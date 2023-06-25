import torch
import numpy as np
import torch.nn.functional as F
from models.rnn import Continuous_Q_Critic, Gaussian_Actor
from utils_.filter import Action_Mapper
from torch.distributions.normal import Normal
from utils_.data_type import TENSOR_FLOAT_TYPE, FLOAT_TYPE
from torch.nn.utils import clip_grad_norm_


class RSS_Agent:
    def __init__(self, params, **kwargs):

        net_params = (params.a_dim, params.o_dim, params.hidden_dim, params.relu)
        device = params.device
        self.actor = Gaussian_Actor(*(net_params + (params.lstm_a,))).to(device)
        self.target_actor = Gaussian_Actor(*(net_params + (params.lstm_a,))).to(device)
        self.target_actor_flag = params.target_a
        self.q1, self.q2, self.target_q1, self.target_q2 = \
            [Continuous_Q_Critic(*net_params).to(device) for _ in range(4)]
        self.actor_list = (self.actor, self.target_actor)
        self.grad_norm = params.grad_norm
        self.device = device
        self._update_target(tau=1.0)
        self.q1_opt = torch.optim.RMSprop(self.q1.parameters(), lr=params.lr_c, weight_decay=params.weight_decay,
                                          alpha=0.99, eps=0.00001)
        self.q2_opt = torch.optim.RMSprop(self.q2.parameters(), lr=params.lr_c, weight_decay=params.weight_decay,
                                          alpha=0.99, eps=0.00001)
        self.actor_opt = torch.optim.RMSprop(self.actor.parameters(), lr=params.lr_a,
                                             weight_decay=params.weight_decay, alpha=0.99, eps=0.00001)
        self.lr_schedulers = [
            torch.optim.lr_scheduler.StepLR(self.q1_opt, params.ep_len, gamma=params.lr_decay),
            torch.optim.lr_scheduler.StepLR(self.q2_opt, params.ep_len, gamma=params.lr_decay),
            torch.optim.lr_scheduler.StepLR(self.actor_opt, params.ep_len, gamma=params.lr_decay)
        ]
        self.tau = params.tau
        self.gamma = params.gamma
        gamma_arr = np.empty((params.ep_len,), dtype=FLOAT_TYPE)
        gamma_arr[0] = 1.
        for i in range(1, params.ep_len):
            gamma_arr[i] = gamma_arr[i - 1] * self.gamma
        self.gamma_arr = gamma_arr
        self.seq_len = params.seq_len
        self.alpha = torch.tensor([params.alpha], requires_grad=False).to(device)
        self.o_dim = params.o_dim
        self.a_dim = params.a_dim
        self.ADM = params.ADM
        self.normal = Normal(torch.zeros(self.a_dim).to(device), torch.ones(self.a_dim).to(device))
        self.obs_normalizer = kwargs["obs_normalizer"]
        self.a_filter = Action_Mapper(
            params,
            action_space=kwargs["action_space"],
            env=kwargs["env"]
        )
        self.q1_h = None
        self.q2_h = None
        self.lr_decay = params.lr_adjust

    def _update_target(self, tau=0.001):
        for target_param, local_param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def _choose_action(self, atm1: torch.Tensor, ot: torch.Tensor, htm1=None, target_actor=0):
        assert len(ot.size()) == 3
        assert len(atm1.size()) == 3
        assert ot.size()[-1] == self.o_dim
        assert atm1.size()[-1] == self.a_dim
        batch_size = ot.size()[0]
        mean, std, ht = self.actor_list[target_actor].forward(atm1, ot, htm1)
        assert list(mean.size()) == [batch_size, ot.size()[1], self.a_dim]
        zt = self.normal.sample((mean.size()[0], mean.size()[1]))
        ut = mean + std * zt
        at = torch.tanh(ut)
        dis = Normal(mean, std)
        return at, ht, mean, std, dis, ut

    def choose_action(self, atm1, ot, htm1=None, env=None):
        assert ot.shape[0] == self.o_dim
        assert atm1.shape[0] == self.a_dim
        with torch.no_grad():
            batch_ot = torch.FloatTensor(ot).view(1, 1, self.o_dim).to(self.device)
            batch_ot = env.normalize_obs(batch_ot)
            batch_atm1 = torch.from_numpy(atm1).view(1, 1, self.a_dim).to(TENSOR_FLOAT_TYPE).to(self.device)
            batch_at_float, ht, _, _, _, _ = self._choose_action(batch_atm1, batch_ot, htm1)
            if self.ADM:
                proto_a = batch_at_float.squeeze().cpu().numpy()
            else:
                proto_a = batch_at_float.squeeze().cpu().numpy()[np.newaxis]
            output_a = self.a_filter.mapper(proto_a)
        return (proto_a, output_a), ht

    def evaluate_action(self, atm1, ot, htm1=None, env=None):
        with torch.no_grad():
            batch_ot = torch.FloatTensor(ot).view(1, 1, self.o_dim).to(self.device)
            batch_ot = env.normalize_obs(batch_ot)
            batch_atm1 = torch.from_numpy(atm1).view(1, 1, self.a_dim).to(TENSOR_FLOAT_TYPE).to(self.device)
            mean, _, ht = self.actor.forward(batch_atm1, batch_ot, htm1)
            batch_at_float = torch.tanh(mean)
            if self.ADM:
                proto_a = batch_at_float.squeeze().cpu().numpy()
            else:
                proto_a = batch_at_float.squeeze().cpu().numpy()[np.newaxis]
            output_a = self.a_filter.mapper(proto_a)
        return (proto_a, output_a), ht

    def train(self, batch, env, log=None, log_fw=None, log_step=None):
        pre_atm1_, pre_ot_, pre_rt_ = batch
        pre_atm1 = pre_atm1_
        pre_ot = env.normalize_obs(pre_ot_)
        pre_rt = env.normalize_r(pre_rt_)
        atm1 = pre_atm1[:, 0:-1, :]
        ot = pre_ot[:, 0:-1, :]
        at = pre_atm1[:, 1:, :]
        rt = pre_rt[:, 0:-1, :]
        otp1 = pre_ot[:, 1:, :]
        with torch.no_grad():
            atp1, _, _, _, dis_atp1, utp1 = self._choose_action(
                at, otp1, target_actor=self.target_actor_flag)
            log_pi_atp1 = (dis_atp1.log_prob(utp1) -
                           torch.log(1.0 - atp1 ** 2 + 1e-6)).sum(dim=-1, keepdim=True)
            next_q1, _ = self.target_q1.forward(at, otp1, atp1)
            next_q2, _ = self.target_q2.forward(at, otp1, atp1)
            min_next_q = torch.min(next_q1, next_q2)
            target = rt + self.gamma * (min_next_q - self.alpha * log_pi_atp1)
        q1, _ = self.q1.forward(atm1, ot, at)
        q2, _ = self.q2.forward(atm1, ot, at)
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        new_at, _, mean, std, dis_at, ut = self._choose_action(atm1, ot)
        new_log_pi_at = (dis_at.log_prob(ut) -
                         torch.log(1.0 - new_at ** 2 + 1e-6)).sum(dim=-1, keepdim=True)
        new_q1, _ = self.q1.forward(atm1, ot, new_at)
        new_q2, _ = self.q2.forward(atm1, ot, new_at)
        min_new_q = torch.min(new_q1, new_q2)
        actor_loss = (self.alpha.detach() * new_log_pi_at - min_new_q).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            clip_grad_norm_(self.actor.parameters(), self.grad_norm)
        self.actor_opt.step()
        self.q1_opt.zero_grad()
        q1_loss.backward()
        if self.grad_norm > 0:
            clip_grad_norm_(self.q1.parameters(), self.grad_norm)
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        q2_loss.backward()
        if self.grad_norm > 0:
            clip_grad_norm_(self.q2.parameters(), self.grad_norm)
        self.q2_opt.step()
        self._update_target(self.tau)
        with torch.no_grad():
            if log:
                log_fw.add_scalar("mean_q", min_new_q.mean(), log_step)
            a_mean_grad, c_mean_grad = self.get_mean_grad()
        if self.lr_decay:
            self.adjust_lr()
        return ((q1_loss + q2_loss) / 2).detach(), actor_loss.detach(), a_mean_grad.detach(), \
            c_mean_grad.detach()

    def get_mean_grad(self):
        a_grad = 0
        c_grad = 0
        cnt = 0
        for w in self.actor.parameters():
            a_grad += w.grad.abs().mean()
            cnt += 1
        a_grad /= cnt
        cnt = 0
        for w in self.q1.parameters():
            c_grad += w.grad.abs().mean()
            cnt += 1
        c_grad /= cnt
        return a_grad, c_grad

    def clear_inner_history(self):
        self.q1_h = None
        self.q2_h = None

    def adjust_lr(self):
        for s in self.lr_schedulers:
            s.step()
