import torch
import numpy as np
import torch.nn.functional as F
from models.rnn import Continuous_Q_Critic, Deterministic_Actor
from utils_.filter import Action_Mapper
from utils_.data_type import TENSOR_FLOAT_TYPE, FLOAT_TYPE
from torch.nn.utils import clip_grad_norm_


class DRDPG_Agent:
    def __init__(self, params, **kwargs):
        net_params = (params.a_dim, params.o_dim, params.hidden_dim, params.relu)
        device = params.device
        self.actor = Deterministic_Actor(*(net_params + (params.lstm_a,))).to(device)
        self.target_actor = Deterministic_Actor(*(net_params + (params.lstm_a,))).to(device)
        self.target_actor_flag = params.target_a
        self.q = Continuous_Q_Critic(*net_params).to(device)
        self.target_q = Continuous_Q_Critic(*net_params).to(device)
        self.actor_list = (self.actor, self.target_actor)
        self.grad_norm = params.grad_norm
        self.device = device
        self._update_target(tau=1.0)
        self.q_opt = torch.optim.RMSprop(self.q.parameters(), lr=params.lr_c, weight_decay=params.weight_decay,
                                         alpha=0.99, eps=0.00001)
        self.actor_opt = torch.optim.RMSprop(self.actor.parameters(), lr=params.lr_a,
                                             weight_decay=params.weight_decay, alpha=0.99, eps=0.00001)
        self.lr_schedulers = [
            torch.optim.lr_scheduler.StepLR(self.q_opt, params.ep_len, gamma=params.lr_decay),
            torch.optim.lr_scheduler.StepLR(self.actor_opt, params.ep_len, gamma=params.lr_decay)
        ]
        self.tau = params.tau
        self.gamma = params.gamma
        gamma_arr = np.empty((params.ep_len,), dtype=FLOAT_TYPE)
        gamma_arr[0] = 1.
        for i in range(1, params.ep_len):
            gamma_arr[i] = gamma_arr[i - 1] * self.gamma
        self.gamma_arr = gamma_arr
        self.recurrent = params.recurrent
        self.seq_len = params.seq_len
        self.o_dim = params.o_dim
        self.a_dim = params.a_dim
        self.ADM = params.ADM
        self.obs_normalizer = kwargs["obs_normalizer"]
        self.a_filter = Action_Mapper(
            params,
            action_space=kwargs["action_space"],
            env=kwargs["env"]
        )
        self.q_h = None
        self.lr_decay = params.lr_adjust

    def _update_target(self, tau=0.001):
        for target_param, local_param in zip(self.target_q.parameters(), self.q.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def _choose_action(self, atm1: torch.Tensor, ot: torch.Tensor, htm1=None, target_actor=0):
        assert len(ot.size()) == 3
        assert len(atm1.size()) == 3
        assert ot.size()[-1] == self.o_dim
        assert atm1.size()[-1] == self.a_dim
        batch_size = ot.size()[0]
        mean, _, ht = self.actor_list[target_actor].forward(atm1, ot, htm1)
        assert list(mean.size()) == [batch_size, ot.size()[1], self.a_dim]
        at = torch.tanh(mean)
        return at, ht, mean, None, None, None

    def choose_action(self, atm1, ot, htm1=None, env=None):
        assert ot.shape[0] == self.o_dim
        assert atm1.shape[0] == self.a_dim
        with torch.no_grad():
            batch_ot = torch.FloatTensor(ot).view(1, 1, self.o_dim).to(self.device)
            batch_ot = env.normalize_obs(batch_ot)
            batch_atm1 = torch.from_numpy(atm1).view(1, 1, self.a_dim).to(TENSOR_FLOAT_TYPE).to(self.device)
            batch_at_float, ht, mean, _, _, _ = self._choose_action(
                batch_atm1, batch_ot, htm1)
            if self.ADM:
                proto_a = batch_at_float.squeeze().cpu().numpy()
            else:
                proto_a = batch_at_float.squeeze().cpu().numpy()[np.newaxis]
            idx_to_train_proto, output_a = self.a_filter.to_int_index(proto_a, mean)
        return (idx_to_train_proto, output_a), ht

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
            next_q, _ = self.target_q.forward(at, otp1, atp1)
            target = rt + self.gamma * next_q
        q, _ = self.q.forward(atm1, ot, at)
        q_loss = F.mse_loss(q, target)
        new_at, _, mean, std, dis_at, ut = self._choose_action(atm1, ot)
        new_q1, _ = self.q.forward(atm1, ot, new_at)
        min_new_q = new_q1
        actor_loss = (-min_new_q).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            clip_grad_norm_(self.actor.parameters(), self.grad_norm)
        self.actor_opt.step()
        self.q_opt.zero_grad()
        q_loss.backward()
        if self.grad_norm > 0:
            clip_grad_norm_(self.q.parameters(), self.grad_norm)
        self.q_opt.step()
        self._update_target(self.tau)
        with torch.no_grad():
            if log:
                log_fw.add_scalar("mean_q", min_new_q.mean(), log_step)
            a_mean_grad, c_mean_grad = self.get_mean_grad()
        return q_loss.detach(), actor_loss.detach(), \
            a_mean_grad.detach(), c_mean_grad.detach()

    def get_mean_grad(self):
        a_grad = 0
        c_grad = 0
        cnt = 0
        for w in self.actor.parameters():
            a_grad += w.grad.abs().mean()
            cnt += 1
        a_grad /= cnt
        cnt = 0
        for w in self.q.parameters():
            c_grad += w.grad.abs().mean()
            cnt += 1
        c_grad /= cnt
        return a_grad, c_grad

    def clear_inner_history(self):
        self.q_h = None

    def adjust_lr(self):
        for s in self.lr_schedulers:
            s.step()
