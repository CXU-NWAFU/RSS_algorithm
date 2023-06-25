import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from envs.CEH import CEH
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import os
import time
import argparse
from torch.nn import init
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard as tb
from utils_.data_type import FLOAT_TYPE, TENSOR_FLOAT_TYPE

parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('--exp_name', default='DRQN', type=str)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--hidden_dim", default=128, type=int)
parser.add_argument("--lr_a", default=0.0005, type=float)
parser.add_argument("--batch_size", default=10, type=int)
parser.add_argument("--buffer_size", default=1000, type=int)
parser.add_argument("--seq_len", default=50, type=int)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--Ehp", default=0.2, type=float)
parser.add_argument("--episode_len", default=1000, type=int)
parser.add_argument("--episode_num", default=710, type=int)
parser.add_argument('--target_update_Episode', default=2, type=int)
parser.add_argument("--test_ep", default=1, type=int)
parser.add_argument("--grad_norm", default=10., type=float)
parser.add_argument('--test_epsilon', default=0.05, type=float)
parser.add_argument("--test_interval", default=1, type=int)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument('--Max_Epsilon', default=1.0, type=float)
parser.add_argument('--Min_Epsilon', default=0.01, type=float)
parser.add_argument('--Epsilon_Decay', default=0.0015, type=float)
parser.add_argument('--episode_start', default=200, type=float)
args = parser.parse_args()

if not torch.cuda.is_available():
    args.device = "cpu"

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.set_default_dtype(TENSOR_FLOAT_TYPE)

epsilon_decay = args.Epsilon_Decay
time_str = time.strftime("%m-%d_%H-%M-%S", time.localtime())

log_dir_name = "_".join([
        time_str,
        args.exp_name,
        "CEH",
        "Ehp", str(args.Ehp),
        "seed", str(args.seed),
])

prefix_dir = 'logs'
log_dir_name = os.path.join(prefix_dir, log_dir_name)
fw = tb.SummaryWriter(log_dir_name)
if not os.path.exists(log_dir_name + '/models'):
    os.makedirs(log_dir_name + '/models')
prams_file = open(log_dir_name + '/prams_table.txt', 'w')
prams_file.writelines(f'{i:50} {v}\n' for i, v in args.__dict__.items())
prams_file.close()

env = CEH(Ehp=args.Ehp, device=args.device, ADM=0)
a_num = env.action_num
o_dim = env.obs_dim
action_dim = 1


class ReplayBuffer:
    def __init__(self, a_dim, s_dim, seq_len, ep_len, cap, batch_size):
        assert type(s_dim) == int
        assert type(a_dim) == int
        self.cap = cap
        self.ep_len = ep_len
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_start_idx = self.ep_len - self.seq_len
        self.s_buffer = np.empty((self.cap, ep_len, s_dim), FLOAT_TYPE)
        self.a_buffer = np.empty((self.cap, ep_len, a_dim), FLOAT_TYPE)
        self.r_buffer = np.empty((self.cap, ep_len, 1), FLOAT_TYPE)
        self.next_s_buffer = np.empty((self.cap, ep_len, s_dim), FLOAT_TYPE)
        self.d_buffer = np.empty((self.cap, ep_len, 1), FLOAT_TYPE)
        self.cap_index = 0
        self.ep_index = 0
        self.size = 0

        self.a_dim = a_dim
        self.s_dim = s_dim

    def push(self, s, a, re, next_s, d):
        assert a.shape[0] == self.a_dim
        assert s.shape[0] == self.s_dim
        assert next_s.shape[0] == self.s_dim
        self.s_buffer[self.cap_index, self.ep_index] = s
        self.a_buffer[self.cap_index, self.ep_index] = a
        self.r_buffer[self.cap_index, self.ep_index] = re
        self.next_s_buffer[self.cap_index, self.ep_index] = next_s
        self.d_buffer[self.cap_index, self.ep_index] = d
        self.ep_index = (self.ep_index + 1) % self.ep_len
        seq_done = int(not self.ep_index)
        self.cap_index = (self.cap_index + seq_done) % self.cap
        self.size = min(self.size + seq_done, self.cap)

    def sample(self, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size

        idx1 = np.random.randint(0, self.size, (batch_size, 1))

        invalid_idx_arr = np.where(idx1 == self.cap_index)[0]
        invalid_idx_num = len(invalid_idx_arr)
        if invalid_idx_num > 0:
            new_idx = np.random.randint(0, self.size, (invalid_idx_num, 1))
            while (new_idx == self.cap_index).any():
                new_idx = np.random.randint(0, self.size, (invalid_idx_num, 1))
            idx1[invalid_idx_arr] = new_idx
        start_idx = np.random.randint(0, self.max_start_idx + 1, batch_size)
        idx2 = np.array([np.arange(start, start + self.seq_len) for start in start_idx])
        batch_s = self.s_buffer[idx1, idx2]
        batch_a = self.a_buffer[idx1, idx2]
        batch_r = self.r_buffer[idx1, idx2]
        batch_next_s = self.next_s_buffer[idx1, idx2]
        batch_d = self.d_buffer[idx1, idx2]

        return batch_s, batch_a, batch_r, batch_next_s, batch_d

    def __len__(self):
        return self.size


class DRQN(nn.Module):
    def __init__(self, in_dim, out_dim, unit_num):
        super(DRQN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.unit_num = unit_num
        self.fc1 = nn.Linear(self.in_dim, self.unit_num)
        self.fc2 = nn.LSTM(self.unit_num, self.unit_num, batch_first=True)
        self.fc3 = nn.Linear(self.unit_num, self.out_dim)
        init.kaiming_normal_(self.fc1.weight)
        init.kaiming_normal_(self.fc3.weight)

    def forward(self, s, hidden=None):
        x = self.fc1(s)
        x, new_hidden = self.fc2(x, hidden)
        x = F.relu(x)
        x = self.fc3(x)
        return x, new_hidden

    def choose_action(self, s, epsilon, hidden_):
        sta = torch.FloatTensor(s).to(args.device)
        sta = env.normalize_obs(sta)
        sta = sta.view(1, 1, o_dim)
        q_values, new_hidden = dqn_net.forward(sta, hidden_)
        if np.random.random() <= epsilon:
            action = np.random.choice(env.action_num)
        else:
            with torch.no_grad():
                action = int(torch.argmax(q_values).squeeze().cpu().numpy())
        return action, new_hidden

    def save_model(self, dir=log_dir_name + '/models'):
        self.target_net.save_weights(dir + '/' +  'DRQN_target_net.h5')
        self.dqn_net.save_weights(dir + '/' + 'DRQN_net.h5')


def target_update():
    for target_param, local_param in zip(target_net.parameters(), dqn_net.parameters()):
        target_param.data.copy_(local_param.data)


def train():
    st, at, rt, stp1, dt = buffer.sample(batch_size)
    s = torch.FloatTensor(st).to(args.device)
    a = torch.LongTensor(at).to(args.device)
    rw = torch.FloatTensor(rt).to(args.device)
    s_ = torch.FloatTensor(stp1).to(args.device)
    dt = torch.FloatTensor(dt).to(args.device)
    s = env.normalize_obs(s)
    s_ = env.normalize_obs(s_)
    s = s.view(args.batch_size, args.seq_len, o_dim)
    s_ = s_.view(args.batch_size, args.seq_len, o_dim)

    with torch.no_grad():
        next_q_values, _ = target_net.forward(s_)

        next_q_value, _ = next_q_values.max(-1, keepdim=True)
        target_q_value = rw + (1. - dt) * args.gamma * next_q_value

    q_values, _ = dqn_net.forward(s)
    q_value = q_values.gather(-1, a)

    loss = F.mse_loss(target_q_value.detach(), q_value)
    optimizer.zero_grad()
    loss.backward()
    if args.grad_norm > 0:
        clip_grad_norm_(dqn_net.parameters(), args.grad_norm)
    optimizer.step()
    return loss


def test_dqn():
    TEST_EP = args.test_ep
    EP_LEN = args.episode_len
    TEST_EPSILON = args.test_epsilon
    mean_r = 0
    for i in range(TEST_EP):
        obs = env.reset()
        state = obs
        h = None
        ep_r = 0
        for _ in range(EP_LEN):
            a, h = dqn_net.choose_action(obs, TEST_EPSILON, h)
            next_state, next_obs, r = env.step(state, np.array([act]))
            obs = next_obs
            state = next_state
            ep_r += r
        mean_r += ep_r / EP_LEN
    return mean_r / TEST_EP


if __name__ == '__main__':
    if not torch.cuda.is_available():
        args.device = "cpu"
    units = args.hidden_dim

    dqn_net = DRQN(o_dim, a_num, units).to(args.device)
    target_net = DRQN(o_dim, a_num, units).to(args.device)
    target_update()
    capacity = args.buffer_size
    buffer = ReplayBuffer(action_dim, o_dim, args.seq_len, args.episode_len, capacity, args.batch_size)
    lr = args.lr_a

    optimizer = torch.optim.RMSprop(dqn_net.parameters(), lr=lr, alpha=0.99, eps=0.00001)
    Epsilon_Decay_Rate = (args.Min_Epsilon - args.Max_Epsilon) / args.buffer_size * args.Epsilon_Decay
    epsilon = None
    batch_size = args.batch_size

    global_step = 0
    summary_step = 0
    loss = 0.
    print("Training starts, training information will be recorded into the tensorboard log file.")
    for i in range(args.episode_num):
        hidden = None
        steps = 0

        if i % args.target_update_Episode == 0:
            target_update()
        state_ = env.reset()
        ob_ = env.reset()
        act = 0
        while True:
            steps += 1
            global_step += 1
            done = 0 if steps < args.episode_len else 1
            epsilon = max(Epsilon_Decay_Rate * global_step + args.Max_Epsilon, args.Min_Epsilon)
            act, hidden = dqn_net.choose_action(ob_, epsilon, hidden)
            next_st, next_ob, r = env.step(state_, np.array([act]))
            buffer.push(ob_, np.array([act]), np.array([r]), next_ob, np.array([0]))
            state_ = next_st
            ob_ = next_ob
            if i > args.episode_start:
                loss = train()
            if done:
                if i >= args.episode_start and i % args.test_interval == 0:
                    mean_r = test_dqn()
                    fw.add_scalar('mean_r', mean_r, summary_step)
                    fw.add_scalar('epsilon', epsilon, summary_step)
                    summary_step += args.test_interval
                    print('episode: {}  epsilon: {:.2f}  mean_reward: {}'.format(i + 1, epsilon, mean_r))
                break
