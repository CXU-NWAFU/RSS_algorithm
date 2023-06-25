import numpy as np
import torch
from utils_.data_type import TENSOR_FLOAT_TYPE, FLOAT_TYPE


class RecurrentReplayBuffer:
    def __init__(self, a_dim, s_dim, seq_len=100, ep_len=1000, cap=100, batch_size=4, device="cuda:0"):
        assert type(s_dim) == int
        assert type(a_dim) == int
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.device = device
        self.cap = cap
        self.ep_len = ep_len
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_start_idx = self.ep_len - 1 - self.seq_len
        self.cap_index = 0
        self.step_idx = 0
        self.size = 0
        self.atm1_buffer_np = np.empty((ep_len, a_dim), dtype=FLOAT_TYPE)
        self.ot_buffer_np = np.empty((ep_len, s_dim), dtype=FLOAT_TYPE)
        self.rt_buffer_np = np.empty((ep_len, 1), dtype=FLOAT_TYPE)
        self.atm1_buffer = torch.empty(self.cap, ep_len, a_dim, dtype=TENSOR_FLOAT_TYPE, device=device)
        self.ot_buffer = torch.empty(self.cap, ep_len, s_dim, dtype=TENSOR_FLOAT_TYPE, device=device)
        self.rt_buffer = torch.empty(self.cap, ep_len, 1, dtype=TENSOR_FLOAT_TYPE, device=device)

    def push(self, atm1, ot, rt):
        assert atm1.shape[0] == self.a_dim
        assert ot.shape[0] == self.s_dim
        self.atm1_buffer_np[self.step_idx] = atm1.copy()
        self.ot_buffer_np[self.step_idx] = ot.copy()
        self.rt_buffer_np[self.step_idx] = np.array([rt])
        self.step_idx += 1

    def store_episode(self):
        self.atm1_buffer[self.cap_index] = torch.from_numpy(self.atm1_buffer_np).to(self.device)
        self.ot_buffer[self.cap_index] = torch.from_numpy(self.ot_buffer_np).to(self.device)
        self.rt_buffer[self.cap_index] = torch.from_numpy(self.rt_buffer_np).to(self.device)
        self.cap_index = (self.cap_index + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
        self.step_idx = 0

    def sample(self, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        ep_idx = np.random.randint(0, self.size, (batch_size, 1))
        start_idx = np.random.randint(0, self.max_start_idx + 1, batch_size)
        ep_range = np.array([np.arange(start, start + self.seq_len + 1) for start in start_idx])
        batch_atm1 = self.atm1_buffer[ep_idx, ep_range]
        batch_ot = self.ot_buffer[ep_idx, ep_range]
        batch_rt = self.rt_buffer[ep_idx, ep_range]
        assert list(batch_atm1.size()) == [batch_size, self.seq_len + 1, self.a_dim]
        assert list(batch_ot.size()) == [batch_size, self.seq_len + 1, self.s_dim]
        return batch_atm1, batch_ot, batch_rt

    def __len__(self):
        return self.size
