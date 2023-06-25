import torch
from utils_.data_type import INT_TYPE
import numpy as np
from torch.distributions.normal import Normal


class Action_Mapper:
    def __init__(self, param, **kwargs):
        action_space = kwargs["action_space"].astype("float32")
        action_space_size = action_space.shape
        self.a_num = action_space_size[0]
        self.max_a = kwargs["env"].a_max.copy()
        self.max_a_num = self.max_a + 1
        self.min_a = kwargs["env"].a_min.copy()
        self.ADM = param.ADM
        self.a_dim = param.a_dim
        device = param.device
        self.normal = Normal(torch.zeros(self.a_dim).to(device),
                             torch.ones(self.a_dim).to(device) * param.std)

    def to_int_index(self, a_float_idx, mean):
        if self.ADM:
            idx = ((a_float_idx + 1.) / 2. * self.max_a_num)
            noise = self.normal.sample((mean.size()[0], mean.size()[1])).squeeze().cpu().numpy()
            idx_to_train = idx + noise
            idx_to_train_proto = (idx_to_train / self.max_a_num) * 2 - 1
            idx = (idx + noise).astype(INT_TYPE)
            return idx_to_train_proto, np.minimum(np.maximum(idx, self.min_a), self.max_a)
        else:
            idx = ((a_float_idx + 1.) / 2. * self.a_num)
            noise = self.normal.sample((mean.size()[0], mean.size()[1])).squeeze().cpu().numpy()
            idx_to_train = idx + noise
            idx_to_train_proto = (idx_to_train / self.a_num) * 2 - 1
            idx = (idx + noise).astype(INT_TYPE)
            return idx_to_train_proto, np.minimum(np.maximum(idx, 0), self.a_num - 1)

    def trainer_to_int_index(self, a_float_idx):
        if self.ADM:
            idx = ((a_float_idx + 1.) / 2. * self.max_a_num).astype(INT_TYPE)
            return np.minimum(np.maximum(idx, self.min_a), self.max_a)
        else:
            idx = ((a_float_idx + 1.) / 2. * self.a_num).astype(INT_TYPE)
            return np.minimum(np.maximum(idx, 0), self.a_num - 1)
