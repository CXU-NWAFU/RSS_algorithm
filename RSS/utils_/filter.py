from utils_.data_type import INT_TYPE
import numpy as np


class Action_Mapper:
    def __init__(self, param, **kwargs):
        action_space = kwargs["action_space"].astype("float32")
        action_space_size = action_space.shape
        self.a_num = action_space_size[0]
        self.max_a = kwargs["env"].a_max.copy()
        self.max_a_num = self.max_a + 1
        self.min_a = kwargs["env"].a_min.copy()
        self.ADM = param.ADM

    def mapper(self, a_float_idx):
        if self.ADM:
            idx = ((a_float_idx + 1.) / 2. * self.max_a_num).astype(INT_TYPE)
            return np.minimum(np.maximum(idx, self.min_a), self.max_a)
        else:
            idx = ((a_float_idx + 1.) / 2. * self.a_num).astype(INT_TYPE)
            return np.minimum(np.maximum(idx, 0), self.a_num - 1)
