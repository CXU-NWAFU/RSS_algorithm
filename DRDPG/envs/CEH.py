import numpy as np
from utils_.data_type import FLOAT_TYPE, INT_TYPE, EPSILON, TENSOR_FLOAT_TYPE
from itertools import product
import torch

_K = 5
_Group_n = 4
_N = _K * _Group_n
_M = 2
_Dnk_list = [
    [0.4, 0.6, 0.8, 1.0],
    [0.4, 0.6, 0.8, 1.0],
    [0.4, 0.6, 0.8, 1.0],
    [0.4, 0.6, 0.8, 1.0],
    [0.4, 0.6, 0.8, 1.0],
]
_Dk0_list = [.999, .999, .999, .999, .999]
_Pn_list = [
    0.05, 0.10, 0.15, 0.2,
    0.05, 0.10, 0.15, 0.2,
    0.05, 0.10, 0.15, 0.2,
    0.05, 0.10, 0.15, 0.2,
    0.05, 0.10, 0.15, 0.2,
]
_E_max = 20
_En = _E_max * np.ones((_N,), dtype=INT_TYPE)
_E0 = 1
_Eh0 = 1
_Ex = 21
_Ehp = np.full((_N,), 0.2, dtype=FLOAT_TYPE)
_Pn = np.array(_Pn_list, dtype=FLOAT_TYPE)
_Dnk = np.array(_Dnk_list, dtype=FLOAT_TYPE)
_Dk0 = np.array(_Dk0_list, dtype=FLOAT_TYPE)
_G_max = _N * _K * 4
_X_max = _N * _K * 4
_AoI_max = _N * _K * 2
_AoI_min = 1
_Slot_len = 1
_Reward_scale = 1.


class CEH:
    def __init__(
            self,
            N=_N,
            K=_K,
            M=_M,
            E_max=_E_max,
            En=_En,
            E0=_E0,
            Eh0=_Eh0,
            Ex=_Ex,
            Ehp=_Ehp,
            Pn=_Pn,
            Dnk=_Dnk,
            Dk0=_Dk0,
            G_max=_G_max,
            X_max=_X_max,
            AoI_max=_AoI_max,
            AoI_min=_AoI_min,
            Slot_len=_Slot_len,
            Reward_scale=_Reward_scale,
            device="cuda:0",
            Group_n=_Group_n,
            ADM=1
    ):
        self.N = N
        self.K = K
        self.M = M
        self.E_max = E_max
        self.En = En
        self.E0 = E0
        self.Eh0 = Eh0
        self.Ex = Ex
        self.Ehp = Ehp
        self.Pn = Pn
        self.Dnk = Dnk
        self.Dk0 = Dk0
        self.G_max = G_max
        self.X_max = X_max
        self.AoI_max = AoI_max
        self.AoI_min = AoI_min
        self.Slot_len = Slot_len
        self.Group_n = Group_n
        self.Action_Space = self._build_action_space()
        self.action_num = len(self.Action_Space)
        self.obs_dim = self.N * 3 + 1
        self.init_state = self._gen_init_state()
        self.r_scale = Reward_scale
        self.normalize_base = torch.empty(
            (self.obs_dim,), dtype=TENSOR_FLOAT_TYPE, device=device)
        self.normalize_base[0:self.N] = self.G_max
        self.normalize_base[self.N:self.N * 2] = self.X_max
        self.normalize_base[self.N * 2:self.N * 3] = self.E_max
        self.normalize_base[-1] = self.AoI_max
        self.revised_a = np.zeros(self.N, INT_TYPE)
        self.a_dim = self.K if ADM else 1
        self.init_action = np.ones((self.a_dim,), dtype=FLOAT_TYPE) * -1
        self.sensor_num = np.array([len(csp) for csp in self.Dnk], INT_TYPE)
        self.Dec_Action_Space = self._build_action_space_dec()
        self.a_min = self.sensor_num * 0
        self.a_max = np.array([
            len(csp_as) - 1 for csp_as in self.Dec_Action_Space], INT_TYPE)

    def _build_action_space_dec(self):
        dec_as = []
        for csp_idx in range(self.K):
            csp_as_list = []
            for possible_a in product(*[(0, 1) for _ in range(self.sensor_num[csp_idx])]):
                a_np = np.array(possible_a, dtype=INT_TYPE)
                if self._check_action_dec(a_np, csp_idx):
                    csp_as_list.append(a_np)
            csp_as = np.array(csp_as_list, dtype=INT_TYPE)
            dec_as.append(csp_as)
        return dec_as

    def _check_action_dec(self, a_vector, csp_idx):
        if a_vector.sum() == 0: return True
        if not self._check_channel_dec(a_vector): return False
        return self._check_importance_dec(a_vector, csp_idx)

    def _check_channel_dec(self, a_vector):
        return a_vector.sum() <= self.M

    def _check_importance_dec(self, a_vector, csp_idx):
        imp = (a_vector * self.Dnk[csp_idx]).sum()
        return (imp - self.Dk0[csp_idx]) >= -EPSILON

    def _build_action_space(self):
        action_space_list = []
        for a_item in product(*[(0, 1) for _ in range(self.N)]):
            a_item_np = np.array(a_item, dtype=INT_TYPE)
            if self._check_action(a_item_np):
                action_space_list.append(a_item_np)
        return np.array(action_space_list, dtype=INT_TYPE)

    def _check_action(self, a_vector):
        if a_vector.sum() == 0: return True
        a_matrix = a_vector.reshape(self.K, self.Group_n)
        if not self._check_channel(a_matrix): return False
        return self._check_importance(a_matrix)

    def _check_channel(self, a_matrix):
        return (a_matrix.sum(-1) <= self.M).all()

    def _check_importance(self, a_matrix):
        imp_count = a_matrix.sum(-1)
        if (imp_count == 0).any():  return False
        imp = (a_matrix * self.Dnk).sum(-1)
        return (imp - self.Dk0 >= -EPSILON).all()

    def _gen_init_state(self):
        init_state = np.zeros((self.obs_dim,), dtype=INT_TYPE)
        init_state[self.N * 2:self.N * 3] = self.E_max
        return init_state

    def reset(self):
        return self.init_state.copy()

    def step(self, st, at):
        st_ = st.copy()
        Gt = st_[0:self.N]
        Xt = st_[self.N:self.N * 2]
        et = st_[self.N * 2:self.N * 3]
        AoIt = st_[-1]
        if len(at) > 1:
            a_list = [self.Dec_Action_Space[csp_id][at[csp_id]] for csp_id in range(self.K)]
            At_final = np.concatenate(a_list)
        else:
            At_final = self.Action_Space[at.squeeze()].copy()
        if not self._check_action(At_final):
            At_final = self.revised_a.copy()
        activated_idx = np.where(At_final > 0)[0]
        etp1 = et.copy()
        etp1[activated_idx] -= self.E0
        no_energy_idx = np.where(etp1 < 0)[0]
        At_final[no_energy_idx] = 0
        etp1[no_energy_idx] = et[no_energy_idx]
        fail_trans_idx = np.where(np.random.rand(self.N) < self.Pn)[0]
        At_final[fail_trans_idx] = 0
        success_idx = np.where(At_final != 0)[0]
        AoItp1 = np.minimum(AoIt + self.Slot_len, self.AoI_max)
        if self._check_importance(At_final.reshape(self.K, self.Group_n)):
            AoItp1 = self.AoI_min
        etp1_o = np.full((self.N,), self.Ex, dtype=INT_TYPE)
        etp1_o[success_idx] = etp1[success_idx].copy()
        eh_idx = np.where(np.random.rand(self.N) < self.Ehp)[0]
        etp1[eh_idx] += self.Eh0
        etp1 = np.minimum(etp1, self.E_max)
        Gtp1 = Gt + self.Slot_len
        Gtp1[success_idx] = 0
        Gtp1 = np.minimum(Gtp1, self.G_max)
        Xtp1 = Xt
        Xtp1[activated_idx] += 1
        Xtp1[success_idx] = 0
        Xtp1 = np.minimum(Xtp1, self.X_max)
        stp1 = np.append(np.concatenate((Gtp1, Xtp1, etp1)), AoItp1)
        otp1 = np.append(np.concatenate((Gtp1, Xtp1, etp1_o)), AoItp1)
        reward = -AoItp1
        return stp1, otp1, reward

    def normalize_obs(self, st):
        return st / self.normalize_base

    def normalize_r(self, r):
        return r / self.r_scale

    def init_a(self):
        return self.init_action.copy()
