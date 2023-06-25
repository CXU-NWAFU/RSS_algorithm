import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init

STD_MIN = 1e-5
STD_MAX = 5.


class Continuous_Q_Critic(nn.Module):
    def __init__(self, a_dim, o_dim, hidden_dim, relu=False):
        super(Continuous_Q_Critic, self).__init__()
        assert type(a_dim) == int
        assert type(o_dim) == int
        self.fc1 = nn.Linear(a_dim + o_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = relu
        init.kaiming_normal_(self.fc1.weight)
        init.kaiming_normal_(self.fc3.weight)
        self.o_dim = o_dim
        self.a_dim = a_dim

    def forward(self, atm1, ot, at, htm1=None):
        assert ot.size()[-1] == self.o_dim
        assert at.size()[-1] == self.a_dim
        input_ = torch.cat((ot, at), dim=-1)
        x = self.fc1(input_)
        x = F.relu(x)
        x, ht = self.lstm(x, htm1)
        x = self.fc3(x)
        return x, ht


class Deterministic_Actor(nn.Module):
    def __init__(self, a_dim, o_dim, hidden_dim, relu=False, lstm_a=False):
        super(Deterministic_Actor, self).__init__()
        assert type(a_dim) == int
        assert type(o_dim) == int
        input_dim = a_dim + o_dim if lstm_a else o_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(hidden_dim, a_dim)
        init.kaiming_normal_(self.fc1.weight)
        init.kaiming_normal_(self.fc3.weight)
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.relu = relu
        self.lstm_a = lstm_a

    def forward(self, atm1, ot, htm1=None):
        assert ot.size()[-1] == self.o_dim
        input_ = torch.cat((atm1, ot), dim=-1) if self.lstm_a else ot
        x = self.fc1(input_)
        x = F.relu(x)
        x, ht = self.lstm(x, htm1)
        mean = self.fc3(x)
        return mean, mean, ht
