import sys
import os
import json

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import random
import time
import torch.utils.tensorboard as tfb
import argparse
from utils_.data_type import *
from utils_.replay_buffer import RecurrentReplayBuffer
from train.trainer import train
from envs.CEH import CEH
import torch
from agents.RSS import RSS_Agent


def add_argparse():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--exp_name', default='RSS', type=str)

    parser.add_argument("--Ehp", default=0.2, type=float)
    parser.add_argument("--ADM", default=1, type=int)
    parser.add_argument("--seed", default=1, type=int)

    parser.add_argument("--alg", default="RSS", type=str)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--alpha", default=0.002, type=float)
    parser.add_argument("--hidden_dim", default=128, type=int)

    parser.add_argument("--lr_c", default=0.0005, type=float)
    parser.add_argument("--lr_a", default=0.0005, type=float)
    parser.add_argument("--lr_adjust", default=0, type=int)
    parser.add_argument("--lr_decay", default=1, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--grad_norm", default=10., type=float)
    parser.add_argument("--relu", default=0, type=int)

    parser.add_argument("--recurrent", default=1, type=int)
    parser.add_argument("--seq_len", default=50, type=int)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--buffer_size", default=1000, type=int)
    parser.add_argument("--start_replay", default=200, type=int)

    parser.add_argument("--ep_len", default=1000, type=int)
    parser.add_argument("--train_ep", default=710, type=int)
    parser.add_argument("--test_ep", default=1, type=int)
    parser.add_argument("--train_interval", default=1, type=int)
    parser.add_argument("--test_interval", default=1, type=int)

    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--save_his", default=1, type=int)
    parser.add_argument("--lstm_a", default=0, type=int)
    parser.add_argument("--target_a", default=0, type=int)

    args_ = parser.parse_args()
    return args_


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log(params):
    time_str = time.strftime("%m-%d_%H-%M-%S")
    log_name = "_".join([
        time_str,
        params.exp_name,
        "CEH",
        "ADM" + str(params.ADM),
        "Ehp", str(params.Ehp),
        "seed", str(params.seed),
    ])
    prefix_dir = "logs"
    log_dir = os.path.join(prefix_dir, log_name)
    log_fw = tfb.SummaryWriter(log_dir)
    dict_params = params.__dict__
    j_content = json.dumps(dict_params, indent=4)
    f = open(os.path.join(log_dir, "params.json"), "w")
    f.write(j_content)
    f.close()
    return log_fw


def get_env(params):
    env = CEH(
        Ehp=params.Ehp,
        device=params.device,
        ADM=params.ADM,
    )
    params.a_dim = env.a_dim
    params.o_dim = env.obs_dim
    params.a_num = env.action_num
    params.sensor_num = env.N
    params.csp_num = env.K
    return env


def run(params):
    if not torch.cuda.is_available():
        params.device = "cpu"
    env = get_env(params)
    set_seed(params.seed)
    torch.set_default_dtype(TENSOR_FLOAT_TYPE)
    log_fw = log(params)
    knn_as = env.Action_Space
    agent = RSS_Agent(
        params, action_space=knn_as,
        obs_normalizer=env.normalize_obs, env=env)
    buffer = RecurrentReplayBuffer(
        params.a_dim, params.o_dim,
        params.seq_len, params.ep_len, params.buffer_size,
        params.batch_size, params.device)

    print("Training starts, training information will be recorded into the tensorboard log file.")

    train(
        log_fw=log_fw,
        params=params,
        env=env,
        agent=agent,
        buffer=buffer
    )


if __name__ == '__main__':
    run(add_argparse())
