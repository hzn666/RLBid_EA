import argparse
import torch
import random
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/')
    parser.add_argument('--dataset_name', default='ipinyou/', help='ipinyou, cretio, yoyi, avazu')
    parser.add_argument('--campaign_id', default='1458', help='1458, 3427')
    parser.add_argument('--episodes', type=int, default=3000)
    parser.add_argument('--model_name', default='FAB')
    parser.add_argument('--lr_A', type=float, default=3e-4)
    parser.add_argument('--lr_C', type=float, default=3e-4)
    parser.add_argument('--neuron_nums', type=list, default=[128, 64])
    parser.add_argument('--tau', type=float, default=0.0005)
    parser.add_argument('--memory_size', type=float, default=100000)
    parser.add_argument('--rl_batch_size', type=int, default=32)
    parser.add_argument('--rl_early_stop_iter', type=int, default=20)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_log_dir', default='../fab/log/')
    parser.add_argument('--result_path', type=str, default='../fab/result/')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--time_fraction', type=int, default=96)
    parser.add_argument('--input_dims', type=int, default=4)

    parser.add_argument('--budget', type=float, default=16e6)
    parser.add_argument('--budget_para', type=int, default=2, help='2,4,8,16')  # 预算调整比例

    parser.add_argument('--reward_type', type=str, default='op', help='op, nop_2.0, clk')
    # op 缩放，nop 不缩放，clk, 直接使用点击数做奖励

    args = parser.parse_args()

    return args
