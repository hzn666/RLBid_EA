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


def init_parser(campaign_id):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/')
    parser.add_argument('--dataset_name', default='ipinyou/', help='ipinyou, cretio, yoyi, avazu')
    parser.add_argument('--campaign_id', default='1458/', help='1458, 3358, 3386, 3427, 3476, avazu')
    parser.add_argument('--ctr_model_name', default='FM', help='LR,FM,FNN...')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_param_dir', default='models/model_params/')
    parser.add_argument('--save_log_dir', default='logs/')
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--sample_type', default='all', help='all')

    # for ctr prediction
    parser.add_argument('--latent_dims', default=10)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--early_stop_type', default='loss', help='auc, loss')
    parser.add_argument('--early_stop_iter', type=int, default=5)
    parser.add_argument('--loss_epsilon', type=float, default=1e-6)
    parser.add_argument('--auc_epsilon', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--data_mprice_index', type=int, default=0)
    parser.add_argument('--data_ctr_index', type=int, default=1)
    parser.add_argument('--data_clk_index', type=int, default=2)

    args = parser.parse_args()
    args.campaign_id = campaign_id

    return args
