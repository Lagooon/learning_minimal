import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from functools import partial
import os
import sys
import time
import random
import yaml
import socket
import wandb
import argparse
import networks.mlp.mlp, networks.mlp_mixer.mlp_mixer, networks.transformer.transformer
from einops.layers.torch import Rearrange, Reduce

import lightgbm as lgb
curdir = os.path.dirname(os.path.abspath(__file__))

def get_args():
    parser = argparse.ArgumentParser()
    #wandb
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--wandb_project", type=str, default="cv-project")
    parser.add_argument("--wandb_group", type=str, default="mlp")
    parser.add_argument("--job_type", type=str, default="training")
    parser.add_argument("--wandb_name", type=str, default="")
    parser.add_argument("--user_name", type=str, default="dl_project_")
    #parser.add_argument("--hyperparameter_search", action='store_true')
    
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--datafolder", type=str, default="26anchors")
    parser.add_argument("--anchors", default=26, type=int)
    #parser.add_argument("--early_stopping", action='store_true')
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    #parser.add_argument("--min_lr", default=1e-6, type=float) 
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--num_epoch", default=300, type=int)
    parser.add_argument("--save_interval", default=3, type=int)
    parser.add_argument("--save_dir", default='models')
    parser.add_argument("--total_updates", default=50000, type=int)
    parser.add_argument("--optimizer", default='sgd', type=str)
    args = parser.parse_args()
    setattr(args, 'save_dir', os.path.join(curdir, args.save_dir, args.model))
    setattr(args, 'datafolder', os.path.join(curdir, args.datafolder))
    '''
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumualte before performing a backward/update pass."
    )
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    '''
    return args

def normalize(x, std, mean):
    sz = x.shape[0]
    return ((x.reshape(sz, -1, 2) - mean) / (std + 1e-10)).reshape(sz, -1)

def lgb_metric(y_pred, y_true):
    y_true = y_true.get_label()
    y_pred = torch.tensor(y_pred.reshape(-1, 26))
    y_true = torch.tensor(y_true.reshape(-1, 26))
    pred = torch.argmax(y_pred, dim=1, keepdim=True)
    corr = torch.gather(y_true, 1, pred)
    c1 = torch.sum(corr > 0).item()
    return 'rate', c1/y_true.shape[0], True

def train(args):

    if args.use_wandb:
        run = wandb.init(config = args,
                        project = args.wandb_project,
                        group = args.wandb_group,
                        entity = args.user_name,
                        notes = socket.gethostname(),
                        name = args.wandb_name,
                        job_type = args.job_type)
        if args.wandb_name == "":
            wandb.run.name = f'lr{args.lr:.2e}-weightdecay{args.weight_decay:.2e}-{args.optimizer}-layers{args.num_layers}-hiddendim{args.hidden_dim}-anchors{args.anchors}-seed{args.seed}'
    
    os.makedirs(args.save_dir, exist_ok=True)
    # set up the network
    # load the training and validation data
    X_train = np.loadtxt(args.datafolder+"/X_train.txt")
    Y_train = np.loadtxt(args.datafolder+"/Y_train.txt")
    X_test = np.loadtxt(args.datafolder+"/X_val.txt")
    Y_test = np.loadtxt(args.datafolder+"/Y_val.txt")
    anchors = np.loadtxt(args.datafolder+"/anchors.txt", skiprows=1)
    anchors = anchors.reshape(1, anchors.shape[0], anchors.shape[1])
    y_train = Y_train.reshape(-1)
    y_test = Y_test.reshape(-1)
    X_train = X_train.repeat(args.anchors,axis=0)
    X_test = X_test.repeat(args.anchors,axis=0)
    anchors_train = anchors.repeat(X_train.shape[0]/args.anchors,axis=0).reshape(-1, anchors.shape[2])
    anchors_test = anchors.repeat(X_test.shape[0]/args.anchors,axis=0).reshape(-1, anchors.shape[2])
    X_train = np.concatenate([X_train, anchors_train], axis=1)
    X_test = np.concatenate([X_test, anchors_test], axis=1)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'dart',
        'seed': 42,
        'num_leaves': 100,
        'learning_rate': 0.01,
        'feature_fraction': 1,
        'scale_pos_weight' : 15,
        'bagging_freq': 10,
        'bagging_fraction': 0.50,
        'n_jobs': -1,
        'lambda_l2': 2,
        'min_data_in_leaf': 40,
    }
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test)
    print(X_train.shape,X_test.shape)
    model = lgb.train(
        params = params,
        train_set = lgb_train,
        num_boost_round = 10500,
        valid_sets = [lgb_train, lgb_test],
        early_stopping_rounds = 1500,
        verbose_eval = 20,
        feval = lgb_metric
    )
    

if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    with open(os.path.join(curdir, 'networks', args.model, 'config.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in config.items():
            if f'--{k}' not in sys.argv[1:]:
                setattr(args, k, v)
    print(args)
    train(args)