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

curdir = os.path.dirname(os.path.abspath(__file__))

def get_args():
	parser = argparse.ArgumentParser()
	#wandb
	parser.add_argument("--use_wandb", action='store_true')
	parser.add_argument("--wandb_project", type=str, default="cv-project")
	parser.add_argument("--wandb_group", type=str, default="mlp")
	parser.add_argument("--job_type", type=str, default="testing")
	parser.add_argument("--wandb_name", type=str, default="")
	parser.add_argument("--user_name", type=str, default="dl_project_")
	#parser.add_argument("--hyperparameter_search", action='store_true')
	
	parser.add_argument("--model", type=str, default="mlp")
	parser.add_argument("--datafolder", type=str, default="23M")
	parser.add_argument("--anchors", default=26, type=int)
	#parser.add_argument("--early_stopping", action='store_true')
	parser.add_argument("--seed", default=1, type=int)
	parser.add_argument("--normalize", action='store_true')
	parser.add_argument("--batch_size", default=512, type=int)
	parser.add_argument("--lr", default=1e-3, type=float)
	#parser.add_argument("--min_lr", default=1e-6, type=float) 
	parser.add_argument("--weight_decay", default=0, type=float)
	parser.add_argument("--momentum", default=0.9, type=float)
	parser.add_argument("--num_epoch", default=300, type=int)
	parser.add_argument("--save_interval", default=3, type=int)
	parser.add_argument("--save_dir", default='models')
	parser.add_argument("--output_dir", default='result')
	parser.add_argument("--ckpt_name", default='hidden_500')
	parser.add_argument("--total_updates", default=50000, type=int)
	parser.add_argument("--optimizer", default='sgd', type=str)
	args = parser.parse_args()
	setattr(args, 'save_dir', os.path.join(curdir, args.save_dir, args.model))
	setattr(args, 'output_dir', os.path.join(curdir, args.output_dir, args.model))
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

@torch.no_grad()
def evaluate(net, test_dataset, f = None):
	test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers = 16)

	c1 = 0
	c2 = 0
	time_nn = 0
	if f is not None:
		f.write(str(len(test_dataset)) + '\n')
	for X, Y in test_dl:
		
		start = time.time()
		outputs = net(X.cuda())
		outputs = F.log_softmax(outputs, dim=1)
		pred = torch.argmax(outputs, dim=1, keepdim=True)
		end = time.time()
		time_nn += end - start
		if f is not None:
			for i in pred:
				f.write(str(i.item()) + '\n')
		#all correct classifications
		corr = torch.gather(Y.cuda(), 1, pred)
		c1 += torch.sum(corr > 0).item()
		#number of data which has at least 1 solution
		coy = torch.sum(Y.cuda(), dim=1)
		c2 += torch.sum(coy > 0).item()

	print(f"total time of NN:{time_nn:.4f}s, ", f"time of NN per data:{time_nn/len(test_dataset)*1e6:.4f}µs \n", "c1:", c1, "c2:", c2, "c", len(test_dataset), "rate:", c1 / len(test_dataset))


if __name__ == "__main__":
	
	args = get_args()
	print(args)
	torch.manual_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)

	os.makedirs(args.output_dir, exist_ok=True)

	basedir = os.path.dirname(os.path.abspath(__file__))
	X_test = np.loadtxt(args.datafolder+"/X_val.txt")
	Y_test = np.loadtxt(args.datafolder+"/Y_val.txt")
	anchors = np.loadtxt(args.datafolder+"/anchors.txt", skiprows=1)
	X_test_tensor = torch.Tensor(X_test)
	y_test_tensor = torch.Tensor(Y_test)
	anchors_tensor = torch.Tensor(anchors)
	Dataset = getattr(getattr(getattr(networks, args.model), args.model), 'Dataset')
	if args.model in ['mlp_mixer', "transformer"]:
		Dataset = partial(Dataset, anchors = anchors_tensor[:, :20])
	test_dataset = Dataset(X_test_tensor,y_test_tensor, args)
	
	net = torch.load(os.path.join(args.save_dir, args.ckpt_name + '.pt'), map_location='cpu').to('cuda')
	f = open(os.path.join(args.output_dir, args.ckpt_name + '.txt'), 'w')
	evaluate(net, test_dataset, f)