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
import networks.mlp.mlp, networks.mlp_mixer.mlp_mixer
from einops.layers.torch import Rearrange, Reduce

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
	parser.add_argument("--dataset_folder", type=str, default=os.path.join(curdir, "MODEL"))
	parser.add_argument("--anchors", default=26, type=int)
	#parser.add_argument("--early_stopping", action='store_true')
	parser.add_argument("--seed", default=1, type=int)
	parser.add_argument("--batch_size", default=128, type=int)
	parser.add_argument("--lr", default=1e-3, type=float)
	#parser.add_argument("--min_lr", default=1e-6, type=float) 
	parser.add_argument("--weight_decay", default=0, type=float)
	parser.add_argument("--momentum", default=0.9, type=float)
	parser.add_argument("--num_epoch", default=80, type=int)
	parser.add_argument("--save_interval", default=3, type=int)
	parser.add_argument("--save_dir", default='models')
	parser.add_argument("--total_updates", default=50000, type=int)
	parser.add_argument("--optimizer", default='sgd', type=str)
	args = parser.parse_args()
	setattr(args, 'save_dir', os.path.join(curdir, args.save_dir, args.model))
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
			wandb.run.name = 'lr{:.2e}-weightdecay{:.2e}-{}-seed{}'.format(args.lr, args.weight_decay, args.optimizer, args.seed)
	
	os.makedirs(args.save_dir, exist_ok=True)
	Net = getattr(getattr(getattr(networks, args.model), args.model), 'Net')
	Dataset = getattr(getattr(getattr(networks, args.model), args.model), 'Dataset')
	# set up the network
	net = Net(args.anchors)
	net = net.cuda()
	#set up the optimizer
	criterion = nn.KLDivLoss(reduction = "batchmean")
	if args.optimizer == 'sgd':
		optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
	elif args.optimizer == 'adam':
		optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	else:
		print('No such optimizer')
		exit(1)
	# load the training and validation data
	X_train = np.loadtxt(args.dataset_folder+"/X_train.txt")
	Y_train = np.loadtxt(args.dataset_folder+"/Y_train.txt")
	X_test = np.loadtxt(args.dataset_folder+"/X_val.txt")
	Y_test = np.loadtxt(args.dataset_folder+"/Y_val.txt")
	anchors = np.loadtxt(args.dataset_folder+"/anchors.txt", skiprows=1)
	X_train_tensor = torch.Tensor(X_train)
	y_train_tensor = torch.Tensor(Y_train)
	X_test_tensor = torch.Tensor(X_test)
	y_test_tensor = torch.Tensor(Y_test)
	anchors_tensor = torch.Tensor(anchors)
	if args.model == 'mlp_mixer':
		Dataset = partial(Dataset, anchors = anchors_tensor[:, :20])
	y_train_tensor = y_train_tensor / (torch.sum(y_train_tensor, dim=1, keepdim=True)+1e-10)
	y_test_tensor = y_test_tensor / (torch.sum(y_test_tensor, dim=1, keepdim=True)+1e-10)
	train_dataset = Dataset(X_train_tensor,y_train_tensor)
	train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle = True, num_workers = 4)
	test_dataset = Dataset(X_test_tensor,y_test_tensor)
	test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers =4)
	'''
	a = np.zeros((27), dtype=np.int32)
	for y in Y_train:
		y = int(y)
		a[y] = a[y] + 1
	print(a)
	'''

	# train the NN
	#best_val = -1
	#best_net = copy.deepcopy(net)
	for epoch in range(args.num_epoch):
		#print("Epoch "+str(epoch))
		net.train()
		losses = []
		for i, (X, Y) in enumerate(train_dl):
			#if i%2000 == 0:
			#	print(i)
			#	gc.collect()
		
			optimizer.zero_grad()
			outputs = net(X.cuda())
			outputs = F.log_softmax(outputs, dim=1)
			loss = criterion(outputs, Y.cuda())
			loss.backward()
			optimizer.step()
			losses.append(loss.item())
			'''
			for i in net.children():
				
				if hasattr(i, 'weight'):

					print(i)
					print(i.weight.grad)
					print(torch.norm(i.weight.grad))
			'''

		if args.use_wandb:  
			wandb.log({"loss": np.mean(losses), 
				"lr": optimizer.param_groups[0]['lr']}, step = epoch + 1)

		#validate
		net.eval()

		start = time.time()

		c1 = 0
		c2 = 0
		for X, Y in test_dl:
		
			outputs = net(X.cuda())
			outputs = F.log_softmax(outputs, dim=1)
			loss = criterion(outputs, Y.cuda())
			losses.append(loss.item())
			
			pred = torch.argmax(outputs, dim=1, keepdim=True)
			#all correct classifications
			corr = torch.gather(Y.cuda(), 1, pred)
			c1 += torch.sum(corr > 0).item()
			#number of data which has at least 1 solution
			coy = torch.sum(Y.cuda(), dim=1)
			c2 += torch.sum(coy > 0).item()

		end = time.time()
		t1 = end-start

		print("epoch:", epoch+1, "c1:", c1, "c2:", c2, "c", y_test_tensor.shape[0], "rate:", c1/y_test_tensor.shape[0])

		if args.use_wandb:  
			wandb.log({"eval_acc": c1 / c2}, step = epoch + 1)
			
		if epoch % args.save_interval == 0:
			torch.save(net, args.save_dir + "/ckpt_{}.pt".format(epoch + 1))

		'''
		if c1_1.numpy() > best_val:
			best_val = c1_1.numpy()
			best_net = copy.deepcopy(net)
			
			print("Saving the network")
			f = open(model_folder+"/nn.txt", "w")
			layers = int(np.round((len(list(net.parameters()))+1)/3))
			f.write(str(layers)+"\n")
			id = 0
			np.set_printoptions(edgeitems=200, linewidth=1000000, precision=7, suppress=True)
			for param in net.parameters():
				print(param)
				if id%3==0:
					print(str(param.size(0))+" "+str(param.size(1)))
					f.write(str(param.size(0))+" "+str(param.size(1))+"\n")
					a = param.detach().numpy()
					for i in range(param.size(0)):
						row = a[i,:]
						f.write(' '.join(map(str, row))+"\n")
				else:
					print(str(param.size(0)))
					f.write(str(param.size(0))+" 1\n")
					a = param.detach().numpy()
					f.write(' '.join(map(str, a))+"\n")
				id = id+1
		'''

if __name__ == "__main__":
	args = get_args()
	torch.manual_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)
	
	with open(os.path.join(curdir, 'networks', args.model, 'config.yaml'), 'r') as f:
		config = yaml.load(f, Loader=yaml.FullLoader)
		for k, v in config.items():
			if f'--{k}' not in sys.argv[1:]:
				setattr(args, k, v)
	print(args)
	train(args)