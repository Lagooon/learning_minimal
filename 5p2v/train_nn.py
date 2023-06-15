import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import time
import random
import yaml
import socket
import wandb
import argparse
import networks.mlp.mlp

curdir = os.path.dirname(os.path.abspath(__file__))

def get_args():
	parser = argparse.ArgumentParser()
	#wandb
	parser.add_argument("--use-wandb", action='store_true')
	parser.add_argument("--wandb-project", type=str, default="cv-project")
	parser.add_argument("--wandb-group", type=str, default="mlp")
	parser.add_argument("--job-type", type=str, default="training")
	parser.add_argument("--wandb-name", type=str, default="")
	parser.add_argument("--user-name", type=str, default="dl_project_")
	#parser.add_argument("--hyperparameter-search", action='store_true')
	
	parser.add_argument("--model", type=str, default="mlp")
	parser.add_argument("--dataset-folder", type=str, default=os.path.join(curdir, "MODEL"))
	parser.add_argument("--anchors", default=26, type=int)
	#parser.add_argument("--early-stopping", action='store_true')
	parser.add_argument("--seed", default=1, type=int)
	parser.add_argument("--batch-size", default=32, type=int)
	parser.add_argument("--lr", default=1e-3, type=float)
	#parser.add_argument("--min-lr", default=1e-6, type=float) 
	parser.add_argument("--weight-decay", default=0, type=float)
	parser.add_argument("--momentum", default=0.9, type=float)
	parser.add_argument("--num-epoch", default=80, type=int)
	parser.add_argument("--save-interval", default=3, type=int)
	parser.add_argument("--save-dir", default='models')
	parser.add_argument("--total-updates", default=50000, type=int)
	parser.add_argument("--optimizer", default='sgd', type=str)
	args = parser.parse_args()
	setattr(args, 'save_dir', os.path.join(curdir, args.save_dir, args.model))
	'''
	parser.add_argument(
		'--gradient-accumulation-steps',
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
	# set up the network
	net = getattr(getattr(getattr(networks, args.model), args.model), 'Net')(args.anchors)
	#set up the optimizer
	criterion = nn.CrossEntropyLoss(ignore_index = 0)
	if args.optimizer == 'sgd':
		optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
	elif args.optimizer == 'adam':
		optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	else:
		print('No such optimizer')
		exit(1)
	# load the training and validation data
	X_train = np.loadtxt(args.dataset_folder+"/X_train.txt")
	Y_train = np.loadtxt(args.dataset_folder+"/Y_train.txt")
	X_test = np.loadtxt(args.dataset_folder+"/X_val.txt")
	Y_test = np.loadtxt(args.dataset_folder+"/Y_val.txt")
	X_train_tensor = torch.Tensor(X_train)
	y_train_tensor = torch.LongTensor(Y_train)
	X_test_tensor = torch.Tensor(X_test)
	y_test_tensor = torch.LongTensor(Y_test)
	train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
	train_dl = torch.utils.data.DataLoader(train_dataset,
											batch_size=args.batch_size,
											shuffle = True,
											num_workers = 8)
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
			outputs = net(X)
			loss = criterion(outputs, Y)
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
		outputs = net(X_test_tensor)
		end = time.time()
		t1 = end-start

		klasse = torch.argmax(outputs, dim=1)
		#all correct classifications
		c1 = sum(klasse==y_test_tensor)
		#correct tracks
		c1_1 = sum((klasse==y_test_tensor) * (klasse != 0))
		#correct trash
		c1_2 = sum((klasse==y_test_tensor) * (klasse == 0))
		#total trash
		c1_3 = sum((klasse == 0))
		#total tracks (non trash)
		c1_4 = sum((klasse != 0))

		print(c1, c1_1, c1_2, c1_3, c1_4, sum((y_test_tensor == 0)), sum((y_test_tensor != 0)))
		print(epoch+1, ' | ', c1_1.numpy(), ' | ', c1_2.numpy(), ' | ', c1_3.numpy() , ' | ', round(t1,6))
		print("")

		if args.use_wandb:  
			wandb.log({"eval_acc": c1_1 / c1_4}, step = epoch + 1)
			
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
			setattr(args, k, v)
	print(args)
	train(args)