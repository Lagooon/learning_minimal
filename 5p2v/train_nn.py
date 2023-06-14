import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import time
import gc
import sys
import copy

class Net(nn.Module):
	def __init__(self, anchors, num_layers = 6, hidden_dim = 100):
		super(Net, self).__init__()

		layers = []
		input_dim = 20
		for i in range(num_layers):
			layers += [
				nn.Linear(input_dim, hidden_dim),
				nn.BatchNorm1d(hidden_dim),
				nn.PReLU(hidden_dim, 0.25)
			]
			input_dim = hidden_dim
		self.MLP = nn.Sequential(*layers)
		self.drop = nn.Dropout(0.5)
		self.fc = nn.Linear(100,anchors+1)

	def forward(self, x):
		x = self.MLP(x)
		x = self.drop(x)
		return self.fc(x)

if __name__ == "__main__":
	print("Neural network training")

	# parse the arguments
	argc = len(sys.argv)
	if argc < 3:
		print("Run as:")
		print("python3 train_nn.py model_folder trainParam")
		exit()
	model_folder = sys.argv[1]
	param_file = sys.argv[2]

	#read and parse the trainParam file
	f = open(param_file, "r")
	for i in range(0,9):
		f.readline()
	line = f.readline().split(" ")
	anchors = int(line[0])

	line = f.readline().split(" ")
	lr_ = float(line[0])

	line = f.readline().split(" ")
	momentum_ = float(line[0])

	line = f.readline().split(" ")
	weight_decay_ = float(line[0])

	line = f.readline().split(" ")
	batch_size_ = int(line[0])

	line = f.readline().split(" ")
	epochs = int(line[0])
	
	f.close()

	# set up the network
	net = Net(anchors)
	print(net)
	#set up the optimizer
	criterion = nn.CrossEntropyLoss(ignore_index = 0)
	optimizer = optim.SGD(net.parameters(), lr=lr_, momentum=momentum_, weight_decay=weight_decay_) # Adam and its variants converge faster but SGD with momentum reaches a better result

	# load the training and validation data
	print("Loading data from folder " + model_folder)
	X_train = np.loadtxt(model_folder+"/X_train.txt")
	Y_train = np.loadtxt(model_folder+"/Y_train.txt")
	a = np.zeros((27), dtype=np.int32)
	for y in Y_train:
		y = int(y)
		a[y] = a[y] + 1
	print(a)
	
	X_test = np.loadtxt(model_folder+"/X_val.txt")
	Y_test = np.loadtxt(model_folder+"/Y_val.txt")
	print("Data loaded.")
	print("Converting data to the pytorch tensors.")
	X_train_tensor = torch.Tensor(X_train)
	y_train_tensor = torch.LongTensor(Y_train)
	X_test_tensor = torch.Tensor(X_test)
	y_test_tensor = torch.LongTensor(Y_test)
	print("Data converted.")

	# create the datset
	train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
	#find the number of the processor cores
	num_workers = os.cpu_count()
	if 'sched_getaffinity' in dir(os):
		num_workers = len(os.sched_getaffinity(0))
	print(str(num_workers) + " processor cores used.")
	#create the data loader
	BS = batch_size_ # batch size
	train_dl = torch.utils.data.DataLoader(train_dataset,
											batch_size= batch_size_,
											shuffle = True, # important thing to do for training.
											num_workers = num_workers)

	# train the NN
	best_val = -1;
	best_net = copy.deepcopy(net)
	for epoch in range(epochs):
		print("Epoch "+str(epoch))
		net.train()
		#iterate over the batches in the training set
		#for i, data2 in en:
		for i, (X, Y) in enumerate(train_dl):
			if i%2000 == 0:
				print(i)
				gc.collect()
		
			#zero the gradient
			optimizer.zero_grad()

			#feed forward
			outputs = net(X)
		
			#compute the loss by backpropagation
			loss = criterion(outputs, Y)
			loss.backward()

			'''
			for i in net.children():
				
				if hasattr(i, 'weight'):

					print(i)
					print(i.weight.grad)
					print(torch.norm(i.weight.grad))
			'''
			#update the model
			optimizer.step()
			

		#validate
		net.eval()
		start = time.time()
		outputs = net(X_test_tensor)
		end = time.time()
		t1 = end-start;

		klasse = torch.argmax(outputs, dim=1)
		#all correct classifications
		c1 = sum(klasse==y_test_tensor);
		#correct tracks
		c1_1 = sum((klasse==y_test_tensor) * (klasse != 0));
		#correct trash
		c1_2 = sum((klasse==y_test_tensor) * (klasse == 0));
		#total trash
		c1_3 = sum((klasse == 0));
		#total tracks (non trash)
		c1_4 = sum((klasse != 0));

		print(c1, c1_1, c1_2, c1_3, c1_4, sum((y_test_tensor == 0)), sum((y_test_tensor != 0)))

		if c1_1.numpy() > best_val:
			best_val = c1_1.numpy()
			best_net = copy.deepcopy(net)
			
			print("Saving the network")
			f = open(model_folder+"/nn.txt", "w")
			layers = int(np.round((len(list(net.parameters()))+1)/3))
			f.write(str(layers)+"\n")
			id = 0
			np.set_printoptions(edgeitems=200, linewidth=1000000, precision=7, suppress=True)
			'''
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
			

		print(epoch+1, ' | ', c1_1.numpy(), ' | ', c1_2.numpy(), ' | ', c1_3.numpy() , ' | ', round(t1,6))
		print("")
