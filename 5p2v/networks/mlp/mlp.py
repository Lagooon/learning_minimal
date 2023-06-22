import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

class Net(nn.Module):
	def __init__(self, num_anchors, num_layers = 6, hidden_dim = 100):
		super(Net, self).__init__()

		layers = []
		input_dim = 20
		for i in range(num_layers):
			layers += [
				nn.Linear(input_dim, hidden_dim),
				nn.BatchNorm1d(hidden_dim),
				nn.GELU(),
				nn.Dropout(0.1),
			]
			input_dim = hidden_dim
		self.MLP = nn.Sequential(*layers)
		self.drop = nn.Dropout(0.5)
		self.fc = nn.Linear(hidden_dim, num_anchors)

	def forward(self, x):
		x = self.MLP(x)
		#x = self.drop(x)
		return self.fc(x)
	
#def Dataset(X, Y):
#	return TensorDataset(X, Y)

def normalize(x, args):
	if not args.normalize:
		return x
	if len(x.shape) == 1:
		return ((x.reshape(-1, 2) - args.mean) / (args.std + 1e-10)).reshape(-1)
	else:
		c = x.shape[0]
		return ((x.reshape(-1, 2) - args.mean) / (args.std + 1e-10)).reshape(c, -1)

class Dataset(Dataset):
	def __init__(self, X, Y, args):
		super().__init__()
		self.X = X
		self.Y = Y
		self.args = args
		
	def __len__(self):
		return len(self.X)
	
	def __getitem__(self, index):
		#print(self.X[index])
		#print(normalize(self.X[index], self.args))
		return normalize(self.X[index], self.args), self.Y[index]