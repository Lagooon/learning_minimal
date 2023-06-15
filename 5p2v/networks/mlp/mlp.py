import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

class Net(nn.Module):
	def __init__(self, anchors, num_layers = 6, hidden_dim = 100):
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
		self.fc = nn.Linear(100,anchors)

	def forward(self, x):
		x = self.MLP(x)
		#x = self.drop(x)
		return self.fc(x)
	
def Dataset(X, Y):
	return TensorDataset(X, Y)