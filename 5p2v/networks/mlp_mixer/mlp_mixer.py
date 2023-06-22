#https://arxiv.org/pdf/2105.01601.pdf
#https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py

import torch
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from torch.utils.data import Dataset
import networks

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
	def __init__(self, dim, fn):
		super().__init__()
		self.fn = fn
		self.norm = nn.LayerNorm(dim)

	def forward(self, x):
		return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		dense(dim, inner_dim),
		nn.GELU(),
		nn.Dropout(dropout),
		dense(inner_dim, dim),
		nn.Dropout(dropout)
	)

# batch_size * (input + all anchors) * 20
def Net(num_anchors, num_layers = 12, hidden_dim = 512, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
	channels = num_anchors + 1
	chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

	return nn.Sequential(
		nn.Linear(20, hidden_dim),
		*[nn.Sequential(
			PreNormResidual(hidden_dim, FeedForward(channels, expansion_factor, dropout, chan_first)),
			PreNormResidual(hidden_dim, FeedForward(hidden_dim, expansion_factor_token, dropout, chan_last))
		) for _ in range(num_layers)],
		nn.LayerNorm(hidden_dim),
		Reduce('b n c -> b c', 'mean'),
		nn.Linear(hidden_dim, num_anchors)
	)

class Dataset(Dataset):
	def __init__(self, X, Y, args, anchors):
		super().__init__()
		self.X = X
		self.Y = Y
		self.args = args
		self.anchors = anchors
		
	def __len__(self):
		return len(self.X)
	
	def __getitem__(self, index):
		#print(networks.mlp.mlp.normalize(torch.cat((self.X[index].unsqueeze(0), self.anchors), dim = 0), self.args))
		return networks.mlp.mlp.normalize(torch.cat((self.X[index].unsqueeze(0), self.anchors), dim = 0), self.args), self.Y[index]