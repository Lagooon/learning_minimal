from typing import Dict, List, Optional, Tuple
from torch import nn, Tensor
import math
import copy
import json
import numpy as np
import random
import torch
import torch.nn.functional as F
import networks

Dataset = networks.mlp_mixer.mlp_mixer.Dataset

ACT2FN = {
	"relu": F.relu,
	"gelu": F.gelu,
	"tanh": torch.tanh,
	"sigmoid": torch.sigmoid,
}


class Config(object):
	dropout = 0.1
	attention_dropout = 0.0
	encoder_layerdrop = 0.0
	decoder_layerdrop = 0.0
	scale_embedding = None
	static_position_embeddings = False
	normalize_before = False
	activation_function = "gelu"
	activation_dropout = 0.0
	normalize_embedding = True
	add_final_layer_norm = False
	init_std = 0.02

	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)

class EncoderLayer(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.embed_dim = config.n_embed
		self.self_attn = Attention(self.embed_dim,
								   config.n_head,
								   dropout=config.attention_dropout)
		self.normalize_before = config.normalize_before
		self.self_attn_layer_norm = LayerNorm(self.embed_dim)
		self.dropout = config.dropout
		self.activation_fn = ACT2FN[config.activation_function]
		self.activation_dropout = config.activation_dropout
		self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim)
		self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim)
		self.final_layer_norm = LayerNorm(self.embed_dim)

	def forward(self, x, encoder_padding_mask):
		"""
		Args:
			x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
			encoder_padding_mask (ByteTensor): binary ByteTensor of shape
				`(batch, src_len)` where padding elements are indicated by ``1``.
			for t_tgt, t_src is excluded (or masked out), =0 means it is
			included in attention

		Returns:
			encoded output of shape `(seq_len, batch, embed_dim)`
		"""
		residual = x
		if self.normalize_before:
			x = self.self_attn_layer_norm(x)
		x = self.self_attn(query=x,
						   key=x,
						   key_padding_mask=encoder_padding_mask)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = residual + x
		if not self.normalize_before:
			x = self.self_attn_layer_norm(x)

		residual = x
		if self.normalize_before:
			x = self.final_layer_norm(x)
		x = self.activation_fn(self.fc1(x))
		x = F.dropout(x, p=self.activation_dropout, training=self.training)
		x = self.fc2(x)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = residual + x
		if not self.normalize_before:
			x = self.final_layer_norm(x)
		if torch.isinf(x).any() or torch.isnan(x).any():
			clamp_value = torch.finfo(x.dtype).max - 1000
			x = torch.clamp(x, min=-clamp_value, max=clamp_value)
		return x


class TransformerEncoder(nn.Module):
	"""
	Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
	is a :class:`EncoderLayer`.

	Args:
		config: 
	"""

	def __init__(self, config, embed_tokens, embed_positions):
		super().__init__()

		self.dropout = config.dropout
		self.layerdrop = config.encoder_layerdrop
		embed_dim = config.n_embed

		self.embed_scale = math.sqrt(
			embed_dim) if config.scale_embedding else 1.0

		self.embed_tokens = embed_tokens
		self.embed_positions = embed_positions
		'''
		if config.static_position_embeddings:
			self.embed_positions = SinusoidalPositionalEmbedding(
				config.max_position_embeddings, embed_dim, self.padding_idx)
		else:
			self.embed_positions = LearnedPositionalEmbedding(
				config.max_position_embeddings,
				embed_dim,
				self.padding_idx,
				config.extra_pos_embeddings,
			)
		'''
		self.layers = nn.ModuleList(
			[EncoderLayer(config) for _ in range(config.n_layer)])
		self.layernorm_embedding = LayerNorm(
			embed_dim) if config.normalize_embedding else nn.Identity()

		self.layer_norm = LayerNorm(
			config.n_embed) if config.add_final_layer_norm else None

	def forward(self, input_ids, attention_mask=None):
		"""
		Args:
			input_ids (LongTensor): tokens in the source language of shape
				`(batch, src_len)`
			attention_mask (torch.LongTensor): indicating which indices are padding tokens.
		"""
		# check attention mask and invert

		inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
		embed_pos = self.embed_positions(input_ids)
		x = inputs_embeds + embed_pos
		x = self.layernorm_embedding(x)
		x = F.dropout(x, p=self.dropout, training=self.training)

		# B x T x C -> T x B x C
		x = x.transpose(0, 1)

		for encoder_layer in self.layers:

			# add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
			dropout_probability = random.uniform(0, 1)
			if self.training and (dropout_probability <
								  self.layerdrop):  # skip the layer
				continue
			else:
				x = encoder_layer(x, attention_mask)

		if self.layer_norm:
			x = self.layer_norm(x)

		# T x B x C -> B x T x C
		x = x.transpose(0, 1)

		return x

class Attention(nn.Module):
	"""Multi-headed attention from 'Attention Is All You Need' paper"""

	def __init__(
			self,
			embed_dim,
			num_heads,
			dropout=0.0,
			bias=True,
			encoder_decoder_attention=False,  # otherwise self_attention
	):
		super().__init__()
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.dropout = dropout
		self.head_dim = embed_dim // num_heads
		assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
		self.scaling = self.head_dim**-0.5

		self.encoder_decoder_attention = encoder_decoder_attention
		self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

	def _shape(self, tensor, seq_len, bsz):
		return tensor.contiguous().view(seq_len, bsz * self.num_heads,
										self.head_dim).transpose(0, 1)

	def forward(
		self,
		query,
		key: Optional[Tensor],
		key_padding_mask: Optional[Tensor] = None,
		attn_mask: Optional[Tensor] = None,
	) -> Tuple[Tensor, Optional[Tensor]]:
		"""
		Compute the attention output. You need to apply key_padding_mask and attn_mask before softmax operation.

		Args:
			query (torch.Tensor): The input query tensor, shape (seq_len, batch_size, embed_dim).
			key (Optional[torch.Tensor]): The input key tensor, shape (seq_len, batch_size, embed_dim).
										 If None, it's assumed to be the same as the query tensor.
			key_padding_mask (Optional[torch.Tensor]): The key padding mask tensor, shape (batch_size, seq_len).
													  Default: None
			attn_mask (Optional[torch.Tensor]): The attention mask tensor, shape (seq_len, seq_len).
											   Default: None

		Returns:
			attn_output (torch.Tensor): The attention output tensor, shape (seq_len, batch_size, embed_dim).

		"""
		##############################################################################
		#				  TODO: You need to complete the code here				  #
		##############################################################################
		
		#if key is None: # In LMmodel, I want to use Decoder layer without cross attention, so if key is None, means this is cross attention, we return zeros
		#	return torch.zeros_like(query, device=query.device)
		
		bsz = query.shape[1]
		q = self.scaling * self.q_proj(query)
		k = self.k_proj(key if self.encoder_decoder_attention else query)
		v = self.v_proj(key if self.encoder_decoder_attention else query)
		attn = torch.bmm(self._shape(q, -1, bsz), self._shape(k, -1, bsz).transpose(1, 2))
		if attn_mask is not None:
			attn += attn_mask
		if key_padding_mask is not None:
			attn = attn.reshape(bsz, self.num_heads, attn.shape[-2], attn.shape[-1])
			attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -torch.inf)
			attn = attn.reshape(-1, attn.shape[-2], attn.shape[-1])
		attn = F.softmax(attn, dim=-1)
		out = torch.bmm(attn, self._shape(v, -1, bsz))
		out = out.transpose(0, 1).reshape(-1, bsz, self.embed_dim)
		attn_output = self.out_proj(out)
		##############################################################################
		#							  END OF YOUR CODE							  #
		##############################################################################
		return attn_output

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):

	return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

class Net(nn.Module):
	def __init__(self, num_anchors, num_layers = 6, hidden_dim = 100):
		super(Net, self).__init__()

		self.config = Config(
            n_embed=hidden_dim,
            n_layer=num_layers,
            n_head=8,
            ffn_dim=hidden_dim,
        )
		self.num_anchors = num_anchors
		self.embed = nn.Linear(20, hidden_dim)
		self.embed_pos = nn.Embedding(2, hidden_dim)
		self.encoder = TransformerEncoder(self.config, self.embed, self.embed_positions)
		self.fc = nn.Linear(hidden_dim, num_anchors)

	# p is 1 and anchors are 0
	def embed_positions(self, input):
		bsz = input.shape[0]
		pos = torch.cat((torch.ones((bsz, 1), dtype = torch.int), torch.zeros((bsz, self.num_anchors), dtype = torch.int)), dim = 1).cuda()
		return self.embed_pos(pos)

	def forward(self, x):
		x = self.encoder(x)[:, 0, :]
		return self.fc(x)
	