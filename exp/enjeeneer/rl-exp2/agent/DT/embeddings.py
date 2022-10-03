import torch
import torch.nn as nn
from typing import Dict, Union


class DiscreteEmbedding(nn.Module):
	"""
	Class for embedding a discrete observation/action token.
	See Figure 13 in Gato paper.
	"""
	def __init__(self, cfg: Dict):
		super(DiscreteEmbedding, self).__init__()

		self.cfg = cfg
		self.embedding = nn.Embedding(
			num_embeddings=1024,  # number of bins in my discretisation (i.e. one-hot encoding)
			embedding_dim=512,  # number of dimensions of embedded vector
			device='cpu'
		)

	def forward(self, inputs):
		"""
		Find embedding vector for a discretised obs/action vector.
		Takes tensor of batches discretised tokens of shape (*, 1), but the value in the last dimenision
		must be an int in the range [0, num_embeddings).

		:param inputs: tensor of shape (?)
		:return: embedding tensor of shape (*, embedding_dim)
		"""
		return self.embedding(inputs)


class PositionEncoding(nn.Module):
	"""
	Class for encoding the local-position of each observation / action dim. E.g. if an observation
	has dimension 5, we find create a local postional embedding for each of the 5 dimensions, to signpost
	where they sit in the observation. NB actions are always given the same positional encoding.
	See Figure 16 in Gato paper.
	"""
	def __init__(self, cfg: Dict):
		super(PositionEncoding, self).__init__()

		self.cfg = cfg
		self.embedding = nn.Embedding(
			num_embeddings=self.cfg.position_table_dim,  # larger value than any obs_dim
			embedding_dim=self.cfg.embedding_dim,
			device=self.cfg.device
		)

	def forward(self, inputs, actions=False):
		"""
		Finds positional encoding input sequence. Figure 16 in Gato paper.
		Takes tensor of observations or actions and finds their local positional embedding. NB: actions
		are always given the same positional embedding which we take, arbitrarily, as the last index in
		the table look-up

		:param inputs: embedding tensor of shape (*, *, obs_dim/act_dim, embedding_dim)
		:return outputs: embedding tensor with positional encoding added of same shape as input
		"""
		if actions:
			pos = torch.tensor([self.cfg.position_table_dim], dtype=torch.int)  # every action gets last index in table
		else:
			input_size = torch.shape(inputs)[-2]  # 2nd to last dimension is number of dims in obs, last is embedding_dim
			pos = torch.range(start=0, end=input_size)

		outputs = inputs + self.embedding(pos)

		return outputs


