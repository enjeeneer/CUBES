import torch
import torch.nn as nn
from typing import Dict, Union


class DiscreteEmbedding(nn.Module):
	"""
	Function for embedding a discrete observation/action token.
	See Figure 13 in Gato paper.
	"""
	def __init__(self, cfg: Dict):
		super(DiscreteEmbedding, self).__init__()

		self.cfg = cfg
		self.embedding = nn.Embedding(
			num_embeddings=self.cfg.emb_num,
			embedding_dim=self.cfg.emb_dim,
			device=self.cfg.device
		)

	def forward(self, inputs):
		"""
		Forward pass of embedding function
		:param inputs: tensor of shape (?)
		:return: embedding of shape (?)
		"""
		return self.embedding(inputs)


class PositionEncoding(nn.Module):
	"""
	Function for encoding the positon of an observation/action in a sequence.
	See Figure 16 in Gato paper.
	"""
	def __init__(self, cfg: Dict):
		super(PositionEncoding, self).__init__()

		self.cfg = cfg
		self.embedding = nn.Embedding(
			num_embeddings=self.cfg.token_seq_length,
			embedding_dim=self.cfg.layer_width,
			device=self.cfg.device
		)

	def forward(self, inputs):
		"""
		Finds positional encoding input sequence. Figure 16 in Gato paper.
		:param inputs: tensor of shape (batch_size, sequence_length, token_dim?)
		:return:
		"""
		input_size = torch.shape(inputs)[1]  # need to check this
		pos = torch.range(start=0, end=input_size)
		outputs = inputs + self.embedding(pos)

		return outputs


