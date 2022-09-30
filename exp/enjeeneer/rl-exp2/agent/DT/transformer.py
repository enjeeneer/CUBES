import torch
import torch.nn as nn
from typing import Dict


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Dict):
        super(TransformerBlock, self).__init__()

        self.cfg = cfg
        self.attention = self.feed_forward = self.dropout = None
        self.layer_norm1 = self.layer_norm2 = None

    def build(self, input_shape):
        """
        Builds one transformer block.
        """
        input_dims = input_shape[-1] # need to confirm what this is

        # attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.cfg.embed_dim,
            num_heads=self.cfg.num_att_heads,
            dropout=self.cfg.dropout,
            device=self.cfg.device)

        # attention dropout
        self.dropout = nn.Dropout(self.cfg.dropout)

        # feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dims, self.cfg.hidden_dims),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dims, self.cfg.hidden_dims)
        )

        # regularisation
        self.layer_norm1 = nn.LayerNorm(normalized_shape=self.cfg.layer_norm_shape,
                                        eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=self.cfg.layer_norm_shape,
                                        eps=1e-6)

    def forward(self, inputs):
        """
        Passes tokensized embedding through
        :param inputs:
        :return:
        """
        residual = inputs
        x = self.layer_norm1(inputs)
        x = self.attention(x, x, x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.layer_norm2(inputs)
        x = self.feed_forward(x)
        x = x + residual

        return x


class OutputPooler(nn.Module):
    def __init__(self, cfg: Dict):
        super(OutputPooler, self).__init__()
        self.cfg = cfg

    def build(self):
        self.outputs = nn.Sequential(
            nn.Linear(self.cfg.hidden_dims, self.cfg.output_dims),
            nn.Tanh()
        )

    def forward(self, inputs):
        x = torch.squeeze(inputs[:, 0:1, :], dim=1) # need to work out why we squeeze
        x = self.outputs(x)

        return x

