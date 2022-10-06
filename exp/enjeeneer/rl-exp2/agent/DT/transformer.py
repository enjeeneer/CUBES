import torch
import torch.nn as nn
import torch.nn.functional
from typing import Dict, Optional, Tuple


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
        input_dims = input_shape[-1]  # TODO: confirm what this is

        # attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.cfg.embed_dim,
            num_heads=self.cfg.heads,
            dropout=self.cfg.dropout,
            device=self.cfg.device,
            kdim=self.cfg.key_value_size,
            vdim=self.cfg.key_value_size
            )

        # attention dropout
        self.dropout = nn.Dropout(self.cfg.dropout)

        # feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dims, self.cfg.hidden_dims),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim)
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
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.layer_norm2(inputs)
        x = self.feed_forward(x)
        x = x + residual

        return x


class OutputPooler(nn.Module):
    """
    Takes output of transformer blocks, predicts distribution over token bins and selects bin with highest probability.
    """
    def __init__(self, cfg: Dict):
        super(OutputPooler, self).__init__()
        self.cfg = cfg
        self.softmax = nn.Softmax()

    def build(self):
        self.outputs = nn.Sequential(
            nn.Linear(self.cfg.hidden_dim, self.cfg.bins),
        )

    def forward(self, inputs: torch.tensor, targets: Optional[torch.tensor], masks: Optional[torch.tensor]):
        """
        Takes output of transformer block and find real-valued action, and loss if targets are provided.
        :param masks:
        :param inputs: [Optional] tensor of masks defining which indices (actions) to include in loss,
                                                                                    shape [sequence_length, batch, bins]
        :param targets: [Optional] tensor of one-hot encoded targets, shape [sequence_length, batch, bins]
        :return:
        """

        x = torch.squeeze(inputs[:, 0:1, :], dim=1)  # TODO: check squeezing, i dont think its necessary
        logits = self.outputs(x)  # bin-wise predictions

        probs = self.softmax(logits)
        y = torch.argmax(probs, dim=-1)

        # if we pass targets calculate loss
        if targets:

            loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')  # [seq_length, batch, 1]

            assert loss.shape(inputs.shape[0], inputs.shape[1], 1)
            masked_loss = torch.sum(masks * targets, dim=0)  # [batch_size]

            return masked_loss

        return y

