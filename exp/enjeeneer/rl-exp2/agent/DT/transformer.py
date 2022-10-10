import torch
import torch.nn as nn
import torch.nn.functional
from typing import Dict, Optional


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Dict, block: str):
        super(TransformerBlock, self).__init__()

        self.layer = block
        self.cfg = cfg

        # attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.cfg.embed_dim,
            num_heads=self.cfg.heads,
            dropout=self.cfg.dropout,
            device=self.cfg.device,
            batch_first=True
        )

        # attention dropout
        self.dropout = nn.Dropout(self.cfg.dropout)

        # feedforward
        self.feed_forward = nn.Sequential(
            nn.Linear(self.cfg.embed_dim, self.cfg.feedforward_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.feedforward_hidden_dim, self.cfg.embed_dim),
            nn.GELU()
        )

        # regularisation
        self.layer_norm1 = nn.LayerNorm(self.cfg.embed_dim,
                                        eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.cfg.embed_dim,
                                        eps=1e-6)

    def forward(self, inputs):
        """
        Passes tokensized embedding through transformer
        :param inputs: tensor of shape [batch, context, embed_dim]
        :return x:
        """
        inputs = inputs.permute(1, 0, 2)
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
        self.softmax = nn.Softmax(dim=-1)
        self.outputs = nn.Sequential(
            nn.Linear(self.cfg.embed_dim, self.cfg.bins),
        )
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, x: torch.tensor,
                targets: Optional[torch.tensor] = None,
                action_mask: Optional[torch.tensor] = None):
        """
        Takes output of transformer block and find real-valued action, and loss if targets are provided.
        :param x: tensor of outputs from transformer block, shape [batch, context_length, hidden_dim]
        :param targets: [Optional] tensor of one-hot encoded targets, shape [context_length, batch, bins]
        :param action_mask: [Optional] tensor of masks defining which indices (actions) to include in loss,
                                                                                shape [context_length, batch]
        :return loss: tensor of predictive loss, shape [batch_size]
        :return y:
        """

        logits = self.outputs(x)  # bin-wise predictions [batch, context_length, bins]
        probs = self.softmax(logits)
        y = torch.argmax(probs, dim=-1)  # [context_length, batch, 1]
        print('probs shape:', logits.shape)

        # if we pass targets calculate loss
        if targets is not None:
            sequence_loss = self.loss(logits, targets)  # [batch, con_length]
            masked_loss = action_mask * sequence_loss  # loss only applied to action predictions
            loss = torch.sum(masked_loss)

            return loss

        return y
