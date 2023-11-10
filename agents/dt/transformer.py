# pylint: disable=[invalid-name, not-callable]
"""Blocks and poolers for transformers."""
import torch
from typing import Tuple, Optional


class TransformerBlock(torch.nn.Module):
    """Transformer block."""

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        dropout: float,
        feedforward_hidden_dimension: int,
        layer_norm_epsilon: float,
        device: torch.device,
    ):
        super().__init__()

        # attention
        self.attention = torch.torch.nn.MultiheadAttention(
            embed_dim=embedding_dimension,
            num_heads=number_of_heads,
            dropout=dropout,
            device=device,
            batch_first=True,
        )

        # attention dropout
        self.dropout = torch.torch.nn.Dropout(dropout)

        # feedforward
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embedding_dimension, feedforward_hidden_dimension),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(feedforward_hidden_dimension, embedding_dimension),
            torch.nn.GELU(),
        ).to(device)

        # regularisation
        self.layer_norm1 = torch.nn.LayerNorm(
            embedding_dimension, eps=layer_norm_epsilon, device=device
        )
        self.layer_norm2 = torch.nn.LayerNorm(
            embedding_dimension, eps=layer_norm_epsilon, device=device
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Passes tokenized embedding through transformer
        Args:
            inputs: tensor of shape [batch, context, embed_dim]
        Returns:
            x: tensor of shape [batch, context, embed_dim]
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


class OutputPooler(torch.nn.Module):
    """
    Takes output of transformer blocks, predicts distribution over
    token bins and selects bin with the highest probability.
    """

    def __init__(
        self,
        embedding_dimension: int,
        bins: int,
        device: torch.device,
    ):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.outputs = torch.nn.Sequential(
            torch.nn.Linear(embedding_dimension, bins),
        ).to(device)
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.bins = bins

    def forward(
        self,
        x: torch.tensor,
        targets: Optional[torch.tensor] = None,
        target_action_mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Takes output of transformer block and finds real-valued action dimension bin,
        and loss if targets are provided.
        Args:
            x: tensor of outputs from transformer block, shape
                    [batch, context_length, hidden_dim]
            targets: [Optional] tensor of targets, shape [batch, context_length]
            target_action_mask: [Optional] tensor of masks defining which indices
                    (actions) to include in loss, shape [context_length, batch]
        Returns:
            y: tensor of predicted action bins, shape [context_length, batch]
            loss: tensor of predictive loss, shape [batch_size]
        """

        logits = self.outputs(x)  # bin-wise predictions [batch, context_length, bins]
        probs = self.softmax(logits)
        y = torch.argmax(probs, dim=-1)  # [context_length, batch, 1]

        # if we pass targets calculate loss
        if targets is not None:
            # one hot encode targets
            print(f"logits nans: {torch.isnan(logits).any()}")
            sequence_loss = self.loss(
                logits.permute(0, 2, 1), targets
            )  # [batch, con_length]
            masked_loss = (
                target_action_mask * sequence_loss
            )  # loss only applied to action predictions
            loss = torch.sum(masked_loss)

        else:
            loss = torch.empty(0)

        return y, loss
