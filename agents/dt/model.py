# pylint: disable=invalid-name
"""Module for defining decision transformer's model."""
import torch
from typing import Optional, Tuple

from agents.dt.transformer import TransformerBlock, OutputPooler
from agents.dt.embeddings import DiscreteEmbedding, PositionEncoding
from agents.dt.tokenizer import Tokenizer


class Model(torch.nn.Module):
    """Transformer based model."""

    def __init__(
        self,
        discretisation_bins: int,
        number_of_blocks: int,
        number_of_heads: int,
        embedding_dimension: int,
        dropout: float,
        feedforward_hidden_dimension: int,
        tokenizer_mu: int,
        tokenizer_M: int,
        positional_encoder_table_dimension: int,
        layer_norm_epsilon: float,
        device: torch.device,
    ):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dimension=embedding_dimension,
                    number_of_heads=number_of_heads,
                    dropout=dropout,
                    feedforward_hidden_dimension=feedforward_hidden_dimension,
                    layer_norm_epsilon=layer_norm_epsilon,
                    device=device,
                )
                for _ in range(number_of_blocks)
            ]
        )

        self.output_pooler = OutputPooler(
            embedding_dimension=embedding_dimension,
            bins=discretisation_bins,
        )
        self.discrete_embedder = DiscreteEmbedding(
            embedding_number=discretisation_bins,
            embedding_dimension=embedding_dimension,
            device=device,
        )
        self.positional_encoder = PositionEncoding(
            table_dimension=positional_encoder_table_dimension,
            embedding_dimension=embedding_dimension,
            device=device,
        )
        self.tokenizer = Tokenizer(
            bins=discretisation_bins,
            mu=tokenizer_mu,
            M=tokenizer_M,
            device=device,
        )

    def predict(
        self,
        input_tokens: torch.tensor,
        obs_mask: torch.tensor,
        act_mask: torch.tensor,
        targets: Optional[torch.tensor] = None,
        target_act_mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Takes sequence, embeds, passes through transformer blocks and pools.
        Args:
            input_tokens: tensor of context-length obs-action tokens
                                of shape [batch_dim, context_length]
            obs_mask: tensor of obs_dim positions in input sequence,
                                shape [batch_dim, context_length]
            act_mask: tensor of act_dim positions in input sequence,
                                shape [batch_dim, context_length]
            targets: tensor of target variables,
                                shape [sequence_length, batch_size, embed_dim]
            target_act_mask: tensor of act_dim positions in target sequence,
        Returns:
            output: output array of shape [sequence_length, batch_size]
                                i.e. real-valued output
            loss: loss tensor of shape [batch_size,]
        """

        # embed sequence
        input_embeddings = self.embed(
            input_tokens=input_tokens, obs_mask=obs_mask, action_mask=act_mask
        )

        x = input_embeddings

        # pass through transformer blocks
        print("x.shape", x.shape)
        for block in self.blocks:
            x = block(x)

        # training
        output_bins, loss = self.output_pooler(
            x=x, targets=targets, target_action_mask=target_act_mask
        )

        output = self.tokenizer.detokenize(output_bins)

        return output, loss

    def embed(
        self,
        input_tokens: torch.tensor,
        obs_mask: torch.tensor,
        action_mask: torch.tensor,
    ):
        """
        Takes sequences of tokens of arbitrary length embeds them ready for model.
        Args:
            input_tokens: tensor of shape [batch_size, context_length]
            obs_mask: tensor of shape [batch_size, context_length]
            action_mask: tensor of shape [batch_size, context_length]
        Returns:
            embedded_sequence: sequence embedding tensor
                        of shape [batch_size, context_length, embed_dim]
        """
        # embed
        embedded_sequence = self.discrete_embedder(
            input_tokens
        )  # [batch, context, embed]

        # add positional encoding
        embedded_sequence = self.positional_encoder.embed(
            embedded_input_sequence=embedded_sequence,
            obs_mask=obs_mask,
            act_mask=action_mask,
        )

        return embedded_sequence
