# pylint: disable=invalid-name
"""Module for decision transformer agent."""

from abc import ABC

import numpy as np
import torch
from typing import List, Dict

from agents.base import AbstractAgent, Batch
from agents.dt.model import Model


class DecisionTransformer(AbstractAgent, ABC):
    """
    Decision Transformer agent.
    """

    def __init__(
        self,
        discretisation_bins: int,
        number_of_blocks: int,
        number_of_heads: int,
        embedding_dimension: int,
        dropout: float,
        feedforward_hidden_dimension: int,
        layer_norm_epsilon: float,
        tokenizer_mu: int,
        positional_encoder_table_dimension: int,
        betas: List[float],
        learning_rate: float,
        weight_decay: float,
        gradient_norm_clip: float,
        optimiser_epsilon: float,
        device: torch.device,
    ):
        super(AbstractAgent, self).__init__(name="DecisionTransformer")

        self.model = Model(
            discretisation_bins=discretisation_bins,
            number_of_blocks=number_of_blocks,
            number_of_heads=number_of_heads,
            embedding_dimension=embedding_dimension,
            dropout=dropout,
            feedforward_hidden_dimension=feedforward_hidden_dimension,
            tokenizer_mu=tokenizer_mu,
            positional_encoder_table_dimension=positional_encoder_table_dimension,
            layer_norm_epsilon=layer_norm_epsilon,
            device=device,
        )

        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=learning_rate,
            eps=optimiser_epsilon,
            betas=betas,
            weight_decay=weight_decay,
        )

        self.gradient_norm_clip = gradient_norm_clip
        self.device = device

    def act(
        self,
        input_sequence: np.array,
        action_dimension: int,
        observation_mask: np.array,
        action_mask: np.array,
    ) -> np.array:
        """
        Takes a sequence of observation-action pairs and returns an action by
        auto-regressively predicting the next action dimension.
        """
        action_dims = []
        input_tokens = self.model.tokenizer.tokenize(input_sequence)

        for _ in range(action_dimension):

            output_sequence, _ = self.model.predict(
                input_sequence=torch.tensor(
                    [input_tokens], dtype=torch.int, device=self.device
                ),
                obs_mask=torch.tensor(
                    [observation_mask], dtype=torch.int, device=self.device
                ),
                act_mask=torch.tensor(
                    [action_mask], dtype=torch.int, device=self.device
                ),
            )
            output_sequence = output_sequence.detach().numpy()
            action_dims.append(
                output_sequence[:, -1]
            )  # action dim is final dim of predicted sequence
            (
                input_tokens,
                observation_mask,
                action_mask,
            ) = self.tokenizer.add_tokens_to_sequence(
                sequence=input_tokens,
                obs_mask=observation_mask,
                act_mask=action_mask,
                tokens=output_sequence[:, -1],
                action=True,
            )

        action = np.array(action_dims, dtype=np.float32).flatten()

        return action

    def update(self, batch: Batch) -> Dict[str, float]:
        """
        Performs one update step on the model using the given batch of data.
        Args:
            batch: batch of sequenced data
        Returns:
            metrics: dictionary of metrics
        """

        _, loss = self.model.predict(
            input_sequence=batch.input_sequences,
            obs_mask=batch.observation_masks,
            act_mask=batch.action_masks,
            targets=batch.targets,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_norm_clip)
        self.optimizer.step()

        return {"loss": loss.item()}
