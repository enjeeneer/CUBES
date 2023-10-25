# pylint: disable=invalid-name, unused-argument
"""Module for decision transformer agent."""
from pathlib import Path

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple

from agents.base import AbstractAgent, Batch
from agents.dt.model import Model


class DecisionTransformer(AbstractAgent):
    """
    Decision Transformer agent.
    """

    def load(self, filepath: Path):
        pass

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
        batch_size: int,
    ):
        super().__init__(name="DecisionTransformer")

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
        self.batch_size = batch_size
        self.device = device

    def act(
        self,
        input_sequence: np.array,
        action_dimension: int,
        observation_mask: np.array,
        action_mask: np.array,
        reward_mask: Optional[np.array] = None,
    ) -> np.array:
        """
        Takes a sequence of observation-action pairs and returns an action by
        auto-regressively predicting the next action dimension.
        """
        action_dims = []

        for _ in range(action_dimension):
            # TODO: check if this input token/sequence bit is correct
            input_tokens = self.model.tokenizer.tokenize(input_sequence)
            output_sequence, _ = self.model.predict(
                input_tokens=input_tokens,
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

            (input_sequence, observation_mask, action_mask,) = self.update_sequences(
                sequence=input_sequence,
                obs_mask=observation_mask,
                act_mask=action_mask,
                values_to_add=output_sequence[:, -1],
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

        # tokenize / convert to tensors
        inputs = (self.model.tokenizer.tokenize(batch.inputs),)
        targets = self.model.tokenizer.tokenize(batch.targets)
        observation_masks = torch.tensor(
            batch.observation_masks, dtype=torch.int, device=self.device
        )
        action_masks = torch.tensor(
            batch.action_masks, dtype=torch.int, device=self.device
        )
        target_action_masks = torch.tensor(
            batch.target_action_masks, dtype=torch.int, device=self.device
        )

        _, loss = self.model.predict(
            input_tokens=inputs,
            obs_mask=observation_masks,
            act_mask=action_masks,
            targets=targets,
            target_act_mask=target_action_masks,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_norm_clip)
        self.optimizer.step()

        return {"loss": loss.item()}

    @staticmethod
    def update_sequences(
        sequence: np.ndarray,
        obs_mask: np.ndarray,
        act_mask: np.ndarray,
        values_to_add: np.ndarray,
        obs: bool = False,
        action: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Add news tokens to sequence and updates masks. Used
        during online rollout.
        Args:
            sequence: array, shape [context_length]
            obs_mask: array, shape [context_length]
            act_mask: array, shape [context_length]
            tokens: array, shape Union[[obs_dim,], [batch_size, act_dim]]
            obs: bool flag to indicate whether tokens are from observation
            action: bool flag to indicate whether tokens are from action
        Returns:
            sequence: array, shape [context_length]
            obs_mask: array, shape [context_length]
            act_mask: array, shape [context_length]
        """

        n_values = values_to_add.shape[0]

        # sequence
        sequence[:-n_values] = sequence[n_values:]
        sequence[-n_values:] = values_to_add

        # masks
        obs_mask[:-n_values] = obs_mask[n_values:]
        act_mask[:-n_values] = act_mask[n_values:]

        if obs:
            obs_mask[-n_values:] = np.arange(start=1, stop=n_values + 1)
            act_mask[-n_values:] = 0

        if action:
            obs_mask[-n_values:] = 0
            act_mask[-n_values:] = 1

        return sequence, obs_mask, act_mask
