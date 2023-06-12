# pylint: disable=invalid-name

"""Module for critic network."""

from typing import Tuple

import torch

from agents.base import AbstractCritic


class DoubleQCritic(torch.nn.Module):
    """Critic network employing double Q learning."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        hidden_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
    ):
        super().__init__()

        self.Q1 = AbstractCritic(
            observation_length=observation_length,
            action_length=action_length,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
        )
        self.Q2 = AbstractCritic(
            observation_length=observation_length,
            action_length=action_length,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
        )
        self.outputs = {}

    def forward(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes obs-action pair through q functions.
        Args:
            observation: tensor of shape [batch_dimension, observation_length]
            action: tensor of shape [batch_dimension, action_length]

        Returns:
            q1: q value from first q function
            q2: q value from second q function
        """
        assert observation.size(0) == action.size(0)

        observation_action = torch.cat([observation, action], dim=-1)
        q1 = self.Q1.forward(observation_action)
        q2 = self.Q2.forward(observation_action)

        self.outputs["q1"] = q1
        self.outputs["q2"] = q2

        return q1, q2
