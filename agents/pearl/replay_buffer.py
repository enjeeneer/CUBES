"""Module for PEARL buffer."""

from typing import Tuple

import numpy as np
import torch

from agents.base import AbstractOnlineReplayBuffer


class PEARLReplayBuffer(AbstractOnlineReplayBuffer):
    """SAC replay buffer."""

    def __init__(
        self,
        capacity: int,
        observation_length: int,
        history_length: int,
        action_length: int,
        device: torch.device,
    ):
        super().__init__(
            capacity=capacity,
            observation_length=observation_length,
            action_length=action_length,
            device=device,
        )
        self.history_length = int(history_length)

        self.observations = np.zeros(
            (self.capacity, self.observation_length),
            dtype=np.float32,
        )

        self.next_observations = np.zeros(
            (self.capacity, self.observation_length),
            dtype=np.float32,
        )

        self.actions = np.empty(
            (
                self.capacity,
                self.action_length,
            ),
            dtype=np.float32,
        )

        self.current_memory_index = int(0)

        self.full_memory = False

    def add(
        self,
        observation: np.array,
        next_observation: np.array,
        action: np.array,
    ) -> None:
        """
        Stores transition in memory.
        Args:
            observation: array of shape [observation_length]
            next_observation: array of shape [observation_length]
            action: array of shape [action_length]
        Returns:
            None
        """

        np.copyto(self.observations[self.current_memory_index], observation)
        np.copyto(self.next_observations[self.current_memory_index], next_observation)
        np.copyto(self.actions[self.current_memory_index], action)

        # update index
        self.current_memory_index = int((self.current_memory_index + 1) % self.capacity)
        self.full_memory = self.full_memory or self.current_memory_index == 0

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples batch_size-many transitions from memory.
        Args:
            batch_size: numbers of transitions to sample.
        Returns:
            observation_histories: tensor of shape
            [batch_size, observation_length * (self.history_length + 1)]
            actions: tensor of shape [batch_size, action_length]
            next_observations: tensor of shape [batch_size, observation_length]
        """

        sample_indices = np.random.randint(
            self.history_length,
            self.capacity if self.full_memory else self.current_memory_index,
            size=batch_size,
        )
        observation_slice = [
            np.arange(idx - self.history_length, idx + 0.1) for idx in sample_indices
        ]  # 0.1 to include idx

        # model inputs
        observation_histories = torch.as_tensor(
            self.observations[observation_slice],
            device=self.device,
        ).float()
        observation_histories = observation_histories.view(batch_size, -1)  # flatten
        actions = torch.as_tensor(self.actions[sample_indices], device=self.device)

        # model outputs
        next_observations = torch.as_tensor(
            self.next_observations[sample_indices],
            device=self.device,
        ).float()

        return observation_histories, actions, next_observations
