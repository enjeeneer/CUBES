"""Module for SAC buffer."""

from typing import Tuple

import numpy as np
import torch

from agents.base import AbstractOnlineReplayBuffer


class SoftActorCriticReplayBuffer(AbstractOnlineReplayBuffer):
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

        self.rewards = np.empty((self.capacity, int(1)), dtype=np.float32)

        self.not_dones = np.empty((self.capacity, int(1)), dtype=np.float32)

        self.current_memory_index = int(0)

        self.full_memory = False

    def add(
        self,
        observation: np.array,
        next_observation: np.array,
        action: np.array,
        reward: np.array,
        done: np.array,
    ) -> None:
        """
        Stores transition in memory.
        Args:
            observation: array of shape [observation_length]
            next_observation: array of shape [observation_length]
            action: array of shape [action_length]
            reward: array of shape [1]
            done: array of shape [1]

        Returns:
            None
        """

        np.copyto(self.observations[self.current_memory_index], observation)
        np.copyto(self.next_observations[self.current_memory_index], next_observation)
        np.copyto(self.actions[self.current_memory_index], action)
        np.copyto(self.rewards[self.current_memory_index], reward)
        np.copyto(self.not_dones[self.current_memory_index], not done)

        # update index
        self.current_memory_index = int((self.current_memory_index + 1) % self.capacity)
        self.full_memory = self.full_memory or self.current_memory_index == 0

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples batch_size-many transitions from memory.
        Args:
            batch_size: numbers of transitions to sample.
        Returns:
            observations: tensor of shape
                            [batch_size, observation_length * self.history_length]
            next_observations: tensor of shape
                            [batch_size, observation_length * self.history_length]
            actions: tensor of shape [batch_size, action_length]
            rewards: tensor of shape [batch_size, 1]
            not_dones: tensor of shape [batch_size, 1]
        """

        sample_indices = np.random.randint(
            self.history_length,
            self.capacity if self.full_memory else self.current_memory_index,
            size=batch_size,
        )
        observation_slice = [
            slice(idx - self.history_length, idx) for idx in sample_indices
        ]

        observation_histories = torch.as_tensor(
            self.observations[observation_slice],
            device=self.device,
        ).float()
        print("observation_histories", observation_histories.shape)

        observation_histories = observation_histories.view(batch_size, -1)  # flatten

        print("observation_histories flattened", observation_histories.shape)
        actions = (torch.as_tensor(self.actions[sample_indices], device=self.device),)
        rewards = (torch.as_tensor(self.rewards[sample_indices], device=self.device),)
        next_observation_histories = (
            torch.as_tensor(
                self.next_observations[
                    sample_indices - self.history_length : sample_indices
                ],
                device=self.device,
            ).float(),
        )
        not_dones = torch.as_tensor(self.not_dones[sample_indices], device=self.device)

        return (
            observation_histories,
            actions,
            rewards,
            next_observation_histories,
            not_dones,
        )
