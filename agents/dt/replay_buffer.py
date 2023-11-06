"""Module for decision transformer replay buffer."""

import numpy as np
import torch
from pathlib import Path
from agents.base import OfflineReplayBuffer, Batch


class DecisionTransformerReplayBuffer(OfflineReplayBuffer):
    """
    Abstract replay buffer class for storing
    transitions from an environment.
    """

    def add(self, *args, **kwargs):
        pass

    def __init__(
        self,
        device: torch.device,
        dataset_path: Path,
        rewards: bool = False,
    ):
        super().__init__(device=device)

        self.storage = {}

        self.load_offline_dataset(dataset_path=dataset_path)
        self.context_length = self.storage["inputs"][0].shape[-1]
        self.rewards = rewards

    def load_offline_dataset(self, dataset_path: Path) -> None:
        """
        Load offline dataset from path into storage.
        """

        dataset = dict(np.load(dataset_path, allow_pickle=True))

        self.storage["inputs"] = dataset["inputs"]
        self.storage["targets"] = dataset["targets"]
        self.storage["observation_masks"] = dataset["observation_masks"]
        self.storage["action_masks"] = dataset["action_masks"]
        self.storage["target_action_masks"] = dataset["target_action_masks"]

        if self.rewards:
            self.storage["reward_masks"] = dataset["reward_masks"]
        else:
            self.storage["reward_masks"] = np.zeros_like(self.storage["action_masks"])

    def sample(self, batch_size: int) -> Batch:
        """
        Samples Batch from the replay buffer.
        Args:
            batch_size: number of transitions to sample
        Returns:
            Batch: batch of transitions
        """

        if len(self.storage) == 0:
            raise ValueError("Replay buffer is empty.")

        batch_indices = np.random.randint(
            low=0, high=len(self.storage["inputs"]), size=batch_size
        )

        return Batch(
            inputs=self.storage["inputs"][batch_indices],
            targets=self.storage["targets"][batch_indices],
            observation_masks=self.storage["observation_masks"][batch_indices],
            action_masks=self.storage["action_masks"][batch_indices],
            reward_masks=self.storage["reward_masks"][batch_indices],
            target_action_masks=self.storage["target_action_masks"][batch_indices],
        )
