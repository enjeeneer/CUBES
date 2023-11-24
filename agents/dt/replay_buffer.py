"""Module for decision transformer replay buffer."""

import numpy as np
import torch
from pathlib import Path
from agents.base import OfflineReplayBuffer, Batch
from typing import Tuple
from loguru import logger


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
        train_split: float = 0.8,
    ):
        super().__init__(device=device)

        self.train_storage = {}
        self.val_storage = {}
        self.rewards = rewards
        self.train_split = train_split
        self.val_split = 1 - train_split
        self.load_offline_dataset(dataset_path=dataset_path)
        self.context_length = self.train_storage["inputs"][0].shape[-1]

    def load_offline_dataset(self, dataset_path: Path) -> None:
        """
        Load offline dataset from path into storage.
        """
        logger.info(f"Loading offline dataset from: {dataset_path}")
        dataset = dict(np.load(dataset_path, allow_pickle=True))

        # split dataset into train and val
        indexes = np.arange(len(dataset["inputs"]))
        np.random.shuffle(indexes)
        train_indexes = indexes[: int(self.train_split * len(indexes))]
        val_indexes = indexes[int(self.train_split * len(indexes)) :]

        self.train_storage["inputs"] = dataset["inputs"][train_indexes]
        self.train_storage["targets"] = dataset["targets"][train_indexes]
        self.train_storage["observation_masks"] = dataset["observation_masks"][
            train_indexes
        ]
        self.train_storage["action_masks"] = dataset["action_masks"][train_indexes]
        self.train_storage["target_action_masks"] = dataset["target_action_masks"][
            train_indexes
        ]

        self.val_storage["inputs"] = dataset["inputs"][val_indexes]
        self.val_storage["targets"] = dataset["targets"][val_indexes]
        self.val_storage["observation_masks"] = dataset["observation_masks"][
            val_indexes
        ]
        self.val_storage["action_masks"] = dataset["action_masks"][val_indexes]
        self.val_storage["target_action_masks"] = dataset["target_action_masks"][
            val_indexes
        ]

        if self.rewards:
            self.train_storage["reward_masks"] = dataset["reward_masks"]
            self.val_storage["reward_masks"] = dataset["reward_masks"]
        else:
            self.train_storage["reward_masks"] = np.zeros_like(
                self.train_storage["action_masks"]
            )
            self.val_storage["reward_masks"] = np.zeros_like(
                self.val_storage["action_masks"]
            )

    def sample(self, batch_size: int) -> Tuple[Batch, Batch]:
        """
        Samples a train Batch and val Batch from the replay buffer.
        Args:
            batch_size: number of sequences to sample
        Returns:
            train_batch: batch of sequences for training
            val_batch: batch of sequences for validation
        """

        if len(self.train_storage) == 0:
            raise ValueError("Replay buffer is empty.")

        train_batch_indices = np.random.randint(
            low=0, high=len(self.train_storage["inputs"]), size=batch_size
        )
        val_batch_indices = np.random.randint(
            low=0, high=len(self.val_storage["inputs"]), size=batch_size
        )

        train_batch = Batch(
            inputs=self.train_storage["inputs"][train_batch_indices],
            targets=self.train_storage["targets"][train_batch_indices],
            observation_masks=self.train_storage["observation_masks"][
                train_batch_indices
            ],
            action_masks=self.train_storage["action_masks"][train_batch_indices],
            reward_masks=self.train_storage["reward_masks"][train_batch_indices],
            target_action_masks=self.train_storage["target_action_masks"][
                train_batch_indices
            ],
        )

        val_batch = Batch(
            inputs=self.val_storage["inputs"][val_batch_indices],
            targets=self.val_storage["targets"][val_batch_indices],
            observation_masks=self.val_storage["observation_masks"][val_batch_indices],
            action_masks=self.val_storage["action_masks"][val_batch_indices],
            reward_masks=self.val_storage["reward_masks"][val_batch_indices],
            target_action_masks=self.val_storage["target_action_masks"][
                val_batch_indices
            ],
        )

        return train_batch, val_batch
