# pylint: disable=C0103
"""Script for reformatting SAC rollouts into sequences for DT training."""

import pandas as pd
import numpy as np
from argparse import ArgumentParser
from cubes.constants import BASE_DIR
from pathlib import Path
from typing import Dict, Tuple
from loguru import logger
from tqdm import tqdm
from os import makedirs

parser = ArgumentParser()
parser.add_argument("--dataset_parent_dir", type=str)
parser.add_argument("--dataset_name", type=str, default="alpha_project")
parser.add_argument("--samples_per_building", type=int, default=1000)
parser.add_argument("--context_length", type=int, default=100)
parser.add_argument("--maintain_rewards", type=bool, default=False)
parser.add_argument("--separator_token", type=bool, default=True)
args = parser.parse_args()

parent_dir = Path(BASE_DIR, "train", args.dataset_parent_dir)
dataset_list = [
    Path(parent_dir / d.name / "rollouts.pickle")
    for d in parent_dir.iterdir()
    if d.is_dir()
]


class DatasetReformatter:
    """
    Takes raw data from performative training and creates
    sequences of trajectories for transformer training.
    """

    def __init__(
        self,
        file_list: list,
        samples_per_building: int,
        context_length: int,
        dataset_name: str,
        maintain_rewards: bool = False,
        separator_token: bool = False,
    ):
        self.file_list = file_list
        self.samples_per_building = samples_per_building
        self.context_length = context_length
        self.maintain_rewards = maintain_rewards
        self.dataset_name = dataset_name
        self.separator_token = separator_token

    def __call__(self):
        """
        Runs the reformatting process.
        """

        logger.info(
            f"Reformatting building-wise data with {self.samples_per_building}"
            f" samples per building and context length {self.context_length}."
        )

        inputs = []
        targets = []
        observation_masks = []
        action_masks = []
        target_action_masks = []
        rewards = []

        # create dictionary of data for all buildings in dataset
        for file in tqdm(self.file_list, desc="Sequencing buildings."):

            # load raw dataframe
            try:
                df = pd.read_pickle(file)
            except FileNotFoundError:
                logger.warning(f"File {file} not found.")
                continue

            # create dictionary of episodes for building
            episodes = self._compile_building_episodes(df)

            # create episode-wise trajectories for building of
            # shape [episodes, timesteps * (obs_dim + act_dim + rew_dim)]
            (
                episode_trajs,
                episode_obs_masks,
                episode_act_masks,
                episode_rew_masks,
            ) = self._compile_episode_trajectories(
                episodes, separator_token=self.separator_token
            )

            # create sequences for building of
            # shape [self.samples_per_building, self.context_length]
            (
                building_input_sequences,
                building_target_sequences,
                building_obs_masks,
                building_act_masks,
                building_target_act_masks,
                building_reward_masks,
            ) = self._compile_sequences(
                episode_trajs=episode_trajs,
                episode_obs_masks=episode_obs_masks,
                episode_act_masks=episode_act_masks,
                episode_rew_masks=episode_rew_masks,
            )

            # add to combined, cross-building dataset
            inputs.append(building_input_sequences)
            targets.append(building_target_sequences)
            observation_masks.append(building_obs_masks)
            action_masks.append(building_act_masks)
            target_action_masks.append(building_target_act_masks)
            rewards.append(building_reward_masks)

        aggregated_data = {
            "inputs": np.concatenate(inputs, axis=0),
            "targets": np.concatenate(targets, axis=0),
            "observation_masks": np.concatenate(observation_masks, axis=0),
            "action_masks": np.concatenate(action_masks, axis=0),
            "target_action_masks": np.concatenate(target_action_masks, axis=0),
        }

        if self.maintain_rewards:
            aggregated_data["reward_masks"] = np.concatenate(rewards, axis=0)

        logger.info(
            f"Saving data to {BASE_DIR}/train/processed_datasets"
            f"/{self.dataset_name}/dataset.npz"
        )
        makedirs(
            Path(BASE_DIR, f"train/processed_datasets/{self.dataset_name}"),
            exist_ok=True,
        )
        np.savez_compressed(
            f"{BASE_DIR}/train/processed_datasets/{self.dataset_name}/dataset.npz",
            **aggregated_data,
        )

    @staticmethod
    def _compile_building_episodes(df: pd.DataFrame) -> Dict:
        """
        Takes DataFrame of performative data for many tasks and creates
        associated dictionary of flattened arrays. The primary key in the
        final dict is the building id, and the secondary key is the episode.
        Args:
            df: DataFrame of high performing data with columns
            [obs, action, obs_, reward, done, episode_no, cfg, mean_reward]
        Returns:
            building_dict: Dictionary of containing all episode data for one building.
        """
        episodes = {}

        # create dictionary entry for each task
        for j, episode in enumerate(df["episode"].unique()):
            episode_data = df[df["episode"] == episode]
            episode = {}

            for variable in ["observation", "action", "reward"]:
                print(df)
                print(variable)
                array = episode_data[variable]
                dimension = array.iloc[0].shape[0]
                episode[variable] = np.concatenate(array).reshape(len(array), dimension)

            episode["done"] = (
                episode_data["done"].to_numpy().reshape(len(episode_data["done"]), 1)
            )

            # store episode data indexed by episode no.
            episodes[j] = episode

        return episodes

    @staticmethod
    def _compile_episode_trajectories(
        episodes: Dict,
        separator_token: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Takes dictionary of data from one building, and creates
        episode-length trajectories of flattened obs, act, rew.
        This function is needlessly longwinded in our case as all
        episodes will be of the same length (1 year). However,
        it allows extensability for episodes of different lengths
        should that be required in the future.
        Args:
            episodes: dictionary of building episodic data
            separator_token: whether to separate actions with
            separator token that in practice takes a large negative value
            such that it could not be seen in the data.
        Returns:
            padded_trajs: array of episode trajectories of shape
            (episodes, timesteps * (obs_dim + act_dim + rew_dim))
            obs_mask: array of obs masks giving dim position of shape
            (episodes, timesteps * (obs_dim + act_dim + rew_dim))
            act_mask: array of action masks of shape
            (episodes, timesteps * (obs_dim + act_dim + rew_dim))
            rew_mask: array of reward masks of shape
            (episodes, timesteps * (obs_dim + act_dim + rew_dim))
        """

        # loop over episodes
        episode_observations = []  # elements of list will be episode-length obs arrays
        episode_separators = (
            []
        )  # elements of list will be episode-length seperator arrays
        episode_actions = []
        episode_rewards = []

        for _, episode_dict in episodes.items():

            episode_observations.append(
                np.expand_dims(episode_dict["observation"], axis=0)
            )
            episode_separators.append(
                np.expand_dims(np.ones_like(episode_dict["observation"]) * -1e9, axis=0)
            )
            episode_actions.append(np.expand_dims(episode_dict["action"], axis=0))
            episode_rewards.append(np.expand_dims(episode_dict["reward"], axis=0))

        observation_dim = episode_observations[0].shape[-1]
        action_dim = episode_actions[0].shape[-1]
        reward_dim = 1
        number_of_episodes = len(episode_observations)
        episode_length = episode_observations[0].shape[1]

        # concat (produces array of shape
        # [no_episodes, max_ep_length, obs_dim + act_dim + rew_dim]
        if separator_token:
            trajectories = np.concatenate(
                [
                    np.concatenate(episode_observations, axis=0),
                    np.concatenate(episode_separators, axis=0),
                    np.concatenate(episode_actions, axis=0),
                    np.concatenate(episode_rewards, axis=0),
                ],
                axis=-1,
            )
        else:
            trajectories = np.concatenate(
                [
                    np.concatenate(episode_observations, axis=0),
                    np.concatenate(episode_actions, axis=0),
                    np.concatenate(episode_rewards, axis=0),
                ],
                axis=-1,
            )

        # masks
        obs_mask = np.zeros(shape=trajectories.shape)
        act_mask = np.zeros(shape=trajectories.shape)
        rew_mask = np.zeros(shape=trajectories.shape)

        obs_mask[:, :, :observation_dim] = np.arange(
            start=1, stop=observation_dim + 1
        )  # obs pos used for positional embedding later
        act_mask[
            :,
            :,
            observation_dim
            + int(separator_token) : observation_dim
            + int(separator_token)
            + action_dim,
        ] = 1
        rew_mask[:, :, -1] = 1

        # reshape into episodes of shape [ep, timesteps * (obs_dim + act_dim + rew_dim)
        trajectories = trajectories.reshape(
            number_of_episodes,
            episode_length
            * (observation_dim + int(separator_token) + action_dim + reward_dim),
        )
        obs_mask = obs_mask.reshape(
            number_of_episodes,
            episode_length
            * (observation_dim + int(separator_token) + action_dim + reward_dim),
        )
        act_mask = act_mask.reshape(
            number_of_episodes,
            episode_length
            * (observation_dim + int(separator_token) + action_dim + reward_dim),
        )
        rew_mask = rew_mask.reshape(
            number_of_episodes,
            episode_length
            * (observation_dim + int(separator_token) + action_dim + reward_dim),
        )

        return trajectories, obs_mask, act_mask, rew_mask

    def _compile_sequences(
        self,
        episode_trajs: np.ndarray,
        episode_obs_masks: np.ndarray,
        episode_act_masks: np.ndarray,
        episode_rew_masks: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Takes episode-length task trajectories and creates sequences
        of trajectories of length self.context_length. We create
        both input and target trajectories for transformer training.
        Args:
            episode_trajs: traj array, shape
                        [*, timesteps * (obs_dim, act_dim, rew_dim)]
            episode_obs_masks: array of obs masks, shape
                        [*, timesteps * (obs_dim, act_dim, rew_dim)]
            episode_act_masks: array of action masks, shape
                        [*, timesteps * (obs_dim, act_dim, rew_dim)]
            episode_rew_masks: array of reward masks, shape
                        [*, timesteps * (obs_dim, act_dim, rew_dim)]
        Returns:
            input_sequences: array, shape [N, self.context_length]
            with N = number of trajs we wish to sample per building
            target_sequences: array of input sequences shifted one
             index to make target, shape [N, self.context_length]
            observation_masks: array of obs masks of shape
                                            [N, self.context_length]
            action_masks: array of input action indices of shape
                                            [N, self.context_length]
            target_action_masks: array of target action indices of
                                        shape [N, self.context_length]
            reward_masks: array of action indices of shape
                                                [N, self.context_length]
        """
        # setup sequence array
        input_sequences = np.empty(
            shape=(self.samples_per_building, self.context_length), dtype=np.int64
        )
        target_sequences = np.empty(
            shape=(self.samples_per_building, self.context_length), dtype=np.int64
        )
        observation_masks = np.empty(
            shape=(self.samples_per_building, self.context_length), dtype=np.int64
        )
        action_masks = np.empty(
            shape=(self.samples_per_building, self.context_length), dtype=np.int64
        )
        target_action_masks = np.empty(
            shape=(self.samples_per_building, self.context_length), dtype=np.int64
        )
        reward_masks = np.empty(
            shape=(self.samples_per_building, self.context_length), dtype=np.int64
        )

        # drop rewards if not required
        if not self.maintain_rewards:
            episode_trajs = episode_trajs[~episode_rew_masks.astype(bool)].reshape(
                (episode_trajs.shape[0], -1)
            )  # except rewards
            episode_obs_masks = episode_obs_masks[
                ~episode_rew_masks.astype(bool)
            ].reshape(
                (episode_trajs.shape[0], -1)
            )  # except rewards
            episode_act_masks = episode_act_masks[
                ~episode_rew_masks.astype(bool)
            ].reshape(
                (episode_trajs.shape[0], -1)
            )  # except rewards

        # sample sequences
        number_of_episodes = episode_trajs.shape[0]
        episode_length = episode_trajs.shape[1]

        # get index of random sub-trajectories
        episode_idxs = np.random.randint(
            low=0, high=number_of_episodes, size=self.samples_per_building
        )
        sequence_start_idxs = np.random.randint(
            low=1,
            high=episode_length - self.context_length,
            size=self.samples_per_building,
        )
        sequence_end_idxs = [
            np.arange(start=i, stop=i + self.context_length)
            for i in sequence_start_idxs
        ]

        # build sequences
        for i, (ep_idx, end_idx) in enumerate(zip(episode_idxs, sequence_end_idxs)):
            input_sequences[i, :] = episode_trajs[
                ep_idx, (end_idx - 1)
            ]  # input shifted one to the left
            target_sequences[i, :] = episode_trajs[ep_idx, end_idx]
            observation_masks[i, :] = episode_obs_masks[ep_idx, (end_idx - 1)]
            action_masks[i, :] = episode_act_masks[ep_idx, (end_idx - 1)]
            target_action_masks[i, :] = episode_act_masks[ep_idx, end_idx]

            if self.maintain_rewards:
                reward_masks[i, :] = episode_rew_masks[ep_idx, (end_idx - 1)]

        return (
            input_sequences,
            target_sequences,
            observation_masks,
            action_masks,
            target_action_masks,
            reward_masks,
        )


reformatter = DatasetReformatter(
    file_list=dataset_list,
    dataset_name=args.dataset_name,
    samples_per_building=args.samples_per_building,
    context_length=args.context_length,
    maintain_rewards=args.maintain_rewards,
)

reformatter()
