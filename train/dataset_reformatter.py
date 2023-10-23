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
parser.add_argument("--dirs", nargs="+")
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--samples_per_building", type=int, default=1000)
parser.add_argument("--context_length", type=int, default=100)
parser.add_argument("--maintain_rewards", type=bool, default=False)
args = parser.parse_args()

test_data = {
    "observation": [np.array([1, 2, 3]) for _ in range(args.samples_per_building + 1)],
    "action": [np.array([4, 5, 6]) for _ in range(args.samples_per_building + 1)],
    "next_observation": [
        np.array([7, 8, 9]) for _ in range(args.samples_per_building + 1)
    ],
    "reward": [np.array([69]) for _ in range(args.samples_per_building + 1)],
    "done": [
        np.array([0]) if (i < args.samples_per_building) else np.array([1])
        for i in range(args.samples_per_building + 1)
    ],
    "building_id": ["0001" for _ in range(args.samples_per_building + 1)],
    "episode": [1 for _ in range(args.samples_per_building + 1)],
    "mean_reward": [np.array([0.5]) for _ in range(args.samples_per_building + 1)],
}

test_df = pd.DataFrame.from_dict(test_data)
test_df.to_pickle(f"{BASE_DIR}/train/data/train.pickle")


class DatasetReformatter:
    """
    Takes raw data from performative training and creates
    sequences of trajectories for transformer training.
    """

    def __init__(
        self,
        dirs: list,
        samples_per_building: int,
        context_length: int,
        dataset_name: str,
        maintain_rewards: bool = False,
    ):
        self.dirs = dirs
        self.samples_per_building = samples_per_building
        self.context_length = context_length
        self.maintain_rewards = maintain_rewards
        self.dataset_name = dataset_name

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
        for dir_index in tqdm(range(len(self.dirs)), desc="Sequencing buildings."):
            # load raw dataframe
            file = Path(BASE_DIR, "train", self.dirs[dir_index], "rollouts.pickle")
            df = pd.read_pickle(file)

            # create dictionary of episodes for building
            episodes = self._create_building_episodes(df)

            # create episode-wise trajectories for building
            (
                episode_trajs,
                episode_obs_masks,
                episode_act_masks,
                episode_rew_masks,
            ) = self._create_episode_trajectories(episodes)

            # create sequences for building
            (
                building_input_sequences,
                building_target_sequences,
                building_obs_masks,
                building_act_masks,
                building_target_act_masks,
                building_reward_masks,
            ) = self._create_sequences(
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

        logger.info(f"Saving data to {BASE_DIR}/train/processed/dataset.npz")
        makedirs(Path(BASE_DIR, "train", "processed"), exist_ok=True)
        np.savez_compressed(
            f"{BASE_DIR}/train/processed/{self.dataset_name}/dataset.npz",
            **aggregated_data,
        )
        print("here")

    @staticmethod
    def _create_building_episodes(df: pd.DataFrame) -> Dict:
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

            for variable in ["observation", "action", "next_observation", "reward"]:
                array = episode_data[variable]
                dimension = episode_data[variable].iloc[0].shape[0]
                episode[variable] = np.concatenate(array).reshape(len(array), dimension)

            episode["done"] = (
                episode_data["done"].to_numpy().reshape(len(episode_data["done"]), 1)
            )

            # store episode data indexed by episode no.
            episodes[j] = episode

        return episodes

    @staticmethod
    def _create_episode_trajectories(
        episodes: Dict,
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
        episode_actions = []
        episode_rewards = []

        for _, episode_dict in episodes.items():

            # get indexes of end of episodes
            term_idx = np.where(episode_dict["done"] == 1)[0]
            term_idx = np.insert(term_idx, 0, 0)

            for i in range(len(term_idx) - 1):
                obs_traj = episode_dict["observation"][term_idx[i] : term_idx[i + 1], :]
                act_traj = episode_dict["action"][term_idx[i] : term_idx[i + 1], :]
                reward_traj = episode_dict["reward"][term_idx[i] : term_idx[i + 1], :]
                episode_observations.append(obs_traj)
                episode_actions.append(act_traj)
                episode_rewards.append(reward_traj)

        episode_lengths = [int(len(ep)) for ep in episode_observations]
        number_of_episodes = len(episode_lengths)
        max_episode_length = int(max(episode_lengths))

        # get dimension info from first entry in lists
        observation_dim = episode_observations[0].shape[1]
        action_dim = episode_actions[0].shape[1]
        reward_dim = episode_rewards[0].shape[1]

        # need to pad trajs as they may be different length depending on episode
        padded_obs_trajs = np.zeros(
            [number_of_episodes, max_episode_length, observation_dim], dtype=np.float32
        )
        padded_act_trajs = np.zeros(
            [number_of_episodes, max_episode_length, action_dim], dtype=np.float32
        )
        padded_rew_trajs = np.zeros(
            [number_of_episodes, max_episode_length, reward_dim], dtype=np.float32
        )

        # build padded trajectories
        for i, (obs, act, rew) in enumerate(
            zip(episode_observations, episode_actions, episode_rewards)
        ):
            padded_obs_trajs[
                i, : episode_lengths[i], :
            ] = obs  # [ep, timestep, obs_dim]
            padded_act_trajs[i, : episode_lengths[i], :] = act
            padded_rew_trajs[i, : episode_lengths[i], :] = rew

        # concat (produces array of shape
        # [no_episodes, max_ep_length, obs_dim + act_dim + rew_dim]
        padded_trajs = np.concatenate(
            [padded_obs_trajs, padded_act_trajs, padded_rew_trajs], axis=-1
        )

        # masks
        obs_mask = np.zeros(shape=padded_trajs.shape)
        act_mask = np.zeros(shape=padded_trajs.shape)
        rew_mask = np.zeros(shape=padded_trajs.shape)
        obs_mask[:, :, :observation_dim] = np.arange(
            start=1, stop=observation_dim + 1
        )  # obs pos used for positional embedding later
        act_mask[:, :, observation_dim : observation_dim + action_dim] = 1
        rew_mask[:, :, -1] = 1

        # reshape into episodes of shape [ep, timesteps * (obs_dim + act_dim + rew_dim)
        padded_trajs = padded_trajs.reshape(
            number_of_episodes,
            max_episode_length * (observation_dim + action_dim + reward_dim),
        )
        obs_mask = obs_mask.reshape(
            number_of_episodes,
            max_episode_length * (observation_dim + action_dim + reward_dim),
        )
        act_mask = act_mask.reshape(
            number_of_episodes,
            max_episode_length * (observation_dim + action_dim + reward_dim),
        )
        rew_mask = rew_mask.reshape(
            number_of_episodes,
            max_episode_length * (observation_dim + action_dim + reward_dim),
        )

        return padded_trajs, obs_mask, act_mask, rew_mask

    def _create_sequences(
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
    dirs=args.dirs,
    dataset_name=args.dataset_name,
    samples_per_building=args.samples_per_building,
    context_length=args.context_length,
    maintain_rewards=args.maintain_rewards,
)

reformatter()
