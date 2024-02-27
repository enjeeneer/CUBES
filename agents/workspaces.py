# pylint: disable=[invalid-name, unused-argument]
"""Module that creates workspaces for training/evaling various agents."""

import gym
import pandas as pd
import torch
import pickle

import wandb
from os import makedirs
from loguru import logger
from tqdm import tqdm
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union, List
from datetime import datetime

from agents.sac.agent import SoftActorCritic
from agents.sac.replay_buffer import SoftActorCriticReplayBuffer

from agents.dt.agent import DecisionTransformer
from agents.dt.replay_buffer import DecisionTransformerReplayBuffer

from agents.pearl.agent import PEARL
from agents.pearl.replay_buffer import PEARLReplayBuffer

from agents.base import AbstractWorkspace

from cubes.rbcs.rbc import GeneralRBC


def transform_sac_battery_action(battery_action: np.ndarray) -> np.ndarray:
    """
    Takes 1D battery action and transforms it into 2D battery action, where
    the first dimension is the charge action and the second dimension is the
    discharge action.
    """

    # TODO: check that charging action is the first one
    # charging
    if battery_action >= 0:
        renormalised_battery_action = (2 * (battery_action - (0))) / (1 - (0)) - 1
        return np.array([renormalised_battery_action, -1])  # -1 unnormalises to 0
    else:
        renormalised_battery_action = (2 * (-battery_action - (0))) / (1 - (0)) - 1
        return np.array([-1, renormalised_battery_action])


class LeidenWorkspace(AbstractWorkspace):
    """
    Workspace for leiden experiments
    """

    def __init__(
        self,
        env,
        eval_rollouts: int,
        wandb_logging: bool,
        wandb_entity: str,
        wandb_project: str,
        wandb_tags: List[str],
    ):
        super().__init__(
            env=env,
            eval_rollouts=eval_rollouts,
            wandb_logging=wandb_logging,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_tags=wandb_tags,
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def eval(
        self,
        agent: Union[SoftActorCritic, GeneralRBC, PEARL],
        replay_buffer: Union[SoftActorCriticReplayBuffer, PEARLReplayBuffer],
        agent_config: Dict = None,
        checkpoints: bool = True,
        full_logging: bool = False,
    ) -> Dict[str, float]:
        """
        Performs eval rollouts and logs metrics for RBC and SAC.
        Args:
            agent: tuple of SAC and RBC agents
            replay_buffer: replay buffer for SAC agent
            agent_config: config for evaled agent
            checkpoints: True if eval is being called during training; False
                        if eval is being called for inference.
            full_logging: True if logging all metrics; False if logging only
        Returns:
            metrics: dict of metrics
        """
        if not checkpoints and self.wandb_logging:
            if full_logging:
                self.wandb_tags = self.wandb_tags + ["eval_rollout"]

            run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                tags=self.wandb_tags,
                config=agent_config,
                reinit=True,
            )

        logger.info("Performing eval.")
        eval_rewards = []
        eval_emissions = []
        eval_ndt_t_violations = {}
        eval_ndt_aq_violations = {}
        eval_heating_dt = {}
        eval_heating_service_dt = {}
        eval_max_heating_service_dt = {}
        eval_heating_beyond_comf_dt = {}
        eval_violation_dt = {}
        eval_violation_daq = {}
        eval_emissions_reward = []
        eval_comfort_reward = []
        eval_aq_reward = []

        if isinstance(agent, SoftActorCritic):
            agent.eval()

        for _ in tqdm(range(self.eval_rollouts)):
            done = False
            rollout_reward = []
            rollout_emissions = 0.0
            rollout_ndt_t_violations = {}
            rollout_ndt_aq_violations = {}
            rollout_heating_dt = {}
            rollout_heating_service_dt = {}
            rollout_max_heating_service_dt = {}
            rollout_violation_daq = {}
            rollout_heating_beyond_comf_dt = {}
            rollout_violation_dt = {}
            rollout_emissions_reward = []
            rollout_comfort_reward = []
            rollout_aq_reward = []

            obs = self.env.reset()
            while not done:
                if isinstance(agent, SoftActorCritic):
                    action = agent.act(
                        obs,
                        sample=False,
                        replay_buffer=replay_buffer,
                    )
                    if not self.battery_demand_levelling:

                        battery_action = transform_sac_battery_action(action[-1])
                        action = np.append(action[:-1], battery_action)

                        if self.battery_only:
                            action = np.append(self.normalised_temp_setpoints, action)
                    else:
                        if self.battery_only:
                            action = np.append(self.normalised_temp_setpoints, action)

                elif isinstance(agent, PEARL):
                    action = agent.act(obs, explore=False)
                else:
                    action = agent.act(obs)

                obs, reward, done, info = self.env.step(action)

                rollout_reward.append(reward)
                rollout_emissions += info["emissions"]

                if full_logging and self.wandb_logging:
                    # get obs dict and action dict
                    obs_dict = self.env.obs_dict
                    action_dict = dict(
                        zip(self.env.variables["action"], info["action_"])
                    )

                    metrics = {**obs_dict, **action_dict}

                    run.log(metrics)

                if not rollout_ndt_t_violations:
                    for k, v in info["t_violation"].items():
                        rollout_ndt_t_violations[k] = v
                else:
                    for k, v in info["t_violation"].items():
                        rollout_ndt_t_violations[k] += v

                if not rollout_ndt_aq_violations:
                    for k, v in info["aq_violation"].items():
                        rollout_ndt_aq_violations[k] = v
                else:
                    for k, v in info["aq_violation"].items():
                        rollout_ndt_aq_violations[k] += v

                if not rollout_heating_dt:
                    for k, v in info["heating_delta_T"].items():
                        rollout_heating_dt[k] = v / 144
                else:
                    for k, v in info["heating_delta_T"].items():
                        rollout_heating_dt[k] += v / 144

                if not rollout_heating_service_dt:
                    for k, v in info["heating_service"].items():
                        rollout_heating_service_dt[k] = v / 144
                else:
                    for k, v in info["heating_service"].items():
                        rollout_heating_service_dt[k] += v / 144

                if not rollout_max_heating_service_dt:
                    for k, v in info["max_heating_service"].items():
                        rollout_max_heating_service_dt[k] = v / 144
                else:
                    for k, v in info["max_heating_service"].items():
                        rollout_max_heating_service_dt[k] += v / 144

                if not rollout_heating_beyond_comf_dt:
                    for k, v in info["heating_beyond_comf_delta_T"].items():
                        rollout_heating_beyond_comf_dt[k] = v / 144
                else:
                    for k, v in info["heating_beyond_comf_delta_T"].items():
                        rollout_heating_beyond_comf_dt[k] += v / 144

                if not rollout_violation_dt:
                    for k, v in info["violation_delta_T"].items():
                        rollout_violation_dt[k] = v / 144
                else:
                    for k, v in info["violation_delta_T"].items():
                        rollout_violation_dt[k] += v / 144

                if not rollout_violation_daq:
                    for k, v in info["violation_delta_aq"].items():
                        rollout_violation_daq[k] = v / 144
                else:
                    for k, v in info["violation_delta_aq"].items():
                        rollout_violation_daq[k] += v / 144

                rollout_emissions_reward.append(info["reward_emissions"])
                rollout_comfort_reward.append(info["reward_comfort"])
                rollout_aq_reward.append(info["reward_air_quality"])

            eval_rewards.append(np.mean(rollout_reward))
            eval_emissions_reward.append(np.mean(rollout_emissions_reward))
            eval_comfort_reward.append(np.mean(rollout_comfort_reward))
            eval_aq_reward.append(np.mean(rollout_aq_reward))
            eval_emissions.append(np.mean(rollout_emissions))

            if not eval_ndt_t_violations:
                for k, v in rollout_ndt_t_violations.items():
                    eval_ndt_t_violations[k] = [v]
            else:
                for k, v in rollout_ndt_t_violations.items():
                    eval_ndt_t_violations[k].append(v)

            if not eval_ndt_aq_violations:
                for k, v in rollout_ndt_aq_violations.items():
                    eval_ndt_aq_violations[k] = [v]
            else:
                for k, v in rollout_ndt_aq_violations.items():
                    eval_ndt_aq_violations[k].append(v)

            if not eval_heating_dt:
                for k, v in rollout_heating_dt.items():
                    eval_heating_dt[k] = [v]
            else:
                for k, v in rollout_heating_dt.items():
                    eval_heating_dt[k].append(v)

            if not eval_heating_service_dt:
                for k, v in rollout_heating_service_dt.items():
                    eval_heating_service_dt[k] = [v]
            else:
                for k, v in rollout_heating_service_dt.items():
                    eval_heating_service_dt[k].append(v)

            if not eval_max_heating_service_dt:
                for k, v in rollout_max_heating_service_dt.items():
                    eval_max_heating_service_dt[k] = [v]
            else:
                for k, v in rollout_heating_service_dt.items():
                    eval_heating_service_dt[k].append(v)

            if not eval_heating_beyond_comf_dt:
                for k, v in rollout_heating_beyond_comf_dt.items():
                    eval_heating_beyond_comf_dt[k] = [v]
            else:
                for k, v in rollout_heating_beyond_comf_dt.items():
                    eval_heating_beyond_comf_dt[k].append(v)

            if not eval_violation_dt:
                for k, v in rollout_violation_dt.items():
                    eval_violation_dt[k] = [v]
            else:
                for k, v in rollout_violation_dt.items():
                    eval_violation_dt[k].append(v)

            if not eval_violation_daq:
                for k, v in rollout_violation_daq.items():
                    eval_violation_daq[k] = [v]
            else:
                for k, v in rollout_violation_daq.items():
                    eval_violation_daq[k].append(v)

        self.env.reset()
        eval_t_violations_means = {}
        for k, v in eval_ndt_t_violations.items():
            eval_t_violations_means[k] = float(np.mean(v))

        eval_aq_violations_means = {}
        for k, v in eval_ndt_aq_violations.items():
            eval_aq_violations_means[k] = float(np.mean(v))

        eval_heating_dt_means = {}
        for k, v in eval_heating_dt.items():
            eval_heating_dt_means[k] = float(np.mean(v))

        eval_heating_service_dt_means = {}
        for k, v in eval_heating_service_dt.items():
            eval_heating_service_dt_means[k] = float(np.mean(v))

        eval_max_heating_service_dt_means = {}
        for k, v in eval_max_heating_service_dt.items():
            eval_max_heating_service_dt_means[k] = float(np.mean(v))

        eval_heating_beyond_comf_dt_means = {}
        for k, v in eval_heating_beyond_comf_dt.items():
            eval_heating_beyond_comf_dt_means[k] = float(np.mean(v))

        eval_violation_dt_means = {}
        for k, v in eval_violation_dt.items():
            eval_violation_dt_means[k] = float(np.mean(v))

        eval_violation_daq_means = {}
        for k, v in eval_violation_daq.items():
            eval_violation_daq_means[k] = float(np.mean(v))

        metrics = {
            "eval/mean_episode_reward": float(np.mean(eval_rewards)),
            "eval/mean_episode_emissions_reward": float(np.mean(eval_emissions_reward)),
            "eval/mean_episode_comfort_reward": float(np.mean(eval_comfort_reward)),
            "eval/mean_episode_air_quality_reward": float(np.mean(eval_aq_reward)),
            "eval/mean_episode_emissions": float(np.mean(eval_emissions)),
            "eval/mean_episode_ndt_t_violations": eval_t_violations_means,
            "eval/mean_episode_ndt_aq_violations": eval_aq_violations_means,
            "eval/mean_episode_heating_degree_days": eval_heating_dt_means,
            "eval/mean_episode_heating_service_degree_days": (
                eval_heating_service_dt_means
            ),
            "eval/mean_episode_max_heating_service_degree_days": (
                eval_max_heating_service_dt_means
            ),
            "eval/mean_episode_heating_beyond_comfort_degree_days": (
                eval_heating_beyond_comf_dt_means
            ),
            "eval/mean_episode_violation_degree_days": eval_violation_dt_means,
            "eval/mean_episode_violation_ppm_days": eval_violation_daq_means,
        }

        if not checkpoints and self.wandb_logging:
            run.log(metrics)
            run.finish()

        return metrics


class LeidenSACWorkspace(LeidenWorkspace):
    """
    Trains/evals/train SAC on one task
    """

    def __init__(
        self,
        env,
        learning_steps: int,
        model_dir: Path,
        eval_frequency: int,
        eval_rollouts: int,
        seed_steps: int,
        wandb_logging: bool,
        log_frequency: int,
        wandb_entity: str,
        wandb_project: str,
        wandb_tags: List[str],
        action_length: int,
        battery_only: bool,
        battery_demand_levelling: bool,
        thermostat_setpoint: float,
        action_variable_names: List[str],
        normalized_observations: bool
    ):
        super().__init__(
            env=env,
            eval_rollouts=eval_rollouts,
            wandb_logging=wandb_logging,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_tags=wandb_tags,
        )

        self.eval_frequency = eval_frequency  # how frequently to eval
        self.model_dir = model_dir
        self.learning_steps = learning_steps
        self.seed_steps = seed_steps
        self.log_frequency = log_frequency
        self.battery_only = battery_only
        self.battery_demand_levelling = battery_demand_levelling
        self.action_length = action_length
        self.normalized_observations = normalized_observations
        action_range_dict = dict(
            zip(
                action_variable_names,
                zip(self.env.setpoints_space.low, self.env.setpoints_space.high),
            )
        )
        self.action_ranges = [*action_range_dict.values()]

        if self.battery_only:
            real_temp_setpoints = [thermostat_setpoint for _ in range(2)]
            normalised_temp_setpoints = []
            for i, temp in enumerate(real_temp_setpoints):
                normalised_temp_setpoints.append(
                    2
                    * (temp - self.action_ranges[i][0])
                    / (self.action_ranges[i][1] - self.action_ranges[i][0])
                    - 1
                )
            self.normalised_temp_setpoints = np.array(normalised_temp_setpoints)

    def train(
        self,
        agent: SoftActorCritic,
        agent_config: Dict,
        replay_buffer: SoftActorCriticReplayBuffer,
    ):
        """
        Trains SAC on one task.
        """
        torch.set_num_threads(1)

        if self.wandb_logging:
            run = wandb.init(
                entity=self.wandb_entity,
                project=self.wandb_project,
                config=agent_config,
                tags=self.wandb_tags,
                reinit=True,
            )

            model_path = self.model_dir / run.name

        else:
            model_path = self.model_dir / "local"

        makedirs(str(model_path), exist_ok=True)

        logger.info("Training SAC.")
        best_eval_reward = -1e8
        done = True

        for i in tqdm(range(self.learning_steps)):

            # reset env
            if done:
                obs = self.env.reset()
            else:
                obs = next_obs

            # sample actions uniformly for seed steps
            if i < self.seed_steps:
                action = np.random.uniform(low=-1, high=1, size=(self.action_length,))

            else:
                action = agent.act(
                    obs,
                    sample=True,
                    replay_buffer=replay_buffer,
                )

            if not self.battery_demand_levelling:
                battery_action = transform_sac_battery_action(action[-1])
                env_action = np.append(action[:-1], battery_action)
                if self.battery_only:
                    env_action = np.append(self.normalised_temp_setpoints, env_action)
            else:
                if self.battery_only:
                    env_action = np.append(self.normalised_temp_setpoints, action)
                else:
                    env_action = action

            next_obs, reward, done, _ = self.env.step(env_action)

            replay_buffer.add(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                done=done,
            )

            eval_metrics = {}
            if (i % self.eval_frequency == 0) & (i > 0):
                eval_metrics = self.eval(agent=agent, replay_buffer=replay_buffer)
                if eval_metrics["eval/mean_episode_reward"] > best_eval_reward:
                    logger.info(
                        f"New max eval reward: {best_eval_reward:.3f} -> "
                        f"{eval_metrics['eval/mean_episode_reward']:.3f}."
                        f" Saving model."
                    )

                    agent.name = i
                    # save locally
                    path = agent.save(model_path)
                    # save observation normalization
                    if self.normalized_observations:
                        on_save_path = (str(path).split(".",maxsplit=1)[0]
                                        +"_obs_norm.pickle")
                        obs_rms = {"mean":self.env.obs_rms.mean,
                                   "var":self.env.obs_rms.var}
                        with open(on_save_path, mode="wb") as f:
                            pickle.dump(obs_rms, f)
                    # save to wandb
                    if self.wandb_logging:
                        run.save(path.as_posix(), base_path=model_path.as_posix())

                    best_eval_reward = eval_metrics["eval/mean_episode_reward"]

                agent.train()

            train_metrics = {}
            if (i % agent.actor_update_frequency == 0) and (i > self.seed_steps):
                train_metrics = agent.update(replay_buffer=replay_buffer, step=i)

            metrics = {**train_metrics, **eval_metrics}

            if self.wandb_logging:
                if i % self.log_frequency == 0:
                    run.log(metrics)

        if self.wandb_logging:
            run.finish()


class LeidenPEARLWorkspace(LeidenWorkspace):
    """
    Trains/evals/train PEARL on one task
    """

    def __init__(
        self,
        env,
        training_steps: int,
        model_dir: Path,
        eval_frequency: int,
        update_frequency: int,
        wandb_logging: bool,
        log_frequency: int,
        wandb_entity: str,
        wandb_project: str,
        wandb_tags: List[str],
        seed_steps: int,
        eval_rollouts: int = 1,
    ):
        super().__init__(
            env=env,
            eval_rollouts=eval_rollouts,
            wandb_logging=wandb_logging,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_tags=wandb_tags,
        )

        self.update_frequency = update_frequency
        self.eval_frequency = eval_frequency  # how frequently to eval
        self.model_dir = model_dir
        self.training_steps = training_steps
        self.log_frequency = log_frequency
        self.seed_steps = seed_steps

    def train(
        self,
        agent: PEARL,
        agent_config: Dict,
        replay_buffer: PEARLReplayBuffer,
    ):
        """
        Trains PEARL on one task.
        """
        torch.set_num_threads(1)

        if self.wandb_logging:
            run = wandb.init(
                entity=self.wandb_entity,
                project=self.wandb_project,
                config=agent_config,
                tags=self.wandb_tags,
                reinit=True,
            )

            model_path = self.model_dir / run.name

        else:
            model_path = self.model_dir / "local"

        makedirs(str(model_path), exist_ok=True)

        logger.info("Training PEARL.")
        best_eval_reward = -1e8
        done = True

        for i in tqdm(range(self.training_steps)):

            # reset env
            if done:
                obs = self.env.reset()
            else:
                obs = next_obs

            # sample actions uniformly for seed steps
            if i < self.seed_steps:
                action = np.random.uniform(
                    low=-1, high=1, size=(self.env.action_space.shape[0],)
                )
            else:
                action = agent.act(obs, explore=False)

            next_obs, _, done, _ = self.env.step(action)

            replay_buffer.add(
                observation=obs,
                action=action,
                next_observation=next_obs,
            )

            # update models periodically, and after sufficient data has been collected
            train_metrics = {}
            # update models at end of seed steps
            if i == (self.seed_steps - 1):
                train_metrics = agent.update(replay_buffer=replay_buffer)

            if (i % self.update_frequency == 0) and (i > (self.seed_steps)):
                train_metrics = agent.update(replay_buffer=replay_buffer)

            eval_metrics = {}
            if (i % self.eval_frequency == 0) and (i > (self.seed_steps)):
                eval_metrics = self.eval(agent=agent, replay_buffer=replay_buffer)

                if eval_metrics["eval/mean_episode_reward"] > best_eval_reward:
                    logger.info(
                        f"New max eval reward: {best_eval_reward:.3f} -> "
                        f"{eval_metrics['eval/mean_episode_reward']:.3f}."
                        f" Saving model."
                    )

                    agent.name = i
                    # save locally
                    path = agent.save(model_path)
                    # save to wandb
                    if self.wandb_logging:
                        run.save(path.as_posix(), base_path=model_path.as_posix())

                    best_eval_reward = eval_metrics["eval/mean_episode_reward"]

                agent.train()

            metrics = {**train_metrics, **eval_metrics}

            if self.wandb_logging:
                if i % self.log_frequency == 0:
                    run.log(metrics)

        if self.wandb_logging:
            run.finish()


class DataCollectionWorkspace:
    """
    Trains/evals/train SAC on one task.
    """

    def __init__(
        self,
        env,
        learning_steps: int,
        run_dir: Path,
        eval_frequency: int,
        eval_rollouts: int,
        seed_steps: int,
        wandb_logging: bool,
        building_config: Dict,
        number_logged_rollouts: int,
        building_id: str,
    ):
        self.env = env
        self.eval_frequency = eval_frequency  # how frequently to eval
        self.eval_rollouts = eval_rollouts  # how many train per eval step
        self.run_dir = run_dir
        self.learning_steps = learning_steps
        self.seed_steps = seed_steps
        self.wandb_logging = wandb_logging
        self.building_config = building_config
        self.eval_metric = "mean_reward"
        self.number_logged_rollouts = number_logged_rollouts
        self.building_id = building_id

        self._STEPS_PER_DAY = 144

    def train(
        self,
        agent: SoftActorCritic,
        agent_config: Dict,
        replay_buffer: SoftActorCriticReplayBuffer,
    ):
        """
        Trains SAC on one task.
        """
        torch.set_num_threads(1)

        dataset = pd.DataFrame()

        if self.wandb_logging:
            run = wandb.init(
                entity="enjeeneer",
                project="cubes",
                config=agent_config,
                tags=["data-collection"],
                reinit=True,
            )

        dataset_path = self.run_dir / "rollouts.parquet"

        logger.info("Training SAC for data collection.")
        best_eval_reward = -1e8
        done = True
        self.eval_episode_no = 0

        for i in tqdm(range(self.learning_steps)):

            # reset env
            if done:
                obs = self.env.reset()
            else:
                obs = next_obs

            # sample actions uniformly for seed steps
            if i < self.seed_steps:
                action = np.random.uniform(
                    low=-1, high=1, size=(self.env.action_space.shape[0],)
                )

            else:
                action = agent.act(
                    obs,
                    sample=True,
                    replay_buffer=replay_buffer,
                )
            next_obs, reward, done, _ = self.env.step(action)

            replay_buffer.add(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                done=done,
            )

            eval_metrics = {}
            if (i % self.eval_frequency == 0) & (i > 0):
                eval_metrics, rollout = self.eval(
                    agent=agent, replay_buffer=replay_buffer
                )
                self.eval_episode_no += 1

                # store data
                dataset = pd.concat([dataset, rollout], ignore_index=True)

                if eval_metrics["eval/mean_episode_reward"] > best_eval_reward:
                    logger.info(
                        f"New max eval reward: {best_eval_reward:.3f} -> "
                        f"{eval_metrics['eval/mean_episode_reward']:.3f}."
                        f" Saving model."
                    )

                    best_eval_reward = eval_metrics["eval/mean_episode_reward"]

                done = True
                agent.train()

            train_metrics = {}

            if (i % agent.actor_update_frequency == 0) and (i > self.seed_steps):
                train_metrics = agent.update(replay_buffer=replay_buffer, step=i)

            metrics = {**train_metrics, **eval_metrics}

            if self.wandb_logging:
                run.log(metrics)

        # slice to only maintain performative dataset and save
        logger.info(
            f"Slicing dataset to {self.number_logged_rollouts} rollouts and saving."
        )
        dataset = self.get_performative(dataset)
        dataset.to_parquet(dataset_path)

        if self.wandb_logging:
            run.finish()

    def eval(
        self, agent: SoftActorCritic, replay_buffer: SoftActorCriticReplayBuffer
    ) -> Tuple[Dict, pd.DataFrame]:
        """Performs eval train."""
        logger.info("Collecting eval rollout.")
        rollout = pd.DataFrame()
        agent.eval()

        done = False
        rollout_reward = []
        rollout_violation_dt = {}
        rollout_emissions = 0.0

        obs = self.env.reset()

        while not done:
            action = agent.act(
                obs,
                sample=False,
                replay_buffer=replay_buffer,
            )
            obs_, reward, done, info = self.env.step(action)
            rollout_reward.append(reward)
            rollout_emissions += info["emissions"]

            if not rollout_violation_dt:
                for k, v in info["violation_delta_T"].items():
                    rollout_violation_dt[k] = v / self._STEPS_PER_DAY
            else:
                for k, v in info["violation_delta_T"].items():
                    rollout_violation_dt[k] += v / self._STEPS_PER_DAY

            # store data
            transition = {
                "building_id": self.building_id,
                "observation": obs,
                "action": action,
                "reward": np.array([reward]),
                "done": done,
                "episode": self.eval_episode_no,
            }
            transition = pd.DataFrame([transition])
            rollout = pd.concat([rollout, transition], ignore_index=True)

            obs = obs_

        mean_reward = np.mean(rollout_reward)
        for k, v in rollout_violation_dt.items():
            rollout_violation_dt[k] = float(np.mean(v))

        rollout[self.eval_metric] = mean_reward

        metrics = {
            "eval/mean_episode_reward": mean_reward,
            "eval/mean_episode_violation_degree_days": rollout_violation_dt,
            "eval/mean_episode_emissions": rollout_emissions,
        }

        return metrics, rollout

    def get_performative(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Takes dataset of tasks and cleans to retain data only data from agents
        performing above some evaluation
        threshold i.e. >= threshold% of converged performance.
        Args:
            dataset: DataFrame of transition data with columns
                        [config, obs, action, obs_, reward,
                        done, mean_reward, episode]
        Returns:
            sliced_data: Sliced DataFrame of high performing data.
              The variables/columns are ['config', 'obs', 'action',
              'obs_', 'reward', 'done', 'mean_reward']
        """

        sliced_dataset = pd.DataFrame(columns=dataset.columns)

        # get the top N rollouts by mean reward for dataset
        runs = np.sort(dataset[self.eval_metric].unique())[::-1][
            : self.number_logged_rollouts
        ]
        for run in runs:
            performative_data = dataset[dataset[self.eval_metric] == run]
            sliced_dataset = pd.concat([sliced_dataset, performative_data])

        return sliced_dataset


class DecisionTransformerWorkspace:
    """Trains and evaluates Decision Transformer on task(s)."""

    def __init__(
        self,
        learning_steps: int,
        eval_frequency: int,
        eval_rollouts: int,
        wandb_logging: bool,
        device: torch.device,
        model_dir: Path,
        eval_env: gym.Env,
        observation_dim: int,
        action_dim: int,
        context_length: int,
        agent_config: Dict,
        steps_per_day: int = 144,
        separator_tokens: bool = True,
    ):

        self.learning_steps = learning_steps
        self.eval_frequency = eval_frequency
        self.eval_rollouts = eval_rollouts
        self.wandb_logging = wandb_logging
        self.device = device
        self.model_dir = model_dir
        self.eval_env = eval_env
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.context_length = context_length
        self.agent_config = agent_config
        self._STEPS_PER_DAY = steps_per_day
        self.separator_tokens = separator_tokens

    def train(
        self,
        agent: DecisionTransformer,
        replay_buffer: DecisionTransformerReplayBuffer,
    ) -> None:
        """
        Trains Decision Transformer on replay buffer.
        """
        if self.wandb_logging:
            run = wandb.init(
                entity="enjeeneer",
                project="cubes-DT",
                config=self.agent_config,
                reinit=True,
            )
            model_path = self.model_dir / run.name
            makedirs(str(model_path))
        else:
            date = datetime.today().strftime("Y-%m-%d-%H-%M-%S")
            model_path = self.model_dir / f"local-run-{date}"
            makedirs(str(model_path))

        logger.info("Training Decision Transformer.")
        best_eval_reward = -np.inf
        best_model_path = None

        for i in tqdm(range(self.learning_steps + 1)):

            batch = replay_buffer.sample(agent.batch_size)
            train_metrics = agent.update(batch=batch)

            eval_metrics = {}
            if (i % self.eval_frequency == 0) and (i > 0):
                eval_metrics = self.eval(agent=agent)
                if eval_metrics["eval/mean_episode_reward"] > best_eval_reward:
                    logger.info(
                        f"New max eval reward: {best_eval_reward:.3f} -> "
                        f"{eval_metrics['eval/mean_episode_reward']:.3f}."
                        f" Saving model."
                    )

                    # delete current best model
                    if best_model_path is not None:
                        best_model_path.unlink(missing_ok=True)

                    agent.name = f"dt_{i}"
                    best_eval_reward = eval_metrics["eval/mean_episode_reward"]
                    best_model_path = agent.save(model_path)

                agent.train()

            metrics = {**train_metrics, **eval_metrics}

            if self.wandb_logging:
                run.log(metrics)

        if self.wandb_logging:
            # save model to wandb
            run.save(best_model_path.as_posix(), base_path=model_path.as_posix())
            run.finish()

        # delete local model
        shutil.rmtree(model_path)

    def eval(
        self,
        agent: DecisionTransformer,
    ) -> Dict[str, Union[float, Dict]]:
        """
        Performs eval train.
        Args:
            agent: Decision Transformer agent.
        Returns:
            eval_metrics: Dictionary of eval metrics.
        """
        logger.info("Performing eval rollouts.")
        eval_rewards = []
        eval_violation_dt = {}
        agent.eval()

        for _ in tqdm(range(self.eval_rollouts)):

            done = False
            rollout_reward = []
            rollout_violation_dt = {}
            input_sequence, obs_mask, act_mask, _ = self._get_prompt()

            while not done:

                action = agent.act(
                    input_sequence=input_sequence,
                    action_dimension=self.action_dim,
                    observation_mask=obs_mask,
                    action_mask=act_mask,
                )

                obs, reward, done, info = self.eval_env.step(action)
                rollout_reward.append(reward)

                if not rollout_violation_dt:
                    for k, v in info["violation_delta_T"].items():
                        rollout_violation_dt[k] = v / self._STEPS_PER_DAY
                else:
                    for k, v in info["violation_delta_T"].items():
                        rollout_violation_dt[k] += v / self._STEPS_PER_DAY

                # add new observation to input sequence
                input_sequence, obs_mask, act_mask = agent.update_sequences(
                    sequence=input_sequence,
                    obs_mask=obs_mask,
                    act_mask=act_mask,
                    values_to_add=obs,
                    obs=True,
                )

                # add new action to input sequence
                input_sequence, obs_mask, act_mask = agent.update_sequences(
                    sequence=input_sequence,
                    obs_mask=obs_mask,
                    act_mask=act_mask,
                    values_to_add=action,
                    action=True,
                )

            eval_rewards.append(np.mean(rollout_reward))
            for k, v in rollout_violation_dt.items():
                eval_violation_dt[k] = float(np.mean(v))

        # average over eval rollouts
        for k, v in eval_violation_dt.items():
            eval_violation_dt[k] = float(np.mean(v))
        eval_rewards = np.mean(eval_rewards)

        # aggregate metrics
        metrics = {
            "eval/mean_episode_reward": np.mean(eval_rewards),
            "eval/mean_episode_violation_degree_days": eval_violation_dt,
        }

        return metrics

    def _get_prompt(self):
        """
        Creates prompt to initialise DT with.
        Returns:
            prompt: Prompt to initialise DT with.
            obs_mask: Observation mask.
            act_mask: Action mask.
            rew_mask: Reward mask.
        """

        prompt_steps = int(
            np.ceil(
                self.context_length
                / (
                    self.observation_dim
                    + int(self.separator_tokens)
                    + self.action_dim
                    + int(self.agent_config["predict_reward"])
                )
            )
        )

        # create masks
        obs_mask = np.zeros(
            shape=(
                int(prompt_steps + 1),
                self.observation_dim
                + int(self.separator_tokens)
                + self.action_dim
                + int(self.agent_config["predict_reward"]),
            )
        )  # +1 because we include final additional obs
        act_mask = np.zeros(
            shape=(
                int(prompt_steps),
                self.observation_dim
                + int(self.separator_tokens)
                + self.action_dim
                + int(self.agent_config["predict_reward"]),
            )
        )
        rew_mask = np.zeros(
            shape=(
                int(prompt_steps),
                self.observation_dim
                + int(self.separator_tokens)
                + self.action_dim
                + int(self.agent_config["predict_reward"]),
            )
        )
        obs_mask[:, : self.observation_dim] = np.arange(
            start=1, stop=self.observation_dim + 1
        )
        act_mask[
            :,
            self.observation_dim
            + int(self.separator_tokens) : self.observation_dim
            + int(self.separator_tokens)
            + self.action_dim,
        ] = 1
        rew_mask[:, -1] = 1
        obs_mask = obs_mask.flatten()[-self.context_length :]
        act_mask = act_mask.flatten()[-self.context_length :]
        rew_mask = rew_mask.flatten()[-self.context_length :]

        prompt_data = []
        obs = self.eval_env.reset()
        for _ in range(prompt_steps):
            prompt_data.append(obs)
            action = self.eval_env.action_space.sample()  # TODO: consider using RBC
            obs, reward, _, _ = self.eval_env.step(action)
            prompt_data.append(action)
            if self.agent_config["predict_reward"]:
                prompt_data.append(reward)

        prompt_data.append(obs)

        # correct masks for last obs
        obs_mask[: -self.observation_dim] = obs_mask[self.observation_dim :]
        obs_mask[-self.observation_dim :] = np.arange(
            start=1, stop=self.observation_dim + 1
        )
        act_mask[: -self.observation_dim] = act_mask[self.observation_dim :]
        act_mask[-self.observation_dim :] = 0
        if self.agent_config["predict_reward"]:
            rew_mask[: -self.observation_dim] = rew_mask[self.observation_dim :]
            rew_mask[-self.observation_dim :] = 0

        prompt = np.concatenate(prompt_data)[-self.context_length :]

        # add batch dimension
        prompt = np.expand_dims(prompt, axis=0)
        obs_mask = np.expand_dims(obs_mask, axis=0)
        act_mask = np.expand_dims(act_mask, axis=0)
        rew_mask = np.expand_dims(rew_mask, axis=0)

        return prompt, obs_mask, act_mask, rew_mask
