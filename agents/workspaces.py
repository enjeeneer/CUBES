# pylint: disable=invalid-name
"""Module that creates workspaces for training/evaling various agents."""
import torch

import wandb
from os import makedirs
from loguru import logger
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict

from agents.sac.agent import SoftActorCritic
from agents.sac.replay_buffer import SoftActorCriticReplayBuffer
from agents.base import AbstractWorkspace


class SACWorkspace(AbstractWorkspace):
    """
    Trains/evals/rollouts SAC on one task
    """

    def __init__(
        self,
        env,
        learning_steps: int,
        model_dir: Path,
        eval_frequency: int,
        eval_rollouts: int,
        seed_steps: int,
    ):
        super().__init__()

        self.env = env
        self.eval_frequency = eval_frequency  # how frequently to eval
        self.eval_rollouts = eval_rollouts  # how many rollouts per eval step
        self.model_dir = model_dir
        self.learning_steps = learning_steps
        self.seed_steps = seed_steps

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

        run = wandb.init(
            entity="enjeeneer",
            project="liden",
            config=agent_config,
            tags=["hannes", "sac"],
            reinit=True,
        )

        model_path = self.model_dir / run.name
        makedirs(str(model_path))

        logger.info("Training SAC.")
        best_eval_reward = -1e8
        done = True

        for i in tqdm(range(self.learning_steps)):

            # reset env
            if done:
                obs = self.env.reset()
                # print(obs)

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
                eval_metrics = self.eval(agent=agent, replay_buffer=replay_buffer)
                if eval_metrics["eval/mean_episode_reward"] > best_eval_reward:
                    logger.info(
                        f"New max eval reward: {best_eval_reward:.3f} -> "
                        f"{eval_metrics['eval/mean_episode_reward']:.3f}."
                        f" Saving model."
                    )

                    name = f"sac_{i}.pickle"
                    # save locally
                    path = agent.save(model_path / name)
                    # save to wandb
                    run.save(path.as_posix(), base_path=model_path.as_posix())

                    best_eval_reward = eval_metrics["eval/mean_episode_reward"]

                agent.train()

            train_metrics = {}
            if (i % agent.actor_update_frequency == 0) and (i > self.seed_steps):
                train_metrics = agent.update(replay_buffer=replay_buffer, step=i)

            metrics = {**train_metrics, **eval_metrics}

            run.log(metrics)

        run.finish()

    def eval(
        self, agent: SoftActorCritic, replay_buffer: SoftActorCriticReplayBuffer
    ) -> Dict[str, float]:
        """Performs eval rollouts."""
        logger.info("Performing eval rollouts.")
        eval_rewards = []
        eval_emissions = []
        eval_ndt_t_violations = {}
        eval_ndt_aq_violations = {}
        eval_heating_dt = {}
        eval_heating_beyond_comf_dt = {}
        eval_violation_dt = {}
        eval_violation_daq = {}
        eval_emissions_reward = []
        eval_comfort_reward = []
        eval_aq_reward = []

        agent.eval()
        for _ in tqdm(range(self.eval_rollouts)):
            done = False
            rollout_reward = 0.0
            rollout_emissions = 0.0
            rollout_ndt_t_violations = {}
            rollout_ndt_aq_violations = {}
            rollout_heating_dt = {}
            rollout_violation_daq = {}
            rollout_heating_beyond_comf_dt = {}
            rollout_violation_dt = {}
            rollout_emissions_reward = 0.0
            rollout_comfort_reward = 0.0
            rollout_aq_reward = 0.0

            obs = self.env.reset()
            while not done:
                action = agent.act(
                    obs,
                    sample=False,
                    replay_buffer=replay_buffer,
                )
                obs, reward, done, info = self.env.step(action)
                rollout_reward += reward
                rollout_emissions += info["emissions"]
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

                rollout_emissions_reward += info["reward_emissions"]
                rollout_comfort_reward += info["reward_comfort"]
                rollout_aq_reward += info["reward_air_quality"]

            eval_rewards.append(rollout_reward)
            eval_emissions.append(rollout_emissions)

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

            eval_emissions_reward.append(rollout_emissions_reward)
            eval_comfort_reward.append(rollout_comfort_reward)
            eval_aq_reward.append(rollout_aq_reward)

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
            "eval/mean_episode_heating_beyond_comfort_degree_days": (
                eval_heating_beyond_comf_dt_means
            ),
            "eval/mean_episode_violation_degree_days": eval_violation_dt_means,
            "eval/mean_episode_violation_ppm_days": eval_violation_daq_means,
        }

        return metrics
