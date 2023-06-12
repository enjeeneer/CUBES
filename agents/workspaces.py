# pylint: disable=invalid-name

"""Module for workspaces for training/evaling various agents."""

import wandb

from os import makedirs
from loguru import logger
from tqdm import tqdm
import numpy as np
import torch
from pathlib import Path
from typing import Dict

from base import AbstractWorkspace
from sac.agent import SoftActorCritic


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

    def train(self, agent: SoftActorCritic, agent_config: Dict):
        """
        Trains SAC on one task.
        """
        run = wandb.init(
            entity="enjeeneer",
            project="liden",
            config=agent_config,
            tags=[],
            reinit=True,
        )

        model_path = self.model_dir / run.name
        makedirs(str(model_path))

        logger.info("Training SAC.")
        best_eval_reward = 0.0
        done = True

        for i in tqdm(range(self.learning_steps)):

            # reset env
            if done:
                observation = self.env.reset()

            # sample actions uniformly for seed steps
            if i < self.seed_steps:
                action = np.random.uniform(
                    low=-1, high=1, size=(self.env.action_spec().shape[0],)
                )

            else:
                action = agent.act(
                    observation,
                    sample=True,
                )

            next_observation, reward, done, _ = self.env.step(action)

            agent.replay_buffer.add(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )

            observation = next_observation

            eval_metrics = {}
            if (i % self.eval_frequency == 0) & (i > 0):
                eval_metrics = self.eval(agent)
                if eval_metrics["eval/mean_episode_reward"] > best_eval_reward:
                    logger.info(
                        f"New max eval reward: {best_eval_reward:.3f} -> "
                        f"{eval_metrics['eval/mean_episode_reward']:.3f}."
                        f" Saving model."
                    )

                    name = f"{i}.pickle"
                    # save locally
                    path = agent.save(model_path / name)
                    # save to wandb
                    run.save(path.as_posix(), base_path=model_path.as_posix())

                    best_eval_reward = eval_metrics["eval/mean_episode_reward"]

                agent.train()

            train_metrics = {}
            if (i % agent.actor_update_frequency == 0) and (i > agent.batch_size):
                train_metrics = agent.update(i)

            metrics = {**train_metrics, **eval_metrics}

            run.log(metrics)

        run.finish()

    def eval(self, agent: SoftActorCritic) -> Dict[str, float]:
        """Performs eval rollouts."""
        logger.info("Performing eval rollouts.")
        eval_rewards = []
        agent.eval()
        for _ in tqdm(range(self.eval_rollouts)):

            rollout_reward = 0.0
            observation = self.env.reset()
            done = False
            while not done:
                action = agent.act(
                    observation,
                    sample=False,
                )
                next_observation, reward, done, _ = self.env.step(action)
                rollout_reward += reward
                observation = next_observation

            eval_rewards.append(rollout_reward)

        metrics = {"eval/mean_episode_reward": float(np.mean(eval_rewards))}

        return metrics

    def collect_dataset(self, num_samples: int) -> Dict[str, np.ndarray]:
        """
        Collects dataset of observations and actions by loading an actor
        and performing rollouts.
        Args:
            num_samples: number of samples to collect.
        Returns:
            dataset: dataset of observations and actions.
        """
        logger.info(f"Collecting dataset of {num_samples} samples.")
        dataset = {
            "observations": [],
            "actions": [],
            "next_observations": [],
            "rewards": [],
        }
        agent = torch.load(self.model_path)
        agent.eval()

        for _ in tqdm(range(num_samples)):
            timestep = self.env.reset()
            while not timestep.last():
                action = agent.act(
                    timestep.observation["observations"],
                    sample=False,
                )
                dataset["observations"].append(timestep.observation["observations"])
                dataset["actions"].append(action)
                timestep = self.env.step(action)
                reward = self.reward_function(self.env.physics)
                dataset["rewards"].append(reward)
                dataset["next_observations"].append(
                    timestep.observation["observations"]
                )

        dataset["observations"] = np.array(dataset["observations"])
        dataset["actions"] = np.array(dataset["actions"])
        dataset["next_observations"] = np.array(dataset["next_observations"])
        dataset["rewards"] = np.array(dataset["rewards"])

        return dataset
