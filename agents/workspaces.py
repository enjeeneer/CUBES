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
            tags=["scott", "sac"],
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
                obs = self.env.reset()
                print(obs)

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
            print(f"reward: {reward}")
            print(f"action: {action}")
            print(f"action shape: {action.shape}")
            print(f"next_obs: {next_obs}")

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
        agent.eval()
        for _ in tqdm(range(self.eval_rollouts)):
            done = False
            rollout_reward = 0.0
            obs = self.env.reset()
            while not done:
                action = agent.act(
                    obs,
                    sample=False,
                    replay_buffer=replay_buffer,
                )
                obs, reward, done, _ = self.env.step(action)
                rollout_reward += reward

            eval_rewards.append(rollout_reward)

        metrics = {"eval/mean_episode_reward": float(np.mean(eval_rewards))}

        return metrics
