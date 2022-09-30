#!/usr/bin/env python3
import torch
import os
import time
from tqdm import tqdm

from agent.sac import SACAgent
from replay_buffer import ReplayBuffer
from cfgs.parser import parse_cfg
import utils

import dmc2gym


def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=True)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = utils.make_env(cfg)

        cfg.sac.obs_dim = self.env.observation_space.shape[0]
        cfg.sac.action_dim = self.env.action_space.shape[0]
        cfg.sac.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = SACAgent(cfg)
        self.replay_buffer = ReplayBuffer(cfg, self.env.observation_space.shape, self.env.action_space.shape)
        self.step = 0
        self.evaluations = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward

            average_episode_reward += episode_reward
        average_episode_reward /= self.cfg.num_eval_episodes
        self.evaluations += 1
        print('Evaluation: {:d} --- Reward: {:.3f}'.format(self.evaluations, average_episode_reward))

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        for _ in tqdm(range(int(self.cfg.num_train_steps))):
            if done:

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.evaluate()

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    cfg = parse_cfg()
    main(cfg)
