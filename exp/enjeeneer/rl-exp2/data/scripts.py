import os
import gym
import pickle
import bauwerk
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import TYPE_CHECKING, Optional, Tuple, Union

import sys
sys.path.append('../agent')
from sac.agent import Agent

sys.path.append('../utils')
from utils import ObsWrapper, Cfg
cfg = Cfg.parse()

class DataCollector:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg.eval_freq = 24 * 7
        self.cfg.collection_episodes = 1
        self.cfg.threshold = 0.8
        self.eval_str = 'mean_eval_reward'
        self.build_dist_b = bauwerk.benchmarks.BuildDistB()
        self.cfg.save_dir = os.getcwd()
        
    def evaluate(self, agent, task) -> Tuple[float, pd.DataFrame]:
        """
        Evaluates the mean stepwise performance of one rollout from our agent, and
        returns the mean evaluation reward and transition data.
        """
        print('...Performing Evaluation Rollout...')
        rollout = pd.DataFrame()
        eval_env = self.build_dist_b.make_env()
        eval_env.set_task(task)
        eval_env = ObsWrapper(eval_env)
        obs = eval_env.reset()
        rewards = 0
        done = False
        steps = 0

        while not done:
            action, _ = agent.act(obs, evaluate=True)
            obs_, reward, done, _ = eval_env.step(action)
            rewards += reward
            steps += 1
            
            # store data
            transition = {
            'obs': obs,
            'action': action,
            'obs_': obs_,
            'reward': np.array([reward], np.float32),
            'done': done,
            'battery_size': eval_env.cfg.battery_size
            }
            transition = pd.DataFrame([transition])
            rollout = pd.concat([rollout, transition], ignore_index=True)

            obs = obs_
            
        mean_reward = rewards / steps
        rollout[self.eval_str] = mean_reward
        
        return mean_reward, rollout
    
    def collect(self) -> pd.DataFrame:
        """
        Collects a dataset of obs, obs_, rewards, dones, eval_rewards from tasks drawn from some distribution. 
        """
        
        data = pd.DataFrame()
        build_dist_b = bauwerk.benchmarks.BuildDistB()
        tasks = [build_dist_b.train_tasks[0]]

        for j, task in enumerate(tasks):
            print('## Collecting Data for Bauwerk Task: {} ##'.format(j))

            # build env
            env = build_dist_b.make_env()
            env.set_task(task)
            env = ObsWrapper(env)

            # build worker
            agent = Agent(cfg=self.cfg, env=env, models_dir='/tmp') 

            for i in tqdm(range(self.cfg.collection_episodes)):
                print('## Episode: {} ##'.format(i))
                eval_reward = -1
                done = False
                obs = env.reset()
                while not done:

                    action, inp = agent.act(obs, evaluate=False)
                    obs_, reward, done, _ = env.step(action)
                    agent.n_steps += 1

                    # modify obs_ to include history
                    if (self.cfg.hist_length > 0) & (agent.n_steps > self.cfg.hist_length):
                        history = agent.memory.get_history()
                        state_ = np.concatenate((obs_, history), axis=0)
                        agent.memory.store_transition(inp, state_, action, reward, done)

                    else:
                        agent.memory.store_transition(inp, obs_, action, reward, done)

                    # update agent
                    if agent.n_steps > self.cfg.learning_starts:
                        value_loss, actor_loss, critic_loss = agent.learn()

                    # evaluate and rollout
                    if agent.n_steps % self.cfg.eval_freq == 0:
                        eval_reward, rollout = self.evaluate(agent, task)
                        data = pd.concat([data, rollout], ignore_index=True)

                    obs = obs_                                          

        return data
    
    def clean(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Takes dataset of obs, action, obs_, reward, done and cleans such that we only retain data from agent 
        at >= 80% of converged performance.
        """
        
        max_return = max(dataset[self.eval_str].unique())
        threshold_return = max_return - np.absolute(max_return * (1 - self.threshold))

        cleaned = dataset[dataset[self.eval_str] >= threshold_return]

        return cleaned
    
    def run(self, dir) -> None:
        """
        Performs training of models, rollouts, evals and saves associated transitions.
        """
        if dir is None:
            dir = self.cfg.save_dir
            
        data = self.collect()
        data = self.clean(data)
        
        with open(os.path.join(dir, 'dataset.pickle'), 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            

DC = DataCollector(cfg)
DC.run()