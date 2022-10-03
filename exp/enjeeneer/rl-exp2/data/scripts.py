import os
import gym
import pickle
import bauwerk
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import TYPE_CHECKING, Optional, Tuple, Union, Dict

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

    def tasks_dict(self, data: pd.DataFrame):
        """
        Takes DataFrame of obs, action, obs_, reward, for many tasks and creates associated dictionary of reshpaed arrays.
        Each primary key in the dictionary represents a task.
        """
        # TODO: inherit keys and tasks from cfg
        task_col = 'battery_size'
        data_dict = {}
        task_dict = {}

        for i, task in enumerate(data[task_col].unique()):
            task_data = data[data[task_col] == task]
            task_dict['cfg'] = {'battery_size': task_data['battery_size'].iloc[0]}

            obs_arr = task_data['obs'].to_numpy()
            obs_dim = task_data['obs'].iloc[0].shape[0]
            task_dict['obs'] = np.concatenate(obs_arr).reshape(len(obs_arr), obs_dim)

            obs_arr_ = task_data['obs_'].to_numpy()
            obs_dim_ = task_data['obs_'].iloc[0].shape[0]
            task_dict['obs_'] = np.concatenate(obs_arr_).reshape(len(obs_arr_), obs_dim_)

            act_arr = task_data['action'].to_numpy()
            act_dim = task_data['action'].iloc[0].shape[0]
            task_dict['action'] = np.concatenate(act_arr).reshape(len(act_arr), act_dim)

            rew_arr = task_data['reward'].to_numpy()
            rew_dim = task_data['reward'].iloc[0].shape[0]
            task_dict['reward'] = np.concatenate(rew_arr).reshape(len(rew_arr), rew_dim)

            print(task_data.head())
            done_arr = task_data['done'].to_numpy()
            task_dict['done'] = done_arr

            data_dict[str(i)] = task_dict

        return data_dict

    def trajectorize(self, data: Dict):
        """
        Takes dictionary of data across many tasks, and creates epsiode-length trajectories of state-action-(opt: reward) pairs/triplets.
        """
        # get indexes of end of episodes
        term_idx = np.where(data['0']['done'] == True)
        term_idx = np.insert(term_idx, 0, 0)

        obs_trajs = []
        act_trajs = []
        rew_trajs = []

        for i in range(len(term_idx) - 1):
            obs_traj = data['0']['obs_'][term_idx[i]: term_idx[i + 1], :]
            act_traj = data['0']['action'][term_idx[i]: term_idx[i + 1], :]
            reward_traj = data['0']['reward'][term_idx[i]: term_idx[i + 1], :]
            obs_trajs.append(obs_traj)
            act_trajs.append(act_traj)
            rew_trajs.append(reward_traj)

        traj_lengths = [int(len(traj)) for traj in obs_trajs]
        num_trajs = len(traj_lengths)
        max_traj = int(max(traj_lengths))

        # need to pad trajs as they may be of different depending on episode
        padded_obs_trajs = np.zeros([num_trajs, max_traj, obs_trajs[0].shape[1]], dtype=np.float32)
        padded_act_trajs = np.zeros([num_trajs, max_traj, act_trajs[0].shape[1]], dtype=np.float32)
        padded_rew_trajs = np.zeros([num_trajs, max_traj, rew_trajs[0].shape[1]], dtype=np.float32)
        early_term_trajs = np.zeros([num_trajs, max_traj, 1], dtype=np.bool)

        i = 0
        for obs, act, rew in zip(obs_trajs, act_trajs, rew_trajs):
            padded_obs_trajs[i, :traj_lengths[i], :] = obs
            padded_act_trajs[i, :traj_lengths[i], :] = act
            padded_rew_trajs[i, :traj_lengths[i], :] = rew
            early_term_trajs[i, traj_lengths[i]:, :] = True
            i += 1

        return padded_obs_trajs, padded_act_trajs, padded_rew_trajs, early_term_trajs

            

DC = DataCollector(cfg)
DC.run()