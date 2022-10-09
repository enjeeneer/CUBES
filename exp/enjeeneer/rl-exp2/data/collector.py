import os
import gym
import pickle
import bauwerk
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Tuple, Dict
from tokenizer import Tokenizer

import sys

sys.path.append('../agent')
from sac.agent import Agent

sys.path.append('../utils')
from utils import ObsWrapper


class DataCollector:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.dataset
        self.eval_str = 'mean_eval_reward'
        self.build_dist_b = bauwerk.benchmarks.BuildDistB()
        self.cfg.save_dir = os.getcwd()
        self.tokenizer = Tokenizer(cfg=cfg.tokenizer)

    def run(self, dir: Optional[str] = None) -> None:
        """
        Performs training of models, rollouts, evals and saves associated transitions.
        """
        if dir is None:
            dir = self.cfg.save_dir

        # collect data and save
        raw_data = self.collect(optimal=self.cfg.optimal)
        print('...saving raw data...')
        with open(os.path.join(dir, self.cfg.raw_name), 'wb') as f:
            pickle.dump(raw_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        performative_data = self.get_performative(raw_data)
        task_dicts = self.create_task_dicts(performative_data)

        # save task-wise data dict
        print('...saving taskwise dictionary ...')
        with open(os.path.join(dir, self.cfg.dict_name), 'wb') as f:
            pickle.dump(task_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)

        # task-wise sequencing
        sequenced_dataset = {}
        for key, _ in task_dicts.items():
            trajs, obs_mask, act_mask, rew_mask = self.create_task_episodes(task_dicts[key])  #
            input_sequences, target_sequencs, obs_masks, act_masks, rew_masks = self.get_sequenced_task_tokens(trajs,
                                                                                                               obs_mask,
                                                                                                               act_mask,
                                                                                                               rew_mask)
            task_dict = {
                'cfg': task_dict[key]['cfg'],
                'inputs': input_sequences,
                'targets': target_sequencs,
                'obs_masks': obs_masks,
                'act_masks': act_masks,
                'rew_masks': rew_masks
            }

            sequenced_dataset[key] = task_dict

        print('...saving sequenced dataset...')
        with open(os.path.join(dir, self.cfg.seq_name), 'wb') as f:
            pickle.dump(sequenced_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    def evaluate(self,
                 task,
                 task_no,
                 eval_episode_no,
                 agent: Optional[Agent] = None,
                 optimal_actions: Optional[np.array] = None) -> Tuple[float, pd.DataFrame]:
        """
        Takes a task and either an agent or optimal sequence of actions from convex solver.
        Evaluates the mean stepwise performance of one rollout from our agent, and
        returns the mean evaluation reward and transition data.
        :param task: task drawn from env distribution
        :param task_no: (int) used as name of task for later indexing
        :param eval_episode_no: (int) count of evaluation episodes performed on task
        :param agent: [Optional] RL agent used for selecting actions
        :param optimal_actions: [Optional] array of actions provided by convex solver, shape [8759, 1]
        :return mean_reward: (float) mean stepwise reward for evaluation rollout
        :return rollout: DataFrame of rollout data where each cell holds an array of shape [var_dim,].
              The variables/columns are ['obs', 'action', 'obs_', 'reward', 'done', 'episode_no', 'cfg', 'mean_reward']
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
            if optimal_actions:
                action = optimal_actions[0][steps]
            else:
                action, _ = agent.act(obs, evaluate=True)

            obs_, reward, done, _ = eval_env.step(action)
            rewards += reward
            steps += 1

            # store data
            transition = {
                self.cfg.task_id: task_no,
                'obs': obs,
                'action': action,
                'obs_': obs_,
                'reward': np.array([reward], np.float32),
                'done': done,
                'epidode_no': eval_episode_no,
                'cfg': eval_env.cfg
            }
            transition = pd.DataFrame([transition])
            rollout = pd.concat([rollout, transition], ignore_index=True)

            obs = obs_

            mean_reward = rewards / steps
            rollout[self.eval_str] = mean_reward

        return mean_reward, rollout

    def collect(self, optimal: Optional[bool] = False) -> pd.DataFrame:
        """
        Collects a dataset of obs, obs_, rewards, dones, eval_rewards for tasks drawn from some distribution.
        The dataset can either be optimal i.e. obtained by evaluating convex solver, or can be obtained by
        training an RL agent to convergence on the task.
        :param optimal: boolean flag that indicates whether we evaluate using bauwerk's convex solver
        :return data: DataFrame of rollout data where each cell holds an array of shape [var_dim,].
              The variables/columns are ['obs', 'action', 'obs_', 'reward', 'done', 'episode_no', 'cfg', 'mean_reward']
        """

        data = pd.DataFrame()
        build_dist_b = bauwerk.benchmarks.BuildDistB()
        tasks = build_dist_b.train_tasks[:2]

        for j, task in enumerate(tasks):
            print('## Collecting Data for Bauwerk Task: {} ##'.format(j))

            # build env
            env = build_dist_b.make_env()
            env.set_task(task)

            if optimal:
                optimal_actions = bauwerk.solve(env)
                eval_reward, rollout = self.evaluate(task, task_no=j, eval_episode_no=0,
                                                     optimal_actions=optimal_actions)
                data = pd.concat([data, rollout], ignore_index=True)

            else:
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
                            eval_reward, rollout = self.evaluate(task, task_no=j, eval_episode_no=i, agent=agent)
                            data = pd.concat([data, rollout], ignore_index=True)

                        obs = obs_

        return data

    def get_performative(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Takes dataset of tasks and cleans to retain data only data from agents performing above some evaluation
        threshold i.e. >= threshold% of converged performance.
        :param dataset: DataFrame of transition data with columns
                                                        [obs, action, obs_, reward, done, episode_no, cfg, mean_reward]
        :return performative_data: DataFrame of high performing data where each cell holds an array of shape [var_dim,].
              The variables/columns are ['obs', 'action', 'obs_', 'reward', 'done', 'episode_no', 'cfg', 'mean_reward']
        """
        performative_data = pd.DataFrame(columns=dataset.columns)
        for task_id in dataset[self.cfg.task_id].unique():
            sliced = dataset[dataset[self.cfg.task_id] == task_id]
            max_return = max(sliced[self.eval_str].unique())
            threshold_return = max_return - np.absolute(max_return * (1 - self.cfg.threshold))
            task_data = sliced[sliced[self.eval_str] >= threshold_return]
            performative_data = pd.concat([performative_data, task_data])

        return performative_data

    def create_task_dicts(self, performative_data: pd.DataFrame):
        """
        Takes DataFrame of performative data for many tasks and creates associated dictionary of reshaped arrays.
        Each primary key in the dictionary represents a task.
        :param performative_data: DataFrame of high performing data with columns
                                                        [obs, action, obs_, reward, done, episode_no, cfg, mean_reward]
        :return data_dict: Dictionary of performative data, with array reshaped to [episodes, timesteps, var_dim]
                            where var_dim is the dimension of the variable in the key-value pair.
        """
        # TODO: change task indexing from numbers to something recognizable
        data_dict = {}

        # create dictionary entry for each task
        for i, task in enumerate(performative_data[self.cfg.task_id].unique()):
            task_dict = {}
            task_data = performative_data[performative_data[self.cfg.task_id] == task]
            task_dict['cfg'] = task_data['cfg'].iloc[0]

            # create dictionary for each episode in task (we do this as each episode may vary in length)
            episode_dict = {}
            for j, episode in enumerate(task_data['episode_no'].unique()):
                episode_data = task_data[task_data['episode_no'] == episode]
                for var in ['obs', 'action', 'obs_', 'reward', 'done']:
                    arr = episode_data[var].to_numpy()
                    dim = episode_data[var].iloc[0].shape[0]
                    episode_dict[var] = np.concatenate(arr).reshape(len(arr), dim)

                # store episode data in task dict, indexed by episode no.
                task_dict[j] = episode_dict

            # store dict of task episodes in data dictionary, indexed by task
            data_dict[str(i)] = task_dict

        return data_dict

    def create_task_episodes(self, task_dict: Dict) -> [np.array, np.array, np.array]:
        """
        Takes dictionary of data from one task, and creates episode-length trajectories of flattened obs, act, rew.
        This function is needlessly longwinded in our case as all episodes will be of the same length (1 year). However
        it allows extensability for episodes of different lengths should that be required in the future.
        :param task_dicts: dictionary of task-specific episodic data
        :return padded_trajs: array of episode trajectories of shape (episodes, timesteps * (obs_dim + act_dim + rew_dim))
        :return obs_mask: array of obs masks giving dim position of shape (episodes, timesteps * (obs_dim + act_dim + rew_dim))
        :return act_mask: array of action masks of shape (episodes, timesteps * (obs_dim + act_dim + rew_dim))
        :return rew_mask: array of reward masks of shape (episodes, timesteps * (obs_dim + act_dim + rew_dim))
        """
        # loop over episodes
        ep_obs = []  # elements of list will be episode-length obs arrays
        ep_act = []
        ep_rew = []
        for episode, _ in task_dict.items():
            episode_dict = task_dict[episode]

            # get indexes of end of episodes
            term_idx = np.where(episode_dict['done'] == True)
            term_idx = np.insert(term_idx, 0, 0)

            for i in range(len(term_idx) - 1):
                obs_traj = episode_dict['obs_'][term_idx[i]: term_idx[i + 1], :]
                act_traj = episode_dict['action'][term_idx[i]: term_idx[i + 1], :]
                reward_traj = episode_dict['reward'][term_idx[i]: term_idx[i + 1], :]
                ep_obs.append(obs_traj)
                ep_act.append(act_traj)
                ep_rew.append(reward_traj)

        ep_lengths = [int(len(ep)) for ep in ep_obs]
        num_eps = len(ep_lengths)
        max_ep_length = int(max(ep_lengths))
        obs_dim = ep_obs[0].shape[1]
        act_dim = ep_act[0].shape[1]
        rew_dim = ep_rew[0].shape[1]

        # need to pad trajs as they may be different length depending on episode
        padded_obs_trajs = np.zeros([num_eps, max_ep_length, obs_dim], dtype=np.float32)
        padded_act_trajs = np.zeros([num_eps, max_ep_length, act_dim], dtype=np.float32)
        padded_rew_trajs = np.zeros([num_eps, max_ep_length, rew_dim], dtype=np.float32)

        for i, (obs, act, rew) in enumerate(zip(ep_obs, ep_act, ep_rew)):
            padded_obs_trajs[i, :ep_lengths[i], :] = obs  # [ep, timestep, obs_dim]
            padded_act_trajs[i, :ep_lengths[i], :] = act
            padded_rew_trajs[i, :ep_lengths[i], :] = rew

        # concat (produces array of shape [no_episodes, max_ep_length, obs_dim + act_dim + rew_dim]
        padded_trajs = np.concatenate([padded_obs_trajs, padded_act_trajs, padded_rew_trajs], axis=-1)

        # masks
        obs_mask = np.zeros(shape=padded_trajs.shape)
        act_mask = np.zeros(shape=padded_trajs.shape)
        rew_mask = np.zeros(shape=padded_trajs.shape)
        obs_mask[:, :, :obs_dim] = np.arange(start=1, stop=obs_dim + 1)  # obs pos used for positional embedding later
        act_mask[:, :, obs_dim: obs_dim + act_dim] = 1
        rew_mask[:, :, -1] = 1

        # reshape into episodes of shape [ep, timesteps * (obs_dim + act_dim + rew_dim)
        padded_trajs = padded_trajs.reshape(num_eps, max_ep_length * (obs_dim + act_dim + rew_dim))
        obs_mask = obs_mask.reshape(num_eps, max_ep_length * (obs_dim + act_dim + rew_dim))
        act_mask = act_mask.reshape(num_eps, max_ep_length * (obs_dim + act_dim + rew_dim))
        rew_mask = rew_mask.reshape(num_eps, max_ep_length * (obs_dim + act_dim + rew_dim))

        return padded_trajs, obs_mask, act_mask, rew_mask

    def get_sequenced_task_tokens(self, padded_trajs: np.array,
                                  obs_mask: np.array,
                                  act_mask: np.array,
                                  rew_mask: np.array) -> [np.array, np.array, np.array, np.array, np.array]:
        """
        Takes episode-length task trajectories and creates sequences of tokenized trajectories of length
        context_length. We create both input and target trajectories for transformer training.
        :param padded_trajs: traj array, shape [*, timesteps * (obs_dim, act_dim, rew_dim)]
        :param obs_mask:
        :param act_mask:
        :param rew_mask:
        :return input_sequences: array, shape [N, context_length] with N = number of trajs we wish to sample
        :return target_sequences: array of input sequences shifted one index to make target, shape [N, context_length]
        :return actions: array of action indices of shape [N, context_length]
        :return rewards: array of action indices of shape [N, context_length]
        """
        # setup sequence array
        input_sequences = np.empty(shape=(self.cfg.task_trajectories, self.cfg.context_length))
        target_sequences = np.empty(shape=(self.cfg.task_trajectories, self.cfg.context_length))
        obs = np.empty(shape=(self.cfg.task_trajectories, self.cfg.context_length))
        actions = np.empty(shape=(self.cfg.task_trajectories, self.cfg.context_length))
        rewards = np.empty(shape=(self.cfg.task_trajectories, self.cfg.context_length))

        # tokenize
        token_trajs = self.tokenizer.tokenize(padded_trajs)

        # drop rewards if not required
        if not self.cfg.rewards:
            token_trajs = token_trajs[~rew_mask.astype(bool)]  # all idxs except rewards
            assert token_trajs.shape == (padded_trajs.shape[0], self.cfg.episode_length * 5 * 1)  # bauwerk only check

        # sample sequences
        eps = token_trajs.shape[0]
        tokens = token_trajs.shape[1]

        # get index of random sub-trajectories
        eps_idxs = np.random.randint(low=0, high=eps - 1, size=self.cfg.task_trajectories)
        seq_idxs = np.random.randint(low=1, high=tokens - 1 - self.cfg.context_length, size=self.cfg.task_trajectories)
        context_idxs = [np.arange(start=i, stop=i + self.cfg.context_length) for i in seq_idxs]

        for i, (ep_idx, cont_idx) in enumerate(zip(eps_idxs, context_idxs)):
            input_sequences[i, :] = token_trajs[ep_idx, (cont_idx - 1)]  # input shifted one to the left
            target_sequences[i, :] = token_trajs[ep_idx, cont_idx]
            obs[i, :] = obs_mask[ep_idx, cont_idx]
            actions[i, :] = act_mask[ep_idx, cont_idx]

            if self.cfg.rewards:
                rewards[i, :] = rew_mask[ep_idx, cont_idx]

        return input_sequences, target_sequences, obs, actions, rewards


def batch(dataset: Dict, cfg) -> [np.array, np.array, np.array, np.array]:
    """
    Takes dataset (as dict) of input_sequences, targets, act_masks, and (optionally) reward_masks
    :param dataset: dictionary of task-wise datasets, composed of input_sequences, target_sequences,
                    action_mask sequences and (optionally) reward_mask sequences, all of shape [N, context_length]
    :return input_batches: array of shape [learning_steps, batch_size, context_length]
    :return target_batches: array of shape [learning_steps, batch_size, context_length]
    :return act_mask_batches: array of shape [learning_steps, batch_size, context_length]
    :return rew_mask_batches: array of shape [learning_steps, batch_size, context_length]
    """
    input_batches = np.empty(shape=(cfg.learning_steps, cfg.batch_size, cfg.context_length))
    target_batches = np.empty(shape=(cfg.learning_steps, cfg.batch_size, cfg.context_length))
    obs_mask_batches = np.empty(shape=(cfg.learning_steps, cfg.batch_size, cfg.context_length))
    act_mask_batches = np.empty(shape=(cfg.learning_steps, cfg.batch_size, cfg.context_length))
    rew_mask_batches = np.empty(shape=(cfg.learning_steps, cfg.batch_size, cfg.context_length))
    tasks = [task for task in dataset.keys()]

    # TODO: will need some way of sampling tasks that reflects their proportion a country / continent
    # for now we'll sample uniformly from tasks
    for i in range(cfg.learning_steps):
        task_idxs = np.random.randint(low=0, high=len(tasks), size=cfg.batch_size)
        seq_idxs = np.random.randint(low=0, high=cfg.task_trajectories, size=cfg.batch_size)

        for j, (task_i, seq_i) in enumerate(zip(task_idxs, seq_idxs)):
            task = tasks[task_i]

            input_batches[i, j, :] = dataset[task]['inputs'][seq_i, :]
            target_batches[i, j, :] = dataset[task]['target'][seq_i, :]
            obs_mask_batches[i, j, :] = dataset[task]['obs_masks'][seq_i, :]
            act_mask_batches[i, j, :] = dataset[task]['act_masks'][seq_i, :]

            if cfg.rewards:
                rew_mask_batches[i, j, :] = dataset[task]['rew_masks'][seq_i, :]

    return input_batches, target_batches, obs_mask_batches, act_mask_batches, rew_mask_batches
