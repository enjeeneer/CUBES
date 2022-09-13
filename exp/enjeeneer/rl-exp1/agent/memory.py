import os
import pickle
import numpy as np

class SACMemory:
    def __init__(self, batch_size, hist_length, state_dim, act_dim):
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.mem_size = int(1000000)
        self.mem_ctr = 0
        self.obs = np.zeros((self.mem_size, state_dim * (hist_length + 1)))
        self.obs_ = np.zeros((self.mem_size, state_dim * (hist_length + 1)))
        self.actions = np.zeros((self.mem_size, act_dim))
        self.rewards = np.zeros(self.mem_size)
        self.dones = np.zeros(self.mem_size, dtype=bool)
        self.hist = np.zeros(hist_length * state_dim)

    def sample(self):
        '''
        Generates batches of training data for agent policy learning
        :return state array: array of previously seen states of shape (batch_size, obs_dim)
        :return action array: array of previously taken actions of shape (batch_size,act_dim)
        :return probs array: array of previous logs probs of actions given policy distribution of shape (batch_size,)
        :return vals array: array of previous critic values of shape (batch_size,)
        :return rewards array: array of previous rewards of shape (batch_size,)
        '''
        max_mem = min(self.mem_ctr, self.mem_size)
        batch = np.random.randint(low=0, high=max_mem, size=self.batch_size, dtype=np.int32)

        return self.obs[batch], \
               self.obs_[batch], \
               self.actions[batch], \
               self.rewards[batch], \
               self.dones[batch]

    def store_transition(self, obs, obs_, action, reward, done):
        '''
        Stores transitory data in memory
        '''
        index = self.mem_ctr % self.mem_size

        self.obs[index] = obs
        self.obs_[index] = obs_
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done

        self.mem_ctr += 1

    def store_history(self, obs):
        '''
        Stores current state observation in history memory. This history is concatenated with the current
        observation to estimate the state. Doing so is a common way of dealing with partially observed states.
        '''
        self.hist[self.state_dim:] = self.hist[:-self.state_dim]
        self.hist[:self.state_dim] = obs

    def get_history(self):
        '''
        Returns the observations stored in history for concatenation with current observation
        '''

        return self.hist

    def save(self, path):
        print('...saving memories...')
        with open(os.path.join(path, 'obs.pickle'), 'wb') as f:
            pickle.dump(self.obs, f)

        with open(os.path.join(path, 'actions.pickle'), 'wb') as f:
            pickle.dump(self.actions, f)

        with open(os.path.join(path, 'rewards.pickle'), 'wb') as f:
            pickle.dump(self.rewards, f)

    def load(self, path):
        print('...loading memories...')
        with open(os.path.join(path, 'obs.pickle'), 'rb') as f:
            self.obs = pickle.load(f)

        with open(os.path.join(path, 'actions.pickle'), 'rb') as f:
            self.actions = pickle.load(f)

        with open(os.path.join(path, 'rewards.pickle'), 'rb') as f:
            self.rewards = pickle.load(f)