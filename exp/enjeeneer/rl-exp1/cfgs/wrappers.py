import gym
import numpy as np


def dict_to_array(obs):

    # modify obs
    vals = []
    for _, value in obs.items():
        vals.append(value)
    obs = np.concatenate(vals, axis=0, dtype=np.float32)

    return obs


class ObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ObsWrapper, self).__init__(env)
        self.env = env

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)

        # modify obs
        vals = []
        for _, value in obs_dict.items():
            vals.append(value)
        obs_array = np.concatenate(vals, axis=0, dtype=np.float32)

        return obs_array, reward, done, obs_dict, info


