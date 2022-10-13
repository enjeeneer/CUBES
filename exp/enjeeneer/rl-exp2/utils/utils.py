import gym
import bauwerk
import numpy as np
from typing import Union
from omegaconf import OmegaConf, DictConfig, ListConfig


class ObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ObsWrapper, self).__init__(env)
        self.env = env
        obs_low = []
        obs_high = []
        for key, value in env.observation_space.items():
            obs_low.append(env.observation_space[key].low)
            obs_high.append(env.observation_space[key].high)

        self.obs_low = np.concatenate(obs_low, axis=0)
        self.obs_high = np.concatenate(obs_high, axis=0)

        # hack obs_low and obs_high until Arduin fixes bauwerk
        self.obs_high[1] = 4.5
        self.obs_high[2] = 3.5

    def step(self, action):
        """
        Steps simulator given action and flattens and normalises observation.
        :param action: array of shape [act_dim]
        :return:
        """
        obs_dict, reward, done, info = self.env.step(action)

        # modify obs
        vals = []
        for _, value in obs_dict.items():
            vals.append(value)
        obs_array = np.concatenate(vals, axis=0, dtype=np.float32)

        # normalise
        obs_array = ((obs_array - self.obs_low) / (self.obs_high - self.obs_low) * 2) - 1

        return obs_array, reward, done, info

    def reset(self):
        obs_dict = self.env.reset()
        
        # modify obs
        vals = []
        for _, value in obs_dict.items():
            vals.append(value)
        obs_array = np.concatenate(vals, axis=0, dtype=np.float32)

        return obs_array


class Cfg:
    def __init__(self):
        super(Cfg, self).__init__()

    def parse(self, model: str) -> Union[DictConfig, ListConfig]:
        """
        Parses agent and env configs files, adds c02 data and returns OmegaConf object
        """
        agent_cfg_path = 'cfgs/' + model + '.yaml'
        # env_cfg_path = 'configs/envs.yaml'
        base = OmegaConf.load(agent_cfg_path)
        # env = OmegaConf.load(env_cfg_path)
        # base.merge_with(env)

        return base
