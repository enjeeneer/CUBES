import gym
import bauwerk
import numpy as np
from typing import Union
from omegaconf import OmegaConf, DictConfig, ListConfig


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
    
class BauwerkEvaluation:
    def __init__(self, cfg):
        self.cfg = cfg

    def rollout(self, model, env):
        # Obtaining model actions and evaluating them
        model_actions = []
        obs = env.reset()
        for i in range(TASK_LEN):
            action, obs_ = model.predict(obs)
            model_actions.append(action)
            obs, _, _, _ = env.step(action)

        p_model = bauwerk.eval.evaluate_actions(model_actions[:TASK_LEN], env)
        return p_model