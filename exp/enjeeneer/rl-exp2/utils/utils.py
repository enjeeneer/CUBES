import gym
import bauwerk
import numpy as np
from typing import Union
from omegaconf import OmegaConf, DictConfig, ListConfig


class ObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ObsWrapper, self).__init__(env)
        self.env = env
        self.obs_low = {}
        self.obs_high = {}
        for key, value in env.observation_space.items():
            self.obs_low[key] = value.low
            self.obs_high[key] = value.high

        # hack obs_low and obs_high until Arduin fixes bauwerk
        self.obs_high['load'] = 4.5
        self.obs_high['pv_gen'] = 3.5

    def step(self, action):
        """
        Steps simulator given action and flattens and normalises observation.
        :param action: array of shape [act_dim]
        :return:
        """
        obs_dict, reward, done, info = self.env.step(action)

        # modify obs
        vals = []
        for key, value in obs_dict.items():
            val = (((value - self.obs_low[key]) / (self.obs_high[key] - self.obs_low[key])) * 2) - 1
            vals.append(val)

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

class TransformerUtils:
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer

    def get_bauewrk_prompt(self, env, obs_dim, act_dim, prompt_steps):
        """
        Creates task-specifc tokenized prompt for DT.
        :param env: Bauwerk environment set to relevant task
        :param cfg: (dict) decision transformer config.
        :return tokenized prompt: array of state-action tokens, shape [context_length,]
        """
        print('...creating prompt...')
        optimal_actions = bauwerk.solve(env)[0]
        state_actions = []

        # create masks
        obs_mask = np.zeros(shape=(prompt_steps + 1, obs_dim + act_dim))  # +1 because we include final additional obs
        act_mask = np.zeros(shape=(prompt_steps, obs_dim + act_dim))
        obs_mask[:, :obs_dim] = np.arange(start=1, stop=obs_dim+1)
        act_mask[:, obs_dim: obs_dim + act_dim] = 1
        obs_mask = obs_mask.flatten()[-self.cfg.transformer.context_length:]
        act_mask = act_mask.flatten()[-self.cfg.transformer.context_length:]

        obs = env.reset()
        for step in range(prompt_steps):
            state_actions.append(obs)
            action = optimal_actions[step]
            obs, _, _, _ = env.step(action)
            state_actions.append(action)

        state_actions.append(obs)

        # correct masks for last obs
        obs_mask[:-obs_dim] = obs_mask[obs_dim:]
        obs_mask[-obs_dim:] = np.arange(start=1, stop=obs_dim+1)
        act_mask[:-obs_dim] = act_mask[obs_dim:]
        act_mask[-obs_dim:] = 0

        prompt = np.concatenate(np.array(state_actions, dtype=object))[-self.cfg.transformer.context_length:]  # flattened array sliced to context length
        tokenised_prompt = self.tokenizer.tokenize(prompt)

        return tokenised_prompt, obs_mask, act_mask
