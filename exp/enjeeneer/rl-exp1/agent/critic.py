import utils
import torch
from torch import nn


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.Q1 = utils.mlp(cfg.critic.obs_dim + cfg.critic.action_dim, cfg.critic.hidden_dim, 1, cfg.critic.hidden_depth)
        self.Q2 = utils.mlp(cfg.critic.obs_dim + cfg.critic.action_dim, cfg.critic.hidden_dim, 1, cfg.critic.hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2


