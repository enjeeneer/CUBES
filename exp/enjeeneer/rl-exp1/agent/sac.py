from agent.critic import DoubleQCritic
from agent.actor import DiagGaussianActor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent import Agent
import utils


class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.sac.device)
        self.critic = DoubleQCritic(cfg).to(self.device)
        self.critic_target = DoubleQCritic(cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor = DiagGaussianActor(cfg).to(self.device)
        self.log_alpha = torch.tensor(np.log(cfg.sac.init_temperature), dtype=torch.float32).to(self.device)
        self.log_alpha.requires_grad = True

        # set target entropy to -|A|
        self.target_entropy = -cfg.sac.action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=cfg.sac.actor_lr,
                                                betas=cfg.sac.actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=cfg.sac.critic_lr,
                                                 betas=cfg.sac.critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=cfg.sac.alpha_lr,
                                                    betas=cfg.sac.alpha_betas)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.cfg.sac.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.cfg.sac.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, step):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.cfg.sac.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()


        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.cfg.sac.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, step):
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.cfg.sac.batch_size)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max, step)

        if step % self.cfg.sac.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, step)

        if step % self.cfg.sac.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.cfg.sac.critic_tau)
