import os
import numpy as np
import torch as T
from sac.networks import Actor, Value, Critic
from sac.memory import SACMemory

class Agent:
    def __init__(self, cfg, env, models_dir):
        self.cfg = cfg
        self.device = T.device('cpu')
        self.models_dir = models_dir
        self.act_dim = env.action_space.shape[0]

        obs_dim = 0
        for _, value in env.observation_space.items():
            obs_dim += value.shape[0]
        self.obs_dim = obs_dim

        self.network_input_dims = self.obs_dim * (1 + cfg.hist_length)
        self.n_steps = 0

        ### NETWORKS ###
        self.actor = Actor(alpha=self.cfg.alpha, input_dims=self.network_input_dims, layer_dims=cfg.layer_dims, max_action=env.action_space.high,
                           act_dim=self.act_dim, device=self.device, path=os.path.join(models_dir, 'actor.pth'))
        self.critic_1 = Critic(beta=self.cfg.alpha, input_dims=self.network_input_dims + self.act_dim, layer_dims=cfg.layer_dims,
                               device=self.device, path=os.path.join(models_dir, 'critic_1.pth'))
        self.critic_2 = Critic(beta=self.cfg.alpha, input_dims=self.network_input_dims + self.act_dim, layer_dims=cfg.layer_dims,
                               device=self.device, path=os.path.join(models_dir, 'critic_2.pth'))
        self.value = Value(beta=self.cfg.alpha, input_dims=self.network_input_dims, layer_dims=cfg.layer_dims,
                           device=self.device, path=os.path.join(models_dir, 'value.pth'))
        self.target_value = Value(beta=self.cfg.alpha, input_dims=self.network_input_dims, layer_dims=cfg.layer_dims,
                                  device=self.device, path=os.path.join(models_dir, 'target_value.pth'))
        self.update_network_parameters(tau=1)
        self.memory = SACMemory(batch_size=self.cfg.batch_size,
                                hist_length=self.cfg.hist_length,
                                state_dim=self.obs_dim,
                                act_dim=self.act_dim)

    def act(self, obs: np.array, evaluate: bool = False):
        '''
        Selects action based on current environment observation.
        :param obs: array of current envnvironment observation of shape (state_dim,)
        '''
        
        if self.cfg.hist_length > 0:
            history = self.memory.get_history()
            state_tensor = T.cat(
                tensors=(T.tensor(obs, dtype=T.float).to(self.device), T.tensor(history, dtype=T.float).to(self.device)),
                dim=0
            )
            self.memory.store_history(obs)
            assert state_tensor.shape[0] == self.network_input_dims
        else:
            state_tensor = T.tensor(obs, dtype=T.float).to(self.device)

        if evaluate:
            _, _, action = self.actor.sample_normal(state_tensor, reparam=False)
        else:
            action, _, _ = self.actor.sample_normal(state_tensor, reparam=False)

        action = action.cpu().detach().numpy()
        inp = state_tensor.cpu().detach().numpy()

        return action, inp

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.cfg.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                                     (1 - tau) * target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)


    def learn(self):
        for i in range(1):
            obs_mem, obs_mem_, actions_mem, rewards_mem, done_mem = self.memory.sample()
        
            obs_T = T.tensor(obs_mem, dtype=T.float).to(self.device)
            actions_T = T.tensor(actions_mem, dtype=T.float).to(self.device)
            rewards_T = T.tensor(rewards_mem, dtype=T.float).to(self.device)
            obs_T_ = T.tensor(obs_mem_, dtype=T.float).to(self.device)
            done_T = T.tensor(done_mem).to(self.device)

            value = self.value.forward(obs_T).view(-1) # collapsing of dimension may not be correct
            value_ = self.target_value.forward(obs_T_).view(-1)
            value_[done_T] = 0.0

            actions, log_probs, _ = self.actor.sample_normal(obs_T, reparam=False)
            log_probs = log_probs.view(-1) # collapsing of dimension may not be correct
            q1_new_policy = self.critic_1.forward(obs_T, actions)
            q2_new_policy = self.critic_2.forward(obs_T, actions)
            critic_value = T.min(q1_new_policy, q2_new_policy).view(-1)

            self.value.optimizer.zero_grad()
            value_target = critic_value - log_probs
            value_loss = 0.5 * T.nn.functional.mse_loss(value, value_target)
            value_loss.backward(retain_graph=True)
            self.value.optimizer.step()

            actions, log_probs, _ = self.actor.sample_normal(obs_T, reparam=True)
            log_probs = log_probs.view(-1)
            q1_new_policy = self.critic_1.forward(obs_T, actions)
            q2_new_policy = self.critic_2.forward(obs_T, actions)
            critic_value = T.min(q1_new_policy, q2_new_policy).view(-1)

            actor_loss = log_probs - critic_value
            actor_loss = T.mean(actor_loss)
            self.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor.optimizer.step()

            self.critic_1.optimizer.zero_grad()
            self.critic_2.optimizer.zero_grad()
            q_hat = self.cfg.scale * rewards_T + self.cfg.gamma * value_
            q1_old_policy = self.critic_1.forward(obs_T, actions_T).view(-1)
            q2_old_policy = self.critic_2.forward(obs_T, actions_T).view(-1)
            critic_1_loss = 0.5 * T.nn.functional.mse_loss(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * T.nn.functional.mse_loss(q2_old_policy, q_hat)

            critic_loss = critic_1_loss + critic_2_loss
            critic_loss.backward()
            self.critic_1.optimizer.step()
            self.critic_2.optimizer.step()

            # if self.n_steps % self.soft_steps:
            #     print('network params')
            self.update_network_parameters()

            return value_loss, actor_loss, critic_loss

    def save_models(self):
        '''
        Saves parameters of each model in ensemble to directory
        '''
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_value.save_checkpoint()
        self.value.save_checkpoint()

    def load_models(self):
        '''
        Loads parameters of pre-trained models from directory
        '''
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_value.load_checkpoint()
        self.value.load_checkpoint()