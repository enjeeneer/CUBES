import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


class Critic(nn.Module):
    def __init__(self, beta, input_dims, layer_dims, device, path):
        super(Critic, self).__init__()
        self.beta = beta
        self.input_dims = input_dims
        self.layer_dims = layer_dims

        self.model = nn.Sequential(
            nn.Linear(self.input_dims, self.layer_dims),
            nn.Tanh(),
            nn.Linear(self.layer_dims, self.layer_dims),
            nn.Tanh(),
            nn.Linear(self.layer_dims, self.layer_dims),
            nn.Tanh(),
            nn.Linear(self.layer_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.path = path
        self.to(device)

    def forward(self, state, action):
        action_value = self.model(T.cat([state, action], dim=-1))

        return action_value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.path)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.path))

class Value(nn.Module):
    def __init__(self, beta, input_dims, layer_dims, device, path):
        super(Value, self).__init__()
        self.beta = beta
        self.input_dims = input_dims
        self.layer_dims = layer_dims

        self.model = nn.Sequential(
            nn.Linear(self.input_dims, self.layer_dims),
            nn.Tanh(),
            nn.Linear(self.layer_dims, self.layer_dims),
            nn.Tanh(),
            nn.Linear(self.layer_dims, self.layer_dims),
            nn.Tanh(),
            nn.Linear(self.layer_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.path = path
        self.to(device)

    def forward(self, state):
        value = self.model(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.path)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.path))

class Actor(nn.Module):
    def __init__(self, alpha, input_dims, layer_dims, act_dim, device, path, max_action):
        super(Actor, self).__init__()
        self.alpha = alpha
        self.input_dims = input_dims
        self.layer_dims = layer_dims
        self.act_dim = act_dim
        self.reparam_noise = 1e-6
        self.log_sig_min = -20
        self.log_sig_max = 2
        self.max_action = T.tensor(max_action, dtype=T.float).to(device)
        self.device = device

        self.model = nn.Sequential(
            nn.Linear(self.input_dims, self.layer_dims),
            nn.Tanh(),
            nn.Linear(self.layer_dims, self.layer_dims),
            nn.Tanh(),
            nn.Linear(self.layer_dims, self.layer_dims),
            nn.Tanh(),
        )

        self.mu = nn.Linear(self.layer_dims, self.act_dim)
        self.log_std = nn.Linear(self.layer_dims, self.act_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.path = path
        self.to(device)

    def forward(self, state):
        x = self.model(state)
        mu = self.mu(x)
        log_std = self.log_std(x)

        log_sigma = T.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)

        return mu, log_sigma

    def sample_normal(self, state, reparam=True):
        mu, log_sigma = self.forward(state)
        sigma = log_sigma.exp()
        normal = Normal(mu, sigma)

        if reparam:
            actions = normal.rsample() # this gives sample plus noise
        else:
            actions = normal.sample()

        action = T.tanh(actions)*self.max_action
        mean = T.tanh(mu)*self.max_action
        log_probs = normal.log_prob(actions)
        log_probs -= T.log((1 - action.pow(2)) + self.reparam_noise) * self.max_action # may need to take max_action outside of log
        # print('pre_summed log probs:', log_probs)
        log_probs = log_probs.sum()
        # print('summed log probs:', log_probs)

        return action, log_probs, mean

    def save_checkpoint(self):
        T.save(self.state_dict(), self.path)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.path))