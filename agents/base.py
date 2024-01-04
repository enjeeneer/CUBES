# pylint: disable=invalid-name
"""Module for holding abstract base classes for all agents."""

import abc
from pathlib import Path
from typing import List, Tuple, Dict

import gym
import numpy as np
import torch
import wandb
import dataclasses

from agents.utils import TruncatedNormal, reparameterise, squashed_gaussian


class AbstractAgent(torch.nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for all agents."""

    def __init__(
        self,
        name: str,
    ):
        super().__init__()
        self.name = name

    @abc.abstractmethod
    def act(self, *args, **kwargs) -> torch.Tensor:
        """
        Returns an action for a given input.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> Dict:
        """
        Updates parameters of model.
        """
        raise NotImplementedError

    def save(self, dir_path: Path) -> Path:
        """
        Saves a copy of the model in a format that can be loaded by load
        """
        dir_path.mkdir(exist_ok=True)
        save_path = dir_path / Path(str(self.name) + ".pickle")
        torch.save(self, save_path)

        return save_path

    @abc.abstractmethod
    def load(self, filepath: Path):
        pass


class AbstractMLP(torch.nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for all feedforward networks."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        hidden_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
        preprocessor: bool = False,
        layernorm: bool = False,
    ):
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension
        self._hidden_dimension = hidden_dimension
        self._hidden_layers = hidden_layers
        self._activation = activation
        self.device = device
        self._preprocessor = preprocessor
        self._layernorm = layernorm

        super().__init__()
        self.trunk = self._build()

    def _build(self) -> torch.nn.Sequential:
        """
        Creates MLP trunk.
        """
        if self.hidden_layers == 0:
            function = [torch.nn.Linear(self.input_dimension, self.output_dimension)]
        else:
            # first layer
            # ICLR paper uses layer norm and tanh for first layer of every network
            if self._layernorm:
                function = [
                    torch.nn.Linear(self.input_dimension, self.hidden_dimension),
                    torch.nn.LayerNorm(self.hidden_dimension),
                    torch.nn.Tanh(),
                ]
            else:
                function = [
                    torch.nn.Linear(self.input_dimension, self.hidden_dimension),
                    self.activation,
                ]

            # hidden layers
            for _ in range(self.hidden_layers - 1):
                function += [
                    torch.nn.Linear(self.hidden_dimension, self.hidden_dimension),
                    self.activation,
                ]

            # last layer
            function.append(
                torch.nn.Linear(self.hidden_dimension, self.output_dimension)
            )

        # add non-linearity to last layer for preprocessor
        if self.preprocessor:
            function.append(self.activation)

        trunk = torch.nn.Sequential(*function).to(self.device)

        return trunk

    @property
    def input_dimension(self) -> int:
        return self._input_dimension

    @property
    def output_dimension(self) -> int:
        return self._output_dimension

    @property
    def hidden_dimension(self) -> int:
        return self._hidden_dimension

    @property
    def hidden_layers(self) -> int:
        return self._hidden_layers

    @property
    def activation(self) -> torch.nn:
        if self._activation == "relu":
            return torch.nn.ReLU()
        elif self._activation == "tanh":
            return torch.nn.Tanh()
        else:
            raise NotImplementedError(f"{self._activation} not implemented.")

    @property
    def preprocessor(self) -> bool:
        return self._preprocessor


class AbstractCritic(AbstractMLP, metaclass=abc.ABCMeta):
    """Abstract critic class."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        hidden_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
    ):
        self._observation_length = observation_length
        self._action_length = action_length
        self._hidden_dimension = hidden_dimension
        self._hidden_layers = hidden_layers
        super().__init__(
            input_dimension=observation_length + action_length,
            output_dimension=int(1),
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            preprocessor=False,
            layernorm=False,
        )

    def forward(self, observation_action: torch.Tensor) -> torch.Tensor:
        """
        Passes observation_action pair through network to predict q value
        Args:
            observation_action: tensor of shape
                                        [batch_dim, observation_length + action_length]

        Returns:
            q: q value tensor of shape [batch_dim, 1]
        """
        q = self.trunk(observation_action)  # pylint: disable=E1102

        return q


class AbstractActor(AbstractMLP, metaclass=abc.ABCMeta):
    """Abstract actor that selects action given input."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        hidden_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
    ):
        super().__init__(
            input_dimension=observation_length,
            output_dimension=action_length,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            layernorm=True,
        )

    def forward(
        self, observation: torch.Tensor, std: float
    ) -> torch.distributions.Distribution:
        """
        Passes input through network to predict action
        Args:
            observation: obs tensor of shape [batch_dim, input_length]
            std: standard deviation of action distribution
        Returns:
            action: action tensor of shape [batch_dim, action_length]
        """
        if observation.shape[-1] != self.input_dimension:
            raise ValueError(
                f"Input shape {observation.shape} does not "
                f"match input dimension {self.input_dimension}"
            )

        mu = self.trunk(observation)  # pylint: disable=E1102
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)

        return dist


class AbstractGaussianActor(AbstractMLP, metaclass=abc.ABCMeta):
    """Abstract gaussian actor that selects action given input."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        hidden_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
        log_std_bounds: Tuple[float] = (-5.0, 2.0),
    ):

        self.log_std_min = log_std_bounds[0]
        self.log_std_max = log_std_bounds[1]

        super().__init__(
            input_dimension=observation_length,
            output_dimension=action_length * 2,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            layernorm=False,
        )

    def forward(self, observation: torch.Tensor, sample=True):
        """
        Takes observation and returns squashed normal distribution over action space.
        Args:
            observation: tensor of shape [batch_dim, observation_length]

        Returns:
            dist: SquashedNormal (multivariate Gaussian) dist over action space.

        """
        # mu, log_std = self.trunk(observation).chunk(2, dim=-1)  # pylint: disable=E1102
        output = self.trunk(observation)
        action, log_prob = squashed_gaussian(x=output, sample=sample)

        return action, log_prob


class PEARLGaussianMLP(AbstractMLP, metaclass=abc.ABCMeta):
    """
    Abstract gaussian MLP that predicts mean and var of each
    output dimension.
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        observation_length: int,
        hidden_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
        observation_space: gym.Space,
        history_length: int,
        log_std_bounds: Tuple[float] = (-20.0, 2.0),
        optimiser: bool = False,
        learning_rate: float = 1e-4,
        betas=None,
        layernorm=False,
    ):

        if betas is None:
            betas = [0.9, 0.99]

        self.log_std_min = log_std_bounds[0]
        self.log_std_max = log_std_bounds[1]
        self.observation_length = observation_length
        print("observation space high", observation_space.high)
        self.observation_upper_bounds = torch.tensor(
            np.tile(observation_space.high, history_length + 1),
            dtype=torch.float32,
            device=device,
        )
        print("upper bounds", self.observation_upper_bounds)
        self.observation_lower_bounds = torch.tensor(
            np.tile(observation_space.low, history_length + 1),
            dtype=torch.float32,
            device=device,
        )
        print("lower bounds", self.observation_lower_bounds)

        super().__init__(
            input_dimension=input_dimension,
            output_dimension=output_dimension * 2,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            layernorm=layernorm,
        )

        if optimiser:
            self.optimiser = torch.optim.Adam(
                self.trunk.parameters(), lr=learning_rate, betas=betas
            )

        self.min_logstd = log_std_bounds[0]
        self.max_logstd = log_std_bounds[1]

    def forward(
        self,
        observation_history: torch.Tensor,
        actions: torch.Tensor,
        sample: bool = True,
    ):
        """
        Takes observation and returns squashed normal distribution over action space.
        Args:
            observation_history: tensor of shape
                [batch_dim, observation_length * history_length]
            sample: whether to sample from distribution or not
        Returns:
            output: sampled output
            log_prob: log probability of sampled output

        """
        # normalise observation
        obsersation_norm = (
            2
            * (
                (observation_history - self.observation_lower_bounds)
                / (self.observation_upper_bounds - self.observation_lower_bounds)
            )
            - 1
        )

        model_input = torch.cat([obsersation_norm, actions], dim=-1)
        hidden = self.trunk(model_input)  # pylint: disable=E1102

        mean, log_std, dist = reparameterise(
            hidden, clamp=("hard", self.min_logstd, self.max_logstd)
        )

        if sample:
            output = dist.rsample()
            output = torch.clamp(
                output, -1.0, 1.0
            )  # incase rsample falls outside bounds
        else:
            output = mean

        # unnormalise predictions
        pred = (output + 1 / 2) * (
            self.observation_upper_bounds - self.observation_lower_bounds
        ) + self.observation_lower_bounds

        return pred, log_std


class AbstractLogger(metaclass=abc.ABCMeta):
    """
    Abstract class for collecting metrics from training
    / eval runs.
    """

    def __init__(
        self, agent_config: Dict, use_wandb: bool = False, wandb_tags: List[str] = None
    ):
        self._agent_config = agent_config
        self.metrics = {}  # overwritten in concrete class

        if use_wandb:
            wandb.init(
                project="zero-shot",
                entity="zero-shot-rl",
                config=agent_config,
                tags=wandb_tags,
                reinit=True,
            )

    def log(self, metrics: Dict[str, float]):
        """Adds metrics to logger."""

        for key, value in metrics.items():
            try:
                self.metrics[key].append(value)
            except KeyError:
                raise KeyError(  # pylint: disable=W0707
                    f"Metric {key} not in metrics dictionary."
                )  # pylint: disable=W0707

        if wandb.run is not None:
            wandb.log(metrics)


class AbstractReplayBuffer(metaclass=abc.ABCMeta):
    """
    Abstract replay buffer class for storing
    transitions from an environment.
    """

    def __init__(self, device: torch.device):
        self.device = device

    @abc.abstractmethod
    def add(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, batch_size: int) -> Dict:
        raise NotImplementedError


class AbstractOnlineReplayBuffer(AbstractReplayBuffer, metaclass=abc.ABCMeta):
    """Abstract buffer for online RL algorithms."""

    def __init__(
        self,
        capacity: int,
        observation_length: int,
        action_length: int,
        device: torch.device,
    ):
        super().__init__(device=device)
        self.observations = NotImplementedError("observations array not defined.")
        self.next_observations = NotImplementedError(
            "next_observations array not defined."
        )
        self.actions = NotImplementedError("actions array not defined.")
        self.rewards = NotImplementedError("rewards array not defined.")
        self.dones = NotImplementedError("dones array not defined.")
        self.current_memory_index = NotImplementedError(
            "current memory index not defined."
        )
        self.full_memory = NotImplementedError("full memory flag not implemented.")

        # properties
        self._capacity = capacity
        self._observation_length = observation_length
        self._action_length = action_length

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def observation_length(self) -> int:
        return self._observation_length

    @property
    def action_length(self) -> int:
        return self._action_length


class AbstractWorkspace(metaclass=abc.ABCMeta):
    """
    Abstract workspace for training and evaluating agents
    in an environment.
    """

    def __init__(
        self,
        env,
        eval_rollouts: int,
        wandb_logging: bool,
        wandb_entity: str,
        wandb_project: str,
        wandb_tags: List[str],
    ):
        self.env = env
        self.eval_rollouts = eval_rollouts
        self.wandb_logging = wandb_logging
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.wandb_tags = wandb_tags

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def eval(self, *args, **kwargs):
        raise NotImplementedError


@dataclasses.dataclass
class Batch:
    """
    Dataclass for batches of offline data.

    Args:
        input_sequences: tensor of shape [batch_dim, context_length]
        targets: tensor of shape [batch_dim, 1]
        observation_masks: tensor of shape [batch_dim, context_length]
        action_masks: tensor of shape [batch_dim, context_length]
        reward_masks: tensor of shape [batch_dim, context_length]
        target_action_masks: tensor of shape [batch_dim, context_length]
    """

    inputs: np.ndarray
    targets: np.ndarray
    observation_masks: np.ndarray
    action_masks: np.ndarray
    reward_masks: np.ndarray
    target_action_masks: np.ndarray


class OfflineReplayBuffer(AbstractReplayBuffer, metaclass=abc.ABCMeta):
    """
    Abstract replay buffer class for storing
    transitions from an environment.
    """

    def __init__(self, device: torch.device):
        super().__init__(device)

        self.storage = NotImplementedError("Storage not implemented in base class.")

    @abc.abstractmethod
    def load_offline_dataset(
        self,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError
