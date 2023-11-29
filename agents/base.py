# pylint: disable=invalid-name
"""Module for holding abstract base classes for all agents."""

import abc
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import wandb
import dataclasses
from loguru import logger
from tqdm import tqdm

from agents.utils import TruncatedNormal, squashed_gaussian
from agents.sac.agent import SoftActorCritic
from agents.sac.replay_buffer import SoftActorCriticReplayBuffer
from cubes.rbcs.rbc import GeneralRBC


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
        save_path = dir_path / Path(str(self.name))
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

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def eval(
        self,
        agent: Tuple[SoftActorCritic, GeneralRBC],
        replay_buffer: SoftActorCriticReplayBuffer,
        checkpoints: bool = True,
    ) -> Dict[str, float]:
        """
        Performs eval rollouts and logs metrics for RBC and SAC.
        Args:
            agent: tuple of SAC and RBC agents
            replay_buffer: replay buffer for SAC agent
            checkpoints: True if eval is being called during training; False
                        if eval is being called for inference.
        Returns:
        """
        if not checkpoints and self.wandb_logging:
            run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                tags=self.wandb_tags,
                reinit=True,
            )

        logger.info("Performing eval train.")
        eval_rewards = []
        eval_emissions = []
        eval_ndt_t_violations = {}
        eval_ndt_aq_violations = {}
        eval_heating_dt = {}
        eval_heating_service_dt = {}
        eval_max_heating_service_dt = {}
        eval_heating_beyond_comf_dt = {}
        eval_violation_dt = {}
        eval_violation_daq = {}
        eval_emissions_reward = []
        eval_comfort_reward = []
        eval_aq_reward = []

        if isinstance(agent, SoftActorCritic):
            agent.eval()

        for _ in tqdm(range(self.eval_rollouts)):
            done = False
            rollout_reward = []
            rollout_emissions = 0.0
            rollout_ndt_t_violations = {}
            rollout_ndt_aq_violations = {}
            rollout_heating_dt = {}
            rollout_heating_service_dt = {}
            rollout_max_heating_service_dt = {}
            rollout_violation_daq = {}
            rollout_heating_beyond_comf_dt = {}
            rollout_violation_dt = {}
            rollout_emissions_reward = []
            rollout_comfort_reward = []
            rollout_aq_reward = []

            obs = self.env.reset()
            while not done:
                if isinstance(agent, SoftActorCritic):
                    action = agent.act(
                        obs,
                        sample=False,
                        replay_buffer=replay_buffer,
                    )
                else:
                    action = agent.act(obs)

                obs, reward, done, info = self.env.step(action)
                rollout_reward.append(reward)
                rollout_emissions += info["emissions"]

                if not rollout_ndt_t_violations:
                    for k, v in info["t_violation"].items():
                        rollout_ndt_t_violations[k] = v
                else:
                    for k, v in info["t_violation"].items():
                        rollout_ndt_t_violations[k] += v

                if not rollout_ndt_aq_violations:
                    for k, v in info["aq_violation"].items():
                        rollout_ndt_aq_violations[k] = v
                else:
                    for k, v in info["aq_violation"].items():
                        rollout_ndt_aq_violations[k] += v

                if not rollout_heating_dt:
                    for k, v in info["heating_delta_T"].items():
                        rollout_heating_dt[k] = v / 144
                else:
                    for k, v in info["heating_delta_T"].items():
                        rollout_heating_dt[k] += v / 144

                if not rollout_heating_service_dt:
                    for k, v in info["heating_service"].items():
                        rollout_heating_service_dt[k] = v / 144
                else:
                    for k, v in info["heating_service"].items():
                        rollout_heating_service_dt[k] += v / 144

                if not rollout_max_heating_service_dt:
                    for k, v in info["max_heating_service"].items():
                        rollout_max_heating_service_dt[k] = v / 144
                else:
                    for k, v in info["max_heating_service"].items():
                        rollout_max_heating_service_dt[k] += v / 144

                if not rollout_heating_beyond_comf_dt:
                    for k, v in info["heating_beyond_comf_delta_T"].items():
                        rollout_heating_beyond_comf_dt[k] = v / 144
                else:
                    for k, v in info["heating_beyond_comf_delta_T"].items():
                        rollout_heating_beyond_comf_dt[k] += v / 144

                if not rollout_violation_dt:
                    for k, v in info["violation_delta_T"].items():
                        rollout_violation_dt[k] = v / 144
                else:
                    for k, v in info["violation_delta_T"].items():
                        rollout_violation_dt[k] += v / 144

                if not rollout_violation_daq:
                    for k, v in info["violation_delta_aq"].items():
                        rollout_violation_daq[k] = v / 144
                else:
                    for k, v in info["violation_delta_aq"].items():
                        rollout_violation_daq[k] += v / 144

                rollout_emissions_reward.append(info["reward_emissions"])
                rollout_comfort_reward.append(info["reward_comfort"])
                rollout_aq_reward.append(info["reward_air_quality"])

            eval_rewards.append(np.mean(rollout_reward))
            eval_emissions_reward.append(np.mean(rollout_emissions_reward))
            eval_comfort_reward.append(np.mean(rollout_comfort_reward))
            eval_aq_reward.append(np.mean(rollout_aq_reward))
            eval_emissions.append(np.mean(rollout_emissions))

            if not eval_ndt_t_violations:
                for k, v in rollout_ndt_t_violations.items():
                    eval_ndt_t_violations[k] = [v]
            else:
                for k, v in rollout_ndt_t_violations.items():
                    eval_ndt_t_violations[k].append(v)

            if not eval_ndt_aq_violations:
                for k, v in rollout_ndt_aq_violations.items():
                    eval_ndt_aq_violations[k] = [v]
            else:
                for k, v in rollout_ndt_aq_violations.items():
                    eval_ndt_aq_violations[k].append(v)

            if not eval_heating_dt:
                for k, v in rollout_heating_dt.items():
                    eval_heating_dt[k] = [v]
            else:
                for k, v in rollout_heating_dt.items():
                    eval_heating_dt[k].append(v)

            if not eval_heating_service_dt:
                for k, v in rollout_heating_service_dt.items():
                    eval_heating_service_dt[k] = [v]
            else:
                for k, v in rollout_heating_service_dt.items():
                    eval_heating_service_dt[k].append(v)

            if not eval_max_heating_service_dt:
                for k, v in rollout_max_heating_service_dt.items():
                    eval_max_heating_service_dt[k] = [v]
            else:
                for k, v in rollout_heating_service_dt.items():
                    eval_heating_service_dt[k].append(v)

            if not eval_heating_beyond_comf_dt:
                for k, v in rollout_heating_beyond_comf_dt.items():
                    eval_heating_beyond_comf_dt[k] = [v]
            else:
                for k, v in rollout_heating_beyond_comf_dt.items():
                    eval_heating_beyond_comf_dt[k].append(v)

            if not eval_violation_dt:
                for k, v in rollout_violation_dt.items():
                    eval_violation_dt[k] = [v]
            else:
                for k, v in rollout_violation_dt.items():
                    eval_violation_dt[k].append(v)

            if not eval_violation_daq:
                for k, v in rollout_violation_daq.items():
                    eval_violation_daq[k] = [v]
            else:
                for k, v in rollout_violation_daq.items():
                    eval_violation_daq[k].append(v)

        self.env.reset()
        eval_t_violations_means = {}
        for k, v in eval_ndt_t_violations.items():
            eval_t_violations_means[k] = float(np.mean(v))

        eval_aq_violations_means = {}
        for k, v in eval_ndt_aq_violations.items():
            eval_aq_violations_means[k] = float(np.mean(v))

        eval_heating_dt_means = {}
        for k, v in eval_heating_dt.items():
            eval_heating_dt_means[k] = float(np.mean(v))

        eval_heating_service_dt_means = {}
        for k, v in eval_heating_service_dt.items():
            eval_heating_service_dt_means[k] = float(np.mean(v))

        eval_max_heating_service_dt_means = {}
        for k, v in eval_max_heating_service_dt.items():
            eval_max_heating_service_dt_means[k] = float(np.mean(v))

        eval_heating_beyond_comf_dt_means = {}
        for k, v in eval_heating_beyond_comf_dt.items():
            eval_heating_beyond_comf_dt_means[k] = float(np.mean(v))

        eval_violation_dt_means = {}
        for k, v in eval_violation_dt.items():
            eval_violation_dt_means[k] = float(np.mean(v))

        eval_violation_daq_means = {}
        for k, v in eval_violation_daq.items():
            eval_violation_daq_means[k] = float(np.mean(v))

        metrics = {
            "eval/mean_episode_reward": float(np.mean(eval_rewards)),
            "eval/mean_episode_emissions_reward": float(np.mean(eval_emissions_reward)),
            "eval/mean_episode_comfort_reward": float(np.mean(eval_comfort_reward)),
            "eval/mean_episode_air_quality_reward": float(np.mean(eval_aq_reward)),
            "eval/mean_episode_emissions": float(np.mean(eval_emissions)),
            "eval/mean_episode_ndt_t_violations": eval_t_violations_means,
            "eval/mean_episode_ndt_aq_violations": eval_aq_violations_means,
            "eval/mean_episode_heating_degree_days": eval_heating_dt_means,
            "eval/mean_episode_heating_service_degree_days": (
                eval_heating_service_dt_means
            ),
            "eval/mean_episode_max_heating_service_degree_days": (
                eval_max_heating_service_dt_means
            ),
            "eval/mean_episode_heating_beyond_comfort_degree_days": (
                eval_heating_beyond_comf_dt_means
            ),
            "eval/mean_episode_violation_degree_days": eval_violation_dt_means,
            "eval/mean_episode_violation_ppm_days": eval_violation_daq_means,
        }

        if not checkpoints and self.wandb_logging:
            run.log(metrics)
            run.finish()

        return metrics


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
