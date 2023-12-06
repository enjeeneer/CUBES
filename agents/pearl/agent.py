# pylint: disable=[invalid-name, unused_argument]
"""Module for PEARL agent."""
import abc
from pathlib import Path

import torch
import numpy as np
from typing import List, Dict, Optional, Union
from loguru import logger

from agents.base import AbstractAgent, AbstractGaussianMLP
from agents.utils import TruncatedNormal
from agents.pearl.reward_function import PEARLRewardFunction
from agents.pearl.replay_buffer import PEARLReplayBuffer


class PEARL(AbstractAgent, metaclass=abc.ABCMeta):
    """Probabilistic Emission-Abating Reinforcement Learning agent."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        history_length: int,
        device: torch.device,
        learning_steps_per_update: int,
        name: str,
        batch_size: int,
        discount: float,
        ensemble_size: int,
        dynamics_hidden_dimension: int,
        dynamics_hidden_layers: int,
        dynamics_learning_rate: float,
        dynamics_activation: str,
        dynamics_betas: List[float],
        planning_particles: int,
        planning_population: int,
        planning_init_mean: float,
        planning_init_var: float,
        planning_horizon: int,
        planning_iterations: int,
        planning_elite_fraction: float,
        planning_temperature: float,
        planning_momentum: float,
        forecast_idxs: Union[List[int], None],
        reward_function: PEARLRewardFunction,
    ):
        super().__init__(name=name)

        self.learning_steps_per_update = learning_steps_per_update
        self.planning_particles = planning_particles
        self.planning_population = planning_population
        self.device = device
        self.batch_size = batch_size
        self.discount = discount
        self.reward_function = reward_function
        self.action_length = action_length
        self.planning_momentum = planning_momentum
        self.forecast_idxs = forecast_idxs
        self.ensemble_size = ensemble_size

        # MPPI planning parameters
        self.planning_init_means = torch.full(
            size=(planning_horizon, action_length),
            fill_value=planning_init_mean,
            device=device,
            dtype=torch.float,
        )
        self.planning_init_vars = torch.full(
            size=(planning_horizon, action_length),
            fill_value=planning_init_var,
            device=device,
            dtype=torch.float,
        )
        self.planning_horizon = planning_horizon
        self.planning_iterations = planning_iterations
        self.planning_elite_fraction = planning_elite_fraction
        self.planning_temperature = planning_temperature

        # Dynamics models
        self.dynamics_ensemble = torch.nn.ModuleList(
            [
                AbstractGaussianMLP(
                    input_dimension=(observation_length + action_length)
                    * (1 + history_length),
                    output_dimension=observation_length,
                    hidden_dimension=dynamics_hidden_dimension,
                    hidden_layers=dynamics_hidden_layers,
                    activation=dynamics_activation,
                    device=self.device,
                    optimiser=True,
                    learning_rate=dynamics_learning_rate,
                    betas=dynamics_betas,
                    layernorm=True,
                )
                for _ in range(ensemble_size)
            ]
        )

        self.model_indices = [
            np.arange(i, planning_particles, ensemble_size)
            for i in range(ensemble_size)
        ]

    def act(
        self,
        observation: np.ndarray,
        explore: bool = False,
        forecasts: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Takes observation and returns action.
        Args:
            observation: numpy array of shape [observation_length]
            explore: whether to explore
            forecasts: numpy array of shape [horizon, observation_length]
        Returns:
            action: numpy array of shape [action_length]
        """
        observation = torch.as_tensor(
            observation, device=self.device, dtype=torch.float
        )

        action = self.plan(
            observation=observation, explore=explore, forecasts=forecasts
        )

        return action.detach().cpu().numpy()

    @torch.no_grad()
    def plan(
        self,
        observation: torch.Tensor,
        explore: bool,
        forecasts: Optional[np.ndarray] = None,  # pylint: disable=W0613
    ) -> torch.Tensor:
        """
        Runs MPPI planning loop to find best action.
        Args:
            observation: numpy array of shape [observation_length]
            explore: whether to explore
            forecasts: numpy array of shape [horizon, observation_length]
        Returns:
            action: numpy array of shape [action_length]
        """

        mean, var, t = self.planning_init_mean, self.planning_init_var, 0

        # tile observation
        observation = torch.tile(
            observation, (self.planning_particles, self.planning_population, 1, 1)
        )

        while t < self.planning_iterations:

            # sample candidate action sequences
            action_dist = TruncatedNormal(
                loc=mean,
                scale=var,
            )  # TODO: if things break check this,
            # and think about changing a and b to {-2, 2}
            action_samples = action_dist.sample(
                sample_shape=torch.Size(
                    self.planning_population,
                )
            )
            action_samples = torch.tile(
                action_samples, (self.planning_particles, 1, 1, 1)
            )

            expected_values = self.rollout_models(
                observation=observation,
                action_samples=action_samples,
                explore=explore,
            )  # [planning_population]

            # select best (elite) actions from rollouts
            elite_values = expected_values[np.argsort(expected_values)][
                -int(self.planning_elite_fraction * self.planning_population) :
            ]
            elite_actions = action_samples[np.argsort(expected_values)][
                -int(self.planning_elite_fraction * self.planning_population) :
            ]

            # update parameters
            max_value = expected_values.max(0)[0]
            min_value = expected_values.min(0)[0]
            norm_values = (np.absolute(elite_values) - np.absolute(min_value)) / (
                np.absolute(max_value) - np.absolute(min_value)
            ) - 1  # scales to range [-1, 0]

            omega = (
                torch.exp(self.planning_temperature * norm_values)
                .view(norm_values.shape[0], 1, 1)
                .to(self.device)
            )
            omega_tile = torch.tile(
                omega, (1, self.planning_horizon, self.action_length)
            ).to(self.device)

            mean_ = torch.sum(omega_tile * elite_actions, dim=0) / (omega.sum(0) + 1e-9)
            var_ = torch.sqrt(
                torch.sum(omega_tile * (elite_actions - mean_.unsqueeze(0)) ** 2, dim=0)
                / (omega.sum(0) + 1e-9)
            )

            mean = self.planning_momentum * mean + (1 - self.planning_momentum) * mean_
            var = self.planning_momentum * var + (1 - self.planning_momentum) * var_

            t += 1

        actions = mean[0].cpu().detach().numpy()  # first action is trajectory`

        return actions

    @torch.no_grad()
    def rollout_models(
        self,
        observation: torch.Tensor,
        action_samples: torch.Tensor,
        explore: bool = False,
        forecasts: Optional[np.ndarray] = None,
    ):
        """
        Takes observation, passes candidate actions through dynamics models
        and returns best action for exploration or exploitation.
        Args:
            observation: numpy array of shape
                    [planning_particles, population_size, 1, observation_length]
            action_samples: numpy array of shape
                    [planning_population, planning_horizon, action_length]
            forecasts: numpy array of shape [horizon, observation_length]
            explore: whether to explore
        Returns:
            tr: numpy array of shape [action_length]
        """
        trajectories = torch.zeros(
            size=(
                self.planning_particles,
                self.planning_population,
                self.planning_horizon,
                self.observation_length,
            ),
        )
        forecasts = torch.tile(
            torch.tensor(forecasts, device=self.device, dtype=torch.float),
            (self.planning_particles, self.planning_population, 1, 1),
        )

        # planning loop
        for i in range(self.planning_horizon):
            actions = action_samples[:, :, i, :]
            trajectories[:, :, i, :] = observation

            inputs = torch.cat((observation, actions), dim=-1)

            for j, model in enumerate(self.dynamics_ensemble):
                model_inputs = inputs[self.model_indices[j]]
                next_observation, _, _ = model.forward(model_inputs, sample=True)
                observation[self.model_indices[j]] = next_observation

        # impute forecasts

        # calculate expected values
        expected_values = self.reward_function(
            trajectories=trajectories, explore=explore
        )

        return expected_values

    def update(self, replay_buffer: PEARLReplayBuffer) -> Dict:
        """
        Updates dynamics models.
        Args:
            replay_buffer: replay buffer
        Returns:
            losses: dictionary of losses
        """
        metrics = {}

        # update dynamics models
        for j, model in enumerate(self.dynamics_ensemble):
            logger.info(f"Updating PEARL dynamics model {j}")
            aggregate_log_probs = []
            aggregate_mses = []

            for _ in range(self.learning_steps_per_update):
                # sample batch
                (state_actions, next_states) = replay_buffer.sample(
                    batch_size=self.batch_size
                )

                pred_next_states, _, dist = model.forward(state_actions)
                log_prob_loss = -dist.log_prob(next_states).mean()

                model.optimiser.zero_grad()
                log_prob_loss.backward()
                model.optimiser.step()

                # MSE for logging
                mse = torch.nn.MSELoss()
                mse_loss = mse(pred_next_states, next_states)

                aggregate_mses.append(log_prob_loss.item())
                aggregate_mses.append(mse_loss.item())

            metrics[f"dynamics_{j}_log_prob_loss"] = np.mean(aggregate_log_probs)
            metrics[f"dynamics_{j}_mse_loss"] = np.mean(aggregate_mses)

        return metrics

    def load(self, filepath: Path):
        pass
