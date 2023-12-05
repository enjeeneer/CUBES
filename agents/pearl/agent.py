# pylint: disable=invalid-name
"""Module for PEARL agent."""

import torch
import numpy as np
from typing import List, Dict

from agents.base import AbstractAgent, AbstractGaussianMLP


class PEARL(AbstractAgent):
    """Probabilistic Emission-Abating Reinforcement Learning agent."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        history_length: int,
        device: torch.device,
        name: str,
        batch_size: int,
        discount: float,
        ensemble_size: int,
        dynamics_hidden_dimension: int,
        dynamics_hidden_layers: int,
        dynamics_learning_rate: float,
        dynamics_activation: str,
        dynamics_betas: List[float, float],
        particle_size: int,
        action_population_size: int,
        planning_init_mean: float,
        planning_init_var: float,
        planning_horizon: int,
        planning_iterations: int,
    ):
        super().__init__(name=name)

        self.particle_size = particle_size
        self.action_population_size = action_population_size
        self.device = device
        self.batch_size = batch_size
        self.discount = discount
        self.planning_init_mean = planning_init_mean
        self.planning_init_var = planning_init_var
        self.planning_horizon = planning_horizon
        self.planning_iterations = planning_iterations

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
                )
                for _ in range(ensemble_size)
            ]
        )

        self.model_indices = [
            np.arange(i, particle_size, ensemble_size) for i in range(ensemble_size)
        ]

    def act(self, observation: np.ndarray) -> np.ndarray:
        pass
        # action = self.plan(observation=observation)

        # return action

    def plan(self, observation: np.ndarray) -> np.ndarray:
        pass

    def update(self, *args, **kwargs) -> Dict:
        pass
