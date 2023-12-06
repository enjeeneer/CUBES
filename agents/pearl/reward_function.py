# pylint: disable=[consider-using-f-string, unused-argument]
"""
This file contains the reward function for the PEARL agent.
"""

import numpy as np
import warnings
from typing import Union, Tuple, List

import torch

from cubes.constants import NATURAL_GAS_EMISSIONS_FACTOR, MJ_TO_KWH

_DEFAULT_VALUE_AT_MARGIN = 0.1


def _sigmoids(
    x: Union[float, np.ndarray], value_at_1: float, sigmoid: str
) -> np.ndarray:
    """
    Reimplemented from the DeepMind Control Suite:
    https://github.com/google-deepmind/dm_control/blob/main/dm_control/utils/rewards.py
    Returns 1 when `x` == 0, between 0 and 1 otherwise.

    Args:
      x: A scalar or numpy array.
      value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
      sigmoid: String, choice of sigmoid type.

    Returns:
      A numpy array with values between 0.0 and 1.0.

    Raises:
      ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
        `quadratic` sigmoids which allow `value_at_1` == 0.
      ValueError: If `sigmoid` is of an unknown type.
    """
    if sigmoid in ("cosine", "linear", "quadratic"):
        if not 0 <= value_at_1 < 1:
            raise ValueError(
                "`value_at_1` must be nonnegative and smaller than 1, "
                "got {}.".format(value_at_1)
            )
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError(
                "`value_at_1` must be strictly between 0 and 1, "
                "got {}.".format(value_at_1)
            )

    if sigmoid == "gaussian":
        scale = np.sqrt(-2 * np.log(value_at_1))
        return np.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == "hyperbolic":
        scale = np.arccosh(1 / value_at_1)
        return 1 / np.cosh(x * scale)

    elif sigmoid == "long_tail":
        scale = np.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == "reciprocal":
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    elif sigmoid == "cosine":
        scale = np.arccos(2 * value_at_1 - 1) / np.pi
        scaled_x = x * scale
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message="invalid value encountered in cos"
            )
            cos_pi_scaled_x = np.cos(np.pi * scaled_x)
        return np.where(abs(scaled_x) < 1, (1 + cos_pi_scaled_x) / 2, 0.0)

    elif sigmoid == "linear":
        scale = 1 - value_at_1
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    elif sigmoid == "quadratic":
        scale = np.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

    elif sigmoid == "tanh_squared":
        scale = np.arctanh(np.sqrt(1 - value_at_1))
        return 1 - np.tanh(x * scale) ** 2

    else:
        raise ValueError("Unknown sigmoid type {!r}.".format(sigmoid))


def tolerance(
    x,
    bounds=(0.0, 0.0),
    margin=0.0,
    sigmoid="gaussian",
    value_at_margin=_DEFAULT_VALUE_AT_MARGIN,
):
    """
    Reimplemented from the DeepMind Control Suite:
    https://github.com/google-deepmind/dm_control/blob/main/dm_control/utils/rewards.py
    Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

    Args:
        x: A scalar or numpy array.
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
          the target interval. These can be infinite if the interval is unbounded
          at one or both ends, or they can be equal to one another if the target
          value is exact.
        margin: Float. Parameter that controls how steeply the output decreases as
          `x` moves out-of-bounds.
          * If `margin == 0` then the output will be 0 for all values of `x`
            outside `bounds`.
          * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
           'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
        value_at_margin: A float between 0 and 1 specifying the output value when
          the distance from `x` to the nearest bound is equal to `margin`. Ignored
          if `margin == 0`.

    Returns:
        A float or numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
    """

    lower, upper = bounds
    if lower > upper:
        raise ValueError("Lower bound must be <= upper bound.")
    if margin < 0:
        raise ValueError("`margin` must be non-negative.")

    in_bounds = np.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = np.where(in_bounds, 1.0, 0.0)
    else:
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

    return float(value) if np.isscalar(x) else value


class PEARLRewardFunction:
    """
    This class evaluates trajectories predicted by PEARL.
    """

    def __init__(
        self,
        observation_variables: List[str],
        action_variables: List[str],
        temperature_variables: List[str],
        air_quality_variables: List[str],
        occupancy_variables: List[str],
        emissions_variables: List[str],
        temp_range_comfort: Tuple[int, int],
        battery_power_rating: float,
        heating_system_capacity: float,  # in W
        max_emissions_factor: float,  # in gCO2e/kWh
        heat_pump: bool,
        battery: bool,
        sleep_hours: Tuple[int, int] = (23, 6),
        negative_emissions_for_export: bool = False,
        timesteps_per_hour: int = 6,
        air_quality_range: Tuple[int, int] = (0, 1000),
        emissions_weight: float = 1.0,
        air_quality_weight: float = 1.0,
        temperature_weight: float = 1.0,
        temperature_margin: float = 3.0,
    ):
        """
        Tolerance based reward function.
        """

        # get the indices of the variables that are used in the reward function
        self.temperature_idxs = []
        self.air_quality_idxs = []
        self.emissions_idxs = []
        self.occupancy_idxs = []

        # here the key is the EPlus zone and value is the variable name
        for key in temperature_variables:
            for act_var in action_variables:
                if key in act_var:
                    idx = observation_variables.index(key)
                    if idx not in self.temperature_idxs:
                        self.temperature_idxs.append(idx)

        # here the key is the EPlus zone and value is the variable name
        for key in air_quality_variables:
            for act_var in action_variables:
                if key in act_var:
                    idx = observation_variables.index(key)
                    if idx not in self.air_quality_idxs:
                        self.air_quality_idxs.append(idx)

        for key in occupancy_variables:
            idx = observation_variables.index(key)
            if idx not in self.occupancy_idxs:
                self.occupancy_idxs.append(idx)

        for key in emissions_variables:
            idx = observation_variables.index(key)
            if idx not in self.emissions_idxs:
                self.emissions_idxs.append(idx)

        # Reward parameters
        self.temp_range_comfort = temp_range_comfort
        self.sleep_hours = sleep_hours
        self.negative_emissions_for_export = negative_emissions_for_export
        self.timesteps_per_hour = timesteps_per_hour
        self.air_quality_range = air_quality_range
        self.emission_weight = emissions_weight
        self.air_quality_weight = air_quality_weight
        self.temperature_weight = temperature_weight
        self.temperature_margin = temperature_margin

        # heating capacity is in W, emissions factor is in gCO2e/kWh
        # convert to kW and kgCO2e/kWh
        heating_system_capacity_kw = heating_system_capacity / 1000  # W -> kW
        max_elec_emissions_factor_kgco2e = (
            max_emissions_factor / 1000
        )  # gCO2e/kWh -> kgCO2e/kWh
        natural_gas_emissions_factor_kgco2e = NATURAL_GAS_EMISSIONS_FACTOR / (
            MJ_TO_KWH * 1000
        )  # g/MJ -> kgCO2e/kWh

        # calculate min/max emissions bounds
        max_heating_emissions = (
            heating_system_capacity_kw
            * max_elec_emissions_factor_kgco2e
            * (1 / timesteps_per_hour)
            if heat_pump
            else heating_system_capacity_kw
            * (natural_gas_emissions_factor_kgco2e)
            * (1 / timesteps_per_hour)
        )
        if battery:
            battery_power_rating_kw = battery_power_rating / 1000  # W -> kW
            battery_charging_emissions = (
                battery_power_rating_kw
                * max_elec_emissions_factor_kgco2e
                * (1 / timesteps_per_hour)
            )
        else:
            battery_charging_emissions = 0

        self.max_emissions = max_heating_emissions + battery_charging_emissions

        if negative_emissions_for_export:
            self.min_emissions = -battery_charging_emissions
        else:
            self.min_emissions = 0

    def __call__(self, trajectories: torch.Tensor, explore: bool) -> np.ndarray:
        """
        Takes tensor of trajectories as predicted by PEARL and calculated
        and expected value, either reward on var.
        Args:
            trajectories: tensor of shape
                    [planning_particles, planning_population,
                    planning_horizon, observation_length]
            explore: whether to use exploration reward i.e.
                    expected variance of trajectories.
        Returns:
            expected_value: numpy array of shape [action_size]
        """
        trajectories = trajectories.cpu().numpy()

        # Occupancy terms
        occupancy_bools = trajectories[..., self.occupancy_idxs] > 0
        temperatures = np.where(
            trajectories[..., self.temperature_idxs],
            occupancy_bools,
            self.temp_range_comfort[0],
        )
        air_qualities = np.where(
            trajectories[..., self.air_quality_idxs],
            occupancy_bools,
            self.air_quality_range[0],
        )
        emissions = trajectories[..., self.emissions_idxs]

        reward_comfort = np.mean(
            tolerance(
                temperatures,
                bounds=self.temp_range_comfort,
                margin=self.temperature_margin,
                sigmoid="gaussian",
            )
        )

        reward_air_quality = np.mean(
            tolerance(
                air_qualities,
                bounds=self.air_quality_range,
                margin=250.0,
                sigmoid="gaussian",
            )
        )

        reward_emissions = tolerance(
            emissions,
            bounds=(self.min_emissions, self.min_emissions),
            margin=self.max_emissions,
            sigmoid="linear",
        )

        # --- AGGREGATE REWARD TERM ---
        trajectory_rewards = (
            self.emission_weight * reward_emissions
            + self.air_quality_weight * reward_air_quality
            + self.temperature_weight * reward_comfort
        ) / (
            self.emission_weight + self.air_quality_weight + self.temperature_weight
        )  # [population_size, action_size, planning_horizon]

        particle_rewards = np.mean(
            trajectory_rewards, axis=[-2, -1]
        )  # [population_size, action_size]

        if explore:
            # expected variance of trajectories
            expected_value = np.var(particle_rewards, axis=0)  # [action_size]
        else:
            # expected reward of trajectories
            expected_value = np.mean(particle_rewards, axis=0)

        return expected_value
