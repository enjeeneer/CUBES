# pylint: disable=[consider-using-f-string, unused-argument]

"""
Define custom reward functions
"""
from sinergym.utils.rewards import BaseReward
from gym import Env
import warnings
import numpy as np
from typing import Any, Dict, Tuple, Union, List
from datetime import datetime
from cubes.package.variables import get_keyword_from_variable_name_with_keyword
from cubes.constants import NATURAL_GAS_EMISSIONS_FACTOR, MJ_TO_KWH


# The value returned by tolerance() at `margin` distance from `bounds` interval.
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


class ToleranceRewardTEAQ(BaseReward):
    """This class implements a normalized reward function based on
    temperature, emissions, and air quality"""

    def __init__(
        self,
        env: Env,
        temperature_variable: Dict[str, list],
        air_quality_variable: Dict[str, list],
        occupancy_variable: Union[str, list],
        emissions_variable: str,
        action_variable: List[str],
        temp_range_comfort_winter: Tuple[int, int],
        temp_range_comfort_summer: Tuple[int, int],
        battery_power_rating: float,
        heating_system_capacity: float,  # in W
        max_emissions_factor: float,  # in gCO2e/kWh
        heat_pump: bool,
        battery: bool,
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        sleep_hours: Tuple[int, int] = (23, 6),
        lambda_emissions: float = 33.0,
        lambda_temperature: float = 0.1,
        lambda_air_quality: float = 0.01,
        negative_emissions_for_export: bool = False,
        timesteps_per_hour: int = 6,
        air_quality_range: Tuple[int, int] = (0, 1000),
        emissions_weight: float = 1.0,
        air_quality_weight: float = 1.0,
        temperature_weight: float = 1.0,
    ):
        """
        Tolerance based reward function.
        """
        super().__init__(env)

        # get reward related variables (parts of the observation space
        # the agent can influence)  # TODO: emissions?
        self.temp_name = []
        self.air_quality_name = []

        # here the key is the EPlus zone and value is the variable name
        for key, value in temperature_variable.items():
            for act_var in action_variable:
                if key in act_var and value[0] not in self.temp_name:
                    self.temp_name.append(value[0])

        # here the key is the EPlus zone and value is the variable name
        for key, value in air_quality_variable.items():
            for act_var in action_variable:
                if key in act_var and value[0] not in self.air_quality_name:
                    self.air_quality_name.append(value[0])

        self.emissions_name = emissions_variable
        self.occupancy_name = occupancy_variable

        self.zone_names = []
        for name in self.occupancy_name:
            self.zone_names.append(get_keyword_from_variable_name_with_keyword(name))

        # Reward parameters
        self.range_comfort_winter = temp_range_comfort_winter
        self.range_comfort_summer = temp_range_comfort_summer
        self.sleep_hours = sleep_hours
        self.lambda_emissions = lambda_emissions
        self.lambda_temp = lambda_temperature
        self.lambda_air_quality = lambda_air_quality
        self.negative_emissions_for_export = negative_emissions_for_export
        self.timesteps_per_hour = timesteps_per_hour
        self.air_quality_range = air_quality_range
        self.emission_weight = emissions_weight
        self.air_quality_weight = air_quality_weight
        self.temperature_weight = temperature_weight

        # heating capacity is in W, emissions factor is in gCO2e/kWh
        # convert to kW and kgCO2e/kWh
        heating_system_capacity_kw = heating_system_capacity / 1000  # W -> kW
        max_elec_emissions_factor_kgco2e = (
            max_emissions_factor / 1000
        )  # gCO2e/kWh -> kgCO2e/kWh
        natural_gas_emissions_factor_kgco2e = NATURAL_GAS_EMISSIONS_FACTOR / (
            MJ_TO_KWH * 1000
        )  # g/MJ -> kgCO2e/kWh

        print("heating_system_capacity kw: ", heating_system_capacity_kw)
        print("max elec emissions factor:", max_elec_emissions_factor_kgco2e)
        print("natural gas emissions factor: ", natural_gas_emissions_factor_kgco2e)
        print("battery power rating: ", battery_power_rating)
        print("heat_pump: ", heat_pump)

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

        print("min emissions: ", self.min_emissions)
        print("max emissions: ", self.max_emissions)

        # Summer period
        self.summer_start = summer_start  # (month,day)
        self.summer_final = summer_final  # (month,day)

    def __call__(self) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the scalar reward given system state.

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary
            with their individual components.
        """
        # Current observation
        obs_dict = self.env.obs_dict.copy()
        # Last observation
        old_obs_dict = None
        if self.env.old_obs_dict:
            old_obs_dict = self.env.old_obs_dict.copy()

        # Occupancy terms
        # get zone occupancy booleans from last observation
        occupancy_bools = []
        zones = []
        if old_obs_dict:
            for k, v in old_obs_dict.items():  # pylint: disable=unused-variable
                if k in self.temp_name:
                    zone_name = get_keyword_from_variable_name_with_keyword(k)
                    zones.append(zone_name)
                    for k2, v2 in old_obs_dict.items():
                        if k2 in self.occupancy_name:
                            if (
                                get_keyword_from_variable_name_with_keyword(k2)
                                == zone_name
                            ):
                                occupancy_bools.append(float(v2 > 0))
        # first timestep (hardcode no occupancy)
        else:
            occupancy_bools = [0] * len(self.occupancy_name)

        occupancy_bools = np.array(occupancy_bools)

        # get temp range from date
        month = obs_dict["month"]
        day = obs_dict["day"]
        year = obs_dict["year"]
        current_dt = datetime(year, month, day)

        # Periods
        summer_start_date = datetime(year, self.summer_start[0], self.summer_start[1])
        summer_final_date = datetime(year, self.summer_final[0], self.summer_final[1])

        if summer_start_date <= current_dt <= summer_final_date:
            temp_range = self.range_comfort_summer
        else:
            temp_range = self.range_comfort_winter

        # --- TEMPERATURE ---
        temp_array = self._get_temperatures(
            obs_dict=obs_dict, temp_range=temp_range, occupancy_bools=occupancy_bools
        )

        reward_comfort = np.mean(
            tolerance(
                temp_array,
                bounds=temp_range,
                margin=3.0,
                sigmoid="gaussian",
            )
        )

        # --- AIR QUALITY ---
        air_quality_array = self._get_air_quality(
            obs_dict=obs_dict,
            air_quality_range=self.air_quality_range,
            occupancy_bools=occupancy_bools,
        )

        reward_air_quality = np.mean(
            tolerance(
                air_quality_array,
                bounds=self.air_quality_range,
                margin=250.0,
                sigmoid="gaussian",
            )
        )

        # --- EMISSIONS ---
        reward_emissions = tolerance(
            obs_dict[self.emissions_name],
            bounds=(self.min_emissions, self.min_emissions),
            margin=self.max_emissions,
            sigmoid="linear",
        )

        # --- AGGREGATE REWARD TERM ---
        reward = (
            self.emission_weight * reward_emissions
            + self.air_quality_weight * reward_air_quality
            + self.temperature_weight * reward_comfort
        ) / (self.emission_weight + self.air_quality_weight + self.temperature_weight)

        # --- LOGGING ---
        # temp-related logging terms
        t_out = obs_dict["Site Outdoor Air Drybulb Temperature(Environment)"]
        heating_on = int(
            obs_dict[
                "Environmental Impact Total CO2 Emissions "
                "Carbon Equivalent Mass(Site)"
            ]
            > 1e-8
        )

        temp_violation_bool = {}
        violation_delta_temp = {}
        heating_delta_temp = {}
        heating_beyond_comf_delta_t = {}
        for occupancy, temp, zone in zip(occupancy_bools, temp_array, zones):
            if temp < temp_range[0]:
                temp_violation_bool[zone] = occupancy
                violation_delta_temp[zone] = temp_range[0] - temp

            elif temp > temp_range[1]:
                temp_violation_bool[zone] = occupancy
                violation_delta_temp[zone] = temp - temp_range[1]
            else:
                temp_violation_bool[zone] = 0
                violation_delta_temp[zone] = 0

            heating_delta_temp[zone] = max(0, temp - t_out) * heating_on
            heating_beyond_comf_delta_t[zone] = (
                max(0, temp - temp_range[0]) * heating_on
            )

        # air quality logging
        aq_violations = {}
        violation_delta_aq = {}

        for occupancy, air_quality, zone in zip(
            occupancy_bools, air_quality_array, zones
        ):
            if air_quality > self.air_quality_range[1]:
                aq_violations[zone] = occupancy
                violation_delta_aq[zone] = air_quality - self.air_quality_range[1]

            else:
                aq_violations[zone] = 0
                violation_delta_aq[zone] = 0

        reward_terms = {
            "reward_emissions": reward_emissions,
            "reward_comfort": reward_comfort,
            "reward_air_quality": reward_air_quality,
            "total_reward": reward,
            "emissions": obs_dict[self.emissions_name],
            "temperatures": temp_array,
            "abs_air_quality": air_quality_array,
            "air_qualities": air_quality_array,
            "t_violation": temp_violation_bool,
            "aq_violation": aq_violations,
            "heating_delta_T": heating_delta_temp,
            "heating_beyond_comf_delta_T": heating_beyond_comf_delta_t,
            "violation_delta_T": violation_delta_temp,
            "violation_delta_aq": violation_delta_aq,
        }

        return reward, reward_terms

    def _get_temperatures(
        self,
        obs_dict: Dict[str, Any],
        temp_range: Tuple[int, int],
        occupancy_bools: np.array,
    ) -> np.array:
        """
        Gets the temperatures in each thermal zone. If the occupancy is 0,
        the temperature is forced to be inside the bounds such that the reward
        is not affected by unoccupied zones.
        Args:
            obs_dict: current observation
            temp_range: temperature comfort range
            occupancy_bools: array of occupancy booleans
        Returns:
            temp_array: array of temperatures
        """

        # get zone temperatures from current observation
        temps = []
        for k, v in obs_dict.items():
            if k in self.temp_name:
                temps.append(v)

        temps = np.array(temps)

        # if zone is unoccupied, force temperature to be inside bounds
        temp_array = np.where(occupancy_bools, temps, temp_range[0])

        return temp_array

    def _get_air_quality(
        self,
        obs_dict: Dict[str, Any],
        air_quality_range: Tuple[int, int],
        occupancy_bools: np.array,
    ) -> np.array:
        """
        Gets air quality values in each thermal zone. If the occupancy is 0,
        the air quality is forced to be inside the bounds such that the reward
        is not affected by unoccupied zones.
        Args:
            obs_dict: current observation
            air_quality_range: air quality comfort range
            occupancy_bools: array of occupancy booleans
        Returns:
            air_quality_array: array of air quality values
        """

        # get air qualities from current observation
        air_quality_array = []
        for k, v in obs_dict.items():
            if k in self.air_quality_name:
                air_quality_array.append(v)

        air_quality_array = np.array(air_quality_array)

        air_quality_array = np.where(
            occupancy_bools, air_quality_array, air_quality_range[0]
        )

        return air_quality_array


class LinearRewardTEAQ(BaseReward):
    """This class implements a linear reward function based on
    temperature, emissions, and air quality"""

    def __init__(
        self,
        env: Env,
        temperature_variable: Union[str, list],
        air_quality_variable: Union[str, list],
        occupancy_variable: Union[str, list],
        emissions_variable: str,
        temp_range_comfort_winter: Tuple[int, int],
        temp_range_comfort_summer: Tuple[int, int],
        action_variable: List[str],
        max_emissions_factor: float,
        battery_power_rating: float,
        heating_system_capacity: float,
        heat_pump: bool,
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        sleep_hours: Tuple[int, int] = (23, 6),
        air_quality_range: Tuple[int, int] = (0, 1000),
        emissions_weight: float = 1.0,
        air_quality_weight: float = 1.0,
        temperature_weight: float = 1.0,
        lambda_emissions: float = 33.0,
        lambda_temperature: float = 0.1,
        lambda_air_quality: float = 0.01,
        negative_emissions_for_export: bool = False,
        timesteps_per_hour: int = 6,
    ):
        """
        Linear reward function.

        It considers the total building emissions,
        the absolute difference to temperature comfort for occupied zones,
        and the air quality in occupied zones.

        .. math::
            R = - W_e * lambda_e * emissions
                - W_aq * lambda_aq * air_quality
                - (1 - W_e - W_aq)*lambda_T * (max(T - T_{low}, 0) + max(T_{up} - T, 0))
        """
        super().__init__(env)

        # get reward related variables (parts of the observation space
        # the agent can influence)
        self.temp_name = []
        self.air_quality_name = []

        # here the key is the EPlus zone and value is the variable name
        for key, value in temperature_variable.items():
            for act_var in action_variable:
                if key in act_var and value[0] not in self.temp_name:
                    self.temp_name.append(value[0])

        # here the key is the EPlus zone and value is the variable name
        for key, value in air_quality_variable.items():
            for act_var in action_variable:
                if key in act_var and value[0] not in self.air_quality_name:
                    self.air_quality_name.append(value[0])

        # Name of the variables
        self.emissions_name = emissions_variable
        self.occupancy_name = occupancy_variable

        self.zone_names = []
        for name in self.occupancy_name:
            self.zone_names.append(get_keyword_from_variable_name_with_keyword(name))

        # Reward parameters
        self.range_comfort_winter = temp_range_comfort_winter
        self.range_comfort_summer = temp_range_comfort_summer
        self.sleep_hours = sleep_hours
        self.air_quality_upper_limit = air_quality_range[1]
        self.w_emissions = emissions_weight
        self.w_air_quality = air_quality_weight
        self.w_temperature = temperature_weight
        self.lambda_emissions = lambda_emissions
        self.lambda_temp = lambda_temperature
        self.lambda_air_quality = lambda_air_quality
        self.negative_emissions_for_export = negative_emissions_for_export
        self.timesteps_per_hour = timesteps_per_hour

        # Summer period
        self.summer_start = summer_start  # (month,day)
        self.summer_final = summer_final  # (month,day)

    def __call__(self) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the reward function.

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary
            with their individual components.
        """
        # Current observation
        obs_dict = self.env.obs_dict.copy()
        # Last observation
        old_obs_dict = None
        if self.env.old_obs_dict:
            old_obs_dict = self.env.old_obs_dict.copy()

        # Emissions term
        emissions = self._get_emissions(obs_dict)
        reward_emissions = -self.lambda_emissions * emissions
        # reward_emissions = 0.

        # Thermal Comfort
        (
            comfort,
            temps,
            t_violation,
            heating_delta_t,
            heating_beyond_comf_delta_t,
            violation_delta_t,
        ) = self._get_comfort(obs_dict, old_obs_dict)
        reward_comfort = -self.lambda_temp * comfort

        # Air quality
        air_quality, aqs, aq_violation, violation_delta_aq = self._get_air_quality(
            obs_dict, old_obs_dict
        )
        reward_air_quality = -self.lambda_air_quality * air_quality

        # Weighted sum of all terms
        reward = (
            self.w_emissions * reward_emissions
            + self.w_air_quality * reward_air_quality
            + self.w_temperature * reward_comfort
        )

        reward_terms = {
            "reward_emissions": self.w_emissions * reward_emissions,
            "reward_comfort": self.w_temperature * reward_comfort,
            "reward_air_quality": self.w_air_quality * reward_air_quality,
            "emissions": emissions,
            "abs_comfort": comfort,
            "temperatures": temps,
            "abs_air_quality": air_quality,
            "air_qualities": aqs,
            "t_violation": t_violation,
            "aq_violation": aq_violation,
            "heating_delta_T": heating_delta_t,
            "heating_beyond_comf_delta_T": heating_beyond_comf_delta_t,
            "violation_delta_T": violation_delta_t,
            "violation_delta_aq": violation_delta_aq,
        }

        return reward, reward_terms

    def _get_emissions(
        self,
        obs_dict: Dict[str, Any],
    ) -> Tuple[float, List[float]]:
        """Calculate the emissions term of the reward.

        Returns:
            float: calculated emissions
        """

        emissions = obs_dict[self.emissions_name]
        if self.negative_emissions_for_export:
            emissions -= (
                obs_dict["Facility Total Surplus Electricity Rate(Whole Building)"]
                / 1000
                / self.timesteps_per_hour
                * obs_dict["Schedule Value(Grid Carbon Intensity Schedule)"]
                / 1000
            )

        return emissions

    def _get_comfort(
        self, obs_dict: Dict[str, Any], old_obs_dict: Dict[str, Any]
    ) -> Tuple[float, List[float]]:
        """Calculate the comfort term of the reward.

        Returns:
            Tuple[float, List[float]]: comfort penalty and List with temperatures used.
        """

        month = obs_dict["month"]
        day = obs_dict["day"]
        year = obs_dict["year"]
        current_dt = datetime(year, month, day)

        t_out = obs_dict["Site Outdoor Air Drybulb Temperature(Environment)"]
        heating_on = int(
            obs_dict[
                "Environmental Impact Total CO2 Emissions "
                "Carbon Equivalent Mass(Site)"
            ]
            > 1e-8
        )

        # Periods
        summer_start_date = datetime(year, self.summer_start[0], self.summer_start[1])
        summer_final_date = datetime(year, self.summer_final[0], self.summer_final[1])

        if summer_start_date <= current_dt <= summer_final_date:
            temp_range = self.range_comfort_summer
        else:
            temp_range = self.range_comfort_winter

        # get zone occupancy weights from last observation
        occs = []
        zones = []
        if old_obs_dict:
            hour = old_obs_dict["hour"]
            for k, v in old_obs_dict.items():
                if k in self.temp_name:
                    zone_name = get_keyword_from_variable_name_with_keyword(k)
                    for k2, v2 in old_obs_dict.items():
                        if k2 in self.occupancy_name:
                            if (
                                get_keyword_from_variable_name_with_keyword(k2)
                                == zone_name
                            ):
                                # no need to heat during sleep hours
                                occs.append(
                                    float(
                                        v2 > 0
                                        and self.sleep_hours[1]
                                        <= hour
                                        < self.sleep_hours[0]
                                    )
                                )
                                zones.append(zone_name)
                                # occs.append(v2)

        # get zone temperatures from current observation
        temps = []
        for k, v in obs_dict.items():
            if k in self.temp_name:
                temps.append(v)

        comfort = 0.0
        t_violation = {}
        violation_delta_t = {}
        heating_delta_t = {}
        heating_beyond_comf_delta_t = {}
        for o, t, z in zip(occs, temps, zones):
            supp = 0
            # if o>0:
            #     supp = 10
            if t < temp_range[0]:
                comfort += o * (temp_range[0] - t) + supp
                t_violation[z] = o
                violation_delta_t[z] = o * (temp_range[0] - t)

            elif t > temp_range[1]:
                comfort += o * (t - temp_range[1]) + supp
                t_violation[z] = o
                violation_delta_t[z] = o * (t - temp_range[1])
            else:
                comfort -= supp
                t_violation[z] = 0
                violation_delta_t[z] = 0

            heating_delta_t[z] = max(0, t - t_out) * heating_on
            heating_beyond_comf_delta_t[z] = max(0, t - temp_range[0]) * heating_on

        return (
            comfort,
            temps,
            t_violation,
            heating_delta_t,
            heating_beyond_comf_delta_t,
            violation_delta_t,
        )

    def _get_air_quality(
        self, obs_dict: Dict[str, Any], old_obs_dict: Dict[str, Any]
    ) -> Tuple[float, List[float]]:
        """Calculate the air quality term of the reward.

        Returns:
            Tuple[float, List[float]]: air quality penalty
                                    and List with air qualities used.
        """

        # get zone occupancy weights from last observation
        occs = []
        zones = []
        if old_obs_dict:
            for k, v in old_obs_dict.items():
                if k in self.air_quality_name:
                    zone_name = get_keyword_from_variable_name_with_keyword(k)
                    for k2, v2 in old_obs_dict.items():
                        if k2 in self.occupancy_name:
                            if (
                                get_keyword_from_variable_name_with_keyword(k2)
                                == zone_name
                            ):
                                occs.append(float(v2 > 0))
                                zones.append(zone_name)
                                # occs.append(v2)

        # get air qualities from current observation
        aqs = []
        for k, v in obs_dict.items():
            if k in self.air_quality_name:
                aqs.append(v)

        comfort = 0.0
        aq_violations = {}
        violation_delta_aq = {}

        for o, aq, z in zip(occs, aqs, zones):
            supp = 0
            # if o>0:
            #     supp = 1000
            if aq > self.air_quality_upper_limit:
                comfort += o * (aq - self.air_quality_upper_limit) + supp
                # comfort += 1. * (aq - self.air_quality_upper_limit)
                aq_violations[z] = o
                violation_delta_aq[z] = o * (aq - self.air_quality_upper_limit)

            else:
                comfort -= supp
                aq_violations[z] = 0
                violation_delta_aq[z] = 0

        return comfort, aqs, aq_violations, violation_delta_aq
