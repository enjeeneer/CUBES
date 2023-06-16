"""
Define custom reward functions
"""
from sinergym.utils.rewards import BaseReward
from gym import Env
from typing import Any, Dict, Tuple, Union, List
from datetime import datetime
from cubes.package.variables import get_keyword_from_variable_name_with_keyword


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
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        air_quality_upper_limit=1000,
        emissions_weight: float = 0.33,
        air_quality_weight: float = 0.33,
        lambda_emissions: float = 1.0,
        lambda_temperature: float = 1.0,
        lambda_air_quality: float = 1.0,
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

        # Name of the variables
        self.temp_name = temperature_variable
        self.emissions_name = emissions_variable
        self.air_quality_name = air_quality_variable
        self.occupancy_name = occupancy_variable

        self.zone_names = []
        for name in self.occupancy_name:
            self.zone_names.append(get_keyword_from_variable_name_with_keyword(name))

        # Reward parameters
        self.range_comfort_winter = temp_range_comfort_winter
        self.range_comfort_summer = temp_range_comfort_summer
        self.air_quality_upper_limit = air_quality_upper_limit
        self.w_emissions = emissions_weight
        self.w_air_quality = air_quality_weight
        self.lambda_emissions = lambda_emissions
        self.lambda_temp = lambda_temperature
        self.lambda_air_quality = lambda_air_quality

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

        # Emissions term
        reward_emissions = -self.lambda_emissions * obs_dict[self.emissions_name]

        # Thermal Comfort
        comfort, temps = self._get_comfort(obs_dict)
        reward_comfort = -self.lambda_temp * comfort

        # Air quality
        air_quality, aqs = self._get_air_quality(obs_dict)
        reward_air_quality = -self.lambda_air_quality * air_quality

        # Weighted sum of all terms
        reward = (
            self.w_emissions * reward_emissions
            + self.w_air_quality * reward_air_quality
            + (1.0 - self.w_emissions - self.w_air_quality) * reward_comfort
        )

        reward_terms = {
            "reward_emissions": reward_emissions,
            "emissions": obs_dict[self.emissions_name],
            "reward_comfort": reward_comfort,
            "abs_comfort": comfort,
            "temperatures": temps,
            "reward_air_quality": reward_air_quality,
            "abs_air_quality": air_quality,
            "air_qualities": aqs,
        }

        return reward, reward_terms

    def _get_comfort(self, obs_dict: Dict[str, Any]) -> Tuple[float, List[float]]:
        """Calculate the comfort term of the reward.

        Returns:
            Tuple[float, List[float]]: comfort penalty and List with temperatures used.
        """

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

        # get zone occupancy weights and temperatures
        occs = []
        temps = []
        for k, v in obs_dict.items():
            if k in self.temp_name:
                zone_name = get_keyword_from_variable_name_with_keyword(k)
                temps.append(v)
                for k2, v2 in obs_dict.items():
                    if k2 in self.occupancy_name:
                        if get_keyword_from_variable_name_with_keyword(k2) == zone_name:
                            occs.append(v2)

        comfort = 0.0
        for o, t in zip(occs, temps):
            if t < temp_range[0] or t > temp_range[1]:
                comfort += o * min(abs(temp_range[0] - t), abs(t - temp_range[1]))

        return comfort, temps

    def _get_air_quality(self, obs_dict: Dict[str, Any]) -> Tuple[float, List[float]]:
        """Calculate the air quality term of the reward.

        Returns:
            Tuple[float, List[float]]: air quality penalty
                                    and List with air qualities used.
        """

        # get zone occupancy weights and air qualitues
        aqs = []
        occs = []
        for k, v in obs_dict.items():
            if k in self.air_quality_name:
                zone_name = get_keyword_from_variable_name_with_keyword(k)
                aqs.append(v)
                for k2, v2 in obs_dict.items():
                    if k2 in self.occupancy_name:
                        if get_keyword_from_variable_name_with_keyword(k2) == zone_name:
                            occs.append(v2)

        comfort = 0.0
        for o, aq in zip(occs, aqs):
            if aq > self.air_quality_upper_limit:
                comfort += o * (aq - self.air_quality_upper_limit)

        return comfort, aqs
