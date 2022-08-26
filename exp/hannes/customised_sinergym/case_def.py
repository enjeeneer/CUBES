"""
    Registering custom Energyplus environments
"""

import gym
from gym.envs.registration import register
from gym import Env
from sinergym.utils.rewards import BaseReward

import numpy as np

from typing import Any, Dict, Tuple

# ----------------------------------CUSTOM--------------------------------- #
ONE_STOREY_TEST_OBSERVATION_VARIABLES = [
    "Site Outdoor Air Drybulb Temperature(Environment)",
    "Site Outdoor Air Relative Humidity(Environment)",
    "Site Wind Speed(Environment)",
    "Site Wind Direction(Environment)",
    "Site Diffuse Solar Radiation Rate per Area(Environment)",
    "Site Direct Solar Radiation Rate per Area(Environment)",
    "Zone Thermostat Heating Setpoint Temperature(Zone-1)",
    "Zone Thermostat Cooling Setpoint Temperature(Zone-1)",
    "Zone Air Temperature(Zone-1)",
    "Zone Air Relative Humidity(Zone-1)",
    "Zone People Occupant Count(Zone-1)",
]

ONE_STOREY_TEST_ACTION_VARIABLES = ["Zone1-natVent-rl"]

ONE_STOREY_TEST_OBSERVATION_SPACE = gym.spaces.Box(
    low=-5e6,
    high=5e6,
    shape=(len(ONE_STOREY_TEST_OBSERVATION_VARIABLES) + 4,),
    dtype=np.float32,
)

ONE_STOREY_TEST_ACTION_SPACE_CONTINUOUS = gym.spaces.Box(
    low=np.array([0.0]), high=np.array([5.0]), shape=(1,), dtype=np.float32
)


# ONE_STOREY_TEST_ACTION_DEFINITION = {
#    "ThermostatSetpoint:DualSetpoint": [
#        {
#            "name": "Space1-DualSetP-RL",
#            "heating_name": "Space1-HtgSetP-RL",
#            "cooling_name": "Space1-ClgSetP-RL",
#            "heating_initial_value": 21.0,
#            "cooling_initial_value": 25.0,
#            "zones": ["space1-1"],
#        }
#    ]
# }


class HCLoadsReward(BaseReward):
    """
    make a simple reward function based on sum of heating and cooling loads
    """

    def __init__(self, env: Env, heating_variable: str, cooling_variable: str):
        super(HCLoadsReward, self).__init__(env)

        # Name of the variables
        self.heat_name = heating_variable
        self.cool_name = cooling_variable

    def __call__(self) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the reward function.

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """
        # Current observation
        obs_dict = self.env.obs_dict.copy()

        reward_heating = -obs_dict[self.heating_name]
        reward_cooling = -obs_dict[self.cooling_name]

        # Weighted sum of both terms
        reward = reward_heating + reward_cooling

        reward_terms = {
            "reward_heating": reward_heating,
            "reward_cooling": reward_cooling,
        }

        return reward, reward_terms


# ---------------------------------------------------------------------------- #
#                          Custom Environments                         #
# ---------------------------------------------------------------------------- #
register(
    id="Eplus-1storeytest-v1",
    entry_point="eplus_env_custom:EplusEnvCustom",
    kwargs={
        "idf_file": "1_storey_test.idf",
        "weather_file": "GBR_ENG_London.Wea.Ctr-St.James.Park.037700_TMYx.2004-2018.epw",
        "observation_space": ONE_STOREY_TEST_OBSERVATION_SPACE,
        "observation_variables": ONE_STOREY_TEST_OBSERVATION_VARIABLES,
        "action_space": ONE_STOREY_TEST_ACTION_SPACE_CONTINUOUS,
        "action_variables": ONE_STOREY_TEST_ACTION_VARIABLES,
        "reward": HCLoadsReward,
        "reward_kwargs": {
            "heating_variable": "Zone Ideal Loads Supply Air Total Heating Energy(Zone-1)",
            "cooling_variable": "Zone Ideal Loads Supply Air Total Cooling Energy(Zone-1)",
        },
        "env_name": "1storeytest-v1",
    },
)
