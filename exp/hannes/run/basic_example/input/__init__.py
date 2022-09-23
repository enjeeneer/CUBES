import gym
from gym.envs.registration import register
import numpy as np
from elhogym.utils.rewards import HCLoadsReward


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
    "Zone Ideal Loads Supply Air Total Heating Energy(Zone-1 IDEAL LOADS AIR SYSTEM)",
    "Zone Ideal Loads Supply Air Total Cooling Energy(Zone-1 IDEAL LOADS AIR SYSTEM)",
    "Zone Ventilation Air Change Rate(Zone-1)",
]

ONE_STOREY_TEST_ACTION_VARIABLES = ["Zone1-natVent-rl"]

ONE_STOREY_TEST_OBSERVATION_SPACE = gym.spaces.Box(
    low=-5e6,
    high=5e6,
    shape=(len(ONE_STOREY_TEST_OBSERVATION_VARIABLES) + 4,),
    dtype=np.float32,
)

ONE_STOREY_TEST_ACTION_SPACE_CONTINUOUS = gym.spaces.Box(
    low=np.array([0.0]), high=np.array([1.0]), shape=(1,), dtype=np.float32
)

register(
    id="Eplus-1storeytest-v2",
    entry_point="elhogym.envs:EplusEnvCustom",
    kwargs={
        "idf_file": "1_storey_test.idf",
        "weather_file": "GBR_ENG_London.Wea.Ctr-St.James.Park.037700_TMYx.2004-2018.epw",
        "observation_space": ONE_STOREY_TEST_OBSERVATION_SPACE,
        "observation_variables": ONE_STOREY_TEST_OBSERVATION_VARIABLES,
        "action_space": ONE_STOREY_TEST_ACTION_SPACE_CONTINUOUS,
        "action_variables": ONE_STOREY_TEST_ACTION_VARIABLES,
        "reward": HCLoadsReward,
        "reward_kwargs": {
            "heating_variable": "Zone Ideal Loads Supply Air Total Heating Energy(Zone-1 IDEAL LOADS AIR SYSTEM)",
            "cooling_variable": "Zone Ideal Loads Supply Air Total Cooling Energy(Zone-1 IDEAL LOADS AIR SYSTEM)",
        },
        "env_name": "1storeytest-v2",
    },
)
