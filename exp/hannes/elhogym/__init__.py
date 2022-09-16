import gym
from gym.envs.registration import register
from elhogym.utils.constants import *
from elhogym.utils.rewards import *
from sinergym.utils.constants import *
from sinergym.utils.rewards import *


# ---------------------------------------------------------------------------- #
#                          Custom Environments                         #
# ---------------------------------------------------------------------------- #
register(
    id="Eplus-1storeytest-v1",
    entry_point="elhogym.envs:EplusEnvCustom",
    kwargs={
        "idf_file": "1_storey_test.idf",
        "weather_file": "GBR_ENG_London.Wea.Ctr-St.James.Park.037700_TMYx.2004-2018.epw",
        "observation_space": ONE_STOREY_TEST_OBSERVATION_SPACE,
        "observation_variables": ONE_STOREY_TEST_OBSERVATION_VARIABLES,
        "action_space": ONE_STOREY_TEST_ACTION_SPACE_CONTINUOUS,
        "action_variables": ONE_STOREY_TEST_ACTION_VARIABLES,
        "action_mapping": ONE_STOREY_TEST_ACTION_MAPPING,
        "reward": HCLoadsReward,
        "reward_kwargs": {
            "heating_variable": "Zone Ideal Loads Supply Air Total Heating Energy(Zone-1)",
            "cooling_variable": "Zone Ideal Loads Supply Air Total Cooling Energy(Zone-1)",
        },
        "env_name": "1storeytest-v1",
    },
)
# 0) Demo environment
register(
    id="Eplus-demo-v1",
    entry_point="elhogym.envs:EplusEnvCustom",
    kwargs={
        "idf_file": "5ZoneAutoDXVAV.idf",
        "weather_file": "USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw",
        "observation_space": DEFAULT_5ZONE_OBSERVATION_SPACE,
        "observation_variables": DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        "action_space": DEFAULT_5ZONE_ACTION_SPACE_DISCRETE,
        "action_variables": DEFAULT_5ZONE_ACTION_VARIABLES,
        "action_mapping": DEFAULT_5ZONE_ACTION_MAPPING,
        "reward": LinearReward,
        "reward_kwargs": {
            "temperature_variable": "Zone Air Temperature(SPACE1-1)",
            "energy_variable": "Facility Total HVAC Electricity Demand Rate(Whole Building)",
            "range_comfort_winter": (20.0, 23.5),
            "range_comfort_summer": (23.0, 26.0),
        },
        "env_name": "demo-v1",
        "action_definition": DEFAULT_5ZONE_ACTION_DEFINITION,
    },
)
