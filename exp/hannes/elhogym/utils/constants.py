# import pathlib
# import os
import gym
import numpy as np
import pkg_resources

# PKG_DATA_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), "data")
PKG_DATA_PATH = pkg_resources.resource_filename("elhogym", "data/")

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
    "Zone Ideal Loads Supply Air Total Heating Energy(Zone-1 IDEAL LOADS AIR SYSTEM)",
    "Zone Ideal Loads Supply Air Total Cooling Energy(Zone-1 IDEAL LOADS AIR SYSTEM)",
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

ONE_STOREY_TEST_ACTION_MAPPING = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

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
