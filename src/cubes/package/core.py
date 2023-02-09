"""Main module to package up IDF files with weather etc and create a gym environment"""
from cubes.construct.core import sample_idf
from cubes.package import weather, utilities, variables, constants, gym_utilities
from cubes.package.envconfig import EnvConfig
from sinergym.utils.rewards import LinearReward
from gym.envs.registration import register


def make_test_env():

    # get idf file
    idf, building_config = sample_idf()
    # save it somewhere
    # idf.save(filename=constants.idf_file_path)
    building = idf.idfobjects["BUILDING"][0]
    building.Name = "Test Building"

    # get weather file and save it
    idf = weather.get_weather_file_and_adapt_idf(idf, building_config)

    # save rdd file and expand idf file
    idf = utilities.get_rdd_and_expand_idf(idf)

    envconfig = EnvConfig()

    # changes to idf file for agent interface
    idf, action_variables = variables.add_control_variables_to_idf(idf, envconfig)
    action_variable_names = variables.get_variable_names(action_variables)

    idf.save(filename=constants.idf_file_path)

    # get observation variables
    (
        observation_variable_names,
        observation_variables,
        temperature_variable_names,
    ) = variables.get_observation_variables(idf, envconfig)

    # define action and observation spaces + rewards
    action_space = gym_utilities.get_space(action_variables, False)
    observation_space = gym_utilities.get_space(observation_variables, True)

    env_name = "cubesgym-test-v1"
    # register environemnt
    register(
        id=env_name,
        entry_point="cubes.cubesgym.envs:EplusEnvCustom",
        kwargs={
            "idf_file": constants.idf_file_path,
            "weather_file": constants.weather_file_path,
            "observation_space": observation_space,
            "observation_variables": observation_variable_names,
            "action_space": action_space,
            "action_variables": action_variable_names,
            "reward": LinearReward,
            "reward_kwargs": {
                "temperature_variable": temperature_variable_names,
                "energy_variable": "Facility Total HVAC Electricity Demand "
                "Rate(Whole Building)",
                "range_comfort_winter": (20, 24),
                "range_comfort_summer": (20, 24),
            },
            "env_name": env_name,
        },
    )

    return env_name
