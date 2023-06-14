"""Main module to package up IDF files with weather etc and create a gym environment"""
from cubes.construct.core import sample_idf
from cubes.construct.buildingconfig import BuildingConfig
from cubes.package import weather, utilities, variables, constants, gym_utilities
from cubes.package.envconfig import EnvConfig
from cubes.cubesgym.utils.rewards import LinearRewardTEAQ
from gym.envs.registration import register

from geomeppy import IDF


def make_test_env():

    # get idf file
    idf, building_config = sample_idf(1)
    envconfig = EnvConfig()

    register_environment("cubesgym-test-v1", idf, building_config, envconfig)


def register_environment(
    env_name: str, idf: IDF, building_config: BuildingConfig, env_config: EnvConfig
):
    # get weather file and save it
    idf = weather.get_weather_file_and_adapt_idf(idf, building_config)

    # save rdd file and expand idf file
    idf = utilities.get_rdd_file(idf)

    # changes to idf file for agent interface
    idf, action_variables = variables.add_control_variables_to_idf(idf, env_config)
    action_variable_names = variables.get_variable_names(action_variables)

    idf.save(filename=constants.idf_file_path)

    # get observation variables
    (
        observation_variable_names,
        observation_variables,
        temperature_variable_names,
        occupancy_variable_names,
        air_quality_variable_names,
    ) = variables.get_observation_variables(idf, building_config, env_config)

    # define action and observation spaces + rewards
    action_space = gym_utilities.get_action_space(action_variables, building_config)
    observation_space = gym_utilities.get_observation_space(observation_variables)

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
            "reward": LinearRewardTEAQ,
            "reward_kwargs": {
                "temperature_variable": temperature_variable_names,
                "air_quality_variable": air_quality_variable_names,
                "occupancy_variable": occupancy_variable_names,
                "emissions_variable": "Environmental Impact Total CO2 Emissions"
                " Carbon Equivalent Mass(Site)",
                "temp_range_comfort_winter": (20, 24),
                "temp_range_comfort_summer": (20, 24),
            },
            "env_name": env_name,
        },
    )
