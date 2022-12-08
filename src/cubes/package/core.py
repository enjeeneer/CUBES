"""Main module to package up IDF files with weather etc and create a gym environment"""
from cubes.construct.core import sample_idf
from cubes.package import weather, utilities, variables, constants
from cubes.package.envconfig import EnvConfig
from cubes.constants import EPLUS_PATH
from sinergym.utils.rewards import LinearReward
from gym.envs.registration import register
from geomeppy import IDF


def make_test_env():

    # get idf file
    idf = sample_idf()
    # save it somewhere
    # idf.save(filename=constants.idf_file_path)

    # get weather file and save it
    weather.get_weather_file()

    # get rdd file and expand idf file
    utilities.get_rdd_and_expand_idf(idf)

    # load expanded idf file
    IDF.setiddname(EPLUS_PATH + "Energy+.idd")
    idf = IDF(constants.idf_file_path)

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
    action_space = variables.get_space(action_variables)
    observation_space = variables.get_space(observation_variables)

    # register environemnt
    register(
        id="cubesgym-test-v1",
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
                "energy_variable": "Environmental Impact Total CO2 Emissions Carbon "
                "Equivalent Mass(Whole Building)",
                "range_comfort_winter": (20, 24),
                "range_comfort_summer": (20, 24),
            },
            "env_name": "cubesgym-test-v1",
        },
    )
