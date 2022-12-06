"""Main module to package up IDF files with weather etc and create a gym environment"""
from cubes.construct.core import sample_idf
from cubes.package import weather, utilities, variables
from cubes.package.envconfig import EnvConfig
from cubes.cubesgym.utils.rewards import HCLoadsReward
from gym.envs.registration import register


def make_test_env():

    # get idf file
    idf = sample_idf()
    # save it somewhere
    # idf.save(filename=constants.idf_file_path)

    # get weather file and save it
    weather.get_weather_file()

    # get rdd file and expand idf file (not yet done)
    utilities.get_rdd_file(idf)

    envconfig = EnvConfig()

    # changes to idf file for agent interface
    idf, action_variables = variables.add_control_variables_to_idf(idf, envconfig)
    action_variable_names = variables.get_variable_names(action_variables)

    # get observation variables
    (
        observation_variable_names,
        observation_variables,
    ) = variables.get_observation_variables(idf, envconfig)

    # define action and observation spaces + rewards
    action_space = variables.get_space(action_variables)
    observation_space = variables.get_space(observation_variables)

    # register environemnt
    register(
        id="Eplus-1storeytest-v2",
        entry_point="cubes.cubesgym.envs:EplusEnvCustom",
        kwargs={
            "idf_file": "1_storey_test.idf",  # put in idf
            "weather_file": "GBR_ENG_London.Wea.Ctr-St.James.Park."
            "037700_TMYx.2004-2018.epw",  # put in epw
            "observation_space": observation_space,
            "observation_variables": observation_variable_names,
            "action_space": action_space,
            "action_variables": action_variable_names,
            "reward": HCLoadsReward,  # change this
            "reward_kwargs": {
                "heating_variable": "Zone Ideal Loads Supply Air "
                "Total Heating Energy(Zone-1 IDEAL LOADS AIR SYSTEM)",
                "cooling_variable": "Zone Ideal Loads Supply Air "
                "Total Cooling Energy(Zone-1 IDEAL LOADS AIR SYSTEM)",
            },
            "env_name": "1storeytest-v2",
        },
    )
