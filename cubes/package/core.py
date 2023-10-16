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
    # set run period
    idf = utilities.set_run_period(idf, env_config)

    # get weather file and save it
    idf = weather.get_weather_file_and_adapt_idf(idf, building_config)

    # save rdd file and expand idf file
    idf = utilities.get_rdd_file(idf)

    # get forecast files
    utilities.get_temperature_forecast_files(
        building_config.weather_file_name,
        env_config.observe_outside_temperature_in_x_hours_forecast,
    )
    utilities.get_grid_carbon_forecast_files(
        building_config.grid_carbon_intensity_file_name,
        env_config.observe_grid_carbon_in_x_hours_forecast,
    )

    # changes to idf file for agent interface
    idf, action_variables = variables.add_control_variables_to_idf(
        idf, building_config, env_config
    )
    action_variable_names = variables.get_variable_names(action_variables)

    # get observation variables
    (
        idf,
        observation_variable_names,
        observation_variables,
        temperature_variable_names,
        occupancy_variable_names,
        air_quality_variable_names,
    ) = variables.get_observation_variables(idf, building_config, env_config)

    # define action and observation spaces + rewards
    action_space = gym_utilities.get_action_space(action_variables, building_config)
    observation_space = gym_utilities.get_observation_space(observation_variables)

    # get action remapping dictionary
    action_remapping = variables.get_action_remapping(
        idf,
        action_variable_names,
        observation_variable_names,
        building_config,
        env_config,
    )

    idf.save(filename=constants.idf_file_path)

    # register environment
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
                "action_variable": action_variable_names,
                "temp_range_comfort_winter": env_config.temp_range_comfort_winter,
                "temp_range_comfort_summer": env_config.temp_range_comfort_summer,
                "air_quality_range": env_config.air_quality_range,
                "emissions_weight": env_config.emissions_weight,
                "air_quality_weight": env_config.air_quality_weight,
                "temperature_weight": env_config.temperature_weight,
            },
            "env_name": env_name,
            "action_remapping": action_remapping,
        },
    )
