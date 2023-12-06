"""Main module to package up IDF files with weather etc and create a gym environment"""
from cubes.construct.core import sample_idf
from cubes.construct.buildingconfig import BuildingConfig
from cubes.package import weather, utilities, variables, gym_utilities
from cubes.package.envconfig import EnvConfig
from cubes.cubesgym.utils.rewards import LinearRewardTEAQ, ToleranceRewardTEAQ
from cubes.constants import BASE_DIR
from gym.envs.registration import register

from agents.pearl.reward_function import PEARLRewardFunction

from geomeppy import IDF


def make_test_env():

    # get idf file
    idf, building_config = sample_idf(1)
    test_name = "cubesgym-test-v1"
    envconfig = EnvConfig(files_dir=BASE_DIR / "inputs" / test_name)

    register_environment(test_name, idf, building_config, envconfig)


def register_environment(
    env_name: str, idf: IDF, building_config: BuildingConfig, env_config: EnvConfig
) -> PEARLRewardFunction:
    """
    Registers gym environment, and returns reward function
    for use inside model-based PEARL agent.
    """

    # set run period
    idf = utilities.set_run_period(idf, env_config)

    # get weather file and save it
    idf = weather.get_weather_file_and_adapt_idf(
        idf=idf,
        building_config=building_config,
        env_config=env_config,
    )

    # save rdd file and expand idf file
    idf, heating_system_capacity = utilities.get_rdd_file(
        idf=idf,
        env_config=env_config,
        building_config=building_config,
    )

    # get forecast files
    utilities.get_temperature_forecast_files(
        building_config.weather_file_name,
        env_config.observe_outside_temperature_in_x_hours_forecast,
        env_files_dir=env_config.files_dir,
    )
    max_emissions_factor = utilities.get_grid_carbon_forecast_files(
        building_config.grid_carbon_intensity_file_name,
        env_config.observe_grid_carbon_in_x_hours_forecast,
        env_files_dir=env_config.files_dir,
    )
    utilities.get_comfort_temperature_forecast_files(
        env_config.observe_comfort_temp_in_x_hours_forecast,
        env_files_dir=env_config.files_dir,
        comfort_temp=building_config.heating_setpoint,
        setback_temp=building_config.heating_setback,
    )
    utilities.get_solar_forecast_files(
        building_config.weather_file_name,
        env_config.observe_solar_irradiance_in_x_hours_forecast,
        env_files_dir=env_config.files_dir,
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
    emissions_variable = (
        "Environmental Impact Total CO2 Emissions Carbon Equivalent Mass(Site)"
    )

    idf.save(filename=env_config.files_dir + "/building_model.idf")

    if env_config.reward_function_type == "Linear":
        reward = LinearRewardTEAQ
        reward_kwargs = {
            "temperature_variable": temperature_variable_names,
            "air_quality_variable": air_quality_variable_names,
            "occupancy_variable": occupancy_variable_names,
            "emissions_variable": "Environmental Impact Total CO2 Emissions"
            " Carbon Equivalent Mass(Site)",
            "action_variable": action_variable_names,
            "temp_range_comfort_winter": env_config.temp_range_comfort_winter,
            "temp_range_comfort_summer": env_config.temp_range_comfort_summer,
            "summer_start": env_config.summer_start,
            "summer_final": env_config.summer_final,
            "air_quality_range": env_config.air_quality_range,
            "emissions_weight": env_config.emissions_weight,
            "air_quality_weight": env_config.air_quality_weight,
            "temperature_weight": env_config.temperature_weight,
            "lambda_emissions": env_config.lambda_emissions,
            "lambda_temperature": env_config.lambda_temperature,
            "lambda_air_quality": env_config.lambda_air_quality,
            "negative_emissions_for_export": (env_config.negative_emissions_for_export),
            "timesteps_per_hour": env_config.timesteps_per_hour,
        }
    elif env_config.reward_function_type == "Tolerance":
        reward = ToleranceRewardTEAQ
        reward_kwargs = {
            "temperature_variable": temperature_variable_names,
            "air_quality_variable": air_quality_variable_names,
            "occupancy_variable": occupancy_variable_names,
            "emissions_variable": emissions_variable,
            "action_variable": action_variable_names,
            "temp_range_comfort_winter": env_config.temp_range_comfort_winter,
            "temp_range_comfort_summer": env_config.temp_range_comfort_summer,
            "summer_start": env_config.summer_start,
            "summer_final": env_config.summer_final,
            "air_quality_range": env_config.air_quality_range,
            "emissions_weight": env_config.emissions_weight,
            "air_quality_weight": env_config.air_quality_weight,
            "temperature_weight": env_config.temperature_weight,
            "lambda_emissions": env_config.lambda_emissions,
            "lambda_temperature": env_config.lambda_temperature,
            "lambda_air_quality": env_config.lambda_air_quality,
            "negative_emissions_for_export": (env_config.negative_emissions_for_export),
            "timesteps_per_hour": env_config.timesteps_per_hour,
            "battery_power_rating": building_config.battery_power_rating,
            "heating_system_capacity": heating_system_capacity,
            "max_emissions_factor": max_emissions_factor,
            "heat_pump": ("heat pump" in building_config.heating_water_loop_equipment),
            "battery": env_config.control_battery_charging,
            "temperature_margin": env_config.temperature_margin,
        }
    else:
        print("Unknown reward_function_type " + env_config.reward_function_type)
        return

    # register environment
    register(
        id=env_name,
        entry_point="cubes.cubesgym.envs:EplusEnvCustom",
        kwargs={
            "idf_file": env_config.files_dir + "/building_model.idf",
            "weather_file": env_config.files_dir + "/weather.epw",
            "observation_space": observation_space,
            "observation_variables": observation_variable_names,
            "action_space": action_space,
            "action_variables": action_variable_names,
            "reward": reward,
            "reward_kwargs": reward_kwargs,
            "env_name": env_name,
            "action_remapping": action_remapping,
        },
    )

    # instantiate pearl reward function

    print(
        f"DEBUGGING: CHECKING TEMP VARIABLES"
        "NAMES INSIDE PEARL REWARD FUNCTION ARE IN SAME"
        f"ORDER AS THE TRUE TEMP: {temperature_variable_names}"
    )

    print(
        f"DEBUGGING: CHECKING ACTION VARIABLES"
        "NAMES INSIDE PEARL REWARD FUNCTION ARE IN SAME"
        f"ORDER AS THE TRUE ACTION SPACE: {action_variable_names}"
    )

    print(
        f"DEBUGGING: CHECKING OCCUPANCY VARIABLES"
        " NAMES INSIDE PEARL REWARD FUNCTION ARE IN SAME"
        f" ORDER AS THE TRUE ACTION SPACE: {occupancy_variable_names}"
    )

    pearl_reward_function = PEARLRewardFunction(
        observation_variables=observation_variable_names,
        action_variables=action_variable_names,
        temperature_variables=temperature_variable_names,
        air_quality_variables=air_quality_variable_names,
        occupancy_variables=occupancy_variable_names,
        emissions_variables=[emissions_variable],
        temp_range_comfort=env_config.temp_range_comfort_summer,
        battery_power_rating=building_config.battery_power_rating,
        heating_system_capacity=heating_system_capacity,  # in W
        max_emissions_factor=max_emissions_factor,  # in gCO2e/kWh
        heat_pump=("heat pump" in building_config.heating_water_loop_equipment),
        battery=env_config.control_battery_charging,
        negative_emissions_for_export=(env_config.negative_emissions_for_export),
        timesteps_per_hour=env_config.timesteps_per_hour,
        emissions_weight=env_config.emissions_weight,
        air_quality_weight=env_config.air_quality_weight,
        temperature_weight=env_config.temperature_weight,
        temperature_margin=env_config.temperature_margin,
    )

    return pearl_reward_function
