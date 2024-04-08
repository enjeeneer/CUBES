"""Main module to package up IDF files with weather etc and create a gym environment"""
from cubes.construct.core import sample_idf
from cubes.construct.buildingconfig import BuildingConfig
from cubes.package import weather, utilities, variables, gym_utilities
from cubes.package.envconfig import EnvConfig
from cubes.cubesgym.utils.rewards import (
    LinearRewardTEAQ,
    ToleranceRewardTEAQ,
    LinearRewardTEAQCOST,
)
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
        sleep_hours=env_config.sleep_hours,
    )
    utilities.get_solar_forecast_files(
        building_config.weather_file_name,
        env_config.observe_solar_irradiance_in_x_hours_forecast,
        env_files_dir=env_config.files_dir,
    )
    utilities.get_gas_price_forecast_files(
        building_config.gas_pricing_file_name,
        env_config.observe_gas_price_in_x_hours_forecast,
        env_files_dir=env_config.files_dir,
    )
    utilities.get_electricity_price_forecast_files(
        building_config.electricity_pricing_file_name,
        env_config.observe_electricity_price_in_x_hours_forecast,
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
        temperature_sepoint_variable_names,
        occupancy_variable_names,
        air_quality_variable_names,
    ) = variables.get_observation_variables(idf, building_config, env_config)

    # get action remapping dictionary
    action_remapping = variables.get_action_remapping(
        idf,
        action_variable_names,
        observation_variable_names,
        building_config,
        env_config,
    )
    # get action discretization dictionary
    action_discretization = variables.get_action_discretization(
        action_variable_names,
        env_config,
    )

    # get action discretization dictionary
    incremental_action = variables.get_incremental_action(
        idf,
        action_variable_names,
        observation_variable_names,
        building_config,
        env_config,
    )

    emissions_variable = (
        "Environmental Impact Total CO2 Emissions Carbon Equivalent Mass(Site)"
    )
    grid_carbon_variable = "Schedule Value(Grid Carbon Intensity Schedule)"

    gas_cost_variable = "Schedule Value(Gas Pricing Schedule)"

    electricity_cost_variable = "Schedule Value(Electricity Pricing Schedule)"

    # get building specifc bounds

    building_specific_bounds = gym_utilities.get_building_specific_bounds(
        emissions_variable=emissions_variable,
        grid_carbon_variable=grid_carbon_variable,
        electricty_purchased_variable=(
            "Facility Net Purchased Electricity Rate(Whole Building)"
        ),
        electricity_demand_variable=(
            "Facility Total Electricity Demand Rate(Whole Building)"
        ),
        heating_system_capacity=heating_system_capacity,
        battery_power_rating=building_config.battery_power_rating,
        max_emissions_factor=max_emissions_factor,
        timesteps_per_hour=env_config.timesteps_per_hour,
        battery=env_config.control_battery_charging,
        heat_pump="heat pump" in building_config.heating_water_loop_equipment,
        negative_emissions_for_export=env_config.negative_emissions_for_export,
        number_of_occupants=building_config.occupant_value,
        occupancy_variables=occupancy_variable_names,
    )

    # define action and observation spaces + rewards
    action_space = gym_utilities.get_action_space(action_variables, building_config)

    (observation_space, ordered_obs_variables,) = gym_utilities.get_observation_space(
        var_list=observation_variables,
        building_specific_bounds=building_specific_bounds,
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
            "temperature_setpoint_variable": temperature_sepoint_variable_names,
            "action_variable": action_variable_names,
            "temp_range_comfort_winter": env_config.temp_range_comfort_winter,
            "temp_range_comfort_summer": env_config.temp_range_comfort_summer,
            "summer_start": env_config.summer_start,
            "summer_final": env_config.summer_final,
            "sleep_hours": env_config.sleep_hours,
            "air_quality_range": env_config.air_quality_range,
            "emissions_weight": env_config.emissions_weight,
            "air_quality_weight": env_config.air_quality_weight,
            "temperature_weight": env_config.temperature_weight,
            "lambda_emissions": env_config.lambda_emissions,
            "lambda_temperature": env_config.lambda_temperature,
            "lambda_air_quality": env_config.lambda_air_quality,
            "negative_emissions_for_export": (env_config.negative_emissions_for_export),
            "timesteps_per_hour": env_config.timesteps_per_hour,
            "emissions_reward_avg_n_timesteps": (
                env_config.emissions_reward_avg_n_timesteps
            ),
            "thermal_comfort_bonus": env_config.thermal_comfort_bonus,  # 1.,#10., #1
            "thermal_comfort_constant_penalty": (
                env_config.thermal_comfort_constant_penalty
            ),
            "air_quality_bonus": 0.0,  # 100.,#300. #100
        }
    if env_config.reward_function_type == "LinearCost":
        reward = LinearRewardTEAQCOST
        reward_kwargs = {
            "temperature_variable": temperature_variable_names,
            "air_quality_variable": air_quality_variable_names,
            "occupancy_variable": occupancy_variable_names,
            "emissions_variable": emissions_variable,
            "gas_cost_variable": gas_cost_variable,
            "electricity_cost_variable": electricity_cost_variable,
            "temperature_setpoint_variable": temperature_sepoint_variable_names,
            "action_variable": action_variable_names,
            "temp_range_comfort_winter": env_config.temp_range_comfort_winter,
            "temp_range_comfort_summer": env_config.temp_range_comfort_summer,
            "summer_start": env_config.summer_start,
            "summer_final": env_config.summer_final,
            "sleep_hours": env_config.sleep_hours,
            "air_quality_range": env_config.air_quality_range,
            "emissions_weight": env_config.emissions_weight,
            "cost_weight": env_config.cost_weight,
            "air_quality_weight": env_config.air_quality_weight,
            "temperature_weight": env_config.temperature_weight,
            "lambda_emissions": env_config.lambda_emissions,
            "lambda_cost": env_config.lambda_cost,
            "lambda_temperature": env_config.lambda_temperature,
            "lambda_air_quality": env_config.lambda_air_quality,
            "negative_emissions_for_export": (env_config.negative_emissions_for_export),
            "timesteps_per_hour": env_config.timesteps_per_hour,
            "emissions_reward_avg_n_timesteps": (
                env_config.emissions_reward_avg_n_timesteps
            ),
            "thermal_comfort_bonus": env_config.thermal_comfort_bonus,  # 1.,#10., #1
            "thermal_comfort_constant_penalty": (
                env_config.thermal_comfort_constant_penalty
            ),
            "air_quality_bonus": 0.0,  # 100.,#300. #100
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
            "emissions_bounds": building_specific_bounds[emissions_variable],
            "timesteps_per_hour": env_config.timesteps_per_hour,
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
            "action_discretization": action_discretization,
            "incremental_action": incremental_action,
        },
    )

    # instantiate pearl reward function
    pearl_reward_function = PEARLRewardFunction(
        observation_variables=ordered_obs_variables,
        action_variables=action_variable_names,
        temperature_variables=temperature_variable_names,
        air_quality_variables=air_quality_variable_names,
        occupancy_variables=occupancy_variable_names,
        emissions_variables=[emissions_variable],
        temp_range_comfort=env_config.temp_range_comfort_summer,
        emissions_bounds=building_specific_bounds[emissions_variable],
        timesteps_per_hour=env_config.timesteps_per_hour,
        emissions_weight=env_config.emissions_weight,
        air_quality_weight=env_config.air_quality_weight,
        temperature_weight=env_config.temperature_weight,
        temperature_margin=env_config.temperature_margin,
    )

    return pearl_reward_function
