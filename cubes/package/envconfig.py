"""
Module containing the configuration of gym environments.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class EnvConfig:
    """holds the parameters defining the interface between controller and building"""

    # path to dir where sim files are stored
    files_dir: str

    # zone air
    observe_zone_temperature: bool = True
    observe_zone_humidity: bool = False
    observe_zone_co2: bool = False
    observe_comfort_temp_in_x_hours_forecast: List[int] = None

    # systems + devices
    observe_zone_thermostat_setpoints: bool = False
    observe_zone_ventilation: bool = False
    observe_electricity_demand: bool = True
    observe_net_purchased_electricity: bool = False
    observe_total_purchased_electricity: bool = False
    observe_total_surplus_electricity: bool = False
    observe_surplus_electricity: bool = False
    observe_co2_emissions: bool = True
    observe_fuel_demand: bool = False
    observe_battery_charge: bool = False
    observe_battery_charging: bool = False
    observe_pv_power: bool = False

    # outside
    observe_solar_irradiance: bool = False
    observe_solar_irradiance_in_x_hours_forecast: List[int] = None
    observe_outside_humidity: bool = False
    observe_outside_pressure: bool = False
    observe_outside_temperature: bool = True
    observe_wind_speed: bool = False
    observe_wind_direction: bool = False
    observe_rain: bool = False

    observe_outside_temperature_in_x_hours_forecast: List[int] = None
    # observe_outside_humidity_in_x_hours_forecast: List[int] = None

    # people
    observe_thermal_comfort: bool = False
    observe_zone_occupancy: bool = False

    # grid
    observe_grid_carbon_intensity: bool = False
    observe_grid_carbon_in_x_hours_forecast: List[int] = None

    # action variables
    control_thermostat_setpoints: bool = False
    control_battery_charging: bool = False
    battery_storage_operation: str = "DemandLevelling"
    control_ventilation: bool = False
    control_water_loop_temperature: bool = False
    map_t_setpoints_to_comfort_space: bool = False
    discrete_window_actions: bool = False
    discrete_battery_actions: bool = False
    incremental_actions: bool = False
    # control_lights: bool = False
    # control_shades: bool = False

    # episode length
    episode_start_date: Tuple[int, int] = (1, 1)
    episode_end_date: Tuple[int, int] = (31, 12)
    timesteps_per_hour: int = 6

    # reward
    reward_function_type: str = "Tolerance"
    temp_range_comfort_winter: Tuple[int, int] = (20, np.inf)
    temp_range_comfort_summer: Tuple[int, int] = (20, np.inf)
    summer_start: Tuple[int, int] = (6, 1)
    summer_final: Tuple[int, int] = (9, 30)
    sleep_hours: Tuple[int, int] = (23, 6)
    air_quality_range = (0, 800)
    emissions_weight: float = 1.0
    air_quality_weight: float = 1.0
    temperature_weight: float = 1.0
    lambda_emissions: float = 30.0  # 1/(1kW*0.2kg/kWh*1/6h)
    lambda_temperature: float = 1.0 # 1/1K
    lambda_air_quality: float = 0.01 # 1/100ppm
    temperature_margin: float = 3.0
    air_quality_margin: float = 250.0
    negative_emissions_for_export: bool = False
    emissions_reward_avg_n_timesteps: int = 6
    thermal_comfort_bonus: float = 0. #1.,#10., #1
    thermal_comfort_constant_penalty: bool = False
