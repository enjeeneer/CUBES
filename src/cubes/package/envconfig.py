"""
Module containing the configuration of gym environments.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class EnvConfig:
    """holds the parameters defining the interface between controller and building"""

    # observation variables

    # zone air
    observe_zone_temperature: bool = True
    observe_zone_humidity: bool = True
    observe_zone_co2: bool = False

    # systems + devices
    observe_zone_thermostat_setpoints: bool = False
    # observe_heat_pump_air_flow_rate: bool = False
    observe_electricity_demand: bool = True
    observe_co2_emissions: bool = True
    # observe_fuel_demand: bool = False
    observe_battery_charge: bool = False
    observe_pv_power: bool = False

    # outside
    observe_solar_irradiance: bool = False
    observe_outside_humidity: bool = False
    observe_outside_pressure: bool = False
    observe_outside_temperature: bool = True
    observe_wind_speed: bool = False
    observe_wind_direction: bool = False

    observe_outside_temperature_in_x_hours_forecast: List[int] = None
    # observe_outside_humidity_in_x_hours_forecast: List[int] = None

    # people
    observe_thermal_comfort: bool = False
    observe_zone_occupancy: bool = False

    # grid
    observe_grid_carbon_intensity: bool = False
    # observe_grid_carbon_in_x_hours_forecast: List[int] = None

    # action variables
    control_thermostat_setpoints: bool = False
    control_battery_charging: bool = False
    control_ventilation: bool = False
    # control_lights: bool = False
    # control_shades: bool = False

    # reward
