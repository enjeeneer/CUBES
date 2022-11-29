"""
Module containing the configuration of gym environments.
"""

from dataclasses import dataclass


@dataclass
class EnvConfig:
    """holds the parameters defining the interface between controller and building"""

    # which environments to make
    make_uncontrolled_env: bool
    make_controlled_env: bool

    # observation variables
    observe_temperature: bool
    observe_humidity: bool
    observe_thermostat_setpoint: bool
    observe_heat_pump_air_flow_rate: bool
    observe_occupancy: bool
    observe_hvac_power: bool
    observe_outside_irradiance: bool
    observe_outside_humidity: bool
    observe_outside_pressure: bool
    observe_outside_temperature: bool
    observe_grid_carbon_intensity: bool
    observe_1h_grid_carbon_forecast: bool
    observe_3h_grid_carbon_forecast: bool
    observe_6h_grid_carbon_forecast: bool
    observe_12h_grid_carbon_forecast: bool
    observe_24h_grid_carbon_forecast: bool
    observe_1h_outside_temperature_forecast: bool
    observe_24h_outside_temperature_forecast: bool
    observe_1h_outside_humidity_forecast: bool
    observe_24h_outside_humidity_forecast: bool

    # action variables
    control_zone_thermostat_setpoints: bool
    control_heat_pump_flow_rate: bool
    control_heating_system_actuation: bool
    control_ventilation: bool
    control_lights: bool
    control_shades: bool
