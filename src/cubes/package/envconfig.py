"""
Module containing the configuration of gym environments.
"""

from dataclasses import dataclass


@dataclass
class EnvConfig:
    """holds the parameters defining the interface between controller and building"""

    # observation variables
    observe_temperature: bool = True
    observe_humidity: bool = True
    observe_co2: bool = False
    observe_thermostat_setpoint: bool = True
    observe_heat_pump_air_flow_rate: bool = True
    observe_occupancy: bool = True
    observe_hvac_power: bool = True
    observe_outside_irradiance: bool = True
    observe_outside_humidity: bool = True
    observe_outside_pressure: bool = True
    observe_outside_temperature: bool = True
    observe_grid_carbon_intensity: bool = True
    observe_1h_grid_carbon_forecast: bool = True
    observe_3h_grid_carbon_forecast: bool = True
    observe_6h_grid_carbon_forecast: bool = True
    observe_12h_grid_carbon_forecast: bool = True
    observe_24h_grid_carbon_forecast: bool = True
    observe_1h_outside_temperature_forecast: bool = True
    observe_24h_outside_temperature_forecast: bool = True
    observe_1h_outside_humidity_forecast: bool = True
    observe_24h_outside_humidity_forecast: bool = True

    # action variables
    control_thermostat_setpoints: bool = True
    control_heat_pump_flow_rate: bool = False
    control_heating_system_actuation: bool = False
    control_ventilation: bool = False
    control_lights: bool = False
    control_shades: bool = False
