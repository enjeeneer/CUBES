"""
Module containing the configuration of gym environments.
"""

from dataclasses import dataclass


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
    observe_heat_pump_air_flow_rate: bool = False
    observe_electricity_demand: bool = True
    observe_co2_emissions: bool = True

    # outside
    observe_solar_irradiance: bool = False
    observe_outside_humidity: bool = False
    observe_outside_pressure: bool = False
    observe_outside_temperature: bool = True
    observe_wind_speed: bool = False
    observe_wind_direction: bool = False

    observe_1h_outside_temperature_forecast: bool = False
    observe_24h_outside_temperature_forecast: bool = False
    observe_1h_outside_humidity_forecast: bool = False
    observe_24h_outside_humidity_forecast: bool = False

    # people
    observe_thermal_comfort: bool = False
    observe_zone_occupancy: bool = False

    # grid
    observe_grid_carbon_intensity: bool = False
    observe_1h_grid_carbon_forecast: bool = False
    observe_3h_grid_carbon_forecast: bool = False
    observe_6h_grid_carbon_forecast: bool = False
    observe_12h_grid_carbon_forecast: bool = False
    observe_24h_grid_carbon_forecast: bool = False

    # action variables
    control_thermostat_setpoints: bool = False
    control_heat_pump_flow_rate: bool = False
    control_heating_system_actuation: bool = False
    control_ventilation: bool = False
    control_lights: bool = False
    control_shades: bool = False

    # reward
