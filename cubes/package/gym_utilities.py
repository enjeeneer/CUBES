"""Utilites using gym package separate from other utilites,
so that gym does not have to be loaded unnecessarily"""

from gym.spaces import Box
import numpy as np
from cubes.construct.buildingconfig import BuildingConfig
from typing import Dict, List, Union
from cubes.constants import NATURAL_GAS_EMISSIONS_FACTOR, MJ_TO_KWH


def get_observation_space(
    var_list, building_specific_bounds: Dict[str, tuple[float, float]]
):
    lower_limits = np.zeros(len(var_list) + 4)  # sinergym adds time info
    upper_limits = np.zeros(len(var_list) + 4)

    lower_limits[0:4] = [0, 0, 0, 0]
    upper_limits[0:4] = [3000, 12, 31, 24]

    for iv, v in enumerate(var_list):
        print("building specific bounds", v)
        print(v in list(building_specific_bounds.keys()))
        if v in list(building_specific_bounds.keys()):

            lower, upper = building_specific_bounds[v]
            print("lower", lower)
            print("upper", upper)
            lower_limits[iv + 4] = lower
            upper_limits[iv + 4] = upper

        else:
            lower_limits[iv + 4], upper_limits[iv + 4] = v.get_range()

    return Box(
        low=lower_limits,
        high=upper_limits,
        dtype=np.float32,
    )


def get_action_space(var_list, building_config: BuildingConfig):
    lower_limits = np.zeros(len(var_list))
    upper_limits = np.zeros(len(var_list))

    for iv, v in enumerate(var_list):
        lower_limits[iv], upper_limits[iv] = v.get_action_range(building_config)

    return Box(
        low=lower_limits,
        high=upper_limits,
        dtype=np.float32,
    )


def get_building_specific_bounds(
    emissions_variable: str,
    grid_carbon_variable: str,
    electricty_purchased_variable: str,
    electricity_demand_variable: str,
    occupancy_variables: List[str],
    heating_system_capacity: float,
    battery_power_rating: float,
    max_emissions_factor: float,
    timesteps_per_hour: int,
    battery: bool,
    heat_pump: bool,
    negative_emissions_for_export: bool,
    number_of_occupants: float,
) -> Dict[str, tuple[Union[int, float], Union[int, float]]]:
    """
    Gets bounds for building specific variables.
    Args:
        emissions_variable: name of emissions variable
        grid_carbon_variable: name of grid carbon variable
        electricty_purchased_variable: name of electricity purchased variable
        electricity_demand_variable: name of electricity demand variable
        occupancy_variables: list of occupancy variables
        heating_system_capacity: heating system capacity in W
        battery_power_rating: battery power rating in W
        max_emissions_factor: max emissions factor in gCO2e/kWh
        timesteps_per_hour: number of timesteps per hour
        battery: whether battery is installed
        heat_pump: whether heat pump is installed
        negative_emissions_for_export: whether negative
                                    emissions for export are allowed
        number_of_occupants: number of occupants
    Returns:
        building_specific_bounds: dict of min/max bounds
    """

    grid_bounds = get_grid_related_bounds(
        emissions_variable=emissions_variable,
        grid_carbon_variable=grid_carbon_variable,
        electricty_purchased_variable=electricty_purchased_variable,
        electricity_demand_variable=electricity_demand_variable,
        heating_system_capacity=heating_system_capacity,
        battery_power_rating=battery_power_rating,
        max_emissions_factor=max_emissions_factor,
        timesteps_per_hour=timesteps_per_hour,
        battery=battery,
        heat_pump=heat_pump,
        negative_emissions_for_export=negative_emissions_for_export,
    )

    occupant_bounds = get_occupant_bounds(
        number_of_occupants=number_of_occupants,
        occupancy_variables=occupancy_variables,
    )

    building_specific_bounds = {
        **grid_bounds,
        **occupant_bounds,
    }

    return building_specific_bounds


def get_occupant_bounds(
    number_of_occupants: int,
    occupancy_variables: List[str],
) -> Dict[str, tuple[int, int]]:
    """
    Gets bounds for occupancy variables, provided number of occupants.
    Args:
        number_of_occupants: number of occupants
        occupancy_variables: list of occupancy variables
    Returns:
        occupant_bounds: dict of min/max occupant bounds
    """
    occupant_bounds = {}
    for occupancy_variable in occupancy_variables:
        occupant_bounds[occupancy_variable] = (0, int(number_of_occupants))

    return occupant_bounds


def get_grid_related_bounds(
    emissions_variable: str,
    grid_carbon_variable: str,
    electricty_purchased_variable: str,
    electricity_demand_variable: str,
    heating_system_capacity: float,
    battery_power_rating: float,
    max_emissions_factor: float,
    timesteps_per_hour: int,
    battery: bool,
    heat_pump: bool,
    negative_emissions_for_export: bool,
):
    """
    Calculate the min/max emissions bounds for the building.
    Args:
        emissions_variable: name of emissions variable
        grid_carbon_variable: name of grid carbon variable
        electricty_purchased_variable: name of electricity purchased variable
        electricity_demand_variable: name of electricity demand variable

        heating_system_capacity: heating system capacity in W
        battery_power_rating: battery power rating in W
        max_emissions_factor: max emissions factor in gCO2e/kWh
        timesteps_per_hour: number of timesteps per hour
        battery: whether battery is installed
        heat_pump: whether heat pump is installed
        negative_emissions_for_export: whether negative
                                    emissions for export are allowed
    Returns:
        emissions_bounds: dict of min/max emissions bounds
    """

    # heating capacity is in W, emissions factor is in gCO2e/kWh
    # convert to kW and kgCO2e/kWh
    heating_system_capacity_kw = heating_system_capacity / 1000  # W -> kW
    max_elec_emissions_factor_kgco2e = (
        max_emissions_factor / 1000
    )  # gCO2e/kWh -> kgCO2e/kWh
    natural_gas_emissions_factor_kgco2e = NATURAL_GAS_EMISSIONS_FACTOR / (
        MJ_TO_KWH * 1000
    )  # g/MJ -> kgCO2e/kWh

    # calculate min/max emissions bounds
    max_heating_emissions = (
        heating_system_capacity_kw
        * max_elec_emissions_factor_kgco2e
        * (1 / timesteps_per_hour)
        if heat_pump
        else heating_system_capacity_kw
        * (natural_gas_emissions_factor_kgco2e)
        * (1 / timesteps_per_hour)
    )
    if battery:
        battery_power_rating_kw = battery_power_rating / 1000  # W -> kW
        battery_charging_emissions = (
            battery_power_rating_kw
            * max_elec_emissions_factor_kgco2e
            * (1 / timesteps_per_hour)
        )
    else:
        battery_charging_emissions = 0

    # get bounds
    max_emissions = max_heating_emissions + battery_charging_emissions
    max_demand = (
        heating_system_capacity + battery_power_rating
    )  # TODO: check with hannes
    max_purchased = heating_system_capacity + battery_power_rating
    min_purchased = -battery_power_rating if battery else 0
    min_demand = -battery_power_rating if battery else 0

    if negative_emissions_for_export:
        min_emissions = -battery_charging_emissions
    else:
        min_emissions = 0

    grid_bounds = {
        emissions_variable: (min_emissions, max_emissions),
        grid_carbon_variable: (0, max_emissions_factor),
        electricty_purchased_variable: (min_purchased, max_purchased),
        electricity_demand_variable: (min_demand, max_demand),
    }

    return grid_bounds
