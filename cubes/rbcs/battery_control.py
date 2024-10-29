# pylint: disable=unused-argument
# pylint: disable=too-many-positional-arguments
# pylint: disable=chained-comparison
"""This module implements several battery control strategies"""
from typing import Dict

from cubes.rbcs.basecontrol import BaseControl


class TrackFacilityElectricDemandStoreExcessOnSite(BaseControl):
    """implements the Energyplus default control strategy.
    Excess generated power is stored onsite if possible
    and used onsitewhen possible"""

    def __init__(
        self,
        battery_capacity: float,
        charging_power: float,
        electricity_demand_variable_name: str,
        electricity_supply_variable_name: str,
        battery_discharge_variable_name: str,
        battery_charge_variable_name: str,  # TODO: roll into one varible in [-1, 1]
        battery_state_variable_name: str,
    ):
        super().__init__()
        self.battery_capacity = battery_capacity
        self.charging_power = charging_power
        self.electricity_demand_variable_name = electricity_demand_variable_name
        self.electricity_supply_variable_name = electricity_supply_variable_name
        self.battery_discharge_variable_name = battery_discharge_variable_name
        self.battery_charge_variable_name = battery_charge_variable_name
        self.battery_state_variable_name = battery_state_variable_name

    def act(
        self,
        obs_dict: Dict[str, float],
        action_dict: Dict[str, float],
        **kwargs,
    ) -> Dict[str, float]:
        """
        Takes observation and returns battery charge/discharge action.
        Args:
            obs_dict: observation dictionary
        Returns:
            action_dict: action dictionary
        """

        # check if supply exceeds demand
        if (
            obs_dict[self.electricity_supply_variable_name]
            > obs_dict[self.electricity_demand_variable_name]
        ):
            action_dict[self.battery_discharge_variable_name] = 0

            # check if battery is not full, if not
            # charge in proportion to excess supply
            if obs_dict[self.battery_state_variable_name] < self.battery_capacity:
                action_dict[self.battery_charge_variable_name] = min(
                    1.0,
                    (
                        obs_dict[self.electricity_supply_variable_name]
                        - obs_dict[self.electricity_demand_variable_name]
                    )
                    / self.charging_power,
                )

            # battery is full, cannot charge
            else:
                action_dict[self.battery_charge_variable_name] = 0

        # if demand exceeds supply discharge in proportion to excess demand
        else:
            action_dict[self.battery_charge_variable_name] = 0
            action_dict[self.battery_discharge_variable_name] = min(
                1.0,
                (
                    obs_dict[self.electricity_demand_variable_name]
                    - obs_dict[self.electricity_supply_variable_name]
                )
                / self.charging_power,
            )

        return action_dict


class DemandLevelling(BaseControl):
    """tries to keep purchased electricity power as low as possible
    by setting demand target to zero at all times"""

    def __init__(self, utility_demand_target_control_name):
        super().__init__()
        self.utility_demand_target_control_name = utility_demand_target_control_name

    def act(
        self,
        obs_dict: Dict[str, float],
        action_dict: Dict[str, float],
        **kwargs,
    ) -> Dict[str, float]:
        """
        Takes obseravtion and returns target utility demand action.
        Args:
            obs_dict: observation dictionary
        Returns:
            action_dict: action dictionary
        """

        action_dict[self.utility_demand_target_control_name] = 1e-6

        return action_dict


class OctopusTimeOfUseBatteryControl(BaseControl):
    """Controls battery based on Octopus's time-of-use periods."""

    def __init__(
        self,
        battery_capacity: float,
        charging_power: float,
        electricity_demand_variable_name: str,
        electricity_supply_variable_name: str,
        battery_discharge_variable_name: str,
        battery_charge_variable_name: str,  # TODO: roll into one varible in [-1, 1]
        battery_state_variable_name: str,
        peak_hours=(16, 19),
        off_peak_hours=(2, 5),
    ):
        super().__init__()
        self.battery_capacity = battery_capacity
        self.charging_power = charging_power
        self.electricity_demand_var = electricity_demand_variable_name
        self.electricity_supply_var = electricity_supply_variable_name
        self.battery_discharge_var = battery_discharge_variable_name
        self.battery_charge_var = battery_charge_variable_name
        self.battery_state_var = battery_state_variable_name
        self.peak_hours = peak_hours
        self.off_peak_hours = off_peak_hours

    def act(
        self,
        obs_dict: Dict[str, float],
        action_dict: Dict[str, float],
        **kwargs,
    ) -> Dict[str, float]:

        hour = obs_dict.get("hour_of_day", 0)
        is_off_peak = self.off_peak_hours[0] <= hour < self.off_peak_hours[1]
        is_peak = self.peak_hours[0] <= hour < self.peak_hours[1]

        battery_state = obs_dict[self.battery_state_var]
        demand = obs_dict[self.electricity_demand_var]
        supply = obs_dict[self.electricity_supply_var]
        excess_power = supply - demand

        # Off-peak: charge if battery not full and supply > demand
        if is_off_peak and battery_state < self.battery_capacity:
            action_dict[self.battery_charge_var] = min(
                1.0, max(0, excess_power) / self.charging_power
            )
            action_dict[self.battery_discharge_var] = 0

        # Peak: discharge if battery has charge and demand > supply
        elif is_peak and battery_state > 0:
            action_dict[self.battery_discharge_var] = min(
                1.0, max(0, demand - supply) / self.charging_power
            )
            action_dict[self.battery_charge_var] = 0

        # Standard hours: balance charge/discharge based on demand/supply
        else:
            if excess_power > 0 and battery_state < self.battery_capacity:
                action_dict[self.battery_charge_var] = min(
                    1.0, excess_power / self.charging_power
                )
                action_dict[self.battery_discharge_var] = 0
            elif excess_power < 0 and battery_state > 0:
                action_dict[self.battery_discharge_var] = min(
                    1.0, -excess_power / self.charging_power
                )
                action_dict[self.battery_charge_var] = 0
            else:
                action_dict[self.battery_charge_var] = 0
                action_dict[self.battery_discharge_var] = 0

        return action_dict
