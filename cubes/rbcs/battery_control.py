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
    ) -> Dict[str, float]:
        """
        Takes obseravtion and returns battery charge/discharge action.
        Args:
            obs_dict: observation dictionary
        Returns:
            actions: action dictionary
        """
        actions = {}

        # check if supply exceeds demand
        if (
            obs_dict[self.electricity_supply_variable_name]
            > obs_dict[self.electricity_demand_variable_name]
        ):
            actions[self.battery_discharge_variable_name] = 0

            # check if battery is not full, if not
            # charge in proportion to excess supply
            if obs_dict[self.battery_state_variable_name] < self.battery_capacity:
                actions[self.battery_charge_variable_name] = min(
                    1.0,
                    (
                        obs_dict[self.electricity_supply_variable_name]
                        - obs_dict[self.electricity_demand_variable_name]
                    )
                    / self.charging_power,
                )

            # battery is full, cannot charge
            else:
                actions[self.battery_charge_variable_name] = 0

        # if demand exceeds supply discharge in proportion to excess demand
        else:
            actions[self.battery_charge_variable_name] = 0
            actions[self.battery_discharge_variable_name] = min(
                1.0,
                (
                    obs_dict[self.electricity_demand_variable_name]
                    - obs_dict[self.electricity_supply_variable_name]
                )
                / self.charging_power,
            )

        return actions
