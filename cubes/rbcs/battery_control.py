"""This module implements several battery control strategies"""
from cubes.rbcs.basecontrol import BaseControl
import cubes.rbcs.constants as c
from cubes.construct.pv_and_battery import get_battery_ah_from_kwh


class TrackFacilityElectricDemandStoreExcessOnSite(BaseControl):
    """implements the Energyplus default control strategy.
    Excess generated power is stored onsite if possible
    and used onsitewhen possible"""

    def __init__(self, battery_capacity, charging_power):
        super().__init__()
        self.battery_capacity = get_battery_ah_from_kwh(battery_capacity)
        self.charging_power = charging_power

    def act(self, obs_dict, action_dict, action_range_dict):
        if obs_dict[c.produced_electricity_name] > obs_dict[c.electricity_demand_name]:
            action_dict[c.discharge_control_name] = 0
            if obs_dict[c.battery_charging_state_name] < self.battery_capacity:
                action_dict[c.charge_control_name] = min(
                    1,
                    (
                        obs_dict[c.produced_electricity_name]
                        - obs_dict[c.electricity_demand_name]
                    )
                    / self.charging_power,
                )
            else:
                action_dict[c.charge_control_name] = 0
        else:
            action_dict[c.charge_control_name] = 0
            action_dict[c.discharge_control_name] = min(
                1,
                (
                    obs_dict[c.electricity_demand_name]
                    - obs_dict[c.produced_electricity_name]
                )
                / self.charging_power,
            )

        return action_dict
