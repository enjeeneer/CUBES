"""module for defining rule based controllers"""
from typing import List, Dict
from abc import ABC, abstractmethod
import numpy as np

from cubes.rbcs.ventilation_control import (
    CO2ControlledVentilation,
    VentilationRateHaldi2017Denmark,
    VentilationRateJones2017,
    VentilationRateRouleau2020,
    DOca2014VentilationRate,
)
from cubes.rbcs.temperature_control import (
    ConstantTemperature,
    OccupancyControlledTemperature,
    DOca2014ThermostatControl,
)
from cubes.rbcs.battery_control import TrackFacilityElectricDemandStoreExcessOnSite


class RuleBasedControllerBase(ABC):
    """base class for rule based controllers"""

    def __init__(
        self,
        action_variable_names: List[str],
        action_ranges: Dict[str, float],
        observation_variable_names: List[str],
    ):
        self.observation_variable_names = observation_variable_names
        self.action_variable_names = action_variable_names
        self.action_range_dict = dict(
            zip(action_variable_names, zip(action_ranges.low, action_ranges.high))
        )
        self.action_ranges = [*self.action_range_dict.values()]

    def _get_observation_dict(self, observations):
        return dict(zip(self.observation_variable_names, observations))

    def _get_action_list(self, action_dict: Dict[str, float]):
        return [*action_dict.values()]

    @abstractmethod
    def act(self, observations: Dict[str, float]):
        pass

    def _normalise_actions(self, real_actions: List[float]):

        normalised_actions = []
        for i, ra in enumerate(real_actions):
            normalised_actions.append(
                2
                * (ra - self.action_ranges[i][0])
                / (self.action_ranges[i][1] - self.action_ranges[i][0])
                - 1
            )
        return normalised_actions


class GeneralRBC(RuleBasedControllerBase):
    """this is a generalised controller which can combine various subcontrollers"""

    def __init__(
        self,
        action_variable_names: List[str],
        action_ranges: Dict[str, float],
        observation_variable_names: List[str],
        zone_names: List[str],
        temp_control_names: Dict[str, str],
        occupancy_variable_names: Dict[str, str],
        electricity_demand_variable_name: str,
        electricity_supply_variable_name: str,
        battery_discharge_variable_name: str,
        battery_charge_variable_name: str,
        battery_state_variable_name: str,
        temperature_control="constant",
        ventilation_control="co2_controlled",
        battery_control="",
        open_window_co2=800,
        close_window_co2=500,
        comfort_temp_setpoint=20,
        setback_temp_setpoint=15,
        battery_capacity=8,
        charging_power=4000,
        user_type_vent="random",
        user_type_temp="random",
    ):
        super().__init__(
            action_variable_names, action_ranges, observation_variable_names
        )

        if ventilation_control == "co2_controlled":
            self.ventilation_controller = CO2ControlledVentilation(
                open_window_co2, close_window_co2
            )
        elif ventilation_control == "Haldi2017":
            self.ventilation_controller = VentilationRateHaldi2017Denmark()
        elif ventilation_control == "Jones2017":
            self.ventilation_controller = VentilationRateJones2017()
        elif ventilation_control == "Rouleau2020":
            self.ventilation_controller = VentilationRateRouleau2020()
        elif ventilation_control == "DOca2014":
            self.ventilation_controller = DOca2014VentilationRate(user_type_vent)
        else:
            if ventilation_control:
                print("no ventilation controller option named " + ventilation_control)
            self.ventilation_controller = None

        if temperature_control == "constant":
            self.temperature_controller = ConstantTemperature(
                temp_setpoint=comfort_temp_setpoint,
                zone_names=zone_names,
                temp_control_names=temp_control_names,
            )
        elif temperature_control == "occupancy":
            self.temperature_controller = OccupancyControlledTemperature(
                zone_names=zone_names,
                temp_control_names=temp_control_names,
                occupancy_variable_names=occupancy_variable_names,
                comfort_temp=comfort_temp_setpoint,
                setback_temp=setback_temp_setpoint,
            )
        elif temperature_control == "DOca2014":
            self.temperature_controller = DOca2014ThermostatControl(user_type_temp)

        else:
            if temperature_control:
                print("no temperature controller option named " + temperature_control)
            self.temperature_controller = None

        if battery_control == "excess_storage":
            self.battery_controller = TrackFacilityElectricDemandStoreExcessOnSite(
                battery_capacity=battery_capacity,
                charging_power=charging_power,
                electricity_demand_variable_name=electricity_demand_variable_name,
                electricity_supply_variable_name=electricity_supply_variable_name,
                battery_discharge_variable_name=battery_discharge_variable_name,
                battery_charge_variable_name=battery_charge_variable_name,
                battery_state_variable_name=battery_state_variable_name,
            )
        else:
            if battery_control:
                print("no battery controller option named " + battery_control)
            self.battery_controller = None

    def act(self, observations: np.ndarray):
        """
        Returns temp/ventilation/battery actions
        given observation.
        Args:
            observations: observation array
        Returns:
            actions: normalised action array
        """
        action_dict = dict(
            zip(self.action_variable_names, [0] * len(self.action_variable_names))
        )
        obs_dict = self._get_observation_dict(observations)

        if self.temperature_controller:
            action_dict = self.temperature_controller.act(
                obs_dict=obs_dict, action_dict=action_dict, action_range_dict=None
            )
        if self.ventilation_controller:
            action_dict = self.ventilation_controller.act(
                obs_dict=obs_dict,
                action_dict=action_dict,
                action_range_dict=self.action_range_dict,
            )
        if self.battery_controller:
            action_dict = self.battery_controller.act(
                obs_dict=obs_dict, action_dict=action_dict
            )

        action_values = self._get_action_list(action_dict)

        return self._normalise_actions(action_values)


# class TrivialRBC(RuleBasedControllerBase):
#     """this controller keeps the temperature constant at comfort level
#     vents whenever the co2 values come within 200ppm of max"""

#     def __init__(
#         self,
#         action_variable_names,
#         action_ranges,
#         observation_variable_names,
#         comfort_temp,
#         max_co2,
#     ):
#         super().__init__(
#             action_variable_names, action_ranges, observation_variable_names
#         )
#         self.comfort_temp = comfort_temp
#         self.max_co2 = max_co2

#     def act(self, observations):
#         obs_dict = self._get_observation_dict(observations)
#         #actions = [0, 0, 0, 0]
#         self.action_dict[c.t_control_living_name] = self.comfort_temp
#         self.action_dict[c.t_control_bedroom_name] = self.comfort_temp

#         if self.max_co2 - 200 < obs_dict[c.co2_living_name]:
#             self.action_dict[c.vent_control_living_name] = (
#                 self.action_range_dict[c.vent_control_living_name][1])
#         else:
#             self.action_dict[c.vent_control_living_name] = (
#                 self.action_range_dict[c.vent_control_living_name][0])

#         if self.max_co2 - 200 < obs_dict[c.co2_bedroom_name]:
#             self.action_dict[c.vent_control_bedroom_name] = (
#                 self.action_range_dict[c.vent_control_bedroom_name][1])
#         else:
#             self.action_dict[c.vent_control_bedroom_name] = (
#                 self.action_range_dict[c.vent_control_bedroom_name][0])

#         return self._normalise_actions(self._get_actions())


# class AggressiveRBC(RuleBasedControllerBase):
#     """this controller sets the temperature constant to comfort level when occ >0
#     else down to setback level;
#     vents whenever the co2 values come within 200ppm of max"""

#     def __init__(
#         self,
#         action_variable_names,
#         action_ranges,
#         observation_variable_names,
#         comfort_temp,
#         setback_temp,
#         max_co2,
#     ):
#         super().__init__(
#             action_variable_names, action_ranges, observation_variable_names
#         )
#         self.comfort_temp = comfort_temp
#         self.setback_temp = setback_temp
#         self.max_co2 = max_co2

#     def act(self, observations):
#         obs_dict = self._get_observation_dict(observations)

#         if obs_dict[c.occ_living_name] > 0:
#             self.action_dict[c.t_control_living_name] = self.comfort_temp + 0.01
#         else:
#             self.action_dict[c.t_control_living_name] = self.setback_temp

#         if obs_dict[c.occ_bedroom_name] > 0:
#             self.action_dict[c.t_control_bedroom_name] = self.comfort_temp + 0.01
#         else:
#             self.action_dict[c.t_control_bedroom_name] = self.setback_temp

#         if self.max_co2 < obs_dict[c.co2_living_name]:
#             self.action_dict[c.vent_control_living_name] = (
#                 self.action_range_dict[c.vent_control_living_name][1])
#         elif 500 > obs_dict[c.co2_living_name]:
#             self.action_dict[c.vent_control_living_name] = (
#                 self.action_range_dict[c.vent_control_living_name][0])
#         else:
#             if obs_dict[c.vent_living_name] > 0:
#                 self.action_dict[
#                     c.vent_control_living_name
#                 ] = (
#                 self.action_range_dict[c.vent_control_living_name][1])
#             else:
#                 self.action_dict[
#                     c.vent_control_living_name
#                 ] = (
#                 self.action_range_dict[c.vent_control_living_name][0])

#         if self.max_co2 < obs_dict[c.co2_bedroom_name]:
#             self.action_dict[c.vent_control_bedroom_name] = (
#                 self.action_range_dict[c.vent_control_bedroom_name][1])
#         elif 500 > obs_dict[c.co2_bedroom_name]:
#             self.action_dict[c.vent_control_bedroom_name] = (
#                 self.action_range_dict[c.vent_control_bedroom_name][0])
#         else:
#             if obs_dict[c.vent_bedroom_name] > 0:
#                 self.action_dict[
#                     c.vent_control_bedroom_name
#                 ] = (
#                 self.action_range_dict[c.vent_control_bedroom_name][1])
#             else:
#                 self.action_dict[
#                     c.vent_control_bedroom_name
#                 ] = (
#                 self.action_range_dict[c.vent_control_bedroom_name][0])

#         return self._normalise_actions(self._get_actions())
