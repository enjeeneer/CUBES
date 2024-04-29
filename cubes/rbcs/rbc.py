"""module for defining rule based controllers"""
from typing import List, Dict, Tuple
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
    ComfortTemperature,
    OccupancyControlledTemperature,
    DOca2014ThermostatControl,
    SwitchOnOFF,
)
from cubes.rbcs.battery_control import (
    TrackFacilityElectricDemandStoreExcessOnSite,
    DemandLevelling,
)


class RuleBasedControllerBase(ABC):
    """base class for rule based controllers"""

    def __init__(
        self,
        action_variable_names: List[str],
        action_ranges: List[Tuple[float, float]],
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
        action_ranges: List[Tuple[float, float]],
        observation_variable_names: List[str],
        zone_names: List[str],
        temp_control_names: Dict[str, str],
        temperature_names: Dict[str, str],
        occupancy_variable_names: Dict[str, str],
        electricity_demand_variable_name: str,
        electricity_supply_variable_name: str,
        battery_discharge_variable_name: str,
        battery_charge_variable_name: str,
        battery_state_variable_name: str,
        utility_demand_target_control_name: str,
        control_ventilation: bool,
        control_battery: bool,
        temperature_control_method: str = "constant",
        ventilation_control_method: str = "co2_controlled",
        battery_control_method: str = "excess_storage",
        open_window_co2: float = 800.0,
        close_window_co2: float = 500.0,
        comfort_temp_setpoint: float = 20.0,
        setback_temp_setpoint: float = 15.0,
        battery_capacity: float = 8.0,
        charging_power: float = 4000.0,
        user_type_vent: str = "random",
        user_type_temp: str = "random",
        t_switch_onoff_times="random",
        sleep_hours: Tuple[int, int] = (23, 6),
    ):
        super().__init__(
            action_variable_names, action_ranges, observation_variable_names
        )

        if control_ventilation:
            if ventilation_control_method == "co2_controlled":
                self.ventilation_controller = CO2ControlledVentilation(
                    open_window_co2=open_window_co2,
                    close_window_co2=close_window_co2,
                )
            elif ventilation_control_method == "Haldi2017":
                self.ventilation_controller = VentilationRateHaldi2017Denmark()
            elif ventilation_control_method == "Jones2017":
                self.ventilation_controller = VentilationRateJones2017(
                    temperature_names=temperature_names
                )
            elif ventilation_control_method == "Rouleau2020":
                self.ventilation_controller = VentilationRateRouleau2020()
            elif ventilation_control_method == "DOca2014":
                self.ventilation_controller = DOca2014VentilationRate(user_type_vent)
            else:
                if ventilation_control_method:
                    print(
                        "no ventilation controller option named "
                        + ventilation_control_method
                    )
                self.ventilation_controller = None
        else:
            self.ventilation_controller = None

        if temperature_control_method == "constant":
            self.temperature_controller = ConstantTemperature(
                temp_setpoint=comfort_temp_setpoint,
                zone_names=zone_names,
                temp_control_names=temp_control_names,
            )
        elif temperature_control_method == "comfort":
            self.temperature_controller = ComfortTemperature(
                comfort_temp_setpoint, setback_temp_setpoint, sleep_hours
            )
        elif temperature_control_method == "switch_onoff":
            self.temperature_controller = SwitchOnOFF(
                comfort_temp_setpoint, setback_temp_setpoint, t_switch_onoff_times
            )

        elif temperature_control_method == "occupancy":
            self.temperature_controller = OccupancyControlledTemperature(
                zone_names=zone_names,
                temp_control_names=temp_control_names,
                occupancy_variable_names=occupancy_variable_names,
                comfort_temp=comfort_temp_setpoint,
                setback_temp=setback_temp_setpoint,
                sleep_hours=sleep_hours,
            )
        elif temperature_control_method == "DOca2014":
            self.temperature_controller = DOca2014ThermostatControl(user_type_temp)

        else:
            print(
                "no temperature controller option named " + temperature_control_method
            )
            self.temperature_controller = None

        if control_battery:
            if battery_control_method == "excess_storage":
                self.battery_controller = TrackFacilityElectricDemandStoreExcessOnSite(
                    battery_capacity=battery_capacity,
                    charging_power=charging_power,
                    electricity_demand_variable_name=electricity_demand_variable_name,
                    electricity_supply_variable_name=electricity_supply_variable_name,
                    battery_discharge_variable_name=battery_discharge_variable_name,
                    battery_charge_variable_name=battery_charge_variable_name,
                    battery_state_variable_name=battery_state_variable_name,
                )
            elif battery_control_method == "demand_levelling":
                self.battery_controller = DemandLevelling(
                    utility_demand_target_control_name=utility_demand_target_control_name  # pylint: disable=line-too-long
                )
            else:
                print("no battery controller option named " + battery_control_method)
                self.battery_controller = None
        else:
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

        print("+" + "-" * (60) + "+")
        print(obs_dict)
        print("+" + "-" * (60) + "+")
        print(action_dict)
        print("+" + "-" * (60) + "+")
        print(action_values)

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
