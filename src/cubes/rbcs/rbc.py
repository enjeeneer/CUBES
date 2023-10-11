"""module for defining rule based controllers"""

from abc import ABC, abstractmethod

t_control_living_name = "Living-Thermostat Dual SP Control-HEATING-EXT"
t_control_bedroom_name = "Bedroom-Thermostat Dual SP Control-HEATING-EXT"
vent_control_living_name = "Living-Ventilation-EXT"
vent_control_bedroom_name = "Bedroom-Ventilation-EXT"

t_out_name = "Site Outdoor Air Drybulb Temperature(Environment)"
t_living_name = "Zone Air Temperature(Living)"
t_bedroom_name = "Zone Air Temperature(Bedroom)"
co2_living_name = "Zone Air CO2 Concentration(Living)"
co2_bedroom_name = "Zone Air CO2 Concentration(Bedroom)"
occ_living_name = "Zone People Occupant Count(Living)"
occ_bedroom_name = "Zone People Occupant Count(Bedroom)"
vent_living_name = "Zone Ventilation Air Change Rate(Living)"
vent_bedroom_name = "Zone Ventilation Air Change Rate(Bedroom)"


class RuleBasedControllerBase(ABC):
    """base class for rule based controllers"""

    def __init__(
        self, action_variable_names, action_ranges, observation_variable_names
    ):
        self.t_control_living_index = -1
        self.t_control_bedroom_index = -1
        self.vent_control_living_name = -1
        self.vent_control_bedroom_index = -1
        for i, avn in enumerate(action_variable_names):
            if avn == t_control_living_name:
                self.t_control_living_index = i
            elif avn == t_control_bedroom_name:
                self.t_control_bedroom_index = i
            elif avn == vent_control_living_name:
                self.vent_control_living_index = i
            elif avn == vent_control_bedroom_name:
                self.vent_control_bedroom_index = i

        self.t_control_living_range = (
            action_ranges.low[self.t_control_living_index],
            action_ranges.high[self.t_control_living_index],
        )
        self.t_control_bedroom_range = (
            action_ranges.low[self.t_control_bedroom_index],
            action_ranges.high[self.t_control_bedroom_index],
        )
        self.vent_control_living_range = (
            action_ranges.low[self.vent_control_living_index],
            action_ranges.high[self.vent_control_living_index],
        )
        self.vent_control_bedroom_range = (
            action_ranges.low[self.vent_control_bedroom_index],
            action_ranges.high[self.vent_control_bedroom_index],
        )
        self.action_range_dict = {
            self.t_control_living_index: self.t_control_living_range,
            self.t_control_bedroom_index: self.t_control_bedroom_range,
            self.vent_control_living_index: self.vent_control_living_range,
            self.vent_control_bedroom_index: self.vent_control_bedroom_range,
        }

        self.t_out_index = -1
        self.t_living_index = -1
        self.t_bedroom_index = -1
        self.co2_living_index = -1
        self.co2_bedroom_index = -1
        self.occupancy_living_index = -1
        self.occupancy_bedroom_index = -1
        self.vent_living_index = -1
        self.vent_bedroom_index = -1

        for i, ovn in enumerate(observation_variable_names):
            if ovn == t_out_name:
                self.t_out_index = i
            elif ovn == t_living_name:
                self.t_living_index = i
            elif ovn == t_bedroom_name:
                self.t_bedroom_index = i
            elif ovn == co2_living_name:
                self.co2_living_index = i
            elif ovn == co2_bedroom_name:
                self.co2_bedroom_index = i
            elif ovn == occ_bedroom_name:
                self.occupancy_bedroom_index = i
            elif ovn == occ_living_name:
                self.occupancy_living_index = i
            elif ovn == vent_living_name:
                self.vent_living_index = i
            elif ovn == vent_bedroom_name:
                self.vent_bedroom_index = i

    @abstractmethod
    def act(self, observations):
        ...

    def normalise_actions(self, real_actions):
        normalised_actions = []
        for i, ra in enumerate(real_actions):
            normalised_actions.append(
                2
                * (ra - self.action_range_dict[i][0])
                / (self.action_range_dict[i][1] - self.action_range_dict[i][0])
                - 1
            )
        return normalised_actions


class TrivialRBC(RuleBasedControllerBase):
    """this controller keeps the temperature constant at comfort level
    vents whenever the co2 values come within 200ppm of max"""

    def __init__(
        self,
        action_variable_names,
        action_ranges,
        observation_variable_names,
        comfort_temp,
        max_co2,
    ):
        super().__init__(
            action_variable_names, action_ranges, observation_variable_names
        )
        self.comfort_temp = comfort_temp
        self.max_co2 = max_co2

    def act(self, observations):
        actions = [0, 0, 0, 0]
        actions[self.t_control_living_index] = self.comfort_temp
        actions[self.t_control_bedroom_index] = self.comfort_temp
        if self.max_co2 - 200 < observations[self.co2_living_index]:
            actions[self.vent_control_living_index] = self.vent_control_living_range[1]
        else:
            actions[self.vent_control_living_index] = self.vent_control_living_range[0]

        if self.max_co2 - 200 < observations[self.co2_bedroom_index]:
            actions[self.vent_control_bedroom_index] = self.vent_control_bedroom_range[
                1
            ]
        else:
            actions[self.vent_control_bedroom_index] = self.vent_control_bedroom_range[
                0
            ]

        return self.normalise_actions(actions)


class AggressiveRBC(RuleBasedControllerBase):
    """this controller sets the temperature constant to comfort level when occ >0
    else down to setback level;
    vents whenever the co2 values come within 200ppm of max"""

    def __init__(
        self,
        action_variable_names,
        action_ranges,
        observation_variable_names,
        comfort_temp,
        setback_temp,
        max_co2,
    ):
        super().__init__(
            action_variable_names, action_ranges, observation_variable_names
        )
        self.comfort_temp = comfort_temp
        self.setback_temp = setback_temp
        self.max_co2 = max_co2

    def act(self, observations):
        actions = [0, 0, 0, 0]
        if observations[self.occupancy_living_index] > 0:
            actions[self.t_control_living_index] = self.comfort_temp + 0.01
        else:
            actions[self.t_control_living_index] = self.setback_temp

        if observations[self.occupancy_bedroom_index] > 0:
            actions[self.t_control_bedroom_index] = self.comfort_temp + 0.01
        else:
            actions[self.t_control_bedroom_index] = self.setback_temp

        if self.max_co2 - 200 < observations[self.co2_living_index]:
            actions[self.vent_control_living_index] = self.vent_control_living_range[1]
        elif 500 > observations[self.co2_living_index]:
            actions[self.vent_control_living_index] = self.vent_control_living_range[0]
        else:
            if observations[self.vent_living_index] > 0:
                actions[
                    self.vent_control_living_index
                ] = self.vent_control_living_range[1]
            else:
                actions[
                    self.vent_control_living_index
                ] = self.vent_control_living_range[0]

        if self.max_co2 - 200 < observations[self.co2_bedroom_index]:
            actions[self.vent_control_bedroom_index] = self.vent_control_bedroom_range[
                1
            ]
        elif 500 > observations[self.co2_bedroom_index]:
            actions[self.vent_control_bedroom_index] = self.vent_control_bedroom_range[
                0
            ]
        else:
            if observations[self.vent_bedroom_index] > 0:
                actions[
                    self.vent_control_bedroom_index
                ] = self.vent_control_bedroom_range[1]
            else:
                actions[
                    self.vent_control_bedroom_index
                ] = self.vent_control_bedroom_range[0]

        return self.normalise_actions(actions)
