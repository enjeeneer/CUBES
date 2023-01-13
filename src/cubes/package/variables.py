"""
Classes to define action and observation variables
"""

from dataclasses import dataclass
import pandas as pd
from cubes.package import constants, utilities
from gym.spaces import Box
import numpy as np


@dataclass
class Variable:
    """This class holds attributes of action and observation variables
    and estimates their ranges"""

    name: str
    description: str
    dimension_or_unit: str

    def get_range(self):
        if self.dimension_or_unit == "C":
            return -40.0, 80.0
        elif self.dimension_or_unit == "%":
            return 0.0, 100.0
        elif self.dimension_or_unit == "m/s":
            return 0.0, 200.0
        elif self.dimension_or_unit == "deg":
            return 0.0, 360.0
        elif self.dimension_or_unit == "W/m2" and "solar" in self.description.lower():
            return 0.0, 1361.0
        elif self.dimension_or_unit == "W":
            return 0.0, 1e8
        elif self.dimension_or_unit == "kg":
            return 0.0, 1e8
        elif self.dimension_or_unit == "":
            return 0.0, 1e6

        return -1e6, 1e6


def get_variable_names(variables):
    return [v.name for v in variables]


def add_control_variables_to_idf(idf, envconfig):
    action_variables = []

    if envconfig.control_ventilation:
        # search through IDF file for ventilation entries
        ventilation_entries = idf.idfobjects["ZONEVENTILATION:DESIGNFLOWRATE"]
        for v in ventilation_entries:
            # add an ExternalInterface:Schedule for each and insert schedule name
            schedule_name = v.Name + "-EXT"
            idf.newidfobject(
                "EXTERNALINTERFACE:SCHEDULE",
                Name=schedule_name,
                Initial_Value=0.0,
            )
            v.Schedule_Name = schedule_name

            action_variables.append(
                Variable(
                    schedule_name,
                    "ZONEVENTILATION:DESIGNFLOWRATE",
                    v.Design_Flow_Rate_Calculation_Method,
                )
            )

    if envconfig.control_thermostat_setpoints:
        objects = [
            "THERMOSTATSETPOINT:SINGLEHEATING",
            "THERMOSTATSETPOINT:SINGLECOOLING",
        ]
        for obj in objects:
            for setpoint_entries in idf.idfobjects[obj]:
                for se in setpoint_entries:
                    schedule_name = se.Name + "-EXT"
                    idf.newidfobject(
                        "EXTERNALINTERFACE:SCHEDULE",
                        Name=schedule_name,
                        Initial_Value=0.0,
                    )
                    se.Schedule_Name = schedule_name

                    action_variables.append(Variable(schedule_name, obj, "C"))

        setpoint_entries = idf.idfobjects["THERMOSTATSETPOINT:DUALSETPOINT"]
        for se in setpoint_entries:
            heating_schedule_name = se.Name + "-HEATING-EXT"
            cooling_schedule_name = se.Name + "-COOLING-EXT"

            idf.newidfobject(
                "EXTERNALINTERFACE:SCHEDULE",
                Name=heating_schedule_name,
                Initial_Value=20.0,
            )
            se.Heating_Setpoint_Temperature_Schedule_Name = heating_schedule_name

            action_variables.append(
                Variable(
                    heating_schedule_name,
                    "THERMOSTATSETPOINT:SINGLEHEATING",
                    "C",
                )
            )

            idf.newidfobject(
                "EXTERNALINTERFACE:SCHEDULE",
                Name=cooling_schedule_name,
                Initial_Value=25.0,
            )
            se.Cooling_Setpoint_Temperature_Schedule_Name = cooling_schedule_name

            action_variables.append(
                Variable(
                    heating_schedule_name,
                    "THERMOSTATSETPOINT:SINGLECOOLING",
                    "C",
                )
            )

    return idf, action_variables


def get_observation_variables(idf, envconfig):
    obs_vars = []
    temp_var_names = []

    if envconfig.observe_outside_temperature:
        name = "Site Outdoor Air Drybulb Temperature(Environment)"
        obs_vars.append(Variable(name, name, "C"))

    if envconfig.observe_outside_humidity:
        name = "Site Outdoor Air Relative Humidity(Environment)"
        obs_vars.append(Variable(name, name, "%"))

    if envconfig.observe_wind_speed:
        name = "Site Wind Speed(Environment)"
        obs_vars.append(Variable(name, name, "m/s"))

    if envconfig.observe_wind_direction:
        name = "Site Wind Direction(Environment)"
        obs_vars.append(Variable(name, name, "deg"))

    if envconfig.observe_solar_irradiance:
        name = "Site Diffuse Solar Radiation Rate per Area(Environment)"
        obs_vars.append(Variable(name, name, "W/m2"))
        name = "Site Direct Solar Radiation Rate per Area(Environment)"
        obs_vars.append(Variable(name, name, "W/m2"))

    if envconfig.observe_co2_emissions:
        name = (
            "Environmental Impact Total CO2 Emissions Carbon "
            "Equivalent Mass(Whole Building)"
        )
        obs_vars.append(Variable(name, name, "kg"))

    if envconfig.observe_electricity_demand:
        name = "Facility Total HVAC Electricity Demand Rate(Whole Building)"
        obs_vars.append(Variable(name, name, "W"))

    idf_zone_names = []
    for zone in idf.idfobjects["ZONE"]:
        idf_zone_names.append(zone.Name)

    if envconfig.observe_zone_temperature:
        for zname in idf_zone_names:
            name = f"Zone Air Temperature({zname})"
            obs_vars.append(Variable(name, name, "C"))
            temp_var_names.append(name)

    if envconfig.observe_zone_humidity:
        for zname in idf_zone_names:
            name = f"Zone Air Relative Humidity({zname})"
            obs_vars.append(Variable(name, name, "%"))

    if envconfig.observe_zone_occupancy:
        for zname in idf_zone_names:
            name = f"Zone People Occupant Count({zname})"
            obs_vars.append(Variable(name, name, ""))

    idf_people_names = []
    for people in idf.idfobjects["PEOPLE"]:
        idf_people_names.append(people.Name)

    if envconfig.observe_thermal_comfort:
        for pn in idf_people_names:
            name = f"Zone Thermal Comfort Mean Radiant Temperature({pn})"
            obs_vars.append(Variable(name, name, "C"))
            name = f"Zone Air Relative Humidity({pn})"
            obs_vars.append(Variable(name, name, "%"))
            name = f"Zone Thermal Comfort Clothing Value({pn})"
            obs_vars.append(Variable(name, name, ""))
            name = f"Zone Thermal Comfort Fanger Model PPD({pn})"
            obs_vars.append(Variable(name, name, ""))
            name = f"People Air Temperature({pn})"
            obs_vars.append(Variable(name, name, "C"))

    if envconfig.observe_zone_thermostat_setpoints:
        if (
            idf.idfobjects["THERMOSTATSETPOINT:DUALSETPOINT"]
            or idf.idfobjects["THERMOSTATSETPOINT:SINGLEHEATING"]
        ):
            for zname in idf_zone_names:
                name = f"Zone Thermostat Heating Setpoint Temperature({zname})"
                obs_vars.append(Variable(name, name, "C"))

        if (
            idf.idfobjects["THERMOSTATSETPOINT:DUALSETPOINT"]
            or idf.idfobjects["THERMOSTATSETPOINT:SINGLECOOLING"]
        ):
            for zname in idf_zone_names:
                name = f"Zone Thermostat Cooling Setpoint Temperature({zname})"
                obs_vars.append(Variable(name, name, "C"))

    # get rdd file
    # Extract rdd observation variables names
    rdd_data = pd.read_csv(constants.rdd_file_path, skiprows=1)
    rdd_variables_names = list(
        map(
            lambda name: name.split(" [")[0], rdd_data["Variable Name [Units]"].tolist()
        )
    )

    obs_var_names = get_variable_names(obs_vars)

    # check that observation variables are viable
    utilities.check_observation_variables(
        obs_var_names, rdd_variables_names, idf_zone_names
    )

    return obs_var_names, obs_vars, temp_var_names


def get_space(var_list, is_observation_space):
    if is_observation_space:
        lower_limits = np.zeros(len(var_list) + 4)  # sinergym adds time info
        upper_limits = np.zeros(len(var_list) + 4)

        lower_limits[0:4] = [0, 0, 0, 0]
        upper_limits[0:4] = [3000, 12, 31, 24]

        for iv, v in enumerate(var_list):
            lower_limits[iv + 4], upper_limits[iv + 4] = v.get_range()

    else:
        lower_limits = np.zeros(len(var_list))
        upper_limits = np.zeros(len(var_list))

        for iv, v in enumerate(var_list):
            lower_limits[iv], upper_limits[iv] = v.get_range()

    return Box(
        low=lower_limits,
        high=upper_limits,
        dtype=np.float32,
    )
