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
    name: str
    eplus_class: str
    dimension: str


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
                Schedule_Type_Limits_Name="Any Number",
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
                        Schedule_Type_Limits_Name="Any Number",
                        Initial_Value=0.0,
                    )
                    se.Schedule_Name = schedule_name

                    action_variables.append(Variable(schedule_name, obj, "Temperature"))

        setpoint_entries = idf.idfobjects["THERMOSTATSETPOINT:DUALSETPOINT"]
        for se in setpoint_entries:
            heating_schedule_name = se.Name + "-HEATING-EXT"
            cooling_schedule_name = se.Name + "-COOLING-EXT"

            idf.newidfobject(
                "EXTERNALINTERFACE:SCHEDULE",
                Name=heating_schedule_name,
                Schedule_Type_Limits_Name="Any Number",
                Initial_Value=0.0,
            )
            se.Heating_Setpoint_Temperature_Schedule_Name = heating_schedule_name

            action_variables.append(
                Variable(
                    heating_schedule_name,
                    "THERMOSTATSETPOINT:SINGLEHEATING",
                    "Temperature",
                )
            )

            idf.newidfobject(
                "EXTERNALINTERFACE:SCHEDULE",
                Name=cooling_schedule_name,
                Schedule_Type_Limits_Name="Any Number",
                Initial_Value=0.0,
            )
            se.Cooling_Setpoint_Temperature_Schedule_Name = cooling_schedule_name

            action_variables.append(
                Variable(
                    heating_schedule_name,
                    "THERMOSTATSETPOINT:SINGLECOOLING",
                    "Temperature",
                )
            )

    return idf, action_variables


def get_observation_variables(idf, envconfig):
    obs_var_names = []

    if envconfig.observe_outside_temperature:
        obs_var_names.append("Site Outdoor Air Drybulb Temperature(Environment)")

    if envconfig.observe_outside_humidity:
        obs_var_names.append("Site Outdoor Air Relative Humidity(Environment)")

    if envconfig.observe_wind_speed:
        obs_var_names.append("Site Wind Speed(Environment)")

    if envconfig.observe_wind_direction:
        obs_var_names.append("Site Wind Direction(Environment)")

    if envconfig.observe_solar_irradiance:
        obs_var_names.append("Site Diffuse Solar Radiation Rate per Area(Environment)")
        obs_var_names.append("Site Direct Solar Radiation Rate per Area(Environment)")

    if envconfig.observe_co2_emissions:
        obs_var_names.append(
            "Environmental Impact Total CO2 Emissions Carbon "
            "Equivalent Mass(Whole Building)"
        )

    if envconfig.observe_electricity_demand:
        obs_var_names.append(
            "Facility Total HVAC Electricity Demand Rate(Whole Building)"
        )

    idf_zone_names = []
    for zone in idf.idfobjects["ZONE"]:
        idf_zone_names.append(zone.Name)

    if envconfig.observe_zone_temperature:
        for zname in idf_zone_names:
            obs_var_names.append(f"Zone Air Temperature({zname})")

    if envconfig.observe_zone_humidity:
        for zname in idf_zone_names:
            obs_var_names.append(f"Zone Air Relative Humidity({zname})")

    if envconfig.observe_zone_occupancy:
        for zname in idf_zone_names:
            obs_var_names.append(f"Zone People Occupant Count({zname})")

    idf_people_names = []
    for people in idf.idfobjects["PEOPLE"]:
        idf_people_names.append(people.Name)

    if envconfig.observe_thermal_comfort:
        for pn in idf_people_names:
            obs_var_names.append(f"Zone Thermal Comfort Mean Radiant Temperature({pn})")
            obs_var_names.append(f"Zone Air Relative Humidity({pn})")
            obs_var_names.append(f"Zone Thermal Comfort Clothing Value({pn})")
            obs_var_names.append(f"Zone Thermal Comfort Fanger Model PPD({pn})")
            obs_var_names.append(f"People Air Temperature({pn})")

    if envconfig.observe_thermostat_setpoints:
        if (
            idf.idfobjects["THERMOSTATSETPOINT:DUALSETPOINT"]
            or idf.idfobjects["THERMOSTATSETPOINT:SINGLEHEATING"]
        ):
            for zname in idf_zone_names:
                obs_var_names.append(
                    f"Zone Thermostat Heating Setpoint Temperature({zname})"
                )

        if (
            idf.idfobjects["THERMOSTATSETPOINT:DUALSETPOINT"]
            or idf.idfobjects["THERMOSTATSETPOINT:SINGLECOOLING"]
        ):
            for zname in idf_zone_names:
                obs_var_names.append(
                    f"Zone Thermostat Cooling Setpoint Temperature({zname})"
                )

    # get rdd file
    # Extract rdd observation variables names
    rdd_data = pd.read_csv(constants.rdd_file_path, skiprows=1)
    rdd_variables_names = list(
        map(
            lambda name: name.split(" [")[0], rdd_data["Variable Name [Units]"].tolist()
        )
    )

    # check that observation variables are viable
    utilities.check_observation_variables(
        obs_var_names, rdd_variables_names, idf_zone_names
    )

    return obs_var_names


def get_space(var_list):

    return Box(
        low=-1e6 * np.ones(len(var_list)),
        high=1e6 * np.ones(len(var_list)),
        dtype=np.float32,
    )
