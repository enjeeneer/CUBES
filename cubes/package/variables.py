"""
Classes to define action and observation variables
"""

from dataclasses import dataclass
import pandas as pd
from cubes.package import constants, utilities
from cubes.package.envconfig import EnvConfig
from cubes.construct.buildingconfig import BuildingConfig
from geomeppy import IDF
from typing import List
import operator


@dataclass
class Variable:
    """This class holds attributes of action and observation variables
    and estimates their ranges"""

    name: str
    keyword: str
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
        elif self.dimension_or_unit == "W/m2" and "solar" in self.name.lower():
            return 0.0, 1361.0
        elif self.dimension_or_unit == "W":
            return 0.0, 1e8
        elif self.dimension_or_unit == "kg":
            return 0.0, 1e8
        elif self.dimension_or_unit == "":
            return 0.0, 1e6
        elif self.dimension_or_unit == "ppm":
            return 0.0, 1e6
        elif self.dimension_or_unit == "fraction":
            return 0.0, 1.0
        elif self.dimension_or_unit == "ach":
            return 0.0, 10.0
        elif self.dimension_or_unit == "0/1":
            return 0.0, 1.0

        return -1e6, 1e6

    def get_action_range(self, building_config: BuildingConfig):
        if self.keyword == "THERMOSTATSETPOINT:SINGLEHEATING":
            return (
                building_config.heating_setback,
                (building_config.heating_setpoint + building_config.cooling_setpoint)
                / 2,
            )
        elif self.keyword == "THERMOSTATSETPOINT:SINGLECOOLING":
            return (
                (building_config.heating_setpoint + building_config.cooling_setpoint)
                / 2,
                building_config.cooling_setback,
            )
        elif self.keyword == "ZONEVENTILATION:DESIGNFLOWRATE":
            return 0.0, 1.0
        else:
            return self.get_range()

    def get_name_with_keyword(self):
        return self.name + "(" + self.keyword + ")"


def get_keyword_from_variable_name_with_keyword(name):
    return name.split("(")[-1].split(")")[0]


def get_variable_names(variables):
    return [v.name for v in variables]


def get_variable_names_with_keywords(variables):
    return [v.get_name_with_keyword() for v in variables]


def add_control_variables_to_idf(
    idf: IDF, building_config: BuildingConfig, envconfig: EnvConfig
):
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
                        Initial_Value=20.0,
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

            if building_config.cooling_system_installed:
                idf.newidfobject(
                    "EXTERNALINTERFACE:SCHEDULE",
                    Name=cooling_schedule_name,
                    Initial_Value=25.0,
                )
                se.Cooling_Setpoint_Temperature_Schedule_Name = cooling_schedule_name

                action_variables.append(
                    Variable(
                        cooling_schedule_name,
                        "THERMOSTATSETPOINT:SINGLECOOLING",
                        "C",
                    )
                )

    if envconfig.control_battery_charging:
        if idf.idfobjects["ELECTRICLOADCENTER:DISTRIBUTION"]:
            elc_dist = idf.idfobjects["ELECTRICLOADCENTER:DISTRIBUTION"][0]
            elc_dist.Storage_Operation_Scheme = "TrackChargeDischargeSchedules"
            elc_dist.Storage_Charge_Power_Fraction_Schedule_Name = (
                "Battery Charge Schedule-EXT"
            )
            elc_dist.Storage_Discharge_Power_Fraction_Schedule_Name = (
                "Battery Discharge Schedule-EXT"
            )

            idf.newidfobject(
                "EXTERNALINTERFACE:SCHEDULE",
                Name="Battery Charge Schedule-EXT",
                Initial_Value=0.0,
            )
            idf.newidfobject(
                "EXTERNALINTERFACE:SCHEDULE",
                Name="Battery Discharge Schedule-EXT",
                Initial_Value=0.0,
            )
            action_variables.append(
                Variable(
                    "Battery Charge Schedule-EXT",
                    "Storage Charge Power Fraction Schedule",
                    "fraction",
                )
            )
            action_variables.append(
                Variable(
                    "Battery Discharge Schedule-EXT",
                    "Storage Discharge Power Fraction Schedule",
                    "fraction",
                )
            )

    return idf, action_variables


def clear_output_variables(idf: IDF):
    variables = idf.idfobjects["OUTPUT:VARIABLE"]
    variables_to_remove = [x for x in variables if x.Key_Value == "*"]
    for var in variables_to_remove:
        variables.remove(var)
    # variables.clear()
    return idf


def add_output_variables_to_idf(idf: IDF, observation_variables: List[Variable]):
    """this is only necessary for cases where sinergym is not used,
    as sinergym adds observation variables automatically"""

    variables = idf.idfobjects["OUTPUT:VARIABLE"]
    var_list = []
    for var in variables:
        var_list.append((var.Key_Value, var.Variable_Name))

    for ov in observation_variables:
        if (ov.keyword, ov.name) not in var_list:
            idf.newidfobject(
                "OUTPUT:VARIABLE",
                Key_Value=ov.keyword,
                Variable_Name=ov.name,
                Reporting_Frequency="Hourly",
            )

    return idf


def remove_duplicate_output_variables(idf: IDF):
    variables = idf.idfobjects["OUTPUT:VARIABLE"]
    variables = list(set(variables))
    return idf


def get_observation_variables(
    idf: IDF, buildingconfig: BuildingConfig, envconfig: EnvConfig
):
    obs_vars = []
    temp_var_names = []
    occ_var_names = []
    aq_var_names = []

    if envconfig.observe_outside_temperature:
        obs_vars.append(
            Variable("Site Outdoor Air Drybulb Temperature", "Environment", "C")
        )

    if envconfig.observe_outside_humidity:
        obs_vars.append(
            Variable("Site Outdoor Air Relative Humidity", "Environment", "%")
        )

    if envconfig.observe_wind_speed:
        obs_vars.append(Variable("Site Wind Speed", "Environment", "m/s"))

    if envconfig.observe_wind_direction:
        obs_vars.append(Variable("Site Wind Direction", "Environment", "deg"))

    if envconfig.observe_solar_irradiance:
        obs_vars.append(
            Variable(
                "Site Diffuse Solar Radiation Rate per Area", "Environment", "W/m2"
            )
        )
        obs_vars.append(
            Variable("Site Direct Solar Radiation Rate per Area", "Environment", "W/m2")
        )

    if envconfig.observe_rain:
        obs_vars.append(Variable("Site Rain Status", "Environment", "0/1"))

    if envconfig.observe_co2_emissions:
        obs_vars.append(
            Variable(
                "Environmental Impact Total CO2 Emissions Carbon Equivalent Mass",
                "Site",
                "kg",
            )
        )

    if envconfig.observe_net_purchased_electricity:
        obs_vars.append(
            Variable("Facility Net Purchased Electricity Rate", "Whole Building", "W")
        )

    if envconfig.observe_electricity_demand:
        obs_vars.append(
            Variable("Facility Total Electricity Demand Rate", "Whole Building", "W")
        )

    if envconfig.observe_surplus_electricity:
        obs_vars.append(
            Variable("Facility Total Surplus Electricity Rate", "Whole Building", "W")
        )

    if envconfig.observe_fuel_demand:
        obs_vars.append(
            Variable("Environmental Impact NaturalGas Source Energy", "Site", "J")
        )

    idf_zone_names = []
    for zone in idf.idfobjects["ZONE"]:
        idf_zone_names.append(zone.Name)

    idf_heated_zone_names = []
    for zone in idf.idfobjects["ZONE"]:
        if (
            zone.Name.upper() == "LOFT" and not buildingconfig.loft_is_heated
        ) or zone.Name.upper() == "SUBFLOOR":
            continue
        idf_heated_zone_names.append(zone.Name)

    if envconfig.observe_zone_temperature:
        for zname in idf_zone_names:
            obs_vars.append(Variable("Zone Air Temperature", zname, "C"))
            temp_var_names.append(obs_vars[-1].get_name_with_keyword())

    if envconfig.observe_zone_humidity:
        for zname in idf_zone_names:
            obs_vars.append(Variable("Zone Air Relative Humidity", zname, "%"))

    if envconfig.observe_zone_co2:
        for zname in idf_zone_names:
            obs_vars.append(Variable("Zone Air CO2 Concentration", zname, "ppm"))
            aq_var_names.append(obs_vars[-1].get_name_with_keyword())

    if envconfig.observe_zone_occupancy:
        for zname in idf_heated_zone_names:
            obs_vars.append(Variable("Zone People Occupant Count", zname, ""))
            occ_var_names.append(obs_vars[-1].get_name_with_keyword())

    idf_people_names = []
    for people in idf.idfobjects["PEOPLE"]:
        idf_people_names.append(people.Name)

    if envconfig.observe_thermal_comfort:
        for pn in idf_people_names:
            obs_vars.append(
                Variable("Zone Thermal Comfort Mean Radiant Temperature", pn, "C")
            )
            obs_vars.append(Variable("Zone Air Relative Humidity", pn, "%"))
            obs_vars.append(Variable("Zone Thermal Comfort Clothing Value", pn, ""))
            obs_vars.append(Variable("Zone Thermal Comfort Fanger Model PPD", pn, ""))
            obs_vars.append(Variable("People Air Temperature", pn, "C"))

    if envconfig.observe_zone_thermostat_setpoints:
        if (
            idf.idfobjects["THERMOSTATSETPOINT:DUALSETPOINT"]
            or idf.idfobjects["THERMOSTATSETPOINT:SINGLEHEATING"]
        ):
            for zname in idf_heated_zone_names:
                obs_vars.append(
                    Variable("Zone Thermostat Heating Setpoint Temperature", zname, "C")
                )

        if (
            idf.idfobjects["THERMOSTATSETPOINT:DUALSETPOINT"]
            or idf.idfobjects["THERMOSTATSETPOINT:SINGLECOOLING"]
        ) and buildingconfig.cooling_system_installed:
            for zname in idf_heated_zone_names:
                obs_vars.append(
                    Variable("Zone Thermostat Cooling Setpoint Temperature", zname, "C")
                )

    if envconfig.observe_zone_ventilation:
        for zname in idf_heated_zone_names:

            obs_vars.append(Variable("Zone Ventilation Air Change Rate", zname, "ach"))

    if envconfig.observe_battery_charge:
        obs_vars.append(
            Variable("Electric Storage Battery Charge State", "SYNERION 24M", "Ah")
        )
    if envconfig.observe_battery_charging:
        obs_vars.append(Variable("Electric Storage Charge Power", "SYNERION 24M", "W"))
        obs_vars.append(
            Variable("Electric Storage Discharge Power", "SYNERION 24M", "W")
        )

    if envconfig.observe_pv_power:
        obs_vars.append(
            Variable("Facility Total Produced Electricity Rate", "Whole Building", "W")
        )
    if envconfig.observe_grid_carbon_intensity:
        obs_vars.append(
            Variable("Schedule Value", "Grid Carbon Intensity Schedule", "kg")
        )

    if envconfig.observe_outside_temperature_in_x_hours_forecast:
        for tfh in envconfig.observe_outside_temperature_in_x_hours_forecast:
            idf.newidfobject(
                "SCHEDULE:FILE",
                Name=str(tfh) + " Hour Temperature Forecast Schedule",
                Schedule_Type_Limits_Name="Any Number",
                File_Name=utilities.get_temperature_forecast_file_path(tfh),
                Column_Number=1,
                Rows_to_Skip_at_Top=0,
                Number_of_Hours_of_Data=8760,
                Minutes_per_Item=10,
                Interpolate_to_Timestep="yes"
            )
            obs_vars.append(
                Variable(
                    "Schedule Value",
                    str(tfh) + " Hour Temperature Forecast Schedule",
                    "C",
                )
            )

    if envconfig.observe_grid_carbon_in_x_hours_forecast:
        for gfh in envconfig.observe_grid_carbon_in_x_hours_forecast:
            idf.newidfobject(
                "SCHEDULE:FILE",
                Name=str(gfh) + " Hour Grid Carbon Forecast Schedule",
                Schedule_Type_Limits_Name="Any Number",
                File_Name=utilities.get_grid_forecast_file_path(gfh),
                Column_Number=1,
                Rows_to_Skip_at_Top=0,
                Number_of_Hours_of_Data=8760,
                Minutes_per_Item=10,
            )
            obs_vars.append(
                Variable(
                    "Schedule Value",
                    str(gfh) + " Hour Grid Carbon Forecast Schedule",
                    "kg",
                )
            )

    # get rdd file
    # Extract rdd observation variables names
    rdd_data = pd.read_csv(constants.rdd_file_path, skiprows=1)
    rdd_variables_names = list(
        map(
            lambda name: name.split(" [")[0], rdd_data["Variable Name [Units]"].tolist()
        )
    )

    obs_var_names = get_variable_names_with_keywords(obs_vars)

    # check that observation variables are viable
    utilities.check_observation_variables(obs_var_names, rdd_variables_names)

    return idf, obs_var_names, obs_vars, temp_var_names, occ_var_names, aq_var_names


def _get_heated_zones(idf: IDF, buildingconfig: BuildingConfig):
    idf_zone_names = []
    for zone in idf.idfobjects["ZONE"]:
        idf_zone_names.append(zone.Name)

    idf_heated_zone_names = []
    for zone in idf.idfobjects["ZONE"]:
        if zone.Name.upper() == "LOFT" and not buildingconfig.loft_is_heated:
            continue
        idf_heated_zone_names.append(zone.Name)
    return idf_heated_zone_names


def get_action_remapping(
    idf: IDF,
    action_variable_names,
    observation_variable_names,
    buildingconfig: BuildingConfig,
    env_config: EnvConfig,
):
    """return a dictionary with the remapped actions as keys and values in the form of
    [[(name of observation1, logic operator1, value1),...],
    mapped lower, mapped upper]"""
    remapping_dict = {}
    if env_config.map_t_setpoints_to_comfort_space:
        for zn in _get_heated_zones(idf, buildingconfig):
            action = ""
            observation = ""
            for avn in action_variable_names:
                if zn.lower() in avn.lower() and "HEATING-EXT" in avn:
                    action = avn
            for ovn in observation_variable_names:
                if zn.lower() in ovn.lower() and "People Occupant Count" in ovn:
                    observation = ovn
            if action and observation:
                remapping_dict[action] = [
                    [(observation,operator.gt,0),
                     ("hour",operator.lt,env_config.sleep_hours[0]),
                     ("hour",operator.ge,env_config.sleep_hours[1])],
                    buildingconfig.heating_setpoint,
                    (buildingconfig.heating_setpoint + buildingconfig.cooling_setpoint)
                    / 2,
                ]
    return remapping_dict
