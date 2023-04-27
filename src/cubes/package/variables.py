"""
Classes to define action and observation variables
"""

from dataclasses import dataclass
import pandas as pd
from cubes.package import constants, utilities
from cubes.package.envconfig import EnvConfig
from cubes.construct.buildingconfig import BuildingConfig
from geomeppy import IDF


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
        elif self.dimension_or_unit == "W/m2" and "solar" in self.description.lower():
            return 0.0, 1361.0
        elif self.dimension_or_unit == "W":
            return 0.0, 1e8
        elif self.dimension_or_unit == "kg":
            return 0.0, 1e8
        elif self.dimension_or_unit == "":
            return 0.0, 1e6

        return -1e6, 1e6

    def get_name_with_keyword(self):
        return self.name + "(" + self.keyword + ")"


def get_variable_names(variables):
    return [v.name for v in variables]


def get_variable_names_with_keywords(variables):
    return [v.get_name_with_keyword() for v in variables]


def add_control_variables_to_idf(idf: IDF, envconfig: EnvConfig):
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


def clear_output_variables(idf: IDF):
    variables = idf.idfobjects["OUTPUT:VARIABLE"]
    variables.clear()
    return idf


def add_output_variables_to_idf(idf: IDF, observation_variables):
    """this is only necessary for cases where sinergym is not used,
    as sinergym adds observation variables automatically"""

    for ov in observation_variables:
        idf.newidfobject(
            "OUTPUT:VARIABLE",
            Key_Value=ov.keyword,
            Variable_Name=ov.name,
            Reporting_Frequency="Hourly",
        )

    return idf


def get_observation_variables(
    idf: IDF, buildingconfig: BuildingConfig, envconfig: EnvConfig
):
    obs_vars = []
    temp_var_names = []

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

    if envconfig.observe_co2_emissions:
        obs_vars.append(
            Variable(
                "Environmental Impact Total CO2 Emissions Carbon Equivalent Mass",
                "Site",
                "kg",
            )
        )

    if envconfig.observe_electricity_demand:
        obs_vars.append(
            Variable(
                "Facility Total Building Electricity Demand Rate", "Whole Building", "W"
            )
        )

    if envconfig.observe_fuel_demand:
        if idf.idfobjects["BOILER:HOTWATER"]:
            if idf.idfobjects["BOILER:HOTWATER"][0].Fuel_Type.lower() == "naturalgas":
                obs_vars.append(Variable("Boiler NaturalGas Rate", "MAIN BOILER", "W"))

    idf_zone_names = []
    for zone in idf.idfobjects["ZONE"]:
        idf_zone_names.append(zone.Name)

    idf_heated_zone_names = []
    for zone in idf.idfobjects["ZONE"]:
        if zone.Name == "ROOF SPACE" and not buildingconfig.attic_is_heated:
            continue
        idf_heated_zone_names.append(zone.Name)

    if envconfig.observe_zone_temperature:
        for zname in idf_zone_names:
            obs_vars.append(Variable("Zone Air Temperature", zname, "C"))
            temp_var_names.append(obs_vars[-1].get_name_with_keyword())

    if envconfig.observe_zone_humidity:
        for zname in idf_zone_names:
            obs_vars.append(Variable("Zone Air Relative Humidity", zname, "%"))

    if envconfig.observe_zone_occupancy:
        for zname in idf_zone_names:
            obs_vars.append(Variable("Zone People Occupant Count", zname, ""))

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
        ):
            for zname in idf_heated_zone_names:
                obs_vars.append(
                    Variable("Zone Thermostat Cooling Setpoint Temperature", zname, "C")
                )

    if envconfig.observe_battery_charge:
        obs_vars.append(
            Variable("Electric Storage Battery Charge State", "SYNERION 24M", "Ah")
        )

    if envconfig.observe_pv_power:
        obs_vars.append(
            Variable("Facility Total Produced Electricity Rate", "Whole Building", "W")
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
    utilities.check_observation_variables(
        obs_var_names, rdd_variables_names, idf_zone_names
    )

    return obs_var_names, obs_vars, temp_var_names
