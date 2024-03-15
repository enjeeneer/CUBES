"""
Classes to define action and observation variables
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from cubes.package import utilities
from cubes.package.envconfig import EnvConfig
from cubes.construct.buildingconfig import BuildingConfig
from cubes.construct import buildingconfig_options as bco
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
    lower_bound: float = None
    upper_bound: float = None

    def __post_init__(self):
        if not self.lower_bound:
            self.lower_bound = self.get_default_obs_range()[0]
        if not self.upper_bound:
            self.upper_bound = self.get_default_obs_range()[1]

    def get_range(self):
        return self.lower_bound, self.upper_bound


    def get_default_obs_range(self):
        if self.dimension_or_unit == "C out":
            return -15.0, 40.0
        elif self.dimension_or_unit == "C in":
            return 10.0,40.0
        elif self.dimension_or_unit == "C":
            return -20.0,100.0
        elif self.dimension_or_unit == "%":
            return 0.0, 100.0
        elif self.dimension_or_unit == "m/s":
            return 0.0, 50.0
        elif self.dimension_or_unit == "deg":
            return 0.0, 360.0
        elif self.dimension_or_unit == (
            "W/m2" and "solar" in self.name.lower() + self.keyword.lower()
        ):
            return 0.0, 1361.0
        elif self.dimension_or_unit == "W":
            return -1e5, 1e5
        elif self.dimension_or_unit == "kg":
            return 0.0, 1.0
        elif self.dimension_or_unit == "gCO2/kWh":
            return 0.0, 500.0
        elif self.dimension_or_unit == "":
            return 0.0, 1e6
        elif self.dimension_or_unit == "ppm":
            return 400., 2000.
        elif self.dimension_or_unit == "fraction":
            return 0.0, 1.0
        elif self.dimension_or_unit == "posneg fraction":
            return -1.0, 1.0
        elif self.dimension_or_unit == "ach":
            return 0.0, 10.0
        elif self.dimension_or_unit == "0/1":
            return 0.0, 1.0
        elif self.dimension_or_unit == "C boiler":
            return 40.0, 85.0
        elif self.dimension_or_unit == "C heatpump":
            return 30.0, 60.0
        elif self.dimension_or_unit == "C heatpump plus DB":
            return 30.0, 65.0

        return -1e8, 1e8

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
                    "fraction",
                )
            )

    if envconfig.control_water_loop_temperature:
        setpoint_manager_entries = idf.idfobjects["SETPOINTMANAGER:SCHEDULED"]
        for sme in setpoint_manager_entries:
            if "DHW" not in sme.Name:
                schedule_name = sme.Name + "-EXT"
                idf.newidfobject(
                    "EXTERNALINTERFACE:SCHEDULE",
                    Name=schedule_name,
                    Initial_Value=building_config.heating_water_loop_temperature,
                )
                sme.Schedule_Name = schedule_name


                if (building_config.heating_water_loop_equipment
                    == bco.HeatingWaterLoopEquipment.ATW_HEAT_PUMP.value):
                    action_variables.append(
                    Variable(
                        schedule_name,
                        "SetpointManager:Scheduled",
                        "C heatpump",
                    )
                )
                    water_heater_mixed_entries = idf.idfobjects["WATERHEATER:MIXED"]
                    for whmw in water_heater_mixed_entries:
                        if "DHW" not in whmw.Name:
                            whmw.Setpoint_Temperature_Schedule_Name = schedule_name

                    #change heat pump compressor setpoint schedule using EMS
                    idf.newidfobject("ENERGYMANAGEMENTSYSTEM:ACTUATOR",
                                     Name = "HP_T_set_override",
                                     Actuated_Component_Unique_Name = (
                                         "Always Radiator Temp Plus DB"),
                                    Actuated_Component_Type = "Schedule:Compact",
                                    Actuated_Component_Control_Type = "Schedule Value")
                    idf.newidfobject("ENERGYMANAGEMENTSYSTEM:SENSOR",
                                     Name="Loop_T_setpoint",
                                     OutputVariable_or_OutputMeter_Index_Key_Name=(
                                         schedule_name
                                     ),
                                     OutputVariable_or_OutputMeter_Name=(
                                         "Schedule Value")
                                     )
                    idf.newidfobject("ENERGYMANAGEMENTSYSTEM:PROGRAMCALLINGMANAGER",
                                     Name="HP T set override prog man",
                                     EnergyPlus_Model_Calling_Point=(
                                         "BeginTimestepBeforePredictor"),
                                     Program_Name_1="HP_T_set_override_prog",)
                    idf.newidfobject("ENERGYMANAGEMENTSYSTEM:PROGRAM",
                                     Name="HP_T_set_override_prog",
                                     Program_Line_1= (
                                         "SET HP_T_set_override = Loop_T_setpoint + 5"))


                else:
                    action_variables.append(
                    Variable(
                        schedule_name,
                        "SetpointManager:Scheduled",
                        "C boiler",
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

                    action_variables.append(Variable(schedule_name, obj, "C in"))

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
                    "C in",
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
                        "C in",
                    )
                )

    if envconfig.control_battery_charging:
        elc_dist = idf.idfobjects["ELECTRICLOADCENTER:DISTRIBUTION"][0]
        if envconfig.battery_storage_operation == "DemandLevelling":
            if idf.idfobjects["ELECTRICLOADCENTER:DISTRIBUTION"]:
                elc_dist.Storage_Operation_Scheme = "FacilityDemandLeveling"
                elc_dist.Storage_Control_Utility_Demand_Target = 10000
                elc_dist.Storage_Control_Utility_Demand_Target_Fraction_Schedule_Name = (  # pylint: disable=line-too-long
                    "Utility Demand Target Schedule-EXT"
                )
                idf.newidfobject(
                    "EXTERNALINTERFACE:SCHEDULE",
                    Name="Utility Demand Target Schedule-EXT",
                    Initial_Value=0.0,
                )
                if envconfig.negative_emissions_for_export:
                    action_variables.append(
                        Variable(
                            "Utility Demand Target Schedule-EXT",
                            "Storage Control Utility Demand Target Fraction Schedule",
                            "posneg fraction",
                        )
                    )
                else:
                    action_variables.append(
                        Variable(
                            "Utility Demand Target Schedule-EXT",
                            "Storage Control Utility Demand Target Fraction Schedule",
                            "fraction",
                        )
                    )
        elif envconfig.battery_storage_operation == "TrackChargeDischargeSchedules":
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
        else:
            raise ValueError(
                f"battery_storage_operation "
                f"{envconfig.battery_storage_operation} not supported"
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
    temp_var_names = {}
    temp_set_var_names = []
    occ_var_names = []
    aq_var_names = {}

    if envconfig.observe_outside_temperature:
        obs_vars.append(
            Variable("Site Outdoor Air Drybulb Temperature", "Environment", "C out")
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
    if envconfig.observe_total_purchased_electricity:
        obs_vars.append(
            Variable("Facility Total Purchased Electricity Rate", "Whole Building", "W")
        )
    if envconfig.observe_total_surplus_electricity:
        obs_vars.append(
            Variable("Facility Total Surplus Electricity Rate", "Whole Building", "W")
        )

    if envconfig.observe_electricity_demand:
        obs_vars.append(
            Variable("Facility Total Electricity Demand Rate", "Whole Building", "W")
        )

    if envconfig.observe_fuel_demand:
        # obs_vars.append(Variable("Boiler NaturalGas Energy","Main Boiler","J"))
        # obs_vars.append(Variable("Boiler Heating Energy","Main Boiler","J"))
        # obs_vars.append(Variable("Boiler Inlet Temperature","Main Boiler","C"))
        #obs_vars.append(Variable("Boiler Outlet Temperature","Main Boiler","C"))
        # obs_vars.append(Variable("Boiler Mass Flow Rate","Main Boiler",""))
        # obs_vars.append(Variable("Boiler Ancillary Electricity Energy","Main Boiler",""))
        #obs_vars.append(Variable("Boiler Efficiency","Main Boiler",""))
        # obs_vars.append(Variable("Baseboard Total Heating Energy","living-baseboard heat",""))
        # obs_vars.append(Variable("Baseboard Convective Heating Energy","living-baseboard heat",""))
        # obs_vars.append(Variable("Baseboard Radiant Heating Energy","living-baseboard heat",""))
        # obs_vars.append(Variable("Baseboard Hot Water Energy","living-baseboard heat",""))
        # obs_vars.append(Variable("Baseboard Hot Water Mass Flow Rate","living-baseboard heat",""))
        # obs_vars.append(Variable("Baseboard Air Mass Flow Rate","living-baseboard heat",""))
        # obs_vars.append(Variable("Baseboard Air Inlet Temperature","living-baseboard heat",""))
        # obs_vars.append(Variable("Baseboard Air Outlet Temperature","living-baseboard heat",""))
        # obs_vars.append(Variable("Baseboard Water Inlet Temperature","living-baseboard heat",""))
        # obs_vars.append(Variable("Baseboard Water Outlet Temperature","living-baseboard heat",""))
        # obs_vars.append(Variable("Zone Air Heat Balance Internal Convective Heat Gain Rate","living","W"))
        # obs_vars.append(Variable("Zone Air Heat Balance Surface Convection Rate","living","W"))
        # obs_vars.append(Variable("Zone Air Heat Balance Interzone Air Transfer Rate","living","W"))
        # obs_vars.append(Variable("Zone Air Heat Balance Outdoor Air Transfer Rate","living","W"))
        # obs_vars.append(Variable("Zone Air Heat Balance System Air Transfer Rate","living","W"))
        # obs_vars.append(Variable("Zone Air Heat Balance System Convective Heat Gain Rate","living","W"))
        # obs_vars.append(Variable("Zone Air Heat Balance Air Energy Storage Rate","living","W"))
        # obs_vars.append(Variable("Zone Air Heat Balance Deviation Rate","living","W"))

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
        for zname in idf_heated_zone_names:
            if buildingconfig.use_operative_temperature:
                obs_vars.append(Variable("Zone Operative Temperature", zname, "C in"))
            else:
                obs_vars.append(Variable("Zone Air Temperature", zname, "C in"))
            if zname not in temp_var_names:
                temp_var_names[zname] = []
            temp_var_names[zname].append(obs_vars[-1].get_name_with_keyword())
            obs_vars.append(Variable("Zone Air Temperature", zname, "C in"))
        # obs_vars.append(Variable("Surface Inside Face Temperature", "storey 1 north wall living", "C"))
        # obs_vars.append(Variable("Surface Inside Face Conduction Heat Transfer Rate", "storey 1 north wall living", "W"))
        # obs_vars.append(Variable("Surface Outside Face Temperature", "storey 1 north wall living", "C"))
        # obs_vars.append(Variable("Surface Outside Face Conduction Heat Transfer Rate", "storey 1 north wall living", "W"))
        # obs_vars.append(Variable("Surface Outside Face Temperature", "storey 1 east wall living", "C"))
        # obs_vars.append(Variable("Surface Outside Face Conduction Heat Transfer Rate", "storey 1 east wall living", "W"))
        # obs_vars.append(Variable("Surface Outside Face Conduction Heat Transfer Rate", "storey 1 west wall living", "W"))
        # obs_vars.append(Variable("Surface Outside Face Conduction Heat Transfer Rate", "storey 1 south wall living", "W"))
        # obs_vars.append(Variable("Surface Outside Face Conduction Heat Transfer Rate", "loft side wall 1", "W"))
        # obs_vars.append(Variable("Surface Outside Face Conduction Heat Transfer Rate", "loft side wall 2", "W"))
        # obs_vars.append(Variable("Surface Outside Face Conduction Heat Transfer Rate", "roof surface 1", "W"))
        # obs_vars.append(Variable("Surface Outside Face Conduction Heat Transfer Rate", "roof surface 2", "W"))
        # obs_vars.append(Variable("Surface Outside Face Conduction Heat Transfer Rate", "storey 1 floor living-subfloor", "W"))
        # obs_vars.append(Variable("Surface Outside Face Convection Heat Gain Rate", "storey 1 north wall living", "W"))
        # obs_vars.append(Variable("Surface Outside Face Convection Heat Gain Rate", "storey 1 east wall living", "W"))
        # obs_vars.append(Variable("Surface Outside Face Convection Heat Gain Rate", "storey 1 west wall living", "W"))
        # obs_vars.append(Variable("Surface Outside Face Convection Heat Gain Rate", "storey 1 south wall living", "W"))
        # obs_vars.append(Variable("Surface Outside Face Convection Heat Gain Rate", "loft side wall 1", "W"))
        # obs_vars.append(Variable("Surface Outside Face Convection Heat Gain Rate", "loft side wall 2", "W"))
        # obs_vars.append(Variable("Surface Outside Face Convection Heat Gain Rate", "roof surface 1", "W"))
        # obs_vars.append(Variable("Surface Outside Face Convection Heat Gain Rate", "roof surface 2", "W"))
        # obs_vars.append(Variable("Surface Outside Face Convection Heat Gain Rate", "storey 1 floor living-subfloor", "W"))
        # obs_vars.append(Variable("Surface Outside Face Convection Heat Gain Rate", "intmass-furniture-living","W"))
        # obs_vars.append(Variable("Surface Outside Face Convection Heat Gain Rate", "intmass-partitions-living","W"))
        # obs_vars.append(Variable("Surface Outside Face Convection Heat Gain Rate", "intmass-living-loft-floor","W"))
        # obs_vars.append(Variable("Surface Outside Face Convection Heat Gain Rate", "intmass-living-loft-ceiling","W"))
        # obs_vars.append(Variable("Surface Outside Face Net Surface Thermal Radiation Heat Gain Rate", "storey 1 east wall living", "W"))
        # obs_vars.append(Variable("Surface Outside Face Net Surface Thermal Radiation Heat Gain Rate", "storey 1 west wall living", "W"))
        # obs_vars.append(Variable("Surface Outside Face Net Surface Thermal Radiation Heat Gain Rate", "storey 1 south wall living", "W"))
        # obs_vars.append(Variable("Surface Outside Face Net Surface Thermal Radiation Heat Gain Rate", "loft side wall 1", "W"))
        # obs_vars.append(Variable("Surface Outside Face Net Surface Thermal Radiation Heat Gain Rate", "loft side wall 2", "W"))
        # obs_vars.append(Variable("Surface Outside Face Net Surface Thermal Radiation Heat Gain Rate", "roof surface 1", "W"))
        # obs_vars.append(Variable("Surface Outside Face Net Surface Thermal Radiation Heat Gain Rate", "roof surface 2", "W"))
        # obs_vars.append(Variable("Surface Outside Face Net Surface Thermal Radiation Heat Gain Rate", "storey 1 floor living-subfloor", "W"))
        # obs_vars.append(Variable("Surface Outside Face Net Surface Thermal Radiation Heat Gain Rate", "storey 1 floor living-subfloor", "W"))



        # obs_vars.append(Variable("Surface Outside Face Temperature", "storey 1 east wall living", "C"))
        # obs_vars.append(Variable("Surface Outside Face Conduction Heat Transfer Rate", "storey 1 east wall living", "W"))
        # obs_vars.append(Variable("Surface Outside Face Conduction Heat Transfer Rate", "intmass-furniture-living","W"))
        # obs_vars.append(Variable("Surface Outside Face Conduction Heat Transfer Rate", "intmass-partitions-living","W"))
        # obs_vars.append(Variable("Surface Outside Face Conduction Heat Transfer Rate", "intmass-living-loft-floor","W"))
        # obs_vars.append(Variable("Surface Outside Face Conduction Heat Transfer Rate", "intmass-living-loft-ceiling","W"))

    if envconfig.control_water_loop_temperature:
        setpoint_manager_entries = idf.idfobjects["SETPOINTMANAGER:SCHEDULED"]
        has_hp = (buildingconfig.heating_water_loop_equipment
                    == bco.HeatingWaterLoopEquipment.ATW_HEAT_PUMP.value)
        for sme in setpoint_manager_entries:
            if "DHW" not in sme.Name:
                schedule_name = sme.Name + "-EXT"
                obs_vars.append(Variable(
                    "Schedule Value",
                    schedule_name,
                    "C boiler" if not has_hp else "C heatpump",
                ))
        if has_hp:
            obs_vars.append(Variable("Schedule Value",
                    "Always Radiator Temp Plus DB",
                    "C heatpump plus DB"))

    if envconfig.observe_zone_humidity:
        for zname in idf_heated_zone_names:
            obs_vars.append(Variable("Zone Air Relative Humidity", zname, "%"))

    if envconfig.observe_zone_co2:
        for zname in idf_heated_zone_names:
            obs_vars.append(Variable("Zone Air CO2 Concentration", zname, "ppm"))
            if zname not in aq_var_names:
                aq_var_names[zname] = []
            aq_var_names[zname].append(obs_vars[-1].get_name_with_keyword())

    if envconfig.observe_zone_occupancy:
        for zname in idf_heated_zone_names:
            obs_vars.append(Variable("Zone People Occupant Count", zname, "",
                                     lower_bound=0,
                                     upper_bound=buildingconfig.occupant_value))
            occ_var_names.append(obs_vars[-1].get_name_with_keyword())

    idf_people_names = []
    for people in idf.idfobjects["PEOPLE"]:
        idf_people_names.append(people.Name)

    if envconfig.observe_thermal_comfort:
        for pn in idf_people_names:
            obs_vars.append(
                Variable("Zone Thermal Comfort Mean Radiant Temperature", pn, "C in")
            )
            obs_vars.append(Variable("Zone Air Relative Humidity", pn, "%"))
            obs_vars.append(Variable("Zone Thermal Comfort Clothing Value", pn, ""))
            obs_vars.append(Variable("Zone Thermal Comfort Fanger Model PPD", pn, ""))
            obs_vars.append(Variable("People Air Temperature", pn, "C in"))


    if envconfig.control_thermostat_setpoints:
        objects = [
            "THERMOSTATSETPOINT:SINGLEHEATING",
            "THERMOSTATSETPOINT:SINGLECOOLING",
        ]
        for obj in objects:
            for setpoint_entries in idf.idfobjects[obj]:
                for se in setpoint_entries:
                    schedule_name = se.Name + "-EXT"
                    obs_vars.append(Variable("Schedule Value",schedule_name, "C in"))

        setpoint_entries = idf.idfobjects["THERMOSTATSETPOINT:DUALSETPOINT"]
        for se in setpoint_entries:
            heating_schedule_name = se.Name + "-HEATING-EXT"
            cooling_schedule_name = se.Name + "-COOLING-EXT"
            se.Heating_Setpoint_Temperature_Schedule_Name = heating_schedule_name

            obs_vars.append(
                Variable(
                    "Schedule Value",
                    heating_schedule_name,
                    "C in",
                )
            )

            if buildingconfig.cooling_system_installed:
                obs_vars.append(
                    Variable(
                        "Schedule Value",
                        cooling_schedule_name,
                        "C in",
                    )
                )

    if envconfig.observe_zone_thermostat_setpoints:
        if (
            idf.idfobjects["THERMOSTATSETPOINT:DUALSETPOINT"]
            or idf.idfobjects["THERMOSTATSETPOINT:SINGLEHEATING"]
        ):
            for zname in idf_heated_zone_names:
                # if buildingconfig.use_operative_temperature:
                #     obs_vars.append(
                #         Variable("Zone Thermostat Operative Temperature", zname, "C in",
                #                  lower_bound= buildingconfig.heating_setback,
                #                  upper_bound= (buildingconfig.heating_setpoint
                #                                + buildingconfig.cooling_setpoint)/ 2)
                #     )
                # else:
                obs_vars.append(
                    Variable(
                        "Zone Thermostat Heating Setpoint Temperature", zname,
                        "C in",
                        lower_bound= buildingconfig.heating_setback,
                        upper_bound= (buildingconfig.heating_setpoint
                                            + buildingconfig.cooling_setpoint)/ 2
                    )
                )
                temp_set_var_names.append(obs_vars[-1].get_name_with_keyword())


        if (
            idf.idfobjects["THERMOSTATSETPOINT:DUALSETPOINT"]
            or idf.idfobjects["THERMOSTATSETPOINT:SINGLECOOLING"]
        ) and buildingconfig.cooling_system_installed:
            for zname in idf_heated_zone_names:
                obs_vars.append(
                    Variable("Zone Thermostat Cooling Setpoint Temperature",
                             zname, "C in",
                             lower_bound= (buildingconfig.heating_setpoint
                                               + buildingconfig.cooling_setpoint)/ 2,
                             upper_bound= buildingconfig.cooling_setback)
                )

    if envconfig.observe_zone_ventilation:
        for zname in idf_heated_zone_names:

            obs_vars.append(Variable("Zone Ventilation Air Change Rate", zname, "ach",
            ))

    if envconfig.control_ventilation:
        # search through IDF file for ventilation entries
        ventilation_entries = idf.idfobjects["ZONEVENTILATION:DESIGNFLOWRATE"]
        for v in ventilation_entries:
            # add an ExternalInterface:Schedule for each and insert schedule name
            schedule_name = v.Name + "-EXT"
            obs_vars.append(Variable("Schedule Value", schedule_name, "fraction",
            ))

    if envconfig.observe_battery_charge:
        obs_vars.append(
            Variable("Electric Storage Battery Charge State", "SYNERION 24M", "Ah",
            )
        )
    if envconfig.observe_battery_charging:
        # obs_vars.append(Variable("Electric Storage Charge Power", "SYNERION 24M", "W"))
        # obs_vars.append(
        #     Variable("Electric Storage Discharge Power", "SYNERION 24M", "W",
        #              lower_bound=0,
        #              upper_bound=battery_charging_power)
        # )
        if envconfig.battery_storage_operation == "DemandLevelling":
            obs_vars.append(
                Variable(
                    "Schedule Value",
                    "Utility Demand Target Schedule-EXT",
                    "posneg fraction",
                )
            )

    if envconfig.observe_pv_power:
        obs_vars.append(
            Variable(
                "Electric Load Center Produced Electricity Rate",
                "DC with inverter and Synerion 24M",
                "W",
            )
        )
        obs_vars.append(
            Variable(
                "Electric Load Center Supplied Electricity Rate",
                "DC with inverter and Synerion 24M",
                "W",
            )
        )
        obs_vars.append(
            Variable(
                "Electric Load Center Drawn Electricity Rate",
                "DC with inverter and Synerion 24M",
                "W",
            )
        )

    if envconfig.observe_grid_carbon_intensity:
        obs_vars.append(
            Variable("Schedule Value", "Grid Carbon Intensity Schedule", "gCO2/kWh")
        )

    if envconfig.observe_outside_temperature_in_x_hours_forecast:
        for tfh in envconfig.observe_outside_temperature_in_x_hours_forecast:
            idf.newidfobject(
                "SCHEDULE:FILE",
                Name=str(tfh) + " Hour Temperature Forecast Schedule",
                Schedule_Type_Limits_Name="Any Number",
                File_Name=utilities.get_temperature_forecast_file_path(
                    env_files_dir=envconfig.files_dir, hours=tfh
                ),
                Column_Number=1,
                Rows_to_Skip_at_Top=0,
                Number_of_Hours_of_Data=8760,
                Minutes_per_Item=10,
                Interpolate_to_Timestep="yes",
            )
            obs_vars.append(
                Variable(
                    "Schedule Value",
                    str(tfh) + " Hour Temperature Forecast Schedule",
                    "C out",
                )
            )

    if envconfig.observe_grid_carbon_in_x_hours_forecast:
        for gfh in envconfig.observe_grid_carbon_in_x_hours_forecast:
            idf.newidfobject(
                "SCHEDULE:FILE",
                Name=str(gfh) + " Hour Grid Carbon Forecast Schedule",
                Schedule_Type_Limits_Name="Any Number",
                File_Name=utilities.get_grid_forecast_file_path(
                    env_files_dir=envconfig.files_dir, hours=gfh
                ),
                Column_Number=1,
                Rows_to_Skip_at_Top=0,
                Number_of_Hours_of_Data=8760,
                Minutes_per_Item=10,
            )
            obs_vars.append(
                Variable(
                    "Schedule Value",
                    str(gfh) + " Hour Grid Carbon Forecast Schedule",
                    "gCO2/kWh",
                )
            )

    if envconfig.observe_comfort_temp_in_x_hours_forecast:
        for cfh in envconfig.observe_comfort_temp_in_x_hours_forecast:
            for zone in idf_heated_zone_names:
                idf.newidfobject(
                    "SCHEDULE:FILE",
                    Name=f"{cfh} Hour {zone} Comfort Temperature Forecast Schedule",
                    Schedule_Type_Limits_Name="Any Number",
                    File_Name=utilities.get_comfort_temp_forecast_file_path(
                        env_files_dir=envconfig.files_dir, hours=cfh, zone=zone
                    ),
                    Column_Number=1,
                    Rows_to_Skip_at_Top=0,
                    Number_of_Hours_of_Data=8760,
                    Minutes_per_Item=10,
                )
                obs_vars.append(
                    Variable(
                        "Schedule Value",
                        f"{cfh} Hour {zone} Comfort Temperature Forecast Schedule",
                        "C in",
                        lower_bound= buildingconfig.heating_setback,
                        upper_bound= (buildingconfig.heating_setpoint
                                        + buildingconfig.cooling_setpoint)/ 2
                    )
                )

    if envconfig.observe_solar_irradiance_in_x_hours_forecast:
        for sfh in envconfig.observe_solar_irradiance_in_x_hours_forecast:
            idf.newidfobject(
                "SCHEDULE:FILE",
                Name=str(sfh) + " Hour Solar Irradiance Forecast Schedule",
                Schedule_Type_Limits_Name="Any Number",
                File_Name=utilities.get_solar_forecast_file_path(
                    env_files_dir=envconfig.files_dir, hours=sfh
                ),
                Column_Number=1,
                Rows_to_Skip_at_Top=0,
                Number_of_Hours_of_Data=8760,
                Minutes_per_Item=10,
            )
            obs_vars.append(
                Variable(
                    "Schedule Value",
                    str(sfh) + " Hour Solar Irradiance Forecast Schedule",
                    "W/m2",
                )
            )

    # get rdd file
    # Extract rdd observation variables names
    rdd_data = pd.read_csv(envconfig.files_dir + "/building_model.rdd", skiprows=1)
    rdd_variables_names = list(
        map(
            lambda name: name.split(" [")[0], rdd_data["Variable Name [Units]"].tolist()
        )
    )

    obs_var_names = get_variable_names_with_keywords(obs_vars)

    # check that observation variables are viable
    utilities.check_observation_variables(obs_var_names, rdd_variables_names)

    return (idf, obs_var_names, obs_vars, temp_var_names, temp_set_var_names,
            occ_var_names, aq_var_names)


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
                    [
                        (observation, operator.gt, 0),
                        ("hour", operator.lt, env_config.sleep_hours[0]),
                        ("hour", operator.ge, env_config.sleep_hours[1]),
                    ],
                    buildingconfig.heating_setpoint,
                    (buildingconfig.heating_setpoint + buildingconfig.cooling_setpoint)
                    / 2,
                ]
    if env_config.enforce_ventilation:
        for zn in _get_heated_zones(idf, buildingconfig):
            action = ""
            observation = ""
            for avn in action_variable_names:
                if zn.lower() in avn.lower() and "VENTILATION-EXT" in avn.upper():
                    action = avn
            # for ovn in observation_variable_names:
            #     if zn.lower() in ovn.lower() and "Zone Air CO2 Concentration" in ovn:
            #         observation = ovn
            for ovn in observation_variable_names:
                if zn.lower() in ovn.lower() and "People Occupant Count" in ovn:
                    observation2 = ovn
            if action and observation:
                # remapping_dict[action] = [
                #     [
                #         (observation, operator.gt, env_config.air_quality_range[1]),
                #     ],
                #     1, 1,
                # ]
                remapping_dict[action] = [
                    [
                        (observation2, operator.lt, 1),
                    ],
                    0, 0,
                ]

    return remapping_dict

def get_action_discretization(
    action_variable_names,
    env_config: EnvConfig,
):
    discretize_dict = {}
    # if env_config.discrete_actions:
    #     for avn in action_variable_names:
    #         if "HEATING-EXT" in avn:
    #             n_points = int((buildingconfig.heating_setpoint
    #                             + buildingconfig.cooling_setpoint)
    #                             / 2 - buildingconfig.heating_setback)+1
    #             discretize_dict[avn] = np.linspace(buildingconfig.heating_setback,
    #                                                (buildingconfig.heating_setpoint
    #                                                 + buildingconfig.cooling_setpoint)
    #                                                 / 2,num=n_points)
    if env_config.discrete_window_actions:
        for avn in action_variable_names:
            if "Ventilation-EXT" in avn:
                n_points = 3
                discretize_dict[avn] = np.linspace(-1,1,num=n_points)
                # n_points = 2
                # discretize_dict[avn] = np.linspace(-1,1,num=n_points)

    if env_config.discrete_battery_actions:
        for avn in action_variable_names:
            if "Utility Demand Target" in avn:
                #if env_config.negative_emissions_for_export:
                n_points = 3
                discretize_dict[avn] = np.linspace(-1,1,num=n_points)
                # else:
                #     n_points = 2
                #     discretize_dict[avn] = np.linspace(0,1,num=n_points)

    return discretize_dict

def get_incremental_action(
    idf: IDF,
    action_variable_names,
    observation_variable_names,
    buildingconfig: BuildingConfig,
    env_config: EnvConfig,
):
    """return a dictionary with the actions as keys
    and the values as [observation_name, max_increment, initial value]"""
    incremental_dict = {}
    if env_config.incremental_actions:
        for zn in _get_heated_zones(idf, buildingconfig):
            action = ""
            observation = ""
            for avn in action_variable_names:
                if zn.lower() in avn.lower() and "HEATING-EXT" in avn:
                    action = avn
            for ovn in observation_variable_names:
                if zn.lower() in ovn.lower() and "HEATING-EXT" in ovn:
                    observation = ovn
            if action and observation:
                incremental_dict[action] = [observation,1,20]

        if env_config.control_water_loop_temperature:
            for avn in action_variable_names:
                if "Hot Water Loop Temp Manager".lower() in avn.lower():
                    action = avn
            for ovn in observation_variable_names:
                if "Hot Water Loop Temp Manager".lower() in ovn.lower():
                    observation = ovn

            if action and observation:
                incremental_dict[action] = [observation,20,
                                        buildingconfig.heating_water_loop_temperature]

        if env_config.control_ventilation:
            for zn in _get_heated_zones(idf, buildingconfig):
                action = ""
                observation = ""
                for avn in action_variable_names:
                    if zn.lower() in avn.lower() and "VENTILATION-EXT" in avn.upper():
                        action = avn
                for ovn in observation_variable_names:
                    if zn.lower() in ovn.lower() and  "VENTILATION-EXT" in ovn.upper():
                        observation = ovn
                if action and observation:
                    incremental_dict[action] = [observation,1,0]

        if env_config.discrete_battery_actions:
            action = ""
            observation = ""
            for avn in action_variable_names:
                if "Utility Demand Target" in avn:
                    action = avn
            for ovn in observation_variable_names:
                if "Utility Demand Target" in ovn:
                    observation = ovn
            if action and observation:
                incremental_dict[action] = [observation,1,0]


    return incremental_dict
