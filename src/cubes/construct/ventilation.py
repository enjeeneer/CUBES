"""This module adds objects for ventilation to an IDF"""
from geomeppy import IDF
import cubes.construct.buildingconfig_options as bco
from cubes.construct.buildingconfig import BuildingConfig
from cubes.constants import package_directory
from cubes.behaviour_models.constants import variables_for_ventilation_models
from cubes.construct import utilities


def add_ventilation(idf: IDF, building_config: BuildingConfig, conditioned_zones):
    """Adds ventilation into e+ for every zone in idf"""

    idf.newidfobject(
        "ZONEAIRCONTAMINANTBALANCE",
        Carbon_Dioxide_Concentration="Yes",
        Outdoor_Carbon_Dioxide_Schedule_Name="Outdoor CO2 Schedule",
    )

    idf.newidfobject(
        "SCHEDULE:CONSTANT", Name="Outdoor CO2 Schedule", Hourly_Value=420.0
    )

    if building_config.ventilation_type == bco.VentilationType.NATURAL.value:

        if building_config.natural_ventilation_method in [
            bco.NaturalVentilationMethod.RATE_PER_OCCUPANT.value,
            bco.NaturalVentilationMethod.RATE_PER_OCCUPANT_PLUS_COOLING.value,
        ]:

            idf.newidfobject(
                "ZONEVENTILATION:DESIGNFLOWRATE",
                Name="Living" + "-Ventilation",
                Zone_or_ZoneList_Name="Living",
                Schedule_Name="Occupancy-Schedule-Living",
                Design_Flow_Rate_Calculation_Method=("Flow/Person"),
                Flow_Rate_per_Person=building_config.ventilation_rate_per_occupant,
                Ventilation_Type="Natural",
            )

            idf.newidfobject(
                "ZONEVENTILATION:DESIGNFLOWRATE",
                Name="Bedroom" + "-Ventilation",
                Zone_or_ZoneList_Name="Living",
                Schedule_Name="Occupancy-Schedule-Bedroom",
                Design_Flow_Rate_Calculation_Method=("Flow/Person"),
                Flow_Rate_per_Person=building_config.ventilation_rate_per_occupant,
                Ventilation_Type="Natural",
            )

        if (
            building_config.natural_ventilation_method
            == bco.NaturalVentilationMethod.RATE_PER_OCCUPANT_PLUS_COOLING.value
        ):
            idf.newidfobject(
                "ZONEVENTILATION:DESIGNFLOWRATE",
                Name="Living" + "-Cooling Ventilation",
                Zone_or_ZoneList_Name="Living",
                Schedule_Name="Occupancy-Schedule-Living",
                Design_Flow_Rate_Calculation_Method=("AirChanges/Hour"),
                Air_Changes_per_Hour=(
                    building_config.natural_ventilation_rate_open_windows
                ),
                Ventilation_Type="Natural",
                Constant_Term_Coefficient=1,
                Temperature_Term_Coefficient=0,
                Velocity_Term_Coefficient=0,
                Velocity_Squared_Term_Coefficient=0,
                Minimum_Indoor_Temperature=(building_config.cooling_setpoint - 1),
                Minimum_Indoor_Temperature_Schedule_Name="",
                Maximum_Indoor_Temperature=(building_config.cooling_setpoint + 3),
                Maximum_Indoor_Temperature_Schedule_Name="",
                Delta_Temperature=1,
            )
            idf.newidfobject(
                "ZONEVENTILATION:DESIGNFLOWRATE",
                Name="Bedroom" + "-Cooling Ventilation",
                Zone_or_ZoneList_Name="Living",
                Schedule_Name="Occupancy-Schedule-Bedroom",
                Design_Flow_Rate_Calculation_Method=("AirChanges/Hour"),
                Air_Changes_per_Hour=(
                    building_config.natural_ventilation_rate_open_windows
                ),
                Ventilation_Type="Natural",
                Constant_Term_Coefficient=1,
                Temperature_Term_Coefficient=0,
                Velocity_Term_Coefficient=0,
                Velocity_Squared_Term_Coefficient=0,
                Minimum_Indoor_Temperature=(building_config.cooling_setpoint - 1),
                Minimum_Indoor_Temperature_Schedule_Name="",
                Maximum_Indoor_Temperature=(building_config.cooling_setpoint + 3),
                Maximum_Indoor_Temperature_Schedule_Name="",
                Delta_Temperature=1,
            )

        elif (
            building_config.natural_ventilation_method
            == bco.NaturalVentilationMethod.RES_WIN_OP_MODEL.value
        ):
            for zone in conditioned_zones:
                idf.newidfobject(
                    "ZONEVENTILATION:DESIGNFLOWRATE",
                    Name=zone.Name + "-Natural Ventilation",
                    Zone_or_ZoneList_Name=zone.Name,
                    Schedule_Name=zone.Name + "-Ventilation-Schedule",
                    Design_Flow_Rate_Calculation_Method=("AirChanges/Hour"),
                    Air_Changes_per_Hour=building_config.nat_vent_rate,
                    Ventilation_Type="Natural",
                    Constant_Term_Coefficient=1,
                    Temperature_Term_Coefficient=0,
                    Velocity_Term_Coefficient=0,
                    Velocity_Squared_Term_Coefficient=0,
                )

                idf.newidfobject(
                    "SCHEDULE:CONSTANT",
                    Name=zone.Name + "-Ventilation-Schedule",
                    Hourly_Value=0.0,
                )
                # add necessary output variables to idf
                for var in variables_for_ventilation_models[
                    building_config.natural_ventilation_model
                ]:
                    if var.split(" ")[0].lower() == "zone":
                        idf.newidfobject(
                            "OUTPUT:VARIABLE",
                            Key_Value=zone.Name,
                            Variable_Name=var,
                            Reporting_Frequency="Hourly",
                        )
                    else:
                        idf.newidfobject(
                            "OUTPUT:VARIABLE",
                            Key_Value=var.split(" ")[0].lower(),
                            Variable_Name=var,
                            Reporting_Frequency="Hourly",
                        )

            idf.newidfobject(
                "PythonPlugin:Instance".upper(),
                Name="Ventilation Override",
                Run_During_Warmup_Days="Yes",
                Python_Module_Name="natural_ventilation_residential",
                Plugin_Class_Name=bco.res_window_PP_map[
                    building_config.natural_ventilation_model
                ],
            )

            idf.newidfobject(
                "PythonPlugin:SearchPaths".upper(),
                Name="PythonPlugin search paths",
                Add_Current_Working_Directory_to_Search_Path="Yes",
                Add_Input_File_Directory_to_Search_Path="No",
                Search_Path_1=package_directory + "/behaviour_models",
            )

    elif building_config.ventilation_type == bco.VentilationType.MECHANICAL.value:
        idf = add_mvhr(idf, building_config, conditioned_zones)

    return idf


def add_mvhr(idf: IDF, building_config: BuildingConfig, conditioned_zones):

    for zone in conditioned_zones:

        idf.newidfobject(
            "ZONEHVAC:ENERGYRECOVERYVENTILATOR",
            Name=zone.Name + " RecVent",
            Availability_Schedule_Name="Always 1",
            Heat_Exchanger_Name="OA Heat Recovery " + zone.Name,
            Supply_Air_Flow_Rate="autosize",
            Exhaust_Air_Flow_Rate="autosize",
            Supply_Air_Fan_Name=zone.Name + " RecVent Supply Fan",
            Exhaust_Air_Fan_Name=zone.Name + " RecVent Exhaust Fan",
            Controller_Name="ERV OA Controller " + zone.Name,
            Ventilation_Rate_per_Occupant=building_config.ventilation_rate_per_occupant,
        )

        idf.newidfobject(
            "ZONEHVAC:ENERGYRECOVERYVENTILATOR:CONTROLLER",
            Name="ERV OA Controller " + zone.Name,
            Temperature_High_Limit=19,
            Temperature_Low_Limit=14,
        )

        # find existing equipment list
        equ_list_idx = -1
        for elidx, el in enumerate(idf.idfobjects["ZONEHVAC:EQUIPMENTLIST"]):
            if el.Name == utilities.get_zone_hvac_equipment_list_name(zone.Name):
                equ_list_idx = elidx

                equ_list = idf.idfobjects["ZONEHVAC:EQUIPMENTLIST"][equ_list_idx]
                # append Ventilator to end of list
                neq = 1
                while getattr(equ_list, "Zone_Equipment_" + str(neq) + "_Object_Type"):
                    neq += 1

                setattr(
                    equ_list,
                    "Zone_Equipment_" + str(neq) + "_Object_Type",
                    "ZoneHVAC:EnergyRecoveryVentilator",
                )
                setattr(
                    equ_list,
                    "Zone_Equipment_" + str(neq) + "_Name",
                    zone.Name + " RecVent",
                )
                setattr(
                    equ_list,
                    "Zone_Equipment_" + str(neq) + "_Cooling_Sequence",
                    str(neq),
                )
                setattr(
                    equ_list,
                    "Zone_Equipment_" + str(neq) + "_Heating_or_NoLoad_Sequence",
                    str(neq),
                )

        if equ_list_idx == -1:
            print("zonehvac:equipmentlist needs to be defined before add_mvhr()")
            return idf

        # find equipment connections
        found_ec = False
        for i_ec, ec in enumerate(idf.idfobjects["ZONEHVAC:EQUIPMENTCONNECTIONS"]):
            if ec.Zone_Name == zone.Name:

                inlet_nl_name = utilities.get_zone_air_inlet_nodelist_name(zone.Name)
                idf.idfobjects["ZONEHVAC:EQUIPMENTCONNECTIONS"][
                    i_ec
                ].Zone_Air_Inlet_Node_or_NodeList_Name = inlet_nl_name

                outlet_nl_name = utilities.get_zone_air_outlet_nodelist_name(zone.Name)
                idf.idfobjects["ZONEHVAC:EQUIPMENTCONNECTIONS"][
                    i_ec
                ].Zone_Air_Exhaust_Node_or_NodeList_Name = outlet_nl_name

                found_ec = True

        if not found_ec:
            print("zonehvac:equipmenconnections needs to be defined before add_mvhr()")
            return idf

        idf = utilities.append_node_to_nodelist(
            idf,
            zone.Name + " RecVent Supply Fan Outlet Node",
            utilities.get_zone_air_inlet_nodelist_name(zone.Name),
        )
        idf = utilities.append_node_to_nodelist(
            idf,
            zone.Name + " Exhaust Node",
            utilities.get_zone_air_outlet_nodelist_name(zone.Name),
        )

        idf.newidfobject(
            "OUTDOORAIR:NODELIST",
            Node_or_NodeList_Name_1=zone.Name + " OutsideAirInletNodes",
        )
        idf.newidfobject(
            "NODELIST",
            Name=zone.Name + " OutsideAirInletNodes",
            Node_1_Name=zone.Name + " RecVent Outdoor Air Inlet Node",
        )
        idf.newidfobject(
            "NODELIST",
            Name=zone.Name + " Heat Exchanger Supply Air Nodes",
            Node_1_Name=zone.Name + " Heat Recovery Outlet Node",
        )

        idf.newidfobject(
            "HEATEXCHANGER:AIRTOAIR:SENSIBLEANDLATENT",
            Name="OA Heat Recovery " + zone.Name,
            Availability_Schedule_Name="Always 1",
            Nominal_Supply_Air_Flow_Rate="autosize",
            Sensible_Effectiveness_at_100_Heating_Air_Flow=(
                building_config.mech_ventilation_heat_recovery_efficiency_sensible
            ),
            Latent_Effectiveness_at_100_Heating_Air_Flow=(
                building_config.mech_ventilation_heat_recovery_efficiency_latent
            ),
            Sensible_Effectiveness_at_75_Heating_Air_Flow=(
                building_config.mech_ventilation_heat_recovery_efficiency_sensible
            ),
            Latent_Effectiveness_at_75_Heating_Air_Flow=(
                building_config.mech_ventilation_heat_recovery_efficiency_latent
            ),
            Sensible_Effectiveness_at_100_Cooling_Air_Flow=(
                building_config.mech_ventilation_heat_recovery_efficiency_sensible
            ),
            Latent_Effectiveness_at_100_Cooling_Air_Flow=(
                building_config.mech_ventilation_heat_recovery_efficiency_latent
            ),
            Sensible_Effectiveness_at_75_Cooling_Air_Flow=(
                building_config.mech_ventilation_heat_recovery_efficiency_sensible
            ),
            Latent_Effectiveness_at_75_Cooling_Air_Flow=(
                building_config.mech_ventilation_heat_recovery_efficiency_latent
            ),
            Supply_Air_Inlet_Node_Name=zone.Name + " RecVent Outdoor Air Inlet Node",
            Supply_Air_Outlet_Node_Name=zone.Name + " Heat Recovery Outlet Node",
            Exhaust_Air_Inlet_Node_Name=zone.Name + " Exhaust Node",
            Exhaust_Air_Outlet_Node_Name=zone.Name
            + " Heat Recovery Secondary Outlet Node",
            Nominal_Electric_Power=50.0,
            Supply_Air_Outlet_Temperature_Control="No",
            Heat_Exchanger_Type="Rotary",
            Frost_Control_Type="MinimumExhaustTemperature",
            Threshold_Temperature=1.7,
        )

        idf.newidfobject(
            "FAN:ONOFF",
            Name=zone.Name + " RecVent Supply Fan",
            Availability_Schedule_Name="Always 1",
            Fan_Total_Efficiency=building_config.mech_ventilation_fan_efficiency,
            Pressure_Rise=building_config.mech_ventilation_fan_pressure_rise,
            Maximum_Flow_Rate="autosize",
            Motor_Efficiency=0.9,
            Motor_In_Airstream_Fraction=1.0,
            Air_Inlet_Node_Name=zone.Name + " Heat Recovery Outlet Node",
            Air_Outlet_Node_Name=zone.Name + " RecVent Supply Fan Outlet Node",
        )

        idf.newidfobject(
            "FAN:ONOFF",
            Name=zone.Name + " RecVent Exhaust Fan",
            Availability_Schedule_Name="Always 1",
            Fan_Total_Efficiency=building_config.mech_ventilation_fan_efficiency,
            Pressure_Rise=building_config.mech_ventilation_fan_pressure_rise,
            Maximum_Flow_Rate="autosize",
            Motor_Efficiency=0.9,
            Motor_In_Airstream_Fraction=1.0,
            Air_Inlet_Node_Name=zone.Name + " Heat Recovery Secondary Outlet Node",
            Air_Outlet_Node_Name=zone.Name + " RecVent Exhaust Fan Outlet Node",
        )

    return idf
