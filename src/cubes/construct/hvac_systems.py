"""This module adds hvac systems to an idf file"""

from geomeppy import IDF
from cubes.construct.buildingconfig import BuildingConfig
from cubes.construct import buildingconfig_options as bco


def add_heating_system(idf: IDF, building_config: BuildingConfig, heated_zones):

    idf.newidfobject("ScheduleTypeLimits".upper(), Name="Limits Any Number")

    idf.newidfobject(
        "Schedule:Compact".upper(),
        Name="Always 4",
        Schedule_Type_Limits_Name="Limits Any Number",
        Field_1="Through: 12/31,  For: AllDays,   Until: 24:00, 4",
    )

    idf.newidfobject(
        "Schedule:Compact".upper(),
        Name="Always 1",
        Field_1="Through: 12/31,   For: AllDays,   Until: 24:00,    1",
    )

    idf.newidfobject(
        "SCHEDULE:COMPACT",
        Name="45degrees",
        Schedule_Type_Limits_Name="Any Number",
        Field_1="Through: 12/31,   For: AllDays,      Until: 24:00, 45",
    )

    idf.idfobjects["SIMULATIONCONTROL"][0].Do_Zone_Sizing_Calculation = "Yes"

    for zone in heated_zones:
        idf.newidfobject(
            "SCHEDULE:COMPACT",
            Name=zone.Name + "-Heating-Setpoints",
            Field_1="Through: 12/31,\n    For: AllDays,\n    Until: 24:00, 20.\n",
        )
        idf.newidfobject(
            "SCHEDULE:COMPACT",
            Name=zone.Name + "-Cooling-Setpoints",
            Field_1="Through: 12/31,\n    For: AllDays,\n    Until: 24:00, 25.\n",
        )

        idf.newidfobject(
            "ThermostatSetpoint:DualSetpoint".upper(),
            Name=zone.Name + "-Thermostat Dual SP Control",
            Heating_Setpoint_Temperature_Schedule_Name=zone.Name + "-Heating-Setpoints",
            Cooling_Setpoint_Temperature_Schedule_Name=zone.Name + "-Cooling-Setpoints",
        )

        idf.newidfobject(
            "ZoneControl:Thermostat".upper(),
            Name=zone.Name + "-Thermostat",
            Zone_or_ZoneList_Name=zone.Name,
            Control_Type_Schedule_Name="Always 4",
            Control_1_Object_Type="ThermostatSetpoint:DualSetpoint",
            Control_1_Name=zone.Name + "-Thermostat Dual SP Control",
        )

    idf = add_equipment_efficiency_curves(idf, building_config)

    idf = add_supply_side_of_all_loops(idf, building_config, heated_zones)
    if building_config.heating_water_loop_equipment:
        idf = add_heating_water_loops_demand_side(idf, building_config, heated_zones)

    if building_config.zoning == bco.Zoning.RESIDENTIAL_DWELLING.value:
        for zone in heated_zones:
            if zone.Name == "Living":
                dhw_zones = [zone]
                break
    else:
        dhw_zones = heated_zones

    if building_config.dhw_heating_equipment:
        idf = add_dhw_loops_demand_side(idf, building_config, dhw_zones)

    return idf


def get_heating_loop_names(building_config: BuildingConfig, heated_zones):
    if not building_config.heating_water_loop_equipment:
        return []
    loop_names = []
    if building_config.heating_water_loop_dimension == "building":
        loop_names.append("Main")
    else:
        for hz in heated_zones:
            loop_names.append(hz.Name)
    return loop_names


def get_dhw_loop_names(building_config: BuildingConfig, heated_zones):
    if not building_config.dhw_heating_equipment:
        return []
    loop_names = []
    if building_config.dhw_heating_loop_dimension == "building":
        loop_names.append("DHW Main")
    else:
        for hz in heated_zones:
            loop_names.append("DHW " + hz.Name)
    return loop_names


def add_supply_side_of_all_loops(
    idf: IDF, building_config: BuildingConfig, heated_zones
):
    for loop in get_heating_loop_names(building_config, heated_zones):
        idf = add_supply_side(
            idf,
            loop,
            building_config.heating_water_loop_equipment,
            building_config.heating_water_loop_equipment_fuel,
            building_config.heating_water_loop_equipment_efficiency,
            building_config.heating_water_loop_temperature,
            building_config.zone_heating_equipment,
            pump_needed=False,
        )

    for loop in get_dhw_loop_names(building_config, heated_zones):
        idf = add_supply_side(
            idf,
            loop,
            building_config.dhw_heating_equipment,
            building_config.dhw_heating_equipment_fuel,
            building_config.dhw_heating_equipment_efficiency,
            45,
            pump_needed=False,
        )

    return idf


def add_supply_side(
    idf: IDF,
    loop_name,
    equipment,
    fuel,
    efficiency,
    temperature,
    zone_heating_equipment="",
    pump_needed=True,
):

    idf.newidfobject(
        "PLANTLOOP",
        Name=loop_name + " Hot Water Loop",
        Fluid_Type="Water",
        User_Defined_Fluid_Type="",
        Plant_Equipment_Operation_Scheme_Name=loop_name + " Hot Water Loop Operation",
        Loop_Temperature_Setpoint_Node_Name=loop_name + " Hot Water Loop Supply Outlet",
        Maximum_Loop_Temperature=100,
        Minimum_Loop_Temperature=10,
        Maximum_Loop_Flow_Rate="autosize",
        Minimum_Loop_Flow_Rate=0,
        Plant_Loop_Volume="autosize",
        Plant_Side_Inlet_Node_Name=loop_name + " Hot Water Loop Supply Inlet",
        Plant_Side_Outlet_Node_Name=loop_name + " Hot Water Loop Supply Outlet",
        Plant_Side_Branch_List_Name=loop_name + " Hot Water Loop Supply Side Branches",
        Plant_Side_Connector_List_Name=loop_name
        + " Hot Water Loop Supply Side Connectors",
        Demand_Side_Inlet_Node_Name=loop_name + " Hot Water Loop Demand Inlet",
        Demand_Side_Outlet_Node_Name=loop_name + " Hot Water Loop Demand Outlet",
        Demand_Side_Branch_List_Name=loop_name + " Hot Water Loop Demand Side Branches",
        Demand_Side_Connector_List_Name=loop_name
        + " Hot Water Loop Demand Side Connectors",
        Load_Distribution_Scheme="SequentialLoad",
        Availability_Manager_List_Name="",
        Plant_Loop_Demand_Calculation_Scheme="SingleSetpoint",
    )

    idf.newidfobject(
        "ConnectorList".upper(),
        Name=loop_name + " Hot Water Loop Supply Side Connectors",
        Connector_1_Object_Type="Connector:Splitter",
        Connector_1_Name=loop_name + " Hot Water Loop Supply Splitter",
        Connector_2_Object_Type="Connector:Mixer",
        Connector_2_Name=loop_name + " Hot Water Loop Supply Mixer",
    )

    idf.newidfobject(
        "BranchList".upper(),
        Name=loop_name + " Hot Water Loop Supply Side Branches",
        Branch_1_Name=loop_name + " Hot Water Loop Supply Inlet Branch",
        Branch_2_Name=loop_name + " Boiler Branch",
        Branch_3_Name=loop_name + " Hot Water Loop Supply Bypass Branch",
        Branch_4_Name=loop_name + " Hot Water Loop Supply Outlet Branch",
    )

    idf.newidfobject(
        "Connector:Splitter".upper(),
        Name=loop_name + " Hot Water Loop Supply Splitter",
        Inlet_Branch_Name=loop_name + " Hot Water Loop Supply Inlet Branch",
        Outlet_Branch_1_Name=loop_name + " Boiler Branch",
        Outlet_Branch_2_Name=loop_name + " Hot Water Loop Supply Bypass Branch",
    )

    idf.newidfobject(
        "Connector:Mixer".upper(),
        Name=loop_name + " Hot Water Loop Supply Mixer",
        Outlet_Branch_Name=loop_name + " Hot Water Loop Supply Outlet Branch",
        Inlet_Branch_1_Name=loop_name + " Boiler Branch",
        Inlet_Branch_2_Name=loop_name + " Hot Water Loop Supply Bypass Branch",
    )

    if equipment in [
        "condensing boiler",
        "non-condensing boiler",
    ]:

        idf.newidfobject(
            "Boiler:HotWater".upper(),
            Name=loop_name + " Boiler",
            Fuel_Type=fuel,
            Nominal_Capacity="autosize",
            Nominal_Thermal_Efficiency=(efficiency),
            Efficiency_Curve_Temperature_Evaluation_Variable="EnteringBoiler",
            Normalized_Boiler_Efficiency_Curve_Name="Boiler Efficiency Curve",
            Design_Water_Flow_Rate="autosize",
            Minimum_Part_Load_Ratio=0,
            Maximum_Part_Load_Ratio=1.1,
            Optimum_Part_Load_Ratio=1,
            Boiler_Water_Inlet_Node_Name=loop_name + " Boiler Inlet",
            Boiler_Water_Outlet_Node_Name=loop_name + " Boiler Outlet",
            Water_Outlet_Upper_Temperature_Limit=100,
            Boiler_Flow_Mode="ConstantFlow",
            Parasitic_Electric_Load=0,
            Sizing_Factor=1,
        )

        idf.newidfobject(
            "Branch".upper(),
            Name=loop_name + " Boiler Branch",
            Pressure_Drop_Curve_Name="",
            Component_1_Object_Type="Boiler:HotWater",
            Component_1_Name=loop_name + " Boiler",
            Component_1_Inlet_Node_Name=loop_name + " Boiler Inlet",
            Component_1_Outlet_Node_Name=loop_name + " Boiler Outlet",
        )

        idf.newidfobject(
            "PlantEquipmentList".upper(),
            Name=loop_name + " Hot Water Loop All Equipment",
            Equipment_1_Object_Type="Boiler:HotWater",
            Equipment_1_Name=loop_name + " Boiler",
        )

        if zone_heating_equipment == "water-to-air heat pump (water loop source)":
            idf.newidfobject(
                "COOLINGTOWER:SINGLESPEED",
                Name=loop_name + " Tower",
                Water_Inlet_Node_Name=loop_name + " Tower Inlet",
                Water_Outlet_Node_Name=loop_name + " Tower Outlet",
                Design_Water_Flow_Rate="autosize",
                Design_Air_Flow_Rate="autosize",
                Design_Fan_Power="autosize",
                Design_UFactor_Times_Area_Value="autosize",
                Free_Convection_Regime_Air_Flow_Rate="autocalculate",
                Free_Convection_Regime_Air_Flow_Rate_Sizing_Factor="",
                Free_Convection_Regime_UFactor_Times_Area_Value="autocalculate",
                Free_Convection_UFactor_Times_Area_Value_Sizing_Factor="",
                Performance_Input_Method="UFactorTimesAreaAndDesignWaterFlowRate",
                Heat_Rejection_Capacity_and_Nominal_Capacity_Sizing_Ratio="",
                Nominal_Capacity="",
                Free_Convection_Capacity="autocalculate",
                Free_Convection_Nominal_Capacity_Sizing_Factor="",
                Design_Inlet_Air_DryBulb_Temperature="",
                Design_Inlet_Air_WetBulb_Temperature="",
                Design_Approach_Temperature="",
                Design_Range_Temperature="",
                Basin_Heater_Capacity="",
                Basin_Heater_Setpoint_Temperature="",
                Basin_Heater_Operating_Schedule_Name="",
                Evaporation_Loss_Mode="SaturatedExit",
                Evaporation_Loss_Factor="",
                Drift_Loss_Percent=0.008,
                Blowdown_Calculation_Mode="ConcentrationRatio",
                Blowdown_Concentration_Ratio=3,
                Blowdown_Makeup_Water_Usage_Schedule_Name="",
                Supply_Water_Storage_Tank_Name="",
                Outdoor_Air_Inlet_Node_Name=loop_name
                + " Tower Cooling Tower Outdoor Air Inlet Node",
                Capacity_Control="FanCycling",
                Number_of_Cells="",
                Cell_Control="",
                Cell_Minimum_Water_Flow_Rate_Fraction="",
                Cell_Maximum_Water_Flow_Rate_Fraction="",
                Sizing_Factor=1.2,
            )

            idf.newidfobject(
                "OUTDOORAIR:NODE",
                Name=loop_name + " Tower Cooling Tower Outdoor Air Inlet Node",
                Height_Above_Ground=-1,
            )

            idf.newidfobject(
                "BRANCH",
                Name=loop_name + " Tower Branch",
                Pressure_Drop_Curve_Name="",
                Component_1_Object_Type="CoolingTower:SingleSpeed",
                Component_1_Name=loop_name + " Tower",
                Component_1_Inlet_Node_Name=loop_name + " Tower Inlet",
                Component_1_Outlet_Node_Name=loop_name + " Tower CndW Outlet",
            )

            supply_side_branches = idf.idfobjects["BranchList".upper()][-1]
            supply_side_branches.Branch_4_Name = loop_name + " Tower Branch"
            supply_side_branches.Branch_5_Name = (
                loop_name + " Hot Water Loop Supply Outlet Branch"
            )

            supply_side_splitter = idf.idfobjects["Connector:Splitter".upper()][-1]
            supply_side_splitter.Outlet_Branch_3_Name = loop_name + " Tower Branch"

            supply_side_mixer = idf.idfobjects["Connector:Mixer".upper()][-1]
            supply_side_mixer.Inlet_Branch_3_Name = loop_name + " Tower Branch"

            idf.newidfobject(
                "PLANTEQUIPMENTOPERATION:COOLINGLOAD",
                Name=loop_name + " Water Loop Cool Operation All Hours",
                Load_Range_1_Lower_Limit=0,
                Load_Range_1_Upper_Limit=1000000000000000,
                Range_1_Equipment_List_Name=loop_name
                + " Water Loop All Cooling Equipment",
            )

            idf.newidfobject(
                "PLANTEQUIPMENTLIST",
                Name=loop_name + " Water Loop All Cooling Equipment",
                Equipment_1_Object_Type="CoolingTower:SingleSpeed",
                Equipment_1_Name="Main Tower",
            )

    elif equipment == "air-to-water heat pump":

        idf.newidfobject(
            "HEATPUMP:PLANTLOOP:EIR:HEATING",
            Name=loop_name + " Heat Pump HW",
            Load_Side_Inlet_Node_Name=loop_name + " Boiler Inlet",
            Load_Side_Outlet_Node_Name=loop_name + " Boiler Outlet",
            Condenser_Type="AirSource",
            Source_Side_Inlet_Node_Name=loop_name + " Outdoor Air Heat Pump HW Inlet",
            Source_Side_Outlet_Node_Name=loop_name + " Outdoor Air Heat Pump HW Outlet",
            Companion_Heat_Pump_Name="",
            Reference_Coefficient_of_Performance=(efficiency),
            Capacity_Modifier_Function_of_Temperature_Curve_Name="CapCurveFuncTemp",
            Load_Side_Reference_Flow_Rate=0.0255,
            Reference_Capacity=10000,
        )
        heatpump_obj = idf.idfobjects["HEATPUMP:PLANTLOOP:EIR:HEATING"][-1]
        setattr(
            heatpump_obj,
            (
                "Electric_Input_to_Output_Ratio_Modifier_"
                "Function_of_Temperature_Curve_Name"
            ),
            "EIRCurveFuncTemp",
        )
        setattr(
            heatpump_obj,
            (
                "Electric_Input_to_Output_Ratio_Modifier_Function"
                "_of_Part_Load_Ratio_Curve_Name"
            ),
            "EIRCurveFuncPLR",
        )

        idf.newidfobject(
            "OUTDOORAIR:NODELIST",
            Node_or_NodeList_Name_1=loop_name + " Outdoor Air Heat Pump HW Inlet",
            Node_or_NodeList_Name_2=loop_name + " Outdoor Air Heat Pump HW Outlet",
        )

        idf.newidfobject(
            "Branch".upper(),
            Name=loop_name + " Boiler Branch",
            Pressure_Drop_Curve_Name="",
            Component_1_Object_Type="HEATPUMP:PLANTLOOP:EIR:HEATING",
            Component_1_Name=loop_name + " Heat Pump HW",
            Component_1_Inlet_Node_Name=loop_name + " Boiler Inlet",
            Component_1_Outlet_Node_Name=loop_name + " Boiler Outlet",
        )

        idf.newidfobject(
            "PlantEquipmentList".upper(),
            Name=loop_name + " Hot Water Loop All Equipment",
            Equipment_1_Object_Type="HEATPUMP:PLANTLOOP:EIR:HEATING",
            Equipment_1_Name=loop_name + " Heat Pump HW",
        )

    if zone_heating_equipment == "water-to-air heat pump (water loop source)":

        idf.idfobjects["PLANTLOOP"][
            -1
        ].Plant_Loop_Demand_Calculation_Scheme = "DualSetPointDeadband"

        idf.newidfobject(
            "SIZING:PLANT",
            Plant_or_Condenser_Loop_Name=loop_name + " Hot Water Loop",
            Loop_Type="Condenser",
            Design_Loop_Exit_Temperature=34,
            Loop_Design_Temperature_Difference=6,
        )

        idf.newidfobject(
            "PLANTEQUIPMENTOPERATIONSCHEMES",
            Name="Main Hot Water Loop Operation",
            Control_Scheme_1_Object_Type="PlantEquipmentOperation:HeatingLoad",
            Control_Scheme_1_Name=loop_name + " Hot Water Loop Operation All Hours",
            Control_Scheme_1_Schedule_Name="Always 1",
            Control_Scheme_2_Object_Type="PlantEquipmentOperation:CoolingLoad",
            Control_Scheme_2_Name=loop_name + " Water Loop Cool Operation All Hours",
            Control_Scheme_2_Schedule_Name="Always 1",
        )

        idf.newidfobject(
            "NODELIST",
            Name="Only Water Loop Mixed Supply Setpoint Nodes",
            Node_1_Name="Main Boiler Outlet",
            Node_2_Name="Main Tower Outlet",
            Node_3_Name="Main Hot Water Loop Supply Outlet",
        )

        idf.newidfobject(
            "Schedule:Compact".upper(),
            Name="Always 34",
            Schedule_Type_Limits_Name="Limits Any Number",
            Field_1="Through: 12/31,  For: AllDays,   Until: 24:00, 34",
        )

        idf.newidfobject(
            "Schedule:Compact".upper(),
            Name="Always 20",
            Schedule_Type_Limits_Name="Limits Any Number",
            Field_1="Through: 12/31,  For: AllDays,   Until: 24:00, 20",
        )

        idf.newidfobject(
            "SETPOINTMANAGER:SCHEDULED:DUALSETPOINT",
            Name="Only Water Loop Mixed Temp Manager",
            Control_Variable="Temperature",
            High_Setpoint_Schedule_Name="Always 34",
            Low_Setpoint_Schedule_Name="Always 20",
            Setpoint_Node_or_NodeList_Name=(
                "Only Water Loop Mixed Supply Setpoint Nodes"
            ),
        )

    else:

        idf.newidfobject(
            "Sizing:Plant".upper(),
            Plant_or_Condenser_Loop_Name=loop_name + " Hot Water Loop",
            Loop_Type="Heating",
            Design_Loop_Exit_Temperature=(temperature),
            Loop_Design_Temperature_Difference=11,
        )
        idf.newidfobject(
            "PlantEquipmentOperationSchemes".upper(),
            Name=loop_name + " Hot Water Loop Operation",
            Control_Scheme_1_Object_Type="PlantEquipmentOperation:HeatingLoad",
            Control_Scheme_1_Name=loop_name + " Hot Water Loop Operation All Hours",
            Control_Scheme_1_Schedule_Name="Always 1",
        )

        idf.newidfobject(
            "NodeList".upper(),
            Name=loop_name + " Hot Water Loop Supply Setpoint Nodes",
            Node_1_Name=loop_name + " Boiler Outlet",
            Node_2_Name=loop_name + " Hot Water Loop Supply Outlet",
        )

        idf.newidfobject(
            "SetpointManager:Scheduled".upper(),
            Name=loop_name + " Hot Water Loop Temp Manager",
            Control_Variable="Temperature",
            Schedule_Name=loop_name + " Hot Water Loop Temperature Schedule",
            Setpoint_Node_or_NodeList_Name=loop_name
            + " Hot Water Loop Supply Setpoint Nodes",
        )

        idf.newidfobject(
            "Schedule:Compact".upper(),
            Name=loop_name + " Hot Water Loop Temperature Schedule",
            Field_1=(
                f"Through: 12/31,   For: AllDays,    Until: 24:00," f"  {temperature}"
            ),
        )

    idf.newidfobject(
        "PlantEquipmentOperation:HeatingLoad".upper(),
        Name=loop_name + " Hot Water Loop Operation All Hours",
        Load_Range_1_Lower_Limit=0,
        Load_Range_1_Upper_Limit=1000000000000000,
        Range_1_Equipment_List_Name=loop_name + " Hot Water Loop All Equipment",
    )

    idf.newidfobject(
        "BRANCH",
        Name=loop_name + " Hot Water Loop Supply Bypass Branch",
        Component_1_Object_Type="Pipe:Adiabatic",
        Component_1_Name=loop_name + " Hot Water Loop Supply Side Bypass Pipe",
        Component_1_Inlet_Node_Name=loop_name + " Hot Water Loop Supply Bypass Inlet",
        Component_1_Outlet_Node_Name=loop_name + " Hot Water Loop Supply Bypass Outlet",
    )

    idf.newidfobject(
        "Pipe:Adiabatic".upper(),
        Name=loop_name + " Hot Water Loop Supply Side Bypass Pipe",
        Inlet_Node_Name=loop_name + " Hot Water Loop Supply Bypass Inlet",
        Outlet_Node_Name=loop_name + " Hot Water Loop Supply Bypass Outlet",
    )

    idf.newidfobject(
        "BRANCH",
        Name=loop_name + " Hot Water Loop Supply Inlet Branch",
        Component_1_Object_Type="Pump:ConstantSpeed",
        Component_1_Name=loop_name + " Hot Water Loop Supply Pump",
        Component_1_Inlet_Node_Name=loop_name + " Hot Water Loop Supply Inlet",
        Component_1_Outlet_Node_Name=loop_name + " Hot Water Loop Pump Outlet",
    )

    pump_head = 179352  # energyplus defaults
    if not pump_needed:
        pump_head = 0

    idf.newidfobject(
        "PUMP:CONSTANTSPEED",
        Name=loop_name + " Hot Water Loop Supply Pump",
        Inlet_Node_Name=loop_name + " Hot Water Loop Supply Inlet",
        Outlet_Node_Name=loop_name + " Hot Water Loop Pump Outlet",
        Design_Flow_Rate="autosize",
        Design_Pump_Head=pump_head,
        Design_Power_Consumption="autosize",
        Motor_Efficiency=0.9,
        Fraction_of_Motor_Inefficiencies_to_Fluid_Stream=0,
        Pump_Control_Type="Intermittent",
        Pump_Flow_Rate_Schedule_Name="",
        # Design_Electric_Power_per_Unit_Flow_Rate=pump_power_per_flow_rate,
    )

    idf.newidfobject(
        "Branch".upper(),
        Name=loop_name + " Hot Water Loop Supply Outlet Branch",
        Component_1_Object_Type="Pipe:Adiabatic",
        Component_1_Name=loop_name + " Hot Water Loop Supply Outlet Pipe",
        Component_1_Inlet_Node_Name=loop_name
        + " Hot Water Loop Supply Outlet Pipe Inlet",
        Component_1_Outlet_Node_Name=loop_name + " Hot Water Loop Supply Outlet",
    )

    idf.newidfobject(
        "Pipe:Adiabatic".upper(),
        Name=loop_name + " Hot Water Loop Supply Outlet Pipe",
        Inlet_Node_Name=loop_name + " Hot Water Loop Supply Outlet Pipe Inlet",
        Outlet_Node_Name=loop_name + " Hot Water Loop Supply Outlet",
    )

    return idf


def add_demand_side_standard_parts(idf: IDF, loop_name):
    idf.newidfobject(
        "ConnectorList".upper(),
        Name=loop_name + " Hot Water Loop Demand Side Connectors",
        Connector_1_Object_Type="Connector:Splitter",
        Connector_1_Name=loop_name + " Hot Water Loop Demand Splitter",
        Connector_2_Object_Type="Connector:Mixer",
        Connector_2_Name=loop_name + " Hot Water Loop Demand Mixer",
    )

    idf.newidfobject(
        "Branch".upper(),
        Name=loop_name + " Hot Water Loop Demand Inlet Branch",
        Component_1_Object_Type="Pipe:Adiabatic",
        Component_1_Name=loop_name + " Hot Water Loop Demand Inlet Pipe",
        Component_1_Inlet_Node_Name=loop_name + " Hot Water Loop Demand Inlet",
        Component_1_Outlet_Node_Name=loop_name
        + " Hot Water Loop Demand Inlet Pipe Outlet",
    )

    idf.newidfobject(
        "Pipe:Adiabatic".upper(),
        Name=loop_name + " Hot Water Loop Demand Inlet Pipe",
        Inlet_Node_Name=loop_name + " Hot Water Loop Demand Inlet",
        Outlet_Node_Name=loop_name + " Hot Water Loop Demand Inlet Pipe Outlet",
    )

    idf.newidfobject(
        "BRANCH",
        Name=loop_name + " Hot Water Loop Demand Bypass Branch",
        Component_1_Object_Type="Pipe:Adiabatic",
        Component_1_Name=loop_name + " Hot Water Loop Demand Side Bypass Pipe",
        Component_1_Inlet_Node_Name=loop_name + " Hot Water Loop Demand Bypass Inlet",
        Component_1_Outlet_Node_Name=loop_name + " Hot Water Loop Demand Bypass Outlet",
    )

    idf.newidfobject(
        "Pipe:Adiabatic".upper(),
        Name=loop_name + " Hot Water Loop Demand Side Bypass Pipe",
        Inlet_Node_Name=loop_name + " Hot Water Loop Demand Bypass Inlet",
        Outlet_Node_Name=loop_name + " Hot Water Loop Demand Bypass Outlet",
    )

    idf.newidfobject(
        "Branch".upper(),
        Name=loop_name + " Hot Water Loop Demand Outlet Branch",
        Component_1_Object_Type="Pipe:Adiabatic",
        Component_1_Name=loop_name + " Hot Water Loop Demand Outlet Pipe",
        Component_1_Inlet_Node_Name=loop_name
        + " Hot Water Loop Demand Outlet Pipe Inlet",
        Component_1_Outlet_Node_Name=loop_name + " Hot Water Loop Demand Outlet",
    )

    idf.newidfobject(
        "Pipe:Adiabatic".upper(),
        Name=loop_name + " Hot Water Loop Demand Outlet Pipe",
        Inlet_Node_Name=loop_name + " Hot Water Loop Demand Outlet Pipe Inlet",
        Outlet_Node_Name=loop_name + " Hot Water Loop Demand Outlet",
    )


def add_dhw_loops_demand_side(idf: IDF, building_config: BuildingConfig, dhw_zones):
    if building_config.heating_water_loop_dimension == "building":
        # one boiler and one branch per zone on demand side

        n_zones = len(dhw_zones)

        loop_names = get_dhw_loop_names(building_config, dhw_zones)

        for ln in loop_names:
            add_demand_side_standard_parts(idf, ln)

        demand_side_branch_list = idf.newidfobject(
            "BRANCHLIST",
            Name="DHW Main Hot Water Loop Demand Side Branches",
            Branch_1_Name="DHW Main Hot Water Loop Demand Inlet Branch",
            Branch_2_Name="DHW Main Hot Water Loop Demand Bypass Branch",
        )
        setattr(
            demand_side_branch_list,
            "Branch_" + str(n_zones + 3) + "_Name",
            "DHW Main Hot Water Loop Demand Outlet Branch",
        )

        demand_side_splitter = idf.newidfobject(
            "Connector:Splitter".upper(),
            Name="DHW Main Hot Water Loop Demand Splitter",
            Inlet_Branch_Name="DHW Main Hot Water Loop Demand Inlet Branch",
            Outlet_Branch_1_Name="DHW Main Hot Water Loop Demand Bypass Branch",
        )

        demand_side_mixer = idf.newidfobject(
            "Connector:Mixer".upper(),
            Name="DHW Main Hot Water Loop Demand Mixer",
            Outlet_Branch_Name="DHW Main Hot Water Loop Demand Outlet Branch",
            Inlet_Branch_1_Name="DHW Main Hot Water Loop Demand Bypass Branch",
        )

        for iz, zone in enumerate(dhw_zones):

            idf, dhw_branch_name = add_dhw_branch_and_tank(idf, building_config, zone)

            setattr(
                demand_side_branch_list,
                "Branch_" + str(3 + iz) + "_Name",
                dhw_branch_name,
            )

            setattr(
                demand_side_splitter,
                "Outlet_Branch_" + str(iz + 2) + "_Name",
                dhw_branch_name,
            )

            setattr(
                demand_side_mixer,
                "Inlet_Branch_" + str(iz + 2) + "_Name",
                dhw_branch_name,
            )

    elif building_config.heating_water_loop_dimension == "zone":

        for iz, zone in enumerate(dhw_zones):

            idf, dhw_branch_name = add_dhw_branch_and_tank(idf, building_config, zone)

            idf.newidfobject(
                "BRANCHLIST",
                Name=zone.Name + " Hot Water Loop Demand Side Branches",
                Branch_1_Name=zone.Name + " Hot Water Loop Demand Inlet Branch",
                Branch_2_Name=zone.Name + " Hot Water Loop Demand Bypass Branch",
                Branch_3_Name=dhw_branch_name,
                Branch_4_Name=zone.Name + " Hot Water Loop Demand Outlet Branch",
            )

            idf.newidfobject(
                "Connector:Splitter".upper(),
                Name=zone.Name + " Hot Water Loop Demand Splitter",
                Inlet_Branch_Name=zone.Name + " Hot Water Loop Demand Inlet Branch",
                Outlet_Branch_1_Name=zone.Name + " Hot Water Loop Demand Bypass Branch",
                Outlet_Branch_2_Name=dhw_branch_name,
            )

            idf.newidfobject(
                "Connector:Mixer".upper(),
                Name=zone.Name + " Hot Water Loop Demand Mixer",
                Outlet_Branch_Name=zone.Name + " Hot Water Loop Demand Outlet Branch",
                Inlet_Branch_1_Name=zone.Name + " Hot Water Loop Demand Bypass Branch",
                Inlet_Branch_2_Name=dhw_branch_name,
            )
    return idf


def add_heating_water_loops_demand_side(
    idf: IDF, building_config: BuildingConfig, heated_zones
):
    loop_names = get_heating_loop_names(building_config, heated_zones)

    for ln in loop_names:
        add_demand_side_standard_parts(idf, ln)

    if building_config.zone_heating_equipment == "radiator":
        idf.newidfobject(
            "ZoneHVAC:Baseboard:RadiantConvective:Water:Design".upper(),
            Name="Baseboard Heat Design",
            Heating_Design_Capacity_Method="HeatingDesignCapacity",
            Convergence_Tolerance=0.001,
            Fraction_Radiant=0.0,
            Fraction_of_Radiant_Energy_Incident_on_People=0.0,
        )

        for zone in heated_zones:
            idf.newidfobject(
                "Sizing:Zone".upper(),
                Zone_or_ZoneList_Name=zone.Name,
                Zone_Cooling_Design_Supply_Air_Temperature_Input_Method=(
                    "SupplyAirTemperature"
                ),
                Zone_Cooling_Design_Supply_Air_Temperature=12.5,
                Zone_Cooling_Design_Supply_Air_Temperature_Difference="",
                Zone_Heating_Design_Supply_Air_Temperature_Input_Method=(
                    "SupplyAirTemperature"
                ),
                Zone_Heating_Design_Supply_Air_Temperature=50,
                Zone_Heating_Design_Supply_Air_Temperature_Difference="",
                Zone_Cooling_Design_Supply_Air_Humidity_Ratio=0.008,
                Zone_Heating_Design_Supply_Air_Humidity_Ratio=0.008,
                Design_Specification_Outdoor_Air_Object_Name="",
                Zone_Heating_Sizing_Factor="",
                Zone_Cooling_Sizing_Factor="",
                Cooling_Design_Air_Flow_Method="DesignDay",
                Cooling_Design_Air_Flow_Rate=0,
                Cooling_Minimum_Air_Flow_per_Zone_Floor_Area="",
                Cooling_Minimum_Air_Flow="",
                Cooling_Minimum_Air_Flow_Fraction=0,
                Heating_Design_Air_Flow_Method="DesignDay",
                Heating_Design_Air_Flow_Rate=0,
                Heating_Maximum_Air_Flow_per_Zone_Floor_Area="",
                Heating_Maximum_Air_Flow="",
                Heating_Maximum_Air_Flow_Fraction=0,
                Design_Specification_Zone_Air_Distribution_Object_Name="",
            )

            idf.newidfobject(
                "ZoneHVAC:EquipmentConnections".upper(),
                Zone_Name=zone.Name,
                Zone_Conditioning_Equipment_List_Name=zone.Name + "-Equipment",
                Zone_Air_Inlet_Node_or_NodeList_Name="",
                Zone_Air_Exhaust_Node_or_NodeList_Name="",
                Zone_Air_Node_Name=zone.Name + "-Zone Air Node",
                Zone_Return_Air_Node_or_NodeList_Name=zone.Name + "-Return Outlet",
            )

            idf.newidfobject(
                "ZoneHVAC:EquipmentList".upper(),
                Name=zone.Name + "-Equipment",
                Load_Distribution_Scheme="SequentialLoad",
                Zone_Equipment_1_Object_Type=(
                    "ZoneHVAC:Baseboard:RadiantConvective:Water"
                ),
                Zone_Equipment_1_Name=zone.Name + "-Baseboard Heat",
                Zone_Equipment_1_Cooling_Sequence=1,
                Zone_Equipment_1_Heating_or_NoLoad_Sequence=1,
            )

            idf.newidfobject(
                "ZoneHVAC:Baseboard:RadiantConvective:Water".upper(),
                Name=zone.Name + "-Baseboard Heat",
                Design_Object="Baseboard Heat Design",
                Availability_Schedule_Name="",
                Inlet_Node_Name=zone.Name + "-Baseboard Inlet",
                Outlet_Node_Name=zone.Name + "-Baseboard Outlet",
                Rated_Average_Water_Temperature=(
                    building_config.heating_water_loop_temperature
                ),
                Rated_Water_Mass_Flow_Rate=0.063,
                Heating_Design_Capacity="autosize",
                Maximum_Water_Flow_Rate="autosize",
            )

            idf.newidfobject(
                "Branch".upper(),
                Name=zone.Name + "-Heating Branch",
                Pressure_Drop_Curve_Name="",
                Component_1_Object_Type="ZoneHVAC:Baseboard:RadiantConvective:Water",
                Component_1_Name=zone.Name + "-Baseboard Heat",
                Component_1_Inlet_Node_Name=zone.Name + "-Baseboard Inlet",
                Component_1_Outlet_Node_Name=zone.Name + "-Baseboard Outlet",
            )

        if building_config.heating_water_loop_dimension == "building":
            # one boiler and one branch per zone on demand side

            n_zones = len(heated_zones)

            demand_side_branch_list = idf.newidfobject(
                "BRANCHLIST",
                Name="Main Hot Water Loop Demand Side Branches",
                Branch_1_Name="Main Hot Water Loop Demand Inlet Branch",
                Branch_2_Name="Main Hot Water Loop Demand Bypass Branch",
            )
            setattr(
                demand_side_branch_list,
                "Branch_" + str(n_zones + 3) + "_Name",
                "Main Hot Water Loop Demand Outlet Branch",
            )

            demand_side_splitter = idf.newidfobject(
                "Connector:Splitter".upper(),
                Name="Main Hot Water Loop Demand Splitter",
                Inlet_Branch_Name="Main Hot Water Loop Demand Inlet Branch",
                Outlet_Branch_1_Name="Main Hot Water Loop Demand Bypass Branch",
            )

            demand_side_mixer = idf.newidfobject(
                "Connector:Mixer".upper(),
                Name="Main Hot Water Loop Demand Mixer",
                Outlet_Branch_Name="Main Hot Water Loop Demand Outlet Branch",
                Inlet_Branch_1_Name="Main Hot Water Loop Demand Bypass Branch",
            )

            for iz, zone in enumerate(heated_zones):

                setattr(
                    demand_side_branch_list,
                    "Branch_" + str(3 + iz) + "_Name",
                    zone.Name + "-Heating Branch",
                )

                setattr(
                    demand_side_splitter,
                    "Outlet_Branch_" + str(iz + 2) + "_Name",
                    zone.Name + "-Heating Branch",
                )

                setattr(
                    demand_side_mixer,
                    "Inlet_Branch_" + str(iz + 2) + "_Name",
                    zone.Name + "-Heating Branch",
                )

        elif building_config.heating_water_loop_dimension == "zone":

            for iz, zone in enumerate(heated_zones):

                idf.newidfobject(
                    "BRANCHLIST",
                    Name=zone.Name + " Hot Water Loop Demand Side Branches",
                    Branch_1_Name=zone.Name + " Hot Water Loop Demand Inlet Branch",
                    Branch_2_Name=zone.Name + " Hot Water Loop Demand Bypass Branch",
                    Branch_3_Name=zone.Name + "-Heating Branch",
                    Branch_4_Name=zone.Name + " Hot Water Loop Demand Outlet Branch",
                )

                idf.newidfobject(
                    "Connector:Splitter".upper(),
                    Name=zone.Name + " Hot Water Loop Demand Splitter",
                    Inlet_Branch_Name=zone.Name + " Hot Water Loop Demand Inlet Branch",
                    Outlet_Branch_1_Name=zone.Name
                    + " Hot Water Loop Demand Bypass Branch",
                    Outlet_Branch_2_Name=zone.Name + "-Heating Branch",
                )

                idf.newidfobject(
                    "Connector:Mixer".upper(),
                    Name=zone.Name + " Hot Water Loop Demand Mixer",
                    Outlet_Branch_Name=zone.Name
                    + " Hot Water Loop Demand Outlet Branch",
                    Inlet_Branch_1_Name=zone.Name
                    + " Hot Water Loop Demand Bypass Branch",
                    Inlet_Branch_2_Name=zone.Name + "-Heating Branch",
                )

    if (
        building_config.heating_water_loop_equipment
        in ["condensing boiler", "non-condensing boiler"]
        and building_config.zone_heating_equipment
        == "water-to-air heat pump (water loop source)"
    ):

        idf.newidfobject(
            "DESIGNSPECIFICATION:OUTDOORAIR",
            Name="Design Spec Outdoor Air",
            Outdoor_Air_Method="Flow/Person",
            Outdoor_Air_Flow_per_Person=0.00944,
            Outdoor_Air_Flow_per_Zone_Floor_Area=0.0,
            Outdoor_Air_Flow_per_Zone=0.0,
        )

        idf.newidfobject(
            "DESIGNSPECIFICATION:ZONEAIRDISTRIBUTION",
            Name="Design Spec Zone Air Distribution",
            Zone_Air_Distribution_Effectiveness_in_Cooling_Mode=1,
            Zone_Air_Distribution_Effectiveness_in_Heating_Mode=1,
        )

        idf.newidfobject(
            "Schedule:Compact".upper(),
            Name="Always 0",
            Schedule_Type_Limits_Name="Limits Any Number",
            Field_1="Through: 12/31,  For: AllDays,   Until: 24:00, 0",
        )

        for iz, zone in enumerate(heated_zones):

            idf.newidfobject(
                "SIZING:ZONE",
                Zone_or_ZoneList_Name=zone.Name,
                Zone_Cooling_Design_Supply_Air_Temperature_Input_Method=(
                    "SupplyAirTemperature"
                ),
                Zone_Cooling_Design_Supply_Air_Temperature=12.5,
                Zone_Cooling_Design_Supply_Air_Temperature_Difference=11.11,
                Zone_Heating_Design_Supply_Air_Temperature_Input_Method=(
                    "SupplyAirTemperature"
                ),
                Zone_Heating_Design_Supply_Air_Temperature=50.0,
                Zone_Heating_Design_Supply_Air_Temperature_Difference="",
                Zone_Cooling_Design_Supply_Air_Humidity_Ratio=0.008,
                Zone_Heating_Design_Supply_Air_Humidity_Ratio=0.008,
                Design_Specification_Outdoor_Air_Object_Name="Design Spec Outdoor Air",
                Zone_Heating_Sizing_Factor=1.2,
                Zone_Cooling_Sizing_Factor=1.2,
                Cooling_Design_Air_Flow_Method="DesignDay",
                Cooling_Design_Air_Flow_Rate=0,
                Cooling_Minimum_Air_Flow_per_Zone_Floor_Area="",
                Cooling_Minimum_Air_Flow="",
                Cooling_Minimum_Air_Flow_Fraction=0,
                Heating_Design_Air_Flow_Method="DesignDay",
                Heating_Design_Air_Flow_Rate=0,
                Heating_Maximum_Air_Flow_per_Zone_Floor_Area="",
                Heating_Maximum_Air_Flow="",
                Heating_Maximum_Air_Flow_Fraction=0,
                Design_Specification_Zone_Air_Distribution_Object_Name=(
                    "Design Spec Zone Air Distribution"
                ),
            )

            idf.newidfobject(
                "ZONEHVAC:EQUIPMENTCONNECTIONS",
                Zone_Name=zone.Name,
                Zone_Conditioning_Equipment_List_Name=zone.Name + " Equipment",
                Zone_Air_Inlet_Node_or_NodeList_Name=zone.Name + " WAHP Supply Inlet",
                Zone_Air_Exhaust_Node_or_NodeList_Name=zone.Name + " WAHP Return",
                Zone_Air_Node_Name=zone.Name + " Zone Air Node",
                Zone_Return_Air_Node_or_NodeList_Name=zone.Name + " Return Outlet",
            )

            idf.newidfobject(
                "ZONEHVAC:EQUIPMENTLIST",
                Name=zone.Name + " Equipment",
                Load_Distribution_Scheme="SequentialLoad",
                Zone_Equipment_1_Object_Type="ZoneHVAC:WaterToAirHeatPump",
                Zone_Equipment_1_Name=zone.Name + " WAHP",
                Zone_Equipment_1_Cooling_Sequence=1,
                Zone_Equipment_1_Heating_or_NoLoad_Sequence=1,
            )

            idf.newidfobject(
                "ZONEHVAC:WATERTOAIRHEATPUMP",
                Name=zone.Name + " WAHP",
                Availability_Schedule_Name="",
                Air_Inlet_Node_Name=zone.Name + " WAHP Return",
                Air_Outlet_Node_Name=zone.Name + " WAHP Supply Inlet",
                Outdoor_Air_Mixer_Object_Type="OutdoorAir:Mixer",
                Outdoor_Air_Mixer_Name=zone.Name + " WAHP OA Mixing Box",
                Cooling_Supply_Air_Flow_Rate="autosize",
                Heating_Supply_Air_Flow_Rate="autosize",
                No_Load_Supply_Air_Flow_Rate="",
                Cooling_Outdoor_Air_Flow_Rate="autosize",
                Heating_Outdoor_Air_Flow_Rate="autosize",
                No_Load_Outdoor_Air_Flow_Rate="autosize",
                Supply_Air_Fan_Object_Type="Fan:OnOff",
                Supply_Air_Fan_Name=zone.Name + " WAHP Supply Fan",
                Heating_Coil_Object_Type="Coil:Heating:WaterToAirHeatPump:EquationFit",
                Heating_Coil_Name=zone.Name + " WAHP Heating Coil",
                Cooling_Coil_Object_Type="Coil:Cooling:WaterToAirHeatPump:EquationFit",
                Cooling_Coil_Name=zone.Name + " WAHP Cooling Coil",
                Maximum_Cycling_Rate=2.5,
                Heat_Pump_Time_Constant=60,
                Fraction_of_OnCycle_Power_Use=0.01,
                Heat_Pump_Fan_Delay_Time=60,
                Supplemental_Heating_Coil_Object_Type="Coil:Heating:Electric",
                Supplemental_Heating_Coil_Name=zone.Name + " WAHP Supp Heating Coil",
                Maximum_Supply_Air_Temperature_from_Supplemental_Heater="autosize",
                Maximum_Outdoor_DryBulb_Temperature_for_Supplemental_Heater_Operation=(
                    20.0
                ),
                Outdoor_DryBulb_Temperature_Sensor_Node_Name=zone.Name
                + " WAHP Outside Air Inlet",
                Fan_Placement="DrawThrough",
                Supply_Air_Fan_Operating_Mode_Schedule_Name="Always 0",
                Availability_Manager_List_Name="",
                Heat_Pump_Coil_Water_Flow_Mode="Cycling",
            )

            idf.newidfobject(
                "FAN:ONOFF",
                Name=zone.Name + " WAHP Supply Fan",
                Availability_Schedule_Name="Always 1",
                Fan_Total_Efficiency=0.7,
                Pressure_Rise=75,
                Maximum_Flow_Rate="autosize",
                Motor_Efficiency=0.9,
                Motor_In_Airstream_Fraction=1,
                Air_Inlet_Node_Name=zone.Name + " WAHP Heating Coil Outlet",
                Air_Outlet_Node_Name=zone.Name + " WAHP Supply Fan Outlet",
            )

            idf.newidfobject(
                "OUTDOORAIR:MIXER",
                Name=zone.Name + " WAHP OA Mixing Box",
                Mixed_Air_Node_Name=zone.Name + " WAHP Mixed Air Outlet",
                Outdoor_Air_Stream_Node_Name=zone.Name + " WAHP Outside Air Inlet",
                Relief_Air_Stream_Node_Name=zone.Name + " WAHP Relief Air Outlet",
                Return_Air_Stream_Node_Name=zone.Name + " WAHP Return",
            )

            idf.newidfobject(
                "OUTDOORAIR:NODE",
                Name=zone.Name + " WAHP Outside Air Inlet",
                Height_Above_Ground=-1,
            )

            idf.newidfobject(
                "COIL:COOLING:WATERTOAIRHEATPUMP:EQUATIONFIT",
                Name=zone.Name + " WAHP Cooling Coil",
                Water_Inlet_Node_Name=zone.Name + " WAHP Cooling Water Inlet Node",
                Water_Outlet_Node_Name=zone.Name + " WAHP Cooling Water Outlet Node",
                Air_Inlet_Node_Name=zone.Name + " WAHP Mixed Air Outlet",
                Air_Outlet_Node_Name=zone.Name + " WAHP Cooling Coil Outlet",
                Rated_Air_Flow_Rate="Autosize",
                Rated_Water_Flow_Rate="Autosize",
                Gross_Rated_Total_Cooling_Capacity="autosize",
                Gross_Rated_Sensible_Cooling_Capacity="Autosize",
                Gross_Rated_Cooling_COP=building_config.cooling_system_efficiency,
                Total_Cooling_Capacity_Curve_Name="WAHP Total Cooling Capacity Curve",
                Sensible_Cooling_Capacity_Curve_Name=(
                    "WAHP Sensible Cooling Capacity Curve"
                ),
                Cooling_Power_Consumption_Curve_Name=(
                    "WAHP Cooling Power Consumption Curve"
                ),
                Nominal_Time_for_Condensate_Removal_to_Begin=0,
            )
            cooling_coil = idf.idfobjects[
                "COIL:COOLING:WATERTOAIRHEATPUMP:EQUATIONFIT"
            ][-1]
            setattr(
                cooling_coil,
                (
                    "Ratio_of_Initial_Moisture_Evaporation_Rate_"
                    "and_Steady_State_Latent_Capacity"
                ),
                0,
            )

            idf.newidfobject(
                "COIL:HEATING:WATERTOAIRHEATPUMP:EQUATIONFIT",
                Name=zone.Name + " WAHP Heating Coil",
                Water_Inlet_Node_Name=zone.Name + " WAHP Heating Water Inlet Node",
                Water_Outlet_Node_Name=zone.Name + " WAHP Heating Water Outlet Node",
                Air_Inlet_Node_Name=zone.Name + " WAHP Cooling Coil Outlet",
                Air_Outlet_Node_Name=zone.Name + " WAHP Heating Coil Outlet",
                Rated_Air_Flow_Rate="Autosize",
                Rated_Water_Flow_Rate="Autosize",
                Gross_Rated_Heating_Capacity="autosize",
                Gross_Rated_Heating_COP=(
                    building_config.heating_water_loop_equipment_efficiency
                ),
                Heating_Capacity_Curve_Name="WAHP Heating Capacity Curve",
                Heating_Power_Consumption_Curve_Name=(
                    "WAHP Heating Power Consumption Curve"
                ),
            )

            idf.newidfobject(
                "COIL:HEATING:ELECTRIC",
                Name=zone.Name + " WAHP Supp Heating Coil",
                Availability_Schedule_Name="",
                Efficiency=1,
                Nominal_Capacity="autosize",
                Air_Inlet_Node_Name=zone.Name + " WAHP Supply Fan Outlet",
                Air_Outlet_Node_Name=zone.Name + " WAHP Supply Inlet",
                Temperature_Setpoint_Node_Name="",
            )

            idf.newidfobject(
                "BRANCH",
                Name=zone.Name + " Cooling Condenser Branch",
                Pressure_Drop_Curve_Name="",
                Component_1_Object_Type="Coil:Cooling:WaterToAirHeatPump:EquationFit",
                Component_1_Name=zone.Name + " WAHP Cooling Coil",
                Component_1_Inlet_Node_Name=zone.Name
                + " WAHP Cooling Water Inlet Node",
                Component_1_Outlet_Node_Name=zone.Name
                + " WAHP Cooling Water Outlet Node",
            )

            idf.newidfobject(
                "BRANCH",
                Name=zone.Name + " Heating Condenser Branch",
                Pressure_Drop_Curve_Name="",
                Component_1_Object_Type="Coil:Heating:WaterToAirHeatPump:EquationFit",
                Component_1_Name=zone.Name + " WAHP Heating Coil",
                Component_1_Inlet_Node_Name=zone.Name
                + " WAHP Heating Water Inlet Node",
                Component_1_Outlet_Node_Name=zone.Name
                + " WAHP Heating Water Outlet Node",
            )

        n_zones = len(heated_zones)

        # wrong branch names here. adapt tp usage above
        demand_side_branch_list = idf.newidfobject(
            "BRANCHLIST",
            Name="Main Hot Water Loop Demand Side Branches",
            Branch_1_Name="Main Hot Water Loop Demand Inlet Branch",
            Branch_2_Name="Main Hot Water Loop Demand Bypass Branch",
        )
        setattr(
            demand_side_branch_list,
            "Branch_" + str(2 * n_zones + 3) + "_Name",
            "Main Hot Water Loop Demand Outlet Branch",
        )

        demand_side_splitter = idf.newidfobject(
            "Connector:Splitter".upper(),
            Name="Main Hot Water Loop Demand Splitter",
            Inlet_Branch_Name="Main Hot Water Loop Demand Inlet Branch",
            Outlet_Branch_1_Name="Main Hot Water Loop Demand Bypass Branch",
        )

        demand_side_mixer = idf.newidfobject(
            "Connector:Mixer".upper(),
            Name="Main Hot Water Loop Demand Mixer",
            Outlet_Branch_Name="Main Hot Water Loop Demand Outlet Branch",
            Inlet_Branch_1_Name="Main Hot Water Loop Demand Bypass Branch",
        )

        for iz, zone in enumerate(heated_zones):

            setattr(
                demand_side_branch_list,
                "Branch_" + str(3 + 2 * iz) + "_Name",
                zone.Name + " Cooling Condenser Branch",
            )
            setattr(
                demand_side_branch_list,
                "Branch_" + str(3 + 2 * iz + 1) + "_Name",
                zone.Name + " Heating Condenser Branch",
            )

            setattr(
                demand_side_splitter,
                "Outlet_Branch_" + str(2 + 2 * iz) + "_Name",
                zone.Name + " Cooling Condenser Branch",
            )
            setattr(
                demand_side_splitter,
                "Outlet_Branch_" + str(2 + 2 * iz + 1) + "_Name",
                zone.Name + " Heating Condenser Branch",
            )

            setattr(
                demand_side_mixer,
                "Inlet_Branch_" + str(2 + 2 * iz) + "_Name",
                zone.Name + " Cooling Condenser Branch",
            )
            setattr(
                demand_side_mixer,
                "Inlet_Branch_" + str(2 + 2 * iz + 1) + "_Name",
                zone.Name + " Heating Condenser Branch",
            )

    return idf


def add_equipment_efficiency_curves(idf: IDF, building_config: BuildingConfig):
    if building_config.heating_water_loop_equipment == "condensing boiler":
        idf.newidfobject(
            "Curve:BiQuadratic".upper(),
            Name="Boiler Efficiency Curve",
            Coefficient1_Constant=1.124970374,
            Coefficient2_x=0.014963852,
            Coefficient3_x2=-0.02599835,
            Coefficient4_y=0.0,
            Coefficient5_y2=-1.40464e-6,
            Coefficient6_xy=-0.00153624,
            Minimum_Value_of_x=0.1,
            Maximum_Value_of_x=1.0,
            Minimum_Value_of_y=30.0,
            Maximum_Value_of_y=85.0,
        )

    elif building_config.heating_water_loop_equipment == "non-condensing boiler":
        idf.newidfobject(
            "Curve:Quadratic".upper(),
            Name="Boiler Efficiency Curve",
            Coefficient1_Constant=0.97,
            Coefficient2_x=0.0633,
            Coefficient3_x2=-0.0333,
            Minimum_Value_of_x=0.0,
            Maximum_Value_of_x=1.0,
        )

    if building_config.dhw_heating_equipment == "condensing boiler":
        idf.newidfobject(
            "Curve:BiQuadratic".upper(),
            Name="DHW Boiler Efficiency Curve",
            Coefficient1_Constant=1.124970374,
            Coefficient2_x=0.014963852,
            Coefficient3_x2=-0.02599835,
            Coefficient4_y=0.0,
            Coefficient5_y2=-1.40464e-6,
            Coefficient6_xy=-0.00153624,
            Minimum_Value_of_x=0.1,
            Maximum_Value_of_x=1.0,
            Minimum_Value_of_y=30.0,
            Maximum_Value_of_y=85.0,
        )

    elif building_config.dhw_heating_equipment == "non-condensing boiler":
        idf.newidfobject(
            "Curve:Quadratic".upper(),
            Name="DHW Boiler Efficiency Curve",
            Coefficient1_Constant=0.97,
            Coefficient2_x=0.0633,
            Coefficient3_x2=-0.0333,
            Minimum_Value_of_x=0.0,
            Maximum_Value_of_x=1.0,
        )

    if "air-to-water heat pump" in [
        building_config.heating_water_loop_equipment,
        building_config.dhw_heating_equipment,
    ]:
        # taken from https://github.com/bsl546/energym/blob/
        # master/simulation/energyplus/apartments2/src/Apartments2_heavy_insulated.idf
        idf.newidfobject(
            "CURVE:BIQUADRATIC",
            Name="CapCurveFuncTemp",
            Coefficient1_Constant=0.369827,
            Coefficient4_y=0.043341,
            Coefficient5_y2=-0.00023,
            Coefficient2_x=0.000466,
            Coefficient3_x2=0.000026,
            Coefficient6_xy=-0.00027,
            Minimum_Value_of_y=-20.0,
            Maximum_Value_of_y=40.0,
            Minimum_Value_of_x=20.0,
            Maximum_Value_of_x=90.0,
            Minimum_Curve_Output="",
            Maximum_Curve_Output="",
            Input_Unit_Type_for_X="Temperature",
            Input_Unit_Type_for_Y="Temperature",
            Output_Unit_Type="Dimensionless",
        )

        # and fitted to inverse
        idf.newidfobject(
            "CURVE:BIQUADRATIC",
            Name="EIRCurveFuncTemp",
            Coefficient1_Constant=2.2238,
            Coefficient4_y=-0.12622,
            Coefficient5_y2=0.00103988,
            Coefficient2_x=-0.008908,
            Coefficient3_x2=-0.0000621397,
            Coefficient6_xy=0.00140904,
            Minimum_Value_of_y=-20.0,
            Maximum_Value_of_y=50.0,
            Minimum_Value_of_x=20.0,
            Maximum_Value_of_x=90.0,
            Minimum_Curve_Output=0.01,
            Maximum_Curve_Output=100,
            Input_Unit_Type_for_X="Temperature",
            Input_Unit_Type_for_Y="Temperature",
            Output_Unit_Type="Dimensionless",
        )

        idf.newidfobject(
            "CURVE:QUADRATIC",
            Name="EIRCurveFuncPLR",
            Coefficient1_Constant=1.176,
            Coefficient2_x=-0.204665,
            Coefficient3_x2=0.02865965,
            Minimum_Value_of_x=0.0,
            Maximum_Value_of_x=1.0,
        )

    if building_config.zone_heating_equipment in [
        "water-to-air heat pump (water loop source)",
        "water-to-air heat pump (ground source)",
    ]:
        idf.newidfobject(
            "CURVE:QUADLINEAR",
            Name="WAHP Total Cooling Capacity Curve",
            Coefficient1_Constant=-9.149069561,
            Coefficient2_w=10.87814026,
            Coefficient3_x=-1.718780157,
            Coefficient4_y=0.746414818,
            Coefficient5_z=0.0,
            Minimum_Value_of_w=0,
            Maximum_Value_of_w=100,
            Minimum_Value_of_x=0,
            Maximum_Value_of_x=100,
            Minimum_Value_of_y=0,
            Maximum_Value_of_y=100,
            Minimum_Value_of_z=0,
            Maximum_Value_of_z=100,
        )

        idf.newidfobject(
            "CURVE:QUINTLINEAR",
            Name="WAHP Sensible Cooling Capacity Curve",
            Coefficient1_Constant=-5.462690012,
            Coefficient2_v=17.95968138,
            Coefficient3_w=-11.87818402,
            Coefficient4_x=-0.980163419,
            Coefficient5_y=0.767285761,
            Coefficient6_z=0.0,
            Minimum_Value_of_v=0,
            Maximum_Value_of_v=100,
            Minimum_Value_of_w=0,
            Maximum_Value_of_w=100,
            Minimum_Value_of_x=0,
            Maximum_Value_of_x=100,
            Minimum_Value_of_y=0,
            Maximum_Value_of_y=100,
            Minimum_Value_of_z=0,
            Maximum_Value_of_z=100,
        )

        idf.newidfobject(
            "CURVE:QUADLINEAR",
            Name="WAHP Cooling Power Consumption Curve",
            Coefficient1_Constant=-3.205409884,
            Coefficient2_w=-0.976409399,
            Coefficient3_x=3.97892546,
            Coefficient4_y=0.938181818,
            Coefficient5_z=0.0,
            Minimum_Value_of_w=0,
            Maximum_Value_of_w=100,
            Minimum_Value_of_x=0,
            Maximum_Value_of_x=100,
            Minimum_Value_of_y=0,
            Maximum_Value_of_y=100,
            Minimum_Value_of_z=0,
            Maximum_Value_of_z=100,
        )

        idf.newidfobject(
            "CURVE:QUADLINEAR",
            Name="WAHP Heating Capacity Curve",
            Coefficient1_Constant=-1.361311959,
            Coefficient2_w=-2.471798046,
            Coefficient3_x=4.173164514,
            Coefficient4_y=0.640757401,
            Coefficient5_z=0.0,
            Minimum_Value_of_w=0,
            Maximum_Value_of_w=100,
            Minimum_Value_of_x=0,
            Maximum_Value_of_x=100,
            Minimum_Value_of_y=0,
            Maximum_Value_of_y=100,
            Minimum_Value_of_z=0,
            Maximum_Value_of_z=100,
        )

        idf.newidfobject(
            "CURVE:QUADLINEAR",
            Name="WAHP Heating Power Consumption Curve",
            Coefficient1_Constant=-2.176941116,
            Coefficient2_w=0.832114286,
            Coefficient3_x=1.570743399,
            Coefficient4_y=0.690793651,
            Coefficient5_z=0.0,
            Minimum_Value_of_w=0,
            Maximum_Value_of_w=100,
            Minimum_Value_of_x=0,
            Maximum_Value_of_x=100,
            Minimum_Value_of_y=0,
            Maximum_Value_of_y=100,
            Minimum_Value_of_z=0,
            Maximum_Value_of_z=100,
        )

    return idf


def add_dhw_branch_and_tank(idf: IDF, building_config: BuildingConfig, zone):
    # this needs to become flexible
    idf.newidfobject(
        "Schedule:Compact".upper(),
        Name=zone.Name + " DHW Flow Rate Fraction Schedule",
        Schedule_Type_Limits_Name="Limits Any Number",
        Field_1=(
            "Through: 12/31,  For: AllDays,   "
            "Until: 8:00, 0,  Until:8:20, 0.5, Until:19:00,0, "
            "Until:19:20,0.5,Until 24:00,0"
        ),
    )

    idf.newidfobject(
        "BRANCH",
        Name=zone.Name + " Hot Water Loop Demand DHW Use",
        Component_1_Object_Type="WATERHEATER:MIXED",
        Component_1_Name=zone.Name + " DHW Water Heater Tank",
        Component_1_Inlet_Node_Name=zone.Name + " Hot Water Loop Demand DHW Inlet",
        Component_1_Outlet_Node_Name=zone.Name + " Hot Water Loop Demand DHW Outlet",
    )

    idf.newidfobject(
        "WATERHEATER:MIXED",
        Name=zone.Name + " DHW Water Heater Tank",
        Tank_Volume=building_config.dhw_water_tank_volume,
        Setpoint_Temperature_Schedule_Name="45degrees",
        Deadband_Temperature_Difference=0,
        Maximum_Temperature_Limit=80,
        Heater_Control_Type="Cycle",
        Heater_Maximum_Capacity=0,
        Heater_Minimum_Capacity=0,
        Heater_Ignition_Minimum_Flow_Rate="",
        Heater_Ignition_Delay="",
        Heater_Fuel_Type=building_config.dhw_heating_equipment_fuel,
        Heater_Thermal_Efficiency=1,
        Part_Load_Factor_Curve_Name="",
        Off_Cycle_Parasitic_Fuel_Consumption_Rate="",
        Off_Cycle_Parasitic_Fuel_Type="",
        Off_Cycle_Parasitic_Heat_Fraction_to_Tank="",
        On_Cycle_Parasitic_Fuel_Consumption_Rate="",
        On_Cycle_Parasitic_Fuel_Type="",
        On_Cycle_Parasitic_Heat_Fraction_to_Tank="",
        Ambient_Temperature_Indicator="Zone",
        Ambient_Temperature_Schedule_Name="",
        Ambient_Temperature_Zone_Name=zone.Name,
        Ambient_Temperature_Outdoor_Air_Node_Name="",
        Off_Cycle_Loss_Coefficient_to_Ambient_Temperature="",
        Off_Cycle_Loss_Fraction_to_Zone=1,
        On_Cycle_Loss_Coefficient_to_Ambient_Temperature="",
        On_Cycle_Loss_Fraction_to_Zone=1,
        Peak_Use_Flow_Rate=0.01 / 60,
        Use_Flow_Rate_Fraction_Schedule_Name=zone.Name
        + " DHW Flow Rate Fraction Schedule",
        Cold_Water_Supply_Temperature_Schedule_Name="",
        Use_Side_Inlet_Node_Name="",
        Use_Side_Outlet_Node_Name="",
        Use_Side_Effectiveness=1,
        Source_Side_Inlet_Node_Name=zone.Name + " Hot Water Loop Demand DHW Inlet",
        Source_Side_Outlet_Node_Name=zone.Name + " Hot Water Loop Demand DHW Outlet",
        Source_Side_Effectiveness=1,
        Use_Side_Design_Flow_Rate="",
        Source_Side_Design_Flow_Rate=1e-003,
        Indirect_Water_Heating_Recovery_Time=1.5,
        Source_Side_Flow_Control_Mode="IndirectHeatPrimarySetpoint",
    )

    return idf, zone.Name + " Hot Water Loop Demand DHW Use"
