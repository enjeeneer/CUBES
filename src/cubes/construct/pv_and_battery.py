"""This module adds PV panels and a battery to an idf file"""

from geomeppy import IDF

# from cubes.construct.buildingconfig import BuildingConfig


def add_pv_and_battery(idf: IDF):

    # add PV panels
    idf.newidfobject(
        "GENERATOR:PHOTOVOLTAIC",
        Name="PVpanels",
        Surface_Name="Pv_surface",
        Photovoltaic_Performance_Object_Type="PhotovoltaicPerformance:Simple",
        Module_Performance_Name="15percentEffPVh83Area",
        Heat_Transfer_Integration_Mode="Decoupled",
        Number_of_Series_Strings_in_Parallel=14,
        Number_of_Modules_in_Series=3,
    )

    idf.newidfobject(
        "PHOTOVOLTAICPERFORMANCE:SIMPLE",
        Name="15percentEffPVh83Area",
        Fraction_of_Surface_Area_with_Active_Solar_Cells=0.83,
        Conversion_Efficiency_Input_Mode="Fixed",
        Value_for_Cell_Efficiency_if_Fixed=0.158,
    )

    idf.newidfobject(
        "ELECTRICLOADCENTER:GENERATORS",
        Name="Generator List",
        Generator_1_Name="PVpanels",
        Generator_1_Object_Type="Generator:Photovoltaic",
        Generator_1_Rated_Electric_Power_Output=10750.0,
        Generator_1_Availability_Schedule_Name="Always-Schedule",
        Generator_1_Rated_Thermal_to_Electrical_Power_Ratio="",
    )

    idf.newidfobject(
        "ELECTRICLOADCENTER:INVERTER:SIMPLE",
        Name="Example Inverter - Simple",
        Availability_Schedule_Name="Always-Schedule",
        Zone_Name="",
        Radiative_Fraction=0.3,
        Inverter_Efficiency=0.95,
    )

    idf.newidfobject(
        "ELECTRICLOADCENTER:STORAGE:BATTERY",
        Name="Synerion 24M",
        Availability_Schedule_Name="",
        Zone_Name="",
        Radiative_Fraction=0,
        Number_of_Battery_Modules_in_Parallel=1,
        Number_of_Battery_Modules_in_Series=5,
        Maximum_Module_Capacity=85,
        Initial_Fractional_State_of_Charge=0,
        Fraction_of_Available_Charge_Capacity=1,
        Change_Rate_from_Bound_Charge_to_Available_Charge=1,
        Fully_Charged_Module_Open_Circuit_Voltage=28,
        Fully_Discharged_Module_Open_Circuit_Voltage=23.25,
        Voltage_Change_Curve_Name_for_Charging="Synerion 24M BatteryChargeCurve",
        Voltage_Change_Curve_Name_for_Discharging="Synerion 24M BatteryDischargeCurve",
        Module_Internal_Electrical_Resistance=0.017,
        Maximum_Module_Discharging_Current=150,
        Module_Cutoff_Voltage=21,
        Module_Charge_Rate_Limit=1,
        Battery_Life_Calculation="No",
    )

    idf.newidfobject(
        "ELECTRICLOADCENTER:DISTRIBUTION",
        Name="DC with inverter and Synerion 24M",
        Generator_List_Name="Generator List",
        Generator_Operation_Scheme_Type="Baseload",
        Generator_Demand_Limit_Scheme_Purchased_Electric_Demand_Limit="",
        Generator_Track_Schedule_Name_Scheme_Schedule_Name="",
        Generator_Track_Meter_Scheme_Meter_Name="",
        Electrical_Buss_Type="DirectCurrentWithInverterDCStorage",
        Inverter_Name="Example Inverter - Simple",
        Electrical_Storage_Object_Name="Synerion 24M",
        Transformer_Object_Name="",
        Storage_Operation_Scheme="TrackFacilityElectricDemandStoreExcessOnSite",
        Storage_Control_Track_Meter_Name="",
        Storage_Converter_Object_Name="ACDCConverter",
        Maximum_Storage_State_of_Charge_Fraction="",
        Minimum_Storage_State_of_Charge_Fraction="",
        Design_Storage_Control_Charge_Power="",
        Storage_Charge_Power_Fraction_Schedule_Name="",
        Design_Storage_Control_Discharge_Power="4000",
        Storage_Discharge_Power_Fraction_Schedule_Name="",
    )

    idf.newidfobject(
        "ELECTRICLOADCENTER:STORAGE:CONVERTER",
        Name="ACDCConverter",
        Availability_Schedule_Name="Always-Schedule",
        Power_Conversion_Efficiency_Method="SimpleFixed",
        Simple_Fixed_Efficiency=0.95,
    )

    idf.newidfobject(
        "SHADING:BUILDING:DETAILED",
        Name="Pv_surface",
        Transmittance_Schedule_Name="",
        Number_of_Vertices=4,
        Vertex_1_Xcoordinate=-2.461095281049,
        Vertex_2_Xcoordinate=-2.461095281049,
        Vertex_3_Xcoordinate=7.538904718951,
        Vertex_4_Xcoordinate=7.538904718951,
        Vertex_1_Ycoordinate=8.936336673095,
        Vertex_2_Ycoordinate=3.000000000000,
        Vertex_3_Ycoordinate=3.000000000000,
        Vertex_4_Ycoordinate=8.936336673095,
        Vertex_1_Zcoordinate=17.709434849632,
        Vertex_2_Zcoordinate=14.000000000000,
        Vertex_3_Zcoordinate=14.000000000000,
        Vertex_4_Zcoordinate=17.709434849632,
    )

    idf.newidfobject(
        "CURVE:RECTANGULARHYPERBOLA2",
        Name="Synerion 24M BatteryDischargeCurve",
        Coefficient1_C1=424,
        Coefficient2_C2=84.9,
        Coefficient3_C3=-9.729,
        Minimum_Value_of_x=0,
        Maximum_Value_of_x=1,
        Minimum_Curve_Output=-100,
        Maximum_Curve_Output=100,
    )

    idf.newidfobject(
        "CURVE:RECTANGULARHYPERBOLA2",
        Name="Synerion 24M BatteryChargeCurve",
        Coefficient1_C1=14.49,
        Coefficient2_C2=-70.95,
        Coefficient3_C3=4.787,
        Minimum_Value_of_x=0,
        Maximum_Value_of_x=1,
        Minimum_Curve_Output=-100,
        Maximum_Curve_Output=100,
    )
    return idf
