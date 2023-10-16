"""This module adds PV panels and a battery to an idf file"""

from geomeppy import IDF

from cubes.construct.buildingconfig import BuildingConfig
from cubes.construct.roof import get_pv_surface_coordinates
from cubes.construct import utilities

battery_modules_in_series = 5
battery_fully_charged_open_circuit_discharge_voltage = 28
battery_charging_power = 4000


def get_battery_ah_from_kwh(kwh):
    return (
        kwh
        * 1000.0
        / battery_modules_in_series
        / battery_fully_charged_open_circuit_discharge_voltage
    )


def add_pv_and_battery(idf: IDF, building_config: BuildingConfig):

    surface_coords = get_pv_surface_coordinates(building_config)

    if not surface_coords[0] and not surface_coords[1]:
        return idf

    pv_areas = [0, 0]

    for isc, sc in enumerate(surface_coords):
        if sc:
            idf.newidfobject(
                "SHADING:BUILDING:DETAILED",
                Name="Pv_surface_" + str(isc),
                Transmittance_Schedule_Name="",
                Number_of_Vertices=4,
                Vertex_1_Xcoordinate=sc["X1"],
                Vertex_2_Xcoordinate=sc["X2"],
                Vertex_3_Xcoordinate=sc["X3"],
                Vertex_4_Xcoordinate=sc["X4"],
                Vertex_1_Ycoordinate=sc["Y1"],
                Vertex_2_Ycoordinate=sc["Y2"],
                Vertex_3_Ycoordinate=sc["Y3"],
                Vertex_4_Ycoordinate=sc["Y4"],
                Vertex_1_Zcoordinate=sc["Z1"],
                Vertex_2_Zcoordinate=sc["Z2"],
                Vertex_3_Zcoordinate=sc["Z3"],
                Vertex_4_Zcoordinate=sc["Z4"],
            )

            pv_areas[isc] = utilities.get_surface_area(
                idf.idfobjects["SHADING:BUILDING:DETAILED"][-1]
            )

            # add PV panels
            idf.newidfobject(
                "GENERATOR:PHOTOVOLTAIC",
                Name="PVpanels_" + str(isc),
                Surface_Name="Pv_surface_" + str(isc),
                Photovoltaic_Performance_Object_Type="PhotovoltaicPerformance:Simple",
                Module_Performance_Name="15percentEffPVh83Area",
                Heat_Transfer_Integration_Mode="Decoupled",
                Number_of_Series_Strings_in_Parallel=14,
                Number_of_Modules_in_Series=3,
            )

    idf.newidfobject(
        "PHOTOVOLTAICPERFORMANCE:SIMPLE",
        Name="15percentEffPVh83Area",
        Fraction_of_Surface_Area_with_Active_Solar_Cells=(
            building_config.pv_active_area_fraction
        ),
        Conversion_Efficiency_Input_Mode="Fixed",
        Value_for_Cell_Efficiency_if_Fixed=building_config.pv_cell_efficiency,
    )

    # put in 1 or 2 solar panels and calculate rated power output
    idf.newidfobject(
        "ELECTRICLOADCENTER:GENERATORS",
        Name="Generator List",
    )
    generator_list = idf.idfobjects["ELECTRICLOADCENTER:GENERATORS"][-1]
    for isc, sc in enumerate(surface_coords):
        if sc:
            setattr(
                generator_list,
                "Generator_" + str(isc + 1) + "_Name",
                "PVpanels_" + str(isc),
            )
            setattr(
                generator_list,
                "Generator_" + str(isc + 1) + "_Object_Type",
                "Generator:Photovoltaic",
            )
            setattr(
                generator_list,
                "Generator_" + str(isc + 1) + "_Rated_Electric_Power_Output",
                pv_areas[isc]
                * building_config.pv_cell_efficiency
                * building_config.pv_active_area_fraction
                * 1000,
            )
            setattr(
                generator_list,
                "Generator_" + str(isc + 1) + "_Availability_Schedule_Name",
                "Always-Schedule",
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
        Number_of_Battery_Modules_in_Series=battery_modules_in_series,
        Maximum_Module_Capacity=get_battery_ah_from_kwh(
            building_config.battery_energy_storage
        ),
        Initial_Fractional_State_of_Charge=0,
        Fraction_of_Available_Charge_Capacity=1,
        Change_Rate_from_Bound_Charge_to_Available_Charge=1,
        Fully_Charged_Module_Open_Circuit_Voltage=(
            battery_fully_charged_open_circuit_discharge_voltage
        ),
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
        Design_Storage_Control_Charge_Power=battery_charging_power,
        Storage_Charge_Power_Fraction_Schedule_Name="",
        Design_Storage_Control_Discharge_Power=battery_charging_power,
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
