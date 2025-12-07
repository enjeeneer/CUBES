# build_coheat.py
from eppy.modeleditor import IDF
from pathlib import Path

def make_coheating_model(idf, zone_temperatures, coheat_period=None):
    """
    Transform an IDF into a co-heating test model:
      - remove HVAC + plant
      - remove internal loads (people, lights, equipment)
      - add IdealLoads systems
      - set fixed heating setpoints per zone
      - override venting schedules
      - build new RunPeriod
      - normalise AFN names
      - add output variables
    """

    # ----------------------------------------
    # 1. Remove HVAC + Plant objects
    # ----------------------------------------
    remove_prefixes = [
        "PLANTLOOP", "AIRLOOPHVAC", "BRANCH", "BRANCHLIST",
        "CONNECTOR:", "PUMP:", "BOILER:", "CHILLER:", "COIL:",
        "AIRTERMINAL:", "ZONEHVAC:", "FAN:", "SIZING:", "CONTROLLER:",
        "SETPOINTMANAGER:", "HVACTEMPLATE:", "PIPE:", "PLANTEQUIPMENT",
        "ENERGYMANAGEMENTSYSTEM:", "CURVE:"
    ]

    for key in list(idf.idfobjects.keys()):
        key_upper = key.upper()
        if any(key_upper.startswith(prefix.replace(":", "")) for prefix in remove_prefixes):
            for obj in list(idf.idfobjects[key]):
                idf.removeidfobject(obj)

    # Remove thermostats to avoid clashes
    for cls in [
        "ZONECONTROL:THERMOSTAT",
        "THERMOSTATSETPOINT:SINGLEHEATING",
        "THERMOSTATSETPOINT:SINGLECOOLING",
        "THERMOSTATSETPOINT:DUALSETPOINT"
    ]:
        for obj in list(idf.idfobjects.get(cls, [])):
            idf.removeidfobject(obj)

    # Remove internal loads (for clean coheat test)
    for cls in ["PEOPLE", "LIGHTS", "ELECTRICEQUIPMENT", "GASEQUIPMENT", "OTHEREQUIPMENT"]:
        for obj in list(idf.idfobjects.get(cls, [])):
            idf.removeidfobject(obj)

    # ----------------------------------------
    # 2. Schedules needed
    # ----------------------------------------
    if not any(s.Name.lower()=="alwaysonschedule"
               for s in idf.idfobjects.get("SCHEDULE:CONSTANT", [])):
        idf.newidfobject(
            "SCHEDULE:CONSTANT",
            Name="AlwaysOnSchedule",
            Schedule_Type_Limits_Name="OnOff",
            Hourly_Value=1
        )

    if not any(s.Name.lower()=="alwaysoffschedule"
               for s in idf.idfobjects.get("SCHEDULE:CONSTANT", [])):
        idf.newidfobject(
            "SCHEDULE:CONSTANT",
            Name="AlwaysOffSchedule",
            Schedule_Type_Limits_Name="OnOff",
            Hourly_Value=0
        )

    # ----------------------------------------
    # 3. Replace RunPeriod
    # ----------------------------------------
    for rp in list(idf.idfobjects.get("RUNPERIOD", [])):
        idf.removeidfobject(rp)

    coheat_period = coheat_period or (11, 23, 2013, 12, 1, 2013)
    bm, bd, by, em, ed, ey = coheat_period

    idf.newidfobject("RUNPERIOD",
                     Name="Coheat_Test_Period",
                     Begin_Month=bm, Begin_Day_of_Month=bd, Begin_Year=by,
                     End_Month=em, End_Day_of_Month=ed, End_Year=ey)

    # ----------------------------------------
    # 4. Fix AFN venting schedules for coheat test
    # ----------------------------------------
    # Coheat test requirements:
    # - Windows: CLOSED (AlwaysOffSchedule)
    # - External doors: CLOSED (AlwaysOffSchedule)
    # - Internal doors: OPEN (AlwaysOnSchedule)

    for surf in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:SURFACE", []):
        if not hasattr(surf, "Venting_Availability_Schedule_Name"):
            continue

        surf_name = surf.Surface_Name.lower() if hasattr(surf, "Surface_Name") else ""

        # Windows and external doors should be closed
        if "_window" in surf_name or "_extdoor" in surf_name:
            surf.Venting_Availability_Schedule_Name = "AlwaysOffSchedule"
        # Internal doors should be open
        elif "_door" in surf_name:
            surf.Venting_Availability_Schedule_Name = "AlwaysOnSchedule"

    # ----------------------------------------
    # 5. Insert thermostats + IdealLoads systems
    # ----------------------------------------
    excluded = {"loft", "subfloor"}

    for zone in idf.idfobjects["ZONE"]:
        zname = zone.Name.lower()

        if any(ex in zname for ex in excluded):
            continue

        if zname in zone_temperatures:

            # Create Schedule
            sched = f"{zone.Name}_coheat_temp_schedule"
            idf.newidfobject("SCHEDULE:COMPACT",
                             Name=sched,
                             Schedule_Type_Limits_Name="Temperature",
                             Field_1="Through: 12/31",
                             Field_2="For: AllDays",
                             Field_3="Until: 24:00",
                             Field_4=str(zone_temperatures[zname]))

            # Create Setpoint
            sp = f"{zone.Name}_Coheat_Setpoint"
            idf.newidfobject("THERMOSTATSETPOINT:SINGLEHEATING",
                             Name=sp,
                             Setpoint_Temperature_Schedule_Name=sched)

            # Create Thermostat Controller
            idf.newidfobject("ZONECONTROL:THERMOSTAT",
                             Name=f"{zone.Name}_Thermostat",
                             Zone_or_ZoneList_Name=zone.Name,
                             Control_Type_Schedule_Name="AlwaysOnSchedule",
                             Control_1_Object_Type="ThermostatSetpoint:SingleHeating",
                             Control_1_Name=sp)

        # Create IdealLoads
        sys = f"{zone.Name}_IdealLoads"
        idf.newidfobject("ZONEHVAC:IDEALLOADSAIRSYSTEM",
                         Name=sys,
                         Availability_Schedule_Name="AlwaysOnSchedule",
                         Zone_Supply_Air_Node_Name=f"{zone.Name}_Supply",
                         Zone_Exhaust_Air_Node_Name=f"{zone.Name}_Exhaust",
                         Maximum_Heating_Supply_Air_Temperature=50,
                         Heating_Limit="NoLimit",
                         Heating_Availability_Schedule_Name="AlwaysOnSchedule",
                         Cooling_Availability_Schedule_Name="AlwaysOffSchedule")

        eq_list = f"{zone.Name}_EquipList"
        idf.newidfobject("ZONEHVAC:EQUIPMENTLIST",
                     Name=eq_list,
                     Load_Distribution_Scheme="SequentialLoad",
                     Zone_Equipment_1_Object_Type="ZoneHVAC:IdealLoadsAirSystem",
                     Zone_Equipment_1_Name=sys,
                     Zone_Equipment_1_Cooling_Sequence=1,
                     Zone_Equipment_1_Heating_or_NoLoad_Sequence=1,
                     Zone_Equipment_1_Sequential_Cooling_Fraction_Schedule_Name="",
                     Zone_Equipment_1_Sequential_Heating_Fraction_Schedule_Name="")


        idf.newidfobject("ZONEHVAC:EQUIPMENTCONNECTIONS",
                         Zone_Name=zone.Name,
                         Zone_Conditioning_Equipment_List_Name=eq_list,
                         Zone_Air_Inlet_Node_or_NodeList_Name=f"{zone.Name}_Supply",
                         Zone_Air_Exhaust_Node_or_NodeList_Name=f"{zone.Name}_Exhaust",
                         Zone_Air_Node_Name=f"{zone.Name}_AirNode",
                         Zone_Return_Air_Node_or_NodeList_Name=f"{zone.Name}_Return")

    # ----------------------------------------
    # 6. Load output variables from coheat_variables.idf
    # ----------------------------------------
    variables_idf_path = Path(__file__).parent / "coheat_variables.idf"
    if variables_idf_path.exists():
        # Load variables.idf as a separate IDF object
        variables_idf = IDF(str(variables_idf_path))

        # Copy all Output:Variable objects from variables.idf to main idf
        for var in variables_idf.idfobjects.get("OUTPUT:VARIABLE", []):
            idf.newidfobject(
                "OUTPUT:VARIABLE",
                Key_Value=var.Key_Value,
                Variable_Name=var.Variable_Name,
                Reporting_Frequency=var.Reporting_Frequency
            )

    # ----------------------------------------
    # 7. SimulationControl
    # ----------------------------------------
    for sc in list(idf.idfobjects.get("SIMULATIONCONTROL", [])):
        idf.removeidfobject(sc)

    idf.newidfobject("SIMULATIONCONTROL",
                     Do_Zone_Sizing_Calculation="No",
                     Do_System_Sizing_Calculation="No",
                     Do_Plant_Sizing_Calculation="No",
                     Run_Simulation_for_Sizing_Periods="No",
                     Run_Simulation_for_Weather_File_Run_Periods="Yes")

    return idf


# ----------------------------------------------------
# Optional CLI: python build_coheat.py input.idf output.idf
# ----------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a co-heating IDF from a base IDF.")
    parser.add_argument("--idd", required=True)
    parser.add_argument("--idf_in", required=True)
    parser.add_argument("--idf_out", required=True)

    args = parser.parse_args()

    IDF.setiddname(args.idd)

    idf = IDF(args.idf_in)

    zone_temps = {
        "front_room": 24.32,
        "backroom": 24.64,
        "kitchen": 25.15,
        "hall_downstairs": 24.0,
        "hall_upstairs": 24.67,
        "bedroom_1": 24.62,
        "bedroom_2": 24.67,
        "bathroom": 23.53,
        "bedroom_3": 24.83,
    }

    idf = make_coheating_model(idf, zone_temps)
    idf.saveas(args.idf_out)

    print(f"✅ Saved co-heating model: {args.idf_out}")
