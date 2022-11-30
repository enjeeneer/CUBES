"""collection of utilities for packaging up files for use with gym
"""
from cubes.package import constants
from cubes.package.variables import ActionVariable
from pathlib import Path
import shutil


def get_rdd_file(idf):
    # make some changes to the idf so that the run time is minimal
    idf.idfobjects["SIMULATIONCONTROL"][0].Do_Zone_Sizing_Calculation = "Yes"
    idf.idfobjects["SIMULATIONCONTROL"][0].Do_System_Sizing_Calculation = "No"
    idf.idfobjects["SIMULATIONCONTROL"][0].Do_Plant_Sizing_Calculation = "No"
    idf.idfobjects["SIMULATIONCONTROL"][0].Run_Simulation_for_Sizing_Periods = "Yes"
    idf.idfobjects["SIMULATIONCONTROL"][
        0
    ].Run_Simulation_for_Weather_File_Run_Periods = "No"
    idf.idfobjects["SIMULATIONCONTROL"][
        0
    ].Do_HVAC_Sizing_Simulation_for_Sizing_Periods = "No"

    idf.idfobjects["BUILDING"][0].Minimum_Number_of_Warmup_Days = 1

    # run idf
    Path(constants.temp_output_path).mkdir(parents=True, exist_ok=True)
    idf.run(
        expandobjects=True,
        weather=constants.weather_file_path,
        output_directory=constants.temp_output_path,
        verbose="q",
    )

    # get rdd file and delete all other data
    shutil.copyfile(
        constants.temp_output_path + "/eplusout.rdd", constants.rdd_file_path
    )
    shutil.rmtree(constants.temp_output_path)


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
                ActionVariable(
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

                    action_variables.append(
                        ActionVariable(schedule_name, obj, "Temperature")
                    )

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
                ActionVariable(
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
                ActionVariable(
                    heating_schedule_name,
                    "THERMOSTATSETPOINT:SINGLECOOLING",
                    "Temperature",
                )
            )

    return idf, action_variables
