"""collection of utilities for packaging up files for use with gym
"""
from cubes.package import constants
from pathlib import Path
import shutil


def get_rdd_and_expand_idf(idf):
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

    # get rdd file
    shutil.copyfile(
        constants.temp_output_path + "/eplusout.rdd", constants.rdd_file_path
    )
    # get expanded idf file
    shutil.copyfile(
        constants.temp_output_path + "/eplusout.expidf", constants.idf_file_path
    )

    # delete all other data
    shutil.rmtree(constants.temp_output_path)


def check_observation_variables(obs_vars, rdd_vars, idf_zone_names) -> None:
    """This method checks whether observation variables names
    are available in building energy simulation"""
    for obs_var in obs_vars:
        obs_name = obs_var.split("(")[0]
        obs_zone = obs_var.split("(")[1][:-1]

        # Check observarion variable names
        assert obs_name in rdd_vars, (
            f"Observation variables: Variable called {obs_name}"
            " in observation variables is not valid for IDF building model",
        )

        # Check observation variable zones
        if (
            obs_zone.lower() != "Environment".lower()
            and obs_zone.lower() != "Whole Building".lower()
        ):

            # sinergym: zones names with people 1 or lights 1, etc. The second name
            # is ignored, only check that zone is a substr from obs zone
            zone_exists = False
            for zone in idf_zone_names:
                if zone.lower() in obs_zone.lower():
                    zone_exists = True
                    break

            assert zone_exists, (
                f"Observation variables: Zone called {obs_zone} "
                "in observation variables does not exist in IDF building model."
            )
