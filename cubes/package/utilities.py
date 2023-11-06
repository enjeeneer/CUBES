"""collection of utilities for packaging up files for use with gym
"""
from cubes.constants import package_directory
from cubes.package.weather import get_weather_file_path
from cubes.package.envconfig import EnvConfig
from pathlib import Path
import shutil
from geomeppy import IDF
import pandas as pd
import numpy as np
from typing import List


def get_rdd_file(idf: IDF, env_config: EnvConfig):
    # setup paths
    temp_output_path = env_config.files_dir + "/temp"
    weather_path = env_config.files_dir + "/weather.epw"
    rdd_file_path = env_config.files_dir + "/building_model.rdd"
    Path(temp_output_path).mkdir(parents=True, exist_ok=True)

    # make some changes to the idf so that the run time is minimal
    idf.idfobjects["SIMULATIONCONTROL"][0].Do_Zone_Sizing_Calculation = "Yes"
    idf.idfobjects["SIMULATIONCONTROL"][0].Do_System_Sizing_Calculation = "Yes"
    idf.idfobjects["SIMULATIONCONTROL"][0].Do_Plant_Sizing_Calculation = "Yes"
    idf.idfobjects["SIMULATIONCONTROL"][0].Run_Simulation_for_Sizing_Periods = "Yes"
    idf.idfobjects["SIMULATIONCONTROL"][
        0
    ].Run_Simulation_for_Weather_File_Run_Periods = "No"
    idf.idfobjects["SIMULATIONCONTROL"][
        0
    ].Do_HVAC_Sizing_Simulation_for_Sizing_Periods = "No"

    idf.idfobjects["BUILDING"][0].Minimum_Number_of_Warmup_Days = 1

    # run idf
    idf.save(temp_output_path + "/dummy.idf")
    idf.run(
        expandobjects=False,
        readvars=True,
        weather=weather_path,
        output_directory=temp_output_path,
        verbose="q",
    )

    # get rdd file
    shutil.copyfile(temp_output_path + "/eplusout.rdd", rdd_file_path)

    # IDF.setiddname(EPLUS_PATH + "Energy+.idd")
    # expanded_idf = IDF(constants.temp_output_path + "/eplusout.expidf")
    # expanded_idf.epw = constants.weather_file_path
    idf = set_simulation_parameters(idf)

    # idf.newidfobject("OUTPUT:SURFACES:DRAWING", Report_Type="DXF")
    # delete all other data
    shutil.rmtree(temp_output_path)

    return idf


def set_run_period(idf: IDF, envconfig: EnvConfig):
    idf.idfobjects["RUNPERIOD"][0].Begin_Month = envconfig.episode_start_date[1]
    idf.idfobjects["RUNPERIOD"][0].Begin_Day_of_Month = envconfig.episode_start_date[0]
    idf.idfobjects["RUNPERIOD"][0].End_Month = envconfig.episode_end_date[1]
    idf.idfobjects["RUNPERIOD"][0].End_Day_of_Month = envconfig.episode_end_date[0]
    idf.idfobjects["TIMESTEP"][
        0
    ].Number_of_Timesteps_per_Hour = envconfig.timesteps_per_hour

    return idf


def set_simulation_parameters(idf: IDF):
    # make some changes to the expanded idf so that the simulation is run normally
    idf.idfobjects["SIMULATIONCONTROL"][0].Do_Zone_Sizing_Calculation = "Yes"
    idf.idfobjects["SIMULATIONCONTROL"][0].Do_System_Sizing_Calculation = "Yes"
    idf.idfobjects["SIMULATIONCONTROL"][0].Do_Plant_Sizing_Calculation = "Yes"
    idf.idfobjects["SIMULATIONCONTROL"][0].Run_Simulation_for_Sizing_Periods = "No"
    idf.idfobjects["SIMULATIONCONTROL"][
        0
    ].Run_Simulation_for_Weather_File_Run_Periods = "Yes"
    idf.idfobjects["SIMULATIONCONTROL"][
        0
    ].Do_HVAC_Sizing_Simulation_for_Sizing_Periods = "No"

    idf.idfobjects["BUILDING"][0].Minimum_Number_of_Warmup_Days = 20
    return idf


def check_observation_variables(obs_vars, rdd_vars) -> None:
    """This method checks whether observation variables names
    are available in building energy simulation"""
    for obs_var in obs_vars:
        obs_name = obs_var.split("(")[0]

        # Check observarion variable names
        assert obs_name in rdd_vars, (
            f"Observation variables: Variable called {obs_name}"
            " in observation variables is not valid for IDF building model",
        )


def get_temperature_forecast_file_path(env_files_dir: str, hours: int):
    return env_files_dir + f"/temperature_forecast_{str(hours)}h.csv"


def get_grid_forecast_file_path(env_files_dir: str, hours: int):
    return env_files_dir + f"/grid_forecast_{str(hours)}h.csv"


def get_temperature_forecast_files(
    weather_file_name: str, temperature_forecast_hours: List[int], env_files_dir: str
):
    """this function produces temperature forecast files
    Numbers based on following assumptions:
    - 92.5% of T forecasts for the next day lie within +-2 degC
    - The standard deviation of the gaussian varies linearly with forecast time"""

    if temperature_forecast_hours:
        sigma_24h = 1.123302474060961  # gaussian based on Met office accuracy

        def sigma(forecast_hours):
            return sigma_24h / 24 * forecast_hours

        temp_data = pd.read_csv(
            get_weather_file_path(weather_file_name),
            skiprows=8,
            usecols=[6],
            names=["T"],
        )

        for tfh in temperature_forecast_hours:
            forecast = np.zeros(len(temp_data))

            for i in range(len(temp_data)):
                if i < len(temp_data) - tfh:
                    forecast[i] = temp_data.loc[i + tfh, "T"] + np.random.normal(
                        0, sigma(tfh), 1
                    )
                else:
                    forecast[i] = temp_data.loc[i, "T"]

            np.savetxt(
                get_temperature_forecast_file_path(
                    env_files_dir=env_files_dir, hours=tfh
                ),
                forecast,
                fmt="%10.2f",
                newline=",\n",
            )


def get_grid_file_path(grid_file_name):
    return package_directory + "/data/grid/" + grid_file_name


def get_grid_carbon_forecast_files(
    grid_carbon_file_name: str,
    grid_carbon_forecast_hours: List[int],
    env_files_dir: str,
):
    """this function produces grid carbon forecast files
    Numbers based on following assumptions:
    - perfect forecast (should be changed)"""

    if grid_carbon_forecast_hours:

        grid_data = pd.read_csv(
            get_grid_file_path(grid_carbon_file_name),
            usecols=[1],
            names=["gCO2/kWh"],
        )

        for gfh in grid_carbon_forecast_hours:
            forecast = np.zeros(len(grid_data))

            for i in range(len(grid_data)):
                if i < len(grid_data) - gfh:
                    forecast[i] = grid_data.loc[i + gfh, "gCO2/kWh"]
                else:
                    forecast[i] = grid_data.loc[i, "gCO2/kWh"]

            np.savetxt(
                get_grid_forecast_file_path(env_files_dir=env_files_dir, hours=gfh),
                forecast,
                fmt="%10.2f",
                newline=",\n",
            )


def get_envconfig_leiden(
    case_number, files_dir: str, obs_for_rbc=False, short_test=False
):
    control_vent = True
    observe_vent = True
    control_observe_battery = False
    if case_number in [3, 4, 8, 9, 13, 14]:
        control_vent = False
        observe_vent = False
    if case_number >= 10:
        control_observe_battery = True
    if case_number < 5:
        observe_outside_temperature_in_x_hours_forecast = [1]
        observe_grid_carbon_in_x_hours_forecast = []
    else:
        observe_outside_temperature_in_x_hours_forecast = [1]
        observe_grid_carbon_in_x_hours_forecast = [1]

    ec = EnvConfig(
        files_dir=files_dir,
        observe_zone_temperature=True,
        observe_electricity_demand=True,
        observe_outside_temperature=True,
        observe_zone_occupancy=True,
        observe_zone_co2=True,
        observe_grid_carbon_intensity=True,
        observe_zone_thermostat_setpoints=True,
        observe_zone_ventilation=observe_vent,
        observe_battery_charge=control_observe_battery,
        observe_battery_charging=control_observe_battery,
        observe_pv_power=control_observe_battery,
        control_battery_charging=control_observe_battery,
        control_ventilation=control_vent,
        control_thermostat_setpoints=True,
        observe_outside_temperature_in_x_hours_forecast=(
            observe_outside_temperature_in_x_hours_forecast
        ),
        observe_grid_carbon_in_x_hours_forecast=(
            observe_grid_carbon_in_x_hours_forecast
        ),
        timesteps_per_hour=6,
        observe_solar_irradiance=obs_for_rbc,
        observe_zone_humidity=obs_for_rbc,
        observe_wind_speed=obs_for_rbc,
        observe_outside_humidity=obs_for_rbc,
    )
    if short_test:
        ec.episode_end_date = (15, 1)
    return ec


def get_envconfig_jack(experiment):

    # outdoor observations
    obs_solar_irradiance = True
    obs_outside_humidity = True
    obs_outside_pressure = True
    obs_outside_temperature = True
    obs_wind_speed = True
    obs_wind_direction = True
    obs_rain = True

    # forecast observations
    obs_outside_temperature_in_x_hours_forecast = [1]
    obs_grid_carbon_in_x_hours_forecast = [1]

    # occupant observations
    obs_thermal_comfort = False
    obs_zone_occupancy = False

    if experiment == "no_outdoor":

        # outdoor observations
        obs_solar_irradiance = False
        obs_outside_humidity = False
        obs_outside_pressure = False
        obs_outside_temperature = False
        obs_wind_speed = False
        obs_wind_direction = False
        obs_rain = False

    elif experiment == "no_forecast":

        # forecast observations
        obs_outside_temperature_in_x_hours_forecast = None
        obs_grid_carbon_in_x_hours_forecast = None

    elif experiment == "no_occupant":

        # occupant observations
        obs_thermal_comfort = False
        obs_zone_occupancy = False

    ec = EnvConfig(
        observe_zone_temperature=True,
        observe_electricity_demand=True,
        observe_outside_temperature=obs_outside_temperature,
        observe_zone_co2=True,
        observe_grid_carbon_intensity=True,
        observe_zone_thermostat_setpoints=True,
        observe_zone_ventilation=True,
        observe_outside_temperature_in_x_hours_forecast=(
            obs_outside_temperature_in_x_hours_forecast
        ),
        observe_grid_carbon_in_x_hours_forecast=(obs_grid_carbon_in_x_hours_forecast),
        observe_zone_occupancy=obs_zone_occupancy,
        observe_thermal_comfort=obs_thermal_comfort,
        timesteps_per_hour=6,
        observe_solar_irradiance=obs_solar_irradiance,
        observe_zone_humidity=True,
        observe_wind_speed=obs_wind_speed,
        observe_outside_humidity=obs_outside_humidity,
        observe_outside_pressure=obs_outside_pressure,
        observe_wind_direction=obs_wind_direction,
        observe_rain=obs_rain,
        control_ventilation=True,
        control_thermostat_setpoints=True,
    )
    return ec
