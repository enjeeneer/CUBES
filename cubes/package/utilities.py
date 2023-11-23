# pylint: disable-all
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
from io import StringIO
from eppy.results import readhtml


def _decodeline(line, encoding="utf-8"):
    """decodes bytes to string, if line is not bytes, line is returned

    It will first attempt to decode line with value of `encoding`. If that fails, it will try with encoding="ISO-8859-2". If that fails, it will return line.

    Why is it trying encoding="ISO-8859-2". Looks like E+ uses this encoding in some example files and which is then output in the HTML file



    Parameters
    ----------
    line : str, bytes
    encoding : str

    Returns
    -------
    line : str
        decoded line
    """
    try:
        return line.decode(encoding)
    except (AttributeError, UnicodeDecodeError) as e:
        if e.__class__ == UnicodeDecodeError:
            # encoding could be ISO-8859-2 in e+ html
            return _decodeline(line, encoding="ISO-8859-2")
        else:
            return line


def getnexttable(fhandle):
    """get the next table in the html file

    Continues to read the file line by line and collects lines from the start of the next table until the end of the table

    Parameters
    ----------
    fhandle : file like object
        A file handle to the E+ HTML table file

    Returns
    -------
    table : str
        The table in HTML format
    """
    lines = fhandle
    tablelines = []
    for line in lines:
        line = _decodeline(line)
        if line.strip().startswith("<table"):
            tablelines.append(line)
            break
    for line in lines:
        line = _decodeline(line)
        tablelines.append(line)
        if line.strip().startswith("</table"):
            break
    return "".join(tablelines)


def tablebyname(filehandle, header):
    """fast extraction of the table using the header to identify the table

    This function reads only one table from the HTML file. This is in contrast to `results.readhtml.titletable` that will read all the tables into memory and allows you to interactively look thru them. The function `results.readhtml.titletable` can be very slow on large HTML files.

    This function is useful when you know which file you are looking for. It looks for the title line that is in bold just before the table. Some tables don't have such a title in bold. This function will not work for tables that don't have a title in bold

    Parameters
    ----------
    fhandle : file like object
        A file handle to the E+ HTML table file
    header: str
        This is the title of the table you are looking for

    Returns
    -------
    titleandtable : (str, list)
        - (title, table)
            - title = previous item with a <b> tag
            - table = rows -> [[cell1, cell2, ..], [cell1, cell2, ..], ..]
    """
    htmlheader = f"<b>{header}</b><br><br>"

    with filehandle:
        for line in filehandle:
            line = _decodeline(line)
            if line.strip() == htmlheader:
                justtable = getnexttable(filehandle)
                thetable = f"{htmlheader}\n{justtable}"
                break

    filehandle = StringIO(thetable)
    htables = readhtml.titletable(filehandle)
    try:
        return list(htables[0])
    except IndexError as e:
        None


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
    # shutil.rmtree(temp_output_path)
    with open(temp_output_path + "/eplustbl.htm", "r") as file:
        table = tablebyname(file, "Component Sizing Information")
        values = pd.DataFrame(table[1][1:], columns=table[1][0])
        print(values)
        print(
            table[["Component Name"] == "MAIN BOILER"]["Design Size Nominal Capacity"]
        )

    # test_data = pd.read_csv(temp_output_path + "/eplusout.csv")

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
        t_idx = np.arange(0, len(temp_data) * 6, 6)
        t_idx = np.append(t_idx, t_idx[-1] + 5)
        temp_data = temp_data.append(
            temp_data.loc[temp_data.index[-1]], ignore_index=True
        )
        t_idx_int = np.arange(0, len(temp_data) * 6)
        temp_data_int = np.interp(t_idx_int, t_idx, temp_data["T"])

        for tfh in temperature_forecast_hours:
            forecast = np.zeros(len(temp_data_int))
            n_ts = int(tfh * 6)

            for i in range(len(temp_data_int)):
                if i < len(temp_data_int) - n_ts:
                    forecast[i] = temp_data_int[i + n_ts] + np.random.normal(
                        0, sigma(tfh), None
                    )
                else:
                    forecast[i] = temp_data_int[i]

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
) -> float:
    """this function produces grid carbon forecast files
    Numbers based on following assumptions:
    - perfect forecast (should be changed)
    Returns:
        max_emissions: maximum emissions in the forecast
    """

    grid_data = pd.read_csv(
        get_grid_file_path(grid_carbon_file_name),
        usecols=[1],
        names=["gCO2/kWh"],
        header=0,
    )

    max_emissions_factor = grid_data["gCO2/kWh"].max()

    if grid_carbon_forecast_hours:

        for gfh in grid_carbon_forecast_hours:
            forecast = np.zeros(len(grid_data))
            n_ts = int(gfh * 6)
            for i in range(len(grid_data)):
                if i < len(grid_data) - n_ts:
                    forecast[i] = grid_data.loc[i + n_ts, "gCO2/kWh"]
                else:
                    forecast[i] = grid_data.loc[i, "gCO2/kWh"]

            np.savetxt(
                get_grid_forecast_file_path(env_files_dir=env_files_dir, hours=gfh),
                forecast,
                fmt="%10.2f",
                newline=",\n",
            )

    return max_emissions_factor


def get_envconfig_leiden(
    case_number, files_dir: str, rbc_setup=False, short_test=False
):
    control_vent = True
    observe_vent = True
    control_observe_battery = False
    negative_emissions_for_export = False
    observe_surplus_electricity = False
    if case_number in [3, 4, 8, 9, 13, 14, 18, 19]:
        control_vent = False
        observe_vent = False
    if case_number >= 10:
        control_observe_battery = True
    if case_number < 5:
        observe_outside_temperature_in_x_hours_forecast = [1]
        observe_grid_carbon_in_x_hours_forecast = []
    else:
        observe_outside_temperature_in_x_hours_forecast = [1, 2, 3, 4, 5, 6, 12]
        observe_grid_carbon_in_x_hours_forecast = [1, 2, 3, 4, 5, 6, 12]
    if case_number >= 15:
        negative_emissions_for_export = True
        observe_surplus_electricity = True

    ec = EnvConfig(
        files_dir=files_dir,
        observe_zone_temperature=True,
        observe_electricity_demand=True,
        observe_net_purchased_electricity=True,
        observe_total_purchased_electricity=True,
        observe_total_surplus_electricity=True,
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
        observe_solar_irradiance=rbc_setup,
        observe_zone_humidity=rbc_setup,
        observe_wind_speed=rbc_setup,
        observe_outside_humidity=rbc_setup,
        observe_rain=rbc_setup,
        negative_emissions_for_export=negative_emissions_for_export,
        observe_surplus_electricity=observe_surplus_electricity,
    )
    if short_test:
        ec.episode_end_date = (15, 1)
    return ec
