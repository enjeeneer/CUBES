"""This module finds a weather file according to specification
and writes it to the case directory"""

from cubes.package import constants
from cubes.constants import package_directory
from cubes.construct.buildingconfig import BuildingConfig

import shutil
from geomeppy import IDF


def get_weather_file_name(building_config: BuildingConfig):

    try:
        weather_file_name = constants.weather_file_dict[building_config.location]
    except LookupError:
        print(
            f"No weather file for location {building_config.location}. "
            "Using Cambridge, UK weather"
        )

        weather_file_name = constants.weather_file_dict["Cambridge"]

    return weather_file_name


def get_weather_file_info(building_config: BuildingConfig):
    weather_file_path = building_config.weather_file_path

    with open(
        weather_file_path,
        encoding="UTF-8",
    ) as f:
        first_line = f.readline().strip("\n").split(",")

    location_etc = {
        "Latitude": float(first_line[-4]),
        "Longitude": float(first_line[-3]),
        "Time Zone": float(first_line[-2]),
        "Elevation": float(first_line[-1]),
    }

    return location_etc


def get_weather_file_and_adapt_idf(idf: IDF, building_config: BuildingConfig):
    """Find a weather file according to specs and copy it into case folder
    This should take arguments in the future, such as
    - location
    - year
    """

    weather_file_path = building_config.weather_file_path

    shutil.copyfile(
        weather_file_path,
        constants.weather_file_path,
    )

    shutil.copyfile(
        package_directory + "/data/weather/dummy.ddy",
        constants.ddy_file_path,
    )

    # read first line of weather file and extract longitude, latitude,
    # time zone, and elevation
    weather_file_info = get_weather_file_info(building_config)

    location = idf.idfobjects["SITE:LOCATION"][0]
    location.Name = building_config.location
    location.Latitude = weather_file_info["Latitude"]
    location.Longitude = weather_file_info["Longitude"]
    location.Time_Zone = weather_file_info["Time Zone"]
    location.Elevation = weather_file_info["Elevation"]

    idf.epw = constants.weather_file_path

    return idf
