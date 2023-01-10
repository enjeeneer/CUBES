"""This module finds a weather file according to specification
and writes it to the case directory"""

from cubes.package import constants
import shutil


def get_weather_file_and_adapt_idf(idf):
    """Find a weather file according to specs and copy it into case folder
    This should take arguments in the future, such as
    - location
    - year
    """

    shutil.copyfile(
        "/workspaces/elizabeth-homes/src/cubes/data/"
        "weather/cambridge_lat=52.25_lng=0.25_period=2021.epw",
        constants.weather_file_path,
    )

    shutil.copyfile(
        "/workspaces/elizabeth-homes/src/cubes/data/weather/dummy.ddy",
        constants.ddy_file_path,
    )

    # read first line of weather file and extract longitude, latitude,
    # time zone, and elevation
    with open(constants.weather_file_path, encoding="UTF-8") as f:
        first_line = f.readline().strip("\n").split(",")

    location = idf.idfobjects["SITE:LOCATION"][0]
    location.Name = "Cambridge 2021"
    location.Latitude = first_line[-4]
    location.Longitude = first_line[-3]
    location.Time_Zone = first_line[-2]
    location.Elevation = first_line[-1]

    return idf
