"""This module finds a weather file according to specification
and writes it to the case directory"""

from cubes.package import constants
import shutil


def get_weather_file():
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
