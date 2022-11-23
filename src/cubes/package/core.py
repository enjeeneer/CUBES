"""Main module to package up IDF files with weather etc and create a gym environment"""
from cubes.construct.core import sample_idf

# import utilities
# import weather
import os
from pathlib import Path

def make_test_env():
    # get idf file
    idf_file = sample_idf()

    # get rdd file
    # rdd_file = utilities.get_rdd_file(idf_file)

    # save it somewhere

    cwd_path = os.getcwd()
    pkg_data_path = os.path.join(cwd_path, "input_case_1")
    Path(pkg_data_path).mkdir(parents=True, exist_ok=True)

    idf_file.save(filename=pkg_data_path + "test1.idf")

    # get weather file and save it
    # weather_file = weather.get_weather_file()

    # changes to idf file for agent interface

    # define action and observation spaces + rewards

    # register environemnt
