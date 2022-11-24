"""Main module to package up IDF files with weather etc and create a gym environment"""
from cubes.construct.core import sample_idf
from cubes.package import weather, utilities


def make_test_env():

    # get idf file
    idf = sample_idf()
    # save it somewhere
    # idf.save(filename=constants.idf_file_path)

    # get weather file and save it
    weather.get_weather_file()

    # get rdd file and expanded
    utilities.get_rdd_file(idf)

    # changes to idf file for agent interface

    # define action and observation spaces + rewards

    # register environemnt
