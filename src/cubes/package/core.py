"""Main module to package up IDF files with weather etc and create a gym environment"""
from cubes.construct.core import sample_idf
from cubes.package import weather, utilities
from cubes.package.envconfig import EnvConfig


def make_test_env():

    # get idf file
    idf = sample_idf()
    # save it somewhere
    # idf.save(filename=constants.idf_file_path)

    # get weather file and save it
    weather.get_weather_file()

    # get rdd file and expanded
    utilities.get_rdd_file(idf)

    envconfig = EnvConfig(
        1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1
    )

    # changes to idf file for agent interface
    utilities.add_control_variables_to_idf(idf, envconfig)

    # define action and observation spaces + rewards

    # register environemnt
