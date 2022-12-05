"""Main module to package up IDF files with weather etc and create a gym environment"""
from cubes.construct.core import sample_idf
from cubes.package import weather, utilities, variables
from cubes.package.envconfig import EnvConfig


def make_test_env():

    # get idf file
    idf = sample_idf()
    # save it somewhere
    # idf.save(filename=constants.idf_file_path)

    # get weather file and save it
    weather.get_weather_file()

    # get rdd file and expand idf file (not yet done)
    utilities.get_rdd_file(idf)

    envconfig = EnvConfig()

    # changes to idf file for agent interface
    idf, action_variables = variables.add_control_variables_to_idf(idf, envconfig)

    # get observation variables

    # define action and observation spaces + rewards
    variables.get_space(action_variables)

    # register environemnt
