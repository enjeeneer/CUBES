"""Functions to package up IDF files with weather
etc without creating a gym environment"""

from cubes.package import weather, constants, utilities, variables


def prepare_simulation(idf, building_config, envconfig):
    idf = weather.get_weather_file_and_adapt_idf(idf, building_config)
    idf = utilities.get_rdd_and_expand_idf(idf)

    # get observation variables
    observation_variables = variables.get_observation_variables(idf, envconfig)[1]

    idf = variables.clear_output_variables(idf)
    idf = variables.add_output_variables_to_idf(idf, observation_variables)

    return idf, constants.weather_file_path
