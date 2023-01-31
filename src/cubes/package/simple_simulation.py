"""Functions to package up IDF files with weather
etc without creating a gym environment"""

from cubes.package import weather, constants, utilities


def prepare_simulation(idf, building_config):
    idf = utilities.set_simulation_parameters(idf)
    idf = weather.get_weather_file_and_adapt_idf(idf, building_config)

    return idf, constants.weather_file_path
