"""Functions to package up IDF files with weather
etc without creating a gym environment"""

from cubes.package import weather, variables, utilities
from cubes.package.envconfig import EnvConfig
from cubes.construct.buildingconfig import BuildingConfig


def prepare_simulation(idf, building_config: BuildingConfig, env_config: EnvConfig):
    idf = weather.get_weather_file_and_adapt_idf(
        idf=idf, building_config=building_config, env_config=env_config
    )
    idf, heating_sys_cap = utilities.get_rdd_file(  # pylint: disable=unused-variable
        idf=idf, env_config=env_config, building_config=building_config
    )
    idf = utilities.set_simulation_parameters(idf)
    # get observation variables
    observation_variables = variables.get_observation_variables(
        idf, building_config, env_config
    )[2]

    idf = variables.clear_output_variables(idf)
    idf = variables.add_output_variables_to_idf(idf, observation_variables)

    # weather_file_path = env_config.files_dir + "/weather.epw"

    return idf  # , weather_file_path
