"""
    Overwrite the constructor of sinergym's Config class to change file paths
"""

from sinergym.utils.config import Config
from typing import Any, Dict, List, Optional

from cubes.cubesgym.utils.constants import (
    PKG_DATA_PATH,
)  # this is the difference between sinergym.utils.config.Config and ConfigCustom
import os


class ConfigCustom(Config):
    """Config object to manage extra configuration in Sinergym experiments.
    Customised to allow own input files

    :param _idf_path: IDF path origin for apply extra configuration.
    :param _weather_path: EPW path origin for apply weather to simulation.
    :param _ddy_path: DDY path origin for get DesignDays and weather Location
    :param experiment_path: Path for Sinergym experiment output
    :param episode_path: Path for Sinergym specific episode
        (before first simulator reset this param is None)
    :param max_ep_store: Number of episodes directories
        will be stored in experiment_path
    :param config: Dict config with extra configuration
        which is required to modify IDF model (may be None)
    :param _idd: IDD opyplus object to set up Epm
    :param building: opyplus Epm object with IDF model
    :param ddy_model: opyplus Epm object with DDY model
    :param weather_data: opyplus WeatherData object with EPW data
    :param action_definition: Dict with action definition to
        automatic building model preparation.
    """

    def __init__(
        self,
        idf_path: str,
        weather_path: str,
        variables: Dict[str, List[str]],
        env_name: str,
        max_ep_store: int,
        action_definition: Optional[Dict[str, Any]],
        extra_config: Dict[str, Any],
    ):

        super().__init__(
            idf_path,
            weather_path,
            variables,
            env_name,
            max_ep_store,
            action_definition,
            extra_config,
        )

        self._rdd_path = os.path.join(
            PKG_DATA_PATH,
            "variables",
            self._idf_path.split("/")[-1].split(".idf")[0] + ".rdd",
        )
