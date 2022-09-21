"""
    Overwrite the constructor of sinergym's Config class to change file paths
"""

from sinergym.utils.config import Config
from typing import Any, Dict, List, Optional
import xml.etree.cElementTree as ElementTree
from opyplus import Epm, Idd, WeatherData

from elhogym.utils.constants import (
    PKG_DATA_PATH,
)  # this is the difference between sinergym.utils.config.Config and ConfigCustom
import os
import pandas


class ConfigCustom(Config):
    """Config object to manage extra configuration in Sinergym experiments.
    Customised to allow own input files

    :param _idf_path: IDF path origin for apply extra configuration.
    :param _weather_path: EPW path origin for apply weather to simulation.
    :param _ddy_path: DDY path origin for get DesignDays and weather Location
    :param experiment_path: Path for Sinergym experiment output
    :param episode_path: Path for Sinergym specific episode (before first simulator reset this param is None)
    :param max_ep_store: Number of episodes directories will be stored in experiment_path
    :param config: Dict config with extra configuration which is required to modify IDF model (may be None)
    :param _idd: IDD opyplus object to set up Epm
    :param building: opyplus Epm object with IDF model
    :param ddy_model: opyplus Epm object with DDY model
    :param weather_data: opyplus WeatherData object with EPW data
    :param action_definition: Dict with action definition to automatic building model preparation.
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

        self._idf_path = idf_path
        self._weather_path = weather_path
        # RDD file name is deducible using idf name (only change .idf by .rdd)
        self._rdd_path = os.path.join(
            PKG_DATA_PATH,
            "variables",
            self._idf_path.split("/")[-1].split(".idf")[0] + ".rdd",
        )

        # DDY path is deducible using weather_path (only change .epw by .ddy)
        self._ddy_path = self._weather_path.split(".epw")[0] + ".ddy"
        self.experiment_path = self.set_experiment_working_dir(env_name)
        self.episode_path = None
        self.max_ep_store = max_ep_store

        # Set config and action definition as config attribute
        self.config = extra_config
        self.action_definition = action_definition

        # Variables XML Tree (empty at the beginning)
        self.variables = variables
        self.variables_tree = ElementTree.Element("BCVTB-variables")

        # Opyplus objects
        self._idd = Idd(os.path.join(os.environ["EPLUS_PATH"], "Energy+.idd"))
        self.building = Epm.from_idf(
            self._idf_path, idd_or_version=self._idd, check_length=False
        )
        self.ddy_model = Epm.from_idf(
            self._ddy_path, idd_or_version=self._idd, check_length=False
        )
        self.weather_data = WeatherData.from_epw(self._weather_path)

        # Extract idf zone names
        self.idf_zone_names = []
        for idf_zone in self.building.Zone:
            self.idf_zone_names.append(idf_zone.name.lower())
        # Extract rdd observation variables names
        data = pandas.read_csv(self._rdd_path, skiprows=1)
        self.rdd_variables_names = list(
            map(
                lambda name: name.split(" [")[0], data["Variable Name [Units]"].tolist()
            )
        )

        # Check observation variables definition
        self._check_observation_variables()
        # Check config definition
        self._check_eplus_config()
