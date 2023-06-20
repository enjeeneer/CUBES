"""
Class for connecting EnergyPlus with Python using Ptolomy server.

..  note::
    Modified from sinergym to be able to use own input files
"""

import os
import socket
import threading
from typing import Any, Dict, List, Optional

from sinergym.utils.logger import Logger

from sinergym.simulators.eplus import EnergyPlus
from cubes.cubesgym.utils.config import (
    ConfigCustom as Config,
)


LOG_LEVEL_MAIN = "INFO"  # "DEBUG"
LOG_LEVEL_EPLS = "FATAL"
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s"


class EnergyPlusCustom(EnergyPlus):
    # pylint: disable=super-init-not-called
    """
    sinergym.simulators.eplus.EnergPlus only modifying file paths
    """

    def __init__(
        self,
        eplus_path: str,
        weather_path: str,
        bcvtb_path: str,
        idf_path: str,
        env_name: str,
        variables: Dict[str, List[str]],
        act_repeat: int = 1,
        max_ep_data_store_num: int = 10,
        action_definition: Optional[Dict[str, Any]] = None,
        config_params: Optional[Dict[str, Any]] = None,
    ):
        """EnergyPlus simulation class.

        Args:
            eplus_path (str):  EnergyPlus installation path.
            weather_path (str): EnergyPlus weather file (.epw) path.
            bcvtb_path (str): BCVTB installation path.
            idf_path (str): EnergyPlus input description file (.idf) path.
            env_name (str): The environment name.
            variables (Dict[str,List[str]]): Variables list with observation and action
                keys in a dictionary.
            act_repeat (int, optional): The number of times to repeat
                the control action. Defaults to 1.
            max_ep_data_store_num (int, optional): The number of simulation results
                to keep. Defaults to 10.
            config_params (Optional[Dict[str, Any]], optional):
                Dictionary with all extra configuration for simulator. Defaults to None.
        """

        self._env_name = env_name
        self._thread_name = threading.current_thread().getName()
        # pylint: disable=consider-using-f-string
        self.logger_main = Logger().getLogger(
            "EPLUS_ENV_%s_%s_ROOT" % (env_name, self._thread_name),
            LOG_LEVEL_MAIN,
            LOG_FMT,
        )

        # Set the environment variable for bcvtb
        os.environ["BCVTB_HOME"] = bcvtb_path
        # Create a socket for communication with the EnergyPlus
        self.logger_main.debug("Creating socket for communication...")
        self._socket = socket.socket()
        # Get local machine name
        self._host = socket.gethostname()
        # Bind to the host and any available port
        self._socket.bind((self._host, 0))
        # Get the port number
        sockname = self._socket.getsockname()
        self._port = sockname[1]
        # Listen on request
        self._socket.listen(60)
        # pylint: disable=logging-not-lazy
        self.logger_main.debug("Socket is listening on host %s port %d" % (sockname))

        # Path attributes
        self._eplus_path = eplus_path
        self._weather_path = weather_path
        self._idf_path = idf_path
        # Episode existed
        self._episode_existed = False

        self._epi_num = 0
        self._act_repeat = act_repeat
        self._max_ep_data_store_num = max_ep_data_store_num
        self._last_action = [0]

        # Creating models config (with extra params if exits)
        self._config = Config(
            idf_path=self._idf_path,
            weather_path=self._weather_path,
            variables=variables,
            env_name=self._env_name,
            max_ep_store=self._max_ep_data_store_num,
            action_definition=action_definition,
            extra_config=config_params,
        )

        # Annotate experiment path in simulator
        self._env_working_dir_parent = self._config.experiment_path
        # Setting an external interface if IDF building has not got.
        self.logger_main.info(
            "Updating idf ExternalInterface object if it is not present..."
        )
        self._config.set_external_interface()
        # Updating IDF file (Location and DesignDays) with EPW file
        self.logger_main.info(
            "Updating idf Site:Location and SizingPeriod:DesignDay(s) "
            "to weather and ddy file..."
        )
        self._config.adapt_idf_to_epw()
        # Updating IDF file Output:Variables with observation variables
        # specified in environment and variables.cfg construction
        self.logger_main.info(
            "Updating idf OutPut:Variable and variables XML tree model "
            "for BVCTB connection."
        )
        self._config.adapt_variables_to_cfg_and_idf()
        # Setting up extra configuration if exists
        self.logger_main.info(
            "Setting up extra configuration in building model if exists..."
        )
        self._config.apply_extra_conf()
        # Setting up action definition automatic manipulation if exists
        self.logger_main.info(
            "Setting up action definition in building model if exists..."
        )
        self._config.adapt_idf_to_action_definition()

        # In this lines Epm model is modified but no IDF is stored anywhere yet

        # Eplus run info
        (
            self._eplus_run_st_mon,
            self._eplus_run_st_day,
            self._eplus_run_st_year,
            self._eplus_run_ed_mon,
            self._eplus_run_ed_day,
            self._eplus_run_ed_year,
            self._eplus_run_st_weekday,
            self._eplus_n_steps_per_hour,
        ) = self._config._get_eplus_run_info()

        # Eplus one epi len
        self._eplus_one_epi_len = self._config._get_one_epi_len()
        # Stepsize in seconds
        self._eplus_run_stepsize = 3600 / self._eplus_n_steps_per_hour

    def get_eplus_run_stepsize(self):
        return self._eplus_run_stepsize
