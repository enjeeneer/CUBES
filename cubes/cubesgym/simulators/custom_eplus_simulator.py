"""
Class for connecting EnergyPlus with Python using Ptolomy server.

..  note::
    Modified from sinergym to be able to use own input files
"""

import _thread
import os
import socket
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from sinergym.utils.logger import Logger
from sinergym.utils.common import get_current_time_info

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


    def reset(
        self, weather_variability: Optional[Tuple[float, float, float]] = None
    ) -> Tuple[float, List[float], bool]:
        """Resets the environment.
        same as sinergym but increased buffer size of socket.recv()
        This method does the following:
        1. Makes a new EnergyPlus working directory.
        2. Copies .idf and variables.cfg file to the working directory.
        3. Creates the socket.cfg file in the working directory.
        4. Creates the EnergyPlus subprocess.
        5. Establishes the socket connection with EnergyPlus.
        6. Reads the first sensor data from the EnergyPlus.
        7. Uses a new weather file if passed.

        Args:
            weather_variability (Optional[Tuple[float, float, float]], optional):
            Tuple with the sigma, mean and tau for OU process. Defaults to None.

        Returns:
            Tuple[float, List[float], bool]: The first element is a value +
            with simulation time elapsed;
            the second element consist on EnergyPlus results in a 1-D list
            corresponding to the variables in
            variables.cfg and year, month, day and hour in simulation.
            The last element is a boolean indicating whether the episode terminates.
        """
        # End the last episode if exists
        if self._episode_existed:
            self._end_episode()
            self.logger_main.info(
                'EnergyPlus episode completed successfully. ')
            self._epi_num += 1

        # Create EnergyPlus simulation process
        self.logger_main.info('Creating new EnergyPlus simulation episode...')
        # Creating episode working dir
        eplus_working_dir = self._config.set_episode_working_dir()
        # Getting IDF, WEATHER, VARIABLES and OUTPUT path for current episode
        eplus_working_idf_path = self._config.save_building_model()
        # pylint: disable=invalid-name
        _ = self._config.save_variables_cfg()
        eplus_working_out_path = (eplus_working_dir + '/' + 'output')
        eplus_working_weather_path = self._config.apply_weather_variability(
            variation=weather_variability)

        self._create_socket_cfg(self._host,
                                self._port,
                                eplus_working_dir)
        # Create the socket.cfg file in the working dir
        self.logger_main.info('EnergyPlus working directory is in %s',eplus_working_dir)
        # Create new random weather file in case variability was specified
        # noise always from original EPW

        # Select new weather if it is passed into the method
        eplus_process = self._create_eplus(
            self._eplus_path,
            eplus_working_weather_path,
            eplus_working_idf_path,
            eplus_working_out_path,
            eplus_working_dir)
        self.logger_main.debug(
            'EnergyPlus process is still running ? %r',
            self._get_is_subprocess_running(eplus_process))
        self._eplus_process = eplus_process

        # Log EnergyPlus output
        eplus_logger = Logger().getLogger(
            f'EPLUS_ENV_{self._env_name}_{self._thread_name}'
            f'-EPLUSPROCESS_EPI_{self._epi_num}',
            LOG_LEVEL_EPLS, LOG_FMT)
        _thread.start_new_thread(self._log_subprocess_info,
                                 (eplus_process.stdout,
                                  eplus_logger))
        _thread.start_new_thread(self._log_subprocess_err,
                                 (eplus_process.stderr,
                                  eplus_logger))

        # Establish connection with EnergyPlus
        # Establish connection with client
        conn, addr = self._socket.accept()
        self.logger_main.debug('Got connection from %s at port %d.', addr[0],addr[1])
        # Start the first data exchange
        rcv_1st = conn.recv(16384).decode(encoding='ISO-8859-1')
        self.logger_main.debug(
            'Got the first message successfully: %s', rcv_1st)
        # pylint: disable=invalid-name
        version, flag, _, _, _, cur_sim_tim, dblist \
            = self._disassembleMsg(rcv_1st)
        # get time info in simulation
        time_info = get_current_time_info(self._config.building, cur_sim_tim)
        # Add time_info date in the end of the Energyplus observation
        dblist = time_info + dblist
        # Remember the message header, useful when send data back to EnergyPlus
        self._eplus_msg_header = [version, flag]
        self._curSimTim = cur_sim_tim
        # Check if episode terminates
        is_terminal = False
        if cur_sim_tim >= self._eplus_one_epi_len:
            is_terminal = True
        # Change some attributes
        self._conn = conn
        self._eplus_working_dir = eplus_working_dir
        self._episode_existed = True
        # Check termination
        if is_terminal:
            self._end_episode()

        return (cur_sim_tim, dblist, is_terminal)

    def step(self, action: Union[int, float, np.integer, np.ndarray, List[Any],
                                 Tuple[Any]]
             ) -> Tuple[float, List[float], bool]:
        """Executes a given action.
        This method does the following:
        1. Sends a list of floats to EnergyPlus.
        2. Receives EnergyPlus results for the next step (state).

        Args:
            action (Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]): Control actions that will be passed to EnergyPlus.

        Raises:
            RuntimeError: When you try to step in an terminated episode (you should be reset before).

        Returns:
            Tuple[float, List[float], bool]: The first element is a float with simulation time elapsed;
            the second element consist on EnergyPlus results in a 1-D list corresponding to the variables in
            variables.cfg. The last element is a boolean indicating whether the episode terminates.
        """
        # Check if terminal
        if self._curSimTim >= self._eplus_one_epi_len:
            raise RuntimeError(
                'You are trying to step in a terminated episode (do reset before).')
        # Send to EnergyPlus
        act_repeat_i = 0
        is_terminal = False
        cur_sim_tim = self._curSimTim

        while act_repeat_i < self._act_repeat and (not is_terminal):
            self.logger_main.debug('Perform one step.')
            header = self._eplus_msg_header
            run_flag = 0  # 0 is normal flag
            tosend = self._assembleMsg(header[0], run_flag, len(action), 0,
                                       0, cur_sim_tim, action)
            self._conn.send(tosend.encode())
            # Recieve from EnergyPlus
            rcv = self._conn.recv(16384).decode(encoding='ISO-8859-1')
            self.logger_main.debug('Got message successfully: %s', rcv)
            # Process received msg
            _,_,_,_,_,cur_sim_tim, dblist = self._disassembleMsg(rcv)
            if cur_sim_tim >= self._eplus_one_epi_len:
                is_terminal = True
                # Remember the last action
                self._last_action = action
            act_repeat_i += 1
        # Construct the return, which is the state observation of the last step
        # plus the integral item
        # get time info in simulation
        time_info = get_current_time_info(self._config.building, cur_sim_tim)
        # Add time_info to the observation (year,month,day and hour) at the
        # beggining
        dblist = time_info + dblist
        # Add terminal state
        # Change some attributes
        self._curSimTim = cur_sim_tim
        self._last_action = action

        return (cur_sim_tim, dblist, is_terminal)
