"""
Class for connecting EnergyPlus with Python using Ptolomy server.

..  note::
    Modified from sinergym to be able to use own input files
"""


from typing import Any, Dict, List, Optional

from sinergym.simulators.eplus import EnergyPlus
from cubes.cubesgym.utils.config import (
    ConfigCustom as Config,
)


LOG_LEVEL_MAIN = "INFO"
LOG_LEVEL_EPLS = "FATAL"
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s"


class EnergyPlusCustom(EnergyPlus):
    """
    inergym.simulators.eplus.EnergPlus only modifying file paths
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
        super().__init__(
            eplus_path,
            weather_path,
            bcvtb_path,
            idf_path,
            env_name,
            variables,
            act_repeat,
            max_ep_data_store_num,
            action_definition,
            config_params,
        )

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
