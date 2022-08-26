"""
Gym environment for customizable simulation with EnergyPlus.
"""

from sinergym.envs import EplusEnv
from sinergym.utils.rewards import LinearReward
import os
import gym
import numpy as np
from constants import PKG_DATA_PATH
from custom_eplus_simulator import EnergyPlusCustom

from typing import Any, Dict, List, Optional, Tuple, Union


class EplusEnvCustom(EplusEnv):
    """_summary_

    Args:
        EplusEnv (_type_): _description_
    """

    def __init__(
        self,
        idf_file: str,
        weather_file: str,
        observation_space: gym.spaces.Box = gym.spaces.Box(
            low=-5e6, high=5e6, shape=(4,)
        ),
        observation_variables: List[str] = [],
        action_space: Union[gym.spaces.Box, gym.spaces.Discrete] = gym.spaces.Box(
            low=0, high=0, shape=(0,)
        ),
        action_variables: List[str] = [],
        action_mapping: Dict[int, Tuple[float, ...]] = {},
        weather_variability: Optional[Tuple[float]] = None,
        reward: Any = LinearReward,
        reward_kwargs: Optional[Dict[str, Any]] = {},
        act_repeat: int = 1,
        max_ep_data_store_num: int = 10,
        action_definition: Optional[Dict[str, Any]] = None,
        env_name: str = "eplus-env-v1",
        config_params: Optional[Dict[str, Any]] = None,
    ):
        """Environment with EnergyPlus simulator. Overwrite base class constructor
        to be allow use of custom input files

        Args:
            idf_file (str): Name of the IDF file with the building definition.
            weather_file (str): Name of the EPW file for weather conditions.
            observation_space (gym.spaces.Box, optional): Gym Observation Space definition. Defaults to an empty observation_space (no control).
            observation_variables (List[str], optional): List with variables names in IDF. Defaults to an empty observation variables (no control).
            action_space (Union[gym.spaces.Box, gym.spaces.Discrete], optional): Gym Action Space definition. Defaults to an empty action_space (no control).
            action_variables (List[str],optional): Action variables to be controlled in IDF, if that actions names have not been configured manually in IDF, you should configure or use extra_config. Default to empty List.
            action_mapping (Dict[int, Tuple[float, ...]], optional): Action mapping list for discrete actions spaces only. Defaults to empty list.
            weather_variability (Optional[Tuple[float]], optional): Tuple with sigma, mu and tao of the Ornstein-Uhlenbeck process to be applied to weather data. Defaults to None.
            reward (Any, optional): Reward function instance used for agent feedback. Defaults to LinearReward.
            reward_kwargs (Optional[Dict[str, Any]], optional): Parameters to be passed to the reward function. Defaults to empty dict.
            act_repeat (int, optional): Number of timesteps that an action is repeated in the simulator, regardless of the actions it receives during that repetition interval.
            max_ep_data_store_num (int, optional): Number of last sub-folders (one for each episode) generated during execution on the simulation.
            env_name (str, optional): Env name used for working directory generation. Defaults to eplus-env-v1.
            config_params (Optional[Dict[str, Any]], optional): Dictionary with all extra configuration for simulator. Defaults to None.
        """

        # ---------------------------------------------------------------------------- #
        #                          Energyplus, BCVTB and paths                         #
        # ---------------------------------------------------------------------------- #
        eplus_path = os.environ["EPLUS_PATH"]
        bcvtb_path = os.environ["BCVTB_PATH"]
        self.pkg_path = PKG_DATA_PATH

        self.idf_path = os.path.join(self.pkg_path, "buildings", idf_file)
        self.weather_path = os.path.join(self.pkg_path, "weather", weather_file)

        # ---------------------------------------------------------------------------- #
        #                             Variables definition                             #
        # ---------------------------------------------------------------------------- #
        self.variables = {}
        self.variables["observation"] = observation_variables
        self.variables["action"] = action_variables

        # ---------------------------------------------------------------------------- #
        #                                   Simulator                                  #
        # ---------------------------------------------------------------------------- #
        self.simulator = EnergyPlusCustom(
            env_name=env_name,
            eplus_path=eplus_path,
            bcvtb_path=bcvtb_path,
            idf_path=self.idf_path,
            weather_path=self.weather_path,
            variables=self.variables,
            act_repeat=act_repeat,
            max_ep_data_store_num=max_ep_data_store_num,
            action_definition=action_definition,
            config_params=config_params,
        )

        # ---------------------------------------------------------------------------- #
        #        Adding simulation date to observation (not needed in simulator)       #
        # ---------------------------------------------------------------------------- #

        self.variables["observation"] = [
            "year",
            "month",
            "day",
            "hour",
        ] + self.variables["observation"]

        # ---------------------------------------------------------------------------- #
        #                              Weather variability                             #
        # ---------------------------------------------------------------------------- #
        self.weather_variability = weather_variability

        # ---------------------------------------------------------------------------- #
        #                               Observation Space                              #
        # ---------------------------------------------------------------------------- #
        self.observation_space = observation_space

        # ---------------------------------------------------------------------------- #
        #                                 Action Space                                 #
        # ---------------------------------------------------------------------------- #
        # Action space type
        self.flag_discrete = isinstance(action_space, gym.spaces.Discrete)

        # Discrete
        if self.flag_discrete:
            self.action_mapping = action_mapping
            self.action_space = action_space
        # Continuous
        else:
            # Defining action values setpoints (one per value)
            self.setpoints_space = action_space

            self.action_space = gym.spaces.Box(
                # continuous_action_def[2] --> shape
                low=np.repeat(-1, action_space.shape[0]),
                high=np.repeat(1, action_space.shape[0]),
                dtype=action_space.dtype,
            )

        # ---------------------------------------------------------------------------- #
        #                                    Reward                                    #
        # ---------------------------------------------------------------------------- #
        self.reward_fn = reward(self, **reward_kwargs)
        self.obs_dict = None

        # ---------------------------------------------------------------------------- #
        #                        Environment definition checker                        #
        # ---------------------------------------------------------------------------- #

        self._check_eplus_env()
