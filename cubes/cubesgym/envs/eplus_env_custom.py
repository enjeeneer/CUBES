"""
Gym environment for customizable simulation with EnergyPlus.
"""

from sinergym.envs import EplusEnv
from sinergym.utils.rewards import LinearReward

import os
import gym
from cubes.cubesgym.simulators.custom_eplus_simulator import EnergyPlusCustom

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class EplusEnvCustom(EplusEnv):
    """_summary_

    Args:
        EplusEnv (_type_): _description_
    """

    def __init__(
        # pylint: disable=super-init-not-called
        self,
        idf_file: str,
        weather_file: str,
        observation_space: gym.spaces.Box = gym.spaces.Box(
            low=-5e6, high=5e6, shape=(4,)
        ),
        observation_variables: List[str] = None,
        action_space: Union[gym.spaces.Box, gym.spaces.Discrete] = gym.spaces.Box(
            low=0, high=0, shape=(0,)
        ),
        action_variables: List[str] = None,
        action_mapping: Dict[int, Tuple[float, ...]] = None,
        weather_variability: Optional[Tuple[float]] = None,
        reward: Any = LinearReward,
        reward_kwargs: Optional[Dict[str, Any]] = None,
        act_repeat: int = 1,
        max_ep_data_store_num: int = 10,
        action_definition: Optional[Dict[str, Any]] = None,
        env_name: str = "eplus-env-v1",
        config_params: Optional[Dict[str, Any]] = None,
        action_remapping: Dict[str, Any] = None,
    ):
        """Environment with EnergyPlus simulator. Overwrite base class constructor
        to be allow use of custom input files

        Args:
            idf_file (str): Name of the IDF file with the building definition.
            weather_file (str): Name of the EPW file for weather conditions.
            observation_space (gym.spaces.Box, optional):
                Gym Observation Space definition.
                Defaults to an empty observation_space (no control).
            observation_variables (List[str], optional):
                List with variables names in IDF.
                Defaults to an empty observation variables (no control).
            action_space (Union[gym.spaces.Box, gym.spaces.Discrete], optional):
                Gym Action Space definition.
                Defaults to an empty action_space (no control).
            action_variables (List[str],optional):
                Action variables to be controlled in IDF,
                if that actions names have not been configured manually in IDF,
                you should configure or use extra_config. Default to empty List.
            action_mapping (Dict[int, Tuple[float, ...]], optional):
                Action mapping list for discrete actions spaces only.
                Defaults to empty list.
            weather_variability (Optional[Tuple[float]], optional):
                Tuple with sigma, mu and tao of the Ornstein-Uhlenbeck process
                to be applied to weather data. Defaults to None.
            reward (Any, optional): Reward function instance used for agent feedback.
                Defaults to LinearReward.
            reward_kwargs (Optional[Dict[str, Any]], optional):
                Parameters to be passed to the reward function. Defaults to empty dict.
            act_repeat (int, optional):
                Number of timesteps that an action is repeated in the simulator,
                regardless of the actions it receives during that repetition interval.
            max_ep_data_store_num (int, optional):
                Number of last sub-folders (one for each episode)
                generated during execution on the simulation.
            env_name (str, optional): Env name used for working directory generation.
                Defaults to eplus-env-v1.
            config_params (Optional[Dict[str, Any]], optional):
                Dictionary with all extra configuration for simulator. Defaults to None.
        """

        # ---------------------------------------------------------------------------- #
        #                          Energyplus, BCVTB and paths                         #
        # ---------------------------------------------------------------------------- #
        eplus_path = os.environ["EPLUS_PATH"]
        bcvtb_path = os.environ["BCVTB_PATH"]

        self.idf_path = idf_file
        self.weather_path = weather_file

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

        self.original_obs = observation_variables
        self.original_obs = [
            "year",
            "month",
            "day",
            "hour",
        ] + self.original_obs

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
            self.action_remapping = action_remapping

        # ---------------------------------------------------------------------------- #
        #                                    Reward                                    #
        # ---------------------------------------------------------------------------- #
        self.reward_fn = reward(self, **reward_kwargs)
        self.obs_dict = None
        self.old_obs_dict = None

        # ---------------------------------------------------------------------------- #
        #                        Environment definition checker                        #
        # ---------------------------------------------------------------------------- #

        self._check_eplus_env()


    # ---------------------------------------------------------------------------- #
    #                                     STEP                                     #
    # ---------------------------------------------------------------------------- #
    def step(
        self, action: Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Sends action to the environment

        Args:
            action (Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]):
            Action selected by the agent.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]:
            Observation for next timestep, reward obtained,
            Whether the episode has ended or not and a dictionary with extra information
        """

        # Save old observations
        if self.obs_dict:
            self.old_obs_dict = self.obs_dict.copy()

        # Get action
        action_ = self._get_action(action)
        # Send action to the simulator
        self.simulator.logger_main.debug(action_)
        # time_info = (current simulation year, month, day, hour, time_elapsed)
        time_elapsed, obs, done = self.simulator.step(action_)

        # Create dictionary with observation
        self.obs_dict = dict(zip(self.original_obs, obs))

        # Calculate reward
        reward, terms = self.reward_fn()

        if "done" in terms.keys():
            done = terms.get("done")

        # Extra info
        info = {
            "timestep": int(time_elapsed / self.simulator.get_eplus_run_stepsize()),
            "time_elapsed": int(time_elapsed),
            "year": self.obs_dict["year"],
            "month": self.obs_dict["month"],
            "day": self.obs_dict["day"],
            "hour": self.obs_dict["hour"],
            "emissions": terms.get("emissions"),
            "reward_emissions": terms.get("reward_emissions"),
            "reward_comfort": terms.get("reward_comfort"),
            "reward_air_quality": terms.get("reward_air_quality"),
            "abs_comfort": terms.get("abs_comfort"),
            "temperatures": terms.get("temperatures"),
            "abs_air_quality": terms.get("abs_air_quality"),
            "air_qualities": terms.get("air_qualities"),
            "t_violation": terms.get("t_violation"),
            "aq_violation": terms.get("aq_violation"),
            "heating_delta_T": terms.get("heating_delta_T"),
            "heating_beyond_comf_delta_T": terms.get("heating_beyond_comf_delta_T"),
            "violation_delta_T": terms.get("violation_delta_T"),
            "violation_delta_aq": terms.get("violation_delta_aq"),
            "heating_service": terms.get("heating_service"),
            "max_heating_service": terms.get("max_heating_service"),
            "out_temperature": self.obs_dict[
                "Site Outdoor Air Drybulb Temperature(Environment)"
            ],
            "action_": action_,
        }

        return np.array(obs, dtype=np.float32), reward, done, info

    def _get_action(self, action: Any):
        """Transform the action for sending it to the simulator."""

        # Get action depending on flag_discrete
        if self.flag_discrete:
            # Index for action_mapping
            if np.issubdtype(type(action), np.integer):
                if isinstance(action, int):
                    setpoints = self.action_mapping[action]
                else:
                    setpoints = self.action_mapping[action.item()]
            # Manual action
            elif isinstance(action, (tuple, list)):
                # stable-baselines DQN bug prevention
                if len(action) == 1:
                    setpoints = self.action_mapping[action.item()]
                else:
                    setpoints = action
            elif isinstance(action, np.ndarray):
                setpoints = self.action_mapping[action.item()]
            else:
                print("ERROR: ", type(action))
            action_ = list(setpoints)
        else:
            # transform action to setpoints simulation
            action_ = self._setpoints_transform(action)

        return action_

    def _setpoints_transform(
        self, action: Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]
    ) -> Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]:
        """This method transforms an action defined in gym
        (-1,1 in all continuous environment) action space
        to simulation real action space.

        Args:
            action (Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]):
            Action received in environment

        Returns:
            Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]:
            Action transformed in simulator action space.
        """
        action_ = []

        for i, value in enumerate(action):
            if self.action_space.low[i] <= value <= self.action_space.high[i]:
                a_max_min = self.action_space.high[i] - self.action_space.low[i]

                # apply action remapping
                override = False
                if self.action_remapping:
                    if self.variables["action"][i] in self.action_remapping.keys():
                        remap = self.action_remapping[self.variables["action"][i]]
                        if self.obs_dict:
                            if self.old_obs_dict:
                                obs_dict = self.old_obs_dict
                            else:
                                obs_dict = self.obs_dict

                            condts_met = True
                            for condt in remap[0]:
                                if not condt[1](obs_dict[condt[0]],condt[2]):
                                    condts_met = False

                            if condts_met:
                                sp_max_min = remap[2] - remap[1]
                                action_.append(
                                    remap[1]
                                    + (value - self.action_space.low[i])
                                    * sp_max_min
                                    / a_max_min
                                )
                                override = True

                if not override:
                    sp_max_min = (
                        self.setpoints_space.high[i] - self.setpoints_space.low[i]
                    )

                    action_.append(
                        self.setpoints_space.low[i]
                        + (value - self.action_space.low[i]) * sp_max_min / a_max_min
                    )
            else:
                # If action is outer action_space already, it don't need
                # transformation
                action_.append(value)

        return action_
