"""custom wrapper to handle cubes reward function"""
import numpy as np
import gym
from copy import deepcopy
from datetime import datetime
from sinergym.utils.wrappers import LoggerWrapper
from cubes.cubesgym.utils.logger import CSVLogger

from typing import Any, Optional, List, Callable


class LoggerWrapperCubes(LoggerWrapper):
    """CSV Logger to interact with environment"""

    def __init__(
        self,
        env: Any,
        logger_class: Callable = CSVLogger,
        monitor_header: Optional[List[str]] = None,
        progress_header: Optional[List[str]] = None,
        flag: bool = True,
    ):
        super().__init__(env, logger_class, monitor_header, progress_header, flag)

        monitor_header_list = (
            monitor_header
            if monitor_header is not None
            else ["timestep"]
            + env.variables["observation"]
            + env.variables["action"]
            + [
                "time (seconds)",
                "reward",
                "emissions",
                "reward_emissions",
                "abs_comfort",
                "reward_comfort",
                "abs_air_quality",
                "reward_air_quality",
                "done",
            ]
        )
        self.monitor_header = ""
        for element_header in monitor_header_list:
            self.monitor_header += element_header + ","
        self.monitor_header = self.monitor_header[:-1]

        progress_header_list = (
            progress_header
            if progress_header is not None
            else [
                "episode_num",
                "cumulative_reward",
                "mean_reward",
                "cumulative_emissions",
                "mean_emissions",
                "cumulative_comfort_penalty",
                "mean_comfort_penalty",
                "cumulative_emissions_penalty",
                "mean_emissions_penalty",
                "cumulative_air_quality_penalty",
                "mean_air_quality_penalty",
                "comfort_violation (%)",
                "mean_comfort_violation",
                "std_comfort_violation",
                "cumulative_comfort_violation",
                "mean_air_quality_violation",
                "std_air_quality_violation",
                "cumulative_air_quality_violation",
                "length(timesteps)",
                "time_elapsed(seconds)",
            ]
        )
        self.progress_header = ""
        for element_header in progress_header_list:
            self.progress_header += element_header + ","
        self.progress_header = self.progress_header[:-1]

        # Create simulation logger, by default is active (flag=True)
        self.logger = logger_class(
            monitor_header=self.monitor_header,
            progress_header=self.progress_header,
            log_progress_file=env.simulator._env_working_dir_parent + "/progress.csv",
            flag=flag,
        )

class ObservationFilterCubes(gym.ObservationWrapper):
    """
    Wrapper to get rid of unwanted observation variablse.
    """

    def __init__(self, env: Any):
        super().__init__(env)
        # Save observation variables before wrapper
        self.original_observation_variables = deepcopy(
            self.variables["observation"]
        )
        self.remove_list = [
            "Environmental Impact Total CO2 Emissions Carbon Equivalent Mass(Site)"]

        # Update new shape
        new_shape = env.observation_space.shape[0] - len(self.remove_list)
        self.observation_space = gym.spaces.Box(
            low=-5e6, high=5e6, shape=(new_shape,), dtype=np.float32
        )


    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Removes observations from remove list

        Args:
            obs (np.ndarray): Original observation.

        Returns:
            np.ndarray: Transformed observation.
        """
        # Get obs_dict with observation variables from unwrapped env
        obs_dict = dict(zip(self.original_observation_variables, observation))

        # New obs dict with same values than obs_dict but with new fields with
        # None
        new_obs = {}
        for (
            key,
            value,
        ) in obs_dict.items():
            if key not in self.remove_list:
                new_obs[key] = value

        return np.array(list(new_obs.values()))


class DatetimeWrapperCubes(gym.ObservationWrapper):
    """
    Wrapper to substitute day value by is_weekend flag, and hour and
    month by sin and cos values. Observation space is updated automatically.
    """

    def __init__(self, env: Any):
        super().__init__(env)
        # Save observation variables before wrapper
        self.original_datetime_observation_variables = deepcopy(
            self.variables["observation"]
        )
        # Update new shape
        new_shape = env.observation_space.shape[0] + 3
        self.observation_space = gym.spaces.Box(
            low=-5e6, high=5e6, shape=(new_shape,), dtype=np.float32
        )
        # Update observation variables
        day_index = self.variables["observation"].index("day")
        self.variables["observation"][day_index] = "is_weekend"
        self.variables["observation"].insert(day_index + 1, "weekday")
        hour_index = self.variables["observation"].index("hour")
        self.variables["observation"][hour_index] = "hour_cos"
        self.variables["observation"].insert(hour_index + 1, "hour_sin")
        month_index = self.variables["observation"].index("month")
        self.variables["observation"][month_index] = "month_cos"
        self.variables["observation"].insert(month_index + 1, "month_sin")
        # Save observation variables after wrapper
        self.datetime_observation_variables = deepcopy(self.variables["observation"])

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Applies calculation in is_weekend flag, and sen and cos in hour and month

        Args:
            obs (np.ndarray): Original observation.

        Returns:
            np.ndarray: Transformed observation.
        """
        # Get obs_dict with observation variables from unwrapped env
        obs_dict = dict(zip(self.original_datetime_observation_variables, observation))

        # New obs dict with same values than obs_dict but with new fields with
        # None
        new_obs = dict.fromkeys(self.datetime_observation_variables)
        for (
            key,
            value,
        ) in obs_dict.items():
            if key in new_obs.keys():  # pylint: disable=consider-iterating-dictionary
                new_obs[key] = value
        dt = datetime(
            int(obs_dict["year"]),
            int(obs_dict["month"]),
            int(obs_dict["day"]),
            int(obs_dict["hour"]),
        )

        # Update obs
        new_obs["is_weekend"] = 1.0 if dt.isoweekday() in [6, 7] else 0.0
        new_obs["weekday"] = dt.weekday()
        new_obs["hour_cos"] = np.cos(2 * np.pi * obs_dict["hour"] / 24)
        new_obs["hour_sin"] = np.sin(2 * np.pi * obs_dict["hour"] / 24)
        new_obs["month_cos"] = np.cos(2 * np.pi * (obs_dict["month"] - 1) / 12)
        new_obs["month_sin"] = np.sin(2 * np.pi * (obs_dict["month"] - 1) / 12)

        return np.array(list(new_obs.values()))


class ScaleObservationCubes(gym.ObservationWrapper):
    """Observations normalized to range [-1, 1]."""

    def __init__(self,
                 env: Any):
        """Observations normalized to range [-1, 1].

        Args:
            env (Any): Original environment.
        """
        super().__init__(env)
        self.unwrapped_observation = None
        self.obs_space = env.observation_space

    def observation(self, observation) -> np.ndarray:
        """Applies normalization to observation.

        Args:
            obs (np.ndarray): Original observation.

        Returns:
            np.ndarray: Normalized observation.
        """
        # Save original obs in class attribute
        self.unwrapped_observation = observation.copy()

        # NOTE: If you want to record day, month and hour, you should add that
        # variables as keys
        for i, _ in enumerate(self.env.variables["observation"]):
            # normalization (handle DivisionbyZero Error)
            # if (self.ranges[variable][1] - self.ranges[variable][0] == 0):
            #     obs[i] = max(
            #         self.ranges[variable][0], min(
            #             obs[i], self.ranges[variable][1]))

            observation[i] = -1.0 + 2.0 * (observation[i] - self.obs_space.low[i]) / \
                    (self.obs_space.high[i] - self.obs_space.low[i])

            # If value is out
            if np.isnan(observation[i]):
                observation[i] = 0
            elif observation[i] > 1:
                observation[i] = 1
            elif observation[i] < -1:
                observation[i] = -1

        return np.array(observation)
    def get_unwrapped_obs(self) -> Optional[np.ndarray]:
        """Get last environment observation without normalization.

        Returns:
            Optional[np.ndarray]: Last original observation. If it is the first observation, this value is None.
        """
        return self.unwrapped_observatio


class NormalizeObservationCUBES(gym.Wrapper):
    """This wrapper will normalize observations using a given variance and mean

    Note:
        This wrapper takes fixed means and variances
        instead of updating these during the run
    """

    def __init__(self, env: gym.Env, obs_rms: dict, epsilon: float = 1e-8):
        """This wrapper will normalize observations

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        super().__init__(env)

        self.obs_rms = obs_rms
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, done, infos = self.env.step(action)
        obs = self.normalize(np.array([obs]))[0]
        return obs, rews, done, infos


    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs = self.env.reset(**kwargs)
        return self.normalize(np.array([obs]))[0]

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        return (obs - self.obs_rms["mean"]) / np.sqrt(self.obs_rms["var"] + self.epsilon)



