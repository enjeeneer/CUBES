"""custom wrapper to handle cubes reward function"""
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
