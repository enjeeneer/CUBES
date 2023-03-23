"""Module for bases classes used throughout package."""

import abc
import pandas as pd
from typing import List


class BaseScheduler(metaclass=abc.ABCMeta):
    """Base class for generating EnergyPlus schedules."""

    def __init__(self, name: str, year: int):

        self._name = name
        self._year = year
        super().__init__()

    @abc.abstractmethod
    def sample_schedule(self):
        """Sample schedule for one building."""
        pass

    @abc.abstractmethod
    def _build_energyplus_schedule(self, sampled_schedule: pd.DataFrame):
        """Takes numpy array holding schedule values and converts to .sch"""
        pass

    @abc.abstractmethod
    def _sample_schedule_df(self):
        """Samples schedule as DataFrame."""
        pass

    @property
    def timestep_length(self) -> int:
        """Length of one schedule timestep in minutes"""
        pass

    @property
    def steps_per_day(self) -> int:
        """
        Number of timesteps in a day given timestep length.
        """
        pass

    @property
    def name(self) -> str:
        """Name of schedule"""
        return self._name

    @property
    def year(self) -> int:
        """Year fo schedule."""
        return self._year

    @property
    def init_schedule_string(self) -> str:
        """Each schedule is initialised with the same string format."""

        schedule_string = f"""
        Schedule:Compact,
        {self.name},        !- Name
        Fraction,           !- Schedule Type Limits Name
        Through: 12/31,     !- Field 1
        """

        return schedule_string

    @property
    def days_of_week(self) -> List[str]:
        """Day strings used for .sch file"""

        return [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
