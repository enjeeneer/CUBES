"""Module for bases classes used throughout package."""

import abc
import pandas as pd


class BaseScheduler(metaclass=abc.ABCMeta):
    """Base class for generating EnergyPlus schedules."""

    def __init__(self, timestep_length: int, name: str, year: int):

        self._timestep_length = timestep_length
        self._name = name
        self._year = year
        super().__init__()

    @abc.abstractmethod
    def sample_schedule(self):
        """Sample schedule for one building."""
        pass

    @abc.abstractmethod
    def _build_energyplus_schedule(self, schedule_df: pd.DataFrame):
        """Takes numpy array holding schedule values and converts to .sch"""
        pass

    @abc.abstractmethod
    def _sample_schedule_df(self):
        """Samples schedule as DataFrame."""
        pass

    @property
    def timestep_length(self):
        """Length of one schedule timestep in minutes"""
        return self._timestep_length

    @property
    def steps_per_day(self) -> int:
        """
        Number of timesteps in a day given timestep length.
        """
        return (24 * 60) / self._timestep_length

    @property
    def name(self):
        """Name of schedule"""
        return self._name

    @property
    def year(self):
        """Year fo schedule."""
        return self._year

    @property
    def init_schedule_string(self):
        """Each schedule is initialised with the same string format."""

        schedule_string = f"""
            Schedule:Compact,
            {self.name},        !- Name
            Fraction,           !- Schedule Type Limits Name
            Through: 12/31,     !- Field 1
            For: AllDays,      !- Field 2
        """

        return schedule_string
