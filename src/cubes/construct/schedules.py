"""Module for generating EnergyPlus schedules."""

from cubes.construct.base import BaseScheduler
import pandas as pd
import numpy as np
from datetime import datetime


class OccupancyScheduler(BaseScheduler):
    """Class for defining occupancy schedules."""

    def __init__(
        self,
        timestep_length: int,
        year: int,
        weekday_init_state_df: pd.DataFrame,
        weekend_init_state_df: pd.DataFrame,
        weekday_transition_matrix_df: pd.DataFrame,
        weekend_transition_matrix_df: pd.DataFrame,
        name: str = "Occupancy Schedule",
    ):
        self._weekday_init_state_df = weekday_init_state_df
        self._weekend_init_state_df = weekend_init_state_df
        self._weekday_transition_matrix_df = weekday_transition_matrix_df
        self._weekend_transition_matrix_df = weekend_transition_matrix_df

        super().__init__(timestep_length=timestep_length, name=name, year=year)

    def sample_schedule(self, number_of_occupants: int):

        schedule_df = self._sample_schedule_df(number_of_occupants)
        schedule_file = self._build_energyplus_schedule(schedule_df)

        return schedule_file

    def _sample_schedule_df(self, number_of_occupants: int) -> pd.DataFrame:
        """
        Samples an occupancy schedule for one year at timestep_length intervals.
        Args:
            number_of_occupants (int): number of occupants in building

        Returns:
            schedule_df (DataFrame): occupancy schedule as fraction of occupants active.
        """

        schedule_df = pd.DataFrame(index=self.date_range)
        active_occupants = []

        x = self._sample_init_state(
            number_of_occupants=number_of_occupants, dt=schedule_df.index[0]
        )
        active_occupants += x

        # loop through each step of the year
        for step, dt in enumerate(schedule_df.index):
            x = self._sample_transition(
                x=x, step=step, dt=dt, number_of_occupants=number_of_occupants
            )
            active_occupants += x

        schedule_df["active_occupant_fraction"] = (
            np.array(active_occupants) / number_of_occupants
        )

        return schedule_df

    def _sample_init_state(self, number_of_occupants: int, dt: datetime) -> int:
        """
        Samples initial state given the day of the week and the number of
        occupants
        Args:
            number_of_occupants (int): no. of occupants in building
            datetime (datetime): current datetime

        Returns:
            x (int): number of initially active occupants
        """

        # ascertain whether day is weekday/weekend
        if dt.weekday() < 5:  # weekday
            init_probabilities = (
                self._weekday_init_state_df[
                    self._weekday_init_state_df["number_of_occupants"]
                    == number_of_occupants
                ]
                .drop("number_of_occupants", axis=1)
                .values.squeeze()
            )

        else:  # weekend
            init_probabilities = (
                self._weekend_init_state_df[
                    self._weekend_init_state_df["number_of_occupants"]
                    == number_of_occupants
                ]
                .drop("number_of_occupants", axis=1)
                .values.squeeze()
            )

        return np.random.choice(self.active_occupant_menu, p=init_probabilities)

    def _sample_transition(
        self, x: int, step: int, dt: datetime, number_of_occupants: int
    ) -> int:
        """
        Samples next state in Markov chain given current
        state and transition matrix.
        Args:
            x (int): current state (number of active occupants)
            step (int): current step index through year.
            dt (datetime): current datetime
            number_of_occupants (int): no. of occupants in building

        Returns:
            y (int): next state (number of active occupants)
        """

        day_step = int(step % self.steps_per_day) + 1

        # ascertain whether day is weekday/weekend
        if dt.weekday() < 5:  # weekday
            transition_probabilities = (
                self._weekday_transition_matrix_df[
                    (
                        self._weekday_transition_matrix_df["number_of_occupants"]
                        == number_of_occupants
                    )
                    & (self._weekday_transition_matrix_df["ten_minute_bin"] == day_step)
                    & (self._weekday_transition_matrix_df["active_occupant_count"] == x)
                ]
                .drop(
                    ["number_of_occupants", "ten_minute_bin", "active_occupant_count"],
                    axis=1,
                )
                .values.squeeze()
            )

        else:  # weekend
            transition_probabilities = (
                self._weekend_transition_matrix_df[
                    (
                        self._weekend_transition_matrix_df["number_of_occupants"]
                        == number_of_occupants
                    )
                    & (self._weekend_transition_matrix_df["ten_minute_bin"] == day_step)
                    & (self._weekend_transition_matrix_df["active_occupant_count"] == x)
                ]
                .drop(
                    ["number_of_occupants", "ten_minute_bin", "active_occupant_count"],
                    axis=1,
                )
                .values.squeeze()
            )

        return np.random.choice(self.active_occupant_menu, p=transition_probabilities)

    def _build_energyplus_schedule(self, schedule_df: pd.DataFrame):
        """
        Builds EnergyPlus .sch file from sampled schedule DataFrame.

        Args:
            schedule_df (pd.DataFrame): sampled occupancy schedule

        Returns:
            schedule_string: string of .sch file.
        """

        schedule_string = self.init_schedule_string.copy()

        for dt in schedule_df.index:

            # get the time string
            datetime_string = (
                f"{dt.month:02d}/{dt.day:02d}{dt.hour:02d}:{dt.minute:02d}:00"
            )

            # get the occupancy value
            occupancy = schedule_df.loc[dt]

            # add occupancy to time string
            schedule_string += f" Until {datetime_string}, {occupancy:.2f}, \n"

        schedule_string += " Until 01/01 24:00:00, occupancy: 0.00; \n "

        return schedule_string

    @property
    def date_range(self):
        """Date range for schedule dataframe."""
        return pd.date_range(
            start=f"{self.year}-01-01", end=f"{self.year}-12-31 23:50:00", freq="10T"
        )

    @property
    def active_occupant_menu(self) -> np.array:
        """
        Defines the number of (possible) active occupants we sample from.
        We are limited to 6 total occupants + 1 (no active occupants) by
        Richardson (2008).
        """
        return np.arange(7)
