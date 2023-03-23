"""Module for generating EnergyPlus schedules."""

from cubes.construct.base import BaseScheduler
import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import datetime


class OccupancyScheduler(BaseScheduler):
    """Class for defining occupancy schedules."""

    def __init__(
        self,
        year: int,
        months_to_sample: int,
        weekday_init_state_df: pd.DataFrame,
        weekend_init_state_df: pd.DataFrame,
        weekday_transition_matrix_df: pd.DataFrame,
        weekend_transition_matrix_df: pd.DataFrame,
        name: str = "Occupancy Schedule",
    ):
        self._months_to_sample = months_to_sample
        self._weekday_init_matrix = weekday_init_state_df.drop(
            ["number_of_occupants"], axis=1
        ).values.reshape(self.max_occupants, len(self.active_occupant_menu))

        self._weekend_init_matrix = weekend_init_state_df.drop(
            ["number_of_occupants"], axis=1
        ).values.reshape(self.max_occupants, len(self.active_occupant_menu))

        self._weekday_transition_matrix = weekday_transition_matrix_df.drop(
            ["number_of_occupants", "ten_minute_bin", "active_occupant_count"], axis=1
        ).values.reshape(
            self.max_occupants,
            self.steps_per_day,
            len(self.active_occupant_menu),
            len(self.active_occupant_menu),
        )

        self._weekend_transition_matrix = weekend_transition_matrix_df.drop(
            ["number_of_occupants", "ten_minute_bin", "active_occupant_count"], axis=1
        ).values.reshape(
            self.max_occupants,
            self.steps_per_day,
            len(self.active_occupant_menu),
            len(self.active_occupant_menu),
        )

        super().__init__(name=name, year=year)

    def sample_schedule(self, number_of_occupants: int):

        schedule_df = self._sample_schedule_df(number_of_occupants)
        schedule_file = self._build_energyplus_schedule(schedule_df)

        return schedule_file

    def _sample_schedule_df(self, number_of_occupants: int) -> pd.DataFrame:
        """
        Samples an occupancy schedule. Occupancy schedule samples are
        hardcoded to be one week in length at 10 minute intervals.
        Args:
            number_of_occupants (int): number of occupants in building

        Returns:
            schedule_df (DataFrame): occupancy schedule as fraction of occupants active.
        """

        schedule_df = pd.DataFrame(index=self.sample_date_range)
        active_occupants = []

        x = self._sample_init_state(
            number_of_occupants=number_of_occupants, dt=schedule_df.index[0]
        )
        active_occupants.append(x)

        # loop through each step of the year
        for step, dt in enumerate(schedule_df.index[1:]):  # skip first as we have init
            x = self._sample_transition(
                x=x, step=step, dt=dt, number_of_occupants=number_of_occupants
            )
            active_occupants.append(x)

        schedule_df["active_occupants"] = (
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
            init_probabilities = self._weekday_init_matrix[number_of_occupants - 1]

        else:  # weekend
            init_probabilities = self._weekend_init_matrix[number_of_occupants - 1]

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

        day_step = int(step % self.steps_per_day)

        # ascertain whether day is weekday/weekend
        if dt.weekday() < 5:  # weekday
            transition_probabilities = self._weekday_transition_matrix[
                number_of_occupants - 1, day_step, x
            ]

        else:  # weekend
            transition_probabilities = self._weekend_transition_matrix[
                number_of_occupants - 1, day_step, x
            ]

        return np.random.choice(self.active_occupant_menu, p=transition_probabilities)

    def _build_energyplus_schedule(self, sampled_schedule: pd.DataFrame):
        """
        Builds EnergyPlus .sch file from sampled schedule. The sampled schedule
        is less than a year so we copy the sc

        Args:
            sampled_schedule (pd.DataFrame): sampled occupancy schedule

        Returns:
            schedule_string: string of .sch file.
        """

        schedule_string = deepcopy(self.init_schedule_string)

        for i, (dt, row) in enumerate(sampled_schedule.iterrows()):

            if i % self.steps_per_day == 0:
                day_string = self.days_of_week[dt.weekday()]
                schedule_string += f" For: {day_string}, \n"

            # get the time string
            datetime_string = f"{dt.hour:02d}:{dt.minute:02d}:00"

            # add occupancy to time string
            schedule_string += f" Until {datetime_string}, {row[0]:.2f}, \n"

        return schedule_string

    @property
    def sample_date_range(self) -> pd.date_range:
        """
        Date range for schedule dataframe. Hardcoded to one week
        in 10 minute intervals.
        """
        return pd.date_range(
            start=f"{self.year}-01-01",
            end=f"{self.year}-01-07 23:50:00",
            freq="10T",
        )

    @property
    def annual_date_range(self) -> pd.date_range:
        """
        Date range for one year at 10 minute intervals.
        Used for EPlus schedule file.
        """
        return pd.date_range(
            start=f"{self.year}-01-01",
            end=f"{self.year}-12-31 23:50:00",
            freq="10T",
        )

    @property
    def active_occupant_menu(self) -> np.array:
        """
        Defines the number of (possible) active occupants we sample from.
        We are limited to 6 total occupants + 1 (no active occupants) by
        Richardson (2008).
        """
        return np.arange(self.max_occupants + 1)

    @property
    def max_occupants(self) -> int:
        """Hard coded maximum occupants of 6."""
        return int(6)

    @property
    def timestep_length(self) -> int:
        """Time between steps in schedule file in minutes.
        Always 10 minutes for occupancy."""
        return int(10)

    @property
    def steps_per_day(self) -> int:
        return int((24 * 60) / self.timestep_length)
