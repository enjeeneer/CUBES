"""Module for generating EnergyPlus schedules."""

from cubes.construct.base import BaseScheduler
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, TypedDict


class DayTime(TypedDict):
    hour: int
    minute: int


class TimeRange(TypedDict):
    start: DayTime
    stop: DayTime


class OccupancyScheduler(BaseScheduler):
    """Class for defining occupancy schedules."""

    def __init__(
        self,
        year: int,
        sample_length: str,
        weekday_init_state_df: pd.DataFrame,
        weekend_init_state_df: pd.DataFrame,
        weekday_transition_matrix_df: pd.DataFrame,
        weekend_transition_matrix_df: pd.DataFrame,
        name: str = "Occupancy Schedule",
    ):
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

        self._sample_length = sample_length

        super().__init__(name=name, year=year)

    def sample(
        self,
        number_of_occupants: int,
        sleep_time_range: TimeRange = None,
    ) -> str:

        schedule_df = self._sample_schedule_df(number_of_occupants)
        schedule_file = self._build_energyplus_schedule(schedule_df)
        sleep_schedule_file = ""
        if sleep_time_range:
            sleeping_schedule_df = self._get_sleeping_schedule_df(
                schedule_df, sleep_time_range
            )
            sleep_schedule_file = self._build_energyplus_schedule(sleeping_schedule_df)

        return schedule_file, sleep_schedule_file

    def _get_sleeping_schedule_df(
        self, active_schedule_df: pd.DataFrame, sleep_time_range: TimeRange
    ):
        print(sleep_time_range)
        sleep_schedule_df = pd.DataFrame(index=active_schedule_df.index)
        sleep_schedule_df["sleeping_occupants"] = 0

        sleep_schedule_df.loc[
            (
                (sleep_schedule_df.index.hour >= sleep_time_range["start"]["hour"])
                & (sleep_schedule_df.index.minute > sleep_time_range["start"]["minute"])
            )
            | (sleep_schedule_df.index.hour > sleep_time_range["start"]["hour"])
            | (sleep_schedule_df.index.hour < sleep_time_range["stop"]["hour"])
            | (
                (sleep_schedule_df.index.hour <= sleep_time_range["stop"]["hour"])
                & (sleep_schedule_df.index.minute <= sleep_time_range["stop"]["minute"])
            ),
            "sleeping_occupants",
        ] = (
            1 - active_schedule_df["active_occupants"]
        )

        return sleep_schedule_df

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

        # initialise markov chain
        x = self._sample_init_state(
            number_of_occupants=number_of_occupants, dt=schedule_df.index[0]
        )
        active_occupants.append(x)

        # loop through each step of the week
        for step, dt in enumerate(schedule_df.index[1:]):  # skip first as we have init
            x = self._sample_transition(
                x=x, step=step, dt=dt, number_of_occupants=number_of_occupants
            )
            active_occupants.append(x)

        # normalise to be in [0, 1]
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

    def _build_energyplus_schedule(self, sampled_schedule: pd.DataFrame) -> str:
        """
        Builds EnergyPlus .sch file from sampled schedule.

        Args:
            sampled_schedule (pd.DataFrame): sampled occupancy schedule

        Returns:
            schedule_string: string of .sch file.
        """

        if self.sample_length == "week":
            schedule_string = self._build_schedule_from_sub_sample(sampled_schedule)

        elif self.sample_length == "month":
            schedule_string = self._build_schedule_from_sub_sample(sampled_schedule)

        else:
            schedule_string = self._build_schedule_from_year_sample(sampled_schedule)

        return schedule_string

    def _build_schedule_from_sub_sample(self, sampled_schedule: pd.DataFrame) -> str:
        """
        Builds an annual EnergyPlus occupancy given a one-week sample from model.
        Schedules are defined for each day of the week once, then copied for the
        rest of the year.
        Args:
            sampled_schedule (pd.DataFrame): one-week occupancy sample.

        Returns:
            str: EnergyPlus .sch file.
        """

        schedule_string = ""
        weekday_numbers = sampled_schedule[sampled_schedule.index.weekday < 5]
        weekend_numbers = sampled_schedule[sampled_schedule.index.weekday >= 5]

        for i, dt in enumerate(self.annual_date_range):

            j = i % self.steps_per_day  # step in day counter
            if j == 0:  # sample new day
                weekday = dt.weekday() < 5
                if weekday:
                    sampled_day = np.random.choice(list(set(weekday_numbers.index.day)))
                else:
                    sampled_day = np.random.choice(list(set(weekend_numbers.index.day)))

                day_sample = sampled_schedule.loc[
                    sampled_schedule.index.day == sampled_day
                ].values.squeeze(-1)

            schedule_string += f"{day_sample[j]:.2f}, \n"
        return schedule_string

    def _build_schedule_from_year_sample(self, sampled_schedule: pd.DataFrame) -> str:
        """
        Builds an annual EnergyPlus occupancy given a one-year sample from model.
        Schedules are defined for every 10 step of the year.
        Args:
            sampled_schedule (pd.DataFrame): one-week occupancy sample.

        Returns:
            str: EnergyPlus .sch file.
        """

        schedule_string = ""

        for _, row in sampled_schedule.iterrows():

            schedule_string += f"{row[0]:.2f}, \n"

        return schedule_string

    @property
    def sample_date_range(self) -> pd.date_range:
        """
        Date range for schedule dataframe conditioned on the sample length
        passed by user.
        """
        if self.sample_length == "week":
            date_range = pd.date_range(
                start=f"{self.year}-01-01",
                end=f"{self.year}-01-07 23:50:00",
                freq="10T",
            )

        elif self.sample_length == "month":
            date_range = pd.date_range(
                start=f"{self.year}-01-01",
                end=f"{self.year}-01-28 23:50:00",
                freq="10T",
            )

        else:  # year
            date_range = pd.date_range(
                start=f"{self.year}-01-01",
                end=f"{self.year}-12-31 23:50:00",
                freq="10T",
            )

        return date_range

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
        """Number of timesteps in a day."""
        return int((24 * 60) / self.timestep_length)

    @property
    def sample_lengths(self) -> List[str]:
        return ["week", "month", "year"]

    @property
    def sample_length(self) -> str:
        if self._sample_length in self.sample_lengths:
            return self._sample_length
        else:
            raise ValueError(
                f"""Sample length {self._sample_length} not in
                list of accepted sample lenghts: {self.sample_lengths}"""
            )
