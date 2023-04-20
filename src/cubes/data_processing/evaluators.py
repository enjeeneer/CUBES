"""Module for data evaluators that merge disparate data sources."""

import pandas as pd

from cubes.data_processing.processors import AbstractProcessor
from pandas import DataFrame
import numpy as np


class BuildingDataEvaluator:
    """Class for merging building data from different sources."""

    def __init__(
        self,
        geometry: AbstractProcessor,
        energy: AbstractProcessor,
        air_infiltration: AbstractProcessor,
    ) -> None:

        self.geometry_df = geometry()
        self.energy_df = energy()
        self.air_infiltration_df = air_infiltration()

    def __call__(self) -> DataFrame:
        """Returns merged building data DataFrame."""

        df = self.geometry_df.copy()
        df = self._merge_energy_to_base(base=df, energy=self.energy_df)
        df = self._merge_air_infiltration_to_base(
            base=df, air_infiltration=self.air_infiltration_df
        )

        return df

    def _merge_energy_to_base(self, base: DataFrame, energy: DataFrame) -> DataFrame:
        """
        Merges energy systems data to base df.
        Args:
            base: Base DataFrame.
            energy: Energy systems DataFrame.
        Returns:
            df: Merged DataFrame.
        """
        df = base.copy()

        df = df.merge(
            energy,
            on=self._energy.merge_features,
            how="left",
        )

        return self._base.merge(
            self._energy,
            on=self._energy.merge_features,
            how="left",
        )

    def _merge_air_infiltration_to_base(
        self, base: DataFrame, air_infiltration: DataFrame
    ) -> DataFrame:
        """
        Merges air infiltration data to base df.
        Args:
            base: Base DataFrame.
            air_infiltration: Air infiltration DataFrame.
        Returns:
            df: Merged DataFrame.
        """
        df = base.copy()

        # get merge integer maxes
        merge_code_maxes = {}
        for code in air_infiltration["REFERENCE BUILDING USE CODE"].unique():
            merge_code_maxes[code] = air_infiltration.loc[
                air_infiltration["REFERENCE BUILDING USE CODE"] == code
            ]["AIR INFILTRATION MERGE INTEGER"].max()

        # get random merge integer for each building
        base_merge_integer = pd.DataFrame(
            index=df.index, columns=["AIR INFILTRATION MERGE INTEGER"]
        )

        for code in df["REFERENCE BUILDING USE CODE"].unique():
            code_index = (df.loc[df["REFERENCE BUILDING USE CODE"] == code]).index
            base_merge_integer.index[code_index] = np.random.randint(
                1, merge_code_maxes[code], len(code_index)
            )

        df = pd.concat([df, base_merge_integer], axis=1)

        # merge in infiltration data
        df = pd.merge(
            df,
            air_infiltration["REFERENCE BUILDING AIR INFILTRATION"],
            on=["REFERENCE BUILDING USE CODE", "AIR INFILTRATION MERGE INTEGER"],
            how="left",
        )

        return df


class OccupantDataEvaluator:
    """Class for processing occupant data."""

    pass


class WeatherDataEvaluator:
    """Class for processing weather data."""

    pass


class Finisher:
    pass
