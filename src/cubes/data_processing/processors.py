"""Module for data_processing row data."""

import pandas as pd
from typing import List, Optional, Union
from pandas import DataFrame
import numpy as np
import abc
import pathlib
from config import ID_COLUMN, RESIDENTIAL_BUILDING_CODES, COMMON_FEATURES


def get_common_features(
    base_df_path: pathlib.Path,
) -> DataFrame:
    """
    Returns common index for all building data. Taken
    from ambience geometry data, which we assume to be
    the most complete.
    """

    df = pd.read_excel(base_df_path)
    df = df.set_index(ID_COLUMN)

    # maintain only residential building codes
    df = df[df["REFERENCE BUILDING USE CODE"].isin(RESIDENTIAL_BUILDING_CODES)]

    # maintain only common channels
    df = df[COMMON_FEATURES]

    return df


class AbstractProcessor(metaclass=abc.ABCMeta):
    """Abstract base class for processing building data."""

    def __init__(
        self, features: List[str], common_features: DataFrame, data_path: pathlib.Path
    ) -> None:
        self._features = features
        self._data_path = data_path
        self._common_features = common_features

    @abc.abstractmethod
    def __call__(self) -> DataFrame:
        """Returns database as DataFrame."""
        raise NotImplementedError

    def check_index_match(self, df: DataFrame) -> bool:
        """Checks if DataFrame index matches common index."""
        return df.index.equals(self.common_features.index)

    @property
    def features(self) -> List[str]:
        """List of features that must be present in DataFrame."""
        return self._features

    @property
    def common_features(self) -> DataFrame:
        """Index to use for concats."""
        return self._common_features

    @property
    def data_path(self) -> pathlib.Path:
        """Path to raw data."""
        return self._data_path

    @property
    def id_column(self) -> str:
        """Column name for unique identifier."""
        return ID_COLUMN

    def _load_raw_data(self, header: Optional[int] = 0) -> DataFrame:
        """Loads raw data."""
        return pd.read_excel(self.data_path, header=header)


class GeometryProcessor(AbstractProcessor):
    """Processes geometric building data."""

    def __init__(
        self, features: List[str], data_path: pathlib.Path, common_features: DataFrame
    ) -> None:

        super().__init__(features, data_path=data_path, common_features=common_features)

    def __call__(self) -> DataFrame:
        """Loads raw data and cleans."""

        df = pd.DataFrame(index=self.common_features.index)
        loaded_df = self._load_raw_data()
        # remove non-resi
        loaded_df = loaded_df[
            loaded_df["REFERENCE BUILDING USE CODE"].isin(RESIDENTIAL_BUILDING_CODES)
        ]

        try:
            loaded_df = loaded_df[self.features]
        except KeyError as e:
            print(f"Raw geometry does not have the required columns: {e}")

        loaded_df = self._calculate_window_to_wall_ratio(loaded_df)
        loaded_df = self._calculate_roof_to_floor_ratio(loaded_df)

        # get mean construction year
        loaded_df["REFERENCE BUILDING CONSTRUCTION YEAR MEAN"] = (
            loaded_df["REFERENCE BUILDING CONSTRUCTION YEAR LOW"]
            + loaded_df["REFERENCE BUILDING CONSTRUCTION YEAR HIGH"]
        ) / 2

        # set index to merge on
        loaded_df = loaded_df.set_index(self.id_column)

        # merge loaded df with common df
        df = pd.concat([df, loaded_df], axis=1)

        return df

    @staticmethod
    def _calculate_window_to_wall_ratio(df: DataFrame) -> DataFrame:
        """Calculates window to wall ratios."""

        df = df.copy()

        df["REFERENCE BUILDING FLOOR ROOF RATIO"] = (
            df["REFERENCE BUILDING ROOF AREA (m2)"]
            / df["REFERENCE BUILDING GROUND FLOOR AREA (m2)"]
        )
        return df

    @staticmethod
    def _calculate_roof_to_floor_ratio(df: DataFrame) -> DataFrame:
        """Calculates roof to floor ratios."""

        df = df.copy()

        df["REFERENCE BUILDING WINDOW WALL RATIO"] = (
            df["REFERENCE BUILDING WINDOW AREA (m2)"]
            / df["REFERENCE BUILDING WALL AREA (m2)"]
        )
        return df


class EnergySystemsProcessor(AbstractProcessor):
    """Processes energy system data."""

    def __init__(
        self,
        features: List[str],
        data_path: pathlib.Path,
        schema_path: pathlib.Path,
        common_features: DataFrame,
    ) -> None:

        self._schema_path = schema_path
        self._ambience_energy_system_name = "HEATING SYSTEM 1 TECHNOLOGY"

        super().__init__(features, data_path=data_path, common_features=common_features)

    def __call__(self) -> DataFrame:
        """Loads raw data and cleans."""

        df = pd.DataFrame(index=self.common_features.index)
        loaded_df = self._load_raw_data()

        try:
            loaded_df = loaded_df[self.features]
        except KeyError as e:
            print(f"Raw energy system does not have the required columns: {e}")

        # set index to merge on
        loaded_df = loaded_df.rename(columns={"Building typology": self.id_column})
        loaded_df = loaded_df.set_index(self.id_column)

        # map heating system to energyplus
        loaded_df = self._map_ambience_to_energyplus(loaded_df)

        # merge loaded df with common df
        df = pd.concat([df, loaded_df], axis=1)

        return df

    def _map_ambience_to_energyplus(self, df: DataFrame) -> DataFrame:
        """Maps ambience types to energyplus."""
        df = df.copy()
        energyplus_system_type_feature = (
            self._ambience_energy_system_name + " ENERGYPLUS"
        )

        mapper = self._load_ambience_to_energyplus_mapping()
        mapper[energyplus_system_type_feature] = (
            mapper["EnergyPlus"] + " " + mapper["Type"]
        )

        df = pd.merge(
            df, mapper, on="HEATING SYSTEM 1 TECHNOLOGY", how="left"
        ).set_index(df.index)

        return df

    def _load_ambience_to_energyplus_mapping(self) -> DataFrame:
        """Loads mapping between ambience and energyplus."""
        df = pd.read_excel(self._schema_path)
        df = df.fillna("")

        return df


class AirInfiltrationProcessor(AbstractProcessor):
    """Processes air infiltration data."""

    def __init__(
        self, features: List[str], data_path: pathlib.Path, common_features: DataFrame
    ) -> None:
        super().__init__(features, data_path=data_path, common_features=common_features)

    def __call__(self) -> Union[DataFrame, dict]:

        df = self.common_features.copy()
        loaded_df = self._load_raw_data(header=9)

        try:
            loaded_df = loaded_df[self.features]
        except KeyError as e:
            print(f"Raw air infiltration data does not have the required columns: {e}")

        # get max merge integer for each building type
        merge_code_maxes = {}
        for code in loaded_df["REFERENCE BUILDING USE CODE"].unique():
            merge_code_maxes[code] = loaded_df.loc[
                loaded_df["REFERENCE BUILDING USE CODE"] == code
            ]["MERGE INTEGER"].max()

        # get random merge integer for each building
        df["MERGE INTEGER"] = pd.NA

        # get random merge integer for each building
        for code in df["REFERENCE BUILDING USE CODE"].unique():
            code_index = (df.loc[df["REFERENCE BUILDING USE CODE"] == code]).index
            df["MERGE INTEGER"].loc[code_index] = np.random.randint(
                1, merge_code_maxes[code], size=(len(code_index))
            )

        # merge air infiltration data with common features
        df = pd.merge(
            df,
            loaded_df,
            on=["REFERENCE BUILDING USE CODE", "MERGE INTEGER"],
            how="left",
        ).set_index(df.index)

        return df
