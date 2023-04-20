"""Module for data_processing row data."""

import pandas as pd
from typing import List, Optional, Union
from pandas import DataFrame
import abc
import pathlib
from config import ID_COLUMN, RESIDENTIAL_BUILDING_CODES


def get_common_index(
    base_df_path: pathlib.Path,
) -> pd.Index:
    """
    Returns common index for all building data. Taken
    from ambience geometry data, which we assume to be
    the most complete.
    """

    df = pd.read_excel(base_df_path)
    df = df.set_index(ID_COLUMN)

    # maintain only residential building codes
    df = df[df["REFERENCE BUILDING CODE"].isin(RESIDENTIAL_BUILDING_CODES)]

    return df.index


class AbstractProcessor(metaclass=abc.ABCMeta):
    """Abstract base class for processing building data."""

    def __init__(
        self, features: List[str], common_index: pd.Index, data_path: pathlib.Path
    ) -> None:
        self._features = features
        self._data_path = data_path
        self._common_index = common_index

    @abc.abstractmethod
    def __call__(self) -> DataFrame:
        """Returns database as DataFrame."""
        raise NotImplementedError

    @property
    def features(self) -> List[str]:
        """List of features that must be present in DataFrame."""
        return self._features

    @property
    def common_index(self) -> pd.Index:
        """Index to use for concats."""
        return self._common_index

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
        self, features: List[str], data_path: pathlib.Path, common_index: pd.Index
    ) -> DataFrame:

        super().__init__(features, data_path=data_path, common_index=common_index)

        self.__call__()

    def __call__(self) -> DataFrame:
        """Loads raw data and cleans."""

        df = self._load_raw_data()

        try:
            df = df[self.features]
        except KeyError as e:
            print(f"Raw geometry does not have the required columns: {e}")

        df = self._calculate_window_to_wall_ratio(df)
        df = self._calculate_roof_to_floor_ratio(df)

        # get mean construction year
        df["REFERENCE BUILDING CONSTRUCTION YEAR MEAN"] = (
            df["REFERENCE BUILDING CONSTRUCTION YEAR LOW"]
            + df["REFERENCE BUILDING CONSTRUCTION YEAR HIGH"]
        ) / 2

        # set index to merge on
        df.set_index(self.id_column)

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
        common_index: pd.Index,
    ) -> None:

        self._schema_path = schema_path
        self._ambience_energy_system_name = "HEATING SYSTEM 1 TECHNOLOGY"

        super().__init__(features, data_path=data_path, common_index=common_index)

        self.__call__()

    def __call__(self) -> DataFrame:
        """Loads raw data and cleans."""

        df = self._load_raw_data()

        try:
            df = df[self.features]
        except KeyError as e:
            print(f"Raw energy system does not have the required columns: {e}")

        # set index to merge on
        df = df.rename(columns={"Building typology": self.id_column})
        df.set_index(self.id_column)

        # map heating system to energyplus
        df = self._map_ambience_to_energyplus(df)

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

        df = pd.merge(df, mapper, on="HEATING SYSTEM 1 TECHNOLOGY", how="left")

        return df

    def _load_ambience_to_energyplus_mapping(self) -> DataFrame:
        """Loads mapping between ambience and energyplus."""
        df = pd.read_excel(self._schema_path)
        df = df.fillna("")

        return df


class AirInfiltrationProcessor(AbstractProcessor):
    """Processes air infiltration data."""

    def __init__(
        self, features: List[str], data_path: pathlib.Path, common_index: pd.Index
    ) -> DataFrame:
        super().__init__(features, data_path=data_path, common_index=common_index)

        self.__call__()

    def __call__(self) -> Union[DataFrame, dict]:

        df = self._load_raw_data(header=9)

        try:
            df = df[self.features]
        except KeyError as e:
            print(f"Raw air infiltration data does not have the required columns: {e}")

        # set index to merge on
        df.set_index("REFERENCE BUILDING USE CODE")

        return df
