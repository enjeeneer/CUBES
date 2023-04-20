"""Module for data_processing row data."""

import pandas as pd
from typing import List, Optional
from pandas import DataFrame
import abc
import pathlib
from config import ID_COLUMN


class AbstractProcessor(metaclass=abc.ABCMeta):
    """Abstract base class for processing building data."""

    def __init__(
        self,
        features: List[str],
        data_path: pathlib.Path,
        merge_features: Optional[List[str]] = None,
    ) -> None:
        self._features = features
        self._merge_features = merge_features
        self._data_path = data_path

    @abc.abstractmethod
    def __call__(self) -> DataFrame:
        """Returns database as DataFrame."""
        raise NotImplementedError

    @property
    def features(self) -> List[str]:
        """List of features that must be present in DataFrame."""
        return self._features

    @property
    def merge_features(self) -> List[str]:
        """List of features to merge database on."""
        return self._merge_features

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


def _calculate_window_to_wall_ratios(df: DataFrame) -> DataFrame:
    """Calculates window to wall ratios."""

    df = df.copy()

    df["REFERENCE BUILDING WINDOW WALL RATIO"] = (
        df["REFERENCE BUILDING WINDOW AREA (m2)"]
        / df["REFERENCE BUILDING WALL AREA (m2)"]
    )
    return df


class GeometryProcessor(AbstractProcessor):
    """Processes geometric building data."""

    def __init__(
        self, features: List[str], merge_features: List[str], data_path: pathlib.Path
    ) -> DataFrame:

        super().__init__(features, merge_features=merge_features, data_path=data_path)

        self.__call__()

    def __call__(self) -> DataFrame:
        """Loads raw data and cleans."""

        df = self._load_raw_data()

        try:
            df = df[self.features]
        except KeyError as e:
            print(f"Raw geometry does not have the required columns: {e}")

        df = _calculate_window_to_wall_ratios(df)
        df = self._calculate_roof_to_floor_ration(df)

        # get mean construction year
        df["REFERENCE BUILDING CONSTRUCTION YEAR MEAN"] = (
            df["REFERENCE BUILDING CONSTRUCTION YEAR LOW"]
            + df["REFERENCE BUILDING CONSTRUCTION YEAR HIGH"]
        ) / 2

        # set index to merge on
        df.set_index(self.id_column)

        return df

    def _calculate_roof_to_floor_ration(self, df: DataFrame) -> DataFrame:
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
        merge_features: List[str],
        data_path: pathlib.Path,
        mapper_path: pathlib.Path,
    ) -> DataFrame:

        self._mapping_path = mapper_path
        self._ambience_system_type = "HEATING SYSTEM 1 TECHNOLOGY"

        super().__init__(features, merge_features=merge_features, data_path=data_path)

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
        energyplus_system_type_feature = self._ambience_system_type + " ENERGYPLUS"

        mapper = self._load_ambience_to_energyplus_mapping()
        for system in df[self._ambience_system_type].unique():
            if system not in mapper["Ambience"].unique():
                print(f"Missing mapping for {system}")

            else:
                df[energyplus_system_type_feature] = (
                    mapper["EnergyPlus"][mapper["Ambience"] == system]
                    + " "
                    + mapper["Type"][mapper["Ambience"] == system]
                )

        return df

    def _load_ambience_to_energyplus_mapping(self) -> DataFrame:
        """Loads mapping between ambience and energyplus."""
        return pd.read_excel(self._mapping_path)


class AirInfiltrationProcessor(AbstractProcessor):
    """Processes air infiltration data."""

    def __init__(
        self,
        features: List[str],
        data_path: pathlib.Path,
        merge_features: List[str],
    ) -> DataFrame:
        super().__init__(features, data_path=data_path, merge_features=merge_features)

        self.__call__()

    def __call__(self) -> DataFrame:

        df = self._load_raw_data(header=9)

        try:
            df = df[self.features]
        except KeyError as e:
            print(f"Raw air infiltration data does not have the required columns: {e}")

        df["REFERENCE BUILDING CONSTRUCTION YEAR MEAN"] = (
            df["REFERENCE BUILDING CONSTRUCTION YEAR LOW"]
            + df["REFERENCE BUILDING CONSTRUCTION YEAR HIGH"]
        ) / 2

        # set index to merge on
        df.set_index(self.merge_features)

        return df
