"""Module for data_processing row data."""

import pandas as pd
from typing import List
from pandas import DataFrame
import abc
import pathlib
from config import (
    ID_COLUMN,
    GEOMETRY_MERGE_FEATURES,
    GEOMETRY_FEATURES,
    GEOMETRY_PATH,
    SYSTEMS_FEATURES,
    SYSTEMS_PATH,
    SYSTEMS_MAP_PATH,
    SYSTEMS_MERGE_FEATURES,
    HEATING_SYSTEM_TYPE_FEATURE,
)


class AbstractProcessor(metaclass=abc.ABCMeta):
    """Abstract base class for processing building data."""

    def __init__(
        self,
        features: List[str],
        merge_features: List[str],
        data_path: pathlib.Path,
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

    def _load_raw_data(self) -> DataFrame:
        """Loads raw data."""
        return pd.read_xlsx(self.data_path)


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
        self,
        features=GEOMETRY_FEATURES,
        merge_features=GEOMETRY_MERGE_FEATURES,
        data_path=GEOMETRY_PATH,
    ) -> None:

        super().__init__(features, merge_features, data_path)

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


class EnergySystemProcessor(AbstractProcessor):
    """Processes energy system data."""

    def __init__(
        self,
        features=SYSTEMS_FEATURES,
        merge_features=SYSTEMS_MERGE_FEATURES,
        data_path=SYSTEMS_PATH,
        mapping_path=SYSTEMS_MAP_PATH,
    ) -> None:

        self._mapping_path = mapping_path
        self._heating_system_type_feature = HEATING_SYSTEM_TYPE_FEATURE

        super().__init__(features, merge_features, data_path)

        self.__call__()

    def __call__(self) -> DataFrame:
        """Loads raw data and cleans."""

        df = self._load_raw_data()

        try:
            df = df[self.features]
        except KeyError as e:
            print(f"Raw energy system does not have the required columns: {e}")

        # set index to merge on
        df.rename(columns={"Building typology": self.id_column})
        df.set_index(self.id_column)

        # map heating system to energyplus
        df = self._map_ambience_to_energyplus(df)

        return df

    def _map_ambience_to_energyplus(self, df: DataFrame) -> DataFrame:
        """Maps ambience types to energyplus."""
        df = df.copy()
        mapper = self._load_ambience_to_energyplus_mapping()
        df[HEATING_SYSTEM_TYPE_FEATURE + " ENERGYPLUS"] = (
            mapper["EnergyPlus"][mapper["Ambience"] == df[HEATING_SYSTEM_TYPE_FEATURE]]
            + " "
            + mapper["Type"][mapper["Ambience"] == df[HEATING_SYSTEM_TYPE_FEATURE]]
        )
        return df

    def _load_ambience_to_energyplus_mapping(self) -> DataFrame:
        """Loads mapping between ambience and energyplus."""
        return pd.read_excel(self._mapping_path)
