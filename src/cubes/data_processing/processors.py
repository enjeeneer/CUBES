"""Module for data_processing row data."""

import pandas as pd
from typing import List
from pandas import DataFrame
import abc
import pathlib


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

    def _load_raw_data(self) -> DataFrame:
        """Loads raw data."""
        return pd.read_xlsx(self.data_path)


class GeometryProcessor(AbstractProcessor):
    """Processes geometric building data."""

    def __init__(
        self,
        features,
        merge_features,
        data_path,
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

        return df
