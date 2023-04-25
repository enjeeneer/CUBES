"""Module for data_processing row data."""

import pandas as pd
from typing import List, Optional, Union, Dict
from pandas import DataFrame
import numpy as np
import abc
import pathlib
from config import ID_COLUMN, RESIDENTIAL_BUILDING_CODES, COMMON_FEATURES, COUNTRIES
from cubes.construct.material import (
    NoMassMaterial,
    Material,
    WindowMaterialSimpleGlazing,
    WindowMaterialGlazing,
)


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

    # maintain reliable country data
    df = df[df["REFERENCE BUILDING COUNTRY CODE"].isin(COUNTRIES)]

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

        # remove non-reliable country data
        loaded_df = loaded_df[
            loaded_df["REFERENCE BUILDING COUNTRY CODE"].isin(COUNTRIES)
        ]

        try:
            loaded_df = loaded_df[self.features]
        except KeyError as e:
            print(f"Raw geometry does not have the required columns: {e}")

        loaded_df = self._calculate_window_to_wall_ratio(loaded_df)
        loaded_df = self._calculate_roof_to_floor_ratio(loaded_df)

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
        loaded_df = loaded_df.set_index(self.id_column)

        # remove non-reliable country data
        loaded_df = loaded_df[
            loaded_df["REFERENCE BUILDING COUNTRY CODE"].isin(COUNTRIES)
        ]

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


class MaterialsProcessor(AbstractProcessor):
    """Processes materials data."""

    def __init__(
        self, features: List[str], data_path: pathlib.Path, common_features: DataFrame
    ) -> None:
        super().__init__(features, data_path=data_path, common_features=common_features)

    def __call__(self) -> Dict:
        """Loads raw data and cleans."""
        loaded_df = self._load_raw_data()

        try:
            loaded_df = loaded_df[self.features]
        except KeyError as e:
            print(f"Raw materials data does not have the required columns: {e}")

        # convert dataframe to dataclasses held as dict
        materials = self._convert_to_dataclasses(loaded_df)

        return materials

    def _convert_to_dataclasses(self, df: DataFrame) -> Dict:
        """Converts dataframe to dataclasses."""

        df = df.copy()

        materials = {}

        no_mass = df[df["NoMass"] is True].copy()
        mass = df[df["NoMass"] is False].copy()

        for _, row in no_mass.iterrows():
            materials[row["Material"]] = NoMassMaterial(
                name=row["Material"],
                roughness=row["Roughness"],
                resistance=row["Thermal_Resistance"],
                thermal_absorptance=row["Thermal_Absorptance"],
                solar_absorptance=row["Solar_Absorptance"],
                visual_absorptance=row["Visual_Absorptance"],
            )

        for _, row in mass.iterrows():
            materials[row["Material"]] = Material(
                name=row["Material"],
                rho=row["Density"],
                cp=row["Specific_Heat_Capacity"],
                k=row["Thermal_Conductivity"],
                roughness=row["Roughness"],
                thermal_absorptance=row["Thermal_Absorptance"],
                solar_absorptance=row["Solar_Absorptance"],
                visual_absorptance=row["Visual_Absorptance"],
            )

        return materials


class WindowsProcessor(AbstractProcessor):
    """Processes windows data."""

    def __init__(
        self, features: List[str], data_path: pathlib.Path, common_features: DataFrame
    ) -> None:
        super().__init__(features, data_path=data_path, common_features=common_features)

    def __call__(self) -> Dict:
        """Loads raw data and cleans."""
        loaded_df = self._load_raw_data()

        try:
            loaded_df = loaded_df[self.features]
        except KeyError as e:
            print(f"Raw windows data does not have the required columns: {e}")

        # convert dataframe to dataclasses held as dict
        windows = self._convert_to_dataclasses(loaded_df)

        return windows

    def _convert_to_dataclasses(self, df: DataFrame) -> Dict:
        """Converts dataframe to dataclasses."""

        df = df.copy()

        windows = {}

        simple_windows = df[df["Simple Glazing"] is True].copy()
        complex_windows = df[df["Simple Glazing"] is False].copy()

        for _, row in complex_windows.iterrows():
            windows[row["Name"]] = WindowMaterialGlazing(
                name=row["Name"],
                optical_data_type=row["Optical_Data_Type"],
                data_set_name=row["Window Glass Spectral Data Set Name"],
                thickness=row["Thickness"],
                solar_transmittance=row["Solar_Transmittance at Normal Incidence"],
                front_side_solar_reflectance=row[
                    "Front Side Solar Reflectance at Normal Incidence"
                ],
                back_side_solar_reflectance=row[
                    "Back Side Solar Reflectance at Normal Incidence"
                ],
                visible_transmittance=row["Visible_Transmittance at Normal Incidence"],
                front_side_visible_reflectance=row[
                    "Front Side Visible Reflectance at Normal"
                ],
                back_side_visible_reflectance=row[
                    "Back Side Visible Reflectance at Normal"
                ],
                infrared_transmittance=row[
                    "Infrared_Transmittance at Normal Incidence"
                ],
                front_side_infrared_emissivity=row[
                    "Front Side Infrared Hemispherical Emissivity"
                ],
                back_side_infrared_emissivity=row[
                    "Back Side Infrared Hemispherical Emissivity"
                ],
                conductivity=row["Conductivity"],
            )

        for _, row in simple_windows.iterrows():
            windows[row["Name"]] = WindowMaterialSimpleGlazing(
                name=row["Name"],
                u_factor=row["U Factor"],
                solar_heat_gain_coefficient=row["Solar Heat Gain Coefficient"],
                visible_transmittance=row["Visible Transmittance at Normal Incidence"],
            )

        return windows
