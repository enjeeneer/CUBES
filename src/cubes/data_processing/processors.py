"""Module for data_processing row data."""

import pandas as pd
from typing import List, Optional, Union, Dict
from pandas import DataFrame
import numpy as np
import abc
import pathlib
from cubes.data_processing.config import (
    LOCATION_PATH,
    GEOMETRY_PATH,
    RESIDENTIAL_BUILDING_CODES,
    COUNTRIES,
)
from cubes.construct.material import (
    NoMassMaterial,
    Material,
    WindowMaterialSimpleGlazing,
    WindowMaterialGlazing,
)


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
        raise NotImplementedError

    def _load_raw_data(self, header: Optional[int] = 0) -> DataFrame:
        """Loads raw data."""
        return pd.read_excel(self.data_path, header=header)


class BaseProcessor:
    """Gets the base DataFrame that all data is merged onto."""

    def __init__(
        self,
        location_df_path: pathlib.Path = LOCATION_PATH,
        geometry_df_path: pathlib.Path = GEOMETRY_PATH,
    ):
        self._location_df_path = location_df_path
        self._geometry_df_path = geometry_df_path

    def __call__(self) -> DataFrame:
        """
        Returns base DataFrame for all building data. The base df creates
        a common index out of the EU NUTS 3 location data (a quantification
        of the number of dwellings in each NUTS 3 region) and the ambience
        geometry data (a quantification of the building archetypes in each
        country). We assume each NUTS 3 region contains every building
        archetype described in the ambience dataset, with their number
        proportional to the national fraction of each archetype in each
        country.

        Returns:
            df: base DataFrame with index representing every archetype
                for every NUTS 3 region.
        """

        location_df = pd.read_excel(self._location_df_path)
        geometry_df = pd.read_excel(self._geometry_df_path)

        # get proportions of archetypes for each country
        geometry_df = self._calculate_archetype_proportions(geometry_df)
        # get common df by combining NUTS 3 regions and building archetypes
        df = self._get_common_df(location_df, geometry_df)

        # merge location and geometry dfs into common df
        merged_location = pd.merge(df, location_df, on="NUTS 3 REGION")
        merged_location = merged_location.set_index(df.index)

        geometry_df = geometry_df.set_index("REFERENCE BUILDING CODE")
        fully_merged = pd.merge(
            merged_location,
            geometry_df,
            left_on="REFERENCE BUILDING COUNTRY CODE",
            right_index=True,
        )

        # get number of dwellings for each region/archetype pair
        fully_merged["NUMBER OF DWELLINGS"] = (
            fully_merged["COUNTRY ARCHETYPE PROPORTION"]
            * fully_merged["Occupied conventional dwellings"]
        )

        return fully_merged

    @staticmethod
    def _get_common_df(location_df: DataFrame, geometry_df: DataFrame) -> DataFrame:
        """
        Gets common index for base df by combining NUTS 3 regions and
        building archetypes.
        Args:
            location_df: DataFrame with location data.
            geometry_df: DataFrame with building archetype data.
        Returns:
            common_index: Index representing every archetype for every NUTS 3 region.
        """

        df = pd.DataFrame()

        # loop through each NUTS 3 region and get the associated country archetypes
        for _, row in location_df.iterrows():
            country = row["COUNTRY CODE"]
            archetypes = geometry_df[
                geometry_df["REFERENCE BUILDING COUNTRY CODE"] == country
            ]["REFERENCE BUILDING CODE"].values

            multi_index = pd.MultiIndex.from_product(
                [[row["NUTS 3 REGION"]], archetypes],
                names=["NUTS 3 REGION", "REFERENCE BUILDING CODE"],
            )

            region_df = pd.DataFrame(index=multi_index)

            df = pd.concat([df, region_df], axis=0)

        return df

    def _calculate_archetype_proportions(self, geometry_df: DataFrame) -> DataFrame:
        """
        Calculates the proportion of each building archetype in each country.
        Args:
            geometry_df: DataFrame with building archetype data.
        Returns:
            df: DataFrame with building archetype proportions.
        """
        geometry_df = geometry_df.copy()

        country_proportion = []

        for country in COUNTRIES:
            country = geometry_df[
                geometry_df["REFERENCE BUILDING COUNTRY CODE"] == country
            ]
            total_dwellings = country[
                "NUMBER OF REFERENCE BUILDINGS IN THE BUILDING STOCK SEGMENT"
            ].sum()

            for index in country.index:
                archetype = country.loc[index]
                proportion = (
                    archetype[
                        "NUMBER OF REFERENCE BUILDINGS IN THE BUILDING STOCK SEGMENT"
                    ]
                    / total_dwellings
                )

                country_proportion.append(proportion)

        geometry_df["COUNTRY ARCHETYPE PROPORTION"] = country_proportion

        return geometry_df


class BuildingLocationProcessor(AbstractProcessor):
    """Processes dwelling location data."""

    def __init__(
        self, features: List[str], data_path: pathlib.Path, common_features: DataFrame
    ) -> None:
        super().__init__(features, data_path=data_path, common_features=common_features)

    def __call__(self) -> DataFrame:
        """Loads raw data and cleans."""
        df = pd.DataFrame(index=self.common_features.index)

        loaded_df = self._load_raw_data()

        try:
            loaded_df = loaded_df[self.features]
        except KeyError as e:
            print(f"Raw location data does not have the required columns: {e}")

        # set index to merge on
        loaded_df = loaded_df.set_index(self.id_column)

        # merge loaded df with common df
        df = pd.concat([df, loaded_df], axis=1)

        return df


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

        # get proportion of each building code in each country
        loaded_df = self._

        # set index to merge on
        loaded_df = loaded_df.set_index(self.id_column)

        upsampled_df = self._upsample(loaded_df)

        # merge loaded df with common df
        df = pd.concat([df, upsampled_df], axis=1)

        return df

    def _calculate_code_proportions(self, loaded_df: DataFrame) -> DataFrame:
        """
        Upsamples the DataFrame to the spatial granularity of the common DataFrame.
        """
        proportions = pd.DataFrame(
            index=COUNTRIES,
            columns=[f"PROPORTION_{code}" for code in RESIDENTIAL_BUILDING_CODES],
        )

        apartment_block_proportions = []
        multi_family_home_proportions = []
        single_family_home_proportions = []
        terrace_house_proportions = []

        for country in COUNTRIES:
            country = loaded_df[loaded_df["REFERENCE BUILDING COUNTRY CODE"] == country]
            total_dwellings = country[
                "NUMBER OF REFERENCE BUILDINGS IN THE BUILDING STOCK SEGMENT"
            ].sum()

            for code in RESIDENTIAL_BUILDING_CODES:
                code_dwellings = country[
                    "NUMBER OF REFERENCE BUILDINGS IN THE BUILDING STOCK SEGMENT"
                ][country["REFERENCE BUILDING USE CODE"] == code].sum()
                if code == "ABL":
                    apartment_block_proportions.append(code_dwellings / total_dwellings)

                elif code == "MFH":
                    multi_family_home_proportions.append(
                        code_dwellings / total_dwellings
                    )

                elif code == "SFH":
                    single_family_home_proportions.append(
                        code_dwellings / total_dwellings
                    )

                elif code == "TH":
                    terrace_house_proportions.append(code_dwellings / total_dwellings)

                else:
                    raise ValueError(f"Unknown code {code}")

        proportions["PROPORTION_ABL"] = apartment_block_proportions
        proportions["PROPORTION_MFH"] = multi_family_home_proportions
        proportions["PROPORTION_SFH"] = single_family_home_proportions
        proportions["PROPORTION_TH"] = terrace_house_proportions

        # merge into loaded_df
        loaded_df = pd.merge(
            loaded_df,
            proportions,
            left_on="REFERENCE BUILDING COUNTRY CODE",
            right_index=True,
        )

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

        simple_windows = df[
            df["Simple Glazing"] == True  # pylint: disable=C0121
        ].copy()
        complex_windows = df[
            df["Simple Glazing"] == False  # pylint: disable=C0121
        ].copy()

        for _, row in complex_windows.iterrows():
            windows[row["Name"]] = WindowMaterialGlazing(
                name=row["Name"],
                optical_data_type=row["Optical Data Type"],
                data_set_name=row["Window Glass Spectral Data Set Name"],
                thickness=row["Thickness"],
                solar_transmittance=row["Solar Transmittance at Normal Incidence"],
                front_side_solar_reflectance=row[
                    "Front Side Solar Reflectance at Normal Incidence"
                ],
                back_side_solar_reflectance=row[
                    "Back Side Solar Reflectance at Normal Incidence"
                ],
                visible_transmittance=row["Visible Transmittance at Normal Incidence"],
                front_side_visible_reflectance=row[
                    "Front Side Visible Reflectance at Normal"
                ],
                back_side_visible_reflectance=row[
                    "Back Side Visible Reflectance at Normal"
                ],
                infrared_transmittance=row[
                    "Infrared Transmittance at Normal Incidence"
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
