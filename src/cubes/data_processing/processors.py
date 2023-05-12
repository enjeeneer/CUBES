"""Module for data_processing row data."""
import itertools

import pandas as pd
from typing import List, Optional, Dict
from pandas import DataFrame
import numpy as np
import abc
import time
import requests
import pathlib
from cubes.data_processing.processor_config import (
    LOCATION_PATH,
    GEOMETRY_PATH,
    COUNTRIES,
    WEATHER_NUTS_3_TRANSFORMATIONS,
    OIKOLAB_API_KEY,
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
        self, features: List[str], base: DataFrame, data_path: pathlib.Path
    ) -> None:
        self._features = features
        self._data_path = data_path
        self._base = base

    @abc.abstractmethod
    def __call__(self) -> DataFrame:
        """Returns database as DataFrame."""
        raise NotImplementedError

    def check_index_match(self, df: DataFrame) -> bool:
        """Checks if DataFrame index matches common index."""
        return df.index.equals(self.base.index)

    @property
    def features(self) -> List[str]:
        """List of features that must be present in DataFrame."""
        return self._features

    @property
    def base(self) -> DataFrame:
        """Index to use for concats."""
        return self._base

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

        # get common df by combining NUTS 3 regions and building archetypes
        df = self._get_common_df(location_df, geometry_df)

        return df

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

        location_df = location_df.set_index("NUTS 3 REGION")
        geometry_df = geometry_df.set_index("REFERENCE BUILDING CODE")

        merged = pd.merge(df, location_df, left_on="NUTS 3 REGION", right_index=True)
        merged = pd.merge(
            merged,
            geometry_df["REFERENCE BUILDING USE CODE"],
            left_on="REFERENCE BUILDING CODE",
            right_index=True,
        )

        return merged


class LocationProcessor(AbstractProcessor):
    """Processes dwelling location data."""

    def __init__(
        self, features: List[str], data_path: pathlib.Path, base: DataFrame
    ) -> None:
        super().__init__(features, data_path=data_path, base=base)

    def __call__(self) -> DataFrame:
        """Loads raw data and cleans."""
        df = pd.DataFrame(index=self.base.index)

        loaded_df = self._load_raw_data()

        try:
            loaded_df = loaded_df[self.features]
        except KeyError as e:
            print(f"Raw location data does not have the required columns: {e}")

        # merge location and geometry dfs into common df
        merged = pd.merge(df, loaded_df, on="NUTS 3 REGION")
        merged = merged.set_index(df.index)

        return merged


class GeometryProcessor(AbstractProcessor):
    """Processes geometric building data."""

    def __init__(
        self, features: List[str], data_path: pathlib.Path, base: DataFrame
    ) -> None:

        super().__init__(features, data_path=data_path, base=base)

    def __call__(self) -> DataFrame:
        """Loads raw data and cleans."""

        df = self.base.copy()
        df = df["REGION OCCUPIED DWELLINGS"]  # keep only conventional dwellings col
        loaded_df = self._load_raw_data()

        try:
            loaded_df = loaded_df[self.features]
        except KeyError as e:
            print(f"Raw geometry does not have the required columns: {e}")

        loaded_df = loaded_df.set_index("REFERENCE BUILDING CODE")
        merged = pd.merge(
            df,
            loaded_df,
            left_on="REFERENCE BUILDING CODE",
            right_index=True,
        )

        # calculate additional features
        merged = self._calculate_archetype_proportions(merged)
        merged = self._calculate_archetypes_per_region(merged)
        merged = self._calculate_window_to_wall_ratio(merged)
        merged = self._calculate_roof_to_floor_ratio(merged)

        merged = merged.drop("REGION OCCUPIED DWELLINGS", axis=1)

        return merged

    @staticmethod
    def _calculate_archetype_proportions(df: DataFrame) -> DataFrame:
        """
        Calculates the proportion of each building archetype in each country.
        Args:
            df: DataFrame with building archetype data.
        Returns:
            df: DataFrame with building archetype proportions.
        """
        df = df.copy()

        country_proportion = []

        for country in COUNTRIES:
            country = df[df["REFERENCE BUILDING COUNTRY CODE"] == country]
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

        df["COUNTRY ARCHETYPE PROPORTION"] = country_proportion

        return df

    @staticmethod
    def _calculate_archetypes_per_region(df: DataFrame) -> DataFrame:
        """Calculates the number of archetypes per NUTS 3 region."""

        df = df.copy()

        df["NUMBER OF DWELLINGS"] = (
            df["COUNTRY ARCHETYPE PROPORTION"] * df["REGION OCCUPIED DWELLINGS"]
        ).astype(int)

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


class HVACProcessor(AbstractProcessor):
    """Processes hvac data."""

    def __init__(
        self,
        features: List[str],
        data_path: pathlib.Path,
        schema_path: pathlib.Path,
        base: DataFrame,
    ) -> None:

        self._schema_path = schema_path
        self._ambience_energy_system_name = "HEATING SYSTEM 1 TECHNOLOGY"

        super().__init__(features, data_path=data_path, base=base)

    def __call__(self) -> DataFrame:
        """Loads raw data and cleans."""

        df = pd.DataFrame(index=self.base.index)
        loaded_df = self._load_raw_data()

        try:
            loaded_df = loaded_df[self.features]
        except KeyError as e:
            print(f"Raw hvac does not have the required columns: {e}")

        # remove non-reliable country data
        loaded_df = loaded_df[
            loaded_df["REFERENCE BUILDING COUNTRY CODE"].isin(COUNTRIES)
        ]

        # map heating system to energyplus
        loaded_df = self._map_ambience_to_energyplus(loaded_df)

        # merge loaded df with common df
        loaded_df = loaded_df.set_index("REFERENCE BUILDING CODE")
        merged = pd.merge(
            df, loaded_df, left_on="REFERENCE BUILDING CODE", right_index=True
        )

        return merged

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
        self, features: List[str], data_path: pathlib.Path, base: DataFrame
    ) -> None:
        super().__init__(features, data_path=data_path, base=base)

    def __call__(self) -> DataFrame:
        """
        Loads and cleans raw air infiltration data.
        We have limited air infiltration data and so assume that, for a given
        building type, a building could have any air infiltration value
        from the dataset. We therefore randomly assign a value to each building.
        """

        df = self.base.copy()
        df = df[["REFERENCE BUILDING USE CODE"]]

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


class WeatherProcessor(AbstractProcessor):
    """
    Processes weather filename data, either by calling the
    OIKOLAB weather API or by loading pre-saved filenames.

    """

    def __init__(
        self,
        features: List[str],
        data_path: pathlib.Path,
        base: DataFrame,
        years: List[str],
    ) -> None:
        super().__init__(features, data_path=data_path, base=base)

        self.years = years

    def __call__(self, call_api: bool = False) -> DataFrame:
        """
        Appends weather file names to base DataFrame. If call_api is True,
        calls OIKOLAB weather API to get EPW files.
        """

        if call_api:
            self._call_api()

        # load weather file names
        loaded_df = self._load_raw_data().set_index(self.base.index)

        try:
            loaded_df = loaded_df[self.features]
        except KeyError as e:
            print(f"Weather data does not have the required columns: {e}")

        return loaded_df

    def _call_api(self) -> None:
        """
        Calls OIKOLAB weather API to get EPW files and writes
        filenames to excel file.
        """

        df = pd.DataFrame(index=self.base.index)

        raw_regions = (
            df.index.get_level_values(0).unique().to_series().reset_index(drop=True)
        )

        # update names of regions with strange names
        regions = raw_regions.replace(WEATHER_NUTS_3_TRANSFORMATIONS)

        # remove '(NUTS {year})' from region names
        regions = regions.apply(lambda x: x.split("(N")[0] + "")
        regions = regions.str.rstrip()

        for year in self.years:
            year_filenames = {}
            for i, region in enumerate(regions):
                print(f"...Retrieving weather data for {region} in {year}...")
                raw_index = raw_regions[i]
                cleaned_region_name = raw_index.replace("/", " ").replace(" ", "_")

                # use the true index, not the cleaned index
                epw_file_name = f"{cleaned_region_name}_{year}.epw"

                r = requests.get(
                    "https://api.oikolab.com/epw",
                    params={"year": year, "location": region},
                    headers={"api-key": OIKOLAB_API_KEY},
                    timeout=10,
                )

                with open(self.data_path.parent / epw_file_name, "wb") as f:
                    f.write(r.content)

                year_filenames[raw_index] = epw_file_name

                # wait 1 seconds between requests
                time.sleep(1)

            # store filenames in dataframe
            year_df = pd.DataFrame.from_dict(
                data=year_filenames,
                orient="index",
                columns=[f"WEATHER FILE {year}"],
            )

            df = pd.merge(
                df,
                year_df,
                left_on="NUTS 3 REGION",
                right_index=True,
            )

        # write dataframe to excel
        df.to_excel(self.data_path)


class SolarPVProcessor(AbstractProcessor):
    """Processes solar PV data."""

    def __init__(
        self, features: List[str], data_path: pathlib.Path, base: DataFrame
    ) -> None:
        super().__init__(features, data_path=data_path, base=base)

    def __call__(self) -> DataFrame:
        """Loads raw data and cleans."""
        base_df = self.base.copy()
        loaded_df = self._load_raw_data(header=5)

        try:
            loaded_df = loaded_df[self.features]
        except KeyError as e:
            print(f"Solar PV data does not have the required columns: {e}")

        # no. of gb buildings = sum of region dwellings indexed by one building code
        gb_df = base_df[(base_df["COUNTRY CODE"] == "GB")]
        one_building_code = gb_df.index.unique(level=1)[0]
        gb_buildings = base_df.loc[pd.IndexSlice[:, one_building_code], :][
            "REGION OCCUPIED DWELLINGS"
        ].sum()

        gb_pv_probability = (
            loaded_df[loaded_df["COUNTRY CODE"] == "GB"][
                "COUNTRY SOLAR PV INSTALLATIONS"
            ]
            / gb_buildings
        )

        loaded_df["SOLAR PV PROBABILITY"] = gb_pv_probability.values[0]

        # merge
        df = pd.merge(base_df, loaded_df, on=["COUNTRY CODE"]).set_index(
            self.base.index
        )

        df = df[
            itertools.chain(self.features, ["SOLAR PV PROBABILITY"])
        ]  # drop duplicate columns

        return df


class BatteriesProcessor(AbstractProcessor):
    """Processes battery data."""

    def __init__(
        self, features: List[str], data_path: pathlib.Path, base: DataFrame
    ) -> None:
        super().__init__(features, data_path=data_path, base=base)

    def __call__(self):
        """Loads raw battery data and cleans."""
        df = self.base.copy()
        df = df["REFERENCE BUILDING USE CODE"]
        loaded_df = self._load_raw_data(header=2)

        try:
            loaded_df = loaded_df[self.features]
        except KeyError as e:
            print(f"Battery data does not have the required columns: {e}")

        # set index
        loaded_df = loaded_df.set_index("REFERENCE BUILDING USE CODE")

        # merge
        df = pd.merge(
            df, loaded_df, left_on="REFERENCE BUILDING USE CODE", right_index=True
        ).drop("REFERENCE BUILDING USE CODE", axis=1)

        return df


class FridgeFreezerProcessor(AbstractProcessor):
    """Processes fridge/freezer data."""

    def __init__(
        self, features: List[str], data_path: pathlib.Path, base: DataFrame
    ) -> None:
        super().__init__(features, data_path=data_path, base=base)

    def __call__(self):
        """Loads raw fridge/freezer data and cleans."""
        loaded_df = self._load_raw_data(header=4)

        try:
            loaded_df = loaded_df[self.features]
        except KeyError as e:
            print(f"Fridge/freezer data does not have the required columns: {e}")

        # fix defrost
        loaded_df["FRIDGE CASE DEFROST TYPE"] = "None"
        loaded_df["FREEZER CASE DEFROST TYPE"] = "None"

        # copy to length of base
        df = loaded_df.loc[loaded_df.index.repeat(len(self.base))]

        # copy base index
        df.index = self.base.index

        return df


class ElectricVehicleProcessor(AbstractProcessor):
    """Processes electric vehicle data."""

    def __init__(self, features: List[str], data_path: pathlib.Path, base: DataFrame):
        super().__init__(features, data_path=data_path, base=base)

    def __call__(self):
        """Loads raw electric vehicle data and cleans."""
        df = self.base.copy()
        df = df["COUNTRY CODE"]
        loaded_df = self._load_raw_data(header=8)

        try:
            loaded_df = loaded_df[self.features]
        except KeyError as e:
            print(f"Electric vehicle data does not have the required columns: {e}")

        # set index
        loaded_df = loaded_df.set_index("COUNTRY CODE")

        # merge
        df = pd.merge(df, loaded_df, left_on="COUNTRY CODE", right_index=True)

        return df


class MaterialsProcessor(AbstractProcessor):
    """Processes materials data."""

    def __init__(
        self, features: List[str], data_path: pathlib.Path, base: DataFrame
    ) -> None:
        super().__init__(features, data_path=data_path, base=base)

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
        self, features: List[str], data_path: pathlib.Path, base: DataFrame
    ) -> None:
        super().__init__(features, data_path=data_path, base=base)

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
