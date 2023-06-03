"""Module for sampling data from mono-dataframe."""

from typing import List, Dict

import pandas as pd
import numpy as np
from pandas import DataFrame


class BuildingDataSampler:
    """Class for concatenating building data from different sources."""

    def __init__(
        self,
        gaussian_sampled_features: List[str],  # columns we sample with gaussian noise
        beta_sampled_features: List[str],  # columns we sample from a bernoulli dist
        gaussian_noise_param: float,  # std dev as fraction of mean
        beta_parameters: Dict[str, float],  # alpha, beta parameters
        weight_column: str = None,  # column used for weighted sampling
    ) -> None:

        if weight_column is None:
            self.weight_column = "NUMBER OF DWELLINGS"

        self.gaussian_sampled_features = gaussian_sampled_features
        self.beta_sampled_features = beta_sampled_features
        self.gaussian_noise_param = gaussian_noise_param
        self.beta_parameters = beta_parameters

    def __call__(self, dataset: DataFrame, n: int) -> DataFrame:
        """Samples n buildings from DataFrame."""

        # get sample weights
        sample_weights = dataset[self.weight_column] / dataset[self.weight_column].sum()

        # index into dataset by sampling region-archetype pair with weights
        sample = dataset.sample(n, weights=sample_weights)

        # sample number of occupants
        sample = self._sample_occupants(sample)

        # add noise
        sample = self._add_noise(sample)

        # sample PV + battery
        sample = self._sample_pv_and_battery(sample)

        # sample electric vehicle
        sample = self._sample_electric_vehicle(sample)

        # sample rotation
        sample = self._sample_rotation(sample)

        # sample year
        sample = self._sample_year(sample)

        cleaned_sampled = pd.DataFrame(data=sample, columns=sample.columns).rename(
            columns=lambda x: x.replace("MEAN ", "")
            if x.replace("MEAN ", "") in self.gaussian_sampled_features
            else x
        )

        return cleaned_sampled

    def _sample_occupants(self, sample: DataFrame) -> DataFrame:
        """Samples occupants and rounds to nearest integer."""
        occupant_mean = sample["REGION MEAN OCCUPANTS PER BUILDING"]

        # sample number of occupants and round
        sample["NUMBER OF OCCUPANTS"] = np.round(
            np.random.normal(
                loc=occupant_mean, scale=occupant_mean * self.gaussian_noise_param
            )
        ).astype("int")

        return sample

    def _add_noise(self, sample: DataFrame) -> DataFrame:
        """Calculates std. dev of sampled features and adds gaussian noise."""

        # find columns which require noise
        # (those whose values are the mean of an assumed gaussian)
        noise_columns = sample.filter(like="MEAN").columns

        # add gaussian noise to gaussian sampled features
        std_dev = sample[noise_columns] * self.gaussian_noise_param
        sample[noise_columns] += np.random.uniform(low=-std_dev, high=std_dev)

        # clip some features
        sample["MEAN SOLAR PV ACTIVE AREA FRACTION"] = np.clip(
            sample["MEAN SOLAR PV ACTIVE AREA FRACTION"], 0, 1
        )
        sample["MEAN HEATING SYSTEM 1 EFFICIENCY"] = np.clip(
            sample["MEAN HEATING SYSTEM 1 EFFICIENCY"], 0, 1
        )
        sample["MEAN HEATING SYSTEM 2 EFFICIENCY"] = np.clip(
            sample["MEAN HEATING SYSTEM 2 EFFICIENCY"], 0, 1
        )
        sample["MEAN HEATING SYSTEM 3 EFFICIENCY"] = np.clip(
            sample["MEAN HEATING SYSTEM 3 EFFICIENCY"], 0, 1
        )
        sample["MEAN DHW SYSTEM 1 EFFICIENCY"] = np.clip(
            sample["MEAN HEATING SYSTEM 1 EFFICIENCY"], 0, 1
        )
        sample["MEAN DHW SYSTEM 2 EFFICIENCY"] = np.clip(
            sample["MEAN HEATING SYSTEM 2 EFFICIENCY"], 0, 1
        )
        sample["MEAN DHW SYSTEM 3 EFFICIENCY"] = np.clip(
            sample["MEAN HEATING SYSTEM 3 EFFICIENCY"], 0, 1
        )
        sample["MEAN SOLAR PV PANEL EFFICIENCY"] = np.clip(
            sample["MEAN SOLAR PV PANEL EFFICIENCY"], 0, 1
        )

        # ensure we havent sampled negative values
        negative_samples = sample[noise_columns] < 0
        if negative_samples.any():
            raise ValueError(
                f"Negative values sampled for "
                f"{negative_samples.columns[negative_samples.any()]}:"
                f" {negative_samples[negative_samples.any()]}"
            )

        return sample

    def _sample_pv_and_battery(self, sample: DataFrame) -> DataFrame:
        """Samples PV and battery."""
        # start with no PV or battery
        sample[["PV PRESENT", "BATTERY PRESENT"]] = False

        # sample PV for non-apartments
        apartment_bool = sample["REFERENCE BUILDING USE CODE"] == "ABL"
        sample.loc[~apartment_bool, ["PV PRESENT", "BATTERY PRESENT"]] = (
            np.random.random() < sample["SOLAR PV PROBABILITY"].values[0]
        )

        # get battery size
        battery_size = sample[
            [
                "SMALL BATTERY SIZE (KWH)",
                "MEDIUM BATTERY SIZE (KWH)",
                "LARGE BATTERY SIZE (KWH)",
            ]
        ].values[0]

        weights = sample[
            [
                "SMALL BATTERY SIZE PROBABILITY",
                "MEDIUM BATTERY SIZE PROBABILITY",
                "LARGE BATTERY SIZE PROBABILITY",
            ]
        ].values

        # sample battery size
        sampled_battery_size = []
        for weight in weights:
            if weight.sum() == 0:  # apartments
                sampled_battery_size.append(0)
                continue
            sampled_battery_size.append(np.random.choice(battery_size, p=weight))

        sample["BATTERY SIZE (KWH)"] = sampled_battery_size

        return sample

    def _sample_distance_to_ground(self, sample: DataFrame) -> DataFrame:
        """Samples storey, then distance to ground for apartments."""

        sample["STOREY"] = np.random.choice(
            sample["NUMBER OF REFERENCE BUILDING STOREYS"], size=len(sample)
        )
        sample["DISTANCE TO GROUND"] = (
            sample["STOREY"] * sample["REFERENCE BUILDING FLOOR TO FLOOR HEIGHT"]
        )  # 3.5m
        apartment_bool = sample["REFERENCE BUILDING USE CODE"] == "ABL"

        # NAN non-apartments
        sample.loc[~apartment_bool, ["STOREY", "DISTANCE TO GROUND"]] = pd.NA

        return sample

    def _sample_electric_vehicle(self, sample: DataFrame) -> DataFrame:
        """Samples electric vehicle."""
        # start with no BEV or PHEV
        sample[["BEV PRESENT", "PHEV PRESENT"]] = False

        # sample BEV
        sample["BEV PRESENT"] = np.random.random() < sample["COUNTRY PROBABILITY BEVS"]

        # for those without BEV, sample PHEV
        sample.loc[~sample["BEV PRESENT"], "PHEV PRESENT"] = (
            np.random.random() < sample["COUNTRY PROBABILITY PHEVS"]
        )

        # sample battery sizes irrespective of BEV or PHEV presence
        sample["BEV BATTERY SIZE"] = sample[
            "BEV MINIMUM BATTERY SIZE (kWh)"
        ] + np.random.beta(
            self.beta_parameters["BEV ALPHA"],
            self.beta_parameters["BEV BETA"],
            size=len(sample),
        ) * (
            sample["BEV MAXIMUM BATTERY SIZE (kWh)"]
            - sample["BEV MINIMUM BATTERY SIZE (kWh)"]
        )

        sample["PHEV BATTERY SIZE"] = sample[
            "PHEV MINIMUM BATTERY SIZE (kWh)"
        ] + np.random.beta(
            self.beta_parameters["PHEV ALPHA"],
            self.beta_parameters["PHEV BETA"],
            size=len(sample),
        ) * (
            sample["PHEV MAXIMUM BATTERY SIZE (kWh)"]
            - sample["PHEV MINIMUM BATTERY SIZE (kWh)"]
        )

        return sample

    def _sample_rotation(self, sample: DataFrame) -> DataFrame:
        """
        Samples rotation angle (in degrees of building).
        """

        # sample rotation angle
        sample["ROTATION"] = np.random.uniform(0, 360, size=len(sample))

        return sample

    def _sample_year(self, sample: DataFrame) -> DataFrame:
        """
        Samples year, and year-dependent features e.g. weather.
        """

        year = np.random.choice([2017, 2018, 2019, 2020, 2021, 2022], size=len(sample))

        sample["SIMULATION YEAR"] = year[0]
        sample["WEATHER FILE"] = sample[f"WEATHER FILE {year[0]}"]
        sample["GRID CARBON FILE"] = sample[f"GRID CARBON FILE {year[0]}"]

        return sample
