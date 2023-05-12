"""Module for sampling data from mono-dataframe."""

from typing import List, Dict

import pandas as pd
import numpy as np
from pandas import DataFrame


class BuildingDataSampler:
    """Class for concatenating building data from different sources."""

    def __init__(
        self,
        dataset: DataFrame,
        gaussian_sampled_features: List[str],  # columns we sample with gaussian noise
        beta_sampled_features: List[str],  # columns we sample from a bernoulli dist
        gaussian_noise_param: float,  # std dev as fraction of mean
        beta_parameters: Dict[str, float],  # alpha, beta parameters
        weight_column=None,  # column used for weighted sampling
    ) -> None:

        if weight_column is None:
            self.weight_column = ["NUMBER OF DWELLINGS"]

        self.dataset = dataset
        self.gaussian_sampled_features = gaussian_sampled_features
        self.beta_sampled_features = beta_sampled_features
        self.gaussian_noise_param = gaussian_noise_param
        self.beta_parameters = beta_parameters

    def __call__(self, n: int) -> DataFrame:
        """Samples n buildings from DataFrame."""

        # index into dataset by sampling region-archetype pair with weights
        sample = self.dataset.sample(n, weights=self.weight_column)

        # sample number of occupants
        sample = self._sample_occupants(sample)

        # add noise
        sample = self._add_noise(sample)

        # sample PV + battery
        sample = self._sample_pv_and_battery(sample)

        # sample electric vehicle
        sample = self._sample_electric_vehicle(sample)

        sample = pd.DataFrame(columns=sample.columns).rename(
            lambda x: x.replace("MEAN ", "")
            if x.replace("MEAN ", "") in self.gaussian_sampled_features
            else x
        )

        return sample

    def _sample_occupants(self, sample: DataFrame) -> DataFrame:
        """Samples occupants and rounds to nearest integer."""
        occupant_mean = sample["MEAN REGION MEAN OCCUPANTS PER BUILDING"]

        # sample number of occupants and round
        sample["NUMBER OF OCCUPANTS"] = np.round(
            np.random.normal(
                loc=occupant_mean, scale=occupant_mean * self.gaussian_noise_param
            )
        )

        return sample

    def _add_noise(self, sample: DataFrame) -> DataFrame:
        """Calculates std. dev of sampled features and adds gaussian noise."""

        # add gaussian noise to gaussian sampled features
        std_dev = sample[self.gaussian_sampled_features] * self.gaussian_noise_param
        sample[self.gaussian_sampled_features] += np.random.normal(0, scale=std_dev)

        return sample

    def _sample_pv_and_battery(self, sample: DataFrame) -> DataFrame:
        """Samples PV and battery."""

        # sample PV
        sample[["PV PRESENT", "BATTERY PRESENT"]] = (
            np.random.random() < sample["SOLAR PV PROBABILITY"]
        )

        # get battery size
        sample["BATTERY SIZE"] = np.random.choice(
            sample[
                [
                    "SMALL BATTERY SIZE (KWH)",
                    "MEDIUM BATTERY SIZE (KWH)",
                    "LARGE BATTERY SIZE (KWH)",
                ]
            ],
            p=sample[
                [
                    "SMALL BATTERY SIZE PROBABILITY",
                    "MEDIUM BATTERY SIZE PROBABILITY",
                    "LARGE BATTERY SIZE PROBABILITY",
                ]
            ],
        )

        return sample

    def _sample_electric_vehicle(self, sample: DataFrame) -> DataFrame:
        """Samples electric vehicle."""

        # sample EV and PHEV
        sample["BEV PRESENT"] = np.random.random() < sample["COUNTRY PROBABILITY EVS"]
        if sample["BEV PRESENT"]:
            sample["PHEV PRESENT"] = False
        else:
            sample["PHEV PRESENT"] = (
                np.random.random() < sample["COUNTRY PROBABILITY PHEVS"]
            )

        # sample battery sizes
        sample["BEV BATTERY SIZE"] = sample[
            "BEV MINIMUM BATTERY SIZE (KWH)"
        ] + np.random.beta(
            self.beta_parameters["ELECTRIC VEHICLE ALPHA"],
            self.beta_parameters["ELECTRIC VEHICLE BETA"],
            size=len(sample),
        ) * (
            sample["BEV MAXIMUM BATTERY SIZE (KWH)"]
            - sample["BEV MINIMUM BATTERY SIZE (KWH)"]
        )

        sample["PHEV BATTERY SIZE"] = sample[
            "PHEV MINIMUM BATTERY SIZE (KWH)"
        ] + np.random.beta(
            self.beta_parameters["ELECTRIC VEHICLE ALPHA"],
            self.beta_parameters["ELECTRIC VEHICLE BETA"],
            size=len(sample),
        ) * (
            sample["PHEV MAXIMUM BATTERY SIZE (KWH)"]
            - sample["PHEV MINIMUM BATTERY SIZE (KWH)"]
        )

        return sample
