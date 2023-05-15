"""Module for data evaluators that merge disparate data sources."""

import pandas as pd
import pickle
from typing import List, Dict
from pandas import DataFrame
from processors import AbstractProcessor, MaterialsProcessor, WindowsProcessor
from loguru import logger
from cubes.data_processing.processor_config import (
    CLEANED_BUILDING_DATASET_PATH,
    CLEANED_MATERIAL_DATASET_PATH,
    CLEANED_WINDOWS_DATASET_PATH,
)


class BuildingDataEvaluator:
    """Class for concatenating building data from different sources."""

    def __init__(
        self, processors: List[AbstractProcessor], base_index: pd.Index
    ) -> None:

        self.processors = processors
        self.base_index = base_index

    def __call__(self) -> DataFrame:
        """Returns merged building data DataFrame."""
        logger.info("Processing building data.")
        df = pd.DataFrame(index=self.base_index)

        for processor in self.processors:
            processed_df = processor()
            df = pd.concat([df, processed_df], axis=1)

        # drop any duplicate columns that have hung around after merge
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

        df.to_csv(CLEANED_BUILDING_DATASET_PATH)

        return df


class MaterialDataEvaluator:
    """Class for processing material data."""

    def __init__(self, processor: MaterialsProcessor):

        self.processor = processor

    def __call__(self) -> Dict:
        logger.info("Processing materials data.")
        materials = self.processor()

        with open(CLEANED_MATERIAL_DATASET_PATH, "wb") as f:
            pickle.dump(materials, f)

        return materials


class WindowsDataEvaluator:
    """Class for processing window data."""

    def __init__(self, processor: WindowsProcessor):

        self.processor = processor

    def __call__(self) -> Dict:
        logger.info("Processing windows data.")
        windows = self.processor()

        with open(CLEANED_WINDOWS_DATASET_PATH, "wb") as f:
            pickle.dump(windows, f)

        return windows
