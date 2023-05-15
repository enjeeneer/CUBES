"""Module for data evaluators that merge disparate data sources."""

import pandas as pd
from typing import List, Dict
from pandas import DataFrame
from processors import AbstractProcessor, MaterialsProcessor, WindowsProcessor
from loguru import logger


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
        logger.info("Building data processed.")

        return df


class MaterialDataEvaluator:
    """Class for processing material data."""

    def __init__(self, processor: MaterialsProcessor):

        self.processor = processor

    def __call__(self) -> Dict:

        return self.processor()


class WindowsDataEvaluator:
    """Class for processing window data."""

    def __init__(self, processor: WindowsProcessor):

        self.processor = processor

    def __call__(self) -> Dict:

        return self.processor()
