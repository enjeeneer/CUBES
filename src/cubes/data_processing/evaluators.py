"""Module for data evaluators that merge disparate data sources."""

import pandas as pd
from typing import List, Dict
from pandas import DataFrame
from processors import AbstractProcessor, MaterialsProcessor, WindowsProcessor


class BuildingDataEvaluator:
    """Class for concatenating building data from different sources."""

    def __init__(
        self,
        processors: List[AbstractProcessor],
        common_index: pd.Index,
    ) -> None:

        self.processors = processors
        self.common_index = common_index

    def __call__(self) -> DataFrame:
        """Returns merged building data DataFrame."""
        df = pd.DataFrame(index=self.common_index)

        for processor in self.processors:
            processed_df = processor()
            df = pd.concat([df, processed_df], axis=1)

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
