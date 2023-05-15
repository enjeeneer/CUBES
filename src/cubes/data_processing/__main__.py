"""Processes all datasets."""
# pylint: disable=invalid-name

from loguru import logger

from evaluators import (
    BuildingDataEvaluator,
    MaterialDataEvaluator,
    WindowsDataEvaluator,
)
from processors import (
    BaseProcessor,
    GeometryProcessor,
    HVACProcessor,
    AirInfiltrationProcessor,
    LocationProcessor,
    MaterialsProcessor,
    WindowsProcessor,
    WeatherProcessor,
    SolarPVProcessor,
    BatteriesProcessor,
    FridgeFreezerProcessor,
    ElectricVehicleProcessor,
)

from processor_config import (
    GEOMETRY_FEATURES,
    GEOMETRY_PATH,
    HVAC_FEATURES,
    HVAC_PATH,
    HVAC_SCHEMA_PATH,
    AIR_INFILTRATION_FEATURES,
    AIR_INFILTRATION_PATH,
    MATERIALS_PATH,
    MATERIALS_FEATURES,
    WINDOWS_PATH,
    WINDOWS_FEATURES,
    LOCATION_FEATURES,
    LOCATION_PATH,
    WEATHER_YEARS,
    WEATHER_PATH,
    WEATHER_FEATURES,
    SOLAR_PV_PATH,
    SOLAR_PV_FEATURES,
    BATTERIES_PATH,
    BATTERIES_FEATURES,
    FRIDGE_FREEZER_FEATURES,
    FRIDGE_FREEZER_PATH,
    ELECTRIC_VEHICLE_PATH,
    ELECTRIC_VEHICLE_FEATURES,
)

base_df = BaseProcessor()()

evaluator = BuildingDataEvaluator(
    processors=[
        BaseProcessor(),
        LocationProcessor(
            features=LOCATION_FEATURES, data_path=LOCATION_PATH, base=base_df
        ),
        GeometryProcessor(
            features=GEOMETRY_FEATURES, data_path=GEOMETRY_PATH, base=base_df
        ),
        HVACProcessor(
            features=HVAC_FEATURES,
            data_path=HVAC_PATH,
            schema_path=HVAC_SCHEMA_PATH,
            base=base_df,
        ),
        AirInfiltrationProcessor(
            features=AIR_INFILTRATION_FEATURES,
            data_path=AIR_INFILTRATION_PATH,
            base=base_df,
        ),
        WeatherProcessor(
            features=WEATHER_FEATURES,
            data_path=WEATHER_PATH,
            years=WEATHER_YEARS,
            base=base_df,
        ),
        ElectricVehicleProcessor(
            features=ELECTRIC_VEHICLE_FEATURES,
            data_path=ELECTRIC_VEHICLE_PATH,
            base=base_df,
        ),
        BatteriesProcessor(
            features=BATTERIES_FEATURES, data_path=BATTERIES_PATH, base=base_df
        ),
        FridgeFreezerProcessor(
            features=FRIDGE_FREEZER_FEATURES,
            data_path=FRIDGE_FREEZER_PATH,
            base=base_df,
        ),
        SolarPVProcessor(
            features=SOLAR_PV_FEATURES, data_path=SOLAR_PV_PATH, base=base_df
        ),
    ],
    base_index=base_df.index,
)

materials_evaluator = MaterialDataEvaluator(
    processor=MaterialsProcessor(
        features=MATERIALS_FEATURES, data_path=MATERIALS_PATH, base=base_df
    )
)

windows_evaluator = WindowsDataEvaluator(
    processor=WindowsProcessor(
        features=WINDOWS_FEATURES, data_path=WINDOWS_PATH, base=base_df
    )
)

if __name__ == "__main__":
    logger.info("Processing datasets.")
    # buildings = evaluator()
    materials = materials_evaluator()
    windows = windows_evaluator()
    logger.info("Processing complete.")
