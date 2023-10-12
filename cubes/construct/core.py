"""outward facing API of construct package"""
from typing import Dict
from loguru import logger

# from cubes.construct import building

from cubes.data_processing.evaluators import (
    BuildingDataEvaluator,
    MaterialDataEvaluator,
    WindowsDataEvaluator,
)
from cubes.data_processing.processors import (
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
    GridCarbonProcessor,
    VentilationProcessor,
)

from cubes.data_processing.processor_config import (
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
    YEARS,
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
    GRID_CARBON_PATH,
    GRID_FEATURES,
    VENTILATION_FEATURES,
    VENTILATION_PATH,
)
from cubes.data_processing.samplers import BuildingDataSampler
from cubes.data_processing.sampler_config import (
    GAUSSIAN_SAMPLED_FEATURES,
    GAUSSIAN_NOISE_STD_DEV_PARAMETER,
    BETA_SAMPLED_FEATURES,
    BETA_PARAMETERS,
)
from cubes.construct.extractor import BuildingConfigExtractor
from cubes.construct.building import Building

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
            years=YEARS,
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
        GridCarbonProcessor(
            features=GRID_FEATURES,
            data_path=GRID_CARBON_PATH,
            base=base_df,
            years=YEARS,
        ),
        VentilationProcessor(
            features=VENTILATION_FEATURES, data_path=VENTILATION_PATH, base=base_df
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

sampler = BuildingDataSampler(
    gaussian_sampled_features=GAUSSIAN_SAMPLED_FEATURES,
    beta_sampled_features=BETA_SAMPLED_FEATURES,
    gaussian_noise_param=GAUSSIAN_NOISE_STD_DEV_PARAMETER,
    beta_parameters=BETA_PARAMETERS,
)


def sample_idf(n: int):
    logger.info("Processing datasets.")

    # evaluate raw data
    buildings = evaluator()
    materials = materials_evaluator()
    windows = windows_evaluator()

    # sample data
    logger.info("Sampling data.")
    sample = sampler(dataset=buildings, n=n)

    # paralleise building config extraction
    # with mp.Pool() as pool:
    #     building_configs =
    #     pool.map(create_building_config_instance, sample.to_dict("records"))

    building_config = create_building_config_instance(sample.to_dict("records")[0])

    logger.info("Building IDF.")
    build = Building(
        building_config=building_config, materials=materials, windows=windows
    )
    build.build()
    idf = build.get_idf()

    logger.info("IDF built.")

    return idf, building_config


# def test_idf():
#     idf1 = sample_idf(n=1)
#     cwd_path = os.getcwd()
#     env_data_path = os.path.join(cwd_path, "input_case_1")
#     Path(env_data_path).mkdir(parents=True, exist_ok=True)
#
#     idf1.save(filename=env_data_path + "test1.idf")
#     idf1.run(
#         expandobjects=True,
#         weather=(
#             "/workspaces/elizabeth-homes/src/cubes/data/"
#             "weather/cambridge_lat=52.25_lng=0.25_period=2021.epw"
#         ),
#         output_directory=env_data_path + "output/",
#     )


def create_building_config_instance(row: Dict):
    """Used for parallelising building config extraction."""
    return BuildingConfigExtractor()(row)


# idf1.to_obj("exp/hannes/construct-tests/test1.obj")
# idf1.view_model()

# idf = sample_idf(n=1)
