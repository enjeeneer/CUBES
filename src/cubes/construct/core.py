"""outward facing API of construct package"""

from cubes.construct import building
from cubes.construct import sample
from cubes.construct import constants

import os
from pathlib import Path


def sample_idf():
    geometry_data, systems_data = sample.sample_database(
        constants.filtered_geometry_data, constants.clean_system_data
    )
    build = building.Building(geometry_data, systems_data)
    build.build()
    idf = build.get_idf()

    return idf


def test_idf():

    idf1 = sample_idf()
    cwd_path = os.getcwd()
    env_data_path = os.path.join(cwd_path, "input_case_1")
    Path(env_data_path).mkdir(parents=True, exist_ok=True)

    idf1.save(filename=env_data_path + "test1.idf")
    idf1.run(
        expandobjects=True,
        weather=(
            "/workspaces/elizabeth-homes/src/cubes/data/"
            "weather/cambridge_lat=52.25_lng=0.25_period=2021.epw"
        ),
        output_directory=env_data_path + "output/",
    )


# idf1.to_obj("exp/hannes/construct-tests/test1.obj")
# idf1.view_model()
