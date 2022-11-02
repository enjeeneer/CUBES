"""outward facing API of construct package"""

import building
import sample
import constants as con


def sample_idf():
    geometry_data, systems_data = sample.sample_database(
        con.filtered_geometry_data, con.clean_system_data
    )
    build = building.Building(geometry_data, systems_data)
    build.build()
    idf = build.get_idf()

    return idf


idf1 = sample_idf()
idf1.save(filename="exp/hannes/construct-tests/test1.idf")
idf1.run(
    expandobjects=True,
    weather=(
        "/workspaces/elizabeth-homes/src/cubes/data/"
        "weather/cambridge_lat=52.25_lng=0.25_period=2021.epw"
    ),
    output_directory="exp/hannes/construct-tests/test1/",
)


# idf1.to_obj("exp/hannes/construct-tests/test1.obj")
# idf1.view_model()
