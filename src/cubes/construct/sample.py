"""This module has functions to sample a building from a database plus
additional attributes from distributions"""

import numpy as np


def calc_roof_floor_ratio(data):
    data["REFERENCE BUILDING FLOOR ROOF RATIO"] = (
        data["REFERENCE BUILDING ROOF AREA (m2)"]
        / data["REFERENCE BUILDING GROUND FLOOR AREA (m2)"]
    )
    return data


def calc_window_wall_ratio(data):
    data["REFERENCE BUILDING WINDOW WALL RATIO"] = (
        data["REFERENCE BUILDING WINDOW AREA (m2)"]
        / data["REFERENCE BUILDING WALL AREA (m2)"]
    )
    return data


def sample_database(geometry_data, systems_data):

    calc_roof_floor_ratio(geometry_data)
    calc_window_wall_ratio(geometry_data)

    archetype_geometry = geometry_data.sample(
        n=1,
        weights=geometry_data[
            "NUMBER OF REFERENCE BUILDINGS IN THE BUILDING STOCK SEGMENT"
        ],
        ignore_index=True,
    )

    # These are the data which have noise added to
    # - can add more elements into the future
    geometry_elements = [
        "REFERENCE BUILDING GROUND FLOOR AREA (m2)",
        "REFERENCE BUILDING WINDOW WALL RATIO",
        "REFERENCE BUILDING FLOOR ROOF RATIO",
    ]

    for element in geometry_elements:

        deviation = 1  # Assumption - This can be altered in the future

        archetype_geometry[element] = np.random.normal(
            archetype_geometry[element], deviation
        )

        if element == "REFERENCE BUILDING FLOOR ROOF RATIO":

            if archetype_geometry[element].values[0] < 1:
                archetype_geometry[element] = 1

    archetype_system = systems_data[
        systems_data["Building typology"]
        == archetype_geometry["REFERENCE BUILDING CODE"].values[0]
    ]

    return archetype_geometry, archetype_system
