"""This module has functions to sample a building from a database plus
additional attributes from distributions"""

import numpy as np


def sample_database(geometry_data, systems_data):

    archetype_geometry = geometry_data.sample(
        n=1,
        weights=geometry_data["BUILDING STOCK SEGMENT NUMBER OF BUILDINGS"],
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

    if "GB" in archetype_geometry["REFERENCE BUILDING CODE"].values[0]:

        archetype_system = systems_data[
            systems_data["Building typology"] == "DE-SFH-2002-2009-00"
        ]

    else:
        archetype_system = systems_data[
            systems_data["Building typology"]
            == archetype_geometry["REFERENCE BUILDING CODE"].values[0]
        ]

    return archetype_geometry, archetype_system
