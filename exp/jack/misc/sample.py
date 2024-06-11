# pylint: disable=all
import pandas as pd
import numpy as np

raw_data = pd.read_excel("../Data/Ambience.xlsx")


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


def sample(data):

    calc_roof_floor_ratio(data)
    calc_window_wall_ratio(data)

    sampled = data.sample(
        n=1, weights=data["NUMBER OF REFERENCE BUILDINGS IN THE BUILDING STOCK SEGMENT"]
    )

    noise = 0.1  # fractional change in geometry value/standard deviation

    geometry_elements = [
        "REFERENCE BUILDING GROUND FLOOR AREA (m2)",
        "REFERENCE BUILDING WINDOW WALL RATIO",
        "REFERENCE BUILDING FLOOR ROOF RATIO",
    ]

    for element in geometry_elements:

        deviation = sampled[element] * noise

        sampled[element] = np.random.normal(sampled[element], deviation)

        if element == "REFERENCE BUILDING FLOOR ROOF RATIO":

            if sampled[element].values[0] < 1:
                sampled[element] = 1

    return sampled
