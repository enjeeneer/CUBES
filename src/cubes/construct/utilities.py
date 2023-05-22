"""some utility functions to be used throughout the construct package"""

import numpy as np
from cubes.constants import package_directory


def get_walls_in_limits(
    idf, x_lims=(-1e4, 1e4), y_lims=(-1e4, 1e4), z_lims=(-1e4, 1e4)
):
    walls = []

    for wall in idf.getsurfaces("wall"):
        x_coords = [
            wall.Vertex_1_Xcoordinate,
            wall.Vertex_2_Xcoordinate,
            wall.Vertex_3_Xcoordinate,
        ]
        y_coords = [
            wall.Vertex_1_Ycoordinate,
            wall.Vertex_2_Ycoordinate,
            wall.Vertex_3_Ycoordinate,
        ]
        z_coords = [
            wall.Vertex_1_Zcoordinate,
            wall.Vertex_2_Zcoordinate,
            wall.Vertex_3_Zcoordinate,
        ]

        if (
            min(x_coords) > x_lims[0]
            and max(x_coords) < x_lims[1]
            and min(y_coords) > y_lims[0]
            and max(y_coords) < y_lims[1]
            and min(z_coords) > z_lims[0]
            and max(z_coords) < z_lims[1]
        ):
            walls.append(wall)

    return walls


def get_shading_surface_start_coordinates(x_idx, y_idx, n_layers_total, lx, ly, d):

    if x_idx < n_layers_total:
        x_min = (
            x_idx * lx - np.ceil((-x_idx) / 2) * d[3] - np.floor((-x_idx) / 2) * d[1]
        )

    else:
        x_min = x_idx * lx + np.ceil(x_idx / 2) * d[1] + np.floor(x_idx / 2) * d[3]

    if y_idx < n_layers_total:
        y_min = (
            y_idx * ly - np.ceil((-y_idx) / 2) * d[2] - np.floor((-y_idx) / 2) * d[0]
        )

    else:
        y_min = y_idx * ly + np.ceil(y_idx / 2) * d[0] + np.floor(y_idx / 2) * d[2]

    return (x_min, y_min)


def get_surface_area(surface_object):
    """only works for quadrilaterals so far:
    area is 1/2 * product of length of the two diagonals"""
    d1 = np.sqrt(
        (surface_object.Vertex_1_Xcoordinate - surface_object.Vertex_3_Xcoordinate) ** 2
        + (surface_object.Vertex_1_Ycoordinate - surface_object.Vertex_3_Ycoordinate)
        ** 2
        + (surface_object.Vertex_1_Zcoordinate - surface_object.Vertex_3_Zcoordinate)
        ** 2
    )
    d2 = np.sqrt(
        (surface_object.Vertex_2_Xcoordinate - surface_object.Vertex_4_Xcoordinate) ** 2
        + (surface_object.Vertex_2_Ycoordinate - surface_object.Vertex_4_Ycoordinate)
        ** 2
        + (surface_object.Vertex_2_Zcoordinate - surface_object.Vertex_4_Zcoordinate)
        ** 2
    )
    return 0.5 * d1 * d2


def get_surface_orientation(surface_object):
    """calculate surface orientation relative to building north
    needs to be adjusted for building rotation"""
    vec12 = [
        surface_object.Vertex_2_Xcoordinate - surface_object.Vertex_1_Xcoordinate,
        surface_object.Vertex_2_Ycoordinate - surface_object.Vertex_1_Ycoordinate,
        surface_object.Vertex_2_Zcoordinate - surface_object.Vertex_1_Zcoordinate,
    ]
    vec23 = [
        surface_object.Vertex_3_Xcoordinate - surface_object.Vertex_2_Xcoordinate,
        surface_object.Vertex_3_Ycoordinate - surface_object.Vertex_2_Ycoordinate,
        surface_object.Vertex_3_Zcoordinate - surface_object.Vertex_2_Zcoordinate,
    ]

    norm = np.cross(vec12, vec23)
    north = [0, 1, 0]
    west = [0, -1, 0]

    angle_to_north = (
        180
        / np.pi
        * np.arccos(
            np.dot(north, norm) / (np.linalg.norm(north) * np.linalg.norm(norm))
        )
    )
    angle_to_west = (
        180
        / np.pi
        * np.arccos(np.dot(west, norm) / (np.linalg.norm(west) * np.linalg.norm(norm)))
    )
    if angle_to_west <= 90:
        return 360 - angle_to_north
    else:
        return angle_to_north


def get_surface_vertical_midpoint(surface_object):
    """calculate surface vertical midpoint"""
    z_coordinates = [
        surface_object.Vertex_1_Zcoordinate,
        surface_object.Vertex_1_Zcoordinate,
        surface_object.Vertex_1_Zcoordinate,
        surface_object.Vertex_1_Zcoordinate,
    ]
    return (max(z_coordinates) + min(z_coordinates)) / 2.0


def get_schedule(name):
    with open(
        package_directory + "/data/schedules/" + name + ".sch", "r", encoding="utf-8"
    ) as file2:
        schedule_str = file2.read()
    return schedule_str
