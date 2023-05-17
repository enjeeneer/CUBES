"""This module adds walls, floors and roof to an IDF and defines the zones"""
from typing import Tuple

from geomeppy import IDF
from cubes.construct.buildingconfig import BuildingConfig
from cubes.construct.buildingconfig_options import Zoning
from cubes.construct.utilities import rotation_changes_north_direction


residential_bedroom_area_ratio = 0.3


def add_geometry_and_zones(idf: IDF, building_config: BuildingConfig):
    if building_config.zoning == Zoning.RESIDENTIAL_DWELLING:
        # work out where the zone boundary is
        total_floor_area = (
            building_config.l_wall_x
            * building_config.l_wall_y
            * building_config.n_storey
        )
        storey_floor_area = building_config.l_wall_x * building_config.l_wall_y
        storey_split_ratio = 0
        storey_split_in_x = building_config.l_wall_x > building_config.l_wall_y

        bedroom_to_place = residential_bedroom_area_ratio * total_floor_area
        for s in range(building_config.n_storey - 1, -1, -1):
            if bedroom_to_place > storey_floor_area:
                bedroom_to_place = bedroom_to_place - storey_floor_area
            else:
                zone_split_storey = s
                storey_split_ratio = bedroom_to_place / storey_floor_area

        north_flip = rotation_changes_north_direction(building_config.rotation)
        if not north_flip:
            storey_split_ratio_flip = (
                1 - storey_split_ratio
            )  # the bedroom zone has the larger coordinate
            front_zone = "Living" if not north_flip else "Bedroom"
            back_zone = "Living" if north_flip else "Bedroom"

        # add the walls to the idf
        for s in range(building_config.n_storey):
            storey_level = (
                building_config.distance_to_ground + s * building_config.h_storey
            )
            if s != zone_split_storey:
                zone = "Living" if s < zone_split_storey else "Bedroom"

                idf = add_external_wall(
                    idf,
                    s + 1,
                    "North",
                    building_config.l_wall_x,
                    (
                        building_config.l_wall_x,
                        building_config.l_wall_y,
                        storey_level + building_config.h_storey,
                    ),
                    building_config.h_storey,
                    zone,
                    building_config.distance_to_neighbour[0] == 0,
                )
                idf = add_external_wall(
                    idf,
                    s + 1,
                    "East",
                    building_config.l_wall_y,
                    (
                        building_config.l_wall_x,
                        0,
                        storey_level + building_config.h_storey,
                    ),
                    building_config.h_storey,
                    zone,
                    building_config.distance_to_neighbour[1] == 0,
                )
                idf = add_external_wall(
                    idf,
                    s + 1,
                    "South",
                    building_config.l_wall_x,
                    (
                        0,
                        0,
                        storey_level + building_config.h_storey,
                    ),
                    building_config.h_storey,
                    zone,
                    building_config.distance_to_neighbour[2] == 0,
                )
                idf = add_external_wall(
                    idf,
                    s + 1,
                    "West",
                    building_config.l_wall_y,
                    (
                        0,
                        building_config.l_wall_y,
                        storey_level + building_config.h_storey,
                    ),
                    building_config.h_storey,
                    zone,
                    building_config.distance_to_neighbour[3] == 0,
                )
            else:

                if storey_split_in_x:
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "North",
                        building_config.l_wall_x * storey_split_ratio_flip,
                        (
                            building_config.l_wall_x * storey_split_ratio_flip,
                            building_config.l_wall_y,
                            storey_level + building_config.h_storey,
                        ),
                        building_config.h_storey,
                        front_zone,
                        building_config.distance_to_neighbour[0] == 0,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "South",
                        building_config.l_wall_x * storey_split_ratio_flip,
                        (
                            0,
                            0,
                            storey_level + building_config.h_storey,
                        ),
                        building_config.h_storey,
                        front_zone,
                        building_config.distance_to_neighbour[2] == 0,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "West",
                        building_config.l_wall_y,
                        (
                            0,
                            building_config.l_wall_y,
                            storey_level + building_config.h_storey,
                        ),
                        building_config.h_storey,
                        front_zone,
                        building_config.distance_to_neighbour[3] == 0,
                    )
                    idf = add_internal_wall(
                        idf,
                        s + 1,
                        "East",
                        building_config.l_wall_y,
                        (
                            building_config.l_wall_x * storey_split_ratio_flip,
                            0,
                            storey_level + building_config.h_storey,
                        ),
                        building_config.h_storey,
                        front_zone,
                        back_zone,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "North",
                        building_config.l_wall_x * (1 - storey_split_ratio_flip),
                        (
                            building_config.l_wall_x,
                            building_config.l_wall_y,
                            storey_level + building_config.h_storey,
                        ),
                        building_config.h_storey,
                        back_zone,
                        building_config.distance_to_neighbour[0] == 0,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "South",
                        building_config.l_wall_x * (1 - storey_split_ratio_flip),
                        (
                            building_config.l_wall_x * storey_split_ratio_flip,
                            0,
                            storey_level + building_config.h_storey,
                        ),
                        building_config.h_storey,
                        back_zone,
                        building_config.distance_to_neighbour[2] == 0,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "East",
                        building_config.l_wall_y,
                        (
                            building_config.l_wall_x,
                            0,
                            storey_level + building_config.h_storey,
                        ),
                        building_config.h_storey,
                        back_zone,
                        building_config.distance_to_neighbour[1] == 0,
                    )

                else:
                    idf = add_internal_wall(
                        idf,
                        s + 1,
                        "North",
                        building_config.l_wall_x,
                        (
                            building_config.l_wall_x,
                            building_config.l_wall_y * storey_split_ratio_flip,
                            storey_level + building_config.h_storey,
                        ),
                        building_config.h_storey,
                        front_zone,
                        back_zone,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "North",
                        building_config.l_wall_x,
                        (
                            building_config.l_wall_x,
                            building_config.l_wall_y,
                            storey_level + building_config.h_storey,
                        ),
                        building_config.h_storey,
                        back_zone,
                        building_config.distance_to_neighbour[0] == 0,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "East",
                        building_config.l_wall_y * (1 - storey_split_ratio_flip),
                        (
                            building_config.l_wall_x,
                            building_config.l_wall_y * storey_split_ratio_flip,
                            storey_level + building_config.h_storey,
                        ),
                        building_config.h_storey,
                        back_zone,
                        building_config.distance_to_neighbour[1] == 0,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "East",
                        building_config.l_wall_y * storey_split_ratio_flip,
                        (
                            building_config.l_wall_x,
                            0,
                            storey_level + building_config.h_storey,
                        ),
                        building_config.h_storey,
                        front_zone,
                        building_config.distance_to_neighbour[1] == 0,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "South",
                        building_config.l_wall_x,
                        (
                            0,
                            0,
                            storey_level + building_config.h_storey,
                        ),
                        building_config.h_storey,
                        front_zone,
                        building_config.distance_to_neighbour[2] == 0,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "West",
                        building_config.l_wall_y * (1 - storey_split_ratio_flip),
                        (
                            0,
                            building_config.l_wall_y,
                            storey_level + building_config.h_storey,
                        ),
                        building_config.h_storey,
                        back_zone,
                        building_config.distance_to_neighbour[3] == 0,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "West",
                        building_config.l_wall_y * storey_split_ratio_flip,
                        (
                            0,
                            building_config.l_wall_y * storey_split_ratio_flip,
                            storey_level + building_config.h_storey,
                        ),
                        building_config.h_storey,
                        front_zone,
                        building_config.distance_to_neighbour[3] == 0,
                    )

        # add ground floor to the idf
        if zone_split_storey != 0:
            idf = add_floor(
                idf,
                1,
                0,
                building_config.l_wall_x,
                0,
                building_config.l_wall_y,
                building_config.distance_to_ground,
                "Living",
                "Ground",
            )
        else:
            if storey_split_in_x:
                idf = add_floor(
                    idf,
                    1,
                    0,
                    building_config.l_wall_x * storey_split_ratio_flip,
                    0,
                    building_config.l_wall_y,
                    building_config.distance_to_ground,
                    front_zone,
                    "Ground",
                )
                idf = add_floor(
                    idf,
                    1,
                    building_config.l_wall_x * storey_split_ratio_flip,
                    building_config.l_wall_x,
                    0,
                    building_config.l_wall_y,
                    building_config.distance_to_ground,
                    back_zone,
                    "Ground",
                )
            else:
                idf = add_floor(
                    idf,
                    1,
                    0,
                    building_config.l_wall_x,
                    0,
                    building_config.l_wall_y * storey_split_ratio_flip,
                    building_config.distance_to_ground,
                    front_zone,
                    "Ground",
                )
                idf = add_floor(
                    idf,
                    1,
                    0,
                    building_config.l_wall_x,
                    building_config.l_wall_y * storey_split_ratio_flip,
                    building_config.l_wall_y,
                    building_config.distance_to_ground,
                    back_zone,
                    "Ground",
                )

        # add internal floors
        for s in range(1, building_config.n_storey):
            if s < zone_split_storey:
                idf = add_floor(
                    idf,
                    s + 1,
                    0,
                    building_config.l_wall_x,
                    0,
                    building_config.l_wall_y,
                    building_config.distance_to_ground,
                    "Living",
                    "Living",
                )
            elif s == zone_split_storey:
                if storey_split_in_x:
                    idf = add_floor(
                        idf,
                        s + 1,
                        0,
                        building_config.l_wall_x * storey_split_ratio_flip,
                        0,
                        building_config.l_wall_y,
                        building_config.distance_to_ground,
                        front_zone,
                        "Living",
                    )
                    idf = add_floor(
                        idf,
                        s + 1,
                        building_config.l_wall_x * storey_split_ratio_flip,
                        building_config.l_wall_x,
                        0,
                        building_config.l_wall_y,
                        building_config.distance_to_ground,
                        back_zone,
                        "Living",
                    )
                else:
                    idf = add_floor(
                        idf,
                        s + 1,
                        0,
                        building_config.l_wall_x,
                        0,
                        building_config.l_wall_y * storey_split_ratio_flip,
                        building_config.distance_to_ground,
                        front_zone,
                        "Living",
                    )
                    idf = add_floor(
                        idf,
                        s + 1,
                        0,
                        building_config.l_wall_x,
                        building_config.l_wall_y * storey_split_ratio_flip,
                        building_config.l_wall_y,
                        building_config.distance_to_ground,
                        back_zone,
                        "Living",
                    )
            elif s - 1 == zone_split_storey:
                if storey_split_in_x:
                    idf = add_floor(
                        idf,
                        s + 1,
                        0,
                        building_config.l_wall_x * storey_split_ratio_flip,
                        0,
                        building_config.l_wall_y,
                        building_config.distance_to_ground,
                        "Bedroom",
                        front_zone,
                    )
                    idf = add_floor(
                        idf,
                        s + 1,
                        building_config.l_wall_x * storey_split_ratio_flip,
                        building_config.l_wall_x,
                        0,
                        building_config.l_wall_y,
                        building_config.distance_to_ground,
                        "Bedroom",
                        back_zone,
                    )
                else:
                    idf = add_floor(
                        idf,
                        s + 1,
                        0,
                        building_config.l_wall_x,
                        0,
                        building_config.l_wall_y * storey_split_ratio_flip,
                        building_config.distance_to_ground,
                        "Bedroom",
                        front_zone,
                    )
                    idf = add_floor(
                        idf,
                        s + 1,
                        0,
                        building_config.l_wall_x,
                        building_config.l_wall_y * storey_split_ratio_flip,
                        building_config.l_wall_y,
                        building_config.distance_to_ground,
                        "Bedroom",
                        back_zone,
                    )

    return idf


def add_internal_wall(
    idf: IDF,
    storey: int,
    direction: str,  # outside normal
    l_wall: float,
    upper_left_corner: Tuple[float, float, float],  # this is either x or y
    storey_height: float,
    inner_zone: str,
    outer_zone: bool,
):
    coords = get_wall_coordinates(l_wall, storey_height, upper_left_corner, direction)
    idf.newidfobject(
        "BuildingSurface:Detailed",
        Name=(
            "Storey " + str(storey) + " Internal Wall " + inner_zone + "-" + outer_zone
        ),
        Surface_Type="Wall",
        Construction_Name="InternalWall",
        Zone_Name=inner_zone,
        Outside_Boundary_Condition="Zone",
        Outside_Boundary_Condition_Object=outer_zone,
        View_Factor_to_Ground=0.5,
        Sun_Exposure="NoSun",
        Wind_Exposure="NoWind",
        Number_of_Vertices=4,
        Vertex_1_X_Coordinate=coords["X1"],
        Vertex_1_Y_Coordinate=coords["Y1"],
        Vertex_1_Z_Coordinate=coords["Z1"],
        Vertex_2_X_Coordinate=coords["X2"],
        Vertex_2_Y_Coordinate=coords["Y2"],
        Vertex_2_Z_Coordinate=coords["Z2"],
        Vertex_3_X_Coordinate=coords["X3"],
        Vertex_3_Y_Coordinate=coords["Y3"],
        Vertex_3_Z_Coordinate=coords["Z3"],
        Vertex_4_X_Coordinate=coords["X4"],
        Vertex_4_Y_Coordinate=coords["Y4"],
        Vertex_4_Z_Coordinate=coords["Z4"],
    )

    return idf


def add_external_wall(
    idf: IDF,
    storey: int,
    direction: str,
    l_wall: float,
    upper_left_corner: Tuple[float, float, float],  # this is either x or y
    storey_height: float,
    zone: str,
    adiabatic: bool,
):
    coords = get_wall_coordinates(l_wall, storey_height, upper_left_corner, direction)
    idf.newidfobject(
        "BuildingSurface:Detailed",
        Name="Storey " + str(storey) + " " + direction + " Wall " + zone,
        Surface_Type="Wall",
        Construction_Name="Wall",
        View_Factor_to_Ground=0.5,
        Number_of_Vertices=4,
        Vertex_1_X_Coordinate=coords["X1"],
        Vertex_1_Y_Coordinate=coords["Y1"],
        Vertex_1_Z_Coordinate=coords["Z1"],
        Vertex_2_X_Coordinate=coords["X2"],
        Vertex_2_Y_Coordinate=coords["Y2"],
        Vertex_2_Z_Coordinate=coords["Z2"],
        Vertex_3_X_Coordinate=coords["X3"],
        Vertex_3_Y_Coordinate=coords["Y3"],
        Vertex_3_Z_Coordinate=coords["Z3"],
        Vertex_4_X_Coordinate=coords["X4"],
        Vertex_4_Y_Coordinate=coords["Y4"],
        Vertex_4_Z_Coordinate=coords["Z4"],
    )
    wall = idf.idfobjects["BuildingSurface:Detailed".upper()][-1]

    if adiabatic:
        wall.Outside_Boundary_Condition = "Adiabatic"
        wall.Sun_Exposure = "NoSun"
        wall.Wind_Exposure = "NoWind"
    else:
        wall.Outside_Boundary_Condition = "Outdoors"
        wall.Sun_Exposure = "SunExposed"
        wall.Wind_Exposure = "WindExposed"

    return idf


def add_floor(
    idf: IDF,
    storey: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    distance_from_ground: float,
    inner_zone: str,
    outer_zone: str,
):
    coords = get_floor_xy_coordinates(xmin, xmax, ymin, ymax)
    idf.newidfobject(
        "BuildingSurface:Detailed",
        Name="Storey " + str(storey) + " " + " Floor " + inner_zone + "-" + outer_zone,
        Surface_Type="Floor",
        Construction_Name="Floor" if not outer_zone == "Ground" else "GroundFloor",
        View_Factor_to_Ground=1.0,
        Number_of_Vertices=4,
        Vertex_1_X_Coordinate=coords["X1"],
        Vertex_1_Y_Coordinate=coords["Y1"],
        Vertex_1_Z_Coordinate=distance_from_ground,
        Vertex_2_X_Coordinate=coords["X2"],
        Vertex_2_Y_Coordinate=coords["Y2"],
        Vertex_2_Z_Coordinate=distance_from_ground,
        Vertex_3_X_Coordinate=coords["X3"],
        Vertex_3_Y_Coordinate=coords["Y3"],
        Vertex_3_Z_Coordinate=distance_from_ground,
        Vertex_4_X_Coordinate=coords["X4"],
        Vertex_4_Y_Coordinate=coords["Y4"],
        Vertex_4_Z_Coordinate=distance_from_ground,
        Sun_Exposure="NoSun",
        Wind_Exposure="NoWind",
    )
    floor = idf.idfobjects["BuildingSurface:Detailed".upper()][-1]

    if outer_zone == "Ground" and distance_from_ground > 0:
        floor.Outside_Boundary_Condition = "Adiabatic"
    elif outer_zone == "Ground" and distance_from_ground == 0:
        floor.Outside_Boundary_Condition = "Ground"
    elif outer_zone == inner_zone:
        floor.Outside_Boundary_Condition = "Adiabatic"
    else:
        floor.Outside_Boundary_Condition = "Zone"
        floor.Outside_Boundary_Condition_Object = outer_zone

    return idf


def get_floor_xy_coordinates(xmin: float, xmax: float, ymin: float, ymax: float):
    # below is outside
    return {
        "X1": xmin,
        "Y1": ymin,
        "X2": xmin,
        "Y2": ymax,
        "X3": xmax,
        "Y3": ymax,
        "X4": xmax,
        "Y4": ymin,
    }


def get_wall_coordinates(
    length: float,
    height: float,
    upper_left_corner: Tuple[float, float, float],
    direction: str,
):
    coords = {
        "X1": upper_left_corner[0],
        "Y1": upper_left_corner[1],
        "Z1": upper_left_corner[2],
    }
    if direction == "North":
        coords["X2"] = coords["X1"]
        coords["Y2"] = coords["Y1"]
        coords["Z2"] = coords["Z1"] - height
        coords["X3"] = coords["X2"] - length
        coords["Y3"] = coords["Y2"]
        coords["Z3"] = coords["Z2"]
        coords["X4"] = coords["X3"]
        coords["Y4"] = coords["Y3"]
        coords["Z4"] = coords["Z3"] + height
    elif direction == "South":
        coords["X2"] = coords["X1"]
        coords["Y2"] = coords["Y1"]
        coords["Z2"] = coords["Z1"] - height
        coords["X3"] = coords["X2"] + length
        coords["Y3"] = coords["Y2"]
        coords["Z3"] = coords["Z2"]
        coords["X4"] = coords["X3"]
        coords["Y4"] = coords["Y3"]
        coords["Z4"] = coords["Z3"] + height
    elif direction == "East":
        coords["X2"] = coords["X1"]
        coords["Y2"] = coords["Y1"]
        coords["Z2"] = coords["Z1"] - height
        coords["X3"] = coords["X2"]
        coords["Y3"] = coords["Y2"] + length
        coords["Z3"] = coords["Z2"]
        coords["X4"] = coords["X3"]
        coords["Y4"] = coords["Y3"]
        coords["Z4"] = coords["Z3"] + height
    elif direction == "West":
        coords["X2"] = coords["X1"]
        coords["Y2"] = coords["Y1"]
        coords["Z2"] = coords["Z1"] - height
        coords["X3"] = coords["X2"]
        coords["Y3"] = coords["Y2"] - length
        coords["Z3"] = coords["Z2"]
        coords["X4"] = coords["X3"]
        coords["Y4"] = coords["Y3"]
        coords["Z4"] = coords["Z3"] + height

    return coords
