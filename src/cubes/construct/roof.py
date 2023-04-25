"""This module holds functions that define the roof geometry and add it to an idf"""

from cubes.construct.buildingconfig import BuildingConfig
from cubes.package.weather import get_weather_file_info
from geomeppy import IDF
import numpy as np


def get_saddleback_roof_coordinates(building_config: BuildingConfig):
    """Determines roof coordinates based on a saddleback roof template

    Returns:
        roof_coords list of lists: roof_coords are the coordinates of each point
                                    of the roof, there are two roof segments, each
                                    with four points, with each point having an
                                    (x,y,z) coordinate
                                    geometry rules: starting upper left corner,
                                    counterclockwise
    """

    roof_coords = [
        {
            "X1": 0,
            "Y1": building_config.l_wall_y / 2,
            "Z1": building_config.n_storey * building_config.h_storey
            + building_config.h_roof,
            "X2": 0,
            "Y2": 0,
            "Z2": building_config.n_storey * building_config.h_storey,
            "X3": building_config.l_wall_x,
            "Y3": 0,
            "Z3": building_config.n_storey * building_config.h_storey,
            "X4": building_config.l_wall_x,
            "Y4": building_config.l_wall_y / 2,
            "Z4": building_config.n_storey * building_config.h_storey
            + building_config.h_roof,
        },
        {
            "X1": building_config.l_wall_x,
            "Y1": building_config.l_wall_y / 2,
            "Z1": building_config.n_storey * building_config.h_storey
            + building_config.h_roof,
            "X2": building_config.l_wall_x,
            "Y2": building_config.l_wall_y,
            "Z2": building_config.n_storey * building_config.h_storey,
            "X3": 0,
            "Y3": building_config.l_wall_y,
            "Z3": building_config.n_storey * building_config.h_storey,
            "X4": 0,
            "Y4": building_config.l_wall_y / 2,
            "Z4": building_config.n_storey * building_config.h_storey
            + building_config.h_roof,
        },
    ]

    return roof_coords


def get_saddleback_roof_wall_coordinates(building_config: BuildingConfig):
    """Determines roof-level wall coordinates based on an idealised pitched roof

    Returns:
        wall_coords list of lists: wall_coords are the coordinates of each point of
                                    the wall, there are two wall segments, each with
                                    four points, with each point having an
                                    (x,y,z) coordinate
    """

    wall_coords = [
        {
            "X1": 0,
            "Y1": 0,
            "Z1": building_config.n_storey * building_config.h_storey,
            "X2": 0,
            "Y2": building_config.l_wall_y,
            "Z2": building_config.n_storey * building_config.h_storey,
            "X3": 0,
            "Y3": building_config.l_wall_y / 2,
            "Z3": building_config.n_storey * building_config.h_storey
            + building_config.h_roof,
            "X4": 0,
            "Y4": building_config.l_wall_y / 2,
            "Z4": building_config.n_storey * building_config.h_storey
            + building_config.h_roof,
        },
        {
            "X1": building_config.l_wall_x,
            "Y1": 0,
            "Z1": building_config.n_storey * building_config.h_storey,
            "X2": building_config.l_wall_x,
            "Y2": building_config.l_wall_y,
            "Z2": building_config.n_storey * building_config.h_storey,
            "X3": building_config.l_wall_x,
            "Y3": building_config.l_wall_y / 2,
            "Z3": building_config.n_storey * building_config.h_storey
            + building_config.h_roof,
            "X4": building_config.l_wall_x,
            "Y4": building_config.l_wall_y / 2,
            "Z4": building_config.n_storey * building_config.h_storey
            + building_config.h_roof,
        },
    ]

    return wall_coords


def rotation_changes_primary_pv_direction(rotation):
    """check if the building rotation is such that the south facing side is changed"""
    if np.cos(rotation / 180 * np.pi) > 0:
        return False
    else:
        return True


def get_roof_pitch(building_config: BuildingConfig):
    """returns roof pitch in radians"""
    if building_config.roof_type == "flat":
        return 0
    else:
        return np.arctan(building_config.h_roof / (building_config.l_wall_y / 2))


def get_pv_surface_coordinates(building_config: BuildingConfig):
    """for a flat roof return an optimally angled PV surface;
    for a saddleback roof returns the side of the roof facing the equator
    and the surface facing away from the equator"""
    latitude = get_weather_file_info(building_config)["Latitude"]

    if building_config.roof_type == "flat":
        if (
            latitude > 0
            and not rotation_changes_primary_pv_direction(building_config.rotation)
        ) or (
            latitude < 0
            and rotation_changes_primary_pv_direction(building_config.rotation)
        ):
            coords = [
                {
                    "X1": 0,
                    "Y1": building_config.l_wall_y
                    * building_config.pv_roof_area_ratio_primary,
                    "Z1": building_config.n_storey * building_config.h_storey
                    + building_config.l_wall_y
                    * building_config.pv_roof_area_ratio_primary
                    * np.tan(latitude / 180 * np.pi),
                    "X2": 0,
                    "Y2": 0,
                    "Z2": building_config.n_storey * building_config.h_storey,
                    "X3": building_config.l_wall_x,
                    "Y3": 0,
                    "Z3": building_config.n_storey * building_config.h_storey,
                    "X4": building_config.l_wall_x,
                    "Y4": building_config.l_wall_y
                    * building_config.pv_roof_area_ratio_primary,
                    "Z4": building_config.n_storey * building_config.h_storey
                    + building_config.l_wall_y
                    * building_config.pv_roof_area_ratio_primary
                    * np.tan(latitude / 180 * np.pi),
                },
                {},
            ]
        else:
            coords = [
                {
                    "X1": building_config.l_wall_x,
                    "Y1": building_config.l_wall_y
                    * (1 - building_config.pv_roof_area_ratio_primary),
                    "Z1": building_config.n_storey * building_config.h_storey
                    + building_config.l_wall_y
                    * building_config.pv_roof_area_ratio_primary
                    * np.tan(latitude / 180 * np.pi),
                    "X2": building_config.l_wall_x,
                    "Y2": building_config.l_wall_y,
                    "Z2": building_config.n_storey * building_config.h_storey,
                    "X3": 0,
                    "Y3": building_config.l_wall_y,
                    "Z3": building_config.n_storey * building_config.h_storey,
                    "X4": 0,
                    "Y4": building_config.l_wall_y
                    * (1 - building_config.pv_roof_area_ratio_primary),
                    "Z4": building_config.n_storey * building_config.h_storey
                    + building_config.l_wall_y
                    * building_config.pv_roof_area_ratio_primary
                    * np.tan(latitude / 180 * np.pi),
                },
                {},
            ]

    else:
        roof_pitch = get_roof_pitch(building_config)

        if (
            latitude > 0
            and not rotation_changes_primary_pv_direction(building_config.rotation)
        ) or (
            latitude < 0
            and rotation_changes_primary_pv_direction(building_config.rotation)
        ):

            if building_config.pv_roof_area_ratio_primary > 0:
                coords1 = get_saddleback_roof_coordinates(building_config)[0]
                coords1["Y2"] = (
                    building_config.l_wall_y
                    / 2
                    * (1 - building_config.pv_roof_area_ratio_primary)
                )
                coords1["Z2"] = (
                    building_config.n_storey * building_config.h_storey
                    + np.tan(roof_pitch) * coords1["Y2"]
                )

                coords1["Y3"] = coords1["Y2"]
                coords1["Z3"] = coords1["Z2"]
            else:
                coords1 = {}

            if building_config.pv_roof_area_ratio_secondary > 0:
                coords2 = get_saddleback_roof_coordinates(building_config)[1]

                coords2["Y2"] = (
                    building_config.l_wall_y / 2
                    + building_config.l_wall_y
                    / 2
                    * building_config.pv_roof_area_ratio_secondary
                )
                coords2[
                    "Z2"
                ] = building_config.n_storey * building_config.h_storey + np.tan(
                    roof_pitch
                ) * (
                    building_config.l_wall_y - coords2["Y2"]
                )

                coords2["Y3"] = coords2["Y2"]
                coords2["Z3"] = coords2["Z2"]

            else:
                coords2 = {}

            coords = [coords1, coords2]
        else:
            if building_config.pv_roof_area_ratio_secondary > 0:
                coords1 = get_saddleback_roof_coordinates(building_config)[0]

                coords1["Y2"] = (
                    building_config.l_wall_y
                    / 2
                    * (1 - building_config.pv_roof_area_ratio_secondary)
                )
                coords1["Z2"] = (
                    building_config.n_storey * building_config.h_storey
                    + np.tan(roof_pitch) * coords1["Y2"]
                )

                coords1["Y3"] = coords1["Y2"]
                coords1["Z3"] = coords1["Z2"]
            else:
                coords1 = {}

            if building_config.pv_roof_area_ratio_primary > 0:
                coords2 = get_saddleback_roof_coordinates(building_config)[1]

                coords2["Y2"] = (
                    building_config.l_wall_y / 2
                    + building_config.l_wall_y
                    / 2
                    * building_config.pv_roof_area_ratio_primary
                )
                coords2[
                    "Z2"
                ] = building_config.n_storey * building_config.h_storey + np.tan(
                    roof_pitch
                ) * (
                    building_config.l_wall_y - coords2["Y2"]
                )

                coords2["Y3"] = coords2["Y2"]
                coords2["Z3"] = coords2["Z2"]
            else:
                coords2 = {}

            coords = [coords2, coords1]

    return coords


def add_roof(idf: IDF, building_config: BuildingConfig):
    """Initially checks if the roof is flat, if it is then the original
    geomeppy flat roof created by the idf.add_block method works. If not then the
    method gets roof height, coordinates of roof and roof space walls, then creates
    a new roof and wall elements in e+ and assigns coordinates of the new
    elements"""

    if building_config.roof_type != "flat":

        for index, surface in enumerate(idf.idfobjects["BUILDINGSURFACE:DETAILED"]):

            if surface.Surface_Type == "roof":

                idf.removeidfobject(idf.idfobjects["BUILDINGSURFACE:DETAILED"][index])

                # search for zone name of last storey
                last_storey_zone_name = "UNKNOWN"
                for zone in idf.idfobjects["ZONE"]:
                    if str(building_config.n_storey - 1) in zone.Name:
                        last_storey_zone_name = zone.Name

                ceiling_name = "storey " + str(building_config.n_storey) + " ceiling"
                idf.newidfobject(
                    "BUILDINGSURFACE:DETAILED",
                    Name=ceiling_name,
                    Surface_Type="ceiling",
                    Zone_Name=last_storey_zone_name,
                    Vertex_1_Xcoordinate=surface.Vertex_1_Xcoordinate,
                    Vertex_1_Ycoordinate=surface.Vertex_1_Ycoordinate,
                    Vertex_1_Zcoordinate=surface.Vertex_1_Zcoordinate,
                    Vertex_2_Xcoordinate=surface.Vertex_2_Xcoordinate,
                    Vertex_2_Ycoordinate=surface.Vertex_2_Ycoordinate,
                    Vertex_2_Zcoordinate=surface.Vertex_2_Zcoordinate,
                    Vertex_3_Xcoordinate=surface.Vertex_3_Xcoordinate,
                    Vertex_3_Ycoordinate=surface.Vertex_3_Ycoordinate,
                    Vertex_3_Zcoordinate=surface.Vertex_3_Zcoordinate,
                    Vertex_4_Xcoordinate=surface.Vertex_4_Xcoordinate,
                    Vertex_4_Ycoordinate=surface.Vertex_4_Ycoordinate,
                    Vertex_4_Zcoordinate=surface.Vertex_4_Zcoordinate,
                    Outside_Boundary_Condition="Surface",
                    Outside_Boundary_Condition_Object="attic floor",
                    Sun_Exposure="NoSun",
                    Wind_Exposure="NoWind",
                )

                idf.newidfobject(
                    "BUILDINGSURFACE:DETAILED",
                    Name="attic floor",
                    Surface_Type="floor",
                    Zone_Name="ROOF SPACE",
                    Vertex_1_Xcoordinate=surface.Vertex_1_Xcoordinate,
                    Vertex_1_Ycoordinate=surface.Vertex_1_Ycoordinate,
                    Vertex_1_Zcoordinate=surface.Vertex_1_Zcoordinate,
                    Vertex_2_Xcoordinate=surface.Vertex_4_Xcoordinate,
                    Vertex_2_Ycoordinate=surface.Vertex_4_Ycoordinate,
                    Vertex_2_Zcoordinate=surface.Vertex_4_Zcoordinate,
                    Vertex_3_Xcoordinate=surface.Vertex_3_Xcoordinate,
                    Vertex_3_Ycoordinate=surface.Vertex_3_Ycoordinate,
                    Vertex_3_Zcoordinate=surface.Vertex_3_Zcoordinate,
                    Vertex_4_Xcoordinate=surface.Vertex_2_Xcoordinate,
                    Vertex_4_Ycoordinate=surface.Vertex_2_Ycoordinate,
                    Vertex_4_Zcoordinate=surface.Vertex_2_Zcoordinate,
                    Outside_Boundary_Condition="Surface",
                    Outside_Boundary_Condition_Object=ceiling_name,
                    Sun_Exposure="NoSun",
                    Wind_Exposure="NoWind",
                )

        roof_coords = get_saddleback_roof_coordinates(building_config)

        wall_coords = get_saddleback_roof_wall_coordinates(building_config)

        idf.newidfobject(
            "ZONE",
            Name="ROOF SPACE",
        )

        # May want to change nomenclature on naming new elements
        # Currently N_X means that there are X of the new elements,
        # and N designates what element you are adding

        idf.newidfobject(
            "BUILDINGSURFACE:DETAILED",
            Name="roof_1_2",
            Construction_Name="ROOF-Construction",
            Surface_Type="ROOF",
            Zone_Name="ROOF SPACE",
            Outside_Boundary_Condition="Outdoors",
        )

        idf.newidfobject(
            "BUILDINGSURFACE:DETAILED",
            Name="roof_2_2",
            Construction_Name="ROOF-Construction",
            Surface_Type="ROOF",
            Zone_Name="ROOF SPACE",
            Outside_Boundary_Condition="Outdoors",
        )

        idf.newidfobject(
            "BUILDINGSURFACE:DETAILED",
            Name="wall_1_2",
            Construction_Name="WALL-Construction",
            Surface_Type="WALL",
            Zone_Name="ROOF SPACE",
            Outside_Boundary_Condition="Outdoors",
            Number_of_Vertices=3,
        )

        idf.newidfobject(
            "BUILDINGSURFACE:DETAILED",
            Name="wall_2_2",
            Construction_Name="WALL-Construction",
            Surface_Type="WALL",
            Zone_Name="ROOF SPACE",
            Outside_Boundary_Condition="Outdoors",
            Number_of_Vertices=3,
        )
        for index, roof in enumerate(idf.getsurfaces("ROOF")):
            roof.Vertex_1_Xcoordinate = roof_coords[index]["X1"]
            roof.Vertex_1_Ycoordinate = roof_coords[index]["Y1"]
            roof.Vertex_1_Zcoordinate = roof_coords[index]["Z1"]
            roof.Vertex_2_Xcoordinate = roof_coords[index]["X2"]
            roof.Vertex_2_Ycoordinate = roof_coords[index]["Y2"]
            roof.Vertex_2_Zcoordinate = roof_coords[index]["Z2"]
            roof.Vertex_3_Xcoordinate = roof_coords[index]["X3"]
            roof.Vertex_3_Ycoordinate = roof_coords[index]["Y3"]
            roof.Vertex_3_Zcoordinate = roof_coords[index]["Z3"]
            roof.Vertex_4_Xcoordinate = roof_coords[index]["X4"]
            roof.Vertex_4_Ycoordinate = roof_coords[index]["Y4"]
            roof.Vertex_4_Zcoordinate = roof_coords[index]["Z4"]

        count = 0
        for index, wall in enumerate(idf.getsurfaces("WALL")):
            if idf.getsurfaces("WALL")[index].Zone_Name == "ROOF SPACE":
                wall.Vertex_1_Xcoordinate = wall_coords[count]["X1"]
                wall.Vertex_1_Ycoordinate = wall_coords[count]["Y1"]
                wall.Vertex_1_Zcoordinate = wall_coords[count]["Z1"]
                wall.Vertex_2_Xcoordinate = wall_coords[count]["X2"]
                wall.Vertex_2_Ycoordinate = wall_coords[count]["Y2"]
                wall.Vertex_2_Zcoordinate = wall_coords[count]["Z2"]
                wall.Vertex_3_Xcoordinate = wall_coords[count]["X3"]
                wall.Vertex_3_Ycoordinate = wall_coords[count]["Y3"]
                wall.Vertex_3_Zcoordinate = wall_coords[count]["Z3"]

                count = count + 1

    return idf
