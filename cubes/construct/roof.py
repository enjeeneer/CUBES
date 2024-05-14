"""This module holds functions that define the roof geometry and add it to an idf"""

from cubes.construct.buildingconfig import BuildingConfig
from cubes.construct.utilities import rotation_changes_north_direction, get_surface_area
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

    if building_config.roof_ridge_along_x:
        roof_coords = [
            {
                "X1": 0,
                "Y1": building_config.length_wall_y / 2,
                "Z1": building_config.number_of_stories * building_config.storey_height
                + building_config.roof_height,
                "X2": 0,
                "Y2": 0,
                "Z2": building_config.number_of_stories * building_config.storey_height,
                "X3": building_config.length_wall_x,
                "Y3": 0,
                "Z3": building_config.number_of_stories * building_config.storey_height,
                "X4": building_config.length_wall_x,
                "Y4": building_config.length_wall_y / 2,
                "Z4": building_config.number_of_stories * building_config.storey_height
                + building_config.roof_height,
            },
            {
                "X1": building_config.length_wall_x,
                "Y1": building_config.length_wall_y / 2,
                "Z1": building_config.number_of_stories * building_config.storey_height
                + building_config.roof_height,
                "X2": building_config.length_wall_x,
                "Y2": building_config.length_wall_y,
                "Z2": building_config.number_of_stories * building_config.storey_height,
                "X3": 0,
                "Y3": building_config.length_wall_y,
                "Z3": building_config.number_of_stories * building_config.storey_height,
                "X4": 0,
                "Y4": building_config.length_wall_y / 2,
                "Z4": building_config.number_of_stories * building_config.storey_height
                + building_config.roof_height,
            },
        ]
    else:
        roof_coords = [
            {
                "X1": building_config.length_wall_x / 2,
                "Y1": 0,
                "Z1": building_config.number_of_stories * building_config.storey_height
                + building_config.roof_height,
                "X2": building_config.length_wall_x,
                "Y2": 0,
                "Z2": building_config.number_of_stories * building_config.storey_height,
                "X3": building_config.length_wall_x,
                "Y3": building_config.length_wall_y,
                "Z3": building_config.number_of_stories * building_config.storey_height,
                "X4": building_config.length_wall_x / 2,
                "Y4": building_config.length_wall_y,
                "Z4": building_config.number_of_stories * building_config.storey_height
                + building_config.roof_height,
            },
            {
                "X1": building_config.length_wall_x / 2,
                "Y1": building_config.length_wall_y,
                "Z1": building_config.number_of_stories * building_config.storey_height
                + building_config.roof_height,
                "X2": 0,
                "Y2": building_config.length_wall_y,
                "Z2": building_config.number_of_stories * building_config.storey_height,
                "X3": 0,
                "Y3": 0,
                "Z3": building_config.number_of_stories * building_config.storey_height,
                "X4": building_config.length_wall_x / 2,
                "Y4": 0,
                "Z4": building_config.number_of_stories * building_config.storey_height
                + building_config.roof_height,
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

    if building_config.roof_ridge_along_x:
        wall_coords = [
            {
                "X1": 0,
                "Y1": building_config.length_wall_y / 2,
                "Z1": building_config.number_of_stories * building_config.storey_height
                + building_config.roof_height,
                "X2": 0,
                "Y2": building_config.length_wall_y,
                "Z2": building_config.number_of_stories * building_config.storey_height,
                "X3": 0,
                "Y3": 0,
                "Z3": building_config.number_of_stories * building_config.storey_height,
            },
            {
                "X1": building_config.length_wall_x,
                "Y1": building_config.length_wall_y / 2,
                "Z1": building_config.number_of_stories * building_config.storey_height
                + building_config.roof_height,
                "X2": building_config.length_wall_x,
                "Y2": 0,
                "Z2": building_config.number_of_stories * building_config.storey_height,
                "X3": building_config.length_wall_x,
                "Y3": building_config.length_wall_y,
                "Z3": building_config.number_of_stories * building_config.storey_height,
            },
        ]
    else:
        wall_coords = [
            {
                "X1": building_config.length_wall_x / 2,
                "Y1": 0,
                "Z1": building_config.number_of_stories * building_config.storey_height
                + building_config.roof_height,
                "X2": 0,
                "Y2": 0,
                "Z2": building_config.number_of_stories * building_config.storey_height,
                "X3": building_config.length_wall_x,
                "Y3": 0,
                "Z3": building_config.number_of_stories * building_config.storey_height,
            },
            {
                "X1": building_config.length_wall_x / 2,
                "Y1": building_config.length_wall_y,
                "Z1": building_config.number_of_stories * building_config.storey_height
                + building_config.roof_height,
                "X2": building_config.length_wall_x,
                "Y2": building_config.length_wall_y,
                "Z2": building_config.number_of_stories * building_config.storey_height,
                "X3": 0,
                "Y3": building_config.length_wall_y,
                "Z3": building_config.number_of_stories * building_config.storey_height,
            },
        ]

    return wall_coords


def get_roof_pitch(building_config: BuildingConfig):
    """returns roof pitch in radians"""
    if building_config.roof_type == "flat":
        return 0
    else:
        return np.arctan(
            building_config.roof_height / (building_config.length_wall_y / 2)
        )


def get_pv_surface_coordinates(building_config: BuildingConfig):
    """for a flat roof return an optimally angled PV surface;
    for a saddleback roof returns the side of the roof facing the equator
    and the surface facing away from the equator"""
    latitude = get_weather_file_info(building_config)["Latitude"]
    pv_distance_from_roof = 0.2

    # TODO add in option for saddleback roof for custom zoning
    if building_config.zoning == "custom":

        z = (
            building_config.number_of_stories
            * building_config.storey_height
            * building_config.pv_roof_area_ratio_primary
            + pv_distance_from_roof
        )

        coords = [
            {
                "X1": 0,
                "Y1": building_config.length_wall_y
                * building_config.pv_roof_area_ratio_primary,
                "Z1": z,
                "X2": 0,
                "Y2": 0,
                "Z2": z,
                "X3": building_config.length_wall_x,
                "Y3": 0,
                "Z3": z,
                "X4": building_config.length_wall_x,
                "Y4": building_config.length_wall_y
                * building_config.pv_roof_area_ratio_primary,
                "Z4": z,
            },
            {},
        ]

        return coords

    if building_config.roof_type == "flat":
        if (
            latitude > 0
            and not rotation_changes_north_direction(building_config.rotation)
        ) or (
            latitude < 0 and rotation_changes_north_direction(building_config.rotation)
        ):
            coords = [
                {
                    "X1": 0,
                    "Y1": building_config.length_wall_y
                    * building_config.pv_roof_area_ratio_primary,
                    "Z1": building_config.number_of_stories
                    * building_config.storey_height
                    + building_config.length_wall_y
                    * building_config.pv_roof_area_ratio_primary
                    * np.tan(latitude / 180 * np.pi)
                    + pv_distance_from_roof,
                    "X2": 0,
                    "Y2": 0,
                    "Z2": building_config.number_of_stories
                    * building_config.storey_height
                    + pv_distance_from_roof,
                    "X3": building_config.length_wall_x,
                    "Y3": 0,
                    "Z3": building_config.number_of_stories
                    * building_config.storey_height
                    + pv_distance_from_roof,
                    "X4": building_config.length_wall_x,
                    "Y4": building_config.length_wall_y
                    * building_config.pv_roof_area_ratio_primary,
                    "Z4": building_config.number_of_stories
                    * building_config.storey_height
                    + building_config.length_wall_y
                    * building_config.pv_roof_area_ratio_primary
                    * np.tan(latitude / 180 * np.pi)
                    + pv_distance_from_roof,
                },
                {},
            ]
        else:
            coords = [
                {
                    "X1": building_config.length_wall_x,
                    "Y1": building_config.length_wall_y
                    * (1 - building_config.pv_roof_area_ratio_primary),
                    "Z1": building_config.number_of_stories
                    * building_config.storey_height
                    + building_config.length_wall_y
                    * building_config.pv_roof_area_ratio_primary
                    * np.tan(latitude / 180 * np.pi)
                    + pv_distance_from_roof,
                    "X2": building_config.length_wall_x,
                    "Y2": building_config.length_wall_y,
                    "Z2": building_config.number_of_stories
                    * building_config.storey_height
                    + pv_distance_from_roof,
                    "X3": 0,
                    "Y3": building_config.length_wall_y,
                    "Z3": building_config.number_of_stories
                    * building_config.storey_height
                    + pv_distance_from_roof,
                    "X4": 0,
                    "Y4": building_config.length_wall_y
                    * (1 - building_config.pv_roof_area_ratio_primary),
                    "Z4": building_config.number_of_stories
                    * building_config.storey_height
                    + building_config.length_wall_y
                    * building_config.pv_roof_area_ratio_primary
                    * np.tan(latitude / 180 * np.pi)
                    + pv_distance_from_roof,
                },
                {},
            ]

    else:

        coords = []

        if (
            latitude > 0
            and not rotation_changes_north_direction(building_config.rotation)
        ) or (
            latitude < 0 and rotation_changes_north_direction(building_config.rotation)
        ):
            roofcoords = (
                get_saddleback_roof_coordinates(building_config)[0],
                get_saddleback_roof_coordinates(building_config)[1],
            )

            if building_config.roof_ridge_along_x:
                wwrs = (
                    building_config.wtw_ratios_loft[2],
                    building_config.wtw_ratios_loft[0],
                )
            else:
                wwrs = (
                    building_config.wtw_ratios_loft[1],
                    building_config.wtw_ratios_loft[3],
                )

        else:
            roofcoords = (
                get_saddleback_roof_coordinates(building_config)[1],
                get_saddleback_roof_coordinates(building_config)[0],
            )

            if building_config.roof_ridge_along_x:
                wwrs = (
                    building_config.wtw_ratios_loft[0],
                    building_config.wtw_ratios_loft[2],
                )
            else:
                wwrs = (
                    building_config.wtw_ratios_loft[3],
                    building_config.wtw_ratios_loft[1],
                )

        pvrs = [
            building_config.pv_roof_area_ratio_primary,
            building_config.pv_roof_area_ratio_secondary,
        ]

        for i_side in range(2):
            if pvrs[i_side] == 0:
                continue

            coords1 = roofcoords[i_side]
            pvr = pvrs[i_side]
            wwr = wwrs[i_side]

            p1 = np.array([coords1["X1"], coords1["Y1"], coords1["Z1"]])
            p2 = np.array([coords1["X2"], coords1["Y2"], coords1["Z2"]])
            p3 = np.array([coords1["X3"], coords1["Y3"], coords1["Z3"]])
            p4 = np.array([coords1["X4"], coords1["Y4"], coords1["Z4"]])

            w1 = p1 + (1 - wwr) / 2 * (p2 - p1)
            w2 = p2 + (1 - wwr) / 2 * (p1 - p2)
            w3 = p3 + (1 - wwr) / 2 * (p4 - p3)
            w4 = p4 + (1 - wwr) / 2 * (p3 - p4)

            a1 = p1
            a2 = p1 + pvr * (w1 - p1)
            a3 = p4 + pvr * (w4 - p4)
            a4 = p4

            b1 = w2
            b2 = w2 + pvr * (p2 - w2)
            b3 = w3 + pvr * (p3 - w3)
            b4 = w3

            coords1a = {}
            coords1b = {}

            coords1a["X1"] = a1[0]
            coords1a["Y1"] = a1[1]
            coords1a["Z1"] = a1[2] + pv_distance_from_roof
            coords1a["X2"] = a2[0]
            coords1a["Y2"] = a2[1]
            coords1a["Z2"] = a2[2] + pv_distance_from_roof
            coords1a["X3"] = a3[0]
            coords1a["Y3"] = a3[1]
            coords1a["Z3"] = a3[2] + pv_distance_from_roof
            coords1a["X4"] = a4[0]
            coords1a["Y4"] = a4[1]
            coords1a["Z4"] = a4[2] + pv_distance_from_roof

            coords1b["X1"] = b1[0]
            coords1b["Y1"] = b1[1]
            coords1b["Z1"] = b1[2] + pv_distance_from_roof
            coords1b["X2"] = b2[0]
            coords1b["Y2"] = b2[1]
            coords1b["Z2"] = b2[2] + pv_distance_from_roof
            coords1b["X3"] = b3[0]
            coords1b["Y3"] = b3[1]
            coords1b["Z3"] = b3[2] + pv_distance_from_roof
            coords1b["X4"] = b4[0]
            coords1b["Y4"] = b4[1]
            coords1b["Z4"] = b4[2] + pv_distance_from_roof

            coords.append(coords1a)
            coords.append(coords1b)

    return coords


def add_saddleback_roof(
    idf: IDF, building_config: BuildingConfig, loft_zone_name="Loft"
):
    """The method takes a flat roof and creates a loft floor,
    loft space and saddleback roof on top of it"""

    # change roof surfaces into ceiling
    for index, surface in enumerate(idf.idfobjects["BUILDINGSURFACE:DETAILED"]):

        if surface.Surface_Type.lower() == "roof":
            # search for zone name of last storey
            zone_name = surface.Zone_Name

            if building_config.attic_floor_layer_materials:
                c_name = "Last ceiling"
                f_name = "Last floor"
            else:
                c_name = "Ceiling"
                f_name = "Floor"
            ceiling = idf.idfobjects["BUILDINGSURFACE:DETAILED"][index]
            if zone_name != loft_zone_name:

                ceiling_name = (
                    "storey "
                    + str(building_config.number_of_stories)
                    + " "
                    + str(zone_name)
                    + " ceiling"
                )

                ceiling.Name = ceiling_name
                ceiling.Surface_Type = "ceiling"
                ceiling.Outside_Boundary_Condition = "Zone"
                ceiling.Outside_Boundary_Condition_Object = loft_zone_name
                ceiling.Sun_Exposure = "NoSun"
                ceiling.Wind_Exposure = "NoWind"

                ceiling.Construction_Name = c_name

            # if zones above and below are the same: delete surface + add internal mass
            else:
                idf.removeidfobject(ceiling)
                idf.newidfobject(
                    "INTERNALMASS",
                    Name="IntMass-" + zone_name + "-loft-ceiling",
                    Construction_Name=c_name,
                    Zone_or_ZoneList_Name=zone_name,
                    Surface_Area=get_surface_area(surface),
                )
                idf.newidfobject(
                    "INTERNALMASS",
                    Name="IntMass-" + zone_name + "-loft-floor",
                    Construction_Name=f_name,
                    Zone_or_ZoneList_Name=loft_zone_name,
                    Surface_Area=get_surface_area(surface),
                )

    roof_coords = get_saddleback_roof_coordinates(building_config)

    wall_coords = get_saddleback_roof_wall_coordinates(building_config)

    if loft_zone_name == "Loft":
        idf.newidfobject(
            "ZONE",
            Name="Loft",
        )

    # May want to change nomenclature on naming new elements
    # Currently N_X means that there are X of the new elements,
    # and N designates what element you are adding

    idf.newidfobject(
        "BUILDINGSURFACE:DETAILED",
        Name="roof surface 1",
        Construction_Name="ROOF-Construction",
        Surface_Type="ROOF",
        Zone_Name=loft_zone_name,
        Outside_Boundary_Condition="Outdoors",
        Sun_Exposure="SunExposed",
        Wind_Exposure="WindExposed",
        Vertex_1_Xcoordinate=roof_coords[0]["X1"],
        Vertex_1_Ycoordinate=roof_coords[0]["Y1"],
        Vertex_1_Zcoordinate=roof_coords[0]["Z1"],
        Vertex_2_Xcoordinate=roof_coords[0]["X2"],
        Vertex_2_Ycoordinate=roof_coords[0]["Y2"],
        Vertex_2_Zcoordinate=roof_coords[0]["Z2"],
        Vertex_3_Xcoordinate=roof_coords[0]["X3"],
        Vertex_3_Ycoordinate=roof_coords[0]["Y3"],
        Vertex_3_Zcoordinate=roof_coords[0]["Z3"],
        Vertex_4_Xcoordinate=roof_coords[0]["X4"],
        Vertex_4_Ycoordinate=roof_coords[0]["Y4"],
        Vertex_4_Zcoordinate=roof_coords[0]["Z4"],
    )

    idf.newidfobject(
        "BUILDINGSURFACE:DETAILED",
        Name="roof surface 2",
        Construction_Name="ROOF-Construction",
        Surface_Type="ROOF",
        Zone_Name=loft_zone_name,
        Outside_Boundary_Condition="Outdoors",
        Sun_Exposure="SunExposed",
        Wind_Exposure="WindExposed",
        Vertex_1_Xcoordinate=roof_coords[1]["X1"],
        Vertex_1_Ycoordinate=roof_coords[1]["Y1"],
        Vertex_1_Zcoordinate=roof_coords[1]["Z1"],
        Vertex_2_Xcoordinate=roof_coords[1]["X2"],
        Vertex_2_Ycoordinate=roof_coords[1]["Y2"],
        Vertex_2_Zcoordinate=roof_coords[1]["Z2"],
        Vertex_3_Xcoordinate=roof_coords[1]["X3"],
        Vertex_3_Ycoordinate=roof_coords[1]["Y3"],
        Vertex_3_Zcoordinate=roof_coords[1]["Z3"],
        Vertex_4_Xcoordinate=roof_coords[1]["X4"],
        Vertex_4_Ycoordinate=roof_coords[1]["Y4"],
        Vertex_4_Zcoordinate=roof_coords[1]["Z4"],
    )

    idf.newidfobject(
        "BUILDINGSURFACE:DETAILED",
        Name="loft side wall 1",
        Construction_Name="WALL-Construction",
        Surface_Type="WALL",
        Zone_Name=loft_zone_name,
        Outside_Boundary_Condition="Outdoors",
        Sun_Exposure="SunExposed",
        Wind_Exposure="WindExposed",
        Number_of_Vertices=3,
        Vertex_1_Xcoordinate=wall_coords[0]["X1"],
        Vertex_1_Ycoordinate=wall_coords[0]["Y1"],
        Vertex_1_Zcoordinate=wall_coords[0]["Z1"],
        Vertex_2_Xcoordinate=wall_coords[0]["X2"],
        Vertex_2_Ycoordinate=wall_coords[0]["Y2"],
        Vertex_2_Zcoordinate=wall_coords[0]["Z2"],
        Vertex_3_Xcoordinate=wall_coords[0]["X3"],
        Vertex_3_Ycoordinate=wall_coords[0]["Y3"],
        Vertex_3_Zcoordinate=wall_coords[0]["Z3"],
    )

    idf.newidfobject(
        "BUILDINGSURFACE:DETAILED",
        Name="loft side wall 2",
        Construction_Name="WALL-Construction",
        Surface_Type="WALL",
        Zone_Name=loft_zone_name,
        Outside_Boundary_Condition="Outdoors",
        Sun_Exposure="SunExposed",
        Wind_Exposure="WindExposed",
        Number_of_Vertices=3,
        Vertex_1_Xcoordinate=wall_coords[1]["X1"],
        Vertex_1_Ycoordinate=wall_coords[1]["Y1"],
        Vertex_1_Zcoordinate=wall_coords[1]["Z1"],
        Vertex_2_Xcoordinate=wall_coords[1]["X2"],
        Vertex_2_Ycoordinate=wall_coords[1]["Y2"],
        Vertex_2_Zcoordinate=wall_coords[1]["Z2"],
        Vertex_3_Xcoordinate=wall_coords[1]["X3"],
        Vertex_3_Ycoordinate=wall_coords[1]["Y3"],
        Vertex_3_Zcoordinate=wall_coords[1]["Z3"],
    )

    return idf


def change_roof_to_adiabatic(idf: IDF) -> IDF:
    for index, surface in enumerate(idf.idfobjects["BUILDINGSURFACE:DETAILED"]):
        if surface.Surface_Type == "roof":
            roof = idf.idfobjects["BUILDINGSURFACE:DETAILED"][index]
            roof.Outside_Boundary_Condition = "Adiabatic"
            roof.Sun_Exposure = "NoSun"
            roof.Wind_Exposure = "NoWind"

    return idf


def add_flat_roof(
    idf: IDF,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    distance_from_ground: float,
    zone: str,
):
    coords = get_roof_xy_coordinates(xmin, xmax, ymin, ymax)
    idf.newidfobject(
        "BuildingSurface:Detailed".upper(),
        Name="Roof " + zone,
        Zone_Name=zone,
        Construction_Name="Roof",
        Surface_Type="Roof",
        View_Factor_to_Ground=0.0,
        Number_of_Vertices=4,
        Vertex_1_Xcoordinate=coords["X1"],
        Vertex_1_Ycoordinate=coords["Y1"],
        Vertex_1_Zcoordinate=distance_from_ground,
        Vertex_2_Xcoordinate=coords["X2"],
        Vertex_2_Ycoordinate=coords["Y2"],
        Vertex_2_Zcoordinate=distance_from_ground,
        Vertex_3_Xcoordinate=coords["X3"],
        Vertex_3_Ycoordinate=coords["Y3"],
        Vertex_3_Zcoordinate=distance_from_ground,
        Vertex_4_Xcoordinate=coords["X4"],
        Vertex_4_Ycoordinate=coords["Y4"],
        Vertex_4_Zcoordinate=distance_from_ground,
        Sun_Exposure="SunExposed",
        Wind_Exposure="WindExposed",
        Outside_Boundary_Condition="Outdoors",
    )

    return idf


def get_roof_xy_coordinates(xmin: float, xmax: float, ymin: float, ymax: float):
    # below is outside
    return {
        "X1": xmin,
        "Y1": ymin,
        "X2": xmax,
        "Y2": ymin,
        "X3": xmax,
        "Y3": ymax,
        "X4": xmin,
        "Y4": ymax,
    }
