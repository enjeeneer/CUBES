# pylint: disable-all
"""some utility functions to be used throughout the construct package"""

import numpy as np
import math
from cubes.constants import package_directory
from geomeppy import IDF
from geomeppy.builder import Block, Zone
from geomeppy.geom.core_perim import core_perim_zone_coordinates
from geomeppy.geom.surfaces import (
    getidfplanes,
    set_coords,
    set_matched_surfaces,
    set_unmatched_surface,
    minimal_set,
)
from geomeppy.geom.intersect_match import sorted_tuple, intersect_idf_surfaces
from itertools import product
from geomeppy.utilities import almostequal
from shapely.geometry import Polygon
from collections import defaultdict
from itertools import combinations
from geomeppy.geom.polygons import Polygon3D


class ModifiedIDF(IDF):
    """This is a modified version of geomeppy's IDF
    The additions are three functions:
    rotate_coords
    new_match_idf_surfaces
    add_block

    Args:
        IDF (_type_): _description_
    """

    def rotate_coords(self, coords, steps):
        """Rotate the list of coordinates by a number of steps."""
        return coords[steps:] + coords[:steps]

    def new_match_idf_surfaces(idf):
        # type: (IDF) -> None
        """Match all surfaces in an IDF.

        :param idf: The IDF.
        """
        surfaces = idf.getsurfaces() + idf.getshadingsurfaces()
        planes = getidfplanes(surfaces)
        matched = {}
        for distance in planes:
            for vector in planes[distance]:
                surfaces = planes[distance][vector]
                for surface in surfaces:
                    set_unmatched_surface(surface, vector)
                matches = planes.get(-distance, {}).get(-vector, [])
                for s, m in product(surfaces, matches):
                    if "roof" in s.Surface_Type and m.Surface_Type:
                        poly_s = Polygon(s.coords).simplify(0.01).buffer(0)
                        poly_m = Polygon(m.coords).simplify(0.01).buffer(0)
                        if poly_s.equals(poly_m):
                            matched[sorted_tuple(m, s)] = (m, s)
                    else:
                        if almostequal(s.coords, reversed(m.coords)):
                            matched[sorted_tuple(m, s)] = (m, s)

        for key in matched:
            set_matched_surfaces(*matched[key])

    # def new_match_idf_surfaces(self):
    #    """Match all surfaces in an IDF."""
    #    print("using new match")
    #    surfaces = self.getsurfaces() + self.getshadingsurfaces()
    #    planes = getidfplanes(surfaces)
    #    print(planes)
    #    matched = {}
    #    for distance in planes:
    #        for vector in planes[distance]:
    #            surfaces = planes[distance][vector]
    #            for surface in surfaces:
    #                set_unmatched_surface(surface, vector)
    #            matches = planes.get(-distance, {}).get(-vector, [])
    #            for s, m in product(surfaces, matches):
    #                if "roof" in s.Surface_Type and m.Surface_Type:
    #                    poly_s = Polygon(s.coords).simplify(0.01).buffer(0)
    #                    poly_m = Polygon(m.coords).simplify(0.01).buffer(0)
    #                    if poly_s.equals(poly_m):
    #                        matched[sorted_tuple(m, s)] = (m, s)
    #                else:
    #
    #                    # Check direct match or mirror match
    #                    for direct in [True, False]:
    #                        for i in range(len(s.coords)):
    #                            rotated = self.rotate_coords(s.coords, i)
    #
    #                            if direct:
    #                                if rotated == m.coords:
    #                                    matched[sorted_tuple(m, s)] = (m, s)
    #                            else:
    #                                if rotated == list(reversed(m.coords)):
    #                                    matched[sorted_tuple(m, s)] = (m, s)
    #                    if almostequal(s.coords, reversed(m.coords)):
    #                    #if almostequal(sorted(s.coords),sorted(m.coords)):
    #                        matched[sorted_tuple(m, s)] = (m, s)
    #
    #    for key in matched:
    #        set_matched_surfaces(*matched[key])

    def get_adjacencies(self, surfaces):
        """Create a dictionary mapping surfaces to their adjacent surfaces.

        :param surfaces: A mutable list of surfaces.
        :returns: Mapping of surfaces to adjacent surfaces.
        """
        adjacencies = defaultdict(list)  # type: defaultdict
        # find all adjacent surfaces
        for s1, s2 in combinations(surfaces, 2):
            adjacencies = self.populate_adjacencies(adjacencies, s1, s2)
        for adjacency, polys in adjacencies.items():
            adjacencies[adjacency] = minimal_set(polys)
        return adjacencies

    def populate_adjacencies(self, adjacencies, s1, s2):
        """Update the adjacencies dict with any intersections between two surfaces.

        :param adjacencies: Dict to contain lists of adjacent surfaces.
        :param s1: Object representing an EnergyPlus surface.
        :param s2: Object representing an EnergyPlus surface.
        :returns: An updated dict of adjacencies.
        """
        poly1 = Polygon3D(s1.coords)
        poly2 = Polygon3D(s2.coords)
        if not almostequal(abs(poly1.distance), abs(poly2.distance), 4):
            return adjacencies
        if not almostequal(poly1.normal_vector, poly2.normal_vector, 4):
            if not almostequal(poly1.normal_vector, -poly2.normal_vector, 4):
                return adjacencies

        intersection = poly1.intersect(poly2)
        if intersection:
            new_surfaces = self.intersect(poly1, poly2)
            new_s1 = [
                s
                for s in new_surfaces
                if almostequal(s.normal_vector, poly1.normal_vector, 4)
            ]
            new_s2 = [
                s
                for s in new_surfaces
                if almostequal(s.normal_vector, poly2.normal_vector, 4)
            ]
            adjacencies[(s1.key, s1.Name)] += new_s1
            adjacencies[(s2.key, s2.Name)] += new_s2
        return adjacencies

    def intersect(self, poly1, poly2):
        """Calculate the polygons to represent the intersection of two polygons.

        :param poly1: The first polygon.
        :param poly2: The second polygon.
        :returns: A list of unique polygons.

        """
        polys = []
        polys.extend(poly1.intersect(poly2))
        polys.extend(poly2.intersect(poly1))

        polys.extend(poly1.difference(poly2))
        polys.extend(poly2.difference(poly1))
        return polys

    def intersect_idf_surfaces(self):
        # type: (IDF) -> None
        """Intersect all surfaces in an IDF.

        :param idf: The IDF.
        """
        surfaces = self.getsurfaces() + self.getshadingsurfaces()
        try:
            ggr = self.idfobjects["GLOBALGEOMETRYRULES"][0]
        except IndexError:
            ggr = None
        # get all the intersected surfaces
        adjacencies = self.get_adjacencies(surfaces)
        for surface in adjacencies:
            key, name = surface
            new_surfaces = adjacencies[surface]
            old_obj = self.getobject(key.upper(), name)
            for i, new_coords in enumerate(new_surfaces, 1):
                new = self.copyidfobject(old_obj)
                new.Name = "%s_%i" % (name, i)
                set_coords(new, new_coords, ggr)
            self.removeidfobject(old_obj)

    def intersect_match(self):
        # type: () -> None
        """Intersect all surfaces in the IDF, then set boundary conditions."""
        self.intersect_idf_surfaces()
        self.new_match_idf_surfaces()

    def add_block(self, *args, **kwargs):
        """Add a block to the IDF."""
        block = Block(*args, **kwargs)
        block.zoning = kwargs.get("zoning", "by_storey")
        if block.zoning == "by_storey":
            zones = [
                Zone("Block %s Storey %i" % (block.name, storey["storey_no"]), storey)
                for storey in block.stories
            ]
        elif block.zoning == "core/perim":
            zones = []
            try:
                for name, coords in core_perim_zone_coordinates(
                    block.coordinates, block.perim_depth
                )[0].items():
                    block = Block(
                        name=name,
                        coordinates=coords,
                        height=block.height,
                        num_stories=block.num_stories,
                    )
                    zones += [
                        Zone(
                            "Block %s Storey %i" % (block.name, storey["storey_no"]),
                            storey,
                        )
                        for storey in block.stories
                    ]
            except NotImplementedError:
                raise ValueError("Perimeter depth is too great")
        else:
            raise ValueError("%s is not a valid zoning rule" % block.zoning)

        for zone in zones:
            self.add_zone(zone)


def get_floor_information(
    zone_info, distance_to_ground, subfloor_height, storey_height
):
    # this function will need to go through each zone and add in the coordinates etc
    # of the floor. the info i need to provide is:
    # storey, xy min max, distance from ground, inner zone, outer zone.
    # TODO slice surfaces for correct outerzone

    zone_names = zone_info["zone"]

    floor_info = {
        "xmin": [],
        "xmax": [],
        "ymin": [],
        "ymax": [],
        "distance_from_ground": [],
        "inner_zone": [],
        "outer_zone": [],
    }

    for i in range(len(zone_info["zone"])):
        xmin = zone_info["zone_bottom_left"][i][0]
        xmax = zone_info["zone_top_right"][i][0]
        ymin = zone_info["zone_bottom_left"][i][1]
        ymax = zone_info["zone_top_right"][i][1]

        floor_info["xmin"].append(xmin)
        floor_info["xmax"].append(xmax)
        floor_info["ymin"].append(ymin)
        floor_info["ymax"].append(ymax)

        if zone_info["storey"][i] == 0:
            if subfloor_height > 0:

                distance_from_ground = distance_to_ground

                floor_info["distance_from_ground"].append(distance_from_ground)
                floor_info["outer_zone"].append("Subfloor")

            else:
                floor_info["distance_from_ground"].append(distance_from_ground)
                floor_info["outer_zone"].append("Ground")

        elif zone_info["storey"][i] > 0:
            distance_from_ground = (
                storey_height * zone_info["storey"][i] + distance_to_ground
            )
            floor_info["distance_from_ground"].append(distance_from_ground)

        floor_info["inner_zone"].append(zone_info["zone"][i])

        # nasty hardcoding incoming, this forces the zones on ground floor to be
        # directly below zones on second floor etc.

        if i > 3:
            floor_info["outer_zone"].append(zone_names[i - 4])

    zone_info["floor"].append(floor_info)

    return zone_info


def label_zone_shared_walls(zone_info):

    zone_coords = zone_info["zone_coords"]
    zone_walls = zone_info["wall"]

    num_zones = len(zone_coords)

    for i, walls_in_zone in enumerate(zone_walls):

        empty_lists = [[] for i in range(num_zones)]  # pylint: disable=unused-variable
        walls_in_zone.append(empty_lists)

        # TODO remove the code below, list comprehension above supercedes
        # for _ in range(num_zones):
        #    walls_in_zone["shared_zone"].append([])

        for wall_coords in walls_in_zone["coordinates"]:
            for j, zc in enumerate(zone_coords):

                if i != j:
                    shared = are_coordinates_shared(zc, wall_coords)

                    if shared:
                        walls_in_zone["shared_zone"][j].append("SHARED")
                    else:
                        walls_in_zone["shared_zone"][j].append("NOT_SHARED")

                else:
                    shared = "IN_ZONE"
                    walls_in_zone["shared_zone"][j].append(shared)

    return zone_info


def are_coordinates_shared(coordinate_list, pairs):
    return all(pair in coordinate_list for pair in pairs)


def get_zone_coords(bottom_left_coord, top_right_coord):
    # zone will have the name and the bottom left and top right coord
    # zone_coords go from bottom left, bottom right, top left, top right, with North up
    zone_coords = []

    zone_coords.append(tuple(bottom_left_coord))
    zone_coords.append((top_right_coord[0], bottom_left_coord[1]))
    zone_coords.append((bottom_left_coord[0], top_right_coord[1]))
    zone_coords.append(tuple(top_right_coord))

    return zone_coords


def get_zone_wall_coords(zone_coords):
    # wall coords are given in an anticlockwise manner, with 0 element being 'north'
    # perspective is if looking at wall from outside, initial coord is left most

    wall_coords = []
    wall_coords.append((zone_coords[3], zone_coords[2]))
    wall_coords.append((zone_coords[1], zone_coords[3]))
    wall_coords.append((zone_coords[0], zone_coords[1]))
    wall_coords.append((zone_coords[2], zone_coords[0]))

    return wall_coords


def get_zone_walls_information(zone_info):

    for i, zone_in in enumerate(zone_info["zone"]):  # pylint: disable=unused-variable

        wall = {
            "storey": [],
            "coordinates": [],
            "direction": [],
            "length": [],
            "upper_left": [],
        }

        wall_coords = get_zone_wall_coords(zone_info["zone_coords"][i])

        for wall_coord in wall_coords:
            wall["storey"].append(zone_info["storey"][i])
            wall["coordinates"].append(wall_coord)

            x_length = abs(wall_coord[-1][0] - wall_coord[0][0])
            y_length = abs(wall_coord[-1][1] - wall_coord[0][1])

            upper_left = wall_coord[0]
            upper_left = tuple(upper_left)
            wall["upper_left"].append(upper_left)

            # check if on same y-coords, if True must be north south

            if wall_coord[0][1] == wall_coord[1][1]:
                wall["length"].append(x_length)
                if wall_coord[0][0] < wall_coord[1][0]:
                    wall["direction"].append("South")
                else:
                    wall["direction"].append("North")
            else:
                wall["length"].append(y_length)
                if wall_coord[0][1] < wall_coord[1][1]:
                    wall["direction"].append("East")
                else:
                    wall["direction"].append("West")

        zone_info["wall"].append(wall)

    return zone_info


def remove_non_unique_wall(zone_info, wall_info):
    for wall in wall_info:
        zone_index = wall[1][0]
        for i, zw in enumerate(zone_info["wall"][zone_index]["coordinates"]):
            if wall[0] == zw:
                wall.append(zone_info["wall"][zone_index]["direction"][i])
                wall.append(zone_info["wall"][zone_index]["length"][i])
                wall.append(zone_info["wall"][zone_index]["upper_left"][i])
                wall.append(zone_info["storey"][zone_index])

    return zone_info, wall_info


def identify_unique_walls(zone_info):

    all_zones = []
    all_walls = []
    unique_walls = []
    unique_zone_indices = []
    wall_info = []

    # get all walls in one list
    for i, zw in enumerate(zone_info["wall"]):
        # go through each wall within zone
        for wall in zw["coordinates"]:
            all_zones.append(i)
            all_walls.append(wall)

    # what does this method do?
    # checks if the wall is in the list of unique walls, if not add it
    # need to check forward and reverse as wall coordinate order depends on orientation
    for index, coordinates in enumerate(all_walls):
        if coordinates not in unique_walls:
            if tuple(reversed(coordinates)) not in unique_walls:
                unique_walls.append(coordinates)
                unique_zone = all_zones[index]
                unique_zone_indices.append(unique_zone)

    for i, uw in enumerate(unique_walls):
        matching = []
        # go through all unique walls, find index of all matching coords
        for j, wall in enumerate(all_walls):
            # eliminate external walls
            if uw == wall:
                matching.append(j)
            elif uw == tuple(reversed(wall)):
                matching.append(j)

        if len(matching) == 1:
            wall_info.append([uw, (unique_zone_indices[i], "external")])
        elif len(matching) > 1:
            matching = matching[1:]
            for match in matching:
                wall_info.append([uw, (unique_zone_indices[i], all_zones[match])])

    return wall_info


def calculate_zone_area(zone_coords):
    if len(zone_coords) < 4:
        return ValueError("Invalid input: Less than 4 zone coords provided.")

    # Sort points based on x-coordinates,
    # then y-coordinates to ensure they are ordered correctly
    sorted_coords = sorted(zone_coords, key=lambda x: (x[0], x[1]))

    # Calculate length and width based on sorted points
    length = abs(sorted_coords[0][0] - sorted_coords[3][0])
    width = abs(sorted_coords[0][1] - sorted_coords[1][1])

    return length * width


def calculate_coordinate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def calculate_coordinate_area(x1, y1, x2, y2):
    # TODO check with Hannes why the area is divided by 2
    area = ((x2 - x1) * (y2 - y1)) / 2
    return area


def get_zone_hvac_equipment_list_name(zone_name):
    return zone_name + "-Equipment"


def get_zone_air_inlet_nodelist_name(zone_name):
    return zone_name + " Inlets"


def get_zone_air_outlet_nodelist_name(zone_name):
    return zone_name + " Exhausts"


def append_node_to_nodelist(idf: IDF, node_name, node_list_name):
    """this function finds a node list in an idf and
    appends a node name to the end of it"""

    i_list = -1
    for il, l in enumerate(idf.idfobjects["NODELIST"]):
        if l.Name == node_list_name:
            i_list = il

    if i_list == -1:
        # node list not found: make new node list
        idf.newidfobject(
            "NODELIST",
            Name=node_list_name,
            Node_1_Name=node_name,
        )
        return idf

    nnode = 1

    while getattr(idf.idfobjects["NODELIST"][i_list], "Node_" + str(nnode) + "_Name"):
        nnode += 1

    setattr(
        idf.idfobjects["NODELIST"][i_list],
        "Node_" + str(nnode) + "_Name",
        node_name,
    )

    return idf


def rotation_changes_north_direction(rotation):
    """check if the building rotation is such that the north facing side is changed"""
    if np.cos(rotation / 180 * np.pi) > 0:
        return False
    else:
        return True


def get_walls_in_limits(
    idf, x_lims=(-1e4, 1e4), y_lims=(-1e4, 1e4), z_lims=(-1e4, 1e4)
):
    walls = []

    for wall in [*idf.getsurfaces("wall"), *idf.getsurfaces("roof")]:
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
        surface_object.Vertex_2_Zcoordinate,
        surface_object.Vertex_3_Zcoordinate,
        surface_object.Vertex_4_Zcoordinate,
    ]
    return (max(z_coordinates) + min(z_coordinates)) / 2.0


def get_schedule(name):
    with open(
        package_directory + "/data/schedules/" + name + ".sch", "r", encoding="utf-8"
    ) as file2:
        schedule_str = file2.read()
    return schedule_str


def write_string_to_file(string, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(string)


def get_grid_carbon_intensity_file_path(filename):
    return package_directory + "/data/grid/" + filename


def get_gas_pricing_file_path(filename):
    return package_directory + "/data/gas/" + filename


def get_electricity_pricing_file_path(filename):
    return package_directory + "/data/electricity/" + filename
