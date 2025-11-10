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


def random_sample(row):
    return np.random.choice(row)


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

    import numpy as np
    from shapely.geometry import Polygon
    from itertools import product
    import matplotlib.pyplot as plt
    from geomeppy.geom.polygons import almostequal

    from itertools import product
    from shapely.geometry import Polygon
    import numpy as np
    from geomeppy.geom.polygons import almostequal

    def match_all_idf_surfaces(idf, tol=0.01, debug_plot=False):
        from geomeppy.geom.polygons import almostequal
        from shapely.geometry import Polygon
        import numpy as np

        surfaces = idf.getsurfaces() + idf.getshadingsurfaces()
        matched = {}

        def normal_vector(coords):
            v1 = np.array(coords[1]) - np.array(coords[0])
            v2 = np.array(coords[2]) - np.array(coords[0])
            n = np.cross(v1, v2)
            return n / np.linalg.norm(n)

        def is_horizontal(normal):
            return almostequal(abs(normal[2]), 1.0, 4)

        # ---- Horizontal (floors & ceilings) ----
        planes = getidfplanes(surfaces)
        for distance, vectors in planes.items():
            for vector, these_surfaces in vectors.items():
                for surface in these_surfaces:
                    set_unmatched_surface(surface, vector)
                matches = planes.get(-distance, {}).get(-vector, [])
                for s, m in product(these_surfaces, matches):
                    if any(x in s.Surface_Type.lower() for x in ["roof", "ceiling"]):
                        if Polygon(s.coords).simplify(tol).buffer(0).equals(
                            Polygon(m.coords).simplify(tol).buffer(0)
                        ):
                            matched[tuple(sorted((s.Name, m.Name)))] = (s, m)

        # ---- Vertical (walls) ----
        for s1, s2 in product(surfaces, repeat=2):
            if s1 == s2:
                continue
            n1 = normal_vector(s1.coords)
            n2 = normal_vector(s2.coords)
            if not almostequal(np.dot(n1, n2), -1.0, 4):
                continue
            if is_horizontal(n1) or is_horizontal(n2):
                continue
            poly1 = Polygon([(x, y) for x, y, z in s1.coords])
            poly2 = Polygon([(x, y) for x, y, z in s2.coords])
            if abs(poly1.area - poly2.area) > tol:
                continue
            if poly1.buffer(tol).equals(poly2.buffer(tol)):
                matched[tuple(sorted((s1.Name, s2.Name)))] = (s1, s2)
                if debug_plot:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    x1, y1 = zip(*poly1.exterior.coords)
                    x2, y2 = zip(*poly2.exterior.coords)
                    ax.fill(x1, y1, alpha=0.5, label=s1.Name)
                    ax.fill(x2, y2, alpha=0.5, label=s2.Name)
                    ax.legend()
                    ax.set_aspect('equal')
                    plt.title(f"Match: {s1.Name} <-> {s2.Name}")
                    plt.show()

        # Mark all matched at once
        for key, (s1, s2) in matched.items():
            set_matched_surfaces(s1, s2)

        return matched

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
        self.match_all_idf_surfaces()

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
    return package_directory + "/data/electricity_import/" + filename


def get_electricity_surplus_file_path(filename):
    return package_directory + "/data/electricity_export/" + filename


def get_heating_pattern_file(filename):
    return package_directory + "/data/heating_pattern/" + filename
