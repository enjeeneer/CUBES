"""This module adds walls, floors and roof to an IDF and defines the zones"""
from typing import Tuple
import numpy as np

from geomeppy import IDF
from cubes.construct.buildingconfig import BuildingConfig
from cubes.construct.buildingconfig_options import Zoning, RoofType
from cubes.construct.utilities import (
    rotation_changes_north_direction,
    get_zone_walls_information,
    get_zone_coords,
    calculate_coordinate_area,
    calculate_zone_area,
    identify_unique_walls,
    remove_non_unique_wall,
    get_floor_information,
)
from cubes.construct.roof import (
    add_flat_roof,
    add_saddleback_roof,
    change_roof_to_adiabatic,
)

residential_bedroom_area_ratio = 0.3
minimum_room_height = 1.525


def add_surfaces_and_zones(idf: IDF, building_config: BuildingConfig) -> IDF:
    area_per_zone = {}
    if building_config.zoning == Zoning.ONE_ZONE_PER_FLOOR.value:
        idf.add_block(
            name="Cube",
            coordinates=[
                (building_config.length_wall_x, 0),
                (building_config.length_wall_x, building_config.length_wall_y),
                (0, building_config.length_wall_y),
                (0, 0),
            ],
            height=building_config.number_of_stories
            * building_config.number_of_stories,
            num_stories=building_config.number_of_stories,
            zoning="by_storey",
        )

    elif building_config.zoning == Zoning.SINGLE_ZONE.value:
        idf.newidfobject(
            "ZONE",
            Name="Living",
        )
        idf = add_external_wall(
            idf,
            1,
            "North",
            building_config.length_wall_x,
            (
                building_config.length_wall_x,
                building_config.length_wall_y,
                building_config.storey_height * building_config.number_of_stories,
            ),
            building_config.storey_height * building_config.number_of_stories,
            "Living",
            building_config.distance_to_neighbour[0] == 0,
        )
        idf = add_external_wall(
            idf,
            1,
            "East",
            building_config.length_wall_y,
            (
                building_config.length_wall_x,
                0,
                building_config.storey_height * building_config.number_of_stories,
            ),
            building_config.storey_height * building_config.number_of_stories,
            "Living",
            building_config.distance_to_neighbour[1] == 0,
        )
        idf = add_external_wall(
            idf,
            1,
            "South",
            building_config.length_wall_x,
            (
                0,
                0,
                building_config.storey_height * building_config.number_of_stories,
            ),
            building_config.storey_height * building_config.number_of_stories,
            "Living",
            building_config.distance_to_neighbour[2] == 0,
        )
        idf = add_external_wall(
            idf,
            1,
            "West",
            building_config.length_wall_y,
            (
                0,
                building_config.length_wall_y,
                building_config.storey_height * building_config.number_of_stories,
            ),
            building_config.storey_height * building_config.number_of_stories,
            "Living",
            building_config.distance_to_neighbour[3] == 0,
        )

        # add subfloor if present
        if building_config.subfloor_height > 0:
            zone = "Subfloor"
            idf.newidfobject(
                "ZONE",
                Name=zone,
            )

            idf = add_external_wall(
                idf,
                -1,
                "North",
                building_config.length_wall_x,
                (
                    building_config.length_wall_x,
                    building_config.length_wall_y,
                    0,
                ),
                building_config.subfloor_height,
                zone,
                building_config.distance_to_neighbour[0] == 0,
            )
            idf = add_external_wall(
                idf,
                -1,
                "East",
                building_config.length_wall_y,
                (
                    building_config.length_wall_x,
                    0,
                    0,
                ),
                building_config.subfloor_height,
                zone,
                building_config.distance_to_neighbour[1] == 0,
            )
            idf = add_external_wall(
                idf,
                -1,
                "South",
                building_config.length_wall_x,
                (
                    0,
                    0,
                    0,
                ),
                building_config.subfloor_height,
                zone,
                building_config.distance_to_neighbour[2] == 0,
            )
            idf = add_external_wall(
                idf,
                -1,
                "West",
                building_config.length_wall_y,
                (
                    0,
                    building_config.length_wall_y,
                    0,
                ),
                building_config.subfloor_height,
                zone,
                building_config.distance_to_neighbour[3] == 0,
            )
            idf = add_floor(
                idf,
                -1,
                0,
                building_config.length_wall_x,
                0,
                building_config.length_wall_y,
                -building_config.subfloor_height,
                "Subfloor",
                "Ground",
            )

        # add ground floor to the idf
        if building_config.subfloor_height > 0:
            under_ground_floor = "SubFloor"
        else:
            under_ground_floor = "Ground"

        idf = add_floor(
            idf,
            1,
            0,
            building_config.length_wall_x,
            0,
            building_config.length_wall_y,
            building_config.distance_to_ground,
            "Living",
            under_ground_floor,
        )

        # internal floors neglected for now

        roof_level = (
            building_config.distance_to_ground
            + building_config.number_of_stories * building_config.storey_height
        )
        idf = add_flat_roof(
            idf,
            0,
            building_config.length_wall_x,
            0,
            building_config.length_wall_y,
            roof_level,
            "Living",
        )

        if building_config.roof_type == RoofType.SADDLEBACK.value:
            if building_config.loft_is_heated:
                idf = add_saddleback_roof(idf, building_config, "Living")
            else:
                idf = add_saddleback_roof(idf, building_config, "Loft")

    elif building_config.zoning == Zoning.RESIDENTIAL_DWELLING.value:
        # add zones
        idf.newidfobject(
            "ZONE",
            Name="Living",
        )
        idf.newidfobject(
            "ZONE",
            Name="Bedroom",
        )

        # work out where the zone boundary is
        total_floor_area = (
            building_config.length_wall_x
            * building_config.length_wall_y
            * building_config.number_of_stories
        )
        if building_config.loft_is_heated:
            if building_config.roof_ridge_along_x:
                loft_area_fraction = (
                    building_config.length_wall_y
                    * minimum_room_height
                    / building_config.roof_height
                )
            else:
                loft_area_fraction = (
                    building_config.length_wall_x
                    * minimum_room_height
                    / building_config.roof_height
                )
            total_floor_area += (
                building_config.length_wall_x
                * building_config.length_wall_y
                * loft_area_fraction
            )
        storey_floor_area = (
            building_config.length_wall_x * building_config.length_wall_y
        )
        storey_split_ratio = 0
        storey_split_in_x = (
            building_config.length_wall_x > building_config.length_wall_y
        )

        bedroom_to_place = residential_bedroom_area_ratio * total_floor_area
        area_per_zone["Living"] = total_floor_area * (
            1 - residential_bedroom_area_ratio
        )
        area_per_zone["Bedroom"] = total_floor_area * residential_bedroom_area_ratio
        # if loft is heated, the bedroom starts in the loft and occupies at least
        # all of the loft plus a third of of the next floor
        # (this is to avoid a zero area split)
        if building_config.loft_is_heated:
            bedroom_to_place -= (
                building_config.length_wall_x
                * building_config.length_wall_y
                * loft_area_fraction
            )
            bedroom_to_place = max(
                bedroom_to_place,
                0.3 * building_config.length_wall_x * building_config.length_wall_y,
            )

            area_per_zone["Bedroom"] = (
                building_config.length_wall_x
                * building_config.length_wall_y
                * loft_area_fraction
                + bedroom_to_place
            )
            area_per_zone["Living"] = total_floor_area - area_per_zone["Bedroom"]

        for s in range(building_config.number_of_stories - 1, -1, -1):
            if bedroom_to_place > storey_floor_area:
                bedroom_to_place = bedroom_to_place - storey_floor_area
            else:
                zone_split_storey = s
                storey_split_ratio = bedroom_to_place / storey_floor_area
                break

        north_flip = rotation_changes_north_direction(building_config.rotation)
        if not north_flip:
            storey_split_ratio_flip = (
                1 - storey_split_ratio
            )  # the bedroom zone has the larger coordinate
        else:
            storey_split_ratio_flip = storey_split_ratio

        front_zone = "Living" if not north_flip else "Bedroom"
        back_zone = "Living" if north_flip else "Bedroom"

        # add the walls to the idf
        for s in range(building_config.number_of_stories):
            storey_level = (
                building_config.distance_to_ground + s * building_config.storey_height
            )
            if s != zone_split_storey:
                zone = "Living" if s < zone_split_storey else "Bedroom"

                idf = add_external_wall(
                    idf,
                    s + 1,
                    "North",
                    building_config.length_wall_x,
                    (
                        building_config.length_wall_x,
                        building_config.length_wall_y,
                        storey_level + building_config.storey_height,
                    ),
                    building_config.storey_height,
                    zone,
                    building_config.distance_to_neighbour[0] == 0,
                )
                idf = add_external_wall(
                    idf,
                    s + 1,
                    "East",
                    building_config.length_wall_y,
                    (
                        building_config.length_wall_x,
                        0,
                        storey_level + building_config.storey_height,
                    ),
                    building_config.storey_height,
                    zone,
                    building_config.distance_to_neighbour[1] == 0,
                )
                idf = add_external_wall(
                    idf,
                    s + 1,
                    "South",
                    building_config.length_wall_x,
                    (
                        0,
                        0,
                        storey_level + building_config.storey_height,
                    ),
                    building_config.storey_height,
                    zone,
                    building_config.distance_to_neighbour[2] == 0,
                )
                idf = add_external_wall(
                    idf,
                    s + 1,
                    "West",
                    building_config.length_wall_y,
                    (
                        0,
                        building_config.length_wall_y,
                        storey_level + building_config.storey_height,
                    ),
                    building_config.storey_height,
                    zone,
                    building_config.distance_to_neighbour[3] == 0,
                )
            else:

                if storey_split_in_x:
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "North",
                        building_config.length_wall_x * storey_split_ratio_flip,
                        (
                            building_config.length_wall_x * storey_split_ratio_flip,
                            building_config.length_wall_y,
                            storey_level + building_config.storey_height,
                        ),
                        building_config.storey_height,
                        front_zone,
                        building_config.distance_to_neighbour[0] == 0,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "South",
                        building_config.length_wall_x * storey_split_ratio_flip,
                        (
                            0,
                            0,
                            storey_level + building_config.storey_height,
                        ),
                        building_config.storey_height,
                        front_zone,
                        building_config.distance_to_neighbour[2] == 0,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "West",
                        building_config.length_wall_y,
                        (
                            0,
                            building_config.length_wall_y,
                            storey_level + building_config.storey_height,
                        ),
                        building_config.storey_height,
                        front_zone,
                        building_config.distance_to_neighbour[3] == 0,
                    )
                    idf = add_internal_wall(
                        idf,
                        s + 1,
                        "East",
                        building_config.length_wall_y,
                        (
                            building_config.length_wall_x * storey_split_ratio_flip,
                            0,
                            storey_level + building_config.storey_height,
                        ),
                        building_config.storey_height,
                        front_zone,
                        back_zone,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "North",
                        building_config.length_wall_x * (1 - storey_split_ratio_flip),
                        (
                            building_config.length_wall_x,
                            building_config.length_wall_y,
                            storey_level + building_config.storey_height,
                        ),
                        building_config.storey_height,
                        back_zone,
                        building_config.distance_to_neighbour[0] == 0,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "South",
                        building_config.length_wall_x * (1 - storey_split_ratio_flip),
                        (
                            building_config.length_wall_x * storey_split_ratio_flip,
                            0,
                            storey_level + building_config.storey_height,
                        ),
                        building_config.storey_height,
                        back_zone,
                        building_config.distance_to_neighbour[2] == 0,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "East",
                        building_config.length_wall_y,
                        (
                            building_config.length_wall_x,
                            0,
                            storey_level + building_config.storey_height,
                        ),
                        building_config.storey_height,
                        back_zone,
                        building_config.distance_to_neighbour[1] == 0,
                    )

                else:
                    idf = add_internal_wall(
                        idf,
                        s + 1,
                        "North",
                        building_config.length_wall_x,
                        (
                            building_config.length_wall_x,
                            building_config.length_wall_y * storey_split_ratio_flip,
                            storey_level + building_config.storey_height,
                        ),
                        building_config.storey_height,
                        front_zone,
                        back_zone,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "North",
                        building_config.length_wall_x,
                        (
                            building_config.length_wall_x,
                            building_config.length_wall_y,
                            storey_level + building_config.storey_height,
                        ),
                        building_config.storey_height,
                        back_zone,
                        building_config.distance_to_neighbour[0] == 0,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "East",
                        building_config.length_wall_y * (1 - storey_split_ratio_flip),
                        (
                            building_config.length_wall_x,
                            building_config.length_wall_y * storey_split_ratio_flip,
                            storey_level + building_config.storey_height,
                        ),
                        building_config.storey_height,
                        back_zone,
                        building_config.distance_to_neighbour[1] == 0,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "East",
                        building_config.length_wall_y * storey_split_ratio_flip,
                        (
                            building_config.length_wall_x,
                            0,
                            storey_level + building_config.storey_height,
                        ),
                        building_config.storey_height,
                        front_zone,
                        building_config.distance_to_neighbour[1] == 0,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "South",
                        building_config.length_wall_x,
                        (
                            0,
                            0,
                            storey_level + building_config.storey_height,
                        ),
                        building_config.storey_height,
                        front_zone,
                        building_config.distance_to_neighbour[2] == 0,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "West",
                        building_config.length_wall_y * (1 - storey_split_ratio_flip),
                        (
                            0,
                            building_config.length_wall_y,
                            storey_level + building_config.storey_height,
                        ),
                        building_config.storey_height,
                        back_zone,
                        building_config.distance_to_neighbour[3] == 0,
                    )
                    idf = add_external_wall(
                        idf,
                        s + 1,
                        "West",
                        building_config.length_wall_y * storey_split_ratio_flip,
                        (
                            0,
                            building_config.length_wall_y * storey_split_ratio_flip,
                            storey_level + building_config.storey_height,
                        ),
                        building_config.storey_height,
                        front_zone,
                        building_config.distance_to_neighbour[3] == 0,
                    )

        # add subfloor if present
        if building_config.subfloor_height > 0:
            zone = "Subfloor"
            idf.newidfobject(
                "ZONE",
                Name=zone,
            )

            idf = add_external_wall(
                idf,
                -1,
                "North",
                building_config.length_wall_x,
                (
                    building_config.length_wall_x,
                    building_config.length_wall_y,
                    0,
                ),
                building_config.subfloor_height,
                zone,
                building_config.distance_to_neighbour[0] == 0,
            )
            idf = add_external_wall(
                idf,
                -1,
                "East",
                building_config.length_wall_y,
                (
                    building_config.length_wall_x,
                    0,
                    0,
                ),
                building_config.subfloor_height,
                zone,
                building_config.distance_to_neighbour[1] == 0,
            )
            idf = add_external_wall(
                idf,
                -1,
                "South",
                building_config.length_wall_x,
                (
                    0,
                    0,
                    0,
                ),
                building_config.subfloor_height,
                zone,
                building_config.distance_to_neighbour[2] == 0,
            )
            idf = add_external_wall(
                idf,
                -1,
                "West",
                building_config.length_wall_y,
                (
                    0,
                    building_config.length_wall_y,
                    0,
                ),
                building_config.subfloor_height,
                zone,
                building_config.distance_to_neighbour[3] == 0,
            )
            idf = add_floor(
                idf,
                -1,
                0,
                building_config.length_wall_x,
                0,
                building_config.length_wall_y,
                -building_config.subfloor_height,
                "Subfloor",
                "Ground",
            )

        # add ground floor to the idf
        if building_config.subfloor_height > 0:
            under_ground_floor = "SubFloor"
        else:
            under_ground_floor = "Ground"

        if zone_split_storey != 0:
            idf = add_floor(
                idf,
                1,
                0,
                building_config.length_wall_x,
                0,
                building_config.length_wall_y,
                building_config.distance_to_ground,
                "Living",
                under_ground_floor,
            )
        else:
            if storey_split_in_x:
                idf = add_floor(
                    idf,
                    1,
                    0,
                    building_config.length_wall_x * storey_split_ratio_flip,
                    0,
                    building_config.length_wall_y,
                    building_config.distance_to_ground,
                    front_zone,
                    under_ground_floor,
                )
                idf = add_floor(
                    idf,
                    1,
                    building_config.length_wall_x * storey_split_ratio_flip,
                    building_config.length_wall_x,
                    0,
                    building_config.length_wall_y,
                    building_config.distance_to_ground,
                    back_zone,
                    under_ground_floor,
                )
            else:
                idf = add_floor(
                    idf,
                    1,
                    0,
                    building_config.length_wall_x,
                    0,
                    building_config.length_wall_y * storey_split_ratio_flip,
                    building_config.distance_to_ground,
                    front_zone,
                    under_ground_floor,
                )
                idf = add_floor(
                    idf,
                    1,
                    0,
                    building_config.length_wall_x,
                    building_config.length_wall_y * storey_split_ratio_flip,
                    building_config.length_wall_y,
                    building_config.distance_to_ground,
                    back_zone,
                    under_ground_floor,
                )

        # add internal floors
        for s in range(1, building_config.number_of_stories):
            if s < zone_split_storey:
                idf = add_floor(
                    idf,
                    s + 1,
                    0,
                    building_config.length_wall_x,
                    0,
                    building_config.length_wall_y,
                    building_config.distance_to_ground
                    + s * building_config.storey_height,
                    "Living",
                    "Living",
                )
            elif s == zone_split_storey:  # storey below is all living area
                if storey_split_in_x:
                    idf = add_floor(
                        idf,
                        s + 1,
                        0,
                        building_config.length_wall_x * storey_split_ratio_flip,
                        0,
                        building_config.length_wall_y,
                        building_config.distance_to_ground
                        + s * building_config.storey_height,
                        front_zone,
                        "Living",
                    )
                    idf = add_floor(
                        idf,
                        s + 1,
                        building_config.length_wall_x * storey_split_ratio_flip,
                        building_config.length_wall_x,
                        0,
                        building_config.length_wall_y,
                        building_config.distance_to_ground
                        + s * building_config.storey_height,
                        back_zone,
                        "Living",
                    )
                else:
                    idf = add_floor(
                        idf,
                        s + 1,
                        0,
                        building_config.length_wall_x,
                        0,
                        building_config.length_wall_y * storey_split_ratio_flip,
                        building_config.distance_to_ground
                        + s * building_config.storey_height,
                        front_zone,
                        "Living",
                    )
                    idf = add_floor(
                        idf,
                        s + 1,
                        0,
                        building_config.length_wall_x,
                        building_config.length_wall_y * storey_split_ratio_flip,
                        building_config.length_wall_y,
                        building_config.distance_to_ground
                        + s * building_config.storey_height,
                        back_zone,
                        "Living",
                    )
            elif s - 1 == zone_split_storey:  # the storey below is split
                if storey_split_in_x:
                    idf = add_floor(
                        idf,
                        s + 1,
                        0,
                        building_config.length_wall_x * storey_split_ratio_flip,
                        0,
                        building_config.length_wall_y,
                        building_config.distance_to_ground
                        + s * building_config.storey_height,
                        "Bedroom",
                        front_zone,
                    )
                    idf = add_floor(
                        idf,
                        s + 1,
                        building_config.length_wall_x * storey_split_ratio_flip,
                        building_config.length_wall_x,
                        0,
                        building_config.length_wall_y,
                        building_config.distance_to_ground
                        + s * building_config.storey_height,
                        "Bedroom",
                        back_zone,
                    )
                else:
                    idf = add_floor(
                        idf,
                        s + 1,
                        0,
                        building_config.length_wall_x,
                        0,
                        building_config.length_wall_y * storey_split_ratio_flip,
                        building_config.distance_to_ground
                        + s * building_config.storey_height,
                        "Bedroom",
                        front_zone,
                    )
                    idf = add_floor(
                        idf,
                        s + 1,
                        0,
                        building_config.length_wall_x,
                        building_config.length_wall_y * storey_split_ratio_flip,
                        building_config.length_wall_y,
                        building_config.distance_to_ground
                        + s * building_config.storey_height,
                        "Bedroom",
                        back_zone,
                    )

            else:  # the storey split is more than one storey below
                idf = add_floor(
                    idf,
                    s + 1,
                    0,
                    building_config.length_wall_x,
                    0,
                    building_config.length_wall_y,
                    building_config.distance_to_ground
                    + s * building_config.storey_height,
                    "Bedroom",
                    "Bedroom",
                )

        # add flat roof (saddleback and loft in a second step)
        roof_level = (
            building_config.distance_to_ground
            + building_config.number_of_stories * building_config.storey_height
        )
        if zone_split_storey != building_config.number_of_stories - 1:
            idf = add_flat_roof(
                idf,
                0,
                building_config.length_wall_x,
                0,
                building_config.length_wall_y,
                roof_level,
                "Bedroom",
            )
        else:
            if storey_split_in_x:
                idf = add_flat_roof(
                    idf,
                    0,
                    building_config.length_wall_x * storey_split_ratio_flip,
                    0,
                    building_config.length_wall_y,
                    roof_level,
                    front_zone,
                )
                idf = add_flat_roof(
                    idf,
                    building_config.length_wall_x * storey_split_ratio_flip,
                    building_config.length_wall_x,
                    0,
                    building_config.length_wall_y,
                    roof_level,
                    back_zone,
                )
            else:
                idf = add_flat_roof(
                    idf,
                    0,
                    building_config.length_wall_x,
                    0,
                    building_config.length_wall_y * storey_split_ratio_flip,
                    roof_level,
                    front_zone,
                )
                idf = add_flat_roof(
                    idf,
                    0,
                    building_config.length_wall_x,
                    building_config.length_wall_y * storey_split_ratio_flip,
                    building_config.length_wall_y,
                    roof_level,
                    back_zone,
                )

        if building_config.roof_type == RoofType.SADDLEBACK.value:
            if building_config.loft_is_heated:
                idf = add_saddleback_roof(idf, building_config, "Bedroom")
            else:
                idf = add_saddleback_roof(idf, building_config, "Loft")

    elif building_config.zoning == Zoning.CUSTOM.value:

        for storey in range(building_config.number_of_stories):
            for i, zone in enumerate(building_config.zone_names[storey]):
                # add zone as a block
                idf.add_block(
                    name=zone,
                    coordinates=[
                        building_config.zone_coords[storey][i][0],
                        building_config.zone_coords[storey][i][1],
                        building_config.zone_coords[storey][i][2],
                        building_config.zone_coords[storey][i][3],
                    ],
                    height=building_config.storey_height,
                )

                idf.idfobjects["ZONE"][-1].Name = zone

                for sf in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
                    if zone in sf.Name:
                        sf.Zone_Name = zone

                # check if zone does not belong on ground floor
                # if True move z coordinate of zone by a height adjustment
                if storey > 0:
                    # adjust height of zone
                    height_adjustment = storey * building_config.storey_height

                    for sf in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
                        if zone in sf.Name:

                            sf.Vertex_1_Zcoordinate = (
                                sf.Vertex_1_Zcoordinate + height_adjustment
                            )
                            sf.Vertex_2_Zcoordinate = (
                                sf.Vertex_2_Zcoordinate + height_adjustment
                            )
                            sf.Vertex_3_Zcoordinate = (
                                sf.Vertex_3_Zcoordinate + height_adjustment
                            )
                            sf.Vertex_4_Zcoordinate = (
                                sf.Vertex_4_Zcoordinate + height_adjustment
                            )

                # calculate zone area
                if zone not in area_per_zone:
                    area_per_zone[zone] = calculate_zone_area(
                        building_config.zone_coords[storey][i]
                    )

        # TODO add in subfloor for each ground floor

        # check if we need a subfloor zone
        # if True, add floor and external walls
        if building_config.subfloor_height > 0:
            zone = "Subfloor"

            idf.newidfobject(
                "ZONE",
                Name=zone,
            )

            idf = add_floor(
                idf,
                -1,
                0,
                building_config.length_wall_x,
                0,
                building_config.length_wall_y,
                -building_config.subfloor_height,
                "Subfloor",
                "Ground",
            )
            idf = add_external_wall(
                idf,
                -1,
                "North",
                building_config.length_wall_x,
                (
                    building_config.length_wall_x,
                    building_config.length_wall_y,
                    0,
                ),
                building_config.subfloor_height,
                zone,
                building_config.distance_to_neighbour[0] == 0,
            )

            idf = add_external_wall(
                idf,
                -1,
                "East",
                building_config.length_wall_y,
                (
                    building_config.length_wall_x,
                    0,
                    0,
                ),
                building_config.subfloor_height,
                zone,
                building_config.distance_to_neighbour[1] == 0,
            )
            idf = add_external_wall(
                idf,
                -1,
                "South",
                building_config.length_wall_x,
                (
                    0,
                    0,
                    0,
                ),
                building_config.subfloor_height,
                zone,
                building_config.distance_to_neighbour[2] == 0,
            )
            idf = add_external_wall(
                idf,
                -1,
                "West",
                building_config.length_wall_y,
                (
                    0,
                    building_config.length_wall_y,
                    0,
                ),
                building_config.subfloor_height,
                zone,
                building_config.distance_to_neighbour[3] == 0,
            )

    elif building_config.zoning == Zoning.LEEDR_H28_ZONING.value:

        zone_info = {
            "zone": [],
            "storey": [],
            "zone_bottom_left": [],
            "zone_top_right": [],
            "zone_coords": [],
            "wall": [],
            "floor": [],
        }

        # go through zones, storey by story
        for storey in range(building_config.number_of_stories):
            for i, zone in enumerate(building_config.zone_names[storey]):

                # add zone
                idf.newidfobject(
                    "ZONE",
                    Name=zone,
                )

                # calculating area per zone
                bottom_left_coord = building_config.zone_coords[storey][i][0]
                top_right_coord = building_config.zone_coords[storey][i][1]

                zone_info["storey"].append(storey)
                zone_info["zone_bottom_left"].append(bottom_left_coord)
                zone_info["zone_top_right"].append(top_right_coord)

                if zone not in area_per_zone:

                    area_per_zone[zone] = calculate_coordinate_area(
                        bottom_left_coord[0],
                        bottom_left_coord[1],
                        top_right_coord[0],
                        top_right_coord[1],
                    )

                zone_info["zone"].append(zone)
                zone_info["zone_coords"].append(
                    get_zone_coords(bottom_left_coord, top_right_coord)
                )

        zone_info = get_floor_information(
            zone_info,
            building_config.distance_to_ground,
            building_config.subfloor_height,
            building_config.storey_height,
        )

        # adding floors
        for i, floor in enumerate(zone_info["floor"][0]["xmin"]):
            idf = add_floor(
                idf,
                zone_info["storey"][i],
                floor,
                zone_info["floor"][0]["xmax"][i],
                zone_info["floor"][0]["ymin"][i],
                zone_info["floor"][0]["ymax"][i],
                zone_info["floor"][0]["distance_from_ground"][i],
                zone_info["floor"][0]["inner_zone"][i],
                zone_info["floor"][0]["outer_zone"][i],
            )

        # add subfloor zone
        if building_config.subfloor_height > 0:
            zone = "Subfloor"

            idf.newidfobject(
                "ZONE",
                Name=zone,
            )

            idf = add_floor(
                idf,
                -1,
                0,
                building_config.length_wall_x,
                0,
                building_config.length_wall_y,
                -building_config.subfloor_height,
                "Subfloor",
                "Ground",
            )

            idf = add_external_wall(
                idf,
                -1,
                "North",
                building_config.length_wall_x,
                (
                    building_config.length_wall_x,
                    building_config.length_wall_y,
                    0,
                ),
                building_config.subfloor_height,
                zone,
                building_config.distance_to_neighbour[0] == 0,
            )

            idf = add_external_wall(
                idf,
                -1,
                "East",
                building_config.length_wall_y,
                (
                    building_config.length_wall_x,
                    0,
                    0,
                ),
                building_config.subfloor_height,
                zone,
                building_config.distance_to_neighbour[1] == 0,
            )
            idf = add_external_wall(
                idf,
                -1,
                "South",
                building_config.length_wall_x,
                (
                    0,
                    0,
                    0,
                ),
                building_config.subfloor_height,
                zone,
                building_config.distance_to_neighbour[2] == 0,
            )
            idf = add_external_wall(
                idf,
                -1,
                "West",
                building_config.length_wall_y,
                (
                    0,
                    building_config.length_wall_y,
                    0,
                ),
                building_config.subfloor_height,
                zone,
                building_config.distance_to_neighbour[3] == 0,
            )

        # get wall information, internal and external
        zone_info = get_zone_walls_information(zone_info)

        wall_info = identify_unique_walls(zone_info)

        zone_info, wall_info = remove_non_unique_wall(zone_info, wall_info)

        for wall in wall_info:
            storey = wall[-1]
            # upper left corner tuple is only 2D, adding in Z axis below
            upper_left_corner = wall[-2]
            upper_left_corner = tuple(
                list(upper_left_corner) + [(storey + 1) * building_config.storey_height]
            )
            direction = wall[2]
            length = wall[3]

            zone_index = wall[1][0]
            zone = zone_info["zone"][zone_index]
            back_zone_index = wall[1][1]

            if back_zone_index == "external":
                # add external walls
                idf = add_external_wall(
                    idf,
                    storey,
                    direction,
                    length,
                    upper_left_corner,
                    building_config.storey_height,
                    zone,
                    building_config.distance_to_neighbour[i] == 0,
                )
            # wall must be internal
            else:
                back_zone = zone_info["zone"][back_zone_index]

                idf = add_internal_wall(
                    idf,
                    storey,
                    direction,
                    length,
                    upper_left_corner,
                    building_config.storey_height,
                    zone,
                    back_zone,
                )

        # add flat roof
        for i, zone in enumerate(zone_info["zone"]):
            roof_level = (
                building_config.distance_to_ground
                + building_config.number_of_stories * building_config.storey_height
            )
            if zone_info["storey"][i] + 1 == building_config.number_of_stories:

                idf = add_flat_roof(
                    idf,
                    zone_info["zone_bottom_left"][i][0],
                    zone_info["zone_top_right"][i][0],
                    zone_info["zone_bottom_left"][i][1],
                    zone_info["zone_top_right"][i][1],
                    roof_level,
                    zone,
                )

        if building_config.roof_type == RoofType.SADDLEBACK.value:
            if building_config.loft_is_heated:
                idf = add_saddleback_roof(idf, building_config, "Bedroom")
            else:
                idf = add_saddleback_roof(idf, building_config, "Loft")

        elif building_config.roof_type == RoofType.ADIABATIC.value:
            idf = change_roof_to_adiabatic(idf)

    return idf, area_per_zone


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
        "BuildingSurface:Detailed".upper(),
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
        Vertex_1_Xcoordinate=coords["X1"],
        Vertex_1_Ycoordinate=coords["Y1"],
        Vertex_1_Zcoordinate=coords["Z1"],
        Vertex_2_Xcoordinate=coords["X2"],
        Vertex_2_Ycoordinate=coords["Y2"],
        Vertex_2_Zcoordinate=coords["Z2"],
        Vertex_3_Xcoordinate=coords["X3"],
        Vertex_3_Ycoordinate=coords["Y3"],
        Vertex_3_Zcoordinate=coords["Z3"],
        Vertex_4_Xcoordinate=coords["X4"],
        Vertex_4_Ycoordinate=coords["Y4"],
        Vertex_4_Zcoordinate=coords["Z4"],
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
        "BuildingSurface:Detailed".upper(),
        Name="Storey " + str(storey) + " " + direction + " Wall " + zone,
        Zone_Name=zone,
        Surface_Type="Wall",
        Construction_Name="Wall",
        View_Factor_to_Ground=0.5,
        Number_of_Vertices=4,
        Vertex_1_Xcoordinate=coords["X1"],
        Vertex_1_Ycoordinate=coords["Y1"],
        Vertex_1_Zcoordinate=coords["Z1"],
        Vertex_2_Xcoordinate=coords["X2"],
        Vertex_2_Ycoordinate=coords["Y2"],
        Vertex_2_Zcoordinate=coords["Z2"],
        Vertex_3_Xcoordinate=coords["X3"],
        Vertex_3_Ycoordinate=coords["Y3"],
        Vertex_3_Zcoordinate=coords["Z3"],
        Vertex_4_Xcoordinate=coords["X4"],
        Vertex_4_Ycoordinate=coords["Y4"],
        Vertex_4_Zcoordinate=coords["Z4"],
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

    if inner_zone != outer_zone:

        if outer_zone == "Ground":
            if inner_zone == "Subfloor":
                c_name = "SubFloor"
            else:
                c_name = "GroundFloor"
        else:
            c_name = "Floor"

        idf.newidfobject(
            "BuildingSurface:Detailed".upper(),
            Name="Storey "
            + str(storey)
            + " "
            + " Floor "
            + inner_zone
            + "-"
            + outer_zone,
            Surface_Type="Floor",
            Zone_Name=inner_zone,
            Construction_Name=c_name,
            View_Factor_to_Ground=1.0,
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
            Sun_Exposure="NoSun",
            Wind_Exposure="NoWind",
        )
        floor = idf.idfobjects["BuildingSurface:Detailed".upper()][-1]

        if outer_zone == "Ground" and distance_from_ground > 0:
            floor.Outside_Boundary_Condition = "Adiabatic"
        elif outer_zone == "Ground":
            floor.Outside_Boundary_Condition = "Ground"
        else:
            floor.Outside_Boundary_Condition = "Zone"
            floor.Outside_Boundary_Condition_Object = outer_zone

    else:
        idf.newidfobject(
            "INTERNALMASS",
            Name=(
                "IntMass-Floor-"
                + inner_zone
                + "-"
                + outer_zone
                + "-storey-"
                + str(storey)
            ),
            Construction_Name="Floor",
            Zone_or_ZoneList_Name=inner_zone,
            Surface_Area=(xmax - xmin) * (ymax - ymin),
        )
        idf.newidfobject(
            "INTERNALMASS",
            Name=(
                "IntMass-Ceiling-"
                + inner_zone
                + "-"
                + outer_zone
                + "-storey-"
                + str(storey)
            ),
            Construction_Name="Ceiling",
            Zone_or_ZoneList_Name=outer_zone,
            Surface_Area=(xmax - xmin) * (ymax - ymin),
        )

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


def add_strip_window_on_wall(idf: IDF, wwr: float, wall):

    p1 = np.array(
        [
            wall.Vertex_1_Xcoordinate,
            wall.Vertex_1_Ycoordinate,
            wall.Vertex_1_Zcoordinate,
        ]
    )
    p2 = np.array(
        [
            wall.Vertex_2_Xcoordinate,
            wall.Vertex_2_Ycoordinate,
            wall.Vertex_2_Zcoordinate,
        ]
    )
    p3 = np.array(
        [
            wall.Vertex_3_Xcoordinate,
            wall.Vertex_3_Ycoordinate,
            wall.Vertex_3_Zcoordinate,
        ]
    )
    p4 = np.array(
        [
            wall.Vertex_4_Xcoordinate,
            wall.Vertex_4_Ycoordinate,
            wall.Vertex_4_Zcoordinate,
        ]
    )
    w1 = p1 + (1 - wwr) / 2 * (p2 - p1)
    w2 = p2 + (1 - wwr) / 2 * (p1 - p2)
    w3 = p3 + (1 - wwr) / 2 * (p4 - p3)
    w4 = p4 + (1 - wwr) / 2 * (p3 - p4)

    idf.newidfobject(
        "FENESTRATIONSURFACE:DETAILED",
        Name=wall.Name + "-Window",
        Surface_Type="Window",
        Construction_Name="Window-Construction",
        Building_Surface_Name=wall.Name,
        View_Factor_to_Ground="autocalculate",
        Number_of_Vertices=4,
        Vertex_1_Xcoordinate=w1[0],
        Vertex_1_Ycoordinate=w1[1],
        Vertex_1_Zcoordinate=w1[2],
        Vertex_2_Xcoordinate=w2[0],
        Vertex_2_Ycoordinate=w2[1],
        Vertex_2_Zcoordinate=w2[2],
        Vertex_3_Xcoordinate=w3[0],
        Vertex_3_Ycoordinate=w3[1],
        Vertex_3_Zcoordinate=w3[2],
        Vertex_4_Xcoordinate=w4[0],
        Vertex_4_Ycoordinate=w4[1],
        Vertex_4_Zcoordinate=w4[2],
    )

    return idf


def add_gable_window_on_triangular_wall(idf: IDF, wwr: float, wall):

    p1 = np.array(
        [
            wall.Vertex_1_Xcoordinate,
            wall.Vertex_1_Ycoordinate,
            wall.Vertex_1_Zcoordinate,
        ]
    )
    p2 = np.array(
        [
            wall.Vertex_2_Xcoordinate,
            wall.Vertex_2_Ycoordinate,
            wall.Vertex_2_Zcoordinate,
        ]
    )
    p3 = np.array(
        [
            wall.Vertex_3_Xcoordinate,
            wall.Vertex_3_Ycoordinate,
            wall.Vertex_3_Zcoordinate,
        ]
    )
    c = 1 / 3 * (p1 + p2 + p3)
    area = 1 / 2 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
    a = np.sqrt(area * wwr) / 2
    i = (p3 - p2) / (np.linalg.norm(p3 - p2))
    j = (p1 - 1 / 2 * (p3 + p2)) / (np.linalg.norm(p1 - 1 / 2 * (p3 + p2)))

    w1 = c - a * i + a * j
    w2 = c - a * i - a * j
    w3 = c + a * i - a * j
    w4 = c + a * i + a * j

    idf.newidfobject(
        "FENESTRATIONSURFACE:DETAILED",
        Name=wall.Name + "-Window",
        Surface_Type="Window",
        Construction_Name="Window-Construction",
        Building_Surface_Name=wall.Name,
        View_Factor_to_Ground="autocalculate",
        Number_of_Vertices=4,
        Vertex_1_Xcoordinate=w1[0],
        Vertex_1_Ycoordinate=w1[1],
        Vertex_1_Zcoordinate=w1[2],
        Vertex_2_Xcoordinate=w2[0],
        Vertex_2_Ycoordinate=w2[1],
        Vertex_2_Zcoordinate=w2[2],
        Vertex_3_Xcoordinate=w3[0],
        Vertex_3_Ycoordinate=w3[1],
        Vertex_3_Zcoordinate=w3[2],
        Vertex_4_Xcoordinate=w4[0],
        Vertex_4_Ycoordinate=w4[1],
        Vertex_4_Zcoordinate=w4[2],
    )

    return idf


def get_surface_height(surface):
    """gives height of surface (z length).
    only works for surfaces along z axis"""

    z_coordinates = [
        surface.Vertex_1_Zcoordinate,
        surface.Vertex_2_Zcoordinate,
        surface.Vertex_3_Zcoordinate,
        surface.Vertex_4_Zcoordinate,
    ]
    return max(z_coordinates) - min(z_coordinates)


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
