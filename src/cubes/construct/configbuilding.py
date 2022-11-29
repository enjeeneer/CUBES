"""This module defines the variables required to generally describe a building and
    stores these variables in a building_data dataclass"""


from dataclasses import dataclass


@dataclass
class BuildingConfig:
    """
    Class which holds data describing a building
    """

    name: str
    a_ground_floor: float
    a_wall: float
    n_storey: int
    a_window: float
    r_floor_roof: float
    h_ceiling: float
    l_wall_front: float
    l_wall_side: float
    h_roof: float

    wall_layer_materials: list
    wall_layer_thickness: list
    roof_construction: list
    floor_construction: list
    ceiling_construction: list
    window_construction: list

    heating_system_efficiency: float
    heating_system_technology: str
    heating_system_fuel: str

    location: str

    occupant_number: int
    occupant_zone: str
    occupant_schedule: str
