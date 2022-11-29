"""This module defines the variables required to generally describe a building and
    stores these variables in a building_data dataclass"""


from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class BuildingConfig:
    """
    Class which holds data describing a building
    """

    name: str
    # a_ground_floor: float # I don't think we need this if we have side lengths
    # a_wall: float # I don't think we need this if we have side lengths and height
    n_storey: int
    # counterclockwise from the top starting with the wall with lower x and lower y
    wtw_ratios: Tuple[float, float, float, float]
    distance_to_neighbour: Tuple[float, float, float, float]
    r_floor_roof: float
    h_storey: float
    l_wall_x: float
    l_wall_y: float
    h_roof: float
    # if this is 0: y is North, x is East.rotation round inverse z-axis
    rotation: float

    zones_per_storey: int  # 0 means whole building is same zone
    location: str  # city or longitude + latitude
    terrain: str

    wall_layer_materials: List[str]
    wall_layer_thickness: List[float]
    roof_layer_materials: List[str]
    roof_layer_thickness: List[float]
    upper_floor_layer_materials: List[str]
    upper_floor_layer_thickness: List[float]
    ground_floor_layer_materials: List[str]
    ground_floor_layer_thickness: List[float]
    window_layer_materials: List[str]
    window_layer_thicknesses: List[float]
    window_shading_type: str

    # heating system
    heating_system_efficiency: float
    heating_system_type: str
    heating_system_fuel: str

    # cooling system
    cooling_system_type: str

    # ventilation
    natural_ventilation: bool
    mechanical_ventilation: bool
    mechanical_ventilation_heat_recovery: float
    ventilation_fan_power: float

    # infiltration
    infiltration_per_area_50pa: float

    # occupants + internal gains
    occupant_number_max: int
    occupant_schedule: str  # could have some fixed schedules or stochastic models
    equipment_gain_type: str  # floor area or occupant or zone
    equipment_gain_value: float
    lighting_power: float
    window_shading_control: str  # need to define a rule
