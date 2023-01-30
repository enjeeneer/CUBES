"""This module defines the variables required to generally describe a building and
    stores these variables in a building_data dataclass"""


from dataclasses import dataclass, asdict
from typing import List, Tuple
import json
from dacite import from_dict


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
    # set to -1 if neighbours should be neglected, set to 0 if attached to neighbour
    distance_to_neighbour: Tuple[float, float, float, float]

    h_storey: float
    l_wall_x: float
    l_wall_y: float

    # roof
    # for the roof we may only need to specify what type of roof it is
    # i.e. flat/saddleback and the roof height and then in building.py the coords
    # are determined by the get_roof_coords method?
    roof_type: str
    h_roof: float
    # r_floor_roof: float #Unsure if needed
    # roof_coordinates = Tuple[float, float, float, float] #unsure if needed
    # roof_wall_coordinates = Tuple[float, float, float, float] #unsure if needed

    # if this is 0: y is North, x is East.rotation round inverse z-axis
    rotation: float

    zones_per_storey: int  # 0 means whole building is same zone
    location: str  # added back in 27/1/23 by Jack
    terrain: str

    ground_floor_layer_materials: List[str]
    ground_floor_layer_thickness: List[float]
    upper_floor_layer_materials: List[str]
    upper_floor_layer_thickness: List[float]
    wall_layer_materials: List[str]
    wall_layer_thickness: List[float]
    roof_layer_materials: List[str]
    roof_layer_thickness: List[float]

    window_type: str
    window_layer_materials: List[str]
    window_layer_thickness: List[float]

    window_shading_device: str  # new

    # heating system
    heating_system_type: str
    heating_system_dimension: str
    heating_system_fuel: str
    heating_system_efficiency: float

    # domestic hot water system
    dhw_system_type: str
    dhw_system_dimension: str
    dhw_system_fuel: str
    dhw_system_efficiency: float

    # cooling system
    cooling_system_type: str

    # ventilation
    natural_ventilation: bool
    mechanical_ventilation: bool
    mech_ventilation_heat_recovery: float
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

    def save_to_file(self, path_to_datafile):
        with open(path_to_datafile, "w", encoding="utf-8") as out_file:
            json.dump(asdict(self), out_file, indent=4)


def load_building_config(path_to_datafile):
    """this function takes a json file and returns a BuildingConfig object"""
    with open(path_to_datafile, encoding="utf-8") as file:
        data = json.loads(file.read())
    data["wtw_ratios"] = tuple(data["wtw_ratios"])
    data["distance_to_neighbour"] = tuple(data["distance_to_neighbour"])

    return from_dict(data_class=BuildingConfig, data=data)
