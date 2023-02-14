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
    n_storey: int
    # counterclockwise, viewed from the top,
    # order: north, east, south, west
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
    attic_is_heated: bool

    # if this is 0: y is North, x is East.rotation round inverse z-axis
    rotation: float

    zones_per_storey: int  # 0 means whole building is same zone

    location: str
    terrain: str

    ground_floor_layer_materials: List[str]
    ground_floor_layer_thickness: List[float]
    upper_floor_layer_materials: List[str]
    upper_floor_layer_thickness: List[float]
    wall_layer_materials: List[str]
    wall_layer_thickness: List[float]
    roof_layer_materials: List[str]
    roof_layer_thickness: List[float]
    # if these are empty then upper floor values are used:
    attic_floor_layer_materials: List[str]
    attic_floor_layer_thickness: List[float]
    partition_layer_materials: List[str]
    partition_layer_thickness: List[float]
    partition_area_per_zone: float

    window_type: str
    window_layer_materials: List[str]
    window_layer_thickness: List[float]

    window_shading_device: str
    window_shading_outside: bool
    window_shading_control: str

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
    cooling_system_dimension: str
    cooling_system_fuel: str
    cooling_system_efficiency: float

    # ventilation
    # this is for additional ventilation to avoid overheating
    natvent_for_cooling_calculation_method: str
    natvent_for_cooling_rate: float
    natvent_for_cooling_indoor_t_range: Tuple[float, float]
    # this is constant ventilation to have enough fresh air
    ventilation_for_air_calculation_method: str
    ventilation_for_air_rate: float
    ventilation_for_air_fan_pressure_rise: float
    ventilation_for_air_fan_efficiency: float
    ventilation_for_air_heat_recovery_efficiency: float
    # this is an alternative mode of ventilation: opening windows
    window_opening_schedule: str

    # infiltration
    infiltration_calculation_method: str
    infiltration_rate: float

    # occupants + internal gains
    occupant_number_calculation_method: str
    occupant_value: float
    occupant_schedule: str
    equipment_gain_calculation_method: str
    equipment_gain_value: float
    equipment_gain_schedule: str
    lighting_power_calculation_method: str
    lighting_power_value: float
    lighting_schedule: str

    # setpoint schedules
    heating_setpoint: float
    heating_setback: float
    heating_setpoint_schedule: str
    cooling_setpoint: float
    cooling_setback: float
    cooling_setpoint_schedule: str

    def save_to_file(self, path_to_datafile):
        with open(path_to_datafile, "w", encoding="utf-8") as out_file:
            json.dump(asdict(self), out_file, indent=4)


def load_building_config(path_to_datafile):
    """this function takes a json file and returns a BuildingConfig object"""
    with open(path_to_datafile, encoding="utf-8") as file:
        data = json.loads(file.read())

    tuple_names = [
        "wtw_ratios",
        "distance_to_neighbour",
        "natvent_for_cooling_indoor_t_range",
    ]
    for tn in tuple_names:
        data[tn] = tuple(data[tn])

    return from_dict(data_class=BuildingConfig, data=data)
