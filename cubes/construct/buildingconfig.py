"""This module defines the variables required to generally describe a building and
    stores these variables in a building_data dataclass"""


from dataclasses import dataclass, asdict
from typing import List, Tuple, Any, Optional
import json
from dacite import from_dict
from os.path import dirname, join

import cubes.construct.buildingconfig_options as bco


@dataclass
class BuildingConfig:
    """
    Class which holds data describing a building
    """

    name: str
    number_of_stories: int
    files_dir: str
    # counterclockwise, viewed from the top,
    # order: north, east, south, west
    wtw_ratios: Tuple[float, float, float, float]
    wtw_ratios_loft: Tuple[float, float, float, float]
    # set to -1 if neighbours should be neglected, set to 0 if attached to neighbour
    distance_to_neighbour: Tuple[float, float, float, float]

    storey_height: float
    length_wall_x: float
    length_wall_y: float
    # this is non-zero for flats which are not on the ground floor
    distance_to_ground: float

    # roof
    # for the roof we may only need to specify what type of roof it is
    # i.e. flat/saddleback and the roof height and then in building.py the coords
    # are determined by the get_roof_coords method?
    roof_type: str
    roof_height: float
    roof_ridge_along_x: bool
    loft_is_heated: bool
    rotation: float  # if this is 0: y is North, x is East.
    # rotation around inverse z-axis    zoning: str
    zoning: str

    zone_names: List[List[str]]
    zone_coords: List[List]
    year: int

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
    partition_wall_area_per_floor_area: float  # 1.6666 in CODE

    # thermal mass allowance for furniture
    furniture_thermal_mass_per_floor_area: float  # in kJ/(K m^2) 30 in CODE
    furniture_material: str
    furniture_thickness: float

    # this is for an optional subfloor (model for airspace below groundfloor)
    subfloor_height: float
    subfloor_layer_materials: List[str]
    subfloor_layer_thickness: List[float]
    subfloor_infiltration_ach: float

    window_type: str
    window_layer_materials: List[str]
    window_layer_thickness: List[float]
    window_simple_values: Optional[
        Tuple[float, float, float]
    ]  # U_factor(incl film), SHGC, Visible Transmittance

    window_shading_device: str
    window_shading_outside: bool
    window_shading_control: str

    # heating and cooling systems
    # heating_water...?
    heating_water_loop_dimension: str  # = "building"
    heating_water_loop_equipment_fuel: str  # = "naturalgas"
    heating_water_loop_equipment: str  # = "condensing boiler"
    heating_water_loop_equipment_efficiency: float  # = 0.9
    heating_water_loop_temperature: float  # = 80  # °C
    heating_heat_pump_tank_volume: float  # 0.05 m^3
    heating_heat_pump_capacity: float  # = 8000 W

    zone_heating_equipment: str  # = "radiator"
    zone_heating_equipment_efficiency: float  # = 1.0

    cooling_system_installed: bool  # = False
    cooling_system_efficiency: float  # = 3.5

    # need to split heating from hot water...
    dhw_heating_loop_dimension: str
    dhw_heating_equipment_fuel: str
    dhw_heating_equipment: str
    dhw_heating_equipment_efficiency: float
    dhw_water_tank_volume: float  # m3 now per zone, should be per dwelling
    dhw_usage_schedule: str

    # ventilation
    ventilation_type: str
    natural_ventilation_method: str
    natural_ventilation_model: str
    ventilation_rate_per_occupant: float  # m3/person/s
    natural_ventilation_rate_open_windows: float  # in ach
    mech_ventilation_heat_recovery_efficiency_sensible: float
    mech_ventilation_heat_recovery_efficiency_latent: float
    mech_ventilation_fan_pressure_rise: float
    mech_ventilation_fan_efficiency: float

    # infiltration
    infiltration_calculation_method: str
    infiltration_rate: float

    # occupants + internal gains
    occupant_number_calculation_method: str
    occupant_value: float
    # comma-separated occupancy fractions in 10 min intervals
    occupant_schedule: List[List[str]]
    # occupant_schedule_living: str
    # occupant_schedule_bedroom: str
    # occupant_schedule_hall: str
    # occupant_schedule_lounge: str
    # occupant_schedule_kitchen: str
    # occupant_schedule_backroom: str
    equipment_gain_calculation_method: str
    equipment_gain_value: float
    equipment_gain_schedule: str
    lighting_power_calculation_method: str
    lighting_power_value: float
    lighting_schedule: str

    # PV and battery
    pv_present: bool
    pv_cell_efficiency: float
    pv_roof_area_ratio_primary: float
    pv_roof_area_ratio_secondary: float
    pv_active_area_fraction: float
    battery_energy_storage: float
    battery_power_rating: float

    # weather
    weather_file_name: str

    # grid
    grid_carbon_intensity_file_name: str

    # setpoint schedules
    use_operative_temperature: bool
    heating_setpoint: float
    heating_setback: float
    heating_setpoint_schedule: str
    cooling_setpoint: float
    cooling_setback: float
    cooling_setpoint_schedule: str

    # vehicle
    bev_present: bool = False
    phev_present: bool = False
    bev_battery_size: float = 0
    phev_battery_size: float = 0

    # refrigeration
    fridge_compressor_refrigerant: str = 0
    fridge_compressor_coefficient_of_performance: float = 0
    fridge_compressor_type: str = ""
    fridge_rack_rated_total_cooling_capacity: float = 0
    fridge_rack_case_length: float = 0
    fridge_rack_case_width: float = 0
    fridge_rack_case_height: float = 0
    fridge_rated_ambient_temperature: float = 0
    fridge_rated_ambient_relative_humidity: float = 0
    fridge_case_defrost_type: str = ""
    fridge_case_operating_temperature: float = 0
    freezer_compressor_refrigerant: str = ""
    freezer_compressor_coefficient_of_performance: float = 0
    freezer_compressor_type: str = ""
    freezer_rack_rated_total_cooling_capacity: float = 0
    freezer_rack_case_length: float = 0
    freezer_rack_case_width: float = 0
    freezer_rack_case_height: float = 0
    freezer_rated_ambient_temperature: float = 0
    freezer_rated_ambient_relative_humidity: float = 0
    freezer_case_defrost_type: str = ""
    freezer_case_operating_temperature: float = 0

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "heating_water_loop_dimension":
            assert (
                value.lower() in valid_dimensions
            ), f"{name} has to be one of {valid_dimensions}, but is {value}"
            self.__dict__[name] = value.lower()
        elif name == "heating_water_loop_equipment_fuel":
            assert (
                value.lower() in valid_fuels
            ), f"{name} has to be one of {valid_fuels}, but is {value}"
            assert value.lower() in implemented_fuels, (
                f"{value} not yet implemented as {name}."
                f" Please use one of {implemented_fuels}"
            )
            self.__dict__[name] = value.lower()
            # elif name == "heating_water_loop_equipment":
            #     assert value.lower() in valid_heating_water_loop_equipment, (
            #         f"{name} has to be one of {valid_heating_water_loop_equipment},"
            #         f"but is {value}"
            #     )
            #     assert value.lower() in implemented_heating_water_loop_equipment, (
            #         f"{value} not yet implemented as {name}."
            #         f"Please use one of {implemented_heating_water_loop_equipment}"
            #     )
            self.__dict__[name] = value.lower()
        elif name == "heating_water_loop_equipment_efficiency":
            assert value > 0, f"{name} has to be > 0, but is {value}"
            self.__dict__[name] = value
        elif name == "heating_water_loop_equipment_temperature":
            assert 0 >= value < 100, (
                f"{name} has to be in range (0,100)," f"but is {value}"
            )
            self.__dict__[name] = value
        elif name == "cooling_system_efficiency":
            assert value > 0, f"{name} has to be > 0, but is {value}"
            self.__dict__[name] = value
        else:
            self.__dict__[name] = value

        if name == "dhw_heating_loop_dimension":
            assert (
                value.lower() in valid_dimensions
            ), f"{name} has to be one of {valid_dimensions}, but is {value}"
            self.__dict__[name] = value.lower()
        elif name == "dhw_heating_equipment_fuel":
            assert (
                value.lower() in valid_fuels
            ), f"{name} has to be one of {valid_fuels}, but is {value}"
            assert value.lower() in implemented_fuels, (
                f"{value} not yet implemented as {name}."
                f" Please use one of {implemented_fuels}"
            )
            self.__dict__[name] = value.lower()
        elif name == "dhw_heating_equipment":
            assert (
                value.lower() in valid_dhw_heating_equipment
            ), f"{name} has to be one of {valid_dhw_heating_equipment},but is {value}"
            assert value.lower() in implemented_dhw_heating_equipment, (
                f"{value} not yet implemented as {name}."
                f"Please use one of {implemented_dhw_heating_equipment}"
            )
            self.__dict__[name] = value.lower()
        elif name == "dhw_heating_equipment_efficiency":
            assert value > 0, f"{name} has to be > 0, but is {value}"
            self.__dict__[name] = value

        elif name == "ventilation_type":
            assert (
                value.lower() in bco.VentilationType
            ), f"{name} has to be one of {bco.VentilationType.list()},but is {value}"
            assert value.lower() in bco.VentilationTypeImplemented, (
                f"{value} not yet implemented as {name}."
                f"Please use one of {bco.VentilationTypeImplemented.list()}"
            )
            self.__dict__[name] = value

        elif name == "ventilation_method":
            assert value.lower() in bco.VentilationMethod, (
                f"{name} has to be one of {bco.VentilationMethod.list()},"
                f"but is '{value}'"
            )
            self.__dict__[name] = value

        elif name == "zoning":
            assert value.lower() in bco.Zoning, (
                f"{name} has to be one of {bco.Zoning.list()}," f"but is '{value}'"
            )
            self.__dict__[name] = value

        elif name == "roof_type":
            assert value.lower() in bco.RoofType, (
                f"{name} has to be one of {bco.RoofType.list()}," f"but is '{value}'"
            )
            self.__dict__[name] = value

    def __post_init__(self):
        if (
            self.zone_heating_equipment == "water-to-air heat pump (water loop source)"
            and self.heating_water_loop_dimension != "building"
        ):
            print(
                "water-to-air heat pump (water loop source) only works with "
                "building wide water loop. "
                "Changing 'heating_system_dimension' to 'building'."
            )
            self.heating_water_loop_equipment_dimension = "building"
        if (
            self.zone_heating_equipment == "water-to-air heat pump (water loop source)"
            and not self.cooling_system_installed
        ):
            print(
                "water-to-air heat pump (water loop source) "
                "always comes with a cooling system. "
                "Setting cooling_system_installed to True."
            )
            self.cooling_system_installed = True
        if (
            self.heating_water_loop_equipment
            in ["air-to-water heat pump", "water-to-water heat pump (ground source)"]
            and self.heating_water_loop_equipment_fuel != "electricity"
        ):
            print(
                "heat pumps are always run on electricity. "
                "Changing fuel to electricity."
            )
            self.heating_water_loop_equipment_fuel = "electricity"

        if (
            self.dhw_heating_equipment
            in ["air-to-water heat pump", "water-to-water heat pump (ground source)"]
            and self.dhw_heating_equipment_fuel != "electricity"
        ):
            print(
                "heat pumps are always run on electricity. "
                "Changing fuel to electricity."
            )
            self.dhw_heating_equipment_fuel = "electricity"

    def save_to_file(self, path_to_datafile):
        with open(path_to_datafile, "w", encoding="utf-8") as out_file:
            json.dump(asdict(self), out_file, indent=4)


valid_dimensions = ["zone", "dwelling", "building", ""]
implemented_dimensions = ["zone", "building", ""]

valid_fuels = ["oil", "naturalgas", "electricity", "biomass", ""]
implemented_fuels = ["naturalgas", "electricity", ""]

valid_heating_water_loop_equipment = [
    "condensing boiler",
    "non-condensing boiler",
    "air-to-water heat pump",
    "water-to-water heat pump (ground source)",
    "district heating",
    "",
]
implemented_heating_water_loop_equipment = [
    "condensing boiler",
    "non-condensing boiler",
    "air-to-water heat pump",
    "",
]

valid_dhw_heating_equipment = [
    "condensing boiler",
    "non-condensing boiler",
    "air-to-water heat pump",
    "water-to-water heat pump (ground source)",
    "district heating",
    "solar collectors",
    "",
]
implemented_dhw_heating_equipment = [
    "condensing boiler",
    "non-condensing boiler",
    "air-to-water heat pump",
    "",
]


valid_zone_heating_equipment = [
    "radiator",
    "stove",
    "air-to-air heat pump",
    "water-to-air heat pump (water loop source)"
    "water-to-air heat pump (ground source)",
    "fan coil unit",
]

implemented_zone_heating_equipment = [
    "radiator",
    "water-to-air heat pump (water loop source)",
]


def load_building_config(path_to_datafile: str, files_dir: str):
    """
    Takes a json file and returns a BuildingConfig object.
    Args:
        path_to_datafile (str): path to json file
        files_dir (str): path to dir where sim files are stored
    Returns:
        BuildingConfig: BuildingConfig object
    """
    with open(path_to_datafile, encoding="utf-8") as file:
        data = json.loads(file.read())

    tuple_names = [
        "wtw_ratios",
        "wtw_ratios_loft",
        "distance_to_neighbour",
        "window_simple_values",
    ]
    for tn in tuple_names:
        if data[tn]:
            data[tn] = tuple(data[tn])

    # TODO: remove this hardcoding
    data["battery_power_rating"] = 4000
    data["files_dir"] = files_dir

    # get schedules which are specified in the schedules.json,
    # and overwrite what is in the building_config
    schedules_path = join(dirname(path_to_datafile), "schedules.json")

    with open(schedules_path, "r", encoding="utf-8") as schedules_file:
        sch_data = json.load(schedules_file)

    for sty, zone_schedule in enumerate(data["occupant_schedule"]):
        for i in range(len(zone_schedule)):
            data["occupant_schedule"][sty][i] = sch_data["schedule"]

    return from_dict(data_class=BuildingConfig, data=data)
