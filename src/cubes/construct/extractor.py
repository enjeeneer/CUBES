"""Module which extracts the relevant information from the datasets and creates an
instance of a buildingconfig dataclass
"""
from typing import Dict

from cubes.construct.buildingconfig import BuildingConfig
from cubes.construct.schedules import OccupancyScheduler
import numpy as np
import pandas as pd


class BuildingConfigExtractor:
    """_summary_"""

    def __init__(self):
        """This constructor is for use with Data from the Ambience database

        Args:
            sample: one row of building dataset
        """

    def __call__(self, sample: Dict) -> BuildingConfig:

        self.sample = sample
        self.name = sample["REFERENCE BUILDING CODE"]
        self.ground_floor_area = sample["REFERENCE BUILDING GROUND FLOOR AREA (m2)"]
        self.wall_area = sample["REFERENCE BUILDING WALL AREA (m2)"]
        self.window_area = sample["REFERENCE BUILDING WINDOW AREA (m2)"]
        self.window_to_wall_ratios = self._get_window_to_wall_ratios()
        self.roof_area = sample["REFERENCE BUILDING ROOF AREA (m2)"]
        self.facade_area = self.wall_area + self.window_area
        self.roof_type = self._get_roof_type()
        self.number_of_stories = int(sample["NUMBER OF REFERENCE BUILDING STOREYS"])
        self.floor_roof_ratio = self.roof_area / self.ground_floor_area
        self.storey_height = self.sample[
            "REFERENCE BUILDING STOREY HEIGHT (m)"
        ]  # tabula default for all buildings ceiling height
        self.length_wall_x, self.length_wall_y = self._calc_wall_length()
        self.roof_height = self._calc_roof_height()
        self.distance_to_neighbour = self._get_distance_to_neighbour()
        self.rotation = self._get_rotation()
        self.zones_per_storey = self._get_zones_per_storey()
        self.location = sample["NUTS 3 REGION"]  # TODO: check this
        self.terrain = self._get_terrain()
        self.wall_layer_materials = self._get_construction_element_materials(
            element="WALL"
        )
        self.roof_layer_materials = self._get_construction_element_materials("ROOF")

        self.ground_floor_layer_materials = self._get_construction_element_materials(
            "FLOOR"
        )
        self.wall_layer_thickness = self._get_construction_element_thickness("WALL")
        self.roof_layer_thickness = self._get_construction_element_thickness("ROOF")
        self.ground_floor_layer_thickness = self._get_construction_element_thickness(
            "FLOOR"
        )
        self.upper_floor_layer_materials = [
            "Cast concrete 2000"
        ]  # TODO: change or add to preprocessors
        self.upper_floor_layer_thickness = [0.2]

        # schedulers
        self.occupancy_scheduler = OccupancyScheduler(
            year=2022,
            sample_length="week",
            weekday_init_state_df=pd.read_parquet(
                "/workspaces/CUBES/src/cubes/data/"
                "occupants/weekday_occupancy_init_states.parquet"
            ),
            weekend_init_state_df=pd.read_parquet(
                "/workspaces/CUBES/src/cubes/data/"
                "occupants/weekend_occupancy_init_states.parquet"
            ),
            weekday_transition_matrix_df=pd.read_parquet(
                "/workspaces/CUBES/src/cubes/data/"
                "occupants/weekday_occupancy_transition.parquet"
            ),
            weekend_transition_matrix_df=pd.read_parquet(
                "/workspaces/CUBES/src/cubes/data/"
                "occupants/weekend_occupancy_transition.parquet"
            ),
        )

        (
            self.partition_layer_materials,
            self.partition_layer_thickness,
            self.partition_area_per_zone,
        ) = self._get_partition_data()

        (
            self.window_type,
            self.window_layer_materials,
            self.window_layer_thickness,
            self.window_simple_values,
            self.window_shading_device,
        ) = self._get_window_construction()

        (
            self.heating_system_type,
            self.heating_system_dimension,
            self.heating_system_fuel,
            self.heating_system_efficiency,
        ) = self._get_heating_system()

        (
            self.dhw_system_type,
            self.dhw_system_dimension,
            self.dhw_system_fuel,
            self.dhw_system_efficiency,
        ) = self._get_dhw_system()

        (
            self.cooling_system_type,
            self.cooling_system_dimension,
            self.cooling_system_fuel,
            self.cooling_system_efficiency,
        ) = self._get_cooling_system()

        (
            self.natvent_for_cooling_calculation_method,
            self.natvent_for_cooling_rate,
            self.natvent_for_cooling_indoor_t_range,
            self.ventilation_for_air_calculation_method,
            self.ventilation_for_air_rate,
            self.ventilation_for_air_fan_pressure_rise,
            self.ventilation_for_air_fan_efficiency,
            self.ventilation_for_air_heat_recovery_efficiency,
        ) = self._get_ventiliation()

        (
            self.infiltration_calculation_method,
            self.infiltration_rate,
        ) = self._get_infiltration()

        (
            self.occupant_schedule,
            self.occupant_value,
            self.occupant_number_calculation_method,
            self.equipment_gain_calculation_method,
            self.equipment_gain_value,
            self.equipment_gain_schedule,
            self.lighting_power_calculation_method,
            self.lighting_power_value,
            self.lighting_schedule,
            self.window_shading_control,
            self.window_shading_outside,
        ) = self._get_occupancy_and_misc()

        (
            self.heating_setpoint,
            self.heating_setback,
            self.heating_setpoint_schedule,
            self.cooling_setpoint,
            self.cooling_setback,
            self.cooling_setpoint_schedule,
        ) = self._get_setpoint_schedule()

        return BuildingConfig(  # pylint: disable=[E1123,E1120]
            name=sample["REFERENCE BUILDING CODE"],
            number_of_stories=int(sample["NUMBER OF REFERENCE BUILDING STOREYS"]),
            wtw_ratios=self.window_to_wall_ratios,
            distance_to_neighbour=self.distance_to_neighbour,
            storey_height=2.5,  # tabula default for all buildings ceiling height
            length_wall_x=self.length_wall_x,
            length_wall_y=self.length_wall_y,
            roof_type=self.roof_type,
            roof_height=self.roof_height,
            attic_is_heated=True,
            rotation=self.rotation,
            zones_per_storey=self.zones_per_storey,
            location=self.location,
            terrain=self.terrain,
            ground_floor_layer_materials=self.ground_floor_layer_materials,
            ground_floor_layer_thickness=self.ground_floor_layer_thickness,
            upper_floor_layer_materials=self.upper_floor_layer_materials,
            upper_floor_layer_thickness=self.upper_floor_layer_thickness,
            wall_layer_materials=self.wall_layer_materials,
            wall_layer_thickness=self.wall_layer_thickness,
            roof_layer_materials=self.roof_layer_materials,
            roof_layer_thickness=self.roof_layer_thickness,
            partition_layer_materials=self.partition_layer_materials,
            partition_layer_thickness=self.partition_layer_thickness,
            partition_area_per_zone=self.partition_area_per_zone,
            attic_floor_layer_materials=[],
            attic_floor_layer_thickness=[],
            window_type=self.window_type,
            window_layer_materials=self.window_layer_materials,
            window_layer_thickness=self.window_layer_thickness,
            window_simple_values=self.window_simple_values,
            window_shading_device=self.window_shading_device,
            window_shading_outside=self.window_shading_outside,
            window_shading_control=self.window_shading_control,
            heating_water_loop_dimension=self.heating_system_dimension,
            heating_water_loop_equipment_fuel=self.heating_system_fuel,
            heating_water_loop_equipment=self.heating_system_type,
            heating_water_loop_equipment_efficiency=self.heating_system_efficiency,
            dhw_heating_loop_dimension="building",
            dhw_heating_equipment_fuel="naturalgas",
            dhw_heating_equipment_efficiency=0.9,
            dhw_heating_equipment="condensing boiler",
            dhw_usage_schedule="",
            dhw_water_tank_volume=0,
            heating_water_loop_temperature=80,
            zone_heating_equipment="radiator",
            zone_heating_equipment_efficiency=1.0,
            cooling_system_installed=False,
            cooling_system_efficiency=self.cooling_system_efficiency,
            ventilation_type="model",
            ventilation_model="",
            natvent_for_cooling_calculation_method=(
                self.natvent_for_cooling_calculation_method
            ),
            natvent_for_cooling_rate=self.natvent_for_cooling_rate,
            natvent_for_cooling_indoor_t_range=self.natvent_for_cooling_indoor_t_range,
            ventilation_for_air_calculation_method=(
                self.ventilation_for_air_calculation_method
            ),
            ventilation_for_air_rate=self.ventilation_for_air_rate,
            ventilation_for_air_fan_pressure_rise=(
                self.ventilation_for_air_fan_pressure_rise
            ),
            ventilation_for_air_fan_efficiency=self.ventilation_for_air_fan_efficiency,
            ventilation_for_air_heat_recovery_efficiency=(
                self.ventilation_for_air_heat_recovery_efficiency
            ),
            window_opening_schedule="",
            infiltration_calculation_method=self.infiltration_calculation_method,
            infiltration_rate=self.infiltration_rate,
            occupant_number_calculation_method=self.occupant_number_calculation_method,
            occupant_value=self.occupant_value,
            occupant_schedule=self.occupant_schedule,
            equipment_gain_calculation_method=self.equipment_gain_calculation_method,
            equipment_gain_value=self.equipment_gain_value,
            equipment_gain_schedule=self.equipment_gain_schedule,
            lighting_power_calculation_method=self.lighting_power_calculation_method,
            lighting_power_value=self.lighting_power_value,
            lighting_schedule=self.lighting_schedule,
            heating_setpoint=self.heating_setpoint,
            heating_setback=self.heating_setback,
            heating_setpoint_schedule=self.heating_setpoint_schedule,
            cooling_setpoint=self.cooling_setpoint,
            cooling_setback=self.cooling_setback,
            cooling_setpoint_schedule=self.cooling_setpoint_schedule,
        )

    def _get_heating_system(self):
        """method which gets the heating system data from ambience and translates it
        into a format for energyplus to understand and use. Currently only dealing with
        boilders, but in future will need to deal with heat pumps, stoves, etc.

        Also will need to check is the assertions are correct
        e.g. see if fuel == liquid
        The link of biomass to coal is obviously not correct but biomass is not an
        option for energplus IIRC

        Returns:
            str: heating_system_type indicates the type of system, e.g. boiler, electric
            str: heating_system_dimension indicates if central or individual
            str: heating_system_fuel indicates the systems fuel
            float: heating_system_efficiency indicates the systems efficiency
        """

        if "GB" in self.name:
            self.heating_system_type = "Central gas condensing boiler"
            self.heating_system_dimension = "Central"
            self.heating_system_fuel = "Gas"
            self.heating_system_efficiency = 0.94

        else:

            self.heating_system_type = self.sample[
                "HEATING SYSTEM 1 TECHNOLOGY ENERGYPLUS"
            ].values[0]

            self.heating_system_dimension = self.sample[
                "HEATING SYSTEM 1 DIMENSIONS"
            ].values[0]

            self.heating_system_fuel = self.sample["HEATING SYSTEM 1 FUEL USED"].values[
                0
            ]

            self.heating_system_efficiency = self.sample[
                "HEATING SYSTEM 1 EFFICIENCY"
            ].values[0]

        if self.heating_system_fuel == "Gas":
            self.heating_system_fuel = "NaturalGas"
        if self.heating_system_fuel == "Liquid":
            self.heating_system_fuel = "FuelOilNo1"
        if self.heating_system_fuel == "Electricity":
            self.heating_system_fuel = "Electricity"
        if self.heating_system_fuel == "Biomass":
            self.heating_system_fuel = "OtherFuel1"

        print(
            "heating system type:",
            self.heating_system_type,
            "heating_system_dimension",
            self.heating_system_dimension,
            "heating_system_fuel",
            self.heating_system_fuel,
            "heating_system_efficiency",
            self.heating_system_efficiency,
        )

        return (
            self.heating_system_type,
            self.heating_system_dimension,
            self.heating_system_fuel,
            self.heating_system_efficiency,
        )

    def _get_dhw_system(self):
        """PLACEHOLDER METHOD
        Need to figure out how to model domestic hot water systems in energyplus before
        we can flesh this method out.

        Returns:
            str: dhw_system_type indicates the type of dhw system, e.g. boiler, electric
            str: dhw_system_dimension indicates if central or individual
            str: dhw_system_fuel indicates the systems fuel
            float: dhw_system_efficiency indicates the systems efficiency

        """
        if "GB" in self.name:
            self.dhw_system_type = "Central gas low temperature non-condensing boiler"
            self.dhw_system_dimension = "Central"
            self.dhw_system_fuel = "Gas"
            self.dhw_system_efficiency = 0.83

        else:

            self.dhw_system_type = self.sample["DHW SYSTEM 1 TECHNOLOGY"].values[0]

            self.dhw_system_dimension = self.sample["DHW SYSTEM 1 DIMENSIONS"].values[0]
            self.dhw_system_fuel = self.sample["DHW SYSTEM 1 FUEL USED"].values[0]
            self.dhw_system_efficiency = self.sample["DHW SYSTEM 1 EFFICIENCY"].values[
                0
            ]

        return (
            self.dhw_system_type,
            self.dhw_system_dimension,
            self.dhw_system_fuel,
            self.dhw_system_efficiency,
        )

    def _get_terrain(self):
        """method which gets the terrain which the building is located. Currently
        hardcoded to be in a town or city.

        Returns:
            str: terrain is where the building in situated
        """

        terrain = "Towns and cities"
        return terrain

    def _get_zones_per_storey(self):
        """method which gets the number of zones per storey, currently hardcoded to 0
        energyplus standard is 1 zone per thermostat, so 0 zones per storey means we
        assume one thermostat per building. Will change into the future.

        Returns:
            int: zones_per_storey is the number of zones per story
        """

        zones_per_storey = 1

        return zones_per_storey

    def _get_rotation(self):
        """method which gets rotation, currently hardcoded

        Returns:
            rotation float: rotation around z-axis, 0 means y is north, x is east
        """

        rotation = 0
        return rotation

    def _get_distance_to_neighbour(self):
        """method which determines the distance to the neighbouring building,
        hardcoded for now

        Future will need to read from Tabula dataset for residential proximity

        Returns:
            distance_to_neighbour tuple(float, float, float, float): distance to
                                                                     neighbour for each
                                                                     cardinal direction
        """
        typical_distance = (
            self.number_of_stories * self.storey_height + self.roof_height
        )
        if self.sample["Code_AttachedNeighbours"] == "B_N1":

            distance_to_neighbour = (
                2 * typical_distance,
                0,
                typical_distance,
                typical_distance,
            )
        elif self.sample["Code_AttachedNeighbours"] == "B_N2":
            distance_to_neighbour = (
                2 * typical_distance,
                0,
                typical_distance,
                0,
            )

        else:
            distance_to_neighbour = (
                2 * typical_distance,
                typical_distance,
                typical_distance,
                typical_distance,
            )

        return distance_to_neighbour

    def _get_window_to_wall_ratios(self):
        """method which gets the window to wall ratio, currently the windows are split
        equally between all four cardinal directions

        Future will need to differentiate the directions cased on archetype

        Returns:
            window_to_wall_ratios tuple(float, float, float, float): window to wall
            ratio for each cardinal direction
        """

        if self.sample["Code_AttachedNeighbours"] == "B_N1":
            wtw_ratio = self.window_area / self.wall_area * 4 / 3

            window_to_wall_ratios = (
                wtw_ratio,
                0,
                wtw_ratio,
                wtw_ratio,
            )
        elif self.sample["Code_AttachedNeighbours"] == "B_N2":
            wtw_ratio = self.window_area / self.wall_area * 2

            window_to_wall_ratios = (
                wtw_ratio,
                0,
                wtw_ratio,
                0,
            )

        else:
            wtw_ratio = self.window_area / self.wall_area
            window_to_wall_ratios = [wtw_ratio, wtw_ratio, wtw_ratio, wtw_ratio]

        return window_to_wall_ratios

    def _get_occupancy_and_misc(self):
        """method which gets occupancy and heat gain data, currently hardcoded so
        future work should update assumptions

        Returns:
            occupant_schedule str: occupant schedule
            equipment_gain_type str: floor area or occupant or zone
            equipment_gain_value float: energy gain from equipment
            lighting_power float: power consumption of lighting
            window_shading_control str: control of window shading
        """

        occupant_schedule = self.occupancy_scheduler.sample(number_of_occupants=2)
        occupant_value = 2
        occupant_number_calculation_method = "People/area"

        equipment_gain_calculation_method = "Watts/person"
        equipment_gain_value = 100
        equipment_gain_schedule = "Always_max"

        lighting_power_calculation_method = "Watts/area"
        lighting_power_value = 1
        lighting_schedule = "Wang_lights"

        window_shading_control = "None"  # need to define a rule
        window_shading_outside = False

        return (
            occupant_schedule,
            occupant_value,
            occupant_number_calculation_method,
            equipment_gain_calculation_method,
            equipment_gain_value,
            equipment_gain_schedule,
            lighting_power_calculation_method,
            lighting_power_value,
            lighting_schedule,
            window_shading_control,
            window_shading_outside,
        )

    def _get_infiltration(self):
        """method which gets infiltration rate, currently hardcoded so assumes
        average infiltration rate taken from UK study on housing infiltration.
        Pasos 2020 https://doi.org/10.1016/j.buildenv.2020.107275
        Paper could be promising as a dataset

        Returns:
            infiltration_per_area float: air permeability in m3 h-1 m-3
        """
        infiltration_calculation_method = "AirChanges/Hour"

        if "GB" in self.name:
            infiltration_rate = self.sample.at[0, "n_air_infiltration"]

        else:
            # will try and get infiltration rate from Tabula for EU residential
            infiltration_rate = 7.92 / 20

        return infiltration_calculation_method, infiltration_rate

    def _get_ventiliation(self):
        """method gets ventilation parameters, currently hardcoded for now as dataset
        needs to be found

        Returns:
            natural_ventilation bool: presence of natural ventilation
            mechanical_ventilation bool: presence of mechanical ventilation
            mech_ventilation_heat_recovery bool: presence of heat recovery
            ventilation_fan_power float: fan power of the ventilation
        """

        natvent_for_cooling_calculation_method = "AirChanges/Hour"
        natvent_for_cooling_rate = 2
        natvent_for_cooling_indoor_t_range = (22, 30)

        ventilation_for_air_calculation_method = "Flow/Person"
        ventilation_for_air_rate = 0.00944
        ventilation_for_air_fan_pressure_rise = 1
        ventilation_for_air_fan_efficiency = 1
        ventilation_for_air_heat_recovery_efficiency = 0

        return (
            natvent_for_cooling_calculation_method,
            natvent_for_cooling_rate,
            natvent_for_cooling_indoor_t_range,
            ventilation_for_air_calculation_method,
            ventilation_for_air_rate,
            ventilation_for_air_fan_pressure_rise,
            ventilation_for_air_fan_efficiency,
            ventilation_for_air_heat_recovery_efficiency,
        )

    def _get_cooling_system(self):
        """method which returns the type of cooling system of archetype. Ambience does
        not disclose the system type, it only indicates if cooling is likely or unlikely
        therefore an assumption needs to be made as to what kind of cooling

        note: air conditioning and None are placeholders

        Returns:
            cooling_system str: describes if the system is air conditioning or none
        """
        cooling_system_presence = self.sample["Cooling presence according to HOTMAPS"]

        if "No" in cooling_system_presence:
            cooling_system_type = "None"
            cooling_system_dimension = ""
            cooling_system_fuel = ""
            cooling_system_efficiency = 1
        else:
            cooling_system_type = "Air Conditioning"
            cooling_system_dimension = ""
            cooling_system_fuel = ""
            cooling_system_efficiency = 1

        return (
            cooling_system_type,
            cooling_system_dimension,
            cooling_system_fuel,
            cooling_system_efficiency,
        )

    def _get_roof_type(self):
        """determines roof type from the ratio between roof and ground floor area,
           the options are flat and saddleback for now, however in the future if the
           ratio is beyond some value (arbitrarily 1.8 below) and has a certain
           aspect ratio then it would make sense to differentiate between a
           pyramid or hip roof

        Returns:
            roof_type str: describes type of roof e.g. saddleback, flat, pyramid, hip
        """

        if 0 < self.floor_roof_ratio <= 1:
            roof_type = "flat"
        elif 1 < self.floor_roof_ratio <= 1.8:
            roof_type = "saddleback"
        else:
            roof_type = (
                "saddleback"  # will need to offer pyramid/hip depending on ratio
            )

        return roof_type

    def _get_construction_element_materials(self, element):
        """method which gets the construction materials of building element (e.g. roof
        or wall) from Ambience geometry dataset

        Args:
            element (str): type of building element, e.g. roof, wall or floor

        Returns:
            element_materials List: the materials using in building element
        """

        element_materials = [
            self.sample["REFERENCE BUILDING " + element + " MATERIAL"],
            self.sample["REFERENCE BUILDING " + element + " INSULATION MATERIAL"],
        ]

        return element_materials

    def _get_construction_element_thickness(self, element):
        """method which gets the thickness of the construction materials within
        building element (e.g. roof or wall) from Ambience geometry dataset

        Args:
            element (str): type of building element, e.g. roof, wall or floor

        Returns:
            element_thickness List: the thickness of materials in element
        """

        element_thickness = [
            self.sample["REFERENCE BUILDING " + element + " MATERIAL THICKNESS (m)"],
            self.sample[
                "REFERENCE BUILDING " + element + " INSULATION MATERIAL THICKNESS (m)"
            ],
        ]

        return element_thickness

    def _get_window_construction(self):
        """Uses description of windows from Ambience dataset to create a window and
            frame construction

        Returns:
            str: Window type (Single,double,simple)
            list of str: window_layers describes the material build-up of window
            list of int: window_thickness describes the build-up thickness of window
            tuple [float,float,float]: if window type=="Simple", this defines U-factor,
                                        SHGC and visible transmittance
            str: window_shading_device describes how the window is shaded e.g. shutters
        """

        window_description = "Double glazed 6 mm   Wood 30 mm thick frame"
        window_material_glazing = "Double"
        window_material_glazing_type = "CLEAR 3MM"
        window_material_gas = "Air"

        # else:

        #    window_description = self.sample[
        #        "REFERENCE BUILDING WINDOW TYPE"
        #    ]

        #    window_material_glazing = self.sample[
        #        "REFERENCE BUILDING WINDOW GLAZING TYPE"
        #    ]

        #    if (
        #        self.sample["REFERENCE BUILDING WINDOW COATED"]
        #        == "Coated"
        #    ):
        #        window_material_glazing_type = "CLEAR 3MM"

        #    else:
        #        window_material_glazing_type = "LoE CLEAR 3MM"

        #    if (
        #        self.sample[
        #            "REFERENCE BUILDING WINDOW FILLING GAS"
        #        ]
        #        == "No gas"
        #    ):
        #        window_material_gas = "Air"
        #    else:
        #        window_material_gas = self.sample[
        #            "REFERENCE BUILDING WINDOW FILLING GAS"
        #        ]

        # Assume windows are 3 mm thick from energyplus window construction data
        window_material_glazing_thickness = 3
        window_shading_device = "None"
        window_layers = [window_material_glazing_type]
        window_thickness = [window_material_glazing_thickness]

        if window_material_glazing != "Single":
            window_material_gap = int(window_description.split("glazed")[1].split()[0])

            window_layers.extend([window_material_gas, window_material_glazing_type])
            window_thickness.extend(
                [window_material_gap, window_material_glazing_thickness]
            )

            if window_material_glazing != "Double":
                window_layers.extend(
                    [window_material_gas, window_material_glazing_type]
                )
                window_thickness.extend(
                    [window_material_gap, window_material_glazing_thickness]
                )

        return (
            window_material_glazing,
            window_layers,
            window_thickness,
            None,
            window_shading_device,
        )

    def _calc_wall_length(self):
        """Calculates wall length assuming a square footprint, the Ambience method
        results in very weird shaped buildings. Their aspect ratio is too extreme.
        Current fix is to assume a square shaped footprint

        Returns:
            length_wall_x float: the length of the wall along the x-axis
            length_wall_y float: the length of the wall along the y-axis
        """

        aspect_ratio = 1.2

        length_wall_x = np.sqrt(self.ground_floor_area / aspect_ratio)
        length_wall_y = np.sqrt(self.ground_floor_area * aspect_ratio)

        return length_wall_x, length_wall_y

    def _calc_roof_height(self):
        """Calculates the roof height, currently only deals with flat or saddleback
        roof types (i.e.pitched and split in two equal sized elements)

        Returns:
            roof_height float: the estimated height of the roof
        """

        if self.roof_type == "flat":
            # if the roof is the flat then attic space is not needed
            roof_height = 0

        elif self.roof_type == "saddleback":
            # double check this formula!
            roof_height = (
                np.sqrt((self.length_wall_y**2) * ((self.floor_roof_ratio**2) - 1))
            ) / 2

        else:
            raise ValueError("roof type not modelled yet")

        return roof_height

    def _get_partition_data(self):
        self.partition_layer_materials = []
        self.partition_layer_thickness = [0.0]
        self.partition_area_per_zone = 0.0

        return (
            self.partition_layer_materials,
            self.partition_layer_thickness,
            self.partition_area_per_zone,
        )

    def _get_setpoint_schedule(self):
        self.heating_setpoint = 20
        self.heating_setback = 15
        self.heating_setpoint_schedule = "Singh_heating_setpoint"
        self.cooling_setpoint = 25
        self.cooling_setback = 30
        self.cooling_setpoint_schedule = "Singh_cooling_setpoint"

        return (
            self.heating_setpoint,
            self.heating_setback,
            self.heating_setpoint_schedule,
            self.cooling_setpoint,
            self.cooling_setback,
            self.cooling_setpoint_schedule,
        )
