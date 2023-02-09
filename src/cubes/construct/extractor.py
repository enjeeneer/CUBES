"""Module which extracts the relevant information from the datasets and creates an
instance of a buildingconfig dataclass
"""
from cubes.construct import buildingconfig as bc
import numpy as np


class Extractor:
    """_summary_"""

    def __init__(self, ambience_geometry_data, ambience_systems_data):
        """This constructor is for use with Data from the Ambience database

        Args:
            geometry_data (_type_): _description_
            systems_data (_type_): _description_
        """

        self.ambience_geometry_data = ambience_geometry_data
        self.ambience_systems_data = ambience_systems_data

        self.name = ambience_geometry_data.loc[0]["REFERENCE BUILDING CODE"]
        self.a_ground_floor = ambience_geometry_data.loc[0][
            "REFERENCE BUILDING GROUND FLOOR AREA (m2)"
        ]
        self.a_wall = ambience_geometry_data.loc[0]["REFERENCE BUILDING WALL AREA (m2)"]
        self.a_window = ambience_geometry_data.loc[0][
            "REFERENCE BUILDING WINDOW AREA (m2)"
        ]
        self.wtw_ratios = self.get_wtw_ratios()
        self.a_roof = ambience_geometry_data.loc[0]["REFERENCE BUILDING ROOF AREA (m2)"]
        self.a_facade = self.a_wall + self.a_window
        self.r_floor_roof = self.a_roof / self.a_ground_floor
        self.roof_type = self.get_roof_type()
        self.n_storey = int(
            ambience_geometry_data.loc[0]["NUMBER OF REFERENCE BUILDING STOREYS"]
        )
        self.h_storey = 2.5  # tabula default for all buildings ceiling height
        self.l_wall_x, self.l_wall_y = self.calc_wall_length()
        self.h_roof = self.calc_roof_height()
        self.distance_to_neighbour = self.get_distance_to_neighbour()
        self.rotation = self.get_rotation()
        self.zones_per_storey = self.get_zones_per_storey()
        self.location = self.get_location()
        self.terrain = self.get_terrain()

        self.roof_layer_materials = ambience_geometry_data.loc[0][
            "REFERENCE BUILDING ROOF MATERIAL"
        ]
        self.wall_layer_materials = self.get_construction_element_materials(
            element="WALL"
        )
        self.roof_layer_materials = self.get_construction_element_materials("ROOF")

        self.ground_floor_layer_materials = self.get_construction_element_materials(
            "FLOOR"
        )
        self.wall_layer_thickness = self.get_construction_element_thickness("WALL")
        self.roof_layer_thickness = self.get_construction_element_thickness("ROOF")
        self.ground_floor_layer_thickness = self.get_construction_element_thickness(
            "FLOOR"
        )

        # hard coded for now, needs to change!
        self.upper_floor_layer_materials = ["Cast concrete 2000"]  # bottom to top
        self.upper_floor_layer_thickness = [0.2]

        (
            self.window_type,
            self.window_layer_materials,
            self.window_layer_thickness,
            self.window_shading_device,
        ) = self.get_window_construction()

        (
            self.heating_system_type,
            self.heating_system_dimension,
            self.heating_system_fuel,
            self.heating_system_efficiency,
        ) = self.get_heating_system()

        (
            self.dhw_system_type,
            self.dhw_system_dimension,
            self.dhw_system_fuel,
            self.dhw_system_efficiency,
        ) = self.get_dhw_system()

        self.cooling_system_type = self.get_cooling_system()

        (
            self.natural_ventilation,
            self.mechanical_ventilation,
            self.mech_ventilation_heat_recovery,
            self.ventilation_fan_power,
        ) = self.get_ventiliation()

        self.infiltration_per_area = self.get_infiltration()

        (
            self.occupant_number_max,
            self.occupant_schedule,
            self.equipment_gain_type,
            self.equipment_gain_value,
            self.lighting_power,
            self.window_shading_control,
        ) = self.get_occupancy_and_misc()

    def get_heating_system(self):
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

        # Future - will need to model more energy systems other than boilers
        # e.g. heat pumps, stoves, electrical heater etc.
        # Future - will need to model biomass and double check if Solid and Liquid fuel
        # in Ambience is actually coal and Diesal etc

        self.heating_system_type = self.ambience_systems_data[
            "HEATING SYSTEM 1 TECHNOLOGY"
        ].values[0]

        self.heating_system_dimension = self.ambience_systems_data[
            "HEATING SYSTEM 1 DIMENSIONS"
        ].values[0]

        self.heating_system_fuel = self.ambience_systems_data[
            "HEATING SYSTEM 1 FUEL USED"
        ].values[0]

        self.heating_system_efficiency = self.ambience_systems_data[
            "HEATING SYSTEM 1 EFFICIENCY"
        ].values[0]

        if "boiler" in self.heating_system_type:

            if "non-condensing" in self.heating_system_type:

                self.heating_system_type = "HotWaterBoiler"
            else:
                self.heating_system_type = "CondensingHotWaterBoiler"

            if self.heating_system_fuel == "Gas":
                self.heating_system_fuel = "NaturalGas"
            if self.heating_system_fuel == "Liquid":
                self.heating_system_fuel = "Diesel"  # Double check
            if self.heating_system_fuel == "Electricity":
                self.heating_system_fuel = "Electricity"
            if self.heating_system_fuel == "Biomass":
                self.heating_system_fuel = "Coal"  # Double check
            if self.heating_system_fuel == "Solid":
                self.heating_system_fuel = "Coal"  # Double check

        else:
            print(self.heating_system_fuel + " not yet handled by " + __name__)

        return (
            self.heating_system_type,
            self.heating_system_dimension,
            self.heating_system_fuel,
            self.heating_system_efficiency,
        )

    def get_dhw_system(self):
        """PLACEHOLDER METHOD
        Need to figure out how to model domestic hot water systems in energyplus before
        we can flesh this method out.

        Returns:
            str: dhw_system_type indicates the type of dhw system, e.g. boiler, electric
            str: dhw_system_dimension indicates if central or individual
            str: dhw_system_fuel indicates the systems fuel
            float: dhw_system_efficiency indicates the systems efficiency

        """

        self.dhw_system_type = self.ambience_systems_data[
            "DHW SYSTEM 1 TECHNOLOGY"
        ].values[0]

        self.dhw_system_dimension = self.ambience_systems_data[
            "DHW SYSTEM 1 DIMENSIONS"
        ].values[0]
        self.dhw_system_fuel = self.ambience_systems_data[
            "DHW SYSTEM 1 FUEL USED"
        ].values[0]
        self.dhw_system_efficiency = self.ambience_systems_data[
            "DHW SYSTEM 1 EFFICIENCY"
        ].values[0]

        return (
            self.dhw_system_type,
            self.dhw_system_dimension,
            self.dhw_system_fuel,
            self.dhw_system_efficiency,
        )

    def get_terrain(self):
        """method which gets the terrain which the building is located. Currently
        hardcoded to be in a town or city.

        Returns:
            str: terrain is where the building in situated
        """

        terrain = "Towns and cities"
        return terrain

    def get_location(self):
        """method which gets the location of the building, currently hardcoded to
        Cambridge. Will change into the future

        Returns:
            str: location is where the building is situated
        """

        location = "cambridge"
        return location

    def get_zones_per_storey(self):
        """method which gets the number of zones per storey, currently hardcoded to 0
        energyplus standard is 1 zone per thermostat, so 0 zones per storey means we
        assume one thermostat per building. Will change into the future.

        Returns:
            int: zones_per_storey is the number of zones per story
        """

        zones_per_storey = 0

        return zones_per_storey

    def get_rotation(self):
        """method which gets rotation, currently hardcoded

        Returns:
            rotation float: rotation around z-axis, 0 means y is north, x is east
        """

        rotation = 0
        return rotation

    def get_distance_to_neighbour(self):
        """method which determines the distance to the neighbouring building,
        hardcoded for now

        Future will need to read from Tabula dataset for residential proximity

        Returns:
            distance_to_neighbour tuple(float, float, float, float): distance to
                                                                     neighbour for each
                                                                     cardinal direction
        """
        distance_to_neighbour = (10, 10, 10, 10)

        return distance_to_neighbour

    def get_wtw_ratios(self):
        """method which gets the window to wall ratio, currently the windows are split
        equally between all four cardinal directions

        Future will need to differentiate the directions cased on archetype

        Returns:
            wtw_ratios tuple(float, float, float, float): window to wall ratio for each
                                                          cardinal direction
        """
        wtw_ratio = self.a_window / self.a_wall
        wtw_ratios = [wtw_ratio, wtw_ratio, wtw_ratio, wtw_ratio]

        return wtw_ratios

    def get_occupancy_and_misc(self):
        """method which gets occupancy and heat gain data, currently hardcoded so
        future work should update assumptions

        Returns:
            occupant_number_max int: maximum number of occupants
            occupant_schedule str: occupant schedule
            equipment_gain_type str: floor area or occupant or zone
            equipment_gain_value float: energy gain from equipment
            lighting_power float: power consumption of lighting
            window_shading_control str: control of window shading
        """
        occupant_number_max = 2
        occupant_schedule = (
            "None"  # could have some fixed schedules or stochastic models
        )
        equipment_gain_type = "floor area"  # floor area or occupant or zone
        equipment_gain_value = 0  # what's an average value?
        lighting_power = 0  # what's an average value?
        window_shading_control = "None"  # need to define a rule

        return (
            occupant_number_max,
            occupant_schedule,
            equipment_gain_type,
            equipment_gain_value,
            lighting_power,
            window_shading_control,
        )

    def get_infiltration(self):
        """method which gets infiltration rate, currently hardcoded so assumes
        average infiltration rate taken from UK study on housing infiltration.
        Pasos 2020 https://doi.org/10.1016/j.buildenv.2020.107275
        Paper could be promising as a dataset

        Returns:
            infiltration_per_area float: air permeability in m3 h-1 m-3
        """

        infiltration_per_area = 7.92 / 20
        return infiltration_per_area

    def get_ventiliation(self):
        """method gets ventilation parameters, currently hardcoded for now as dataset
        needs to be found

        Returns:
            natural_ventilation bool: presence of natural ventilation
            mechanical_ventilation bool: presence of mechanical ventilation
            mech_ventilation_heat_recovery bool: presence of heat recovery
            ventilation_fan_power float: fan power of the ventilation
        """

        natural_ventilation = True
        mechanical_ventilation = False
        mech_ventilation_heat_recovery = False
        ventilation_fan_power = 0

        return (
            natural_ventilation,
            mechanical_ventilation,
            mech_ventilation_heat_recovery,
            ventilation_fan_power,
        )

    def get_cooling_system(self):
        """method which returns the type of cooling system of arcehtype. Ambience does
        not disclose the system type, it only indicates if cooling is likely or unlikely
        therefore an assumption needs to be made as to what kind of cooling

        note: air conditioning and None are placeholders

        Returns:
            cooling_system str: describes if the system is air conditioning or none
        """

        cooling_system_presence = self.ambience_systems_data[
            "Cooling presence according to HOTMAPS"
        ].values[0]

        if "No" in cooling_system_presence:
            cooling_system = "None"
        else:
            cooling_system = "Air Conditioning"

        return cooling_system

    def get_roof_type(self):
        """determines roof type from the ratio between roof and ground floor area,
           the options are flat and saddleback for now, however in the future if the
           ratio is beyond some value (arbitrarily 1.8 below) and has a certain
           aspect ratio then it would make sense to differentiate between a
           pyramid or hip roof

        Returns:
            roof_type str: describes type of roof e.g. saddleback, flat, pyramid, hip
        """

        if 0 < self.r_floor_roof <= 1:
            roof_type = "flat"
        elif 1 < self.r_floor_roof <= 1.8:
            roof_type = "saddleback"
        else:
            roof_type = (
                "saddleback"  # will need to offer pyramid/hip depending on ratio
            )

        return roof_type

    def get_construction_element_materials(self, element):
        """method which gets the construction materials of building element (e.g. roof
        or wall) from Ambience geometry dataset

        Args:
            element (str): type of building element, e.g. roof, wall or floor

        Returns:
            element_materials List: the materials using in building element
        """
        # from outside in
        element_materials = [
            self.ambience_geometry_data.loc[0][
                "REFERENCE BUILDING " + element + " MATERIAL"
            ],
            self.ambience_geometry_data.loc[0][
                "REFERENCE BUILDING " + element + " INSULATION MATERIAL"
            ],
        ]

        return element_materials

    def get_construction_element_thickness(self, element):
        """method which gets the thickness of the construction materials within
        building element (e.g. roof or wall) from Ambience geometry dataset

        Args:
            element (str): type of building element, e.g. roof, wall or floor

        Returns:
            element_thickness List: the thickness of materials in element
        """

        element_thickness = [
            self.ambience_geometry_data.loc[0][
                "REFERENCE BUILDING " + element + " MATERIAL THICKNESS (m)"
            ],
            self.ambience_geometry_data.loc[0][
                "REFERENCE BUILDING " + element + " INSULATION MATERIAL THICKNESS (m)"
            ],
        ]

        return element_thickness

    def get_window_construction(self):
        """Uses description of windows from Ambience dataset to create a window and
            frame construction

        Returns:
            list of str: window_layers describes the material build-up of window
            list of int: window_thickness describes the build-up thickness of window
            str: window_material_frame describes the material build-up of frame
            int: window_frame_thickness describes the build-up thickness of frame
            str: window_shading_device describes how the window is shaded e.g. shutters
        """

        window_description = self.ambience_geometry_data.loc[0][
            "REFERENCE BUILDING WINDOW TYPE"
        ]

        window_material_glazing = self.ambience_geometry_data.loc[0][
            "REFERENCE BUILDING WINDOW GLAZING TYPE"
        ]

        if (
            self.ambience_geometry_data.loc[0]["REFERENCE BUILDING WINDOW COATED"]
            == "Coated"
        ):
            window_material_glazing_type = "CLEAR 3MM"

        else:
            window_material_glazing_type = "LoE CLEAR 3MM"

        if (
            self.ambience_geometry_data.loc[0]["REFERENCE BUILDING WINDOW FILLING GAS"]
            == "No gas"
        ):
            window_material_gas = "Air"
        else:
            window_material_gas = self.ambience_geometry_data.loc[0][
                "REFERENCE BUILDING WINDOW FILLING GAS"
            ]

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
            window_shading_device,
        )

    def calc_wall_length(self):
        """Calculates wall length assuming a square footprint, the Ambience method
        results in very weird shaped buildings. Their aspect ratio is too extreme.
        Current fix is to assume a square shaped footprint

        Returns:
            l_wall_x float: the length of the wall along the x-axis
            l_wall_y float: the length of the wall along the y-axis
        """

        l_wall_x = np.sqrt(self.a_ground_floor)
        l_wall_y = np.sqrt(self.a_ground_floor)

        return l_wall_x, l_wall_y

    def calc_wall_length_ambience(self):
        """Calculates wall length using formula from Ambience

        Returns:
            l_wall_x float: the length of the wall along the x-axis
            l_wall_y float: the length of the wall along the y-axis
        """

        determinant = (
            self.a_facade / (2 * self.n_storey * self.h_storey)
        ) ** 2 - 4 * self.a_ground_floor

        # checks if determinant is positive
        if determinant < 0:
            # negative
            # follow ambience's assumption of an aspect ratio of 1.5
            l_wall_y = np.sqrt(self.a_ground_floor / 1.5)
            l_wall_x = 1.5 * l_wall_y

        else:
            # positive
            # follow ambience's equation for wall lengths
            # assumes wall_x is the longer wall
            l_wall_x = (
                (self.a_facade / (2 * self.n_storey * self.h_storey))
                + np.sqrt(determinant)
            ) / 2

            l_wall_y = (
                (self.a_facade / (2 * self.n_storey * self.h_storey))
                - np.sqrt(determinant)
            ) / 2

        return l_wall_x, l_wall_y

    def calc_roof_height(self):
        """Calculates the roof height, currently only deals with flat or saddleback
        roof types (i.e.pitched and split in two equal sized elements)

        Returns:
            h_roof float: the estimated height of the roof
        """

        if self.roof_type == "flat":
            # if the roof is the flat then attic space is not needed
            h_roof = 0

        elif self.roof_type == "saddleback":
            h_roof = (
                np.sqrt((self.l_wall_y**2) * ((self.r_floor_roof**2) - 1))
            ) / 2

        else:
            raise ValueError("roof type not modelled yet")

        return h_roof

    def create_building_config_object(self):
        building_config = bc.BuildingConfig(
            name=self.name,
            n_storey=self.n_storey,
            wtw_ratios=self.wtw_ratios,
            distance_to_neighbour=self.distance_to_neighbour,
            h_storey=self.h_storey,
            l_wall_x=self.l_wall_x,
            l_wall_y=self.l_wall_y,
            roof_type=self.roof_type,
            h_roof=self.h_roof,
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
            partition_layer_materials=[],
            partition_layer_thickness=[0.0],
            partition_area_per_zone=0.0,
            window_type=self.window_type,
            window_layer_materials=self.window_layer_materials,
            window_layer_thickness=self.window_layer_thickness,
            window_shading_device=self.window_shading_device,
            window_shading_outside=False,
            window_shading_control=self.window_shading_control,
            heating_system_type=self.heating_system_type,
            heating_system_dimension=self.heating_system_dimension,
            heating_system_fuel=self.heating_system_fuel,
            heating_system_efficiency=self.heating_system_efficiency,
            dhw_system_type=self.dhw_system_type,
            dhw_system_dimension=self.dhw_system_dimension,
            dhw_system_fuel=self.dhw_system_fuel,
            dhw_system_efficiency=self.dhw_system_efficiency,
            cooling_system_type=self.cooling_system_type,
            cooling_system_dimension="",
            cooling_system_fuel="",
            cooling_system_efficiency=1,
            natvent_for_cooling_calculation_method=str,
            natvent_for_cooling_rate=2,
            natvent_for_cooling_indoor_t_range=(22, 30),
            ventilation_for_air_calculation_method="Flow/Person",
            ventilation_for_air_rate=0.00944,
            ventilation_for_air_fan_pressure_rise=1,
            ventilation_for_air_fan_efficiency=1,
            ventilation_for_air_heat_recovery_efficiency=0,
            infiltration_calculation_method="Flow/ExteriorArea",
            infiltration_rate=self.infiltration_per_area,
            occupant_number_calculation_method="People",
            occupant_value=self.occupant_number_max,
            occupant_schedule=self.occupant_schedule,
            equipment_gain_calculation_method="Watts/area",
            equipment_gain_value=self.equipment_gain_value,
            equipment_gain_schedule="",
            lighting_power_calculation_method="Watts/area",
            lighting_power_value=self.lighting_power,
            lighting_schedule="",
            heating_setpoint=20,
            heating_setback=15,
            heating_setpoint_schedule="",
            cooling_setpoint=25,
            cooling_setback=30,
            cooling_setpoint_schedule="",
        )

        return building_config
