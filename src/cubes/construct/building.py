"""Defines the Building class """

from cubes.construct.constants import EPLUS_PATH
from cubes.construct import constants as con
from cubes.construct import material as mat

from geomeppy import IDF
import numpy as np


class Building:

    """This class holds all the information and methods to produce an IDF file"""

    def __init__(self, geometry_data, systems_data):
        """This constructor is for use with Data from the Ambience database

        Args:
            geometry_data (_type_): _description_
            systems_data (_type_): _description_
        """

        self.a_ground_floor = geometry_data.loc[0][
            "REFERENCE BUILDING GROUND FLOOR AREA (m2)"
        ]
        self.a_wall = geometry_data.loc[0]["REFERENCE BUILDING WALL AREA (m2)"]
        self.n_storey = int(
            geometry_data.loc[0]["NUMBER OF REFERENCE BUILDING STOREYS"]
        )
        self.a_window = geometry_data.loc[0]["REFERENCE BUILDING WINDOW AREA (m2)"]
        self.r_floor_roof = geometry_data.loc[0]["REFERENCE BUILDING FLOOR ROOF RATIO"]

        self.h_ceiling = 2.5  # tabula default for all buildings

        self.l_wall_front, self.l_wall_side = self.calc_wall_length()
        self.h_roof = self.get_roof_height()

        # read in materials, outside to inside
        self.wall_materials = [
            con.MATERIALS[geometry_data.loc[0]["REFERENCE BUILDING WALL MATERIAL"]],
            con.MATERIALS[
                geometry_data.loc[0]["REFERENCE BUILDING WALL INSULATION MATERIAL"]
            ],
        ]

        self.wall_layer_thicknesses = [
            geometry_data.loc[0]["REFERENCE BUILDING WALL MATERIAL THICKNESS (m)"],
            geometry_data.loc[0][
                "REFERENCE BUILDING WALL INSULATION MATERIAL THICKNESS (m)"
            ],
        ]

        self.roof_materials = [
            con.MATERIALS[geometry_data.loc[0]["REFERENCE BUILDING ROOF MATERIAL"]],
            con.MATERIALS[
                geometry_data.loc[0]["REFERENCE BUILDING ROOF INSULATION MATERIAL"]
            ],
        ]
        self.roof_layer_thicknesses = [
            geometry_data.loc[0]["REFERENCE BUILDING WALL MATERIAL THICKNESS (m)"],
            geometry_data.loc[0][
                "REFERENCE BUILDING WALL INSULATION MATERIAL THICKNESS (m)"
            ],
        ]

        self.ground_floor_materials = [
            con.MATERIALS[geometry_data.loc[0]["REFERENCE BUILDING FLOOR MATERIAL"]],
            con.MATERIALS[
                geometry_data.loc[0]["REFERENCE BUILDING FLOOR INSULATION MATERIAL"]
            ],
        ]
        self.ground_floor_layer_thicknesses = [
            geometry_data.loc[0]["REFERENCE BUILDING WALL MATERIAL THICKNESS (m)"],
            geometry_data.loc[0][
                "REFERENCE BUILDING WALL INSULATION MATERIAL THICKNESS (m)"
            ],
        ]

        # hard coded for now, needs to change!
        self.floor_materials = [con.MATERIALS["Cast concrete 2000"]]  # bottom to top
        self.floor_layer_thicknesses = [0.2]
        self.ceiling_materials = self.floor_materials[::-1]  # top to bottom
        self.ceiling_layer_thicknesses = self.floor_layer_thicknesses[::-1]

        self.wall_construction = mat.Construction(
            "Wall", self.wall_materials, self.wall_layer_thicknesses
        )
        self.ground_floor_construction = mat.Construction(
            "GroundFloor",
            self.ground_floor_materials,
            self.ground_floor_layer_thicknesses,
        )
        self.roof_construction = mat.Construction(
            "Roof", self.roof_materials, self.roof_layer_thicknesses
        )
        self.floor_construction = mat.Construction(
            "Floor", self.floor_materials, self.floor_layer_thicknesses
        )
        self.ceiling_construction = mat.Construction(
            "Ceiling", self.ceiling_materials, self.ceiling_layer_thicknesses
        )

        # windows
        self.window_construction = mat.WindowConstruction(
            geometry_data["REFERENCE BUILDING WINDOW GLAZING TYPE"].values[0],
            geometry_data["REFERENCE BUILDING WINDOW COATED"].values[0] == "Coated",
            con.get_window_gap_width(
                geometry_data["REFERENCE BUILDING WINDOW TYPE"].values[0]
            ),
        )

        self.heating_system_efficiency = systems_data[
            "HEATING SYSTEM 1 EFFICIENCY"
        ].values[0]
        self.heating_system_type = systems_data["HEATING SYSTEM 1 TECHNOLOGY"].values[0]
        self.heating_system_fuel = systems_data["HEATING SYSTEM 1 FUEL USED"].values[0]

        IDF.setiddname(EPLUS_PATH + "Energy+.idd")
        self.idf = IDF(EPLUS_PATH + "ExampleFiles/Minimal.idf")
        # Future - Will need to automatically add in weather file based on locations
        self.idf.epw = EPLUS_PATH + "WeatherData/USA_CO_Golden-NREL.724666_TMY3.epw"

    def calc_wall_length(self):
        """Calculates wall length using formula from Ambience"""
        a_facade = self.a_wall + self.a_window
        l_walls = []

        determinant = (
            a_facade / (2 * self.n_storey * self.h_ceiling)
        ) ** 2 - 4 * self.a_ground_floor

        # checks if determinant is positive
        if determinant < 0:
            # negative
            # follow ambience's assumption of an aspect ratio of 1.5
            l_wall_side = np.sqrt(self.a_ground_floor / 1.5)
            l_wall_front = 1.5 * l_wall_side

        else:
            # positive
            # follow ambience's equation for wall lengths
            l_walls.append(
                (
                    (a_facade / (2 * self.n_storey * self.h_ceiling))
                    + np.sqrt(
                        (a_facade / (2 * self.n_storey * self.h_ceiling)) ** 2
                        - 4 * self.a_ground_floor
                    )
                )
                / 2
            )

            l_walls.append(
                (
                    (a_facade / (2 * self.n_storey * self.h_ceiling))
                    - np.sqrt(
                        (a_facade / (2 * self.n_storey * self.h_ceiling)) ** 2
                        - 4 * self.a_ground_floor
                    )
                )
                / 2
            )

            l_wall_front = l_walls[0]
            l_wall_side = l_walls[1]

        return l_wall_front, l_wall_side

    def get_roof_height(self):
        """Calculates the roof height provided the roof is
        split in two equal sized elements"""
        return (np.sqrt((self.l_wall_side**2) * ((self.r_floor_roof**2) - 1))) / 2

    def get_roof_coordinates(self):
        """Determines roof coordinates based on an idealised pitched roof"""

        # Future improvement will need to deal
        # with different aspect ratios and different roof shapes

        roof_coords = [
            [
                [self.l_wall_front, 0, self.n_storey * self.h_ceiling],
                [
                    self.l_wall_front,
                    self.l_wall_side / 2,
                    self.n_storey * self.h_ceiling + self.h_roof,
                ],
                [0, self.l_wall_side / 2, self.n_storey * self.h_ceiling + self.h_roof],
                [0, 0, self.n_storey * self.h_ceiling],
            ],
            [
                [self.l_wall_front, self.l_wall_side, self.n_storey * self.h_ceiling],
                [
                    self.l_wall_front,
                    self.l_wall_side / 2,
                    self.n_storey * self.h_ceiling + self.h_roof,
                ],
                [0, self.l_wall_side / 2, self.n_storey * self.h_ceiling + self.h_roof],
                [0, self.l_wall_front, self.n_storey * self.h_ceiling],
            ],
        ]

        return roof_coords

    def get_roof_wall_coordinates(self):
        """Determines roof-level wall coordinates based on an idealised pitched roof"""

        wall_coords = [
            [
                [0, 0, self.n_storey * self.h_ceiling],
                [0, self.l_wall_side, self.n_storey * self.h_ceiling],
                [0, self.l_wall_side / 2, self.n_storey * self.h_ceiling + self.h_roof],
                [0, self.l_wall_side / 2, self.n_storey * self.h_ceiling + self.h_roof],
            ],
            [
                [self.l_wall_front, 0, self.n_storey * self.h_ceiling],
                [self.l_wall_front, self.l_wall_side, self.n_storey * self.h_ceiling],
                [
                    self.l_wall_front,
                    self.l_wall_side / 2,
                    self.n_storey * self.h_ceiling + self.h_roof,
                ],
                [
                    self.l_wall_front,
                    self.l_wall_side / 2,
                    self.n_storey * self.h_ceiling + self.h_roof,
                ],
            ],
        ]

        return wall_coords

    def add_roof(self):
        """Gets roof height, coordinates of roof and walls,
        then creates new roof and wall elements in e+
        then assigns coordinates of the new elements"""

        roof_coords = self.get_roof_coordinates()

        wall_coords = self.get_roof_wall_coordinates()

        self.idf.newidfobject(
            "ZONE",
            Name="Roof Space",
        )

        # May want to change nomenclature on naming new elements
        # Currently N_X means that there are X of the new elements,
        # and N designates what element you are adding

        self.idf.newidfobject(
            "BUILDINGSURFACE:DETAILED",
            Name="roof_1_2",
            Construction_Name="REFERENCE ROOF",
            Surface_Type="roof",
            Zone_Name="Roof Space",
        )

        self.idf.newidfobject(
            "BUILDINGSURFACE:DETAILED",
            Name="roof_2_2",
            Construction_Name="REFERENCE ROOF",
            Surface_Type="roof",
            Zone_Name="Roof Space",
        )

        self.idf.newidfobject(
            "BUILDINGSURFACE:DETAILED",
            Name="wall_1_2",
            Construction_Name="REFERENCE WALL",
            Surface_Type="wall",
            Zone_Name="Roof Space",
        )

        self.idf.newidfobject(
            "BUILDINGSURFACE:DETAILED",
            Name="wall_2_2",
            Construction_Name="REFERENCE WALL",
            Surface_Type="wall",
            Zone_Name="Roof Space",
        )

        for index, roof in enumerate(self.idf.getsurfaces("roof")):
            roof.Vertex_1_Xcoordinate = roof_coords[index][0][0]
            roof.Vertex_1_Ycoordinate = roof_coords[index][0][1]
            roof.Vertex_1_Zcoordinate = roof_coords[index][0][2]
            roof.Vertex_2_Xcoordinate = roof_coords[index][1][0]
            roof.Vertex_2_Ycoordinate = roof_coords[index][1][1]
            roof.Vertex_2_Zcoordinate = roof_coords[index][1][2]
            roof.Vertex_3_Xcoordinate = roof_coords[index][2][0]
            roof.Vertex_3_Ycoordinate = roof_coords[index][2][1]
            roof.Vertex_3_Zcoordinate = roof_coords[index][2][2]
            roof.Vertex_4_Xcoordinate = roof_coords[index][3][0]
            roof.Vertex_4_Ycoordinate = roof_coords[index][3][1]
            roof.Vertex_4_Zcoordinate = roof_coords[index][3][2]

        count = 0
        for index, wall in enumerate(self.idf.getsurfaces("wall")):
            if self.idf.getsurfaces("wall")[index].Zone_Name == "Roof Space":
                wall.Vertex_1_Xcoordinate = wall_coords[count][0][0]
                wall.Vertex_1_Ycoordinate = wall_coords[count][0][1]
                wall.Vertex_1_Zcoordinate = wall_coords[count][0][2]
                wall.Vertex_2_Xcoordinate = wall_coords[count][1][0]
                wall.Vertex_2_Ycoordinate = wall_coords[count][1][1]
                wall.Vertex_2_Zcoordinate = wall_coords[count][1][2]
                wall.Vertex_3_Xcoordinate = wall_coords[count][2][0]
                wall.Vertex_3_Ycoordinate = wall_coords[count][2][1]
                wall.Vertex_3_Zcoordinate = wall_coords[count][2][2]
                wall.Vertex_4_Xcoordinate = wall_coords[count][3][0]
                wall.Vertex_4_Ycoordinate = wall_coords[count][3][1]
                wall.Vertex_4_Zcoordinate = wall_coords[count][3][2]
                count = count + 1

    def set_constructions(self):
        """adds materials and constructions to IDF
        then assigns each of the constructions to surfaces"""

        for c in [
            self.wall_construction,
            self.roof_construction,
            self.ground_floor_construction,
            self.floor_construction,
            self.ceiling_construction,
        ]:
            if c.materials:
                self.idf = c.add_to_idf(self.idf)

        for surface in self.idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
            if surface.Surface_Type == "wall":
                surface.Construction_Name = self.wall_construction.get_name()
            elif surface.Surface_Type == "roof":
                surface.Construction_Name = self.roof_construction.get_name()
            elif surface.Surface_Type == "floor":
                if surface.Vertex_1_Zcoordinate < 0.1:
                    surface.Construction_Name = (
                        self.ground_floor_construction.get_name()
                    )
                else:
                    surface.Construction_Name = self.floor_construction.get_name()
            elif surface.Surface_Type == "ceiling":
                surface.Construction_Name = self.ceiling_construction.get_name()

        # windows
        self.idf = self.window_construction.add_to_idf(self.idf)
        for window in self.idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
            window.Construction_Name = self.window_construction.get_name()

    def get_system_data(self):
        """Reads the sampled building system information and puts it in a
        format for e+ to read"""

        # Future - will need to model more energy systems other than boilers
        # e.g. heat pumps, stoves, electrical heater etc.
        # Future - will need to model biomass and double check if Solid and Liquid fuel
        # in Ambience is actually coal and Diesal etc

        if "boiler" in self.heating_system_type:

            if "non-condensing" in self.heating_system_type:

                technology = "HotWaterBoiler"
            else:
                technology = "CondensingHotWaterBoiler"

            if self.heating_system_fuel == "Gas":
                fuel = "NaturalGas"
            if self.heating_system_fuel == "Liquid":
                fuel = "Diesel"  # Double check
            if self.heating_system_fuel == "Electricity":
                fuel = "Electricity"
            if self.heating_system_fuel == "Biomass":
                fuel = "Coal"  # Double check
            if self.heating_system_fuel == "Solid":
                fuel = "Coal"  # Double check

        else:
            print(self.heating_system_fuel + " not yet handled by " + __name__)

        return technology, fuel

    def add_heating_system(self):
        """Gets the system data from get_system_data method and then adds
        in a heating system"""

        # Future - will need to use templates for other energy systems

        tech, fuel = self.get_system_data()

        stat = self.idf.newidfobject(
            "HVACTEMPLATE:THERMOSTAT",
            Name="Thermostat",
            Heating_Setpoint_Schedule_Name="Heating-Setpoints",
            Cooling_Setpoint_Schedule_Name="Cooling-Setpoints",
        )

        for zone in self.idf.idfobjects["ZONE"]:
            self.idf.newidfobject(
                "HVACTEMPLATE:ZONE:BASEBOARDHEAT",
                Zone_Name=zone.Name,
                Baseboard_Heating_Type="HotWater",
                Template_Thermostat_Name=stat.Name,
            )

        self.idf.newidfobject("HVACTEMPLATE:PLANT:HOTWATERLOOP", Name="Hot Water Loop")

        self.idf.newidfobject(
            "HVACTEMPLATE:PLANT:BOILER",
            Name="Main Boiler",
            Boiler_Type=tech,
            Efficiency=self.heating_system_efficiency,
            Fuel_Type=fuel,
        )

        self.idf.idfobjects["SIMULATIONCONTROL"][0].Do_Zone_Sizing_Calculation = "Yes"

    def add_schedules(self):
        """Adds schedules into e+"""
        self.idf.newidfobject(
            "SCHEDULE:COMPACT",
            Name="People-Schedule",
            Field_1=(
                "Through: 12/31,\n    "
                "For: Weekdays,\n    Until: 9:00, 1.0,\n"
                "    Until:17:00, 0.5,\n    Until:24:00, 1.,\n "
                "   For:AllOtherDays,\n    Until:24:00,1."
            ),
        )
        self.idf.newidfobject(
            "SCHEDULE:COMPACT",
            Name="Always-Schedule",
            Field_1="Through: 12/31,\n    For: AllDays,\n    Until: 24:00, 1.0\n",
        )
        self.idf.newidfobject(
            "SCHEDULE:COMPACT",
            Name="Activity-Schedule",
            Field_1="Through: 12/31,\n    For: AllDays,\n    Until: 24:00, 100.\n",
        )
        self.idf.newidfobject(
            "SCHEDULE:COMPACT",
            Name="Heating-Setpoints",
            Field_1="Through: 12/31,\n    For: AllDays,\n    Until: 24:00, 20.\n",
        )
        self.idf.newidfobject(
            "SCHEDULE:COMPACT",
            Name="Cooling-Setpoints",
            Field_1="Through: 12/31,\n    For: AllDays,\n    Until: 24:00, 25.\n",
        )

    def add_people(self):
        """Adds people into e+ for every zone in idf"""

        for zone in self.idf.idfobjects["ZONE"]:
            self.idf.newidfobject(
                "PEOPLE",
                Name=zone.Name + "-People",
                Zone_or_ZoneList_Name=zone.Name,
                Number_of_People_Schedule_Name="People-Schedule",
                Number_of_People=2,
                Activity_Level_Schedule_Name="Activity-Schedule",
            )

    def add_ventilation(self):
        """Adds ventilation into e+ for every zone in idf"""
        for zone in self.idf.idfobjects["ZONE"]:
            self.idf.newidfobject(
                "ZONEVENTILATION:DESIGNFLOWRATE",
                Name=zone.Name + "-Ventilation",
                Zone_or_ZoneList_Name=zone.Name,
                Design_Flow_Rate_Calculation_Method="Flow/Person",
                Flow_Rate_per_Person=0.01,
                Schedule_Name="Always-Schedule",
            )

    def add_infiltration(self):
        """Adds infiltration into e+ for every zone in idf"""
        for zone in self.idf.idfobjects["ZONE"]:
            self.idf.newidfobject(
                "ZONEINFILTRATION:DESIGNFLOWRATE",
                Name=zone.Name + "-Infiltration",
                Zone_or_ZoneList_Name=zone.Name,
                Design_Flow_Rate_Calculation_Method="Flow/ExteriorArea",
                Flow_per_Exterior_Surface_Area=15 * 0.07,
                Constant_Term_Coefficient=0.606,
                Temperature_Term_Coefficient=0.03636,
                Velocity_Term_Coeﬀicient=0.1177,
                Velocity_Squared_Term_Coefficient=0.0,
                Schedule_Name="Always-Schedule",
            )

    def add_internal_gains(self):
        """Adds internal gains into e+ for every zone in idf"""
        for zone in self.idf.idfobjects["ZONE"]:
            self.idf.newidfobject(
                "LIGHTS",
                Name=zone.Name + "-Lights",
                Zone_or_ZoneList_Name=zone.Name,
                Schedule_Name="Always-Schedule",
                Design_Level_Calculation_Method="Watts/area",
                Watts_per_Zone_Floor_Area=1,
            )

            self.idf.newidfobject(
                "ELECTRICEQUIPMENT",
                Name=zone.Name + "-Equipment",
                Zone_or_ZoneList_Name=zone.Name,
                Schedule_Name="Always-Schedule",
                Design_Level_Calculation_Method="Watts/area",
                Watts_per_Zone_Floor_Area=5,
            )

    def build(self):
        """Creates the building"""
        # May want to change the block name - again nomenclature
        self.idf.add_block(
            name="Living",
            coordinates=[
                (self.l_wall_front, 0),
                (self.l_wall_front, self.l_wall_side),
                (0, self.l_wall_side),
                (0, 0),
            ],
            height=self.n_storey * self.h_ceiling,
            num_stories=self.n_storey,
        )

        # self.idf.set_default_constructions()

        if self.r_floor_roof != 1:

            for index, surface in enumerate(
                self.idf.idfobjects["BUILDINGSURFACE:DETAILED"]
            ):
                if surface.Surface_Type == "roof":
                    self.idf.removeidfobject(
                        self.idf.idfobjects["BUILDINGSURFACE:DETAILED"][index]
                    )

            self.add_roof()

        self.idf.intersect_match()

        self.idf.set_wwr(
            wwr=self.a_window / self.a_wall, construction="Project External Window"
        )

        if self.r_floor_roof != 1:
            self.idf.idfobjects["FENESTRATIONSURFACE:DETAILED"].pop(-1)
            self.idf.idfobjects["FENESTRATIONSURFACE:DETAILED"].pop(-1)

        self.set_constructions()
        self.add_heating_system()
        self.add_schedules()
        self.add_people()
        self.add_ventilation()
        self.add_infiltration()
        self.add_internal_gains()

        # remove design days:
        self.idf.idfobjects["SIZINGPERIOD:DESIGNDAY"].clear()
        # add design period:
        self.idf.newidfobject(
            "SIZINGPERIOD:WEATHERFILEDAYS",
            Name="Winter Design Day",
            Begin_Month=1,
            Begin_Day_of_Month=1,
            End_Month=1,
            End_Day_of_Month=14,
        )

        self.idf.newidfobject(
            "SIZINGPERIOD:WEATHERFILEDAYS",
            Name="Summer Design Day",
            Begin_Month=7,
            Begin_Day_of_Month=1,
            End_Month=7,
            End_Day_of_Month=14,
        )

        return self.idf

    def get_idf(self):
        return self.idf
