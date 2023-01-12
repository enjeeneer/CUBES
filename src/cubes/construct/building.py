"""Defines the Building class """

from cubes.construct.constants import EPLUS_PATH
from cubes.construct import material as mat

from geomeppy import IDF


class Building:

    """This class holds all the information and methods to produce an IDF file"""

    def __init__(self, building_config):
        """This constructor is for use with Data from the Ambience database

        Args:
            building_config (buildingconfig object): this is an instance of the
                                                     buildingconfig dataclass
        """
        self.building_config = building_config

        self.wall_construction = mat.Construction(
            "Wall",
            building_config.wall_layer_materials,
            building_config.wall_layer_thickness,
        )
        self.ground_floor_construction = mat.Construction(
            "GroundFloor",
            building_config.ground_floor_layer_materials,
            building_config.ground_floor_layer_thickness,
        )
        self.roof_construction = mat.Construction(
            "Roof",
            building_config.roof_layer_materials,
            building_config.roof_layer_thickness,
        )
        self.upper_floor_construction = mat.Construction(
            "Floor",
            building_config.upper_floor_layer_materials,
            building_config.upper_floor_layer_thickness,
        )
        self.ceiling_construction = mat.Construction(
            "Ceiling",
            building_config.ceiling_layer_materials,
            building_config.ceiling_layer_thickness,
        )
        self.window_construction = mat.WindowConstruction(
            building_config.window_type,
            building_config.window_layer_materials,
            building_config.window_layer_thickness,
        )

        IDF.setiddname(EPLUS_PATH + "Energy+.idd")
        self.idf = IDF(EPLUS_PATH + "ExampleFiles/Minimal.idf")
        # Future - Will need to automatically add in weather file based on locations
        self.idf.epw = EPLUS_PATH + "WeatherData/USA_CO_Golden-NREL.724666_TMY3.epw"

    def set_constructions(self):
        """adds materials and constructions to IDF
        then assigns each of the constructions to surfaces
        """

        for c in [
            self.wall_construction,
            self.roof_construction,
            self.ground_floor_construction,
            self.upper_floor_construction,
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
                    surface.Construction_Name = self.upper_floor_construction.get_name()
            elif surface.Surface_Type == "ceiling":
                surface.Construction_Name = self.ceiling_construction.get_name()

        # windows
        self.idf = self.window_construction.add_to_idf(self.idf)
        for window in self.idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
            window.Construction_Name = self.window_construction.get_name()

    def add_heating_system(self):
        """Gets the system data from get_system_data method and then adds
        in a heating system. Current template knowledge limits us to boilers"""

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
            Boiler_Type=self.building_config.heating_system_type,
            Efficiency=self.building_config.heating_system_efficiency,
            Fuel_Type=self.building_config.heating_system_fuel,
        )

        self.idf.idfobjects["SIMULATIONCONTROL"][0].Do_Zone_Sizing_Calculation = "Yes"

    def add_schedules(self):
        """Adds schedules into e+. Currently hardcoded, will need to add a feature later
        on
        """

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

    def add_environmental_impact_factors(self):
        self.idf.newidfobject(
            "FUELFACTORS",
            Existing_Fuel_Resource_Name="NaturalGas",
            CO2_Emission_Factor=56,
        )
        self.idf.newidfobject(
            "FUELFACTORS",
            Existing_Fuel_Resource_Name="Electricity",
            CO2_Emission_Factor=56,
        )
        self.idf.newidfobject("ENVIRONMENTALIMPACTFACTORS")

    def set_design_days(self):

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

    def build(self):
        """method which can construct or 'build' our archetypal building

        Returns:
            idf: idf is the input data file which can be used by energyplus
        """
        # Nomenclature on block can be changed in future
        self.idf.add_block(
            name="Living",
            coordinates=[
                (self.building_config.l_wall_x, 0),
                (self.building_config.l_wall_x, self.building_config.l_wall_y),
                (0, self.building_config.l_wall_y),
                (0, 0),
            ],
            height=self.building_config.n_storey * self.building_config.h_storey,
            num_stories=self.building_config.n_storey,
        )

        self.add_roof()
        self.idf.intersect_match()
        self.add_windows()
        self.set_constructions()
        self.add_heating_system()
        self.add_schedules()
        self.add_people()
        self.add_ventilation()
        self.add_infiltration()
        self.add_internal_gains()
        self.add_environmental_impact_factors()
        self.set_design_days()

        return self.idf

    def add_windows(self):
        """method which adds window strips into idf and then deletes the windows added
        to roof space"""

        self.idf.set_wwr(
            wwr=0.00001,
            wwr_map={
                0: self.building_config.wtw_ratios[0],
                90: self.building_config.wtw_ratios[1],
                180: self.building_config.wtw_ratios[2],
                270: self.building_config.wtw_ratios[3],
            },
            construction="Window-Construction",
        )

        # the code above adds a strip of windows to each storey, including roof space
        # this needs to be removed

        if self.building_config.roof_type != "flat":
            self.idf.idfobjects["FENESTRATIONSURFACE:DETAILED"].pop(-1)
            self.idf.idfobjects["FENESTRATIONSURFACE:DETAILED"].pop(-1)

    def get_roof_coordinates(self):
        """Determines roof coordinates based on a saddleback roof template

        Returns:
            roof_coords list of lists: roof_coords are the coordinates of each point
                                       of the roof, there are two roof segments, each
                                       with four points, with each point having an
                                       (x,y,z) coordinate
                                       geometry rules: starting upper left corner,
                                       counterclockwise
        """

        roof_coords = [
            [
                [
                    0,
                    self.building_config.l_wall_y / 2,
                    self.building_config.n_storey * self.building_config.h_storey
                    + self.building_config.h_roof,
                ],
                [0, 0, self.building_config.n_storey * self.building_config.h_storey],
                [
                    self.building_config.l_wall_x,
                    0,
                    self.building_config.n_storey * self.building_config.h_storey,
                ],
                [
                    self.building_config.l_wall_x,
                    self.building_config.l_wall_y / 2,
                    self.building_config.n_storey * self.building_config.h_storey
                    + self.building_config.h_roof,
                ],
            ],
            [
                [
                    self.building_config.l_wall_x,
                    self.building_config.l_wall_y / 2,
                    self.building_config.n_storey * self.building_config.h_storey
                    + self.building_config.h_roof,
                ],
                [
                    self.building_config.l_wall_x,
                    self.building_config.l_wall_y,
                    self.building_config.n_storey * self.building_config.h_storey,
                ],
                [
                    0,
                    self.building_config.l_wall_y,
                    self.building_config.n_storey * self.building_config.h_storey,
                ],
                [
                    0,
                    self.building_config.l_wall_y / 2,
                    self.building_config.n_storey * self.building_config.h_storey
                    + self.building_config.h_roof,
                ],
            ],
        ]

        return roof_coords

    def get_roof_wall_coordinates(self):
        """Determines roof-level wall coordinates based on an idealised pitched roof

        Returns:
            wall_coords list of lists: wall_coords are the coordinates of each point of
                                       the wall, there are two wall segments, each with
                                       four points, with each point having an
                                       (x,y,z) coordinate
        """

        wall_coords = [
            [
                [0, 0, self.building_config.n_storey * self.building_config.h_storey],
                [
                    0,
                    self.building_config.l_wall_y,
                    self.building_config.n_storey * self.building_config.h_storey,
                ],
                [
                    0,
                    self.building_config.l_wall_y / 2,
                    self.building_config.n_storey * self.building_config.h_storey
                    + self.building_config.h_roof,
                ],
                [
                    0,
                    self.building_config.l_wall_y / 2,
                    self.building_config.n_storey * self.building_config.h_storey
                    + self.building_config.h_roof,
                ],
            ],
            [
                [
                    self.building_config.l_wall_x,
                    0,
                    self.building_config.n_storey * self.building_config.h_storey,
                ],
                [
                    self.building_config.l_wall_x,
                    self.building_config.l_wall_y,
                    self.building_config.n_storey * self.building_config.h_storey,
                ],
                [
                    self.building_config.l_wall_x,
                    self.building_config.l_wall_y / 2,
                    self.building_config.n_storey * self.building_config.h_storey
                    + self.building_config.h_roof,
                ],
                [
                    self.building_config.l_wall_x,
                    self.building_config.l_wall_y / 2,
                    self.building_config.n_storey * self.building_config.h_storey
                    + self.building_config.h_roof,
                ],
            ],
        ]

        return wall_coords

    def add_roof(self):
        """Initially checks if the roof is flat, if it is then the original
        geomeppy flat roof created by the idf.add_block method works. If not then the
        method gets roof height, coordinates of roof and roof space walls, then creates
        a new roof and wall elements in e+ and assigns coordinates of the new
        elements"""

        if self.building_config.roof_type != "flat":

            for index, surface in enumerate(
                self.idf.idfobjects["BUILDINGSURFACE:DETAILED"]
            ):

                if surface.Surface_Type == "roof":

                    self.idf.removeidfobject(
                        self.idf.idfobjects["BUILDINGSURFACE:DETAILED"][index]
                    )

                    self.idf.newidfobject(
                        "BUILDINGSURFACE:DETAILED",
                        Name="attic floor",
                        Surface_Type="floor",
                        Zone_Name="ROOF SPACE",
                        Vertex_1_Xcoordinate=surface.Vertex_1_Xcoordinate,
                        Vertex_1_Ycoordinate=surface.Vertex_1_Ycoordinate,
                        Vertex_1_Zcoordinate=surface.Vertex_1_Zcoordinate,
                        Vertex_2_Xcoordinate=surface.Vertex_2_Xcoordinate,
                        Vertex_2_Ycoordinate=surface.Vertex_2_Ycoordinate,
                        Vertex_2_Zcoordinate=surface.Vertex_2_Zcoordinate,
                        Vertex_3_Xcoordinate=surface.Vertex_3_Xcoordinate,
                        Vertex_3_Ycoordinate=surface.Vertex_3_Ycoordinate,
                        Vertex_3_Zcoordinate=surface.Vertex_3_Zcoordinate,
                        Vertex_4_Xcoordinate=surface.Vertex_4_Xcoordinate,
                        Vertex_4_Ycoordinate=surface.Vertex_4_Ycoordinate,
                        Vertex_4_Zcoordinate=surface.Vertex_4_Zcoordinate,
                    )

                    # search for zone name of last storey
                    last_storey_zone_name = "UNKNOWN"
                    for zone in self.idf.idfobjects["ZONE"]:
                        if str(self.building_config.n_storey - 1) in zone.Name:
                            last_storey_zone_name = zone.Name

                    self.idf.newidfobject(
                        "BUILDINGSURFACE:DETAILED",
                        Name="storey "
                        + str(self.building_config.n_storey)
                        + " ceiling",
                        Surface_Type="ceiling",
                        Zone_Name=last_storey_zone_name,
                        Vertex_1_Xcoordinate=surface.Vertex_1_Xcoordinate,
                        Vertex_1_Ycoordinate=surface.Vertex_1_Ycoordinate,
                        Vertex_1_Zcoordinate=surface.Vertex_1_Zcoordinate,
                        Vertex_2_Xcoordinate=surface.Vertex_2_Xcoordinate,
                        Vertex_2_Ycoordinate=surface.Vertex_2_Ycoordinate,
                        Vertex_2_Zcoordinate=surface.Vertex_2_Zcoordinate,
                        Vertex_3_Xcoordinate=surface.Vertex_3_Xcoordinate,
                        Vertex_3_Ycoordinate=surface.Vertex_3_Ycoordinate,
                        Vertex_3_Zcoordinate=surface.Vertex_3_Zcoordinate,
                        Vertex_4_Xcoordinate=surface.Vertex_4_Xcoordinate,
                        Vertex_4_Ycoordinate=surface.Vertex_4_Ycoordinate,
                        Vertex_4_Zcoordinate=surface.Vertex_4_Zcoordinate,
                    )

            roof_coords = self.get_roof_coordinates()

            wall_coords = self.get_roof_wall_coordinates()

            self.idf.newidfobject(
                "ZONE",
                Name="ROOF SPACE",
            )

            # May want to change nomenclature on naming new elements
            # Currently N_X means that there are X of the new elements,
            # and N designates what element you are adding

            self.idf.newidfobject(
                "BUILDINGSURFACE:DETAILED",
                Name="roof_1_2",
                Construction_Name="ROOF-Construction",
                Surface_Type="ROOF",
                Zone_Name="ROOF SPACE",
            )

            self.idf.newidfobject(
                "BUILDINGSURFACE:DETAILED",
                Name="roof_2_2",
                Construction_Name="ROOF-Construction",
                Surface_Type="ROOF",
                Zone_Name="ROOF SPACE",
            )

            self.idf.newidfobject(
                "BUILDINGSURFACE:DETAILED",
                Name="wall_1_2",
                Construction_Name="WALL-Construction",
                Surface_Type="WALL",
                Zone_Name="ROOF SPACE",
            )

            self.idf.newidfobject(
                "BUILDINGSURFACE:DETAILED",
                Name="wall_2_2",
                Construction_Name="WALL-Construction",
                Surface_Type="WALL",
                Zone_Name="ROOF SPACE",
            )

            for index, roof in enumerate(self.idf.getsurfaces("ROOF")):
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
                for index, wall in enumerate(self.idf.getsurfaces("WALL")):
                    if self.idf.getsurfaces("WALL")[index].Zone_Name == "ROOF SPACE":
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

    def get_idf(self):
        return self.idf
