"""Defines the Building class """

from typing import Dict
from cubes.construct import material as mat
from cubes.construct import utilities
from cubes.construct.buildingconfig import BuildingConfig
from cubes.construct.hvac_systems import add_heating_system
from cubes.construct.pv_and_battery import add_pv_and_battery
from cubes.construct.roof import add_roof
from cubes.construct.utilities import get_schedule
import cubes.construct.buildingconfig_options as bco
from cubes.constants import package_directory, EPLUS_PATH
from cubes.package.constants import env_files_path
from geomeppy import IDF


class Building:

    """This class holds all the information and methods to produce an IDF file"""

    building_config: BuildingConfig

    def __init__(self, building_config: BuildingConfig, materials: Dict, windows: Dict):
        """This constructor is for with a BuildingConfig object

        Args:
            building_config (buildingconfig object): an instance of the
                                                     buildingconfig dataclass
            materials (dict): a dictionary of materials data
            windows (dict): a dictionary of windows data
        """
        self.building_config = building_config
        self.materials = materials
        self.windows = windows

        self.wall_construction = mat.Construction(
            "Wall",
            [materials[x] for x in building_config.wall_layer_materials],
            building_config.wall_layer_thickness,
        )
        self.ground_floor_construction = mat.Construction(
            "GroundFloor",
            [materials[x] for x in building_config.ground_floor_layer_materials],
            building_config.ground_floor_layer_thickness,
        )
        self.roof_construction = mat.Construction(
            "Roof",
            [materials[x] for x in building_config.roof_layer_materials],
            building_config.roof_layer_thickness,
        )
        self.upper_floor_construction = mat.Construction(
            "Floor",
            [materials[x] for x in building_config.upper_floor_layer_materials],
            building_config.upper_floor_layer_thickness,
        )
        self.ceiling_construction = mat.Construction(
            "Ceiling",
            [materials[x] for x in building_config.upper_floor_layer_materials[::-1]],
            building_config.upper_floor_layer_thickness[::-1],
        )
        self.partition_construction = mat.Construction(
            "InternalMass",
            [materials[x] for x in building_config.partition_layer_materials[::-1]],
            building_config.partition_layer_thickness[::-1],
        )

        self.all_constructions = [
            self.wall_construction,
            self.roof_construction,
            self.ground_floor_construction,
            self.upper_floor_construction,
            self.ceiling_construction,
        ]

        if self.building_config.attic_floor_layer_materials:
            self.last_floor_construction = mat.Construction(
                "Last floor",
                [materials[x] for x in building_config.attic_floor_layer_materials],
                building_config.attic_floor_layer_thickness,
            )
            self.all_constructions.append(self.last_floor_construction)
            self.last_ceiling_construction = mat.Construction(
                "Last ceiling",
                [
                    materials[x]
                    for x in building_config.attic_floor_layer_materials[::-1]
                ],
                building_config.attic_floor_layer_thickness[::-1],
            )
            self.all_constructions.append(self.last_ceiling_construction)

        if building_config.window_type == "Simple":
            if building_config.window_simple_values:
                self.window_system_simple = mat.WindowMaterialSimpleGlazing(
                    "Simple glazing", *building_config.window_simple_values
                )
            else:
                print(
                    "Simple glazing selected, but no values given. "
                    "Using default values."
                )
                self.window_system_simple = mat.WindowMaterialSimpleGlazing(
                    "Default glazing", 3, 0.8, 0.8
                )
        else:
            self.window_construction = mat.WindowConstruction(
                building_config.window_type,
                building_config.window_layer_materials,
                building_config.window_layer_thickness,
            )

        IDF.setiddname(EPLUS_PATH + "Energy+.idd")
        self.idf = IDF(EPLUS_PATH + "ExampleFiles/Minimal.idf")

        self.idf.idfobjects["GLOBALGEOMETRYRULES"][0].Coordinate_System = "Relative"
        self.idf.idfobjects["BUILDING"][0].Solar_Distribution = "FullExterior"
        self.idf.idfobjects["TIMESTEP"][0].Number_of_Timesteps_per_Hour = 6

    def set_constructions(self):
        """adds materials and constructions to IDF
        then assigns each of the constructions to surfaces
        """

        for c in self.all_constructions:
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
                elif (
                    self.building_config.roof_type != "flat"
                    and surface.Vertex_1_Zcoordinate
                    > self.building_config.storey_height
                    * self.building_config.number_of_stories
                    - 0.1
                    and self.building_config.attic_floor_layer_materials
                ):
                    surface.Construction_Name = self.last_floor_construction.get_name()
                else:
                    surface.Construction_Name = self.upper_floor_construction.get_name()
            elif surface.Surface_Type == "ceiling":
                if (
                    self.building_config.roof_type != "flat"
                    and surface.Vertex_1_Zcoordinate
                    > self.building_config.storey_height
                    * self.building_config.number_of_stories
                    - 0.1
                    and self.building_config.attic_floor_layer_materials
                ):
                    surface.Construction_Name = (
                        self.last_ceiling_construction.get_name()
                    )

                else:
                    surface.Construction_Name = self.ceiling_construction.get_name()

        # windows
        if self.building_config.window_type != "Simple":
            self.idf = self.window_construction.add_to_idf(
                self.idf, windows=self.windows
            )
            for window in self.idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
                window.Construction_Name = self.window_construction.get_name()
        else:
            self.idf = self.window_system_simple.add_to_idf(self.idf)
            for window in self.idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
                window.Construction_Name = "Glazing"

    def zone_not_heated(self, zone_name):
        return zone_name == "ROOF SPACE" and self.building_config.attic_is_heated

    def get_heated_zones(self):
        zones = []
        for zone in self.idf.idfobjects["ZONE"]:
            if self.zone_not_heated(zone.Name):
                continue
            zones.append(zone)
        return zones

    def add_schedules(self):
        """Adds schedules into e+.
        on
        """

        # occupants
        if self.building_config.occupant_schedule:
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="People-Schedule",
                Field_1=get_schedule(self.building_config.occupant_schedule),
            )
        else:
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

        # defaults
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
        # temperature setpoints
        if self.building_config.heating_setpoint_schedule:
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Heating-Setpoint-Schedule",
                Field_1=get_schedule(self.building_config.heating_setpoint_schedule),
            )
        else:
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Heating-Setpoint-Schedule",
                Field_1="Through: 12/31,\n    For: AllDays,\n    Until: 24:00, 20.\n",
            )

        if self.building_config.cooling_setpoint_schedule:
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Cooling-Setpoint-Schedule",
                Field_1=get_schedule(self.building_config.cooling_setpoint_schedule),
            )
        else:
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Cooling-Setpoint-Schedule",
                Field_1="Through: 12/31,\n    For: AllDays,\n    Until: 24:00, 25.\n",
            )
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Radiant-System-Schedule",
                Field_1="Through: 12/31,\n    For: AllDays,\n    Until: 24:00, 20.\n",
            )

        # lighting
        if self.building_config.lighting_schedule:
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Lighting-Schedule",
                Field_1=get_schedule(self.building_config.lighting_schedule),
            )
        else:
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Lighting-Schedule",
                Field_1=(
                    "Through: 12/31,\n    "
                    "For: AllDays,\n    Until: 6:00, 0.1,\n"
                    "    Until:23:00, 1,\n    Until:24:00, 0.1;"
                ),
            )

        # equipment
        if self.building_config.equipment_gain_schedule:
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Equipment-Schedule",
                Field_1=get_schedule(self.building_config.equipment_gain_schedule),
            )
        else:
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Equipment-Schedule",
                Field_1=(
                    "Through: 12/31,\n    "
                    "For: AllDays,\n    Until: 6:00, 0.1,\n"
                    "    Until:23:00, 1,\n    Until:24:00, 0.1;"
                ),
            )

    def add_people(self):
        """Adds people into e+ for every zone in idf"""

        for zone in self.get_heated_zones():

            self.idf.newidfobject(
                "PEOPLE",
                Name=zone.Name + "-People",
                Zone_or_ZoneList_Name=zone.Name,
                Number_of_People_Calculation_Method=(
                    self.building_config.occupant_number_calculation_method
                ),
                Number_of_People_Schedule_Name="People-Schedule",
                Number_of_People=self.building_config.occupant_value,
                People_per_Zone_Floor_Area=self.building_config.occupant_value,
                Zone_Floor_Area_per_Person=self.building_config.occupant_value,
                Activity_Level_Schedule_Name="Activity-Schedule",
            )

    def add_ventilation(self):
        """Adds ventilation into e+ for every zone in idf"""

        self.idf.newidfobject(
            "ZONEAIRCONTAMINANTBALANCE",
            Carbon_Dioxide_Concentration="Yes",
            Outdoor_Carbon_Dioxide_Schedule_Name="Outdoor CO2 Schedule",
        )

        self.idf.newidfobject(
            "SCHEDULE:CONSTANT", Name="Outdoor CO2 Schedule", Hourly_Value=420.0
        )

        if self.building_config.ventilation_method in [
            bco.VentilationMethod.RATE_PER_OCCUPANT.value,
            bco.VentilationMethod.RATE_PER_OCCUPANT_PLUS_COOLING.value,
        ]:
            if (
                self.building_config.ventilation_type
                == bco.VentilationType.MECHANICAL.value
            ):
                vtype = "Balanced"
            else:
                vtype = "Natural"

            for zone in self.get_heated_zones():
                self.idf.newidfobject(
                    "ZONEVENTILATION:DESIGNFLOWRATE",
                    Name=zone.Name + "-Ventilation",
                    Zone_or_ZoneList_Name=zone.Name,
                    Schedule_Name="People-Schedule",
                    Design_Flow_Rate_Calculation_Method=("Flow/Person"),
                    Flow_Rate_per_Person=(
                        self.building_config.ventilation_rate_per_occupant
                    ),
                    Ventilation_Type=vtype,
                    Fan_Pressure_Rise=(
                        self.building_config.mech_vent_fan_pressure_rise
                    ),
                    Fan_Total_Efficiency=(
                        self.building_config.mech_vent_fan_efficiency
                    ),
                )

        if (
            self.building_config.ventilation_method
            == bco.VentilationMethod.RATE_PER_OCCUPANT_PLUS_COOLING.value
        ):
            for zone in self.get_heated_zones():
                self.idf.newidfobject(
                    "ZONEVENTILATION:DESIGNFLOWRATE",
                    Name=zone.Name + "-Cooling Ventilation",
                    Zone_or_ZoneList_Name=zone.Name,
                    Schedule_Name="People-Schedule",
                    Design_Flow_Rate_Calculation_Method=("AirChanges/Hour"),
                    Air_Changes_per_Hour=self.building_config.nat_vent_rate,
                    Ventilation_Type="Natural",
                    Constant_Term_Coefficient=1,
                    Temperature_Term_Coefficient=0,
                    Velocity_Term_Coefficient=0,
                    Velocity_Squared_Term_Coefficient=0,
                    Minimum_Indoor_Temperature=(
                        self.building_config.cooling_setpoint - 1
                    ),
                    Minimum_Indoor_Temperature_Schedule_Name="",
                    Maximum_Indoor_Temperature=(
                        self.building_config.cooling_setpoint + 3
                    ),
                    Maximum_Indoor_Temperature_Schedule_Name="",
                    Delta_Temperature=1,
                )

        elif (
            self.building_config.ventilation_method
            == bco.VentilationMethod.RES_WIN_OP_MODEL.value
        ):
            for zone in self.get_heated_zones():
                self.idf.newidfobject(
                    "ZONEVENTILATION:DESIGNFLOWRATE",
                    Name=zone.Name + "-Natural Ventilation",
                    Zone_or_ZoneList_Name=zone.Name,
                    Schedule_Name=zone.Name + "-Ventilation-Schedule",
                    Design_Flow_Rate_Calculation_Method=("AirChanges/Hour"),
                    Air_Changes_per_Hour=self.building_config.nat_vent_rate,
                    Ventilation_Type="Natural",
                    Constant_Term_Coefficient=1,
                    Temperature_Term_Coefficient=0,
                    Velocity_Term_Coefficient=0,
                    Velocity_Squared_Term_Coefficient=0,
                )

                self.idf.newidfobject(
                    "SCHEDULE:CONSTANT",
                    Name=zone.Name + "-Ventilation-Schedule",
                    Hourly_Value=0.0,
                )

            self.idf.newidfobject(
                "PythonPlugin:Instance".upper(),
                Name="Ventilation Override",
                Run_During_Warmup_Days="Yes",
                Python_Module_Name="natural_ventilation_residential",
                Plugin_Class_Name=bco.res_window_PP_map[
                    self.building_config.ventilation_model
                ],
            )

            self.idf.newidfobject(
                "PythonPlugin:SearchPaths".upper(),
                Name="PythonPlugin search paths",
                Add_Current_Working_Directory_to_Search_Path="Yes",
                Add_Input_File_Directory_to_Search_Path="No",
                Search_Path_1=package_directory + "/behaviour_models",
            )

            with open(
                env_files_path + "/list_of_zones.txt", "w", encoding="utf-8"
            ) as filehandle:
                for listitem in self.get_heated_zones():
                    filehandle.write(f"{listitem.Name}\n")

    def add_infiltration(self):
        """Adds infiltration into e+ for every zone in idf"""
        for zone in self.idf.idfobjects["ZONE"]:
            self.idf.newidfobject(
                "ZONEINFILTRATION:DESIGNFLOWRATE",
                Name=zone.Name + "-Infiltration",
                Zone_or_ZoneList_Name=zone.Name,
                Design_Flow_Rate_Calculation_Method=(
                    self.building_config.infiltration_calculation_method
                ),
                Design_Flow_Rate=(self.building_config.infiltration_rate),
                Flow_per_Zone_Floor_Area=(self.building_config.infiltration_rate),
                Flow_per_Exterior_Surface_Area=(self.building_config.infiltration_rate),
                Air_Changes_per_Hour=(self.building_config.infiltration_rate),
                Constant_Term_Coefficient=0.606,
                Temperature_Term_Coefficient=0.03636,
                Velocity_Term_Coeﬀicient=0.1177,
                Velocity_Squared_Term_Coefficient=0.0,
                Schedule_Name="Always-Schedule",
            )

    def add_internal_gains(self):
        """Adds internal gains into e+ for every zone in idf"""
        for zone in self.get_heated_zones():
            self.idf.newidfobject(
                "LIGHTS",
                Name=zone.Name + "-Lights",
                Zone_or_ZoneList_Name=zone.Name,
                Schedule_Name="Lighting-Schedule",
                Design_Level_Calculation_Method=(
                    self.building_config.lighting_power_calculation_method
                ),
                Lighting_Level=(self.building_config.lighting_power_value),
                Watts_per_Zone_Floor_Area=(self.building_config.lighting_power_value),
                Watts_per_Person=(self.building_config.lighting_power_value),
            )

            self.idf.newidfobject(
                "ELECTRICEQUIPMENT",
                Name=zone.Name + "-Equipment",
                Zone_or_ZoneList_Name=zone.Name,
                Schedule_Name="Equipment-Schedule",
                Design_Level_Calculation_Method=(
                    self.building_config.equipment_gain_calculation_method
                ),
                Design_Level=(self.building_config.equipment_gain_value),
                Watts_per_Zone_Floor_Area=(self.building_config.equipment_gain_value),
                Watts_per_Person=(self.building_config.equipment_gain_value),
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
                (self.building_config.length_wall_x, 0),
                (
                    self.building_config.length_wall_x,
                    self.building_config.length_wall_y,
                ),
                (0, self.building_config.length_wall_y),
                (0, 0),
            ],
            height=self.building_config.number_of_stories
            * self.building_config.storey_height,
            num_stories=self.building_config.number_of_stories,
        )

        # set rotation
        self.idf.idfobjects["BUILDING"][0].North_Axis = self.building_config.rotation

        self.idf.intersect_match()
        self.idf = add_roof(self.idf, self.building_config)
        self.add_windows()
        self.set_boundary_conditions()
        self.add_neighbours()
        self.set_constructions()
        self.idf = add_heating_system(
            self.idf, self.building_config, self.get_heated_zones()
        )
        self.add_schedules()
        self.add_people()
        self.add_ventilation()
        self.add_infiltration()
        self.add_internal_gains()
        self.add_environmental_impact_factors()
        self.set_design_days()

        self.idf = add_pv_and_battery(self.idf, self.building_config)

        return self.idf

    def add_windows(self):
        """method which adds window strips into idf and then deletes the windows added
        to roof space"""

        # self.idf.set_wwr(
        #     wwr=0.00001,
        #     wwr_map={
        #         0: self.building_config.wtw_ratios[0],
        #         90: self.building_config.wtw_ratios[1],
        #         180: self.building_config.wtw_ratios[2],
        #         270: self.building_config.wtw_ratios[3],
        #     },
        #     construction="Window-Construction",
        # )
        self.idf.set_wwr(wwr=self.building_config.wtw_ratios[0], orientation="north")
        self.idf.set_wwr(wwr=self.building_config.wtw_ratios[1], orientation="east")
        self.idf.set_wwr(wwr=self.building_config.wtw_ratios[2], orientation="south")
        self.idf.set_wwr(wwr=self.building_config.wtw_ratios[3], orientation="west")

        # the code above adds a strip of windows to each storey, including roof space
        # this needs to be removed
        # Hannes: this seems to take out the windows on the west facade

        # if self.building_config.roof_type != "flat":
        #     self.idf.idfobjects["FENESTRATIONSURFACE:DETAILED"].pop(-1)
        #     self.idf.idfobjects["FENESTRATIONSURFACE:DETAILED"].pop(-1)

    def set_boundary_conditions(self):

        for floor_surface in self.idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
            if (
                floor_surface.Surface_Type == "floor"
                and floor_surface.Zone_Name != "ROOF SPACE"
            ):
                floor_zone_nr = int(floor_surface.Zone_Name.split()[-1])
                if floor_zone_nr in range(1, self.building_config.number_of_stories):
                    # find ceiling of zone below
                    for ceil_surface in self.idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
                        if ceil_surface.Surface_Type == "ceiling":
                            ceil_zone_nr = int(ceil_surface.Zone_Name.split()[-1])
                            if ceil_zone_nr == floor_zone_nr - 1:
                                floor_surface.Outside_Boundary_Condition = "Surface"
                                floor_surface.Outside_Boundary_Condition_Object = (
                                    ceil_surface.Name
                                )

                                floor_surface.Sun_Exposure = "NoSun"
                                floor_surface.Wind_Exposure = "NoWind"
                                ceil_surface.Outside_Boundary_Condition = "Surface"
                                ceil_surface.Outside_Boundary_Condition_Object = (
                                    floor_surface.Name
                                )

                                ceil_surface.Sun_Exposure = "NoSun"
                                ceil_surface.Wind_Exposure = "NoWind"

        if self.building_config.distance_to_neighbour[0] == 0:
            self.idf.set_wwr(wwr=0, orientation="north")
            # change boundary conditions of all north facing walls
            for wall in utilities.get_walls_in_limits(
                self.idf,
                y_lims=(
                    -1e-4 + self.building_config.length_wall_y,
                    1e-4 + self.building_config.length_wall_y,
                ),
            ):
                wall.Outside_Boundary_Condition = "Adiabatic"
                wall.Sun_Exposure = "NoSun"
                wall.Wind_Exposure = "NoWind"

        if self.building_config.distance_to_neighbour[1] == 0:
            self.idf.set_wwr(wwr=0, orientation="east")
            # change boundary conditions of all east facing walls
            for wall in utilities.get_walls_in_limits(
                self.idf,
                x_lims=(
                    -1e-4 + self.building_config.length_wall_x,
                    1e-4 + self.building_config.length_wall_x,
                ),
            ):
                wall.Outside_Boundary_Condition = "Adiabatic"
                wall.Sun_Exposure = "NoSun"
                wall.Wind_Exposure = "NoWind"

        if self.building_config.distance_to_neighbour[2] == 0:
            self.idf.set_wwr(wwr=0, orientation="south")
            # change boundary conditions of all south facing walls
            for wall in utilities.get_walls_in_limits(
                self.idf,
                y_lims=(
                    -1e-4,
                    1e-4,
                ),
            ):
                wall.Outside_Boundary_Condition = "Adiabatic"
                wall.Sun_Exposure = "NoSun"
                wall.Wind_Exposure = "NoWind"

        if self.building_config.distance_to_neighbour[3] == 0:
            self.idf.set_wwr(wwr=0, orientation="west")
            # change boundary conditions of all west facing walls
            for wall in utilities.get_walls_in_limits(
                self.idf,
                x_lims=(-1e-4, 1e-4),
            ):
                wall.Outside_Boundary_Condition = "Adiabatic"
                wall.Sun_Exposure = "NoSun"
                wall.Wind_Exposure = "NoWind"

    def add_neighbours(self):

        neighbour_layers = 2
        d = self.building_config.distance_to_neighbour
        lx = self.building_config.length_wall_x
        ly = self.building_config.length_wall_y
        h = (
            self.building_config.storey_height * self.building_config.number_of_stories
            + self.building_config.roof_height
        )

        for x_idx in range(-neighbour_layers, 1 + neighbour_layers):
            for y_idx in range(-neighbour_layers, 1 + neighbour_layers):
                if abs(x_idx) != 1 and y_idx == 0:
                    continue
                if abs(y_idx) != 1 and x_idx == 0:
                    continue

                faces = []
                if x_idx < 0 and y_idx < 0:
                    faces = ["N", "E"]
                elif x_idx == 0 and y_idx < 0:
                    faces = ["N"]
                elif y_idx < 0 < x_idx:
                    faces = ["N", "W"]
                elif x_idx < 0 and y_idx == 0:
                    faces = ["E"]
                elif x_idx > 0 and y_idx == 0:
                    faces = ["W"]
                elif x_idx < 0 < y_idx:
                    faces = ["S", "E"]
                elif x_idx == 0 and y_idx > 0:
                    faces = ["S"]
                elif x_idx > 0 and y_idx > 0:
                    faces = ["S", "W"]

                xi_min, yi_min = utilities.get_shading_surface_start_coordinates(
                    x_idx, y_idx, neighbour_layers, lx, ly, d
                )

                # exclude attached neighbours
                if xi_min in [0, lx] and yi_min in [0, ly]:
                    continue

                if "N" in faces:
                    self.idf.newidfobject(
                        "SHADING:BUILDING",
                        Name=f"NEIGHBOUR-L{neighbour_layers}-X{x_idx}-Y{y_idx}-NORTH",
                        Azimuth_Angle=180,
                        Tilt_Angle=90,
                        Starting_X_Coordinate=xi_min,
                        Starting_Y_Coordinate=yi_min + ly,
                        Starting_Z_Coordinate=0,
                        Length=lx,
                        Height=h,
                    )

                if "S" in faces:
                    self.idf.newidfobject(
                        "SHADING:BUILDING",
                        Name=f"NEIGHBOUR-L{neighbour_layers}-X{x_idx}-Y{y_idx}-SOUTH",
                        Azimuth_Angle=180,
                        Tilt_Angle=90,
                        Starting_X_Coordinate=xi_min,
                        Starting_Y_Coordinate=yi_min,
                        Starting_Z_Coordinate=0,
                        Length=lx,
                        Height=h,
                    )

                if "E" in faces:
                    self.idf.newidfobject(
                        "SHADING:BUILDING",
                        Name=f"NEIGHBOUR-L{neighbour_layers}-X{x_idx}-Y{y_idx}-EAST",
                        Azimuth_Angle=90,
                        Tilt_Angle=90,
                        Starting_X_Coordinate=xi_min + lx,
                        Starting_Y_Coordinate=yi_min,
                        Starting_Z_Coordinate=0,
                        Length=ly,
                        Height=h,
                    )

                if "W" in faces:
                    self.idf.newidfobject(
                        "SHADING:BUILDING",
                        Name=f"NEIGHBOUR-L{neighbour_layers}-X{x_idx}-Y{y_idx}-WEST",
                        Azimuth_Angle=90,
                        Tilt_Angle=90,
                        Starting_X_Coordinate=xi_min,
                        Starting_Y_Coordinate=yi_min,
                        Starting_Z_Coordinate=0,
                        Length=ly,
                        Height=h,
                    )

    def get_floor_area(self):
        return (
            self.building_config.length_wall_x
            * self.building_config.length_wall_y
            * self.building_config.number_of_stories
        )

    def get_idf(self):
        return self.idf
