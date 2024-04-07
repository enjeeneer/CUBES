"""Defines the Building class """

from typing import Dict
from cubes.construct import material as mat
from cubes.construct import utilities
from cubes.construct.buildingconfig import BuildingConfig
from cubes.construct.hvac_systems import add_heating_system
from cubes.construct.pv_and_battery import add_pv_and_battery
from cubes.construct.geometry import (
    add_surfaces_and_zones,
    add_strip_window_on_wall,
    add_gable_window_on_triangular_wall,
)
from cubes.construct.utilities import get_schedule, get_grid_carbon_intensity_file_path
import cubes.construct.buildingconfig_options as bco
from cubes.constants import EPLUS_PATH, BASE_DIR
from cubes.construct.ventilation import add_ventilation
from cubes.constants import NATURAL_GAS_EMISSIONS_FACTOR
from geomeppy import IDF
import pandas as pd


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
        self.adiabatic_wall_construction = mat.Construction(
            "Adiabatic Wall",
            [
                *[materials[x] for x in building_config.wall_layer_materials],
                materials["Adiabatic_insulation"],
            ],
            [*building_config.wall_layer_thickness, 1.0],
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
            "InternalWall",
            [materials[x] for x in building_config.partition_layer_materials],
            building_config.partition_layer_thickness,
        )

        self.all_constructions = [
            self.wall_construction,
            self.adiabatic_wall_construction,
            self.roof_construction,
            self.ground_floor_construction,
            self.upper_floor_construction,
            self.ceiling_construction,
            self.partition_construction,
        ]
        if self.building_config.furniture_material:
            self.furniture_construction = mat.Construction(
                "Furniture",
                [materials[building_config.furniture_material]],
                [building_config.furniture_thickness],
            )
            self.all_constructions.append(self.furniture_construction)

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

        else:
            self.last_floor_construction = self.upper_floor_construction
            self.last_ceiling_construction = self.ceiling_construction

        if building_config.subfloor_layer_materials:
            self.subfloor_construction = mat.Construction(
                "SubFloor",
                [materials[x] for x in building_config.subfloor_layer_materials],
                building_config.subfloor_layer_thickness,
            )
            self.all_constructions.append(self.subfloor_construction)

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

        # get schdeules which are described in occupancy schedules

        if self.building_config.occupant_schedule is not None:
            # get path to where schedules are specified
            schedule_directory = BASE_DIR / "cubes/data/schedules/"
            schedule_file_name = self.building_config.occupant_schedule_file_name
            schedule_path = schedule_directory / schedule_file_name

            for zones_in_storey in self.building_config.zone_names:
                for i, zone in enumerate(zones_in_storey):

                    dataframe = pd.read_csv(schedule_path, index_col=0)
                    dataframe = dataframe.reset_index(drop=True)

                    schedule_to_write = dataframe.iloc[:, i].to_string(index=False)

                    occupancy_schedule_file = (
                        building_config.files_dir
                        + "/occupancy_schedule_"
                        + zone
                        + ".sch"
                    )

                    utilities.write_string_to_file(
                        schedule_to_write,
                        occupancy_schedule_file,
                    )

        # TODO delete below
        # below is code which I (JACK) have commented out as it is not generalisable
        # for different number of zones

        # if self.building_config.occupant_schedule_living is not None:
        #    self.occupancy_schedule_living_file = (
        #        building_config.files_dir + "/occupancy_living.sch"
        #    )

        #    utilities.write_string_to_file(
        #        self.building_config.occupant_schedule_living,
        #        self.occupancy_schedule_living_file,
        #    )

        # if self.building_config.occupant_schedule_bedroom is not None:
        #    self.occupancy_schedule_bedroom_file = (
        #        building_config.files_dir + "/occupancy_bedroom.sch"
        #    )

        #    utilities.write_string_to_file(
        #        self.building_config.occupant_schedule_bedroom,
        #        self.occupancy_schedule_bedroom_file,
        #    )

        IDF.setiddname(EPLUS_PATH + "Energy+.idd")
        self.idf = IDF(EPLUS_PATH + "ExampleFiles/Minimal.idf")

        self.idf.idfobjects["GLOBALGEOMETRYRULES"][0].Coordinate_System = "Relative"
        self.idf.idfobjects["BUILDING"][0].Solar_Distribution = "FullExterior"
        self.idf.idfobjects["TIMESTEP"][0].Number_of_Timesteps_per_Hour = 6
        self.idf.idfobjects["BUILDING"][0].Name = self.building_config.name
        self.idf.idfobjects["RUNPERIOD"][0].Begin_Year = self.building_config.year
        self.idf.idfobjects["RUNPERIOD"][0].End_Year = self.building_config.year
        self.idf.newidfobject(
            "ZoneAirHeatBalanceAlgorithm".upper(), Algorithm="AnalyticalSolution"
        )

        self.idf.newidfobject("SURFACECONVECTIONALGORITHM:INSIDE", Algorithm="Simple")

    def set_constructions(self):
        """adds materials and constructions to IDF
        then assigns each of the constructions to surfaces
        """

        for c in self.all_constructions:
            if c.materials:
                self.idf = c.add_to_idf(self.idf)

        for surface in self.idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
            if surface.Surface_Type.lower() == "wall":
                if surface.Outside_Boundary_Condition.lower() == "zone":
                    surface.Construction_Name = self.partition_construction.get_name()
                else:
                    surface.Construction_Name = self.wall_construction.get_name()
            elif surface.Surface_Type.lower() == "roof":
                surface.Construction_Name = self.roof_construction.get_name()
            elif surface.Surface_Type.lower() == "floor":
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
            elif surface.Surface_Type.lower() == "ceiling":
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

    def zone_not_conditioned(self, zone_name):
        return (
            zone_name == "Loft" and not self.building_config.loft_is_heated
        ) or zone_name == "Subfloor"

    def get_conditioned_zones(self):
        zones = []
        for zone in self.idf.idfobjects["ZONE"]:
            if self.zone_not_conditioned(zone.Name):
                continue
            zones.append(zone)
        return zones

    def add_schedules(self):
        """Adds schedules into e+."""
        # add schedule types
        self.idf.newidfobject(
            "SCHEDULETYPELIMITS",
            Name="Fraction",
            Lower_Limit_Value=0,
            Upper_Limit_Value=1,
            Numeric_Type="Continuous",
            Unit_Type="Dimensionless",
        )
        # add schedule types
        self.idf.newidfobject("SCHEDULETYPELIMITS", Name="Any Number")

        # add zone schedules in a generalised manner

        # TODO add in some checks/tests to ensure there are sch for each zone name,
        # otherwise this will break!!
        # TODO Check with Hannes that my approach is correct for scheduling

        # TODO DELETE the commented out lines below
        # Define the directory and target file name
        # schedule_directory = "/workspaces/CUBES/cubes/data/schedules/"

        ## Construct the full path to the target file
        # occupancy_schedule_file = self.building_config.occupant_schedule_file_name
        # occupancy_schedule_file_path = os.path.join(
        #     schedule_directory, occupancy_schedule_file)

        ## Check if the target file exists at the specified path
        # if os.path.exists(occupancy_schedule_file_path):
        #    occupancy_schedules = pd.read_csv(
        #        occupancy_schedule_file_path, index_col=0)
        #    occupancy_schedules = occupancy_schedules.reset_index(drop=True)
        # else:
        #    # Raise an error or handle the case where the file does not exist
        #    raise ValueError(
        #        f"No occupant schedule file named {occupancy_schedule_file_path}
        #        in building config")

        for zones in self.building_config.zone_names:
            for zone in zones:
                if zone:
                    occupancy_schedule_file_path = (
                        self.building_config.files_dir
                        + "/occupancy_schedule_"
                        + zone
                        + ".sch"
                    )

                    self.idf.newidfobject(
                        "SCHEDULE:FILE",
                        Name="Occupancy-Schedule-" + zone,
                        Schedule_Type_Limits_Name="Fraction",
                        File_Name=occupancy_schedule_file_path,
                        Column_Number=1,
                        Rows_to_Skip_at_Top=0,
                        Number_of_Hours_of_Data=8760,
                        Minutes_per_Item=10,
                    )

                    if "bedroom" in zone.lower():
                        self.idf.newidfobject(
                            "SCHEDULE:COMPACT",
                            Name="Activity-Schedule-" + zone,
                            Field_1=(
                                "Through: 12/31,\n    "
                                "For: AllDays,\n    Until: 7:00, 80.,\n   "
                                "Until: 22:00, 120.,\n    Until: 24:00, 80.,\n"
                            ),
                        )
                    else:
                        self.idf.newidfobject(
                            "SCHEDULE:COMPACT",
                            Name="Activity-Schedule-" + zone,
                            Field_1=(
                                "Through: 12/31,\n    "
                                "For: AllDays,\n    Until: 24:00, 120.\n"
                            ),
                        )
                else:
                    self.idf.newidfobject(
                        "SCHEDULE:COMPACT",
                        Name="Occupancy-Schedule-" + zone,
                        Field_1=(
                            "Through: 12/31,\n    "
                            "For: Weekdays,\n    Until: 9:00, 1.0,\n"
                            "    Until:17:00, 0.5,\n    Until:24:00, 1.,\n "
                            "   For:AllOtherDays,\n    Until:24:00,1."
                        ),
                    )

        # I (JACK) have commented out below as the adding of schedules is not
        # generalised

        # occupants living room
        # if self.building_config.occupant_schedule_living:
        #    self.idf.newidfobject(
        #        "SCHEDULE:FILE",
        #        Name="Occupancy-Schedule-Living",
        #        Schedule_Type_Limits_Name="Fraction",
        #        File_Name=self.occupancy_schedule_living_file,
        #        Column_Number=1,
        #        Rows_to_Skip_at_Top=0,
        #        Number_of_Hours_of_Data=8760,
        #        Minutes_per_Item=10,
        #    )
        # else:
        #    self.idf.newidfobject(
        #        "SCHEDULE:COMPACT",
        #        Name="Occupancy-Schedule-Living",
        #        Field_1=(
        #            "Through: 12/31,\n    "
        #            "For: Weekdays,\n    Until: 9:00, 1.0,\n"
        #            "    Until:17:00, 0.5,\n    Until:24:00, 1.,\n "
        #            "   For:AllOtherDays,\n    Until:24:00,1."
        #        ),
        #    )

        ## occupants bedroom room
        # if self.building_config.occupant_schedule_bedroom:
        #    self.idf.newidfobject(
        #        "SCHEDULE:FILE",
        #        Name="Occupancy-Schedule-Bedroom",
        #        Schedule_Type_Limits_Name="Fraction",
        #        File_Name=self.occupancy_schedule_bedroom_file,
        #        Column_Number=1,
        #        Rows_to_Skip_at_Top=0,
        #        Number_of_Hours_of_Data=8760,
        #        Minutes_per_Item=10,
        #    )

        # else:
        #    self.idf.newidfobject(
        #        "SCHEDULE:COMPACT",
        #        Name="Occupancy-Schedule-Bedroom",
        #        Field_1=(
        #            "Through: 12/31,\n    "
        #            "For: Weekdays,\n    Until: 9:00, 1.0,\n"
        #            "    Until:17:00, 0.5,\n    Until:24:00, 1.,\n "
        #            "   For:AllOtherDays,\n    Until:24:00,1."
        #        ),
        #    )

        # defaults
        # TODO unsure what I (JACK) need to do with the old zone activity schedules
        self.idf.newidfobject(
            "SCHEDULE:COMPACT",
            Name="Always-Schedule",
            Field_1="Through: 12/31,\n    For: AllDays,\n    Until: 24:00, 1.0\n",
        )
        self.idf.newidfobject(
            "SCHEDULE:COMPACT",
            Name="Activity-Schedule-Living",
            Field_1="Through: 12/31,\n    For: AllDays,\n    Until: 24:00, 120.\n",
        )
        self.idf.newidfobject(
            "SCHEDULE:COMPACT",
            Name="Activity-Schedule-Bedroom",
            Field_1=(
                "Through: 12/31,\n    For: AllDays,\n    Until: 7:00, 80.,\n    "
                "Until: 22:00, 120.,\n    Until: 24:00, 80.,\n"
            ),
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

        if self.building_config.zoning == bco.Zoning.RESIDENTIAL_DWELLING.value:
            self.idf.newidfobject(
                "PEOPLE",
                Name="Living-People",
                Zone_or_ZoneList_Name="Living",
                Number_of_People_Calculation_Method=(
                    self.building_config.occupant_number_calculation_method
                ),
                Number_of_People_Schedule_Name="Occupancy-Schedule-Living",
                Number_of_People=self.building_config.occupant_value,
                People_per_Zone_Floor_Area=self.building_config.occupant_value,
                Zone_Floor_Area_per_Person=self.building_config.occupant_value,
                Activity_Level_Schedule_Name="Activity-Schedule-Living",
            )

            self.idf.newidfobject(
                "PEOPLE",
                Name="Bedroom-People",
                Zone_or_ZoneList_Name="Bedroom",
                Number_of_People_Calculation_Method=(
                    self.building_config.occupant_number_calculation_method
                ),
                Number_of_People_Schedule_Name="Occupancy-Schedule-Bedroom",
                Number_of_People=self.building_config.occupant_value,
                People_per_Zone_Floor_Area=self.building_config.occupant_value,
                Zone_Floor_Area_per_Person=self.building_config.occupant_value,
                Activity_Level_Schedule_Name="Activity-Schedule-Bedroom",
            )

        elif self.building_config.zoning == bco.Zoning.SINGLE_ZONE.value:
            self.idf.newidfobject(
                "PEOPLE",
                Name="Living-People",
                Zone_or_ZoneList_Name="Living",
                Number_of_People_Calculation_Method=(
                    self.building_config.occupant_number_calculation_method
                ),
                Number_of_People_Schedule_Name="Occupancy-Schedule-Living",
                Number_of_People=self.building_config.occupant_value,
                People_per_Zone_Floor_Area=self.building_config.occupant_value,
                Zone_Floor_Area_per_Person=self.building_config.occupant_value,
                Activity_Level_Schedule_Name="Activity-Schedule-Living",
            )

        # added by JACK for zoning generalisation

        elif self.building_config.zoning == bco.Zoning.CUSTOM.value:
            for zones_in_storey in self.building_config.zone_names:
                for zone in zones_in_storey:

                    self.idf.newidfobject(
                        "PEOPLE",
                        Name=zone + "-People",
                        Zone_or_ZoneList_Name=zone,
                        Number_of_People_Calculation_Method=(
                            self.building_config.occupant_number_calculation_method
                        ),
                        Number_of_People_Schedule_Name="Occupancy-Schedule-" + zone,
                        Number_of_People=self.building_config.occupant_value,
                        People_per_Zone_Floor_Area=self.building_config.occupant_value,
                        Zone_Floor_Area_per_Person=self.building_config.occupant_value,
                        Activity_Level_Schedule_Name="Activity-Schedule-" + zone,
                    )

        else:

            for zone in self.get_conditioned_zones():

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

    def add_infiltration(self):
        """Adds infiltration into e+ for every zone in idf"""
        for zone in self.idf.idfobjects["ZONE"]:

            if zone.Name == "Subfloor":
                self.idf.newidfobject(
                    "ZONEINFILTRATION:DESIGNFLOWRATE",
                    Name=zone.Name + "-Infiltration",
                    Zone_or_ZoneList_Name=zone.Name,
                    Design_Flow_Rate_Calculation_Method="airchanges/hour",
                    Air_Changes_per_Hour=(
                        self.building_config.subfloor_infiltration_ach
                    ),
                    Constant_Term_Coefficient=0.606,
                    Temperature_Term_Coefficient=0.03636,
                    Velocity_Term_Coeﬀicient=0.1177,
                    Velocity_Squared_Term_Coefficient=0.0,
                    Schedule_Name="Always-Schedule",
                )
            else:
                self.idf.newidfobject(
                    "ZONEINFILTRATION:DESIGNFLOWRATE",
                    Name=zone.Name + "-Infiltration",
                    Zone_or_ZoneList_Name=zone.Name,
                    Design_Flow_Rate_Calculation_Method=(
                        self.building_config.infiltration_calculation_method
                    ),
                    Design_Flow_Rate=(self.building_config.infiltration_rate),
                    Flow_per_Zone_Floor_Area=(self.building_config.infiltration_rate),
                    Flow_per_Exterior_Surface_Area=(
                        self.building_config.infiltration_rate
                    ),
                    Air_Changes_per_Hour=(self.building_config.infiltration_rate),
                    Constant_Term_Coefficient=0.606,
                    Temperature_Term_Coefficient=0.03636,
                    Velocity_Term_Coeﬀicient=0.1177,
                    Velocity_Squared_Term_Coefficient=0.0,
                    Schedule_Name="Always-Schedule",
                )

    def add_internal_mass(self, zone_areas):
        """adds internal thermal mass of partitions and furniture"""
        for zone in self.get_conditioned_zones():
            if zone_areas:
                za = zone_areas[zone.Name]
            else:  # one zone per floor
                za = (
                    self.building_config.length_wall_x
                    * self.building_config.length_wall_y
                )

            self.idf.newidfobject(
                "INTERNALMASS",
                Name=("IntMass-Partitions-" + zone.Name),
                Construction_Name="InternalWall",
                Zone_or_ZoneList_Name=zone.Name,
                Surface_Area=za,
            )
            if self.furniture_construction:
                furn_mat = self.furniture_construction.materials[0]
                furn_t = self.furniture_construction.thicknesses[0]
                tm_furniture = (
                    self.building_config.furniture_thermal_mass_per_floor_area
                    / (furn_mat.rho * furn_mat.cp / 1000 * furn_t)
                )
                self.idf.newidfobject(
                    "INTERNALMASS",
                    Name=("IntMass-Furniture-" + zone.Name),
                    Construction_Name="Furniture",
                    Zone_or_ZoneList_Name=zone.Name,
                    Surface_Area=za * tm_furniture,
                )

        # internal mass constructions
        for im in self.idf.idfobjects["INTERNALMASS"]:
            if im.Construction_Name.lower() == "ceiling":
                im.Construction_Name = self.ceiling_construction.get_name()
            elif im.Construction_Name.lower() == "floor":
                im.Construction_Name = self.upper_floor_construction.get_name()
            elif im.Construction_Name.lower() == "internalwall":
                im.Construction_Name = self.partition_construction.get_name()
            elif im.Construction_Name.lower() == "furniture":
                im.Construction_Name = self.furniture_construction.get_name()

    def add_internal_gains(self):
        """Adds internal gains into e+ for every zone in idf"""
        for zone in self.get_conditioned_zones():
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
            CO2_Emission_Factor=NATURAL_GAS_EMISSIONS_FACTOR,
            Source_Energy_Factor=1,
        )
        self.idf.newidfobject(
            "FUELFACTORS",
            Existing_Fuel_Resource_Name="Electricity",
            CO2_Emission_Factor=1 / 3.6,  # conversion MJ to kWh
            CO2_Emission_Factor_Schedule_Name="Grid Carbon Intensity Schedule",
        )
        self.idf.newidfobject(
            "SCHEDULE:FILE",
            Name="Grid Carbon Intensity Schedule",
            Schedule_Type_Limits_Name="Any Number",
            File_Name=get_grid_carbon_intensity_file_path(
                self.building_config.grid_carbon_intensity_file_name
            ),
            Column_Number=2,
            Rows_to_Skip_at_Top=1,
            Number_of_Hours_of_Data=8760,
            Minutes_per_Item=10,
            Interpolate_to_Timestep="yes",
        )

        self.idf.newidfobject(
            "ENVIRONMENTALIMPACTFACTORS",
            Total_Carbon_Equivalent_Emission_Factor_From_N2O=298,
            Total_Carbon_Equivalent_Emission_Factor_From_CH4=25,
            Total_Carbon_Equivalent_Emission_Factor_From_CO2=1,
        )

    def set_design_days(self):

        # remove design days:
        self.idf.idfobjects["SIZINGPERIOD:DESIGNDAY"].clear()
        # # add design period:
        # self.idf.newidfobject(
        #     "SIZINGPERIOD:WEATHERFILEDAYS",
        #     Name="Winter Design Day",
        #     Begin_Month=1,
        #     Begin_Day_of_Month=1,
        #     End_Month=1,
        #     End_Day_of_Month=14,
        # )

        # self.idf.newidfobject(
        #     "SIZINGPERIOD:WEATHERFILEDAYS",
        #     Name="Summer Design Day",
        #     Begin_Month=7,
        #     Begin_Day_of_Month=1,
        #     End_Month=7,
        #     End_Day_of_Month=14,
        # )
        # add Cambridge design day
        self.idf.newidfobject(
            "SIZINGPERIOD:DESIGNDAY",
            Name="Cambridge.AP Ann Htg 99.6p Condns DB",
            Month=2,
            Day_of_Month=21,
            Day_Type="WinterDesignDay",
            Maximum_DryBulb_Temperature=-8.8,
            Daily_DryBulb_Temperature_Range=0.0,
            DryBulb_Temperature_Range_Modifier_Type="DefaultMultipliers",
            Humidity_Condition_Type="Wetbulb",
            Wetbulb_or_DewPoint_at_Maximum_DryBulb=-10.8,
            Barometric_Pressure=101153.0,
            Wind_Speed=6.71,
            Wind_Direction=0,
        )

    def build(self):
        """method which can construct or 'build' our archetypal building

        Returns:
            idf: idf is the input data file which can be used by energyplus
        """

        self.idf, zone_areas = add_surfaces_and_zones(self.idf, self.building_config)

        # set rotation
        self.idf.idfobjects["BUILDING"][0].North_Axis = self.building_config.rotation
        if self.building_config.zoning == bco.Zoning.CUSTOM.value:
            self.idf.intersect_match()
        else:
            self.set_boundary_conditions()
        self.add_windows()
        self.add_neighbours()
        self.set_constructions()
        self.idf = add_heating_system(
            self.idf, self.building_config, self.get_conditioned_zones()
        )
        self.add_schedules()
        self.add_people()
        self.idf = add_ventilation(
            self.idf, self.building_config, self.get_conditioned_zones()
        )
        self.add_infiltration()
        self.add_internal_gains()
        self.add_internal_mass(zone_areas)

        self.add_environmental_impact_factors()
        self.set_design_days()

        if self.building_config.pv_present:
            self.idf = add_pv_and_battery(self.idf, self.building_config)

        return self.idf

    def add_windows(self):
        """method which adds window strips into idf"""

        for i_s in range(self.building_config.number_of_stories):
            if self.building_config.wtw_ratios[0] > 0:
                for wall in utilities.get_walls_in_limits(
                    self.idf,
                    y_lims=(
                        -1e-4 + self.building_config.length_wall_y,
                        1e-4 + self.building_config.length_wall_y,
                    ),
                    z_lims=(
                        -1e-4 + i_s * self.building_config.storey_height,
                        1e-4 + (i_s + 1) * self.building_config.storey_height,
                    ),
                ):
                    self.idf = add_strip_window_on_wall(
                        self.idf, self.building_config.wtw_ratios[0], wall
                    )
            if self.building_config.wtw_ratios[1] > 0:
                for wall in utilities.get_walls_in_limits(
                    self.idf,
                    x_lims=(
                        -1e-4,
                        1e-4,
                    ),
                    z_lims=(
                        -1e-4 + i_s * self.building_config.storey_height,
                        1e-4 + (i_s + 1) * self.building_config.storey_height,
                    ),
                ):
                    self.idf = add_strip_window_on_wall(
                        self.idf, self.building_config.wtw_ratios[1], wall
                    )

            if self.building_config.wtw_ratios[2] > 0:
                for wall in utilities.get_walls_in_limits(
                    self.idf,
                    y_lims=(
                        -1e-4,
                        1e-4,
                    ),
                    z_lims=(
                        -1e-4 + i_s * self.building_config.storey_height,
                        1e-4 + (i_s + 1) * self.building_config.storey_height,
                    ),
                ):
                    self.idf = add_strip_window_on_wall(
                        self.idf, self.building_config.wtw_ratios[2], wall
                    )

            if self.building_config.wtw_ratios[3] > 0:
                for wall in utilities.get_walls_in_limits(
                    self.idf,
                    x_lims=(
                        -1e-4 + self.building_config.length_wall_x,
                        1e-4 + self.building_config.length_wall_x,
                    ),
                    z_lims=(
                        -1e-4 + i_s * self.building_config.storey_height,
                        1e-4 + (i_s + 1) * self.building_config.storey_height,
                    ),
                ):
                    self.idf = add_strip_window_on_wall(
                        self.idf, self.building_config.wtw_ratios[3], wall
                    )

        if self.building_config.wtw_ratios_loft[0] > 0:
            for wall in utilities.get_walls_in_limits(
                self.idf,
                y_lims=(
                    -1e-4 + self.building_config.length_wall_y / 2,
                    1e-4 + self.building_config.length_wall_y,
                ),
                z_lims=(
                    (
                        -1e-4
                        + self.building_config.number_of_stories
                        * self.building_config.storey_height
                    ),
                    (
                        1e-4
                        + self.building_config.number_of_stories
                        * self.building_config.storey_height
                        + self.building_config.roof_height
                    ),
                ),
            ):
                if self.building_config.roof_ridge_along_x:
                    self.idf = add_strip_window_on_wall(
                        self.idf, self.building_config.wtw_ratios_loft[0], wall
                    )
                else:
                    self.idf = add_gable_window_on_triangular_wall(
                        self.idf, self.building_config.wtw_ratios_loft[0], wall
                    )

        if self.building_config.wtw_ratios_loft[1] > 0:
            for wall in utilities.get_walls_in_limits(
                self.idf,
                x_lims=(
                    -1e-4,
                    1e-4 + self.building_config.length_wall_x / 2,
                ),
                z_lims=(
                    (
                        -1e-4
                        + self.building_config.number_of_stories
                        * self.building_config.storey_height
                    ),
                    (
                        1e-4
                        + self.building_config.number_of_stories
                        * self.building_config.storey_height
                        + self.building_config.roof_height
                    ),
                ),
            ):
                if not self.building_config.roof_ridge_along_x:
                    self.idf = add_strip_window_on_wall(
                        self.idf, self.building_config.wtw_ratios_loft[1], wall
                    )
                else:
                    self.idf = add_gable_window_on_triangular_wall(
                        self.idf, self.building_config.wtw_ratios_loft[1], wall
                    )

        if self.building_config.wtw_ratios_loft[2] > 0:
            for wall in utilities.get_walls_in_limits(
                self.idf,
                y_lims=(
                    -1e-4,
                    1e-4 + self.building_config.length_wall_y / 2,
                ),
                z_lims=(
                    (
                        -1e-4
                        + self.building_config.number_of_stories
                        * self.building_config.storey_height
                    ),
                    (
                        1e-4
                        + self.building_config.number_of_stories
                        * self.building_config.storey_height
                        + self.building_config.roof_height
                    ),
                ),
            ):
                if self.building_config.roof_ridge_along_x:
                    self.idf = add_strip_window_on_wall(
                        self.idf, self.building_config.wtw_ratios_loft[2], wall
                    )
                else:
                    self.idf = add_gable_window_on_triangular_wall(
                        self.idf, self.building_config.wtw_ratios_loft[2], wall
                    )

        if self.building_config.wtw_ratios_loft[3] > 0:
            for wall in utilities.get_walls_in_limits(
                self.idf,
                x_lims=(
                    -1e-4 + self.building_config.length_wall_x / 2,
                    1e-4 + self.building_config.length_wall_x,
                ),
                z_lims=(
                    (
                        -1e-4
                        + self.building_config.number_of_stories
                        * self.building_config.storey_height
                    ),
                    (
                        1e-4
                        + self.building_config.number_of_stories
                        * self.building_config.storey_height
                        + self.building_config.roof_height
                    ),
                ),
            ):
                if not self.building_config.roof_ridge_along_x:
                    self.idf = add_strip_window_on_wall(
                        self.idf, self.building_config.wtw_ratios_loft[3], wall
                    )
                else:
                    self.idf = add_gable_window_on_triangular_wall(
                        self.idf, self.building_config.wtw_ratios_loft[3], wall
                    )

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
            # change boundary conditions of all north facing walls
            for wall in utilities.get_walls_in_limits(
                self.idf,
                y_lims=(
                    -1e-4 + self.building_config.length_wall_y,
                    1e-4 + self.building_config.length_wall_y,
                ),
            ):
                wall.Construction_Name = self.adiabatic_wall_construction.get_name()
                # wall.Outside_Boundary_Condition = "Adiabatic"
                wall.Sun_Exposure = "NoSun"
                wall.Wind_Exposure = "NoWind"

        if self.building_config.distance_to_neighbour[1] == 0:
            # change boundary conditions of all east facing walls
            for wall in utilities.get_walls_in_limits(
                self.idf,
                x_lims=(
                    -1e-4 + self.building_config.length_wall_x,
                    1e-4 + self.building_config.length_wall_x,
                ),
            ):
                wall.Construction_Name = self.adiabatic_wall_construction.get_name()
                # wall.Outside_Boundary_Condition = "Adiabatic"
                wall.Sun_Exposure = "NoSun"
                wall.Wind_Exposure = "NoWind"

        if self.building_config.distance_to_neighbour[2] == 0:
            # change boundary conditions of all south facing walls
            for wall in utilities.get_walls_in_limits(
                self.idf,
                y_lims=(
                    -1e-4,
                    1e-4,
                ),
            ):
                wall.Construction_Name = self.adiabatic_wall_construction.get_name()
                # wall.Outside_Boundary_Condition = "Adiabatic"
                wall.Sun_Exposure = "NoSun"
                wall.Wind_Exposure = "NoWind"

        if self.building_config.distance_to_neighbour[3] == 0:
            # change boundary conditions of all west facing walls
            for wall in utilities.get_walls_in_limits(
                self.idf,
                x_lims=(-1e-4, 1e-4),
            ):
                wall.Construction_Name = self.adiabatic_wall_construction.get_name()
                # wall.Outside_Boundary_Condition = "Adiabatic"
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
                elif xi_min + lx in [0, lx] and yi_min + ly in [0, ly]:
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
