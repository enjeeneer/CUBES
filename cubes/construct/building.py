"""Defines the Building class """

from typing import Dict
from pathlib import Path
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
from cubes.construct.utilities import (
    get_schedule,
    get_grid_carbon_intensity_file_path,
    get_gas_pricing_file_path,
    get_electricity_pricing_file_path,
    get_electricity_surplus_file_path,
    ModifiedIDF as IDF,
)
import cubes.construct.buildingconfig_options as bco
from cubes.constants import EPLUS_PATH, BASE_DIR
from cubes.construct.ventilation import add_ventilation
from cubes.constants import NATURAL_GAS_EMISSIONS_FACTOR
from cubes.construct.airflow_network import add_airflow_network

# from geomeppy import IDF
import pandas as pd
import numpy as np


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
        self.subfloor_roof_construction = mat.Construction(
            "SubFloorRoof",
            [materials[x] for x in building_config.ground_floor_layer_materials[::-1]],
            building_config.ground_floor_layer_thickness[::-1],
        )
        self.partition_construction = mat.Construction(
            "InternalWall",
            [materials[x] for x in building_config.partition_layer_materials],
            building_config.partition_layer_thickness,
        )

        # --- Doors and Holes ---
        self.hole_construction = mat.Construction(
            "HoleConstruction",
            [materials[x] for x in building_config.hole_layer_material],
            []  # IRTMaterial has no thickness
        )

        self.external_door_construction = mat.Construction(
            "ExternalDoor",
            [materials[x] for x in building_config.external_door_layer_materials],
            building_config.external_door_layer_thickness,
        )

        self.partition_door_construction = mat.Construction(
            "PartitionDoor",
            [materials[x] for x in building_config.partition_door_layer_materials],
            building_config.partition_door_layer_thickness,
        )

        self.all_constructions = [
            self.wall_construction,
            self.adiabatic_wall_construction,
            self.roof_construction,
            self.ground_floor_construction,
            self.upper_floor_construction,
            self.ceiling_construction,
            self.partition_construction,
            self.subfloor_roof_construction,
            self.hole_construction,
            self.external_door_construction,
            self.partition_door_construction,
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

        if self.building_config.heating_pattern_schedule_file_name:
            schedule_directory = BASE_DIR / "cubes/data/heating_pattern/"
            schedule_file_name = self.building_config.heating_pattern_schedule_file_name
            schedule_path = schedule_directory / schedule_file_name

            for zones_in_storey in self.building_config.zone_names:
                for zone in zones_in_storey:
                    zone = zone.lower()

                    dataframe = pd.read_csv(schedule_path)

                    # Skip first row entry so we act on current timestep
                    dataframe = dataframe.iloc[1:]

                    # Reset index to remove 'UTC_Time' from the output
                    dataframe.reset_index(drop=True, inplace=True)

                    dataframe.columns = dataframe.columns.str.lower()

                    schedule_to_write = dataframe.loc[:, zone].to_string(index=False)

                    # Prepend the column name to the string
                    schedule_to_write = f"{zone}\n{schedule_to_write}"

                    heating_schedule_filename = (
                        building_config.files_dir + "/heating_pattern_" + zone + ".sch"
                    )

                    utilities.write_string_to_file(
                        schedule_to_write,
                        heating_schedule_filename,
                    )

        if self.building_config.temperature_schedulue_file_name:
            # get path to where schedules are specified
            schedule_directory = BASE_DIR / "cubes/data/schedules/"
            schedule_file_name = self.building_config.temperature_schedulue_file_name
            schedule_path = schedule_directory / schedule_file_name

            for zones_in_storey in self.building_config.zone_names:
                for zone in zones_in_storey:

                    dataframe = pd.read_csv(schedule_path, index_col=0)

                    schedule_to_write = dataframe.loc[:, zone].to_string(index=False)

                    temperature_schedule_file = (
                        building_config.files_dir
                        + "/temperature_schedule_"
                        + zone
                        + ".sch"
                    )

                    utilities.write_string_to_file(
                        schedule_to_write,
                        temperature_schedule_file,
                    )

        if building_config.occupant_schedule_file_name:
            schedule_directory = BASE_DIR / "cubes/data/occupants"
            schedule_file_name = building_config.occupant_schedule_file_name
            schedule_path = schedule_directory / schedule_file_name

            df = pd.read_csv(schedule_path)

            # Ensure uniform formatting
            df.columns = df.columns.str.lower()

            # Use timestamp if present
            if "utc_time" in df.columns:
                df["utc_time"] = pd.to_datetime(df["utc_time"])
                df.set_index("utc_time", inplace=True)
            else:
                raise ValueError(
                    "Occupancy schedule must contain 'UTC_Time' for resampling."
                )

            timestep = (
                building_config.timesteps_per_hour
            )  # e.g., 6 for 10-min, 60 for 1-min

            # Determine resample rule based on timestep
            if timestep == 60:
                downsampled_df = df.copy()
            elif timestep == 6:
                # Resample to 10-minute intervals, treating any occupancy as '1'
                downsampled_df = df.resample("10T").mean()
            else:
                raise ValueError(f"Unsupported timestep: {timestep}")

            # Now write out a schedule file for each zone
            for zones_in_storey in building_config.zone_names:
                for zone in zones_in_storey:
                    zone = zone.lower()

                    if zone not in downsampled_df.columns:
                        raise KeyError(f"Zone '{zone}' not found in occupancy file.")

                    schedule_series = downsampled_df[zone]

                    # Create .sch string: zone name followed by occupancy values
                    schedule_string = f"{zone}\n" + "\n".join(
                        schedule_series.astype(str).tolist()
                    )

                    occupancy_schedule_file = (
                        Path(building_config.files_dir) / f"occupancy_{zone}.sch"
                    )

                    utilities.write_string_to_file(
                        schedule_string, occupancy_schedule_file
                    )

        # TODO Below is Hannes' way, I (Jack) have used the custom zoning above
        if self.building_config.occupant_schedule_living:
            self.occupancy_schedule_living_file = (
                building_config.files_dir + "/occupancy_living.sch"
            )

            utilities.write_string_to_file(
                self.building_config.occupant_schedule_living,
                self.occupancy_schedule_living_file,
            )

        if self.building_config.occupant_schedule_bedroom:
            self.occupancy_schedule_bedroom_file = (
                building_config.files_dir + "/occupancy_bedroom.sch"
            )

            utilities.write_string_to_file(
                self.building_config.occupant_schedule_bedroom,
                self.occupancy_schedule_bedroom_file,
            )

        IDF.setiddname(EPLUS_PATH + "Energy+.idd")
        self.idf = IDF(EPLUS_PATH + "ExampleFiles/Minimal.idf")

        self.idf.idfobjects["GLOBALGEOMETRYRULES"][0].Coordinate_System = "Relative"
        self.idf.idfobjects["GLOBALGEOMETRYRULES"][
            0
        ].Vertex_Entry_Direction = "CounterClockWise"
        self.idf.idfobjects["GLOBALGEOMETRYRULES"][
            0
        ].Starting_Vertex_Position = "LowerLeftCorner"

        self.idf.idfobjects["BUILDING"][0].Solar_Distribution = "FullExterior"
        self.idf.idfobjects["TIMESTEP"][0].Number_of_Timesteps_per_Hour = 60
        self.idf.idfobjects["BUILDING"][0].Name = self.building_config.name
        self.idf.idfobjects["RUNPERIOD"][0].Begin_Year = self.building_config.year
        self.idf.idfobjects["RUNPERIOD"][0].End_Year = self.building_config.year
        self.idf.newidfobject(
            "ZoneAirHeatBalanceAlgorithm".upper(), Algorithm="AnalyticalSolution"
        )

        self.idf.newidfobject("SURFACECONVECTIONALGORITHM:INSIDE", Algorithm="Simple")
        self.idf.newidfobject("OUTPUT:DIAGNOSTICS")
        self.idf.idfobjects["OUTPUT:DIAGNOSTICS"][0].Key_1 = "DisplayExtraWarnings"
        self.idf.idfobjects["OUTPUT:DIAGNOSTICS"][0].Key_1 = "DisplayAllWarnings"

    def set_constructions(self):
        """adds materials and constructions to IDF
        then assigns each of the constructions to surfaces
        """
        for c in self.all_constructions:
            if c.materials:
                self.idf = c.add_to_idf(self.idf)

        # as CUSTOM uses geomeppy's add_block function, the surface types are
        # different to Hannes' approach, so a different approach is needed
        if self.building_config.zoning == bco.Zoning.CUSTOM.value:
            surfaces_to_remove = []
            for surface in self.idf.idfobjects["BUILDINGSURFACE:DETAILED"]:

                # Handle Wall Surfaces
                if "wall" in surface.Surface_Type.lower():
                    if "surface" in surface.Outside_Boundary_Condition:
                        surface.Construction_Name = (
                            self.partition_construction.get_name()
                        )
                        surface.Sun_Exposure = "NoSun"
                        surface.Wind_Exposure = "NoWind"
                    elif "subfloor" in surface.Name.lower():
                        surface.Sun_Exposure = "NoSun"
                        surface.Wind_Exposure = "NoWind"
                        surface.Construction_Name = self.wall_construction.get_name()
                    elif "adiabatic" in surface.Outside_Boundary_Condition.lower():
                        surface.Construction_Name = (
                            self.adiabatic_wall_construction.get_name()
                        )
                        surface.Sun_Exposure = "NoSun"
                        surface.Wind_Exposure = "NoWind"
                    else:
                        surface.Construction_Name = self.wall_construction.get_name()
                        surface.Sun_Exposure = "SunExposed"
                        surface.Wind_Exposure = "WindExposed"

                # Handle Ceiling Surfaces
                elif "ceiling" in surface.Surface_Type.lower():

                    if "loft" in surface.Zone_Name.lower():
                        surfaces_to_remove.append(surface)

                    elif "subfloor" in surface.Name.lower():
                        surface.Construction_Name = self.subfloor_roof_construction.get_name()
                        surface.Sun_Exposure = "NoSun"
                        surface.Wind_Exposure = "NoWind"

                    elif "storey 2" in surface.Name.lower():
                        surface.Construction_Name = self.last_ceiling_construction.get_name()
                        surface.Sun_Exposure = "NoSun"
                        surface.Wind_Exposure = "NoWind"

                    else:
                        surface.Construction_Name = self.ceiling_construction.get_name()
                        surface.Sun_Exposure = "NoSun"
                        surface.Wind_Exposure = "NoWind"


                # Handle Roof Surfaces
                elif "roof" in surface.Surface_Type.lower():
                    if "subfloor" in surface.Name.lower():
                        surface.Construction_Name = (
                            self.subfloor_roof_construction.get_name()
                        )
                        surface.Sun_Exposure = "NoSun"
                        surface.Wind_Exposure = "NoWind"
                    elif "surface" in surface.Outside_Boundary_Condition:
                        surface.Surface_Type = "ceiling"
                        surface.Construction_Name = self.ceiling_construction.get_name()
                        surface.Sun_Exposure = "NoSun"
                        surface.Wind_Exposure = "NoWind"
                    else:
                        surface.Construction_Name = self.roof_construction.get_name()
                        surface.Sun_Exposure = "SunExposed"
                        surface.Wind_Exposure = "WindExposed"

                # Handle Floor Surfaces
                elif "floor" in surface.Surface_Type.lower():
                    if "loft" in surface.Name.lower():
                        surface.Construction_Name = (
                            self.last_floor_construction.get_name()
                        )
                        surface.Sun_Exposure = "NoSun"
                        surface.Wind_Exposure = "NoWind"

                    elif surface.Vertex_1_Zcoordinate < 0:  # Subfloor
                        surface.Construction_Name = (
                            self.subfloor_construction.get_name()
                        )
                        surface.Outside_Boundary_Condition = "Adiabatic"
                    elif surface.Vertex_1_Zcoordinate == 0:  # Ground Floor
                        surface.Construction_Name = (
                            self.ground_floor_construction.get_name()
                        )
                        surface.Sun_Exposure = "NoSun"
                        surface.Wind_Exposure = "NoWind"
                    else:  # Upper Floors
                        surface.Construction_Name = (
                            self.upper_floor_construction.get_name()
                        )
                        surface.Sun_Exposure = "NoSun"
                        surface.Wind_Exposure = "NoWind"


                # Handle Unknown Surface Types
                else:
                    raise ValueError(f"Unknown surface type: {surface.Surface_Type}")

            for surface in surfaces_to_remove:
                self.idf.removeidfobject(surface)

            if self.building_config.window_type != "Simple":
                self.idf = self.window_construction.add_to_idf(
                    self.idf, windows=self.windows
                )
                for window in self.idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
                    if window.Surface_Type.lower() == "door":
                        continue
                    else:
                        window.Construction_Name = self.window_construction.get_name()

        else:
        # follow conventional approach laid out by Hannes
            for surface in self.idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
                if surface.Surface_Type.lower() == "wall":
                    if surface.Outside_Boundary_Condition.lower() == "zone":
                        surface.Construction_Name = (
                            self.partition_construction.get_name()
                        )  # pylint: disable=line-too-long
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
                    if window.Surface_Type.lower() == "door":
                        continue
                    else:
                        window.Construction_Name = self.window_construction.get_name()
            else:
                self.idf = self.window_system_simple.add_to_idf(self.idf)
                for window in self.idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
                    if window.Surface_Type.lower() == "door":
                        continue
                    else:
                        window.Construction_Name = "Glazing"

    def zone_not_conditioned(self, zone_name):
        return (
            "Loft" in zone_name and not self.building_config.loft_is_heated
        ) or "Subfloor" in zone_name

    def get_conditioned_zones(self):
        zone_names = self.building_config.controlled_zones  # List of controlled zones
        zones = []

        for zone in self.idf.idfobjects["ZONE"]:
            if zone.Name in zone_names:  # Corrected syntax
                zones.append(zone)

        return zones

    def add_schedules(self):
        """Adds occupancy, heating pattern, and activity schedules to the EnergyPlus IDF."""

        # Add common schedule type limits
        self.idf.newidfobject(
            "SCHEDULETYPELIMITS",
            Name="Fraction",
            Lower_Limit_Value=0,
            Upper_Limit_Value=1,
            Numeric_Type="Continuous",
            Unit_Type="Dimensionless",
        )
        self.idf.newidfobject("SCHEDULETYPELIMITS", Name="Any Number")

        self.idf.newidfobject(
            "ScheduleTypeLimits".upper(),
            Name="ActivityLevel",
            Lower_Limit_Value=0,
            Upper_Limit_Value=1000,
            Numeric_Type="CONTINUOUS",
            Unit_Type="ActivityLevel",
        )

        # Get timestep and calculate Minutes_per_Item
        timestep = self.building_config.timesteps_per_hour  # 6 = 10-min, 60 = 1-min
        minutes_per_item = 60 // timestep

        if self.building_config.zoning == bco.Zoning.CUSTOM.value:
            for zones_in_storey in self.building_config.zone_names:
                for zone in zones_in_storey:
                    if not zone:
                        continue  # Skip empty zone names

                    zone = zone.lower()
                    files_dir = Path(self.building_config.files_dir)

                    # Add heating pattern schedule if provided
                    if self.building_config.heating_pattern_schedule_file_name:
                        heating_pattern_file = files_dir / f"heating_pattern_{zone}.sch"
                        self.idf.newidfobject(
                            "SCHEDULE:FILE",
                            Name=f"Heating-Pattern-Schedule-{zone}",
                            Schedule_Type_Limits_Name="Fraction",
                            File_Name=str(heating_pattern_file),
                            Column_Number=1,
                            Rows_to_Skip_at_Top=0,
                            Number_of_Hours_of_Data=8760,
                            Minutes_per_Item=minutes_per_item,
                        )

                    # Add occupancy schedule (.sch starts with zone name on first line)
                    occupancy_file = files_dir / f"occupancy_{zone}.sch"
                    self.idf.newidfobject(
                        "SCHEDULE:FILE",
                        Name=f"Occupancy-Schedule-{zone}",
                        Schedule_Type_Limits_Name="Fraction",
                        File_Name=str(occupancy_file),
                        Column_Number=1,
                        Rows_to_Skip_at_Top=1,  # skip zone name
                        Number_of_Hours_of_Data=8760,
                        Minutes_per_Item=minutes_per_item,
                    )

                    # Add activity schedule (hardcoded, zone-dependent)
                    if "bedroom" in zone:
                        activity_lines = [
                            "Through: 12/31",
                            "For: AllDays",
                            "Until: 7:00, 80.",
                            "Until: 22:00, 120.",
                            "Until: 24:00, 80.",
                        ]
                    else:
                        activity_lines = [
                            "Through: 12/31",
                            "For: AllDays",
                            "Until: 24:00, 120.",
                        ]

                    compact = self.idf.newidfobject(
                        "SCHEDULE:COMPACT",
                        Name=f"Activity-Schedule-{zone}",
                        Schedule_Type_Limits_Name="ActivityLevel",
                    )

                    for i, line in enumerate(activity_lines, start=1):
                        compact[f"Field_{i}"] = line

        else:
            # occupants living room
            if self.building_config.occupant_schedule_living:
                self.idf.newidfobject(
                    "SCHEDULE:FILE",
                    Name="Occupancy-Schedule-Living",
                    Schedule_Type_Limits_Name="Fraction",
                    File_Name=self.occupancy_schedule_living_file,
                    Column_Number=1,
                    Rows_to_Skip_at_Top=0,
                    Number_of_Hours_of_Data=8760,
                    Minutes_per_Item=10,
                )
                self.idf.newidfobject(
                    "SCHEDULE:COMPACT",
                    Name="Activity-Schedule-Living",
                    Field_1="Through: 12/31,\n    For: AllDays,\n    Until: 24:00, 120.\n",  # pylint: disable=line-too-long
                )

            else:
                self.idf.newidfobject(
                    "SCHEDULE:COMPACT",
                    Name="Occupancy-Schedule-Living",
                    Field_1=(
                        "Through: 12/31,\n    "
                        "For: Weekdays,\n    Until: 9:00, 1.0,\n"
                        "    Until:17:00, 0.5,\n    Until:24:00, 1.,\n "
                        "   For:AllOtherDays,\n    Until:24:00,1."
                    ),
                )
                self.idf.newidfobject(
                    "SCHEDULE:COMPACT",
                    Name="Activity-Schedule-Living",
                    Field_1="Through: 12/31,\n    For: AllDays,\n    Until: 24:00, 120.\n",  # pylint: disable=line-too-long
                )

            # occupants bedroom room
            if self.building_config.occupant_schedule_bedroom:
                self.idf.newidfobject(
                    "SCHEDULE:FILE",
                    Name="Occupancy-Schedule-Bedroom",
                    Schedule_Type_Limits_Name="Fraction",
                    File_Name=self.occupancy_schedule_bedroom_file,
                    Column_Number=1,
                    Rows_to_Skip_at_Top=0,
                    Number_of_Hours_of_Data=8760,
                    Minutes_per_Item=10,
                )
                self.idf.newidfobject(
                    "SCHEDULE:COMPACT",
                    Name="Activity-Schedule-Bedroom",
                    Field_1=(
                        "Through: 12/31,\n    For: AllDays,\n    Until: 7:00, 80.,\n    "  # pylint: disable=line-too-long
                        "Until: 22:00, 120.,\n    Until: 24:00, 80.,\n"
                    ),
                )
            else:
                self.idf.newidfobject(
                    "SCHEDULE:COMPACT",
                    Name="Occupancy-Schedule-Bedroom",
                    Field_1=(
                        "Through: 12/31,\n    "
                        "For: Weekdays,\n    Until: 9:00, 1.0,\n"
                        "    Until:17:00, 0.5,\n    Until:24:00, 1.,\n "
                        "   For:AllOtherDays,\n    Until:24:00,1."
                    ),
                )
                self.idf.newidfobject(
                    "SCHEDULE:COMPACT",
                    Name="Activity-Schedule-Bedroom",
                    Field_1=(
                        "Through: 12/31,\n    For: AllDays,\n    Until: 7:00, 80.,\n    "  # pylint: disable=line-too-long
                        "Until: 22:00, 120.,\n    Until: 24:00, 80.,\n"
                    ),
                )

        self.idf.newidfobject(
            "SCHEDULE:COMPACT",
            Name="Always-Schedule",
            Schedule_Type_Limits_Name="onOff",
            Field_1="Through: 12/31,\n    For: AllDays,\n    Until: 24:00, 1.0\n",
        )

        # TODO In future remove these, or move to hvac_systems.py
        # temperature setpoints
        if self.building_config.heating_setpoint_schedule:
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Heating-Setpoint-Schedule",
                Field_1=get_schedule(self.building_config.heating_setpoint_schedule),
            )

        if self.building_config.cooling_setpoint_schedule:
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Cooling-Setpoint-Schedule",
                Field_1=get_schedule(self.building_config.cooling_setpoint_schedule),
            )

        # lighting
        if self.building_config.lighting_schedule:
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Lighting-Schedule",
                Schedule_Type_Limits_Name="any number",
                Field_1=get_schedule(self.building_config.lighting_schedule),
            )
        else:
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Lighting-Schedule",
                Schedule_Type_Limits_Name="any number",
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
                Schedule_Type_Limits_Name="any number",
                Field_1=get_schedule(self.building_config.equipment_gain_schedule),
            )
        else:
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Equipment-Schedule",
                Schedule_Type_Limits_Name="any number",
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

    def convert_to_flow(self, zone_volume, n_50):
        return (2 * zone_volume * n_50 * 0.03) / 3600

    def add_infiltration(self):
        """Adds infiltration into e+ for every zone in idf"""
        for zone in self.idf.idfobjects["ZONE"]:

            if zone.Name.lower() == "subfloor":
                self.idf.newidfobject(
                    "ZONEINFILTRATION:DESIGNFLOWRATE",
                    Name=zone.Name + "-Infiltration",
                    Zone_or_ZoneList_Name=zone.Name,
                    Design_Flow_Rate_Calculation_Method="airchanges/hour",
                    Air_Changes_per_Hour=(
                        self.building_config.subfloor_infiltration_ach
                    ),
                    Constant_Term_Coefficient=0.0,
                    Temperature_Term_Coefficient=0.0,
                    Velocity_Term_Coeﬀicient=0.224,
                    Velocity_Squared_Term_Coefficient=0.0,
                    Schedule_Name="Always-Schedule",
                )
            elif zone.Name.lower() == "loft":
                self.idf.newidfobject(
                    "ZONEINFILTRATION:DESIGNFLOWRATE",
                    Name=zone.Name + "-Infiltration",
                    Zone_or_ZoneList_Name=zone.Name,
                    Design_Flow_Rate_Calculation_Method="airchanges/hour",
                    Air_Changes_per_Hour=(self.building_config.loft_infiltration_ach),
                    Constant_Term_Coefficient=0.0,
                    Temperature_Term_Coefficient=0.0,
                    Velocity_Term_Coeﬀicient=0.224,
                    Velocity_Squared_Term_Coefficient=0.0,
                    Schedule_Name="Always-Schedule",
                )
            else:
                self.idf.newidfobject(
                    "ZONEINFILTRATION:DESIGNFLOWRATE",
                    Name=zone.Name + "-Infiltration",
                    Zone_or_ZoneList_Name=zone.Name,
                    Design_Flow_Rate_Calculation_Method="airchanges/hour",
                    Air_Changes_per_Hour=(self.building_config.infiltration_rate),
                    Constant_Term_Coefficient=0.0,
                    Temperature_Term_Coefficient=0.0,
                    Velocity_Term_Coeﬀicient=0.224,
                    Velocity_Squared_Term_Coefficient=0.0,
                    Schedule_Name="Always-Schedule",
                )

    def add_zone_capacitance_multiplier(self):
        """adds temp capacitance multiplier to increase heating time"""
        for zone in self.get_conditioned_zones():
            self.idf.newidfobject(
                "ZONECAPACITANCEMULTIPLIER:RESEARCHSPECIAL",
                Name=zone.Name + "capacitance_multiplier",
                Zone_or_ZoneList_Name=zone.Name,
                Temperature_Capacity_Multiplier=(
                    self.building_config.capacitance_multiplier
                ),
            )

    # def add_air_flow_network(self):
    #     """Method to add air flow network"""

    #     # Enable the AirflowNetwork model
    #     self.idf.newidfobject(
    #         "AirflowNetwork:SimulationControl".upper(),
    #         Name="AFNControl",
    #         AirflowNetwork_Control="MultizoneWithoutDistribution",
    #         Wind_Pressure_Coefficient_Type="SurfaceAverageCalculation",
    #         Height_Selection_for_Local_Wind_Pressure_Calculation="OpeningHeight",
    #         Building_Type="LOWRISE",
    #         Maximum_Number_of_Iterations=500,
    #         Initialization_Type="ZeroNodePressures",
    #         Relative_Airflow_Convergence_Tolerance=1.0e-4,
    #         Absolute_Airflow_Convergence_Tolerance=1.0e-6,
    #         Convergence_Acceleration_Limit=-0.5,
    #         Azimuth_Angle_of_Long_Axis_of_Building=0.0,
    #         Ratio_of_Building_Width_Along_Short_Axis_to_Width_Along_Long_Axis=1.0,
    #     )

    #     # # Add AirflowNetwork:MultiZone:WindPressureCoefficientValues
    #     self.idf.newidfobject(
    #         "AirflowNetwork:MultiZone:WindPressureCoefficientValues".upper(),
    #         Name="VerticalFacade_WPCValues",
    #         AirflowNetworkMultiZoneWindPressureCoefficientArray_Name="Every 45 Degrees",
    #         Wind_Pressure_Coefficient_Value_1=0.4,
    #         Wind_Pressure_Coefficient_Value_2=0.1,
    #         Wind_Pressure_Coefficient_Value_3=-0.3,
    #         Wind_Pressure_Coefficient_Value_4=-0.35,
    #         Wind_Pressure_Coefficient_Value_5=-0.2,
    #         Wind_Pressure_Coefficient_Value_6=-0.35,
    #         Wind_Pressure_Coefficient_Value_7=-0.3,
    #         Wind_Pressure_Coefficient_Value_8=-0.1,
    #     )
    #     self.idf.newidfobject(
    #         "AirflowNetwork:MultiZone:WindPressureCoefficientArray".upper(),
    #         Name="Every 45 Degrees",
    #         Wind_Direction_1=0,
    #         Wind_Direction_2=45,
    #         Wind_Direction_3=90,
    #         Wind_Direction_4=135,
    #         Wind_Direction_5=180,
    #         Wind_Direction_6=225,
    #         Wind_Direction_7=270,
    #         Wind_Direction_8=315,
    #     )

    #     # # Add an external node for outdoors, referencing the wind pressure coefficient values
    #     self.idf.newidfobject(
    #         "AirflowNetwork:MultiZone:ExternalNode".upper(),
    #         Name="outdoors",  # External node name
    #         External_Node_Height=1.54,  # Adjust as needed for the building height
    #         Wind_Pressure_Coefficient_Curve_Name="VerticalFacade_WPCValues",  # Reference the WPC values
    #     )

    #     # Add AirflowNetwork:MultiZone:Zone for all zones
    #     for zone in self.idf.idfobjects["ZONE"]:
    #         self.idf.newidfobject(
    #             "AirflowNetwork:MultiZone:Zone".upper(),
    #             Zone_Name=zone.Name,  # The name of the thermal zone
    #             Ventilation_Control_Mode="NoVent",  # Cracks operate passively
    #             Venting_Availability_Schedule_Name="Always-Schedule",  # Irrelevant
    #         )

    #     # Define reusable crack templates with properties
    #     crack_definitions = {
    #         "ExternalWallCrack": {"Cq": 0.002, "n": 0.7},
    #         "InternalWallCrack": {"Cq": 0.005, "n": 0.75},
    #         "FloorCeilingCrack": {"Cq": 0.002, "n": 0.7},
    #         "RoofCrack": {"Cq": 0.00015, "n": 0.7},
    #         "WindowCrack": {"Cq": 0.01, "n": 0.65},
    #     }

    #     # Add crack templates to the IDF
    #     for crack_name, properties in crack_definitions.items():
    #         self.idf.newidfobject(
    #             "AirflowNetwork:MultiZone:Surface:Crack".upper(),
    #             Name=crack_name,
    #             Air_Mass_Flow_Coefficient_at_Reference_Conditions=properties["Cq"],
    #             Air_Mass_Flow_Exponent=properties["n"],
    #         )

    #     # Helper function to classify surface types
    #     def classify_surface(surface_type, boundary_condition):
    #         if surface_type.lower() == "wall":
    #             return (
    #                 "ExternalWallCrack"
    #                 if boundary_condition.lower() == "outdoors"
    #                 else "InternalWallCrack"
    #             )
    #         elif surface_type.lower() in ["floor", "ceiling"]:
    #             return "FloorCeilingCrack"
    #         elif surface_type.lower() == "roof":
    #             return "RoofCrack"
    #         else:
    #             return None

    #     # Keep track of processed surface pairs to avoid duplication
    #     processed_surface_pairs = set()

    #     # Loop through all BuildingSurface:Detailed objects
    #     for surface in self.idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
    #         surface_name = surface.Name
    #         surface_type = surface.Surface_Type
    #         boundary_condition = surface.Outside_Boundary_Condition
    #         boundary_object = surface.Outside_Boundary_Condition_Object

    #         # Skip surfaces with boundary condition Ground or Adiabatic
    #         if boundary_condition.lower() in ["ground", "adiabatic"]:
    #             continue

    #         # Avoid duplication for shared surfaces
    #         if boundary_condition.lower() == "surface" and boundary_object:
    #             # Create a unique key for the surface pair (order-independent)
    #             surface_pair = tuple(sorted([surface_name, boundary_object]))
    #             if surface_pair in processed_surface_pairs:
    #                 continue  # Skip if this pair has already been processed
    #             processed_surface_pairs.add(surface_pair)

    #         # Classify the surface and get the appropriate crack template
    #         crack_name = classify_surface(surface_type, boundary_condition)
    #         if not crack_name:
    #             continue  # Skip surfaces that don't match any category

    #         # Add the AirflowNetwork:MultiZone:Surface object for this surface
    #         self.idf.newidfobject(
    #             "AirflowNetwork:MultiZone:Surface".upper(),
    #             Surface_Name=surface_name,
    #             Leakage_Component_Name=crack_name,
    #             External_Node_Name=(
    #                 "Outdoors" if boundary_condition.lower() == "outdoors" else ""
    #             ),
    #         )

    #     # Loop through all Windows
    #     for window in self.idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
    #         window_name = window.Name
    #         boundary_condition = window.Outside_Boundary_Condition_Object

    #         if boundary_condition.lower() == "surface":
    #             continue

    #         self.idf.newidfobject(
    #             "AirflowNetwork:MultiZone:Surface".upper(),
    #             Surface_Name=window_name,
    #             Leakage_Component_Name="WindowCrack",
    #             External_Node_Name="Outdoors",
    #         )

    def add_zone_mixing_for_doors(self, mixing_flow_rate=0.01):
        """Adds ZONECROSSMIXING objects for all doors that connect two zones."""

        added_mixing_pairs = set()  # Track added pairs to avoid duplicates

        # Iterate through all fenestration surfaces to find doors
        for door in self.idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
            if door.Surface_Type.lower() == "door":
                # Get the parent surface
                parent_surface_name = door.Building_Surface_Name

                # Find the corresponding building surface object
                parent_surface = next(
                    (
                        surf
                        for surf in self.idf.idfobjects["BUILDINGSURFACE:DETAILED"]
                        if surf.Name == parent_surface_name
                    ),
                    None,
                )

                if not parent_surface:
                    continue  # Skip if no matching parent surface found

                # Get the zone the door belongs to
                zone_name = parent_surface.Zone_Name

                # Get the adjacent fenestration surface object
                adjacent_door = next(
                    (
                        other_door
                        for other_door in self.idf.idfobjects[
                            "BUILDINGSURFACE:DETAILED"
                        ]
                        if other_door.Name == door.Outside_Boundary_Condition_Object
                    ),
                    None,
                )

                if not adjacent_door:
                    continue  # Skip if no adjacent door found

                # Find the parent building surface of the adjacent door
                adjacent_surface = next(
                    (
                        surf
                        for surf in self.idf.idfobjects["BUILDINGSURFACE:DETAILED"]
                        if surf.Name == adjacent_door.Building_Surface_Name
                    ),
                    None,
                )

                if not adjacent_surface:
                    continue  # Skip if no adjacent building surface found

                adjacent_zone_name = adjacent_surface.Zone_Name

                # Ensure we have two different zones (not an external door)
                if zone_name and adjacent_zone_name and zone_name != adjacent_zone_name:
                    zone_pair = tuple(
                        sorted([zone_name, adjacent_zone_name])
                    )  # Sort to maintain consistency

                    if zone_pair in added_mixing_pairs:
                        continue  # Skip if this zone pair is already processed

                    print(
                        f"Adding zone mixing between {zone_name} and {adjacent_zone_name}"
                    )

                    # Create bidirectional mixing (only once per unique zone pair)
                    self.idf.newidfobject(
                        "ZONECROSSMIXING",
                        Name=f"{zone_name}_to_{adjacent_zone_name}_Mixing",
                        Zone_Name=zone_name,
                        Design_Flow_Rate=mixing_flow_rate,  # Set a constant value for now
                        Schedule_Name="AlwaysOnSchedule",
                        Source_Zone_Name=adjacent_zone_name,
                        Delta_Temperature=0.0,
                    )

                    self.idf.newidfobject(
                        "ZONECROSSMIXING",
                        Name=f"{adjacent_zone_name}_to_{zone_name}_Mixing",
                        Zone_Name=adjacent_zone_name,
                        Design_Flow_Rate=mixing_flow_rate,  # Set a constant value for now
                        Schedule_Name="AlwaysOnSchedule",
                        Source_Zone_Name=zone_name,
                        Delta_Temperature=0.0,
                    )

                    added_mixing_pairs.add(zone_pair)  # Mark this pair as added

    def add_zone_mixing(self):
        self.idf.newidfobject(
            "ZONECROSSMIXING",
            Zone_Name="Test",
            Schedule_Name="Always-Schedule",
            Design_Flow_Rate_Calculation_Method="AirChanges/Hour",
            Air_Changes_Per_Hour=0.5,
            Source_Zone_Name="Testing1",
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

    def add_beizaee_gains(self):
        """"""

        def add_back_room_gains_and_schedules(self):
            """Add schedules and internal gains for the back room (dining room)."""

            # Morning Schedule (08:00–08:30 on weekdays, 09:30–10:00 on weekends)
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="BackRoom-MorningGains-Schedule",
                Schedule_Type_Limits_Name="Any Number",
                Field_1="Through: 12/31",
                Field_2="For: Weekdays",
                Field_3="Until: 08:00, 0",
                Field_4="Until: 08:30, 1",  # Total actual gains (Hot food)
                Field_5="Until: 24:00, 0",
                Field_6="For: Weekends",
                Field_7="Until: 09:30, 0",
                Field_8="Until: 10:00, 1",  # Total actual gains (Hot food)
                Field_9="Until: 24:00, 0",
            )

            # Evening Schedule (17:00–18:00)
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="BackRoom-EveningGains-Schedule",
                Schedule_Type_Limits_Name="Any Number",
                Field_1="Through: 12/31",
                Field_2="For: Weekdays Weekends",
                Field_3="Until: 17:00, 0",
                Field_4="Until: 18:00, 1",  # Total actual gains (Hot food + Lighting)
                Field_5="Until: 24:00, 0",
            )

            # Add Morning Gains (Hot Food)
            self.idf.newidfobject(
                "ELECTRICEQUIPMENT",
                Name="BackRoom-HotFood-Morning",
                Zone_or_ZoneList_Name="backroom",
                Schedule_Name="BackRoom-MorningGains-Schedule",
                Design_Level_Calculation_Method="EquipmentLevel",
                Design_Level=460,  # Total actual gains
                Fraction_Latent=0.0,
                Fraction_Radiant=0.3,  # Adjust as needed
                Fraction_Lost=0.1,  # Adjust as needed
                EndUse_Subcategory="Cooking",
            )

            # Add Evening Gains (Hot Food)
            self.idf.newidfobject(
                "ELECTRICEQUIPMENT",
                Name="BackRoom-HotFood-Evening",
                Zone_or_ZoneList_Name="backroom",
                Schedule_Name="BackRoom-EveningGains-Schedule",
                Design_Level_Calculation_Method="EquipmentLevel",
                Design_Level=450,  # Hot food portion of the gains
                Fraction_Latent=0.0,
                Fraction_Radiant=0.3,  # Adjust as needed
                Fraction_Lost=0.1,  # Adjust as needed
                EndUse_Subcategory="Cooking",
            )

            # Add Evening Gains (Lighting)
            self.idf.newidfobject(
                "LIGHTS",
                Name="BackRoom-Lighting-Evening",
                Zone_or_ZoneList_Name="backroom",
                Schedule_Name="BackRoom-EveningGains-Schedule",
                Design_Level_Calculation_Method="LightingLevel",
                Lighting_Level=30,  # Lighting power in watts
                Fraction_Radiant=0.7,
                Fraction_Visible=0.3,
                Fraction_Replaceable=1.0,
                EndUse_Subcategory="Lighting",
            )

        def add_bedroom1_gains_and_schedules(self):
            """Add schedules and internal gains for Bedroom 1."""

            # Afternoon Schedule (16:00–17:00)
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Bedroom1-AfternoonGains-Schedule",
                Schedule_Type_Limits_Name="Any Number",
                Field_1="Through: 12/31",
                Field_2="For: Weekdays Weekends",
                Field_3="Until: 16:00, 0",
                Field_4="Until: 17:00, 1",  # Total actual gains (Lighting + others)
                Field_5="Until: 24:00, 0",
            )

            # Evening Schedule (19:00–20:00)
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Bedroom1-EveningGains-Schedule",
                Schedule_Type_Limits_Name="Any Number",
                Field_1="Through: 12/31",
                Field_2="For: Weekdays Weekends",
                Field_3="Until: 19:00, 0",
                Field_4="Until: 20:00, 1",  # Total actual gains (Lighting + others)
                Field_5="Until: 24:00, 0",
            )

            # Late Evening Schedule (20:00–22:30)
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Bedroom1-LateEveningGains-Schedule",
                Schedule_Type_Limits_Name="Any Number",
                Field_1="Through: 12/31",
                Field_2="For: Weekdays Weekends",
                Field_3="Until: 20:00, 0",
                Field_4="Until: 22:30, 1",  # Total actual gains (Lighting + Computer)
                Field_5="Until: 24:00, 0",
            )

            # Add Gains for Afternoon
            self.idf.newidfobject(
                "LIGHTS",
                Name="Bedroom1-Lighting-Afternoon",
                Zone_or_ZoneList_Name="bedroom_1",
                Schedule_Name="Bedroom1-AfternoonGains-Schedule",
                Design_Level_Calculation_Method="LightingLevel",
                Lighting_Level=30,  # Lighting power in watts
                Fraction_Radiant=0.7,
                Fraction_Visible=0.3,
                Fraction_Replaceable=1.0,
                EndUse_Subcategory="Lighting",
            )

            # Add Gains for Evening
            self.idf.newidfobject(
                "LIGHTS",
                Name="Bedroom1-Lighting-Evening",
                Zone_or_ZoneList_Name="bedroom_1",
                Schedule_Name="Bedroom1-EveningGains-Schedule",
                Design_Level_Calculation_Method="LightingLevel",
                Lighting_Level=30,  # Lighting power in watts
                Fraction_Radiant=0.7,
                Fraction_Visible=0.3,
                Fraction_Replaceable=1.0,
                EndUse_Subcategory="Lighting",
            )

            # Add Gains for Late Evening (Lighting)
            self.idf.newidfobject(
                "LIGHTS",
                Name="Bedroom1-Lighting-LateEvening",
                Zone_or_ZoneList_Name="bedroom_1",
                Schedule_Name="Bedroom1-LateEveningGains-Schedule",
                Design_Level_Calculation_Method="LightingLevel",
                Lighting_Level=30,  # Lighting power in watts
                Fraction_Radiant=0.7,
                Fraction_Visible=0.3,
                Fraction_Replaceable=1.0,
                EndUse_Subcategory="Lighting",
            )

            # Add Gains for Late Evening (Computer)
            self.idf.newidfobject(
                "ELECTRICEQUIPMENT",
                Name="Bedroom1-Computer-LateEvening",
                Zone_or_ZoneList_Name="bedroom_1",
                Schedule_Name="Bedroom1-LateEveningGains-Schedule",
                Design_Level_Calculation_Method="EquipmentLevel",
                Design_Level=100,  # Computer power in watts
                Fraction_Latent=0.0,
                Fraction_Radiant=0.4,  # Adjust as needed
                Fraction_Lost=0.2,  # Adjust as needed
                EndUse_Subcategory="Computer",
            )

        def add_front_room_gains_and_schedules(self):
            """Add schedules and internal gains for the front room (living room)."""

            # Early Evening Schedule (18:00–19:00)
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="FrontRoom-EarlyEveningGains-Schedule",
                Schedule_Type_Limits_Name="Any Number",
                Field_1="Through: 12/31",
                Field_2="For: Weekdays Weekends",
                Field_3="Until: 18:00, 0",
                Field_4="Until: 19:00, 1",  # Total actual gains (TV: 150 W + Lighting: 30 W + others)
                Field_5="Until: 24:00, 0",
            )

            # Late Evening Schedule (19:00–22:30)
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="FrontRoom-LateEveningGains-Schedule",
                Schedule_Type_Limits_Name="Any Number",
                Field_1="Through: 12/31",
                Field_2="For: Weekdays Weekends",
                Field_3="Until: 19:00, 0",
                Field_4="Until: 22:30, 1",  # Total actual gains (TV: 150 W + Lighting: 30 W + others)
                Field_5="Until: 24:00, 0",
            )

            # Add TV Gains for Early Evening
            self.idf.newidfobject(
                "ELECTRICEQUIPMENT",
                Name="FrontRoom-TV-EarlyEvening",
                Zone_or_ZoneList_Name="front_room",
                Schedule_Name="FrontRoom-EarlyEveningGains-Schedule",
                Design_Level_Calculation_Method="EquipmentLevel",
                Design_Level=150,  # TV power in watts
                Fraction_Latent=0.0,
                Fraction_Radiant=0.2,  # Adjust as needed
                Fraction_Lost=0.1,  # Adjust as needed
                EndUse_Subcategory="TV",
            )

            # Add TV Gains for Late Evening
            self.idf.newidfobject(
                "ELECTRICEQUIPMENT",
                Name="FrontRoom-TV-LateEvening",
                Zone_or_ZoneList_Name="front_room",
                Schedule_Name="FrontRoom-LateEveningGains-Schedule",
                Design_Level_Calculation_Method="EquipmentLevel",
                Design_Level=150,  # TV power in watts
                Fraction_Latent=0.0,
                Fraction_Radiant=0.2,  # Adjust as needed
                Fraction_Lost=0.1,  # Adjust as needed
                EndUse_Subcategory="TV",
            )

            # Add Lighting Gains for Early Evening
            self.idf.newidfobject(
                "LIGHTS",
                Name="FrontRoom-Lighting-EarlyEvening",
                Zone_or_ZoneList_Name="front_room",
                Schedule_Name="FrontRoom-EarlyEveningGains-Schedule",
                Design_Level_Calculation_Method="LightingLevel",
                Lighting_Level=30,  # Lighting power in watts
                Fraction_Radiant=0.7,
                Fraction_Visible=0.3,
                Fraction_Replaceable=1.0,
                EndUse_Subcategory="Lighting",
            )

            # Add Lighting Gains for Late Evening
            self.idf.newidfobject(
                "LIGHTS",
                Name="FrontRoom-Lighting-LateEvening",
                Zone_or_ZoneList_Name="front_room",
                Schedule_Name="FrontRoom-LateEveningGains-Schedule",
                Design_Level_Calculation_Method="LightingLevel",
                Lighting_Level=30,  # Lighting power in watts
                Fraction_Radiant=0.7,
                Fraction_Visible=0.3,
                Fraction_Replaceable=1.0,
                EndUse_Subcategory="Lighting",
            )

        def add_kitchen_gains_and_schedules(self):
            """Add schedules and internal gains for the kitchen."""

            # Morning Cooking Schedule
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Kitchen-MorningCooking-Schedule",
                Schedule_Type_Limits_Name="Any Number",
                Field_1="Through: 12/31",
                Field_2="For: Weekdays",
                Field_3="Until: 07:30, 0",
                Field_4="Until: 08:00, 1",  # Morning cooking
                Field_5="Until: 24:00, 0",
                Field_6="For: Weekends",
                Field_7="Until: 09:00, 0",
                Field_8="Until: 09:30, 1",  # Morning cooking
                Field_9="Until: 24:00, 0",
            )

            # Evening Cooking Schedule
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Kitchen-EveningCooking-Schedule",
                Schedule_Type_Limits_Name="Any Number",
                Field_1="Through: 12/31",
                Field_2="For: Weekdays Weekends",
                Field_3="Until: 16:00, 0",
                Field_4="Until: 17:00, 1",  # Evening cooking
                Field_5="Until: 24:00, 0",
            )

            # Lighting Schedule
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Kitchen-Lighting-Schedule",
                Schedule_Type_Limits_Name="Any Number",
                Field_1="Through: 12/31",
                Field_2="For: Weekdays Weekends",
                Field_3="Until: 16:00, 0",
                Field_4="Until: 17:00, 1",  # Lighting
                Field_5="Until: 24:00, 0",
            )

            # Fridge Schedule
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Kitchen-Fridge-Schedule",
                Schedule_Type_Limits_Name="Any Number",
                Field_1="Through: 12/31",
                Field_2="For: AllDays",
                Field_3="Until: 24:00, 1",  # Fridge (constant all day)
            )

            # Add Morning Cooking Gains
            self.idf.newidfobject(
                "ELECTRICEQUIPMENT",
                Name="Kitchen-MorningCooking",
                Zone_or_ZoneList_Name="kitchen",
                Schedule_Name="Kitchen-MorningCooking-Schedule",
                Design_Level_Calculation_Method="EquipmentLevel",
                Design_Level=160,
                Fraction_Latent=0.0,
                Fraction_Radiant=0.3,  # Adjust as needed
                Fraction_Lost=0.1,  # Adjust as needed
                EndUse_Subcategory="Cooking",
            )

            # Add Evening Cooking Gains
            self.idf.newidfobject(
                "ELECTRICEQUIPMENT",
                Name="Kitchen-EveningCooking",
                Zone_or_ZoneList_Name="kitchen",
                Schedule_Name="Kitchen-EveningCooking-Schedule",
                Design_Level_Calculation_Method="EquipmentLevel",
                Design_Level=1600,
                Fraction_Latent=0.0,
                Fraction_Radiant=0.3,  # Adjust as needed
                Fraction_Lost=0.1,  # Adjust as needed
                EndUse_Subcategory="Cooking",
            )

            # Add Lighting Gains
            self.idf.newidfobject(
                "LIGHTS",
                Name="Kitchen-Lighting",
                Zone_or_ZoneList_Name="kitchen",
                Schedule_Name="Kitchen-Lighting-Schedule",
                Design_Level_Calculation_Method="LightingLevel",
                Lighting_Level=54,
                Fraction_Radiant=0.7,
                Fraction_Visible=0.3,
                Fraction_Replaceable=1.0,
                EndUse_Subcategory="Lighting",
            )

            # Add Fridge Gains
            self.idf.newidfobject(
                "ELECTRICEQUIPMENT",
                Name="Kitchen-Fridge",
                Zone_or_ZoneList_Name="kitchen",
                Schedule_Name="Kitchen-Fridge-Schedule",
                Design_Level_Calculation_Method="EquipmentLevel",
                Design_Level=60,
                Fraction_Latent=0.0,
                Fraction_Radiant=0.2,
                Fraction_Lost=0.2,
                EndUse_Subcategory="Fridge",
            )

        add_kitchen_gains_and_schedules(self)
        add_front_room_gains_and_schedules(self)
        add_back_room_gains_and_schedules(self)
        add_bedroom1_gains_and_schedules(self)

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
            Interpolate_to_Timestep="no",
        )
        self.idf.newidfobject(
            "SCHEDULE:FILE",
            Name="Gas Pricing Schedule",
            Schedule_Type_Limits_Name="Any Number",
            File_Name=get_gas_pricing_file_path(
                self.building_config.gas_pricing_file_name
            ),
            Column_Number=2,
            Rows_to_Skip_at_Top=1,
            Number_of_Hours_of_Data=8760,
            Minutes_per_Item=60,
            Interpolate_to_Timestep="no",
        )
        self.idf.newidfobject(
            "SCHEDULE:FILE",
            Name="Electricity Pricing Schedule",
            Schedule_Type_Limits_Name="Any Number",
            File_Name=get_electricity_pricing_file_path(
                self.building_config.electricity_pricing_file_name
            ),
            Column_Number=2,
            Rows_to_Skip_at_Top=1,
            Number_of_Hours_of_Data=8760,
            Minutes_per_Item=30,
            Interpolate_to_Timestep="no",
        )
        self.idf.newidfobject(
            "SCHEDULE:FILE",
            Name="Electricity Surplus Schedule",
            Schedule_Type_Limits_Name="Any Number",
            File_Name=get_electricity_surplus_file_path(
                self.building_config.electricity_surplus_file_name
            ),
            Column_Number=2,
            Rows_to_Skip_at_Top=1,
            Number_of_Hours_of_Data=8760,
            Minutes_per_Item=30,
            Interpolate_to_Timestep="no",
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
            Month=1,
            Day_of_Month=16,
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
        # Will need to comment out the surfaces, boundary conditions, but keep
        self.idf, zone_areas = add_surfaces_and_zones(self.idf, self.building_config)

        # Get the geometry
        # geometry_path = str(BASE_DIR / "cubes/data/geometry/geometry.idf")

        # geometry = IDF(geometry_path)

        # # Copy objects from 'geometry.idf' to 'self.idf'
        # for key in geometry.idfobjects:
        #     for idfobject in geometry.idfobjects[key]:
        #         self.idf.copyidfobject(idfobject)

        # Extract floor area for each zone
        # zone_areas = {}
        # for zone in self.idf.idfobjects["ZONE"]:
        #     zone_areas[zone.Name] = zone.Floor_Area

        # set rotation
        self.idf.idfobjects["BUILDING"][0].North_Axis = self.building_config.rotation
        self.idf.translate_to_origin()

        # self.add_neighbours()


        self.add_schedules()
        self.add_people()
        self.idf = add_ventilation(
            self.idf, self.building_config, self.get_conditioned_zones()
        )
        self.add_infiltration()

        self.add_internal_mass(zone_areas)
        self.add_zone_capacitance_multiplier()
        self.add_zone_mixing_for_doors()
        self.add_internal_gains()

        self.add_environmental_impact_factors()
        self.set_design_days()

        if self.building_config.pv_present:
            self.idf = add_pv_and_battery(self.idf, self.building_config)

        # Unneeded now
        # self.add_beizaee_gains()
        # self.add_air_flow_network()
        self.set_boundary_conditions()
        self.add_windows()
        self.set_constructions()
        self.add_openings()
        self.add_airflow_network()

        self.idf = add_heating_system(
            self.idf, self.building_config, self.get_conditioned_zones()
        )

        # HACK
        self.idf.translate([0, 0, 0.6])

        return self.idf

    def add_airflow_network(self):
        self.idf = add_airflow_network(self.idf, self.building_config)


    def add_openings(self):
        """Add door and hole openings by matching inter-zone surfaces and scaling down for opening size.

        Uses building_config.openings which defines zone pairs, orientation (vertical/horizontal),
        material name, and target area. Automatically finds the correct surface pair and creates
        fenestration surfaces scaled to the requested area.
        """
        openings = getattr(self.building_config, "openings", [])
        if not openings:
            print("No openings defined in building configuration.")
            return

        # 1. Collect true inter-zone surfaces (surface boundary condition)
        true_surfaces = [
            sf for sf in self.idf.idfobjects["BUILDINGSURFACE:DETAILED"]
            if sf.Outside_Boundary_Condition.lower() == "surface"
        ]

        # 2. Loop through each requested opening
        for opening in openings:
            zones = [z.lower() for z in opening["zones"]]
            area = opening["area"]
            orientation = opening["orientation"].lower()

            # --- find matching surface pair ---
            matched_pair = None
            for sf in true_surfaces:
                if sf.Zone_Name.lower() == zones[0]:
                    opp = sf.Outside_Boundary_Condition_Object
                    if not opp:
                        continue
                    opp_sf = next(
                        (s for s in true_surfaces if s.Name == opp and s.Zone_Name.lower() == zones[1]),
                        None
                    )
                    if opp_sf:
                        matched_pair = (sf, opp_sf)
                        break
                elif sf.Zone_Name.lower() == zones[1]:
                    opp = sf.Outside_Boundary_Condition_Object
                    if not opp:
                        continue
                    opp_sf = next(
                        (s for s in true_surfaces if s.Name == opp and s.Zone_Name.lower() == zones[0]),
                        None
                    )
                    if opp_sf:
                        matched_pair = (sf, opp_sf)
                        break

            if not matched_pair:
                print(f"No matching surface pair for opening {zones}")
                continue

            sf_from, sf_to = matched_pair

            # --- area check ---
            if sf_from.area < area:
                print(f"Warning: requested opening area {area:.2f} m² "
                    f"is larger than surface {sf_from.Name} ({sf_from.area:.2f} m²). "
                    f"Using maximum available area instead.")
                area = sf_from.area * 0.95  # slightly smaller than full surface

            # --- orientation check ---
            if orientation == "horizontal":
                suffix = "hole"
                construction_name = self.hole_construction.get_name()
                if not ("floor" in sf_from.Surface_Type.lower() or "ceiling" in sf_from.Surface_Type.lower()):
                    print(f"Skipping {sf_from.Name}: not horizontal surface for horizontal opening")
                    continue
            elif orientation == "vertical":
                suffix = "door"
                construction_name = self.partition_door_construction.get_name()
                if "wall" not in sf_from.Surface_Type.lower():
                    print(f"Skipping {sf_from.Name}: not vertical surface for vertical opening")
                    continue

            # --- create scaled polygon (simple centroid shrink) ---
            coords = np.array(sf_from.coords)
            centroid = coords.mean(axis=0)
            scale = np.sqrt(area / sf_from.area)
            opening_coords = centroid + (coords - centroid) * scale

            # --- add fenestration surfaces ---
            self._add_fenestration(sf_from, sf_to, opening_coords, suffix, construction_name)
            print(f"Added opening between {zones[0]} and {zones[1]} "
                f"({orientation}, {area:.2f} m²)")


    def _add_fenestration(self, surf_from, surf_to, door_coords, suffix, construction_name):
        """Helper to add fenestration for a matched surface pair."""
        name_from = f"{surf_from.Name}_{suffix}"
        name_to = f"{surf_to.Name}_{suffix}"

        self.idf.newidfobject(
            "FENESTRATIONSURFACE:DETAILED",
            Name=name_from,
            Surface_Type="Door",
            Construction_Name=construction_name,
            Building_Surface_Name=surf_from.Name,
            Outside_Boundary_Condition_Object=name_to,
            View_Factor_to_Ground="AutoCalculate",
            Multiplier=1,
            Number_of_Vertices=4,
            **{f"Vertex_{i+1}_Xcoordinate": door_coords[i][0] for i in range(4)},
            **{f"Vertex_{i+1}_Ycoordinate": door_coords[i][1] for i in range(4)},
            **{f"Vertex_{i+1}_Zcoordinate": door_coords[i][2] for i in range(4)},
        )

        # --- For the opposite zone door, reverse vertex order to flip normal ---
        door_coords_reversed = door_coords[::-1]

        self.idf.newidfobject(
            "FENESTRATIONSURFACE:DETAILED",
            Name=name_to,
            Surface_Type="Door",
            Construction_Name=construction_name,
            Building_Surface_Name=surf_to.Name,
            Outside_Boundary_Condition_Object=name_from,
            View_Factor_to_Ground="AutoCalculate",
            Multiplier=1,
            Number_of_Vertices=len(door_coords_reversed),
            **{f"Vertex_{i+1}_Xcoordinate": door_coords_reversed[i][0] for i in range(len(door_coords_reversed))},
            **{f"Vertex_{i+1}_Ycoordinate": door_coords_reversed[i][1] for i in range(len(door_coords_reversed))},
            **{f"Vertex_{i+1}_Zcoordinate": door_coords_reversed[i][2] for i in range(len(door_coords_reversed))},
        )


    def add_windows(self):
        """Method which adds window strips into IDF and applies thermal bridging."""

        if self.building_config.zoning == bco.Zoning.CUSTOM.value:

            # Retrieve surfaces by type
            floors = self.idf.getsurfaces("floor")
            walls = self.idf.getsurfaces("wall")

            # Set Window-to-Wall Ratios (WWR) for each orientation
            for idx, orient in enumerate(["north", "east", "south", "west"]):
                if self.building_config.wtw_ratios[idx] > 0:
                    self.idf.set_wwr(
                        wwr=self.building_config.wtw_ratios[idx],
                        orientation=orient
                    )

            # Thermal bridging correction factors for different junctions
            window_reveal_factor = self.building_config.thermal_bridging_coefficient
            window_sill_factor = self.building_config.thermal_bridging_coefficient
            corner_window_factor = self.building_config.thermal_bridging_coefficient

            # Function to determine if a window is adjacent to a given type of surface
            def is_window_adjacent_to(window, other_surfaces):
                """Check if a window is adjacent to any of the given surfaces."""
                window_surface = self.idf.getobject(
                    "BUILDINGSURFACE:DETAILED", window.Building_Surface_Name
                )
                if not window_surface:
                    return False

                for other_surface in other_surfaces:
                    for win_vertex in window_surface.coords:
                        for other_vertex in other_surface.coords:
                            if (
                                abs(win_vertex[0] - other_vertex[0]) < 0.1
                                and abs(win_vertex[1] - other_vertex[1]) < 0.1
                            ):
                                return True
                return False

            # Function to create a modified construction with adjusted U-value
            # for windows
            def create_modified_window_construction(window, bridge_factor):
                """Create a new construction with an effective U-value including
                thermal bridging."""
                original = self.idf.getobject("CONSTRUCTION", window.Construction_Name)
                if not original or "_ThermalBridge" in original.Name:
                    return None

                # Calculate the base U-value from the original construction layers
                total_thickness = sum(
                    self.idf.getobject("MATERIAL", layer).Thickness
                    for layer in original.Material_Layers
                    if self.idf.getobject("MATERIAL", layer)
                )
                u_value_base = sum(
                    1 / (material.Thickness / material.Conductivity)
                    for material in (
                        self.idf.getobject("MATERIAL", layer)
                        for layer in original.Material_Layers
                        if self.idf.getobject("MATERIAL", layer)
                    )
                )

                # Calculate the effective U-value by adding the combined thermal
                # bridge factor
                u_value_effective = u_value_base + bridge_factor

                # Create a new construction name with the combined factor in the name
                new_construction_name = (
                    f"{original.Name}" f"_ThermalBridge_{bridge_factor:.2f}"
                )

                # Create a new construction object with the modified U-value
                new_construction = self.idf.newidfobject(
                    "CONSTRUCTION", Name=new_construction_name
                )
                new_construction.Material_Layers = original.Material_Layers[:]

                # Adjust the first material's conductivity to achieve the
                # effective U-value
                for layer_name in new_construction.Material_Layers:
                    material = self.idf.getobject("MATERIAL", layer_name)
                    if material and total_thickness > 0:
                        # Adjust conductivity to achieve the effective U-value
                        material.Conductivity = total_thickness / (
                            1 / u_value_effective
                        )
                        break  # Modify only one layer for simplicity

                return new_construction_name

            # Collect windows to remove in a separate list
            windows_to_remove = []

            # Loop through each window in the IDF and apply thermal bridging conditions
            for window in self.idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
                # Identify windows to remove
                if "loft" in window.Name.lower():
                    windows_to_remove.append(window)
                    continue

                # Check junction types for thermal bridging
                reveal_adjacent = True  # By default, all windows are adjacent to walls
                sill_adjacent = is_window_adjacent_to(window, floors)
                corner_adjacent = is_window_adjacent_to(window, walls)

                # Calculate the total thermal bridging factor for the window
                total_bridge_factor = 0
                if reveal_adjacent:
                    total_bridge_factor += window_reveal_factor
                if sill_adjacent:
                    total_bridge_factor += window_sill_factor
                if corner_adjacent:
                    total_bridge_factor += corner_window_factor

                # Create a new construction for the window if there is a thermal
                # bridge factor
                # if total_bridge_factor > 0:
                #     new_construction_name = create_modified_window_construction(
                #         window, total_bridge_factor
                #     )
                #     if new_construction_name:
                #         window.Construction_Name = new_construction_name

            # Remove all collected windows (e.g., loft windows)
            for window in windows_to_remove:
                self.idf.removeidfobject(window)

        else:
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
        if self.building_config.zoning == bco.Zoning.CUSTOM.value:

            self.idf.intersect_match()

            # # Retrieve surfaces by type
            # floors = self.idf.getsurfaces("floor")
            # walls = self.idf.getsurfaces("wall")
            # roofs = self.idf.getsurfaces("roof")
            # ceilings = self.idf.getsurfaces("ceiling")

            # # Define the thermal bridging correction factors for different junctions
            # # Fraction is used as a hack...
            # wall_bridge_factor = (
            #     0.25 / 0.15
            # ) * self.building_config.thermal_bridging_coefficient
            # subfloor_bridge_factor = (
            #     0.30 / 0.15
            # ) * self.building_config.thermal_bridging_coefficient

            # # Function to check if a wall is external or party wall
            # def is_external_or_party_wall(wall):
            #     """Check if a wall is external or a party wall."""
            #     return wall.Outside_Boundary_Condition in [
            #         "Outdoors",
            #         "OtherSideConditionsModel",
            #         "Adiabatic",
            #     ]

            # # Function to check if a surface is a subfloor
            # # (e.g., in an unconditioned space)
            # def is_subfloor(floor):
            #     """Check if the floor is a subfloor
            #     (e.g., ground contact or unconditioned)."""
            #     return floor.Outside_Boundary_Condition in [
            #         "Ground",
            #         "OtherSideConditionsModel",
            #     ]

            # # Function to check if a surface is adjacent to a given set of surfaces
            # def is_surface_adjacent_to(surface, other_surfaces):
            #     """Check if a surface is adjacent to any of the given surfaces."""
            #     for other_surface in other_surfaces:
            #         for surface_vertex in surface.coords:
            #             for other_vertex in other_surface.coords:
            #                 if (
            #                     abs(surface_vertex[0] - other_vertex[0]) < 0.1
            #                     and abs(surface_vertex[1] - other_vertex[1]) < 0.1
            #                 ):
            #                     return True
            #     return False

            # # Function to create a modified construction with combined U-value
            # # for multiple factors
            # def create_modified_construction(construction_name, combined_bridge_factor):
            #     """Create a new construction with an effective U-value
            #     including thermal bridging."""
            #     construction = self.idf.getobject("CONSTRUCTION", construction_name)
            #     if not construction or "_ThermalBridge" in construction_name:
            #         return None

            #     # Calculate the base U-value from material layers
            #     total_thickness = sum(
            #         self.idf.getobject("MATERIAL", layer).Thickness
            #         for layer in construction.Material_Layers
            #         if self.idf.getobject("MATERIAL", layer)
            #     )
            #     u_value_base = sum(
            #         1 / (material.Thickness / material.Conductivity)
            #         for material in (
            #             self.idf.getobject("MATERIAL", layer)
            #             for layer in construction.Material_Layers
            #             if self.idf.getobject("MATERIAL", layer)
            #         )
            #     )

            #     # Calculate the effective U-value by adding the combined thermal
            #     # bridge factor
            #     u_value_effective = u_value_base + combined_bridge_factor

            #     # Create a new construction name with a suffix
            #     new_construction_name = (
            #         f"{construction_name}_ThermalBridge_{combined_bridge_factor:.2f}"
            #     )

            #     # Create a new construction object with adjusted U-value
            #     new_construction = self.idf.newidfobject(
            #         "CONSTRUCTION", Name=new_construction_name
            #     )
            #     new_construction.Material_Layers = construction.Material_Layers[:]

            #     # Adjust the first material's conductivity to achieve the
            #     # effective U-value
            #     for layer_name in new_construction.Material_Layers:
            #         material = self.idf.getobject("MATERIAL", layer_name)
            #         if material and total_thickness > 0:
            #             # Adjust conductivity to achieve the effective U-value
            #             material.Conductivity = total_thickness / (
            #                 1 / u_value_effective
            #             )
            #             break  # Modify only one layer for simplicity

            #     return new_construction_name

            # # Loop through floors and apply thermal bridging to combined junctions
            # for floor in floors:
            #     subfloor_adjacent = is_surface_adjacent_to(
            #         floor, [f for f in floors if is_subfloor(f)]
            #     )
            #     wall_adjacent = is_surface_adjacent_to(floor, walls)

            #     # Determine the total bridging factor
            #     total_bridge_factor = 0
            #     if subfloor_adjacent:
            #         total_bridge_factor += subfloor_bridge_factor
            #     if wall_adjacent:
            #         total_bridge_factor += wall_bridge_factor

            #     # Apply the combined thermal bridging factor if both conditions are met
            #     if total_bridge_factor > 0 and not is_subfloor(floor):
            #         new_construction_name = create_modified_construction(
            #             floor.Construction_Name, total_bridge_factor
            #         )
            #         if new_construction_name:
            #             floor.Construction_Name = new_construction_name

            # # Apply modified constructions to ceilings and roofs adjacent to walls
            # for ceiling in ceilings:
            #     if is_surface_adjacent_to(ceiling, walls):
            #         new_construction_name = create_modified_construction(
            #             ceiling.Construction_Name, wall_bridge_factor
            #         )
            #         if new_construction_name:
            #             ceiling.Construction_Name = new_construction_name

            # for roof in roofs:
            #     if is_surface_adjacent_to(roof, walls):
            #         new_construction_name = create_modified_construction(
            #             roof.Construction_Name, wall_bridge_factor
            #         )
            #         if new_construction_name:
            #             roof.Construction_Name = new_construction_name

            # # Apply smaller thermal bridging factor for walls adjacent
            # # to floors or ceilings
            # for wall in walls:
            #     if is_external_or_party_wall(wall):
            #         new_construction_name = create_modified_construction(
            #             wall.Construction_Name, combined_bridge_factor=0.10
            #         )
            #         if new_construction_name:
            #             wall.Construction_Name = new_construction_name

        else:
            for floor_surface in self.idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
                if (
                    floor_surface.Surface_Type == "floor"
                    and floor_surface.Zone_Name != "ROOF SPACE"
                ):
                    floor_zone_nr = int(floor_surface.Zone_Name.split()[-1])
                    if floor_zone_nr in range(
                        1, self.building_config.number_of_stories
                    ):
                        # find ceiling of zone below
                        for ceil_surface in self.idf.idfobjects[
                            "BUILDINGSURFACE:DETAILED"
                        ]:
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
