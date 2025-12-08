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
        self.idf.idfobjects["TIMESTEP"][0].Number_of_Timesteps_per_Hour = self.building_config.timesteps_per_hour
        self.idf.idfobjects["BUILDING"][0].Name = self.building_config.name
        self.idf.idfobjects["RUNPERIOD"][0].Begin_Year = self.building_config.year
        self.idf.idfobjects["RUNPERIOD"][0].End_Year = self.building_config.year
        self.idf.newidfobject(
            "ZoneAirHeatBalanceAlgorithm".upper(), Algorithm="ThirdOrderBackwardDifference"
        )

        self.idf.newidfobject("SURFACECONVECTIONALGORITHM:INSIDE", Algorithm="TARP")
        self.idf.newidfobject("SURFACECONVECTIONALGORITHM:OUTSIDE", Algorithm="DOE-2")
        self.idf.newidfobject(
            "HEATBALANCEALGORITHM",
            Algorithm="ConductionTransferFunction",
            Surface_Temperature_Upper_Limit = 2000,
            Minimum_Surface_Convection_Heat_Transfer_Coefficient_Value=1e-08,
            Maximum_Surface_Convection_Heat_Transfer_Coefficient_Value=1000
            )



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
                        # surface.Sun_Exposure = "NoSun"
                        # surface.Wind_Exposure = "NoWind"
                        surface.Sun_Exposure = "SunExposed"
                        surface.Wind_Exposure = "WindExposed"
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
                        # surface.Sun_Exposure = "NoSun"
                        # surface.Wind_Exposure = "NoWind"
                        surface.Sun_Exposure = "SunExposed"
                        surface.Wind_Exposure = "WindExposed"

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
                        # surface.Sun_Exposure = "NoSun"
                        # surface.Wind_Exposure = "NoWind"
                        surface.Sun_Exposure = "SunExposed"
                        surface.Wind_Exposure = "WindExposed"
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
                        surface.Outside_Boundary_Condition = "Ground"
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

            # --- At the very end of set_constructions ---
            # Post-pass: enforce adiabatic for neighbour-shared walls
            azimuth_tol = 5.0  # degrees tolerance

            for wall in self.idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
                if wall.Surface_Type.lower() == "wall" and wall.Outside_Boundary_Condition.lower() == "outdoors":
                    azi = wall.azimuth

                    # North neighbour
                    if self.building_config.distance_to_neighbour[0] == 0:
                        diff = (azi - 0 + 360) % 360
                        diff = min(diff, 360 - diff)  # wrap-around fix
                        if diff <= azimuth_tol:
                            wall.Outside_Boundary_Condition = "Adiabatic"
                            wall.Construction_Name = self.adiabatic_wall_construction.get_name()
                            wall.Sun_Exposure = "NoSun"
                            wall.Wind_Exposure = "NoWind"

                    # East neighbour
                    if self.building_config.distance_to_neighbour[1] == 0:
                        diff = (azi - 90 + 360) % 360
                        diff = min(diff, 360 - diff)
                        if diff <= azimuth_tol:
                            wall.Outside_Boundary_Condition = "Adiabatic"
                            wall.Construction_Name = self.adiabatic_wall_construction.get_name()
                            wall.Sun_Exposure = "NoSun"
                            wall.Wind_Exposure = "NoWind"

                    # South neighbour
                    if self.building_config.distance_to_neighbour[2] == 0:
                        diff = (azi - 180 + 360) % 360
                        diff = min(diff, 360 - diff)
                        if diff <= azimuth_tol:
                            wall.Outside_Boundary_Condition = "Adiabatic"
                            wall.Construction_Name = self.adiabatic_wall_construction.get_name()
                            wall.Sun_Exposure = "NoSun"
                            wall.Wind_Exposure = "NoWind"

                    # West neighbour
                    if self.building_config.distance_to_neighbour[3] == 0:
                        diff = (azi - 270 + 360) % 360
                        diff = min(diff, 360 - diff)
                        if diff <= azimuth_tol:
                            wall.Outside_Boundary_Condition = "Adiabatic"
                            wall.Construction_Name = self.adiabatic_wall_construction.get_name()
                            wall.Sun_Exposure = "NoSun"
                            wall.Wind_Exposure = "NoWind"


        else:
        # follow conventional approach laid out by Hannes
            for surface in self.idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
                if surface.Surface_Type.lower() == "wall":
                    if surface.Outside_Boundary_Condition.lower() == "zone":
                        surface.Construction_Name = (
                            self.partition_construction.get_name()
                        )
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


    def add_ground_temperatures(self):
        """adds monthly ground temperatures for ground-contact surfaces"""
        self.idf.newidfobject(
            "Site:GroundTemperature:BuildingSurface".upper(),
            January_Ground_Temperature=6.4,
            February_Ground_Temperature=5.9,
            March_Ground_Temperature=6.9,
            April_Ground_Temperature=8.9,
            May_Ground_Temperature=11.0,
            June_Ground_Temperature=13.5,
            July_Ground_Temperature=16.1,
            August_Ground_Temperature=16.0,
            September_Ground_Temperature=15.3,
            October_Ground_Temperature=13.5,
            November_Ground_Temperature=10.8,
            December_Ground_Temperature=7.6
        )
    def add_internal_mass(self, zone_areas):
        """adds internal thermal mass of partitions and furniture"""
        FURNITURE_TM_OVERRIDE = {
            "kitchen": 600,
            "front_room": 600,
            "backroom": 600,
            "hall_downstairs": 600,
            "bedroom_1": 60,
            "bedroom_2": 60,
            "bedroom_3": 60,
            "hall_upstairs": 600,
            "bathroom": 60
        }
        # FURNITURE_TM_OVERRIDE = {
        #     "kitchen": 60,
        #     "front_room": 60,
        #     "backroom": 60,
        #     "hall_downstairs": 60,
        #     "bedroom_1": 60,
        #     "bedroom_2": 60,
        #     "bedroom_3": 60,
        #     "hall_upstairs": 60,
        #     "bathroom": 60
        # }

        furniture_override = FURNITURE_TM_OVERRIDE

        for zone in self.get_conditioned_zones():
            za = zone_areas.get(zone.Name) if zone_areas else (
                self.building_config.length_wall_x * self.building_config.length_wall_y
            )

            # self.idf.newidfobject(
            #     "INTERNALMASS",
            #     Name=f"IntMass-Partitions-{zone.Name}",
            #     Construction_Name="InternalWall",
            #     Zone_or_ZoneList_Name=zone.Name,
            #     Surface_Area=za
            # )

            if self.furniture_construction:
                override_val = furniture_override.get(zone.Name.lower())
                if override_val is None or override_val <= 0:
                    continue

                furn_mat = self.furniture_construction.materials[0]
                furn_t = self.furniture_construction.thicknesses[0]

                tm_furniture = override_val / (furn_mat.rho * furn_mat.cp / 1000 * furn_t)

                self.idf.newidfobject(
                    "INTERNALMASS",
                    Name=f"IntMass-Furniture-{zone.Name}",
                    Construction_Name="Furniture",
                    Zone_or_ZoneList_Name=zone.Name,
                    Surface_Area=za * tm_furniture
                )

        for im in self.idf.idfobjects["INTERNALMASS"]:
            name = im.Construction_Name.lower()
            if name == "ceiling":
                im.Construction_Name = self.ceiling_construction.get_name()
            elif name == "floor":
                im.Construction_Name = self.upper_floor_construction.get_name()
            elif name == "internalwall":
                im.Construction_Name = self.partition_construction.get_name()
            elif name == "furniture":
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

        self.add_schedules()
        # self.add_people()
        self.idf = add_ventilation(
            self.idf, self.building_config, self.get_conditioned_zones()
        )

        self.add_internal_mass(zone_areas)
        self.add_zone_capacitance_multiplier()
        self.add_ground_temperatures()
        # self.add_internal_gains()

        self.add_environmental_impact_factors()
        self.set_design_days()

        if self.building_config.pv_present:
            self.idf = add_pv_and_battery(self.idf, self.building_config)

        # Unneeded now
        # self.add_beizaee_gains()
        # self.add_infiltration()
        # HACK
        self.idf.translate([0, 0, 0.6])
        self.set_boundary_conditions()
        self.idf.translate([0, 0, -0.6])
        # self.add_windows()
        self.set_constructions()
        self.idf.translate([0, 0, 0.6])
        self.add_openings()
        self.add_airflow_network()

        # self.add_neighbours()

        self.idf = add_heating_system(
            self.idf, self.building_config, self.get_conditioned_zones()
        )

        return self.idf

    def add_airflow_network(self):
        self.idf = add_airflow_network(self.idf, self.building_config)




    def add_openings(self):
        openings = getattr(self.building_config, "openings", [])
        if not openings:
            print("No openings defined in building configuration.")
            return

        type_props = {
            "internal_door": {"surface_type": "Door", "get_construction": lambda: self.partition_door_construction.get_name(), "suffix": "door", "aspect": 0.45},
            "external_door": {"surface_type": "Door", "get_construction": lambda: self.external_door_construction.get_name(), "suffix": "extdoor", "aspect": 0.45},
            "window": {"surface_type": "Window", "get_construction": lambda: "Single Glazing", "suffix": "window", "aspect": 1.2},
            "vent": {"surface_type": "Door", "get_construction": lambda: self._ensure_air_brick_construction(), "suffix": "vent", "aspect": 5.0},
            "hole": {"surface_type": "Door", "get_construction": lambda: self.hole_construction.get_name(), "suffix": "hole", "aspect": 1.0}
        }

        true_surfaces = [sf for sf in self.idf.idfobjects["BUILDINGSURFACE:DETAILED"] if sf.Outside_Boundary_Condition.lower() == "surface"]
        self.wall_registry = {}

        for opening in openings:
            opening_type = opening["type"]
            zones = [z.lower() for z in opening["zones"]]
            area = opening["area"]

            if opening_type not in type_props:
                print(f"⚠️ Unknown opening type: {opening_type}, skipping.")
                continue

            props = type_props[opening_type]
            construction_name = props["get_construction"]()
            surface_type = props["surface_type"]
            suffix = props["suffix"]
            aspect = props["aspect"]

            # --- EXTERNAL OPENINGS ---
            if "outdoors" in zones:
                zone = zones[0] if zones[1] == "outdoors" else zones[1]
                azimuth = opening.get("azimuth")
                if azimuth is None:
                    print(f"⚠️ External opening {opening_type} for {zone} missing azimuth, skipping.")
                    continue

                ext_walls = [w for w in self.idf.idfobjects["BUILDINGSURFACE:DETAILED"]
                            if w.Zone_Name.lower() == zone.lower()
                            and w.Outside_Boundary_Condition.lower() == "outdoors"
                            and "wall" in w.Surface_Type.lower()]
                if not ext_walls:
                    print(f"⚠️ No external wall found for {zone}, skipping {opening_type}.")
                    continue

                wall = min(ext_walls, key=lambda w: abs((float(w.azimuth) - azimuth + 180) % 360 - 180))
                opening_coords = self._make_rectangular_opening(wall, area, aspect, opening_type)
                if opening_coords is None:
                    print(f"⚠️ Could not place {opening_type} on {wall.Name}, skipping.")
                    continue

                self._add_fenestration(wall, None, opening_coords, suffix, construction_name, surface_type)
                print(f"✅ Added {opening_type} to {zone} on {wall.Name} (az={azimuth}°, {area:.2f} m²)")
                continue

            # --- INTERNAL OPENINGS ---
            matched_pair = self._find_surface_pair(true_surfaces, zones, opening_type)
            if not matched_pair:
                print(f"⚠️ No matching surface pair for {opening_type} between {zones}")
                continue

            sf_from, sf_to = matched_pair
            opening_coords = self._make_rectangular_opening(sf_from, area, aspect, opening_type)
            if opening_coords is None:
                print(f"⚠️ Could not place {opening_type} between {zones[0]} ↔ {zones[1]}, skipping.")
                continue

            self._add_fenestration(sf_from, sf_to, opening_coords, suffix, construction_name, surface_type)
            print(f"✅ Added {opening_type} between {zones[0]} ↔ {zones[1]} ({area:.2f} m²)")


    def _make_rectangular_opening(self, wall, area, aspect, otype):
        import numpy as np, math
        from shapely.geometry import Polygon

        verts = np.array(wall.coords)
        p1, p2, p3 = verts[0], verts[1], verts[2]
        n = np.cross(p2 - p1, p3 - p1); n /= np.linalg.norm(n)
        u = (p2 - p1); u /= np.linalg.norm(u)
        v = np.cross(n, u)

        proj = [(np.dot(vt - p1, u), np.dot(vt - p1, v)) for vt in verts]
        wall_poly = Polygon(proj)
        if not wall_poly.is_valid: wall_poly = wall_poly.buffer(0)
        wall_area = float(getattr(wall, "area", wall_poly.area))

        rect_w = math.sqrt(area / aspect)
        rect_h = rect_w * aspect
        margin = 0.02 * math.sqrt(wall_area)
        inset_poly = wall_poly.buffer(-margin)
        if inset_poly.is_empty: inset_poly = wall_poly

        minx, miny, maxx, maxy = inset_poly.bounds
        step_x = rect_w * 0.2
        step_y = rect_h * 0.2

        def try_place(rw, rh):
            for x0 in np.arange(minx, maxx - rw, step_x):
                for y0 in np.arange(miny, maxy - rh, step_y):
                    rect = Polygon([(x0, y0), (x0 + rw, y0), (x0 + rw, y0 + rh), (x0, y0 + rh)])
                    if inset_poly.contains(rect):
                        coords3d = [p1 + u * x + v * y - n * 1e-4 for (x, y) in rect.exterior.coords[:-1]]
                        return np.round(np.array(coords3d), 4), rw, rh
            return None, None, None

        coords3d, rw, rh = try_place(rect_w, rect_h)
        if coords3d is None:
            for shrink in np.linspace(0.95, 0.5, 10):
                coords3d, rw, rh = try_place(rect_w * shrink, rect_h * shrink)
                if coords3d is not None:
                    print(f"🪟 {otype.upper()} | {wall.Name}: auto-fit scaled to {shrink:.2f}× ({rw:.2f}×{rh:.2f} m, {rw*rh:.2f} m²) from requested {area:.2f} m²")
                    return coords3d
            maxx, maxy = inset_poly.bounds[2], inset_poly.bounds[3]
            usable_w = maxx - minx
            usable_h = maxy - miny
            coords3d, rw, rh = try_place(usable_w * 0.95, usable_h * 0.95)
            if coords3d is not None:
                print(f"🪟 {otype.upper()} | {wall.Name}: fallback used max available space ({rw:.2f}×{rh:.2f} m, {rw*rh:.2f} m²)")
                return coords3d
            print(f"⚠️ {otype.upper()} | {wall.Name}: failed even with fallback, placing minimal patch (0.2×0.2 m²)")
            coords3d, rw, rh = try_place(0.2, 0.2)
        if coords3d is not None:
            print(f"🪟 {otype.upper()} | {wall.Name}: placed ({rw:.2f}×{rh:.2f} m, {rw*rh:.2f} m²) vs requested {area:.2f} m² | wall_area={wall_area:.2f} m²")
        return coords3d





    def _find_surface_pair(self, true_surfaces, zones, opening_type):
        """Find matching surface pair for internal opening."""
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
                    return (sf, opp_sf)
            elif sf.Zone_Name.lower() == zones[1]:
                opp = sf.Outside_Boundary_Condition_Object
                if not opp:
                    continue
                opp_sf = next(
                    (s for s in true_surfaces if s.Name == opp and s.Zone_Name.lower() == zones[0]),
                    None
                )
                if opp_sf:
                    return (sf, opp_sf)
        return None


    def _add_fenestration(self, surf_from, surf_to, door_coords, suffix, construction_name, surface_type):
        """Helper to add fenestration for a matched surface pair or external wall."""
        name_from = f"{surf_from.Name}_{suffix}"

        self.idf.newidfobject(
            "FENESTRATIONSURFACE:DETAILED",
            Name=name_from,
            Surface_Type=surface_type,
            Construction_Name=construction_name,
            Building_Surface_Name=surf_from.Name,
            Outside_Boundary_Condition_Object="" if surf_to is None else f"{surf_to.Name}_{suffix}",
            View_Factor_to_Ground="AutoCalculate",
            Multiplier=1,
            Number_of_Vertices=len(door_coords),
            **{f"Vertex_{i+1}_Xcoordinate": door_coords[i][0] for i in range(len(door_coords))},
            **{f"Vertex_{i+1}_Ycoordinate": door_coords[i][1] for i in range(len(door_coords))},
            **{f"Vertex_{i+1}_Zcoordinate": door_coords[i][2] for i in range(len(door_coords))},
        )

        if surf_to is None:
            return

        door_coords_reversed = door_coords[::-1]
        name_to = f"{surf_to.Name}_{suffix}"

        self.idf.newidfobject(
            "FENESTRATIONSURFACE:DETAILED",
            Name=name_to,
            Surface_Type=surface_type,
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


    def _ensure_air_brick_construction(self):
        """Ensure air brick construction exists, create if missing."""
        cons_name = "AFN_AirBrickConstruction"
        mat_name = "AFN_AirBrickPanel"

        if not any(m.Name == mat_name for m in self.idf.idfobjects.get("MATERIAL:NOMASS", [])):
            self.idf.newidfobject(
                "MATERIAL:NOMASS",
                Name=mat_name,
                Roughness="Rough",
                Thermal_Resistance=0.1,
            )

        if not any(c.Name == cons_name for c in self.idf.idfobjects.get("CONSTRUCTION", [])):
            self.idf.newidfobject(
                "CONSTRUCTION",
                Name=cons_name,
                Outside_Layer=mat_name,
            )

        return cons_name

    def add_windows(self):
        """Method which adds window strips into IDF"""

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
            # --- Remove any windows in Subfloor zone walls ---
            to_remove = []
            for win in self.idf.idfobjects.get("FENESTRATIONSURFACE:DETAILED", []):
                parent = win.Building_Surface_Name
                parent_surface = next(
                    (s for s in self.idf.idfobjects["BUILDINGSURFACE:DETAILED"] if s.Name == parent),
                    None
                )
                if parent_surface and parent_surface.Zone_Name.lower() == "subfloor":
                    to_remove.append(win)

            for win in to_remove:
                self.idf.removeidfobject(win)

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
