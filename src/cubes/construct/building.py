"""Defines the Building class """

from cubes.construct.constants import (
    EPLUS_PATH,
    MATERIALS,
    SIMPLE_GLAZINGS,
    get_schedule,
)
from cubes.construct import material as mat
from cubes.construct import utilities

from geomeppy import IDF
from eppy import idf_helpers


class Building:

    """This class holds all the information and methods to produce an IDF file"""

    def __init__(self, building_config):
        """This constructor is for with a BuildingConfig object

        Args:
            building_config (buildingconfig object): this is an instance of the
                                                     buildingconfig dataclass
        """
        self.building_config = building_config

        self.wall_construction = mat.Construction(
            "Wall",
            [MATERIALS[x] for x in building_config.wall_layer_materials],
            building_config.wall_layer_thickness,
        )
        self.ground_floor_construction = mat.Construction(
            "GroundFloor",
            [MATERIALS[x] for x in building_config.ground_floor_layer_materials],
            building_config.ground_floor_layer_thickness,
        )
        self.roof_construction = mat.Construction(
            "Roof",
            [MATERIALS[x] for x in building_config.roof_layer_materials],
            building_config.roof_layer_thickness,
        )
        self.upper_floor_construction = mat.Construction(
            "Floor",
            [MATERIALS[x] for x in building_config.upper_floor_layer_materials],
            building_config.upper_floor_layer_thickness,
        )
        self.ceiling_construction = mat.Construction(
            "Ceiling",
            [MATERIALS[x] for x in building_config.upper_floor_layer_materials[::-1]],
            building_config.upper_floor_layer_thickness[::-1],
        )
        self.partition_construction = mat.Construction(
            "InternalMass",
            [MATERIALS[x] for x in building_config.partition_layer_materials[::-1]],
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
                [MATERIALS[x] for x in building_config.attic_floor_layer_materials],
                building_config.attic_floor_layer_thickness,
            )
            self.all_constructions.append(self.last_floor_construction)
            self.last_ceiling_construction = mat.Construction(
                "Last ceiling",
                [
                    MATERIALS[x]
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
            elif self.building_config.window_layer_materials:
                self.window_system_simple = SIMPLE_GLAZINGS[
                    self.building_config.window_layer_materials[0]
                ]
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
                    > self.building_config.h_storey * self.building_config.n_storey
                    - 0.1
                ):
                    surface.Construction_Name = self.last_floor_construction.get_name()
                else:
                    surface.Construction_Name = self.upper_floor_construction.get_name()
            elif surface.Surface_Type == "ceiling":
                if (
                    self.building_config.roof_type != "flat"
                    and surface.Vertex_1_Zcoordinate
                    > self.building_config.h_storey * self.building_config.n_storey
                    - 0.1
                ):
                    surface.Construction_Name = (
                        self.last_ceiling_construction.get_name()
                    )

                else:
                    surface.Construction_Name = self.ceiling_construction.get_name()

        # windows
        if self.building_config.window_type != "Simple":
            self.idf = self.window_construction.add_to_idf(self.idf)
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

    def add_heating_system(self):
        """Adds in thermostats for each zone and"""

        # heating_system = self.building_config.heating_system_type.str.split()
        # template = heating_system.values[0][0]
        # boiler_type = heating_system.values[0][1]
        # heating_system = self.building_config.heating_system_type.str.split()
        # template = heating_system.values[0][0]
        # boiler_type = heating_system.values[0][1]

        if self.building_config.heating_system_type == "Water to air heat pump":
            for zone in self.get_heated_zones():

                stat = self.idf.newidfobject(
                    "HVACTEMPLATE:THERMOSTAT",
                    Name="Thermostat-" + zone.Name,
                    Heating_Setpoint_Schedule_Name="Heating-Setpoint-Schedule",
                    Cooling_Setpoint_Schedule_Name="Cooling-Setpoint-Schedule",
                )

                self.idf.newidfobject(
                    "HVACTEMPLATE:ZONE:WATERTOAIRHEATPUMP",
                    Zone_Name=zone.Name,
                    Template_Thermostat_Name=stat.Name,
                    Cooling_Supply_Air_Flow_Rate="autosize",
                    Heating_Supply_Air_Flow_Rate="autosize",
                    Zone_Heating_Sizing_Factor=1.2,
                    Zone_Cooling_Sizing_Factor=1.2,
                    Supply_Fan_Placement="DrawThrough",
                    Supply_Fan_Total_Efficiency=0.7,
                    Supply_Fan_Delta_Pressure=75,
                    Supply_Fan_Motor_Efficiency=0.9,
                    Cooling_Coil_Type="Coil:Cooling:WaterToAirHeatPump:EquationFit",
                    Cooling_Coil_Gross_Rated_Total_Capacity="autosize",
                    Cooling_Coil_Gross_Rated_Sensible_Heat_Ratio="autosize",
                    Cooling_Coil_Gross_Rated_COP=(
                        self.building_config.cooling_system_efficiency
                    ),
                    Heat_Pump_Heating_Coil_Type=(
                        "Coil:Heating:WaterToAirHeatPump:EquationFit"
                    ),
                    Heat_Pump_Heating_Coil_Gross_Rated_Capacity="autosize",
                    Heat_Pump_Heating_Coil_Gross_Rated_COP=(
                        self.building_config.heating_system_efficiency
                    ),
                    Supplemental_Heating_Coil_Capacity="autosize",
                    Maximum_Cycling_Rate=2.5,
                    Heat_Pump_Time_Constant=60,
                    Fraction_of_OnCycle_Power_Use=0.01,
                    Heat_Pump_Fan_Delay_Time=60,
                    Supplemental_Heating_Coil_Type="Electric",
                    Zone_Cooling_Design_Supply_Air_Temperature_Input_Method=(
                        "SupplyAirTemperature"
                    ),
                    Zone_Cooling_Design_Supply_Air_Temperature=12.5,
                    Zone_Heating_Design_Supply_Air_Temperature_Input_Method=(
                        "SupplyAirTemperature"
                    ),
                    Zone_Heating_Design_Supply_Air_Temperature=50.0,
                )

            self.idf.newidfobject(
                "HVACTEMPLATE:PLANT:MIXEDWATERLOOP",
                Name="Only Water Loop",
                Pump_Control_Type="Intermittent",
                Operation_Scheme_Type="Default",
                High_Temperature_Design_Setpoint=34,
                Low_Temperature_Design_Setpoint=20,
                Water_Pump_Configuration="ConstantFlow",
                Water_Pump_Rated_Head=179352,
                Water_Pump_Type="SinglePump",
                Supply_Side_Bypass_Pipe="Yes",
                Demand_Side_Bypass_Pipe="Yes",
                Fluid_Type="Water",
                Loop_Design_Delta_Temperature=6,
                Load_Distribution_Scheme="SequentialLoad",
            )

            self.idf.newidfobject(
                "HVACTEMPLATE:PLANT:TOWER",
                Name="Main Tower",
                Tower_Type="SingleSpeed",
                High_Speed_Nominal_Capacity="autosize",
                High_Speed_Fan_Power="autosize",
                Low_Speed_Nominal_Capacity="autosize",
                Low_Speed_Fan_Power="autosize",
                Free_Convection_Capacity="autosize",
                Priority=1,
                Sizing_Factor=1.2,
            )

            self.idf.newidfobject(
                "HVACTEMPLATE:PLANT:BOILER",
                Name="Main Boiler",
                Boiler_Type="HotWaterBoiler",
                Capacity="autosize",
                Efficiency=0.95,
                Fuel_Type="Electricity",
                Priority=1,
                Sizing_Factor=1.2,
                Minimum_Part_Load_Ratio=0.1,
                Maximum_Part_Load_Ratio=1.1,
                Optimum_Part_Load_Ratio=0.9,
                Water_Outlet_Upper_Temperature_Limit=99.9,
            )

        else:
            for zone in self.get_heated_zones():

                stat = self.idf.newidfobject(
                    "HVACTEMPLATE:THERMOSTAT",
                    Name="Thermostat-" + zone.Name,
                    Heating_Setpoint_Schedule_Name="Heating-Setpoint-Schedule",
                    Cooling_Setpoint_Schedule_Name="Cooling-Setpoint-Schedule",
                )
                self.idf.newidfobject(
                    "HVACTEMPLATE:ZONE:BASEBOARDHEAT",
                    Zone_Name=zone.Name,
                    Baseboard_Heating_Type="HotWater",
                    Template_Thermostat_Name=stat.Name,
                )
            self.idf.newidfobject(
                "HVACTEMPLATE:PLANT:HOTWATERLOOP", Name="Hot Water Loop"
            )
            self.idf.newidfobject(
                "HVACTEMPLATE:PLANT:BOILER",
                Name="Main Boiler",
                Boiler_Type="CondensingHotWaterBoiler",
                # self.building_config.heating_system_type,
                Efficiency=0.8,
                Fuel_Type="NaturalGas",
            )

        self.idf.idfobjects["SIMULATIONCONTROL"][0].Do_Zone_Sizing_Calculation = "Yes"

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

        # windows
        if self.building_config.window_opening_schedule:
            self.idf.newidfobject(
                "SCHEDULE:COMPACT",
                Name="Window-Opening-Schedule",
                Field_1=get_schedule(self.building_config.window_opening_schedule),
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
        for zone in self.get_heated_zones():
            self.idf.newidfobject(
                "ZONEVENTILATION:DESIGNFLOWRATE",
                Name=zone.Name + "-Ventilation",
                Zone_or_ZoneList_Name=zone.Name,
                Schedule_Name="People-Schedule",
                Design_Flow_Rate_Calculation_Method=(
                    self.building_config.ventilation_for_air_calculation_method
                ),
                Design_Flow_Rate=self.building_config.ventilation_for_air_rate,
                Flow_Rate_per_Zone_Floor_Area=(
                    self.building_config.ventilation_for_air_rate
                ),
                Flow_Rate_per_Person=self.building_config.ventilation_for_air_rate,
                Air_Changes_per_Hour=self.building_config.ventilation_for_air_rate,
                Ventilation_Type="Balanced",
                Fan_Pressure_Rise=(
                    self.building_config.ventilation_for_air_fan_pressure_rise
                ),
                Fan_Total_Efficiency=(
                    self.building_config.ventilation_for_air_fan_efficiency
                ),
            )

            self.idf.newidfobject(
                "ZONEVENTILATION:DESIGNFLOWRATE",
                Name=zone.Name + "-Cooling Ventilation",
                Zone_or_ZoneList_Name=zone.Name,
                Schedule_Name="People-Schedule",
                Design_Flow_Rate_Calculation_Method=(
                    self.building_config.natvent_for_cooling_calculation_method
                ),
                Design_Flow_Rate=(self.building_config.natvent_for_cooling_rate),
                Flow_Rate_per_Zone_Floor_Area=(
                    self.building_config.natvent_for_cooling_rate
                ),
                Flow_Rate_per_Person=(self.building_config.natvent_for_cooling_rate),
                Air_Changes_per_Hour=(self.building_config.natvent_for_cooling_rate),
                Ventilation_Type="Natural",
                Fan_Pressure_Rise=0,
                Fan_Total_Efficiency=1,
                Constant_Term_Coefficient=1,
                Temperature_Term_Coefficient=0,
                Velocity_Term_Coefficient=0,
                Velocity_Squared_Term_Coefficient=0,
                Minimum_Indoor_Temperature=(
                    self.building_config.natvent_for_cooling_indoor_t_range[0]
                ),
                Minimum_Indoor_Temperature_Schedule_Name="",
                Maximum_Indoor_Temperature=(
                    self.building_config.natvent_for_cooling_indoor_t_range[1]
                ),
                Maximum_Indoor_Temperature_Schedule_Name="",
                Delta_Temperature=1,
            )

        if self.building_config.window_opening_schedule:
            window_count = 0
            for window in self.idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
                window_count += 1
                zone_name = idf_helpers.name2idfobject(
                    self.idf, Name=window.Building_Surface_Name
                ).Zone_Name
                self.idf.newidfobject(
                    "ZONEVENTILATION:WINDANDSTACKOPENAREA",
                    Name=zone_name + "-Open Windows" + str(window_count),
                    Zone_Name=zone_name,
                    Opening_Area=utilities.get_surface_area(window),
                    Opening_Area_Fraction_Schedule_Name="Window-Opening-Schedule",
                    Opening_Effectiveness="Autocalculate",
                    Effective_Angle=(
                        (
                            utilities.get_surface_orientation(window)
                            + self.building_config.rotation
                        )
                        % 360
                    ),
                    Height_Difference=abs(
                        utilities.get_surface_vertical_midpoint(window)
                        - (
                            self.building_config.n_storey
                            * self.building_config.h_storey
                            + self.building_config.h_roof
                        )
                        / 2.0
                    ),
                    Discharge_Coefficient_for_Opening="Autocalculate",
                )

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
                (self.building_config.l_wall_x, 0),
                (self.building_config.l_wall_x, self.building_config.l_wall_y),
                (0, self.building_config.l_wall_y),
                (0, 0),
            ],
            height=self.building_config.n_storey * self.building_config.h_storey,
            num_stories=self.building_config.n_storey,
        )

        # set rotation
        self.idf.idfobjects["BUILDING"][0].North_Axis = self.building_config.rotation

        self.idf.intersect_match()
        self.add_roof()
        self.add_windows()
        self.set_boundary_conditions()
        self.add_neighbours()
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

                    # search for zone name of last storey
                    last_storey_zone_name = "UNKNOWN"
                    for zone in self.idf.idfobjects["ZONE"]:
                        if str(self.building_config.n_storey - 1) in zone.Name:
                            last_storey_zone_name = zone.Name

                    ceiling_name = (
                        "storey " + str(self.building_config.n_storey) + " ceiling"
                    )
                    self.idf.newidfobject(
                        "BUILDINGSURFACE:DETAILED",
                        Name=ceiling_name,
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
                        Outside_Boundary_Condition="Surface",
                        Outside_Boundary_Condition_Object="attic floor",
                        Sun_Exposure="NoSun",
                        Wind_Exposure="NoWind",
                    )

                    self.idf.newidfobject(
                        "BUILDINGSURFACE:DETAILED",
                        Name="attic floor",
                        Surface_Type="floor",
                        Zone_Name="ROOF SPACE",
                        Vertex_1_Xcoordinate=surface.Vertex_1_Xcoordinate,
                        Vertex_1_Ycoordinate=surface.Vertex_1_Ycoordinate,
                        Vertex_1_Zcoordinate=surface.Vertex_1_Zcoordinate,
                        Vertex_2_Xcoordinate=surface.Vertex_4_Xcoordinate,
                        Vertex_2_Ycoordinate=surface.Vertex_4_Ycoordinate,
                        Vertex_2_Zcoordinate=surface.Vertex_4_Zcoordinate,
                        Vertex_3_Xcoordinate=surface.Vertex_3_Xcoordinate,
                        Vertex_3_Ycoordinate=surface.Vertex_3_Ycoordinate,
                        Vertex_3_Zcoordinate=surface.Vertex_3_Zcoordinate,
                        Vertex_4_Xcoordinate=surface.Vertex_2_Xcoordinate,
                        Vertex_4_Ycoordinate=surface.Vertex_2_Ycoordinate,
                        Vertex_4_Zcoordinate=surface.Vertex_2_Zcoordinate,
                        Outside_Boundary_Condition="Surface",
                        Outside_Boundary_Condition_Object=ceiling_name,
                        Sun_Exposure="NoSun",
                        Wind_Exposure="NoWind",
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
                Outside_Boundary_Condition="Outdoors",
            )

            self.idf.newidfobject(
                "BUILDINGSURFACE:DETAILED",
                Name="roof_2_2",
                Construction_Name="ROOF-Construction",
                Surface_Type="ROOF",
                Zone_Name="ROOF SPACE",
                Outside_Boundary_Condition="Outdoors",
            )

            self.idf.newidfobject(
                "BUILDINGSURFACE:DETAILED",
                Name="wall_1_2",
                Construction_Name="WALL-Construction",
                Surface_Type="WALL",
                Zone_Name="ROOF SPACE",
                Outside_Boundary_Condition="Outdoors",
                Number_of_Vertices=3,
            )

            self.idf.newidfobject(
                "BUILDINGSURFACE:DETAILED",
                Name="wall_2_2",
                Construction_Name="WALL-Construction",
                Surface_Type="WALL",
                Zone_Name="ROOF SPACE",
                Outside_Boundary_Condition="Outdoors",
                Number_of_Vertices=3,
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

                        count = count + 1

    def set_boundary_conditions(self):

        for floor_surface in self.idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
            if (
                floor_surface.Surface_Type == "floor"
                and floor_surface.Zone_Name != "ROOF SPACE"
            ):
                floor_zone_nr = int(floor_surface.Zone_Name.split()[-1])
                if floor_zone_nr in range(1, self.building_config.n_storey):
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
                    -1e-4 + self.building_config.l_wall_y,
                    1e-4 + self.building_config.l_wall_y,
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
                    -1e-4 + self.building_config.l_wall_x,
                    1e-4 + self.building_config.l_wall_x,
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
        lx = self.building_config.l_wall_x
        ly = self.building_config.l_wall_y
        h = (
            self.building_config.h_storey * self.building_config.n_storey
            + self.building_config.h_roof
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
            self.building_config.l_wall_x
            * self.building_config.l_wall_y
            * self.building_config.n_storey
        )

    def get_idf(self):
        return self.idf
