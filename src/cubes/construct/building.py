'''Defines the Building class '''

class Building:
    def __init__(self,geometry_data,systems_data):
        self.geometry_data = geometry_data
        self.system_data = systems_data

        self.a_ground_floor = self.geometry_data.loc[0]["REFERENCE BUILDING GROUND FLOOR AREA (m2)"]
        self.a_wall = self.geometry_data.loc[0]["REFERENCE BUILDING WALL AREA (m2)"]
        self.n_storey = int(self.geometry_data.loc[0]["NUMBER OF REFERENCE BUILDING STOREYS"])
        self.a_window = self.geometry_data.loc[0]["REFERENCE BUILDING WINDOW AREA (m2)"]
        self.r_floor_roof = self.geometry_data.loc[0]["Floor/Roof_ratio"]
        
        self.h_ceiling = 2.5 #tabula default for all buildings
        self.l_wall_front, self.l_wall_side = self.calc_wall_length()
        self.h_roof = self.get_roof_height(self.l_wall_side)

        self.idf = IDF('exp/jack/Data/Minimal.idf')
        #Future - Will need to automatically add in weather file based on locations
        self.idf.epw = "exp/jack/Data/USA_CO_Golden-NREL.724666_TMY3.epw"

    

    def calc_wall_length(self):
        """Calculates wall length using formula from Ambience"""

        a_facade = self.a_wall+self.a_window
        l_walls = []

        l_walls.append((a_facade/(2*self.n_storey*self.h_ceiling)) +
                        np.sqrt(((a_facade/(2*self.n_storey*self.h_ceiling))
                                -4*self.a_ground_floor)/2))
        l_walls.append((a_facade/(2*self.n_storey*self.h_ceiling)) - 
                        np.sqrt(((a_facade/(2*self.n_storey*self.h_ceiling))
                                -4*self.a_ground_floor)/2))

        #Currently assuming the longer wall is the front facing wall
        #...though in future this could be changed depending on the archetype 
        # e.g. a terraced house may be the opposite, so an if statement is needed here

        l_wall_front = l_walls[0]
        l_wall_side = l_walls[1]

        return l_wall_front, l_wall_side
        
    def get_roof_height(self, l_wall_side):
        """Calculates the roof height provided the roof is split in two equal sized elements"""
        return (np.sqrt((l_wall_side**2)*((self.r_floor_roof**2)-1)))/2

    def get_roof_coordinates(self):
        """Determines roof coordinates based on an idealised pitched roof"""

        #Future improvement will need to deal with different aspect ratios and different roof shapes
        
        roof_coords = [[[self.l_wall_front, 0, self.n_storey*self.h_ceiling],
                        [self.l_wall_front, self.l_wall_side/2, self.n_storey*self.h_ceiling+self.h_roof],
                        [0,self.l_wall_side/2,self.n_storey*self.h_ceiling+self.h_roof],
                        [0,0,self.n_storey*self.h_ceiling]],
                        [[self.l_wall_front,self.l_wall_side,self.n_storey*self.h_ceiling],
                        [self.l_wall_front,self.l_wall_side/2,self.n_storey*self.h_ceiling+self.h_roof],
                        [0,self.l_wall_side/2,self.n_storey*self.h_ceiling+self.h_roof],
                        [0,self.l_wall_front,self.n_storey*self.h_ceiling]]]

        return roof_coords

    def get_roof_wall_coordinates(self):
        """Determines roof-level wall coordinates based on an idealised pitched roof"""
    
        wall_coords = [[[0, 0, self.n_storey*self.h_ceiling],
                        [0, self.l_wall_side, self.n_storey*self.h_ceiling],
                        [0,self.l_wall_side/2,self.n_storey*self.h_ceiling+self.h_roof],
                        [0,self.l_wall_side/2,self.n_storey*self.h_ceiling+self.h_roof]],
                        [[self.l_wall_front, 0, self.n_storey*self.h_ceiling],
                        [self.l_wall_front, self.l_wall_side, self.n_storey*self.h_ceiling],
                        [self.l_wall_front,self.l_wall_side/2,self.n_storey*self.h_ceiling+self.h_roof],
                        [self.l_wall_front,self.l_wall_side/2,self.n_storey*self.h_ceiling+self.h_roof]]]
    
        return wall_coords

    def add_roof(self):
        """Gets roof height, coordinates of roof and walls, then creates new roof and wall elements in e+
            then assigns coordinates of the new elements"""
        
        roof_coords = self.get_roof_coordinates()
        
        wall_coords = self.get_roof_wall_coordinates()
        
        
        roof_construction = self.idf.newidfobject('CONSTRUCTION',
                                            Name="REFERENCE ROOF",
                                            Outside_Layer = "DefaultMaterial")
        
        roof_zone = self.idf.newidfobject('ZONE',
                                    Name="Roof Space",)

        #May want to change nomenclature on naming new elements
        #Currently N_X means that there are X of the new elements, and N designates what element you are adding
            
        roof_1_2 = self.idf.newidfobject('BUILDINGSURFACE:DETAILED',
                                    Name='roof_1_2', 
                                    Construction_Name = "REFERENCE ROOF",
                                    Surface_Type = 'roof',
                                    Zone_Name='Roof Space')
        
        roof_2_2 = self.idf.newidfobject('BUILDINGSURFACE:DETAILED',
                                    Name='roof_2_2', 
                                    Construction_Name = "REFERENCE ROOF",
                                    Surface_Type = 'roof',
                                    Zone_Name='Roof Space')
        
        wall_1_2 = self.idf.newidfobject('BUILDINGSURFACE:DETAILED',
                                    Name='wall_1_2', 
                                    Construction_Name = "REFERENCE WALL",
                                    Surface_Type = 'wall',
                                    Zone_Name='Roof Space')
        
        wall_2_2 = self.idf.newidfobject('BUILDINGSURFACE:DETAILED',
                                    Name='wall_2_2', 
                                    Construction_Name = "REFERENCE WALL",
                                    Surface_Type = 'wall',
                                    Zone_Name='Roof Space')
        
        
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
                count = count+1

    def load_materials(self):
        """Reads the materials database, which contains all of the materials
            and their properties, used by Ambience and then stores them in e+"""
        
        #Should maybe move this into the constants module and have a separate construction_data dataframe?
        #Would then be split into two methods, one which creates the dictionary in the constants
        #and another which belongs in building.py to read them into e+?

        #Path for this needs to be properly defined in either location
        materials = pd.read_excel("../Data/Materials.xlsx")
        
        used_materials = {"Material":[self.geometry_data.loc[0]["REFERENCE BUILDING WALL MATERIAL"],
                                    self.geometry_data.loc[0]["REFERENCE BUILDING WALL INSULATION MATERIAL"],
                                    self.geometry_data.loc[0]["REFERENCE BUILDING ROOF MATERIAL"],
                                    self.geometry_data.loc[0]["REFERENCE BUILDING ROOF INSULATION MATERIAL"],
                                    self.geometry_data.loc[0]["REFERENCE BUILDING FLOOR MATERIAL"],
                                    self.geometry_data.loc[0]["REFERENCE BUILDING FLOOR INSULATION MATERIAL"]],
                        
                        "Element": ["Wall_0", 
                                    "Wall_1",
                                    "Roof_0", 
                                    "Roof_1",
                                    "Floor_0", 
                                    "Floor_1"],
                        
                        "Thickness": [self.geometry_data.loc[0]["REFERENCE BUILDING WALL MATERIAL THICKNESS (m)"],
                                    self.geometry_data.loc[0]["REFERENCE BUILDING WALL INSULATION MATERIAL THICKNESS (m)"],
                                    self.geometry_data.loc[0]["REFERENCE BUILDING ROOF MATERIAL THICKNESS (m)"],
                                    self.geometry_data.loc[0]["REFERENCE BUILDING ROOF INSULATION MATERIAL THICKNESS (m)"],
                                    self.geometry_data.loc[0]["REFERENCE BUILDING FLOOR MATERIAL THICKNESS (m)"],
                                    self.geometry_data.loc[0]["REFERENCE BUILDING FLOOR INSULATION MATERIAL THICKNESS (m)"]]
                        }

        for i, used_mat in enumerate(used_materials["Material"]):
            
            if used_materials["Thickness"][i] != 0:
                
                self.idf.newidfobject("MATERIAL", 
                                Name = used_mat+" "+used_materials["Element"][i],
                                Roughness= 'MediumRough',
                                Thickness= used_materials["Thickness"][i],
                                Conductivity= materials.loc[materials['Material']==used_mat].Thermal_Conductivity,
                                Density= materials.loc[materials['Material']==used_mat].Density,
                                Specific_Heat= materials.loc[materials['Material']==used_mat].Specific_Heat_Capacity,
                                )

    def determine_buildup(self):
        """Loads buildup and buildup thicknesses from Ambience then cleans the buildup if the thickness is equal to 0"""

        build_up = {"Wall":[self.geometry_data.loc[0]["REFERENCE BUILDING WALL MATERIAL"],
                                self.geometry_data.loc[0]["REFERENCE BUILDING WALL INSULATION MATERIAL"]],
                        "Roof":[self.geometry_data.loc[0]["REFERENCE BUILDING ROOF MATERIAL"],
                                self.geometry_data.loc[0]["REFERENCE BUILDING ROOF INSULATION MATERIAL"]],
                        "Floor":[self.geometry_data.loc[0]["REFERENCE BUILDING FLOOR MATERIAL"],
                                self.geometry_data.loc[0]["REFERENCE BUILDING FLOOR INSULATION MATERIAL"]]}

        build_up_thickness = {"Wall": [self.geometry_data.loc[0]["REFERENCE BUILDING WALL MATERIAL THICKNESS (m)"],
                                    self.geometry_data.loc[0]["REFERENCE BUILDING WALL INSULATION MATERIAL THICKNESS (m)"]],
                            "Roof": [self.geometry_data.loc[0]["REFERENCE BUILDING WALL MATERIAL THICKNESS (m)"],
                                    self.geometry_data.loc[0]["REFERENCE BUILDING WALL INSULATION MATERIAL THICKNESS (m)"]],
                            "Floor": [self.geometry_data.loc[0]["REFERENCE BUILDING WALL MATERIAL THICKNESS (m)"],
                                    self.geometry_data.loc[0]["REFERENCE BUILDING WALL INSULATION MATERIAL THICKNESS (m)"]]}

        for element in build_up_thickness:
            for i, layer in enumerate(build_up_thickness[element]):
                if layer == 0:
                    build_up[element].remove(i)
            
        return build_up

    def set_constructions(self):
        """Loads materials used in building into e+, loads those materials into a buildup for each construction element
            then assigns each the buildups as construction objects in e+"""

        self.load_materials()
        build_up = self.determine_buildup() 
        
        construction_name = ["REFERENCE WALL", 
                            "REFERENCE ROOF",
                            "REFERENCE FLOOR"]
        
        for i, element in enumerate(build_up):
            
            if len(build_up[element]) > 1:
                
                self.idf.newidfobject(
                    "CONSTRUCTION",
                    Name=construction_name[i],
                    Outside_Layer= build_up[element][0],
                    Layer_2 = build_up[element][1])
                                                
            else:
                self.idf.newidfobject(
                    "CONSTRUCTION",
                    Name=construction_name[i],
                    Outside_Layer= build_up[element][0])                                 

        for surface in self.idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
            if surface.Surface_Type =="wall":
                surface.Construction_Name = "REFERENCE WALL"
            if surface.Surface_Type =="roof":
                surface.Construction_Name = "REFERENCE ROOF"
            if surface.Surface_Type =="floor":
                surface.Construction_Name = "REFERENCE FLOOR"

    def get_system_data(self):
        """Reads the sampled building system information and puts it in a format for e+ to read"""
    
        #Future - will need to model more energy systems other than boilers e.g. heat pumps, stoves, electrical heater etc.
        #Future - will need to model biomass and double check if Solid and Liquid fuel in Ambience is actually coal and Diesal etc
        
        efficiency = self.system_data["HEATING SYSTEM 1 EFFICIENCY"]
        
        if self.system_data["HEATING SYSTEM 1 TECHNOLOGY"].str.contains("boiler").values[0]:
        
            if self.system_data["HEATING SYSTEM 1 TECHNOLOGY"].str.contains("non-condensing").values[0]:

                technology = "HotWaterBoiler"
            else:
                technology = "CondensingHotWaterBoiler"

            if self.system_data["HEATING SYSTEM 1 FUEL USED"].values[0] == "Gas":
                fuel = "NaturalGas"
            if self.system_data["HEATING SYSTEM 1 FUEL USED"].values[0] == "Liquid":
                fuel = "Diesel" #Double check
            if self.system_data["HEATING SYSTEM 1 FUEL USED"].values[0] == "Electricity":
                fuel = "Electricity"
            if self.system_data["HEATING SYSTEM 1 FUEL USED"].values[0] == "Biomass":
                fuel = "Coal" #Double check
            if self.system_data["HEATING SYSTEM 1 FUEL USED"].values[0] == "Solid":
                fuel = "Coal" #Double check
                
        return technology, efficiency, fuel
    
    def add_heating_system(self):
        """Gets the system data from get_system_data method and then adds in a heating system"""
        
        #Future - will need to use templates for other energy systems
        
        tech, efficiency, fuel = self.get_system_data()
            
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
            Efficiency=efficiency,
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
                Name=zone.Name+"-People",
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
                Name=zone.Name+"-Ventilation",
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
                Name=zone.Name+"-Infiltration",
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
                Name=zone.Name+"-Lights",
                Zone_or_ZoneList_Name=zone.Name,
                Schedule_Name="Always-Schedule",
                Design_Level_Calculation_Method="Watts/area",
                Watts_per_Zone_Floor_Area=1,
            )

            self.idf.newidfobject(
                "ELECTRICEQUIPMENT",
                Name=zone.Name+"-Equipment",
                Zone_or_ZoneList_Name=zone.Name,
                Schedule_Name="Always-Schedule",
                Design_Level_Calculation_Method="Watts/area",
                Watts_per_Zone_Floor_Area=5,
            )
    
    def build(self):
        """Creates the building"""
        #May want to change the block name - again nomenclature
        self.idf.add_block(name='Living',
                            coordinates=[(self.l_wall_front,0),
                                        (self.l_wall_front,self.l_wall_side),
                                        (0,self.l_wall_side),
                                        (0,0)],
                            height=self.n_storey*self.h_ceiling,
                            num_stories = self.n_storey)
        
        self.idf.set_default_constructions()
        
        if self.r_floor_roof != 1:
            
            for index,surface in enumerate(self.idf.idfobjects['BUILDINGSURFACE:DETAILED']):
                if surface.Surface_Type == "roof":
                    self.idf.removeidfobject(self.idf.idfobjects['BUILDINGSURFACE:DETAILED'][index])
        
            self.add_roof()
                
        self.idf.intersect_match()
        
        self.idf.set_wwr(wwr=self.a_window/self.a_wall+self.a_window, construction="Project External Window")
        
        if self.r_floor_roof != 1:
            self.idf.idfobjects['FENESTRATIONSURFACE:DETAILED'].pop(-1)
            self.idf.idfobjects['FENESTRATIONSURFACE:DETAILED'].pop(-1)
            
        self.set_constructions()
        self.add_heating_system()
        self.add_schedules()
        self.add_people()
        self.add_ventilation
        self.add_infiltration()        
        self.add_internal_gains()
    
        return self.idf

    def get_idf(self):
        return (self.systems_data)