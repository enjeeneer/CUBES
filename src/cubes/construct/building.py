'''Defines the Building class '''

class Building:
    def __init__(self,geometry_data,systems_data):
        self.geometry_data = geometry_data
        self.system_data = systems_data

        self.a_ground_floor = self.geometry_data.loc[0]["Ground_Floor_Area_m2"]
        self.n_storey = int(self.geometry_data.loc[0]["Num_Storeys"])
        self.a_window = self.geometry_data.loc[0]["Window_Area_m2"]
        self.r_floor_roof = self.geometry_data.loc[0]["Floor/Roof_ratio"]
        
        self.h_ceiling = 2.5 #tabula default for all buildings
        self.l_wall = self.calc_wall_length()
        self.a_wall = self.calc_wall_area()
        self.h_roof = self.get_roof_height()

        self.idf = IDF('exp/jack/Data/Minimal.idf')
        #Future - Will need to automatically add in weather file based on locations
        self.idf.epw = "exp/jack/Data/USA_CO_Golden-NREL.724666_TMY3.epw"

    def calc_wall_area(self):
        return (self.l_wall*self.n_storey*self.h_ceiling)*4

    def calc_wall_length(self):
        #Will need to change this method depending on aspect ratio 
        #currently modelling footprint as square
        return np.sqrt(self.a_ground_floor)
        
    def get_roof_height(self):
        return (np.sqrt((self.l_wall**2)*((self.r_floor_roof**2)-1)))/2

    def get_roof_coordinates(self, h_roof):
        
        roof_coords = [[[self.l_wall, 0, self.n_storey*self.h_ceiling],
                        [self.l_wall, self.l_wall/2, self.n_storey*self.h_ceiling+h_roof],
                        [0,self.l_wall/2,self.n_storey*self.h_ceiling+h_roof],
                        [0,0,self.n_storey*self.h_ceiling]],
                        [[self.l_wall,self.l_wall,self.n_storey*self.h_ceiling],
                        [self.l_wall,self.l_wall/2,self.n_storey*self.h_ceiling+h_roof],
                        [0,self.l_wall/2,self.n_storey*self.h_ceiling+h_roof],
                        [0,self.l_wall,self.n_storey*self.h_ceiling]]]

        return roof_coords

    def get_roof_wall_coordinates(self, h_roof):
    
        wall_coords = [[[0, 0, self.n_storey*self.h_ceiling],
                        [0, self.l_wall, self.n_storey*self.h_ceiling],
                        [0,self.l_wall/2,self.n_storey*self.h_ceiling+h_roof],
                        [0,self.l_wall/2,self.n_storey*self.h_ceiling+h_roof]],
                        [[self.l_wall, 0, self.n_storey*self.h_ceiling],
                        [self.l_wall, self.l_wall, self.n_storey*self.h_ceiling],
                        [self.l_wall,self.l_wall/2,self.n_storey*self.h_ceiling+h_roof],
                        [self.l_wall,self.l_wall/2,self.n_storey*self.h_ceiling+h_roof]]]
    
    
        return wall_coords

    def add_roof(self):

        h_roof = self.get_roof_height()
        
        roof_coords = self.get_roof_coordinates(h_roof)
        
        wall_coords = self.get_roof_wall_coordinates(h_roof)
        
        
        roof_construction = self.idf.newidfobject('CONSTRUCTION',
                                            Name="REFERENCE ROOF",
                                            Outside_Layer = "DefaultMaterial")
        
        roof_zone = self.idf.newidfobject('ZONE',
                                    Name="Roof Space",)

        #May want to change nomenclature on naming new elements
        #Currently N_X means that there are X of the new elements, and N designates what one it is
            
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
        """This method reads the materials database, which contains all of the materials
            and their properties, which have been used by Ambience and then stores them in e+"""
        
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
    
        build_up_thickness = {"Wall": [self.geometry_data.loc[0]["REFERENCE BUILDING WALL MATERIAL THICKNESS (m)"],
                                    self.geometry_data.loc[0]["REFERENCE BUILDING WALL INSULATION MATERIAL THICKNESS (m)"]],
                            "Roof": [self.geometry_data.loc[0]["REFERENCE BUILDING WALL MATERIAL THICKNESS (m)"],
                                    self.geometry_data.loc[0]["REFERENCE BUILDING WALL INSULATION MATERIAL THICKNESS (m)"]],
                            "Floor": [self.geometry_data.loc[0]["REFERENCE BUILDING WALL MATERIAL THICKNESS (m)"],
                                    self.geometry_data.loc[0]["REFERENCE BUILDING WALL INSULATION MATERIAL THICKNESS (m)"]]}

        build_up = {"Wall":[self.geometry_data.loc[0]["REFERENCE BUILDING WALL MATERIAL"],
                            self.geometry_data.loc[0]["REFERENCE BUILDING WALL INSULATION MATERIAL"]],
                    "Roof":[self.geometry_data.loc[0]["REFERENCE BUILDING ROOF MATERIAL"],
                            self.geometry_data.loc[0]["REFERENCE BUILDING ROOF INSULATION MATERIAL"]],
                    "Floor":[self.geometry_data.loc[0]["REFERENCE BUILDING FLOOR MATERIAL"],
                            self.geometry_data.loc[0]["REFERENCE BUILDING FLOOR INSULATION MATERIAL"]]}
        
        for element in build_up_thickness:
            for i, layer in enumerate(build_up_thickness[element]):
                if layer == 0:
                    build_up[element].remove(i)
            
        return build_up

    def set_constructions(self):
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

    def add_heating_system(self):
              
        stat = self.idf.newidfobject(
            "HVACTEMPLATE:THERMOSTAT",
            Name="Thermostat",
            Constant_Heating_Setpoint=20,
            Constant_Cooling_Setpoint=25,
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
            Boiler_Type="CondensingHotWaterBoiler",
            Efficiency=0.8,
            Fuel_Type="NaturalGas",
        )
        
        self.idf.idfobjects["SIMULATIONCONTROL"][0].Do_Zone_Sizing_Calculation = "Yes"
    
    def build(self):
        
        #May want to change the block name - again nomenclature
        self.idf.add_block(name='Living',
                    coordinates=[(self.l_wall,0),
                                (self.l_wall,self.l_wall),
                                (0,self.l_wall),
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
        
        self.idf.set_wwr(wwr=self.a_window/self.a_wall, construction="Project External Window")
        
        if self.r_floor_roof != 1:
            self.idf.idfobjects['FENESTRATIONSURFACE:DETAILED'].pop(-1)
            self.idf.idfobjects['FENESTRATIONSURFACE:DETAILED'].pop(-1)
            
        self.set_constructions()
        self.add_heating_system()
    
        return self.idf

    def get_idf(self):
        return (self.systems_data)