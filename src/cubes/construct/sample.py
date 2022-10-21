'''This module has functions to sample a building from a database plus additional attributes from distributions'''

import constants as con
import random

#Path will need changed when we get a data folder in construct
raw_data = pd.read_excel("exp/jack/Data/AmBIENCe_Geometry_Constructions.xlsx")
raw_system_data = pd.read_excel("exp/jack/Data/AmBIENCe_Energy_Systems.xlsx")

def clean_ambience_system_data(sy_dt):
    sy_dt.columns = sy_dt.iloc[0,:]
    sy_dt = sy_dt.iloc[1:,:]
    sy_dt = sy_dt.drop([1793,1794])
    sy_dt = sy_dt.reset_index(drop=True)
    return sy_dt

def filter_geometry_data_by_boiler(gm_dt, sy_dt):
    sy_dt = sy_dt[sy_dt["HEATING SYSTEM 1 TECHNOLOGY"].str.contains("boiler")]    
    sy_dt = pd.merge(sy_dt, gm_dt, left_index=True, right_index=True)
    gm_dt = pd.DataFrame(sy_dt.iloc[:,34:])
    return gm_dt

raw_system_data = clean_ambience_system_data(raw_system_data)

raw_data = filter_geometry_data_by_boiler(raw_data, raw_system_data)

def calc_roof_floor_ratio(data):
    data["REFERENCE BUILDING FLOOR ROOF RATIO"] = data["REFERENCE BUILDING ROOF AREA (m2)"]/data["REFERENCE BUILDING GROUND FLOOR AREA (m2)"]
    return data

def calc_window_wall_ratio(data):
    data["REFERENCE BUILDING WINDOW WALL RATIO"] = data["REFERENCE BUILDING WINDOW AREA (m2)"]/data["REFERENCE BUILDING WALL AREA (m2)"]
    return data
    
def sample_database(data):
    
    calc_roof_floor_ratio(data)
    calc_window_wall_ratio(data)    
    
    archetype_geometry = data.sample(n=1, 
                                     weights=data["NUMBER OF REFERENCE BUILDINGS IN THE BUILDING STOCK SEGMENT"], 
                                    ignore_index=True)
    
    noise = 0.1 #fractional change in geometry value/standard deviation 
    
    #These are the data which have noise added to - can add more elements into the future
    geometry_elements = ["REFERENCE BUILDING GROUND FLOOR AREA (m2)",
                         "REFERENCE BUILDING WINDOW WALL RATIO",
                         "REFERENCE BUILDING FLOOR ROOF RATIO"]
    
    for element in geometry_elements:
        
        deviation = archetype_geometry[element]*noise
        
        archetype_geometry[element] = np.random.normal(archetype_geometry[element], deviation)
        
        if element == "REFERENCE BUILDING FLOOR ROOF RATIO":
            
            if archetype_geometry[element].values[0] < 1:
                archetype_geometry[element] = 1
                
                
    archetype_system = raw_system_data[raw_system_data["Building typology"]==archetype_geometry["REFERENCE BUILDING CODE"].values[0]]
    
    return archetype_geometry, archetype_system

#def sample_from_database():

    #geometry_data = random.sample(con.DATABASE,1)
    #systems_data = random.sample(con.DATABASE,1)
    #return geometry_data,systems_data