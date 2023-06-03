"""coonstants used throughot package package"""
from cubes.constants import env_files_path

weather_file_path = env_files_path + "/weather.epw"
ddy_file_path = env_files_path + "/weather.ddy"
idf_file_path = env_files_path + "/building_model.idf"
rdd_file_path = env_files_path + "/building_model.rdd"
temp_output_path = env_files_path + "/temp"

weather_file_dict = {
    "Cambridge": "cambridge_lat=52.25_lng=0.25_period=2021",
    "Munich": "Munich",
    "Nottingham": "Nottingham_2022",
}
