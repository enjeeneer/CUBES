'''outward facing API of construct package'''

import building
import sample


def sample_idf():
    geometry_data,systems_data = sample.sample_from_database()
    build = building.Building(geometry_data,systems_data)
    idf = build.get_idf()


    return idf

print(sample_idf())