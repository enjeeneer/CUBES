'''This module has functions to sample a building from a database plus additional attributes from distributions'''

import constants as con
import random

def sample_from_database():

    geometry_data = random.sample(con.DATABASE,1)
    systems_data = random.sample(con.DATABASE,1)
    return geometry_data,systems_data