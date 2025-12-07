# idf_generation/pipelines/build_coheat_idf.py

from eppy.modeleditor import IDF
from idf_generation.base_building.generate_base_building import generate_base_building
from idf_generation.transforms.build_coheat import make_coheating_model
from pathlib import Path

zone_setpoints = {
    "front_room": 24.32,
    "backroom": 24.64,
    "kitchen": 25.15,
    "hall_downstairs": 24.0,
    "hall_upstairs": 24.67,
    "bedroom_1": 24.62,
    "bedroom_2": 24.67,
    "bathroom": 23.53,
    "bedroom_3": 24.83,
}

def build_coheat_idf(save=False):
    idf = generate_base_building()
    idf = make_coheating_model(idf, zone_setpoints)

    if save:
        out = Path("idf_generation/output/coheat.idf")
        idf.save(str(out))
        return out

    return idf
