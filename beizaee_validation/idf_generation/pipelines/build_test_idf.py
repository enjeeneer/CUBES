# idf_generation/pipelines/build_test_idf.py

from eppy.modeleditor import IDF
from idf_generation.base_building.generate_base_building import generate_base_building
from idf_generation.transforms.clean_experiment_objects import clean_experiment_objects
from idf_generation.transforms.add_blinds import add_always_closed_blinds

def build_test_idf(save=False):
    idf = generate_base_building()
    clean_experiment_objects(idf)
    add_always_closed_blinds(idf, "bedroom_3")

    if save:
        out = "idf_generation/output/test.idf"
        idf.save(out)
        return out

    return idf
