# idf_generation/transforms/thermostat.py

from eppy.modeleditor import IDF
from pathlib import Path

def apply_thermostat_logic(idf, experiment, base_dir):
    """
    Add heating control objects depending on experiment:
        - 'conventional'
        - 'zonal'
        - 'occupancy_control'
    """

    exp_file = {
        "conventional": "conventional/thermostat_control.idf",
        "zonal_control": "zonal/thermostat_control.idf",
        "occupancy_control": "occupancy/thermostat_control.idf",
    }.get(experiment)

    if exp_file is None:
        raise ValueError(f"Unknown thermostat experiment: {experiment}")

    control_path = base_dir / "experiments" / "thermostat" / exp_file
    control_idf = IDF(str(control_path))

    for key in control_idf.idfobjects:
        for obj in control_idf.idfobjects[key]:
            idf.copyidfobject(obj)

    return idf
