# idf_generation/transforms/schedules.py

from eppy.modeleditor import IDF
from pathlib import Path

def _copy_all_objects(src_idf, dst_idf):
    for key in src_idf.idfobjects:
        for obj in src_idf.idfobjects[key]:
            dst_idf.copyidfobject(obj)

def apply_validation_schedules(idf, base_dir):
    """
    Apply the Beizaee validation schedules:
        - Door schedules (manual diary)
        - Occupancy schedules (Beizaee)
    """
    door_sched = base_dir / "experiments" / "thermostat" / "shared" / "door_schedule.idf"
    occ_sched  = base_dir / "experiments" / "thermostat" / "shared" / "occupancy_schedule.idf"

    door_idf = IDF(str(door_sched))
    occ_idf  = IDF(str(occ_sched))

    _copy_all_objects(door_idf, idf)
    _copy_all_objects(occ_idf, idf)

    return idf


def apply_evaluation_schedules(idf, base_dir):
    """
    Apply the PIR-based evaluation schedules:
        - occupancy derived from PIR sensors
        - door schedules inferred
    """
    door_sched = base_dir / "experiments" / "thermostat" / "shared" / "evaluation_door_schedule.idf"
    occ_sched  = base_dir / "experiments" / "thermostat" / "shared" / "evaluation_occupancy.idf"

    door_idf = IDF(str(door_sched))
    occ_idf  = IDF(str(occ_sched))

    _copy_all_objects(door_idf, idf)
    _copy_all_objects(occ_idf, idf)

    return idf
