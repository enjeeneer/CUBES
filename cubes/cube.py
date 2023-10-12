"""
Module containing a building control environment a.k.a. a cube.
"""

from dataclasses import dataclass


@dataclass
class EnvConfig:
    """
    Skeleton configuration for building env (cube).
    Full definition TBC.
    """

    # general
    timestep_len: float  # minutes

    # geometry
    roof_mat_thickness: float
    roof_ins_thickness: float
    floor_area: float
    roof_pitch: float

    # materials
    insulation: str
    concrete: str
    brick: str
    stone: str
    wood: str

    # equipment
    heating: str  # e.g. ['heat_pump', 'boiler']

    # sensors
    sensor_no: int

    # weather
    location: str  # e.g. 'cambridge'
