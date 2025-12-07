from pathlib import Path

EPLUS_ROOT = Path("/usr/local/EnergyPlus-9-5-0/")
IDD_PATH = EPLUS_ROOT / "Energy+.idd"

# Weather files
WEATHER_COHEAT = Path("/workspaces/CUBES/cubes/data/weather/loughborough_beizaee_oiko_2013.epw")
WEATHER_THERMOSTAT = Path("/workspaces/CUBES/cubes/data/weather/loughborough_beizaee_oiko.epw")

PROJECT_ROOT = Path("/workspaces/CUBES/beizaee_validation")

RUN_ROOT = PROJECT_ROOT / "experiments"
SCHEDULE_SOURCE = PROJECT_ROOT / "input_data"
