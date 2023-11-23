"""constants used across the whole package"""
import os
from pathlib import Path

EPLUS_PATH = "/usr/local/EnergyPlus-9-5-0/"
package_directory = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = Path(__file__).parent.parent

cwd_path = os.getcwd()
env_files_path = os.path.join(cwd_path, "input")
Path(env_files_path).mkdir(parents=True, exist_ok=True)

CH4_EMISSIONS_FACTOR = 52  # units?
