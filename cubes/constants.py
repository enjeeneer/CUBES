"""constants used across the whole package"""
import os
from pathlib import Path

EPLUS_PATH = "/usr/local/EnergyPlus-9-5-0/"
package_directory = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = Path(__file__).parent.parent

NATURAL_GAS_EMISSIONS_FACTOR = 52  # gCO2eq/MJ
MJ_TO_KWH = 0.277778
