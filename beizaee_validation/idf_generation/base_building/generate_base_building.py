import pandas as pd
from pathlib import Path
from cubes.construct.buildingconfig import load_building_config
from cubes.construct.building import Building

BASE_DIR = Path("/workspaces/CUBES/beizaee_validation")


def generate_base_building():
    """Return an unsaved IDF object for the base building."""
    files_dir = BASE_DIR / "input_data"
    bc_file_path = Path(
        "/workspaces/CUBES/cubes/data/buildingconfigs/thermostat_experiment/2023/case0.json"
    )

    materials = pd.read_pickle("/workspaces/CUBES/cubes/materials.pickle")
    windows = pd.read_pickle("/workspaces/CUBES/cubes/windows.pickle")

    BC = load_building_config(str(bc_file_path), files_dir=str(files_dir))

    building = Building(BC, materials, windows)
    building.build()
    return building.get_idf()


def main():
    from eppy.modeleditor import IDF
    IDF.setiddname("/usr/local/EnergyPlus-9-5-0/Energy+.idd")

    idf = generate_base_building()

    output_dir = BASE_DIR / "idf_generation" / "output"
    output_dir.mkdir(exist_ok=True)

    out_path = output_dir / "base_building.idf"
    idf.save(str(out_path))

    print(f"Base building IDF saved to {out_path}")


if __name__ == "__main__":
    main()
