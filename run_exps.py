from pathlib import Path
import os
import argparse
import shutil
from glob import glob
from eppy.modeleditor import IDF
from cubes.constants import BASE_DIR

# === Base Paths ===
VALIDATION_DIR = BASE_DIR / "beizaee_validation"
WEATHER_FILE = BASE_DIR / "cubes/data/weather/loughborough_beizaee.epw"
EPLUS_PATH = "/usr/local/EnergyPlus-9-5-0/"

# Set IDD path
IDF.setiddname(str(Path(EPLUS_PATH) / "Energy+.idd"))


def test_idf(experiment, period, part_load=None, efficiency=None):
    """Runs an EnergyPlus simulation with a given configuration."""

    # === Load base model ===
    base_model_path = VALIDATION_DIR / "building_model_full_afn.idf"
    base_idf = IDF(str(base_model_path), str(WEATHER_FILE))

    # === Add heating control ===
    heating_path = VALIDATION_DIR / "thermostat_control" / f"heating_{experiment}.idf"
    heating_idf = IDF(str(heating_path), str(WEATHER_FILE))
    for key in heating_idf.idfobjects:
        for obj in heating_idf.idfobjects[key]:
            base_idf.copyidfobject(obj)

    # === Add run period ===
    run_period_path = VALIDATION_DIR / "run_period" / f"{period}_run.idf"
    run_idf = IDF(str(run_period_path), str(WEATHER_FILE))
    base_idf.idfobjects["RUNPERIOD"] = []
    for run_obj in run_idf.idfobjects["RUNPERIOD"]:
        base_idf.copyidfobject(run_obj)

    # === Optional: Boiler part load ===
    if part_load:
        pl_path = VALIDATION_DIR / "boiler" / f"minimum_part_load_{part_load}.idf"
        pl_idf = IDF(str(pl_path), str(WEATHER_FILE))
        for key in pl_idf.idfobjects:
            for obj in pl_idf.idfobjects[key]:
                base_idf.copyidfobject(obj)

    # === Optional: Boiler efficiency curve ===
    if efficiency:
        eff_path = VALIDATION_DIR / "boiler" / "efficiency" / f"{efficiency}.idf"
        eff_idf = IDF(str(eff_path), str(WEATHER_FILE))
        for key in eff_idf.idfobjects:
            for obj in eff_idf.idfobjects[key]:
                base_idf.copyidfobject(obj)

    # === Output directory inside beizaee_validation ===
    folder_name = (
        f"{experiment}__part_{part_load or 'none'}__eff_{efficiency or 'none'}"
    )
    output_dir = VALIDATION_DIR / "runs" / period / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Copy .csv and .sch files to output_dir ===
    external_source_dir = VALIDATION_DIR / "building_model-external"
    external_target_dir = output_dir / "building_model-external"
    external_target_dir.mkdir(parents=True, exist_ok=True)

    for ext in ["*.csv", "*.sch"]:
        for filepath in glob(str(external_source_dir / ext)):
            filename = Path(filepath).name
            shutil.copy(filepath, external_target_dir / filename)

    # === Patch Schedule:File references to use absolute paths ===
    for obj in base_idf.idfobjects["SCHEDULE:FILE"]:
        fname = Path(obj.File_Name).name  # just the filename
        absolute_path = external_target_dir / fname
        obj.File_Name = str(absolute_path.resolve())

    # Remove old one if present
    base_idf.idfobjects["GLOBALGEOMETRYRULES"] = []

    # Add full version
    base_idf.newidfobject(
        "GLOBALGEOMETRYRULES",
        Starting_Vertex_Position="lowerleftcorner",
        Vertex_Entry_Direction="CounterClockWise",
        Coordinate_System="relative",
        Daylighting_Reference_Point_Coordinate_System="relative",
        Rectangular_Surface_Coordinate_System="relative",
    )

    # === Save IDF and run ===
    idf_save_path = output_dir / "modified_test.idf"
    base_idf.save(filename=str(idf_save_path))

    base_idf.run(
        weather=str(WEATHER_FILE),
        output_directory=str(output_dir),
        output_prefix="eplus",
        output_suffix="L",
        expandobjects=True,
        epmacro=True,
        readvars=True,
    )

    print(f"✅ Simulation complete: {experiment} with '{period}' run period")
    print(f"📁 Results saved to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run EnergyPlus simulation with dynamic inputs."
    )
    parser.add_argument(
        "experiment",
        choices=["zonal_control", "conventional_control", "occupancy_control"],
        help="Heating control strategy",
    )
    parser.add_argument(
        "period", choices=["test", "beizaee", "lynch"], help="Simulation period"
    )
    parser.add_argument(
        "--part_load", default=None, help="Boiler minimum part load (e.g., 0_0, 0_2)"
    )
    parser.add_argument(
        "--efficiency",
        default=None,
        choices=["constant", "cubic", "quadratic"],
        help="Boiler efficiency curve",
    )

    args = parser.parse_args()
    test_idf(args.experiment, args.period, args.part_load, args.efficiency)
