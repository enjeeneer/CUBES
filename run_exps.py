from pathlib import Path
import argparse
import shutil
from glob import glob
from eppy.modeleditor import IDF
from cubes.constants import BASE_DIR

# === Base Paths ===
VALIDATION_DIR = BASE_DIR / "beizaee_validation"
# WEATHER_FILE = BASE_DIR / "cubes/data/weather/loughborough_beizaee.epw"
WEATHER_FILE = BASE_DIR / "cubes/data/weather/loughborough_beizaee_oiko.epw"
EPLUS_PATH = "/usr/local/EnergyPlus-9-5-0/"

# Set IDD path
IDF.setiddname(str(Path(EPLUS_PATH) / "Energy+.idd"))


def test_idf(experiment, period, part_load=None, efficiency=None, mode="validation"):
    base_model_path = VALIDATION_DIR / "h28_test.idf"
    base_model_name = base_model_path.stem  # gives "h28_model_afn"

    base_idf = IDF(str(base_model_path), str(WEATHER_FILE))

    # === Heating control strategy ===
    heating_path = VALIDATION_DIR / "thermostat_control" / f"heating_{experiment}.idf"
    heating_idf = IDF(str(heating_path), str(WEATHER_FILE))
    for key in heating_idf.idfobjects:
        for obj in heating_idf.idfobjects[key]:
            base_idf.copyidfobject(obj)

    # === Run period ===
    run_period_path = VALIDATION_DIR / "run_period" / f"{period}_run.idf"
    run_idf = IDF(str(run_period_path), str(WEATHER_FILE))
    base_idf.idfobjects["RUNPERIOD"] = []
    for run_obj in run_idf.idfobjects["RUNPERIOD"]:
        base_idf.copyidfobject(run_obj)

    # === Always-include components ===
    always_include = [
        VALIDATION_DIR / "baseboard" / "h28_baseboard.idf",
        VALIDATION_DIR / "equipment_gains" / "equipment_gains.idf",
        VALIDATION_DIR / "people" / "people.idf",
    ]
    for include_path in always_include:
        extra_idf = IDF(str(include_path), str(WEATHER_FILE))
        for key in extra_idf.idfobjects:
            for obj in extra_idf.idfobjects[key]:
                base_idf.copyidfobject(obj)

    # === Mode-specific schedules ===
    if mode == "validation":
        sched_files = [
            VALIDATION_DIR / "door_schedule" / "beizaee_door_schedule.idf",
            VALIDATION_DIR / "occupancy_schedule" / "beizaee_occupancy.idf",
        ]
    elif mode == "evaluation":
        sched_files = [
            VALIDATION_DIR / "door_schedule" / "occupancy_door_schedule.idf",
            VALIDATION_DIR / "occupancy_schedule" / "real_occupancy.idf",
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for sched_path in sched_files:
        sched_idf = IDF(str(sched_path), str(WEATHER_FILE))
        for key in sched_idf.idfobjects:
            for obj in sched_idf.idfobjects[key]:
                base_idf.copyidfobject(obj)

    # === Optional boiler part load ===
    if part_load:
        pl_path = VALIDATION_DIR / "boiler" / f"minimum_part_load_{part_load}.idf"
        pl_idf = IDF(str(pl_path), str(WEATHER_FILE))
        for key in pl_idf.idfobjects:
            for obj in pl_idf.idfobjects[key]:
                base_idf.copyidfobject(obj)

    # === Optional boiler efficiency curve ===
    if efficiency:
        eff_path = VALIDATION_DIR / "boiler" / "efficiency" / f"{efficiency}.idf"
        eff_idf = IDF(str(eff_path), str(WEATHER_FILE))
        for key in eff_idf.idfobjects:
            for obj in eff_idf.idfobjects[key]:
                base_idf.copyidfobject(obj)

    # === Output directory ===
    folder_name = f"{experiment}__part_{part_load or 'none'}__eff_{efficiency or 'none'}"
    output_root = VALIDATION_DIR / f"runs_{base_model_name}_{mode}"
    output_dir = output_root / period / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Copy .csv and .sch files ===
    external_source_dir = VALIDATION_DIR / "building_model-external"
    external_target_dir = output_dir / "building_model-external"
    external_target_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["*.csv", "*.sch"]:
        for filepath in glob(str(external_source_dir / ext)):
            shutil.copy(filepath, external_target_dir / Path(filepath).name)
    for obj in base_idf.idfobjects["SCHEDULE:FILE"]:
        fname = Path(obj.File_Name).name
        obj.File_Name = str((external_target_dir / fname).resolve())

    # === Reset GlobalGeometryRules ===
    base_idf.idfobjects["GLOBALGEOMETRYRULES"] = []
    base_idf.newidfobject(
        "GLOBALGEOMETRYRULES",
        Starting_Vertex_Position="LowerLeftCorner",
        Vertex_Entry_Direction="CounterClockWise",
        Coordinate_System="World",
        Daylighting_Reference_Point_Coordinate_System="World",
        Rectangular_Surface_Coordinate_System="World",
    )

    # === Save & run ===
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
    print(f"✅ Simulation complete: {experiment} ({mode}) with '{period}' run period")
    print(f"📁 Results saved to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EnergyPlus simulation with dynamic inputs.")
    parser.add_argument(
        "--experiment",
        choices=["zonal_control", "conventional_control", "occupancy_control"],
        help="Heating control strategy",
    )
    parser.add_argument(
        "--period",
        choices=["test", "beizaee", "lynch"],
        help="Simulation period",
    )
    parser.add_argument(
        "--part_load",
        default=None,
        help="Boiler minimum part load (e.g., 0_0, 0_2)",
    )
    parser.add_argument(
        "--efficiency",
        default=None,
        choices=["constant", "cubic", "quadratic"],
        help="Boiler efficiency curve",
    )
    parser.add_argument(
        "--mode",
        choices=["validation", "evaluation"],
        default="validation",
        help="Which schedules to use: validation (Beizaee) or evaluation (PIR-based)",
    )

    args = parser.parse_args()
    test_idf(args.experiment, args.period, args.part_load, args.efficiency, args.mode)
