# experiments/thermostat/run_thermostat.py

import argparse
import shutil
from pathlib import Path
from glob import glob
from eppy.modeleditor import IDF

from experiments.shared.yaml_loader import ExperimentConfig
from experiments.shared.config import WEATHER_THERMOSTAT, RUN_ROOT, IDD_PATH, SCHEDULE_SOURCE
from idf_generation.pipelines.build_thermostat_idf import build_thermostat_idf
from experiments.shared.runner import run_energyplus
from cubes.constants import BASE_DIR

# === Base Paths ===
WEATHER_THERMOSTAT = BASE_DIR / "cubes/data/weather/loughborough_beizaee_oiko.epw"


def run_thermostat_experiment(experiment_name, mode=None, run_period=None,
                              boiler_part_load=None, boiler_efficiency=None):
    """
    Run a thermostat experiment from YAML configuration.

    Args:
        experiment_name: Name of experiment (conventional_control, zonal_control, occupancy_control)
        mode: Override default mode (validation/evaluation)
        run_period: Override default run period (test/beizaee/lynch)
        boiler_part_load: Override default boiler part load (0_0/0_2/0_4)
        boiler_efficiency: Override default boiler efficiency (constant/cubic/quadratic)
    """
    # Map experiment names to directory names
    experiment_dir_map = {
        'conventional_control': 'conventional',
        'zonal_control': 'zonal',
        'occupancy_control': 'occupancy'
    }

    dir_name = experiment_dir_map.get(experiment_name, experiment_name)

    # Load experiment configuration
    experiment_dir = Path(__file__).parent / dir_name
    yaml_path = experiment_dir / "experiment.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"Experiment YAML not found: {yaml_path}")

    config = ExperimentConfig.from_yaml(yaml_path)

    # Build IDF using the pipeline
    idf, run_config = build_thermostat_idf(
        config,
        yaml_dir=experiment_dir,
        mode=mode,
        run_period=run_period,
        boiler_part_load=boiler_part_load,
        boiler_efficiency=boiler_efficiency
    )

    # Create output directory structure matching old run_exps.py
    # Format: runs_{mode}/{run_period}/{experiment}__{part_load}__{efficiency}
    folder_name = f"{config.name}__part_{run_config['boiler_part_load']}__eff_{run_config['boiler_efficiency']}"
    output_root = RUN_ROOT / "thermostat" / f"runs_{run_config['mode']}" / run_config['run_period']
    output_dir = output_root / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy external files (.csv and .sch) to output directory
    external_target_dir = output_dir / "building_model-external"
    external_target_dir.mkdir(parents=True, exist_ok=True)

    for ext in ["*.csv", "*.sch"]:
        for filepath in glob(str(SCHEDULE_SOURCE / ext)):
            shutil.copy(filepath, external_target_dir / Path(filepath).name)

    # Update SCHEDULE:FILE paths to point to copied files
    for obj in idf.idfobjects.get("SCHEDULE:FILE", []):
        if hasattr(obj, "File_Name"):
            fname = Path(obj.File_Name).name
            obj.File_Name = str((external_target_dir / fname).resolve())

    # Save IDF to output directory
    idf_save_path = output_dir / "modified_test.idf"
    idf.saveas(str(idf_save_path))

    # Set IDF attributes and run
    idf.epw = str(WEATHER_THERMOSTAT)
    idf.idfname = str(idf_save_path)

    print(f"{str(WEATHER_THERMOSTAT)=}")

    # Run EnergyPlus
    idf.run(
        weather=str(WEATHER_THERMOSTAT),
        output_directory=str(output_dir),
        output_prefix="eplus",
        output_suffix="L",
        expandobjects=True,
        epmacro=True,
        readvars=True,
    )

    print(f"✅ Simulation complete: {config.name} ({run_config['mode']}) with '{run_config['run_period']}' run period")
    print(f"📁 Results saved to: {output_dir}/")

    return output_dir


if __name__ == "__main__":
    # Set IDD path
    IDF.setiddname(str(IDD_PATH))

    parser = argparse.ArgumentParser(
        description="Run thermostat experiment from YAML configuration"
    )
    parser.add_argument(
        "experiment",
        choices=["conventional_control", "zonal_control", "occupancy_control"],
        help="Experiment name (matches directory name)"
    )
    parser.add_argument(
        "--mode",
        choices=["validation", "evaluation"],
        default=None,
        help="Override default mode (validation=Beizaee schedules, evaluation=PIR schedules)"
    )
    parser.add_argument(
        "--period",
        choices=["test", "beizaee", "lynch"],
        default=None,
        help="Override default run period"
    )
    parser.add_argument(
        "--part_load",
        choices=["0_0", "0_2", "0_4"],
        default=None,
        help="Override default boiler minimum part load"
    )
    parser.add_argument(
        "--efficiency",
        choices=["constant", "cubic", "quadratic"],
        default=None,
        help="Override default boiler efficiency curve"
    )

    args = parser.parse_args()

    run_thermostat_experiment(
        experiment_name=args.experiment,
        mode=args.mode,
        run_period=args.period,
        boiler_part_load=args.part_load,
        boiler_efficiency=args.efficiency
    )
