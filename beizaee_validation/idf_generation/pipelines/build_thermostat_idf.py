# idf_generation/pipelines/build_thermostat_idf.py

from pathlib import Path
from eppy.modeleditor import IDF
from idf_generation.base_building.generate_base_building import generate_base_building
from idf_generation.transforms.clean_experiment_objects import clean_experiment_objects
from idf_generation.transforms.add_blinds import add_always_closed_blinds
from experiments.shared.yaml_loader import ExperimentConfig

def build_thermostat_idf(config: ExperimentConfig, yaml_dir: Path,
                        mode=None, run_period=None,
                        boiler_part_load=None, boiler_efficiency=None):
    """
    Build a thermostat experiment IDF from YAML configuration.

    Args:
        config: ExperimentConfig loaded from YAML
        yaml_dir: Directory containing the experiment YAML file
        mode: Override default mode (validation/evaluation)
        run_period: Override default run period (test/beizaee/lynch)
        boiler_part_load: Override default boiler part load (0_0/0_2/0_4)
        boiler_efficiency: Override default boiler efficiency (constant/cubic/quadratic)

    Returns:
        IDF object ready to run
    """
    # Get resolved configuration with defaults
    run_config = config.get_run_config(mode, run_period, boiler_part_load, boiler_efficiency)

    # Start with base building
    idf = generate_base_building()
    clean_experiment_objects(idf)
    add_always_closed_blinds(idf, "bedroom_3")


    # Add thermostat control strategy
    thermostat_path = config.resolve_file_path(config.thermostat_control, yaml_dir)
    merge_idf_file(idf, thermostat_path)

    # Add always-include components
    for component in config.always_include:
        for name, path in component.items():
            component_path = config.resolve_file_path(path, yaml_dir)
            merge_idf_file(idf, component_path)

    # Add mode-specific schedules
    mode_schedules = config.modes[run_config['mode']]
    for schedule_name, schedule_path in mode_schedules.items():
        # Handle cross-experiment references (e.g., "occupancy/door_schedule.idf")
        # If path starts with experiment name, resolve relative to parent thermostat directory
        if schedule_path.startswith(('occupancy/', 'conventional/', 'zonal/')):
            resolved_path = yaml_dir.parent / schedule_path
        else:
            resolved_path = config.resolve_file_path(schedule_path, yaml_dir)
        merge_idf_file(idf, resolved_path)

    # Add run period
    run_period_path = config.resolve_file_path(
        config.run_periods[run_config['run_period']],
        yaml_dir
    )
    # Clear existing run periods before adding new one
    idf.idfobjects["RUNPERIOD"] = []
    merge_idf_file(idf, run_period_path)

    # Add boiler part load if specified
    if run_config['boiler_part_load'] != 'none':
        # Find the part load option
        for option_dict in config.boiler['part_load']['options']:
            if run_config['boiler_part_load'] in option_dict:
                part_load_path = config.resolve_file_path(
                    option_dict[run_config['boiler_part_load']],
                    yaml_dir
                )
                merge_idf_file(idf, part_load_path)
                break

    # Add boiler efficiency if specified
    if run_config['boiler_efficiency'] != 'none':
        # Find the efficiency option
        for option_dict in config.boiler['efficiency']['options']:
            if run_config['boiler_efficiency'] in option_dict:
                efficiency_path = config.resolve_file_path(
                    option_dict[run_config['boiler_efficiency']],
                    yaml_dir
                )
                merge_idf_file(idf, efficiency_path)
                break

    # ----------------------------------------
    # Load output variables from coheat_variables.idf
    # ----------------------------------------
    variables_idf_path = Path("/workspaces/CUBES/beizaee_validation/idf_generation/transforms/variables.idf")
    if variables_idf_path.exists():
        # Load variables.idf as a separate IDF object
        variables_idf = IDF(str(variables_idf_path))

        # Copy all Output:Variable objects from variables.idf to main idf
        for var in variables_idf.idfobjects.get("OUTPUT:VARIABLE", []):
            idf.newidfobject(
                "OUTPUT:VARIABLE",
                Key_Value=var.Key_Value,
                Variable_Name=var.Variable_Name,
                Reporting_Frequency=var.Reporting_Frequency
            )

    # Reset GlobalGeometryRules (matching old run_exps.py behavior)
    idf.idfobjects["GLOBALGEOMETRYRULES"] = []
    idf.newidfobject(
        "GLOBALGEOMETRYRULES",
        Starting_Vertex_Position="LowerLeftCorner",
        Vertex_Entry_Direction="CounterClockWise",
        Coordinate_System="World",
        Daylighting_Reference_Point_Coordinate_System="World",
        Rectangular_Surface_Coordinate_System="World",
    )

    return idf, run_config


def merge_idf_file(base_idf: IDF, file_path: Path):
    """Merge all objects from an IDF file into the base IDF."""
    if not file_path.exists():
        raise FileNotFoundError(f"IDF component not found: {file_path}")

    # Load the component IDF
    component_idf = IDF(str(file_path))

    # Copy all objects from component to base
    for key in component_idf.idfobjects:
        for obj in component_idf.idfobjects[key]:
            base_idf.copyidfobject(obj)
