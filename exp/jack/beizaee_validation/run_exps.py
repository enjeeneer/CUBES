from pathlib import Path
import os
from eppy.modeleditor import IDF

def test_idf(experiment, period):
    # Paths
    base_model_path = "/workspaces/CUBES/exp/jack/beizaee_validation/building_model.idf"

    if experiment == 'zonal_control':
        heating_path = "/workspaces/CUBES/exp/jack/beizaee_validation/thermostat_control/heating_zonal.idf"
    elif experiment == 'conventional_control':
        heating_path = "/workspaces/CUBES/exp/jack/beizaee_validation/thermostat_control/heating_conventional.idf"
    else:
        heating_path = "/workspaces/CUBES/exp/jack/beizaee_validation/thermostat_control/heating_occupancy.idf"

    if period == 'test':
        run_period_path = "/workspaces/CUBES/exp/jack/beizaee_validation/run_period/test_run.idf"
    elif period == 'beizaee':
        run_period_path = "/workspaces/CUBES/exp/jack/beizaee_validation/run_period/beizaee_run.idf"
    else:
        run_period_path = "/workspaces/CUBES/exp/jack/beizaee_validation/run_period/lynch_run.idf"

    weather_file_path = "/workspaces/CUBES/cubes/data/weather/loughborough_beizaee.epw"
    EPLUS_PATH = "/usr/local/EnergyPlus-9-5-0/"
    iddfile = os.path.join(EPLUS_PATH, "Energy+.idd")
    IDF.setiddname(iddfile)

    # Load base model
    base_idf = IDF(base_model_path, weather_file_path)

    # Load heating objects
    heating_idf = IDF(heating_path, weather_file_path)
    for key in heating_idf.idfobjects:
        for obj in heating_idf.idfobjects[key]:
            base_idf.copyidfobject(obj)

    # Load run period
    run_idf = IDF(run_period_path, weather_file_path)

    # Remove any existing RunPeriod(s) in base model
    base_idf.idfobjects['RUNPERIOD'] = []

    # Copy RunPeriod(s) from selected file
    for run_obj in run_idf.idfobjects['RUNPERIOD']:
        base_idf.copyidfobject(run_obj)

    # ✅ Updated output directory structure: run_period/experiment
    output_dir = os.path.join(period, experiment)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    idf_save_path = os.path.join(output_dir, "modified_test.idf")
    base_idf.save(filename=idf_save_path)
    base_idf.run(output_directory=output_dir)

    print(f"✅ {experiment} simulation completed with '{period}' run period. Results saved to: {output_dir}/")


if __name__ == "__main__":
    run = "lynch"
    test_idf("conventional_control", run)
    test_idf("zonal_control", run)
    test_idf("occupancy_control", run)
