# experiments/shared/runner.py

from pathlib import Path

def run_energyplus(idf, weather, output_root, label):
    output_root.mkdir(parents=True, exist_ok=True)

    run_dir = output_root / f"{label}_run"
    run_dir.mkdir(exist_ok=True)

    # Save IDF to run directory before running
    idf_path = run_dir / "in.idf"
    idf.saveas(str(idf_path))

    # Set the weather file on the IDF object (required for programmatically generated IDFs)
    idf.epw = str(weather)
    idf.idfname = str(idf_path)

    idf.run(
        weather=str(weather),
        output_directory=str(run_dir),
        output_prefix="eplus",
        output_suffix="L",
        expandobjects=True,
        epmacro=True,
        readvars=True,
    )
    return run_dir
