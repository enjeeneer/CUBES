# experiments/coheat/run_coheat.py

from eppy.modeleditor import IDF
from idf_generation.pipelines.build_coheat_idf import build_coheat_idf
from experiments.shared.runner import run_energyplus
from experiments.shared.config import WEATHER_COHEAT, RUN_ROOT, IDD_PATH

def run_coheat():
    idf = build_coheat_idf()

    run_dir = run_energyplus(
        idf,
        weather=WEATHER_COHEAT,
        output_root=RUN_ROOT / "coheat" / "runs",
        label="coheat"
    )

    print(f"✔ Coheat complete → {run_dir}")
    return run_dir

if __name__ == "__main__":
    IDF.setiddname(str(IDD_PATH))
    run_coheat()
