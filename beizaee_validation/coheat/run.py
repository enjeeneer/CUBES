from pathlib import Path
from eppy.modeleditor import IDF
import os

EPLUS_PATH = "/usr/local/EnergyPlus-9-5-0"
IDF.setiddname(str(Path(EPLUS_PATH) / "Energy+.idd"))

idf_path = Path("/workspaces/CUBES/beizaee_validation/h28_coheat.idf")
epw_path = Path("/workspaces/CUBES/cubes/data/weather/loughborough_beizaee_oiko_2013.epw")
out_dir = Path("/workspaces/CUBES/beizaee_validation/coheat/output")

out_dir.mkdir(parents=True, exist_ok=True)

idf = IDF(str(idf_path), str(epw_path))

idf.run(
    weather=str(epw_path),
    output_directory=str(out_dir),
    output_prefix="eplus",
    output_suffix="L",
    expandobjects=True,
    epmacro=True,
    readvars=True,  # ✅ ensures eplusout.csv is generated
)

print(f"✅ Simulation complete for: {idf_path.name}")
print(f"📁 Outputs located at: {out_dir}")
print(f"📄 eplusout.csv will be found here: {out_dir / 'eplusout.csv'}")
