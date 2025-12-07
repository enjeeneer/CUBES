from pathlib import Path
import shutil

def copy_schedules(source_folder: Path, dest_folder: Path):
    dest_folder.mkdir(parents=True, exist_ok=True)
    for ext in ["*.csv", "*.sch"]:
        for f in source_folder.glob(ext):
            shutil.copy(f, dest_folder / f.name)
