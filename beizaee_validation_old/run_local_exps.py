import subprocess

period="lynch"
mode="evaluation"
part_load="0_2"
efficiency="quadratic"
controls=["conventional_control" ,"zonal_control" ,"occupancy_control"]

for control in controls:
    print(f"▶ {control} | period={period} | part={part_load} | eff={efficiency} | mode={mode}")
    cmd=[
        "python3","run_exps.py",
        "--experiment",control,
        "--period",period,
        "--part_load",part_load,
        "--efficiency",efficiency,
        "--mode",mode
    ]
    subprocess.run(cmd,check=True)
