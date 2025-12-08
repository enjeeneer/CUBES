# Thermostat Control Experiments

## Running Experiments

### Basic Usage

```bash
cd /workspaces/CUBES/beizaee

# Run conventional control (validation mode, beizaee period)
python -m experiments.thermostat.run_thermostat conventional_control --mode validation --period beizaee

# Run zonal control (validation mode, beizaee period)
python -m experiments.thermostat.run_thermostat zonal_control --mode validation --period beizaee

# Run occupancy control (evaluation mode, lynch period)
python -m experiments.thermostat.run_thermostat occupancy_control --mode evaluation --period lynch
```

### Available Options

**Control Strategies:**
- `conventional_control` - Single building-wide thermostat
- `zonal_control` - Room-by-room manual schedules
- `occupancy_control` - Motion-activated PIR sensor control

**Modes:**
- `--mode validation` - Uses Beizaee diary-based occupancy schedules
- `--mode evaluation` - Uses PIR sensor-based occupancy schedules

**Run Periods:**
- `--period test` - 3-day test run
- `--period beizaee` - Beizaee validation period (Feb-Apr 2013)
- `--period lynch` - Lynch validation period

**Boiler Configuration:**
- `--part_load 0_0` / `0_2` / `0_4` - Minimum part load ratio
- `--efficiency constant` / `cubic` / `quadratic` - Efficiency curve type

### Example: Full Parameter Sweep

```bash
# Test all boiler configurations for conventional control
for part in 0_0 0_2 0_4; do
  for eff in constant cubic quadratic; do
    python -m experiments.thermostat.run_thermostat conventional_control \
      --mode validation --period beizaee \
      --part_load $part --efficiency $eff
  done
done
```

## Analyzing Results

### Validation Analysis (Beizaee Comparison)

```bash
cd /workspaces/CUBES/beizaee
jupyter notebook notebooks/thermostat_validation_example.ipynb
```

**Key plots in validation notebook:**
- Temperature deltas (Conventional - Smart Controls) vs Beizaee/Cockroft references
- Absolute temperatures by heating period (WholeDay, HeatingOn, HeatingOff)
- Gas consumption comparison with reference overlays
- Daily gas consumption time series
- Infiltration (ACH) analysis

### Evaluation Analysis (PIR-based)

```bash
cd /workspaces/CUBES/beizaee
jupyter notebook notebooks/thermostat_evaluation_analysis.ipynb
```

## Output Structure

Results are saved to:
```
/workspaces/CUBES/beizaee/experiments/thermostat/
  runs_validation/          # Beizaee diary-based schedules
    beizaee/               # Beizaee period results
      conventional_control__part_0_2__eff_quadratic/
        eplusout.csv       # Main results file
        modified_test.idf  # IDF used for this run
        building_model-external/  # Schedules and external files
    lynch/                 # Lynch period results
    test/                  # 3-day test results

  runs_evaluation/          # PIR sensor-based schedules
    lynch/
      occupancy_control__part_0_2__eff_quadratic/
        ...
```

## Key Analysis Functions

### Loading Results

```python
from analysis.shared import load_run_csv

df_conv = load_run_csv("/path/to/conventional_control__part_0_2__eff_quadratic/eplusout.csv")
df_zonal = load_run_csv("/path/to/zonal_control__part_0_2__eff_quadratic/eplusout.csv")
```

### Temperature Comparison

```python
from analysis.thermostat import compare_room_temperatures
from analysis.thermostat.plotting import plot_boiler_config_comparison

results = {
    "conventional_control__part_0_2__eff_quadratic": df_conv,
    "zonal_control__part_0_2__eff_quadratic": df_zonal
}

# Compare temperatures (filtered by baseboard activity)
df_temps, energy_use = compare_room_temperatures(results, baseboard_filter=1)

# Plot deltas with Beizaee/Cockroft overlays
fig, ax = plot_boiler_config_comparison(
    results_dict=results,
    config_tag="part_0_2__eff_quadratic",
    controls=["conventional_control", "zonal_control"]
)
```

### Gas Consumption

```python
from analysis.thermostat.plotting import (
    plot_gas_consumption_by_config,
    plot_daily_gas_consumption
)

# Multi-config comparison (includes reference overlays)
df_energy = plot_gas_consumption_by_config(
    results_dict=results,
    show_references=True
)

# Daily time series
fig, ax = plot_daily_gas_consumption(results)
```

### Infiltration Analysis

```python
from analysis.coheat.analysis_pipeline import run_coheat_analysis, plot_infiltration_results

idf_path = "/path/to/modified_test.idf"
csv_path = "/path/to/eplusout.csv"
idd_path = "/usr/local/EnergyPlus-9-5-0/Energy+.idd"

df_analysis, ach, daily_kwh = run_coheat_analysis(idf_path, csv_path, idd_path)
plot_infiltration_results(ach)
```

## Quick Validation Test

Run a complete validation workflow:

```bash
# 1. Run all three control strategies (validation mode, beizaee period)
python -m experiments.thermostat.run_thermostat conventional_control --mode validation --period beizaee
python -m experiments.thermostat.run_thermostat zonal_control --mode validation --period beizaee
python -m experiments.thermostat.run_thermostat occupancy_control --mode validation --period beizaee

# 2. Open analysis notebook
jupyter notebook notebooks/thermostat_validation_example.ipynb

# 3. Update paths in notebook to point to new runs, then execute all cells
```

## Experiment Configuration

Experiments are defined in YAML files:
```
/workspaces/CUBES/beizaee/experiments/thermostat/
  conventional/experiment.yaml
  zonal/experiment.yaml
  occupancy/experiment.yaml
  shared/                   # Shared components (boiler configs, schedules, etc.)
```

Each YAML specifies:
- Thermostat control strategy IDF
- Mode-specific schedules (validation vs evaluation)
- Run period options
- Boiler configuration sweep parameters
- Default settings

## Troubleshooting

**Error: "IDF component not found"**
- Check paths in experiment.yaml are correct
- Verify shared files exist in `experiments/thermostat/shared/`

**Error: "No gas energy column found"**
- Check EnergyPlus simulation completed successfully
- Look for errors in `eplusout.err` file

**Weather data shows incorrect temperatures**
- Verify `SimulationControl` was removed during IDF build
- Check `eplusout.err` for weather file loading messages

## Contact

For issues or questions, refer to the main CUBES documentation or analysis notebooks for detailed examples.
