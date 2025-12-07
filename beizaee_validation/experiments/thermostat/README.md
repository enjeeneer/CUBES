# Thermostat Experiments

YAML-based configuration system for running thermostat control experiments with EnergyPlus.

## Overview

Three experiment types are supported:
- **conventional_control**: Single building-wide thermostat with constant setpoints
- **zonal_control**: Independent temperature schedules for each zone (room-by-room control)
- **occupancy_control**: Motion-activated heating control based on PIR sensor data

## Quick Start

### Run with defaults

```bash
# Conventional control with defaults (validation mode, lynch period, part_load 0_2, efficiency quadratic)
python3.9 -m experiments.thermostat.run_thermostat conventional_control

# Zonal control with defaults
python3.9 -m experiments.thermostat.run_thermostat zonal_control

# Occupancy control with defaults (uses evaluation mode by default)
python3.9 -m experiments.thermostat.run_thermostat occupancy_control
```

### Override parameters

```bash
# Run conventional control with evaluation mode (PIR-based schedules)
python3.9 -m experiments.thermostat.run_thermostat conventional_control --mode evaluation

# Run with different run period
python3.9 -m experiments.thermostat.run_thermostat conventional_control --period beizaee

# Run with different boiler configuration
python3.9 -m experiments.thermostat.run_thermostat conventional_control --part_load 0_0 --efficiency cubic

# Combine multiple overrides
python3.9 -m experiments.thermostat.run_thermostat zonal_control --mode evaluation --period test --part_load 0_4
```

## Parameters

### Experiment types
- `conventional_control`: Single thermostat for entire building
- `zonal_control`: Room-by-room thermostat control
- `occupancy_control`: Motion-activated heating

### Mode (--mode)
- `validation`: Uses Beizaee diary-based schedules (default for conventional/zonal)
- `evaluation`: Uses PIR sensor-based schedules (default for occupancy)

### Run period (--period)
- `test`: 3-day test period
- `beizaee`: Beizaee validation period
- `lynch`: Lynch validation period (default)

### Boiler part load (--part_load)
- `0_0`: No minimum part load
- `0_2`: 20% minimum part load (default)
- `0_4`: 40% minimum part load

### Boiler efficiency (--efficiency)
- `constant`: Constant efficiency
- `cubic`: Cubic efficiency curve
- `quadratic`: Quadratic efficiency curve (default)

## Output Structure

Results are saved to:
```
experiments/thermostat/runs_{mode}/{run_period}/{experiment}__part_{part_load}__eff_{efficiency}/
```

Example:
```
experiments/thermostat/runs_validation/lynch/conventional_control__part_0_2__eff_quadratic/
├── building_model-external/     # External schedule files (.csv, .sch)
├── modified_test.idf            # Generated IDF file
├── eplusout.csv                 # EnergyPlus output
├── eplustbl.htm                 # Summary table
└── ...                          # Other EnergyPlus outputs
```

## Parameter Sweeps

To run multiple configurations, use a shell loop:

```bash
# Sweep boiler efficiencies
for eff in constant cubic quadratic; do
    python3.9 -m experiments.thermostat.run_thermostat conventional_control --efficiency $eff
done

# Sweep all boiler configurations
for part in 0_0 0_2 0_4; do
    for eff in constant cubic quadratic; do
        python3.9 -m experiments.thermostat.run_thermostat conventional_control --part_load $part --efficiency $eff
    done
done

# Full parameter sweep (3 experiments × 2 modes × 3 periods × 3 part loads × 3 efficiencies = 162 runs)
for exp in conventional_control zonal_control occupancy_control; do
    for mode in validation evaluation; do
        for period in test beizaee lynch; do
            for part in 0_0 0_2 0_4; do
                for eff in constant cubic quadratic; do
                    python3.9 -m experiments.thermostat.run_thermostat $exp --mode $mode --period $period --part_load $part --efficiency $eff
                done
            done
        done
    done
done
```

## Configuration

Each experiment directory contains:
- `experiment.yaml`: Configuration defining all options and defaults
- `thermostat_control.idf`: Experiment-specific thermostat configuration
- `door_schedule.idf` (occupancy only): PIR-based door schedule
- `occupancy_schedule.idf` (occupancy only): PIR-based occupancy schedule

Shared components are in `shared/`:
- `people.idf`: Internal gains from people
- `equipment_gains.idf`: Internal gains from equipment
- `door_schedule.idf`: Beizaee diary-based door schedule
- `occupancy_schedule.idf`: Beizaee diary-based occupancy
- `boiler/part_load/`: Boiler minimum part load configurations
- `boiler/efficiency/`: Boiler efficiency curve configurations
- `run_period/`: Run period definitions

## Modifying Experiments

To add a new experiment:

1. Create directory: `experiments/thermostat/{experiment_name}/`
2. Add `experiment.yaml` (copy from existing and modify)
3. Add `thermostat_control.idf` with thermostat configuration
4. Add mode-specific schedules if needed

## Legacy Comparison

This replaces the old `run_exps.py` workflow:

**Old:**
```bash
python run_exps.py --experiment conventional_control --period lynch --part_load 0_2 --efficiency quadratic --mode validation
```

**New:**
```bash
python3.9 -m experiments.thermostat.run_thermostat conventional_control --period lynch --part_load 0_2 --efficiency quadratic --mode validation
```

The new system:
- Uses YAML for configuration (easier to modify)
- Modular IDF composition (base building + transforms)
- Consistent with coheat experiment structure
- Cleaner separation of concerns
