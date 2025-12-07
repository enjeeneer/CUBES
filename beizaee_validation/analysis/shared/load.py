import pandas as pd
from pathlib import Path
from glob import glob
from eppy.modeleditor import IDF
from .constants import ROOM_MAPPING


def load_idf(idf_path, idd_path):
    """Load an IDF file using eppy.

    Args:
        idf_path: Path to IDF file
        idd_path: Path to IDD file

    Returns:
        IDF object
    """
    IDF.setiddname(idd_path)
    return IDF(idf_path)


def load_run_csv(path):
    """Load EnergyPlus CSV output and parse datetime.

    Args:
        path: Path to eplusout.csv file

    Returns:
        DataFrame with datetime index
    """
    df = pd.read_csv(path)
    dt = df['Date/Time'].str.strip().str.replace('  ', ' ')
    df['datetime'] = pd.to_datetime('2013 ' + dt, format='%Y %m/%d %H:%M:%S', errors='coerce')
    return df.dropna(subset=['datetime']).set_index('datetime')


def clean_zone_columns(df, room_mapping=None):
    """Standardize zone column names from EnergyPlus format.

    Removes ':Zone Mean Air Temperature [C](TimeStep)' suffixes and applies
    room mapping if provided.

    Args:
        df: DataFrame with EnergyPlus column names
        room_mapping: Optional dict mapping zone names to room categories

    Returns:
        DataFrame with cleaned column names
    """
    if room_mapping is None:
        room_mapping = ROOM_MAPPING

    # Create column name mapping
    col_map = {}
    for col in df.columns:
        # Extract zone name from EnergyPlus format
        if ':Zone Mean Air Temperature' in col:
            zone = col.split(':')[0]
            if zone in room_mapping:
                col_map[col] = room_mapping[zone]
            else:
                col_map[col] = zone

    return df.rename(columns=col_map)


def load_simulation_results(base_path, experiment_path, run):
    """Load simulation results from experiment directory structure.

    Loads CSV files from all control types in the experiment directory,
    extracting temperature, gas, and baseboard columns.

    Args:
        base_path: Base directory for simulation runs
        experiment_path: Relative path to experiment (e.g., 'runs_validation/lynch')
        run: Run identifier

    Returns:
        dict: Dictionary mapping control types to DataFrames with columns:
            - Temperature columns for each zone
            - Gas consumption columns
            - Baseboard availability columns
    """
    results = {}
    run_dir = Path(base_path) / experiment_path

    # Find all result directories
    for result_path in run_dir.glob("*/eplusout.csv"):
        control_type = result_path.parent.name.split("__")[0]  # Extract control type from folder name

        # Load CSV
        df = load_run_csv(str(result_path))

        # Extract relevant columns
        temp_cols = [c for c in df.columns if 'Zone Mean Air Temperature' in c]
        gas_cols = [c for c in df.columns if 'NaturalGas' in c or 'Gas' in c]
        baseboard_cols = [c for c in df.columns if 'Baseboard' in c and 'Availability' in c]

        # Create result DataFrame with relevant columns
        result_df = df[temp_cols + gas_cols + baseboard_cols].copy()

        results[control_type] = result_df

    return results


def load_all_run_results(base_path, experiment_path, run, control_filter=None, room_mapping=None):
    """Load and clean simulation results for all or specific control types.

    Wrapper function that loads simulation results and applies column name
    standardization.

    Args:
        base_path: Base directory for simulation runs
        experiment_path: Relative path to experiment
        run: Run identifier
        control_filter: Optional control type to filter (e.g., 'conventional_control')
        room_mapping: Optional dict mapping zone names to room categories

    Returns:
        dict or DataFrame: If control_filter is None, returns dict of DataFrames
                          by control type. If control_filter is specified, returns
                          single DataFrame for that control type.
    """
    if room_mapping is None:
        room_mapping = ROOM_MAPPING

    # Load all results
    results = load_simulation_results(base_path, experiment_path, run)

    # Clean column names
    cleaned_results = {}
    for control_type, df in results.items():
        cleaned_results[control_type] = clean_zone_columns(df, room_mapping)

    # Apply filter if specified
    if control_filter:
        return cleaned_results.get(control_filter)

    return cleaned_results
