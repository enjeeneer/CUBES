# analysis/thermostat/comparison.py

import pandas as pd
import numpy as np
from analysis.shared.constants import ROOM_MAPPING, FLOOR_AREAS


def compare_room_temperatures(results_dict, room_mapping=None, floor_areas=None, baseboard_filter=None):
    """Compare temperatures across control types with floor-area weighting.

    EXACT REPLICATION from play.ipynb - matches original signature and behavior.

    Args:
        results_dict: Dict of DataFrames by control type
        room_mapping: Optional zone→category mapping. If None, uses ROOM_MAPPING constant
        floor_areas: Optional dict of floor areas by zone. If None, uses FLOOR_AREAS constant
        baseboard_filter: Optional filter for baseboard availability (0, 1, or None)

    Returns:
        (temp_df, energy_use)
    """
    if room_mapping is None:
        room_mapping = ROOM_MAPPING
    if floor_areas is None:
        floor_areas = FLOOR_AREAS

    results = {}
    energy_use = {}

    # Aggregate floor areas by category
    beizaee_floor_areas = {}
    for zone, area in floor_areas.items():
        cat = room_mapping.get(zone, zone)
        beizaee_floor_areas[cat] = beizaee_floor_areas.get(cat, 0) + area
    total_area = sum(beizaee_floor_areas.values())

    for label, df in results_dict.items():
        # ---- Energy use ----
        from analysis.thermostat.boiler import calculate_kwh_per_day
        gas_cols = [
            c for c in df.columns
            if "naturalgas" in c.lower() or ("gas" in c.lower() and "energy" in c.lower())
        ]
        if gas_cols:
            energy_use[label] = calculate_kwh_per_day(df)

        # ---- Baseboard filtering ----
        df_filtered = df.copy()
        if baseboard_filter is not None:
            baseboard_cols = [
                c for c in df.columns
                if "baseboard availability" in c.lower() and "schedule" in c.lower()
            ]
            if baseboard_cols:
                df_filtered = df[df_filtered[baseboard_cols[0]] == baseboard_filter]

        # ---- Zone temperatures ----
        zone_means = {}
        for zone in room_mapping.keys():
            zone_l = zone.lower()
            temp_cols = [
                c for c in df_filtered.columns
                if zone_l in c.lower()
                and "zone air temperature" in c.lower()
            ]
            if temp_cols:
                zone_means[zone] = df_filtered[temp_cols].mean().mean()

        # ---- Aggregate to categories ----
        category_means = pd.Series(zone_means)
        category_means.index = category_means.index.map(room_mapping)
        grouped = category_means.groupby(level=0).mean()

        # ---- Whole-house weighted T ----
        weighted_sum = 0.0
        for cat, area in beizaee_floor_areas.items():
            if cat in grouped.index:
                weighted_sum += grouped.loc[cat] * area
        grouped["Whole House"] = weighted_sum / total_area

        results[label] = grouped

    return pd.DataFrame(results).round(2), energy_use


def compute_temperature_deltas(conventional_df, smart_df, room_mapping=None):
    """Calculate temperature differences between conventional and smart controls.

    Delta = Conventional Temperature - Smart Control Temperature

    Args:
        conventional_df: DataFrame for conventional control
        smart_df: DataFrame for smart control (zonal or occupancy)
        room_mapping: Optional zone to category mapping

    Returns:
        Series: Temperature deltas by room category
    """
    if room_mapping is None:
        room_mapping = ROOM_MAPPING

    # Get mean temperatures for each control type
    conv_temps = {}
    smart_temps = {}

    for zone in room_mapping.keys():
        conv_cols = [c for c in conventional_df.columns if zone in c and 'Temperature' in c]
        smart_cols = [c for c in smart_df.columns if zone in c and 'Temperature' in c]

        if conv_cols and smart_cols:
            conv_temps[zone] = conventional_df[conv_cols].mean().mean()
            smart_temps[zone] = smart_df[smart_cols].mean().mean()

    # Map to categories
    deltas = {}
    for zone in room_mapping.keys():
        if zone in conv_temps and zone in smart_temps:
            category = room_mapping[zone]
            delta = conv_temps[zone] - smart_temps[zone]

            if category in deltas:
                deltas[category] = (deltas[category] + delta) / 2
            else:
                deltas[category] = delta

    return pd.Series(deltas).sort_values(ascending=False)


def compare_gas_consumption(results_dict, period_days=None):
    """Compare gas consumption across control types.

    Calculates total and daily average gas consumption, plus relative savings.

    Args:
        results_dict: Dict of DataFrames by control type
        period_days: Number of days in simulation period (for averaging)

    Returns:
        DataFrame: Gas consumption comparison with columns:
                  - Total (J)
                  - Daily Average (kWh/day)
                  - Savings vs Conventional (%)
    """
    from analysis.shared.constants import MJ_TO_KWH, J_TO_MJ

    gas_summary = {}

    for control_type, df in results_dict.items():
        # Find gas energy column
        gas_cols = [c for c in df.columns if ('NaturalGas' in c or 'Gas' in c) and 'Energy' in c]

        if not gas_cols:
            continue

        total_gas_j = df[gas_cols[0]].sum()

        # Convert to kWh
        total_gas_kwh = total_gas_j * J_TO_MJ * MJ_TO_KWH

        # Calculate daily average
        if period_days:
            daily_avg_kwh = total_gas_kwh / period_days
        else:
            # Estimate from data duration
            duration_days = (df.index[-1] - df.index[0]).total_seconds() / (24 * 3600)
            daily_avg_kwh = total_gas_kwh / duration_days if duration_days > 0 else 0

        gas_summary[control_type] = {
            'Total_J': total_gas_j,
            'Total_kWh': total_gas_kwh,
            'Daily_kWh': daily_avg_kwh
        }

    summary_df = pd.DataFrame(gas_summary).T

    # Calculate savings vs conventional
    if 'conventional_control' in summary_df.index:
        conventional_kwh = summary_df.loc['conventional_control', 'Daily_kWh']
        summary_df['Savings_pct'] = ((conventional_kwh - summary_df['Daily_kWh']) /
                                     conventional_kwh * 100)
    else:
        summary_df['Savings_pct'] = 0

    return summary_df
