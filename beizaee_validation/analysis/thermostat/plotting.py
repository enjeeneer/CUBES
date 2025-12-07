# analysis/thermostat/plotting.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from analysis.shared.constants import (
    CONTROL_COLORS,
    BEIZAEE_REFERENCE,
    BEIZAEE_COCKROFT_DELTAS,
    FLOOR_AREAS,
    ROOM_MAPPING
)


def plot_boiler_config_comparison(results_dict, config_tag, room_mapping=None,
                                   beizaee_cockroft_deltas=None, run_label=None,
                                   controls=None, floor_areas=None):
    """
    Plots temperature differences between selected smart control(s) and conventional control.
    'Whole House' value is computed as floor-area-weighted mean aggregated by Beizaee room categories.

    EXACT REPLICATION from play.ipynb - maintains negative y-axis and Beizaee/Cockroft overlay.

    Args:
        results_dict: Dict of DataFrames by control type
        config_tag: Config string to filter results (e.g., "part_2__eff_quadratic")
        room_mapping: Zone to Beizaee category mapping
        beizaee_cockroft_deltas: Reference delta values
        run_label: Optional label for title
        controls: List of control types to compare
        floor_areas: Dict of floor areas by zone

    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    from analysis.thermostat.comparison import compare_room_temperatures

    if room_mapping is None:
        room_mapping = ROOM_MAPPING
    if floor_areas is None:
        floor_areas = FLOOR_AREAS
    if beizaee_cockroft_deltas is None:
        beizaee_cockroft_deltas = BEIZAEE_COCKROFT_DELTAS

    # === Extract relevant runs ===
    filtered = {k: v for k, v in results_dict.items() if config_tag in k}
    if not filtered:
        print(f"⚠️ Warning: No configurations found for {config_tag}.")
        return

    # === Compute temperature means during baseboard heating ===
    df_compare, _ = compare_room_temperatures(filtered, room_mapping, baseboard_filter=1)
    df_compare.columns = [col.split('__')[0] for col in df_compare.columns]

    # === Controls ===
    if controls is None:
        controls = ["zonal_control", "occupancy_control"]
    if "conventional_control" not in controls:
        controls = ["conventional_control"] + controls

    for control in controls:
        if control not in df_compare.columns:
            df_compare[control] = pd.Series([0] * len(df_compare), index=df_compare.index)

    # === Aggregate floor areas by Beizaee room categories ===
    beizaee_floor_areas = {}
    for zone, area in floor_areas.items():
        cat = room_mapping.get(zone, zone)
        beizaee_floor_areas[cat] = beizaee_floor_areas.get(cat, 0) + area

    total_area = sum(beizaee_floor_areas.values())

    # === Recompute 'Whole House' weighted average for each control ===
    for control in controls:
        weighted_sum = 0.0
        for cat, area in beizaee_floor_areas.items():
            if cat in df_compare.index:
                weighted_sum += df_compare.loc[cat, control] * area
        whole_house_val = weighted_sum / total_area
        if "Whole House" not in df_compare.index:
            df_compare.loc["Whole House"] = 0
        df_compare.loc["Whole House", control] = whole_house_val

    # === Compute deltas (Conventional - Smart) ===
    delta_dict = {
        ctrl: df_compare[ctrl] - df_compare["conventional_control"]
        for ctrl in controls if ctrl != "conventional_control"
    }


    # === Sort zones by delta magnitude ===
    first_delta = next(iter(delta_dict.values()))
    sort_order = first_delta.drop("Whole House", errors="ignore").sort_values().index.tolist()
    if "Whole House" in first_delta:
        sort_order.append("Whole House")

    x = np.arange(len(sort_order))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))

    offsets = {"zonal_control": -width / 2, "occupancy_control": width / 2}
    colors = {"zonal_control": "tab:blue", "occupancy_control": "tab:orange"}
    labels = {"zonal_control": "Conv - Zonal", "occupancy_control": "Conv - Occupancy"}

    # Plot bars for simulated deltas
    for control in delta_dict:
        delta = delta_dict[control]
        ax.bar(x + offsets.get(control, 0), delta.loc[sort_order], width=width,
               label=f"{labels.get(control, control)} (Simulated)",
               color=colors.get(control, "gray"))

    # === Overlay Beizaee & Cockroft references ===
    if beizaee_cockroft_deltas:
        for label, color, marker in [("Beizaee", "black", "o"), ("Cockroft", "darkgreen", "s")]:
            values = {k: v[label] for k, v in beizaee_cockroft_deltas.items() if k in sort_order and label in v}
            idxs = [i for i, room in enumerate(sort_order) if room in values]
            y_vals = [values[room] for room in sort_order if room in values]
            ax.scatter([x[i] for i in idxs], y_vals, color=color, marker=marker,
                      label=f"Conv - Zonal ({label})", s=100, zorder=5)
            # Add value labels
            for i, val in zip(idxs, y_vals):
                ax.text(x[i], val + 0.1, f"{val:.1f}°C", ha='center', va='bottom',
                       fontsize=9, color=color)

    # Formatting
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sort_order, rotation=45, ha='right')
    ax.set_ylabel("ΔT (Conventional - Zonal Control) [°C]")
    ax.legend(title="Control Type", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

    if run_label:
        ax.set_title(f"Temperature Differences: {run_label}")

    plt.tight_layout()
    return fig, ax


def plot_absolute_temperatures_with_beizaee_period(results_dict, config_tag, room_mapping=None,
                                                   beizaee_full_df=None, period="WholeDay",
                                                   controls=None, baseboard_only=None,
                                                   run_label=None, floor_areas=None):
    """
    Plot absolute temperatures vs Beizaee reference for a specific heating period.

    EXACT REPLICATION from play.ipynb - filters simulation data by period to match Beizaee reference.

    Args:
        results_dict: Dict of DataFrames by control type
        config_tag: Config string to filter results
        room_mapping: Zone to Beizaee category mapping
        beizaee_full_df: DataFrame with columns [Room, Period, CC, ZC] for reference data
        period: Heating period ("WholeDay", "HeatingOn", "HeatingOff", "Occupied", "Unoccupied")
        controls: List of control types to plot
        baseboard_only: Deprecated - period parameter now controls filtering
        run_label: Optional label for title
        floor_areas: Dict of floor areas by zone

    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    from analysis.thermostat.comparison import compare_room_temperatures

    if room_mapping is None:
        room_mapping = ROOM_MAPPING
    if floor_areas is None:
        floor_areas = FLOOR_AREAS
    if controls is None:
        controls = ["conventional_control", "zonal_control"]

    # === Map period to baseboard filter ===
    # This ensures simulation data is filtered to match the Beizaee period definition
    period_to_baseboard = {
        "WholeDay": None,        # All times
        "HeatingOn": 1,          # When baseboard is active
        "HeatingOff": 0,         # When baseboard is inactive
        "Occupied": 1,           # During occupied hours (typically with heating)
        "Unoccupied": 0,         # During unoccupied hours (typically no heating)
    }

    # Use period to determine filter, unless baseboard_only explicitly provided
    if baseboard_only is None:
        baseboard_filter = period_to_baseboard.get(period, None)
    else:
        baseboard_filter = baseboard_only

    # === Extract matching runs ===
    filtered = {k: v for k, v in results_dict.items() if config_tag in k}
    if not filtered:
        print(f"⚠️ Warning: No configurations found for {config_tag}")
        return

    # === Compute room temps with period-aligned filtering ===
    df_compare, _ = compare_room_temperatures(filtered, room_mapping, baseboard_filter=baseboard_filter)
    df_compare.columns = [col.split('__')[0] for col in df_compare.columns]

    # === Aggregate floor areas by Beizaee room category ===
    beizaee_floor_areas = {}
    for zone, area in floor_areas.items():
        cat = room_mapping.get(zone, zone)
        beizaee_floor_areas[cat] = beizaee_floor_areas.get(cat, 0) + area
    total_area = sum(beizaee_floor_areas.values())

    # === Recompute Whole House weighted averages ===
    for control in controls:
        if control not in df_compare.columns:
            continue
        weighted_sum = 0.0
        for cat, area in beizaee_floor_areas.items():
            if cat in df_compare.index:
                weighted_sum += df_compare.loc[cat, control] * area
        whole_house_val = weighted_sum / total_area
        if "Whole House" not in df_compare.index:
            df_compare.loc["Whole House"] = 0
        df_compare.loc["Whole House", control] = whole_house_val

    # === Prepare plotting dataframe ===
    df_plot = df_compare[[c for c in controls if c in df_compare.columns]].copy()

    # Add Beizaee reference values if provided
    if beizaee_full_df is not None:
        period_df = beizaee_full_df[beizaee_full_df["Period"] == period].set_index("Room")
        for control in controls:
            if control not in df_compare.columns:
                continue
            label = "CC" if control == "conventional_control" else "ZC"
            if label in period_df.columns:
                df_plot[f"Beizaee_{control}"] = period_df[label]
    else:
        # Use BEIZAEE_REFERENCE constant if no dataframe provided
        for control in controls:
            if control not in df_compare.columns:
                continue
            beizaee_col = []
            for room in df_plot.index:
                if room in BEIZAEE_REFERENCE and period in BEIZAEE_REFERENCE[room]:
                    # Index: 0=zonal, 1=conventional
                    idx = 1 if control == "conventional_control" else 0
                    val = BEIZAEE_REFERENCE[room][period][idx]
                    beizaee_col.append(val)
                else:
                    beizaee_col.append(np.nan)
            df_plot[f"Beizaee_{control}"] = beizaee_col

    # Ensure Whole House is last row
    if "Whole House" in df_plot.index:
        ordered_idx = [idx for idx in df_plot.index if idx != "Whole House"] + ["Whole House"]
        df_plot = df_plot.loc[ordered_idx]

    # === Plot ===
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(df_plot.index))
    width = 0.3

    active_controls = [c for c in controls if c in df_compare.columns]

    for i, control in enumerate(active_controls):
        bar_pos = x + i * width - width * len(active_controls) / 2 + width / 2
        label_name = control.replace('_control', '').capitalize()
        ax.bar(bar_pos, df_plot[control], width=width,
              label=f"{label_name} (Sim)", color=f"C{i}", alpha=0.7)

        # Overlay Beizaee reference
        beizaee_col = f"Beizaee_{control}"
        if beizaee_col in df_plot.columns:
            ax.scatter(bar_pos, df_plot[beizaee_col], color="black", marker="o",
                      label=f"{label_name} (Beizaee)" if i == 0 else "", zorder=5, s=80)

            # Label points
            for xi, temp in zip(bar_pos, df_plot[beizaee_col]):
                if pd.notna(temp):
                    ax.text(xi, temp + 0.3, f"{temp:.1f}", ha='center', va='bottom',
                           fontsize=8, color="black")

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot.index, rotation=45, ha='right')
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(f"Absolute Temperatures: {period}" + (f" ({run_label})" if run_label else ""))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_temp_differences_with_beizaee(sim_deltas, beizaee_cockroft_deltas=None):
    """Compare simulated deltas against Beizaee and Cockroft references.

    Args:
        sim_deltas: Series of simulated temperature differences
        beizaee_cockroft_deltas: Optional reference data dict

    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    if beizaee_cockroft_deltas is None:
        beizaee_cockroft_deltas = BEIZAEE_COCKROFT_DELTAS

    fig, ax = plt.subplots(figsize=(12, 6))

    rooms = list(sim_deltas.index)
    x = np.arange(len(rooms))
    width = 0.25

    # Simulated values
    ax.bar(x - width, sim_deltas.values, width, label='Simulated', alpha=0.8, color='steelblue')

    # Beizaee reference
    beizaee_vals = [beizaee_cockroft_deltas.get(r, {}).get('Beizaee', 0) for r in rooms]
    ax.bar(x, beizaee_vals, width, label='Beizaee', alpha=0.8, color='orange')

    # Cockroft reference
    cockroft_vals = [beizaee_cockroft_deltas.get(r, {}).get('Cockroft', 0) for r in rooms]
    ax.bar(x + width, cockroft_vals, width, label='Cockroft', alpha=0.8, color='green')

    ax.set_xlabel('Room')
    ax.set_ylabel('ΔT: Conventional - Zonal (°C)')
    ax.set_title('Temperature Difference Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(rooms, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_gas_consumption_by_config(results_dict,
                                   base_title="Gas Use and Relative Savings by Control & Boiler Config",
                                   controls=None,
                                   show_references=True):
    """
    Plots gas consumption comparison across different boiler configurations.

    EXACT REPLICATION from play.ipynb - compares part-load ratios and efficiency curves.

    Creates 2-panel plot:
    - Absolute daily gas use (kWh/day)
    - Relative % reduction vs conventional
    Optional: overlay Beizaee and Cockroft reference values.

    Args:
        results_dict: Dict of DataFrames by run name (e.g., "conventional_control__part_0_2__eff_quadratic")
        base_title: Title for the figure
        controls: List of control types to include (filters by run name prefix)
        show_references: Whether to overlay Beizaee/Cockroft reference points

    Returns:
        DataFrame: Energy consumption by config and control type
    """

    def calculate_kwh_per_day(gas_series, timesteps_per_hour=6):
        """Local helper to calculate kWh/day from gas Series."""
        total_mj = gas_series.sum() / 1e6
        total_days = len(gas_series) / (timesteps_per_hour * 24)
        return (total_mj * 0.277777778) / total_days

    # === Colour config ===
    control_colours = {
        "Conventional": "grey",
        "Zonal": "lightblue",
        "Occupancy": "purple"
    }

    # === Reference values ===
    ref_points = {
        "Beizaee": {"Conventional": 62.4, "Zonal": 53.6},
        "Cockroft": {"Conventional": 62.6, "Zonal": 55.2},
    }
    highlight_config = "Part-load: 0_2 Eff. Curve: quadratic"

    # === Build DataFrame of daily energy use ===
    energy_data = {}
    for name, df in results_dict.items():
        control = name.split("__")[0]
        if controls and control not in controls:
            continue
        part = name.split("__part_")[1].split("__")[0]
        eff = name.split("__eff_")[1]
        config = f"Part-load: {part} Eff. Curve: {eff}"

        # Find gas column
        gas_cols = [c for c in df.columns if 'Gas' in c and 'Energy' in c] or \
                   [c for c in df.columns if c == 'Gas']
        if gas_cols:
            kwh = calculate_kwh_per_day(df[gas_cols[0]])
        else:
            kwh = 0

        if config not in energy_data:
            energy_data[config] = {}
        energy_data[config][control] = kwh

    df_energy = pd.DataFrame(energy_data).T.sort_index()
    df_energy = df_energy[[c for c in ["conventional_control", "zonal_control", "occupancy_control"] if c in df_energy.columns]]
    df_energy.columns = [c.replace("_control", "").capitalize() for c in df_energy.columns]

    # === Sort the configurations (custom order) ===
    custom_order = [
        "Part-load: 0_2 Eff. Curve: quadratic",
        # Add more configs as needed
    ]
    # Only keep configs that exist in the data
    custom_order = [c for c in custom_order if c in df_energy.index]
    if custom_order:
        df_energy = df_energy.loc[custom_order]

    # === Calculate relative savings ===
    if "Conventional" in df_energy.columns:
        baseline = df_energy["Conventional"]
        df_relative = (1 - df_energy.divide(baseline, axis=0)) * 100
    else:
        df_relative = None

    # === Plot ===
    fig, axes = plt.subplots(nrows=2, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1]})

    # Absolute gas use
    control_order = ["Conventional", "Zonal", "Occupancy"]
    available_controls = [c for c in control_order if c in df_energy.columns]
    df_energy[available_controls].plot(kind="bar", ax=axes[0],
                                       color=[control_colours[c] for c in available_controls])
    axes[0].set_ylabel("Daily Gas Use (kWh/day)")
    axes[0].set_title("Absolute Gas Consumption")
    axes[0].legend(title="Control Type", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    # Reference dots (optional)
    if show_references:
        for i, config in enumerate(df_energy.index):
            axes[0].scatter(i - 0.1, ref_points["Beizaee"]["Conventional"], color="black", marker="o", label="Beizaee — Conv", s=100)
            axes[0].scatter(i + 0.1, ref_points["Beizaee"]["Zonal"], facecolors='none', edgecolors='black', marker="o", label="Beizaee — Zonal", s=100)
            axes[0].scatter(i - 0.1, ref_points["Cockroft"]["Conventional"], color="grey", marker="o", label="Cockroft — Conv", s=100)
            axes[0].scatter(i + 0.1, ref_points["Cockroft"]["Zonal"], facecolors='none', edgecolors='grey', marker="o", label="Cockroft — Zonal", s=100)

    # Relative savings
    if df_relative is not None:
        rel_controls = [c for c in control_order if c in df_relative.columns and c != "Conventional"]
        df_relative_subset = df_relative[rel_controls]
        df_relative_subset.plot(kind="bar", ax=axes[1],
                                color=[control_colours[c] for c in df_relative_subset.columns])
        axes[1].axhline(0, color="grey", linewidth=1)
        axes[1].set_ylabel("Relative Saving vs Conventional (%)")
        axes[1].set_title("Relative Energy Reduction")
        axes[1].grid(axis="y", linestyle="--", alpha=0.3)
        axes[1].get_legend().set_visible(False)

        if show_references:
            for i, config in enumerate(df_relative.index):
                rel_b = (1 - ref_points["Beizaee"]["Zonal"] / ref_points["Beizaee"]["Conventional"]) * 100
                rel_c = (1 - ref_points["Cockroft"]["Zonal"] / ref_points["Cockroft"]["Conventional"]) * 100
                axes[1].scatter(i, rel_b, facecolors='none', edgecolors='black', marker='o', label="Beizaee ΔT", s=100)
                axes[1].scatter(i, rel_c, facecolors='none', edgecolors='grey', marker='o', label="Cockroft ΔT", s=100)

    # X-axis labels
    labels = [f"Part-load: {config.split(' ')[1].replace('_', '.')}" for config in df_energy.index]
    axes[1].set_xticklabels(labels, rotation=0)

    # Efficiency labels between bars
    eff_labels, x_coords = [], []
    for i, config in enumerate(df_energy.index):
        if i % 2 == 0:
            if "quadratic" in config:
                eff_labels.append("Curve: quadratic")
            elif "constant" in config:
                eff_labels.append("Curve: constant")
            else:
                eff_labels.append("Curve: cubic")
            x_coords.append(i + 0.5)
    for i, label in zip(x_coords, eff_labels):
        axes[1].text(i, -2, label, ha='center', va='top', fontsize=10, color='black')

    fig.suptitle(base_title, fontsize=14)

    # Deduplicated legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 0.3), loc='upper left', frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return df_energy.round(2)


def plot_daily_gas_consumption(results_dict, control_colors=None,
                               fill_alpha=0.15, show_markers=True, marker_size=4):
    """Plot time series of daily gas consumption by control type with filled areas.

    REWRITTEN to match notebook style - accepts results dict and calculates daily gas automatically.

    Args:
        results_dict: Dict of {control_type: DataFrame with gas column}
                     e.g., {"conventional_control": df_conv, "zonal_control": df_zonal}
        control_colors: Optional color mapping dict
        fill_alpha: Alpha for filled area under curves (0-1)
        show_markers: Whether to show markers on line plots
        marker_size: Size of markers if shown

    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    if control_colors is None:
        control_colors = CONTROL_COLORS

    fig, ax = plt.subplots(figsize=(14, 6))

    # Calculate daily gas consumption for each control type
    for control_type, df in results_dict.items():
        # Find gas column
        gas_cols = [c for c in df.columns if 'Gas' in c and 'Energy' in c] or \
                   [c for c in df.columns if c == 'Gas']

        if not gas_cols:
            print(f"Warning: No gas column found for {control_type}")
            continue

        # Calculate daily consumption (J -> kWh/day)
        gas_joules = df[gas_cols[0]]
        daily_gas_joules = gas_joules.resample('D').sum()
        gas_kwh_per_day = daily_gas_joules / 3.6e6  # J to kWh

        # Get color for this control type
        color = control_colors.get(control_type, 'steelblue')
        label = control_type.replace('_control', '').replace('_', ' ').title()

        # Plot line with optional markers
        if show_markers:
            ax.plot(gas_kwh_per_day.index, gas_kwh_per_day.values,
                   color=color, linewidth=2, marker='o', markersize=marker_size,
                   label=label, alpha=0.9)
        else:
            ax.plot(gas_kwh_per_day.index, gas_kwh_per_day.values,
                   color=color, linewidth=2, label=label, alpha=0.9)

        # Fill area under curve
        ax.fill_between(gas_kwh_per_day.index, gas_kwh_per_day.values,
                       alpha=fill_alpha, color=color)

    ax.set_xlabel('Date')
    ax.set_ylabel('Gas Consumption (kWh/day)')
    ax.set_title('Daily Gas Consumption by Control Strategy')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig, ax


def plot_simulation_day(results_dict, day, control_colors=None):
    """Plot temperatures for a single day across control types.

    Args:
        results_dict: Dict of DataFrames by control type
        day: Datetime or date string to plot
        control_colors: Optional color mapping

    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    if control_colors is None:
        control_colors = CONTROL_COLORS

    fig, ax = plt.subplots(figsize=(14, 6))

    for control_type, df in results_dict.items():
        # Filter to specific day
        day_data = df[df.index.date == pd.to_datetime(day).date()]

        # Get mean temperature across zones
        temp_cols = [c for c in day_data.columns if 'Temperature' in c]
        if temp_cols:
            mean_temp = day_data[temp_cols].mean(axis=1)
            color = control_colors.get(control_type, 'gray')
            ax.plot(day_data.index, mean_temp, label=control_type,
                   color=color, linewidth=2)

    ax.set_xlabel('Time')
    ax.set_ylabel('Mean Building Temperature (°C)')
    ax.set_title(f'Temperature Comparison: {day}')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_loop_temperatures(df, control_type):
    """Visualize boiler loop temperatures.

    Args:
        df: DataFrame with boiler loop temperature columns
        control_type: Control strategy name

    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Find loop temperature columns
    loop_cols = [c for c in df.columns if 'Loop' in c and 'Temperature' in c]

    for col in loop_cols:
        ax.plot(df.index, df[col], label=col, alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'Boiler Loop Temperatures: {control_type}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    return fig, ax
