# analysis/shared/plotting.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compare_plot_multi(measured, modeled_dict, title="ACH Comparison"):
    """Create multi-model ACH comparison plot.

    Args:
        measured: Dict of measured ACH values by zone
        modeled_dict: Dict of dicts, {model_name: {zone: ACH}}
        title: Plot title

    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    zones = list(measured.keys())
    x = np.arange(len(zones))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot measured values
    measured_values = [measured[z] for z in zones]
    ax.scatter(x, measured_values, color='black', s=100, marker='o',
              label='Measured', zorder=3)

    # Plot modeled values
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i, (model_name, modeled) in enumerate(modeled_dict.items()):
        modeled_values = [modeled.get(z, 0) for z in zones]
        offset = (i - len(modeled_dict)/2 + 0.5) * width
        ax.bar(x + offset, modeled_values, width, label=model_name,
              alpha=0.7, color=colors[i % len(colors)])

    ax.set_xlabel('Zone')
    ax.set_ylabel('ACH')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(zones, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_zone_breakdown(breakdown, zones, title="Heat Flow Breakdown"):
    """Create stacked bar chart of heat flows by zone.

    Args:
        breakdown: Dict of DataFrames with heat flow breakdown by zone
        zones: List of zone names to plot
        title: Plot title

    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Aggregate heat flows across zones
    heat_sources = set()
    for zone in zones:
        if zone in breakdown:
            heat_sources.update(breakdown[zone].columns)

    heat_sources = sorted(list(heat_sources))

    # Prepare data for stacking
    x = np.arange(len(zones))
    bottoms = np.zeros(len(zones))

    colors = plt.cm.Set3(np.linspace(0, 1, len(heat_sources)))

    for i, source in enumerate(heat_sources):
        values = []
        for zone in zones:
            if zone in breakdown and source in breakdown[zone].columns:
                values.append(breakdown[zone][source].sum())
            else:
                values.append(0)

        ax.bar(x, values, bottom=bottoms, label=source, color=colors[i])
        bottoms += np.array(values)

    ax.set_xlabel('Zone')
    ax.set_ylabel('Heat Flow (W)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(zones, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_surface(df, surface_cols, title="Surface Heat Transfer"):
    """Plot time series of surface heat transfer.

    Args:
        df: DataFrame with surface columns
        surface_cols: List of column names to plot
        title: Plot title

    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    for col in surface_cols:
        if col in df.columns:
            ax.plot(df.index, df[col], label=col, alpha=0.7)

    ax.set_xlabel('Time')
    ax.set_ylabel('Heat Transfer Rate (W)')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    return fig, ax


def surface_timeseries(df, surfaces):
    """Helper function for surface-level time series plotting.

    Args:
        df: DataFrame with time series data
        surfaces: List of surface names

    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    surface_cols = [c for c in df.columns if any(s in c for s in surfaces)]
    return plot_surface(df, surface_cols)


def format_axis_dates(ax, days_interval=5):
    """Format x-axis with dates at specified interval.

    Args:
        ax: Matplotlib axis object
        days_interval: Interval between date labels (days)
    """
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=days_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def add_reference_markers(ax, reference_data, color='black', marker='o'):
    """Overlay reference data markers on existing plot.

    Args:
        ax: Matplotlib axis object
        reference_data: Dict or Series of reference values
        color: Marker color
        marker: Marker style
    """
    if isinstance(reference_data, dict):
        x_pos = range(len(reference_data))
        y_vals = list(reference_data.values())
    elif isinstance(reference_data, pd.Series):
        x_pos = range(len(reference_data))
        y_vals = reference_data.values
    else:
        return

    ax.scatter(x_pos, y_vals, color=color, marker=marker, s=100,
              zorder=3, edgecolors='white', linewidths=1.5)
