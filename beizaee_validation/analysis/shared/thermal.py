# analysis/shared/thermal.py

import pandas as pd
import numpy as np
from .constants import HLC_MIN, HLC_MAX, zone_volumes, FLOOR_AREAS, ROOM_MAPPING


def calc_HLC(df, outdoor_temp_col='Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'):
    """Calculate Heat Loss Coefficient from heating rate and temperature difference.

    HLC = Heating Rate / (Indoor Temp - Outdoor Temp)

    Values are clipped to the range [HLC_MIN, HLC_MAX] to exclude outliers.

    Args:
        df: DataFrame with heating and temperature columns
        outdoor_temp_col: Column name for outdoor temperature

    Returns:
        Series: Heat Loss Coefficient (W/K), clipped to valid range
    """
    # Find heating rate column (ideal loads or baseboard)
    heating_cols = [c for c in df.columns if 'Heating' in c and 'Rate' in c]
    if not heating_cols:
        raise ValueError("No heating rate column found in DataFrame")

    heating_rate = df[heating_cols[0]]  # W

    # Calculate indoor-outdoor temperature difference
    indoor_temp_cols = [c for c in df.columns if 'Zone Mean Air Temperature' in c]
    if not indoor_temp_cols:
        raise ValueError("No zone temperature columns found")

    # Use average indoor temperature
    indoor_temp = df[indoor_temp_cols].mean(axis=1)
    outdoor_temp = df[outdoor_temp_col]

    delta_t = indoor_temp - outdoor_temp

    # Calculate HLC, avoiding division by zero
    hlc = heating_rate / delta_t.replace(0, np.nan)

    # Clip to valid range
    return hlc.clip(lower=HLC_MIN, upper=HLC_MAX)


def weighted_temp(df, volumes=None):
    """Calculate zone-volume-weighted average building temperature.

    Args:
        df: DataFrame with zone temperature columns
        volumes: Optional dict of zone volumes. If None, uses zone_volumes constant

    Returns:
        Series: Volume-weighted average temperature across all zones
    """
    if volumes is None:
        volumes = zone_volumes

    # Find temperature columns
    temp_cols = [c for c in df.columns if 'Zone Mean Air Temperature' in c or c in volumes.keys()]

    total_volume = 0
    weighted_sum = 0

    for col in temp_cols:
        # Extract zone name
        if ':' in col:
            zone = col.split(':')[0]
        else:
            zone = col

        if zone in volumes:
            vol = volumes[zone]
            weighted_sum += df[col] * vol
            total_volume += vol

    return weighted_sum / total_volume if total_volume > 0 else pd.Series(0, index=df.index)


def get_floor_areas_by_beizaee_category(floor_areas=None, room_mapping=None):
    """Map zone floor areas to Beizaee categories with aggregation.

    Handles cases where multiple zones map to the same category
    (e.g., HALL_DOWNSTAIRS + HALL_UPSTAIRS → Circulation Areas).

    Args:
        floor_areas: Optional dict of floor areas by zone. If None, uses FLOOR_AREAS constant
        room_mapping: Optional dict mapping zones to categories. If None, uses ROOM_MAPPING constant

    Returns:
        dict: Floor areas aggregated by Beizaee category
    """
    if floor_areas is None:
        floor_areas = FLOOR_AREAS
    if room_mapping is None:
        room_mapping = ROOM_MAPPING

    category_areas = {}

    for zone, area in floor_areas.items():
        if zone in room_mapping:
            category = room_mapping[zone]

            if category in category_areas:
                category_areas[category] += area
            else:
                category_areas[category] = area

    return category_areas
