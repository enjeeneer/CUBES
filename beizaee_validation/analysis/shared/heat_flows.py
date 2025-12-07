# analysis/shared/heat_flows.py

import pandas as pd
import numpy as np


def extract_zone_afn(df, zone):
    """Extract AirflowNetwork (AFN) infiltration/exfiltration data for a zone.

    Args:
        df: DataFrame with EnergyPlus output columns
        zone: Zone name (e.g., 'FRONT_ROOM')

    Returns:
        DataFrame: AFN columns for the specified zone
    """
    afn_cols = [c for c in df.columns if zone.upper() in c and 'AFN' in c]
    return df[afn_cols]


def extract_zone_baseboard(df, zone):
    """Extract baseboard heating output for a zone.

    Args:
        df: DataFrame with EnergyPlus output columns
        zone: Zone name

    Returns:
        DataFrame: Baseboard columns for the specified zone
    """
    baseboard_cols = [c for c in df.columns if zone.upper() in c and 'Baseboard' in c]
    return df[baseboard_cols]


def extract_zone_fabric(df, zone):
    """Extract fabric heat losses for a zone grouped by construction type.

    Args:
        df: DataFrame with EnergyPlus output columns
        zone: Zone name

    Returns:
        DataFrame: Fabric heat transfer columns grouped by construction
    """
    fabric_cols = [c for c in df.columns if zone.upper() in c and
                  ('Surface' in c or 'Wall' in c or 'Floor' in c or 'Roof' in c) and
                  'Heat' in c]
    return df[fabric_cols]


def extract_zone_windows(df, zone):
    """Extract window heat losses for a zone.

    Args:
        df: DataFrame with EnergyPlus output columns
        zone: Zone name

    Returns:
        DataFrame: Window heat transfer columns
    """
    window_cols = [c for c in df.columns if zone.upper() in c and
                  ('Window' in c or 'Glazing' in c) and 'Heat' in c]
    return df[window_cols]


def get_surface_groups(idf):
    """Group all building surfaces by type.

    Args:
        idf: IDF object from eppy

    Returns:
        dict: Surface groups {'walls': [], 'roof': [], 'floor': [], 'windows': []}
    """
    groups = {
        'walls': [],
        'roof': [],
        'floor': [],
        'windows': []
    }

    # Group BuildingSurface:Detailed objects
    for surf in idf.idfobjects.get('BUILDINGSURFACE:DETAILED', []):
        surf_type = surf.Surface_Type.lower() if hasattr(surf, 'Surface_Type') else ''

        if 'wall' in surf_type:
            groups['walls'].append(surf.Name)
        elif 'roof' in surf_type or 'ceiling' in surf_type:
            groups['roof'].append(surf.Name)
        elif 'floor' in surf_type:
            groups['floor'].append(surf.Name)

    # Group FenestrationSurface:Detailed objects (windows)
    for surf in idf.idfobjects.get('FENESTRATIONSURFACE:DETAILED', []):
        groups['windows'].append(surf.Name)

    return groups


def get_zone_surface_groups(idf, zone):
    """Get surface groups for a specific zone.

    Args:
        idf: IDF object from eppy
        zone: Zone name

    Returns:
        dict: Surface groups for the specified zone
    """
    zone_groups = {
        'walls': [],
        'roof': [],
        'floor': [],
        'windows': []
    }

    # Filter BuildingSurface:Detailed by zone
    for surf in idf.idfobjects.get('BUILDINGSURFACE:DETAILED', []):
        if hasattr(surf, 'Zone_Name') and surf.Zone_Name == zone:
            surf_type = surf.Surface_Type.lower() if hasattr(surf, 'Surface_Type') else ''

            if 'wall' in surf_type:
                zone_groups['walls'].append(surf.Name)
            elif 'roof' in surf_type or 'ceiling' in surf_type:
                zone_groups['roof'].append(surf.Name)
            elif 'floor' in surf_type:
                zone_groups['floor'].append(surf.Name)

    # Filter FenestrationSurface:Detailed by zone
    for surf in idf.idfobjects.get('FENESTRATIONSURFACE:DETAILED', []):
        # Windows reference their building surface, need to check surface's zone
        if hasattr(surf, 'Building_Surface_Name'):
            building_surf_name = surf.Building_Surface_Name
            # Find the parent surface
            for bs in idf.idfobjects.get('BUILDINGSURFACE:DETAILED', []):
                if bs.Name == building_surf_name and hasattr(bs, 'Zone_Name') and bs.Zone_Name == zone:
                    zone_groups['windows'].append(surf.Name)
                    break

    return zone_groups


def compute_zone_heat_breakdown(df, zone, idf=None):
    """Decompose zone heat flows by source.

    Calculates heat contributions from:
    - Baseboard heating
    - Fabric losses (walls, floor, roof)
    - Windows
    - AFN infiltration/exfiltration
    - Solar gains
    - Internal radiation

    Args:
        df: DataFrame with EnergyPlus output columns
        zone: Zone name
        idf: Optional IDF object for surface grouping

    Returns:
        DataFrame: Heat flow breakdown with columns for each source
    """
    breakdown = pd.DataFrame(index=df.index)

    # Baseboard heating
    baseboard_cols = [c for c in df.columns if zone.upper() in c and 'Baseboard' in c and 'Rate' in c]
    if baseboard_cols:
        breakdown['Baseboard'] = df[baseboard_cols].sum(axis=1)

    # Fabric losses
    fabric_cols = [c for c in df.columns if zone.upper() in c and
                  ('Surface' in c or 'Wall' in c or 'Floor' in c or 'Roof' in c) and
                  ('Heat' in c or 'Loss' in c)]
    if fabric_cols:
        breakdown['Fabric'] = df[fabric_cols].sum(axis=1)

    # Windows
    window_cols = [c for c in df.columns if zone.upper() in c and
                  ('Window' in c or 'Glazing' in c) and 'Heat' in c]
    if window_cols:
        breakdown['Windows'] = df[window_cols].sum(axis=1)

    # AFN
    afn_cols = [c for c in df.columns if zone.upper() in c and 'AFN' in c and
               ('Heat' in c or 'Sensible' in c)]
    if afn_cols:
        breakdown['AFN'] = df[afn_cols].sum(axis=1)

    # Solar gains
    solar_cols = [c for c in df.columns if zone.upper() in c and 'Solar' in c]
    if solar_cols:
        breakdown['Solar'] = df[solar_cols].sum(axis=1)

    # Radiation
    rad_cols = [c for c in df.columns if zone.upper() in c and 'Radiation' in c]
    if rad_cols:
        breakdown['Radiation'] = df[rad_cols].sum(axis=1)

    return breakdown


def heat_flow_breakdown(df, zones, idf=None):
    """Calculate heat flow breakdown for multiple zones.

    Args:
        df: DataFrame with EnergyPlus output columns
        zones: List of zone names
        idf: Optional IDF object

    Returns:
        dict: Heat flow breakdown DataFrames by zone
    """
    breakdowns = {}
    for zone in zones:
        breakdowns[zone] = compute_zone_heat_breakdown(df, zone, idf)

    return breakdowns


def summarise_heat_flows(breakdown):
    """Aggregate heat flow breakdown results into summary statistics.

    Args:
        breakdown: DataFrame from compute_zone_heat_breakdown

    Returns:
        Series: Total heat flow by source (sum over time period)
    """
    return breakdown.sum()
