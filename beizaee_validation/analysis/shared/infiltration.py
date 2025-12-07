import numpy as np
import pandas as pd
from .constants import AIR_DENSITY, AIR_SPECIFIC_HEAT, DT_MIN_INFILTRATION


def compute_afn_infiltration_exfiltration(df, zone_links, zone_volumes, measured_inf, measured_exf, rho=1.2):
    """Compute infiltration/exfiltration ACH from AFN mass flow rates.

    Args:
        df: DataFrame with AFN mass flow rate columns
        zone_links: Dict mapping zones to their outdoor-facing surfaces
        zone_volumes: Dict of zone volumes (m³)
        measured_inf: Dict of measured infiltration rates (ACH)
        measured_exf: Dict of measured exfiltration rates (ACH)
        rho: Air density (kg/m³), default 1.2

    Returns:
        DataFrame: ACH comparison by zone (AFN, EnergyPlus, Measured)
    """
    ach_means = []
    for zone in zone_links.keys():
        vol = zone_volumes[zone.upper()]
        net_flow = pd.Series(0.0, index=df.index)
        for sf in zone_links[zone.upper()]:
            flow_21 = df[f"{sf.upper()}:AFN Linkage Node 2 to Node 1 Mass Flow Rate [kg/s](TimeStep)"]
            flow_12 = df[f"{sf.upper()}:AFN Linkage Node 1 to Node 2 Mass Flow Rate [kg/s](TimeStep)"]
            net_flow += (flow_21 - flow_12)
        infil_total = net_flow.clip(lower=0)
        exfil_total = (-net_flow).clip(lower=0)
        infil_ach = (infil_total / (rho * vol)) * 3600
        exfil_ach = (exfil_total / (rho * vol)) * 3600
        eplus_col = f"{zone.upper()}:AFN Zone Infiltration Air Change Rate [ach](TimeStep)"
        eplus_infil_ach = df[eplus_col] if eplus_col in df.columns else pd.Series(np.nan, index=df.index)
        ach_means.append({
            "Zone": zone.upper(),
            "AFN_infil": infil_ach.mean(),
            "AFN_exfil": exfil_ach.mean(),
            "Eplus_infil": eplus_infil_ach.mean(),
            "Measured_infil": measured_inf.get(zone.upper(), np.nan),
            "Measured_exfil": measured_exf.get(zone.upper(), np.nan)
        })
    return pd.DataFrame(ach_means).set_index("Zone").round(3)


def compute_infiltration_exfiltration_ach(df, zone_volumes, outdoor_temp_col='Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)',
                                         rho=None, cp=None, dt_min=None):
    """Compute infiltration/exfiltration ACH from energy balance.

    Calculates ACH from exfiltration heat transfer using:
        Q = rho * cp * m_dot * dT
    where:
        Q = exfiltration heat transfer rate (W)
        rho = air density (kg/m³)
        cp = air specific heat (J/kg·K)
        m_dot = mass flow rate (kg/s)
        dT = zone_temp - outdoor_temp (K)

    Then converts to ACH:
        ACH = (m_dot / (rho * V)) * 3600

    Args:
        df: DataFrame with AFN exfiltration heat transfer and temperature columns
        zone_volumes: Dict of zone volumes (m³)
        outdoor_temp_col: Column name for outdoor temperature
        rho: Air density (kg/m³). If None, uses AIR_DENSITY constant
        cp: Air specific heat (J/kg·K). If None, uses AIR_SPECIFIC_HEAT constant
        dt_min: Minimum temperature difference (°C). If None, uses DT_MIN_INFILTRATION constant

    Returns:
        DataFrame: ACH by zone calculated from energy balance
    """
    if rho is None:
        rho = AIR_DENSITY
    if cp is None:
        cp = AIR_SPECIFIC_HEAT
    if dt_min is None:
        dt_min = DT_MIN_INFILTRATION

    results = []

    for zone, volume in zone_volumes.items():
        # Find exfiltration heat transfer column
        exfil_heat_col = f"{zone}:AFN Zone Exfiltration Sensible Heat Transfer Rate [W](TimeStep)"

        if exfil_heat_col not in df.columns:
            continue

        # Find zone temperature column
        zone_temp_col = f"{zone}:Zone Mean Air Temperature [C](TimeStep)"
        if zone_temp_col not in df.columns:
            continue

        # Calculate temperature difference
        zone_temp = df[zone_temp_col]
        outdoor_temp = df[outdoor_temp_col]
        dt = zone_temp - outdoor_temp

        # Only calculate where dT > dt_min to avoid noise
        valid_dt = dt > dt_min

        # Calculate mass flow rate from heat balance: m_dot = Q / (cp * dT)
        exfil_heat = df[exfil_heat_col]  # W
        m_dot = np.where(valid_dt, exfil_heat / (cp * dt), 0)  # kg/s

        # Convert to ACH
        ach = (m_dot / (rho * volume)) * 3600  # air changes per hour

        results.append({
            'Zone': zone,
            'ACH_mean': ach.mean(),
            'ACH_std': ach.std()
        })

    return pd.DataFrame(results).set_index('Zone')