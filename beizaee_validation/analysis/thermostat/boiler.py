# analysis/thermostat/boiler.py

import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from analysis.shared.constants import BOILER_EFFICIENCY_CALIBRATION, MJ_TO_KWH, J_TO_MJ


def boiler_efficiency(df):
    """Calculate real-time boiler efficiency from heating energy and gas consumption.

    Efficiency = Heating Energy / Gas Energy

    Args:
        df: DataFrame with boiler heating and gas energy columns

    Returns:
        Series: Boiler efficiency (dimensionless, typically 0.8-1.0 for condensing boilers)
    """
    # Find heating energy column
    heating_cols = [c for c in df.columns if 'Heating' in c and 'Energy' in c]
    if not heating_cols:
        raise ValueError("No heating energy column found")

    # Find gas energy column
    gas_cols = [c for c in df.columns if ('NaturalGas' in c or 'Gas' in c) and 'Energy' in c]
    if not gas_cols:
        raise ValueError("No gas energy column found")

    heating_energy = df[heating_cols[0]]  # J
    gas_energy = df[gas_cols[0]]  # J

    # Calculate efficiency, avoiding division by zero
    efficiency = heating_energy / gas_energy.replace(0, np.nan)

    return efficiency


def daily_summary(df):
    """Aggregate daily boiler performance metrics.

    Calculates daily totals for:
    - Heating energy (J)
    - Gas consumption (J)
    - Average efficiency

    Args:
        df: DataFrame with boiler columns and datetime index

    Returns:
        DataFrame: Daily summary with heating, gas, and efficiency columns
    """
    # Resample to daily
    daily = df.resample('D')

    # Find columns
    heating_cols = [c for c in df.columns if 'Heating' in c and 'Energy' in c]
    gas_cols = [c for c in df.columns if ('NaturalGas' in c or 'Gas' in c) and 'Energy' in c]

    if not heating_cols or not gas_cols:
        raise ValueError("Required boiler columns not found")

    summary = pd.DataFrame()
    summary['Heating_J'] = daily[heating_cols[0]].sum()
    summary['Gas_J'] = daily[gas_cols[0]].sum()

    # Daily average efficiency
    summary['Efficiency'] = summary['Heating_J'] / summary['Gas_J']

    return summary


def calculate_kwh_per_day(df):
    """Convert gas energy to kWh per day.

    Args:
        df: DataFrame with gas energy column (J)

    Returns:
        Series: Gas consumption in kWh/day
    """
    # Find gas energy column
    gas_cols = [c for c in df.columns if ('NaturalGas' in c or 'Gas' in c) and 'Energy' in c]
    if not gas_cols:
        raise ValueError("No gas energy column found")

    gas_j = df[gas_cols[0]]

    # Convert J → MJ → kWh
    gas_mj = gas_j * J_TO_MJ
    gas_kwh = gas_mj * MJ_TO_KWH

    # Resample to daily
    g=gas_kwh.resample('D')
    gas_kwh_per_day=g.sum().where(g.count()>0)


    return gas_kwh_per_day


def fit_biquadratic(calibration_data=None):
    """Fit biquadratic efficiency function to calibration data.

    Fits: eta = a + b*PLR + c*PLR² + d*T + e*T² + f*PLR*T

    where:
        PLR = Part Load Ratio (0-1)
        T = Flow Temperature (°C)
        eta = Efficiency (HHV basis)

    Args:
        calibration_data: List of [PLR, T, eta] points.
                         If None, uses BOILER_EFFICIENCY_CALIBRATION constant

    Returns:
        dict: Fitted coefficients {a, b, c, d, e, f}
    """
    if calibration_data is None:
        calibration_data = BOILER_EFFICIENCY_CALIBRATION

    # Convert to arrays
    data = np.array(calibration_data)
    plr = data[:, 0]
    temp = data[:, 1]
    eta = data[:, 2]

    # Build design matrix for least squares
    # eta = a + b*PLR + c*PLR² + d*T + e*T² + f*PLR*T
    X = np.column_stack([
        np.ones_like(plr),  # a
        plr,                # b
        plr**2,             # c
        temp,               # d
        temp**2,            # e
        plr * temp          # f
    ])

    # Solve least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(X, eta, rcond=None)

    return {
        'a': coeffs[0],
        'b': coeffs[1],
        'c': coeffs[2],
        'd': coeffs[3],
        'e': coeffs[4],
        'f': coeffs[5]
    }


def eta_func(coefficients):
    """Return callable efficiency function from fitted coefficients.

    Args:
        coefficients: Dict of coefficients {a, b, c, d, e, f}

    Returns:
        function: eta(PLR, T) that computes efficiency for given part load ratio and temperature
    """
    def efficiency(plr, temp):
        """Calculate boiler efficiency.

        Args:
            plr: Part load ratio (0-1)
            temp: Flow temperature (°C)

        Returns:
            float: Efficiency (HHV basis)
        """
        return (coefficients['a'] +
                coefficients['b'] * plr +
                coefficients['c'] * plr**2 +
                coefficients['d'] * temp +
                coefficients['e'] * temp**2 +
                coefficients['f'] * plr * temp)

    return efficiency
