# analysis/thermostat/__init__.py

# Boiler analysis
from .boiler import (
    boiler_efficiency,
    daily_summary,
    calculate_kwh_per_day,
    fit_biquadratic,
    eta_func
)

# Control strategy comparison
from .comparison import (
    compare_room_temperatures,
    compute_temperature_deltas,
    compare_gas_consumption
)

# Plotting (import module for access to all plotting functions)
from . import plotting
