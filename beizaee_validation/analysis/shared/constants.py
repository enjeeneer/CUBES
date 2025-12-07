# ============================================================================
# PHYSICAL PROPERTIES
# ============================================================================

# Air properties for infiltration/exfiltration calculations
AIR_DENSITY = 1.2  # kg/m³
AIR_SPECIFIC_HEAT = 1005  # J/kg·K
DT_MIN_INFILTRATION = 2  # °C (minimum ΔT for ACH calculation to avoid noise)

# Energy conversions
MJ_TO_KWH = 0.277777778  # 1 MJ ≈ 0.2778 kWh
J_TO_MJ = 1e-6  # 1 J = 1e-6 MJ

# ============================================================================
# BUILDING GEOMETRY
# ============================================================================

# Zone volumes (m³)
zone_volumes = {
    'HALL_DOWNSTAIRS': 19.66,
    'FRONT_ROOM': 39.31,
    'KITCHEN': 31.75,
    'BACKROOM': 31.75,
    'BEDROOM_1': 32.21,
    'BEDROOM_2': 34.69,
    'BEDROOM_3': 19.21,
    'HALL_UPSTAIRS': 16.46,
    'BATHROOM': 19.89,
    'SUBFLOOR': 8.75,
    'LOFT': 50.30,
}

# Floor areas by zone (m²)
FLOOR_AREAS = {
    'HALL_DOWNSTAIRS': 7.02,
    'FRONT_ROOM': 14.04,
    'KITCHEN': 11.34,
    'BACKROOM': 11.34,
    'BEDROOM_3': 6.86,
    'BEDROOM_1': 11.51,
    'HALL_UPSTAIRS': 4.00,
    'BATHROOM': 7.11,
    'BEDROOM_2': 12.39
}

# ============================================================================
# MEASURED INFILTRATION/EXFILTRATION (from blower door tests)
# ============================================================================

# Measured infiltration rates by zone (ACH)
measured_inf = {
    "FRONT_ROOM": 0.23,
    "BACKROOM": 0.55,
    "KITCHEN": 1.65,
    "HALL_DOWNSTAIRS": 2.02,
    "HALL_UPSTAIRS": 0.02,
    "BEDROOM_1": 0.0,
    "BEDROOM_2": 0.0,
    "BEDROOM_3": 0.15,
    "BATHROOM": 0.02,
    "SUBFLOOR": 21.0,
    "LOFT": 0.0
}

# Measured exfiltration rates by zone (ACH)
measured_exf = {
    "FRONT_ROOM": 0.37,
    "BACKROOM": 0.05,
    "KITCHEN": 0.25,
    "HALL_DOWNSTAIRS": 0.18,
    "HALL_UPSTAIRS": 1.28,
    "BEDROOM_1": 0.6,
    "BEDROOM_2": 1.7,
    "BEDROOM_3": 2.15,
    "BATHROOM": 1.88,
    "SUBFLOOR": 0,
    "LOFT": 2.1
}

# ============================================================================
# ROOM MAPPING
# ============================================================================

# Zone to Beizaee category mapping
# Maps EnergyPlus zone names to Beizaee reference study room categories
ROOM_MAPPING = {
    'FRONT_ROOM': 'Living Room',
    'BACKROOM': 'Dining Room',
    'BEDROOM_1': 'Bedroom 1',
    'BEDROOM_2': 'Bedroom 2',
    'BEDROOM_3': 'Unoccupied Room',
    'BATHROOM': 'Bathroom',
    'KITCHEN': 'Kitchen',
    'HALL_DOWNSTAIRS': 'Circulation Areas',
    'HALL_UPSTAIRS': 'Circulation Areas'
}

# ============================================================================
# VALIDATION REFERENCE DATA (from Beizaee study)
# ============================================================================

# Beizaee reference temperatures by period (°C)
# Structure: BEIZAEE_REFERENCE[room][period] = (zonal_temp, conventional_temp)
BEIZAEE_REFERENCE = {
    'Living Room': {
        'WholeDay': (19.2, 20.0),
        'HeatingOn': (20.3, 21.5),
        'Occupied': (22.3, 22.5),
        'Unoccupied': (18.7, 20.5),
        'HeatingOff': (18.0, 18.4)
    },
    'Dining Room': {
        'WholeDay': (18.2, 18.7),
        'HeatingOn': (19.0, 19.5),
        'Occupied': (20.4, 20.1),
        'Unoccupied': (18.8, 19.4),
        'HeatingOff': (17.4, 17.7)
    },
    'Bedroom 1': {
        'WholeDay': (18.0, 18.3),
        'HeatingOn': (18.9, 19.2),
        'Occupied': (18.9, 19.2),
        'Unoccupied': (18.7, 19.4),
        'HeatingOff': (17.1, 17.3)
    },
    'Bedroom 2': {
        'WholeDay': (17.2, 18.2),
        'HeatingOn': (17.6, 19.1),
        'Occupied': (16.3, 18.1),
        'Unoccupied': (17.9, 19.3),
        'HeatingOff': (16.5, 17.1)
    },
    'Bathroom': {
        'WholeDay': (16.5, 17.7),
        'HeatingOn': (17.3, 18.9),
        'Occupied': (19.7, 19.1),
        'Unoccupied': (17.2, 18.9),
        'HeatingOff': (15.5, 16.4)
    },
    'Unoccupied Room': {
        'WholeDay': (14.8, 15.3),
        'HeatingOn': (14.9, 15.5),
        'Occupied': (None, None),  # Not occupied
        'Unoccupied': (14.9, 15.5),
        'HeatingOff': (14.6, 15.0)
    },
    'Circulation Areas': {
        'WholeDay': (19.1, 19.5),
        'HeatingOn': (20.3, 20.8),
        'Occupied': (None, None),  # Not specifically occupied
        'Unoccupied': (20.3, 20.8),
        'HeatingOff': (17.8, 18.1)
    },
    'Kitchen': {
        'WholeDay': (19.6, 20.0),
        'HeatingOn': (20.7, 21.2),
        'Occupied': (23.0, 23.6),
        'Unoccupied': (20.4, 20.8),
        'HeatingOff': (18.4, 18.6)
    },
    'Whole House': {
        'WholeDay': (18.1, 18.7),
        'HeatingOn': (18.9, 19.7),
        'Occupied': (19.7, 20.1),
        'Unoccupied': (18.6, 19.6),
        'HeatingOff': (17.1, 17.5)
    }
}

# Beizaee vs Cockroft temperature delta references (Conventional - Zonal, °C)
BEIZAEE_COCKROFT_DELTAS = {
    "Living room": {"Beizaee": 1.193, "Cockroft": 1.230},
    "Dining room": {"Beizaee": 0.496, "Cockroft": 0.733},
    "Kitchen": {"Beizaee": 0.496, "Cockroft": 0.348},
    "Circulation areas": {"Beizaee": 0.496, "Cockroft": 0.141},
    "Bedroom 2": {"Beizaee": 1.496, "Cockroft": 0.822},
    "Unoccupied Room": {"Beizaee": 0.600, "Cockroft": 0.289},
    "Bedroom 1": {"Beizaee": 0.296, "Cockroft": 0.363},
    "Bathroom": {"Beizaee": 1.593, "Cockroft": 1.807},
    "Whole House": {"Beizaee": 0.800, "Cockroft": 0.667}
}

# Normalise keys for consistency
BEIZAEE_COCKROFT_DELTAS = {
    k.title(): {
        "Beizaee": -v["Beizaee"],
        "Cockroft": -v["Cockroft"]
    }
    for k, v in BEIZAEE_COCKROFT_DELTAS.items()
}

# ============================================================================
# BOILER CALIBRATION DATA
# ============================================================================

# Viessmann boiler efficiency calibration points (HHV basis)
# Format: [Part Load Ratio, Flow Temperature (°C), Efficiency]
# Source: Viessmann manufacturer data
BOILER_EFFICIENCY_CALIBRATION = [
    [1.0, 80, 0.90],   # 75/60°C return (non-condensing)
    [0.33, 80, 0.91],
    [1.0, 60, 0.95],   # 60/40°C return (transition)
    [0.33, 60, 0.97],
    [1.0, 40, 1.03],   # 40/30°C return (condensing mode)
    [0.33, 40, 1.05]
]

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================

# Control strategy colors for consistent visualization
CONTROL_COLORS = {
    "Conventional": "grey",
    "Zonal": "lightblue",
    "Occupancy": "purple"
}

# ============================================================================
# ANALYSIS THRESHOLDS
# ============================================================================

# Heat Loss Coefficient (HLC) valid range (W/K)
# Values outside this range are considered outliers
HLC_MIN = 10
HLC_MAX = 800
