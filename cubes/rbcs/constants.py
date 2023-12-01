"""constanst shared across rbc methods"""

zone_names = ["Living", "Bedroom"]

t_control_name = {
    "Living": "Living-Thermostat Dual SP Control-HEATING-EXT",
    "Bedroom": "Bedroom-Thermostat Dual SP Control-HEATING-EXT",
}

vent_control_name = {
    "Living": "Living-Ventilation-EXT",
    "Bedroom": "Bedroom-Ventilation-EXT",
}

charge_control_name = "Battery Charge Schedule-EXT"
discharge_control_name = "Battery Discharge Schedule-EXT"

utility_demand_target_control_name = "Utility Demand Target Schedule-EXT"

t_out_name = "Site Outdoor Air Drybulb Temperature(Environment)"
humidity_out_name = "Site Outdoor Air Relative Humidity(Environment)"
t_name = {
    "Living": "Zone Air Temperature(Living)",
    "Bedroom": "Zone Air Temperature(Bedroom)",
}
t_name_operative = {
    "Living": "Zone Operative Temperature(Living)",
    "Bedroom": "Zone Operative Temperature(Bedroom)",
}
def get_temp_name(use_operative:bool):
    if use_operative:
        return t_name_operative
    else:
        return t_name

t_set_name = {
    "Living": "Zone Thermostat Heating Setpoint Temperature(Living)",
    "Bedroom": "Zone Thermostat Heating Setpoint Temperature(Bedroom)",
}
co2_name = {
    "Living": "Zone Air CO2 Concentration(Living)",
    "Bedroom": "Zone Air CO2 Concentration(Bedroom)",
}
occ_name = {
    "Living": "Zone People Occupant Count(Living)",
    "Bedroom": "Zone People Occupant Count(Bedroom)",
}
vent_name = {
    "Living": "Zone Ventilation Air Change Rate(Living)",
    "Bedroom": "Zone Ventilation Air Change Rate(Bedroom)",
}
humidity_name = {
    "Living": "Zone Air Relative Humidity(Living)",
    "Bedroom": "Zone Air Relative Humidity(Bedroom)",
}

rainfall_name = "Site Rain Status(Environment)"
diffuse_solar_radiation_name = "Site Diffuse Solar Radiation Rate per Area(Environment)"
direct_solar_radiation_name = "Site Direct Solar Radiation Rate per Area(Environment)"
windspeed_name = "Site Wind Speed(Environment)"

produced_electricity_name = ("Electric Load Center Produced Electricity Rate"
                     "(DC with inverter and Synerion 24M)")
electricity_demand_name = "Facility Total Electricity Demand Rate(Whole Building)"
battery_charging_state_name = "Electric Storage Battery Charge State(SYNERION 24M)"

hour_name = "hour"
month_name = "month"
day_name = "day"
