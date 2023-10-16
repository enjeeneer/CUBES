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

t_out_name = "Site Outdoor Air Drybulb Temperature(Environment)"
humidity_out_name = "Site Outdoor Air Relative Humidity(Environment)"
t_name = {
    "Living": "Zone Air Temperature(Living)",
    "Bedroom": "Zone Air Temperature(Bedroom)",
}
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
    "Bedroom": "Zone People Occupant Count(Bedroom)",
}
humidity_name = {
    "Living": "Zone Air Relative Humidity(Living)",
    "Bedroom": "Zone Air Relative Humidity(Bedroom)",
}

rainfall_name = "Site Rain Status(Environment)"
diffuse_solar_radiation_name = "Site Diffuse Solar Radiation Rate per Area(Environment)"
direct_solar_radiation_name = "Site Direct Solar Radiation Rate per Area(Environment)"
windspeed_name = "Site Wind Speed(Environment)"

produced_electricity_name = "Facility Total Produced Electricity Rate(Whole Building)"
electricity_demand_name = "Facility Total Electricity Demand Rate(Whole Building)"
battery_charging_state_name = "Electric Storage Battery Charge State(SYNERION 24M)"

hour_name = "hour"
month_name = "month"
day_name = "day"
