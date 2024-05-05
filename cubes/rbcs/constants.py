"""constanst shared across rbc methods"""

zone_names = ["Living", "Bedroom"]

# t_control_name = {
#     "Living": "Living-Thermostat Dual SP Control-HEATING-EXT",
#     "Bedroom": "Bedroom-Thermostat Dual SP Control-HEATING-EXT",
# }
#
# vent_control_name = {
#     "Living": "Living-Ventilation-EXT",
#     "Bedroom": "Bedroom-Ventilation-EXT",
# }
#
# t_name = {
#     "Living": "Zone Air Temperature(Living)",
#     "Bedroom": "Zone Air Temperature(Bedroom)",
# }
#
# t_name_operative = {
#     "Living": "Zone Operative Temperature(Living)",
#     "Bedroom": "Zone Operative Temperature(Bedroom)",
# }
#
# t_set_name = {
#     "Living": "Zone Thermostat Heating Setpoint Temperature(Living)",
#     "Bedroom": "Zone Thermostat Heating Setpoint Temperature(Bedroom)",
# }
#
# occ_name = {
#     "Living": "Zone People Occupant Count(Living)",
#     "Bedroom": "Zone People Occupant Count(Bedroom)",
# }
#
# co2_name = {
#     "Living": "Zone Air CO2 Concentration(Living)",
#     "Bedroom": "Zone Air CO2 Concentration(Bedroom)",
# }
#
# vent_name = {
#     "Living": "Zone Ventilation Air Change Rate(Living)",
#     "Bedroom": "Zone Ventilation Air Change Rate(Bedroom)",
# }
#
# humidity_name = {
#     "Living": "Zone Air Relative Humidity(Living)",
#     "Bedroom": "Zone Air Relative Humidity(Bedroom)",
# }


def get_zone_names(zoning):
    if zoning == "single zone":
        return ["Living"]
    else:
        return zone_names


def get_t_control_name(zones):
    t_control_name = {}
    for zone in zones:
        t_control_name[zone] = zone + "-Thermostat Dual SP Control-HEATING-EXT"
    return t_control_name


def get_vent_control_name(zones):
    vent_control_name = {}
    for zone in zones:
        vent_control_name[zone] = zone + "-Ventilation-EXT"
    return vent_control_name


charge_control_name = "Battery Charge Schedule-EXT"
discharge_control_name = "Battery Discharge Schedule-EXT"

utility_demand_target_control_name = "Utility Demand Target Schedule-EXT"

t_out_name = "Site Outdoor Air Drybulb Temperature(Environment)"
humidity_out_name = "Site Outdoor Air Relative Humidity(Environment)"


def get_t_name(zones):
    t_name = {}
    for zone in zones:
        t_name[zone] = f"Zone Air Temperature({zone})"
    return t_name


def get_t_name_operative(zones):
    t_name = {}
    for zone in zones:
        t_name[zone] = f"Zone Operative Temperature({zone})"
    return t_name


def get_temp_name(use_operative: bool, zones):
    if use_operative:
        return get_t_name_operative(zones)
    else:
        return get_t_name(zones)


def get_t_set_name(zones):
    t_set_name = {}
    for zone in zones:
        t_set_name[zone] = f"Zone Thermostat Heating Setpoint Temperature({zone})"
    return t_set_name


def get_co2_name(zones):
    co2_name = {}
    for zone in zones:
        co2_name[zone] = f"Zone Air CO2 Concentration({zone})"
    return co2_name


def get_occ_name(zones):
    occ_name = {}
    for zone in zones:
        occ_name[zone] = f"Zone People Occupant Count({zone})"
    return occ_name


def get_vent_name(zones):
    vent_name = {}
    for zone in zones:
        vent_name[zone] = f"Zone Ventilation Air Change Rate({zone})"
    return vent_name


def get_humidity_name(zones):
    humidity_name = {}
    for zone in zones:
        humidity_name[zone] = f"Zone Air Relative Humidity({zone})"
    return humidity_name


rainfall_name = "Site Rain Status(Environment)"
diffuse_solar_radiation_name = "Site Diffuse Solar Radiation Rate per Area(Environment)"
direct_solar_radiation_name = "Site Direct Solar Radiation Rate per Area(Environment)"
windspeed_name = "Site Wind Speed(Environment)"

produced_electricity_name = (
    "Electric Load Center Produced Electricity Rate"
    "(DC with inverter and Synerion 24M)"
)
electricity_demand_name = "Facility Total Electricity Demand Rate(Whole Building)"
battery_charging_state_name = "Electric Storage Battery Charge State(SYNERION 24M)"

hour_name = "hour"
month_name = "month"
day_name = "day"
