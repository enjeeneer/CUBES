"""This module adds schedules, ventilation, infiltration, systems, people,
and electric equipment to an idf file"""


def add_schedules(idf):
    # schedules
    idf.newidfobject(
        "SCHEDULE:COMPACT",
        Name="People-Schedule",
        Field_1=(
            "Through: 12/31,\n    "
            "For: Weekdays,\n    Until: 9:00, 1.0,\n"
            "    Until:17:00, 0.5,\n    Until:24:00, 1.,\n "
            "   For:AllOtherDays,\n    Until:24:00,1."
        ),
    )
    idf.newidfobject(
        "SCHEDULE:COMPACT",
        Name="Always-Schedule",
        Field_1="Through: 12/31,\n    For: AllDays,\n    Until: 24:00, 1.0\n",
    )
    idf.newidfobject(
        "SCHEDULE:COMPACT",
        Name="Activity-Schedule",
        Field_1="Through: 12/31,\n    For: AllDays,\n    Until: 24:00, 100.\n",
    )
    idf.newidfobject(
        "SCHEDULE:COMPACT",
        Name="Heating-Setpoints",
        Field_1="Through: 12/31,\n    For: AllDays,\n    Until: 24:00, 20.\n",
    )
    idf.newidfobject(
        "SCHEDULE:COMPACT",
        Name="Cooling-Setpoints",
        Field_1="Through: 12/31,\n    For: AllDays,\n    Until: 24:00, 25.\n",
    )
    return idf


def add_people(idf):
    # people
    idf.newidfobject(
        "PEOPLE",
        Name="People-1",
        Zone_or_ZoneList_Name="Zone-1",
        Number_of_People_Schedule_Name="People-Schedule",
        Number_of_People=2,
        Activity_Level_Schedule_Name="Activity-Schedule",
    )
    return idf


def add_ventilation(idf):
    # ventilation
    idf.newidfobject(
        "ZONEVENTILATION:DESIGNFLOWRATE",
        Name="Zone-1-Ventilation",
        Zone_or_ZoneList_Name="Zone-1",
        Design_Flow_Rate_Calculation_Method="Flow/Person",
        Flow_Rate_per_Person=0.01,
        Schedule_Name="Always-Schedule",
    )
    return idf


def add_infiltration(idf):
    # infiltration
    idf.newidfobject(
        "ZONEINFILTRATION:DESIGNFLOWRATE",
        Name="Zone-1-Infiltration",
        Zone_or_ZoneList_Name="Zone-1",
        Design_Flow_Rate_Calculation_Method="Flow/ExteriorArea",
        Flow_per_Exterior_Surface_Area=15 * 0.07,
        Constant_Term_Coefficient=0.606,
        Temperature_Term_Coefficient=0.03636,
        Velocity_Term_Coeﬀicient=0.1177,
        Velocity_Squared_Term_Coefficient=0.0,
        Schedule_Name="Always-Schedule",
    )
    return idf


def add_internal_gains(idf):
    # internal gains
    idf.newidfobject(
        "LIGHTS",
        Name="Lights-Zone-1",
        Zone_or_ZoneList_Name="Zone-1",
        Schedule_Name="Always-Schedule",
        Design_Level_Calculation_Method="Watts/area",
        Watts_per_Zone_Floor_Area=1,
    )

    idf.newidfobject(
        "ELECTRICEQUIPMENT",
        Name="Equipment-Zone-1",
        Zone_or_ZoneList_Name="Zone-1",
        Schedule_Name="Always-Schedule",
        Design_Level_Calculation_Method="Watts/area",
        Watts_per_Zone_Floor_Area=5,
    )

    return idf


def add_heating_system(idf):

    # heating system
    idf.newidfobject(
        "HVACTEMPLATE:THERMOSTAT",
        Name="Zone-1-Thermostat",
        Heating_Setpoint_Schedule_Name="Heating-Setpoints",
        Cooling_Setpoint_Schedule_Name="Cooling-Setpoints",
    )

    idf.newidfobject(
        "HVACTEMPLATE:ZONE:BASEBOARDHEAT",
        Zone_Name="Zone-1",
        Baseboard_Heating_Type="HotWater",
        Template_Thermostat_Name="Zone-1-Thermostat",
    )

    idf.newidfobject("HVACTEMPLATE:PLANT:HOTWATERLOOP", Name="Hot Water Loop")

    idf.newidfobject(
        "HVACTEMPLATE:PLANT:BOILER",
        Name="Main Boiler",
        Boiler_Type="CondensingHotWaterBoiler",
        Efficiency=0.8,
        Fuel_Type="NaturalGas",
    )

    return idf
