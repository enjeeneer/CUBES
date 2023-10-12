"""some constants for bhvaioural models"""

from cubes.construct.buildingconfig_options import ResWindowOpeningModel

variables_for_ventilation_models = {
    ResWindowOpeningModel.HALDI_2017.value: [
        "Zone Air co2 Concentration",
        "Zone Mean Air Temperature",
        "Zone Air Relative Humidity",
        "Zone Ventilation Air Change Rate",
        "Zone People Occupant Count",
    ],
    ResWindowOpeningModel.ANDERSEN_2013_BR.value: [
        "Zone Air co2 Concentration",
        "Zone Mean Air Temperature",
        "Zone Air Relative Humidity",
        "Zone Ventilation Air Change Rate",
        "Zone People Occupant Count",
        "Site Outdoor Air Drybulb Temperature",
    ],
    ResWindowOpeningModel.ANDERSEN_2013_LR.value: [
        "Zone Air co2 Concentration",
        "Zone Mean Air Temperature",
        "Zone Air Relative Humidity",
        "Zone Ventilation Air Change Rate",
        "Zone People Occupant Count",
        "Site Outdoor Air Drybulb Temperature",
    ],
    ResWindowOpeningModel.JONES_2017.value: [
        "Zone Mean Air Temperature",
        "Zone Air Relative Humidity",
        "Zone Ventilation Air Change Rate",
        "Zone People Occupant Count",
    ],
}
