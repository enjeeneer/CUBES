"""This class defines the valid values for building config parameter options"""

from enum import Enum, EnumMeta


class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)  # pylint: disable=no-value-for-parameter
        except ValueError:
            return False
        return True


class BaseEnum(Enum, metaclass=MetaEnum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class HeatingWaterLoopEquipment(BaseEnum):
    CONDENSING_BOILER = "condensing boiler"
    NON_CONDENSING_BOILER = "non-condensing boiler"
    ATW_HEAT_PUMP = "air-to-water heat pump"
    WTW_HEAT_PUMP = "water-to-water heat pump (ground source)"
    DISTRICT_HEATING = "district heating"


class VentilationType(BaseEnum):
    NATURAL = "natural"
    MECHANICAL = "mechanical"


class VentilationTypeImplemented(BaseEnum):
    NATURAL = "natural"
    MECHANICAL = "mechanical"


class NaturalVentilationMethod(BaseEnum):
    RATE_PER_OCCUPANT = "rate per occupant"
    RATE_PER_OCCUPANT_PLUS_COOLING = "rate per occupant plus cooling"
    RES_WIN_OP_MODEL = "residential window opening model"


res_window_PP_map = {
    "RES-WINDOW:Haldi-2017-Denmark": "VentilationRateHaldi2017Denmark",
    "RES-WINDOW:Andersen-2013-Group3-livingroom": (
        "VentilationRateAndersen2013Group3Livingroom"
    ),
    "RES-WINDOW:Andersen-2013-Group3-bedroom": (
        "VentilationRateAndersen2013Group3Bedroom"
    ),
    "RES-WINDOW:Jones-2017": "VentilationRateJones2017",
}


class ResWindowOpeningModel(BaseEnum):
    HALDI_2017 = "RES-WINDOW:Haldi-2017-Denmark"
    ANDERSEN_2013_LR = "RES-WINDOW:Andersen-2013-Group3-livingroom"
    ANDERSEN_2013_BR = "RES-WINDOW:Andersen-2013-Group3-bedroom"
    JONES_2017 = "RES-WINDOW:Jones-2017"


class Zoning(BaseEnum):
    RESIDENTIAL_DWELLING = "residential dwelling"
    ONE_ZONE_PER_FLOOR = "one zone per floor"


class ZoningImplemented(BaseEnum):
    RESIDENTIAL_DWELLING = "residential dwelling"


class RoofType(BaseEnum):
    SADDLEBACK = "saddleback"
    FLAT = "flat"
    ADIABATIC = "adiabatic"
