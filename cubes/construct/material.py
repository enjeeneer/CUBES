"""This module defines the material classes which facilitate import of materials from a
database and adding materials and constructions to an IDF file"""

from dataclasses import dataclass
from typing import List


@dataclass
class Material:
    """
    A class which holds material properties
    """

    name: str
    rho: float
    cp: float
    k: float
    roughness: str
    thermal_absorptance: float
    solar_absorptance: float
    visual_absorptance: float

    def add_to_idf(self, idf, element, thickness):
        idf.newidfobject("MATERIAL")
        new_mat = idf.idfobjects["MATERIAL"][-1]
        new_mat.Name = self.get_idf_material_name(element, thickness)
        new_mat.Roughness = self.roughness
        new_mat.Thickness = thickness
        new_mat.Conductivity = self.k
        new_mat.Density = self.rho
        new_mat.Specific_Heat = self.cp
        new_mat.Thermal_Absorptance = self.thermal_absorptance
        new_mat.Solar_Absorptance = self.solar_absorptance
        new_mat.Visible_Absorptance = self.visual_absorptance
        return idf

    def get_idf_material_name(self, element, thickness):
        return self.name + "-" + self.transform_element(element) + "-" + str(thickness)

    def transform_element(self, element):
        if element.lower() == "ceiling":
            return "floor"
        elif element.lower() == "last ceiling":
            return "last floor"
        else:
            return element

    def get_thermal_resistance(self, thickness):
        return thickness / self.k


@dataclass
class NoMassMaterial:
    """
    A class which holds properties of no-mass materials
    """

    name: str
    roughness: str
    resistance: float
    thermal_absorptance: float
    solar_absorptance: float
    visual_absorptance: float

    def add_to_idf(self, idf, element, thickness):
        idf.newidfobject("MATERIAL:NOMASS", Thermal_Resistance=self.resistance)
        new_mat = idf.idfobjects["MATERIAL:NOMASS"][-1]
        new_mat.Name = self.get_idf_material_name(element, thickness)
        new_mat.Roughness = self.roughness
        new_mat.Thermal_Absorptance = self.thermal_absorptance
        new_mat.Solar_Absorptance = self.solar_absorptance
        new_mat.Visible_Absorptance = self.visual_absorptance

        return idf

    def get_idf_material_name(self, element, thickness):
        return self.name + "-" + self.transform_element(element) + "-" + str(thickness)

    def transform_element(self, element):
        if element.lower() == "ceiling":
            return "floor"
        elif element.lower() == "last ceiling":
            return "last floor"
        else:
            return element

    def get_thermal_resistance(self, *_):
        return self.resistance


@dataclass
class WindowMaterialGlazing:
    """
    This class holds properties for window materials
    """

    name: str
    optical_data_type: str
    data_set_name: str
    thickness: float
    solar_transmittance: float
    front_side_solar_reflectance: float
    back_side_solar_reflectance: float
    visible_transmittance: float
    front_side_visible_reflectance: float
    back_side_visible_reflectance: float
    infrared_transmittance: float
    front_side_infrared_emissivity: float
    back_side_infrared_emissivity: float
    conductivity: float

    def add_to_idf(self, idf):
        idf.newidfobject("WINDOWMATERIAL:GLAZING")
        new_mat = idf.idfobjects["WINDOWMATERIAL:GLAZING"][-1]
        new_mat.Name = self.name
        new_mat.Optical_Data_Type = self.optical_data_type
        new_mat.Thickness = self.thickness
        new_mat.Solar_Transmittance_at_Normal_Incidence = self.solar_transmittance
        new_mat.Front_Side_Solar_Reflectance_at_Normal_Incidence = (
            self.front_side_solar_reflectance
        )
        new_mat.Back_Side_Solar_Reflectance_at_Normal_Incidence = (
            self.back_side_solar_reflectance
        )
        new_mat.Visible_Transmittance_at_Normal_Incidence = self.visible_transmittance
        new_mat.Front_Side_Visible_Reflectance_at_Normal_Incidence = (
            self.front_side_visible_reflectance
        )
        new_mat.Back_Side_Visible_Reflectance_at_Normal_Incidence = (
            self.back_side_visible_reflectance
        )
        new_mat.Infrared_Transmittance_at_Normal_Incidence = self.infrared_transmittance
        new_mat.Front_Side_Infrared_Hemispherical_Emissivity = (
            self.front_side_infrared_emissivity
        )
        new_mat.Back_Side_Infrared_Hemispherical_Emissivity = (
            self.back_side_infrared_emissivity
        )
        new_mat.Conductivity = self.conductivity

        return idf


@dataclass
class WindowMaterialSimpleGlazing:
    """
    This class holds properties for "simple glazing systems"
    """

    name: str
    u_factor: str
    solar_heat_gain_coefficient: str
    visible_transmittance: float

    def add_to_idf(self, idf):
        idf.newidfobject("WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM")
        new_mat = idf.idfobjects["WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM"][-1]
        new_mat.Name = self.name
        new_mat.UFactor = self.u_factor
        new_mat.Solar_Heat_Gain_Coefficient = self.solar_heat_gain_coefficient
        new_mat.Visible_Transmittance = self.visible_transmittance

        idf.newidfobject("CONSTRUCTION")
        new_con = idf.idfobjects["CONSTRUCTION"][-1]
        new_con.Name = "Glazing"
        new_con.Outside_Layer = self.name

        return idf


@dataclass
class WindowConstruction:
    """Class for window constructions. This is currently only dealing with
    single and double glazing and two coating options."""

    window_type: str
    window_layers: List[str]
    window_thickness: List[float]

    def get_name(self):

        return self.window_type + " Glazing "

    def add_to_idf(self, idf, windows: dict):
        idf = windows[self.window_layers[0]].add_to_idf(idf)

        if self.window_type != "Single":
            idf.newidfobject(
                "WINDOWMATERIAL:GAS",
                Name=self.window_layers[1],
                Gas_Type=self.window_layers[1],
                Thickness=self.window_thickness[1],
            )

        idf.newidfobject("CONSTRUCTION")
        new_con = idf.idfobjects["CONSTRUCTION"][-1]

        new_con.Name = self.get_name()

        new_con.Outside_Layer = self.window_layers[0]

        # Maximum glazing is triple - Ambience's max is double
        if self.window_type != "Single":
            new_con.Layer_2 = self.window_layers[1]
            new_con.Layer_3 = self.window_layers[2]
            if self.window_layers[2] != self.window_layers[0]:
                idf = windows[self.window_layers[2]].add_to_idf(idf)
            if self.window_type != "Double":
                new_con.Layer_4 = self.window_layers[3]
                new_con.Layer_5 = self.window_layers[4]
                if (self.window_layers[4] != self.window_layers[2]) and (
                    self.window_layers[4] != self.window_layers[0]
                ):
                    idf = windows[self.window_layers[4]].add_to_idf(idf)

        return idf


@dataclass
class WindowFrameConstruction:
    """Class for window frame constructions. Default values used for all inputs except
    for frame width and conductivity"""

    name: str
    frame_width: int
    frame_u_value: float

    def add_to_idf(self, idf):
        idf.newidfobject("WINDOWPROPERTY:FRAMEANDDIVIDER")
        new_mat = idf.idfobjects["WINDOWPROPERTY:FRAMEANDDIVIDER"][-1]
        new_mat.Name = self.name + "-Frame"
        new_mat.Frame_Width = self.frame_width
        new_mat.Frame_Conductance = 1 / self.frame_u_value

        return idf


@dataclass
class Construction:
    """
    This class holds the properties of a construction
    and can add a construction to and IDF
    """

    element: str
    materials: List[Material]
    thicknesses: List[float]

    def get_name(self):
        return self.element + "-Construction"

    def get_u_value(self):
        thermal_resistance = 0
        for m, t in zip(self.materials, self.thicknesses):
            thermal_resistance += m.get_thermal_resistance(t)
        return 1 / thermal_resistance

    def add_to_idf(self, idf):
        idf.newidfobject("CONSTRUCTION")
        new_con = idf.idfobjects["CONSTRUCTION"][-1]
        new_con.Name = self.get_name()

        non_zero_layers = []
        for i, t in enumerate(self.thicknesses):
            if t > 1e-8:
                non_zero_layers.append(i)

        self.materials = [self.materials[i] for i in non_zero_layers]
        self.thicknesses = [self.thicknesses[i] for i in non_zero_layers]

        # check for identical layers:
        duplicate_idxs = []
        for idx in range(len(self.materials)):
            for cidx in range(idx + 1, len(self.materials)):
                if (
                    self.materials[idx].name == self.materials[cidx].name
                    and self.thicknesses[idx] == self.thicknesses[cidx]
                ):
                    duplicate_idxs.append(idx)

        # add materials to idf:
        for i, (m, t) in enumerate(zip(self.materials, self.thicknesses)):
            if (
                t > 1e-8
                and self.element not in ["Ceiling", "Last ceiling"]
                and i not in duplicate_idxs
            ):
                idf = m.add_to_idf(idf, self.element, t)

        new_con.Outside_Layer = self.materials[0].get_idf_material_name(
            self.element, self.thicknesses[0]
        )
        if len(self.materials) > 1:
            new_con.Layer_2 = self.materials[1].get_idf_material_name(
                self.element, self.thicknesses[1]
            )
            if len(self.materials) > 2:
                new_con.Layer_3 = self.materials[2].get_idf_material_name(
                    self.element, self.thicknesses[2]
                )
                if len(self.materials) > 3:
                    new_con.Layer_4 = self.materials[3].get_idf_material_name(
                        self.element, self.thicknesses[3]
                    )
                    if len(self.materials) > 4:
                        new_con.Layer_5 = self.materials[4].get_idf_material_name(
                            self.element, self.thicknesses[4]
                        )
                        if len(self.materials) > 5:
                            new_con.Layer_6 = self.materials[5].get_idf_material_name(
                                self.element, self.thicknesses[5]
                            )

        return idf
