"""This module defines the material classes which facilitate import of materials from a
database and adding materials and constructions to an IDF file"""

from dataclasses import dataclass
from typing import List
import constants as con


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
        return self.name + "-" + element + "-" + str(thickness)


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
        new_mat.Thermal_Absorptance = self.thermalAbsorptance
        new_mat.Solar_Absorptance = self.solarAbsorptance
        new_mat.Visible_Absorptance = self.visualAbsorptance

        return idf

    def get_idf_material_name(self, element, thickness):
        return self.name + "-" + element + "-" + str(thickness)


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
    fron_side_visible_reflectance: float
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
            self.fron_side_visible_reflectance
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
class WindowConstruction:
    """Class for window constructions. This is currently only dealing with
    single and double glazing and two coating options."""

    window_type: str
    low_e_coating: bool
    gap_width: float

    def get_name(self):
        if self.low_e_coating:
            coating = " lowE"
        else:
            coating = ""

        if self.gap_width > 1e-8:
            gap = str(self.gap_width) + "mm"
        else:
            gap = ""
        return self.window_type + " Glazing " + gap + coating

    def add_to_idf(self, idf):
        if not self.low_e_coating:
            glass_material = "CLEAR 3MM"
        else:
            glass_material = "LoE CLEAR 3MM"

        idf = con.WINDOW_GLASS_MATERIALS[glass_material].add_to_idf(idf)

        if self.window_type != "Single":
            idf.newidfobject(
                "WINDOWMATERIAL:GAS",
                Name="Argon",
                Gas_Type="Argon",
                Thickness=self.gap_width,
            )

        idf.newidfobject("CONSTRUCTION")
        new_con = idf.idfobjects["CONSTRUCTION"][-1]
        new_con.Name = self.get_name()

        new_con.Outside_Layer = glass_material

        if self.window_type == "Double":
            new_con.Layer_2 = "Argon"
            new_con.Layer_3 = glass_material

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

    def add_to_idf(self, idf):
        idf.newidfobject("CONSTRUCTION")
        new_con = idf.idfobjects["CONSTRUCTION"][-1]
        new_con.Name = self.get_name()

        # add materials to idf:
        for m, t in zip(self.materials, self.thicknesses):
            if t > 1e-8:
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
