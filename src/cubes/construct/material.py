"""This module defines the material classes which facilitate import of materials from a
database and adding materials and constructions to an IDF file"""

from dataclasses import dataclass


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
        new_mat.Thermal_Absorptance = self.thermalAbsorptance
        new_mat.Solar_Absorptance = self.solarAbsorptance
        new_mat.Visible_Absorptance = self.visualAbsorptance
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


@dataclass
class Construction:
    """
    This class holds the properties of a construction
    and can add a construction to and IDF
    """

    element: str
    materials: list(Material)
    thicknesses: list(float)

    def add_to_idf(self, idf):
        idf.newidfobject("CONSTRUCTION")
        new_con = idf.idfobjects["CONSTRUCTION"][-1]
        new_con.Name = self.element + "-Construction"

        new_con.Outside_Layer = self.materials[0].get_idf_material_name(
            self.element, self.thicknesses[0]
        )
        if len(self.layers) > 1:
            new_con.Layer2 = self.materials[1].get_idf_material_name(
                self.element, self.thicknesses[1]
            )
            if len(self.layers) > 2:
                new_con.Layer3 = self.materials[2].get_idf_material_name(
                    self.element, self.thicknesses[2]
                )
                if len(self.layers) > 3:
                    new_con.Layer4 = self.materials[3].get_idf_material_name(
                        self.element, self.thicknesses[3]
                    )
                    if len(self.layers) > 4:
                        new_con.Layer5 = self.materials[4].get_idf_material_name(
                            self.element, self.thicknesses[4]
                        )
                        if len(self.layers) > 5:
                            new_con.Layer6 = self.materials[5].get_idf_material_name(
                                self.element, self.thicknesses[5]
                            )

        return idf
