""" air flow network.py """
import os
from collections import defaultdict
from eppy.modeleditor import IDF

def ensure_basic_schedules(idf):
    """Ensure AlwaysOnSchedule and AlwaysOffSchedule exist."""
    existing_schedules = {s.Name.lower() for s in idf.idfobjects.get("SCHEDULE:CONSTANT", [])}

    if "alwaysonschedule" not in existing_schedules:
        idf.newidfobject(
            "SCHEDULE:CONSTANT",
            Name="AlwaysOnSchedule",
            Schedule_Type_Limits_Name="OnOff",
            Hourly_Value=1
        )
        print("🗓️ Added AlwaysOnSchedule.")

    if "alwaysoffschedule" not in existing_schedules:
        idf.newidfobject(
            "SCHEDULE:CONSTANT",
            Name="AlwaysOffSchedule",
            Schedule_Type_Limits_Name="OnOff",
            Hourly_Value=0
        )
        print("🗓️ Added AlwaysOffSchedule.")

def resolve_opening_schedule(opening):
    schedule = opening.get("schedule", "1").lower()
    if schedule == "1":
        return "AlwaysOnSchedule"
    elif schedule == "0":
        return "AlwaysOffSchedule"
    else:
        return f"Occupancy-Schedule-{schedule}"

def remove_old_airflow_objects(idf):
    """Remove old infiltration, ventilation, and cross-mixing objects safely."""
    for objtype in [
        "ZONEINFILTRATION:DESIGNFLOWRATE",
        "ZONEVENTILATION:DESIGNFLOWRATE",
        "ZONECROSSMIXING"
    ]:
        old_objects = list(idf.idfobjects.get(objtype, []))
        for obj in old_objects:
            idf.removeidfobject(obj)
    print("🗑️ Removed old infiltration, ventilation, and mixing objects.")

def setup_afn_control(idf):
    """Set up AFN simulation controls and basic settings."""
    cp_array_name = "NormalExposureCpArray"
    idf.newidfobject(
        "AIRFLOWNETWORK:SIMULATIONCONTROL",
        Name="AFN_SimulationControl",
        AirflowNetwork_Control="MultizoneWithoutDistribution",
        Wind_Pressure_Coefficient_Type="Input",
        Height_Selection_for_Local_Wind_Pressure_Calculation="ExternalNode",
        Building_Type="LowRise",
        Maximum_Number_of_Iterations=500,
        Initialization_Type="ZeroNodePressures",
        Relative_Airflow_Convergence_Tolerance=1e-5,
        Absolute_Airflow_Convergence_Tolerance=1e-6,
        Convergence_Acceleration_Limit=-0.5,
        Azimuth_Angle_of_Long_Axis_of_Building=0,
        Ratio_of_Building_Width_Along_Short_Axis_to_Width_Along_Long_Axis=1,
        Height_Dependence_of_External_Node_Temperature="No"
    )
    idf.newidfobject(
        "AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTARRAY",
        Name=cp_array_name,
        Wind_Direction_1=0, Wind_Direction_2=45, Wind_Direction_3=90, Wind_Direction_4=135,
        Wind_Direction_5=180, Wind_Direction_6=225, Wind_Direction_7=270, Wind_Direction_8=315
    )
    if not idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:REFERENCECRACKCONDITIONS"):
        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:REFERENCECRACKCONDITIONS",
            Name="ReferenceCrackConditions",
            Reference_Temperature=20.0,
            Reference_Barometric_Pressure=101325,
            Reference_Humidity_Ratio=0.0
        )
    for zone in idf.idfobjects["ZONE"]:
        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:ZONE",
            Zone_Name=zone.Name,
            Ventilation_Control_Mode="Constant",
            Minimum_Venting_Open_Factor=1,
            Indoor_and_Outdoor_Temperature_Difference_Lower_Limit_For_Maximum_Venting_Open_Factor=0,
            Indoor_and_Outdoor_Temperature_Difference_Upper_Limit_for_Minimum_Venting_Open_Factor=100,
            Indoor_and_Outdoor_Enthalpy_Difference_Lower_Limit_For_Maximum_Venting_Open_Factor=0,
            Indoor_and_Outdoor_Enthalpy_Difference_Upper_Limit_for_Minimum_Venting_Open_Factor=100000,
            Venting_Availability_Schedule_Name="AlwaysOnSchedule",
            Single_Sided_Wind_Pressure_Coefficient_Algorithm="Standard",
            Facade_Width=0
        )
    print("🔧 AFN controls set up.")


def add_openings_for_internal_doors_and_holes(idf, building_config):
    """Add openings for internal doors and horizontal holes based on
       building_config.openings. Ensures one AFN surface per door/hole pair."""
    count = 0

    for opening in building_config.openings:
        schedule_name = resolve_opening_schedule(opening)
        comp_name = "_".join(opening["zones"]) + "_Opening"

        # Create component: Horizontal or Vertical
        if opening.get("orientation", "vertical").lower() == "horizontal":
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:COMPONENT:HORIZONTALOPENING",
                Name=comp_name,
                Air_Mass_Flow_Coefficient_When_Opening_is_Closed=0.001,
                Air_Mass_Flow_Exponent_When_Opening_is_Closed=0.65,
                Sloping_Plane_Angle=90.0,
                Discharge_Coefficient=0.65,
            )
        else:
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:COMPONENT:SIMPLEOPENING",
                Name=comp_name,
                Air_Mass_Flow_Coefficient_When_Opening_is_Closed=0.001,
                Air_Mass_Flow_Exponent_When_Opening_is_Closed=0.65,
                Minimum_Density_Difference_for_TwoWay_Flow=0.0001,
                Discharge_Coefficient=0.65,
            )

        # Find one fenestration surface representing this connection
        matched_surface = None
        for fen in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
            z1, z2 = opening["zones"]
            if (
                z1.lower() in fen.Building_Surface_Name.lower()
                and z2.lower() in fen.Outside_Boundary_Condition_Object.lower()
            ):
                matched_surface = fen
                break
            if (
                z2.lower() in fen.Building_Surface_Name.lower()
                and z1.lower() in fen.Outside_Boundary_Condition_Object.lower()
            ):
                matched_surface = fen
                break

        # Attach AFN surface if a match was found
        if matched_surface:
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:SURFACE",
                Surface_Name=matched_surface.Name,
                Leakage_Component_Name=comp_name,
                External_Node_Name="",
                Ventilation_Control_Mode="Constant",
                Venting_Availability_Schedule_Name=schedule_name,
            )
            count += 1

    print(f"🔵 Internal Door & Hole Openings Created: {count}")
    return count


def calculate_surface_mid_height(surface):
    zcoords = [float(getattr(surface, f"Vertex_{i}_Zcoordinate")) for i in range(1, 5) if getattr(surface, f"Vertex_{i}_Zcoordinate", "") != ""]
    return sum(zcoords) / len(zcoords) if zcoords else 1.5

def is_surface_horizontal(surface):
    zcoords = [float(getattr(surface, f"Vertex_{i}_Zcoordinate")) for i in range(1, 5) if getattr(surface, f"Vertex_{i}_Zcoordinate", "") != ""]
    return (max(zcoords) - min(zcoords)) < 0.1 if len(zcoords) >= 2 else False


def add_cracks_to_building_surfaces(idf, cp_array_name):
    """Add cracks and external nodes to external walls and roofs."""
    # Crack coefficients for different building elements (from the thesis)
    crack_characteristics = {
        'external_wall': {'flow_coef': 0.0002, 'exponent': 0.7},
        'internal_wall': {'flow_coef': 0.005, 'exponent': 0.75},  # Updated for internal walls
        'internal_floor': {'flow_coef': 0.002, 'exponent': 0.7},
        'external_floor': {'flow_coef': 0.001, 'exponent': 1.0},
        'roof': {'flow_coef': 0.00015, 'exponent': 0.7}
    }

    count = 0
    for surface in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
        if surface.Outside_Boundary_Condition.lower() != "outdoors":
            continue

        # Choose crack properties based on surface type
        surface_type = surface.Surface_Type.lower()
        if surface_type == "wall":
            crack_flow_coef, crack_exponent = crack_characteristics['external_wall'].values()
        elif surface_type == "roof":
            crack_flow_coef, crack_exponent = crack_characteristics['roof'].values()
        elif surface_type == "floor":
            if "external" in surface.Name.lower():
                crack_flow_coef, crack_exponent = crack_characteristics['external_floor'].values()
            else:
                crack_flow_coef, crack_exponent = crack_characteristics['internal_floor'].values()
        elif surface_type == "internalwall":  # Add check for internal walls
            crack_flow_coef, crack_exponent = crack_characteristics['internal_wall'].values()  # Internal wall properties
        else:
            continue  # Skip if no matching surface type

        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:SURFACE:CRACK",
            Name=f"{surface.Name}_Crack",
            Air_Mass_Flow_Coefficient_at_Reference_Conditions=crack_flow_coef,
            Air_Mass_Flow_Exponent=crack_exponent,
            Reference_Crack_Conditions="ReferenceCrackConditions"
        )
        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:SURFACE",
            Surface_Name=surface.Name,
            Leakage_Component_Name=f"{surface.Name}_Crack",
            External_Node_Name=surface.Name,
            Ventilation_Control_Mode="ZoneLevel"
        )
        count += 1
    return count



def add_cracks_to_building_surfaces(idf, crack_flow_coef, crack_exponent, cp_array_name):
    """Add cracks and external nodes to external walls and roofs."""
    cp_values_wall = [0.4, 0.1, -0.3, -0.35, -0.2, -0.35, -0.3, -0.1]
    cp_values_roof = [-0.6, -0.5, -0.4, -0.5, -0.6, -0.5, -0.4, -0.5]

    count = 0
    for surface in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
        if surface.Outside_Boundary_Condition.lower() != "outdoors":
            continue

        cp_values = cp_values_wall if surface.Surface_Type.lower() == "wall" else cp_values_roof

        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTVALUES",
            Name=surface.Name,
            AirflowNetworkMultiZoneWindPressureCoefficientArray_Name=cp_array_name,
            **{f"Wind_Pressure_Coefficient_Value_{i+1}": v for i, v in enumerate(cp_values)}
        )
        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE",
            Name=surface.Name,
            External_Node_Height=calculate_surface_mid_height(surface),
            Wind_Pressure_Coefficient_Curve_Name=surface.Name
        )

        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:SURFACE:CRACK",
            Name=f"{surface.Name}_Crack",
            Air_Mass_Flow_Coefficient_at_Reference_Conditions=crack_flow_coef,
            Air_Mass_Flow_Exponent=crack_exponent,
            Reference_Crack_Conditions="ReferenceCrackConditions"
        )
        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:SURFACE",
            Surface_Name=surface.Name,
            Leakage_Component_Name=f"{surface.Name}_Crack",
            External_Node_Name=surface.Name,
            Ventilation_Control_Mode="ZoneLevel"
        )
        count += 1
    return count

def add_cracks_to_external_fenestrations(idf, crack_flow_coef, crack_exponent):
    """Add cracks to external windows, vents, and external doors (not internal doors)."""
    count = 0
    building_surfaces = idf.idfobjects["BUILDINGSURFACE:DETAILED"]
    for fen in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
        if (fen.Surface_Type.lower() == "door" and "partition" in fen.Name.lower()) or 'hole' in fen.Name.lower():
            continue

        parent = next((s for s in building_surfaces if s.Name == fen.Building_Surface_Name), None)
        if not parent or parent.Outside_Boundary_Condition.lower() != "outdoors":
            continue

        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:SURFACE:CRACK",
            Name=f"{fen.Name}_Crack",
            Air_Mass_Flow_Coefficient_at_Reference_Conditions=crack_flow_coef,
            Air_Mass_Flow_Exponent=crack_exponent,
            Reference_Crack_Conditions="ReferenceCrackConditions"
        )
        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:SURFACE",
            Surface_Name=fen.Name,
            Leakage_Component_Name=f"{fen.Name}_Crack",
            External_Node_Name=parent.Name,
            Ventilation_Control_Mode="ZoneLevel"
        )
        count += 1
    return count

def add_subfloor_cracks(idf, subfloor_zone_name="Subfloor",
                        wall_flow_coef=0.0015, wall_flow_exp=0.65,
                        ceiling_flow_coef=0.0005, ceiling_flow_exp=0.65):
    """
    Add AFN cracks for subfloor walls (to ground) and ceilings (to ground floor zones).

    Parameters
    ----------
    idf : eppy.modeleditor.IDF
        The IDF object to modify.
    subfloor_zone_name : str, optional
        The name of the subfloor zone. Defaults to "Subfloor".
    wall_flow_coef : float, optional
        Crack flow coefficient for subfloor external walls (m3/s @ 1 Pa).
    wall_flow_exp : float, optional
        Flow exponent for subfloor external walls.
    ceiling_flow_coef : float, optional
        Crack flow coefficient for subfloor ceiling (to ground floor zones).
    ceiling_flow_exp : float, optional
        Flow exponent for subfloor ceiling.
    """

    # Ensure ReferenceCrackConditions exists (needed for AFN)
    if not idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:REFERENCECRACKCONDITIONS"):
        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:REFERENCECRACKCONDITIONS",
            Name="ReferenceCrackConditions",
            Reference_Temperature=20.0,
            Reference_Barometric_Pressure=101325,
            Reference_Humidity_Ratio=0.0
        )

    count_walls = 0
    count_ceilings = 0

    for surface in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
        if surface.Zone_Name.lower() != subfloor_zone_name.lower():
            continue

        surf_type = surface.Surface_Type.lower()
        bc = surface.Outside_Boundary_Condition.lower()

        if surf_type == "wall":
            # External wall (to ground) -> add airbrick cracks
            crack_name = f"{surface.Name}_SubfloorWallCrack"
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:SURFACE:CRACK",
                Name=crack_name,
                Air_Mass_Flow_Coefficient_at_Reference_Conditions=wall_flow_coef,
                Air_Mass_Flow_Exponent=wall_flow_exp,
                Reference_Crack_Conditions="ReferenceCrackConditions"
            )
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:SURFACE",
                Surface_Name=surface.Name,
                Leakage_Component_Name=crack_name,
                External_Node_Name="",  # No external node for ground
                Ventilation_Control_Mode="Constant",
                Venting_Availability_Schedule_Name="AlwaysOnSchedule"
            )
            count_walls += 1

        elif surf_type == "ceiling" and bc == "surface":
            # Ceiling (shared with zone above) -> add small leakage cracks
            crack_name = f"{surface.Name}_SubfloorCeilingLeak"
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:SURFACE:CRACK",
                Name=crack_name,
                Air_Mass_Flow_Coefficient_at_Reference_Conditions=ceiling_flow_coef,
                Air_Mass_Flow_Exponent=ceiling_flow_exp,
                Reference_Crack_Conditions="ReferenceCrackConditions"
            )
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:SURFACE",
                Surface_Name=surface.Name,
                Leakage_Component_Name=crack_name,
                External_Node_Name="",  # Internal connection
                Ventilation_Control_Mode="Constant",
                Venting_Availability_Schedule_Name="AlwaysOnSchedule"
            )
            count_ceilings += 1

    print(f"🔵 Added {count_walls} subfloor wall cracks and {count_ceilings} ceiling leakage cracks.")
    return count_walls, count_ceilings

def add_loft_floor_cracks(idf):
    """Add cracks between loft floors and the ceilings of rooms below."""
    crack_flow_coef = 0.0001   # smaller than external walls (hatch, penetrations)
    crack_exponent = 0.65

    count = 0
    for surface in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
        if ("loft" in surface.Zone_Name.lower() and
            surface.Surface_Type.lower() == "floor" and
            surface.Outside_Boundary_Condition.lower() == "surface"):

            # Create a crack component
            crack_name = f"{surface.Name}_Crack"
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:SURFACE:CRACK",
                Name=crack_name,
                Air_Mass_Flow_Coefficient_at_Reference_Conditions=crack_flow_coef,
                Air_Mass_Flow_Exponent=crack_exponent,
                Reference_Crack_Conditions="ReferenceCrackConditions"
            )

            # Add the airflow network surface
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:SURFACE",
                Surface_Name=surface.Name,
                Leakage_Component_Name=crack_name,
                External_Node_Name="",  # internal surface, no external node
                Ventilation_Control_Mode="ZoneLevel"
            )

            count += 1

    print(f"🔵 Added {count} cracks between loft and first-floor ceilings.")
    return count

def convert_subfloor_walls_to_outdoors(idf):
    """Convert all subfloor external walls from 'Ground' to 'Outdoors' for AFN compatibility."""
    count = 0
    for surface in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
        if "subfloor" in surface.Zone_Name.lower() and surface.Surface_Type.lower() == "wall":
            if surface.Outside_Boundary_Condition.lower() == "ground":
                surface.Outside_Boundary_Condition = "Outdoors"
                surface.Sun_Exposure = "NoSun"
                surface.Wind_Exposure = "NoWind"
                count += 1
    print(f"🔵 Converted {count} subfloor walls to Outdoors boundary.")
    return count


def add_cracks_and_openings(idf, building_config):
    """Adds cracks and user-defined openings to IDF for AFN model."""
    crack_flow_coef = 0.0002
    crack_exponent = 0.7
    cp_array_name = "NormalExposureCpArray"

    external = add_cracks_to_building_surfaces(idf, crack_flow_coef, crack_exponent, cp_array_name)
    fenestrations = add_cracks_to_external_fenestrations(idf, crack_flow_coef, crack_exponent)
    user_openings = add_openings_for_internal_doors_and_holes(idf, building_config)

    print(f"🔵 External Surfaces with Cracks: {external}")
    print(f"🔵 External Fenestrations with Cracks: {fenestrations}")
    print(f"🔵 User-defined openings created: {user_openings}")


def add_airflow_network(idf, building_config):

    ensure_basic_schedules(idf)
    remove_old_airflow_objects(idf)
    setup_afn_control(idf)
    add_cracks_and_openings(idf, building_config)
    add_loft_floor_cracks(idf)
    convert_subfloor_walls_to_outdoors(idf)
    add_subfloor_cracks(idf, subfloor_zone_name="Subfloor")
    return idf
