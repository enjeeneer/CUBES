"""airflow_network.py

AFN builder:
- DB "poor" crack template applied per m² (multiplied by geometric area) on Outdoors walls/roofs.
- Optional internal openings (Simple/HorizontalOpening).
- Two "air brick" vents (0.01 m² each) auto-added to each Subfloor external wall (except azimuth=90°),
  with AFN crack and ExternalNode linkage to the parent wall.
- Audits to ensure each zone has ≥ 2 AFN surfaces.

Requires: eppy.modeleditor.IDF
"""

from collections import defaultdict
from typing import Optional, Tuple, List
from eppy.modeleditor import IDF
import math


CRACK_TEMPLATES = {
    "poor": {
        "external_wall":  {"cq_per_m2": 0.0002,  "n": 0.7},
        "internal_wall":  {"cq_per_m2": 0.005,   "n": 0.75},
        "internal_floor": {"cq_per_m2": 0.002,   "n": 0.7},
        "internal_ceiling": {"cq_per_m2": 0.002,   "n": 0.7},
        "external_floor": {"cq_per_m2": 0.001,   "n": 1.0},
        "external_roof":  {"cq_per_m2": 0.00015, "n": 0.7},
        "external_window":{"cq_per_m":  0.001,   "n": 0.6},
        "external_door":  {"cq_per_m":  0.0018,  "n": 0.66},
        "internal_door":  {"cq_per_m":  0.02,    "n": 0.6},
        "external_vent":  {"cq_per_m":  0.01,    "n": 0.66, "cd": 0.65}
    },
    "medium": {
        "external_wall":  {"cq_per_m2": 0.0001, "n": 0.7},
        "internal_wall":  {"cq_per_m2": 0.003,  "n": 0.75},
        "internal_floor": {"cq_per_m2": 0.0009, "n": 0.7},
        "internal_ceiling": {"cq_per_m2": 0.0009, "n": 0.7},
        "external_floor": {"cq_per_m2": 0.0007, "n": 1.0},
        "external_roof":  {"cq_per_m2": 0.0001,  "n": 0.7},
        "external_window":{"cq_per_m":  0.00014,"n": 0.65},
        "external_door":  {"cq_per_m":  0.0014, "n": 0.65},
        "internal_door":  {"cq_per_m":  0.02,   "n": 0.6},
        "external_vent":  {"cq_per_m":  0.008,  "n": 0.66, "cd": 0.65}
    },
    "tight": {
        "external_wall":  {"cq_per_m2": 0.0002,  "n": 0.7},
        "internal_wall":  {"cq_per_m2": 0.005,   "n": 0.75},
        "internal_floor": {"cq_per_m2": 0.002,   "n": 0.7},
        "internal_ceiling": {"cq_per_m2": 0.002,   "n": 0.7},
        "external_floor": {"cq_per_m2": 0.001,   "n": 1.0},
        "external_roof":  {"cq_per_m2": 0.00015, "n": 0.7},
        "external_window":{"cq_per_m":  0.001,   "n": 0.6},
        "external_door":  {"cq_per_m":  0.0018,  "n": 0.66},
        "internal_door":  {"cq_per_m":  0.02,    "n": 0.6},
        "external_vent":  {"cq_per_m":  0.01,    "n": 0.66, "cd": 0.65}
    },
}


# --------------------------
# Utilities
# --------------------------


# --- Helper functions ---
def perimeter(verts):
    if len(verts) < 2:
        return 0.0
    per = 0.0
    for i in range(len(verts)):
        x1, y1, z1 = verts[i]
        x2, y2, z2 = verts[(i + 1) % len(verts)]
        dx, dy, dz = (x2 - x1, y2 - y1, z2 - z1)
        per += math.sqrt(dx * dx + dy * dy + dz * dz)
    return per

def get_vertices(fen):
    vs = []
    for i in range(1, 501):
        x = getattr(fen, f"Vertex_{i}_Xcoordinate", "")
        if x == "":
            break
        y = getattr(fen, f"Vertex_{i}_Ycoordinate")
        z = getattr(fen, f"Vertex_{i}_Zcoordinate")
        vs.append((float(x), float(y), float(z)))
    return vs

def _get_vertices(s) -> List[Tuple[float, float, float]]:
    """Return list of (x,y,z) vertices from a BUILDINGSURFACE or FENESTRATIONSURFACE EpBunch."""
    verts = []
    for i in range(1, 501):  # plenty of headroom
        try:
            x = getattr(s, f"Vertex_{i}_Xcoordinate")
        except Exception:
            break
        if x == "":
            break
        y = getattr(s, f"Vertex_{i}_Ycoordinate")
        z = getattr(s, f"Vertex_{i}_Zcoordinate")
        verts.append((float(x), float(y), float(z)))
    return verts

def _polygon_area_3d(verts: List[Tuple[float, float, float]]) -> float:
    """Area of a planar polygon from ordered 3D vertices (Newell’s method)."""
    if len(verts) < 3:
        return 0.0
    nx = ny = nz = 0.0
    for (x1, y1, z1), (x2, y2, z2) in zip(verts, verts[1:] + verts[:1]):
        nx += (y1 - y2) * (z1 + z2)
        ny += (z1 - z2) * (x1 + x2)
        nz += (x1 - x2) * (y1 + y2)
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)

def _unit(v):
    x, y, z = v
    n = math.sqrt(x*x + y*y + z*z)
    if n == 0:
        return (0.0, 0.0, 0.0)
    return (x/n, y/n, z/n)

def _add(p, a, scale=1.0):
    return (p[0] + a[0]*scale, p[1] + a[1]*scale, p[2] + a[2]*scale)

def _sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def _approx(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(float(a) - float(b)) <= tol

def _get_azimuth(surface) -> float:
    # eppy exposes either 'Azimuth' or 'azimuth' depending on IDD label
    return float(getattr(surface, "Azimuth", getattr(surface, "azimuth", 0.0)))

# --------------------------
# Schedules & AFN scaffolding
# --------------------------

def ensure_basic_schedules(idf: IDF) -> None:
    names = {s.Name.lower() for s in idf.idfobjects.get("SCHEDULE:CONSTANT", [])}
    if "alwaysonschedule" not in names:
        idf.newidfobject(
            "SCHEDULE:CONSTANT",
            Name="AlwaysOnSchedule",
            Schedule_Type_Limits_Name="OnOff",
            Hourly_Value=1,
        )
    if "alwaysoffschedule" not in names:
        idf.newidfobject(
            "SCHEDULE:CONSTANT",
            Name="AlwaysOffSchedule",
            Schedule_Type_Limits_Name="OnOff",
            Hourly_Value=0,
        )

def ensure_reference_crack_conditions(idf: IDF) -> None:
    if not idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:REFERENCECRACKCONDITIONS"):
        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:REFERENCECRACKCONDITIONS",
            Name="ReferenceCrackConditions",
            Reference_Temperature=20.0,
            Reference_Barometric_Pressure=101325,
            Reference_Humidity_Ratio=0.0,
        )

def setup_afn_controls(idf: IDF, cp_array_name: str = "NormalExposureCpArray") -> None:
    if not idf.idfobjects.get("AIRFLOWNETWORK:SIMULATIONCONTROL"):
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
            Height_Dependence_of_External_Node_Temperature="No",
        )
    if not idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTARRAY"):
        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTARRAY",
            Name=cp_array_name,
            Wind_Direction_1=0,
            Wind_Direction_2=45,
            Wind_Direction_3=90,
            Wind_Direction_4=135,
            Wind_Direction_5=180,
            Wind_Direction_6=225,
            Wind_Direction_7=270,
            Wind_Direction_8=315,
        )

def ensure_afn_zones(idf: IDF) -> None:
    existing = {z.Zone_Name for z in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:ZONE", [])}
    for z in idf.idfobjects["ZONE"]:
        if z.Name in existing:
            continue
        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:ZONE",
            Zone_Name=z.Name,
            Ventilation_Control_Mode="Constant",
            Minimum_Venting_Open_Factor=1.0,
            Indoor_and_Outdoor_Temperature_Difference_Lower_Limit_For_Maximum_Venting_Open_Factor=0,
            Indoor_and_Outdoor_Temperature_Difference_Upper_Limit_for_Minimum_Venting_Open_Factor=100,
            Indoor_and_Outdoor_Enthalpy_Difference_Lower_Limit_For_Maximum_Venting_Open_Factor=0,
            Indoor_and_Outdoor_Enthalpy_Difference_Upper_Limit_for_Minimum_Venting_Open_Factor=100000,
            Venting_Availability_Schedule_Name="AlwaysOnSchedule",
            Single_Sided_Wind_Pressure_Coefficient_Algorithm="Standard",
            Facade_Width=0.0,
        )

def surface_mid_height(s) -> float:
    zs = [v[2] for v in _get_vertices(s)]
    return sum(zs)/len(zs) if zs else 1.5

# --------------------------
# External leakage (DB "poor" per m² × area)
# --------------------------

def get_template(cracks_cfg, zone, element_key):
    zone_cfg = cracks_cfg.get(zone.lower(), {})
    default_cfg = cracks_cfg.get("default", {})

    # Get either a dict of modifiers or a string label
    zone_entry = zone_cfg.get(element_key)
    if isinstance(zone_entry, dict):
        template_label = default_cfg.get(element_key)  # fallback to default type like "poor"
        modifiers = zone_entry
    else:
        template_label = zone_entry or default_cfg.get(element_key)
        modifiers = {}

    if not template_label:
        print(f"⚠️ No crack template found for {zone}:{element_key}")
        return None

    # Fetch the actual template from the correct category (poor/medium/tight)
    base_tmpl = CRACK_TEMPLATES.get(template_label, {}).get(element_key)
    if not base_tmpl:
        print(f"⚠️ Missing crack definition for {template_label}:{element_key}")
        return None

    # Apply modifiers if any
    cq_factor = modifiers.get("cq_per_m2_factor", modifiers.get("cq_per_m_factor", 1.0))
    n_factor = modifiers.get("n_factor", 1.0)

    tmpl = base_tmpl.copy()
    if "cq_per_m2" in tmpl:
        tmpl["cq_per_m2"] *= cq_factor
    if "cq_per_m" in tmpl:
        tmpl["cq_per_m"] *= cq_factor
    if "n" in tmpl:
        tmpl["n"] *= n_factor

    return tmpl




def add_surface_leakage(
    idf: IDF,
    building_config,
    cp_array_name: str = "NormalExposureCpArray",
) -> int:
    """
    Add AFN leakage to all building surfaces using crack templates.
    - Uses the building_config.cracks definitions to map elements to template names.
    - Zone-specific scaling factors (cq_per_m2_factor, n_factor) are applied if provided.
    - Ground and adiabatic surfaces are skipped.
    """

    cracks_cfg = building_config.cracks
    cp_vals = [0.4, 0.1, -0.3, -0.35, -0.2, -0.35, -0.3, -0.1]
    made = 0


    for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
        ext_node = ""
        stype = s.Surface_Type.lower()
        bc = s.Outside_Boundary_Condition.lower()

        if bc == "outdoors":
            scope = "outdoors"
            element_key = f"external_{stype}"
            ext_node = s.Name
        elif bc in ("adiabatic", "ground"):
            continue
        elif bc == "surface" and s.Outside_Boundary_Condition_Object:
            scope = "indoors"
            element_key = f"internal_{stype}"
            ext_node = s.Outside_Boundary_Condition_Object
        else:
            scope = "indoors"
            element_key = f"internal_{stype}"

        tmpl = get_template(cracks_cfg, s.Zone_Name, element_key)
        if not tmpl:
            continue

        coef_pm2 = tmpl.get("cq_per_m2", 0.0)
        expn = tmpl.get("n", 0.65)
        target_coef = coef_pm2 * s.area
        crack_name = f"{s.Name}_Surface_Crack"

        cracks = idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:SURFACE:CRACK", [])
        c = next((x for x in cracks if x.Name == crack_name), None)
        if c is None:
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:SURFACE:CRACK",
                Name=crack_name,
                Air_Mass_Flow_Coefficient_at_Reference_Conditions=target_coef,
                Air_Mass_Flow_Exponent=expn,
                Reference_Crack_Conditions="ReferenceCrackConditions",
            )
        else:
            c.Air_Mass_Flow_Coefficient_at_Reference_Conditions = target_coef
            c.Air_Mass_Flow_Exponent = expn

        if scope == "outdoors":
            if not any(v.Name == s.Name for v in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTVALUES", [])):
                idf.newidfobject(
                    "AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTVALUES",
                    Name=s.Name,
                    AirflowNetworkMultiZoneWindPressureCoefficientArray_Name=cp_array_name,
                    **{f"Wind_Pressure_Coefficient_Value_{i+1}": v for i, v in enumerate(cp_vals)}
                )
            if not any(n.Name == s.Name for n in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE", [])):
                idf.newidfobject(
                    "AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE",
                    Name=s.Name,
                    External_Node_Height=surface_mid_height(s),
                    Wind_Pressure_Coefficient_Curve_Name=s.Name,
                )
            ext_node = s.Name

        if not any(afn.Surface_Name == s.Name for afn in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:SURFACE", [])):
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:SURFACE",
                Surface_Name=s.Name,
                Leakage_Component_Name=crack_name,
                External_Node_Name=ext_node,
                Ventilation_Control_Mode="ZoneLevel",
            )
            made += 1

    return made



# --------------------------
# Internal openings (zone-to-zone)
# --------------------------

def resolve_opening_schedule(opening: dict) -> str:
    val = opening.get("schedule", "1")
    s = str(val).lower()
    if s == "1":
        return "AlwaysOnSchedule"
    if s == "0":
        return "AlwaysOffSchedule"
    return f"door-schedule-{s}"

def add_internal_openings(idf: IDF, building_config) -> int:

    openings_cfg = building_config.openings
    made_components = set()
    count = 0
    for op in openings_cfg or []:
        z1, z2 = op["zones"]
        comp_name = f"{z1}_{z2}_Opening"
        sched = resolve_opening_schedule(op)
        if comp_name not in made_components:
            if str(op.get("orientation", "vertical")).lower() == "horizontal":
                idf.newidfobject(
                    "AIRFLOWNETWORK:MULTIZONE:COMPONENT:HORIZONTALOPENING",
                    Name=comp_name,
                    Air_Mass_Flow_Coefficient_When_Opening_is_Closed=0.001,
                    Air_Mass_Flow_Exponent_When_Opening_is_Closed=0.65,
                    Sloping_Plane_Angle=90.0,
                    Discharge_Coefficient=0.2,
                )
            else:
                idf.newidfobject(
                    "AIRFLOWNETWORK:MULTIZONE:COMPONENT:SIMPLEOPENING",
                    Name=comp_name,
                    Air_Mass_Flow_Coefficient_When_Opening_is_Closed=0.001,
                    Air_Mass_Flow_Exponent_When_Opening_is_Closed=0.65,
                    Minimum_Density_Difference_for_TwoWay_Flow=0.0001,
                    Discharge_Coefficient=0.2,
                )
            made_components.add(comp_name)

        # Find the shared partition fenestration (your existing approach)
        match = None
        for fen in idf.idfobjects.get("FENESTRATIONSURFACE:DETAILED", []):
            bs = fen.Building_Surface_Name.lower()
            ob = fen.Outside_Boundary_Condition_Object.lower()
            if z1.lower() in bs and z2.lower() in ob:
                match = fen
                break
            if z2.lower() in bs and z1.lower() in ob:
                match = fen
                break
        if not match:
            continue

        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:SURFACE",
            Surface_Name=match.Name,
            Leakage_Component_Name=comp_name,
            External_Node_Name="",  # inter-zone
            Ventilation_Control_Mode="Constant",
            Venting_Availability_Schedule_Name=sched,
        )
        count += 1
    return count

# --------------------------
# Validation / patch helpers
# --------------------------

def validate_afn(idf: IDF) -> list[str]:
    issues = []
    ext_nodes = {n.Name for n in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE", [])}
    for afn in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:SURFACE", []):
        name = afn.Surface_Name
        geo = next((s for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"] if s.Name == name), None)
        if not geo:
            geo = next((f for f in idf.idfobjects.get("FENESTRATIONSURFACE:DETAILED", []) if f.Name == name), None)
        if not geo:
            issues.append(f"Missing geometry for AFN surface: {name}")
            continue
        if hasattr(geo, "Outside_Boundary_Condition"):
            bc = geo.Outside_Boundary_Condition.lower()
        else:
            parent = next((s for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"] if s.Name == geo.Building_Surface_Name), None)
            bc = parent.Outside_Boundary_Condition.lower() if parent else ""
        if bc == "outdoors" and not afn.External_Node_Name:
            issues.append(f"Outdoors surface needs external node: {name}")
        if bc == "outdoors" and afn.External_Node_Name and afn.External_Node_Name not in ext_nodes:
            issues.append(f"External node not defined: {afn.External_Node_Name}")
        if bc != "outdoors" and afn.External_Node_Name:
            issues.append(f"Internal surface must not set external node: {name}")
    return issues


def add_fenestration_cracks(
    idf: IDF,
    building_config,
    cp_array_name: str = "NormalExposureCpArray",
) -> int:
    """
    Add AFN cracks to all fenestrations (windows, doors, vents), both internal and external.
    Uses crack templates referenced in building_config['cracks'], scaled by perimeter length.
    """

    cracks_cfg = building_config.cracks
    bsurfs = {s.Name: s for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]}
    wpc_vals = {v.Name for v in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTVALUES", [])}
    ext_nodes = {n.Name for n in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE", [])}
    cp_vals = [0.4, 0.1, -0.3, -0.35, -0.2, -0.35, -0.3, -0.1]
    added = 0

    for fen in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
        st = fen.Surface_Type.lower()
        nm = fen.Name.lower()
        if st not in ("window", "door"):
            continue

        parent = bsurfs.get(fen.Building_Surface_Name)
        if not parent:
            continue

        bc = parent.Outside_Boundary_Condition.lower()
        is_external = bc == "outdoors"

        if st == "window" and is_external:
            element_key = "external_window"
        elif st == "door" and is_external and "vent" not in nm:
            element_key = "external_door"
        elif st == "door" and not is_external:
            element_key = "internal_door"
        else:
            element_key = "external_vent"

        tmpl = get_template(cracks_cfg, parent.Zone_Name, element_key)
        if not tmpl:
            continue

        ccq_per_m = tmpl.get("cq_per_m", 0.0)
        n = tmpl.get("n", 0.65)
        verts = get_vertices(fen)
        perim = perimeter(verts)
        coef = ccq_per_m * perim
        if coef <= 0:
            continue

        ext_node = ""
        if is_external:
            if fen.Name not in wpc_vals:
                idf.newidfobject(
                    "AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTVALUES",
                    Name=fen.Name,
                    AirflowNetworkMultiZoneWindPressureCoefficientArray_Name=cp_array_name,
                    **{f"Wind_Pressure_Coefficient_Value_{i+1}": v for i, v in enumerate(cp_vals)}
                )
                wpc_vals.add(fen.Name)
            if fen.Name not in ext_nodes:
                idf.newidfobject(
                    "AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE",
                    Name=fen.Name,
                    External_Node_Height=surface_mid_height(parent),
                    Wind_Pressure_Coefficient_Curve_Name=fen.Name,
                )
                ext_nodes.add(fen.Name)
            ext_node = fen.Name
        elif fen.Outside_Boundary_Condition_Object:
            ext_node = fen.Outside_Boundary_Condition_Object

        crack_name = f"{fen.Name}_FrameCrack"
        if not any(c.Name == crack_name for c in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:SURFACE:CRACK", [])):
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:SURFACE:CRACK",
                Name=crack_name,
                Air_Mass_Flow_Coefficient_at_Reference_Conditions=coef,
                Air_Mass_Flow_Exponent=n,
                Reference_Crack_Conditions="ReferenceCrackConditions",
            )

        if not any(a.Surface_Name == fen.Name for a in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:SURFACE", [])):
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:SURFACE",
                Surface_Name=fen.Name,
                Leakage_Component_Name=crack_name,
                External_Node_Name=ext_node,
                WindowDoor_Opening_Factor_or_Crack_Factor=1.0,
                Ventilation_Control_Mode="Constant",
                Venting_Availability_Schedule_Name="AlwaysOnSchedule",
            )
            added += 1

        if not is_external and fen.Outside_Boundary_Condition_Object:
            paired = fen.Outside_Boundary_Condition_Object
            if not any(a.Surface_Name == paired for a in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:SURFACE", [])):
                idf.newidfobject(
                    "AIRFLOWNETWORK:MULTIZONE:SURFACE",
                    Surface_Name=paired,
                    Leakage_Component_Name=crack_name,
                    External_Node_Name=fen.Name,
                    WindowDoor_Opening_Factor_or_Crack_Factor=1.0,
                    Ventilation_Control_Mode="Constant",
                    Venting_Availability_Schedule_Name="AlwaysOnSchedule",
                )
                added += 1

    return added




def ensure_vent_construction(idf: IDF) -> str:
    """Create a simple opaque construction for vents if missing; return its name."""
    cons_name = "AFN_AirBrickConstruction"
    mat_name = "AFN_AirBrickPanel"
    if not any(m.Name == mat_name for m in idf.idfobjects.get("MATERIAL:NOMASS", [])):
        idf.newidfobject(
            "MATERIAL:NOMASS",
            Name=mat_name,
            Roughness="Rough",
            Thermal_Resistance=0.1,  # arbitrary; tiny area so thermal impact is negligible
        )
    if not any(c.Name == cons_name for c in idf.idfobjects.get("CONSTRUCTION", [])):
        idf.newidfobject(
            "CONSTRUCTION",
            Name=cons_name,
            Outside_Layer=mat_name,
        )
    return cons_name

def add_subfloor_air_bricks(
    idf: IDF,
    per_wall: int = 2,
    vent_area: float = 0.01,        # m²
    exclude_azimuth: float = 90.0,  # degrees (skip walls ≈ 90°)
    z_clear: float = 0.20           # vent bottom above wall bottom (m)
) -> int:
    """
    For each Subfloor external wall (except azimuth≈exclude_azimuth), add `per_wall`
    fenestration vents (Door surfaces) of area `vent_area`, place near bottom, and
    attach AFN as a SimpleOpening held open (opening factor = 1.0).
    """
    cons = ensure_vent_construction(idf)
    made = 0

    def _get_vertices(s):
        verts = []
        for i in range(1, 501):
            x = getattr(s, f"Vertex_{i}_Xcoordinate", "")
            if x == "":
                break
            y = getattr(s, f"Vertex_{i}_Ycoordinate")
            z = getattr(s, f"Vertex_{i}_Zcoordinate")
            verts.append((float(x), float(y), float(z)))
        return verts

    def _unit(v):
        import math
        x, y, z = v
        n = math.sqrt(x*x + y*y + z*z)
        return (0.0, 0.0, 0.0) if n == 0 else (x/n, y/n, z/n)

    def _sub(a, b):
        return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

    def _add(p, a, scale=1.0):
        return (p[0] + a[0]*scale, p[1] + a[1]*scale, p[2] + a[2]*scale)

    def _approx(a, b, tol=1.0):
        return abs(float(a) - float(b)) <= tol

    def _get_azimuth(surface):
        return float(getattr(surface, "Azimuth", getattr(surface, "azimuth", 0.0)))

    # pick Subfloor external walls to ventilate
    walls = [s for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]
             if s.Zone_Name.lower() == "subfloor"
             and s.Surface_Type.lower() == "wall"
             and s.Outside_Boundary_Condition.lower() == "outdoors"
             and not _approx(_get_azimuth(s), exclude_azimuth)]

    # ensure each parent wall has an ExternalNode
    for s in walls:
        if not any(n.Name == s.Name for n in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE", [])):
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE",
                Name=s.Name,
                External_Node_Height=surface_mid_height(s),
                Wind_Pressure_Coefficient_Curve_Name=s.Name,
            )

    import math
    for s in walls:
        verts = _get_vertices(s)
        if len(verts) < 4:
            continue

        # bottom edge: two lowest-z vertices
        sort_by_z = sorted(verts, key=lambda t: t[2])
        low1, low2 = sort_by_z[0], sort_by_z[1]
        base_z = min(low1[2], low2[2]) + z_clear

        # along-wall unit vector + vertical
        u = _unit(_sub(low2, low1))
        v = (0.0, 0.0, 1.0)

        # rectangle size to get area = vent_area (choose width, compute height)
        width = 0.20
        height = vent_area / max(width, 1e-6)

        # positions along the wall (fractions 1/3 and 2/3 from one end)
        edge_vec = _sub(low2, low1)
        edge_len = math.sqrt(edge_vec[0]**2 + edge_vec[1]**2 + edge_vec[2]**2)
        frac_positions = [(i+1)/(per_wall+1) for i in range(per_wall)]

        for idx, f in enumerate(frac_positions, start=1):
            center = _add(low1, u, f * edge_len)
            center = (center[0], center[1], base_z + height/2.0)

            # CCW rectangle vertices
            v1 = _add(_add(center, u, -width/2.0), v, -height/2.0)
            v2 = _add(_add(center, u,  width/2.0), v, -height/2.0)
            v3 = _add(_add(center, u,  width/2.0), v,  height/2.0)
            v4 = _add(_add(center, u, -width/2.0), v,  height/2.0)

            fen_name = f"{s.Name}_Vent_{idx}"
            if any(fen.Name == fen_name for fen in idf.idfobjects.get("FENESTRATIONSURFACE:DETAILED", [])):
                continue

            # Opaque “Door” fenestration to represent the vent opening (area = 0.01 m²)
            idf.newidfobject(
                "FENESTRATIONSURFACE:DETAILED",
                Name=fen_name,
                Surface_Type="Door",
                Construction_Name=cons,
                Building_Surface_Name=s.Name,
                View_Factor_to_Ground="Autocalculate",
                Frame_and_Divider_Name="",
                Multiplier=1.0,
                Number_of_Vertices=4,
                Vertex_1_Xcoordinate=v1[0], Vertex_1_Ycoordinate=v1[1], Vertex_1_Zcoordinate=v1[2],
                Vertex_2_Xcoordinate=v2[0], Vertex_2_Ycoordinate=v2[1], Vertex_2_Zcoordinate=v2[2],
                Vertex_3_Xcoordinate=v3[0], Vertex_3_Ycoordinate=v3[1], Vertex_3_Zcoordinate=v3[2],
                Vertex_4_Xcoordinate=v4[0], Vertex_4_Ycoordinate=v4[1], Vertex_4_Zcoordinate=v4[2],
            )

            # SimpleOpening component (fixed-open behavior handled by opening factor = 1.0)
            comp_name = f"{fen_name}_SimpleOpening"
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:COMPONENT:SIMPLEOPENING",
                Name=comp_name,
                Air_Mass_Flow_Coefficient_When_Opening_is_Closed=0.001,
                Air_Mass_Flow_Exponent_When_Opening_is_Closed=0.65,
                Minimum_Density_Difference_for_TwoWay_Flow=0.0001,
                Discharge_Coefficient=0.65,
            )

            # AFN surface linking fenestration to SimpleOpening, always fully open
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:SURFACE",
                Surface_Name=fen_name,
                Leakage_Component_Name=comp_name,
                External_Node_Name=s.Name,  # parent wall node
                WindowDoor_Opening_Factor_or_Crack_Factor=1.0,  # 100% open
                Ventilation_Control_Mode="Constant",
                Venting_Availability_Schedule_Name="AlwaysOnSchedule",
            )

            made += 1

    return made


# --------------------------
# Audits & guard rails
# --------------------------

def audit_afn_counts(idf: IDF):
    surf_by_name = {s.Name: s for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]}
    counts = defaultdict(int)
    for afn in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:SURFACE", []):
        parent = surf_by_name.get(afn.Surface_Name)
        if parent:
            counts[parent.Zone_Name] += 1
    print("AFN surfaces per zone:")
    for z in sorted(counts):
        print(f"  {z}: {counts[z]}")
    missing = [z.Name for z in idf.idfobjects["ZONE"] if counts.get(z.Name, 0) < 2]
    if missing:
        print("Zones with <2 AFN surfaces:", missing)
    return counts

def ensure_min_two_afn_per_zone(idf: IDF, tiny_coef=5e-5, tiny_exp=0.65):
    surf_by_name = {s.Name: s for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]}
    afn_surfs = idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:SURFACE", [])
    by_zone = defaultdict(list)
    for afn in afn_surfs:
        parent = surf_by_name.get(afn.Surface_Name)
        if parent:
            by_zone[parent.Zone_Name].append(afn)

    existing_afn_names = {a.Surface_Name for a in afn_surfs}
    ext_nodes = {n.Name for n in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE", [])}
    wpc_vals = {v.Name for v in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTVALUES", [])}
    cp_vals = [0.4, 0.1, -0.3, -0.35, -0.2, -0.35, -0.3, -0.1]

    for z in idf.idfobjects["ZONE"]:
        n = len(by_zone.get(z.Name, []))
        if n >= 2:
            continue
        candidates = [s for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]
                      if s.Zone_Name == z.Name
                      and s.Outside_Boundary_Condition.lower() == "outdoors"
                      and s.Name not in existing_afn_names]
        if not candidates:
            print(f"⚠️ No extra Outdoors surface found for zone '{z.Name}' to reach >=2 AFN paths.")
            continue

        s = candidates[0]
        crack = f"{s.Name}_TinyCrack"
        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:SURFACE:CRACK",
            Name=crack,
            Air_Mass_Flow_Coefficient_at_Reference_Conditions=tiny_coef,
            Air_Mass_Flow_Exponent=tiny_exp,
            Reference_Crack_Conditions="ReferenceCrackConditions",
        )
        if s.Name not in wpc_vals:
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTVALUES",
                Name=s.Name,
                AirflowNetworkMultiZoneWindPressureCoefficientArray_Name="NormalExposureCpArray",
                **{f"Wind_Pressure_Coefficient_Value_{i+1}": v for i, v in enumerate(cp_vals)}
            )
            wpc_vals.add(s.Name)
        if s.Name not in ext_nodes:
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE",
                Name=s.Name,
                External_Node_Height=surface_mid_height(s),
                Wind_Pressure_Coefficient_Curve_Name=s.Name,
            )
            ext_nodes.add(s.Name)
        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:SURFACE",
            Surface_Name=s.Name,
            Leakage_Component_Name=crack,
            External_Node_Name=s.Name,
            Ventilation_Control_Mode="ZoneLevel",
        )
        print(f"Added tiny AFN path on '{s.Name}' for zone '{z.Name}' to reach >=2.")

# --------------------------
# Main entry
# --------------------------

def add_airflow_network(idf: IDF, building_config=None) -> IDF:
    """Build AFN: controls, zones, DB-poor per-m² cracks on Outdoors walls/roofs,
    internal openings, plus subfloor vents."""
    ensure_basic_schedules(idf)
    ensure_reference_crack_conditions(idf)
    setup_afn_controls(idf)
    ensure_afn_zones(idf)

    # Per-m² cracks on surfaces (walls, roofs, etc.)
    surface_cracks = add_surface_leakage(idf, building_config)

    print("Added AFN Surface cracks:", surface_cracks)

    # Add air bricks on Subfloor external walls (skip azimuth ≈ 90°)
    add_subfloor_air_bricks(idf, per_wall=2, vent_area=0.01, exclude_azimuth=90.0)

    # Internal openings (if provided)
    add_internal_openings(idf, building_config)

    # Per-m cracks on fenestration surfaces (windows, doors, vents)
    n_added = add_fenestration_cracks(
        idf,
        building_config
    )

    print("Added AFN fenestration cracks:", n_added)

    # Audit & guard rail
    audit_afn_counts(idf)
    ensure_min_two_afn_per_zone(idf)

    return idf
