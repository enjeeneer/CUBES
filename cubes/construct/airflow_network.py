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

FACT = 0.75

CRACK_TEMPLATES = {
    "not_bad": {
        # Building surfaces - area-based (kg/s·m² at 1 Pa)
        "external_wall":  {"cq_per_m2": 0.0002 *FACT,  "n": 0.7},
        "internal_wall":  {"cq_per_m2": 0.005 *FACT,   "n": 0.75},
        "internal_floor": {"cq_per_m2": 0.002 *FACT,   "n": 0.7},
        "internal_ceiling": {"cq_per_m2": 0.002 *FACT,   "n": 0.7},
        "external_floor": {"cq_per_m2": 0.001 *FACT,   "n": 1.0},
        "external_roof":  {"cq_per_m2": 0.00015 *FACT, "n": 0.7},

        # Fenestrations - perimeter-based (kg/s·m at 1 Pa)
        "external_window": {"cq_per_m": 0.001 * FACT,  "n": 0.6},
        "external_door":   {"cq_per_m": 0.0018 * FACT, "n": 0.66},
        "internal_door":   {"cq_per_m": 0.02 * FACT,   "n": 0.6, "cd": 0.65},
        "external_vent":   {"cq_per_m": 0.01 * FACT,   "n": 0.66, "cd": 0.65}
    },
    "poor": {
        # Building surfaces - area-based (kg/s·m² at 1 Pa)
        "external_wall":  {"cq_per_m2": 0.0002,  "n": 0.7},
        "internal_wall":  {"cq_per_m2": 0.005,   "n": 0.75},
        "internal_floor": {"cq_per_m2": 0.002,   "n": 0.7},
        "internal_ceiling": {"cq_per_m2": 0.002,   "n": 0.7},
        "external_floor": {"cq_per_m2": 0.001,   "n": 1.0},
        "external_roof":  {"cq_per_m2": 0.00015, "n": 0.7},

        # Fenestrations - perimeter-based (kg/s·m at 1 Pa)
        "external_window": {"cq_per_m": 0.001,  "n": 0.6},
        "external_door":   {"cq_per_m": 0.0018, "n": 0.66},
        "internal_door":   {"cq_per_m": 0.02,   "n": 0.6, "cd": 0.65},
        "external_vent":   {"cq_per_m": 0.01,   "n": 0.66, "cd": 0.65}
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
    "good": {
        "external_wall":  {"cq_per_m2": 0.0002,  "n": 0.7},
        "internal_wall":  {"cq_per_m2": 0.005,   "n": 0.75},
        "internal_floor": {"cq_per_m2": 0.002,   "n": 0.7},
        "internal_ceiling": {"cq_per_m2": 0.002,   "n": 0.7},
        "external_floor": {"cq_per_m2": 0.001,   "n": 1.0},
        "external_roof":  {"cq_per_m2": 0.00015, "n": 0.7},
        "external_window":{"cq_per_m":  0.00006,   "n": 0.6}, # actually from beizaee thesis
        "external_door":  {"cq_per_m":  0.0018,  "n": 0.66},
        "internal_door":  {"cq_per_m":  0.02,    "n": 0.6},
        "external_vent":  {"cq_per_m":  0.01,    "n": 0.66, "cd": 0.65}
    },
}


# Opening type → Crack template key mapping
OPENING_TEMPLATE_MAP = {
    "internal_door": "internal_door",
    "external_door": "external_door",
    "window": "external_window",
    "vent": "external_vent",
    "hole": "internal_door"  # holes use door template
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
    """Get crack template with zone-specific scaling factors applied."""
    zone_cfg = cracks_cfg.get(zone.lower(), {})
    default_cfg = cracks_cfg.get("default", {})

    zone_entry = zone_cfg.get(element_key)
    if isinstance(zone_entry, dict):
        template_label = default_cfg.get(element_key)
        modifiers = zone_entry
    else:
        template_label = zone_entry or default_cfg.get(element_key)
        modifiers = {}

    if not template_label:
        print(f"⚠️ No crack template found for {zone}:{element_key}")
        return None

    base_tmpl = CRACK_TEMPLATES.get(template_label, {}).get(element_key)
    if not base_tmpl:
        print(f"⚠️ Missing crack definition for {template_label}:{element_key}")
        return None

    # Apply modifiers
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
            ext_node = ""  # ✅ must be blank for interzone surfaces
        else:
            scope = "indoors"
            element_key = f"internal_{stype}"
            ext_node = ""


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

def resolve_opening_schedule(schedule_val: str) -> str:
    """Convert schedule value to EnergyPlus schedule name."""
    s = str(schedule_val).lower()
    if s == "1":
        return "AlwaysOnSchedule"
    if s == "0":
        return "AlwaysOffSchedule"
    return f"door-schedule-{s}"

def add_opening_components(idf: IDF, building_config, cp_array_name: str = "NormalExposureCpArray") -> int:
    """
    Add AFN opening components to all fenestrations defined in openings config.
    Creates SimpleOpening or HorizontalOpening with template-based crack coefficients.

    Returns: Number of AFN surfaces created
    """
    openings_cfg = getattr(building_config, "openings", [])
    if not openings_cfg:
        print("No openings defined in building configuration.")
        return 0

    cracks_cfg = building_config.cracks
    bsurfs = {s.Name: s for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]}
    fens = {f.Name: f for f in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]}

    wpc_vals = {v.Name for v in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTVALUES", [])}
    ext_nodes = {n.Name for n in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE", [])}
    cp_vals = [0.4, 0.1, -0.3, -0.35, -0.2, -0.35, -0.3, -0.1]

    count = 0

    for opening in openings_cfg:
        opening_type = opening["type"]
        zones = [z.lower() for z in opening["zones"]]
        schedule = resolve_opening_schedule(opening.get("schedule", "1"))

        # Map opening type to crack template key
        element_key = OPENING_TEMPLATE_MAP.get(opening_type)
        if not element_key:
            print(f"⚠️ Unknown opening type: {opening_type}, skipping.")
            continue

        # Determine zone for template lookup
        zone = zones[0] if zones[0] != "outdoors" else zones[1]

        # Get crack template
        tmpl = get_template(cracks_cfg, zone, element_key)
        if not tmpl:
            tmpl = {"cq_per_m": 0.001, "n": 0.65, "cd": 0.65}  # fallback

        # Find corresponding fenestration surface
        fen = _find_fenestration_by_opening(fens, opening, zones)
        if not fen:
            print(f"⚠️ No fenestration found for {opening_type} in {zones}")
            continue


        # Doors/windows/holes use perimeter-based cracks
        perim = perimeter(get_vertices(fen))
        closed_coef = tmpl.get("cq_per_m", 0.001) * perim
        closed_exp = tmpl.get("n", 0.65)
        cd = tmpl.get("cd", 0.65)

        # Create opening component
        comp_name = f"{fen.Name}_Opening"
        if opening_type == "hole":
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:COMPONENT:HORIZONTALOPENING",
                Name=comp_name,
                Air_Mass_Flow_Coefficient_When_Opening_is_Closed=closed_coef,
                Air_Mass_Flow_Exponent_When_Opening_is_Closed=closed_exp,
                Sloping_Plane_Angle=90.0,
                Discharge_Coefficient=0.6
            )
        elif opening_type == "internal_door":
            verts = get_vertices(fen)
            w = abs(verts[0][0] - verts[1][0]) if abs(verts[0][0] - verts[1][0]) > 0 else abs(verts[0][1] - verts[1][1])
            h = abs(verts[0][2] - verts[2][2])
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:COMPONENT:DETAILEDOPENING",
                Name=comp_name,
                Air_Mass_Flow_Coefficient_When_Opening_is_Closed=closed_coef,
                Air_Mass_Flow_Exponent_When_Opening_is_Closed=closed_exp,
                Type_of_Rectangular_Large_Vertical_Opening_LVO="NonPivoted",
                Extra_Crack_Length_or_Height_of_Pivoting_Axis=0.0,
                Number_of_Sets_of_Opening_Factor_Data=2,
                Opening_Factor_1=0.0,
                Discharge_Coefficient_for_Opening_Factor_1=0.001,
                Width_Factor_for_Opening_Factor_1=0.0,
                Height_Factor_for_Opening_Factor_1=0.0,
                Start_Height_Factor_for_Opening_Factor_1=0.0,
                Opening_Factor_2=1.0,
                Discharge_Coefficient_for_Opening_Factor_2=cd,
                Width_Factor_for_Opening_Factor_2=1.0,
                Height_Factor_for_Opening_Factor_2=1.0,
                Start_Height_Factor_for_Opening_Factor_2=0.0
            )

        else:
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:COMPONENT:SIMPLEOPENING",
                Name=comp_name,
                Air_Mass_Flow_Coefficient_When_Opening_is_Closed=closed_coef,
                Air_Mass_Flow_Exponent_When_Opening_is_Closed=closed_exp,
                Minimum_Density_Difference_for_TwoWay_Flow=0.0001,
                Discharge_Coefficient=cd
            )


        print(f"✅ Created {opening_type} component {comp_name}: coef={closed_coef:.6f}, n={closed_exp:.2f}")

        # Setup external node if needed
        parent = bsurfs.get(fen.Building_Surface_Name)
        if not parent:
            print(f"⚠️ Parent surface not found for {fen.Name}")
            continue

        is_external = parent.Outside_Boundary_Condition.lower() == "outdoors"

        if is_external:
            # Setup wind pressure coefficients
            if fen.Name not in wpc_vals:
                idf.newidfobject(
                    "AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTVALUES",
                    Name=fen.Name,
                    AirflowNetworkMultiZoneWindPressureCoefficientArray_Name=cp_array_name,
                    **{f"Wind_Pressure_Coefficient_Value_{i+1}": v for i, v in enumerate(cp_vals)}
                )
                wpc_vals.add(fen.Name)

            # Setup external node
            if fen.Name not in ext_nodes:
                idf.newidfobject(
                    "AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE",
                    Name=fen.Name,
                    External_Node_Height=surface_mid_height(parent),
                    Wind_Pressure_Coefficient_Curve_Name=fen.Name,
                )
                ext_nodes.add(fen.Name)

            ext_node = fen.Name
        else:
            ext_node = ""

        # Create AFN surface
        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:SURFACE",
            Surface_Name=fen.Name,
            Leakage_Component_Name=comp_name,
            External_Node_Name=ext_node,
            WindowDoor_Opening_Factor_or_Crack_Factor=1.0,
            Ventilation_Control_Mode="Constant",
            Venting_Availability_Schedule_Name=schedule,
        )
        count += 1
        print(f"✅ Added AFN surface for {fen.Name} with schedule {schedule}")

    return count


def _find_fenestration_by_opening(fens: dict, opening: dict, zones: list) -> Optional[object]:
    """Find fenestration surface matching the opening definition."""
    opening_type = opening["type"]

    suffix_map = {
        "internal_door": "door",
        "external_door": "extdoor",
        "window": "window",
        "vent": "vent",
        "hole": "hole"
    }
    suffix = suffix_map.get(opening_type, "")

    # Just search for any fenestration ending with the suffix and containing zone name
    zone = zones[0] if "outdoors" not in zones else (zones[0] if zones[1] == "outdoors" else zones[1])

    for fen in fens.values():
        fen_name = fen.Name.lower()
        # Match if fenestration name contains zone and ends with suffix
        if opening.get('azimuth'):
            if zone.lower() in fen_name and fen_name.endswith(f"_{suffix}") and fen.azimuth == opening.get('azimuth'):
                return fen
        else:
            if zone.lower() in fen_name and fen_name.endswith(f"_{suffix}"):
                return fen


    return None

    # Pick based on minimal azimuth difference
    def az_diff(wall_az, target):
        return abs((float(wall_az) - target + 180) % 360 - 180)

    return min(
        candidates,
        key=lambda fen: az_diff(bsurfs[fen.Building_Surface_Name].azimuth, target_az)
    )

# def _find_fenestration_by_opening(fens, opening, zones, bsurfs):
#     """
#     Correct AFN matching:
#       - windows/external doors → external fenestrations with matching azimuth
#       - internal doors → fenestrations whose parent surfaces connect the two zones
#       - vents → fenestrations containing 'vent' in their name
#       - holes → parent surfaces between two zones (not fenestrations)
#     """

#     opening_type = opening["type"]
#     zone_a, zone_b = zones[0].lower(), zones[1].lower()
#     target_az = opening.get("azimuth")

#     # ---------------------------------------------------------
#     # Helper: identify which two zones a fenestration connects
#     # ---------------------------------------------------------
#     def zones_connected_by_fenestration(f):
#         parent = bsurfs[f.Building_Surface_Name]

#         if parent.Outside_Boundary_Condition.lower() != "surface":
#             # external → only one zone
#             return (parent.Zone_Name.lower(), None)

#         other_parent_name = parent.Outside_Boundary_Condition_Object
#         other_parent = bsurfs.get(other_parent_name)

#         if not other_parent:
#             return (parent.Zone_Name.lower(), None)

#         return (parent.Zone_Name.lower(), other_parent.Zone_Name.lower())

#     # ---------------------------------------------------------
#     # 1) WINDOWS + EXTERNAL DOORS (external fenestrations)
#     # ---------------------------------------------------------
#     if opening_type in {"window", "external_door"}:

#         candidates = [
#             f for f in fens.values()
#             if bsurfs[f.Building_Surface_Name].Zone_Name.lower() == zone_a
#             and bsurfs[f.Building_Surface_Name].Outside_Boundary_Condition.lower() == "outdoors"
#         ]

#         if not candidates:
#             return None

#         if target_az is None:
#             return max(candidates, key=lambda f: f.area)

#         def az_diff(wall_az, target):
#             return abs((wall_az - target + 180) % 360 - 180)

#         return min(
#             candidates,
#             key=lambda fen: az_diff(bsurfs[fen.Building_Surface_Name].azimuth, target_az)
#         )

#     # ---------------------------------------------------------
#     # 2) INTERNAL DOORS
#     # ---------------------------------------------------------
#     if opening_type == "internal_door":

#         candidates = []
#         for f in fens.values():
#             z1, z2 = zones_connected_by_fenestration(f)

#             if {z1, z2} == {zone_a, zone_b}:  # two-way match
#                 candidates.append(f)

#         if not candidates:
#             return None

#         req_area = opening.get("area")
#         if req_area:
#             return min(candidates, key=lambda f: abs(f.area - req_area))

#         return max(candidates, key=lambda f: f.area)

#     # ---------------------------------------------------------
#     # 3) VENTS (fenestrations containing "vent")
#     # ---------------------------------------------------------
#     if opening_type == "vent":
#         candidates = [
#             f for f in fens.values()
#             if "vent" in f.Name.lower()
#             and bsurfs[f.Building_Surface_Name].Zone_Name.lower() == zone_a
#         ]

#         if not candidates:
#             return None

#         return min(candidates, key=lambda f: f.area)

#     # ---------------------------------------------------------
#     # 4) HOLES (fenestrations of type Door with "hole" in name)
#     # ---------------------------------------------------------
#     if opening_type == "hole":
#         candidates = []
#         for f in fens.values():
#             if "hole" not in f.Name.lower():
#                 continue
#             if f.Surface_Type.lower() != "door":
#                 continue

#             parent = bsurfs[f.Building_Surface_Name]

#             if parent.Outside_Boundary_Condition.lower() != "surface":
#                 continue

#             other_name = parent.Outside_Boundary_Condition_Object
#             other_parent = bsurfs.get(other_name)
#             if not other_parent:
#                 continue

#             z1 = parent.Zone_Name.lower()
#             z2 = other_parent.Zone_Name.lower()

#             if {z1, z2} == {zone_a, zone_b}:
#                 candidates.append(f)

#         if not candidates:
#             return None

#         req_area = opening.get("area", 1.0)
#         return min(candidates, key=lambda f: abs(f.area - req_area))


#     # ---------------------------------------------------------
#     # 5) fallback – should not be needed
#     # ---------------------------------------------------------
#     return None




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


def audit_paired_fenestrations(idf: IDF):
    fens = {f.Name: f for f in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]}
    for f in fens.values():
        paired = f.Outside_Boundary_Condition_Object
        if paired and paired in fens:
            if fens[paired].Outside_Boundary_Condition_Object != f.Name:
                print(f"⚠️ Mismatch: {f.Name} ↔ {paired} not reciprocally linked")

# --------------------------
# Main entry
# --------------------------

def add_airflow_network(idf: IDF, building_config=None) -> IDF:
    """
    Build AFN network:
    1. Setup controls and zones
    2. Add cracks to all building surfaces
    3. Add opening components to all fenestrations
    4. Validate
    """
    ensure_basic_schedules(idf)
    ensure_reference_crack_conditions(idf)
    setup_afn_controls(idf)
    ensure_afn_zones(idf)

    # Add cracks to building surfaces (walls, roofs, floors)
    surface_cracks = add_surface_leakage(idf, building_config)
    print(f"✅ Added {surface_cracks} AFN surface cracks")

    # Add opening components to fenestrations (doors, windows, vents)
    opening_count = add_opening_components(idf, building_config)
    print(f"✅ Added {opening_count} AFN opening components")

    # Validate
    audit_paired_fenestrations(idf)
    audit_afn_counts(idf)
    ensure_min_two_afn_per_zone(idf)

    return idf