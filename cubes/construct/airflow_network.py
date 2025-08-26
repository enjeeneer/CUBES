"""airflow_network.py

Minimal AFN builder for façade leakage on external walls and explicit internal
openings. Uses WPC=Input with ExternalNode height selection. Internal leakage
is ignored except for user-specified internal openings.

Requires: eppy.modeleditor.IDF
"""

from collections import defaultdict
from eppy.modeleditor import IDF
from typing import Optional


def ensure_basic_schedules(idf: IDF) -> None:
    """Ensure AlwaysOnSchedule and AlwaysOffSchedule exist."""
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
    """Create AFN reference crack conditions once."""
    if not idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:REFERENCECRACKCONDITIONS"):
        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:REFERENCECRACKCONDITIONS",
            Name="ReferenceCrackConditions",
            Reference_Temperature=20.0,
            Reference_Barometric_Pressure=101325,
            Reference_Humidity_Ratio=0.0,
        )


def setup_afn_controls(idf: IDF, cp_array_name: str = "NormalExposureCpArray") -> None:
    """Create AFN simulation control and the shared WPC array."""
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
    """Create one AFN zone object per Zone."""
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
    """Return mid-height from up to four Z vertices; fallback to 1.5 m."""
    zs = []
    for i in range(1, 5):
        val = getattr(s, f"Vertex_{i}_Zcoordinate", "")
        if val != "":
            zs.append(float(val))
    return sum(zs) / len(zs) if zs else 1.5


def add_external_wall_leakage(
    idf: IDF,
    cp_array_name: str = "NormalExposureCpArray",
    crack_params: Optional[dict] = None,
) -> int:
    """Add façade leakage on external walls only with per-type coefficients."""
    defaults = {"wall": (0.0002, 0.70)}
    params = {"wall": tuple(crack_params.get("wall", defaults["wall"]))} if \
        crack_params else defaults
    cp_wall = [0.4, 0.1, -0.3, -0.35, -0.2, -0.35, -0.3, -0.1]
    count = 0
    for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
        if s.Outside_Boundary_Condition.lower() != "outdoors":
            continue
        if s.Surface_Type.lower() != "wall":
            continue
        if not any(v.Name == s.Name for v in
                   idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTVALUES", [])):
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTVALUES",
                Name=s.Name,
                AirflowNetworkMultiZoneWindPressureCoefficientArray_Name=cp_array_name,
                **{f"Wind_Pressure_Coefficient_Value_{i+1}": v
                   for i, v in enumerate(cp_wall)}
            )
        if not any(n.Name == s.Name for n in
                   idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE", [])):
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE",
                Name=s.Name,
                External_Node_Height=surface_mid_height(s),
                Wind_Pressure_Coefficient_Curve_Name=s.Name,
            )
        coef, expn = params["wall"]
        crack_name = f"{s.Name}_Crack"
        if not any(c.Name == crack_name for c in
                   idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:SURFACE:CRACK", [])):
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:SURFACE:CRACK",
                Name=crack_name,
                Air_Mass_Flow_Coefficient_at_Reference_Conditions=coef,
                Air_Mass_Flow_Exponent=expn,
                Reference_Crack_Conditions="ReferenceCrackConditions",
            )
        if not any(afn.Surface_Name == s.Name for afn in
                   idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:SURFACE", [])):
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:SURFACE",
                Surface_Name=s.Name,
                Leakage_Component_Name=crack_name,
                External_Node_Name=s.Name,
                Ventilation_Control_Mode="ZoneLevel",
            )
        count += 1
    return count


def resolve_opening_schedule(opening: dict) -> str:
    """Map opening.schedule to a schedule name."""
    val = opening.get("schedule", "1")
    s = str(val).lower()
    if s == "1":
        return "AlwaysOnSchedule"
    if s == "0":
        return "AlwaysOffSchedule"
    return f"Occupancy-Schedule-{s}"


def add_internal_openings(idf: IDF, openings_cfg) -> int:
    """Add internal SimpleOpening or HorizontalOpening for zone-to-zone links.

    openings_cfg should be an iterable of dicts like:
      {"zones":["zone_a","zone_b"],"orientation":"vertical|horizontal",
       "schedule":"1|0|<name>"}
    """
    made_components = set()
    count = 0
    print(openings_cfg)
    print(type(openings_cfg))
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
            made_components.add(comp_name)
        match = None
        for fen in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
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
            External_Node_Name="",
            Ventilation_Control_Mode="Constant",
            Venting_Availability_Schedule_Name=sched,
        )
        count += 1
    return count


def validate_afn(idf: IDF) -> list[str]:
    """Return a list of human-readable AFN validation issues."""
    issues = []
    ext_nodes = {n.Name for n in
                 idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE", [])}
    for afn in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:SURFACE", []):
        name = afn.Surface_Name
        geo = next((s for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]
                    if s.Name == name), None)
        if not geo:
            geo = next((f for f in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]
                        if f.Name == name), None)
        if not geo:
            issues.append(f"Missing geometry for AFN surface: {name}")
            continue
        if hasattr(geo, "Outside_Boundary_Condition"):
            bc = geo.Outside_Boundary_Condition.lower()
        else:
            parent = next((s for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]
                           if s.Name == geo.Building_Surface_Name), None)
            bc = parent.Outside_Boundary_Condition.lower() if parent else ""
        if bc == "outdoors" and not afn.External_Node_Name:
            issues.append(f"Outdoors surface needs external node: {name}")
        if bc == "outdoors" and afn.External_Node_Name not in ext_nodes:
            issues.append(f"External node not defined: {afn.External_Node_Name}")
        if bc != "outdoors" and afn.External_Node_Name:
            issues.append(f"Internal surface must not set external node: {name}")
    return issues

def patch_outdoor_afn_surfaces(idf: IDF,
                               cp_array_name: str = "NormalExposureCpArray") -> int:
    """Ensure AFN surfaces with Outdoors boundary have valid ExternalNode/WPC."""
    cp_wall = [0.4, 0.1, -0.3, -0.35, -0.2, -0.35, -0.3, -0.1]
    ext_nodes = {n.Name for n in
                 idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE", [])}
    wpc_vals = {v.Name for v in
                idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTVALUES", [])}
    bsurfs = {s.Name: s for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]}
    fixed = 0
    for afn in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:SURFACE", []):
        geo = bsurfs.get(afn.Surface_Name)
        if not geo or geo.Outside_Boundary_Condition.lower() != "outdoors":
            continue
        if geo.Name not in wpc_vals:
            idf.newidfobject("AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTVALUES",
                             Name=geo.Name,
                             AirflowNetworkMultiZoneWindPressureCoefficientArray_Name=cp_array_name,
                             **{f"Wind_Pressure_Coefficient_Value_{i+1}": v
                                for i, v in enumerate(cp_wall)})
            wpc_vals.add(geo.Name)
        if geo.Name not in ext_nodes:
            idf.newidfobject("AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE",
                             Name=geo.Name,
                             External_Node_Height=surface_mid_height(geo),
                             Wind_Pressure_Coefficient_Curve_Name=geo.Name)
            ext_nodes.add(geo.Name)
        if not afn.External_Node_Name:
            afn.External_Node_Name = geo.Name
            fixed += 1

    print(f"Fixed {fixed}")
    return fixed

def add_subfloor_cracks_minimal(
    idf: IDF,
    zone_name: str = "Subfloor",
    cp_array_name: str = "NormalExposureCpArray",
    wall_coef: float = 1e-4,
    wall_exp: float = 0.65,
    ceil_coef: float = 5e-5,
    ceil_exp: float = 0.65,
) -> int:
    """Add minimal AFN cracks for the subfloor to ensure >=2 AFN surfaces.

    - Walls: add cracks; if boundary is Outdoors, create WPC+ExternalNode and set it.
    - Ceilings/Floors with Surface boundary: add cracks with blank external node.
    - Coefficients are small to avoid impacting results materially.
    Returns number of AFN surfaces created for the subfloor.
    """
    cp_wall = [0.4, 0.1, -0.3, -0.35, -0.2, -0.35, -0.3, -0.1]
    bsurfs = [s for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]
              if s.Zone_Name.lower() == zone_name.lower()]
    made = 0

    def ensure_wpc_and_node(name: str, surf) -> None:
        vals = idf.idfobjects.get(
            "AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTVALUES", []
        )
        nodes = idf.idfobjects.get(
            "AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE", []
        )
        if not any(v.Name == name for v in vals):
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTVALUES",
                Name=name,
                AirflowNetworkMultiZoneWindPressureCoefficientArray_Name=cp_array_name,
                **{f"Wind_Pressure_Coefficient_Value_{i+1}": v
                   for i, v in enumerate(cp_wall)}
            )
        if not any(n.Name == name for n in nodes):
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE",
                Name=name,
                External_Node_Height=surface_mid_height(surf),
                Wind_Pressure_Coefficient_Curve_Name=name,
            )

    # Prefer walls first (may be Outdoors), then ceilings/floors (Surface).
    ordered = [s for s in bsurfs if s.Surface_Type.lower() == "wall"] + \
              [s for s in bsurfs if s.Surface_Type.lower() in ("ceiling", "floor")]

    for s in ordered:
        afn_surfs = idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:SURFACE", [])
        if any(a.Surface_Name == s.Name for a in afn_surfs):
            continue
        is_wall = s.Surface_Type.lower() == "wall"
        crack = f"{s.Name}_Subfloor{'Wall' if is_wall else 'Surf'}Crack"
        coef, expn = (wall_coef, wall_exp) if is_wall else (ceil_coef, ceil_exp)
        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:SURFACE:CRACK",
            Name=crack,
            Air_Mass_Flow_Coefficient_at_Reference_Conditions=coef,
            Air_Mass_Flow_Exponent=expn,
            Reference_Crack_Conditions="ReferenceCrackConditions",
        )
        ext = ""
        if s.Outside_Boundary_Condition.lower() == "outdoors":
            ensure_wpc_and_node(s.Name, s)
            ext = s.Name
        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:SURFACE",
            Surface_Name=s.Name,
            Leakage_Component_Name=crack,
            External_Node_Name=ext,
            Ventilation_Control_Mode="ZoneLevel",
        )
        made += 1
        # Stop once we have at least two AFN surfaces for the subfloor
        if made >= 2:
            break
    return made


def add_airflow_network(idf: IDF, building_config=None, crack_params=None) -> IDF:
    """Build AFN: controls, zones, façade leakage on external walls, openings."""
    ensure_basic_schedules(idf)
    ensure_reference_crack_conditions(idf)
    setup_afn_controls(idf)
    ensure_afn_zones(idf)
    add_subfloor_cracks_minimal(idf, zone_name="Subfloor")
    add_external_wall_leakage(idf, crack_params=crack_params)
    add_internal_openings(idf, building_config.openings)
    patch_outdoor_afn_surfaces(idf)
    return idf
