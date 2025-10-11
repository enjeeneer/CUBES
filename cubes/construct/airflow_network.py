"""
airflow_network.py

Airflow Network builder with crack templates:
- Zone & element-specific crack templates
- External/internal surface cracks (per m²) from crack_templates.
- Fenestration frame cracks (per metre perimeter) from crack_templates.
- Internal openings (Simple/Horizontal) with fixed discharge coefficient 0.65.
- Subfloor vents with always-open SimpleOpenings.
- Audits to ensure each zone has ≥2 AFN surfaces.

Requires: eppy.modeleditor.IDF
"""

from collections import defaultdict
from typing import Optional, Dict, Union, List, Tuple
from eppy.modeleditor import IDF
import math

# --------------------------
# Utilities
# --------------------------

def perimeter(verts):
    if len(verts) < 2: return 0.0
    per = 0.0
    for i in range(len(verts)):
        x1,y1,z1 = verts[i]; x2,y2,z2 = verts[(i+1)%len(verts)]
        dx,dy,dz = x2-x1,y2-y1,z2-z1
        per += math.sqrt(dx*dx+dy*dy+dz*dz)
    return per

def get_vertices(fen):
    vs=[]
    for i in range(1,501):
        x=getattr(fen,f"Vertex_{i}_Xcoordinate","")
        if x=="": break
        y=getattr(fen,f"Vertex_{i}_Ycoordinate")
        z=getattr(fen,f"Vertex_{i}_Zcoordinate")
        vs.append((float(x),float(y),float(z)))
    return vs

def _get_vertices(s):
    verts=[]
    for i in range(1,501):
        try: x=getattr(s,f"Vertex_{i}_Xcoordinate")
        except Exception: break
        if x=="": break
        y=getattr(s,f"Vertex_{i}_Ycoordinate")
        z=getattr(s,f"Vertex_{i}_Zcoordinate")
        verts.append((float(x),float(y),float(z)))
    return verts

def _polygon_area_3d(verts):
    if len(verts)<3: return 0.0
    nx=ny=nz=0.0
    for (x1,y1,z1),(x2,y2,z2) in zip(verts,verts[1:]+verts[:1]):
        nx+=(y1-y2)*(z1+z2); ny+=(z1-z2)*(x1+x2); nz+=(x1-x2)*(y1+y2)
    return 0.5*math.sqrt(nx*nx+ny*ny+nz*nz)

def _approx(a,b,tol=1e-3): return abs(float(a)-float(b))<=tol
def _get_azimuth(surface): return float(getattr(surface,"Azimuth",getattr(surface,"azimuth",0.0)))
def surface_mid_height(s): zs=[v[2] for v in _get_vertices(s)]; return sum(zs)/len(zs) if zs else 1.5

# --------------------------
# AFN scaffolding
# --------------------------

def ensure_basic_schedules(idf:IDF):
    names={s.Name.lower() for s in idf.idfobjects.get("SCHEDULE:CONSTANT",[])}
    if "alwaysonschedule" not in names:
        idf.newidfobject("SCHEDULE:CONSTANT",Name="AlwaysOnSchedule",Schedule_Type_Limits_Name="OnOff",Hourly_Value=1)
    if "alwaysoffschedule" not in names:
        idf.newidfobject("SCHEDULE:CONSTANT",Name="AlwaysOffSchedule",Schedule_Type_Limits_Name="OnOff",Hourly_Value=0)

def ensure_reference_crack_conditions(idf:IDF):
    if not idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:REFERENCECRACKCONDITIONS"):
        idf.newidfobject("AIRFLOWNETWORK:MULTIZONE:REFERENCECRACKCONDITIONS",
            Name="ReferenceCrackConditions",
            Reference_Temperature=20.0,
            Reference_Barometric_Pressure=101325,
            Reference_Humidity_Ratio=0.0
        )

def setup_afn_controls(idf:IDF,cp_array_name="NormalExposureCpArray"):
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
            Height_Dependence_of_External_Node_Temperature="No"
        )
    if not idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTARRAY"):
        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTARRAY",
            Name=cp_array_name,
            Wind_Direction_1=0,Wind_Direction_2=45,Wind_Direction_3=90,Wind_Direction_4=135,
            Wind_Direction_5=180,Wind_Direction_6=225,Wind_Direction_7=270,Wind_Direction_8=315
        )

def ensure_afn_zones(idf:IDF):
    existing={z.Zone_Name for z in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:ZONE",[])}
    for z in idf.idfobjects["ZONE"]:
        if z.Name in existing: continue
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
            Facade_Width=0.0
        )

# --------------------------
# Crack template resolver
# --------------------------

def get_crack_params(zone:str,element_key:str,templates:Dict)->Optional[Dict[str,Union[float,int]]]:
    zone=zone.lower()
    if zone in templates and element_key in templates[zone]:
        return templates[zone][element_key]
    return templates.get("default",{}).get(element_key)

# --------------------------
# Surface cracks
# --------------------------

def add_surface_leakage(idf:IDF,building_config,cp_array_name="NormalExposureCpArray"):
    cp_vals=[0.4,0.1,-0.3,-0.35,-0.2,-0.35,-0.3,-0.1]
    templates=getattr(building_config,"crack_templates",{})
    made=0

    for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
        zone=s.Zone_Name
        stype=s.Surface_Type.lower()
        bc=s.Outside_Boundary_Condition.lower()

        if bc=="outdoors":
            if stype=="wall": key="external_wall"
            elif stype=="roof": key="roof"
            elif stype in("floor","ceiling"): key="external_floor"
            else: continue
        elif bc in("adiabatic","ground"):
            continue
        else:
            if stype=="wall": key="internal_wall"
            elif stype in("floor","ceiling"): key="internal_floor"
            else: continue

        crack=get_crack_params(zone,key,templates)
        if not crack: continue
        coef=crack.get("cq_per_m2",0)*s.area
        expn=crack.get("n",0.65)
        crack_name=f"{s.Name}_Crack"

        if not any(c.Name==crack_name for c in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:SURFACE:CRACK",[])):
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:SURFACE:CRACK",
                Name=crack_name,
                Air_Mass_Flow_Coefficient_at_Reference_Conditions=coef,
                Air_Mass_Flow_Exponent=expn,
                Reference_Crack_Conditions="ReferenceCrackConditions"
            )

        ext_node=""
        if bc=="outdoors":
            if not any(v.Name==s.Name for v in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTVALUES",[])):
                idf.newidfobject(
                    "AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTVALUES",
                    Name=s.Name,
                    AirflowNetworkMultiZoneWindPressureCoefficientArray_Name=cp_array_name,
                    **{f"Wind_Pressure_Coefficient_Value_{i+1}": v for i,v in enumerate(cp_vals)}
                )
            if not any(n.Name==s.Name for n in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE",[])):
                idf.newidfobject(
                    "AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE",
                    Name=s.Name,
                    External_Node_Height=surface_mid_height(s),
                    Wind_Pressure_Coefficient_Curve_Name=s.Name
                )
            ext_node=s.Name

        if not any(a.Surface_Name==s.Name for a in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:SURFACE",[])):
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:SURFACE",
                Surface_Name=s.Name,
                Leakage_Component_Name=crack_name,
                External_Node_Name=ext_node,
                Ventilation_Control_Mode="ZoneLevel"
            )
            made+=1
    return made

# --------------------------
# Internal openings
# --------------------------

def resolve_opening_schedule(opening):
    val=opening.get("schedule","1"); s=str(val).lower()
    if s=="1": return "AlwaysOnSchedule"
    if s=="0": return "AlwaysOffSchedule"
    return f"door-schedule-{s}"

def add_internal_openings(idf:IDF,openings_cfg):
    made=set();count=0
    for op in openings_cfg or []:
        z1,z2=op["zones"]; comp_name=f"{z1}_{z2}_Opening"; sched=resolve_opening_schedule(op)
        if comp_name not in made:
            if str(op.get("orientation","vertical")).lower()=="horizontal":
                idf.newidfobject(
                    "AIRFLOWNETWORK:MULTIZONE:COMPONENT:HORIZONTALOPENING",
                    Name=comp_name,
                    Air_Mass_Flow_Coefficient_When_Opening_is_Closed=0.001,
                    Air_Mass_Flow_Exponent_When_Opening_is_Closed=0.65,
                    Sloping_Plane_Angle=90.0,
                    Discharge_Coefficient=0.65
                )
            else:
                idf.newidfobject(
                    "AIRFLOWNETWORK:MULTIZONE:COMPONENT:SIMPLEOPENING",
                    Name=comp_name,
                    Air_Mass_Flow_Coefficient_When_Opening_is_Closed=0.001,
                    Air_Mass_Flow_Exponent_When_Opening_is_Closed=0.65,
                    Minimum_Density_Difference_for_TwoWay_Flow=0.0001,
                    Discharge_Coefficient=0.65
                )
            made.add(comp_name)

        match=None
        for fen in idf.idfobjects.get("FENESTRATIONSURFACE:DETAILED",[]):
            bs=fen.Building_Surface_Name.lower(); ob=fen.Outside_Boundary_Condition_Object.lower()
            if z1.lower() in bs and z2.lower() in ob: match=fen; break
            if z2.lower() in bs and z1.lower() in ob: match=fen; break
        if not match: continue

        idf.newidfobject(
            "AIRFLOWNETWORK:MULTIZONE:SURFACE",
            Surface_Name=match.Name,
            Leakage_Component_Name=comp_name,
            External_Node_Name="",
            Ventilation_Control_Mode="Constant",
            Venting_Availability_Schedule_Name=sched
        )
        count+=1
    return count

# --------------------------
# Fenestration frame cracks
# --------------------------

def add_fenestration_frame_cracks(idf:IDF,building_config,cp_array_name="NormalExposureCpArray"):
    bsurfs={s.Name:s for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]}
    wpc_vals={v.Name for v in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTVALUES",[])}
    ext_nodes={n.Name for n in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE",[])}
    cp_vals=[0.4,0.1,-0.3,-0.35,-0.2,-0.35,-0.3,-0.1]
    templates=getattr(building_config,"crack_templates",{})
    added=0

    for fen in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
        st=fen.Surface_Type.lower()
        if st not in ("window","door"): continue
        parent=bsurfs.get(fen.Building_Surface_Name)
        if not parent: continue
        bc=parent.Outside_Boundary_Condition.lower()
        is_external=bc=="outdoors"
        zone=parent.Zone_Name

        if st=="window" and is_external: key="external_window"
        elif st=="door" and is_external and "vent" not in fen.Name.lower(): key="external_door"
        elif st=="door" and not is_external: key="internal_door"
        else: key="external_vent"

        crack=get_crack_params(zone,key,templates)
        if not crack: continue
        cq=crack.get("cq_per_m",0); n=crack.get("n",0.65)
        verts=get_vertices(fen); perim=perimeter(verts); coef=cq*perim
        if coef<=0: continue

        ext_node=""
        if is_external:
            if parent.Name not in wpc_vals:
                idf.newidfobject(
                    "AIRFLOWNETWORK:MULTIZONE:WINDPRESSURECOEFFICIENTVALUES",
                    Name=parent.Name,
                    AirflowNetworkMultiZoneWindPressureCoefficientArray_Name=cp_array_name,
                    **{f"Wind_Pressure_Coefficient_Value_{i+1}":v for i,v in enumerate(cp_vals)}
                )
                wpc_vals.add(parent.Name)
            if parent.Name not in ext_nodes:
                idf.newidfobject(
                    "AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE",
                    Name=parent.Name,
                    External_Node_Height=surface_mid_height(parent),
                    Wind_Pressure_Coefficient_Curve_Name=parent.Name
                )
                ext_nodes.add(parent.Name)
            ext_node=parent.Name

        crack_name=f"{fen.Name}_FrameCrack"
        if not any(c.Name==crack_name for c in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:SURFACE:CRACK",[])):
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:SURFACE:CRACK",
                Name=crack_name,
                Air_Mass_Flow_Coefficient_at_Reference_Conditions=coef,
                Air_Mass_Flow_Exponent=n,
                Reference_Crack_Conditions="ReferenceCrackConditions"
            )

        if not any(a.Surface_Name==fen.Name for a in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:SURFACE",[])):
            idf.newidfobject(
                "AIRFLOWNETWORK:MULTIZONE:SURFACE",
                Surface_Name=fen.Name,
                Leakage_Component_Name=crack_name,
                External_Node_Name=ext_node,
                WindowDoor_Opening_Factor_or_Crack_Factor=1.0,
                Ventilation_Control_Mode="Constant",
                Venting_Availability_Schedule_Name="AlwaysOnSchedule"
            )
            added+=1
    return added

# --------------------------
# Validation / audits
# --------------------------

def validate_afn(idf:IDF)->List[str]:
    issues=[]
    ext_nodes={n.Name for n in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE",[])}
    for afn in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:SURFACE",[]):
        name=afn.Surface_Name
        geo=next((s for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"] if s.Name==name),None)
        if not geo:
            geo=next((f for f in idf.idfobjects.get("FENESTRATIONSURFACE:DETAILED",[]) if f.Name==name),None)
        if not geo:
            issues.append(f"Missing geometry for AFN surface: {name}")
            continue
        bc=geo.Outside_Boundary_Condition.lower() if hasattr(geo,"Outside_Boundary_Condition") else ""
        if bc=="outdoors" and not afn.External_Node_Name:
            issues.append(f"Outdoors surface needs external node: {name}")
        if bc=="outdoors" and afn.External_Node_Name and afn.External_Node_Name not in ext_nodes:
            issues.append(f"External node not defined: {afn.External_Node_Name}")
        if bc!="outdoors" and afn.External_Node_Name:
            issues.append(f"Internal surface must not set external node: {name}")
    return issues

def audit_afn_counts(idf:IDF):
    surf_by_name={s.Name:s for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]}
    counts=defaultdict(int)
    for afn in idf.idfobjects.get("AIRFLOWNETWORK:MULTIZONE:SURFACE",[]):
        parent=surf_by_name.get(afn.Surface_Name)
        if parent: counts[parent.Zone_Name]+=1
    print("AFN surfaces per zone:")
    for z in sorted(counts): print(f"  {z}: {counts[z]}")
    missing=[z.Name for z in idf.idfobjects["ZONE"] if counts.get(z.Name,0)<2]
    if missing: print("Zones with <2 AFN surfaces:",missing)
    return counts

# --------------------------
# Main entry
# --------------------------

def add_airflow_network(idf:IDF,building_config=None)->IDF:
    ensure_basic_schedules(idf)
    ensure_reference_crack_conditions(idf)
    setup_afn_controls(idf)
    ensure_afn_zones(idf)
    add_surface_leakage(idf,building_config)
    add_subfloor_air_bricks(idf,per_wall=2,vent_area=0.01,exclude_azimuth=90.0)
    add_internal_openings(idf,getattr(building_config,"openings",None))
    n_added=add_fenestration_frame_cracks(idf,building_config)
    print("Added AFN fenestration cracks:",n_added)
    patch_outdoor_afn_surfaces(idf)
    audit_afn_counts(idf)
    ensure_min_two_afn_per_zone(idf)
    return idf
