def get_zone_outdoor_links(idf):
    mapping = {}
    outdoor_nodes = {n.Name.lower() for n in idf.idfobjects["AIRFLOWNETWORK:MULTIZONE:EXTERNALNODE"]}
    build_surfs = {s.Name.lower(): s for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]}
    fen_surfs = {s.Name.lower(): s for s in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]}
    all_surfs = {**build_surfs, **fen_surfs}
    for afn_surf in idf.idfobjects["AIRFLOWNETWORK:MULTIZONE:SURFACE"]:
        surf_name = afn_surf.Surface_Name.lower()
        ext_node = getattr(afn_surf, "External_Node_Name", "").lower()
        s_obj = all_surfs.get(surf_name)
        zone_name, is_outdoor = None, False
        if s_obj:
            if s_obj.key.upper() == "FENESTRATIONSURFACE:DETAILED":
                parent = build_surfs.get(s_obj.Building_Surface_Name.lower())
                if parent:
                    zone_name = parent.Zone_Name.strip()
                    if getattr(parent, "Outside_Boundary_Condition", "").lower() == "outdoors":
                        is_outdoor = True
            else:
                zone_name = s_obj.Zone_Name.strip()
                if getattr(s_obj, "Outside_Boundary_Condition", "").lower() == "outdoors":
                    is_outdoor = True
        if not is_outdoor and ext_node in outdoor_nodes:
            is_outdoor = True
        if is_outdoor and zone_name:
            mapping.setdefault(zone_name.upper(), []).append(afn_surf.Surface_Name.strip())
    return mapping