# add_blinds.py
from pathlib import Path
from eppy.modeleditor import IDF

def add_always_closed_blinds(idf, zone_name, blind_name="always_closed_blind_mat"):
    """
    Adds:
      - a WINDOWMATERIAL:SHADE (if missing)
      - an AlwaysOnSchedule (if missing)
      - a WINDOWSHADINGCONTROL attaching this shade
      - applied to every window in the specified zone
    """

    # --- Ensure shade material exists ---
    shade_exists = any(
        m.Name.lower() == blind_name.lower()
        for m in idf.idfobjects["WINDOWMATERIAL:SHADE"]
    )
    if not shade_exists:
        idf.newidfobject(
            "WINDOWMATERIAL:SHADE",
            Name=blind_name,
            Solar_Transmittance=0.20,
            Solar_Reflectance=0.20,
            Visible_Transmittance=0.05,
            Visible_Reflectance=0.30,
            Infrared_Hemispherical_Emissivity=0.90,
            Infrared_Transmittance=0.0,
            Thickness=0.9,    # 9 mm
            Conductivity=0.10,
            Shade_to_Glass_Distance=0.10,
            Top_Opening_Multiplier=0.0,
            Bottom_Opening_Multiplier=0.0
        )

    # --- Find all surfaces in the target zone ---
    zone_surfaces = [
        s for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]
        if s.Zone_Name and s.Zone_Name.lower() == zone_name.lower()
    ]
    parent_names = {s.Name for s in zone_surfaces}

    # --- Find windows attached to those surfaces ---
    windows = [
        f for f in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]
        if f.Surface_Type.lower() == "window"
        and f.Building_Surface_Name in parent_names
    ]

    if not windows:
        print(f"⚠️ No windows found in zone '{zone_name}'. No blinds added.")
        return

    # --- Ensure AlwaysOnSchedule exists ---
    always_on_exists = any(
        s.Name.lower() == "alwaysonschedule"
        for s in idf.idfobjects["SCHEDULE:CONSTANT"]
    )
    if not always_on_exists:
        idf.newidfobject(
            "SCHEDULE:CONSTANT",
            Name="AlwaysOnSchedule",
            Schedule_Type_Limits_Name="OnOff",
            Hourly_Value=1
        )

    # --- Create the shading control object if missing ---
    ctrl_name = f"{zone_name}_blinds"
    ctrl_exists = any(
        sc.Name.lower() == ctrl_name.lower()
        for sc in idf.idfobjects["WINDOWSHADINGCONTROL"]
    )

    if not ctrl_exists:
        sc = idf.newidfobject(
            "WINDOWSHADINGCONTROL",
            Name=ctrl_name,
            Zone_Name=zone_name,
            Shading_Control_Sequence_Number=1,
            Shading_Type="InteriorShade",
            Shading_Control_Type="AlwaysOn",
            Schedule_Name="AlwaysOnSchedule",
            Shading_Control_Is_Scheduled="Yes",
            Glare_Control_Is_Active="No",
            Shading_Device_Material_Name=blind_name,
            Multiple_Surface_Control_Type="Group"
        )
        for i, win in enumerate(windows, 1):
            setattr(sc, f"Fenestration_Surface_{i}_Name", win.Name)

    print(f"✅ Added blinds to {len(windows)} window(s) in zone '{zone_name}'.")


# ----------------------------------------------------
# Optional CLI
# ----------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Add always-closed blinds to a zone.")
    parser.add_argument("--idd", required=True, help="Path to Energy+.idd")
    parser.add_argument("--idf_in", required=True, help="Input IDF")
    parser.add_argument("--idf_out", required=True, help="Output IDF")
    parser.add_argument("--zone", required=True, help="Zone to apply blinds to")

    args = parser.parse_args()

    IDF.setiddname(args.idd)
    idf = IDF(args.idf_in)

    add_always_closed_blinds(idf, args.zone)
    idf.saveas(args.idf_out)

    print(f"💾 Saved with blinds: {args.idf_out}")


if __name__ == "__main__":
    main()
