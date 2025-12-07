from pathlib import Path
from eppy.modeleditor import IDF

# ----------------------------------------------------
#  CLEAN EXPERIMENT OBJECTS
# ----------------------------------------------------

def clean_experiment_objects(idf):
    """
    Remove all objects that vary between experiments:
      - thermostat setpoints
      - injected schedules
      - boiler efficiency curves
      - baseboards
      - other experiment-dependent controls

    Leaves intact:
      - geometry
      - AFN structure
      - materials & constructions
      - HVAC plant nodes, branches, equipment
      - zone definitions
    """

    # Classes we explicitly allow removal of
    removable_classes = [
        "SITE:LOCATION",
        "SIMULATIONCONTROL",
        "ZONEVENTILATION:DESIGNFLOWRATE",
        # thermostat controls
        "HVACTEMPLATE:THERMOSTAT",

        # schedules varied by experiments/modes
        "SCHEDULE:FILE",
        "SCHEDULE:COMPACT",

        # boiler curves
        "BOILER:HOTWATER",
        "CURVE:LINEAR",
        "CURVE:QUADRATIC",
        "CURVE:CUBIC",
        "CURVE:BIQUADRATIC",

        # baseboards (you always re-inject h28_baseboard.idf)
        "ZONEHVAC:BASEBOARD:RADIANTCONVECTIVE:WATER",

        # experiment-specific branches (e.g. plant modifications)
        # "BRANCH",
    ]

    for cls in removable_classes:
        if cls in idf.idfobjects:
            for obj in list(idf.idfobjects[cls]):
                idf.removeidfobject(obj)

    return idf


# ----------------------------------------------------
#  OPTIONAL: CLI USAGE (for debugging)
#  e.g.  python clean_experiment_objects.py file.idf
# ----------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remove experiment-dependent objects from an IDF.")
    parser.add_argument("idf_path", help="Path to IDF file to clean")
    parser.add_argument("--save_as", default=None, help="Save cleaned file as new IDF")

    args = parser.parse_args()

    IDF.setiddname("/usr/local/EnergyPlus-9-5-0/Energy+.idd")
    idf = IDF(args.idf_path)

    clean_experiment_objects(idf)

    out = args.save_as or args.idf_path.replace(".idf", "_cleaned.idf")
    idf.saveas(out)
    print(f"Cleaned IDF saved to: {out}")
