{
"variant":"standard",
"id":"82934"
}
import argparse
from analysis.coheat.analysis_pipeline import run_coheat_analysis, plot_infiltration_results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idf", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--idd", required=True)
    args = p.parse_args()

    df, ach, daily_kwh = run_coheat_analysis(args.idf, args.csv, args.idd)
    print("\n--- Infiltration / Exfiltration Table ---")
    print(ach)
    plot_infiltration_results(ach)

if __name__ == "__main__":
    main()
