{
"variant":"standard",
"id":"39502"
}
import pandas as pd
from analysis.shared import (
    load_idf, load_run_csv, get_zone_outdoor_links,
    compute_afn_infiltration_exfiltration,
    zone_volumes, measured_inf, measured_exf
)
from analysis.coheat.energy_use import compute_daily_heating_kwh
from analysis.coheat.plotting import compare_plot, plot_daily_heating_energy


def run_coheat_analysis(idf_path, csv_path, idd_path):
    idf = load_idf(idf_path, idd_path)
    df = load_run_csv(csv_path)
    zone_links = get_zone_outdoor_links(idf)
    ach = compute_afn_infiltration_exfiltration(
        df, zone_links, zone_volumes, measured_inf, measured_exf
    )
    daily_kwh = compute_daily_heating_kwh(df)
    return df, ach, daily_kwh

def plot_infiltration_results(ach):
    compare_plot(ach["AFN_infil"], measured_inf, "Infiltration: Measured vs Modelled")
    compare_plot(ach["AFN_exfil"], measured_exf, "Exfiltration: Measured vs Modelled")
