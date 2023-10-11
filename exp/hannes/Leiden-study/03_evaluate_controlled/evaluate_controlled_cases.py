"""assemble results for controlled cases"""
import pandas as pd
import numpy as np
import sys

# gas_column = 'Site:Environmental Impact NaturalGas Source Energy [J](Hourly)'
elec_column = "Facility Net Purchased Electricity Rate(Whole Building)"
co2_column = "Environmental Impact Total CO2 Emissions Carbon Equivalent Mass(Site)"

installation_costs = [
    0,
    3800,
    16000,
    51000,
    71000,
    23000,
    26800,
    39000,
    74000,
    94000,
    33000,
    36800,
    49000,
    84000,
    104000,
]
# cases=15
i_case = int(sys.argv[1])
reps = 1
years = np.arange(2017, 2023)
dir_name = "Eplus-env-evaluate_SAC"

gas_price_per_kwh = 0.1
elec_price_per_kwh = 0.35
elec_export_price_per_kwh = 0.15
co2_to_gas = 52 * 1e-6 / 0.277778

results = []
for y in years:
    for r in range(reps):
        data = pd.read_csv(
            dir_name
            + "_case_"
            + str(i_case)
            + "_year_"
            + str(y)
            + "_rep_"
            + str(r)
            + "-v1-res1/"
            + "Eplus-env-sub_run1/monitor.csv",
            skiprows=lambda x: x == 1,
        )
        co2_sum = data[co2_column].sum() / 1000
        elec_sum = data[elec_column].sum() / 1000

        if i_case > 4:
            gas_sum = 0
        else:
            gas_sum = co2_sum / co2_to_gas

        if elec_sum > 0:
            bill_sum = gas_sum * gas_price_per_kwh + elec_sum * elec_price_per_kwh
        else:
            bill_sum = (
                gas_sum * gas_price_per_kwh + elec_sum * elec_export_price_per_kwh
            )

        row_dict = {
            "case": i_case,
            "year": y,
            "rep": r,
            "gas [kWh]": gas_sum,
            "electricity [kWh]": elec_sum,
            "bills [GBP]": bill_sum,
            "CO2 [t]": co2_sum,
            "costs [GBP]": installation_costs[i_case],
        }

        results.append(row_dict)

results_df = pd.DataFrame(results)
results_df.to_csv("results_controlled_case_" + str(i_case) + ".csv")
