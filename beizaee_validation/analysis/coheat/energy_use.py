import pandas as pd

MJ_TO_KWH = 0.277778
J_TO_MJ = 1e-6

def compute_daily_heating_kwh(df):
    ideal = df.filter(like="IDEALLOADS:Zone Ideal Loads Supply Air Total Heating Energy")
    daily_kwh = ideal.resample("1D").sum().sum(axis=1) * J_TO_MJ * MJ_TO_KWH
    return daily_kwh
