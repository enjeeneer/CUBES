import matplotlib.pyplot as plt
import numpy as np

def compare_plot(modelled, measured, title):
    zones = list(modelled.index)   # keep ACH order exactly as provided
    fig, ax = plt.subplots(figsize=(10,5))

    model_vals = modelled.values
    meas_vals = [measured.get(z, np.nan) for z in zones]

    ax.plot(zones, model_vals, 'o-', label="Modelled")
    ax.plot(zones, meas_vals, 'ks--', label="Measured")

    for i, (z, mv, ev) in enumerate(zip(zones, model_vals, meas_vals)):
        if not np.isnan(mv):
            ax.text(i, mv + 0.1, f"{mv:.2f}", ha='center', va='bottom', fontsize=8)
        if not np.isnan(ev):
            ax.text(i, ev - 0.1, f"{ev:.2f}", ha='center', va='top', fontsize=8)

    ax.set(title=title, ylabel="ACH (1/h)")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_daily_heating_energy(daily_kwh):
    plt.figure(figsize=(10,5))
    plt.plot(daily_kwh.index, daily_kwh.values, linewidth=1.8)

    plt.title("Daily Total Energy Consumption (Building)")
    plt.ylabel("Energy [kWh]")
    plt.xlabel("Date")

    plt.ylim([0, max(daily_kwh)*1.1])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

