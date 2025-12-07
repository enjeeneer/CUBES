# analysis/thermostat/run_analysis.py

import argparse
from pathlib import Path
from analysis.shared.load import load_all_run_results
from analysis.thermostat.comparison import compare_room_temperatures, compute_temperature_deltas, compare_gas_consumption
from analysis.thermostat.plotting import (
    plot_boiler_config_comparison,
    plot_temp_differences_with_beizaee,
    plot_absolute_temperatures_with_beizaee_period,
    plot_gas_consumption_by_config
)


def run_thermostat_analysis(results_dir, control_types=None, period='lynch', output_dir=None):
    """Main analysis pipeline for thermostat experiments.

    Performs comprehensive analysis:
    1. Loads results for all control types
    2. Compares temperatures across strategies
    3. Calculates temperature deltas vs conventional
    4. Compares gas consumption
    5. Generates comparative plots
    6. Prints summary statistics

    Args:
        results_dir: Path to experiment results directory
        control_types: List of control strategies to analyze
        period: Analysis period (test/beizaee/lynch)
        output_dir: Optional directory to save plots
    """
    if control_types is None:
        control_types = ['conventional_control', 'zonal_control', 'occupancy_control']

    print(f"Loading thermostat experiment results from {results_dir}")
    print(f"Period: {period}")
    print(f"Control types: {', '.join(control_types)}\n")

    # ========================================
    # 1. Load Results
    # ========================================
    print("=" * 60)
    print("LOADING SIMULATION RESULTS")
    print("=" * 60)

    results = load_all_run_results(
        base_path=results_dir,
        experiment_path=f"runs_validation/{period}",
        run=None
    )

    # Filter to requested control types
    results = {k: v for k, v in results.items() if any(ct in k for ct in control_types)}

    if not results:
        print("ERROR: No results found for specified control types")
        return

    print(f"Loaded {len(results)} control configurations")
    for control_type in results.keys():
        print(f"  - {control_type}: {len(results[control_type])} timesteps")

    # ========================================
    # 2. Temperature Comparison
    # ========================================
    print("\n" + "=" * 60)
    print("TEMPERATURE COMPARISON")
    print("=" * 60)

    temp_comparison = compare_room_temperatures(results)
    print("\nMean Temperatures by Room (°C):")
    print(temp_comparison.round(2))

    # ========================================
    # 3. Temperature Deltas
    # ========================================
    print("\n" + "=" * 60)
    print("TEMPERATURE DELTAS (Conventional - Smart)")
    print("=" * 60)

    if 'conventional_control' in results:
        for control_type in results.keys():
            if control_type != 'conventional_control' and 'control' in control_type:
                deltas = compute_temperature_deltas(
                    results['conventional_control'],
                    results[control_type]
                )
                print(f"\nConventional - {control_type}:")
                print(deltas.round(3))

                # Plot deltas
                if output_dir:
                    fig, ax = plot_boiler_config_comparison(deltas)
                    fig.savefig(Path(output_dir) / f"deltas_{control_type}.png", dpi=300, bbox_inches='tight')
                    print(f"  → Saved plot to {output_dir}/deltas_{control_type}.png")

    # ========================================
    # 4. Gas Consumption Comparison
    # ========================================
    print("\n" + "=" * 60)
    print("GAS CONSUMPTION COMPARISON")
    print("=" * 60)

    gas_summary = compare_gas_consumption(results)
    print("\nGas Consumption Summary:")
    print(gas_summary.round(2))

    # Plot gas consumption
    if output_dir:
        fig, ax1, ax2 = plot_gas_consumption_by_config(gas_summary)
        fig.savefig(Path(output_dir) / "gas_consumption.png", dpi=300, bbox_inches='tight')
        print(f"  → Saved plot to {output_dir}/gas_consumption.png")

    # ========================================
    # 5. Summary Statistics
    # ========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nKey Findings:")
    if 'conventional_control' in gas_summary.index:
        conv_gas = gas_summary.loc['conventional_control', 'Daily_kWh']
        print(f"  Conventional control: {conv_gas:.2f} kWh/day")

        for control_type in gas_summary.index:
            if control_type != 'conventional_control':
                savings = gas_summary.loc[control_type, 'Savings_pct']
                daily = gas_summary.loc[control_type, 'Daily_kWh']
                print(f"  {control_type}: {daily:.2f} kWh/day ({savings:+.1f}% vs conventional)")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze thermostat experiment results"
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to experiment results directory"
    )
    parser.add_argument(
        "--controls",
        nargs="+",
        default=["conventional_control", "zonal_control", "occupancy_control"],
        help="Control types to analyze"
    )
    parser.add_argument(
        "--period",
        choices=["test", "beizaee", "lynch"],
        default="lynch",
        help="Analysis period"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Directory to save plots (optional)"
    )

    args = parser.parse_args()

    # Create output directory if specified
    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)

    run_thermostat_analysis(
        results_dir=args.results,
        control_types=args.controls,
        period=args.period,
        output_dir=args.output
    )
