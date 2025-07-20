import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import random
import os
from .performance_metrics import calculate_performance_metrics


def monte_carlo_simulation(
    trades: List[Dict[str, Any]],
    initial_balance: float,
    simulations: int = 1000,
    confidence_interval: float = 0.95,
) -> Dict[str, Any]:
    """
    Perform Monte Carlo simulation by randomizing the order of trades
    to test the robustness of a trading strategy.

    Args:
        trades: List of trade dictionaries containing trade details
        initial_balance: Starting account balance
        simulations: Number of Monte Carlo simulations to run
        confidence_interval: Confidence interval for the results

    Returns:
        Dict containing simulation results
    """
    if not trades:
        return {
            "mean_final_equity": initial_balance,
            "median_final_equity": initial_balance,
            "min_final_equity": initial_balance,
            "max_final_equity": initial_balance,
            "confidence_interval_lower": initial_balance,
            "confidence_interval_upper": initial_balance,
            "mean_max_drawdown": 0,
            "median_max_drawdown": 0,
            "worst_drawdown": 0,
            "mean_win_rate": 0,
            "mean_profit_factor": 0,
            "profitable_simulations_pct": 0,
        }

    # Extract profits from trades - handle both 'profit' and 'profit_loss' field names
    profits = [
        t.get("profit", t.get("profit_loss", 0))
        for t in trades
        if ("profit" in t or "profit_loss" in t)
    ]

    # Handle case with no profits data
    if not profits:
        return {
            "mean_final_equity": initial_balance,
            "median_final_equity": initial_balance,
            "min_final_equity": initial_balance,
            "max_final_equity": initial_balance,
            "confidence_interval_lower": initial_balance,
            "confidence_interval_upper": initial_balance,
            "mean_max_drawdown": 0,
            "median_max_drawdown": 0,
            "worst_drawdown": 0,
            "mean_win_rate": 0,
            "mean_profit_factor": 0,
            "profitable_simulations_pct": 0,
        }

    # Store simulation results
    final_equities = []
    max_drawdowns = []
    win_rates = []
    profit_factors = []

    for _ in range(simulations):
        # Shuffle the trades to simulate different market conditions
        shuffled_profits = random.sample(profits, len(profits))

        # Recalculate equity curve with shuffled trades
        equity = initial_balance
        equity_curve = [equity]

        for profit in shuffled_profits:
            equity += profit
            equity_curve.append(equity)

        final_equities.append(equity_curve[-1])

        # Calculate drawdown for this simulation
        running_max = equity_curve[0]
        drawdowns = []

        for e in equity_curve:
            if e > running_max:
                running_max = e
            drawdown = (running_max - e) / running_max * 100 if running_max > 0 else 0
            drawdowns.append(drawdown)

        max_drawdowns.append(max(drawdowns) if drawdowns else 0)

        # Calculate win rate
        winning_trades = len([p for p in shuffled_profits if p > 0])
        win_rate = winning_trades / len(shuffled_profits) if shuffled_profits else 0
        win_rates.append(win_rate)

        # Calculate profit factor
        gains = sum([p for p in shuffled_profits if p > 0])
        losses = sum([p for p in shuffled_profits if p < 0])
        profit_factor = gains / abs(losses) if losses < 0 else float("inf")
        profit_factors.append(profit_factor)

    # Sort final equities to calculate percentiles
    final_equities.sort()

    # Calculate confidence intervals
    lower_percentile = (1 - confidence_interval) / 2
    upper_percentile = 1 - lower_percentile

    lower_idx = int(lower_percentile * simulations)
    upper_idx = int(upper_percentile * simulations)

    # Ensure indices are within bounds
    lower_idx = max(0, lower_idx)
    upper_idx = min(simulations - 1, upper_idx)

    # Calculate percentage of simulations that were profitable
    profitable_sims = len([e for e in final_equities if e > initial_balance])
    profitable_pct = profitable_sims / simulations * 100

    results = {
        "mean_final_equity": np.mean(final_equities),
        "median_final_equity": np.median(final_equities),
        "min_final_equity": min(final_equities),
        "max_final_equity": max(final_equities),
        "confidence_interval_lower": final_equities[lower_idx],
        "confidence_interval_upper": final_equities[upper_idx],
        "mean_max_drawdown": np.mean(max_drawdowns),
        "median_max_drawdown": np.median(max_drawdowns),
        "worst_drawdown": max(max_drawdowns),
        "mean_win_rate": np.mean(win_rates),
        "mean_profit_factor": np.mean(profit_factors),
        "profitable_simulations_pct": profitable_pct,
    }

    return results


def plot_monte_carlo_results(
    simulation_results: Dict[str, Any],
    trades: List[Dict[str, Any]],
    initial_balance: float,
    simulations: int = 100,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Plot Monte Carlo simulation results with confidence intervals.

    Args:
        simulation_results: Results from monte_carlo_simulation function
        trades: List of trade dictionaries
        initial_balance: Starting account balance
        simulations: Number of simulation paths to display
        save_path: Path to save the chart image
        show_plot: Whether to display the chart
    """
    # Extract profits from trades - handle both 'profit' and 'profit_loss' field names
    profits = [
        t.get("profit", t.get("profit_loss", 0))
        for t in trades
        if ("profit" in t or "profit_loss" in t)
    ]

    if not profits:
        print("No profit data available for Monte Carlo visualization")
        return

    # Generate simulations for visualization
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot original equity curve
    equity = initial_balance
    original_equity = [equity]
    for profit in profits:
        equity += profit
        original_equity.append(equity)

    ax.plot(original_equity, color="blue", linewidth=2, label="Original Equity Curve")

    # Plot Monte Carlo simulations
    for _ in range(min(simulations, 100)):  # Limit to 100 displayed paths
        # Shuffle the trades
        shuffled_profits = random.sample(profits, len(profits))

        # Calculate equity curve
        equity = initial_balance
        equity_curve = [equity]
        for profit in shuffled_profits:
            equity += profit
            equity_curve.append(equity)

        # Plot this simulation with low opacity
        ax.plot(equity_curve, color="gray", alpha=0.1)

    # Add confidence intervals
    x = range(len(profits) + 1)
    lower_bound = [initial_balance]
    upper_bound = [initial_balance]

    # Calculate confidence interval at each trade
    for i in range(1, len(profits) + 1):
        sim_results = []
        for _ in range(100):  # Use 100 simulations for each point
            sample = random.sample(profits, i)
            sim_equity = initial_balance + sum(sample)
            sim_results.append(sim_equity)

        sim_results.sort()
        lower_idx = int(0.05 * len(sim_results))  # 5th percentile
        upper_idx = int(0.95 * len(sim_results))  # 95th percentile

        lower_bound.append(sim_results[lower_idx])
        upper_bound.append(sim_results[upper_idx])

    # Plot confidence interval
    ax.fill_between(
        x,
        lower_bound,
        upper_bound,
        color="green",
        alpha=0.2,
        label="95% Confidence Interval",
    )

    # Add final equity information
    ax.axhline(
        y=simulation_results["mean_final_equity"],
        color="red",
        linestyle="--",
        label=f"Mean Final: ${simulation_results['mean_final_equity']:.2f}",
    )

    # Annotate key metrics
    plt.annotate(
        f"Mean Final: ${simulation_results['mean_final_equity']:.2f}\n"
        + f"95% CI: [${simulation_results['confidence_interval_lower']:.2f}, "
        + f"${simulation_results['confidence_interval_upper']:.2f}]\n"
        + f"Mean Max DD: {simulation_results['mean_max_drawdown']:.2f}%\n"
        + f"Profitable: {simulation_results['profitable_simulations_pct']:.1f}%",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
    )

    ax.set_title(f"Monte Carlo Simulation - {len(profits)} Trades")
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Account Balance")
    ax.legend(loc="upper left")
    ax.grid(True)

    # Save if path provided
    if save_path:
        plt.savefig(save_path)

    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def run_monte_carlo_analysis(
    trades: List[Dict[str, Any]],
    initial_balance: float,
    final_balance: float,
    simulations: int = 1000,
    output_dir: str = "./reports",
    report_name: str = "monte_carlo_analysis",
) -> Dict[str, Any]:
    """
    Run a complete Monte Carlo analysis and generate reports.

    Args:
        trades: List of trade dictionaries
        initial_balance: Starting account balance
        final_balance: Final account balance
        simulations: Number of Monte Carlo simulations
        output_dir: Directory to save the report
        report_name: Base name for the report files

    Returns:
        Dict containing simulation results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run Monte Carlo simulation
    results = monte_carlo_simulation(trades, initial_balance, simulations)

    # Generate Monte Carlo plot
    chart_path = os.path.join(output_dir, f"{report_name}_plot.png")
    plot_monte_carlo_results(results, trades, initial_balance, 100, chart_path, False)

    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_csv_path = os.path.join(output_dir, f"{report_name}_results.csv")
    results_df.to_csv(results_csv_path, index=False)

    return results
