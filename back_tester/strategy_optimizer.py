import numpy as np
import pandas as pd
import random
import itertools
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional, Callable
import os
import time
import concurrent.futures
from datetime import datetime

from .performance_metrics import calculate_performance_metrics


class StrategyOptimizer:
    """
    Optimizer for trading strategy parameters using various methods
    including grid search, random search, and adaptive methods.
    """

    def __init__(
        self,
        backtest_func: Callable,
        param_ranges: Dict[str, Any],
        optimization_target: str = "sharpe_ratio",
        maximize: bool = True,
    ):
        """
        Initialize strategy optimizer.

        Args:
            backtest_func: Function that performs backtesting with parameters
            param_ranges: Dictionary of parameter ranges to optimize
                For continuous parameters: (min_value, max_value)
                For discrete parameters: list of possible values
            optimization_target: Metric to optimize (e.g. "sharpe_ratio", "total_return")
            maximize: Whether to maximize or minimize the target metric
        """
        self.backtest_func = backtest_func
        self.param_ranges = param_ranges
        self.optimization_target = optimization_target
        self.maximize = maximize
        self.results = []

    def grid_search(self, max_combinations: int = 100) -> Dict[str, Any]:
        """
        Perform grid search over parameter space.

        Args:
            max_combinations: Maximum number of combinations to test

        Returns:
            Dictionary with best parameters and results
        """
        print(f"Starting grid search with max {max_combinations} combinations...")

        # Create parameter grid
        param_grid = {}

        for param_name, param_range in self.param_ranges.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                # For continuous parameters, create evenly spaced values
                min_val, max_val = param_range

                # Determine number of steps based on range
                if isinstance(min_val, int) and isinstance(max_val, int):
                    # For integers, step size is 1
                    param_grid[param_name] = list(range(min_val, max_val + 1))
                else:
                    # For floats, create a reasonable number of steps
                    num_steps = min(
                        int(max_combinations ** (1 / len(self.param_ranges))), 10
                    )
                    param_grid[param_name] = np.linspace(
                        min_val, max_val, num_steps
                    ).tolist()
            else:
                # For discrete parameters, use the provided list
                param_grid[param_name] = param_range

        # Calculate total combinations
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)

        print(f"Total parameter combinations: {total_combinations}")

        # If too many combinations, sample a subset
        if total_combinations > max_combinations:
            print(f"Reducing to {max_combinations} combinations...")
            return self.random_search(max_combinations)

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        combinations = list(itertools.product(*param_values))

        # Run backtests
        return self._run_parameter_combinations(param_names, combinations)

    def random_search(self, num_trials: int = 100) -> Dict[str, Any]:
        """
        Perform random search over parameter space.

        Args:
            num_trials: Number of random parameter combinations to try

        Returns:
            Dictionary with best parameters and results
        """
        print(f"Starting random search with {num_trials} trials...")

        # Generate random parameter combinations
        param_names = list(self.param_ranges.keys())
        combinations = []

        for _ in range(num_trials):
            combination = []
            for param_name in param_names:
                param_range = self.param_ranges[param_name]

                if isinstance(param_range, tuple) and len(param_range) == 2:
                    min_val, max_val = param_range

                    if isinstance(min_val, int) and isinstance(max_val, int):
                        # Integer parameter
                        value = random.randint(min_val, max_val)
                    else:
                        # Float parameter
                        value = min_val + random.random() * (max_val - min_val)
                else:
                    # Discrete parameter, select random value
                    value = random.choice(param_range)

                combination.append(value)

            combinations.append(combination)

        # Run backtests
        return self._run_parameter_combinations(param_names, combinations)

    def _run_parameter_combinations(
        self, param_names: List[str], combinations: List[tuple]
    ) -> Dict[str, Any]:
        """
        Run backtests for multiple parameter combinations.

        Args:
            param_names: List of parameter names
            combinations: List of parameter value combinations

        Returns:
            Dictionary with best parameters and results
        """
        results = []
        start_time = time.time()

        print(f"Running {len(combinations)} backtest combinations...")

        # Run backtests with different parameters
        for i, combination in enumerate(combinations):
            if i > 0 and i % 10 == 0:
                elapsed = time.time() - start_time
                remaining = (elapsed / i) * (len(combinations) - i)
                print(
                    f"Progress: {i}/{len(combinations)} combinations ({i/len(combinations)*100:.1f}%) - ETA: {remaining:.1f}s"
                )

            # Create parameter dictionary for this combination
            params = {param_names[j]: combination[j] for j in range(len(param_names))}

            # Run backtest with these parameters
            try:
                final_balance, trades, metrics = self.backtest_func(**params)

                # If metrics not provided by backtest function, calculate them
                if metrics is None and trades:
                    initial_balance = params.get("initial_balance", 10000)
                    metrics = calculate_performance_metrics(
                        trades, initial_balance, final_balance
                    )

                # Extract target metric
                target_value = (
                    metrics.get(self.optimization_target, 0) if metrics else 0
                )

                results.append(
                    {
                        "parameters": params,
                        "final_balance": final_balance,
                        "num_trades": len(trades) if trades else 0,
                        "metrics": metrics,
                        "target_value": target_value,
                    }
                )
            except Exception as e:
                print(f"Error with parameters {params}: {str(e)}")

        # Store all results
        self.results = results

        # Find best result
        if self.maximize:
            best_result = (
                max(results, key=lambda x: x["target_value"]) if results else None
            )
        else:
            best_result = (
                min(results, key=lambda x: x["target_value"]) if results else None
            )

        if best_result:
            print(
                f"\nOptimization complete. Best {self.optimization_target}: {best_result['target_value']}"
            )
            print(f"Best parameters: {best_result['parameters']}")
        else:
            print("Optimization failed - no valid results")

        return {
            "best_result": best_result,
            "all_results": results,
            "param_names": param_names,
        }

    def plot_parameter_impact(
        self,
        optimization_results: Dict[str, Any],
        top_n: int = 5,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> None:
        """
        Plot the impact of individual parameters on the optimization target.

        Args:
            optimization_results: Results from grid_search or random_search
            top_n: Number of top parameters to show
            save_path: Path to save the chart image
            show_plot: Whether to display the chart
        """
        results = optimization_results.get("all_results", [])
        if not results:
            print("No results to plot")
            return

        # Extract all parameters and values
        param_names = optimization_results.get("param_names", [])

        # Create figure with subplots for each parameter
        fig, axs = plt.subplots(len(param_names), 1, figsize=(12, 4 * len(param_names)))

        if len(param_names) == 1:
            axs = [axs]  # Make iterable if only one parameter

        for i, param_name in enumerate(param_names):
            # Extract parameter values and corresponding target values
            param_values = [r["parameters"].get(param_name, None) for r in results]
            target_values = [r["target_value"] for r in results]

            # Plot parameter vs target metric
            axs[i].scatter(param_values, target_values, alpha=0.6)
            axs[i].set_xlabel(param_name)
            axs[i].set_ylabel(self.optimization_target)
            axs[i].set_title(f"Impact of {param_name} on {self.optimization_target}")
            axs[i].grid(True, linestyle="--", alpha=0.7)

            # Add trend line if parameter is numeric
            if all(isinstance(v, (int, float)) for v in param_values):
                try:
                    # Filter out None values
                    valid_indices = [
                        j for j, v in enumerate(param_values) if v is not None
                    ]
                    valid_param_values = [param_values[j] for j in valid_indices]
                    valid_target_values = [target_values[j] for j in valid_indices]

                    if valid_param_values:
                        z = np.polyfit(valid_param_values, valid_target_values, 1)
                        p = np.poly1d(z)

                        # Add trend line
                        x_range = np.linspace(
                            min(valid_param_values), max(valid_param_values), 100
                        )
                        axs[i].plot(x_range, p(x_range), "r--")
                except:
                    # Skip trend line if it can't be calculated
                    pass

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path)

        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_parameter_heatmap(
        self,
        optimization_results: Dict[str, Any],
        param1: str,
        param2: str,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> None:
        """
        Plot heatmap showing the relationship between two parameters.

        Args:
            optimization_results: Results from grid_search or random_search
            param1: First parameter name
            param2: Second parameter name
            save_path: Path to save the chart image
            show_plot: Whether to display the chart
        """
        results = optimization_results.get("all_results", [])
        if not results:
            print("No results to plot")
            return

        # Extract parameter values
        param1_values = [r["parameters"].get(param1, None) for r in results]
        param2_values = [r["parameters"].get(param2, None) for r in results]
        target_values = [r["target_value"] for r in results]

        # Check if parameters are numeric
        if not all(isinstance(v, (int, float)) for v in param1_values + param2_values):
            print("Heatmap requires numeric parameters")
            return

        # Create heatmap data
        unique_param1 = sorted(set(param1_values))
        unique_param2 = sorted(set(param2_values))

        # Initialize heatmap with NaNs
        heatmap_data = np.full((len(unique_param2), len(unique_param1)), np.nan)

        # Fill in heatmap data
        for p1_val, p2_val, target_val in zip(
            param1_values, param2_values, target_values
        ):
            p1_idx = unique_param1.index(p1_val)
            p2_idx = unique_param2.index(p2_val)

            # If multiple values for same parameter combination, take the best one
            if np.isnan(heatmap_data[p2_idx, p1_idx]) or (
                self.maximize
                and target_val > heatmap_data[p2_idx, p1_idx]
                or not self.maximize
                and target_val < heatmap_data[p2_idx, p1_idx]
            ):
                heatmap_data[p2_idx, p1_idx] = target_val

        # Create figure
        plt.figure(figsize=(10, 8))

        # Plot heatmap
        heatmap = plt.imshow(
            heatmap_data, interpolation="nearest", aspect="auto", cmap="viridis"
        )
        plt.colorbar(heatmap, label=self.optimization_target)

        # Set labels
        plt.xlabel(param1)
        plt.ylabel(param2)
        plt.title(f"{self.optimization_target} Heatmap: {param1} vs {param2}")

        # Set tick labels
        plt.xticks(
            np.arange(len(unique_param1)),
            [f"{x:.2f}" if isinstance(x, float) else str(x) for x in unique_param1],
            rotation=45,
        )
        plt.yticks(
            np.arange(len(unique_param2)),
            [f"{y:.2f}" if isinstance(y, float) else str(y) for y in unique_param2],
        )

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path)

        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()


def optimize_strategy(
    backtest_func: Callable,
    param_ranges: Dict[str, Any],
    optimization_target: str = "sharpe_ratio",
    search_method: str = "random",
    num_trials: int = 100,
    output_dir: str = "./reports/optimization",
    maximize: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to run strategy optimization and generate reports.

    Args:
        backtest_func: Function that performs backtesting with parameters
        param_ranges: Dictionary of parameter ranges to optimize
        optimization_target: Metric to optimize
        search_method: "grid" or "random"
        num_trials: Number of trials for random search
        output_dir: Directory to save reports
        maximize: Whether to maximize or minimize the target metric

    Returns:
        Dictionary with optimization results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize optimizer
    optimizer = StrategyOptimizer(
        backtest_func=backtest_func,
        param_ranges=param_ranges,
        optimization_target=optimization_target,
        maximize=maximize,
    )

    # Run optimization
    if search_method.lower() == "grid":
        results = optimizer.grid_search(num_trials)
    else:
        results = optimizer.random_search(num_trials)

    # Generate impact plots
    impact_path = os.path.join(output_dir, "parameter_impact.png")
    optimizer.plot_parameter_impact(results, save_path=impact_path, show_plot=False)

    # Generate heatmaps for pairs of parameters
    param_names = list(param_ranges.keys())
    if len(param_names) >= 2:
        for i in range(len(param_names)):
            for j in range(i + 1, len(param_names)):
                param1 = param_names[i]
                param2 = param_names[j]

                # Skip if either parameter is not numeric
                param1_values = [
                    r["parameters"].get(param1) for r in results["all_results"]
                ]
                param2_values = [
                    r["parameters"].get(param2) for r in results["all_results"]
                ]

                if all(
                    isinstance(v, (int, float)) for v in param1_values + param2_values
                ):
                    heatmap_path = os.path.join(
                        output_dir, f"heatmap_{param1}_{param2}.png"
                    )
                    optimizer.plot_parameter_heatmap(
                        results, param1, param2, save_path=heatmap_path, show_plot=False
                    )

    # Save results to CSV
    if results["best_result"]:
        best_params = results["best_result"]["parameters"]
        best_df = pd.DataFrame([best_params])
        best_df["target_value"] = results["best_result"]["target_value"]
        best_df["final_balance"] = results["best_result"]["final_balance"]
        best_df["num_trades"] = results["best_result"]["num_trades"]

        best_df.to_csv(os.path.join(output_dir, "best_parameters.csv"), index=False)

    # Save all results
    all_results_list = []
    for result in results["all_results"]:
        row = result["parameters"].copy()
        row["target_value"] = result["target_value"]
        row["final_balance"] = result["final_balance"]
        row["num_trades"] = result["num_trades"]
        all_results_list.append(row)

    if all_results_list:
        all_df = pd.DataFrame(all_results_list)
        all_df.to_csv(os.path.join(output_dir, "all_results.csv"), index=False)

    return results
