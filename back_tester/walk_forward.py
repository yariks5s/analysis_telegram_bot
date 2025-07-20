import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Callable
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

from .performance_metrics import calculate_performance_metrics, plot_equity_curve


class WalkForwardAnalysis:
    """
    Implements walk-forward analysis to prevent overfitting and validate trading strategies.
    This approach uses temporal data segmentation to train on in-sample data and test on
    out-of-sample data in chronological order.
    """

    def __init__(
        self,
        backtest_func: Callable,
        train_size: float = 0.7,
        window_type: str = "expanding",
        num_windows: int = 5,
        overlap: float = 0.0,
    ):
        """
        Initialize walk-forward analysis.

        Args:
            backtest_func: Function that performs backtesting (must accept start_date and end_date parameters)
            train_size: Proportion of window to use for training (default 70%)
            window_type: "expanding" or "rolling" windows
            num_windows: Number of train/test windows to use
            overlap: Overlap between consecutive windows (0.0 to 0.5)
        """
        self.backtest_func = backtest_func
        self.train_size = train_size
        self.window_type = window_type
        self.num_windows = num_windows
        self.overlap = overlap

    def generate_windows(
        self, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Generate train/test windows for walk-forward analysis.

        Args:
            start_date: Start date for the entire dataset
            end_date: End date for the entire dataset

        Returns:
            List of dictionaries containing start/end dates for train and test periods
        """
        total_days = (end_date - start_date).days
        window_days = total_days // self.num_windows

        # Calculate overlap in days
        overlap_days = int(window_days * self.overlap)

        windows = []
        current_start = start_date

        for i in range(self.num_windows):
            # For expanding window, always start from the original start date
            if self.window_type == "expanding" and i > 0:
                window_start = start_date
            else:
                window_start = current_start

            # Calculate end of current window
            window_end = window_start + timedelta(days=window_days)

            # Ensure we don't go beyond the overall end date
            window_end = min(window_end, end_date)

            # Calculate train/test split point
            train_days = int((window_end - window_start).days * self.train_size)
            train_end = window_start + timedelta(days=train_days)

            windows.append(
                {
                    "window_num": i + 1,
                    "train_start": window_start,
                    "train_end": train_end,
                    "test_start": train_end + timedelta(days=1),
                    "test_end": window_end,
                }
            )

            # Move to next window start, accounting for overlap
            current_start = window_start + timedelta(days=window_days - overlap_days)

            # Break if we've reached the end
            if window_end >= end_date:
                break

        return windows

    def run_analysis(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_balance: float = 10000.0,
        **backtest_kwargs,
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis over the specified date range.

        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            initial_balance: Initial account balance
            **backtest_kwargs: Additional arguments to pass to backtest function

        Returns:
            Dictionary with analysis results
        """
        windows = self.generate_windows(start_date, end_date)
        results = []

        all_test_trades = []
        current_balance = initial_balance

        for window in windows:
            print(f"Processing window {window['window_num']}/{len(windows)}...")

            # Run backtesting on training period
            train_balance, train_trades, train_metrics = self.backtest_func(
                start_date=window["train_start"],
                end_date=window["train_end"],
                initial_balance=initial_balance,
                **backtest_kwargs,
            )

            # Run backtesting on test period using parameters optimized on training period
            test_balance, test_trades, test_metrics = self.backtest_func(
                start_date=window["test_start"],
                end_date=window["test_end"],
                initial_balance=current_balance,
                **backtest_kwargs,
            )

            # Update balance for next window (if using expanding window)
            if self.window_type == "expanding":
                current_balance = test_balance

            # Store results
            window_result = {
                "window_num": window["window_num"],
                "train_start": window["train_start"],
                "train_end": window["train_end"],
                "test_start": window["test_start"],
                "test_end": window["test_end"],
                "train_return_pct": (train_balance - initial_balance)
                / initial_balance
                * 100,
                "test_return_pct": (test_balance - current_balance)
                / current_balance
                * 100,
                "train_num_trades": len(train_trades),
                "test_num_trades": len(test_trades),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "train_balance": train_balance,
                "test_balance": test_balance,
            }

            results.append(window_result)
            all_test_trades.extend(test_trades)

        # Calculate overall metrics from all test periods
        overall_metrics = {
            "total_windows": len(results),
            "profitable_windows": sum(1 for r in results if r["test_return_pct"] > 0),
            "avg_test_return": np.mean([r["test_return_pct"] for r in results]),
            "median_test_return": np.median([r["test_return_pct"] for r in results]),
            "consistency_score": self._calculate_consistency(
                [r["test_return_pct"] for r in results]
            ),
            "robustness_score": self._calculate_robustness(results),
            "final_balance": (
                results[-1]["test_balance"] if results else initial_balance
            ),
            "total_return_pct": (
                (results[-1]["test_balance"] - initial_balance) / initial_balance * 100
                if results
                else 0
            ),
        }

        return {
            "windows": results,
            "overall": overall_metrics,
            "all_test_trades": all_test_trades,
        }

    def _calculate_consistency(self, returns: List[float]) -> float:
        """
        Calculate consistency score based on how consistently the strategy performs.

        Args:
            returns: List of returns for each window

        Returns:
            Consistency score between 0 and 1
        """
        # Simple consistency measure: percentage of positive returns
        positive_returns = sum(1 for r in returns if r > 0)
        return positive_returns / len(returns) if returns else 0

    def _calculate_robustness(self, window_results: List[Dict[str, Any]]) -> float:
        """
        Calculate robustness score based on comparison of train/test performance.

        Args:
            window_results: List of results for each window

        Returns:
            Robustness score between 0 and 1
        """
        if not window_results:
            return 0

        # Calculate ratio of test returns to training returns
        # Closer to 1 means more robust (test performs similar to train)
        ratios = []

        for window in window_results:
            train_return = window["train_return_pct"]
            test_return = window["test_return_pct"]

            # Skip windows where training return is near zero to avoid division issues
            if abs(train_return) < 0.001:
                continue

            if train_return > 0 and test_return > 0:
                # Both positive, ratio of smaller to larger
                ratio = min(train_return, test_return) / max(train_return, test_return)
                ratios.append(ratio)
            elif train_return < 0 and test_return < 0:
                # Both negative, ratio of larger to smaller (less negative is better)
                ratio = max(train_return, test_return) / min(train_return, test_return)
                ratios.append(ratio)
            else:
                # One positive, one negative - no robustness
                ratios.append(0)

        return np.mean(ratios) if ratios else 0

    def plot_results(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> None:
        """
        Plot walk-forward analysis results.

        Args:
            results: Results from run_analysis
            save_path: Path to save the chart image
            show_plot: Whether to display the chart
        """
        windows = results["windows"]

        # Create figure with subplots
        fig, axs = plt.subplots(
            2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Plot 1: Returns by window
        train_returns = [w["train_return_pct"] for w in windows]
        test_returns = [w["test_return_pct"] for w in windows]
        window_nums = [w["window_num"] for w in windows]

        x = np.arange(len(window_nums))
        width = 0.35

        axs[0].bar(x - width / 2, train_returns, width, label="In-Sample (Train)")
        axs[0].bar(x + width / 2, test_returns, width, label="Out-of-Sample (Test)")

        axs[0].set_ylabel("Return (%)")
        axs[0].set_title("Walk-Forward Analysis Results")
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(window_nums)
        axs[0].legend()
        axs[0].grid(True, linestyle="--", alpha=0.7)

        # Add horizontal line at 0
        axs[0].axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Plot 2: Consistency metrics
        consistency_metric = []
        robustness_metric = []

        for i in range(len(windows)):
            # Calculate cumulative consistency up to this window
            curr_windows = windows[: i + 1]
            curr_test_returns = [w["test_return_pct"] for w in curr_windows]

            consistency = sum(1 for r in curr_test_returns if r > 0) / len(
                curr_test_returns
            )
            consistency_metric.append(consistency)

            # Calculate cumulative robustness up to this window
            robustness = self._calculate_robustness(curr_windows)
            robustness_metric.append(robustness)

        axs[1].plot(
            window_nums, consistency_metric, "b-", marker="o", label="Consistency"
        )
        axs[1].plot(
            window_nums, robustness_metric, "r-", marker="s", label="Robustness"
        )

        axs[1].set_ylim([0, 1.1])
        axs[1].set_xlabel("Window Number")
        axs[1].set_ylabel("Score")
        axs[1].legend()
        axs[1].grid(True, linestyle="--", alpha=0.7)

        # Add annotations for overall metrics
        overall = results["overall"]
        plt.figtext(
            0.01,
            0.01,
            f"Overall Metrics:\n"
            f"Total Return: {overall['total_return_pct']:.2f}%\n"
            f"Profitable Windows: {overall['profitable_windows']}/{overall['total_windows']}\n"
            f"Consistency Score: {overall['consistency_score']:.2f}\n"
            f"Robustness Score: {overall['robustness_score']:.2f}",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7),
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


def walk_forward_backtest(
    backtest_func: Callable,
    start_date: datetime,
    end_date: datetime,
    initial_balance: float = 10000.0,
    train_size: float = 0.7,
    window_type: str = "expanding",
    num_windows: int = 5,
    output_dir: str = "./reports/walk_forward",
    **backtest_kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to run a walk-forward backtest and generate reports.

    Args:
        backtest_func: Function that performs backtesting
        start_date: Start date for analysis
        end_date: End date for analysis
        initial_balance: Initial account balance
        train_size: Proportion to use for training
        window_type: "expanding" or "rolling"
        num_windows: Number of windows to use
        output_dir: Directory to save reports
        **backtest_kwargs: Additional arguments for backtest function

    Returns:
        Dictionary with analysis results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize walk-forward analyzer
    wfa = WalkForwardAnalysis(
        backtest_func=backtest_func,
        train_size=train_size,
        window_type=window_type,
        num_windows=num_windows,
    )

    # Run analysis
    results = wfa.run_analysis(
        start_date=start_date,
        end_date=end_date,
        initial_balance=initial_balance,
        **backtest_kwargs,
    )

    # Generate plots
    plot_path = os.path.join(output_dir, "walk_forward_results.png")
    wfa.plot_results(results, plot_path, False)

    # Generate equity curve from all test trades
    if results["all_test_trades"]:
        equity_path = os.path.join(output_dir, "walk_forward_equity.png")
        plot_equity_curve(
            results["all_test_trades"], initial_balance, equity_path, False
        )

    # Save results to CSV
    results_df = pd.DataFrame([results["overall"]])
    results_df.to_csv(os.path.join(output_dir, "overall_results.csv"), index=False)

    windows_df = pd.DataFrame(
        [
            {
                "window_num": w["window_num"],
                "train_start": w["train_start"],
                "train_end": w["train_end"],
                "test_start": w["test_start"],
                "test_end": w["test_end"],
                "train_return_pct": w["train_return_pct"],
                "test_return_pct": w["test_return_pct"],
                "train_num_trades": w["train_num_trades"],
                "test_num_trades": w["test_num_trades"],
            }
            for w in results["windows"]
        ]
    )

    windows_df.to_csv(os.path.join(output_dir, "window_results.csv"), index=False)

    return results
