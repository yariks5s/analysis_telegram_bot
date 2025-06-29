import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional, Callable
import os
from datetime import datetime

from .performance_metrics import calculate_performance_metrics

class StrategyComparison:
    """
    Comparative analysis of multiple trading strategies to identify
    the most effective approach under different market conditions.
    """
    
    def __init__(self):
        """Initialize strategy comparison"""
        self.strategies = {}
        self.benchmark = None
    
    def add_strategy(
        self,
        name: str,
        backtest_func: Callable,
        parameters: Dict[str, Any] = {}
    ) -> None:
        """
        Add a strategy to the comparison.
        
        Args:
            name: Strategy name for identification
            backtest_func: Function that performs backtesting
            parameters: Parameters to pass to the backtest function
        """
        self.strategies[name] = {
            "func": backtest_func,
            "parameters": parameters,
            "results": None
        }
    
    def add_benchmark(
        self,
        name: str = "Buy and Hold",
        backtest_func: Callable = None,
        parameters: Dict[str, Any] = {}
    ) -> None:
        """
        Add a benchmark strategy (e.g., buy and hold) for comparison.
        
        Args:
            name: Benchmark name
            backtest_func: Function that performs benchmark backtesting
            parameters: Parameters to pass to the benchmark function
        """
        self.benchmark = {
            "name": name,
            "func": backtest_func,
            "parameters": parameters,
            "results": None
        }
    
    def run_comparison(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        initial_balance: float = 10000.0,
        **shared_params
    ) -> Dict[str, Any]:
        """
        Run backtest for all strategies with the same initial conditions.
        
        Args:
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_balance: Initial account balance
            **shared_params: Additional parameters to pass to all backtest functions
            
        Returns:
            Dictionary with comparison results
        """
        # Store common parameters
        common_params = {
            "initial_balance": initial_balance,
            **shared_params
        }
        
        if start_date:
            common_params["start_date"] = start_date
        if end_date:
            common_params["end_date"] = end_date
        
        # Run benchmark if available
        if self.benchmark:
            print(f"Running benchmark: {self.benchmark['name']}...")
            params = {**common_params, **self.benchmark["parameters"]}
            
            try:
                benchmark_balance, benchmark_trades, benchmark_metrics = self.benchmark["func"](**params)
                
                # If metrics not provided by backtest function, calculate them
                if benchmark_metrics is None and benchmark_trades:
                    benchmark_metrics = calculate_performance_metrics(
                        benchmark_trades, initial_balance, benchmark_balance
                    )
                
                self.benchmark["results"] = {
                    "balance": benchmark_balance,
                    "trades": benchmark_trades,
                    "metrics": benchmark_metrics
                }
            except Exception as e:
                print(f"Error running benchmark: {str(e)}")
        
        # Run backtest for each strategy
        for name, strategy in self.strategies.items():
            print(f"Running strategy: {name}...")
            params = {**common_params, **strategy["parameters"]}
            
            try:
                balance, trades, metrics = strategy["func"](**params)
                
                # If metrics not provided by backtest function, calculate them
                if metrics is None and trades:
                    metrics = calculate_performance_metrics(
                        trades, initial_balance, balance
                    )
                
                self.strategies[name]["results"] = {
                    "balance": balance,
                    "trades": trades,
                    "metrics": metrics
                }
            except Exception as e:
                print(f"Error running strategy {name}: {str(e)}")
        
        # Compile comparison results
        comparison_results = self._compile_comparison_results()
        
        return comparison_results
    
    def _compile_comparison_results(self) -> Dict[str, Any]:
        """
        Compile results for all strategies into a comparable format.
        
        Returns:
            Dictionary with comparative metrics
        """
        # Collect metrics for each strategy
        strategy_metrics = {}
        
        for name, strategy in self.strategies.items():
            if strategy["results"] and strategy["results"].get("metrics"):
                strategy_metrics[name] = strategy["results"]["metrics"]
        
        # Add benchmark if available
        if self.benchmark and self.benchmark["results"] and self.benchmark["results"].get("metrics"):
            strategy_metrics[self.benchmark["name"]] = self.benchmark["results"]["metrics"]
        
        # Calculate relative performance vs benchmark
        if self.benchmark and self.benchmark["name"] in strategy_metrics:
            benchmark_name = self.benchmark["name"]
            benchmark_return = strategy_metrics[benchmark_name].get("total_return", 0)
            
            for name in strategy_metrics:
                if name != benchmark_name:
                    strategy_return = strategy_metrics[name].get("total_return", 0)
                    strategy_metrics[name]["vs_benchmark"] = strategy_return - benchmark_return
        
        return {
            "strategy_metrics": strategy_metrics,
            "strategies": self.strategies,
            "benchmark": self.benchmark
        }
    
    def plot_equity_curves(
        self,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> None:
        """
        Plot equity curves for all strategies on the same chart.
        
        Args:
            save_path: Path to save the chart image
            show_plot: Whether to display the chart
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Add equity curve for each strategy
        for name, strategy in self.strategies.items():
            if not strategy["results"] or not strategy["results"].get("trades"):
                continue
                
            trades = strategy["results"]["trades"]
            initial_balance = strategy["parameters"].get("initial_balance", 10000)
            
            # Build equity curve
            equity = [initial_balance]
            
            for trade in trades:
                profit = trade.get("profit", trade.get("profit_loss", 0))
                equity.append(equity[-1] + profit)
            
            # Plot this strategy's equity curve
            plt.plot(equity, label=name)
        
        # Add benchmark if available
        if self.benchmark and self.benchmark["results"] and self.benchmark["results"].get("trades"):
            benchmark_trades = self.benchmark["results"]["trades"]
            initial_balance = self.benchmark["parameters"].get("initial_balance", 10000)
            
            # Build benchmark equity curve
            equity = [initial_balance]
            
            for trade in benchmark_trades:
                profit = trade.get("profit", trade.get("profit_loss", 0))
                equity.append(equity[-1] + profit)
            
            # Plot benchmark equity curve
            plt.plot(equity, label=f"{self.benchmark['name']}", linestyle="--")
        
        plt.title("Strategy Performance Comparison")
        plt.xlabel("Trade Number")
        plt.ylabel("Account Balance")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_comparative_metrics(
        self,
        metrics: List[str] = ["sharpe_ratio", "max_drawdown", "win_rate", "profit_factor"],
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> None:
        """
        Plot comparative metrics for all strategies.
        
        Args:
            metrics: List of metrics to compare
            save_path: Path to save the chart image
            show_plot: Whether to display the chart
        """
        # Collect results
        comparison_results = self._compile_comparison_results()
        strategy_metrics = comparison_results.get("strategy_metrics", {})
        
        if not strategy_metrics:
            print("No strategy metrics available for plotting")
            return
        
        # Create figure with subplots for each metric
        fig, axs = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        
        if len(metrics) == 1:
            axs = [axs]  # Make iterable if only one metric
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            # Extract metric values for each strategy
            strategy_names = []
            metric_values = []
            colors = []
            
            for name, metrics_dict in strategy_metrics.items():
                if metric in metrics_dict:
                    strategy_names.append(name)
                    metric_values.append(metrics_dict[metric])
                    
                    # Use different color for benchmark
                    if self.benchmark and name == self.benchmark["name"]:
                        colors.append("orange")
                    else:
                        colors.append("blue")
            
            # Skip if no values for this metric
            if not metric_values:
                axs[i].set_title(f"{metric} - No data available")
                continue
                
            # Create bar chart
            bars = axs[i].bar(strategy_names, metric_values, color=colors)
            axs[i].set_title(f"Comparison: {metric}")
            axs[i].set_ylabel(metric)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axs[i].text(
                    bar.get_x() + bar.get_width()/2.,
                    height * 1.01,
                    f"{height:.2f}",
                    ha='center', va='bottom', rotation=0
                )
            
            # Adjust for better readability
            if len(strategy_names) > 3:
                plt.setp(axs[i].get_xticklabels(), rotation=45, ha="right")
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def generate_comparison_report(
        self,
        output_dir: str = "./reports/comparison",
        report_name: str = "strategy_comparison"
    ) -> str:
        """
        Generate comprehensive comparison report with charts and tables.
        
        Args:
            output_dir: Directory to save the report
            report_name: Base name for report files
            
        Returns:
            Path to the report directory
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate equity curve chart
        equity_path = os.path.join(output_dir, f"{report_name}_equity_curves.png")
        self.plot_equity_curves(equity_path, False)
        
        # Generate comparative metrics chart
        metrics_path = os.path.join(output_dir, f"{report_name}_metrics.png")
        self.plot_comparative_metrics(save_path=metrics_path, show_plot=False)
        
        # Compile results for CSV
        comparison_results = self._compile_comparison_results()
        strategy_metrics = comparison_results.get("strategy_metrics", {})
        
        # Create summary dataframe
        summary_data = []
        for name, metrics in strategy_metrics.items():
            row = {"Strategy": name}
            row.update(metrics)
            summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_csv_path = os.path.join(output_dir, f"{report_name}_summary.csv")
            summary_df.to_csv(summary_csv_path, index=False)
        
        return output_dir

def compare_strategies(
    strategies: Dict[str, Dict[str, Any]],
    benchmark_func: Optional[Callable] = None,
    benchmark_params: Optional[Dict[str, Any]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    initial_balance: float = 10000.0,
    output_dir: str = "./reports/comparison",
    **shared_params
) -> Dict[str, Any]:
    """
    Convenience function to compare multiple strategies and generate reports.
    
    Args:
        strategies: Dict mapping strategy names to dicts with 'func' and 'parameters' keys
        benchmark_func: Optional benchmark function
        benchmark_params: Parameters for benchmark function
        start_date: Start date for backtesting
        end_date: End date for backtesting
        initial_balance: Initial account balance
        output_dir: Directory to save reports
        **shared_params: Additional parameters for all backtest functions
        
    Returns:
        Dictionary with comparison results
    """
    # Initialize comparison
    comparison = StrategyComparison()
    
    # Add strategies
    for name, strategy_info in strategies.items():
        comparison.add_strategy(
            name=name,
            backtest_func=strategy_info.get("func"),
            parameters=strategy_info.get("parameters", {})
        )
    
    # Add benchmark if provided
    if benchmark_func:
        comparison.add_benchmark(
            backtest_func=benchmark_func,
            parameters=benchmark_params or {}
        )
    
    # Run comparison
    results = comparison.run_comparison(
        start_date=start_date,
        end_date=end_date,
        initial_balance=initial_balance,
        **shared_params
    )
    
    # Generate reports
    comparison.generate_comparison_report(output_dir=output_dir)
    
    return results
