#!/usr/bin/env python3
"""
Example usage of the enhanced backtesting system.

This file contains examples showing how to use the various
features of the enhanced backtesting system.
"""

import os
import pandas as pd
from datetime import datetime, timedelta

from back_tester.enhanced_backtester import EnhancedBacktester, run_enhanced_backtest
from back_tester.performance_metrics import calculate_performance_metrics, plot_equity_curve
from back_tester.monte_carlo import monte_carlo_simulation
from back_tester.transaction_costs import TransactionCostsModel
from back_tester.risk_management import AdaptiveRiskManager
from back_tester.walk_forward import walk_forward_backtest
from back_tester.strategy_optimizer import optimize_strategy
from back_tester.strategy_comparison import compare_strategies
from back_tester.correlation_analysis import analyze_correlations

# Example 1: Basic Enhanced Backtest
def example_basic_backtest():
    """Run a basic enhanced backtest with full reporting"""
    print("\n=== Example 1: Basic Enhanced Backtest ===")
    
    # Run backtest with default parameters
    summary = run_enhanced_backtest(
        symbol="BTCUSDT",
        interval="1h",
        candles=600,  # Reduced for faster execution
        window=300,
        initial_balance=10000.0,
        risk_percentage=1.0,
        output_dir="./reports/examples"
    )
    
    print("\nBacktest Summary:")
    print(f"Final Balance: ${summary['final_balance']:.2f}")
    print(f"Return: {summary['return_pct']:.2f}%")
    print(f"Win Rate: {summary['win_rate']:.2f}%")
    print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {summary['max_drawdown']:.2f}%")
    print(f"Report saved to: {summary['report_directory']}")

# Example 2: Parameter Optimization
def example_parameter_optimization():
    """Optimize strategy parameters"""
    print("\n=== Example 2: Parameter Optimization ===")
    
    # Initialize backtester
    backtester = EnhancedBacktester(output_dir="./reports/examples")
    
    # Define parameter ranges to optimize
    param_ranges = {
        "window": (200, 400),        # Test windows from 200 to 400
        "risk_percentage": (0.5, 2.0) # Test risk from 0.5% to 2%
    }
    
    # Run optimization with fewer trials for demonstration
    results = backtester.optimize_parameters(
        symbol="BTCUSDT",
        interval="1h",
        param_ranges=param_ranges,
        optimization_target="sharpe_ratio",
        search_method="random",
        num_trials=20,  # Reduced for faster execution
        candles=600     # Reduced for faster execution
    )
    
    # Display best parameters
    if results["best_result"]:
        best_params = results["best_result"]["parameters"]
        best_value = results["best_result"]["target_value"]
        print(f"\nBest parameters found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"Sharpe Ratio: {best_value:.2f}")
        print(f"Final Balance: ${results['best_result']['final_balance']:.2f}")
    else:
        print("Optimization did not find valid results")

# Example 3: Strategy Comparison
def example_strategy_comparison():
    """Compare different strategy configurations"""
    print("\n=== Example 3: Strategy Comparison ===")
    
    # Initialize backtester
    backtester = EnhancedBacktester(output_dir="./reports/examples")
    
    # Define different strategies (different parameter configurations)
    strategies = {
        "Conservative": {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "candles": 600,
            "window": 300,
            "risk_percentage": 0.5,
            "initial_balance": 10000.0
        },
        "Balanced": {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "candles": 600,
            "window": 250,
            "risk_percentage": 1.0,
            "initial_balance": 10000.0
        },
        "Aggressive": {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "candles": 600,
            "window": 200,
            "risk_percentage": 2.0,
            "initial_balance": 10000.0
        }
    }
    
    # Define buy and hold benchmark
    benchmark_params = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "candles": 600,
        "window": 300,
        "risk_percentage": 100.0,  # Use a large risk percentage to simulate buy and hold
        "initial_balance": 10000.0
    }
    
    # Run comparison
    results = backtester.compare_strategies(
        strategies=strategies,
        benchmark_params=benchmark_params,
        initial_balance=10000.0
    )
    
    print("\nComparison completed. Results saved to: ./reports/examples/strategy_comparison")

# Example 4: Walk-Forward Analysis
def example_walk_forward():
    """Demonstrate walk-forward analysis"""
    print("\n=== Example 4: Walk-Forward Analysis ===")
    
    # Initialize backtester
    backtester = EnhancedBacktester(output_dir="./reports/examples")
    
    # Define date range (using relative dates for demo)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months ago
    
    # Run walk-forward analysis
    results = backtester.run_walk_forward_analysis(
        symbol="BTCUSDT",
        interval="1h",
        start_date=start_date,
        end_date=end_date,
        train_size=0.7,
        num_windows=3,
        window_type="expanding",
        candles=600,
        window=300,
        risk_percentage=1.0,
        initial_balance=10000.0
    )
    
    print("\nWalk-Forward Analysis completed. Results saved to: ./reports/examples/walk_forward_BTCUSDT_1h")

# Example 5: Monte Carlo Simulation
def example_monte_carlo():
    """Run Monte Carlo simulation on backtest results"""
    print("\n=== Example 5: Monte Carlo Simulation ===")
    
    # Initialize backtester and run a backtest
    backtester = EnhancedBacktester(output_dir="./reports/examples")
    final_balance, trades, metrics = backtester.run_backtest(
        symbol="BTCUSDT",
        interval="1h",
        candles=600,
        window=300,
        initial_balance=10000.0,
        risk_percentage=1.0
    )
    
    # Debug: Check the structure of trades
    print("\nExamining trade structure...")
    if trades:
        print(f"Number of trades: {len(trades)}")
        print(f"Keys in first trade: {list(trades[0].keys())}")
        
    # Run Monte Carlo simulation on the trades
    print("\nRunning Monte Carlo simulations...")
    mc_results = monte_carlo_simulation(
        trades=trades,
        initial_balance=10000.0,
        simulations=500,
        confidence_interval=0.95
    )
    
    # Display results
    print(f"Original final balance: ${final_balance:.2f}")
    
    # Debug: Print all available keys in mc_results
    print("\nMonte Carlo result keys:")
    for key in mc_results.keys():
        print(f"  - {key}")
    
    # Use appropriate key names with fallbacks
    mean_balance_key = next((k for k in ['mean_final_balance', 'mean_final_equity', 'mean_equity'] if k in mc_results), None)
    lower_ci_key = next((k for k in ['confidence_interval_lower', 'ci_lower', 'lower_bound'] if k in mc_results), None)
    upper_ci_key = next((k for k in ['confidence_interval_upper', 'ci_upper', 'upper_bound'] if k in mc_results), None)
    drawdown_key = next((k for k in ['mean_max_drawdown', 'mean_drawdown', 'max_drawdown'] if k in mc_results), None)
    profitable_key = next((k for k in ['profitable_simulations_pct', 'profitable_pct', 'win_rate'] if k in mc_results), None)
    
    if mean_balance_key:
        print(f"Mean final balance from MC: ${mc_results[mean_balance_key]:.2f}")
    
    if lower_ci_key and upper_ci_key:
        print(f"95% confidence interval: [${mc_results[lower_ci_key]:.2f}, ${mc_results[upper_ci_key]:.2f}]")
    
    if drawdown_key:
        print(f"Mean maximum drawdown: {mc_results[drawdown_key]:.2f}%")
    
    if profitable_key:
        print(f"Profitable simulations: {mc_results[profitable_key]:.1f}%")

def main():
    """Run all examples"""
    # Create reports directory
    os.makedirs("./reports/examples", exist_ok=True)
    
    # Run examples with better error isolation
    run_example("Basic Enhanced Backtest", example_basic_backtest)
    run_example("Parameter Optimization", example_parameter_optimization)
    run_example("Strategy Comparison", example_strategy_comparison)
    run_example("Walk-Forward Analysis", example_walk_forward)
    run_example("Monte Carlo Simulation", example_monte_carlo)
    
    print("\nExamples run complete!")
    print("Check the ./reports/examples directory for detailed reports.")
    
def run_example(name, func):
    """Run an example function with error handling"""
    print(f"\n--- Running {name} Example ---")
    try:
        func()
        print(f"✅ {name} example completed successfully")
    except Exception as e:
        import traceback
        print(f"❌ Error in {name} example: {str(e)}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
