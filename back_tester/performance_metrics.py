import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import os

def calculate_performance_metrics(
    trades: List[Dict[str, Any]], 
    initial_balance: float, 
    final_balance: float,
    benchmark_returns: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics for backtesting results.
    
    Args:
        trades: List of trade dictionaries containing trade details
        initial_balance: Starting account balance
        final_balance: Ending account balance
        benchmark_returns: Optional list of benchmark returns (e.g. buy and hold)
        
    Returns:
        Dict containing performance metrics
    """
    # Extract trade data
    if not trades:
        return {
            "win_rate": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "sortino_ratio": 0,
            "total_return": 0,
            "total_trades": 0,
            "avg_profit_per_trade": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "expectancy": 0,
        }
    
    profits = [t.get('profit', 0) for t in trades if 'profit' in t]
    
    # Handle case with no profits data
    if not profits:
        return {
            "win_rate": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "sortino_ratio": 0,
            "total_return": 0,
            "total_trades": 0,
            "avg_profit_per_trade": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "expectancy": 0,
        }
    
    # Win rate
    winning_trades = [p for p in profits if p > 0]
    losing_trades = [p for p in profits if p < 0]
    win_rate = len(winning_trades) / len(profits) if profits else 0
    
    # Profit factor
    profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if sum(losing_trades) < 0 else float('inf')
    
    # Drawdown calculation
    equity_curve = [initial_balance]
    for profit in profits:
        equity_curve.append(equity_curve[-1] + profit)
    
    running_max = equity_curve[0]
    drawdowns = []
    
    for equity in equity_curve:
        if equity > running_max:
            running_max = equity
        drawdown_pct = (running_max - equity) / running_max * 100 if running_max > 0 else 0
        drawdowns.append(drawdown_pct)
    
    max_drawdown = max(drawdowns) if drawdowns else 0
    
    # Calculate returns
    returns = [(equity_curve[i] - equity_curve[i-1])/equity_curve[i-1] for i in range(1, len(equity_curve)) if equity_curve[i-1] > 0]
    
    # Sharpe ratio (assuming 252 trading days per year, risk-free rate = 0)
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if returns and np.std(returns) > 0 else 0
    
    # Sortino ratio (only considers downside deviation)
    negative_returns = [r for r in returns if r < 0]
    downside_deviation = np.std(negative_returns) if negative_returns else 0.0001
    sortino = np.mean(returns) / downside_deviation * np.sqrt(252) if returns else 0
    
    # Average metrics
    avg_profit = sum(profits) / len(profits) if profits else 0
    avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
    
    # Largest win/loss
    largest_win = max(winning_trades) if winning_trades else 0
    largest_loss = min(losing_trades) if losing_trades else 0
    
    # Expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # Total return
    total_return = (final_balance - initial_balance) / initial_balance * 100
    
    # Compare against benchmark if provided
    benchmark_comparison = {}
    if benchmark_returns and len(benchmark_returns) > 0:
        benchmark_return = (benchmark_returns[-1] - benchmark_returns[0]) / benchmark_returns[0] * 100
        alpha = total_return - benchmark_return
        benchmark_comparison = {
            "benchmark_return": benchmark_return,
            "alpha": alpha,
        }
    
    metrics = {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "total_return": total_return,
        "total_trades": len(profits),
        "avg_profit_per_trade": avg_profit,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "expectancy": expectancy,
        **benchmark_comparison
    }
    
    return metrics

def plot_equity_curve(
    trades: List[Dict[str, Any]], 
    initial_balance: float,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    benchmark_returns: Optional[List[float]] = None
) -> None:
    """
    Generate and display equity curve with drawdown visualization.
    
    Args:
        trades: List of trade dictionaries
        initial_balance: Initial account balance
        save_path: Path to save the chart image (optional)
        show_plot: Whether to display the chart
        benchmark_returns: Optional benchmark data for comparison
    """
    profits = [t.get('profit', 0) for t in trades if 'profit' in t]
    timestamps = [t.get('timestamp', i) for i, t in enumerate(trades) if 'profit' in t]
    
    equity_curve = [initial_balance]
    for profit in profits:
        equity_curve.append(equity_curve[-1] + profit)
    
    # Calculate drawdowns
    running_max = [equity_curve[0]]
    for i in range(1, len(equity_curve)):
        running_max.append(max(running_max[i-1], equity_curve[i]))
    
    drawdowns = [(running_max[i] - equity_curve[i]) / running_max[i] * 100 if running_max[i] > 0 else 0 
                for i in range(len(equity_curve))]
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curve
    ax1.plot(equity_curve, label='Strategy', color='blue', linewidth=2)
    
    # Add benchmark comparison if provided
    if benchmark_returns and len(benchmark_returns) > 0:
        # Scale benchmark to start at initial_balance
        scale_factor = initial_balance / benchmark_returns[0]
        scaled_benchmark = [b * scale_factor for b in benchmark_returns]
        
        # Ensure both arrays are the same length
        min_length = min(len(equity_curve), len(scaled_benchmark))
        ax1.plot(scaled_benchmark[:min_length], label='Benchmark', color='green', linestyle='--')
    
    ax1.set_title('Equity Curve')
    ax1.set_ylabel('Account Balance')
    ax1.grid(True)
    ax1.legend()
    
    # Plot drawdowns
    ax2.fill_between(range(len(drawdowns)), drawdowns, 0, color='red', alpha=0.3)
    ax2.set_title('Drawdowns')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Trade Number')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

def generate_performance_report(
    trades: List[Dict[str, Any]], 
    initial_balance: float, 
    final_balance: float,
    output_dir: str = "./reports",
    report_name: str = "backtest_report",
    benchmark_returns: Optional[List[float]] = None
) -> str:
    """
    Generate a comprehensive performance report with metrics and visualizations.
    
    Args:
        trades: List of trade dictionaries
        initial_balance: Initial account balance
        final_balance: Final account balance
        output_dir: Directory to save the report
        report_name: Base name for the report files
        benchmark_returns: Optional benchmark data for comparison
        
    Returns:
        Path to the generated report files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(trades, initial_balance, final_balance, benchmark_returns)
    
    # Generate equity curve chart
    chart_path = os.path.join(output_dir, f"{report_name}_equity_curve.png")
    plot_equity_curve(trades, initial_balance, chart_path, False, benchmark_returns)
    
    # Create trade analysis dataframe
    trade_df = pd.DataFrame(trades)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_csv_path = os.path.join(output_dir, f"{report_name}_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    
    # Save trades to CSV if they contain timestamps
    if 'timestamp' in trades[0] if trades else False:
        trades_csv_path = os.path.join(output_dir, f"{report_name}_trades.csv")
        trade_df.to_csv(trades_csv_path, index=False)
    
    return output_dir
