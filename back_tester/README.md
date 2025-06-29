# Enhanced Backtesting System

This enhanced backtesting system provides comprehensive tools for strategy development, testing, and optimization for cryptocurrency trading.

## Features

- **Performance Metrics** - Advanced metrics including Sharpe ratio, Sortino ratio, win rate, profit factor, and maximum drawdown
- **Monte Carlo Simulation** - Test strategy robustness through randomization
- **Transaction Costs** - Realistic modeling of trading fees and slippage
- **Risk Management** - Adaptive position sizing based on volatility and market conditions
- **Walk-Forward Analysis** - Out-of-sample testing to reduce overfitting
- **Parameter Optimization** - Find optimal strategy parameters using grid or random search
- **Strategy Comparison** - Compare multiple strategies against benchmarks
- **Correlation Analysis** - Analyze relationship between strategy performance and market movements

## Modules

### 1. Performance Metrics (`performance_metrics.py`)

Provides comprehensive performance evaluation including:
- Win rate and profit factor
- Drawdown analysis
- Sharpe and Sortino ratios
- Equity curve visualization

```python
from performance_metrics import calculate_performance_metrics, plot_equity_curve

# Calculate metrics
metrics = calculate_performance_metrics(trades, initial_balance, final_balance)
print(f"Win Rate: {metrics['win_rate']:.2f}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

# Plot equity curve
plot_equity_curve(trades, initial_balance)
```

### 2. Monte Carlo Simulation (`monte_carlo.py`)

Test strategy robustness through randomized trade sequences:
- Statistical confidence intervals for returns
- Maximum drawdown distribution
- Probability of profitability

```python
from monte_carlo import monte_carlo_simulation, plot_monte_carlo_results

# Run simulation
results = monte_carlo_simulation(trades, initial_balance, simulations=1000)
print(f"95% Confidence Interval: [{results['confidence_interval_lower']:.2f}, {results['confidence_interval_upper']:.2f}]")

# Visualize results
plot_monte_carlo_results(results, trades, initial_balance)
```

### 3. Transaction Costs (`transaction_costs.py`)

Model realistic trading frictions:
- Exchange fees (maker/taker)
- Slippage based on volatility or fixed percentage
- Market impact estimation

```python
from transaction_costs import TransactionCostsModel, apply_transaction_costs_to_backtest

# Create model
costs_model = TransactionCostsModel(
    maker_fee_pct=0.1,
    taker_fee_pct=0.1,
    slippage_model="volatility",
    slippage_vol_factor=0.5
)

# Apply to backtest results
adjusted_trades = apply_transaction_costs_to_backtest(trades_df, costs_model)
```

### 4. Risk Management (`risk_management.py`)

Advanced risk management:
- Adaptive position sizing based on volatility
- Correlation-based risk adjustment
- Dynamic stop-loss placement

```python
from risk_management import AdaptiveRiskManager, DynamicStopLossManager

# Initialize risk manager
risk_manager = AdaptiveRiskManager(
    base_risk_pct=1.0,
    max_risk_pct=2.0,
    min_risk_pct=0.2
)

# Calculate position size
position_data = risk_manager.calculate_position_size(
    account_balance=10000,
    entry_price=50000,
    stop_loss=48000,
    atr=1500,
    correlation_to_btc=0.8
)

print(f"Position Size: {position_data['position_size']}")
print(f"Adjusted Risk: {position_data['risk_percentage']}%")
```

### 5. Walk-Forward Analysis (`walk_forward.py`)

Out-of-sample testing to prevent overfitting:
- Train on in-sample data, validate on out-of-sample
- Multiple window approaches (expanding, rolling)
- Consistency and robustness metrics

```python
from walk_forward import walk_forward_backtest

# Run walk-forward analysis
results = walk_forward_backtest(
    backtest_func=my_backtest_function,
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    train_size=0.7,
    window_type="expanding",
    num_windows=5
)
```

### 6. Strategy Optimizer (`strategy_optimizer.py`)

Find optimal parameters for your strategy:
- Grid search for exhaustive exploration
- Random search for efficiency
- Parameter impact visualization

```python
from strategy_optimizer import optimize_strategy

# Define parameter ranges
param_ranges = {
    "window": (100, 500),
    "risk_percentage": (0.5, 2.0)
}

# Run optimization
results = optimize_strategy(
    backtest_func=my_backtest_function,
    param_ranges=param_ranges,
    optimization_target="sharpe_ratio",
    search_method="random",
    num_trials=100
)

# Access best parameters
best_params = results["best_result"]["parameters"]
```

### 7. Strategy Comparison (`strategy_comparison.py`)

Compare multiple strategies:
- Side-by-side metric comparison
- Benchmark comparisons
- Equity curve visualization

```python
from strategy_comparison import compare_strategies

# Define strategies to compare
strategies = {
    "Conservative": {
        "func": strategy_function,
        "parameters": {"risk_percentage": 0.5}
    },
    "Aggressive": {
        "func": strategy_function,
        "parameters": {"risk_percentage": 2.0}
    }
}

# Run comparison
results = compare_strategies(
    strategies=strategies,
    benchmark_func=benchmark_function,
    benchmark_params={"risk_percentage": 0}
)
```

### 8. Correlation Analysis (`correlation_analysis.py`)

Analyze strategy relationship to market:
- Correlation with market benchmarks
- Performance in different market regimes
- Market condition dependencies

```python
from correlation_analysis import analyze_correlations

# Run analysis
results = analyze_correlations(
    strategy_trades=trades,
    price_data={"BTC": btc_df, "ETH": eth_df},
    initial_balance=10000.0
)

# View correlations
print(results["correlations"])
```

### 9. Enhanced Backtester (`enhanced_backtester.py`)

Integrated interface to all features:
- Unified configuration
- Comprehensive reporting
- Streamlined workflow

```python
from enhanced_backtester import EnhancedBacktester

# Create backtester
backtester = EnhancedBacktester()

# Run comprehensive backtest
summary = backtester.generate_comprehensive_report(
    symbol="BTCUSDT",
    interval="1h",
    candles=1000,
    window=300,
    risk_percentage=1.0
)

print(f"Final Balance: ${summary['final_balance']:.2f}")
print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
```

## Quick Start

See the `examples.py` file for complete usage examples of all features.

```bash
# Run all examples
python examples.py

# Explore generated reports
cd reports/examples
```

## Best Practices

1. **Always account for transaction costs** - Real-world trading includes fees and slippage
2. **Validate with walk-forward analysis** - Prevent overfitting by testing on unseen data
3. **Use Monte Carlo to assess robustness** - Understand the range of possible outcomes
4. **Compare to benchmarks** - Ensure your strategy outperforms simple alternatives
5. **Optimize carefully** - Be aware of the risk of curve-fitting to historical data
6. **Adapt to market conditions** - Different strategies work in different market regimes
7. **Monitor correlations** - Understand how your strategy relates to market movements
8. **Use adaptive position sizing** - Adjust risk based on volatility and confidence
9. **Track drawdowns carefully** - Maximum drawdown is often more important than returns
10. **Test over multiple time periods** - Market conditions change over time

## Next Steps for Improvement

1. Implement machine learning for signal enhancement
2. Add cross-validation for more robust parameter testing
3. Implement portfolio-level backtesting for multiple assets
4. Add regime switching capabilities based on market conditions
5. Develop adaptive strategy selection based on market regime
