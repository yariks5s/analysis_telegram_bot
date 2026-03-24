#!/usr/bin/env python3
"""
Self-Learning Backtesting Examples

This script demonstrates how to use the new self-learning capabilities
of the backtesting system:

1. Basic self-learning backtest
2. Running multiple backtests to train the system
3. Viewing performance analytics
4. Triggering weight adjustments manually
5. Understanding which indicators work best
"""

import os
import sys
from datetime import datetime

# Add project root to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from back_tester.strategy import backtest_strategy
from back_tester.adaptive_learning import (
    SelfLearningBacktester,
    SignalPerformanceTracker,
    AdaptiveWeightAdjuster,
    LearningMetricsAnalyzer,
)

# Weight names for reference
WEIGHT_NAMES = SignalPerformanceTracker.WEIGHT_NAMES

try:
    from back_tester.enhanced_metrics import EnhancedMetricsCalculator
except ImportError:
    EnhancedMetricsCalculator = None


def example_1_basic_self_learning_backtest():
    """
    Example 1: Run a basic backtest with self-learning enabled.
    
    The system will:
    - Track each signal and its contributing indicators
    - Record whether signals were successful or not
    - Automatically adjust weights based on performance
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Self-Learning Backtest")
    print("="*70)
    
    # Create the self-learning backtester
    learner = SelfLearningBacktester(
        storage_path="./data/learning",
        auto_adjust_weights=True,
        adjustment_frequency=50  # Adjust weights every 50 signals
    )
    
    # Get current adaptive weights
    weights = learner.get_current_weights()
    print(f"\nCurrent adaptive weights: {weights}")
    
    # Run backtest with learning enabled
    final_balance, trades, _ = backtest_strategy(
        symbol="BTCUSDT",
        interval="1h",
        candles=500,
        window=200,
        initial_balance=10000.0,
        risk_percentage=1.0,
        enable_learning=True,
        learner=learner,
    )
    
    print(f"\nBacktest Results:")
    print(f"  Final Balance: ${final_balance:.2f}")
    print(f"  Total Trades: {len([t for t in trades if t['type'] == 'entry'])}")
    
    # View learning status
    learner.print_status()
    
    return learner


def example_2_multiple_backtests_training():
    """
    Example 2: Run multiple backtests to train the system.
    
    The more backtests you run, the better the system learns
    which indicators work in different market conditions.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Training with Multiple Backtests")
    print("="*70)
    
    # Create the self-learning backtester
    learner = SelfLearningBacktester(
        storage_path="./data/learning_training",
        auto_adjust_weights=True,
        adjustment_frequency=20  # More aggressive adjustment
    )
    
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    intervals = ["1h", "4h"]
    
    total_trades = 0
    total_profit = 0
    
    print("\nRunning multiple backtests for training...")
    
    for symbol in symbols:
        for interval in intervals:
            try:
                final_balance, trades, _ = backtest_strategy(
                    symbol=symbol,
                    interval=interval,
                    candles=400,
                    window=150,
                    initial_balance=10000.0,
                    risk_percentage=1.0,
                    enable_learning=True,
                    learner=learner,
                )
                
                entry_trades = [t for t in trades if t['type'] == 'entry']
                profit = final_balance - 10000.0
                
                print(f"  {symbol} {interval}: {len(entry_trades)} trades, "
                      f"${profit:.2f} profit")
                
                total_trades += len(entry_trades)
                total_profit += profit
                
            except Exception as e:
                print(f"  {symbol} {interval}: Error - {e}")
    
    print(f"\nTraining Summary:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Total Profit: ${total_profit:.2f}")
    
    # Force learning update
    learner.force_learning()
    
    # View updated weights
    new_weights = learner.get_current_weights()
    print(f"\nUpdated weights after training: {new_weights}")
    
    # View detailed report
    learner.print_status()
    
    return learner


def example_3_analyze_indicator_performance():
    """
    Example 3: Analyze which indicators perform best.
    
    After running backtests, you can see detailed analytics
    on which indicators contribute to winning trades.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Indicator Performance Analysis")
    print("="*70)
    
    # Create tracker with existing data
    tracker = SignalPerformanceTracker(storage_path="./data/learning/signals")
    
    # Get indicator performance breakdown
    indicator_perf = tracker.get_indicator_performance()
    
    if indicator_perf.empty:
        print("\nNo signal data available yet. Run some backtests first!")
        return
    
    print("\n📊 Indicator Performance Ranking:")
    print("-" * 60)
    print(f"{'Indicator':<25} {'Signals':>8} {'Win Rate':>10} {'Avg P/L':>10}")
    print("-" * 60)
    
    for _, row in indicator_perf.iterrows():
        print(f"{row['indicator']:<25} {row['signals']:>8} "
              f"{row['win_rate']:>9.1f}% ${row['avg_pnl']:>9.2f}")
    
    # Identify weak and strong indicators
    weak = tracker.identify_weak_indicators()
    strong = tracker.identify_strong_indicators()
    
    if strong:
        print(f"\n🟢 Strong Indicators (>60% win rate): {', '.join(strong)}")
    
    if weak:
        print(f"🔴 Weak Indicators (<40% win rate): {', '.join(weak)}")
    
    # Get market regime analysis
    regime_perf = tracker.get_regime_performance()
    if not regime_perf.empty:
        print("\n📈 Performance by Market Regime:")
        print("-" * 40)
        for _, row in regime_perf.iterrows():
            print(f"  {row['regime']}: {row['win_rate']:.1f}% win rate, "
                  f"${row['total_pnl']:.2f} total")
    
    return tracker


def example_4_manual_weight_adjustment():
    """
    Example 4: Manually trigger weight adjustments.
    
    Sometimes you want to manually review and apply
    weight adjustments instead of automatic updates.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Manual Weight Adjustment")
    print("="*70)
    
    # Create tracker and adjuster
    tracker = SignalPerformanceTracker(storage_path="./data/learning/signals")
    adjuster = AdaptiveWeightAdjuster(
        performance_tracker=tracker,
        learning_rate=0.05,
        storage_path="./data/learning/weights"
    )
    
    # Get current weights
    current_weights = adjuster.current_weights
    print("\nCurrent Weights:")
    for name, value in current_weights.items():
        print(f"  {name}: {value:.3f}")
    
    # Calculate suggested adjustments
    adjustments = adjuster.calculate_weight_adjustments()
    
    if adjustments:
        print("\n📝 Suggested Adjustments:")
        for indicator, adjustment in sorted(adjustments.items(), 
                                            key=lambda x: abs(x[1]), reverse=True):
            direction = "↑" if adjustment > 0 else "↓"
            print(f"  {indicator}: {direction} {abs(adjustment):.4f}")
        
        # Apply adjustments
        confirm = input("\nApply these adjustments? (y/n): ").lower()
        if confirm == 'y':
            new_weights = adjuster.apply_adjustments(adjustments)
            print("\n✅ Weights updated!")
            
            # Show changes
            print("\nWeight Changes:")
            for name in new_weights:
                old = current_weights.get(name, 1.0)
                new = new_weights[name]
                if old != new:
                    print(f"  {name}: {old:.3f} → {new:.3f}")
    else:
        print("\nNo adjustments needed based on current data.")
    
    return adjuster


def example_5_enhanced_backtest_with_metrics():
    """
    Example 5: Run enhanced backtest with detailed metrics.
    
    This example shows how to use the EnhancedMetricsCalculator
    for detailed trade analysis.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Enhanced Metrics Analysis")
    print("="*70)
    
    # Create metrics calculator
    metrics_calc = EnhancedMetricsCalculator()
    
    # Create self-learning backtester
    learner = SelfLearningBacktester(
        storage_path="./data/learning_metrics",
        auto_adjust_weights=True
    )
    
    # Run backtest
    final_balance, trades, _ = backtest_strategy(
        symbol="BTCUSDT",
        interval="1h",
        candles=600,
        window=250,
        initial_balance=10000.0,
        risk_percentage=1.0,
        enable_learning=True,
        learner=learner,
        metrics_calculator=metrics_calc,
    )
    
    # Print detailed metrics report
    if trades:
        metrics_calc.print_detailed_report(10000.0, final_balance)
    else:
        print("\nNo trades executed in this backtest.")
    
    return metrics_calc


def example_6_continuous_learning_loop():
    """
    Example 6: Continuous learning loop.
    
    This example shows how to set up a continuous learning loop
    that runs backtests, learns from results, and improves over time.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Continuous Learning Loop")
    print("="*70)
    
    # Create the self-learning backtester
    learner = SelfLearningBacktester(
        storage_path="./data/learning_continuous",
        auto_adjust_weights=True,
        adjustment_frequency=30
    )
    
    symbols = ["BTCUSDT", "ETHUSDT"]
    iterations = 3
    
    print(f"\nRunning {iterations} learning iterations...")
    
    for iteration in range(1, iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        
        # Get current weights (which adapt over time)
        weights = learner.get_current_weights()
        print(f"Using weights: {dict(zip(SignalPerformanceTracker.WEIGHT_NAMES[:5], weights[:5]))}...")
        
        iteration_profit = 0
        iteration_trades = 0
        
        for symbol in symbols:
            try:
                final_balance, trades, _ = backtest_strategy(
                    symbol=symbol,
                    interval="1h",
                    candles=300,
                    window=100,
                    initial_balance=10000.0,
                    risk_percentage=1.0,
                    enable_learning=True,
                    learner=learner,
                    weights=weights,
                )
                
                entry_trades = [t for t in trades if t['type'] == 'entry']
                profit = final_balance - 10000.0
                
                iteration_profit += profit
                iteration_trades += len(entry_trades)
                
            except Exception as e:
                print(f"  {symbol}: Error - {e}")
        
        print(f"  Iteration {iteration} results: {iteration_trades} trades, "
              f"${iteration_profit:.2f} profit")
        
        # Get performance after each iteration
        recent_perf = learner.tracker.get_recent_performance(50)
        print(f"  Recent win rate: {recent_perf.get('win_rate', 0):.1f}%")
    
    # Final report
    print("\n" + "="*50)
    print("FINAL LEARNING REPORT")
    print("="*50)
    learner.print_status()
    
    return learner


if __name__ == "__main__":
    print("\n🔄 Self-Learning Backtesting System Examples")
    print("=" * 70)
    print("\nAvailable examples:")
    print("  1. Basic self-learning backtest")
    print("  2. Training with multiple backtests")
    print("  3. Analyze indicator performance")
    print("  4. Manual weight adjustment")
    print("  5. Enhanced metrics analysis")
    print("  6. Continuous learning loop")
    print("  0. Run all examples")
    
    choice = input("\nSelect example (0-6): ").strip()
    
    if choice == "1":
        example_1_basic_self_learning_backtest()
    elif choice == "2":
        example_2_multiple_backtests_training()
    elif choice == "3":
        example_3_analyze_indicator_performance()
    elif choice == "4":
        example_4_manual_weight_adjustment()
    elif choice == "5":
        example_5_enhanced_backtest_with_metrics()
    elif choice == "6":
        example_6_continuous_learning_loop()
    elif choice == "0":
        # Run all examples
        example_1_basic_self_learning_backtest()
        example_2_multiple_backtests_training()
        example_3_analyze_indicator_performance()
        example_5_enhanced_backtest_with_metrics()
        example_6_continuous_learning_loop()
    else:
        print("Invalid choice. Running example 1...")
        example_1_basic_self_learning_backtest()

