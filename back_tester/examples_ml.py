#!/usr/bin/env python3
"""
Machine Learning Signal Enhancement Examples

This file demonstrates how to use the ML signal enhancement module
to improve trading signals and backtest performance.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from back_tester.enhanced_backtester import EnhancedBacktester, run_enhanced_backtest
from back_tester.ml_signal_enhancement import MLSignalEnhancer
from back_tester.performance_metrics import calculate_performance_metrics, plot_equity_curve
from back_tester.strategy_comparison import compare_strategies


def example_ml_feature_engineering():
    """Demonstrate ML feature engineering on cryptocurrency data"""
    print("\n=== Example 1: ML Feature Engineering ===")
    
    # Initialize the backtester to get some data
    backtester = EnhancedBacktester(output_dir="./reports/ml_examples")
    
    # Fetch data
    symbol = "BTCUSDT"
    interval = "1h"
    candles = 1000
    
    print(f"Fetching {candles} candles of {symbol} {interval} data...")
    
    # Get data from the backtester
    data = backtester._fetch_data(symbol, interval, candles)
    
    # Initialize ML enhancer
    ml_enhancer = MLSignalEnhancer(output_dir="./reports/ml_examples/models")
    
    # Generate features
    print("Generating ML features...")
    features_df = ml_enhancer.generate_features(data)
    
    # Display feature information
    print(f"\nGenerated {len(features_df.columns)} features from price data")
    print(f"Example features: {list(features_df.columns)[5:10]}")
    print(f"Data shape: {features_df.shape}")
    
    # Create a correlation heatmap of key features
    key_features = ['close', 'ma_20', 'ema_20', 'rsi_14', 'macd', 'bb_width_20', 'volume_ratio']
    key_features = [col for col in key_features if col in features_df.columns]
    
    corr = features_df[key_features].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    plt.matshow(corr, fignum=1)
    plt.xticks(range(len(key_features)), key_features, rotation=45, ha='left')
    plt.yticks(range(len(key_features)), key_features)
    plt.colorbar()
    
    # Save the correlation heatmap
    os.makedirs("./reports/ml_examples", exist_ok=True)
    plt.savefig("./reports/ml_examples/feature_correlation.png")
    plt.close()
    
    print("Feature correlation heatmap saved to: ./reports/ml_examples/feature_correlation.png")


def example_ml_model_training():
    """Train an ML model for signal prediction"""
    print("\n=== Example 2: ML Model Training ===")
    
    # Initialize the backtester to get some data
    backtester = EnhancedBacktester(output_dir="./reports/ml_examples")
    
    # Fetch data
    symbol = "BTCUSDT"
    interval = "1h"
    candles = 5000  # Need more data for proper training
    
    print(f"Fetching {candles} candles of {symbol} {interval} data...")
    
    # Get data from the backtester
    data = backtester._fetch_data(symbol, interval, candles)
    
    # Initialize ML enhancer
    ml_enhancer = MLSignalEnhancer(
        model_type="random_forest",
        feature_selection="kbest",
        n_features=20,
        output_dir="./reports/ml_examples/models"
    )
    
    # Train model with time series cross-validation
    print("\nTraining ML model with time series cross-validation...")
    training_results = ml_enhancer.train(
        data=data,
        prediction_horizon=12,  # Predict price movement 12 hours ahead
        threshold=0.01,  # Signal for 1% or more price increase
        use_time_series_split=True,
        n_splits=5
    )
    
    # Display training results
    print("\nTraining Results:")
    print(f"Accuracy: {training_results['accuracy']:.4f} (±{training_results.get('accuracy_std', 0):.4f})")
    print(f"Precision: {training_results['precision']:.4f} (±{training_results.get('precision_std', 0):.4f})")
    print(f"Recall: {training_results['recall']:.4f} (±{training_results.get('recall_std', 0):.4f})")
    print(f"F1 Score: {training_results['f1']:.4f} (±{training_results.get('f1_std', 0):.4f})")
    
    # Save model
    model_path = ml_enhancer.save_model(f"{symbol}_{interval}_ml_model")
    print(f"\nModel saved to: {model_path}")
    
    # Display feature importance if available
    if 'feature_importance' in training_results:
        # Sort features by importance
        importance = training_results['feature_importance']
        sorted_importance = {k: v for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}
        
        # Print top 10 features
        print("\nTop 10 Important Features:")
        for i, (feature, importance) in enumerate(list(sorted_importance.items())[:10]):
            print(f"{i+1}. {feature}: {importance:.4f}")
    
    return ml_enhancer


def example_ml_signal_enhancement(ml_enhancer=None):
    """Enhance trading signals using a trained ML model"""
    print("\n=== Example 3: ML Signal Enhancement ===")
    
    # Initialize the backtester to get some data
    backtester = EnhancedBacktester(output_dir="./reports/ml_examples")
    
    # Fetch data
    symbol = "BTCUSDT"
    interval = "1h"
    candles = 1000  # Test on a different dataset
    
    print(f"Fetching {candles} candles of {symbol} {interval} data...")
    
    # Get data from the backtester
    data = backtester._fetch_data(symbol, interval, candles)
    
    # Load or use provided ML enhancer
    if ml_enhancer is None or not ml_enhancer.is_trained:
        model_path = f"./reports/ml_examples/models/{symbol}_{interval}_ml_model.joblib"
        
        if os.path.exists(model_path):
            print(f"\nLoading existing model from {model_path}")
            ml_enhancer = MLSignalEnhancer(output_dir="./reports/ml_examples/models")
            ml_enhancer.load_model(model_path)
        else:
            print("\nNo existing model found. Training a new model...")
            ml_enhancer = example_ml_model_training()
    
    # Generate basic signals (here we use a simple moving average crossover for demonstration)
    print("\nGenerating basic trading signals...")
    data['ma_fast'] = data['close'].rolling(window=20).mean()
    data['ma_slow'] = data['close'].rolling(window=50).mean()
    data = data.dropna()
    
    # Generate basic signals: 1 for buy, -1 for sell, 0 for hold
    basic_signals = pd.Series(0, index=data.index)
    basic_signals[data['ma_fast'] > data['ma_slow']] = 1  # Buy signal
    basic_signals[data['ma_fast'] < data['ma_slow']] = -1  # Sell signal
    
    # Enhance signals with ML
    print("Enhancing signals with ML predictions...")
    enhanced_signals = ml_enhancer.enhance_signals(
        data=data,
        signals=basic_signals,
        threshold=0.7  # High confidence threshold
    )
    
    # Compare signal counts
    basic_buy_signals = (basic_signals == 1).sum()
    enhanced_buy_signals = (enhanced_signals == 1).sum()
    
    print(f"\nBasic buy signals: {basic_buy_signals}")
    print(f"Enhanced buy signals: {enhanced_buy_signals}")
    print(f"Change in signal count: {enhanced_buy_signals - basic_buy_signals}")
    
    # Visualize a section of signals
    sample_size = min(100, len(data))
    plt.figure(figsize=(15, 10))
    
    # Plot price
    plt.subplot(2, 1, 1)
    plt.plot(data['close'].tail(sample_size), label='Close Price')
    plt.title(f"{symbol} {interval} Price Chart with Signals")
    plt.grid(True)
    
    # Plot signals
    plt.subplot(2, 1, 2)
    plt.plot(basic_signals.tail(sample_size), label='Basic Signals', marker='o', alpha=0.6)
    plt.plot(enhanced_signals.tail(sample_size), label='Enhanced Signals', marker='x')
    plt.legend()
    plt.grid(True)
    plt.title("Trading Signals Comparison")
    
    # Save visualization
    plt.tight_layout()
    plt.savefig("./reports/ml_examples/signal_comparison.png")
    plt.close()
    
    print("\nSignal comparison visualization saved to: ./reports/ml_examples/signal_comparison.png")
    
    return data, basic_signals, enhanced_signals


def example_ml_enhanced_backtest(ml_enhancer=None):
    """Run a backtest with ML enhanced signals"""
    print("\n=== Example 4: ML Enhanced Backtest ===")
    
    # Get data and signals from the previous example
    data, basic_signals, enhanced_signals = example_ml_signal_enhancement(ml_enhancer)
    
    # Initialize backtester
    backtester = EnhancedBacktester(output_dir="./reports/ml_examples")
    
    # Define strategies to compare
    strategies = {
        "Original Strategy": {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "candles": 1000,
            "window": 300,
            "risk_percentage": 1.0,
            "initial_balance": 10000.0
        },
        "ML Enhanced Strategy": {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "candles": 1000,
            "window": 300,
            "risk_percentage": 1.0,
            "initial_balance": 10000.0,
            "use_ml_signals": True  # This would need to be handled in your backtesting code
        }
    }
    
    # At this point, you would need to modify your backtesting code to accept ML signals
    # For demonstration purposes, we'll just print how this would be set up
    
    print("\nSetting up ML-enhanced backtest comparison...")
    print("This would compare the original strategy against an ML-enhanced version")
    print("Note: To fully implement this, you would need to modify your backtester to:")
    print("  1. Accept pre-computed ML signals")
    print("  2. Use these signals to override or enhance the built-in signal generation")
    print("  3. Run comparable backtests on the same data")
    
    # Ideally, you would implement this in your backtester and then run:
    # results = backtester.compare_strategies(strategies=strategies, initial_balance=10000.0)


def main():
    """Run all ML examples"""
    # Create reports directory
    os.makedirs("./reports/ml_examples/models", exist_ok=True)
    
    # Run examples with better error isolation
    run_example("ML Feature Engineering", example_ml_feature_engineering)
    ml_enhancer = run_example("ML Model Training", example_ml_model_training)
    run_example("ML Signal Enhancement", lambda: example_ml_signal_enhancement(ml_enhancer))
    run_example("ML Enhanced Backtest", lambda: example_ml_enhanced_backtest(ml_enhancer))
    
    print("\nML Examples run complete!")
    print("Check the ./reports/ml_examples directory for detailed reports.")


def run_example(name, func):
    """Run an example function with error handling"""
    print(f"\n--- Running {name} Example ---")
    result = None
    try:
        result = func()
        print(f"✅ {name} example completed successfully")
    except Exception as e:
        import traceback
        print(f"❌ Error in {name} example: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
    return result


if __name__ == "__main__":
    main()
