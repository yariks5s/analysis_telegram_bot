"""
Enhanced Backtesting System for Cryptocurrency Trading

This package provides comprehensive tools for strategy development,
testing, optimization, and self-learning for cryptocurrency trading.

New in v1.1.0:
- Adaptive Learning System: Automatically adjusts signal weights based on performance
- Signal Performance Tracking: Tracks which indicators contribute to winning/losing trades
- Enhanced Metrics: Detailed breakdown by indicator, market regime, and symbol
"""

__version__ = "1.1.0"

# Core modules
from .strategy import backtest_strategy
from .performance_metrics import calculate_performance_metrics, generate_performance_report

# Enhanced features
from .enhanced_backtester import EnhancedBacktester, run_enhanced_backtest

# Self-learning modules
try:
    from .adaptive_learning import (
        SelfLearningBacktester,
        SignalPerformanceTracker,
        AdaptiveWeightAdjuster,
        LearningMetricsAnalyzer,
        SignalRecord,
        SignalOutcome,
        extract_indicator_contributions,
    )
    from .enhanced_metrics import (
        EnhancedMetricsCalculator,
        TradeMetrics,
        create_trade_metrics,
    )
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False

__all__ = [
    # Core
    "backtest_strategy",
    "calculate_performance_metrics",
    "generate_performance_report",
    "EnhancedBacktester",
    "run_enhanced_backtest",
    # Learning (if available)
    "SelfLearningBacktester",
    "SignalPerformanceTracker",
    "AdaptiveWeightAdjuster",
    "EnhancedMetricsCalculator",
    "LEARNING_AVAILABLE",
]
