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

# Self-learning modules (optional — graceful fallback if dependencies missing)
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
    "LEARNING_AVAILABLE",
]

if LEARNING_AVAILABLE:
    __all__ += [
        "SelfLearningBacktester",
        "SignalPerformanceTracker",
        "AdaptiveWeightAdjuster",
        "LearningMetricsAnalyzer",
        "EnhancedMetricsCalculator",
    ]
