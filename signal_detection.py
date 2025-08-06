"""
Adapter module to maintain compatibility with back_tester modules.
This file imports functions from src.telegram.signals.detection and exports them
under names expected by the backtesting system.
"""

from src.telegram.signals.detection import (
    TradingSignal,
    generate_price_prediction_signal_proba,
    calculate_position_size,
)

# Re-export the imported functions and classes to maintain interface
__all__ = [
    "TradingSignal",
    "generate_price_prediction_signal_proba",
    "calculate_position_size",
]
