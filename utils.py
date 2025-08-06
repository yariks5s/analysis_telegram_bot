"""
Adapter module to maintain compatibility with back_tester modules.
This file imports functions from src.core.utils and exports them
under names expected by the backtesting system.
"""

from src.core.utils import create_true_preferences

# Re-export the imported functions to maintain interface
__all__ = ["create_true_preferences"]
