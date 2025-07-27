"""
Core utility functions for the CryptoBot application.
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Create console handler for logs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)

# Prevent propagation to root logger
logger.propagate = False

# Time intervals mapping
VALID_INTERVALS = {
    "1m": "1",    # 1 minute
    "5m": "5",    # 5 minutes
    "15m": "15",  # 15 minutes
    "30m": "30",  # 30 minutes
    "1h": "60",   # 1 hour
    "4h": "240",  # 4 hours
    "1d": "D",    # 1 day
    "1w": "W",    # 1 week
}

# API endpoint constants
API_URL = "https://api.bybit.com/v5/market/kline"

# Global storage for user preferences and jobs
user_selected_indicators = {}
auto_signal_jobs = {}


def plural_helper(num: int) -> str:
    """
    Helper function to determine plural form based on number.
    
    Args:
        num: The number to check
        
    Returns:
        str: "s" for plural, "" for singular
    """
    if num != 1:
        return "s"
    return ""


def create_true_preferences():
    """
    Create a default preferences dictionary with all indicators enabled.
    
    Returns:
        dict: Default preferences dictionary
    """
    preferences = {}
    preferences["order_blocks"] = True
    preferences["fvgs"] = True
    preferences["liquidity_levels"] = True
    preferences["breaker_blocks"] = True
    preferences["liquidity_pools"] = True
    return preferences


def is_bullish(candle):
    """
    Detects whether the candle is bullish.
    
    Args:
        candle: Dictionary or pandas Series with Open and Close prices
        
    Returns:
        bool: True if Close > Open (bullish candle)
    """
    return candle["Close"] > candle["Open"]


def is_bearish(candle):
    """
    Detects whether the candle is bearish.
    
    Args:
        candle: Dictionary or pandas Series with Open and Close prices
        
    Returns:
        bool: True if Close < Open (bearish candle)
    """
    return candle["Close"] < candle["Open"]
