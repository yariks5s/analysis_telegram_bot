"""
Enhanced risk management module for the backtester.

This module provides advanced risk management functionality:
- Dynamic ATR-based stop losses
- Adaptive position sizing
- Improved risk-reward calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, Union

def calculate_tighter_stop_loss(
    df: pd.DataFrame,
    index: int,
    entry_price: float,
    signal_type: str,
    atr_period: int = 14,
    baseline_multiplier: float = 1.5,
    min_distance_percent: float = 0.005,
    max_distance_percent: float = 0.02
) -> float:
    """
    Calculate a tighter, smarter stop loss based on ATR and market structure.
    
    Args:
        df: Price data DataFrame with OHLC
        index: Current candle index
        entry_price: Trade entry price
        signal_type: 'Bullish' or 'Bearish'
        atr_period: Period for ATR calculation
        baseline_multiplier: Base ATR multiplier
        min_distance_percent: Minimum distance for stop loss as percent of price
        max_distance_percent: Maximum distance for stop loss as percent of price
        
    Returns:
        Calculated stop loss price
    """
    if index < atr_period:
        # fallback if not enough data
        stop_distance = entry_price * max_distance_percent
        return entry_price - stop_distance if signal_type == "Bullish" else entry_price + stop_distance
    
    # ATR
    high_low = df["High"].iloc[index-atr_period:index] - df["Low"].iloc[index-atr_period:index]
    high_close = np.abs(df["High"].iloc[index-atr_period:index] - df["Close"].iloc[index-atr_period-1:index-1])
    low_close = np.abs(df["Low"].iloc[index-atr_period:index] - df["Close"].iloc[index-atr_period-1:index-1])
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.mean()
    
    # Adjust ATR multiplier based on recent volatility
    recent_volatility = df["Close"].iloc[index-10:index].std() / df["Close"].iloc[index-10:index].mean()
    volatility_factor = 1.0
    
    if recent_volatility > 0.015:  # High volatility
        volatility_factor = 1.2
    elif recent_volatility < 0.005:  # Low volatility
        volatility_factor = 0.8
    
    # Calculate stop loss distance with adjusted multiplier
    stop_distance = atr * baseline_multiplier * volatility_factor
    
    # Apply min/max constraints
    min_stop_distance = entry_price * min_distance_percent
    max_stop_distance = entry_price * max_distance_percent
    
    stop_distance = max(min_stop_distance, min(stop_distance, max_stop_distance))
    
    # Calculate stop loss price based on signal type
    if signal_type == "Bullish":
        stop_loss = entry_price - stop_distance
    else:
        stop_loss = entry_price + stop_distance
    
    return stop_loss

def calculate_dynamic_position_size(
    balance: float,
    risk_percentage: float,
    entry_price: float,
    stop_loss: float,
    market_volatility: float = 1.0
) -> Dict[str, Union[float, str]]:
    """
    Calculate position size with dynamic risk adjustment based on market conditions.
    
    Args:
        balance: Account balance
        risk_percentage: Base percentage of account to risk
        entry_price: Entry price for the trade
        stop_loss: Stop loss price
        market_volatility: Volatility factor (>1 means higher volatility)
        
    Returns:
        Dictionary with position size and risk details
    """
    # Adjust risk percentage based on market volatility
    adjusted_risk = risk_percentage
    
    if market_volatility > 1.5:  # High volatility
        adjusted_risk = risk_percentage * 0.7  # Reduce risk
    elif market_volatility < 0.7:  # Low volatility
        adjusted_risk = risk_percentage * 1.2  # Increase risk slightly
    
    # Ensure risk stays within reasonable bounds
    adjusted_risk = min(adjusted_risk, risk_percentage * 1.5)
    adjusted_risk = max(adjusted_risk, risk_percentage * 0.5)
    
    # Calculate dollar risk amount
    risk_amount = balance * (adjusted_risk / 100.0)
    
    # Calculate risk per unit
    risk_per_unit = abs(entry_price - stop_loss)
    
    if risk_per_unit <= 0:
        return {
            "position_size": 0.0,
            "risk_amount": 0.0,
            "risk_percentage": 0.0,
            "error": "Invalid risk: entry price and stop loss are too close"
        }
    
    # Calculate position size
    position_size = risk_amount / risk_per_unit
    
    return {
        "position_size": position_size,
        "risk_amount": risk_amount,
        "risk_percentage": adjusted_risk,
        "error": None
    }
