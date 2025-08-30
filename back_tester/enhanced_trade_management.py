"""
Enhanced trade management module for the backtester.

This module provides advanced trade management functionality:
- Trailing stops
- Time-based exits
- Volatility-based take profits
"""

import numpy as np
import pandas as pd
from typing import Tuple

def calculate_trailing_stop(
    entry_price: float,
    current_price: float,
    initial_stop: float,
    trail_percent: float = 0.5,
    activation_threshold: float = 1.0
) -> float:
    """
    Calculate a trailing stop price.
    
    Args:
        entry_price: The entry price of the trade
        current_price: The current price of the asset
        initial_stop: The initial stop loss price
        trail_percent: What percentage of profits to protect (0-1)
        activation_threshold: How far price needs to move in profit (R multiple) to activate trailing
        
    Returns:
        The updated stop loss price
    """
    initial_risk = abs(entry_price - initial_stop)
    profit_distance = current_price - entry_price
    risk_multiple = profit_distance / initial_risk if initial_risk > 0 else 0
    
    # Only activate trailing stop if price has moved **up** sufficiently in profit
    if risk_multiple >= activation_threshold:
        risk_to_protect = profit_distance * trail_percent
        new_stop = entry_price + risk_to_protect
        
        return max(new_stop, initial_stop)
    else:
        return initial_stop
        
def calculate_atr_based_stops(
    df: pd.DataFrame,
    multiplier: float = 1.5,
    period: int = 14
) -> pd.Series:
    """
    Calculate ATR-based stop loss values.
    
    Args:
        df: DataFrame with OHLC data
        multiplier: ATR multiplier for stop distance
        period: Period for ATR calculation
        
    Returns:
        Series of stop loss values
    """
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    # Fill NaN values with a reasonable default (1% of price)
    if atr.isna().all():
        atr = df["Close"] * 0.01
    else:
        atr = atr.bfill().ffill()
        if atr.isna().any():
            atr = atr.fillna(df["Close"] * 0.01)
    
    # Calculate stop loss distances based on ATR
    return atr * multiplier

def calculate_volatility_based_take_profits(*args, **kwargs) -> Tuple[float, float, float]:
    """
    Calculate volatility-adjusted take profit levels.
    
    Function supports two calling methods for backward compatibility:
    
    Method 1 (recommended):
        entry_price: The entry price of the trade
        stop_loss: The stop loss price
        atr: The current ATR value
        signal_type: "Bullish" or "Bearish"
        risk_reward_ratios: Base risk-reward ratios for the three targets (tuple of 3 floats)
    
    Method 2 (legacy):
        df: DataFrame with OHLC data
        i: Current index in dataframe
        current_price: Current price
        signal: "Bullish" or "Bearish"
        atr_period: Period for ATR calculation
        tp_multipliers: List of TP multipliers
        min/max_distance_percent: Min/max TP distance
        
    Returns:
        Tuple of three take profit levels
    """
    # Check which calling method is being used
    if len(args) > 0 and isinstance(args[0], pd.DataFrame):
        # Legacy method with dataframe
        df = args[0]
        i = args[1]
        current_price = args[2]
        signal_type = args[3]
        
        # Extract parameters from kwargs
        atr_period = kwargs.get('atr_period', 14)
        tp_multipliers = kwargs.get('tp_multipliers', [2.0, 3.5, 5.0])
        
        # Calculate ATR for volatility
        atr_values = calculate_atr_based_stops(df.iloc[:i+1], multiplier=1.0, period=atr_period)
        atr = atr_values.iloc[-1] if not atr_values.empty else current_price * 0.01
        
        # Calculate stop loss if not provided
        if "stop_loss" in kwargs:
            stop_loss = kwargs["stop_loss"]
        else:
            stop_distance = atr * 1.5  # Default multiplier
            if signal_type == "Bullish":
                stop_loss = current_price * 0.95  # Default 5% below entry
            else:
                stop_loss = current_price * 1.05  # Default 5% above entry
        
        entry_price = current_price
        risk_reward_ratios = tuple(tp_multipliers)
        
    else:
        # New method with direct parameters
        entry_price = kwargs.get('entry_price', args[0])
        stop_loss = kwargs.get('stop_loss', args[1])
        atr = kwargs.get('atr', args[2])
        signal_type = kwargs.get('signal_type', args[3])
        risk_reward_ratios = kwargs.get('risk_reward_ratios', (1.5, 2.5, 3.5))
        if len(args) > 4:
            risk_reward_ratios = args[4]
    
    # Calculate initial risk
    risk = abs(entry_price - stop_loss)
    
    # Adjust risk-reward ratio based on volatility
    volatility_factor = atr / (entry_price * 0.01)  # Normalize to 1% of price
    adjusted_rr = [
        max(rr - (0.2 * (volatility_factor - 1)), rr * 0.7)
        if volatility_factor > 1 else rr
        for rr in risk_reward_ratios
    ]
    
    # Calculate take profit levels
    if signal_type == "Bullish":
        tp1 = entry_price + (risk * adjusted_rr[0])
        tp2 = entry_price + (risk * adjusted_rr[1])
        tp3 = entry_price + (risk * adjusted_rr[2])
    else:
        tp1 = entry_price - (risk * adjusted_rr[0])
        tp2 = entry_price - (risk * adjusted_rr[1])
        tp3 = entry_price - (risk * adjusted_rr[2])
        
    return tp1, tp2, tp3

def should_exit_based_on_time(
    entry_index: int,
    current_index: int,
    max_trade_duration: int = 24
) -> bool:
    """
    Determine if a trade should be exited based on time duration.
    
    Args:
        entry_index: The candle index when the trade was entered
        current_index: The current candle index
        max_trade_duration: Maximum duration to hold a trade in candles
        
    Returns:
        Boolean indicating whether to exit the trade
    """
    trade_duration = current_index - entry_index
    return trade_duration >= max_trade_duration
