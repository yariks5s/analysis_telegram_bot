"""
Market filters module for the backtester.

This module provides functionality to filter trading opportunities based on:
- Market regime analysis
- Time-of-day filtering
- Enhanced entry confirmation criteria
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union


def detect_market_regime(
    df: pd.DataFrame,
    lookback_period: int = 20,
    atr_period: int = 14,
    rsi_period: int = 14,
    atr_threshold: float = 1.5,
    rsi_overbought: float = 70,
    rsi_oversold: float = 30,
) -> Dict[str, Union[str, float]]:
    """
    Detect the current market regime based on price action and indicators.

    Args:
        df: Price data DataFrame with OHLC
        lookback_period: Period to analyze for regime detection
        atr_period: Period for ATR calculation
        rsi_period: Period for RSI calculation
        atr_threshold: Threshold for ATR ratio to detect volatility
        rsi_overbought: RSI threshold for overbought
        rsi_oversold: RSI threshold for oversold

    Returns:
        Dictionary with market regime details
    """
    if len(df) < lookback_period + atr_period:
        return {"regime": "unknown", "strength": 0.0, "details": "Not enough data"}

    # Use the last lookback_period candles for analysis
    analysis_window = df.iloc[-lookback_period:]

    # Calculate price direction
    start_price = analysis_window.iloc[0]["Close"]
    end_price = analysis_window.iloc[-1]["Close"]
    price_change = (end_price - start_price) / start_price

    # Calculate ATR for volatility
    high_low = analysis_window["High"] - analysis_window["Low"]
    high_close = np.abs(analysis_window["High"] - analysis_window["Close"].shift())
    low_close = np.abs(analysis_window["Low"] - analysis_window["Close"].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=atr_period).mean().fillna(true_range.mean())

    # Calculate average ATR as percentage of price
    avg_atr_pct = atr.mean() / analysis_window["Close"].mean()

    # Calculate historical ATR for comparison
    if len(df) > lookback_period + atr_period:
        historical_window = df.iloc[-(lookback_period + atr_period) : -lookback_period]
        historical_high_low = historical_window["High"] - historical_window["Low"]
        historical_high_close = np.abs(
            historical_window["High"] - historical_window["Close"].shift()
        )
        historical_low_close = np.abs(
            historical_window["Low"] - historical_window["Close"].shift()
        )

        historical_tr = pd.concat(
            [historical_high_low, historical_high_close, historical_low_close], axis=1
        ).max(axis=1)
        historical_atr = historical_tr.mean()
        historical_atr_pct = historical_atr / historical_window["Close"].mean()

        atr_ratio = avg_atr_pct / historical_atr_pct if historical_atr_pct > 0 else 1.0
    else:
        atr_ratio = 1.0

    # Calculate RSI
    delta = analysis_window["Close"].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)

    avg_gain = gain.rolling(window=rsi_period).mean().fillna(0)
    avg_loss = loss.rolling(window=rsi_period).mean().fillna(0)

    rs = avg_gain / avg_loss.replace(0, 1e-9)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]

    # Calculate price range as percentage
    price_range = (
        analysis_window["High"].max() - analysis_window["Low"].min()
    ) / analysis_window["Low"].min()

    # Determine market regime
    if atr_ratio > atr_threshold:
        # Volatile market
        regime = "volatile"
        strength = min(atr_ratio / atr_threshold, 3.0) / 3.0  # Normalize to 0-1
    elif abs(price_change) > 0.05:  # 5% price change
        # Trending market
        if price_change > 0:
            regime = "trending_up"
        else:
            regime = "trending_down"
        strength = min(abs(price_change) / 0.05, 3.0) / 3.0  # Normalize to 0-1
    elif price_range < 0.02:  # Very tight range
        regime = "quiet"
        strength = 1.0 - (price_range / 0.02)  # Higher strength for tighter ranges
    else:
        # Ranging market
        regime = "ranging"
        strength = min(price_range / 0.02, 2.0) / 2.0  # Normalize to 0-1

    return {
        "regime": regime,
        "strength": strength,
        "price_change": price_change,
        "atr_ratio": atr_ratio,
        "rsi": current_rsi,
        "price_range": price_range,
    }


def should_trade_in_regime(
    market_regime: Dict[str, Union[str, float]],
    signal_type: str,
    min_regime_strength: float = 0.6,
    allow_volatile: bool = False,
) -> Tuple[bool, str]:
    """
    Determine if we should trade in the current market regime.

    Args:
        market_regime: Dictionary with market regime details
        signal_type: 'Bullish' or 'Bearish'
        min_regime_strength: Minimum regime strength to consider valid
        allow_volatile: Whether to allow trading in volatile markets

    Returns:
        Tuple of (should_trade, reason)
    """
    regime = market_regime.get("regime", "unknown")
    strength = market_regime.get("strength", 0.0)
    rsi = market_regime.get("rsi", 50.0)

    # Check regime strength
    if strength < min_regime_strength and regime != "quiet":
        return False, f"Weak {regime} regime (strength: {strength:.2f})"

    # Check for regime and signal alignment
    if regime == "trending_up" and signal_type == "Bullish":
        return True, f"Strong uptrend (strength: {strength:.2f})"

    elif regime == "trending_down" and signal_type == "Bearish":
        return True, f"Strong downtrend (strength: {strength:.2f})"

    elif regime == "volatile":
        if not allow_volatile:
            return False, f"Avoiding volatile market (strength: {strength:.2f})"
        # If we do allow volatile markets, check for extremes
        if signal_type == "Bullish" and rsi < 30:
            return True, f"Volatile market, oversold condition (RSI: {rsi:.2f})"
        elif signal_type == "Bearish" and rsi > 70:
            return True, f"Volatile market, overbought condition (RSI: {rsi:.2f})"
        else:
            return False, f"Volatile market without extreme RSI condition"

    elif regime == "ranging":
        # In ranging markets, trade counter-trend
        if signal_type == "Bullish" and rsi < 40:
            return True, f"Ranging market, lower range (RSI: {rsi:.2f})"
        elif signal_type == "Bearish" and rsi > 60:
            return True, f"Ranging market, upper range (RSI: {rsi:.2f})"
        else:
            return False, f"Ranging market without favorable RSI position"

    elif regime == "quiet":
        return False, "Market too quiet, avoiding low volatility"

    return False, f"Regime {regime} not aligned with {signal_type} signal"


def is_favorable_trading_time(
    timestamp: pd.Timestamp, timezone: str = "UTC", favorable_hours: List[int] = None
) -> Tuple[bool, str]:
    """
    Determine if the current time is favorable for trading.

    Args:
        timestamp: The timestamp to check
        timezone: Timezone to convert timestamp to
        favorable_hours: List of hours (0-23) considered favorable for trading

    Returns:
        Tuple of (is_favorable, reason)
    """
    if favorable_hours is None:
        # Default to major market hours (rough approximation)
        favorable_hours = [1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22]

    # Convert timestamp to specified timezone
    local_time = timestamp.tz_localize("UTC").tz_convert(timezone)
    hour = local_time.hour
    day_of_week = local_time.dayofweek  # 0=Monday, 6=Sunday

    # Check if weekend
    if day_of_week >= 5:  # Saturday or Sunday
        return False, f"Weekend trading (day {day_of_week+1})"

    # Check hour
    if hour in favorable_hours:
        return True, f"Favorable trading hour ({hour}:00)"

    return False, f"Unfavorable trading hour ({hour}:00)"


def confirm_entry_criteria(
    df: pd.DataFrame,
    index: int,
    signal_type: str,
    confidence: float,
    minimum_confidence: float = 0.6,
) -> Tuple[bool, str]:
    """
    Apply enhanced entry confirmation criteria.

    Args:
        df: Price data DataFrame
        index: Current candle index
        signal_type: 'Bullish' or 'Bearish'
        confidence: Signal confidence value
        minimum_confidence: Minimum confidence required

    Returns:
        Tuple of (criteria_met, reason)
    """
    # Check basic confidence threshold
    if confidence < minimum_confidence:
        return False, f"Low confidence ({confidence:.2f} < {minimum_confidence:.2f})"

    # Make sure we have enough historical data
    if index < 20:
        return False, "Not enough historical data"

    # Recent price action
    recent_candles = df.iloc[index - 5 : index]

    # Check for trend confirmation in recent candles
    if signal_type == "Bullish":
        # Check if most recent close is above most recent open
        last_candle_bullish = df["Close"].iloc[index - 1] > df["Open"].iloc[index - 1]

        # Check if at least 3 of last 5 candles are bullish
        bullish_count = sum(recent_candles["Close"] > recent_candles["Open"])
        trend_confirmed = bullish_count >= 3 and last_candle_bullish

        if not trend_confirmed:
            return (
                False,
                f"Bullish signal not confirmed by recent price action ({bullish_count}/5 bullish candles)",
            )

    elif signal_type == "Bearish":
        # Check if most recent close is below most recent open
        last_candle_bearish = df["Close"].iloc[index - 1] < df["Open"].iloc[index - 1]

        # Check if at least 3 of last 5 candles are bearish
        bearish_count = sum(recent_candles["Close"] < recent_candles["Open"])
        trend_confirmed = bearish_count >= 3 and last_candle_bearish

        if not trend_confirmed:
            return (
                False,
                f"Bearish signal not confirmed by recent price action ({bearish_count}/5 bearish candles)",
            )

    # Volume confirmation
    # Check if volume is increasing
    recent_volume = (
        df["Volume"].iloc[index - 5 : index] if "Volume" in df.columns else None
    )

    if recent_volume is not None:
        avg_recent_volume = recent_volume.mean()
        avg_prior_volume = (
            df["Volume"].iloc[index - 10 : index - 5].mean()
            if index >= 10
            else avg_recent_volume
        )

        volume_increasing = avg_recent_volume > avg_prior_volume

        if not volume_increasing:
            return False, "Volume not confirming the signal"

    return True, "All entry criteria confirmed"
