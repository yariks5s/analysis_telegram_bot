"""
Analysis utility helpers for CryptoBot.

This module contains various utility functions for technical analysis,
data validation, and processing market data.
"""

import pandas as pd  # type: ignore

from src.api.data_fetcher import fetch_ohlc_data, analyze_data, fetch_candles
from src.core.utils import VALID_INTERVALS
from src.database.operations import check_user_preferences, get_all_user_signal_requests


def calculate_macd(data: pd.DataFrame):
    """
    Calculate MACD and Signal Line.
    
    Args:
        data: DataFrame with Close prices
        
    Returns:
        Tuple of (MACD, Signal Line)
    """
    short_ema = data["Close"].ewm(span=12, adjust=False).mean()
    long_ema = data["Close"].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def calculate_rsi(data: pd.DataFrame, period: int = 14):
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        data: DataFrame with Close prices
        period: RSI calculation period
        
    Returns:
        Series of RSI values
    """
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


async def input_sanity_check_show(args, update) -> tuple:
    """
    Validate input arguments for chart display.
    
    Args:
        args: Command arguments
        update: Telegram update object
        
    Returns:
        Tuple of (symbol, hours, interval, liq_lev_tolerance) or empty tuple if invalid
    """
    # Default values
    symbol = "BTCUSDT"
    hours = 24
    interval = "1h"
    liq_lev_tolerance = 0

    if len(args) >= 1:
        symbol = args[0].upper()
    if len(args) >= 2:
        try:
            hours = int(args[1])
        except ValueError:
            await update.message.reply_text(
                "Invalid hours specified. Please provide a valid number."
            )
            return tuple()
    if len(args) >= 3:
        interval = args[2].lower()
        if interval not in VALID_INTERVALS:
            await update.message.reply_text(
                f"Invalid interval specified. Valid intervals are: {', '.join(VALID_INTERVALS.keys())}"
            )
            return tuple()
    if len(args) >= 4:
        try:
            liq_lev_tolerance = float(args[3])
            if liq_lev_tolerance < 0 or liq_lev_tolerance > 1:
                await update.message.reply_text(
                    f"Invalid liquidity level tolerance specified. It should be a number between 0 and 1"
                )
                return tuple()
        except ValueError:
            await update.message.reply_text(
                "Invalid tolerance value. Please provide a valid number."
            )
            return tuple()

    return (symbol, hours, interval, liq_lev_tolerance)


async def input_sanity_check_analyzing(text) -> dict:
    """
    Validate input for signal creation or deletion.
    
    Args:
        text: Command text to analyze
        
    Returns:
        Dict with validation results and parsed values
    """
    parts = text.strip().upper().split()
    result = {
        "is_valid": False,
        "error_message": "",
        "symbol": "",
        "frequency": 0,
        "is_with_chart": False
    }
    
    # Check basic format
    if len(parts) < 2:
        result["error_message"] = "Missing required arguments (format: SYMBOL MINUTES [with_chart])"
        return result
        
    # Parse symbol
    result["symbol"] = parts[0]
    
    # Parse frequency
    try:
        result["frequency"] = int(parts[1])
        if result["frequency"] <= 0:
            result["error_message"] = "Frequency must be a positive number"
            return result
    except ValueError:
        result["error_message"] = "Frequency must be a valid number"
        return result
        
    # Check for chart option
    if len(parts) >= 3 and "WITH_CHART" in " ".join(parts[2:]):
        result["is_with_chart"] = True
        
    result["is_valid"] = True
    return result


async def input_sanity_check_historical(args, update) -> tuple:
    """
    Parse and validate arguments for historical data command.
    Usage: /history <symbol> <length> <interval> <tolerance> <timestamp>
    Timestamp must be Unix epoch in seconds.
    
    Args:
        args: Command arguments
        update: Telegram update object
        
    Returns:
        Tuple of validated arguments or empty tuple if invalid
    """
    if len(args) < 5:
        await update.message.reply_text(
            "Usage: /history <symbol> <length> <interval> <tolerance> <timestamp>"
        )
        return tuple()
    if len(args) > 5:
        await update.message.reply_text(
            "❌ Invalid number of arguments. Please provide exactly 5 arguments."
        )
        return tuple()
    symbol = args[0].upper()
    try:
        length = int(args[1])
    except ValueError:
        await update.message.reply_text("❌ Invalid length. Must be an integer.")
        return tuple()
    interval = args[2].lower()
    if interval not in VALID_INTERVALS:
        await update.message.reply_text(
            f"❌ Invalid interval. Valid intervals: {', '.join(VALID_INTERVALS.keys())}"
        )
        return tuple()
    try:
        tolerance = float(args[3])
    except ValueError:
        await update.message.reply_text(
            "❌ Invalid tolerance. Must be a number between 0 and 1."
        )
        return tuple()
    if tolerance < 0 or tolerance > 1:
        await update.message.reply_text(
            "❌ Invalid tolerance. Must be between 0 and 1."
        )
        return tuple()
    try:
        timestamp_sec = int(args[4])
    except ValueError:
        await update.message.reply_text(
            "❌ Invalid timestamp. Must be Unix epoch seconds."
        )
        return tuple()
    return (symbol, length, interval, tolerance, timestamp_sec)


async def fetch_data_and_get_indicators(symbol, hours, interval, preferences, liq_lev_tolerance=0, update=None):
    """
    Fetch market data and calculate indicators.
    
    Args:
        symbol: Trading pair symbol
        hours: Number of hours/periods to fetch
        interval: Time interval
        preferences: User preferences for indicators
        liq_lev_tolerance: Tolerance for liquidity level detection
        update: Optional Telegram update object for notifications
        
    Returns:
        Tuple of (indicators, dataframe)
    """
    if update:
        await update.message.reply_text(
            f"Fetching {symbol} price data for the last {hours} periods with interval {interval}, please wait..."
        )

    df = None
    if hours <= 200:
        df = fetch_ohlc_data(symbol, hours, interval)
    else:
        df = fetch_candles(symbol, hours, interval)
        
    if (df is None or df.empty):
        if update:
            await update.message.reply_text(
                f"Error fetching data for {symbol}. Please check the pair and try again."
            )
        return (None, None)

    indicators = analyze_data(df, preferences, liq_lev_tolerance)
    return (indicators, df)


async def check_and_analyze(update, user_id, preferences, args):
    """
    Validate input, check user preferences, and fetch data with indicators.
    
    Args:
        update: Telegram update object
        user_id: User ID
        preferences: User preferences
        args: Command arguments
        
    Returns:
        Tuple of (indicators, dataframe) or None if error
    """
    res = await input_sanity_check_show(args, update)

    if not res:
        return None, None

    # Check if user selected indicators
    if not check_user_preferences(user_id):
        await update.message.reply_text(
            "Please select indicators using /preferences before requesting a chart."
        )
        return None, None

    return await fetch_data_and_get_indicators(res[0], res[1], res[2], preferences, res[3], update)


async def check_signal_limit(user_id):
    """
    Check if the user has reached the signal limit.
    
    Args:
        user_id: User ID to check
        
    Returns:
        Tuple of (is_limit_reached, message)
    """
    previous_signals = get_all_user_signal_requests(user_id)
    
    if len(previous_signals) >= 10:
        return (False, f"You've reached the limit of signals ({len(previous_signals)}). If you want to add a new signal, please remove some of existing signals.")
    
    return (True, "")
