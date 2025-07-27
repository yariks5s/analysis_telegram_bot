"""
Data fetching module for CryptoBot.

This module provides functions to retrieve cryptocurrency market data
from various exchanges and API endpoints.
"""

import pandas as pd  # type: ignore
from datetime import datetime
import requests  # type: ignore

from src.core.utils import VALID_INTERVALS, API_URL, logger
from src.model_classes.indicators import Indicators


def fetch_from_json(data):
    """
    Parse API response JSON into a pandas DataFrame.
    
    Args:
        data: JSON data from the API response
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data or None if error
    """
    if data.get("retCode") != 0:
        logger.error(f"API returned error: {data}")
        return None

    kline_data = data.get("result", {}).get("list", [])
    if not kline_data:
        logger.error("Empty kline_data from Bybit API.")
        return None

    dates, opens, highs, lows, closes, volumes = [], [], [], [], [], []

    for entry in kline_data[::-1]:
        ts_millis = int(entry[0])
        open_price = float(entry[1])
        high_price = float(entry[2])
        low_price = float(entry[3])
        close_price = float(entry[4])
        volume = float(entry[5])

        dt = datetime.utcfromtimestamp(ts_millis / 1000)

        dates.append(dt)
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(volume)

    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=pd.DatetimeIndex(dates),
    )

    return df


def fetch_candles(
    symbol: str,
    desired_total: int,
    interval: str,
    timestamp: float = datetime.utcnow().timestamp(),
):
    """
    Fetch up to the specified number of candles for a given symbol and interval.
    
    Since each request is limited to 200 candles, multiple requests are made as needed.
    
    Args:
        symbol: The trading pair symbol (e.g., "BTCUSD")
        desired_total: The number of candles to fetch
        interval: The interval for each candle (e.g., "1m", "5m", "1h")
        timestamp: End timestamp for the data range (defaults to current time)
        
    Returns:
        pd.DataFrame: DataFrame containing up to desired_total candles
    """
    # Ensure desired_total is an integer to avoid type comparison issues
    try:
        desired_total = int(desired_total)
    except (ValueError, TypeError):
        logger.error(
            f"Invalid desired_total parameter: {desired_total}. Using default of 100."
        )
        desired_total = 100

    # Initialize an empty DataFrame with the expected columns and dtypes
    all_candles = pd.DataFrame(
        {
            "Open": pd.Series(dtype="float64"),
            "High": pd.Series(dtype="float64"),
            "Low": pd.Series(dtype="float64"),
            "Close": pd.Series(dtype="float64"),
            "Volume": pd.Series(dtype="float64"),
        }
    )

    # Current time in milliseconds since epoch
    end_time_ms = int(timestamp * 1000)

    # Define the maximum number of candles per batch request
    batch_limit = 200

    while len(all_candles) < desired_total:
        # Determine how many candles to fetch in this batch
        remaining = desired_total - len(all_candles)
        current_limit = min(batch_limit, remaining)

        # Fetch a batch of candles
        df_batch = fetch_ohlc_data(symbol, current_limit, interval, end=end_time_ms)

        if df_batch is None or df_batch.empty:
            logger.info("No more data returned. Stopping early.")
            break

        # Ensure df_batch has the same dtypes as all_candles
        df_batch = df_batch.astype(all_candles.dtypes)

        # Concatenate the new batch with the existing DataFrame
        if all_candles.empty:
            all_candles = df_batch
        else:
            all_candles = pd.concat([all_candles, df_batch], axis=0)
            all_candles = all_candles[~all_candles.index.duplicated(keep="last")]

        logger.info(
            f"Fetched {len(df_batch)} new candles. Total in memory: {len(all_candles)}"
        )

        # If the batch returned fewer candles than requested, assume no more data is available
        if len(df_batch) < current_limit:
            logger.info("Reached the earliest available data from the exchange.")
            break

        # Update `end_time_ms` to the timestamp of the earliest candle in the current batch minus 1 ms
        earliest_candle_time = df_batch.index.min()  # Assuming the index is datetime
        earliest_candle_ms = int(earliest_candle_time.timestamp() * 1000)
        end_time_ms = earliest_candle_ms - 1  # Move back in time to avoid overlap

    # Sort the DataFrame by ascending time
    all_candles.sort_index(inplace=True)

    # If more candles were fetched than desired (due to overlap), trim the DataFrame
    if len(all_candles) > desired_total:
        all_candles = all_candles.tail(desired_total)

    return all_candles


def fetch_ohlc_data(
    symbol: str, limit: int, interval: str, start: int = None, end: int = None
):
    """
    Fetch historical OHLCV data for a given crypto pair, time period, and interval.
    
    Args:
        symbol: The trading pair symbol (e.g., "BTCUSD")
        limit: Number of candles to fetch
        interval: The interval for each candle (e.g., "1m", "5m", "1h")
        start: Optional start timestamp
        end: Optional end timestamp
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data or None if error
    """
    url = API_URL
    params = {
        "category": "spot",
        "symbol": symbol.upper(),
        "interval": VALID_INTERVALS.get(interval, "60"),  # Default to 1h
        "limit": limit,  # Number of candles to fetch
    }
    if start:
        params["start"] = start
    if end:
        params["end"] = end

    try:
        response = requests.get(url, params=params, timeout=10)
        logger.info(f"Response Status: {response.status_code}")
        logger.info(f"Response Content: {response.text}")

        if response.status_code != 200:
            logger.error(f"Non-200 response from API: {response.status_code}")
            return None

        return fetch_from_json(response.json())

    except Exception as e:
        logger.error(f"Error while fetching data: {e}")
        return None


def analyze_data(df: pd.DataFrame, preferences, liq_lev_tolerance):
    """
    Analyze price data and detect technical indicators based on user preferences.
    
    Args:
        df: DataFrame with OHLCV data
        preferences: Dictionary of indicator preferences
        liq_lev_tolerance: Tolerance for liquidity level detection
        
    Returns:
        Indicators: Object containing all detected indicators
    """
    # Import from the new module structure
    from src.analysis.detection.indicators import (
        detect_order_blocks,
        detect_fvgs,
        detect_liquidity_levels,
        detect_breaker_blocks,
        detect_liquidity_pools,
    )
    
    indicators = Indicators()

    if preferences["order_blocks"]:
        order_blocks = detect_order_blocks(df)
        logger.info(f"Detected Order Blocks: {order_blocks}")
        indicators.order_blocks = order_blocks

    if preferences["fvgs"]:
        fvgs = detect_fvgs(df)
        logger.info(f"Detected FVGs: {fvgs}")
        indicators.fvgs = fvgs

    liquidity_levels = {}
    if preferences["liquidity_levels"]:
        liquidity_levels = detect_liquidity_levels(
            df, window=len(df), stdev_multiplier=liq_lev_tolerance
        )
        logger.info(f"Detected Liquidity Levels: {liquidity_levels}")
        indicators.liquidity_levels = liquidity_levels

    if preferences["breaker_blocks"] and preferences["liquidity_levels"]:
        breaker_blocks = detect_breaker_blocks(df, liquidity_levels)
        logger.info(f"Detected Breaker Blocks: {breaker_blocks}")
        indicators.breaker_blocks = breaker_blocks

    if preferences.get("liquidity_pools"):
        liquidity_pools = detect_liquidity_pools(df)
        logger.info(f"Detected Liquidity Pools: {liquidity_pools}")
        indicators.liquidity_pools = liquidity_pools

    return indicators
