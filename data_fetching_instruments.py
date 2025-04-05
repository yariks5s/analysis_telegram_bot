import pandas as pd  # type: ignore
from datetime import datetime

import requests  # type: ignore

from utils import VALID_INTERVALS, logger, API_URL
from indicators import (
    detect_order_blocks,
    detect_fvgs,
    detect_liquidity_levels,
    detect_breaker_blocks,
)
from IndicatorUtils.indicators import Indicators


def fetch_from_json(data):
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
    Fetch up to the specified number of candles for a given `symbol` and `interval`.
    Since each request is limited to 200 candles, multiple requests are made as needed.

    Parameters:
        symbol (str): The trading pair symbol (e.g., "BTCUSD").
        interval (str): The interval for each candle (e.g., "1m", "5m", "1h").
        desired_total (int): The number of candles to fetch.

    Returns:
        pd.DataFrame: A DataFrame containing up to `desired_total` recent candles.
                      If fewer are available, returns as many as possible.
    """
    # Initialize an empty DataFrame with the expected columns
    all_candles = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

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
            print("No more data returned. Stopping early.")
            break

        # Concatenate the new batch with the existing DataFrame
        all_candles = pd.concat([all_candles, df_batch]).drop_duplicates()

        print(
            f"Fetched {len(df_batch)} new candles. Total in memory: {len(all_candles)}"
        )

        # If the batch returned fewer candles than requested, assume no more data is available
        if len(df_batch) < current_limit:
            print("Reached the earliest available data from the exchange.")
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

    return indicators
