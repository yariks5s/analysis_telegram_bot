import pandas as pd
from datetime import datetime

import requests

from utils import VALID_INTERVALS, logger, API_URL
from indicators import detect_order_blocks, detect_multi_candle_order_blocks, detect_fvgs, detect_support_resistance_levels, detect_breaker_blocks
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

    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    }, index=pd.DatetimeIndex(dates))

    return df


def fetch_ohlc_data(symbol: str, limit: int, interval: str, start: int = None, end: int = None):
    """
    Fetch historical OHLCV data for a given crypto pair, time period, and interval.
    """
    url = API_URL
    params = {
        "category": "spot",
        "symbol": symbol.upper(),
        "interval": VALID_INTERVALS.get(interval, "60"),  # Default to 1h
        "limit": limit  # Number of candles to fetch
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

    if (preferences["order_blocks"]):
        order_blocks = detect_multi_candle_order_blocks(df)
        logger.info(f"Detected Order Blocks: {order_blocks}")
        indicators.order_blocks = order_blocks
    
    if (preferences["fvgs"]):
        fvgs = detect_fvgs(df)
        logger.info(f"Detected FVGs: {fvgs}")
        indicators.fvgs = fvgs

    if (not liq_lev_tolerance):
        liq_lev_tolerance = 0.05;

    liquidity_levels = {}
    if (preferences["liquidity_levels"]):
        liquidity_levels = detect_support_resistance_levels(df, window=len(df), tolerance=liq_lev_tolerance)
        logger.info(f"Detected Liquidity Levels: {liquidity_levels}")
        indicators.liquidity_levels = liquidity_levels

    if (preferences["breaker_blocks"] and preferences["liquidity_levels"]):
        breaker_blocks = detect_breaker_blocks(df, liquidity_levels)
        logger.info(f"Detected Breaker Blocks: {breaker_blocks}")
        indicators.breaker_blocks = breaker_blocks

    return indicators


def fetch_last_1000_candles(symbol: str, interval: str):
    """
    Fetch up to the last 1000 candles for a given `symbol` and `interval`.
    Because each request is limited to 200 candles, we will do multiple requests.

    Returns a DataFrame with up to 1000 recent candles. If fewer are available, returns as many as possible.
    """
    all_candles = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    # We'll fetch from "now" going backward in time until we accumulate 1000 rows or run out of data.
    # Bybit allows using the `end` parameter to specify the latest time we want data for,
    # then it will return earlier candles. We can shift `end` backward each time.
    
    # Let's define "end" as the current time in milliseconds
    end_time_ms = int(datetime.utcnow().timestamp() * 1000)

    # We'll keep fetching in batches of 200 until we gather 1000 or no more data.
    batch_limit = 200
    desired_total = 1000

    while len(all_candles) < desired_total:
        df_batch = fetch_ohlc_data(symbol, batch_limit, interval, end=end_time_ms)
        if df_batch is None or df_batch.empty:
            print("No more data returned. Stopping early.")
            break

        # Merge the new batch with our existing DataFrame
        # Because we are going backward in time, the "new" batch will likely be older candles than we have.
        # We'll just do a union and keep the unique ones.
        # You can also handle duplicates if needed.
        before_merge_len = len(all_candles)
        all_candles = pd.concat([all_candles, df_batch]).drop_duplicates()
        after_merge_len = len(all_candles)

        print(f"Fetched {len(df_batch)} new candles. Total in memory: {after_merge_len}")

        # If the batch is less than 200, we probably can't go further back
        if len(df_batch) < batch_limit:
            print("We likely reached earliest available data from Bybit.")
            break

        # Update end_time_ms to the earliest candle's timestamp in the batch minus 1 ms,
        # so the next batch will be older than that.
        earliest_candle_time = df_batch.index[0]  # because df_batch is sorted by date ascending
        earliest_candle_ms = int(earliest_candle_time.timestamp() * 1000)
        end_time_ms = earliest_candle_ms - 1  # go one millisecond before to avoid overlap/duplicate

    # Now we have up to 1000 candles in all_candles (or fewer if the exchange didn't have more).
    # We'll sort them by ascending time.
    all_candles.sort_index(inplace=True)

    # Finally, if we got more than 1000 (possible duplicates or overlap?), slice the last 1000
    # e.g. "recent 1000" means the last 1000 rows in ascending order => tail(1000)
    if len(all_candles) > desired_total:
        all_candles = all_candles.tail(desired_total)

    return all_candles
