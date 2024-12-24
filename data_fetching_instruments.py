import pandas as pd
from datetime import datetime

import requests

from utils import VALID_INTERVALS, logger
from indicators import detect_order_blocks, detect_fvgs, detect_support_resistance_levels, detect_breaker_blocks
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

def fetch_ohlc_data(symbol: str, limit: int, interval: str):
    """
    Fetch historical OHLCV data for a given crypto pair, time period, and interval.
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "spot",
        "symbol": symbol.upper(),
        "interval": VALID_INTERVALS.get(interval, "60"),  # Default to 1h
        "limit": limit  # Number of candles to fetch
    }
    
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
    
def analyze_data(df: pd.DataFrame, preferences, liq_lev_tolerance = 0.05):
    indicators = Indicators()

    if (preferences["order_blocks"]):
        order_blocks = detect_order_blocks(df)
        logger.info(f"Detected Order Blocks: {order_blocks}")
        indicators.order_blocks = order_blocks
    
    if (preferences["fvgs"]):
        fvgs = detect_fvgs(df)
        logger.info(f"Detected FVGs: {fvgs}")
        indicators.fvgs = fvgs

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
