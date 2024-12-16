import pandas as pd
from datetime import datetime

import requests

from helpers import VALID_INTERVALS, logger

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
        
        data = response.json()
        
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
    
    except Exception as e:
        logger.error(f"Error while fetching data: {e}")
        return None