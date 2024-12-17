import pandas as pd
import logging

logger = logging.getLogger(__name__)

def calculate_macd(data: pd.DataFrame):
    """
    Calculate MACD and Signal Line.
    """
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_rsi(data: pd.DataFrame, period: int = 14):
    """
    Calculate Relative Strength Index (RSI).
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

VALID_INTERVALS = {
    "1m": "1",  # 1 minute
    "5m": "5",  # 5 minutes
    "15m": "15",  # 15 minutes
    "30m": "30",  # 30 minutes
    "1h": "60",  # 1 hour
    "4h": "240",  # 4 hours
    "1d": "D",  # 1 day
    "1w": "W",  # 1 week
}

async def input_sanity_check(args, update) -> tuple:
    # Default values
    symbol = "BTCUSDT"
    hours = 24
    interval = "1h"
    liq_lev_tolerance=0

    if len(args) >= 1:
        symbol = args[0].upper()
    if len(args) >= 2:
        try:
            hours = int(args[1])
        except ValueError:
            await update.message.reply_text("Invalid hours specified. Please provide a valid number.")
            return tuple()
    if len(args) >= 3:
        interval = args[2].lower()
        if interval not in VALID_INTERVALS:
            await update.message.reply_text(f"Invalid interval specified. Valid intervals are: {', '.join(VALID_INTERVALS.keys())}")
            return tuple()
    if len(args) >= 4:
        liq_lev_tolerance = float(args[3])
        if liq_lev_tolerance < 0 or liq_lev_tolerance > 1:
            await update.message.reply_text(f"Invalid liquidity level tolerance specified. It should be a number between 0 and 1")
            return tuple()

    if hours < 1 or hours > 200:
        await update.message.reply_text("Amount of intervals must be between 1 and 200 (due to API limits).")
        return tuple()

    return (symbol, hours, interval, liq_lev_tolerance)
