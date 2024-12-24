import pandas as pd

from IndicatorUtils.order_block_utils import OrderBlock, OrderBlocks
from IndicatorUtils.fvg_utils import FVG, FVGs
from IndicatorUtils.liquidity_level_utils import LiquidityLevel, LiquidityLevels
from IndicatorUtils.breaker_block_utils import BreakerBlock, BreakerBlocks

def detect_order_blocks(df: pd.DataFrame, volume_threshold=1.5, body_percentage=0.5, breakout_factor=1.01):
    """
    Detect less sensitive order blocks based on bullish or bearish breakouts.

    Parameters:
        df (pd.DataFrame): A DataFrame containing OHLCV data with columns ['Open', 'High', 'Low', 'Close', 'Volume'].
        volume_threshold (float): Multiplier to determine significant volume (e.g., 1.5x the average).
        body_percentage (float): Minimum body size as a percentage of the candle's range to be considered significant.
        breakout_factor (float): Multiplier to determine the strength of the breakout (e.g., 1.01 for 1% breakout).

    Returns:
        list: A list of tuples representing order blocks, where each tuple is:
              (index, high, low, type)
              - index: Index of the order block candle
              - high: High of the candle
              - low: Low of the candle
              - type: 'bullish' or 'bearish'
    """
    order_blocks = OrderBlocks()
    avg_volume = df['Volume'].mean()

    for i in range(1, len(df) - 3):
        high, low, close, open_price = df['High'].iloc[i], df['Low'].iloc[i], df['Close'].iloc[i], df['Open'].iloc[i]
        volume = df['Volume'].iloc[i]

        body_size = abs(close - open_price)
        range_size = high - low

        # Skip small candles
        if range_size == 0 or body_size / range_size < body_percentage:
            continue

        # Skip low-volume candles
        if volume < volume_threshold * avg_volume:
            continue

        # Detect order blocks
        if close < open_price and df['Close'].iloc[i + 1] > high * breakout_factor:
            order_blocks.add(OrderBlock('bearish', i, high, low))
        elif close > open_price and df['Close'].iloc[i + 1] < low * breakout_factor:
            order_blocks.add(OrderBlock('bullish', i, high, low))

    return order_blocks


def detect_fvgs(df: pd.DataFrame):
    """
    Detect Fair Value Gaps (FVGs) in price action and check if they are covered later.

    Parameters:
        df (pd.DataFrame): A DataFrame containing OHLCV data with columns ['Open', 'High', 'Low', 'Close'].

    Returns:
        list: A list of tuples representing FVGs, where each tuple is:
              (start_index, end_index, start_price, end_price, type, covered)
              - start_index: Start of the gap
              - end_index: End of the gap
              - start_price: Price at the start of the gap
              - end_price: Price at the end of the gap
              - type: 'bullish' or 'bearish'
              - covered: Boolean indicating whether the FVG was later covered
    """
    fvgs = FVGs()

    for i in range(2, len(df)):
        # Bullish FVG
        if df['Low'].iloc[i] > df['High'].iloc[i - 2]:
            is_covered = False
            for j in range(i + 1, len(df)):
                if df['Low'].iloc[j] <= df['Low'].iloc[i]:
                   is_covered = True 
                   break

            fvgs.add(FVG(i - 2, i, df['High'].iloc[i - 2], df['Low'].iloc[i], 'bullish', is_covered))

        # Bearish FVG
        elif df['High'].iloc[i] < df['Low'].iloc[i - 2]:
            is_covered = False
            for j in range(i + 1, len(df)):
                if df['High'].iloc[j] >= df['High'].iloc[i]:
                   is_covered = True 
                   break

            fvgs.add(FVG(i - 2, i, df['Low'].iloc[i - 2], df['High'].iloc[i], 'bearish', is_covered))

    return fvgs


### NOTE: Instead of support and resistance levels we can do just a liquidity levels

def detect_support_resistance_levels(df: pd.DataFrame, window: int = 50, tolerance: float = 0.05):
    """
    Identifies support and resistance levels within a given range of candlesticks.

    Parameters:
        df (pd.DataFrame): OHLC data (Open, High, Low, Close).
        window (int): The number of recent candlesticks to analyze (between 50 and 200).
        tolerance (float): The tolerance for grouping nearby levels (default is 0.2%).

    Returns:
        Tuple[List[float], List[float]]: Lists of support and resistance levels.
    """
    # Ensure the window size does not exceed the DataFrame size
    recent_df = df.tail(window)
    levels = LiquidityLevels()

    for i in range(1, len(recent_df) - 1):
        low, high = recent_df['Low'].iloc[i], recent_df['High'].iloc[i]

        # Local minima as support
        if low < recent_df['Low'].iloc[i - 1] and low < recent_df['Low'].iloc[i + 1]:
            levels.add(LiquidityLevel('support', low))

        # Local maxima as resistance
        if high > recent_df['High'].iloc[i - 1] and high > recent_df['High'].iloc[i + 1]:
            levels.add(LiquidityLevel('resistance', high))

    # Group levels
    grouped_levels = LiquidityLevels()
    for level in levels.list:
        if not grouped_levels or abs(level.price - grouped_levels.list[-1].price) > tolerance * level.price:
            grouped_levels.add(level)

    return grouped_levels

def detect_breaker_blocks(df: pd.DataFrame, liquidity_levels: LiquidityLevels):
    """
    Detect breaker blocks based on liquidity sweeps and reversals.

    Parameters:
        df (pd.DataFrame): OHLCV DataFrame with columns ['Open', 'High', 'Low', 'Close'].
        liquidity_levels (list): A list of LiquidityLevel objects (support/resistance).

    Returns:
        list: A list of BreakerBlock objects.
    """
    breaker_blocks = BreakerBlocks()

    if (not liquidity_levels.list):
        return breaker_blocks

    # Iterate over each candle
    for i in range(2, len(df)):
        low, high, close = df['Low'].iloc[i], df['High'].iloc[i], df['Close'].iloc[i]

        # Check for bullish and bearish breaker blocks
        for level in liquidity_levels.list:
            # Bullish breaker: sweeps support and reverses upward
            if level.is_support() and low < level.price and close > level.price:
                breaker_blocks.add(BreakerBlock(
                    block_type='bullish',
                    index=i,
                    zone=(low, high)
                ))

            # Bearish breaker: sweeps resistance and reverses downward
            elif level.is_resistance() and high > level.price and close < level.price:
                breaker_blocks.add(BreakerBlock(
                    block_type='bearish',
                    index=i,
                    zone=(low, high)
                ))

    return breaker_blocks
