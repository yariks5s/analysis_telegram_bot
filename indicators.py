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

def detect_multi_candle_order_blocks(
    df: pd.DataFrame,
    min_consolidation_candles=2,
    volume_multiplier=1.2,
    breakout_factor=1.01
):
    """
    1) Find 'consolidation zones' of at least min_consolidation_candles
       where the range is small and overlapping.
    2) Confirm a breakout from that zone with volume above average * volume_multiplier.
    3) Mark that consolidation zone as an 'order block zone.'
    """
    blocks = OrderBlocks()
    if df.empty:
        return blocks

    avg_volume = df['Volume'].mean()
    i = 0
    while i < len(df) - min_consolidation_candles:
        # Step A: Check for a consolidation region of N candles
        # e.g. consecutive overlapping candles with small range
        region_start = i
        region_end = i + min_consolidation_candles - 1

        # Are these candles overlapping enough to be called 'consolidation'?
        # Example logic:
        highest_high = df['High'].iloc[region_start:region_end+1].max()
        lowest_low = df['Low'].iloc[region_start:region_end+1].min()
        # overlap or small range check, e.g., range < 0.5% of last close
        range_size = highest_high - lowest_low
        if range_size < 0.005 * df['Close'].iloc[region_end]:
            # Potential zone
            # Step B: Look for a breakout candle right after the zone
            if region_end + 1 < len(df):
                breakout_candle = region_end + 1
                candle_high = df['High'].iloc[breakout_candle]
                candle_low = df['Low'].iloc[breakout_candle]
                candle_volume = df['Volume'].iloc[breakout_candle]
                previous_high = highest_high
                previous_low = lowest_low

                # Check volume
                if candle_volume > volume_multiplier * avg_volume:
                    # Check if bullish breakout (e.g. candle_high > previous_high * breakout_factor)
                    if candle_high > (previous_high * breakout_factor):
                        # Mark region as bullish order block
                        blocks.add(OrderBlock(
                            block_type='bullish',
                            index=region_end,
                            high=previous_high,
                            low=previous_low
                        ))
                        i = breakout_candle + 1
                        continue

                    # Check if bearish breakout
                    if candle_low < (previous_low * (2 - breakout_factor)): 
                        # 2 - breakout_factor might be 0.99 => 1% lower, etc.
                        blocks.add(OrderBlock(
                            block_type='bearish',
                            index=region_end,
                            high=previous_high,
                            low=previous_low
                        ))
                        i = breakout_candle + 1
                        continue

        i += 1

    return blocks


def detect_fvgs(df: pd.DataFrame, min_fvg_ratio=0.005):
    """
    Detect Fair Value Gaps (FVGs) in price action and check if they are covered later.
    Filter out relatively small FVGs, i.e. those whose size is less than
    (min_fvg_ratio * last_close_price).
    """
    fvgs = FVGs()

    if df.empty:
        return fvgs

    last_close_price = df['Close'].iloc[-1]

    for i in range(2, len(df)):
        # Bullish FVG
        if df['Low'].iloc[i] > df['High'].iloc[i - 2]:
            gap_size = df['Low'].iloc[i] - df['High'].iloc[i - 2]
            # Filter condition: skip if gap_size < min_fvg_ratio * last_close_price
            if gap_size < min_fvg_ratio * last_close_price:
                continue

            is_covered = False
            for j in range(i + 1, len(df)):
                if df['Low'].iloc[j] <= df['Low'].iloc[i]:
                    is_covered = True 
                    break

            fvgs.add(FVG(i - 2, i, df['High'].iloc[i - 2], df['Low'].iloc[i], 'bullish', is_covered))

        # Bearish FVG
        elif df['High'].iloc[i] < df['Low'].iloc[i - 2]:
            gap_size = df['Low'].iloc[i - 2] - df['High'].iloc[i]
            if gap_size < min_fvg_ratio * last_close_price:
                continue

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
