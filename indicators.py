import pandas as pd  # type: ignore

from IndicatorUtils.order_block_utils import OrderBlock, OrderBlocks
from IndicatorUtils.fvg_utils import FVG, FVGs
from IndicatorUtils.liquidity_level_utils import LiquidityLevel, LiquidityLevels
from IndicatorUtils.breaker_block_utils import BreakerBlock, BreakerBlocks

from utils import is_bullish, is_bearish


def detect_order_blocks(
    df: pd.DataFrame,
    max_candles: int = 7,
    impulse_threshold_multiplier: float = 1.2,
    min_gap: int = 5
):
    """
    Detects order blocks and evaluates impulse strength by comparing the average price change
    per confirming candle during the impulse with the average price change of regular candles.
    
    For a bullish order block candidate (candidate is bearish: Close < Open):
      - Checks up to max_candles following candles.
      - Counts confirmations: a candle confirms if its Close > candidate's High.
      - The counting is done from the end of the window backward.
      - Computes impulse_change as the difference between the last confirming candle's Close 
        and candidate's High.
      - Computes average_impulse_change = impulse_change / impulse_count.
      - The candidate is considered a valid bullish order block if average_impulse_change 
        >= impulse_threshold_multiplier * baseline_change.
    
    For a bearish order block candidate (candidate is bullish: Close > Open):
      - Checks up to max_candles following candles.
      - A candle confirms if its Close < candidate's Low.
      - Counts confirmations from the end backward.
      - Computes impulse_change as the difference between candidate's Low and the last confirming 
        candle's Close.
      - Computes average_impulse_change = impulse_change / impulse_count.
      - The candidate is considered valid if average_impulse_change 
        >= impulse_threshold_multiplier * baseline_change.
    
    Additionally, if two order blocks are detected too close to each other (i.e. within min_gap candles),
    the second one replaces the first.
    
    Args:
        df (pd.DataFrame): DataFrame containing candlestick data with columns 'Open', 'High', 'Low', 'Close'.
        max_candles (int): Maximum number of candles to consider for confirmation (default is 7).
        impulse_threshold_multiplier (float): Multiplier to compare the impulse's average change 
                                              with the baseline average change.
        min_gap (int): Minimum gap (in candle count) required between order blocks.
    
    Returns:
        OrderBlocks: An instance of the OrderBlocks class containing detected OrderBlock objects.
    """
    order_blocks = OrderBlocks()
    
    # Compute baseline average change (absolute difference between Close and Open) for all candles
    baseline_change = (df["Close"] - df["Open"]).abs().mean()
    
    # Loop over candidate candles ensuring enough candles for confirmation exist
    for i in range(len(df) - max_candles):
        candidate = df.iloc[i]
        
        # Process bullish order block candidate:
        # Candidate candle is bearish (Close < Open) and we expect confirmation with Close > candidate's High.
        if is_bearish(candidate):
            window = df.iloc[i+1:i+max_candles+1]
            confirmations = [candle["Close"] > candidate["High"] for _, candle in window.iterrows()]
            impulse_count = 0
            last_confirming_close = None
            # Count consecutive confirmations starting from the end of the window
            for condition, close_value in zip(reversed(confirmations), reversed(window["Close"].tolist())):
                if condition:
                    impulse_count += 1
                    if last_confirming_close is None:
                        last_confirming_close = close_value
                else:
                    break
            
            if impulse_count > 0 and last_confirming_close is not None:
                impulse_change = last_confirming_close - candidate["High"]
                average_impulse_change = impulse_change / impulse_count
                # Compare with baseline change to determine if this is a valid impulse
                if average_impulse_change >= impulse_threshold_multiplier * baseline_change:
                    # Now we check if there are any more bearish candles ahead to get the last one
                    index = i
                    while (is_bearish(df.iloc[index + 1])):
                        candidate = df.iloc[index + 1]
                        index += 1
                    ready_candidate = OrderBlock("bullish", index, candidate["High"], candidate["Low"])
                    # Check if the last accepted order block is too close.
                    if order_blocks.list and (i - order_blocks.list[-1].index < min_gap):
                        # Replace the previous order block with the current one.
                        order_blocks.list[-1] = ready_candidate
                    else:
                        order_blocks.add(ready_candidate)
        
        # Process bearish order block candidate:
        # Candidate candle is bullish (Close > Open) and we expect confirmation with Close < candidate's Low.
        elif is_bullish(candidate):
            window = df.iloc[i+1:i+max_candles+1]
            confirmations = [candle["Close"] < candidate["Low"] for _, candle in window.iterrows()]
            impulse_count = 0
            last_confirming_close = None
            for condition, close_value in zip(reversed(confirmations), reversed(window["Close"].tolist())):
                if condition:
                    impulse_count += 1
                    if last_confirming_close is None:
                        last_confirming_close = close_value
                else:
                    break
            
            if impulse_count > 0 and last_confirming_close is not None:
                impulse_change = candidate["Low"] - last_confirming_close
                average_impulse_change = impulse_change / impulse_count
                if average_impulse_change >= impulse_threshold_multiplier * baseline_change:
                    # Now we check if there are any more bullish candles ahead to get the last one
                    index = i
                    while (is_bullish(df.iloc[index + 1])):
                        candidate = df.iloc[index + 1]
                        index += 1
                    ready_candidate = OrderBlock("bearish", index, candidate["High"], candidate["Low"])
                    if order_blocks.list and (i - order_blocks.list[-1].index < min_gap):
                        order_blocks.list[-1] = ready_candidate
                    else:
                        order_blocks.add(ready_candidate)
    
    return order_blocks


def detect_multi_candle_order_blocks(
    df: pd.DataFrame,
    min_consolidation_candles=2,
    volume_multiplier=1.2,
    breakout_factor=1.01,
):
    """
    Detect order blocks by:
      1. Identifying consolidation zones where at least `min_consolidation_candles` have overlapping ranges.
      2. Confirming a breakout from that zone with volume above average * volume_multiplier.
      3. Marking the zone as an order block zone.
    
    For a bullish order block, the breakout candle closes above the consolidation zone’s high.
    For a bearish order block, the breakout candle closes below the consolidation zone’s low.

    Returns:
        OrderBlocks: A collection of detected multi-candle order blocks.
    """
    blocks = OrderBlocks()
    if df.empty:
        return blocks

    avg_volume = df["Volume"].mean()
    i = 0
    while i < len(df) - min_consolidation_candles:
        # Step A: Check for a consolidation region of N candles
        # e.g. consecutive overlapping candles with small range
        region_start = i
        region_end = i + min_consolidation_candles - 1

        # Are these candles overlapping enough to be called 'consolidation'?
        # Example logic:
        highest_high = df["High"].iloc[region_start : region_end + 1].max()
        lowest_low = df["Low"].iloc[region_start : region_end + 1].min()
        # overlap or small range check, e.g., range < 0.5% of last close
        range_size = highest_high - lowest_low
        if range_size < 0.005 * df["Close"].iloc[region_end]:
            # Potential zone
            # Step B: Look for a breakout candle right after the zone
            if region_end + 1 < len(df):
                breakout_candle = region_end + 1
                candle_high = df["High"].iloc[breakout_candle]
                candle_low = df["Low"].iloc[breakout_candle]
                candle_volume = df["Volume"].iloc[breakout_candle]
                previous_high = highest_high
                previous_low = lowest_low

                # Check volume
                if candle_volume > volume_multiplier * avg_volume:
                    # Check if bullish breakout (e.g. candle_high > previous_high * breakout_factor)
                    if candle_high > (previous_high * breakout_factor):
                        # Mark region as bullish order block
                        blocks.add(
                            OrderBlock(
                                block_type="bullish",
                                index=region_end,
                                high=previous_high,
                                low=previous_low,
                            )
                        )
                        i = breakout_candle + 1
                        continue

                    # Check if bearish breakout
                    if candle_low < (previous_low * (2 - breakout_factor)):
                        # 2 - breakout_factor might be 0.99 => 1% lower, etc.
                        blocks.add(
                            OrderBlock(
                                block_type="bearish",
                                index=region_end,
                                high=previous_high,
                                low=previous_low,
                            )
                        )
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

    last_close_price = df["Close"].iloc[-1]

    for i in range(2, len(df)):
        # Bullish FVG
        if df["Low"].iloc[i] > df["High"].iloc[i - 2]:
            gap_size = df["Low"].iloc[i] - df["High"].iloc[i - 2]
            # Filter condition: skip if gap_size < min_fvg_ratio * last_close_price
            if gap_size < min_fvg_ratio * last_close_price:
                continue

            is_covered = False
            for j in range(i + 1, len(df)):
                if df["Low"].iloc[j] <= df["Low"].iloc[i]:
                    is_covered = True
                    break

            fvgs.add(
                FVG(
                    i - 2,
                    i,
                    df["High"].iloc[i - 2],
                    df["Low"].iloc[i],
                    "bullish",
                    is_covered,
                )
            )

        # Bearish FVG
        elif df["High"].iloc[i] < df["Low"].iloc[i - 2]:
            gap_size = df["Low"].iloc[i - 2] - df["High"].iloc[i]
            if gap_size < min_fvg_ratio * last_close_price:
                continue

            is_covered = False
            for j in range(i + 1, len(df)):
                if df["High"].iloc[j] >= df["High"].iloc[i]:
                    is_covered = True
                    break

            fvgs.add(
                FVG(
                    i - 2,
                    i,
                    df["Low"].iloc[i - 2],
                    df["High"].iloc[i],
                    "bearish",
                    is_covered,
                )
            )

    return fvgs


### NOTE: Instead of support and resistance levels we can do just a liquidity levels


def detect_support_resistance_levels(
    df: pd.DataFrame, window: int = 50, tolerance: float = 0.05
):
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
        low, high = recent_df["Low"].iloc[i], recent_df["High"].iloc[i]

        # Local minima as support
        if low < recent_df["Low"].iloc[i - 1] and low < recent_df["Low"].iloc[i + 1]:
            levels.add(LiquidityLevel("support", low))

        # Local maxima as resistance
        if (
            high > recent_df["High"].iloc[i - 1]
            and high > recent_df["High"].iloc[i + 1]
        ):
            levels.add(LiquidityLevel("resistance", high))

    # Group levels
    grouped_levels = LiquidityLevels()
    for level in levels.list:
        if (
            not grouped_levels
            or abs(level.price - grouped_levels.list[-1].price)
            > tolerance * level.price
        ):
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

    if not liquidity_levels.list:
        return breaker_blocks

    # Iterate over each candle
    for i in range(2, len(df)):
        low, high, close = df["Low"].iloc[i], df["High"].iloc[i], df["Close"].iloc[i]

        # Check for bullish and bearish breaker blocks
        for level in liquidity_levels.list:
            # Bullish breaker: sweeps support and reverses upward
            if level.is_support() and low < level.price and close > level.price:
                breaker_blocks.add(
                    BreakerBlock(block_type="bullish", index=i, zone=(low, high))
                )

            # Bearish breaker: sweeps resistance and reverses downward
            elif level.is_resistance() and high > level.price and close < level.price:
                breaker_blocks.add(
                    BreakerBlock(block_type="bearish", index=i, zone=(low, high))
                )

    return breaker_blocks
