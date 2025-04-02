import pandas as pd  # type: ignore

from IndicatorUtils.order_block_utils import OrderBlock, OrderBlocks
from IndicatorUtils.fvg_utils import FVG, FVGs
from IndicatorUtils.liquidity_level_utils import LiquidityLevel, LiquidityLevels
from IndicatorUtils.breaker_block_utils import BreakerBlock, BreakerBlocks

from utils import is_bullish, is_bearish


def detect_impulses(
    df: pd.DataFrame, overall_impulse_multiplier: float = 1.5, min_chain_length: int = 2
):
    """
    Detect impulses in the DataFrame as chains of consecutive candles in the same direction.

    For a bullish impulse:
      - The chain is built from consecutive bullish candles.
      - The overall move is: last candle's Close - first candle's Open.

    For a bearish impulse:
      - The chain is built from consecutive bearish candles.
      - The overall move is: first candle's Close - last candle's Close.

    An impulse is accepted if the overall move is at least overall_impulse_multiplier times
    the expected cumulative move (chain_length * baseline_change), where baseline_change is the
    average absolute change of all candles.

    Returns:
        A list of impulses. Each impulse is a tuple:
          (chain_start_index, chain_end_index, impulse_type, overall_move, chain_length)
    """
    baseline_change = (df["Close"] - df["Open"]).abs().mean()
    impulses = []
    n = len(df)
    i = 0
    while i < n:
        candle = df.iloc[i]
        # Detect bullish impulses: consecutive bullish candles.
        if is_bullish(candle):
            chain_start = i
            chain_end = i
            while chain_end + 1 < n and is_bullish(df.iloc[chain_end + 1]):
                chain_end += 1
            chain_length = chain_end - chain_start + 1
            if chain_length >= min_chain_length:
                overall_move = (
                    df.iloc[chain_end]["Close"] - df.iloc[chain_start]["Open"]
                )
                expected_move = chain_length * baseline_change
                if overall_move >= overall_impulse_multiplier * expected_move:
                    impulses.append(
                        (
                            chain_start,
                            chain_end,
                            "bullish",
                            overall_move,
                            chain_length,
                            df.iloc[chain_start]["Open"],
                            df.iloc[chain_end]["Close"],
                        )
                    )
            i = chain_end + 1
        # Detect bearish impulses: consecutive bearish candles.
        elif is_bearish(candle):
            chain_start = i
            chain_end = i
            while chain_end + 1 < n and is_bearish(df.iloc[chain_end + 1]):
                chain_end += 1
            chain_length = chain_end - chain_start + 1
            if chain_length >= min_chain_length:
                overall_move = (
                    df.iloc[chain_start]["Close"] - df.iloc[chain_end]["Close"]
                )
                expected_move = chain_length * baseline_change
                if overall_move >= overall_impulse_multiplier * expected_move:
                    impulses.append(
                        (
                            chain_start,
                            chain_end,
                            "bearish",
                            overall_move,
                            chain_length,
                            df.iloc[chain_start]["Open"],
                            df.iloc[chain_end]["Close"],
                        )
                    )
            i = chain_end + 1
        else:
            i += 1
    return impulses


def detect_order_blocks_from_impulses(
    df: pd.DataFrame, impulses, min_gap: int = 5
) -> "OrderBlocks":
    """
    For each detected impulse, find the candidate order block by searching backwards from the start
    of the impulse until a candle with the opposite trend is found. For a bullish impulse (upward move),
    the candidate must be bearish; for a bearish impulse (downward move), the candidate must be bullish.

    If two order blocks are too close (within min_gap candles), the later one replaces the earlier.

    Args:
        df (pd.DataFrame): DataFrame with candlestick data.
        impulses (list): List of impulses as returned by detect_impulses().
        min_gap (int): Minimum gap (in candle count) required between order blocks.

    Returns:
        OrderBlocks: An instance containing detected OrderBlock objects.
    """
    order_blocks = OrderBlocks()

    for impulse in impulses:
        chain_start, chain_end, impulse_type, overall_move, chain_length, _, _ = impulse

        # Start from the candle immediately before the impulse chain
        candidate_index = chain_start - 1
        found_candidate = None

        # For bullish impulse, search backwards for the closest bearish candle.
        if impulse_type == "bullish":
            while candidate_index >= 0:
                candidate = df.iloc[candidate_index]
                if is_bearish(candidate):
                    found_candidate = candidate_index
                    break
                candidate_index -= 1

        # For bearish impulse, search backwards for the closest bullish candle.
        elif impulse_type == "bearish":
            while candidate_index >= 0:
                candidate = df.iloc[candidate_index]
                if is_bullish(candidate):
                    found_candidate = candidate_index
                    break
                candidate_index -= 1

        if found_candidate is None:
            # No suitable candidate found; skip this impulse.
            continue

        candidate_index = found_candidate
        candidate = df.iloc[candidate_index]

        # The order block's type is derived from the impulse type.
        # For bullish impulse, we label the order block as "bullish" (indicating a potential supply zone).
        # For bearish impulse, we label it as "bearish" (indicating a potential demand zone).
        block_type = impulse_type

        new_block = OrderBlock(
            block_type, candidate_index, candidate["High"], candidate["Low"]
        )
        new_block.pos = candidate_index  # store positional index

        # Enforce the minimum gap between order blocks.
        if (
            order_blocks.list
            and (candidate_index - order_blocks.list[-1].index < min_gap)
            and order_blocks.list[-1].block_type == new_block.block_type
        ):
            order_blocks.list[-1] = new_block
        else:
            order_blocks.add(new_block)

    return order_blocks


def detect_order_blocks(
    df: pd.DataFrame,
    overall_impulse_multiplier: float = 1.5,
    min_chain_length: int = 2,
    min_gap: int = 5,
):
    """
    Detect potential order blocks by first detecting impulses in the DataFrame and then determining
    the candidate order block for each impulse. The candidate order block is the candle immediately preceding
    the impulse chain.

    Args:
        df (pd.DataFrame): DataFrame with candlestick data ('Open', 'High', 'Low', 'Close').
        overall_impulse_multiplier (float): Multiplier threshold for impulse detection.
        min_chain_length (int): Minimum number of candles in an impulse chain.
        min_gap (int): Minimum gap (in candle count) required between order blocks.

    Returns:
        OrderBlocks: An instance containing the detected OrderBlock objects.
    """
    impulses = detect_impulses(df, overall_impulse_multiplier, min_chain_length)
    return detect_order_blocks_from_impulses(df, impulses, min_gap)


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
