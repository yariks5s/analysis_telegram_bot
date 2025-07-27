"""
Technical indicator detection functions for CryptoBot.

This module contains functions to detect various technical indicators
including order blocks, fair value gaps, liquidity levels, and breaker blocks.
"""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from src.analysis.utils.order_block_utils import OrderBlock, OrderBlocks
from src.analysis.utils.fvg_utils import FVG, FVGs
from src.analysis.utils.liquidity_level_utils import LiquidityLevel, LiquidityLevels
from src.analysis.utils.breaker_block_utils import BreakerBlock, BreakerBlocks
from src.analysis.utils.liquidity_pool_utils import LiquidityPool, LiquidityPools

from sklearn.cluster import DBSCAN  # type: ignore

from src.core.utils import is_bullish, is_bearish


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
):
    """
    Find order blocks from detected impulses by searching for opposing candles.
    
    Args:
        df: DataFrame with OHLC data
        impulses: List of detected impulses
        min_gap: Minimum candles between order block and impulse
        
    Returns:
        OrderBlocks: Container with detected order blocks
    """
    order_blocks = OrderBlocks()

    for impulse in impulses:
        chain_start, chain_end, impulse_type, _, _, _, _ = impulse

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
        new_block.pos = candidate_index

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
    min_chain_length: int = 1,
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


def detect_fvgs(df: pd.DataFrame, min_fvg_ratio=0.0005):
    """
    Detect Fair Value Gaps (FVGs) in price action and check if they are covered later.

    An FVG is detected when a gap exists between candle i-2 and candle i that is
    larger than (min_fvg_ratio * last_close_price). This function also checks if the
    gap is later "covered" by subsequent price action.

    Optimization:
      - Compute the positional indices of the global highest high and the global lowest low.
      - Determine a single extreme index (global_extreme_idx) as the maximum of the two.
      - For any FVG candidate, if the starting candle (i-2) occurs before global_extreme_idx,
        skip it because it will be covered by one (or both) of the extrema.

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['Open', 'High', 'Low', 'Close'].
        min_fvg_ratio (float): Minimum gap size as a fraction of the last close price.

    Returns:
        FVGs: A container holding all detected FVG objects.
    """
    fvgs = FVGs()

    if df.empty:
        return fvgs

    last_close_price = df["Close"].iloc[-1]

    # TODO
    # Optimization suggestion: it is possible to build a two sorted lists with some amount of candles, sorted
    # by low and high, which are stored along with their indices. If we pass that index when iterating over candles,
    # just remove that candle from sorted list of candles. And ta-da, you don't need to recalculate the min/max again.

    # Convert the index of the global high and global low into positional indices
    global_high_idx = df.index.get_loc(df["High"].idxmax())
    global_low_idx = df.index.get_loc(df["Low"].idxmin())
    # Candidate FVG must occur after both extremes
    global_extreme_idx = min(global_high_idx, global_low_idx)

    if global_extreme_idx < 5:
        global_extreme_idx = 5

    for i in range(
        global_extreme_idx - 3, len(df)
    ):  # Skip candidates that occur before the later of the global extremes (including extrema candle)
        # Bullish FVG: current candle's low is greater than candle (i-2)'s high
        if df["Low"].iloc[i] > df["High"].iloc[i - 2]:
            # We'll track coverage by scanning forward for the next MIN Low
            next_min = df["Low"].iloc[i]
            is_covered = False
            top_boundary = df["Low"].iloc[i]  # top boundary of bullish gap
            bottom_boundary = df["High"].iloc[i - 2]  # bottom boundary of gap

            for j in range(i + 1, len(df)):
                if df["Low"].iloc[j] < next_min:
                    next_min = df["Low"].iloc[j]
                else:
                    continue

                # If next_min <= bottom_boundary => fully covered
                if next_min <= bottom_boundary:
                    is_covered = True
                    break

            if is_covered:
                continue

            # If partly covered, we can tighten the top boundary to 'next_min'
            # (meaning price only went down to next_min, never fully to bottom_boundary)
            if next_min < top_boundary:
                top_boundary = next_min

            gap_size = top_boundary - df["High"].iloc[i - 2]
            if gap_size < min_fvg_ratio * last_close_price:
                continue

            fvgs.add(
                FVG(
                    i - 2,
                    i,
                    df["High"].iloc[i - 2],
                    top_boundary,
                    "bullish",
                )
            )

        # Bearish FVG: current candle's high is less than candle (i-2)'s low
        elif df["High"].iloc[i] < df["Low"].iloc[i - 2]:
            # We'll track coverage by scanning forward for the next MAX High
            next_max = df["High"].iloc[i]
            is_covered = False
            top_boundary = df["Low"].iloc[i - 2]  # top boundary of gap
            bottom_boundary = df["High"].iloc[i]  # bottom boundary

            for j in range(i + 1, len(df)):
                # Always track the maximum High so far
                if df["High"].iloc[j] > next_max:
                    next_max = df["High"].iloc[j]
                else:
                    continue

                # If next_max >= top_boundary => fully covered
                if next_max >= top_boundary:
                    is_covered = True
                    break

            if is_covered:
                continue

            # If partly covered, we adjust the FVG accordingly
            if next_max > bottom_boundary:
                bottom_boundary = next_max

            gap_size = df["Low"].iloc[i - 2] - bottom_boundary
            if gap_size < min_fvg_ratio * last_close_price:
                continue

            fvgs.add(
                FVG(
                    i - 2,
                    i,
                    df["Low"].iloc[i - 2],
                    bottom_boundary,
                    "bearish",
                )
            )

    return fvgs


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute the Average True Range (ATR) over a specified period.

    ATR is used to measure volatility. Here it helps determine the significance
    of pivots and scales the touch tolerances.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
        period (int): Number of periods for the rolling ATR.

    Returns:
        pd.Series: ATR values computed as the rolling mean of the true range.
    """
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    # True Range is the maximum of the three measures per bar
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def is_swing_high(
    df: pd.DataFrame, i: int, left_bars: int, right_bars: int, min_move: float
) -> bool:
    """
    Determine if the **candle body** at index i qualifies as a swing high pivot.

    Here we consider the candle's body:
      - 'BodyHigh' is defined as max(Open, Close)
      - 'BodyLow' is defined as min(Open, Close)

    A swing high pivot is defined by:
      - The current BodyHigh is greater than the BodyHigh of the previous 'left_bars' candles.
      - The current BodyHigh is greater than or equal to the BodyHigh of the next 'right_bars' candles.
      - The difference between the current BodyHigh and the corresponding BodyLow from neighboring candles
        is at least min_move.

    Parameters:
        df (pd.DataFrame): DataFrame with 'BodyHigh' and 'BodyLow' columns.
        i (int): Index of the potential pivot.
        left_bars (int): Number of candles to the left for validation.
        right_bars (int): Number of candles to the right for validation.
        min_move (float): Minimum required move (typically a fraction of ATR).

    Returns:
        bool: True if conditions are met, otherwise False
    """
    pivot_condition = all(
        df["BodyHigh"].iloc[i] > df["BodyHigh"].iloc[i - j]
        for j in range(1, left_bars + 1)
    ) and all(
        df["BodyHigh"].iloc[i] >= df["BodyHigh"].iloc[i + j]
        for j in range(1, right_bars + 1)
    )
    if not pivot_condition:
        return False

    if (df["BodyHigh"].iloc[i] - df["BodyLow"].iloc[i - left_bars]) < min_move:
        return False
    if (df["BodyHigh"].iloc[i] - df["BodyLow"].iloc[i + right_bars]) < min_move:
        return False

    return True


def is_swing_low(
    df: pd.DataFrame, i: int, left_bars: int, right_bars: int, min_move: float
) -> bool:
    """
    Determine if the candle body at index i qualifies as a swing low pivot.

    A swing low pivot is defined by:
      - The current BodyLow is lower than the BodyLow of the previous 'left_bars' candles.
      - The current BodyLow is lower than or equal to the BodyLow of the next 'right_bars' candles.
      - The difference between the corresponding BodyHigh of neighboring candles and the current BodyLow
        is at least min_move.

    Parameters:
        df (pd.DataFrame): DataFrame with 'BodyHigh' and 'BodyLow' columns.
        i (int): Index of the potential pivot.
        left_bars (int): Number of candles to the left for validation.
        right_bars (int): Number of candles to the right for validation.
        min_move (float): Minimum required move (typically a fraction of ATR).

    Returns:
        bool: True if conditions are met, otherwise False
    """
    pivot_condition = all(
        df["BodyLow"].iloc[i] < df["BodyLow"].iloc[i - j]
        for j in range(1, left_bars + 1)
    ) and all(
        df["BodyLow"].iloc[i] <= df["BodyLow"].iloc[i + j]
        for j in range(1, right_bars + 1)
    )
    if not pivot_condition:
        return False

    if (df["BodyHigh"].iloc[i - left_bars] - df["BodyLow"].iloc[i]) < min_move:
        return False
    if (df["BodyHigh"].iloc[i + right_bars] - df["BodyLow"].iloc[i]) < min_move:
        return False

    return True


def find_pivots(
    df: pd.DataFrame,
    left_bars: int = 2,
    right_bars: int = 2,
    significance_multiplier: float = 0.5,
    atr_period: int = 14,
):
    """
    Identify pivot points (swing highs and swing lows) using an ATR-based fractal method,
    considering only the candle body.

    For each candle (ensuring enough candles exist on both sides), the function computes a
    minimum required move (min_move) as significance_multiplier multiplied by the ATR at that candle.
    If the candle's body qualifies as a pivot (via is_swing_high or is_swing_low), it is recorded.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close' columns.
        left_bars (int): Number of candles to the left required for a valid pivot.
        right_bars (int): Number of candles to the right required for a valid pivot.
        significance_multiplier (float): Multiplier for ATR to set the minimum move threshold.
        atr_period (int): Lookback period for computing the ATR.

    Returns:
        tuple: Two lists containing the detected pivot low prices and pivot high prices (based on canlde body)
    """
    atr_series = compute_atr(df, period=atr_period).fillna(0)
    pivot_lows, pivot_highs = [], []

    # Iterate over the DataFrame, ensuring we have enough candles on either side
    for i in range(left_bars, len(df) - right_bars):
        # Determine the minimum required move for this bar
        min_move = significance_multiplier * atr_series.iloc[i]
        if is_swing_high(df, i, left_bars, right_bars, min_move):
            pivot_highs.append(df["BodyHigh"].iloc[i])
        if is_swing_low(df, i, left_bars, right_bars, min_move):
            pivot_lows.append(df["BodyLow"].iloc[i])

    return pivot_lows, pivot_highs


def detect_liquidity_levels(
    df: pd.DataFrame,
    stdev_multiplier: float,  # default is 0.05 if not provided
    window: int = 200,
    left_bars: int = 2,
    right_bars: int = 2,
    significance_multiplier: float = 0.5,
    atr_period: int = 14,
    min_samples: int = 2,
    min_touches: int = 3,
    atr_touch_multiplier: float = 0.2,
) -> LiquidityLevels:
    """
    Detect liquidity levels using an ATR-based fractal pivot method with DBSCAN clustering,
    considering only the candle body. If a custom stdev_multiplier is used (i.e. not provided),
    the function will adjust it via binary search to ensure that the final number of liquidity levels
    falls between 3 and 8

    The process is as follows:
      1. Limit analysis to the last 'window' candles and compute 'BodyHigh' and 'BodyLow'
      2. Identify significant pivot points (both swing highs and swing lows) using the ATR-based method
      3. Cluster the pivot points using DBSCAN with dynamic epsilon = (std_dev of pivots) * stdev_multiplier
      4. Filter clusters based on "touch frequency": retain a cluster only if the price touches its centroid
         at least 'min_touches' times. The touch tolerance is set as (latest ATR) * atr_touch_multiplier
      5. If a custom stdev_multiplier was used and the resulting number of liquidity levels is not in the specified range,
         adjust stdev_multiplier and recalculate the liquidity levels
      6. Return the final liquidity levels

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['Open', 'High', 'Low', 'Close'].
        stdev_multiplier (float): Multiplier for the standard deviation of pivot levels to set DBSCAN's eps.
        window (int): Number of recent candles to analyze.
        left_bars (int): Number of candles to the left for validating a pivot.
        right_bars (int): Number of candles to the right for validating a pivot.
        significance_multiplier (float): Multiplier for ATR to determine the minimum move for a pivot.
        atr_period (int): Lookback period for ATR calculation.
        min_samples (int): Minimum number of samples for DBSCAN to form a cluster.
        min_touches (int): Minimum number of times price must "touch" a level to be valid.
        atr_touch_multiplier (float): Multiplier for ATR to set the touch tolerance.

    Returns:
        LiquidityLevels: A container object holding all valid liquidity levels
    """
    is_stdev_specified = True
    if not stdev_multiplier:
        is_stdev_specified = False
        stdev_multiplier = 0.05
    df = df.tail(window).copy().reset_index(drop=True)
    df["BodyHigh"] = df[["Open", "Close"]].max(axis=1)
    df["BodyLow"] = df[["Open", "Close"]].min(axis=1)

    pivot_lows, pivot_highs = find_pivots(
        df, left_bars, right_bars, significance_multiplier, atr_period
    )
    all_pivots = pivot_lows + pivot_highs
    if not all_pivots:
        return LiquidityLevels()

    data = np.array(all_pivots).reshape(-1, 1)
    std_dev = np.std(all_pivots)
    eps = std_dev * stdev_multiplier
    if eps <= 0:
        eps = 1e-5

    atr_series = compute_atr(df, atr_period).fillna(0)
    last_atr = atr_series.iloc[-1] if not atr_series.empty else 0.0
    base_touch_tolerance = last_atr * atr_touch_multiplier

    def calc_final_levels(multiplier: float) -> list:
        """
        Given a stdev_multiplier value, perform clustering and touch filtering
        to return the final liquidity level centroids
        """
        trial_eps = std_dev * multiplier
        if trial_eps <= 0:
            trial_eps = 1e-5
        model = DBSCAN(eps=trial_eps, min_samples=min_samples)
        model.fit(data)
        clusters_dict = {}
        for lvl, lbl in zip(all_pivots, model.labels_):
            if lbl == -1:
                continue
            clusters_dict.setdefault(lbl, []).append(lvl)
        if not clusters_dict:
            return []
        clustered_levels = [np.mean(vals) for vals in clusters_dict.values()]

        # Count touches using candle body values
        def count_touches(level):
            count = 0
            for _, row in df.iterrows():
                if (level - base_touch_tolerance) <= row["BodyLow"] <= (
                    level + base_touch_tolerance
                ) or (level - base_touch_tolerance) <= row["BodyHigh"] <= (
                    level + base_touch_tolerance
                ):
                    count += 1
            return count

        final = [lvl for lvl in clustered_levels if count_touches(lvl) >= min_touches]
        return sorted(final)

    final_levels = calc_final_levels(stdev_multiplier)

    # If custom stdev is not specified, adjust it to yield between 3 and 6 liquidity levels
    # Motivation: if we will have too many liquidity levels, using them will not make any sense
    # So we might want either to rely on this logic and make the computer decide, how many
    # liquidity levels it is there, or to go into the "pro" mode and try to specify the multiplier by yourself
    if not is_stdev_specified:
        target_min, target_max = 3, 6
        # Define search range for stdev_multiplier
        low_multiplier, high_multiplier = 0.01, 0.2
        # Perform binary search over 10 iterations
        for _ in range(10):
            trial_multiplier = (low_multiplier + high_multiplier) / 2.0
            trial_levels = calc_final_levels(trial_multiplier)
            count = len(trial_levels)
            if target_min <= count <= target_max:
                stdev_multiplier = trial_multiplier
                final_levels = trial_levels
                break
            elif count < target_min:
                # Too few levels: merge is too aggressive; decrease eps -> lower multiplier
                high_multiplier = trial_multiplier
            else:  # count > target_max
                # Too many levels: clusters are too fragmented; increase multiplier to merge more
                low_multiplier = trial_multiplier

    result = LiquidityLevels()
    for lvl in sorted(final_levels):
        result.add(LiquidityLevel(lvl))

    return result


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

    for i in range(2, len(df)):
        low, high, close = df["Low"].iloc[i], df["High"].iloc[i], df["Close"].iloc[i]

        for level in liquidity_levels.list:
            # Bullish breaker: sweeps support and reverses upward
            if low < level.price and close > level.price:
                breaker_blocks.add(
                    BreakerBlock(block_type="bullish", index=i, zone=(low, high))
                )

    return breaker_blocks


def detect_liquidity_pools(
    df: pd.DataFrame,
    volume_threshold: float = 1.5,  # Volume multiplier above average
    min_pool_size: int = 3,  # Minimum number of candles to form a pool
    atr_multiplier: float = 0.5,  # Multiplier for ATR to determine price range
    atr_period: int = 14,
) -> LiquidityPools:
    """
    Detect liquidity pools based on volume concentration and price action.

    A liquidity pool is identified when:
    1. There is a concentration of volume above the average
    2. Price action stays within a tight range (defined by ATR)
    3. The pattern persists for at least min_pool_size candles

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume'].
        volume_threshold (float): Multiplier above average volume to consider significant.
        min_pool_size (int): Minimum number of candles required to form a pool.
        atr_multiplier (float): Multiplier for ATR to determine the price range.
        atr_period (int): Period for ATR calculation.

    Returns:
        LiquidityPools: A container object holding all detected liquidity pools.
    """
    if "Volume" not in df.columns:
        return LiquidityPools()

    avg_volume = df["Volume"].mean()
    atr = compute_atr(df, period=atr_period)
    price_range = atr * atr_multiplier

    pools = LiquidityPools()
    i = 0
    while i < len(df) - min_pool_size:
        # Check if we have enough volume concentration
        if (
            df["Volume"].iloc[i : i + min_pool_size].mean()
            < avg_volume * volume_threshold
        ):
            i += 1
            continue

        # Calculate price range for potential pool
        high = df["High"].iloc[i : i + min_pool_size].max()
        low = df["Low"].iloc[i : i + min_pool_size].min()

        if high - low > price_range.iloc[i]:
            i += 1
            continue

        # Calculate pool metrics
        pool_price = (high + low) / 2
        pool_volume = df["Volume"].iloc[i : i + min_pool_size].sum()
        pool_strength = min(
            1.0, pool_volume / (avg_volume * min_pool_size * volume_threshold)
        )

        pool = LiquidityPool(pool_price, pool_volume, pool_strength)
        pools.add(pool)

        i += min_pool_size

    return pools
