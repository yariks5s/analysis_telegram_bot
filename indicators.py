import pandas as pd  # type: ignore
import numpy as np # type: ignore

from IndicatorUtils.order_block_utils import OrderBlock, OrderBlocks
from IndicatorUtils.fvg_utils import FVG, FVGs
from IndicatorUtils.liquidity_level_utils import LiquidityLevel, LiquidityLevels
from IndicatorUtils.breaker_block_utils import BreakerBlock, BreakerBlocks

from sklearn.cluster import DBSCAN # type: ignore
from typing import Tuple, List

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


def is_local_min(df: pd.DataFrame, i: int, n: int = 3) -> bool:
    return all(df["Low"].iloc[i] < df["Low"].iloc[i - j] and df["Low"].iloc[i] < df["Low"].iloc[i + j] for j in range(1, n + 1))

def is_local_max(df: pd.DataFrame, i: int, n: int = 3) -> bool:
    return all(df["High"].iloc[i] > df["High"].iloc[i - j] and df["High"].iloc[i] > df["High"].iloc[i + j] for j in range(1, n + 1))

def find_extrema(df: pd.DataFrame, n: int = 3):
    supports, resistances = [], []
    for i in range(n, len(df) - n):
        if is_local_min(df, i, n):
            supports.append(df["Low"].iloc[i])
        if is_local_max(df, i, n):
            resistances.append(df["High"].iloc[i])
    return supports, resistances

def cluster_levels(levels, eps: float = 0.03, min_samples: int = 2):
    if not levels:
        return []

    data = np.array(levels).reshape(-1, 1)
    model = DBSCAN(eps=eps * np.mean(levels), min_samples=min_samples)
    model.fit(data)

    clusters = {}
    for level, label in zip(levels, model.labels_):
        if label == -1:
            continue  # noise
        clusters.setdefault(label, []).append(level)

    clustered = [np.mean(cluster) for cluster in clusters.values()]
    return sorted(clustered)

def count_touches(df: pd.DataFrame, level: float, tolerance: float = 0.005) -> int:
    touches = 0
    for _, row in df.iterrows():
        if abs(row["Low"] - level) / level < tolerance or abs(row["High"] - level) / level < tolerance:
            touches += 1
    return touches

def filter_by_touch_frequency(df, levels, min_touches: int = 2, tolerance: float = 0.005):
    return [level for level in levels if count_touches(df, level, tolerance) >= min_touches]


def detect_support_resistance_levels(
    df: pd.DataFrame,
    window: int = 200,
    extrema_window: int = 3,
    eps: float = 0.01,
    min_cluster_size: int = 2,
    min_touches: int = 2,
    tolerance: float = 0.005
) -> LiquidityLevels:
    """
    Detects support and resistance levels and returns LiquidityLevels object.
    """
    df = df.tail(window).copy().reset_index(drop=True)
    supports, resistances = find_extrema(df, n=extrema_window)

    clustered_supports = cluster_levels(supports, eps=eps, min_samples=min_cluster_size)
    clustered_resistances = cluster_levels(resistances, eps=eps, min_samples=min_cluster_size)

    filtered_supports = filter_by_touch_frequency(df, clustered_supports, min_touches, tolerance)
    filtered_resistances = filter_by_touch_frequency(df, clustered_resistances, min_touches, tolerance)

    result = LiquidityLevels()
    for level in filtered_supports:
        result.add(LiquidityLevel("support", level))
    for level in filtered_resistances:
        result.add(LiquidityLevel("resistance", level))

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
