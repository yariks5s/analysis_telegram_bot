import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from IndicatorUtils.order_block_utils import OrderBlock, OrderBlocks
from IndicatorUtils.fvg_utils import FVG, FVGs
from IndicatorUtils.liquidity_level_utils import LiquidityLevel, LiquidityLevels
from IndicatorUtils.breaker_block_utils import BreakerBlock, BreakerBlocks

from sklearn.cluster import DBSCAN  # type: ignore

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
    # Calculate differences needed for True Range
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
    # Check if current high is greater than previous left_bars highs and at least equal to subsequent right_bars highs
    pivot_condition = all(
        df["BodyHigh"].iloc[i] > df["BodyHigh"].iloc[i - j]
        for j in range(1, left_bars + 1)
    ) and all(
        df["BodyHigh"].iloc[i] >= df["BodyHigh"].iloc[i + j]
        for j in range(1, right_bars + 1)
    )
    if not pivot_condition:
        return False

    # Ensure the price difference is significant compared to ATR-based min_move
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
    # Check if current low is less than previous left_bars lows and less than or equal to subsequent right_bars lows
    pivot_condition = all(
        df["BodyLow"].iloc[i] < df["BodyLow"].iloc[i - j]
        for j in range(1, left_bars + 1)
    ) and all(
        df["BodyLow"].iloc[i] <= df["BodyLow"].iloc[i + j]
        for j in range(1, right_bars + 1)
    )
    if not pivot_condition:
        return False

    # Ensure the price difference is significant compared to ATR-based min_move
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
    # Compute ATR for volatility; ATR uses the full candle range
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
    # Determine if we need to auto-adjust stdev_multiplier or it is set by user and we should not touch it
    is_stdev_specified = True
    if not stdev_multiplier:
        is_stdev_specified = False
        stdev_multiplier = 0.05

    # 1. Limit analysis to the last 'window' candles and compute body values
    df = df.tail(window).copy().reset_index(drop=True)
    df["BodyHigh"] = df[["Open", "Close"]].max(axis=1)
    df["BodyLow"] = df[["Open", "Close"]].min(axis=1)

    # 2. Identify pivot lows and pivot highs based on candle body
    pivot_lows, pivot_highs = find_pivots(
        df, left_bars, right_bars, significance_multiplier, atr_period
    )
    all_pivots = pivot_lows + pivot_highs
    if not all_pivots:
        return LiquidityLevels()

    # Pre-calculate some values that remain constant for different stdev_multiplier trials
    data = np.array(all_pivots).reshape(-1, 1)
    std_dev = np.std(all_pivots)
    # Calculate eps; if std_dev is zero, set eps to a small positive value to avoid DBSCAN error
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

    # Initially calculate final levels with the given stdev_multiplier
    final_levels = calc_final_levels(stdev_multiplier)

    # 3. If custom stdev is not specified, adjust it to yield between 3 and 6 liquidity levels
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

    # 4. Wrap the final levels into the LiquidityLevels container
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

    # Iterate over each candle
    for i in range(2, len(df)):
        low, high, close = df["Low"].iloc[i], df["High"].iloc[i], df["Close"].iloc[i]

        # Check for bullish and bearish breaker blocks
        for level in liquidity_levels.list:
            # Bullish breaker: sweeps support and reverses upward
            if low < level.price and close > level.price:
                breaker_blocks.add(
                    BreakerBlock(block_type="bullish", index=i, zone=(low, high))
                )

    return breaker_blocks
