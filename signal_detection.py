from datetime import timedelta
from typing import List, Dict, Tuple
import pandas as pd  # type: ignore

from helpers import fetch_candles, analyze_data, fetch_data_and_get_indicators
from database import (
    get_user_preferences,
    upsert_user_signal_request,
    delete_user_signal_request,
)
from database import (
    get_chat_id_for_user,
    get_signal_requests,
    user_signal_request_exists,
)
from utils import auto_signal_jobs, create_true_preferences, logger

from plot_build_helpers import plot_price_chart


def detect_support_resistance(
    df: pd.DataFrame, window: int = 20, threshold: float = 0.02
) -> Tuple[List[float], List[float]]:
    """
    Detect support and resistance levels using swing highs/lows and price clustering.

    Args:
        df: DataFrame with OHLCV data
        window: Window size for swing detection
        threshold: Price threshold for clustering

    Returns:
        Tuple of (support_levels, resistance_levels)
    """
    highs = df["High"].values
    lows = df["Low"].values

    # Find swing highs and lows
    swing_highs = []
    swing_lows = []

    for i in range(window, len(df) - window):
        # Check for swing high
        if all(highs[i] > highs[i - window : i]) and all(
            highs[i] > highs[i + 1 : i + window + 1]
        ):
            swing_highs.append(highs[i])
        # Check for swing low
        if all(lows[i] < lows[i - window : i]) and all(
            lows[i] < lows[i + 1 : i + window + 1]
        ):
            swing_lows.append(lows[i])

    # Cluster price levels
    def cluster_levels(levels: List[float], threshold: float) -> List[float]:
        if not levels:
            return []
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]

        for level in levels[1:]:
            if (level - current_cluster[0]) / current_cluster[0] <= threshold:
                current_cluster.append(level)
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]

        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))

        return clusters

    support_levels = cluster_levels(swing_lows, threshold)
    resistance_levels = cluster_levels(swing_highs, threshold)

    return support_levels, resistance_levels


def detect_trend(df: pd.DataFrame, window: int = 20) -> str:
    """
    Detect the current trend using moving averages and price action.

    Returns:
        "bullish", "bearish", or "neutral"
    """
    # Calculate EMAs
    df["EMA20"] = df["Close"].ewm(span=window).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    # Get recent values
    current_price = df["Close"].iloc[-1]
    ema20 = df["EMA20"].iloc[-1]
    ema50 = df["EMA50"].iloc[-1]

    # Calculate trend strength
    price_above_ema20 = current_price > ema20
    ema20_above_ema50 = ema20 > ema50

    # Calculate momentum
    momentum = (current_price - df["Close"].iloc[-window]) / df["Close"].iloc[-window]

    if price_above_ema20 and ema20_above_ema50 and momentum > 0.01:
        return "bullish"
    elif not price_above_ema20 and not ema20_above_ema50 and momentum < -0.01:
        return "bearish"
    else:
        return "neutral"


def analyze_order_blocks(df: pd.DataFrame, window: int = 3) -> List[Dict]:
    """
    Enhanced order block detection with volume confirmation and price action patterns.

    Returns:
        List of order blocks with their properties
    """
    order_blocks = []

    for i in range(window, len(df) - window):
        # Bullish order block conditions
        if (
            df["Close"].iloc[i] < df["Open"].iloc[i]  # Bearish candle
            and df["Close"].iloc[i + 1] > df["Open"].iloc[i + 1]  # Bullish candle
            and df["Volume"].iloc[i]
            > df["Volume"].iloc[i - 1 : i + 1].mean()  # Higher volume
            and df["Low"].iloc[i + 1 : i + window].min() > df["Low"].iloc[i]
        ):  # Price holds above

            order_blocks.append(
                {
                    "type": "bullish",
                    "index": i,
                    "price": df["Low"].iloc[i],
                    "volume": df["Volume"].iloc[i],
                    "strength": df["Volume"].iloc[i]
                    / df["Volume"].iloc[i - 5 : i].mean(),
                }
            )

        # Bearish order block conditions
        elif (
            df["Close"].iloc[i] > df["Open"].iloc[i]  # Bullish candle
            and df["Close"].iloc[i + 1] < df["Open"].iloc[i + 1]  # Bearish candle
            and df["Volume"].iloc[i]
            > df["Volume"].iloc[i - 1 : i + 1].mean()  # Higher volume
            and df["High"].iloc[i + 1 : i + window].max() < df["High"].iloc[i]
        ):  # Price holds below

            order_blocks.append(
                {
                    "type": "bearish",
                    "index": i,
                    "price": df["High"].iloc[i],
                    "volume": df["Volume"].iloc[i],
                    "strength": df["Volume"].iloc[i]
                    / df["Volume"].iloc[i - 5 : i].mean(),
                }
            )

    return order_blocks


def analyze_breaker_blocks(df: pd.DataFrame, window: int = 3) -> List[Dict]:
    """
    Enhanced breaker block detection with volume and price action confirmation.

    Returns:
        List of breaker blocks with their properties
    """
    breaker_blocks = []

    for i in range(window, len(df) - window):
        # Bullish breaker block conditions
        if (
            df["Close"].iloc[i] > df["Open"].iloc[i]  # Bullish candle
            and df["Close"].iloc[i]
            > df["High"].iloc[i - 1 : i].max()  # Breaks above previous high
            and df["Volume"].iloc[i]
            > df["Volume"].iloc[i - 1 : i + 1].mean()  # Higher volume
            and df["Low"].iloc[i + 1 : i + window].min() > df["Low"].iloc[i]
        ):  # Price holds above

            breaker_blocks.append(
                {
                    "type": "bullish",
                    "index": i,
                    "price": df["Low"].iloc[i],
                    "volume": df["Volume"].iloc[i],
                    "strength": df["Volume"].iloc[i]
                    / df["Volume"].iloc[i - 5 : i].mean(),
                    "breakout_size": (
                        df["Close"].iloc[i] - df["High"].iloc[i - 1 : i].max()
                    )
                    / df["High"].iloc[i - 1 : i].max(),
                }
            )

        # Bearish breaker block conditions
        elif (
            df["Close"].iloc[i] < df["Open"].iloc[i]  # Bearish candle
            and df["Close"].iloc[i]
            < df["Low"].iloc[i - 1 : i].min()  # Breaks below previous low
            and df["Volume"].iloc[i]
            > df["Volume"].iloc[i - 1 : i + 1].mean()  # Higher volume
            and df["High"].iloc[i + 1 : i + window].max() < df["High"].iloc[i]
        ):  # Price holds below

            breaker_blocks.append(
                {
                    "type": "bearish",
                    "index": i,
                    "price": df["High"].iloc[i],
                    "volume": df["Volume"].iloc[i],
                    "strength": df["Volume"].iloc[i]
                    / df["Volume"].iloc[i - 5 : i].mean(),
                    "breakout_size": (
                        df["Low"].iloc[i - 1 : i].min() - df["Close"].iloc[i]
                    )
                    / df["Low"].iloc[i - 1 : i].min(),
                }
            )

    return breaker_blocks


def detect_sweep_of_highs(df: pd.DataFrame, window: int = 20) -> bool:
    """
    Detect if price has swept through a previous high.
    A sweep occurs when price moves above a previous high and then reverses.
    """
    if len(df) < window:
        return False

    # Get the last window of data
    recent_data = df.iloc[-window:].copy()
    recent_data.reset_index(drop=True, inplace=True)

    # Find the highest high in the window
    highest_high = recent_data["High"].max()
    highest_high_idx = recent_data["High"].idxmax()

    # Check if price has moved above the highest high and then reversed
    if (
        highest_high_idx < len(recent_data) - 2
    ):  # Ensure we have enough candles after the high
        # Check if price moved above the high and then reversed
        if (
            recent_data["High"].iloc[-1] > highest_high
            and recent_data["Close"].iloc[-1] < highest_high
        ):
            return True

    return False


def detect_sweep_of_lows(df: pd.DataFrame, window: int = 20) -> bool:
    """
    Detect if price has swept through a previous low.
    A sweep occurs when price moves below a previous low and then reverses.
    """
    if len(df) < window:
        return False

    # Get the last window of data
    recent_data = df.iloc[-window:].copy()
    recent_data.reset_index(drop=True, inplace=True)

    # Find the lowest low in the window
    lowest_low = recent_data["Low"].min()
    lowest_low_idx = recent_data["Low"].idxmin()

    # Check if price has moved below the lowest low and then reversed
    if (
        lowest_low_idx < len(recent_data) - 2
    ):  # Ensure we have enough candles after the low
        # Check if price moved below the low and then reversed
        if (
            recent_data["Low"].iloc[-1] < lowest_low
            and recent_data["Close"].iloc[-1] > lowest_low
        ):
            return True

    return False


def detect_structure_break_up(df: pd.DataFrame, window: int = 20) -> bool:
    """
    Detect if price has broken the structure upward.
    A structure break up occurs when price makes a higher high after a series of lower highs.
    """
    if len(df) < window:
        return False

    # Get the last window of data
    recent_data = df.iloc[-window:]

    # Find the highest high before the last candle
    prev_highs = recent_data["High"].iloc[:-1]
    if len(prev_highs) < 3:  # Need at least 3 candles to establish a structure
        return False

    # Check if the last candle made a higher high
    last_high = recent_data["High"].iloc[-1]
    if last_high > prev_highs.max():
        return True

    return False


def detect_structure_break_down(df: pd.DataFrame, window: int = 20) -> bool:
    """
    Detect if price has broken the structure downward.
    A structure break down occurs when price makes a lower low after a series of higher lows.
    """
    if len(df) < window:
        return False

    # Get the last window of data
    recent_data = df.iloc[-window:]

    # Find the lowest low before the last candle
    prev_lows = recent_data["Low"].iloc[:-1]
    if len(prev_lows) < 3:  # Need at least 3 candles to establish a structure
        return False

    # Check if the last candle made a lower low
    last_low = recent_data["Low"].iloc[-1]
    if last_low < prev_lows.min():
        return True

    return False


def detect_bullish_pin_bar(df: pd.DataFrame) -> bool:
    """
    Detect a bullish pin bar (hammer) pattern.
    A bullish pin bar has a small body at the top and a long lower wick.
    """
    if len(df) < 1:
        return False

    last_candle = df.iloc[-1]
    body_size = abs(last_candle["Close"] - last_candle["Open"])
    lower_wick = min(last_candle["Open"], last_candle["Close"]) - last_candle["Low"]
    upper_wick = last_candle["High"] - max(last_candle["Open"], last_candle["Close"])

    # Check if it's a bullish pin bar
    if (
        body_size < lower_wick * 0.3  # Small body compared to lower wick
        and upper_wick < lower_wick * 0.3  # Small upper wick
        and last_candle["Close"] > last_candle["Open"]
    ):  # Bullish close
        return True

    return False


def detect_bearish_engulfing(df: pd.DataFrame) -> bool:
    """
    Detect a bearish engulfing pattern.
    A bearish engulfing pattern occurs when a bearish candle completely engulfs the previous bullish candle.
    """
    if len(df) < 2:
        return False

    prev_candle = df.iloc[-2]
    curr_candle = df.iloc[-1]

    # Check if it's a bearish engulfing pattern
    if (
        prev_candle["Close"] > prev_candle["Open"]  # Previous candle is bullish
        and curr_candle["Close"] < curr_candle["Open"]  # Current candle is bearish
        and curr_candle["Open"]
        > prev_candle["Close"]  # Current open is above previous close
        and curr_candle["Close"] < prev_candle["Open"]
    ):  # Current close is below previous open
        return True

    return False


def generate_price_prediction_signal_proba(
    df: pd.DataFrame, indicators, weights: list = []
) -> Tuple[str, float, float, str]:
    """
    Generates a single-timeframe signal with bullish/bearish/neutral outcome.

    Returns:
        (signal, probability_of_bullish, confidence, reason_str)
    """
    last_close = df["Close"].iloc[-1]
    reasons = []

    # Weights for each condition
    W_BULLISH_OB = 1.0
    W_BEARISH_OB = 1.0
    W_BULLISH_BREAKER = 1.0
    W_BEARISH_BREAKER = 1.0
    W_ABOVE_SUPPORT = 0.7
    W_BELOW_RESISTANCE = 0.7
    W_FVG_ABOVE = 0.5
    W_FVG_BELOW = 0.5
    W_TREND = 0.8
    W_SWEEP_HIGHS = 1.2
    W_SWEEP_LOWS = 1.2
    W_STRUCTURE_BREAK = 1.0
    W_PIN_BAR = 0.8
    W_ENGULFING = 0.8
    W_LIQUIDITY_POOL_ABOVE = 1.2  # Weight for liquidity pool above current price
    W_LIQUIDITY_POOL_BELOW = 1.2  # Weight for liquidity pool below current price
    W_LIQUIDITY_POOL_ROUND = 1.5  # Weight for liquidity pool at round numbers

    if weights and len(weights) == 18:  # Updated for new weights
        W_BULLISH_OB = weights[0]
        W_BEARISH_OB = weights[1]
        W_BULLISH_BREAKER = weights[2]
        W_BEARISH_BREAKER = weights[3]
        W_ABOVE_SUPPORT = weights[4]
        W_BELOW_RESISTANCE = weights[5]
        W_FVG_ABOVE = weights[6]
        W_FVG_BELOW = weights[7]
        W_TREND = weights[8]
        W_SWEEP_HIGHS = weights[9]
        W_SWEEP_LOWS = weights[10]
        W_STRUCTURE_BREAK = weights[11]
        W_PIN_BAR = weights[12]
        W_ENGULFING = weights[13]
        W_LIQUIDITY_POOL_ABOVE = weights[14]
        W_LIQUIDITY_POOL_BELOW = weights[15]
        W_LIQUIDITY_POOL_ROUND = weights[16]

    bullish_score = 0.0
    bearish_score = 0.0

    # Detect trend
    trend = detect_trend(df)
    if trend == "bullish":
        bullish_score += W_TREND
        reasons.append("Price is in a bullish trend")
    elif trend == "bearish":
        bearish_score += W_TREND
        reasons.append("Price is in a bearish trend")

    # Enhanced support/resistance analysis
    support_levels, resistance_levels = detect_support_resistance(df)

    # Find nearest support and resistance
    nearest_support = max([s for s in support_levels if s < last_close], default=None)
    nearest_resistance = min(
        [r for r in resistance_levels if r > last_close], default=None
    )

    if nearest_support:
        support_distance = (last_close - nearest_support) / last_close
        if support_distance < 0.02:  # Within 2% of support
            bullish_score += W_ABOVE_SUPPORT
            reasons.append(f"Price near support level at {nearest_support:.2f}")

    if nearest_resistance:
        resistance_distance = (nearest_resistance - last_close) / last_close
        if resistance_distance < 0.02:  # Within 2% of resistance
            bearish_score += W_BELOW_RESISTANCE
            reasons.append(f"Price near resistance level at {nearest_resistance:.2f}")

    # Enhanced order block analysis
    order_blocks = analyze_order_blocks(df)
    recent_blocks = [block for block in order_blocks if block["index"] >= len(df) - 10]

    for block in recent_blocks:
        if (
            block["type"] == "bullish" and block["strength"] > 1.2
        ):  # Strong bullish block
            bullish_score += W_BULLISH_OB * block["strength"]
            reasons.append(
                f"Strong bullish order block found (strength: {block['strength']:.2f})"
            )
        elif (
            block["type"] == "bearish" and block["strength"] > 1.2
        ):  # Strong bearish block
            bearish_score += W_BEARISH_OB * block["strength"]
            reasons.append(
                f"Strong bearish order block found (strength: {block['strength']:.2f})"
            )

    # Enhanced breaker block analysis
    breaker_blocks = analyze_breaker_blocks(df)
    recent_breakers = [
        block for block in breaker_blocks if block["index"] >= len(df) - 10
    ]

    for block in recent_breakers:
        if (
            block["type"] == "bullish"
            and block["strength"] > 1.2
            and block["breakout_size"] > 0.01
        ):
            bullish_score += (
                W_BULLISH_BREAKER * block["strength"] * (1 + block["breakout_size"])
            )
            reasons.append(
                f"Strong bullish breaker block found (strength: {block['strength']:.2f}, "
                f"breakout: {block['breakout_size']*100:.1f}%)"
            )
        elif (
            block["type"] == "bearish"
            and block["strength"] > 1.2
            and block["breakout_size"] > 0.01
        ):
            bearish_score += (
                W_BEARISH_BREAKER * block["strength"] * (1 + block["breakout_size"])
            )
            reasons.append(
                f"Strong bearish breaker block found (strength: {block['strength']:.2f}, "
                f"breakout: {block['breakout_size']*100:.1f}%)"
            )

    # FVG Logic (Based on Position Relative to Current Price)
    if indicators.fvgs and indicators.fvgs.list:
        for fvg in indicators.fvgs.list:
            # Determine if FVG is above or below the current price
            # For Bullish FVG: start_price is the High of two periods ago, end_price is current Low
            # For Bearish FVG: start_price is the Low of two periods ago, end_price is current High
            # To determine position, compare current price with FVG range
            if last_close > fvg.start_price and last_close > fvg.end_price:
                # FVG is below the current price
                bearish_score += W_FVG_BELOW
                reasons.append("Unfilled FVG below current price")
            elif last_close < fvg.start_price and last_close < fvg.end_price:
                # FVG is above the current price
                bullish_score += W_FVG_ABOVE
                reasons.append("Unfilled FVG above current price")

    # New pattern detection logic
    if detect_sweep_of_highs(df):
        bearish_score += W_SWEEP_HIGHS
        reasons.append("Price swept through previous highs")

    if detect_sweep_of_lows(df):
        bullish_score += W_SWEEP_LOWS
        reasons.append("Price swept through previous lows")

    if detect_structure_break_up(df):
        bullish_score += W_STRUCTURE_BREAK
        reasons.append("Price broke structure upward")

    if detect_structure_break_down(df):
        bearish_score += W_STRUCTURE_BREAK
        reasons.append("Price broke structure downward")

    if detect_bullish_pin_bar(df):
        bullish_score += W_PIN_BAR
        reasons.append("Bullish pin bar pattern detected")

    if detect_bearish_engulfing(df):
        bearish_score += W_ENGULFING
        reasons.append("Bearish engulfing pattern detected")

    # Enhanced liquidity pool analysis
    if indicators.liquidity_pools and indicators.liquidity_pools.list:
        # Define round numbers based on the current price level
        current_price = last_close
        price_magnitude = len(str(int(current_price)))
        round_numbers = [
            round(current_price / 10**i) * 10**i
            for i in range(price_magnitude - 1, price_magnitude + 2)
        ]

        # Find local maximums and minimums for context
        local_max_window = 20  # Window for detecting local extremes
        local_maximums = []
        local_minimums = []

        for i in range(local_max_window, len(df) - local_max_window):
            # Check for local maximum
            if all(
                df["High"].iloc[i] > df["High"].iloc[i - local_max_window : i]
            ) and all(
                df["High"].iloc[i] > df["High"].iloc[i + 1 : i + local_max_window + 1]
            ):
                local_maximums.append((i, df["High"].iloc[i]))

            # Check for local minimum
            if all(
                df["Low"].iloc[i] < df["Low"].iloc[i - local_max_window : i]
            ) and all(
                df["Low"].iloc[i] < df["Low"].iloc[i + 1 : i + local_max_window + 1]
            ):
                local_minimums.append((i, df["Low"].iloc[i]))

        for pool in indicators.liquidity_pools.list:
            # Calculate distance to current price as a percentage
            distance = abs(pool.price - last_close) / last_close

            # Skip if pool is too far from current price
            if distance > 0.05:  # 5% threshold
                continue

            # Determine if pool is at a round number
            is_round_number = any(
                abs(pool.price - round_num) / round_num < 0.001
                for round_num in round_numbers
            )

            # Calculate pool impact based on strength and volume
            pool_impact = pool.strength * (pool.volume / df["Volume"].mean())

            # Generate detailed reason based on pool characteristics
            pool_reason = []

            # Check if pool is near a local maximum or minimum
            is_near_max = any(
                abs(pool.price - max_price) / max_price < 0.001
                for _, max_price in local_maximums[-5:]  # Check last 5 maximums
            )
            is_near_min = any(
                abs(pool.price - min_price) / min_price < 0.001
                for _, min_price in local_minimums[-5:]  # Check last 5 minimums
            )

            # Add local extreme context
            if is_near_max:
                pool_reason.append("Located at recent local maximum")
            elif is_near_min:
                pool_reason.append("Located at recent local minimum")

            # Add round number context if applicable
            if is_round_number:
                pool_reason.append(f"Round number level {pool.price:.2f}")

            # Add volume context
            volume_ratio = pool.volume / df["Volume"].mean()
            if volume_ratio > 1.5:
                pool_reason.append("High volume concentration")
            elif volume_ratio > 1.2:
                pool_reason.append("Above average volume")

            # Add strength context
            if pool.strength > 0.8:
                pool_reason.append("Very strong pool")
            elif pool.strength > 0.6:
                pool_reason.append("Strong pool")

            # Add distance context
            if distance < 0.01:
                pool_reason.append("Very close to current price")
            elif distance < 0.02:
                pool_reason.append("Close to current price")

            if last_close > pool.price:
                # Pool is below current price - potential support
                if is_round_number:
                    bullish_score += W_LIQUIDITY_POOL_ROUND * pool_impact
                    reasons.append(
                        f"Strong liquidity pool at round number {pool.price:.2f} below price "
                        f"(strength: {pool.strength:.2f}, volume: {pool.volume:.2f})"
                        f"\n  • {' | '.join(pool_reason)}"
                    )
                else:
                    bullish_score += W_LIQUIDITY_POOL_BELOW * pool_impact
                    reasons.append(
                        f"Liquidity pool at {pool.price:.2f} below price "
                        f"(strength: {pool.strength:.2f}, volume: {pool.volume:.2f})"
                        f"\n  • {' | '.join(pool_reason)}"
                    )
            else:
                # Pool is above current price - potential resistance
                if is_round_number:
                    bearish_score += W_LIQUIDITY_POOL_ROUND * pool_impact
                    reasons.append(
                        f"Strong liquidity pool at round number {pool.price:.2f} above price "
                        f"(strength: {pool.strength:.2f}, volume: {pool.volume:.2f})"
                        f"\n  • {' | '.join(pool_reason)}"
                    )
                else:
                    bearish_score += W_LIQUIDITY_POOL_ABOVE * pool_impact
                    reasons.append(
                        f"Liquidity pool at {pool.price:.2f} above price "
                        f"(strength: {pool.strength:.2f}, volume: {pool.volume:.2f})"
                        f"\n  • {' | '.join(pool_reason)}"
                    )

    # Convert scores to final probability
    eps = 1e-9
    total_score = bullish_score + bearish_score
    if total_score < eps:
        probability_of_bullish = 0.5
    else:
        probability_of_bullish = bullish_score / total_score

    # Clamp probability
    probability_of_bullish = max(0.001, min(probability_of_bullish, 0.999))

    # Decide final signal with trend confirmation
    if probability_of_bullish >= 0.66 and trend != "bearish":
        signal = "Bullish"
    elif probability_of_bullish <= 0.33 and trend != "bullish":
        signal = "Bearish"
    else:
        signal = "Neutral"

    # Calculate confidence
    confidence = abs(probability_of_bullish - 0.5) * 2.0

    # Compile reason string
    reason_str = (
        f"Signal: {signal}\n"
        f"Trend: {trend}\n"
        f"Bullish Score: {bullish_score:.2f} | Bearish Score: {bearish_score:.2f}\n"
        f"Probability of Bullish: {probability_of_bullish:.3f}\n"
        f"Confidence: {confidence:.3f}\n\n"
    )
    if reasons:
        reason_str += "Reasons:\n- " + "\n- ".join(reasons)

    return signal, probability_of_bullish, confidence, reason_str


###############################################################################
# Multi-timeframe Analysis
###############################################################################
async def multi_timeframe_analysis(
    symbol: str,
    preferences: Dict[str, bool],
    timeframes: List[str],
    candles_per_tf: int = 300,
    liq_lev_tolerance: float = 0.05,
) -> Dict[str, Dict[str, any]]:
    """
    Fetches OHLCV data for multiple timeframes (e.g., 1h, 4h, 1d) and analyzes
    them using your 'analyze_data' function.

    Parameters:
        symbol (str): The symbol/currency pair to fetch (e.g. "BTCUSDT")
        preferences (dict): The user indicator preferences (e.g., order_blocks, fvgs, etc.)
        timeframes (List[str]): List of timeframes to analyze (e.g., ["1h", "4h", "1d"])
        candles_per_tf (int): How many candles to fetch for each timeframe
        liq_lev_tolerance (float): The tolerance for liquidity level detection

    Returns:
        A dictionary containing, for each timeframe:
            {
                "df": pandas DataFrame with the OHLCV data,
                "indicators": your Indicators object
            }
    """
    mtf_results = {}

    for tf in timeframes:
        # 1) Fetch historical data for the desired timeframe
        df = fetch_candles(symbol, candles_per_tf, tf)

        if df is None or df.empty:
            # In case no data is returned or an error happened, skip
            logger.info(f"[multi_timeframe_analysis] No data for {symbol} on {tf}")
            continue

        # 2) Analyze the data using your existing 'analyze_data' function,
        #    which returns an Indicators() object with order blocks, FVG, etc.
        indicators = analyze_data(df, preferences, liq_lev_tolerance)

        # 3) Store the results
        mtf_results[tf] = {"df": df, "indicators": indicators}

    return mtf_results


###############################################################################
# Multi-timeframe Aggregation of Signals
###############################################################################
def generate_multi_tf_signal_proba(
    mtf_results: Dict[str, Dict[str, any]],
) -> (str, float, float, str):  # type: ignore
    """
    Aggregates signals from multiple timeframes. For each timeframe, we use
    'generate_price_prediction_signal_proba()' to produce an individual signal
    (Bullish/Bearish/Neutral) along with a probability of bullishness.

    Then we weight them together for a final probability & final signal.

    Parameters:
        mtf_results (dict): As returned by `multi_timeframe_analysis`.
            Example format:
            {
                "1h": {
                    "df": <DataFrame>,
                    "indicators": <Indicators>
                },
                "4h": {
                    "df": <DataFrame>,
                    "indicators": <Indicators>
                },
                ...
            }

    Returns:
        (final_signal, final_probability_of_bullish, confidence, reason_str)

        - final_signal (str): "Bullish", "Bearish", or "Neutral"
        - final_probability_of_bullish (float): aggregated probability in [0.001..0.999]
        - confidence (float): how far from 0.5 the final probability is, scaled to [0..1]
        - reason_str (str): textual explanation that includes breakdown from each timeframe
    """
    # Example weighting for each timeframe:
    # You can tweak these or make them user-configurable.
    timeframe_weights = {
        "1m": 0.10,  # Just an example if you want 1m
        "5m": 0.15,
        "15m": 0.20,
        "1h": 0.25,
        "4h": 0.30,
        "1d": 0.40,
        "1w": 0.50,
    }

    reasons = []
    total_weight = 0.0
    weighted_prob_sum = 0.0

    # 1) Loop through each timeframe's results
    for tf, data in mtf_results.items():
        df = data["df"]
        indicators = data["indicators"]

        # Use your existing single-timeframe function to get the signal
        signal, prob_bullish, confidence, reason_str = (
            generate_price_prediction_signal_proba(df, indicators)
        )

        # 2) Retrieve a weight for that timeframe, default to 0.2 if not specified
        w = timeframe_weights.get(tf, 0.2)
        total_weight += w

        # 3) Accumulate weighted probability
        weighted_prob_sum += prob_bullish * w

        # 4) Collect textual explanation for each timeframe
        reasons.append(
            f"Timeframe: {tf}\n"
            f"Signal: {signal}\n"
            f"Probability of Bullish: {prob_bullish:.3f}\n"
            f"Confidence: {confidence:.3f}\n"
            f"Reasons:\n{reason_str}\n"
            "-----------------------------------------"
        )

    # If no data/timeframes processed, return a default "Neutral" signal
    if total_weight == 0:
        return (
            "Neutral",
            0.5,
            0.0,
            "No timeframes data were available, defaulting to Neutral",
        )

    # 5) Compute the final aggregated bullish probability
    final_prob = weighted_prob_sum / total_weight

    # 6) Clamp the probability within [0.001..0.999]
    final_prob = max(0.001, min(final_prob, 0.999))

    # 7) Determine final signal by thresholds
    if final_prob >= 0.66:
        final_signal = "Bullish"
    elif final_prob <= 0.33:
        final_signal = "Bearish"
    else:
        final_signal = "Neutral"

    # 8) Calculate overall confidence
    #    (distance from 0.5 scaled to [0..1])
    confidence = abs(final_prob - 0.5) * 2.0

    # 9) Combine all timeframe reasons into one final reason string
    final_reasons_str = (
        f"Final Aggregated Signal: {final_signal}\n"
        f"Aggregated Probability of Bullish: {final_prob:.3f}\n"
        f"Confidence: {confidence:.3f}\n\n"
        "Detailed breakdown by timeframe:\n" + "\n".join(reasons)
    )

    return final_signal, final_prob, confidence, final_reasons_str


###############################################################################
# Updated Auto-Signal Job with Multi-Timeframe Analysis
###############################################################################
async def auto_signal_job(context):
    """
    This function is called periodically by the Telegram JobQueue (run_repeating).
    Instead of analyzing a single timeframe, it uses the multi-timeframe analysis
    to produce a more reliable signal. Then it decides whether to send a message.
    """
    job_data = context.job.data
    user_id = job_data["user_id"]
    chat_id = job_data["chat_id"]
    currency_pair = job_data["currency_pair"]
    is_with_chart = job_data["is_with_chart"]

    # 1) Fetch user preferences from the database
    preferences = get_user_preferences(user_id)

    # If the user has not selected any indicators, enable all by default
    if all(not value for value in preferences.values()):
        preferences = {k: True for k in preferences}

    # 2) Perform multi-timeframe analysis
    #    For example, let's fetch "15m", "1h", "4h" data. Adjust as needed.
    mtf_results = await multi_timeframe_analysis(
        symbol=currency_pair,
        preferences=preferences,
        timeframes=["15m", "1h", "4h"],
        candles_per_tf=300,
        liq_lev_tolerance=0.05,
    )

    # If we have no data/timeframes, abort
    if not mtf_results:
        await context.bot.send_message(
            chat_id=chat_id, text=f"[Auto-Signal] No data found for {currency_pair}."
        )
        return

    # 3) Generate an aggregated signal from all timeframes
    final_signal, final_prob, confidence, reason_str = generate_multi_tf_signal_proba(
        mtf_results
    )

    # 4) Decide if we want to send the signal to the user
    #    For instance, we can require a minimum confidence or a non-neutral signal
    if confidence > 0.0 or final_signal != "Neutral":
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(f"[Auto-Signal for {currency_pair}]\n\n" f"{reason_str}"),
            )
            if is_with_chart:
                interval_count = 200
                interval = "1h"
                input = [currency_pair, interval_count, interval]
                (indicators, df) = await fetch_data_and_get_indicators(
                    input, create_true_preferences(), ()
                )

                chart_path = plot_price_chart(
                    df,
                    indicators,
                    show_legend=preferences["show_legend"],
                    show_volume=preferences["show_volume"],
                )

                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"Below is a chart for {currency_pair} for the last {interval_count} intervals with {interval} interval:"
                    ),
                )

                # Send the chart to the user
                with open(chart_path, "rb") as chart_file:
                    await context.bot.send_photo(chat_id=chat_id, photo=chart_file)
        except Exception as e:
            logger.info(
                f"Error sending auto-signal message to user {user_id}: {str(e)}"
            )
    else:
        # Optionally, no message is sent if the signal is too weak
        pass


###############################################################################
# Creating and Deleting Signal Jobs (Example usage remains similar)
###############################################################################
async def createSignalJob(
    symbol: str, period_minutes: int, is_with_chart: bool, update, context
):
    """
    Creates a repeating job for auto-signal analysis (multi-timeframe).
    The code below is largely the same as your existing function.
    """
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if user_signal_request_exists(user_id, symbol):
        await update.message.reply_text(
            f"❌ You already have an auto-signal for {symbol}. "
            "Please delete it first with /delete_signal or choose another pair."
        )
        return

    # Update or insert into DB
    signals_request = {
        "currency_pair": symbol,
        "frequency_minutes": period_minutes,
        "is_with_chart": is_with_chart,
    }
    upsert_user_signal_request(user_id, signals_request)

    job_key = (user_id, symbol)

    # If there's an existing job for the same user & symbol, remove it
    if job_key in auto_signal_jobs:
        old_job = auto_signal_jobs[job_key]
        old_job.schedule_removal()
        del auto_signal_jobs[job_key]

    # Create a new repeating job
    job_ref = context.application.job_queue.run_repeating(
        callback=auto_signal_job,
        interval=timedelta(minutes=period_minutes),
        first=0,
        name=f"signal_job_{user_id}_{symbol}",
        data={
            "user_id": user_id,
            "chat_id": chat_id,
            "currency_pair": symbol,
            "is_with_chart": is_with_chart,
        },
    )

    # Save the job reference
    auto_signal_jobs[job_key] = job_ref

    await update.message.reply_text(
        f"✅ Auto-signals started for {symbol}, every {period_minutes} minute(s)."
    )


async def deleteSignalJob(currency_pair, update):
    """
    Stops a specific user's auto-signal job for a given symbol.
    """
    user_id = update.effective_user.id

    # Remove from the database
    delete_user_signal_request(user_id, currency_pair)

    job_key = (user_id, currency_pair)
    if job_key in auto_signal_jobs:
        job_ref = auto_signal_jobs[job_key]
        job_ref.schedule_removal()
        del auto_signal_jobs[job_key]
        await update.message.reply_text(f"✅ Auto-signals for {currency_pair} stopped.")
    else:
        await update.message.reply_text(f"No auto-signals running for {currency_pair}.")


###############################################################################
# Initialization of all Jobs at Startup (remains as in your code)
###############################################################################
async def initialize_jobs(application):
    """
    Called once at bot start-up to restore all jobs from the database.
    """
    signal_requests = get_signal_requests()

    for req in signal_requests:
        user_id = req["user_id"]
        currency_pair = req["currency_pair"]
        frequency_minutes = req["frequency_minutes"]
        chat_id = get_chat_id_for_user(user_id)

        if not chat_id:
            logger.info(f"No chat_id found for user {user_id}. Skipping job creation.")
            continue

        job_key = (user_id, currency_pair)
        if job_key in auto_signal_jobs:
            logger.info(
                f"Job for user_id {user_id}, pair {currency_pair} already exists."
            )
            continue

        # Create a job
        job_data = {
            "user_id": user_id,
            "chat_id": chat_id,
            "currency_pair": currency_pair,
        }
        job_ref = application.job_queue.run_repeating(
            callback=auto_signal_job,
            interval=timedelta(minutes=frequency_minutes),
            first=0,
            name=f"signal_job_{user_id}_{currency_pair}",
            data=job_data,
        )

        auto_signal_jobs[job_key] = job_ref

    logger.info("All user signal jobs have been initialized.")
