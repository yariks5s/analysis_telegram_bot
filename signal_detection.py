from datetime import timedelta
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from dataclasses import dataclass
from enum import Enum

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


@dataclass
class TradingSignal:
    """Enhanced signal class with risk management"""

    symbol: str
    signal_type: str  # "Bullish", "Bearish", "Neutral"
    probability: float
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_reward_ratio: float
    position_size: float
    max_risk_amount: float
    reasons: List[str]
    market_conditions: Dict[str, any]
    timestamp: pd.Timestamp


class MarketRegime(Enum):
    """Market regime classification"""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range for volatility"""
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI for momentum confirmation"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_market_structure_score(df: pd.DataFrame) -> float:
    """Calculate market structure score based on higher highs/lows"""
    window = 20
    highs = df["High"].rolling(window=window).max()
    lows = df["Low"].rolling(window=window).min()

    # Count higher highs and higher lows
    hh = (highs.diff() > 0).sum()
    hl = (lows.diff() > 0).sum()

    # Count lower highs and lower lows
    lh = (highs.diff() < 0).sum()
    ll = (lows.diff() < 0).sum()

    # Calculate structure score (-1 to 1)
    bullish_score = (hh + hl) / (window * 2)
    bearish_score = (lh + ll) / (window * 2)

    return bullish_score - bearish_score


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
    """Detect trend using EMAs"""
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()

    # Calculate EMAs using .loc to avoid the warning
    df.loc[:, "EMA20"] = df["Close"].ewm(span=window).mean()
    df.loc[:, "EMA50"] = df["Close"].ewm(span=50).mean()

    # Get the last values
    ema20 = df["EMA20"].iloc[-1]
    ema50 = df["EMA50"].iloc[-1]

    # Determine trend
    if ema20 > ema50:
        return "uptrend"
    elif ema20 < ema50:
        return "downtrend"
    else:
        return "sideways"


# Enhanced market regime detection
def detect_market_regime(df: pd.DataFrame) -> MarketRegime:
    """Detect current market regime for better signal filtering"""
    # Calculate volatility
    returns = df["Close"].pct_change()
    volatility = returns.rolling(window=20).std().iloc[-1]
    avg_volatility = returns.rolling(window=100).std().mean()

    # Calculate trend strength using ADX
    atr = calculate_atr(df)
    dm_plus = df["High"].diff()
    dm_minus = -df["Low"].diff()

    dm_plus = dm_plus.where(dm_plus > 0, 0)
    dm_minus = dm_minus.where(dm_minus > 0, 0)

    di_plus = 100 * (dm_plus.rolling(14).mean() / atr)
    di_minus = 100 * (dm_minus.rolling(14).mean() / atr)

    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(14).mean().iloc[-1]

    # Determine regime
    if volatility > avg_volatility * 1.5:
        return MarketRegime.VOLATILE
    elif volatility < avg_volatility * 0.5:
        return MarketRegime.QUIET
    elif adx > 25:
        if di_plus.iloc[-1] > di_minus.iloc[-1]:
            return MarketRegime.TRENDING_UP
        else:
            return MarketRegime.TRENDING_DOWN
    else:
        return MarketRegime.RANGING


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
            and df["Close"].iloc[i]
            < df["Low"].iloc[i - 1 : i].min()  # Breaks below previous low
            and df["Volume"].iloc[i]
            > df["Volume"].iloc[i - 1 : i + 1].mean()  # Higher volume
            and df["High"].iloc[i + 1 : i + window].max() > df["High"].iloc[i]
        ):  # Price holds above

            order_blocks.append(
                {
                    "type": "bullish",
                    "index": i,
                    "price": df["Low"].iloc[i],
                    "volume": df["Volume"].iloc[i],
                    "strength": df["Volume"].iloc[i]
                    / max(df["Volume"].iloc[i - 5 : i].mean(), 1e-10),
                }
            )

        # Bearish order block conditions
        elif (
            df["Close"].iloc[i] > df["Open"].iloc[i]  # Bullish candle
            and df["Close"].iloc[i]
            > df["High"].iloc[i - 1 : i].max()  # Breaks above previous high
            and df["Volume"].iloc[i]
            > df["Volume"].iloc[i - 1 : i + 1].mean()  # Higher volume
            and df["Low"].iloc[i + 1 : i + window].min() < df["Low"].iloc[i]
        ):  # Price holds below

            order_blocks.append(
                {
                    "type": "bearish",
                    "index": i,
                    "price": df["High"].iloc[i],
                    "volume": df["Volume"].iloc[i],
                    "strength": df["Volume"].iloc[i]
                    / max(df["Volume"].iloc[i - 5 : i].mean(), 1e-10),
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

            prev_high = df["High"].iloc[i - 1 : i].max()
            breakout_size = (df["Close"].iloc[i] - prev_high) / max(prev_high, 1e-10)

            breaker_blocks.append(
                {
                    "type": "bullish",
                    "index": i,
                    "price": df["Low"].iloc[i],
                    "volume": df["Volume"].iloc[i],
                    "strength": df["Volume"].iloc[i]
                    / max(df["Volume"].iloc[i - 5 : i].mean(), 1e-10),
                    "breakout_size": breakout_size,
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

            prev_low = df["Low"].iloc[i - 1 : i].min()
            breakout_size = (prev_low - df["Close"].iloc[i]) / max(prev_low, 1e-10)

            breaker_blocks.append(
                {
                    "type": "bearish",
                    "index": i,
                    "price": df["High"].iloc[i],
                    "volume": df["Volume"].iloc[i],
                    "strength": df["Volume"].iloc[i]
                    / max(df["Volume"].iloc[i - 5 : i].mean(), 1e-10),
                    "breakout_size": breakout_size,
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


# Risk Management Functions
def calculate_dynamic_stop_loss(
    df: pd.DataFrame, signal_type: str, entry_price: float, atr_multiplier: float = 2.0
) -> float:
    """Calculate dynamic stop loss based on ATR and market structure"""
    atr = calculate_atr(df)
    current_atr = atr.iloc[-1]

    # Find recent support/resistance levels
    support_levels, resistance_levels = detect_support_resistance(df)

    if signal_type == "Bullish":
        # ATR-based stop
        atr_stop = entry_price - (current_atr * atr_multiplier)

        # Structure-based stop (below recent support)
        structure_stop = max([s for s in support_levels if s < entry_price], default=0)
        if structure_stop > 0:
            structure_stop *= 0.995  # 0.5% below support

        # Use the higher of the two (closer to entry)
        stop_loss = max(atr_stop, structure_stop) if structure_stop > 0 else atr_stop

    else:  # Bearish
        # ATR-based stop
        atr_stop = entry_price + (current_atr * atr_multiplier)

        # Structure-based stop (above recent resistance)
        structure_stop = min(
            [r for r in resistance_levels if r > entry_price], default=float("inf")
        )
        if structure_stop != float("inf"):
            structure_stop *= 1.005  # 0.5% above resistance

        # Use the lower of the two (closer to entry)
        stop_loss = (
            min(atr_stop, structure_stop)
            if structure_stop != float("inf")
            else atr_stop
        )

    return stop_loss


def calculate_take_profit_levels(
    df: pd.DataFrame,
    signal_type: str,
    entry_price: float,
    stop_loss: float,
    risk_reward_ratios: List[float] = [1.5, 2.0, 3.0],
) -> Tuple[float, float, float]:
    """Calculate multiple take profit levels based on R:R and market structure"""
    risk = abs(entry_price - stop_loss)

    # Basic R:R based targets
    if signal_type == "Bullish":
        tp1 = entry_price + (risk * risk_reward_ratios[0])
        tp2 = entry_price + (risk * risk_reward_ratios[1])
        tp3 = entry_price + (risk * risk_reward_ratios[2])
    else:
        tp1 = entry_price - (risk * risk_reward_ratios[0])
        tp2 = entry_price - (risk * risk_reward_ratios[1])
        tp3 = entry_price - (risk * risk_reward_ratios[2])

    # Adjust based on market structure
    support_levels, resistance_levels = detect_support_resistance(df)

    if signal_type == "Bullish":
        # Find resistance levels that could act as targets
        potential_targets = [r for r in resistance_levels if r > entry_price]
        if potential_targets:
            # Adjust targets to be just below resistance, but ensure they're still above entry
            tp1 = max(
                entry_price * 1.001,  # Ensure at least 0.1% above entry
                (
                    min(tp1, potential_targets[0] * 0.995)
                    if len(potential_targets) > 0
                    else tp1
                ),
            )
            tp2 = max(
                tp1 * 1.001,  # Ensure at least 0.1% above tp1
                (
                    min(tp2, potential_targets[1] * 0.995)
                    if len(potential_targets) > 1
                    else tp2
                ),
            )
            tp3 = max(
                tp2 * 1.001,  # Ensure at least 0.1% above tp2
                (
                    min(tp3, potential_targets[2] * 0.995)
                    if len(potential_targets) > 2
                    else tp3
                ),
            )
    else:
        # Find support levels that could act as targets
        potential_targets = [s for s in support_levels if s < entry_price]
        if potential_targets:
            # Adjust targets to be just above support, but ensure they're still below entry
            tp1 = min(
                entry_price * 0.999,  # Ensure at least 0.1% below entry
                (
                    max(tp1, potential_targets[0] * 1.005)
                    if len(potential_targets) > 0
                    else tp1
                ),
            )
            tp2 = min(
                tp1 * 0.999,  # Ensure at least 0.1% below tp1
                (
                    max(tp2, potential_targets[1] * 1.005)
                    if len(potential_targets) > 1
                    else tp2
                ),
            )
            tp3 = min(
                tp2 * 0.999,  # Ensure at least 0.1% below tp2
                (
                    max(tp3, potential_targets[2] * 1.005)
                    if len(potential_targets) > 2
                    else tp3
                ),
            )

    return tp1, tp2, tp3


def calculate_position_size(
    account_balance: float, risk_percentage: float, entry_price: float, stop_loss: float
) -> Dict[str, float]:
    """Calculate position size based on account risk"""
    risk_amount = account_balance * (risk_percentage / 100)
    price_difference = abs(entry_price - stop_loss)
    position_size = risk_amount / price_difference

    return {
        "position_size": position_size,
        "position_value": position_size * entry_price,
        "risk_amount": risk_amount,
        "risk_percentage": risk_percentage,
    }


# Market context analysis
def analyze_market_correlation(df: pd.DataFrame, btc_df: pd.DataFrame = None) -> Dict:
    """Analyze correlation with BTC and market conditions"""
    market_analysis = {}

    # Volume analysis
    current_volume = df["Volume"].iloc[-1]
    avg_volume = df["Volume"].rolling(window=20).mean().iloc[-1]
    market_analysis["volume_ratio"] = current_volume / max(
        avg_volume, 1e-10
    )  # Prevent division by zero

    # Volatility analysis
    atr = calculate_atr(df)
    market_analysis["volatility"] = atr.iloc[-1] / max(
        df["Close"].iloc[-1], 1e-10
    )  # Prevent division by zero

    # RSI for momentum
    market_analysis["rsi"] = calculate_rsi(df["Close"]).iloc[-1]

    # If BTC data is provided, calculate correlation
    if btc_df is not None and len(btc_df) == len(df):
        correlation = df["Close"].pct_change().corr(btc_df["Close"].pct_change())
        market_analysis["btc_correlation"] = correlation

    # Market structure
    market_analysis["structure_score"] = calculate_market_structure_score(df)

    return market_analysis


def calculate_position_size(
    account_balance: float, risk_percentage: float, entry_price: float, stop_loss: float
) -> Dict[str, float]:
    """Calculate position size based on account risk"""
    risk_amount = account_balance * (risk_percentage / 100)
    price_difference = abs(entry_price - stop_loss)
    position_size = risk_amount / max(
        price_difference, 1e-10
    )  # Prevent division by zero

    return {
        "position_size": position_size,
        "position_value": position_size * entry_price,
        "risk_amount": risk_amount,
        "risk_percentage": risk_percentage,
    }


def generate_price_prediction_signal_proba(
    df: pd.DataFrame,
    indicators,
    weights: list = [],
    account_balance: float = 10000,
    risk_percentage: float = 1.0,
    btc_df: pd.DataFrame = None,
) -> Tuple[str, float, float, str, Optional[TradingSignal]]:
    """
    Generates a single-timeframe signal with bullish/bearish/neutral outcome.
    Now includes risk management and returns a TradingSignal object.

    Returns:
        (signal, probability_of_bullish, confidence, reason_str, trading_signal)
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
    W_RSI_EXTREME = 0.6  # NEW: Weight for RSI extremes

    if weights and len(weights) >= 16:  # Updated for new weights
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
        W_LIQUIDITY_POOL_ROUND = weights[16] if len(weights) > 16 else 1.5
        W_RSI_EXTREME = weights[17] if len(weights) > 17 else 0.6

    bullish_score = 0.0
    bearish_score = 0.0

    # NEW: Market context analysis
    market_context = analyze_market_correlation(df, btc_df)
    market_regime = detect_market_regime(df)

    # NEW: RSI analysis
    if "rsi" in market_context:
        if market_context["rsi"] < 30:
            bullish_score += W_RSI_EXTREME
            reasons.append(f"RSI oversold at {market_context['rsi']:.1f}")
        elif market_context["rsi"] > 70:
            bearish_score += W_RSI_EXTREME
            reasons.append(f"RSI overbought at {market_context['rsi']:.1f}")

    # Detect trend
    trend = detect_trend(df)
    if trend == "uptrend":
        bullish_score += W_TREND
        reasons.append("Price is in an uptrend")
    elif trend == "downtrend":
        bearish_score += W_TREND
        reasons.append("Price is in a downtrend")

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
                abs(pool.price - round_num) / max(abs(round_num), 1e-10) < 0.001
                for round_num in round_numbers
            )

            # Calculate pool impact based on strength and volume
            pool_impact = pool.strength * (
                pool.volume / max(df["Volume"].mean(), 1e-10)
            )

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
    if probability_of_bullish >= 0.66 and trend != "downtrend":
        signal = "Bullish"
    elif probability_of_bullish <= 0.33 and trend != "uptrend":
        signal = "Bearish"
    else:
        signal = "Neutral"

    # Calculate confidence
    confidence = abs(probability_of_bullish - 0.5) * 2.0

    # Add market context to reasons
    reasons.append(f"Market Regime: {market_regime.value}")
    reasons.append(f"Volume Ratio: {market_context['volume_ratio']:.2f}")
    reasons.append(f"Volatility: {market_context['volatility']*100:.2f}%")

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

    # Create TradingSignal object if signal is not neutral or confidence is high
    trading_signal = None
    if signal != "Neutral" or confidence > 0.3:
        entry_price = last_close

        # Calculate stop loss
        stop_loss = calculate_dynamic_stop_loss(df, signal, entry_price)

        # Calculate take profit levels
        tp1, tp2, tp3 = calculate_take_profit_levels(df, signal, entry_price, stop_loss)

        # Calculate risk/reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(tp2 - entry_price)  # Using TP2 as primary target
        risk_reward_ratio = reward / risk if risk > 0 else 0

        # Calculate position size
        position_sizing = calculate_position_size(
            account_balance, risk_percentage, entry_price, stop_loss
        )

        trading_signal = TradingSignal(
            symbol=df.attrs.get("symbol", "UNKNOWN"),
            signal_type=signal,
            probability=probability_of_bullish,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            risk_reward_ratio=risk_reward_ratio,
            position_size=position_sizing["position_size"],
            max_risk_amount=position_sizing["risk_amount"],
            reasons=reasons,
            market_conditions=market_context,
            timestamp=pd.Timestamp.now(),
        )

    return signal, probability_of_bullish, confidence, reason_str, trading_signal


# Multi-timeframe Analysis
async def multi_timeframe_analysis(
    symbol: str,
    preferences: Dict[str, bool],
    timeframes: List[str],
    candles_per_tf: int = 300,
    liq_lev_tolerance: float = 0.05,
    account_balance: float = 10000,
    risk_percentage: float = 1.0,
) -> Dict[str, Dict[str, any]]:
    """
    Fetches OHLCV data for multiple timeframes (e.g., 1h, 4h, 1d) and analyzes
    them using your 'analyze_data' function. Now includes risk management.

    Parameters:
        symbol (str): The symbol/currency pair to fetch (e.g. "BTCUSDT")
        preferences (dict): The user indicator preferences (e.g., order_blocks, fvgs, etc.)
        timeframes (List[str]): List of timeframes to analyze (e.g., ["1h", "4h", "1d"])
        candles_per_tf (int): How many candles to fetch for each timeframe
        liq_lev_tolerance (float): The tolerance for liquidity level detection
        account_balance (float): Account balance for position sizing
        risk_percentage (float): Risk percentage per trade

    Returns:
        A dictionary containing, for each timeframe:
            {
                "df": pandas DataFrame with the OHLCV data,
                "indicators": your Indicators object,
                "trading_signal": TradingSignal object if applicable
            }
    """
    mtf_results = {}

    # Fetch BTC data for correlation (if not BTC itself)
    btc_df = None
    if symbol != "BTCUSDT":
        btc_df = fetch_candles("BTCUSDT", candles_per_tf, "1h")

    for tf in timeframes:
        # 1) Fetch historical data for the desired timeframe
        df = fetch_candles(symbol, candles_per_tf, tf)

        if df is None or df.empty:
            # In case no data is returned or an error happened, skip
            logger.info(f"[multi_timeframe_analysis] No data for {symbol} on {tf}")
            continue

        # Add symbol attribute for reference
        df.attrs["symbol"] = symbol

        # 2) Analyze the data using your existing 'analyze_data' function,
        #    which returns an Indicators() object with order blocks, FVG, etc.
        indicators = analyze_data(df, preferences, liq_lev_tolerance)

        # 3) Generate enhanced signal with risk management
        signal, prob, conf, reasons, trading_signal = (
            generate_price_prediction_signal_proba(
                df, indicators, [], account_balance, risk_percentage, btc_df
            )
        )

        # 4) Store the results
        mtf_results[tf] = {
            "df": df,
            "indicators": indicators,
            "signal": signal,
            "probability": prob,
            "confidence": conf,
            "reasons": reasons,
            "trading_signal": trading_signal,
        }

    return mtf_results


# Multi-timeframe Aggregation of Signals
def generate_multi_tf_signal_proba(
    mtf_results: Dict[str, Dict[str, any]],
) -> Tuple[str, float, float, str, Optional[TradingSignal]]:
    """
    Aggregates signals from multiple timeframes. For each timeframe, we use
    'generate_price_prediction_signal_proba()' to produce an individual signal
    (Bullish/Bearish/Neutral) along with a probability of bullishness.

    Now also returns the best TradingSignal object.

    Parameters:
        mtf_results (dict): As returned by `multi_timeframe_analysis`.
            Example format:
            {
                "1h": {
                    "df": <DataFrame>,
                    "indicators": <Indicators>,
                    "trading_signal": <TradingSignal>
                },
                "4h": {
                    "df": <DataFrame>,
                    "indicators": <Indicators>,
                    "trading_signal": <TradingSignal>
                },
                ...
            }

    Returns:
        (final_signal, final_probability_of_bullish, confidence, reason_str, best_trading_signal)

        - final_signal (str): "Bullish", "Bearish", or "Neutral"
        - final_probability_of_bullish (float): aggregated probability in [0.001..0.999]
        - confidence (float): how far from 0.5 the final probability is, scaled to [0..1]
        - reason_str (str): textual explanation that includes breakdown from each timeframe
        - best_trading_signal (TradingSignal): The best signal object for execution
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
    trading_signals = []

    # 1) Loop through each timeframe's results
    for tf, data in mtf_results.items():
        signal = data.get("signal", "Neutral")
        prob_bullish = data.get("probability", 0.5)
        confidence = data.get("confidence", 0.0)
        reason_str = data.get("reasons", "")
        trading_signal = data.get("trading_signal")

        # 2) Retrieve a weight for that timeframe, default to 0.2 if not specified
        w = timeframe_weights.get(tf, 0.2)
        total_weight += w

        # 3) Accumulate weighted probability
        weighted_prob_sum += prob_bullish * w

        # 4) Collect trading signals
        if trading_signal:
            trading_signals.append((tf, trading_signal))

        # 5) Collect textual explanation for each timeframe
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
            None,
        )

    # 6) Compute the final aggregated bullish probability
    final_prob = weighted_prob_sum / total_weight

    # 7) Clamp the probability within [0.001..0.999]
    final_prob = max(0.001, min(final_prob, 0.999))

    # 8) Determine final signal by thresholds
    if final_prob >= 0.66:
        final_signal = "Bullish"
    elif final_prob <= 0.33:
        final_signal = "Bearish"
    else:
        final_signal = "Neutral"

    # 9) Calculate overall confidence
    #    (distance from 0.5 scaled to [0..1])
    confidence = abs(final_prob - 0.5) * 2.0

    # 10) Select the best trading signal (highest R:R with good confidence)
    best_trading_signal = None
    if trading_signals:
        # Sort by risk/reward ratio * confidence
        trading_signals.sort(
            key=lambda x: x[1].risk_reward_ratio * x[1].confidence, reverse=True
        )
        best_trading_signal = trading_signals[0][1]

    # 11) Combine all timeframe reasons into one final reason string
    final_reasons_str = (
        f"Final Aggregated Signal: {final_signal}\n"
        f"Aggregated Probability of Bullish: {final_prob:.3f}\n"
        f"Confidence: {confidence:.3f}\n\n"
        "Detailed breakdown by timeframe:\n" + "\n".join(reasons)
    )

    return final_signal, final_prob, confidence, final_reasons_str, best_trading_signal


# Signal quality validation
def validate_signal_quality(signal: TradingSignal) -> Tuple[bool, List[str]]:
    """
    Validate signal quality before sending to user
    """
    issues = []

    # Check risk/reward ratio
    if signal.risk_reward_ratio < 1.5:
        issues.append("Risk/Reward ratio below 1.5")

    # Check confidence
    if signal.confidence < 0.5:
        issues.append("Low confidence signal")

    # Check market conditions
    if signal.market_conditions.get("volume_ratio", 0) < 0.5:
        issues.append("Low volume - potential false signal")

    # Check position size
    if signal.position_size * signal.entry_price > signal.max_risk_amount * 10:
        issues.append("Position size too large for account")

    # Signal is valid if no critical issues
    is_valid = len(issues) == 0 or (len(issues) == 1 and signal.confidence > 0.7)

    return is_valid, issues


# Enhanced message formatting
def format_enhanced_signal_message(
    signal: TradingSignal, issues: List[str] = []
) -> str:
    """Format enhanced signal message with all details"""
    message = f"""
🎯 **SIGNAL ALERT - {signal.symbol}** 🎯

📊 **Signal Type:** {signal.signal_type}
💪 **Confidence:** {signal.confidence:.1%}
🎲 **Probability:** {signal.probability:.1%}

💰 **TRADE SETUP:**
━━━━━━━━━━━━━━━━━━
📍 **Entry Price:** ${signal.entry_price:.4f}
🛑 **Stop Loss:** ${signal.stop_loss:.4f} ({abs(signal.entry_price - signal.stop_loss)/signal.entry_price*100:.1f}%)
🎯 **Take Profit 1:** ${signal.take_profit_1:.4f} ({abs(signal.take_profit_1 - signal.entry_price)/signal.entry_price*100:.1f}%)
🎯 **Take Profit 2:** ${signal.take_profit_2:.4f} ({abs(signal.take_profit_2 - signal.entry_price)/signal.entry_price*100:.1f}%)
🎯 **Take Profit 3:** ${signal.take_profit_3:.4f} ({abs(signal.take_profit_3 - signal.entry_price)/signal.entry_price*100:.1f}%)

📈 **Risk Management:**
━━━━━━━━━━━━━━━━━━
⚖️ **Risk/Reward:** {signal.risk_reward_ratio:.2f}
💵 **Position Size:** {signal.position_size:.4f} units
💸 **Risk Amount:** ${signal.max_risk_amount:.2f}

🔄 **Market Conditions:**
━━━━━━━━━━━━━━━━━━
📊 **Volume:** {signal.market_conditions.get('volume_ratio', 0):.1f}x average
📉 **Volatility:** {signal.market_conditions.get('volatility', 0)*100:.1f}%
📈 **Structure:** {signal.market_conditions.get('structure_score', 0):.2f}
📊 **RSI:** {signal.market_conditions.get('rsi', 50):.1f}
"""

    if issues:
        message += f"\n⚠️ **Warnings:**\n"
        for issue in issues[:3]:
            message += f"• {issue}\n"

    # Add top 3 reasons
    message += f"\n📝 **Key Reasons:**\n"
    for reason in signal.reasons[:3]:
        message += f"• {reason}\n"

    message += (
        f"\n⏰ *Generated at {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC*"
    )

    return message


# Updated Auto-Signal Job with Multi-Timeframe Analysis
async def auto_signal_job(context):
    """
    This function is called periodically by the Telegram JobQueue (run_repeating).
    Instead of analyzing a single timeframe, it uses the multi-timeframe analysis
    to produce a more reliable signal. Then it decides whether to send a message.
    Now includes risk management and signal validation.
    """
    job_data = context.job.data
    user_id = job_data["user_id"]
    chat_id = job_data["chat_id"]
    currency_pair = job_data["currency_pair"]
    is_with_chart = job_data["is_with_chart"]
    account_balance = job_data.get("account_balance", 10000)  # NEW
    risk_percentage = job_data.get("risk_percentage", 1.0)  # NEW

    # 1) Fetch user preferences from the database
    preferences = get_user_preferences(user_id)

    # If the user has not selected any indicators, enable all by default
    if all(not value for value in preferences.values()):
        preferences = {k: True for k in preferences}

    # 2) Perform multi-timeframe analysis with risk management
    mtf_results = await multi_timeframe_analysis(
        symbol=currency_pair,
        preferences=preferences,
        timeframes=["15m", "1h", "4h"],
        candles_per_tf=300,
        liq_lev_tolerance=0.05,
        account_balance=account_balance,
        risk_percentage=risk_percentage,
    )

    # If we have no data/timeframes, abort
    if not mtf_results:
        await context.bot.send_message(
            chat_id=chat_id, text=f"[Auto-Signal] No data found for {currency_pair}."
        )
        return

    # 3) Generate an aggregated signal from all timeframes
    final_signal, final_prob, confidence, reason_str, best_trading_signal = (
        generate_multi_tf_signal_proba(mtf_results)
    )

    # 4) NEW: Validate signal quality if we have a trading signal
    if best_trading_signal:
        is_valid, issues = validate_signal_quality(best_trading_signal)

        # Skip if signal is invalid and confidence is low
        if not is_valid and best_trading_signal.confidence < 0.8:
            return

        # Format enhanced message
        message = format_enhanced_signal_message(best_trading_signal, issues)
    else:
        # Use basic message format
        message = f"[Auto-Signal for {currency_pair}]\n\n{reason_str}"

    # 5) Decide if we want to send the signal to the user
    #    For instance, we can require a minimum confidence or a non-neutral signal
    if confidence > 0.0 or final_signal != "Neutral":
        try:
            await context.bot.send_message(
                chat_id=chat_id, text=message, parse_mode="Markdown"
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


# Creating and Deleting Signal Jobs
async def createSignalJob(
    symbol: str,
    period_minutes: int,
    is_with_chart: bool,
    update,
    context,
    account_balance: float = 10000,
    risk_percentage: float = 1.0,
):
    """
    Creates a repeating job for auto-signal analysis (multi-timeframe).
    Now includes risk management parameters.
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
        "account_balance": account_balance,  # NEW
        "risk_percentage": risk_percentage,  # NEW
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
            "account_balance": account_balance,  # NEW
            "risk_percentage": risk_percentage,  # NEW
        },
    )

    # Save the job reference
    auto_signal_jobs[job_key] = job_ref

    await update.message.reply_text(
        f"✅ Auto-signals started for {symbol}, every {period_minutes} minute(s)."
        f"\n💰 Account Balance: ${account_balance}"
        f"\n⚠️ Risk per trade: {risk_percentage}%"
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


# Initialization of all Jobs at Startup
async def initialize_jobs(application):
    """
    Called once at bot start-up to restore all jobs from the database.
    """
    signal_requests = get_signal_requests()

    for req in signal_requests:
        user_id = req["user_id"]
        currency_pair = req["currency_pair"]
        frequency_minutes = req["frequency_minutes"]
        is_with_chart = req.get("is_with_chart", False)
        account_balance = req.get("account_balance", 10000)
        risk_percentage = req.get("risk_percentage", 1.0)
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
            "is_with_chart": is_with_chart,
            "account_balance": account_balance,
            "risk_percentage": risk_percentage,
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


# Performance Tracking (Optional - for future implementation)
class PerformanceTracker:
    """Track and analyze signal performance for optimization"""

    def __init__(self):
        self.signals_history = []
        self.performance_metrics = {}

    def record_signal(self, signal: TradingSignal):
        """Record a signal for tracking"""
        self.signals_history.append(
            {"timestamp": signal.timestamp, "signal": signal, "status": "open"}
        )

    def calculate_performance_metrics(self) -> Dict:
        """Calculate win rate, average R:R, etc."""
        # Implementation for tracking performance
        pass


def analyze_liquidity_levels(
    df: pd.DataFrame, tolerance: float = 0.001
) -> List[Dict[str, Any]]:
    """Analyze liquidity levels in the market"""
    liquidity_levels = []

    # Round numbers to analyze
    round_numbers = [
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
        100.0,
        200.0,
        500.0,
        1000.0,
    ]

    # Analyze each price point
    for i in range(len(df)):
        price = df["Close"].iloc[i]
        volume = df["Volume"].iloc[i]

        # Check if price is near a round number
        for round_num in round_numbers:
            # Avoid division by zero by checking if round_num is zero
            if round_num == 0:
                continue

            # Calculate relative distance to round number
            relative_distance = abs(price - round_num) / round_num

            # If price is close to round number and volume is significant
            if relative_distance < tolerance and volume > df["Volume"].mean():
                liquidity_levels.append(
                    {
                        "price": price,
                        "round_number": round_num,
                        "volume": volume,
                        "timestamp": df.index[i],
                        "relative_distance": relative_distance,
                    }
                )

    return liquidity_levels
