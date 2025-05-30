from datetime import timedelta
from typing import List, Dict, Tuple

import pandas as pd  # type: ignore

from helpers import fetch_candles, analyze_data, fetch_data_and_get_indicators
from database import (
    get_user_preferences,
    upsert_user_signal_request,
    delete_user_signal_request,
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
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=window).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    price = df["Close"].iloc[-1]
    ema20, ema50 = df["EMA20"].iloc[-1], df["EMA50"].iloc[-1]
    momentum = (
        price - df["Close"].iloc[-window]
    ) / df["Close"].iloc[-window]

    if price > ema20 and ema20 > ema50 and momentum > 0.01:
        return "bullish"
    if price < ema20 and ema20 < ema50 and momentum < -0.01:
        return "bearish"
    return "neutral"


def generate_price_prediction_signal_proba(
    df: pd.DataFrame, indicators, weights: List[float] = []
) -> Tuple[str, float, float, str]:
    """
    Single-timeframe bullish/bearish/neutral signal with probability and reasons.
    """
    last_close = df["Close"].iloc[-1]
    reasons: List[str] = []

    # Default weights
    W_OB, W_BB, W_SR, W_FVG, W_TREND = 1.0, 1.0, 0.7, 0.5, 0.8
    if len(weights) == 5:
        W_OB, W_BB, W_SR, W_FVG, W_TREND = weights

    bullish_score = bearish_score = 0.0

    # 1) Trend
    trend = detect_trend(df)
    if trend == "bullish":
        bullish_score += W_TREND
        reasons.append("Price is in a bullish trend")
    elif trend == "bearish":
        bearish_score += W_TREND
        reasons.append("Price is in a bearish trend")

    # 2) Support/Resistance via custom detection
    supports, resistances = detect_support_resistance(df)
    nearest_support = max((s for s in supports if s < last_close), default=None)
    nearest_resistance = min((r for r in resistances if r > last_close), default=None)
    if nearest_support:
        dist = (last_close - nearest_support) / last_close
        if dist < 0.02:
            bullish_score += W_SR
            reasons.append(f"Price near support @ {nearest_support:.2f}")
    if nearest_resistance:
        dist = (nearest_resistance - last_close) / last_close
        if dist < 0.02:
            bearish_score += W_SR
            reasons.append(f"Price near resistance @ {nearest_resistance:.2f}")

    # 3) Order & Breaker Blocks from indicators
    for block in getattr(indicators, "order_blocks", []).list or []:
        if block.block_type == "bullish" and block.index >= len(df) - 3:
            bullish_score += W_OB
            reasons.append("Recent bullish order block found")
            break
    for block in getattr(indicators, "order_blocks", []).list or []:
        if block.block_type == "bearish" and block.index >= len(df) - 3:
            bearish_score += W_OB
            reasons.append("Recent bearish order block found")
            break
    for block in getattr(indicators, "breaker_blocks", []).list or []:
        if block.block_type == "bullish" and block.index >= len(df) - 4:
            bullish_score += W_BB
            reasons.append("Recent bullish breaker block found")
            break
    for block in getattr(indicators, "breaker_blocks", []).list or []:
        if block.block_type == "bearish" and block.index >= len(df) - 4:
            bearish_score += W_BB
            reasons.append("Recent bearish breaker block found")
            break

    # 4) FVGs
    for fvg in getattr(indicators, "fvgs", []).list or []:
        if last_close > fvg.start_price and last_close > fvg.end_price:
            bearish_score += W_FVG
            reasons.append("Unfilled FVG below price")
        elif last_close < fvg.start_price and last_close < fvg.end_price:
            bullish_score += W_FVG
            reasons.append("Unfilled FVG above price")

    # 5) Compute probabilities
    total = bullish_score + bearish_score
    prob = 0.5 if total < 1e-9 else bullish_score / total
    prob = max(0.001, min(prob, 0.999))
    if prob >= 0.66:
        signal = "Bullish"
    elif prob <= 0.33:
        signal = "Bearish"
    else:
        signal = "Neutral"
    confidence = abs(prob - 0.5) * 2.0

    # 6) Reason string
    reason_str = (
        f"Signal: {signal}\n"
        f"Trend: {trend}\n"
        f"Bullish Score: {bullish_score:.2f} | Bearish Score: {bearish_score:.2f}\n"
        f"Probability of Bullish: {prob:.3f}\n"
        f"Confidence: {confidence:.3f}\n\n"
    )
    if reasons:
        reason_str += "Reasons:\n- " + "\n- ".join(reasons)

    return signal, prob, confidence, reason_str


async def multi_timeframe_analysis(
    symbol: str,
    preferences: Dict[str, bool],
    timeframes: List[str],
    candles_per_tf: int = 300,
    liq_lev_tolerance: float = 0.05,
) -> Dict[str, Dict[str, any]]:
    results = {}
    for tf in timeframes:
        df = fetch_candles(symbol, candles_per_tf, tf)
        if df is None or df.empty:
            logger.info(f"No data for {symbol} on {tf}")
            continue
        indicators = analyze_data(df, preferences, liq_lev_tolerance)
        results[tf] = {"df": df, "indicators": indicators}
    return results


def generate_multi_tf_signal_proba(
    mtf: Dict[str, Dict[str, any]]
) -> Tuple[str, float, float, str]:
    weights = {"15m": 0.2, "1h": 0.3, "4h": 0.5}
    total_w = sum(weights.get(tf, 0.2) for tf in mtf)
    weighted = 0.0
    reasons = []
    for tf, data in mtf.items():
        sig, p, conf, rstr = generate_price_prediction_signal_proba(
            data["df"], data["indicators"]
        )
        w = weights.get(tf, 0.2)
        weighted += p * w
        reasons.append(f"[{tf}] {sig} ({p:.3f})\n" + rstr)
    if total_w == 0:
        return "Neutral", 0.5, 0.0, "No data available"
    final_p = max(0.001, min(weighted / total_w, 0.999))
    if final_p >= 0.66:
        final_signal = "Bullish"
    elif final_p <= 0.33:
        final_signal = "Bearish"
    else:
        final_signal = "Neutral"
    final_conf = abs(final_p - 0.5) * 2.0
    final_reasons = (
        f"Final: {final_signal} ({final_p:.3f}, conf={final_conf:.3f})\n\n"
        + "\n---\n".join(reasons)
    )
    return final_signal, final_p, final_conf, final_reasons

async def auto_signal_job(context):
    data = context.job.data
    uid, cid, pair = data["user_id"], data["chat_id"], data["currency_pair"]
    chart = data.get("is_with_chart", False)
    prefs = get_user_preferences(uid)
    if all(not v for v in prefs.values()):
        prefs = {k: True for k in prefs}
    mtf = await multi_timeframe_analysis(pair, prefs, ["15m", "1h", "4h"] )
    if not mtf:
        await context.bot.send_message(cid, text=f"[Auto-Signal] No data for {pair}.")
        return
    sig, p, conf, rstr = generate_multi_tf_signal_proba(mtf)
    if conf > 0 or sig != "Neutral":
        await context.bot.send_message(cid, text=f"[Signal {pair}]\n{rstr}")
        if chart:
            inds, df = await fetch_data_and_get_indicators(
                [pair, 200, "1h"], create_true_preferences(), ()
            )
            path = plot_price_chart(df, inds, True, True)
            with open(path, "rb") as f:
                await context.bot.send_photo(cid, photo=f)

async def createSignalJob(symbol: str, period: int, chart: bool, update, ctx):
    uid, cid = update.effective_user.id, update.effective_chat.id
    if user_signal_request_exists(uid, symbol):
        await update.message.reply_text(
            f"❌ Auto-signal exists for {symbol}. Delete first."
        )
        return
    upsert_user_signal_request(uid, {"currency_pair": symbol, "frequency_minutes": period, "is_with_chart": chart})
    key = (uid, symbol)
    if key in auto_signal_jobs:
        auto_signal_jobs[key].schedule_removal()
        del auto_signal_jobs[key]
    job = ctx.application.job_queue.run_repeating(
        auto_signal_job,
        interval=timedelta(minutes=period), first=0,
        name=f"sig_{uid}_{symbol}",
        data={"user_id": uid, "chat_id": cid, "currency_pair": symbol, "is_with_chart": chart}
    )
    auto_signal_jobs[key] = job
    await update.message.reply_text(f"✅ Started auto-signals for {symbol}." )

async def deleteSignalJob(symbol: str, update):
    uid = update.effective_user.id
    delete_user_signal_request(uid, symbol)
    key = (uid, symbol)
    if key in auto_signal_jobs:
        auto_signal_jobs[key].schedule_removal()
        del auto_signal_jobs[key]
        await update.message.reply_text(f"✅ Stopped auto-signals for {symbol}.")
    else:
        await update.message.reply_text(f"No auto-signals for {symbol}.")

async def initialize_jobs(app):
    for req in get_signal_requests():
        uid, pair, period = req["user_id"], req["currency_pair"], req["frequency_minutes"]
        cid = get_chat_id_for_user(uid)
        if not cid:
            logger.info(f"Skip {uid}:{pair}, no chat_id.")
            continue
        key = (uid, pair)
        if key in auto_signal_jobs:
            continue
        job = app.job_queue.run_repeating(
            auto_signal_job,
            interval=timedelta(minutes=period), first=0,
            name=f"sig_{uid}_{pair}",
            data={"user_id": uid, "chat_id": cid, "currency_pair": pair}
        )
        auto_signal_jobs[key] = job
    logger.info("Initialized all signal jobs.")
