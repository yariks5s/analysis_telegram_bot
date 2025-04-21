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
from utils import auto_signal_jobs, create_true_preferences

from plot_build_helpers import plot_price_chart

from datetime import timedelta


def generate_price_prediction_signal_proba(
    df: pd.DataFrame, indicators, weights: list = []
) -> Tuple[str, float, float, str]:
    """
    Generates a single-timeframe signal with bullish/bearish/neutral outcome.
    Treats all FVGs based on their position relative to the current price.

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
    W_FVG_ABOVE = 0.5  # Weight for FVGs above price
    W_FVG_BELOW = 0.5  # Weight for FVGs below price

    if weights and len(weights) == 8:
        W_BULLISH_OB = weights[0]
        W_BEARISH_OB = weights[1]
        W_BULLISH_BREAKER = weights[2]
        W_BEARISH_BREAKER = weights[3]
        W_ABOVE_SUPPORT = weights[4]
        W_BELOW_RESISTANCE = weights[5]
        W_FVG_ABOVE = weights[6]
        W_FVG_BELOW = weights[7]

    bullish_score = 0.0
    bearish_score = 0.0

    # -------------------------------------------------------------
    # 1) Order Blocks and Breaker Blocks
    # -------------------------------------------------------------
    if indicators.order_blocks and indicators.order_blocks.list:
        for block in indicators.order_blocks.list[-3:]:
            if block.block_type == "bullish" and block.index >= (len(df) - 3):
                bullish_score += W_BULLISH_OB
                reasons.append("Recent bullish order block found")
                break

    if indicators.order_blocks and indicators.order_blocks.list:
        for block in indicators.order_blocks.list[-3:]:
            if block.block_type == "bearish" and block.index >= (len(df) - 3):
                bearish_score += W_BEARISH_OB
                reasons.append("Recent bearish order block found")
                break

    if indicators.breaker_blocks and indicators.breaker_blocks.list:
        for block in indicators.breaker_blocks.list[-3:]:
            if block.block_type == "bullish" and block.index >= (len(df) - 4):
                bullish_score += W_BULLISH_BREAKER
                reasons.append("Recent bullish breaker block found")
                break

    if indicators.breaker_blocks and indicators.breaker_blocks.list:
        for block in indicators.breaker_blocks.list[-3:]:
            if block.block_type == "bearish" and block.index >= (len(df) - 4):
                bearish_score += W_BEARISH_BREAKER
                reasons.append("Recent bearish breaker block found")
                break

    # -------------------------------------------------------------
    # 2) Liquidity Levels (Support / Resistance)
    # -------------------------------------------------------------
    above_support = False
    below_resistance = False

    # TODO!
    # if indicators.liquidity_levels and indicators.liquidity_levels.list:
    # Find last discovered support
    # supports = [
    #     lvl.price for lvl in indicators.liquidity_levels.list if lvl.is_support()
    # ]
    # if supports:
    #     last_support = supports[-1]
    #     if last_close > last_support:
    #         above_support = True
    #         reasons.append(
    #             f"Price {last_close:.2f} is above support {last_support:.2f}"
    #         )

    # Find last discovered resistance
    # resistances = [
    #     lvl.price for lvl in indicators.liquidity_levels.list if lvl.is_resistance()
    # ]
    # if resistances:
    #     last_resistance = resistances[-1]
    #     if last_close < last_resistance:
    #         below_resistance = True
    #         reasons.append(
    #             f"Price {last_close:.2f} is below resistance {last_resistance:.2f}"
    #        )

    if above_support:
        bullish_score += W_ABOVE_SUPPORT
    if below_resistance:
        bearish_score += W_BELOW_RESISTANCE

    # -------------------------------------------------------------
    # 3) FVG Logic (Based on Position Relative to Current Price)
    # -------------------------------------------------------------
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
            else:
                # Price is within the FVG range; treat as neutral or consider specific logic
                # For simplicity, you might skip or handle differently
                reasons.append("Price is within FVG range; no score adjustment")

    # -------------------------------------------------------------
    # 4) Convert scores to final probability, confidence, signal
    # -------------------------------------------------------------
    eps = 1e-9
    total_score = bullish_score + bearish_score
    if total_score < eps:
        probability_of_bullish = 0.5
    else:
        probability_of_bullish = bullish_score / total_score

    # Clamp probability
    probability_of_bullish = max(0.001, min(probability_of_bullish, 0.999))

    # Decide final signal
    if probability_of_bullish >= 0.66:
        signal = "Bullish"
    elif probability_of_bullish <= 0.33:
        signal = "Bearish"
    else:
        signal = "Neutral"

    # Confidence is how far from 0.5 we are
    confidence = abs(probability_of_bullish - 0.5) * 2.0

    # Compile reason string
    reason_str = (
        f"Signal: {signal}\n"
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
            print(f"[multi_timeframe_analysis] No data for {symbol} on {tf}")
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
            print(f"Error sending auto-signal message to user {user_id}: {str(e)}")
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
            print(f"No chat_id found for user {user_id}. Skipping job creation.")
            continue

        job_key = (user_id, currency_pair)
        if job_key in auto_signal_jobs:
            print(f"Job for user_id {user_id}, pair {currency_pair} already exists.")
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

    print("All user signal jobs have been initialized.")
