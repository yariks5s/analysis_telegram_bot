from utils import auto_signal_jobs
from datetime import timedelta
from database import get_user_preferences, upsert_user_signal_request, delete_user_signal_request, get_chat_id_for_user, get_signal_requests

import sqlite3

def generate_price_prediction_signal_proba(df, indicators):
    """
    Generate a more complex weighted bullish/bearish signal
    based on:
        - Order Blocks (bullish/bearish)
        - Breaker Blocks (bullish/bearish)
        - Liquidity Levels (support/resistance)
        - FVGs (bullish/bearish + covered/uncovered)

    Returns:
        (signal, probability, confidence, reason_str)
            - signal (str): "Bullish", "Bearish", or "Neutral"
            - probability (float): 0.0 to 1.0 (probability price will move up)
                                  (clamped to [0.001..0.999], never 100%)
            - confidence (float): absolute distance from 0.5 (range [0..1]),
                                  so 1.0 is max confidence (never exactly reached)
            - reason_str (str): textual explanation
    """

    # --------------- 1) Extract last price, set up some memory ---------------
    last_close = df['Close'].iloc[-1]
    reasons = []

    # --------------- 2) Define weights for each condition ---------------
    #Tune these weights based on backtesting or heuristics.
    W_BULLISH_OB          = 1.0  # weight for a recent bullish order block
    W_BEARISH_OB          = 1.0  # weight for a recent bearish order block
    W_BULLISH_BREAKER     = 1.0
    W_BEARISH_BREAKER     = 1.0
    W_ABOVE_SUPPORT       = 0.7  # if price is above support
    W_BELOW_RESISTANCE    = 0.7  # if price is below resistance
    W_BULLISH_FVG         = 0.5  # unfilled bullish FVG below price
    W_BEARISH_FVG         = 0.5  # unfilled bearish FVG above price

    bullish_score = 0.0
    bearish_score = 0.0

    # --------------- 3) Check conditions similarly to the naive approach ---------------

    # --- Bullish Order Block ---
    bullish_ob = False
    if indicators.order_blocks and len(indicators.order_blocks.list) > 0:
        # e.g. check the last 3 blocks
        last_idx = len(df) - 1
        for block in indicators.order_blocks.list[-3:]:
            if block.block_type == 'bullish' and block.index >= (last_idx - 3):
                bullish_ob = True
                reasons.append("Recent bullish order block found")
                break
    if bullish_ob:
        bullish_score += W_BULLISH_OB

    # --- Bearish Order Block ---
    bearish_ob = False
    if indicators.order_blocks and len(indicators.order_blocks.list) > 0:
        last_idx = len(df) - 1
        for block in indicators.order_blocks.list[-3:]:
            if block.block_type == 'bearish' and block.index >= (last_idx - 3):
                bearish_ob = True
                reasons.append("Recent bearish order block found")
                break
    if bearish_ob:
        bearish_score += W_BEARISH_OB

    # --- Bullish Breaker ---
    bullish_breaker = False
    if indicators.breaker_blocks and len(indicators.breaker_blocks.list) > 0:
        for block in indicators.breaker_blocks.list[-3:]:
            if block.block_type == 'bullish' and block.index >= (len(df) - 4):
                bullish_breaker = True
                reasons.append("Recent bullish breaker block found")
                break
    if bullish_breaker:
        bullish_score += W_BULLISH_BREAKER

    # --- Bearish Breaker ---
    bearish_breaker = False
    if indicators.breaker_blocks and len(indicators.breaker_blocks.list) > 0:
        for block in indicators.breaker_blocks.list[-3:]:
            if block.block_type == 'bearish' and block.index >= (len(df) - 4):
                bearish_breaker = True
                reasons.append("Recent bearish breaker block found")
                break
    if bearish_breaker:
        bearish_score += W_BEARISH_BREAKER

    # --- Price Above Support Level ---
    above_support = False
    if indicators.liquidity_levels and len(indicators.liquidity_levels.list) > 0:
        # find the last discovered support
        supports = [lvl.price for lvl in indicators.liquidity_levels.list if lvl.level_type == 'support']
        if supports:
            last_support = supports[-1]
            if last_close > last_support:
                above_support = True
                reasons.append(f"Price {last_close:.2f} is above support {last_support:.2f}")
    if above_support:
        bullish_score += W_ABOVE_SUPPORT

    # --- Price Below Resistance Level ---
    below_resistance = False
    if indicators.liquidity_levels and len(indicators.liquidity_levels.list) > 0:
        # find the last discovered resistance
        resistances = [lvl.price for lvl in indicators.liquidity_levels.list if lvl.level_type == 'resistance']
        if resistances:
            last_resistance = resistances[-1]
            if last_close < last_resistance:
                below_resistance = True
                reasons.append(f"Price {last_close:.2f} is below resistance {last_resistance:.2f}")
    if below_resistance:
        bearish_score += W_BELOW_RESISTANCE

    # --- Unfilled Bullish FVG below price ---
    unfilled_bullish_fvg = False
    if indicators.fvgs and indicators.fvgs.list:
        for fvg in indicators.fvgs.list:
            if (
                fvg.fvg_type == 'bullish' and
                not fvg.covered and
                fvg.start_price < last_close and 
                fvg.end_price < last_close
            ):
                unfilled_bullish_fvg = True
                reasons.append("Unfilled bullish FVG below current price")
                break
    if unfilled_bullish_fvg:
        bullish_score += W_BULLISH_FVG

    # --- Unfilled Bearish FVG above price ---
    unfilled_bearish_fvg = False
    if indicators.fvgs and indicators.fvgs.list:
        for fvg in indicators.fvgs.list:
            if (
                fvg.fvg_type == 'bearish' and
                not fvg.covered and
                fvg.start_price > last_close and
                fvg.end_price > last_close
            ):
                unfilled_bearish_fvg = True
                reasons.append("Unfilled bearish FVG above current price")
                break
    if unfilled_bearish_fvg:
        bearish_score += W_BEARISH_FVG

    # --------------- 4) Convert scores to a probability ---------------
    eps = 1e-9  # tiny to avoid division by zero
    total_score = bullish_score + bearish_score

    if total_score < eps:
        # If no signals are triggered at all, treat it as neutral => 50%
        probability_of_bullish = 0.5
    else:
        probability_of_bullish = bullish_score / (bullish_score + bearish_score)

    # --------------- 5) Clamp probability so it's never 0% or 100% ---------------
    probability_of_bullish = max(0.001, min(probability_of_bullish, 0.999))

    # --------------- 6) Decide final signal ---------------
    # For example, thresholds:
    #   If P(bullish) >= 0.66 => "Bullish"
    #   If P(bullish) <= 0.33 => "Bearish"
    #   Otherwise => "Neutral"
    signal = "Neutral"
    if probability_of_bullish >= 0.66:
        signal = "Bullish"
    elif probability_of_bullish <= 0.33:
        signal = "Bearish"

    # --------------- 7) Calculate confidence ---------------
    # Confidence is how far from 0.5 we are, scaled to max 1.0
    # Because of the clamp, it will never be exactly 1.0
    confidence = abs(probability_of_bullish - 0.5) * 2.0

    # --------------- 8) Build reason string ---------------
    reason_str = (
        f"Signal: {signal}\n"
        f"Bullish Score: {bullish_score:.2f} | Bearish Score: {bearish_score:.2f}\n"
        f"Probability of Bullish: {probability_of_bullish:.3f}\n"
        f"Confidence: {confidence:.3f}\n\n"
    )
    if reasons:
        reason_str += "Reasons:\n- " + "\n- ".join(reasons)

    return signal, probability_of_bullish, confidence, reason_str


async def auto_signal_job(context):
    """
    This function is called periodically by the JobQueue (run_repeating).
    It fetches signals and sends them to the user if needed.
    """
    job_data = context.job.data
    user_id = job_data["user_id"]
    chat_id = job_data["chat_id"]
    currency_pair = job_data["currency_pair"]

    # 1) Fetch user preferences from DB
    preferences = get_user_preferences(user_id)

    if all(not value for value in preferences.values()):  # Check if all items are False
        preferences = {key: True for key in preferences}  # Set all to True

    # 2) Perform the analysis (placeholder)
    # Replace with your actual analysis function
    # indicators, df = await check_and_analyze(currency_pair)
    indicators = None  # Replace with actual indicators
    df = None  # Replace with actual data frame

    # 3) Generate a signal
    # signal, prob_bullish, confidence, reason_str = generate_price_prediction_signal_proba(df, indicators)

    # 4) Decide whether to send the signal based on confidence or signal
    # if confidence > 0.6 or signal != "Neutral":
    #     # 5) Build the message
    #     now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    #     message_text = (
    #         f"[Auto-Signal Check @ {now_str}]\n"
    #         f"Symbol: {currency_pair}\n"
    #         f"Signal: {signal}\n"
    #         f"Probability of Bullish: {prob_bullish:.3f}\n"
    #         f"Confidence: {confidence:.3f}\n\n"
    #         f"Reasons:\n{reason_str}"
    #     )

        # 6) Send the message
        # try:
    await context.bot.send_message(chat_id=chat_id, text=f"Signal for {currency_pair} for 1 minute")
        # except Exception as e:
        #     print(f"Error sending message to user {user_id}: {e}")


async def createSignalJob(symbol: str, period_minutes: int, update, context):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    currency_pair = symbol

    # Update the database with the new signal request
    signals_request = {
        "currency_pair": currency_pair,
        "frequency_minutes": period_minutes,
    }
    upsert_user_signal_request(user_id, signals_request)

    job_key = (user_id, currency_pair)

    # If a job is already running for this user and currency pair, cancel it
    if job_key in auto_signal_jobs:
        old_job = auto_signal_jobs[job_key]
        old_job.schedule_removal()
        del auto_signal_jobs[job_key]

    # Create a job to run periodically
    job_ref = context.application.job_queue.run_repeating(
        callback=auto_signal_job,
        interval=timedelta(minutes=period_minutes),
        first=0,  # Start immediately
        name=f"signal_job_{user_id}_{currency_pair}",
        data={
            "user_id": user_id,
            "chat_id": chat_id,
            "currency_pair": currency_pair,
        },
    )

    # Save reference
    auto_signal_jobs[job_key] = job_ref

    await update.message.reply_text(
        f"✅ Auto-signals started for {currency_pair}, every {period_minutes} minute{'s' if period_minutes > 1 else ''}."
    )


async def deleteSignalJob(currency_pair, update):
    user_id = update.effective_user.id

    # Remove from the database
    delete_user_signal_request(user_id, currency_pair)

    # Remove the job from the job queue
    job_key = (user_id, currency_pair)
    if job_key in auto_signal_jobs:
        job_ref = auto_signal_jobs[job_key]
        job_ref.schedule_removal()
        del auto_signal_jobs[job_key]
        await update.message.reply_text(f"✅ Auto-signals for {currency_pair} stopped.")
    else:
        await update.message.reply_text(f"No auto-signals are running for {currency_pair}.")


async def initialize_jobs(application):
    """
    Initialize all user signal jobs from the database when the application starts.
    """
    signal_requests = get_signal_requests()

    for request in signal_requests:
        user_id = request["user_id"]
        currency_pair = request["currency_pair"]
        frequency_minutes = request["frequency_minutes"]
        chat_id = get_chat_id_for_user(user_id)

        if not chat_id:
            print(f"No chat_id found for user_id {user_id}. Skipping job creation.")
            continue

        job_key = (user_id, currency_pair)

        if job_key in auto_signal_jobs:
            print(f"Job for user_id {user_id}, currency_pair {currency_pair} already exists.")
            continue

        # Create job data
        job_data = {
            "user_id": user_id,
            "chat_id": chat_id,
            "currency_pair": currency_pair,
        }

        # Create a job to run periodically
        job_ref = application.job_queue.run_repeating(
            callback=auto_signal_job,
            interval=timedelta(minutes=frequency_minutes),
            first=0,  # Start immediately
            name=f"signal_job_{user_id}_{currency_pair}",
            data=job_data,
        )

        # Save reference
        auto_signal_jobs[job_key] = job_ref

    print("All user signal jobs have been initialized.")

