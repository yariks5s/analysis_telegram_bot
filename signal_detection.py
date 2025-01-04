from utils import auto_signal_jobs
from datetime import datetime, timedelta
from database import get_user_preferences

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
    symbol = job_data["symbol"]

    # 1) Fetch user preferences (if you need them) from DB
    preferences = get_user_preferences(user_id)

    if (all(not value for value in preferences.values())): # Check if all items are False
        preferences = {key: True for key in preferences} # If yes, sett all to True

    # 2) Perform the analysis (example):
    # (indicators, df) = await check_and_analyze(...)

    # 3) Generate a signal
    # signal, prob_bullish, confidence, reason_str = generate_price_prediction_signal_proba(df, indicators)

    # 4) If there's a specific condition to alert:
    # e.g., only alert if confidence > 0.6 or if signal != "Neutral"
    # For demonstration, let's say we send a message every time for now:
    # --------------------------------
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    message_text = (
        f"[Auto-Signal Check @ {now_str}]\n"
        f"Symbol: {symbol}\n"
        f"**(Add your signal details here)**"
    )

    # 5) Send the message
    await context.bot.send_message(chat_id=chat_id, text=message_text)


async def createSignalJob(symbol, period_minutes, update, context):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # If a job is already running for this user, cancel it
    if user_id in auto_signal_jobs:
        old_job = auto_signal_jobs[user_id]
        old_job.schedule_removal()
        del auto_signal_jobs[user_id]

    # Create a job to run periodically
    job_queue = context.application.job_queue
    job_ref = job_queue.run_repeating(
        callback=auto_signal_job,              # the function to call
        interval=timedelta(minutes=period_minutes),
        first=0,                               # start immediately
        name=f"signal_job_{user_id}",
        data={
            "user_id": user_id,
            "chat_id": chat_id,
            "symbol": symbol,
        },
    )

    # Save reference so we can stop it later
    auto_signal_jobs[user_id] = job_ref
    await update.message.reply_text(
        f"✅ Auto-signals started for {symbol}, every {period_minutes} minute{'s' if period_minutes > 1 else ''}."
    )


async def deleteSignalJob(update):
    user_id = update.effective_user.id

    if user_id in auto_signal_jobs:
        job_ref = auto_signal_jobs[user_id]
        job_ref.schedule_removal()
        del auto_signal_jobs[user_id]
        await update.message.reply_text("✅ Auto-signals stopped.")
    else:
        await update.message.reply_text("No auto-signals are running for you.")

