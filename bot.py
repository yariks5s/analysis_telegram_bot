from helpers import check_and_analyze, get_1000
from plot_build_helpers import plot_price_chart
from message_handlers import select_indicators, handle_indicator_selection
from signal_detection import generate_price_prediction_signal_proba
from database import get_user_preferences

from utils import auto_signal_jobs

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackContext,
    CallbackQueryHandler,
    JobQueue,
    Job,
)

from dotenv import load_dotenv

import os
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

load_dotenv()

async def send_crypto_chart(update: Update, context: CallbackContext):
    """
    Telegram handler to fetch OHLC data, analyze indicators, and send the chart back to the user.
    """
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    preferences = get_user_preferences(user_id)

    # Analyze data
    (indicators, df) = await check_and_analyze(update, user_id, preferences, context.args)


    # Plot chart
    chart_path = plot_price_chart(df, indicators)
    if chart_path is None:
        await update.message.reply_text("Error generating the chart. Please try again.")
        return
    
    # 1) Generate the probability-based signal:
    _, _, _, reason_str = generate_price_prediction_signal_proba(df, indicators)

    # Send the chart to the user
    with open(chart_path, 'rb') as f:
        await context.bot.send_photo(chat_id=chat_id, photo=f)

    await update.message.reply_text(f"  {reason_str}")  # Indentation needed for correct representation of the message
    
    df = df.reset_index(drop=True)

# Needs to be deprecated later
async def send_crypto_chart_1000(update: Update, context: CallbackContext):
    """
    Telegram handler to fetch OHLC data, analyze indicators, and send the chart back to the user.
    """
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    preferences = get_user_preferences(user_id)

    # Analyze data
    (indicators, df) = await get_1000(update, user_id, preferences, context.args)


    # Plot chart
    chart_path = plot_price_chart(df, indicators)
    if chart_path is None:
        await update.message.reply_text("Error generating the chart. Please try again.")
        return
    
    # 1) Generate the probability-based signal:
    _, _, _, reason_str = generate_price_prediction_signal_proba(df, indicators)

    # Send the chart to the user
    with open(chart_path, 'rb') as f:
        await context.bot.send_photo(chat_id=chat_id, photo=f)

    await update.message.reply_text(f"  {reason_str}")  # Indentation needed for correct representation of the message
    
    df = df.reset_index(drop=True)


async def send_text_data(update: Update, context: CallbackContext):
    """
    Telegram handler to fetch OHLC data for a user-specified crypto pair, time period, interval, and liquidity level detection tolerance
    plot the candlestick chart, and send it back to the user.
    Usage: /text_result <symbol> <hours> <interval> <tolerance>, e.g. /text_result BTCUSDT 42 15m 0.03
    """
    user_id = update.effective_user.id

    preferences = get_user_preferences(user_id)
    # Analyze data
    (indicators, df) = await check_and_analyze(update, user_id, preferences, context.args)

    await update.message.reply_text(str(indicators))
    
    df = df.reset_index(drop=True)

async def start_signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /start_signals <symbol> <period_in_minutes>
    Example: /start_signals BTCUSDT 5
    - This means: "Check BTCUSDT on timeframe every 5 minutes"
    """
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Parse user arguments
    args = context.args  # list of strings
    if len(args) < 2:
        await update.message.reply_text(
            "Usage: /start_signals <symbol> <period_in_minutes>"
        )
        return

    symbol = args[0]
    try:
        period_minutes = int(args[1])
    except ValueError:
        await update.message.reply_text("Invalid period. Must be an integer (minutes).")
        return

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
        f"✅ Auto-signals started for {symbol}, every {period_minutes} minutes."
    )

async def stop_signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /stop_signals
    Cancels the user's signal job if it exists.
    """
    user_id = update.effective_user.id

    if user_id in auto_signal_jobs:
        job_ref = auto_signal_jobs[user_id]
        job_ref.schedule_removal()
        del auto_signal_jobs[user_id]
        await update.message.reply_text("✅ Auto-signals stopped.")
    else:
        await update.message.reply_text("No auto-signals are running for you.")

async def auto_signal_job(context: CallbackContext):
    """
    This function is called periodically by the JobQueue (run_repeating).
    It fetches signals and sends them to the user if needed.
    """
    job_data = context.job.data
    user_id = job_data["user_id"]
    chat_id = job_data["chat_id"]
    symbol = job_data["symbol"]

    # 1) Fetch user preferences (if you need them) from DB
    # preferences = get_user_preferences(user_id)

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

if __name__ == "__main__":
    from database import init_db

    init_db()
    TOKEN = os.getenv('API_TELEGRAM_KEY')

    app = ApplicationBuilder().token(TOKEN).build()

    job_queue = app.job_queue

    app.add_handler(CommandHandler("chart", send_crypto_chart))
    app.add_handler(CommandHandler("chart_1000", send_crypto_chart_1000)) # obsolete
    app.add_handler(CommandHandler("text_result", send_text_data))
    app.add_handler(CommandHandler("select_indicators", select_indicators))
    app.add_handler(CallbackQueryHandler(handle_indicator_selection))

    app.add_handler(CommandHandler("start_signals", start_signals_command))
    app.add_handler(CommandHandler("stop_signals", stop_signals_command))

    print("Bot is running...")
    app.run_polling()
