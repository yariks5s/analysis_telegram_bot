import os
from dotenv import load_dotenv
import logging

# Telegram imports
from telegram import Update
from telegram.ext import CommandHandler, CallbackContext, ApplicationBuilder, CallbackQueryHandler

from helpers import check_and_analyze, get_1000
from plot_build_helpers import plot_price_chart
from message_handlers import select_indicators, handle_indicator_selection
from signal_detection import generate_price_prediction_signal_proba

from database import get_user_preferences

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

if __name__ == "__main__":
    from database import init_db

    init_db()
    TOKEN = os.getenv('API_TELEGRAM_KEY')

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("chart", send_crypto_chart))
    app.add_handler(CommandHandler("chart_1000", send_crypto_chart_1000))
    app.add_handler(CommandHandler("text_result", send_text_data))
    app.add_handler(CommandHandler("select_indicators", select_indicators))
    app.add_handler(CallbackQueryHandler(handle_indicator_selection))

    print("Bot is running...")
    app.run_polling()
