import os
from dotenv import load_dotenv
import logging

# Telegram imports
from telegram import Update
from telegram.ext import CommandHandler, CallbackContext, ApplicationBuilder

from helpers import check_and_analyze
from plot_build_helpers import plot_price_chart

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

load_dotenv()

async def send_crypto_chart(update: Update, context: CallbackContext):
    """
    Telegram handler to fetch OHLC data for a user-specified crypto pair, time period, interval, and liquidity level detection tolerance
    plot the candlestick chart, and send it back to the user.
    Usage: /chart <symbol> <hours> <interval> <tolerance>, e.g. /chart BTCUSDT 42 15m 0.03
    """
    chat_id = update.effective_chat.id
    (indicators, df) = await check_and_analyze(update, context)

    # Plot the chart with detected order blocks
    chart_path = plot_price_chart(df, indicators)
    if chart_path is None:
        await update.message.reply_text("Error generating the chart. Please try again.")
        return

    # Send the chart to the user
    with open(chart_path, 'rb') as f:
        await context.bot.send_photo(chat_id=chat_id, photo=f)
    
    df = df.reset_index(drop=True)

async def send_text_data(update: Update, context: CallbackContext):
    """
    Telegram handler to fetch OHLC data for a user-specified crypto pair, time period, interval, and liquidity level detection tolerance
    plot the candlestick chart, and send it back to the user.
    Usage: /text_result <symbol> <hours> <interval> <tolerance>, e.g. /text_result BTCUSDT 42 15m 0.03
    """
    (indicators, df) = await check_and_analyze(update, context)

    await update.message.reply_text(str(indicators))
    
    df = df.reset_index(drop=True)

if __name__ == "__main__":
    TOKEN = os.getenv('API_TELEGRAM_KEY')

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("chart", send_crypto_chart))
    app.add_handler(CommandHandler("text_result", send_text_data))

    print("Bot is running...")
    app.run_polling()
