import os
from dotenv import load_dotenv
import logging

# Telegram imports
from telegram import Update
from telegram.ext import CommandHandler, CallbackContext, ApplicationBuilder

from helpers import calculate_macd, calculate_rsi, input_sanity_check, VALID_INTERVALS, logger
from plot_build_helpers import plot_price_chart
from data_fetching_instruments import fetch_ohlc_data

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

load_dotenv()

async def send_crypto_chart(update: Update, context: CallbackContext):
    """
    Telegram handler to fetch OHLC data for a user-specified crypto pair, time period, and interval,
    plot the candlestick chart, and send it back to the user.
    Usage: /chart <symbol> <hours> <interval>, e.g. /chart BTCUSDT 42 15m
    """
    chat_id = update.effective_chat.id
    args = context.args

    res = await input_sanity_check(args, update)

    if (not res):
        return
    else:
        symbol = res[0]
        hours = res[1]
        interval = res[2]

    limit = min(hours, 200)  # Ensure limit respects Bybit's constraints
    await update.message.reply_text(f"Fetching {symbol} price data for the last {hours} periods with interval {interval}, please wait...")
    
    df = fetch_ohlc_data(symbol, limit, interval)
    if df is None or df.empty:
        await update.message.reply_text(f"Error fetching data for {symbol}. Please check the pair and try again.")
        return

    # Plot the chart with detected order blocks
    chart_path = plot_price_chart(df, symbol)
    if chart_path is None:
        await update.message.reply_text("Error generating the chart. Please try again.")
        return

    # Send the chart to the user
    with open(chart_path, 'rb') as f:
        await context.bot.send_photo(chat_id=chat_id, photo=f)


if __name__ == "__main__":
    TOKEN = os.getenv('API_TELEGRAM_KEY')

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("chart", send_crypto_chart))

    print("Bot is running...")
    app.run_polling()
