from telegram import Update
from telegram.ext import ContextTypes, CallbackContext
from helpers import check_and_analyze, input_sanity_check_analyzing
from plot_build_helpers import plot_price_chart
from database import get_user_preferences
from signal_detection import generate_price_prediction_signal_proba


async def send_crypto_chart(update: Update, context: CallbackContext):
    """
    Telegram handler to fetch OHLC data, analyze indicators, and send the chart back to the user.
    """
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    preferences = get_user_preferences(user_id)

    # Analyze data
    (indicators, df) = await check_and_analyze(
        update, user_id, preferences, context.args
    )

    # Plot chart
    chart_path = plot_price_chart(
        df,
        indicators,
        show_legend=preferences["show_legend"],
        show_volume=preferences["show_volume"],
    )
    if chart_path is None:
        await update.message.reply_text("Error generating the chart. Please try again.")
        return

    # Generate the probability-based signal
    _, _, _, reason_str = generate_price_prediction_signal_proba(df, indicators)

    # Send the chart to the user
    with open(chart_path, "rb") as f:
        await context.bot.send_photo(chat_id=chat_id, photo=f)

    await update.message.reply_text(f"  {reason_str}")

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
    (indicators, df) = await check_and_analyze(
        update, user_id, preferences, context.args
    )

    await update.message.reply_text(str(indicators))

    df = df.reset_index(drop=True)
