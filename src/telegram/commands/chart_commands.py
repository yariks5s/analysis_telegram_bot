from telegram import Update  # type: ignore
from telegram.ext import ContextTypes, CallbackContext  # type: ignore

# These imports will need to be updated once all files are restructured
from src.analysis.utils.helpers import (
    check_and_analyze,
    input_sanity_check_analyzing,
    input_sanity_check_historical,
)
from src.visualization.plot_builder import plot_price_chart
from src.database.operations import get_user_preferences
from src.telegram.signals.detection import generate_price_prediction_signal_proba
from src.api.data_fetcher import fetch_candles, analyze_data


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
        dark_mode=preferences["dark_mode"],
    )
    if chart_path is None:
        await update.message.reply_text("Error generating the chart. Please try again.")
        return

    # Generate the probability-based signal
    _, _, _, reason_str, _ = generate_price_prediction_signal_proba(df, indicators)

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


async def send_historical_chart(update: Update, context: CallbackContext):
    """
    Telegram handler to fetch historical OHLC data for a user-specified crypto pair, interval, and tolerance at a specified timestamp.
    Usage: /history <symbol> <length> <interval> <tolerance> <timestamp>
    """
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    preferences = get_user_preferences(user_id)

    res = await input_sanity_check_historical(context.args, update)
    if not res:
        return
    symbol, length, interval, tolerance, timestamp_sec = res

    # Fetch historical candles ending at the specified timestamp
    df = fetch_candles(symbol, length, interval, timestamp=timestamp_sec)
    if df is None or df.empty:
        await update.message.reply_text(
            f"No data returned for {symbol} at timestamp {timestamp_sec}."
        )
        return

    # Analyze data
    indicators = analyze_data(df, preferences, tolerance)

    # Plot chart
    chart_path = plot_price_chart(
        df,
        indicators,
        show_legend=preferences["show_legend"],
        show_volume=preferences["show_volume"],
        dark_mode=preferences["dark_mode"],
    )
    if chart_path is None:
        await update.message.reply_text(
            "Error generating the historical chart. Please try again."
        )
        return

    # Send the chart to the user
    with open(chart_path, "rb") as f:
        await context.bot.send_photo(chat_id=chat_id, photo=f)

    await update.message.reply_text(f"Historical data for timestamp {timestamp_sec}")
