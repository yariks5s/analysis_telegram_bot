import datetime
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
from src.core.error_handler import handle_error


async def send_crypto_chart(update: Update, context: CallbackContext):
    """
    Telegram handler to fetch OHLC data, analyze indicators, and send the chart back to the user.
    """
    try:
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id

        try:
            preferences = get_user_preferences(user_id)
        except Exception as e:
            await handle_error(update, "database", 
                              "Failed to retrieve user preferences. Please try setting your preferences again using /preferences.", 
                              exception=e)
            return

        # Analyze data
        try:
            (indicators, df) = await check_and_analyze(
                update, user_id, preferences, context.args
            )
            if indicators is None or df is None or df.empty:
                # The check_and_analyze function already handles user messaging for specific errors
                return
        except Exception as e:
            await handle_error(update, "data_processing", 
                              "Failed to analyze data. Please check your inputs and try again.", 
                              exception=e)
            return

        # Plot chart
        try:
            chart_path = plot_price_chart(
                df,
                indicators,
                show_legend=preferences["show_legend"],
                show_volume=preferences["show_volume"],
                dark_mode=preferences["dark_mode"],
            )
            if chart_path is None:
                await handle_error(update, "chart_generation")
                return
        except Exception as e:
            await handle_error(update, "chart_generation", exception=e)
            return

        # Generate the probability-based signal
        try:
            _, _, _, reason_str, _ = generate_price_prediction_signal_proba(df, indicators)
        except Exception as e:
            await handle_error(update, "data_processing", 
                              "Could not generate price prediction signal. Sending chart without analysis.", 
                              exception=e)
            reason_str = "Analysis not available"

        # Send the chart to the user
        try:
            with open(chart_path, "rb") as f:
                await context.bot.send_photo(chat_id=chat_id, photo=f)
        except Exception as e:
            await handle_error(update, "unknown", "Failed to send chart. Please try again.", exception=e)
            return

        await update.message.reply_text(f"  {reason_str}")

        df = df.reset_index(drop=True)
    except Exception as e:
        await handle_error(update, "unknown", exception=e)


async def send_text_data(update: Update, context: CallbackContext):
    """
    Telegram handler to fetch OHLC data for a user-specified crypto pair, time period, interval, and liquidity level detection tolerance
    plot the candlestick chart, and send it back to the user.
    Usage: /text_result <symbol> <hours> <interval> <tolerance>, e.g. /text_result BTCUSDT 42 15m 0.03
    """
    try:
        user_id = update.effective_user.id

        try:
            preferences = get_user_preferences(user_id)
        except Exception as e:
            await handle_error(update, "database", 
                              "Failed to retrieve user preferences. Please try setting your preferences again using /preferences.", 
                              exception=e)
            return

        # Analyze data
        try:
            (indicators, df) = await check_and_analyze(
                update, user_id, preferences, context.args
            )
            if indicators is None or df is None or df.empty:
                # The check_and_analyze function already handles user messaging for specific errors
                return
        except Exception as e:
            await handle_error(update, "data_processing", 
                              "Failed to analyze data. Please check your inputs and try again.", 
                              exception=e)
            return

        try:
            await update.message.reply_text(str(indicators))
        except Exception as e:
            await handle_error(update, "unknown", "Failed to send analysis results. Please try again.", exception=e)
            return

        df = df.reset_index(drop=True)
    except Exception as e:
        await handle_error(update, "unknown", exception=e)


async def send_historical_chart(update: Update, context: CallbackContext):
    """
    Telegram handler to fetch historical OHLC data for a user-specified crypto pair, interval, and tolerance at a specified timestamp.
    Usage: /history <symbol> <length> <interval> <tolerance> <timestamp>
    """
    try:
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id

        try:
            preferences = get_user_preferences(user_id)
        except Exception as e:
            await handle_error(update, "database", 
                              "Failed to retrieve user preferences. Please try setting your preferences again using /preferences.", 
                              exception=e)
            return

        try:
            res = await input_sanity_check_historical(context.args, update)
            if not res:
                # The input_sanity_check_historical function already handles user messaging for validation errors
                return
            symbol, length, interval, tolerance, timestamp_sec = res
        except Exception as e:
            await handle_error(update, "invalid_input", 
                              "Invalid command parameters. Usage: /history <symbol> <length> <interval> <tolerance> <timestamp>", 
                              exception=e)
            return

        if timestamp_sec > datetime.datetime.now().timestamp():
            await update.message.reply_text(
                "❌ Invalid timestamp. Must be a Unix epoch timestamp in the past."
            )
            return

        # Fetch historical candles ending at the specified timestamp
        try:
            df = fetch_candles(symbol, length, interval, timestamp=timestamp_sec)
            if df is None or df.empty:
                await handle_error(update, "data_fetch", 
                                  f"No data returned for {symbol} at timestamp {timestamp_sec}. The data may not be available for that period.")
                return
        except Exception as e:
            await handle_error(update, "data_fetch", 
                              f"Failed to fetch historical data for {symbol}. Please check the symbol and timestamp.", 
                              exception=e)
            return

        # Analyze data
        try:
            indicators = analyze_data(df, preferences, tolerance)
        except Exception as e:
            await handle_error(update, "data_processing", 
                              "Failed to analyze historical data. Please try with different parameters.", 
                              exception=e)
            return

        # Plot chart
        try:
            chart_path = plot_price_chart(
                df,
                indicators,
                show_legend=preferences["show_legend"],
                show_volume=preferences["show_volume"],
                dark_mode=preferences["dark_mode"],
            )
            if chart_path is None:
                await handle_error(update, "chart_generation")
                return
        except Exception as e:
            await handle_error(update, "chart_generation", exception=e)
            return

        # Send the chart to the user
        try:
            with open(chart_path, "rb") as f:
                await context.bot.send_photo(chat_id=chat_id, photo=f)
        except Exception as e:
            await handle_error(update, "unknown", "Failed to send chart. Please try again.", exception=e)
            return

        await update.message.reply_text(f"Historical data for timestamp {timestamp_sec}")
    except Exception as e:
        await handle_error(update, "unknown", exception=e)
