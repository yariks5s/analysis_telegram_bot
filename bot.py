import asyncio

from helpers import check_and_analyze, input_sanity_check_analyzing, check_signal_limit
from plot_build_helpers import plot_price_chart
from message_handlers import (
    select_indicators,
    handle_indicator_selection,
    manage_signals,
    handle_signal_menu_callback,
    handle_signal_text_input,
    CHOOSING_ACTION,
    TYPING_SIGNAL_DATA,
)

from signal_detection import (
    generate_price_prediction_signal_proba,
    createSignalJob,
    deleteSignalJob,
    initialize_jobs,
)

from database import get_user_preferences
from utils import plural_helper

from telegram import Update  # type: ignore
from telegram.ext import (  # type: ignore
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackContext,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ConversationHandler,
)

from dotenv import load_dotenv  # type: ignore

import os
import logging

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
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

    # 1) Generate the probability-based signal:
    _, _, _, reason_str = generate_price_prediction_signal_proba(df, indicators)

    # Send the chart to the user
    with open(chart_path, "rb") as f:
        await context.bot.send_photo(chat_id=chat_id, photo=f)

    await update.message.reply_text(
        f"  {reason_str}"
    )  # Indentation needed for correct representation of the message

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


async def create_signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Command handler to create a signal job.
    Usage: /create_signal <SYMBOL> <MINUTES> [<IS_WITH_CHART>]
    Example: /create_signal BTCUSDT 60 True

    Note: is_with_chart is an optional argunent. Default value is false
    """
    if await check_signal_limit(update):
        return

    args = context.args
    pair = await input_sanity_check_analyzing(True, args, update)
    if not pair:
        await update.message.reply_text(
            f"Usage: /create_signal <symbol> <period_in_minutes> [<is_with_chart>], you've sent {len(args)} argument{plural_helper(len(args))}."
        )
    else:
        try:
            await createSignalJob(pair[0], pair[1], pair[2], update, context)
        except Exception as e:
            print(f"Unexpected error: {e}")
            await update.message.reply_text("❌ An unexpected error occurred.")


async def delete_signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Command handler to delete a specific signal job.
    Usage: /delete_signal <SYMBOL>
    Example: /delete_signal BTCUSDT
    """
    args = context.args
    pair = await input_sanity_check_analyzing(False, args, update)
    if not pair:
        await update.message.reply_text(
            f"Usage: /delete_signal <symbol>, you've sent {len(args)} argument{plural_helper(len(args))}."
        )
    else:
        try:
            await deleteSignalJob(pair[0], update)
        except Exception as e:
            print(f"Unexpected error: {e}")
            await update.message.reply_text("❌ An unexpected error occurred.")


async def initialize_jobs_handler(application):
    """
    Initialize all user signal jobs from the database when the application starts.
    """
    await initialize_jobs(application)


if __name__ == "__main__":
    from database import init_db

    init_db()
    TOKEN = os.getenv("API_TELEGRAM_KEY")

    app = ApplicationBuilder().token(TOKEN).build()

    job_queue = app.job_queue

    app.add_handler(CommandHandler("chart", send_crypto_chart))
    app.add_handler(CommandHandler("text_result", send_text_data))
    app.add_handler(CommandHandler("preferences", select_indicators))
    app.add_handler(
        CallbackQueryHandler(handle_indicator_selection, pattern=r"^indicator_")
    )

    app.add_handler(CommandHandler("create_signal", create_signal_command))
    app.add_handler(CommandHandler("delete_signal", delete_signal_command))

    manage_signals_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("manage_signals", manage_signals)],
        states={
            CHOOSING_ACTION: [
                CallbackQueryHandler(handle_signal_menu_callback),
            ],
            TYPING_SIGNAL_DATA: [
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND, handle_signal_text_input
                ),
            ],
        },
        fallbacks=[
            # Could add a "/cancel" fallback
        ],
    )
    app.add_handler(manage_signals_conv_handler)

    # Initialize jobs after the bot starts
    app.job_queue.run_once(
        lambda _: asyncio.create_task(initialize_jobs_handler(app)), when=0
    )

    print("Bot is running...")
    app.run_polling()
