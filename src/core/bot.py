"""
Core bot module for CryptoBot.

This is the main module for the Telegram bot, handling initialization, setup,
and running the bot with all necessary handlers.
"""

import asyncio
import os
import logging
from dotenv import load_dotenv

from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ConversationHandler,
)

# These imports will need to be updated once we've reorganized all files
from src.database.operations import init_db
from src.telegram.handlers import (
    select_indicators,
    handle_indicator_selection,
    manage_signals,
    handle_signal_menu_callback,
    handle_signal_text_input,
    CHOOSING_ACTION,
    TYPING_SIGNAL_DATA,
)
from src.telegram.signals.detection import initialize_jobs
from src.telegram.commands.chart_commands import (
    send_crypto_chart,
    send_text_data,
    send_historical_chart,
)
from src.telegram.commands.signal_commands import (
    create_signal_command,
    delete_signal_command,
)
from src.telegram.commands.help_commands import help_command
from src.telegram.commands.db_commands import (
    execute_sql_command,
    show_tables_command,
    describe_table_command,
)
from src.core.utils import logger
from src.core.error_handler import global_error_handler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def initialize_jobs_handler(application):
    """
    Initialize all user signal jobs from the database when the application starts.
    """
    await initialize_jobs(application)


def setup_handlers(app):
    """Setup all command and callback handlers for the bot"""
    # Chart commands
    app.add_handler(CommandHandler("chart", send_crypto_chart))
    app.add_handler(CommandHandler("text_result", send_text_data))
    app.add_handler(CommandHandler("history", send_historical_chart))

    # Preference commands
    app.add_handler(CommandHandler("preferences", select_indicators))
    app.add_handler(
        CallbackQueryHandler(handle_indicator_selection, pattern=r"^indicator_")
    )

    # Signal commands
    app.add_handler(CommandHandler("create_signal", create_signal_command))
    app.add_handler(CommandHandler("delete_signal", delete_signal_command))

    # Database commands
    app.add_handler(CommandHandler("sql", execute_sql_command))
    app.add_handler(CommandHandler("tables", show_tables_command))
    app.add_handler(CommandHandler("schema", describe_table_command))

    # Signal management conversation handler
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
        fallbacks=[],
    )
    app.add_handler(manage_signals_conv_handler)

    # Help command
    app.add_handler(CommandHandler("help", help_command))


def main():
    """Main function to initialize and run the bot"""
    # Load environment variables
    load_dotenv()

    # Initialize database
    init_db()

    # Get token from environment
    TOKEN = os.getenv("API_TELEGRAM_KEY")

    # Create bot application
    app = ApplicationBuilder().token(TOKEN).build()

    # Register global error handler
    app.add_error_handler(global_error_handler)

    # Setup all handlers
    setup_handlers(app)

    # Initialize jobs after the bot starts
    app.job_queue.run_once(
        lambda _: asyncio.create_task(initialize_jobs_handler(app)), when=0
    )

    # Start the bot
    logger.info("Bot is starting...")
    app.run_polling()
    logger.info("Bot started successfully")
