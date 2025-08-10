"""
Error handling utilities for CryptoBot.

This module provides centralized error handling functionality for logging errors
and formatting user-friendly error messages.
"""

import logging
import traceback
from typing import Optional, Union
from telegram import Update

logger = logging.getLogger(__name__)

# Error message templates
ERROR_MESSAGES = {
    "network": "Network error occurred. Please try again later.",
    "data_fetch": "Failed to fetch data. Please check the symbol and try again.",
    "data_processing": "Error processing data. Please try with different parameters.",
    "chart_generation": "Error generating chart. Please try again with different settings.",
    "invalid_input": "Invalid input provided. Please check your command syntax.",
    "database": "Database operation failed. Please try again.",
    "timeout": "Operation timed out. Please try again.",
    "api_limit": "API rate limit reached. Please try again later.",
    "permission": "You don't have permission to perform this action.",
    "unknown": "An unexpected error occurred. Please try again later.",
}


async def handle_error(
    update: Update,
    error_type: str = "unknown",
    custom_message: Optional[str] = None,
    exception: Optional[Exception] = None,
    notify_user: bool = True,
) -> None:
    """
    Handle errors by logging them and optionally sending a user-friendly message.

    Args:
        update: The Telegram update object to respond to
        error_type: Type of error from ERROR_MESSAGES dictionary
        custom_message: Optional custom message to override the template
        exception: The exception object if available
        notify_user: Whether to send a message to the user
    """
    # Log the error with details
    if exception:
        logger.error(f"Error: {error_type} - {str(exception)}")
        logger.debug(traceback.format_exc())
    else:
        logger.error(
            f"Error: {error_type} - {custom_message or ERROR_MESSAGES.get(error_type, ERROR_MESSAGES['unknown'])}"
        )

    if notify_user and update and update.effective_chat:
        message = custom_message or ERROR_MESSAGES.get(
            error_type, ERROR_MESSAGES["unknown"]
        )
        try:
            await update.message.reply_text(f"❌ {message}")
        except Exception as e:
            # Fallback if message reply fails
            try:
                await update.callback_query.message.reply_text(f"❌ {message}")
            except Exception:
                logger.error(f"Failed to send error message to user: {e}")


async def global_error_handler(update: object, context):
    """
    Global error handler for the application.
    This catches all unhandled exceptions in handlers.

    Args:
        update: The update that triggered the error
        context: The context containing the error
    """
    logger.error(f"Unhandled exception: {context.error}")
    logger.debug(traceback.format_exc())
    
    if hasattr(context, 'error') and isinstance(context.error, Exception):
        from src.core.logging_utils import log_error_with_stacktrace
        log_error_with_stacktrace("unhandled_exception", context.error, 
                                 update if isinstance(update, Update) else None)

    # Ensure we have a valid update object
    if isinstance(update, Update) and update.effective_chat:
        try:
            await update.effective_chat.send_message(
                "❌ An unexpected error occurred while processing your request. "
                "Please try again later.\n\n"
                "If you think this is a bug, please report it here: https://github.com/yariks5s/analysis_telegram_bot/issues"
            )
        except Exception as e:
            logger.error(f"Failed to send global error message: {e}")
            from src.core.logging_utils import log_error_with_stacktrace
            log_error_with_stacktrace("notification_error", e, 
                                     update if isinstance(update, Update) else None)
