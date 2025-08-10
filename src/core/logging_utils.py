"""
Logging utilities for CryptoBot.

This module provides enhanced logging functionality to track user interactions
and capture detailed error information.
"""

import logging
import traceback
import json
import os
from datetime import datetime
from functools import wraps
from telegram import Update
from telegram.ext import ContextTypes

# ensure logs directory exists
log_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"
)
os.makedirs(log_dir, exist_ok=True)

query_logger = logging.getLogger("user_queries")
query_logger.setLevel(logging.INFO)

query_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# rotating file handler for query logs - 5MB per file, keeping 5 backup files
from logging.handlers import RotatingFileHandler

query_file_handler = RotatingFileHandler(
    "logs/user_queries.log", maxBytes=5 * 1024 * 1024, backupCount=5  # 5MB
)
query_file_handler.setFormatter(query_formatter)
query_logger.addHandler(query_file_handler)

# this logger doesn't propagate to the root logger
query_logger.propagate = False

error_logger = logging.getLogger("error_details")
error_logger.setLevel(logging.ERROR)

error_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

error_file_handler = RotatingFileHandler(
    "logs/error_details.log", maxBytes=5 * 1024 * 1024, backupCount=10  # 5MB
)
error_file_handler.setFormatter(error_formatter)
error_logger.addHandler(error_file_handler)

error_logger.propagate = False


def log_user_query(update: Update):
    """
    Log user query details to a dedicated log file.

    Args:
        update: The Telegram update object containing user info and message
    """
    if update and update.effective_message and update.effective_user:
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        chat_id = update.effective_chat.id if update.effective_chat else "Unknown"
        message_text = update.effective_message.text or "Unknown"

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "username": username,
            "chat_id": chat_id,
            "query": message_text,
        }

        query_logger.info(f"User Query: {json.dumps(log_entry)}")


def log_error_with_stacktrace(
    error_type: str, exception: Exception, update: Update = None
):
    """
    Log error with full stacktrace and user details if available.

    Args:
        error_type: Type of error
        exception: The exception object
        update: Optional Telegram update object for user context
    """
    stack_trace = traceback.format_exc()

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "error_type": error_type,
        "error_message": str(exception),
    }

    if update and update.effective_user:
        log_entry["user_id"] = update.effective_user.id
        log_entry["username"] = update.effective_user.username or "Unknown"
        if update.effective_message:
            log_entry["query"] = update.effective_message.text or "Unknown"

    error_logger.error(f"Error: {json.dumps(log_entry)}\nStacktrace:\n{stack_trace}")


def log_message_decorator(func):
    """
    Decorator to log user messages before they are processed by handlers.

    Args:
        func: The handler function to decorate

    Returns:
        Decorated function that logs user queries
    """

    @wraps(func)
    async def wrapper(
        update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs
    ):
        log_user_query(update)

        try:
            return await func(update, context, *args, **kwargs)
        except Exception as e:
            log_error_with_stacktrace("handler_exception", e, update)
            raise

    return wrapper
