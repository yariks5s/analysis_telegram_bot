"""
Telegram module for CryptoBot.

This module provides Telegram bot functionality and notifications.
"""

try:
    from src.telegram.notifier import TelegramNotifier, create_notifier
    __all__ = ["TelegramNotifier", "create_notifier"]
except ImportError:
    pass
