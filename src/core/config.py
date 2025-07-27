"""
Configuration module for CryptoBot.

This module centralizes configuration settings and constants for the application.
"""

import os
import logging
from dotenv import load_dotenv
from src.core.utils import API_URL, VALID_INTERVALS

load_dotenv()

TELEGRAM_API_TOKEN = os.getenv("API_TELEGRAM_KEY")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

DATABASE_PATH = "preferences.db"

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1h"
DEFAULT_HOURS = 24
DEFAULT_CANDLE_LIMIT = 200
MAX_SIGNALS_PER_USER = 10


def setup_logging():
    """Configure the logger for the application."""
    logger = logging.getLogger("cryptobot")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Prevent propagation to root logger
    logger.propagate = False

    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger


logger = setup_logging()
