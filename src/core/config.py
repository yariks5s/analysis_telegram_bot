"""
Configuration module for CryptoBot.

This module centralizes configuration settings and constants for the application.
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API keys and tokens
TELEGRAM_API_TOKEN = os.getenv("API_TELEGRAM_KEY")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

# API endpoints
API_URL = "https://api.bybit.com/v5/market/kline"

# Database settings
DATABASE_PATH = "preferences.db"

# Time intervals mapping
VALID_INTERVALS = {
    "1m": "1",    # 1 minute
    "5m": "5",    # 5 minutes
    "15m": "15",  # 15 minutes
    "30m": "30",  # 30 minutes
    "1h": "60",   # 1 hour
    "4h": "240",  # 4 hours
    "1d": "D",    # 1 day
    "1w": "W",    # 1 week
}

# Application defaults
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1h"
DEFAULT_HOURS = 24
DEFAULT_CANDLE_LIMIT = 200
MAX_SIGNALS_PER_USER = 10

# Logging configuration
def setup_logging():
    """Configure the logger for the application."""
    logger = logging.getLogger("cryptobot")
    logger.setLevel(logging.INFO)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Add handlers to logger if not already added
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger

# Create logger instance
logger = setup_logging()
