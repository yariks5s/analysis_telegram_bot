import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# NOTE: Uncomment to enable logging to a file
# Create file handler for training logs
# log_file = f'logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
# file_handler = logging.FileHandler(log_file)
# file_handler.setLevel(logging.INFO)
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

# Create console handler for training logs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)

# Prevent propagation to root logger
logger.propagate = False

VALID_INTERVALS = {
    "1m": "1",  # 1 minute
    "5m": "5",  # 5 minutes
    "15m": "15",  # 15 minutes
    "30m": "30",  # 30 minutes
    "1h": "60",  # 1 hour
    "4h": "240",  # 4 hours
    "1d": "D",  # 1 day
    "1w": "W",  # 1 week
}

# Store user-selected indicators temporarily
user_selected_indicators = {}

auto_signal_jobs = {}

API_URL = "https://api.bybit.com/v5/market/kline"


def plural_helper(num: int) -> str:
    """
    Will help to decide whether we need to use the plural form of the words when communicating with the user
    """
    if num != 1:
        return "s"
    return ""


def create_true_preferences():
    preferences = {}

    preferences["order_blocks"] = True
    preferences["fvgs"] = True
    preferences["liquidity_levels"] = True
    preferences["breaker_blocks"] = True

    return preferences


def is_bullish(candle):
    """
    Detects whether the candle is bullish
    """
    return candle["Close"] > candle["Open"]


def is_bearish(candle):
    """
    Detects whether the candle is bullish
    """
    return candle["Close"] < candle["Open"]
