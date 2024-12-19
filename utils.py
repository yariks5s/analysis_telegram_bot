import logging

logger = logging.getLogger(__name__)

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
