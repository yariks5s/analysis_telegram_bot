"""
Analysis utility helpers for CryptoBot.

This module contains various utility functions for technical analysis,
data validation, and processing market data.
"""

import logging
import pandas as pd  # type: ignore

from src.api.data_fetcher import fetch_ohlc_data, analyze_data, fetch_candles
from src.core.utils import VALID_INTERVALS
from src.database.operations import check_user_preferences, get_all_user_signal_requests
from src.core.error_handler import handle_error

logger = logging.getLogger(__name__)


def calculate_macd(data: pd.DataFrame):
    """
    Calculate MACD and Signal Line.

    Args:
        data: DataFrame with Close prices

    Returns:
        Tuple of (MACD, Signal Line)
    """
    short_ema = data["Close"].ewm(span=12, adjust=False).mean()
    long_ema = data["Close"].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def calculate_rsi(data: pd.DataFrame, period: int = 14):
    """
    Calculate Relative Strength Index (RSI).

    Args:
        data: DataFrame with Close prices
        period: RSI calculation period

    Returns:
        Series of RSI values
    """
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


async def input_sanity_check_show(args, update) -> tuple:
    """
    Validate input arguments for chart display.

    Args:
        args: Command arguments
        update: Telegram update object

    Returns:
        Tuple of (symbol, hours, interval, liq_lev_tolerance) or empty tuple if invalid
    """
    # Default values
    symbol = "BTCUSDT"
    hours = 24
    interval = "1h"
    liq_lev_tolerance = 0

    if len(args) >= 1:
        symbol = args[0].upper()
    if len(args) >= 2:
        try:
            hours = int(args[1])
        except ValueError:
            await update.message.reply_text(
                "Invalid hours specified. Please provide a valid number."
            )
            return tuple()
    if len(args) >= 3:
        interval = args[2].lower()
        if interval not in VALID_INTERVALS:
            await update.message.reply_text(
                f"Invalid interval specified. Valid intervals are: {', '.join(VALID_INTERVALS.keys())}"
            )
            return tuple()
    if len(args) >= 4:
        try:
            liq_lev_tolerance = float(args[3])
            if liq_lev_tolerance < 0 or liq_lev_tolerance > 1:
                await update.message.reply_text(
                    f"Invalid liquidity level tolerance specified. It should be a number between 0 and 1"
                )
                return tuple()
        except ValueError:
            await update.message.reply_text(
                "Invalid tolerance value. Please provide a valid number."
            )
            return tuple()

    return (symbol, hours, interval, liq_lev_tolerance)


async def input_sanity_check_analyzing(is_start: bool, args, update) -> tuple:
    # Default values
    symbol = "BTCUSDT"
    period_minutes = 60
    is_with_chart = False

    if is_start:
        if len(args) < 2:
            await update.message.reply_text(
                f"❌ Please specify the currency pair and sending period to create a signal."
            )
            return tuple()
    else:
        if len(args) != 1:
            await update.message.reply_text(
                f"❌ Please specify the currency pair to delete."
            )
            return tuple()

    if len(args) >= 1:
        symbol = str(args[0]).upper()
    if len(args) >= 2:
        try:
            period_minutes = int(args[1])
        except ValueError:
            await update.message.reply_text(
                "❌ Invalid period. Must be an integer (minutes)."
            )
            return tuple()
    if len(args) >= 3:
        if str(args[2]).lower() in ["true", "yes", "1"]:
            is_with_chart = True
        elif str(args[2]).lower() in ["false", "no", "0"]:
            is_with_chart = False
        else:
            await update.message.reply_text(
                "❌ Invalid value for the third argument. Must be a true/false (yes/no, 1/0) depending if you want to receive a chart along with a signal."
            )
            return tuple()

    if len(args) > (3 if is_start else 1):
        await update.message.reply_text(
            "❌ Invalid number of arguments. Please check your request."
        )
        return tuple()

    return (symbol, period_minutes, is_with_chart)


async def input_sanity_check_historical(args, update) -> tuple:
    """
    Parse and validate arguments for historical data command.
    Usage: /history <symbol> <length> <interval> <tolerance> <timestamp>
    Timestamp must be Unix epoch in seconds.

    Args:
        args: Command arguments
        update: Telegram update object

    Returns:
        Tuple of validated arguments or empty tuple if invalid
    """
    if len(args) < 5:
        await update.message.reply_text(
            "Usage: /history <symbol> <length> <interval> <tolerance> <timestamp>"
        )
        return tuple()
    if len(args) > 5:
        await update.message.reply_text(
            "❌ Invalid number of arguments. Please provide exactly 5 arguments."
        )
        return tuple()
    symbol = args[0].upper()
    try:
        length = int(args[1])
    except ValueError:
        await update.message.reply_text("❌ Invalid length. Must be an integer.")
        return tuple()
    interval = args[2].lower()
    if interval not in VALID_INTERVALS:
        await update.message.reply_text(
            f"❌ Invalid interval. Valid intervals: {', '.join(VALID_INTERVALS.keys())}"
        )
        return tuple()
    try:
        tolerance = float(args[3])
    except ValueError:
        await update.message.reply_text(
            "❌ Invalid tolerance. Must be a number between 0 and 1."
        )
        return tuple()
    if tolerance < 0 or tolerance > 1:
        await update.message.reply_text(
            "❌ Invalid tolerance. Must be between 0 and 1."
        )
        return tuple()
    try:
        timestamp_sec = int(args[4])
    except ValueError:
        await update.message.reply_text(
            "❌ Invalid timestamp. Must be Unix epoch seconds."
        )
        return tuple()
    return (symbol, length, interval, tolerance, timestamp_sec)


async def fetch_data_and_get_indicators(
    symbol, hours, interval, preferences, liq_lev_tolerance=0, update=None
):
    """
    Fetch market data and calculate indicators.

    Args:
        symbol: Trading pair symbol
        hours: Number of hours/periods to fetch
        interval: Time interval
        preferences: User preferences for indicators
        liq_lev_tolerance: Tolerance for liquidity level detection
        update: Optional Telegram update object for notifications

    Returns:
        Tuple of (indicators, dataframe)
    """
    try:
        if update:
            await update.message.reply_text(
                f"Fetching {symbol} price data for the last {hours} periods with interval {interval}, please wait..."
            )

        df = None
        try:
            if hours <= 200:
                df = fetch_ohlc_data(symbol, hours, interval)
            else:
                df = fetch_candles(symbol, hours, interval)
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            if update:
                await handle_error(
                    update,
                    "data_fetch",
                    f"Error fetching data for {symbol}. The pair may not exist or the service might be temporarily unavailable.",
                    exception=e,
                )
            return (None, None)

        if df is None or df.empty:
            if update:
                await handle_error(
                    update,
                    "data_fetch",
                    f"No data returned for {symbol}. Please check the pair name and try again.",
                )
            return (None, None)

        try:
            indicators = analyze_data(df, preferences, liq_lev_tolerance)
            return (indicators, df)
        except Exception as e:
            logger.error(f"Error analyzing data: {str(e)}")
            if update:
                await handle_error(
                    update,
                    "data_processing",
                    "Error analyzing market data. Please try different parameters or contact support.",
                    exception=e,
                )
            return (None, None)

    except Exception as e:
        logger.error(f"Unexpected error in fetch_data_and_get_indicators: {str(e)}")
        if update:
            await handle_error(update, "unknown", exception=e)
        return (None, None)


async def check_and_analyze(update, user_id, preferences, args):
    """
    Validate input, check user preferences, and fetch data with indicators.

    Args:
        update: Telegram update object
        user_id: User ID
        preferences: User preferences
        args: Command arguments

    Returns:
        Tuple of (indicators, dataframe) or None if error
    """
    try:
        # Input validation
        try:
            res = await input_sanity_check_show(args, update)
            if not res:
                # input_sanity_check_show already sends appropriate error messages
                return None, None
        except Exception as e:
            logger.error(f"Error in input validation: {str(e)}")
            await handle_error(
                update,
                "invalid_input",
                "Invalid command parameters. Please check the syntax and try again.",
                exception=e,
            )
            return None, None

        # Check if user selected indicators
        try:
            if not check_user_preferences(user_id):
                await update.message.reply_text(
                    "Please select indicators using /preferences before requesting a chart."
                )
                return None, None
        except Exception as e:
            logger.error(f"Error checking user preferences: {str(e)}")
            await handle_error(
                update,
                "database",
                "Could not verify your preferences. Please try setting your preferences again using /preferences.",
                exception=e,
            )
            return None, None

        # Fetch and analyze data
        return await fetch_data_and_get_indicators(
            res[0], res[1], res[2], preferences, res[3], update
        )
    except Exception as e:
        logger.error(f"Unexpected error in check_and_analyze: {str(e)}")
        await handle_error(update, "unknown", exception=e)
        return None, None


async def check_signal_limit(update):
    """
    Check if the user has reached the signal limit.

    Args:
        update: Telegram update object

    Returns:
        bool: True if limit reached, False otherwise
    """
    user_id = update.effective_user.id
    previous_signals = get_all_user_signal_requests(user_id)

    if len(previous_signals) >= 10:
        await update.message.reply_text(
            f"You've reached the limit of signals ({len(previous_signals)}). "
            f"If you want to add a new signal, please remove some of existing signals."
        )
        return True

    return False
