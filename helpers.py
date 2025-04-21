import pandas as pd  # type: ignore

from data_fetching_instruments import fetch_ohlc_data, analyze_data, fetch_candles
from utils import VALID_INTERVALS
from database import check_user_preferences, get_all_user_signal_requests


def calculate_macd(data: pd.DataFrame):
    """
    Calculate MACD and Signal Line.
    """
    short_ema = data["Close"].ewm(span=12, adjust=False).mean()
    long_ema = data["Close"].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def calculate_rsi(data: pd.DataFrame, period: int = 14):
    """
    Calculate Relative Strength Index (RSI).
    """
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


async def input_sanity_check_show(args, update) -> tuple:
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
        liq_lev_tolerance = float(args[3])
        if liq_lev_tolerance < 0 or liq_lev_tolerance > 1:
            await update.message.reply_text(
                f"Invalid liquidity level tolerance specified. It should be a number between 0 and 1"
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


async def fetch_data_and_get_indicators(res, preferences, update):
    symbol = {}
    hours = {}
    interval = {}
    liq_lev_tolerance = {}

    if len(res) >= 1:
        symbol = res[0]
    if len(res) >= 2:
        hours = res[1]
    if len(res) >= 3:
        interval = res[2]
    if len(res) >= 4:
        liq_lev_tolerance = res[3]

    if update:
        await update.message.reply_text(
            f"Fetching {symbol} price data for the last {hours} periods with interval {interval}, please wait..."
        )

    df = []
    if hours <= 200:
        df = fetch_ohlc_data(symbol, hours, interval)
    else:
        df = fetch_candles(symbol, hours, interval)
    if (df is None or df.empty) and update:
        await update.message.reply_text(
            f"Error fetching data for {symbol}. Please check the pair and try again."
        )
        return

    indicators = analyze_data(df, preferences, liq_lev_tolerance)
    return (indicators, df)


async def check_and_analyze(update, user_id, preferences, args):
    res = await input_sanity_check_show(args, update)

    if not res:
        return

    # Check if user selected indicators
    if not check_user_preferences(user_id):
        await update.message.reply_text(
            "Please select indicators using /preferences before requesting a chart."
        )
        return

    return await fetch_data_and_get_indicators(res, preferences, update)


async def check_signal_limit(update, build_signal_list_keyboard=None) -> bool:
    previous_signals = get_all_user_signal_requests(update.effective_user.id)

    if len(previous_signals) >= 10:
        if build_signal_list_keyboard is not None:
            await update.message.reply_text(
                text=f"You've reached the limit of signals ({len(previous_signals)}). If you want to add a new signal, please remove some of existing signals.",
                reply_markup=build_signal_list_keyboard(update.effective_user.id),
            )
        else:
            await update.message.reply_text(
                text=f"You've reached the limit of signals ({len(previous_signals)}). If you want to add a new signal, please remove some of existing signals."
            )

        return True

    return False
