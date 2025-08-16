"""
Command handlers for retrieving and displaying signal history.
"""

import json
import os
import logging
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from telegram.error import BadRequest

logger = logging.getLogger(__name__)

from src.database.operations import (
    get_user_signal_history,
    get_signal_history_by_date_range,
)
from src.analysis.utils.export import export_user_signals

from src.core.error_handler import handle_error

HISTORY_PERIOD = "hist_period:"
HISTORY_PAIR = "hist_pair:"
HISTORY_PAGE = "hist_page:"
EXPORT_CSV = "export_csv:"
EXPORT_JSON = "export_json:"

EXPORT_DIR = "exports"


async def command_signal_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for /signal_history command.
    Shows the user's signal history with filtering options.
    """
    try:
        user_id = update.effective_user.id

        signals = get_user_signal_history(user_id, limit=10)

        keyboard = [
            [
                InlineKeyboardButton("Last 24h", callback_data=f"{HISTORY_PERIOD}24h"),
                InlineKeyboardButton("Last 7d", callback_data=f"{HISTORY_PERIOD}7d"),
                InlineKeyboardButton("Last 30d", callback_data=f"{HISTORY_PERIOD}30d"),
            ]
        ]

        if signals:
            # Get unique currency pairs
            currency_pairs = set(signal["currency_pair"] for signal in signals)
            pair_buttons = []
            for pair in currency_pairs:
                pair_buttons.append(
                    InlineKeyboardButton(pair, callback_data=f"{HISTORY_PAIR}{pair}")
                )
            for i in range(0, len(pair_buttons), 3):
                keyboard.append(pair_buttons[i : i + 3])

        reply_markup = InlineKeyboardMarkup(keyboard)

        if signals:
            message = "📜 *Your Signal History*\n\n"
            message += format_signal_history(signals)
            message += "\n\nFilter by time period or currency pair:"
        else:
            message = "No signal history found. Signals will be recorded when they are generated."

        await update.message.reply_text(
            message, reply_markup=reply_markup, parse_mode="Markdown"
        )

    except Exception as e:
        await handle_error(
            update, "data_fetch", "Failed to retrieve signal history.", exception=e
        )


async def button_history_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles callbacks from the history command buttons.
    """
    query = update.callback_query

    await query.answer()

    user_id = update.effective_user.id
    callback_data = query.data

    try:
        if callback_data.startswith(HISTORY_PERIOD):
            period = callback_data[len(HISTORY_PERIOD) :]
            signals = filter_by_period(user_id, period)

            keyboard = build_history_keyboard(signals)
            reply_markup = InlineKeyboardMarkup(keyboard)

            if signals:
                message = f"📜 *Signal History - Last {period}*\n\n"
                message += format_signal_history(signals)
            else:
                message = f"No signals found in the last {period}."

            await safe_edit_message(query, message, reply_markup)

        # Handle currency pair filtering
        elif callback_data.startswith(HISTORY_PAIR):
            pair = callback_data[len(HISTORY_PAIR) :]
            signals = get_user_signal_history(user_id, limit=20, currency_pair=pair)

            keyboard = build_history_keyboard(signals, selected_pair=pair)
            reply_markup = InlineKeyboardMarkup(keyboard)

            if signals:
                message = f"📜 *Signal History - {pair}*\n\n"
                message += format_signal_history(signals)
            else:
                message = f"No signals found for {pair}."

            await safe_edit_message(query, message, reply_markup)

        elif callback_data.startswith(HISTORY_PAGE):
            # TODO: Implement pagination for large result sets
            pass

        # export requests
        elif callback_data.startswith(EXPORT_CSV):
            pair = callback_data[len(EXPORT_CSV) :]
            currency_pair = None if pair == "all" else pair
            await handle_export(update, context, user_id, "csv", currency_pair)

        elif callback_data.startswith(EXPORT_JSON):
            pair = callback_data[len(EXPORT_JSON) :]
            currency_pair = None if pair == "all" else pair
            await handle_export(update, context, user_id, "json", currency_pair)

    except Exception as e:
        await handle_error(
            update,
            "data_fetch",
            "Failed to retrieve filtered signal history.",
            exception=e,
        )


def filter_by_period(user_id, period):
    """
    Filter signals by time period.

    Args:
        user_id: User ID
        period: Time period string (24h, 7d, 30d)

    Returns:
        List of signals in the given period
    """
    now = datetime.now()

    if period == "24h":
        start_date = (now - timedelta(days=1)).isoformat()
    elif period == "7d":
        start_date = (now - timedelta(days=7)).isoformat()
    elif period == "30d":
        start_date = (now - timedelta(days=30)).isoformat()
    else:
        # Default to 24h
        start_date = (now - timedelta(days=1)).isoformat()

    end_date = now.isoformat()

    return get_signal_history_by_date_range(user_id, start_date, end_date)


def build_history_keyboard(signals, selected_pair=None):
    """
    Build the keyboard for history filtering.

    Args:
        signals: List of signals
        selected_pair: Currently selected currency pair

    Returns:
        List of button rows for the keyboard
    """
    keyboard = [
        [
            InlineKeyboardButton("Last 24h", callback_data=f"{HISTORY_PERIOD}24h"),
            InlineKeyboardButton("Last 7d", callback_data=f"{HISTORY_PERIOD}7d"),
            InlineKeyboardButton("Last 30d", callback_data=f"{HISTORY_PERIOD}30d"),
        ]
    ]

    if signals:
        currency_pairs = set(signal["currency_pair"] for signal in signals)
        pair_buttons = []
        for pair in currency_pairs:
            label = f"▶ {pair}" if pair == selected_pair else pair
            pair_buttons.append(
                InlineKeyboardButton(label, callback_data=f"{HISTORY_PAIR}{pair}")
            )
        for i in range(0, len(pair_buttons), 3):
            keyboard.append(pair_buttons[i : i + 3])

    # Add 'Show All' button if a pair is selected
    if selected_pair:
        keyboard.append(
            [InlineKeyboardButton("Show All", callback_data=f"{HISTORY_PERIOD}all")]
        )
    
    export_buttons = []
    if selected_pair:
        export_buttons.extend(
            [
                InlineKeyboardButton(
                    "Export CSV", callback_data=f"{EXPORT_CSV}{selected_pair}"
                ),
                InlineKeyboardButton(
                    "Export JSON", callback_data=f"{EXPORT_JSON}{selected_pair}"
                ),
            ]
        )
    else:
        export_buttons.extend(
            [
                InlineKeyboardButton(
                    "Export CSV", callback_data=f"{EXPORT_CSV}all"
                ),
                InlineKeyboardButton(
                    "Export JSON", callback_data=f"{EXPORT_JSON}all"
                ),
            ]
        )
    
    if signals:
        keyboard.append(export_buttons)

    return keyboard


def format_signal_history(signals, max_signals=10):
    """
    Format the signal history for display.

    Args:
        signals: List of signals
        max_signals: Maximum number of signals to display

    Returns:
        Formatted message string
    """
    message = ""

    signals = signals[:max_signals]

    for i, signal in enumerate(signals):
        timestamp = datetime.fromisoformat(signal["timestamp"]).strftime(
            "%Y-%m-%d %H:%M"
        )
        pair = signal["currency_pair"]
        signal_type = signal["signal_type"]
        probability = signal["probability"]

        line = f"{i+1}. *{pair}* - {timestamp}\n"
        line += f"   Type: {signal_type}, Probability: {probability:.2f}\n"

        line += (
            f"   Entry: {signal['entry_price']:.2f}, SL: {signal['stop_loss']:.2f}, "
        )

        if "take_profit_1" in signal:
            take_profit = signal["take_profit_1"]
        elif "take_profit" in signal:
            take_profit = signal["take_profit"]
        else:
            take_profit = 0.0

        line += f"TP1: {take_profit:.2f}\n"

        reasons = (
            json.loads(signal["reasons"])
            if isinstance(signal["reasons"], str)
            else signal["reasons"]
        )
        if reasons and len(reasons) > 0:
            line += f"   Reasons: {', '.join(reasons[:3])}"
            if len(reasons) > 3:
                line += f" (+{len(reasons)-3} more)"

        message += line + "\n\n"

    if len(signals) == max_signals:
        message += f"_Showing {max_signals} most recent signals._"

    return message


async def safe_edit_message(query, text, reply_markup=None):
    """
    Safely edit a message, handling common Telegram API errors.

    Args:
        query: The callback query
        text: The new text content
        reply_markup: Optional keyboard markup
    """
    try:
        await query.edit_message_text(
            text=text, reply_markup=reply_markup, parse_mode="Markdown"
        )
    except BadRequest as e:
        if "Message is not modified" in str(e):
            if reply_markup:
                try:
                    await query.edit_message_reply_markup(reply_markup=reply_markup)
                except Exception:
                    pass
        elif "can't parse entities" in str(e).lower():
            try:
                await query.edit_message_text(text=text, reply_markup=reply_markup)
            except Exception:
                # Last resort - acknowledge the interaction but don't change anything
                await query.answer("Couldn't update message")
        else:
            # For any other errors, log but continue
            logger.error(f"Error editing message: {str(e)}")
            try:
                await query.edit_message_text(text=text, reply_markup=reply_markup)
            except Exception as inner_e:
                logger.error(f"Second attempt to edit message failed: {str(inner_e)}")
                await query.answer("Couldn't update message")
    except Exception as e:
        logger.error(f"Unexpected error editing message: {str(e)}")
        await query.answer("Couldn't update message")


async def handle_export(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    format_type: str,
    currency_pair: str = None,
):
    logger.info(f"handle_export called with format_type={format_type}, currency_pair={currency_pair}")
    """
    Handle exporting signal history to a file and send it to the user.

    Args:
        update: Telegram update object
        context: Telegram context object
        user_id: User ID for which to export data
        format_type: Export format ('csv' or 'json')
        currency_pair: Optional currency pair to filter by
    """
    try:
        query = update.callback_query
        message_id = query.message.message_id
        chat_id = query.message.chat_id

        await query.answer(f"Preparing {format_type.upper()} export...")

        if currency_pair:
            signals = get_user_signal_history(
                user_id, limit=100, currency_pair=currency_pair
            )
            description = f"for {currency_pair}"
        else:
            signals = get_user_signal_history(user_id, limit=100)
            description = "for all pairs"

        if not signals:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"No signal data available {description} to export.",
                reply_to_message_id=message_id,
            )
            return

        filepath = export_user_signals(
            signals, user_id, format_type, EXPORT_DIR, currency_pair
        )

        with open(filepath, "rb") as file:
            await context.bot.send_document(
                chat_id=chat_id,
                document=file,
                filename=os.path.basename(filepath),
                caption=f"📊 Signal history {description} exported as {format_type.upper()}",
                reply_to_message_id=message_id,
            )

        try:
            os.remove(filepath)
        except Exception as e:
            logger.error(f"Error removing temporary export file {filepath}: {str(e)}")

    except Exception as e:
        error_message = f"Failed to export signal history as {format_type.upper()}"
        await handle_error(update, "export_error", error_message, exception=e)
