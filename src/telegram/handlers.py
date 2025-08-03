"""
Telegram message handlers module for CryptoBot.

This module contains handlers for processing Telegram bot messages,
including conversation handlers, callback queries, and indicator selection.
"""

from telegram import InlineKeyboardMarkup, InlineKeyboardButton, Update  # type: ignore
from telegram.ext import (  # type: ignore
    ContextTypes,
    ConversationHandler,
    CallbackQueryHandler,
)

from src.core.utils import auto_signal_jobs, logger, plural_helper
from src.database.operations import (
    get_all_user_signal_requests,
    delete_user_signal_request,
    get_user_preferences,
    update_user_preferences,
)
from src.telegram.signals.detection import createSignalJob

# Imports with updated module paths
from src.analysis.utils.helpers import input_sanity_check_analyzing, check_signal_limit
from src.core.preferences import get_formatted_preferences

###############################################################################
# States for Conversation
###############################################################################
CHOOSING_ACTION, TYPING_SIGNAL_DATA = range(2)


async def manage_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /manage_signals - The entry point to display signals for the user with inline buttons.
    """
    user_id = update.effective_user.id

    await update.message.reply_text(
        text="Manage your signals:", reply_markup=build_signal_list_keyboard(user_id)
    )

    return CHOOSING_ACTION


def build_signal_list_keyboard(user_id: int) -> InlineKeyboardMarkup:
    """
    Builds a dynamic inline keyboard listing all signals for a user:
      - Each row: [<Pair> (<freq>m)] [Delete <Pair>]
      - 'Add New Signal' button
      - 'Done' button

    Args:
        user_id: Telegram user ID

    Returns:
        InlineKeyboardMarkup with signal management options
    """
    signals = get_all_user_signal_requests(user_id)
    keyboard = []

    if signals:
        for s in signals:
            pair = s["currency_pair"]
            freq = s["frequency_minutes"]
            is_with_chart = s["is_with_chart"]
            display_text = (
                f"{pair} {freq}m chart {'✅' if str(is_with_chart) == '1' else '❌'}"
            )
            # We use 'delete_signal_<pair>' as callback data for the delete button
            del_button = InlineKeyboardButton(
                text=f"Delete {pair}", callback_data=f"delete_signal_{pair}"
            )
            # 'no_op' is a do-nothing callback if user clicks on the display_text
            keyboard.append(
                [InlineKeyboardButton(display_text, callback_data="no_op"), del_button]
            )
    else:
        # If no signals, show a placeholder row
        keyboard.append(
            [InlineKeyboardButton("No signals found", callback_data="no_op")]
        )

    # Add row for [Add New Signal]
    keyboard.append(
        [InlineKeyboardButton("➕ Add New Signal", callback_data="add_signal")]
    )

    # Add row for [Done]
    keyboard.append([InlineKeyboardButton("Done", callback_data="signal_menu_done")])

    return InlineKeyboardMarkup(keyboard)


async def handle_signal_menu_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    """
    CallbackQueryHandler for the signal management UI. This is where we CHECK callback data.

    Args:
        update: Telegram update object
        context: Telegram context object

    Returns:
        Conversation state
    """
    query = update.callback_query
    data = query.data  # The callback_data from the button
    user_id = query.from_user.id

    logger.info(f"handle_signal_menu_callback -> callback_data = {data}")
    await query.answer()  # Acknowledge callback, so Telegram stops the spinning loader

    if data == "no_op":
        # Do nothing
        logger.debug("No operation button clicked.")
        return CHOOSING_ACTION

    elif data.startswith("delete_signal_"):
        # Example data: "delete_signal_BTCUSDT"
        pair = data.replace("delete_signal_", "")

        # 1) Delete from DB
        delete_user_signal_request(user_id, pair)

        # 2) Also stop any running job
        job_key = (user_id, pair)
        if job_key in auto_signal_jobs:
            auto_signal_jobs[job_key].schedule_removal()
            del auto_signal_jobs[job_key]

        await query.edit_message_text(
            text="Signal deleted. Updated list:",
            reply_markup=build_signal_list_keyboard(user_id),
        )
        return CHOOSING_ACTION

    elif data == "add_signal":
        await query.edit_message_text(
            "Please enter your desired signal in the format: \n\n"
            "`SYMBOL INTERVAL [with_chart]`\n\n"
            "Example: `BTCUSDT 60` or `ETHUSDT 15 with_chart`\n\n"
            "SYMBOL should be a valid trading pair (e.g., BTCUSDT).\n"
            "INTERVAL should be in minutes (e.g., 5, 15, 60)."
        )
        return TYPING_SIGNAL_DATA

    elif data == "signal_menu_done":
        n_signals = len(get_all_user_signal_requests(user_id))
        signal_text = f"signal{plural_helper(n_signals)}"
        await query.edit_message_text(f"You have {n_signals} active {signal_text}.")
        return ConversationHandler.END

    # Catch-all
    return CHOOSING_ACTION


async def handle_signal_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    After user taps "Add New Signal", they must type "SYMBOL MINUTES". We parse it and create the job.

    Args:
        update: Telegram update object
        context: Telegram context object

    Returns:
        Conversation state
    """
    user_id = update.effective_user.id
    text = update.message.text.strip().upper()

    # Pass user input through the sanity check
    parse_result = input_sanity_check_analyzing(text)
    if not parse_result["is_valid"]:
        await update.message.reply_text(
            f"Invalid input: {parse_result['error_message']}. Try again or /cancel."
        )
        return TYPING_SIGNAL_DATA

    # Check if user has reached their signal limit
    limit_ok, limit_msg = check_signal_limit(user_id)
    if not limit_ok:
        await update.message.reply_text(limit_msg)
        # Exit to the list of signals
        await update.message.reply_text(
            text="Current signals:", reply_markup=build_signal_list_keyboard(user_id)
        )
        return CHOOSING_ACTION

    # Create the job and return to signal management
    await createSignalJob(update, context)
    await update.message.reply_text(
        text="Signal created! Current signals:",
        reply_markup=build_signal_list_keyboard(user_id),
    )
    return CHOOSING_ACTION


# Dictionary to store temporary preferences for each menu session
# Format: {menu_id: {preference_key: value}}
_menu_preferences = {}


def get_indicator_selection_keyboard(user_id, menu_id=None):
    """
    Create an inline keyboard for selecting indicators with a checkmark for selected ones.

    Args:
        user_id: Telegram user ID
        menu_id: Optional unique identifier for this preference menu instance
        (was created to provide multiple menus being opened at the same time)

    Returns:
        InlineKeyboardMarkup with indicator selection options and the menu_id
    """
    current_preferences = get_user_preferences(user_id)

    if menu_id is None:
        import uuid

        menu_id = str(uuid.uuid4())[:8]
        _menu_preferences[menu_id] = current_preferences.copy()

    selected = _menu_preferences.get(menu_id, current_preferences)

    def create_callback(action):
        return f"{action}:{menu_id}"

    keyboard = [
        [
            InlineKeyboardButton(
                f"{'✔️ ' if selected['order_blocks'] else ''}Order Blocks",
                callback_data=create_callback("indicator_order_blocks"),
            ),
            InlineKeyboardButton(
                f"{'✔️ ' if selected['fvgs'] else ''}FVGs",
                callback_data=create_callback("indicator_fvgs"),
            ),
        ],
        [
            InlineKeyboardButton(
                f"{'✔️ ' if selected['liquidity_levels'] else ''}Liquidity Levels",
                callback_data=create_callback("indicator_liquidity_levels"),
            ),
            InlineKeyboardButton(
                f"{'✔️ ' if selected['breaker_blocks'] else ''}Breaker Blocks",
                callback_data=create_callback("indicator_breaker_blocks"),
            ),
        ],
        [
            InlineKeyboardButton(
                f"{'✔️ ' if selected['show_legend'] else ''}Show Legend",
                callback_data=create_callback("indicator_show_legend"),
            ),
            InlineKeyboardButton(
                f"{'✔️ ' if selected['show_volume'] else ''}Show Volume",
                callback_data=create_callback("indicator_show_volume"),
            ),
        ],
        [
            InlineKeyboardButton(
                f"{'✔️ ' if selected['liquidity_pools'] else ''}Liquidity Pools",
                callback_data=create_callback("indicator_liquidity_pools"),
            ),
        ],
        [
            InlineKeyboardButton(
                f"{'🌙 ' if selected['dark_mode'] else '☀️ '}{'Dark Mode' if selected['dark_mode'] else 'Light Mode'}",
                callback_data=create_callback("indicator_dark_mode"),
            ),
        ],
        [InlineKeyboardButton("Done", callback_data=create_callback("indicator_done"))],
    ]
    return InlineKeyboardMarkup(keyboard), menu_id


async def handle_indicator_selection(update, _):
    """
    Handle the user's selection of indicators and update the inline keyboard dynamically.

    Args:
        update: Telegram update object
        _: Context object (unused)

    Returns:
        Conversation state
    """
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id

    try:
        data_parts = query.data.split(":")
        action = data_parts[0]
        menu_id = data_parts[1] if len(data_parts) > 1 else str(id(query))
        logger.info(f"Processing action: {action}, menu_id: {menu_id}")
    except Exception as e:
        logger.error(f"Invalid callback data format: {query.data}, error: {str(e)}")
        return

    # Make sure this menu exists in our session store
    if menu_id not in _menu_preferences:
        _menu_preferences[menu_id] = get_user_preferences(user_id).copy()

    preferences = _menu_preferences[menu_id].copy()

    if action == "indicator_done":
        # When Done is clicked, commit these preferences to the database
        # This ensures that the most recently closed window always wins
        final_preferences = preferences.copy()

        logger.info(
            f"Indicator done - preferences before formatting: {final_preferences}"
        )

        update_user_preferences(user_id, final_preferences)

        selected_pretty = get_formatted_preferences(final_preferences)
        logger.info(f"Formatted preferences: {selected_pretty}")

        # Clean up by removing this menu from our session store
        if menu_id in _menu_preferences:
            del _menu_preferences[menu_id]

        await query.edit_message_text(
            f"You selected: {', '.join(selected_pretty) or 'None'}"
        )
        return

    # Otherwise, toggle the appropriate preference
    elif action == "indicator_order_blocks":
        preferences["order_blocks"] = not preferences["order_blocks"]
    elif action == "indicator_fvgs":
        preferences["fvgs"] = not preferences["fvgs"]
    elif action == "indicator_liquidity_levels":
        preferences["liquidity_levels"] = not preferences["liquidity_levels"]
    elif action == "indicator_breaker_blocks":
        preferences["breaker_blocks"] = not preferences["breaker_blocks"]
    elif action == "indicator_liquidity_pools":
        preferences["liquidity_pools"] = not preferences["liquidity_pools"]
    elif action == "indicator_show_legend":
        preferences["show_legend"] = not preferences["show_legend"]
    elif action == "indicator_show_volume":
        preferences["show_volume"] = not preferences["show_volume"]
    elif action == "indicator_dark_mode":
        preferences["dark_mode"] = not preferences["dark_mode"]

    # Save the updated preferences back to the session store
    _menu_preferences[menu_id] = preferences

    new_markup, _ = get_indicator_selection_keyboard(user_id, menu_id)

    await query.edit_message_reply_markup(reply_markup=new_markup)

    return CHOOSING_ACTION


async def select_indicators(update, _):
    """
    Start the process of selecting indicators.

    Args:
        update: Telegram update object
        _: Context object (unused)

    Returns:
        None
    """
    user_id = update.effective_user.id

    # Generate a new keyboard with a unique menu ID
    # This ensures each preferences window has its own identifier
    keyboard, menu_id = get_indicator_selection_keyboard(user_id)

    await update.message.reply_text(
        "Please choose the indicators you'd like to include:",
        reply_markup=keyboard,
    )
