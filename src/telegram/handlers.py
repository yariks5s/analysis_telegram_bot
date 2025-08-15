"""
Telegram message handlers module for CryptoBot.

This module contains handlers for processing Telegram bot messages,
including conversation handlers, callback queries, and indicator selection.
"""

import time

from telegram import InlineKeyboardMarkup, InlineKeyboardButton, Update  # type: ignore
from telegram.ext import (  # type: ignore
    ContextTypes,
    ConversationHandler,
)

from src.core.utils import auto_signal_jobs, logger, plural_helper
from src.database.operations import (
    get_all_user_signal_requests,
    delete_user_signal_request,
    get_user_preferences,
    update_user_preferences,
)
from src.core.preferences import INDICATOR_PARAMS
from src.telegram.signals.detection import createSignalJob

from src.analysis.utils.helpers import input_sanity_check_text, check_signal_limit_by_id
from src.core.preferences import get_formatted_preferences

###############################################################################
# States for Conversation
###############################################################################
CHOOSING_ACTION, TYPING_SIGNAL_DATA, TYPING_PARAM_VALUE = range(3)

# Dictionary to store temporary parameter editing states
# Format: {user_id: {"param": param_name, "menu_id": menu_id}}
_param_edit_states = {}

# Dictionary to store temporary preferences for each menu session
# Format: {menu_id: {preference_key: value}}
_menu_preferences = {}

# Dictionary to store temporary parameter values for each menu session
# Format: {menu_id: {param_name: value}}
_param_preferences = {}


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
            del_button = InlineKeyboardButton(
                text=f"Delete {pair}", callback_data=f"delete_signal_{pair}"
            )
            keyboard.append(
                [InlineKeyboardButton(display_text, callback_data="no_op"), del_button]
            )
    else:
        keyboard.append(
            [InlineKeyboardButton("No signals found", callback_data="no_op")]
        )

    keyboard.append(
        [InlineKeyboardButton("➕ Add New Signal", callback_data="add_signal")]
    )

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
    data = query.data
    user_id = query.from_user.id

    logger.info(f"handle_signal_menu_callback -> callback_data = {data}")
    await query.answer()

    if data == "no_op":
        logger.debug("No operation button clicked.")
        return CHOOSING_ACTION

    elif data.startswith("delete_signal_"):
        pair = data.replace("delete_signal_", "")

        delete_user_signal_request(user_id, pair)

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

    parse_result = input_sanity_check_text(text)
    if not parse_result["is_valid"]:
        await update.message.reply_text(
            f"Invalid input: {parse_result['error_message']}. Try again or /cancel."
        )
        return TYPING_SIGNAL_DATA

    limit_ok, limit_msg = check_signal_limit_by_id(user_id)
    if not limit_ok:
        await update.message.reply_text(limit_msg)
        await update.message.reply_text(
            text="Current signals:", reply_markup=build_signal_list_keyboard(user_id)
        )
        return CHOOSING_ACTION

    symbol = parse_result["symbol"]
    period_minutes = parse_result["period_minutes"]
    is_with_chart = parse_result["is_with_chart"]

    await createSignalJob(symbol, period_minutes, is_with_chart, update, context)
    await update.message.reply_text(
        text="Signal created! Current signals:",
        reply_markup=build_signal_list_keyboard(user_id),
    )
    return CHOOSING_ACTION


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
        [
            InlineKeyboardButton(
                "Done", callback_data=create_callback("indicator_done")
            ),
        ],
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

        if len(data_parts) >= 3 and data_parts[0] == "indicator":
            menu_id = data_parts[1]
            action = data_parts[2]
            logger.info(f"Processing parameter action: {action}, menu_id: {menu_id}")
        else:
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
        final_preferences = preferences.copy()

        logger.info(
            f"Indicator done - preferences before formatting: {final_preferences}"
        )

        update_user_preferences(user_id, final_preferences)

        selected_pretty = get_formatted_preferences(final_preferences)
        logger.info(f"Formatted preferences: {selected_pretty}")

        if menu_id in _menu_preferences:
            del _menu_preferences[menu_id]

        await query.edit_message_text(
            f"You selected: {', '.join(selected_pretty) or 'None'}"
        )
        return

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

    _menu_preferences[menu_id] = preferences

    new_markup, _ = get_indicator_selection_keyboard(user_id, menu_id)

    await query.edit_message_reply_markup(reply_markup=new_markup)

    return CHOOSING_ACTION


def get_parameter_keyboard(user_id, menu_id=None):
    """
    Create an inline keyboard for adjusting indicator parameters.

    Args:
        user_id: Telegram user ID
        menu_id: Optional unique identifier for this parameter menu instance

    Returns:
        InlineKeyboardMarkup with parameter setting options and the menu_id
    """
    if menu_id is None:
        menu_id = str(id(user_id)) + str(time.time())

    if menu_id not in _param_preferences:
        _param_preferences[menu_id] = get_user_preferences(user_id).copy()

    preferences = _param_preferences[menu_id]

    keyboard = []

    for param_name, param_info in INDICATOR_PARAMS.items():
        current_value = preferences.get(param_name, param_info["default"])
        keyboard.append(
            [
                InlineKeyboardButton(
                    f"{param_info['display_name']}: {current_value}",
                    callback_data=f"param:{menu_id}:edit_{param_name}",
                )
            ]
        )

    keyboard.append(
        [InlineKeyboardButton("Save", callback_data=f"param:{menu_id}:save")]
    )

    return InlineKeyboardMarkup(keyboard), menu_id


async def handle_parameter_input(update, context):
    """
    Handle user input for parameter values.

    Args:
        update: Telegram update object
        context: Telegram context object

    Returns:
        Conversation state
    """
    user_id = update.effective_user.id
    user_text = update.message.text.strip()

    if user_id not in _param_edit_states:
        await update.message.reply_text(
            "Sorry, I don't have an active parameter editing session. "
            "Please use /select_indicators to start over."
        )
        return ConversationHandler.END

    param_info = _param_edit_states[user_id]
    param_name = param_info["param"]
    menu_id = param_info["menu_id"]

    try:
        if param_name == "atr_period":
            new_value = int(user_text)
        else:  # fvg_min_size or other float params
            new_value = float(user_text)

        param_config = INDICATOR_PARAMS[param_name]
        if new_value < param_config["min"] or new_value > param_config["max"]:
            await update.message.reply_text(
                f"Value must be between {param_config['min']} and {param_config['max']}. "
                "Please try again."
            )
            return TYPING_PARAM_VALUE

        _menu_preferences[menu_id][param_name] = new_value

        # Clear the edit state
        del _param_edit_states[user_id]

        param_markup = get_parameter_keyboard(user_id, menu_id)
        await update.message.reply_text(
            f"{INDICATOR_PARAMS[param_name]['display_name']} set to {new_value}.",
            reply_markup=param_markup,
        )

        return CHOOSING_ACTION

    except (ValueError, TypeError):
        await update.message.reply_text(
            f"Invalid input. Please enter a valid number for {INDICATOR_PARAMS[param_name]['display_name']}."
        )
        return TYPING_PARAM_VALUE


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

    keyboard, _ = get_indicator_selection_keyboard(user_id)

    await update.message.reply_text(
        "Please choose the indicators you'd like to include:",
        reply_markup=keyboard,
    )


async def handle_parameter_selection(update, _):
    """
    Handle the user's selection of parameter values.

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
        if len(data_parts) >= 3 and data_parts[0] == "param":
            menu_id = data_parts[1]
            action = data_parts[2]
            logger.info(f"Processing parameter action: {action}, menu_id: {menu_id}")
        else:
            logger.error(f"Invalid callback data format: {query.data}")
            return
    except Exception as e:
        logger.error(f"Invalid callback data format: {query.data}, error: {str(e)}")
        return

    if menu_id not in _param_preferences:
        _param_preferences[menu_id] = get_user_preferences(user_id).copy()

    preferences = _param_preferences[menu_id].copy()

    if action == "save":
        update_user_preferences(user_id, preferences)

        if menu_id in _param_preferences:
            del _param_preferences[menu_id]

        await query.edit_message_text("Parameter settings saved successfully!")
        return

    elif action.startswith("edit_"):
        param_name = action.replace("edit_", "")
        _param_edit_states[user_id] = {"param": param_name, "menu_id": menu_id}

        param_info = INDICATOR_PARAMS[param_name]
        current_value = preferences.get(param_name, param_info["default"])

        await query.edit_message_text(
            f"Enter a new value for {param_info['display_name']}:\n\n"
            f"Current value: {current_value}\n"
            f"Valid range: {param_info['min']} to {param_info['max']} (step: {param_info['step']})\n\n"
            f"Reply with a number to set the new value."
        )

        return TYPING_PARAM_VALUE

    # If we got here, something unexpected happened
    await query.edit_message_text(
        "Sorry, I couldn't process that action. Please try again with /parameters."
    )
    return


async def handle_parameter_input(update, context):
    """
    Handle user input for parameter values.

    Args:
        update: Telegram update object
        context: Telegram context object

    Returns:
        Conversation state
    """
    user_id = update.effective_user.id
    user_text = update.message.text.strip()

    if user_id not in _param_edit_states:
        await update.message.reply_text(
            "Sorry, I don't have an active parameter editing session. "
            "Please use /parameters to start over."
        )
        return ConversationHandler.END

    param_info = _param_edit_states[user_id]
    param_name = param_info["param"]
    menu_id = param_info["menu_id"]

    try:
        if param_name == "atr_period":
            new_value = int(user_text)
        else:  # fvg_min_size or other float params
            new_value = float(user_text)

        param_config = INDICATOR_PARAMS[param_name]
        if new_value < param_config["min"] or new_value > param_config["max"]:
            await update.message.reply_text(
                f"Value must be between {param_config['min']} and {param_config['max']}. "
                "Please try again."
            )
            return TYPING_PARAM_VALUE

        _param_preferences[menu_id][param_name] = new_value

        del _param_edit_states[user_id]

        param_markup, _ = get_parameter_keyboard(user_id, menu_id)
        await update.message.reply_text(
            f"{INDICATOR_PARAMS[param_name]['display_name']} set to {new_value}.",
            reply_markup=param_markup,
        )

        return CHOOSING_ACTION

    except (ValueError, TypeError):
        await update.message.reply_text(
            f"Invalid input. Please enter a valid number for {INDICATOR_PARAMS[param_name]['display_name']}."
        )
        return TYPING_PARAM_VALUE


async def show_parameters(update, _):
    """
    Start the process of setting indicator parameters.

    Args:
        update: Telegram update object
        _: Context object (unused)

    Returns:
        Conversation state
    """
    user_id = update.effective_user.id

    keyboard, _ = get_parameter_keyboard(user_id)

    await update.message.reply_text(
        "Configure indicator parameters:",
        reply_markup=keyboard,
    )

    return CHOOSING_ACTION
