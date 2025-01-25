from helpers import input_sanity_check_analyzing
from telegram import InlineKeyboardMarkup, InlineKeyboardButton, Update, ReplyKeyboardRemove # type: ignore
from telegram.ext import ( # type: ignore
    ContextTypes,
    ConversationHandler, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters
)

from database import (
    get_all_user_signal_requests,
    delete_user_signal_request,
    get_user_preferences,
    update_user_preferences,
)
from signal_detection import createSignalJob
from utils import auto_signal_jobs, logger
from utils import plural_helper

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
        text="Manage your signals:",
        reply_markup=build_signal_list_keyboard(user_id)
    )

    return CHOOSING_ACTION


def build_signal_list_keyboard(user_id: int) -> InlineKeyboardMarkup:
    """
    Builds a dynamic inline keyboard listing all signals for a user:
      - Each row: [<Pair> (<freq>m)] [Delete <Pair>]
      - 'Add New Signal' button
      - 'Done' button
    """
    signals = get_all_user_signal_requests(user_id)
    keyboard = []

    if signals:
        for s in signals:
            pair = s["currency_pair"]
            freq = s["frequency_minutes"]
            display_text = f"{pair} ({freq}m)"
            # We use 'delete_signal_<pair>' as callback data for the delete button
            del_button = InlineKeyboardButton(
                text=f"Delete {pair}",
                callback_data=f"delete_signal_{pair}"
            )
            # 'no_op' is a do-nothing callback if user clicks on the display_text
            keyboard.append([
                InlineKeyboardButton(display_text, callback_data="no_op"),
                del_button
            ])
    else:
        # If no signals, show a placeholder row
        keyboard.append([
            InlineKeyboardButton("No signals found", callback_data="no_op")
        ])

    # Add row for [Add New Signal]
    keyboard.append([
        InlineKeyboardButton("➕ Add New Signal", callback_data="add_signal")
    ])

    # Add row for [Done]
    keyboard.append([
        InlineKeyboardButton("Done", callback_data="signal_menu_done")
    ])

    return InlineKeyboardMarkup(keyboard)


async def handle_signal_menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    CallbackQueryHandler for the signal management UI. This is where we CHECK callback data.
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
            reply_markup=build_signal_list_keyboard(user_id)
        )
        return CHOOSING_ACTION

    elif data == "add_signal":
        # Switch to TYPING_SIGNAL_DATA state: we expect user to type "SYMBOL MINUTES"
        await query.edit_message_text(
            "Enter new signal as: SYMBOL MINUTES (e.g. BTCUSDT 60)"
        )
        return TYPING_SIGNAL_DATA

    elif data == "signal_menu_done":
        # End the conversation
        await query.edit_message_text("Signal management window closed.")
        return ConversationHandler.END

    # Default fallback: remain in CHOOSING_ACTION
    return CHOOSING_ACTION


async def handle_signal_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    After user taps "Add New Signal", they must type "SYMBOL MINUTES". We parse it and create the job.
    """
    user_id = update.effective_user.id
    text = update.message.text.strip()
    parts = text.split()

    pair = await input_sanity_check_analyzing(True, parts, update)
    if (not pair):
        await update.message.reply_text(f"Usage: <symbol> <period_in_minutes> [<is_with_photo>], you've sent {len(parts)} argument{plural_helper(len(parts))}.")
        return TYPING_SIGNAL_DATA

    await createSignalJob(pair[0], pair[1], update, context)

    # Finally show the updated list
    await update.message.reply_text(
        text="Updated signals list:",
        reply_markup=build_signal_list_keyboard(user_id)
    )

    return CHOOSING_ACTION

def get_indicator_selection_keyboard(user_id):
    """
    Create an inline keyboard for selecting indicators with a checkmark for selected ones.
    """
    selected = get_user_preferences(user_id)

    # Add a checkmark if the indicator is selected
    keyboard = [
        [
            InlineKeyboardButton(
                f"{'✔️ ' if selected['order_blocks'] else ''}Order Blocks", 
                callback_data="indicator_order_blocks"
            ),
            InlineKeyboardButton(
                f"{'✔️ ' if selected['fvgs'] else ''}FVGs", 
                callback_data="indicator_fvgs"
            ),
        ],
        [
            InlineKeyboardButton(
                f"{'✔️ ' if selected['liquidity_levels'] else ''}Liquidity Levels", 
                callback_data="indicator_liquidity_levels"
            ),
            InlineKeyboardButton(
                f"{'✔️ ' if selected['breaker_blocks'] else ''}Breaker Blocks", 
                callback_data="indicator_breaker_blocks"
            ),
        ],
        [InlineKeyboardButton("Done", callback_data="indicator_done")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def handle_indicator_selection(update, _):
    """
    Handle the user's selection of indicators and update the inline keyboard dynamically.
    """
    query = update.callback_query
    await query.answer()

    # Get user ID and initialize preferences
    user_id = query.from_user.id
    preferences = get_user_preferences(user_id)

    # Update preferences based on user action
    data = query.data
    if data == "indicator_order_blocks":
        preferences["order_blocks"] = not preferences["order_blocks"]
    elif data == "indicator_fvgs":
        preferences["fvgs"] = not preferences["fvgs"]
    elif data == "indicator_liquidity_levels":
        preferences["liquidity_levels"] = not preferences["liquidity_levels"]
    elif data == "indicator_breaker_blocks":
        preferences["breaker_blocks"] = not preferences["breaker_blocks"]
    elif data == "indicator_done":
        selected = [key for key, val in preferences.items() if val]
        await query.edit_message_text(f"You selected: {', '.join(selected) or 'None'}")
        return

    # Save updated preferences in the database
    update_user_preferences(user_id, preferences)

    # Update message with current selections and checkmarks
    await query.edit_message_reply_markup(
        reply_markup=get_indicator_selection_keyboard(user_id)
    )

async def select_indicators(update, _):
    """
    Start the process of selecting indicators.
    """
    user_id = update.effective_user.id
    await update.message.reply_text(
        "Please choose the indicators you'd like to include:",
        reply_markup=get_indicator_selection_keyboard(user_id)
    )
