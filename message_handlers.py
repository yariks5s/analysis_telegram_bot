from telegram import InlineKeyboardMarkup, InlineKeyboardButton

from database import get_user_preferences, update_user_preferences

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
