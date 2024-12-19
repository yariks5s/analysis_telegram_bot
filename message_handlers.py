from telegram import InlineKeyboardMarkup, InlineKeyboardButton

from utils import user_selected_indicators

def get_indicator_selection_keyboard(user_id):
    """
    Create an inline keyboard for selecting indicators with a checkmark for selected ones.
    """
    selected = user_selected_indicators.get(user_id, {
        "order_blocks": False,
        "fvgs": False,
        "liquidity_levels": False,
        "breaker_blocks": False,
    })

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
    if user_id not in user_selected_indicators:
        user_selected_indicators[user_id] = {
            "order_blocks": False,
            "fvgs": False,
            "liquidity_levels": False,
            "breaker_blocks": False,
        }

    # Update preferences based on selection
    data = query.data
    if data == "indicator_order_blocks":
        user_selected_indicators[user_id]["order_blocks"] = not user_selected_indicators[user_id]["order_blocks"]
    elif data == "indicator_fvgs":
        user_selected_indicators[user_id]["fvgs"] = not user_selected_indicators[user_id]["fvgs"]
    elif data == "indicator_liquidity_levels":
        user_selected_indicators[user_id]["liquidity_levels"] = not user_selected_indicators[user_id]["liquidity_levels"]
    elif data == "indicator_breaker_blocks":
        user_selected_indicators[user_id]["breaker_blocks"] = not user_selected_indicators[user_id]["breaker_blocks"]
    elif data == "indicator_done":
        selected = user_selected_indicators[user_id]
        await query.edit_message_text(
            f"You selected: {', '.join([key for key, val in selected.items() if val]) or 'None'}"
        )
        return

    # Update message with current selections and checkmarks
    await query.edit_message_reply_markup(
        reply_markup=get_indicator_selection_keyboard(user_id)
    )

async def select_indicators(update, _):
    """
    Start the process of selecting indicators.
    """
    user_id = update.effective_user.id
    user_selected_indicators[user_id] = {
        "order_blocks": False,
        "fvgs": False,
        "liquidity_levels": False,
        "breaker_blocks": False,
    }
    await update.message.reply_text(
        "Please choose the indicators you'd like to include:",
        reply_markup=get_indicator_selection_keyboard(user_id)
    )
