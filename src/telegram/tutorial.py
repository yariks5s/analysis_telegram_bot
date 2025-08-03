"""
Tutorial system for CryptoBot.

This module provides onboarding tutorials for new users to help them learn
how to use the bot effectively.
"""

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ContextTypes, ConversationHandler

from src.database.operations import get_user_preferences, update_user_preferences

TUTORIAL_NOT_STARTED = 0
TUTORIAL_WELCOME = 1
TUTORIAL_CHART_COMMAND = 2
TUTORIAL_PREFERENCES = 3
TUTORIAL_SIGNALS = 4
TUTORIAL_PARAMETERS = 5
TUTORIAL_COMPLETED = 6

CHOOSING_TUTORIAL_ACTION = 0

TUTORIAL_MESSAGES = {
    TUTORIAL_WELCOME: {
        "title": "🎉 Welcome to CryptoBot!",
        "content": (
            "I'm your personal crypto market analysis assistant. I can help you analyze "
            "cryptocurrency charts with advanced technical indicators.\n\n"
            "This short tutorial will show you how to use my main features. You can skip "
            "the tutorial or come back to it anytime with the /tutorial command."
        ),
        "buttons": [
            [InlineKeyboardButton("Continue", callback_data="tutorial_next")],
            [InlineKeyboardButton("Skip Tutorial", callback_data="tutorial_skip")]
        ]
    },
    TUTORIAL_CHART_COMMAND: {
        "title": "📊 Chart Analysis",
        "content": (
            "The basic command to get a chart analysis is:\n\n"
            "<code>/chart SYMBOL LENGTH INTERVAL TOLERANCE</code>\n\n"
            "For example: <code>/chart BTCUSDT 48 1h 0.05</code>\n\n"
            "This gives you a 48-candle, 1-hour chart for BTC/USDT with a tolerance of 0.05 "
            "for detecting liquidity levels.\n\n"
            "Try it out or continue to learn more features!"
        ),
        "buttons": [
            [InlineKeyboardButton("Continue", callback_data="tutorial_next")],
            [InlineKeyboardButton("Skip to End", callback_data="tutorial_skip")]
        ]
    },
    TUTORIAL_PREFERENCES: {
        "title": "⚙️ Customizing Indicators",
        "content": (
            "You can customize which indicators appear on your charts with the "
            "<code>/preferences</code> command.\n\n"
            "This opens an interactive menu where you can enable or disable:\n"
            "• Order Blocks\n"
            "• Fair Value Gaps (FVGs)\n"
            "• Liquidity Levels\n"
            "• Breaker Blocks\n"
            "• And change display options\n\n"
            "Your preferences will be saved for future chart requests."
        ),
        "buttons": [
            [InlineKeyboardButton("Continue", callback_data="tutorial_next")],
            [InlineKeyboardButton("Skip to End", callback_data="tutorial_skip")]
        ]
    },
    TUTORIAL_SIGNALS: {
        "title": "🔔 Automated Signals",
        "content": (
            "Want to receive automatic notifications? Set up signal alerts with:\n\n"
            "<code>/create_signal SYMBOL MINUTES [CHART]</code>\n\n"
            "For example: <code>/create_signal BTCUSDT 60 true</code>\n\n"
            "This will send you a notification with a chart every 60 minutes for BTC/USDT.\n\n"
            "You can manage your signals with the <code>/manage_signals</code> command."
        ),
        "buttons": [
            [InlineKeyboardButton("Continue", callback_data="tutorial_next")],
            [InlineKeyboardButton("Skip to End", callback_data="tutorial_skip")]
        ]
    },
    TUTORIAL_PARAMETERS: {
        "title": "🔧 Advanced Parameters",
        "content": (
            "Fine-tune indicator parameters with the <code>/parameters</code> command.\n\n"
            "You can adjust:\n"
            "• ATR Period: for calculating Average True Range\n"
            "• FVG Min Size: minimum size for Fair Value Gaps\n\n"
            "These settings affect how indicators are calculated and displayed."
        ),
        "buttons": [
            [InlineKeyboardButton("Complete Tutorial", callback_data="tutorial_next")]
        ]
    },
    TUTORIAL_COMPLETED: {
        "title": "🚀 Tutorial Completed!",
        "content": (
            "Great job! You've completed the CryptoBot tutorial.\n\n"
            "Remember, you can always get help with:\n"
            "• <code>/help</code> - General help information\n"
            "• <code>/help COMMAND</code> - Help for a specific command\n\n"
            "Ready to start? Try <code>/chart BTCUSDT 48 1h 0.05</code> to see BTC/USDT analysis!"
        ),
        "buttons": [
            [InlineKeyboardButton("View Available Commands", callback_data="tutorial_commands")],
            [InlineKeyboardButton("Finish", callback_data="tutorial_finish")]
        ]
    }
}


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Start the bot and begin the onboarding tutorial.
    This is the entry point for new users.
    """
    user_id = update.effective_user.id
    user_prefs = get_user_preferences(user_id)
    
    if user_prefs["tutorial_stage"] < TUTORIAL_COMPLETED:
        user_prefs["tutorial_stage"] = TUTORIAL_WELCOME
        update_user_preferences(user_id, user_prefs)
        
        tutorial = TUTORIAL_MESSAGES[TUTORIAL_WELCOME]
        await update.message.reply_html(
            f"<b>{tutorial['title']}</b>\n\n{tutorial['content']}",
            reply_markup=InlineKeyboardMarkup(tutorial["buttons"])
        )
        return CHOOSING_TUTORIAL_ACTION
    else:
        # User has already completed the tutorial
        await update.message.reply_html(
            "<b>Welcome back to CryptoBot!</b>\n\n"
            "You've already completed the tutorial. What would you like to do?\n\n"
            "• Get a chart with <code>/chart BTCUSDT 48 1h 0.05</code>\n"
            "• Set preferences with <code>/preferences</code>\n"
            "• Create signals with <code>/create_signal</code>\n"
            "• Review the tutorial with <code>/tutorial</code>"
        )
        return ConversationHandler.END


async def tutorial_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Restart or continue the tutorial at any point.
    """
    user_id = update.effective_user.id
    user_prefs = get_user_preferences(user_id)
    
    has_completed_before = user_prefs["tutorial_stage"] >= TUTORIAL_COMPLETED
    
    current_view_stage = TUTORIAL_WELCOME
    
    original_stage = user_prefs["tutorial_stage"]
    
    if user_prefs["tutorial_stage"] == TUTORIAL_NOT_STARTED:
        user_prefs["tutorial_stage"] = TUTORIAL_WELCOME
        update_user_preferences(user_id, user_prefs)
    elif not has_completed_before and user_prefs["tutorial_stage"] > TUTORIAL_NOT_STARTED:
        # If in progress but not completed, just continue from current stage
        current_view_stage = user_prefs["tutorial_stage"]
        
    # For users who already completed, we'll show the tutorial from the beginning
    # but won't update their stage in the database unless they complete it again
    tutorial = TUTORIAL_MESSAGES[current_view_stage]
    
    context.user_data["has_completed_tutorial_before"] = has_completed_before
    context.user_data["original_tutorial_stage"] = original_stage
    
    await update.message.reply_html(
        f"<b>{tutorial['title']}</b>\n\n{tutorial['content']}",
        reply_markup=InlineKeyboardMarkup(tutorial["buttons"])
    )
    return CHOOSING_TUTORIAL_ACTION


async def handle_tutorial_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle callbacks from tutorial inline buttons.
    """
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    user_prefs = get_user_preferences(user_id)
    
    # Check if user completed tutorial before (from context or preferences)
    has_completed_before = context.user_data.get(
        "has_completed_tutorial_before", 
        user_prefs["tutorial_stage"] >= TUTORIAL_COMPLETED
    )
    
    original_stage = context.user_data.get(
        "original_tutorial_stage", 
        user_prefs["tutorial_stage"]
    )
    
    current_stage = user_prefs["tutorial_stage"]
    
    if query.data == "tutorial_next":
        next_stage = current_stage + 1
        if next_stage > TUTORIAL_COMPLETED:
            next_stage = TUTORIAL_COMPLETED
        
        # If user has completed tutorial before, only update DB if they haven't progressed this far
        if not has_completed_before or next_stage > original_stage:
            user_prefs["tutorial_stage"] = next_stage
            update_user_preferences(user_id, user_prefs)
        
        tutorial = TUTORIAL_MESSAGES[next_stage]
        await query.edit_message_text(
            f"<b>{tutorial['title']}</b>\n\n{tutorial['content']}",
            reply_markup=InlineKeyboardMarkup(tutorial["buttons"]),
            parse_mode="HTML"
        )
        
        if next_stage == TUTORIAL_COMPLETED:
            return ConversationHandler.END
            
    elif query.data == "tutorial_skip":
        # If they've completed it before, don't need to update to completed again
        if not has_completed_before:
            user_prefs["tutorial_stage"] = TUTORIAL_COMPLETED
            update_user_preferences(user_id, user_prefs)
        
        tutorial = TUTORIAL_MESSAGES[TUTORIAL_COMPLETED]
        await query.edit_message_text(
            f"<b>{tutorial['title']}</b>\n\n{tutorial['content']}",
            reply_markup=InlineKeyboardMarkup(tutorial["buttons"]),
            parse_mode="HTML"
        )
        return ConversationHandler.END
        
    elif query.data == "tutorial_commands": # Todo: make it reusable
        await query.edit_message_text(
            "<b>Available Commands:</b>\n\n"
            "<code>/chart</code> - Get a chart with indicators\n"
            "<code>/text_result</code> - Get a text analysis\n"
            "<code>/preferences</code> - Set indicator preferences\n"
            "<code>/parameters</code> - Fine-tune indicator settings\n"
            "<code>/create_signal</code> - Set up automatic signals\n"
            "<code>/manage_signals</code> - Manage your signal alerts\n"
            "<code>/help</code> - Get help with commands\n"
            "<code>/tutorial</code> - Revisit this tutorial",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("Back to Tutorial", callback_data="tutorial_back_to_end")
            ]])
        )
    
    elif query.data == "tutorial_back_to_end":
        tutorial = TUTORIAL_MESSAGES[TUTORIAL_COMPLETED]
        await query.edit_message_text(
            f"<b>{tutorial['title']}</b>\n\n{tutorial['content']}",
            reply_markup=InlineKeyboardMarkup(tutorial["buttons"]),
            parse_mode="HTML"
        )
    
    elif query.data == "tutorial_finish":
        await query.edit_message_text(
            "<b>Happy trading!</b> 📈\n\n"
            "Use <code>/help</code> anytime you need assistance.",
            parse_mode="HTML"
        )
        return ConversationHandler.END
    
    return CHOOSING_TUTORIAL_ACTION
