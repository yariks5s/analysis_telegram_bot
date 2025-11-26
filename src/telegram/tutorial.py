"""
Tutorial system for CryptoBot.

This module provides onboarding tutorials for new users to help them learn
how to use the bot effectively.
"""

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ContextTypes, ConversationHandler

from src.database.operations import get_user_preferences, update_user_preferences
from src.i18n import t, get_user_language

TUTORIAL_NOT_STARTED = 0
TUTORIAL_WELCOME = 1
TUTORIAL_CHART_COMMAND = 2
TUTORIAL_PREFERENCES = 3
TUTORIAL_SIGNALS = 4
TUTORIAL_PARAMETERS = 5
TUTORIAL_COMPLETED = 6

CHOOSING_TUTORIAL_ACTION = 0


def get_tutorial_message(stage: int, lang: str = "en") -> dict:
    """
    Get the tutorial message for a specific stage in the specified language.
    
    Args:
        stage: Tutorial stage number
        lang: Language code
    
    Returns:
        Dictionary with title, content, and buttons for the tutorial stage
    """
    messages = {
        TUTORIAL_WELCOME: {
            "title": t("tutorial.welcome.title", lang),
            "content": t("tutorial.welcome.content", lang),
            "buttons": [
                [InlineKeyboardButton(t("tutorial.buttons.continue", lang), callback_data="tutorial_next")],
                [InlineKeyboardButton(t("tutorial.buttons.skip_tutorial", lang), callback_data="tutorial_skip")],
            ],
        },
        TUTORIAL_CHART_COMMAND: {
            "title": t("tutorial.chart.title", lang),
            "content": t("tutorial.chart.content", lang),
            "buttons": [
                [InlineKeyboardButton(t("tutorial.buttons.continue", lang), callback_data="tutorial_next")],
                [InlineKeyboardButton(t("tutorial.buttons.skip_to_end", lang), callback_data="tutorial_skip")],
            ],
        },
        TUTORIAL_PREFERENCES: {
            "title": t("tutorial.preferences.title", lang),
            "content": t("tutorial.preferences.content", lang),
            "buttons": [
                [InlineKeyboardButton(t("tutorial.buttons.continue", lang), callback_data="tutorial_next")],
                [InlineKeyboardButton(t("tutorial.buttons.skip_to_end", lang), callback_data="tutorial_skip")],
            ],
        },
        TUTORIAL_SIGNALS: {
            "title": t("tutorial.signals.title", lang),
            "content": t("tutorial.signals.content", lang),
            "buttons": [
                [InlineKeyboardButton(t("tutorial.buttons.continue", lang), callback_data="tutorial_next")],
                [InlineKeyboardButton(t("tutorial.buttons.skip_to_end", lang), callback_data="tutorial_skip")],
            ],
        },
        TUTORIAL_PARAMETERS: {
            "title": t("tutorial.parameters.title", lang),
            "content": t("tutorial.parameters.content", lang),
            "buttons": [
                [InlineKeyboardButton(t("tutorial.buttons.complete", lang), callback_data="tutorial_next")]
            ],
        },
        TUTORIAL_COMPLETED: {
            "title": t("tutorial.completed.title", lang),
            "content": t("tutorial.completed.content", lang),
            "buttons": [
                [
                    InlineKeyboardButton(
                        t("tutorial.buttons.view_commands", lang), callback_data="tutorial_commands"
                    )
                ],
                [InlineKeyboardButton(t("common.finish", lang), callback_data="tutorial_finish")],
            ],
        },
    }
    return messages.get(stage, messages[TUTORIAL_WELCOME])


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Start the bot and begin the onboarding tutorial.
    This is the entry point for new users.
    """
    user_id = update.effective_user.id
    user_prefs = get_user_preferences(user_id)
    lang = get_user_language(user_id)

    if user_prefs["tutorial_stage"] < TUTORIAL_COMPLETED:
        user_prefs["tutorial_stage"] = TUTORIAL_WELCOME
        update_user_preferences(user_id, user_prefs)

        tutorial = get_tutorial_message(TUTORIAL_WELCOME, lang)
        await update.message.reply_html(
            f"<b>{tutorial['title']}</b>\n\n{tutorial['content']}",
            reply_markup=InlineKeyboardMarkup(tutorial["buttons"]),
        )
        return CHOOSING_TUTORIAL_ACTION
    else:
        # User has already completed the tutorial
        await update.message.reply_html(
            f"<b>{t('tutorial.welcome_back.title', lang)}</b>\n\n{t('tutorial.welcome_back.content', lang)}"
        )
        return ConversationHandler.END


async def tutorial_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Restart or continue the tutorial at any point.
    """
    user_id = update.effective_user.id
    user_prefs = get_user_preferences(user_id)
    lang = get_user_language(user_id)

    has_completed_before = user_prefs["tutorial_stage"] >= TUTORIAL_COMPLETED

    current_view_stage = TUTORIAL_WELCOME

    original_stage = user_prefs["tutorial_stage"]

    if user_prefs["tutorial_stage"] == TUTORIAL_NOT_STARTED:
        user_prefs["tutorial_stage"] = TUTORIAL_WELCOME
        update_user_preferences(user_id, user_prefs)
    elif (
        not has_completed_before and user_prefs["tutorial_stage"] > TUTORIAL_NOT_STARTED
    ):
        # If in progress but not completed, just continue from current stage
        current_view_stage = user_prefs["tutorial_stage"]

    # For users who already completed, we'll show the tutorial from the beginning
    # but won't update their stage in the database unless they complete it again
    tutorial = get_tutorial_message(current_view_stage, lang)

    context.user_data["has_completed_tutorial_before"] = has_completed_before
    context.user_data["original_tutorial_stage"] = original_stage

    await update.message.reply_html(
        f"<b>{tutorial['title']}</b>\n\n{tutorial['content']}",
        reply_markup=InlineKeyboardMarkup(tutorial["buttons"]),
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
    lang = get_user_language(user_id)

    # Check if user completed tutorial before (from context or preferences)
    has_completed_before = context.user_data.get(
        "has_completed_tutorial_before",
        user_prefs["tutorial_stage"] >= TUTORIAL_COMPLETED,
    )

    original_stage = context.user_data.get(
        "original_tutorial_stage", user_prefs["tutorial_stage"]
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

        tutorial = get_tutorial_message(next_stage, lang)
        await query.edit_message_text(
            f"<b>{tutorial['title']}</b>\n\n{tutorial['content']}",
            reply_markup=InlineKeyboardMarkup(tutorial["buttons"]),
            parse_mode="HTML",
        )

        if next_stage == TUTORIAL_COMPLETED:
            return ConversationHandler.END

    elif query.data == "tutorial_skip":
        # If they've completed it before, don't need to update to completed again
        if not has_completed_before:
            user_prefs["tutorial_stage"] = TUTORIAL_COMPLETED
            update_user_preferences(user_id, user_prefs)

        tutorial = get_tutorial_message(TUTORIAL_COMPLETED, lang)
        await query.edit_message_text(
            f"<b>{tutorial['title']}</b>\n\n{tutorial['content']}",
            reply_markup=InlineKeyboardMarkup(tutorial["buttons"]),
            parse_mode="HTML",
        )
        return ConversationHandler.END

    elif query.data == "tutorial_commands":
        await query.edit_message_text(
            f"<b>{t('tutorial.commands_list.title', lang)}</b>\n\n{t('tutorial.commands_list.content', lang)}",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            t("tutorial.buttons.back_to_tutorial", lang), 
                            callback_data="tutorial_back_to_end"
                        )
                    ]
                ]
            ),
        )

    elif query.data == "tutorial_back_to_end":
        tutorial = get_tutorial_message(TUTORIAL_COMPLETED, lang)
        await query.edit_message_text(
            f"<b>{tutorial['title']}</b>\n\n{tutorial['content']}",
            reply_markup=InlineKeyboardMarkup(tutorial["buttons"]),
            parse_mode="HTML",
        )

    elif query.data == "tutorial_finish":
        await query.edit_message_text(
            t("tutorial.happy_trading", lang),
            parse_mode="HTML",
        )
        return ConversationHandler.END

    return CHOOSING_TUTORIAL_ACTION
