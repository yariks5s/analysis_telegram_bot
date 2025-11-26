"""
Language selection command handlers for the Telegram bot.

This module provides commands for users to change their preferred language.
"""

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from src.i18n import (
    t,
    get_user_language,
    set_user_language,
    get_language_keyboard_data,
    SUPPORTED_LANGUAGES,
)


async def language_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /language - Opens the language selection menu.
    
    Args:
        update: Telegram update object
        context: Telegram context object
    """
    user_id = update.effective_user.id
    lang = get_user_language(user_id)
    
    keyboard = build_language_keyboard(lang)
    
    await update.message.reply_text(
        t("language.select_prompt", lang),
        reply_markup=keyboard,
    )


def build_language_keyboard(current_lang: str) -> InlineKeyboardMarkup:
    """
    Build the inline keyboard for language selection.
    
    Args:
        current_lang: Currently selected language code
    
    Returns:
        InlineKeyboardMarkup with language options
    """
    keyboard_data = get_language_keyboard_data()
    keyboard = []
    
    row = []
    for code, name, flag in keyboard_data:
        # Mark current language with a checkmark
        if code == current_lang:
            label = f"✓ {flag} {name}"
        else:
            label = f"{flag} {name}"
        
        row.append(
            InlineKeyboardButton(label, callback_data=f"lang_set:{code}")
        )
        
        # Create rows of 2 buttons each
        if len(row) == 2:
            keyboard.append(row)
            row = []
    
    # Add remaining buttons if any
    if row:
        keyboard.append(row)
    
    return InlineKeyboardMarkup(keyboard)


async def handle_language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle language selection callback from inline keyboard.
    
    Args:
        update: Telegram update object
        context: Telegram context object
    """
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    callback_data = query.data
    
    if callback_data.startswith("lang_set:"):
        new_lang = callback_data.replace("lang_set:", "")
        
        if new_lang in SUPPORTED_LANGUAGES:
            # Update user's language preference
            success = set_user_language(user_id, new_lang)
            
            if success:
                # Reply in the new language
                await query.edit_message_text(
                    t("language.changed", new_lang),
                )
            else:
                # Fallback message
                await query.edit_message_text(
                    "❌ Failed to update language. Please try again.",
                )
        else:
            await query.edit_message_text(
                "❌ Invalid language selection.",
            )

