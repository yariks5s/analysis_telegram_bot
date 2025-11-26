"""
Help command handlers for the Telegram bot.

This module provides the /help command with multi-language support.
"""

from telegram import Update
from telegram.ext import ContextTypes

from src.i18n import t, get_user_language

# Map command names to translation keys
COMMAND_HELP_KEYS = {
    "parameters": "help.commands.parameters",
    "chart": "help.commands.chart",
    "text_result": "help.commands.text_result",
    "history": "help.commands.history",
    "preferences": "help.commands.preferences",
    "create_signal": "help.commands.create_signal",
    "delete_signal": "help.commands.delete_signal",
    "manage_signals": "help.commands.manage_signals",
    "help": "help.commands.help",
    "sql": "help.commands.sql",
    "tables": "help.commands.tables",
    "schema": "help.commands.schema",
    "language": "help.commands.language",
}


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Sends a help message with usage instructions for the bot,
    or detailed help for a specific command if provided.
    """
    user_id = update.effective_user.id
    lang = get_user_language(user_id)
    
    args = context.args
    if args and args[0].lower() in COMMAND_HELP_KEYS:
        command = args[0].lower()
        help_text = t(COMMAND_HELP_KEYS[command], lang)
    else:
        help_text = f"<b>{t('help.title', lang)}</b>\n\n{t('help.main', lang)}"
    
    await update.message.reply_text(help_text, parse_mode="HTML")
