"""
Status command handlers for the Telegram bot.

This module contains commands related to showing system status,
including rate limits and quotas.
"""

from telegram import Update
from telegram.ext import ContextTypes

from src.core.rate_limiter import get_rate_limit_stats
from src.i18n import t, get_user_language


async def rate_limit_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Show current rate limit status for the user.

    Args:
        update: The update object from Telegram
        context: The context object from Telegram
    """
    user_id = update.effective_user.id
    lang = get_user_language(user_id)
    quota = get_rate_limit_stats(user_id)

    status_emoji = "✅" if quota["remaining"] > 5 else "⚠️"
    suspicious_suffix = t("rate_limit.suspicious_suffix", lang) if quota["is_suspicious"] else ""

    message = (
        f"{status_emoji} {t('rate_limit.title', lang)}\n\n"
        f"{t('rate_limit.remaining', lang, remaining=quota['remaining'], limit=quota['limit'], suffix=suspicious_suffix)}\n"
        f"{t('rate_limit.reset_in', lang, seconds=quota['reset_seconds'])}\n\n"
    )

    if quota["is_suspicious"]:
        message += f"{t('rate_limit.reduced_notice', lang)}\n\n"

    message += t("rate_limit.footer", lang)

    await update.message.reply_text(message, parse_mode="Markdown")
