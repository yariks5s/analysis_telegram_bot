"""
Status command handlers for the Telegram bot.

This module contains commands related to showing system status,
including rate limits and quotas.
"""

from telegram import Update
from telegram.ext import ContextTypes

from src.core.rate_limiter import get_rate_limit_stats


async def rate_limit_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Show current rate limit status for the user.

    Args:
        update: The update object from Telegram
        context: The context object from Telegram
    """
    user_id = update.effective_user.id
    quota = get_rate_limit_stats(user_id)

    status_emoji = "✅" if quota["remaining"] > 5 else "⚠️"
    suspicious_status = (
        " (reduced due to suspicious activity)" if quota["is_suspicious"] else ""
    )

    message = (
        f"{status_emoji} **Rate Limit Status**\n\n"
        f"• Remaining: {quota['remaining']}/{quota['limit']} requests{suspicious_status}\n"
        f"• Reset in: {quota['reset_seconds']} seconds\n\n"
    )

    if quota["is_suspicious"]:
        message += (
            "ℹ️ Your rate limit is temporarily reduced due to detected high-frequency activity. "
            "This will automatically return to normal after a cooling period.\n\n"
        )

    message += "The system uses rate limiting to ensure fair API usage for all users."

    await update.message.reply_text(message, parse_mode="Markdown")
