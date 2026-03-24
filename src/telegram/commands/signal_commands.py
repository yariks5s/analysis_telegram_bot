import logging

from telegram import Update
from telegram.ext import ContextTypes

from src.analysis.utils.helpers import check_signal_limit, input_sanity_check_analyzing
from src.core.utils import plural_helper
from src.telegram.signals.detection import createSignalJob, deleteSignalJob

logger = logging.getLogger(__name__)


async def create_signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Command handler to create a signal job.
    Usage: /create_signal <SYMBOL> <MINUTES> [<IS_WITH_CHART>]
    Example: /create_signal BTCUSDT 60 True
    """
    if await check_signal_limit(update):
        return

    args = context.args
    pair = await input_sanity_check_analyzing(True, args, update)
    if not pair:
        await update.message.reply_text(
            f"Usage: /create_signal <symbol> <period_in_minutes> [<is_with_chart>], "
            f"you've sent {len(args)} argument{plural_helper(len(args))}."
        )
    else:
        try:
            await createSignalJob(pair[0], pair[1], pair[2], update, context)
        except Exception as e:
            logger.error(f"Unexpected error creating signal: {e}")
            await update.message.reply_text("❌ An unexpected error occurred.")


async def delete_signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Command handler to delete a specific signal job.
    Usage: /delete_signal <SYMBOL>
    Example: /delete_signal BTCUSDT
    """
    args = context.args

    if not args:
        await update.message.reply_text("Usage: /delete_signal <symbol>")
        return

    try:
        await deleteSignalJob(args[0].upper(), update)
    except ValueError as e:
        logger.error(f"Value error in delete_signal: {e}")
        await update.message.reply_text(f"❌ Invalid symbol: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in delete_signal: {e}")
        await update.message.reply_text("❌ An unexpected error occurred.")
