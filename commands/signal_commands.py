from telegram import Update
from telegram.ext import ContextTypes
from helpers import check_signal_limit, input_sanity_check_analyzing
from utils import plural_helper
from signal_detection import createSignalJob, deleteSignalJob

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
            f"Usage: /create_signal <symbol> <period_in_minutes> [<is_with_chart>], you've sent {len(args)} argument{plural_helper(len(args))}."
        )
    else:
        try:
            await createSignalJob(pair[0], pair[1], pair[2], update, context)
        except Exception as e:
            print(f"Unexpected error: {e}")
            await update.message.reply_text("❌ An unexpected error occurred.")

async def delete_signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Command handler to delete a specific signal job.
    Usage: /delete_signal <SYMBOL>
    Example: /delete_signal BTCUSDT
    """
    args = context.args
    pair = await input_sanity_check_analyzing(False, args, update)
    if not pair:
        await update.message.reply_text(
            f"Usage: /delete_signal <symbol>, you've sent {len(args)} argument{plural_helper(len(args))}."
        )
    else:
        try:
            await deleteSignalJob(pair[0], update)
        except Exception as e:
            print(f"Unexpected error: {e}")
            await update.message.reply_text("❌ An unexpected error occurred.") 