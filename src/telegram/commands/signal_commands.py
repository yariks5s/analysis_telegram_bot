from telegram import Update
from telegram.ext import ContextTypes
from src.analysis.utils.helpers import check_signal_limit, input_sanity_check_analyzing
from src.telegram.signals.detection import createSignalJob, deleteSignalJob
from src.i18n import t, get_user_language


async def create_signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Command handler to create a signal job.
    Usage: /create_signal <SYMBOL> <MINUTES> [<IS_WITH_CHART>]
    Example: /create_signal BTCUSDT 60 True
    """
    user_id = update.effective_user.id
    lang = get_user_language(user_id)

    if await check_signal_limit(update):
        return

    args = context.args
    pair = await input_sanity_check_analyzing(True, args, update)
    if not pair:
        await update.message.reply_text(
            t("signals.usage.create", lang, count=len(args))
        )
    else:
        try:
            await createSignalJob(pair[0], pair[1], pair[2], update, context)
        except Exception as e:
            print(f"Unexpected error: {e}")
            await update.message.reply_text(t("signals.errors.unexpected", lang))


async def delete_signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Command handler to delete a specific signal job.
    Usage: /delete_signal <SYMBOL>
    Example: /delete_signal BTCUSDT
    """
    user_id = update.effective_user.id
    lang = get_user_language(user_id)
    args = context.args

    # Check if args list is empty
    if not args:
        await update.message.reply_text(t("signals.usage.delete", lang))
        return

    # Format and store the command in the update object for deleteSignalJob to parse
    # The original command structure expected by deleteSignalJob is:
    # "/delete_signal SYMBOL"

    # We'll manually modify the update object's message text to match what deleteSignalJob expects
    original_text = update.message.text
    update.message.text = f"/delete_signal {args[0].upper()}"

    try:
        await deleteSignalJob(args[0].upper(), update)
    except ValueError as e:
        print(f"Value error in delete_signal: {e}")
        await update.message.reply_text(t("signals.errors.invalid_symbol", lang, symbol=str(e)))
    except Exception as e:
        print(f"Unexpected error in delete_signal: {e}")
        await update.message.reply_text(t("signals.errors.unexpected", lang))
    finally:
        # Restore the original text in case it's needed elsewhere
        update.message.text = original_text
