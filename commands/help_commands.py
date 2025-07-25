from telegram import Update
from telegram.ext import ContextTypes

COMMAND_HELPS = {
    "chart": (
        "<b>/chart &lt;symbol&gt; &lt;length&gt; &lt;interval&gt; &lt;tolerance&gt;</b>\n"
        "Get a candlestick chart with all enabled indicators.\n\n"
        "- <b>symbol</b>: The trading pair, e.g. BTCUSDT\n"
        "- <b>length</b>: Amount of candles to analyze (e.g. 48)\n"
        "- <b>interval</b>: Candle interval (e.g. 1h, 15m)\n"
        "- <b>tolerance</b>: Sensitivity for liquidity level detection (0-1, e.g. 0.05 = more levels, 0.2 = fewer, only strongest)\n\n"
        "Example: <code>/chart BTCUSDT 48 1h 0.05</code>\n\n"
        "The chart will include all indicators you have enabled in your preferences."
    ),
    "text_result": (
        "<b>/text_result &lt;symbol&gt; &lt;length&gt; &lt;interval&gt; &lt;tolerance&gt;</b>\n"
        "Get a text summary of all detected indicators for the specified symbol and timeframe.\n\n"
        "Example: <code>/text_result ETHUSDT 24 15m 0.03</code>"
    ),
    "history": (
        "<b>/history &lt;symbol&gt; &lt;length&gt; &lt;interval&gt; &lt;tolerance&gt; &lt;timestamp&gt;</b>\n"
        "Get a candlestick chart for historical data at a specified timestamp.\n\n"
        "- <b>symbol</b>: The trading pair, e.g. BTCUSDT\n"
        "- <b>length</b>: Number of candles to analyze\n"
        "- <b>interval</b>: Candle interval (e.g., 1h, 15m)\n"
        "- <b>tolerance</b>: Sensitivity for liquidity level detection (0-1)\n"
        "- <b>timestamp</b>: Unix epoch seconds for the end of the period\n\n"
        "Example: <code>/history BTCUSDT 48 1h 0.05 1700000000</code>"
    ),
    "preferences": (
        "<b>/preferences</b>\n"
        "Open an interactive menu to select which indicators to use and chart options.\n\n"
        "Available options:\n"
        "- Technical indicators (Order Blocks, FVGs, Liquidity Levels, etc.)\n"
        "- Display settings (Show Legend, Show Volume)\n"
        "- Theme selection (Light/Dark mode)\n\n"
        "Your preferences are saved per user."
    ),
    "create_signal": (
        "<b>/create_signal &lt;symbol&gt; &lt;minutes&gt; [&lt;show_chart&gt;]</b>\n"
        "Start receiving auto-signals for a pair at a given frequency (in minutes).\n\n"
        "- <b>symbol</b>: The trading pair, e.g. BTCUSDT\n"
        "- <b>minutes</b>: Frequency in minutes to receive signals\n"
        "- <b>show_chart</b> (optional): true/false, whether to include a chart with each signal (default: false)\n\n"
        "Example: <code>/create_signal BTCUSDT 60 true</code>\n\n"
        "You can have up to 10 active signal jobs per user."
    ),
    "delete_signal": (
        "<b>/delete_signal &lt;symbol&gt;</b>\n"
        "Stop auto-signals for a pair.\n\n"
        "Example: <code>/delete_signal BTCUSDT</code>"
    ),
    "manage_signals": (
        "<b>/manage_signals</b>\n"
        "Open an interactive menu to view, add, or delete your signal jobs.\n\n"
        "You can have up to 10 active signal jobs per user."
    ),
    "help": (
        "<b>/help</b>\nShow the general help message.\n\n"
        "<b>/help &lt;command&gt;</b>\nShow detailed help for a specific command.\n\n"
        "Example: <code>/help chart</code>"
    ),
    "sql": (
        "<b>/sql &lt;query&gt;</b>\n"
        "Execute a custom SQL query against the database.\n\n"
        "Example queries:\n"
        "- <code>/sql SELECT * FROM trades LIMIT 5</code>\n"
        "- <code>/sql SELECT symbol, COUNT(*) as trades FROM trades GROUP BY symbol</code>\n"
        "- <code>/sql SELECT AVG(profit_loss) as avg_profit FROM trades WHERE symbol = 'BTCUSDT'</code>\n\n"
        "Note: Results are limited to 50 rows for readability."
    ),
    "tables": (
        "<b>/tables</b>\n"
        "Show a list of all available tables in the database.\n\n"
        "Use this command to see what tables you can query with /sql."
    ),
    "schema": (
        "<b>/schema &lt;table_name&gt;</b>\n"
        "Show the schema (structure) of a specific table.\n\n"
        "Example: <code>/schema trades</code>\n\n"
        "This will show you all columns and their data types in the specified table."
    ),
}


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Sends a help message with usage instructions for the bot, or detailed help for a specific command if provided.
    """
    args = context.args
    if args and args[0].lower() in COMMAND_HELPS:
        help_text = COMMAND_HELPS[args[0].lower()]
    else:
        help_text = (
            "<b>CryptoBot Help</b>\n\n"
            "Here are the main commands you can use:\n\n"
            "<b>/chart &lt;symbol&gt; &lt;length&gt; &lt;interval&gt; &lt;tolerance&gt;</b>\n"
            "<b>/text_result &lt;symbol&gt; &lt;length&gt; &lt;interval&gt; &lt;tolerance&gt;</b>\n"
            "<b>/history &lt;symbol&gt; &lt;length&gt; &lt;interval&gt; &lt;tolerance&gt; &lt;timestamp&gt;</b>\n"
            "<b>/preferences</b>\n"
            "<b>/create_signal &lt;symbol&gt; &lt;minutes&gt; [&lt;show_chart&gt;]</b>\n"
            "<b>/delete_signal &lt;symbol&gt;</b>\n"
            "<b>/manage_signals</b>\n"
            "<b>/sql &lt;query&gt;</b>\n"
            "<b>/tables</b>\n"
            "<b>/schema &lt;table_name&gt;</b>\n"
            "<b>/help &lt;command&gt;</b>\n\n"
            "Type <code>/help &lt;command&gt;</code> to get detailed help for a specific command.\n\n"
            "Example: <code>/help chart</code>\n\n"
            "For more details, see the README or contact the maintainer (@yarik_is_working)."
        )
    await update.message.reply_text(help_text, parse_mode="HTML")
