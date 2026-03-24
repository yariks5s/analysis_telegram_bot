import sqlite3
import logging

from telegram import Update
from telegram.ext import ContextTypes

from src.core.config import DATABASE_PATH, ADMIN_IDS

logger = logging.getLogger(__name__)

# Only SELECT queries are allowed through the /sql command
_ALLOWED_STATEMENTS = ("SELECT",)


def _is_admin(user_id: int) -> bool:
    return bool(ADMIN_IDS) and user_id in ADMIN_IDS


async def execute_sql_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Admin command to execute read-only SQL queries against the bot database.
    Usage: /sql <query>
    Example: /sql SELECT * FROM user_preferences LIMIT 5
    """
    user_id = update.effective_user.id
    if not _is_admin(user_id):
        await update.message.reply_text("❌ This command is restricted to admins.")
        return

    if not context.args:
        await update.message.reply_text(
            "Please provide a SQL query.\n"
            "Usage: /sql <query>\n"
            "Example: /sql SELECT * FROM user_preferences LIMIT 5"
        )
        return

    query = " ".join(context.args).strip()

    # Only allow SELECT statements to prevent data modification
    if not query.upper().lstrip().startswith(_ALLOWED_STATEMENTS):
        await update.message.reply_text(
            "❌ Only SELECT queries are allowed."
        )
        return

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchmany(50)  # cap at 50 rows
        conn.close()

        if not rows:
            await update.message.reply_text("Query returned no results.")
            return

        col_names = [description[0] for description in cursor.description]
        header = " | ".join(col_names)
        separator = "-" * len(header)
        body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
        message = f"`{header}\n{separator}\n{body}`"

        # Telegram message limit is 4096 chars
        if len(message) > 4000:
            message = message[:4000] + "\n... (truncated)"

        await update.message.reply_text(message, parse_mode="Markdown")

    except sqlite3.Error as e:
        logger.error(f"Error executing SQL query: {e}")
        await update.message.reply_text(f"❌ Database error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error executing SQL: {e}")
        await update.message.reply_text(f"❌ Error: {e}")


async def show_tables_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Admin command to show available tables.
    Usage: /tables
    """
    user_id = update.effective_user.id
    if not _is_admin(user_id):
        await update.message.reply_text("❌ This command is restricted to admins.")
        return

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            await update.message.reply_text("No tables found in the database.")
            return

        message = "Available tables:\n\n" + "\n".join(f"• {row[0]}" for row in rows)
        await update.message.reply_text(message)

    except Exception as e:
        logger.error(f"Error getting tables: {e}")
        await update.message.reply_text(f"❌ Error: {e}")


async def describe_table_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Admin command to show table schema.
    Usage: /schema <table_name>
    Example: /schema user_preferences
    """
    user_id = update.effective_user.id
    if not _is_admin(user_id):
        await update.message.reply_text("❌ This command is restricted to admins.")
        return

    if not context.args:
        await update.message.reply_text(
            "Please provide a table name.\n"
            "Usage: /schema <table_name>\n"
            "Example: /schema user_preferences"
        )
        return

    table_name = context.args[0]

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({sqlite3.escape_string(table_name)})")
        columns = cursor.fetchall()
        conn.close()

        if not columns:
            await update.message.reply_text(
                f"No schema found for table '{table_name}'."
            )
            return

        message = f"Schema for table '{table_name}':\n\n"
        for col in columns:
            # col: (cid, name, type, notnull, default_value, pk)
            line = f"• {col[1]}: {col[2]}"
            if col[4] is not None:
                line += f" (default: {col[4]})"
            if col[5]:
                line += " [PK]"
            message += line + "\n"

        await update.message.reply_text(message)

    except Exception as e:
        logger.error(f"Error getting table schema: {e}")
        await update.message.reply_text(f"❌ Error: {e}")
