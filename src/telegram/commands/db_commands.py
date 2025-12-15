from telegram import Update
from telegram.ext import ContextTypes
from back_tester.db_operations import ClickHouseDB
import logging

from src.i18n import t, get_user_language

logger = logging.getLogger(__name__)

# Initialize database connection
db = ClickHouseDB()


async def execute_sql_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Command handler to execute SQL queries.
    Usage: /sql <query>
    Example: /sql SELECT * FROM trades LIMIT 5
    """
    user_id = update.effective_user.id
    lang = get_user_language(user_id)

    if not context.args:
        await update.message.reply_text(t("database.sql.no_query", lang))
        return

    # Join all arguments to form the complete query
    query = " ".join(context.args)

    try:
        # Add FORMAT TabSeparated if not already present
        if "FORMAT" not in query.upper():
            query = f"{query} FORMAT PrettyCompactMonoBlock"

        # Execute the query and get results
        result = db.execute_query(query)

        if isinstance(result, str) and result.startswith("Error:"):
            await update.message.reply_text(f"❌ {result}")
            return

        if not result:
            await update.message.reply_text(t("database.sql.success_no_results", lang))
            return

        # Convert result to string
        if isinstance(result, list):
            # If result is a list of rows, join them with newlines
            formatted_result = "\n".join(
                "\t".join(str(val) for val in row) for row in result
            )
        else:
            formatted_result = str(result)

        # Format the message
        message = t("database.sql.results", lang, results=formatted_result)

        # Send the formatted message
        await update.message.reply_text(message)

    except Exception as e:
        logger.error(f"Error executing SQL query: {str(e)}")
        await update.message.reply_text(t("database.sql.error", lang, error=str(e)))


async def show_tables_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Command handler to show available tables.
    Usage: /tables
    """
    user_id = update.effective_user.id
    lang = get_user_language(user_id)

    try:
        tables = db.get_available_tables()
        if isinstance(tables, str):  # Error occurred
            await update.message.reply_text(t("database.tables.error", lang, error=tables))
            return

        if not tables:
            await update.message.reply_text(t("database.tables.empty", lang))
            return

        message = t("database.tables.title", lang)
        for table in tables:
            message += f"• {table}\n"

        await update.message.reply_text(message)

    except Exception as e:
        logger.error(f"Error getting tables: {str(e)}")
        await update.message.reply_text(t("database.tables.error", lang, error=str(e)))


async def describe_table_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Command handler to show table schema.
    Usage: /schema <table_name>
    Example: /schema trades
    """
    user_id = update.effective_user.id
    lang = get_user_language(user_id)

    if not context.args:
        await update.message.reply_text(t("database.schema.no_table", lang))
        return

    table_name = context.args[0]

    try:
        schema = db.get_table_schema(table_name)
        if isinstance(schema, str):  # Error occurred
            await update.message.reply_text(t("database.schema.error", lang, error=schema))
            return

        if not schema:
            await update.message.reply_text(t("database.schema.not_found", lang, table=table_name))
            return

        message = t("database.schema.title", lang, table=table_name)
        for column in schema:
            message += f"• {column['name']}: {column['type']}\n"
            if column["default"]:
                message += f"  Default: {column['default']}\n"

        await update.message.reply_text(message)

    except Exception as e:
        logger.error(f"Error getting table schema: {str(e)}")
        await update.message.reply_text(t("database.schema.error", lang, error=str(e)))
