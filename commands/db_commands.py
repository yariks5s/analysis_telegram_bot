from telegram import Update
from telegram.ext import ContextTypes
from back_tester.db_operations import ClickHouseDB
import logging

logger = logging.getLogger(__name__)

# Initialize database connection
db = ClickHouseDB()


async def execute_sql_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Command handler to execute SQL queries.
    Usage: /sql <query>
    Example: /sql SELECT * FROM trades LIMIT 5
    """
    if not context.args:
        await update.message.reply_text(
            "Please provide a SQL query.\n"
            "Usage: /sql <query>\n"
            "Example: /sql SELECT * FROM trades LIMIT 5"
        )
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
            await update.message.reply_text(
                "Query executed successfully. No results returned."
            )
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
        message = f"Query Results:\n\n{formatted_result}"

        # Send the formatted message
        await update.message.reply_text(message)

    except Exception as e:
        logger.error(f"Error executing SQL query: {str(e)}")
        await update.message.reply_text(f"❌ Error executing query: {str(e)}")


async def show_tables_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Command handler to show available tables.
    Usage: /tables
    """
    try:
        tables = db.get_available_tables()
        if isinstance(tables, str):  # Error occurred
            await update.message.reply_text(f"❌ Error: {tables}")
            return

        if not tables:
            await update.message.reply_text("No tables found in the database.")
            return

        message = "Available tables:\n\n"
        for table in tables:
            message += f"• {table}\n"

        await update.message.reply_text(message)

    except Exception as e:
        logger.error(f"Error getting tables: {str(e)}")
        await update.message.reply_text(f"❌ Error: {str(e)}")


async def describe_table_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Command handler to show table schema.
    Usage: /schema <table_name>
    Example: /schema trades
    """
    if not context.args:
        await update.message.reply_text(
            "Please provide a table name.\n"
            "Usage: /schema <table_name>\n"
            "Example: /schema trades"
        )
        return

    table_name = context.args[0]

    try:
        schema = db.get_table_schema(table_name)
        if isinstance(schema, str):  # Error occurred
            await update.message.reply_text(f"❌ Error: {schema}")
            return

        if not schema:
            await update.message.reply_text(
                f"No schema found for table '{table_name}'."
            )
            return

        message = f"Schema for table '{table_name}':\n\n"
        for column in schema:
            message += f"• {column['name']}: {column['type']}\n"
            if column["default"]:
                message += f"  Default: {column['default']}\n"

        await update.message.reply_text(message)

    except Exception as e:
        logger.error(f"Error getting table schema: {str(e)}")
        await update.message.reply_text(f"❌ Error: {str(e)}")
