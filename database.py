import sqlite3

def init_db() -> None:
    """
    Initialize the SQLite database and create the table if it doesn't exist.
    """
    conn = sqlite3.connect("preferences.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id INTEGER PRIMARY KEY,
            order_blocks BOOLEAN DEFAULT 0,
            fvgs BOOLEAN DEFAULT 0,
            liquidity_levels BOOLEAN DEFAULT 0,
            breaker_blocks BOOLEAN DEFAULT 0
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_signals_requests (
            user_id INTEGER PRIMARY KEY,
            currency_pair VARCHAR DEFAULT BTCUSDT,
            frequency_minutes INTEGER DEFAULT 60
        )
    """)
    conn.commit()
    conn.close()

def get_user_preferences(user_id: int) -> dict[str: bool]:
    """
    Retrieve the user's indicator preferences from the database.
    """
    conn = sqlite3.connect("preferences.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "order_blocks": bool(row[1]),
            "fvgs": bool(row[2]),
            "liquidity_levels": bool(row[3]),
            "breaker_blocks": bool(row[4]),
        }
    else:
        return {
            "order_blocks": False,
            "fvgs": False,
            "liquidity_levels": False,
            "breaker_blocks": False,
        }


def check_user_preferences(user_id: int) -> bool:
    """
    Retrieve the user's indicator preferences from the database.
    """
    conn = sqlite3.connect("preferences.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return True
    else:
        return False


def update_user_preferences(user_id: int, preferences: dict[str: bool]) -> None:
    """
    Update or insert the user's indicator preferences in the database.
    """
    conn = sqlite3.connect("preferences.db")
    cursor = conn.cursor()

    # Check if the user already exists
    cursor.execute("SELECT 1 FROM user_preferences WHERE user_id = ?", (user_id,))
    exists = cursor.fetchone()

    if exists:
        # Update preferences
        cursor.execute("""
            UPDATE user_preferences
            SET order_blocks = ?, fvgs = ?, liquidity_levels = ?, breaker_blocks = ?
            WHERE user_id = ?
        """, (preferences["order_blocks"], preferences["fvgs"],
              preferences["liquidity_levels"], preferences["breaker_blocks"], user_id))
    else:
        # Insert new user preferences
        cursor.execute("""
            INSERT INTO user_preferences (user_id, order_blocks, fvgs, liquidity_levels, breaker_blocks)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, preferences["order_blocks"], preferences["fvgs"],
              preferences["liquidity_levels"], preferences["breaker_blocks"]))

    conn.commit()
    conn.close()


def get_user_signal_requests(user_id: int) -> dict:
    """
    Retrieve the user's indicator preferences from the database.
    """
    conn = sqlite3.connect("preferences.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_signals_requests WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "currency_pair": row[5],
            "frequency_minutes": int(row[6]),
        }
    else:
        return {
            "currency_pair": "",
            "frequency_minutes": 0,
        }


def check_user_signals_requests(user_id: int) -> bool:
    """
    Retrieve the user's indicator signal requests from the database.
    """
    conn = sqlite3.connect("preferences.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_signals_requests WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return True
    else:
        return False


def update_user_signals_requests(user_id: int, signals_requests: dict[str: bool]) -> None:
    """
    Update or insert the user's indicator preferences in the database.
    """
    conn = sqlite3.connect("preferences.db")
    cursor = conn.cursor()

    # Check if the user already exists
    cursor.execute("SELECT 1 FROM user_signals_requests WHERE user_id = ?", (user_id,))
    exists = cursor.fetchone()

    if exists:
        # Update preferences
        cursor.execute("""
            UPDATE user_signals_requests
            SET currency_pair = ?, frequency_minutes = ?
            WHERE user_id = ?
        """, (signals_requests["currency_pair"], signals_requests["frequency_minutes"]))
    else:
        # Insert new user preferences
        cursor.execute("""
            INSERT INTO user_signals_requests (user_id, currency_pair, frequency_minutes)
            VALUES (?, ?, ?)
        """, (signals_requests["currency_pair"], signals_requests["frequency_minutes"]))

    conn.commit()
    conn.close()
