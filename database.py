import sqlite3
from typing import List, Dict

def init_db() -> None:
    """
    Initialize the SQLite database and create the tables if they don't exist.
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
            user_id INTEGER,
            currency_pair VARCHAR DEFAULT 'BTCUSDT',
            frequency_minutes INTEGER DEFAULT 60,
            PRIMARY KEY (user_id, currency_pair)
        )
    """)
    # Example table for user-chat mapping
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_chats (
            user_id INTEGER PRIMARY KEY,
            chat_id INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def get_user_preferences(user_id: int) -> Dict[str, bool]:
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
    Check if the user has any preferences set.
    """
    conn = sqlite3.connect("preferences.db")
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM user_preferences WHERE user_id = ?", (user_id,))
    exists = cursor.fetchone()
    conn.close()
    return bool(exists)


def update_user_preferences(user_id: int, preferences: Dict[str, bool]) -> None:
    """
    Update or insert the user's indicator preferences in the database.
    """
    try:
        conn = sqlite3.connect("preferences.db")
        cursor = conn.cursor()

        cursor.execute("SELECT 1 FROM user_preferences WHERE user_id = ?", (user_id,))
        exists = cursor.fetchone()

        if exists:
            cursor.execute("""
                UPDATE user_preferences
                SET order_blocks = ?, fvgs = ?, liquidity_levels = ?, breaker_blocks = ?
                WHERE user_id = ?
            """, (preferences["order_blocks"], preferences["fvgs"],
                  preferences["liquidity_levels"], preferences["breaker_blocks"], user_id))
        else:
            cursor.execute("""
                INSERT INTO user_preferences (user_id, order_blocks, fvgs, liquidity_levels, breaker_blocks)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, preferences["order_blocks"], preferences["fvgs"],
                  preferences["liquidity_levels"], preferences["breaker_blocks"]))

        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()


def get_all_user_signal_requests(user_id: int) -> List[Dict[str, any]]:
    """
    Retrieve all signal request preferences for a user from the database.
    """
    conn = sqlite3.connect("preferences.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT currency_pair, frequency_minutes 
        FROM user_signals_requests 
        WHERE user_id = ?
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "currency_pair": row[0],
            "frequency_minutes": row[1],
        } for row in rows
    ]


def upsert_user_signal_request(user_id: int, signals_request: Dict[str, any]) -> None:
    """
    Update or insert a user's signal request for a specific currency pair in the database.
    """
    try:
        conn = sqlite3.connect("preferences.db")
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO user_signals_requests (user_id, currency_pair, frequency_minutes)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, currency_pair) DO UPDATE SET
                frequency_minutes=excluded.frequency_minutes
        """, (user_id, signals_request["currency_pair"], signals_request["frequency_minutes"]))

        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()


def delete_user_signal_request(user_id: int, currency_pair: str) -> None:
    """
    Delete a specific signal request for a user from the database.
    """
    try:
        conn = sqlite3.connect("preferences.db")
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM user_signals_requests 
            WHERE user_id = ? AND currency_pair = ?
        """, (user_id, currency_pair))

        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()


def delete_all_user_signal_requests(user_id: int) -> None:
    """
    Delete all signal requests for a user from the database.
    """
    try:
        conn = sqlite3.connect("preferences.db")
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM user_signals_requests 
            WHERE user_id = ?
        """, (user_id,))

        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()


def get_chat_id_for_user(user_id: int) -> int:
    """
    Retrieve the chat_id for a given user_id.
    """
    conn = sqlite3.connect("preferences.db")
    cursor = conn.cursor()
    cursor.execute("SELECT chat_id FROM user_chats WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0]
    else:
        return None  # Handle appropriately
    
def get_signal_requests():
    signal_requests = []
    try:
        conn = sqlite3.connect("preferences.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_id, currency_pair, frequency_minutes 
            FROM user_signals_requests
        """)
        rows = cursor.fetchall()
        signal_requests = [
            {
                "user_id": row[0],
                "currency_pair": row[1],
                "frequency_minutes": row[2],
            } for row in rows
        ]
    except sqlite3.Error as e:
        print(f"Database error during job initialization: {e}")
    finally:
        conn.close()

    return signal_requests

