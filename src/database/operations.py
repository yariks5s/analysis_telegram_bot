"""
Database operations module for CryptoBot.

This module handles all database interactions, including initialization,
user preferences management, and signal request management.
"""

import sqlite3
from typing import List, Dict, Optional, Any, Union

from src.core.config import DATABASE_PATH, logger
from src.database.models import create_tables
from src.core.preferences import INDICATOR_PARAMS


def init_db() -> None:
    """
    Initialize the SQLite database and create the tables if they don't exist.
    """
    create_tables()


def get_user_preferences(user_id: int) -> Dict[str, Union[bool, int, float]]:
    """
    Retrieve the user's indicator preferences from the database.

    Args:
        user_id: Telegram user ID

    Returns:
        Dictionary of user preferences
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        prefs = {
            "order_blocks": bool(row[1]),
            "fvgs": bool(row[2]),
            "liquidity_levels": bool(row[3]),
            "breaker_blocks": bool(row[4]),
            "show_legend": bool(row[5]),
            "show_volume": bool(row[6]),
            "liquidity_pools": bool(row[7]) if len(row) > 7 else True,
            "dark_mode": bool(row[8]) if len(row) > 8 else False,
            "tutorial_stage": int(row[11]) if len(row) > 11 else 0,
        }

        # Handle indicator parameters if they exist in the database
        if len(row) > 9:
            prefs["atr_period"] = (
                int(row[9])
                if row[9] is not None
                else INDICATOR_PARAMS["atr_period"]["default"]
            )
            prefs["fvg_min_size"] = (
                float(row[10])
                if row[10] is not None
                else INDICATOR_PARAMS["fvg_min_size"]["default"]
            )
        else:
            prefs["atr_period"] = INDICATOR_PARAMS["atr_period"]["default"]
            prefs["fvg_min_size"] = INDICATOR_PARAMS["fvg_min_size"]["default"]

        return prefs
    else:
        prefs = {
            "order_blocks": False,
            "fvgs": False,
            "liquidity_levels": False,
            "breaker_blocks": False,
            "show_legend": True,
            "show_volume": True,
            "liquidity_pools": False,
            "dark_mode": False,
            "tutorial_stage": 0,
        }

        prefs["atr_period"] = INDICATOR_PARAMS["atr_period"]["default"]
        prefs["fvg_min_size"] = INDICATOR_PARAMS["fvg_min_size"]["default"]

        return prefs


def check_user_preferences(user_id: int) -> bool:
    """
    Check if the user has any preferences set.

    Args:
        user_id: Telegram user ID

    Returns:
        True if user has preferences, False otherwise
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM user_preferences WHERE user_id = ?", (user_id,))
    exists = cursor.fetchone()
    conn.close()
    return bool(exists)


def update_user_preferences(user_id: int, preferences: Dict[str, Any]) -> None:
    """
    Update or insert the user's indicator preferences in the database.

    Args:
        user_id: Telegram user ID
        preferences: Dictionary of user preferences to update
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT 1 FROM user_preferences WHERE user_id = ?", (user_id,))
        exists = cursor.fetchone()

        if exists:
            cursor.execute(
                """
                UPDATE user_preferences
                SET order_blocks = ?, fvgs = ?, liquidity_levels = ?, breaker_blocks = ?,
                    show_legend = ?, show_volume = ?, liquidity_pools = ?, dark_mode = ?,
                    atr_period = ?, fvg_min_size = ?, tutorial_stage = ?
                WHERE user_id = ?
            """,
                (
                    preferences["order_blocks"],
                    preferences["fvgs"],
                    preferences["liquidity_levels"],
                    preferences["breaker_blocks"],
                    preferences["show_legend"],
                    preferences["show_volume"],
                    preferences["liquidity_pools"],
                    preferences["dark_mode"],
                    preferences.get(
                        "atr_period", INDICATOR_PARAMS["atr_period"]["default"]
                    ),
                    preferences.get(
                        "fvg_min_size", INDICATOR_PARAMS["fvg_min_size"]["default"]
                    ),
                    preferences.get("tutorial_stage", 0),
                    user_id,
                ),
            )
        else:
            cursor.execute(
                """
                INSERT INTO user_preferences (
                    user_id, order_blocks, fvgs, liquidity_levels, breaker_blocks,
                    show_legend, show_volume, liquidity_pools, dark_mode,
                    atr_period, fvg_min_size, tutorial_stage
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    user_id,
                    preferences["order_blocks"],
                    preferences["fvgs"],
                    preferences["liquidity_levels"],
                    preferences["breaker_blocks"],
                    preferences["show_legend"],
                    preferences["show_volume"],
                    preferences["liquidity_pools"],
                    preferences["dark_mode"],
                    preferences.get(
                        "atr_period", INDICATOR_PARAMS["atr_period"]["default"]
                    ),
                    preferences.get(
                        "fvg_min_size", INDICATOR_PARAMS["fvg_min_size"]["default"]
                    ),
                ),
            )

        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
    finally:
        conn.close()


def get_all_user_signal_requests(user_id: int) -> List[Dict[str, any]]:
    """
    Retrieve all signal request preferences for a user from the database.

    Args:
        user_id: Telegram user ID

    Returns:
        List of signal request dictionaries
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT currency_pair, frequency_minutes, is_with_chart
        FROM user_signals_requests 
        WHERE user_id = ?
    """,
        (user_id,),
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        {"currency_pair": row[0], "frequency_minutes": row[1], "is_with_chart": row[2]}
        for row in rows
    ]


def upsert_user_signal_request(user_id: int, signals_request: Dict[str, any]) -> None:
    """
    Update or insert a user's signal request for a specific currency pair in the database.

    Args:
        user_id: Telegram user ID
        signals_request: Signal request configuration
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO user_signals_requests (user_id, currency_pair, frequency_minutes, is_with_chart)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, currency_pair) DO UPDATE SET
                frequency_minutes=excluded.frequency_minutes
        """,
            (
                user_id,
                signals_request["currency_pair"],
                signals_request["frequency_minutes"],
                signals_request["is_with_chart"],
            ),
        )

        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
    finally:
        conn.close()


def delete_user_signal_request(user_id: int, currency_pair: str) -> None:
    """
    Delete a specific signal request for a user from the database.

    Args:
        user_id: Telegram user ID
        currency_pair: Currency pair to delete (e.g. "BTCUSDT")
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM user_signals_requests 
            WHERE user_id = ? AND currency_pair = ?
        """,
            (user_id, currency_pair),
        )

        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
    finally:
        conn.close()


def delete_all_user_signal_requests(user_id: int) -> None:
    """
    Delete all signal requests for a user from the database.

    Args:
        user_id: Telegram user ID
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM user_signals_requests 
            WHERE user_id = ?
        """,
            (user_id,),
        )

        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
    finally:
        conn.close()


def get_chat_id_for_user(user_id: int) -> Optional[int]:
    """
    Retrieve the chat_id for a given user_id.

    Args:
        user_id: Telegram user ID

    Returns:
        Chat ID associated with the user or None if not found
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT chat_id FROM user_chats WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def get_signal_requests() -> List[Dict[str, any]]:
    """
    Get all signal requests from the database.

    Returns:
        List of all signal request configurations
    """
    signal_requests = []
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT user_id, currency_pair, frequency_minutes 
            FROM user_signals_requests
        """
        )
        rows = cursor.fetchall()
        signal_requests = [
            {
                "user_id": row[0],
                "currency_pair": row[1],
                "frequency_minutes": row[2],
            }
            for row in rows
        ]
    except sqlite3.Error as e:
        logger.error(f"Database error during job initialization: {e}")
    finally:
        conn.close()

    return signal_requests


def user_signal_request_exists(user_id: int, currency_pair: str) -> bool:
    """
    Check if a user already has a signal for the specified currency pair.

    Args:
        user_id: Telegram user ID
        currency_pair: Currency pair to check (e.g. "BTCUSDT")

    Returns:
        True if the signal request exists, False otherwise
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT 1 FROM user_signals_requests
        WHERE user_id = ? AND currency_pair = ?
    """,
        (user_id, currency_pair),
    )
    row = cursor.fetchone()
    conn.close()
    return bool(row)
