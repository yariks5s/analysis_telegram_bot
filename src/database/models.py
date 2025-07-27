"""
Database models for CryptoBot.

This module contains database table definitions and schema information.
"""

import sqlite3
from dataclasses import dataclass
from typing import Dict, List

from src.core.config import DATABASE_PATH, logger


@dataclass
class UserPreference:
    """User chart and indicator preferences."""
    user_id: int
    order_blocks: bool = False
    fvgs: bool = False
    liquidity_levels: bool = False
    breaker_blocks: bool = False
    show_legend: bool = True
    show_volume: bool = True
    liquidity_pools: bool = True
    dark_mode: bool = False


@dataclass
class SignalRequest:
    """User signal request configuration."""
    user_id: int
    currency_pair: str
    frequency_minutes: int
    is_with_chart: bool = False
    account_balance: float = 10000.0
    risk_percentage: float = 1.0


@dataclass
class UserChat:
    """User-chat ID mapping."""
    user_id: int
    chat_id: int


def create_tables():
    """
    Create the database tables if they don't exist.
    """
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id INTEGER PRIMARY KEY,
            order_blocks BOOLEAN DEFAULT 0,
            fvgs BOOLEAN DEFAULT 0,
            liquidity_levels BOOLEAN DEFAULT 0,
            breaker_blocks BOOLEAN DEFAULT 0,
            show_legend BOOLEAN DEFAULT 1,
            show_volume BOOLEAN DEFAULT 1,
            liquidity_pools BOOLEAN DEFAULT 1,
            dark_mode BOOLEAN DEFAULT 0
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_signals_requests (
            user_id INTEGER,
            currency_pair VARCHAR DEFAULT 'BTCUSDT',
            frequency_minutes INTEGER DEFAULT 60,
            is_with_chart BOOL default 0,
            PRIMARY KEY (user_id, currency_pair)
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_chats (
            user_id INTEGER PRIMARY KEY,
            chat_id INTEGER NOT NULL
        )
        """)
        
        conn.commit()
        logger.info("Database tables created successfully")
        
    except sqlite3.Error as e:
        logger.error(f"Database error during table creation: {e}")
    finally:
        if conn:
            conn.close()


def get_tables_info() -> List[Dict]:
    """
    Get information about all tables in the database.
    
    Returns:
        List of dictionaries with table information
    """
    tables_info = []
    conn = None
    
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            column_info = []
            for col in columns:
                column_info.append({
                    "name": col[1],
                    "type": col[2],
                    "notnull": bool(col[3]),
                    "default": col[4],
                    "primary_key": bool(col[5])
                })
            
            tables_info.append({
                "name": table_name,
                "columns": column_info
            })
        
    except sqlite3.Error as e:
        logger.error(f"Error getting table information: {e}")
    finally:
        if conn:
            conn.close()
            
    return tables_info
