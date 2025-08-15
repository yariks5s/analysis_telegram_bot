#!/usr/bin/env python3
"""
Test script for validating signal history functionality.
"""

import sys
import os
import json
from datetime import datetime
import pandas as pd
import unittest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.models import SignalHistory
from src.database.operations import (
    save_signal_history,
    get_user_signal_history,
    get_signal_history_by_date_range,
)
from src.telegram.signals.detection import TradingSignal


class TestSignalHistory(unittest.TestCase):
    """Test case for signal history functionality"""

    def setUp(self):
        """Set up test environment"""
        self.test_user_id = 12345
        self.test_currency_pair = "BTC/USDT"

        self.trading_signal = TradingSignal(
            symbol="BTC/USDT",
            signal_type="Bullish",
            probability=0.75,
            confidence=0.8,
            entry_price=50000.0,
            stop_loss=48000.0,
            take_profit_1=52000.0,
            take_profit_2=54000.0,
            take_profit_3=56000.0,
            risk_reward_ratio=2.0,
            position_size=0.5,
            max_risk_amount=1000.0,
            reasons=["RSI oversold", "Price above support"],
            market_conditions={"volume_ratio": 1.2, "volatility": 0.05},
            timestamp=pd.Timestamp.now(),
        )

        self.signal_data = {
            "user_id": self.test_user_id,
            "currency_pair": self.test_currency_pair,
            "signal_type": self.trading_signal.signal_type,
            "probability": self.trading_signal.probability,
            "confidence": self.trading_signal.confidence,
            "entry_price": self.trading_signal.entry_price,
            "stop_loss": self.trading_signal.stop_loss,
            "take_profit_1": self.trading_signal.take_profit_1,
            "take_profit_2": self.trading_signal.take_profit_2,
            "take_profit_3": self.trading_signal.take_profit_3,
            "risk_reward_ratio": self.trading_signal.risk_reward_ratio,
            "position_size": self.trading_signal.position_size,
            "max_risk_amount": self.trading_signal.max_risk_amount,
            "reasons": self.trading_signal.reasons,
            "market_conditions": self.trading_signal.market_conditions,
            "timestamp": self.trading_signal.timestamp.isoformat(),
        }

    @patch("src.database.operations.get_db_connection")
    def test_save_signal_history(self, mock_get_conn):
        """Test saving a signal to history"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.lastrowid = 1

        result = save_signal_history(self.signal_data)

        mock_cursor.execute.assert_called_once()
        self.assertTrue(result)

    @patch("src.database.operations.get_db_connection")
    def test_get_user_signal_history(self, mock_get_conn):
        """Test retrieving signal history for a user"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchall.return_value = [
            (
                1,
                self.test_user_id,
                self.test_currency_pair,
                self.trading_signal.signal_type,
                self.trading_signal.probability,
                self.trading_signal.confidence,
                self.trading_signal.entry_price,
                self.trading_signal.stop_loss,
                self.trading_signal.take_profit_1,
                self.trading_signal.take_profit_2,
                self.trading_signal.take_profit_3,
                self.trading_signal.risk_reward_ratio,
                self.trading_signal.position_size,
                self.trading_signal.max_risk_amount,
                json.dumps(self.trading_signal.reasons),
                json.dumps(self.trading_signal.market_conditions),
                self.trading_signal.timestamp.isoformat(),
            )
        ]

        result = get_user_signal_history(self.test_user_id)

        mock_cursor.execute.assert_called_once()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["user_id"], self.test_user_id)
        self.assertEqual(result[0]["currency_pair"], self.test_currency_pair)

    @patch("src.database.operations.get_db_connection")
    def test_get_signal_history_by_date_range(self, mock_get_conn):
        """Test retrieving signal history by date range"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchall.return_value = [
            (
                1,
                self.test_user_id,
                self.test_currency_pair,
                self.trading_signal.signal_type,
                self.trading_signal.probability,
                self.trading_signal.confidence,
                self.trading_signal.entry_price,
                self.trading_signal.stop_loss,
                self.trading_signal.take_profit_1,
                self.trading_signal.take_profit_2,
                self.trading_signal.take_profit_3,
                self.trading_signal.risk_reward_ratio,
                self.trading_signal.position_size,
                self.trading_signal.max_risk_amount,
                json.dumps(self.trading_signal.reasons),
                json.dumps(self.trading_signal.market_conditions),
                self.trading_signal.timestamp.isoformat(),
            )
        ]

        start_date = (
            datetime.now()
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .isoformat()
        )
        end_date = (
            datetime.now()
            .replace(hour=23, minute=59, second=59, microsecond=999999)
            .isoformat()
        )

        result = get_signal_history_by_date_range(
            self.test_user_id, start_date, end_date
        )

        mock_cursor.execute.assert_called_once()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["user_id"], self.test_user_id)
        self.assertEqual(result[0]["currency_pair"], self.test_currency_pair)
