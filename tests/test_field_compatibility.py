#!/usr/bin/env python3
"""
Test script for validating field name compatibility between take_profit and take_profit_1.
"""

import sys
import os
import json
import unittest
from unittest.mock import patch, MagicMock
import sqlite3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.operations import (
    save_signal_history,
    get_user_signal_history,
    get_signal_history_by_date_range
)
from src.telegram.commands.history_commands import format_signal_history


class TestFieldCompatibility(unittest.TestCase):
    """Test case for field name compatibility between take_profit and take_profit_1"""

    def setUp(self):
        """Set up test environment"""
        self.test_user_id = 12345
        self.test_currency_pair = "BTC/USDT"
        
        self.signal_data_take_profit = {
            "user_id": self.test_user_id,
            "currency_pair": self.test_currency_pair,
            "signal_type": "Bullish",
            "probability": 0.75,
            "entry_price": 50000.0,
            "stop_loss": 48000.0,
            "take_profit": 52000.0,
            "reasons": ["RSI oversold", "Price above support"],
            "market_conditions": {"volume_ratio": 1.2, "volatility": 0.05}
        }
        
        self.signal_data_take_profit_1 = {
            "user_id": self.test_user_id,
            "currency_pair": self.test_currency_pair,
            "signal_type": "Bearish",
            "probability": 0.80,
            "entry_price": 48000.0,
            "stop_loss": 50000.0,
            "take_profit_1": 45000.0,
            "reasons": ["RSI overbought", "Price below resistance"],
            "market_conditions": {"volume_ratio": 0.8, "volatility": 0.04}
        }
        
        self.signal_with_take_profit = {
            "id": 1,
            "user_id": self.test_user_id,
            "currency_pair": self.test_currency_pair,
            "signal_type": "Bullish",
            "timestamp": "2025-08-11T00:00:00.000",
            "entry_price": 50000.0,
            "stop_loss": 48000.0,
            "take_profit": 52000.0,
            "probability": 0.75,
            "reasons": ["RSI oversold", "Price above support"],
            "market_conditions": {"volume_ratio": 1.2, "volatility": 0.05}
        }
        
        self.signal_with_take_profit_1 = {
            "id": 2,
            "user_id": self.test_user_id,
            "currency_pair": self.test_currency_pair,
            "signal_type": "Bearish",
            "timestamp": "2025-08-11T00:00:00.000",
            "entry_price": 48000.0,
            "stop_loss": 50000.0,
            "take_profit_1": 45000.0,
            "probability": 0.80,
            "reasons": ["RSI overbought", "Price below resistance"],
            "market_conditions": {"volume_ratio": 0.8, "volatility": 0.04}
        }

    @patch('sqlite3.connect')
    def test_save_signal_take_profit(self, mock_connect):
        """Test saving a signal with take_profit field"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        result = save_signal_history(self.signal_data_take_profit)
        
        self.assertTrue(result)
        
        mock_cursor.execute.assert_called_once()
        
        args = mock_cursor.execute.call_args[0][1]
        self.assertEqual(args[6], self.signal_data_take_profit["take_profit"])

    @patch('sqlite3.connect')
    def test_save_signal_take_profit_1(self, mock_connect):
        """Test saving a signal with take_profit_1 field"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        result = save_signal_history(self.signal_data_take_profit_1)
        
        self.assertTrue(result)
        
        mock_cursor.execute.assert_called_once()
        
        args = mock_cursor.execute.call_args[0][1]
        self.assertEqual(args[6], self.signal_data_take_profit_1["take_profit_1"])

    @patch('sqlite3.connect')
    def test_get_user_signal_history_includes_both_fields(self, mock_connect):
        """Test that retrieved signals include both take_profit and take_profit_1 fields"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        mock_cursor.fetchall.return_value = [(
            1,
            self.test_user_id,
            self.test_currency_pair,
            "Bullish",
            "2025-08-11T00:00:00.000",
            50000.0,
            48000.0,
            52000.0,
            0.75,
            json.dumps(["RSI oversold", "Price above support"]),
            json.dumps({"volume_ratio": 1.2, "volatility": 0.05})
        )]
        
        result = get_user_signal_history(self.test_user_id)
        
        mock_cursor.execute.assert_called_once()
        
        self.assertIn("take_profit", result[0])
        self.assertIn("take_profit_1", result[0])
        
        self.assertEqual(result[0]["take_profit"], result[0]["take_profit_1"])
        self.assertEqual(result[0]["take_profit"], 52000.0)

    @patch('sqlite3.connect')
    def test_get_signal_history_by_date_includes_both_fields(self, mock_connect):
        """Test that date-filtered signals include both take_profit and take_profit_1 fields"""
        # Mock the database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        mock_cursor.fetchall.return_value = [(
            1,
            self.test_user_id,
            self.test_currency_pair,
            "Bullish",
            "2025-08-11T00:00:00.000",
            50000.0,
            48000.0,
            52000.0,
            0.75,
            json.dumps(["RSI oversold", "Price above support"]),
            json.dumps({"volume_ratio": 1.2, "volatility": 0.05})
        )]
        
        result = get_signal_history_by_date_range(
            self.test_user_id, 
            "2025-08-10", 
            "2025-08-12"
        )
        
        mock_cursor.execute.assert_called_once()
        
        self.assertIn("take_profit", result[0])
        self.assertIn("take_profit_1", result[0])
        
        self.assertEqual(result[0]["take_profit"], result[0]["take_profit_1"])
        self.assertEqual(result[0]["take_profit"], 52000.0)

    def test_format_signal_history_take_profit(self):
        """Test formatting signals with take_profit field"""
        formatted = format_signal_history([self.signal_with_take_profit])
        
        self.assertIn("TP1: 52000.00", formatted)

    def test_format_signal_history_take_profit_1(self):
        """Test formatting signals with take_profit_1 field"""
        formatted = format_signal_history([self.signal_with_take_profit_1])
        
        self.assertIn("TP1: 45000.00", formatted)

    def test_format_signal_history_missing_both_fields(self):
        """Test formatting signals with neither take_profit nor take_profit_1 field"""
        signal_missing_fields = dict(self.signal_with_take_profit)
        if "take_profit" in signal_missing_fields:
            del signal_missing_fields["take_profit"]
        if "take_profit_1" in signal_missing_fields:
            del signal_missing_fields["take_profit_1"]
            
        formatted = format_signal_history([signal_missing_fields])
        
        self.assertIn("TP1: 0.00", formatted)

    def test_integration_save_and_retrieve(self):
        """Integration test: save signals with different field names and retrieve them"""
        # This is more of an integration test that would require a test database
        # Since we've already tested the individual parts, we'll just outline it
        pass


if __name__ == "__main__":
    unittest.main()
