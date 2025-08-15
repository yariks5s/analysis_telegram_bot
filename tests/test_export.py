#!/usr/bin/env python3
"""
Test script for validating signal export functionality.
"""

import sys
import os
import json
import csv
import unittest
import tempfile
from unittest.mock import patch, mock_open
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.utils.export import (
    signal_data_to_json,
    signal_data_to_csv,
    generate_export_filename,
    export_user_signals
)


class TestExportFunctionality(unittest.TestCase):
    """Test case for signal export functionality"""

    def setUp(self):
        """Set up test environment"""
        self.user_id = 12345
        self.test_signals = [
            {
                "id": 1,
                "user_id": self.user_id,
                "currency_pair": "BTC/USDT",
                "signal_type": "Bullish",
                "timestamp": datetime.now().isoformat(),
                "entry_price": 50000.0,
                "stop_loss": 48000.0,
                "take_profit": 52000.0,
                "take_profit_1": 52000.0,
                "probability": 0.75,
                "reasons": json.dumps(["RSI oversold", "Price above support"]),
                "market_conditions": json.dumps({"volume_ratio": 1.2, "volatility": 0.05}),
            },
            {
                "id": 2,
                "user_id": self.user_id,
                "currency_pair": "ETH/USDT",
                "signal_type": "Bearish",
                "timestamp": datetime.now().isoformat(),
                "entry_price": 3000.0,
                "stop_loss": 3200.0,
                "take_profit": 2800.0,
                "take_profit_1": 2800.0,
                "probability": 0.8,
                "reasons": json.dumps(["RSI overbought", "Resistance rejection"]),
                "market_conditions": json.dumps({"volume_ratio": 0.9, "volatility": 0.07}),
            }
        ]

        self.test_signals_parsed = [
            {
                "id": 1,
                "user_id": self.user_id,
                "currency_pair": "BTC/USDT",
                "signal_type": "Bullish",
                "timestamp": datetime.now().isoformat(),
                "entry_price": 50000.0,
                "stop_loss": 48000.0,
                "take_profit": 52000.0,
                "take_profit_1": 52000.0,
                "probability": 0.75,
                "reasons": ["RSI oversold", "Price above support"],
                "market_conditions": {"volume_ratio": 1.2, "volatility": 0.05},
            }
        ]

        self.test_export_dir = tempfile.mkdtemp()

    def test_signal_data_to_json(self):
        """Test converting signal data to JSON"""
        json_data = signal_data_to_json(self.test_signals)
        parsed_data = json.loads(json_data)
        
        self.assertEqual(len(parsed_data), 2)
        self.assertEqual(parsed_data[0]["currency_pair"], "BTC/USDT")
        self.assertEqual(parsed_data[1]["currency_pair"], "ETH/USDT")
        
        self.assertIsInstance(parsed_data[0]["reasons"], list)
        self.assertEqual(parsed_data[0]["reasons"][0], "RSI oversold")
        
        self.assertIsInstance(parsed_data[0]["market_conditions"], dict)
        self.assertEqual(parsed_data[0]["market_conditions"]["volume_ratio"], 1.2)
        json_data = signal_data_to_json(self.test_signals_parsed)
        parsed_data = json.loads(json_data)
        
        self.assertEqual(len(parsed_data), 1)
        self.assertIsInstance(parsed_data[0]["reasons"], list)
        self.assertIsInstance(parsed_data[0]["market_conditions"], dict)

    def test_signal_data_to_json_with_file(self):
        """Test JSON export with file writing"""
        with patch("builtins.open", mock_open()) as mock_file:
            filepath = "/tmp/test_export.json"
            json_data = signal_data_to_json(self.test_signals, filepath)
            
            mock_file.assert_called_once_with(filepath, 'w')
            mock_file().write.assert_called_once_with(json_data)

    def test_signal_data_to_csv(self):
        """Test converting signal data to CSV"""
        csv_data = signal_data_to_csv(self.test_signals)
        
        self.assertIn("currency_pair", csv_data)
        self.assertIn("signal_type", csv_data)
        self.assertIn("entry_price", csv_data)
        
        self.assertIn("BTC/USDT", csv_data)
        self.assertIn("ETH/USDT", csv_data)
        
        lines = csv_data.strip().split("\n")
        reader = csv.DictReader(lines)
        rows = list(reader)
        
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["currency_pair"], "BTC/USDT")
        self.assertEqual(rows[1]["currency_pair"], "ETH/USDT")

    def test_signal_data_to_csv_with_file(self):
        """Test CSV export with file writing"""
        with patch("builtins.open", mock_open()) as mock_file:
            filepath = "/tmp/test_export.csv"
            csv_data = signal_data_to_csv(self.test_signals, filepath)
            
            mock_file.assert_called_once_with(filepath, 'w')
            mock_file().write.assert_called_once_with(csv_data)

    def test_generate_export_filename(self):
        """Test generating export filenames"""
        filename = generate_export_filename(self.user_id, "csv", "BTC/USDT")
        self.assertTrue(filename.startswith(f"signal_history_user_{self.user_id}_BTC/USDT_"))
        self.assertTrue(filename.endswith(".csv"))
        
        filename = generate_export_filename(self.user_id, "json")
        self.assertTrue(filename.startswith(f"signal_history_user_{self.user_id}_"))
        self.assertTrue(filename.endswith(".json"))
        self.assertNotIn("None", filename)

    def test_export_user_signals(self):
        """Test the full export functionality"""
        filepath = export_user_signals(
            self.test_signals, 
            self.user_id, 
            "csv", 
            self.test_export_dir,
            "BTC/USDT"
        )
        
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith(".csv"))
        
        with open(filepath, 'r') as f:
            content = f.read()
            self.assertIn("BTC/USDT", content)
            
        os.remove(filepath)
        
        filepath = export_user_signals(
            self.test_signals, 
            self.user_id, 
            "json", 
            self.test_export_dir
        )
        
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith(".json"))
        
        with open(filepath, 'r') as f:
            content = json.load(f)
            self.assertEqual(len(content), 2)
            self.assertEqual(content[0]["currency_pair"], "BTC/USDT")
            self.assertEqual(content[1]["currency_pair"], "ETH/USDT")
            
        os.remove(filepath)

    def test_empty_signals(self):
        """Test behavior with empty signals list"""
        json_data = signal_data_to_json([])
        self.assertEqual(json_data, "[]")
        
        csv_data = signal_data_to_csv([])
        self.assertEqual(csv_data, "")
        
        filepath = export_user_signals(
            [], 
            self.user_id, 
            "csv", 
            self.test_export_dir
        )
        self.assertEqual(filepath, "")

    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.test_export_dir):
            try:
                os.rmdir(self.test_export_dir)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
