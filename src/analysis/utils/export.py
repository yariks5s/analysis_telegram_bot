"""
Export utilities for CryptoBot.

This module contains functions for exporting signal data in various formats.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

def signal_data_to_json(signals: List[Dict[str, Any]], filepath: Optional[str] = None) -> str:
    """
    Convert signal data to JSON format and optionally save to a file.

    Args:
        signals: List of signal history records
        filepath: Optional path to save the JSON file

    Returns:
        JSON string of the signal data
    """
    signals_copy = []
    
    for signal in signals:
        signal_copy = signal.copy()
        
        if "reasons" in signal_copy and isinstance(signal_copy["reasons"], str):
            try:
                signal_copy["reasons"] = json.loads(signal_copy["reasons"])
            except json.JSONDecodeError:
                pass
                
        if "market_conditions" in signal_copy and isinstance(signal_copy["market_conditions"], str):
            try:
                signal_copy["market_conditions"] = json.loads(signal_copy["market_conditions"])
            except json.JSONDecodeError:
                pass
        
        signals_copy.append(signal_copy)
    
    json_data = json.dumps(signals_copy, indent=2, default=str)
    
    if filepath:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w') as json_file:
            json_file.write(json_data)
    
    return json_data

def signal_data_to_csv(signals: List[Dict[str, Any]], filepath: Optional[str] = None) -> str:
    """
    Convert signal data to CSV format and optionally save to a file.

    Args:
        signals: List of signal history records
        filepath: Optional path to save the CSV file

    Returns:
        CSV string of the signal data
    """
    if not signals:
        return ""
    
    df = pd.DataFrame(signals)
    
    for idx, signal in enumerate(signals):
        if "reasons" in signal:
            if isinstance(signal["reasons"], str):
                try:
                    df.at[idx, "reasons"] = json.dumps(json.loads(signal["reasons"]))
                except json.JSONDecodeError:
                    df.at[idx, "reasons"] = signal["reasons"]
            else:
                df.at[idx, "reasons"] = json.dumps(signal["reasons"])
                
        if "market_conditions" in signal:
            if isinstance(signal["market_conditions"], str):
                try:
                    df.at[idx, "market_conditions"] = json.dumps(json.loads(signal["market_conditions"]))
                except json.JSONDecodeError:
                    df.at[idx, "market_conditions"] = signal["market_conditions"]
            else:
                df.at[idx, "market_conditions"] = json.dumps(signal["market_conditions"])
    
    csv_data = df.to_csv(index=False)
    
    if filepath:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w') as csv_file:
            csv_file.write(csv_data)
    
    return csv_data

def generate_export_filename(user_id: int, format_type: str, currency_pair: Optional[str] = None) -> str:
    """
    Generate a filename for the exported data.

    Args:
        user_id: Telegram user ID
        format_type: File format ('csv' or 'json')
        currency_pair: Optional currency pair to include in filename

    Returns:
        Filename string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    currency_str = f"_{currency_pair}" if currency_pair else ""
    return f"signal_history_user_{user_id}{currency_str}_{timestamp}.{format_type.lower()}"

def export_user_signals(signals: List[Dict[str, Any]], user_id: int, format_type: str, 
                   export_dir: str, currency_pair: str = None, auto_cleanup: bool = False) -> str:
    """
    Export user signals to a file.

    Args:
        signals: List of signal history records
        user_id: Telegram user ID
        format_type: Export format ('csv' or 'json')
        export_dir: Directory to save the export
        currency_pair: Optional currency pair filter
        auto_cleanup: If True, register the file for deletion when the program exits

    Returns:
        Path to the exported file
    """
    if not signals:
        return ""
    
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    
    filename = generate_export_filename(user_id, format_type, currency_pair)
    filepath = os.path.join(export_dir, filename)
    
    if format_type.lower() == "json":
        signal_data_to_json(signals, filepath)
    else:
        signal_data_to_csv(signals, filepath)
    
    # Register file for cleanup on program exit if requested
    if auto_cleanup and os.path.exists(filepath):
        import atexit
        atexit.register(lambda file_path: os.remove(file_path) if os.path.exists(file_path) else None, filepath)
    
    return filepath
