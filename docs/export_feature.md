# Signal Data Export Feature

## Overview

The Signal Data Export feature allows users to download their trading signal history in CSV or JSON format. This provides users with the ability to further analyze their signal data in spreadsheet applications, data analysis tools, or integrate with other systems.

## Supported Formats

- **CSV**
- **JSON**

## How to Use

1. Use the `/signal_history` command to view your trading signal history
2. The bot will display your recent signals with a set of filtering options
3. To export your data, use one of the export buttons at the bottom of the message:
   - **Export CSV**: Download your signal history in CSV format
   - **Export JSON**: Download your signal history in JSON format

## Export Options

- **Export All Signals**: When viewing the complete signal history, you can export all signals across all currency pairs
- **Export Filtered Signals**: When viewing signals for a specific currency pair or time period, you can export only those filtered signals

## File Contents

Exported files contain the following information for each signal:

- `id`: Unique identifier for the signal
- `user_id`: Your user identifier
- `currency_pair`: The trading pair (e.g., "BTC/USDT")
- `signal_type`: Type of signal (e.g., "Bullish", "Bearish")
- `timestamp`: Date and time when the signal was generated
- `entry_price`: Suggested entry price
- `stop_loss`: Suggested stop loss price
- `take_profit` / `take_profit_1`: Suggested take profit price
- `probability`: Signal confidence level
- `reasons`: List of technical reasons for the signal
- `market_conditions`: Market context information when the signal was generated

## Technical Notes

- Exports are generated on-demand and sent as downloadable files
- Large history exports may take a few seconds to prepare
- Files are temporarily stored on the server and automatically deleted after sending
