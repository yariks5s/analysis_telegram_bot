# CryptoBot Adaptive Learning Dashboard

A web-based dashboard for visualizing and analyzing the adaptive learning trading system's performance.

## Features

- **Interactive Candlestick Charts**: View price action with TradingView's Lightweight Charts
- **Signal Visualization**: See all trading signals with entry/exit markers
- **Indicator Overlays**: Toggle visibility of:
  - Order Blocks (bullish/bearish)
  - Breaker Blocks
  - Support/Resistance levels
  - Fair Value Gaps (FVG)
  - Trade levels (Entry, SL, TP1-3)
- **Trade History**: Browse and filter all historical signals
- **Performance Metrics**: View aggregate statistics and P/L
- **Flexible Candle Loading**: Request more candles before a trade to see context

## Quick Start

### Option 1: Using the startup script

```bash
cd dashboard
./start_dashboard.sh
```

### Option 2: Manual start

```bash
# Activate virtual environment
source ../venv/bin/activate

# Install dependencies (if needed)
pip install flask flask-cors

# Start the server
python app.py
```

Then open http://localhost:5050 in your browser.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/signals` | List all signals with pagination |
| `GET /api/signals/<id>` | Get a specific signal by ID |
| `GET /api/candles` | Fetch candlestick data with indicators |
| `GET /api/candles/for-signal/<id>` | Fetch candles around a specific signal |
| `GET /api/stats` | Get aggregate statistics |
| `GET /api/realtime-tests` | List all realtime test results |
| `GET /api/weights` | Get current optimized weights |
| `GET /api/symbols` | List available symbols |
| `GET /api/intervals` | List available intervals |

### Query Parameters

**GET /api/signals**
- `page` - Page number (default: 1)
- `per_page` - Items per page (default: 50)
- `symbol` - Filter by symbol (e.g., BTCUSDT)
- `interval` - Filter by interval (e.g., 1h)
- `outcome` - Filter by outcome (tp1_hit, tp2_hit, tp3_hit, stop_loss, etc.)

**GET /api/candles**
- `symbol` - Trading pair (default: BTCUSDT)
- `interval` - Candle interval (default: 1h)
- `limit` - Number of candles (default: 200)
- `end_time` - Optional end timestamp for historical data

**GET /api/candles/for-signal/<id>**
- `before` - Candles before entry (default: 50)
- `after` - Candles after exit (default: 20)

## Usage Tips

1. **Select a Signal**: Click on any signal in the left sidebar to load its chart
2. **Load More Context**: Use "Candles Before Trade" to see more price action before the entry
3. **Toggle Indicators**: Click indicator chips to show/hide different overlays
4. **Zoom & Pan**: Use mouse scroll to zoom, drag to pan the chart
5. **Fit Chart**: Click "Fit" button to auto-fit the chart to all data

## Data Sources

The dashboard reads from:
- `back_tester/data/btc_eth_learning/signals/signal_history.json` - Signal history
- `back_tester/data/btc_eth_learning/final_weights.json` - Optimized weights
- `back_tester/data/realtime_test_*.json` - Realtime test results

Live candle data is fetched from the Bybit API.

## Architecture

```
dashboard/
├── app.py           # Flask backend API
├── static/
│   └── index.html   # Frontend (single-page app)
├── requirements.txt # Python dependencies
├── start_dashboard.sh # Startup script
└── README.md        # This file
```

## Troubleshooting

**Dashboard won't start**
- Make sure you're in the project's virtual environment
- Check that Flask is installed: `pip install flask flask-cors`

**No signals showing**
- Verify signal_history.json exists in the correct path
- Check the Flask console for any errors

**Chart not loading**
- Check browser console for JavaScript errors
- Ensure you have internet access (for fetching live candle data)




