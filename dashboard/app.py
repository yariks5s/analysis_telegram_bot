#!/usr/bin/env python3
"""
Adaptive Learning Dashboard - Flask API Backend

Provides endpoints for:
- Fetching trade history and signal data from ALL sources
- Fetching candlestick data with indicators
- Real-time data updates
"""

import os
import sys
import json
import csv
import glob
import uuid
from datetime import datetime, timedelta, timezone
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Add project root to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_dir)

from src.api.data_fetcher import fetch_candles, analyze_data
from src.telegram.signals.detection import (
    calculate_rsi,
    calculate_atr,
    detect_support_resistance,
    detect_market_regime,
    analyze_order_blocks,
    analyze_breaker_blocks,
)
from utils import create_true_preferences

app = Flask(__name__, static_folder='static')
CORS(app)

# Data paths
SIGNAL_HISTORY_PATH = os.path.join(project_dir, "back_tester/data/btc_eth_learning/signals/signal_history.json")
REALTIME_DATA_DIR = os.path.join(project_dir, "back_tester/data")
REPORTS_DIRS = [
    os.path.join(project_dir, "reports"),
    os.path.join(project_dir, "back_tester/reports"),
]
WEIGHTS_PATH = os.path.join(project_dir, "back_tester/data/btc_eth_learning/final_weights.json")

# Cache for all trades (loaded once and refreshed on demand)
_all_trades_cache = None
_cache_timestamp = None


def load_signal_history():
    """Load signal history from JSON file"""
    signals = []
    try:
        with open(SIGNAL_HISTORY_PATH, 'r') as f:
            data = json.load(f)
            for s in data.get('signals', []):
                s['source'] = 'adaptive_learning'
                s['source_file'] = 'signal_history.json'
                signals.append(s)
    except Exception as e:
        print(f"Error loading signal history: {e}")
    return signals


def load_realtime_tests():
    """Load all realtime test result files"""
    tests = []
    try:
        for filename in os.listdir(REALTIME_DATA_DIR):
            if filename.startswith('realtime_test_') and filename.endswith('.json'):
                filepath = os.path.join(REALTIME_DATA_DIR, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    data['filename'] = filename
                    tests.append(data)
    except Exception as e:
        print(f"Error loading realtime tests: {e}")
    return sorted(tests, key=lambda x: x.get('timestamp', ''), reverse=True)


def load_realtime_trades():
    """Load trades from all realtime test files"""
    trades = []
    try:
        for filename in os.listdir(REALTIME_DATA_DIR):
            if filename.startswith('realtime_test_') and filename.endswith('.json'):
                filepath = os.path.join(REALTIME_DATA_DIR, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    for trade in data.get('trades', []):
                        # Convert realtime trade format to signal format
                        signal = {
                            'signal_id': str(uuid.uuid4()),
                            'timestamp': trade.get('entry_time', data.get('timestamp')),
                            'symbol': trade.get('symbol', 'UNKNOWN'),
                            'interval': trade.get('interval', '1h'),
                            'signal_type': 'Bullish',  # Realtime only does long trades
                            'entry_price': trade.get('entry_price', 0),
                            'exit_price': trade.get('exit_price', 0),
                            'stop_loss': trade.get('entry_price', 0) * 0.99,  # Approximate
                            'take_profit_1': trade.get('entry_price', 0) * 1.01,
                            'take_profit_2': trade.get('entry_price', 0) * 1.02,
                            'take_profit_3': trade.get('entry_price', 0) * 1.03,
                            'profit_loss': trade.get('profit_loss', 0),
                            'profit_loss_percent': trade.get('profit_percent', 0),
                            'duration_candles': int(trade.get('duration_minutes', 0) / 60),
                            'outcome': trade.get('exit_type', 'unknown'),
                            'source': 'realtime_test',
                            'source_file': filename,
                            'indicator_contributions': {},
                            'reasons': [],
                        }
                        trades.append(signal)
    except Exception as e:
        print(f"Error loading realtime trades: {e}")
    return trades


def load_backtest_trades():
    """Load trades from all backtesting report CSV files"""
    trades = []
    try:
        # Find all trades.csv files in all reports directories
        all_csv_files = []
        for reports_dir in REPORTS_DIRS:
            pattern = os.path.join(reports_dir, "**/*_trades.csv")
            all_csv_files.extend(glob.glob(pattern, recursive=True))
        
        print(f"Found {len(all_csv_files)} trade CSV files")
        
        for csv_path in all_csv_files:
            try:
                # Extract symbol and interval from path
                folder_name = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
                parts = folder_name.split('_')
                if len(parts) >= 2:
                    symbol = parts[0]
                    interval = parts[1]
                else:
                    symbol = 'UNKNOWN'
                    interval = '1h'
                
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    current_entry = None
                    
                    for row in reader:
                        trade_type = row.get('type', '')
                        
                        if trade_type == 'entry':
                            # Start a new trade - convert timestamp to ISO format
                            raw_ts = row.get('timestamp', '')
                            # Convert "2023-11-26 07:00:00" to "2023-11-26T07:00:00"
                            iso_ts = raw_ts.replace(' ', 'T') if raw_ts else ''
                            
                            current_entry = {
                                'signal_id': str(uuid.uuid4()),
                                'timestamp': iso_ts,
                                'symbol': symbol,
                                'interval': interval,
                                'signal_type': 'Bullish' if 'Bullish' in row.get('signal', '') else 'Bearish',
                                'entry_price': float(row.get('price', 0) or 0),
                                'stop_loss': float(row.get('stop_loss', 0) or 0),
                                'take_profit_1': float(row.get('take_profit_1', 0) or 0),
                                'take_profit_2': float(row.get('take_profit_2', 0) or 0),
                                'take_profit_3': float(row.get('take_profit_3', 0) or 0),
                                'source': 'backtest',
                                'source_file': os.path.basename(csv_path),
                                'indicator_contributions': {},
                                'reasons': [row.get('signal', '')],
                            }
                        elif trade_type in ['stop_loss', 'take_profit_1', 'take_profit_2', 'take_profit_3', 'trailing_stop'] and current_entry:
                            # Complete the trade
                            exit_price = float(row.get('price', 0) or 0)
                            profit = float(row.get('profit', 0) or 0)
                            
                            current_entry['exit_price'] = exit_price
                            current_entry['profit_loss'] = profit
                            current_entry['profit_loss_percent'] = (exit_price - current_entry['entry_price']) / current_entry['entry_price'] * 100 if current_entry['entry_price'] > 0 else 0
                            current_entry['outcome'] = trade_type
                            # Convert exit timestamp to ISO format
                            exit_ts = row.get('timestamp', '')
                            current_entry['exit_time'] = exit_ts.replace(' ', 'T') if exit_ts else ''
                            
                            # Calculate duration in candles (approximate)
                            try:
                                entry_idx = int(current_entry.get('entry_index', 0) or 0)
                                exit_idx = int(row.get('index', 0) or 0)
                                current_entry['duration_candles'] = exit_idx - entry_idx
                            except:
                                current_entry['duration_candles'] = 0
                            
                            trades.append(current_entry)
                            current_entry = None
                            
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
                
    except Exception as e:
        print(f"Error loading backtest trades: {e}")
    
    return trades


def load_all_trades(force_refresh=False):
    """Load and combine trades from all sources"""
    global _all_trades_cache, _cache_timestamp
    
    # Use cache if available and less than 5 minutes old
    if not force_refresh and _all_trades_cache is not None and _cache_timestamp is not None:
        if (datetime.now() - _cache_timestamp).total_seconds() < 300:
            return _all_trades_cache
    
    all_trades = []
    
    # Load from each source
    print("Loading trades from all sources...")
    
    # 1. Signal history (adaptive learning)
    signal_trades = load_signal_history()
    print(f"  - Adaptive learning signals: {len(signal_trades)}")
    all_trades.extend(signal_trades)
    
    # 2. Realtime tests
    realtime_trades = load_realtime_trades()
    print(f"  - Realtime test trades: {len(realtime_trades)}")
    all_trades.extend(realtime_trades)
    
    # 3. Backtest reports
    backtest_trades = load_backtest_trades()
    print(f"  - Backtest trades: {len(backtest_trades)}")
    all_trades.extend(backtest_trades)
    
    print(f"Total trades loaded: {len(all_trades)}")
    
    # Update cache
    _all_trades_cache = all_trades
    _cache_timestamp = datetime.now()
    
    return all_trades


def load_weights():
    """Load current weights"""
    try:
        with open(WEIGHTS_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return {}


@app.route('/')
def index():
    """Serve the main dashboard page"""
    return send_from_directory('static', 'index.html')


@app.route('/api/signals')
def get_signals():
    """Get all signals with pagination from all data sources"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    symbol = request.args.get('symbol', None)
    interval = request.args.get('interval', None)
    outcome = request.args.get('outcome', None)
    source = request.args.get('source', None)  # New: filter by data source
    refresh = request.args.get('refresh', 'false').lower() == 'true'
    
    # Load all trades from all sources
    signals = load_all_trades(force_refresh=refresh)
    
    # Filter signals
    if symbol:
        signals = [s for s in signals if s.get('symbol') == symbol]
    if interval:
        signals = [s for s in signals if s.get('interval') == interval]
    if outcome:
        signals = [s for s in signals if s.get('outcome') == outcome]
    if source:
        signals = [s for s in signals if s.get('source') == source]
    
    # Sort by timestamp (most recent first)
    signals = sorted(signals, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    # Paginate
    total = len(signals)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_signals = signals[start:end]
    
    return jsonify({
        'signals': paginated_signals,
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': (total + per_page - 1) // per_page
    })


@app.route('/api/signals/<signal_id>')
def get_signal(signal_id):
    """Get a specific signal by ID from all sources"""
    signals = load_all_trades()
    signal = next((s for s in signals if s.get('signal_id') == signal_id), None)
    
    if signal:
        return jsonify(signal)
    return jsonify({'error': 'Signal not found'}), 404


@app.route('/api/candles')
def get_candles():
    """
    Fetch candlestick data with indicators
    
    Query params:
    - symbol: Trading pair (e.g., BTCUSDT)
    - interval: Candle interval (e.g., 1h, 15m)
    - limit: Number of candles (default 200)
    - end_time: Optional end timestamp (for fetching historical data)
    """
    symbol = request.args.get('symbol', 'BTCUSDT')
    interval = request.args.get('interval', '1h')
    limit = request.args.get('limit', 200, type=int)
    end_time = request.args.get('end_time', None)
    
    try:
        if end_time:
            timestamp = datetime.fromisoformat(end_time.replace('Z', '+00:00')).timestamp()
        else:
            timestamp = datetime.now(timezone.utc).timestamp()
        
        # Fetch candles
        df = fetch_candles(symbol, limit, interval, timestamp)
        
        if df is None or df.empty:
            return jsonify({'error': 'Failed to fetch candles'}), 500
        
        # Calculate indicators
        preferences = create_true_preferences()
        indicators = analyze_data(df, preferences, 0.05)
        
        # Calculate additional indicators for the chart
        rsi = calculate_rsi(df['Close']).fillna(50).tolist()
        atr = calculate_atr(df).fillna(0).tolist()
        
        # Detect support/resistance levels
        support_levels, resistance_levels = detect_support_resistance(df)
        
        # Detect order blocks and breaker blocks
        order_blocks = analyze_order_blocks(df)
        breaker_blocks = analyze_breaker_blocks(df)
        
        # Market regime
        market_regime = detect_market_regime(df)
        
        # Format candles for response
        candles = []
        for i, (idx, row) in enumerate(df.iterrows()):
            candle = {
                'time': idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                'timestamp': int(idx.timestamp() * 1000) if hasattr(idx, 'timestamp') else 0,
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': float(row['Volume']),
                'rsi': rsi[i] if i < len(rsi) else 50,
                'atr': atr[i] if i < len(atr) else 0,
            }
            candles.append(candle)
        
        # Format indicators
        indicator_data = {
            'order_blocks': [],
            'breaker_blocks': [],
            'fvgs': [],
            'liquidity_levels': [],
            'liquidity_pools': [],
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'market_regime': market_regime.value if hasattr(market_regime, 'value') else str(market_regime),
        }
        
        # Add order blocks
        for ob in order_blocks:
            indicator_data['order_blocks'].append({
                'type': ob['type'],
                'index': ob['index'],
                'price': ob['price'],
                'strength': ob.get('strength', 1.0),
            })
        
        # Add breaker blocks
        for bb in breaker_blocks:
            indicator_data['breaker_blocks'].append({
                'type': bb['type'],
                'index': bb['index'],
                'price': bb['price'],
                'strength': bb.get('strength', 1.0),
            })
        
        # Add FVGs if available
        if indicators and hasattr(indicators, 'fvgs') and indicators.fvgs and hasattr(indicators.fvgs, 'list'):
            for fvg in indicators.fvgs.list:
                indicator_data['fvgs'].append({
                    'start_index': getattr(fvg, 'start_index', 0),
                    'end_index': getattr(fvg, 'end_index', 0),
                    'start_price': getattr(fvg, 'start_price', 0),
                    'end_price': getattr(fvg, 'end_price', 0),
                    'type': getattr(fvg, 'type', 'unknown'),
                })
        
        # Add liquidity levels
        if indicators and hasattr(indicators, 'liquidity_levels') and indicators.liquidity_levels and hasattr(indicators.liquidity_levels, 'list'):
            for ll in indicators.liquidity_levels.list:
                indicator_data['liquidity_levels'].append({
                    'price': getattr(ll, 'price', 0),
                    'strength': getattr(ll, 'strength', 1.0),
                    'type': getattr(ll, 'type', 'unknown'),
                })
        
        # Add liquidity pools
        if indicators and hasattr(indicators, 'liquidity_pools') and indicators.liquidity_pools and hasattr(indicators.liquidity_pools, 'list'):
            for lp in indicators.liquidity_pools.list:
                indicator_data['liquidity_pools'].append({
                    'price': getattr(lp, 'price', 0),
                    'volume': getattr(lp, 'volume', 0),
                    'strength': getattr(lp, 'strength', 1.0),
                })
        
        return jsonify({
            'symbol': symbol,
            'interval': interval,
            'candles': candles,
            'indicators': indicator_data,
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/candles/for-signal/<signal_id>')
def get_candles_for_signal(signal_id):
    """
    Fetch candles for displaying a signal's trade.
    
    Since backtesting signals don't have actual historical timestamps,
    we fetch recent candles to visualize the trade pattern.
    
    Query params:
    - before: Number of candles before entry (default 150)
    - after: Number of candles after exit (default 20)
    """
    before = request.args.get('before', 150, type=int)
    after = request.args.get('after', 20, type=int)
    
    # Search in all trades (not just signal_history)
    all_signals = load_all_trades()
    signal = next((s for s in all_signals if s.get('signal_id') == signal_id), None)
    
    if not signal:
        return jsonify({'error': 'Signal not found'}), 404
    
    symbol = signal.get('symbol', 'BTCUSDT')
    interval = signal.get('interval', '1h')
    duration = signal.get('duration_candles', 10) or 10
    
    # Calculate total candles needed
    total_candles = before + duration + after
    
    try:
        # Fetch recent candles (we can't match historical backtesting data,
        # so we just show recent market data with trade markers)
        df = fetch_candles(symbol, total_candles, interval)
        
        if df is None or df.empty:
            return jsonify({'error': 'Failed to fetch candles'}), 500
        
        # Calculate indicators
        preferences = create_true_preferences()
        indicators = analyze_data(df, preferences, 0.05)
        
        # Calculate additional indicators
        rsi = calculate_rsi(df['Close']).fillna(50).tolist()
        atr = calculate_atr(df).fillna(0).tolist()
        support_levels, resistance_levels = detect_support_resistance(df)
        order_blocks = analyze_order_blocks(df)
        breaker_blocks = analyze_breaker_blocks(df)
        market_regime = detect_market_regime(df)
        
        # Format candles
        candles = []
        for i, (idx, row) in enumerate(df.iterrows()):
            candle = {
                'time': idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                'timestamp': int(idx.timestamp() * 1000) if hasattr(idx, 'timestamp') else 0,
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': float(row['Volume']),
                'rsi': rsi[i] if i < len(rsi) else 50,
                'atr': atr[i] if i < len(atr) else 0,
            }
            candles.append(candle)
        
        # Format indicators
        indicator_data = {
            'order_blocks': [{'type': ob['type'], 'index': ob['index'], 'price': ob['price'], 'strength': ob.get('strength', 1.0)} for ob in order_blocks],
            'breaker_blocks': [{'type': bb['type'], 'index': bb['index'], 'price': bb['price'], 'strength': bb.get('strength', 1.0)} for bb in breaker_blocks],
            'fvgs': [],
            'liquidity_levels': [],
            'liquidity_pools': [],
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'market_regime': market_regime.value if hasattr(market_regime, 'value') else str(market_regime),
        }
        
        # Add FVGs if available
        if indicators and hasattr(indicators, 'fvgs') and indicators.fvgs and hasattr(indicators.fvgs, 'list'):
            for fvg in indicators.fvgs.list:
                indicator_data['fvgs'].append({
                    'start_index': getattr(fvg, 'start_index', 0),
                    'end_index': getattr(fvg, 'end_index', 0),
                    'start_price': getattr(fvg, 'start_price', 0),
                    'end_price': getattr(fvg, 'end_price', 0),
                    'type': getattr(fvg, 'type', 'unknown'),
                })
        
        return jsonify({
            'symbol': symbol,
            'interval': interval,
            'candles': candles,
            'indicators': indicator_data,
            'signal': signal,
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/realtime-tests')
def get_realtime_tests():
    """Get all realtime test results"""
    tests = load_realtime_tests()
    return jsonify({'tests': tests})


@app.route('/api/weights')
def get_weights():
    """Get current weights and their names"""
    weights_data = load_weights()
    return jsonify(weights_data)


@app.route('/api/stats')
def get_stats():
    """Get aggregate statistics from all data sources"""
    signals = load_all_trades()
    
    # Calculate stats
    total_signals = len(signals)
    outcomes = {}
    symbols = {}
    intervals = {}
    sources = {}
    total_pnl = 0
    
    for s in signals:
        # Count outcomes
        outcome = s.get('outcome', 'unknown')
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        # Count by symbol
        symbol = s.get('symbol', 'unknown')
        if symbol not in symbols:
            symbols[symbol] = {'count': 0, 'pnl': 0, 'wins': 0, 'losses': 0}
        symbols[symbol]['count'] += 1
        symbols[symbol]['pnl'] += s.get('profit_loss', 0)
        if s.get('profit_loss', 0) > 0:
            symbols[symbol]['wins'] += 1
        elif s.get('profit_loss', 0) < 0:
            symbols[symbol]['losses'] += 1
        
        # Count by interval
        interval = s.get('interval', 'unknown')
        if interval not in intervals:
            intervals[interval] = {'count': 0, 'pnl': 0, 'wins': 0, 'losses': 0}
        intervals[interval]['count'] += 1
        intervals[interval]['pnl'] += s.get('profit_loss', 0)
        if s.get('profit_loss', 0) > 0:
            intervals[interval]['wins'] += 1
        elif s.get('profit_loss', 0) < 0:
            intervals[interval]['losses'] += 1
        
        # Count by source
        source = s.get('source', 'unknown')
        if source not in sources:
            sources[source] = {'count': 0, 'pnl': 0, 'wins': 0, 'losses': 0}
        sources[source]['count'] += 1
        sources[source]['pnl'] += s.get('profit_loss', 0)
        if s.get('profit_loss', 0) > 0:
            sources[source]['wins'] += 1
        elif s.get('profit_loss', 0) < 0:
            sources[source]['losses'] += 1
        
        total_pnl += s.get('profit_loss', 0)
    
    return jsonify({
        'total_signals': total_signals,
        'total_pnl': total_pnl,
        'outcomes': outcomes,
        'by_symbol': symbols,
        'by_interval': intervals,
        'by_source': sources,
    })


@app.route('/api/symbols')
def get_symbols():
    """Get list of available symbols from all sources"""
    signals = load_all_trades()
    symbols = list(set(s.get('symbol', 'BTCUSDT') for s in signals if s.get('symbol')))
    return jsonify({'symbols': sorted(symbols)})


@app.route('/api/intervals')
def get_intervals():
    """Get list of available intervals from all sources"""
    signals = load_all_trades()
    intervals = list(set(s.get('interval', '1h') for s in signals if s.get('interval')))
    # Sort intervals by duration
    interval_order = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    intervals = sorted(intervals, key=lambda x: interval_order.index(x) if x in interval_order else 999)
    return jsonify({'intervals': intervals})


@app.route('/api/sources')
def get_sources():
    """Get list of available data sources"""
    signals = load_all_trades()
    sources = list(set(s.get('source', 'unknown') for s in signals if s.get('source')))
    source_info = {
        'adaptive_learning': 'Adaptive Learning Signals',
        'realtime_test': 'Realtime Test Trades',
        'backtest': 'Backtesting Reports'
    }
    return jsonify({
        'sources': [{'id': s, 'name': source_info.get(s, s)} for s in sorted(sources)]
    })


if __name__ == '__main__':
    print("Starting Adaptive Learning Dashboard...")
    print(f"Signal history path: {SIGNAL_HISTORY_PATH}")
    print(f"Realtime data dir: {REALTIME_DATA_DIR}")
    app.run(host='0.0.0.0', port=5050, debug=True)

