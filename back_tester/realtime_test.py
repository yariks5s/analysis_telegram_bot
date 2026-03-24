#!/usr/bin/env python3
"""
Real-Time Trading Strategy Tester

This script monitors multiple symbol/interval combinations in real-time,
generates signals using the optimized weights from backtesting, and tracks
open positions with stop-loss and take-profit management.

Usage:
    python realtime_test.py [--paper] [--duration HOURS]
"""

import os
import sys
import json
import time
import signal
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from threading import Thread, Event
import uuid

# Add project root to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_dir)

import pandas as pd
import numpy as np

# Import from the project
from src.analysis.utils.helpers import fetch_candles
from src.telegram.signals.detection import (
    analyze_data,
    generate_price_prediction_signal_proba,
    TradingSignal,
    calculate_position_size,
)
from src.core.utils import create_true_preferences
import requests

# Try to import indicator contribution tracking
try:
    from back_tester.adaptive_learning import extract_indicator_contributions
    INDICATOR_TRACKING_AVAILABLE = True
except ImportError:
    INDICATOR_TRACKING_AVAILABLE = False
    def extract_indicator_contributions(*args, **kwargs):
        return {}


def fetch_current_price(symbol: str) -> float:
    """Fetch real-time current price from Bybit ticker API"""
    try:
        url = f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("retCode") == 0:
                result = data.get("result", {}).get("list", [])
                if result:
                    return float(result[0].get("lastPrice", 0))
    except Exception as e:
        logger.debug(f"Error fetching ticker for {symbol}: {e}")
    return 0.0

# Try to import charting libraries
try:
    import mplfinance as mpf
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    CHARTING_AVAILABLE = True
except ImportError:
    CHARTING_AVAILABLE = False
    logger.warning("mplfinance not available - trade charts will not be generated")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)


@dataclass
class OpenPosition:
    """Represents an open trading position"""
    position_id: str
    symbol: str
    interval: str
    entry_price: float
    entry_time: datetime
    position_size: float
    stop_loss: float
    initial_stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    highest_price: float
    trailing_stop_active: bool = False
    trailing_stop_level: float = 0.0
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    initial_position: float = 0.0
    risk_reward_ratio: float = 2.0
    signal_reason: str = ""
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    confidence: float = 0.0  # Signal confidence level (0.0 - 1.0)
    indicator_contributions: Dict[str, float] = field(default_factory=dict)  # Which indicators triggered signal


@dataclass
class TradeResult:
    """Represents a completed trade"""
    position_id: str
    symbol: str
    interval: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    profit_loss: float
    profit_percent: float
    exit_type: str  # stop_loss, trailing_stop, tp1, tp2, tp3
    duration_minutes: float
    chart_path: str = ""  # Path to the trade chart image
    entry_index: int = 0
    exit_index: int = 0
    confidence: float = 0.0  # Signal confidence level at entry
    indicator_contributions: Dict[str, float] = field(default_factory=dict)  # Which indicators triggered signal


@dataclass
class CombinationStats:
    """Statistics for a symbol/interval combination"""
    symbol: str
    interval: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_profit: float = 0.0
    signals_generated: int = 0
    last_signal_time: Optional[datetime] = None


class RealTimeTrader:
    """Real-time trading strategy tester"""
    
    # Trading combinations to monitor (based on production recommendations)
    COMBINATIONS = [
        # Safe options (0% fail rate in backtesting)
        {"symbol": "BTCUSDT", "interval": "1h", "priority": "SAFE"},
        {"symbol": "ETHUSDT", "interval": "5m", "priority": "SAFE"},
        {"symbol": "ETHUSDT", "interval": "15m", "priority": "SAFE"},
        # Higher risk/reward options
        {"symbol": "BTCUSDT", "interval": "4h", "priority": "AGGRESSIVE"},
        {"symbol": "ETHUSDT", "interval": "4h", "priority": "AGGRESSIVE"},
        # Additional intervals for completeness
        {"symbol": "BTCUSDT", "interval": "5m", "priority": "EXTRA"},
        {"symbol": "BTCUSDT", "interval": "15m", "priority": "EXTRA"},
        {"symbol": "ETHUSDT", "interval": "1h", "priority": "EXTRA"},
    ]
    
    # Interval to seconds mapping
    INTERVAL_SECONDS = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }
    
    # Maximum percentage of balance to use per trade (prevents all-in trades)
    MAX_POSITION_PERCENT = 20.0  # Max 20% of balance per trade
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_percentage: float = 1.0,
        use_trailing_stop: bool = True,
        trailing_stop_distance: float = 0.5,
        weights_file: str = None,
        safe_only: bool = False,
        max_position_percent: float = 20.0,  # Max % of balance per trade
        candle_window: int = 500,  # Number of candles for analysis
        disable_worst_patterns: bool = False,
        worst_pattern_threshold: float = -100.0,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_percentage = risk_percentage
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_distance = trailing_stop_distance
        self.safe_only = safe_only
        self.max_position_percent = max_position_percent
        self.candle_window = candle_window
        self.disable_worst_patterns = disable_worst_patterns
        self.worst_pattern_threshold = worst_pattern_threshold
        
        # Load optimized weights
        self.weights = self._load_weights(weights_file)
        
        # Load disabled patterns if enabled
        self.disabled_patterns = set()
        if disable_worst_patterns:
            self._load_disabled_patterns(weights_file)
        
        # Trading state
        self.open_positions: Dict[str, OpenPosition] = {}  # key = position_id
        self.completed_trades: List[TradeResult] = []
        self.combination_stats: Dict[str, CombinationStats] = {}
        
        # Initialize stats for each combination
        for combo in self.COMBINATIONS:
            if safe_only and combo["priority"] != "SAFE":
                continue
            key = f"{combo['symbol']}_{combo['interval']}"
            self.combination_stats[key] = CombinationStats(
                symbol=combo['symbol'],
                interval=combo['interval']
            )
        
        # Control flags
        self.running = False
        self.stop_event = Event()
        
        # Preferences for signal generation
        self.preferences = create_true_preferences()
        
        # Track last data fetch time per combination
        self.last_fetch: Dict[str, datetime] = {}
        
        # Start time
        self.start_time = None
        
        # Track invested amount
        self.invested_amount = 0.0
        
        # Track indicator performance
        self.indicator_stats: Dict[str, Dict[str, Any]] = {}
        # Initialize stats for each indicator
        for indicator in [
            "W_BULLISH_OB", "W_BEARISH_OB", "W_BULLISH_BREAKER", "W_BEARISH_BREAKER",
            "W_ABOVE_SUPPORT", "W_BELOW_RESISTANCE", "W_FVG_ABOVE", "W_FVG_BELOW",
            "W_TREND", "W_SWEEP_HIGHS", "W_SWEEP_LOWS", "W_STRUCTURE_BREAK",
            "W_PIN_BAR", "W_ENGULFING", "W_LIQUIDITY_POOL_ABOVE", "W_LIQUIDITY_POOL_BELOW",
            "W_LIQUIDITY_POOL_ROUND", "W_RSI_EXTREME"
        ]:
            self.indicator_stats[indicator] = {
                "signals": 0,
                "wins": 0,
                "losses": 0,
                "total_profit": 0.0,
                "win_rate": 0.0
            }
    
    def get_total_equity(self) -> float:
        """Calculate total account equity (available + invested)"""
        return self.balance + self.invested_amount
    
    def get_unrealized_pnl(self, current_prices: Dict[str, float] = None) -> float:
        """Calculate unrealized P/L from open positions"""
        if not current_prices:
            return 0.0
        unrealized = 0.0
        for pos in self.open_positions.values():
            current = current_prices.get(pos.symbol, pos.entry_price)
            unrealized += pos.position_size * (current - pos.entry_price)
        return unrealized
    
    def _generate_trade_chart(self, trade: TradeResult, df: pd.DataFrame) -> str:
        """Generate a chart image for a completed trade"""
        if not CHARTING_AVAILABLE:
            return ""
        
        try:
            # Create charts directory
            charts_dir = os.path.join(os.path.dirname(__file__), "data", "trade_charts")
            os.makedirs(charts_dir, exist_ok=True)
            
            # Generate filename
            timestamp = trade.exit_time.strftime("%Y%m%d_%H%M%S")
            result = "WIN" if trade.profit_loss > 0 else "LOSS"
            filename = f"{trade.symbol}_{trade.interval}_{timestamp}_{result}.png"
            filepath = os.path.join(charts_dir, filename)
            
            # Prepare data for mplfinance
            # Make sure df has the right index and columns
            plot_df = df.copy()
            if not isinstance(plot_df.index, pd.DatetimeIndex):
                plot_df.index = pd.to_datetime(plot_df.index)
            
            # Calculate RSI for the chart
            delta = plot_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, 1e-10)
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(50)
            
            # Create entry/exit markers
            entry_idx = None
            exit_idx = None
            
            # Find closest indices to entry and exit times
            for i, idx in enumerate(plot_df.index):
                if entry_idx is None and idx >= trade.entry_time:
                    entry_idx = i
                if exit_idx is None and idx >= trade.exit_time:
                    exit_idx = i
                    break
            
            if entry_idx is None:
                entry_idx = 0
            if exit_idx is None:
                exit_idx = len(plot_df) - 1
            
            # Create markers for entry and exit
            entry_marker = [np.nan] * len(plot_df)
            exit_marker = [np.nan] * len(plot_df)
            entry_marker[entry_idx] = plot_df['Low'].iloc[entry_idx] * 0.998
            exit_marker[exit_idx] = plot_df['High'].iloc[exit_idx] * 1.002
            
            # Create horizontal lines for SL and TP levels
            sl_line = [trade.entry_price * (1 - 0.01)] * len(plot_df)  # Approximate SL
            tp_line = [trade.entry_price * (1 + 0.02)] * len(plot_df)  # Approximate TP
            
            # Add plots
            add_plots = [
                mpf.make_addplot(entry_marker, type='scatter', markersize=200, marker='^', color='green'),
                mpf.make_addplot(exit_marker, type='scatter', markersize=200, marker='v', 
                               color='green' if trade.profit_loss > 0 else 'red'),
                mpf.make_addplot(rsi, panel=1, color='purple', ylabel='RSI'),
                mpf.make_addplot([30] * len(plot_df), panel=1, color='gray', linestyle='--'),
                mpf.make_addplot([70] * len(plot_df), panel=1, color='gray', linestyle='--'),
            ]
            
            # Style
            mc = mpf.make_marketcolors(
                up='#26a69a', down='#ef5350',
                edge='inherit',
                wick='inherit',
                volume='in'
            )
            style = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle=':',
                gridcolor='gray',
                facecolor='#1e1e1e',
                figcolor='#1e1e1e',
                edgecolor='#333333'
            )
            
            # Create the chart
            fig, axes = mpf.plot(
                plot_df,
                type='candle',
                style=style,
                addplot=add_plots,
                volume=True,
                volume_panel=2,
                title=f"\n{trade.symbol} {trade.interval} - {result}: ${trade.profit_loss:+,.2f} ({trade.profit_percent:+.2f}%)",
                figsize=(14, 10),
                panel_ratios=(3, 1, 1),
                returnfig=True
            )
            
            # Add annotations
            axes[0].axhline(y=trade.entry_price, color='blue', linestyle='--', alpha=0.7, label='Entry')
            axes[0].axhline(y=trade.exit_price, color='green' if trade.profit_loss > 0 else 'red', 
                          linestyle='--', alpha=0.7, label='Exit')
            
            # Add text annotations
            axes[0].text(0.02, 0.98, f"Entry: ${trade.entry_price:,.2f}", transform=axes[0].transAxes,
                        fontsize=10, color='blue', verticalalignment='top')
            axes[0].text(0.02, 0.94, f"Exit: ${trade.exit_price:,.2f}", transform=axes[0].transAxes,
                        fontsize=10, color='green' if trade.profit_loss > 0 else 'red', verticalalignment='top')
            axes[0].text(0.02, 0.90, f"Duration: {trade.duration_minutes:.1f} min", transform=axes[0].transAxes,
                        fontsize=10, color='white', verticalalignment='top')
            axes[0].text(0.02, 0.86, f"Exit Type: {trade.exit_type}", transform=axes[0].transAxes,
                        fontsize=10, color='white', verticalalignment='top')
            
            # Save the figure
            fig.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='#1e1e1e')
            plt.close(fig)
            
            logger.info(f"📊 Trade chart saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating trade chart: {e}")
            return ""
    
    # Store candle data for chart generation
    candle_cache: Dict[str, pd.DataFrame] = {}
        
    def _load_weights(self, weights_file: str = None) -> list:
        """Load optimized weights from file"""
        if weights_file is None:
            weights_file = os.path.join(
                os.path.dirname(__file__),
                "data/btc_eth_learning/final_weights.json"
            )
        
        try:
            with open(weights_file, 'r') as f:
                data = json.load(f)
                weights = data.get('weights', [])
                logger.info(f"✓ Loaded {len(weights)} optimized weights from {weights_file}")
                return weights
        except FileNotFoundError:
            logger.warning(f"Weights file not found: {weights_file}. Using default weights.")
            return []
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            return []
    
    def _load_disabled_patterns(self, weights_file: str = None):
        """Load worst-performing patterns from saved stats and disable them"""
        if weights_file is None:
            weights_file = os.path.join(
                os.path.dirname(__file__),
                "data/btc_eth_learning/final_weights.json"
            )
        
        try:
            with open(weights_file, 'r') as f:
                data = json.load(f)
                pattern_stats = data.get('pattern_stats', {})
                
                # Patterns that are ALWAYS present (not specific signals) should be excluded
                always_present_patterns = {
                    'Uptrend', 'Downtrend', 'Market Regime', 'Volume Analysis'
                }
                
                for pattern, stats in pattern_stats.items():
                    # Skip always-present patterns
                    if pattern in always_present_patterns:
                        continue
                        
                    total_profit = stats.get('total_profit', 0)
                    trades = stats.get('detections', 0) or stats.get('trades', 0)
                    
                    # Only disable if we have enough data (>5 trades) and pattern is losing
                    if trades >= 5 and total_profit < self.worst_pattern_threshold:
                        self.disabled_patterns.add(pattern)
                
                if self.disabled_patterns:
                    logger.warning(f"⚠️  Disabled worst-performing patterns:")
                    for p in sorted(self.disabled_patterns):
                        stats = pattern_stats.get(p, {})
                        logger.warning(f"   ❌ {p}: ${stats.get('total_profit', 0):+.2f}")
                else:
                    logger.info(f"✓ No patterns disabled (all above threshold ${self.worst_pattern_threshold})")
                    
        except FileNotFoundError:
            logger.warning(f"Pattern stats not found in {weights_file}")
        except Exception as e:
            logger.error(f"Error loading pattern stats: {e}")
    
    def _is_pattern_disabled(self, reason: str) -> tuple:
        """Check if any disabled patterns are present in the signal reason"""
        if not self.disabled_patterns:
            return False, []
        
        reason_lower = reason.lower() if reason else ""
        
        # Pattern keywords mapping
        pattern_keywords = {
            "bullish order block": "Bullish Order Block",
            "bearish order block": "Bearish Order Block", 
            "bullish breaker block": "Bullish Breaker Block",
            "bearish breaker block": "Bearish Breaker Block",
            "breaker block": "Breaker Block",
            "order block": "Order Block",
            "fvg below": "FVG Below",
            "fvg above": "FVG Above",
            "unfilled fvg": "FVG",
            "near support": "Support Level",
            "near resistance": "Resistance Level",
            "swept through previous highs": "Liquidity Sweep (Highs)",
            "swept through previous lows": "Liquidity Sweep (Lows)",
            "broke structure upward": "Structure Break (Bullish)",
            "broke structure downward": "Structure Break (Bearish)",
            "broke structure": "Structure Break",
            "bullish pin bar": "Bullish Pin Bar",
            "bearish engulfing": "Bearish Engulfing",
            "pin bar": "Pin Bar",
            "engulfing": "Engulfing Pattern",
            "rsi oversold": "RSI Oversold",
            "rsi overbought": "RSI Overbought",
            "liquidity pool": "Liquidity Pool",
        }
        
        # Detect patterns in reason
        detected_patterns = []
        for keyword, pattern_name in pattern_keywords.items():
            if keyword in reason_lower and pattern_name not in detected_patterns:
                detected_patterns.append(pattern_name)
        
        # Check if any disabled pattern is present
        disabled_found = [p for p in detected_patterns if p in self.disabled_patterns]
        
        # Only block if majority of detected patterns are disabled
        if disabled_found and len(disabled_found) >= len(detected_patterns) / 2:
            return True, disabled_found
        
        return False, disabled_found
    
    def _get_candle_window(self, symbol: str, interval: str, window: int = None) -> Optional[pd.DataFrame]:
        """Fetch the latest candles for analysis"""
        if window is None:
            window = self.candle_window
        try:
            df = fetch_candles(symbol, window, interval)
            if df is None or df.empty:
                return None
            df.sort_index(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol} {interval}: {e}")
            return None
    
    def _should_fetch(self, combo_key: str, interval: str) -> bool:
        """Check if enough time has passed to fetch new data"""
        if combo_key not in self.last_fetch:
            return True
        
        elapsed = (datetime.now() - self.last_fetch[combo_key]).total_seconds()
        interval_seconds = self.INTERVAL_SECONDS.get(interval, 60)
        
        # Fetch slightly before the candle closes to be ready
        return elapsed >= (interval_seconds * 0.9)
    
    def _check_position_updates(self, position: OpenPosition, current_price: float) -> Optional[TradeResult]:
        """Check if position needs updating (stop-loss, take-profit, trailing stop)"""
        now = datetime.now()
        
        # Update current price and unrealized P/L
        position.current_price = current_price
        position.unrealized_pnl = position.position_size * (current_price - position.entry_price)
        
        # Update highest price for trailing stop
        if current_price > position.highest_price:
            position.highest_price = current_price
            
            # Activate trailing stop at TP1
            if not position.trailing_stop_active and current_price >= position.take_profit_1:
                position.trailing_stop_active = True
                logger.info(f"🔄 [{position.symbol} {position.interval}] Trailing stop ACTIVATED at {current_price:.2f}")
            
            # Update trailing stop level
            if position.trailing_stop_active and self.use_trailing_stop:
                new_stop = current_price * (1 - self.trailing_stop_distance / 100)
                if new_stop > position.trailing_stop_level:
                    old_stop = position.trailing_stop_level
                    position.trailing_stop_level = new_stop
                    position.stop_loss = new_stop
                    logger.info(f"📈 [{position.symbol} {position.interval}] Trailing stop updated: {old_stop:.2f} → {new_stop:.2f}")
        
        # Check take profit levels
        if current_price >= position.take_profit_3 and not position.tp3_hit:
            # Full target hit - close position
            profit = position.position_size * (current_price - position.entry_price)
            duration = (now - position.entry_time).total_seconds() / 60
            
            close_value = position.position_size * current_price
            self.balance += close_value
            self.invested_amount -= (position.position_size * position.entry_price)
            
            logger.info(f"🎯 [{position.symbol} {position.interval}] TP3 HIT at {current_price:.2f} | Profit: ${profit:+.2f}")
            
            # Update indicator stats for this trade
            is_win = profit > 0
            for indicator in position.indicator_contributions:
                if indicator in self.indicator_stats:
                    if is_win:
                        self.indicator_stats[indicator]["wins"] += 1
                    else:
                        self.indicator_stats[indicator]["losses"] += 1
                    self.indicator_stats[indicator]["total_profit"] += profit
            
            return TradeResult(
                position_id=position.position_id,
                symbol=position.symbol,
                interval=position.interval,
                entry_price=position.entry_price,
                exit_price=current_price,
                entry_time=position.entry_time,
                exit_time=now,
                profit_loss=profit,
                profit_percent=(current_price - position.entry_price) / position.entry_price * 100,
                exit_type="tp3_hit",
                duration_minutes=duration,
                confidence=position.confidence,
                indicator_contributions=position.indicator_contributions
            )
        
        elif current_price >= position.take_profit_2 and not position.tp2_hit:
            # TP2 hit - close 50% of remaining
            close_amount = position.position_size / 2
            profit = close_amount * (current_price - position.entry_price)
            self.balance += close_amount * current_price
            position.position_size -= close_amount
            position.tp2_hit = True
            
            logger.info(f"🎯 [{position.symbol} {position.interval}] TP2 HIT at {current_price:.2f} | Partial profit: ${profit:+.2f}")
        
        elif current_price >= position.take_profit_1 and not position.tp1_hit:
            # TP1 hit - close 33% of position
            close_amount = position.initial_position / 3
            profit = close_amount * (current_price - position.entry_price)
            self.balance += close_amount * current_price
            position.position_size -= close_amount
            position.tp1_hit = True
            
            logger.info(f"🎯 [{position.symbol} {position.interval}] TP1 HIT at {current_price:.2f} | Partial profit: ${profit:+.2f}")
        
        # Check stop loss
        elif current_price <= position.stop_loss:
            profit = position.position_size * (current_price - position.entry_price)
            duration = (now - position.entry_time).total_seconds() / 60
            
            close_value = position.position_size * current_price
            self.balance += close_value
            self.invested_amount -= (position.position_size * position.entry_price)
            
            stop_type = "trailing_stop" if position.trailing_stop_active else "stop_loss"
            emoji = "🛑" if profit < 0 else "✅"
            
            logger.info(f"{emoji} [{position.symbol} {position.interval}] {stop_type.upper()} at {current_price:.2f} | P/L: ${profit:+.2f}")
            
            # Update indicator stats for this trade
            is_win = profit > 0
            for indicator in position.indicator_contributions:
                if indicator in self.indicator_stats:
                    if is_win:
                        self.indicator_stats[indicator]["wins"] += 1
                    else:
                        self.indicator_stats[indicator]["losses"] += 1
                    self.indicator_stats[indicator]["total_profit"] += profit
            
            return TradeResult(
                position_id=position.position_id,
                symbol=position.symbol,
                interval=position.interval,
                entry_price=position.entry_price,
                exit_price=current_price,
                entry_time=position.entry_time,
                exit_time=now,
                profit_loss=profit,
                profit_percent=(current_price - position.entry_price) / position.entry_price * 100,
                exit_type=stop_type,
                duration_minutes=duration,
                confidence=position.confidence,
                indicator_contributions=position.indicator_contributions
            )
        
        return None
    
    def _process_combination(self, symbol: str, interval: str):
        """Process a single symbol/interval combination"""
        combo_key = f"{symbol}_{interval}"
        
        # Check if we should fetch new data
        if not self._should_fetch(combo_key, interval):
            return
        
        # Fetch candle data
        df = self._get_candle_window(symbol, interval)
        if df is None or len(df) < 100:
            return
        
        self.last_fetch[combo_key] = datetime.now()
        current_price = df["Close"].iloc[-1]
        
        # Check existing positions for this combination
        positions_to_close = []
        for pos_id, position in self.open_positions.items():
            if position.symbol == symbol and position.interval == interval:
                result = self._check_position_updates(position, current_price)
                if result:
                    positions_to_close.append((pos_id, result))
        
        # Close completed positions
        for pos_id, result in positions_to_close:
            del self.open_positions[pos_id]
            
            # Generate trade chart
            if df is not None and len(df) > 0:
                chart_path = self._generate_trade_chart(result, df)
                result.chart_path = chart_path
            
            self.completed_trades.append(result)
            
            # Update stats
            stats = self.combination_stats[combo_key]
            stats.trades += 1
            stats.total_profit += result.profit_loss
            if result.profit_loss > 0:
                stats.wins += 1
            else:
                stats.losses += 1
        
        # Check for new signal (only if no open position for this combo)
        has_open_position = any(
            p.symbol == symbol and p.interval == interval 
            for p in self.open_positions.values()
        )
        
        if not has_open_position:
            # Generate signal
            indicators = analyze_data(df, self.preferences, 0.05)
            signal, prob, confidence, reason, trading_signal = generate_price_prediction_signal_proba(
                df, indicators, self.weights, self.balance, self.risk_percentage
            )
            
            # Get confidence threshold from weights (index 18) or use default 0.3
            confidence_threshold = self.weights[18] if len(self.weights) > 18 else 0.3
            # Clamp to valid range [0.1, 0.9]
            confidence_threshold = max(0.1, min(0.9, confidence_threshold))
            
            if signal == "Bullish" and trading_signal and confidence > confidence_threshold:
                # Check for disabled patterns
                if self.disable_worst_patterns and reason:
                    is_disabled, disabled_list = self._is_pattern_disabled(reason)
                    if is_disabled:
                        logger.info(f"⏭️  {symbol} {interval}: Skipped - Disabled pattern(s): {', '.join(disabled_list)}")
                        return
                
                # Calculate position size based on risk
                position_sizing = calculate_position_size(
                    self.balance, self.risk_percentage, current_price, trading_signal.stop_loss
                )
                
                position_size = position_sizing["position_size"]
                amount_to_invest = position_size * current_price
                
                # Cap at maximum position percentage of TOTAL balance (not just available)
                max_position_value = self.initial_balance * (self.max_position_percent / 100)
                if amount_to_invest > max_position_value:
                    amount_to_invest = max_position_value
                    position_size = amount_to_invest / current_price
                    logger.info(f"⚠️ Position capped at {self.max_position_percent}% of balance (${max_position_value:,.2f})")
                
                # Also cap at available balance
                if amount_to_invest > self.balance:
                    amount_to_invest = self.balance
                    position_size = self.balance / current_price
                
                if position_size > 0 and amount_to_invest >= 10:  # Minimum $10 trade
                    self.balance -= amount_to_invest
                    self.invested_amount += amount_to_invest
                    
                    # Extract indicator contributions from the signal
                    reasons_list = reason.split('\n') if reason else []
                    indicator_contribs = extract_indicator_contributions(
                        bullish_score=prob if signal == "Bullish" else 0,
                        bearish_score=0 if signal == "Bullish" else prob,
                        signal_type=signal,
                        reasons=reasons_list
                    )
                    
                    # Update indicator stats for signal generation
                    for indicator in indicator_contribs:
                        if indicator in self.indicator_stats:
                            self.indicator_stats[indicator]["signals"] += 1
                    
                    position = OpenPosition(
                        position_id=str(uuid.uuid4()),
                        symbol=symbol,
                        interval=interval,
                        entry_price=current_price,
                        entry_time=datetime.now(),
                        position_size=position_size,
                        initial_position=position_size,
                        stop_loss=float(trading_signal.stop_loss),
                        initial_stop_loss=float(trading_signal.stop_loss),
                        take_profit_1=float(trading_signal.take_profit_1),
                        take_profit_2=float(trading_signal.take_profit_2),
                        take_profit_3=float(trading_signal.take_profit_3),
                        highest_price=current_price,
                        trailing_stop_level=float(trading_signal.stop_loss),
                        risk_reward_ratio=float(trading_signal.risk_reward_ratio),
                        signal_reason=reason.split('\n')[0] if reason else "",
                        current_price=current_price,
                        unrealized_pnl=0.0,
                        confidence=confidence,
                        indicator_contributions=indicator_contribs
                    )
                    
                    self.open_positions[position.position_id] = position
                    self.combination_stats[combo_key].signals_generated += 1
                    self.combination_stats[combo_key].last_signal_time = datetime.now()
                    
                    position_percent = (amount_to_invest / self.initial_balance) * 100
                    logger.info(f"""
{'='*60}
🚀 NEW SIGNAL: {symbol} {interval}
{'='*60}
   Entry Price:  ${current_price:,.2f}
   Stop Loss:    ${trading_signal.stop_loss:,.2f} ({(trading_signal.stop_loss/current_price-1)*100:.2f}%)
   TP1:          ${trading_signal.take_profit_1:,.2f} ({(trading_signal.take_profit_1/current_price-1)*100:.2f}%)
   TP2:          ${trading_signal.take_profit_2:,.2f} ({(trading_signal.take_profit_2/current_price-1)*100:.2f}%)
   TP3:          ${trading_signal.take_profit_3:,.2f} ({(trading_signal.take_profit_3/current_price-1)*100:.2f}%)
   Position:     {position_size:.6f} units (${amount_to_invest:,.2f} = {position_percent:.1f}% of balance)
   R/R Ratio:    {trading_signal.risk_reward_ratio:.2f}
   Confidence:   {confidence:.1%}
{'='*60}
""")
    
    def _print_status(self):
        """Print current status summary"""
        now = datetime.now()
        runtime = (now - self.start_time).total_seconds() / 60 if self.start_time else 0
        
        total_profit = sum(t.profit_loss for t in self.completed_trades)
        total_trades = len(self.completed_trades)
        wins = sum(1 for t in self.completed_trades if t.profit_loss > 0)
        
        # Fetch real-time prices for open positions
        if self.open_positions:
            symbols = set(pos.symbol for pos in self.open_positions.values())
            live_prices = {symbol: fetch_current_price(symbol) for symbol in symbols}
            for pos in self.open_positions.values():
                if pos.symbol in live_prices and live_prices[pos.symbol] > 0:
                    pos.current_price = live_prices[pos.symbol]
                    pos.unrealized_pnl = pos.position_size * (pos.current_price - pos.entry_price)
        
        # Calculate total equity and unrealized P/L
        total_equity = self.get_total_equity()
        total_unrealized = sum(pos.unrealized_pnl for pos in self.open_positions.values())
        
        print(f"""
{'═'*70}
📊 REAL-TIME TRADING STATUS - {now.strftime('%Y-%m-%d %H:%M:%S')}
{'═'*70}
  Runtime:           {runtime:.1f} minutes
  
  💰 BALANCE (Started: ${self.initial_balance:,.2f}):
     Available:      ${self.balance:,.2f}
     Invested:       ${self.invested_amount:,.2f}
     Base Total:     ${self.balance + self.invested_amount:,.2f} {'✓' if abs(self.balance + self.invested_amount - self.initial_balance - total_profit) < 1 else '⚠️ MISMATCH!'}
     Unrealized P/L: ${total_unrealized:+,.2f}
     Total Equity:   ${total_equity + total_unrealized:,.2f}
     
  📊 REALIZED P/L:   ${total_profit:+,.2f} ({total_profit/self.initial_balance*100:+.2f}%)
  
  Completed:         {total_trades} trades ({wins} wins, {total_trades-wins} losses)
  Win Rate:          {wins/total_trades*100:.1f}%
  Open Positions:    {len(self.open_positions)}
{'─'*70}
  COMBINATION STATS:
""" if total_trades > 0 else f"""
{'═'*70}
📊 REAL-TIME TRADING STATUS - {now.strftime('%Y-%m-%d %H:%M:%S')}
{'═'*70}
  Runtime:           {runtime:.1f} minutes
  
  💰 BALANCE (Started: ${self.initial_balance:,.2f}):
     Available:      ${self.balance:,.2f}
     Invested:       ${self.invested_amount:,.2f}
     Base Total:     ${self.balance + self.invested_amount:,.2f} {'✓' if abs(self.balance + self.invested_amount - self.initial_balance) < 1 else '⚠️ MISMATCH!'}
     Unrealized P/L: ${total_unrealized:+,.2f}
     Total Equity:   ${total_equity + total_unrealized:,.2f}
     
  Open Positions:    {len(self.open_positions)}
  Signals Generated: {sum(s.signals_generated for s in self.combination_stats.values())}
{'─'*70}
  COMBINATION STATS:
""")
        
        for key, stats in sorted(self.combination_stats.items()):
            wr = stats.wins / stats.trades * 100 if stats.trades > 0 else 0
            status = "🟢" if stats.total_profit > 0 else "🔴" if stats.total_profit < 0 else "⚪"
            print(f"    {status} {key:<15}: {stats.trades:>3} trades | ${stats.total_profit:>+10.2f} | {wr:>5.1f}% WR | {stats.signals_generated} signals")
        
        if self.open_positions:
            # Prices already updated at the top of this method
            # Calculate actual invested from positions (for verification)
            actual_invested = sum(pos.position_size * pos.entry_price for pos in self.open_positions.values())
            print(f"\n{'─'*70}\n  OPEN POSITIONS (Unrealized P/L: ${total_unrealized:+,.2f}):")
            for pos in self.open_positions.values():
                duration = (now - pos.entry_time).total_seconds() / 60
                pnl_pct = ((pos.current_price - pos.entry_price) / pos.entry_price * 100) if pos.current_price > 0 else 0
                pnl_emoji = "📈" if pos.unrealized_pnl >= 0 else "📉"
                pos_value = pos.position_size * pos.entry_price
                print(f"    {pnl_emoji} {pos.symbol} {pos.interval}:")
                print(f"       Entry: ${pos.entry_price:,.2f} → Current: ${pos.current_price:,.2f} ({pnl_pct:+.2f}%)")
                print(f"       Size: {pos.position_size:.6f} units (${pos_value:,.2f}) | P/L: ${pos.unrealized_pnl:+,.2f}")
                print(f"       SL: ${pos.stop_loss:,.2f} | Duration: {duration:.1f}m")
            
            # Show verification
            if abs(actual_invested - self.invested_amount) > 0.01:
                print(f"\n    ⚠️ ACCOUNTING MISMATCH:")
                print(f"       Tracked invested: ${self.invested_amount:,.2f}")
                print(f"       Actual from positions: ${actual_invested:,.2f}")
                print(f"       Difference: ${self.invested_amount - actual_invested:,.2f}")
        
        print(f"{'═'*70}\n")
    
    def run(self, duration_hours: float = None):
        """Run the real-time trading test"""
        self.running = True
        self.start_time = datetime.now()
        
        # Determine which combinations to monitor
        combinations = [
            c for c in self.COMBINATIONS 
            if not self.safe_only or c["priority"] == "SAFE"
        ]
        
        # Get confidence threshold from weights
        conf_threshold = self.weights[18] if len(self.weights) > 18 else 0.3
        conf_threshold = max(0.1, min(0.9, conf_threshold))
        
        logger.info(f"""
{'═'*70}
🚀 STARTING REAL-TIME TRADING TEST
{'═'*70}
  Initial Balance: ${self.initial_balance:,.2f}
  Risk per Trade:  {self.risk_percentage}%
  Max Position:    {self.max_position_percent}% of balance (${self.initial_balance * self.max_position_percent / 100:,.2f} max per trade)
  Candle Window:   {self.candle_window} candles
  Confidence:      >{conf_threshold:.0%} (learned threshold)
  Trailing Stop:   {'Enabled' if self.use_trailing_stop else 'Disabled'} ({self.trailing_stop_distance}%)
  Mode:            {'SAFE ONLY' if self.safe_only else 'ALL COMBINATIONS'}
  Duration:        {f'{duration_hours} hours' if duration_hours else 'Unlimited (Ctrl+C to stop)'}
  
  Monitoring {len(combinations)} combinations:
""")
        for c in combinations:
            logger.info(f"    • {c['symbol']} {c['interval']} [{c['priority']}]")
        
        logger.info(f"{'═'*70}\n")
        
        end_time = datetime.now() + timedelta(hours=duration_hours) if duration_hours else None
        last_status_print = datetime.now()
        status_interval = 60  # Print status every 60 seconds
        
        try:
            while self.running:
                if end_time and datetime.now() >= end_time:
                    logger.info("Duration reached. Stopping...")
                    break
                
                if self.stop_event.is_set():
                    break
                
                # Process each combination
                for combo in combinations:
                    if self.stop_event.is_set():
                        break
                    
                    try:
                        self._process_combination(combo["symbol"], combo["interval"])
                    except Exception as e:
                        logger.error(f"Error processing {combo['symbol']} {combo['interval']}: {e}")
                
                # Print status periodically
                if (datetime.now() - last_status_print).total_seconds() >= status_interval:
                    self._print_status()
                    last_status_print = datetime.now()
                
                # Sleep briefly between cycles
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("\n⚠️ Interrupted by user")
        finally:
            self.running = False
            self._print_final_report()
    
    def _print_final_report(self):
        """Print final trading report"""
        runtime = (datetime.now() - self.start_time).total_seconds() / 60 if self.start_time else 0
        
        total_profit = sum(t.profit_loss for t in self.completed_trades)
        total_trades = len(self.completed_trades)
        wins = sum(1 for t in self.completed_trades if t.profit_loss > 0)
        
        print(f"""
{'═'*70}
📊 FINAL REAL-TIME TRADING REPORT
{'═'*70}

  Duration:         {runtime:.1f} minutes ({runtime/60:.2f} hours)
  Initial Balance:  ${self.initial_balance:,.2f}
  Final Balance:    ${self.balance:,.2f}
  
  PERFORMANCE:
  ─────────────────────────────────
  Total Profit:     ${total_profit:+,.2f}
  ROI:              {total_profit/self.initial_balance*100:+.2f}%
  Total Trades:     {total_trades}
  Win Rate:         {wins/total_trades*100:.1f}% ({wins}/{total_trades})
  Avg Profit/Trade: ${total_profit/total_trades if total_trades else 0:.2f}
  
  BY COMBINATION:
  ─────────────────────────────────
""" if total_trades > 0 else f"""
{'═'*70}
📊 FINAL REAL-TIME TRADING REPORT
{'═'*70}

  Duration:         {runtime:.1f} minutes
  Initial Balance:  ${self.initial_balance:,.2f}
  Final Balance:    ${self.balance:,.2f}
  
  No completed trades during this session.
  Open positions: {len(self.open_positions)}
  Signals generated: {sum(s.signals_generated for s in self.combination_stats.values())}
  
{'═'*70}
""")
        
        if total_trades > 0:
            sorted_stats = sorted(
                self.combination_stats.items(),
                key=lambda x: x[1].total_profit,
                reverse=True
            )
            
            for key, stats in sorted_stats:
                if stats.trades > 0:
                    wr = stats.wins / stats.trades * 100
                    emoji = "🟢" if stats.total_profit > 0 else "🔴"
                    print(f"  {emoji} {key:<15}: {stats.trades:>3} trades | ${stats.total_profit:>+10.2f} | {wr:>5.1f}% WR")
            
            print(f"\n{'═'*70}")
        
        # Print indicator performance stats
        active_indicators = {k: v for k, v in self.indicator_stats.items() if v["signals"] > 0}
        if active_indicators:
            print(f"""
{'═'*70}
📈 INDICATOR CONTRIBUTION ANALYSIS
{'═'*70}
  (Sorted by profit contribution - worst performing first)
  
""")
            sorted_indicators = sorted(
                active_indicators.items(),
                key=lambda x: x[1]["total_profit"]
            )
            
            for indicator, stats in sorted_indicators:
                wr = (stats["wins"] / stats["signals"] * 100) if stats["signals"] > 0 else 0
                emoji = "🟢" if stats["total_profit"] > 0 else "🔴" if stats["total_profit"] < 0 else "⚪"
                print(f"  {emoji} {indicator:<25}: {stats['signals']:>3} signals | ${stats['total_profit']:>+10.2f} | {wr:>5.1f}% WR")
            
            print(f"\n{'═'*70}")
        
        # Save results to file
        self._save_results()
    
    def _save_results(self):
        """Save results to JSON file"""
        # Calculate unrealized P/L for open positions
        total_unrealized = sum(pos.unrealized_pnl for pos in self.open_positions.values())
        
        # Get confidence threshold from weights
        confidence_threshold = self.weights[18] if len(self.weights) > 18 else 0.3
        confidence_threshold = max(0.1, min(0.9, confidence_threshold))
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "runtime_minutes": (datetime.now() - self.start_time).total_seconds() / 60 if self.start_time else 0,
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "invested_amount": self.invested_amount,
            "total_equity": self.balance + self.invested_amount + total_unrealized,
            "realized_profit": sum(t.profit_loss for t in self.completed_trades),
            "unrealized_profit": total_unrealized,
            "total_completed_trades": len(self.completed_trades),
            "total_open_positions": len(self.open_positions),
            "confidence_threshold": confidence_threshold,
            
            # Completed trades
            "completed_trades": [
                {
                    "position_id": t.position_id,
                    "symbol": t.symbol,
                    "interval": t.interval,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "profit_loss": t.profit_loss,
                    "profit_percent": t.profit_percent,
                    "exit_type": t.exit_type,
                    "duration_minutes": t.duration_minutes,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "chart_path": getattr(t, 'chart_path', ''),
                    "confidence": t.confidence,
                    "indicator_contributions": t.indicator_contributions,
                }
                for t in self.completed_trades
            ],
            
            # Open positions (for potential continuation)
            "open_positions": [
                {
                    "position_id": pos.position_id,
                    "symbol": pos.symbol,
                    "interval": pos.interval,
                    "entry_price": pos.entry_price,
                    "entry_time": pos.entry_time.isoformat(),
                    "position_size": pos.position_size,
                    "position_value": pos.position_size * pos.entry_price,
                    "stop_loss": pos.stop_loss,
                    "initial_stop_loss": pos.initial_stop_loss,
                    "take_profit_1": pos.take_profit_1,
                    "take_profit_2": pos.take_profit_2,
                    "take_profit_3": pos.take_profit_3,
                    "highest_price": pos.highest_price,
                    "trailing_stop_active": pos.trailing_stop_active,
                    "trailing_stop_level": pos.trailing_stop_level,
                    "tp1_hit": pos.tp1_hit,
                    "tp2_hit": pos.tp2_hit,
                    "tp3_hit": pos.tp3_hit,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "duration_minutes": (datetime.now() - pos.entry_time).total_seconds() / 60,
                    "signal_reason": pos.signal_reason,
                    "confidence": pos.confidence,
                    "indicator_contributions": pos.indicator_contributions,
                }
                for pos in self.open_positions.values()
            ],
            
            # Indicator performance stats (sorted by profit, worst first)
            "indicator_stats": {
                indicator: {
                    **stats,
                    "win_rate": (stats["wins"] / stats["signals"] * 100) if stats["signals"] > 0 else 0.0
                }
                for indicator, stats in sorted(
                    self.indicator_stats.items(),
                    key=lambda x: x[1]["total_profit"]
                )
            },
            
            # Stats per combination
            "combination_stats": {
                key: {
                    "trades": s.trades,
                    "wins": s.wins,
                    "losses": s.losses,
                    "total_profit": s.total_profit,
                    "signals_generated": s.signals_generated,
                }
                for key, s in self.combination_stats.items()
            }
        }
        
        output_file = os.path.join(
            os.path.dirname(__file__),
            f"data/realtime_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✓ Results saved to: {output_file}")
        
        if self.open_positions:
            logger.info(f"   📍 {len(self.open_positions)} open positions saved for reference")


def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print("\n⚠️ Received interrupt signal. Shutting down gracefully...")
    raise KeyboardInterrupt


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-Time Trading Strategy Tester")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance (default: 10000)")
    parser.add_argument("--risk", type=float, default=1.0, help="Risk percentage per trade (default: 1.0)")
    parser.add_argument("--max-position", type=float, default=20.0, help="Max position size as %% of balance (default: 20)")
    parser.add_argument("--candles", type=int, default=500, help="Number of candles for analysis window (default: 500)")
    parser.add_argument("--duration", type=float, default=None, help="Test duration in hours (default: unlimited)")
    parser.add_argument("--safe-only", action="store_true", help="Only use safe combinations (0%% fail rate)")
    parser.add_argument("--no-trailing", action="store_true", help="Disable trailing stop")
    parser.add_argument("--disable-worst-patterns", action="store_true", 
                        help="Disable recognition of worst-performing patterns from backtest")
    parser.add_argument("--pattern-threshold", type=float, default=-100.0,
                        help="Profit threshold below which patterns are disabled (default: -100)")
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    trader = RealTimeTrader(
        initial_balance=args.balance,
        risk_percentage=args.risk,
        use_trailing_stop=not args.no_trailing,
        safe_only=args.safe_only,
        max_position_percent=args.max_position,
        candle_window=args.candles,
        disable_worst_patterns=args.disable_worst_patterns,
        worst_pattern_threshold=args.pattern_threshold,
    )
    
    trader.run(duration_hours=args.duration)


if __name__ == "__main__":
    main()

