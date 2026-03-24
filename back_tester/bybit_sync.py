#!/usr/bin/env python3
"""
Bybit Live Trading Synchronization Module

This module synchronizes the realtime test signals with actual Bybit trading.
When a signal is generated, it executes the trade on your real Bybit account.

WARNING: This module executes REAL trades with REAL money.
Make sure you understand the risks before using it.

Usage:
    python bybit_sync.py [--testnet] [--balance AMOUNT] [--risk PERCENT]
    
Example:
    # Paper trading with testnet
    python bybit_sync.py --testnet --balance 1000
    
    # Live trading (use with caution!)
    python bybit_sync.py --balance 5000 --risk 0.5
"""

import os
import sys
import json
import time
import signal
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from threading import Thread, Event, Lock
import uuid

# Add project root to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_dir)

import pandas as pd
import numpy as np

# Import trading client
from src.trading.bybit_client import BybitClient, OrderSide, OrderType, OrderResult

# Import from the project
from src.analysis.utils.helpers import fetch_candles
from src.telegram.signals.detection import (
    analyze_data,
    generate_price_prediction_signal_proba,
    TradingSignal,
    calculate_position_size,
)
from src.core.utils import create_true_preferences

# Try to import indicator contribution tracking
try:
    from back_tester.adaptive_learning import extract_indicator_contributions
    INDICATOR_TRACKING_AVAILABLE = True
except ImportError:
    INDICATOR_TRACKING_AVAILABLE = False
    def extract_indicator_contributions(*args, **kwargs):
        return {}

# Import Telegram notifier for alerts
try:
    from src.telegram.notifier import TelegramNotifier
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    TelegramNotifier = None

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
class LivePosition:
    """Represents a live trading position"""
    position_id: str
    symbol: str
    interval: str
    entry_price: float
    entry_time: datetime
    quantity: float  # Actual quantity bought
    invested_amount: float  # Amount in USDT invested
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
    initial_quantity: float = 0.0
    risk_reward_ratio: float = 2.0
    signal_reason: str = ""
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    confidence: float = 0.0
    order_id: str = ""  # Bybit order ID
    indicator_contributions: Dict[str, float] = field(default_factory=dict)


@dataclass
class LiveTradeResult:
    """Represents a completed live trade"""
    position_id: str
    symbol: str
    interval: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    quantity: float
    invested_amount: float
    realized_pnl: float
    profit_percent: float
    exit_type: str  # stop_loss, trailing_stop, tp1, tp2, tp3
    duration_minutes: float
    entry_order_id: str = ""
    exit_order_id: str = ""
    confidence: float = 0.0
    indicator_contributions: Dict[str, float] = field(default_factory=dict)


class BybitLiveTrader:
    """
    Live trading synchronization with Bybit.
    
    This class monitors signals from the trading strategy and executes
    real trades on your Bybit account.
    """
    
    # Trading combinations to monitor
    COMBINATIONS = [
        {"symbol": "BTCUSDT", "interval": "1h", "priority": "SAFE"},
        {"symbol": "ETHUSDT", "interval": "5m", "priority": "SAFE"},
        {"symbol": "ETHUSDT", "interval": "15m", "priority": "SAFE"},
        {"symbol": "BTCUSDT", "interval": "4h", "priority": "AGGRESSIVE"},
        {"symbol": "ETHUSDT", "interval": "4h", "priority": "AGGRESSIVE"},
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
    
    def __init__(
        self,
        bybit_client: BybitClient,
        max_position_usdt: float = 100.0,  # Max USDT per trade
        risk_percentage: float = 1.0,  # Risk per trade
        use_trailing_stop: bool = True,
        trailing_stop_distance: float = 0.5,
        weights_file: str = None,
        safe_only: bool = False,
        candle_window: int = 500,
        min_confidence: float = 0.3,
        max_open_positions: int = 3,
        position_check_interval: float = 10.0,  # Seconds between position checks
        telegram_chat_id: int = 840355587,  # Your Telegram ID for notifications
    ):
        """
        Initialize the live trader.
        
        Args:
            bybit_client: Initialized BybitClient instance
            max_position_usdt: Maximum USDT amount per trade
            risk_percentage: Risk percentage per trade
            use_trailing_stop: Enable trailing stop
            trailing_stop_distance: Trailing stop distance in %
            weights_file: Path to optimized weights file
            safe_only: Only trade SAFE combinations
            candle_window: Number of candles for analysis
            min_confidence: Minimum signal confidence to trade
            max_open_positions: Maximum concurrent open positions
            position_check_interval: Seconds between position price checks
            telegram_chat_id: Telegram user ID for notifications (your ID: 840355587)
        """
        self.client = bybit_client
        self.max_position_usdt = max_position_usdt
        self.risk_percentage = risk_percentage
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_distance = trailing_stop_distance
        self.safe_only = safe_only
        self.candle_window = candle_window
        self.min_confidence = min_confidence
        self.max_open_positions = max_open_positions
        self.position_check_interval = position_check_interval
        
        # Load optimized weights
        self.weights = self._load_weights(weights_file)
        
        # Trading state
        self.open_positions: Dict[str, LivePosition] = {}
        self.completed_trades: List[LiveTradeResult] = []
        self.position_lock = Lock()
        
        # Track total P/L
        self.total_realized_pnl = 0.0
        
        # Control flags
        self.running = False
        self.stop_event = Event()
        
        # Preferences for signal generation
        self.preferences = create_true_preferences()
        
        # Track last data fetch time per combination
        self.last_fetch: Dict[str, datetime] = {}
        
        # Start time
        self.start_time = None
        
        # Initial balance tracking
        self.initial_balance = 0.0
        
        # Initialize Telegram notifier for alerts
        self.notifier = None
        if TELEGRAM_AVAILABLE and telegram_chat_id > 0:
            try:
                self.notifier = TelegramNotifier(chat_id=telegram_chat_id)
                if self.notifier.enabled:
                    logger.info(f"✓ Telegram alerts enabled → chat ID: {telegram_chat_id}")
                else:
                    logger.warning("Telegram alerts disabled (check API_TELEGRAM_KEY in .env)")
                    self.notifier = None
            except Exception as e:
                logger.warning(f"Could not initialize Telegram notifier: {e}")
                self.notifier = None
        
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
                logger.info(f"✓ Loaded {len(weights)} optimized weights")
                return weights
        except FileNotFoundError:
            logger.warning(f"Weights file not found: {weights_file}")
            return []
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            return []
    
    def _get_candle_window(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch the latest candles for analysis"""
        try:
            df = fetch_candles(symbol, self.candle_window, interval)
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
        
        return elapsed >= (interval_seconds * 0.9)
    
    def _get_base_currency(self, symbol: str) -> str:
        """Extract base currency from symbol (e.g., BTCUSDT -> BTC)"""
        if symbol.endswith("USDT"):
            return symbol[:-4]
        elif symbol.endswith("USD"):
            return symbol[:-3]
        return symbol
    
    def _execute_buy(
        self,
        symbol: str,
        usdt_amount: float,
        current_price: float,
    ) -> Tuple[bool, str, float, float]:
        """
        Execute a buy order on Bybit.
        
        Args:
            symbol: Trading pair
            usdt_amount: Amount in USDT to spend
            current_price: Current market price
            
        Returns:
            Tuple of (success, order_id, actual_qty, actual_price)
        """
        try:
            # Calculate quantity to buy
            quantity = usdt_amount / current_price
            
            # Get minimum order size
            min_qty, max_qty, qty_step = self.client.get_lot_size_filter(symbol)
            
            # Round to valid quantity
            if qty_step > 0:
                quantity = round(quantity / qty_step) * qty_step
            
            if quantity < min_qty:
                logger.warning(f"Quantity {quantity} below minimum {min_qty} for {symbol}")
                return False, "", 0, 0
            
            # Place market buy order
            result = self.client.place_market_buy(
                symbol=symbol,
                qty=quantity,
                category="spot",
            )
            
            if result.success:
                logger.info(f"✅ Buy order executed: {quantity} {symbol} @ ~${current_price:.2f}")
                return True, result.order_id, quantity, current_price
            else:
                logger.error(f"❌ Buy order failed: {result.message}")
                return False, "", 0, 0
                
        except Exception as e:
            logger.error(f"❌ Error executing buy: {e}")
            return False, "", 0, 0
    
    def _execute_sell(
        self,
        symbol: str,
        quantity: float,
        current_price: float,
    ) -> Tuple[bool, str, float]:
        """
        Execute a sell order on Bybit.
        
        Args:
            symbol: Trading pair
            quantity: Quantity to sell
            current_price: Current market price
            
        Returns:
            Tuple of (success, order_id, actual_price)
        """
        try:
            # Get available balance
            base_currency = self._get_base_currency(symbol)
            available = self.client.get_spot_balance(base_currency)
            
            # Adjust quantity if we don't have enough
            if quantity > available:
                logger.warning(f"Adjusting sell quantity from {quantity} to {available}")
                quantity = available
            
            if quantity <= 0:
                logger.warning(f"No {base_currency} available to sell")
                return False, "", 0
            
            # Place market sell order
            result = self.client.place_market_sell(
                symbol=symbol,
                qty=quantity,
                category="spot",
            )
            
            if result.success:
                logger.info(f"✅ Sell order executed: {quantity} {symbol} @ ~${current_price:.2f}")
                return True, result.order_id, current_price
            else:
                logger.error(f"❌ Sell order failed: {result.message}")
                return False, "", 0
                
        except Exception as e:
            logger.error(f"❌ Error executing sell: {e}")
            return False, "", 0
    
    def _check_position_updates(self, position: LivePosition) -> Optional[LiveTradeResult]:
        """Check if position needs updating (stop-loss, take-profit, trailing stop)"""
        now = datetime.now()
        
        # Get current price
        current_price = self.client.get_current_price(position.symbol)
        if current_price <= 0:
            return None
        
        # Update position state
        position.current_price = current_price
        position.unrealized_pnl = position.quantity * (current_price - position.entry_price)
        
        # Update highest price for trailing stop
        if current_price > position.highest_price:
            position.highest_price = current_price
            
            # Activate trailing stop at TP1
            if not position.trailing_stop_active and current_price >= position.take_profit_1:
                position.trailing_stop_active = True
                logger.info(f"🔄 [{position.symbol}] Trailing stop ACTIVATED at ${current_price:.2f}")
            
            # Update trailing stop level
            if position.trailing_stop_active and self.use_trailing_stop:
                new_stop = current_price * (1 - self.trailing_stop_distance / 100)
                if new_stop > position.trailing_stop_level:
                    position.trailing_stop_level = new_stop
                    position.stop_loss = new_stop
                    logger.info(f"📈 [{position.symbol}] Trailing stop updated to ${new_stop:.2f}")
        
        exit_type = None
        exit_quantity = 0
        
        # Check take profit levels
        if current_price >= position.take_profit_3 and not position.tp3_hit:
            # Full target hit - close entire position
            exit_type = "tp3_hit"
            exit_quantity = position.quantity
            logger.info(f"🎯 [{position.symbol}] TP3 HIT at ${current_price:.2f}")
        
        elif current_price >= position.take_profit_2 and not position.tp2_hit:
            # TP2 hit - close 50% of remaining
            exit_type = "tp2_partial"
            exit_quantity = position.quantity / 2
            position.tp2_hit = True
            logger.info(f"🎯 [{position.symbol}] TP2 HIT - Selling 50%")
        
        elif current_price >= position.take_profit_1 and not position.tp1_hit:
            # TP1 hit - close 33% of position
            exit_type = "tp1_partial"
            exit_quantity = position.initial_quantity / 3
            position.tp1_hit = True
            logger.info(f"🎯 [{position.symbol}] TP1 HIT - Selling 33%")
        
        # Check stop loss
        elif current_price <= position.stop_loss:
            stop_type = "trailing_stop" if position.trailing_stop_active else "stop_loss"
            exit_type = stop_type
            exit_quantity = position.quantity
            emoji = "🛑" if position.unrealized_pnl < 0 else "✅"
            logger.info(f"{emoji} [{position.symbol}] {stop_type.upper()} at ${current_price:.2f}")
        
        # Execute exit if needed
        if exit_type and exit_quantity > 0:
            success, order_id, actual_price = self._execute_sell(
                position.symbol,
                exit_quantity,
                current_price
            )
            
            if success:
                realized_pnl = exit_quantity * (actual_price - position.entry_price)
                
                # Send Telegram notification for TP/SL
                if self.notifier:
                    try:
                        if exit_type.startswith("tp"):
                            # Extract TP level (tp1_partial -> 1, tp2_partial -> 2, tp3_hit -> 3)
                            tp_level = int(exit_type[2]) if len(exit_type) > 2 and exit_type[2].isdigit() else 1
                            self.notifier.send_tp_hit(
                                symbol=position.symbol,
                                tp_level=tp_level,
                                exit_price=actual_price,
                                entry_price=position.entry_price,
                                quantity_sold=exit_quantity,
                                realized_pnl=realized_pnl,
                                remaining_quantity=position.quantity - exit_quantity,
                            )
                        elif exit_type in ["stop_loss", "trailing_stop"]:
                            self.notifier.send_stop_loss(
                                symbol=position.symbol,
                                exit_price=actual_price,
                                entry_price=position.entry_price,
                                quantity=exit_quantity,
                                realized_pnl=realized_pnl,
                                is_trailing=(exit_type == "trailing_stop"),
                            )
                    except Exception as e:
                        logger.error(f"Failed to send Telegram alert: {e}")
                
                # Update position quantity
                position.quantity -= exit_quantity
                
                # If fully closed, create trade result
                if position.quantity <= 0.0001 or exit_type in ["tp3_hit", "stop_loss", "trailing_stop"]:
                    duration = (now - position.entry_time).total_seconds() / 60
                    
                    return LiveTradeResult(
                        position_id=position.position_id,
                        symbol=position.symbol,
                        interval=position.interval,
                        entry_price=position.entry_price,
                        exit_price=actual_price,
                        entry_time=position.entry_time,
                        exit_time=now,
                        quantity=position.initial_quantity,
                        invested_amount=position.invested_amount,
                        realized_pnl=realized_pnl,
                        profit_percent=(actual_price - position.entry_price) / position.entry_price * 100,
                        exit_type=exit_type,
                        duration_minutes=duration,
                        entry_order_id=position.order_id,
                        exit_order_id=order_id,
                        confidence=position.confidence,
                        indicator_contributions=position.indicator_contributions,
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
        current_price = self.client.get_current_price(symbol)
        
        if current_price <= 0:
            current_price = df["Close"].iloc[-1]
        
        # Check existing positions for this combination
        with self.position_lock:
            positions_to_close = []
            for pos_id, position in list(self.open_positions.items()):
                if position.symbol == symbol and position.interval == interval:
                    result = self._check_position_updates(position)
                    if result:
                        positions_to_close.append((pos_id, result))
            
            # Remove closed positions
            for pos_id, result in positions_to_close:
                del self.open_positions[pos_id]
                self.completed_trades.append(result)
                self.total_realized_pnl += result.realized_pnl
                
                emoji = "💰" if result.realized_pnl > 0 else "💸"
                logger.info(f"{emoji} Trade closed: ${result.realized_pnl:+.2f} ({result.profit_percent:+.2f}%)")
        
        # Check for new signal
        with self.position_lock:
            has_open_position = any(
                p.symbol == symbol and p.interval == interval
                for p in self.open_positions.values()
            )
            
            # Check max positions limit
            if len(self.open_positions) >= self.max_open_positions:
                return
        
        if not has_open_position:
            # Generate signal
            indicators = analyze_data(df, self.preferences, 0.05)
            signal, prob, confidence, reason, trading_signal = generate_price_prediction_signal_proba(
                df, indicators, self.weights, 10000, self.risk_percentage
            )
            
            # Get confidence threshold from weights
            confidence_threshold = self.weights[18] if len(self.weights) > 18 else 0.3
            confidence_threshold = max(0.1, min(0.9, confidence_threshold))
            
            if signal == "Bullish" and trading_signal and confidence > max(confidence_threshold, self.min_confidence):
                # Check available balance
                available_usdt = self.client.get_spot_balance("USDT")
                
                # Calculate position size (use smaller of max_position or available)
                position_usdt = min(self.max_position_usdt, available_usdt * 0.95)  # Keep 5% buffer
                
                if position_usdt < 10:  # Minimum $10 trade
                    logger.warning(f"Insufficient balance: ${available_usdt:.2f}")
                    return
                
                # Execute buy order
                success, order_id, quantity, actual_price = self._execute_buy(
                    symbol, position_usdt, current_price
                )
                
                if success and quantity > 0:
                    # Extract indicator contributions
                    reasons_list = reason.split('\n') if reason else []
                    indicator_contribs = extract_indicator_contributions(
                        bullish_score=prob,
                        bearish_score=0,
                        signal_type=signal,
                        reasons=reasons_list
                    )
                    
                    position = LivePosition(
                        position_id=str(uuid.uuid4()),
                        symbol=symbol,
                        interval=interval,
                        entry_price=actual_price,
                        entry_time=datetime.now(),
                        quantity=quantity,
                        invested_amount=position_usdt,
                        initial_quantity=quantity,
                        stop_loss=float(trading_signal.stop_loss),
                        initial_stop_loss=float(trading_signal.stop_loss),
                        take_profit_1=float(trading_signal.take_profit_1),
                        take_profit_2=float(trading_signal.take_profit_2),
                        take_profit_3=float(trading_signal.take_profit_3),
                        highest_price=actual_price,
                        trailing_stop_level=float(trading_signal.stop_loss),
                        risk_reward_ratio=float(trading_signal.risk_reward_ratio),
                        signal_reason=reason.split('\n')[0] if reason else "",
                        current_price=actual_price,
                        confidence=confidence,
                        order_id=order_id,
                        indicator_contributions=indicator_contribs,
                    )
                    
                    with self.position_lock:
                        self.open_positions[position.position_id] = position
                    
                    logger.info(f"""
{'='*60}
🚀 LIVE TRADE OPENED: {symbol} {interval}
{'='*60}
   Order ID:     {order_id}
   Entry Price:  ${actual_price:,.2f}
   Quantity:     {quantity:.6f} ({position_usdt:.2f} USDT)
   Stop Loss:    ${trading_signal.stop_loss:,.2f} ({(trading_signal.stop_loss/actual_price-1)*100:.2f}%)
   TP1:          ${trading_signal.take_profit_1:,.2f} ({(trading_signal.take_profit_1/actual_price-1)*100:.2f}%)
   TP2:          ${trading_signal.take_profit_2:,.2f} ({(trading_signal.take_profit_2/actual_price-1)*100:.2f}%)
   TP3:          ${trading_signal.take_profit_3:,.2f} ({(trading_signal.take_profit_3/actual_price-1)*100:.2f}%)
   Confidence:   {confidence:.1%}
{'='*60}
""")
                    
                    # Send Telegram notification for new trade
                    if self.notifier:
                        try:
                            self.notifier.send_signal_alert(
                                symbol=symbol,
                                interval=interval,
                                entry_price=actual_price,
                                stop_loss=float(trading_signal.stop_loss),
                                take_profit_1=float(trading_signal.take_profit_1),
                                take_profit_2=float(trading_signal.take_profit_2),
                                take_profit_3=float(trading_signal.take_profit_3),
                                confidence=confidence,
                                risk_reward=float(trading_signal.risk_reward_ratio),
                                invested_amount=position_usdt,
                                quantity=quantity,
                                order_id=order_id,
                            )
                        except Exception as e:
                            logger.error(f"Failed to send Telegram alert: {e}")
    
    def _position_monitor_loop(self):
        """Background thread to monitor open positions"""
        while self.running and not self.stop_event.is_set():
            try:
                with self.position_lock:
                    positions_to_close = []
                    
                    for pos_id, position in list(self.open_positions.items()):
                        result = self._check_position_updates(position)
                        if result:
                            positions_to_close.append((pos_id, result))
                    
                    for pos_id, result in positions_to_close:
                        del self.open_positions[pos_id]
                        self.completed_trades.append(result)
                        self.total_realized_pnl += result.realized_pnl
                        
                        emoji = "💰" if result.realized_pnl > 0 else "💸"
                        logger.info(f"{emoji} Trade closed: ${result.realized_pnl:+.2f}")
                
            except Exception as e:
                logger.error(f"Error in position monitor: {e}")
            
            time.sleep(self.position_check_interval)
    
    def _print_status(self):
        """Print current status summary"""
        now = datetime.now()
        runtime = (now - self.start_time).total_seconds() / 60 if self.start_time else 0
        
        # Get current balance
        current_balance = self.client.get_spot_balance("USDT")
        
        # Calculate unrealized P/L
        total_unrealized = sum(pos.unrealized_pnl for pos in self.open_positions.values())
        
        total_trades = len(self.completed_trades)
        wins = sum(1 for t in self.completed_trades if t.realized_pnl > 0)
        
        print(f"""
{'═'*70}
📊 LIVE TRADING STATUS - {now.strftime('%Y-%m-%d %H:%M:%S')}
{'═'*70}
  Runtime:           {runtime:.1f} minutes
  
  💰 BALANCE:
     USDT Available: ${current_balance:,.2f}
     Initial:        ${self.initial_balance:,.2f}
     Realized P/L:   ${self.total_realized_pnl:+,.2f}
     Unrealized P/L: ${total_unrealized:+,.2f}
  
  📊 TRADING STATS:
     Completed:      {total_trades} trades
     Win Rate:       {wins/total_trades*100:.1f}% ({wins}/{total_trades}) {'' if total_trades == 0 else ''}
     Open Positions: {len(self.open_positions)}
{'─'*70}""")
        
        if self.open_positions:
            print("  OPEN POSITIONS:")
            for pos in self.open_positions.values():
                duration = (now - pos.entry_time).total_seconds() / 60
                pnl_pct = ((pos.current_price - pos.entry_price) / pos.entry_price * 100) if pos.current_price > 0 else 0
                pnl_emoji = "📈" if pos.unrealized_pnl >= 0 else "📉"
                print(f"    {pnl_emoji} {pos.symbol} {pos.interval}:")
                print(f"       Entry: ${pos.entry_price:,.2f} → ${pos.current_price:,.2f} ({pnl_pct:+.2f}%)")
                print(f"       Qty: {pos.quantity:.6f} | P/L: ${pos.unrealized_pnl:+,.2f}")
                print(f"       SL: ${pos.stop_loss:,.2f} | Duration: {duration:.1f}m")
        
        print(f"{'═'*70}\n")
    
    def run(self, duration_hours: float = None):
        """Run the live trading system"""
        # Test connection
        if not self.client.test_connection():
            logger.error("❌ Failed to connect to Bybit. Check your API credentials.")
            return
        
        # Get initial balance
        self.initial_balance = self.client.get_spot_balance("USDT")
        
        self.running = True
        self.start_time = datetime.now()
        
        # Determine which combinations to monitor
        combinations = [
            c for c in self.COMBINATIONS
            if not self.safe_only or c["priority"] == "SAFE"
        ]
        
        # Get confidence threshold
        conf_threshold = self.weights[18] if len(self.weights) > 18 else 0.3
        conf_threshold = max(0.1, min(0.9, conf_threshold))
        
        logger.info(f"""
{'═'*70}
🚀 STARTING LIVE TRADING {'(TESTNET)' if self.client.testnet else '(MAINNET)'}
{'═'*70}
  ⚠️  WARNING: This will execute REAL trades!
  
  Initial Balance: ${self.initial_balance:,.2f} USDT
  Max per Trade:   ${self.max_position_usdt:,.2f} USDT
  Risk per Trade:  {self.risk_percentage}%
  Max Positions:   {self.max_open_positions}
  Confidence:      >{max(conf_threshold, self.min_confidence):.0%}
  Trailing Stop:   {'Enabled' if self.use_trailing_stop else 'Disabled'} ({self.trailing_stop_distance}%)
  Mode:            {'SAFE ONLY' if self.safe_only else 'ALL COMBINATIONS'}
  Duration:        {f'{duration_hours} hours' if duration_hours else 'Unlimited (Ctrl+C to stop)'}
  
  Monitoring {len(combinations)} combinations:
""")
        for c in combinations:
            logger.info(f"    • {c['symbol']} {c['interval']} [{c['priority']}]")
        
        logger.info(f"{'═'*70}\n")
        
        # Send Telegram startup notification
        if self.notifier:
            try:
                self.notifier.send_startup(
                    testnet=self.client.testnet,
                    initial_balance=self.initial_balance,
                    max_position=self.max_position_usdt,
                    max_positions=self.max_open_positions,
                    combinations=combinations,
                )
            except Exception as e:
                logger.error(f"Failed to send Telegram startup alert: {e}")
        
        # Start position monitor thread
        monitor_thread = Thread(target=self._position_monitor_loop, daemon=True)
        monitor_thread.start()
        
        end_time = datetime.now() + timedelta(hours=duration_hours) if duration_hours else None
        last_status_print = datetime.now()
        status_interval = 60
        
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
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("\n⚠️ Interrupted by user")
        finally:
            self.running = False
            self._print_final_report()
    
    def _print_final_report(self):
        """Print final trading report"""
        runtime = (datetime.now() - self.start_time).total_seconds() / 60 if self.start_time else 0
        final_balance = self.client.get_spot_balance("USDT")
        
        total_trades = len(self.completed_trades)
        wins = sum(1 for t in self.completed_trades if t.realized_pnl > 0)
        
        print(f"""
{'═'*70}
📊 FINAL LIVE TRADING REPORT
{'═'*70}

  Duration:         {runtime:.1f} minutes ({runtime/60:.2f} hours)
  Initial Balance:  ${self.initial_balance:,.2f}
  Final Balance:    ${final_balance:,.2f}
  
  PERFORMANCE:
  ─────────────────────────────────
  Realized P/L:     ${self.total_realized_pnl:+,.2f}
  ROI:              {(self.total_realized_pnl/self.initial_balance)*100 if self.initial_balance > 0 else 0:+.2f}%
  Total Trades:     {total_trades}
  Win Rate:         {wins/total_trades*100 if total_trades > 0 else 0:.1f}% ({wins}/{total_trades})
  
  Open Positions:   {len(self.open_positions)}
  
{'═'*70}
""")
        
        # Warn about open positions
        if self.open_positions:
            logger.warning(f"⚠️ {len(self.open_positions)} positions still open!")
            for pos in self.open_positions.values():
                logger.warning(f"   • {pos.symbol}: {pos.quantity:.6f} @ ${pos.entry_price:.2f}")
        
        # Send Telegram shutdown notification
        if self.notifier:
            try:
                self.notifier.send_shutdown(
                    runtime_minutes=runtime,
                    total_trades=total_trades,
                    total_pnl=self.total_realized_pnl,
                    open_positions=len(self.open_positions),
                )
            except Exception as e:
                logger.error(f"Failed to send Telegram shutdown alert: {e}")
        
        # Save results
        self._save_results()
    
    def _save_results(self):
        """Save trading results to JSON file"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "runtime_minutes": (datetime.now() - self.start_time).total_seconds() / 60 if self.start_time else 0,
            "initial_balance": self.initial_balance,
            "final_balance": self.client.get_spot_balance("USDT"),
            "total_realized_pnl": self.total_realized_pnl,
            "total_trades": len(self.completed_trades),
            "testnet": self.client.testnet,
            
            "completed_trades": [
                {
                    "position_id": t.position_id,
                    "symbol": t.symbol,
                    "interval": t.interval,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "invested_amount": t.invested_amount,
                    "realized_pnl": t.realized_pnl,
                    "profit_percent": t.profit_percent,
                    "exit_type": t.exit_type,
                    "duration_minutes": t.duration_minutes,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "entry_order_id": t.entry_order_id,
                    "exit_order_id": t.exit_order_id,
                }
                for t in self.completed_trades
            ],
            
            "open_positions": [
                {
                    "position_id": pos.position_id,
                    "symbol": pos.symbol,
                    "interval": pos.interval,
                    "entry_price": pos.entry_price,
                    "quantity": pos.quantity,
                    "invested_amount": pos.invested_amount,
                    "stop_loss": pos.stop_loss,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "order_id": pos.order_id,
                }
                for pos in self.open_positions.values()
            ],
        }
        
        output_file = os.path.join(
            os.path.dirname(__file__),
            f"data/live_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✓ Results saved to: {output_file}")


def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print("\n⚠️ Received interrupt signal. Shutting down gracefully...")
    raise KeyboardInterrupt


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Bybit Live Trading Synchronization")
    parser.add_argument("--testnet", action="store_true", help="Use Bybit testnet")
    parser.add_argument("--max-position", type=float, default=100.0, 
                        help="Maximum USDT per trade (default: 100)")
    parser.add_argument("--risk", type=float, default=1.0, 
                        help="Risk percentage per trade (default: 1.0)")
    parser.add_argument("--max-positions", type=int, default=3,
                        help="Maximum concurrent open positions (default: 3)")
    parser.add_argument("--duration", type=float, default=None, 
                        help="Trading duration in hours (default: unlimited)")
    parser.add_argument("--safe-only", action="store_true", 
                        help="Only use safe combinations")
    parser.add_argument("--min-confidence", type=float, default=0.3,
                        help="Minimum signal confidence (default: 0.3)")
    parser.add_argument("--no-trailing", action="store_true", 
                        help="Disable trailing stop")
    parser.add_argument("--confirm", action="store_true",
                        help="Confirm live trading (required for mainnet)")
    parser.add_argument("--telegram-id", type=int, default=840355587,
                        help="Telegram chat ID for alerts (default: 840355587)")
    parser.add_argument("--no-telegram", action="store_true",
                        help="Disable Telegram notifications")
    
    args = parser.parse_args()
    
    # Safety check for mainnet
    if not args.testnet and not args.confirm:
        print("""
⚠️  WARNING: You are about to start LIVE TRADING on MAINNET!
    
    This will execute REAL trades with REAL money.
    
    To confirm, add the --confirm flag:
    
    python bybit_sync.py --confirm [other options]
    
    Or use testnet for testing:
    
    python bybit_sync.py --testnet [other options]
""")
        return
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create Bybit client
    client = BybitClient(testnet=args.testnet)
    
    # Create trader
    trader = BybitLiveTrader(
        bybit_client=client,
        max_position_usdt=args.max_position,
        risk_percentage=args.risk,
        use_trailing_stop=not args.no_trailing,
        safe_only=args.safe_only,
        min_confidence=args.min_confidence,
        max_open_positions=args.max_positions,
        telegram_chat_id=0 if args.no_telegram else args.telegram_id,
    )
    
    # Run trading
    trader.run(duration_hours=args.duration)


if __name__ == "__main__":
    main()

