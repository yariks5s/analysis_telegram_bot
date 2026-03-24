"""
Telegram Notifier for Live Trading Alerts

This module provides simple notification functionality for trading events.
It sends messages directly to a specified Telegram chat/user ID.

Usage:
    notifier = TelegramNotifier(chat_id=840355587)
    notifier.send_signal_alert(symbol, entry, sl, tp1, tp2, tp3, confidence)
    notifier.send_trade_closed(symbol, entry, exit, pnl, exit_type)
"""

import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class TradeEventType(Enum):
    """Types of trade events for notifications"""
    NEW_SIGNAL = "new_signal"
    TRADE_OPENED = "trade_opened"
    TP1_HIT = "tp1_hit"
    TP2_HIT = "tp2_hit"
    TP3_HIT = "tp3_hit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TRADE_CLOSED = "trade_closed"
    ERROR = "error"


class TelegramNotifier:
    """
    Simple Telegram notifier for trading alerts.
    
    Sends notifications about:
    - New trading signals
    - Trade entries
    - Take-profit hits (TP1, TP2, TP3)
    - Stop-loss hits
    - Trade closures with P/L summary
    """
    
    def __init__(
        self,
        chat_id: int = None,
        bot_token: str = None,
        enabled: bool = True,
    ):
        """
        Initialize the notifier.
        
        Args:
            chat_id: Telegram chat/user ID to send notifications to
            bot_token: Telegram bot token (or from env API_TELEGRAM_KEY)
            enabled: Whether notifications are enabled
        """
        self.chat_id = chat_id or int(os.getenv("TELEGRAM_CHAT_ID", "0"))
        self.bot_token = bot_token or os.getenv("API_TELEGRAM_KEY", "")
        self.enabled = enabled and bool(self.bot_token) and self.chat_id > 0
        
        if not self.enabled:
            if not self.bot_token:
                logger.warning("Telegram notifier disabled: No bot token configured")
            elif self.chat_id <= 0:
                logger.warning("Telegram notifier disabled: No chat ID configured")
        else:
            logger.info(f"✓ Telegram notifier enabled for chat ID: {self.chat_id}")
    
    @property
    def api_url(self) -> str:
        """Get Telegram API URL"""
        return f"https://api.telegram.org/bot{self.bot_token}"
    
    def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False,
    ) -> bool:
        """
        Send a message to the configured chat.
        
        Args:
            text: Message text (supports HTML formatting)
            parse_mode: Parse mode (HTML or Markdown)
            disable_notification: Send silently
            
        Returns:
            True if message sent successfully
        """
        if not self.enabled:
            return False
        
        try:
            url = f"{self.api_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_notification": disable_notification,
            }
            
            response = requests.post(url, json=payload, timeout=10)
            result = response.json()
            
            if result.get("ok"):
                return True
            else:
                logger.error(f"Telegram API error: {result.get('description')}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_signal_alert(
        self,
        symbol: str,
        interval: str,
        entry_price: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float,
        take_profit_3: float,
        confidence: float,
        risk_reward: float = 0,
        invested_amount: float = 0,
        quantity: float = 0,
        order_id: str = "",
        reasons: str = "",
    ) -> bool:
        """
        Send notification for a new trading signal/trade entry.
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit_1: First take profit level
            take_profit_2: Second take profit level
            take_profit_3: Third take profit level
            confidence: Signal confidence (0-1)
            risk_reward: Risk/reward ratio
            invested_amount: Amount invested in USDT
            quantity: Quantity bought
            order_id: Bybit order ID
            reasons: Signal reasons (optional)
        """
        sl_pct = ((stop_loss / entry_price) - 1) * 100
        tp1_pct = ((take_profit_1 / entry_price) - 1) * 100
        tp2_pct = ((take_profit_2 / entry_price) - 1) * 100
        tp3_pct = ((take_profit_3 / entry_price) - 1) * 100
        
        message = f"""
🚀 <b>NEW TRADE OPENED</b>

<b>Symbol:</b> {symbol} ({interval})
<b>Entry:</b> ${entry_price:,.2f}
<b>Confidence:</b> {confidence:.0%}

📊 <b>Targets:</b>
├ SL: ${stop_loss:,.2f} ({sl_pct:+.2f}%)
├ TP1: ${take_profit_1:,.2f} ({tp1_pct:+.2f}%)
├ TP2: ${take_profit_2:,.2f} ({tp2_pct:+.2f}%)
└ TP3: ${take_profit_3:,.2f} ({tp3_pct:+.2f}%)
"""
        
        if invested_amount > 0:
            message += f"""
💰 <b>Position:</b>
├ Invested: ${invested_amount:,.2f}
├ Quantity: {quantity:.6f}
└ R/R: {risk_reward:.2f}
"""
        
        if order_id:
            message += f"\n🆔 Order: <code>{order_id}</code>"
        
        message += f"\n\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send_message(message)
    
    def send_tp_hit(
        self,
        symbol: str,
        tp_level: int,
        exit_price: float,
        entry_price: float,
        quantity_sold: float,
        realized_pnl: float,
        remaining_quantity: float = 0,
    ) -> bool:
        """
        Send notification for take-profit hit.
        
        Args:
            symbol: Trading pair
            tp_level: Take profit level (1, 2, or 3)
            exit_price: Exit price
            entry_price: Original entry price
            quantity_sold: Quantity sold at this TP
            realized_pnl: Realized P/L from this sale
            remaining_quantity: Remaining position quantity
        """
        profit_pct = ((exit_price / entry_price) - 1) * 100
        
        if tp_level == 3:
            emoji = "🎯🎯🎯"
            action = "FULL EXIT"
        else:
            emoji = "🎯" * tp_level
            action = f"PARTIAL EXIT ({33 if tp_level == 1 else 50}%)"
        
        message = f"""
{emoji} <b>TP{tp_level} HIT - {action}</b>

<b>Symbol:</b> {symbol}
<b>Entry:</b> ${entry_price:,.2f}
<b>Exit:</b> ${exit_price:,.2f} ({profit_pct:+.2f}%)

💰 <b>Result:</b>
├ Sold: {quantity_sold:.6f}
├ P/L: ${realized_pnl:+,.2f}
"""
        
        if remaining_quantity > 0 and tp_level < 3:
            message += f"└ Remaining: {remaining_quantity:.6f}"
        else:
            message += "└ Position closed"
        
        message += f"\n\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send_message(message)
    
    def send_stop_loss(
        self,
        symbol: str,
        exit_price: float,
        entry_price: float,
        quantity: float,
        realized_pnl: float,
        is_trailing: bool = False,
    ) -> bool:
        """
        Send notification for stop-loss hit.
        
        Args:
            symbol: Trading pair
            exit_price: Exit price
            entry_price: Original entry price
            quantity: Position quantity
            realized_pnl: Realized P/L
            is_trailing: Whether this was a trailing stop
        """
        loss_pct = ((exit_price / entry_price) - 1) * 100
        
        if is_trailing:
            emoji = "📉" if realized_pnl < 0 else "✅"
            title = "TRAILING STOP HIT"
        else:
            emoji = "🛑"
            title = "STOP LOSS HIT"
        
        result_emoji = "💸" if realized_pnl < 0 else "💰"
        
        message = f"""
{emoji} <b>{title}</b>

<b>Symbol:</b> {symbol}
<b>Entry:</b> ${entry_price:,.2f}
<b>Exit:</b> ${exit_price:,.2f} ({loss_pct:+.2f}%)

{result_emoji} <b>Result:</b>
├ Quantity: {quantity:.6f}
└ P/L: ${realized_pnl:+,.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_message(message)
    
    def send_trade_summary(
        self,
        symbol: str,
        interval: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        invested_amount: float,
        realized_pnl: float,
        exit_type: str,
        duration_minutes: float,
        entry_order_id: str = "",
        exit_order_id: str = "",
    ) -> bool:
        """
        Send trade closure summary.
        
        Args:
            symbol: Trading pair
            interval: Timeframe
            entry_price: Entry price
            exit_price: Final exit price
            quantity: Position quantity
            invested_amount: Amount invested
            realized_pnl: Total realized P/L
            exit_type: How the trade was closed
            duration_minutes: Trade duration
            entry_order_id: Entry order ID
            exit_order_id: Exit order ID
        """
        profit_pct = ((exit_price / entry_price) - 1) * 100
        roi = (realized_pnl / invested_amount) * 100 if invested_amount > 0 else 0
        
        is_profit = realized_pnl > 0
        emoji = "💰" if is_profit else "💸"
        result = "PROFIT" if is_profit else "LOSS"
        
        # Format duration
        if duration_minutes < 60:
            duration_str = f"{duration_minutes:.0f}m"
        else:
            hours = int(duration_minutes // 60)
            mins = int(duration_minutes % 60)
            duration_str = f"{hours}h {mins}m"
        
        message = f"""
{emoji} <b>TRADE CLOSED - {result}</b>

<b>Symbol:</b> {symbol} ({interval})
<b>Duration:</b> {duration_str}
<b>Exit Type:</b> {exit_type.replace('_', ' ').title()}

📊 <b>Price Action:</b>
├ Entry: ${entry_price:,.2f}
└ Exit: ${exit_price:,.2f} ({profit_pct:+.2f}%)

💵 <b>Financial Result:</b>
├ Invested: ${invested_amount:,.2f}
├ P/L: ${realized_pnl:+,.2f}
└ ROI: {roi:+.2f}%

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_message(message)
    
    def send_daily_summary(
        self,
        total_trades: int,
        wins: int,
        losses: int,
        total_pnl: float,
        best_trade: float,
        worst_trade: float,
        current_balance: float,
        initial_balance: float,
    ) -> bool:
        """
        Send daily trading summary.
        """
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        roi = ((current_balance - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0
        
        emoji = "📈" if total_pnl > 0 else "📉"
        
        message = f"""
{emoji} <b>DAILY TRADING SUMMARY</b>

📊 <b>Trading Stats:</b>
├ Total Trades: {total_trades}
├ Wins: {wins} ✅
├ Losses: {losses} ❌
└ Win Rate: {win_rate:.1f}%

💰 <b>P/L Summary:</b>
├ Total P/L: ${total_pnl:+,.2f}
├ Best Trade: ${best_trade:+,.2f}
└ Worst Trade: ${worst_trade:+,.2f}

💵 <b>Account:</b>
├ Initial: ${initial_balance:,.2f}
├ Current: ${current_balance:,.2f}
└ ROI: {roi:+.2f}%

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_message(message)
    
    def send_error(self, error_message: str, context: str = "") -> bool:
        """
        Send error notification.
        
        Args:
            error_message: Error description
            context: Additional context about the error
        """
        message = f"""
⚠️ <b>TRADING ERROR</b>

<b>Error:</b> {error_message}
"""
        
        if context:
            message += f"\n<b>Context:</b> {context}"
        
        message += f"\n\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send_message(message)
    
    def send_startup(
        self,
        testnet: bool,
        initial_balance: float,
        max_position: float,
        max_positions: int,
        combinations: list,
    ) -> bool:
        """
        Send notification when trading bot starts.
        """
        mode = "🧪 TESTNET" if testnet else "💵 LIVE"
        
        pairs = "\n".join([f"├ {c['symbol']} ({c['interval']})" for c in combinations[:-1]])
        pairs += f"\n└ {combinations[-1]['symbol']} ({combinations[-1]['interval']})" if combinations else ""
        
        message = f"""
🤖 <b>TRADING BOT STARTED</b>

<b>Mode:</b> {mode}
<b>Balance:</b> ${initial_balance:,.2f}
<b>Max per Trade:</b> ${max_position:,.2f}
<b>Max Positions:</b> {max_positions}

📊 <b>Monitoring:</b>
{pairs}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_message(message)
    
    def send_shutdown(
        self,
        runtime_minutes: float,
        total_trades: int,
        total_pnl: float,
        open_positions: int,
    ) -> bool:
        """
        Send notification when trading bot stops.
        """
        hours = int(runtime_minutes // 60)
        mins = int(runtime_minutes % 60)
        
        emoji = "💰" if total_pnl > 0 else "💸" if total_pnl < 0 else "⚪"
        
        message = f"""
🛑 <b>TRADING BOT STOPPED</b>

<b>Runtime:</b> {hours}h {mins}m
<b>Trades:</b> {total_trades}
<b>P/L:</b> ${total_pnl:+,.2f} {emoji}
<b>Open Positions:</b> {open_positions}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_message(message)
    
    def test_connection(self) -> bool:
        """
        Test Telegram bot connection by sending a test message.
        
        Returns:
            True if test message sent successfully
        """
        return self.send_message(
            "✅ <b>Connection Test</b>\n\nTelegram notifier is working!",
            disable_notification=True
        )


# Default notifier instance (can be imported and used directly)
def create_notifier(chat_id: int = None) -> TelegramNotifier:
    """
    Create a notifier instance with the specified chat ID.
    
    Args:
        chat_id: Telegram chat/user ID (default from TELEGRAM_CHAT_ID env var)
        
    Returns:
        Configured TelegramNotifier instance
    """
    return TelegramNotifier(chat_id=chat_id)

