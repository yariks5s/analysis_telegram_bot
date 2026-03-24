"""
Bybit Trading Client for CryptoBot.

This module provides a client for executing trades on Bybit exchange,
including market orders, limit orders, and conditional orders (TP/SL).

Uses Bybit V5 API: https://bybit-exchange.github.io/docs/v5/intro
"""

import os
import time
import hmac
import hashlib
import json
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# API Endpoints
MAINNET_URL = "https://api.bybit.com"
TESTNET_URL = "https://api-testnet.bybit.com"


class OrderSide(Enum):
    """Order side (buy/sell)"""
    BUY = "Buy"
    SELL = "Sell"


class OrderType(Enum):
    """Order types supported by Bybit"""
    MARKET = "Market"
    LIMIT = "Limit"


class PositionMode(Enum):
    """Position mode for trading"""
    ONE_WAY = 0  # Merge long and short positions
    HEDGE = 3   # Separate long and short positions


class TriggerBy(Enum):
    """Trigger price type for conditional orders"""
    LAST_PRICE = "LastPrice"
    INDEX_PRICE = "IndexPrice"
    MARK_PRICE = "MarkPrice"


@dataclass
class OrderResult:
    """Result of an order execution"""
    success: bool
    order_id: str
    order_link_id: str
    symbol: str
    side: str
    order_type: str
    price: float
    qty: float
    status: str
    message: str
    raw_response: Dict[str, Any]


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    side: str
    size: float
    entry_price: float
    leverage: float
    unrealized_pnl: float
    realized_pnl: float
    position_value: float
    liq_price: float
    tp_sl_mode: str
    take_profit: float
    stop_loss: float
    created_time: datetime


class BybitClient:
    """
    Bybit exchange client for executing trades.
    
    Supports:
    - Spot trading
    - Market and limit orders
    - Take-profit and stop-loss orders
    - Position management
    """
    
    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        testnet: bool = False,
        recv_window: int = 5000,
    ):
        """
        Initialize the Bybit client.
        
        Args:
            api_key: Bybit API key (or from env BYBIT_API_KEY)
            api_secret: Bybit API secret (or from env BYBIT_API_SECRET)
            testnet: Use testnet instead of mainnet
            recv_window: Request timeout window in milliseconds
        """
        self.api_key = api_key or os.getenv("BYBIT_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BYBIT_API_SECRET", "")
        self.base_url = TESTNET_URL if testnet else MAINNET_URL
        self.recv_window = recv_window
        self.testnet = testnet
        
        if not self.api_key or not self.api_secret:
            logger.warning("Bybit API credentials not configured. Trading will fail.")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in milliseconds"""
        return str(int(time.time() * 1000))
    
    def _generate_signature(self, timestamp: str, params: Dict[str, Any]) -> str:
        """
        Generate HMAC SHA256 signature for API authentication.
        
        Args:
            timestamp: Current timestamp
            params: Request parameters
            
        Returns:
            Hex-encoded signature
        """
        param_str = timestamp + self.api_key + str(self.recv_window) + json.dumps(params, separators=(',', ':'))
        return hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _generate_get_signature(self, timestamp: str, query_string: str) -> str:
        """
        Generate signature for GET requests.
        
        Args:
            timestamp: Current timestamp
            query_string: URL query string
            
        Returns:
            Hex-encoded signature
        """
        param_str = timestamp + self.api_key + str(self.recv_window) + query_string
        return hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None,
        signed: bool = True,
    ) -> Dict[str, Any]:
        """
        Make an API request to Bybit.
        
        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint
            params: Request parameters
            signed: Whether to sign the request
            
        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        if signed:
            timestamp = self._get_timestamp()
            headers["X-BAPI-API-KEY"] = self.api_key
            headers["X-BAPI-TIMESTAMP"] = timestamp
            headers["X-BAPI-RECV-WINDOW"] = str(self.recv_window)
            
            if method == "GET":
                query_string = "&".join([f"{k}={v}" for k, v in (params or {}).items()])
                headers["X-BAPI-SIGN"] = self._generate_get_signature(timestamp, query_string)
                response = requests.get(url, params=params, headers=headers, timeout=10)
            else:
                headers["X-BAPI-SIGN"] = self._generate_signature(timestamp, params or {})
                response = requests.post(url, json=params, headers=headers, timeout=10)
        else:
            if method == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=10)
            else:
                response = requests.post(url, json=params, headers=headers, timeout=10)
        
        try:
            data = response.json()
            if data.get("retCode") != 0:
                logger.error(f"Bybit API error: {data.get('retMsg')} (code: {data.get('retCode')})")
            return data
        except Exception as e:
            logger.error(f"Error parsing Bybit response: {e}")
            return {"retCode": -1, "retMsg": str(e)}
    
    # ========== Account Methods ==========
    
    def get_wallet_balance(self, account_type: str = "UNIFIED", coin: str = None) -> Dict[str, Any]:
        """
        Get wallet balance.
        
        Args:
            account_type: Account type (UNIFIED, SPOT, etc.)
            coin: Optional specific coin to query
            
        Returns:
            Wallet balance information
        """
        params = {"accountType": account_type}
        if coin:
            params["coin"] = coin
        
        return self._request("GET", "/v5/account/wallet-balance", params)
    
    def get_available_balance(self, coin: str = "USDT") -> float:
        """
        Get available balance for a specific coin.
        
        Args:
            coin: Coin symbol (default: USDT)
            
        Returns:
            Available balance as float
        """
        result = self.get_wallet_balance(coin=coin)
        
        try:
            if result.get("retCode") == 0:
                accounts = result.get("result", {}).get("list", [])
                for account in accounts:
                    coins = account.get("coin", [])
                    for c in coins:
                        if c.get("coin") == coin:
                            return float(c.get("availableToWithdraw", 0))
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
        
        return 0.0
    
    # ========== Market Data Methods ==========
    
    def get_ticker(self, symbol: str, category: str = "spot") -> Dict[str, Any]:
        """
        Get current ticker information.
        
        Args:
            symbol: Trading pair symbol
            category: Category (spot, linear, inverse)
            
        Returns:
            Ticker information
        """
        params = {"category": category, "symbol": symbol}
        return self._request("GET", "/v5/market/tickers", params, signed=False)
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get current market price for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Current price as float
        """
        result = self.get_ticker(symbol)
        
        try:
            if result.get("retCode") == 0:
                tickers = result.get("result", {}).get("list", [])
                if tickers:
                    return float(tickers[0].get("lastPrice", 0))
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
        
        return 0.0
    
    def get_instrument_info(self, symbol: str, category: str = "spot") -> Dict[str, Any]:
        """
        Get trading instrument information (min qty, tick size, etc.)
        
        Args:
            symbol: Trading pair symbol
            category: Category (spot, linear, inverse)
            
        Returns:
            Instrument information
        """
        params = {"category": category, "symbol": symbol}
        return self._request("GET", "/v5/market/instruments-info", params, signed=False)
    
    def get_lot_size_filter(self, symbol: str) -> Tuple[float, float, float]:
        """
        Get lot size constraints for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (min_qty, max_qty, qty_step)
        """
        result = self.get_instrument_info(symbol)
        
        try:
            if result.get("retCode") == 0:
                instruments = result.get("result", {}).get("list", [])
                if instruments:
                    lot_size = instruments[0].get("lotSizeFilter", {})
                    return (
                        float(lot_size.get("minOrderQty", 0)),
                        float(lot_size.get("maxOrderQty", 0)),
                        float(lot_size.get("basePrecision", 0.001)),
                    )
        except Exception as e:
            logger.error(f"Error getting lot size for {symbol}: {e}")
        
        return (0.0, 0.0, 0.001)
    
    # ========== Order Methods ==========
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        qty: float,
        price: float = None,
        category: str = "spot",
        time_in_force: str = "GTC",
        order_link_id: str = None,
        reduce_only: bool = False,
        take_profit: float = None,
        stop_loss: float = None,
    ) -> OrderResult:
        """
        Place a trading order.
        
        Args:
            symbol: Trading pair symbol
            side: Order side (Buy/Sell)
            order_type: Order type (Market/Limit)
            qty: Order quantity
            price: Limit price (required for limit orders)
            category: Category (spot, linear, inverse)
            time_in_force: Time in force (GTC, IOC, FOK)
            order_link_id: Custom order ID
            reduce_only: Reduce only order (for derivatives)
            take_profit: Take profit price
            stop_loss: Stop loss price
            
        Returns:
            OrderResult with execution details
        """
        # Round quantity to valid precision
        min_qty, max_qty, qty_step = self.get_lot_size_filter(symbol)
        
        # Adjust quantity to valid step
        if qty_step > 0:
            qty = round(qty / qty_step) * qty_step
        
        # Validate quantity
        if qty < min_qty:
            return OrderResult(
                success=False,
                order_id="",
                order_link_id=order_link_id or "",
                symbol=symbol,
                side=side.value,
                order_type=order_type.value,
                price=price or 0,
                qty=qty,
                status="REJECTED",
                message=f"Quantity {qty} below minimum {min_qty}",
                raw_response={}
            )
        
        params = {
            "category": category,
            "symbol": symbol,
            "side": side.value,
            "orderType": order_type.value,
            "qty": str(qty),
            "timeInForce": time_in_force,
        }
        
        if order_link_id:
            params["orderLinkId"] = order_link_id
        
        if order_type == OrderType.LIMIT and price:
            params["price"] = str(price)
        
        if category != "spot":
            if reduce_only:
                params["reduceOnly"] = True
            if take_profit:
                params["takeProfit"] = str(take_profit)
            if stop_loss:
                params["stopLoss"] = str(stop_loss)
        
        result = self._request("POST", "/v5/order/create", params)
        
        if result.get("retCode") == 0:
            order_result = result.get("result", {})
            return OrderResult(
                success=True,
                order_id=order_result.get("orderId", ""),
                order_link_id=order_result.get("orderLinkId", ""),
                symbol=symbol,
                side=side.value,
                order_type=order_type.value,
                price=price or 0,
                qty=qty,
                status="CREATED",
                message="Order placed successfully",
                raw_response=result
            )
        else:
            return OrderResult(
                success=False,
                order_id="",
                order_link_id=order_link_id or "",
                symbol=symbol,
                side=side.value,
                order_type=order_type.value,
                price=price or 0,
                qty=qty,
                status="FAILED",
                message=result.get("retMsg", "Unknown error"),
                raw_response=result
            )
    
    def place_market_buy(
        self,
        symbol: str,
        qty: float,
        category: str = "spot",
        order_link_id: str = None,
    ) -> OrderResult:
        """
        Place a market buy order.
        
        Args:
            symbol: Trading pair symbol
            qty: Order quantity (in base currency)
            category: Category (spot, linear, inverse)
            order_link_id: Custom order ID
            
        Returns:
            OrderResult with execution details
        """
        return self.place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=qty,
            category=category,
            order_link_id=order_link_id,
        )
    
    def place_market_sell(
        self,
        symbol: str,
        qty: float,
        category: str = "spot",
        order_link_id: str = None,
    ) -> OrderResult:
        """
        Place a market sell order.
        
        Args:
            symbol: Trading pair symbol
            qty: Order quantity (in base currency)
            category: Category (spot, linear, inverse)
            order_link_id: Custom order ID
            
        Returns:
            OrderResult with execution details
        """
        return self.place_order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            qty=qty,
            category=category,
            order_link_id=order_link_id,
        )
    
    def cancel_order(
        self,
        symbol: str,
        order_id: str = None,
        order_link_id: str = None,
        category: str = "spot",
    ) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to cancel
            order_link_id: Custom order ID to cancel
            category: Category (spot, linear, inverse)
            
        Returns:
            Cancellation result
        """
        params = {"category": category, "symbol": symbol}
        
        if order_id:
            params["orderId"] = order_id
        elif order_link_id:
            params["orderLinkId"] = order_link_id
        else:
            return {"retCode": -1, "retMsg": "Either orderId or orderLinkId required"}
        
        return self._request("POST", "/v5/order/cancel", params)
    
    def cancel_all_orders(self, symbol: str = None, category: str = "spot") -> Dict[str, Any]:
        """
        Cancel all open orders.
        
        Args:
            symbol: Optional specific symbol to cancel
            category: Category (spot, linear, inverse)
            
        Returns:
            Cancellation result
        """
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        
        return self._request("POST", "/v5/order/cancel-all", params)
    
    def get_open_orders(
        self,
        symbol: str = None,
        category: str = "spot",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get list of open orders.
        
        Args:
            symbol: Optional specific symbol
            category: Category (spot, linear, inverse)
            limit: Maximum number of orders to return
            
        Returns:
            List of open orders
        """
        params = {"category": category, "limit": limit}
        if symbol:
            params["symbol"] = symbol
        
        result = self._request("GET", "/v5/order/realtime", params)
        
        if result.get("retCode") == 0:
            return result.get("result", {}).get("list", [])
        return []
    
    def get_order_history(
        self,
        symbol: str = None,
        category: str = "spot",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get order history.
        
        Args:
            symbol: Optional specific symbol
            category: Category (spot, linear, inverse)
            limit: Maximum number of orders to return
            
        Returns:
            List of historical orders
        """
        params = {"category": category, "limit": limit}
        if symbol:
            params["symbol"] = symbol
        
        result = self._request("GET", "/v5/order/history", params)
        
        if result.get("retCode") == 0:
            return result.get("result", {}).get("list", [])
        return []
    
    # ========== Spot Trading with TP/SL ==========
    
    def place_spot_order_with_tp_sl(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        take_profit: float = None,
        stop_loss: float = None,
        entry_price: float = None,
    ) -> Tuple[OrderResult, Optional[str], Optional[str]]:
        """
        Place a spot market order and set up monitoring for TP/SL.
        
        Since spot trading doesn't support native TP/SL orders,
        this returns the necessary information for manual monitoring.
        
        Args:
            symbol: Trading pair symbol
            side: Order side (Buy/Sell)
            qty: Order quantity
            take_profit: Take profit price
            stop_loss: Stop loss price
            entry_price: Expected entry price (for logging)
            
        Returns:
            Tuple of (OrderResult, tp_order_id, sl_order_id)
            Note: For spot, tp_order_id and sl_order_id will be None
        """
        # Place the market order
        order_result = self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            qty=qty,
            category="spot",
        )
        
        if order_result.success:
            logger.info(
                f"✅ Spot order placed: {side.value} {qty} {symbol} "
                f"(TP: {take_profit}, SL: {stop_loss})"
            )
        
        # For spot trading, we can't set native TP/SL
        # The caller needs to monitor and manage exits manually
        return order_result, None, None
    
    # ========== Position Management ==========
    
    def get_positions(
        self,
        symbol: str = None,
        category: str = "linear",
    ) -> List[Position]:
        """
        Get open positions (for derivatives).
        
        Args:
            symbol: Optional specific symbol
            category: Category (linear, inverse)
            
        Returns:
            List of Position objects
        """
        params = {"category": category, "settleCoin": "USDT"}
        if symbol:
            params["symbol"] = symbol
        
        result = self._request("GET", "/v5/position/list", params)
        positions = []
        
        if result.get("retCode") == 0:
            for pos in result.get("result", {}).get("list", []):
                if float(pos.get("size", 0)) > 0:
                    positions.append(Position(
                        symbol=pos.get("symbol"),
                        side=pos.get("side"),
                        size=float(pos.get("size", 0)),
                        entry_price=float(pos.get("avgPrice", 0)),
                        leverage=float(pos.get("leverage", 1)),
                        unrealized_pnl=float(pos.get("unrealisedPnl", 0)),
                        realized_pnl=float(pos.get("cumRealisedPnl", 0)),
                        position_value=float(pos.get("positionValue", 0)),
                        liq_price=float(pos.get("liqPrice", 0)),
                        tp_sl_mode=pos.get("tpSlMode", ""),
                        take_profit=float(pos.get("takeProfit", 0)),
                        stop_loss=float(pos.get("stopLoss", 0)),
                        created_time=datetime.fromtimestamp(
                            int(pos.get("createdTime", 0)) / 1000
                        ) if pos.get("createdTime") else datetime.now(),
                    ))
        
        return positions
    
    # ========== Spot Balance Methods ==========
    
    def get_spot_balance(self, coin: str) -> float:
        """
        Get available spot balance for a specific coin.
        
        Args:
            coin: Coin symbol (e.g., "BTC", "ETH", "USDT")
            
        Returns:
            Available balance as float
        """
        result = self.get_wallet_balance(account_type="UNIFIED", coin=coin)
        
        try:
            if result.get("retCode") == 0:
                accounts = result.get("result", {}).get("list", [])
                for account in accounts:
                    coins = account.get("coin", [])
                    for c in coins:
                        if c.get("coin") == coin:
                            return float(c.get("walletBalance", 0))
        except Exception as e:
            logger.error(f"Error getting spot balance for {coin}: {e}")
        
        return 0.0
    
    def get_all_spot_balances(self) -> Dict[str, float]:
        """
        Get all non-zero spot balances.
        
        Returns:
            Dictionary of coin -> balance
        """
        result = self.get_wallet_balance(account_type="UNIFIED")
        balances = {}
        
        try:
            if result.get("retCode") == 0:
                accounts = result.get("result", {}).get("list", [])
                for account in accounts:
                    coins = account.get("coin", [])
                    for c in coins:
                        balance = float(c.get("walletBalance", 0))
                        if balance > 0:
                            balances[c.get("coin")] = balance
        except Exception as e:
            logger.error(f"Error getting spot balances: {e}")
        
        return balances
    
    # ========== Utility Methods ==========
    
    def test_connection(self) -> bool:
        """
        Test API connection and authentication.
        
        Returns:
            True if connection successful
        """
        try:
            result = self.get_wallet_balance()
            if result.get("retCode") == 0:
                logger.info("✅ Bybit API connection successful")
                return True
            else:
                logger.error(f"❌ Bybit API error: {result.get('retMsg')}")
                return False
        except Exception as e:
            logger.error(f"❌ Bybit API connection failed: {e}")
            return False
    
    def get_server_time(self) -> int:
        """
        Get Bybit server time.
        
        Returns:
            Server timestamp in milliseconds
        """
        result = self._request("GET", "/v5/market/time", signed=False)
        
        if result.get("retCode") == 0:
            return int(result.get("result", {}).get("timeSecond", 0)) * 1000
        return 0

