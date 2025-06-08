import os
import sys
from typing import List, Tuple, Optional, Dict, Any
import random
from datetime import datetime, timedelta
import uuid
import numpy as np

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

import pandas as pd  # type: ignore
from data_fetching_instruments import fetch_candles, analyze_data
from signal_detection import (
    generate_price_prediction_signal_proba,
    TradingSignal,
    calculate_position_size,
)
from utils import create_true_preferences
from db_operations import ClickHouseDB


def backtest_strategy(
    df: pd.DataFrame,
    weights: List[float],
    db: Optional[ClickHouseDB] = None,
    iteration_id: Optional[str] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Backtest strategy with the given weights"""
    # Generate UUIDs for database tracking
    sub_iteration_id = str(uuid.uuid4())
    if iteration_id is None:
        iteration_id = str(uuid.uuid4())

    # Initialize variables
    balance = 10000.0  # Starting balance
    position = None
    trade_log = []
    current_trade = None

    # Ensure all price columns are float
    df = df.copy()
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].astype(float)

    # Process each candle
    for i in range(len(df)):
        # Get current candle and ensure price is float
        candle = df.iloc[i]
        current_price = float(candle["Close"])

        # If we have an open position, check for take profit or stop loss
        if position is not None:
            # Check take profit levels
            if float(current_price) >= float(
                position["take_profit_1"]
            ) and not position.get("tp1_hit", False):
                # Take profit 1 hit
                profit = (
                    float(position["take_profit_1"]) - float(position["entry_price"])
                ) * float(position["size"])
                balance += profit
                position["tp1_hit"] = True
                position["tp1_hit_price"] = current_price
                position["tp1_hit_time"] = candle.name

                # Log TP1 trade
                trade_data = {
                    "trade_id": str(uuid.uuid4()),
                    "sub_iteration_id": sub_iteration_id,
                    "iteration_id": iteration_id,
                    "trade_type": "tp1",
                    "price": float(position["take_profit_1"]),
                    "amount": float(
                        position["size"] * 0.4
                    ),  # Take 40% of position at TP1
                    "timestamp": candle.name,
                    "index": i,
                    "balance": float(balance),
                    "profit": float(profit),
                    "entry_price": float(position["entry_price"]),
                    "stop_loss": float(position["stop_loss"]),
                    "take_profit_1": float(position["take_profit_1"]),
                    "take_profit_2": float(position["take_profit_2"]),
                    "take_profit_3": float(position["take_profit_3"]),
                    "risk_percentage": float(position["risk_percentage"]),
                }
                trade_log.append(trade_data)
                if db:
                    db.insert_trade(trade_data)

            elif float(current_price) >= float(
                position["take_profit_2"]
            ) and not position.get("tp2_hit", False):
                # Take profit 2 hit
                profit = (
                    float(position["take_profit_2"]) - float(position["entry_price"])
                ) * float(position["size"])
                balance += profit
                position["tp2_hit"] = True
                position["tp2_hit_price"] = current_price
                position["tp2_hit_time"] = candle.name

                # Log TP2 trade
                trade_data = {
                    "trade_id": str(uuid.uuid4()),
                    "sub_iteration_id": sub_iteration_id,
                    "iteration_id": iteration_id,
                    "trade_type": "tp2",
                    "price": float(position["take_profit_2"]),
                    "amount": float(
                        position["size"] * 0.4
                    ),  # Take 40% of position at TP2
                    "timestamp": candle.name,
                    "index": i,
                    "balance": float(balance),
                    "profit": float(profit),
                    "entry_price": float(position["entry_price"]),
                    "stop_loss": float(position["stop_loss"]),
                    "take_profit_1": float(position["take_profit_1"]),
                    "take_profit_2": float(position["take_profit_2"]),
                    "take_profit_3": float(position["take_profit_3"]),
                    "risk_percentage": float(position["risk_percentage"]),
                }
                trade_log.append(trade_data)
                if db:
                    db.insert_trade(trade_data)

            elif float(current_price) >= float(
                position["take_profit_3"]
            ) and not position.get("tp3_hit", False):
                # Take profit 3 hit
                profit = (
                    float(position["take_profit_3"]) - float(position["entry_price"])
                ) * float(position["size"])
                balance += profit
                position["tp3_hit"] = True
                position["tp3_hit_price"] = current_price
                position["tp3_hit_time"] = candle.name

                # Log TP3 trade
                trade_data = {
                    "trade_id": str(uuid.uuid4()),
                    "sub_iteration_id": sub_iteration_id,
                    "iteration_id": iteration_id,
                    "trade_type": "tp3",
                    "price": float(position["take_profit_3"]),
                    "amount": float(
                        position["size"] * 0.2
                    ),  # Take remaining 20% of position at TP3
                    "timestamp": candle.name,
                    "index": i,
                    "balance": float(balance),
                    "profit": float(profit),
                    "entry_price": float(position["entry_price"]),
                    "stop_loss": float(position["stop_loss"]),
                    "take_profit_1": float(position["take_profit_1"]),
                    "take_profit_2": float(position["take_profit_2"]),
                    "take_profit_3": float(position["take_profit_3"]),
                    "risk_percentage": float(position["risk_percentage"]),
                }
                trade_log.append(trade_data)
                if db:
                    db.insert_trade(trade_data)

                # Close position after TP3
                position = None

            # Check stop loss
            elif float(current_price) <= float(position["stop_loss"]):
                # Stop loss hit
                loss = (
                    float(position["stop_loss"]) - float(position["entry_price"])
                ) * float(position["size"])
                balance += loss

                # Log stop loss trade
                trade_data = {
                    "trade_id": str(uuid.uuid4()),
                    "sub_iteration_id": sub_iteration_id,
                    "iteration_id": iteration_id,
                    "trade_type": "stop_loss",
                    "price": float(position["stop_loss"]),
                    "amount": float(position["size"]),
                    "timestamp": candle.name,
                    "index": i,
                    "balance": float(balance),
                    "profit": float(loss),
                    "entry_price": float(position["entry_price"]),
                    "stop_loss": float(position["stop_loss"]),
                    "take_profit_1": float(position["take_profit_1"]),
                    "take_profit_2": float(position["take_profit_2"]),
                    "take_profit_3": float(position["take_profit_3"]),
                    "risk_percentage": float(position["risk_percentage"]),
                }
                trade_log.append(trade_data)
                if db:
                    db.insert_trade(trade_data)

                position = None

        # Check for new entry signals
        if position is None:
            # Get signal for current candle
            current_df = df.iloc[
                : i + 1
            ].copy()  # Create a copy of the data up to current candle
            signal, prob_bullish, confidence, reason, trading_signal = (
                generate_price_prediction_signal_proba(
                    current_df,  # Use data up to current candle
                    None,  # No indicators needed
                    weights=weights,
                    account_balance=balance,
                    risk_percentage=1.0,
                )
            )

            # Enter position if we have a valid trading signal
            if (
                trading_signal
                and trading_signal.signal_type == "Bullish"
                and confidence > 0.3
            ):
                # Calculate position size based on risk
                risk_amount = balance * 0.01  # 1% risk per trade
                price_difference = abs(
                    float(trading_signal.entry_price) - float(trading_signal.stop_loss)
                )
                if price_difference > 0:
                    position_size = risk_amount / price_difference
                else:
                    position_size = 0

                # Store initial trade data
                current_trade = {
                    "trade_id": str(uuid.uuid4()),
                    "sub_iteration_id": sub_iteration_id,
                    "iteration_id": iteration_id,
                    "trade_type": "buy",
                    "price": float(trading_signal.entry_price),
                    "amount": float(position_size),
                    "timestamp": candle.name,
                    "index": i,
                    "balance": float(balance),
                    "profit": 0.0,
                    "entry_price": float(trading_signal.entry_price),
                    "stop_loss": float(trading_signal.stop_loss),
                    "take_profit_1": float(trading_signal.take_profit_1),
                    "take_profit_2": float(trading_signal.take_profit_2),
                    "take_profit_3": float(trading_signal.take_profit_3),
                    "risk_percentage": 1.0,
                }

                # Store trade in database if available
                if db:
                    db.insert_trade(current_trade)

                # Update trade log
                trade_log.append(current_trade)

                # Open position
                position = {
                    "entry_price": float(trading_signal.entry_price),
                    "size": float(position_size),
                    "stop_loss": float(trading_signal.stop_loss),
                    "take_profit_1": float(trading_signal.take_profit_1),
                    "take_profit_2": float(trading_signal.take_profit_2),
                    "take_profit_3": float(trading_signal.take_profit_3),
                    "risk_percentage": 1.0,
                }

                print(
                    f"Buy at {i}: {trading_signal.entry_price}, size: {position_size}, reason: {reason}"
                )

    # Close any remaining position at the end
    if position is not None:
        final_profit = (current_price - float(position["entry_price"])) * float(
            position["size"]
        )
        if not np.isnan(final_profit):
            balance += final_profit

    return balance, trade_log


if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = "1h"
    final_balance, trades = backtest_strategy(
        df=pd.DataFrame(), weights=[0.3, 0.3, 0.4], db=None, iteration_id=None
    )
    print(f"Final Balance: {final_balance:.2f}")
    print("Trade Log:")
    for trade in trades:
        print(trade)
