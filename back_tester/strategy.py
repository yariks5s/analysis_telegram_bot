import os
import sys
from typing import List, Tuple, Optional, Dict
import random
from datetime import datetime, timedelta
import uuid

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
    symbol: str,
    interval: str,
    candles: int = 1000,
    window: int = 300,
    initial_balance: float = 10000.0,
    liq_lev_tolerance: float = 0.05,
    weights: list = [],
    risk_percentage: float = 1.0,
    iteration_id: Optional[str] = None,
    db: Optional[ClickHouseDB] = None,
) -> Tuple[float, list, Optional[str]]:
    """
    Backtest a strategy with risk management:
      - At each step, generate a signal with risk management parameters
      - Use position sizing based on risk percentage
      - Implement multiple take profit levels
      - Use dynamic stop loss

    Parameters:
      symbol: The trading pair symbol (e.g. "BTCUSDT")
      interval: The candle interval (e.g. "1h")
      candles: Total number of candles to fetch
      window: Lookback period (number of candles) to use for generating the signal
      initial_balance: Starting capital in USD (or your base currency)
      liq_lev_tolerance: Tolerance for liquidity level detection
      risk_percentage: Percentage of account to risk per trade
      weights: List of weights for signal generation

    Returns:
      final_balance: The simulated portfolio balance at the end
      trade_log: A list of trade events for review
    """
    # Generate UUIDs for database tracking
    sub_iteration_id = str(uuid.uuid4()) if db else None
    if db and not iteration_id:
        iteration_id = str(uuid.uuid4())

    # Determine a random start time for fetching candles
    end_time = datetime.now()
    start_time = end_time - timedelta(days=730)  # Assuming 2 years of data availability
    random_start_time = start_time + timedelta(
        seconds=random.randint(0, int((end_time - start_time).total_seconds()))
    )

    # Convert datetime to timestamp for fetch_candles
    timestamp = random_start_time.timestamp()

    # Fetch candles starting from the random start time
    df = fetch_candles(symbol, candles, interval, timestamp=timestamp)
    if df is None or df.empty:
        raise ValueError("No data returned for backtesting.")

    # Ensure the DataFrame is sorted by time
    df.sort_index(inplace=True)

    balance = initial_balance
    position = 0.0  # quantity of asset held
    entry_price = None
    trade_log = []  # logs of trades for later analysis
    entry_time = None
    entry_index = None
    entry_signal = None
    current_trade = None  # Store current trade details

    # Use all indicators enabled by default in backtesting
    preferences = create_true_preferences()

    for i in range(window, len(df)):
        current_window = df.iloc[i - window : i]
        current_time = df.index[i]
        current_price = df["Close"].iloc[i]

        # Compute technical indicators and generate a signal
        indicators = analyze_data(current_window, preferences, liq_lev_tolerance)
        signal, prob, confidence, reason, trading_signal = (
            generate_price_prediction_signal_proba(
                current_window, indicators, weights, balance, risk_percentage
            )
        )

        # Handle existing position
        if position > 0 and current_trade:
            # Check take profit levels
            if (
                current_price >= current_trade["take_profit_1"]
                and not current_trade["tp1_hit"]
            ):
                # Close 1/3 of position at TP1
                close_amount = position / 3
                balance += close_amount * current_price
                position -= close_amount
                current_trade["tp1_hit"] = True
                trade_log.append(
                    {
                        "type": "tp1",
                        "price": current_price,
                        "index": i,
                        "signal": "Take Profit 1",
                        "timestamp": current_time,
                        "amount": close_amount,
                    }
                )
                print(f"[Index {i}] TP1 hit at {current_price:.5f}")

            elif (
                current_price >= current_trade["take_profit_2"]
                and not current_trade["tp2_hit"]
            ):
                # Close 1/2 of remaining position at TP2
                close_amount = position / 2
                balance += close_amount * current_price
                position -= close_amount
                current_trade["tp2_hit"] = True
                trade_log.append(
                    {
                        "type": "tp2",
                        "price": current_price,
                        "index": i,
                        "signal": "Take Profit 2",
                        "timestamp": current_time,
                        "amount": close_amount,
                    }
                )
                print(f"[Index {i}] TP2 hit at {current_price:.5f}")

            elif (
                current_price >= current_trade["take_profit_3"]
                and not current_trade["tp3_hit"]
            ):
                # Close remaining position at TP3
                balance += position * current_price
                trade_log.append(
                    {
                        "type": "tp3",
                        "price": current_price,
                        "index": i,
                        "signal": "Take Profit 3",
                        "timestamp": current_time,
                        "amount": position,
                    }
                )
                print(f"[Index {i}] TP3 hit at {current_price:.5f}")

                # Store complete trade in database if available
                if db and entry_time and entry_index:
                    trade_data = {
                        "iteration_id": iteration_id,
                        "sub_iteration_id": sub_iteration_id,
                        "symbol": symbol,
                        "interval": interval,
                        "trade_type": "tp3",
                        "entry_timestamp": entry_time,
                        "exit_timestamp": current_time,
                        "entry_index": entry_index,
                        "exit_index": i,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "profit_loss": current_price - entry_price,
                        "trade_duration": i - entry_index,
                        "entry_signal": entry_signal,
                        "exit_signal": "Take Profit 3",
                        "risk_reward_ratio": current_trade["risk_reward_ratio"],
                        "position_size": current_trade["position_size"],
                        "stop_loss": current_trade["stop_loss"],
                        "take_profit_1": current_trade["take_profit_1"],
                        "take_profit_2": current_trade["take_profit_2"],
                        "take_profit_3": current_trade["take_profit_3"],
                        "risk_percentage": risk_percentage,
                    }
                    db.insert_trade(trade_data)

                position = 0
                entry_price = None
                entry_time = None
                entry_index = None
                entry_signal = None
                current_trade = None

            # Check stop loss
            elif current_price <= current_trade["stop_loss"]:
                # Close entire position at stop loss
                balance += position * current_price
                trade_log.append(
                    {
                        "type": "stop_loss",
                        "price": current_price,
                        "index": i,
                        "signal": "Stop Loss",
                        "timestamp": current_time,
                        "amount": position,
                    }
                )
                print(f"[Index {i}] Stop Loss hit at {current_price:.5f}")

                # Store trade in database if available
                if db and entry_time and entry_index:
                    trade_data = {
                        "iteration_id": iteration_id,
                        "sub_iteration_id": sub_iteration_id,
                        "symbol": symbol,
                        "interval": interval,
                        "trade_type": "stop_loss",
                        "entry_timestamp": entry_time,
                        "exit_timestamp": current_time,
                        "entry_index": entry_index,
                        "exit_index": i,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "profit_loss": current_price - entry_price,
                        "trade_duration": i - entry_index,
                        "entry_signal": entry_signal,
                        "exit_signal": "Stop Loss",
                        "risk_reward_ratio": current_trade["risk_reward_ratio"],
                        "position_size": current_trade["position_size"],
                        "stop_loss": current_trade["stop_loss"],
                        "take_profit_1": current_trade["take_profit_1"],
                        "take_profit_2": current_trade["take_profit_2"],
                        "take_profit_3": current_trade["take_profit_3"],
                        "risk_percentage": risk_percentage,
                    }
                    db.insert_trade(trade_data)

                position = 0
                entry_price = None
                entry_time = None
                entry_index = None
                entry_signal = None
                current_trade = None

        # Handle new signal
        if signal == "Bullish" and position == 0 and trading_signal:
            # Calculate position size based on risk management
            position_sizing = calculate_position_size(
                balance, risk_percentage, current_price, trading_signal.stop_loss
            )

            # Enter position with calculated size
            position = position_sizing["position_size"]
            balance -= position * current_price
            entry_price = current_price
            entry_time = current_time
            entry_index = i
            entry_signal = signal

            # Store trade details
            current_trade = {
                "stop_loss": float(trading_signal.stop_loss),
                "take_profit_1": float(trading_signal.take_profit_1),
                "take_profit_2": float(trading_signal.take_profit_2),
                "take_profit_3": float(trading_signal.take_profit_3),
                "risk_reward_ratio": float(trading_signal.risk_reward_ratio),
                "position_size": float(position),
                "tp1_hit": False,
                "tp2_hit": False,
                "tp3_hit": False,
            }

            trade_log.append(
                {
                    "type": "buy",
                    "price": float(entry_price),
                    "index": int(i),
                    "signal": signal,
                    "timestamp": current_time,
                    "amount": float(position),
                    "stop_loss": float(trading_signal.stop_loss),
                    "take_profit_1": float(trading_signal.take_profit_1),
                    "take_profit_2": float(trading_signal.take_profit_2),
                    "take_profit_3": float(trading_signal.take_profit_3),
                }
            )

            # Store initial trade in database if available
            if db:
                trade_data = {
                    "iteration_id": str(iteration_id),
                    "sub_iteration_id": str(sub_iteration_id),
                    "symbol": str(symbol),
                    "interval": str(interval),
                    "trade_type": "buy",
                    "entry_timestamp": current_time,
                    "exit_timestamp": current_time,  # Will be updated on exit
                    "entry_index": int(i),
                    "exit_index": int(i),  # Will be updated on exit
                    "entry_price": float(entry_price),
                    "exit_price": float(entry_price),  # Will be updated on exit
                    "profit_loss": 0.0,  # Will be updated on exit
                    "trade_duration": 0,  # Will be updated on exit
                    "entry_signal": str(signal),
                    "exit_signal": "",  # Will be updated on exit
                    "risk_reward_ratio": float(trading_signal.risk_reward_ratio),
                    "position_size": float(position),
                    "stop_loss": float(trading_signal.stop_loss),
                    "take_profit_1": float(trading_signal.take_profit_1),
                    "take_profit_2": float(trading_signal.take_profit_2),
                    "take_profit_3": float(trading_signal.take_profit_3),
                    "risk_percentage": float(risk_percentage),
                }
                db.insert_trade(trade_data)

            print(
                f"[Index {i}] BUY at {entry_price:.5f} | Size: {position:.5f} | Reason: {reason.splitlines()[0]}"
            )

    # If still holding a position at the end, liquidate at the final price
    if position > 0:
        final_price = df["Close"].iloc[-1]
        final_time = df.index[-1]
        balance += position * final_price
        trade_log.append(
            {
                "type": "sell_end",
                "price": final_price,
                "index": len(df) - 1,
                "signal": "Final Sell",
                "timestamp": final_time,
                "amount": position,
            }
        )
        print(f"[Final] SELL at {final_price:.2f}")

        # Store final trade in database if available
        if db and entry_time and entry_index:
            trade_data = {
                "iteration_id": iteration_id,
                "sub_iteration_id": sub_iteration_id,
                "symbol": symbol,
                "interval": interval,
                "trade_type": "buy",
                "entry_timestamp": entry_time,
                "exit_timestamp": final_time,
                "entry_index": entry_index,
                "exit_index": len(df) - 1,
                "entry_price": entry_price,
                "exit_price": final_price,
                "profit_loss": final_price - entry_price,
                "trade_duration": len(df) - 1 - entry_index,
                "entry_signal": entry_signal,
                "exit_signal": "Final Sell",
                "risk_reward_ratio": (
                    current_trade["risk_reward_ratio"] if current_trade else 0
                ),
                "position_size": current_trade["position_size"] if current_trade else 0,
            }
            db.insert_trade(trade_data)

    return balance, trade_log, sub_iteration_id


if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = "1h"
    final_balance, trades, _ = backtest_strategy(
        symbol, interval, candles=600, window=300, risk_percentage=1.0
    )
    print(f"Final Balance: {final_balance:.2f}")
    print("Trade Log:")
    for trade in trades:
        print(trade)
