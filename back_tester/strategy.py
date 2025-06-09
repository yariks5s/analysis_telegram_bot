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
    symbol: str,
    interval: str,
    candles: pd.DataFrame,
    window: int,
    initial_balance: float,
    liq_lev_tolerance: float,
    risk_percentage: float,
    weights: List[float] = None,
    iteration_id: Optional[str] = None,
    db: Optional[ClickHouseDB] = None,
) -> Tuple[float, List[Dict[str, Any]], Optional[str]]:
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
    df = fetch_candles(symbol, len(candles), interval, timestamp=timestamp)
    if df is None or df.empty:
        raise ValueError("No data returned for backtesting.")

    # Ensure the DataFrame is sorted by time
    df.sort_index(inplace=True)

    balance = initial_balance
    position = None
    trade_log = []
    current_trade = None

    # Use all indicators enabled by default in backtesting
    preferences = create_true_preferences()

    # Generate signals
    signals = generate_signals(df, window, liq_lev_tolerance, weights)

    for i in range(window, len(df)):
        current_price = df["Close"].iloc[i]

        # Handle existing position
        if position:
            # Check take profit levels
            if current_price >= position["take_profit_3"]:
                close_amount = position["amount"]
                profit = (current_price - position["entry_price"]) * close_amount
                balance += profit if not np.isnan(profit) else 0.0

                trade_log.append(
                    {
                        "type": "tp3",
                        "index": i,
                        "price": current_price,
                        "amount": close_amount,
                        "profit": profit,
                        "balance": balance,
                    }
                )
                position = None
                current_trade = None

            elif current_price >= position["take_profit_2"]:
                close_amount = position["amount"] * 0.5
                profit = (current_price - position["entry_price"]) * close_amount
                balance += profit if not np.isnan(profit) else 0.0
                position["amount"] -= close_amount

                trade_log.append(
                    {
                        "type": "tp2",
                        "index": i,
                        "price": current_price,
                        "amount": close_amount,
                        "profit": profit,
                        "balance": balance,
                    }
                )

            elif current_price >= position["take_profit_1"]:
                close_amount = position["amount"] * 0.5
                profit = (current_price - position["entry_price"]) * close_amount
                balance += profit if not np.isnan(profit) else 0.0
                position["amount"] -= close_amount

                trade_log.append(
                    {
                        "type": "tp1",
                        "index": i,
                        "price": current_price,
                        "amount": close_amount,
                        "profit": profit,
                        "balance": balance,
                    }
                )

            # Check stop loss
            elif current_price <= position["stop_loss"]:
                close_amount = position["amount"]
                profit = (current_price - position["entry_price"]) * close_amount
                balance += profit if not np.isnan(profit) else 0.0

                trade_log.append(
                    {
                        "type": "stop_loss",
                        "index": i,
                        "price": current_price,
                        "amount": close_amount,
                        "profit": profit,
                        "balance": balance,
                    }
                )
                position = None
                current_trade = None

        # Check for new signals
        if not position and i < len(signals):
            signal = signals[i]
            if signal and signal.signal_type == "Bullish":
                # Calculate position size based on risk
                risk_amount = balance * (risk_percentage / 100)
                price_difference = abs(signal.entry_price - signal.stop_loss)
                if price_difference > 0:  # Prevent division by zero
                    position_size = risk_amount / price_difference
                else:
                    position_size = 0

                if position_size > 0:
                    position = {
                        "entry_price": signal.entry_price,
                        "amount": position_size,
                        "stop_loss": signal.stop_loss,
                        "take_profit_1": signal.take_profit_1,
                        "take_profit_2": signal.take_profit_2,
                        "take_profit_3": signal.take_profit_3,
                    }

                    current_trade = {
                        "type": "buy",
                        "index": i,
                        "price": signal.entry_price,
                        "amount": position_size,
                        "stop_loss": signal.stop_loss,
                        "take_profit_1": signal.take_profit_1,
                        "take_profit_2": signal.take_profit_2,
                        "take_profit_3": signal.take_profit_3,
                        "risk_percentage": risk_percentage,
                    }
                    trade_log.append(current_trade)

    # If still holding a position at the end, liquidate at the final price
    if position:
        final_price = df["Close"].iloc[-1]
        final_time = df.index[-1]
        final_profit = (final_price - position["entry_price"]) * position["amount"]
        balance += final_profit if not np.isnan(final_profit) else 0.0
        trade_log.append(
            {
                "type": "sell_end",
                "index": len(df) - 1,
                "price": final_price,
                "amount": position["amount"],
                "profit": final_profit,
                "balance": balance,
            }
        )
        print(f"[Final] SELL at {final_price:.2f}")

        # Store final trade in database if available
        if db:
            trade_data = {
                "iteration_id": iteration_id,
                "sub_iteration_id": sub_iteration_id,
                "symbol": symbol,
                "interval": interval,
                "trade_type": "sell_end",
                "entry_timestamp": df.index[0],
                "exit_timestamp": final_time,
                "entry_index": df.index[0],
                "exit_index": len(df) - 1,
                "entry_price": position["entry_price"],
                "exit_price": final_price,
                "profit_loss": final_profit,
                "trade_duration": len(df) - 1 - df.index[0],
                "entry_signal": "Bullish",
                "exit_signal": "Final Sell",
                "risk_reward_ratio": (position["risk_reward_ratio"] if position else 0),
                "position_size": position["amount"],
                "stop_loss": position["stop_loss"],
                "take_profit_1": position["take_profit_1"],
                "take_profit_2": position["take_profit_2"],
                "take_profit_3": position["take_profit_3"],
                "risk_percentage": risk_percentage,
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
