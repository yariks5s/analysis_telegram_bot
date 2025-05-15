import os
import sys
from typing import List
import random
from datetime import datetime, timedelta

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

import pandas as pd  # type: ignore
from data_fetching_instruments import fetch_candles, analyze_data
from signal_detection import generate_price_prediction_signal_proba
from utils import create_true_preferences


def backtest_strategy(  # TODO: LOGGING INSTEAD OF PRINTING
    symbol: str,
    interval: str,
    candles: int = 1000,
    window: int = 300,
    initial_balance: float = 10000.0,
    liq_lev_tolerance: float = 0.05,
    weights: list = [],
) -> (float, list):  # type: ignore
    """
    Backtest a simple strategy:
      - At each step (after an initial window of historical data), generate a signal
      - If the signal is Bullish and you are not in the market, buy (go all in)
      - If the signal is Bearish and you are in a position, sell (go to cash)

    Parameters:
      symbol: The trading pair symbol (e.g. "BTCUSDT")
      interval: The candle interval (e.g. "1h")
      candles: Total number of candles to fetch
      window: Lookback period (number of candles) to use for generating the signal
      initial_balance: Starting capital in USD (or your base currency)
      liq_lev_tolerance: Tolerance for liquidity level detection

    Returns:
      final_balance: The simulated portfolio balance at the end
      trade_log: A list of trade events for review
    """

    # Determine a random start time for fetching candles
    end_time = datetime.now()
    start_time = end_time - timedelta(days=730)  # Assuming 2 years of data availability
    random_start_time = start_time + timedelta(
        seconds=random.randint(0, int((end_time - start_time).total_seconds()))
    )

    # Fetch candles starting from the random start time
    df = fetch_candles(symbol, candles, interval, start_time=random_start_time)
    if df is None or df.empty:
        raise ValueError("No data returned for backtesting.")

    # Ensure the DataFrame is sorted by time
    df.sort_index(inplace=True)

    balance = initial_balance
    position = 0.0  # quantity of asset held
    entry_price = None
    trade_log = []  # logs of trades for later analysis

    # Use all indicators enabled by default in backtesting
    preferences = create_true_preferences()

    for i in range(window, len(df)):
        current_window = df.iloc[i - window : i]

        # Compute technical indicators and generate a signal
        indicators = analyze_data(current_window, preferences, liq_lev_tolerance)
        signal, prob, confidence, reason = generate_price_prediction_signal_proba(
            current_window, indicators, weights
        )

        # Use the current close price as the trade price
        price = df["Close"].iloc[i]

        # Simple trading logic: go long when Bullish; exit when Bearish.
        if signal == "Bullish" and position == 0:
            # Buy signal: use all available balance to buy the asset
            position = balance / price
            entry_price = price
            balance = 0  # all-in
            trade_log.append(
                {"type": "buy", "price": entry_price, "index": i, "signal": signal}
            )
            print(
                f"[Index {i}] BUY at {entry_price:.5f} | Reason: {reason.splitlines()[0]}"
            )
        elif signal == "Bearish" and position > 0:
            # Sell signal: liquidate the position
            balance = position * price
            trade_log.append(
                {"type": "sell", "price": price, "index": i, "signal": signal}
            )
            print(f"[Index {i}] SELL at {price:.5f} | Reason: {reason.splitlines()[0]}")
            position = 0
            entry_price = None

    # If still holding a position at the end, liquidate at the final price.
    if position > 0:
        final_price = df["Close"].iloc[-1]
        balance = position * final_price
        trade_log.append(
            {
                "type": "sell_end",
                "price": final_price,
                "index": len(df) - 1,
                "signal": "Final Sell",
            }
        )
        print(f"[Final] SELL at {final_price:.2f}")

    return balance, trade_log


if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = "1h"
    final_balance, trades = backtest_strategy(symbol, interval, candles=600, window=300)
    print(f"Final Balance: {final_balance:.2f}")
    print("Trade Log:")
    for trade in trades:
        print(trade)
