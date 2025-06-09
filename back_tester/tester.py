import requests  # type: ignore
import random
import os
import sys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from strategy import backtest_strategy
from getTradingPairs import get_trading_pairs

# --- Main Routine: Run Multiple Backtests ---
if __name__ == "__main__":
    try:
        # Fetch all trading pairs
        pairs = get_trading_pairs()
        if not pairs:
            raise Exception("No trading pairs found.")

        # Randomly select a subset of trading pairs (e.g., 5)
        num_pairs = min(5, len(pairs))  # Ensure we don't exceed available pairs
        selected_pairs = random.sample(pairs, num_pairs)

        # Define intervals
        intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

        # Set initial balance (assumed same for each test)
        initial_balance = 1000.0

        # Risk management parameters
        risk_percentage = 1.0  # Risk 1% per trade

        # Run backtest for each selected pair with different intervals
        results = []
        total_revenue_percent = 0

        for symbol in selected_pairs:
            interval = random.choice(intervals)  # Select a random interval
            candles = random.randint(300, 600)  # Choose a random total candle count
            window = int(candles * 0.6)  # Lookback window (60% of total candles)

            print("\n--------------------------------------")
            print(f"Running Backtest for {symbol} | Interval: {interval}")
            print(f"Total Candles: {candles}, Lookback Window: {window}")
            print(f"Risk per trade: {risk_percentage}%")

            # Run the backtest with risk management
            final_balance, trades, _ = backtest_strategy(
                symbol,
                interval,
                candles,
                window,
                initial_balance,
                risk_percentage=risk_percentage,
            )

            # Calculate revenue percentage for this test
            revenue_percent = (
                (final_balance - initial_balance) / initial_balance
            ) * 100
            total_revenue_percent += revenue_percent

            # Count different trade types
            entry_trades = [t for t in trades if t["type"] == "entry"]
            tp1_trades = [t for t in trades if t["type"] == "take_profit_1"]
            tp2_trades = [t for t in trades if t["type"] == "take_profit_2"]
            tp3_trades = [t for t in trades if t["type"] == "take_profit_3"]
            sl_trades = [t for t in trades if t["type"] == "stop_loss"]
            exit_end_trades = [t for t in trades if t["type"] == "exit_end_of_period"]

            # Calculate total profit/loss
            total_profit = sum(t.get("profit", 0) for t in trades if "profit" in t)

            # Store results
            results.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "final_balance": final_balance,
                    "num_trades": len(entry_trades),
                    "tp1_hits": len(tp1_trades),
                    "tp2_hits": len(tp2_trades),
                    "tp3_hits": len(tp3_trades),
                    "stop_losses": len(sl_trades),
                    "end_exits": len(exit_end_trades),
                    "revenue_percent": revenue_percent,
                    "total_profit": total_profit,
                    "trades": trades,
                }
            )

            # Print results
            print("\n--- Backtest Complete ---")
            print(f"Final Balance: ${final_balance:.2f}")
            print(f"Total Entry Trades: {len(entry_trades)}")
            print(
                f"Take Profit Hits - TP1: {len(tp1_trades)}, TP2: {len(tp2_trades)}, TP3: {len(tp3_trades)}"
            )
            print(f"Stop Loss Hits: {len(sl_trades)}")
            print(f"End of Period Exits: {len(exit_end_trades)}")
            print(f"Total Profit/Loss: ${total_profit:.2f}")
            print(f"Revenue %: {revenue_percent:.2f}%")

            # Show sample trades
            print("\nSample Trades (First 5 shown):")
            sample_trades = trades[:5]
            for trade in sample_trades:
                if trade["type"] == "entry":
                    print(
                        f"  ENTRY @ {trade['price']:.5f} | Amount: {trade['amount']:.5f} | "
                        f"SL: {trade['stop_loss']:.5f} | TP1: {trade['take_profit_1']:.5f} | "
                        f"TP2: {trade['take_profit_2']:.5f} | TP3: {trade['take_profit_3']:.5f}"
                    )
                elif trade["type"] in [
                    "take_profit_1",
                    "take_profit_2",
                    "take_profit_3",
                ]:
                    print(
                        f"  {trade['type'].upper()} @ {trade['price']:.5f} | "
                        f"Amount: {trade['amount']:.5f} | Profit: ${trade.get('profit', 0):.2f}"
                    )
                elif trade["type"] == "stop_loss":
                    print(
                        f"  STOP LOSS @ {trade['price']:.5f} | "
                        f"Amount: {trade['amount']:.5f} | Loss: ${trade.get('profit', 0):.2f}"
                    )
                elif trade["type"] == "exit_end_of_period":
                    print(
                        f"  END EXIT @ {trade['price']:.5f} | "
                        f"Amount: {trade['amount']:.5f} | P/L: ${trade.get('profit', 0):.2f}"
                    )

            print("--------------------------------------")

        print("\n====== Summary of All Backtests ======")
        for res in results:
            print(
                f"Pair: {res['symbol']} | Interval: {res['interval']} | "
                f"Final Balance: ${res['final_balance']:.2f} | "
                f"Trades: {res['num_trades']} | "
                f"TP Hits: {res['tp1_hits']+res['tp2_hits']+res['tp3_hits']} | "
                f"SL Hits: {res['stop_losses']} | "
                f"Revenue %: {res['revenue_percent']:.2f}%"
            )

        # Compute the average revenue percentage
        average_revenue_percent = total_revenue_percent / len(results)

        # Calculate overall statistics
        total_trades = sum(res["num_trades"] for res in results)
        total_tp_hits = sum(
            res["tp1_hits"] + res["tp2_hits"] + res["tp3_hits"] for res in results
        )
        total_sl_hits = sum(res["stop_losses"] for res in results)

        print(f"\n======= Overall Statistics =======")
        print(f"Average Revenue %: {average_revenue_percent:.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Total TP Hits: {total_tp_hits}")
        print(f"Total SL Hits: {total_sl_hits}")
        if total_trades > 0:
            tp_rate = (
                (total_tp_hits / (total_tp_hits + total_sl_hits) * 100)
                if (total_tp_hits + total_sl_hits) > 0
                else 0
            )
            print(f"TP Success Rate: {tp_rate:.1f}%")

    except Exception as e:
        print("Error:", e)
        import traceback

        traceback.print_exc()
