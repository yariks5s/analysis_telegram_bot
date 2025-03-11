import requests # type: ignore
import random
import os
import sys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from strategy import backtest_strategy

def get_trading_pairs():
    url = "https://api.bybit.com/spot/v1/symbols"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch trading pairs.")
    data = response.json()
    return [item["name"] for item in data.get("result", [])] # add this to getTradingPairs.py

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
            
            # Run the backtest
            final_balance, trades = backtest_strategy(symbol, interval, candles, window, initial_balance)

            # Calculate revenue percentage for this test
            revenue_percent = ((final_balance - initial_balance) / initial_balance) * 100
            total_revenue_percent += revenue_percent  # Accumulate total revenue percentage

            # Store results
            results.append({
                "symbol": symbol,
                "interval": interval,
                "final_balance": final_balance,
                "num_trades": len(trades),
                "revenue_percent": revenue_percent,
                "trades": trades
            })
            
            # Print results
            print("\n--- Backtest Complete ---")
            print(f"Final Balance: {final_balance:.2f}")
            print(f"Total Trades: {len(trades)}")
            print(f"Revenue %: {revenue_percent:.2f}%")
            print("Trade Log (First 5 trades shown):")
            for trade in trades[:5]:  # Display only the first 5 trades for readability
                print(trade)
            print("--------------------------------------")

        print("\n====== Summary of All Backtests ======")
        for res in results:
            print(f"Pair: {res['symbol']} | Interval: {res['interval']} | Final Balance: {res['final_balance']:.2f} | Trades: {res['num_trades']} | Revenue %: {res['revenue_percent']:.2f}%")

        # Compute the average revenue percentage
        average_revenue_percent = total_revenue_percent / len(results)
        print(f"\nTotal Revenue % across all backtests: {average_revenue_percent:.2f}%")
    
    except Exception as e:
        print("Error:", e)
