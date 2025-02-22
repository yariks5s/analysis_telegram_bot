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
    return [item["name"] for item in data.get("result", [])]

# --- Main Routine: Select Random Pair and Period, then Run Backtest ---
if __name__ == "__main__":
    try:
        # Fetch all trading pairs and randomly select one
        pairs = get_trading_pairs()
        if not pairs:
            raise Exception("No trading pairs found.")
        symbol = random.choice(pairs)
        
        # Randomly select an interval from a list of common intervals
        intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        interval = random.choice(intervals)
        
        # Choose a random total candle count between 300 and 600
        candles = random.randint(300, 600)
        # Choose a window as 60% of candles (ensure window < candles)
        window = int(candles * 0.6)
        
        print(f"Selected Symbol: {symbol}")
        print(f"Interval: {interval}")
        print(f"Total Candles: {candles}, Lookback Window: {window}")
        
        # Run the backtest strategy
        final_balance, trades = backtest_strategy(symbol, interval, candles, window)
        print("\n--- Backtest Complete ---")
        print(f"Final Balance: {final_balance:.2f}")
        print("Trade Log:")
        for trade in trades:
            print(trade)
    
    except Exception as e:
        print("Error:", e)
