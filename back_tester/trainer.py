import random
import os
import sys
import numpy as np # type: ignore

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from strategy import backtest_strategy
from getTradingPairs import get_trading_pairs

# --- Initial Weight Configuration ---
weights = [1.0, 1.0, 1.0, 1.0, 0.7, 0.7, 0.5, 0.5]
learning_rate = 0.05  # Step size for adjusting weights
iterations = 1000  # Number of iterations for tuning
evaluation_runs = 20  # How many tests per weight set


# --- Function to evaluate weights ---
def evaluate_weights(weights):
    try:
        pairs = get_trading_pairs()
        if not pairs:
            raise Exception("No trading pairs found.")

        num_pairs = min(5, len(pairs))
        selected_pairs = random.sample(pairs, num_pairs)
        intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        initial_balance = 1000.0

        total_revenue_percent = 0

        for _ in range(evaluation_runs):
            for symbol in selected_pairs:
                interval = random.choice(intervals)
                candles = random.randint(300, 600)
                window = int(candles * 0.6)

                final_balance, trades = backtest_strategy(
                    symbol, interval, candles, window, initial_balance, weights=weights
                )

                revenue_percent = (
                    (final_balance - initial_balance) / initial_balance
                ) * 100
                total_revenue_percent += revenue_percent

        avg_revenue_percent = total_revenue_percent / (evaluation_runs * num_pairs)
        return avg_revenue_percent

    except Exception as e:
        print("Error:", e)
        return -9999  # Return a large negative value in case of failure


# --- AI Optimization Process ---
def optimize_weights(weights, iterations, learning_rate):
    best_weights = weights[:]
    best_score = evaluate_weights(best_weights)

    print(f"Initial Score: {best_score:.2f}% with weights: {best_weights}")

    for _ in range(iterations):
        index = random.randint(0, len(weights) - 1)  # Choose a random weight to adjust
        adjustment = random.choice([-learning_rate, learning_rate])  # Adjust up or down
        new_weights = best_weights[:]
        new_weights[index] += adjustment

        # Ensure weights stay within a reasonable range (0 to 2.0)
        new_weights[index] = max(0, min(2.0, new_weights[index]))

        new_score = evaluate_weights(new_weights)

        if new_score > best_score:  # If the new weights perform better, keep them
            best_weights = new_weights
            best_score = new_score
            print(f"New Best Score: {best_score:.2f}% with weights: {best_weights}")

    print("\nOptimization Complete!")
    print(f"Final Optimized Score: {best_score:.2f}%")
    print(f"Optimized Weights: {best_weights}")

    return best_weights


if __name__ == "__main__":
    optimized_weights = optimize_weights(weights, iterations, learning_rate)
