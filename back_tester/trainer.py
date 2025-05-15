import random
import os
import sys
import numpy as np  # type: ignore
from typing import List, Tuple, Optional, Dict
import logging
from datetime import datetime
from collections import deque

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from strategy import backtest_strategy
from getTradingPairs import get_trading_pairs

# Set up focused logging for training
logger = logging.getLogger("training")
logger.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Create file handler for training logs
training_log_file = f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
file_handler = logging.FileHandler(training_log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Create console handler for training logs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Prevent propagation to root logger
logger.propagate = False

# --- Initial Weight Configuration ---
weights = [1.0, 1.0, 1.0, 1.0, 0.7, 0.7, 0.5, 0.5]
learning_rate = 0.05
iterations = 1000
evaluation_runs = 20
min_candles_required = 300
max_candles = 600

# Optimization parameters
momentum = 0.9
velocity = [0.0] * len(weights)
min_learning_rate = 0.001
max_learning_rate = 0.1
patience = 50
history_size = 10


class TrainingMetrics:
    def __init__(self):
        self.total_trades = 0
        self.winning_trades = 0
        self.total_revenue = 0
        self.max_drawdown = 0
        self.current_drawdown = 0
        self.peak_balance = 0
        self.trade_durations = []
        self.profit_per_trade = []
        self.loss_per_trade = []

    def update(
        self, trade_log: List[dict], initial_balance: float, final_balance: float
    ):
        self.total_trades += len(trade_log)
        self.total_revenue += final_balance - initial_balance

        if trade_log:
            entry_prices = [t["price"] for t in trade_log if t["type"] == "buy"]
            exit_prices = [
                t["price"] for t in trade_log if t["type"] in ["sell", "sell_end"]
            ]
            entry_indices = [t["index"] for t in trade_log if t["type"] == "buy"]
            exit_indices = [
                t["index"] for t in trade_log if t["type"] in ["sell", "sell_end"]
            ]

            for entry, exit, entry_idx, exit_idx in zip(
                entry_prices, exit_prices, entry_indices, exit_indices
            ):
                profit = exit - entry
                if profit > 0:
                    self.winning_trades += 1
                    self.profit_per_trade.append(profit)
                else:
                    self.loss_per_trade.append(profit)
                self.trade_durations.append(exit_idx - entry_idx)

        if final_balance > self.peak_balance:
            self.peak_balance = final_balance
            self.current_drawdown = 0
        else:
            self.current_drawdown = (
                self.peak_balance - final_balance
            ) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

    def get_metrics(self) -> dict:
        win_rate = (
            (self.winning_trades / self.total_trades * 100)
            if self.total_trades > 0
            else 0
        )
        avg_profit = np.mean(self.profit_per_trade) if self.profit_per_trade else 0
        avg_loss = np.mean(self.loss_per_trade) if self.loss_per_trade else 0
        avg_duration = np.mean(self.trade_durations) if self.trade_durations else 0
        profit_factor = (
            abs(sum(self.profit_per_trade) / sum(self.loss_per_trade))
            if self.loss_per_trade and sum(self.loss_per_trade) != 0
            else float("inf")
        )

        return {
            "total_trades": self.total_trades,
            "win_rate": win_rate,
            "total_revenue": self.total_revenue,
            "max_drawdown": self.max_drawdown * 100,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_trade_duration": avg_duration,
        }


def calculate_fitness(metrics: TrainingMetrics) -> float:
    """Calculate a comprehensive fitness score based on multiple metrics"""
    if metrics.total_trades == 0:
        return -9999

    # Normalize metrics
    win_rate = metrics.winning_trades / metrics.total_trades
    profit_factor = (
        abs(sum(metrics.profit_per_trade) / sum(metrics.loss_per_trade))
        if metrics.loss_per_trade and sum(metrics.loss_per_trade) != 0
        else 1
    )

    # Weighted combination of metrics
    fitness = (
        0.3 * win_rate
        + 0.2 * (1 - metrics.max_drawdown)
        + 0.2 * min(profit_factor, 5) / 5  # Cap profit factor at 5
        + 0.2 * (metrics.total_revenue / 1000)  # Normalize revenue
        + 0.1 * (1 - min(metrics.current_drawdown, 1))  # Current drawdown penalty
    )

    return fitness * 100  # Scale to percentage


def evaluate_weights(
    weights: List[float], test_pairs: Optional[List[str]] = None
) -> Tuple[float, Optional[TrainingMetrics]]:
    try:
        pairs = test_pairs if test_pairs else get_trading_pairs()
        if not pairs:
            logger.error("No trading pairs found")
            return -9999, None

        num_pairs = min(5, len(pairs))
        selected_pairs = random.sample(pairs, num_pairs)
        intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        initial_balance = 1000.0
        metrics = TrainingMetrics()

        successful_runs = 0
        total_revenue_percent = 0

        for _ in range(evaluation_runs):
            for symbol in selected_pairs:
                interval = random.choice(intervals)
                candles = random.randint(min_candles_required, max_candles)
                window = int(candles * 0.6)

                try:
                    final_balance, trades = backtest_strategy(
                        symbol,
                        interval,
                        candles,
                        window,
                        initial_balance,
                        weights=weights,
                    )

                    if trades:  # Only count runs with actual trades
                        successful_runs += 1
                        revenue_percent = (
                            (final_balance - initial_balance) / initial_balance
                        ) * 100
                        total_revenue_percent += revenue_percent
                        metrics.update(trades, initial_balance, final_balance)

                except Exception as e:
                    logger.warning(
                        f"Error in backtest for {symbol} {interval}: {str(e)}"
                    )
                    continue

        if successful_runs == 0:
            logger.warning("No successful runs in evaluation")
            return -9999, None

        avg_revenue_percent = total_revenue_percent / successful_runs
        fitness_score = calculate_fitness(metrics)
        return fitness_score, metrics

    except Exception as e:
        logger.error(f"Error in evaluate_weights: {str(e)}")
        return -9999, None


def optimize_weights(
    weights: List[float], iterations: int, learning_rate: float
) -> List[float]:
    best_weights = weights[:]
    best_score, best_metrics = evaluate_weights(best_weights)

    if best_metrics:
        logger.info(f"Initial Score: {best_score:.5f}%")
        logger.info(f"Initial Metrics: {best_metrics.get_metrics()}")
        logger.info(f"Initial Weights: {best_weights}")

    # Optimization state
    current_learning_rate = learning_rate
    no_improvement_count = 0
    score_history = deque(maxlen=history_size)
    velocity = [0.0] * len(weights)

    for iteration in range(iterations):
        # Randomly select multiple weights to adjust
        num_weights_to_adjust = random.randint(1, 3)
        indices = random.sample(range(len(weights)), num_weights_to_adjust)

        new_weights = best_weights[:]
        for index in indices:
            # Momentum-based weight adjustment
            gradient = random.choice([-1, 1])
            velocity[index] = momentum * velocity[index] + (1 - momentum) * gradient
            adjustment = current_learning_rate * velocity[index]
            new_weights[index] = max(0, min(2.0, new_weights[index] + adjustment))

        new_score, new_metrics = evaluate_weights(new_weights)
        score_history.append(new_score)

        if new_score > best_score:
            best_weights = new_weights
            best_score = new_score
            best_metrics = new_metrics
            no_improvement_count = 0
            logger.info(f"Iteration {iteration + 1}: New Best Score: {best_score:.5f}%")
            logger.info(
                f"Metrics: {best_metrics.get_metrics() if best_metrics else 'N/A'}"
            )
            logger.info(f"Weights: {best_weights}")
        else:
            no_improvement_count += 1

            # Adaptive learning rate adjustment
            if no_improvement_count >= patience:
                if len(score_history) >= history_size:
                    recent_trend = np.mean(list(score_history)[-5:]) - np.mean(
                        list(score_history)[:-5]
                    )
                    if recent_trend < 0:
                        current_learning_rate = max(
                            min_learning_rate, current_learning_rate * 0.5
                        )
                    else:
                        current_learning_rate = min(
                            max_learning_rate, current_learning_rate * 1.1
                        )
                no_improvement_count = 0
                logger.info(f"Adjusting learning rate to {current_learning_rate}")

        # Early stopping if we've found a good solution
        if best_score > 50 and no_improvement_count > patience * 2:
            logger.info("Early stopping: Found good solution")
            break

    logger.info("\nOptimization Complete!")
    logger.info(f"Final Optimized Score: {best_score:.5f}%")
    if best_metrics:
        logger.info(f"Final Metrics: {best_metrics.get_metrics()}")
    logger.info(f"Optimized Weights: {best_weights}")

    return best_weights


if __name__ == "__main__":
    try:
        optimized_weights = optimize_weights(weights, iterations, learning_rate)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)
