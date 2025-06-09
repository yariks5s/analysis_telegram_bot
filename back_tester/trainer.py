import random
import os
import sys
import numpy as np  # type: ignore
from typing import List, Tuple, Optional, Dict, Any
import logging
from datetime import datetime
from collections import deque
import uuid

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from strategy import backtest_strategy
from getTradingPairs import get_trading_pairs
from db_operations import ClickHouseDB

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

# Initialize ClickHouse database
db = ClickHouseDB()

# --- Initial Weight Configuration ---
# Updated for new weights including risk management signals
weights = [
    1.0,  # W_BULLISH_OB
    1.0,  # W_BEARISH_OB
    1.0,  # W_BULLISH_BREAKER
    1.0,  # W_BEARISH_BREAKER
    0.7,  # W_ABOVE_SUPPORT
    0.7,  # W_BELOW_RESISTANCE
    0.5,  # W_FVG_ABOVE
    0.5,  # W_FVG_BELOW
    0.8,  # W_TREND
    1.2,  # W_SWEEP_HIGHS
    1.2,  # W_SWEEP_LOWS
    1.5,  # W_STRUCTURE_BREAK
    0.6,  # W_PIN_BAR
]  # Updated for new weights

learning_rate = 0.05
iterations = 1000
evaluation_runs = 20
min_candles_required = 300
max_candles = 600
risk_percentage = 1.0  # Default risk per trade

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
        self.tp1_hits = 0
        self.tp2_hits = 0
        self.tp3_hits = 0
        self.stop_loss_hits = 0
        self.risk_reward_ratios = []
        self.position_sizes = []

    def update(
        self, trade_log: List[dict], initial_balance: float, final_balance: float
    ):
        self.total_revenue += final_balance - initial_balance

        if trade_log:
            # Track take profit and stop loss hits
            for trade in trade_log:
                if trade["type"] == "take_profit_1":
                    self.tp1_hits += 1
                elif trade["type"] == "take_profit_2":
                    self.tp2_hits += 1
                elif trade["type"] == "take_profit_3":
                    self.tp3_hits += 1
                elif trade["type"] == "stop_loss":
                    self.stop_loss_hits += 1

            # Calculate profits and losses for completed trades
            entry_trades = [t for t in trade_log if t["type"] == "entry"]
            self.total_trades += len(entry_trades)

            for entry_trade in entry_trades:
                # Find all exit trades for this entry
                entry_idx = entry_trade["index"]
                exit_trades = [
                    t
                    for t in trade_log
                    if t["type"]
                    in [
                        "take_profit_1",
                        "take_profit_2",
                        "take_profit_3",
                        "stop_loss",
                        "exit_end_of_period",
                    ]
                    and t["index"] > entry_idx
                ]

                if exit_trades:
                    # Calculate total profit/loss for this trade
                    total_profit = sum(t.get("profit", 0) for t in exit_trades)

                    if total_profit > 0:
                        self.winning_trades += 1
                        self.profit_per_trade.append(total_profit)
                    else:
                        self.loss_per_trade.append(total_profit)

                    # Track trade duration (to the last exit)
                    last_exit = max(exit_trades, key=lambda x: x["index"])
                    self.trade_durations.append(last_exit["index"] - entry_idx)

                    # Track risk/reward ratio and position size
                    if "stop_loss" in entry_trade:
                        entry_price = entry_trade["price"]
                        stop_loss = entry_trade["stop_loss"]
                        tp2 = entry_trade.get("take_profit_2", entry_price)
                        risk = abs(entry_price - stop_loss)
                        reward = abs(tp2 - entry_price)
                        if risk > 0:
                            self.risk_reward_ratios.append(reward / risk)

                    if "amount" in entry_trade:
                        self.position_sizes.append(entry_trade["amount"])

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
        avg_risk_reward = (
            np.mean(self.risk_reward_ratios) if self.risk_reward_ratios else 0
        )
        avg_position_size = np.mean(self.position_sizes) if self.position_sizes else 0

        return {
            "total_trades": self.total_trades,
            "win_rate": win_rate,
            "total_revenue": self.total_revenue,
            "max_drawdown": self.max_drawdown * 100,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_trade_duration": avg_duration,
            "tp1_hits": self.tp1_hits,
            "tp2_hits": self.tp2_hits,
            "tp3_hits": self.tp3_hits,
            "stop_loss_hits": self.stop_loss_hits,
            "avg_risk_reward": avg_risk_reward,
            "avg_position_size": avg_position_size,
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

    # Calculate take profit success rate
    total_exits = (
        metrics.tp1_hits + metrics.tp2_hits + metrics.tp3_hits + metrics.stop_loss_hits
    )
    tp_success_rate = (
        (metrics.tp1_hits + metrics.tp2_hits + metrics.tp3_hits) / total_exits
        if total_exits > 0
        else 0
    )

    # Get average risk/reward from metrics
    avg_risk_reward = (
        np.mean(metrics.risk_reward_ratios) if metrics.risk_reward_ratios else 0
    )

    # Weighted combination of metrics
    fitness = (
        0.25 * win_rate
        + 0.15 * (1 - metrics.max_drawdown)
        + 0.15 * min(profit_factor, 5) / 5  # Cap profit factor at 5
        + 0.15 * (metrics.total_revenue / 1000)  # Normalize revenue
        + 0.10 * (1 - min(metrics.current_drawdown, 1))  # Current drawdown penalty
        + 0.10 * tp_success_rate  # Take profit success rate
        + 0.10 * min(avg_risk_reward, 3) / 3  # Risk/reward ratio (capped at 3)
    )

    return fitness * 100  # Scale to percentage


def evaluate_weights(
    weights: List[float],
    test_pairs: Optional[List[str]] = None,
    iteration_id: Optional[str] = None,
) -> Tuple[float, Optional[TrainingMetrics]]:
    try:
        pairs = test_pairs if test_pairs else get_trading_pairs()
        if not pairs:
            logger.error("No trading pairs found")
            return -9999, None

        num_pairs = min(30, len(pairs))
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
                    final_balance, trades, sub_iteration_id = backtest_strategy(
                        symbol,
                        interval,
                        candles,
                        window,
                        initial_balance,
                        weights=weights,
                        risk_percentage=risk_percentage,
                        iteration_id=iteration_id,
                        db=db,
                    )

                    if trades:  # Only count runs with actual trades
                        successful_runs += 1
                        revenue_percent = (
                            (final_balance - initial_balance) / initial_balance
                        ) * 100
                        total_revenue_percent += revenue_percent
                        metrics.update(trades, initial_balance, final_balance)

                        # Store sub-iteration data
                        if db and sub_iteration_id:
                            metrics_data = metrics.get_metrics()

                            # Calculate TP success rate for sub-iteration
                            total_exits = (
                                metrics.tp1_hits
                                + metrics.tp2_hits
                                + metrics.tp3_hits
                                + metrics.stop_loss_hits
                            )
                            tp_success_rate = (
                                (metrics.tp1_hits + metrics.tp2_hits + metrics.tp3_hits)
                                / total_exits
                                if total_exits > 0
                                else 0
                            )

                            sub_iteration_data = {
                                "iteration_id": iteration_id,
                                "symbol": symbol,
                                "interval": interval,
                                "candles": candles,
                                "window": window,
                                "initial_balance": initial_balance,
                                "final_balance": final_balance,
                                "total_trades": len(
                                    [t for t in trades if t["type"] == "entry"]
                                ),
                                "win_rate": metrics_data["win_rate"],
                                "revenue": final_balance - initial_balance,
                                "risk_percentage": risk_percentage,
                                "avg_risk_reward": metrics_data["avg_risk_reward"],
                                "tp_success_rate": tp_success_rate,
                            }
                            db.insert_sub_iteration(sub_iteration_data)

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
    """Optimize weights using gradient descent with momentum"""
    best_weights = weights.copy()
    best_fitness = -float("inf")
    no_improvement_count = 0
    history = deque(maxlen=history_size)
    iteration_id = str(uuid.uuid4())

    # Initial evaluation
    fitness, metrics = evaluate_weights(best_weights, iteration_id=iteration_id)
    if fitness > best_fitness:
        best_fitness = fitness
        logger.info(f"Initial fitness: {fitness:.2f}")
        if metrics:
            logger.info(f"Initial metrics: {metrics.get_metrics()}")

            # Store initial iteration
            iteration_data = {
                "iteration_id": iteration_id,
                "iteration_number": 0,
                "weights": best_weights,
                "fitness_score": fitness,
                **metrics.get_metrics(),
                "risk_percentage": risk_percentage,
            }
            db.insert_iteration(iteration_data)

    for i in range(iterations):
        # Generate random perturbations
        perturbations = np.random.normal(0, 0.1, len(weights))

        # Apply perturbations to weights
        test_weights = [max(0, min(2.0, w + p)) for w, p in zip(weights, perturbations)]

        # Evaluate fitness
        fitness, metrics = evaluate_weights(test_weights, iteration_id=iteration_id)

        if fitness > best_fitness:
            best_fitness = fitness
            best_weights = test_weights.copy()
            no_improvement_count = 0

            # Log improvement
            logger.info(f"Iteration {i}: New best fitness {fitness:.2f}")
            if metrics:
                logger.info(f"Metrics: {metrics.get_metrics()}")

                # Store iteration data
                iteration_data = {
                    "iteration_id": iteration_id,
                    "iteration_number": i + 1,
                    "weights": test_weights,
                    "fitness_score": fitness,
                    **metrics.get_metrics(),
                    "risk_percentage": risk_percentage,
                }
                db.insert_iteration(iteration_data)
        else:
            no_improvement_count += 1

        # Early stopping
        if no_improvement_count >= patience:
            logger.info(f"Early stopping at iteration {i}")
            break

        # Update weights using momentum
        for j in range(len(weights)):
            velocity[j] = momentum * velocity[j] + learning_rate * perturbations[j]
            weights[j] = max(0, min(2.0, weights[j] + velocity[j]))

        # Adjust learning rate
        if len(history) >= history_size:
            if all(f < fitness for f in history):
                learning_rate = max(learning_rate * 0.95, min_learning_rate)
            elif all(f > fitness for f in history):
                learning_rate = min(learning_rate * 1.05, max_learning_rate)

        history.append(fitness)

    logger.info(f"Best weights found: {best_weights}")
    logger.info(f"Best fitness: {best_fitness:.2f}")

    return best_weights


if __name__ == "__main__":
    # Train the model
    try:
        best_weights = optimize_weights(weights, iterations, learning_rate)
        logger.info(f"Training completed successfully")

        # Evaluate final performance
        fitness, metrics = evaluate_weights(best_weights)
        if metrics:
            logger.info(f"Final metrics: {metrics.get_metrics()}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)
