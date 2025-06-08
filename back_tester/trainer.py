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
weights = [
    1.0,
    1.0,
    1.0,
    1.0,
    0.7,
    0.7,
    0.5,
    0.5,
    0.8,
    1.2,
    1.2,
    1.5,
    0.6,
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
    """Class to track training metrics"""

    def __init__(self):
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_balance = 0.0
        self.avg_risk_reward = 0.0  # Added missing attribute
        self.avg_position_size = 0.0  # Added missing attribute
        self.tp_success_rate = 0.0  # Added missing attribute
        self.tp1_hits = 0  # Added missing attribute
        self.tp2_hits = 0  # Added missing attribute
        self.tp3_hits = 0  # Added missing attribute
        self.stop_loss_hits = 0  # Added missing attribute
        self.risk_percentage = 0.0  # Added missing attribute

    def update(self, trade_result: Dict[str, Any]):
        """Update metrics with trade result"""
        self.total_trades += 1
        profit = trade_result.get("profit_loss", 0.0)
        self.total_profit += profit

        if profit > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Update drawdown
        current_balance = trade_result.get("balance", 0.0)
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        else:
            self.current_drawdown = (
                self.peak_balance - current_balance
            ) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        # Update risk management metrics
        self.avg_risk_reward = (
            self.avg_risk_reward * (self.total_trades - 1)
            + trade_result.get("risk_reward_ratio", 0.0)
        ) / self.total_trades
        self.avg_position_size = (
            self.avg_position_size * (self.total_trades - 1)
            + trade_result.get("position_size", 0.0)
        ) / self.total_trades

        # Update take profit and stop loss hits
        trade_type = trade_result.get("trade_type", "")
        if trade_type == "tp1":
            self.tp1_hits += 1
        elif trade_type == "tp2":
            self.tp2_hits += 1
        elif trade_type == "tp3":
            self.tp3_hits += 1
        elif trade_type == "stop_loss":
            self.stop_loss_hits += 1

        # Calculate take profit success rate
        total_exits = (
            self.tp1_hits + self.tp2_hits + self.tp3_hits + self.stop_loss_hits
        )
        if total_exits > 0:
            self.tp_success_rate = (
                self.tp1_hits + self.tp2_hits + self.tp3_hits
            ) / total_exits

        # Update risk percentage
        self.risk_percentage = trade_result.get("risk_percentage", 0.0)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics as dictionary"""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_profit": self.total_profit,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.winning_trades / max(1, self.total_trades),
            "avg_risk_reward": self.avg_risk_reward,
            "avg_position_size": self.avg_position_size,
            "tp_success_rate": self.tp_success_rate,
            "tp1_hits": self.tp1_hits,
            "tp2_hits": self.tp2_hits,
            "tp3_hits": self.tp3_hits,
            "stop_loss_hits": self.stop_loss_hits,
            "risk_percentage": self.risk_percentage,
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

    # Weighted combination of metrics
    fitness = (
        0.25 * win_rate
        + 0.15 * (1 - metrics.max_drawdown)
        + 0.15 * min(profit_factor, 5) / 5  # Cap profit factor at 5
        + 0.15 * (metrics.total_profit / 1000)  # Normalize revenue
        + 0.10 * (1 - min(metrics.current_drawdown, 1))  # Current drawdown penalty
        + 0.10 * tp_success_rate  # Take profit success rate
        + 0.10 * min(metrics.avg_risk_reward, 3) / 3  # Risk/reward ratio (capped at 3)
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
                        metrics.update(trades)

                        # Store sub-iteration data
                        if db and sub_iteration_id:
                            sub_iteration_data = {
                                "iteration_id": iteration_id,
                                "symbol": symbol,
                                "interval": interval,
                                "candles": candles,
                                "window": window,
                                "initial_balance": initial_balance,
                                "final_balance": final_balance,
                                "total_trades": len(trades),
                                "win_rate": (
                                    metrics.winning_trades / metrics.total_trades * 100
                                    if metrics.total_trades > 0
                                    else 0
                                ),
                                "revenue": final_balance - initial_balance,
                                "risk_percentage": risk_percentage,
                                "avg_risk_reward": metrics.get_metrics()[
                                    "avg_risk_reward"
                                ],
                                "tp_success_rate": metrics.get_metrics()[
                                    "tp_success_rate"
                                ],
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

    for i in range(iterations):
        # Generate random perturbations
        perturbations = np.random.normal(0, 0.1, len(weights))

        # Apply perturbations to weights
        test_weights = [w + p for w, p in zip(weights, perturbations)]

        # Evaluate fitness
        fitness, metrics = evaluate_weights(test_weights)

        logger.info(f"Iteration {i}: New fitness {fitness:.2f}")
        if metrics:
            logger.info(f"Metrics: {metrics.get_metrics()}")

        if fitness > best_fitness:
            best_fitness = fitness
            best_weights = test_weights.copy()
            no_improvement_count = 0

            # Insert iteration data into database
            if metrics:
                iteration_data = {
                    "iteration_id": str(uuid.uuid4()),
                    "iteration_number": i,
                    "weights": test_weights,
                    "fitness_score": fitness,
                    "total_trades": metrics.total_trades,
                    "win_rate": metrics.winning_trades
                    / max(1, metrics.total_trades)
                    * 100,
                    "total_revenue": metrics.total_profit,
                    "max_drawdown": metrics.max_drawdown,
                    "avg_profit": (
                        metrics.total_profit / max(1, metrics.winning_trades)
                        if metrics.winning_trades > 0
                        else 0
                    ),
                    "avg_loss": (
                        abs(metrics.total_profit) / max(1, metrics.losing_trades)
                        if metrics.losing_trades > 0
                        else 0
                    ),
                    "profit_factor": (
                        abs(metrics.total_profit) / max(1, abs(metrics.total_profit))
                        if metrics.total_profit < 0
                        else float("inf")
                    ),
                    "avg_trade_duration": 0,  # This would need to be tracked in metrics
                    "tp1_hits": metrics.tp1_hits,
                    "tp2_hits": metrics.tp2_hits,
                    "tp3_hits": metrics.tp3_hits,
                    "stop_loss_hits": metrics.stop_loss_hits,
                    "avg_risk_reward": metrics.avg_risk_reward,
                    "avg_position_size": metrics.avg_position_size,
                    "risk_percentage": metrics.risk_percentage,
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
            weights[j] += velocity[j]

        # Adjust learning rate
        if len(history) >= history_size:
            if all(f < fitness for f in history):
                learning_rate = max(learning_rate * 0.95, min_learning_rate)
            elif all(f > fitness for f in history):
                learning_rate = min(learning_rate * 1.05, max_learning_rate)

        history.append(fitness)

    return best_weights


if __name__ == "__main__":
    # Train the model
    best_weights = optimize_weights(weights, iterations, learning_rate)
    logger.info(f"Best weights found: {best_weights}")

    # Evaluate final performance
    fitness, metrics = evaluate_weights(best_weights)
    if metrics:
        logger.info(f"Final metrics: {metrics.get_metrics()}")
