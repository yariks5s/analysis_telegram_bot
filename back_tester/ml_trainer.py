import os
import sys
import random
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from collections import deque
import uuid
from typing import List, Dict, Any, Tuple, Optional
import joblib

# Add project root to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from back_tester.ml_signal_enhancement import MLSignalEnhancer
from back_tester.enhanced_backtester import EnhancedBacktester
from back_tester.db_operations import ClickHouseDB

# Set up logging
logger = logging.getLogger("ml_training")
logger.setLevel(logging.INFO)

# Create formatters and handlers (similar to trainer.py)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
training_log_file = f'logs/ml_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
os.makedirs('logs', exist_ok=True)

file_handler = logging.FileHandler(training_log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False

# Initialize database
db = ClickHouseDB()

# Initial ML hyperparameters (similar to weights in trainer.py)
ml_params = {
    "model_type": "random_forest",  # Model algorithm
    "n_estimators": 100,            # Number of trees
    "max_depth": 10,                # Max tree depth
    "min_samples_split": 2,         # Min samples to split
    "min_samples_leaf": 1,          # Min samples in leaf
    "feature_selection": "kbest",   # Feature selection method
    "n_features": 20,               # Number of features to select
    "prediction_horizon": 12,       # Periods ahead to predict
    "threshold": 0.01,              # Price movement threshold
    "ml_threshold": 0.7             # Signal confidence threshold
}

# Training parameters
iterations = 50
patience = 10
history_size = 5


class MLTrainingMetrics:
    """Similar to TrainingMetrics in trainer.py but for ML models"""
    def __init__(self):
        self.total_trades = 0
        self.winning_trades = 0
        self.win_rate = 0.0
        self.total_revenue = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.profit_factor = 0.0
        self.ml_accuracy = 0.0
        self.ml_precision = 0.0
        self.ml_recall = 0.0
        self.ml_f1 = 0.0
    
    def update(self, backtest_metrics: Dict[str, Any], ml_metrics: Dict[str, Any]):
        """Update metrics from backtest and ML training results"""
        # Update backtest metrics
        self.total_trades = backtest_metrics.get('total_trades', 0)
        self.winning_trades = backtest_metrics.get('winning_trades', 0)
        self.win_rate = backtest_metrics.get('win_rate', 0.0)
        self.total_revenue = backtest_metrics.get('total_return', 0.0)
        self.max_drawdown = backtest_metrics.get('max_drawdown', 0.0)
        self.sharpe_ratio = backtest_metrics.get('sharpe_ratio', 0.0)
        self.sortino_ratio = backtest_metrics.get('sortino_ratio', 0.0)
        self.profit_factor = backtest_metrics.get('profit_factor', 0.0)
        
        # Update ML metrics
        self.ml_accuracy = ml_metrics.get('accuracy', 0.0)
        self.ml_precision = ml_metrics.get('precision', 0.0)
        self.ml_recall = ml_metrics.get('recall', 0.0)
        self.ml_f1 = ml_metrics.get('f1', 0.0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return all metrics as a dictionary"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.win_rate,
            'total_revenue': self.total_revenue,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'profit_factor': self.profit_factor,
            'ml_accuracy': self.ml_accuracy,
            'ml_precision': self.ml_precision,
            'ml_recall': self.ml_recall,
            'ml_f1': self.ml_f1
        }


def calculate_fitness(metrics: MLTrainingMetrics) -> float:
    """Calculate fitness score from metrics (similar to calculate_fitness in trainer.py)"""
    # Balance between ML model accuracy and trading performance
    # Convert to int to avoid type comparison issues
    try:
        total_trades = int(metrics.total_trades)
    except (ValueError, TypeError):
        total_trades = 0
        
    if total_trades < 5:
        return -999  # Penalize if too few trades
        
    # Combine ML and trading metrics with weights
    ml_score = (
        0.3 * metrics.ml_precision +  # Focus on precision (avoid false signals)
        0.2 * metrics.ml_recall +     # But also consider recall
        0.5 * metrics.ml_f1           # F1 balances precision and recall
    )
    
    trading_score = (
        0.4 * min(3.0, metrics.sharpe_ratio) +  # Cap Sharpe at 3.0
        0.3 * min(2.0, metrics.profit_factor) + # Cap profit factor at 2.0
        0.2 * min(80, metrics.win_rate) / 100 - # Win rate as proportion
        0.1 * min(30, metrics.max_drawdown) / 100  # Penalize drawdown
    )
    
    # Weight trading performance higher than pure ML metrics
    final_score = 0.3 * ml_score + 0.7 * trading_score
    return final_score


def evaluate_ml_params(
    params: Dict[str, Any],
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    candles: int = 2000,
    initial_balance: float = 10000.0,
    risk_percentage: float = 1.0,
    iteration_id: Optional[str] = None
) -> Tuple[float, Optional[MLTrainingMetrics]]:
    """Evaluate ML parameters through backtesting (similar to evaluate_weights)"""
    try:
        # Debug log parameters and their types
        logger.info(f"Evaluating ML parameters: {params}")
        logger.info(f"Parameter types: symbol={type(symbol)}, interval={type(interval)}, "
                   f"candles={type(candles)}, initial_balance={type(initial_balance)}, "
                   f"risk_percentage={type(risk_percentage)}")
        logger.info(f"ML params types: model_type={type(params.get('model_type'))}, "
                   f"n_estimators={type(params.get('n_estimators'))}, "
                   f"n_features={type(params.get('n_features'))}, "
                   f"ml_threshold={type(params.get('ml_threshold'))}")
                   
        # Ensure all parameters are of the expected type
        candles = int(candles)
        initial_balance = float(initial_balance)
        risk_percentage = float(risk_percentage)
        n_features = int(params.get('n_features', 20))
        ml_threshold = float(params.get('ml_threshold', 0.7))
        metrics = MLTrainingMetrics()
        
        # Initialize ML enhancer with current parameters
        ml_enhancer = MLSignalEnhancer(
            model_type=params["model_type"],
            feature_selection=params["feature_selection"],
            n_features=params["n_features"],
            model_params={
                "n_estimators": params["n_estimators"],
                "max_depth": params["max_depth"],
                "min_samples_split": params["min_samples_split"],
                "min_samples_leaf": params["min_samples_leaf"]
            },
            output_dir="./models/training"
        )
        
        # Initialize backtester
        backtester = EnhancedBacktester(output_dir="./reports/training")
        
        # Get data for training
        data = backtester._fetch_data(symbol, interval, candles)
        
        # Ensure we have numeric values for calculations
        try:
            data_length = int(len(data))
            train_size = int(data_length * 0.7)
            test_size = data_length - train_size
        except (ValueError, TypeError):
            logger.error("Invalid data length or conversion error")
            return -9999, None
        
        # Train on first 70% of data
        try:
            # Ensure data has the expected column names (lowercase)
            # This handles differences between column naming conventions
            training_data = data.iloc[:train_size].copy()
            
            # Standardize column names to lowercase for ML processing
            column_mapping = {
                'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
                'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'
            }
            
            # Only rename columns that actually exist
            rename_dict = {}
            for old_col, new_col in column_mapping.items():
                if old_col in training_data.columns:
                    rename_dict[old_col] = new_col
                    
            if rename_dict:
                training_data.rename(columns=rename_dict, inplace=True)
            
            # Check if we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in training_data.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}. Available columns: {training_data.columns.tolist()}")
                return -9999, None
                
            # Train the ML model
            logger.info(f"Training ML model with {len(training_data)} rows of data")
            ml_results = ml_enhancer.train(
                data=training_data,
                prediction_horizon=params["prediction_horizon"],
                threshold=params["threshold"],
                use_time_series_split=True,
                n_splits=5
            )
        except KeyError as e:
            logger.error(f"KeyError accessing data columns: {str(e)}")
            return -9999, None
        except ValueError as e:
            logger.error(f"ValueError during ML training: {str(e)}")
            return -9999, None
        except Exception as e:
            logger.error(f"Unexpected error during ML training: {str(e)}")
            return -9999, None
        
        # Test on remaining 30% of data
        # Use a fixed value for test_candles to avoid type comparison issues
        # This is a safer approach than using calculated values that might cause type errors
        test_candles = 300  # Use a reasonable fixed size for testing
        
        try:
            final_balance, trades, backtest_metrics = backtester.run_backtest(
                symbol=symbol,
                interval=interval,
                candles=test_candles,
                initial_balance=float(initial_balance),
                risk_percentage=float(risk_percentage),
                use_ml_signals=True,
                ml_threshold=float(params["ml_threshold"])
            )
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            return -9999, None
        
        # Update metrics
        metrics.update(backtest_metrics, ml_results)
        
        # Calculate fitness
        fitness = calculate_fitness(metrics)
        
        # Store results in database if iteration_id provided
        if iteration_id and db:
            iteration_data = {
                "iteration_id": iteration_id,
                "model_type": params["model_type"],
                "params": str(params),
                "fitness_score": fitness,
                **metrics.get_metrics()
            }
            db.insert_ml_iteration(iteration_data)
            
        return fitness, metrics
        
    except Exception as e:
        logger.error(f"Error in evaluate_ml_params: {str(e)}")
        return -9999, None


def optimize_ml_params(
    params: Dict[str, Any], 
    iterations: int
) -> Dict[str, Any]:
    """Optimize ML parameters using similar approach to optimize_weights"""
    best_params = params.copy()
    best_fitness = float("-inf")
    no_improvement_count = 0
    history = deque(maxlen=history_size)
    iteration_id = str(uuid.uuid4())
    
    # Initial evaluation
    fitness, metrics = evaluate_ml_params(best_params, iteration_id=iteration_id)
    if fitness > best_fitness:
        best_fitness = fitness
        logger.info(f"Initial fitness: {fitness:.4f}")
        if metrics:
            logger.info(f"Initial metrics: {metrics.get_metrics()}")
    
    # Parameter ranges for mutation
    param_ranges = {
        "n_estimators": (50, 500),
        "max_depth": (3, 20),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 10),
        "n_features": (10, 50),
        "prediction_horizon": (6, 24),
        "threshold": (0.005, 0.03),
        "ml_threshold": (0.55, 0.9)
    }
    
    # Discrete parameters that should only take certain values
    discrete_params = ["model_type", "feature_selection"]
    model_options = ["random_forest", "gradient_boosting", "logistic"]
    feature_selection_options = ["none", "kbest", "rfe", "pca"]
    
    for i in range(iterations):
        # Start with current best parameters
        test_params = best_params.copy()
        
        # Randomly select parameters to modify (1-3 parameters)
        num_params_to_modify = random.randint(1, 3)
        params_to_modify = random.sample(list(test_params.keys()), num_params_to_modify)
        
        # Modify selected parameters
        for param in params_to_modify:
            if param == "model_type":
                test_params[param] = random.choice(model_options)
            elif param == "feature_selection":
                test_params[param] = random.choice(feature_selection_options)
            elif param in param_ranges:
                min_val, max_val = param_ranges[param]
                
                # Get current value
                current = test_params[param]
                
                # Determine mutation size (smaller for later iterations)
                mutation_factor = max(0.1, 1.0 - (i / iterations))
                
                if isinstance(current, int):
                    # For integer parameters
                    mutation_range = int((max_val - min_val) * mutation_factor)
                    change = random.randint(-mutation_range, mutation_range)
                    test_params[param] = max(min_val, min(max_val, current + change))
                else:
                    # For float parameters
                    mutation_range = (max_val - min_val) * mutation_factor
                    change = random.uniform(-mutation_range, mutation_range)
                    test_params[param] = max(min_val, min(max_val, current + change))
        
        # Evaluate fitness
        logger.info(f"Iteration {i+1}: Testing parameters: {test_params}")
        fitness, metrics = evaluate_ml_params(test_params, iteration_id=iteration_id)
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_params = test_params.copy()
            no_improvement_count = 0
            
            # Log improvement
            logger.info(f"Iteration {i+1}: New best fitness {fitness:.4f}")
            if metrics:
                logger.info(f"Metrics: {metrics.get_metrics()}")
                
            # Save model
            ml_enhancer = MLSignalEnhancer(
                model_type=best_params["model_type"],
                feature_selection=best_params["feature_selection"],
                n_features=best_params["n_features"],
                model_params={
                    "n_estimators": best_params["n_estimators"],
                    "max_depth": best_params["max_depth"],
                    "min_samples_split": best_params["min_samples_split"],
                    "min_samples_leaf": best_params["min_samples_leaf"]
                },
                output_dir="./models/training"
            )
            
            # Get data and train best model
            backtester = EnhancedBacktester(output_dir="./reports/training")
            data = backtester._fetch_data("BTCUSDT", "1h", 2000)
            ml_enhancer.train(
                data=data,
                prediction_horizon=best_params["prediction_horizon"],
                threshold=best_params["threshold"]
            )
            
            # Save model
            model_path = ml_enhancer.save_model(f"best_model_iter_{i+1}")
            logger.info(f"Saved best model to {model_path}")
            
        else:
            no_improvement_count += 1
            
        # Early stopping
        if no_improvement_count >= patience:
            logger.info(f"Early stopping at iteration {i+1}")
            break
            
        history.append(fitness)
    
    logger.info(f"Best parameters found: {best_params}")
    logger.info(f"Best fitness: {best_fitness:.4f}")
    
    return best_params


def test_simplified_evaluation():
    """A simplified test function to isolate the error"""
    logger.info("Running simplified evaluation test")
    
    # Create a simple test params object with all values explicitly typed
    test_params = {
        "model_type": "random_forest",
        "n_estimators": 50,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "feature_selection": "none",
        "n_features": 10,
        "prediction_horizon": 6,
        "threshold": 0.01,
        "ml_threshold": 0.6
    }
    
    # Create ML enhancer manually
    try:
        logger.info("Creating ML enhancer")
        os.makedirs("./models/test", exist_ok=True)
        ml_enhancer = MLSignalEnhancer(
            model_type="random_forest",  # Hardcoded for testing
            feature_selection="none",    # Simplest option
            n_features=10,              # Low number for faster processing
            output_dir="./models/test"
        )
        logger.info("ML enhancer created successfully")
    except Exception as e:
        logger.error(f"Error creating ML enhancer: {str(e)}")
        return False
    
    # Create backtester manually
    try:
        logger.info("Creating backtester")
        os.makedirs("./reports/test", exist_ok=True)
        backtester = EnhancedBacktester(output_dir="./reports/test")
        logger.info("Backtester created successfully")
    except Exception as e:
        logger.error(f"Error creating backtester: {str(e)}")
        return False
        
    # Attempt to fetch data
    try:
        logger.info("Fetching data")
        data = backtester._fetch_data("BTCUSDT", "1h", 500)  # Small dataset for testing
        logger.info(f"Data fetched successfully, shape: {data.shape}")
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        return False
        
    # Skip ML training and directly test backtesting
    try:
        logger.info("Running backtest with fixed parameters")
        final_balance, trades, backtest_metrics = backtester.run_backtest(
            symbol="BTCUSDT",
            interval="1h",
            candles=200,  # Fixed integer
            initial_balance=10000.0,  # Fixed float
            risk_percentage=1.0,      # Fixed float
            use_ml_signals=False      # Disable ML to simplify testing
        )
        logger.info("Backtest completed successfully")
        logger.info(f"Final balance: {final_balance}, Trades: {len(trades)}")
        return True
    except Exception as e:
        logger.error(f"Error in backtest: {str(e)}")
        return False

if __name__ == "__main__":
    # First run the simplified test to see if we can isolate the error
    if test_simplified_evaluation():
        logger.info("Simplified test passed, proceeding with full ML training")
        try:
            # Train the ML model
            best_params = optimize_ml_params(ml_params, iterations)
            logger.info(f"ML training completed successfully")
            
            # Evaluate final performance
            fitness, metrics = evaluate_ml_params(best_params)
            if metrics:
                logger.info(f"Final metrics: {metrics.get_metrics()}")
                
            # Save best parameters
            os.makedirs("models", exist_ok=True)
            joblib.dump(best_params, f"models/best_ml_params_{datetime.now().strftime('%Y%m%d_%H%M')}.joblib")
            
        except Exception as e:
            logger.error(f"ML training failed: {str(e)}")
            sys.exit(1)
    else:
        logger.error("Simplified test failed, cannot proceed with ML training")
        sys.exit(1)
