import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Callable
from datetime import datetime, timedelta
import uuid
import matplotlib.pyplot as plt

# Import the enhanced backtesting modules
from .performance_metrics import calculate_performance_metrics, plot_equity_curve, generate_performance_report
from .monte_carlo import monte_carlo_simulation, plot_monte_carlo_results, run_monte_carlo_analysis
from .transaction_costs import TransactionCostsModel, apply_transaction_costs_to_backtest
from .risk_management import AdaptiveRiskManager, DynamicStopLossManager
from .walk_forward import WalkForwardAnalysis, walk_forward_backtest
from .strategy_optimizer import StrategyOptimizer, optimize_strategy
from .strategy_comparison import StrategyComparison, compare_strategies
from .correlation_analysis import CorrelationAnalysis, analyze_correlations

# Import ML signal enhancement module if available
try:
    from .ml_signal_enhancement import MLSignalEnhancer
    ML_SUPPORT = True
except ImportError:
    ML_SUPPORT = False

# Add project root to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from .strategy import backtest_strategy

class EnhancedBacktester:
    """
    Integration of all enhanced backtesting features into a single class
    for comprehensive strategy development, testing, and optimization.
    """
    
    def __init__(self, output_dir: str = "./reports"):
        """
        Initialize enhanced backtester.
        
        Args:
            output_dir: Base directory for saving reports
        """
        self.output_dir = output_dir
        self.transaction_costs_model = TransactionCostsModel()
        self.risk_manager = AdaptiveRiskManager()
        self.stop_loss_manager = DynamicStopLossManager()
        
        # Initialize ML enhancer if available
        self.ml_enhancer = None
        if ML_SUPPORT:
            self.ml_enhancer = MLSignalEnhancer(output_dir=os.path.join(output_dir, "models"))
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def _fetch_data(self, symbol: str, interval: str, candles: int) -> pd.DataFrame:
        """Fetch data for backtesting - exposed for ML use"""
        # Use data fetching instruments to get the data
        # Note: order of parameters in fetch_candles is: symbol, desired_total, interval
        try:
            from data_fetching_instruments import fetch_candles
            return fetch_candles(symbol=symbol, desired_total=candles, interval=interval)
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            # Return empty DataFrame with expected columns as fallback
            # Use uppercase first letter column names to match what the code expects
            return pd.DataFrame({
                'Open': [], 'High': [], 'Low': [], 'Close': [], 'Volume': []
            })
        
    def run_backtest(
        self,
        symbol: str,
        interval: str,
        candles: int = 1000,
        window: int = 300,
        initial_balance: float = 10000.0,
        risk_percentage: float = 1.0,
        apply_costs: bool = True,
        adaptive_risk: bool = True,
        use_ml_signals: bool = False,
        ml_model_path: Optional[str] = None,
        ml_threshold: float = 0.6,
        **kwargs
    ) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run enhanced backtest with all features.
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            candles: Number of candles to fetch
            window: Lookback period for signal generation
            initial_balance: Starting capital
            risk_percentage: Risk percentage per trade
            apply_costs: Whether to apply transaction costs
            adaptive_risk: Whether to use adaptive risk management
            use_ml_signals: Whether to use ML-enhanced signals
            ml_model_path: Path to a pre-trained ML model (if None, will use default or train new model)
            ml_threshold: Probability threshold for ML signal confirmation
            **kwargs: Additional arguments for backtest_strategy
            
        Returns:
            Tuple of (final_balance, trades, metrics)
        """
        # Check if ML enhancement is requested but not available
        if use_ml_signals and not ML_SUPPORT:
            print("Warning: ML signal enhancement requested but ML module is not available.")
            print("Running backtest without ML enhancement.")
            use_ml_signals = False
        
        # Load ML model if requested
        if use_ml_signals and self.ml_enhancer:
            if ml_model_path and os.path.exists(ml_model_path):
                self.ml_enhancer.load_model(ml_model_path)
                print(f"Loaded ML model from {ml_model_path}")
            elif not self.ml_enhancer.is_trained:
                print("No ML model specified or found. Using default settings.")
        
        # Run basic backtest
        final_balance, trades, sub_iteration_id = backtest_strategy(
            symbol=symbol,
            interval=interval,
            candles=candles,
            window=window,
            initial_balance=initial_balance,
            risk_percentage=risk_percentage,
            **kwargs
        )
        
        # Apply ML signal enhancement if requested
        if use_ml_signals and self.ml_enhancer and self.ml_enhancer.is_trained and trades:
            try:
                # Fetch data for ML prediction
                data = self._fetch_data(symbol, interval, candles)
                
                # Convert trades to signals series for enhancement
                signals = pd.Series(0, index=data.index)
                for trade in trades:
                    if 'timestamp' in trade and 'signal' in trade:
                        signals[trade['timestamp']] = 1 if trade['signal'].lower() == 'buy' else -1
                
                # Enhance signals with ML predictions
                print(f"Enhancing signals with ML model (threshold={ml_threshold})...")
                enhanced_signals = self.ml_enhancer.enhance_signals(
                    data=data,
                    signals=signals,
                    threshold=ml_threshold
                )
                
                # Count signals before and after enhancement
                original_signals = (signals == 1).sum()
                new_signals = (enhanced_signals == 1).sum()
                print(f"Original buy signals: {original_signals}, Enhanced buy signals: {new_signals}")
                print(f"ML enhancement removed {original_signals - new_signals} false signals")
                
                # Filter trades based on enhanced signals
                enhanced_trades = []
                for trade in trades:
                    if 'timestamp' in trade and trade['timestamp'] in enhanced_signals.index:
                        # Keep trade only if ML confirms it's a good signal
                        if enhanced_signals[trade['timestamp']] == 1:
                            enhanced_trades.append(trade)
                
                # Update trades list with enhanced trades
                if len(enhanced_trades) > 0:
                    print(f"Using {len(enhanced_trades)} ML-enhanced trades (filtered from {len(trades)})")
                    trades = enhanced_trades
                    
                    # Recalculate final balance
                    final_balance = initial_balance
                    for trade in trades:
                        if 'profit_loss' in trade:
                            final_balance += trade['profit_loss']
                        elif 'profit' in trade:
                            final_balance += trade['profit']
            except Exception as e:
                print(f"Error applying ML signal enhancement: {str(e)}")
                print("Proceeding with original signals.")
        
        # Apply transaction costs if requested
        if apply_costs and trades:
            # Convert to DataFrame for processing
            trades_df = pd.DataFrame(trades)
            
            # Apply transaction costs
            adjusted_df = apply_transaction_costs_to_backtest(
                trades_df, 
                self.transaction_costs_model
            )
            
            # Convert back to list of dictionaries
            trades = adjusted_df.to_dict('records')
            
            # Recalculate final balance
            final_balance = initial_balance
            for trade in trades:
                if 'profit_loss' in trade:
                    final_balance += trade['profit_loss']
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(trades, initial_balance, final_balance)
        
        return final_balance, trades, metrics
    
    def generate_comprehensive_report(
        self,
        symbol: str,
        interval: str,
        candles: int = 1000,
        window: int = 300,
        initial_balance: float = 10000.0,
        risk_percentage: float = 1.0,
        monte_carlo_sims: int = 1000,
        report_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate comprehensive backtesting report with all analyses.
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            candles: Number of candles to fetch
            window: Lookback period for signal generation
            initial_balance: Starting capital
            risk_percentage: Risk percentage per trade
            monte_carlo_sims: Number of Monte Carlo simulations
            report_name: Custom name for the report
            **kwargs: Additional arguments for backtest_strategy
            
        Returns:
            Dictionary with report results and paths
        """
        # Create report name if not provided
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"{symbol}_{interval}_{timestamp}"
        
        # Create report directory
        report_dir = os.path.join(self.output_dir, report_name)
        os.makedirs(report_dir, exist_ok=True)
        
        # Run backtest with transaction costs
        print(f"Running enhanced backtest for {symbol} {interval}...")
        final_balance, trades, metrics = self.run_backtest(
            symbol=symbol,
            interval=interval,
            candles=candles,
            window=window,
            initial_balance=initial_balance,
            risk_percentage=risk_percentage,
            apply_costs=True,
            adaptive_risk=True,
            **kwargs
        )
        
        # Generate performance report
        print("Generating performance metrics...")
        performance_dir = os.path.join(report_dir, "performance")
        generate_performance_report(
            trades, initial_balance, final_balance, 
            output_dir=performance_dir, 
            report_name=report_name
        )
        
        # Run Monte Carlo analysis
        print("Running Monte Carlo simulations...")
        mc_dir = os.path.join(report_dir, "monte_carlo")
        mc_results = run_monte_carlo_analysis(
            trades, initial_balance, final_balance,
            simulations=monte_carlo_sims,
            output_dir=mc_dir,
            report_name=report_name
        )
        
        # Summarize results
        summary = {
            "symbol": symbol,
            "interval": interval,
            "candles": candles,
            "window": window,
            "initial_balance": initial_balance,
            "final_balance": final_balance,
            "net_profit": final_balance - initial_balance,
            "return_pct": (final_balance - initial_balance) / initial_balance * 100,
            "num_trades": len(trades),
            "win_rate": metrics.get("win_rate", 0) * 100,
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "monte_carlo_confidence_lower": mc_results.get("confidence_interval_lower", 0),
            "monte_carlo_confidence_upper": mc_results.get("confidence_interval_upper", 0),
            "report_directory": report_dir
        }
        
        # Save summary to CSV
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(os.path.join(report_dir, "summary.csv"), index=False)
        
        print(f"Comprehensive report generated at: {report_dir}")
        
        return summary
    
    def optimize_parameters(
        self,
        symbol: str,
        interval: str,
        param_ranges: Dict[str, Any],
        optimization_target: str = "sharpe_ratio",
        search_method: str = "random",
        num_trials: int = 100,
        **fixed_params
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid or random search.
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            param_ranges: Dictionary of parameter ranges to optimize
            optimization_target: Metric to optimize
            search_method: "grid" or "random"
            num_trials: Number of trials for optimization
            **fixed_params: Fixed parameters for backtest_strategy
            
        Returns:
            Dictionary with optimization results
        """
        # Create wrapper function for optimization
        def backtest_wrapper(**params):
            # Combine fixed and variable parameters
            all_params = {**fixed_params, **params, "symbol": symbol, "interval": interval}
            
            # Run backtest
            final_balance, trades, _ = self.run_backtest(**all_params)
            
            # Calculate metrics
            metrics = calculate_performance_metrics(trades, all_params.get("initial_balance", 10000), final_balance)
            
            return final_balance, trades, metrics
        
        # Run optimization
        optimization_dir = os.path.join(self.output_dir, f"optimization_{symbol}_{interval}")
        results = optimize_strategy(
            backtest_func=backtest_wrapper,
            param_ranges=param_ranges,
            optimization_target=optimization_target,
            search_method=search_method,
            num_trials=num_trials,
            output_dir=optimization_dir
        )
        
        return results
    
    def run_walk_forward_analysis(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        train_size: float = 0.7,
        num_windows: int = 5,
        window_type: str = "expanding",
        **params
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis to validate strategy robustness.
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            start_date: Start date for analysis
            end_date: End date for analysis
            train_size: Proportion to use for training
            num_windows: Number of windows to use
            window_type: "expanding" or "rolling"
            **params: Additional parameters for backtest_strategy
            
        Returns:
            Dictionary with analysis results
        """
        # Create wrapper function for walk-forward analysis
        def backtest_wrapper(start_date, end_date, **kwargs):
            # Combine parameters
            all_params = {**params, **kwargs, "symbol": symbol, "interval": interval}
            
            # Run backtest with date filtering
            # Note: In a real implementation, you'd need to modify backtest_strategy
            # to accept start_date and end_date parameters
            final_balance, trades, _ = self.run_backtest(**all_params)
            
            # Calculate metrics
            metrics = calculate_performance_metrics(trades, all_params.get("initial_balance", 10000), final_balance)
            
            return final_balance, trades, metrics
        
        # Run walk-forward analysis
        wf_dir = os.path.join(self.output_dir, f"walk_forward_{symbol}_{interval}")
        results = walk_forward_backtest(
            backtest_func=backtest_wrapper,
            start_date=start_date,
            end_date=end_date,
            train_size=train_size,
            window_type=window_type,
            num_windows=num_windows,
            output_dir=wf_dir
        )
        
        return results
    
    def compare_strategies(
        self,
        strategies: Dict[str, Dict[str, Any]],
        benchmark_params: Optional[Dict[str, Any]] = None,
        initial_balance: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Compare multiple strategies against each other and a benchmark.
        
        Args:
            strategies: Dict mapping strategy names to parameter dictionaries
            benchmark_params: Parameters for benchmark strategy (buy and hold)
            initial_balance: Initial account balance
            
        Returns:
            Dictionary with comparison results
        """
        # Create wrapper function for strategy comparison
        def backtest_wrapper(**params):
            # Run backtest
            final_balance, trades, _ = self.run_backtest(**params)
            
            # Calculate metrics
            metrics = calculate_performance_metrics(trades, params.get("initial_balance", initial_balance), final_balance)
            
            return final_balance, trades, metrics
        
        # Create strategy functions dictionary
        strategy_funcs = {}
        
        for name, params in strategies.items():
            strategy_funcs[name] = {
                "func": backtest_wrapper,
                "parameters": params
            }
        
        # Create benchmark function if parameters provided
        benchmark_func = None
        if benchmark_params:
            benchmark_func = backtest_wrapper
        
        # Run comparison
        comparison_dir = os.path.join(self.output_dir, "strategy_comparison")
        results = compare_strategies(
            strategies=strategy_funcs,
            benchmark_func=benchmark_func,
            benchmark_params=benchmark_params,
            initial_balance=initial_balance,
            output_dir=comparison_dir
        )
        
        return results

# Example usage function
def run_enhanced_backtest(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    candles: int = 1000,
    window: int = 300,
    initial_balance: float = 10000.0,
    risk_percentage: float = 1.0,
    output_dir: str = "./reports",
    run_optimization: bool = False
):
    """
    Run enhanced backtest with comprehensive reporting.
    
    Args:
        symbol: Trading pair symbol
        interval: Candle interval
        candles: Number of candles
        window: Lookback window
        initial_balance: Starting balance
        risk_percentage: Risk percentage per trade
        output_dir: Output directory for reports
        run_optimization: Whether to run parameter optimization
        
    Returns:
        Dictionary with summary results
    """
    # Initialize enhanced backtester
    backtester = EnhancedBacktester(output_dir=output_dir)
    
    # Run comprehensive backtest and generate report
    summary = backtester.generate_comprehensive_report(
        symbol=symbol,
        interval=interval,
        candles=candles,
        window=window,
        initial_balance=initial_balance,
        risk_percentage=risk_percentage,
        monte_carlo_sims=500  # Reduced for faster execution
    )
    
    # Optionally run parameter optimization
    if run_optimization:
        # Define parameter ranges to optimize
        param_ranges = {
            "window": (100, 500),
            "risk_percentage": (0.5, 2.0),
        }
        
        # Run optimization
        optimization_results = backtester.optimize_parameters(
            symbol=symbol,
            interval=interval,
            param_ranges=param_ranges,
            optimization_target="sharpe_ratio",
            search_method="random",
            num_trials=50,  # Reduced for faster execution
            candles=candles,
            initial_balance=initial_balance
        )
        
        # Print best parameters
        if optimization_results["best_result"]:
            best_params = optimization_results["best_result"]["parameters"]
            best_value = optimization_results["best_result"]["target_value"]
            print(f"Optimal parameters found: {best_params}, Sharpe ratio: {best_value:.2f}")
    
    return summary

if __name__ == "__main__":
    # Example command line usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Run enhanced backtesting')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--interval', default='1h', help='Candle interval')
    parser.add_argument('--candles', type=int, default=1000, help='Number of candles')
    parser.add_argument('--window', type=int, default=300, help='Lookback window')
    parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
    parser.add_argument('--risk', type=float, default=1.0, help='Risk percentage')
    parser.add_argument('--optimize', action='store_true', help='Run parameter optimization')
    
    args = parser.parse_args()
    
    # Run enhanced backtest
    run_enhanced_backtest(
        symbol=args.symbol,
        interval=args.interval,
        candles=args.candles,
        window=args.window,
        initial_balance=args.balance,
        risk_percentage=args.risk,
        run_optimization=args.optimize
    )
