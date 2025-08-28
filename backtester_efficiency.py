#!/usr/bin/env python3
"""
Backtester Efficiency Evaluation Script

This script evaluates the efficiency of the backtester by running a series of ClickHouse
queries to analyze trading performance, risk management, and execution metrics.
It generates an overall efficiency score and saves the results to a file.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from clickhouse_driver import Client
import argparse
from typing import Dict, List, Any


class BacktesterEfficiencyEvaluator:
    """
    Evaluates backtester efficiency using ClickHouse queries and calculates
    a single efficiency score based on multiple metrics.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,
        database: str = "crypto_bot",
        output_file: str = "backtester_efficiency_results.json",
    ):
        """
        Initialize the evaluator with connection settings.

        Args:
            host: ClickHouse host address
            port: ClickHouse port
            database: Database name
            output_file: Path to save results
        """
        self.host = host
        self.port = port
        self.database = database
        self.output_file = output_file
        self.client = None
        self.connect()

    def connect(self) -> None:
        """Establish connection to ClickHouse"""
        try:
            self.client = Client(
                host=self.host,
                port=self.port,
                database=self.database,
                send_receive_timeout=60,
                connect_timeout=10,
            )
            # Test connection
            self.client.execute("SELECT 1")
            print(f"Successfully connected to ClickHouse at {self.host}:{self.port}")
        except Exception as e:
            print(f"Error connecting to ClickHouse: {str(e)}")
            print("Please ensure ClickHouse is running and accessible.")
            sys.exit(1)

    def run_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Run a ClickHouse query and return results.

        Args:
            query: SQL query string

        Returns:
            List of dictionaries containing query results
        """
        try:
            result = self.client.execute(query, with_column_types=True)
            columns = [col[0] for col in result[1]]
            data = [dict(zip(columns, row)) for row in result[0]]
            return data
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            print(f"Query: {query}")
            return []

    def calculate_trading_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate trading performance metrics from trades table.
        
        Returns:
            Dictionary of performance metrics
        """
        # Get overall win rate, profit factor, and trade metrics
        query = """
        SELECT
            COUNT(*) AS total_trades,
            countIf(profit_loss > 0) AS winning_trades,
            countIf(profit_loss < 0) AS losing_trades,
            round(countIf(profit_loss > 0) / count(*), 4) AS win_rate,
            round(sumIf(profit_loss, profit_loss > 0) / abs(sumIf(profit_loss, profit_loss < 0)), 4) AS profit_factor,
            round(avg(profit_loss), 2) AS avg_profit_loss,
            round(avgIf(profit_loss, profit_loss > 0), 2) AS avg_win,
            round(avgIf(profit_loss, profit_loss < 0), 2) AS avg_loss,
            round(max(profit_loss), 2) AS largest_win,
            round(min(profit_loss), 2) AS largest_loss,
            round(sum(profit_loss), 2) AS total_profit_loss,
            round(avg(trade_duration), 2) AS avg_trade_duration,
            round(median(trade_duration), 2) AS median_trade_duration
        FROM trades
        WHERE exit_timestamp > entry_timestamp
            AND profit_loss != 0
        """
        trading_metrics = self.run_query(query)
        
        if not trading_metrics:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "avg_profit_loss": 0,
                "trading_performance_score": 0
            }
            
        metrics = trading_metrics[0]
        
        # Calculate trading performance score (0-100)
        win_rate_score = min(metrics["win_rate"] * 100, 100) if "win_rate" in metrics and metrics["win_rate"] is not None else 0
        profit_factor_score = min(metrics["profit_factor"] * 10, 50) if "profit_factor" in metrics and metrics["profit_factor"] is not None else 0
        
        # Calculate expectancy score
        if "avg_win" in metrics and "avg_loss" in metrics and "win_rate" in metrics:
            avg_win = abs(metrics["avg_win"]) if metrics["avg_win"] is not None else 0
            avg_loss = abs(metrics["avg_loss"]) if metrics["avg_loss"] is not None else 0
            win_rate = metrics["win_rate"] if metrics["win_rate"] is not None else 0
            
            if avg_loss > 0:
                reward_risk_ratio = avg_win / avg_loss if avg_loss > 0 else 1
                expectancy = (win_rate * reward_risk_ratio) - (1 - win_rate)
                expectancy_score = min(expectancy * 25, 50)
            else:
                expectancy_score = 50  # Maximum score if no losses
        else:
            expectancy_score = 0
            
        trading_performance_score = win_rate_score * 0.4 + profit_factor_score * 0.3 + expectancy_score * 0.3
        metrics["trading_performance_score"] = round(trading_performance_score, 2)
        
        return metrics

    def calculate_risk_management_metrics(self) -> Dict[str, Any]:
        """
        Calculate risk management metrics from trades and iterations tables.
        
        Returns:
            Dictionary of risk management metrics
        """
        # Query for risk reward ratios, stop loss and take profit hit rates
        query = """
        WITH
            trade_counts AS (
                SELECT
                    COUNT(*) AS total_trades,
                    countIf(trade_type = 'stop_loss') AS sl_hits,
                    countIf(trade_type = 'take_profit_1') AS tp1_hits,
                    countIf(trade_type = 'take_profit_2') AS tp2_hits,
                    countIf(trade_type = 'take_profit_3') AS tp3_hits,
                    avg(risk_reward_ratio) AS avg_risk_reward_ratio
                FROM trades
                WHERE trade_type IN ('entry', 'stop_loss', 'take_profit_1', 'take_profit_2', 'take_profit_3')
            ),
            max_drawdowns AS (
                SELECT
                    iteration_id,
                    max_drawdown
                FROM iterations
                ORDER BY created_at DESC
                LIMIT 100
            )
        
        SELECT
            tc.total_trades,
            tc.sl_hits,
            tc.tp1_hits,
            tc.tp2_hits,
            tc.tp3_hits,
            round(tc.sl_hits / tc.total_trades, 4) AS sl_hit_rate,
            round((tc.tp1_hits + tc.tp2_hits + tc.tp3_hits) / tc.total_trades, 4) AS tp_hit_rate,
            round(tc.avg_risk_reward_ratio, 2) AS avg_risk_reward_ratio,
            round(avg(md.max_drawdown), 4) AS avg_max_drawdown
        FROM trade_counts tc
        CROSS JOIN max_drawdowns md
        """
        
        risk_metrics = self.run_query(query)
        
        if not risk_metrics:
            return {
                "sl_hit_rate": 0,
                "tp_hit_rate": 0,
                "avg_risk_reward_ratio": 0,
                "avg_max_drawdown": 0,
                "risk_management_score": 0
            }
            
        metrics = risk_metrics[0]
        
        # Calculate risk management score (0-100)
        # Lower stop loss hit rate is better (but not zero)
        sl_hit_rate = metrics.get("sl_hit_rate", 0) or 0
        sl_score = 100 - (sl_hit_rate * 100) if sl_hit_rate > 0 else 70  # Penalize if no SL hits (might indicate improper risk management)
        
        # Higher take profit hit rate is better
        tp_hit_rate = metrics.get("tp_hit_rate", 0) or 0
        tp_score = tp_hit_rate * 100
        
        # Higher risk-reward ratio is better (up to a point)
        risk_reward = metrics.get("avg_risk_reward_ratio", 0) or 0
        rr_score = min(risk_reward * 20, 100)
        
        # Lower max drawdown is better
        max_dd = metrics.get("avg_max_drawdown", 0) or 0
        dd_score = 100 - (max_dd * 100) if max_dd <= 1 else 0
        
        risk_management_score = sl_score * 0.2 + tp_score * 0.3 + rr_score * 0.3 + dd_score * 0.2
        metrics["risk_management_score"] = round(risk_management_score, 2)
        
        return metrics

    def calculate_consistency_metrics(self) -> Dict[str, Any]:
        """
        Calculate consistency metrics across iterations.
        
        Returns:
            Dictionary of consistency metrics
        """
        # Query for consistency across iterations
        query = """
        WITH 
            iteration_metrics AS (
                SELECT
                    iteration_id,
                    win_rate,
                    profit_factor,
                    total_revenue,
                    max_drawdown,
                    sharpe_ratio
                FROM iterations
                WHERE created_at > now() - INTERVAL 30 DAY
                ORDER BY created_at DESC
                LIMIT 100
            )
            
        SELECT
            count() AS total_iterations,
            round(avg(win_rate), 4) AS avg_win_rate,
            round(stddevPop(win_rate), 4) AS stddev_win_rate,
            round(avg(profit_factor), 4) AS avg_profit_factor,
            round(stddevPop(profit_factor), 4) AS stddev_profit_factor,
            round(avg(total_revenue), 2) AS avg_revenue,
            round(stddevPop(total_revenue), 2) AS stddev_revenue,
            round(avg(max_drawdown), 4) AS avg_max_drawdown,
            round(stddevPop(max_drawdown), 4) AS stddev_max_drawdown,
            round(avg(sharpe_ratio), 4) AS avg_sharpe_ratio,
            round(stddevPop(sharpe_ratio), 4) AS stddev_sharpe_ratio
        FROM iteration_metrics
        """
        
        consistency_metrics = self.run_query(query)
        
        if not consistency_metrics:
            return {
                "total_iterations": 0,
                "coefficient_variation_win_rate": 0,
                "coefficient_variation_profit_factor": 0,
                "consistency_score": 0
            }
            
        metrics = consistency_metrics[0]
        
        # Calculate coefficient of variation (lower is better - indicates consistency)
        metrics["coefficient_variation_win_rate"] = round(metrics["stddev_win_rate"] / metrics["avg_win_rate"], 4) if metrics["avg_win_rate"] > 0 else 0
        metrics["coefficient_variation_profit_factor"] = round(metrics["stddev_profit_factor"] / metrics["avg_profit_factor"], 4) if metrics["avg_profit_factor"] > 0 else 0
        metrics["coefficient_variation_revenue"] = round(metrics["stddev_revenue"] / metrics["avg_revenue"], 4) if metrics["avg_revenue"] > 0 else 0
        metrics["coefficient_variation_max_drawdown"] = round(metrics["stddev_max_drawdown"] / metrics["avg_max_drawdown"], 4) if metrics["avg_max_drawdown"] > 0 else 0
        metrics["coefficient_variation_sharpe"] = round(metrics["stddev_sharpe_ratio"] / metrics["avg_sharpe_ratio"], 4) if metrics["avg_sharpe_ratio"] > 0 else 0
        
        # Calculate consistency score (0-100)
        # Lower coefficient of variation is better (more consistent)
        win_rate_consistency = 100 - min(metrics["coefficient_variation_win_rate"] * 100, 100) if "coefficient_variation_win_rate" in metrics else 0
        pf_consistency = 100 - min(metrics["coefficient_variation_profit_factor"] * 100, 100) if "coefficient_variation_profit_factor" in metrics else 0
        revenue_consistency = 100 - min(metrics["coefficient_variation_revenue"] * 100, 100) if "coefficient_variation_revenue" in metrics else 0
        sharpe_consistency = 100 - min(metrics["coefficient_variation_sharpe"] * 100, 100) if "coefficient_variation_sharpe" in metrics else 0
        
        consistency_score = (win_rate_consistency * 0.25 + 
                            pf_consistency * 0.25 + 
                            revenue_consistency * 0.25 + 
                            sharpe_consistency * 0.25)
        
        metrics["consistency_score"] = round(consistency_score, 2)
        
        return metrics

    def calculate_market_conditions_adaptability(self) -> Dict[str, Any]:
        """
        Calculate how well the backtester adapts to different market conditions.
        
        Returns:
            Dictionary of adaptability metrics
        """
        # Query for performance across different market conditions (by symbol)
        query = """
        WITH 
            symbol_metrics AS (
                SELECT
                    symbol,
                    avg(win_rate) AS avg_win_rate,
                    avg(revenue) AS avg_revenue,
                    avg(risk_reward_ratio) AS avg_risk_reward,
                    count() AS iterations_count
                FROM sub_iterations
                WHERE created_at > now() - INTERVAL 30 DAY
                GROUP BY symbol
                HAVING iterations_count > 5
            ),
            
            overall_metrics AS (
                SELECT
                    avg(avg_win_rate) AS overall_win_rate,
                    avg(avg_revenue) AS overall_revenue,
                    stddevPop(avg_win_rate) AS stddev_win_rate_by_symbol,
                    stddevPop(avg_revenue) AS stddev_revenue_by_symbol,
                    count() AS symbol_count
                FROM symbol_metrics
            )
            
        SELECT
            symbol_count,
            round(overall_win_rate, 4) AS overall_win_rate,
            round(overall_revenue, 2) AS overall_revenue,
            round(stddev_win_rate_by_symbol, 4) AS stddev_win_rate_by_symbol,
            round(stddev_revenue_by_symbol, 2) AS stddev_revenue_by_symbol,
            round(stddev_win_rate_by_symbol / overall_win_rate, 4) AS win_rate_variation_across_symbols,
            round(stddev_revenue_by_symbol / overall_revenue, 4) AS revenue_variation_across_symbols
        FROM overall_metrics
        """
        
        adaptability_metrics = self.run_query(query)
        
        if not adaptability_metrics:
            return {
                "symbol_count": 0,
                "adaptability_score": 0
            }
            
        metrics = adaptability_metrics[0]
        
        # Calculate adaptability score (0-100)
        # Lower variation across symbols is better (indicates good adaptability to different market conditions)
        win_rate_adaptability = 100 - min(metrics.get("win_rate_variation_across_symbols", 1) * 100, 100)
        revenue_adaptability = 100 - min(metrics.get("revenue_variation_across_symbols", 1) * 100, 100)
        
        # Only consider adaptability if we have data for multiple symbols
        symbol_count = metrics.get("symbol_count", 0)
        if symbol_count <= 1:
            adaptability_score = 0  # Not enough data to calculate adaptability
        else:
            adaptability_score = (win_rate_adaptability * 0.5 + revenue_adaptability * 0.5)
        
        metrics["adaptability_score"] = round(adaptability_score, 2)
        
        return metrics

    def calculate_execution_metrics(self) -> Dict[str, Any]:
        """
        Calculate execution metrics for the backtester.
        
        Returns:
            Dictionary of execution metrics
        """
        # Calculate time-based metrics from trades
        query = """
        WITH
            date_metrics AS (
                SELECT
                    toDate(created_at) AS trade_date,
                    count() AS trades_per_day
                FROM trades
                GROUP BY trade_date
                ORDER BY trade_date DESC
                LIMIT 30
            ),
            trade_durations AS (
                SELECT
                    trade_duration,
                    count() AS count
                FROM trades
                WHERE exit_timestamp > entry_timestamp
                GROUP BY trade_duration
                ORDER BY trade_duration
            )
            
        SELECT
            round(avg(trades_per_day), 2) AS avg_trades_per_day,
            max(trades_per_day) AS max_trades_per_day,
            (SELECT count() FROM trade_durations) AS distinct_trade_durations,
            (SELECT count() FROM trades) AS total_trades
        FROM date_metrics
        """
        
        execution_metrics = self.run_query(query)
        
        if not execution_metrics:
            return {
                "avg_trades_per_day": 0,
                "execution_score": 0
            }
            
        metrics = execution_metrics[0]
        
        # Calculate execution speed and diversity score (0-100)
        # More trades per day is better (up to a reasonable limit)
        trades_per_day_score = min(metrics.get("avg_trades_per_day", 0) * 5, 100)
        
        # More diverse trade durations is better (indicates flexible strategy)
        duration_diversity_score = min(metrics.get("distinct_trade_durations", 0), 100)
        
        execution_score = trades_per_day_score * 0.5 + duration_diversity_score * 0.5
        metrics["execution_score"] = round(execution_score, 2)
        
        return metrics

    def calculate_overall_efficiency_score(self, all_metrics: Dict[str, Dict[str, Any]]) -> float:
        """
        Calculate overall efficiency score from individual metrics.
        
        Args:
            all_metrics: Dictionary containing all calculated metrics
            
        Returns:
            Overall efficiency score from 0-100
        """
        # Define weights for each category
        weights = {
            "trading_performance": 0.35,
            "risk_management": 0.25,
            "consistency": 0.20,
            "adaptability": 0.10,
            "execution": 0.10
        }
        
        # Extract category scores
        scores = {
            "trading_performance": all_metrics["trading_performance"].get("trading_performance_score", 0),
            "risk_management": all_metrics["risk_management"].get("risk_management_score", 0),
            "consistency": all_metrics["consistency"].get("consistency_score", 0),
            "adaptability": all_metrics["adaptability"].get("adaptability_score", 0),
            "execution": all_metrics["execution"].get("execution_score", 0)
        }
        
        # Calculate weighted score
        overall_score = sum(scores[category] * weights[category] for category in weights)
        
        # Round to 2 decimal places
        return round(overall_score, 2)

    def evaluate(self) -> Dict[str, Any]:
        """
        Run all evaluations and calculate overall efficiency score.
        
        Returns:
            Dictionary with all results and overall score
        """
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {}
        }
        
        # Run all metric calculations
        results["metrics"]["trading_performance"] = self.calculate_trading_performance_metrics()
        results["metrics"]["risk_management"] = self.calculate_risk_management_metrics()
        results["metrics"]["consistency"] = self.calculate_consistency_metrics()
        results["metrics"]["adaptability"] = self.calculate_market_conditions_adaptability()
        results["metrics"]["execution"] = self.calculate_execution_metrics()
        
        # Calculate overall efficiency score
        results["overall_efficiency_score"] = self.calculate_overall_efficiency_score(results["metrics"])
        
        # Add qualitative rating based on score
        if results["overall_efficiency_score"] >= 90:
            results["rating"] = "Excellent"
        elif results["overall_efficiency_score"] >= 80:
            results["rating"] = "Very Good"
        elif results["overall_efficiency_score"] >= 70:
            results["rating"] = "Good"
        elif results["overall_efficiency_score"] >= 60:
            results["rating"] = "Satisfactory"
        elif results["overall_efficiency_score"] >= 50:
            results["rating"] = "Adequate"
        elif results["overall_efficiency_score"] >= 40:
            results["rating"] = "Needs Improvement"
        elif results["overall_efficiency_score"] >= 30:
            results["rating"] = "Poor"
        else:
            results["rating"] = "Critically Deficient"
            
        return results

    def save_results(self, results: Dict[str, Any]) -> None:
        """
        Save evaluation results to file.
        
        Args:
            results: Dictionary of evaluation results
        """
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(self.output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Save as JSON
            with open(self.output_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            # Also save as CSV for easy import into spreadsheets
            csv_file = os.path.splitext(self.output_file)[0] + '.csv'
            
            # Flatten the dictionary structure for CSV
            flat_dict = {
                "timestamp": results["timestamp"],
                "overall_efficiency_score": results["overall_efficiency_score"],
                "rating": results["rating"]
            }
            
            # Add all metrics
            for category, metrics in results["metrics"].items():
                for metric_name, value in metrics.items():
                    flat_dict[f"{category}_{metric_name}"] = value
                    
            # Convert to DataFrame and save
            pd.DataFrame([flat_dict]).to_csv(csv_file, index=False)
            
            print(f"Results saved to {self.output_file} and {csv_file}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")


def main():
    """Main function to parse arguments and run evaluation"""
    parser = argparse.ArgumentParser(description='Evaluate backtester efficiency using ClickHouse queries')
    parser.add_argument('--host', default='localhost', help='ClickHouse host address')
    parser.add_argument('--port', type=int, default=9000, help='ClickHouse port')
    parser.add_argument('--database', default='crypto_bot', help='Database name')
    parser.add_argument('--output', default='backtester_efficiency_results.json', help='Output file path')
    
    args = parser.parse_args()
    
    evaluator = BacktesterEfficiencyEvaluator(
        host=args.host,
        port=args.port,
        database=args.database,
        output_file=args.output
    )
    
    print("Evaluating backtester efficiency...")
    results = evaluator.evaluate()
    
    evaluator.save_results(results)
    
    print("\nBacktester Efficiency Summary:")
    print(f"Overall Score: {results['overall_efficiency_score']}/100 - {results['rating']}")
    print("\nCategory Scores:")
    for category, metrics in results["metrics"].items():
        score_key = f"{category}_score"
        for key in metrics:
            if key.endswith("_score"):
                print(f"- {category.replace('_', ' ').title()}: {metrics[key]}/100")
                break
    
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
