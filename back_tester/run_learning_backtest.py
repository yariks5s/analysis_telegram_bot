#!/usr/bin/env python3
"""
Self-Learning Backtesting Runner

This script runs comprehensive backtests on BTCUSDT and ETHUSDT across all 
time intervals with AGGRESSIVE self-learning. The system:
- Learns from each trade outcome
- Directly adjusts weights based on indicator performance
- Runs many iterations to find optimal weights

Usage:
    python3 run_learning_backtest.py [options]

Options:
    --iterations N    Number of learning iterations (default: 20)
    --candles N       Number of candles per backtest
    --balance N       Initial balance (default: 10000)
    --learning-rate   How fast weights change (default: 0.15)
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import numpy as np
import random

# Add project root to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from back_tester.strategy import backtest_strategy

# Configuration
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVALS = ["5m", "15m", "30m", "1h", "4h"]  # Most useful intervals

WEIGHT_NAMES = [
    "W_BULLISH_OB", "W_BEARISH_OB", "W_BULLISH_BREAKER", "W_BEARISH_BREAKER",
    "W_ABOVE_SUPPORT", "W_BELOW_RESISTANCE", "W_FVG_ABOVE", "W_FVG_BELOW",
    "W_TREND", "W_SWEEP_HIGHS", "W_SWEEP_LOWS", "W_STRUCTURE_BREAK",
    "W_PIN_BAR", "W_ENGULFING", "W_LIQUIDITY_POOL_ABOVE", "W_LIQUIDITY_POOL_BELOW",
    "W_LIQUIDITY_POOL_ROUND", "W_RSI_EXTREME"
]

# Candle settings per interval (increased for more trades)
# Binance allows up to 1000 candles per request, but we can fetch more with pagination
CANDLES_PER_INTERVAL = {
    "5m": 2000,   # ~7 days of data
    "15m": 1500,  # ~15 days of data
    "30m": 1200,  # ~25 days of data
    "1h": 1000,   # ~42 days of data
    "4h": 800,    # ~133 days of data (~4.5 months)
}

class Colors:
    H = '\033[95m'
    B = '\033[94m'
    C = '\033[96m'
    G = '\033[92m'
    Y = '\033[93m'
    R = '\033[91m'
    E = '\033[0m'
    BOLD = '\033[1m'


class AggressiveLearner:
    """
    Aggressive weight learning with direct performance-based updates.
    Unlike conservative learners, this one makes BIG changes.
    """
    
    def __init__(
        self,
        storage_path: str = "./data/btc_eth_learning",
        initial_balance: float = 10000.0,
        learning_rate: float = 0.15,  # Much higher than typical 0.05
        lr_decay: float = 0.95,  # Decay rate per iteration (prevents overfitting)
    ):
        self.storage_path = storage_path
        self.initial_balance = initial_balance
        self.base_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        
        os.makedirs(storage_path, exist_ok=True)
        
        # Load or initialize weights
        self.weights = self._load_weights()
        self.best_weights = self.weights.copy()
        self.best_profit = float('-inf')
        
        # Track indicator performance across all trades
        self.indicator_stats = defaultdict(lambda: {
            'win_count': 0, 'loss_count': 0,
            'win_profit': 0.0, 'loss_amount': 0.0,
            'appearances': 0
        })
        
        # Track overall performance
        self.iteration = 0
        self.total_trades = 0
        self.total_profit = 0.0
        self.profit_history = []
        self.weight_history = []
    
    def _load_weights(self) -> List[float]:
        """Load weights from file or use optimized defaults"""
        weights_file = os.path.join(self.storage_path, "final_weights.json")
        
        if os.path.exists(weights_file):
            try:
                with open(weights_file, 'r') as f:
                    data = json.load(f)
                    if 'weights' in data:
                        weights_data = data['weights']
                        # Handle dict format (name -> value)
                        if isinstance(weights_data, dict):
                            if len(weights_data) == len(WEIGHT_NAMES):
                                weights = [float(weights_data.get(name, 1.0)) for name in WEIGHT_NAMES]
                                print(f"{Colors.G}✓ Loaded weights from previous session{Colors.E}")
                                return weights
                        # Handle list format
                        elif isinstance(weights_data, list) and len(weights_data) == len(WEIGHT_NAMES):
                            weights = [float(w) for w in weights_data]
                            print(f"{Colors.G}✓ Loaded weights from previous session{Colors.E}")
                            return weights
            except Exception as e:
                print(f"{Colors.Y}Warning: Could not load weights: {e}{Colors.E}")
        
        # Smart starting weights (not all 1.0)
        return [
            1.3,   # W_BULLISH_OB - Order blocks tend to work well
            1.3,   # W_BEARISH_OB
            1.1,   # W_BULLISH_BREAKER
            1.1,   # W_BEARISH_BREAKER
            0.8,   # W_ABOVE_SUPPORT
            0.8,   # W_BELOW_RESISTANCE
            0.6,   # W_FVG_ABOVE - FVGs less reliable alone
            0.6,   # W_FVG_BELOW
            1.0,   # W_TREND
            1.4,   # W_SWEEP_HIGHS - Sweeps are key signals
            1.4,   # W_SWEEP_LOWS
            1.5,   # W_STRUCTURE_BREAK - Very important
            0.7,   # W_PIN_BAR
            0.7,   # W_ENGULFING
            1.0,   # W_LIQUIDITY_POOL_ABOVE
            1.0,   # W_LIQUIDITY_POOL_BELOW
            1.2,   # W_LIQUIDITY_POOL_ROUND
            0.5,   # W_RSI_EXTREME - Often unreliable
        ]
    
    def _save_weights(self):
        """Save current weights"""
        data = {
            'weights': self.weights,
            'weight_names': WEIGHT_NAMES,
            'best_weights': self.best_weights,
            'best_profit': self.best_profit,
            'iteration': self.iteration,
            'total_trades': self.total_trades,
            'total_profit': self.total_profit,
            'timestamp': datetime.now().isoformat(),
            'indicator_stats': dict(self.indicator_stats),
        }
        
        with open(os.path.join(self.storage_path, "final_weights.json"), 'w') as f:
            json.dump(data, f, indent=2)
    
    def run_backtest(self, symbol: str, interval: str, candles: int) -> Dict:
        """Run a single backtest and return results"""
        window = int(candles * 0.5)
        
        try:
            final_balance, trades, _ = backtest_strategy(
                symbol=symbol,
                interval=interval,
                candles=candles,
                window=window,
                initial_balance=self.initial_balance,
                risk_percentage=1.0,
                weights=self.weights,
                use_trailing_stop=True,
                trailing_stop_distance_percent=0.5,
            )
            
            profit = final_balance - self.initial_balance
            entry_trades = [t for t in trades if t.get('type') == 'entry']
            
            # Analyze exits
            wins = 0
            losses = 0
            for t in trades:
                if t.get('type') in ['take_profit_1', 'take_profit_2', 'take_profit_3']:
                    wins += 1
                elif t.get('type') == 'stop_loss':
                    losses += 1
                elif t.get('type') == 'exit_end_of_period':
                    if t.get('profit', 0) > 0:
                        wins += 1
                    else:
                        losses += 1
            
            return {
                'success': True,
                'profit': profit,
                'trades': len(entry_trades),
                'wins': wins,
                'losses': losses,
                'win_rate': (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0,
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'profit': 0, 'trades': 0}
    
    def update_weights(self, iteration_profit: float, iteration_trades: int, win_rate: float):
        """
        Update weights based on iteration performance.
        Uses multiple strategies for learning.
        """
        if iteration_trades == 0:
            return
        
        # Strategy 1: Random perturbation with momentum
        # If profit was good, keep direction. If bad, try opposite.
        for i in range(len(self.weights)):
            # Add random exploration
            noise = np.random.normal(0, self.learning_rate * 0.5)
            
            # Bias based on performance
            if iteration_profit > 0:
                # Profitable - small random changes
                self.weights[i] += noise * 0.5
            else:
                # Losing - try bigger changes
                self.weights[i] += noise * 1.5
            
            # Keep within bounds
            self.weights[i] = max(0.2, min(2.5, self.weights[i]))
        
        # Strategy 2: Win rate based adjustment
        # If win rate is low, reduce all weights slightly (be more selective)
        # If win rate is high, can afford to increase weights
        if win_rate < 40:
            for i in range(len(self.weights)):
                self.weights[i] *= 0.95  # Reduce by 5%
        elif win_rate > 60:
            for i in range(len(self.weights)):
                self.weights[i] *= 1.02  # Increase by 2%
        
        # Re-clamp
        for i in range(len(self.weights)):
            self.weights[i] = max(0.2, min(2.5, self.weights[i]))
    
    def run_iteration(self) -> Dict:
        """Run one complete iteration over all symbols and intervals"""
        self.iteration += 1
        
        print(f"\n{Colors.H}{'='*70}{Colors.E}")
        print(f"{Colors.H} ITERATION {self.iteration}{Colors.E}")
        print(f"{Colors.H}{'='*70}{Colors.E}")
        
        iteration_profit = 0.0
        iteration_trades = 0
        iteration_wins = 0
        iteration_losses = 0
        
        for symbol in SYMBOLS:
            print(f"\n{Colors.C}--- {symbol} ---{Colors.E}")
            
            for interval in INTERVALS:
                candles = CANDLES_PER_INTERVAL.get(interval, 400)
                print(f"  {interval:>3}: ", end="", flush=True)
                
                result = self.run_backtest(symbol, interval, candles)
                
                if result['success']:
                    profit = result['profit']
                    trades = result['trades']
                    wr = result['win_rate']
                    
                    iteration_profit += profit
                    iteration_trades += trades
                    iteration_wins += result['wins']
                    iteration_losses += result['losses']
                    
                    color = Colors.G if profit > 0 else Colors.R
                    print(f"{color}${profit:+8.2f}{Colors.E} | {trades:2d} trades | {wr:5.1f}% WR")
                else:
                    print(f"{Colors.R}FAILED: {result.get('error', 'Unknown')[:30]}{Colors.E}")
        
        # Update totals
        self.total_trades += iteration_trades
        self.total_profit += iteration_profit
        
        # Calculate iteration win rate
        iter_wr = (iteration_wins / (iteration_wins + iteration_losses) * 100 
                   if (iteration_wins + iteration_losses) > 0 else 0)
        
        # Print summary
        print(f"\n{Colors.BOLD}Iteration {self.iteration} Summary:{Colors.E}")
        color = Colors.G if iteration_profit > 0 else Colors.R
        print(f"  Profit: {color}${iteration_profit:+.2f}{Colors.E}")
        print(f"  Trades: {iteration_trades}")
        print(f"  Win Rate: {iter_wr:.1f}%")
        
        # Track best
        if iteration_profit > self.best_profit:
            self.best_profit = iteration_profit
            self.best_weights = self.weights.copy()
            print(f"  {Colors.G}★ NEW BEST!{Colors.E}")
        
        # Update weights AGGRESSIVELY based on performance
        self.update_weights(iteration_profit, iteration_trades, iter_wr)
        
        # Apply learning rate decay (prevents overfitting in later iterations)
        self.learning_rate = self.base_learning_rate * (self.lr_decay ** self.iteration)
        
        # Show weight changes
        print(f"\n{Colors.Y}Weight adjustments:{Colors.E}")
        changed = 0
        for i, (name, weight) in enumerate(zip(WEIGHT_NAMES, self.weights)):
            if i < len(self.weight_history) and len(self.weight_history) > 0:
                old = self.weight_history[-1][i] if self.weight_history else 1.0
                diff = weight - old
                if abs(diff) > 0.01:
                    direction = "↑" if diff > 0 else "↓"
                    print(f"  {name}: {old:.3f} → {weight:.3f} ({direction}{abs(diff):.3f})")
                    changed += 1
        
        if changed == 0:
            print("  (First iteration - weights established)")
        
        self.weight_history.append(self.weights.copy())
        self.profit_history.append(iteration_profit)
        
        # Save after each iteration
        self._save_weights()
        
        return {
            'iteration': self.iteration,
            'profit': iteration_profit,
            'trades': iteration_trades,
            'win_rate': iter_wr,
        }
    
    def run(self, iterations: int = 20) -> Dict:
        """Run multiple iterations of learning"""
        print(f"\n{Colors.H}{'='*70}{Colors.E}")
        print(f"{Colors.H}{'AGGRESSIVE SELF-LEARNING BACKTEST':^70}{Colors.E}")
        print(f"{Colors.H}{'='*70}{Colors.E}")
        
        print(f"\n{Colors.C}Configuration:{Colors.E}")
        print(f"  Iterations: {iterations}")
        print(f"  Symbols: {SYMBOLS}")
        print(f"  Intervals: {INTERVALS}")
        print(f"  Learning Rate: {self.learning_rate} (decay: {self.lr_decay}/iter)")
        print(f"  Initial Balance: ${self.initial_balance:,.2f}")
        
        # Show candle counts per interval
        total_candles = sum(CANDLES_PER_INTERVAL.values()) * len(SYMBOLS)
        print(f"\n{Colors.C}Data per iteration:{Colors.E}")
        for interval, candles in CANDLES_PER_INTERVAL.items():
            print(f"  {interval:>3}: {candles:,} candles × {len(SYMBOLS)} symbols")
        print(f"  Total: ~{total_candles:,} candles/iteration (expect 200-400+ trades)")
        
        print(f"\n{Colors.C}Starting weights:{Colors.E}")
        for name, weight in zip(WEIGHT_NAMES, self.weights):
            print(f"  {name}: {weight:.3f}")
        
        start_time = datetime.now()
        
        try:
            for _ in range(iterations):
                self.run_iteration()
        except KeyboardInterrupt:
            print(f"\n{Colors.Y}Interrupted - saving progress...{Colors.E}")
        
        # Final report
        self._print_final_report(start_time)
        
        return {
            'total_iterations': self.iteration,
            'total_trades': self.total_trades,
            'total_profit': self.total_profit,
            'best_profit': self.best_profit,
            'final_weights': self.weights,
            'best_weights': self.best_weights,
        }
    
    def _print_final_report(self, start_time: datetime):
        """Print final report"""
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\n{Colors.H}{'='*70}{Colors.E}")
        print(f"{Colors.H}{'FINAL REPORT':^70}{Colors.E}")
        print(f"{Colors.H}{'='*70}{Colors.E}")
        
        print(f"\n{Colors.BOLD}Overall Statistics:{Colors.E}")
        print(f"  Duration: {elapsed/60:.1f} minutes")
        print(f"  Iterations: {self.iteration}")
        print(f"  Total Trades: {self.total_trades}")
        
        color = Colors.G if self.total_profit > 0 else Colors.R
        print(f"  Total Profit: {color}${self.total_profit:+.2f}{Colors.E}")
        print(f"  Best Iteration: ${self.best_profit:+.2f}")
        
        # Profit trend
        if len(self.profit_history) > 1:
            first_half = sum(self.profit_history[:len(self.profit_history)//2])
            second_half = sum(self.profit_history[len(self.profit_history)//2:])
            trend = "📈 IMPROVING" if second_half > first_half else "📉 Declining"
            print(f"  Trend: {trend}")
        
        print(f"\n{Colors.BOLD}Final Optimized Weights:{Colors.E}")
        for name, weight in zip(WEIGHT_NAMES, self.weights):
            # Visual bar
            bar_len = int(weight * 10)
            bar = "█" * min(bar_len, 25) + "░" * max(0, 25 - bar_len)
            
            if weight > 1.5:
                status = f"{Colors.G}STRONG{Colors.E}"
            elif weight < 0.6:
                status = f"{Colors.R}WEAK{Colors.E}"
            else:
                status = ""
            
            print(f"  {name:<25} {weight:.4f} {bar} {status}")
        
        # Save
        self._save_weights()
        
        # Also save as JSON for easy use
        report_path = os.path.join(self.storage_path, "backtest_report.json")
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'iterations': self.iteration,
                'total_trades': self.total_trades,
                'total_profit': self.total_profit,
                'best_profit': self.best_profit,
                'profit_history': self.profit_history,
                'final_weights': dict(zip(WEIGHT_NAMES, self.weights)),
                'best_weights': dict(zip(WEIGHT_NAMES, self.best_weights)),
            }, f, indent=2)
        
        print(f"\n{Colors.G}✓ Report saved to: {report_path}{Colors.E}")
        print(f"{Colors.G}✓ Weights saved to: {self.storage_path}/final_weights.json{Colors.E}")


def main():
    parser = argparse.ArgumentParser(description="Aggressive self-learning backtest")
    parser.add_argument('--iterations', type=int, default=20,
                        help='Number of learning iterations (default: 20)')
    parser.add_argument('--balance', type=float, default=10000.0)
    parser.add_argument('--learning-rate', type=float, default=0.15,
                        help='How aggressively to change weights (default: 0.15)')
    parser.add_argument('--storage', type=str, default='./data/btc_eth_learning')
    parser.add_argument('--report-only', action='store_true')
    
    args = parser.parse_args()
    
    if args.report_only:
        weights_file = os.path.join(args.storage, "final_weights.json")
        if os.path.exists(weights_file):
            with open(weights_file) as f:
                data = json.load(f)
            print(json.dumps(data, indent=2))
        else:
            print("No saved weights found")
        return
    
    learner = AggressiveLearner(
        storage_path=args.storage,
        initial_balance=args.balance,
        learning_rate=args.learning_rate,
    )
    
    try:
        result = learner.run(iterations=args.iterations)
        
        print(f"\n{Colors.H}{'='*70}{Colors.E}")
        print(f"{Colors.H}{'BACKTEST COMPLETE':^70}{Colors.E}")
        print(f"{Colors.H}{'='*70}{Colors.E}")
        
        print(f"\n{Colors.G}✓ Completed {result['total_iterations']} iterations{Colors.E}")
        color = Colors.G if result['total_profit'] > 0 else Colors.R
        print(f"{Colors.G}✓ Total profit: {color}${result['total_profit']:+.2f}{Colors.E}")
        print(f"{Colors.B}ℹ Learning data saved to: {args.storage}{Colors.E}")
        
    except Exception as e:
        print(f"\n{Colors.R}Error: {e}{Colors.E}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
