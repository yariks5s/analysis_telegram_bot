#!/usr/bin/env python3
"""
Walk-Forward Validation Test

This script performs proper out-of-sample testing by:
1. Splitting historical data into TRAIN (70%) and TEST (30%) periods
2. Training/optimizing weights ONLY on the training data
3. Validating the learned weights on the test data (no further learning)

This tells you if your strategy generalizes or if it's overfitted to historical data.

Usage:
    python walk_forward_test.py [options]
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

# Add project root to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from src.analysis.utils.helpers import fetch_candles
from src.telegram.signals.detection import (
    generate_price_prediction_signal_proba,
    analyze_data,
    calculate_position_size,
)
from src.core.utils import create_true_preferences

# Configuration
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVALS = ["5m", "15m", "30m", "1h", "4h"]

WEIGHT_NAMES = [
    "W_BULLISH_OB", "W_BEARISH_OB", "W_BULLISH_BREAKER", "W_BEARISH_BREAKER",
    "W_ABOVE_SUPPORT", "W_BELOW_RESISTANCE", "W_FVG_ABOVE", "W_FVG_BELOW",
    "W_TREND", "W_SWEEP_HIGHS", "W_SWEEP_LOWS", "W_STRUCTURE_BREAK",
    "W_PIN_BAR", "W_ENGULFING", "W_LIQUIDITY_POOL_ABOVE", "W_LIQUIDITY_POOL_BELOW",
    "W_LIQUIDITY_POOL_ROUND", "W_RSI_EXTREME",
    "W_CONFIDENCE_THRESHOLD"
]

# Candle counts per interval
CANDLE_COUNTS = {
    "5m": 3000,   # ~10 days
    "15m": 2000,  # ~21 days  
    "30m": 1500,  # ~31 days
    "1h": 1200,   # ~50 days
    "4h": 1000,   # ~167 days
}

class Colors:
    H = '\033[95m'  # Header
    B = '\033[94m'  # Blue
    C = '\033[96m'  # Cyan
    G = '\033[92m'  # Green
    Y = '\033[93m'  # Yellow
    R = '\033[91m'  # Red
    E = '\033[0m'   # End
    BOLD = '\033[1m'


class WalkForwardValidator:
    """Performs walk-forward validation on trading strategy"""
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        initial_balance: float = 10000.0,
        risk_percentage: float = 1.0,
        learning_iterations: int = 10,
        learning_rate: float = 0.15,
        weights_file: str = None,
    ):
        self.train_ratio = train_ratio
        self.initial_balance = initial_balance
        self.risk_percentage = risk_percentage
        self.learning_iterations = learning_iterations
        self.learning_rate = learning_rate
        self.weights_file = weights_file
        
        self.preferences = create_true_preferences()
        
        # Results storage
        self.train_results = {}
        self.test_results = {}
        
        # Pattern tracking
        self.train_patterns = defaultdict(lambda: {'wins': 0, 'losses': 0, 'profit': 0.0})
        self.test_patterns = defaultdict(lambda: {'wins': 0, 'losses': 0, 'profit': 0.0})
        
        # Pattern keywords for detection
        self.PATTERN_KEYWORDS = {
            "bullish order block": "Bullish Order Block",
            "bearish order block": "Bearish Order Block",
            "order block": "Order Block",
            "bullish breaker block": "Bullish Breaker Block",
            "bearish breaker block": "Bearish Breaker Block",
            "breaker block": "Breaker Block",
            "fvg below": "FVG Below",
            "fvg above": "FVG Above",
            "unfilled fvg": "FVG",
            "near support": "Support Level",
            "near resistance": "Resistance Level",
            "swept through previous highs": "Liquidity Sweep (Highs)",
            "swept through previous lows": "Liquidity Sweep (Lows)",
            "broke structure upward": "Structure Break (Bullish)",
            "broke structure downward": "Structure Break (Bearish)",
            "broke structure": "Structure Break",
            "bullish pin bar": "Bullish Pin Bar",
            "bearish engulfing": "Bearish Engulfing",
            "pin bar": "Pin Bar",
            "engulfing": "Engulfing Pattern",
            "rsi oversold": "RSI Oversold",
            "rsi overbought": "RSI Overbought",
            "in an uptrend": "Uptrend",
            "in a downtrend": "Downtrend",
        }
    
    def _load_weights(self) -> List[float]:
        """Load weights from file or use defaults"""
        if self.weights_file and os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, 'r') as f:
                    data = json.load(f)
                    weights = data.get('weights', [])
                    if weights:
                        print(f"{Colors.G}✓ Loaded weights from {self.weights_file}{Colors.E}")
                        return weights
            except Exception as e:
                print(f"{Colors.Y}Warning: Could not load weights: {e}{Colors.E}")
        
        # Default starting weights
        return [
            1.3, 1.3, 1.1, 1.1,  # Order blocks, breakers
            0.8, 0.8, 0.6, 0.6,  # Support/resistance, FVG
            1.0, 1.4, 1.4, 1.5,  # Trend, sweeps, structure
            0.7, 0.7, 1.0, 1.0,  # Pin bar, engulfing, liq pools
            1.2, 0.5, 0.3        # Liq round, RSI, confidence
        ]
    
    def _extract_patterns(self, reason: str) -> List[str]:
        """Extract patterns from signal reason"""
        patterns = []
        reason_lower = reason.lower() if reason else ""
        for keyword, pattern_name in self.PATTERN_KEYWORDS.items():
            if keyword in reason_lower and pattern_name not in patterns:
                patterns.append(pattern_name)
        return patterns
    
    def _run_backtest_on_data(
        self, 
        df, 
        symbol: str, 
        interval: str,
        weights: List[float],
        is_training: bool = True
    ) -> Dict:
        """Run backtest on a specific dataframe"""
        
        balance = self.initial_balance
        position = 0
        entry_price = 0
        entry_reason = ""
        trades = []
        current_trade = None
        
        window_size = min(300, len(df) // 3)
        
        confidence_threshold = weights[18] if len(weights) > 18 else 0.3
        confidence_threshold = max(0.1, min(0.9, confidence_threshold))
        
        for i in range(window_size, len(df)):
            current_window = df.iloc[i - window_size:i + 1].copy()
            current_price = df["Close"].iloc[i]
            current_time = df.index[i]
            
            # Get indicators and signal
            try:
                indicators = analyze_data(current_window, self.preferences, 0.05)
                signal, prob, confidence, reason, trading_signal = (
                    generate_price_prediction_signal_proba(
                        current_window, indicators, weights, balance, self.risk_percentage
                    )
                )
            except Exception:
                continue
            
            # Check existing position
            if position > 0 and current_trade:
                # Check stop loss
                if current_price <= current_trade["stop_loss"]:
                    loss = position * (current_price - entry_price)
                    balance += position * current_price
                    
                    patterns = self._extract_patterns(entry_reason)
                    pattern_tracker = self.train_patterns if is_training else self.test_patterns
                    for p in patterns:
                        pattern_tracker[p]['losses'] += 1
                        pattern_tracker[p]['profit'] += loss
                    
                    trades.append({
                        "type": "stop_loss",
                        "entry": entry_price,
                        "exit": current_price,
                        "profit": loss,
                        "patterns": patterns
                    })
                    
                    position = 0
                    current_trade = None
                    continue
                
                # Check take profits
                if not current_trade.get("tp1_hit") and current_price >= current_trade["take_profit_1"]:
                    close_amount = current_trade["initial_position"] / 3
                    profit = close_amount * (current_price - entry_price)
                    balance += close_amount * current_price
                    position -= close_amount
                    current_trade["tp1_hit"] = True
                
                if not current_trade.get("tp2_hit") and current_price >= current_trade["take_profit_2"]:
                    close_amount = position / 2
                    profit = close_amount * (current_price - entry_price)
                    balance += close_amount * current_price
                    position -= close_amount
                    current_trade["tp2_hit"] = True
                
                if not current_trade.get("tp3_hit") and current_price >= current_trade["take_profit_3"]:
                    profit = position * (current_price - entry_price)
                    balance += position * current_price
                    
                    patterns = self._extract_patterns(entry_reason)
                    pattern_tracker = self.train_patterns if is_training else self.test_patterns
                    for p in patterns:
                        pattern_tracker[p]['wins'] += 1
                        pattern_tracker[p]['profit'] += profit
                    
                    trades.append({
                        "type": "take_profit",
                        "entry": entry_price,
                        "exit": current_price,
                        "profit": profit,
                        "patterns": patterns
                    })
                    
                    position = 0
                    current_trade = None
                    continue
            
            # New entry
            if signal == "Bullish" and position == 0 and trading_signal and confidence >= confidence_threshold:
                position_sizing = calculate_position_size(
                    balance, self.risk_percentage, current_price, trading_signal.stop_loss
                )
                
                position = position_sizing["position_size"]
                amount_to_invest = position * current_price
                
                # Cap position
                max_position = self.initial_balance * 0.2
                if amount_to_invest > max_position:
                    amount_to_invest = max_position
                    position = amount_to_invest / current_price
                
                if amount_to_invest > balance:
                    amount_to_invest = balance
                    position = balance / current_price
                
                if position > 0 and amount_to_invest >= 10:
                    balance -= amount_to_invest
                    entry_price = current_price
                    entry_reason = reason
                    
                    current_trade = {
                        "stop_loss": float(trading_signal.stop_loss),
                        "take_profit_1": float(trading_signal.take_profit_1),
                        "take_profit_2": float(trading_signal.take_profit_2),
                        "take_profit_3": float(trading_signal.take_profit_3),
                        "initial_position": position,
                        "tp1_hit": False,
                        "tp2_hit": False,
                        "tp3_hit": False,
                    }
        
        # Close remaining position
        if position > 0:
            final_price = df["Close"].iloc[-1]
            profit = position * (final_price - entry_price)
            balance += position * final_price
            trades.append({
                "type": "end_close",
                "entry": entry_price,
                "exit": final_price,
                "profit": profit,
            })
        
        total_profit = balance - self.initial_balance
        wins = len([t for t in trades if t.get("profit", 0) > 0])
        losses = len([t for t in trades if t.get("profit", 0) < 0])
        
        return {
            "profit": total_profit,
            "trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / len(trades) * 100) if trades else 0,
            "final_balance": balance
        }
    
    def _optimize_weights(self, train_data: Dict[str, Dict]) -> List[float]:
        """Run learning iterations on training data to optimize weights"""
        
        weights = self._load_weights()
        best_weights = weights.copy()
        best_profit = float('-inf')
        
        print(f"\n{Colors.C}Starting weight optimization ({self.learning_iterations} iterations)...{Colors.E}")
        
        for iteration in range(self.learning_iterations):
            total_profit = 0
            total_trades = 0
            total_wins = 0
            total_losses = 0
            
            for key, data in train_data.items():
                symbol, interval = key.split("_")
                result = self._run_backtest_on_data(
                    data['df'], symbol, interval, weights, is_training=True
                )
                total_profit += result['profit']
                total_trades += result['trades']
                total_wins += result['wins']
                total_losses += result['losses']
            
            win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
            
            # Track best
            if total_profit > best_profit:
                best_profit = total_profit
                best_weights = weights.copy()
            
            # Update weights based on performance
            for i in range(len(weights) - 1):  # Exclude confidence threshold
                noise = np.random.normal(0, self.learning_rate * 0.5)
                if total_profit > 0:
                    weights[i] += noise * 0.5
                else:
                    weights[i] += noise * 1.5
                weights[i] = max(0.2, min(2.5, weights[i]))
            
            # Adjust confidence threshold
            if win_rate < 40:
                weights[-1] = min(0.9, weights[-1] + 0.02)
            elif win_rate > 60:
                weights[-1] = max(0.1, weights[-1] - 0.01)
            weights[-1] = max(0.1, min(0.9, weights[-1]))
            
            color = Colors.G if total_profit > 0 else Colors.R
            print(f"  Iteration {iteration + 1:>2}: {color}${total_profit:>+10.2f}{Colors.E} | "
                  f"{total_trades} trades | {win_rate:.1f}% WR")
        
        print(f"\n{Colors.G}✓ Best training profit: ${best_profit:+.2f}{Colors.E}")
        return best_weights
    
    def run(self):
        """Run the complete walk-forward validation"""
        
        print(f"\n{Colors.H}{'='*70}{Colors.E}")
        print(f"{Colors.H}{'WALK-FORWARD VALIDATION':^70}{Colors.E}")
        print(f"{Colors.H}{'='*70}{Colors.E}")
        
        print(f"\n{Colors.C}Configuration:{Colors.E}")
        print(f"  Train/Test Split: {self.train_ratio*100:.0f}% / {(1-self.train_ratio)*100:.0f}%")
        print(f"  Learning Iterations: {self.learning_iterations}")
        print(f"  Initial Balance: ${self.initial_balance:,.2f}")
        print(f"  Symbols: {SYMBOLS}")
        print(f"  Intervals: {INTERVALS}")
        
        # Phase 1: Fetch and split data
        print(f"\n{Colors.H}{'='*70}{Colors.E}")
        print(f"{Colors.H}{'PHASE 1: DATA COLLECTION & SPLITTING':^70}{Colors.E}")
        print(f"{Colors.H}{'='*70}{Colors.E}")
        
        train_data = {}
        test_data = {}
        
        for symbol in SYMBOLS:
            for interval in INTERVALS:
                candles = CANDLE_COUNTS.get(interval, 1000)
                print(f"\n  Fetching {symbol} {interval} ({candles} candles)...", end=" ")
                
                try:
                    df = fetch_candles(symbol, candles, interval)
                    if df is None or df.empty:
                        print(f"{Colors.R}FAILED{Colors.E}")
                        continue
                    
                    df.sort_index(inplace=True)
                    
                    # Split data
                    split_idx = int(len(df) * self.train_ratio)
                    train_df = df.iloc[:split_idx].copy()
                    test_df = df.iloc[split_idx:].copy()
                    
                    key = f"{symbol}_{interval}"
                    train_data[key] = {
                        'df': train_df,
                        'start': train_df.index[0],
                        'end': train_df.index[-1],
                    }
                    test_data[key] = {
                        'df': test_df,
                        'start': test_df.index[0],
                        'end': test_df.index[-1],
                    }
                    
                    print(f"{Colors.G}OK{Colors.E} (Train: {len(train_df)}, Test: {len(test_df)})")
                    
                except Exception as e:
                    print(f"{Colors.R}ERROR: {e}{Colors.E}")
        
        if not train_data:
            print(f"\n{Colors.R}No data fetched. Exiting.{Colors.E}")
            return
        
        # Phase 2: Training
        print(f"\n{Colors.H}{'='*70}{Colors.E}")
        print(f"{Colors.H}{'PHASE 2: TRAINING (Weight Optimization)':^70}{Colors.E}")
        print(f"{Colors.H}{'='*70}{Colors.E}")
        
        # Show training period
        all_train_starts = [d['start'] for d in train_data.values()]
        all_train_ends = [d['end'] for d in train_data.values()]
        print(f"\n{Colors.C}Training Period:{Colors.E}")
        print(f"  From: {min(all_train_starts)}")
        print(f"  To:   {max(all_train_ends)}")
        
        optimized_weights = self._optimize_weights(train_data)
        
        # Calculate training results with optimized weights
        print(f"\n{Colors.C}Training Results (with optimized weights):{Colors.E}")
        train_total_profit = 0
        train_total_trades = 0
        train_total_wins = 0
        train_total_losses = 0
        
        for key, data in train_data.items():
            symbol, interval = key.split("_")
            result = self._run_backtest_on_data(
                data['df'], symbol, interval, optimized_weights, is_training=True
            )
            train_total_profit += result['profit']
            train_total_trades += result['trades']
            train_total_wins += result['wins']
            train_total_losses += result['losses']
            
            color = Colors.G if result['profit'] > 0 else Colors.R
            print(f"  {symbol} {interval}: {color}${result['profit']:>+10.2f}{Colors.E} | "
                  f"{result['trades']} trades | {result['win_rate']:.1f}% WR")
        
        train_win_rate = (train_total_wins / train_total_trades * 100) if train_total_trades > 0 else 0
        
        # Phase 3: Testing (Out-of-Sample)
        print(f"\n{Colors.H}{'='*70}{Colors.E}")
        print(f"{Colors.H}{'PHASE 3: TESTING (Out-of-Sample Validation)':^70}{Colors.E}")
        print(f"{Colors.H}{'='*70}{Colors.E}")
        
        all_test_starts = [d['start'] for d in test_data.values()]
        all_test_ends = [d['end'] for d in test_data.values()]
        print(f"\n{Colors.C}Test Period (UNSEEN DATA):{Colors.E}")
        print(f"  From: {min(all_test_starts)}")
        print(f"  To:   {max(all_test_ends)}")
        
        print(f"\n{Colors.C}Test Results:{Colors.E}")
        test_total_profit = 0
        test_total_trades = 0
        test_total_wins = 0
        test_total_losses = 0
        
        for key, data in test_data.items():
            symbol, interval = key.split("_")
            result = self._run_backtest_on_data(
                data['df'], symbol, interval, optimized_weights, is_training=False
            )
            test_total_profit += result['profit']
            test_total_trades += result['trades']
            test_total_wins += result['wins']
            test_total_losses += result['losses']
            
            color = Colors.G if result['profit'] > 0 else Colors.R
            print(f"  {symbol} {interval}: {color}${result['profit']:>+10.2f}{Colors.E} | "
                  f"{result['trades']} trades | {result['win_rate']:.1f}% WR")
        
        test_win_rate = (test_total_wins / test_total_trades * 100) if test_total_trades > 0 else 0
        
        # Final Report
        print(f"\n{Colors.H}{'='*70}{Colors.E}")
        print(f"{Colors.H}{'WALK-FORWARD VALIDATION RESULTS':^70}{Colors.E}")
        print(f"{Colors.H}{'='*70}{Colors.E}")
        
        print(f"\n{Colors.BOLD}Comparison:{Colors.E}")
        print(f"{'':20} {'TRAINING':>15} {'TESTING':>15} {'DIFF':>15}")
        print(f"{'-'*65}")
        
        train_color = Colors.G if train_total_profit > 0 else Colors.R
        test_color = Colors.G if test_total_profit > 0 else Colors.R
        diff_color = Colors.G if test_total_profit > train_total_profit * 0.5 else Colors.Y if test_total_profit > 0 else Colors.R
        
        print(f"{'Total Profit':20} {train_color}${train_total_profit:>+14,.2f}{Colors.E} "
              f"{test_color}${test_total_profit:>+14,.2f}{Colors.E} "
              f"{diff_color}{(test_total_profit/train_total_profit*100 if train_total_profit else 0):>+14.1f}%{Colors.E}")
        
        print(f"{'Total Trades':20} {train_total_trades:>15,} {test_total_trades:>15,} "
              f"{test_total_trades/train_total_trades*100 if train_total_trades else 0:>14.1f}%")
        
        print(f"{'Win Rate':20} {train_win_rate:>14.1f}% {test_win_rate:>14.1f}% "
              f"{test_win_rate - train_win_rate:>+14.1f}%")
        
        if train_total_trades > 0 and test_total_trades > 0:
            train_avg = train_total_profit / train_total_trades
            test_avg = test_total_profit / test_total_trades
            print(f"{'Avg Profit/Trade':20} ${train_avg:>14.2f} ${test_avg:>14.2f} "
                  f"${test_avg - train_avg:>+14.2f}")
        
        # Verdict
        print(f"\n{Colors.BOLD}Verdict:{Colors.E}")
        
        if test_total_profit > 0 and test_total_profit >= train_total_profit * 0.3:
            print(f"  {Colors.G}✅ STRATEGY GENERALIZES WELL{Colors.E}")
            print(f"  The strategy maintained {test_total_profit/train_total_profit*100:.1f}% of training profit on unseen data.")
            print(f"  This suggests the edge is REAL, not overfitted.")
        elif test_total_profit > 0:
            print(f"  {Colors.Y}⚠️  MODERATE GENERALIZATION{Colors.E}")
            print(f"  The strategy is profitable on test data but with reduced performance.")
            print(f"  Some overfitting may be present.")
        else:
            print(f"  {Colors.R}❌ POOR GENERALIZATION - LIKELY OVERFITTED{Colors.E}")
            print(f"  The strategy lost money on unseen data.")
            print(f"  The backtest profits were likely due to overfitting, not a real edge.")
        
        # Pattern analysis
        print(f"\n{Colors.BOLD}Pattern Performance (Test Data Only):{Colors.E}")
        if self.test_patterns:
            sorted_patterns = sorted(
                self.test_patterns.items(),
                key=lambda x: x[1]['profit']
            )
            for pattern, stats in sorted_patterns:
                total = stats['wins'] + stats['losses']
                if total > 0:
                    wr = stats['wins'] / total * 100
                    emoji = "🟢" if stats['profit'] > 0 else "🔴"
                    print(f"  {emoji} {pattern:<25}: {total:>3} trades | ${stats['profit']:>+10.2f} | {wr:>5.1f}% WR")
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "train_ratio": self.train_ratio,
            "training": {
                "profit": train_total_profit,
                "trades": train_total_trades,
                "wins": train_total_wins,
                "losses": train_total_losses,
                "win_rate": train_win_rate,
            },
            "testing": {
                "profit": test_total_profit,
                "trades": test_total_trades,
                "wins": test_total_wins,
                "losses": test_total_losses,
                "win_rate": test_win_rate,
            },
            "optimized_weights": dict(zip(WEIGHT_NAMES, optimized_weights)),
            "generalization_ratio": test_total_profit / train_total_profit if train_total_profit else 0,
        }
        
        output_path = os.path.join(
            os.path.dirname(__file__),
            f"data/walk_forward_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{Colors.G}✓ Results saved to: {output_path}{Colors.E}")
        print(f"{Colors.H}{'='*70}{Colors.E}")


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Validation Test")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Ratio of data to use for training (default: 0.7)")
    parser.add_argument("--balance", type=float, default=10000.0,
                        help="Initial balance (default: 10000)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Learning iterations for training phase (default: 10)")
    parser.add_argument("--learning-rate", type=float, default=0.15,
                        help="Learning rate for weight optimization (default: 0.15)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to starting weights file (optional)")
    
    args = parser.parse_args()
    
    validator = WalkForwardValidator(
        train_ratio=args.train_ratio,
        initial_balance=args.balance,
        learning_iterations=args.iterations,
        learning_rate=args.learning_rate,
        weights_file=args.weights,
    )
    
    validator.run()


if __name__ == "__main__":
    main()



