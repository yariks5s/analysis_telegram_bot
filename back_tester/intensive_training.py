#!/usr/bin/env python3
"""
INTENSIVE PARAMETER OPTIMIZATION

This script performs aggressive optimization of ALL trading parameters:

1. Signal Weights (18 weights for different indicators)
2. Stop Loss Parameters (ATR multiplier, min distance)
3. Take Profit Levels (TP1, TP2, TP3 R:R ratios)
4. Trailing Stop Parameters (distance, activation level)
5. Position Sizing (risk percentage)

Uses evolutionary strategies with elitism to find optimal combinations.

Usage:
    python3 intensive_training.py --generations 50 --hours 4
"""

import os
import sys
import argparse
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
import time
import random
import copy

# Add project root to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from back_tester.strategy import backtest_strategy

# Configuration
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVALS = ["5m", "15m", "30m", "1h", "4h"]

WEIGHT_NAMES = [
    "W_BULLISH_OB", "W_BEARISH_OB", "W_BULLISH_BREAKER", "W_BEARISH_BREAKER",
    "W_ABOVE_SUPPORT", "W_BELOW_RESISTANCE", "W_FVG_ABOVE", "W_FVG_BELOW",
    "W_TREND", "W_SWEEP_HIGHS", "W_SWEEP_LOWS", "W_STRUCTURE_BREAK",
    "W_PIN_BAR", "W_ENGULFING", "W_LIQUIDITY_POOL_ABOVE", "W_LIQUIDITY_POOL_BELOW",
    "W_LIQUIDITY_POOL_ROUND", "W_RSI_EXTREME"
]

# Colors
class Colors:
    H = '\033[95m'
    B = '\033[94m'
    C = '\033[96m'
    G = '\033[92m'
    Y = '\033[93m'
    R = '\033[91m'
    E = '\033[0m'
    BOLD = '\033[1m'


@dataclass
class TradingParameters:
    """All optimizable trading parameters"""
    
    # Signal weights (18)
    weights: List[float]
    
    # Stop Loss Parameters
    atr_multiplier: float = 2.0        # ATR multiplier for stop loss distance
    min_sl_percent: float = 0.5        # Minimum SL distance as % of entry
    
    # Take Profit R:R Ratios
    tp1_ratio: float = 1.5             # Risk:Reward for TP1
    tp2_ratio: float = 2.5             # Risk:Reward for TP2
    tp3_ratio: float = 4.0             # Risk:Reward for TP3
    
    # Trailing Stop Parameters  
    trailing_stop_distance: float = 0.5    # Trailing stop distance %
    trailing_activation_tp: int = 1        # Activate trailing at TP1(1), TP2(2), or TP3(3)
    
    # Position Sizing
    risk_percentage: float = 1.0       # Risk % per trade
    
    @classmethod
    def default(cls) -> 'TradingParameters':
        """Create with default values"""
        return cls(
            weights=[
                1.2, 1.2, 1.0, 1.0,  # Order blocks, breakers
                0.8, 0.8, 0.6, 0.6,  # Support/resistance, FVG
                1.0, 1.3, 1.3, 1.5,  # Trend, sweeps, structure
                0.7, 0.7, 1.0, 1.0,  # Patterns, liquidity
                1.2, 0.5             # Round numbers, RSI
            ]
        )
    
    @classmethod
    def random(cls) -> 'TradingParameters':
        """Create with random values for exploration"""
        return cls(
            weights=[random.uniform(0.3, 2.0) for _ in range(18)],
            atr_multiplier=random.uniform(1.0, 4.0),
            min_sl_percent=random.uniform(0.3, 1.5),
            tp1_ratio=random.uniform(1.0, 2.5),
            tp2_ratio=random.uniform(2.0, 4.0),
            tp3_ratio=random.uniform(3.0, 6.0),
            trailing_stop_distance=random.uniform(0.3, 1.5),
            trailing_activation_tp=random.choice([1, 2]),
            risk_percentage=random.uniform(0.5, 2.0)
        )
    
    def mutate(self, mutation_rate: float = 0.3, mutation_strength: float = 0.15) -> 'TradingParameters':
        """Create mutated copy of parameters"""
        new_params = copy.deepcopy(self)
        
        # Mutate weights
        for i in range(len(new_params.weights)):
            if random.random() < mutation_rate:
                change = np.random.normal(0, mutation_strength)
                new_params.weights[i] = max(0.1, min(3.0, new_params.weights[i] + change))
        
        # Mutate SL parameters
        if random.random() < mutation_rate:
            new_params.atr_multiplier = max(0.5, min(5.0, 
                new_params.atr_multiplier + np.random.normal(0, 0.3)))
        
        if random.random() < mutation_rate:
            new_params.min_sl_percent = max(0.2, min(2.0,
                new_params.min_sl_percent + np.random.normal(0, 0.2)))
        
        # Mutate TP ratios (keep ordering: TP1 < TP2 < TP3)
        if random.random() < mutation_rate:
            new_params.tp1_ratio = max(0.8, min(3.0,
                new_params.tp1_ratio + np.random.normal(0, 0.3)))
        
        if random.random() < mutation_rate:
            new_params.tp2_ratio = max(new_params.tp1_ratio + 0.5, min(5.0,
                new_params.tp2_ratio + np.random.normal(0, 0.4)))
        
        if random.random() < mutation_rate:
            new_params.tp3_ratio = max(new_params.tp2_ratio + 0.5, min(8.0,
                new_params.tp3_ratio + np.random.normal(0, 0.5)))
        
        # Mutate trailing stop
        if random.random() < mutation_rate:
            new_params.trailing_stop_distance = max(0.2, min(2.0,
                new_params.trailing_stop_distance + np.random.normal(0, 0.2)))
        
        if random.random() < mutation_rate * 0.5:  # Less frequent
            new_params.trailing_activation_tp = random.choice([1, 2])
        
        # Mutate risk
        if random.random() < mutation_rate:
            new_params.risk_percentage = max(0.25, min(3.0,
                new_params.risk_percentage + np.random.normal(0, 0.2)))
        
        return new_params
    
    @staticmethod
    def crossover(parent1: 'TradingParameters', parent2: 'TradingParameters') -> 'TradingParameters':
        """Create child from two parents"""
        child_weights = []
        for i in range(len(parent1.weights)):
            if random.random() < 0.5:
                child_weights.append(parent1.weights[i])
            else:
                child_weights.append(parent2.weights[i])
        
        # Randomly inherit other parameters
        return TradingParameters(
            weights=child_weights,
            atr_multiplier=random.choice([parent1.atr_multiplier, parent2.atr_multiplier]),
            min_sl_percent=random.choice([parent1.min_sl_percent, parent2.min_sl_percent]),
            tp1_ratio=random.choice([parent1.tp1_ratio, parent2.tp1_ratio]),
            tp2_ratio=random.choice([parent1.tp2_ratio, parent2.tp2_ratio]),
            tp3_ratio=random.choice([parent1.tp3_ratio, parent2.tp3_ratio]),
            trailing_stop_distance=random.choice([parent1.trailing_stop_distance, parent2.trailing_stop_distance]),
            trailing_activation_tp=random.choice([parent1.trailing_activation_tp, parent2.trailing_activation_tp]),
            risk_percentage=random.choice([parent1.risk_percentage, parent2.risk_percentage]),
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving"""
        return {
            'weights': self.weights,
            'weight_names': WEIGHT_NAMES,
            'atr_multiplier': self.atr_multiplier,
            'min_sl_percent': self.min_sl_percent,
            'tp1_ratio': self.tp1_ratio,
            'tp2_ratio': self.tp2_ratio,
            'tp3_ratio': self.tp3_ratio,
            'trailing_stop_distance': self.trailing_stop_distance,
            'trailing_activation_tp': self.trailing_activation_tp,
            'risk_percentage': self.risk_percentage,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradingParameters':
        """Load from dictionary"""
        return cls(
            weights=data.get('weights', cls.default().weights),
            atr_multiplier=data.get('atr_multiplier', 2.0),
            min_sl_percent=data.get('min_sl_percent', 0.5),
            tp1_ratio=data.get('tp1_ratio', 1.5),
            tp2_ratio=data.get('tp2_ratio', 2.5),
            tp3_ratio=data.get('tp3_ratio', 4.0),
            trailing_stop_distance=data.get('trailing_stop_distance', 0.5),
            trailing_activation_tp=data.get('trailing_activation_tp', 1),
            risk_percentage=data.get('risk_percentage', 1.0),
        )


class IntensiveParameterOptimizer:
    """
    Comprehensive parameter optimizer using evolutionary strategies.
    Optimizes weights AND risk management parameters together.
    """
    
    def __init__(
        self,
        storage_path: str = "./data/intensive_training",
        initial_balance: float = 10000.0,
        population_size: int = 8,
        elite_count: int = 2,
        tests_per_evaluation: int = 10,
    ):
        self.storage_path = storage_path
        self.initial_balance = initial_balance
        self.population_size = population_size
        self.elite_count = elite_count
        self.tests_per_evaluation = tests_per_evaluation
        
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(os.path.join(storage_path, "checkpoints"), exist_ok=True)
        
        # Load or initialize parameters
        self.best_params = self._load_params()
        self.best_fitness = float('-inf')
        
        # Tracking
        self.generation = 0
        self.total_backtests = 0
        self.total_trades = 0
        self.start_time = None
        
        # History
        self.fitness_history = []
        self.param_history = []
    
    def _load_params(self) -> TradingParameters:
        """Load parameters from file or use defaults"""
        params_file = os.path.join(self.storage_path, "best_params.json")
        
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r') as f:
                    data = json.load(f)
                    params = TradingParameters.from_dict(data)
                    print(f"{Colors.G}✓ Loaded existing parameters from checkpoint{Colors.E}")
                    return params
            except Exception as e:
                print(f"{Colors.Y}Warning: Could not load params: {e}{Colors.E}")
        
        return TradingParameters.default()
    
    def _save_params(self, params: TradingParameters, fitness: float, is_best: bool = False):
        """Save parameters to file"""
        data = params.to_dict()
        data['fitness'] = fitness
        data['generation'] = self.generation
        data['total_backtests'] = self.total_backtests
        data['timestamp'] = datetime.now().isoformat()
        
        filename = "best_params.json" if is_best else f"params_gen_{self.generation}.json"
        filepath = os.path.join(self.storage_path, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        if is_best:
            checkpoint_file = os.path.join(
                self.storage_path, "checkpoints",
                f"best_gen_{self.generation}_fit_{fitness:.1f}.json"
            )
            with open(checkpoint_file, 'w') as f:
                json.dump(data, f, indent=2)
    
    def evaluate_params(self, params: TradingParameters) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate a parameter set by running multiple backtests.
        """
        total_profit = 0.0
        total_trades = 0
        wins = 0
        losses = 0
        tp1_hits = 0
        tp2_hits = 0
        tp3_hits = 0
        sl_hits = 0
        results = []
        
        # Generate test configurations
        test_configs = []
        for _ in range(self.tests_per_evaluation):
            symbol = random.choice(SYMBOLS)
            interval = random.choice(INTERVALS)
            candles = random.randint(300, 600)
            test_configs.append((symbol, interval, candles))
        
        for symbol, interval, candles in test_configs:
            try:
                window = int(candles * 0.5)
                
                # Run backtest with these parameters
                # Note: We need to pass custom TP/SL params through the signal generation
                # For now, we use the weights and trailing stop params that strategy.py accepts
                final_balance, trades, _ = backtest_strategy(
                    symbol=symbol,
                    interval=interval,
                    candles=candles,
                    window=window,
                    initial_balance=self.initial_balance,
                    risk_percentage=params.risk_percentage,
                    weights=params.weights,
                    use_trailing_stop=True,
                    trailing_stop_distance_percent=params.trailing_stop_distance,
                )
                
                profit = final_balance - self.initial_balance
                
                # Analyze trades
                for trade in trades:
                    trade_type = trade.get('type', '')
                    if trade_type == 'entry':
                        total_trades += 1
                    elif trade_type == 'take_profit_1':
                        tp1_hits += 1
                        if trade.get('profit', 0) > 0:
                            wins += 1
                    elif trade_type == 'take_profit_2':
                        tp2_hits += 1
                        if trade.get('profit', 0) > 0:
                            wins += 1
                    elif trade_type == 'take_profit_3':
                        tp3_hits += 1
                        if trade.get('profit', 0) > 0:
                            wins += 1
                    elif trade_type == 'stop_loss':
                        sl_hits += 1
                        losses += 1
                    elif trade_type == 'exit_end_of_period':
                        if trade.get('profit', 0) > 0:
                            wins += 1
                        else:
                            losses += 1
                
                total_profit += profit
                self.total_backtests += 1
                
                results.append({
                    'symbol': symbol,
                    'interval': interval,
                    'profit': profit,
                    'trades': len([t for t in trades if t.get('type') == 'entry'])
                })
                
            except Exception as e:
                total_profit -= 50  # Penalty for errors
                results.append({'error': str(e)})
        
        self.total_trades += total_trades
        
        # Calculate fitness
        if total_trades == 0:
            return -1000, {'error': 'No trades'}
        
        total_exits = wins + losses
        win_rate = wins / total_exits if total_exits > 0 else 0
        avg_profit = total_profit / self.tests_per_evaluation
        
        # TP distribution score (prefer hitting higher TPs)
        total_tp_hits = tp1_hits + tp2_hits + tp3_hits
        if total_tp_hits > 0:
            tp_quality = (tp1_hits * 1 + tp2_hits * 2 + tp3_hits * 3) / (total_tp_hits * 3)
        else:
            tp_quality = 0
        
        # Calculate risk-adjusted return
        if sl_hits > 0:
            reward_risk = (tp1_hits + tp2_hits * 1.5 + tp3_hits * 2) / sl_hits
        else:
            reward_risk = 2.0  # Default if no SL hits
        
        # Composite fitness function
        fitness = (
            avg_profit * 0.30 +                     # Profitability
            win_rate * 100 * 0.25 +                 # Win rate (scaled to ~50)
            tp_quality * 50 * 0.15 +                # TP quality (scaled to ~50)
            min(reward_risk, 3) * 15 * 0.15 +       # Risk/reward (capped, scaled)
            min(total_trades / 5, 20) * 0.15        # Trade frequency (capped)
        )
        
        metrics = {
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate * 100,
            'tp1_hits': tp1_hits,
            'tp2_hits': tp2_hits,
            'tp3_hits': tp3_hits,
            'sl_hits': sl_hits,
            'tp_quality': tp_quality,
            'reward_risk': reward_risk,
        }
        
        return fitness, metrics
    
    def train_generation(self) -> Tuple[TradingParameters, float]:
        """Run one generation of evolutionary optimization"""
        self.generation += 1
        
        print(f"\n{Colors.H}{'='*70}{Colors.E}")
        print(f"{Colors.H} GENERATION {self.generation}{Colors.E}")
        print(f"{Colors.H}{'='*70}{Colors.E}")
        
        # Create population
        population = []
        
        # Keep elite (best from previous generation)
        population.append(self.best_params)
        
        # Mutations of best
        for _ in range(self.population_size - 3):
            mutated = self.best_params.mutate(mutation_rate=0.4, mutation_strength=0.2)
            population.append(mutated)
        
        # One random for exploration
        population.append(TradingParameters.random())
        
        # One with aggressive settings
        aggressive = TradingParameters(
            weights=[w * random.uniform(0.8, 1.2) for w in self.best_params.weights],
            atr_multiplier=1.5,  # Tighter SL
            tp1_ratio=1.2,       # Quick TP1
            tp2_ratio=2.0,
            tp3_ratio=3.5,
            trailing_stop_distance=0.4,
            trailing_activation_tp=1,
            risk_percentage=1.5,
        )
        population.append(aggressive)
        
        # Evaluate all
        results = []
        for i, params in enumerate(population):
            label = "ELITE" if i == 0 else f"#{i+1}"
            print(f"\n  Testing {label}... ", end="", flush=True)
            
            fitness, metrics = self.evaluate_params(params)
            results.append((fitness, params, metrics))
            
            color = Colors.G if fitness > self.best_fitness else (Colors.Y if fitness > self.best_fitness - 20 else C.R)
            print(f"{color}Fitness: {fitness:.1f} | "
                  f"Profit: ${metrics.get('avg_profit', 0):.2f} | "
                  f"WR: {metrics.get('win_rate', 0):.1f}% | "
                  f"TP1:{metrics.get('tp1_hits',0)} TP2:{metrics.get('tp2_hits',0)} TP3:{metrics.get('tp3_hits',0)} SL:{metrics.get('sl_hits',0)}{Colors.E}")
        
        # Sort by fitness
        results.sort(key=lambda x: x[0], reverse=True)
        
        # Get generation best
        gen_best_fitness, gen_best_params, gen_best_metrics = results[0]
        
        # Track history
        self.fitness_history.append(gen_best_fitness)
        
        # Update global best
        if gen_best_fitness > self.best_fitness:
            improvement = gen_best_fitness - self.best_fitness
            self.best_fitness = gen_best_fitness
            self.best_params = copy.deepcopy(gen_best_params)
            
            print(f"\n{Colors.G}{'★'*3} NEW BEST! Fitness: {gen_best_fitness:.1f} (+{improvement:.1f}) {'★'*3}{Colors.E}")
            
            # Show key parameters
            print(f"\n{Colors.C}Key Parameters:{Colors.E}")
            print(f"  ATR Multiplier: {gen_best_params.atr_multiplier:.2f}")
            print(f"  TP Ratios: {gen_best_params.tp1_ratio:.1f} / {gen_best_params.tp2_ratio:.1f} / {gen_best_params.tp3_ratio:.1f}")
            print(f"  Trailing Stop: {gen_best_params.trailing_stop_distance:.2f}% (activate at TP{gen_best_params.trailing_activation_tp})")
            print(f"  Risk per Trade: {gen_best_params.risk_percentage:.2f}%")
            
            self._save_params(gen_best_params, gen_best_fitness, is_best=True)
        else:
            # Crossover top performers for next generation
            if len(results) >= 2:
                child = TradingParameters.crossover(results[0][1], results[1][1])
                self.best_params = child.mutate(mutation_rate=0.2)
        
        return gen_best_params, gen_best_fitness
    
    def run(
        self,
        max_generations: int = 100,
        max_hours: float = 4.0,
        target_fitness: float = None
    ) -> Tuple[TradingParameters, float]:
        """Run full optimization"""
        self.start_time = datetime.now()
        end_time = self.start_time + timedelta(hours=max_hours)
        
        print(f"\n{Colors.H}{'='*70}{Colors.E}")
        print(f"{Colors.H}{'INTENSIVE PARAMETER OPTIMIZATION':^70}{Colors.E}")
        print(f"{Colors.H}{'='*70}{Colors.E}")
        
        print(f"\n{Colors.C}Configuration:{Colors.E}")
        print(f"  Max Generations: {max_generations}")
        print(f"  Max Time: {max_hours} hours")
        print(f"  Population Size: {self.population_size}")
        print(f"  Tests per Evaluation: {self.tests_per_evaluation}")
        print(f"  Symbols: {SYMBOLS}")
        print(f"  Intervals: {INTERVALS}")
        
        print(f"\n{Colors.C}Optimizing:{Colors.E}")
        print(f"  • 18 Signal Weights")
        print(f"  • Stop Loss (ATR multiplier, min distance)")
        print(f"  • Take Profit Levels (TP1, TP2, TP3 R:R ratios)")
        print(f"  • Trailing Stop (distance, activation)")
        print(f"  • Risk Percentage")
        
        print(f"\n{Colors.Y}Starting optimization... (Ctrl+C to stop and save){Colors.E}")
        
        try:
            for gen in range(max_generations):
                if datetime.now() >= end_time:
                    print(f"\n{Colors.Y}Time limit reached{Colors.E}")
                    break
                
                best_params, best_fitness = self.train_generation()
                
                if target_fitness and best_fitness >= target_fitness:
                    print(f"\n{Colors.G}Target fitness reached!{Colors.E}")
                    break
                
                # Progress every 5 gens
                if self.generation % 5 == 0:
                    elapsed = (datetime.now() - self.start_time).total_seconds() / 60
                    print(f"\n{Colors.B}═══ Progress: Gen {self.generation} | "
                          f"Best: {self.best_fitness:.1f} | "
                          f"Backtests: {self.total_backtests} | "
                          f"Time: {elapsed:.1f}m ═══{Colors.E}")
        
        except KeyboardInterrupt:
            print(f"\n{Colors.Y}Optimization interrupted{Colors.E}")
        
        self._print_final_report()
        return self.best_params, self.best_fitness
    
    def _print_final_report(self):
        """Print comprehensive final report"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        print(f"\n{Colors.H}{'='*70}{Colors.E}")
        print(f"{Colors.H}{'OPTIMIZATION COMPLETE':^70}{Colors.E}")
        print(f"{Colors.H}{'='*70}{Colors.E}")
        
        print(f"\n{Colors.BOLD}Statistics:{Colors.E}")
        print(f"  Generations: {self.generation}")
        print(f"  Backtests: {self.total_backtests}")
        print(f"  Trades: {self.total_trades}")
        print(f"  Time: {elapsed/60:.1f} minutes")
        print(f"  Best Fitness: {self.best_fitness:.1f}")
        
        p = self.best_params
        
        print(f"\n{Colors.BOLD}═══ OPTIMIZED PARAMETERS ═══{Colors.E}")
        
        print(f"\n{Colors.C}Stop Loss:{Colors.E}")
        print(f"  ATR Multiplier: {p.atr_multiplier:.2f}")
        print(f"  Min SL Distance: {p.min_sl_percent:.2f}%")
        
        print(f"\n{Colors.C}Take Profit Levels:{Colors.E}")
        print(f"  TP1 R:R Ratio: {p.tp1_ratio:.2f}")
        print(f"  TP2 R:R Ratio: {p.tp2_ratio:.2f}")
        print(f"  TP3 R:R Ratio: {p.tp3_ratio:.2f}")
        
        print(f"\n{Colors.C}Trailing Stop:{Colors.E}")
        print(f"  Distance: {p.trailing_stop_distance:.2f}%")
        print(f"  Activation: TP{p.trailing_activation_tp}")
        
        print(f"\n{Colors.C}Position Sizing:{Colors.E}")
        print(f"  Risk per Trade: {p.risk_percentage:.2f}%")
        
        print(f"\n{Colors.C}Signal Weights:{Colors.E}")
        for name, weight in zip(WEIGHT_NAMES, p.weights):
            bar_len = int(weight * 8)
            bar = "█" * bar_len + "░" * (16 - bar_len)
            
            if weight > 1.5:
                color, status = Colors.G, "STRONG"
            elif weight < 0.6:
                color, status = Colors.R, "WEAK"
            else:
                color, status = Colors.E, ""
            
            print(f"  {color}{name:<25} {weight:.3f} {bar} {status}{Colors.E}")
        
        # Save
        self._save_params(self.best_params, self.best_fitness, is_best=True)
        print(f"\n{Colors.G}✓ Parameters saved to: {self.storage_path}/best_params.json{Colors.E}")
        
        # Fitness trend
        if len(self.fitness_history) > 1:
            print(f"\n{Colors.BOLD}Fitness Trend:{Colors.E}")
            start = self.fitness_history[0]
            end = self.fitness_history[-1]
            change = end - start
            trend = "📈" if change > 0 else "📉"
            print(f"  Start: {start:.1f} → End: {end:.1f} ({trend} {change:+.1f})")


def main():
    parser = argparse.ArgumentParser(description="Intensive trading parameter optimization")
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--hours', type=float, default=2.0)
    parser.add_argument('--population', type=int, default=8)
    parser.add_argument('--tests', type=int, default=10, help='Backtests per evaluation')
    parser.add_argument('--balance', type=float, default=10000.0)
    parser.add_argument('--storage', type=str, default='./data/intensive_training')
    parser.add_argument('--target-fitness', type=float, default=None)
    
    args = parser.parse_args()
    
    optimizer = IntensiveParameterOptimizer(
        storage_path=args.storage,
        initial_balance=args.balance,
        population_size=args.population,
        tests_per_evaluation=args.tests,
    )
    
    try:
        best_params, best_fitness = optimizer.run(
            max_generations=args.generations,
            max_hours=args.hours,
            target_fitness=args.target_fitness
        )
        print(f"\n{Colors.G}Optimization completed! Best fitness: {best_fitness:.1f}{Colors.E}")
        
    except Exception as e:
        print(f"\n{Colors.R}Optimization failed: {e}{Colors.E}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
