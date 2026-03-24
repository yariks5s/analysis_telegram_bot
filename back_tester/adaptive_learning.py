"""
Adaptive Learning Module for Backtesting System

This module provides self-learning capabilities for the backtesting system:
1. SignalPerformanceTracker - Tracks individual signal outcomes with indicator attribution
2. AdaptiveWeightAdjuster - Adjusts weights based on signal performance
3. IndicatorAttributor - Attributes success/failure to specific indicators
4. LearningMetricsAnalyzer - Provides detailed analytics on what works and what doesn't

The system learns from failures and automatically adjusts coefficients to improve future signals.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class SignalOutcome(Enum):
    """Possible outcomes for a trading signal"""
    TP1_HIT = "tp1_hit"
    TP2_HIT = "tp2_hit"
    TP3_HIT = "tp3_hit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    END_OF_PERIOD = "end_of_period"
    BREAKEVEN = "breakeven"
    PENDING = "pending"


@dataclass
class SignalRecord:
    """Complete record of a signal with its contributing factors and outcome"""
    signal_id: str
    timestamp: datetime
    symbol: str
    interval: str
    signal_type: str  # Bullish/Bearish
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    
    # Contributing indicators and their scores
    indicator_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Reasons that triggered the signal
    reasons: List[str] = field(default_factory=list)
    
    # Market context at signal time
    market_regime: str = ""
    volatility: float = 0.0
    volume_ratio: float = 1.0
    rsi: float = 50.0
    trend: str = ""
    
    # Outcome tracking
    outcome: SignalOutcome = SignalOutcome.PENDING
    exit_price: float = 0.0
    profit_loss: float = 0.0
    profit_loss_percent: float = 0.0
    duration_candles: int = 0
    max_favorable_excursion: float = 0.0  # Max profit during trade
    max_adverse_excursion: float = 0.0    # Max loss during trade
    
    # Weights used at signal time
    weights_at_signal: List[float] = field(default_factory=list)
    
    def is_successful(self) -> bool:
        """Determine if the signal was successful"""
        return self.outcome in [SignalOutcome.TP1_HIT, SignalOutcome.TP2_HIT, 
                                SignalOutcome.TP3_HIT, SignalOutcome.TRAILING_STOP]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['outcome'] = self.outcome.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SignalRecord':
        """Create from dictionary"""
        data = data.copy()
        data['outcome'] = SignalOutcome(data['outcome'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class SignalPerformanceTracker:
    """
    Tracks individual signal performance and attributes outcomes to specific indicators.
    This enables learning from both successes and failures.
    """
    
    # Weight names for attribution
    WEIGHT_NAMES = [
        "W_BULLISH_OB", "W_BEARISH_OB", "W_BULLISH_BREAKER", "W_BEARISH_BREAKER",
        "W_ABOVE_SUPPORT", "W_BELOW_RESISTANCE", "W_FVG_ABOVE", "W_FVG_BELOW",
        "W_TREND", "W_SWEEP_HIGHS", "W_SWEEP_LOWS", "W_STRUCTURE_BREAK",
        "W_PIN_BAR", "W_ENGULFING", "W_LIQUIDITY_POOL_ABOVE", "W_LIQUIDITY_POOL_BELOW",
        "W_LIQUIDITY_POOL_ROUND", "W_RSI_EXTREME"
    ]
    
    def __init__(self, storage_path: str = "./data/signal_history"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        self.signals: List[SignalRecord] = []
        self.performance_by_indicator: Dict[str, Dict] = defaultdict(
            lambda: {"wins": 0, "losses": 0, "total_pnl": 0.0, "count": 0}
        )
        self.performance_by_market_regime: Dict[str, Dict] = defaultdict(
            lambda: {"wins": 0, "losses": 0, "total_pnl": 0.0, "count": 0}
        )
        self.performance_by_symbol: Dict[str, Dict] = defaultdict(
            lambda: {"wins": 0, "losses": 0, "total_pnl": 0.0, "count": 0}
        )
        
        self._load_history()
    
    def _load_history(self):
        """Load signal history from storage"""
        history_file = os.path.join(self.storage_path, "signal_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.signals = [SignalRecord.from_dict(s) for s in data.get('signals', [])]
                    self._rebuild_performance_stats()
                logger.info(f"Loaded {len(self.signals)} historical signals")
            except Exception as e:
                logger.warning(f"Failed to load signal history: {e}")
    
    def _save_history(self):
        """Save signal history to storage"""
        history_file = os.path.join(self.storage_path, "signal_history.json")
        try:
            with open(history_file, 'w') as f:
                json.dump({
                    'signals': [s.to_dict() for s in self.signals[-10000:]],  # Keep last 10k
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save signal history: {e}")
    
    def _rebuild_performance_stats(self):
        """Rebuild performance statistics from loaded signals"""
        for signal in self.signals:
            if signal.outcome != SignalOutcome.PENDING:
                self._update_stats(signal)
    
    def _update_stats(self, signal: SignalRecord):
        """Update performance statistics for a completed signal"""
        is_win = signal.is_successful()
        
        # Update by indicator
        for indicator, contribution in signal.indicator_contributions.items():
            if contribution > 0:
                stats = self.performance_by_indicator[indicator]
                stats["count"] += 1
                stats["total_pnl"] += signal.profit_loss
                if is_win:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1
        
        # Update by market regime
        regime_stats = self.performance_by_market_regime[signal.market_regime]
        regime_stats["count"] += 1
        regime_stats["total_pnl"] += signal.profit_loss
        if is_win:
            regime_stats["wins"] += 1
        else:
            regime_stats["losses"] += 1
        
        # Update by symbol
        symbol_stats = self.performance_by_symbol[signal.symbol]
        symbol_stats["count"] += 1
        symbol_stats["total_pnl"] += signal.profit_loss
        if is_win:
            symbol_stats["wins"] += 1
        else:
            symbol_stats["losses"] += 1
    
    def record_signal(self, 
                      signal_id: str,
                      symbol: str,
                      interval: str,
                      signal_type: str,
                      entry_price: float,
                      stop_loss: float,
                      take_profit_1: float,
                      take_profit_2: float,
                      take_profit_3: float,
                      indicator_contributions: Dict[str, float],
                      reasons: List[str],
                      market_context: Dict[str, Any],
                      weights: List[float]) -> SignalRecord:
        """Record a new signal with its contributing factors"""
        
        signal = SignalRecord(
            signal_id=signal_id,
            timestamp=datetime.now(),
            symbol=symbol,
            interval=interval,
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            indicator_contributions=indicator_contributions,
            reasons=reasons,
            market_regime=market_context.get("market_regime", "unknown"),
            volatility=market_context.get("volatility", 0.0),
            volume_ratio=market_context.get("volume_ratio", 1.0),
            rsi=market_context.get("rsi", 50.0),
            trend=market_context.get("trend", ""),
            weights_at_signal=weights.copy() if weights else []
        )
        
        self.signals.append(signal)
        return signal
    
    def update_signal_outcome(self, 
                              signal_id: str,
                              outcome: SignalOutcome,
                              exit_price: float,
                              profit_loss: float,
                              duration_candles: int,
                              max_favorable: float = 0.0,
                              max_adverse: float = 0.0):
        """Update a signal with its outcome"""
        
        for signal in reversed(self.signals):  # Search from most recent
            if signal.signal_id == signal_id:
                signal.outcome = outcome
                signal.exit_price = exit_price
                signal.profit_loss = profit_loss
                if signal.signal_type == "Bearish":
                    signal.profit_loss_percent = (signal.entry_price - exit_price) / signal.entry_price * 100
                else:
                    signal.profit_loss_percent = (exit_price - signal.entry_price) / signal.entry_price * 100
                signal.duration_candles = duration_candles
                signal.max_favorable_excursion = max_favorable
                signal.max_adverse_excursion = max_adverse
                
                self._update_stats(signal)
                self._save_history()
                
                logger.info(f"Signal {signal_id} outcome: {outcome.value}, P/L: {profit_loss:.2f}")
                return signal
        
        logger.warning(f"Signal {signal_id} not found for outcome update")
        return None
    
    def get_indicator_performance(self) -> pd.DataFrame:
        """Get performance breakdown by indicator"""
        data = []
        for indicator, stats in self.performance_by_indicator.items():
            if stats["count"] > 0:
                win_rate = stats["wins"] / stats["count"] * 100
                avg_pnl = stats["total_pnl"] / stats["count"]
                data.append({
                    "indicator": indicator,
                    "signals": stats["count"],
                    "wins": stats["wins"],
                    "losses": stats["losses"],
                    "win_rate": win_rate,
                    "total_pnl": stats["total_pnl"],
                    "avg_pnl": avg_pnl,
                    "score": win_rate * 0.4 + (avg_pnl / 100) * 0.6 if avg_pnl else win_rate
                })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values("score", ascending=False)
        return df
    
    def get_regime_performance(self) -> pd.DataFrame:
        """Get performance breakdown by market regime"""
        data = []
        for regime, stats in self.performance_by_market_regime.items():
            if stats["count"] > 0:
                win_rate = stats["wins"] / stats["count"] * 100
                data.append({
                    "regime": regime,
                    "signals": stats["count"],
                    "win_rate": win_rate,
                    "total_pnl": stats["total_pnl"],
                    "avg_pnl": stats["total_pnl"] / stats["count"]
                })
        return pd.DataFrame(data)
    
    def get_recent_performance(self, n: int = 100) -> Dict[str, Any]:
        """Get performance of recent N signals"""
        recent = [s for s in self.signals[-n:] if s.outcome != SignalOutcome.PENDING]
        
        if not recent:
            return {"signals": 0, "win_rate": 0, "avg_pnl": 0}
        
        wins = sum(1 for s in recent if s.is_successful())
        total_pnl = sum(s.profit_loss for s in recent)
        
        return {
            "signals": len(recent),
            "wins": wins,
            "losses": len(recent) - wins,
            "win_rate": wins / len(recent) * 100,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(recent),
            "avg_duration": sum(s.duration_candles for s in recent) / len(recent)
        }
    
    def identify_weak_indicators(self, min_signals: int = 10, 
                                  win_rate_threshold: float = 40.0) -> List[str]:
        """Identify indicators that consistently underperform"""
        weak = []
        for indicator, stats in self.performance_by_indicator.items():
            if stats["count"] >= min_signals:
                win_rate = stats["wins"] / stats["count"] * 100
                if win_rate < win_rate_threshold:
                    weak.append(indicator)
        return weak
    
    def identify_strong_indicators(self, min_signals: int = 10,
                                    win_rate_threshold: float = 60.0) -> List[str]:
        """Identify indicators that consistently outperform"""
        strong = []
        for indicator, stats in self.performance_by_indicator.items():
            if stats["count"] >= min_signals:
                win_rate = stats["wins"] / stats["count"] * 100
                if win_rate >= win_rate_threshold:
                    strong.append(indicator)
        return strong


class AdaptiveWeightAdjuster:
    """
    Adjusts signal weights based on historical performance.
    Uses online learning to continuously improve signal quality.
    """
    
    def __init__(self, 
                 performance_tracker: SignalPerformanceTracker,
                 learning_rate: float = 0.05,
                 momentum: float = 0.9,
                 min_weight: float = 0.1,
                 max_weight: float = 3.0,
                 decay_factor: float = 0.95,
                 storage_path: str = "./data/weights"):
        
        self.tracker = performance_tracker
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.decay_factor = decay_factor  # Decay old performance influence
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # Weight velocity for momentum-based updates
        self.velocity = defaultdict(float)
        
        # Historical weight adjustments
        self.adjustment_history: List[Dict] = []
        
        # Load last known weights
        self.current_weights = self._load_weights()
    
    def _load_weights(self) -> Dict[str, float]:
        """Load current weights from storage"""
        weights_file = os.path.join(self.storage_path, "adaptive_weights.json")
        default_weights = {name: 1.0 for name in SignalPerformanceTracker.WEIGHT_NAMES}
        
        if os.path.exists(weights_file):
            try:
                with open(weights_file, 'r') as f:
                    data = json.load(f)
                    return data.get('weights', default_weights)
            except Exception as e:
                logger.warning(f"Failed to load weights from {weights_file}: {e}")
        return default_weights
    
    def _save_weights(self):
        """Save current weights to storage"""
        weights_file = os.path.join(self.storage_path, "adaptive_weights.json")
        with open(weights_file, 'w') as f:
            json.dump({
                'weights': self.current_weights,
                'last_updated': datetime.now().isoformat(),
                'learning_rate': self.learning_rate,
                'adjustment_count': len(self.adjustment_history)
            }, f, indent=2)
    
    def calculate_weight_adjustments(self, recent_window: int = 100) -> Dict[str, float]:
        """
        Calculate weight adjustments based on recent signal performance.
        
        For each indicator:
        - If win rate > 60%: increase weight
        - If win rate < 40%: decrease weight
        - Scale adjustment by confidence (number of signals)
        """
        adjustments = {}
        indicator_perf = self.tracker.get_indicator_performance()
        
        if indicator_perf.empty:
            return adjustments
        
        for _, row in indicator_perf.iterrows():
            indicator = row['indicator']
            if indicator not in self.current_weights:
                continue
            
            signals = row['signals']
            win_rate = row['win_rate']
            avg_pnl = row['avg_pnl']
            
            # Calculate confidence based on sample size
            confidence = min(1.0, signals / 50)  # Full confidence at 50 signals
            
            # Calculate adjustment direction and magnitude
            if win_rate > 60:
                # Positive adjustment - increase weight
                magnitude = (win_rate - 50) / 100 * confidence
                adjustment = magnitude * self.learning_rate
            elif win_rate < 40:
                # Negative adjustment - decrease weight
                magnitude = (50 - win_rate) / 100 * confidence
                adjustment = -magnitude * self.learning_rate
            else:
                adjustment = 0
            
            # Factor in P/L performance
            if avg_pnl != 0:
                pnl_factor = np.clip(avg_pnl / 100, -0.5, 0.5)
                adjustment += pnl_factor * 0.5 * self.learning_rate * confidence
            
            # Apply momentum
            self.velocity[indicator] = (self.momentum * self.velocity[indicator] + 
                                        (1 - self.momentum) * adjustment)
            adjustments[indicator] = self.velocity[indicator]
        
        return adjustments
    
    def apply_adjustments(self, 
                          adjustments: Optional[Dict[str, float]] = None,
                          force: bool = False) -> Dict[str, float]:
        """
        Apply calculated adjustments to current weights.
        
        Args:
            adjustments: Optional pre-calculated adjustments
            force: If True, apply even small adjustments
            
        Returns:
            Dictionary of new weights
        """
        if adjustments is None:
            adjustments = self.calculate_weight_adjustments()
        
        if not adjustments:
            return self.current_weights
        
        old_weights = self.current_weights.copy()
        changes_made = False
        
        for indicator, adjustment in adjustments.items():
            if indicator not in self.current_weights:
                continue
            
            # Only apply significant adjustments unless forced
            if not force and abs(adjustment) < 0.01:
                continue
            
            old_weight = self.current_weights[indicator]
            new_weight = old_weight + adjustment
            
            # Clamp to valid range
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))
            
            if new_weight != old_weight:
                self.current_weights[indicator] = new_weight
                changes_made = True
                logger.info(f"Weight adjusted: {indicator} {old_weight:.3f} -> {new_weight:.3f}")
        
        if changes_made:
            self.adjustment_history.append({
                'timestamp': datetime.now().isoformat(),
                'old_weights': old_weights,
                'new_weights': self.current_weights.copy(),
                'adjustments': adjustments
            })
            self._save_weights()
        
        return self.current_weights
    
    def get_weights_as_list(self) -> List[float]:
        """Convert weights dictionary to list format for backtesting"""
        return [self.current_weights.get(name, 1.0) 
                for name in SignalPerformanceTracker.WEIGHT_NAMES]
    
    def reset_weights(self):
        """Reset all weights to default values"""
        self.current_weights = {name: 1.0 for name in SignalPerformanceTracker.WEIGHT_NAMES}
        self.velocity = defaultdict(float)
        self._save_weights()
        logger.info("Weights reset to defaults")
    
    def get_adjustment_summary(self) -> Dict[str, Any]:
        """Get summary of recent weight adjustments"""
        if not self.adjustment_history:
            return {"adjustments": 0, "last_adjustment": None}
        
        recent = self.adjustment_history[-10:]
        return {
            "total_adjustments": len(self.adjustment_history),
            "recent_adjustments": len(recent),
            "last_adjustment": recent[-1]['timestamp'] if recent else None,
            "current_weights": self.current_weights
        }


class LearningMetricsAnalyzer:
    """
    Analyzes learning metrics and provides actionable insights
    for improving signal quality.
    """
    
    def __init__(self, 
                 tracker: SignalPerformanceTracker,
                 adjuster: AdaptiveWeightAdjuster):
        self.tracker = tracker
        self.adjuster = adjuster
    
    def generate_learning_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "indicator_analysis": {},
            "regime_analysis": {},
            "recommendations": []
        }
        
        # Overall summary
        recent_perf = self.tracker.get_recent_performance(100)
        report["summary"] = {
            "total_signals_tracked": len(self.tracker.signals),
            "recent_win_rate": recent_perf.get("win_rate", 0),
            "recent_avg_pnl": recent_perf.get("avg_pnl", 0),
            "weight_adjustments": len(self.adjuster.adjustment_history)
        }
        
        # Indicator analysis
        indicator_df = self.tracker.get_indicator_performance()
        if not indicator_df.empty:
            report["indicator_analysis"] = {
                "best_performers": indicator_df.head(5).to_dict('records'),
                "worst_performers": indicator_df.tail(5).to_dict('records'),
                "weak_indicators": self.tracker.identify_weak_indicators(),
                "strong_indicators": self.tracker.identify_strong_indicators()
            }
        
        # Regime analysis
        regime_df = self.tracker.get_regime_performance()
        if not regime_df.empty:
            report["regime_analysis"] = regime_df.to_dict('records')
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations()
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on performance data"""
        recommendations = []
        
        # Check for weak indicators
        weak = self.tracker.identify_weak_indicators()
        if weak:
            recommendations.append(
                f"Consider reducing weights for underperforming indicators: {', '.join(weak)}"
            )
        
        # Check for strong indicators
        strong = self.tracker.identify_strong_indicators()
        if strong:
            recommendations.append(
                f"High-performing indicators (consider increasing weight): {', '.join(strong)}"
            )
        
        # Check recent performance trend
        recent_perf = self.tracker.get_recent_performance(50)
        older_perf = self.tracker.get_recent_performance(200)
        
        if recent_perf.get("win_rate", 0) < older_perf.get("win_rate", 0) - 10:
            recommendations.append(
                "Recent performance declining - consider resetting weights or retraining"
            )
        
        # Regime-specific recommendations
        regime_df = self.tracker.get_regime_performance()
        if not regime_df.empty:
            worst_regime = regime_df.loc[regime_df['win_rate'].idxmin()]
            if worst_regime['win_rate'] < 40 and worst_regime['signals'] > 20:
                recommendations.append(
                    f"Poor performance in '{worst_regime['regime']}' regime "
                    f"({worst_regime['win_rate']:.1f}% win rate) - consider filtering signals"
                )
        
        return recommendations
    
    def print_report(self):
        """Print formatted learning report to console"""
        report = self.generate_learning_report()
        
        print("\n" + "="*70)
        print("📊 ADAPTIVE LEARNING REPORT")
        print("="*70)
        
        print(f"\n📈 Summary:")
        print(f"   Total signals tracked: {report['summary']['total_signals_tracked']}")
        print(f"   Recent win rate: {report['summary']['recent_win_rate']:.1f}%")
        print(f"   Recent avg P/L: ${report['summary']['recent_avg_pnl']:.2f}")
        print(f"   Weight adjustments made: {report['summary']['weight_adjustments']}")
        
        if report.get('indicator_analysis', {}).get('best_performers'):
            print(f"\n🟢 Top Performing Indicators:")
            for ind in report['indicator_analysis']['best_performers'][:3]:
                print(f"   {ind['indicator']}: {ind['win_rate']:.1f}% win rate, "
                      f"${ind['avg_pnl']:.2f} avg P/L ({ind['signals']} signals)")
        
        if report.get('indicator_analysis', {}).get('weak_indicators'):
            print(f"\n🔴 Underperforming Indicators:")
            for ind in report['indicator_analysis']['weak_indicators']:
                print(f"   {ind}")
        
        if report.get('recommendations'):
            print(f"\n💡 Recommendations:")
            for rec in report['recommendations']:
                print(f"   • {rec}")
        
        print("\n" + "="*70)


class SelfLearningBacktester:
    """
    Enhanced backtester with self-learning capabilities.
    Integrates signal tracking, weight adjustment, and performance analysis.
    """
    
    def __init__(self, 
                 storage_path: str = "./data/learning",
                 auto_adjust_weights: bool = True,
                 adjustment_frequency: int = 50):  # Adjust every N signals
        
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        self.tracker = SignalPerformanceTracker(
            storage_path=os.path.join(storage_path, "signals")
        )
        self.adjuster = AdaptiveWeightAdjuster(
            performance_tracker=self.tracker,
            storage_path=os.path.join(storage_path, "weights")
        )
        self.analyzer = LearningMetricsAnalyzer(self.tracker, self.adjuster)
        
        self.auto_adjust = auto_adjust_weights
        self.adjustment_frequency = adjustment_frequency
        self.signals_since_adjustment = 0
    
    def get_current_weights(self) -> List[float]:
        """Get current adaptive weights as list"""
        return self.adjuster.get_weights_as_list()
    
    def record_signal_entry(self,
                            signal_id: str,
                            symbol: str,
                            interval: str,
                            signal_type: str,
                            entry_price: float,
                            stop_loss: float,
                            tp1: float, tp2: float, tp3: float,
                            indicator_contributions: Dict[str, float],
                            reasons: List[str],
                            market_context: Dict[str, Any]) -> SignalRecord:
        """Record a new signal entry"""
        
        signal = self.tracker.record_signal(
            signal_id=signal_id,
            symbol=symbol,
            interval=interval,
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            indicator_contributions=indicator_contributions,
            reasons=reasons,
            market_context=market_context,
            weights=self.get_current_weights()
        )
        
        return signal
    
    def record_signal_exit(self,
                           signal_id: str,
                           outcome: str,  # "tp1", "tp2", "tp3", "stop_loss", "trailing_stop", "end"
                           exit_price: float,
                           profit_loss: float,
                           duration: int):
        """Record signal exit and trigger learning if needed"""
        
        # Map outcome string to enum
        outcome_map = {
            "tp1": SignalOutcome.TP1_HIT,
            "tp2": SignalOutcome.TP2_HIT,
            "tp3": SignalOutcome.TP3_HIT,
            "stop_loss": SignalOutcome.STOP_LOSS,
            "trailing_stop": SignalOutcome.TRAILING_STOP,
            "end": SignalOutcome.END_OF_PERIOD
        }
        outcome_enum = outcome_map.get(outcome, SignalOutcome.END_OF_PERIOD)
        
        self.tracker.update_signal_outcome(
            signal_id=signal_id,
            outcome=outcome_enum,
            exit_price=exit_price,
            profit_loss=profit_loss,
            duration_candles=duration
        )
        
        self.signals_since_adjustment += 1
        
        # Auto-adjust weights if enabled and threshold reached
        if self.auto_adjust and self.signals_since_adjustment >= self.adjustment_frequency:
            self._trigger_learning()
            self.signals_since_adjustment = 0
    
    def _trigger_learning(self):
        """Trigger weight adjustment based on recent performance"""
        logger.info("Triggering adaptive weight adjustment...")
        
        adjustments = self.adjuster.calculate_weight_adjustments()
        if adjustments:
            self.adjuster.apply_adjustments(adjustments)
            logger.info(f"Applied {len(adjustments)} weight adjustments")
    
    def force_learning(self):
        """Force immediate weight adjustment"""
        self._trigger_learning()
        self.signals_since_adjustment = 0
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return self.analyzer.generate_learning_report()
    
    def print_status(self):
        """Print current learning status"""
        self.analyzer.print_report()
    
    def save_state(self):
        """Save all learning state"""
        self.tracker._save_history()
        self.adjuster._save_weights()
        logger.info("Learning state saved")


# Helper function to extract indicator contributions from signal generation
def extract_indicator_contributions(
    bullish_score: float,
    bearish_score: float,
    signal_type: str,
    reasons: List[str]
) -> Dict[str, float]:
    """
    Extract which indicators contributed to the signal.
    This parses the reasons list to determine contributions.
    """
    contributions = {}
    
    # Parse reasons to identify contributing indicators
    indicator_keywords = {
        "order block": ("W_BULLISH_OB" if signal_type == "Bullish" else "W_BEARISH_OB"),
        "breaker block": ("W_BULLISH_BREAKER" if signal_type == "Bullish" else "W_BEARISH_BREAKER"),
        "support": "W_ABOVE_SUPPORT",
        "resistance": "W_BELOW_RESISTANCE",
        "FVG above": "W_FVG_ABOVE",
        "FVG below": "W_FVG_BELOW",
        "uptrend": "W_TREND",
        "downtrend": "W_TREND",
        "swept through previous highs": "W_SWEEP_HIGHS",
        "swept through previous lows": "W_SWEEP_LOWS",
        "broke structure": "W_STRUCTURE_BREAK",
        "pin bar": "W_PIN_BAR",
        "engulfing": "W_ENGULFING",
        "liquidity pool": "W_LIQUIDITY_POOL_ABOVE",
        "round number": "W_LIQUIDITY_POOL_ROUND",
        "RSI oversold": "W_RSI_EXTREME",
        "RSI overbought": "W_RSI_EXTREME"
    }
    
    for reason in reasons:
        reason_lower = reason.lower()
        for keyword, indicator in indicator_keywords.items():
            if keyword.lower() in reason_lower:
                # Score is proportional to contribution to final signal
                total_score = bullish_score + bearish_score if bullish_score + bearish_score > 0 else 1
                score = bullish_score if signal_type == "Bullish" else bearish_score
                contributions[indicator] = score / total_score
    
    return contributions


# Usage example
if __name__ == "__main__":
    # Initialize self-learning backtester
    learner = SelfLearningBacktester(
        storage_path="./data/learning",
        auto_adjust_weights=True,
        adjustment_frequency=50
    )
    
    # Get current adaptive weights for signal generation
    weights = learner.get_current_weights()
    print(f"Current weights: {weights}")
    
    # Print performance status
    learner.print_status()

