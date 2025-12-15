"""
Enhanced Metrics Module for Backtesting System

Provides comprehensive, informative metrics including:
1. Indicator-level performance breakdown
2. Market regime analysis
3. Signal quality scoring
4. Trade attribution and root cause analysis
5. Learning suggestions based on performance patterns
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json
import os


@dataclass
class TradeMetrics:
    """Detailed metrics for a single trade"""
    trade_id: str
    symbol: str
    interval: str
    signal_type: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit_targets: List[float]
    
    # Outcome
    outcome: str  # tp1, tp2, tp3, stop_loss, trailing_stop, end_of_period
    profit_loss: float
    profit_loss_percent: float
    risk_reward_achieved: float
    
    # Duration
    entry_index: int
    exit_index: int
    duration_candles: int
    
    # Market context
    market_regime: str = ""
    volatility: float = 0.0
    volume_ratio: float = 1.0
    trend: str = ""
    
    # Indicator contributions
    contributing_indicators: List[str] = field(default_factory=list)
    indicator_scores: Dict[str, float] = field(default_factory=dict)
    
    # Trade quality metrics
    max_favorable_excursion: float = 0.0  # Best profit during trade
    max_adverse_excursion: float = 0.0    # Worst drawdown during trade
    r_multiple: float = 0.0               # Actual R achieved
    
    def is_winner(self) -> bool:
        return self.outcome in ['tp1', 'tp2', 'tp3', 'trailing_stop'] or self.profit_loss > 0
    
    def is_full_target(self) -> bool:
        return self.outcome == 'tp3'


class EnhancedMetricsCalculator:
    """
    Calculates and aggregates enhanced trading metrics with 
    detailed breakdowns for learning and optimization.
    """
    
    def __init__(self):
        self.trades: List[TradeMetrics] = []
        self.indicator_performance: Dict[str, Dict] = defaultdict(
            lambda: {
                'wins': 0, 'losses': 0, 'total_pnl': 0.0, 
                'avg_r': 0.0, 'r_values': [],
                'tp1_hits': 0, 'tp2_hits': 0, 'tp3_hits': 0, 'stop_hits': 0
            }
        )
        self.regime_performance: Dict[str, Dict] = defaultdict(
            lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'trades': 0}
        )
        self.symbol_performance: Dict[str, Dict] = defaultdict(
            lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'trades': 0}
        )
        self.interval_performance: Dict[str, Dict] = defaultdict(
            lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'trades': 0}
        )
    
    def add_trade(self, trade: TradeMetrics):
        """Add a completed trade to metrics"""
        self.trades.append(trade)
        self._update_indicator_stats(trade)
        self._update_regime_stats(trade)
        self._update_symbol_stats(trade)
        self._update_interval_stats(trade)
    
    def _update_indicator_stats(self, trade: TradeMetrics):
        """Update indicator-level statistics"""
        is_win = trade.is_winner()
        
        for indicator in trade.contributing_indicators:
            stats = self.indicator_performance[indicator]
            if is_win:
                stats['wins'] += 1
            else:
                stats['losses'] += 1
            
            stats['total_pnl'] += trade.profit_loss
            stats['r_values'].append(trade.r_multiple)
            
            # Track specific outcomes
            if trade.outcome == 'tp1':
                stats['tp1_hits'] += 1
            elif trade.outcome == 'tp2':
                stats['tp2_hits'] += 1
            elif trade.outcome == 'tp3':
                stats['tp3_hits'] += 1
            elif trade.outcome in ['stop_loss', 'trailing_stop']:
                stats['stop_hits'] += 1
    
    def _update_regime_stats(self, trade: TradeMetrics):
        """Update market regime statistics"""
        stats = self.regime_performance[trade.market_regime or 'unknown']
        stats['trades'] += 1
        stats['total_pnl'] += trade.profit_loss
        if trade.is_winner():
            stats['wins'] += 1
        else:
            stats['losses'] += 1
    
    def _update_symbol_stats(self, trade: TradeMetrics):
        """Update symbol-level statistics"""
        stats = self.symbol_performance[trade.symbol]
        stats['trades'] += 1
        stats['total_pnl'] += trade.profit_loss
        if trade.is_winner():
            stats['wins'] += 1
        else:
            stats['losses'] += 1
    
    def _update_interval_stats(self, trade: TradeMetrics):
        """Update interval/timeframe statistics"""
        stats = self.interval_performance[trade.interval]
        stats['trades'] += 1
        stats['total_pnl'] += trade.profit_loss
        if trade.is_winner():
            stats['wins'] += 1
        else:
            stats['losses'] += 1
    
    def calculate_comprehensive_metrics(self, 
                                         initial_balance: float,
                                         final_balance: float) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics with full breakdown"""
        
        if not self.trades:
            return self._empty_metrics()
        
        # Basic metrics
        total_trades = len(self.trades)
        winners = [t for t in self.trades if t.is_winner()]
        losers = [t for t in self.trades if not t.is_winner()]
        
        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
        
        # P/L calculations
        total_profit = sum(t.profit_loss for t in winners)
        total_loss = sum(t.profit_loss for t in losers)
        net_profit = total_profit + total_loss
        
        # Profit factor
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # R-multiple statistics
        r_multiples = [t.r_multiple for t in self.trades if t.r_multiple != 0]
        avg_r = np.mean(r_multiples) if r_multiples else 0
        
        # Drawdown calculation
        equity_curve = self._calculate_equity_curve(initial_balance)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Expectancy
        avg_win = np.mean([t.profit_loss for t in winners]) if winners else 0
        avg_loss = np.mean([t.profit_loss for t in losers]) if losers else 0
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
        
        # Exit type breakdown
        exit_breakdown = self._calculate_exit_breakdown()
        
        # Indicator performance breakdown
        indicator_breakdown = self._calculate_indicator_breakdown()
        
        # Regime performance breakdown
        regime_breakdown = self._calculate_regime_breakdown()
        
        # Symbol performance breakdown
        symbol_breakdown = self._calculate_symbol_breakdown()
        
        # Generate insights and recommendations
        insights = self._generate_insights()
        
        return {
            # Summary
            'summary': {
                'total_trades': total_trades,
                'winners': len(winners),
                'losers': len(losers),
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'net_profit': net_profit,
                'total_return_pct': (final_balance - initial_balance) / initial_balance * 100,
                'max_drawdown_pct': max_drawdown,
                'expectancy': expectancy,
                'avg_r_multiple': avg_r
            },
            
            # Detailed trade statistics
            'trade_stats': {
                'avg_winner': avg_win,
                'avg_loser': avg_loss,
                'largest_winner': max((t.profit_loss for t in winners), default=0),
                'largest_loser': min((t.profit_loss for t in losers), default=0),
                'avg_duration_candles': np.mean([t.duration_candles for t in self.trades]),
                'avg_mfe': np.mean([t.max_favorable_excursion for t in self.trades]),
                'avg_mae': np.mean([t.max_adverse_excursion for t in self.trades])
            },
            
            # Exit type breakdown
            'exit_breakdown': exit_breakdown,
            
            # Indicator performance
            'indicator_performance': indicator_breakdown,
            
            # Market regime performance
            'regime_performance': regime_breakdown,
            
            # Symbol performance
            'symbol_performance': symbol_breakdown,
            
            # Timeframe performance
            'interval_performance': self._calculate_interval_breakdown(),
            
            # Learning insights
            'insights': insights,
            
            # Equity curve for visualization
            'equity_curve': equity_curve
        }
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            'summary': {
                'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
                'net_profit': 0, 'max_drawdown_pct': 0, 'expectancy': 0
            },
            'trade_stats': {},
            'exit_breakdown': {},
            'indicator_performance': {},
            'regime_performance': {},
            'symbol_performance': {},
            'interval_performance': {},
            'insights': {'recommendations': ['Insufficient data for analysis']},
            'equity_curve': []
        }
    
    def _calculate_equity_curve(self, initial_balance: float) -> List[float]:
        """Calculate cumulative equity curve"""
        equity = [initial_balance]
        for trade in self.trades:
            equity.append(equity[-1] + trade.profit_loss)
        return equity
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown percentage"""
        if not equity_curve:
            return 0
        
        peak = equity_curve[0]
        max_dd = 0
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_exit_breakdown(self) -> Dict[str, Dict]:
        """Calculate breakdown by exit type"""
        breakdown = defaultdict(lambda: {'count': 0, 'total_pnl': 0.0, 'avg_pnl': 0.0})
        
        for trade in self.trades:
            stats = breakdown[trade.outcome]
            stats['count'] += 1
            stats['total_pnl'] += trade.profit_loss
        
        for outcome, stats in breakdown.items():
            if stats['count'] > 0:
                stats['avg_pnl'] = stats['total_pnl'] / stats['count']
                stats['percentage'] = stats['count'] / len(self.trades) * 100
        
        return dict(breakdown)
    
    def _calculate_indicator_breakdown(self) -> Dict[str, Dict]:
        """Calculate detailed indicator performance breakdown"""
        breakdown = {}
        
        for indicator, stats in self.indicator_performance.items():
            total = stats['wins'] + stats['losses']
            if total == 0:
                continue
            
            win_rate = stats['wins'] / total * 100
            avg_r = np.mean(stats['r_values']) if stats['r_values'] else 0
            
            # Calculate score for ranking
            # Score = win_rate * 0.4 + normalized_pnl * 0.3 + avg_r * 0.3
            pnl_score = min(1.0, max(-1.0, stats['total_pnl'] / 1000))  # Normalize to -1 to 1
            r_score = min(1.0, max(-1.0, avg_r / 3))  # Normalize R to -1 to 1
            score = win_rate / 100 * 0.4 + (pnl_score + 1) / 2 * 0.3 + (r_score + 1) / 2 * 0.3
            
            breakdown[indicator] = {
                'total_signals': total,
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': win_rate,
                'total_pnl': stats['total_pnl'],
                'avg_pnl': stats['total_pnl'] / total,
                'avg_r': avg_r,
                'tp1_hits': stats['tp1_hits'],
                'tp2_hits': stats['tp2_hits'],
                'tp3_hits': stats['tp3_hits'],
                'stop_hits': stats['stop_hits'],
                'score': score,
                'status': self._get_indicator_status(win_rate, stats['total_pnl'])
            }
        
        # Sort by score
        return dict(sorted(breakdown.items(), key=lambda x: x[1]['score'], reverse=True))
    
    def _get_indicator_status(self, win_rate: float, total_pnl: float) -> str:
        """Get indicator health status"""
        if win_rate >= 60 and total_pnl > 0:
            return 'excellent'
        elif win_rate >= 50 and total_pnl >= 0:
            return 'good'
        elif win_rate >= 40:
            return 'average'
        else:
            return 'poor'
    
    def _calculate_regime_breakdown(self) -> Dict[str, Dict]:
        """Calculate performance by market regime"""
        breakdown = {}
        
        for regime, stats in self.regime_performance.items():
            if stats['trades'] == 0:
                continue
            
            win_rate = stats['wins'] / stats['trades'] * 100
            breakdown[regime] = {
                'trades': stats['trades'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': win_rate,
                'total_pnl': stats['total_pnl'],
                'avg_pnl': stats['total_pnl'] / stats['trades'],
                'recommendation': self._get_regime_recommendation(regime, win_rate, stats['total_pnl'])
            }
        
        return breakdown
    
    def _get_regime_recommendation(self, regime: str, win_rate: float, pnl: float) -> str:
        """Get recommendation for trading in specific regime"""
        if win_rate >= 55 and pnl > 0:
            return f"Good performance in {regime} - continue trading"
        elif win_rate < 40 or pnl < 0:
            return f"Poor performance in {regime} - consider filtering signals or reducing position size"
        else:
            return f"Average performance in {regime} - monitor closely"
    
    def _calculate_symbol_breakdown(self) -> Dict[str, Dict]:
        """Calculate performance by symbol"""
        breakdown = {}
        
        for symbol, stats in self.symbol_performance.items():
            if stats['trades'] == 0:
                continue
            
            win_rate = stats['wins'] / stats['trades'] * 100
            breakdown[symbol] = {
                'trades': stats['trades'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': win_rate,
                'total_pnl': stats['total_pnl'],
                'avg_pnl': stats['total_pnl'] / stats['trades']
            }
        
        return dict(sorted(breakdown.items(), key=lambda x: x[1]['total_pnl'], reverse=True))
    
    def _calculate_interval_breakdown(self) -> Dict[str, Dict]:
        """Calculate performance by timeframe"""
        breakdown = {}
        
        for interval, stats in self.interval_performance.items():
            if stats['trades'] == 0:
                continue
            
            win_rate = stats['wins'] / stats['trades'] * 100
            breakdown[interval] = {
                'trades': stats['trades'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': win_rate,
                'total_pnl': stats['total_pnl'],
                'avg_pnl': stats['total_pnl'] / stats['trades']
            }
        
        return breakdown
    
    def _generate_insights(self) -> Dict[str, Any]:
        """Generate actionable insights from performance data"""
        insights = {
            'strong_indicators': [],
            'weak_indicators': [],
            'best_regimes': [],
            'worst_regimes': [],
            'recommendations': []
        }
        
        # Indicator insights
        for indicator, stats in self._calculate_indicator_breakdown().items():
            if stats['total_signals'] >= 10:  # Minimum sample size
                if stats['status'] == 'excellent':
                    insights['strong_indicators'].append({
                        'name': indicator,
                        'win_rate': stats['win_rate'],
                        'avg_pnl': stats['avg_pnl']
                    })
                elif stats['status'] == 'poor':
                    insights['weak_indicators'].append({
                        'name': indicator,
                        'win_rate': stats['win_rate'],
                        'avg_pnl': stats['avg_pnl']
                    })
        
        # Regime insights
        regime_breakdown = self._calculate_regime_breakdown()
        sorted_regimes = sorted(regime_breakdown.items(), 
                                key=lambda x: x[1]['win_rate'], reverse=True)
        
        if sorted_regimes:
            insights['best_regimes'] = [sorted_regimes[0][0]] if sorted_regimes else []
            insights['worst_regimes'] = [sorted_regimes[-1][0]] if len(sorted_regimes) > 1 else []
        
        # Generate recommendations
        if insights['weak_indicators']:
            weak_names = [i['name'] for i in insights['weak_indicators'][:3]]
            insights['recommendations'].append(
                f"Consider reducing weights for: {', '.join(weak_names)}"
            )
        
        if insights['strong_indicators']:
            strong_names = [i['name'] for i in insights['strong_indicators'][:3]]
            insights['recommendations'].append(
                f"Consider increasing weights for: {', '.join(strong_names)}"
            )
        
        if insights['worst_regimes']:
            worst_regime = insights['worst_regimes'][0]
            regime_stats = regime_breakdown.get(worst_regime, {})
            if regime_stats.get('win_rate', 100) < 40:
                insights['recommendations'].append(
                    f"Consider avoiding signals in '{worst_regime}' market regime"
                )
        
        # Check for recent performance decline
        if len(self.trades) >= 50:
            recent_trades = self.trades[-25:]
            older_trades = self.trades[-50:-25]
            
            recent_win_rate = sum(1 for t in recent_trades if t.is_winner()) / len(recent_trades) * 100
            older_win_rate = sum(1 for t in older_trades if t.is_winner()) / len(older_trades) * 100
            
            if recent_win_rate < older_win_rate - 10:
                insights['recommendations'].append(
                    f"Performance declining: Recent win rate {recent_win_rate:.1f}% vs "
                    f"previous {older_win_rate:.1f}%. Consider retraining weights."
                )
        
        return insights
    
    def print_detailed_report(self, initial_balance: float, final_balance: float):
        """Print detailed performance report"""
        metrics = self.calculate_comprehensive_metrics(initial_balance, final_balance)
        
        print("\n" + "="*80)
        print("📊 ENHANCED PERFORMANCE REPORT")
        print("="*80)
        
        # Summary
        s = metrics['summary']
        print(f"\n📈 SUMMARY")
        print(f"   Total Trades: {s['total_trades']}")
        print(f"   Win Rate: {s['win_rate']:.1f}% ({s['winners']}W / {s['losers']}L)")
        print(f"   Profit Factor: {s['profit_factor']:.2f}")
        print(f"   Net Profit: ${s['net_profit']:.2f}")
        print(f"   Total Return: {s['total_return_pct']:.2f}%")
        print(f"   Max Drawdown: {s['max_drawdown_pct']:.2f}%")
        print(f"   Expectancy: ${s['expectancy']:.2f}")
        print(f"   Avg R-Multiple: {s['avg_r_multiple']:.2f}R")
        
        # Exit breakdown
        if metrics['exit_breakdown']:
            print(f"\n📤 EXIT TYPE BREAKDOWN")
            for exit_type, stats in metrics['exit_breakdown'].items():
                print(f"   {exit_type}: {stats['count']} ({stats['percentage']:.1f}%) | "
                      f"Avg P/L: ${stats['avg_pnl']:.2f}")
        
        # Indicator performance
        if metrics['indicator_performance']:
            print(f"\n🎯 INDICATOR PERFORMANCE (Top 5)")
            for i, (indicator, stats) in enumerate(list(metrics['indicator_performance'].items())[:5]):
                status_emoji = {'excellent': '🟢', 'good': '🟡', 'average': '🟠', 'poor': '🔴'}
                emoji = status_emoji.get(stats['status'], '⚪')
                print(f"   {emoji} {indicator}: {stats['win_rate']:.1f}% WR | "
                      f"${stats['avg_pnl']:.2f} avg | {stats['avg_r']:.2f}R | "
                      f"TP1:{stats['tp1_hits']} TP2:{stats['tp2_hits']} TP3:{stats['tp3_hits']} SL:{stats['stop_hits']}")
        
        # Weak indicators
        weak_indicators = [ind for ind, stats in metrics['indicator_performance'].items() 
                          if stats['status'] == 'poor']
        if weak_indicators:
            print(f"\n🔴 UNDERPERFORMING INDICATORS")
            for ind in weak_indicators[:5]:
                stats = metrics['indicator_performance'][ind]
                print(f"   {ind}: {stats['win_rate']:.1f}% WR, ${stats['total_pnl']:.2f} total")
        
        # Regime performance
        if metrics['regime_performance']:
            print(f"\n🌍 MARKET REGIME PERFORMANCE")
            for regime, stats in metrics['regime_performance'].items():
                print(f"   {regime}: {stats['win_rate']:.1f}% WR ({stats['trades']} trades) | "
                      f"${stats['total_pnl']:.2f}")
        
        # Recommendations
        if metrics['insights']['recommendations']:
            print(f"\n💡 RECOMMENDATIONS")
            for rec in metrics['insights']['recommendations']:
                print(f"   • {rec}")
        
        print("\n" + "="*80)
        
        return metrics


def create_trade_metrics(
    trade_log: List[Dict],
    entry_trade: Dict,
    exit_trades: List[Dict],
    market_context: Dict[str, Any],
    indicator_contributions: Dict[str, float]
) -> TradeMetrics:
    """
    Helper to create TradeMetrics from trade log data.
    
    Args:
        trade_log: Full trade log
        entry_trade: Entry trade dictionary
        exit_trades: List of exit trades for this entry
        market_context: Market context at entry
        indicator_contributions: Indicator contributions to signal
    """
    
    # Calculate final outcome
    if not exit_trades:
        outcome = 'end_of_period'
        exit_price = entry_trade['price']
        profit_loss = 0
    else:
        last_exit = exit_trades[-1]
        outcome = last_exit.get('type', 'end_of_period')
        # Map trade types to outcome strings
        outcome_map = {
            'take_profit_1': 'tp1',
            'take_profit_2': 'tp2',
            'take_profit_3': 'tp3',
            'stop_loss': 'stop_loss',
            'trailing_stop': 'trailing_stop',
            'exit_end_of_period': 'end_of_period'
        }
        outcome = outcome_map.get(outcome, outcome)
        exit_price = last_exit.get('price', entry_trade['price'])
        profit_loss = sum(t.get('profit', 0) for t in exit_trades)
    
    # Calculate R-multiple
    risk = abs(entry_trade['price'] - entry_trade.get('stop_loss', entry_trade['price']))
    r_multiple = profit_loss / risk if risk > 0 else 0
    
    return TradeMetrics(
        trade_id=entry_trade.get('trade_id', str(entry_trade.get('index', 0))),
        symbol=entry_trade.get('symbol', 'UNKNOWN'),
        interval=entry_trade.get('interval', '1h'),
        signal_type=entry_trade.get('signal', '').split()[0] if entry_trade.get('signal') else 'Unknown',
        entry_price=entry_trade['price'],
        exit_price=exit_price,
        stop_loss=entry_trade.get('stop_loss', 0),
        take_profit_targets=[
            entry_trade.get('take_profit_1', 0),
            entry_trade.get('take_profit_2', 0),
            entry_trade.get('take_profit_3', 0)
        ],
        outcome=outcome,
        profit_loss=profit_loss,
        profit_loss_percent=(exit_price - entry_trade['price']) / entry_trade['price'] * 100 if entry_trade['price'] > 0 else 0,
        risk_reward_achieved=abs(profit_loss / risk) if risk > 0 else 0,
        entry_index=entry_trade.get('index', 0),
        exit_index=exit_trades[-1].get('index', 0) if exit_trades else entry_trade.get('index', 0),
        duration_candles=exit_trades[-1].get('index', 0) - entry_trade.get('index', 0) if exit_trades else 0,
        market_regime=market_context.get('market_regime', ''),
        volatility=market_context.get('volatility', 0),
        volume_ratio=market_context.get('volume_ratio', 1),
        trend=market_context.get('trend', ''),
        contributing_indicators=list(indicator_contributions.keys()),
        indicator_scores=indicator_contributions,
        r_multiple=r_multiple
    )


# Usage example
if __name__ == "__main__":
    # Create calculator
    calc = EnhancedMetricsCalculator()
    
    # Example trade (normally these would come from backtest)
    example_trade = TradeMetrics(
        trade_id="test_1",
        symbol="BTCUSDT",
        interval="1h",
        signal_type="Bullish",
        entry_price=50000,
        exit_price=51500,
        stop_loss=49000,
        take_profit_targets=[51000, 52000, 53000],
        outcome="tp2",
        profit_loss=150,
        profit_loss_percent=3.0,
        risk_reward_achieved=1.5,
        entry_index=100,
        exit_index=115,
        duration_candles=15,
        market_regime="trending_up",
        volatility=0.02,
        volume_ratio=1.2,
        trend="uptrend",
        contributing_indicators=["W_BULLISH_OB", "W_TREND", "W_STRUCTURE_BREAK"],
        indicator_scores={"W_BULLISH_OB": 1.2, "W_TREND": 0.8, "W_STRUCTURE_BREAK": 1.5},
        r_multiple=1.5
    )
    
    calc.add_trade(example_trade)
    calc.print_detailed_report(10000, 10150)

