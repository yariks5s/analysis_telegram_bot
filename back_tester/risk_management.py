from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


class AdaptiveRiskManager:
    """
    Enhanced risk management system with adaptive position sizing
    based on market conditions, volatility, and correlation.
    """

    def __init__(
        self,
        base_risk_pct: float = 1.0,
        max_risk_pct: float = 2.0,
        min_risk_pct: float = 0.2,
        volatility_factor: float = 1.0,
        correlation_factor: float = 0.5,
        market_regime_factor: float = 0.7,
        win_streak_factor: float = 0.1,
        loss_streak_factor: float = 0.15,
    ):
        """
        Initialize adaptive risk manager with specified parameters.

        Args:
            base_risk_pct: Base risk percentage per trade
            max_risk_pct: Maximum allowed risk percentage
            min_risk_pct: Minimum allowed risk percentage
            volatility_factor: How much volatility affects position sizing
            correlation_factor: How much correlation affects position sizing
            market_regime_factor: How much market regime affects position sizing
            win_streak_factor: Factor for increasing risk after consecutive wins
            loss_streak_factor: Factor for decreasing risk after consecutive losses
        """
        self.base_risk_pct = base_risk_pct
        self.max_risk_pct = max_risk_pct
        self.min_risk_pct = min_risk_pct
        self.volatility_factor = volatility_factor
        self.correlation_factor = correlation_factor
        self.market_regime_factor = market_regime_factor
        self.win_streak_factor = win_streak_factor
        self.loss_streak_factor = loss_streak_factor

        # Track trade performance
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.total_trades = 0
        self.winning_trades = 0

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        atr: Optional[float] = None,
        correlation_to_btc: Optional[float] = None,
        market_regime: Optional[str] = None,
        recent_trades: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate adaptive position size based on multiple factors.

        Args:
            account_balance: Current account balance
            entry_price: Entry price for the trade
            stop_loss: Stop loss price for the trade
            atr: Average True Range (optional)
            correlation_to_btc: Correlation with BTC (optional)
            market_regime: Current market regime (optional)
            recent_trades: List of recent trades for streaks (optional)

        Returns:
            Dictionary with position sizing details
        """
        # Update streaks if recent trades are provided
        if recent_trades:
            self._update_streaks(recent_trades)

        # Start with base risk
        adjusted_risk_pct = self.base_risk_pct
        risk_adjustments = {"base_risk": self.base_risk_pct}

        # 1. Adjust for volatility
        if atr is not None:
            volatility_adjustment = self._adjust_for_volatility(entry_price, atr)
            adjusted_risk_pct *= volatility_adjustment
            risk_adjustments["volatility_adjustment"] = volatility_adjustment

        # 2. Adjust for correlation
        if correlation_to_btc is not None:
            correlation_adjustment = self._adjust_for_correlation(correlation_to_btc)
            adjusted_risk_pct *= correlation_adjustment
            risk_adjustments["correlation_adjustment"] = correlation_adjustment

        # 3. Adjust for market regime
        if market_regime is not None:
            regime_adjustment = self._adjust_for_market_regime(market_regime)
            adjusted_risk_pct *= regime_adjustment
            risk_adjustments["regime_adjustment"] = regime_adjustment

        # 4. Adjust for win/loss streaks
        streak_adjustment = self._adjust_for_streaks()
        adjusted_risk_pct *= streak_adjustment
        risk_adjustments["streak_adjustment"] = streak_adjustment

        # Ensure risk stays within bounds
        adjusted_risk_pct = max(
            min(adjusted_risk_pct, self.max_risk_pct), self.min_risk_pct
        )

        # Calculate risk amount in currency
        risk_amount = account_balance * (adjusted_risk_pct / 100)

        # Calculate position size based on stop distance
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance <= 0:
            stop_distance = entry_price * 0.01  # Default to 1% if invalid stop

        position_size = risk_amount / stop_distance

        return {
            "position_size": position_size,
            "risk_percentage": adjusted_risk_pct,
            "risk_amount": risk_amount,
            "stop_distance": stop_distance,
            "risk_adjustments": risk_adjustments,
        }

    def _adjust_for_volatility(self, price: float, atr: float) -> float:
        """
        Adjust risk based on market volatility (ATR).
        Lower risk during high volatility periods.

        Args:
            price: Current price
            atr: Average True Range

        Returns:
            Adjustment multiplier for risk percentage
        """
        # Calculate normalized ATR as percentage of price
        normalized_atr = (atr / price) * 100

        # Benchmark: 1% ATR is "normal" volatility
        normal_volatility = 1.0

        # If volatility is higher than normal, reduce risk
        if normalized_atr > normal_volatility:
            # Inverse relationship: higher volatility = lower position size
            adjustment = normal_volatility / (normalized_atr * self.volatility_factor)
            return max(0.5, adjustment)  # Don't reduce below 50%

        # If volatility is lower than normal, can slightly increase risk
        elif normalized_atr < normal_volatility:
            adjustment = (
                1 + ((normal_volatility - normalized_atr) / normal_volatility) * 0.2
            )
            return min(1.2, adjustment)  # Don't increase above 120%

        return 1.0

    def _adjust_for_correlation(self, correlation: float) -> float:
        """
        Adjust risk based on correlation to BTC.
        Lower risk during high correlation periods (market-wide moves).

        Args:
            correlation: Correlation coefficient to BTC (-1 to 1)

        Returns:
            Adjustment multiplier for risk percentage
        """
        # High absolute correlation (positive or negative) suggests systemic risk
        abs_corr = abs(correlation)

        if abs_corr > 0.7:
            # Strong correlation - reduce risk
            reduction = (abs_corr - 0.7) / 0.3 * self.correlation_factor
            return max(0.7, 1.0 - reduction)

        # Low correlation means more idiosyncratic price action - can increase risk slightly
        elif abs_corr < 0.3:
            increase = (0.3 - abs_corr) / 0.3 * 0.1
            return min(1.1, 1.0 + increase)

        return 1.0

    def _adjust_for_market_regime(self, regime: str) -> float:
        """
        Adjust risk based on market regime.

        Args:
            regime: Current market regime classification

        Returns:
            Adjustment multiplier for risk percentage
        """
        # Reduce risk in volatile or ranging markets
        if regime.lower() in ["volatile", "ranging"]:
            return max(0.7, 1.0 - self.market_regime_factor * 0.3)

        # Slightly increase risk in trending markets
        elif regime.lower() in ["trending_up", "trending_down"]:
            return min(1.2, 1.0 + self.market_regime_factor * 0.2)

        # Default - no adjustment
        return 1.0

    def _adjust_for_streaks(self) -> float:
        """
        Adjust risk based on consecutive win/loss streaks.

        Returns:
            Adjustment multiplier for risk percentage
        """
        # Increase risk slightly after consecutive wins
        if self.consecutive_wins > 2:
            win_adjustment = min(self.consecutive_wins * self.win_streak_factor, 0.3)
            return 1.0 + win_adjustment

        # Decrease risk after consecutive losses
        elif self.consecutive_losses > 1:
            loss_adjustment = min(
                self.consecutive_losses * self.loss_streak_factor, 0.5
            )
            return max(0.5, 1.0 - loss_adjustment)

        return 1.0

    def _update_streaks(self, recent_trades: List[Dict[str, Any]]) -> None:
        """
        Update win/loss streaks based on recent trades.

        Args:
            recent_trades: List of recent trades
        """
        if not recent_trades:
            return

        # Sort trades by timestamp if available
        sorted_trades = sorted(
            recent_trades,
            key=lambda t: t.get("timestamp", t.get("exit_timestamp", 0)),
            reverse=True,
        )

        # Reset streaks
        self.consecutive_wins = 0
        self.consecutive_losses = 0

        # Count consecutive wins/losses
        for trade in sorted_trades:
            profit = trade.get("profit", trade.get("profit_loss", 0))

            if profit > 0:
                self.consecutive_wins += 1
                self.winning_trades += 1
                if self.consecutive_losses > 0:
                    break
            elif profit < 0:
                self.consecutive_losses += 1
                if self.consecutive_wins > 0:
                    break
            else:
                # Profit = 0 breaks the streak
                break

        self.total_trades = len(sorted_trades)

    def update_after_trade(self, trade: Dict[str, Any]) -> None:
        """
        Update risk manager's state after a completed trade.

        Args:
            trade: Completed trade information
        """
        profit = trade.get("profit", trade.get("profit_loss", 0))

        if profit > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.winning_trades += 1
        elif profit < 0:
            self.consecutive_wins = 0
            self.consecutive_losses += 1
        else:
            # Profit = 0 resets both streaks
            self.consecutive_wins = 0
            self.consecutive_losses = 0

        self.total_trades += 1


class DynamicStopLossManager:
    """
    Manager for dynamic stop-loss placement and adjustment.
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        min_atr_multiplier: float = 1.0,
        max_atr_multiplier: float = 4.0,
        support_resistance_factor: float = 0.1,
        trailing_activation_pct: float = 1.0,
        trailing_step_pct: float = 0.5,
    ):
        """
        Initialize the dynamic stop loss manager.

        Args:
            atr_multiplier: Base multiplier for ATR-based stops
            min_atr_multiplier: Minimum ATR multiplier
            max_atr_multiplier: Maximum ATR multiplier
            support_resistance_factor: Weight given to support/resistance levels
            trailing_activation_pct: Percentage of profit target at which trailing stop activates
            trailing_step_pct: Step size for trailing stop as percentage of price
        """
        self.atr_multiplier = atr_multiplier
        self.min_atr_multiplier = min_atr_multiplier
        self.max_atr_multiplier = max_atr_multiplier
        self.support_resistance_factor = support_resistance_factor
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_step_pct = trailing_step_pct

    def calculate_initial_stop_loss(
        self,
        df: pd.DataFrame,
        signal_type: str,
        entry_price: float,
        atr: float,
        support_levels: Optional[List[float]] = None,
        resistance_levels: Optional[List[float]] = None,
    ) -> float:
        """
        Calculate initial stop loss based on ATR and market structure.

        Args:
            df: DataFrame with OHLCV data
            signal_type: "buy" or "sell"
            entry_price: Entry price for the trade
            atr: Average True Range
            support_levels: List of support levels (optional)
            resistance_levels: List of resistance levels (optional)

        Returns:
            Stop loss price
        """
        # Base ATR stop
        if signal_type.lower() == "buy":
            base_stop = entry_price - (atr * self.atr_multiplier)
        else:
            base_stop = entry_price + (atr * self.atr_multiplier)

        # Adjust based on support/resistance if available
        if signal_type.lower() == "buy" and support_levels:
            # Find nearest support level below entry
            valid_supports = [s for s in support_levels if s < entry_price]
            if valid_supports:
                nearest_support = max(valid_supports)
                # Blend ATR stop with support level
                blended_stop = (
                    base_stop * (1 - self.support_resistance_factor)
                    + nearest_support * self.support_resistance_factor
                )
                # Don't place stop too far away
                return max(blended_stop, entry_price - (atr * self.max_atr_multiplier))

        elif signal_type.lower() == "sell" and resistance_levels:
            # Find nearest resistance level above entry
            valid_resistances = [r for r in resistance_levels if r > entry_price]
            if valid_resistances:
                nearest_resistance = min(valid_resistances)
                # Blend ATR stop with resistance level
                blended_stop = (
                    base_stop * (1 - self.support_resistance_factor)
                    + nearest_resistance * self.support_resistance_factor
                )
                # Don't place stop too far away
                return min(blended_stop, entry_price + (atr * self.max_atr_multiplier))

        return base_stop

    def adjust_trailing_stop(
        self,
        current_price: float,
        entry_price: float,
        current_stop: float,
        take_profit: float,
        signal_type: str,
    ) -> float:
        """
        Adjust trailing stop loss as price moves in favorable direction.

        Args:
            current_price: Current market price
            entry_price: Original entry price
            current_stop: Current stop loss level
            take_profit: Take profit target
            signal_type: "buy" or "sell"

        Returns:
            Updated stop loss price
        """
        # For long positions
        if signal_type.lower() == "buy":
            # Calculate how far we've moved toward take profit
            price_movement = current_price - entry_price
            target_distance = take_profit - entry_price

            # Only activate trailing stop if we've moved sufficiently toward target
            if price_movement <= 0 or target_distance <= 0:
                return current_stop

            movement_percentage = price_movement / target_distance

            if movement_percentage >= self.trailing_activation_pct / 100:
                # Calculate new stop based on trailing step
                trailing_distance = current_price * (self.trailing_step_pct / 100)
                new_stop = current_price - trailing_distance

                # Only move stop up, never down
                if new_stop > current_stop:
                    return new_stop

        # For short positions
        else:
            # Calculate how far we've moved toward take profit
            price_movement = entry_price - current_price
            target_distance = entry_price - take_profit

            # Only activate trailing stop if we've moved sufficiently toward target
            if price_movement <= 0 or target_distance <= 0:
                return current_stop

            movement_percentage = price_movement / target_distance

            if movement_percentage >= self.trailing_activation_pct / 100:
                # Calculate new stop based on trailing step
                trailing_distance = current_price * (self.trailing_step_pct / 100)
                new_stop = current_price + trailing_distance

                # Only move stop down, never up
                if new_stop < current_stop:
                    return new_stop

        # If conditions for adjusting stop aren't met, return current stop
        return current_stop
