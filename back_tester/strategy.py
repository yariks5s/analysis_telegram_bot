import os
import sys
from typing import List, Tuple, Optional, Dict
import random
from datetime import datetime, timedelta
import uuid

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)


# These imports use the system path we added above
from data_fetching_instruments import fetch_candles, analyze_data
from signal_detection import (
    generate_price_prediction_signal_proba,
    calculate_position_size,
)

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from utils import create_true_preferences
from .db_operations import ClickHouseDB
from .enhanced_trade_management import (
    calculate_trailing_stop,
    should_exit_based_on_time,
    calculate_volatility_based_take_profits,
    calculate_atr_based_stops,
)
from .enhanced_risk_management import calculate_tighter_stop_loss
from .market_filters import (
    detect_market_regime,
    should_trade_in_regime,
    is_favorable_trading_time,
    confirm_entry_criteria,
)


def backtest_strategy(
    symbol: str,
    interval: str,
    candles: int = 1000,
    window: int = 300,
    initial_balance: float = 10000.0,
    liq_lev_tolerance: float = 0.05,
    weights: list = [],
    risk_percentage: float = 1.0,
    # Trade entry/exit parameters
    use_trailing_stop: bool = True,
    trail_activation_threshold: float = 1.0,
    trail_percent: float = 0.5,
    use_tighter_stops: bool = True,
    atr_stop_multiplier: float = 1.5,
    min_stop_distance_percent: float = 0.005,
    max_stop_distance_percent: float = 0.02,
    max_trade_duration: int = 20,
    # Market regime parameters
    use_market_regime_filter: bool = True,
    min_regime_strength: float = 0.6,
    allow_volatile_regime: bool = False,
    # Time of day parameters
    use_time_filter: bool = True,
    trading_timezone: str = "UTC",
    favorable_hours: List[int] = None,
    # Entry confirmation parameters
    strengthen_entry_criteria: bool = True,
    minimum_confidence: float = 0.7,
    # Volatility-based take profit parameters
    use_volatility_based_tps: bool = True,
    atr_tp_multipliers: List[float] = None,  # Multipliers for ATR to set TPs
    min_tp_distance_percent: float = 0.01,  # Minimum TP distance as percent
    max_tp_distance_percent: float = 0.05,  # Maximum TP distance as percent
    # Multi-stage exit parameters
    use_partial_exits: bool = True,
    tp1_exit_percentage: float = 0.33,  # Percentage of position to exit at TP1
    tp2_exit_percentage: float = 0.50,  # Percentage of remaining position to exit at TP2
    tp3_exit_percentage: float = 1.0,  # Percentage of remaining position to exit at TP3
    iteration_id: Optional[str] = None,
    db: Optional[ClickHouseDB] = None,
) -> Tuple[float, list, Optional[str]]:
    """
    Backtest a strategy with risk management:
      - At each step, generate a signal with risk management parameters
      - Use position sizing based on risk percentage
      - Implement multiple take profit levels
      - Use dynamic stop loss

    Parameters:
      symbol: The trading pair symbol (e.g. "BTCUSDT")
      interval: The candle interval (e.g. "1h")
      candles: Total number of candles to fetch
      window: Lookback period (number of candles) to use for generating the signal
      initial_balance: Starting capital in USD (or your base currency)
      liq_lev_tolerance: Tolerance for liquidity level detection
      risk_percentage: Percentage of account to risk per trade
      weights: List of weights for signal generation

    Returns:
      final_balance: The simulated portfolio balance at the end
      trade_log: A list of trade events for review
    """
    # Generate UUIDs for database tracking
    sub_iteration_id = str(uuid.uuid4()) if db else None
    if db and not iteration_id:
        iteration_id = str(uuid.uuid4())

    # Determine a random start time for fetching candles
    end_time = datetime.now()
    start_time = end_time - timedelta(days=730)  # Assuming 2 years of data availability
    random_start_time = start_time + timedelta(
        seconds=random.randint(0, int((end_time - start_time).total_seconds()))
    )

    # Convert datetime to timestamp for fetch_candles
    timestamp = random_start_time.timestamp()

    # Fetch candles starting from the random start time
    df = fetch_candles(symbol, candles, interval, timestamp=timestamp)
    if df is None or df.empty:
        raise ValueError("No data returned for backtesting.")

    # Ensure the DataFrame is sorted by time
    df.sort_index(inplace=True)

    balance = initial_balance
    position = 0.0  # quantity of asset held
    entry_price = None
    trade_log = []  # logs of trades for later analysis
    entry_time = None
    entry_index = None
    entry_signal = None
    current_trade = None  # Store current trade details
    parent_trade_id = None  # Track parent trade for TP/SL entries

    # Use all indicators enabled by default in backtesting
    preferences = create_true_preferences()

    for i in range(window, len(df)):
        current_window = df.iloc[i - window : i]
        current_time = df.index[i]
        current_price = df["Close"].iloc[i]

        # Compute technical indicators and generate a signal
        indicators = analyze_data(current_window, preferences, liq_lev_tolerance)
        signal, prob, confidence, reason, trading_signal = (
            generate_price_prediction_signal_proba(
                current_window, indicators, weights, balance, risk_percentage
            )
        )

        # Handle existing position
        if position > 0 and current_trade:
            # Check take profit levels
            if (
                current_trade["take_profit_1"] is not None
                and current_price >= current_trade["take_profit_1"]
                and not current_trade["tp1_hit"]
            ):
                current_trade["tp1_hit"] = True

                # Calculate amount to close based on configured percentage
                if use_partial_exits:
                    # Close the configured percentage of initial position at TP1
                    close_amount = (
                        current_trade["initial_position"] * tp1_exit_percentage
                    )
                    close_amount = min(
                        close_amount, position
                    )  # Don't close more than we have

                    profit = close_amount * (current_price - entry_price)
                    balance += close_amount * current_price
                    position -= close_amount

                    trade_log.append(
                        {
                            "type": "take_profit_1",
                            "price": current_price,
                            "index": i,
                            "signal": f"Take Profit 1 - {tp1_exit_percentage*100:.0f}% position closed",
                            "timestamp": current_time,
                            "amount": close_amount,
                            "remaining": position,
                            "profit": profit,
                        }
                    )
                    print(
                        f"[Index {i}] TP1 hit at {current_price:.5f} - Closed {close_amount:.5f} units ({tp1_exit_percentage*100:.0f}%) - Remaining: {position:.5f}"
                    )
                else:
                    # Close entire position if partial exits are disabled
                    close_amount = position
                    profit = close_amount * (current_price - entry_price)
                    balance += close_amount * current_price
                    position = 0

                    trade_log.append(
                        {
                            "type": "take_profit_1",
                            "price": current_price,
                            "index": i,
                            "signal": "Take Profit 1 - 100% position closed",
                            "timestamp": current_time,
                            "amount": close_amount,
                            "profit": profit,
                        }
                    )
                    print(
                        f"[Index {i}] TP1 hit at {current_price:.5f} - Closed entire position ({close_amount:.5f} units)"
                    )

                    # Reset trade tracking since position is closed
                    entry_price = None
                    entry_time = None
                    entry_index = None
                    entry_signal = None
                    current_trade = None
                    parent_trade_id = None

                # Store TP1 trade in database
                if db:
                    trade_id = (
                        str(uuid.uuid4()) if use_partial_exits else parent_trade_id
                    )

                    trade_data = {
                        "trade_id": trade_id,
                        "iteration_id": str(iteration_id),
                        "sub_iteration_id": str(sub_iteration_id),
                        "symbol": str(symbol),
                        "interval": str(interval),
                        "trade_type": "take_profit_1",
                        "entry_timestamp": entry_time,
                        "exit_timestamp": current_time,
                        "entry_index": int(entry_index),
                        "exit_index": int(i),
                        "entry_price": float(entry_price),
                        "exit_price": float(current_price),
                        "profit_loss": float(profit),
                        "trade_duration": int(i - entry_index),
                        "entry_signal": str(entry_signal),
                        "exit_signal": "Take Profit 1",
                        "risk_reward_ratio": float(current_trade["risk_reward_ratio"]),
                        "position_size": float(close_amount),
                        "stop_loss": float(current_trade["stop_loss"]),
                        "take_profit_1": float(current_trade["take_profit_1"]),
                        "take_profit_2": float(current_trade["take_profit_2"]),
                        "take_profit_3": float(current_trade["take_profit_3"]),
                        "risk_percentage": float(risk_percentage),
                        "amount_traded": float(close_amount * current_price),
                        "parent_trade_id": (
                            parent_trade_id if use_partial_exits else None
                        ),
                    }

                    if use_partial_exits:
                        # For partial exits, insert a new trade record
                        db.insert_trade(trade_data)
                    else:
                        # For full exit, update the original trade record
                        db.update_trade(
                            parent_trade_id,
                            {
                                "exit_timestamp": current_time,
                                "exit_index": int(i),
                                "exit_price": float(current_price),
                                "profit_loss": float(profit),
                                "trade_duration": int(i - entry_index),
                                "exit_signal": "Take Profit 1",
                            },
                        )

            elif (
                current_trade["take_profit_2"] is not None
                and current_price >= current_trade["take_profit_2"]
                and current_trade["tp1_hit"]
                and not current_trade["tp2_hit"]
                and position > 0
            ):
                current_trade["tp2_hit"] = True

                # Calculate amount to close at TP2
                if use_partial_exits:
                    # Calculate percentage of remaining position to close
                    remaining_percentage = tp2_exit_percentage
                    close_amount = position * remaining_percentage
                    close_amount = min(
                        close_amount, position
                    )  # Don't close more than we have

                    profit = close_amount * (current_price - entry_price)
                    balance += close_amount * current_price
                    position -= close_amount

                    trade_log.append(
                        {
                            "type": "take_profit_2",
                            "price": current_price,
                            "index": i,
                            "signal": f"Take Profit 2 - {remaining_percentage*100:.0f}% of remaining position closed",
                            "timestamp": current_time,
                            "amount": close_amount,
                            "remaining": position,
                            "profit": profit,
                        }
                    )
                    print(
                        f"[Index {i}] TP2 hit at {current_price:.5f} - Closed {close_amount:.5f} units ({remaining_percentage*100:.0f}% of remaining) - Remaining: {position:.5f}"
                    )
                else:
                    # Close entire position if partial exits are disabled
                    close_amount = position
                    profit = close_amount * (current_price - entry_price)
                    balance += close_amount * current_price
                    position = 0

                    trade_log.append(
                        {
                            "type": "take_profit_2",
                            "price": current_price,
                            "index": i,
                            "signal": "Take Profit 2 - 100% position closed",
                            "timestamp": current_time,
                            "amount": close_amount,
                            "profit": profit,
                        }
                    )
                    print(
                        f"[Index {i}] TP2 hit at {current_price:.5f} - Closed entire position ({close_amount:.5f} units)"
                    )

                    # Reset trade tracking since position is closed
                    entry_price = None
                    entry_time = None
                    entry_index = None
                    entry_signal = None
                    current_trade = None
                    parent_trade_id = None

                # Store TP2 trade in database
                if db:
                    # Generate a new trade ID for this partial exit if using partial exits
                    trade_id = (
                        str(uuid.uuid4()) if use_partial_exits else parent_trade_id
                    )

                    trade_data = {
                        "trade_id": trade_id,
                        "iteration_id": str(iteration_id),
                        "sub_iteration_id": str(sub_iteration_id),
                        "symbol": str(symbol),
                        "interval": str(interval),
                        "trade_type": "take_profit_2",
                        "entry_timestamp": entry_time,
                        "exit_timestamp": current_time,
                        "entry_index": int(entry_index),
                        "exit_index": int(i),
                        "entry_price": float(entry_price),
                        "exit_price": float(current_price),
                        "profit_loss": float(profit),
                        "trade_duration": int(i - entry_index),
                        "entry_signal": str(entry_signal),
                        "exit_signal": "Take Profit 2",
                        "risk_reward_ratio": float(current_trade["risk_reward_ratio"]),
                        "position_size": float(close_amount),
                        "stop_loss": float(current_trade["stop_loss"]),
                        "take_profit_1": float(current_trade["take_profit_1"]),
                        "take_profit_2": float(current_trade["take_profit_2"]),
                        "take_profit_3": float(current_trade["take_profit_3"]),
                        "risk_percentage": float(risk_percentage),
                        "amount_traded": float(close_amount * current_price),
                        "parent_trade_id": (
                            parent_trade_id if use_partial_exits else None
                        ),
                    }

                    if use_partial_exits:
                        # For partial exits, insert a new trade record
                        db.insert_trade(trade_data)
                    else:
                        # For full exit, update the original trade record
                        db.update_trade(
                            parent_trade_id,
                            {
                                "exit_timestamp": current_time,
                                "exit_index": int(i),
                                "exit_price": float(current_price),
                                "profit_loss": float(profit),
                                "trade_duration": int(i - entry_index),
                                "exit_signal": "Take Profit 2",
                            },
                        )

            # Check TP3 - always closes the full remaining position
            elif (
                current_trade["take_profit_3"] is not None
                and current_price >= current_trade["take_profit_3"]
                and current_trade["tp1_hit"]
                and current_trade["tp2_hit"]
                and not current_trade["tp3_hit"]
                and position > 0
            ):
                current_trade["tp3_hit"] = True

                # Always close the full remaining position at TP3
                close_amount = position
                profit = close_amount * (current_price - entry_price)
                balance += close_amount * current_price
                position = 0

                trade_log.append(
                    {
                        "type": "take_profit_3",
                        "price": current_price,
                        "index": i,
                        "signal": "Take Profit 3 - 100% remaining position closed",
                        "timestamp": current_time,
                        "amount": close_amount,
                        "profit": profit,
                    }
                )
                print(
                    f"[Index {i}] TP3 hit at {current_price:.5f} - Closed remaining position ({close_amount:.5f} units)"
                )

                # Reset trade tracking since position is fully closed
                entry_price = None
                entry_time = None
                entry_index = None
                entry_signal = None
                parent_trade_id = None

                # Store TP3 trade in database
                if db and parent_trade_id:
                    # Always close the parent trade when TP3 is hit
                    trade_data = {
                        "exit_timestamp": current_time,
                        "exit_index": int(i),
                        "exit_price": float(current_price),
                        "profit_loss": float(profit),
                        "trade_duration": int(i - entry_index),
                        "exit_signal": "Take Profit 3",
                    }
                    db.update_trade(parent_trade_id, trade_data)

                    # Also insert a TP3 trade record if using partial exits
                    if use_partial_exits:
                        tp3_trade_id = str(uuid.uuid4())
                        tp3_trade_data = {
                            "trade_id": tp3_trade_id,
                            "iteration_id": str(iteration_id),
                            "sub_iteration_id": str(sub_iteration_id),
                            "symbol": str(symbol),
                            "interval": str(interval),
                            "trade_type": "take_profit_3",
                            "entry_timestamp": entry_time,
                            "exit_timestamp": current_time,
                            "entry_index": int(entry_index),
                            "exit_index": int(i),
                            "entry_price": float(entry_price),
                            "exit_price": float(current_price),
                            "profit_loss": float(profit),
                            "trade_duration": int(i - entry_index),
                            "entry_signal": str(entry_signal),
                            "exit_signal": "Take Profit 3",
                            "risk_reward_ratio": float(
                                current_trade["risk_reward_ratio"]
                            ),
                            "position_size": float(close_amount),
                            "stop_loss": float(current_trade["stop_loss"]),
                            "take_profit_1": float(current_trade["take_profit_1"]),
                            "take_profit_2": float(current_trade["take_profit_2"]),
                            "take_profit_3": float(current_trade["take_profit_3"]),
                            "risk_percentage": float(risk_percentage),
                            "amount_traded": float(close_amount * current_price),
                            "parent_trade_id": parent_trade_id,
                        }
                        db.insert_trade(tp3_trade_data)

                # Reset all trade tracking since position is fully closed at TP3
                current_trade = None
                position = 0
                entry_price = None
                entry_time = None
                entry_index = None
                entry_signal = None
                parent_trade_id = None
                
                # Skip the rest
                return

            # Update trailing stop if enabled
            if use_trailing_stop and current_trade and current_price > entry_price:
                updated_stop = calculate_trailing_stop(
                    entry_price,
                    current_price,
                    current_trade["stop_loss"],
                    trail_percent,
                    trail_activation_threshold,
                )
                current_trade["stop_loss"] = updated_stop

            # Check time-based exit
            time_exit_triggered = should_exit_based_on_time(
                entry_index, i, max_trade_duration
            )

            # Check stop loss
            if current_price <= current_trade["stop_loss"] or time_exit_triggered:
                # Close entire position at stop loss or time-based exit
                loss = position * (current_price - entry_price)
                balance += position * current_price

                exit_reason = (
                    "Stop Loss"
                    if current_price <= current_trade["stop_loss"]
                    else "Time-Based Exit"
                )

                trade_log.append(
                    {
                        "type": exit_reason.lower().replace(" ", "_"),
                        "price": current_price,
                        "index": i,
                        "signal": f"{exit_reason} - Position closed",
                        "timestamp": current_time,
                        "amount": position,
                        "profit": loss,
                    }
                )
                print(
                    f"[Index {i}] {exit_reason} at {current_price:.5f} - Profit/Loss: {loss:.2f}"
                )

                # Store stop loss trade in database
                if db:
                    trade_data = {
                        "iteration_id": iteration_id,
                        "sub_iteration_id": sub_iteration_id,
                        "symbol": symbol,
                        "interval": interval,
                        "trade_type": exit_reason.lower().replace(" ", "_"),
                        "entry_timestamp": entry_time,
                        "exit_timestamp": current_time,
                        "entry_index": entry_index,
                        "exit_index": i,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "profit_loss": loss,
                        "trade_duration": i - entry_index,
                        "entry_signal": entry_signal,
                        "exit_signal": exit_reason,
                        "risk_reward_ratio": current_trade["risk_reward_ratio"],
                        "position_size": position,
                        "stop_loss": current_trade["stop_loss"],
                        "take_profit_1": current_trade["take_profit_1"],
                        "take_profit_2": current_trade["take_profit_2"],
                        "take_profit_3": current_trade["take_profit_3"],
                        "risk_percentage": risk_percentage,
                        "amount_traded": position * current_price,
                        "parent_trade_id": parent_trade_id,
                    }
                    db.insert_trade(trade_data)
                
                # Reset trade tracking since position is fully closed at stop loss
                current_trade = None
                position = 0
                entry_price = None
                entry_time = None
                entry_index = None
                entry_signal = None
                parent_trade_id = None
                
                # Skip the rest
                return

        # Handle new signal
        if signal in ["Bullish", "Bearish"] and position == 0 and trading_signal:
            # Market regime filter disabled as it's too restrictive
            # We'll keep the code but bypass the filter logic
            if False and use_market_regime_filter:  # Added False to bypass
                # Detect current market regime
                market_regime = detect_market_regime(
                    df.iloc[: i + 1], lookback_period=20
                )
                should_trade, regime_reason = should_trade_in_regime(
                    market_regime,
                    signal,
                    min_regime_strength=min_regime_strength,
                    allow_volatile=allow_volatile_regime,
                )

                if not should_trade:
                    print(
                        f"[Index {i}] Skipping trade due to market regime: {regime_reason}"
                    )
                    continue

            # Time filter disabled as it's too restrictive
            # We'll keep the code but bypass the filter logic
            if False and use_time_filter:  # Added False to bypass
                # Set default favorable hours if not provided
                if favorable_hours is None:
                    # Default to common active market hours (UTC)
                    favorable_hours = [
                        1,
                        2,
                        3,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        20,
                        21,
                        22,
                    ]

                favorable_time, time_reason = is_favorable_trading_time(
                    current_time,
                    timezone=trading_timezone,
                    favorable_hours=favorable_hours,
                )

                if not favorable_time:
                    print(
                        f"[Index {i}] Skipping trade due to time filter: {time_reason}"
                    )
                    continue

            # Entry criteria filter disabled as it's too restrictive
            # We'll keep the code but bypass the filter logic
            if False and strengthen_entry_criteria:
                criteria_met, criteria_reason = confirm_entry_criteria(
                    df.iloc[: i + 1],
                    i,
                    signal,
                    min_confidence=min_entry_confidence,
                    min_confirmation_candles=min_price_action_confirmation,
                    require_volume_confirmation=require_volume_confirmation,
                )

                if not criteria_met:
                    print(
                        f"[Index {i}] Skipping trade due to entry criteria: {criteria_reason}"
                    )
                    continue

            # Proceed with the trade if it passes all filters
            if signal == "Bullish":
                # Validate price is reasonable
                if current_price < 0.00000001:  # Skip if price is too small
                    continue

                # Calculate position size based on risk management
                position_sizing = calculate_position_size(
                    balance, risk_percentage, current_price, trading_signal.stop_loss
                )

                # Apply tighter stops if enabled
                if use_tighter_stops:
                    tighter_stop = calculate_tighter_stop_loss(
                        df,
                        i,
                        current_price,
                        signal,
                        atr_period=14,
                        baseline_multiplier=atr_stop_multiplier,
                        min_distance_percent=min_stop_distance_percent,
                        max_distance_percent=max_stop_distance_percent,
                    )
                    # Only update if tighter stop is actually better (closer but still safe)
                    if signal == "Bullish" and tighter_stop > trading_signal.stop_loss:
                        trading_signal.stop_loss = tighter_stop
                    elif (
                        signal == "Bearish" and tighter_stop < trading_signal.stop_loss
                    ):
                        trading_signal.stop_loss = tighter_stop

                # Apply volatility-based take profits if enabled
                if use_volatility_based_tps:
                    # Default ATR multipliers if not provided
                    if atr_tp_multipliers is None:
                        atr_tp_multipliers = [
                            2.0,
                            3.5,
                            5.0,
                        ]  # Default multipliers for 3 TPs

                    # Calculate volatility-based take profit levels
                    vol_based_tps = calculate_volatility_based_take_profits(
                        df,
                        i,
                        current_price,
                        signal,
                        atr_period=14,
                        tp_multipliers=atr_tp_multipliers,
                        min_distance_percent=min_tp_distance_percent,
                        max_distance_percent=max_tp_distance_percent,
                    )

                    # Update take profit levels based on volatility
                    # For bullish signals, only update if the volatility-based TP is lower (closer)
                    if signal == "Bullish":
                        if vol_based_tps[0] < trading_signal.take_profit_1:
                            trading_signal.take_profit_1 = vol_based_tps[0]
                        if vol_based_tps[1] < trading_signal.take_profit_2:
                            trading_signal.take_profit_2 = vol_based_tps[1]
                        if vol_based_tps[2] < trading_signal.take_profit_3:
                            trading_signal.take_profit_3 = vol_based_tps[2]

                # Recalculate position size with potentially updated stop loss
                position_sizing = calculate_position_size(
                    balance, risk_percentage, current_price, trading_signal.stop_loss
                )

                # Enter position with calculated size
                position = position_sizing["position_size"]
                amount_to_invest = position * current_price

                # Ensure we don't invest more than available balance
                if amount_to_invest > balance:
                    position = balance / current_price
                    amount_to_invest = balance

                # Skip if position size is unreasonably large
                if position > 1e12:  # 1 trillion units max
                    continue

                balance -= amount_to_invest
                entry_price = current_price
                entry_time = current_time
                entry_index = i
                entry_signal = signal
                parent_trade_id = str(uuid.uuid4())  # Generate parent trade ID

                # Store trade details
                current_trade = {
                    "stop_loss": float(trading_signal.stop_loss),
                    "take_profit_1": float(trading_signal.take_profit_1),
                    "take_profit_2": float(trading_signal.take_profit_2),
                    "take_profit_3": float(trading_signal.take_profit_3),
                    "risk_reward_ratio": float(trading_signal.risk_reward_ratio),
                    "position_size": float(position),
                    "initial_position": float(position),  # Store initial position size
                    "tp1_hit": False,
                    "tp2_hit": False,
                    "tp3_hit": False,
                }

                trade_log.append(
                    {
                        "type": "entry",
                        "price": float(entry_price),
                        "index": int(i),
                        "signal": f"{signal} - {reason.splitlines()[0]}",
                        "timestamp": current_time,
                        "amount": float(position),
                        "stop_loss": float(trading_signal.stop_loss),
                        "take_profit_1": float(trading_signal.take_profit_1),
                        "take_profit_2": float(trading_signal.take_profit_2),
                        "take_profit_3": float(trading_signal.take_profit_3),
                    }
                )

                # Store entry trade in database
                if db:
                    trade_data = {
                        "trade_id": parent_trade_id,
                        "iteration_id": str(iteration_id),
                        "sub_iteration_id": str(sub_iteration_id),
                        "symbol": str(symbol),
                        "interval": str(interval),
                        "trade_type": "entry",
                        "entry_timestamp": current_time,
                        "exit_timestamp": current_time,  # Will be updated on exit
                        "entry_index": int(i),
                        "exit_index": int(i),  # Will be updated on exit
                        "entry_price": float(entry_price),
                        "exit_price": float(entry_price),  # Will be updated on exit
                        "profit_loss": 0.0,  # Will be calculated on exit
                        "trade_duration": 0,  # Will be updated on exit
                        "entry_signal": str(signal),
                        "exit_signal": "",  # Will be updated on exit
                        "risk_reward_ratio": float(trading_signal.risk_reward_ratio),
                        "position_size": float(position),
                        "stop_loss": float(trading_signal.stop_loss),
                        "take_profit_1": float(trading_signal.take_profit_1),
                        "take_profit_2": float(trading_signal.take_profit_2),
                        "take_profit_3": float(trading_signal.take_profit_3),
                        "risk_percentage": float(risk_percentage),
                        "amount_traded": float(amount_to_invest),
                        "parent_trade_id": None,  # This is the parent trade
                    }
                    db.insert_trade(trade_data)

                print(
                    f"[Index {i}] {symbol}: ENTRY at {entry_price:.5f} | Size: {position:.5f} | Risk/Reward: {trading_signal.risk_reward_ratio:.2f} | Reason: {reason.splitlines()[0]}"
                )

            # Handle bearish signals - mirror of bullish signal handling
            elif signal == "Bearish":
                # Validate price is reasonable
                if current_price < 0.00000001:  # Skip if price is too small
                    continue

                # Calculate position size based on risk management
                position_sizing = calculate_position_size(
                    balance, risk_percentage, current_price, trading_signal.stop_loss
                )

                # Apply tighter stops if enabled
                if use_tighter_stops:
                    tighter_stop = calculate_tighter_stop_loss(
                        df,
                        i,
                        current_price,
                        signal,
                        atr_period=14,
                        baseline_multiplier=atr_stop_multiplier,
                        min_distance_percent=min_stop_distance_percent,
                        max_distance_percent=max_stop_distance_percent,
                    )
                    # Only update if tighter stop is actually better (closer but still safe)
                    if signal == "Bearish" and tighter_stop < trading_signal.stop_loss:
                        trading_signal.stop_loss = tighter_stop

                # Apply volatility-based take profits if enabled
                if use_volatility_based_tps:
                    # Default ATR multipliers if not provided
                    if atr_tp_multipliers is None:
                        atr_tp_multipliers = [
                            2.0,
                            3.5,
                            5.0,
                        ]  # Default multipliers for 3 TPs

                    # Calculate volatility-based take profit levels
                    vol_based_tps = calculate_volatility_based_take_profits(
                        df,
                        i,
                        current_price,
                        signal,
                        atr_period=14,
                        tp_multipliers=atr_tp_multipliers,
                        min_distance_percent=min_tp_distance_percent,
                        max_distance_percent=max_tp_distance_percent,
                    )

                    # Update take profit levels based on volatility
                    # For bearish signals, only update if the volatility-based TP is higher (closer)
                    if signal == "Bearish":
                        if vol_based_tps[0] > trading_signal.take_profit_1:
                            trading_signal.take_profit_1 = vol_based_tps[0]
                        if vol_based_tps[1] > trading_signal.take_profit_2:
                            trading_signal.take_profit_2 = vol_based_tps[1]
                        if vol_based_tps[2] > trading_signal.take_profit_3:
                            trading_signal.take_profit_3 = vol_based_tps[2]

                # Recalculate position size with potentially updated stop loss
                position_sizing = calculate_position_size(
                    balance, risk_percentage, current_price, trading_signal.stop_loss
                )

                # Enter position with calculated size
                position = position_sizing["position_size"]
                amount_to_invest = position * current_price

                # Ensure we don't invest more than available balance
                if amount_to_invest > balance:
                    position = balance / current_price
                    amount_to_invest = balance

                # Skip if position size is unreasonably large
                if position > 1e12:  # 1 trillion units max
                    continue

                balance -= amount_to_invest
                entry_price = current_price
                entry_time = current_time
                entry_index = i
                entry_signal = signal
                parent_trade_id = str(uuid.uuid4())  # Generate parent trade ID

                # Store trade details
                current_trade = {
                    "stop_loss": float(trading_signal.stop_loss),
                    "take_profit_1": float(trading_signal.take_profit_1),
                    "take_profit_2": float(trading_signal.take_profit_2),
                    "take_profit_3": float(trading_signal.take_profit_3),
                    "risk_reward_ratio": float(trading_signal.risk_reward_ratio),
                    "position_size": float(position),
                    "initial_position": float(position),  # Store initial position size
                    "tp1_hit": False,
                    "tp2_hit": False,
                    "tp3_hit": False,
                }

                trade_log.append(
                    {
                        "type": "entry",
                        "price": float(entry_price),
                        "index": int(i),
                        "signal": f"{signal} - {reason.splitlines()[0]}",
                        "timestamp": current_time,
                        "amount": float(position),
                        "stop_loss": float(trading_signal.stop_loss),
                        "take_profit_1": float(trading_signal.take_profit_1),
                        "take_profit_2": float(trading_signal.take_profit_2),
                        "take_profit_3": float(trading_signal.take_profit_3),
                    }
                )

                # Store entry trade in database
                if db:
                    trade_data = {
                        "trade_id": parent_trade_id,
                        "iteration_id": str(iteration_id),
                        "sub_iteration_id": str(sub_iteration_id),
                        "symbol": str(symbol),
                        "interval": str(interval),
                        "trade_type": "entry",
                        "entry_timestamp": current_time,
                        "exit_timestamp": current_time,  # Will be updated on exit
                        "entry_index": int(i),
                        "exit_index": int(i),  # Will be updated on exit
                        "entry_price": float(entry_price),
                        "exit_price": float(entry_price),  # Will be updated on exit
                        "profit_loss": 0.0,  # Will be calculated on exit
                        "trade_duration": 0,  # Will be updated on exit
                        "entry_signal": str(signal),
                        "exit_signal": "",  # Will be updated on exit
                        "risk_reward_ratio": float(trading_signal.risk_reward_ratio),
                        "position_size": float(position),
                        "stop_loss": float(trading_signal.stop_loss),
                        "take_profit_1": float(trading_signal.take_profit_1),
                        "take_profit_2": float(trading_signal.take_profit_2),
                        "take_profit_3": float(trading_signal.take_profit_3),
                        "risk_percentage": float(risk_percentage),
                        "amount_traded": float(amount_to_invest),
                        "parent_trade_id": None,  # This is the parent trade
                    }
                    db.insert_trade(trade_data)

                print(
                    f"[Index {i}] {symbol}: ENTRY at {entry_price:.5f} | Size: {position:.5f} | Risk/Reward: {trading_signal.risk_reward_ratio:.2f} | Reason: {reason.splitlines()[0]}"
                )

            trade_log.append(
                {
                    "type": "entry",
                    "price": float(entry_price),
                    "index": int(i),
                    "signal": f"{signal} - {reason.splitlines()[0]}",
                    "timestamp": current_time,
                    "amount": float(position),
                    "stop_loss": float(trading_signal.stop_loss),
                    "take_profit_1": float(trading_signal.take_profit_1),
                    "take_profit_2": float(trading_signal.take_profit_2),
                    "take_profit_3": float(trading_signal.take_profit_3),
                }
            )

            # Store entry trade in database
            if db:
                trade_data = {
                    "trade_id": parent_trade_id,
                    "iteration_id": str(iteration_id),
                    "sub_iteration_id": str(sub_iteration_id),
                    "symbol": str(symbol),
                    "interval": str(interval),
                    "trade_type": "entry",
                    "entry_timestamp": current_time,
                    "exit_timestamp": current_time,  # Will be updated on exit
                    "entry_index": int(i),
                    "exit_index": int(i),  # Will be updated on exit
                    "entry_price": float(entry_price),
                    "exit_price": float(entry_price),  # Will be updated on exit
                    "profit_loss": 0.0,  # Will be calculated on exit
                    "trade_duration": 0,  # Will be updated on exit
                    "entry_signal": str(signal),
                    "exit_signal": "",  # Will be updated on exit
                    "risk_reward_ratio": float(trading_signal.risk_reward_ratio),
                    "position_size": float(position),
                    "stop_loss": float(trading_signal.stop_loss),
                    "take_profit_1": float(trading_signal.take_profit_1),
                    "take_profit_2": float(trading_signal.take_profit_2),
                    "take_profit_3": float(trading_signal.take_profit_3),
                    "risk_percentage": float(risk_percentage),
                    "amount_traded": float(amount_to_invest),
                    "parent_trade_id": None,  # This is the parent trade
                }
                db.insert_trade(trade_data)

            print(
                f"[Index {i}] {symbol}: ENTRY at {entry_price:.5f} | Size: {position:.5f} | Risk/Reward: {trading_signal.risk_reward_ratio:.2f} | Reason: {reason.splitlines()[0]}"
            )

    # If still holding a position at the end, liquidate at the final price
    if position > 0:
        final_price = df["Close"].iloc[-1]
        final_time = df.index[-1]
        profit = position * (final_price - entry_price)
        balance += position * final_price

        trade_log.append(
            {
                "type": "exit_end_of_period",
                "price": final_price,
                "index": len(df) - 1,
                "signal": "End of Period - Position closed",
                "timestamp": final_time,
                "amount": position,
                "profit": profit,
            }
        )
        print(f"[Final] EXIT at {final_price:.5f} - Profit/Loss: {profit:.2f}")

        # Store final trade in database if available
        if db and entry_time and entry_index:
            trade_data = {
                "iteration_id": iteration_id,
                "sub_iteration_id": sub_iteration_id,
                "symbol": symbol,
                "interval": interval,
                "trade_type": "exit_end_of_period",
                "entry_timestamp": entry_time,
                "exit_timestamp": final_time,
                "entry_index": entry_index,
                "exit_index": len(df) - 1,
                "entry_price": entry_price,
                "exit_price": final_price,
                "profit_loss": profit,
                "trade_duration": len(df) - 1 - entry_index,
                "entry_signal": entry_signal,
                "exit_signal": "End of Period",
                "risk_reward_ratio": (
                    current_trade["risk_reward_ratio"] if current_trade else 0
                ),
                "position_size": position,
                "stop_loss": current_trade["stop_loss"] if current_trade else 0,
                "take_profit_1": current_trade["take_profit_1"] if current_trade else 0,
                "take_profit_2": current_trade["take_profit_2"] if current_trade else 0,
                "take_profit_3": current_trade["take_profit_3"] if current_trade else 0,
                "risk_percentage": risk_percentage,
                "amount_traded": position * final_price,
                "parent_trade_id": parent_trade_id,
            }
            db.insert_trade(trade_data)

    return balance, trade_log, sub_iteration_id


if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = "1h"
    final_balance, trades, _ = backtest_strategy(
        symbol, interval, candles=600, window=300, risk_percentage=1.0
    )
    print(f"Final Balance: {final_balance:.2f}")
    print("Trade Log:")
    for trade in trades:
        print(trade)
