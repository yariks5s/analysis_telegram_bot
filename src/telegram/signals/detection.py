"""
Signal detection module for CryptoBot.

This module contains functions for detecting trading signals, managing signal jobs,
and notifying users about market conditions.
"""

from datetime import timedelta
from typing import List, Dict, Optional, Any
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from dataclasses import dataclass
from enum import Enum

from src.core.utils import auto_signal_jobs, create_true_preferences, logger
from src.database.operations import (
    get_user_preferences,
    upsert_user_signal_request,
    delete_user_signal_request,
    get_chat_id_for_user,
    get_signal_requests,
    user_signal_request_exists,
)

# Imports with updated module paths
from src.visualization.plot_builder import plot_price_chart
from src.analysis.utils.helpers import fetch_candles, analyze_data, fetch_data_and_get_indicators


@dataclass
class TradingSignal:
    """Enhanced signal class with risk management"""

    symbol: str
    signal_type: str  # "Bullish", "Bearish", "Neutral"
    probability: float
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_reward_ratio: float
    position_size: float
    max_risk_amount: float
    reasons: List[str]
    market_conditions: Dict[str, any]
    timestamp: pd.Timestamp


class MarketRegime(Enum):
    """Market regime classification"""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range for volatility
    
    Args:
        df: DataFrame with OHLC data
        period: Period for ATR calculation
    
    Returns:
        ATR values as a pandas Series
    """
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    # Fill NaN values with a reasonable default (1% of price)
    if atr.isna().all():
        atr = df["Close"] * 0.01
    else:
        atr = atr.bfill().ffill()
        if atr.isna().any():
            atr = atr.fillna(df["Close"] * 0.01)

    return atr


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI for momentum confirmation
    
    Args:
        series: Price series (typically Close prices)
        period: RSI calculation period
        
    Returns:
        RSI values as a pandas Series
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Prevent division by zero
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Fill NaN values with neutral RSI (50)
    rsi = rsi.fillna(50)

    return rsi


def detect_market_regime(df: pd.DataFrame) -> MarketRegime:
    """
    Detect current market regime for better signal filtering
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        Market regime classification
    """
    # Calculate indicators for regime detection
    atr = calculate_atr(df)
    rsi = calculate_rsi(df["Close"])
    
    # Get recent values
    recent_atr = atr.iloc[-5:].mean()
    recent_rsi = rsi.iloc[-5:].mean()
    avg_atr = atr.mean()
    price_range_percent = (df["High"].iloc[-20:].max() - df["Low"].iloc[-20:].min()) / df["Close"].iloc[-1]
    
    # Determine market regime
    if recent_atr > avg_atr * 1.5:
        return MarketRegime.VOLATILE
    elif recent_atr < avg_atr * 0.5:
        return MarketRegime.QUIET
    elif recent_rsi > 60 and df["Close"].pct_change(20).iloc[-1] > 0.05:
        return MarketRegime.TRENDING_UP
    elif recent_rsi < 40 and df["Close"].pct_change(20).iloc[-1] < -0.05:
        return MarketRegime.TRENDING_DOWN
    elif price_range_percent < 0.05:
        return MarketRegime.RANGING
    else:
        # Default to ranging if no clear regime is detected
        return MarketRegime.RANGING


async def auto_signal_job(context):
    """
    Scheduled job to check for signals and notify the user.
    
    This is called periodically based on the frequency set by the user.
    
    Args:
        context: Telegram context with job data
    """
    job_data = context.job.data
    user_id = job_data["user_id"]
    chat_id = job_data["chat_id"]
    currency_pair = job_data["currency_pair"]
    is_with_chart = job_data.get("is_with_chart", False)
    
    logger.info(f"Running auto signal job for {user_id}, {currency_pair}")
    
    try:
        # Get user's indicator preferences, or use default if none
        preferences = get_user_preferences(user_id)
        if not any(preferences.values()):
            preferences = create_true_preferences()
        
        # Fetch data for analysis
        df, indicators = fetch_data_and_get_indicators(currency_pair, 100, "1h", preferences)
        
        if df is None or df.empty:
            await context.bot.send_message(
                chat_id=chat_id, 
                text=f"❌ Error fetching data for {currency_pair}"
            )
            return
            
        # Generate analysis result
        analysis_result = generate_signal_analysis(df, indicators, currency_pair)
        
        # Generate chart if requested
        if is_with_chart:
            chart_path = plot_price_chart(
                df, 
                indicators, 
                preferences.get("show_legend", True),
                preferences.get("show_volume", True),
                preferences.get("dark_mode", False)
            )
            
            # Send chart with analysis
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=open(chart_path, "rb"),
                caption=analysis_result
            )
        else:
            # Send text analysis only
            await context.bot.send_message(
                chat_id=chat_id,
                text=analysis_result
            )
            
    except Exception as e:
        logger.error(f"Error in auto_signal_job: {e}")
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Error analyzing {currency_pair}: {str(e)}"
        )


def generate_signal_analysis(df: pd.DataFrame, indicators, currency_pair: str) -> str:
    """
    Generate a comprehensive analysis of the current market conditions.
    
    Args:
        df: DataFrame with OHLC data
        indicators: Indicator objects containing detected patterns
        currency_pair: The trading pair symbol
        
    Returns:
        Formatted analysis text
    """
    # Detect market regime
    regime = detect_market_regime(df)
    
    # Calculate key technical indicators
    rsi = calculate_rsi(df["Close"]).iloc[-1]
    atr = calculate_atr(df).iloc[-1]
    
    # Current price
    current_price = df["Close"].iloc[-1]
    
    # Format the analysis text
    analysis = f"📊 *{currency_pair} Analysis*\n\n"
    analysis += f"*Price*: {current_price:.2f} USDT\n"
    analysis += f"*Market Regime*: {regime.value}\n"
    analysis += f"*RSI*: {rsi:.1f}\n\n"
    
    # Add indicator information if available
    if hasattr(indicators, 'order_blocks') and indicators.order_blocks.list:
        analysis += f"*Order Blocks*: {len(indicators.order_blocks.list)} detected\n"
        
    if hasattr(indicators, 'fvgs') and indicators.fvgs.list:
        analysis += f"*Fair Value Gaps*: {len(indicators.fvgs.list)} detected\n"
        
    if hasattr(indicators, 'liquidity_levels') and indicators.liquidity_levels.list:
        analysis += f"*Liquidity Levels*: {len(indicators.liquidity_levels.list)} detected\n"
    
    # Add a conclusion based on market regime
    analysis += "\n*Analysis*:\n"
    if regime == MarketRegime.TRENDING_UP:
        analysis += "Market is in an uptrend. Watch for bullish continuation patterns."
    elif regime == MarketRegime.TRENDING_DOWN:
        analysis += "Market is in a downtrend. Watch for bearish continuation patterns."
    elif regime == MarketRegime.RANGING:
        analysis += "Market is ranging. Watch for breakouts or rejections at range boundaries."
    elif regime == MarketRegime.VOLATILE:
        analysis += "Market is volatile. Consider reducing position sizes and using wider stops."
    elif regime == MarketRegime.QUIET:
        analysis += "Market is quiet with low volatility. Potential buildup for a larger move."
        
    return analysis


async def createSignalJob(update, context):
    """
    Create a new signal monitoring job based on user input.
    
    Args:
        update: Telegram update object
        context: Telegram context object
    """
    text = update.message.text.strip().upper()
    user_id = update.effective_user.id
    chat_id = update.effective_message.chat_id
    
    # Parse user input
    parts = text.split()
    currency_pair = parts[0]  # e.g., "BTCUSDT"
    frequency = int(parts[1])  # e.g., 60 (minutes)
    is_with_chart = "WITH_CHART" in text.upper()
    
    # Save request to database
    signal_request = {
        "currency_pair": currency_pair,
        "frequency_minutes": frequency,
        "is_with_chart": is_with_chart
    }
    upsert_user_signal_request(user_id, signal_request)
    
    # Create job
    job_data = {
        "user_id": user_id,
        "chat_id": chat_id,
        "currency_pair": currency_pair,
        "is_with_chart": is_with_chart
    }
    
    # Add to job queue
    job = context.job_queue.run_repeating(
        callback=auto_signal_job,
        interval=timedelta(minutes=frequency),
        first=10,  # Start after 10 seconds
        name=f"signal_job_{user_id}_{currency_pair}",
        data=job_data,
    )
    
    # Store job reference
    job_key = (user_id, currency_pair)
    auto_signal_jobs[job_key] = job
    
    await update.message.reply_text(
        f"✅ Signal alert set for {currency_pair} every {frequency} minutes."
        f"{' With chart.' if is_with_chart else ''}"
    )


async def initialize_jobs(application):
    """
    Called once at bot start-up to restore all jobs from the database.
    
    Args:
        application: Telegram bot application
    """
    signal_requests = get_signal_requests()

    for req in signal_requests:
        user_id = req["user_id"]
        currency_pair = req["currency_pair"]
        frequency_minutes = req["frequency_minutes"]
        is_with_chart = req.get("is_with_chart", False)
        account_balance = req.get("account_balance", 10000)
        risk_percentage = req.get("risk_percentage", 1.0)
        chat_id = get_chat_id_for_user(user_id)

        if not chat_id:
            logger.info(f"No chat_id found for user {user_id}. Skipping job creation.")
            continue

        job_key = (user_id, currency_pair)
        if job_key in auto_signal_jobs:
            logger.info(
                f"Job for user_id {user_id}, pair {currency_pair} already exists."
            )
            continue

        # Create a job
        job_data = {
            "user_id": user_id,
            "chat_id": chat_id,
            "currency_pair": currency_pair,
            "is_with_chart": is_with_chart,
            "account_balance": account_balance,
            "risk_percentage": risk_percentage,
        }
        job_ref = application.job_queue.run_repeating(
            callback=auto_signal_job,
            interval=timedelta(minutes=frequency_minutes),
            first=0,
            name=f"signal_job_{user_id}_{currency_pair}",
            data=job_data,
        )

        auto_signal_jobs[job_key] = job_ref

    logger.info("All user signal jobs have been initialized.")
