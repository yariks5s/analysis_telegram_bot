import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import os
from datetime import datetime, timedelta


class CorrelationAnalysis:
    """
    Analyzes the correlation between trading strategy performance
    and various market factors to identify dependencies and risks.
    """

    def __init__(self):
        """Initialize correlation analysis"""
        self.price_data = {}
        self.trade_data = None
        self.strategy_returns = None

    def load_price_data(
        self, symbol: str, df: pd.DataFrame, rename_col: Optional[str] = None
    ) -> None:
        """
        Load price data for analysis.

        Args:
            symbol: Symbol/name for the price data (e.g. "BTC", "ETH", "SPY")
            df: DataFrame with price data (must have datetime index and Close column)
            rename_col: Optional name to use for the price column
        """
        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        # Ensure we have a Close column
        if "Close" not in df.columns:
            raise ValueError("DataFrame must have 'Close' column")

        # Create copy with just Close prices
        price_series = df["Close"].copy()

        # Rename if requested
        col_name = rename_col if rename_col else symbol

        # Store in dictionary
        self.price_data[col_name] = price_series

    def load_trade_data(
        self, trades: List[Dict[str, Any]], initial_balance: float
    ) -> None:
        """
        Load trade data for strategy returns calculation.

        Args:
            trades: List of trade dictionaries with timestamps and profits
            initial_balance: Initial account balance
        """
        if not trades:
            raise ValueError("No trades provided")

        # Convert to DataFrame for easier manipulation
        trade_df = pd.DataFrame(trades)

        # Use exit_timestamp if available, otherwise timestamp
        timestamp_col = (
            "exit_timestamp" if "exit_timestamp" in trade_df.columns else "timestamp"
        )

        if timestamp_col not in trade_df.columns:
            raise ValueError("Trades must have timestamp or exit_timestamp")

        # Profit column might be named differently
        profit_col = "profit" if "profit" in trade_df.columns else "profit_loss"

        if profit_col not in trade_df.columns:
            raise ValueError("Trades must have profit or profit_loss")

        # Ensure timestamp is datetime
        trade_df[timestamp_col] = pd.to_datetime(trade_df[timestamp_col])

        # Sort by timestamp
        trade_df = trade_df.sort_values(timestamp_col)

        # Set timestamp as index
        trade_df.set_index(timestamp_col, inplace=True)

        # Store trade data
        self.trade_data = trade_df

        # Calculate strategy returns
        self._calculate_strategy_returns(initial_balance)

    def _calculate_strategy_returns(self, initial_balance: float) -> None:
        """
        Calculate strategy returns from trade data.

        Args:
            initial_balance: Initial account balance
        """
        if self.trade_data is None:
            return

        # Profit column might be named differently
        profit_col = "profit" if "profit" in self.trade_data.columns else "profit_loss"

        # Calculate equity curve
        equity = initial_balance
        equity_series = [equity]
        timestamps = [
            self.trade_data.index[0] - timedelta(days=1)
        ]  # Start one day before first trade

        for idx, row in self.trade_data.iterrows():
            equity += row[profit_col]
            equity_series.append(equity)
            timestamps.append(idx)

        # Create equity curve series
        equity_curve = pd.Series(equity_series, index=timestamps)

        # Resample to daily and forward fill
        daily_equity = equity_curve.resample("D").last().ffill()

        # Calculate daily returns
        strategy_returns = daily_equity.pct_change().dropna()

        # Store strategy returns
        self.strategy_returns = strategy_returns

    def calculate_market_correlations(self) -> pd.DataFrame:
        """
        Calculate correlations between strategy returns and market factors.

        Returns:
            DataFrame with correlation statistics
        """
        if self.strategy_returns is None:
            raise ValueError("Strategy returns not available. Load trade data first.")

        if not self.price_data:
            raise ValueError("No price data available. Load price data first.")

        # Create DataFrame for all price data
        all_prices = pd.DataFrame()

        # Add each price series
        for symbol, price_series in self.price_data.items():
            all_prices[symbol] = price_series

        # Resample to daily and forward fill if needed
        if not isinstance(all_prices.index, pd.DatetimeIndex):
            raise ValueError("Price data must have DatetimeIndex")

        daily_prices = all_prices.resample("D").last().ffill()

        # Calculate daily returns
        market_returns = daily_prices.pct_change().dropna()

        # Align dates with strategy returns
        common_dates = market_returns.index.intersection(self.strategy_returns.index)
        aligned_market = market_returns.loc[common_dates]
        aligned_strategy = self.strategy_returns.loc[common_dates]

        # Calculate correlations
        correlations = {}

        for symbol in aligned_market.columns:
            # Pearson correlation
            pearson_corr = aligned_strategy.corr(aligned_market[symbol])

            # Calculate rolling correlations (30-day window)
            combined = pd.DataFrame(
                {"strategy": aligned_strategy, "market": aligned_market[symbol]}
            )
            rolling_corr = (
                combined["strategy"].rolling(window=30).corr(combined["market"])
            )

            # Calculate upside/downside correlations
            up_market = combined[combined["market"] > 0]
            down_market = combined[combined["market"] < 0]

            up_corr = (
                up_market["strategy"].corr(up_market["market"])
                if len(up_market) > 5
                else None
            )
            down_corr = (
                down_market["strategy"].corr(down_market["market"])
                if len(down_market) > 5
                else None
            )

            # Store results
            correlations[symbol] = {
                "pearson_correlation": pearson_corr,
                "avg_30d_rolling_correlation": rolling_corr.mean(),
                "max_30d_correlation": rolling_corr.max(),
                "min_30d_correlation": rolling_corr.min(),
                "upside_correlation": up_corr,
                "downside_correlation": down_corr,
                "rolling_correlation": rolling_corr,
            }

        # Convert to DataFrame
        corr_df = pd.DataFrame(
            {
                symbol: {k: v for k, v in data.items() if k != "rolling_correlation"}
                for symbol, data in correlations.items()
            }
        )

        return corr_df.T, correlations

    def analyze_market_regime_performance(
        self, volatility_percentile: float = 75, trend_periods: int = 20
    ) -> pd.DataFrame:
        """
        Analyze strategy performance across different market regimes.

        Args:
            volatility_percentile: Percentile to determine high volatility
            trend_periods: Number of periods to determine trend

        Returns:
            DataFrame with performance metrics by regime
        """
        if self.strategy_returns is None or not self.price_data:
            raise ValueError("Strategy returns and price data required")

        # Use BTC or first available asset as benchmark
        benchmark_symbol = (
            "BTC" if "BTC" in self.price_data else list(self.price_data.keys())[0]
        )
        benchmark_prices = self.price_data[benchmark_symbol]

        # Resample to daily
        daily_prices = benchmark_prices.resample("D").last().ffill()
        daily_returns = daily_prices.pct_change().dropna()

        # Calculate volatility (20-day rolling std)
        volatility = daily_returns.rolling(window=20).std()

        # Determine high/low volatility regimes
        high_vol_threshold = volatility.quantile(volatility_percentile / 100)
        volatility_regime = volatility.apply(
            lambda x: "high_volatility" if x > high_vol_threshold else "low_volatility"
        )

        # Determine trend regimes
        sma_short = daily_prices.rolling(window=trend_periods).mean()
        sma_long = daily_prices.rolling(window=trend_periods * 2).mean()

        trend_regime = pd.Series(index=daily_prices.index)
        trend_regime[sma_short > sma_long] = "uptrend"
        trend_regime[sma_short < sma_long] = "downtrend"
        trend_regime = trend_regime.ffill().bfill()

        # Combine regimes
        combined_regime = pd.DataFrame(
            {"volatility": volatility_regime, "trend": trend_regime}
        )

        # Create combined regime column
        combined_regime["regime"] = (
            combined_regime["trend"] + "_" + combined_regime["volatility"]
        )

        # Align with strategy returns
        common_dates = combined_regime.index.intersection(self.strategy_returns.index)
        aligned_regime = combined_regime.loc[common_dates]
        aligned_strategy = self.strategy_returns.loc[common_dates]

        # Combine regime and returns
        combined_data = pd.DataFrame(
            {"regime": aligned_regime["regime"], "strategy_return": aligned_strategy}
        )

        # Calculate performance by regime
        regime_performance = {}

        for regime in combined_data["regime"].unique():
            regime_returns = combined_data[combined_data["regime"] == regime][
                "strategy_return"
            ]

            if len(regime_returns) < 5:
                continue

            regime_performance[regime] = {
                "avg_daily_return": regime_returns.mean(),
                "cumulative_return": (1 + regime_returns).prod() - 1,
                "volatility": regime_returns.std(),
                "sharpe": (
                    regime_returns.mean() / regime_returns.std() * np.sqrt(252)
                    if regime_returns.std() > 0
                    else 0
                ),
                "win_rate": (regime_returns > 0).mean(),
                "num_days": len(regime_returns),
            }

        # Convert to DataFrame
        regime_df = pd.DataFrame(regime_performance).T

        return regime_df

    def plot_rolling_correlations(
        self, save_path: Optional[str] = None, show_plot: bool = True
    ) -> None:
        """
        Plot rolling correlations between strategy and market factors.

        Args:
            save_path: Path to save the chart image
            show_plot: Whether to display the chart
        """
        _, correlations = self.calculate_market_correlations()

        plt.figure(figsize=(12, 6))

        for symbol, data in correlations.items():
            rolling_corr = data["rolling_correlation"]
            plt.plot(rolling_corr.index, rolling_corr, label=f"{symbol} (30-day)")

        plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        plt.axhline(y=0.5, color="red", linestyle="--", alpha=0.3)
        plt.axhline(y=-0.5, color="red", linestyle="--", alpha=0.3)

        plt.title("Rolling 30-Day Correlation: Strategy vs. Markets")
        plt.ylabel("Correlation Coefficient")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)

        # Save if path provided
        if save_path:
            plt.savefig(save_path)

        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_regime_performance(
        self, save_path: Optional[str] = None, show_plot: bool = True
    ) -> None:
        """
        Plot strategy performance across different market regimes.

        Args:
            save_path: Path to save the chart image
            show_plot: Whether to display the chart
        """
        regime_performance = self.analyze_market_regime_performance()

        if regime_performance.empty:
            print("Not enough data for regime analysis")
            return

        # Create figure with subplots
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Plot average daily return by regime
        returns_by_regime = regime_performance["avg_daily_return"].sort_values()
        returns_by_regime.plot(kind="bar", ax=axs[0])
        axs[0].set_title("Average Daily Return by Market Regime")
        axs[0].set_ylabel("Daily Return")
        axs[0].axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Format y-axis as percentage
        axs[0].set_yticklabels([f"{x:.2%}" for x in axs[0].get_yticks()])

        # Plot Sharpe ratio by regime
        sharpe_by_regime = regime_performance["sharpe"].sort_values()
        sharpe_by_regime.plot(kind="bar", ax=axs[1])
        axs[1].set_title("Sharpe Ratio by Market Regime")
        axs[1].set_ylabel("Sharpe Ratio")
        axs[1].axhline(y=0, color="black", linestyle="-", alpha=0.3)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path)

        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()

    def generate_correlation_report(
        self,
        output_dir: str = "./reports/correlation",
        report_name: str = "correlation_analysis",
    ) -> str:
        """
        Generate comprehensive correlation analysis report.

        Args:
            output_dir: Directory to save the report
            report_name: Base name for report files

        Returns:
            Path to the report directory
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Calculate correlations
        corr_df, _ = self.calculate_market_correlations()

        # Generate rolling correlation chart
        rolling_path = os.path.join(output_dir, f"{report_name}_rolling.png")
        self.plot_rolling_correlations(rolling_path, False)

        # Generate regime performance chart
        try:
            regime_path = os.path.join(output_dir, f"{report_name}_regime.png")
            self.plot_regime_performance(regime_path, False)

            # Export regime performance
            regime_df = self.analyze_market_regime_performance()
            regime_csv = os.path.join(output_dir, f"{report_name}_regime.csv")
            regime_df.to_csv(regime_csv)
        except:
            print("Not enough data for regime analysis")

        # Export correlation statistics
        corr_csv = os.path.join(output_dir, f"{report_name}_correlations.csv")
        corr_df.to_csv(corr_csv)

        return output_dir


def analyze_correlations(
    strategy_trades: List[Dict[str, Any]],
    price_data: Dict[str, pd.DataFrame],
    initial_balance: float = 10000.0,
    output_dir: str = "./reports/correlation",
) -> Dict[str, Any]:
    """
    Convenience function to run correlation analysis and generate reports.

    Args:
        strategy_trades: List of trades from strategy backtest
        price_data: Dict mapping symbols to price DataFrames
        initial_balance: Initial account balance
        output_dir: Directory to save reports

    Returns:
        Dictionary with correlation results
    """
    # Initialize correlation analyzer
    analyzer = CorrelationAnalysis()

    # Load strategy trade data
    analyzer.load_trade_data(strategy_trades, initial_balance)

    # Load price data for each symbol
    for symbol, df in price_data.items():
        analyzer.load_price_data(symbol, df)

    # Generate report
    analyzer.generate_correlation_report(output_dir)

    # Return correlation results
    corr_df, corr_data = analyzer.calculate_market_correlations()

    try:
        regime_df = analyzer.analyze_market_regime_performance()
    except:
        regime_df = pd.DataFrame()

    return {"correlations": corr_df, "regime_performance": regime_df}
