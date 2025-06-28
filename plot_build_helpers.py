import matplotlib.patches as mpatches  # type: ignore
import matplotlib.lines as mlines  # type: ignore

import mplfinance as mpf  # type: ignore
import pandas as pd  # type: ignore

from typing import List, Dict
from utils import logger


def plot_price_chart(
    df: pd.DataFrame,
    indicators: Dict[str, bool],
    show_legend: bool = True,
    show_volume: bool = True,
    dark_mode: bool = False,
):
    """
    Generate a candlestick chart with detected order blocks, FVGs, and support/resistance levels as horizontal lines.
    """

    # Define styles for light and dark mode
    if dark_mode:
        mc = mpf.make_marketcolors(
            up="#8AFF80",
            down="#FF5555",
            edge="inherit",
            wick={"up": "#8AFF80", "down": "#FF5555"},
            volume={"up": "#8AFF80", "down": "#FF5555"},
        )
        s = mpf.make_mpf_style(
            # Use a valid base style - 'nightclouds' is a built-in dark theme
            base_mpf_style="nightclouds",
            marketcolors=mc,
            gridstyle=":",
            gridcolor="#44475A",
            facecolor="#282A36",
            edgecolor="#44475A",
            figcolor="#282A36",
            y_on_right=False,
        )
    else:
        # Default light style (yahoo)
        s = "yahoo"

    fig, ax = mpf.plot(
        df,
        type="candle",
        style=s,
        volume=show_volume,
        returnfig=True,
        ylabel="Price (USDT)",
        ylabel_lower="Volume",
        tight_layout=True,
        figsize=(12, 8),
    )

    if indicators.fvgs:
        add_fvgs(ax, indicators.fvgs)

    if indicators.order_blocks:
        add_order_blocks(ax, indicators.order_blocks, df)

    if indicators.liquidity_levels:
        add_liquidity_levels(ax, indicators.liquidity_levels)

    if indicators.breaker_blocks:
        add_breaker_blocks(ax, indicators.breaker_blocks)

    if indicators.liquidity_pools:
        add_liquidity_pools(ax, indicators.liquidity_pools)

    if show_legend:
        add_legend(ax, indicators)

    fig.subplots_adjust(left=0.05, right=0.95)
    fig.savefig("crypto_chart.png", bbox_inches="tight", pad_inches=0.1)

    return "crypto_chart.png"


def add_legend(ax, indicators):
    handles = []
    if indicators.order_blocks:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="blue",
                marker="o",
                markersize=10,
                label="Bearish Order Block",
                linestyle="None",
            )
        )
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="purple",
                marker="o",
                markersize=10,
                label="Bullish Order Block",
                linestyle="None",
            )
        )

    if indicators.fvgs:
        handles.append(mpatches.Patch(color="purple", alpha=0.3, label="FVG"))

    if indicators.breaker_blocks:
        handles.append(
            mpatches.Patch(color="green", alpha=0.05, label="Bullish Breaker Block")
        )
        handles.append(
            mpatches.Patch(color="red", alpha=0.05, label="Bearish Breaker Block")
        )

    if indicators.liquidity_levels:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="orange",
                linestyle="--",
                markersize=10,
                label="Liquidity level",
            )
        )

    if indicators.liquidity_pools:
        handles.append(mpatches.Patch(color="cyan", alpha=0.2, label="Liquidity Pool"))

    ax[0].legend(handles=handles, loc="upper left")


def add_order_blocks(ax, order_blocks, df):
    for block in order_blocks.list:
        idx = block.index
        low = block.low

        if idx < 0 or idx >= len(df):
            logger.warning(f"Invalid order block index: {idx}")
            continue

        circle_y_position = low * 0.995  # Slightly above the high for visibility

        ax[0].scatter(
            idx,
            circle_y_position,
            color="purple" if block.is_bullish() else "blue",
            s=20,
            zorder=5,
            label="Order Block",
        )


def add_fvgs(ax, fvgs):
    for fvg in fvgs.list:
        start_idx = fvg.start_index
        end_idx = fvg.end_index
        start_price = fvg.start_price
        end_price = fvg.end_price

        # Plot the FVG as a horizontal rectangle
        ax[0].fill_betweenx(
            y=[start_price, end_price],
            x1=start_idx,
            x2=end_idx,
            # color='blue' if start_price < end_price else 'orange',
            color="blue",
            alpha=0.2,
            label="FVG",
        )


def add_liquidity_levels(ax, liquidity_levels):
    for level in liquidity_levels.list:
        ax[0].axhline(
            y=level.price,
            color="orange",
            linestyle="--",
            linewidth=1,
        )


def add_breaker_blocks(ax, breaker_blocks):
    for block in breaker_blocks.list:
        # Start time and end time (extend the block forward by 1-3 candles)
        start_index = block.index
        end_index = start_index + 2  # Add ~2 candles forward

        # Breaker block price range
        y1, y2 = block.zone

        # Choose color based on type
        color = "green" if block.is_bullish() else "red"

        # Fill between x-axis and y-axis values
        ax[0].fill_betweenx(
            y=[y1, y2],  # Price range (y-axis)
            x1=start_index,  # Start time of the block
            x2=end_index,  # End time of the block
            color=color,
            alpha=0.05,
        )


def add_liquidity_pools(ax, liquidity_pools):
    """
    Add liquidity pools to the chart as semi-transparent horizontal bands.
    The opacity of each band is determined by the pool's strength.
    """
    for pool in liquidity_pools.list:
        # Calculate the price range for the pool (using ATR or a fixed percentage)
        price_range = pool.price * 0.001  # 0.1% of price as default range

        # Create a horizontal band for the pool
        ax[0].fill_between(
            x=[0, len(ax[0].get_lines()[0].get_xdata())],
            y1=[pool.price - price_range, pool.price - price_range],
            y2=[pool.price + price_range, pool.price + price_range],
            color="cyan",
            alpha=0.2 * pool.strength,  # Scale opacity by pool strength
            label="Liquidity Pool",
        )

        # Add a horizontal line at the pool's price level
        ax[0].axhline(y=pool.price, color="cyan", linestyle=":", alpha=0.5, linewidth=1)
