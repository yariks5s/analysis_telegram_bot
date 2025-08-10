"""
Plot builder module for CryptoBot.

This module contains functions for generating and customizing cryptocurrency charts
with various technical analysis indicators.
"""

import matplotlib.patches as mpatches  # type: ignore
import matplotlib.lines as mlines  # type: ignore
import mplfinance as mpf  # type: ignore
import pandas as pd  # type: ignore

from typing import Dict

from src.core.utils import logger
from src.visualization.chart_styles import ChartTheme


def plot_price_chart(
    df: pd.DataFrame,
    indicators: Dict[str, bool],
    show_legend: bool = True,
    show_volume: bool = True,
    dark_mode: bool = False,
):
    """
    Generate a candlestick chart with detected order blocks, FVGs, and support/resistance levels.

    Args:
        df: DataFrame with OHLCV data
        indicators: Dict of indicator objects
        show_legend: Whether to show chart legend
        show_volume: Whether to show volume
        dark_mode: Whether to use dark mode styling

    Returns:
        str: Path to the saved chart image
    """
    s = ChartTheme.get_mpf_style(dark_mode)

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
        add_fvgs(ax, indicators.fvgs, dark_mode)

    if indicators.order_blocks:
        add_order_blocks(ax, indicators.order_blocks, df, dark_mode)

    if indicators.liquidity_levels:
        add_liquidity_levels(ax, indicators.liquidity_levels, dark_mode)

    if indicators.breaker_blocks:
        add_breaker_blocks(ax, indicators.breaker_blocks, dark_mode)

    if indicators.liquidity_pools:
        add_liquidity_pools(ax, indicators.liquidity_pools, dark_mode)

    if show_legend:
        add_legend(ax, indicators, dark_mode)

    fig.subplots_adjust(left=0.05, right=0.95)
    fig.savefig("crypto_chart.png", bbox_inches="tight", pad_inches=0.1)

    return "crypto_chart.png"


def add_legend(ax, indicators, dark_mode=False):
    """
    Add legend to the chart based on available indicators.

    Args:
        ax: Matplotlib axes object
        indicators: Dict of indicator objects
        dark_mode: Whether to use dark mode styling
    """
    handles = []
    if indicators.order_blocks:
        if dark_mode:
            bearish_color = "#8BE9FD"  # Bright cyan for dark mode
            bullish_color = "#FF79C6"  # Bright pink for dark mode
            markersize = 10
        else:
            bearish_color = "blue"  # Original colors for light mode
            bullish_color = "purple"  # Original colors for light mode
            markersize = 10

        handles.append(
            mlines.Line2D(
                [],
                [],
                color=bearish_color,
                marker="o",
                markersize=markersize,
                label="Bearish Order Block",
                linestyle="None",
            )
        )
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=bullish_color,
                marker="o",
                markersize=markersize,
                label="Bullish Order Block",
                linestyle="None",
            )
        )

    if indicators.fvgs:
        if dark_mode:
            fvg_color = "#BD93F9"  # Bright purple for dark mode
            fvg_alpha = 0.5
        else:
            fvg_color = "blue"  # Original color
            fvg_alpha = 0.3

        handles.append(mpatches.Patch(color=fvg_color, alpha=fvg_alpha, label="FVG"))

    if indicators.breaker_blocks:
        if dark_mode:
            bullish_color = "#50FA7B"  # Bright green for dark mode
            bearish_color = "#FF5555"  # Bright red for dark mode
            alpha = 0.3
        else:
            bullish_color = "green"  # Original colors
            bearish_color = "red"  # Original colors
            alpha = 0.05

        handles.append(
            mpatches.Patch(
                color=bullish_color, alpha=alpha, label="Bullish Breaker Block"
            )
        )
        handles.append(
            mpatches.Patch(
                color=bearish_color, alpha=alpha, label="Bearish Breaker Block"
            )
        )

    if indicators.liquidity_levels:
        if dark_mode:
            liq_level_color = "#FFB86C"  # Bright orange for dark mode
        else:
            liq_level_color = "orange"  # Original color

        handles.append(
            mlines.Line2D(
                [],
                [],
                color=liq_level_color,
                linestyle="--",
                label="Liquidity Level",
            )
        )

    if indicators.liquidity_pools:
        if dark_mode:
            pool_color = "#8BE9FD"  # Bright cyan for dark mode
            pool_alpha = 0.3
        else:
            pool_color = "cyan"  # Original color
            pool_alpha = 0.2

        handles.append(
            mpatches.Patch(color=pool_color, alpha=pool_alpha, label="Liquidity Pool")
        )

    ax[0].legend(handles=handles, loc="upper left")


def add_order_blocks(ax, order_blocks, df, dark_mode=False):
    """
    Add order blocks to the chart.

    Args:
        ax: Matplotlib axes object
        order_blocks: OrderBlocks object containing order blocks to plot
        df: DataFrame with price data
        dark_mode: Whether to use dark mode styling
    """
    for block in order_blocks.list:
        idx = block.index
        low = block.low

        if idx < 0 or idx >= len(df):
            logger.warning(f"Invalid order block index: {idx}")
            continue

        circle_y_position = low * 0.995

        style = ChartTheme.get_order_block_style(dark_mode, block.is_bullish())

        ax[0].scatter(
            idx,
            circle_y_position,
            color=style["color"],
            s=style["marker_size"],
            zorder=5,
            label="Order Block",
        )


def add_fvgs(ax, fvgs, dark_mode=False):
    """
    Add Fair Value Gaps (FVGs) to the chart.

    Args:
        ax: Matplotlib axes object
        fvgs: FVGs object containing FVGs to plot
        dark_mode: Whether to use dark mode styling
    """
    for fvg in fvgs.list:
        start_idx = fvg.start_index
        end_idx = fvg.end_index
        start_price = fvg.start_price
        end_price = fvg.end_price

        style = ChartTheme.get_fvg_style(dark_mode)

        ax[0].fill_betweenx(
            y=[start_price, end_price],
            x1=start_idx,
            x2=end_idx,
            color=style["color"],
            alpha=style["alpha"],
            label="FVG",
        )


def add_liquidity_levels(ax, liquidity_levels, dark_mode=False):
    """
    Add liquidity levels to the chart.

    Args:
        ax: Matplotlib axes object
        liquidity_levels: LiquidityLevels object containing levels to plot
        dark_mode: Whether to use dark mode styling
    """
    for level in liquidity_levels.list:
        style = ChartTheme.get_liquidity_level_style(dark_mode)

        ax[0].axhline(
            y=level.price,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )


def add_breaker_blocks(ax, breaker_blocks, dark_mode=False):
    """
    Add breaker blocks to the chart.

    Args:
        ax: Matplotlib axes object
        breaker_blocks: BreakerBlocks object containing blocks to plot
        dark_mode: Whether to use dark mode styling
    """
    for block in breaker_blocks.list:
        start_index = block.index
        end_index = start_index + 2

        y1, y2 = block.zone

        style = ChartTheme.get_breaker_block_style(dark_mode, block.is_bullish())

        # Fill between x-axis and y-axis values
        ax[0].fill_betweenx(
            y=[y1, y2],  # Price range (y-axis)
            x1=start_index,  # Start time of the block
            x2=end_index,  # End time of the block
            color=style["color"],
            alpha=style["alpha"],
        )


def add_liquidity_pools(ax, liquidity_pools, dark_mode=False):
    """
    Add liquidity pools to the chart as semi-transparent horizontal bands.
    The opacity of each band is determined by the pool's strength.

    Args:
        ax: Matplotlib axes object
        liquidity_pools: LiquidityPools object containing pools to plot
        dark_mode: Whether to use dark mode styling
    """
    for pool in liquidity_pools.list:
        price_range = pool.price * 0.001  # 0.1% of price as default range

        style = ChartTheme.get_liquidity_pool_style(dark_mode)

        try:
            if len(ax[0].get_lines()) > 0:
                x_range = [0, len(ax[0].get_lines()[0].get_xdata())]
            else:
                x_range = [0, 100]
        except (IndexError, AttributeError):
            logger.warning("Could not determine x-axis range, using default value")
            x_range = [0, 100]  # Default length as fallback

        ax[0].fill_between(
            x=x_range,
            y1=[pool.price - price_range, pool.price - price_range],
            y2=[pool.price + price_range, pool.price + price_range],
            color=style["color"],
            alpha=style["base_alpha"] * pool.strength,
            label="Liquidity Pool",
        )

        ax[0].axhline(
            y=pool.price,
            color=style["color"],
            linestyle=style["linestyle"],
            alpha=style["line_alpha"],
            linewidth=style["linewidth"],
        )
