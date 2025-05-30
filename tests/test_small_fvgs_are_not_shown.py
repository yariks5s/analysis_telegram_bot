import pytest  # type: ignore
from indicators import (
    detect_order_blocks,
    detect_fvgs,
    detect_liquidity_levels,
    detect_breaker_blocks,
)
from IndicatorUtils.breaker_block_utils import BreakerBlocks
from IndicatorUtils.fvg_utils import FVGs
from IndicatorUtils.liquidity_level_utils import LiquidityLevels
from IndicatorUtils.order_block_utils import OrderBlocks

from data_fetching_instruments import fetch_from_json


@pytest.fixture
def setup_data(small_fvgs):
    """Fixture to provide test data."""
    args = small_fvgs
    update = {}

    return args, update


@pytest.fixture
def dataframe(setup_data):
    args, update = setup_data
    # Default values
    return fetch_from_json(args)


def test_detect_fvgs(dataframe):
    """Test detect_fvgs function."""
    fvgs = detect_fvgs(dataframe)
    assert isinstance(fvgs, FVGs), "FVGs should return a FVGs instance."
    assert (
        str(fvgs)
        == """
None."""
    )
