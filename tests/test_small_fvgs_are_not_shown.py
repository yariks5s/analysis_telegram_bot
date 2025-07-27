import pytest  # type: ignore
from src.analysis.detection.indicators import (
    detect_order_blocks,
    detect_fvgs,
    detect_liquidity_levels,
    detect_breaker_blocks,
)
from src.analysis.utils.breaker_block_utils import BreakerBlocks
from src.analysis.utils.fvg_utils import FVGs
from src.analysis.utils.liquidity_level_utils import LiquidityLevels
from src.analysis.utils.order_block_utils import OrderBlocks

from src.api.data_fetcher import fetch_from_json


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
