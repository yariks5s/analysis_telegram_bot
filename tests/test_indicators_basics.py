import sys
sys.path.append("/Users/yaroslav/cryptoBot")
from indicators import detect_order_blocks, detect_fvgs, detect_support_resistance_levels, detect_breaker_blocks

def test_detect_order_blocks(sample_dataframe):
    order_blocks = detect_order_blocks(sample_dataframe)
    assert isinstance(order_blocks.list, list)

def test_detect_fvgs(sample_dataframe):
    fvgs = detect_fvgs(sample_dataframe)
    assert isinstance(fvgs.list, list)

def test_detect_support_resistance_levels(sample_dataframe):
    levels = detect_support_resistance_levels(sample_dataframe, window=2, tolerance=0.1)
    assert isinstance(levels.list, list)

def test_detect_breaker_blocks(sample_dataframe):
    breaker_blocks = detect_breaker_blocks(sample_dataframe, {})
    assert isinstance(breaker_blocks.list, list)
