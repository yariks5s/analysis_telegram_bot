import pytest  # type: ignore

from data_fetching_instruments import fetch_from_json, analyze_data


@pytest.fixture
def setup_data(sample_response_for_ll):
    """Fixture to provide test data."""
    args = sample_response_for_ll
    update = {}

    return args, update


@pytest.fixture
def dataframe(setup_data):
    args, update = setup_data
    # Default values
    return fetch_from_json(args)


def test_all_preferences(dataframe, sample_preferences_all):
    indicators = analyze_data(dataframe, sample_preferences_all, liq_lev_tolerance={})
    assert indicators.liquidity_levels.list
    assert indicators.breaker_blocks.list
    assert indicators.order_blocks.list
    assert indicators.fvgs.list


def test_fvg_preferences(dataframe, sample_preferences_fvg):
    indicators = analyze_data(dataframe, sample_preferences_fvg, liq_lev_tolerance=0.05)
    assert not indicators.liquidity_levels.list
    assert not indicators.breaker_blocks.list
    assert not indicators.order_blocks.list
    assert indicators.fvgs.list


def test_ob_preferences(dataframe, sample_preferences_ob):
    indicators = analyze_data(dataframe, sample_preferences_ob, liq_lev_tolerance=0.05)
    assert not indicators.liquidity_levels.list
    assert not indicators.breaker_blocks.list
    assert indicators.order_blocks.list
    assert not indicators.fvgs.list


def test_ll_preferences(dataframe, sample_preferences_ll):
    indicators = analyze_data(dataframe, sample_preferences_ll, liq_lev_tolerance=0.05)
    assert indicators.liquidity_levels.list
    assert not indicators.breaker_blocks.list
    assert not indicators.order_blocks.list
    assert not indicators.fvgs.list


def test_ll_bb_preferences(dataframe, sample_preferences_ll_bb):
    indicators = analyze_data(
        dataframe, sample_preferences_ll_bb, liq_lev_tolerance=0.05
    )
    assert indicators.liquidity_levels.list
    assert indicators.breaker_blocks.list
    assert not indicators.order_blocks.list
    assert not indicators.fvgs.list


def test_none_preferences(dataframe, sample_preferences_none):
    indicators = analyze_data(
        dataframe, sample_preferences_none, liq_lev_tolerance=0.05
    )
    assert not indicators.liquidity_levels.list
    assert not indicators.breaker_blocks.list
    assert not indicators.order_blocks.list
    assert not indicators.fvgs.list


def test_bb_only_preferences(dataframe, sample_preferences_bb):
    indicators = analyze_data(dataframe, sample_preferences_bb, liq_lev_tolerance=0.05)
    assert not indicators.liquidity_levels.list
    assert not indicators.breaker_blocks.list
    assert not indicators.order_blocks.list
    assert not indicators.fvgs.list
