import pytest  # type: ignore

from data_fetching_instruments import fetch_from_json, analyze_data


@pytest.fixture
def setup_data():
    """Fixture to provide test data."""
    args = {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "category": "spot",
            "symbol": "ADAUSDT",
            "list": [
                [
                    "1734717600000",
                    "0.8864",
                    "0.8918",
                    "0.8802",
                    "0.8806",
                    "1718006.72",
                    "1524108.952454",
                ],
                [
                    "1734714000000",
                    "0.8784",
                    "0.8905",
                    "0.8748",
                    "0.8864",
                    "2357034.61",
                    "2080101.00459",
                ],
                [
                    "1734710400000",
                    "0.8759",
                    "0.896",
                    "0.8694",
                    "0.8784",
                    "4540774.46",
                    "3995051.10712",
                ],
                [
                    "1734706800000",
                    "0.8548",
                    "0.8912",
                    "0.8529",
                    "0.8759",
                    "6785736.93",
                    "5915563.667385",
                ],
                [
                    "1734703200000",
                    "0.861",
                    "0.8649",
                    "0.8444",
                    "0.8548",
                    "3392823.91",
                    "2902433.142977",
                ],
                [
                    "1734699600000",
                    "0.8309",
                    "0.8662",
                    "0.7987",
                    "0.861",
                    "13772176.91",
                    "11572106.340974",
                ],
                [
                    "1734696000000",
                    "0.7623",
                    "0.8349",
                    "0.7618",
                    "0.8309",
                    "10831041.14",
                    "8668030.270388",
                ],
                [
                    "1734692400000",
                    "0.7941",
                    "0.8097",
                    "0.7623",
                    "0.7623",
                    "11119622.19",
                    "8708597.281615",
                ],
                [
                    "1734688800000",
                    "0.8184",
                    "0.8236",
                    "0.7846",
                    "0.7941",
                    "9528974.5",
                    "7623319.707574",
                ],
                [
                    "1734685200000",
                    "0.8421",
                    "0.8503",
                    "0.8184",
                    "0.8184",
                    "8772707.09",
                    "7307436.993643",
                ],
                [
                    "1734681600000",
                    "0.8889",
                    "0.8897",
                    "0.8362",
                    "0.8421",
                    "5302445.37",
                    "4531357.077279",
                ],
                [
                    "1734678000000",
                    "0.8785",
                    "0.906",
                    "0.8784",
                    "0.8889",
                    "2915921.25",
                    "2615134.590824",
                ],
                [
                    "1734674400000",
                    "0.8847",
                    "0.8947",
                    "0.871",
                    "0.8785",
                    "1791737.69",
                    "1582550.93245",
                ],
                [
                    "1734670800000",
                    "0.9",
                    "0.9041",
                    "0.8839",
                    "0.8847",
                    "1306534.55",
                    "1168168.901247",
                ],
                [
                    "1734667200000",
                    "0.9047",
                    "0.9089",
                    "0.8983",
                    "0.9",
                    "2209536.75",
                    "1997962.670153",
                ],
            ],
        },
        "retExtInfo": {},
        "time": 1734719712626,
    }
    update = {}

    return args, update


@pytest.fixture
def dataframe(setup_data):
    args, update = setup_data
    # Default values
    return fetch_from_json(args)


def test_all_preferences(dataframe, sample_preferences_all):
    indicators = analyze_data(dataframe, sample_preferences_all, liq_lev_tolerance=0.05)
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
