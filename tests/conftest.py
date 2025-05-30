import pytest  # type: ignore
import pandas as pd  # type: ignore
import sys
import os
import json

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from database import init_db


@pytest.fixture(scope="session", autouse=True)
def setup_environment():
    init_db()


@pytest.fixture
def sample_dataframe():
    data = {
        "Open": [29000, 29500],
        "High": [30000, 31000],
        "Low": [28000, 29000],
        "Close": [29500, 30000],
        "Volume": [1000, 1500],
    }
    dates = ["2021-01-01 00:00:00", "2021-01-01 01:00:00"]
    return pd.DataFrame(data, index=pd.to_datetime(dates))


@pytest.fixture
def sample_preferences_all():
    return {
        "order_blocks": True,
        "fvgs": True,
        "liquidity_levels": True,
        "breaker_blocks": True,
        "liquidity_pools": True,
    }


@pytest.fixture
def sample_preferences_ob():
    return {
        "order_blocks": True,
        "fvgs": False,
        "liquidity_levels": False,
        "breaker_blocks": False,
        "liquidity_pools": False,
    }


@pytest.fixture
def sample_preferences_fvg():
    return {
        "order_blocks": False,
        "fvgs": True,
        "liquidity_levels": False,
        "breaker_blocks": False,
        "liquidity_pools": False,
    }


@pytest.fixture
def sample_preferences_ll():
    return {
        "order_blocks": False,
        "fvgs": False,
        "liquidity_levels": True,
        "breaker_blocks": False,
        "liquidity_pools": False,
    }


@pytest.fixture
def sample_preferences_ll_bb():
    return {
        "order_blocks": False,
        "fvgs": False,
        "liquidity_levels": True,
        "breaker_blocks": True,
        "liquidity_pools": False,
    }


@pytest.fixture
def sample_preferences_bb():
    return {
        "order_blocks": False,
        "fvgs": False,
        "liquidity_levels": False,
        "breaker_blocks": True,
        "liquidity_pools": False,
    }


@pytest.fixture
def sample_preferences_lp():
    return {
        "order_blocks": False,
        "fvgs": False,
        "liquidity_levels": False,
        "breaker_blocks": False,
        "liquidity_pools": True,
    }


@pytest.fixture
def sample_preferences_none():
    return {
        "order_blocks": False,
        "fvgs": False,
        "liquidity_levels": False,
        "breaker_blocks": False,
        "liquidity_pools": False,
    }


@pytest.fixture
def three_ll_input():
    json_path = os.path.join(
        os.path.dirname(__file__), "test_context/three_ll_input.json"
    )
    with open(json_path, "r") as f:
        return json.load(f)


@pytest.fixture
def sample_response_for_ll():
    json_path = os.path.join(
        os.path.dirname(__file__), "test_context/sample_response_for_ll.json"
    )
    with open(json_path, "r") as f:
        return json.load(f)


@pytest.fixture
def advanced_indicators():
    json_path = os.path.join(
        os.path.dirname(__file__), "test_context/advanced_indicators.json"
    )
    with open(json_path, "r") as f:
        return json.load(f)


@pytest.fixture
def small_fvgs():
    json_path = os.path.join(os.path.dirname(__file__), "test_context/small_fvgs.json")
    with open(json_path, "r") as f:
        return json.load(f)
