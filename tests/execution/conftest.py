"""Shared fixtures for execution-layer tests."""

from __future__ import annotations
import pytest

from tests.execution.fixtures.mock_mt5 import MockMT5
from tests.execution.fixtures.mock_orders import (  # noqa: F401
    make_intent, make_signal, make_signal_sell,
)
from tests.execution.fixtures.mock_positions import (  # noqa: F401
    make_griff_open, make_griff_pending, make_position, make_tick,
)


@pytest.fixture
def mock_mt5():
    return MockMT5()


@pytest.fixture
def patch_live_broker_mt5(mock_mt5, monkeypatch):
    from execution import live_broker
    monkeypatch.setattr(live_broker, "mt5", mock_mt5)
    return mock_mt5


@pytest.fixture
def patch_router_mt5(mock_mt5, monkeypatch):
    from execution import order_router as griff_order_router
    monkeypatch.setattr(griff_order_router, "mt5", mock_mt5)
    return mock_mt5


@pytest.fixture
def patch_connector_mt5(mock_mt5, monkeypatch):
    from data import mt5_connector
    monkeypatch.setattr(mt5_connector, "mt5", mock_mt5)
    return mock_mt5


@pytest.fixture
def intent():
    return make_intent()


@pytest.fixture
def buy_signal():
    return make_signal()


@pytest.fixture
def sell_signal():
    return make_signal_sell()


@pytest.fixture
def tick():
    return make_tick()


@pytest.fixture
def open_position():
    return make_position()
