import math

import pandas as pd
import pytest

from fin_agent.agent.graph import build_graph
from fin_agent.data.providers.vnstock_provider import market_data_from_frames
from fin_agent.llm.providers import FakeProvider
from fin_agent.metrics import technical as ta
from fin_agent.metrics.library import compute_metrics
from tests.conftest import scripted_handler


def s(values):
    return pd.Series(values, dtype=float)


def test_rsi_extremes_and_balance():
    assert ta.rsi(s(range(1, 40))) == 100.0  # only gains
    assert ta.rsi(s(range(40, 1, -1))) == pytest.approx(0.0)  # only losses
    # equal ups and downs: near 50, tilted towards the last move (Wilder smoothing weights recent moves)
    assert 50 < ta.rsi(s([10, 11] * 20)) < 55
    assert 45 < ta.rsi(s([11, 10] * 20)) < 50
    assert math.isnan(ta.rsi(s(range(10))))  # fewer than 15 closes


def test_rsi_matches_wilder_reference():
    # hand-rolled Wilder recursion on a fixed series, independent of the implementation
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61,
              46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64]
    d = [b - a for a, b in zip(closes, closes[1:])]
    g = sum(max(x, 0) for x in d[:14]) / 14
    l = sum(max(-x, 0) for x in d[:14]) / 14
    for x in d[14:]:
        g, l = (g * 13 + max(x, 0)) / 14, (l * 13 + max(-x, 0)) / 14
    assert ta.rsi(s(closes)) == pytest.approx(100 - 100 / (1 + g / l))


def test_macd_flat_series_is_zero_and_needs_history():
    assert ta.macd(s([50.0] * 60))[2] == pytest.approx(0.0)
    assert all(math.isnan(x) for x in ta.macd(s([50.0] * 30)))


def test_returns_drawdown_range_volume():
    c = s([100] * 252 + [110])
    assert ta.period_return(c, 21) == pytest.approx(0.10)
    peak_then_fall = s([100, 120, 90, 108] + [108] * 248)
    assert ta.max_drawdown(peak_then_fall) == pytest.approx(90 / 120 - 1)
    hi, lo = ta.range_position(peak_then_fall)
    assert hi == pytest.approx(108 / 120 - 1) and lo == pytest.approx(108 / 90 - 1)
    assert ta.volume_ratio(s([100] * 40 + [200] * 20)) == pytest.approx(200 / ((100 * 40 + 200 * 20) / 60))
    assert ta.volatility(s([100.0] * 61)) == 0.0


def test_relative_return_aligns_on_common_dates():
    days = pd.bdate_range("2025-01-01", periods=70)
    stock = pd.Series([100.0] * 69 + [110.0], index=days)
    bench = pd.Series([1000.0] * 69 + [1050.0], index=days).drop(days[10])  # a missing index day
    assert ta.relative_return(stock, bench, 63) == pytest.approx(0.10 - 0.05)


def test_demo_technical_metrics(demo):
    ms = {m.name: m for m in compute_metrics(demo) if m.lens == "technical"}
    assert {m.period for m in ms.values()} == {"2025-03-31"}
    assert {"sma_200", "rsi_14", "return_12m", "rel_return_12m", "max_drawdown_12m"} <= set(ms)
    assert 0 <= ms["rsi_14"].value <= 100 and ms["rsi_14"].display().count(".") == 1
    assert ms["sma_50"].unit == "VND"
    assert ms["max_drawdown_12m"].value < 0 < ms["pct_from_52w_low"].value


def test_no_technical_metrics_without_history(demobank):
    assert not [m for m in compute_metrics(demobank) if m.lens == "technical"]


def test_vnstock_bars_use_one_scale_for_the_whole_series():
    ov = pd.DataFrame([{"outstanding_shares": 1e9, "as_of_date": "2025-12-31"}])
    hist = pd.DataFrame({"time": pd.to_datetime(["2026-11-04", "2026-11-05"]), "close": [54.8, 54.2], "volume": [1e6, 2e6]})
    md = market_data_from_frames(hist, ov, source="t")
    assert [b["close"] for b in md.history] == [54800.0, 54200.0]
    # a stock that crosses 1,000 must not get mixed scales within its history
    hist2 = hist.assign(close=[990.0, 1010.0])
    assert [b["close"] for b in market_data_from_frames(hist2, ov, source="t").history] == [990.0, 1010.0]


def test_technical_lens_runs_with_history_and_is_skipped_without():
    out = build_graph(FakeProvider(scripted_handler())).invoke({"user_input": "Analyze DEMO"},
                                                               {"configurable": {"thread_id": "ta"}})
    assert not out["findings"]["technical"].get("not_assessed")
    assert out["findings"]["technical"]["findings"]
    bank = build_graph(FakeProvider(scripted_handler(tickers=["DEMOBANK"]))).invoke(
        {"user_input": "Analyze the demo bank"}, {"configurable": {"thread_id": "ta-bank"}})
    assert bank["findings"]["technical"]["not_assessed"] is True
    assert "technical" in bank["recommendation"]["not_assessed"]
