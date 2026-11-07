"""Price scenarios lens: ranges, trend, bear/base/bull, wiring into the graph, UI view and vault report."""

import math

import numpy as np
import pandas as pd
import pytest

from fin_agent.agent.graph import build_graph
from fin_agent.llm.providers import FakeProvider
from fin_agent.metrics import scenarios as sc
from fin_agent.metrics.library import compute_metrics
from fin_agent.ui.scenario_view import scenario_view
from fin_agent.vault.render import render_report
from tests.conftest import scripted_handler


def closes(values):
    return pd.Series(values, index=pd.bdate_range("2025-01-01", periods=len(values)), dtype=float)


def test_range_is_symmetric_in_log_space_and_widens_with_horizon():
    rng = np.random.default_rng(0)
    c = closes(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 260))))
    r = sc.price_range(c)
    last = c.iloc[-1]
    assert r["range_3m_low"] < last < r["range_3m_high"]
    assert r["range_12m_low"] < r["range_3m_low"] and r["range_12m_high"] > r["range_3m_high"]
    assert math.isclose(math.log(r["range_12m_high"] / last), -math.log(r["range_12m_low"] / last))
    assert 0.25 < r["range_volatility"] < 0.40  # 2% daily ~ 32% annualized
    assert sc.price_range(closes([100] * 30)) == {}  # too short: nothing guessed


def test_trend_fits_a_clean_exponential_path():
    c = closes(100 * np.exp(0.001 * np.arange(252)))  # +0.1% a day
    t = sc.trend(c)
    assert t["trend_r2"] > 0.999
    assert math.isclose(t["trend_annual"], math.exp(0.252) - 1, rel_tol=1e-6)
    assert math.isclose(t["trend_3m_price"], c.iloc[-1] * math.exp(0.063), rel_tol=1e-6)


def test_growth_percentiles_cap_and_skip_non_positive_bases():
    assert sc.growth_percentiles(pd.Series([100, 110])) is None  # one observation is not a distribution
    p25, p50, p75 = sc.growth_percentiles(pd.Series([100, 110, 121, 400, -5, 50]))
    assert p75 == sc.GROWTH_CAP  # +230% capped at +50%
    assert p25 <= p50 <= p75


def test_scenarios_are_ordered_and_anchor_on_low_last_high():
    c = closes(np.concatenate([np.linspace(80, 120, 150), np.linspace(120, 100, 102)]))
    s = sc.scenarios(c, "earnings", pd.Series([100.0, 110.0, 121.0, 133.1]), multiple_now=10.0)
    assert s["scenario_bear_price"] <= s["scenario_base_price"] <= s["scenario_bull_price"]
    g = s["scenario_base_earnings_growth"]
    assert math.isclose(s["scenario_base_price"], 100 * (1 + g))
    assert math.isclose(s["scenario_bull_pe"], 10.0 * 120 / 100) and math.isclose(s["scenario_bear_pe"], 10.0 * 80 / 100)
    assert "scenario_base_book_growth" not in s


def test_demo_metrics_cite_ids_and_bank_without_prices_has_none(demo, demobank):
    ms = {m.name: m for m in compute_metrics(demo) if m.lens == "scenarios"}
    assert {"range_12m_low", "trend_r2", "scenario_bear_price", "scenario_bull_pe"} <= set(ms)
    assert ms["scenario_base_price"].id.startswith("DEMO.scenario_base_price:")
    assert ms["scenario_base_price"].unit == "VND"
    assert not [m for m in compute_metrics(demobank) if m.lens == "scenarios"]


@pytest.fixture  # function scope: the autouse offline fixture (fixture data, no network) must apply first
def demo_run():
    return build_graph(FakeProvider(scripted_handler())).invoke({"user_input": "Analyze DEMO"},
                                                                {"configurable": {"thread_id": "scen"}})


def test_scenarios_lens_runs_in_standard_depth(demo_run):
    out = demo_run["findings"]["scenarios"]
    assert not out.get("not_assessed") and out["findings"]


def test_scenario_view_and_report_section(demo_run):
    view = scenario_view(demo_run, "DEMO")
    assert list(view["table"]["Scenario"]) == ["Bear", "Base", "Bull"]
    assert len(view["band"]) > 40 and view["band"]["high"].iloc[-1] > view["band"]["low"].iloc[-1]
    assert set(view["points"]["case"]) == {"Bear", "Base", "Bull"}
    report = render_report(demo_run, "r1", day="2026-11-07")
    assert "## Price scenarios (12 months)" in report and "not forecasts" in report
    assert view["table"]["12-month price"].iloc[1] in report


def test_scenarios_not_assessed_without_prices():
    bank = build_graph(FakeProvider(scripted_handler(tickers=["DEMOBANK"]))).invoke(
        {"user_input": "Analyze the demo bank"}, {"configurable": {"thread_id": "scen-bank"}})
    assert bank["findings"]["scenarios"]["not_assessed"] is True
    assert scenario_view(bank, "DEMOBANK") is None
    assert "## Price scenarios" not in render_report(bank, "r2", day="2026-11-07")
