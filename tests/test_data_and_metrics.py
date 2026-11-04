import math

import pandas as pd
import pytest

from fin_agent.data.providers.edgar import parse_companyfacts
from fin_agent.data.providers.vnstock_provider import normalize_frames
from fin_agent.data.schema import DataQualityError, Sector, assert_quality, run_dq_checks
from fin_agent.metrics.library import compute_metrics, line_item_metrics, trend


def by_name(metrics, name, period):
    return next(m for m in metrics if m.name == name and m.period == period).value


def test_dq_passes_for_balanced_fixture(demo):
    assert assert_quality(demo) == []


def test_dq_catches_balance_gap(demo):
    demo.data.loc[demo.data.index[-1], "equity"] += 1000
    with pytest.raises(DataQualityError) as e:
        assert_quality(demo)
    assert "balance_identity" in str(e.value)


def test_dq_flags_missing_core_lines(demo):
    demo.data = demo.data.drop(columns=["cfo"])
    assert any(i.check == "core_lines" for i in run_dq_checks(demo))


def test_general_metrics_hand_computed(demo):
    m = compute_metrics(demo)
    p = "2024-12-31"
    assert by_name(m, "gross_margin", p) == pytest.approx(5530 / 15800)
    assert by_name(m, "roe", p) == pytest.approx(2320 / ((11000 + 12000) / 2))
    assert by_name(m, "dso", p) == pytest.approx((4500 + 5500) / 2 / 15800 * 365)
    dio = (2700 + 3200) / 2 / 10270 * 365
    dpo = (1950 + 2100) / 2 / 10270 * 365
    assert by_name(m, "ccc", p) == pytest.approx(by_name(m, "dso", p) + dio - dpo)
    assert by_name(m, "cfo_to_net_income", p) == pytest.approx(1500 / 2320)
    assert by_name(m, "dividend_fcf_cover", p) == pytest.approx((1500 - 1300) / 800)
    assert by_name(m, "debt_to_equity", p) == pytest.approx((3300 + 1700) / 12000)
    # first period uses ending balances (no prior)
    assert by_name(m, "roe", "2020-12-31") == pytest.approx(1720 / 8000)
    assert all(m_.id.startswith("DEMO.") for m_ in m)


def test_dupont_identity(demo):
    m = compute_metrics(demo)
    p = "2023-12-31"
    product = by_name(m, "net_margin", p) * by_name(m, "asset_turnover", p) * by_name(m, "equity_multiplier", p)
    assert product == pytest.approx(by_name(m, "roe", p))


def test_bank_metrics_and_sector_filter(demobank):
    m = compute_metrics(demobank)
    names = {x.name for x in m}
    assert "gross_margin" not in names and "ccc" not in names
    assert by_name(m, "ldr", "2024-12-31") == pytest.approx(760000 / 780000)
    assert by_name(m, "npl_ratio", "2024-12-31") == pytest.approx(15200 / 760000)
    assert trend(m, "npl_ratio") == "up"


def test_line_items_are_citable(demo):
    ids = {m.id for m in line_item_metrics(demo)}
    assert "DEMO.revenue:2024-12-31" in ids


def test_edgar_parser_uses_frames_and_signs():
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {"end": "2023-12-31", "val": 1_000_000_000, "frame": "CY2023", "form": "10-K"},
            {"end": "2023-12-31", "val": 1_000_000_000, "form": "10-K"},  # duplicate without frame: ignored
            {"end": "2023-09-30", "val": 250_000_000, "frame": "CY2023Q3", "form": "10-Q"},  # quarterly: ignored
        ]}},
        "Assets": {"units": {"USD": [{"end": "2023-12-31", "val": 5_000_000_000, "frame": "CY2023Q4I"}]}},
        "StockholdersEquity": {"units": {"USD": [{"end": "2023-12-31", "val": 2_000_000_000, "frame": "CY2023Q4I"}]}},
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [{"end": "2023-12-31", "val": 100_000_000, "frame": "CY2023"}]}},
    }}}
    df = parse_companyfacts(facts, years=5)
    row = df.iloc[-1]
    assert row["revenue"] == 1000 and row["total_assets"] == 5000
    assert row["capex"] == -100  # outflow sign
    assert row["total_liabilities"] == 3000  # derived


def test_vnstock_column_mapping_and_rescale():
    income = pd.DataFrame({
        "ticker": ["XYZ", "XYZ"], "yearReport": [2023, 2024],
        "Revenue (Bn. VND)": [1.0e12, 1.2e12], "Cost of Sales": [-0.7e12, -0.8e12],
        "Net Profit For the Year": [0.1e12, 0.12e12],
    })
    balance = pd.DataFrame({
        "yearReport": [2023, 2024], "TOTAL ASSETS (Bn. VND)": [2.0e12, 2.4e12],
        "LIABILITIES (Bn. VND)": [1.2e12, 1.4e12], "OWNER'S EQUITY(Bn.VND)": [0.8e12, 1.0e12],
    })
    df, notes, sector = normalize_frames([income, balance], years=5, freq="annual")
    assert sector is Sector.GENERAL
    assert df.loc[df.index[-1], "revenue"] == pytest.approx(1200)  # rescaled to VND bn
    assert df.loc[df.index[-1], "gross_profit"] == pytest.approx(400)
    assert df.loc[df.index[-1], "equity"] == pytest.approx(1000)
    assert any("rescaled" in n for n in notes)
