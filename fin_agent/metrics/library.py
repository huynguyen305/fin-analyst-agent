"""Deterministic metrics library (ADR-004: LLMs reason, Python counts).

Every figure the agent may mention is computed here and gets a stable
``metric_id`` (``<TICKER>.<name>:<period_end>``). The LLM cites ids; the critic
reconciles the narrative against these values.

Conventions:
- Ratios are fractions (0.25 = 25%), days are days, amounts are in the
  statements' unit (VND bn / USD mn).
- Averages use (prior + current) / 2 where a prior period exists, otherwise the
  ending balance; the ``formula`` string says which.
- Missing inputs give no metric (never a zero).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

from fin_agent.data.schema import Sector, Statements


@dataclass(frozen=True)
class Metric:
    id: str
    name: str
    lens: str
    period: str
    value: float
    unit: str  # "ratio" | "days" | "x" | currency unit
    formula: str

    def display(self) -> str:
        if self.unit == "ratio":
            return f"{self.value * 100:.1f}%"
        if self.unit == "days":
            return f"{self.value:.0f} days"
        if self.unit == "x":
            return f"{self.value:.2f}x"
        return f"{self.value:,.1f} {self.unit}"

    def to_dict(self) -> dict:
        return {**asdict(self), "display": self.display()}


@dataclass(frozen=True)
class MetricDef:
    name: str
    lens: str
    unit: str
    formula: str
    sectors: tuple[Sector, ...]


LENSES = ("profitability", "health", "operational", "cashflow_quality", "valuation", "risk_governance")

DEFS: dict[str, MetricDef] = {}


def _d(name, lens, unit, formula, sectors=(Sector.GENERAL,)):
    DEFS[name] = MetricDef(name, lens, unit, formula, tuple(sectors))


ALL_SECTORS = tuple(Sector)
# profitability
_d("gross_margin", "profitability", "ratio", "gross_profit / revenue")
_d("operating_margin", "profitability", "ratio", "operating_income / revenue")
_d("net_margin", "profitability", "ratio", "net_income / revenue")
_d("roe", "profitability", "ratio", "net_income / avg(equity)", ALL_SECTORS)
_d("roa", "profitability", "ratio", "net_income / avg(total_assets)", ALL_SECTORS)
_d("asset_turnover", "profitability", "x", "revenue / avg(total_assets)")
_d("equity_multiplier", "profitability", "x", "avg(total_assets) / avg(equity)", ALL_SECTORS)
_d("revenue_growth", "profitability", "ratio", "revenue_t / revenue_t-1 - 1")
_d("net_income_growth", "profitability", "ratio", "net_income_t / net_income_t-1 - 1", ALL_SECTORS)
# financial health
_d("current_ratio", "health", "x", "current_assets / current_liabilities")
_d("quick_ratio", "health", "x", "(current_assets - inventory) / current_liabilities")
_d("debt_to_equity", "health", "x", "(short_term_debt + long_term_debt) / equity")
_d("net_debt_to_equity", "health", "x", "(short_term_debt + long_term_debt - cash) / equity")
_d("interest_coverage", "health", "x", "operating_income / interest_expense")
_d("liabilities_to_assets", "health", "ratio", "total_liabilities / total_assets", ALL_SECTORS)
# operational
_d("dso", "operational", "days", "avg(receivables) / revenue * 365")
_d("dio", "operational", "days", "avg(inventory) / cogs * 365")
_d("dpo", "operational", "days", "avg(accounts_payable) / cogs * 365")
_d("ccc", "operational", "days", "dso + dio - dpo")
_d("sga_to_revenue", "operational", "ratio", "sga / revenue")
_d("capex_to_revenue", "operational", "ratio", "-capex / revenue")
# cash flow and earnings quality
_d("cfo_to_net_income", "cashflow_quality", "x", "cfo / net_income")
_d("accruals_ratio", "cashflow_quality", "ratio", "(net_income - cfo) / avg(total_assets)")
_d("fcf", "cashflow_quality", "amount", "cfo + capex (capex is negative)")
_d("fcf_margin", "cashflow_quality", "ratio", "(cfo + capex) / revenue")
_d("dividend_fcf_cover", "cashflow_quality", "x", "(cfo + capex) / -dividends_paid")
# banks
BANK = (Sector.BANK,)
_d("nii_to_avg_loans", "profitability", "ratio", "net_interest_income / avg(loans)  (NIM proxy; earning assets not in statements)", BANK)
_d("loan_growth", "operational", "ratio", "loans_t / loans_t-1 - 1", BANK)
_d("deposit_growth", "operational", "ratio", "deposits_t / deposits_t-1 - 1", BANK)
_d("ldr", "health", "ratio", "loans / deposits", BANK)
_d("equity_to_assets", "health", "ratio", "equity / total_assets", BANK)
_d("npl_ratio", "health", "ratio", "npl / loans", BANK)
_d("reserve_coverage", "health", "ratio", "loan_loss_reserve / npl", BANK)
_d("cost_of_risk", "health", "ratio", "provision_expense / avg(loans)", BANK)


def _avg(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Average with prior period; returns (values, used_prior_flag)."""
    prior = s.shift(1)
    used = prior.notna() & s.notna()
    return s.where(~used, (s + prior) / 2), used


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b = b.where(b != 0)
    return a / b


def compute_series(st: Statements) -> dict[str, pd.Series]:
    g = st.get
    avg_eq, _ = _avg(g("equity"))
    avg_ta, _ = _avg(g("total_assets"))
    days = 365.0 if st.freq == "annual" else 91.25
    debt = g("short_term_debt").fillna(0) + g("long_term_debt").fillna(0)
    debt = debt.where(g("short_term_debt").notna() | g("long_term_debt").notna())
    fcf = g("cfo") + g("capex")
    s: dict[str, pd.Series] = {
        "gross_margin": _safe_div(g("gross_profit"), g("revenue")),
        "operating_margin": _safe_div(g("operating_income"), g("revenue")),
        "net_margin": _safe_div(g("net_income"), g("revenue")),
        "roe": _safe_div(g("net_income"), avg_eq),
        "roa": _safe_div(g("net_income"), avg_ta),
        "asset_turnover": _safe_div(g("revenue"), avg_ta),
        "equity_multiplier": _safe_div(avg_ta, avg_eq),
        "revenue_growth": _safe_div(g("revenue"), g("revenue").shift(1)) - 1,
        "net_income_growth": _safe_div(g("net_income"), g("net_income").shift(1).where(g("net_income").shift(1) > 0)) - 1,
        "current_ratio": _safe_div(g("current_assets"), g("current_liabilities")),
        "quick_ratio": _safe_div(g("current_assets") - g("inventory").fillna(0), g("current_liabilities")),
        "debt_to_equity": _safe_div(debt, g("equity")),
        "net_debt_to_equity": _safe_div(debt - g("cash"), g("equity")),
        "interest_coverage": _safe_div(g("operating_income"), g("interest_expense").abs()),
        "liabilities_to_assets": _safe_div(g("total_liabilities"), g("total_assets")),
        "dso": _safe_div(_avg(g("receivables"))[0], g("revenue")) * days,
        "dio": _safe_div(_avg(g("inventory"))[0], g("cogs").abs()) * days,
        "dpo": _safe_div(_avg(g("accounts_payable"))[0], g("cogs").abs()) * days,
        "sga_to_revenue": _safe_div(g("sga").abs(), g("revenue")),
        "capex_to_revenue": _safe_div(-g("capex"), g("revenue")),
        "cfo_to_net_income": _safe_div(g("cfo"), g("net_income").where(g("net_income") > 0)),
        "accruals_ratio": _safe_div(g("net_income") - g("cfo"), avg_ta),
        "fcf": fcf,
        "fcf_margin": _safe_div(fcf, g("revenue")),
        "dividend_fcf_cover": _safe_div(fcf, -g("dividends_paid").where(g("dividends_paid") < 0)),
        "nii_to_avg_loans": _safe_div(g("net_interest_income"), _avg(g("loans"))[0]),
        "loan_growth": _safe_div(g("loans"), g("loans").shift(1)) - 1,
        "deposit_growth": _safe_div(g("deposits"), g("deposits").shift(1)) - 1,
        "ldr": _safe_div(g("loans"), g("deposits")),
        "equity_to_assets": _safe_div(g("equity"), g("total_assets")),
        "npl_ratio": _safe_div(g("npl"), g("loans")),
        "reserve_coverage": _safe_div(g("loan_loss_reserve").abs(), g("npl")),
        "cost_of_risk": _safe_div(g("provision_expense").abs(), _avg(g("loans"))[0]),
    }
    s["ccc"] = s["dso"] + s["dio"] - s["dpo"]
    return s


def compute_metrics(st: Statements, lenses: list[str] | None = None) -> list[Metric]:
    """All metrics applicable to the sector (and optional lens filter) as Metric objects."""
    series = compute_series(st)
    out: list[Metric] = []
    for name, d in DEFS.items():
        if st.sector not in d.sectors:
            continue
        if lenses and d.lens not in lenses:
            continue
        unit = st.unit if d.unit == "amount" else d.unit
        for period, v in series[name].items():
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) or pd.isna(v):
                continue
            out.append(Metric(f"{st.ticker}.{name}:{period}", name, d.lens, str(period), float(v), unit, d.formula))
    return out


LINE_ITEMS_FOR_CITATION = (
    "revenue", "gross_profit", "operating_income", "net_income", "total_assets", "equity",
    "cfo", "capex", "dividends_paid", "cash", "receivables", "inventory",
    "net_interest_income", "loans", "deposits",
)


def line_item_metrics(st: Statements) -> list[Metric]:
    """Raw statement lines as citable metrics (lens 'data'), so narratives can quote levels."""
    out = []
    for item in LINE_ITEMS_FOR_CITATION:
        for period, v in st.get(item).items():
            if pd.notna(v):
                out.append(Metric(f"{st.ticker}.{item}:{period}", item, "data", str(period), float(v), st.unit, f"reported {item}"))
    return out


def metric_table(metrics: list[Metric]) -> pd.DataFrame:
    """Wide table (metric x period) of display strings, for prompts and reports."""
    if not metrics:
        return pd.DataFrame()
    df = pd.DataFrame([{"metric": m.name, "period": m.period, "display": m.display()} for m in metrics])
    return df.pivot(index="metric", columns="period", values="display").fillna("n/a")


def trend(metrics: list[Metric], name: str) -> str:
    """'up' | 'down' | 'flat' | 'n/a' comparing first and last available period."""
    vals = sorted((m.period, m.value) for m in metrics if m.name == name)
    if len(vals) < 2:
        return "n/a"
    first, last = vals[0][1], vals[-1][1]
    if first == 0:
        return "up" if last > 0 else "down" if last < 0 else "flat"
    change = (last - first) / abs(first)
    return "flat" if abs(change) < 0.02 else ("up" if change > 0 else "down")
