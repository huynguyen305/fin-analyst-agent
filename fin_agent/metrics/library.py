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
from fin_agent.metrics.scenarios import scenario_values
from fin_agent.metrics.technical import indicators


@dataclass(frozen=True)
class Metric:
    id: str
    name: str
    lens: str
    period: str
    value: float
    unit: str  # "ratio" | "days" | "x" | "pts" (index points, e.g. RSI) | currency unit
    formula: str

    def display(self) -> str:
        if self.unit == "ratio":
            return f"{self.value * 100:.1f}%"
        if self.unit == "days":
            return f"{self.value:.0f} days"
        if self.unit == "x":
            return f"{self.value:.2f}x"
        if self.unit == "pts":
            return f"{self.value:.1f}"
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


LENSES = ("profitability", "health", "operational", "cashflow_quality", "valuation", "technical", "scenarios",
          "risk_governance")

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
# valuation: one value at the price date (period = price date). Earnings = latest fiscal year,
# or the sum of the last four quarters (TTM) for quarterly data; balances = latest period.
_d("share_price", "valuation", "price", "last close per share", ALL_SECTORS)
_d("market_cap", "valuation", "amount", "share_price * shares_outstanding", ALL_SECTORS)
_d("pe", "valuation", "x", "market_cap / net_income (latest FY or TTM; positive earnings only)", ALL_SECTORS)
_d("pb", "valuation", "x", "market_cap / equity (latest period)", ALL_SECTORS)
_d("earnings_yield", "valuation", "ratio", "net_income (latest FY or TTM) / market_cap", ALL_SECTORS)
_d("dividend_yield", "valuation", "ratio", "-dividends_paid (latest FY or TTM) / market_cap", ALL_SECTORS)
_d("ps", "valuation", "x", "market_cap / revenue (latest FY or TTM)")
_d("ev_to_ebit", "valuation", "x", "(market_cap + debt - cash) / operating_income (latest FY or TTM)")
_d("fcf_yield", "valuation", "ratio", "(cfo + capex) (latest FY or TTM) / market_cap")
# technical: daily closes, one value at the price date; windows in trading days (fin_agent.metrics.technical)
_t = lambda name, unit, formula: _d(name, "technical", unit, formula, ALL_SECTORS)  # noqa: E731
_t("sma_50", "price", "mean of last 50 closes")
_t("sma_200", "price", "mean of last 200 closes")
_t("price_vs_sma50", "ratio", "close / sma_50 - 1")
_t("price_vs_sma200", "ratio", "close / sma_200 - 1 (long-term trend)")
_t("sma50_vs_sma200", "ratio", "sma_50 / sma_200 - 1 (> 0: golden-cross regime, < 0: death-cross regime)")
_t("rsi_14", "pts", "Wilder RSI, 14 days (> 70 overbought, < 30 oversold)")
_t("macd_hist_pct", "ratio", "(MACD 12/26 - signal 9) / close (> 0: positive momentum)")
_t("return_1m", "ratio", "close / close 21 trading days ago - 1")
_t("return_3m", "ratio", "close / close 63 trading days ago - 1")
_t("return_6m", "ratio", "close / close 126 trading days ago - 1")
_t("return_12m", "ratio", "close / close 252 trading days ago - 1")
_t("volatility_60d", "ratio", "stdev of daily log returns over 60 days * sqrt(252)")
_t("max_drawdown_12m", "ratio", "worst peak-to-trough fall of closes over 252 days")
_t("pct_from_52w_high", "ratio", "close / highest close of 252 days - 1")
_t("pct_from_52w_low", "ratio", "close / lowest close of 252 days - 1")
_t("volume_ratio_20_60", "x", "average volume 20 days / average volume 60 days")
_t("rel_return_3m", "ratio", "return_3m - VN-Index return over the same days")
_t("rel_return_6m", "ratio", "return_6m - VN-Index return over the same days")
_t("rel_return_12m", "ratio", "return_12m - VN-Index return over the same days (index precision about 0.5%)")
# price scenarios (fin_agent.metrics.scenarios): ranges, trend and 12-month bear/base/bull, at the price date.
# Conditional on stated assumptions; none is a forecast of the most likely price.
_s = lambda name, unit, formula: _d(name, "scenarios", unit, formula, ALL_SECTORS)  # noqa: E731
_s("range_volatility", "ratio", "annualized stdev of daily log returns over up to 252 days (used for the ranges)")
for _h, _days in (("3m", 63), ("12m", 252)):
    _s(f"range_{_h}_low", "price", f"10th percentile price in {_days} trading days: close * exp(-1.2816 * range_volatility * sqrt({_days}/252)); zero drift, past volatility persists")
    _s(f"range_{_h}_high", "price", f"90th percentile price in {_days} trading days: close * exp(+1.2816 * range_volatility * sqrt({_days}/252)); zero drift, past volatility persists")
_s("trend_annual", "ratio", "annualized slope of a log-linear fit to up to 252 daily closes")
_s("trend_r2", "ratio", "R-squared of that fit (above about 60%: a clear trend; below about 30%: no trend)")
_s("trend_3m_price", "price", "close * exp(daily trend slope * 63): where the price would be if the trend simply continued")
_s("trend_12m_price", "price", "close * exp(daily trend slope * 252): where the price would be if the trend simply continued")
_REF = {"bear": "12-month low close", "base": "last close", "bull": "12-month high close"}
_PCT = {"bear": "25th", "base": "50th", "bull": "75th"}
for _c in ("bear", "base", "bull"):
    _s(f"scenario_{_c}_price", "price", f"{_c} 12-month scenario: {_REF[_c]} * (1 + {_c} driver growth); driver = earnings, or book value for banks and loss-makers; constant share count")
    _s(f"scenario_{_c}_return", "ratio", f"scenario_{_c}_price / last close - 1")
    _s(f"scenario_{_c}_earnings_growth", "ratio", f"{_PCT[_c]} percentile of annual net income growth (positive prior years only), capped at +/-50%")
    _s(f"scenario_{_c}_book_growth", "ratio", f"{_PCT[_c]} percentile of annual equity (book value) growth, capped at +/-50%")
    _s(f"scenario_{_c}_pe", "x", f"P/E implied by the {_REF[_c]} with the latest earnings: pe * reference close / last close")
    _s(f"scenario_{_c}_pb", "x", f"P/B implied by the {_REF[_c]} with the latest book value: pb * reference close / last close")


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
    s.update(valuation_series(st))
    s.update(technical_series(st))
    s.update(scenario_series(st, s))
    return s


def technical_series(st: Statements) -> dict[str, pd.Series]:
    """Technical indicators as one-point series at the last bar's date (empty without history)."""
    names = [n for n, d in DEFS.items() if d.lens == "technical"]
    md = st.market_data
    values = indicators(md.history, md.benchmark) if md and md.history else {}
    if not values:
        return {n: pd.Series(dtype=float) for n in names}
    idx = [md.history[-1]["date"]]
    return {n: pd.Series([values.get(n, float("nan"))], index=idx, dtype=float) for n in names}



def scenario_series(st: Statements, s: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Scenario-lens values as one-point series at the last bar's date (empty without price history).

    The driver is earnings when the company is not a bank and its latest earnings are positive,
    otherwise book value (equity); ``s`` provides today's P/E or P/B for the implied multiples.
    """
    names = [n for n, d in DEFS.items() if d.lens == "scenarios"]
    md = st.market_data
    if not md or not md.history or st.data.empty:
        return {n: pd.Series(dtype=float) for n in names}
    ni = st.get("net_income")
    earnings = st.sector != Sector.BANK and pd.notna(ni.iloc[-1]) and ni.iloc[-1] > 0
    driver, levels = ("earnings", ni) if earnings else ("book", st.get("equity"))
    mult = s.get("pe" if earnings else "pb")
    multiple_now = float(mult.iloc[-1]) if mult is not None and len(mult) else float("nan")
    values = scenario_values(md.history, driver=driver, driver_levels=levels, multiple_now=multiple_now, freq=st.freq)
    if not values:
        return {n: pd.Series(dtype=float) for n in names}
    idx = [md.history[-1]["date"]]
    return {n: pd.Series([values.get(n, float("nan"))], index=idx, dtype=float) for n in names}


UNIT_SCALE = {"bn": 1e9, "mn": 1e6}


def valuation_series(st: Statements) -> dict[str, pd.Series]:
    """Valuation inputs as one-point series indexed by the price date (empty without market data).

    Flows (earnings, revenue, cash flows) use the latest fiscal year, or the trailing four
    quarters for quarterly data (no figure when fewer than four quarters are available).
    """
    names = [n for n, d in DEFS.items() if d.lens == "valuation"]
    md = st.market_data
    if md is None or st.data.empty or not md.price or not md.shares_outstanding:
        return {n: pd.Series(dtype=float) for n in names}
    g, last = st.get, st.data.index[-1]

    def flow(item: str) -> float:
        s = g(item)
        if st.freq == "annual":
            return s.iloc[-1]
        tail = s.tail(4)
        return tail.sum() if len(tail) == 4 and tail.notna().all() else float("nan")

    def pos(v: float) -> float:
        return v if pd.notna(v) and v > 0 else float("nan")

    def div(a: float, b: float) -> float:
        return a / b if pd.notna(a) and pd.notna(b) and b != 0 else float("nan")

    mcap = md.price * md.shares_outstanding / UNIT_SCALE.get(st.unit, 1.0)
    ni, rev, ebit = flow("net_income"), flow("revenue"), flow("operating_income")
    std, ltd, cash = g("short_term_debt")[last], g("long_term_debt")[last], g("cash")[last]
    debt = (0 if pd.isna(std) else std) + (0 if pd.isna(ltd) else ltd) if pd.notna(std) or pd.notna(ltd) else float("nan")
    ev = mcap + debt - cash if pd.notna(debt) and pd.notna(cash) else float("nan")
    divs = flow("dividends_paid")
    values = {
        "share_price": md.price,
        "market_cap": mcap,
        "pe": div(mcap, pos(ni)),
        "pb": div(mcap, pos(g("equity")[last])),
        "earnings_yield": div(ni, mcap),
        "dividend_yield": div(-divs, mcap) if pd.notna(divs) and divs <= 0 else float("nan"),
        "ps": div(mcap, pos(rev)),
        "ev_to_ebit": div(ev, pos(ebit)),
        "fcf_yield": div(flow("cfo") + flow("capex"), mcap),
    }
    idx = [md.price_date or str(last)]
    return {n: pd.Series([values[n]], index=idx, dtype=float) for n in names}


def compute_metrics(st: Statements, lenses: list[str] | None = None) -> list[Metric]:
    """All metrics applicable to the sector (and optional lens filter) as Metric objects."""
    series = compute_series(st)
    out: list[Metric] = []
    for name, d in DEFS.items():
        if st.sector not in d.sectors:
            continue
        if lenses and d.lens not in lenses:
            continue
        unit = st.unit if d.unit == "amount" else st.currency if d.unit == "price" else d.unit
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
