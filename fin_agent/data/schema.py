"""Canonical financial statement schema and data-quality gate.

Every provider (vnstock, SEC EDGAR, fixtures) normalizes into ``Statements``:
one tidy table of ``(period_end, line_item) -> value`` for one ticker. VND values
are stored in billions; USD values in millions. Downstream code never sees
provider-specific column names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import pandas as pd


class Sector(str, Enum):
    GENERAL = "general"  # non-financial corporates
    BANK = "bank"
    INSURANCE = "insurance"
    SECURITIES = "securities"


# Canonical line items. Providers map into these names; anything else is dropped.
INCOME_ITEMS = [
    "revenue", "cogs", "gross_profit", "sga", "operating_income",
    "interest_expense", "pretax_income", "net_income", "depreciation",
]
BALANCE_ITEMS = [
    "total_assets", "current_assets", "cash", "receivables", "inventory",
    "total_liabilities", "current_liabilities", "accounts_payable",
    "short_term_debt", "long_term_debt", "equity",
]
CASHFLOW_ITEMS = ["cfo", "capex", "dividends_paid"]
BANK_ITEMS = [
    "interest_income", "net_interest_income", "loans", "deposits",
    "npl", "loan_loss_reserve", "provision_expense",
]
ALL_ITEMS = INCOME_ITEMS + BALANCE_ITEMS + CASHFLOW_ITEMS + BANK_ITEMS

CORE_ITEMS = {
    Sector.GENERAL: ["revenue", "net_income", "total_assets", "total_liabilities", "equity", "cfo"],
    Sector.BANK: ["net_interest_income", "net_income", "total_assets", "total_liabilities", "equity", "loans", "deposits"],
    Sector.INSURANCE: ["revenue", "net_income", "total_assets", "equity"],
    Sector.SECURITIES: ["revenue", "net_income", "total_assets", "equity"],
}


@dataclass
class Statements:
    """Normalized statements for one ticker.

    ``data`` is a DataFrame indexed by ``period_end`` (ascending) with one column
    per canonical line item. Missing items are NaN, never zero.
    """

    ticker: str
    market: str  # HOSE | HNX | UPCOM | US
    currency: str  # VND | USD
    unit: str  # "bn" for VND, "mn" for USD
    sector: Sector
    freq: str  # "annual" | "quarterly"
    data: pd.DataFrame
    source: str
    fetched_at: str
    notes: list[str] = field(default_factory=list)

    @property
    def periods(self) -> list[date]:
        return list(self.data.index)

    def get(self, item: str) -> pd.Series:
        if item in self.data.columns:
            return self.data[item]
        return pd.Series([float("nan")] * len(self.data), index=self.data.index, name=item)

    def last_n(self, n: int) -> "Statements":
        return Statements(**{**self.__dict__, "data": self.data.tail(n)})

    def to_records(self) -> list[dict]:
        """Long format for snapshots: one row per (period, item)."""
        out = []
        for period, row in self.data.iterrows():
            for item, value in row.items():
                if pd.notna(value):
                    out.append({
                        "ticker": self.ticker, "period_end": str(period), "freq": self.freq,
                        "line_item": item, "value": float(value), "unit": self.unit,
                        "currency": self.currency, "source": self.source,
                    })
        return out


@dataclass
class DQIssue:
    severity: str  # "error" stops the run, "warning" is reported
    period: str
    check: str
    message: str


class DataQualityError(Exception):
    def __init__(self, issues: list[DQIssue]):
        self.issues = issues
        super().__init__("; ".join(f"[{i.period}] {i.check}: {i.message}" for i in issues))


def run_dq_checks(st: Statements, balance_tolerance: float = 0.005) -> list[DQIssue]:
    """Data-quality gate (Guardrails layer 2). Returns all issues found.

    Checks: data present, core lines present, balance identity
    (assets = liabilities + equity within tolerance), sign conventions.
    """
    issues: list[DQIssue] = []
    if st.data.empty:
        return [DQIssue("error", "-", "empty", f"No statements returned for {st.ticker} from {st.source}")]

    for period, row in st.data.iterrows():
        p = str(period)
        missing = [c for c in CORE_ITEMS[st.sector] if c not in row or pd.isna(row.get(c))]
        if missing:
            issues.append(DQIssue("error", p, "core_lines", f"Missing core lines: {', '.join(missing)}"))

        a, l, e = row.get("total_assets"), row.get("total_liabilities"), row.get("equity")
        if all(pd.notna(x) for x in (a, l, e)) and a:
            gap = abs(a - (l + e)) / abs(a)
            if gap > balance_tolerance:
                issues.append(DQIssue(
                    "error", p, "balance_identity",
                    f"Assets {a:,.1f} vs liabilities+equity {l + e:,.1f} (gap {gap:.2%})",
                ))

        for item in ("total_assets", "revenue", "cash", "inventory", "loans", "deposits"):
            v = row.get(item)
            if pd.notna(v) and v < 0:
                issues.append(DQIssue("error", p, "sign", f"{item} is negative ({v:,.1f})"))
        for item in ("capex", "dividends_paid"):
            v = row.get(item)
            if pd.notna(v) and v > 0:
                issues.append(DQIssue("warning", p, "sign", f"{item} should be an outflow (negative); got {v:,.1f}"))

    return issues


def assert_quality(st: Statements) -> list[DQIssue]:
    """Raise on errors, return warnings."""
    issues = run_dq_checks(st)
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise DataQualityError(errors)
    return issues
