"""Vietnam provider (HOSE / HNX / UPCoM) via the vnstock library.

vnstock wraps broker data sources (VCI, TCBS). Its column names change between
versions and sources, so mapping is done with regex patterns rather than exact
names, and ``fetch`` records which source column fed each canonical item in
``Statements.notes``. **Verify the mapping on the first real run**; adjust
``COLUMN_PATTERNS`` if a line is missed.

Install: ``pip install vnstock`` (or ``vnstock3`` for the older 3.x line).
As of 2026-11-04 ``vnstock`` returned 404 on PyPI from our build machine while
``vnstock3`` 3.2.1 was available; check the project's README for the current
install route.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone

import pandas as pd

from fin_agent.data.providers.base import ProviderError, StatementProvider
from fin_agent.data.schema import Sector, Statements

# canonical item -> regex patterns tried in order against lower-cased column names
COLUMN_PATTERNS: dict[str, list[str]] = {
    # income statement
    "revenue": [r"^net (sales|revenue)", r"^revenue \(bn", r"^revenue$", r"^net revenue"],
    "cogs": [r"^cost of (goods sold|sales)"],
    "gross_profit": [r"^gross profit"],
    "sga": [r"general (&|and) admin"],
    "operating_income": [r"^operating profit", r"^net operating profit"],
    "interest_expense": [r"^interest expenses?$", r"of which.*interest expense"],
    "pretax_income": [r"^profit before tax", r"accounting profit before tax"],
    "net_income": [r"attributable to parent", r"^net profit for the year", r"^net profit after tax", r"^net profit"],
    # balance sheet
    "total_assets": [r"^total assets", r"^total resources"],
    "current_assets": [r"^current assets", r"^short-term assets"],
    "cash": [r"^cash and cash equivalents"],
    "receivables": [r"^(short-term )?accounts receivable", r"^short-term receivables"],
    "inventory": [r"^(net )?inventor(y|ies)"],
    "total_liabilities": [r"^liabilities \(bn", r"^total liabilities", r"^liabilities$"],
    "current_liabilities": [r"^current liabilities", r"^short-term liabilities"],
    "accounts_payable": [r"^(short-term )?(trade )?accounts payable", r"^short-term trade payables"],
    "short_term_debt": [r"^short-term borrowings"],
    "long_term_debt": [r"^long-term borrowings"],
    "equity": [r"^owner'?s'? equity", r"^total equity", r"^equity \(bn"],
    # cash flow
    "cfo": [r"net cash (inflows/outflows|flows?) from operating"],
    "capex": [r"^purchase of fixed assets", r"purchases? of (fixed|long-term) assets"],
    "dividends_paid": [r"^dividends paid"],
    "depreciation": [r"^depreciation"],
    # banks
    "interest_income": [r"^interest and similar income"],
    "net_interest_income": [r"^net interest income"],
    "loans": [r"^loans and advances to customers, net", r"^loans and advances to customers"],
    "deposits": [r"^deposits from customers"],
    "loan_loss_reserve": [r"provision for (credit|loan) losses on loans", r"^less: provision"],
    "provision_expense": [r"^provision for credit losses$", r"^provision expenses? for credit"],
}

BANK_HINT_COLUMNS = ("net interest income", "deposits from customers")


def _norm(col: str) -> str:
    return re.sub(r"\s+", " ", str(col)).strip().lower()


def map_columns(df: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    """Return {canonical_item: source_column} and human-readable notes."""
    cols = {_norm(c): c for c in df.columns}
    mapping: dict[str, str] = {}
    for item, patterns in COLUMN_PATTERNS.items():
        for pat in patterns:
            hit = next((orig for n, orig in cols.items() if re.search(pat, n)), None)
            if hit is not None and hit not in mapping.values():
                mapping[item] = hit
                break
    notes = [f"{item} <- '{col}'" for item, col in mapping.items()]
    return mapping, notes


def normalize_frames(frames: list[pd.DataFrame], years: int, freq: str) -> tuple[pd.DataFrame, list[str], Sector]:
    """Merge income/balance/cash-flow frames from vnstock into canonical columns (VND bn)."""
    merged: dict[date, dict[str, float]] = {}
    notes: list[str] = []
    sector = Sector.GENERAL
    for df in frames:
        if df is None or df.empty:
            continue
        if any(h in _norm(c) for c in df.columns for h in BANK_HINT_COLUMNS):
            sector = Sector.BANK
        year_col = next((c for c in df.columns if _norm(c) in ("yearreport", "year")), None)
        q_col = next((c for c in df.columns if _norm(c) in ("lengthreport", "quarter")), None)
        if year_col is None:
            raise ProviderError(f"No year column in vnstock frame; columns: {list(df.columns)[:8]}")
        mapping, n = map_columns(df)
        notes += n
        for _, row in df.iterrows():
            y = int(row[year_col])
            if freq == "quarterly" and q_col is not None and int(row[q_col]) in (1, 2, 3, 4):
                q = int(row[q_col])
                end = date(y, 3 * q, 31 if q in (1, 4) else 30)
            else:
                end = date(y, 12, 31)
            rec = merged.setdefault(end, {})
            for item, col in mapping.items():
                v = pd.to_numeric(row[col], errors="coerce")
                if pd.notna(v):
                    rec[item] = float(v)

    if not merged:
        return pd.DataFrame(), notes, sector
    out = pd.DataFrame.from_dict(merged, orient="index").sort_index().tail(years if freq == "annual" else years * 4)
    out.index.name = "period_end"

    # Units: some sources label columns "Bn. VND" but return raw VND. Detect and rescale.
    if "total_assets" in out and out["total_assets"].median() > 1e9:
        out = out / 1e9
        notes.append("values rescaled from VND to VND bn")

    for item in ("capex", "dividends_paid"):
        if item in out:
            out[item] = -out[item].abs()
    if "gross_profit" not in out and {"revenue", "cogs"} <= set(out.columns):
        out["gross_profit"] = out["revenue"] - out["cogs"].abs()
    if "cogs" in out:
        out["cogs"] = out["cogs"].abs()
    return out, notes, sector


class VnstockProvider(StatementProvider):
    name = "vnstock"
    markets = ("HOSE", "HNX", "UPCOM")

    def __init__(self, source: str = "VCI"):
        self.source = source
        try:
            from vnstock import Vnstock  # type: ignore
        except ImportError:
            try:
                from vnstock3 import Vnstock  # type: ignore
            except ImportError as exc:
                raise ProviderError("vnstock is not installed: pip install vnstock (or vnstock3)") from exc
        self._client = Vnstock()

    def fetch(self, ticker: str, years: int, freq: str = "annual") -> Statements:
        period = "year" if freq == "annual" else "quarter"
        try:
            stock = self._client.stock(symbol=ticker.upper(), source=self.source)
            frames = [
                stock.finance.income_statement(period=period, lang="en"),
                stock.finance.balance_sheet(period=period, lang="en"),
                stock.finance.cash_flow(period=period, lang="en"),
            ]
        except Exception as exc:
            raise ProviderError(f"vnstock ({self.source}) failed for {ticker}: {exc}") from exc
        df, notes, sector = normalize_frames(frames, years, freq)
        if df.empty:
            raise ProviderError(f"vnstock returned no statements for {ticker}")
        return Statements(
            ticker=ticker.upper(), market="HOSE", currency="VND", unit="bn", sector=sector,
            freq=freq, data=df, source=f"vnstock:{self.source}",
            fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), notes=notes,
        )
