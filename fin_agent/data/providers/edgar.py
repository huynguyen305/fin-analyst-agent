"""SEC EDGAR XBRL provider (US firms, fallback market).

Uses the free ``companyfacts`` API. SEC asks every client to send a descriptive
User-Agent with contact details; set ``SEC_USER_AGENT`` in ``.env``.
Annual values are taken from facts that carry a calendar ``frame``
(``CY2023`` for flows, ``CY2023Q4I`` for balances), which de-duplicates the
many restated copies of the same number across filings.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from fin_agent.data.providers.base import ProviderError, StatementProvider
from fin_agent.data.schema import Sector, Statements

# canonical item -> US-GAAP tags in priority order
TAG_MAP: dict[str, list[str]] = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "operating_income": ["OperatingIncomeLoss"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating"],
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "net_income": ["NetIncomeLoss"],
    "depreciation": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization"],
    "total_assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "receivables": ["AccountsReceivableNetCurrent"],
    "inventory": ["InventoryNet"],
    "total_liabilities": ["Liabilities"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "accounts_payable": ["AccountsPayableCurrent"],
    "short_term_debt": ["LongTermDebtCurrent", "ShortTermBorrowings", "DebtCurrent"],
    "long_term_debt": ["LongTermDebtNoncurrent"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
}
OUTFLOW_ITEMS = {"capex", "dividends_paid"}  # SEC reports these as positive payments


def parse_companyfacts(facts: dict, years: int) -> pd.DataFrame:
    """Turn a companyfacts JSON document into annual canonical columns (USD millions)."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    columns: dict[str, dict[int, tuple[str, float]]] = {}

    for item, tags in TAG_MAP.items():
        by_year: dict[int, tuple[str, float]] = {}
        for tag in tags:
            entries = gaap.get(tag, {}).get("units", {}).get("USD", [])
            for e in entries:
                frame = e.get("frame", "")
                if not frame.startswith("CY"):
                    continue
                is_annual_flow = len(frame) == 6  # CY2023
                is_year_end_balance = frame.endswith("Q4I")
                if not (is_annual_flow or is_year_end_balance):
                    continue
                year = int(frame[2:6])
                if year not in by_year:  # first tag in priority order wins
                    by_year[year] = (e["end"], float(e["val"]))
        if by_year:
            columns[item] = by_year

    if not columns:
        return pd.DataFrame()

    all_years = sorted({y for col in columns.values() for y in col})[-years:]
    rows = {}
    for y in all_years:
        ends = [col[y][0] for col in columns.values() if y in col]
        period_end = pd.Timestamp(max(ends)).date()
        row = {}
        for item, col in columns.items():
            if y in col:
                v = col[y][1] / 1e6
                row[item] = -abs(v) if item in OUTFLOW_ITEMS else v
        rows[period_end] = row
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "period_end"

    if "gross_profit" not in df and {"revenue", "cogs"} <= set(df.columns):
        df["gross_profit"] = df["revenue"] - df["cogs"]
    if "total_liabilities" not in df and {"total_assets", "equity"} <= set(df.columns):
        # Some filers omit the Liabilities total; derive it and say so in notes.
        df["total_liabilities"] = df["total_assets"] - df["equity"]
    return df


class EdgarProvider(StatementProvider):
    name = "sec_edgar"
    markets = ("US",)
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

    def __init__(self, user_agent: str | None = None, timeout: int = 30):
        self.user_agent = user_agent or os.getenv("SEC_USER_AGENT", "")
        self.timeout = timeout
        if not self.user_agent:
            raise ProviderError("Set SEC_USER_AGENT (e.g. 'fin-analyst-agent you@example.com') before using EDGAR")

    def _get_json(self, url: str) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except Exception as exc:  # network, 404, rate limit
            raise ProviderError(f"EDGAR request failed for {url}: {exc}") from exc

    def cik_for(self, ticker: str) -> int:
        for row in self._get_json(self.TICKERS_URL).values():
            if row["ticker"].upper() == ticker.upper():
                return int(row["cik_str"])
        raise ProviderError(f"Ticker {ticker} not found in SEC company list")

    def fetch(self, ticker: str, years: int, freq: str = "annual") -> Statements:
        if freq != "annual":
            raise ProviderError("EDGAR provider currently supports annual data only")
        facts = self._get_json(self.FACTS_URL.format(cik=self.cik_for(ticker)))
        df = parse_companyfacts(facts, years)
        if df.empty:
            raise ProviderError(f"No US-GAAP facts for {ticker}")
        return Statements(
            ticker=ticker.upper(), market="US", currency="USD", unit="mn", sector=Sector.GENERAL,
            freq="annual", data=df, source=self.name,
            fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
