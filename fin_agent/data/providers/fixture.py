"""Offline provider reading JSON fixtures. Used by tests, evals and demos.

Fixture format::

    {"ticker": "DEMO", "market": "HOSE", "currency": "VND", "unit": "bn",
     "sector": "general", "freq": "annual",
     "periods": {"2021-12-31": {"revenue": 100.0, ...}, ...}}
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fin_agent.data.providers.base import ProviderError, StatementProvider
from fin_agent.data.schema import Sector, Statements


class FixtureProvider(StatementProvider):
    name = "fixture"
    markets = ("HOSE", "HNX", "UPCOM", "US")

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def fetch(self, ticker: str, years: int, freq: str = "annual") -> Statements:
        path = self.directory / f"{ticker.upper()}.json"
        if not path.exists():
            raise ProviderError(f"No fixture for {ticker} in {self.directory}")
        doc = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame.from_dict(doc["periods"], orient="index")
        df.index = pd.to_datetime(df.index).date
        df.index.name = "period_end"
        df = df.sort_index().tail(years)
        return Statements(
            ticker=doc["ticker"], market=doc["market"], currency=doc["currency"], unit=doc["unit"],
            sector=Sector(doc.get("sector", "general")), freq=doc.get("freq", "annual"),
            data=df, source=f"fixture:{path.name}", fetched_at=doc.get("fetched_at", "fixture"),
            notes=[doc["note"]] if doc.get("note") else [],
        )
