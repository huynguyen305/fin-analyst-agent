"""Provider registry: pick a source by market, falling back in order."""

from __future__ import annotations

import os

from fin_agent.data.providers.base import ProviderError, StatementProvider
from fin_agent.data.schema import Statements


def build_providers(market: str) -> list[StatementProvider]:
    """Providers to try for a market, in priority order (ADR-007).

    ``FIN_AGENT_FIXTURES`` set to a directory forces offline fixtures first,
    which is how tests and evals run without network access.
    """
    providers: list[StatementProvider] = []
    fixtures = os.getenv("FIN_AGENT_FIXTURES")
    if fixtures:
        from fin_agent.data.providers.fixture import FixtureProvider
        providers.append(FixtureProvider(fixtures))
    market = market.upper()
    if market in ("HOSE", "HNX", "UPCOM"):
        try:
            from fin_agent.data.providers.vnstock_provider import VnstockProvider
            providers.append(VnstockProvider(os.getenv("VNSTOCK_SOURCE", "VCI")))
        except ProviderError:
            pass
    if market == "US":
        try:
            from fin_agent.data.providers.edgar import EdgarProvider
            providers.append(EdgarProvider())
        except ProviderError:
            pass
    return providers


def fetch_statements(ticker: str, market: str, years: int, freq: str = "annual") -> Statements:
    errors = []
    for p in build_providers(market):
        try:
            return p.fetch(ticker, years, freq)
        except ProviderError as exc:
            errors.append(f"{p.name}: {exc}")
    raise ProviderError(f"No provider could supply {ticker} ({market}). Tried: " + (" | ".join(errors) or "none configured"))


__all__ = ["ProviderError", "StatementProvider", "build_providers", "fetch_statements"]
