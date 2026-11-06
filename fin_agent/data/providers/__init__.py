"""Provider registry: pick a source by market, falling back in order."""

from __future__ import annotations

import os

from fin_agent.data.providers.base import ProviderError, StatementProvider
from fin_agent.data.schema import Statements


def build_providers(market: str, setup_errors: list[str] | None = None) -> list[StatementProvider]:
    """Providers to try for a market, in priority order (ADR-007).

    ``FIN_AGENT_FIXTURES`` set to a directory forces offline fixtures first,
    which is how tests and evals run without network access. Providers that
    cannot start (library missing, bad config) are skipped and the reason is
    appended to ``setup_errors`` so the user sees why.
    """
    setup_errors = setup_errors if setup_errors is not None else []
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
        except ProviderError as exc:
            setup_errors.append(f"vnstock not available: {exc}")
    if market == "US":
        try:
            from fin_agent.data.providers.edgar import EdgarProvider
            providers.append(EdgarProvider())
        except ProviderError as exc:
            setup_errors.append(f"sec_edgar not available: {exc}")
    return providers


def fetch_statements(ticker: str, market: str, years: int, freq: str = "annual") -> Statements:
    errors: list[str] = []
    for p in build_providers(market, errors):
        try:
            st = p.fetch(ticker, years, freq)
        except ProviderError as exc:
            errors.append(f"{p.name}: {exc}")
            continue
        if st.market_data is None:
            try:
                st.market_data = p.fetch_market_data(ticker)
            except ProviderError as exc:  # prices are optional: valuation is then not assessed
                st.notes.append(f"market data unavailable: {exc}")
        return st
    raise ProviderError(f"No provider could supply {ticker} ({market}). Tried: " + (" | ".join(errors) or "no provider for this market"))


__all__ = ["ProviderError", "StatementProvider", "build_providers", "fetch_statements"]
