"""Provider interface. A provider fetches raw statements and normalizes them."""

from __future__ import annotations

from abc import ABC, abstractmethod

from fin_agent.data.schema import MarketData, Statements


class ProviderError(Exception):
    """The source could not return usable data (network, unknown ticker, schema change)."""


class StatementProvider(ABC):
    name: str = "base"
    markets: tuple[str, ...] = ()

    @abstractmethod
    def fetch(self, ticker: str, years: int, freq: str = "annual") -> Statements:
        """Return normalized statements, most recent ``years`` periods, ascending."""

    def fetch_market_data(self, ticker: str) -> MarketData | None:
        """Latest price and share count, or None when the provider has no market data.
        Raise ``ProviderError`` on a failed lookup; callers treat it as non-fatal."""
        return None

    def supports(self, market: str) -> bool:
        return market.upper() in self.markets
