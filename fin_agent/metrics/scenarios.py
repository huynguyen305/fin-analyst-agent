"""Price scenarios and trend ranges (pure functions, ADR-004: Python counts).

Three views of where the price could be, each with its assumptions in the formula, never a
single target:

1. **Statistical range** (``range_*``): the 10th-90th percentile band of a lognormal price
   with zero drift and the realized volatility of up to 252 daily closes. It answers "how
   far does this stock usually move in 3 or 12 months", not "where will it go".
2. **Trend** (``trend_*``): a log-linear fit of up to 252 closes. ``trend_r2`` says how
   well a straight trend describes the year; ``trend_3m_price`` / ``trend_12m_price`` are
   where the price would be *if that trend simply continued* from today's close.
3. **12-month scenarios** (``scenario_*``): price = reference close × (1 + driver growth).
   The driver is earnings (per-share earnings, assuming a constant share count) or, for
   banks and loss-makers, book value. Growth rates are the 25th / 50th / 75th percentiles
   of the company's own annual growth history, capped at ±50%. Bear starts from the
   12-month low close (the market's least generous multiple of the year), base from the
   last close (today's multiple), bull from the 12-month high close (its most generous).
   So each scenario = "the year's multiple range × the company's own growth range".

By construction bear ≤ base ≤ bull. Every function returns ``{}`` or NaN when the history
is too short; nothing is guessed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252
Z90 = 1.2815515655446004  # standard normal 90th percentile: the band covers 80% of outcomes
HORIZONS = {"3m": 63, "12m": 252}
MIN_BARS_RANGE = 60
MIN_BARS_TREND = 120
GROWTH_CAP = 0.5
MIN_GROWTH_OBS = 2


def _closes(history: list[dict]) -> pd.Series:
    df = pd.DataFrame(history)
    df = df.assign(date=pd.to_datetime(df["date"])).sort_values("date").drop_duplicates("date").set_index("date")
    return df["close"].astype(float).dropna()


def price_range(close: pd.Series) -> dict[str, float]:
    """80% lognormal bands at 3 and 12 months, zero drift, realized volatility of up to 252 days."""
    if len(close) < MIN_BARS_RANGE + 1:
        return {}
    r = np.log(close.iloc[-TRADING_DAYS - 1:]).diff().dropna()
    sigma = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))
    last = float(close.iloc[-1])
    out = {"range_volatility": sigma}
    for k, h in HORIZONS.items():
        width = Z90 * sigma * math.sqrt(h / TRADING_DAYS)
        out[f"range_{k}_low"] = last * math.exp(-width)
        out[f"range_{k}_high"] = last * math.exp(width)
    return out


def trend(close: pd.Series) -> dict[str, float]:
    """Log-linear trend of up to 252 closes: annualized slope, R², and its continuation from the last close."""
    w = close.iloc[-TRADING_DAYS:]
    if len(w) < MIN_BARS_TREND:
        return {}
    y = np.log(w.to_numpy())
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res, ss_tot = float(((y - fitted) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    last = float(w.iloc[-1])
    return {
        "trend_annual": math.exp(slope * TRADING_DAYS) - 1,
        "trend_r2": 1 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "trend_3m_price": last * math.exp(slope * HORIZONS["3m"]),
        "trend_12m_price": last * math.exp(slope * HORIZONS["12m"]),
    }


def growth_percentiles(levels: pd.Series, freq: str = "annual") -> tuple[float, float, float] | None:
    """25th / 50th / 75th percentile of period-on-period growth (year-on-year for quarterly data),
    from positive prior levels only, each capped at ±50%. None with fewer than 2 observations."""
    s = levels.dropna()
    lag = 4 if freq == "quarterly" else 1
    prev = s.shift(lag)
    g = (s / prev - 1)[prev > 0].dropna()
    if len(g) < MIN_GROWTH_OBS:
        return None
    p25, p50, p75 = (float(np.clip(np.percentile(g, q), -GROWTH_CAP, GROWTH_CAP)) for q in (25, 50, 75))
    return p25, p50, p75


def scenarios(close: pd.Series, driver: str, driver_levels: pd.Series, multiple_now: float,
              freq: str = "annual") -> dict[str, float]:
    """12-month bear / base / bull prices. ``driver`` is "earnings" or "book"; ``multiple_now`` is
    today's P/E or P/B (the implied scenario multiples are read off the 12-month low and high)."""
    w = close.iloc[-TRADING_DAYS:]
    pct = growth_percentiles(driver_levels, freq)
    if len(w) < MIN_BARS_TREND or pct is None:
        return {}
    last, low, high = float(w.iloc[-1]), float(w.min()), float(w.max())
    tag = "earnings" if driver == "earnings" else "book"
    mult = "pe" if driver == "earnings" else "pb"
    out: dict[str, float] = {}
    for case, ref, g in (("bear", low, pct[0]), ("base", last, pct[1]), ("bull", high, pct[2])):
        price = ref * (1 + g)
        out[f"scenario_{case}_price"] = price
        out[f"scenario_{case}_return"] = price / last - 1
        out[f"scenario_{case}_{tag}_growth"] = g
        if multiple_now == multiple_now and multiple_now > 0:
            out[f"scenario_{case}_{mult}"] = multiple_now * ref / last
    return out


def scenario_values(history: list[dict], *, driver: str, driver_levels: pd.Series, multiple_now: float,
                    freq: str = "annual") -> dict[str, float]:
    """All scenario-lens values at the last bar (empty without price history)."""
    if not history:
        return {}
    close = _closes(history)
    return {**price_range(close), **trend(close), **scenarios(close, driver, driver_levels, multiple_now, freq)}


def fan(last_price: float, sigma: float, days: int = TRADING_DAYS, step: int = 5) -> list[dict]:
    """The 80% band for every ``step`` trading days ahead (for charts): [{"t", "low", "high"}]."""
    rows = []
    for t in range(0, days + 1, step):
        width = Z90 * sigma * math.sqrt(t / TRADING_DAYS)
        rows.append({"t": t, "low": last_price * math.exp(-width), "high": last_price * math.exp(width)})
    return rows
