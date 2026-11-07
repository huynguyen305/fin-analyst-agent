"""Price-based technical indicators (pure functions, ADR-004: Python counts).

Inputs are daily closes (and volumes) in ascending date order. Every function
returns NaN when the history is too short, never a guess. Windows are in
trading days: 21 ~ 1 month, 63 ~ 3 months, 126 ~ 6 months, 252 ~ 1 year.

All levels are on a closing basis (52-week high/low use closes, not intraday
highs), which keeps them reproducible from the stored bars.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

NAN = float("nan")
TRADING_DAYS = 252
RETURN_WINDOWS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}


def sma(close: pd.Series, n: int) -> float:
    return float(close.iloc[-n:].mean()) if len(close) >= n else NAN


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> float:
    """Wilder's RSI: seed with the simple average of the first ``n`` moves, then smooth by 1/n."""
    if len(close) < n + 1:
        return NAN
    delta = close.diff().dropna().to_numpy()
    gains, losses = np.clip(delta, 0, None), np.clip(-delta, 0, None)
    avg_g, avg_l = gains[:n].mean(), losses[:n].mean()
    for g, l in zip(gains[n:], losses[n:]):
        avg_g = (avg_g * (n - 1) + g) / n
        avg_l = (avg_l * (n - 1) + l) / n
    if avg_l == 0:
        return 100.0 if avg_g > 0 else 50.0
    return float(100 - 100 / (1 + avg_g / avg_l))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float, float]:
    """(macd line, signal line, histogram) at the last bar; NaN without slow + signal bars."""
    if len(close) < slow + signal:
        return NAN, NAN, NAN
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return float(line.iloc[-1]), float(sig.iloc[-1]), float(line.iloc[-1] - sig.iloc[-1])


def period_return(close: pd.Series, n: int) -> float:
    return float(close.iloc[-1] / close.iloc[-1 - n] - 1) if len(close) > n and close.iloc[-1 - n] else NAN


def volatility(close: pd.Series, n: int = 60) -> float:
    """Annualized standard deviation of daily log returns over the last ``n`` days."""
    if len(close) < n + 1:
        return NAN
    r = np.log(close.iloc[-n - 1:]).diff().dropna()
    return float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))


def max_drawdown(close: pd.Series, n: int = TRADING_DAYS) -> float:
    """Worst peak-to-trough fall over the last ``n`` days (negative fraction)."""
    if len(close) < n:
        return NAN
    w = close.iloc[-n:]
    return float((w / w.cummax() - 1).min())


def range_position(close: pd.Series, n: int = TRADING_DAYS) -> tuple[float, float]:
    """(distance below the n-day high, distance above the n-day low), closing basis."""
    if len(close) < n:
        return NAN, NAN
    w, last = close.iloc[-n:], close.iloc[-1]
    return float(last / w.max() - 1), float(last / w.min() - 1)


def volume_ratio(volume: pd.Series, short: int = 20, long: int = 60) -> float:
    v = volume.dropna()
    if len(v) < long or v.iloc[-long:].mean() == 0:
        return NAN
    return float(v.iloc[-short:].mean() / v.iloc[-long:].mean())


def relative_return(close: pd.Series, bench: pd.Series, n: int) -> float:
    """Stock return minus benchmark return over the last ``n`` common trading days."""
    both = pd.concat([close.rename("s"), bench.rename("b")], axis=1, join="inner").dropna()
    if len(both) <= n:
        return NAN
    return period_return(both["s"], n) - period_return(both["b"], n)


def indicators(history: list[dict], benchmark: list[dict] | None = None) -> dict[str, float]:
    """All technical indicators at the last bar, keyed by metric name."""
    if not history:
        return {}
    df = pd.DataFrame(history)
    df = df.assign(date=pd.to_datetime(df["date"])).sort_values("date").drop_duplicates("date").set_index("date")
    close = df["close"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df else pd.Series(dtype=float)
    last = float(close.iloc[-1])
    s50, s200 = sma(close, 50), sma(close, 200)
    _, _, hist = macd(close)
    from_high, from_low = range_position(close)
    out = {
        "sma_50": s50,
        "sma_200": s200,
        "price_vs_sma50": last / s50 - 1 if s50 == s50 else NAN,
        "price_vs_sma200": last / s200 - 1 if s200 == s200 else NAN,
        "sma50_vs_sma200": s50 / s200 - 1 if s50 == s50 and s200 == s200 else NAN,
        "rsi_14": rsi(close),
        "macd_hist_pct": hist / last if hist == hist and last else NAN,
        **{f"return_{k}": period_return(close, n) for k, n in RETURN_WINDOWS.items()},
        "volatility_60d": volatility(close),
        "max_drawdown_12m": max_drawdown(close),
        "pct_from_52w_high": from_high,
        "pct_from_52w_low": from_low,
        "volume_ratio_20_60": volume_ratio(volume),
    }
    if benchmark:
        b = pd.DataFrame(benchmark)
        b = b.assign(date=pd.to_datetime(b["date"])).drop_duplicates("date").set_index("date")["close"].astype(float)
        for k in ("3m", "6m", "12m"):
            out[f"rel_return_{k}"] = relative_return(close, b, RETURN_WINDOWS[k])
    return out
