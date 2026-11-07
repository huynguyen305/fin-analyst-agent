"""Data for the "Price scenarios" view: price history, the 80% band, the trend path and the
12-month scenario points, all derived from computed metrics (no new numbers). UI-agnostic:
returns plain DataFrames that any chart library can draw."""

from __future__ import annotations

import math

import pandas as pd

from fin_agent.metrics.scenarios import TRADING_DAYS, fan

CASES = (("bear", "Bear"), ("base", "Base"), ("bull", "Bull"))


def latest_values(metrics: list[dict], ticker: str) -> dict[str, dict]:
    """Latest metric per name for one ticker (the scenario lens has one period: the price date)."""
    out: dict[str, dict] = {}
    for m in metrics:
        if m["id"].startswith(f"{ticker}.") and (m["name"] not in out or m["period"] > out[m["name"]]["period"]):
            out[m["name"]] = m
    return out


def scenario_view(state: dict, ticker: str) -> dict | None:
    """``{"history", "band", "trend", "points", "table", "price_date"}`` or None when the lens has no data."""
    ds = (state.get("datasets") or {}).get(ticker) or {}
    md = (ds.get("meta") or {}).get("market_data") or {}
    m = latest_values(state.get("metrics") or [], ticker)
    if not md.get("history") or "range_volatility" not in m:
        return None
    hist = pd.DataFrame(md["history"])[["date", "close"]]
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date")
    last_date, last = hist["date"].iloc[-1], float(hist["close"].iloc[-1])
    future = pd.bdate_range(last_date, periods=TRADING_DAYS + 1)  # trading-day approximation (no holiday calendar)

    band = pd.DataFrame(fan(last, m["range_volatility"]["value"]))
    band["date"] = future[band["t"]]

    trend = None
    if "trend_annual" in m:
        g = m["trend_annual"]["value"]
        t = list(range(0, TRADING_DAYS + 1, 5))
        trend = pd.DataFrame({"date": future[t], "price": [last * math.pow(1 + g, d / TRADING_DAYS) for d in t]})

    end = future[-1]
    points = pd.DataFrame([
        {"date": end, "case": label, "price": m[f"scenario_{key}_price"]["value"],
         "display": m[f"scenario_{key}_price"]["display"]}
        for key, label in CASES if f"scenario_{key}_price" in m
    ])

    rows = []
    for key, label in CASES:
        if f"scenario_{key}_price" not in m:
            continue
        growth = m.get(f"scenario_{key}_earnings_growth") or m.get(f"scenario_{key}_book_growth")
        mult = m.get(f"scenario_{key}_pe") or m.get(f"scenario_{key}_pb")
        rows.append({
            "Scenario": label,
            "12-month price": m[f"scenario_{key}_price"]["display"],
            "vs today": m[f"scenario_{key}_return"]["display"],
            ("Earnings growth" if growth and "earnings" in growth["name"] else "Book value growth"): growth["display"] if growth else "n/a",
            ("Implied P/E" if mult and mult["name"].endswith("_pe") else "Implied P/B"): mult["display"] if mult else "n/a",
            "Assumption": m[f"scenario_{key}_price"]["formula"].split(":", 1)[-1].split(";")[0].strip(),
        })
    ranges = [
        {"Horizon": h, "Low (10th pct)": m[f"range_{k}_low"]["display"], "High (90th pct)": m[f"range_{k}_high"]["display"],
         "If trend continued": m[f"trend_{k}_price"]["display"] if f"trend_{k}_price" in m else "n/a"}
        for k, h in (("3m", "3 months"), ("12m", "12 months")) if f"range_{k}_low" in m
    ]
    return {"history": hist, "band": band, "trend": trend, "points": points, "table": pd.DataFrame(rows),
            "ranges": pd.DataFrame(ranges), "price_date": m["range_volatility"]["period"],
            "trend_r2": m.get("trend_r2", {}).get("display"), "trend_annual": m.get("trend_annual", {}).get("display"),
            "volatility": m["range_volatility"]["display"]}
