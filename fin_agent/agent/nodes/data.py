"""Data Gatherer, Quantifier and Skill Loader nodes (no LLM)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from fin_agent.agent.state import DEPTH_LENSES, AgentState
from fin_agent.data.providers import ProviderError, fetch_statements
from fin_agent.data.schema import DataQualityError, MarketData, Sector, Statements, assert_quality
from fin_agent.metrics.library import compute_metrics, line_item_metrics
from fin_agent.skills.loader import load_skill_packs


def statements_to_state(st: Statements) -> dict:
    meta = {k: getattr(st, k) for k in ("ticker", "market", "currency", "unit", "freq", "source", "fetched_at", "notes")}
    meta["sector"] = st.sector.value
    meta["market_data"] = st.market_data.to_dict() if st.market_data else None
    return {"meta": meta, "records": st.to_records()}


def statements_from_state(ds: dict) -> Statements:
    meta = ds["meta"]
    rows: dict = {}
    for r in ds["records"]:
        rows.setdefault(pd.Timestamp(r["period_end"]).date(), {})[r["line_item"]] = r["value"]
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "period_end"
    return Statements(
        ticker=meta["ticker"], market=meta["market"], currency=meta["currency"], unit=meta["unit"],
        sector=Sector(meta["sector"]), freq=meta["freq"], data=df, source=meta["source"],
        fetched_at=meta["fetched_at"], notes=list(meta.get("notes", [])),
        market_data=MarketData.from_dict(meta.get("market_data")),
    )


def gather(state: AgentState) -> dict:
    req = state["request"]
    datasets, warnings, log = {}, [], []
    tickers = req["tickers"]

    def fetch(ticker):
        try:
            return fetch_statements(ticker, req["market"], req["years"], req.get("freq", "annual")), None
        except ProviderError as exc:
            return None, exc

    # every ticker in parallel (network-bound); results are handled in request order
    with ThreadPoolExecutor(max_workers=max(1, min(4, len(tickers)))) as pool:
        fetched = list(pool.map(fetch, tickers))
    for ticker, (st, fetch_error) in zip(tickers, fetched):
        try:
            if fetch_error:
                raise fetch_error
            for w in assert_quality(st):
                warnings.append({"ticker": ticker, **w.__dict__})
        except ProviderError as exc:
            log.append({"layer": "process", "check": "data_fetch", "action": "stop", "detail": str(exc)[:300]})
            return {"error": f"Could not get data for {ticker}: {exc}", "guardrail_log": log}
        except DataQualityError as exc:
            log.append({"layer": "process", "check": "dq_gate", "action": "stop", "detail": str(exc)[:300]})
            return {"error": f"Data-quality gate failed for {ticker}: {exc}", "guardrail_log": log}
        if len(st.data) < req["years"]:
            log.append({"layer": "input", "check": "period_feasibility", "action": "shortened",
                        "detail": f"{ticker}: asked {req['years']}y, source has {len(st.data)}"})
        datasets[ticker] = statements_to_state(st)
    return {"datasets": datasets, "dq_warnings": warnings, "guardrail_log": log}


def quantify(state: AgentState) -> dict:
    metrics = []
    for ds in state["datasets"].values():
        st = statements_from_state(ds)
        metrics += [m.to_dict() for m in compute_metrics(st) + line_item_metrics(st)]
    return {"metrics": metrics}


def load_skills(state: AgentState) -> dict:
    req = state["request"]
    first = next(iter(state["datasets"].values()))
    sector = Sector(first["meta"]["sector"])
    packs = load_skill_packs(DEPTH_LENSES[req["depth"]], sector)
    return {"skills": {k: {"text": p.text, "sources": p.sources} for k, p in packs.items()}}


def route_after_gather(state: AgentState) -> str:
    return "end" if state.get("error") else "quantify"
