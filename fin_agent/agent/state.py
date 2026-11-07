"""Graph state. Everything is JSON-serializable so LangGraph checkpoints work."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge_dict(old: dict | None, new: dict | None) -> dict:
    """Reducer for per-lens outputs: a lens re-run replaces only its own entry."""
    return {**(old or {}), **(new or {})}


DEPTH_LENSES = {
    "quick": ["profitability", "health"],
    "standard": ["profitability", "health", "operational", "cashflow_quality", "valuation", "technical", "scenarios"],
    "360": ["profitability", "health", "operational", "cashflow_quality", "valuation", "technical", "scenarios",
            "risk_governance"],
}
DEPTH_YEARS = {"quick": 3, "standard": 5, "360": 8}
MAX_CRITIC_RETRIES = 2
LLM_CALL_BUDGET = {"quick": 15, "standard": 40, "360": 90}  # per run (guardrails layer 2)


class Request(TypedDict, total=False):
    intent: str  # full_analysis | qa | compare | screen
    tickers: list[str]
    market: str  # HOSE | HNX | UPCOM | US
    years: int
    freq: str  # annual | quarterly
    depth: str  # quick | standard | 360
    question: str
    language: str  # en | vi
    search_query: str | None  # English concept query for the vault knowledge base (from intake)


class AgentState(TypedDict, total=False):
    user_input: str
    request: Request
    clarification: str
    datasets: dict[str, dict]  # ticker -> {"meta": {...}, "records": [...]} (normalized statements)
    dq_warnings: list[dict]
    metrics: list[dict]  # Metric.to_dict()
    skills: dict[str, dict]  # lens -> {"text", "sources"}
    knowledge: dict[str, Any]  # {"status", "lens": {lens: [hit]}, "company": [hit]} (GraphRAG over the vault)
    findings: Annotated[dict[str, dict], merge_dict]  # lens -> {"findings": [...], "summary": str}
    critic: dict[str, Any]  # {"issues": {lens: [...]}, "retry": [lenses]}
    retries: int
    recommendation: dict
    answer: dict
    guardrail_log: Annotated[list[dict], operator.add]
    llm_calls: Annotated[int, operator.add]
    llm_usage: Annotated[list[dict], operator.add]  # one row per LLM call (fin_agent.llm.usage.usage_entry)
    error: str
    sync: bool  # set by the caller: run the review checkpoint and vault sync
    run_id: str
    review: dict  # {"action": "approve" | "reject", "feedback": str}
    vault_sync: dict  # SyncResult as dict
