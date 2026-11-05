"""Lens analysts (fan-out), Critic, Recommender and Analyst Q&A."""

from __future__ import annotations

from langgraph.types import Send

from fin_agent.agent.prompts import (
    ANALYST_SYSTEM, LANG_NAMES, LENS_PROMPT, LENS_SCHEMA, QA_PROMPT, QA_SCHEMA,
    RECOMMEND_PROMPT, RECOMMEND_SCHEMA,
)
from fin_agent.agent.state import DEPTH_LENSES, LLM_CALL_BUDGET, MAX_CRITIC_RETRIES, AgentState
from fin_agent.guards.guards import ensure_disclaimer, guard_answer, guard_claim, guard_recommendation
from fin_agent.llm.base import LLMError, LLMProvider
from fin_agent.llm.usage import usage_entry

NEEDS_MARKET_DATA = {"valuation"}  # until a price provider exists


def _system(req: dict) -> str:
    return ANALYST_SYSTEM.format(language_name=LANG_NAMES.get(req.get("language", "en"), "English"))


def _metrics_block(metrics: list[dict], lens: str | None = None) -> str:
    rows = [m for m in metrics if lens in (None, "risk_governance") or m["lens"] in (lens, "data")]
    return "\n".join(f"{m['id']} | {m['display']} | {m['formula']}" for m in rows)


# ---------- fan-out ----------

def fan_out_lenses(state: AgentState) -> list[Send]:
    req = state["request"]
    retry = (state.get("critic") or {}).get("retry")
    lenses = retry if retry else DEPTH_LENSES[req["depth"]]
    issues = (state.get("critic") or {}).get("issues", {})
    return [
        Send("lens", {
            "lens": lens, "request": req, "metrics": state["metrics"], "skills": state["skills"],
            "datasets": {t: {"meta": d["meta"]} for t, d in state["datasets"].items()},
            "feedback": issues.get(lens, []),
        })
        for lens in lenses
    ]


def make_lens(llm: LLMProvider):
    def lens_node(payload: dict) -> dict:
        lens, req = payload["lens"], payload["request"]
        meta = next(iter(payload["datasets"].values()))["meta"]
        if lens in NEEDS_MARKET_DATA:
            return {"findings": {lens: {"not_assessed": True, "summary": "Not assessed: market price data is not connected yet.", "findings": []}}}
        feedback = ""
        if payload.get("feedback"):
            feedback = "\nYour previous answer failed review. Fix these issues:\n- " + "\n- ".join(payload["feedback"]) + "\n"
        prompt = LENS_PROMPT.format(
            lens=lens, ticker=meta["ticker"], market=meta["market"], sector=meta["sector"], unit=meta["unit"],
            skill=payload["skills"].get(lens, {}).get("text", ""), metrics=_metrics_block(payload["metrics"], lens),
            feedback=feedback,
        )
        label = lens + (" (retry)" if payload.get("feedback") else "")
        try:
            res = llm.complete(_system(req), prompt, LENS_SCHEMA)
            data = res.data if isinstance(res.data, dict) else {"summary": "", "findings": []}
            usage = usage_entry(res.usage, "lens", label)
        except LLMError as exc:
            data = {"summary": f"Lens failed: {exc}", "findings": [], "error": True}
            usage = usage_entry(exc.usage, "lens", label, ok=False)
        return {"findings": {lens: data}, "llm_calls": 1, "llm_usage": [usage]}

    return lens_node


# ---------- critic (claim-to-number reconciliation, guardrails layer 3) ----------

def check_claim(claim: str, metric_ids: list[str], by_id: dict[str, dict], direction: str | None = None) -> list[str]:
    """Run the claim guard (numeric reconciliation, direction, no guarantees)."""
    return guard_claim({"claim": claim, "metric_ids": metric_ids, "direction": direction}, by_id)


def critic(state: AgentState) -> dict:
    by_id = {m["id"]: m for m in state["metrics"]}
    issues: dict[str, list[str]] = {}
    cleaned: dict[str, dict] = {}
    log = []
    for lens, out in (state.get("findings") or {}).items():
        if out.get("not_assessed"):
            continue
        lens_issues, kept = [], []
        for f in out.get("findings", []):
            problems = check_claim(f.get("claim", ""), f.get("metric_ids", []), by_id, f.get("direction"))
            if problems:
                lens_issues += problems
            else:
                kept.append(f)
        if lens_issues:
            issues[lens] = lens_issues
            log.append({"layer": "output", "check": "numeric_reconciliation", "action": "reask", "detail": f"{lens}: {len(lens_issues)} issue(s)"})
        cleaned[lens] = {**out, "findings": kept, "rejected": len(out.get("findings", [])) - len(kept)}

    retries = state.get("retries", 0)
    budget = LLM_CALL_BUDGET[state["request"]["depth"]]
    over_budget = state.get("llm_calls", 0) + len(issues) + 1 > budget
    if issues and over_budget:
        log.append({"layer": "process", "check": "llm_budget", "action": "stop_retries",
                    "detail": f"{state.get('llm_calls', 0)} calls used of {budget}"})
    if issues and retries < MAX_CRITIC_RETRIES and not over_budget:
        return {"critic": {"issues": issues, "retry": list(issues)}, "retries": retries + 1, "guardrail_log": log}
    if issues:
        log.append({"layer": "process", "check": "loop_limit", "action": "ship_low_confidence", "detail": f"lenses still failing: {list(issues)}"})
        for lens in issues:
            cleaned[lens]["low_confidence"] = True
    return {"critic": {"issues": issues, "retry": []}, "findings": cleaned, "guardrail_log": log}


def route_after_critic(state: AgentState):
    if (state.get("critic") or {}).get("retry"):
        return fan_out_lenses(state)
    return "recommend"


# ---------- recommender ----------

def make_recommender(llm: LLMProvider):
    def recommend(state: AgentState) -> dict:
        req = state["request"]
        meta = next(iter(state["datasets"].values()))["meta"]
        findings = state.get("findings") or {}
        not_assessed = [l for l, o in findings.items() if o.get("not_assessed")]
        lines = []
        for lens, out in findings.items():
            if out.get("not_assessed"):
                continue
            flag = " (low confidence)" if out.get("low_confidence") else ""
            lines.append(f"## {lens}{flag}: {out.get('summary', '')}")
            lines += [f"- [{f['severity']}, {f['direction']}] {f['claim']} ({', '.join(f['metric_ids'])})" for f in out.get("findings", [])]
        prompt = RECOMMEND_PROMPT.format(
            ticker=meta["ticker"], market=meta["market"], sector=meta["sector"], depth=req["depth"],
            skill=state["skills"].get("recommendation", {}).get("text", ""),
            findings="\n".join(lines) or "(none)", not_assessed=", ".join(not_assessed) or "none",
        )
        log, calls, rec, feedback, usage = [], 0, {}, "", []
        for attempt in range(2):  # one re-ask if guards fail
            label = "re-ask" if attempt else None
            try:
                res = llm.complete(_system(req), prompt + feedback, RECOMMEND_SCHEMA)
            except LLMError as exc:
                usage.append(usage_entry(exc.usage, "recommend", label, ok=False))
                return {"error": f"Recommender failed: {exc}", "llm_calls": calls + 1, "llm_usage": usage}
            rec = res.data or {}
            usage.append(usage_entry(res.usage, "recommend", label))
            calls += 1
            problems = guard_recommendation(rec)
            if not problems:
                break
            log.append({"layer": "output", "check": "recommendation_guard", "action": "reask" if attempt == 0 else "flag",
                        "detail": "; ".join(problems)[:300]})
            feedback = "\n\nYour previous answer failed review. Fix:\n- " + "\n- ".join(problems)
        else:
            rec["guardrail_warning"] = problems
        if any(o.get("low_confidence") for o in findings.values()) and rec.get("confidence") == "high":
            rec["confidence"] = "medium"
            log.append({"layer": "output", "check": "confidence_cap", "action": "fix", "detail": "high -> medium (low-confidence lens)"})
        lang = req.get("language", "en")
        rec["disclaimer"] = ensure_disclaimer(rec.get("disclaimer", ""), lang).strip()
        rec["not_assessed"] = not_assessed
        return {"recommendation": rec, "guardrail_log": log, "llm_calls": calls, "llm_usage": usage}

    return recommend


# ---------- analyst Q&A ----------

def make_qa(llm: LLMProvider):
    def qa(state: AgentState) -> dict:
        req = state["request"]
        prompt = QA_PROMPT.format(
            question=req.get("question") or state["user_input"],
            skill=state["skills"].get("qa", {}).get("text", ""), metrics=_metrics_block(state["metrics"]),
        )
        by_id = {m["id"]: m for m in state["metrics"]}
        log, calls, ans, usage = [], 0, {}, []
        feedback = ""
        for attempt in range(MAX_CRITIC_RETRIES + 1):
            label = f"re-ask {attempt}" if attempt else None
            try:
                res = llm.complete(_system(req), prompt + feedback, QA_SCHEMA)
            except LLMError as exc:
                usage.append(usage_entry(exc.usage, "qa", label, ok=False))
                return {"error": f"Q&A failed: {exc}", "llm_calls": calls + 1, "llm_usage": usage}
            ans = res.data or {}
            usage.append(usage_entry(res.usage, "qa", label))
            calls += 1
            problems = guard_answer(ans, by_id)
            if not problems:
                break
            log.append({"layer": "output", "check": "numeric_reconciliation", "action": "reask", "detail": "; ".join(problems)[:300]})
            feedback = "\n\nYour previous answer failed review:\n- " + "\n- ".join(problems)
        else:
            ans["low_confidence"] = True
        ans["disclaimer"] = ensure_disclaimer("", req.get("language", "en")).strip()
        return {"answer": ans, "guardrail_log": log, "llm_calls": calls, "llm_usage": usage}

    return qa


def route_after_skills(state: AgentState):
    if state["request"].get("intent") == "qa":
        return "qa"
    return fan_out_lenses(state)
