"""Lens analysts (fan-out), Critic, Recommender and Analyst Q&A."""

from __future__ import annotations

import re

from langgraph.types import Send

from fin_agent.agent.prompts import (
    ANALYST_SYSTEM, DISCLAIMER, LANG_NAMES, LENS_PROMPT, LENS_SCHEMA, QA_PROMPT, QA_SCHEMA,
    RECOMMEND_PROMPT, RECOMMEND_SCHEMA,
)
from fin_agent.agent.state import DEPTH_LENSES, MAX_CRITIC_RETRIES, AgentState
from fin_agent.llm.base import LLMError, LLMProvider

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
        try:
            res = llm.complete(_system(req), prompt, LENS_SCHEMA)
            data = res.data if isinstance(res.data, dict) else {"summary": "", "findings": []}
        except LLMError as exc:
            data = {"summary": f"Lens failed: {exc}", "findings": [], "error": True}
        return {"findings": {lens: data}, "llm_calls": 1}

    return lens_node


# ---------- critic (claim-to-number reconciliation, guardrails layer 3) ----------

NUM_RE = re.compile(r"(-?\d[\d,]*\.?\d*)\s*(%|x\b|days?\b|ngày)?", re.I)


def _numbers_in(text: str) -> list[tuple[float, str, float]]:
    """(value, kind, half_precision) for numbers worth checking: %, x, days, or decimals/thousands."""
    out = []
    for m in NUM_RE.finditer(text):
        raw, suffix = m.group(1), (m.group(2) or "").lower()
        clean = raw.replace(",", "")
        try:
            v = float(clean)
        except ValueError:
            continue
        if not suffix and ("." not in clean and "," not in raw):
            continue  # bare integers: years, counts, list numbers
        if not suffix and 1900 <= v <= 2100:
            continue
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        kind = "pct" if suffix == "%" else "x" if suffix == "x" else "days" if suffix.startswith(("day", "ngày")) else "amount"
        out.append((v, kind, 0.5 * 10 ** (-decimals)))
    return out


def _matches(v: float, kind: str, half: float, m: dict) -> bool:
    target = m["value"] * 100 if m["unit"] == "ratio" else m["value"]
    if kind == "pct" and m["unit"] != "ratio":
        return False
    tol = max(half, abs(target) * 0.005)
    return abs(abs(v) - abs(target)) <= tol + 1e-9


def check_claim(claim: str, metric_ids: list[str], by_id: dict[str, dict]) -> list[str]:
    issues = []
    unknown = [i for i in metric_ids if i not in by_id]
    if unknown:
        issues.append(f"Unknown metric ids: {', '.join(unknown)}")
    cited = [by_id[i] for i in metric_ids if i in by_id]
    if not cited:
        issues.append(f"No valid metric cited for: {claim[:80]!r}")
        return issues
    for v, kind, half in _numbers_in(claim):
        if not any(_matches(v, kind, half, m) for m in cited):
            issues.append(f"Number {v:g}{'%' if kind == 'pct' else ''} in {claim[:60]!r} does not match any cited metric")
    return issues


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
            problems = check_claim(f.get("claim", ""), f.get("metric_ids", []), by_id)
            if problems:
                lens_issues += problems
            else:
                kept.append(f)
        if lens_issues:
            issues[lens] = lens_issues
            log.append({"layer": "output", "check": "numeric_reconciliation", "action": "reask", "detail": f"{lens}: {len(lens_issues)} issue(s)"})
        cleaned[lens] = {**out, "findings": kept, "rejected": len(out.get("findings", [])) - len(kept)}

    retries = state.get("retries", 0)
    if issues and retries < MAX_CRITIC_RETRIES:
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
        log = []
        try:
            rec = llm.complete(_system(req), prompt, RECOMMEND_SCHEMA).data or {}
        except LLMError as exc:
            return {"error": f"Recommender failed: {exc}", "llm_calls": 1}
        if len(rec.get("key_risks", [])) < 3:
            log.append({"layer": "output", "check": "risk_balance", "action": "flag", "detail": f"only {len(rec.get('key_risks', []))} risks"})
            rec["risk_balance_warning"] = True
        if any(o.get("low_confidence") for o in findings.values()) and rec.get("confidence") == "high":
            rec["confidence"] = "medium"
            log.append({"layer": "output", "check": "confidence_cap", "action": "fix", "detail": "high -> medium (low-confidence lens)"})
        rec["disclaimer"] = DISCLAIMER.get(req.get("language", "en"), DISCLAIMER["en"])
        rec["not_assessed"] = not_assessed
        return {"recommendation": rec, "guardrail_log": log, "llm_calls": 1}

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
        log, calls, ans = [], 0, {}
        feedback = ""
        for attempt in range(MAX_CRITIC_RETRIES + 1):
            try:
                ans = llm.complete(_system(req), prompt + feedback, QA_SCHEMA).data or {}
            except LLMError as exc:
                return {"error": f"Q&A failed: {exc}", "llm_calls": calls + 1}
            calls += 1
            problems = check_claim(ans.get("answer", ""), ans.get("metric_ids", []), by_id)
            if not problems:
                break
            log.append({"layer": "output", "check": "numeric_reconciliation", "action": "reask", "detail": "; ".join(problems)[:300]})
            feedback = "\n\nYour previous answer failed review:\n- " + "\n- ".join(problems)
        else:
            ans["low_confidence"] = True
        ans["disclaimer"] = DISCLAIMER.get(req.get("language", "en"), DISCLAIMER["en"])
        return {"answer": ans, "guardrail_log": log, "llm_calls": calls}

    return qa


def route_after_skills(state: AgentState):
    if state["request"].get("intent") == "qa":
        return "qa"
    return fan_out_lenses(state)
