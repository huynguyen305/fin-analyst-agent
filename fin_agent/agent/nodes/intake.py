"""Intake and Router: natural language -> Request (with input guardrails)."""

from __future__ import annotations

import re
from datetime import date

from langgraph.types import interrupt

from fin_agent.agent.prompts import INTAKE_SCHEMA, INTAKE_SYSTEM
from fin_agent.agent.state import DEPTH_YEARS, AgentState
from fin_agent.guards.guards import guard_input
from fin_agent.llm.base import LLMError, LLMProvider
from fin_agent.llm.usage import usage_entry

VI_CHARS = re.compile(r"[ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ]", re.I)
STOPWORDS = {"THE", "AND", "FOR", "ROE", "ROA", "NIM", "NPL", "CCC", "DSO", "EPS", "USD", "VND", "CEO", "CFO", "BCTC"}


def heuristic_parse(text: str) -> dict:
    """Fallback parser used when the LLM is unavailable or returns junk."""
    lower = text.lower()
    tickers = list(dict.fromkeys(t for t in re.findall(r"\b[A-Z]{3,8}\b", text) if t not in STOPWORDS))
    depth = "standard"
    if any(k in lower for k in ("360", "deep", "toàn diện", "chi tiết", "full")):
        depth = "360"
    elif any(k in lower for k in ("quick", "nhanh", "brief")):
        depth = "quick"
    years = None
    m = re.search(r"(\d{1,2})\s*(?:y\b|yr|years?|năm)", lower)
    if m:
        years = int(m.group(1))
    is_question = "?" in text or lower.startswith(("why", "what", "how", "is ", "are ", "tại sao", "vì sao"))
    return {
        "intent": "compare" if len(tickers) > 1 and not is_question else ("qa" if is_question else "full_analysis"),
        "tickers": tickers[:5],
        "market": "HOSE",
        "depth": depth,
        "years": years,
        "question": text if is_question else None,
        "language": "vi" if VI_CHARS.search(text) else "en",
        "missing": None if tickers else "Which company (ticker) should I analyze?",
    }


def make_intake(llm: LLMProvider):
    def intake(state: AgentState) -> dict:
        text = state["user_input"]
        if state.get("clarification"):
            text = f"{text}\n(User clarification: {state['clarification']})"
        text, log = guard_input(text)
        calls, usage = 0, []
        try:
            res = llm.complete(INTAKE_SYSTEM.format(today=date.today().isoformat()), text, INTAKE_SCHEMA)
            calls = 1
            usage.append(usage_entry(res.usage, "intake"))
            parsed = res.data or {}
            if not isinstance(parsed, dict) or "tickers" not in parsed:
                raise LLMError("intake output missing tickers")
        except LLMError as exc:
            if exc.usage:
                calls = 1
                usage.append(usage_entry(exc.usage, "intake", ok=False))
            parsed = heuristic_parse(text)
            log.append({"layer": "input", "check": "intake_fallback", "action": "heuristic", "detail": str(exc)[:200]})

        tickers = [t.strip().upper() for t in parsed.get("tickers") or [] if re.fullmatch(r"[A-Za-z0-9.]{1,10}", t.strip())]
        if not tickers:
            question = parsed.get("missing") or "Which company (ticker) should I analyze?"
            log.append({"layer": "input", "check": "ticker_missing", "action": "ask_user", "detail": question})
            answer = interrupt({"type": "clarify", "question": question})
            return {"clarification": str(answer), "guardrail_log": log, "llm_calls": calls, "llm_usage": usage, "request": {}}

        depth = parsed.get("depth") or "standard"
        req = {
            "intent": parsed.get("intent") or "full_analysis",
            "tickers": tickers,
            "market": parsed.get("market") or "HOSE",
            "depth": depth,
            "years": int(parsed.get("years") or DEPTH_YEARS[depth]),
            "freq": "annual",
            "question": parsed.get("question") or (state["user_input"] if parsed.get("intent") == "qa" else None),
            "language": parsed.get("language") or "en",
        }
        if req["years"] > 15:
            log.append({"layer": "input", "check": "period_feasibility", "action": "capped", "detail": f"{req['years']}y capped to 15y"})
            req["years"] = 15
        return {"request": req, "guardrail_log": log, "llm_calls": calls, "llm_usage": usage}

    return intake


def route_after_intake(state: AgentState) -> str:
    return "gather" if state.get("request", {}).get("tickers") else "intake"
