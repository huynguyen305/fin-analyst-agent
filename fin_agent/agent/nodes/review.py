"""Human review checkpoint and vault sync (guardrails layer 2 and 4)."""

from __future__ import annotations

import os
import uuid

from langgraph.types import interrupt

from fin_agent.agent.state import AgentState
from fin_agent.vault.writer import VaultSyncError, VaultWriter
from fin_agent.vault.sync import sync_run

APPROVE_WORDS = {"", "y", "yes", "ok", "approve", "approved", "đồng ý", "duyệt"}


def review(state: AgentState) -> dict:
    """Pause for Huy's approval before anything is written to the vault."""
    rec = state.get("recommendation") or {}
    ans = state.get("answer") or {}
    resume = interrupt({
        "type": "review",
        "question": "Approve syncing this result to the vault? (approve / reject, plus optional feedback)",
        "call": rec.get("call"), "confidence": rec.get("confidence"),
        "thesis": rec.get("thesis") or ans.get("answer", "")[:400],
    })
    if isinstance(resume, dict):
        decision = {"action": resume.get("action", "approve"), "feedback": resume.get("feedback", "")}
    else:
        text = str(resume).strip()
        decision = {"action": "approve", "feedback": ""} if text.lower() in APPROVE_WORDS else (
            {"action": "reject", "feedback": text[6:].strip(" :")} if text.lower().startswith("reject")
            else {"action": "approve", "feedback": text})
    return {"review": decision, "guardrail_log": [{"layer": "process", "check": "human_review", "action": decision["action"],
                                                   "detail": decision["feedback"][:200]}]}


def sync(state: AgentState) -> dict:
    vault = os.getenv("VAULT_PATH")
    if not vault:
        return {"guardrail_log": [{"layer": "sync", "check": "vault_path", "action": "skip", "detail": "VAULT_PATH not set"}]}
    run_id = state.get("run_id") or uuid.uuid4().hex[:8]
    try:
        writer = VaultWriter(vault, push=os.getenv("FIN_AGENT_VAULT_PUSH", "1") != "0")
        res = sync_run(dict(state), run_id, writer, state.get("review"))
    except VaultSyncError as exc:
        return {"guardrail_log": [{"layer": "sync", "check": "vault_sync", "action": "stop", "detail": str(exc)[:300]}],
                "vault_sync": {"error": str(exc)}}
    events = [{"layer": "sync", "check": "link_check", "action": "warn", "detail": w} for w in res.warnings]
    return {"vault_sync": res.__dict__, "guardrail_log": events}


def route_to_review(state: AgentState) -> str:
    if state.get("error"):
        return "end"
    return "review" if state.get("sync") else "end"
