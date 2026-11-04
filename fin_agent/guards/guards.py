"""Guards composed from the validators, with one function per use in the graph.

Every function returns guardrail-log events in the graph's format
(``{"layer", "check", "action", "detail"}``) so the UI and the vault run log show
exactly what fired.
"""

from __future__ import annotations

import json

from guardrails import Guard, OnFailAction

from fin_agent.guards.validators import (
    PII, Disclaimer, DirectionConsistency, NoGuarantees, NumericReconciliation, PromptInjection, RiskBalance,
)

_input_guard = Guard().use(PromptInjection(on_fail=OnFailAction.FIX), PII(on_fail=OnFailAction.FIX))
_content_guard = Guard().use(PromptInjection(on_fail=OnFailAction.FIX))
_claim_guard = Guard().use(
    NumericReconciliation(on_fail=OnFailAction.NOOP),
    DirectionConsistency(on_fail=OnFailAction.NOOP),
    NoGuarantees(on_fail=OnFailAction.NOOP),
)
_answer_guard = Guard().use(NumericReconciliation(on_fail=OnFailAction.NOOP), NoGuarantees(on_fail=OnFailAction.NOOP))
_risk_guard = Guard().use(RiskBalance(on_fail=OnFailAction.NOOP))
_text_guard = Guard().use(NoGuarantees(on_fail=OnFailAction.NOOP))
_disclaimer_guard = Guard().use(Disclaimer(on_fail=OnFailAction.FIX))

CHECK_NAMES = {
    "NumericReconciliation": "numeric_reconciliation", "DirectionConsistency": "direction_check",
    "NoGuarantees": "no_guarantees", "RiskBalance": "risk_balance", "PromptInjection": "prompt_injection",
    "PII": "pii", "Disclaimer": "disclaimer",
}


def _failures(outcome) -> list[tuple[str, str]]:
    return [
        (CHECK_NAMES.get(s.validator_name, s.validator_name), s.failure_reason or "")
        for s in (outcome.validation_summaries or []) if s.validator_status == "fail"
    ]


def guard_input(text: str, layer: str = "input") -> tuple[str, list[dict]]:
    """Clean user text (injection, PII). Returns cleaned text and log events."""
    out = _input_guard.validate(text)
    events = [{"layer": layer, "check": c, "action": "fix", "detail": r[:200]} for c, r in _failures(out)]
    return (out.validated_output if out.validated_output is not None else text), events


def guard_fetched_content(text: str) -> tuple[str, list[dict]]:
    """For news, PDFs and other fetched text: always treated as data, injection stripped."""
    out = _content_guard.validate(text)
    events = [{"layer": "input", "check": c, "action": "strip", "detail": r[:200]} for c, r in _failures(out)]
    return (out.validated_output if out.validated_output is not None else text), events


def guard_claim(finding: dict, metrics_by_id: dict) -> list[str]:
    """Problems with one lens finding (empty list = passes)."""
    out = _claim_guard.validate(
        finding.get("claim", ""),
        metadata={"metric_ids": finding.get("metric_ids", []), "metrics_by_id": metrics_by_id,
                  "direction": finding.get("direction")},
    )
    return [r for _, r in _failures(out)]


def guard_answer(answer: dict, metrics_by_id: dict) -> list[str]:
    out = _answer_guard.validate(
        answer.get("answer", ""), metadata={"metric_ids": answer.get("metric_ids", []), "metrics_by_id": metrics_by_id},
    )
    return [r for _, r in _failures(out)]


def guard_recommendation(rec: dict) -> list[str]:
    """Structural (risk balance) and language (no guarantees) problems in a recommendation."""
    problems = [r for _, r in _failures(_risk_guard.validate(json.dumps(rec, ensure_ascii=False)))]
    narrative = " ".join(
        [rec.get("thesis", "")] + [x for k in ("bull_case", "base_case", "bear_case") for x in rec.get(k, [])]
    )
    problems += [r for _, r in _failures(_text_guard.validate(narrative))]
    return problems


def ensure_disclaimer(text: str, language: str) -> str:
    if not text.strip():
        from fin_agent.agent.prompts import DISCLAIMER
        return DISCLAIMER.get(language, DISCLAIMER["en"])
    out = _disclaimer_guard.validate(text, metadata={"language": language})
    return out.validated_output if out.validated_output is not None else text
