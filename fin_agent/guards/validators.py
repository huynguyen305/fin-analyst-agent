"""Guardrails AI validators (Guardrails and Anti-Patterns note, layers 1 and 3).

Each check is a Guardrails AI ``Validator`` so it can be composed into ``Guard``s,
reused from LangChain, and reported uniformly. Where a check needs graph context
(computed metrics), it is passed through ``metadata``.

Guardrails AI ships hub telemetry that exports traces to a remote endpoint by
default. Financial prompts must stay local, so this module disables
OpenTelemetry export and Guardrails metrics before the library is imported.
"""

from __future__ import annotations

import json
import os
import re

os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from guardrails.settings import settings  # noqa: E402

settings.rc.enable_metrics = False

from guardrails.validators import (  # noqa: E402
    FailResult, PassResult, ValidationResult, Validator, register_validator,
)

from fin_agent.agent.prompts import DISCLAIMER  # noqa: E402

# ---------------------------------------------------------------- numbers

NUM_RE = re.compile(r"(-?\d[\d,]*\.?\d*)\s*(%|x\b|days?\b|ngày)?", re.I)

# metrics where a higher value is better (others: lower is better)
HIGHER_IS_BETTER = {
    "gross_margin", "operating_margin", "net_margin", "roe", "roa", "asset_turnover",
    "revenue_growth", "net_income_growth", "current_ratio", "quick_ratio", "interest_coverage",
    "cfo_to_net_income", "fcf", "fcf_margin", "dividend_fcf_cover", "nii_to_avg_loans",
    "equity_to_assets", "reserve_coverage", "loan_growth", "deposit_growth",
    "revenue", "gross_profit", "operating_income", "net_income", "cfo", "cash", "net_interest_income",
}
LOWER_IS_BETTER = {
    "debt_to_equity", "net_debt_to_equity", "liabilities_to_assets", "dso", "dio", "ccc",
    "sga_to_revenue", "accruals_ratio", "npl_ratio", "cost_of_risk",
}


def numbers_in(text: str) -> list[tuple[float, str, float]]:
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


def number_matches(v: float, kind: str, half: float, m: dict) -> bool:
    target = m["value"] * 100 if m["unit"] == "ratio" else m["value"]
    if kind == "pct" and m["unit"] != "ratio":
        return False
    tol = max(half, abs(target) * 0.005)
    return abs(abs(v) - abs(target)) <= tol + 1e-9


def reconcile(text: str, metric_ids: list[str], by_id: dict[str, dict]) -> list[str]:
    """Issues found when checking a claim's numbers against the metrics it cites."""
    issues = []
    unknown = [i for i in metric_ids if i not in by_id]
    if unknown:
        issues.append(f"Unknown metric ids: {', '.join(unknown)}")
    cited = [by_id[i] for i in metric_ids if i in by_id]
    if not cited:
        issues.append(f"No valid metric cited for: {text[:80]!r}")
        return issues
    for v, kind, half in numbers_in(text):
        if not any(number_matches(v, kind, half, m) for m in cited):
            issues.append(f"Number {v:g}{'%' if kind == 'pct' else ''} in {text[:60]!r} does not match any cited metric")
    return issues


@register_validator(name="fin-agent/numeric-reconciliation", data_type="string")
class NumericReconciliation(Validator):
    """Every number in the text must match a metric it cites (AP-01, AP-02).

    metadata: ``metric_ids`` (list) and ``metrics_by_id`` (dict id -> metric dict).
    """

    def _validate(self, value: str, metadata: dict) -> ValidationResult:
        issues = reconcile(value, metadata.get("metric_ids", []), metadata.get("metrics_by_id", {}))
        return FailResult(error_message=" | ".join(issues)) if issues else PassResult()


@register_validator(name="fin-agent/direction-consistency", data_type="string")
class DirectionConsistency(Validator):
    """'improving' / 'deteriorating' must agree with how the cited metric moved.

    Applies only when the cited ids are one metric over several periods and the
    change is material (more than 5%). metadata: ``direction``, ``metric_ids``,
    ``metrics_by_id``.
    """

    def _validate(self, value: str, metadata: dict) -> ValidationResult:
        direction = metadata.get("direction")
        by_id = metadata.get("metrics_by_id", {})
        cited = [by_id[i] for i in metadata.get("metric_ids", []) if i in by_id]
        names = {m.get("name") for m in cited}
        if direction not in ("improving", "deteriorating") or len(names) != 1 or None in names or len(cited) < 2:
            return PassResult()
        name = names.pop()
        if name not in HIGHER_IS_BETTER and name not in LOWER_IS_BETTER:
            return PassResult()
        series = sorted(cited, key=lambda m: m["period"])
        first, last = series[0]["value"], series[-1]["value"]
        if first == 0 or abs(last - first) / abs(first) < 0.05:
            return PassResult()
        better = (last > first) == (name in HIGHER_IS_BETTER)
        expected = "improving" if better else "deteriorating"
        if expected != direction:
            return FailResult(error_message=f"Direction '{direction}' contradicts {name} moving from {first:.4g} to {last:.4g} ({expected})")
        return PassResult()


# ---------------------------------------------------------------- language

GUARANTEE_PATTERNS = [
    r"\bguarantee[sd]?\b", r"\brisk[- ]free\b", r"\bcan(?:'|no)t lose\b", r"\bsure (?:to|bet)\b",
    r"\bwill (?:definitely|certainly) (?:rise|go up|double)\b", r"\bno downside\b",
    r"chắc chắn (?:tăng|lãi|có lời)", r"không có rủi ro", r"đảm bảo (?:lợi nhuận|lãi)", r"cam kết lợi nhuận",
]


@register_validator(name="fin-agent/no-guarantees", data_type="string")
class NoGuarantees(Validator):
    """Blocks promises of returns or certainty (EN and VI)."""

    def _validate(self, value: str, metadata: dict) -> ValidationResult:
        hits = [p for p in GUARANTEE_PATTERNS if re.search(p, value, re.I)]
        if hits:
            return FailResult(error_message=f"Certainty or guarantee language found: {hits}")
        return PassResult()


@register_validator(name="fin-agent/disclaimer", data_type="string")
class Disclaimer(Validator):
    """The not-financial-advice disclaimer must be present; FIX appends it."""

    def _validate(self, value: str, metadata: dict) -> ValidationResult:
        lang = metadata.get("language", "en")
        text = DISCLAIMER.get(lang, DISCLAIMER["en"])
        if text in value:
            return PassResult()
        return FailResult(error_message="Disclaimer missing", fix_value=f"{value.rstrip()}\n\n{text}")


@register_validator(name="fin-agent/risk-balance", data_type="string")
class RiskBalance(Validator):
    """A recommendation (JSON string) needs >= 3 risks, a bear case and view-changers (AP-09)."""

    def _validate(self, value: str, metadata: dict) -> ValidationResult:
        try:
            rec = json.loads(value)
        except json.JSONDecodeError:
            return FailResult(error_message="Recommendation is not valid JSON")
        problems = []
        if len(rec.get("key_risks") or []) < 3:
            problems.append(f"needs at least 3 key risks (has {len(rec.get('key_risks') or [])})")
        if not rec.get("bear_case"):
            problems.append("bear case is empty")
        if not rec.get("what_would_change_view"):
            problems.append("'what would change the view' is empty")
        return FailResult(error_message="; ".join(problems)) if problems else PassResult()


# ---------------------------------------------------------------- input

INJECTION_PATTERNS = [
    r"ignore (?:all |any )?(?:previous|prior|above) (?:instructions|rules)",
    r"disregard (?:the |your )?(?:system|previous) (?:prompt|instructions)",
    r"you are now\b", r"new instructions?:", r"reveal (?:your|the) (?:system )?prompt",
    r"</?(?:data|skill|system)>",
    r"bỏ qua (?:các |mọi )?(?:hướng dẫn|chỉ dẫn)", r"quên (?:các |mọi )?(?:hướng dẫn|quy tắc)",
]


@register_validator(name="fin-agent/prompt-injection", data_type="string")
class PromptInjection(Validator):
    """Detects instruction-like text in user input or fetched content (AP-11). FIX strips the matches."""

    def _validate(self, value: str, metadata: dict) -> ValidationResult:
        hits = [p for p in INJECTION_PATTERNS if re.search(p, value, re.I)]
        if not hits:
            return PassResult()
        cleaned = value
        for p in hits:
            cleaned = re.sub(p, "[removed]", cleaned, flags=re.I)
        return FailResult(error_message=f"Possible prompt injection: {hits}", fix_value=cleaned)


PII_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "vn_citizen_id": r"\b0\d{11}\b",  # CCCD: 12 digits starting with 0
    "phone": r"(?<!\d)(?:\+84|0)(?:3|5|7|8|9)\d{8}(?!\d)",
    "account_number": r"\b\d{9,16}\b",
}


@register_validator(name="fin-agent/pii", data_type="string")
class PII(Validator):
    """Redacts personal identifiers before text reaches prompts or logs. FIX replaces them."""

    def _validate(self, value: str, metadata: dict) -> ValidationResult:
        found, cleaned = [], value
        for label, pat in PII_PATTERNS.items():
            if re.search(pat, cleaned):
                found.append(label)
                cleaned = re.sub(pat, f"[{label}]", cleaned)
        if found:
            return FailResult(error_message=f"PII redacted: {found}", fix_value=cleaned)
        return PassResult()
