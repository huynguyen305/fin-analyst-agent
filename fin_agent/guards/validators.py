"""Finance validators (Guardrails and Anti-Patterns note, layers 1 and 3).

Each check is a Guardrails AI ``Validator`` so it can be composed into ``Guard``s,
reused from LangChain, and reported uniformly. Where a check needs graph context
(computed metrics), it is passed through ``metadata``.

The domain-free validators (prompt injection, PII, certainty language, disclaimer)
live in ``hubkit.guards.validators`` and are re-exported here. Importing hubkit first
also disables Guardrails telemetry before the library loads.
"""

from __future__ import annotations

import json
import re

from hubkit.guards.validators import (  # noqa: F401  (re-exported for existing imports)
    GUARANTEE_PATTERNS, INJECTION_PATTERNS, PII, PII_PATTERNS, Disclaimer, NoGuarantees, PromptInjection,
)
from guardrails.validators import (  # noqa: E402
    FailResult, PassResult, ValidationResult, Validator, register_validator,
)

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
# valuation metrics have one period (the price date), so no direction check applies to them
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


# a rule threshold is quoted as a comparison: "above the 15% bar", "below 0.8x", "dưới ngưỡng 30%"
COMPARISON_CUE = re.compile(
    r"\b(above|below|over|under|than|exceed\w*|beyond|threshold|bar|guideline|benchmark|limit|cap|rule|"
    r"at least|at most|trên|dưới|ngưỡng|vượt|hơn|tối thiểu|tối đa)\b", re.I)


def rule_thresholds(skill_text: str) -> set[tuple[float, str]]:
    """Numbers stated in a lens's skill rules (curated thresholds such as 0.8x, 85%, -30%)."""
    return {(abs(v), kind) for v, kind, _ in numbers_in(skill_text or "")}


def quoted_thresholds(text: str, thresholds: set[tuple[float, str]]) -> set[tuple[float, str]]:
    """Threshold values the claim quotes as a comparison (cue word within 40 characters before it)."""
    out = set()
    for m in NUM_RE.finditer(text):
        found = numbers_in(m.group(0))
        if not found:
            continue
        v, kind, _ = found[0]
        if (abs(v), kind) in thresholds and COMPARISON_CUE.search(text[max(0, m.start() - 40):m.start()]):
            out.add((abs(v), kind))
    return out


def reconcile(text: str, metric_ids: list[str], by_id: dict[str, dict],
              thresholds: set[tuple[float, str]] | None = None) -> list[str]:
    """Issues found when checking a claim's numbers against the metrics it cites.

    ``thresholds`` (from the lens's skill rules) lets a claim compare a metric with a curated
    rule ("CFO/net income of 0.72x is below the 0.8x guideline"): such a number passes only
    when it is a rule value *and* quoted as a comparison. Any other unmatched number fails.
    """
    issues = []
    unknown = [i for i in metric_ids if i not in by_id]
    if unknown:
        issues.append(f"Unknown metric ids: {', '.join(unknown)}")
    cited = [by_id[i] for i in metric_ids if i in by_id]
    if not cited:
        issues.append(f"No valid metric cited for: {text[:80]!r}")
        return issues
    allowed = quoted_thresholds(text, thresholds) if thresholds else set()
    for v, kind, half in numbers_in(text):
        if (abs(v), kind) in allowed:
            continue
        if not any(number_matches(v, kind, half, m) for m in cited):
            issues.append(f"Number {v:g}{'%' if kind == 'pct' else ''} in {text[:60]!r} does not match any cited metric")
    return issues


@register_validator(name="fin-agent/numeric-reconciliation", data_type="string")
class NumericReconciliation(Validator):
    """Every number in the text must match a metric it cites (AP-01, AP-02).

    metadata: ``metric_ids`` (list) and ``metrics_by_id`` (dict id -> metric dict).
    """

    def _validate(self, value: str, metadata: dict) -> ValidationResult:
        issues = reconcile(value, metadata.get("metric_ids", []), metadata.get("metrics_by_id", {}),
                           metadata.get("thresholds"))
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
