"""Agent evaluation with Jev in place of an LLM judge.

The critic already checks numbers deterministically (claim figures vs metric
values). What it cannot check is judgement: does the claim's interpretation
follow from the figures, is the severity label sensible, does the call follow
from the findings? Those are typed decisions, so Jev answers them with
calibrated probabilities at a fraction of an LLM judge's cost.

Bias control: where the agent made a labelled choice (direction, severity,
call), Jev is asked "blind" (the agent's label is left out of the state) and
the two are compared afterwards as an agreement rate.

Probabilities between ``REVIEW_BAND`` and low-confidence choices are flagged
``needs_review`` for a human, instead of being trusted either way.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from fin_agent.evals.jev import JevClient, JevError, choice, noul, score
from fin_agent.llm.usage import summarize, usage_entry

REVIEW_BAND = (0.3, 0.7)  # noul probabilities in this range are too uncertain to auto-grade
MIN_CHOICE_CONFIDENCE = 0.6
PASS = 0.5

DIRECTIONS = {
    "improving": "The metric moved in the favourable direction over the periods shown",
    "deteriorating": "The metric moved in the unfavourable direction over the periods shown",
    "stable": "The metric changed little over the periods shown",
    "mixed": "The metric moved up and down without a clear direction",
}
SEVERITIES = {
    "positive": "A strength for the company",
    "neutral": "Neither a strength nor a concern",
    "watch": "A weakness worth monitoring but not yet serious",
    "red_flag": "A serious problem that could hurt the investment case",
}
CALLS = {
    "Accumulate": "Business quality and (if assessed) valuation are attractive; build a position",
    "Hold": "Strengths and weaknesses roughly balance; keep the position",
    "Reduce": "Weaknesses outweigh strengths; trim the position",
    "Avoid": "Serious problems; do not own",
}
INSIGHT_LEVELS = [
    "Only restates a number",
    "Judges whether the level or trend is good or bad",
    "Explains the driver or the implication for an investor",
]


# ---------------------------------------------------------------- task builders
# Each task is (kind, ref, state, questions, grade) where grade(result) -> item dict.

def _metric_context(ids: list[str], metrics: list[dict]) -> list[dict]:
    """Cited metrics plus the same metric's other periods, so trend claims can be judged."""
    by_id = {m["id"]: m for m in metrics}
    names = {(i.split(".", 1)[0], by_id[i]["name"]) for i in ids if i in by_id}
    rows = [m for m in metrics if (m["id"].split(".", 1)[0], m["name"]) in names]
    return [{"id": m["id"], "metric": m["name"], "period": m["period"], "value": m["display"],
             "formula": m["formula"], "cited": m["id"] in ids} for m in sorted(rows, key=lambda m: (m["name"], m["period"]))]


def finding_tasks(out: dict) -> list[tuple]:
    metrics = out.get("metrics") or []
    tasks = []
    for lens, block in (out.get("findings") or {}).items():
        if block.get("not_assessed"):
            continue
        for i, f in enumerate(block.get("findings") or []):
            state = {"lens": lens, "claim": f.get("claim", ""), "metrics": _metric_context(f.get("metric_ids", []), metrics)}
            questions = {
                "supported": noul(
                    "Does the claim's interpretation follow from the metric values and trend in the state, "
                    "without asserting facts that are not in the state?",
                    true="Every judgement in the claim is backed by the metrics shown",
                    false="The claim overstates, contradicts, or goes beyond the metrics shown"),
                "direction": choice("Which best describes the trend of the cited metric across the periods shown?", DIRECTIONS),
                "severity": choice("For an equity investor, how should this claim be classified?", SEVERITIES),
                "insight": score("How much analysis does the claim add beyond the raw figure?", INSIGHT_LEVELS),
            }

            def grade(res, f=f):
                sev, sev_c = res.choice("severity")
                d, d_c = res.choice("direction")
                ins, _ = res.score("insight")
                sup = res.noul("supported")
                review = _uncertain(sup) or min(sev_c, d_c) < MIN_CHOICE_CONFIDENCE
                return {"supported": sup, "direction_agrees": d == f.get("direction"), "jev_direction": d,
                        "severity_agrees": sev == f.get("severity"), "jev_severity": sev,
                        "insight": ins / (len(INSIGHT_LEVELS) - 1), "needs_review": review}

            tasks.append(("finding", f"{lens}[{i}]", state, questions, grade))
    return tasks


def recommendation_tasks(out: dict) -> list[tuple]:
    rec = out.get("recommendation") or {}
    if not rec.get("call"):
        return []
    lenses = {lens: {"summary": b.get("summary", ""), "not_assessed": bool(b.get("not_assessed")),
                     "findings": [{"claim": f.get("claim"), "severity": f.get("severity")} for f in b.get("findings") or []]}
              for lens, b in (out.get("findings") or {}).items()}
    blind = {"lenses": lenses}
    full = {"lenses": lenses, "not_assessed": rec.get("not_assessed") or [],
            **{k: rec.get(k) for k in ("call", "confidence", "thesis", "bull_case", "base_case", "bear_case",
                                         "key_risks", "what_would_change_view")}}

    def grade_blind(res):
        c, conf = res.choice("call")
        return {"jev_call": c, "call_agrees": c == rec["call"], "needs_review": conf < MIN_CHOICE_CONFIDENCE}

    def grade_full(res):
        vals = {k: res.noul(k) for k in ("consistent", "balanced", "discloses_gaps", "specific_risks")}
        return {**vals, "needs_review": any(_uncertain(v) for v in vals.values())}

    return [
        ("recommendation_blind", "call", blind,
         {"call": choice("Given only these lens findings, which investment call fits best?", CALLS)}, grade_blind),
        ("recommendation", "rationale", full, {
            "consistent": noul("Is the call consistent with the balance of strengths and concerns across the lenses?"),
            "balanced": noul("Do both the bull case and the bear case contain specific, substantive points rather than generic ones?"),
            "discloses_gaps": noul("If any lens is listed as not assessed, does the thesis or the cases acknowledge that gap? "
                                   "Answer yes if nothing is listed as not assessed."),
            "specific_risks": noul("Are the key risks tied to this company's figures rather than generic market risks?"),
        }, grade_full),
    ]


def qa_tasks(out: dict) -> list[tuple]:
    a = out.get("answer") or {}
    if not a.get("answer"):
        return []
    state = {"question": (out.get("request") or {}).get("question") or out.get("user_input", ""),
             "answer": a["answer"], "data_gaps": a.get("data_gaps") or [],
             "metrics": _metric_context(a.get("metric_ids") or [], out.get("metrics") or [])}

    def grade(res):
        vals = {k: res.noul(k) for k in ("answers_question", "grounded")}
        return {**vals, "needs_review": any(_uncertain(v) for v in vals.values())}

    return [("qa", "answer", state, {
        "answers_question": noul("Does the answer directly address the question asked?"),
        "grounded": noul("Is every figure and fact in the answer present in the metrics shown, or declared as a data gap?"),
    }, grade)]


def _uncertain(p: float) -> bool:
    return REVIEW_BAND[0] < p < REVIEW_BAND[1]


# ---------------------------------------------------------------- run

def evaluate(out: dict, client: JevClient, workers: int = 4) -> dict:
    """Score one final agent state. Returns ``{"items", "summary", "usage"}``."""
    tasks = finding_tasks(out) + recommendation_tasks(out) + qa_tasks(out)
    items, usage = _run(tasks, client, workers)
    return {"items": items, "summary": summarize_items(items), "usage": summarize(usage), "jev_model": client.model}


def _run(tasks: list[tuple], client: JevClient, workers: int) -> tuple[list[dict], list[dict]]:
    def one(task):
        kind, ref, state, questions, grade = task
        try:
            res = client.system_one(state, questions)
        except JevError as exc:
            return {"kind": kind, "ref": ref, "error": str(exc)}, usage_entry(exc.usage, "eval", f"{kind}:{ref}", ok=False)
        try:
            item = {"kind": kind, "ref": ref, **grade(res)}
        except (KeyError, TypeError, ValueError) as exc:  # answer missing or malformed
            item = {"kind": kind, "ref": ref, "error": f"unexpected Jev answer: {exc}"}
        return item, usage_entry(res.usage, "eval", f"{kind}:{ref}", ok="error" not in item)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pairs = list(pool.map(one, tasks))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _mean(rows: list[dict], key: str, fn: Callable = float) -> float | None:
    vals = [fn(r[key]) for r in rows if key in r]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarize_items(items: list[dict]) -> dict:
    ok = [i for i in items if "error" not in i]
    f = [i for i in ok if i["kind"] == "finding"]
    rec = next((i for i in ok if i["kind"] == "recommendation"), {})
    blind = next((i for i in ok if i["kind"] == "recommendation_blind"), {})
    qa = next((i for i in ok if i["kind"] == "qa"), {})
    return {
        "findings": len(f),
        "finding_support": _mean(f, "supported"),  # mean probability the interpretation is supported
        "finding_pass_rate": _mean(f, "supported", lambda p: p >= PASS),
        "direction_agreement": _mean(f, "direction_agrees"),
        "severity_agreement": _mean(f, "severity_agrees"),
        "insight": _mean(f, "insight"),  # 0 = restates numbers, 1 = explains drivers/implications
        "recommendation": {k: rec.get(k) for k in ("consistent", "balanced", "discloses_gaps", "specific_risks")} if rec else None,
        "call_agreement": blind.get("call_agrees"), "jev_call": blind.get("jev_call"),
        "qa": {k: qa.get(k) for k in ("answers_question", "grounded")} if qa else None,
        "needs_review": [f"{i['kind']}:{i['ref']}" for i in ok if i.get("needs_review")],
        "errors": [f"{i['kind']}:{i['ref']}: {i['error']}" for i in items if "error" in i],
    }
