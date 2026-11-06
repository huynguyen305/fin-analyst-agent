"""Evaluate agent runs with Jev.

Examples::

    # run the agent on offline fixtures, then score the output
    python -m fin_agent.evals "Analyze DEMO" "Analyze the demo bank" --fixtures

    # score saved runs (output of `python -m fin_agent "..." --json > run.json`)
    python -m fin_agent.evals --from run.json --out logs/evals/report.json

Exit code 1 when any run's finding pass rate is below ``--min-pass`` (for CI).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from fin_agent.config import load_env
from fin_agent.evals.jev import JevClient, JevError
from fin_agent.evals.judge import evaluate
from fin_agent.llm.usage import one_line, summarize

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def run_agent(request: str, llm: str | None) -> dict:
    from fin_agent.agent.graph import build_graph
    from fin_agent.llm.providers import get_provider

    app = build_graph(get_provider(llm))
    return app.invoke({"user_input": request, "run_id": uuid.uuid4().hex[:8]},
                      {"configurable": {"thread_id": uuid.uuid4().hex}})


def fmt_summary(s: dict) -> list[str]:
    def pct(v):
        return "n/a" if v is None else f"{v:.0%}"

    lines = [f"  findings {s['findings']}: support {pct(s['finding_support'])}, pass {pct(s['finding_pass_rate'])}, "
             f"direction agree {pct(s['direction_agreement'])}, severity agree {pct(s['severity_agreement'])}, "
             f"insight {pct(s['insight'])}"]
    if s.get("recommendation"):
        r = s["recommendation"]
        lines.append("  recommendation: " + ", ".join(f"{k} {pct(v)}" for k, v in r.items())
                     + f"; blind call {s.get('jev_call')} ({'agrees' if s.get('call_agreement') else 'differs'})")
    if s.get("qa"):
        lines.append("  qa: " + ", ".join(f"{k} {pct(v)}" for k, v in s["qa"].items()))
    if s["needs_review"]:
        lines.append("  needs human review: " + ", ".join(s["needs_review"]))
    for e in s["errors"]:
        lines.append("  error: " + e)
    return lines


def main(argv: list[str] | None = None) -> int:
    load_env()
    ap = argparse.ArgumentParser(description="Score agent runs with TypeSafe Jev")
    ap.add_argument("requests", nargs="*", help="natural-language requests to run and score")
    ap.add_argument("--from", dest="from_json", action="append", default=[], help="saved run JSON (repeatable)")
    ap.add_argument("--fixtures", action="store_true", help="run the agent on offline fixtures (tests/fixtures)")
    ap.add_argument("--llm", default=None, help="agent LLM provider (see python -m fin_agent --help)")
    ap.add_argument("--out", default=None, help="write the full JSON report here (default logs/evals/<timestamp>.json)")
    ap.add_argument("--min-pass", type=float, default=None, help="fail if a run's finding pass rate is below this (0-1)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    if not args.requests and not args.from_json:
        ap.error("give at least one request or --from file")
    if args.fixtures:
        os.environ["FIN_AGENT_FIXTURES"] = str(FIXTURES)

    try:
        client = JevClient()
    except JevError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    runs = [(p, json.loads(Path(p).read_text(encoding="utf-8"))) for p in args.from_json]
    runs += [(r, run_agent(r, args.llm)) for r in args.requests]

    report, failed = [], False
    with client:
        for label, out in runs:
            print(f"# {label}")
            if out.get("error"):
                print(f"  agent error, not scored: {out['error']}")
                report.append({"run": label, "error": out["error"]})
                continue
            ev = evaluate(out, client, workers=args.workers)
            agent_usage = summarize(out.get("llm_usage"))
            print("\n".join(fmt_summary(ev["summary"])))
            print(f"  agent LLM: {one_line(agent_usage)}")
            print(f"  Jev eval:  {one_line(ev['usage'])}")
            rate = ev["summary"]["finding_pass_rate"]
            if args.min_pass is not None and rate is not None and rate < args.min_pass:
                failed = True
            report.append({"run": label, "eval": ev, "agent_usage": agent_usage})

    path = Path(args.out or f"logs/evals/{datetime.now():%Y%m%d-%H%M%S}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
