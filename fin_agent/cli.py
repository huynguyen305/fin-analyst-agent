"""Command line entry point: ``python -m fin_agent "Analyze FPT, 360 view"``."""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from langgraph.types import Command

from fin_agent.agent.graph import build_graph
from fin_agent.config import load_env
from fin_agent.llm.providers import get_provider


def render(out: dict) -> str:
    if out.get("error"):
        return f"ERROR: {out['error']}"
    lines = []
    req = out.get("request", {})
    if out.get("answer"):
        a = out["answer"]
        lines += [a.get("answer", ""), "", "Sources: " + ", ".join(a.get("metric_ids", []))]
        if a.get("data_gaps"):
            lines.append("Data gaps: " + "; ".join(a["data_gaps"]))
        lines += ["", a.get("disclaimer", "")]
        return "\n".join(lines)
    rec = out.get("recommendation", {})
    lines.append(f"# {', '.join(req.get('tickers', []))}: {rec.get('call')} (confidence: {rec.get('confidence')})")
    lines += ["", rec.get("thesis", ""), ""]
    for lens, f in (out.get("findings") or {}).items():
        flag = " [low confidence]" if f.get("low_confidence") else ""
        lines.append(f"## {lens}{flag}")
        lines.append(f.get("summary", ""))
        for x in f.get("findings", []):
            lines.append(f"- ({x['severity']}) {x['claim']}  [{', '.join(x['metric_ids'])}]")
        lines.append("")
    for k in ("bull_case", "base_case", "bear_case", "key_risks", "what_would_change_view"):
        if rec.get(k):
            lines.append(f"**{k.replace('_', ' ').title()}:** " + "; ".join(rec[k]))
    lines += ["", rec.get("disclaimer", "")]
    lines.append(f"\nLLM calls: {out.get('llm_calls', 0)} · guardrail events: {len(out.get('guardrail_log', []))}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    load_env()
    ap = argparse.ArgumentParser(description="Fin Analyst Agent")
    ap.add_argument("request", help='Natural-language request, e.g. "Phân tích FPT 5 năm, 360"')
    ap.add_argument("--json", action="store_true", help="print the raw final state as JSON")
    ap.add_argument("--llm", default=None, help="claude_subscription (default) | anthropic_api")
    ap.add_argument("--sync", action="store_true", help="review the result, then write it to the vault as a PR (needs VAULT_PATH)")
    args = ap.parse_args(argv)

    app = build_graph(get_provider(args.llm))
    run_id = uuid.uuid4().hex[:8]
    cfg = {"configurable": {"thread_id": run_id}}
    out = app.invoke({"user_input": args.request, "sync": args.sync, "run_id": run_id}, cfg)
    while "__interrupt__" in out:  # human-in-the-loop: ask on the terminal
        payload = out["__interrupt__"][0].value
        if payload.get("type") == "review":
            print(render(app.get_state(cfg).values))
        out = app.invoke(Command(resume=input(f"\n{payload.get('question', '?')} > ")), cfg)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    elif not args.sync:
        print(render(out))
    vs = out.get("vault_sync")
    if vs:
        print("\nVault:", vs.get("error") or f"branch {vs['branch']} ({len(vs['files'])} files) "
              + (vs.get("pr_url") or vs.get("compare_url") or "committed locally"))
    return 1 if out.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
