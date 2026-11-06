"""Render run outputs as Obsidian notes (Vault Sync Contract, section 2).

Pure functions: state in, ``{vault_relative_path: content}`` out. The writer
handles git. Notes follow the vault's conventions: YAML frontmatter, full-path
wikilinks, ``up`` links to the hub, English or Vietnamese per request.
"""

from __future__ import annotations

import csv
import io
from datetime import date

import yaml

from fin_agent.llm.usage import fmt_tokens, one_line, summarize

COMPANY_ROOT = "Intelligence/Investment_Analysis/Company_Analysis"
QA_ROOT = "Intelligence/Investment_Analysis/Analyst_QA"
PROJECT_ROOT = "Projects/FinAnalystAgent"
PROJECT_HUB = f"{PROJECT_ROOT}/_INDEX-FinAnalystAgent-Design-Guardrails-Evals"
IA_INDEX = "Intelligence/Investment_Analysis/_INDEX-IA-Fund-Tech-Behav-Portfolio"

ALLOWED_PREFIXES = (f"{COMPANY_ROOT}/", f"{QA_ROOT}/", f"{PROJECT_ROOT}/Runs/", f"{PROJECT_ROOT}/Learnings/", f"{PROJECT_ROOT}/Evals/")

SENTIMENT = {"Accumulate": "bullish", "Hold": "neutral", "Reduce": "bearish", "Avoid": "bearish"}


def _fm(meta: dict) -> str:
    return "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip() + "\n---\n"


def _wl(path: str, alias: str | None = None) -> str:
    return f"[[{path}|{alias}]]" if alias else f"[[{path}]]"


def company_paths(ticker: str, day: str, depth: str) -> dict[str, str]:
    base = f"{COMPANY_ROOT}/{ticker}"
    return {
        "hub": f"{base}/_INDEX-{ticker}-Reports-Data",
        "report": f"{base}/{day}_{depth}",
        "snapshot": f"{base}/Data/{day}_statements",
    }


def render_report(state: dict, run_id: str, day: str | None = None) -> str:
    day = day or date.today().isoformat()
    req, rec = state["request"], state.get("recommendation") or {}
    ticker = req["tickers"][0]
    meta = state["datasets"][ticker]["meta"]
    paths = company_paths(ticker, day, req["depth"])
    skill_sources = sorted({s for p in (state.get("skills") or {}).values() for s in p.get("sources", [])})
    fm = _fm({
        "tags": ["fin-analyst-agent", "company-analysis", ticker.lower()],
        "date": day,
        "up": [_wl(paths["hub"])],
        "fa_run_id": run_id, "fa_ticker": ticker, "fa_depth": req["depth"],
        "fa_call": rec.get("call"), "fa_confidence": rec.get("confidence"),
        "sentiment": SENTIMENT.get(rec.get("call", ""), "neutral"),
        "source": meta["source"],
    })
    by_id = {m["id"]: m for m in state.get("metrics", [])}
    lines = [
        f"# {ticker}: {rec.get('call', 'n/a')} ({rec.get('confidence', 'n/a')} confidence)",
        "",
        f"**Company hub:** {_wl(paths['hub'], ticker)} · **Data:** {_wl(paths['snapshot'], 'statements and metrics')} · "
        f"**Run:** {_wl(f'{PROJECT_ROOT}/Runs/{day}_{run_id}', run_id)}",
        f"**Market:** {meta['market']} · **Sector template:** {meta['sector']} · **Unit:** {meta['currency']} {meta['unit']} · "
        f"**Depth:** {req['depth']} · **History:** {req['years']}y {req.get('freq', 'annual')}",
        "",
        "## Thesis",
        rec.get("thesis", ""),
        "",
    ]
    for k, title in (("bull_case", "Bull case"), ("base_case", "Base case"), ("bear_case", "Bear case")):
        if rec.get(k):
            lines += [f"### {title}"] + [f"- {x}" for x in rec[k]] + [""]
    lines += ["## Key risks"] + [f"- {x}" for x in rec.get("key_risks", [])] + [""]
    lines += ["## What would change the view"] + [f"- {x}" for x in rec.get("what_would_change_view", [])] + [""]
    lines.append("## Findings by lens")
    for lens, out in (state.get("findings") or {}).items():
        flag = " ⚠️ low confidence" if out.get("low_confidence") else ""
        lines += ["", f"### {lens.replace('_', ' ').title()}{flag}", out.get("summary", "")]
        for f in out.get("findings", []):
            evidence = ", ".join(f"`{i}` = {by_id[i]['display']}" if i in by_id else f"`{i}`" for i in f["metric_ids"])
            lines.append(f"- **{f['severity']}**, {f['direction']}: {f['claim']}  ")
            lines.append(f"  <small>Evidence: {evidence}</small>")
    if rec.get("not_assessed"):
        lines += ["", f"> Not assessed: {', '.join(rec['not_assessed'])}"]
    lines += ["", "## Method", f"- Skills used: {', '.join(skill_sources) or 'none'}",
              f"- Guardrail events: {len(state.get('guardrail_log', []))} (see run log)",
              f"- LLM usage: {one_line(summarize(state.get('llm_usage')))}", "", f"*{rec.get('disclaimer', '')}*", ""]
    return fm + "\n".join(lines)


def render_snapshot(state: dict, day: str | None = None) -> tuple[str, str]:
    """Markdown note + CSV of the normalized statements and all computed metrics."""
    day = day or date.today().isoformat()
    req = state["request"]
    ticker = req["tickers"][0]
    ds = state["datasets"][ticker]
    paths = company_paths(ticker, day, req["depth"])
    periods = sorted({r["period_end"] for r in ds["records"]})
    items = sorted({r["line_item"] for r in ds["records"]})
    val = {(r["line_item"], r["period_end"]): r["value"] for r in ds["records"]}
    head = "| line item | " + " | ".join(periods) + " |\n|---|" + "---:|" * len(periods)
    rows = [f"| {i} | " + " | ".join(f"{val[(i, p)]:,.1f}" if (i, p) in val else "" for p in periods) + " |" for i in items]
    metrics = [m for m in state.get("metrics", []) if m["lens"] != "data"]
    mperiods = sorted({m["period"] for m in metrics})
    mval = {(m["name"], m["period"]): m["display"] for m in metrics}
    mnames = sorted({m["name"] for m in metrics})
    mhead = "| metric | " + " | ".join(mperiods) + " |\n|---|" + "---:|" * len(mperiods)
    mrows = [f"| {n} | " + " | ".join(mval.get((n, p), "") for p in mperiods) + " |" for n in mnames]
    fm = _fm({"tags": ["fin-analyst-agent", "data-snapshot", ticker.lower()], "date": day,
              "up": [_wl(paths["hub"])], "fa_ticker": ticker, "source": ds["meta"]["source"],
              "fetched_at": ds["meta"]["fetched_at"]})
    notes = ds["meta"].get("notes") or []
    body = [f"# {ticker} statements snapshot ({day})", "",
            f"Source: `{ds['meta']['source']}` · unit: {ds['meta']['currency']} {ds['meta']['unit']} · CSV: `{day}_statements.csv`", "",
            "## Normalized statements", head, *rows, "", "## Computed metrics", mhead, *mrows]
    md = ds["meta"].get("market_data")
    if md:
        body += ["", f"Market data: price {md['price']:,.0f} {ds['meta']['currency']} on {md['price_date']} · "
                     f"{md['shares_outstanding']:,.0f} shares (as of {md.get('shares_as_of') or 'n/a'}) · `{md['source']}`"]
    if notes:
        body += ["", "## Source mapping notes"] + [f"- {n}" for n in notes]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["ticker", "period_end", "freq", "line_item", "value", "unit", "currency", "source"])
    w.writeheader()
    w.writerows(ds["records"])
    return fm + "\n".join(body) + "\n", buf.getvalue()


def render_hub(ticker: str, existing: str | None, report_path: str, day: str, call: str | None, depth: str) -> str:
    entry = f"- {_wl(report_path, f'{day} {depth}')}: {call or 'n/a'}"
    if existing:
        return existing if entry in existing else existing.rstrip() + "\n" + entry + "\n"
    fm = _fm({"tags": ["fin-analyst-agent", "company-hub", ticker.lower()], "up": [_wl(IA_INDEX)]})
    return fm + f"# {ticker}\n\nReports and data produced by {_wl(PROJECT_HUB, 'Fin Analyst Agent')}.\n\n## Reports\n{entry}\n"


def render_run_log(state: dict, run_id: str, day: str | None = None, review: dict | None = None) -> str:
    day = day or date.today().isoformat()
    req = state.get("request", {})
    fm = _fm({"tags": ["fin-analyst-agent", "run-log"], "date": day, "up": [_wl(PROJECT_HUB)],
              "fa_run_id": run_id, "fa_tickers": req.get("tickers", []), "fa_intent": req.get("intent")})
    lines = [f"# Run {run_id}", "", f"**Request:** {state.get('user_input', '')}", "",
             "```yaml", yaml.safe_dump(req, allow_unicode=True, sort_keys=False).strip(), "```", "",
             f"LLM calls: {state.get('llm_calls', 0)} · critic retries: {state.get('retries', 0)}", ""]
    rows = state.get("llm_usage") or []
    s = summarize(rows)
    lines += ["## Token usage", f"**Total:** {one_line(s)}", ""]
    if rows:
        lines += ["| node | label | model | in | out | cache write | cache read | cost | ms |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            cost = f"${r['cost_usd']:.4f}" if r.get("cost_usd") is not None else "n/a"
            lines.append(f"| {r.get('node')}{'' if r.get('ok', True) else ' ✗'} | {r.get('label') or ''} | {r.get('model') or ''} "
                         f"| {fmt_tokens(r['input_tokens'])} | {fmt_tokens(r['output_tokens'])} "
                         f"| {fmt_tokens(r['cache_creation_input_tokens'])} | {fmt_tokens(r['cache_read_input_tokens'])} "
                         f"| {cost} | {r.get('duration_ms') or ''} |")
        lines.append("")
        if "sdk_estimate" in s["cost_sources"]:
            lines += ["*Cost is the Claude Agent SDK's API-equivalent estimate; on a subscription it counts against plan limits, not billed.*", ""]
    if review:
        lines += ["## Human review", f"- action: {review.get('action')}", f"- feedback: {review.get('feedback') or '(none)'}", ""]
    lines += ["## Guardrail log", "| layer | check | action | detail |", "|---|---|---|---|"]
    for e in state.get("guardrail_log", []):
        detail = str(e.get("detail", "")).replace("|", "\\|").replace("\n", " ")[:200]
        lines.append(f"| {e.get('layer')} | {e.get('check')} | {e.get('action')} | {detail} |")
    if state.get("dq_warnings"):
        lines += ["", "## Data-quality warnings"] + [f"- {w['ticker']} {w['period']}: {w['message']}" for w in state["dq_warnings"]]
    return fm + "\n".join(lines) + "\n"


def render_learning(run_id: str, day: str, slug: str, kind: str, observed: str, evidence: list[str],
                    proposal: str, target: str, confidence: str) -> str:
    fm = _fm({"tags": ["fin-analyst-agent", "learning-proposal"], "date": day,
              "up": [_wl(f"{PROJECT_ROOT}/Learnings/_Review-Queue")], "fa_run_id": run_id,
              "fa_kind": kind, "fa_status": "proposed", "fa_target": target, "fa_confidence": confidence})
    return fm + "\n".join([
        f"# Learning proposal: {slug.replace('-', ' ')}", "",
        f"**Kind:** {kind} · **Target note:** `{target}` · **Confidence:** {confidence}", "",
        "## Observed", observed, "", "## Evidence"] + [f"- {e}" for e in evidence] + [
        "", "## Proposed change", proposal, "",
        "> Merge this PR to accept; edit the target note by hand or ask the agent to apply it. Close the PR to reject.", ""])


def render_review_queue(existing: str | None, learning_path: str, slug: str) -> str:
    entry = f"- [ ] {_wl(learning_path, slug)}"
    if existing:
        return existing if entry in existing else existing.rstrip() + "\n" + entry + "\n"
    fm = _fm({"tags": ["fin-analyst-agent", "review-queue"], "up": [_wl(PROJECT_HUB)]})
    return fm + "# Learning review queue\n\nProposals the agent made from its runs. Tick when merged or rejected.\n\n" + entry + "\n"


def render_qa_entry(state: dict, run_id: str) -> str:
    a = state.get("answer") or {}
    req = state.get("request", {})
    by_id = {m["id"]: m for m in state.get("metrics", [])}
    ev = ", ".join(f"`{i}` = {by_id[i]['display']}" if i in by_id else f"`{i}`" for i in a.get("metric_ids", []))
    gaps = "; ".join(a.get("data_gaps") or []) or "none"
    flag = " ⚠️ low confidence" if a.get("low_confidence") else ""
    return (f"\n## {', '.join(req.get('tickers', []))}: {req.get('question') or state.get('user_input', '')}{flag}\n\n"
            f"{a.get('answer', '')}\n\n<small>Evidence: {ev} · Data gaps: {gaps} · Run: `{run_id}`</small>\n")


def qa_note(existing: str | None, day: str, entry: str) -> str:
    if existing:
        return existing.rstrip() + "\n" + entry
    fm = _fm({"tags": ["fin-analyst-agent", "analyst-qa"], "date": day, "up": [_wl(IA_INDEX)]})
    return fm + f"# Analyst Q&A {day}\n" + entry
