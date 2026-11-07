"""Local Streamlit UI: ``streamlit run app/ui.py`` then open http://localhost:8501.

Chat in English or Vietnamese. Each graph node shows as a live step; clarifying
questions and the vault review checkpoint appear inline.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fin_agent.agent.graph import build_graph  # noqa: E402
from fin_agent.config import load_env  # noqa: E402
from fin_agent.llm.providers import get_provider  # noqa: E402
from fin_agent.llm.routing import table as routing_table  # noqa: E402
from fin_agent.llm.usage import fmt_cost, fmt_tokens, one_line, summarize  # noqa: E402
from fin_agent.ui.genui import AgentActivity  # noqa: E402
from genui_render import Renderer  # noqa: E402

load_env()
st.set_page_config(page_title="Fin Analyst Agent", page_icon="📊", layout="wide")

# ---------------------------------------------------------------- session

ss = st.session_state
ss.setdefault("messages", [])  # (role, kind, payload)
ss.setdefault("thread", None)
ss.setdefault("pending", None)  # interrupt payload awaiting the user


@st.cache_resource(show_spinner=False)
def get_app(provider_name: str):
    return build_graph(get_provider(provider_name))


with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("LLM", ["claude_subscription", "anthropic_api"],
                            index=0 if os.getenv("FIN_AGENT_LLM", "claude_subscription") == "claude_subscription" else 1)
    sync_on = st.toggle("Review and sync results to the vault", value=bool(os.getenv("VAULT_PATH")),
                        disabled=not os.getenv("VAULT_PATH"),
                        help="Needs VAULT_PATH in .env. Results go to a branch + PR, never to main.")
    st.caption(f"Vault: `{os.getenv('VAULT_PATH') or 'not set'}`")
    if os.getenv("FIN_AGENT_FIXTURES"):
        st.warning("Offline fixtures are on (FIN_AGENT_FIXTURES): data is synthetic.")
    session_rows = [r for _, kind, p in ss.messages if kind == "state" for r in (p.get("llm_usage") or [])]
    if session_rows:
        tot = summarize(session_rows)
        st.divider()
        st.markdown("**Session usage**")
        st.caption(f"{fmt_tokens(tot['total_tokens'])} tokens · {fmt_cost(tot)} · {tot['calls']} LLM calls")
    with st.expander("Model per agent"):
        st.caption("Cheapest model that does each job; override with FIN_AGENT_MODEL_<NODE> in .env")
        st.dataframe(pd.DataFrame(routing_table()), hide_index=True, use_container_width=True)
    st.divider()
    st.markdown("**Try**\n- `Phân tích HPG 5 năm, toàn diện`\n- `Quick view of VCB`\n- `Why is FPT's DSO rising?`")
    if st.button("New conversation"):
        ss.messages, ss.thread, ss.pending = [], None, None
        st.rerun()

app = get_app(provider)


# ---------------------------------------------------------------- rendering

def render_result(state: dict) -> None:
    if state.get("error"):
        st.error(state["error"])
        return
    by_id = {m["id"]: m for m in state.get("metrics", [])}
    if state.get("answer"):
        a = state["answer"]
        st.markdown(a.get("answer", ""))
        if a.get("metric_ids"):
            st.caption("Evidence: " + " · ".join(f"`{i}` = {by_id[i]['display']}" for i in a["metric_ids"] if i in by_id))
        if a.get("data_gaps"):
            st.info("Data gaps: " + "; ".join(a["data_gaps"]))
        st.caption(a.get("disclaimer", ""))
        return
    rec = state.get("recommendation") or {}
    if not rec:
        return
    ticker = ", ".join(state["request"]["tickers"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Ticker", ticker)
    c2.metric("Call", rec.get("call", "n/a"))
    c3.metric("Confidence", rec.get("confidence", "n/a"))
    st.markdown(f"**Thesis.** {rec.get('thesis', '')}")
    t_sum, t_lens, t_metrics, t_audit = st.tabs(["Scenarios and risks", "Lenses", "Metrics", "Sources and guardrails"])
    with t_sum:
        cols = st.columns(3)
        for col, (k, title) in zip(cols, (("bull_case", "🟢 Bull"), ("base_case", "⚪ Base"), ("bear_case", "🔴 Bear"))):
            col.markdown(f"**{title}**\n" + "\n".join(f"- {x}" for x in rec.get(k, [])))
        st.markdown("**Key risks**\n" + "\n".join(f"- {x}" for x in rec.get("key_risks", [])))
        st.markdown("**What would change the view**\n" + "\n".join(f"- {x}" for x in rec.get("what_would_change_view", [])))
        if rec.get("not_assessed"):
            st.info("Not assessed: " + ", ".join(rec["not_assessed"]))
    with t_lens:
        icon = {"positive": "🟢", "neutral": "⚪", "watch": "🟡", "red_flag": "🔴"}
        for lens, out in (state.get("findings") or {}).items():
            label = lens.replace("_", " ").title() + (" (low confidence)" if out.get("low_confidence") else "")
            with st.expander(label, expanded=False):
                st.write(out.get("summary", ""))
                for f in out.get("findings", []):
                    st.markdown(f"{icon.get(f['severity'], '•')} {f['claim']}")
                    st.caption(" · ".join(f"`{i}` = {by_id[i]['display']}" for i in f["metric_ids"] if i in by_id))
    with t_metrics:
        metrics = [m for m in state.get("metrics", []) if m["lens"] != "data"]
        if metrics:
            df = pd.DataFrame(metrics)
            wide = df.pivot_table(index="name", columns="period", values="display", aggfunc="first").fillna("")
            st.dataframe(wide, use_container_width=True)
            choice = st.selectbox("Trend", sorted(df["name"].unique()), key=f"trend-{ss.thread}")
            series = df[df["name"] == choice].set_index("period")["value"].sort_index()
            st.line_chart(series)
    with t_audit:
        skills = sorted({s for p in (state.get("skills") or {}).values() for s in p.get("sources", [])})
        st.markdown("**Skills used**\n" + "\n".join(f"- `{s}`" for s in skills))
        kn = state.get("knowledge") or {}
        hits = [h for hs in (kn.get("lens") or {}).values() for h in hs] + (kn.get("company") or [])
        if hits:
            rows = sorted({(h["path"], h["heading"] or h["title"], h["depth"], h["score"]) for h in hits}, key=lambda r: (r[2], -r[3]))
            st.markdown("**Vault knowledge retrieved** (hop 0 = similarity seed, 1+ = reached via links, tags, ticker or lens)")
            st.dataframe(pd.DataFrame(rows, columns=["note", "section", "hop", "similarity"]), use_container_width=True, hide_index=True)
        elif kn.get("status") and kn["status"] != "ok":
            st.caption(f"Vault knowledge: {kn['status']}")
        ds = next(iter(state.get("datasets", {}).values()), {})
        if ds:
            st.markdown(f"**Data source:** `{ds['meta']['source']}` · fetched {ds['meta']['fetched_at']}")
        st.markdown(f"**LLM calls:** {state.get('llm_calls', 0)} · **critic retries:** {state.get('retries', 0)} "
                    "· token usage below")
        if state.get("guardrail_log"):
            st.dataframe(pd.DataFrame(state["guardrail_log"]), use_container_width=True, hide_index=True)
    st.caption(rec.get("disclaimer", ""))


def render_usage(state: dict) -> None:
    """Token usage and cost for this run (every run, including Q&A and failed ones)."""
    rows = state.get("llm_usage") or []
    s = summarize(rows)
    with st.expander(f"🔢 Usage: {fmt_tokens(s['total_tokens'])} tokens · {fmt_cost(s)} · {s['calls']} LLM calls", expanded=False):
        st.markdown(one_line(s) + (f"  \nModels: {', '.join(s['models'])}" if s["models"] else ""))
        if rows:
            df = pd.DataFrame(rows)[["node", "label", "model", "input_tokens", "output_tokens",
                                     "cache_creation_input_tokens", "cache_read_input_tokens", "total_tokens",
                                     "cost_usd", "duration_ms", "ok"]]
            st.dataframe(df, use_container_width=True, hide_index=True)
        if s["failed_calls"]:
            st.caption(f"{s['failed_calls']} call(s) failed after spending tokens; they are included above.")
        if "sdk_estimate" in s["cost_sources"]:
            st.caption("Cost is the Claude Agent SDK's API-equivalent estimate. On a Claude subscription it is not billed; "
                       "usage counts against your plan limits.")


def render_sync(state: dict) -> None:
    vs = state.get("vault_sync")
    if not vs:
        return
    if vs.get("error"):
        st.error(f"Vault sync failed: {vs['error']}")
        return
    link = vs.get("pr_url") or vs.get("compare_url")
    st.success(f"Vault: branch `{vs['branch']}` with {len(vs['files'])} files" + (f" · [open PR]({link})" if link else ""))
    for w in vs.get("warnings", []):
        st.warning(w)


def run_graph(payload, resume: bool = False) -> None:
    """Stream the graph into the animated agent panel (AgenticGenUI-style component specs);
    store the result or the pending interrupt."""
    cfg = {"configurable": {"thread_id": ss.thread}}
    activity = ss.get(f"activity-{ss.thread}") if resume else None
    activity = activity or AgentActivity()
    activity.waiting = activity.finished = False  # resuming after the review checkpoint
    renderer, panel = Renderer(), st.empty()
    panel.html(renderer.render(activity.specs(), activity.seq))
    for chunk in app.stream(payload, cfg, stream_mode="updates"):
        for node, update in chunk.items():
            activity.update(node, update)
        panel.html(renderer.render(activity.specs(), activity.seq))
    snapshot = app.get_state(cfg)
    interrupts = [i for t in snapshot.tasks for i in (t.interrupts or [])]
    ss.pending = interrupts[0].value if interrupts else None
    activity.finish(waiting=bool(interrupts))
    panel.html(renderer.render(activity.specs(), activity.seq))
    ss[f"activity-{ss.thread}"] = activity
    specs = activity.specs()
    if resume and len(ss.messages) >= 2 and ss.messages[-2][1] == "genui":
        ss.messages[-2] = ("assistant", "genui", specs)
    if resume and ss.messages and ss.messages[-1][1] == "state":
        ss.messages[-1] = ("assistant", "state", snapshot.values)  # same run: replace, don't duplicate
    else:
        ss.messages.append(("assistant", "genui", specs))
        ss.messages.append(("assistant", "state", snapshot.values))


# ---------------------------------------------------------------- page

st.title("📊 Fin Analyst Agent")
st.caption("Financial statement analysis for HOSE (and US) companies. Numbers come from code; the model explains them.")

for role, kind, payload in ss.messages:
    with st.chat_message(role):
        if kind == "text":
            st.markdown(payload)
        elif kind == "genui":
            with st.expander("Agent steps", expanded=False):
                st.html(Renderer().render(payload, current_seq=-1))  # history: no entry animations
        else:
            render_result(payload)
            render_sync(payload)
            render_usage(payload)

pending = ss.pending
if pending and pending.get("type") == "review":
    with st.chat_message("assistant"):
        st.markdown("**Review before syncing to the vault.** Approve to write the report, data and run log as a PR; "
                    "reject to log only your feedback as a learning proposal.")
        fb = st.text_area("Feedback (optional, becomes a learning proposal)", key=f"fb-{ss.thread}")
        a, r = st.columns(2)
        if a.button("✅ Approve and sync", use_container_width=True):
            ss.pending = None
            run_graph(Command(resume={"action": "approve", "feedback": fb}), resume=True)
            st.rerun()
        if r.button("❌ Reject", use_container_width=True):
            ss.pending = None
            run_graph(Command(resume={"action": "reject", "feedback": fb}), resume=True)
            st.rerun()

prompt = st.chat_input("Ask about a company, e.g. 'Phân tích FPT 5 năm, 360'")
if prompt:
    ss.messages.append(("user", "text", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    if pending and pending.get("type") == "clarify":
        run_graph(Command(resume=prompt))
    else:
        ss.thread = uuid.uuid4().hex[:8]
        run_graph({"user_input": prompt, "sync": sync_on, "run_id": ss.thread})
    if ss.pending and ss.pending.get("type") == "clarify":
        ss.messages.append(("assistant", "text", ss.pending.get("question", "Which company?")))
    st.rerun()
