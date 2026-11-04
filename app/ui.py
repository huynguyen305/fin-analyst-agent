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

from fin_agent.agent.graph import build_graph  # noqa: E402
from fin_agent.config import load_env  # noqa: E402
from fin_agent.llm.providers import get_provider  # noqa: E402

load_env()
st.set_page_config(page_title="Fin Analyst Agent", page_icon="📊", layout="wide")

NODE_LABELS = {
    "intake": "Understanding the request", "gather": "Fetching statements + data-quality gate",
    "quantify": "Computing metrics", "load_skills": "Loading skills from the vault",
    "lens": "Lens analysis", "critic": "Checking every number", "recommend": "Forming the view",
    "qa": "Answering the question", "review": "Waiting for your review", "sync": "Writing to the vault",
}

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
        ds = next(iter(state.get("datasets", {}).values()), {})
        if ds:
            st.markdown(f"**Data source:** `{ds['meta']['source']}` · fetched {ds['meta']['fetched_at']}")
        st.markdown(f"**LLM calls:** {state.get('llm_calls', 0)} · **critic retries:** {state.get('retries', 0)}")
        if state.get("guardrail_log"):
            st.dataframe(pd.DataFrame(state["guardrail_log"]), use_container_width=True, hide_index=True)
    st.caption(rec.get("disclaimer", ""))


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
    """Stream the graph, showing each node as a live step; store result or pending interrupt."""
    cfg = {"configurable": {"thread_id": ss.thread}}
    with st.status("Working…", expanded=True) as status:
        for chunk in app.stream(payload, cfg, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    continue
                label = NODE_LABELS.get(node, node)
                if node == "lens" and isinstance(update, dict):
                    label += ": " + ", ".join((update.get("findings") or {}).keys())
                status.write(f"✓ {label}")
        snapshot = app.get_state(cfg)
        interrupts = [i for t in snapshot.tasks for i in (t.interrupts or [])]
        if interrupts:
            ss.pending = interrupts[0].value
            status.update(label="Waiting for you", state="running", expanded=False)
        else:
            ss.pending = None
            status.update(label="Done", state="complete", expanded=False)
    if resume and ss.messages and ss.messages[-1][1] == "state":
        ss.messages[-1] = ("assistant", "state", snapshot.values)  # same run: replace, don't duplicate
    else:
        ss.messages.append(("assistant", "state", snapshot.values))


# ---------------------------------------------------------------- page

st.title("📊 Fin Analyst Agent")
st.caption("Financial statement analysis for HOSE (and US) companies. Numbers come from code; the model explains them.")

for role, kind, payload in ss.messages:
    with st.chat_message(role):
        if kind == "text":
            st.markdown(payload)
        else:
            render_result(payload)
            render_sync(payload)

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
