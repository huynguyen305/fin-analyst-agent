"""Agentic generative UI: LangGraph events -> component specs (AgenticGenUI protocol).

Follows the protocol of AgenticGenUI (github.com/vivek100/AgenticGenUI, MIT): the
agent side emits JSON component specs ``{"type": <component name>, "props": {...}}``
and a renderer looks each ``type`` up in a component registry, falling back to raw
text for unknown types. Component names and props follow that library's catalogue
(Timeline, ChecklistWithProgress, ProgressBar, MetricCard, InfoBanner, ToastStack).

This module is framework-free and has no UI dependency: ``AgentActivity`` consumes
``graph.stream(..., stream_mode="updates")`` events and returns the specs to draw.
Streamlit renders them in ``app/genui_render.py``; a React front end using
AgenticGenUI's ``AgentRenderer`` could render the same JSON unchanged.

``updates`` events only say a node *finished*; the running step is inferred from the
pipeline order (the lens analysts run as one parallel fan-out).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ANALYSIS_STEPS = [
    ("intake", "Understanding the request"),
    ("gather", "Fetching statements and prices"),
    ("quantify", "Computing metrics in Python"),
    ("load_skills", "Loading skill packs"),
    ("retrieve", "Retrieving vault knowledge"),
    ("lens", "Lens analysts (parallel)"),
    ("critic", "Checking every number"),
    ("recommend", "Forming the view"),
]
QA_STEPS = ANALYSIS_STEPS[:5] + [("qa", "Answering the question")]
REVIEW_STEPS = [("review", "Waiting for your review"), ("sync", "Writing to the vault")]

# headline metrics shown as cards when present (first match per slot)
CARD_SLOTS = [
    ("Price", ["share_price"]),
    ("P/E", ["pe"]),
    ("P/B", ["pb"]),
    ("ROE", ["roe"]),
    ("RSI 14", ["rsi_14"]),
    ("12-month return", ["return_12m"]),
]
SEVERITY_ICON = {"positive": "▲", "neutral": "●", "watch": "◆", "red_flag": "✖"}


def spec(type_: str, **props: Any) -> dict:
    return {"type": type_, "props": props}


@dataclass
class AgentActivity:
    """Accumulates graph events into a live picture of what the agent is doing."""

    intent: str = "full_analysis"
    lenses: list[str] = field(default_factory=list)
    done: list[str] = field(default_factory=list)  # finished steps, in order
    lens_status: dict[str, dict] = field(default_factory=dict)  # lens -> {status, findings, retries}
    banners: list[dict] = field(default_factory=list)
    toasts: list[dict] = field(default_factory=list)
    cards: list[dict] = field(default_factory=list)
    usage: list[dict] = field(default_factory=list)
    finished: bool = False
    waiting: bool = False
    error: str | None = None
    seq: int = 0  # bumps on every event; items created in the latest event animate in

    # ---------------------------------------------------------------- events
    def update(self, node: str, payload: Any) -> None:
        self.seq += 1
        upd = payload if isinstance(payload, dict) else {}
        if node == "__interrupt__":
            self.waiting = True
            return
        self.usage += upd.get("llm_usage") or []
        if upd.get("error"):
            self.error = upd["error"]
        handler = getattr(self, f"_on_{node}", None)
        if handler:
            handler(upd)
        if node not in self.done and node != "lens":
            self.done.append(node)

    def _on_intake(self, upd: dict) -> None:
        req = upd.get("request") or {}
        if not req.get("tickers"):
            return
        from fin_agent.agent.state import DEPTH_LENSES  # local import keeps this module UI-light

        self.intent = req.get("intent", "full_analysis")
        self.lenses = [] if self.intent == "qa" else list(DEPTH_LENSES.get(req.get("depth", "standard"), []))
        self.lens_status = {l: {"status": "pending", "findings": 0, "retries": 0} for l in self.lenses}
        what = req.get("question") if self.intent == "qa" else f"{req.get('depth', '')} analysis"
        self._banner("info", f"{', '.join(req['tickers'])} · {what}",
                     f"{req.get('market', '')} · {req.get('years', '')} years · language {req.get('language', 'en')}"
                     + (f" · search: “{req['search_query']}”" if req.get("search_query") else ""))

    def _on_gather(self, upd: dict) -> None:
        for ticker, ds in (upd.get("datasets") or {}).items():
            meta = ds.get("meta", {})
            md = meta.get("market_data") or {}
            periods = len({r["period_end"] for r in ds.get("records", [])})
            price = f" · last close {md['price']:,.0f} {meta.get('currency', '')} ({md.get('price_date', '')})" if md else ""
            self._banner("success", f"{ticker}: {periods} periods from {meta.get('source', '?')}",
                         f"sector {meta.get('sector', '?')}{price}"
                         + (f" · {len(md.get('history') or [])} daily bars" if md.get("history") else " · no price history"))
        for w in upd.get("dq_warnings") or []:
            self._toast("warning", f"Data check {w.get('period', '')}: {w.get('message', '')}")

    def _on_quantify(self, upd: dict) -> None:
        metrics = upd.get("metrics") or []
        latest: dict[str, dict] = {}
        for m in metrics:
            if m["lens"] == "data":
                continue
            if m["name"] not in latest or m["period"] > latest[m["name"]]["period"]:
                latest[m["name"]] = m
        prior = {}
        for m in metrics:  # previous period, for the change arrow
            cur = latest.get(m["name"])
            if cur and m["period"] < cur["period"] and (m["name"] not in prior or m["period"] > prior[m["name"]]["period"]):
                prior[m["name"]] = m
        self.cards = []
        for title, names in CARD_SLOTS:
            m = next((latest[n] for n in names if n in latest), None)
            if not m:
                continue
            p = prior.get(m["name"])
            change, kind = "", "neutral"
            if p and p["value"]:
                delta = m["value"] - p["value"]
                kind = "increase" if delta > 0 else "decrease" if delta < 0 else "neutral"
                change = f"vs {p['period'][:4]}: {p['display']}"
            self.cards.append(spec("MetricCard", title=title, value=m["display"], change=change, changeType=kind,
                                   description=f"{m['period']} · {m['formula'][:60]}", seq=self.seq))
        self._banner("info", f"{len(metrics)} figures computed", "every number the agent may cite comes from here")

    def _on_retrieve(self, upd: dict) -> None:
        kn = upd.get("knowledge") or {}
        for l in self.lens_status.values():  # the fan-out starts right after retrieval
            l["status"] = "running"
        if kn.get("status") != "ok":
            self._banner("muted", "Vault knowledge skipped", kn.get("status", ""))
            return
        hits = [h for hs in (kn.get("lens") or {}).values() for h in hs]
        notes = sorted({h["path"].rsplit("/", 1)[-1].removesuffix(".md") for h in hits})
        hops = sum(1 for h in hits if h.get("depth", 0) > 0)
        self._banner("info", f"Vault knowledge: {len(hits)} passages from {len(notes)} notes",
                     f"{hops} reached through links or tags · " + ", ".join(notes[:4]) + (" …" if len(notes) > 4 else "")
                     + (f" · {len(kn.get('company') or [])} earlier reports" if kn.get("company") else ""))

    def _on_lens(self, upd: dict) -> None:
        for lens, out in (upd.get("findings") or {}).items():
            st = self.lens_status.setdefault(lens, {"status": "pending", "findings": 0, "retries": 0})
            if st["status"] == "retrying":
                st["retries"] += 1
            st["findings"] = len(out.get("findings") or [])
            st["status"] = "skipped" if out.get("not_assessed") else "error" if out.get("error") else "done"
            st["severity"] = [f.get("severity") for f in out.get("findings") or []]
            st["seq"] = self.seq
        if all(s["status"] in ("done", "skipped", "error") for s in self.lens_status.values()) and "lens" not in self.done:
            self.done.append("lens")

    def _on_critic(self, upd: dict) -> None:
        critic = upd.get("critic") or {}
        retry = critic.get("retry") or []
        for lens in retry:
            n = len((critic.get("issues") or {}).get(lens, []))
            self._toast("warning", f"Critic sent {lens} back: {n} claim(s) did not match the computed numbers")
            self.lens_status.setdefault(lens, {"findings": 0, "retries": 0})["status"] = "retrying"
        if retry and "lens" in self.done:  # lenses run again; the timeline shows that step in progress
            self.done.remove("lens")
        for g in upd.get("guardrail_log") or []:
            if g.get("action") == "ship_low_confidence":
                self._toast("error", f"Shipped with low confidence: {g.get('detail', '')}")

    def _on_recommend(self, upd: dict) -> None:
        rec = upd.get("recommendation") or {}
        if rec.get("call"):
            self._banner("success", f"View: {rec['call']} ({rec.get('confidence', '?')} confidence)", rec.get("thesis", "")[:220])

    def _on_qa(self, upd: dict) -> None:
        ans = upd.get("answer") or {}
        if ans.get("answer"):
            self._banner("success", "Answer ready", f"{len(ans.get('metric_ids') or [])} metrics cited")

    def _banner(self, variant: str, title: str, description: str = "") -> None:
        self.banners.append(spec("InfoBanner", variant=variant, title=title, description=description, seq=self.seq))

    def _toast(self, variant: str, message: str) -> None:
        self.toasts.append({"variant": variant, "message": message, "seq": self.seq})

    def finish(self, waiting: bool = False) -> None:
        self.seq += 1
        self.finished, self.waiting = not waiting, waiting

    # ---------------------------------------------------------------- specs
    def steps(self) -> list[tuple[str, str]]:
        base = QA_STEPS if self.intent == "qa" else ANALYSIS_STEPS
        extra = [s for s in REVIEW_STEPS if s[0] in self.done or self.waiting]
        return base + extra

    def specs(self) -> list[dict]:
        steps = self.steps()
        running_set = False
        events = []
        for key, label in steps:
            if key in self.done:
                status = "completed"
            elif self.error or (self.finished and not self.waiting):
                status = "skipped" if self.finished else "pending"
            elif not running_set:
                status, running_set = ("waiting" if self.waiting and key == "review" else "in-progress"), True
            else:
                status = "pending"
            events.append({"id": key, "title": label, "status": status, "detail": self._step_detail(key)})
        done = sum(1 for e in events if e["status"] == "completed")
        cost = sum(r.get("cost_usd") or 0 for r in self.usage)
        models = sorted({(r.get("model") or "").replace("claude-", "") for r in self.usage if r.get("model")})
        out = [spec("ProgressBar", value=round(100 * done / max(1, len(events))),
                    label="Error" if self.error else "Done" if self.finished else "Waiting for you" if self.waiting else "Working",
                    detail=f"{len(self.usage)} LLM calls · ${cost:.3f} est." + (f" · {', '.join(models)}" if models else ""),
                    variant="error" if self.error else "success" if self.finished else "default"),
               spec("Timeline", events=events)]
        if self.lens_status:
            items = [{"id": l, "label": l.replace("_", " "), "status": s["status"],
                      "detail": self._lens_detail(s), "seq": s.get("seq", 0)} for l, s in self.lens_status.items()]
            out.append(spec("ChecklistWithProgress", title="Lens analysts", items=items,
                            completed=sum(1 for i in items if i["status"] in ("done", "skipped", "error"))))
        if self.cards:
            out.append(spec("MetricCardGrid", cards=self.cards))
        out += self.banners
        if self.toasts:
            out.append(spec("ToastStack", toasts=self.toasts[-4:], currentSeq=self.seq))
        if self.error:
            out.append(spec("InfoBanner", variant="error", title="Run stopped", description=self.error[:300], seq=self.seq))
        return out

    def _step_detail(self, key: str) -> str:
        rows = [r for r in self.usage if r.get("node") == key]
        if not rows:
            return ""
        cost = sum(r.get("cost_usd") or 0 for r in rows)
        model = (rows[-1].get("model") or "").replace("claude-", "")
        return f"{len(rows)} call{'s' if len(rows) > 1 else ''} · {model} · ${cost:.3f}"

    @staticmethod
    def _lens_detail(s: dict) -> str:
        if s["status"] == "skipped":
            return "not assessed (no data)"
        if s["status"] in ("pending", "running", "retrying"):
            return "re-checking numbers" if s["status"] == "retrying" else ""
        icons = " ".join(SEVERITY_ICON.get(x, "•") for x in s.get("severity") or [])
        return f"{s['findings']} findings {icons}" + (f" · {s['retries']} retry" if s.get("retries") else "")
