"""Streamlit renderer for agentic UI specs (AgenticGenUI protocol: ``{"type", "props"}``).

``REGISTRY`` maps a component ``type`` to a function returning HTML, like AgenticGenUI's
``componentRegistry``; unknown types fall back to their raw JSON, as its ``AgentRenderer``
falls back to raw text. Components animate with CSS only (no scripts): items created by
the latest agent event (``seq``) enter with a fade or slide, the running step pulses,
running lenses spin, retries shake, and the progress bar grows from its previous value.
"""

from __future__ import annotations

import html
import json
from typing import Callable

CSS = """
<style>
.ga{font-family:inherit;display:flex;flex-direction:column;gap:12px;margin:4px 0 8px}
.ga *{box-sizing:border-box}
.ga-card{border:1px solid rgba(128,128,128,.28);border-radius:10px;padding:12px 14px;background:rgba(128,128,128,.05)}
.ga-enter{animation:ga-in .45s ease-out both}
@keyframes ga-in{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.ga-pbar{height:8px;border-radius:99px;background:rgba(128,128,128,.18);overflow:hidden;position:relative}
.ga-pfill{height:100%;border-radius:99px;background:#3b82f6;animation:ga-grow .6s ease-out both;width:var(--to)}
.ga-pfill.success{background:#16a34a}.ga-pfill.error{background:#dc2626}
@keyframes ga-grow{from{width:var(--from)}to{width:var(--to)}}
.ga-pbar.working::after{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.35),transparent);animation:ga-shim 1.4s infinite}
@keyframes ga-shim{from{transform:translateX(-100%)}to{transform:translateX(100%)}}
.ga-row{display:flex;justify-content:space-between;align-items:baseline;gap:8px;font-size:.85rem;margin-bottom:6px}
.ga-muted{opacity:.65;font-size:.78rem}
.ga-tl{list-style:none;margin:0;padding:0}
.ga-tl li{display:flex;gap:10px;align-items:flex-start;position:relative;padding:0 0 10px 0}
.ga-tl li:not(:last-child)::before{content:"";position:absolute;left:8px;top:18px;bottom:0;width:2px;background:rgba(128,128,128,.25)}
.ga-tl li.completed:not(:last-child)::before{background:#16a34a}
.ga-dot{flex:0 0 18px;height:18px;border-radius:50%;border:2px solid rgba(128,128,128,.45);display:flex;align-items:center;justify-content:center;font-size:11px;line-height:1;margin-top:1px}
.completed .ga-dot{background:#16a34a;border-color:#16a34a;color:#fff;animation:ga-pop .35s ease-out}
.in-progress .ga-dot{border-color:#3b82f6;box-shadow:0 0 0 0 rgba(59,130,246,.6);animation:ga-pulse 1.3s infinite}
.waiting .ga-dot{border-color:#d97706;background:rgba(217,119,6,.15);animation:ga-pulse-amber 1.6s infinite}
.skipped .ga-dot{border-style:dashed}
.skipped .ga-tlt,.pending .ga-tlt{opacity:.55}
@keyframes ga-pop{0%{transform:scale(.4)}70%{transform:scale(1.2)}100%{transform:scale(1)}}
@keyframes ga-pulse{0%{box-shadow:0 0 0 0 rgba(59,130,246,.55)}100%{box-shadow:0 0 0 9px rgba(59,130,246,0)}}
@keyframes ga-pulse-amber{0%{box-shadow:0 0 0 0 rgba(217,119,6,.55)}100%{box-shadow:0 0 0 9px rgba(217,119,6,0)}}
.ga-tlt{font-size:.88rem}.in-progress .ga-tlt{font-weight:600}
.ga-cl{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:8px;margin-top:8px}
.ga-item{border:1px solid rgba(128,128,128,.25);border-radius:8px;padding:8px 10px;font-size:.84rem;display:flex;gap:8px;align-items:flex-start}
.ga-item .ga-ic{flex:0 0 16px;height:16px;margin-top:2px}
.ga-spin{border:2px solid rgba(59,130,246,.25);border-top-color:#3b82f6;border-radius:50%;animation:ga-rot .8s linear infinite}
@keyframes ga-rot{to{transform:rotate(360deg)}}
.ga-item.done{border-color:rgba(22,163,74,.45)}.ga-item.done .ga-ic{color:#16a34a}
.ga-item.just-done{animation:ga-flash .9s ease-out}
@keyframes ga-flash{from{background:rgba(22,163,74,.22)}to{background:transparent}}
.ga-item.retrying{border-color:rgba(217,119,6,.6);animation:ga-shake .5s ease-in-out}
@keyframes ga-shake{0%,100%{transform:none}25%{transform:translateX(-3px)}75%{transform:translateX(3px)}}
.ga-item.skipped{opacity:.6;border-style:dashed}.ga-item.error{border-color:rgba(220,38,38,.6)}
.ga-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.ga-metric{border:1px solid rgba(128,128,128,.25);border-radius:10px;padding:10px 12px}
.ga-metric.ga-enter{animation:ga-flip .55s ease-out both}
@keyframes ga-flip{from{opacity:0;transform:perspective(400px) rotateX(-35deg)}to{opacity:1;transform:none}}
.ga-mv{font-size:1.25rem;font-weight:650;margin:2px 0}
.ga-up{color:#16a34a}.ga-down{color:#dc2626}
.ga-banner{border-left:4px solid #3b82f6;border-radius:6px;padding:8px 12px;background:rgba(59,130,246,.08);font-size:.86rem}
.ga-banner.success{border-color:#16a34a;background:rgba(22,163,74,.08)}
.ga-banner.warning{border-color:#d97706;background:rgba(217,119,6,.09)}
.ga-banner.error{border-color:#dc2626;background:rgba(220,38,38,.09)}
.ga-banner.muted{border-color:rgba(128,128,128,.5);background:rgba(128,128,128,.06)}
.ga-toasts{display:flex;flex-direction:column;gap:6px;align-items:flex-end}
.ga-toast{max-width:560px;border-radius:8px;padding:7px 12px;font-size:.82rem;background:rgba(217,119,6,.12);border:1px solid rgba(217,119,6,.45)}
.ga-toast.error{background:rgba(220,38,38,.1);border-color:rgba(220,38,38,.5)}
.ga-toast.new{animation:ga-slide .45s ease-out both}
.ga-toast.old{opacity:.6}
@keyframes ga-slide{from{opacity:0;transform:translateX(24px)}to{opacity:1;transform:none}}
.ga-raw{font-family:monospace;font-size:.75rem;white-space:pre-wrap;opacity:.8}
@media (prefers-reduced-motion:reduce){.ga *{animation:none!important}}
</style>
"""

ICONS = {"done": "✓", "skipped": "–", "error": "!", "pending": "○", "retrying": "↻"}


def e(x) -> str:
    return html.escape(str(x if x is not None else ""))


class Renderer:
    """Holds state between redraws (previous progress) so animations continue smoothly."""

    def __init__(self) -> None:
        self.prev_progress = 0

    def render(self, specs: list[dict], current_seq: int | None = None) -> str:
        seq = current_seq if current_seq is not None else max((s.get("props", {}).get("seq", 0) for s in specs), default=0)
        parts = [REGISTRY.get(s.get("type"), _raw)(self, s.get("props") or {}, seq) for s in specs]
        return CSS + '<div class="ga">' + "".join(parts) + "</div>"


def _enter(props: dict, seq: int) -> str:
    return " ga-enter" if props.get("seq") == seq else ""


def _progress(r: Renderer, p: dict, seq: int) -> str:
    value = int(p.get("value", 0))
    frm, r.prev_progress = r.prev_progress, value
    working = p.get("variant") == "default"
    return (f'<div class="ga-card"><div class="ga-row"><b>{e(p.get("label"))}</b><span class="ga-muted">{e(p.get("detail"))}</span></div>'
            f'<div class="ga-pbar{" working" if working else ""}"><div class="ga-pfill {e(p.get("variant", ""))}" '
            f'style="--from:{frm}%;--to:{value}%"></div></div></div>')


def _timeline(r: Renderer, p: dict, seq: int) -> str:
    rows = []
    for ev in p.get("events", []):
        status = ev.get("status", "pending")
        mark = "✓" if status == "completed" else ""
        detail = f'<div class="ga-muted">{e(ev.get("detail"))}</div>' if ev.get("detail") else ""
        rows.append(f'<li class="{e(status)}"><span class="ga-dot">{mark}</span><div><div class="ga-tlt">{e(ev.get("title"))}</div>{detail}</div></li>')
    return f'<div class="ga-card"><ul class="ga-tl">{"".join(rows)}</ul></div>'


def _checklist(r: Renderer, p: dict, seq: int) -> str:
    items = p.get("items", [])
    cells = []
    for it in items:
        st = it.get("status", "pending")
        icon = '<span class="ga-ic ga-spin"></span>' if st == "running" else f'<span class="ga-ic">{ICONS.get(st, "○")}</span>'
        just = " just-done" if st == "done" and it.get("seq") == seq else ""
        detail = f'<div class="ga-muted">{e(it.get("detail"))}</div>' if it.get("detail") else ""
        cells.append(f'<div class="ga-item {e(st)}{just}">{icon}<div><div>{e(it.get("label"))}</div>{detail}</div></div>')
    return (f'<div class="ga-card"><div class="ga-row"><b>{e(p.get("title"))}</b>'
            f'<span class="ga-muted">{int(p.get("completed", 0))}/{len(items)}</span></div><div class="ga-cl">{"".join(cells)}</div></div>')


def _metric(r: Renderer, p: dict, seq: int) -> str:
    kind = p.get("changeType")
    arrow = {"increase": '<span class="ga-up">▲</span> ', "decrease": '<span class="ga-down">▼</span> '}.get(kind, "")
    change = f'<div class="ga-muted">{arrow}{e(p.get("change"))}</div>' if p.get("change") else ""
    return (f'<div class="ga-metric{_enter(p, seq)}" title="{e(p.get("formula"))}"><div class="ga-muted">{e(p.get("title"))}</div>'
            f'<div class="ga-mv">{e(p.get("value"))}</div>{change}<div class="ga-muted">{e(p.get("description"))}</div></div>')


def _metric_grid(r: Renderer, p: dict, seq: int) -> str:
    return '<div class="ga-grid">' + "".join(_metric(r, c.get("props", {}), seq) for c in p.get("cards", [])) + "</div>"


def _banner(r: Renderer, p: dict, seq: int) -> str:
    desc = f'<div class="ga-muted">{e(p.get("description"))}</div>' if p.get("description") else ""
    return f'<div class="ga-banner {e(p.get("variant", "info"))}{_enter(p, seq)}"><b>{e(p.get("title"))}</b>{desc}</div>'


def _toasts(r: Renderer, p: dict, seq: int) -> str:
    cur = p.get("currentSeq", seq)
    rows = [f'<div class="ga-toast {e(t.get("variant", ""))} {"new" if t.get("seq") == cur else "old"}">{e(t.get("message"))}</div>'
            for t in p.get("toasts", [])]
    return '<div class="ga-toasts">' + "".join(rows) + "</div>"


def _raw(r: Renderer, p: dict, seq: int) -> str:
    return f'<div class="ga-card ga-raw">{e(json.dumps(p, ensure_ascii=False)[:2000])}</div>'


REGISTRY: dict[str, Callable[[Renderer, dict, int], str]] = {
    "ProgressBar": _progress,
    "Timeline": _timeline,
    "ChecklistWithProgress": _checklist,
    "MetricCard": _metric,
    "MetricCardGrid": _metric_grid,
    "InfoBanner": _banner,
    "ToastStack": _toasts,
}
