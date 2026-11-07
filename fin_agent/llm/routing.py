"""Model routing: the cheapest model that does each agent's job well.

| node      | job                                                        | default            | effort |
|-----------|------------------------------------------------------------|--------------------|--------|
| intake    | parse a request into a small JSON object, translate terms  | claude-haiku-5-5   | low    |
| lens      | interpret ~20-60 computed metrics, 2-5 cited findings     | claude-sonnet-5-5  | medium |
| qa        | answer one question from cited metrics                     | claude-sonnet-5-5  | medium |
| recommend | weigh all lenses into one call, cases, risks (1 call/run)  | claude-opus-5-5    | medium |

Numbers never depend on the model (ADR-004: Python computes, the critic checks every
figure), so the cheaper tiers only need to reason and cite; the one synthesis call that
decides the investment view keeps the strongest model.

Resolution is ``hubkit.llm.ModelRouter`` (agent ``fin``). Override per node with
``FIN_AGENT_MODEL_<NODE>`` / ``FIN_AGENT_EFFORT_<NODE>`` (or ``HUB_MODEL_FIN_<NODE>`` /
``HUB_MODEL_<NODE>``); ``FIN_AGENT_MODEL`` sets every node not overridden individually.
"""

from __future__ import annotations

from hubkit.llm.routing import EFFORTS, ModelRouter  # noqa: F401  (EFFORTS re-exported)

ROUTES: dict[str, tuple[str, str]] = {
    "intake": ("claude-haiku-5-5", "low"),
    "lens": ("claude-sonnet-5-5", "medium"),
    "qa": ("claude-sonnet-5-5", "medium"),
    "recommend": ("claude-opus-5-5", "medium"),
}
ROUTER = ModelRouter("fin", ROUTES)


def route(node: str) -> dict:
    """``{"model": ..., "effort": ...}`` for a node, after environment overrides."""
    return ROUTER.route(node)


def table() -> list[dict]:
    """Resolved routing for display (UI, CLI)."""
    return [{"node": n, **route(n)} for n in ROUTES]
