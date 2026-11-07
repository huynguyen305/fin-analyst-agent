import sys
from pathlib import Path

from fin_agent.agent.graph import build_graph
from fin_agent.llm.providers import FakeProvider
from fin_agent.llm.routing import ROUTES, route
from fin_agent.ui.genui import AgentActivity
from tests.conftest import scripted_handler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from genui_render import REGISTRY, Renderer  # noqa: E402


def stream(handler, text, thread):
    app = build_graph(FakeProvider(handler))
    act, history = AgentActivity(), []
    for chunk in app.stream({"user_input": text}, {"configurable": {"thread_id": thread}}, stream_mode="updates"):
        for node, update in chunk.items():
            act.update(node, update)
        history.append(act.specs())
    act.finish()
    return act, history


def by_type(specs, t):
    return [s for s in specs if s["type"] == t]


def test_full_run_produces_agentic_ui_specs():
    act, history = stream(scripted_handler(), "Analyze DEMO", "g1")
    final = act.specs()
    assert all({"type", "props"} <= set(s) for s in final)  # AgenticGenUI protocol shape
    progress = by_type(final, "ProgressBar")[0]["props"]
    assert progress["value"] == 100 and progress["label"] == "Done" and "LLM calls" in progress["detail"]
    events = by_type(final, "Timeline")[0]["props"]["events"]
    assert [e["status"] for e in events] == ["completed"] * len(events)
    lens = by_type(final, "ChecklistWithProgress")[0]["props"]
    assert lens["completed"] == len(lens["items"]) and {"valuation", "technical"} <= {i["id"] for i in lens["items"]}
    cards = by_type(final, "MetricCardGrid")[0]["props"]["cards"]
    assert {"Price", "P/E", "RSI 14"} <= {c["props"]["title"] for c in cards}
    assert any("View: Hold" in b["props"]["title"] for b in by_type(final, "InfoBanner"))
    # while running, exactly one step is in progress and progress only grows
    mid = by_type(history[1], "Timeline")[0]["props"]["events"]
    assert sum(e["status"] == "in-progress" for e in mid) == 1
    values = [by_type(h, "ProgressBar")[0]["props"]["value"] for h in history]
    assert values == sorted(values)


def test_critic_retry_shows_toast_and_retry_count():
    act, _ = stream(scripted_handler(bad_first_lens="health"), "Analyze DEMO quick", "g2")
    toasts = by_type(act.specs(), "ToastStack")[0]["props"]["toasts"]
    assert any("Critic sent health back" in t["message"] for t in toasts)
    health = next(i for i in by_type(act.specs(), "ChecklistWithProgress")[0]["props"]["items"] if i["id"] == "health")
    assert health["status"] == "done" and "1 retry" in health["detail"]


def test_qa_path_uses_qa_steps():
    act, _ = stream(scripted_handler(), "Why is DEMO's DSO rising?", "g3")
    titles = [e["id"] for e in by_type(act.specs(), "Timeline")[0]["props"]["events"]]
    assert titles[-1] == "qa" and "lens" not in titles
    assert not by_type(act.specs(), "ChecklistWithProgress")


def test_renderer_escapes_and_falls_back():
    r = Renderer()
    specs = [{"type": "InfoBanner", "props": {"title": "<script>alert(1)</script>", "seq": 1}},
             {"type": "MysteryWidget", "props": {"x": "<b>"}}]
    out = r.render(specs, current_seq=1)
    assert "<script>alert" not in out and "&lt;script&gt;" in out
    assert "ga-raw" in out and "&lt;b&gt;" in out  # unknown type -> raw JSON, escaped
    assert 'ga-enter"' in out  # created by the current event: animates in (class attribute, not the CSS)
    assert 'ga-enter"' not in Renderer().render(specs[:1], current_seq=-1)  # history: no animation


def test_progress_animates_from_previous_value():
    r = Renderer()
    r.render([{"type": "ProgressBar", "props": {"value": 25, "variant": "default"}}])
    out = r.render([{"type": "ProgressBar", "props": {"value": 60, "variant": "default"}}])
    assert "--from:25%;--to:60%" in out and "working" in out
    assert set(REGISTRY) >= {"Timeline", "ChecklistWithProgress", "MetricCard", "ToastStack"}


def test_renderer_theme_class():
    spec = [{"type": "InfoBanner", "props": {"title": "x"}}]
    assert '<div class="ga ga-dark">' in Renderer(theme="dark").render(spec)
    assert '<div class="ga ga-light">' in Renderer(theme="light").render(spec)
    assert '<div class="ga">' in Renderer().render(spec)  # unknown: CSS follows prefers-color-scheme
    assert '<div class="ga">' in Renderer(theme="neon").render(spec)  # invalid values are ignored


def test_model_routing_defaults_and_overrides(monkeypatch):
    for node in ROUTES:
        monkeypatch.delenv(f"FIN_AGENT_MODEL_{node.upper()}", raising=False)
    monkeypatch.delenv("FIN_AGENT_MODEL", raising=False)
    assert route("intake") == {"model": "claude-haiku-5-5", "effort": "low"}
    assert route("lens")["model"] == "claude-sonnet-5-5" and route("recommend")["model"] == "claude-opus-5-5"
    monkeypatch.setenv("FIN_AGENT_MODEL_RECOMMEND", "claude-sonnet-5-5")
    assert route("recommend")["model"] == "claude-sonnet-5-5"
    monkeypatch.setenv("FIN_AGENT_MODEL", "claude-opus-5-5")
    assert route("lens")["model"] == "claude-opus-5-5" and route("recommend")["model"] == "claude-sonnet-5-5"


def test_each_node_calls_its_routed_model(monkeypatch):
    for node in ROUTES:
        monkeypatch.delenv(f"FIN_AGENT_MODEL_{node.upper()}", raising=False)
    monkeypatch.delenv("FIN_AGENT_MODEL", raising=False)
    llm = FakeProvider(scripted_handler())
    build_graph(llm).invoke({"user_input": "Analyze DEMO quick"}, {"configurable": {"thread_id": "route"}})
    by_schema = {}
    from fin_agent.agent.prompts import INTAKE_SCHEMA, LENS_SCHEMA, RECOMMEND_SCHEMA
    for c in llm.calls:
        by_schema.setdefault(id(c["schema"]), set()).add((c["model"], c["effort"]))
    assert by_schema[id(INTAKE_SCHEMA)] == {("claude-haiku-5-5", "low")}
    assert by_schema[id(LENS_SCHEMA)] == {("claude-sonnet-5-5", "medium")}
    assert by_schema[id(RECOMMEND_SCHEMA)] == {("claude-opus-5-5", "medium")}


def test_overhead_detector_and_strict_schema():
    from fin_agent.llm.base import strict_schema
    from fin_agent.llm.providers import expected_tokens, flag_overhead
    from fin_agent.llm.usage import summarize, usage_entry
    exp = expected_tokens("system", "x" * 9000, {"type": "object"})
    normal = flag_overhead({"cache_creation_input_tokens": exp}, exp)
    injected = flag_overhead({"cache_creation_input_tokens": 169_200}, exp)
    assert "overhead_tokens" not in normal and injected["overhead_tokens"] > 150_000
    s = summarize([usage_entry(normal, "lens"), usage_entry(injected, "lens")])
    assert s["overhead_calls"] == 1
    sch = strict_schema({"type": "object", "properties": {"a": {"type": "array", "items": {"type": "object", "properties": {}}}}})
    assert sch["additionalProperties"] is False and sch["properties"]["a"]["items"]["additionalProperties"] is False


def test_api_request_body_uses_structured_outputs_effort_and_caching():
    from fin_agent.llm.providers import AnthropicAPIProvider
    p = AnthropicAPIProvider.__new__(AnthropicAPIProvider)  # no client or key needed to build the body
    p.model, p.max_tokens = None, 4096
    body = p.request("sys", "prompt", {"type": "object", "properties": {"x": {"type": "string"}}}, "claude-haiku-5-5", "low")
    assert body["model"] == "claude-haiku-5-5" and body["cache_control"] == {"type": "ephemeral"}
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["schema"]["additionalProperties"] is False


def test_subscription_sdk_errors_become_llm_errors(monkeypatch):
    import pytest
    import claude_agent_sdk
    from fin_agent.llm.base import LLMError
    from fin_agent.llm.providers import ClaudeSubscriptionProvider

    class ResultError(Exception):
        pass

    async def boom(**kwargs):
        raise ResultError("You've hit your 5-hour limit · resets 8:30am")
        yield  # pragma: no cover

    monkeypatch.setattr(claude_agent_sdk, "query", boom)
    with pytest.raises(LLMError, match="usage limit reached"):
        ClaudeSubscriptionProvider().complete("s", "p", {"type": "object"}, model="claude-haiku-5-5")
