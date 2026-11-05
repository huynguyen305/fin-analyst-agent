import pytest

from fin_agent.agent.graph import build_graph
from fin_agent.llm.base import LLMError
from fin_agent.llm.providers import FakeProvider
from fin_agent.llm.usage import PRICES, normalize, one_line, price_cost, summarize, usage_entry
from fin_agent.vault.render import render_run_log
from tests.conftest import scripted_handler


def test_price_table_cost():
    tokens = {"input_tokens": 1_000_000, "output_tokens": 100_000,
              "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1_000_000}
    inp, out, _, read = PRICES["claude-opus-5-5"]
    assert price_cost("claude-opus-5-5", tokens) == pytest.approx(inp + out / 10 + read)
    assert price_cost("some-unknown-model", tokens) is None


def test_normalize_prefers_reported_cost_and_reads_objects():
    class U:  # SDK usage objects expose attributes, not keys
        input_tokens, output_tokens = 120, 30
        cache_creation_input_tokens, cache_read_input_tokens = None, 50

    u = normalize(U(), provider="anthropic_api", model="claude-sonnet-5-5")
    assert (u["input_tokens"], u["output_tokens"], u["cache_creation_input_tokens"], u["cache_read_input_tokens"]) == (120, 30, 0, 50)
    assert u["cost_source"] == "price_table" and u["cost_usd"] > 0
    sdk = normalize({"input_tokens": 10}, provider="claude_subscription", model="x", cost_usd=0.5, cost_source="sdk_estimate")
    assert sdk["cost_usd"] == 0.5 and sdk["cost_source"] == "sdk_estimate"


def test_summarize_totals_partial_cost_and_failures():
    rows = [
        usage_entry({"input_tokens": 100, "output_tokens": 10, "cost_usd": 0.01, "cost_source": "price_table"}, "lens", "health"),
        usage_entry({"input_tokens": 50, "output_tokens": 5}, "recommend"),  # no cost known
        usage_entry(None, "qa", ok=False),  # failed before any usage was reported
    ]
    s = summarize(rows)
    assert s["calls"] == 3 and s["failed_calls"] == 1
    assert s["input_tokens"] == 150 and s["output_tokens"] == 15 and s["total_tokens"] == 165
    assert s["cost_usd"] == pytest.approx(0.01) and s["cost_partial"]
    assert s["by_node"]["lens"]["calls"] == 1
    assert "partial" in one_line(s)
    assert summarize([])["cost_usd"] is None


def test_every_llm_call_is_in_the_ledger():
    llm = FakeProvider(scripted_handler(bad_first_lens="health"), model="claude-sonnet-5-5")
    app = build_graph(llm)
    out = app.invoke({"user_input": "Analyze DEMO quick"}, {"configurable": {"thread_id": "usage"}})
    rows = out["llm_usage"]
    assert len(rows) == len(llm.calls) == out["llm_calls"]
    nodes = [r["node"] for r in rows]
    assert nodes.count("intake") == 1 and nodes.count("recommend") >= 1
    assert any(r["label"] == "health (retry)" for r in rows)  # critic re-ask is counted too
    s = summarize(rows)
    assert s["input_tokens"] > 0 and s["output_tokens"] > 0 and s["cost_usd"] > 0
    log = render_run_log(out, "usage")
    assert "## Token usage" in log and "| intake |" in log


def test_failed_call_tokens_are_still_counted():
    def handler(system, prompt, schema):
        if "lens" in prompt.lower() and "profitability" in prompt.lower():
            return "not json at all"  # model ran, output unusable
        return scripted_handler()(system, prompt, schema)

    llm = FakeProvider(handler)
    try:
        llm.complete("s", "profitability lens", {"type": "object"})
    except LLMError as exc:
        assert exc.usage and exc.usage["input_tokens"] > 0
    else:
        pytest.fail("expected LLMError")
