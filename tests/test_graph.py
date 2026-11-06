from langgraph.types import Command

from fin_agent.agent.graph import build_graph
from fin_agent.agent.nodes.analysis import check_claim
from fin_agent.llm.providers import FakeProvider
from tests.conftest import scripted_handler

CFG = {"configurable": {"thread_id": "t1"}}


def test_check_claim_rules():
    by_id = {"A.roe:2024": {"value": 0.2017, "unit": "ratio"}, "A.dso:2024": {"value": 115.5, "unit": "days"}}
    assert check_claim("ROE reached 20.2% in 2024", ["A.roe:2024"], by_id) == []
    assert check_claim("ROE was about 20%", ["A.roe:2024"], by_id) == []  # rounding allowed
    assert check_claim("ROE hit 25.0%", ["A.roe:2024"], by_id)  # wrong number
    assert check_claim("DSO rose to 116 days", ["A.dso:2024"], by_id) == []
    assert check_claim("ROE is strong", ["A.nope:2024"], by_id)  # unknown id
    assert check_claim("Something", [], by_id)  # no citation


def test_full_analysis_end_to_end():
    llm = FakeProvider(scripted_handler())
    app = build_graph(llm)
    out = app.invoke({"user_input": "Analyze DEMO, quick view"}, CFG | {"configurable": {"thread_id": "e2e"}})
    assert out.get("error") is None, out.get("error")
    assert out["request"]["depth"] == "quick"
    assert set(out["findings"]) == {"profitability", "health"}
    rec = out["recommendation"]
    assert rec["call"] == "Hold" and "not licensed" in rec["disclaimer"]
    assert out["skills"]["profitability"]["sources"] == ["builtin:profitability"]


def test_valuation_assessed_when_market_data_present():
    app = build_graph(FakeProvider(scripted_handler()))
    out = app.invoke({"user_input": "Analyze DEMO"}, {"configurable": {"thread_id": "std"}})
    assert out.get("error") is None, out.get("error")
    val = out["findings"]["valuation"]
    assert not val.get("not_assessed")
    assert val["findings"]
    assert "valuation" not in (out["recommendation"].get("not_assessed") or [])
    assert out["datasets"]["DEMO"]["meta"]["market_data"]["price"] == 30000.0


def test_valuation_not_assessed_without_market_data():
    app = build_graph(FakeProvider(scripted_handler(tickers=["DEMOBANK"])))
    out = app.invoke({"user_input": "Analyze the demo bank"}, {"configurable": {"thread_id": "std-bank"}})
    assert out["findings"]["valuation"]["not_assessed"] is True
    assert "valuation" in out["recommendation"]["not_assessed"]


def test_critic_rejects_wrong_number_then_retry_fixes_it():
    llm = FakeProvider(scripted_handler(bad_first_lens="health"))
    app = build_graph(llm)
    out = app.invoke({"user_input": "Analyze DEMO quick"}, {"configurable": {"thread_id": "retry"}})
    checks = [g["check"] for g in out["guardrail_log"]]
    assert "numeric_reconciliation" in checks
    assert out["retries"] == 1
    assert out["findings"]["health"]["findings"], "retry should produce a valid finding"
    assert not out["findings"]["health"].get("low_confidence")


def test_bank_uses_bank_skills_and_metrics():
    app = build_graph(FakeProvider(scripted_handler(tickers=["DEMOBANK"])))
    out = app.invoke({"user_input": "Analyze the demo bank quickly"}, {"configurable": {"thread_id": "bank"}})
    assert out.get("error") is None, out.get("error")
    assert "NIM proxy" in out["skills"]["profitability"]["text"] or "banks" in out["skills"]["profitability"]["text"]
    assert not any(m["name"] == "gross_margin" for m in out["metrics"])


def test_qa_path_cites_metric():
    app = build_graph(FakeProvider(scripted_handler()))
    out = app.invoke({"user_input": "What is DEMO's DSO trend?"}, {"configurable": {"thread_id": "qa"}})
    assert out["request"]["intent"] == "qa"
    assert out["answer"]["metric_ids"][0].startswith("DEMO.dso:")
    assert not out["answer"].get("low_confidence")


def test_missing_ticker_interrupts_and_resumes():
    app = build_graph(FakeProvider(scripted_handler()))
    cfg = {"configurable": {"thread_id": "clarify"}}
    first = app.invoke({"user_input": "analyze this company for me"}, cfg)
    assert "__interrupt__" in first
    out = app.invoke(Command(resume="DEMO"), cfg)
    assert out["request"]["tickers"] == ["DEMO"]
    assert out.get("recommendation")


def test_unknown_ticker_stops_with_error():
    app = build_graph(FakeProvider(scripted_handler(tickers=["ZZZ"])))
    out = app.invoke({"user_input": "Analyze ZZZ"}, {"configurable": {"thread_id": "unk"}})
    assert "Could not get data for ZZZ" in out["error"]
    assert "recommendation" not in out
