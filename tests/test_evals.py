import json

import httpx
import pytest

from fin_agent.agent.graph import build_graph
from fin_agent.evals import __main__ as evals_cli
from fin_agent.evals.jev import INPUT_PRICE_PER_M, JevClient, JevError
from fin_agent.evals.judge import evaluate
from fin_agent.llm.providers import FakeProvider
from tests.conftest import scripted_handler


def fake_jev(noul_value=0.9, choice_label=None, status=200, calls=None):
    """MockTransport answering every question; choice = choice_label or the first label."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if calls is not None:
            calls.append(body)
        if status != 200:
            return httpx.Response(status, json={"detail": "bad request"})
        answers = {}
        for key, q in body["questions"].items():
            if q["type"] == "noul":
                answers[key] = {"type": "noul", "noul": noul_value}
            elif q["type"] == "choice":
                label = choice_label if choice_label in q["criteria"] else next(iter(q["criteria"]))
                answers[key] = {"type": "choice", "choice": label, "confidence": 0.9}
            else:
                answers[key] = {"type": "score", "score": len(q["criteria"]) - 1, "confidence": 0.8}
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": answers,
                                         "usage": {"input_tokens": 1000, "output_tokens": 50}})

    return httpx.MockTransport(handler)


@pytest.fixture(scope="module")
def run_out():
    import os
    from pathlib import Path
    os.environ["FIN_AGENT_FIXTURES"] = str(Path(__file__).parent / "fixtures")
    app = build_graph(FakeProvider(scripted_handler()))
    return app.invoke({"user_input": "Analyze DEMO"}, {"configurable": {"thread_id": "eval"}})


def test_client_requires_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JevError, match="TYPESAFE_API_KEY"):
        JevClient()


def test_client_cost_and_no_retry_on_400():
    calls = []
    c = JevClient(api_key="k", transport=fake_jev(calls=calls))
    res = c.system_one("state", {"q": {"type": "noul", "instructions": "x"}})
    assert res.noul("q") == 0.9 and res.model == "jev-1.13.0"
    assert res.usage["cost_usd"] == pytest.approx(1000 * INPUT_PRICE_PER_M / 1e6)  # output tokens are free
    calls.clear()
    bad = JevClient(api_key="k", transport=fake_jev(status=422, calls=calls), retries=2)
    with pytest.raises(JevError, match="422"):
        bad.system_one("s", {"q": {"type": "noul"}})
    assert len(calls) == 1


def test_evaluate_scores_findings_and_recommendation(run_out):
    calls = []
    ev = evaluate(run_out, JevClient(api_key="k", transport=fake_jev(choice_label="Hold", calls=calls)))
    s = ev["summary"]
    n_findings = sum(len(b.get("findings") or []) for b in run_out["findings"].values() if not b.get("not_assessed"))
    assert s["findings"] == n_findings > 0
    assert s["finding_pass_rate"] == 1.0 and s["insight"] == 1.0
    assert s["call_agreement"] is True and s["recommendation"]["consistent"] == 0.9
    assert ev["usage"]["calls"] == n_findings + 2
    assert ev["usage"]["cost_usd"] == pytest.approx(ev["usage"]["calls"] * 1000 * INPUT_PRICE_PER_M / 1e6)
    # blind grading: the agent's own labels are not shown to Jev
    blind = next(c for c in calls if set(c["questions"]) == {"call"})
    assert "call" not in blind["state"] and "thesis" not in json.dumps(blind["state"])
    finding_req = next(c for c in calls if "severity" in c["questions"])
    assert "severity" not in finding_req["state"] and "direction" not in finding_req["state"]


def test_uncertain_answers_are_flagged_for_review(run_out):
    ev = evaluate(run_out, JevClient(api_key="k", transport=fake_jev(noul_value=0.5)))
    assert ev["summary"]["needs_review"]
    assert ev["summary"]["finding_pass_rate"] == 1.0  # 0.5 counts as pass but goes to review


def test_jev_errors_do_not_abort_the_eval(run_out):
    ev = evaluate(run_out, JevClient(api_key="k", transport=fake_jev(status=400), retries=0))
    assert ev["summary"]["errors"] and ev["summary"]["findings"] == 0
    assert ev["usage"]["failed_calls"] == ev["usage"]["calls"]


def test_cli_scores_saved_run(tmp_path, monkeypatch, run_out, capsys):
    saved = tmp_path / "run.json"
    saved.write_text(json.dumps(run_out, default=str), encoding="utf-8")
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    real = JevClient.__init__
    monkeypatch.setattr(JevClient, "__init__", lambda self, **kw: real(self, transport=fake_jev(noul_value=0.2)))
    out = tmp_path / "report.json"
    code = evals_cli.main(["--from", str(saved), "--out", str(out), "--min-pass", "0.8"])
    assert code == 1  # pass rate 0 < 0.8
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report[0]["eval"]["summary"]["finding_pass_rate"] == 0.0
    assert "Jev eval:" in capsys.readouterr().out
