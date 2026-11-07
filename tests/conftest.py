import re
from pathlib import Path

import pytest

from fin_agent.agent.nodes.intake import heuristic_parse
from fin_agent.agent.prompts import INTAKE_SCHEMA, LENS_SCHEMA, QA_SCHEMA, RECOMMEND_SCHEMA
from fin_agent.data.providers.fixture import FixtureProvider
from fin_agent.llm.providers import FakeProvider

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    from fin_agent.agent.nodes.knowledge import reset_kb

    monkeypatch.setenv("FIN_AGENT_FIXTURES", str(FIXTURES))
    monkeypatch.delenv("VAULT_PATH", raising=False)
    monkeypatch.setenv("FIN_AGENT_KB", "off")  # never touch the real index or load a model; KB tests opt in
    reset_kb()
    yield
    reset_kb()


@pytest.fixture
def demo():
    return FixtureProvider(FIXTURES).fetch("DEMO", 5)


@pytest.fixture
def demobank():
    return FixtureProvider(FIXTURES).fetch("DEMOBANK", 5)


METRIC_LINE = re.compile(r"^(\S+\.(\w+):\S+) \| (.+?) \| ", re.M)


def scripted_handler(bad_first_lens: str | None = None, tickers=None):
    """Fake LLM: cites the first non-'data' metric it sees, quoting its display value.

    ``bad_first_lens`` makes that lens quote a wrong number on its first attempt,
    to exercise the critic retry loop.
    """
    state = {"bad_done": False}

    def handler(system, prompt, schema):
        if schema is INTAKE_SCHEMA:
            parsed = heuristic_parse(prompt)
            if tickers:
                parsed["tickers"] = tickers
            return parsed
        if schema is LENS_SCHEMA:
            lens = re.search(r"^Lens: (\w+)", prompt, re.M).group(1)
            lines = [m for m in METRIC_LINE.finditer(prompt) if "| reported" not in m.group(0)]
            mid, _, display = lines[-1].group(1), lines[-1].group(2), lines[-1].group(3)
            if lens == bad_first_lens and not state["bad_done"] and "failed review" not in prompt:
                state["bad_done"] = True
                display = "99.9%"
            return {"summary": f"{lens} summary", "findings": [
                {"claim": f"Latest value is {display}.", "metric_ids": [mid], "direction": "stable", "severity": "neutral"}]}
        if schema is RECOMMEND_SCHEMA:
            return {"call": "Hold", "confidence": "high", "thesis": "Solid growth, weakening cash conversion.",
                    "bull_case": ["a"], "base_case": ["b"], "bear_case": ["c"],
                    "key_risks": ["receivables", "leverage", "margins"], "what_would_change_view": ["CFO recovers"]}
        if schema is QA_SCHEMA:
            m = next(m for m in METRIC_LINE.finditer(prompt) if m.group(2) == "dso")
            return {"answer": f"DSO is {m.group(3)}.", "metric_ids": [m.group(1)], "data_gaps": []}
        raise AssertionError("unexpected schema")

    return handler


@pytest.fixture
def fake_llm():
    return FakeProvider(scripted_handler())
