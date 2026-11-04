"""Smoke test of the Streamlit UI with the fake LLM (no browser)."""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

from fin_agent.llm.providers import FakeProvider  # noqa: E402
from tests.conftest import scripted_handler  # noqa: E402

UI = str(Path(__file__).resolve().parents[1] / "app" / "ui.py")


@pytest.fixture
def fake_provider(monkeypatch):
    import fin_agent.llm.providers as providers
    monkeypatch.setattr(providers, "get_provider", lambda name=None: FakeProvider(scripted_handler()))


def test_chat_runs_full_analysis(fake_provider):
    at = st_testing.AppTest.from_file(UI, default_timeout=60)
    at.run()
    at.chat_input[0].set_value("Analyze DEMO quick").run()
    assert not at.exception
    assert [m.value for m in at.metric] == ["DEMO", "Hold", "high"]


def test_missing_ticker_asks_then_resumes(fake_provider):
    at = st_testing.AppTest.from_file(UI, default_timeout=60)
    at.run()
    at.chat_input[0].set_value("analyze this company please").run()
    assert any("Which company" in m.value for m in at.markdown)
    at.chat_input[0].set_value("DEMO").run()
    assert not at.exception
    assert at.metric[1].value == "Hold"
