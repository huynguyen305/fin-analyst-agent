import subprocess
from pathlib import Path

import pytest
from langgraph.types import Command

from fin_agent.agent.graph import build_graph
from fin_agent.llm.providers import FakeProvider
from fin_agent.vault.writer import VaultSyncError, VaultWriter, check_paths, scan_secrets
from tests.conftest import scripted_handler

SEED = {
    "Intelligence/Investment_Analysis/_INDEX-IA-Fund-Tech-Behav-Portfolio.md": "# IA index\n",
    "Projects/FinAnalystAgent/_INDEX-FinAnalystAgent-Design-Guardrails-Evals.md": "# hub\n",
}


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def vault(tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    v = tmp_path / "vault"
    git(tmp_path, "init", "-q", "-b", "main", str(v))
    for p, c in SEED.items():
        (v / p).parent.mkdir(parents=True, exist_ok=True)
        (v / p).write_text(c)
    git(v, "add", ".")
    git(v, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "seed")
    git(v, "remote", "add", "origin", str(remote))
    git(v, "push", "-q", "origin", "main")
    monkeypatch.setenv("VAULT_PATH", str(v))
    monkeypatch.setattr("shutil.which", lambda name: None)  # no gh CLI in tests
    return v, remote


def files_on(remote, branch):
    return set(git(remote, "ls-tree", "-r", "--name-only", branch).split())


def test_approved_run_syncs_report_snapshot_hub_and_log(vault):
    v, remote = vault
    app = build_graph(FakeProvider(scripted_handler()))
    cfg = {"configurable": {"thread_id": "sync1"}}
    first = app.invoke({"user_input": "Analyze DEMO quick", "sync": True, "run_id": "r1"}, cfg)
    assert first["__interrupt__"][0].value["type"] == "review"
    out = app.invoke(Command(resume={"action": "approve", "feedback": ""}), cfg)
    res = out["vault_sync"]
    assert res["pushed"] and res["branch"].startswith("fa/") and res["branch"].endswith("-r1")
    names = files_on(remote, res["branch"])
    day = res["branch"][3:13]
    base = "Intelligence/Investment_Analysis/Company_Analysis/DEMO"
    assert f"{base}/{day}_quick.md" in names
    assert f"{base}/Data/{day}_statements.csv" in names
    assert f"{base}/_INDEX-DEMO-Reports-Data.md" in names
    assert f"Projects/FinAnalystAgent/Runs/{day}_r1.md" in names
    assert res["warnings"] == [], res["warnings"]  # every generated wikilink resolves
    report = git(remote, "show", f"{res['branch']}:{base}/{day}_quick.md")
    assert "fa_call: Hold" in report and "not licensed" in report
    assert git(v, "status", "--porcelain") == ""  # the user's checkout is untouched
    assert "fa/" not in git(v, "branch")  # pushed branch is removed locally


def test_rejected_run_writes_log_and_learning_only(vault):
    v, remote = vault
    app = build_graph(FakeProvider(scripted_handler()))
    cfg = {"configurable": {"thread_id": "sync2"}}
    app.invoke({"user_input": "Analyze DEMO quick", "sync": True, "run_id": "r2"}, cfg)
    out = app.invoke(Command(resume="reject: ROE story ignores the buyback"), cfg)
    names = files_on(remote, out["vault_sync"]["branch"])
    assert not any("Company_Analysis" in n for n in names)
    learning = [n for n in names if "/Learnings/" in n and "human-correction" in n]
    assert learning and "Projects/FinAnalystAgent/Learnings/_Review-Queue.md" in names


def test_qa_sync_appends_daily_qa_note(vault):
    v, remote = vault
    app = build_graph(FakeProvider(scripted_handler()))
    cfg = {"configurable": {"thread_id": "sync3"}}
    app.invoke({"user_input": "What is DEMO's DSO?", "sync": True, "run_id": "r3"}, cfg)
    out = app.invoke(Command(resume="approve"), cfg)
    names = files_on(remote, out["vault_sync"]["branch"])
    assert any(n.startswith("Intelligence/Investment_Analysis/Analyst_QA/") for n in names)


def test_path_allowlist_and_secret_scan():
    with pytest.raises(VaultSyncError):
        check_paths({"Home.md": "x"})
    with pytest.raises(VaultSyncError):
        check_paths({"Intelligence/Investment_Analysis/Company_Analysis/../../Home.md": "x"})
    with pytest.raises(VaultSyncError):
        scan_secrets({"a.md": "key sk-ant-abcdefghijklmnop"})


def test_writer_requires_git_repo(tmp_path):
    with pytest.raises(VaultSyncError):
        VaultWriter(tmp_path)
