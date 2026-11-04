# fin-analyst-agent

An agentic **Expert Financial Analyst** for listed companies: HOSE first, with US firms as a fallback. You type a request in plain language. The agent fetches the financial statements, computes every figure deterministically, runs lens analysts in parallel, checks each claim against the computed numbers, and returns a recommendation with its risks.

The design, decisions and roadmap live in the Obsidian vault (`MyObsidian/Projects/FinAnalystAgent/`). This repo holds only the code and runtime.

> **Core rule (ADR-004): LLMs reason, Python counts.** The model never computes numbers. It cites `metric_id`s, and the critic rejects any figure in the narrative that does not match a computed metric.

## Status

| Step | Scope | State |
| --- | --- | --- |
| 1 | Design spec in the vault | ✅ in review |
| 2 | Data layer, metrics, LLM adapter, LangGraph agents, CLI | ✅ |
| 3 | Guardrails AI validators, review checkpoint + vault PR writer, Streamlit UI | ✅ this version |
| 4 | ADK evals, 3-ticker HOSE pilot | ⏳ |

## Quick start (Windows PowerShell)

```powershell
git clone https://github.com/huynguyen305/fin-analyst-agent.git
cd fin-analyst-agent
python -m venv .venv
.venv\Scripts\pip install -e ".[dev,ui]"
copy .env.example .env      # then edit VAULT_PATH, SEC_USER_AGENT

# Uses your Claude subscription through the Claude Code CLI login (run `claude` once to sign in)
.venv\Scripts\python -m fin_agent "Phân tích HPG 5 năm, toàn diện"
.venv\Scripts\python -m fin_agent "Why are FPT's receivables growing faster than revenue?"

# Local UI (http://localhost:8501)
.venv\Scripts\streamlit run app/ui.py

# Review the result, then write it to the vault as a branch + PR (needs VAULT_PATH; `gh` CLI opens the PR, else you get a compare link)
.venv\Scripts\python -m fin_agent "Analyze FPT, 360" --sync

# Offline demo on a synthetic company (no market data needed)
$env:FIN_AGENT_FIXTURES="tests/fixtures"; .venv\Scripts\python -m fin_agent "Analyze DEMO, standard depth"

.venv\Scripts\python -m pytest -q
```

On macOS or Linux, use `make install`, `make test` and `make demo`.

## How it works

```
request ─► intake (NL → ticker, period, depth; asks you if unclear)
        ─► gather (vnstock / SEC EDGAR → canonical schema → data-quality gate)
        ─► quantify (metrics library: DuPont, CCC, accruals, FCF cover, bank NIM proxy/LDR/NPL…)
        ─► load skills (vault Analyst_Skills notes, built-in fallback)
        ─► lens analysts in parallel ─► critic (claim ↔ number check, max 2 retries) ─► recommender
        └► (questions) analyst Q&A ─► same number check
        ─► (with --sync / UI toggle) human review ─► vault writer (temp git worktree → branch → PR, never main)
```

### Guardrails (Guardrails AI validators in `fin_agent/guards/`)

| Layer | Checks |
| --- | --- |
| Input | prompt injection (EN/VI) stripped, PII redacted, missing ticker → asks you, period capped |
| Process | data-quality gate, critic retries max 2, LLM-call budget per depth, human review before any vault write |
| Output | numeric reconciliation, direction vs metric polarity, no guarantees (EN/VI), risk balance, disclaimer |
| Vault | path allowlist, secret scan, wikilink check, PR only |

Guardrails AI's remote telemetry is switched off in code (`OTEL_SDK_DISABLED`, `enable_metrics=False`) so prompts never leave your machine except to Claude.

### What gets written to the vault

Report, statements snapshot (`.md` + `.csv`) and company hub under `Intelligence/Investment_Analysis/Company_Analysis/<TICKER>/`; Q&A under `Analyst_QA/`; run log and learning proposals under `Projects/FinAnalystAgent/`. Skills are read from `Intelligence/Investment_Analysis/Analyst_Skills/` (sector notes override generic ones).

| Depth | Lenses | History |
| --- | --- | --- |
| `quick` | profitability, health | 3y |
| `standard` | + operational, cash-flow quality, valuation* | 5y |
| `360` | + risk and governance | 8y |

\* Valuation reports *not assessed* until a market-price provider is added. The agent says so instead of guessing.

## Layout

```
fin_agent/
  agent/       graph.py (LangGraph), state.py, prompts.py, nodes/{intake,data,analysis}.py
  data/        schema.py (canonical schema + DQ gate), providers/{vnstock_provider,edgar,fixture}.py
  metrics/     library.py (pure, tested ratio functions with metric ids)
  llm/         base.py, providers.py (Claude subscription | Anthropic API | fake), langchain_adapter.py
  guards/      validators.py (Guardrails AI validators), guards.py (composed guards used by the graph)
  vault/       render.py (notes), writer.py (git worktree + PR), sync.py (run → files + learning proposals)
app/ui.py      Streamlit UI
  skills/      loader.py (vault → skill packs), defaults.py (fallback rules)
tests/         offline tests on synthetic fixtures (DEMO, DEMOBANK: invented numbers, not real issuers)
```

## Configuration (`.env`)

| Variable | Meaning |
| --- | --- |
| `FIN_AGENT_LLM` | `claude_subscription` (default) or `anthropic_api` |
| `FIN_AGENT_MODEL` | Optional model override |
| `VAULT_PATH` | Local clone of the Obsidian vault |
| `VNSTOCK_SOURCE` | `VCI` (default) or `TCBS` |
| `SEC_USER_AGENT` | Required by SEC EDGAR: `app-name your-email` |
| `FIN_AGENT_FIXTURES` | Directory of JSON fixtures; forces offline data |
| `FIN_AGENT_VAULT_PUSH` | `0` = commit to a local branch only, no push |

## Known limits and things to verify

- **vnstock is quarantined on PyPI.** As of 2026-11-04 PyPI marks `vnstock` as *quarantined* (PyPI does this while a malware or security report is investigated), and `vnstock3` cannot install because its dependency `vnai` was removed. Do not install either from another source until the reason is clear. The HOSE provider is being replaced by a dependency-free one; run `python scripts/probe_vn_sources.py FPT` to see which public sources answer from your network.
- **Not tested against live market data yet.** The build machine had no access to VN broker APIs or SEC EDGAR. Live runs happen on your PC, and the step 4 pilot will verify them.
- **Claude subscription.** Works through the Claude Agent SDK and your local Claude Code login (personal, local use, within your plan's rate limits). Large eval sweeps may need `anthropic_api`.
- **Banks.** NPL and CAR usually sit in the notes, not the main statements, so they show as missing unless a source provides them.

This is analysis, not licensed financial advice.
