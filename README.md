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
| 4 | Valuation from market prices; Jev agent evals (`python -m fin_agent.evals`); 3-ticker HOSE pilot | 🚧 |

## Quick start (Windows PowerShell)

```powershell
git clone https://github.com/huynguyen305/fin-analyst-agent.git
cd fin-analyst-agent
python -m venv .venv
.venv\Scripts\pip install -e ".[dev,ui]"
# HOSE/HNX/UPCoM data: vnstock 5 is only on the vendor's index (vnstock/vnai left PyPI in 2026-11)
.venv\Scripts\pip install --extra-index-url https://vnstocks.com/api/packages vnstock
copy .env.example .env      # then edit VAULT_PATH, SEC_USER_AGENT, VNSTOCK_API_KEY

# Uses your Claude subscription through the Claude Code CLI login (run `claude` once to sign in)
.venv\Scripts\python -m fin_agent "Phân tích HPG 5 năm, toàn diện"
.venv\Scripts\python -m fin_agent "Why are FPT's receivables growing faster than revenue?"

# Local UI (http://localhost:8501), foreground
.venv\Scripts\streamlit run app/ui.py
# ...or as a hidden background server that outlives the terminal (logs in logs/), and to stop it
powershell -ExecutionPolicy Bypass -File scripts\start_ui.ps1     # add -NoBrowser to skip opening a tab
powershell -ExecutionPolicy Bypass -File scripts\stop_ui.ps1
powershell -ExecutionPolicy Bypass -File scripts\restart_ui.ps1   # stop + start; use after code or .env changes
# Shortcuts: point a Desktop shortcut at start_ui.ps1 (and one at restart_ui.ps1); put one with -NoBrowser in shell:startup to run it at logon

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

\* Valuation uses the latest close and shares outstanding (vnstock 5 free endpoints `Quote.history` and `Company.overview`): share price, market cap, P/E, P/B, earnings yield, and for non-financials P/S, EV/EBIT, FCF yield and dividend yield when the statement lines exist. Multiples use the latest fiscal year (TTM for quarterly data). If the price lookup fails, valuation is reported *not assessed* rather than guessed.

### Evaluating the agent with Jev

`python -m fin_agent.evals` scores agent output with [TypeSafe Jev](https://docs.typesafe.ai/api), a calibrated classifier that is far cheaper than an LLM judge (about $0.042 per million input tokens; output tokens free). The critic already checks the numbers. Jev checks the judgement, one request per finding plus two for the recommendation:

| Item | Questions | Metric |
| --- | --- | --- |
| Finding | interpretation supported by the figures (yes/no probability); trend and severity, asked without showing the agent's labels; insight level (scored scale) | support, pass rate, direction and severity agreement, insight |
| Recommendation | call chosen from the lens findings alone (agent's call not shown); consistent, balanced, gaps disclosed, risks specific | call agreement, rationale scores |
| Q&A | answers the question, grounded in cited metrics | answer quality |

Answers with a probability between 0.3 and 0.7, or an option chosen with confidence below 0.6, are listed under *needs human review*. The report prints the agent's LLM cost next to the Jev evaluation cost and is saved to `logs/evals/`.

```powershell
python -m fin_agent.evals "Analyze DEMO" "Analyze the demo bank" --fixtures   # run + score
python -m fin_agent "Analyze FPT" --json > run.json; python -m fin_agent.evals --from run.json --min-pass 0.8
```

## Layout

```
fin_agent/
  agent/       graph.py (LangGraph), state.py, prompts.py, nodes/{intake,data,analysis}.py
  data/        schema.py (canonical schema + DQ gate), providers/{vnstock_provider,edgar,fixture}.py
  metrics/     library.py (pure, tested ratio functions with metric ids)
  llm/         base.py, providers.py (Claude subscription | Anthropic API | fake), langchain_adapter.py
  evals/       jev.py (TypeSafe Jev client), judge.py (Jev rubric for findings/recommendation/Q&A), __main__.py (CLI)
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
| `TYPESAFE_API_KEY` | Jev key for `python -m fin_agent.evals` |
| `JEV_MODEL` | Jev model; pin e.g. `jev-1.13.0` when tuning thresholds (default `jev-latest`) |

## Known limits and things to verify

- **vnstock is quarantined on PyPI.** As of 2026-11-04 PyPI marks `vnstock` as *quarantined* (PyPI does this while a malware or security report is investigated), and `vnstock3` cannot install because its dependency `vnai` was removed. Do not install either from another source until the reason is clear. The HOSE provider is being replaced by a dependency-free one; run `python scripts/probe_vn_sources.py FPT` to see which public sources answer from your network.
- **Not tested against live market data yet.** The build machine had no access to VN broker APIs or SEC EDGAR. Live runs happen on your PC, and the step 4 pilot will verify them.
- **Claude subscription.** Works through the Claude Agent SDK and your local Claude Code login (personal, local use, within your plan's rate limits). Large eval sweeps may need `anthropic_api`.
- **Banks.** NPL and CAR usually sit in the notes, not the main statements, so they show as missing unless a source provides them. vnstock 5's free tier also omits customer deposits (LDR and deposit growth come out n/a) and blocks the notes endpoint; CAR is in no vnstock tier.
- **Free-tier statements are summaries.** Borrowings, capex and dividends lines are missing for non-financials, so EV/EBIT, FCF yield and dividend yield need a paid vnstock tier.
- **Jev rubric not yet calibrated on real runs.** Thresholds (pass 0.5, review band 0.3-0.7) are starting points; compare against a small hand-labelled set before using `--min-pass` as a gate.

This is analysis, not licensed financial advice.
