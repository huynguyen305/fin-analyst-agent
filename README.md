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

# Optional: vault knowledge base (GraphRAG). Local embeddings, nothing leaves the machine.
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -e ".[kb]"
.venv\Scripts\python -m fin_agent.knowledge index        # first run downloads the model and embeds ~3k chunks

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
        ─► retrieve knowledge (GraphRAG over the vault: per-lens concepts + this company's past reports)
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

`standard` and `360` also run the **technical** lens when daily prices are available.

\* Valuation uses the latest close and shares outstanding (vnstock 5 free endpoints `Quote.history` and `Company.overview`): share price, market cap, P/E, P/B, earnings yield, and for non-financials P/S, EV/EBIT, FCF yield and dividend yield when the statement lines exist. Multiples use the latest fiscal year (TTM for quarterly data). If the price lookup fails, valuation is reported *not assessed* rather than guessed.

### Technical analysis

The technical lens uses about 13 months of daily closes and volumes (free `Quote.history`) and the VN-Index. Every indicator is computed in Python (`fin_agent/metrics/technical.py`), and the LLM only interprets them:

| Group | Metrics |
| --- | --- |
| Trend | 50- and 200-day averages, price vs each, 50 vs 200 (golden or death-cross regime) |
| Momentum | RSI 14 (Wilder), MACD 12/26/9 histogram as % of price, returns over 1, 3, 6 and 12 months |
| Risk | 60-day annualized volatility, 12-month max drawdown, distance from the 52-week high and low (closing basis) |
| Participation | 20/60-day volume ratio |
| Relative strength | 3, 6 and 12-month return minus the VN-Index (the index comes rounded, about 0.5% precision, so no beta) |

Technicals inform timing and position size, not the quality call, and the lens never gives price targets.

### Vault knowledge (GraphRAG)

After intake parses the request (and data and skills are loaded), the **Knowledge Retriever** runs graph retrieval over the vault with [`langchain-graph-retriever`](https://docs.langchain.com/oss/python/integrations/retrievers/graph_rag). The graph is the vault's own structure, so no LLM extraction is needed:

| Edge (metadata) | Meaning |
| --- | --- |
| `links` → `note` and back | wikilinks and `same`/`related` frontmatter, both directions; `up` links and index/MOC notes are skipped as hubs |
| `tags` ↔ `tags` | shared tags, except hub tags on more than 15% of notes |
| `ticker` ↔ `ticker` | the agent's own reports and data snapshots per company |
| `fa_lens` ↔ `fa_lens` | skill notes of the same lens |

- **Traversal:** `Eager` uses 4 similarity seeds and up to 2 hops. Results are ranked seeds first, then by hop and similarity, with at most 2 chunks per note.
- **Skill notes:** already in the prompt, so they are dropped from the output, but they still seed traversal into the concept notes they link to.
- **What counts as knowledge:** lens and Q&A retrieval filter *inside the store* on reference notes (concepts, skill rules, learnings), so dated daily briefs, index notes and company reports are never seeds or traversal steps. Filtering after retrieval returned nothing when every seed was a brief.
- **Vietnamese questions:** intake also writes a short English `search_query`, and Q&A searches with it alone. The embedding model favours same-language matches, so even a mixed English+Vietnamese query drifted to the nearest Vietnamese note.
- **Per lens:** 3 chunks go into a `<knowledge>` block. The recommender also gets this company's earlier reports (`ticker` filter) to say what changed. Q&A retrieves on the question itself.
- **Safety:** retrieved text passes the injection guard. The prompt says its numbers are not metrics, and the critic still rejects any figure that doesn't match a cited metric id.
- **Store:** Chroma, persistent in `data/knowledge/`. List metadata goes through the `ShreddingTransformer`, and a small adapter makes `langchain-graph-retriever` 0.8 work with chromadb 1.x.
- **Embeddings:** `intfloat/multilingual-e5-small`, run locally, so Vietnamese questions match English notes.
- **Indexing:** incremental per chunk. Unchanged text is never re-embedded, link or tag changes rewrite metadata only, and changing the model rebuilds the index.

```powershell
python -m fin_agent.knowledge index                 # incremental; --full to rebuild
python -m fin_agent.knowledge search "cổ phiếu có quá mua không?" --ticker FPT
python -m fin_agent.knowledge stats
```

Without an index, the agent logs `knowledge_base: skip` and runs as before. The UI's *Sources and guardrails* tab lists the retrieved notes with their hop, and vault reports link them under *Method*.

### Cost and speed

Each agent uses the cheapest model that does its job (`fin_agent/llm/routing.py`):

| Agent | Model | Effort | Why |
| --- | --- | --- | --- |
| intake | Haiku 5.5 | low | parse the request into JSON, write an English search query |
| lens analysts | Sonnet 5.5 | medium | interpret computed metrics; the critic checks every number |
| Q&A | Sonnet 5.5 | medium | answer from cited metrics |
| recommender | Opus 5.5 | medium | one call per run that weighs every lens into the view |

Override per agent in `.env` with `FIN_AGENT_MODEL_<NODE>` and `FIN_AGENT_EFFORT_<NODE>` (e.g. `FIN_AGENT_MODEL_RECOMMEND=claude-sonnet-5-5`); the sidebar shows the routing.

- **CLI isolation:** subscription calls run the Claude Code CLI with no MCP servers, claude.ai connectors, plugins or skills. Before, 2-3 calls per run carried 54k-169k tokens of injected tool definitions, which was 46-83% of each run's cost. A detector flags any call whose input is far larger than what was sent; it appears as a WARNING in the usage line.
- **Critic:** a claim may quote a threshold from its lens's skill rules as a comparison ("below the 0.8x guideline"). Any other number must match a cited metric. Lens prompts forbid self-computed differences ("rose by 3,020.5 bn"); the analyst quotes both levels instead.
- **Concurrency:** the lens analysts run in parallel (as before); tickers are fetched in parallel, each ticker's statements and prices concurrently; the knowledge base loads in a background thread while intake and data fetching run, and its searches run in parallel. Non-LLM time on a warm process: 6.5 s (was 29 s).

| Run (FPT, standard depth) | LLM calls | Lens retries | Cost (SDK estimate) | Wall time |
| --- | --- | --- | --- | --- |
| Before: all Opus, injected CLI context | 9-15 | 3 | $1.97-2.87 | n/a |
| Model routing + CLI isolation | 17 | 10 | $0.67 | 110 s |
| + critic accepts quoted rule thresholds | 14 | 6 | $0.58 | 145 s |
| + prompt: no self-computed deltas | 10 | 2 | $0.41 | 105 s |

### Agent activity panel

The UI shows the run as an animated panel, using the protocol of [AgenticGenUI](https://github.com/vivek100/AgenticGenUI): `fin_agent/ui/genui.py` turns LangGraph events into component specs (`{"type", "props"}`: Timeline, ChecklistWithProgress, ProgressBar, MetricCard, InfoBanner, ToastStack), and `app/genui_render.py` maps each type to an HTML/CSS component (unknown types fall back to raw JSON). The running step pulses, lens analysts spin and show findings as they finish, critic retries shake and raise a toast, and the progress bar shows live LLM calls, cost and models. AgenticGenUI itself is a React/Next.js + CopilotKit library; the same specs could drive its `AgentRenderer` if the UI moves to React.

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
  agent/       graph.py (LangGraph), state.py, prompts.py, nodes/{intake,data,knowledge,analysis}.py
  data/        schema.py (canonical schema + DQ gate), providers/{vnstock_provider,edgar,fixture}.py
  metrics/     library.py (pure, tested ratio functions with metric ids), technical.py (price indicators)
  llm/         base.py, providers.py (Claude subscription | Anthropic API | fake), routing.py (model per agent), usage.py, langchain_adapter.py
  ui/          genui.py (graph events → AgenticGenUI-style component specs)
  knowledge/   vault_docs.py (notes → chunks + graph metadata), store.py (embeddings, Chroma, graph retrieval, incremental sync), __main__.py (CLI)
  evals/       jev.py (TypeSafe Jev client), judge.py (Jev rubric for findings/recommendation/Q&A), __main__.py (CLI)
  guards/      validators.py (Guardrails AI validators), guards.py (composed guards used by the graph)
  vault/       render.py (notes), writer.py (git worktree + PR), sync.py (run → files + learning proposals)
app/ui.py      Streamlit UI · app/genui_render.py (animated component registry)
  skills/      loader.py (vault → skill packs), defaults.py (fallback rules)
tests/         offline tests on synthetic fixtures (DEMO, DEMOBANK: invented numbers, not real issuers)
```

## Configuration (`.env`)

| Variable | Meaning |
| --- | --- |
| `FIN_AGENT_LLM` | `claude_subscription` (default) or `anthropic_api` |
| `FIN_AGENT_MODEL` | Model for every agent not overridden individually |
| `FIN_AGENT_MODEL_<NODE>` / `FIN_AGENT_EFFORT_<NODE>` | Per-agent model and effort (`INTAKE`, `LENS`, `QA`, `RECOMMEND`) |
| `VAULT_PATH` | Local clone of the Obsidian vault |
| `VNSTOCK_SOURCE` | `VCI` (default) or `TCBS` |
| `SEC_USER_AGENT` | Required by SEC EDGAR: `app-name your-email` |
| `FIN_AGENT_FIXTURES` | Directory of JSON fixtures; forces offline data |
| `FIN_AGENT_VAULT_PUSH` | `0` = commit to a local branch only, no push |
| `FIN_AGENT_KB` | `auto` (use the vault knowledge base when indexed) or `off` |
| `FIN_AGENT_EMBEDDINGS` | `hf:intfloat/multilingual-e5-small` (default) or `ollama:<model>`; changing it rebuilds the index |
| `FIN_AGENT_KB_DIR` / `FIN_AGENT_KB_INCLUDE` | Index location (default `data/knowledge`) and vault folders to index |
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
