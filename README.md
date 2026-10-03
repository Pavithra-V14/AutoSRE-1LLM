# AutoSRE — Autonomous Incident Commander

An agentic AI system that investigates production incidents across synthetic
metrics, logs, and deploy history — using a real plan → act → observe →
replan loop, not a fixed playbook. The investigation path isn't known ahead
of time, so no rule-based runbook or single-shot lookup can solve it.

The focus of this project is the **control surface** around the agent —
tool validation, guardrails, graceful degradation, human approval, and cost
accounting — since that's the layer that decides whether an agent survives
contact with production.

Full design rationale: [`docs/architecture.md`](docs/architecture.md)
Full stack rationale ("why this, not that" per layer): [`docs/tool-stack.md`](docs/tool-stack.md)

---

## Architecture

```
incident alert
     │
     ▼
[prompt guardrail: NeMo Guardrails]  ───► reject / escalate
     │
     ▼
agent plan (LangGraph state graph)
     │
     ▼
[tool validation: pydantic]  schema · policy · budget · authz
     │
     ▼
tool execution (simulator: metrics / logs / deploys / restart / rollback)
     │
     ▼
[output validator: guardrails-ai]  citation check, causal graph schema
[PII check: Presidio]
     │
     ▼
[human approval queue: Slack]  triggered on irreversible / large blast radius / low confidence
     │
     ▼
response ──► traces table (Supabase) ──► dashboard (Next.js/Vercel)
```

## Stack

| Layer | Tool |
|---|---|
| Agent orchestration | LangGraph |
| LLM (fallback chain) | Groq → Gemini → OpenRouter |
| Guardrails | Presidio (PII) + guardrails-ai (output schema) + NeMo Guardrails (scope/injection) |
| Incident simulator | FastAPI (Render free tier) |
| Control API | FastAPI — approve/deny + dashboard metrics (`services/api/`) |
| Database | Supabase (Postgres, free tier) |
| Human approval | Slack Incoming Webhook + dashboard Approve/Deny buttons |
| Dashboard | Next.js (Vercel free tier) |
| CI / benchmark | GitHub Actions |

## Repo layout

```
├── services/
│   ├── agent/          # LangGraph agent: graph, tools, guardrails, trust/correlation/dedup
│   ├── api/             # Control API (FastAPI) — dashboard approve/deny + metrics
│   └── simulator/       # FastAPI synthetic production environment
├── apps/
│   └── dashboard/       # Next.js: incidents, approvals, notifications, metrics
├── infra/
│   └── db/schema.sql    # Supabase schema
├── tools/               # operational scripts (benchmark, guardrail tests, CLI approve/deny)
├── tests/               # pytest suite for the control API
├── .github/workflows/   # benchmark, escalation, keepalive
├── docs/
│   ├── architecture.md
│   └── tool-stack.md
├── pyproject.toml
└── .env.example
```

---

## Installation

Requires [`uv`](https://docs.astral.sh/uv/) and Node.js 18+.

```bash
git clone <your-repo-url>
cd autosre

# Python deps
uv venv
uv sync
```

Copy `.env.example` to `.env` and add:
- Groq, Gemini, and OpenRouter API keys
- Supabase URL + keys (run `infra/db/schema.sql` in the Supabase SQL editor first)
- A Slack Incoming Webhook URL

## Running

Each service runs in its own terminal:

```bash
# 1. Simulator
uv run uvicorn services.simulator.main:app --reload --port 8100

# 2. Control API
uv run uvicorn services.api.main:app --reload --port 8101

# 3. Dashboard
cd apps/dashboard
npm install
cp .env.local.example .env.local   # add Supabase anon key + control API URL
npm run dev

# 4. (Optional) escalation monitor
uv run python tools/escalation_monitor.py
```

Run the agent against a scripted incident:

```bash
uv run python -m services.agent.graph --scenario db_timeout
```

## Benchmark

```bash
uv run python tools/run_benchmark.py
```

Runs the 19 scripted incidents in `services/simulator/scenarios.py` and
reports success rate and CPSO (Cost Per Successful Outcome) — CI runs this
on every push.

```bash
uv run python tools/run_benchmark.py --scenario db_timeout   # one scenario
uv run python tools/run_benchmark.py --runs 3                # each scenario 3x: pass rate, 95% CI, flaky list
uv run python tools/run_benchmark.py --no-judge              # offline: no judge LLM calls
uv run python tools/run_benchmark.py --min-success-rate 0.5  # exit 1 below threshold (CI)
```

### How runs are scored

| Scenario kind | Graded by | Pass condition |
|---|---|---|
| Causal (deploy, traffic, cascade, deadlock, ...) | **LLM judge** (`services/agent/judge.py`) | Verdict `correct`: same root cause and mechanism as `ground_truth_cause`, regardless of naming |
| `expects_low_confidence` | **Flags** (`services/agent/scoring.py`) | Reached a conclusion with confidence below `LOW_CONFIDENCE_THRESHOLD` |
| `expects_fail_closed` | **Flags** | Low confidence **and** the rationale names every tool in `degraded_tools` |
| Any run whose synthesis aborted | — | Always a failure (`aborted`) |

The report prints how each run was scored. If the judge is unreachable the run
falls back to the old word-overlap scorer and is labelled `lexical_fallback`
with a warning, so degraded grading is never silent. Judge cost is reported
separately and is not part of CPSO. Set `JUDGE_PROVIDER=gemini` to grade with a
different model family than the agent (avoids self-preference bias).

### Benchmark integrity

- **The agent only sees symptoms.** Each scenario has an `alert` (see `ALERTS` in
  `scenarios.py`) written like a real monitoring alert. The `description` field
  states the cause and is *never* sent to the agent; `tests/test_scenarios.py`
  fails if an alert contains a deploy id, copies the description, or uses cause
  vocabulary.
- **One run is not a result.** The agent is non-deterministic, so use `--runs 3`
  (or more) before concluding anything. The report gives each scenario's pass
  rate (`stable_pass` / `stable_fail` / `flaky`) and a 95% confidence interval on
  the overall rate, so a change inside the interval is treated as noise.
- **Repeatable by default.** `LLM_TEMPERATURE` and `JUDGE_TEMPERATURE` default to 0.

### Agent reliability behaviours

- **Malformed JSON** is repaired where it is structural (fences, trailing
  commas, truncation) and otherwise re-asked once with the exact parse error
  (`LLM_JSON_MAX_ATTEMPTS`). Groq and Gemini are called in JSON mode.
- **No wasted steps:** the planner may answer `{"action": "conclude"}` once it has
  evidence, calls are de-duplicated by what the tool actually honors (only
  `service` and `since`), and a planner that stalls is stopped
  (`MAX_CONSECUTIVE_NO_PROGRESS`). `AGENT_FAST_PATH_MAX_STEPS` is a ceiling,
  not a target.
- **Fail closed in code:** if deploy history or metrics were unavailable,
  confidence is capped below the human-review threshold and the gap is written
  into the rationale, whatever the model claimed. A missing log source fails
  open (disclosed, not capped).

### Untrusted tool output

Logs, metric labels and deploy descriptions are written by outside systems, so
`services/agent/untrusted.py` treats them as data, not instructions. Every
observation is redacted (secrets; PII types that cannot destroy evidence), scanned
for injection-shaped text, and defanged against forged delimiters; prompts fence
tool output in `<untrusted_tool_output>` tags with an instruction never to follow
it. A flagged observation also fails the run closed (confidence capped, human
review). The `prompt_injection_in_logs` scenario plants a hijack attempt and
fails any run that repeats the planted claim.

### Node vocabulary

Causal-graph nodes are `<kind>:<slug>` (`deploy:cfg-4471 -> mechanism:pool_exhaustion
-> symptom:latency_spike`); see `services/agent/vocabulary.py`. Near-miss
spellings are normalized, anything else is re-asked. Pattern trust is keyed on
trigger kind + first effect (`deploy/pool_exhaustion`), so two different deploys
causing the same failure build up one trust history instead of two. Trust events
recorded under the old free-text keys are not migrated.

### Unit tests (offline)

```bash
uv run python -m pytest tests -q --ignore=tests/test_control_api.py
```

## License

MIT