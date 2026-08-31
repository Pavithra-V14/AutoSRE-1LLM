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
uv run uvicorn services.simulator.main:app --reload --port 8000

# 2. Control API
uv run uvicorn services.api.main:app --reload --port 8001

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

Runs the 18 scripted incidents in `services/simulator/scenarios.py` and
reports success rate and CPSO (Cost Per Successful Outcome) — CI runs this
on every push.

## License

MIT