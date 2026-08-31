# Autonomous Incident Commander — Reference Architecture
*Enterprise-grade agentic AI*

---

## 0. The one decision that comes before architecture

Per the "wrong question" framework: don't ask *RAG vs Agent vs Fine-tune — which is best*. Ask **what kind of failure am I trying to control**, per component of the system.

| Component of your project | Verb | Right architecture | Why |
|---|---|---|---|
| "What does our runbook say about DB timeouts?" | Retrieve | **RAG** | Knowledge lives in changing external docs |
| "Investigate this incident, decide what to check next, take action" | Decide / Act | **Agent** | Knowledge lives in live tools; path is unpredictable |
| "Always format the postmortem in our exact house style" | Behave | **Fine-tune** (skip for this project — not worth the cost at your scale) | Stable behavior, not knowledge |

---

## 1. The control surface (this is the actual product)

> "The agent is not the product. The control surface is." — demos showcase the agent; production runs on the surface around it.

```
incident alert
     │
     ▼
[prompt guardrail] ──────────────► reject / escalate
     │
     ▼
agent plan (hypothesis + next tool to call)
     │
     ▼
[tool validation]  schema · policy(role×action) · budget · authz
     │
     ▼
tool execution (metrics / logs / deploys / restart / rollback)
     │
     ▼
[output validator + guardrail]  fabrication check, PII, compliance
     │
     ▼
[human approval queue]  triggered if: irreversible ∧ large blast radius ∧ low confidence
     │
     ▼
response ──► observability log (trace, cost, latency, causal graph)
```

Every layer below maps directly to a feature you can point at in a demo and defend in an interview.

---

## 2. Architecture layers → features → free-tier tools

### Layer 1 — Tool validation (pre-flight, before any call executes)
**Features to build:**
- Parameter/schema validation on every tool call (types, ranges, enums)
- Policy check: role × action (e.g., "read-only investigator" can't call `rollback_deploy`)
- Budget check: per-incident and per-day call/cost ceiling
- Row-level authz: agent can only act on the service it's investigating, not the whole fleet

**Tools (free tier):**
- `pydantic` for schema validation (local, free)
- A simple JSON policy file (`role → allowed_actions`) — no paid policy engine needed at this scale
- SQLite counter for budget tracking

---

### Layer 2 — Guardrails
**Features to build:**
- Prompt-injection guard on incoming alert text / Slack input
- Output guard: block fabricated root causes not backed by a cited tool observation
- PII redaction on any log data passed through the agent
- Compliance rule: never auto-execute an action tagged `data-destructive`

**Tools (free tier):**
- Simple regex/keyword + a cheap LLM-based classifier call for injection/PII detection (Claude Haiku is cheap enough to run as a guard model)
- A "citation validator": every claim in the final report must reference a `tool_call_id` — reject/flag if not

---

### Layer 3 — Observability (you cannot operate what you cannot see)
**Features to build:**
- Structured per-incident trace (unique trace ID, every step logged)
- Immutable tool-call log — this **is** your postmortem source data
- Token + cost telemetry per incident
- Stage-level latency (investigation phase, remediation phase)
- Failure tracking by class, with alert thresholds

**Tools (free tier):**
- SQLite/Supabase free tier table: `traces(trace_id, step, tool, input, output, tokens, cost, latency_ms, ts)`
- A simple `/dashboard` route in your Next.js frontend reading straight from that table — no paid observability vendor needed for a portfolio project

---

### Layer 4 — Human approval
**Features to build:**
- Approval trigger fires on **any one** of: irreversible action, large blast radius, low agent confidence
- Approval queue entry carries full context: evidence, alternative hypotheses considered, confidence score
- Human decision feeds back into the trace log (this becomes your "training data" narrative even though you're not literally fine-tuning)

**Tools (free tier):**
- Slack Incoming Webhook + Slash Command (free) for the approve/deny UI — realistic, zero infra
- Store the decision back in the same traces table

---

## 3. Cost model — build this in, it's your strongest differentiator

Most portfolio agent projects report **$/call**. Real systems report **CPSO — Cost Per Successful Outcome**:

```
CPSO = (inference + tool-call fanout + retry/replan overhead) / success_rate
```

Why this matters for your project specifically: an agent that's 4¢/call but wrong 20% of the time (needing a costly "senior correction" — in your case, a human re-investigation) is *more* expensive than one at 12¢/call but wrong 2% of the time. Compute and report **both** numbers for your synthetic incident benchmark. This single metric is the most senior-sounding thing you can put on a resume line, because it shows you know a raw token-cost number is misleading.

**Concretely, log for every synthetic incident run:**
- `$ spent` (sum of all inference + tool costs for that trace)
- `success: bool` (did the causal graph match the injected ground-truth cause?)
- Then `CPSO = mean($ spent) / success_rate` across your 15–20 scenario benchmark

**Two levers to actually lower it** (mention both in your README as "future work" even if you only implement one):
- Numerator: compress context, cache repeated tool results within an incident, route cheap-model-first then escalate to a stronger model only if confidence is low
- Denominator: the self-critique/verification step (already in your Tier-1 list) — raising success rate lowers CPSO more reliably than cutting tokens

---

## 4. Single-agent vs multi-agent — decide with data, not vibes

The PDF's numbers are a good sanity check: a 4-agent planner/worker split ran **2,600 tokens / 6–8s latency** vs a single agent with tool selection at **900 tokens / 2–3s** for a comparable task — multi-agent was **~3x more expensive and 2x slower** for no accuracy gain in that example.

**Decision tree for your project:**
```
Q1: Are the investigation steps truly parallel (metrics + logs + deploys
    can be fetched independently before reasoning about them)?
      NO  → single agent
      YES → continue

Q2: Does specializing sub-agents (a "metrics agent", "logs agent") actually
    improve hypothesis quality, or just add orchestration?
      NO  → single agent
      YES → continue

Q3: Does the benefit exceed the orchestration tax (state management,
    retry coordination, debugging complexity, cost multiplier)?
      NO  → single agent
      YES → multi-agent
```

---

## 5. Graceful degradation — load-bearing vs decorative tools

This is the one piece from the PDF that almost no portfolio project has, and it's cheap to build:

> A tool call returning `200 OK` with an empty or stale body is **not a failure** to a health check — but it can be a lie to the user. The mesh sees green. The agent must not.

**Feature to build:** at planning time (not inside an error handler), the agent declares each tool call's role for *this specific investigation step*:
- `REQUIRED` (load-bearing) — e.g., the deploy-history tool when checking "was this caused by a recent deploy." If unreachable → **fail closed**: agent explicitly states "cannot confirm/deny deploy correlation, deploy-history tool unavailable" rather than guessing.
- `PREFERRED` (decorative) — e.g., an enrichment lookup that improves confidence but isn't required. If unreachable → **fail open**: continue, but disclose in the trace/output that this signal was skipped.


**Implementation :** a simple `tool_role` field set by the planner LLM call itself as part of its plan output (`{"tool": "deploy_history", "role": "required"}`), checked in your existing tool-validation layer.

---

## 6. Full stack

| Layer | Tool | Why free-tier fits |
|---|---|---|
| Agent orchestration | Claude API, plain Python loop (or LangGraph free tier) | No framework tax needed for single-agent MVP |
| Guard model | Claude Haiku for injection/PII/fabrication checks | Cheap enough to call on every step |
| Incident simulator | FastAPI on Render/Railway free tier | Serves synthetic metrics/logs/deploys, scriptable chaos injection |
| Storage / traces | Supabase or Neon Postgres free tier (or SQLite for pure local dev) | Structured traces, cost telemetry, approval queue |
| Human approval UI | Slack Incoming Webhook + Slash Command (free) | Realistic HITL without building a custom UI |
| Frontend / dashboard | Next.js on Vercel free tier | Reasoning trail viewer, causal graph, cost dashboard |
| CI | GitHub Actions free tier | Runs your 15–20 scenario benchmark on every commit — this is your "eval & regression" story |

---

## 7. Why this differs from what's already out there 

| Gap in existing tools (per prior research) | What you build instead |
|---|---|
| Correlation, not causation | Structured causal graph, evidence-cited |
| Investigation ships without a visible trust boundary | Full replayable reasoning trail + load-bearing/decorative tool disclosure |
| Vendor lock-in (Datadog-only, PagerDuty-only) | Thin adapter interface, swappable backend |
| No cost accountability beyond $/call | CPSO reported on a real benchmark |
| Multi-agent by default | Single-agent baseline, multi-agent only if data justifies it |

Most AI-SRE tools optimize for demo-impressiveness — unrestricted tool calls, no budget ceiling, no disclosed confidence. I built the control surface first: validated tool calls, graceful degradation between load-bearing and decorative signals, a human-approval queue gated on blast radius and confidence, and CPSO-based cost accounting on a 20-scenario benchmark — the layer that's the actual difference between a demo and something that survives 24 hours in production."
