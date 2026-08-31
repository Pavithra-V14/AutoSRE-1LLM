# Master Tool Stack — Autonomous Incident Commander
*Every layer, final tool choices,and rationale — single reference*

---

## Full stack at a glance

| # | Layer | Tool | Cost | Setup needed |
|---|---|---|---|---|
| 1 | Agent orchestration | **LangGraph** | Free (OSS) | `pip install langgraph langchain-core` |
| 2 | LLM inference | **Groq → Gemini → OpenRouter** (fallback chain) | Free, rate-limited | 3 free accounts, no card |
| 3 | Guardrails | **Presidio + guardrails-ai + NeMo Guardrails** | Free (OSS) | `pip install presidio-analyzer presidio-anonymizer guardrails-ai nemoguardrails` |
| 4 | Incident simulator | **FastAPI on Render (free web service)** | Free | Render account, connect GitHub repo |
| 5 | Database / traces | **Supabase (Postgres)** | Free (permanent tier) | Supabase account, 1 project |
| 6 | Human approval | **Slack Incoming Webhook + Slash Command** | Free | Slack workspace, webhook URL |
| 7 | Observability dashboard | **Next.js on Vercel** | Free | Vercel account, connect GitHub repo |
| 8 | CI / benchmark runner | **GitHub Actions** | Free (public repo) | Already have GitHub |
| 9 | Tool validation | **Hand-rolled (pydantic)** | Free | `pip install pydantic` |

---

## Layer-by-layer detail

### 1 — Agent orchestration: LangGraph
**What it does:** manages the plan → act → observe → replan loop as an explicit state graph, and routes between your three LLM providers.
**Why not a hand-rolled loop:** with 3 LLM providers, 3 guardrail libraries, and a human-approval branch, you now have real conditional branching — LangGraph's graph model makes that legible and debuggable. A plain `while` loop is fine for one provider; it gets messy fast with fallback chains.
**Why not CrewAI/AutoGen:** those are multi-agent-first frameworks. You're single-agent (per your earlier decision tree) — they'd add orchestration weight for a problem you don't have yet.

### 2 — LLM inference: Groq → Gemini → OpenRouter
**What it does:** primary planner (Groq, fastest), automatic fallback to Gemini (largest context, most generous daily quota) or OpenRouter (broadest model variety) when the primary is rate-limited.

| Provider | Free limit (verified Aug 2026) | Role |
|---|---|---|
| Groq | openai/gpt-oss-120b: 30 RPM / 1,000 RPD / 12,000 TPM | Primary — speed |
| Gemini 2.5 Flash | 1,500 RPD / 15 RPM / 1M TPM, 1M-token context | Fallback — volume + long-context |
| OpenRouter | openrouter/free / 20 RPM / 50–1,000 RPD across 28+ free models | Second fallback — model diversity |

**Why a chain, not one provider:** this is your strongest single design decision — it makes your agent's *own* inference layer exhibit the same graceful-degradation pattern you designed for tools (load-bearing vs. decorative), which is a rare, senior-sounding detail to walk an interviewer through.

### 3 — Guardrails: Presidio + guardrails-ai + NeMo Guardrails
**What each does, and why they don't overlap:**
- **Presidio** — PII detection/redaction on any log or metric text the agent ingests.
- **guardrails-ai** — validates the agent's *output shape*: enforces that every root-cause claim cites a real `tool_call_id`, rejects malformed causal-graph JSON.
- **NeMo Guardrails** — controls *conversational scope*: blocks prompt injection from alert text, keeps the agent from being steered off-task.
**Why three libraries instead of one:** each solves a structurally different problem (data leakage vs. output correctness vs. injection/scope). One general-purpose framework would mean re-implementing what the other two already do well.

### 4 — Incident simulator: FastAPI on Render
**What it does:** serves synthetic metrics/logs/deploy-history endpoints and supports scripted "chaos injection" (bad deploy, DB timeout, traffic spike).
**Free-tier reality:** 512 MB RAM / 0.1 CPU, spins down after 15 min idle, ~30–60s cold start, 750 shared hours/month.
**Mitigation:** a free GitHub Actions cron ping every 10 minutes keeps it warm before a live demo — costs nothing.
**Why not Vercel Functions:** serverless functions don't hold persistent state between calls, which breaks a simulator that needs to "remember" an injected incident across multiple agent tool calls.

### 5 — Database: Supabase
**What it does:** stores traces (every plan step, tool call, observation, cost, latency), the causal graph, and the approval queue.
**Why Supabase over Render's free Postgres:** Render's free database expires 30 days after creation — it will silently delete your project data. Supabase's free tier is permanent (with a 500 MB cap and a pause after 7 days of no API traffic — same free-cron mitigation applies).
**Why Supabase over Neon:** Supabase bundles auth in the same free project, useful if you build a login-gated approval dashboard later. Neon is the better pick only if you specifically want database branching, which isn't essential here.

### 6 — Human approval: Slack
**What it does:** posts an approval request (with evidence, alternatives considered, confidence score) when the agent's action is irreversible, high blast-radius, or low-confidence; a Slack command records the human's decision back into the trace table.
**Why Slack over a custom dashboard for MVP:** it's realistic (this is how PagerDuty/incident.io-class tools actually do HITL) and a fraction of the build time. Build a custom approval UI later as a polish pass, not for the MVP.

### 7 — Dashboard: Next.js on Vercel
**What it does:** renders the reasoning trail, causal graph, cost (CPSO) dashboard, and load-bearing/decorative tool disclosures — this is your demo centerpiece.
**Why not a paid observability vendor:** using one would directly undercut your own differentiation pitch (you're positioning against vendor-locked, opaque tools). A custom dashboard is also the only way to show your project's unique concepts — no off-the-shelf tool renders "load-bearing vs decorative" disclosures.

### 8 — CI: GitHub Actions
**What it does:** runs your 15–20 synthetic-incident benchmark on every commit, tracking success rate and CPSO over time — this is your "eval & regression" story, and it's free on public repos.

### 9 — Tool validation: pydantic (hand-rolled)
**What it does:** schema/type/range validation on every tool call before execution, plus the `tool_role` (`required`/`preferred`) declaration used for graceful degradation.
**Why hand-rolled, not a library:** this is deliberately small and specific to your tool set — a general validation library adds nothing a few pydantic models don't already give you for free.

---

## What to build yourself vs. what to wire together

| Build yourself (your differentiation) | Use as-is (undifferentiated, don't reinvent) |
|---|---|
| Agent loop logic inside LangGraph | Database (Supabase) |
| Incident simulator + chaos injection | Hosting (Render, Vercel) |
| Causal-graph construction | LLM inference itself (Groq/Gemini/OpenRouter) |
| Tool-role (load-bearing/decorative) declaration | PII detection internals (Presidio) |
| CPSO cost tracking | Approval messaging (Slack API) |
| Trace/reasoning-trail dashboard UI | |
