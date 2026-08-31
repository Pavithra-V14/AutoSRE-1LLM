# Dashboard

Next.js 15 (App Router, TypeScript, Tailwind) dashboard that reads directly
from Supabase.

## Routes

| Route | Shows |
|---|---|
| `/` | Incident list, cost summary per incident |
| `/incident/[id]` | Reasoning trail (every plan/observation/synthesis step, with load-bearing/decorative disclosure callouts), causal graph, cost panel |

## Setup

```bash
cd dashboard
npm install
cp .env.local.example .env.local   # fill in your Supabase project URL + anon key
npm run dev
```

Uses the Supabase **anon/public key only** — this is a read-only dashboard.
Never put the service role key in a `NEXT_PUBLIC_*` env var.

## Deploy

Push to Vercel (free tier), set the project root directory to `dashboard/`,
and add the same two env vars in the Vercel project settings:

- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`

## Notable implementation choices

- **Causal graph** (`components/CausalGraph.tsx`) — a hand-rolled
  left-to-right SVG layout, not a graph library. This project's causal
  chains are effectively always linear sequences, so a force-directed
  layout library would add weight without any visual benefit.
- **Trace viewer** (`components/TraceViewer.tsx`) — gives the loudest visual
  treatment to the `disclosure` field (load-bearing/decorative tool
  degradation), since that's the most differentiated data in the project:
  a bordered red/amber callout, not just another line of JSON.
- **No Google Fonts import** — the default `create-next-app` template pulls
  Geist from Google Fonts at build time; removed to avoid an external
  network dependency at build time and keep the app fully self-contained.