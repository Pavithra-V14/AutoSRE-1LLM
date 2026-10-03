-- Persist WHY a benchmark run passed or failed, not just whether it did.
-- Safe to run more than once. Apply in the Supabase SQL editor.
-- Until applied, insert_benchmark_run() falls back to saving pass/fail only.

alter table benchmark_runs add column if not exists score_method text;
alter table benchmark_runs add column if not exists score numeric(4, 3);
alter table benchmark_runs add column if not exists score_rationale text;
alter table benchmark_runs add column if not exists run_index int;

comment on column benchmark_runs.score_method is 'judge | flags | lexical | lexical_fallback | injection | aborted';
