-- Run in the Supabase SQL editor (or any Postgres instance).

create table if not exists traces (
    id bigserial primary key,
    incident_id uuid not null,
    step_index int not null,
    step_type text not null check (step_type in ('plan', 'observation', 'synthesis')),
    tool text,
    tool_role text check (tool_role in ('required', 'preferred')),
    status text,
    provider text,              
    content jsonb not null,
    input_tokens int default 0,
    output_tokens int default 0,
    cost_usd numeric(10, 6) default 0,
    latency_ms int,
    created_at timestamptz default now()
);

create index if not exists idx_traces_incident on traces (incident_id);

create table if not exists approval_queue (
    id bigserial primary key,
    incident_id uuid not null,
    action text not null,
    blast_radius text not null check (blast_radius in ('low', 'medium', 'high')),
    confidence numeric(4, 3),
    confidence_rationale text, 
    pattern_key text,            
    denial_reason text,          
    escalated_at timestamptz,                                    
    service text,                  
    evidence jsonb,
    alternatives_considered jsonb,
    status text default 'pending' check (status in ('pending', 'approved', 'denied')),
    decided_by text,
    decided_at timestamptz,
    created_at timestamptz default now()
);

create table if not exists benchmark_runs (
    id bigserial primary key,
    run_id uuid not null,
    scenario text not null,
    success boolean,
    ground_truth_match boolean,
    total_cost_usd numeric(10, 6),
    cpso numeric(10, 6),         
    steps_taken int,
    tool_selection_f1 numeric(4, 3),  
    needed_approval boolean,          
    had_degradation boolean,         
    recovered boolean,               
    created_at timestamptz default now()
);

create table if not exists pattern_trust_events (
    id bigserial primary key,
    pattern_key text not null,
    event_type text not null check (
        event_type in ('approved', 'denied', 'benchmark_success', 'benchmark_fail', 'auto_approved')
    ),
    incident_id uuid,
    created_at timestamptz default now()
);

create index if not exists idx_pattern_trust_key on pattern_trust_events (pattern_key);

create table if not exists notifications (
    id bigserial primary key,
    incident_id uuid not null,
    action text not null,
    message text not null,
    created_at timestamptz default now()
);

create table if not exists active_incidents (
    id bigserial primary key,
    fingerprint text not null,
    incident_id uuid not null,
    created_at timestamptz default now()
);

create index if not exists idx_active_incidents_fingerprint on active_incidents (fingerprint, created_at desc);

create or replace view incident_cost_summary as
select
    incident_id,
    sum(cost_usd) as total_cost_usd,
    sum(input_tokens) as total_input_tokens,
    sum(output_tokens) as total_output_tokens,
    count(*) filter (where step_type = 'plan') as plan_steps,
    max(created_at) - min(created_at) as duration,
    max(created_at) as last_activity_at  
from traces
group by incident_id;

create or replace view pattern_trust_summary as
select
    pattern_key,
    count(*) filter (where event_type in ('approved', 'benchmark_success', 'auto_approved')) as positive_events,
    count(*) filter (where event_type in ('denied', 'benchmark_fail')) as negative_events,
    count(*) as total_events,
    round(
        count(*) filter (where event_type in ('approved', 'benchmark_success', 'auto_approved'))::numeric
        / nullif(count(*), 0),
        3
    ) as trust_score
from pattern_trust_events
group by pattern_key;
