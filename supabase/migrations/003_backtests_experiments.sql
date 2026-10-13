-- TrendLab V2 Phase 1: durable backtest / experiment result storage (optional Supabase).
-- Python still writes local JSON under data/backtests/; sync compact rows here when configured.

create table if not exists backtest_runs (
  id text primary key,
  created_at timestamptz not null default now(),
  code_sha text,
  timeframe text not null,
  symbols text[] not null,
  benchmark_symbol text,
  start_ts timestamptz,
  end_ts timestamptz,
  bars_tested integer,
  execution_model text not null,
  risk_json jsonb not null default '{}'::jsonb,
  status text not null default 'completed'
);

create table if not exists backtest_metrics (
  run_id text not null references backtest_runs(id) on delete cascade,
  agent_id text not null,
  metrics_json jsonb not null,
  primary key (run_id, agent_id)
);

create table if not exists experiment_variants (
  id text primary key,
  created_at timestamptz not null default now(),
  hypothesis text not null,
  control_config jsonb not null,
  variant_config jsonb not null,
  decision text check (decision in ('accept', 'reject', 'pending')),
  evidence_json jsonb not null default '{}'::jsonb
);

create table if not exists prediction_outcomes (
  id bigserial primary key,
  symbol text not null,
  timeframe text not null,
  predicted_at timestamptz not null,
  direction text not null,
  confidence double precision,
  expected_move_pct double precision,
  actual_1h_pct double precision,
  actual_4h_pct double precision,
  actual_24h_pct double precision,
  scored_at timestamptz
);

create index if not exists prediction_outcomes_symbol_time_idx
  on prediction_outcomes (symbol, predicted_at desc);
