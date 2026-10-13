-- TrendLab V2 Phase 2: market regimes + walk-forward run storage (optional Supabase).

create table if not exists market_regimes (
  symbol text not null,
  timeframe text not null,
  ts timestamptz not null,
  regime text not null,
  adx double precision,
  volatility_20 double precision,
  primary key (symbol, timeframe, ts)
);

create index if not exists market_regimes_symbol_ts_idx
  on market_regimes (symbol, timeframe, ts desc);

create table if not exists walk_forward_runs (
  id text primary key,
  created_at timestamptz not null default now(),
  code_sha text,
  timeframe text not null,
  symbols text[] not null,
  warm_up integer,
  calibrate_bars integer,
  test_bars integer,
  fold_count integer,
  risk_json jsonb not null default '{}'::jsonb,
  status text not null default 'completed'
);

create table if not exists walk_forward_metrics (
  run_id text not null references walk_forward_runs(id) on delete cascade,
  agent_id text not null,
  metrics_json jsonb not null,
  regime_json jsonb not null default '{}'::jsonb,
  primary key (run_id, agent_id)
);
