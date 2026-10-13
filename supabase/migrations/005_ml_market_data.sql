-- TrendLab ML + deep market-data schema (non-destructive)
-- Extends candles for extraction metadata and adds ML training tables.
-- Apply after 001–004. Does not drop existing objects.

alter table if exists candles
  add column if not exists provider_updated_at timestamptz;

alter table if exists candles
  add column if not exists updated_at timestamptz not null default now();

create index if not exists candles_symbol_tf_ts_idx
  on candles (symbol, timeframe, ts desc);

create table if not exists ingestion_runs (
  id uuid primary key default gen_random_uuid(),
  job_name text not null,
  started_at timestamptz not null default now(),
  finished_at timestamptz,
  status text not null,
  details jsonb not null default '{}'::jsonb
);

create table if not exists market_features (
  symbol text not null references instruments(symbol),
  timeframe text not null,
  ts timestamptz not null,
  feature_version text not null,
  features jsonb not null,
  created_at timestamptz not null default now(),
  primary key (symbol, timeframe, ts, feature_version)
);

create index if not exists market_features_lookup_idx
  on market_features (symbol, timeframe, feature_version, ts desc);

create table if not exists model_versions (
  model_version text primary key,
  symbol text not null references instruments(symbol),
  timeframe text not null,
  feature_version text not null,
  algorithm text not null,
  status text not null check (status in ('candidate', 'approved', 'rejected', 'retired')),
  artifact_path text,
  metrics jsonb not null default '{}'::jsonb,
  trained_at timestamptz not null,
  approved_at timestamptz,
  dataset_start timestamptz,
  dataset_end timestamptz,
  notes text
);

create index if not exists model_versions_symbol_tf_idx
  on model_versions (symbol, timeframe, status);

-- Distinct from legacy rule-based `predictions` table.
create table if not exists ml_predictions (
  id uuid primary key default gen_random_uuid(),
  symbol text not null references instruments(symbol),
  timeframe text not null,
  model_version text not null references model_versions(model_version),
  feature_version text not null,
  predicted_at timestamptz not null,
  horizon_bars integer not null,
  reference_close numeric not null,
  prob_up numeric not null,
  prob_not_up numeric not null,
  features jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (symbol, timeframe, model_version, predicted_at, horizon_bars)
);

create index if not exists ml_predictions_symbol_tf_idx
  on ml_predictions (symbol, timeframe, predicted_at desc);

create table if not exists ml_prediction_outcomes (
  prediction_id uuid primary key references ml_predictions(id),
  actual_close numeric,
  realized_return numeric,
  direction_correct boolean,
  evaluated_at timestamptz not null default now()
);

create table if not exists paper_trades (
  id uuid primary key default gen_random_uuid(),
  agent_id text references agents(id),
  symbol text not null references instruments(symbol),
  side text not null check (side in ('BUY', 'SELL')),
  qty numeric not null,
  price numeric not null,
  fee numeric not null default 0,
  slippage numeric not null default 0,
  reason text,
  timeframe text,
  ts timestamptz not null,
  model_version text,
  prediction_id uuid references ml_predictions(id),
  idempotency_key text not null unique
);

create index if not exists paper_trades_symbol_ts_idx
  on paper_trades (symbol, ts desc);
