-- TrendLab schema for Supabase PostgreSQL (free tier)
-- Apply in Supabase SQL editor or via CLI when credentials are available.

create extension if not exists "pgcrypto";

create table if not exists instruments (
  symbol text primary key,
  name text not null,
  sector text not null,
  asset_class text not null default 'equity',
  active boolean not null default true
);

create table if not exists candles (
  symbol text not null references instruments(symbol),
  timeframe text not null,
  ts timestamptz not null,
  open numeric not null,
  high numeric not null,
  low numeric not null,
  close numeric not null,
  volume numeric not null,
  source text not null,
  adjusted boolean not null default true,
  created_at timestamptz not null default now(),
  primary key (symbol, timeframe, ts)
);

create index if not exists candles_symbol_tf_ts_idx
  on candles (symbol, timeframe, ts desc);

create table if not exists indicators (
  symbol text not null,
  timeframe text not null,
  ts timestamptz not null,
  ema_20 numeric,
  ema_50 numeric,
  ema_200 numeric,
  rsi_14 numeric,
  adx_14 numeric,
  atr_14 numeric,
  volatility_20 numeric,
  volume_sma_20 numeric,
  trend_label text,
  primary key (symbol, timeframe, ts)
);

create table if not exists agents (
  id text primary key,
  name text not null,
  strategy text not null,
  description text not null,
  starting_cash numeric not null default 10000,
  cash numeric not null default 10000,
  updated_at timestamptz not null default now()
);

create table if not exists positions (
  id uuid primary key default gen_random_uuid(),
  agent_id text not null references agents(id),
  symbol text not null references instruments(symbol),
  qty numeric not null,
  avg_entry numeric not null,
  stop_loss numeric,
  opened_at timestamptz not null,
  updated_at timestamptz not null default now(),
  unique (agent_id, symbol)
);

create table if not exists trades (
  id uuid primary key default gen_random_uuid(),
  agent_id text not null references agents(id),
  symbol text not null,
  side text not null check (side in ('BUY', 'SELL')),
  qty numeric not null,
  price numeric not null,
  fee numeric not null default 0,
  slippage numeric not null default 0,
  stop_loss numeric,
  reason text not null,
  timeframe text not null,
  ts timestamptz not null,
  idempotency_key text not null unique
);

create table if not exists equity_snapshots (
  id bigserial primary key,
  agent_id text not null references agents(id),
  ts timestamptz not null,
  cash numeric not null,
  equity numeric not null,
  unique (agent_id, ts)
);

create table if not exists trend_events (
  id uuid primary key default gen_random_uuid(),
  symbol text not null,
  timeframe text not null,
  detected_at timestamptz not null,
  direction text not null,
  strength numeric,
  volatility numeric,
  volume_zscore numeric,
  support numeric,
  resistance numeric,
  features jsonb not null default '{}'::jsonb,
  forward_return_5 numeric,
  forward_return_10 numeric,
  forward_return_20 numeric,
  max_favorable_excursion numeric,
  max_adverse_excursion numeric
);

create table if not exists job_runs (
  id uuid primary key default gen_random_uuid(),
  job_name text not null,
  started_at timestamptz not null default now(),
  finished_at timestamptz,
  status text not null,
  details jsonb not null default '{}'::jsonb
);

create table if not exists system_meta (
  key text primary key,
  value text not null,
  updated_at timestamptz not null default now()
);
