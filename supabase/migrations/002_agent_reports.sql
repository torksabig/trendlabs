-- Shared AI research reports (Qwen via Ollama on Mac). Never used for order routing.

create table if not exists agent_reports (
  id bigint generated always as identity primary key,
  symbol text not null,
  timeframe text not null,
  model text not null,
  report jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists agent_reports_symbol_created_idx
  on agent_reports (symbol, created_at desc);

alter table agent_reports enable row level security;

-- No public policies by default. Workers write with the service role key.
-- Dashboard reads should go through an authenticated Next.js server route.
