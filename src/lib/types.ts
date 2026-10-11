export type Candle = {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export type Market = {
  symbol: string;
  name: string;
  sector: string;
  timeframe: string;
  last_close: number;
  trend_label: string;
  ema_20: number | null;
  ema_50: number | null;
  ema_200: number | null;
  rsi_14: number | null;
  adx_14: number | null;
  atr_14: number | null;
  source: string;
  as_of: string;
  candles: Candle[];
};

export type AgentRow = {
  id: string;
  name: string;
  strategy: string;
  description: string;
  cash: number;
  equity: number;
  return_pct: number;
  max_drawdown_pct: number;
  trades: number;
  open_positions: number;
  positions: Array<{
    agent_id: string;
    symbol: string;
    qty: number;
    avg_entry: number;
    stop_loss: number | null;
    opened_at: string;
  }>;
};

export type Trade = {
  id: number;
  agent_id: string;
  symbol: string;
  side: "BUY" | "SELL";
  qty: number;
  price: number;
  fee: number;
  slippage: number;
  stop_loss: number | null;
  reason: string;
  timeframe: string;
  ts: string;
};

export type Snapshot = {
  generated_at: string;
  timeframe: string;
  data_source: string;
  last_collection: string | null;
  last_simulation: string | null;
  starting_cash: number;
  totals: {
    agents: number;
    instruments: number;
    total_equity: number;
    avg_return_pct: number;
  };
  instruments: Array<{ symbol: string; name: string; sector: string }>;
  markets: Market[];
  leaderboard: AgentRow[];
  equity_curves: Record<string, Array<{ ts: string; equity: number }>>;
  trades: Trade[];
  research: {
    bullish: { matches: number; hit_rate: number | null; avg_forward_10: number | null };
    bearish: { matches: number; hit_rate: number | null; avg_forward_10: number | null };
    recent_events: Array<Record<string, unknown>>;
  };
  status: {
    budget: string;
    deployment: string;
    real_money_trading: boolean;
  };
};
