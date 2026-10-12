export type Instrument = {
  symbol: string;
  name: string;
  sector: string;
};

/** Universe TrendLab paper-trades and charts. */
export const INSTRUMENTS: Instrument[] = [
  { symbol: "AAPL", name: "Apple Inc.", sector: "Technology" },
  { symbol: "MSFT", name: "Microsoft Corp.", sector: "Technology" },
  { symbol: "NVDA", name: "NVIDIA Corp.", sector: "Semiconductors" },
  { symbol: "AMD", name: "Advanced Micro Devices", sector: "Semiconductors" },
  { symbol: "JPM", name: "JPMorgan Chase", sector: "Financial services" },
  { symbol: "GS", name: "Goldman Sachs", sector: "Financial services" },
  { symbol: "XOM", name: "Exxon Mobil", sector: "Energy" },
  { symbol: "CVX", name: "Chevron", sector: "Energy" },
  { symbol: "JNJ", name: "Johnson & Johnson", sector: "Healthcare" },
  { symbol: "UNH", name: "UnitedHealth Group", sector: "Healthcare" },
  { symbol: "SPY", name: "SPDR S&P 500 ETF", sector: "Market benchmarks" },
  { symbol: "QQQ", name: "Invesco QQQ Trust", sector: "Market benchmarks" },
];

export const SYMBOLS = INSTRUMENTS.map((i) => i.symbol);

export const TIMEFRAMES = ["15m", "1h", "4h", "1d"] as const;
export type Timeframe = (typeof TIMEFRAMES)[number];

export function isTimeframe(value: string): value is Timeframe {
  return (TIMEFRAMES as readonly string[]).includes(value);
}

export function instrumentFor(symbol: string): Instrument {
  const upper = symbol.toUpperCase();
  return (
    INSTRUMENTS.find((i) => i.symbol === upper) ?? {
      symbol: upper,
      name: upper,
      sector: "Unknown",
    }
  );
}
