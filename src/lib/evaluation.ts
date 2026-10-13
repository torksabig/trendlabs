import { promises as fs } from "fs";
import path from "path";

export type RiskMetrics = {
  total_return_pct?: number;
  sharpe?: number | null;
  mean_fold_sharpe?: number | null;
  max_drawdown_pct?: number;
  trade_count?: number;
  alpha_pct?: number | null;
};

export type RiskRow = {
  agent_id: string;
  name: string;
  metrics: RiskMetrics;
};

export type RiskBoardPayload = {
  source: string;
  rows: RiskRow[];
};

async function readJson(relPaths: string[]) {
  for (const rel of relPaths) {
    try {
      const file = path.join(process.cwd(), rel);
      const raw = await fs.readFile(file, "utf8");
      return JSON.parse(raw) as Record<string, unknown>;
    } catch {
      // try next
    }
  }
  return null;
}

export async function loadRiskBoard(): Promise<RiskBoardPayload | null> {
  const wf = await readJson([
    "public/data/walk_forward/latest.json",
    "data/walk_forward/latest.json",
  ]);
  if (wf?.agents && Array.isArray(wf.agents) && wf.agents.length) {
    return {
      source: `Walk-forward OOS · ${String(wf.fold_count ?? "?")} folds`,
      rows: (wf.agents as RiskRow[]).map((a) => ({
        agent_id: a.agent_id,
        name: a.name,
        metrics: a.metrics || {},
      })),
    };
  }

  const bt = await readJson([
    "public/data/backtests/latest.json",
    "data/backtests/latest.json",
  ]);
  if (bt?.agents && Array.isArray(bt.agents) && bt.agents.length) {
    return {
      source: "Full-period backtest",
      rows: (bt.agents as RiskRow[]).map((a) => ({
        agent_id: a.agent_id,
        name: a.name,
        metrics: a.metrics || {},
      })),
    };
  }
  return null;
}
