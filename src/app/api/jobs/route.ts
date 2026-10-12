import { spawn } from "child_process";
import path from "path";
import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";
// Hobby / anonymous deploy plans cap serverless maxDuration at 60s.
export const maxDuration = 60;

const ALLOWED = new Set(["seed", "simulate", "research", "collect"]);

function pythonBin(): string {
  // Keep this env-driven so the bundler does not try to resolve .venv symlinks.
  return process.env.TRENDLAB_PYTHON || "python3";
}

function runWorker(moduleName: string, args: string[]): Promise<{ code: number; output: string }> {
  const root = process.cwd();
  const cwd = path.join(root, "workers");
  const env = {
    ...process.env,
    PYTHONPATH: cwd,
  };

  return new Promise((resolve) => {
    const child = spawn(/* turbopackIgnore: true */ pythonBin(), ["-m", moduleName, ...args], {
      cwd,
      env,
    });
    let output = "";
    child.stdout.on("data", (chunk) => {
      output += String(chunk);
    });
    child.stderr.on("data", (chunk) => {
      output += String(chunk);
    });
    child.on("close", (code) => resolve({ code: code ?? 1, output }));
    child.on("error", (err) => resolve({ code: 1, output: String(err) }));
  });
}

export async function POST(request: NextRequest) {
  // Local interactive controls (Trend Arena style). Disabled on Vercel by default.
  if (process.env.VERCEL || process.env.TRENDLAB_ALLOW_JOBS === "0") {
    return NextResponse.json(
      {
        ok: false,
        error:
          "Job controls run only in local Next.js + Python worker mode. On Vercel, refresh the committed snapshot via GitHub Actions or npm run seed locally.",
      },
      { status: 403 },
    );
  }

  const body = (await request.json().catch(() => ({}))) as {
    job?: string;
    symbol?: string;
    timeframe?: string;
  };
  const job = body.job || "simulate";
  if (!ALLOWED.has(job)) {
    return NextResponse.json({ ok: false, error: "Unknown job" }, { status: 400 });
  }

  const symbol = (body.symbol || "AAPL").toUpperCase();
  const timeframe = body.timeframe || "1h";

  let moduleName = "trendlabs.simulate";
  let args: string[] = ["--timeframe", timeframe];

  if (job === "seed") {
    moduleName = "trendlabs.seed";
    args = ["--timeframe", timeframe, "--research-symbol", symbol, "--no-ollama"];
  } else if (job === "collect") {
    moduleName = "trendlabs.collect";
    args = ["--timeframe", timeframe, "--symbols", symbol];
  } else if (job === "research") {
    moduleName = "trendlabs.research";
    args = ["--symbol", symbol, "--timeframe", timeframe, "--no-ollama"];
  } else if (job === "simulate") {
    moduleName = "trendlabs.simulate";
    args = ["--timeframe", timeframe];
  }

  const result = await runWorker(moduleName, args);
  return NextResponse.json({
    ok: result.code === 0,
    job,
    symbol,
    timeframe,
    code: result.code,
    output: result.output.slice(-4000),
  });
}
