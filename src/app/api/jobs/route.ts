import { spawn } from "child_process";
import path from "path";
import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";
// Hobby / anonymous deploy plans cap serverless maxDuration at 60s.
export const maxDuration = 60;

const ALLOWED = new Set(["seed", "simulate", "research", "collect"]);
const WORKFLOW_FILE = "trendlabs.yml";

function pythonBin(): string {
  // Keep this env-driven so the bundler does not try to resolve .venv symlinks.
  return process.env.TRENDLAB_PYTHON || "python3";
}

function githubToken(): string | undefined {
  return process.env.GH_JOB_TOKEN || process.env.GITHUB_TOKEN || undefined;
}

function githubRepository(): string | undefined {
  return process.env.GITHUB_REPOSITORY || undefined;
}

function shouldUseGitHubActions(): boolean {
  if (process.env.TRENDLAB_ALLOW_JOBS === "0") return false;
  if (process.env.TRENDLAB_JOB_MODE === "local") return false;
  if (process.env.TRENDLAB_JOB_MODE === "github_actions") return true;
  // Prefer Actions whenever credentials are present (Vercel production path).
  if (githubToken() && githubRepository()) return true;
  // On Vercel without credentials, fall through so we can return a clear config error.
  return Boolean(process.env.VERCEL);
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

async function dispatchGitHubActions(opts: {
  job: string;
  symbol: string;
  timeframe: string;
}): Promise<{ ok: boolean; run_url?: string; error?: string; status?: number }> {
  const token = githubToken();
  const repo = githubRepository();
  if (!token || !repo) {
    return {
      ok: false,
      status: 503,
      error:
        "GitHub Actions job mode needs GITHUB_TOKEN (or GH_JOB_TOKEN) and GITHUB_REPOSITORY set on the server.",
    };
  }

  const [owner, name] = repo.split("/");
  if (!owner || !name) {
    return {
      ok: false,
      status: 500,
      error: "GITHUB_REPOSITORY must be owner/repo (e.g. torksabig/trendlabs).",
    };
  }

  const ref = process.env.GITHUB_REF || process.env.VERCEL_GIT_COMMIT_REF || "main";
  const url = `https://api.github.com/repos/${owner}/${name}/actions/workflows/${WORKFLOW_FILE}/dispatches`;

  const res = await fetch(url, {
    method: "POST",
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      "X-GitHub-Api-Version": "2022-11-28",
      "Content-Type": "application/json",
      "User-Agent": "trendlabs-jobs",
    },
    body: JSON.stringify({
      ref,
      inputs: {
        job: opts.job,
        symbol: opts.symbol,
        timeframe: opts.timeframe,
      },
    }),
  });

  if (res.status === 204) {
    return {
      ok: true,
      run_url: `https://github.com/${owner}/${name}/actions/workflows/${WORKFLOW_FILE}`,
    };
  }

  const bodyText = await res.text().catch(() => "");
  let detail = bodyText.slice(0, 500);
  try {
    const parsed = JSON.parse(bodyText) as { message?: string };
    if (parsed.message) detail = parsed.message;
  } catch {
    // keep raw text
  }

  return {
    ok: false,
    status: res.status >= 400 ? res.status : 502,
    error: `GitHub workflow dispatch failed (${res.status}): ${detail || res.statusText}`,
  };
}

function workerArgs(job: string, symbol: string, timeframe: string): { moduleName: string; args: string[] } {
  if (job === "seed") {
    return {
      moduleName: "trendlabs.seed",
      args: ["--timeframe", timeframe, "--research-symbol", symbol, "--no-ollama"],
    };
  }
  if (job === "collect") {
    return {
      moduleName: "trendlabs.collect",
      args: ["--timeframe", timeframe, "--symbols", symbol],
    };
  }
  if (job === "research") {
    return {
      moduleName: "trendlabs.research",
      args: ["--symbol", symbol, "--timeframe", timeframe, "--no-ollama"],
    };
  }
  return {
    moduleName: "trendlabs.simulate",
    args: ["--timeframe", timeframe],
  };
}

export async function POST(request: NextRequest) {
  if (process.env.TRENDLAB_ALLOW_JOBS === "0") {
    return NextResponse.json(
      { ok: false, error: "Job controls are disabled (TRENDLAB_ALLOW_JOBS=0)." },
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

  if (shouldUseGitHubActions()) {
    const dispatched = await dispatchGitHubActions({ job, symbol, timeframe });
    if (!dispatched.ok) {
      return NextResponse.json(
        {
          ok: false,
          mode: "github_actions",
          job,
          symbol,
          timeframe,
          error: dispatched.error,
        },
        { status: dispatched.status || 502 },
      );
    }
    return NextResponse.json({
      ok: true,
      mode: "github_actions",
      job,
      symbol,
      timeframe,
      run_url: dispatched.run_url,
      message:
        "Job queued on GitHub Actions. The dashboard refreshes after the workflow commits an updated snapshot and Vercel redeploys.",
    });
  }

  const { moduleName, args } = workerArgs(job, symbol, timeframe);
  const result = await runWorker(moduleName, args);
  return NextResponse.json({
    ok: result.code === 0,
    mode: "local",
    job,
    symbol,
    timeframe,
    code: result.code,
    output: result.output.slice(-4000),
  });
}
