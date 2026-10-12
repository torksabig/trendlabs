# TrendLab production checklist

One-time setup to run the dashboard on Vercel and workers on GitHub Actions.

## Links

| Resource | URL |
| --- | --- |
| GitHub repo | https://github.com/torksabig/trendlabs |
| Temporary Vercel preview | https://temporary-instant-boron-2zl7waq.vercel.app (claim within ~60 minutes — see below) |
| Claim deployment | https://vercel.com/claim-deployment?code=8ab2712f-6aab-4dce-b55d-d5acb64a8fce |

After you claim the deployment and connect the GitHub repo, Vercel will assign a stable production URL (typically `https://<project>.vercel.app`).

## 1. GitHub (code + Actions)

- [x] Application source is in `torksabig/trendlabs` on branch `main` (large/binary files: run `GITHUB_TOKEN=… python3 scripts/push_to_github.py package-lock.json public/data/snapshot.json data/snapshots/latest.json src/app/favicon.ico` if anything is missing).
- [ ] Add **Repository secrets** (Settings → Secrets and variables → Actions) for optional data/LLM:
  - `ALPACA_API_KEY`, `ALPACA_API_SECRET` (optional; Yahoo fallback works without them)
  - `GOOGLE_AI_API_KEY` / `GEMINI_API_KEY`, `GROQ_API_KEY`, etc. (optional; research can use `fallback`)

**Run a worker manually:** Actions → **TrendLab scheduled research** → **Run workflow** → choose job (`simulate`, `research`, `seed`, …).

## 2. Vercel (dashboard)

1. Open the [claim URL](https://vercel.com/claim-deployment?code=8ab2712f-6aab-4dce-b55d-d5acb64a8fce) and sign in, or import https://github.com/torksabig/trendlabs in the Vercel dashboard.
2. Set **Environment variables** (Project → Settings → Environment Variables), **Production** (and Preview if you use Arena there):

| Variable | Value | Required for Arena buttons |
| --- | --- | --- |
| `GITHUB_REPOSITORY` | `torksabig/trendlabs` | Yes |
| `GITHUB_TOKEN` or `GH_JOB_TOKEN` | PAT with `actions:write` + `contents:write` | Yes |
| `GITHUB_REF` | `main` | Optional |

Do **not** prefix these with `NEXT_PUBLIC_`.

3. Redeploy after env vars are saved.

## 3. Verify

- Dashboard loads snapshot data at `/`.
- `GET /api/status` returns JSON (job mode should mention GitHub Actions on Vercel).
- Arena **Run simulation** triggers Actions (check the repo’s Actions tab).

## Notes

- Paper trading only; no live brokerage keys on Vercel.
- Python workers run on GitHub Actions, not on Vercel Hobby.
- Snapshot updates commit `public/data/snapshot.json`; Vercel redeploys on push when Git is connected.
