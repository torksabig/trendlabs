# TrendLab production checklist

One-time setup to run the dashboard on Vercel and workers on GitHub Actions.

## Links

| Resource | URL |
| --- | --- |
| GitHub repo | https://github.com/torksabig/trendlabs |
| Temporary Vercel preview | https://temporary-brisk-bassoon-or1i9hz.vercel.app (claim within ~60 minutes — see below) |
| Claim deployment | https://vercel.com/claim-deployment?code=a2709737-8720-4e54-b6e7-4158713ccd32 |

After you claim the deployment and connect the GitHub repo, Vercel will assign a stable production URL (typically `https://<project>.vercel.app`).

## 1. GitHub (code + Actions)

- [x] Application source is in `torksabig/trendlabs` on branch `main` (large/binary files: run `GITHUB_TOKEN=… python3 scripts/push_to_github.py package-lock.json public/data/snapshot.json data/snapshots/latest.json src/app/favicon.ico` if anything is missing).
- [ ] Add **Repository secrets** (Settings → Secrets and variables → Actions) for optional data/LLM:
  - `ALPACA_API_KEY`, `ALPACA_API_SECRET` (optional; Yahoo fallback works without them)
  - `GOOGLE_AI_API_KEY` / `GEMINI_API_KEY`, `GROQ_API_KEY`, etc. (optional; research can use `fallback`)

**Run a worker manually:** Actions → **TrendLab scheduled research** → **Run workflow** → choose job (`simulate`, `research`, `seed`, …).

## 2. Vercel (dashboard)

**Temporary/claim deploys cannot hold secrets.** Claim (or import) the project first, then set env vars on the owned project.

1. Open the [claim URL](https://vercel.com/claim-deployment?code=a2709737-8720-4e54-b6e7-4158713ccd32) and sign in, or import https://github.com/torksabig/trendlabs in the Vercel dashboard.
2. Project → **Settings** → **Environment Variables** → scope **Production** (add Preview too if you use Arena there).
3. Copy-paste these three (create a fine-grained or classic PAT with `actions:write` + `contents:write` on `torksabig/trendlabs`):

```text
GITHUB_REPOSITORY=torksabig/trendlabs
GITHUB_TOKEN=ghp_your_pat_here
GITHUB_REF=main
```

| Variable | Value | Required for Arena buttons |
| --- | --- | --- |
| `GITHUB_REPOSITORY` | `torksabig/trendlabs` | Yes |
| `GITHUB_TOKEN` or `GH_JOB_TOKEN` | PAT with `actions:write` + `contents:write` | Yes |
| `GITHUB_REF` | `main` | Optional (defaults to git branch / `main`) |

Do **not** prefix these with `NEXT_PUBLIC_`. Prefer `GH_JOB_TOKEN` if you already use `GITHUB_TOKEN` for something else on the same project.

4. **Redeploy** after saving (Deployments → … → Redeploy), then retry Arena job buttons.

## 3. Verify

- Dashboard loads snapshot data at `/`.
- `GET /api/status` returns JSON (job mode should mention GitHub Actions on Vercel).
- Arena **Run simulation** triggers Actions (check the repo’s Actions tab).

## Notes

- Paper trading only; no live brokerage keys on Vercel.
- Python workers run on GitHub Actions, not on Vercel Hobby.
- Snapshot updates commit `public/data/snapshot.json`; Vercel redeploys on push when Git is connected.
- GitHub `main` may ship a minimal snapshot and no lockfile for Vercel bootstrap; after `GITHUB_TOKEN` is set, sync the full `package-lock.json` and snapshots with `python3 scripts/push_to_github.py package-lock.json public/data/snapshot.json data/snapshots/latest.json`.
