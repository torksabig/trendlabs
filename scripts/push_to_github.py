#!/usr/bin/env python3
"""Upload local tracked files to github.com/torksabig/trendlabs via REST API.

Requires GITHUB_TOKEN or GH_TOKEN with repo scope (contents: write).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

OWNER = "torksabig"
REPO = "trendlabs"
BRANCH = "main"
API = f"https://api.github.com/repos/{OWNER}/{REPO}"


def token() -> str:
    for key in ("GITHUB_TOKEN", "GH_TOKEN", "GH_JOB_TOKEN"):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    print("Missing GITHUB_TOKEN or GH_TOKEN", file=sys.stderr)
    sys.exit(1)


def api(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{API}{path}"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token()}",
            "User-Agent": "trendlabs-sync",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req) as resp:
        raw = resp.read().decode()
        return json.loads(raw) if raw else {}


def get_sha(path: str) -> str | None:
    try:
        meta = api("GET", f"/contents/{path}?ref={BRANCH}")
        return meta.get("sha")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def put_file(path: str, content: bytes, message: str) -> None:
    sha = get_sha(path)
    body: dict = {
        "message": message,
        "content": base64.b64encode(content).decode(),
        "branch": BRANCH,
    }
    if sha:
        body["sha"] = sha
    api("PUT", f"/contents/{path}", body)
    print(f"OK {path} ({len(content)} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Repo-relative file paths")
    parser.add_argument("-m", "--message", default="chore: sync files via API")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for rel in args.paths:
        p = root / rel
        if not p.is_file():
            print(f"skip missing {rel}", file=sys.stderr)
            continue
        put_file(rel, p.read_bytes(), args.message)


if __name__ == "__main__":
    main()
