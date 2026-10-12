#!/usr/bin/env python3
"""Push a prebuilt push_files payload JSON to GitHub (Git Data API).

Usage:
  GITHUB_TOKEN=... github_push_files_batch.py /tmp/mcp_out/single_batch_02.json

Implements the same multi-file commit flow as GitHub MCP push_files.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request

OWNER = "torksabig"
REPO = "trendlabs"
API = f"https://api.github.com/repos/{OWNER}/{REPO}"


def token() -> str:
    for k in ("GITHUB_TOKEN", "GITHUB_PERSONAL_ACCESS_TOKEN", "GH_TOKEN"):
        v = (os.environ.get(k) or "").strip()
        if v:
            return v
    raise SystemExit("Missing GITHUB_TOKEN / GITHUB_PERSONAL_ACCESS_TOKEN / GH_TOKEN")


def req(method: str, url: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token()}",
            "User-Agent": "trendlabs-batch-push",
            "X-GitHub-Api-Version": "2022-11-28",
            **({"Content-Type": "application/json"} if body else {}),
        },
    )
    with urllib.request.urlopen(r, timeout=180) as resp:
        raw = resp.read().decode()
        return json.loads(raw) if raw else {}


def push_payload(payload: dict) -> str:
    branch = payload["branch"]
    message = payload["message"]
    files = payload["files"]

    ref = req("GET", f"{API}/git/ref/heads/{branch}")
    base_commit = ref["object"]["sha"]
    base_tree = req("GET", f"{API}/git/commits/{base_commit}")["tree"]["sha"]

    tree_items = []
    for f in files:
        path = f["path"]
        content = f["content"]
        blob = req(
            "POST",
            f"{API}/git/blobs",
            {"content": content, "encoding": "utf-8"},
        )
        tree_items.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})

    tree = req(
        "POST",
        f"{API}/git/trees",
        {"base_tree": base_tree, "tree": tree_items},
    )
    commit = req(
        "POST",
        f"{API}/git/commits",
        {
            "message": message,
            "tree": tree["sha"],
            "parents": [base_commit],
        },
    )
    req(
        "PATCH",
        f"{API}/git/refs/heads/{branch}",
        {"sha": commit["sha"]},
    )
    return commit["sha"]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <payload.json>")
    payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
    sha = push_payload(payload)
    print("OK", sha, payload.get("message", ""))


if __name__ == "__main__":
    main()
