#!/usr/bin/env python3
"""Push local files to GitHub via REST API (Contents API). Requires GITHUB_TOKEN or GITHUB_PERSONAL_ACCESS_TOKEN."""
import base64
import json
import os
import sys
import urllib.error
import urllib.request

OWNER = "torksabig"
REPO = "trendlabs"
BRANCH = "main"


def token():
    for k in ("GITHUB_PERSONAL_ACCESS_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v.strip()
    return None


def api(method, path, data=None):
    tok = token()
    if not tok:
        raise SystemExit("No GitHub token in environment")
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{path}"
    headers = {
        "Authorization": f"Bearer {tok}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "trendlabs-sync",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        raise SystemExit(f"GitHub API {method} {path}: {e.code} {err[:500]}")


def get_sha(path):
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{path}?ref={BRANCH}"
    tok = token()
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {tok}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "trendlabs-sync",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode()).get("sha")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def put_file(path, local_path, message, binary=False):
    with open(local_path, "rb") as f:
        raw = f.read()
    if binary:
        content = base64.b64encode(raw).decode()
        encoding = "base64"
    else:
        content = raw.decode("utf-8")
        encoding = "utf-8"
    payload = {
        "message": message,
        "content": content,
        "branch": BRANCH,
    }
    sha = get_sha(path)
    if sha:
        payload["sha"] = sha
    if encoding == "base64":
        # GitHub contents API expects base64 in content field for binary; utf-8 for text
        payload["content"] = content
    return api("PUT", path, payload)


def main():
    if len(sys.argv) < 2:
        print("Usage: github_sync_push.py <repo-relative-path> ...", file=sys.stderr)
        sys.exit(2)
    for rel in sys.argv[1:]:
        local = os.path.join("/workspace", rel)
        if not os.path.isfile(local):
            raise SystemExit(f"Missing local file: {local}")
        binary = rel.endswith(".ico")
        msg = f"chore: sync {rel} from Cursor workspace"
        print(f"Pushing {rel} ({os.path.getsize(local)} bytes)...")
        res = put_file(rel, local, msg, binary=binary)
        print("  OK", res.get("commit", {}).get("sha", "")[:12])


if __name__ == "__main__":
    main()
