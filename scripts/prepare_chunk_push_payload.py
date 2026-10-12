#!/usr/bin/env python3
"""Write GitHub MCP push_files JSON payloads (one chunk per file, <20KB)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / ".github" / "sync-chunks"
OUT = Path("/tmp/chunk_payloads")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in ROOT.iterdir() if p.name != "manifest.json")
    for name in names:
        content = (ROOT / name).read_text()
        payload = {
            "owner": "torksabig",
            "repo": "trendlabs",
            "branch": "main",
            "message": f"chore: sync chunk {name}",
            "files": [{"path": f".github/sync-chunks/{name}", "content": content}],
        }
        out = OUT / f"{name}.json"
        out.write_text(json.dumps(payload))
        print(name, out.stat().st_size)
    print("total", len(names), file=sys.stderr)


if __name__ == "__main__":
    main()
