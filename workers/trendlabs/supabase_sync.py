"""Optional Supabase upload for research reports (server-side secret only)."""

from __future__ import annotations

import os
from typing import Any

from trendlabs.config import SUPABASE_SECRET_KEY, SUPABASE_URL


def upload_agent_report(payload: dict[str, Any]) -> Any:
    url = SUPABASE_URL or os.getenv("SUPABASE_URL")
    key = SUPABASE_SECRET_KEY or os.getenv("SUPABASE_SECRET_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SECRET_KEY are required for upload")

    from supabase import create_client

    client = create_client(url, key)
    row = {
        "symbol": payload["symbol"],
        "timeframe": payload["timeframe"],
        "model": payload["model"],
        "report": {
            "source": payload.get("source"),
            "snapshot": payload.get("snapshot"),
            "analysis": payload.get("report"),
            "created_at": payload.get("created_at"),
        },
    }
    return client.table("agent_reports").insert(row).execute()
