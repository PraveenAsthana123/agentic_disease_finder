#!/usr/bin/env python3
"""Backfill ALL operator inputs from Claude Code transcripts into the Request Inbox,
so the full project history (every prompt, since day 1) is in the DB. Dedupes."""
import json, glob, sqlite3, re
from pathlib import Path
ROOT = Path("/media/praveen/Asthana4/rajveer/agenticfinder")
TX = "/home/praveen/.claude/projects/-media-praveen-Asthana4-rajveer-agenticfinder/*.jsonl"

c = sqlite3.connect(str(ROOT / "data/clinical.db"))
c.execute("""CREATE TABLE IF NOT EXISTS operator_requests (id INTEGER PRIMARY KEY AUTOINCREMENT,
  request_text TEXT, category TEXT, status TEXT DEFAULT 'open', notes TEXT, source TEXT,
  ts_utc TEXT, ts_local TEXT, updated_at TEXT)""")
existing = set(r[0] for r in c.execute("SELECT request_text FROM operator_requests").fetchall())

prompts = []
for f in sorted(glob.glob(TX)):
    for line in open(f, errors="ignore"):
        try:
            o = json.loads(line)
            if o.get("type") != "user":
                continue
            ct = o.get("message", {}).get("content", "")
            txt = ct if isinstance(ct, str) else " ".join(b.get("text", "") for b in ct if isinstance(b, dict) and b.get("type") == "text")
            txt = txt.strip()
            if not txt or txt.startswith("<") or "tool_result" in str(ct) or "[Request interrupted" in txt:
                continue
            ts = o.get("timestamp", "")
            prompts.append((txt, ts))
        except Exception:
            continue

added = 0
seen = set()
for txt, ts in prompts:
    key = txt[:200]
    if key in seen or txt in existing:
        continue
    seen.add(key)
    c.execute("INSERT INTO operator_requests(request_text,category,status,source,ts_utc,ts_local,updated_at) "
              "VALUES(?,?,'logged','transcript',?,?,?)", (txt[:2000], "history", ts, ts, ts))
    added += 1
c.commit()
# report
rows = c.execute("SELECT ts_local FROM operator_requests WHERE ts_local!='' ORDER BY ts_local").fetchall()
first = rows[0][0][:10] if rows else "?"
from collections import Counter
dist = dict(Counter(r[0] for r in c.execute("SELECT status FROM operator_requests").fetchall()))
total = c.execute("SELECT COUNT(*) FROM operator_requests").fetchone()[0]
print(f"backfilled {added} historical prompts")
print(f"PROJECT START: {first} · TOTAL PROMPTS: {total} · status: {dist}")
