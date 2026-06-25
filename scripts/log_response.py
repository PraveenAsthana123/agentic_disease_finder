#!/usr/bin/env python3
"""Stop hook target — logs the assistant's last response to the conversation log (DB + MD)."""
import sys, json, re
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path("/media/praveen/Asthana4/rajveer/agenticfinder"); sys.path.insert(0, str(ROOT))
import clinical_db as cdb
try:
    data = json.load(sys.stdin)
    tpath = data.get("transcript_path", "")
except Exception:
    tpath = ""
text = ""
if tpath and Path(tpath).exists():
    # find last assistant text message in the JSONL transcript
    for line in reversed(Path(tpath).read_text(errors="ignore").splitlines()):
        try:
            o = json.loads(line)
            if o.get("type") == "assistant":
                c = o.get("message", {}).get("content", [])
                text = " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
                if text.strip():
                    break
        except Exception:
            continue
if text.strip():
    cdb.log_convo("assistant", text)
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    with open(ROOT / "prompt_inputs" / "CONVERSATION.md", "a") as f:
        f.write(f"\n### [{now}] ASSISTANT\n{text[:2000]}\n")
