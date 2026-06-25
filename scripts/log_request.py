#!/usr/bin/env python3
"""UserPromptSubmit hook target — triple-writes every operator input:
 1. DB (operator_requests table)  2. master MD (prompt_inputs/ALL_INPUTS.md)  3. per-input file."""
import sys, json, re
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path("/media/praveen/Asthana4/rajveer/agenticfinder")
sys.path.insert(0, str(ROOT))
import clinical_db as cdb

text = ""
try:
    data = json.load(sys.stdin); text = data.get("prompt") or data.get("user_input") or ""
except Exception:
    text = " ".join(sys.argv[1:])
text = text.strip()
if any(k in text for k in ("scripts/","§159","NEVER force-push","safe_push.sh")):
    sys.exit(0)  # internal-skip: not an operator input
if not text:
    sys.exit(0)

res = cdb.save_request(text, source="chat")
if res.get("deduped"):
    sys.exit(0)
rid = res.get("id", 0)
now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
folder = ROOT / "prompt_inputs"; folder.mkdir(exist_ok=True)
# 2. master MD (append-only)
with open(folder / "ALL_INPUTS.md", "a") as f:
    f.write(f"- [ ] **#{rid}** [{now}] {text}\n")
# 3. per-input file
safe = re.sub(r"[^a-z0-9]+", "-", text.lower())[:50].strip("-")
(folder / f"{rid:05d}_{safe}.md").write_text(f"# Request #{rid}\n\n- When: {now}\n- Status: open\n\n## Input\n{text}\n")
print(f"logged request #{rid}")
