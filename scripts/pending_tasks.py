#!/usr/bin/env python3
"""Terminal pending-task tracker — always shows the FULL done/pending list.

Scans every config registry for built/partial/planned status + checks crons +
key features, then prints a complete status board to the terminal.

Usage: python scripts/pending_tasks.py [--pending]   (--pending = only show open items)
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "config"
G, Y, R, B, D, X = "\033[92m", "\033[93m", "\033[91m", "\033[94m", "\033[2m", "\033[0m"


def status_counts(obj):
    """Recursively count built/partial/planned/n-a in any registry JSON."""
    c = {"built": 0, "partial": 0, "planned": 0, "n/a": 0, "other": 0}
    def walk(o):
        if isinstance(o, dict):
            s = o.get("status")
            if isinstance(s, str):
                key = s if s in c else ("built" if s == "pass" else "other")
                c[key] = c.get(key, 0) + 1
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(obj)
    return c


def main():
    only_pending = "--pending" in sys.argv
    ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(f"\n{B}╔══════════════════════════════════════════════════════════╗{X}")
    print(f"{B}║  AGENTIC EPILEPSY FINDER — TASK STATUS BOARD             ║{X}")
    print(f"{B}╚══════════════════════════════════════════════════════════╝{X}")
    print(f"{D}{ts}{X}\n")

    # 1. Registry status
    print(f"{B}■ REGISTRIES (built / partial / planned){X}")
    tot = {"built": 0, "partial": 0, "planned": 0, "n/a": 0}
    for f in sorted(CFG.glob("*.json")):
        try:
            obj = json.loads(f.read_text())
        except Exception:
            continue
        c = status_counts(obj)
        if sum(c.values()) - c["other"] == 0:
            continue
        for k in tot:
            tot[k] += c.get(k, 0)
        open_ct = c["partial"] + c["planned"]
        if only_pending and open_ct == 0:
            continue
        line = (f"  {f.stem:32s} {G}✓{c['built']}{X} {Y}~{c['partial']}{X} "
                f"{D}○{c['planned']}{X}" + (f" {D}n/a{c['n/a']}{X}" if c['n/a'] else ""))
        print(line)
    print(f"  {B}{'TOTAL':32s} {G}✓{tot['built']} built{X}  {Y}~{tot['partial']} partial{X}  {D}○{tot['planned']} planned{X}\n")

    # 2. Crons
    print(f"{B}■ SCHEDULED JOBS{X}")
    try:
        cr = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
        jobs = [l for l in cr.splitlines() if "agenticfinder" in l]
        for j in jobs:
            tag = j.split("# ")[-1].replace(" (agenticfinder)", "")
            print(f"  {G}✓{X} {tag}")
        if not jobs:
            print(f"  {R}○ no cron jobs installed{X}")
    except Exception:
        print(f"  {D}(crontab unavailable){X}")
    print()

    # 3. Pending highlights (planned items that are the real gaps)
    print(f"{B}■ TOP PENDING (the real gaps){X}")
    gaps = [
        "More CHB-MIT subjects (only 4 on disk → need 10-20, PhysioNet download)",
        "Hard 2nd dataset (TUSZ/Siena — Bonn too easy)",
        "Prospective clinical/reader study (needs IRB)",
        "Voice AI (Whisper/TTS)", "Video-EEG sync player",
        "Multi-user auth / RBAC", "EMR/FHIR + real device streaming",
        "External integrations (Slack/Drive/etc. via MCP)",
    ]
    for g in gaps:
        print(f"  {Y}○{X} {g}")
    print()
    print(f"{D}Run `python scripts/pending_tasks.py --pending` for open items only.{X}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
