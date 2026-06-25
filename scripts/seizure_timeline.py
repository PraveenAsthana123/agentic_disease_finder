#!/usr/bin/env python3
"""Real seizure timeline from CHB-MIT ground-truth annotations (chbNN-summary.txt).
Parses per-file seizure start/end times → per-subject events + burden stats. No synthetic."""
import re, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
SUMM = ROOT / "data" / "real_eeg" / "epilepsy_physionet"


def parse_summary(txt: str):
    """Parse one chbNN-summary.txt → list of file blocks with seizures."""
    files, cur = [], None
    sz_start = None
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith("File Name:"):
            if cur:
                files.append(cur)
            cur = {"file": line.split(":", 1)[1].strip(), "seizures": []}
        elif cur is not None and line.startswith("Seizure") and "Start Time" in line:
            m = re.search(r"(\d+)", line)
            sz_start = int(m.group(1)) if m else None
        elif cur is not None and line.startswith("Seizure") and "End Time" in line:
            m = re.search(r"(\d+)", line)
            if m and sz_start is not None:
                end = int(m.group(1))
                cur["seizures"].append({"start_sec": sz_start, "end_sec": end, "duration_sec": end - sz_start})
                sz_start = None
    if cur:
        files.append(cur)
    return files


def build() -> dict:
    subjects = []
    all_events, total_sz, total_files = [], 0, 0
    for summ in sorted(SUMM.glob("chb*/chb*-summary.txt")):
        subj = summ.parent.name
        files = parse_summary(summ.read_text(errors="ignore"))
        sz = [(f["file"], s) for f in files for s in f["seizures"]]
        durs = [s["duration_sec"] for _, s in sz]
        subjects.append({
            "subject": subj, "n_files": len(files), "n_seizures": len(sz),
            "total_seizure_sec": sum(durs),
            "mean_seizure_sec": round(sum(durs) / len(durs), 1) if durs else 0,
            "files_with_seizures": sum(1 for f in files if f["seizures"]),
        })
        for fname, s in sz:
            all_events.append({"subject": subj, "file": fname, **s})
        total_sz += len(sz); total_files += len(files)
    subjects.sort(key=lambda x: x["n_seizures"], reverse=True)
    durs = [e["duration_sec"] for e in all_events]
    return {
        "available": bool(subjects),
        "source": "CHB-MIT chbNN-summary.txt (ground-truth clinical annotations)",
        "n_subjects": len(subjects), "total_seizures": total_sz, "total_files": total_files,
        "total_seizure_minutes": round(sum(durs) / 60, 1),
        "mean_seizure_sec": round(sum(durs) / len(durs), 1) if durs else 0,
        "min_seizure_sec": min(durs) if durs else 0, "max_seizure_sec": max(durs) if durs else 0,
        "by_subject": subjects,
        "events": sorted(all_events, key=lambda e: e["duration_sec"], reverse=True)[:50],
        "note": "Ground-truth seizure onset/offset per recording — the labels AI seizure-detection is evaluated against.",
    }


if __name__ == "__main__":
    r = build()
    print(f"CHB-MIT seizure timeline: {r['total_seizures']} seizures across {r['n_subjects']} subjects "
          f"({r['total_seizure_minutes']} min total, mean {r['mean_seizure_sec']}s)")
    for s in r["by_subject"]:
        print(f"  {s['subject']}: {s['n_seizures']} seizures in {s['files_with_seizures']}/{s['n_files']} files")
