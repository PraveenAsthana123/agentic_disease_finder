#!/usr/bin/env python3
"""Collect thesis-ready assets into one ``thesis/`` folder.

Gathers the already-generated figures (PDF/PNG/SVG), result tables
(CSV/JSON), and writes a Markdown index so they can be dropped straight
into a thesis/dissertation. Read-only over sources; idempotent (safe to
re-run, e.g. from cron).

Usage:
    python scripts/collect_thesis_assets.py [--root .] [--out thesis]
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Source globs relative to the project root.
FIGURE_DIRS = ["paper/figures", "papers/figures"]
RESULT_GLOBS = [
    "disease_results.json",
    "results/all_diseases_summary.csv",
    "results/all_diseases_summary.json",
    "results/detailed_metrics.json",
    "analysis_results/*.json",
]
FIGURE_EXTS = (".pdf", ".png", ".svg")


def _stamp() -> str:
    now = datetime.now(timezone.utc).astimezone()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


def collect(root: Path, out: Path) -> dict:
    figs_out = out / "figures"
    data_out = out / "data"
    figs_out.mkdir(parents=True, exist_ok=True)
    data_out.mkdir(parents=True, exist_ok=True)

    copied_figs: list[str] = []
    for d in FIGURE_DIRS:
        src_dir = root / d
        if not src_dir.is_dir():
            continue
        for f in sorted(src_dir.iterdir()):
            if f.suffix.lower() in FIGURE_EXTS and f.is_file():
                dest = figs_out / f.name
                # De-dup: if same name from a second dir, prefix with parent.
                if dest.exists() and dest.stat().st_size != f.stat().st_size:
                    dest = figs_out / f"{f.parent.name}__{f.name}"
                shutil.copy2(f, dest)
                copied_figs.append(dest.name)

    copied_data: list[str] = []
    for pattern in RESULT_GLOBS:
        for f in sorted(root.glob(pattern)):
            if f.is_file():
                shutil.copy2(f, data_out / f.name)
                copied_data.append(f.name)

    return {
        "figures": sorted(set(copied_figs)),
        "data": sorted(set(copied_data)),
        "generated": _stamp(),
    }


def write_index(out: Path, manifest: dict) -> None:
    figs = manifest["figures"]
    data = manifest["data"]
    pdfs = [f for f in figs if f.endswith(".pdf")]

    lines = [
        "# Thesis Assets",
        "",
        f"_Auto-collected {manifest['generated']} by `scripts/collect_thesis_assets.py`._",
        "",
        f"- **{len(figs)}** figure files in `figures/` ({len(pdfs)} PDF, "
        f"{len([f for f in figs if f.endswith('.png')])} PNG, "
        f"{len([f for f in figs if f.endswith('.svg')])} SVG)",
        f"- **{len(data)}** result/data files in `data/`",
        "",
        "> ⚠️ Verify result numbers before citing: `disease_results.json` (epilepsy 99.02%) "
        "and `results/all_diseases_summary.csv` (epilepsy 100%) disagree, and a separate run "
        "shows 81.67%. Perfect/near-perfect EEG accuracy is a data-leakage red flag.",
        "",
        "## Figures (PDF — best for LaTeX/Word)",
        "",
    ]
    lines += [f"- `figures/{f}`" for f in pdfs] or ["- (none)"]
    lines += ["", "## Result / data tables", ""]
    lines += [f"- `data/{f}`" for f in data] or ["- (none)"]
    lines += [""]

    (out / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="thesis")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = (root / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest = collect(root, out)
    write_index(out, manifest)

    print(f"[collect_thesis_assets] {manifest['generated']}")
    print(f"  figures: {len(manifest['figures'])} -> {out / 'figures'}")
    print(f"  data:    {len(manifest['data'])} -> {out / 'data'}")
    print(f"  index:   {out / 'INDEX.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
