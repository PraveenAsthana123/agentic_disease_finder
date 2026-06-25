#!/usr/bin/env python3
"""Clinical Data Manager — Video Validation.

Validates the REAL extracted video frames on disk (data/frames + data/frames_clean):
per-frame readability, dimensions, colour mode, file size, and brightness — with
flags for corrupt, blank/near-black, over-exposed, and dimension-inconsistent
frames. Backs the seizure-video QC pipeline (frames feed the CV lifecycle).

100% real (reads actual JPG/PNG bytes via PIL) — no synthetic, no mutation.
"""

import glob
import os
import statistics
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME_DIRS = ["data/frames", "data/frames_clean"]
BLANK_MEAN = 12.0      # mean luma below this ⇒ near-black/blank
BRIGHT_MEAN = 245.0    # mean luma above this ⇒ over-exposed/washed-out


def _frame_paths(d):
    out = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        out += glob.glob(os.path.join(ROOT, d, "**", ext), recursive=True)
    return sorted(out)


def _inspect(path):
    """Real per-frame inspection — never raises (corruption ⇒ flagged honestly)."""
    rel = os.path.relpath(path, ROOT)
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()  # detect truncation/corruption
        with Image.open(path) as im:
            w, h = im.size
            mode = im.mode
            gray = im.convert("L")
            # sample mean luma (downscale for speed, still real)
            gray.thumbnail((64, 64))
            px = list(gray.getdata())
            mean_luma = round(sum(px) / len(px), 1) if px else 0.0
        return {"frame": rel, "readable": True, "width": w, "height": h, "mode": mode,
                "bytes": os.path.getsize(path), "mean_luma": mean_luma,
                "blank": mean_luma < BLANK_MEAN, "overexposed": mean_luma > BRIGHT_MEAN}
    except Exception as e:  # noqa: BLE001 — QC must report corruption, not crash
        return {"frame": rel, "readable": False, "issue": f"{type(e).__name__}: {e}"}


def validate_frames():
    dirs = []
    all_dims = []
    total_frames = total_blank = total_corrupt = 0
    for d in FRAME_DIRS:
        paths = _frame_paths(d)
        if not paths:
            continue
        frames = [_inspect(p) for p in paths]
        readable = [f for f in frames if f["readable"]]
        dims = Counter((f["width"], f["height"]) for f in readable)
        all_dims += [(f["width"], f["height"]) for f in readable]
        blanks = [f["frame"] for f in readable if f.get("blank")]
        overexp = [f["frame"] for f in readable if f.get("overexposed")]
        corrupt = [f["frame"] for f in frames if not f["readable"]]
        flags = []
        if len(dims) > 1:
            flags.append(f"inconsistent dimensions: {dict(dims)}")
        if blanks:
            flags.append(f"{len(blanks)} blank/near-black frame(s)")
        if overexp:
            flags.append(f"{len(overexp)} over-exposed frame(s)")
        if corrupt:
            flags.append(f"{len(corrupt)} unreadable/corrupt frame(s)")
        total_frames += len(frames)
        total_blank += len(blanks)
        total_corrupt += len(corrupt)
        dirs.append({
            "directory": d, "n_frames": len(frames), "readable": len(readable),
            "dimensions": {f"{w}x{h}": n for (w, h), n in dims.items()},
            "blank_frames": blanks, "overexposed_frames": overexp, "corrupt_frames": corrupt,
            "flags": flags or ["clean"],
        })
    overall_dims = Counter(all_dims)
    return {
        "available": bool(dirs),
        "directories": dirs,
        "summary": {
            "total_frames": total_frames,
            "blank_frames": total_blank,
            "corrupt_frames": total_corrupt,
            "distinct_resolutions": len(overall_dims),
            "resolution_distribution": {f"{w}x{h}": n for (w, h), n in overall_dims.items()},
            "validation": "PASS" if (total_corrupt == 0 and total_blank == 0 and len(overall_dims) <= 1)
                          else "REVIEW",
        },
        "thresholds": {"blank_mean_luma": BLANK_MEAN, "overexposed_mean_luma": BRIGHT_MEAN},
        "note": "Per-frame integrity/brightness QC over real extracted frames (PIL). Report only.",
    }


if __name__ == "__main__":
    r = validate_frames()
    if not r["available"]:
        print("No frames found.")
    else:
        print("Video frame validation:", r["summary"])
        for d in r["directories"]:
            print(f"  {d['directory']}: {d['readable']}/{d['n_frames']} readable, flags={d['flags']}")
