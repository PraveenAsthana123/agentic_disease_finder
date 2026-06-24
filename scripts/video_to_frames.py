#!/usr/bin/env python3
"""Video → image-frame extraction job. Pulls keyframes from uploaded EEG/seizure
videos (data/uploads/videos + patient folders) so they can be used for CV analysis.
Writes jobs/reports/video_frames_latest.json. Run by cron (hourly) or manually."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "jobs" / "reports"
FRAMES = ROOT / "data" / "frames"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".3gp", ".m4v", ".flv", ".wmv"}


def have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
        return True
    except Exception:
        return False


def extract(video: Path, fps: float = 0.5) -> int:
    """Extract ~1 frame every 2s into data/frames/<stem>/. Returns frame count."""
    out = FRAMES / video.stem
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(video), "-vf", f"fps={fps}",
                    str(out / "frame_%04d.jpg")], capture_output=True, timeout=600)
    return len(list(out.glob("*.jpg")))


def main():
    now = datetime.now(timezone.utc).astimezone()
    REPORTS.mkdir(parents=True, exist_ok=True)
    vids = []
    for base in [ROOT / "data" / "uploads" / "videos", ROOT / "data" / "patients", ROOT / "Epilepsy"]:
        if base.exists():
            vids += [p for p in base.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    ffmpeg = have_ffmpeg()
    results = []
    if ffmpeg:
        for v in vids[:50]:
            try:
                n = extract(v)
                results.append({"video": v.name, "frames": n, "ok": True})
            except Exception as e:  # noqa: BLE001
                results.append({"video": v.name, "frames": 0, "ok": False, "error": str(e)[:120]})
    report = {
        "run_at_local": now.isoformat(timespec="seconds"),
        "run_at_utc": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "ffmpeg_available": ffmpeg,
        "videos_found": len(vids), "processed": len(results),
        "total_frames": sum(r["frames"] for r in results),
        "results": results,
        "note": "" if ffmpeg else "ffmpeg not installed — install with: sudo apt install ffmpeg",
    }
    (REPORTS / "video_frames_latest.json").write_text(json.dumps(report, indent=2))
    try:
        sys.path.insert(0, str(ROOT)); import clinical_db as cdb
        cdb.log_transaction("_system", component="video_frames", action="extract",
                            detail=f"{len(vids)} videos → {report['total_frames']} frames (ffmpeg={ffmpeg})")
    except Exception:
        pass
    print(f"[{report['run_at_local']}] videos={len(vids)} frames={report['total_frames']} ffmpeg={ffmpeg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
