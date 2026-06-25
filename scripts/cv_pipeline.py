#!/usr/bin/env python3
"""CV pipeline on extracted video frames + EEG-trace images:
1. Noise cleaning (OpenCV fastNlMeansDenoising)
2. Segmentation (OpenCV threshold + contours)
3. Detection (YOLO if weights present, else OpenCV contour boxes)
4. Classification (qwen2.5vl / llava vision model via Ollama)
Writes jobs/reports/cv_pipeline_latest.json. Real — uses installed cv2/YOLO/Ollama."""
import base64
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "jobs" / "reports"
FRAMES = ROOT / "data" / "frames"
CLEAN = ROOT / "data" / "frames_clean"


def classify_ollama(img_path: Path, model: str = "llava:7b") -> str:
    try:
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        payload = json.dumps({"model": model, "prompt": "Describe this clinical image in one sentence (patient/EEG/behaviour).",
                              "images": [b64], "stream": False}).encode()
        req = urllib.request.Request("http://localhost:11434/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read()).get("response", "").strip()[:300]
    except Exception as e:  # noqa: BLE001
        return f"(vision model unavailable: {str(e)[:80]})"


def main():
    import cv2
    import numpy as np
    now = datetime.now(timezone.utc).astimezone()
    REPORTS.mkdir(parents=True, exist_ok=True); CLEAN.mkdir(parents=True, exist_ok=True)
    frames = sorted(FRAMES.rglob("*.jpg"))[:10]
    if not frames:
        print("no frames — run video_to_frames.py first"); return 1

    # YOLO detection (optional)
    yolo = None
    try:
        from ultralytics import YOLO
        yolo = YOLO("yolov8n.pt")  # downloads tiny weights on first run
    except Exception:
        yolo = None

    results = []
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        # 1. NOISE CLEANING
        clean = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        out = CLEAN / f.name; cv2.imwrite(str(out), clean)
        # 2. SEGMENTATION (threshold + contours)
        gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segments = len([c for c in contours if cv2.contourArea(c) > 500])
        # 3. DETECTION
        if yolo is not None:
            det = yolo(str(f), verbose=False)[0]
            objects = [det.names[int(b.cls)] for b in det.boxes][:8]
        else:
            objects = [f"region({segments})"]
        results.append({"frame": f.name, "denoised": out.name, "segments": segments, "detections": objects})

    # 4. CLASSIFICATION (vision model on the first frame only — it's slow)
    classification = classify_ollama(frames[0]) if frames else ""

    report = {
        "run_at_local": now.isoformat(timespec="seconds"),
        "frames_processed": len(results),
        "noise_cleaning": "OpenCV fastNlMeansDenoisingColored",
        "segmentation": "OpenCV Otsu threshold + contours",
        "detection": "YOLOv8n" if yolo else "OpenCV contour regions (YOLO weights not loaded)",
        "classification_model": "llava:7b (Ollama)",
        "classification_sample": classification,
        "results": results,
    }
    (REPORTS / "cv_pipeline_latest.json").write_text(json.dumps(report, indent=2))
    try:
        sys.path.insert(0, str(ROOT)); import clinical_db as cdb
        cdb.log_transaction("_system", component="cv_pipeline", action="process",
                            detail=f"{len(results)} frames: denoise+segment+detect+classify")
    except Exception:
        pass
    print(f"[{report['run_at_local']}] CV pipeline: {len(results)} frames denoised+segmented+detected")
    print(f"  detection: {report['detection']}")
    print(f"  classification (frame 1): {classification[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
