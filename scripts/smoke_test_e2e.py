#!/usr/bin/env python3
"""End-to-end process smoke test — upload → predict → report → council.

Covers the 'Process' testing dimension in config/stories_and_tests.json:
  "end-to-end pipelines run (upload→predict→report; council)"

Exit 0 = all checks pass.  Exit 1 = any failure.
Usage:
  python3 scripts/smoke_test_e2e.py [--port 8010]
"""
import argparse
import io
import json
import sys
import tempfile
import time
from pathlib import Path

# ── stdlib-only HTTP (no requests dependency required) ──────────────────────
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent.parent
PORT = 8010
BASE = f"http://127.0.0.1:{PORT}"

# ── Helpers ──────────────────────────────────────────────────────────────────

def _get(path: str, timeout: int = 15) -> dict:
    url = BASE + path
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def _post_json(path: str, payload: dict, timeout: int = 20) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _post_multipart(path: str, fields: dict, files: dict, timeout: int = 60) -> dict:
    """Minimal multipart/form-data POST — no external deps."""
    boundary = "----SmokeBoundary7F3A9E"
    body_parts = []
    for name, value in fields.items():
        body_parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        )
    for name, (filename, content, ctype) in files.items():
        body_parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f'Content-Type: {ctype}\r\n\r\n'
        )
        body_parts.append(content if isinstance(content, str) else content.decode("latin-1"))
        body_parts.append("\r\n")
    body_parts.append(f"--{boundary}--\r\n")
    body = "".join(body_parts).encode("latin-1")
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ── Synthetic EEG generator ──────────────────────────────────────────────────

def _make_synthetic_eeg_csv(n_channels: int = 4, duration_sec: int = 2, sfreq: int = 256) -> bytes:
    """Return a minimal CSV EEG (header row + signal rows) as bytes."""
    import math
    n_samples = duration_sec * sfreq
    ch_names = [f"EEG{i+1}" for i in range(n_channels)]
    lines = [",".join(ch_names)]
    for t in range(n_samples):
        row = [str(round(10 * math.sin(2 * math.pi * 10 * t / sfreq + i * 0.5), 4))
               for i in range(n_channels)]
        lines.append(",".join(row))
    return "\n".join(lines).encode()


# ── Test cases ───────────────────────────────────────────────────────────────

RESULTS = []

def _check(name: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    RESULTS.append((name, status, detail))
    icon = "✅" if ok else "❌"
    print(f"  {icon} [{status}] {name}" + (f" — {detail}" if detail else ""))


def test_backend_alive():
    """Backend must return a valid JSON root response."""
    try:
        r = _get("/")
        _check("backend_alive", "message" in r or "version" in r, str(r)[:80])
    except Exception as e:
        _check("backend_alive", False, str(e))


def test_analyze_upload():
    """POST a synthetic CSV to /api/analyze-upload; expect status=success + prediction."""
    csv_bytes = _make_synthetic_eeg_csv()
    try:
        r = _post_multipart(
            "/api/analyze-upload",
            fields={"disease": "epilepsy", "patient_id": "smoke_test_pt"},
            files={"file": ("smoke_eeg.csv", csv_bytes, "text/csv")},
        )
        ok_status = r.get("status") == "success"
        has_prediction = "prediction" in r and r["prediction"] is not None
        has_features = "features" in r and isinstance(r["features"], dict) and len(r["features"]) > 0
        _check("analyze_upload.status_success", ok_status, r.get("status", "?"))
        _check("analyze_upload.has_prediction", has_prediction, str(r.get("prediction", {}))[:80])
        _check("analyze_upload.has_features", has_features, f"{len(r.get('features', {}))} features")
        return r
    except Exception as e:
        _check("analyze_upload", False, str(e))
        return {}


def test_report_layout():
    """GET /api/report-layout — must return a usable structure."""
    try:
        r = _get("/api/report-layout")
        _check("report_layout", isinstance(r, (dict, list)), str(r)[:80])
    except Exception as e:
        _check("report_layout", False, str(e))


def test_council_run():
    """POST /api/council/run with a clinical query — must return a governed response."""
    try:
        r = _post_json(
            "/api/council/run",
            {"query": "Is this EEG consistent with epilepsy?", "patient_id": "smoke_test_pt"},
        )
        has_resp = bool(r)  # any non-empty dict
        _check("council_run.responds", has_resp, str(r)[:120])
    except Exception as e:
        _check("council_run", False, str(e))


def test_data_manager():
    """GET /api/data-manager — CDM overview must return summary counts."""
    try:
        r = _get("/api/data-manager")
        has_summary = "summary" in r or "role" in r
        _check("data_manager", has_summary, str(r.get("summary", {}))[:80])
    except Exception as e:
        _check("data_manager", False, str(e))


def test_pipeline_status():
    """GET /api/automatic-pipelines/overview — pipeline registry must respond."""
    try:
        r = _get("/api/automatic-pipelines/overview")
        _check("automatic_pipelines", isinstance(r, (dict, list)), str(r)[:80])
    except Exception as e:
        _check("automatic_pipelines", False, str(e))


# ── Runner ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Process E2E smoke test")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    global BASE
    BASE = f"http://127.0.0.1:{args.port}"

    print(f"\n{'='*60}")
    print("PROCESS E2E SMOKE TEST — upload→predict→report→council")
    print(f"Backend: {BASE}   ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'='*60}\n")

    test_backend_alive()
    test_analyze_upload()
    test_report_layout()
    test_council_run()
    test_data_manager()
    test_pipeline_status()

    passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
    failed = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    total = len(RESULTS)

    print(f"\n{'─'*60}")
    print(f"RESULT: {passed}/{total} passed  ·  {failed} failed")
    if failed == 0:
        print("✅  ALL CHECKS PASS — Process E2E pipeline verified")
    else:
        print("❌  SOME CHECKS FAILED — review output above")
    print(f"{'─'*60}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
