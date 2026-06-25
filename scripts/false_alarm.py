#!/usr/bin/env python3
"""False Alarm Review — validate a simple seizure detector against ground truth.

Runs a power-based detector across an annotated CHB-MIT recording (sliding
windows flagged when broadband power exceeds a robust threshold), then scores
each detection against the annotated ictal windows: true positive (overlaps a
seizure) vs false alarm (interictal). Reports sensitivity + false-alarm rate/hour.

100% real (annotated EDF via MNE/SciPy) — runs under the canonical venv.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def review(file: str = None, window_s: float = 5.0, k: float = 6.0) -> dict:
    import mne
    import numpy as np
    from scipy import signal as sps
    import scripts.ictal_analysis as ia

    ann = [a for a in ia.parse_seizure_annotations() if a["edf_on_disk"] and a["seizures"]]
    if not ann:
        return {"available": False, "error": "no annotated seizure EDF on disk"}
    rec = next((a for a in ann if file and (a["file"] == file or a["edf_path"] == file)), ann[0])
    p = ROOT / rec["edf_path"]
    raw = mne.io.read_raw_edf(str(p), preload=True, verbose="ERROR")
    sf = float(raw.info["sfreq"]); dur = float(raw.times[-1])
    data = raw.get_data()
    win = int(window_s * sf)
    n_win = data.shape[1] // win

    # per-window mean broadband (1-30Hz) power across channels
    powers = []
    for w in range(n_win):
        seg = data[:, w * win:(w + 1) * win]
        f, ps = sps.welch(seg, fs=sf, nperseg=int(min(sf * 2, seg.shape[1])))
        powers.append(float(ps[:, (f >= 1) & (f < 30)].sum(axis=1).mean()))
    powers = np.array(powers)
    # robust threshold: median + k×MAD
    med = np.median(powers); mad = np.median(np.abs(powers - med)) + 1e-20
    thr = med + k * mad
    det_idx = np.where(powers > thr)[0]

    sz_windows = [(s["start_s"], s["end_s"]) for s in rec["seizures"]]
    def in_seizure(w):
        ws, we = w * window_s, (w + 1) * window_s
        return any(we >= s and ws <= e for s, e in sz_windows)

    tp_windows = [int(w) for w in det_idx if in_seizure(w)]
    fp_windows = [{"window": int(w), "time_s": round(w * window_s, 1)} for w in det_idx if not in_seizure(w)]
    seizures_detected = sum(1 for s, e in sz_windows
                            if any(in_seizure(w) and (w * window_s) <= e and ((w + 1) * window_s) >= s for w in tp_windows))
    hours = dur / 3600.0
    return {
        "available": True, "file": rec["file"], "recording_hours": round(hours, 2),
        "detector": {"method": "5s windows, broadband power > median + 6×MAD", "threshold_k": k},
        "n_seizures_annotated": len(sz_windows),
        "seizures_detected": seizures_detected,
        "sensitivity": round(seizures_detected / len(sz_windows), 2) if sz_windows else None,
        "true_positive_windows": len(tp_windows),
        "false_alarms": len(fp_windows),
        "false_alarms_per_hour": round(len(fp_windows) / hours, 1) if hours else None,
        "false_alarm_windows": fp_windows[:20],
        "verdict": ("acceptable" if (len(fp_windows) / hours if hours else 99) <= 6 else
                    "high false-alarm rate — detector needs tuning"),
        "note": ("Validates a SIMPLE power detector vs CHB-MIT annotations — illustrates the "
                 "sensitivity/false-alarm tradeoff, not a production seizure detector."),
        "source": "CHB-MIT annotated EDF via MNE/SciPy.",
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    r = review()
    if r["available"]:
        print("False Alarm Review:", r["file"])
        print(f"  sensitivity {r['sensitivity']} | TP windows {r['true_positive_windows']} | "
              f"false alarms {r['false_alarms']} ({r['false_alarms_per_hour']}/hr) → {r['verdict']}")
    else:
        print(r)
