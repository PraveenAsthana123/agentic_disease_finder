#!/usr/bin/env python3
"""Sleep State Dashboard — sleep architecture from real Sleep-EDF hypnograms.

Parses Sleep-EDFx PSG + Hypnogram.edf annotation pairs (MNE read_annotations)
into a sleep-stage profile: time + % per stage (W/N1/N2/N3/REM), Total Sleep
Time, Sleep Efficiency, REM%, deep-sleep%, and stage-transition count.

100% real (annotated hypnograms via MNE) — runs under the canonical venv.
"""
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLEEP_DIRS = [
    "data/eeg_datasets/sleep/sleep_edf/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette",
    "data/eeg_datasets/sleep/sleep_edf/physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry",
]
# Sleep-EDF stage label → canonical stage
STAGE_MAP = {
    "Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
    "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM",
    "Sleep stage ?": "?", "Movement time": "?",
}
SLEEP_STAGES = {"N1", "N2", "N3", "REM"}


def list_sleep_recordings(limit: int = 40):
    recs = []
    for d in SLEEP_DIRS:
        hyps = sorted(glob.glob(str(ROOT / d / "*Hypnogram.edf")))
        for h in hyps:
            base = Path(h).name.replace("-Hypnogram.edf", "")
            psg = glob.glob(str(ROOT / d / f"{base[:7]}*-PSG.edf"))
            recs.append({"hypnogram": str(Path(h).relative_to(ROOT)),
                         "psg": (str(Path(psg[0]).relative_to(ROOT)) if psg else None),
                         "dataset": "cassette" if "cassette" in d else "telemetry"})
    return {"available": bool(recs), "n_total": len(recs), "recordings": recs[:limit]}


def sleep_architecture(hypnogram: str = None) -> dict:
    import mne

    recs = list_sleep_recordings(limit=9999)["recordings"]
    if not recs:
        return {"available": False, "error": "no hypnograms on disk"}
    rec = next((r for r in recs if hypnogram and r["hypnogram"] == hypnogram), recs[0])

    ann = mne.read_annotations(str(ROOT / rec["hypnogram"]))
    # Sleep-EDF records ~24h with long daytime wake padding → clip to the SLEEP
    # PERIOD (first → last sleep epoch) so efficiency/WASO are clinically sensible.
    epochs = [(STAGE_MAP.get(str(d), "?"), float(dur)) for d, dur in zip(ann.description, ann.duration)]
    sleep_idx = [i for i, (st, _) in enumerate(epochs) if st in SLEEP_STAGES]
    if sleep_idx:
        epochs = epochs[sleep_idx[0]:sleep_idx[-1] + 1]
    secs, seq = {}, []
    for st, dur in epochs:
        secs[st] = secs.get(st, 0.0) + dur
        seq.append(st)
    tst = sum(v for k, v in secs.items() if k in SLEEP_STAGES)
    spt = sum(v for k, v in secs.items() if k != "?")  # sleep period time (excludes daytime wake)
    time_in_bed = spt
    # stage transitions (ignore repeats + unknown)
    trans = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1] and seq[i] != "?" and seq[i - 1] != "?")

    def pct(s):
        return round(100 * secs.get(s, 0.0) / (tst or 1), 1)
    # sleep stages as % of TST; W within sleep period = WASO, as % of SPT
    stages = {s: {"minutes": round(secs.get(s, 0.0) / 60, 1),
                  "pct_of_sleep": round(100 * secs.get(s, 0.0) / (spt or 1), 1) if s == "W" else pct(s)}
              for s in ["W", "N1", "N2", "N3", "REM"]}
    flags = []
    if pct("REM") < 15:
        flags.append("low REM (<15% of sleep)")
    if pct("N3") < 10:
        flags.append("low deep sleep N3 (<10%)")
    se = round(100 * tst / (time_in_bed or 1), 1)
    if se < 80:
        flags.append(f"reduced sleep efficiency ({se}%)")
    return {
        "available": True, "hypnogram": Path(rec["hypnogram"]).name, "dataset": rec["dataset"],
        "total_sleep_time_min": round(tst / 60, 1),
        "time_in_bed_min": round(time_in_bed / 60, 1),
        "sleep_efficiency_pct": se,
        "stages": stages,
        "rem_pct": pct("REM"), "deep_sleep_pct": pct("N3"),
        "stage_transitions": trans,
        "flags": flags or ["within normative ranges"],
        "quality": "PASS" if not flags else "REVIEW",
        "n_recordings": len(recs),
        "source": "Sleep-EDFx Hypnogram annotations via MNE (real staging).",
        "note": "Sleep architecture from expert-scored hypnogram. Normative flags are screening-grade.",
    }


if __name__ == "__main__":
    r = sleep_architecture()
    print("Sleep architecture:", r["hypnogram"])
    print("  TST:", r["total_sleep_time_min"], "min | efficiency:", r["sleep_efficiency_pct"], "%")
    print("  stages:", {k: v["pct_of_sleep"] for k, v in r["stages"].items()})
    print("  quality:", r["quality"], r["flags"])
