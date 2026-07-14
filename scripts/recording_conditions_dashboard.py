"""Recording Conditions Dashboard — EEG recording environment and activation procedure analytics.

EEG recording conditions are critical determinants of diagnostic yield in epilepsy.
Standardised activation procedures (hyperventilation, photic stimulation, sleep recording,
eyes-open/closed testing) can provoke epileptiform discharges that would otherwise remain
undetected during routine awake EEG.  The ACNS and ILAE recommend that all four procedures
be attempted in every routine EEG unless medically contraindicated.  Patient state (awake,
drowsy, asleep) and cooperation level directly affect signal quality; drowsiness and sleep
can unmask centro-temporal spikes, while poor cooperation introduces movement artefact that
degrades interpretation.  Tracking these conditions per recording ensures protocol adherence,
supports quality-improvement programmes, and strengthens the evidentiary basis of each report.

All data from REAL recording_conditions rows in data/clinical.db.
"""
import sqlite3, json
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    with _conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def _parse_rows():
    """Fetch all recording_conditions rows and parse fields_json."""
    raw = _rows("SELECT id, patient_id, fields_json, created_at FROM recording_conditions ORDER BY patient_id")
    parsed = []
    for r in raw:
        fields = json.loads(r["fields_json"]) if r["fields_json"] else {}
        parsed.append({
            "id": r["id"],
            "patient_id": r["patient_id"],
            "eyes_open": bool(fields.get("eyes_open", False)),
            "hyperventilation": bool(fields.get("hyperventilation", False)),
            "photic_stimulation": bool(fields.get("photic_stimulation", False)),
            "sleep_recorded": bool(fields.get("sleep_recorded", False)),
            "patient_state": fields.get("patient_state", "unknown"),
            "cooperation": fields.get("cooperation", "unknown"),
            "created_at": r["created_at"],
        })
    return parsed


def overview():
    """Aggregate recording-condition statistics across all patients.

    Returns dict with total_recordings, activation_rates, patient_state_distribution,
    cooperation_distribution, protocol_completeness, and quality_summary.
    """
    rows = _parse_rows()
    if not rows:
        return {"total_recordings": 0, "message": "No recording conditions data yet"}

    total = len(rows)

    # Activation rates (percentage of recordings including each procedure)
    eyes_ct = sum(1 for r in rows if r["eyes_open"])
    hv_ct = sum(1 for r in rows if r["hyperventilation"])
    ps_ct = sum(1 for r in rows if r["photic_stimulation"])
    sl_ct = sum(1 for r in rows if r["sleep_recorded"])

    activation_rates = {
        "eyes_open_pct": round(eyes_ct / total * 100, 1),
        "hyperventilation_pct": round(hv_ct / total * 100, 1),
        "photic_stimulation_pct": round(ps_ct / total * 100, 1),
        "sleep_recorded_pct": round(sl_ct / total * 100, 1),
    }

    # Patient state distribution
    state_dist = {}
    for r in rows:
        st = r["patient_state"]
        state_dist[st] = state_dist.get(st, 0) + 1

    # Cooperation distribution
    coop_dist = {}
    for r in rows:
        co = r["cooperation"]
        coop_dist[co] = coop_dist.get(co, 0) + 1

    # Protocol completeness: all 4 activation procedures true
    complete_ct = sum(
        1 for r in rows
        if r["eyes_open"] and r["hyperventilation"] and r["photic_stimulation"] and r["sleep_recorded"]
    )
    protocol_completeness = round(complete_ct / total * 100, 1)

    # Quality summary: excellent+good vs fair+poor cooperation
    eg_ct = sum(1 for r in rows if r["cooperation"] in ("excellent", "good"))
    fp_ct = sum(1 for r in rows if r["cooperation"] in ("fair", "poor"))
    quality_summary = {
        "excellent_good_pct": round(eg_ct / total * 100, 1),
        "fair_poor_pct": round(fp_ct / total * 100, 1),
    }

    return {
        "total_recordings": total,
        "activation_rates": activation_rates,
        "patient_state_distribution": state_dist,
        "cooperation_distribution": coop_dist,
        "protocol_completeness": protocol_completeness,
        "quality_summary": quality_summary,
    }


def breakdown():
    """Per-patient recording condition detail.

    Returns dict with patients list containing each patient's activation
    procedures, state, cooperation, activations_completed count, and
    protocol_complete flag.
    """
    rows = _parse_rows()
    if not rows:
        return {"patients": []}

    patients = []
    for r in rows:
        bools = [r["eyes_open"], r["hyperventilation"], r["photic_stimulation"], r["sleep_recorded"]]
        activations_completed = sum(1 for b in bools if b)
        protocol_complete = activations_completed == 4
        patients.append({
            "patient_id": r["patient_id"],
            "eyes_open": r["eyes_open"],
            "hyperventilation": r["hyperventilation"],
            "photic_stimulation": r["photic_stimulation"],
            "sleep_recorded": r["sleep_recorded"],
            "patient_state": r["patient_state"],
            "cooperation": r["cooperation"],
            "activations_completed": activations_completed,
            "protocol_complete": protocol_complete,
            "created_at": r["created_at"],
        })

    return {"patients": patients}


def definitions():
    """Clinical definitions for EEG recording condition fields.

    Returns title, description, and glossary of terms with their clinical
    significance in epilepsy EEG interpretation.
    """
    return {
        "title": "EEG Recording Conditions",
        "description": (
            "Recording conditions document the environment, activation procedures, "
            "and patient state during an EEG study.  These factors directly influence "
            "the diagnostic yield of the recording and must be reported per ACNS and "
            "ILAE guidelines to ensure reproducibility and clinical validity."
        ),
        "terms": [
            {
                "term": "Eyes Open / Closed",
                "definition": (
                    "Testing posterior dominant rhythm (PDR) reactivity by asking the "
                    "patient to open and close their eyes.  A well-formed, reactive PDR "
                    "that attenuates with eye opening is a key marker of normal cortical "
                    "function; its absence or asymmetry may indicate focal pathology."
                ),
            },
            {
                "term": "Hyperventilation",
                "definition": (
                    "Three minutes of deep, rapid breathing that lowers PaCO2 and "
                    "produces cerebral vasoconstriction.  Provokes generalised spike-wave "
                    "discharges in absence epilepsy and may enhance focal slowing or "
                    "epileptiform activity in structural lesions."
                ),
            },
            {
                "term": "Photic Stimulation",
                "definition": (
                    "Intermittent stroboscopic light delivered at frequencies from 1 to "
                    "30 Hz.  Assesses for photoparoxysmal response (PPR) — generalised "
                    "spike-wave or polyspike-wave discharges time-locked to the flash, "
                    "seen in photosensitive epilepsies such as juvenile myoclonic epilepsy."
                ),
            },
            {
                "term": "Sleep Recording",
                "definition": (
                    "Capture of drowsiness and sleep stages during the EEG.  NREM sleep "
                    "(especially N2) activates epileptiform discharges in many focal "
                    "epilepsies, particularly benign epilepsy with centrotemporal spikes "
                    "(BECTS) and frontal-lobe epilepsy.  Sleep deprivation protocols "
                    "further increase yield."
                ),
            },
            {
                "term": "Patient State",
                "definition": (
                    "Classification of the patient as awake, drowsy, or asleep at the "
                    "time of recording.  Background EEG patterns differ markedly across "
                    "states; accurate labelling is essential for correct interpretation "
                    "of normal variants (e.g., vertex sharp waves in drowsiness) versus "
                    "pathological discharges."
                ),
            },
            {
                "term": "Cooperation",
                "definition": (
                    "Qualitative rating (excellent, good, fair, poor) of the patient's "
                    "ability to remain still and follow instructions.  Poor cooperation "
                    "introduces muscle and movement artefact that can obscure epileptiform "
                    "activity and reduce recording interpretability."
                ),
            },
            {
                "term": "Protocol Completeness",
                "definition": (
                    "Whether all four standard activation procedures (eyes open/closed, "
                    "hyperventilation, photic stimulation, sleep recording) were performed.  "
                    "ACNS and ILAE minimum standards recommend completion of all procedures "
                    "in every routine EEG to maximise sensitivity for epileptiform "
                    "abnormalities."
                ),
            },
        ],
    }
