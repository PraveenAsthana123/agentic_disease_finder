"""
seed_hospitalization.py
Seed the `hospitalization` table in data/clinical.db with realistic
epilepsy hospitalization data for 30 patients (EPAT001–EPAT030).
~120 rows, spanning 2024-01 through 2026-06.
"""

import json
import random
import sqlite3
from datetime import date, timedelta

DB_PATH = "data/clinical.db"
random.seed(42)

# ─── constants ────────────────────────────────────────────────────────────────

PATIENT_IDS = [f"EPAT{i:03d}" for i in range(1, 31)]

ADMISSION_TYPES = ["emergency", "planned", "transfer", "observation"]
ADMISSION_REASONS = [
    "status_epilepticus",
    "seizure_cluster",
    "medication_adjustment",
    "pre_surgical_evaluation",
    "post_surgical_monitoring",
    "breakthrough_seizure",
    "aed_toxicity",
    "first_seizure",
    "video_eeg_monitoring",
]
WARDS = [
    "Epilepsy Monitoring Unit",
    "Neurology Ward",
    "ICU",
    "Emergency",
    "Surgical Recovery",
]
PHYSICIANS = [
    "Dr. Sharma",
    "Dr. Patel",
    "Dr. Chen",
    "Dr. Kim",
    "Dr. Wilson",
    "Dr. Rodriguez",
]
DISCHARGE_DISPOSITIONS = ["home", "rehabilitation", "transferred", "ama"]
COMPLICATIONS = [None, "none", "none", "none", "infection", "medication_reaction", "fall"]
INSURANCE_TYPES = ["public", "private", "private", "self_pay", "military"]

# Ward-based daily cost multiplier (USD / day)
WARD_DAILY_COST = {
    "Epilepsy Monitoring Unit": 3500,
    "Neurology Ward": 2200,
    "ICU": 6500,
    "Emergency": 4000,
    "Surgical Recovery": 4800,
}

# Date range: 2024-01-01 to 2026-06-01
START_DATE = date(2024, 1, 1)
END_DATE   = date(2026, 6, 1)


# ─── helpers ──────────────────────────────────────────────────────────────────

def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def admission_type_for_reason(reason: str) -> str:
    mapping = {
        "status_epilepticus":       "emergency",
        "seizure_cluster":          "emergency",
        "aed_toxicity":             "emergency",
        "first_seizure":            "emergency",
        "breakthrough_seizure":     random.choice(["emergency", "observation"]),
        "medication_adjustment":    random.choice(["planned", "observation"]),
        "pre_surgical_evaluation":  "planned",
        "post_surgical_monitoring": "planned",
        "video_eeg_monitoring":     random.choice(["planned", "observation"]),
    }
    return mapping.get(reason, random.choice(ADMISSION_TYPES))


def ward_for_reason(reason: str) -> str:
    mapping = {
        "status_epilepticus":       "ICU",
        "aed_toxicity":             random.choice(["ICU", "Emergency"]),
        "first_seizure":            "Emergency",
        "seizure_cluster":          random.choice(["Emergency", "Neurology Ward"]),
        "breakthrough_seizure":     random.choice(["Emergency", "Neurology Ward"]),
        "medication_adjustment":    random.choice(["Neurology Ward", "Epilepsy Monitoring Unit"]),
        "pre_surgical_evaluation":  "Epilepsy Monitoring Unit",
        "post_surgical_monitoring": "Surgical Recovery",
        "video_eeg_monitoring":     "Epilepsy Monitoring Unit",
    }
    return mapping.get(reason, random.choice(WARDS))


def los_for_reason_and_ward(reason: str, ward: str) -> int:
    """Length of stay in days — realistic per ward/reason."""
    if ward == "ICU":
        return random.randint(3, 10)
    if ward == "Emergency":
        return random.randint(1, 3)
    if ward == "Surgical Recovery":
        return random.randint(4, 12)
    if ward == "Epilepsy Monitoring Unit":
        return random.randint(3, 7)
    # Neurology Ward
    if reason in ("medication_adjustment", "aed_toxicity"):
        return random.randint(2, 6)
    return random.randint(2, 8)


def compute_cost(ward: str, los: int) -> int:
    base = WARD_DAILY_COST[ward] * los
    # Add fixed overhead (procedures, labs, etc.)
    overhead = random.randint(800, 5000)
    total = base + overhead
    # Clamp to [3000, 85000]
    return max(3000, min(85000, total))


def seizure_free_at_discharge(reason: str, admission_type: str) -> bool:
    """
    Patients admitted for surgical or monitoring reasons more likely seizure-free.
    Emergency admissions less likely.
    """
    if reason in ("post_surgical_monitoring", "video_eeg_monitoring", "pre_surgical_evaluation"):
        return random.random() < 0.70
    if admission_type == "emergency":
        return random.random() < 0.45
    return random.random() < 0.60


# ─── build rows ───────────────────────────────────────────────────────────────

def build_rows(n_still_admitted: int = 4) -> list[dict]:
    rows = []

    # Decide admission count per patient: mix of 1 and 2-6
    # 30 patients, target ~120 total
    admission_counts = {}
    for pid in PATIENT_IDS:
        admission_counts[pid] = random.choices(
            [1, 2, 3, 4, 5, 6],
            weights=[20, 30, 25, 15, 7, 3],
        )[0]

    # Normalise total toward 120
    total = sum(admission_counts.values())
    # Small nudge to get close to 120 without breaking per-patient realism
    while total < 115:
        pid = random.choice(PATIENT_IDS)
        if admission_counts[pid] < 6:
            admission_counts[pid] += 1
            total += 1
    while total > 125:
        pid = random.choice([p for p in PATIENT_IDS if admission_counts[p] > 1])
        admission_counts[pid] -= 1
        total -= 1

    # Pick patients who are currently admitted (no discharge yet)
    still_admitted_patients = random.sample(PATIENT_IDS, n_still_admitted)

    for pid in PATIENT_IDS:
        count = admission_counts[pid]
        # Give patient a "primary" reason tendency
        primary_reason = random.choice(ADMISSION_REASONS)

        for adm_idx in range(count):
            reason = (
                primary_reason
                if random.random() < 0.55
                else random.choice(ADMISSION_REASONS)
            )
            admission_type = admission_type_for_reason(reason)
            ward = ward_for_reason(reason)
            los = los_for_reason_and_ward(reason, ward)
            physician = random.choice(PHYSICIANS)
            insurance = random.choices(
                INSURANCE_TYPES,
                weights=[30, 45, 10, 5, 10],
            )[0]

            # Admission date — spread across date range, keep patient's admissions ordered
            admission_date = random_date(START_DATE, END_DATE - timedelta(days=los + 1))

            # Last admission for a "still admitted" patient → no discharge
            is_last = adm_idx == count - 1
            still_in = pid in still_admitted_patients and is_last

            if still_in:
                # Admitted recently (within last 10 days) → no discharge
                admission_date = date(2026, 7, 16) - timedelta(days=random.randint(1, 10))
                discharge_date = None
                discharge_disposition = None
                los_actual = None
                complications = random.choice(COMPLICATIONS)
                total_cost = compute_cost(ward, random.randint(1, 5))  # partial stay cost
                sf = None
                readmit_30d = False
            else:
                discharge_date = admission_date + timedelta(days=los)
                discharge_disposition = random.choices(
                    DISCHARGE_DISPOSITIONS,
                    weights=[70, 10, 10, 10],
                )[0]
                los_actual = los
                complications = random.choice(COMPLICATIONS)
                total_cost = compute_cost(ward, los)
                sf = seizure_free_at_discharge(reason, admission_type)
                # Readmission within 30 days — more likely for emergency / status_epilepticus
                readmit_base = 0.18 if reason in ("status_epilepticus", "seizure_cluster") else 0.08
                readmit_30d = random.random() < readmit_base

            fields = {
                "admission_date":          str(admission_date),
                "discharge_date":          str(discharge_date) if discharge_date else None,
                "admission_type":          admission_type,
                "admission_reason":        reason,
                "ward":                    ward,
                "attending_physician":     physician,
                "length_of_stay_days":     los_actual,
                "discharge_disposition":   discharge_disposition,
                "readmission_within_30d":  readmit_30d,
                "seizure_free_at_discharge": sf,
                "complications":           complications,
                "insurance_type":          insurance,
                "total_cost_usd":          total_cost,
            }

            rows.append({
                "patient_id":  pid,
                "fields_json": json.dumps(fields),
                "created_at":  str(admission_date),
            })

    return rows


# ─── insert ───────────────────────────────────────────────────────────────────

def seed(db_path: str = DB_PATH):
    rows = build_rows(n_still_admitted=4)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executemany(
        "INSERT INTO hospitalization (patient_id, fields_json, created_at) VALUES (?, ?, ?)",
        [(r["patient_id"], r["fields_json"], r["created_at"]) for r in rows],
    )
    conn.commit()

    # ── summary stats ──────────────────────────────────────────────────────────
    cur.execute("SELECT COUNT(*) FROM hospitalization")
    total = cur.fetchone()[0]

    parsed = [json.loads(r["fields_json"]) for r in rows]

    reasons     = {}
    wards       = {}
    types_count = {}
    costs       = []
    los_list    = []
    still_in    = 0
    readmit     = 0
    seizure_free = 0
    discharged   = 0

    for f in parsed:
        reasons[f["admission_reason"]]  = reasons.get(f["admission_reason"], 0) + 1
        wards[f["ward"]]                = wards.get(f["ward"], 0) + 1
        types_count[f["admission_type"]] = types_count.get(f["admission_type"], 0) + 1
        costs.append(f["total_cost_usd"])
        if f["discharge_date"] is None:
            still_in += 1
        else:
            discharged += 1
            los_list.append(f["length_of_stay_days"])
            if f["readmission_within_30d"]:
                readmit += 1
            if f["seizure_free_at_discharge"]:
                seizure_free += 1

    print(f"\n{'='*55}")
    print(f"  Hospitalization Seed — Summary")
    print(f"{'='*55}")
    print(f"  Total rows inserted : {total}")
    print(f"  Patients covered    : {len(set(r['patient_id'] for r in rows))}")
    print(f"  Currently admitted  : {still_in}")
    print(f"  Discharged          : {discharged}")
    print(f"  Avg LOS (days)      : {sum(los_list)/len(los_list):.1f}")
    print(f"  Readmit within 30d  : {readmit} ({100*readmit/discharged:.1f}%)")
    print(f"  Seizure-free @ d/c  : {seizure_free} ({100*seizure_free/discharged:.1f}%)")
    print(f"  Avg cost (USD)      : ${sum(costs)/len(costs):,.0f}")
    print(f"\n  Admission reasons:")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {k:<35} {v}")
    print(f"\n  Wards:")
    for k, v in sorted(wards.items(), key=lambda x: -x[1]):
        print(f"    {k:<35} {v}")
    print(f"\n  Admission types:")
    for k, v in sorted(types_count.items(), key=lambda x: -x[1]):
        print(f"    {k:<35} {v}")
    print(f"{'='*55}\n")

    conn.close()


if __name__ == "__main__":
    seed()
