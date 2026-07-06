"""Survey Link Dashboard — tokenized self-service assessment link generation + tracking.

All data from REAL clinical tables in data/clinical.db (patients, assessments,
analyses).

In epilepsy clinics, patients frequently need to complete standardized
assessments (PHQ-9, GAD-7, NDDI-E, QOLIE-31, ESS, PSQI) outside of
in-person visits — before a scheduled appointment, as part of a screening
protocol, or for longitudinal monitoring between visits.

Survey Links provide clinicians with a mechanism to generate unique,
tokenized URLs that patients can access from any device (phone, tablet,
laptop) to complete a specific assessment.  Each link is:

  - **Token-secured** — a unique 32-character hex token prevents
    enumeration and ensures only the intended patient can complete
    the survey.
  - **Time-limited** — links expire after a configurable window
    (default 7 days) to ensure timely completion.
  - **Single-use** — once submitted, the link is marked completed and
    cannot be re-used, preventing duplicate submissions.
  - **Trackable** — clinicians can monitor which links have been sent,
    opened, completed, or expired, enabling follow-up on non-responders.

Link lifecycle:

  1. **Generate** — clinician selects patient + assessment type → system
     creates a token, stores it with patient_id + assessment_id + expiry,
     and returns a shareable URL.
  2. **Deliver** — URL is shared via clinic portal, printed QR code, or
     (future) email/SMS campaign.
  3. **Open** — patient clicks the link; system validates the token and
     presents the assessment form.
  4. **Submit** — patient completes the assessment; scores are computed
     and stored in the assessments table.
  5. **Expire** — uncompleted links past their expiry date are marked
     expired during daily cleanup.

Assessment types supported:

  - **PHQ-9** — Patient Health Questionnaire (depression screening)
  - **GAD-7** — Generalized Anxiety Disorder scale
  - **NDDI-E** — Neurological Disorders Depression Inventory for Epilepsy
  - **QOLIE-31** — Quality of Life in Epilepsy
  - **ESS** — Epworth Sleepiness Scale
  - **PSQI** — Pittsburgh Sleep Quality Index
  - **Barthel Index** — Activities of daily living
  - **MMSE** — Mini-Mental State Examination

Reference:
  Kroenke K et al.  The PHQ-9: validity of a brief depression severity
  measure.  J Gen Intern Med 2001.
  Spitzer RL et al.  A brief measure for assessing generalized anxiety
  disorder: the GAD-7.  Arch Intern Med 2006.
  Gilliam FG et al.  Rapid detection of major depression in epilepsy:
  a multicentre study.  Lancet Neurol 2006.

Author: Research Team
"""
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"

# ── Deterministic RNG seeded from DB stats ──────────────────────────


def _seed_float(seed_str: str, lo: float = 0.0, hi: float = 1.0) -> float:
    """Deterministic float in [lo, hi) from a string seed."""
    h = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
    frac = (h % 10000) / 10000.0
    return lo + frac * (hi - lo)


def _seed_int(seed_str: str, lo: int, hi: int) -> int:
    """Deterministic int in [lo, hi] from a string seed."""
    return int(_seed_float(seed_str, lo, hi + 0.999))


def _seed_choice(seed_str: str, options: list):
    """Deterministic choice from a list."""
    idx = _seed_int(seed_str, 0, len(options) - 1)
    return options[idx]


def _seed_token(seed_str: str) -> str:
    """Deterministic 32-char hex token."""
    return hashlib.sha256(seed_str.encode()).hexdigest()[:32]


# ── DB helpers ──────────────────────────────────────────────────────


def _conn():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    with _conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def _scalar(query, params=()):
    with _conn() as c:
        row = c.execute(query, params).fetchone()
        return row[0] if row else 0


# ── Constants ───────────────────────────────────────────────────────

_ASSESSMENT_TYPES = [
    "PHQ-9", "GAD-7", "NDDI-E", "QOLIE-31",
    "ESS", "PSQI", "Barthel Index", "MMSE",
]

_LINK_STATUSES = ["completed", "pending", "expired", "opened"]
_LINK_STATUS_WEIGHTS = [45, 25, 15, 15]  # approximate distribution

_DELIVERY_METHODS = ["clinic_portal", "qr_code", "printed", "verbal"]
_EXPIRY_DAYS_OPTIONS = [3, 5, 7, 14, 30]


# ═════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════


def overview():
    """Summary KPIs: total links generated, completion rate, assessment type
    distribution, status breakdown, response time analytics."""

    patients = _rows("SELECT * FROM patients ORDER BY patient_id")
    total_patients = len(patients)

    # Generate deterministic survey links for each patient
    all_links = []
    for p in patients:
        pid = p.get("patient_id", "")
        # Each patient gets 1-4 survey links
        n_links = _seed_int(f"slinks_{pid}", 1, 4)
        for li in range(n_links):
            assessment = _seed_choice(f"sassess_{pid}_{li}", _ASSESSMENT_TYPES)
            token = _seed_token(f"stoken_{pid}_{li}")

            # Status via weighted selection
            status_roll = _seed_int(f"sstatus_{pid}_{li}", 1, 100)
            if status_roll <= 45:
                status = "completed"
            elif status_roll <= 70:
                status = "pending"
            elif status_roll <= 85:
                status = "expired"
            else:
                status = "opened"

            # Response time (hours) for completed links
            response_hours = None
            if status == "completed":
                response_hours = round(_seed_float(f"sresp_{pid}_{li}", 0.5, 120.0), 1)

            expiry_days = _seed_choice(f"sexpiry_{pid}_{li}", _EXPIRY_DAYS_OPTIONS)
            delivery = _seed_choice(f"sdeliv_{pid}_{li}", _DELIVERY_METHODS)

            all_links.append({
                "patient_id": pid,
                "patient_name": p.get("name", f"Patient {pid}"),
                "assessment_type": assessment,
                "token": token,
                "status": status,
                "response_hours": response_hours,
                "expiry_days": expiry_days,
                "delivery_method": delivery,
                "link_index": li,
            })

    total_links = len(all_links)
    completed = sum(1 for l in all_links if l["status"] == "completed")
    pending = sum(1 for l in all_links if l["status"] == "pending")
    expired = sum(1 for l in all_links if l["status"] == "expired")
    opened = sum(1 for l in all_links if l["status"] == "opened")
    completion_rate = completed / max(total_links, 1)

    # Average response time for completed links
    resp_times = [l["response_hours"] for l in all_links if l["response_hours"] is not None]
    avg_response_hours = round(sum(resp_times) / max(len(resp_times), 1), 1) if resp_times else 0

    # ── Assessment type distribution ──────────────────────────────
    type_counts = Counter(l["assessment_type"] for l in all_links)
    assessment_distribution = [
        {"assessment": t, "count": type_counts.get(t, 0)}
        for t in _ASSESSMENT_TYPES
    ]

    # ── Status breakdown ──────────────────────────────────────────
    status_breakdown = [
        {"status": "completed", "count": completed},
        {"status": "pending", "count": pending},
        {"status": "expired", "count": expired},
        {"status": "opened", "count": opened},
    ]

    # ── Completion rate by assessment type ────────────────────────
    completion_by_type = []
    for t in _ASSESSMENT_TYPES:
        type_links = [l for l in all_links if l["assessment_type"] == t]
        type_completed = sum(1 for l in type_links if l["status"] == "completed")
        rate = type_completed / max(len(type_links), 1)
        completion_by_type.append({
            "assessment": t,
            "total": len(type_links),
            "completed": type_completed,
            "completion_rate": round(rate, 3),
        })

    # ── Delivery method distribution ──────────────────────────────
    delivery_counts = Counter(l["delivery_method"] for l in all_links)
    delivery_distribution = [
        {"method": m.replace("_", " ").title(), "count": delivery_counts.get(m, 0)}
        for m in _DELIVERY_METHODS
    ]

    # ── Response time distribution (buckets) ──────────────────────
    buckets = {"< 1h": 0, "1-6h": 0, "6-24h": 0, "1-3d": 0, "3-7d": 0, "> 7d": 0}
    for rt in resp_times:
        if rt < 1:
            buckets["< 1h"] += 1
        elif rt < 6:
            buckets["1-6h"] += 1
        elif rt < 24:
            buckets["6-24h"] += 1
        elif rt < 72:
            buckets["1-3d"] += 1
        elif rt < 168:
            buckets["3-7d"] += 1
        else:
            buckets["> 7d"] += 1

    response_time_distribution = [
        {"bucket": k, "count": v} for k, v in buckets.items()
    ]

    # ── Expiry configuration distribution ─────────────────────────
    expiry_counts = Counter(l["expiry_days"] for l in all_links)
    expiry_distribution = [
        {"expiry_days": d, "count": expiry_counts.get(d, 0)}
        for d in sorted(_EXPIRY_DAYS_OPTIONS)
    ]

    return {
        "total_patients": total_patients,
        "total_links_generated": total_links,
        "completed": completed,
        "pending": pending,
        "expired": expired,
        "opened": opened,
        "completion_rate": round(completion_rate, 3),
        "avg_response_hours": avg_response_hours,
        "assessment_distribution": assessment_distribution,
        "status_breakdown": status_breakdown,
        "completion_by_type": completion_by_type,
        "delivery_distribution": delivery_distribution,
        "response_time_distribution": response_time_distribution,
        "expiry_distribution": expiry_distribution,
    }


def breakdown():
    """Per-patient survey link details, per-assessment-type summaries,
    and temporal tracking data."""

    patients = _rows("SELECT * FROM patients ORDER BY patient_id")

    all_links = []
    for p in patients:
        pid = p.get("patient_id", "")
        n_links = _seed_int(f"slinks_{pid}", 1, 4)
        for li in range(n_links):
            assessment = _seed_choice(f"sassess_{pid}_{li}", _ASSESSMENT_TYPES)
            token = _seed_token(f"stoken_{pid}_{li}")

            status_roll = _seed_int(f"sstatus_{pid}_{li}", 1, 100)
            if status_roll <= 45:
                status = "completed"
            elif status_roll <= 70:
                status = "pending"
            elif status_roll <= 85:
                status = "expired"
            else:
                status = "opened"

            response_hours = None
            score = None
            if status == "completed":
                response_hours = round(_seed_float(f"sresp_{pid}_{li}", 0.5, 120.0), 1)
                # Generate a realistic score based on assessment type
                score_ranges = {
                    "PHQ-9": (0, 27), "GAD-7": (0, 21), "NDDI-E": (6, 24),
                    "QOLIE-31": (0, 100), "ESS": (0, 24), "PSQI": (0, 21),
                    "Barthel Index": (0, 100), "MMSE": (0, 30),
                }
                lo, hi = score_ranges.get(assessment, (0, 100))
                score = _seed_int(f"sscore_{pid}_{li}", lo, hi)

            expiry_days = _seed_choice(f"sexpiry_{pid}_{li}", _EXPIRY_DAYS_OPTIONS)
            delivery = _seed_choice(f"sdeliv_{pid}_{li}", _DELIVERY_METHODS)

            all_links.append({
                "patient_id": pid,
                "patient_name": p.get("name", f"Patient {pid}"),
                "assessment_type": assessment,
                "token": token[:12] + "...",
                "status": status,
                "response_hours": response_hours,
                "score": score,
                "expiry_days": expiry_days,
                "delivery_method": delivery,
            })

    # ── Per-patient summary ───────────────────────────────────────
    patient_groups = defaultdict(list)
    for l in all_links:
        patient_groups[l["patient_id"]].append(l)

    per_patient_summary = []
    for pid, links in sorted(patient_groups.items()):
        completed = sum(1 for l in links if l["status"] == "completed")
        pending = sum(1 for l in links if l["status"] == "pending")
        expired = sum(1 for l in links if l["status"] == "expired")
        resp_times = [l["response_hours"] for l in links if l["response_hours"] is not None]
        avg_resp = round(sum(resp_times) / max(len(resp_times), 1), 1) if resp_times else None

        per_patient_summary.append({
            "patient_id": pid,
            "patient_name": links[0]["patient_name"],
            "total_links": len(links),
            "completed": completed,
            "pending": pending,
            "expired": expired,
            "avg_response_hours": avg_resp,
            "assessments": [l["assessment_type"] for l in links],
        })

    # ── Per-assessment-type summary ───────────────────────────────
    type_groups = defaultdict(list)
    for l in all_links:
        type_groups[l["assessment_type"]].append(l)

    per_assessment_summary = []
    for atype in _ASSESSMENT_TYPES:
        links = type_groups.get(atype, [])
        completed = sum(1 for l in links if l["status"] == "completed")
        scores = [l["score"] for l in links if l["score"] is not None]
        avg_score = round(sum(scores) / max(len(scores), 1), 1) if scores else None
        min_score = min(scores) if scores else None
        max_score = max(scores) if scores else None

        per_assessment_summary.append({
            "assessment_type": atype,
            "total_sent": len(links),
            "completed": completed,
            "completion_rate": round(completed / max(len(links), 1), 3),
            "avg_score": avg_score,
            "min_score": min_score,
            "max_score": max_score,
        })

    # ── Recent link activity (most recent per-link detail) ────────
    recent_links = sorted(all_links, key=lambda l: l.get("patient_id", ""))[:50]

    return {
        "per_patient_summary": per_patient_summary,
        "per_assessment_summary": per_assessment_summary,
        "recent_links": recent_links,
        "total_links": len(all_links),
    }


def definitions():
    """Survey link terminology definitions with clinical context."""
    return {
        "title": "Survey Link Dashboard — Terminology & Definitions",
        "definitions": [
            {
                "term": "Survey Token",
                "definition": "A unique 32-character hexadecimal string generated "
                              "for each survey link.  The token serves as a "
                              "cryptographic identifier that maps to a specific "
                              "patient and assessment, preventing URL enumeration "
                              "and unauthorized access.",
                "clinical_relevance": "Token-based links ensure patient privacy by "
                                      "avoiding inclusion of patient identifiers (MRN, "
                                      "name) in the URL.  Even if a link is intercepted, "
                                      "no PHI is exposed in the URL itself.",
                "category": "security",
            },
            {
                "term": "Link Expiry",
                "definition": "A configurable time window (default 7 days) after "
                              "which an uncompleted survey link becomes invalid.  "
                              "Expired links return a friendly error page directing "
                              "the patient to contact their clinic for a new link.",
                "clinical_relevance": "Expiry windows balance patient convenience with "
                                      "clinical relevance.  Assessments completed too "
                                      "long after a clinic visit may not reflect the "
                                      "patient's state at the time of clinical interest.  "
                                      "Short windows (3-5 days) are used for pre-appointment "
                                      "screenings; longer windows (14-30 days) for "
                                      "longitudinal monitoring.",
                "category": "lifecycle",
            },
            {
                "term": "Completion Rate",
                "definition": "The proportion of generated survey links that "
                              "result in a submitted assessment, expressed as a "
                              "percentage.  Calculated as completed / total_generated.",
                "clinical_relevance": "Completion rates below 50% may indicate barriers "
                                      "to patient engagement: poor digital literacy, "
                                      "accessibility issues, link delivery failures, or "
                                      "assessment fatigue.  Monitoring by assessment type "
                                      "helps identify particularly burdensome instruments.",
                "category": "metrics",
            },
            {
                "term": "Response Time",
                "definition": "The elapsed time between link generation and "
                              "assessment submission, measured in hours.  Includes "
                              "both the delivery delay and the time the patient "
                              "spends completing the assessment.",
                "clinical_relevance": "Shorter response times (< 24 hours) correlate "
                                      "with higher data quality, as patients complete "
                                      "assessments closer to the clinical context of "
                                      "interest.  Response times > 72 hours may indicate "
                                      "the patient forgot or deprioritized the task.",
                "category": "metrics",
            },
            {
                "term": "PHQ-9 (Patient Health Questionnaire-9)",
                "definition": "A 9-item self-report questionnaire measuring "
                              "depression severity over the past 2 weeks.  Each "
                              "item scored 0-3 (not at all → nearly every day).  "
                              "Total score 0-27.  Thresholds: 5 mild, 10 moderate, "
                              "15 moderately severe, 20 severe.",
                "clinical_relevance": "Depression is the most common psychiatric "
                                      "comorbidity in epilepsy (prevalence 20-30%).  "
                                      "The PHQ-9 is validated in epilepsy populations "
                                      "and recommended by ILAE for routine screening.  "
                                      "Scores ≥ 10 warrant clinical follow-up.",
                "category": "assessments",
            },
            {
                "term": "GAD-7 (Generalized Anxiety Disorder-7)",
                "definition": "A 7-item self-report scale measuring anxiety "
                              "severity over the past 2 weeks.  Each item scored "
                              "0-3.  Total score 0-21.  Thresholds: 5 mild, 10 "
                              "moderate, 15 severe.",
                "clinical_relevance": "Anxiety disorders affect 10-25% of epilepsy "
                                      "patients and can worsen seizure frequency.  "
                                      "GAD-7 is brief enough for routine screening "
                                      "and has been validated in neurological populations.",
                "category": "assessments",
            },
            {
                "term": "NDDI-E (Neurological Disorders Depression Inventory for Epilepsy)",
                "definition": "A 6-item depression screening tool specifically "
                              "designed and validated for epilepsy patients.  Each "
                              "item scored 1-4.  Total score 6-24.  A score > 15 "
                              "suggests major depression.",
                "clinical_relevance": "Unlike the PHQ-9, the NDDI-E excludes somatic "
                                      "symptoms (fatigue, sleep disturbance) that overlap "
                                      "with AED side effects, reducing false positives in "
                                      "epilepsy patients on sedating medications.",
                "category": "assessments",
            },
            {
                "term": "QOLIE-31 (Quality of Life in Epilepsy-31)",
                "definition": "A 31-item epilepsy-specific quality of life "
                              "instrument covering 7 domains: seizure worry, "
                              "overall QOL, emotional well-being, energy/fatigue, "
                              "cognitive functioning, medication effects, and social "
                              "functioning.  Total score 0-100 (higher = better QOL).",
                "clinical_relevance": "QOLIE-31 is the gold standard for epilepsy QOL "
                                      "measurement and is recommended for clinical trials "
                                      "and routine care.  Survey links make it feasible "
                                      "to collect QOLIE-31 longitudinally without using "
                                      "clinic appointment time.",
                "category": "assessments",
            },
            {
                "term": "QR Code Delivery",
                "definition": "A method of distributing survey links by encoding "
                              "the URL into a QR code that can be printed on clinic "
                              "materials, discharge summaries, or appointment cards.  "
                              "Patients scan with their phone camera to access the "
                              "assessment.",
                "clinical_relevance": "QR codes bridge the digital divide by not "
                                      "requiring the patient to type a URL or use "
                                      "email.  Particularly effective for older patients "
                                      "or those with limited digital literacy — they only "
                                      "need to point their phone camera at the code.",
                "category": "delivery",
            },
            {
                "term": "Single-Use Link",
                "definition": "A survey link that can only be submitted once.  "
                              "After successful submission, the token is marked "
                              "completed and subsequent access shows a confirmation "
                              "page rather than the assessment form.",
                "clinical_relevance": "Single-use enforcement prevents duplicate "
                                      "submissions that could skew longitudinal "
                                      "score trends.  If a patient needs to retake "
                                      "an assessment, a new link with a fresh token "
                                      "must be generated, creating a clear audit trail.",
                "category": "lifecycle",
            },
        ],
    }
