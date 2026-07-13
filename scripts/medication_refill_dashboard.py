"""Medication Refill Dashboard — refill analytics from clinical.db.

Tracks medication refill patterns, drug distribution, pharmacy usage,
auto-refill rates, days supply analysis, and gap detection for AED drugs.

Sources:
- medication_refills table (patient_id, drug_name, refill_date, quantity,
  pharmacy, auto_refill, days_supply)
- 226 records, 30 patients, 8 AED drugs, 6 pharmacies
"""

import sqlite3
import json
import os
from datetime import datetime, timezone, timedelta

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql, params=(), default=0):
    try:
        cur.execute(sql, params)
        return cur.fetchone()[0]
    except Exception:
        return default


def _safe_rows(cur, sql, params=()):
    try:
        cur.execute(sql, params)
        return cur.fetchall()
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────
#  /api/medication-refills/overview
# ──────────────────────────────────────────────────────────────

def overview():
    """Aggregate refill health: total refills, drug and pharmacy distribution,
    monthly trends, auto-refill breakdown, KPIs."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    total_refills = _safe(cur, "SELECT COUNT(*) FROM medication_refills")
    total_patients = _safe(cur, "SELECT COUNT(DISTINCT patient_id) FROM medication_refills")
    total_drugs = _safe(cur, "SELECT COUNT(DISTINCT drug_name) FROM medication_refills")
    avg_quantity = _safe(cur, "SELECT ROUND(AVG(quantity), 1) FROM medication_refills")
    avg_days_supply = _safe(cur, "SELECT ROUND(AVG(days_supply), 1) FROM medication_refills")
    auto_refill_count = _safe(cur, "SELECT COUNT(*) FROM medication_refills WHERE auto_refill = 1")
    auto_refill_pct = round(auto_refill_count / total_refills * 100, 1) if total_refills else 0
    total_pharmacies = _safe(cur, "SELECT COUNT(DISTINCT pharmacy) FROM medication_refills")

    # Drug distribution
    drug_distribution = []
    for row in _safe_rows(cur,
            """SELECT drug_name, COUNT(*) as cnt
               FROM medication_refills GROUP BY drug_name ORDER BY cnt DESC"""):
        drug_distribution.append({
            "drug_name": row[0], "count": row[1],
            "pct": round(row[1] / total_refills * 100, 1) if total_refills else 0
        })

    # Pharmacy distribution
    pharmacy_distribution = []
    for row in _safe_rows(cur,
            """SELECT pharmacy, COUNT(*) as cnt
               FROM medication_refills GROUP BY pharmacy ORDER BY cnt DESC"""):
        pharmacy_distribution.append({
            "pharmacy": row[0], "count": row[1],
            "pct": round(row[1] / total_refills * 100, 1) if total_refills else 0
        })

    # Monthly trend
    monthly_trend = []
    for row in _safe_rows(cur,
            """SELECT strftime('%Y-%m', refill_date) as month, COUNT(*) as refills
               FROM medication_refills GROUP BY month ORDER BY month"""):
        monthly_trend.append({
            "month": row[0], "refills": row[1]
        })

    # Auto-refill breakdown
    auto_refill_breakdown = []
    manual_count = total_refills - auto_refill_count
    auto_refill_breakdown.append({
        "type": "Auto", "count": auto_refill_count,
        "pct": round(auto_refill_count / total_refills * 100, 1) if total_refills else 0
    })
    auto_refill_breakdown.append({
        "type": "Manual", "count": manual_count,
        "pct": round(manual_count / total_refills * 100, 1) if total_refills else 0
    })

    conn.close()

    return {
        "available": True,
        "kpis": {
            "total_refills": total_refills,
            "total_patients": total_patients,
            "total_drugs": total_drugs,
            "avg_quantity": avg_quantity,
            "avg_days_supply": avg_days_supply,
            "auto_refill_pct": auto_refill_pct,
            "total_pharmacies": total_pharmacies
        },
        "drug_distribution": drug_distribution,
        "pharmacy_distribution": pharmacy_distribution,
        "monthly_trend": monthly_trend,
        "auto_refill_breakdown": auto_refill_breakdown
    }


# ──────────────────────────────────────────────────────────────
#  /api/medication-refills/breakdown
# ──────────────────────────────────────────────────────────────

def breakdown():
    """Per-patient refill matrix, recent refills, drug details,
    and gap analysis for adherence monitoring."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # Per-patient summary (top 20 by refills)
    per_patient = []
    for row in _safe_rows(cur,
            """SELECT patient_id, COUNT(*) as refills,
                      COUNT(DISTINCT drug_name) as drugs,
                      ROUND(AVG(quantity), 1) as avg_quantity,
                      ROUND(AVG(days_supply), 1) as avg_days_supply
               FROM medication_refills GROUP BY patient_id
               ORDER BY refills DESC LIMIT 20"""):
        per_patient.append({
            "patient_id": row[0], "refills": row[1],
            "drugs": row[2], "avg_quantity": row[3],
            "avg_days_supply": row[4]
        })

    # Recent refills (last 20)
    recent_refills = []
    for row in _safe_rows(cur,
            """SELECT patient_id, drug_name, refill_date, quantity,
                      pharmacy, auto_refill, days_supply
               FROM medication_refills ORDER BY refill_date DESC LIMIT 20"""):
        recent_refills.append({
            "patient_id": row[0], "drug_name": row[1],
            "refill_date": row[2], "quantity": row[3],
            "pharmacy": row[4], "auto_refill": row[5],
            "days_supply": row[6]
        })

    # Drug details
    drug_details = []
    for row in _safe_rows(cur,
            """SELECT drug_name,
                      COUNT(*) as total_refills,
                      COUNT(DISTINCT patient_id) as unique_patients,
                      ROUND(AVG(quantity), 1) as avg_quantity,
                      ROUND(AVG(days_supply), 1) as avg_days_supply,
                      ROUND(SUM(CASE WHEN auto_refill = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as auto_refill_pct
               FROM medication_refills GROUP BY drug_name
               ORDER BY total_refills DESC"""):
        drug_details.append({
            "drug_name": row[0], "total_refills": row[1],
            "unique_patients": row[2], "avg_quantity": row[3],
            "avg_days_supply": row[4], "auto_refill_pct": row[5]
        })

    # Gap analysis — find patients with potential refill gaps
    # A gap occurs when the interval between consecutive refills of the
    # same drug for the same patient exceeds the days_supply of the earlier refill.
    gap_analysis = []
    refill_rows = _safe_rows(cur,
        """SELECT patient_id, drug_name, refill_date, days_supply
           FROM medication_refills
           ORDER BY patient_id, drug_name, refill_date""")

    prev = {}  # key: (patient_id, drug_name) -> (refill_date, days_supply)
    for row in refill_rows:
        patient_id, drug_name, refill_date, days_supply = row
        key = (patient_id, drug_name)
        if key in prev:
            prev_date_str, prev_days_supply = prev[key]
            try:
                prev_dt = datetime.strptime(prev_date_str, '%Y-%m-%d')
                curr_dt = datetime.strptime(refill_date, '%Y-%m-%d')
                gap_days = (curr_dt - prev_dt).days
                status = 'gap' if gap_days > prev_days_supply else 'on_time'
                if status == 'gap':
                    gap_analysis.append({
                        "patient_id": patient_id,
                        "drug_name": drug_name,
                        "last_refill": refill_date,
                        "previous_refill": prev_date_str,
                        "gap_days": gap_days,
                        "days_supply": prev_days_supply,
                        "status": status
                    })
            except (ValueError, TypeError):
                pass
        prev[key] = (refill_date, days_supply)

    # Sort gap analysis by gap_days descending, limit to top entries
    gap_analysis.sort(key=lambda x: x['gap_days'], reverse=True)

    conn.close()

    return {
        "available": True,
        "per_patient": per_patient,
        "recent_refills": recent_refills,
        "drug_details": drug_details,
        "gap_analysis": gap_analysis
    }


# ──────────────────────────────────────────────────────────────
#  /api/medication-refills/definitions
# ──────────────────────────────────────────────────────────────

def definitions():
    """Metric definitions for tooltip overlays."""
    return {
        "auto_refill_note": "Auto-refill indicates the prescription is set up for automatic "
                            "renewal at the pharmacy without requiring a new request each cycle. "
                            "Higher auto-refill rates correlate with better medication adherence.",
        "days_supply_note": "Days supply is the number of days a single refill quantity is expected "
                            "to last. Standard supplies are 30, 60, or 90 days. Shorter supplies may "
                            "indicate titration or cost management.",
        "gap_analysis_note": "Gap analysis identifies patients whose interval between consecutive "
                             "refills of the same drug exceeds the days_supply of the previous refill. "
                             "A gap suggests potential non-adherence and increased seizure risk.",
        "drug_classes": {
            "Oxcarbazepine": "Sodium channel blocker (triazine derivative). Treats focal seizures.",
            "Lamotrigine": "Sodium channel blocker / glutamate release inhibitor. Broad-spectrum AED.",
            "Lacosamide": "Sodium channel blocker (slow inactivation enhancer). Treats focal seizures.",
            "Topiramate": "Broad-spectrum AED (multiple mechanisms: sodium channels, GABA enhancement, "
                          "glutamate antagonism, carbonic anhydrase inhibition).",
            "Valproate": "Broad-spectrum AED (GABA potentiation, sodium/calcium channel modulation). "
                         "First-line for generalized epilepsy.",
            "Carbamazepine": "Sodium channel blocker. Classic first-line AED for focal seizures.",
            "Levetiracetam": "SV2A-binding AED. Broad-spectrum with favorable side-effect profile. "
                             "Most prescribed AED worldwide.",
            "Clobazam": "Benzodiazepine (GABA-A modulator). Adjunctive therapy for Lennox-Gastaut "
                        "syndrome and refractory seizures."
        },
        "pharmacy_note": "Pharmacy distribution tracks which pharmacies fill the most AED prescriptions. "
                         "Consolidating refills at a single pharmacy improves drug interaction screening "
                         "and adherence tracking."
    }


if __name__ == '__main__':
    import pprint
    print("=== Overview ===")
    pprint.pprint(overview())
    print("\n=== Breakdown ===")
    pprint.pprint(breakdown())
    print("\n=== Definitions ===")
    pprint.pprint(definitions())
