"""
Clinical Dietitian / Nutritionist Module
==========================================
Nutrition assessment for epilepsy patients from clinical.db — medications,
patients, assessments (BARTHEL), and seizure_diary tables.

All nutrient-depletion data and food-interaction data are from published
clinical evidence in AED pharmacology (Epilepsia, Neurology, J Clin Pharmacol,
Am J Clin Nutr).  Key references:
  - Johannessen Landmark CJ (2008). Antiepileptic drugs in non-epilepsy
    disorders. CNS Drugs.
  - Fong CY, et al. (2012). Effects of AEDs on nutrient status. CNS Drugs.
  - Sato Y, et al. (2001). Vitamin D deficiency and AEDs. Epilepsia.
  - Pack AM (2004). Bone disease with AEDs. Neurology.

Endpoints:
  /api/dietitian                             — full dashboard (all 4 sub-analyses)
  /api/dietitian/ketogenic-diet              — Ketogenic diet eligibility/monitoring
  /api/dietitian/malnutrition-screening      — MNA/MUST-style malnutrition risk screening
  /api/dietitian/nutrient-analysis           — AED-specific nutrient/vitamin depletion analysis
  /api/dietitian/medication-nutrition        — Medication-nutrition interaction counseling

All data from REAL patients, medications, assessments, and seizure_diary
in data/clinical.db.
"""

import sqlite3
import os
import json
from collections import defaultdict

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "clinical.db",
)

# ─── AED nutrient depletion table (published clinical evidence) ─────────
# Each entry: drug_name_lower → list of {nutrient, mechanism, recommendation}
AED_NUTRIENT_DEPLETIONS = {
    "valproic acid": [
        {
            "nutrient": "L-Carnitine",
            "mechanism": "VPA inhibits carnitine transporter and increases renal excretion; risk of hyperammonemia",
            "recommendation": "Supplement L-carnitine 10-20 mg/kg/day; monitor ammonia levels",
            "severity": "high",
        },
        {
            "nutrient": "Folate (Vitamin B9)",
            "mechanism": "VPA inhibits folate metabolism; risk of megaloblastic anemia and teratogenicity",
            "recommendation": "Supplement folic acid 1 mg/day (5 mg/day if childbearing age)",
            "severity": "high",
        },
    ],
    "carbamazepine": [
        {
            "nutrient": "Vitamin D",
            "mechanism": "CYP450 enzyme induction accelerates vitamin D catabolism → reduced 25(OH)D levels",
            "recommendation": "Supplement vitamin D 1000-2000 IU/day; monitor 25(OH)D annually",
            "severity": "high",
        },
        {
            "nutrient": "Folate (Vitamin B9)",
            "mechanism": "Enzyme induction increases folate catabolism; impairs absorption",
            "recommendation": "Supplement folic acid 1 mg/day",
            "severity": "moderate",
        },
        {
            "nutrient": "Vitamin B12",
            "mechanism": "Impaired B12 absorption with long-term use",
            "recommendation": "Monitor B12 levels annually; supplement if <300 pg/mL",
            "severity": "moderate",
        },
    ],
    "phenytoin": [
        {
            "nutrient": "Vitamin D",
            "mechanism": "Potent CYP450 inducer → accelerated vitamin D degradation → osteomalacia risk",
            "recommendation": "Supplement vitamin D 2000 IU/day; DEXA scan if >2 years on therapy",
            "severity": "high",
        },
        {
            "nutrient": "Folate (Vitamin B9)",
            "mechanism": "Impairs folate absorption and increases catabolism; risk of megaloblastic anemia",
            "recommendation": "Supplement folic acid 1 mg/day (caution: high-dose folate may reduce phenytoin levels)",
            "severity": "high",
        },
        {
            "nutrient": "Calcium",
            "mechanism": "Secondary to vitamin D depletion; impaired calcium absorption",
            "recommendation": "Supplement calcium 1000-1200 mg/day with vitamin D",
            "severity": "moderate",
        },
    ],
    "phenobarbital": [
        {
            "nutrient": "Vitamin D",
            "mechanism": "CYP450 enzyme induction → accelerated 25(OH)D catabolism",
            "recommendation": "Supplement vitamin D 1000-2000 IU/day; bone density monitoring",
            "severity": "high",
        },
        {
            "nutrient": "Folate (Vitamin B9)",
            "mechanism": "Impaired folate absorption and increased catabolism",
            "recommendation": "Supplement folic acid 1 mg/day",
            "severity": "moderate",
        },
        {
            "nutrient": "Calcium",
            "mechanism": "Secondary to vitamin D depletion → reduced calcium absorption",
            "recommendation": "Supplement calcium 1000-1200 mg/day",
            "severity": "moderate",
        },
    ],
    "levetiracetam": [
        {
            "nutrient": "Vitamin B6 (Pyridoxine)",
            "mechanism": "Levetiracetam may interfere with pyridoxine metabolism; behavioral side effects may respond to B6",
            "recommendation": "Consider pyridoxine 50-100 mg/day if behavioral side effects present",
            "severity": "low",
        },
    ],
    "topiramate": [
        {
            "nutrient": "Bicarbonate",
            "mechanism": "Carbonic anhydrase inhibition → metabolic acidosis (reduced serum bicarbonate)",
            "recommendation": "Monitor serum bicarbonate; supplement with oral bicarbonate if <18 mEq/L; ensure adequate hydration",
            "severity": "high",
        },
    ],
    "lamotrigine": [
        {
            "nutrient": "Folate (Vitamin B9)",
            "mechanism": "Weak dihydrofolate reductase inhibitor; mild folate reduction",
            "recommendation": "Supplement folic acid 0.4-1 mg/day (especially if childbearing age)",
            "severity": "low",
        },
    ],
    "oxcarbazepine": [
        {
            "nutrient": "Vitamin D",
            "mechanism": "Mild CYP450 induction → vitamin D catabolism (less than carbamazepine)",
            "recommendation": "Monitor 25(OH)D; supplement 1000 IU/day if low",
            "severity": "moderate",
        },
        {
            "nutrient": "Sodium",
            "mechanism": "Risk of hyponatremia (SIADH-like effect); not dietary depletion per se",
            "recommendation": "Monitor serum sodium; avoid excessive free water intake",
            "severity": "moderate",
        },
    ],
    "zonisamide": [
        {
            "nutrient": "Bicarbonate",
            "mechanism": "Carbonic anhydrase inhibition → metabolic acidosis (similar to topiramate)",
            "recommendation": "Monitor serum bicarbonate; supplement if <18 mEq/L",
            "severity": "moderate",
        },
    ],
}

# ─── AED-food interaction table (published clinical evidence) ───────────
AED_FOOD_INTERACTIONS = {
    "carbamazepine": [
        {
            "interaction": "Grapefruit juice inhibits CYP3A4 → increased carbamazepine levels → toxicity risk",
            "counseling": "AVOID grapefruit and grapefruit juice entirely while on carbamazepine",
            "severity": "high",
        },
        {
            "interaction": "High-fat meals may increase absorption rate",
            "counseling": "Take consistently with or without food; report dizziness/nausea if taken with large fatty meals",
            "severity": "low",
        },
    ],
    "phenytoin": [
        {
            "interaction": "Enteral (tube) feeding reduces phenytoin absorption by 50-75% (binds to casein/calcium in formula)",
            "counseling": "HOLD enteral feeds 2 hours before AND 2 hours after phenytoin dose; flush tube with water",
            "severity": "high",
        },
        {
            "interaction": "High-protein diets may alter phenytoin binding and free fraction",
            "counseling": "Maintain consistent protein intake; avoid drastic dietary changes",
            "severity": "moderate",
        },
        {
            "interaction": "Calcium-rich foods/supplements if taken simultaneously may reduce absorption",
            "counseling": "Separate calcium supplements by 2 hours from phenytoin dose",
            "severity": "moderate",
        },
    ],
    "valproic acid": [
        {
            "interaction": "GI side effects (nausea, cramping) common on empty stomach",
            "counseling": "TAKE WITH FOOD to reduce GI side effects; use enteric-coated formulation if available",
            "severity": "moderate",
        },
        {
            "interaction": "Carbonated beverages may dissolve enteric coating prematurely → increased GI upset",
            "counseling": "Avoid taking with carbonated drinks",
            "severity": "low",
        },
    ],
    "levetiracetam": [
        {
            "interaction": "No significant food interactions established",
            "counseling": "May be taken with or without food; maintain adequate hydration",
            "severity": "none",
        },
    ],
    "topiramate": [
        {
            "interaction": "Carbonic anhydrase inhibition increases kidney stone risk (calcium phosphate)",
            "counseling": "CRITICAL: Maintain adequate hydration (≥2-3 L/day); avoid high-oxalate foods (spinach, rhubarb, nuts); increase citrate-rich foods (lemon water)",
            "severity": "high",
        },
        {
            "interaction": "Appetite suppression and weight loss are common dose-dependent effects",
            "counseling": "Monitor weight weekly; ensure adequate caloric intake; small frequent meals if appetite poor",
            "severity": "moderate",
        },
    ],
    "lamotrigine": [
        {
            "interaction": "No significant food interactions established",
            "counseling": "May be taken with or without food",
            "severity": "none",
        },
    ],
    "phenobarbital": [
        {
            "interaction": "Alcohol potentiates CNS depression synergistically",
            "counseling": "AVOID alcohol completely; even small amounts may cause excessive sedation and respiratory depression",
            "severity": "high",
        },
        {
            "interaction": "Caffeine may reduce sedative effect but does not affect AED efficacy",
            "counseling": "Moderate caffeine is acceptable; avoid excess which may lower seizure threshold",
            "severity": "low",
        },
    ],
    "oxcarbazepine": [
        {
            "interaction": "Grapefruit juice may mildly increase levels (less than carbamazepine)",
            "counseling": "Limit grapefruit juice; monitor for dizziness if consumed",
            "severity": "moderate",
        },
    ],
    "zonisamide": [
        {
            "interaction": "Kidney stone risk similar to topiramate (carbonic anhydrase inhibition)",
            "counseling": "Maintain adequate hydration (≥2 L/day); avoid high-oxalate foods",
            "severity": "moderate",
        },
    ],
}

# ─── Medications that cause appetite suppression (malnutrition risk) ────
APPETITE_SUPPRESSING_AEDS = {
    "topiramate": "Significant appetite suppression and weight loss (5-10% body weight common)",
    "zonisamide": "Moderate appetite suppression and weight loss",
    "felbamate": "Appetite loss and nausea common",
}

# ─── Ketogenic diet types ───────────────────────────────────────────────
KETO_DIET_TYPES = [
    {
        "name": "Classic Ketogenic Diet (4:1)",
        "ratio": "4:1 fat to (protein+carb)",
        "carb_pct": "~3-4%",
        "indication": "Drug-resistant epilepsy; highest efficacy; requires dietitian supervision",
        "suitability": "Pediatric primary; adult feasible with motivation",
    },
    {
        "name": "Modified Atkins Diet (MAD)",
        "ratio": "~1-2:1",
        "carb_limit": "10-20 g/day",
        "indication": "Adult-friendly alternative; easier compliance; nearly comparable efficacy",
        "suitability": "Adults and adolescents; outpatient initiation",
    },
    {
        "name": "Medium-Chain Triglyceride (MCT) Diet",
        "ratio": "~3:1 with MCT oil",
        "carb_pct": "~15-19%",
        "indication": "Allows more carbs/protein than classic; MCT oil provides additional ketones",
        "suitability": "Patients who cannot tolerate strict fat ratios; GI tolerance may limit",
    },
    {
        "name": "Low Glycemic Index Treatment (LGIT)",
        "ratio": "~1:1",
        "carb_limit": "40-60 g/day (GI <50 only)",
        "indication": "Least restrictive; suitable for patients unwilling/unable to follow stricter protocols",
        "suitability": "Adults; outpatient; minimal supervision needed",
    },
]

# ─── AEDs with known keto diet contraindications/interactions ──────────
KETO_AED_INTERACTIONS = {
    "valproic acid": {
        "concern": "VPA + ketogenic diet increases risk of carnitine depletion and hepatotoxicity; monitor LFTs closely",
        "severity": "high",
        "action": "Mandatory L-carnitine supplementation; weekly LFTs during initiation",
    },
    "topiramate": {
        "concern": "Both topiramate and keto diet increase kidney stone risk (additive carbonic anhydrase inhibition + high fat)",
        "severity": "high",
        "action": "Aggressive hydration (≥3 L/day); urine alkalinization; renal ultrasound at baseline",
    },
    "zonisamide": {
        "concern": "Similar kidney stone risk as topiramate when combined with keto diet",
        "severity": "moderate",
        "action": "Increased hydration; monitor urine calcium/citrate ratio",
    },
    "phenobarbital": {
        "concern": "Enzyme induction may alter ketone metabolism; sedation may mask keto-flu symptoms",
        "severity": "low",
        "action": "Monitor ketone levels more frequently during initiation",
    },
}


def _conn():
    return sqlite3.connect(DB_PATH)


def _rows_as_dicts(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _parse_medications(med_rows):
    """Parse medications rows → list of dicts with drug_name, dose_mg, frequency, aed, patient_id."""
    meds = []
    for mr in med_rows:
        fields = {}
        if mr.get("fields_json"):
            try:
                fields = json.loads(mr["fields_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        meds.append({
            "patient_id": mr["patient_id"],
            "drug_name": fields.get("drug_name", ""),
            "dose_mg": fields.get("dose_mg", ""),
            "frequency": fields.get("frequency", ""),
            "aed": fields.get("aed", ""),
        })
    return meds


def _get_medications(c, patient_id=None):
    """Fetch and parse medications from DB."""
    where = "WHERE patient_id = ?" if patient_id else ""
    params = (patient_id,) if patient_id else ()
    cur = c.execute(f"SELECT patient_id, fields_json FROM medications {where}", params)
    return _parse_medications(_rows_as_dicts(cur))


def _get_patients(c, patient_id=None):
    """Fetch patients from DB."""
    where = "WHERE patient_id = ?" if patient_id else ""
    params = (patient_id,) if patient_id else ()
    cur = c.execute(f"SELECT patient_id, name, age, gender, disease FROM patients {where}", params)
    return _rows_as_dicts(cur)


def _match_drug(drug_name, lookup_dict):
    """Match a drug name (case-insensitive) against a lookup dictionary."""
    if not drug_name:
        return None
    lower = drug_name.strip().lower()
    for key in lookup_dict:
        if key in lower:
            return key
    return None


# ─── 1. Ketogenic Diet Eligibility ─────────────────────────────────────

def ketogenic_diet_eligibility(patient_id=None):
    """
    Ketogenic diet eligibility and monitoring assessment.

    Evaluates:
    - Seizure frequency from seizure_diary (high frequency = candidate)
    - Current AEDs and keto-specific drug interactions
    - Age considerations (pediatric vs adult protocol selection)
    - Produces eligibility score, contraindications, recommended diet type
    """
    c = _conn()
    patients = _get_patients(c, patient_id)
    meds_all = _get_medications(c, patient_id)

    # Seizure frequency per patient
    where_sz = "WHERE patient_id = ?" if patient_id else ""
    params = (patient_id,) if patient_id else ()
    cur = c.execute(f"SELECT patient_id, event_date, severity FROM seizure_diary {where_sz}", params)
    seizures = _rows_as_dicts(cur)
    sz_counts = defaultdict(int)
    sz_severe = defaultdict(int)
    for s in seizures:
        sz_counts[s["patient_id"]] += 1
        if s.get("severity") in ("Severe", "severe", "High", "high"):
            sz_severe[s["patient_id"]] += 1

    # Group meds by patient
    meds_by_patient = defaultdict(list)
    for m in meds_all:
        meds_by_patient[m["patient_id"]].append(m)

    # Build per-patient eligibility
    profiles = []
    for p in patients:
        pid = p["patient_id"]
        age = p.get("age") or 0
        sz_count = sz_counts.get(pid, 0)
        severe_count = sz_severe.get(pid, 0)
        patient_meds = meds_by_patient.get(pid, [])

        # --- Seizure frequency score (0-40 points) ---
        # Higher seizure count → higher eligibility
        if sz_count >= 4:
            sz_score = 40
            sz_rationale = f"High seizure frequency ({sz_count} events) — strong candidate for dietary therapy"
        elif sz_count >= 2:
            sz_score = 25
            sz_rationale = f"Moderate seizure frequency ({sz_count} events) — consider dietary therapy if AEDs suboptimal"
        elif sz_count >= 1:
            sz_score = 10
            sz_rationale = f"Low seizure frequency ({sz_count} event) — dietary therapy as adjunct only"
        else:
            sz_score = 0
            sz_rationale = "No seizure events recorded — dietary therapy not indicated for seizure control"

        # --- Drug-resistance proxy (0-30 points) ---
        # ≥2 AEDs and still seizing = drug-resistant (ILAE definition)
        aed_count = sum(1 for m in patient_meds if m.get("aed") in ("Yes", "yes", "1", True, "true"))
        if aed_count == 0:
            aed_count = len(patient_meds)  # fallback: count all meds

        if aed_count >= 3 and sz_count >= 2:
            resist_score = 30
            resist_rationale = f"On {aed_count} AEDs with ongoing seizures — meets drug-resistant criteria (ILAE)"
        elif aed_count >= 2 and sz_count >= 1:
            resist_score = 20
            resist_rationale = f"On {aed_count} AEDs with seizures — likely drug-resistant"
        elif aed_count >= 2:
            resist_score = 10
            resist_rationale = f"On {aed_count} AEDs — polytherapy but seizure-free"
        else:
            resist_score = 5
            resist_rationale = f"On {aed_count} AED(s) — monotherapy"

        # --- Age-based diet recommendation (0-15 points) ---
        if age <= 12:
            age_score = 15
            age_rationale = "Pediatric — classic 4:1 ketogenic diet is first-line; strong evidence base"
            recommended_diet = "Classic Ketogenic Diet (4:1)"
        elif age <= 18:
            age_score = 12
            age_rationale = "Adolescent — Modified Atkins Diet (MAD) preferred for compliance; classic keto also feasible"
            recommended_diet = "Modified Atkins Diet (MAD)"
        elif age <= 65:
            age_score = 10
            age_rationale = "Adult — MAD or LGIT recommended for adherence; classic keto if highly motivated"
            recommended_diet = "Modified Atkins Diet (MAD)"
        else:
            age_score = 5
            age_rationale = "Elderly — LGIT safest option; monitor renal function and bone density"
            recommended_diet = "Low Glycemic Index Treatment (LGIT)"

        # --- Severity bonus (0-15 points) ---
        if severe_count >= 2:
            sev_score = 15
            sev_rationale = f"{severe_count} severe seizures — urgent need for additional therapy"
        elif severe_count >= 1:
            sev_score = 8
            sev_rationale = f"{severe_count} severe seizure — consider add-on dietary therapy"
        else:
            sev_score = 0
            sev_rationale = "No severe seizures recorded"

        # --- Contraindications from current medications ---
        contraindications = []
        for m in patient_meds:
            matched = _match_drug(m["drug_name"], KETO_AED_INTERACTIONS)
            if matched:
                info = KETO_AED_INTERACTIONS[matched]
                contraindications.append({
                    "medication": m["drug_name"],
                    "concern": info["concern"],
                    "severity": info["severity"],
                    "action": info["action"],
                })

        # --- Composite eligibility score (0-100) ---
        total_score = sz_score + resist_score + age_score + sev_score
        if total_score >= 70:
            eligibility = "Strong candidate"
        elif total_score >= 45:
            eligibility = "Moderate candidate"
        elif total_score >= 20:
            eligibility = "Weak candidate — consider only if other options exhausted"
        else:
            eligibility = "Not currently indicated"

        profiles.append({
            "patient_id": pid,
            "name": p.get("name", ""),
            "age": age,
            "eligibility_score": total_score,
            "eligibility_category": eligibility,
            "recommended_diet_type": recommended_diet,
            "score_breakdown": {
                "seizure_frequency": {"score": sz_score, "max": 40, "rationale": sz_rationale},
                "drug_resistance": {"score": resist_score, "max": 30, "rationale": resist_rationale},
                "age_factor": {"score": age_score, "max": 15, "rationale": age_rationale},
                "severity_factor": {"score": sev_score, "max": 15, "rationale": sev_rationale},
            },
            "contraindications": contraindications,
            "aed_count": aed_count,
            "seizure_count": sz_count,
        })

    profiles.sort(key=lambda p: p["eligibility_score"], reverse=True)

    # Cohort summary
    scores = [p["eligibility_score"] for p in profiles]
    cat_dist = defaultdict(int)
    for p in profiles:
        cat_dist[p["eligibility_category"]] += 1

    c.close()
    return {
        "assessment": "Ketogenic Diet Eligibility (composite: seizure frequency + drug resistance + age + severity)",
        "unique_patients": len(profiles),
        "cohort_summary": {
            "mean_eligibility_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "category_distribution": [
                {"category": k, "count": v, "pct": round(100 * v / len(profiles), 1)}
                for k, v in sorted(cat_dist.items())
            ],
        },
        "diet_types_reference": KETO_DIET_TYPES,
        "patient_profiles": profiles,
    }


# ─── 2. Malnutrition Screening (MNA/MUST-style) ───────────────────────

def malnutrition_screening(patient_id=None):
    """
    Malnutrition risk screening using available clinical data.

    Risk factors assessed:
    - Age ≥65 (elderly at higher risk per MNA/MUST criteria)
    - Appetite-suppressing AEDs (topiramate, zonisamide)
    - Low Barthel scores (ADL impairment → feeding difficulty)
    - High medication burden (polypharmacy → drug-nutrient interactions)
    """
    c = _conn()
    patients = _get_patients(c, patient_id)
    meds_all = _get_medications(c, patient_id)

    # Barthel scores for ADL/feeding assessment
    where_b = "WHERE instrument='BARTHEL'" + (" AND patient_id = ?" if patient_id else "")
    params = (patient_id,) if patient_id else ()
    cur = c.execute(
        f"SELECT patient_id, answers_json, score FROM assessments {where_b} ORDER BY created_at DESC",
        params,
    )
    barthel_rows = _rows_as_dicts(cur)
    barthel_latest = {}
    for r in barthel_rows:
        pid = r["patient_id"]
        if pid not in barthel_latest:
            barthel_latest[pid] = r

    # Group meds by patient
    meds_by_patient = defaultdict(list)
    for m in meds_all:
        meds_by_patient[m["patient_id"]].append(m)

    profiles = []
    for p in patients:
        pid = p["patient_id"]
        age = p.get("age") or 0
        patient_meds = meds_by_patient.get(pid, [])
        risk_points = 0
        risk_factors = []

        # --- Age risk ---
        if age >= 80:
            risk_points += 3
            risk_factors.append({
                "factor": "Advanced age (≥80)",
                "points": 3,
                "detail": f"Age {age} — very high malnutrition risk per MNA criteria",
            })
        elif age >= 65:
            risk_points += 2
            risk_factors.append({
                "factor": "Elderly (≥65)",
                "points": 2,
                "detail": f"Age {age} — elevated malnutrition risk; recommend MNA-SF screening",
            })
        elif age <= 12:
            risk_points += 1
            risk_factors.append({
                "factor": "Pediatric (<12)",
                "points": 1,
                "detail": f"Age {age} — growth monitoring essential; keto diet may restrict growth",
            })

        # --- Appetite-suppressing AEDs ---
        for m in patient_meds:
            matched = _match_drug(m["drug_name"], APPETITE_SUPPRESSING_AEDS)
            if matched:
                risk_points += 2
                risk_factors.append({
                    "factor": f"Appetite-suppressing AED: {m['drug_name']}",
                    "points": 2,
                    "detail": APPETITE_SUPPRESSING_AEDS[matched],
                })

        # --- Barthel feeding and total score ---
        barthel = barthel_latest.get(pid)
        if barthel:
            total_barthel = barthel.get("score") or 0
            answers = json.loads(barthel["answers_json"]) if barthel.get("answers_json") else {}
            feeding_score = float(answers.get("item1", 0) or 0)

            if feeding_score < 10:
                risk_points += 3
                risk_factors.append({
                    "factor": "Impaired feeding (Barthel feeding <10)",
                    "points": 3,
                    "detail": f"Feeding score {feeding_score}/10 — needs assistance with eating; high aspiration/malnutrition risk",
                })
            if total_barthel <= 60:
                risk_points += 2
                risk_factors.append({
                    "factor": "Severe ADL dependence (Barthel ≤60)",
                    "points": 2,
                    "detail": f"Barthel total {total_barthel}/100 — dependent for meal preparation and self-feeding",
                })
            elif total_barthel <= 90:
                risk_points += 1
                risk_factors.append({
                    "factor": "Moderate ADL dependence (Barthel 61-90)",
                    "points": 1,
                    "detail": f"Barthel total {total_barthel}/100 — may need help with meal preparation",
                })

        # --- Polypharmacy (≥3 medications) ---
        if len(patient_meds) >= 4:
            risk_points += 2
            risk_factors.append({
                "factor": "High polypharmacy (≥4 medications)",
                "points": 2,
                "detail": f"{len(patient_meds)} medications — increased drug-nutrient interaction risk and GI side effects",
            })
        elif len(patient_meds) >= 3:
            risk_points += 1
            risk_factors.append({
                "factor": "Polypharmacy (≥3 medications)",
                "points": 1,
                "detail": f"{len(patient_meds)} medications — moderate drug-nutrient interaction risk",
            })

        # --- Risk classification ---
        if risk_points >= 6:
            risk_level = "High"
            action = "Urgent dietitian referral; detailed nutritional assessment; consider oral nutritional supplements"
        elif risk_points >= 3:
            risk_level = "Medium"
            action = "Dietitian review within 2 weeks; dietary counseling; monitor weight monthly"
        else:
            risk_level = "Low"
            action = "Routine dietary advice; annual nutritional review"

        profiles.append({
            "patient_id": pid,
            "name": p.get("name", ""),
            "age": age,
            "risk_score": risk_points,
            "risk_level": risk_level,
            "recommended_action": action,
            "risk_factors": risk_factors,
            "medication_count": len(patient_meds),
            "barthel_total": barthel.get("score") if barthel else None,
        })

    profiles.sort(key=lambda p: p["risk_score"], reverse=True)

    # Cohort summary
    risk_dist = defaultdict(int)
    for p in profiles:
        risk_dist[p["risk_level"]] += 1

    c.close()
    return {
        "assessment": "Malnutrition Risk Screening (MNA/MUST-style, adapted for epilepsy population)",
        "scoring": "Points: age (0-3) + appetite-suppressing AEDs (0-2 each) + Barthel feeding/ADL (0-5) + polypharmacy (0-2). High ≥6, Medium 3-5, Low 0-2.",
        "unique_patients": len(profiles),
        "cohort_summary": {
            "risk_distribution": [
                {"level": k, "count": v, "pct": round(100 * v / len(profiles), 1)}
                for k, v in sorted(risk_dist.items())
            ],
            "mean_risk_score": round(sum(p["risk_score"] for p in profiles) / len(profiles), 1) if profiles else 0,
            "patients_needing_referral": sum(1 for p in profiles if p["risk_level"] == "High"),
        },
        "patient_profiles": profiles,
    }


# ─── 3. Nutrient/Vitamin Depletion Analysis ────────────────────────────

def nutrient_analysis(patient_id=None):
    """
    AED-specific nutrient depletion analysis.

    Cross-references each patient's actual medications against the
    published AED nutrient depletion table to produce per-patient
    supplement recommendations.

    Evidence sources:
    - Fong CY et al. (2012) CNS Drugs: AED effects on nutritional status
    - Sato Y et al. (2001) Epilepsia: Vitamin D and AEDs
    - Pack AM (2004) Neurology: Bone health and AEDs
    - Johannessen Landmark CJ (2008): AED drug interactions
    """
    c = _conn()
    patients = _get_patients(c, patient_id)
    meds_all = _get_medications(c, patient_id)

    # Group meds by patient
    meds_by_patient = defaultdict(list)
    for m in meds_all:
        meds_by_patient[m["patient_id"]].append(m)

    profiles = []
    nutrient_cohort_counts = defaultdict(int)  # how many patients affected per nutrient

    for p in patients:
        pid = p["patient_id"]
        patient_meds = meds_by_patient.get(pid, [])

        depletions = []
        supplements_needed = {}
        affected_nutrients = set()

        for m in patient_meds:
            matched = _match_drug(m["drug_name"], AED_NUTRIENT_DEPLETIONS)
            if matched:
                for depletion in AED_NUTRIENT_DEPLETIONS[matched]:
                    nutrient = depletion["nutrient"]
                    affected_nutrients.add(nutrient)
                    depletions.append({
                        "medication": m["drug_name"],
                        "dose_mg": m.get("dose_mg", ""),
                        "nutrient_depleted": nutrient,
                        "mechanism": depletion["mechanism"],
                        "severity": depletion["severity"],
                    })
                    # Track unique supplement recommendations (keep highest severity)
                    if nutrient not in supplements_needed or _sev_rank(depletion["severity"]) > _sev_rank(supplements_needed[nutrient]["severity"]):
                        supplements_needed[nutrient] = {
                            "nutrient": nutrient,
                            "recommendation": depletion["recommendation"],
                            "severity": depletion["severity"],
                            "caused_by": [m["drug_name"]],
                        }
                    else:
                        if m["drug_name"] not in supplements_needed[nutrient]["caused_by"]:
                            supplements_needed[nutrient]["caused_by"].append(m["drug_name"])

        for nutrient in affected_nutrients:
            nutrient_cohort_counts[nutrient] += 1

        # Sort supplements by severity
        supplement_list = sorted(
            supplements_needed.values(),
            key=lambda s: _sev_rank(s["severity"]),
            reverse=True,
        )

        profiles.append({
            "patient_id": pid,
            "name": p.get("name", ""),
            "medications": [{"drug_name": m["drug_name"], "dose_mg": m.get("dose_mg", "")} for m in patient_meds],
            "depletions_identified": len(depletions),
            "depletion_details": depletions,
            "supplement_recommendations": supplement_list,
            "high_severity_count": sum(1 for d in depletions if d["severity"] == "high"),
        })

    profiles.sort(key=lambda p: p["depletions_identified"], reverse=True)

    # Cohort-level nutrient gap analysis
    nutrient_summary = [
        {
            "nutrient": k,
            "patients_affected": v,
            "pct_of_cohort": round(100 * v / len(profiles), 1) if profiles else 0,
        }
        for k, v in sorted(nutrient_cohort_counts.items(), key=lambda x: -x[1])
    ]

    c.close()
    return {
        "assessment": "AED-Specific Nutrient Depletion Analysis",
        "evidence_basis": "Published AED pharmacology: Fong CY et al. 2012 (CNS Drugs), Sato Y et al. 2001 (Epilepsia), Pack AM 2004 (Neurology)",
        "unique_patients": len(profiles),
        "cohort_summary": {
            "patients_with_depletions": sum(1 for p in profiles if p["depletions_identified"] > 0),
            "patients_with_high_severity": sum(1 for p in profiles if p["high_severity_count"] > 0),
            "nutrient_gap_frequency": nutrient_summary,
        },
        "patient_profiles": profiles,
    }


def _sev_rank(severity):
    """Numeric rank for severity sorting."""
    return {"high": 3, "moderate": 2, "low": 1, "none": 0}.get(severity, 0)


# ─── 4. Medication-Nutrition Interaction Analysis ──────────────────────

def medication_nutrition_interaction(patient_id=None):
    """
    AED-food/nutrition interaction analysis.

    Cross-references each patient's actual medications against the
    published AED-food interaction table to produce per-patient
    dietary counseling points.

    Evidence sources:
    - Epilepsia clinical practice guidelines
    - Drug interaction databases (Lexicomp, Micromedex)
    - FDA prescribing information for each AED
    """
    c = _conn()
    patients = _get_patients(c, patient_id)
    meds_all = _get_medications(c, patient_id)

    # Group meds by patient
    meds_by_patient = defaultdict(list)
    for m in meds_all:
        meds_by_patient[m["patient_id"]].append(m)

    profiles = []
    interaction_cohort_counts = defaultdict(int)

    for p in patients:
        pid = p["patient_id"]
        patient_meds = meds_by_patient.get(pid, [])

        interactions = []
        counseling_points = []
        seen_counseling = set()

        for m in patient_meds:
            matched = _match_drug(m["drug_name"], AED_FOOD_INTERACTIONS)
            if matched:
                for ix in AED_FOOD_INTERACTIONS[matched]:
                    interactions.append({
                        "medication": m["drug_name"],
                        "dose_mg": m.get("dose_mg", ""),
                        "interaction": ix["interaction"],
                        "severity": ix["severity"],
                    })
                    # Unique counseling points
                    counsel_key = ix["counseling"]
                    if counsel_key not in seen_counseling:
                        seen_counseling.add(counsel_key)
                        counseling_points.append({
                            "medication": m["drug_name"],
                            "counseling": ix["counseling"],
                            "severity": ix["severity"],
                        })

        # Sort counseling by severity
        counseling_points.sort(key=lambda cp: _sev_rank(cp["severity"]), reverse=True)

        # Count high-severity interactions per drug name for cohort
        for ix in interactions:
            if ix["severity"] in ("high", "moderate"):
                interaction_cohort_counts[ix["medication"]] += 1

        profiles.append({
            "patient_id": pid,
            "name": p.get("name", ""),
            "medications": [{"drug_name": m["drug_name"], "dose_mg": m.get("dose_mg", "")} for m in patient_meds],
            "interactions_found": len(interactions),
            "interaction_details": interactions,
            "dietary_counseling_points": counseling_points,
            "high_severity_count": sum(1 for ix in interactions if ix["severity"] == "high"),
        })

    profiles.sort(key=lambda p: p["high_severity_count"], reverse=True)

    # Cohort summary
    drug_interaction_freq = [
        {"medication": k, "interaction_events": v}
        for k, v in sorted(interaction_cohort_counts.items(), key=lambda x: -x[1])
    ]

    c.close()
    return {
        "assessment": "Medication-Nutrition Interaction Analysis",
        "evidence_basis": "AED prescribing information (FDA), Lexicomp, Micromedex drug interaction databases",
        "unique_patients": len(profiles),
        "cohort_summary": {
            "patients_with_interactions": sum(1 for p in profiles if p["interactions_found"] > 0),
            "patients_with_high_severity": sum(1 for p in profiles if p["high_severity_count"] > 0),
            "most_common_interactions_by_drug": drug_interaction_freq,
        },
        "patient_profiles": profiles,
    }


# ─── Full Dashboard ────────────────────────────────────────────────────

def full_dashboard(patient_id=None):
    """Combined Clinical Dietitian dashboard — all 4 modules for the patient(s)."""
    return {
        "role": "Clinical Dietitian / Nutritionist",
        "description": (
            "Nutritional assessment for epilepsy patients: ketogenic diet eligibility, "
            "malnutrition risk screening, AED-specific nutrient depletion analysis, "
            "and medication-nutrition interaction counseling"
        ),
        "modules": {
            "ketogenic_diet_eligibility": ketogenic_diet_eligibility(patient_id),
            "malnutrition_screening": malnutrition_screening(patient_id),
            "nutrient_analysis": nutrient_analysis(patient_id),
            "medication_nutrition_interaction": medication_nutrition_interaction(patient_id),
        },
    }
