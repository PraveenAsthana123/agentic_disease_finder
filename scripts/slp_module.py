"""
Speech-Language Pathologist (SLP) Module — Epilepsy Clinical Dashboard
======================================================================
Language assessment, dysphagia screening, cognitive-communication evaluation,
AED speech/language side-effect monitoring, and therapy goal tracking — all
from REAL patient data in data/clinical.db.

Clinical evidence base:
  - Helmstaedter C & Witt JA (2017). Clinical neuropsychology in epilepsy:
    theoretical and practical issues. Handb Clin Neurol 139:437-459.
  - Bell BD et al. (2011). Confrontation naming after anterior temporal
    lobectomy: Influences of age at onset, education, and surgery type.
    Epilepsy Behav 22:391-397.
  - Hamberger MJ & Cole J (2011). Language organization and reorganization
    in epilepsy. Neuropsychol Rev 21:240-251.
  - ASHA Practice Portal: Speech-Language Pathology in Epilepsy —
    assessment of language lateralization, naming, verbal fluency,
    discourse, and swallowing in epilepsy populations.
  - Perucca P & Gilliam FG (2012). Adverse effects of antiepileptic drugs.
    Lancet Neurol 11:792-802.
  - Mula M & Trimble MR (2009). Antiepileptic drug-induced cognitive
    adverse effects. CNS Drugs 23:121-137.

Endpoints:
  /api/slp                          — full dashboard (all sub-analyses)
  /api/slp/language-assessment      — language lateralization, BNT, Token Test, fluency
  /api/slp/swallowing               — dysphagia screening, aspiration risk
  /api/slp/cognitive-communication  — word-finding, naming, discourse, pragmatics
  /api/slp/aed-speech-effects       — AED-related speech/language side effects
  /api/slp/therapy-goals            — active therapy goals + progress tracking
  /api/slp/definitions              — metric definitions + clinical references

All data from REAL patients, medications, and assessments in data/clinical.db.
"""

import sqlite3
import os
import json
from collections import defaultdict
from datetime import datetime, timedelta

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "clinical.db",
)

# ─── AED speech/language side-effect knowledge base ──────────────────
# Sourced from Helmstaedter & Witt 2017, Perucca & Gilliam 2012,
# Mula & Trimble 2009. Clinically validated incidence data.
AED_SPEECH_EFFECTS = {
    "topiramate": {
        "brand": "Topamax",
        "speech_effects": [
            "Word-finding difficulty (20-40% of patients)",
            "Verbal fluency impairment",
            "Reduced phonemic and semantic fluency",
            "Expressive language slowing",
            "Anomia (particularly at doses >200mg/day)",
        ],
        "severity": "high",
        "incidence_pct": "20-40",
        "mechanism": "Carbonic anhydrase inhibition + glutamate antagonism affecting language networks",
        "clinical_note": "Most common AED to cause language complaints; dose-dependent; often reason for discontinuation",
        "reference": "Helmstaedter & Witt 2017; Mula & Trimble 2009",
    },
    "zonisamide": {
        "brand": "Zonegran",
        "speech_effects": [
            "Cognitive slowing affecting speech rate",
            "Word-finding difficulty (10-20% of patients)",
            "Reduced verbal processing speed",
        ],
        "severity": "moderate",
        "incidence_pct": "10-20",
        "mechanism": "Carbonic anhydrase inhibition (similar to topiramate but less pronounced)",
        "clinical_note": "Similar mechanism to topiramate; speech effects generally milder but monitor in combination therapy",
        "reference": "Perucca & Gilliam 2012",
    },
    "phenobarbital": {
        "brand": "Luminal",
        "speech_effects": [
            "Cognitive slowing with secondary speech rate reduction",
            "Dysarthria at high serum levels",
            "Reduced verbal output / spontaneity",
            "Sedation-related speech imprecision",
        ],
        "severity": "moderate",
        "incidence_pct": "15-30",
        "mechanism": "GABAergic potentiation causing global CNS depression affecting motor speech and language processing",
        "clinical_note": "Older barbiturate AED with well-documented cognitive and speech effects; avoid in patients requiring high verbal function",
        "reference": "Helmstaedter & Witt 2017",
    },
    "levetiracetam": {
        "brand": "Keppra",
        "speech_effects": [
            "Generally speech-neutral",
            "Occasional irritability/behavioral changes affecting pragmatic communication",
            "Rare: reduced verbal output in context of mood change",
        ],
        "severity": "low",
        "incidence_pct": "<5",
        "mechanism": "SV2A modulation — minimal direct language impact; behavioral effects may indirectly affect social communication",
        "clinical_note": "Low speech risk; irritability (5-10%) may affect pragmatic/social communication skills",
        "reference": "Perucca & Gilliam 2012; Helmstaedter & Witt 2017",
    },
    "lamotrigine": {
        "brand": "Lamictal",
        "speech_effects": [
            "Generally speech-positive",
            "May improve cognitive function and verbal fluency compared to other AEDs",
            "Minimal word-finding impairment",
            "Rare: insomnia-related concentration effects on speech",
        ],
        "severity": "low",
        "incidence_pct": "<5",
        "mechanism": "Sodium channel blockade with relatively favorable cognitive profile; may enhance alertness",
        "clinical_note": "Often preferred when preserving language function is priority; good option for patients with pre-existing language concerns",
        "reference": "Helmstaedter & Witt 2017; Mula & Trimble 2009",
    },
    "valproic acid": {
        "brand": "Depakote",
        "speech_effects": [
            "Mild cognitive slowing at high doses",
            "Tremor affecting speech precision (dose-dependent)",
            "Encephalopathy with dysarthria (rare, at toxic levels)",
        ],
        "severity": "low-moderate",
        "incidence_pct": "5-15",
        "mechanism": "Multi-mechanism; tremor via cerebellar effects; encephalopathy from hyperammonemia",
        "clinical_note": "Speech effects usually mild; tremor most common concern; monitor ammonia if cognitive/speech decline",
        "reference": "Perucca & Gilliam 2012",
    },
    "carbamazepine": {
        "brand": "Tegretol",
        "speech_effects": [
            "Diplopia/ataxia may impair speech coordination at high levels",
            "Mild cognitive slowing",
            "Dysarthria at supratherapeutic levels",
        ],
        "severity": "low-moderate",
        "incidence_pct": "5-15",
        "mechanism": "Sodium channel blockade; dose-dependent cerebellar effects",
        "clinical_note": "Speech effects typically dose-related; check serum levels if speech deterioration reported",
        "reference": "Perucca & Gilliam 2012",
    },
    "phenytoin": {
        "brand": "Dilantin",
        "speech_effects": [
            "Cerebellar dysarthria at supratherapeutic levels",
            "Nystagmus and ataxia affecting speech coordination",
            "Cognitive slowing with chronic use",
        ],
        "severity": "moderate",
        "incidence_pct": "10-20",
        "mechanism": "Sodium channel blockade with narrow therapeutic index; cerebellar toxicity at high levels",
        "clinical_note": "Narrow therapeutic window — small dose changes can produce speech/coordination toxicity; monitor levels closely",
        "reference": "Helmstaedter & Witt 2017",
    },
    "oxcarbazepine": {
        "brand": "Trileptal",
        "speech_effects": [
            "Mild cognitive effects, generally less than carbamazepine",
            "Dizziness may affect speech at high doses",
        ],
        "severity": "low",
        "incidence_pct": "<10",
        "mechanism": "Sodium channel blockade (MHD active metabolite); better cognitive profile than CBZ",
        "clinical_note": "Generally favorable speech profile compared to carbamazepine",
        "reference": "Perucca & Gilliam 2012",
    },
    "perampanel": {
        "brand": "Fycompa",
        "speech_effects": [
            "Dizziness and somnolence affecting speech clarity",
            "Aggression/irritability affecting pragmatic communication",
            "Dysarthria reported at higher doses (8-12mg)",
        ],
        "severity": "moderate",
        "incidence_pct": "10-15",
        "mechanism": "AMPA receptor antagonism — cerebellar and behavioral effects",
        "clinical_note": "Dose-dependent; titrate slowly; monitor speech and behavioral changes",
        "reference": "Perucca & Gilliam 2012",
    },
}

# ─── Language assessment normative references ────────────────────────
LANGUAGE_NORMS = {
    "boston_naming_test": {
        "full_name": "Boston Naming Test (BNT)",
        "max_score": 60,
        "normal_cutoff": 48,
        "impaired_cutoff": 38,
        "description": "Confrontation naming of line drawings; sensitive to anomia in temporal lobe epilepsy",
        "reference": "Kaplan E et al. 1983; Bell et al. 2011",
    },
    "token_test": {
        "full_name": "Token Test (Revised)",
        "max_score": 36,
        "normal_cutoff": 29,
        "impaired_cutoff": 22,
        "description": "Auditory comprehension of commands of increasing complexity; detects receptive language deficits",
        "reference": "De Renzi & Faglioni 1978; ASHA Practice Portal",
    },
    "phonemic_fluency": {
        "full_name": "Phonemic / Letter Fluency (FAS/CFL)",
        "normal_range": [30, 60],
        "impaired_cutoff": 20,
        "unit": "words per 3 minutes (F+A+S)",
        "description": "Generate words beginning with given letters in 60s each; frontal-executive language measure",
        "reference": "Benton & Hamsher 1976; Helmstaedter & Witt 2017",
    },
    "semantic_fluency": {
        "full_name": "Semantic / Category Fluency (Animals)",
        "normal_range": [15, 30],
        "impaired_cutoff": 10,
        "unit": "words per 60 seconds",
        "description": "Generate words from a semantic category (animals); temporal lobe language measure",
        "reference": "Tombaugh et al. 1999; Hamberger & Cole 2011",
    },
}

# ─── Cognitive-communication domains ─────────────────────────────────
COGNITIVE_COMM_DOMAINS = [
    {
        "domain": "Word-finding / Anomia",
        "description": "Ability to retrieve words during spontaneous speech and confrontation naming",
        "assessment_tools": ["Boston Naming Test", "Responsive naming", "Discourse analysis"],
        "epilepsy_relevance": "Temporal lobe epilepsy strongly associated with naming deficits; lateralized to dominant hemisphere",
        "reference": "Bell et al. 2011; Hamberger & Cole 2011",
    },
    {
        "domain": "Verbal fluency",
        "description": "Speed and efficiency of word generation under phonemic and semantic constraints",
        "assessment_tools": ["FAS/CFL letter fluency", "Category fluency (animals, fruits)"],
        "epilepsy_relevance": "Phonemic fluency: frontal lobe function; Semantic fluency: temporal lobe function — dissociation aids lateralization",
        "reference": "Helmstaedter & Witt 2017; Hamberger & Cole 2011",
    },
    {
        "domain": "Discourse / Narrative",
        "description": "Coherence, informativeness, and organization of connected speech",
        "assessment_tools": ["Story retell", "Picture description", "Conversational analysis"],
        "epilepsy_relevance": "Patients with TLE may show reduced informativeness, circumlocution, and topic maintenance difficulties",
        "reference": "ASHA Practice Portal; Bell et al. 2011",
    },
    {
        "domain": "Auditory comprehension",
        "description": "Understanding of spoken language at word, sentence, and discourse levels",
        "assessment_tools": ["Token Test", "BDAE auditory comprehension", "Functional listening tasks"],
        "epilepsy_relevance": "Dominant temporal lobe dysfunction may impair auditory-verbal comprehension",
        "reference": "De Renzi & Faglioni 1978; Hamberger & Cole 2011",
    },
    {
        "domain": "Pragmatic communication",
        "description": "Social language use: turn-taking, topic management, inference, humor, sarcasm",
        "assessment_tools": ["Pragmatic Protocol", "Conversational observation", "Social communication checklist"],
        "epilepsy_relevance": "AED behavioral side effects (irritability, mood changes) and frontal lobe involvement can impair pragmatics",
        "reference": "ASHA Practice Portal; Mula & Trimble 2009",
    },
    {
        "domain": "Reading / Written language",
        "description": "Reading comprehension, written expression, and spelling",
        "assessment_tools": ["Reading comprehension passages", "Written narrative sample", "GORT-5"],
        "epilepsy_relevance": "Left hemisphere epilepsy may affect reading; AED cognitive effects compound reading difficulties",
        "reference": "Helmstaedter & Witt 2017",
    },
]

# ─── Dysphagia risk factors in epilepsy ──────────────────────────────
DYSPHAGIA_RISK_FACTORS = {
    "post_ictal_aspiration": {
        "description": "Risk of aspiration during post-ictal confusion / reduced consciousness",
        "weight": 2.5,
        "reference": "Epilepsia reviews: aspiration pneumonia is a recognized seizure-related mortality factor",
    },
    "aed_sedation": {
        "description": "AED-related sedation impairing swallow coordination",
        "weight": 1.5,
        "reference": "Perucca & Gilliam 2012: sedating AEDs (phenobarbital, clobazam, benzodiazepines) impair pharyngeal coordination",
    },
    "vagus_nerve_stimulator": {
        "description": "VNS therapy — vocal cord paresis / swallowing difficulty",
        "weight": 2.0,
        "reference": "Morris GL (1999): VNS side effects include hoarseness, cough, dysphagia in 10-20% of patients",
    },
    "seizure_related_injury": {
        "description": "Oral/dental injury from seizures affecting mastication and swallowing",
        "weight": 1.5,
        "reference": "Epilepsy Foundation: dental/oral injuries common in tonic-clonic seizures",
    },
    "elderly_with_polytherapy": {
        "description": "Age > 65 with multiple AEDs — compounded swallowing and sedation risk",
        "weight": 2.0,
        "reference": "Helmstaedter & Witt 2017: elderly patients at highest risk of AED-related swallowing dysfunction",
    },
    "intellectual_disability": {
        "description": "Co-occurring intellectual disability with epilepsy",
        "weight": 1.5,
        "reference": "ASHA Practice Portal: epilepsy + ID population has elevated dysphagia prevalence",
    },
}


# ─── Helper: get DB connection ─────────────────────────────────────────
def _get_conn():
    """Return a sqlite3 connection to clinical.db. Returns None if DB missing."""
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn, table_name):
    """Check if a table exists in the DB."""
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return c.fetchone() is not None


def _get_patients(conn, patient_id=None):
    """Fetch patient rows. Filter by patient_id if given."""
    c = conn.cursor()
    if patient_id:
        c.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,))
    else:
        c.execute("SELECT * FROM patients")
    return [dict(r) for r in c.fetchall()]


def _get_medications(conn, patient_id=None):
    """Fetch medications with parsed fields_json."""
    c = conn.cursor()
    if patient_id:
        c.execute("SELECT * FROM medications WHERE patient_id=?", (patient_id,))
    else:
        c.execute("SELECT * FROM medications")
    rows = []
    for r in c.fetchall():
        d = dict(r)
        try:
            d["fields"] = json.loads(d.get("fields_json", "{}"))
        except Exception:
            d["fields"] = {}
        rows.append(d)
    return rows


def _get_assessments(conn, patient_id=None, instrument=None):
    """Fetch assessments, optionally filtered by patient and/or instrument."""
    if not _table_exists(conn, "assessments"):
        return []
    c = conn.cursor()
    q = "SELECT * FROM assessments WHERE 1=1"
    params = []
    if patient_id:
        q += " AND patient_id=?"
        params.append(patient_id)
    if instrument:
        q += " AND instrument=?"
        params.append(instrument)
    c.execute(q, params)
    return [dict(r) for r in c.fetchall()]


def _get_seizure_diary(conn, patient_id=None):
    """Fetch seizure diary entries."""
    if not _table_exists(conn, "seizure_diary"):
        return []
    c = conn.cursor()
    if patient_id:
        c.execute("SELECT * FROM seizure_diary WHERE patient_id=?", (patient_id,))
    else:
        c.execute("SELECT * FROM seizure_diary")
    return [dict(r) for r in c.fetchall()]


def _extract_drug_names(meds):
    """Extract all drug names from medication records."""
    drug_names = []
    for m in meds:
        f = m.get("fields", {})
        dn = f.get("drug_name", "")
        if dn:
            drug_names.append(dn)
        aed_list = f.get("aed", [])
        if isinstance(aed_list, list):
            for a in aed_list:
                if a and a not in drug_names:
                    drug_names.append(a)
    return drug_names


# ═══════════════════════════════════════════════════════════════════════
#  1. LANGUAGE ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════

def generate_slp_language_assessment(patient_id=None):
    """Language lateralization, Boston Naming Test, Token Test, verbal fluency
    scores from assessments table. Cross-references epilepsy lateralization
    data from patients table where available."""
    conn = _get_conn()
    if not conn:
        return {"error": "clinical.db not found", "patients": []}
    try:
        patients = _get_patients(conn, patient_id)
        if not patients:
            return {
                "title": "SLP Language Assessment",
                "subtitle": "No patient data found" + (f" for {patient_id}" if patient_id else ""),
                "total_patients": 0,
                "patients": [],
                "normative_references": LANGUAGE_NORMS,
            }

        patient_results = []
        for p in patients:
            pid = p["patient_id"]

            # Fetch language-relevant assessments
            bnt_assessments = _get_assessments(conn, pid, "BNT")
            token_assessments = _get_assessments(conn, pid, "TokenTest")
            phonemic_assessments = _get_assessments(conn, pid, "PhonemicFluency")
            semantic_assessments = _get_assessments(conn, pid, "SemanticFluency")
            # Also try alternate instrument names
            if not bnt_assessments:
                bnt_assessments = _get_assessments(conn, pid, "BostonNaming")
            if not phonemic_assessments:
                phonemic_assessments = _get_assessments(conn, pid, "FAS")
            if not phonemic_assessments:
                phonemic_assessments = _get_assessments(conn, pid, "VERBAL_FLUENCY")
            if not semantic_assessments:
                semantic_assessments = _get_assessments(conn, pid, "CategoryFluency")

            # All language assessments for this patient
            all_lang = _get_assessments(conn, pid)
            # Filter to language-related instruments
            lang_instruments = [
                a for a in all_lang
                if any(kw in (a.get("instrument") or "").lower()
                       for kw in ["bnt", "boston", "naming", "token", "fluency",
                                  "fas", "category", "language", "wab", "bdae",
                                  "slp", "speech", "verbal"])
            ]

            # BNT results
            bnt_result = None
            if bnt_assessments:
                latest = max(bnt_assessments, key=lambda x: x.get("created_at", ""))
                score = latest.get("score")
                norms = LANGUAGE_NORMS["boston_naming_test"]
                if score is not None:
                    if score >= norms["normal_cutoff"]:
                        status = "normal"
                    elif score >= norms["impaired_cutoff"]:
                        status = "borderline"
                    else:
                        status = "impaired"
                else:
                    status = "score_unavailable"
                bnt_result = {
                    "score": score,
                    "max_score": norms["max_score"],
                    "status": status,
                    "date": latest.get("created_at"),
                    "normal_cutoff": norms["normal_cutoff"],
                    "impaired_cutoff": norms["impaired_cutoff"],
                }

            # Token Test results
            token_result = None
            if token_assessments:
                latest = max(token_assessments, key=lambda x: x.get("created_at", ""))
                score = latest.get("score")
                norms = LANGUAGE_NORMS["token_test"]
                if score is not None:
                    if score >= norms["normal_cutoff"]:
                        status = "normal"
                    elif score >= norms["impaired_cutoff"]:
                        status = "borderline"
                    else:
                        status = "impaired"
                else:
                    status = "score_unavailable"
                token_result = {
                    "score": score,
                    "max_score": norms["max_score"],
                    "status": status,
                    "date": latest.get("created_at"),
                    "normal_cutoff": norms["normal_cutoff"],
                    "impaired_cutoff": norms["impaired_cutoff"],
                }

            # Phonemic fluency results
            phonemic_result = None
            if phonemic_assessments:
                latest = max(phonemic_assessments, key=lambda x: x.get("created_at", ""))
                score = latest.get("score")
                norms = LANGUAGE_NORMS["phonemic_fluency"]
                if score is not None:
                    if score >= norms["normal_range"][0]:
                        status = "normal"
                    elif score >= norms["impaired_cutoff"]:
                        status = "borderline"
                    else:
                        status = "impaired"
                else:
                    status = "score_unavailable"
                phonemic_result = {
                    "score": score,
                    "unit": norms["unit"],
                    "status": status,
                    "date": latest.get("created_at"),
                    "normal_range": norms["normal_range"],
                    "impaired_cutoff": norms["impaired_cutoff"],
                }

            # Semantic fluency results
            semantic_result = None
            if semantic_assessments:
                latest = max(semantic_assessments, key=lambda x: x.get("created_at", ""))
                score = latest.get("score")
                norms = LANGUAGE_NORMS["semantic_fluency"]
                if score is not None:
                    if score >= norms["normal_range"][0]:
                        status = "normal"
                    elif score >= norms["impaired_cutoff"]:
                        status = "borderline"
                    else:
                        status = "impaired"
                else:
                    status = "score_unavailable"
                semantic_result = {
                    "score": score,
                    "unit": norms["unit"],
                    "status": status,
                    "date": latest.get("created_at"),
                    "normal_range": norms["normal_range"],
                    "impaired_cutoff": norms["impaired_cutoff"],
                }

            # Language lateralization inference
            disease = (p.get("disease") or "").lower()
            lateralization = "unknown"
            if "left" in disease or "left temporal" in disease:
                lateralization = "left_hemisphere"
            elif "right" in disease or "right temporal" in disease:
                lateralization = "right_hemisphere"
            elif "bilateral" in disease:
                lateralization = "bilateral"

            lateralization_note = {
                "left_hemisphere": "Left temporal lobe epilepsy — higher risk of naming and verbal memory deficits (Hamberger & Cole 2011)",
                "right_hemisphere": "Right temporal lobe epilepsy — relatively preserved naming; monitor visuospatial language",
                "bilateral": "Bilateral epilepsy — language lateralization may be atypical; Wada test / fMRI recommended",
                "unknown": "Lateralization not determined from available data — consider fMRI language mapping if surgery planned",
            }.get(lateralization, "")

            # Assessments available flag
            has_assessments = any([bnt_result, token_result, phonemic_result, semantic_result])

            # Summary impairment count
            impairment_count = 0
            for result in [bnt_result, token_result, phonemic_result, semantic_result]:
                if result and result.get("status") == "impaired":
                    impairment_count += 1

            # Recommendations
            recommendations = []
            if not has_assessments:
                recommendations.append("No language assessment data available — recommend baseline BNT, Token Test, and verbal fluency evaluation")
            if bnt_result and bnt_result["status"] == "impaired":
                recommendations.append(f"BNT score {bnt_result['score']}/{bnt_result['max_score']} (impaired) — confrontation naming therapy indicated")
            if token_result and token_result["status"] == "impaired":
                recommendations.append(f"Token Test score {token_result['score']}/{token_result['max_score']} (impaired) — auditory comprehension intervention needed")
            if phonemic_result and phonemic_result["status"] == "impaired":
                recommendations.append(f"Phonemic fluency score {phonemic_result['score']} (impaired) — executive-language intervention indicated")
            if semantic_result and semantic_result["status"] == "impaired":
                recommendations.append(f"Semantic fluency score {semantic_result['score']} (impaired) — semantic network strengthening recommended")
            if lateralization == "left_hemisphere":
                recommendations.append("Left TLE — monitor naming closely pre/post-surgery; consider Wada test for language dominance")
            if impairment_count >= 2:
                recommendations.append(f"Multiple language domains impaired ({impairment_count}/4) — comprehensive SLP intervention recommended")

            patient_results.append({
                "patient_id": pid,
                "age": p.get("age"),
                "gender": p.get("gender"),
                "disease": p.get("disease"),
                "lateralization": lateralization,
                "lateralization_note": lateralization_note,
                "boston_naming_test": bnt_result,
                "token_test": token_result,
                "phonemic_fluency": phonemic_result,
                "semantic_fluency": semantic_result,
                "has_language_assessments": has_assessments,
                "impaired_domain_count": impairment_count,
                "all_language_assessments": lang_instruments,
                "recommendations": recommendations,
            })

        # Aggregate stats
        total_with_assessments = sum(1 for p in patient_results if p["has_language_assessments"])
        total_impaired = sum(1 for p in patient_results if p["impaired_domain_count"] > 0)

        return {
            "title": "SLP Language Assessment",
            "subtitle": "Language lateralization, naming, comprehension, fluency — Bell et al. 2011, Hamberger & Cole 2011",
            "total_patients": len(patient_results),
            "patients_with_assessments": total_with_assessments,
            "patients_with_impairment": total_impaired,
            "normative_references": LANGUAGE_NORMS,
            "patients": patient_results,
        }
    except Exception as exc:
        return {"error": str(exc), "patients": []}
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════
#  2. SWALLOWING / DYSPHAGIA SCREENING
# ═══════════════════════════════════════════════════════════════════════

def generate_slp_swallowing(patient_id=None):
    """Dysphagia screening: aspiration risk from post-ictal state, sedating AEDs,
    VNS therapy, seizure-related oral injuries, age + polytherapy."""
    conn = _get_conn()
    if not conn:
        return {"error": "clinical.db not found", "patients": []}
    try:
        patients = _get_patients(conn, patient_id)
        meds = _get_medications(conn, patient_id)
        seizures = _get_seizure_diary(conn, patient_id)

        if not patients:
            return {
                "title": "SLP Swallowing / Dysphagia Screening",
                "subtitle": "No patient data found" + (f" for {patient_id}" if patient_id else ""),
                "total_patients": 0,
                "patients": [],
                "risk_factors_legend": DYSPHAGIA_RISK_FACTORS,
            }

        meds_by_patient = defaultdict(list)
        for m in meds:
            meds_by_patient[m["patient_id"]].append(m)

        seiz_by_patient = defaultdict(list)
        for s in seizures:
            seiz_by_patient[s["patient_id"]].append(s)

        # Sedating AEDs known to affect swallowing
        sedating_aeds = {"phenobarbital", "clobazam", "pregabalin", "gabapentin",
                         "perampanel", "clonazepam", "diazepam", "lorazepam"}

        patient_results = []
        for p in patients:
            pid = p["patient_id"]
            p_meds = meds_by_patient.get(pid, [])
            p_seizures = seiz_by_patient.get(pid, [])
            age = p.get("age")

            drug_names = _extract_drug_names(p_meds)
            drug_names_lower = [d.lower().strip() for d in drug_names]
            n_drugs = len(set(drug_names))

            factors = {}
            total_score = 0

            # 1. Post-ictal aspiration risk
            severe_seizures = sum(1 for s in p_seizures if (s.get("severity") or "").lower() == "severe")
            if severe_seizures > 0:
                score = DYSPHAGIA_RISK_FACTORS["post_ictal_aspiration"]["weight"]
                factors["post_ictal_aspiration"] = {
                    "present": True,
                    "value": f"{severe_seizures} severe seizures recorded",
                    "score": round(score, 2),
                }
                total_score += score
            else:
                factors["post_ictal_aspiration"] = {"present": len(p_seizures) > 0, "value": f"{len(p_seizures)} seizures (none severe)", "score": 0}

            # 2. AED sedation risk
            sedating_count = sum(1 for d in drug_names_lower if d in sedating_aeds)
            if sedating_count > 0:
                score = DYSPHAGIA_RISK_FACTORS["aed_sedation"]["weight"]
                factors["aed_sedation"] = {
                    "present": True,
                    "value": f"{sedating_count} sedating AED(s) on board",
                    "drugs": [d for d in drug_names if d.lower().strip() in sedating_aeds],
                    "score": round(score, 2),
                }
                total_score += score
            else:
                factors["aed_sedation"] = {"present": False, "value": "No sedating AEDs identified", "score": 0}

            # 3. VNS therapy (check medication/device notes)
            vns_present = False
            for m in p_meds:
                f = m.get("fields", {})
                for val in f.values():
                    if isinstance(val, str) and "vns" in val.lower():
                        vns_present = True
                        break
            # Also check disease field
            if "vns" in (p.get("disease") or "").lower():
                vns_present = True

            if vns_present:
                score = DYSPHAGIA_RISK_FACTORS["vagus_nerve_stimulator"]["weight"]
                factors["vagus_nerve_stimulator"] = {"present": True, "value": "VNS therapy detected", "score": round(score, 2)}
                total_score += score
            else:
                factors["vagus_nerve_stimulator"] = {"present": False, "value": "No VNS detected", "score": 0}

            # 4. Seizure-related oral/dental injury
            oral_injuries = 0
            for s in p_seizures:
                injury = (s.get("injury") or "").lower()
                if any(kw in injury for kw in ["oral", "dental", "tongue", "bite", "mouth", "teeth", "jaw"]):
                    oral_injuries += 1
            if oral_injuries > 0:
                score = DYSPHAGIA_RISK_FACTORS["seizure_related_injury"]["weight"]
                factors["seizure_related_injury"] = {"present": True, "value": f"{oral_injuries} oral/dental injuries", "score": round(score, 2)}
                total_score += score
            else:
                factors["seizure_related_injury"] = {"present": False, "value": "No oral injuries recorded", "score": 0}

            # 5. Elderly with polytherapy
            if age and age >= 65 and n_drugs >= 2:
                score = DYSPHAGIA_RISK_FACTORS["elderly_with_polytherapy"]["weight"]
                factors["elderly_with_polytherapy"] = {"present": True, "value": f"Age {age}, {n_drugs} AEDs", "score": round(score, 2)}
                total_score += score
            else:
                factors["elderly_with_polytherapy"] = {"present": False, "value": f"Age {age or 'unknown'}, {n_drugs} AEDs", "score": 0}

            # 6. Intellectual disability (check disease field)
            disease = (p.get("disease") or "").lower()
            id_present = any(kw in disease for kw in ["intellectual disability", "learning disability", "mental retardation", "developmental delay"])
            if id_present:
                score = DYSPHAGIA_RISK_FACTORS["intellectual_disability"]["weight"]
                factors["intellectual_disability"] = {"present": True, "value": "Co-occurring intellectual disability", "score": round(score, 2)}
                total_score += score
            else:
                factors["intellectual_disability"] = {"present": False, "value": "Not identified", "score": 0}

            total_score = round(min(total_score, 10), 2)

            # Risk level
            if total_score >= 5:
                risk_level = "High"
                risk_color = "red"
                action = "Urgent bedside swallowing assessment; consider videofluoroscopy (VFSS) or FEES; NPO precautions during post-ictal periods; SLP swallowing therapy"
            elif total_score >= 2.5:
                risk_level = "Moderate"
                risk_color = "amber"
                action = "Bedside dysphagia screening recommended; educate on aspiration precautions during post-ictal recovery; review sedating AEDs"
            else:
                risk_level = "Low"
                risk_color = "green"
                action = "Routine swallowing awareness; educate on post-ictal aspiration risk; re-screen if medications change or new seizure types emerge"

            patient_results.append({
                "patient_id": pid,
                "age": age,
                "gender": p.get("gender"),
                "dysphagia_risk_score": total_score,
                "dysphagia_risk_max": 10,
                "risk_level": risk_level,
                "risk_color": risk_color,
                "recommended_action": action,
                "factors": factors,
                "aed_count": n_drugs,
                "seizure_count": len(p_seizures),
            })

        return {
            "title": "SLP Swallowing / Dysphagia Screening",
            "subtitle": "Aspiration risk from seizures, sedating AEDs, VNS, oral injury — ASHA Practice Portal",
            "total_patients": len(patient_results),
            "high_risk_count": sum(1 for p in patient_results if p["risk_level"] == "High"),
            "moderate_risk_count": sum(1 for p in patient_results if p["risk_level"] == "Moderate"),
            "low_risk_count": sum(1 for p in patient_results if p["risk_level"] == "Low"),
            "risk_factors_legend": DYSPHAGIA_RISK_FACTORS,
            "patients": patient_results,
        }
    except Exception as exc:
        return {"error": str(exc), "patients": []}
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════
#  3. COGNITIVE-COMMUNICATION ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════

def generate_slp_cognitive_communication(patient_id=None):
    """Cognitive-communication assessment: word-finding, naming, discourse,
    pragmatics, reading — integrated with epilepsy type and AED profile."""
    conn = _get_conn()
    if not conn:
        return {"error": "clinical.db not found", "patients": []}
    try:
        patients = _get_patients(conn, patient_id)
        meds = _get_medications(conn, patient_id)

        if not patients:
            return {
                "title": "SLP Cognitive-Communication Assessment",
                "subtitle": "No patient data found" + (f" for {patient_id}" if patient_id else ""),
                "total_patients": 0,
                "patients": [],
                "domains_catalog": COGNITIVE_COMM_DOMAINS,
            }

        meds_by_patient = defaultdict(list)
        for m in meds:
            meds_by_patient[m["patient_id"]].append(m)

        patient_results = []
        for p in patients:
            pid = p["patient_id"]
            p_meds = meds_by_patient.get(pid, [])
            age = p.get("age")
            disease = (p.get("disease") or "").lower()

            drug_names = _extract_drug_names(p_meds)
            drug_names_lower = [d.lower().strip() for d in drug_names]

            # Determine epilepsy type for risk stratification
            is_temporal = "temporal" in disease
            is_left = "left" in disease
            is_frontal = "frontal" in disease

            # Assess each cognitive-communication domain
            domain_assessments = []
            total_risk_score = 0
            high_risk_domains = 0

            for dom in COGNITIVE_COMM_DOMAINS:
                domain_risk = "low"
                risk_notes = []

                if dom["domain"] == "Word-finding / Anomia":
                    # Check for topiramate/zonisamide (word-finding drugs)
                    wf_drugs = [d for d in drug_names_lower if d in ("topiramate", "zonisamide")]
                    if wf_drugs:
                        domain_risk = "high"
                        risk_notes.append(f"On {', '.join(wf_drugs)} — known word-finding impairment risk")
                    if is_temporal and is_left:
                        domain_risk = "high"
                        risk_notes.append("Left temporal lobe epilepsy — anomia is hallmark deficit (Bell et al. 2011)")
                    elif is_temporal:
                        if domain_risk != "high":
                            domain_risk = "moderate"
                        risk_notes.append("Temporal lobe epilepsy — naming deficits common")

                    # Check for BNT assessment
                    bnt = _get_assessments(conn, pid, "BNT") or _get_assessments(conn, pid, "BostonNaming")
                    if bnt:
                        latest = max(bnt, key=lambda x: x.get("created_at", ""))
                        score = latest.get("score")
                        if score is not None and score < LANGUAGE_NORMS["boston_naming_test"]["impaired_cutoff"]:
                            domain_risk = "high"
                            risk_notes.append(f"BNT score {score} — below impaired cutoff")

                elif dom["domain"] == "Verbal fluency":
                    if is_frontal:
                        domain_risk = "moderate"
                        risk_notes.append("Frontal lobe epilepsy — phonemic fluency vulnerability")
                    if is_temporal:
                        domain_risk = "moderate"
                        risk_notes.append("Temporal lobe epilepsy — semantic fluency vulnerability")
                    if any(d in drug_names_lower for d in ("topiramate", "zonisamide", "phenobarbital")):
                        domain_risk = "high"
                        risk_notes.append("AED(s) with known fluency impairment effects on board")

                elif dom["domain"] == "Discourse / Narrative":
                    if is_temporal:
                        domain_risk = "moderate"
                        risk_notes.append("TLE — may show reduced informativeness and circumlocution")
                    if any(d in drug_names_lower for d in ("phenobarbital", "topiramate")):
                        if domain_risk != "high":
                            domain_risk = "moderate"
                        risk_notes.append("Sedating/cognitive AEDs may slow discourse production")

                elif dom["domain"] == "Auditory comprehension":
                    if is_temporal and is_left:
                        domain_risk = "high"
                        risk_notes.append("Left TLE — dominant temporal lobe dysfunction may impair auditory comprehension")
                    elif is_temporal:
                        domain_risk = "moderate"
                        risk_notes.append("Temporal lobe epilepsy — monitor receptive language")

                elif dom["domain"] == "Pragmatic communication":
                    if any(d in drug_names_lower for d in ("levetiracetam", "perampanel")):
                        domain_risk = "moderate"
                        risk_notes.append("AED(s) with behavioral/irritability effects — may impact pragmatic skills")
                    if is_frontal:
                        domain_risk = "moderate"
                        risk_notes.append("Frontal lobe epilepsy — pragmatic/social communication vulnerability")

                elif dom["domain"] == "Reading / Written language":
                    if is_left:
                        domain_risk = "moderate"
                        risk_notes.append("Left hemisphere epilepsy — reading may be affected")
                    if any(d in drug_names_lower for d in ("topiramate", "phenobarbital", "phenytoin")):
                        if domain_risk != "high":
                            domain_risk = "moderate"
                        risk_notes.append("Cognitive AED effects may compound reading difficulties")

                if not risk_notes:
                    risk_notes.append("No specific elevated risk factors identified from available data")

                if domain_risk == "high":
                    high_risk_domains += 1
                    total_risk_score += 3
                elif domain_risk == "moderate":
                    total_risk_score += 2
                else:
                    total_risk_score += 1

                domain_assessments.append({
                    "domain": dom["domain"],
                    "description": dom["description"],
                    "epilepsy_relevance": dom["epilepsy_relevance"],
                    "risk_level": domain_risk,
                    "risk_notes": risk_notes,
                    "assessment_tools": dom["assessment_tools"],
                })

            # Overall cognitive-communication profile
            avg_risk = round(total_risk_score / len(COGNITIVE_COMM_DOMAINS), 1)
            if avg_risk >= 2.5 or high_risk_domains >= 2:
                overall_risk = "High"
                overall_color = "red"
            elif avg_risk >= 1.5 or high_risk_domains >= 1:
                overall_risk = "Moderate"
                overall_color = "amber"
            else:
                overall_risk = "Low"
                overall_color = "green"

            recommendations = []
            if overall_risk == "High":
                recommendations.append("Comprehensive cognitive-communication evaluation recommended — multiple domains at risk")
            if high_risk_domains >= 2:
                recommendations.append(f"{high_risk_domains} high-risk domains identified — prioritize assessment and intervention")
            if any(d in drug_names_lower for d in ("topiramate", "zonisamide")):
                recommendations.append("AED speech review: consider alternative if language deficits are functionally limiting")
            if is_temporal and is_left:
                recommendations.append("Pre-surgical language mapping (fMRI/Wada) recommended if surgery is planned")

            patient_results.append({
                "patient_id": pid,
                "age": age,
                "gender": p.get("gender"),
                "disease": p.get("disease"),
                "epilepsy_type": {
                    "temporal": is_temporal,
                    "left_hemisphere": is_left,
                    "frontal": is_frontal,
                },
                "aed_names": list(set(drug_names)),
                "overall_cognitive_comm_risk": overall_risk,
                "overall_risk_color": overall_color,
                "high_risk_domain_count": high_risk_domains,
                "domain_assessments": domain_assessments,
                "recommendations": recommendations,
            })

        return {
            "title": "SLP Cognitive-Communication Assessment",
            "subtitle": "Word-finding, naming, discourse, pragmatics — Helmstaedter & Witt 2017, Hamberger & Cole 2011",
            "total_patients": len(patient_results),
            "high_risk_count": sum(1 for p in patient_results if p["overall_cognitive_comm_risk"] == "High"),
            "moderate_risk_count": sum(1 for p in patient_results if p["overall_cognitive_comm_risk"] == "Moderate"),
            "low_risk_count": sum(1 for p in patient_results if p["overall_cognitive_comm_risk"] == "Low"),
            "domains_catalog": COGNITIVE_COMM_DOMAINS,
            "patients": patient_results,
        }
    except Exception as exc:
        return {"error": str(exc), "patients": []}
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════
#  4. AED SPEECH/LANGUAGE EFFECTS
# ═══════════════════════════════════════════════════════════════════════

def generate_slp_aed_speech_effects(patient_id=None):
    """AED-related speech/language side effects per patient: topiramate word-finding,
    phenobarbital cognitive slowing, etc. Cross-references actual medications."""
    conn = _get_conn()
    if not conn:
        return {"error": "clinical.db not found", "patients": []}
    try:
        patients = _get_patients(conn, patient_id)
        meds = _get_medications(conn, patient_id)

        if not patients:
            return {
                "title": "SLP — AED Speech/Language Effects",
                "subtitle": "No patient data found" + (f" for {patient_id}" if patient_id else ""),
                "total_patients": 0,
                "patients": [],
                "aed_effects_catalog": AED_SPEECH_EFFECTS,
            }

        meds_by_patient = defaultdict(list)
        for m in meds:
            meds_by_patient[m["patient_id"]].append(m)

        patient_results = []
        for p in patients:
            pid = p["patient_id"]
            p_meds = meds_by_patient.get(pid, [])

            drug_names = _extract_drug_names(p_meds)
            unique_drugs = list(set(drug_names))

            # Match each drug against speech effects catalog
            drug_effects = []
            total_speech_risk = 0
            high_risk_drugs = []
            moderate_risk_drugs = []

            for dn in unique_drugs:
                key = dn.lower().strip()
                effects = AED_SPEECH_EFFECTS.get(key)
                if effects:
                    drug_effects.append({
                        "drug_name": dn,
                        "brand": effects["brand"],
                        "speech_effects": effects["speech_effects"],
                        "severity": effects["severity"],
                        "incidence_pct": effects["incidence_pct"],
                        "mechanism": effects["mechanism"],
                        "clinical_note": effects["clinical_note"],
                        "reference": effects["reference"],
                    })
                    if effects["severity"] == "high":
                        total_speech_risk += 3
                        high_risk_drugs.append(dn)
                    elif effects["severity"] in ("moderate", "low-moderate"):
                        total_speech_risk += 2
                        moderate_risk_drugs.append(dn)
                    else:
                        total_speech_risk += 1
                else:
                    drug_effects.append({
                        "drug_name": dn,
                        "brand": "—",
                        "speech_effects": ["Not in SLP speech-effects catalog — monitor and report any speech changes"],
                        "severity": "unknown",
                        "incidence_pct": "—",
                        "mechanism": "—",
                        "clinical_note": "Drug not in current speech-effects knowledge base",
                        "reference": "—",
                    })

            # Additive risk: multiple speech-affecting AEDs compound effects
            additive_risk = False
            if len(high_risk_drugs) + len(moderate_risk_drugs) >= 2:
                additive_risk = True
                total_speech_risk += 2  # bonus for polypharmacy speech risk

            # Overall speech risk
            if total_speech_risk >= 5 or len(high_risk_drugs) >= 1:
                overall_risk = "High"
                risk_color = "red"
            elif total_speech_risk >= 3 or len(moderate_risk_drugs) >= 1:
                overall_risk = "Moderate"
                risk_color = "amber"
            else:
                overall_risk = "Low"
                risk_color = "green"

            recommendations = []
            if high_risk_drugs:
                recommendations.append(f"HIGH speech risk from: {', '.join(high_risk_drugs)} — baseline and periodic language assessment recommended")
            if additive_risk:
                recommendations.append("Multiple speech-affecting AEDs — additive impairment risk; discuss with neurology re: regimen simplification")
            if "topiramate" in [d.lower().strip() for d in unique_drugs]:
                recommendations.append("Topiramate on board (20-40% word-finding difficulty) — monitor BNT and verbal fluency; consider dose reduction if functionally limiting")
            if not drug_effects:
                recommendations.append("No AED records found — verify medication list for speech-effect screening")
            if overall_risk == "Low":
                recommendations.append("Low AED speech risk profile — routine monitoring; reassess if medications change")

            patient_results.append({
                "patient_id": pid,
                "age": p.get("age"),
                "gender": p.get("gender"),
                "aed_count": len(unique_drugs),
                "aed_names": unique_drugs,
                "drug_speech_effects": drug_effects,
                "high_risk_drugs": high_risk_drugs,
                "moderate_risk_drugs": moderate_risk_drugs,
                "additive_risk": additive_risk,
                "overall_speech_risk": overall_risk,
                "risk_color": risk_color,
                "recommendations": recommendations,
            })

        # Aggregate
        total_high = sum(1 for p in patient_results if p["overall_speech_risk"] == "High")
        total_on_topiramate = sum(1 for p in patient_results if "topiramate" in [d.lower().strip() for d in p["aed_names"]])

        return {
            "title": "SLP — AED Speech/Language Effects",
            "subtitle": "AED-related speech side effects — Helmstaedter & Witt 2017, Perucca & Gilliam 2012, Mula & Trimble 2009",
            "total_patients": len(patient_results),
            "high_risk_count": total_high,
            "moderate_risk_count": sum(1 for p in patient_results if p["overall_speech_risk"] == "Moderate"),
            "low_risk_count": sum(1 for p in patient_results if p["overall_speech_risk"] == "Low"),
            "patients_on_topiramate": total_on_topiramate,
            "aed_effects_catalog": AED_SPEECH_EFFECTS,
            "patients": patient_results,
        }
    except Exception as exc:
        return {"error": str(exc), "patients": []}
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════
#  5. THERAPY GOALS & PROGRESS TRACKING
# ═══════════════════════════════════════════════════════════════════════

def generate_slp_therapy_goals(patient_id=None):
    """Active SLP therapy goals and progress tracking. Derives goals from
    language assessment results, AED speech effects, and swallowing risk.
    Uses assessments table for progress data where available."""
    conn = _get_conn()
    if not conn:
        return {"error": "clinical.db not found", "patients": []}
    try:
        patients = _get_patients(conn, patient_id)
        meds = _get_medications(conn, patient_id)
        seizures = _get_seizure_diary(conn, patient_id)

        if not patients:
            return {
                "title": "SLP Therapy Goals & Progress",
                "subtitle": "No patient data found" + (f" for {patient_id}" if patient_id else ""),
                "total_patients": 0,
                "patients": [],
            }

        meds_by_patient = defaultdict(list)
        for m in meds:
            meds_by_patient[m["patient_id"]].append(m)

        seiz_by_patient = defaultdict(list)
        for s in seizures:
            seiz_by_patient[s["patient_id"]].append(s)

        patient_results = []
        for p in patients:
            pid = p["patient_id"]
            p_meds = meds_by_patient.get(pid, [])
            p_seizures = seiz_by_patient.get(pid, [])
            age = p.get("age")
            disease = (p.get("disease") or "").lower()

            drug_names = _extract_drug_names(p_meds)
            drug_names_lower = [d.lower().strip() for d in drug_names]

            # Derive therapy goals from patient profile
            goals = []
            goal_id = 0

            # Goal: Word-finding (if on topiramate/zonisamide or TLE)
            wf_drugs = [d for d in drug_names_lower if d in ("topiramate", "zonisamide")]
            is_tle = "temporal" in disease
            if wf_drugs or is_tle:
                goal_id += 1
                # Check for BNT progress data
                bnt = _get_assessments(conn, pid, "BNT") or _get_assessments(conn, pid, "BostonNaming")
                bnt_scores = []
                for a in bnt:
                    if a.get("score") is not None:
                        bnt_scores.append({"score": a["score"], "date": a.get("created_at")})
                bnt_scores.sort(key=lambda x: x.get("date", ""))

                progress = "no_data"
                if len(bnt_scores) >= 2:
                    diff = bnt_scores[-1]["score"] - bnt_scores[0]["score"]
                    if diff > 0:
                        progress = "improving"
                    elif diff < 0:
                        progress = "declining"
                    else:
                        progress = "stable"
                elif len(bnt_scores) == 1:
                    progress = "baseline_only"

                goals.append({
                    "goal_id": goal_id,
                    "domain": "Word-finding / Naming",
                    "goal": "Improve confrontation naming accuracy to functional level (BNT >= 48/60)",
                    "rationale": f"{'AED-related word-finding risk (' + ', '.join(wf_drugs) + ')' if wf_drugs else ''}"
                                 f"{' + ' if wf_drugs and is_tle else ''}"
                                 f"{'Temporal lobe epilepsy — naming deficit risk' if is_tle else ''}",
                    "interventions": [
                        "Semantic feature analysis (SFA) for naming therapy",
                        "Phonological component analysis (PCA)",
                        "Spaced retrieval training",
                        "Compensatory strategies (circumlocution, self-cueing hierarchies)",
                    ],
                    "measure": "Boston Naming Test (BNT)",
                    "target": ">= 48/60",
                    "progress_data": bnt_scores,
                    "progress_trend": progress,
                    "status": "active",
                })

            # Goal: Verbal fluency
            fluency_drugs = [d for d in drug_names_lower if d in ("topiramate", "zonisamide", "phenobarbital")]
            if fluency_drugs or is_tle:
                goal_id += 1
                # Check fluency progress
                flu = (_get_assessments(conn, pid, "PhonemicFluency")
                       or _get_assessments(conn, pid, "FAS")
                       or _get_assessments(conn, pid, "VERBAL_FLUENCY")
                       or _get_assessments(conn, pid, "SemanticFluency")
                       or _get_assessments(conn, pid, "CategoryFluency"))
                flu_scores = []
                for a in flu:
                    if a.get("score") is not None:
                        flu_scores.append({"score": a["score"], "date": a.get("created_at")})
                flu_scores.sort(key=lambda x: x.get("date", ""))

                progress = "no_data"
                if len(flu_scores) >= 2:
                    diff = flu_scores[-1]["score"] - flu_scores[0]["score"]
                    progress = "improving" if diff > 0 else ("declining" if diff < 0 else "stable")
                elif len(flu_scores) == 1:
                    progress = "baseline_only"

                goals.append({
                    "goal_id": goal_id,
                    "domain": "Verbal Fluency",
                    "goal": "Improve verbal fluency to age-appropriate range (phonemic >= 30 words/3min; semantic >= 15 words/min)",
                    "rationale": f"AED fluency risk ({', '.join(fluency_drugs)})" if fluency_drugs else "TLE — semantic fluency vulnerability",
                    "interventions": [
                        "Category generation practice (timed trials)",
                        "Phonemic cueing strategies",
                        "Semantic clustering training",
                        "Divergent naming exercises",
                    ],
                    "measure": "FAS/CFL + Category Fluency (Animals)",
                    "target": "Phonemic >= 30; Semantic >= 15",
                    "progress_data": flu_scores,
                    "progress_trend": progress,
                    "status": "active",
                })

            # Goal: Dysphagia management (if risk factors present)
            severe_seizures = sum(1 for s in p_seizures if (s.get("severity") or "").lower() == "severe")
            sedating_on_board = [d for d in drug_names_lower if d in ("phenobarbital", "clobazam", "pregabalin", "clonazepam")]
            if severe_seizures > 0 or sedating_on_board or (age and age >= 65):
                goal_id += 1
                rationale_parts = []
                if severe_seizures > 0:
                    rationale_parts.append(f"Severe seizures ({severe_seizures}) — post-ictal aspiration risk")
                if sedating_on_board:
                    rationale_parts.append(f"Sedating AEDs ({', '.join(sedating_on_board)})")
                if age and age >= 65:
                    rationale_parts.append("Age >= 65")
                goals.append({
                    "goal_id": goal_id,
                    "domain": "Swallowing Safety",
                    "goal": "Maintain safe swallowing function; prevent aspiration during post-ictal periods",
                    "rationale": "; ".join(rationale_parts),
                    "interventions": [
                        "Bedside swallowing assessment (baseline + post-seizure protocol)",
                        "Post-ictal swallowing precautions education for caregivers",
                        "Compensatory swallowing strategies (chin tuck, small bolus, thickened liquids if needed)",
                        "Oral motor exercises if weakness identified",
                    ],
                    "measure": "Clinical swallowing evaluation; aspiration pneumonia incidence",
                    "target": "Zero aspiration events; safe PO diet maintained",
                    "progress_data": [],
                    "progress_trend": "no_data",
                    "status": "active",
                })

            # Goal: Pragmatic communication (if on levetiracetam/perampanel or frontal epilepsy)
            pragmatic_drugs = [d for d in drug_names_lower if d in ("levetiracetam", "perampanel")]
            is_frontal = "frontal" in disease
            if pragmatic_drugs or is_frontal:
                goal_id += 1
                rationale_parts = []
                if pragmatic_drugs:
                    rationale_parts.append(f"AED behavioral effects ({', '.join(pragmatic_drugs)} — irritability/mood)")
                if is_frontal:
                    rationale_parts.append("Frontal lobe epilepsy — pragmatic vulnerability")
                goals.append({
                    "goal_id": goal_id,
                    "domain": "Pragmatic Communication",
                    "goal": "Maintain functional social communication skills; manage AED behavioral effects on pragmatics",
                    "rationale": "; ".join(rationale_parts),
                    "interventions": [
                        "Social communication skills training",
                        "Behavioral self-monitoring strategies",
                        "Role-play and video-modeling for conversation skills",
                        "Collaboration with neuropsychology re: behavioral AED effects",
                    ],
                    "measure": "Pragmatic Protocol; conversational observation",
                    "target": "Functional social communication maintained; AED behavioral effects managed",
                    "progress_data": [],
                    "progress_trend": "no_data",
                    "status": "active",
                })

            # Goal: Compensatory communication strategies (for all patients with language risk)
            if goals:
                goal_id += 1
                goals.append({
                    "goal_id": goal_id,
                    "domain": "Compensatory Strategies",
                    "goal": "Patient independently uses compensatory communication strategies in daily life",
                    "rationale": "Epilepsy-related language risk factors identified — proactive strategy training",
                    "interventions": [
                        "Self-cueing hierarchy training (semantic then phonemic then written cues)",
                        "Circumlocution strategies for word-retrieval failures",
                        "Note-taking and external memory aids for important conversations",
                        "Communication partner training for family/caregivers",
                    ],
                    "measure": "Functional communication assessment; patient self-report",
                    "target": "Independent use of >= 3 compensatory strategies in daily communication",
                    "progress_data": [],
                    "progress_trend": "no_data",
                    "status": "active",
                })

            # If no specific goals generated, provide general monitoring goal
            if not goals:
                goal_id += 1
                goals.append({
                    "goal_id": goal_id,
                    "domain": "General Monitoring",
                    "goal": "Maintain baseline communication function; monitor for AED or seizure-related changes",
                    "rationale": "No high-risk speech/language factors identified — routine monitoring recommended",
                    "interventions": [
                        "Annual language screening (BNT, verbal fluency)",
                        "Patient education on reporting speech/language changes",
                        "Re-assess if AEDs change or new seizure types emerge",
                    ],
                    "measure": "BNT + verbal fluency screening",
                    "target": "Maintain within normal limits",
                    "progress_data": [],
                    "progress_trend": "no_data",
                    "status": "monitoring",
                })

            patient_results.append({
                "patient_id": pid,
                "age": age,
                "gender": p.get("gender"),
                "disease": p.get("disease"),
                "aed_names": list(set(drug_names)),
                "total_goals": len(goals),
                "active_goals": sum(1 for g in goals if g["status"] == "active"),
                "monitoring_goals": sum(1 for g in goals if g["status"] == "monitoring"),
                "goals": goals,
            })

        return {
            "title": "SLP Therapy Goals & Progress",
            "subtitle": "Active therapy goals, progress tracking — evidence-based interventions for epilepsy-related communication disorders",
            "total_patients": len(patient_results),
            "total_goals": sum(p["total_goals"] for p in patient_results),
            "total_active_goals": sum(p["active_goals"] for p in patient_results),
            "patients": patient_results,
        }
    except Exception as exc:
        return {"error": str(exc), "patients": []}
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════
#  6. DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

def generate_slp_definitions():
    """Return metric definitions, clinical references, and scoring explanations."""
    return {
        "title": "Speech-Language Pathologist (SLP) Module — Metric Definitions",
        "modules": {
            "language_assessment": {
                "purpose": "Evaluate language lateralization, confrontation naming, auditory comprehension, and verbal fluency in epilepsy patients",
                "data_source": "patients + assessments tables in clinical.db",
                "key_metrics": {
                    "boston_naming_test": LANGUAGE_NORMS["boston_naming_test"],
                    "token_test": LANGUAGE_NORMS["token_test"],
                    "phonemic_fluency": LANGUAGE_NORMS["phonemic_fluency"],
                    "semantic_fluency": LANGUAGE_NORMS["semantic_fluency"],
                },
                "references": [
                    "Bell BD et al. Epilepsy Behav 2011;22:391-397",
                    "Hamberger MJ & Cole J. Neuropsychol Rev 2011;21:240-251",
                ],
            },
            "swallowing_screening": {
                "purpose": "Screen for dysphagia risk from post-ictal state, sedating AEDs, VNS, oral injuries, age, and comorbidities",
                "data_source": "patients + medications + seizure_diary tables in clinical.db",
                "scoring": {
                    "range": "0-10",
                    "low": "<2.5 — routine swallowing awareness",
                    "moderate": "2.5-4.99 — bedside screening recommended",
                    "high": ">=5 — urgent swallowing assessment + NPO precautions",
                },
                "risk_factors": DYSPHAGIA_RISK_FACTORS,
                "references": [
                    "ASHA Practice Portal: Speech-Language Pathology in Epilepsy",
                    "Morris GL. Neurology 1999 — VNS side effects",
                    "Perucca P & Gilliam FG. Lancet Neurol 2012;11:792-802",
                ],
            },
            "cognitive_communication": {
                "purpose": "Assess word-finding, naming, discourse, pragmatics, and reading integrated with epilepsy type and AED profile",
                "data_source": "patients + medications + assessments tables in clinical.db",
                "domains": [d["domain"] for d in COGNITIVE_COMM_DOMAINS],
                "risk_levels": {
                    "low": "No specific elevated risk factors",
                    "moderate": "Epilepsy type or AED profile suggests vulnerability",
                    "high": "Multiple risk factors — active assessment and intervention indicated",
                },
                "references": [
                    "Helmstaedter C & Witt JA. Handb Clin Neurol 2017;139:437-459",
                    "Hamberger MJ & Cole J. Neuropsychol Rev 2011;21:240-251",
                    "ASHA Practice Portal: Speech-Language Pathology in Epilepsy",
                ],
            },
            "aed_speech_effects": {
                "purpose": "Monitor AED-related speech and language side effects per patient's actual medication regimen",
                "data_source": "patients + medications tables in clinical.db",
                "catalog_size": len(AED_SPEECH_EFFECTS),
                "high_risk_drugs": ["topiramate (20-40% word-finding)", "phenobarbital (15-30% cognitive slowing)"],
                "speech_neutral_drugs": ["levetiracetam (<5%)", "lamotrigine (<5%, may improve)"],
                "references": [
                    "Helmstaedter C & Witt JA. Handb Clin Neurol 2017;139:437-459",
                    "Perucca P & Gilliam FG. Lancet Neurol 2012;11:792-802",
                    "Mula M & Trimble MR. CNS Drugs 2009;23:121-137",
                ],
            },
            "therapy_goals": {
                "purpose": "Generate evidence-based SLP therapy goals and track progress using assessment data",
                "data_source": "patients + medications + assessments + seizure_diary tables in clinical.db",
                "goal_domains": [
                    "Word-finding / Naming",
                    "Verbal Fluency",
                    "Swallowing Safety",
                    "Pragmatic Communication",
                    "Compensatory Strategies",
                    "General Monitoring",
                ],
                "interventions_referenced": [
                    "Semantic Feature Analysis (SFA) — Boyle 2004",
                    "Phonological Component Analysis (PCA) — Leonard et al. 2008",
                    "Spaced Retrieval Training — Brush & Camp 1998",
                    "ASHA evidence-based practice guidelines for cognitive-communication disorders",
                ],
            },
        },
        "clinical_references": [
            {
                "authors": "Helmstaedter C & Witt JA",
                "year": 2017,
                "title": "Clinical neuropsychology in epilepsy: theoretical and practical issues",
                "journal": "Handb Clin Neurol 139:437-459",
                "relevance": "Comprehensive review of cognitive and language effects of epilepsy and AEDs",
            },
            {
                "authors": "Bell BD et al.",
                "year": 2011,
                "title": "Confrontation naming after anterior temporal lobectomy",
                "journal": "Epilepsy Behav 22:391-397",
                "relevance": "Naming deficits in temporal lobe epilepsy pre- and post-surgery",
            },
            {
                "authors": "Hamberger MJ & Cole J",
                "year": 2011,
                "title": "Language organization and reorganization in epilepsy",
                "journal": "Neuropsychol Rev 21:240-251",
                "relevance": "Language lateralization and naming organization in epilepsy",
            },
            {
                "authors": "ASHA Practice Portal",
                "year": 2023,
                "title": "Speech-Language Pathology in Epilepsy",
                "journal": "American Speech-Language-Hearing Association",
                "relevance": "Clinical guidelines for SLP assessment and intervention in epilepsy populations",
            },
            {
                "authors": "Perucca P & Gilliam FG",
                "year": 2012,
                "title": "Adverse effects of antiepileptic drugs",
                "journal": "Lancet Neurol 11:792-802",
                "relevance": "AED side effects including speech, cognitive, and swallowing effects",
            },
            {
                "authors": "Mula M & Trimble MR",
                "year": 2009,
                "title": "Antiepileptic drug-induced cognitive adverse effects",
                "journal": "CNS Drugs 23:121-137",
                "relevance": "Detailed review of AED cognitive and language side effects with incidence data",
            },
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
#  FULL DASHBOARD (combines all sub-analyses)
# ═══════════════════════════════════════════════════════════════════════

def generate_slp_dashboard(patient_id=None):
    """Return combined SLP dashboard with all sub-analyses + summary KPIs."""
    language = generate_slp_language_assessment(patient_id)
    swallowing = generate_slp_swallowing(patient_id)
    cog_comm = generate_slp_cognitive_communication(patient_id)
    aed_effects = generate_slp_aed_speech_effects(patient_id)
    therapy = generate_slp_therapy_goals(patient_id)

    # Summary KPIs
    total_patients = language.get("total_patients", 0)
    patients_with_lang_assessments = language.get("patients_with_assessments", 0)
    patients_with_impairment = language.get("patients_with_impairment", 0)
    high_dysphagia = swallowing.get("high_risk_count", 0)
    high_cog_comm = cog_comm.get("high_risk_count", 0)
    high_aed_speech = aed_effects.get("high_risk_count", 0)
    total_goals = therapy.get("total_goals", 0)
    active_goals = therapy.get("total_active_goals", 0)

    return {
        "title": "Speech-Language Pathologist (SLP) Dashboard",
        "subtitle": "Language assessment, dysphagia screening, cognitive-communication, AED speech effects, therapy goals — all from real clinical.db data",
        "summary": {
            "total_patients": total_patients,
            "patients_with_language_assessments": patients_with_lang_assessments,
            "patients_with_language_impairment": patients_with_impairment,
            "high_dysphagia_risk": high_dysphagia,
            "moderate_dysphagia_risk": swallowing.get("moderate_risk_count", 0),
            "high_cognitive_comm_risk": high_cog_comm,
            "high_aed_speech_risk": high_aed_speech,
            "patients_on_topiramate": aed_effects.get("patients_on_topiramate", 0),
            "total_therapy_goals": total_goals,
            "active_therapy_goals": active_goals,
        },
        "language_assessment": language,
        "swallowing": swallowing,
        "cognitive_communication": cog_comm,
        "aed_speech_effects": aed_effects,
        "therapy_goals": therapy,
    }
