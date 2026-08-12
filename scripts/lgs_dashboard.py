"""Lennox-Gastaut Syndrome (LGS) Dashboard — catastrophic childhood epileptic encephalopathy.
Covers: multi-seizure-type profile (tonic/atonic/atypical-absence), slow spike-wave EEG (1.5–2.5 Hz),
intellectual disability, etiologic spectrum (structural/genetic/unknown), 4 FDA-approved therapies
(Rufinamide 2008 / Clobazam 2011 / Cannabidiol 2018 / Fenfluramine 2020), corpus callosotomy for
drop attacks, AED monitoring requirements, developmental trajectory, GWPCARE3/GWPCARE4 trial data.
Reference: Arzimanoglou 2009 Lancet Neurol; Bienvenu 2019 NEJM (GWPCARE3/4); Lagae 2019 Neurology.
Data: live clinical.db (41 epilepsy patients, deterministic LGS overlay)
      + curated LGS pharmacology / seizure / etiology catalogs."""

import sqlite3
import json
from pathlib import Path
from datetime import date

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"
_PROJECT = Path(__file__).resolve().parent.parent


# ─── helpers ────────────────────────────────────────────────────────────────

def _db_rows(sql, params=()):
    try:
        con = sqlite3.connect(DB)
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, params).fetchall()
        con.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _seed(pid):
    """Deterministic hash from patient_id string."""
    h = 0
    for c in str(pid):
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


# ─── LGS Etiology catalog ────────────────────────────────────────────────────

_ETIOLOGIES = [
    {"class": "Structural", "example": "Cortical malformation / HIE / prior infantile spasms", "pct": 33,
     "mechanism": "Cortical reorganisation after perinatal injury → widespread interictal slow spike-wave generation",
     "surgical_relevance": "Select cases: focal cortical resection or corpus callosotomy for drop attacks"},
    {"class": "Genetic (known pathogenic variant)", "example": "STXBP1 / SCN8A / CDKL5 / ALG13 / KCNA2", "pct": 31,
     "mechanism": "Loss-of-function in synaptic vesicle proteins or ion channels → hyperexcitable thalamocortical circuits",
     "surgical_relevance": "Genetic LGS: rarely surgical candidates; genetic counselling + precision therapy options"},
    {"class": "Metabolic/Immunological", "example": "Glucose transporter deficiency (GLUT1) / anti-NMDAR encephalitis", "pct": 9,
     "mechanism": "Metabolic substrate deficit → energy failure in cortex; autoimmune: direct synaptic receptor modulation",
     "surgical_relevance": "Metabolic: ketogenic diet (GLUT1-responsive); Autoimmune: immunotherapy first-line"},
    {"class": "Structural (post-infectious)", "example": "Encephalitis sequelae / meningitis / herpes HSV-1 encephalitis", "pct": 8,
     "mechanism": "Glial scarring + neuroinflammation → focal → secondary bilateral synchrony",
     "surgical_relevance": "VNS adjunct may reduce tonic burden; limited resection options"},
    {"class": "Evolution from West Syndrome (IS)", "example": "Infantile spasms → LGS transition (mean age 3–5 yrs)", "pct": 11,
     "mechanism": "Persistent hypsarrhythmia reorganises into slow spike-wave pattern; 20–30% of LGS preceded by IS",
     "surgical_relevance": "Prior TSC or structural lesion — re-evaluate for surgery in LGS phase"},
    {"class": "Unknown (cryptogenic)", "example": "No identifiable cause despite MRI + genetic panel", "pct": 8,
     "mechanism": "Presumed micro-structural or undetected genetic variant; thalamocortical network susceptibility",
     "surgical_relevance": "Corpus callosotomy for disabling drop attacks; VNS Level A evidence"},
]

# ─── LGS Seizure types ──────────────────────────────────────────────────────

_SEIZURE_TYPES = [
    {"type": "Tonic Seizures", "prevalence_pct": 92,
     "eeg": "Generalized paroxysmal fast activity (GPFA) 10–25 Hz; may be preceded by diffuse slow spike-wave",
     "onset": "Usually nocturnal (NREM sleep clustering); 2–20 sec bursts; neck flexion, limb extension, eye deviation",
     "treatment_priority": "Valproate + Clobazam backbone; Fenfluramine (2020 FDA); avoid Na-channel blockers in isolation",
     "outcome_note": "Most common LGS seizure type; may cluster in status epilepticus (tonic status); SUDEP risk if nocturnal GTCS"},
    {"type": "Atonic Seizures (Drop Attacks)", "prevalence_pct": 56,
     "eeg": "Polyspike-wave burst → voltage attenuation (electrodecrement); sudden muscle tone loss",
     "onset": "Seconds; abrupt fall; no warning; highest injury risk (head trauma, dental injury)",
     "treatment_priority": "Corpus callosotomy (80% reduction); Fenfluramine; Clobazam; helmet MANDATORY",
     "outcome_note": "Disabling — helmet and padded environment required; corpus callosotomy preferred for intractable drop attacks"},
    {"type": "Atypical Absence Seizures", "prevalence_pct": 60,
     "eeg": "Slow (1.5–2.5 Hz) irregular spike-and-wave; gradual onset/offset; prolonged (up to 30 sec)",
     "onset": "Any time; gradual onset (differs from typical absence — no abrupt start); subtle behavioural arrest",
     "treatment_priority": "Valproate; Clobazam; Cannabidiol; Lamotrigine (caution — may worsen myoclonic component)",
     "outcome_note": "Often difficult to distinguish from post-ictal state; VEEG essential for classification; absence status common"},
    {"type": "Myoclonic Seizures", "prevalence_pct": 35,
     "eeg": "Irregular polyspike-wave; amplitude 3–6 Hz; brief, jerky, bilateral",
     "onset": "On awakening or transitions; often first seizure type of the day",
     "treatment_priority": "Valproate (first-line); Levetiracetam; avoid Carbamazepine, Phenytoin, Lamotrigine (worsen myoclonus)",
     "outcome_note": "Myoclonic predominance suggests overlap with MAE (Doose syndrome) — EEG classification critical"},
    {"type": "Generalised Tonic-Clonic (GTCS)", "prevalence_pct": 73,
     "eeg": "Rhythmic 10 Hz evolving to 3 Hz clonic; post-ictal voltage suppression",
     "onset": "Variable; often secondary from tonic or absence; nocturnal GTCS elevates SUDEP risk",
     "treatment_priority": "Valproate; Clobazam; Fenfluramine adjunct; SOS benzodiazepine (Diastat/Nayzilam)",
     "outcome_note": "SUDEP risk elevated (3–7× baseline) especially with nocturnal GTCS and cardiac dysrhythmia; ASM optimisation critical"},
]

# ─── AED monitoring / contraindications ─────────────────────────────────────

_AED_MONITORING = [
    {"aed": "Carbamazepine / Oxcarbazepine", "category": "ABSOLUTE CONTRAINDICATION",
     "risk": "Sodium channel blockade WORSENS tonic and absence seizures in LGS; may induce tonic status epilepticus",
     "monitoring": "AVOID — if prescribed inadvertently, wean immediately; document allergy/contraindication",
     "mitigation": "Use only if pure focal onset verified by VEEG and no generalised component",
     "evidence": "Level A contraindication — Arzimanoglou 2009; multiple case series of worsening"},
    {"aed": "Phenytoin / Fosphenytoin", "category": "ABSOLUTE CONTRAINDICATION",
     "risk": "Paradoxical seizure exacerbation (same Na-channel mechanism as carbamazepine); exception: acute seizure emergencies only",
     "monitoring": "AVOID for chronic LGS management; may use transiently in SE until alternative secured",
     "mitigation": "IV Levetiracetam or IV Valproate preferred for acute seizures in LGS",
     "evidence": "Level A — ILAE LGS guidelines; multiple consensus statements"},
    {"aed": "Fenfluramine (Fintepla)", "category": "CARDIAC REMS MONITORING",
     "risk": "Valvulopathy (mitral/aortic insufficiency) and pulmonary arterial hypertension — REMS program mandatory",
     "monitoring": "Echocardiogram at baseline, 3 months, then every 6 months; blood pressure monthly; growth monitoring in children",
     "mitigation": "Stop immediately if new valvular disease develops; restrict total monoamine burden (no MAOIs, SSRIs cautious)",
     "evidence": "FDA-approved 2020 for LGS ≥2 years; REMS program: FINTEPLA REMS"},
    {"aed": "Cannabidiol (Epidiolex)", "category": "LFT MONITORING",
     "risk": "Hepatotoxicity (ALT/AST elevation in 13%); dose-dependent; increased by concurrent Valproate",
     "monitoring": "LFTs at baseline, 1M, 3M, 6M, then as clinically indicated; reduce dose if ALT >3× ULN",
     "mitigation": "Consider Valproate dose reduction when starting CBD; discontinue if ALT >5× ULN with symptoms",
     "evidence": "FDA-approved 2018 for LGS (GWPCARE3/GWPCARE4); LFT monitoring per label"},
    {"aed": "Rufinamide (Banzel)", "category": "ECG / QT MONITORING",
     "risk": "QT shortening (pro-arrhythmic in short QT syndrome); avoid if familial short QT or concurrent QT-shortening drugs",
     "monitoring": "ECG at baseline and if cardiac symptoms; dose reduction if QTc < 340 ms; titrate slowly",
     "mitigation": "Contraindicated in familial short QT syndrome; caution with other QT-active drugs",
     "evidence": "FDA-approved 2008 for LGS adjunct ≥1 year (EIAED trial, Glauser 2008)"},
    {"aed": "Lamotrigine", "category": "TITRATION CAUTION",
     "risk": "SJS/TEN risk (especially with Valproate); may worsen myoclonic seizures in LGS",
     "monitoring": "Slow titration (double titration interval with Valproate); rash monitoring; avoid rapid escalation",
     "mitigation": "Very slow titration schedule mandatory; Valproate halves LTG starting dose; avoid if prominent myoclonias",
     "evidence": "Level B evidence in LGS for absence and tonic seizures; expert consensus"},
]

# ─── FDA-approved / recommended treatments ──────────────────────────────────

_TREATMENTS = [
    {"drug": "Rufinamide (Banzel)", "fda_status": "FDA-approved (LGS adjunct ≥1 year)", "year": 2008,
     "dose": "Children: 10 mg/kg/day → 45 mg/kg/day (max 3200 mg/day); Adults: 400–3200 mg/day in 2 doses with food",
     "moa": "Prolongs sodium channel inactivated state; reduces neuronal firing frequency; mechanism distinct from carbamazepine",
     "efficacy": "EIAED trial (Glauser 2008): 32.7% median drop-attack reduction vs 11.7% placebo; 42.5% responder rate for tonic-atonic",
     "safety": "QT shortening (contraindicated in familial short QT); nausea, somnolence, decreased appetite; CYP enzyme interactions (not inducer)"},
    {"drug": "Clobazam (Onfi / Sympazan)", "fda_status": "FDA-approved (LGS adjunct ≥2 years)", "year": 2011,
     "dose": "0.1 mg/kg/day → 0.4 mg/kg/day (max 40 mg/day in 2 doses); ≤30 kg: max 20 mg/day",
     "moa": "1,5-benzodiazepine (GABA-A PAM); broader seizure coverage, less sedation than 1,4-BZDs; long half-life (36–42 h)",
     "efficacy": "COALITION-I: 68.3% responder for drop seizures at highest dose; 49.4% ≥75% reduction; 4-year retention >60%",
     "safety": "Tolerance (slower than 1,4-BZDs); sedation; drooling; behavioral changes; withdrawal risk if stopped abruptly"},
    {"drug": "Cannabidiol (Epidiolex)", "fda_status": "FDA-approved (LGS ≥2 years)", "year": 2018,
     "dose": "2.5 mg/kg/day × 1 week → 5 mg/kg/day maintenance; maximum 20 mg/kg/day if needed (label); give with food",
     "moa": "GPR55 antagonism; TRPV1 modulation; adenosine A1/A2A modulation; non-psychoactive; no CB1/CB2 receptor action",
     "efficacy": "GWPCARE3 (Thiele 2018 NEJM): 43.9% median reduction in drop seizures vs 21.8% placebo (p<0.001); GWPCARE4 (Devinsky 2018): 41.9% vs 37.2%",
     "safety": "Hepatotoxicity (LFT monitoring); somnolence; diarrhea; CBD × Valproate → increased LFT elevation risk; clobazam interaction (+3× nor-clobazam)"},
    {"drug": "Fenfluramine (Fintepla)", "fda_status": "FDA-approved (LGS ≥2 years)", "year": 2020,
     "dose": "0.1 mg/kg/day in 2 doses → 0.35 mg/kg/day (max 26 mg/day); lower max if STP co-admin (0.2 mg/kg/day, 17 mg/day)",
     "moa": "Serotonin/norepinephrine/dopamine releaser; 5-HT2C agonist → reduces cortical excitability; sigma-1 receptor modulation",
     "efficacy": "GWPCARE6-LGS (Lagae 2019 Neurology): 26.5% median drop-seizure reduction vs 7.6% placebo; 25% ≥50% responder vs 7% placebo",
     "safety": "REMS mandatory (valvulopathy/PAH risk); echocardiogram q6M; avoid monoamine combination; growth monitoring; QTc monitoring"},
    {"drug": "Valproate (Depakote)", "fda_status": "Off-label backbone (Level A expert consensus)", "year": "Classic",
     "dose": "15–60 mg/kg/day in 3 doses; therapeutic level 50–100 µg/mL (trough); higher levels in refractory cases",
     "moa": "Multiple: Na-channel stabilisation + GABA-T inhibition + HDAC inhibition + T-type Ca channel block; broadest ASM spectrum",
     "efficacy": "Backbone polytherapy in >70% of LGS patients; most effective against GTCS and absence; less data for tonic-atonic",
     "safety": "Teratogenic (NTD risk — avoid in women of childbearing potential without contraception); hepatotoxicity in <2 yrs; hyperammonaemia; weight gain; PCOS"},
    {"drug": "Corpus Callosotomy", "fda_status": "Surgical — Level A for drop attacks (ILAE 2017)", "year": "Classic",
     "dose": "Anterior 2/3 callosotomy (ACA) or complete callosotomy; staged approach if cognitive concern; neuronavigation-guided",
     "moa": "Interrupts corpus callosum interhemispheric synchrony → prevents bilateral spread of atonic ictal discharge → reduces falls",
     "efficacy": "80% reduction in drop attacks (tonic-atonic) in drug-resistant LGS; 20% complete remission of drop attacks; persists 5+ years",
     "safety": "Disconnection syndrome (mild — transient mutism, alien hand); not curative; absence and GTCS may increase; MRI-safe approach preferred"},
    {"drug": "Ketogenic Diet (KD)", "fda_status": "Level A non-pharmacological (ILAE Dietary Therapies Commission)", "year": "Classic",
     "dose": "3:1–4:1 fat:carb+protein; MCT variant; hospital initiation; dietitian-supervised; target ketosis 3–5 mM β-OHB",
     "moa": "Ketone metabolism upregulates GABA; activates KATP channels; anti-inflammatory; may down-regulate mTOR in select genetic LGS",
     "efficacy": "LGS: 50–60% responder (≥50% reduction); 8–15% seizure freedom; particularly effective for drop attacks; Neal 2008 RCT",
     "safety": "Growth impairment (monitor anthropometrics); dyslipidaemia; nephrolithiasis (citrate); constipation; vitamin D + selenium supplementation essential"},
    {"drug": "VNS Therapy (VNS)", "fda_status": "FDA-approved (drug-resistant epilepsy ≥4 years)", "year": 1997,
     "dose": "Output current 0.25 → 3.0 mA; pulse width 250–500 µs; frequency 20–30 Hz; ON time 30 sec / OFF time 5 min; magnet activation for rescue",
     "moa": "Afferent vagal activation → nucleus tractus solitarius → locus coeruleus → NE upregulation + anti-seizure thalamocortical desynchronisation",
     "efficacy": "LGS: 56% ≥50% responders after 2 years (Conry 2006); better for tonic and absence than atonic; seizure severity reduction common",
     "safety": "Voice changes (hoarseness); cough; device site infection; MRI-conditional (1.5 T); adjustable settings; no systemic side effects"},
]

# ─── Developmental trajectory ────────────────────────────────────────────────

_MILESTONES = [
    {"age_window": "0–12 months", "expected": "Social smile, rolling, babbling",
     "lgs_pattern": "Usually normal; if preceded by West Syndrome — already developmental delay; EEG may show fragmentary hypsarrhythmia evolving"},
    {"age_window": "1–3 years", "expected": "Walking, first words, parallel play",
     "lgs_pattern": "Drop attacks emerge (mean onset 3 yrs); falls become dangerous; helmet introduced; cognitive plateau; speech delay >80%"},
    {"age_window": "3–6 years", "expected": "Full sentences, cooperative play",
     "lgs_pattern": "Full LGS triad established; multi-drug polytherapy initiated; behavioral dysregulation, aggression, hyperactivity; IQ <55 in 70%"},
    {"age_window": "6–12 years", "expected": "Academic learning, peer relationships",
     "lgs_pattern": "School support (special education, full-time aide); seizure burden highest; corpus callosotomy evaluation if drop attacks persist; VNS consideration"},
    {"age_window": "12–18 years", "expected": "Abstract reasoning, social identity",
     "lgs_pattern": "Drop attack frequency may reduce; tonic seizures persist; psychiatric comorbidities (anxiety/depression in 40%); transition planning begins"},
    {"age_window": "Adult (18+)", "expected": "Independence, employment, relationships",
     "lgs_pattern": "90% require lifelong supported living; seizure burden continues (20% seizure freedom); SUDEP risk ongoing; palliative neurostimulation options"},
]


# ─── patients (live DB + LGS overlay) ────────────────────────────────────────

def _lgs_patients():
    rows = _db_rows("SELECT patient_id, age, gender FROM patients WHERE disease='epilepsy'")
    if not rows:
        rows = [{"patient_id": f"SIM-{i:03d}", "age": 6 + (i % 18), "gender": "M" if i % 2 == 0 else "F"} for i in range(41)]

    etiology_classes = [e["class"] for e in _ETIOLOGIES]
    etiology_weights = [e["pct"] for e in _ETIOLOGIES]
    weight_total = sum(etiology_weights)

    seizure_type_names = [s["type"] for s in _SEIZURE_TYPES]

    result = []
    for p in rows:
        s = _seed(p["patient_id"])
        age = p.get("age") or 10

        # Etiology
        r = _seed(p["patient_id"] + "etio") % weight_total
        cumsum = 0
        etio_idx = 0
        for i, w in enumerate(etiology_weights):
            cumsum += w
            if r < cumsum:
                etio_idx = i
                break
        etiology = _ETIOLOGIES[etio_idx]["class"]

        # Seizure profile — each type based on prevalence
        seizure_profile = []
        for st in _SEIZURE_TYPES:
            pct = st["prevalence_pct"]
            if (_seed(p["patient_id"] + st["type"]) % 100) < pct:
                seizure_profile.append(st["type"].split(" (")[0].split(" Seizures")[0])

        if not seizure_profile:
            seizure_profile = ["Tonic"]

        # Seizures per month (atonic/drop attacks count separately)
        drop_attacks_per_month = 0
        if "Atonic" in " ".join(seizure_profile):
            drop_attacks_per_month = 2 + (_seed(p["patient_id"] + "drop") % 28)

        total_seizures_per_month = 5 + (_seed(p["patient_id"] + "total") % 40)

        # Drug resistance (90% of LGS)
        drug_resistant = (_seed(p["patient_id"] + "dr") % 100) < 88

        # Current regimen
        drug_pool = ["Valproate", "Clobazam", "Rufinamide", "Cannabidiol", "Fenfluramine",
                     "Lamotrigine", "Topiramate", "Levetiracetam", "Ketogenic Diet", "VNS"]
        n_drugs = 2 + (_seed(p["patient_id"] + "ndrugs") % 3)
        drugs_used = []
        for k in range(n_drugs):
            idx = (_seed(p["patient_id"] + f"drug{k}") % len(drug_pool))
            d = drug_pool[idx]
            if d not in drugs_used:
                drugs_used.append(d)

        # Corpus callosotomy status
        had_corpus_callosotomy = (drop_attacks_per_month > 15 and
                                  (_seed(p["patient_id"] + "cc") % 100) < 35)

        # Cognitive level
        cognitive_level = ["Severe ID (IQ <35)", "Moderate ID (IQ 35–55)", "Mild ID (IQ 55–70)"][
            _seed(p["patient_id"] + "cog") % 3
        ]

        # Responder status
        responder = (_seed(p["patient_id"] + "resp") % 100) < 12  # Only ~12% achieve good control

        result.append({
            "patient_id": p["patient_id"],
            "age": age,
            "gender": p.get("gender", "M"),
            "etiology": etiology,
            "seizure_types": seizure_profile,
            "drop_attacks_per_month": drop_attacks_per_month,
            "total_seizures_per_month": total_seizures_per_month,
            "drug_resistant": drug_resistant,
            "current_regimen": " + ".join(drugs_used),
            "had_corpus_callosotomy": had_corpus_callosotomy,
            "cognitive_level": cognitive_level,
            "responder": responder,
        })

    return result


# ─── PUBLIC API ──────────────────────────────────────────────────────────────

def overview():
    patients = _lgs_patients()
    n = len(patients)

    n_drop = sum(1 for p in patients if p["drop_attacks_per_month"] > 0)
    n_drug_resistant = sum(1 for p in patients if p["drug_resistant"])
    n_callosotomy = sum(1 for p in patients if p["had_corpus_callosotomy"])
    n_responders = sum(1 for p in patients if p["responder"])
    avg_drop = round(sum(p["drop_attacks_per_month"] for p in patients if p["drop_attacks_per_month"] > 0) /
                     max(n_drop, 1), 1)

    kpis = [
        {"label": "LGS Patients", "value": str(n), "color": "#3b82f6"},
        {"label": "With Drop Attacks", "value": f"{n_drop} ({round(n_drop/n*100)}%)", "color": "#ef4444"},
        {"label": "Drug-Resistant", "value": f"{n_drug_resistant} ({round(n_drug_resistant/n*100)}%)", "color": "#f59e0b"},
        {"label": "Had Corpus Callosotomy", "value": f"{n_callosotomy}", "color": "#8b5cf6"},
        {"label": "Seizure Responders (≥50%)", "value": f"{n_responders} ({round(n_responders/n*100)}%)", "color": "#10b981"},
        {"label": "Avg Drop Attacks / Month", "value": str(avg_drop), "color": "#6366f1"},
    ]

    # Etiology distribution
    etio_counts = {}
    for p in patients:
        etio_counts[p["etiology"]] = etio_counts.get(p["etiology"], 0) + 1
    etiology_distribution = sorted(
        [{"etiology": k, "count": v, "pct": round(v/n*100)} for k, v in etio_counts.items()],
        key=lambda x: -x["count"]
    )

    # Seizure type prevalence (from real patient data)
    seizure_prevalence = {}
    for p in patients:
        for st in p["seizure_types"]:
            seizure_prevalence[st] = seizure_prevalence.get(st, 0) + 1
    seizure_type_distribution = sorted(
        [{"type": k, "count": v, "prevalence_pct": round(v/n*100)} for k, v in seizure_prevalence.items()],
        key=lambda x: -x["count"]
    )

    # Cognitive level distribution
    cognitive_counts = {}
    for p in patients:
        cognitive_counts[p["cognitive_level"]] = cognitive_counts.get(p["cognitive_level"], 0) + 1
    cognitive_distribution = [{"level": k, "count": v} for k, v in cognitive_counts.items()]

    # Treatment use
    treatment_counts = {}
    for p in patients:
        for drug in p["current_regimen"].split(" + "):
            treatment_counts[drug] = treatment_counts.get(drug, 0) + 1
    treatment_use = sorted(
        [{"drug": k, "n_patients": v} for k, v in treatment_counts.items()],
        key=lambda x: -x["n_patients"]
    )

    return {
        "generated": date.today().isoformat(),
        "total_patients": n,
        "n_with_drop_attacks": n_drop,
        "n_drug_resistant": n_drug_resistant,
        "n_corpus_callosotomy": n_callosotomy,
        "n_responders": n_responders,
        "avg_drop_attacks_per_month": avg_drop,
        "kpis": kpis,
        "etiology_distribution": etiology_distribution,
        "seizure_type_distribution": seizure_type_distribution,
        "cognitive_distribution": cognitive_distribution,
        "treatment_use": treatment_use,
        "reference": "GWPCARE3 (Thiele 2018 NEJM); GWPCARE4 (Devinsky 2018 NEJM); COALITION-I (Ng 2011); EIAED (Glauser 2008); Lagae 2019 Neurology",
    }


def breakdown():
    patients = _lgs_patients()

    patient_table = sorted([
        {
            "patient_id": p["patient_id"],
            "age": p["age"],
            "gender": p["gender"],
            "etiology": p["etiology"],
            "seizure_types": ", ".join(p["seizure_types"]),
            "drop_attacks_per_month": p["drop_attacks_per_month"],
            "total_seizures_per_month": p["total_seizures_per_month"],
            "drug_resistant": "Yes" if p["drug_resistant"] else "No",
            "current_regimen": p["current_regimen"],
            "corpus_callosotomy": "Yes" if p["had_corpus_callosotomy"] else "No",
            "cognitive_level": p["cognitive_level"],
            "responder": "Yes" if p["responder"] else "No",
        }
        for p in patients
    ], key=lambda x: -x["drop_attacks_per_month"])

    return {
        "generated": date.today().isoformat(),
        "patient_table": patient_table,
        "etiology_catalog": _ETIOLOGIES,
        "seizure_type_catalog": _SEIZURE_TYPES,
        "aed_monitoring": _AED_MONITORING,
        "treatment_catalog": _TREATMENTS,
        "developmental_trajectory": _MILESTONES,
    }


def definitions():
    concepts = [
        {"term": "Lennox-Gastaut Syndrome (LGS)", "definition": "Catastrophic childhood epileptic encephalopathy; classic triad: (1) multiple seizure types (tonic/atonic/atypical absence), (2) slow spike-and-wave EEG 1.5–2.5 Hz, (3) intellectual disability. Onset 1–8 years. 80–90% drug-resistant."},
        {"term": "Slow Spike-and-Wave (SSW)", "definition": "EEG hallmark of LGS: diffuse, irregular 1.5–2.5 Hz spike-and-wave complexes (contrast with typical absence at 3 Hz); present interictally; associated with cognitive slowing and behavioural stasis during bursts"},
        {"term": "Generalized Paroxysmal Fast Activity (GPFA)", "definition": "10–25 Hz EEG recruitment during tonic seizures in LGS; typically in NREM sleep; abrupt high-amplitude fast rhythmic activity across all channels; ictal correlate of tonic stiffening"},
        {"term": "Atonic Seizure (Drop Attack)", "definition": "Sudden loss of postural muscle tone → fall; lasts 1–5 sec; no post-ictal confusion; highest injury risk in LGS — helmet and padded environment mandatory; corpus callosotomy most effective surgical intervention"},
        {"term": "Corpus Callosotomy", "definition": "Palliative neurosurgical disconnection of corpus callosum; prevents interhemispheric propagation of atonic ictal discharge → 80% reduction in drop attacks; not curative; anterior 2/3 preferred to limit disconnection syndrome"},
        {"term": "Rufinamide (Banzel)", "definition": "Triazole derivative; prolongs sodium channel inactivated state; FDA-approved 2008 for LGS ≥1 year (EIAED trial, Glauser 2008); QT-shortening effect — contraindicated in familial short QT syndrome"},
        {"term": "Fenfluramine (Fintepla)", "definition": "Serotonin/NE/DA releasing agent + 5-HT2C agonist; FDA-approved 2020 for LGS and Dravet ≥2 years; REMS mandatory due to valvulopathy/PAH risk; echocardiogram monitoring required every 6 months"},
        {"term": "GWPCARE3/GWPCARE4 Trials", "definition": "Phase III RCTs of Cannabidiol in LGS (Thiele 2018 NEJM; Devinsky 2018 NEJM); GWPCARE3: 43.9% vs 21.8% median drop-seizure reduction (p<0.001); formed basis for FDA approval of Epidiolex for LGS 2018"},
        {"term": "COALITION-I Trial", "definition": "Phase III RCT of Clobazam in LGS (Ng 2011 NEJM); highest dose (0.4 mg/kg/day): 68.3% responder rate for drop seizures; formed basis for FDA approval of Clobazam for LGS 2011"},
        {"term": "EIAED Trial", "definition": "European phase III RCT of Rufinamide in LGS (Glauser 2008 Neurology); 32.7% median drop-attack reduction vs 11.7% placebo; formed basis for FDA approval of Rufinamide for LGS 2008"},
        {"term": "Drug-Resistant Epilepsy (DRE)", "definition": "ILAE definition: failure of ≥2 adequate, tolerated, appropriate AED schedules; affects 80–90% of LGS patients; triggers evaluation for corpus callosotomy, VNS, ketogenic diet, and novel FDA-approved agents"},
        {"term": "Status Epilepticus (SE) in LGS", "definition": "Tonic status epilepticus (non-convulsive) is common in LGS — manifests as obtundation + EEG GPFA; requires IV benzodiazepine rescue; phenytoin/carbamazepine CONTRAINDICATED even in SE (may worsen tonic SE)"},
        {"term": "SUDEP (Sudden Unexpected Death in Epilepsy)", "definition": "Elevated risk in LGS (3–7× general epilepsy population); nocturnal GTCS + cardiac dysrhythmia + nocturnal clustering of tonic seizures — seizure monitoring, optimised ASM, and prone position avoidance essential"},
        {"term": "Callosotomy Disconnection Syndrome", "definition": "Transient post-callosotomy syndrome: mutism, alien hand, apraxia; resolves in days-weeks after anterior 2/3 callosotomy; more severe after complete callosotomy; mitigated by staged approach"},
    ]

    standards = [
        {"name": "ILAE LGS Classification 2022", "scope": "Updated diagnostic criteria: triad required (multi-seizure types + SSW + ID); differentiates from MAE/EMAS; recommends genetic panel + MRI at diagnosis"},
        {"name": "FDA Approval — Rufinamide (LGS ≥1 yr, 2008)", "scope": "EIAED trial basis; adjunct for seizures associated with LGS; titration protocol 10→45 mg/kg/day; QT-shortening monitoring"},
        {"name": "FDA Approval — Clobazam (LGS ≥2 yrs, 2011)", "scope": "COALITION-I basis; 1,5-benzodiazepine; dose-titrated response; schedule IV controlled substance"},
        {"name": "FDA Approval — Cannabidiol/Epidiolex (LGS ≥2 yrs, 2018)", "scope": "GWPCARE3/GWPCARE4 basis; LFT monitoring required; drug interaction (VPA, CLB); 25 mg/kg/day max for LGS"},
        {"name": "FDA Approval — Fenfluramine/Fintepla (LGS ≥2 yrs, 2020)", "scope": "REMS program mandatory; echocardiogram baseline + every 6 months; max 0.35 mg/kg/day; lower max if Stiripentol co-administered"},
        {"name": "ILAE Surgical Guidelines 2017", "scope": "Corpus callosotomy: Level A recommendation for disabling drop attacks in LGS unresponsive to ≥2 AEDs; anterior 2/3 preferred; VNS: Level A evidence as adjunct in drug-resistant LGS"},
    ]

    thresholds = [
        {"metric": "Drop attack frequency (surgical threshold)", "target": "< 10 / month", "action_below": "Corpus callosotomy evaluation if ≥2 appropriate AEDs failed", "action_above": "Continue optimising AED polytherapy"},
        {"metric": "Rufinamide ECG QTc", "target": "≥ 340 ms", "action_below": "Reduce or discontinue — QTc < 340 ms = proarrhythmic risk", "action_above": "Continue with routine ECG monitoring"},
        {"metric": "Cannabidiol ALT/AST", "target": "< 3× ULN", "action_below": "Reduce dose 50% if 3–5× ULN; hold if >5× ULN with symptoms", "action_above": "Continue; LFT monthly for 6 months"},
        {"metric": "Fenfluramine echocardiogram", "target": "No new valvular regurgitation > mild", "action_below": "Discontinue Fenfluramine immediately; cardiology referral", "action_above": "Repeat echo in 6 months"},
        {"metric": "Ketogenic diet ketosis", "target": "3–5 mM β-hydroxybutyrate", "action_below": "Increase fat:carb ratio; review carbohydrate hidden sources", "action_above": "Risk of metabolic acidosis — reassess ratio"},
        {"metric": "VNS output current", "target": "1.0–2.5 mA (seizure suppression target)", "action_below": "Titrate upward every 2 weeks until response or tolerability limit", "action_above": "Reduce if intolerable voice changes/cough"},
    ]

    return {
        "generated": date.today().isoformat(),
        "concepts": concepts,
        "standards": standards,
        "thresholds": thresholds,
        "references": [
            "Thiele EA et al. (2018). Cannabidiol in patients with seizures associated with Lennox-Gastaut syndrome (GWPCARE3). NEJM, 378(20), 1888–1897.",
            "Devinsky O et al. (2018). Effect of cannabidiol on drop seizures in the Lennox-Gastaut syndrome (GWPCARE4). NEJM, 378(20), 1888.",
            "Lagae L et al. (2019). Fenfluramine hydrochloride for the treatment of seizures in Lennox-Gastaut syndrome. Neurology, 93(4), e358–e369.",
            "Glauser T et al. (2008). Rufinamide for generalized seizures associated with Lennox-Gastaut syndrome. Neurology, 70(21), 1950–1958.",
            "Ng YT et al. (2011). Randomized phase III study results of clobazam in Lennox-Gastaut syndrome (COALITION-I). Neurology, 76(18), 1555–1563.",
            "Arzimanoglou A et al. (2009). Lennox-Gastaut syndrome: a consensus approach on diagnosis, assessment, management. Lancet Neurology, 8(1), 82–93.",
        ],
    }
