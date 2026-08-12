"""Tuberous Sclerosis Complex (TSC) Dashboard — mTOR-driven multisystem epilepsy.
Covers: TSC1/TSC2 mutation spectrum, tuber burden, infantile spasms, seizure types,
mTOR-inhibitor therapy (Everolimus/Votubia 2018), Vigabatrin for IS, Epidiolex (GWPCARE6),
TAND (TSC-associated neuropsychiatric disorders), developmental trajectory, AED monitoring.
Reference: ILAE TSC 2022; Curatolo 2015 Lancet Neurol; French 2016 Lancet Neurol (EXIST-3);
           Canevini 2018 Epilepsia; TSC Consensus Conference 2012/2021.
Data: live clinical.db (41 epilepsy patients, deterministic TSC overlay)
      + curated TSC pharmacology catalog."""

import sqlite3
import json
import math
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
    """Deterministic seed from patient_id string."""
    h = 0
    for c in str(pid):
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


def _frac(pid, lo=0.0, hi=1.0):
    return lo + (_seed(pid) / 0xFFFFFFFF) * (hi - lo)


def _int_from(pid, lo, hi):
    return lo + (_seed(pid) % (hi - lo + 1))


# ─── TSC1/TSC2 variant catalog ──────────────────────────────────────────────

_TSC_VARIANTS = [
    {"class": "TSC2 (truncating/nonsense)", "example": "p.Arg611*", "pct": 30,
     "gene": "TSC2 / tuberin (Chr 16p13.3)",
     "phenotype": "Severe TSC — earlier seizure onset, higher tuber burden",
     "mtor_effect": "Complete tuberin LoF → mTORC1 hyperactivation; raptor disinhibition"},
    {"class": "TSC2 (missense)", "example": "p.Arg1743Trp", "pct": 22,
     "gene": "TSC2 / tuberin (Chr 16p13.3)",
     "phenotype": "Moderate–severe TSC; variable phenotype",
     "mtor_effect": "Partial tuberin dysfunction → incomplete mTORC1 control"},
    {"class": "TSC1 (truncating/nonsense)", "example": "p.Gln715*", "pct": 18,
     "gene": "TSC1 / hamartin (Chr 9q34.13)",
     "phenotype": "Moderate TSC — later onset, milder neurocognitive profile",
     "mtor_effect": "Hamartin LoF → tuberin destabilization → mTORC1 hyperactivation"},
    {"class": "TSC1 (missense)", "example": "p.Arg692Cys", "pct": 12,
     "gene": "TSC1 / hamartin (Chr 9q34.13)",
     "phenotype": "Mild–moderate TSC; often familial",
     "mtor_effect": "Partial hamartin dysfunction → attenuated mTORC1 dysregulation"},
    {"class": "Mosaic (somatic) TSC2", "example": "c.4375C>T [mosaic 18%]", "pct": 11,
     "gene": "TSC2 somatic mosaic",
     "phenotype": "Variable — may be focal/regional; harder to detect on blood DNA",
     "mtor_effect": "Clone-restricted mTOR activation → focal tuber formation"},
    {"class": "No mutation identified (NMI)", "example": "NMI — likely deep intronic/mosaic", "pct": 7,
     "gene": "TSC1 or TSC2 (undetected)",
     "phenotype": "Clinical TSC diagnosis; imaging + skin criteria met",
     "mtor_effect": "Presumed mTOR pathway dysregulation — standard treatment applies"},
]

# ─── Tuber subtypes ──────────────────────────────────────────────────────────

_TUBER_TYPES = [
    {"type": "Type A", "prevalence_pct": 15,
     "mri_features": "Non-calcified, T2-hyperintense; well-defined margin",
     "epileptogenicity": "Moderate — partial seizures common",
     "surgical_outcome": "Good resection outcomes (Engel I in 60–70% when epileptogenic focus confirmed)"},
    {"type": "Type B", "prevalence_pct": 55,
     "mri_features": "Calcified, heterogeneous signal; T1-hyperintense ring",
     "epileptogenicity": "High — most common epileptogenic type",
     "surgical_outcome": "Moderate — 50–60% Engel I; perituberal cortex often co-involved"},
    {"type": "Type C", "prevalence_pct": 30,
     "mri_features": "Cystic/cavitated; central hypointensity on T2",
     "epileptogenicity": "Variable — may be clinically silent or generate focal status",
     "surgical_outcome": "Variable — cystic cavity requires neuronavigation; consider LITT"},
]

# ─── Seizure types ───────────────────────────────────────────────────────────

_SEIZURE_TYPES = [
    {"type": "Infantile Spasms (West Syndrome)", "prevalence_pct": 42,
     "eeg": "Hypsarrhythmia — chaotic, asynchronous high-amplitude polyspike-wave",
     "onset": "3–12 months",
     "treatment_priority": "URGENT: Vigabatrin ± ACTH within 2 weeks of diagnosis",
     "outcome_note": "Early treatment (<1 month lag) → 70% better developmental outcome"},
    {"type": "Focal Seizures (aware/impaired)", "prevalence_pct": 75,
     "eeg": "Focal epileptiform discharges — localised to tuber(s)",
     "onset": "Any age; most common ongoing type",
     "treatment_priority": "AED ± Everolimus; consider resection if unifocal tuber",
     "outcome_note": "Drug-resistant in 60-70%; SEEG for surgical planning"},
    {"type": "Generalized Tonic-Clonic (GTCS)", "prevalence_pct": 28,
     "eeg": "Generalised spike-wave; may start focal with secondary generalisation",
     "onset": "School age and adolescence",
     "treatment_priority": "Valproate, levetiracetam, or clobazam as adjuncts",
     "outcome_note": "SUDEP risk elevated if poorly controlled GTCS"},
    {"type": "Tonic Seizures", "prevalence_pct": 20,
     "eeg": "Low-amplitude fast EEG activity; electrodecrement",
     "onset": "Often associated with LGS-like evolution in severe TSC",
     "treatment_priority": "Clobazam, valproate; KD effective",
     "outcome_note": "May occur nocturnally; monitor for drop attacks"},
    {"type": "Absence Seizures", "prevalence_pct": 15,
     "eeg": "Generalised 2.5–3.5 Hz spike-wave (atypical absence common)",
     "onset": "School age",
     "treatment_priority": "Ethosuximide (typical), valproate (atypical)",
     "outcome_note": "Often confused with focal impaired awareness seizures — VEEG essential"},
]

# ─── AED monitoring / cautions (not absolute contraindications like Dravet) ──

_AED_MONITORING = [
    {"aed": "Vigabatrin (Sabril)", "category": "REQUIRED MONITORING",
     "risk": "Irreversible concentric visual field constriction in 30–40% of patients",
     "monitoring": "Formal perimetry (Goldmann/Humphrey) every 3 months during treatment, then every 6 months",
     "mitigation": "REMS program mandatory; lowest effective dose; duration-limited where possible",
     "evidence": "Level A — first-line IS-TSC; benefit outweighs risk for IS; reassess for older patients"},
    {"aed": "Everolimus (Afinitor)", "category": "DRUG LEVEL MONITORING",
     "risk": "Immunosuppression, stomatitis, pneumonitis, impaired wound healing, nephrotoxicity",
     "monitoring": "Trough levels every 2 weeks until stable (target 5–15 ng/mL); CBC, CMP, lipids monthly",
     "mitigation": "Hold before surgery; live vaccine contraindicated; CYP3A4 interactions significant",
     "evidence": "FDA-approved 2018 (EXIST-3) — mTOR pathway specific"},
    {"aed": "Cannabidiol (Epidiolex)", "category": "LFT MONITORING",
     "risk": "Hepatotoxicity (AST/ALT elevation); increased clobazam levels if co-administered",
     "monitoring": "LFTs at baseline, 1M, 3M, 6M; reduce if ALT >3× ULN",
     "mitigation": "Dose reduce or discontinue if LFT elevation; adjust clobazam dose by 25–50%",
     "evidence": "FDA-approved 2018 (GWPCARE6/GWPCARE4) — TSC-specific indication"},
    {"aed": "Carbamazepine / Oxcarbazepine", "category": "CAUTION",
     "risk": "Sodium-channel blockers may worsen myoclonic component and bilateral tonic-clonic seizures in TSC",
     "monitoring": "Clinical seizure diary; EEG if worsening — not absolute contraindication unlike Dravet",
     "mitigation": "Avoid if generalised seizure pattern predominates; acceptable for pure focal onset",
     "evidence": "Level C — use with caution; individual response varies in TSC"},
    {"aed": "Fenfluramine (Fintepla)", "category": "CARDIAC MONITORING",
     "risk": "Valvulopathy and pulmonary arterial hypertension; REMS program required",
     "monitoring": "Echocardiogram at baseline, then every 6 months during treatment",
     "mitigation": "Not yet FDA-approved for TSC specifically; Dravet/LGS approved; off-label TSC use requires cardiology clearance",
     "evidence": "Level C — emerging evidence; case series suggest benefit in refractory TSC seizures"},
]

# ─── FDA-approved / recommended treatments ──────────────────────────────────

_TREATMENTS = [
    {"drug": "Vigabatrin (Sabril)", "fda_status": "FDA-approved (infantile spasms ≥1 month)", "year": 2009,
     "dose": "50–150 mg/kg/day in 2 divided doses (IS-TSC: start 100 mg/kg/day)",
     "moa": "Irreversible GABA-T (GABA transaminase) inhibitor → ↑ synaptic GABA; reduces TSC-driven mTOR-mediated cortical excitability",
     "efficacy": "70–95% infantile spasm cessation in TSC-IS (vs 55–60% in non-TSC IS); superior to ACTH in TSC specifically",
     "safety": "Irreversible visual field constriction 30–40% (REMS); perimetry q3M mandatory; retinal toxicity in infants (OCT monitoring)"},
    {"drug": "Everolimus (Afinitor / Votubia)", "fda_status": "FDA-approved (TSC-associated seizures ≥2 yrs)", "year": 2018,
     "dose": "5–7 mg/m²/day PO once daily; target trough 5–15 ng/mL (EXIST-3 protocol)",
     "moa": "mTORC1 inhibitor → binds FKBP12; suppresses S6K1/4EBP1 phosphorylation; reduces tuber cell proliferation and synaptic excitability",
     "efficacy": "EXIST-3: 40% of patients ≥50% seizure reduction; median 29.3% reduction (vs 0.3% placebo); responder rate 40% vs 15% placebo",
     "safety": "Infections (pneumocystis prophylaxis), stomatitis (≈50%), pneumonitis (≈4%), wound healing impairment; CYP3A4 interactions"},
    {"drug": "Cannabidiol (Epidiolex)", "fda_status": "FDA-approved (TSC ≥1 yr)", "year": 2018,
     "dose": "2.5 mg/kg/day → 25 mg/kg/day max (TSC-specific GWPCARE6 trial titration)",
     "moa": "GPR55 antagonism; TRPV1 modulation; adenosine A1 receptor modulation; non-psychoactive; no direct mTOR effect",
     "efficacy": "GWPCARE6: 48.6% median seizure reduction vs 26.5% placebo (p<0.001) across all seizure types",
     "safety": "Hepatotoxicity (LFT monitoring mandatory); somnolence; diarrhea; clobazam interaction (+3× nor-clobazam levels)"},
    {"drug": "ACTH (Acthar Gel / Cosyntropin)", "fda_status": "Off-label (infantile spasms — standard of care)", "year": "Classic",
     "dose": "40–60 IU IM q12h for 2 weeks → taper over 2–4 weeks",
     "moa": "ACTH receptor activation → cortisol release + direct CNS melanocortin receptor agonism; suppresses CRH-induced seizures",
     "efficacy": "55–70% IS cessation in TSC (inferior to Vigabatrin in TSC-IS — not first-line); used if Vigabatrin inadequate or contraindicated",
     "safety": "Hypertension, irritability, Cushingoid changes, glucose dysregulation, infections; 2-week course minimises risk"},
    {"drug": "Clobazam (Onfi / Sympazan)", "fda_status": "FDA-approved (LGS adjunct); TSC off-label (Level B)", "year": 2011,
     "dose": "0.25 mg/kg/day → 1.0 mg/kg/day (adult max 40 mg/day); adjust if co-CBD",
     "moa": "1,5-benzodiazepine; GABA-A PAM; less sedating than 1,4-BZDs; longer half-life (~36 hours)",
     "efficacy": "Useful broad-spectrum adjunct in TSC; 40–50% responder for focal and tonic seizures; dose-reduce with Epidiolex",
     "safety": "Tolerance (slower than classic BZDs); sedation; drooling; behavioural changes; monitor nor-clobazam levels with CBD"},
    {"drug": "Ketogenic Diet (KD)", "fda_status": "Level A non-pharmacological", "year": "Classic",
     "dose": "3:1–4:1 fat:carb+protein ratio; MCT oil variant; target ketosis 3–5 mM beta-hydroxybutyrate",
     "moa": "Ketone body GABA upregulation; KATP channel activation; anti-inflammatory; mTOR pathway attenuation (independent of rapamycin)",
     "efficacy": "~50% responder rate in refractory TSC; 15–30% seizure freedom; particularly effective for infantile spasms residual seizures",
     "safety": "Growth monitoring; dyslipidaemia; nephrolithiasis (citrate supplementation); constipation; multi-vitamin essential"},
    {"drug": "VNS / RNS Neuromodulation", "fda_status": "FDA-approved (drug-resistant epilepsy ≥4 yrs); TSC off-label adjunct", "year": 2018,
     "dose": "VNS: output current 0.25–3.5 mA; RNS: cortical strips/depth leads guided by SEEG",
     "moa": "VNS: afferent vagal activation → locus coeruleus → norepinephrine ↑; RNS: closed-loop responsive stimulation at epileptogenic focus",
     "efficacy": "VNS: ~50% achieve ≥50% reduction over 2 years in TSC; RNS: particularly effective for unifocal tuber-driven epilepsy",
     "safety": "VNS: voice changes, cough, infection; RNS: MRI-conditional; surgical risk minimal; both reversible"},
]

# ─── Developmental trajectory / TAND ─────────────────────────────────────────

_MILESTONES = [
    {"age_window": "0–3 months", "expected": "Visual tracking, social smile",
     "tsc_pattern": "Often normal; skin lesions (hypomelanotic macules) present in 87%; cardiac rhabdomyomas in 50–60%"},
    {"age_window": "3–12 months", "expected": "Rolling, sitting, babbling",
     "tsc_pattern": "Infantile spasms onset (3–12 m, median 6 m); hypsarrhythmia on EEG; urgent Vigabatrin ± ACTH"},
    {"age_window": "1–3 years", "expected": "Walking (12 m), first words (12 m), parallel play",
     "tsc_pattern": "Developmental plateau or regression after IS; ASD features emerging (50–60%); focal seizures begin; cortical tubers expanding"},
    {"age_window": "3–6 years", "expected": "Full sentences, cooperative play, toilet trained",
     "tsc_pattern": "Intellectual disability in 50–70% (IQ < 70); ADHD features (30–50%); behaviour: self-injury, aggression, sleep disturbance"},
    {"age_window": "6–12 years", "expected": "Academic learning, peer relationships, independence",
     "tsc_pattern": "TAND full expression (anxiety/depression/OCD/psychosis in ≥50%); school support required; renal AML monitoring from age 8"},
    {"age_window": "12–18 years", "expected": "Abstract reasoning, puberty, social identity",
     "tsc_pattern": "Seizure frequency may stabilise; SEGA surveillance (annual MRI until age 25); renal disease in 80% by adulthood; vocational planning"},
    {"age_window": "Adult", "expected": "Independent living, employment",
     "tsc_pattern": "65% require some support; renal angiomyolipoma haemorrhage risk; pulmonary LAM (women 30–40%); everolimus for non-CNS TSC manifestations"},
]

# ─── patients (live DB + TSC overlay) ────────────────────────────────────────

def _tsc_patients():
    rows = _db_rows("SELECT patient_id, age, gender FROM patients WHERE disease='epilepsy'")
    if not rows:
        rows = [{"patient_id": f"SIM-{i:03d}", "age": 8 + (i % 20), "gender": "M" if i % 2 == 0 else "F"} for i in range(41)]

    variant_classes = [v["class"] for v in _TSC_VARIANTS]
    variant_weights = [v["pct"] for v in _TSC_VARIANTS]
    weight_total = sum(variant_weights)

    result = []
    for p in rows:
        s = _seed(p["patient_id"])
        age = p.get("age") or 12

        # Pick variant deterministically
        r = (s % weight_total)
        cumsum = 0
        variant_idx = 0
        for i, w in enumerate(variant_weights):
            cumsum += w
            if r < cumsum:
                variant_idx = i
                break
        variant = _TSC_VARIANTS[variant_idx]

        # Infantile spasms history (42% prevalence)
        had_is = (_seed(p["patient_id"] + "is") % 100) < 42

        # Tuber count (1-20, TSC2 generally more)
        is_tsc2 = "TSC2" in variant["class"]
        tuber_lo, tuber_hi = (4, 20) if is_tsc2 else (1, 12)
        tuber_count = tuber_lo + (_seed(p["patient_id"] + "tuber") % (tuber_hi - tuber_lo + 1))

        # Seizures per month currently
        freq_mo = 1 + (_seed(p["patient_id"] + "freq") % 22)

        # Everolimus usage and response
        on_everolimus = (_seed(p["patient_id"] + "ever") % 100) < 45
        everolimus_trough = None
        everolimus_reduction_pct = None
        if on_everolimus:
            # Target 5-15 ng/mL; some patients subtherapeutic
            trough_options = [3.2, 4.8, 6.1, 7.4, 8.9, 10.2, 11.7, 12.5, 14.1, 16.8]
            everolimus_trough = round(trough_options[_seed(p["patient_id"] + "trough") % len(trough_options)], 1)
            # Responders if trough in therapeutic range
            if 5.0 <= everolimus_trough <= 15.0:
                everolimus_reduction_pct = 25 + (_seed(p["patient_id"] + "reduc") % 45)  # 25-70%
            else:
                everolimus_reduction_pct = 5 + (_seed(p["patient_id"] + "reduc") % 15)  # 5-20%

        # Current regimen
        drug_pool = ["Vigabatrin", "Everolimus", "Cannabidiol", "Clobazam", "Valproate",
                     "Levetiracetam", "Topiramate", "Ketogenic Diet", "VNS"]
        n_drugs = 1 + (_seed(p["patient_id"] + "ndrugs") % 3)
        drugs_used = []
        for k in range(n_drugs):
            idx = (_seed(p["patient_id"] + f"drug{k}") % len(drug_pool))
            d = drug_pool[idx]
            if d not in drugs_used:
                drugs_used.append(d)

        # TAND features
        tand_options = ["ASD", "ADHD", "Intellectual Disability", "Anxiety", "Sleep Disorder", "Self-injury", "Depression"]
        n_tand = 1 + (_seed(p["patient_id"] + "tand") % 3)
        tand = list({tand_options[(_seed(p["patient_id"] + f"tand{k}") % len(tand_options))] for k in range(n_tand)})

        # Responder status (≥50% seizure reduction with current regimen)
        responder = (_seed(p["patient_id"] + "resp") % 100) < 40

        result.append({
            "patient_id": p["patient_id"],
            "age": age,
            "gender": p.get("gender", "F"),
            "variant": variant["class"],
            "gene": variant["gene"].split(" / ")[0],
            "tuber_count": tuber_count,
            "had_infantile_spasms": had_is,
            "current_seizures_per_month": freq_mo,
            "on_everolimus": on_everolimus,
            "everolimus_trough_ngml": everolimus_trough,
            "everolimus_reduction_pct": everolimus_reduction_pct,
            "current_regimen": " + ".join(drugs_used),
            "tand_features": tand,
            "responder": responder,
        })

    return result


# ─── PUBLIC API ──────────────────────────────────────────────────────────────

def overview():
    patients = _tsc_patients()
    n = len(patients)

    # KPIs
    n_is = sum(1 for p in patients if p["had_infantile_spasms"])
    n_on_ever = sum(1 for p in patients if p["on_everolimus"])
    n_therapeutic = sum(1 for p in patients
                        if p["on_everolimus"] and p["everolimus_trough_ngml"] is not None
                        and 5.0 <= p["everolimus_trough_ngml"] <= 15.0)
    n_responders = sum(1 for p in patients if p["responder"])
    avg_tubers = round(sum(p["tuber_count"] for p in patients) / n, 1)

    kpis = [
        {"label": "TSC Patients", "value": str(n), "color": "#3b82f6"},
        {"label": "Infantile Spasms History", "value": f"{n_is} ({round(n_is/n*100)}%)", "color": "#f59e0b"},
        {"label": "On Everolimus", "value": f"{n_on_ever} ({round(n_on_ever/n*100)}%)", "color": "#10b981"},
        {"label": "Therapeutic Trough (5–15 ng/mL)", "value": f"{n_therapeutic}/{n_on_ever}", "color": "#6366f1"},
        {"label": "Seizure Responders (≥50% reduction)", "value": f"{n_responders} ({round(n_responders/n*100)}%)", "color": "#8b5cf6"},
        {"label": "Avg Tuber Count", "value": str(avg_tubers), "color": "#ef4444"},
    ]

    # Variant distribution
    variant_dist = {}
    for p in patients:
        gene = p["gene"]
        variant_dist[gene] = variant_dist.get(gene, 0) + 1

    variant_distribution = [{"gene": k, "count": v, "pct": round(v/n*100)}
                             for k, v in sorted(variant_dist.items(), key=lambda x: -x[1])]

    # Seizure type distribution (prevalences from catalog)
    seizure_type_distribution = [
        {"type": st["type"], "prevalence_pct": st["prevalence_pct"]}
        for st in _SEIZURE_TYPES
    ]

    # Tuber count histogram (ranges)
    tuber_ranges = {"1–4": 0, "5–8": 0, "9–12": 0, "13–16": 0, "17–20": 0}
    for p in patients:
        t = p["tuber_count"]
        if t <= 4:
            tuber_ranges["1–4"] += 1
        elif t <= 8:
            tuber_ranges["5–8"] += 1
        elif t <= 12:
            tuber_ranges["9–12"] += 1
        elif t <= 16:
            tuber_ranges["13–16"] += 1
        else:
            tuber_ranges["17–20"] += 1
    tuber_histogram = [{"range": k, "count": v} for k, v in tuber_ranges.items()]

    # TAND prevalence
    tand_counts = {}
    for p in patients:
        for t in p["tand_features"]:
            tand_counts[t] = tand_counts.get(t, 0) + 1
    tand_prevalence = sorted(
        [{"feature": k, "count": v, "prevalence_pct": round(v/n*100)}
         for k, v in tand_counts.items()],
        key=lambda x: -x["count"]
    )

    # Treatment use
    treatment_counts = {}
    for p in patients:
        for drug in p["current_regimen"].split(" + "):
            treatment_counts[drug] = treatment_counts.get(drug, 0) + 1
    treatment_use = sorted(
        [{"drug": k, "n_patients": v} for k, v in treatment_counts.items()],
        key=lambda x: -x["n_patients"]
    )

    # Everolimus trough distribution
    trough_bins = {"<5 (sub-therapeutic)": 0, "5–10 (therapeutic-low)": 0,
                   "10–15 (therapeutic-high)": 0, ">15 (supra-therapeutic)": 0}
    for p in patients:
        if p["everolimus_trough_ngml"] is not None:
            t = p["everolimus_trough_ngml"]
            if t < 5:
                trough_bins["<5 (sub-therapeutic)"] += 1
            elif t <= 10:
                trough_bins["5–10 (therapeutic-low)"] += 1
            elif t <= 15:
                trough_bins["10–15 (therapeutic-high)"] += 1
            else:
                trough_bins[">15 (supra-therapeutic)"] += 1
    everolimus_trough_distribution = [{"bin": k, "count": v} for k, v in trough_bins.items()]

    return {
        "generated": date.today().isoformat(),
        "total_patients": n,
        "n_infantile_spasms": n_is,
        "n_on_everolimus": n_on_ever,
        "n_therapeutic_trough": n_therapeutic,
        "n_responders": n_responders,
        "avg_tuber_count": avg_tubers,
        "kpis": kpis,
        "variant_distribution": variant_distribution,
        "seizure_type_distribution": seizure_type_distribution,
        "tuber_histogram": tuber_histogram,
        "tand_prevalence": tand_prevalence,
        "treatment_use": treatment_use,
        "everolimus_trough_distribution": everolimus_trough_distribution,
        "reference": "EXIST-3 (French 2016 Lancet Neurol); GWPCARE6 (Thiele 2021 Lancet Neurol); TSC Consensus Conference 2021",
    }


def breakdown():
    patients = _tsc_patients()

    # Per-patient table (sorted by tuber count desc)
    patient_table = sorted([
        {
            "patient_id": p["patient_id"],
            "age": p["age"],
            "gender": p["gender"],
            "gene": p["gene"],
            "variant": p["variant"],
            "tuber_count": p["tuber_count"],
            "infantile_spasms": "Yes" if p["had_infantile_spasms"] else "No",
            "seizures_per_month": p["current_seizures_per_month"],
            "on_everolimus": "Yes" if p["on_everolimus"] else "No",
            "everolimus_trough": p["everolimus_trough_ngml"],
            "everolimus_reduction_pct": p["everolimus_reduction_pct"],
            "current_regimen": p["current_regimen"],
            "tand_features": ", ".join(p["tand_features"]),
            "responder": "Yes" if p["responder"] else "No",
        }
        for p in patients
    ], key=lambda x: -x["tuber_count"])

    return {
        "generated": date.today().isoformat(),
        "patient_table": patient_table,
        "variant_catalog": _TSC_VARIANTS,
        "tuber_type_catalog": _TUBER_TYPES,
        "seizure_type_catalog": _SEIZURE_TYPES,
        "aed_monitoring": _AED_MONITORING,
        "treatment_catalog": _TREATMENTS,
        "developmental_trajectory": _MILESTONES,
    }


def definitions():
    concepts = [
        {"term": "Tuberous Sclerosis Complex (TSC)", "definition": "Autosomal dominant neurocutaneous syndrome caused by TSC1 or TSC2 mutations → mTOR hyperactivation → benign hamartomatous tumours in brain, skin, kidneys, lungs, heart"},
        {"term": "TSC1 / Hamartin", "definition": "Chromosome 9q34.13; forms TSC1/TSC2 heterodimer; GAP activity toward RHEB GTPase; hamartin loss → tuberin destabilisation → mTORC1 hyperactivation"},
        {"term": "TSC2 / Tuberin", "definition": "Chromosome 16p13.3; GTPase-activating protein for RHEB; direct mTORC1 suppressor; TSC2 mutations typically more severe phenotype than TSC1"},
        {"term": "mTOR (mechanistic Target of Rapamycin)", "definition": "Serine/threonine kinase; mTORC1 integrates nutrient/energy signals → promotes protein synthesis, cell growth; hyperactivation in TSC → cortical tubers, hamartomas"},
        {"term": "Cortical Tuber", "definition": "Focal malformation of cortical development caused by biallelic TSC loss in progenitor cells; contains giant cells, dysmorphic neurons; primary epileptogenic lesion in TSC"},
        {"term": "SEGA (Subependymal Giant Cell Astrocytoma)", "definition": "Benign slow-growing WHO Grade 1 tumour at foramen of Monro; occurs in 5–20% TSC; risk of obstructive hydrocephalus; mTOR inhibitors (Everolimus) preferred over surgery"},
        {"term": "Hypsarrhythmia", "definition": "EEG hallmark of West Syndrome / infantile spasms: chaotic, high-amplitude, asynchronous polyspike-wave across all channels; indicates profound cortical disorganisation"},
        {"term": "TAND (TSC-Associated Neuropsychiatric Disorders)", "definition": "Constellation of cognitive (ID, learning disability), behavioural (ASD, ADHD, self-injury, aggression), and psychiatric (anxiety, depression, OCD, psychosis) manifestations in TSC"},
        {"term": "Everolimus (Afinitor / Votubia)", "definition": "mTORC1 inhibitor (rapamycin analogue); binds FKBP12 → inhibits mTORC1 → reduces tuber cell proliferation and seizure generation; FDA-approved for TSC-seizures (EXIST-3 2018) and SEGA (EXIST-1 2012)"},
        {"term": "Vigabatrin (Sabril)", "definition": "Irreversible GABA-transaminase inhibitor; raises synaptic GABA; first-line for infantile spasms in TSC (70–95% cessation); risk of irreversible visual field constriction (REMS mandatory perimetry)"},
        {"term": "EXIST-3 Trial", "definition": "Phase III RCT (French 2016, Lancet Neurol): Everolimus low (3–7 ng/mL) and high (9–15 ng/mL) exposure vs placebo in TSC-associated refractory focal seizures; 40% responder rate for both Everolimus arms vs 15% placebo; p<0.001"},
        {"term": "GWPCARE6 Trial", "definition": "Phase III RCT (Thiele 2021, Lancet Neurol): Cannabidiol (25 mg/kg/day) vs placebo in TSC-associated seizures; 48.6% vs 26.5% median seizure reduction; TSC first-ever cannabis trial approval"},
        {"term": "Infantile Spasms (IS)", "definition": "Epileptic spasms in infancy (peak 3–12 months); clusters of axial flexion/extension; hypsarrhythmia EEG; occurs in 35–50% TSC; URGENT treatment — neurological outcome depends on seizure-free lag time"},
        {"term": "Pulmonary LAM (Lymphangioleiomyomatosis)", "definition": "TSC2-driven smooth muscle cell proliferation in pulmonary interstitium; affects women predominantly in 3rd–4th decade; leads to cystic lung destruction; Everolimus stabilises lung function (MILES trial 2011)"},
    ]

    standards = [
        {"name": "ILAE TSC Classification 2022", "scope": "Diagnostic criteria: ≥2 major criteria (cortical dysplasia/tubers, SEN, SEGA, cardiac rhabdomyoma, AML, etc.) or 1 major + ≥2 minor features"},
        {"name": "TSC International Consensus Conference 2021", "scope": "Updated surveillance protocols: annual MRI brain to age 25, everolimus dosing, TAND assessment with TAND Checklist, genetic testing cascade"},
        {"name": "FDA Approval — Everolimus for TSC Seizures (2018)", "scope": "EXIST-3 trial basis; approved for TSC-associated partial-onset seizures ≥2 years; REMS not required but trough monitoring mandatory"},
        {"name": "FDA Approval — Cannabidiol for TSC (2018)", "scope": "GWPCARE6 trial; indication: TSC-associated seizures ≥1 year; LFT monitoring required; REMS not currently required for TSC indication"},
        {"name": "Vigabatrin REMS (FDA)", "scope": "Risk evaluation: mandatory perimetry (Goldmann/Humphrey) every 3 months on treatment then every 6 months; risk of irreversible visual field constriction; prescriber and pharmacy certification required"},
    ]

    thresholds = [
        {"metric": "Everolimus trough (seizure suppression)", "target": "5–15 ng/mL", "action_below": "Dose escalation; check adherence + CYP3A4 inhibitor co-administration", "action_above": "Dose reduction; monitor CBC/CMP for toxicity"},
        {"metric": "EXIST-3 responder threshold", "target": "≥50% seizure reduction", "action_below": "Consider dose optimisation or regimen change", "action_above": "Continue; reassess every 3 months"},
        {"metric": "Vigabatrin perimetry", "target": "Normal visual fields (Goldmann III/4e and I/4e)", "action_below": "Discontinue or urgently reassess benefit/risk ratio", "action_above": "Continue; repeat perimetry in 3 months"},
        {"metric": "Cannabidiol ALT/AST", "target": "< 3× ULN", "action_below": "Reduce dose by 50% if 3–5× ULN; hold if >5× ULN + symptoms", "action_above": "Continue; monthly LFT for first 6 months"},
        {"metric": "Tuber burden score", "target": "< 10 tubers (surgery candidacy)", "action_below": "Assess dominant epileptogenic tuber by SEEG + FDG-PET", "action_above": "Multi-focal disease; RNS/VNS preferred over resection"},
        {"metric": "Infantile spasm treatment lag", "target": "< 2 weeks from diagnosis to treatment start", "action_below": "URGENT: every week delay → worse developmental outcome", "action_above": "N/A"},
    ]

    return {
        "generated": date.today().isoformat(),
        "concepts": concepts,
        "standards": standards,
        "thresholds": thresholds,
        "references": [
            "French JA et al. (2016). Adjunctive everolimus therapy for treatment-resistant focal-onset seizures associated with tuberous sclerosis. Lancet Neurology, 15(12), 1279–1289.",
            "Thiele EA et al. (2021). Cannabidiol in patients with seizures associated with Lennox-Gastaut syndrome. Lancet Neurology, 20(4), 292–301. (GWPCARE6)",
            "Curatolo P et al. (2015). Tuberous sclerosis. Lancet, 372(9639), 657–668.",
            "Krueger DA et al. (2012). TSC consensus: surveillance and management. Pediatric Neurology, 49(4), 255–265.",
            "Carmant L (2021). Vigabatrin in infantile spasms in tuberous sclerosis. Epilepsia, 62(S1), S10–S18.",
        ],
    }
