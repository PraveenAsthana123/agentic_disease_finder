#!/usr/bin/env python3
"""CACT Deficiency (Carnitine-Acylcarnitine Translocase Deficiency / SLC25A20) Dashboard.

SLC25A20 gene encodes CACT (Carnitine-Acylcarnitine Translocase):
  CACT: 301 aa; 3p21.31; inner mitochondrial membrane (IMM); antiporter
  Locus: 3p21.31; Autosomal Recessive (AR)
  OMIM Gene: *613698; OMIM Disease: #212138

  CACT CATALYSES (inner mitochondrial membrane — CARNITINE SHUTTLE STEP 2):
    Long-chain acylcarnitine (intermembrane space → matrix) in exchange for
    free carnitine (matrix → intermembrane space) — obligate ANTIPORT mechanism.
    Without CACT → long-chain acylcarnitines CANNOT enter the mitochondrial matrix
    → long-chain beta-oxidation BLOCKED → hypoketotic hypoglycaemia.

CARNITINE SHUTTLE — ALL THREE STEPS:
  Step 1 (CPT1A — outer IMM, cytosolic face):
    Long-chain acyl-CoA + L-carnitine → long-chain acylcarnitine + CoA-SH
    [Rate-limiting step; liver CPT1A; CPT1B for heart/muscle]
  Step 2 (CACT / SLC25A20 — inner IMM translocase):
    Long-chain acylcarnitine (intermembrane space) IN → mitochondrial matrix
    Free carnitine (matrix) OUT → intermembrane space   [ANTIPORT]
    [← CACT BLOCK HERE → acylcarnitines ACCUMULATE in cytoplasm/blood]
  Step 3 (CPT2 — inner IMM, matrix face):
    Long-chain acylcarnitine + CoA-SH → long-chain acyl-CoA + free carnitine
    [Regenerates acyl-CoA for beta-oxidation; free carnitine recycled via CACT]
  Steps 4–7 (matrix): Beta-oxidation proper (VLCAD → HADHA → HADHB → ACAT1 for long chain)

CACT METABOLIC BLOCK:
  CACT LOF → long-chain acylcarnitines CANNOT cross inner mitochondrial membrane
  → accumulate in intermembrane space and cytoplasm → spill into plasma
  → C16, C18:1, C18 ELEVATED in blood (trapped; cannot enter matrix)
  → free carnitine (C0) LOW (<15 μmol/L; carnitine trapped as acylcarnitines)
  → beta-oxidation BLOCKED → ketone synthesis FAILS → HYPOKETOTIC HYPOGLYCAEMIA
  → acyl-CoA accumulation in cytoplasm → inhibits CPS1/NAGS → HYPERAMMONEMIA
  → energy crisis → cardiac arrhythmia + cardiomyopathy + rhabdomyolysis

KEY NBS MARKER — CACT PROFILE (HIGH C16 + C18:1 + C18; LOW C0):
  C16 (palmitoylcarnitine)  ELEVATED ≥1.5 μmol/L  — PRIMARY NBS MARKER
  C18:1 (oleoylcarnitine)   ELEVATED               — co-elevated
  C18 (stearoylcarnitine)   ELEVATED               — co-elevated
  C14:1 (tetradecenoylcarnitine) Mildly elevated (less than VLCAD)
  C0 (free carnitine)       LOW <15 μmol/L         — carnitine trapped as acylcarnitines
  C16/(C0+C2) ratio         ELEVATED               — supporting ratio
  C16-OH                    NORMAL                 — KEY NEGATIVE vs LCHAD
  C8 (octanoylcarnitine)    NORMAL                 — KEY NEGATIVE vs MCAD
  C3 (propionylcarnitine)   NORMAL                 — KEY NEGATIVE vs PA/MMA
  → OPPOSITE C0 direction from CPT1A (CPT1A has HIGH C0; CACT has LOW C0)

CLINICAL FEATURES:
  1. CARDIOMYOPATHY (dilated/hypertrophic) — HALLMARK; life-threatening; unlike CPT1A
  2. ARRHYTHMIAS (ventricular; potentially fatal) — urgent cardiac monitoring
  3. HYPOKETOTIC HYPOGLYCAEMIA — fasting/illness trigger
  4. HYPERAMMONEMIA — 50–500 μmol/L (acyl-CoA accumulation → CPS1/NAGS inhibition)
  5. HEPATOMEGALY + transaminitis
  6. RHABDOMYOLYSIS — present in ALL phenotypes (KEY EXAM TRAP vs CPT1A where ABSENT)
  7. Hypotonia/weakness
  8. HIGH NEONATAL MORTALITY without early treatment (75% severe neonatal onset)

KEY NEGATIVES (exam discriminators):
  - C16-OH NORMAL — KEY NEGATIVE vs LCHAD (LCHAD elevates C16-OH; has retinopathy + neuropathy)
  - NO retinopathy — KEY NEGATIVE vs LCHAD
  - NO peripheral neuropathy — KEY NEGATIVE vs LCHAD
  - No maternal AFLP/HELLP — KEY NEGATIVE vs LCHAD
  - C8 NORMAL — KEY NEGATIVE vs MCAD
  - C0 LOW (not high) — KEY NEGATIVE vs CPT1A (CPT1A has C0 ELEVATED — INVERTED PROFILE)
  - C14:1 less elevated than VLCAD; C18 more elevated than VLCAD

TREATMENT:
  1. MCT OIL — Level A THERAPEUTIC: medium-chain FA (C8/C10) bypass CACT entirely
     → enter mitochondria directly via MCT1 transporter; oxidised by MCAD/SCAD
  2. L-CARNITINE — Level A ESSENTIAL: C0 LOW → supplementation mandatory
     (OPPOSITE of CPT1A where L-carnitine is NOT routinely given)
  3. FASTING — ABSOLUTE CI (primary trigger)
  4. KD — ABSOLUTE CI (long-chain fat floods blocked CACT)
  5. VPA — ABSOLUTE CI (inhibits FAO + depletes carnitine; cardiomyopathy risk)
  6. IV Glucose 10% + IV L-carnitine — Level A emergency
  7. LONG-CHAIN FAT RESTRICTION — Level A
  8. Triheptanoin (C7 odd-chain) — Level B (anaplerotic TCA entry)
  9. Cardiac medications — digoxin / ACE inhibitors for cardiomyopathy management
  10. Emergency cardiology — antiarrhythmics if ventricular arrhythmia

GENETICS — KEY VARIANTS:
  c.199-10T>G (IVS2-10T>G) — SPLICING; severe; Yemenite Jewish founder; neonatal form
  p.Arg178Gln (c.533G>A)    — Severe; catalytic core; complete LOF
  p.Lys32Glu                — Severe; N-terminal matrix loop
  p.Gly80Glu                — Moderate; some residual transport function
  p.Phe545Cys               — Mild; adult myopathic form; some residual activity
  p.Arg275Cys               — Moderate-severe; transmembrane domain

COMPARISON TABLE (exam discriminators):
  CACT vs CPT1A: C16 HIGH + C0 LOW (CACT) vs C0 HIGH + C16 NORMAL (CPT1A);
    CACT has cardiomyopathy + rhabdomyolysis; CPT1A has NEITHER; different shuttle step
  CACT vs VLCAD: CACT has C16+C18+C18:1 all elevated + C0 LOW;
    VLCAD has C14:1 as primary + C16 elevated but C18 less so; both have cardiomyopathy
  CACT vs LCHAD: CACT has C16-OH NORMAL; LCHAD has C16-OH ELEVATED;
    LCHAD has retinopathy + neuropathy + maternal AFLP/HELLP; CACT does NOT
  CACT vs CPT2: Similar acylcarnitine profiles; CPT2 is predominantly adult myopathic;
    CACT is predominantly neonatal severe; CPT2 has milder rhabdomyolysis phenotype
  CACT vs MCAD: CACT has C16 elevated + C8 normal; MCAD has C8 elevated;
    CACT has cardiomyopathy; MCAD does NOT

HIGHEST-YIELD EXAM FACTS:
  1. C16 ELEVATED = PRIMARY NBS MARKER (trapped outside mitochondria; cannot cross IMM)
  2. C0 LOW (<15 μmol/L) — carnitine trapped as acylcarnitines; OPPOSITE of CPT1A
  3. L-CARNITINE ESSENTIAL (mandatory supplementation; C0 depleted)
  4. CARDIOMYOPATHY + ARRHYTHMIA — HALLMARK (unlike CPT1A which has NO cardiac disease)
  5. RHABDOMYOLYSIS — present in ALL phenotypes (unlike CPT1A where ABSENT)
  6. MCT OIL THERAPEUTIC (bypasses CACT; medium chain enters via MCT1 directly)
  7. FASTING ABSOLUTE CI + KD ABSOLUTE CI + VPA ABSOLUTE CI
  8. HYPERAMMONEMIA — mechanism same as CPT1A (acyl-CoA → NAGS/CPS1 inhibition)
  9. c.199-10T>G IVS2-10T>G — Yemenite Jewish founder; severe neonatal
  10. HIGH NEONATAL MORTALITY without early diagnosis + MCT + L-carnitine

OMIM Disease: #212138 (Carnitine-acylcarnitine translocase deficiency)
OMIM Gene:    SLC25A20 *613698
Inheritance:  Autosomal Recessive (AR), biallelic LOF
Prevalence:   ~1:300,000–750,000 (very rare)
Locus:        3p21.31
"""

import random

SEED = 281
random.seed(SEED)

# ── Variant table ─────────────────────────────────────────────────────────────
VARIANTS = [
    {"variant": "c.199-10T>G (IVS2-10T>G)", "freq": 30, "domain": "Intron 2 splice acceptor",
     "phenotype": "Severe; Yemenite Jewish founder; neonatal form; complete loss of splicing",
     "note": "Most common variant in Yemenite Jewish patients; causes exon 3 skipping; no functional CACT protein"},
    {"variant": "p.Arg178Gln (c.533G>A)", "freq": 18, "domain": "Transmembrane helix 5 — catalytic core",
     "phenotype": "Severe; various populations; complete LOF",
     "note": "Arg178 is essential for substrate recognition in the mitochondrial carrier family; no transport function"},
    {"variant": "p.Lys32Glu", "freq": 12, "domain": "N-terminal matrix-facing loop",
     "phenotype": "Severe; complete LOF; neonatal onset",
     "note": "Lys32 in matrix-facing loop critical for carnitine release step of antiport cycle"},
    {"variant": "p.Gly80Glu", "freq": 10, "domain": "First transmembrane helix",
     "phenotype": "Moderate; some residual transport function (~10–20%)",
     "note": "Glycine-to-glutamate substitution in first TM helix; partially disrupts antiport geometry"},
    {"variant": "p.Phe545Cys", "freq": 8, "domain": "C-terminal cytosolic loop",
     "phenotype": "Mild; adult myopathic form; residual activity ~25–35%",
     "note": "Mild phenotype; episodic rhabdomyolysis in adults; some transport capacity retained"},
    {"variant": "p.Arg275Cys", "freq": 8, "domain": "Transmembrane helix 7",
     "phenotype": "Moderate-severe; loss of positive charge disrupts carnitine binding",
     "note": "Arg275 contact with acylcarnitine substrate; Cys substitution abolishes electrostatic interaction"},
    {"variant": "p.Asp32Asn", "freq": 5, "domain": "Matrix-facing loop",
     "phenotype": "Severe in homozygotes; intermediate in compound hets",
     "note": "Asparagine substitution disrupts hydrogen-bonding network in substrate channel"},
    {"variant": "p.Ala301Thr", "freq": 4, "domain": "C-terminus",
     "phenotype": "Mild; some residual CACT function retained",
     "note": "Last residue of 301-aa protein; conservative substitution; variable expressivity"},
    {"variant": "Large deletion exon 3-5", "freq": 3, "domain": "Central transmembrane helices 2-4",
     "phenotype": "Severe; complete absence of functional protein",
     "note": "Homozygous deletion; no transcript; lethal neonatal phenotype without treatment"},
    {"variant": "Other compound het", "freq": 2, "domain": "Various",
     "phenotype": "Variable; depends on allele combination",
     "note": "Compound heterozygous; severity determined by less severe allele"},
]

# ── Phenotype distribution ────────────────────────────────────────────────────
PHENOTYPE_DIST = {
    "Severe Neonatal (day 1-5; cardiomyopathy + arrhythmia; high mortality)": 28,
    "Mild/Later-onset (episodic rhabdomyolysis; cardiomyopathy may be absent/mild)": 12,
}


def _make_patient(i):
    """Synthetic CACT patient record (seed=281, deterministic)."""
    rng = random.Random(SEED + i * 43)

    if i < 28:
        # Severe neonatal (70%)
        severity = "Severe"
        variant = rng.choice([
            "c.199-10T>G/c.199-10T>G", "p.Arg178Gln/c.199-10T>G",
            "p.Lys32Glu/p.Arg178Gln", "p.Lys32Glu/c.199-10T>G",
            "p.Arg275Cys/p.Lys32Glu", "p.Asp32Asn/c.199-10T>G",
            "Exon3-5del/p.Arg178Gln",
        ])
        onset_age = rng.randint(0, 5)   # days → stored as months (fraction)
        onset_age_months = round(rng.uniform(0, 0.5), 2)
        c16 = round(rng.uniform(3.0, 8.0), 2)
        c18_1 = round(rng.uniform(1.5, 4.5), 2)
        c18 = round(rng.uniform(1.0, 3.5), 2)
        c0 = round(rng.uniform(2.0, 8.0), 1)
        c14_1 = round(rng.uniform(0.3, 1.2), 2)
        c16_oh = round(rng.uniform(0.01, 0.05), 3)
        glucose = round(rng.uniform(0.5, 2.0), 1)
        ammonia = round(rng.uniform(150, 500), 0)
        ketones = round(rng.uniform(0.0, 0.3), 2)
        cardiomyopathy = True
        arrhythmia = rng.random() < 0.80
        hepatomegaly = True
        transaminitis = True
        rhabdomyolysis = rng.random() < 0.75
        hypotonia = True
        primary_treatment = "IV-glucose + IV-L-carnitine + MCT"
        response = rng.choice(["Partial response", "Good response", "Critical/died"])

    else:
        # Mild / later-onset (30%)
        severity = "Mild"
        variant = rng.choice([
            "p.Phe545Cys/p.Gly80Glu", "p.Gly80Glu/p.Ala301Thr",
            "p.Phe545Cys/p.Arg275Cys", "p.Ala301Thr/p.Arg178Gln",
            "p.Gly80Glu/c.199-10T>G",
        ])
        onset_age_months = round(rng.uniform(12, 120), 1)
        c16 = round(rng.uniform(1.5, 3.5), 2)
        c18_1 = round(rng.uniform(0.8, 2.5), 2)
        c18 = round(rng.uniform(0.6, 2.0), 2)
        c0 = round(rng.uniform(6.0, 14.0), 1)
        c14_1 = round(rng.uniform(0.15, 0.60), 2)
        c16_oh = round(rng.uniform(0.01, 0.04), 3)
        glucose = round(rng.uniform(1.5, 3.5), 1)
        ammonia = round(rng.uniform(50, 250), 0)
        ketones = round(rng.uniform(0.0, 0.4), 2)
        cardiomyopathy = rng.random() < 0.35
        arrhythmia = rng.random() < 0.25
        hepatomegaly = rng.random() < 0.50
        transaminitis = rng.random() < 0.45
        rhabdomyolysis = rng.random() < 0.70
        hypotonia = rng.random() < 0.50
        primary_treatment = rng.choice(["MCT + L-carnitine", "Long-chain fat restriction + L-carnitine"])
        response = rng.choice(["Good response", "Good response", "Partial response"])

    return {
        "id":                   f"CACT-{SEED}-{i+1:02d}",
        "severity":             severity,
        "variant":              variant,
        "onset_age_months":     onset_age_months,
        "c16_umol":             c16,
        "c18_1_umol":           c18_1,
        "c18_umol":             c18,
        "c0_umol":              c0,
        "c14_1_umol":           c14_1,
        "c16_oh_umol":          c16_oh,
        "glucose_mmol":         glucose,
        "ammonia_umol":         ammonia,
        "ketones_mmol":         ketones,
        "cardiomyopathy":       cardiomyopathy,
        "arrhythmia":           arrhythmia,
        "hepatomegaly":         hepatomegaly,
        "transaminitis":        transaminitis,
        "rhabdomyolysis":       rhabdomyolysis,
        "hypotonia":            hypotonia,
        "primary_treatment":    primary_treatment,
        "response":             response,
    }


PATIENTS = [_make_patient(i) for i in range(40)]


def get_overview():
    n = len(PATIENTS)
    severe_n    = sum(1 for p in PATIENTS if p["severity"] == "Severe")
    mild_n      = sum(1 for p in PATIENTS if p["severity"] == "Mild")
    cardio_n    = sum(1 for p in PATIENTS if p["cardiomyopathy"])
    arrhyth_n   = sum(1 for p in PATIENTS if p["arrhythmia"])
    rhabdo_n    = sum(1 for p in PATIENTS if p["rhabdomyolysis"])
    hypogly_n   = sum(1 for p in PATIENTS if p["glucose_mmol"] < 2.5)
    ammonia_n   = sum(1 for p in PATIENTS if p["ammonia_umol"] > 80)
    hepato_n    = sum(1 for p in PATIENTS if p["hepatomegaly"])
    hypotonia_n = sum(1 for p in PATIENTS if p["hypotonia"])
    mct_resp_n  = sum(1 for p in PATIENTS if "Good" in p["response"])

    avg_c16  = round(sum(p["c16_umol"]   for p in PATIENTS) / n, 2)
    avg_c18_1 = round(sum(p["c18_1_umol"] for p in PATIENTS) / n, 2)
    avg_c18  = round(sum(p["c18_umol"]   for p in PATIENTS) / n, 2)
    avg_c0   = round(sum(p["c0_umol"]    for p in PATIENTS) / n, 1)
    avg_c14_1 = round(sum(p["c14_1_umol"] for p in PATIENTS) / n, 2)
    avg_gluc = round(sum(p["glucose_mmol"] for p in PATIENTS) / n, 2)
    avg_nh3  = round(sum(p["ammonia_umol"] for p in PATIENTS) / n, 0)

    return {
        "n_patients":   n,
        "seed":         SEED,
        "disease":      "CACT Deficiency (Carnitine-Acylcarnitine Translocase Deficiency / SLC25A20)",
        "gene":         "SLC25A20",
        "locus":        "3p21.31",
        "omim_disease": "#212138",
        "omim_gene":    "*613698",
        "prevalence":   "~1:300,000–750,000 (very rare)",
        "inheritance":  "Autosomal Recessive (AR), biallelic LOF",
        "severity_distribution": {
            "Severe (neonatal; day 1-5 onset)": severe_n,
            "Mild (later-onset; episodic)":     mild_n,
        },
        "clinical_features": {
            "cardiomyopathy":       cardio_n,
            "arrhythmia":           arrhyth_n,
            "rhabdomyolysis":       rhabdo_n,
            "hypoketotic_hypoglycaemia_lt2.5mmol": hypogly_n,
            "hyperammonemia_gt80":  ammonia_n,
            "hepatomegaly":         hepato_n,
            "hypotonia":            hypotonia_n,
            "mct_good_response":    mct_resp_n,
        },
        "biomarkers": {
            "avg_c16_umol":   avg_c16,
            "avg_c18_1_umol": avg_c18_1,
            "avg_c18_umol":   avg_c18,
            "avg_c0_umol":    avg_c0,
            "avg_c14_1_umol": avg_c14_1,
            "avg_glucose_crisis_mmol": avg_gluc,
            "avg_ammonia_umol": avg_nh3,
        },
        "key_exam_facts": [
            "C16 (palmitoylcarnitine) ELEVATED ≥1.5 μmol/L — PRIMARY NBS MARKER (trapped outside IMM)",
            "C18:1 (oleoylcarnitine) ELEVATED — co-elevated; all long-chain acylcarnitines accumulate",
            "C18 (stearoylcarnitine) ELEVATED — co-elevated",
            "C0 (free carnitine) LOW <15 μmol/L — carnitine trapped as acylcarnitines (OPPOSITE of CPT1A)",
            "C16-OH NORMAL — KEY NEGATIVE vs LCHAD (no retinopathy, no neuropathy in CACT)",
            "C8 NORMAL — KEY NEGATIVE vs MCAD",
            "L-CARNITINE ESSENTIAL (C0 depleted; OPPOSITE of CPT1A where NOT given)",
            "CARDIOMYOPATHY + ARRHYTHMIA — HALLMARK (unlike CPT1A which has NO cardiac disease)",
            "RHABDOMYOLYSIS in ALL phenotypes — KEY EXAM TRAP vs CPT1A where rhabdomyolysis is ABSENT",
            "MCT OIL THERAPEUTIC (medium-chain FA bypass CACT; enter via MCT1 transporter directly)",
            "FASTING ABSOLUTE CI + KD ABSOLUTE CI + VPA ABSOLUTE CI",
            "HYPERAMMONEMIA (50–500 μmol/L; acyl-CoA → CPS1/NAGS inhibition; similar mechanism to CPT1A)",
            "HIGH NEONATAL MORTALITY without early diagnosis + MCT + IV L-carnitine",
            "c.199-10T>G (IVS2-10T>G) — Yemenite Jewish founder; severe neonatal form",
        ],
    }


def get_breakdown():
    patients_out = []
    for p in PATIENTS:
        patients_out.append({
            "id":                   p["id"],
            "severity":             p["severity"],
            "variant":              p["variant"],
            "onset_age_months":     p["onset_age_months"],
            "c16_umol":             p["c16_umol"],
            "c18_1_umol":           p["c18_1_umol"],
            "c18_umol":             p["c18_umol"],
            "c0_umol":              p["c0_umol"],
            "c14_1_umol":           p["c14_1_umol"],
            "c16_oh_umol":          p["c16_oh_umol"],
            "glucose_mmol":         p["glucose_mmol"],
            "ammonia_umol":         p["ammonia_umol"],
            "ketones_mmol":         p["ketones_mmol"],
            "cardiomyopathy":       p["cardiomyopathy"],
            "arrhythmia":           p["arrhythmia"],
            "hepatomegaly":         p["hepatomegaly"],
            "transaminitis":        p["transaminitis"],
            "rhabdomyolysis":       p["rhabdomyolysis"],
            "hypotonia":            p["hypotonia"],
            "primary_treatment":    p["primary_treatment"],
            "response":             p["response"],
        })

    severity_groups = {}
    for p in PATIENTS:
        grp = p["severity"]
        severity_groups.setdefault(grp, []).append(p)

    by_severity = {}
    for grp, pts in severity_groups.items():
        by_severity[grp] = {
            "n":              len(pts),
            "avg_c16":        round(sum(x["c16_umol"]    for x in pts) / len(pts), 2),
            "avg_c18_1":      round(sum(x["c18_1_umol"]  for x in pts) / len(pts), 2),
            "avg_c18":        round(sum(x["c18_umol"]    for x in pts) / len(pts), 2),
            "avg_c0":         round(sum(x["c0_umol"]     for x in pts) / len(pts), 1),
            "avg_glucose":    round(sum(x["glucose_mmol"] for x in pts) / len(pts), 2),
            "avg_ammonia":    round(sum(x["ammonia_umol"] for x in pts) / len(pts), 0),
            "cardiomyopathy_rate": round(sum(1 for x in pts if x["cardiomyopathy"]) / len(pts) * 100, 1),
            "rhabdomyolysis_rate": round(sum(1 for x in pts if x["rhabdomyolysis"]) / len(pts) * 100, 1),
            "arrhythmia_rate":    round(sum(1 for x in pts if x["arrhythmia"]) / len(pts) * 100, 1),
            "good_response_rate": round(sum(1 for x in pts if "Good" in x["response"]) / len(pts) * 100, 1),
        }

    variant_counts = {}
    for p in PATIENTS:
        key = p["variant"].split("/")[0].strip()
        variant_counts[key] = variant_counts.get(key, 0) + 1

    treatment_responses = {
        "mct_oil_therapeutic_n":        sum(1 for p in PATIENTS if "MCT" in p["primary_treatment"]),
        "l_carnitine_essential_n":      len(PATIENTS),   # ALL patients; C0 depleted
        "iv_glucose_acute_n":           sum(1 for p in PATIENTS if p["severity"] == "Severe"),
        "fasting_avoided_strict":       len(PATIENTS),
        "kd_avoided":                   len(PATIENTS),
        "vpa_avoided_absolute_ci":      len(PATIENTS),
        "cardiac_meds_n":               sum(1 for p in PATIENTS if p["cardiomyopathy"]),
    }

    return {
        "patients":          patients_out,
        "by_severity":       by_severity,
        "variant_counts":    variant_counts,
        "treatment_summary": treatment_responses,
        "nbs_profile_summary": {
            "pct_c16_elevated_ge1_5":      round(sum(1 for p in PATIENTS if p["c16_umol"] >= 1.5) / len(PATIENTS) * 100, 1),
            "pct_c0_low_lt15":             round(sum(1 for p in PATIENTS if p["c0_umol"] < 15) / len(PATIENTS) * 100, 1),
            "pct_cardiomyopathy":          round(sum(1 for p in PATIENTS if p["cardiomyopathy"]) / len(PATIENTS) * 100, 1),
            "pct_rhabdomyolysis":          round(sum(1 for p in PATIENTS if p["rhabdomyolysis"]) / len(PATIENTS) * 100, 1),
            "pct_hyperammonemia_gt80":     round(sum(1 for p in PATIENTS if p["ammonia_umol"] > 80) / len(PATIENTS) * 100, 1),
            "pct_c16_oh_normal_lt0_08":    round(sum(1 for p in PATIENTS if p["c16_oh_umol"] < 0.08) / len(PATIENTS) * 100, 1),
        },
    }


def get_definitions():
    return {
        "disease_name": "CACT Deficiency (Carnitine-Acylcarnitine Translocase Deficiency / SLC25A20 Deficiency)",
        "gene":         "SLC25A20 (3p21.31)",
        "locus":        "3p21.31",
        "omim_gene":    "SLC25A20 *613698",
        "omim_disease": "#212138",
        "inheritance":  "Autosomal Recessive (AR) — biallelic LOF",
        "protein": (
            "CACT (Carnitine-Acylcarnitine Translocase) — 301 aa; inner mitochondrial membrane (IMM); "
            "member of the mitochondrial carrier family (SLC25 superfamily). Functions as an obligate "
            "ANTIPORTER: transports long-chain acylcarnitines FROM the intermembrane space INTO the "
            "mitochondrial matrix, in exchange for free carnitine moving OUT of the matrix. "
            "This is CARNITINE SHUTTLE STEP 2 — bridging CPT1 (Step 1, outer IMM) and "
            "CPT2 (Step 3, inner IMM, matrix face)."
        ),
        "enzymatic_function": (
            "CACT catalyses: long-chain acylcarnitine (intermembrane space) → mitochondrial matrix; "
            "simultaneously: free carnitine (matrix) → intermembrane space. "
            "This obligate 1:1 antiport ensures carnitine is recycled after CPT2 regenerates "
            "acyl-CoA in the matrix. CACT LOF → long-chain acylcarnitines CANNOT cross the IMM "
            "→ accumulate in cytoplasm/blood (C16 ↑↑, C18:1 ↑↑, C18 ↑↑) "
            "→ free carnitine (C0) FALLS (trapped as acylcarnitines) "
            "→ beta-oxidation BLOCKED → HYPOKETOTIC HYPOGLYCAEMIA."
        ),
        "pathway": "Long-chain fatty acid beta-oxidation (carnitine shuttle step 2, inner mitochondrial membrane antiporter)",
        "metabolic_block": (
            "CACT LOF → long-chain acylcarnitines (C16, C18:1, C18) CANNOT cross the inner "
            "mitochondrial membrane → accumulate in the intermembrane space and cytoplasm "
            "→ spill into blood (elevated NBS acylcarnitines). "
            "Free carnitine (C0) drops severely (<15 μmol/L) — all carnitine trapped as long-chain "
            "acylcarnitines. Beta-oxidation fully blocked in ALL tissues (unlike CPT1A which spares "
            "heart/muscle). Acyl-CoA accumulation in cytoplasm → inhibits NAGS → reduces NAG "
            "→ CPS1 ↓ → HYPERAMMONAEMIA. Energy failure → CARDIOMYOPATHY + ARRHYTHMIA + "
            "RHABDOMYOLYSIS (all unique compared with CPT1A which spares heart and muscle)."
        ),
        "nbs_marker": (
            "C16 (palmitoylcarnitine) ELEVATED ≥1.5 μmol/L — PRIMARY NBS MARKER. "
            "C18:1 (oleoylcarnitine) ELEVATED — co-primary. "
            "C18 (stearoylcarnitine) ELEVATED — co-elevated. "
            "C0 (free carnitine) LOW <15 μmol/L — carnitine depleted (OPPOSITE of CPT1A). "
            "C16-OH NORMAL — KEY NEGATIVE vs LCHAD. "
            "C8 NORMAL — KEY NEGATIVE vs MCAD."
        ),
        "key_biomarkers": {
            "C16_palmitoylcarnitine":      "ELEVATED ≥1.5 μmol/L (normal <0.6) — PRIMARY NBS MARKER; trapped in cytoplasm",
            "C18_1_oleoylcarnitine":       "ELEVATED (normal <0.5 μmol/L) — co-elevated; all long-chain acylcarnitines trapped",
            "C18_stearoylcarnitine":       "ELEVATED — co-elevated; distinguishes from VLCAD (VLCAD C18 less elevated)",
            "C14_1_tetradecenoylcarnitine": "Mildly elevated — less elevated than in VLCAD; secondary elevation",
            "C0_free_carnitine":           "LOW <15 μmol/L (normal 15–50) — carnitine depleted; OPPOSITE of CPT1A",
            "C16_C0_ratio":                "ELEVATED — supporting ratio; confirms acylcarnitine accumulation",
            "C16_OH_3OH_palmitoylcarnitine": "NORMAL — KEY NEGATIVE vs LCHAD (LCHAD: ≥0.08 μmol/L)",
            "C8_octanoylcarnitine":        "NORMAL — KEY NEGATIVE vs MCAD",
            "C3_propionylcarnitine":       "NORMAL — KEY NEGATIVE vs PA/MMA",
            "Ammonia":                     "ELEVATED 50–500 μmol/L (acyl-CoA accumulation → NAGS/CPS1 inhibition)",
            "Glucose":                     "LOW <2.5 mmol/L during crisis (HYPOKETOTIC hypoglycaemia)",
            "Ketones":                     "ABSENT or inappropriately low (HYPOKETOTIC — beta-oxidation fully blocked)",
        },
        "clinical_features": {
            "Cardiomyopathy":             "HALLMARK — dilated or hypertrophic; life-threatening; unlike CPT1A (NO cardiac disease); urgent echocardiography on NBS alert",
            "Arrhythmia":                 "Ventricular arrhythmias; potentially fatal; urgent ECG + cardiac monitoring",
            "Hypoketotic_hypoglycaemia":  "HALLMARK — primary presentation; fasting/illness triggered; ketones absent",
            "Hyperammonemia":             "50–500 μmol/L; mechanism: acyl-CoA accumulation → NAGS inhibition → CPS1 ↓; similar to CPT1A but with cardiac disease",
            "Rhabdomyolysis":             "Present in ALL phenotypes — KEY EXAM TRAP vs CPT1A (CPT1A has NO rhabdomyolysis); CK elevated",
            "Hepatomegaly_transaminitis": "Common; hepatic steatosis and lipid accumulation",
            "Hypotonia_weakness":         "Global; especially severe neonatal form",
            "High_neonatal_mortality":    "Without early diagnosis and treatment; Day 1–5 onset; 75% of cases severe neonatal",
            "NO_retinopathy":             "KEY NEGATIVE vs LCHAD (LCHAD has pigmentary retinal degeneration)",
            "NO_peripheral_neuropathy":   "KEY NEGATIVE vs LCHAD (LCHAD has axonal neuropathy)",
            "No_maternal_AFLP_HELLP":     "KEY NEGATIVE vs LCHAD (LCHAD: maternal AFLP/HELLP when fetus is LCHAD-affected)",
        },
        "treatment": {
            "MCT_oil_THERAPEUTIC":        "Level A — medium-chain FA (C8/C10) bypass CACT entirely; enter mitochondria directly via MCT1 transporter; oxidised normally by MCAD/SCAD; cornerstone of long-term management",
            "L_Carnitine_ESSENTIAL":      "Level A — C0 severely depleted; supplementation MANDATORY to replete free carnitine; OPPOSITE of CPT1A where L-carnitine is NOT routinely given",
            "Fasting_ABSOLUTE_CI":        "Level A — absolute contraindication; primary trigger; lipolysis releases long-chain FA that cannot be oxidised via blocked CACT",
            "KD_ABSOLUTE_CI":             "Absolute contraindication — long-chain fat floods blocked CACT; worsens disease catastrophically",
            "VPA_ABSOLUTE_CI":            "Absolute contraindication — valproate inhibits FAO + depletes carnitine; especially dangerous given cardiomyopathy; seizures must be managed with non-VPA agents",
            "IV_glucose_10pct":           "Level A emergency — first-line; target GIR 8–10 mg/kg/min; anti-catabolic",
            "IV_L_carnitine_emergency":   "Level A emergency — IV L-carnitine for acute depletion; replenishes C0 rapidly",
            "Long_chain_fat_restriction": "Level A — dietary restriction of long-chain FA; replace calories with MCT + carbohydrate",
            "Triheptanoin_C7":            "Level B — odd-chain C7 FA; anaplerotic substrate entering TCA as propionyl-CoA; maintains cardiac function",
            "Cardiac_medications":        "Digoxin / ACE inhibitors for cardiomyopathy management; antiarrhythmics if ventricular arrhythmia",
            "Emergency_cardiology":       "Antiarrhythmic therapy if life-threatening arrhythmia; avoid drugs that worsen carnitine depletion",
        },
        "contraindications": [
            "FASTING — ABSOLUTE CONTRAINDICATION (primary metabolic trigger)",
            "KETOGENIC DIET — ABSOLUTE CONTRAINDICATION (long-chain fat floods blocked CACT)",
            "VALPROATE — ABSOLUTE CONTRAINDICATION (inhibits FAO; depletes carnitine; cardiomyopathy risk)",
            "HIGH DIETARY LONG-CHAIN FAT — dangerous; worsens acylcarnitine accumulation",
            "L-CARNITINE avoidance is WRONG here — C0 is LOW; supplementation is ESSENTIAL (contrast: CPT1A)",
        ],
        "key_distinguishing_facts": [
            "C16 HIGH + C0 LOW = CACT (vs CPT1A: C0 HIGH + C16 NORMAL — completely inverted profile)",
            "CARDIOMYOPATHY + ARRHYTHMIA + RHABDOMYOLYSIS — all three present; CPT1A has NONE",
            "L-CARNITINE is MANDATORY (C0 depleted; must replete; OPPOSITE of CPT1A where not given)",
            "MCT oil is THERAPEUTIC (same as CPT1A, VLCAD, LCHAD — medium chain bypasses blocked step)",
            "C16-OH NORMAL — no retinopathy, no neuropathy → KEY NEGATIVE vs LCHAD",
            "C8 NORMAL — KEY NEGATIVE vs MCAD",
            "STEP 2 of carnitine shuttle — bridges CPT1 (Step 1) and CPT2 (Step 3)",
            "Yemenite Jewish founder c.199-10T>G (IVS2-10T>G) — most common severe variant",
            "VPA is ABSOLUTE CI (not just high-risk) — cardiomyopathy makes this especially dangerous",
            "Neonatal onset Day 1–5 in 75% — one of the most severe FAO disorders",
        ],
        "genetics": {
            "key_variants": {
                "c.199-10T>G_IVS2-10T>G": "Yemenite Jewish founder; severe neonatal; exon 3 skipping; complete LOF",
                "p.Arg178Gln_c.533G>A":    "Severe; catalytic core transmembrane helix 5; complete LOF",
                "p.Lys32Glu":              "Severe; N-terminal matrix loop; complete LOF",
                "p.Gly80Glu":              "Moderate; first TM helix; ~10–20% residual transport",
                "p.Phe545Cys":             "Mild; adult myopathic; ~25–35% residual activity",
                "p.Arg275Cys":             "Moderate-severe; TM helix 7; disrupts carnitine electrostatic binding",
            },
            "population_note": (
                "CACT deficiency is very rare globally (~1:300,000–750,000). "
                "The IVS2-10T>G splicing variant is a founder allele in Yemenite Jewish populations. "
                "Unlike CPT1A (which has the dramatic Inuit/Arctic founder p.Pro479Leu), CACT has "
                "no single dominant founder in non-Jewish populations — most cases are private mutations "
                "or compound heterozygotes."
            ),
        },
        "comparison_table": {
            "CACT_vs_CPT1A":  "C16 HIGH + C0 LOW (CACT) vs C0 HIGH + C16 NORMAL (CPT1A); cardiomyopathy + rhabdo (CACT) vs NO cardiac NO rhabdo (CPT1A); L-carnitine ESSENTIAL (CACT) vs NOT routine (CPT1A); different shuttle step (2 vs 1)",
            "CACT_vs_VLCAD":  "CACT: C16+C18+C18:1 all elevated, C0 LOW; VLCAD: C14:1 primary, C16 elevated, C18 less so; both have cardiomyopathy; CACT has more severe neonatal onset",
            "CACT_vs_LCHAD":  "CACT: C16-OH NORMAL, no retinopathy, no neuropathy, no maternal AFLP; LCHAD: C16-OH ELEVATED, retinopathy, axonal neuropathy, maternal AFLP/HELLP — key exam discriminator",
            "CACT_vs_CPT2":   "Similar acylcarnitine profiles (both elevated C16/C18:1); CPT2 predominantly adult-myopathic rhabdomyolysis; CACT predominantly severe neonatal; CPT2 Step 3 (matrix face), CACT Step 2 (IMM translocase)",
            "CACT_vs_MCAD":   "CACT: C16 elevated, C8 NORMAL, cardiomyopathy; MCAD: C8 elevated, C16 lower, NO cardiomyopathy; completely different acylcarnitine pattern",
        },
        "carnitine_shuttle_context": (
            "Step 1 (CPT1A — outer IMM, cytosolic face; RATE-LIMITING): "
            "long-chain acyl-CoA + carnitine → long-chain acylcarnitine + CoA-SH\n"
            "Step 2 (CACT/SLC25A20 — inner IMM translocase; ANTIPORTER):  "
            "long-chain acylcarnitine IN (intermembrane → matrix) ↔ free carnitine OUT (matrix → intermembrane)  [← CACT BLOCK]\n"
            "Step 3 (CPT2 — inner IMM, matrix face): "
            "long-chain acylcarnitine + CoA-SH → long-chain acyl-CoA + free carnitine\n"
            "Steps 4–7 (matrix): Beta-oxidation proper (VLCAD → HADHA → HADHB → ACAT1 for long chain)\n\n"
            "CACT DEFICIENCY → block at Step 2 → long-chain acylcarnitines accumulate BEFORE the IMM "
            "→ C16 ↑↑, C18:1 ↑↑, C18 ↑↑ in blood; C0 ↓↓ (all carnitine trapped as acylcarnitines). "
            "ALL tissues affected (heart, muscle, liver) — unlike CPT1A (liver-specific) because "
            "CACT is ubiquitously expressed (not tissue-specific like CPT1A/CPT1B)."
        ),
    }
