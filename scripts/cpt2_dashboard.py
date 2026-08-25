#!/usr/bin/env python3
"""CPT2 Deficiency (Carnitine Palmitoyltransferase II Deficiency) Dashboard.

CPT2 gene encodes CPT2 (Carnitine Palmitoyltransferase II):
  CPT2: 590 aa; 1p32.3; inner mitochondrial membrane (IMM), MATRIX FACE; monomer
  Locus: 1p32.3; Autosomal Recessive (AR)
  OMIM Gene: *600650
  OMIM Disease: #255110 (Myopathic, adult-onset)
                #600649 (Severe infantile hepatocardiomuscular)
                #608836 (Lethal neonatal)

  CPT2 CATALYSES (inner mitochondrial membrane, MATRIX face — CARNITINE SHUTTLE STEP 3):
    Long-chain acylcarnitine + CoA-SH → long-chain acyl-CoA + free carnitine
    (Regenerates acyl-CoA for entry into beta-oxidation proper; free carnitine
    recycled back to intermembrane space via CACT for the next transport cycle)
    Without CPT2 → long-chain acylcarnitines CANNOT be converted to acyl-CoA
    → beta-oxidation BLOCKED at final step of carnitine shuttle
    → acyl-CoA substrates UNAVAILABLE for VLCAD/HADHA/HADHB

CARNITINE SHUTTLE — ALL THREE STEPS:
  Step 1 (CPT1A/CPT1B — outer IMM, cytosolic face; RATE-LIMITING):
    Long-chain acyl-CoA + L-carnitine → long-chain acylcarnitine + CoA-SH
    [CPT1A liver; CPT1B heart/muscle; CPT1C brain]
  Step 2 (CACT/SLC25A20 — inner IMM translocase; ANTIPORTER):
    Long-chain acylcarnitine (intermembrane space) IN → mitochondrial matrix
    Free carnitine (matrix) OUT → intermembrane space  [ANTIPORT]
  Step 3 (CPT2 — inner IMM, MATRIX FACE):
    Long-chain acylcarnitine + CoA-SH → long-chain acyl-CoA + free carnitine
    [← CPT2 BLOCK HERE → acylcarnitines ACCUMULATE in matrix / cannot be oxidised]
  Steps 4–7 (matrix): Beta-oxidation proper (VLCAD → HADHA → HADHB → ACAT1 for long chain)

CPT2 METABOLIC BLOCK:
  CPT2 LOF → long-chain acylcarnitines CANNOT be converted to acyl-CoA in matrix
  → C16, C18:1, C18 ELEVATED in blood (similar profile to CACT, but from different step)
  → Free carnitine (C0) low-to-normal in mild myopathic; very low in severe neonatal
  → Beta-oxidation BLOCKED → hypoketotic hypoglycaemia (severe phenotypes)
  → Energy crisis → cardiomyopathy/hepatopathy (neonatal/infantile only)

THREE PHENOTYPES (key exam discriminator):
  1. MYOPATHIC (most common, >90–95% of CPT2):
     - Onset teens/young adults; episodic exercise-induced rhabdomyolysis; myoglobinuria
     - Triggers: prolonged exercise, cold, fasting, fever, infections, general anaesthesia
     - Normal between episodes; NO cardiomyopathy; NO hypoglycaemia between episodes
     - NBS often NORMAL (C16 only mildly elevated; may normalise between episodes)
     - Classic first presentation: dark-brown urine (myoglobinuria) after prolonged exercise
     - p.Ser113Leu HYPOMORPHIC allele: temperature-sensitive reduction in CPT2 activity
  2. SEVERE INFANTILE HEPATOCARDIOMUSCULAR (intermediate):
     - Onset months to years; cardiomyopathy, hepatomegaly, hypoketotic hypoglycaemia
     - Similar to CACT in clinical features but CPT2 distinct locus/step
  3. LETHAL NEONATAL (rarest):
     - Day 1–5; respiratory failure; cardiomyopathy; renal cysts; brain malformations
     - Dysembryogenetic features unique to CPT2 lethal neonatal (vs CACT)
     - Poor prognosis even with treatment

KEY NBS MARKER — CPT2 PROFILE (HIGH C16 + C18:1 + C18; C0 variable):
  C16 (palmitoylcarnitine)  ELEVATED — co-primary NBS MARKER (mild in myopathic; marked in severe)
  C18:1 (oleoylcarnitine)   ELEVATED — co-elevated
  C18 (stearoylcarnitine)   ELEVATED — co-elevated
  C14:1                     Mildly elevated in severe forms
  C0 (free carnitine)       LOW in severe/neonatal; normal-low in myopathic (key vs CACT)
  C16-OH                    NORMAL — KEY NEGATIVE vs LCHAD
  C8                        NORMAL — KEY NEGATIVE vs MCAD
  → Similar acylcarnitine profile to CACT but DIFFERENT CLINICAL PHENOTYPE (myopathic vs neonatal)

KEY EXAM DISCRIMINATORS:
  CPT2 MYOPATHIC vs CACT:
    - CPT2 myopathic: adult-onset rhabdomyolysis; NO cardiomyopathy; NBS often NORMAL
    - CACT: neonatal severe; HALLMARK cardiomyopathy; C0 profoundly LOW; NBS positive
  CPT2 NEONATAL vs CACT NEONATAL:
    - CPT2 lethal neonatal: RENAL CYSTS + BRAIN MALFORMATIONS (dysembryogenesis) — UNIQUE
    - CACT: no structural malformations; purely metabolic; c.199-10T>G Yemenite founder
  CPT2 vs CPT1A:
    - CPT1A: C0 HIGH + C16 NORMAL (inverted); hepatic only; no rhabdomyolysis; Arctic founder
    - CPT2: C16 HIGH + C18:1 HIGH; rhabdomyolysis; muscle involvement
  p.Ser113Leu: TEMPERATURE-SENSITIVE allele — CPT2 activity reduced at 41°C (fever/exercise)
    explains why fever/exercise triggers crisis; key exam fact

TREATMENT:
  MYOPATHIC FORM:
  1. AVOID PROLONGED EXERCISE — primary prevention; pace activities; warm-up essential
  2. HIGH-CARBOHYDRATE PRE-EXERCISE diet — maintains glycogen; reduces FAO demand
  3. MCT OIL — Level B (medium-chain FA bypass CPT2 in myopathic; reduces long-chain FAO demand)
  4. L-CARNITINE — Level C (controversial in myopathic; may worsen by delivering more acylcarnitines;
     use only if C0 depleted; NOT routine unlike CACT where ESSENTIAL)
  5. AVOID TRIGGERS: prolonged fasting, cold exposure, fever, general anaesthesia
  6. IV GLUCOSE + IV FLUIDS — emergency (rhabdomyolysis; maintain urine output; prevent AKI)
  7. BICARBONATE — if metabolic acidosis during rhabdomyolysis

  SEVERE/NEONATAL FORMS (same as other long-chain FAO):
  8. MCT OIL — Level A (bypasses CPT2; medium chain enters mitochondria directly via MCT1)
  9. LONG-CHAIN FAT RESTRICTION — Level A
  10. FASTING — ABSOLUTE CI (all severe forms)
  11. KD — ABSOLUTE CI (long-chain fat floods blocked pathway)
  12. VPA — ABSOLUTE CI (inhibits FAO; carnitine depletion; especially dangerous with cardiomyopathy)
  13. L-CARNITINE — Level A if C0 depleted (severe forms; myopathic — Level C only)
  14. GENERAL ANAESTHESIA — HIGH RISK in myopathic; pre-op glucose; short-acting agents preferred

KEY CONTRAINIDCATIONS (myopathic form — different from neonatal-severe forms):
  - Prolonged fasting (NOT ABSOLUTE CI in myopathic between episodes)
  - KD is ABSOLUTE CI in severe forms (but less established for myopathic)
  - VPA — ABSOLUTE CI in severe forms; HIGH RISK in myopathic
  - Statins — HIGH RISK (myopathy/rhabdomyolysis risk on background CPT2 deficiency)

GENETICS — KEY VARIANTS:
  p.Ser113Leu (c.338C>T)   — MOST COMMON myopathic allele; 30–50% allele frequency;
                              temperature-sensitive hypomorph; reduces activity at febrile temps;
                              ~3–5% heterozygote frequency general population (common variant!)
  p.Pro50His (c.149C>A)    — Severe; neonatal/infantile; European; near-complete LOF
  p.Arg151Gln (c.452G>A)   — Severe infantile hepatocardiomuscular; complete LOF
  p.Arg503Cys (c.1507C>T)  — Myopathic; European; partial LOF
  p.Val368Ile               — Mild; common compound-het partner in myopathic form
  p.Tyr628Ser               — Lethal neonatal; complete LOF; dysembryogenetic features
  p.Ile475Thr               — Moderate; infantile form

COMPARISON TABLE (exam discriminators):
  CPT2 vs CACT: Both C16/C18:1 elevated; CPT2 myopathic = adult rhabdomyolysis, NO cardiac,
    NBS sometimes normal; CACT = neonatal severe, hallmark cardiomyopathy, C0 profoundly LOW
  CPT2 vs CPT1A: CPT2 C16 HIGH + rhabdomyolysis; CPT1A C0 HIGH + C16 NORMAL; inverse profile;
    CPT1A hepatic-only (no muscle); CPT2 Step 3, CPT1A Step 1
  CPT2 lethal neonatal vs CACT neonatal: CPT2 has RENAL CYSTS + BRAIN MALFORMATIONS
    (dysembryogenesis); CACT has NO structural anomalies; Yemenite founder c.199-10T>G
  CPT2 vs VLCAD: Both C14:1 elevated in severe; VLCAD more C14:1-dominant; CPT2 more C16/C18
  CPT2 vs LCHAD: LCHAD has C16-OH elevated (3-OH acylcarnitines); CPT2 C16-OH NORMAL;
    LCHAD has retinopathy + neuropathy + maternal AFLP; CPT2 does NOT

HIGHEST-YIELD EXAM FACTS:
  1. CPT2 = CARNITINE SHUTTLE STEP 3 (matrix face; reverses CPT1A action; regenerates acyl-CoA)
  2. MYOPATHIC FORM = MOST COMMON (>90%); exercise-induced rhabdomyolysis; teens/young adults
  3. p.Ser113Leu = temperature-sensitive allele; 30-50% myopathic allele freq; COMMON in general pop
  4. THREE PHENOTYPES: Myopathic (adult, mild) / Severe Infantile / Lethal Neonatal (severe)
  5. RENAL CYSTS + BRAIN MALFORMATIONS in LETHAL NEONATAL CPT2 — KEY NEGATIVE for CACT
  6. NBS often NORMAL in myopathic (crisis-dependent; C16 may normalise between episodes)
  7. C16 HIGH + C18:1 HIGH — similar to CACT but different clinical phenotype (adult muscle vs neonatal)
  8. L-CARNITINE NOT ROUTINE in myopathic (Level C; may worsen) — unlike CACT where ESSENTIAL
  9. AVOID TRIGGERS: prolonged exercise, cold, fever, fasting, general anaesthesia, statins
  10. MCT OIL: Level A severe forms / Level B myopathic; bypasses CPT2 (medium chain via MCT1)

OMIM Disease: #255110 (myopathic), #600649 (severe infantile), #608836 (lethal neonatal)
OMIM Gene:    CPT2 *600650
Inheritance:  Autosomal Recessive (AR), biallelic LOF (p.Ser113Leu hypomorphic — temperature-sensitive)
Prevalence:   ~1:100,000 (myopathic; one of most common FAO rhabdomyolysis disorders)
Locus:        1p32.3
"""

import random

SEED = 283
random.seed(SEED)

# ── Variant table ─────────────────────────────────────────────────────────────
VARIANTS = [
    {"variant": "p.Ser113Leu (c.338C>T)", "freq": 38, "domain": "Substrate-binding tunnel",
     "phenotype": "Myopathic; temperature-sensitive; 30-50% allele freq; ~3-5% heterozygote in pop",
     "note": "MOST COMMON CPT2 variant. Hypomorphic: normal activity at 37°C; near-abolished at 41°C. "
             "Explains fever/exercise triggers. Biallelic required for clinical disease; single copy = carrier."},
    {"variant": "p.Arg503Cys (c.1507C>T)", "freq": 14, "domain": "C-terminal domain",
     "phenotype": "Myopathic; European; partial LOF; episodic rhabdomyolysis",
     "note": "Often compound heterozygous with p.Ser113Leu; moderate residual activity"},
    {"variant": "p.Val368Ile", "freq": 12, "domain": "Catalytic core",
     "phenotype": "Mild; frequent compound-het partner; some residual CPT2 activity",
     "note": "Mild phenotype alone; pathogenic when combined with severe allele"},
    {"variant": "p.Pro50His (c.149C>A)", "freq": 10, "domain": "N-terminal mitochondrial targeting sequence",
     "phenotype": "Severe; neonatal/infantile; European; near-complete LOF",
     "note": "Disrupts targeting sequence; reduced mitochondrial import; almost no residual activity"},
    {"variant": "p.Arg151Gln (c.452G>A)", "freq": 8, "domain": "Catalytic core",
     "phenotype": "Severe infantile hepatocardiomuscular; complete LOF",
     "note": "Arginine critical for CoA-SH binding step; no catalytic activity; severe phenotype"},
    {"variant": "p.Ile475Thr", "freq": 6, "domain": "Transmembrane anchor region",
     "phenotype": "Moderate; infantile form; some residual activity",
     "note": "Less severe than p.Pro50His; intermediate phenotype; age of onset childhood"},
    {"variant": "p.Tyr628Ser", "freq": 4, "domain": "C-terminal tail (final residue region)",
     "phenotype": "Lethal neonatal; complete LOF; renal cysts; brain malformations",
     "note": "Rare; complete LOF; associated with lethal neonatal CPT2 with dysembryogenetic features"},
    {"variant": "Other compound het", "freq": 5, "domain": "Various",
     "phenotype": "Variable; depends on allele combination",
     "note": "Private mutations; variable expressivity; severity determined by less severe allele"},
    {"variant": "Large intragenic deletion", "freq": 3, "domain": "Exon 4-6",
     "phenotype": "Severe; complete absence of functional protein; infantile onset",
     "note": "Homozygous deletion; no transcript; severe infantile form"},
]

# ── Phenotype distribution ────────────────────────────────────────────────────
PHENOTYPE_DIST = {
    "Myopathic (adult-onset; exercise-induced rhabdomyolysis; NO cardiomyopathy)": 32,
    "Severe Infantile (cardiomyopathy; hepatopathy; hypoglycaemia; <1y onset)": 5,
    "Lethal Neonatal (cardiomyopathy; renal cysts; brain malformations; day 1-5)": 3,
}


def _make_patient(i):
    """Synthetic CPT2 patient record (seed=283, deterministic)."""
    rng = random.Random(SEED + i * 47)

    if i < 32:
        # Myopathic form (80%)
        phenotype = "Myopathic"
        variant = rng.choice([
            "p.Ser113Leu/p.Ser113Leu", "p.Ser113Leu/p.Arg503Cys",
            "p.Ser113Leu/p.Val368Ile",  "p.Arg503Cys/p.Val368Ile",
            "p.Ser113Leu/p.Ile475Thr",  "p.Ser113Leu/p.Pro50His",
        ])
        onset_age_months = round(rng.uniform(120, 360), 1)   # teens–young adults
        c16 = round(rng.uniform(0.7, 2.5), 2)   # often mildly elevated or borderline
        c18_1 = round(rng.uniform(0.4, 1.8), 2)
        c18 = round(rng.uniform(0.3, 1.5), 2)
        c0 = round(rng.uniform(12.0, 42.0), 1)  # often normal-low between episodes
        c14_1 = round(rng.uniform(0.05, 0.35), 2)
        c16_oh = round(rng.uniform(0.01, 0.04), 3)
        glucose = round(rng.uniform(3.5, 6.0), 1)   # normal between episodes
        ammonia = round(rng.uniform(20, 60), 0)      # normal between episodes
        ketones = round(rng.uniform(0.1, 1.2), 2)    # may be present between episodes
        cardiomyopathy = False       # KEY EXAM POINT: NO cardiomyopathy in myopathic CPT2
        hepatomegaly = rng.random() < 0.10   # unusual in myopathic
        transaminitis = True        # CK/ALT elevation during rhabdo
        rhabdomyolysis = True       # defining feature
        myoglobinuria = rng.random() < 0.85
        renal_cysts = False
        brain_malformation = False
        trigger = rng.choice([
            "Prolonged exercise", "Cold exposure + exercise",
            "Fever/infection + exercise", "Prolonged fasting + exercise",
            "General anaesthesia", "Viral illness",
        ])
        primary_treatment = rng.choice([
            "Avoid prolonged exercise + high-carb pre-exercise",
            "High-carb diet + MCT supplement",
            "Avoid triggers + rest during crisis + IV fluids",
        ])
        response = rng.choice(["Good response", "Good response", "Partial response"])

    elif i < 37:
        # Severe infantile (12.5%)
        phenotype = "Severe Infantile"
        variant = rng.choice([
            "p.Arg151Gln/p.Pro50His", "p.Pro50His/p.Pro50His",
            "p.Arg151Gln/p.Ile475Thr", "p.Pro50His/p.Ile475Thr",
        ])
        onset_age_months = round(rng.uniform(2, 18), 1)
        c16 = round(rng.uniform(2.5, 7.0), 2)
        c18_1 = round(rng.uniform(1.5, 4.0), 2)
        c18 = round(rng.uniform(1.0, 3.5), 2)
        c0 = round(rng.uniform(4.0, 14.0), 1)
        c14_1 = round(rng.uniform(0.2, 0.9), 2)
        c16_oh = round(rng.uniform(0.01, 0.05), 3)
        glucose = round(rng.uniform(0.8, 2.5), 1)
        ammonia = round(rng.uniform(80, 350), 0)
        ketones = round(rng.uniform(0.0, 0.3), 2)
        cardiomyopathy = rng.random() < 0.90
        hepatomegaly = True
        transaminitis = True
        rhabdomyolysis = rng.random() < 0.50
        myoglobinuria = rng.random() < 0.30
        renal_cysts = rng.random() < 0.20   # occasional in severe infantile
        brain_malformation = False
        trigger = "Fasting/illness/fever"
        primary_treatment = "MCT + L-carnitine + avoid fasting"
        response = rng.choice(["Partial response", "Good response"])

    else:
        # Lethal neonatal (7.5%)
        phenotype = "Lethal Neonatal"
        variant = rng.choice([
            "p.Tyr628Ser/p.Arg151Gln", "p.Tyr628Ser/p.Pro50His",
            "p.Pro50His/Exon4-6del",
        ])
        onset_age_months = round(rng.uniform(0, 0.2), 3)   # Day 1–5
        c16 = round(rng.uniform(4.0, 9.0), 2)
        c18_1 = round(rng.uniform(2.5, 5.5), 2)
        c18 = round(rng.uniform(1.5, 4.5), 2)
        c0 = round(rng.uniform(1.5, 7.0), 1)
        c14_1 = round(rng.uniform(0.4, 1.2), 2)
        c16_oh = round(rng.uniform(0.01, 0.05), 3)
        glucose = round(rng.uniform(0.4, 1.5), 1)
        ammonia = round(rng.uniform(150, 600), 0)
        ketones = round(rng.uniform(0.0, 0.1), 2)
        cardiomyopathy = True
        hepatomegaly = True
        transaminitis = True
        rhabdomyolysis = rng.random() < 0.60
        myoglobinuria = False
        renal_cysts = True    # HALLMARK of lethal neonatal CPT2 — KEY NEGATIVE vs CACT
        brain_malformation = rng.random() < 0.75   # dysembryogenesis — KEY vs CACT
        trigger = "Neonatal metabolic crisis (Day 1-5)"
        primary_treatment = "IV glucose + MCT + L-carnitine (palliative; poor prognosis)"
        response = rng.choice(["Critical/died", "Critical/died", "Partial response"])

    return {
        "id":                   f"CPT2-{SEED}-{i+1:02d}",
        "phenotype":            phenotype,
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
        "hepatomegaly":         hepatomegaly,
        "transaminitis":        transaminitis,
        "rhabdomyolysis":       rhabdomyolysis,
        "myoglobinuria":        myoglobinuria,
        "renal_cysts":          renal_cysts,
        "brain_malformation":   brain_malformation,
        "trigger":              trigger,
        "primary_treatment":    primary_treatment,
        "response":             response,
    }


PATIENTS = [_make_patient(i) for i in range(40)]


def get_overview():
    n = len(PATIENTS)
    myopathic_n   = sum(1 for p in PATIENTS if p["phenotype"] == "Myopathic")
    infantile_n   = sum(1 for p in PATIENTS if p["phenotype"] == "Severe Infantile")
    neonatal_n    = sum(1 for p in PATIENTS if p["phenotype"] == "Lethal Neonatal")
    cardio_n      = sum(1 for p in PATIENTS if p["cardiomyopathy"])
    rhabdo_n      = sum(1 for p in PATIENTS if p["rhabdomyolysis"])
    myoglobin_n   = sum(1 for p in PATIENTS if p["myoglobinuria"])
    hepato_n      = sum(1 for p in PATIENTS if p["hepatomegaly"])
    renal_cysts_n = sum(1 for p in PATIENTS if p["renal_cysts"])
    brain_mal_n   = sum(1 for p in PATIENTS if p["brain_malformation"])
    hypogly_n     = sum(1 for p in PATIENTS if p["glucose_mmol"] < 2.5)
    good_resp_n   = sum(1 for p in PATIENTS if "Good" in p["response"])

    avg_c16  = round(sum(p["c16_umol"]  for p in PATIENTS) / n, 2)
    avg_c18_1 = round(sum(p["c18_1_umol"] for p in PATIENTS) / n, 2)
    avg_c18  = round(sum(p["c18_umol"]  for p in PATIENTS) / n, 2)
    avg_c0   = round(sum(p["c0_umol"]   for p in PATIENTS) / n, 1)
    avg_c14_1 = round(sum(p["c14_1_umol"] for p in PATIENTS) / n, 2)
    avg_gluc = round(sum(p["glucose_mmol"] for p in PATIENTS) / n, 2)
    avg_nh3  = round(sum(p["ammonia_umol"] for p in PATIENTS) / n, 0)

    return {
        "n_patients":       n,
        "seed":             SEED,
        "disease":          "CPT2 Deficiency (Carnitine Palmitoyltransferase II Deficiency / CPT2 Deficiency)",
        "gene":             "CPT2",
        "locus":            "1p32.3",
        "omim_disease":     "#255110 (myopathic) / #600649 (severe infantile) / #608836 (lethal neonatal)",
        "omim_gene":        "*600650",
        "prevalence":       "~1:100,000 (myopathic; one of the most common FAO rhabdomyolysis disorders)",
        "inheritance":      "Autosomal Recessive (AR), biallelic LOF",
        "phenotype_distribution": {
            "Myopathic (adult-onset rhabdomyolysis; NO cardiomyopathy)": myopathic_n,
            "Severe Infantile (cardiomyopathy; hepatopathy)":             infantile_n,
            "Lethal Neonatal (renal cysts; brain malformations)":         neonatal_n,
        },
        "clinical_features": {
            "rhabdomyolysis":                  rhabdo_n,
            "myoglobinuria":                   myoglobin_n,
            "cardiomyopathy":                  cardio_n,
            "hepatomegaly":                    hepato_n,
            "renal_cysts":                     renal_cysts_n,
            "brain_malformation":              brain_mal_n,
            "hypoketotic_hypoglycaemia_lt2.5": hypogly_n,
            "good_treatment_response":         good_resp_n,
        },
        "biomarkers": {
            "avg_c16_umol":              avg_c16,
            "avg_c18_1_umol":            avg_c18_1,
            "avg_c18_umol":              avg_c18,
            "avg_c0_umol":               avg_c0,
            "avg_c14_1_umol":            avg_c14_1,
            "avg_glucose_mmol":          avg_gluc,
            "avg_ammonia_umol":          avg_nh3,
        },
        "key_exam_facts": [
            "CPT2 = CARNITINE SHUTTLE STEP 3 (inner IMM, MATRIX FACE — converts acylcarnitine back to acyl-CoA)",
            "MYOPATHIC FORM MOST COMMON (>90%): exercise-induced rhabdomyolysis; teens/adults; NO cardiomyopathy",
            "p.Ser113Leu = TEMPERATURE-SENSITIVE hypomorphic allele; 30–50% myopathic allele frequency",
            "p.Ser113Leu ~3–5% heterozygote frequency in GENERAL POPULATION (common allele!)",
            "THREE PHENOTYPES: Myopathic / Severe Infantile Hepatocardiomuscular / Lethal Neonatal",
            "RENAL CYSTS + BRAIN MALFORMATIONS — hallmark of LETHAL NEONATAL CPT2 — KEY NEGATIVE vs CACT",
            "NBS often NORMAL in myopathic between episodes (C16 may normalise — crisis-dependent elevation)",
            "C16 HIGH + C18:1 HIGH — similar to CACT but DIFFERENT clinical presentation (adult muscle vs neonatal)",
            "L-CARNITINE NOT ROUTINE in myopathic (Level C; may worsen rhabdo by increasing acylcarnitine load)",
            "UNLIKE CACT: C0 often normal-low in myopathic; L-carnitine NOT essential (CACT C0 always LOW)",
            "Triggers (myopathic): prolonged exercise · cold · fever · fasting · general anaesthesia · statins",
            "MCT OIL: Level A severe forms / Level B myopathic (medium chain bypasses CPT2 via MCT1)",
            "AVOID STATINS — high risk for rhabdomyolysis on background CPT2 deficiency",
            "CPT2 STEP 3 vs CPT1A STEP 1: CPT2 C16 HIGH (similar to CACT); CPT1A C0 HIGH + C16 NORMAL",
        ],
    }


def get_breakdown():
    patients_out = []
    for p in PATIENTS:
        patients_out.append({
            "id":                   p["id"],
            "phenotype":            p["phenotype"],
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
            "hepatomegaly":         p["hepatomegaly"],
            "transaminitis":        p["transaminitis"],
            "rhabdomyolysis":       p["rhabdomyolysis"],
            "myoglobinuria":        p["myoglobinuria"],
            "renal_cysts":          p["renal_cysts"],
            "brain_malformation":   p["brain_malformation"],
            "trigger":              p["trigger"],
            "primary_treatment":    p["primary_treatment"],
            "response":             p["response"],
        })

    phenotype_groups = {}
    for p in PATIENTS:
        grp = p["phenotype"]
        phenotype_groups.setdefault(grp, []).append(p)

    by_phenotype = {}
    for grp, pts in phenotype_groups.items():
        by_phenotype[grp] = {
            "n":               len(pts),
            "avg_c16":         round(sum(x["c16_umol"]    for x in pts) / len(pts), 2),
            "avg_c18_1":       round(sum(x["c18_1_umol"]  for x in pts) / len(pts), 2),
            "avg_c18":         round(sum(x["c18_umol"]    for x in pts) / len(pts), 2),
            "avg_c0":          round(sum(x["c0_umol"]     for x in pts) / len(pts), 1),
            "avg_glucose":     round(sum(x["glucose_mmol"] for x in pts) / len(pts), 2),
            "avg_ammonia":     round(sum(x["ammonia_umol"] for x in pts) / len(pts), 0),
            "cardiomyopathy_rate": round(sum(1 for x in pts if x["cardiomyopathy"]) / len(pts) * 100, 1),
            "rhabdomyolysis_rate": round(sum(1 for x in pts if x["rhabdomyolysis"]) / len(pts) * 100, 1),
            "myoglobinuria_rate":  round(sum(1 for x in pts if x["myoglobinuria"])  / len(pts) * 100, 1),
            "renal_cysts_rate":    round(sum(1 for x in pts if x["renal_cysts"])    / len(pts) * 100, 1),
            "good_response_rate":  round(sum(1 for x in pts if "Good" in x["response"]) / len(pts) * 100, 1),
        }

    variant_counts = {}
    for p in PATIENTS:
        key = p["variant"].split("/")[0].strip()
        variant_counts[key] = variant_counts.get(key, 0) + 1

    treatment_summary = {
        "avoid_prolonged_exercise_n":       sum(1 for p in PATIENTS if p["phenotype"] == "Myopathic"),
        "mct_oil_n":                        len(PATIENTS),
        "iv_glucose_fluids_n":              sum(1 for p in PATIENTS if p["phenotype"] in ["Severe Infantile", "Lethal Neonatal"]),
        "fasting_avoided_severe_forms":     sum(1 for p in PATIENTS if p["phenotype"] != "Myopathic"),
        "kd_avoided_severe_forms":          sum(1 for p in PATIENTS if p["phenotype"] != "Myopathic"),
        "vpa_absolute_ci_severe":           sum(1 for p in PATIENTS if p["phenotype"] != "Myopathic"),
        "statins_avoided_all":              len(PATIENTS),
    }

    return {
        "patients":           patients_out,
        "by_phenotype":       by_phenotype,
        "variant_counts":     variant_counts,
        "treatment_summary":  treatment_summary,
        "nbs_profile_summary": {
            "pct_c16_elevated_ge1_5":      round(sum(1 for p in PATIENTS if p["c16_umol"] >= 1.5) / n * 100, 1)
                                            if (n := len(PATIENTS)) else 0,
            "pct_c0_normal_ge12":          round(sum(1 for p in PATIENTS if p["c0_umol"] >= 12) / len(PATIENTS) * 100, 1),
            "pct_rhabdomyolysis":          round(sum(1 for p in PATIENTS if p["rhabdomyolysis"]) / len(PATIENTS) * 100, 1),
            "pct_cardiomyopathy":          round(sum(1 for p in PATIENTS if p["cardiomyopathy"]) / len(PATIENTS) * 100, 1),
            "pct_renal_cysts":             round(sum(1 for p in PATIENTS if p["renal_cysts"]) / len(PATIENTS) * 100, 1),
            "pct_brain_malformation":      round(sum(1 for p in PATIENTS if p["brain_malformation"]) / len(PATIENTS) * 100, 1),
            "pct_c16_oh_normal_lt0_08":    round(sum(1 for p in PATIENTS if p["c16_oh_umol"] < 0.08) / len(PATIENTS) * 100, 1),
            "pct_myopathic_no_cardiomyo":  round(sum(1 for p in PATIENTS if p["phenotype"] == "Myopathic" and not p["cardiomyopathy"]) / len(PATIENTS) * 100, 1),
        },
    }


def get_definitions():
    return {
        "disease_name": "CPT2 Deficiency (Carnitine Palmitoyltransferase II Deficiency)",
        "gene":         "CPT2 (1p32.3)",
        "locus":        "1p32.3",
        "omim_gene":    "CPT2 *600650",
        "omim_disease": "#255110 (Myopathic) / #600649 (Severe Infantile Hepatocardiomuscular) / #608836 (Lethal Neonatal)",
        "inheritance":  "Autosomal Recessive (AR) — biallelic LOF (p.Ser113Leu hypomorphic common allele)",
        "protein": (
            "CPT2 (Carnitine Palmitoyltransferase II) — 590 aa; inner mitochondrial membrane (IMM), "
            "MATRIX FACE; monomer; member of the carnitine palmitoyltransferase family. "
            "CPT2 is the MATRIX-SIDE enzyme that REVERSES CPT1 action: it converts long-chain "
            "acylcarnitines BACK to long-chain acyl-CoAs + free carnitine inside the mitochondrial matrix. "
            "This is CARNITINE SHUTTLE STEP 3 — the final step that regenerates the acyl-CoA substrate "
            "for entry into beta-oxidation proper (VLCAD → HADHA/MTP → HADHB → ACAT1)."
        ),
        "enzymatic_function": (
            "CPT2 catalyses: long-chain acylcarnitine + CoA-SH → long-chain acyl-CoA + free carnitine "
            "(inside mitochondrial matrix). Free carnitine generated is recycled back to the intermembrane "
            "space via CACT (Step 2) for the next transport cycle. CPT2 LOF → long-chain acylcarnitines "
            "cannot be converted to acyl-CoA in the matrix → beta-oxidation BLOCKED (cannot enter "
            "VLCAD/HADHA/HADHB pathway) → C16, C18:1, C18 accumulate in blood."
        ),
        "pathway": "Long-chain fatty acid beta-oxidation (carnitine shuttle step 3, inner mitochondrial membrane, matrix face)",
        "metabolic_block": (
            "CPT2 LOF → long-chain acylcarnitines trapped in mitochondrial matrix CANNOT be converted "
            "to acyl-CoA → beta-oxidation cannot proceed (VLCAD needs acyl-CoA not acylcarnitine) → "
            "C16 (palmitoylcarnitine), C18:1 (oleoylcarnitine), C18 (stearoylcarnitine) accumulate in "
            "blood. C0 (free carnitine): variable — normal-low in myopathic (less severe; adequate "
            "carnitine recycling during rest); markedly low in severe neonatal/infantile forms. "
            "Acyl-CoA deficit → reduced ketogenesis → hypoketotic hypoglycaemia (severe forms only). "
            "In MYOPATHIC form: between episodes everything is near-normal; NBS may be false-negative."
        ),
        "three_phenotypes": {
            "Myopathic_adult_onset": (
                "Most common (>90–95%). Onset: teens to young adults. "
                "Exercise-induced rhabdomyolysis + myoglobinuria (dark-brown urine). "
                "Triggers: prolonged exercise, cold, fasting, fever, infections, general anaesthesia. "
                "Normal between episodes — no cardiomyopathy, no hypoglycaemia at rest. "
                "NBS often NORMAL (C16 may normalise between episodes; crisis-dependent elevation). "
                "p.Ser113Leu temperature-sensitive: CPT2 activity near-normal at 37°C, markedly "
                "reduced at 41°C (fever/exercise temperature). CK markedly elevated during rhabdo."
            ),
            "Severe_infantile_hepatocardiomuscular": (
                "Rare (<5%). Onset: months to 2 years. Cardiomyopathy (dilated/hypertrophic), "
                "hepatomegaly, hepatopathy, hypoketotic hypoglycaemia, rhabdomyolysis. "
                "Similar clinically to CACT but different gene (CPT2 vs SLC25A20) and step (3 vs 2). "
                "Renal cysts occasionally present (unlike CACT). NBS: C16 elevated, C18:1 elevated."
            ),
            "Lethal_neonatal": (
                "Rarest (<1–2%). Onset: Day 1–5. Respiratory failure, severe cardiomyopathy, "
                "hepatomegaly, hypoketotic hypoglycaemia. "
                "HALLMARK: RENAL CYSTS + BRAIN MALFORMATIONS (dysembryogenesis) — "
                "KEY EXAM DISCRIMINATOR vs CACT which has NO structural anomalies. "
                "Poor prognosis despite treatment. Complete CPT2 LOF (null alleles)."
            ),
        },
        "nbs_marker": (
            "C16 (palmitoylcarnitine) ELEVATED — co-primary NBS marker. "
            "C18:1 (oleoylcarnitine) ELEVATED — co-elevated. "
            "C18 (stearoylcarnitine) ELEVATED — co-elevated. "
            "C0 (free carnitine): VARIABLE — normal-low in myopathic; low in severe forms. "
            "C16-OH NORMAL — KEY NEGATIVE vs LCHAD. "
            "C8 NORMAL — KEY NEGATIVE vs MCAD. "
            "IMPORTANT: NBS may be NORMAL in myopathic CPT2 between episodes (crisis-dependent)."
        ),
        "key_biomarkers": {
            "C16_palmitoylcarnitine":         "ELEVATED — co-primary NBS marker; mildly elevated in myopathic (may normalise); markedly elevated in severe",
            "C18_1_oleoylcarnitine":          "ELEVATED — co-elevated; similar profile to CACT but different clinical presentation",
            "C18_stearoylcarnitine":          "ELEVATED — co-elevated",
            "C14_1_tetradecenoylcarnitine":   "Mildly elevated in severe forms; less elevated than in VLCAD",
            "C0_free_carnitine":              "VARIABLE — normal-low in myopathic (between episodes); LOW in severe neonatal/infantile",
            "C16_OH_3OH_palmitoylcarnitine":  "NORMAL — KEY NEGATIVE vs LCHAD (LCHAD: ≥0.08 μmol/L elevated)",
            "C8_octanoylcarnitine":           "NORMAL — KEY NEGATIVE vs MCAD",
            "CK_creatine_kinase":             "MARKEDLY ELEVATED during rhabdomyolysis (often >10,000 U/L); KEY diagnostic in myopathic",
            "Myoglobin_urine":                "PRESENT during rhabdo — dark-brown/cola-coloured urine; risk of AKI",
            "Glucose":                        "NORMAL between episodes (myopathic); LOW during crisis in severe forms",
            "Ketones":                        "Present between episodes in myopathic (unlike CACT where always hypoketotic)",
            "Ammonia":                        "NORMAL in myopathic; ELEVATED in severe neonatal/infantile (mechanism same as CPT1A/CACT)",
        },
        "clinical_features": {
            "Rhabdomyolysis_myopathic":           "HALLMARK of myopathic CPT2 — exercise-induced; episodic; CK markedly elevated; myoglobinuria",
            "Myoglobinuria":                      "Dark-brown/cola urine — classic presentation; AKI risk if severe",
            "NO_cardiomyopathy_myopathic":        "KEY EXAM FACT — NO cardiomyopathy in myopathic form (unlike CACT, LCHAD, VLCAD, GA2 severe forms)",
            "Cardiomyopathy_severe_forms":        "Present in severe infantile and lethal neonatal; absent in myopathic",
            "Renal_cysts":                        "HALLMARK of lethal neonatal CPT2 — KEY DISCRIMINATOR vs CACT (CACT has no renal cysts)",
            "Brain_malformations_dysembryogenesis": "Lethal neonatal CPT2 only — neuronal migration defects; absent in CACT",
            "Hepatomegaly_hepatopathy":           "Severe infantile and lethal neonatal; unusual in myopathic",
            "Hypoketotic_hypoglycaemia":          "Severe forms; absent between episodes in myopathic",
            "Fever_exercise_cold_triggers":       "Myopathic: prolonged exercise, cold exposure, fever, fasting, general anaesthesia",
            "Normal_between_episodes":            "Myopathic: completely normal examination, labs, and NBS between crises",
            "NBS_false_negative_risk":            "Myopathic CPT2: C16 may normalise between episodes — screening may be missed",
        },
        "treatment": {
            "Avoid_prolonged_exercise":       "PRIMARY PREVENTION (myopathic) — pace activities; warm-up; avoid exhaustion; paced increasing exercise tolerance",
            "High_carb_pre_exercise_diet":    "Level B (myopathic) — maintains glycogen; reduces long-chain FAO demand during exercise; glucose loading before exertion",
            "MCT_oil":                        "Level A (severe forms) / Level B (myopathic) — medium-chain FA (C8/C10) bypass CPT2; enter mitochondria via MCT1 transporter",
            "L_Carnitine":                    "Level A if C0 depleted (severe forms) / Level C (myopathic — NOT routine; may increase rhabdomyolysis by elevating acylcarnitine load)",
            "IV_glucose_IV_fluids":           "Level A emergency — rhabdomyolysis: high IV fluid rate (3–4 ml/kg/hr) to maintain urine output ≥3 ml/kg/hr; prevent myoglobin-induced AKI",
            "Bicarbonate":                    "If metabolic acidosis during acute rhabdomyolysis episode",
            "Fasting_avoidance":              "ABSOLUTE CI (severe neonatal/infantile) / Prudent avoidance (myopathic) — less strict than CACT in well-compensated myopathic between episodes",
            "KD_avoidance":                   "ABSOLUTE CI (severe forms) / Avoid (myopathic) — long-chain fat worsens blocked pathway",
            "VPA_avoidance":                  "ABSOLUTE CI (severe forms with cardiomyopathy) / HIGH RISK (myopathic) — inhibits FAO; carnitine depletion",
            "Statins_avoidance":              "HIGH RISK in ALL CPT2 — increases rhabdomyolysis risk; contraindicated",
            "General_anaesthesia_caution":    "HIGH RISK (myopathic) — metabolic trigger; use short-acting agents; maintain glucose; avoid prolonged fasting pre-op",
            "Bezafibrate_investigational":    "Level C investigational — PPAR-α agonist; may upregulate residual CPT2 activity in p.Ser113Leu patients",
        },
        "contraindications": [
            "STATINS — HIGH RISK in ALL CPT2 phenotypes (rhabdomyolysis risk on background CPT2 deficiency)",
            "KETOGENIC DIET — ABSOLUTE CONTRAINDICATION (severe forms); avoid (myopathic)",
            "VALPROATE — ABSOLUTE CONTRAINDICATION (severe forms with cardiomyopathy); HIGH RISK (myopathic)",
            "FASTING — ABSOLUTE CI (severe forms); prudent avoidance (myopathic between episodes)",
            "PROLONGED EXERCISE without high-carb pre-loading — PRIMARY TRIGGER (myopathic)",
            "GENERAL ANAESTHESIA without metabolic preparation — HIGH RISK (myopathic)",
            "L-CARNITINE routine use in myopathic is NOT recommended (Level C only; may worsen rhabdo)",
        ],
        "key_distinguishing_facts": [
            "CPT2 = STEP 3 (matrix face) — FINAL step of carnitine shuttle; after CACT step 2; converts acylcarnitine → acyl-CoA",
            "MYOPATHIC FORM (>90%): adult rhabdomyolysis; NO cardiomyopathy; NBS often NORMAL between episodes",
            "p.Ser113Leu: temperature-sensitive; 30–50% myopathic allele; MOST COMMON; ~3–5% in general population",
            "RENAL CYSTS + BRAIN MALFORMATIONS = lethal neonatal CPT2 — ABSENT in CACT (no dysembryogenesis)",
            "C0 NORMAL-LOW in myopathic (between episodes) — unlike CACT where C0 is always PROFOUNDLY LOW",
            "L-CARNITINE NOT ROUTINE in myopathic (Level C; may worsen by increasing acylcarnitine load)",
            "NBS may be FALSE-NEGATIVE in myopathic CPT2 — crisis-dependent; C16 normalises between episodes",
            "STATINS are HIGH RISK — unique to CPT2 (and all FAO disorders) — increases rhabdomyolysis",
            "CPT2 vs CACT: same acylcarnitine profile (C16/C18:1 HIGH); DIFFERENT PHENOTYPE (adult muscle vs neonatal cardiac)",
            "CPT2 vs CPT1A: CPT2 C16 HIGH (like CACT); CPT1A C0 HIGH + C16 NORMAL — inverse profile; different shuttle step",
        ],
        "genetics": {
            "key_variants": {
                "p.Ser113Leu_c.338C>T":     "MOST COMMON; temperature-sensitive hypomorph; 30–50% myopathic allele freq; ~3–5% general pop heterozygote; normal activity at 37°C, markedly reduced at 41°C",
                "p.Arg503Cys_c.1507C>T":    "Myopathic; European; partial LOF; often compound-het with p.Ser113Leu",
                "p.Val368Ile":              "Mild; frequent compound-het partner; some residual CPT2 activity",
                "p.Pro50His_c.149C>A":      "Severe; neonatal/infantile; European; disrupts mitochondrial targeting; near-complete LOF",
                "p.Arg151Gln_c.452G>A":     "Severe infantile hepatocardiomuscular; catalytic core; complete LOF",
                "p.Ile475Thr":              "Moderate; infantile form; some residual activity",
                "p.Tyr628Ser":              "Lethal neonatal; complete LOF; associated with renal cysts and brain malformations",
            },
            "population_note": (
                "CPT2 myopathic is the most common FAO disorder causing exercise-induced rhabdomyolysis "
                "in adults. The p.Ser113Leu allele is remarkably common (~3–5% of the general population "
                "are heterozygous carriers) — requiring a second pathogenic allele for clinical disease. "
                "Unlike CPT1A (strong Arctic/Inuit founder) and CACT (Yemenite Jewish founder), CPT2 "
                "myopathic is pan-ethnic with p.Ser113Leu as the dominant allele across European, "
                "Asian, and North American populations."
            ),
        },
        "comparison_table": {
            "CPT2_vs_CACT":   "Both: C16 HIGH + C18:1 HIGH; CPT2 myopathic = adult rhabdomyolysis + NO cardiac; CACT = neonatal severe + hallmark cardiomyopathy + C0 profoundly LOW; CPT2 lethal neonatal = renal cysts + brain malformations (ABSENT in CACT); CACT c.199-10T>G Yemenite founder; CPT2 p.Ser113Leu common allele",
            "CPT2_vs_CPT1A":  "CPT2: C16 HIGH + rhabdomyolysis + myopathic; CPT1A: C0 HIGH + C16 NORMAL (inverted) + hepatic only + NO rhabdomyolysis + Arctic founder p.Pro479Leu; CPT2 is Step 3 (matrix), CPT1A is Step 1 (outer IMM)",
            "CPT2_vs_VLCAD":  "Both: C16/C14:1 elevated; VLCAD: C14:1 more dominant + cardiomyopathy in all ages; CPT2 myopathic: muscle only + NO cardiomyopathy + NBS may be normal",
            "CPT2_vs_LCHAD":  "LCHAD: C16-OH ELEVATED (3-OH acylcarnitines) + retinopathy + axonal neuropathy + maternal AFLP/HELLP; CPT2: C16-OH NORMAL + no retinopathy + no neuropathy",
            "CPT2_vs_MCAD":   "MCAD: C8 elevated + NO C16 elevation (primary); CPT2: C16 HIGH + C8 NORMAL; completely different chain-length acylcarnitine pattern",
        },
        "carnitine_shuttle_context": (
            "Step 1 (CPT1A — outer IMM, cytosolic face; RATE-LIMITING): "
            "long-chain acyl-CoA + carnitine → long-chain acylcarnitine + CoA-SH\n"
            "Step 2 (CACT/SLC25A20 — inner IMM translocase; ANTIPORTER): "
            "long-chain acylcarnitine IN (intermembrane → matrix) ↔ free carnitine OUT (matrix → intermembrane)\n"
            "Step 3 (CPT2 — inner IMM, MATRIX FACE):  [← CPT2 BLOCK HERE]\n"
            "  long-chain acylcarnitine + CoA-SH → long-chain acyl-CoA + free carnitine\n"
            "  [Regenerates acyl-CoA for beta-oxidation; free carnitine recycled via CACT]\n"
            "Steps 4–7 (matrix): Beta-oxidation proper (VLCAD → HADHA → HADHB → ACAT1)\n\n"
            "CPT2 DEFICIENCY → block at Step 3 → long-chain acylcarnitines accumulate IN the matrix "
            "AND in blood (C16 ↑, C18:1 ↑, C18 ↑). C0 variable (normal-low in myopathic; LOW in severe). "
            "MYOPATHIC form: CPT1B (heart/muscle Step 1) + CPT2 deficiency → muscle-selective FAO failure "
            "during high-energy demand (exercise); heart relatively spared in myopathic because "
            "cardiomyocytes use glucose/short-chain FA as primary fuel at rest."
        ),
    }
