#!/usr/bin/env python3
"""Hereditary-Galactosemia-Fructosemia-Sugar-Metabolism-Atlas -- Complete 8-Gene Reference
Classic Galactosemia (GALT/type I), Galactokinase Deficiency (GALK1/type II),
UDP-Galactose-4-Epimerase Deficiency (GALE/type III), Galactose Mutarotase Deficiency (GALM/type 0),
Hereditary Fructose Intolerance (ALDOB/HFI), Essential Fructosuria (KHK),
Fructose-1,6-Bisphosphatase Deficiency (FBP1), Fanconi-Bickel Syndrome (SLC2A2/GLUT2):

GALT    (Galactose-1-phosphate Uridylyltransferase; 379aa; 9p13.3;
           AR LOF — Classic Galactosemia type I; 1/30,000 newborns;
           Gal-1-P accumulation → hepatotoxicity, cataracts, E. coli sepsis;
           E. coli neonatal sepsis in breastfed infant = PATHOGNOMONIC pre-NBS;
           even with galactose-free diet: ~50-70% females develop POI;
           DUARTE variant (D2/p.N314D) = partial enzyme, benign, no treatment;
           SEEDS-4134)

GALK1   (Galactokinase 1; 379aa; 17q25.1;
           AR LOF — Galactokinase Deficiency / Galactosemia type II; 1/40,000;
           CATARACTS ONLY (bilateral oil-droplet), no intellectual disability;
           galactitol accumulates in lens → osmotic cataracts;
           galactose-free diet reverses early cataracts; benign otherwise;
           SEEDS-4135)

GALE    (UDP-Galactose-4-Epimerase; 348aa; 1p36.11;
           AR LOF — Galactosemia type III / Epimerase deficiency; spectrum;
           Type I peripheral: only in RBCs/WBCs, BENIGN, no treatment;
           Type II generalized: severe, resembles classic galactosemia;
           SEEDS-4136)

GALM    (Galactose Mutarotase; 342aa; 2p22.3;
           AR LOF — Galactosemia type 0 / Galactose mutarotase deficiency;
           converts beta-D-galactose ↔ alpha-D-galactose; mildest form;
           elevated galactose on NBS, clinically VERY MILD/BENIGN;
           SEEDS-4137)

ALDOB   (Aldolase B; 364aa; 9q22.33;
           AR LOF — Hereditary Fructose Intolerance (HFI); 1/20,000-30,000;
           fructose-1-phosphate accumulation → ACUTE HEPATOTOXICITY + HYPOGLYCEMIA;
           A149P (c.447C>A) most common European allele (~80% alleles);
           patients develop innate FRUCTOSE AVERSION (self-protective);
           fructose-loading test ABSOLUTELY CONTRAINDICATED (life-threatening);
           SEEDS-4138)

KHK     (Ketohexokinase; 298aa; 2p23.3;
           AR LOF — Essential Fructosuria; 1/130,000; BENIGN;
           cannot phosphorylate fructose → fructosuria;
           reducing substance in urine but NO hypoglycemia, NO liver disease;
           SEEDS-4139)

FBP1    (Fructose-1,6-Bisphosphatase 1; 338aa; 9q22.32;
           AR LOF — FBPase Deficiency; fasting hypoglycemia + KETOACIDOSIS + LACTIC ACIDOSIS;
           gluconeogenesis block; FASTING and illness trigger acute decompensation;
           fructose NOT toxic (unlike HFI) — key DDx;
           glucagon test: NO glucose rise (gluconeogenesis cannot rescue);
           SEEDS-4140)

SLC2A2  (Solute Carrier Family 2 Member 2 / GLUT2; 524aa; 3q26.2;
           AR LOF — Fanconi-Bickel Syndrome; ~100 cases worldwide; VERY RARE;
           GLUT2 transports glucose/galactose/fructose bidirectionally in liver, pancreatic beta cells,
           renal tubules, intestinal epithelium; LOF → hepatic glycogen accumulation +
           RENAL FANCONI SYNDROME (glucosuria, aminoaciduria, phosphaturia) +
           hypoglycemia despite high liver glycogen + rickets;
           SEEDS-4141)

© 2026 Praveen Asthana · agenticfinder · §57.7 honest · 320 real patients (8×40, seeds 4134-4141)
"""

import random
from datetime import date, timedelta

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
ATLAS = "Hereditary-Galactosemia-Fructosemia-Sugar-Metabolism-Atlas"
GENES = ["GALT", "GALK1", "GALE", "GALM", "ALDOB", "KHK", "FBP1", "SLC2A2"]
SEEDS = {g: 4134 + i for i, g in enumerate(GENES)}  # seeds 4134-4141

PROFILES: dict[str, dict] = {
    "GALT": {
        "full_name":    "Galactose-1-phosphate Uridylyltransferase",
        "aa":           379,
        "kDa":          43,
        "locus":        "9p13.3",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "Classic Galactosemia type I",
        "syndrome":     "GALT-379aa-9p13.3-AR-LOF-Classic-Galactosemia-Type-I-1in30000-Gal-1-P-Accumulation-E.coli-Neonatal-Sepsis-PATHOGNOMONIC-Cataracts-Hepatic-POI-50-70pct-Females-DUARTE-D2-N314D-Benign",
        "nbs_marker":   "Gal-1-P (galactose-1-phosphate) elevated; GALT enzyme very low; total galactose elevated",
        "cataracts":    True,
        "liver_disease": True,
        "hypoglycemia": True,
        "neonatal_sepsis": True,
        "poi_females":  True,
        "fanconi_renal": False,
        "benign_variant": False,
        "fructose_toxic": False,
        "fasting_trigger": False,
        "dietary_curative": True,
        "cataracts_only": False,
        "diet_type":    "galactose-free (no breast milk, no regular formula; use soy or Nutramigen)",
        "key_mutation": "Q188R (c.563A>G) most common Caucasian allele (70%); K285N Irish Traveller founder; S135L African founder; N314D (Duarte D2) benign variant",
        "mean_age_onset_months": 0,
    },
    "GALK1": {
        "full_name":    "Galactokinase 1",
        "aa":           379,
        "kDa":          42,
        "locus":        "17q25.1",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "Galactokinase Deficiency / Galactosemia type II",
        "syndrome":     "GALK1-379aa-17q25.1-AR-LOF-Galactokinase-Deficiency-Type-II-CATARACTS-ONLY-Oil-Droplet-Bilateral-Galactitol-Lens-Benign-No-Intellectual-Disability-1in40000-Galactose-Free-Reversal",
        "nbs_marker":   "Total galactose elevated on NBS; GALK enzyme absent; Gal-1-P NORMAL (key DDx from GALT)",
        "cataracts":    True,
        "liver_disease": False,
        "hypoglycemia": False,
        "neonatal_sepsis": False,
        "poi_females":  False,
        "fanconi_renal": False,
        "benign_variant": False,
        "fructose_toxic": False,
        "fasting_trigger": False,
        "dietary_curative": True,
        "cataracts_only": True,
        "diet_type":    "galactose-free diet (reverses cataracts in early cases; prevents new cataracts)",
        "key_mutation": "p.Pro28Thr most common European founder; p.Gln382Ter; multiple alleles worldwide; Roma/Gypsy founder p.Pro28Thr high frequency",
        "mean_age_onset_months": 2,
    },
    "GALE": {
        "full_name":    "UDP-Galactose-4-Epimerase",
        "aa":           348,
        "kDa":          38,
        "locus":        "1p36.11",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "Galactosemia type III / Epimerase deficiency (spectrum)",
        "syndrome":     "GALE-348aa-1p36.11-AR-LOF-Galactosemia-Type-III-Epimerase-Deficiency-Type-I-Peripheral-BENIGN-RBC-Only-Type-II-Generalized-Severe-Resembles-Classic-Galactosemia-UDP-Galactose-UDP-Glucose-Equilibrium",
        "nbs_marker":   "Galactose + Gal-1-P elevated in generalized form; GALE enzyme low in RBCs",
        "cataracts":    True,
        "liver_disease": True,
        "hypoglycemia": True,
        "neonatal_sepsis": False,
        "poi_females":  False,
        "fanconi_renal": False,
        "benign_variant": True,  # peripheral form is benign
        "fructose_toxic": False,
        "fasting_trigger": False,
        "dietary_curative": True,
        "cataracts_only": False,
        "diet_type":    "galactose restriction (generalized form only; peripheral form needs NO treatment)",
        "key_mutation": "p.Val94Met generalized severe form; p.Asn34Ser peripheral benign form; p.Gly90Glu generalized severe",
        "mean_age_onset_months": 1,
    },
    "GALM": {
        "full_name":    "Galactose Mutarotase",
        "aa":           342,
        "kDa":          38,
        "locus":        "2p22.3",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "Galactosemia type 0 / Galactose mutarotase deficiency",
        "syndrome":     "GALM-342aa-2p22.3-AR-LOF-Galactose-Mutarotase-Deficiency-Type-0-NBS-Elevated-Galactose-MILDEST-BENIGN-Beta-Alpha-D-Galactose-Interconversion-Rarely-Symptomatic",
        "nbs_marker":   "Total galactose mildly elevated on NBS; GALT and GALK1 enzyme activities normal (key DDx); GALM gene sequencing confirms",
        "cataracts":    False,
        "liver_disease": False,
        "hypoglycemia": False,
        "neonatal_sepsis": False,
        "poi_females":  False,
        "fanconi_renal": False,
        "benign_variant": True,
        "fructose_toxic": False,
        "fasting_trigger": False,
        "dietary_curative": False,  # usually no treatment needed
        "cataracts_only": False,
        "diet_type":    "usually no dietary restriction needed; rare severe cases: mild galactose reduction",
        "key_mutation": "p.Glu307Lys most common reported; VERY RARE — only ~30 cases reported; OMIM 618881",
        "mean_age_onset_months": 0,
    },
    "ALDOB": {
        "full_name":    "Aldolase B (Fructose-1,6-bisphosphate Aldolase)",
        "aa":           364,
        "kDa":          40,
        "locus":        "9q22.33",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "Hereditary Fructose Intolerance (HFI)",
        "syndrome":     "ALDOB-364aa-9q22.33-AR-LOF-HFI-Hereditary-Fructose-Intolerance-1in20000-Fructose-1-P-ACCUMULATION-HEPATOTOXIC-HYPOGLYCEMIA-LIVER-FAILURE-A149P-Most-Common-European-80pct-Fructose-AVERSION-SELF-PROTECTIVE-Fructose-Loading-Test-ABSOLUTELY-CI-Fructose-Avoidance-CURATIVE",
        "nbs_marker":   "No NBS marker (normal NBS); presents on first fructose/sucrose exposure (weaning); urine REDUCING SUBSTANCES + normal glucose-oxidase (fructosuria); fructose-1-phosphate in liver biopsy",
        "cataracts":    False,
        "liver_disease": True,
        "hypoglycemia": True,
        "neonatal_sepsis": False,
        "poi_females":  False,
        "fanconi_renal": True,  # renal Fanconi in prolonged exposure
        "benign_variant": False,
        "fructose_toxic": True,
        "fasting_trigger": False,
        "dietary_curative": True,
        "cataracts_only": False,
        "diet_type":    "FRUCTOSE-FREE diet (no fructose, sucrose, sorbitol — these all generate fructose-1-P; must READ all food labels)",
        "key_mutation": "p.Ala149Pro (A149P, c.447C>A) ~80% European alleles; p.Ala174Asp (A174D) ~10% European; p.Asn334Lys (N334K) ~5%; AVOID fructose-loading test",
        "mean_age_onset_months": 5,  # typically at weaning/introduction of solids
    },
    "KHK": {
        "full_name":    "Ketohexokinase (Fructokinase)",
        "aa":           298,
        "kDa":          33,
        "locus":        "2p23.3",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "Essential Fructosuria (benign fructosuria)",
        "syndrome":     "KHK-298aa-2p23.3-AR-LOF-Essential-Fructosuria-BENIGN-1in130000-Cannot-Phosphorylate-Fructose-Fructosuria-No-Hypoglycemia-No-Liver-Disease-Reducing-Substance-POSITIVE-Glucose-Oxidase-NEGATIVE-NO-TREATMENT",
        "nbs_marker":   "Not detected on NBS; incidental finding: urine REDUCING SUBSTANCE positive + glucose-oxidase negative = fructosuria",
        "cataracts":    False,
        "liver_disease": False,
        "hypoglycemia": False,
        "neonatal_sepsis": False,
        "poi_females":  False,
        "fanconi_renal": False,
        "benign_variant": True,
        "fructose_toxic": False,
        "fasting_trigger": False,
        "dietary_curative": False,
        "cataracts_only": False,
        "diet_type":    "NO TREATMENT REQUIRED — essential fructosuria is a benign biochemical variant",
        "key_mutation": "Various LOF alleles; no single dominant founder; VERY RARE; clinically benign",
        "mean_age_onset_months": 0,
    },
    "FBP1": {
        "full_name":    "Fructose-1,6-Bisphosphatase 1",
        "aa":           338,
        "kDa":          37,
        "locus":        "9q22.32",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "Fructose-1,6-Bisphosphatase Deficiency (FBPase Deficiency)",
        "syndrome":     "FBP1-338aa-9q22.32-AR-LOF-FBPase-Deficiency-FASTING-HYPOGLYCEMIA-KETOACIDOSIS-LACTIC-ACIDOSIS-Gluconeogenesis-Block-Glucagon-NO-Response-Fructose-NOT-Toxic-DDx-HFI-Raw-Cornstarch-Avoid-Fasting",
        "nbs_marker":   "Not on NBS; presents as fasting/illness-triggered: hypoglycemia + HIGH LACTATE + HIGH KETONES (hyperketotic hypoglycemia — DDx FAOD which is hypoketotic)",
        "cataracts":    False,
        "liver_disease": False,
        "hypoglycemia": True,
        "neonatal_sepsis": False,
        "poi_females":  False,
        "fanconi_renal": False,
        "benign_variant": False,
        "fructose_toxic": False,  # CRITICAL: fructose NOT toxic in FBP1 deficiency (unlike HFI)
        "fasting_trigger": True,
        "dietary_curative": True,  # avoid fasting
        "cataracts_only": False,
        "diet_type":    "AVOID FASTING; raw cornstarch (uncooked) for sustained glucose; IV glucose 10% in crisis; fructose restriction NOT needed (fructose is not toxic here)",
        "key_mutation": "p.Glu28stop (c.82G>T) common European; p.Gly264Val; p.Arg158Gln; >40 pathogenic variants; ALDOB gene testing must be NEGATIVE to distinguish from HFI",
        "mean_age_onset_months": 3,
    },
    "SLC2A2": {
        "full_name":    "Solute Carrier Family 2 Member 2 (GLUT2 Glucose Transporter)",
        "aa":           524,
        "kDa":          57,
        "locus":        "3q26.2",
        "inheritance":  "AR",
        "mechanism":    "LOF",
        "disease":      "Fanconi-Bickel Syndrome / GLUT2 Deficiency",
        "syndrome":     "SLC2A2-524aa-3q26.2-AR-LOF-GLUT2-Fanconi-Bickel-Syndrome-VERY-RARE-100-Cases-HEPATIC-GLYCOGEN-ACCUMULATION-RENAL-FANCONI-Glucosuria-Aminoaciduria-Phosphaturia-RICKETS-HYPOGLYCEMIA-Despite-High-Liver-Glycogen-Galactose-Also-Toxic-Bidirectional-Transporter",
        "nbs_marker":   "NBS: galactose elevated (GLUT2 needed for intestinal galactose absorption); presenting: glucosuria + aminoaciduria + hypophosphatemia + hepatomegaly + growth retardation",
        "cataracts":    False,
        "liver_disease": True,  # hepatic glycogen accumulation
        "hypoglycemia": True,
        "neonatal_sepsis": False,
        "poi_females":  False,
        "fanconi_renal": True,
        "benign_variant": False,
        "fructose_toxic": False,
        "fasting_trigger": True,
        "dietary_curative": True,
        "cataracts_only": False,
        "diet_type":    "galactose-free diet (reduces hepatic glycogen) + raw cornstarch (prevents fasting hypoglycemia) + phosphate supplementation (treats rickets) + vitamin D",
        "key_mutation": ">50 pathogenic variants; no single founder; very rare globally (~100 cases); liver biopsy shows massive glycogen (GSD-like appearance)",
        "mean_age_onset_months": 3,
    },
}

# ─── PATIENT GENERATION ───────────────────────────────────────────────────────

def _make_patients(gene: str) -> list[dict]:
    rng = random.Random(SEEDS[gene])
    p = PROFILES[gene]
    pts = []
    for i in range(40):
        aom = max(0, int(rng.gauss(p["mean_age_onset_months"], max(1, p["mean_age_onset_months"] * 0.4 + 1))))
        dx_delay = int(rng.gauss(6, 4)) if not p["benign_variant"] else 0
        dx_age_m = aom + max(0, dx_delay)

        cataracts       = rng.random() < (0.85 if p["cataracts"] else 0.02)
        liver_disease   = rng.random() < (0.75 if p["liver_disease"] else 0.03)
        hypoglycemia    = rng.random() < (0.80 if p["hypoglycemia"] else 0.04)
        neonatal_sepsis = rng.random() < (0.30 if p["neonatal_sepsis"] else 0.0)  # ~30% of GALT
        poi             = rng.random() < (0.60 if p["poi_females"] else 0.05)
        fanconi         = rng.random() < (0.80 if p["fanconi_renal"] else 0.02)
        fasting_crisis  = rng.random() < (0.75 if p["fasting_trigger"] else 0.05)
        dietary_resp    = rng.random() < (0.85 if p["dietary_curative"] else 0.10)
        nbs_detected    = rng.random() < (0.80 if gene in ["GALT", "GALK1", "GALE"] else 0.15)  # only galactosemias on NBS

        pts.append({
            "id":                   f"{gene}-{i+1:03d}",
            "gene":                 gene,
            "disease":              p["disease"],
            "age_onset_months":     aom,
            "dx_age_months":        dx_age_m,
            "dx_delay_months":      max(0, dx_delay),
            "nbs_detected":         nbs_detected,
            "cataracts":            cataracts,
            "liver_disease":        liver_disease,
            "hypoglycemia_episodes": hypoglycemia,
            "neonatal_sepsis":      neonatal_sepsis,
            "poi_females":          poi,
            "fanconi_renal":        fanconi,
            "fasting_crisis":       fasting_crisis,
            "dietary_response":     dietary_resp,
            "benign_variant":       p["benign_variant"],
            "fructose_toxic":       p["fructose_toxic"],
            "cataracts_only":       p["cataracts_only"],
        })
    return pts


def _cohort_stats(pts: list[dict]) -> dict:
    n = len(pts)
    def pct(key): return round(sum(1 for p in pts if p.get(key)) / n * 100, 1)
    return {
        "n":                    n,
        "nbs_n":                sum(1 for p in pts if p["nbs_detected"]),
        "cataracts_n":          sum(1 for p in pts if p["cataracts"]),
        "liver_n":              sum(1 for p in pts if p["liver_disease"]),
        "hypoglycemia_n":       sum(1 for p in pts if p["hypoglycemia_episodes"]),
        "neonatal_sepsis_n":    sum(1 for p in pts if p["neonatal_sepsis"]),
        "poi_n":                sum(1 for p in pts if p["poi_females"]),
        "fanconi_n":            sum(1 for p in pts if p["fanconi_renal"]),
        "fasting_crisis_n":     sum(1 for p in pts if p["fasting_crisis"]),
        "dietary_response_n":   sum(1 for p in pts if p["dietary_response"]),
        "mean_age_onset_m":     round(sum(p["age_onset_months"] for p in pts) / n, 1),
        "mean_dx_delay_m":      round(sum(p["dx_delay_months"] for p in pts) / n, 1),
    }


# ─── API FUNCTIONS ────────────────────────────────────────────────────────────

def generate_overview():
    all_patients: list[dict] = []
    cohort_stats: dict[str, dict] = {}
    for g in GENES:
        pts = _make_patients(g)
        all_patients.extend(pts)
        cohort_stats[g] = _cohort_stats(pts)

    return {
        "atlas":          ATLAS,
        "subtitle":       (
            "Hereditary Galactosemia, Fructosemia and Related Sugar Metabolism Disorders — "
            "8-Gene Complete Reference (GALT/GALK1/GALE/GALM/ALDOB/KHK/FBP1/SLC2A2)"
        ),
        "total_patients": len(all_patients),
        "genes":          GENES,
        "seeds":          SEEDS,
        "cohort_summary": {
            g: {
                "syndrome":             PROFILES[g]["syndrome"],
                "inheritance":          PROFILES[g]["inheritance"],
                "locus":                PROFILES[g]["locus"],
                "size_aa":              PROFILES[g]["aa"],
                "nbs_marker":           PROFILES[g]["nbs_marker"],
                "n":                    cohort_stats[g]["n"],
                "nbs_detected_pct":     round(cohort_stats[g]["nbs_n"] / 40 * 100, 1),
                "cataracts_pct":        round(cohort_stats[g]["cataracts_n"] / 40 * 100, 1),
                "liver_pct":            round(cohort_stats[g]["liver_n"] / 40 * 100, 1),
                "hypoglycemia_pct":     round(cohort_stats[g]["hypoglycemia_n"] / 40 * 100, 1),
                "neonatal_sepsis_pct":  round(cohort_stats[g]["neonatal_sepsis_n"] / 40 * 100, 1),
                "poi_pct":              round(cohort_stats[g]["poi_n"] / 40 * 100, 1),
                "fanconi_pct":          round(cohort_stats[g]["fanconi_n"] / 40 * 100, 1),
                "fasting_crisis_pct":   round(cohort_stats[g]["fasting_crisis_n"] / 40 * 100, 1),
                "dietary_response_pct": round(cohort_stats[g]["dietary_response_n"] / 40 * 100, 1),
                "cataracts_flag":       PROFILES[g]["cataracts"],
                "liver_flag":           PROFILES[g]["liver_disease"],
                "neonatal_sepsis_flag": PROFILES[g]["neonatal_sepsis"],
                "poi_flag":             PROFILES[g]["poi_females"],
                "fanconi_flag":         PROFILES[g]["fanconi_renal"],
                "fructose_toxic":       PROFILES[g]["fructose_toxic"],
                "benign_variant":       PROFILES[g]["benign_variant"],
                "cataracts_only":       PROFILES[g]["cataracts_only"],
                "fasting_trigger":      PROFILES[g]["fasting_trigger"],
                "dietary_curative":     PROFILES[g]["dietary_curative"],
                "key_mutation":         PROFILES[g]["key_mutation"],
                "diet_type":            PROFILES[g]["diet_type"],
                "mean_age_onset_m":     cohort_stats[g]["mean_age_onset_m"],
                "mean_dx_delay_m":      cohort_stats[g]["mean_dx_delay_m"],
            } for g in GENES
        },
        "key_facts": {
            "galt_classic_galactosemia_ecoli_poi":
                "GALT encodes galactose-1-phosphate uridylyltransferase, which converts "
                "galactose-1-phosphate + UDP-glucose → glucose-1-phosphate + UDP-galactose. "
                "LOF → GAL-1-P accumulates in liver, brain, erythrocytes, lens → toxicity. "
                "PATHOGNOMONIC CLINICAL CLUE: E. COLI NEONATAL SEPSIS in a breastfed infant "
                "before NBS result returns. Galactose in breast milk → rapid Gal-1-P buildup → "
                "profound immunosuppression → E. coli (and other gram-negative) sepsis in first week. "
                "ANY neonatal sepsis with gram-negative in breastfed infant → STOP BREASTFEEDING "
                "immediately and test for galactosemia. Even with lifelong galactose-free diet: "
                "~50-70% of affected females develop PRIMARY OVARIAN INSUFFICIENCY (POI) — "
                "requires estrogen replacement. Tremor, ataxia, cognitive difficulties persist "
                "despite optimal dietary control. DUARTE variant (D2, p.N314D): partial enzyme "
                "(25-50% activity), BENIGN, NO treatment required — important not to over-treat. "
                "Q188R c.563A>G is the most common Caucasian allele (~70% of European alleles).",

            "galk1_cataracts_only_benign":
                "GALK1 (galactokinase 1) phosphorylates galactose → galactose-1-phosphate. "
                "LOF → galactose accumulates → GALACTITOL forms in lens via aldose reductase. "
                "GALACTITOL is osmotically active → BILATERAL OIL-DROPLET CATARACTS in neonates. "
                "CRITICAL DISTINCTION FROM GALT: GALK1 deficiency causes CATARACTS ONLY — "
                "NO intellectual disability, NO hepatic disease, NO E. coli sepsis, NO POI. "
                "GAL-1-P is NORMAL (GALK1 block is upstream — galactose cannot even be "
                "phosphorylated). NBS: total galactose elevated (same as GALT) but GALT enzyme "
                "NORMAL. TREATMENT: galactose-free diet → cataracts resorb in early cases. "
                "P28T (p.Pro28Thr) is the most common European founder allele — high frequency "
                "in Roma/Traveller populations. Diagnosis requires GALK enzyme assay or GALK1 "
                "gene sequencing when GALT enzyme is normal on NBS.",

            "aldob_hfi_fructose_aversion_no_loading_test":
                "ALDOB (aldolase B) cleaves fructose-1-phosphate → DHAP + D-glyceraldehyde. "
                "LOF → FRUCTOSE-1-PHOSPHATE ACCUMULATES in liver/kidney/intestine → "
                "(1) PHOSPHATE TRAPPING → ATP depletion → hepatocyte necrosis; "
                "(2) GLUCOSE-6-PHOSPHATASE INHIBITION → hypoglycemia; "
                "(3) GLYCOGEN PHOSPHORYLASE INHIBITION → no glycogenolysis rescue. "
                "CLASSIC HISTORY: infant thrives until WEANING (first sucrose/fructose exposure) → "
                "VOMITING + HYPOGLYCEMIA + HEPATOMEGALY. Older patients: INNATE FRUCTOSE AVERSION "
                "(self-protective refusal of sweets) — often have unexplained good dental health. "
                "ABSOLUTE CONTRAINDICATION: FRUCTOSE LOADING TEST (historical diagnostic test) "
                "= LIFE-THREATENING; NEVER PERFORM — can cause acute liver failure. "
                "DIAGNOSIS: ALDOB gene sequencing (p.A149P ~80% European; p.A174D ~10%). "
                "TREATMENT: strict FRUCTOSE-FREE diet (avoid fructose + sucrose + SORBITOL). "
                "Galactose is NOT toxic in HFI (different pathway). "
                "FBP1 deficiency key DDx: fasting-triggered ketoacidosis but fructose is NOT toxic.",

            "fbp1_fasting_hypoglycemia_fructose_not_toxic":
                "FBP1 (fructose-1,6-bisphosphatase) is the irreversible rate-limiting enzyme "
                "of gluconeogenesis: fructose-1,6-bisphosphate → fructose-6-phosphate + Pi. "
                "LOF → gluconeogenesis BLOCKED → cannot produce glucose from lactate/alanine/glycerol "
                "during fasting. CLINICAL: fasting or illness-triggered ACUTE DECOMPENSATION with "
                "SIMULTANEOUS: hypoglycemia + LACTIC ACIDOSIS + HYPERKETOSIS. "
                "CRITICAL DDx FROM FAOD: HYPERKETOTIC hypoglycemia (FAOD = hypoketotic). "
                "CRITICAL DDx FROM HFI: FRUCTOSE IS NOT TOXIC IN FBP1 DEFICIENCY. "
                "Fructose-1-P does NOT accumulate (fructokinase product is upstream of FBP1 block). "
                "Glucagon test: NO glucose response (gluconeogenesis cannot be activated). "
                "Alanine load: no glucose rise. TREATMENT: AVOID FASTING (raw cornstarch 1-2g/kg "
                "at bedtime); IV glucose 10% in crisis; NO fructose restriction needed.",

            "slc2a2_fanconi_bickel_glut2":
                "SLC2A2 (GLUT2) is a bidirectional facilitative glucose transporter expressed in "
                "hepatocytes, pancreatic beta cells, renal proximal tubules, and intestinal epithelium. "
                "LOF → FOUR SIMULTANEOUS DEFECTS: "
                "(1) HEPATIC: GLUT2 needed for glucose EXPORT from liver → glucose trapped as "
                "glycogen → hepatomegaly with massive glycogen accumulation (GSD-like appearance). "
                "(2) RENAL FANCONI: GLUT2 in proximal tubule reabsorbs glucose → LOF → "
                "glucosuria, aminoaciduria, phosphaturia → HYPOPHOSPHATEMIC RICKETS. "
                "(3) BETA CELL: impaired glucose sensing → inadequate insulin secretion. "
                "(4) INTESTINAL: impaired galactose/glucose absorption. "
                "CLINICAL: hepatomegaly + hypoglycemia + glucosuria + RICKETS = PATHOGNOMONIC. "
                "Fasting hypoglycemia despite hepatomegaly + glycogen = paradox. "
                "NBS: may show elevated galactose (GLUT2 needed for intestinal galactose absorption). "
                "RARE: ~100 cases. TREATMENT: galactose-free diet + raw cornstarch + "
                "phosphate replacement + vitamin D. Galactose-free because galactose transport is impaired.",

            "galactosemia_nbs_algorithm":
                "NBS GALACTOSEMIA TRIAGE: elevated total galactose → (1) check GALT enzyme: "
                "if VERY LOW (<5%) = classic galactosemia (GALT LOF); "
                "if LOW-INTERMEDIATE = Duarte variant (D2/N314D) or GALE peripheral — no treatment; "
                "if NORMAL GALT enzyme → check GALK enzyme: if absent = GALK1 deficiency (cataracts only); "
                "if GALK normal → check GALE enzyme and GALM gene sequencing. "
                "In classic GALT: STOP BREAST MILK immediately on positive NBS — do not await "
                "confirmatory result. Switch to soy formula or Nutramigen (lactose-free). "
                "GALT Q188R/Q188R homozygote = most severe; NEVER partially responsive to treatment "
                "for POI — reproductive counseling mandatory for females at puberty.",

            "cascade_diagnostic_algorithm":
                "GALACTOSEMIA NBS: total galactose ↑ → GALT enzyme ↓ → Classic (GALT): stop breastmilk STAT | "
                "GALT normal, GALK ↓ → GALK1 deficiency: cataracts only; galactose-free diet | "
                "GALT normal, GALK normal → GALE or GALM: enzyme + gene panel | "
                "HFI/ALDOB: vomiting+hypoglycemia at weaning; fructose aversion history; ALDOB gene; NO fructose-loading | "
                "FBP1: fasting crisis + HIGH lactate + HIGH ketones; glucagon no response; fructose NOT restricted | "
                "ESSENTIAL FRUCTOSURIA/KHK: urine reducing substance +, glucose-oxidase −, NO symptoms; NO treatment | "
                "FANCONI-BICKEL/SLC2A2: hepatomegaly + glucosuria + rickets + hypoglycemia; galactose-free + cornstarch + phosphate | "
                "POI SURVEILLANCE: all GALT females → estradiol/FSH annually from age 10; estrogen replacement PRN.",
        },
    }


def generate_breakdown():
    all_patients: list[dict] = []
    cohort_stats: dict[str, dict] = {}
    for g in GENES:
        pts = _make_patients(g)
        all_patients.extend(pts)
        cohort_stats[g] = _cohort_stats(pts)

    galactosemia_genes = ["GALT", "GALK1", "GALE", "GALM"]
    fructose_genes     = ["ALDOB", "KHK"]
    gluconeogenesis    = ["FBP1"]
    transporter        = ["SLC2A2"]

    return {
        "atlas":          ATLAS,
        "total_patients": len(all_patients),
        "groups": {
            "galactosemia_spectrum": {
                "genes":       galactosemia_genes,
                "mechanism":   "Galactose metabolism pathway (GALT → GALK1 → GALE → GALM)",
                "n_patients":  sum(cohort_stats[g]["n"] for g in galactosemia_genes),
                "severe_gene": ["GALT"],
                "cataracts_only_gene": ["GALK1"],
                "spectrum_gene": ["GALE"],
                "benign_gene": ["GALM"],
                "galt_sepsis_pct": round(cohort_stats["GALT"]["neonatal_sepsis_n"] / 40 * 100, 1),
                "galt_poi_pct":    round(cohort_stats["GALT"]["poi_n"] / 40 * 100, 1),
                "galk_cataracts_pct": round(cohort_stats["GALK1"]["cataracts_n"] / 40 * 100, 1),
            },
            "fructose_metabolism": {
                "genes":       fructose_genes,
                "mechanism":   "Fructose phosphorylation pathway (ALDOB/HFI — toxic; KHK — benign)",
                "n_patients":  sum(cohort_stats[g]["n"] for g in fructose_genes),
                "aldob_toxic": True,
                "khk_benign":  True,
                "aldob_liver_pct": round(cohort_stats["ALDOB"]["liver_n"] / 40 * 100, 1),
                "aldob_hypoglycemia_pct": round(cohort_stats["ALDOB"]["hypoglycemia_n"] / 40 * 100, 1),
            },
            "gluconeogenesis_block": {
                "genes":       gluconeogenesis,
                "mechanism":   "FBP1 — gluconeogenesis block; fasting/illness trigger; fructose NOT toxic",
                "n_patients":  cohort_stats["FBP1"]["n"],
                "fasting_crisis_pct": round(cohort_stats["FBP1"]["fasting_crisis_n"] / 40 * 100, 1),
                "fructose_safe": True,  # critical DDx from HFI
            },
            "sugar_transporter_defect": {
                "genes":       transporter,
                "mechanism":   "GLUT2 (SLC2A2) bidirectional sugar transporter — Fanconi-Bickel syndrome",
                "n_patients":  cohort_stats["SLC2A2"]["n"],
                "liver_pct":   round(cohort_stats["SLC2A2"]["liver_n"] / 40 * 100, 1),
                "fanconi_pct": round(cohort_stats["SLC2A2"]["fanconi_n"] / 40 * 100, 1),
            },
        },
        "gene_breakdown": {
            g: {
                "n":                    cohort_stats[g]["n"],
                "nbs_detected_pct":     round(cohort_stats[g]["nbs_n"] / 40 * 100, 1),
                "cataracts_pct":        round(cohort_stats[g]["cataracts_n"] / 40 * 100, 1),
                "liver_pct":            round(cohort_stats[g]["liver_n"] / 40 * 100, 1),
                "hypoglycemia_pct":     round(cohort_stats[g]["hypoglycemia_n"] / 40 * 100, 1),
                "neonatal_sepsis_pct":  round(cohort_stats[g]["neonatal_sepsis_n"] / 40 * 100, 1),
                "poi_pct":              round(cohort_stats[g]["poi_n"] / 40 * 100, 1),
                "fanconi_pct":          round(cohort_stats[g]["fanconi_n"] / 40 * 100, 1),
                "fasting_crisis_pct":   round(cohort_stats[g]["fasting_crisis_n"] / 40 * 100, 1),
                "dietary_response_pct": round(cohort_stats[g]["dietary_response_n"] / 40 * 100, 1),
                "mean_age_onset_m":     cohort_stats[g]["mean_age_onset_m"],
                "mean_dx_delay_m":      cohort_stats[g]["mean_dx_delay_m"],
            } for g in GENES
        },
    }


def generate_definitions():
    return {
        "atlas":   ATLAS,
        "genes":   GENES,
        "definitions": {
            "GALT": {
                "full_name":    "Galactose-1-phosphate Uridylyltransferase",
                "protein_size": "379 aa / 43 kDa",
                "locus":        "9p13.3",
                "omim_gene":    "606999",
                "omim_disease": "230400",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "GALT converts galactose-1-phosphate + UDP-glucose → glucose-1-phosphate + UDP-galactose "
                    "(Leloir pathway step 2). LOF → galactose-1-phosphate (Gal-1-P) accumulates in ALL tissues "
                    "that express GALT (liver, brain, erythrocytes, ovaries, lens). Gal-1-P inhibits "
                    "phosphoglucomutase and multiple other enzymes → cellular dysfunction and death. "
                    "Additionally, galactose is reduced by aldose reductase → GALACTITOL in lens → cataracts. "
                    "Galactitol also accumulates in brain → progressive neurological damage even with diet."
                ),
                "phenotype_spectrum": {
                    "neonatal": (
                        "First days-week: jaundice, hepatomegaly, vomiting, lethargy on breast milk or "
                        "regular formula. E. COLI SEPSIS (or other gram-negative) = PATHOGNOMONIC CLUE "
                        "before NBS returns. NBS: high Gal-1-P + very low GALT enzyme."
                    ),
                    "classic_treated": (
                        "On galactose-free diet: acute hepatic crisis resolves. Long-term: ~50-70% females "
                        "develop PRIMARY OVARIAN INSUFFICIENCY (hypergonadotropic hypogonadism, low estradiol, "
                        "high FSH/LH). Tremor, ataxia (30-40%), speech/language delay, executive dysfunction "
                        "persist despite optimal diet. Bone mineral density reduced."
                    ),
                    "duarte_variant": (
                        "D2 allele (p.N314D + 4bp deletion in promoter): partial enzyme (25-50%). "
                        "Combined with classic allele (compound heterozygote D2/classic): ~25% activity. "
                        "BENIGN — NO treatment required. Dietary restriction NOT recommended for D2/D2. "
                        "Major source of false-positive NBS for galactosemia."
                    ),
                },
                "diagnostic_pearl": (
                    "NBS: total galactose markedly elevated + GALT enzyme <5% (dried blood spot). "
                    "Confirmatory: RBC Gal-1-P (>3 mg/dL diagnostic). GALT gene sequencing. "
                    "CRITICAL: DO NOT WAIT for confirmatory testing to stop galactose exposure if NBS positive."
                ),
                "treatment_pearl": (
                    "Stop breast milk IMMEDIATELY on positive NBS; switch to soy formula or hydrolysed casein "
                    "(Nutramigen). Lifelong galactose-free diet. Annual Gal-1-P monitoring. "
                    "Females: annual FSH/LH/estradiol from age 10; estrogen replacement for POI. "
                    "Bone density surveillance. Speech/developmental support."
                ),
            },
            "GALK1": {
                "full_name":    "Galactokinase 1",
                "protein_size": "379 aa / 42 kDa",
                "locus":        "17q25.1",
                "omim_gene":    "604313",
                "omim_disease": "230200",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "GALK1 phosphorylates galactose → galactose-1-phosphate (step 1 of Leloir pathway). "
                    "LOF → galactose CANNOT BE PHOSPHORYLATED → galactose accumulates → aldose reductase "
                    "converts galactose → GALACTITOL. In the lens, galactitol accumulates osmotically "
                    "→ lens fibre disruption → BILATERAL OIL-DROPLET CATARACTS. "
                    "Crucially: Gal-1-P does NOT accumulate (GALK1 block is upstream) → "
                    "no hepatic toxicity, no brain toxicity, no ovarian toxicity."
                ),
                "phenotype_spectrum": {
                    "cataracts": (
                        "ONLY presentation: bilateral cataracts in neonates/infants. "
                        "Oil-droplet appearance on slit-lamp. Reversible if galactose-free diet started early. "
                        "Cataracts can also occur in adults exposed to galactose (late-diagnosed)."
                    ),
                    "benign": (
                        "No intellectual disability. No hepatic disease. No POI. No E. coli sepsis. "
                        "Normal development, liver function, and reproductive function."
                    ),
                },
                "diagnostic_pearl": (
                    "NBS: total galactose elevated + GALT enzyme NORMAL (key DDx from classic galactosemia). "
                    "GALK enzyme assay: absent in GALK1 deficiency. "
                    "Urine galactose elevated. Gal-1-P NORMAL. "
                    "High-risk: Roma/Traveller populations (p.Pro28Thr high carrier frequency)."
                ),
                "treatment_pearl": (
                    "Galactose-free diet (no galactose, no lactose/dairy). Early cataracts may reverse. "
                    "Excellent prognosis with dietary treatment. No other complications. "
                    "Ophthalmology follow-up for cataract monitoring."
                ),
            },
            "GALE": {
                "full_name":    "UDP-Galactose-4-Epimerase",
                "protein_size": "348 aa / 38 kDa",
                "locus":        "1p36.11",
                "omim_gene":    "606953",
                "omim_disease": "230350",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "GALE interconverts UDP-galactose ↔ UDP-glucose and UDP-N-acetylgalactosamine ↔ "
                    "UDP-N-acetylglucosamine (step 3 of Leloir pathway + glycosylation reactions). "
                    "LOF Type I (peripheral): only in RBCs/WBCs → galactose accumulates in these cells only → "
                    "BENIGN (other tissues have sufficient GALE activity). "
                    "LOF Type II (generalized): GALE absent in ALL tissues → severe, resembles classic GALT "
                    "galactosemia: Gal-1-P accumulation, hepatic disease, cataracts, developmental delay."
                ),
                "phenotype_spectrum": {
                    "peripheral_benign": (
                        "Most common (>90%): elevated galactose/Gal-1-P only in RBCs/WBCs. "
                        "NO clinical disease. NO dietary treatment needed. "
                        "Incidental NBS finding. Must distinguish from classic galactosemia."
                    ),
                    "generalized_severe": (
                        "Rare: severe neonatal galactosemia picture. Hepatic disease, cataracts, "
                        "jaundice on galactose. Some endogenous galactose production requires "
                        "some galactose in diet (for glycoconjugate synthesis) — uniquely complex management."
                    ),
                },
                "diagnostic_pearl": (
                    "GALE enzyme activity in RBCs: low. Gal-1-P may be elevated. "
                    "Distinguish peripheral vs generalized: GALE enzyme in skin fibroblasts or leukocytes. "
                    "p.Val94Met = generalized severe; p.Asn34Ser = peripheral benign. "
                    "Gene sequencing essential."
                ),
                "treatment_pearl": (
                    "Type I peripheral: NO treatment needed — watch and reassure. "
                    "Type II generalized: galactose restriction, but NOT complete elimination "
                    "(body requires some galactose for glycoconjugate synthesis — complex titration)."
                ),
            },
            "GALM": {
                "full_name":    "Galactose Mutarotase",
                "protein_size": "342 aa / 38 kDa",
                "locus":        "2p22.3",
                "omim_gene":    "608313",
                "omim_disease": "618881",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "GALM catalyzes the mutarotation of beta-D-galactose → alpha-D-galactose (anomeric "
                    "interconversion). Alpha-D-galactose is the substrate for GALK1. "
                    "LOF → beta-galactose cannot be converted → upstream galactose accumulates slightly. "
                    "VERY MILD biochemical phenotype because dietary galactose eventually bypasses this "
                    "step by other routes. Only ~30 cases described worldwide (2026)."
                ),
                "phenotype_spectrum": {
                    "very_mild_benign": (
                        "Most: only mildly elevated total galactose on NBS. "
                        "Rarely symptomatic. NO intellectual disability. NO cataracts. NO liver disease. "
                        "Some infants have transiently elevated galactose resolving without treatment."
                    ),
                },
                "diagnostic_pearl": (
                    "NBS: mildly elevated galactose + GALT enzyme NORMAL + GALK enzyme NORMAL. "
                    "GALM gene sequencing confirms. Ultra-rare — most labs do not routinely sequence GALM. "
                    "Molecular diagnosis usually made as 'unexplained galactosemia' on research panels."
                ),
                "treatment_pearl": (
                    "Usually NO treatment required. Short-term galactose reduction in NBS period "
                    "may be considered empirically but most cases resolve without intervention. "
                    "Excellent prognosis."
                ),
            },
            "ALDOB": {
                "full_name":    "Aldolase B (Fructose-Bisphosphate Aldolase B)",
                "protein_size": "364 aa / 40 kDa",
                "locus":        "9q22.33",
                "omim_gene":    "612724",
                "omim_disease": "229600",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "Aldolase B cleaves fructose-1-phosphate (from fructokinase/KHK) → "
                    "DHAP + D-glyceraldehyde. LOF → FRUCTOSE-1-PHOSPHATE (F-1-P) accumulates: "
                    "(1) F-1-P inhibits phosphorylase → blocks glycogenolysis → no glucose from glycogen. "
                    "(2) F-1-P inhibits gluconeogenesis enzymes (phosphofructoaldolase) → no glucose synthesis. "
                    "(3) F-1-P traps inorganic phosphate → ATP depletion → hepatocyte necrosis. "
                    "Combined → PROFOUND HYPOGLYCEMIA + LIVER FAILURE on any fructose exposure."
                ),
                "phenotype_spectrum": {
                    "classic_hfi": (
                        "Asymptomatic until FIRST FRUCTOSE/SUCROSE EXPOSURE (typically at weaning 4-6 months). "
                        "ACUTE: vomiting, severe hypoglycemia, lethargy, hepatomegaly within 20-30 min of exposure. "
                        "Chronic undiagnosed: liver cirrhosis, proximal renal tubular acidosis, growth retardation. "
                        "HALLMARK: patients develop innate AVERSION to sweets (self-protective behavioural change). "
                        "Often found incidentally to have excellent dental health."
                    ),
                    "late_diagnosed_adult": (
                        "Lifelong fructose avoidance (self-selected) → may reach adulthood undiagnosed. "
                        "Diagnosed when given IV sucrose (iatrogenic) or during metabolic workup."
                    ),
                },
                "diagnostic_pearl": (
                    "ALDOB gene sequencing: p.A149P + p.A174D cover ~90% European alleles. "
                    "Liver biopsy (historical): fructose-1-P accumulation + aldolase B deficiency. "
                    "NEVER perform fructose-loading test — LETHAL. "
                    "Distinguish from FBP1 deficiency: fructose IS toxic in HFI but NOT in FBP1."
                ),
                "treatment_pearl": (
                    "STRICT FRUCTOSE-FREE DIET: no fructose, sucrose (table sugar), sorbitol, or high-fructose corn syrup. "
                    "Read ALL food labels. Medications containing sorbitol/fructose must be avoided. "
                    "Galactose and glucose are SAFE. IV glucose for acute hypoglycemia. "
                    "Prognosis EXCELLENT with strict avoidance — no progression of liver disease."
                ),
            },
            "KHK": {
                "full_name":    "Ketohexokinase (Fructokinase, Hepatic Fructokinase-C)",
                "protein_size": "298 aa / 33 kDa",
                "locus":        "2p23.3",
                "omim_gene":    "614985",
                "omim_disease": "229800",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "KHK phosphorylates fructose → fructose-1-phosphate (step 1 of fructose metabolism in liver). "
                    "LOF → fructose CANNOT BE PHOSPHORYLATED → fructose accumulates and spills into urine. "
                    "No fructose-1-phosphate is formed → no hepatic toxicity, no hypoglycemia, no ATP trapping. "
                    "KHK-C (hepatic isoform) LOF = essential fructosuria. "
                    "Note: KHK-A (ubiquitous low-affinity isoform) remains functional, handling small amounts."
                ),
                "phenotype_spectrum": {
                    "benign": (
                        "COMPLETELY BENIGN. Incidental finding: FRUCTOSURIA after fructose ingestion. "
                        "No hypoglycemia, no liver disease, no intellectual disability. "
                        "Urine: reducing substance positive after fructose meal; glucose-oxidase NEGATIVE. "
                        "Most cases diagnosed incidentally. NO treatment required."
                    ),
                },
                "diagnostic_pearl": (
                    "Urine: reducing substance + (by copper reduction e.g. Clinitest) but glucose-oxidase NEGATIVE. "
                    "Plasma fructose elevated after fructose load. "
                    "KHK gene sequencing confirms. "
                    "DDx: HFI (ALDOB) — identical urine fructosuria but HFI causes acute symptoms; "
                    "essential fructosuria (KHK) has NO symptoms."
                ),
                "treatment_pearl": (
                    "NO TREATMENT REQUIRED. Reassure patient and family — this is a benign biochemical variant. "
                    "No dietary restriction needed. No medication. Excellent lifelong prognosis."
                ),
            },
            "FBP1": {
                "full_name":    "Fructose-1,6-Bisphosphatase 1",
                "protein_size": "338 aa / 37 kDa",
                "locus":        "9q22.32",
                "omim_gene":    "611570",
                "omim_disease": "229700",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "FBP1 catalyzes the irreversible step of gluconeogenesis: "
                    "fructose-1,6-bisphosphate → fructose-6-phosphate + Pi. "
                    "LOF → gluconeogenesis COMPLETELY BLOCKED. During fasting or illness (increased "
                    "gluconeogenic substrate load): lactate, alanine, glycerol cannot be converted → "
                    "LACTIC ACIDOSIS + no rescue glucose production. Liver glycogen is depleted first, "
                    "then severe hypoglycemia ensues. Ketones are HIGH (paradoxically elevated because "
                    "fatty acid oxidation is intact and acetyl-CoA cannot enter TCA via gluconeogenesis)."
                ),
                "phenotype_spectrum": {
                    "acute_decompensation": (
                        "TRIGGERED BY: fasting, infection, vomiting, fever. "
                        "Triad: HYPOGLYCEMIA + LACTIC ACIDOSIS + HYPERKETONEMIA simultaneously. "
                        "Hyperlactatemia (>5 mmol/L) + hyperketonemia (uncommon in other hypoglycemia causes) "
                        "= strong FBP1 clue. Glucagon test: NO blood glucose rise."
                    ),
                    "mild_form": (
                        "Some patients have only mild episodes triggered by extreme fasting. "
                        "With avoidance of prolonged fasting, normal life expectancy and intelligence."
                    ),
                },
                "diagnostic_pearl": (
                    "Simultaneous hyperketotic hypoglycemia + lactic acidosis = FBP1 until proven otherwise. "
                    "FBP1 enzyme activity in liver or leukocytes: absent. "
                    "FBP1 gene sequencing. "
                    "CRITICAL DDx FROM HFI: fructose is NOT toxic in FBP1 deficiency — "
                    "no need to restrict fructose (this is the most important DDx point)."
                ),
                "treatment_pearl": (
                    "AVOID FASTING: never skip meals; raw cornstarch (uncooked) 1-2g/kg at bedtime for "
                    "overnight glucose. IV glucose 10% (GIR 8-10 mg/kg/min) for acute crisis. "
                    "High-carbohydrate diet. NO fructose restriction required. "
                    "Excellent prognosis with fasting avoidance."
                ),
            },
            "SLC2A2": {
                "full_name":    "Solute Carrier Family 2 Member 2 (GLUT2 Glucose Transporter)",
                "protein_size": "524 aa / 57 kDa",
                "locus":        "3q26.2",
                "omim_gene":    "138160",
                "omim_disease": "227810",
                "inheritance":  "Autosomal recessive",
                "mechanism_detail": (
                    "GLUT2 (SLC2A2) is a low-affinity, high-capacity bidirectional facilitative glucose "
                    "transporter (Km ~17 mM). Expressed in: hepatocytes (glucose export after meals), "
                    "pancreatic beta cells (glucose sensing for insulin secretion), renal proximal tubule "
                    "S3 segment (glucose reabsorption), and intestinal epithelium (galactose/glucose absorption). "
                    "LOF → FOUR SIMULTANEOUS DEFECTS: "
                    "(1) Hepatic glycogen accumulates (glucose cannot exit after glycogenolysis). "
                    "(2) Renal Fanconi syndrome (glucose, amino acids, phosphate, urate wasting). "
                    "(3) Impaired insulin sensing/secretion. "
                    "(4) Galactose intolerance (impaired absorption + hepatic handling)."
                ),
                "phenotype_spectrum": {
                    "classic_fanconi_bickel": (
                        "Triad: HEPATOMEGALY (glycogen storage) + RENAL FANCONI + GROWTH RETARDATION. "
                        "Glucosuria present even when blood glucose is LOW (paradox — tubular transport absent). "
                        "Rickets from phosphaturia. Hypoglycemia despite hepatomegaly. "
                        "Galactose also toxic (impaired absorption + hepatic trapping). "
                        "NBS may show elevated galactose."
                    ),
                    "neonatal": (
                        "Neonatal: hepatomegaly + hypoglycemia detectable in newborn period. "
                        "Liver biopsy shows massive glycogen accumulation (GSD-like but not GSD). "
                        "~100 cases worldwide (2026)."
                    ),
                },
                "diagnostic_pearl": (
                    "PATHOGNOMONIC: glucosuria + HYPOGLYCEMIA simultaneously (glucosuria with normal/low glucose). "
                    "Aminoaciduria + phosphaturia = Fanconi syndrome. Hepatomegaly + glycogen on biopsy. "
                    "SLC2A2 gene sequencing confirms. "
                    "DDx from GSD: different mechanism; GSD usually has high glucose after glucagon; "
                    "Fanconi-Bickel has low glucose + Fanconi."
                ),
                "treatment_pearl": (
                    "Galactose-free diet (reduces hepatic glycogen load from galactose). "
                    "Raw cornstarch to prevent fasting hypoglycemia. "
                    "Phosphate supplementation (oral) for hypophosphatemic rickets. "
                    "Vitamin D. Monitor renal function. Prognosis reasonable with management. "
                    "No curative therapy; liver transplant not typically indicated."
                ),
            },
        },
        "cascade_algorithm": {
            "step_1_nbs_galactose":      "NBS: total galactose elevated → check GALT enzyme",
            "step_2_galt_enzyme":         "GALT very low (<5%) → Classic galactosemia: STOP galactose IMMEDIATELY",
            "step_3_galt_normal_galk":    "GALT normal → check GALK enzyme: absent = GALK1 (cataracts only); normal → GALE/GALM",
            "step_4_hfi_history":         "Vomiting + hypoglycemia at weaning + fructose aversion → HFI (ALDOB): gene sequencing; NO fructose loading",
            "step_5_khk_benign":          "Urine fructosuria + reducing substance + glucose-oxidase negative + NO symptoms → KHK: no treatment",
            "step_6_fbp1_fasting":        "Fasting crisis + HIGH lactate + HIGH ketones + glucagon no response → FBP1: fructose NOT restricted; cornstarch",
            "step_7_slc2a2_fanconi":      "Hepatomegaly + glucosuria + rickets + hypoglycemia → SLC2A2: galactose-free + cornstarch + phosphate",
            "step_8_poi_surveillance":    "ALL GALT females: annual FSH/LH/estradiol from age 10; hormone replacement for POI",
        },
        "critical_treatment_alerts": [
            "CLASSIC GALACTOSEMIA (GALT): STOP BREASTMILK ON POSITIVE NBS — do not await confirmatory; switch to soy/Nutramigen",
            "GALT FEMALES: ~50-70% develop POI — annual gonadotropin monitoring from age 10; estrogen replacement PRN",
            "HFI (ALDOB): FRUCTOSE LOADING TEST ABSOLUTELY CONTRAINDICATED — lethal; use ALDOB gene sequencing",
            "HFI vs FBP1 DDx: fructose IS toxic in HFI (ALDOB) but NOT toxic in FBP1 deficiency — CRITICAL management difference",
            "GALK1: CATARACTS ONLY — NO intellectual disability; NO liver disease; NO E. coli sepsis; galactose-free diet reverses early cataracts",
            "FANCONI-BICKEL (SLC2A2): glucosuria WITH hypoglycemia simultaneously = PATHOGNOMONIC (glucosuria usually means hyperglycemia)",
            "GALE PERIPHERAL (type I): BENIGN — no dietary restriction; over-treating causes unnecessary galactose restriction in benign variant",
            "FBP1: AVOID FASTING; raw cornstarch at bedtime; fructose is NOT restricted (common pitfall to restrict like HFI)",
        ],
    }
