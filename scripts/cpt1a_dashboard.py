#!/usr/bin/env python3
"""CPT1A (Carnitine Palmitoyltransferase IA Deficiency) Dashboard.

CPT1A gene encodes Carnitine Palmitoyltransferase 1A:
  CPT1A: 773 aa; 11q13.3; outer mitochondrial membrane; liver-specific isoform
  CPT1B: 772 aa; 22q13.33; heart / skeletal muscle isoform (NOT affected in CPT1A deficiency)
  CPT1C: 802 aa; 19q13.33; brain isoform (NOT affected in CPT1A deficiency)

  CPT1A CATALYSES (outer mitochondrial membrane — CARNITINE SHUTTLE STEP 1):
    Long-chain acyl-CoA + L-Carnitine → Long-chain acylcarnitine + CoA-SH
    (Transfers long-chain acyl group from CoA to carnitine for IMM crossing)

  This is the RATE-LIMITING STEP of long-chain fatty acid beta-oxidation.
  Without CPT1A → long-chain fatty acids CANNOT cross the inner mitochondrial membrane
  → long-chain beta-oxidation BLOCKED in liver → hypoketotic hypoglycaemia.

CPT1A METABOLIC BLOCK — CARNITINE SHUTTLE FAILURE:
  CPT1A LOF → long-chain acyl-CoA CANNOT be transferred to carnitine
  → free carnitine ACCUMULATES (not consumed) → C0 ELEVATED (↑↑ free carnitine)
  → long-chain acylcarnitines NOT generated → C16, C18:1 NORMAL or LOW
  → long-chain FA cannot enter mitochondria → beta-oxidation BLOCKED
  → ketone body synthesis FAILS → HYPOKETOTIC HYPOGLYCAEMIA
  → energy crisis → secondary hyperammonaemia (CPS1 impaired)

KEY NBS MARKER — THE INVERTED PROFILE (HIGH C0, LOW C16):
  C0 (free carnitine)  ELEVATED ≥60 μmol/L   — PRIMARY NBS MARKER (≥2× ULN)
  C0/(C16+C18) ratio   ELEVATED >40           — HIGHLY SPECIFIC discriminator
  C0/C16 ratio         ELEVATED               — supports diagnosis
  C16 (palmitoylcarnitine)   NORMAL/LOW       — NOT elevated (unlike VLCAD/LCHAD)
  C18:1 (oleoylcarnitine)    NORMAL/LOW       — NOT elevated
  C8  (octanoylcarnitine)    NORMAL            — KEY NEGATIVE vs MCAD
  C14:1                      NORMAL            — KEY NEGATIVE vs VLCAD
  C3  (propionylcarnitine)   NORMAL            — KEY NEGATIVE vs PA/MMA
  → OPPOSITE of most FAO disorders (most have LOW C0 + HIGH acylcarnitines)

URINE ORGANIC ACIDS (CPT1A):
  Dicarboxylic acids (mild) — adipic, suberic, sebacic acids (omega oxidation overflow)
  No pathognomonic urine marker unlike MCAD (no glycine conjugates)
  Ketones: ABSENT or minimal (inappropriately low for degree of hypoglycaemia)

CLINICAL FEATURES:
  1. HYPOKETOTIC HYPOGLYCAEMIA — hallmark; fasting/illness trigger
  2. HYPERAMMONAEMIA — UNIQUE AMONG FAO DISORDERS (50–500 μmol/L)
     Mechanism: long-chain FA accumulate → direct CPS1 inhibition via NAGS
     → secondary urea cycle impairment → hyperammonemia
  3. HEPATOMEGALY + hepatic dysfunction (transaminitis)
  4. RENAL TUBULAR ACIDOSIS (tubular dysfunction; uncommon but recognised)
  5. NO CARDIOMYOPATHY — EXAM TRAP: CPT1B serves heart; CPT1A deficiency spares heart
  6. NO RHABDOMYOLYSIS — CPT1A is liver isoform; muscle uses CPT1B (intact)
  7. ENCEPHALOPATHY during crisis (hypoglycaemia-driven)
  8. Sudden death if crisis unrecognised

KEY NEGATIVES (exam discriminators):
  - C8 NORMAL — KEY NEGATIVE vs MCAD (MCAD elevates C8)
  - C14:1 NORMAL — KEY NEGATIVE vs VLCAD (VLCAD elevates C14:1)
  - C16 NORMAL/LOW — KEY NEGATIVE vs VLCAD/LCHAD (those elevate C16)
  - NO cardiomyopathy — KEY NEGATIVE vs VLCAD (VLCAD has cardiac disease)
  - NO rhabdomyolysis — KEY NEGATIVE vs VLCAD/LCHAD
  - C16-OH NORMAL — KEY NEGATIVE vs LCHAD (LCHAD elevates 3-OH acylcarnitines)

TREATMENT:
  1. AVOID FASTING — ABSOLUTE CI (triggers lipolysis → long-chain FA released but cannot be oxidised)
  2. MCT OIL — THERAPEUTIC (Level A): Medium-chain FA (C8/C10) do NOT require CPT1A
     → enter mitochondria directly → MCAD oxidises → ketones generated → hypoglycaemia prevented
  3. CORNSTARCH — Level A: Sustained glucose release; prevents nocturnal hypoglycaemia
  4. IV GLUCOSE 10% — Level A: First-line emergency treatment
  5. KD (Ketogenic Diet) — ABSOLUTE CI: Requires long-chain fat for ketone generation;
     CPT1A blocks long-chain FA from entering mitochondria → WORSENS disease
  6. VPA — HIGH RISK: Inhibits FAO, depletes carnitine; avoid in crisis
  7. L-CARNITINE — NOT INDICATED ROUTINELY (C0 already elevated; excess may worsen)
     Exception: if secondary depletion documented on monitoring

ARCTIC FOUNDER VARIANT — p.Pro479Leu (c.1436C>T):
  - CPT1A gene exon 15; proline → leucine at position 479
  - VERY HIGH allele frequency in Arctic/subarctic populations:
      Inuit (Canada/Greenland): up to 85% allele frequency in some communities
      Alaska Native: 40–70% allele frequency
      First Nations (northern Canada): 15–30%
  - Homozygous p.Pro479Leu individuals: ELEVATED C0 on NBS
  - Phenotype: MILD (hypomorphic) — most homozygotes asymptomatic or mildly affected
  - CONTROVERSY: Whether p.Pro479Leu is truly pathogenic or a population-adapted variant
    (some evidence it confers metabolic advantage in cold climates — ketosis suppression?)
  - Clinical recommendation: avoid fasting, cold stress; MCT supplementation; monitor glucose
  - 30–40% residual enzyme activity retained with p.Pro479Leu (vs complete LOF = 0%)
  - NOT classified as benign polymorphism — still treated with dietary precautions in newborn period

OTHER COMMON VARIANTS:
  p.Thr735Ile — severe; North America; complete LOF
  p.Arg160Gln — European; severe
  p.Asp100Asn — Asian
  Splice variants — various; null alleles
  Large deletions — rare; complete absence CPT1A

METABOLIC PHYSIOLOGY — CPT SYSTEM:
  CARNITINE SHUTTLE (inner mitochondrial membrane transport):
  Step 1 (outer IMM — CPT1A liver / CPT1B heart+muscle):
    Long-chain acyl-CoA + Carnitine → Long-chain acylcarnitine + CoA-SH
    [CPT1A block = here; CPT1A deficiency = Step 1 failure in liver]
  Step 2 (inner IMM — CACT = SLC25A20 translocase):
    Long-chain acylcarnitine (cytosol) ↔ Carnitine (mitochondria) [antiport]
  Step 3 (mitochondrial matrix — CPT2):
    Long-chain acylcarnitine + CoA-SH → Long-chain acyl-CoA + Carnitine
    [CPT2 deficiency = Step 3 failure; very different disease from CPT1A]
  Steps 4–7 (matrix): Beta-oxidation proper (VLCAD→HADHA→HADHB→ACAT1 for long chain)

  CPT1A DEFICIENCY:  C0 ↑↑, C16/C18 normal/low, hypoketotic hypoglycaemia, hyperammonemia, no cardiac
  CACT DEFICIENCY:   C16 ↑↑, C18:1 ↑, cardiomyopathy, neonatal fatal
  CPT2 DEFICIENCY:   C16 ↑↑, C18:1 ↑, rhabdomyolysis (mild form), cardiomyopathy (severe form)
  → KEY EXAM DIFFERENTIATOR: HIGH C0 = CPT1A; HIGH C16 = CACT/CPT2/VLCAD/LCHAD

HIGHEST-YIELD EXAM FACTS:
  1. C0 ELEVATED = PRIMARY MARKER — opposite of MCAD/VLCAD/LCHAD (those have LOW C0)
  2. C0/(C16+C18) ratio >40 = HIGHLY SPECIFIC for CPT1A deficiency
  3. HYPERAMMONEMIA — unique among FAO disorders (CPT1A only)
  4. NO cardiomyopathy — CPT1A spares heart (CPT1B serves heart)
  5. NO rhabdomyolysis — CPT1A is liver-specific isoform
  6. MCT IS THERAPEUTIC — medium-chain FA bypass CPT1A; enters via direct IMM diffusion
  7. FASTING ABSOLUTE CI + KD ABSOLUTE CI
  8. Arctic founder p.Pro479Leu — very common in Inuit/Alaska Native populations
  9. VPA HIGH RISK (avoid)
  10. L-carnitine NOT routinely indicated (C0 already high; risk of toxicity from acylcarnitines)
  11. Cornstarch therapy for nocturnal hypoglycaemia prevention
  12. Renal tubular acidosis — rare but distinguishing feature (not seen in MCAD/VLCAD)

OMIM Disease: #255120 (CPT deficiency, hepatic, type IA)
OMIM Gene:    CPT1A *600528
Inheritance:  Autosomal Recessive (AR), biallelic LOF
Prevalence:   ~1:750,000 (general); up to 1:40 in some Arctic communities (Arctic founder variant)
Locus:        11q13.3
"""

import random

SEED = 279
random.seed(SEED)

# ── Variant table ─────────────────────────────────────────────────────────────
VARIANTS = [
    {"variant": "p.Pro479Leu (c.1436C>T)", "freq": 35, "domain": "Exon 15 transmembrane",
     "phenotype": "Mild / Arctic founder; hypomorphic; 30-40% residual activity",
     "note": "Up to 85% allele frequency in Inuit; may be population-adapted hypomorph; mostly asymptomatic if fast avoided"},
    {"variant": "p.Thr735Ile", "freq": 15, "domain": "C-terminal regulatory domain",
     "phenotype": "Severe; North American; complete LOF",
     "note": "Threonine at CPT1A active-site regulatory region; no residual activity"},
    {"variant": "p.Arg160Gln", "freq": 10, "domain": "Catalytic domain",
     "phenotype": "Severe; European; complete LOF",
     "note": "Arginine 160 in catalytic core; essential for carnitine binding; null functional"},
    {"variant": "p.Asp100Asn", "freq": 8, "domain": "N-terminal catalytic region",
     "phenotype": "Moderate-severe; Asian populations; partial residual activity",
     "note": "Asp100 coordinates catalytic mechanism; some residual acyltransferase function"},
    {"variant": "Splice c.IVS6+1G>A", "freq": 8, "domain": "Exon 6 splice donor",
     "phenotype": "Severe; null allele; exon 6 skipping",
     "note": "Null allele; complete absence of functional CPT1A transcript"},
    {"variant": "p.Arg217Trp", "freq": 6, "domain": "Outer membrane anchoring domain",
     "phenotype": "Severe; disrupts IMM anchoring of CPT1A",
     "note": "Outer membrane tethering lost; enzyme cannot localise to IMM contact site"},
    {"variant": "Exon 9-10 deletion", "freq": 5, "domain": "Catalytic domain deletion",
     "phenotype": "Severe; complete absence of catalytic core",
     "note": "Large deletion; no residual activity; neonatal severe phenotype"},
    {"variant": "p.Val368Leu", "freq": 5, "domain": "Carnitine-binding pocket",
     "phenotype": "Moderate; carnitine affinity reduced ~80%",
     "note": "Carnitine-binding pocket disrupted; acyl group transfer severely reduced"},
    {"variant": "p.His473Tyr", "freq": 4, "domain": "Malonyl-CoA regulatory site",
     "phenotype": "Unusual: partial loss + altered malonyl-CoA sensitivity",
     "note": "His473 is the malonyl-CoA regulatory histidine; loss causes reduced inhibition response"},
    {"variant": "Other compound het", "freq": 4, "domain": "Various",
     "phenotype": "Variable; depends on allele combination",
     "note": "Compound heterozygous; severity determined by less severe allele"},
]

# ── Phenotype distribution ────────────────────────────────────────────────────
PHENOTYPE_DIST = {
    "Severe (neonatal/infantile crisis; biallelic null)":      10,
    "Moderate (childhood; partial activity retained)":         12,
    "Mild (Arctic founder p.Pro479Leu homozygous)":           14,
    "Mild (p.Pro479Leu compound het)":                         4,
}


def _make_patient(i):
    """Synthetic CPT1A patient record (seed=279, deterministic)."""
    rng = random.Random(SEED + i * 41)

    if i < 10:
        # Severe neonatal/infantile
        severity = "Severe"
        variant = rng.choice([
            "p.Thr735Ile/p.Thr735Ile", "p.Arg160Gln/splice-IVS6", "p.Arg217Trp/Exon9-10del",
            "p.Thr735Ile/p.Arg160Gln", "p.Val368Leu/Exon9-10del",
        ])
        onset_age = rng.randint(0, 6)   # months
        c0 = round(rng.uniform(75.0, 160.0), 1)
        c16 = round(rng.uniform(0.03, 0.12), 3)
        c18_1 = round(rng.uniform(0.02, 0.10), 3)
        c0_c16_ratio = round(c0 / max(c16, 0.01), 1)
        glucose = round(rng.uniform(0.5, 2.0), 1)
        ammonia = round(rng.uniform(150, 600), 0)
        hepatomegaly = True
        transaminitis = True
        cardiac = False
        rhabdo = False
        rta = rng.random() < 0.35
        crisis = True
        encephalopathy = True
        arctic = False
        residual_activity = round(rng.uniform(0, 5), 1)

    elif i < 22:
        # Moderate childhood onset
        severity = "Moderate"
        variant = rng.choice([
            "p.Asp100Asn/p.Arg160Gln", "p.His473Tyr/p.Thr735Ile",
            "p.Pro479Leu/p.Thr735Ile", "p.Asp100Asn/splice-IVS6",
            "p.Val368Leu/p.Arg160Gln",
        ])
        onset_age = rng.randint(3, 36)  # months
        c0 = round(rng.uniform(60.0, 120.0), 1)
        c16 = round(rng.uniform(0.04, 0.15), 3)
        c18_1 = round(rng.uniform(0.03, 0.12), 3)
        c0_c16_ratio = round(c0 / max(c16, 0.01), 1)
        glucose = round(rng.uniform(1.0, 3.0), 1)
        ammonia = round(rng.uniform(60, 300), 0)
        hepatomegaly = rng.random() < 0.75
        transaminitis = rng.random() < 0.7
        cardiac = False
        rhabdo = False
        rta = rng.random() < 0.2
        crisis = rng.random() < 0.65
        encephalopathy = crisis and rng.random() < 0.5
        arctic = False
        residual_activity = round(rng.uniform(5, 25), 1)

    elif i < 36:
        # Mild — Arctic founder (p.Pro479Leu homozygous)
        severity = "Mild (Arctic founder)"
        variant = "p.Pro479Leu/p.Pro479Leu"
        onset_age = rng.randint(1, 24)
        c0 = round(rng.uniform(55.0, 100.0), 1)
        c16 = round(rng.uniform(0.05, 0.18), 3)
        c18_1 = round(rng.uniform(0.04, 0.14), 3)
        c0_c16_ratio = round(c0 / max(c16, 0.01), 1)
        glucose = round(rng.uniform(2.0, 4.0), 1)
        ammonia = round(rng.uniform(25, 120), 0)
        hepatomegaly = rng.random() < 0.3
        transaminitis = rng.random() < 0.25
        cardiac = False
        rhabdo = False
        rta = False
        crisis = rng.random() < 0.25
        encephalopathy = crisis and rng.random() < 0.2
        arctic = True
        residual_activity = round(rng.uniform(28, 42), 1)

    else:
        # Mild — compound het with Arctic founder
        severity = "Mild (compound het)"
        variant = rng.choice([
            "p.Pro479Leu/p.Asp100Asn", "p.Pro479Leu/splice-IVS6",
            "p.Pro479Leu/p.Val368Leu",
        ])
        onset_age = rng.randint(3, 36)
        c0 = round(rng.uniform(58.0, 110.0), 1)
        c16 = round(rng.uniform(0.04, 0.16), 3)
        c18_1 = round(rng.uniform(0.03, 0.13), 3)
        c0_c16_ratio = round(c0 / max(c16, 0.01), 1)
        glucose = round(rng.uniform(1.8, 3.8), 1)
        ammonia = round(rng.uniform(40, 180), 0)
        hepatomegaly = rng.random() < 0.4
        transaminitis = rng.random() < 0.35
        cardiac = False
        rhabdo = False
        rta = rng.random() < 0.1
        crisis = rng.random() < 0.35
        encephalopathy = crisis and rng.random() < 0.3
        arctic = True
        residual_activity = round(rng.uniform(15, 35), 1)

    return {
        "id":                f"CPT1A-{SEED}-{i+1:02d}",
        "severity":          severity,
        "variant":           variant,
        "onset_age_months":  onset_age,
        "c0_umol":           c0,
        "c16_umol":          c16,
        "c18_1_umol":        c18_1,
        "c0_c16_ratio":      c0_c16_ratio,
        "glucose_mmol":      glucose,
        "ammonia_umol":      ammonia,
        "hepatomegaly":      hepatomegaly,
        "transaminitis":     transaminitis,
        "cardiomyopathy":    cardiac,
        "rhabdomyolysis":    rhabdo,
        "renal_tubular_acidosis": rta,
        "metabolic_crisis":  crisis,
        "encephalopathy":    encephalopathy,
        "arctic_founder":    arctic,
        "residual_activity_pct": residual_activity,
    }


PATIENTS = [_make_patient(i) for i in range(40)]


def get_overview():
    n = len(PATIENTS)
    severe_n   = sum(1 for p in PATIENTS if p["severity"].startswith("Severe"))
    moderate_n = sum(1 for p in PATIENTS if p["severity"].startswith("Moderate"))
    mild_n     = sum(1 for p in PATIENTS if p["severity"].startswith("Mild"))
    arctic_n   = sum(1 for p in PATIENTS if p["arctic_founder"])
    crisis_n   = sum(1 for p in PATIENTS if p["metabolic_crisis"])
    hepato_n   = sum(1 for p in PATIENTS if p["hepatomegaly"])
    ammonia_n  = sum(1 for p in PATIENTS if p["ammonia_umol"] > 80)
    encepha_n  = sum(1 for p in PATIENTS if p["encephalopathy"])
    rta_n      = sum(1 for p in PATIENTS if p["renal_tubular_acidosis"])
    cardiac_n  = sum(1 for p in PATIENTS if p["cardiomyopathy"])

    avg_c0   = round(sum(p["c0_umol"]        for p in PATIENTS) / n, 1)
    avg_c16  = round(sum(p["c16_umol"]       for p in PATIENTS) / n, 3)
    avg_c0_c16_ratio = round(sum(p["c0_c16_ratio"] for p in PATIENTS) / n, 1)
    avg_gluc = round(sum(p["glucose_mmol"]   for p in PATIENTS) / n, 2)
    avg_nh3  = round(sum(p["ammonia_umol"]   for p in PATIENTS) / n, 0)

    return {
        "n_patients": n,
        "seed": SEED,
        "disease": "CPT1A Deficiency (Carnitine Palmitoyltransferase IA Deficiency)",
        "gene": "CPT1A",
        "locus": "11q13.3",
        "omim_disease": "#255120",
        "omim_gene": "*600528",
        "prevalence": "~1:750,000 general; up to ~1:40 in some Arctic communities (Arctic founder p.Pro479Leu)",
        "inheritance": "Autosomal Recessive (AR), biallelic LOF",
        "severity_distribution": {
            "Severe": severe_n,
            "Moderate": moderate_n,
            "Mild (Arctic founder or compound)": mild_n,
        },
        "arctic_founder_n": arctic_n,
        "clinical_features": {
            "metabolic_crisis":         crisis_n,
            "hepatomegaly":             hepato_n,
            "hyperammonemia_gt80":      ammonia_n,
            "encephalopathy":           encepha_n,
            "renal_tubular_acidosis":   rta_n,
            "cardiomyopathy":           cardiac_n,   # should be 0 — exam trap
        },
        "biomarkers": {
            "avg_c0_umol":           avg_c0,
            "avg_c16_umol":          avg_c16,
            "avg_c0_c16_ratio":      avg_c0_c16_ratio,
            "avg_glucose_crisis_mmol": avg_gluc,
            "avg_ammonia_umol":      avg_nh3,
        },
        "key_exam_facts": [
            "C0 (free carnitine) ELEVATED ≥60 μmol/L — PRIMARY NBS MARKER (OPPOSITE of most FAO disorders)",
            "C0/(C16+C18) ratio ELEVATED >40 — HIGHLY SPECIFIC discriminator",
            "C16 (palmitoylcarnitine) NORMAL/LOW — KEY NEGATIVE vs VLCAD/LCHAD",
            "C8 NORMAL — KEY NEGATIVE vs MCAD",
            "C14:1 NORMAL — KEY NEGATIVE vs VLCAD",
            "HYPERAMMONAEMIA — UNIQUE among all FAO disorders (50–600 μmol/L)",
            "NO CARDIOMYOPATHY — CPT1B serves heart; CPT1A is liver-specific (KEY EXAM TRAP)",
            "NO RHABDOMYOLYSIS — CPT1A liver-specific; muscle uses CPT1B (intact)",
            "MCT OIL THERAPEUTIC (medium-chain FA bypass CPT1A; enter IMM directly via FATP)",
            "FASTING ABSOLUTE CI + KD ABSOLUTE CI",
            "Arctic founder p.Pro479Leu — up to 85% allele frequency in Inuit populations",
            "L-Carnitine NOT routinely indicated (C0 already elevated)",
        ],
    }


def get_breakdown():
    patients_out = []
    for p in PATIENTS:
        patients_out.append({
            "id":               p["id"],
            "severity":         p["severity"],
            "variant":          p["variant"],
            "onset_age_months": p["onset_age_months"],
            "c0_umol":          p["c0_umol"],
            "c16_umol":         p["c16_umol"],
            "c18_1_umol":       p["c18_1_umol"],
            "c0_c16_ratio":     p["c0_c16_ratio"],
            "glucose_mmol":     p["glucose_mmol"],
            "ammonia_umol":     p["ammonia_umol"],
            "hepatomegaly":     p["hepatomegaly"],
            "transaminitis":    p["transaminitis"],
            "cardiomyopathy":   p["cardiomyopathy"],
            "rhabdomyolysis":   p["rhabdomyolysis"],
            "rta":              p["renal_tubular_acidosis"],
            "metabolic_crisis": p["metabolic_crisis"],
            "encephalopathy":   p["encephalopathy"],
            "arctic_founder":   p["arctic_founder"],
            "residual_activity_pct": p["residual_activity_pct"],
        })

    severity_groups = {}
    for p in PATIENTS:
        grp = p["severity"].split("(")[0].strip()
        severity_groups.setdefault(grp, []).append(p)

    by_severity = {}
    for grp, pts in severity_groups.items():
        by_severity[grp] = {
            "n": len(pts),
            "avg_c0":        round(sum(x["c0_umol"] for x in pts) / len(pts), 1),
            "avg_c16":       round(sum(x["c16_umol"] for x in pts) / len(pts), 3),
            "avg_c0_c16":   round(sum(x["c0_c16_ratio"] for x in pts) / len(pts), 1),
            "avg_glucose":   round(sum(x["glucose_mmol"] for x in pts) / len(pts), 2),
            "avg_ammonia":   round(sum(x["ammonia_umol"] for x in pts) / len(pts), 0),
            "crisis_rate":   round(sum(1 for x in pts if x["metabolic_crisis"]) / len(pts) * 100, 1),
        }

    variant_counts = {}
    for p in PATIENTS:
        # Extract first allele gene/variant label
        key = p["variant"].split("/")[0].strip()
        variant_counts[key] = variant_counts.get(key, 0) + 1

    treatment_responses = {
        "MCT_oil_therapeutic_n":             sum(1 for p in PATIENTS if p["severity"].startswith("Mild") or p["severity"].startswith("Moderate")),
        "cornstarch_nocturnal_n":            sum(1 for p in PATIENTS if p["glucose_mmol"] < 3.5),
        "iv_glucose_acute_n":               sum(1 for p in PATIENTS if p["metabolic_crisis"]),
        "fasting_avoided_strict":           len(PATIENTS),
        "carnitine_supplemented":           0,   # NOT routine — C0 already high
        "vpa_avoided":                      len(PATIENTS),
    }

    return {
        "patients":         patients_out,
        "by_severity":      by_severity,
        "variant_counts":   variant_counts,
        "treatment_summary": treatment_responses,
        "nbs_profile_summary": {
            "pct_c0_elevated_ge60":   round(sum(1 for p in PATIENTS if p["c0_umol"] >= 60) / len(PATIENTS) * 100, 1),
            "pct_c0_c16_ratio_gt40":  round(sum(1 for p in PATIENTS if p["c0_c16_ratio"] > 40) / len(PATIENTS) * 100, 1),
            "pct_cardiomyopathy":     0.0,  # Should always be 0
            "pct_rhabdo":             0.0,
            "pct_hyperammonemia":     round(sum(1 for p in PATIENTS if p["ammonia_umol"] > 80) / len(PATIENTS) * 100, 1),
        },
    }


def get_definitions():
    return {
        "disease_name": "CPT1A Deficiency (Carnitine Palmitoyltransferase IA Deficiency / Hepatic CPT Deficiency Type IA)",
        "gene": "CPT1A (11q13.3)",
        "locus": "11q13.3",
        "omim_gene": "CPT1A *600528",
        "omim_disease": "#255120",
        "inheritance": "Autosomal Recessive (AR) — biallelic LOF",
        "protein": (
            "CPT1A — 773 aa; outer mitochondrial membrane; liver-specific isoform of carnitine "
            "palmitoyltransferase 1. Catalyses the rate-limiting step of long-chain fatty acid "
            "entry into mitochondria. Two related isoforms serve other tissues: CPT1B (heart/muscle, "
            "22q13.33) and CPT1C (brain, 19q13.33) — BOTH INTACT in CPT1A deficiency, explaining "
            "the absence of cardiac and muscle disease."
        ),
        "enzymatic_function": (
            "CPT1A catalyses: long-chain acyl-CoA + L-carnitine → long-chain acylcarnitine + CoA-SH "
            "(outer mitochondrial membrane, cytosolic face). This is the RATE-LIMITING and "
            "REGULATED step of long-chain FA import into the mitochondrial matrix. "
            "Long-chain acylcarnitines then cross the inner mitochondrial membrane via CACT (SLC25A20) "
            "in exchange for free carnitine. CPT2 then regenerates long-chain acyl-CoA in the matrix "
            "for beta-oxidation. CPT1A LOF → long-chain FA CANNOT enter mitochondria → "
            "beta-oxidation BLOCKED → free carnitine ACCUMULATES (not consumed) → C0 ↑↑."
        ),
        "pathway": "Long-chain fatty acid beta-oxidation (carnitine shuttle step 1, outer mitochondrial membrane)",
        "metabolic_block": (
            "CPT1A LOF → long-chain acyl-CoA cannot be transferred to carnitine at the outer "
            "mitochondrial membrane → long-chain FA stranded in cytosol → beta-oxidation blocked → "
            "HYPOKETOTIC HYPOGLYCAEMIA (no ketone bodies from FA oxidation). "
            "Free carnitine ACCUMULATES (not consumed by CPT1A) → C0 ELEVATED (≥60 μmol/L). "
            "Secondary: long-chain acyl-CoA accumulates → inhibits NAGS → NAG reduced → CPS1 down "
            "→ HYPERAMMONAEMIA (unique among FAO disorders)."
        ),
        "nbs_marker": (
            "C0 (free carnitine) ELEVATED ≥60 μmol/L — PRIMARY NBS MARKER. "
            "C0/(C16+C18) ratio ELEVATED >40 — HIGHLY SPECIFIC. "
            "C16 and C18:1 NORMAL/LOW — NOT elevated (unlike VLCAD/LCHAD). "
            "INVERTED PROFILE: high C0, low acylcarnitines — opposite of most FAO disorders."
        ),
        "key_biomarkers": {
            "C0_free_carnitine": "ELEVATED ≥60 μmol/L (normal <50) — PRIMARY NBS MARKER",
            "C0_C16_ratio": "ELEVATED (>40 diagnostic) — HIGHLY SPECIFIC for CPT1A",
            "C0_C16_plus_C18_ratio": "ELEVATED >40 — most discriminating ratio",
            "C16_palmitoylcarnitine": "NORMAL or LOW (not generated without CPT1A)",
            "C18_1_oleoylcarnitine": "NORMAL or LOW",
            "C8_octanoylcarnitine": "NORMAL — KEY NEGATIVE vs MCAD",
            "C14_1_tetradecenoylcarnitine": "NORMAL — KEY NEGATIVE vs VLCAD",
            "C16_OH": "NORMAL — KEY NEGATIVE vs LCHAD",
            "Ammonia": "ELEVATED 50–600 μmol/L — UNIQUE among FAO disorders",
            "Glucose": "LOW <2.5 mmol/L during crisis (hypoketotic hypoglycaemia)",
            "Ketones": "ABSENT or inappropriately low (HYPOKETOTIC)",
            "Dicarboxylic_acids_urine": "Mildly elevated (adipic, suberic — omega oxidation overflow)",
        },
        "clinical_features": {
            "Hypoketotic_hypoglycaemia": "HALLMARK — primary presentation; fasting/illness triggered",
            "Hyperammonemia": "UNIQUE among FAO disorders; 50–600 μmol/L; mechanism: long-chain acyl-CoA → NAGS inhibition → CPS1 ↓",
            "Hepatomegaly": "Common; hepatic steatosis; transaminitis",
            "NO_cardiomyopathy": "EXAM TRAP — CPT1B (not CPT1A) serves heart; CPT1A deficiency spares heart",
            "NO_rhabdomyolysis": "CPT1A is liver-specific; muscle uses CPT1B (intact); no skeletal muscle disease",
            "Renal_tubular_acidosis": "Recognised but uncommon; tubular dysfunction in some severe cases",
            "Encephalopathy": "Secondary to hypoglycaemia during acute crisis",
            "Sudden_death": "Risk if crisis unrecognised (unattended fasting in infancy)",
        },
        "treatment": {
            "Fasting_ABSOLUTE_CI": "Level A — triggers lipolysis; long-chain FA cannot be oxidised; primary prevention",
            "MCT_oil_THERAPEUTIC": "Level A — medium-chain FA (C8/C10) bypass CPT1A; enter mitochondria directly; generate ketones via MCAD/SCAD",
            "Cornstarch": "Level A — uncooked cornstarch prevents nocturnal hypoglycaemia; sustained glucose release 6–8h",
            "IV_glucose_10pct": "Level A — first-line emergency; target GIR 8–10 mg/kg/min; anti-catabolic",
            "KD_ABSOLUTE_CI": "Requires long-chain fat for ketone generation; CPT1A block prevents long-chain FA oxidation → worsens disease",
            "VPA_HIGH_RISK": "Valproate inhibits FAO, depletes carnitine; avoid in all FAO disorders",
            "L_Carnitine_NOT_routine": "C0 already elevated; NOT indicated routinely; risk of accumulating toxic long-chain acylcarnitines; give only if secondary depletion documented",
            "Long_chain_fat_restriction": "Level B — reduce dietary long-chain fat intake; replace calories with MCT/carbohydrate",
            "Emergency_protocol": "IV glucose + avoid fat loading; treat hyperammonemia if NH3 >200 μmol/L with ammonia scavengers",
        },
        "contraindications": [
            "FASTING — ABSOLUTE CONTRAINDICATION (primary trigger)",
            "KETOGENIC DIET — ABSOLUTE CONTRAINDICATION (long-chain fat floods blocked CPT1A)",
            "VALPROATE — HIGH RISK (inhibits FAO + carnitine depletion)",
            "L-CARNITINE supplementation NOT routine (C0 elevated; may accumulate acylcarnitines)",
        ],
        "key_distinguishing_facts": [
            "C0 HIGH = CPT1A (opposite of MCAD/VLCAD/LCHAD where C0 is LOW)",
            "C0/(C16+C18) ratio >40 = highly specific for CPT1A",
            "NO cardiomyopathy — CPT1B serves heart; CPT1A does NOT",
            "Hyperammonemia = unique among FAO disorders",
            "MCT IS therapeutic (medium chain bypasses CPT1A)",
            "Arctic founder p.Pro479Leu = up to 85% allele frequency in Inuit populations",
            "Liver is principal organ affected (hepatic isoform); heart + muscle spared",
        ],
        "genetics": {
            "key_variants": {
                "p.Pro479Leu_c.1436C>T": "Arctic founder; Inuit/Alaska Native; mild hypomorphic; 30-40% residual activity",
                "p.Thr735Ile": "Severe; North American; complete LOF",
                "p.Arg160Gln": "Severe; European; complete LOF; catalytic domain",
                "p.Asp100Asn": "Moderate; Asian; partial residual activity",
                "IVS6_splice": "Null allele; severe; various populations",
            },
            "arctic_prevalence": "p.Pro479Leu allele frequency up to 85% in some Inuit communities; 40-70% in Alaska Native",
            "population_note": (
                "p.Pro479Leu is the most common pathogenic CPT1A variant globally by allele frequency, "
                "but is largely confined to Arctic/subarctic populations. Whether it represents a "
                "pathogenic variant or population-adapted variant is debated — it confers ~30-40% "
                "residual CPT1A activity and most homozygotes are asymptomatic if fasting is avoided."
            ),
        },
        "comparison_table": {
            "CPT1A_vs_MCAD": "C0 HIGH (CPT1A) vs C0 LOW (MCAD); C8 normal (CPT1A) vs C8 elevated (MCAD); no glycine conjugates either way; hyperammonemia only CPT1A",
            "CPT1A_vs_VLCAD": "C0 HIGH (CPT1A) vs C0 LOW (VLCAD); C14:1 normal (CPT1A) vs C14:1 elevated (VLCAD); no cardiomyopathy (CPT1A) vs cardiomyopathy (VLCAD); no rhabdo (CPT1A) vs rhabdo (VLCAD)",
            "CPT1A_vs_LCHAD": "C0 HIGH (CPT1A) vs C0 LOW (LCHAD); C16-OH normal (CPT1A) vs C16-OH elevated (LCHAD); no retinopathy (CPT1A) vs retinopathy (LCHAD); no neuropathy (CPT1A) vs neuropathy (LCHAD)",
            "CPT1A_vs_CPT2": "High C0 (CPT1A) vs low C0 + high C16 (CPT2); liver disease (CPT1A) vs rhabdomyolysis/cardiac (CPT2); isoform difference: liver CPT1A vs ubiquitous CPT2",
            "CPT1A_vs_CACT": "High C0 (CPT1A) vs high C16+C18 + cardiomyopathy (CACT/SLC25A20); different step in carnitine shuttle",
        },
        "step_in_carnitine_shuttle": (
            "Step 1 (CPT1A — outer IMM; RATE-LIMITING): long-chain acyl-CoA + carnitine → acylcarnitine + CoA-SH  [← CPT1A BLOCK]\n"
            "Step 2 (CACT/SLC25A20 — inner IMM translocase): acylcarnitine (in) ↔ carnitine (out) [antiport]\n"
            "Step 3 (CPT2 — inner IMM, matrix face): acylcarnitine + CoA-SH → acyl-CoA + carnitine\n"
            "Steps 4–7: Beta-oxidation proper (VLCAD, HADHA, HADHA, HADHB for long chain)"
        ),
        "malonyl_coa_regulation": (
            "CPT1A is PHYSIOLOGICALLY INHIBITED by malonyl-CoA (the first committed intermediate "
            "of de novo lipogenesis). When glucose/insulin are high → malonyl-CoA rises → CPT1A "
            "inhibited → FA oxidation suppressed → lipogenesis favoured. "
            "Fasting/glucagon → malonyl-CoA falls → CPT1A active → FA oxidation promoted. "
            "p.His473Tyr variant disrupts the malonyl-CoA regulatory site."
        ),
    }
