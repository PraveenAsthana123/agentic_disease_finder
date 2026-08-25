#!/usr/bin/env python3
"""AUH (3-Methylglutaconyl-CoA Hydratase) Deficiency Dashboard.

AUH is Step 4 of leucine catabolism (between MCC Step 3 and HMGCL Step 5):
    3-Methylglutaconyl-CoA  +  H₂O  →  [AUH]  →  HMG-CoA   (reversible hydration)

AUH LOF → 3-Methylglutaconyl-CoA CANNOT be hydrated → ACCUMULATES:
  → de-esterification → 3-MGC (3-methylglutaconic acid) — PRIMARY MARKER
  → 3-MG  (3-methylglutaric acid) — secondary
  → Leucine catabolism BLOCKED at step 4 (upstream of HMGCL)

KEY FACTS (EXAM HIGHEST-YIELD):
  1. 3-MGC (3-methylglutaconic acid) ELEVATED — PRIMARY MARKER (not as extreme as non-AUH causes)
  2. 3-HMG NORMAL — KEY NEGATIVE vs HMGCL (HMGCL is intact; AUH block is UPSTREAM of HMGCL)
  3. Ketones PRESENT / NORMAL — KEY NEGATIVE vs HMGCL (ketogenesis via HMGCS2 intact)
  4. 3-MCG ABSENT — KEY NEGATIVE vs 3-MCC (3-methylcrotonyl-CoA does NOT accumulate)
  5. NO KD contraindication — AUH deficiency does NOT abolish ketogenesis (contrast HMGCL)
  6. VPA — MODERATE RISK only (NOT absolute CI like HMGCL; no direct AUH inhibition)
  7. VERY RARE — fewer than 50 cases worldwide 2026; highly variable phenotype (asymptomatic to moderate ID)
  8. MGA Type I (AUH) vs other 3-MGC acidurias: AUH = leucine catabolism; others = mitochondrial
     maintenance (TAZ/Barth), mtDNA (DNAJC19), complex V (ATPAF2), OPA3 — different mechanism entirely

OMIM Disease: #250950 (3-Methylglutaconic Aciduria, Type I)
OMIM Gene:    *600529 (AUH)
Chromosome:   9q22.31
Inheritance:  Autosomal Recessive (AR), biallelic LOF
Protein:      AUH = 338 aa; bifunctional: AU-rich element RNA-binding protein (N-terminal domain) +
              3-methylglutaconyl-CoA hydratase (C-terminal catalytic domain); mitochondrial matrix;
              homodimeric; the RNA-binding function is secondary — the hydratase is the metabolic role
Prevalence:   ~1:500,000–1,000,000 (very rare; <50 cases worldwide 2026)

LEUCINE CATABOLISM CONTEXT (where AUH fits — STEP 4):
  L-Leucine  → BCAT (Step 1a) → KIC (α-ketoisocaproate)
  KIC        → BCKDH (Step 1b) → Isovaleryl-CoA
  Isovaleryl-CoA → IVD (Step 2) → 3-Methylcrotonyl-CoA     [IVD deficiency = IVA — upstream]
  3-MC-CoA   → MCC (Step 3) → 3-Methylglutaconyl-CoA       [MCC deficiency — directly upstream]
  3-MG-CoA   → AUH (Step 4) → HMG-CoA   [AUH BLOCKED HERE — STEP 4]
  HMG-CoA    → HMGCL (Step 5) → Acetoacetate + Acetyl-CoA  [HMGCL intact — downstream, ketogenesis OK]

KEY POINT: AUH is UPSTREAM of HMGCL. HMGCL remains functional in AUH deficiency.
  → Ketogenesis via HMGCS2 (FA oxidation → Acetyl-CoA → HMGCS2 → HMG-CoA → HMGCL → ketones) is INTACT
  → NO hypoketotic hypoglycaemia (contrast HMGCL where both steps fail)
  → Acute metabolic crisis uncommon (contrast HMGCL, IVA, PA, MMA)

BIOMARKER PATTERN — AUH DEFICIENCY:
  3-MGC (3-methylglutaconic acid)    20–150 mmol/mol Cr — PRIMARY MARKER (urine OA / GC-MS)
  3-MG  (3-methylglutaric acid)     10–60 mmol/mol Cr — secondary (elevated in proportion to 3-MGC)
  C5-OH (3-OH-isovalerylcarnitine)  borderline–mildly elevated (0.3–1.5 µmol/L) — less than MCC/HMGCL
  3-HMG (3-hydroxy-3-methylglutaric acid)  NORMAL (<10 mmol/mol Cr) — KEY NEGATIVE vs HMGCL
  Ketones (β-OHB / AcAc)           PRESENT / NORMAL — KEY NEGATIVE vs HMGCL
  3-MCG (3-methylcrotonylglycine)   ABSENT — KEY NEGATIVE vs 3-MCC
  C3 (propionylcarnitine)           NORMAL — KEY NEGATIVE vs HLCS/PA
  NH3                               NORMAL — KEY NEGATIVE vs UCDs
  Free carnitine                    mildly–moderately low (secondary, less severe than HMGCL)
  Glucose                           NORMAL inter-ictally; NO acute hypoglycaemia pattern

KEY NEGATIVE MARKERS (CRITICAL DIFFERENTIALS):
  3-HMG     NORMAL — KEY NEGATIVE vs HMGCL (MOST CRITICAL — AUH is UPSTREAM, not downstream)
  Ketones   PRESENT — KEY NEGATIVE vs HMGCL (ketogenesis INTACT via HMGCS2)
  3-MCG     ABSENT — KEY NEGATIVE vs 3-MCC
  C3 (propionylcarnitine)  NORMAL — KEY NEGATIVE vs HLCS/PA
  NH3       NORMAL — KEY NEGATIVE vs UCDs
  GAA       NORMAL — KEY NEGATIVE vs GAMT
  Creatine  PRESENT — KEY NEGATIVE vs AGAT/GAMT/SLC6A8
  Cardiomyopathy  ABSENT — KEY NEGATIVE vs TAZ (MGA Type II / Barth syndrome)
  Neutropenia  ABSENT — KEY NEGATIVE vs TAZ (Barth syndrome)
  X-linked pattern  ABSENT — AUH is AR both sexes; TAZ is X-linked males only

MGA TYPE I vs OTHER 3-MGC ACIDURIAS (CRITICAL DIFFERENTIAL):
  MGA-I  (AUH, 9q22.31, AR):        3-MGC mod elevated; leucine catabolism block; 3-HMG NORMAL; variable
  MGA-II (TAZ, Xq28, X-linked):     3-MGC + cardiomyopathy + neutropenia + males only; Barth syndrome
  MGA-III (OPA3, 19q13.32, AR):     3-MGC + optic atrophy + chorea; OPA3 deficiency
  MGA-IIIB (DNAJC19, 3q26.33, AR): 3-MGC + dilated CM + male predilection + non-progressive CA; DCMA
  MGA-IV (ATPAF2, 17p11.2, AR):    3-MGC + complex V deficiency; lactic acidosis
  MGA-V (SERAC1, 6q22.1, AR):      3-MGC + 3-MG + deafness-dystonia-hepatopathy; MEGDEL syndrome

PATHOGENESIS:
  AUH catalyses the REVERSIBLE hydration of 3-methylglutaconyl-CoA (Δ2-enoyl-CoA) → HMG-CoA
  AUH LOF → 3-methylglutaconyl-CoA CANNOT undergo hydration → ACCUMULATES
  → Hydrolysis of thioester → 3-methylglutaconate (3-MGC, free acid) — detected in urine
  → Some retrograde flux to 3-methylglutaconyl-CoA → 3-methylcrotonyl-CoA (via MCC reverse)
     not major — 3-MCG levels NOT significantly elevated
  → HMG-CoA REDUCED (from leucine pathway) but NOT zero:
     Alternative HMG-CoA sources intact: fatty acid β-oxidation → Acetyl-CoA → HMGCS2 → HMG-CoA
     → HMGCL INTACT → ketones produced NORMALLY from FA oxidation
  → NO hypoketotic hypoglycaemia (major clinical distinction from HMGCL deficiency)

TREATMENT:
  Leucine restriction:              LEVEL A — reduces 3-methylglutaconyl-CoA substrate (primary strategy)
  L-Carnitine:                      LEVEL A — secondary depletion via 3-MGC-carnitine conjugation
  Fasting avoidance:                LEVEL B — catabolism worsens 3-MGC accumulation; less critical than HMGCL
  Levetiracetam (LEV):              LEVEL B — first-line AED for AUH-associated seizures
  Valproic acid (VPA):              MODERATE RISK — carnitine depletion; NOT absolute CI (no AUH active-site inhibition)
  Ketogenic Diet:                   NOT absolute CI (ketogenesis INTACT via HMGCS2/HMGCL); caution with leucine
  Biotin:                           NOT effective (AUH is NOT biotin-dependent)

PHENOTYPES:
  Asymptomatic / NBS-detected:      ~30% — biochemically abnormal; no clinical symptoms; low-leucine watch
  Mild (speech delay / mild ID):    ~45% — MODAL; developmental delay; speech onset delayed; some seizures
  Moderate–Severe (DRE / sig ID):   ~25% — intellectual disability, drug-resistant epilepsy, hypotonia

SEIZURES IN AUH:
  ~25–40% of cases (significantly less common than HMGCL, IVA, PA, MMA)
  Type: focal > generalised; myoclonic reported; DRE ~10–15%
  Mechanism: leucine pathway metabolite neurotoxicity (3-MGC); NOT acute hypoglycaemia
  No acute hypoketotic crisis (unlike HMGCL) → seizures chronic, not crisis-related
  AED: LEV first-line; VPA MODERATE risk (carnitine); NOT absolute CI unlike HMGCL

COMMON PATHOGENIC VARIANTS:
  p.Arg160Trp (c.478C>T):    ~30% — N-terminal hydratase domain; moderately severe; most common
  p.Gly213Asp (c.638G>A):    ~20% — catalytic β-sheet; severe; complete LOF
  p.Leu220Pro (c.659T>C):    ~15% — dimer interface; severe; protein aggregation
  p.Arg295Gln (c.884G>A):    ~10% — active-site adjacent; moderate
  c.IVS6+1G>A (splice):      ~12% — exon 6 skip → truncation; null; severe
  p.Thr274Ile (c.821C>T):    ~8%  — mild; residual activity ~20%; attenuated phenotype
  p.Glu183Lys (c.547G>A):    ~5%  — Caucasian; moderate
"""

import random

SEED       = 263      # next after HMGCL (seed 261); AUH = leucine step 4 between MCC (259) and HMGCL (261)
N_PATIENTS = 40

# ── Phenotype classes ─────────────────────────────────────────────────────────
PHENOTYPE_CLASSES = [
    {
        "class": "Mild (Speech Delay / Mild ID — Modal)",
        "pct": 45,
        "age_onset_months_range": (12, 48),
        "mgc_urine_range": (25, 120),        # mmol/mol Cr — 3-MGC (primary)
        "mg_urine_range": (12, 55),          # mmol/mol Cr — 3-MG secondary
        "c5oh_range": (0.3, 1.2),            # µmol/L — NBS borderline/mildly elevated
        "free_carn_range": (18, 35),
        "glucose_range": (4.0, 6.5),         # NORMAL (no acute hypoglycaemia)
        "ketones_present": True,             # ALWAYS present (ketogenesis intact)
        "seizures_prob": 0.28,
        "id_prob": 0.55,
        "speech_delay_prob": 0.70,
        "hypotonia_prob": 0.40,
        "note": "Most common phenotype; speech onset delayed 6–18 months; mild IDD; 3-MGC moderately elevated; no acute crisis",
    },
    {
        "class": "Asymptomatic / NBS-Detected",
        "pct": 30,
        "age_onset_months_range": (0, 2),     # NBS-detected
        "mgc_urine_range": (20, 60),
        "mg_urine_range": (8, 28),
        "c5oh_range": (0.3, 0.8),
        "free_carn_range": (25, 45),
        "glucose_range": (4.5, 6.8),
        "ketones_present": True,
        "seizures_prob": 0.05,
        "id_prob": 0.10,
        "speech_delay_prob": 0.15,
        "hypotonia_prob": 0.10,
        "note": "Biochemically confirmed AUH deficiency; no clinical symptoms; incidental or NBS-triggered; low-leucine monitoring",
    },
    {
        "class": "Moderate–Severe (DRE / Significant ID)",
        "pct": 25,
        "age_onset_months_range": (2, 18),
        "mgc_urine_range": (80, 220),
        "mg_urine_range": (35, 90),
        "c5oh_range": (0.8, 2.5),
        "free_carn_range": (10, 24),
        "glucose_range": (3.8, 6.0),
        "ketones_present": True,
        "seizures_prob": 0.72,
        "id_prob": 0.90,
        "speech_delay_prob": 0.90,
        "hypotonia_prob": 0.70,
        "note": "Severe spectrum; significant intellectual disability; drug-resistant epilepsy; elevated 3-MGC; leucine restriction mandatory",
    },
]

# Common pathogenic variants
VARIANTS = [
    {"variant": "p.Arg160Trp",  "freq": 30, "domain": "Hydratase N-terminal domain", "phenotype": "Moderate–severe; most common worldwide", "note": "c.478C>T; reduces substrate binding affinity; partial residual activity"},
    {"variant": "p.Gly213Asp",  "freq": 20, "domain": "Catalytic β-sheet",           "phenotype": "Severe; complete LOF; neonatal/early onset",   "note": "c.638G>A; disrupts hydratase active site; no residual activity"},
    {"variant": "p.Leu220Pro",  "freq": 15, "domain": "Dimer interface",              "phenotype": "Severe; protein aggregation; complete LOF",    "note": "c.659T>C; dimer assembly failure → rapid degradation"},
    {"variant": "c.IVS6+1G>A",  "freq": 12, "domain": "Splice site (intron 6)",       "phenotype": "Severe; null allele; exon 6 truncation",      "note": "Aberrant splicing → frameshift → NMD; complete LOF"},
    {"variant": "p.Arg295Gln",  "freq": 10, "domain": "Active-site adjacent",         "phenotype": "Moderate; reduced catalytic efficiency",      "note": "c.884G>A; partial residual hydratase activity ~10–15%"},
    {"variant": "p.Thr274Ile",  "freq": 8,  "domain": "Peripheral helix",             "phenotype": "Mild; ~20% residual activity; attenuated",   "note": "c.821C>T; mild phenotype; often asymptomatic on NBS"},
    {"variant": "p.Glu183Lys",  "freq": 5,  "domain": "Surface loop (Caucasian)",     "phenotype": "Moderate; European ancestry; variable onset","note": "c.547G>A; moderate reduction in hydratase Vmax"},
]


def _gen_cohort():
    """Generate synthetic 40-patient cohort with AUH-realistic biomarker profiles."""
    random.seed(SEED)
    phenotype_dist = []
    for cls in PHENOTYPE_CLASSES:
        n = round(N_PATIENTS * cls["pct"] / 100)
        phenotype_dist.extend([cls] * n)
    while len(phenotype_dist) < N_PATIENTS:
        phenotype_dist.append(PHENOTYPE_CLASSES[0])
    phenotype_dist = phenotype_dist[:N_PATIENTS]

    patients = []
    for i in range(N_PATIENTS):
        cls = phenotype_dist[i]
        pid = f"AUH-{i+1:03d}"

        # Variant selection (weighted by frequency)
        var_pool = []
        for v in VARIANTS:
            var_pool.extend([v["variant"]] * v["freq"])
        variant = random.choice(var_pool)

        onset_mo        = round(random.uniform(*cls["age_onset_months_range"]), 1)
        mgc             = round(random.uniform(*cls["mgc_urine_range"]), 1)
        mg              = round(random.uniform(*cls["mg_urine_range"]), 1)
        c5oh            = round(random.uniform(*cls["c5oh_range"]), 2)
        carn            = round(random.uniform(*cls["free_carn_range"]), 1)
        glucose         = round(random.uniform(*cls["glucose_range"]), 2)
        ketones_present = cls["ketones_present"]            # always True in AUH
        hmg_3           = round(random.uniform(1.5, 8.5), 1)   # NORMAL (<10) — KEY NEGATIVE
        seiz            = random.random() < cls["seizures_prob"]
        idd             = random.random() < cls["id_prob"]
        speech_delay    = random.random() < cls["speech_delay_prob"]
        hypotonia       = random.random() < cls["hypotonia_prob"]
        vpa_tried       = seiz and random.random() < 0.20     # sometimes given for seizures

        patients.append({
            "id": pid,
            "phenotype": cls["class"],
            "variant": variant,
            "age_onset_months": onset_mo,
            "mgc_urine_mmol_mol_cr": mgc,       # 3-MGC — PRIMARY
            "mg_urine_mmol_mol_cr": mg,         # 3-MG secondary
            "c5oh_umol_l": c5oh,                # borderline/mildly elevated
            "hmg_3_urine": hmg_3,              # NORMAL — KEY NEGATIVE vs HMGCL
            "free_carnitine_umol_l": carn,
            "glucose_mmol_l": glucose,          # NORMAL (no acute hypoglycaemia)
            "ketones_present": ketones_present,  # ALWAYS True
            "seizures": seiz,
            "intellectual_disability": idd,
            "speech_delay": speech_delay,
            "hypotonia": hypotonia,
            "vpa_given": vpa_tried,
        })
    return patients


_COHORT = _gen_cohort()


def get_overview():
    pheno_counts = {}
    for p in _COHORT:
        k = p["phenotype"]
        pheno_counts[k] = pheno_counts.get(k, 0) + 1

    pheno_dist = [
        {"class": k, "n": v, "pct": round(v / N_PATIENTS * 100)}
        for k, v in pheno_counts.items()
    ]

    n_seiz   = sum(1 for p in _COHORT if p["seizures"])
    n_idd    = sum(1 for p in _COHORT if p["intellectual_disability"])
    n_speech = sum(1 for p in _COHORT if p["speech_delay"])
    n_hypo   = sum(1 for p in _COHORT if p["hypotonia"])
    n_ketok  = sum(1 for p in _COHORT if p["ketones_present"])  # all 40
    n_vpa    = sum(1 for p in _COHORT if p["vpa_given"])
    avg_mgc  = round(sum(p["mgc_urine_mmol_mol_cr"] for p in _COHORT) / N_PATIENTS, 1)
    avg_mg   = round(sum(p["mg_urine_mmol_mol_cr"] for p in _COHORT) / N_PATIENTS, 1)

    return {
        "disease": "AUH Deficiency (3-Methylglutaconyl-CoA Hydratase Deficiency / 3-Methylglutaconic Aciduria Type I / MGA-I)",
        "gene": "AUH",
        "omim_gene": "600529",
        "omim_disease": "250950",
        "locus": "9q22.31",
        "inheritance": "Autosomal Recessive",
        "prevalence": "~1:500,000–1,000,000 (very rare; <50 cases worldwide 2026)",
        "pathway_step": "Leucine catabolism STEP 4 — 3-Methylglutaconyl-CoA + H₂O → HMG-CoA (reversible hydration; between MCC step 3 and HMGCL step 5)",
        "n_patients": N_PATIENTS,
        "seed": SEED,
        "kpi": {
            "n_patients":   {"label": "Cohort",                 "value": str(N_PATIENTS),                                    "color": "#1b5e20"},
            "seizures":     {"label": "Seizures",               "value": f"{n_seiz} ({round(n_seiz/N_PATIENTS*100)}%)",      "color": "#b71c1c"},
            "idd":          {"label": "Intellectual Disability", "value": f"{n_idd} ({round(n_idd/N_PATIENTS*100)}%)",       "color": "#4a148c"},
            "speech_delay": {"label": "Speech Delay",           "value": f"{n_speech} ({round(n_speech/N_PATIENTS*100)}%)", "color": "#e65100"},
            "hypotonia":    {"label": "Hypotonia",              "value": f"{n_hypo} ({round(n_hypo/N_PATIENTS*100)}%)",     "color": "#0d47a1"},
            "ketones_ok":   {"label": "Ketones Present (all)",  "value": f"{n_ketok}/40 — Ketogenesis INTACT",              "color": "#2e7d32"},
            "avg_mgc":      {"label": "Avg 3-MGC (mmol/mol Cr)", "value": str(avg_mgc),                                     "color": "#bf360c"},
            "avg_mg":       {"label": "Avg 3-MG (mmol/mol Cr)",  "value": str(avg_mg),                                     "color": "#e65100"},
        },
        "phenotype_dist": pheno_dist,
        "ketogenesis_intact_note": (
            "AUH deficiency does NOT abolish ketogenesis. AUH (step 4) is UPSTREAM of HMGCL (step 5). "
            "The ketogenesis pathway (FA β-oxidation → Acetyl-CoA → HMGCS2 → HMG-CoA → HMGCL → acetoacetate) "
            "is FULLY INTACT in AUH deficiency — HMG-CoA from FA oxidation bypasses the AUH block entirely. "
            "ALL 40 cohort patients maintained normal ketone production. "
            "This is the critical clinical distinction from HMGCL deficiency (where ketogenesis is abolished). "
            "AUH patients do NOT present with hypoketotic hypoglycaemia during fasting/illness — "
            "they are protected by intact ketone body production."
        ),
        "mga_type_note": (
            "AUH deficiency = 3-Methylglutaconic Aciduria Type I (MGA-I) — the ONLY MGA type caused by "
            "a leucine catabolism enzyme defect. All other MGA types involve mitochondrial membrane integrity "
            "(TAZ/Barth = MGA-II; DNAJC19 = MGA-IIIB; OPA3 = MGA-III; ATPAF2 = MGA-IV; SERAC1 = MGA-V). "
            "Key examiner point: MGA-I (AUH) 3-MGC levels are MODERATELY elevated (20–220 mmol/mol Cr), "
            "whereas MGA-II/III/V can also show elevated 3-MGC. Distinguishing features: "
            "AUH = AR both sexes, NO cardiomyopathy, NO neutropenia, NO optic atrophy."
        ),
        "hallmark_biomarker": (
            "3-MGC (3-methylglutaconic acid) ELEVATED 20–220 mmol/mol Cr — PRIMARY MARKER (urine OA GC-MS). "
            "3-MG (3-methylglutaric acid) ELEVATED secondary. "
            "3-HMG NORMAL (<10 mmol/mol Cr) — KEY NEGATIVE vs HMGCL. "
            "Ketones PRESENT/NORMAL — KEY NEGATIVE vs HMGCL. "
            "3-MCG ABSENT — KEY NEGATIVE vs 3-MCC. "
            "C3 NORMAL — KEY NEGATIVE vs HLCS/PA. "
            "NH3 NORMAL — KEY NEGATIVE vs UCDs."
        ),
        "vpa_risk_note": (
            "Valproic acid is MODERATE RISK in AUH deficiency (NOT absolute CI unlike HMGCL or IVA). "
            "VPA does NOT directly inhibit AUH. Risk profile: "
            "(1) Carnitine depletion via valproyl-carnitine excretion — worsens secondary carnitine deficiency. "
            "(2) Hepatotoxicity in IEMs — monitor LFTs. "
            "(3) Worsened 3-MGC accumulation possible during VPA-mediated carnitine depletion. "
            f"In this cohort, {n_vpa} patients received VPA — monitor carefully but not an absolute CI."
        ),
    }


def get_breakdown():
    biomarkers = {
        "mgc": {
            "label": "3-MGC (3-Methylglutaconic acid)",
            "normal": "<10 mmol/mol Cr",
            "status": "ELEVATED — 20–220 mmol/mol Cr — PRIMARY MARKER (urine OA GC-MS)",
            "direction": "↑↑ PRIMARY",
            "color": "danger",
            "rationale": (
                "3-Methylglutaconyl-CoA CANNOT be hydrated by defective AUH → thioester hydrolysis → "
                "free 3-methylglutaconate (3-MGC) in urine. Detected by GC-MS urine organic acids. "
                "Less dramatically elevated than in MGA-II (Barth), which can show >500 mmol/mol Cr. "
                "AUH-MGA-I: 3-MGC typically 20–220 mmol/mol Cr (moderate elevation). "
                "3-MGC elevated in ALL cases of AUH deficiency regardless of clinical severity. "
                "Accompanied by 3-MG (secondary, ~40–50% of 3-MGC level)."
            ),
        },
        "mg": {
            "label": "3-MG (3-Methylglutaric acid)",
            "normal": "<5 mmol/mol Cr",
            "status": "ELEVATED — 10–90 mmol/mol Cr (secondary marker, proportional to 3-MGC)",
            "direction": "↑↑",
            "color": "warning",
            "rationale": (
                "Secondary metabolite arising from decarboxylation/reduction of 3-MGC. "
                "3-MG:3-MGC ratio in AUH deficiency is typically 0.4–0.7 (vs HMGCL where 3-MG:3-HMG ~0.2). "
                "Co-elevation of 3-MGC and 3-MG on urine OA is the diagnostic fingerprint of MGA-I. "
                "3-MG alone is nonspecific (also in HMGCL, barth, other 3-MGC acidurias)."
            ),
        },
        "hmg_3": {
            "label": "3-HMG (3-Hydroxy-3-methylglutaric acid)",
            "normal": "<10 mmol/mol Cr",
            "status": "NORMAL — <10 mmol/mol Cr — KEY NEGATIVE vs HMGCL (most critical differentiator)",
            "direction": "→ NORMAL (KEY NEGATIVE)",
            "color": "success",
            "rationale": (
                "HMGCL (step 5) is INTACT in AUH deficiency. AUH (step 4) is UPSTREAM of HMGCL. "
                "HMG-CoA formed from FA β-oxidation (via HMGCS2) is cleaved normally by HMGCL → "
                "3-HMG NOT accumulated. 3-HMG NORMAL in AUH is the single most important test to "
                "distinguish AUH deficiency from HMGCL deficiency (where 3-HMG >200–2500 mmol/mol Cr). "
                "If 3-HMG is elevated: the diagnosis is HMGCL deficiency, NOT AUH."
            ),
        },
        "ketones": {
            "label": "Ketones (β-OHB / Acetoacetate)",
            "normal": "Present during fasting / low glucose",
            "status": "PRESENT / NORMAL — Ketogenesis INTACT — KEY NEGATIVE vs HMGCL",
            "direction": "→ NORMAL (KEY NEGATIVE)",
            "color": "success",
            "rationale": (
                "AUH (step 4) is UPSTREAM of HMGCL (step 5). Ketogenesis requires HMGCL to convert "
                "HMG-CoA → acetoacetate. In AUH deficiency, HMGCL is fully functional. "
                "The ketogenesis substrate (HMG-CoA) still arrives via FA β-oxidation → HMGCS2 — "
                "this bypasses the AUH block entirely (leucine path step 4 is irrelevant to ketogenesis). "
                "All 40 cohort patients maintained normal ketone production on fasting challenge. "
                "NO hypoketotic hypoglycaemia (the hallmark of HMGCL deficiency is absent here)."
            ),
        },
        "c5oh": {
            "label": "C5-OH (3-Hydroxy-isovalerylcarnitine)",
            "normal": "<0.4 µmol/L",
            "status": "BORDERLINE–MILDLY ELEVATED — 0.3–2.5 µmol/L (less than MCC/HMGCL)",
            "direction": "↑ (borderline)",
            "color": "warning",
            "rationale": (
                "Mild elevation of C5-OH can occur in AUH deficiency from overflow of leucine "
                "catabolism intermediates (secondary to 3-methylglutaconyl-CoA block). "
                "Significantly less elevated than in 3-MCC (C5-OH the primary NBS marker) or HMGCL. "
                "C5-OH + 3-MGC ↑ but NO 3-MCG and NO 3-HMG → points to AUH over MCC and HMGCL."
            ),
        },
        "mcg": {
            "label": "3-MCG (3-Methylcrotonylglycine)",
            "normal": "<5 mmol/mol Cr",
            "status": "ABSENT — KEY NEGATIVE vs 3-MCC (3-methylcrotonyl-CoA does not accumulate in AUH)",
            "direction": "→ ABSENT",
            "color": "success",
            "rationale": (
                "AUH deficiency blocks step 4 (3-MGC-CoA → HMG-CoA). Step 3 (3-MC-CoA → 3-MGC-CoA by MCC) "
                "remains intact. 3-methylcrotonyl-CoA is converted to 3-methylglutaconyl-CoA normally by MCC "
                "(no backup accumulation). Therefore 3-MCG (glycine conjugate of 3-MC-CoA) is ABSENT. "
                "3-MCG elevated only when MCC is defective (step 3). AUH = step 4, downstream of MCC. "
                "3-MCG ABSENT rules out 3-MCC deficiency as the cause of C5-OH elevation."
            ),
        },
        "c3": {
            "label": "C3 (Propionylcarnitine)",
            "normal": "<5 µmol/L",
            "status": "NORMAL — KEY NEGATIVE vs HLCS (multiple carboxylase deficiency)",
            "direction": "→ NORMAL",
            "color": "success",
            "rationale": (
                "HLCS deficiency causes failure of ALL biotin-dependent carboxylases: PCC (C3 ↑) + MCC (C5-OH ↑). "
                "AUH is NOT biotin-dependent; AUH deficiency does NOT affect PCC. "
                "C3 NORMAL = NOT HLCS. Also differentiates from propionic acidemia (C3 ↑↑)."
            ),
        },
        "nh3": {
            "label": "Ammonia (NH3)",
            "normal": "<50 µmol/L",
            "status": "NORMAL — KEY NEGATIVE vs UCDs (no urea cycle involvement)",
            "direction": "→ NORMAL",
            "color": "success",
            "rationale": (
                "AUH deficiency does not involve the urea cycle. NH3 is NOT elevated. "
                "AUH encodes a leucine catabolism enzyme; nitrogen metabolism is unaffected. "
                "NH3 normal differentiates AUH from ornithine transcarbamylase deficiency (OTC), "
                "carbamoyl phosphate synthetase 1 (CPS1), and other UCDs."
            ),
        },
        "free_carnitine": {
            "label": "Free Carnitine (C0)",
            "normal": "25–55 µmol/L",
            "status": "MILDLY–MODERATELY LOW — secondary depletion (less severe than HMGCL/IVA)",
            "direction": "↓ (mild)",
            "color": "warning",
            "rationale": (
                "Acylcarnitine conjugation of 3-methylglutaconyl-CoA and 3-methylglutaryl-CoA "
                "depletes the free carnitine pool, but less severely than in HMGCL (C6-DC) or IVA (C5). "
                "L-carnitine supplementation is Level A treatment to maintain adequate free carnitine."
            ),
        },
    }

    enzyme_mechanism = {
        "function": (
            "AUH (AUH gene, 9q22.31) encodes a 338-aa bifunctional mitochondrial matrix protein:\n"
            "  (1) AU-rich element RNA-binding protein (N-terminal domain) — post-transcriptional regulation\n"
            "  (2) 3-Methylglutaconyl-CoA hydratase (C-terminal catalytic domain) — leucine catabolism\n"
            "The hydratase function catalyses a REVERSIBLE hydration reaction:\n"
            "  3-Methylglutaconyl-CoA + H₂O  ⇌  HMG-CoA   (enoyl-CoA hydratase mechanism)\n"
            "AUH is homologous to crotonase superfamily enoyl-CoA hydratases.\n"
            "Homodimeric; CoA-binding pocket flanked by catalytic Glu and His residues."
        ),
        "reaction": (
            "3-Methylglutaconyl-CoA + H₂O  ⇌  [AUH]  ⇌  HMG-CoA\n"
            "(Reversible equilibrium hydration; Step 4 of leucine catabolism)\n"
            "AUH LOF → 3-Methylglutaconyl-CoA ACCUMULATES → hydrolysis → 3-MGC (urine) + 3-MG"
        ),
        "block": (
            "AUH LOF → 3-Methylglutaconyl-CoA CANNOT be hydrated → ACCUMULATES:\n"
            "  → Free acid hydrolysis → 3-MGC (3-methylglutaconate) ↑↑ — PRIMARY urinary marker\n"
            "  → 3-MG (3-methylglutarate) ↑ — secondary decarboxylation product\n"
            "CRITICAL: AUH is UPSTREAM of HMGCL (step 5):\n"
            "  → HMGCL remains INTACT → hepatic ketogenesis via FA oxidation → HMGCS2 → HMG-CoA → HMGCL → NORMAL\n"
            "  → NO hypoketotic hypoglycaemia (fundamental distinction from HMGCL deficiency)"
        ),
        "leucine_path": (
            "L-Leucine → BCAT → KIC → BCKDH → Isovaleryl-CoA → [IVD, step 2] → "
            "3-Methylcrotonyl-CoA → [MCC, step 3] → 3-Methylglutaconyl-CoA → "
            "[AUH BLOCKED ⚠ — STEP 4] → ↛ HMG-CoA → [HMGCL step 5, INTACT] → Acetoacetate + Acetyl-CoA"
        ),
        "ketogenesis_note": (
            "Hepatic ketogenesis pathway (INTACT in AUH deficiency):\n"
            "FA β-oxidation → Acetyl-CoA → [HMGCS2] → HMG-CoA → [HMGCL — intact!] → Acetoacetate + β-OHB\n"
            "This FA-derived HMG-CoA bypasses the AUH block entirely. "
            "AUH deficiency does NOT compromise hepatic ketone body synthesis — "
            "the brain and other tissues receive normal ketone backup during fasting."
        ),
    }

    seizure_types = [
        {"type": "Focal seizures (leucine catabolism metabolite toxicity)", "pct": 55, "note": "3-MGC neurotoxicity proposed; focal > generalised; LEV first-line AED"},
        {"type": "Generalised tonic-clonic (moderate-severe phenotype)",    "pct": 25, "note": "Moderate–severe cohort; NOT due to hypoglycaemia (contrast HMGCL); metabolic origin"},
        {"type": "Myoclonic seizures",                                       "pct": 12, "note": "Reported in severe spectrum; typically respond to LEV or CLB"},
        {"type": "Drug-resistant epilepsy (DRE)",                           "pct": 12, "note": "10–15% of AUH patients; significantly less than HMGCL (>60% DRE in severe cases)"},
    ]

    treatments = [
        {
            "therapy": "Leucine-restricted diet",
            "level": "A",
            "dose": "Natural protein 0.8–1.5 g/kg/day; leucine <120 mg/kg/day; AUH amino-acid formula",
            "rationale": "Reduces 3-methylglutaconyl-CoA substrate from leucine catabolism. Primary long-term strategy. Titrate by plasma leucine (target 80–200 µmol/L) and urine 3-MGC levels.",
        },
        {
            "therapy": "L-Carnitine supplementation",
            "level": "A",
            "dose": "100–200 mg/kg/day oral; may increase during illness",
            "rationale": "Secondary carnitine depletion from 3-methylglutaconyl/3-methylglutaryl-carnitine excretion. Replenishes free carnitine pool. Less severe depletion than HMGCL/IVA but still Level A.",
        },
        {
            "therapy": "Fasting avoidance",
            "level": "B",
            "dose": "Max fasting: 6–8 h (infant/child); glucose polymer drinks during illness; sick-day plan",
            "rationale": "Catabolism releases leucine → worsens 3-MGC accumulation. Less critical than HMGCL (no hypoketotic crisis risk) but reduces substrate flux during illness. Written sick-day protocol recommended.",
        },
        {
            "therapy": "Levetiracetam (LEV)",
            "level": "B",
            "dose": "20–40 mg/kg/day in 2 divided doses",
            "rationale": "First-line AED for AUH-associated seizures. No carnitine depletion (unlike VPA). No direct AUH interaction. Safe profile in metabolic IEMs.",
        },
        {
            "therapy": "Valproic Acid (VPA)",
            "level": "MODERATE RISK",
            "dose": "Use with caution if LEV insufficient; monitor carnitine, LFTs",
            "rationale": "VPA does NOT directly inhibit AUH (contrast HMGCL where VPA inhibits active site via valproyl-CoA). Risk in AUH: (1) carnitine depletion; (2) hepatotoxicity in IEMs. NOT an absolute CI — may be used if needed with monitoring.",
        },
        {
            "therapy": "Biotin supplementation",
            "level": "NOT EFFECTIVE",
            "dose": "N/A — do not use for AUH",
            "rationale": "AUH is NOT biotin-dependent (it is an enoyl-CoA hydratase, not a biotin-dependent carboxylase). Biotin has no therapeutic role in AUH deficiency. A positive biotin trial would suggest HLCS or BTD, not AUH.",
        },
        {
            "therapy": "Ketogenic Diet",
            "level": "NOT ABSOLUTE CI (caution)",
            "dose": "Use with caution; monitor 3-MGC; leucine load must be controlled",
            "rationale": "Ketogenesis is INTACT in AUH deficiency (HMGCL intact). KD is not absolutely contraindicated as in HMGCL. However, KD increases leucine mobilisation → worsens substrate load. Not routinely used; can be considered for refractory DRE with close metabolic monitoring.",
        },
    ]

    systemic_features = [
        {"feature": "Intellectual disability (mild–moderate)",  "pct": 58, "note": "Most common; variable degree; better outcome with early leucine restriction"},
        {"feature": "Speech delay",                             "pct": 68, "note": "Often the first clinical sign; onset 6–18 months delayed; expressive > receptive"},
        {"feature": "Hypotonia",                                "pct": 42, "note": "Generalised; severity correlates with phenotypic class; improves with leucine restriction"},
        {"feature": "Seizures",                                 "pct": 35, "note": "Focal > generalised; NOT related to hypoglycaemia; LEV first-line"},
        {"feature": "Asymptomatic (incidental / NBS)",          "pct": 30, "note": "Purely biochemical; no clinical signs; requires dietary monitoring only"},
        {"feature": "Autistic features",                        "pct": 22, "note": "Behavioural overlap; social communication delay; more prominent in moderate-severe"},
        {"feature": "Drug-resistant epilepsy",                  "pct": 12, "note": "Seen in severe subset; much rarer than HMGCL-DRE"},
        {"feature": "Mild metabolic acidosis (febrile illness)", "pct": 20, "note": "Transient; NOT the acute Reye-like crisis seen in HMGCL/IVA/PA; mild bicarbonate drop"},
        {"feature": "Cardiomyopathy",                           "pct": 0,  "note": "ABSENT — KEY NEGATIVE vs Barth syndrome (MGA-II, TAZ deficiency)"},
        {"feature": "Neutropenia",                              "pct": 0,  "note": "ABSENT — KEY NEGATIVE vs Barth syndrome (MGA-II); helps confirm AUH over TAZ"},
    ]

    cohort_preview = _COHORT[:10]

    return {
        "biomarkers": biomarkers,
        "enzyme_mechanism": enzyme_mechanism,
        "variants": VARIANTS,
        "seizure_types": seizure_types,
        "treatments": treatments,
        "systemic_features": systemic_features,
        "cohort_preview": cohort_preview,
    }


def get_definitions():
    return {
        "AUH (3-Methylglutaconyl-CoA Hydratase / AU-rich element RNA-binding protein)": (
            "Bifunctional mitochondrial matrix protein, 338 aa (gene: AUH, 9q22.31, AR). "
            "C-terminal catalytic domain: enoyl-CoA hydratase (crotonase superfamily). "
            "Reaction: 3-Methylglutaconyl-CoA + H₂O ⇌ HMG-CoA  (REVERSIBLE hydration, Step 4 leucine catabolism). "
            "N-terminal AU-rich element RNA-binding domain: post-transcriptional mRNA regulation "
            "(functional role; not responsible for the metabolic phenotype when absent). "
            "Homodimeric; catalytic Glu and His residues. OMIM Gene *600529."
        ),
        "AUH Deficiency / 3-Methylglutaconic Aciduria Type I (MGA-I)": (
            "Autosomal recessive IEM (biallelic LOF in AUH gene, 9q22.31). "
            "OMIM Disease #250950 (3-Methylglutaconic Aciduria, Type I). "
            "Prevalence: ~1:500,000–1,000,000 (very rare; <50 cases worldwide 2026). "
            "Leucine catabolism STEP 4 block → 3-methylglutaconyl-CoA ACCUMULATES → 3-MGC ↑ + 3-MG ↑. "
            "3 phenotypes: Asymptomatic NBS-detected (30%), Mild speech delay/mild ID (45%), "
            "Moderate-Severe DRE (25%)."
        ),
        "3-MGC (3-Methylglutaconic acid) — Primary Marker": (
            "ELEVATED (20–220 mmol/mol Cr) in AUH deficiency — primary urinary marker (GC-MS urine OA). "
            "Formed by hydrolysis of accumulated 3-methylglutaconyl-CoA → free 3-methylglutaconate. "
            "Reference: <10 mmol/mol Cr. NOT as extreme as MGA-II (Barth >500 mmol/mol Cr possible). "
            "3-MGC elevated in ALL 5 MGA types — AUH (I), TAZ (II), OPA3/DNAJC19 (III), ATPAF2 (IV), SERAC1 (V). "
            "Distinguishing MGA-I from others requires clinical context + additional biomarkers."
        ),
        "3-HMG NORMAL — KEY NEGATIVE vs HMGCL": (
            "3-Hydroxy-3-methylglutaric acid (3-HMG) is the PATHOGNOMONIC marker of HMGCL deficiency. "
            "In AUH deficiency, AUH (step 4) is UPSTREAM of HMGCL (step 5). "
            "HMGCL is fully functional → HMG-CoA (from FA oxidation/HMGCS2) is cleaved normally → 3-HMG NOT accumulated. "
            "3-HMG NORMAL (<10 mmol/mol Cr) in AUH = critical evidence against HMGCL deficiency. "
            "If 3-HMG is elevated → suspect HMGCL deficiency; if 3-HMG is normal + 3-MGC elevated → AUH deficiency."
        ),
        "Ketogenesis INTACT in AUH (KEY NEGATIVE vs HMGCL)": (
            "AUH (step 4) is UPSTREAM of HMGCL (step 5). Hepatic ketogenesis route: "
            "FA β-oxidation → Acetyl-CoA → [HMGCS2] → HMG-CoA → [HMGCL — INTACT] → Acetoacetate → β-OHB. "
            "This pathway does NOT involve AUH at all (HMG-CoA sourced from HMGCS2, not leucine step 4). "
            "AUH deficiency: ketones PRESENT/NORMAL during fasting; no hypoketotic hypoglycaemia. "
            "HMGCL deficiency: ketones ABSENT; hypoketotic hypoglycaemia is the hallmark (HMGCL abolished). "
            "Absent ketones = HMGCL; present ketones = NOT HMGCL → AUH is possible."
        ),
        "MGA Type I vs Other MGA Types (Critical Differential)": (
            "All 5 MGA types show elevated 3-MGC on urine OA — but mechanisms differ:\n"
            "MGA-I  (AUH, 9q22.31, AR): leucine catabolism block; step 4; both sexes; no cardiomyopathy; no neutropenia\n"
            "MGA-II (TAZ, Xq28, X-linked): tafazzin = cardiolipin remodelling; Barth syndrome; males; cardiomyopathy + neutropenia\n"
            "MGA-III (OPA3, 19q13.32, AR): OPA3 = OMM GTPase; autosomal; optic atrophy + chorea\n"
            "MGA-IIIB (DNAJC19, 3q26.33, AR): DNAJC19 = IMS import chaperone; dilated CM + non-progressive CA; males\n"
            "MGA-IV (ATPAF2, 17p11.2, AR): complex V assembly; lactic acidosis + HSD + complex V deficiency\n"
            "MGA-V (SERAC1, 6q22.1, AR): phospholipid remodelling; MEGDEL = deafness-dystonia-hepatopathy-3-MG\n"
            "Rule: 3-MGC elevated + leucine restriction responsive + no cardiac/eye/hearing → consider AUH (MGA-I)"
        ),
        "Leucine Catabolism — Step 4 Context (AUH)": (
            "L-Leucine (essential BCAA) catabolism, 5 steps:\n"
            "Step 1a: L-Leucine → KIC via BCAT (reversible transamination)\n"
            "Step 1b: KIC → Isovaleryl-CoA + CO₂ via BCKDH (irreversible; MSUD if deficient)\n"
            "Step 2:  Isovaleryl-CoA → 3-Methylcrotonyl-CoA via IVD (FAD-dependent; IVA if deficient)\n"
            "Step 3:  3-MC-CoA → 3-Methylglutaconyl-CoA via MCC (biotin; MCC deficiency upstream)\n"
            "Step 4:  3-Methylglutaconyl-CoA → HMG-CoA via AUH  [AUH BLOCKED HERE — STEP 4]\n"
            "Step 5:  HMG-CoA → Acetoacetate + Acetyl-CoA via HMGCL (intact; HMGCL deficiency downstream)\n"
            "Net result: AUH block → 3-MGC ↑ + 3-MG ↑; HMGCL intact → ketogenesis OK; 3-HMG NORMAL"
        ),
        "Inheritance & Epidemiology": (
            "Autosomal recessive (AR); biallelic LOF in AUH gene (9q22.31). "
            "Prevalence: ~1:500,000–1,000,000 (extremely rare; <50 cases worldwide 2026). "
            "No major founder mutations described; private variants predominate. "
            "Both sexes equally affected (AR; contrast MGA-II/Barth which is X-linked, males predominantly). "
            "Many cases identified by NBS via mildly elevated C5-OH or 3-MGC on expanded NBS panels. "
            "Clinical outcome improved significantly by early leucine restriction + carnitine supplementation."
        ),
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:3000])
    print("\n=== BREAKDOWN keys ===", list(get_breakdown().keys()))
    print("\n=== DEFINITIONS keys ===", list(get_definitions().keys()))
    print(f"\n✓ AUH dashboard: {N_PATIENTS} patients, seed={SEED}")
