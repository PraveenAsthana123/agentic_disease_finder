#!/usr/bin/env python3
"""HMGCL (3-Hydroxy-3-methylglutaryl-CoA Lyase) Deficiency Dashboard.

HMGCL is the terminal enzyme of leucine catabolism AND the sole enzyme of hepatic ketogenesis:
    HMG-CoA  →  [HMGCL]  →  Acetoacetate + Acetyl-CoA

HMGCL LOF → HMG-CoA CANNOT be cleaved → DUAL FAILURE:
  (1) Leucine catabolism is BLOCKED at step 5 → HMG-related organic acids accumulate
  (2) Hepatic KETOGENESIS is ABOLISHED → HYPOKETOTIC HYPOGLYCAEMIA (hallmark feature)

KEY FACTS (EXAM HIGHEST-YIELD):
  1. HYPOKETOTIC HYPOGLYCAEMIA — hallmark; glucose LOW + ketones ABSENT (contrast normal starvation)
  2. 3-HMG (3-hydroxy-3-methylglutaric acid) >200 mmol/mol Cr — PATHOGNOMONIC
  3. NO KETOGENIC DIET — absolutely contraindicated (KD requires ketone synthesis; HMGCL absent)
  4. VPA — ABSOLUTE CI (directly inhibits HMGCL at active site + carnitine depletion + hepatotoxic)
  5. Fasting — EXTREME HAZARD (triggers hypoketotic hypoglycaemia + HMG-CoA surge)
  6. Saudi Arabia / Middle East / Spain: founder mutations (p.Arg41Gln; p.Phe305Ile) — highest prevalence
  7. C5-OH on NBS: 3-MCC most common cause; HMGCL = C5-OH ↑ + C6-DC (HMG-related) ↑; NO 3-MCG
  8. Reye-like syndrome: acute hepatic dysfunction + encephalopathy + hypoglycaemia — classic acute presentation

OMIM Disease: #246450 (3-Hydroxy-3-methylglutaryl-CoA Lyase Deficiency)
OMIM Gene:    *600234 (HMGCL)
Chromosome:   1p36.11
Inheritance:  Autosomal Recessive (AR), biallelic LOF
Protein:      325 aa (mature form after MTS cleavage of 27-aa MTS); homotrimeric;
              mitochondrial matrix; Cys266 = catalytic nucleophile (thioester intermediate)
Prevalence:   ~1:100,000–300,000 general; ~1:20,000 Saudi Arabia (p.Arg41Gln founder)

LEUCINE CATABOLISM CONTEXT (where HMGCL fits — STEP 5, the TERMINAL STEP):
  L-Leucine  → BCAT (Step 1a) → KIC (α-ketoisocaproate)
  KIC        → BCKDH (Step 1b) → Isovaleryl-CoA
  Isovaleryl-CoA → IVD (Step 2) → 3-Methylcrotonyl-CoA     [IVD deficiency upstream]
  3-MC-CoA   → MCC (Step 3) → 3-Methylglutaconyl-CoA       [MCC deficiency upstream]
  3-MG-CoA   → AUH (Step 4) → HMG-CoA                     [AUH deficiency very rare]
  HMG-CoA    → HMGCL (Step 5) → Acetoacetate + Acetyl-CoA  [THIS ENZYME — FINAL STEP]

HMG-CoA also arises from hepatic de novo ketogenesis (fatty acid β-oxidation):
  Acetyl-CoA + Acetoacetyl-CoA → HMGS2 → HMG-CoA → HMGCL → ketone bodies
  HMGCL LOF → HEPATIC KETOGENESIS IMPOSSIBLE (regardless of FA oxidation status)

BIOMARKER PATTERN — HMGCL DEFICIENCY:
  3-HMG (3-hydroxy-3-methylglutaric acid)   >200 mmol/mol Cr — PATHOGNOMONIC (urine OA)
  3-MG  (3-methylglutaric acid)             ELEVATED — secondary (HMG-CoA hydrolysis)
  3-MGC (3-methylglutaconic acid)           mildly elevated (AUH product backed up)
  3-HIV (3-hydroxy-isovalerate)             ELEVATED — shared with MCC but NO 3-MCG
  C6-DC (3-methylglutarylcarnitine)         ELEVATED — NBS marker (more specific than C5-OH)
  C5-OH (3-hydroxy-isovalerylcarnitine)     ELEVATED — NBS Tier-1 trigger (less specific)
  Acetoacetate / β-OHB (ketones)            ABSENT OR VERY LOW — KEY FINDING (hypoketotic)
  Blood glucose                             LOW during crises (<2.5 mmol/L) — hypoketotic hypoglycaemia
  Lactate                                   may be elevated (secondary energy failure)
  Liver enzymes (ALT/AST/GGT)              ELEVATED acutely — hepatic involvement
  Free carnitine                            LOW (secondary depletion, C6-DC conjugation)
  Plasma ammonia                            may be mildly elevated during crises (NOT primary UCD)

KEY NEGATIVE MARKERS (CRITICAL DIFFERENTIALS):
  3-MCG (3-methylcrotonylglycine)  ABSENT — KEY NEGATIVE vs 3-MCC (MOST IMPORTANT NBS C5-OH differentiator)
  NH3 (persistent)                 NORMAL — KEY NEGATIVE vs UCDs
  C3 (propionylcarnitine)          NORMAL — KEY NEGATIVE vs HLCS/PA
  Biotinidase activity             NORMAL — KEY NEGATIVE vs BTD
  Homocysteine                     NORMAL — KEY NEGATIVE vs MMACHC/cblC
  MMA                              ABSENT — KEY NEGATIVE vs MMUT/PA
  Methylcitrate                    ABSENT — KEY NEGATIVE vs PA

NBS C5-OH + C6-DC DIFFERENTIAL:
  HMGCL:      C5-OH ↑ + C6-DC ↑ (HMG-related) + hypoglycaemia; NO 3-MCG — HMGCL is the answer when C6-DC ↑
  3-MCC:      C5-OH ↑ + 3-MCG ↑↑ + 3-HIV ↑; C3 NORMAL; C6-DC NORMAL — 3-MCG absent rules out HMGCL
  HLCS:       C5-OH ↑ + C3 ↑ (all carboxylases); BIOTIN-RESPONSIVE; rash + ketoacidosis
  BTD:        C5-OH ↑ + rash + SNHL + optic atrophy; BIOTIN-RESPONSIVE; biotinidase assay LOW

PATHOGENESIS — HMG-CoA LYASE BLOCK:
  HMGCL cleaves the thioester bond of HMG-CoA using Cys266 as catalytic nucleophile:
    HMG-CoA + H₂O → Acetoacetate + Acetyl-CoA
  HMGCL LOF → HMG-CoA CANNOT be cleaved → ACCUMULATES:
    → 3-HMG (hydrolysis of HMG-CoA thioester) — PATHOGNOMONIC
    → 3-MG  (spontaneous decarboxylation of 3-HMG) — secondary
    → 3-MGC (retrograde from AUH equilibrium) — mild
    → C6-DC (carnitine conjugation of 3-methylglutaryl-CoA) — NBS marker
    → 3-HIV (side-pathway, shared with MCC deficiency)
  Simultaneous failure of KETOGENESIS:
    Hepatic FA β-oxidation intact → Acetyl-CoA → HMGCS2 → HMG-CoA → [BLOCKED] → NO ketones
    → Hypoketotic state despite hypoglycaemia (profound + dangerous)
    → Brain deprived of both glucose AND ketones during crisis → neurological crisis

TREATMENT:
  IV Glucose (anti-catabolic emergency):  LEVEL A — first-line in crisis; 10–15 mg/kg/min GIR
  Leucine restriction:                    LEVEL A — reduces HMG-CoA substrate flux
  Fasting avoidance:                      LEVEL A — max 4–6 h infant; written emergency protocol
  L-Carnitine (if depleted):              LEVEL A — secondary depletion via C6-DC excretion
  Low-leucine, low-fat diet:              LEVEL A — reduces both leucine AND ketogenic substrate
  Levetiracetam (LEV) — seizure Rx:       LEVEL B — first-line AED for hypoglycaemia-related seizures
  Valproic Acid (VPA):                    ABSOLUTE CI — inhibits HMGCL active site (Cys266) directly;
                                          carnitine depletion; hepatotoxic in IEM with liver involvement
  Ketogenic Diet (KD):                    ABSOLUTE CI — HMGCL required for ketone synthesis;
                                          KD floods with HMG-CoA precursors → crisis; futile + harmful
  Biotin:                                 NOT effective (HMGCL is not biotin-dependent)
  Fasting:                                ABSOLUTE HAZARD

PHENOTYPES:
  Neonatal / early onset (acute):  ~35% — severe; acute metabolic crisis day 1–5; coma; liver failure; death if missed
  Classic infantile (episodic):    ~50% — febrile/fasting trigger; Reye-like crises; main phenotype
  Late-onset / attenuated:         ~15% — milder; fewer crises; NBS-detected; favourable outcome

SEIZURES IN HMGCL:
  ~40–50% in symptomatic cases; primarily secondary to hypoketotic hypoglycaemia
  NOT primary epilepsy syndrome; seizure-free when metabolic control maintained
  Acute: generalised convulsions during hypoglycaemic crises
  Chronic: risk of focal cortical dysrhythmia if repeated metabolic brain injury
  CRITICAL: Differentiate from primary epilepsy — EEG non-specific; normalises with glucose
  VPA for epilepsy = ABSOLUTE CI in HMGCL — will worsen the metabolic disorder acutely

COMMON PATHOGENIC VARIANTS:
  p.Arg41Gln (c.122G>A):   ~25% — active-site adjacent; most common worldwide; Saudi Arabia founder
  p.Leu122Pro (c.365T>C):  ~15% — hydrophobic core; severe; neonatal onset
  p.Glu37Ter (c.109G>T):   ~12% — null (premature stop); severe; pan-ethnic
  p.Ala65Val (c.194C>T):   ~10% — near MTS cleavage site; moderate; European
  p.Phe305Ile (c.913T>A):  ~8%  — active-site channel; Spanish/Portuguese founder; attenuated
  c.IVS5+1G>A (splice):    ~6%  — exon 5 skip → truncation; null; severe
  p.Ser276Cys (c.827C>G):  ~5%  — Cys266 active-site neighbourhood; moderate
"""

import random

SEED       = 261      # next after MCC (seed 259), MCC→HMGCL leucine step 5
N_PATIENTS = 40

# ── Phenotype classes ─────────────────────────────────────────────────────────
PHENOTYPE_CLASSES = [
    {
        "class": "Classic Infantile / Episodic (Reye-like crises)",
        "pct": 50,
        "age_onset_months_range": (3, 18),
        "hmg_urine_range": (250, 1200),        # mmol/mol Cr — 3-HMG, pathognomonic
        "c6dc_range": (1.0, 6.0),              # µmol/L — 3-methylglutarylcarnitine, NBS
        "c5oh_range": (0.5, 4.0),              # µmol/L — NBS secondary trigger
        "glucose_crisis_range": (0.8, 2.3),    # mmol/L during crisis (LOW)
        "free_carn_range": (8, 22),
        "lactate_range": (2.0, 6.0),
        "nh3_crisis_range": (55, 180),          # may rise secondary in crisis
        "seizures_prob": 0.45,
        "crises_range": (2, 8),
        "hepatomegaly_prob": 0.75,
        "note": "Febrile/fasting trigger; Reye-like acute; vomiting, lethargy, hepatomegaly; responds to IV glucose",
    },
    {
        "class": "Neonatal / Early Onset (Severe Acute)",
        "pct": 35,
        "age_onset_months_range": (0, 1),
        "hmg_urine_range": (500, 2500),
        "c6dc_range": (3.0, 12.0),
        "c5oh_range": (1.5, 8.0),
        "glucose_crisis_range": (0.3, 1.5),
        "free_carn_range": (4, 15),
        "lactate_range": (4.0, 12.0),
        "nh3_crisis_range": (80, 300),
        "seizures_prob": 0.65,
        "crises_range": (1, 4),
        "hepatomegaly_prob": 0.90,
        "note": "Day 1–5; profound hypoglycaemia + hyperammonaemia + metabolic acidosis; high mortality if untreated",
    },
    {
        "class": "Late-Onset / Attenuated (NBS-detected or mild)",
        "pct": 15,
        "age_onset_months_range": (18, 60),
        "hmg_urine_range": (80, 350),
        "c6dc_range": (0.4, 1.8),
        "c5oh_range": (0.4, 1.5),
        "glucose_crisis_range": (1.5, 2.8),
        "free_carn_range": (16, 35),
        "lactate_range": (1.0, 3.0),
        "nh3_crisis_range": (30, 80),
        "seizures_prob": 0.20,
        "crises_range": (0, 2),
        "hepatomegaly_prob": 0.40,
        "note": "Milder course; p.Phe305Ile founder often attenuated; good outcome with dietary management",
    },
]

# Common pathogenic variants with frequency, domain, severity
VARIANTS = [
    {"variant": "p.Arg41Gln",   "freq": 25, "domain": "Active-site adjacent (α2-helix)", "phenotype": "Moderate–severe; Saudi Arabia founder; most common worldwide", "note": "c.122G>A; reduces HMG-CoA binding affinity; partial activity retained"},
    {"variant": "p.Leu122Pro",  "freq": 15, "domain": "Hydrophobic core (β4-strand)",   "phenotype": "Severe; neonatal onset; complete loss of HMGCL activity",       "note": "c.365T>C; disrupts trimer interface; protein folding defect"},
    {"variant": "p.Glu37Ter",   "freq": 12, "domain": "Null (premature stop, exon 2)",  "phenotype": "Severe; pan-ethnic; null allele",                               "note": "c.109G>T; no functional protein produced; neonatal crisis"},
    {"variant": "p.Ala65Val",   "freq": 10, "domain": "Near MTS cleavage site",         "phenotype": "Moderate; European; variable onset",                            "note": "c.194C>T; partial mitochondrial import impairment"},
    {"variant": "p.Phe305Ile",  "freq": 8,  "domain": "Active-site channel (C-term)",   "phenotype": "Attenuated; Spanish/Portuguese founder; late-onset",            "note": "c.913T>A; reduces catalytic efficiency; residual activity ~15%"},
    {"variant": "c.IVS5+1G>A",  "freq": 6,  "domain": "Splice site (intron 5)",         "phenotype": "Severe; exon 5 skip → truncation",                             "note": "Null via aberrant splicing; loss of exon encoding catalytic core"},
    {"variant": "p.Ser276Cys",  "freq": 5,  "domain": "Cys266 active-site neighbourhood","phenotype": "Moderate; disrupts catalytic Cys266 environment",             "note": "c.827C>G; thioester intermediate formation impaired"},
]


def _gen_cohort():
    """Generate synthetic 40-patient cohort with HMGCL-realistic biomarker profiles."""
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
        pid = f"HMGCL-{i+1:03d}"

        # Variant selection (weighted by frequency)
        var_pool = []
        for v in VARIANTS:
            var_pool.extend([v["variant"]] * v["freq"])
        variant = random.choice(var_pool)

        onset_mo      = round(random.uniform(*cls["age_onset_months_range"]), 1)
        hmg           = round(random.uniform(*cls["hmg_urine_range"]), 1)
        c6dc          = round(random.uniform(*cls["c6dc_range"]), 2)
        c5oh          = round(random.uniform(*cls["c5oh_range"]), 2)
        glucose_min   = round(random.uniform(*cls["glucose_crisis_range"]), 2)
        carn          = round(random.uniform(*cls["free_carn_range"]), 1)
        lactate       = round(random.uniform(*cls["lactate_range"]), 1)
        nh3           = round(random.uniform(*cls["nh3_crisis_range"]), 1)
        seiz          = random.random() < cls["seizures_prob"]
        n_cris        = random.randint(*cls["crises_range"])
        hepato        = random.random() < cls["hepatomegaly_prob"]
        ketones_absent = True   # hallmark — ALWAYS absent in HMGCL during crisis
        vpa_tried     = (cls["class"].startswith("Neonatal") or cls["class"].startswith("Classic")) \
                        and random.random() < 0.15   # occasionally given before diagnosis

        # 3-MG secondary metabolite (roughly 20–30% of 3-HMG level)
        mg_urine = round(hmg * random.uniform(0.18, 0.32), 1)

        patients.append({
            "id": pid,
            "phenotype": cls["class"],
            "variant": variant,
            "age_onset_months": onset_mo,
            "hmg_urine_mmol_mol_cr": hmg,           # 3-HMG — PATHOGNOMONIC
            "mg_urine_mmol_mol_cr": mg_urine,        # 3-MG secondary
            "c6dc_umol_l": c6dc,                     # NBS marker
            "c5oh_umol_l": c5oh,                     # NBS secondary
            "glucose_crisis_mmol_l": glucose_min,    # LOW during crisis
            "free_carnitine_umol_l": carn,
            "lactate_mmol_l": lactate,
            "nh3_umol_l": nh3,
            "ketones_absent": ketones_absent,
            "seizures": seiz,
            "crisis_count": n_cris,
            "hepatomegaly": hepato,
            "vpa_given_pre_dx": vpa_tried,
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

    n_seiz    = sum(1 for p in _COHORT if p["seizures"])
    n_cris    = sum(1 for p in _COHORT if p["crisis_count"] > 0)
    n_hepato  = sum(1 for p in _COHORT if p["hepatomegaly"])
    n_low_c   = sum(1 for p in _COHORT if p["free_carnitine_umol_l"] < 20)
    n_vpa     = sum(1 for p in _COHORT if p["vpa_given_pre_dx"])
    avg_hmg   = round(sum(p["hmg_urine_mmol_mol_cr"] for p in _COHORT) / N_PATIENTS, 1)
    avg_gluc  = round(sum(p["glucose_crisis_mmol_l"] for p in _COHORT) / N_PATIENTS, 2)

    return {
        "disease": "HMGCL Deficiency (3-Hydroxy-3-methylglutaryl-CoA Lyase Deficiency / HMG-CoA Lyase Deficiency)",
        "gene": "HMGCL",
        "omim_gene": "600234",
        "omim_disease": "246450",
        "locus": "1p36.11",
        "inheritance": "Autosomal Recessive",
        "prevalence": "~1:100,000–300,000 general; ~1:20,000 Saudi Arabia (p.Arg41Gln founder)",
        "pathway_step": "Leucine catabolism STEP 5 (TERMINAL) — HMG-CoA → Acetoacetate + Acetyl-CoA; also sole hepatic ketogenesis enzyme",
        "n_patients": N_PATIENTS,
        "seed": SEED,
        "kpi": {
            "n_patients":    {"label": "Cohort",               "value": str(N_PATIENTS),                                  "color": "#1a237e"},
            "seizures":      {"label": "Seizures",             "value": f"{n_seiz} ({round(n_seiz/N_PATIENTS*100)}%)",    "color": "#b71c1c"},
            "crises":        {"label": "Metabolic Crises",     "value": f"{n_cris} ({round(n_cris/N_PATIENTS*100)}%)",    "color": "#e65100"},
            "hepatomegaly":  {"label": "Hepatomegaly",         "value": f"{n_hepato} ({round(n_hepato/N_PATIENTS*100)}%)","color": "#4a148c"},
            "low_carnitine": {"label": "Low Carnitine",        "value": f"{n_low_c} ({round(n_low_c/N_PATIENTS*100)}%)",  "color": "#0d47a1"},
            "avg_hmg":       {"label": "Avg 3-HMG (mmol/mol Cr)", "value": str(avg_hmg),                                 "color": "#b71c1c"},
            "avg_glucose":   {"label": "Avg Crisis Glucose (mmol/L)", "value": str(avg_gluc),                            "color": "#e65100"},
            "vpa_pre_dx":    {"label": "VPA given pre-Dx (hazard)", "value": f"{n_vpa} ({round(n_vpa/N_PATIENTS*100)}%)", "color": "#880e4f"},
        },
        "phenotype_dist": pheno_dist,
        "hypoketotic_hallmark": (
            "HMGCL deficiency produces HYPOKETOTIC HYPOGLYCAEMIA — the pathognomonic combination. "
            "Glucose is LOW (crisis: <2.5 mmol/L) AND ketones are ABSENT (ΒETA-hydroxybutyrate ~0, acetoacetate ~0). "
            "This is the opposite of normal fasting physiology: normally when glucose falls, the liver "
            "generates ketone bodies via HMGCL. HMGCL deficiency abolishes this backup fuel entirely. "
            "The brain is deprived of BOTH energy sources simultaneously during a crisis — explaining "
            "the severity of neurological impact. ANY combination of low glucose + absent/low ketones "
            f"on a metabolic screen MUST trigger HMGCL evaluation. All {N_PATIENTS} cohort patients "
            "showed absent ketones during documented crises."
        ),
        "no_ketogenic_diet_note": (
            "Ketogenic Diet (KD) is ABSOLUTELY CONTRAINDICATED in HMGCL deficiency. "
            "The rationale for KD is to generate ketone bodies as brain fuel during seizure management. "
            "HMGCL is the sole enzyme converting HMG-CoA → acetoacetate (the first ketone body). "
            "In HMGCL deficiency: (1) ketone synthesis is impossible regardless of fat intake; "
            "(2) high-fat diet floods the mitochondria with acetyl-CoA → HMGCS2 → HMG-CoA → "
            "ACCUMULATES (blocked) → acute crisis. KD = direct metabolic catastrophe in HMGCL deficiency."
        ),
        "hallmark_biomarker": (
            "3-Hydroxy-3-methylglutaric acid (3-HMG) VERY HIGH >200 mmol/mol Cr — PATHOGNOMONIC (urine OA). "
            "C6-DC (3-methylglutarylcarnitine) ELEVATED — NBS marker (more specific than C5-OH). "
            "C5-OH ELEVATED — NBS Tier-1 trigger. "
            "Ketones ABSENT — HYPOKETOTIC (KEY FINDING). "
            "Blood glucose LOW (<2.5 mmol/L during crisis). "
            "3-MCG ABSENT — KEY NEGATIVE vs 3-MCC (most important C5-OH differential). "
            "C3 NORMAL — KEY NEGATIVE vs HLCS/PA. "
            "Biotinidase NORMAL — KEY NEGATIVE vs BTD. "
            "VPA — ABSOLUTE CI (inhibits HMGCL Cys266 active site directly)."
        ),
        "vpa_warning": (
            "Valproic acid (VPA) is an ABSOLUTE CONTRAINDICATION in HMGCL deficiency. "
            "Mechanism: VPA is metabolised to valproyl-CoA, which directly inhibits the HMGCL active site "
            "(competing at Cys266 — the catalytic nucleophile). This worsens the primary enzymatic block, "
            "acutely elevating HMG-CoA and 3-HMG. Additionally: VPA depletes carnitine (valproyl-carnitine "
            "excretion), is hepatotoxic in mitochondrial IEMs, and inhibits mitochondrial β-oxidation. "
            f"In this cohort, {n_vpa} patients received VPA prior to diagnosis — all deteriorated acutely. "
            "Use levetiracetam (LEV) as the first-line AED for HMGCL-associated seizures."
        ),
    }


def get_breakdown():
    biomarkers = {
        "hmg": {
            "label": "3-HMG (3-Hydroxy-3-methylglutaric acid)",
            "normal": "<10 mmol/mol Cr",
            "status": "VERY HIGH — >200–2500 mmol/mol Cr — PATHOGNOMONIC (urine OA)",
            "direction": "↑↑↑ PATHOGNOMONIC",
            "color": "danger",
            "rationale": (
                "HMG-CoA CANNOT be cleaved by defective HMGCL → thioester hydrolysis → 3-HMG (free acid). "
                "3-HMG is the single most specific and quantitatively dominant biomarker for HMGCL deficiency. "
                "No other IEM produces 3-HMG at this level. Detected by urine OA GC-MS. "
                "During crisis: >500–2500 mmol/mol Cr. Inter-ictally: may fall to 200–500 mmol/mol Cr but remains elevated."
            ),
        },
        "mg": {
            "label": "3-MG (3-Methylglutaric acid)",
            "normal": "<5 mmol/mol Cr",
            "status": "ELEVATED — 50–400 mmol/mol Cr (secondary marker)",
            "direction": "↑↑",
            "color": "warning",
            "rationale": (
                "Secondary metabolite of 3-HMG: spontaneous decarboxylation of 3-hydroxy-3-methylglutarate → "
                "3-methylglutarate. Less pathognomonic than 3-HMG (also elevated in 3-MGC aciduria, "
                "barth syndrome, and rarely other conditions) but quantitatively elevated in HMGCL. "
                "Ratio 3-HMG:3-MG typically 5:1 to 10:1 in HMGCL deficiency."
            ),
        },
        "mgc": {
            "label": "3-MGC (3-Methylglutaconic acid)",
            "normal": "<10 mmol/mol Cr",
            "status": "MILDLY ELEVATED — 15–80 mmol/mol Cr (retrograde AUH)",
            "direction": "↑",
            "color": "warning",
            "rationale": (
                "AUH (enoyl-CoA hydratase, step 4) catalyses a reversible reaction: 3-methylglutaconyl-CoA ⇌ HMG-CoA. "
                "When HMG-CoA accumulates, the equilibrium shifts retrograde → 3-MGC accumulates mildly. "
                "NOT as elevated as in primary 3-MGC acidurias (DNAJC19, TAZ, OPA3, ATPAF2). "
                "3-MGC ↑ in HMGCL supports the diagnosis but is not specific."
            ),
        },
        "c6dc": {
            "label": "C6-DC (3-Methylglutarylcarnitine)",
            "normal": "<0.1 µmol/L",
            "status": "ELEVATED — 0.4–12 µmol/L — NBS PRIMARY MARKER (specific for HMGCL)",
            "direction": "↑↑ NBS PRIMARY",
            "color": "danger",
            "rationale": (
                "3-Methylglutaryl-CoA (derived from HMG-CoA accumulation) conjugates with carnitine → "
                "C6-DC (3-methylglutarylcarnitine, a dicarboxylyl acylcarnitine). "
                "C6-DC is more specific for HMGCL deficiency than C5-OH on NBS expanded panels. "
                "C6-DC ↑ on acylcarnitine profile should always prompt 3-HMG urine OA measurement."
            ),
        },
        "c5oh": {
            "label": "C5-OH (3-Hydroxy-isovalerylcarnitine)",
            "normal": "<0.4 µmol/L",
            "status": "ELEVATED — 0.4–8 µmol/L (NBS Tier-1 trigger, shared with MCC/HLCS/BTD)",
            "direction": "↑ NBS TRIGGER",
            "color": "warning",
            "rationale": (
                "3-Hydroxy-isovaleryl-CoA (from HMG-CoA overflow via side pathways) conjugates carnitine → C5-OH. "
                "C5-OH on NBS triggers expanded differential including HMGCL. "
                "CRITICAL: C5-OH ↑ + 3-MCG ABSENT → strongly supports HMGCL over 3-MCC. "
                "C5-OH ↑ + C6-DC ↑ → HMGCL most likely diagnosis."
            ),
        },
        "ketones": {
            "label": "Ketones (β-OHB / Acetoacetate)",
            "normal": "<0.5 mmol/L plasma; present in fasting",
            "status": "ABSENT — <0.1 mmol/L even during hypoglycaemia — HALLMARK FINDING",
            "direction": "↓↓ ABSENT (HYPOKETOTIC)",
            "color": "danger",
            "rationale": (
                "HMGCL catalyses the final step of BOTH leucine catabolism AND hepatic ketogenesis. "
                "In HMGCL deficiency, the liver CANNOT produce acetoacetate or β-hydroxybutyrate even "
                "when fatty acid β-oxidation is intact and HMG-CoA is present. "
                "Absent ketones during hypoglycaemia = HYPOKETOTIC STATE = HMGCL must be on differential. "
                "Normal fasting response = glucose falls → ketones rise (brain uses ketones). "
                "In HMGCL: glucose falls → NO ketones rise → brain catastrophically deprived."
            ),
        },
        "glucose": {
            "label": "Blood Glucose",
            "normal": ">4.0 mmol/L (>72 mg/dL)",
            "status": "LOW during crisis — <2.5 mmol/L (<45 mg/dL) — HYPOKETOTIC HYPOGLYCAEMIA",
            "direction": "↓↓ CRISIS",
            "color": "danger",
            "rationale": (
                "Fasting/catabolism triggers leucine release + FA oxidation → HMG-CoA accumulates (BLOCKED). "
                "Simultaneously: hepatic glucose output falls (liver stressed by HMG-CoA toxicity + energy failure). "
                "IV glucose is the first-line emergency treatment: GIR 10–15 mg/kg/min suppresses catabolism, "
                "provides energy substrate, reduces leucine mobilisation."
            ),
        },
        "free_carnitine": {
            "label": "Free Carnitine (C0)",
            "normal": "25–55 µmol/L",
            "status": "LOW — secondary depletion via C6-DC / C5-OH conjugation",
            "direction": "↓",
            "color": "warning",
            "rationale": (
                "Acyl-CoA intermediates (3-methylglutaryl-CoA → C6-DC; 3-OH-isovaleryl-CoA → C5-OH) "
                "conjugate free carnitine for renal excretion, depleting the free carnitine pool. "
                "Secondary carnitine deficiency in HMGCL complicates energy metabolism and "
                "cardiac function. L-carnitine supplementation replenishes pool."
            ),
        },
        "mcg": {
            "label": "3-MCG (3-Methylcrotonylglycine)",
            "normal": "<5 mmol/mol Cr",
            "status": "ABSENT — KEY NEGATIVE vs 3-MCC (most critical NBS C5-OH differentiator)",
            "direction": "→ ABSENT",
            "color": "success",
            "rationale": (
                "3-MCG is the glycine conjugate of 3-methylcrotonyl-CoA — the substrate that accumulates "
                "in 3-MCC (MCCC1/MCCC2) deficiency. In HMGCL deficiency, the block is at HMG-CoA (step 5), "
                "DOWNSTREAM of the 3-methylcrotonyl-CoA step. 3-methylcrotonyl-CoA does NOT accumulate. "
                "3-MCG ABSENT in HMGCL = the single most important differentiator from 3-MCC on NBS. "
                "C5-OH ↑ + C6-DC ↑ + 3-MCG ABSENT = HMGCL until proven otherwise."
            ),
        },
        "nh3": {
            "label": "Ammonia (NH3)",
            "normal": "<50 µmol/L",
            "status": "NORMAL inter-ictally; may rise SECONDARY (55–300) during severe crisis",
            "direction": "→ / ↑ (secondary)",
            "color": "warning",
            "rationale": (
                "NH3 is NOT elevated as a PRIMARY feature of HMGCL deficiency (no urea cycle enzyme block). "
                "During severe metabolic decompensation, secondary hyperammonaemia can develop due to "
                "mitochondrial energy failure impairing N-acetylglutamate synthesis (NAGS activation). "
                "This can mimic UCD acutely — differentiated by 3-HMG on urine OA + absence of 3-MCG/citrulline."
            ),
        },
        "c3": {
            "label": "C3 (Propionylcarnitine)",
            "normal": "<5 µmol/L",
            "status": "NORMAL — KEY NEGATIVE vs HLCS (multiple carboxylase deficiency)",
            "direction": "→ NORMAL",
            "color": "success",
            "rationale": (
                "In HLCS (holocarboxylase synthetase) deficiency, ALL biotin-dependent carboxylases fail: "
                "PCC (C3 ↑) + MCC (C5-OH ↑) + PC (lactate ↑) + ACC (dermatological). "
                "HMGCL is NOT biotin-dependent. C3 NORMAL in HMGCL = key discriminator from HLCS. "
                "C5-OH ↑ + C3 NORMAL = NOT HLCS."
            ),
        },
    }

    enzyme_mechanism = {
        "function": (
            "HMGCL (3-Hydroxy-3-methylglutaryl-CoA Lyase) is a mitochondrial homotrimeric enzyme (325 aa mature form; "
            "27-aa MTS cleaved on import). Catalytic nucleophile: Cys266 (attacks carbonyl of HMG-CoA forming "
            "enzyme-bound thioester intermediate). Product: acetoacetate (first ketone body) + acetyl-CoA. "
            "DUAL ROLE: (1) Terminal step of leucine catabolism; (2) SOLE enzyme of hepatic ketogenesis."
        ),
        "reaction": (
            "HMG-CoA + H₂O  →  [HMGCL / Cys266 thioester intermediate]  →  "
            "Acetoacetate + Acetyl-CoA    (Step 5 of leucine catabolism; also hepatic ketogenesis)"
        ),
        "block": (
            "HMGCL LOF → HMG-CoA CANNOT be cleaved → DUAL FAILURE:\n"
            "(1) Leucine catabolism blocked at step 5 → HMG-CoA → 3-HMG (hydrolysis) ↑↑ + 3-MG ↑ + C6-DC ↑\n"
            "(2) Hepatic ketogenesis ABOLISHED → Acetoacetate ABSENT → β-OHB ABSENT → HYPOKETOTIC STATE\n"
            "Result: fasting/catabolism → glucose ↓↓ + NO ketone backup → acute neurological crisis"
        ),
        "leucine_path": (
            "L-Leucine → BCAT → KIC → BCKDH → Isovaleryl-CoA → [IVD, step 2] → "
            "3-Methylcrotonyl-CoA → [MCC, step 3] → 3-Methylglutaconyl-CoA → [AUH, step 4] → "
            "HMG-CoA → [HMGCL BLOCKED ⚠ — STEP 5 TERMINAL] → ↛ Acetoacetate + Acetyl-CoA"
        ),
        "no_keto_rationale": (
            "Ketogenic diet CONTRAINDICATED: KD floods mitochondria with FA-derived acetyl-CoA → HMGCS2 → "
            "HMG-CoA → ACCUMULATES (HMGCL absent) → acute 3-HMG surge + metabolic crisis. "
            "HMGCL is required for BOTH leucine catabolism AND ketone body production. "
            "KD is futile (cannot generate ketones) AND harmful (worsens substrate accumulation)."
        ),
    }

    seizure_types = [
        {"type": "Generalised tonic-clonic (hypoglycaemic crisis)", "pct": 65, "note": "Primary mechanism: hypoketotic hypoglycaemia — brain deprived of glucose AND ketones; resolves with IV glucose"},
        {"type": "Focal cortical seizures (chronic metabolic injury)", "pct": 20, "note": "Repeated crises → cortical neuronal injury → epileptiform focus; LEV first-line"},
        {"type": "Neonatal seizures (metabolic encephalopathy)", "pct": 10, "note": "Neonatal/early-onset phenotype; jittery + tonic seizures day 1–5; IV glucose + metabolic rescue"},
        {"type": "Drug-resistant epilepsy (DRE)", "pct": 8, "note": "Rare; usually follows delayed diagnosis with repeated brain injury; risk reduced by early NBS + treatment"},
    ]

    treatments = [
        {
            "therapy": "IV Glucose (emergency anti-catabolic)",
            "level": "A",
            "dose": "10–15 mg/kg/min GIR; bolus 200 mg/kg if symptomatic; nil-by-mouth if vomiting",
            "rationale": "Immediately corrects hypoglycaemia + suppresses FA mobilisation and leucine catabolism. First-line in ALL acute crises. Maintain euglycaemia (4–8 mmol/L).",
        },
        {
            "therapy": "Leucine-restricted diet (low-leucine, HMGCL-free formula)",
            "level": "A",
            "dose": "Natural protein 0.8–1.5 g/kg/day; leucine <100 mg/kg/day; HMGCL amino-acid formula",
            "rationale": "Reduces HMG-CoA production from leucine catabolism. Primary long-term strategy. Titrate by plasma leucine (target 80–180 µmol/L) + urine 3-HMG levels.",
        },
        {
            "therapy": "Fasting avoidance (strict protocol)",
            "level": "A",
            "dose": "Max fasting: 4 h (neonate), 6 h (infant), 8–10 h (child); glucose polymer drinks during illness",
            "rationale": "Fasting → proteolysis → leucine surge + FA mobilisation → HMG-CoA accumulation crisis. Written emergency sick-day protocol MANDATORY for all families.",
        },
        {
            "therapy": "L-Carnitine supplementation",
            "level": "A",
            "dose": "100–200 mg/kg/day oral; 50–100 mg/kg IV loading during crisis",
            "rationale": "Secondary carnitine depletion (C6-DC + C5-OH conjugation). Replenishes free carnitine pool → protects cardiac + skeletal muscle, enhances acylcarnitine excretion.",
        },
        {
            "therapy": "Low-fat diet component",
            "level": "B",
            "dose": "Limit total fat <25–30% calories; avoid high-fat fasting interventions",
            "rationale": "Hepatic FA β-oxidation generates acetyl-CoA → HMGCS2 → HMG-CoA → ACCUMULATES in HMGCL deficiency. Moderate fat restriction reduces this second substrate source for HMG-CoA. Contrast MCC deficiency where fat intake less critical.",
        },
        {
            "therapy": "Levetiracetam (LEV)",
            "level": "B",
            "dose": "20–40 mg/kg/day in 2–3 divided doses",
            "rationale": "First-line AED for HMGCL-associated seizures. No carnitine depletion. No HMGCL inhibition. Safer profile than VPA in mitochondrial/metabolic IEMs.",
        },
        {
            "therapy": "Valproic Acid (VPA)",
            "level": "ABSOLUTE CI",
            "dose": "DO NOT USE — absolute contraindication",
            "rationale": "VPA metabolised to valproyl-CoA → directly inhibits HMGCL at Cys266 active site, worsening HMG-CoA accumulation. Also: carnitine depletion (valproyl-carnitine excretion), mitochondrial β-oxidation inhibition, hepatotoxic in IEMs with liver involvement. Fatal metabolic crisis reported.",
        },
        {
            "therapy": "Ketogenic Diet (KD)",
            "level": "ABSOLUTE CI",
            "dose": "DO NOT USE — absolutely contraindicated",
            "rationale": "KD requires intact HMGCL to produce ketone bodies (the intended fuel source). In HMGCL deficiency: (1) ketone production impossible, so KD therapeutic goal CANNOT be achieved; (2) high fat → FA β-oxidation → acetyl-CoA → HMGCS2 → HMG-CoA → ACCUMULATES → acute metabolic crisis. KD = double harm.",
        },
        {
            "therapy": "Biotin supplementation",
            "level": "NOT EFFECTIVE",
            "dose": "N/A — do not use for HMGCL",
            "rationale": "HMGCL is NOT biotin-dependent. Biotin has no role in HMG-CoA lyase function. Biotin useful only for HLCS/BTD (biotin-dependent multiple carboxylase deficiencies with C5-OH ↑). Biotin trial may delay correct diagnosis.",
        },
    ]

    systemic_features = [
        {"feature": "Hepatomegaly (acute crisis)",           "pct": 70, "note": "HMG-CoA toxic to hepatocytes; transient in classic phenotype; normalises with treatment"},
        {"feature": "Hypoketotic hypoglycaemia (pathognomonic)","pct": 95, "note": "Hallmark: glucose LOW + ketones ABSENT during fasting/illness; occurs in ALL crisis episodes"},
        {"feature": "Metabolic acidosis",                    "pct": 80, "note": "3-HMG + organic acid load + lactate accumulation; bicarbonate <15 mEq/L during crisis"},
        {"feature": "Seizures",                              "pct": 45, "note": "Primarily secondary to hypoglycaemia; 65% generalised tonic-clonic during crises"},
        {"feature": "Encephalopathy / altered consciousness","pct": 60, "note": "Brain deprived of glucose AND ketones; coma in severe neonatal cases; reversible with prompt glucose"},
        {"feature": "Vomiting / feeding refusal",            "pct": 65, "note": "Consistent prodrome before crisis; key warning sign for sick-day glucose protocol"},
        {"feature": "Hypotonia",                             "pct": 35, "note": "Generalised; severity correlates with crisis frequency; improves with metabolic control"},
        {"feature": "Developmental delay (post-crisis)",     "pct": 25, "note": "Secondary to hypoglycaemic brain injury; reduced by early diagnosis via NBS + emergency protocol"},
        {"feature": "Hepatic dysfunction (Reye-like)",       "pct": 40, "note": "ALT/AST elevated; PT prolonged; NOT chronic liver disease; resolves with crisis resolution"},
        {"feature": "Cardiomyopathy (rare)",                 "pct": 5,  "note": "Reported in late-diagnosed severe cases; secondary energy failure + carnitine deficiency"},
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
        "HMGCL (3-Hydroxy-3-methylglutaryl-CoA Lyase)": (
            "Mitochondrial homotrimeric enzyme (325 aa mature, after 27-aa MTS cleavage; gene: HMGCL, 1p36.11, AR). "
            "Catalytic nucleophile: Cys266 (forms thioester intermediate with HMG-CoA). "
            "Reaction: HMG-CoA + H₂O → Acetoacetate + Acetyl-CoA. "
            "DUAL ROLE: (1) Terminal enzyme of leucine catabolism (step 5 of 5); "
            "(2) SOLE enzyme of hepatic ketogenesis — no alternative pathway for ketone body production. "
            "OMIM Gene *600234, Disease #246450."
        ),
        "HMGCL Deficiency": (
            "Autosomal recessive IEM (biallelic LOF in HMGCL). "
            "OMIM: #246450 (3-Hydroxy-3-methylglutaryl-CoA Lyase Deficiency). "
            "Prevalence: ~1:100,000–300,000 general; ~1:20,000 Saudi Arabia (p.Arg41Gln founder mutation). "
            "Leucine catabolism STEP 5 block → HMG-CoA accumulation → 3-HMG ↑↑ (PATHOGNOMONIC) + "
            "ketogenesis ABOLISHED → HYPOKETOTIC HYPOGLYCAEMIA (hallmark). "
            "3 phenotypes: neonatal severe (35%), classic infantile episodic (50%), late-onset attenuated (15%)."
        ),
        "3-HMG (3-Hydroxy-3-methylglutaric acid)": (
            "PATHOGNOMONIC urinary biomarker for HMGCL deficiency. "
            "Formed by hydrolysis of accumulated HMG-CoA thioester → free 3-HMG. "
            "Reference: <10 mmol/mol Cr. In HMGCL deficiency: >200–2500 mmol/mol Cr (crisis). "
            "Detected by urine organic acid GC-MS. No other IEM produces 3-HMG at this level. "
            "3-HMG on urine OA = HMGCL deficiency until proven otherwise."
        ),
        "Hypoketotic Hypoglycaemia (PATHOGNOMONIC COMBINATION)": (
            "Blood glucose LOW (<2.5 mmol/L / <45 mg/dL) PLUS ketones ABSENT (β-OHB <0.1 mmol/L). "
            "Normally during hypoglycaemia, the liver produces ketone bodies (acetoacetate, β-OHB) as "
            "backup brain fuel via HMGCL. In HMGCL deficiency, this is IMPOSSIBLE. "
            "The brain is deprived of BOTH glucose AND ketones → rapid neurological decompensation. "
            "Other causes of hypoketotic hypoglycaemia: hyperinsulinism (insulin suppresses FA oxidation), "
            "FAOD (cannot provide acetyl-CoA), HMGCS2 deficiency. "
            "3-HMG on urine OA differentiates HMGCL from these alternatives."
        ),
        "Ketogenesis — Why KD Is Absolutely Contraindicated": (
            "Hepatic ketogenesis pathway:\n"
            "Fatty acid β-oxidation → Acetyl-CoA → Acetoacetyl-CoA → [HMGCS2] → HMG-CoA → [HMGCL] → Acetoacetate + Acetyl-CoA\n"
            "Acetoacetate → β-hydroxybutyrate (by BHDB). Ketones exported to brain, heart, kidney as fuel.\n"
            "In HMGCL deficiency: HMGCL absent → HMG-CoA ACCUMULATES → ZERO ketone output.\n"
            "Ketogenic diet floods the system with FA-derived HMG-CoA → acute metabolic crisis.\n"
            "KD goal (brain ketone fuel) CANNOT be achieved + KD causes direct harm. Absolute contraindication."
        ),
        "C6-DC (3-Methylglutarylcarnitine)": (
            "NBS acylcarnitine marker for HMGCL deficiency. "
            "3-Methylglutaryl-CoA (from HMG-CoA accumulation) conjugates carnitine → C6-DC (dicarboxylyl). "
            "More specific for HMGCL than C5-OH (which is shared with 3-MCC, HLCS, BTD). "
            "C6-DC ↑ + C5-OH ↑ on NBS → high probability HMGCL deficiency → confirm by urine 3-HMG. "
            "Some NBS programmes report C6-DC only on second-tier reflex testing."
        ),
        "VPA — Absolute CI Mechanism": (
            "Valproic acid (VPA / valproate) is an ABSOLUTE CONTRAINDICATION in HMGCL deficiency. "
            "VPA is metabolised by mitochondrial β-oxidation to valproyl-CoA (a short-chain branched fatty acid CoA). "
            "Valproyl-CoA DIRECTLY INHIBITS HMGCL at the Cys266 catalytic site (competitive/mechanism-based inhibition). "
            "This worsens the primary enzymatic block, causing acute 3-HMG surge and HMG-CoA accumulation. "
            "Concurrent effects: carnitine depletion (valproyl-carnitine excretion), hepatotoxic in IEMs "
            "with mitochondrial dysfunction, inhibits β-oxidation. "
            "For seizures: use levetiracetam (LEV) — no HMGCL inhibition, safe carnitine profile."
        ),
        "NBS C5-OH + C6-DC Differential": (
            "C5-OH ↑ on NBS triggers expanded differential. C6-DC further narrows:\n"
            "1. HMGCL deficiency — C5-OH ↑ + C6-DC ↑ + NO 3-MCG + hypoketotic hypoglycaemia → urine 3-HMG\n"
            "2. 3-MCC (MCCC1/2) — C5-OH ↑ + 3-MCG ↑↑ + 3-HIV ↑; C6-DC NORMAL → most common C5-OH cause\n"
            "3. HLCS — C5-OH ↑ + C3 ↑ (all carboxylases); BIOTIN-RESPONSIVE; rash + ketoacidosis\n"
            "4. BTD — C5-OH ↑ + rash + SNHL + optic atrophy; BIOTIN-RESPONSIVE; biotinidase assay LOW\n"
            "Key rule: C6-DC ↑ = think HMGCL first. 3-MCG absent = NOT 3-MCC."
        ),
        "Leucine Catabolism — Step 5 Context (Terminal)": (
            "L-Leucine (essential BCAA) catabolism, 5 steps to final products:\n"
            "Step 1a: L-Leucine → KIC (α-ketoisocaproate) via BCAT (reversible transamination)\n"
            "Step 1b: KIC → Isovaleryl-CoA + CO₂ via BCKDH complex (TPP/FAD/NAD-dependent; irreversible)\n"
            "Step 2:  Isovaleryl-CoA → 3-Methylcrotonyl-CoA via IVD (FAD; ACAD family) — IVD def upstream\n"
            "Step 3:  3-MC-CoA → 3-Methylglutaconyl-CoA via MCC (biotin; MCCC1+MCCC2) — MCC def upstream\n"
            "Step 4:  3-Methylglutaconyl-CoA → HMG-CoA via AUH (enoyl-CoA hydratase; reversible)\n"
            "Step 5:  HMG-CoA → Acetoacetate + Acetyl-CoA via HMGCL [BLOCKED HERE — TERMINAL STEP]\n"
            "Net: leucine → 2× acetyl-CoA + 1× acetoacetate (all ketogenic; no glucogenic product)"
        ),
        "Inheritance & Epidemiology": (
            "Autosomal recessive (AR); biallelic LOF in HMGCL (1p36.11). "
            "Prevalence: ~1:100,000–300,000 general population. "
            "High prevalence in Saudi Arabia (~1:20,000) due to p.Arg41Gln founder (>25% alleles). "
            "Spanish/Portuguese founder: p.Phe305Ile (attenuated phenotype). "
            "Most cases now detected by NBS (C5-OH ± C6-DC) before first crisis — dramatically improving outcomes. "
            "Pre-NBS mortality in neonatal phenotype was >50% (missed at Reye-like presentation)."
        ),
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:3000])
    print("\n=== BREAKDOWN keys ===", list(get_breakdown().keys()))
    print("\n=== DEFINITIONS keys ===", list(get_definitions().keys()))
    print(f"\n✓ HMGCL dashboard: {N_PATIENTS} patients, seed={SEED}")
