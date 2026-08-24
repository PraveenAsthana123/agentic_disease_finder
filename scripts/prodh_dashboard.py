#!/usr/bin/env python3
"""PRODH (Hyperprolinemia Type I / Proline Dehydrogenase Deficiency) Epilepsy Dashboard.

PRODH (also PRODH1) encodes Proline Oxidase / Proline Dehydrogenase, the first
mitochondrial enzyme of the proline catabolic pathway.

PROLINE CATABOLIC PATHWAY:
  Proline
    → [PRODH / Proline Oxidase, FAD-dependent, inner mitochondrial membrane]
  Delta-1-pyrroline-5-carboxylate (P5C)
    → [ALDH4A1 / P5CDH, NAD-dependent, mitochondrial matrix]
  L-Glutamate
    → (TCA via alpha-KG; gluconeogenesis; GABA synthesis)

PRODH ENZYMATIC FUNCTION:
  PRODH catalyses: L-Proline + FAD → Delta-1-pyrroline-5-carboxylate (P5C) + FADH2
  (oxidation at C5 of the pyrrolidine ring; inner mitochondrial membrane, electron donor
  to ubiquinone/CoQ10 in the mitochondrial electron transport chain)

PRODH LOF — PROLINE ACCUMULATES; P5C DOES NOT:
  Since PRODH cannot oxidise proline to P5C, proline accumulates in plasma/urine.
  P5C does NOT accumulate — ALDH4A1 (step 2) is INTACT and converts any residual P5C
  to glutamate normally.
  This is the MOST FUNDAMENTAL DISTINCTION between Type I and Type II Hyperprolinemia.

THE EPILEPTOGENIC MECHANISMS IN PRODH DEFICIENCY (distinct from Type II):
  PRODH LOF causes epilepsy by DIFFERENT and WEAKER mechanisms than ALDH4A1:

  Mechanism 1 — Proline as partial NMDA receptor agonist:
    Proline (especially at >600–1000 µmol/L) acts as a partial agonist at the glycine
    co-agonist site and as a weak direct NMDA agonist, increasing NMDA receptor activity
    → neuroexcitatory → lowers seizure threshold.

  Mechanism 2 — Proline inhibits GABA transporter re-uptake:
    Proline is a competitive inhibitor of GABA transporters SLC6A1 (GAT-1) and SLC6A11 (GAT-3).
    High synaptic proline competes with GABA re-uptake → local GABA accumulation acutely
    (paradoxically), but also disrupts synchrony of inhibitory post-synaptic potentials.
    Net effect: altered GABA signalling architecture → seizure susceptibility.

  Mechanism 3 — Mitochondrial dysfunction at high loads:
    At very high proline concentrations, the PRODH enzyme reversal and associated
    FAD/FADH2 redox buffering perturbation can impair mitochondrial membrane potential
    in a concentration-dependent manner.

  WHY MILDER THAN TYPE II:
    There is NO P5C accumulation → NO P5C-PLP adduct formation → PLP pool is INTACT.
    GAD65/GAD67 are NOT impaired → GABA synthesis is NORMAL.
    No secondary B6 deficiency → NO B6/pyridoxine response expected or observed.
    Epilepsy prevalence: 25–35% (Type I) vs 60–80% (Type II).
    Drug-resistant epilepsy: <15% (Type I) vs 25–40% (Type II).

DISTINGUISHING HYPERPROLINEMIA TYPE I vs TYPE II:
  Feature                   Type I (PRODH)              Type II (ALDH4A1)
  Proline (plasma)          ELEVATED (350–1000 µmol/L)  MARKEDLY ELEVATED (>1000–2200+)
  P5C (plasma/urine)        NORMAL (ALDH4A1 intact)     ELEVATED — PATHOGNOMONIC
  PLP (plasma)              NORMAL — NO inactivation     LOW — secondary B6 deficiency
  B6/pyridoxine response    NONE — PLP not depleted      PARTIAL (30–50%) — PLP restoration
  Seizure prevalence        25–35% (mild mechanisms)     60–80% (P5C-PLP GABA collapse)
  Drug-resistant epilepsy   <15%                         25–40%
  IDD severity              Mild-moderate (30–40%)       Moderate (50–70%)
  Psychiatric features      YES: schizophrenia-like 15–25%; bipolar 10–15%  Less prominent
  alpha-AASA                NORMAL                       NORMAL (both vs ALDH7A1)
  Pipecolic acid            NORMAL                       NORMAL (both vs ALDH7A1)

KEY BIOMARKERS IN PRODH DEFICIENCY:
  ELEVATED (Type I constellation):
    Proline (plasma): ELEVATED (350–1000 µmol/L; normal <260 µmol/L) — milder than Type II
    Urine proline: ELEVATED (prolinuria — overflow transport)
    Urine hydroxyproline: mildly elevated (overflow from high proline pool → hydroxylation flux)

  NORMAL (critical key negatives — the diagnostic workhorses):
    P5C (plasma/urine): NORMAL — single most important differential from Type II (ALDH4A1)
    PLP (plasma):       NORMAL — no P5C-PLP inactivation
    alpha-AASA (urine): NORMAL — KEY NEGATIVE vs ALDH7A1/PDE (antiquitin deficiency)
    Pipecolic acid:     NORMAL — KEY NEGATIVE vs ALDH7A1/PDE
    MMA:                NORMAL — propionate/B12 pathway intact
    Homocysteine (tHcy): NORMAL — CBS and folate cycle intact
    Methionine:         NORMAL — SAM cycle unaffected
    Lactate/pyruvate:   NORMAL — no primary mitochondrial RC involvement at typical proline levels
    Ammonia:            NORMAL — urea cycle intact
    CSF glycine:        NORMAL — glycine cleavage system intact (vs NKH)

PRODH AND SCHIZOPHRENIA / PSYCHIATRIC PHENOTYPE:
  PRODH resides at 22q13.2 and has a functional relationship with the 22q11.2 deletion
  syndrome (DiGeorge/velocardiofacial syndrome) region. The 22q11.2 deletion commonly
  includes DGCR6/PRODH-related regulatory elements, and PRODH haploinsufficiency has been
  studied as a susceptibility factor for schizophrenia.
  In biallelic PRODH LOF (hyperprolinemia Type I):
    - Schizophrenia-like features: 15–25% of symptomatic patients
    - Bipolar-like features: 10–15%
    - ADHD-like: 20–30%
  This psychiatric overlap is UNIQUE to Type I and is less prominent in Type II (ALDH4A1).
  The mechanistic link is proposed to involve NMDA receptor hypofunction (proline partial
  agonist → desensitisation; dysregulation of dopaminergic/glutamatergic balance).

PRODH vs ALDH7A1 (PDE — Pyridoxine-Dependent Epilepsy / Antiquitin Deficiency):
  Both PRODH and ALDH7A1 are in the proline/lysine catabolic metabolic neighbourhood,
  but they are entirely different diseases:
    PRODH: proline elevated; P5C NORMAL; PLP NORMAL; alpha-AASA NORMAL; pipecolic NORMAL
    ALDH7A1: alpha-AASA MARKEDLY elevated (PATHOGNOMONIC); pipecolic ELEVATED; proline NORMAL
    ALDH7A1 responds 85%+ to B6; PRODH has NO B6 response.

NBS STATUS:
  Standard NBS: PARTIALLY detected — proline mildly elevated (350–1000 µmol/L) is above
  normal (<260) but less striking than Type II (>1000). Some expanded NBS programs flag
  proline >350 µmol/L as borderline; many Type I cases are missed.
  P5C: NOT routinely measured; NORMAL in Type I (not useful for Type I NBS).
  Diagnosis often triggered by incidental plasma amino acid testing, NBS borderline flag,
  or psychiatric/neurodevelopmental evaluation in symptomatic patients.

TREATMENTS:
  Proline-restricted diet — Level B:
    Reduce proline intake: low-proline proteins (avoid collagen, gelatin, casein-rich foods).
    Reduces plasma proline → less NMDA agonism, less GABA transport competition.
    Moderate benefit; diet adherence challenging.
  Levetiracetam (LEV) — Level B:
    First-line AED; no specific interaction with proline pathway; well-tolerated.
  Standard AEDs — Level C:
    For seizures not controlled by LEV; VPA has a specific cautionary note (see below).
  Psychiatric medication — Level C:
    Antipsychotics (2nd generation) for schizophrenia-like features; mood stabilisers for
    bipolar-like; stimulants/atomoxetine for ADHD-like; clinical psychiatry consultation needed.
  Betaine (trimethylglycine) — Level C, experimental:
    May promote alternative methylation pathways affecting proline metabolism;
    limited evidence; under study.
  NO B6/pyridoxine:
    PLP is NORMAL; there is no P5C-PLP inactivation mechanism; B6 supplementation
    has NO indication and will NOT benefit seizure control in Type I.

HIGH-RISK DRUGS:
  VPA (valproate) — MODERATE RISK:
    VPA inhibits PRODH in vitro (raises proline → more accumulation in an already-elevated pool).
    VPA also causes carnitine depletion (metabolic complication).
    B6/PLP depletion by VPA is less concerning here (PLP normal baseline) but still noteworthy.
    Risk is MODERATE (not HIGH-RISK triple mechanism of Type II) — use with monitoring.
  B6 antagonists (INH, cycloserine) — MODERATE RISK:
    Less hazardous than in Type II (PLP is not already depleted), but prudent to avoid
    additional PLP interference; risk is reduced compared to Type II.
  Phenylalanine-heavy protein loads — MODERATE RISK:
    Competes with proline for shared transport (LAT1/SLC7A5 and intestinal IMINO transporter);
    may paradoxically reduce proline clearance; avoid high-phenylalanine formula excess.

INHERITANCE AND EPIDEMIOLOGY:
  Gene: PRODH (also PRODH1), 19q13.2, 601 amino acids
  Protein: inner mitochondrial membrane; FAD-dependent flavoenzyme; monotopic membrane protein
  Inheritance: Autosomal Recessive — biallelic LOF required for full hyperprolinemia.
    Heterozygotes: some AR heterozygotes may show mildly elevated proline (350–500 µmol/L)
    without full clinical syndrome; psychiatric susceptibility reported in some studies.
  Prevalence: ~200–400 cases worldwide (2026); may be underdiagnosed (many asymptomatic).
  Asymptomatic cases: ~40% — detected by NBS or family cascade screening.

OMIM:
  Gene:    *606810 (PRODH)
  Disease: #239500 (Hyperprolinemia, Type I)
"""

import random

# Deterministic cohort — seed 147
_rng = random.Random(147)

# ---------------------------------------------------------------------------
# Phenotypic classes
# ---------------------------------------------------------------------------
PHENOTYPE_CLASSES = [
    "Asymptomatic-Incidental (NBS or family screening; no seizures; proline 350–700 µmol/L; cognitively normal)",
    "Mild-Symptomatic (mild seizures or behavioral/mild-IDD; proline 600–900 µmol/L; psychiatric features possible)",
    "Classic-Symptomatic (seizures + IDD + psychiatric features; proline 800–1000 µmol/L)",
]

PHENO_WEIGHTS = [0.40, 0.45, 0.15]

VARIANTS = [
    "p.Pro406Leu",    # mitochondrial targeting region; 22q11-like; most common ~25%; mild
    "p.Arg185Trp",    # FAD binding domain; ~20%; moderate-severe
    "p.Leu441Pro",    # active site; ~15%; moderate
    "p.Ala239Val",    # substrate binding; ~12%; mild-moderate
    "p.Arg564Cys",    # membrane anchoring; ~10%; moderate
    "c.IVS4+1G>A",   # splice-null; ~10%; severe-for-Type-I
    "p.Gly372Ser",    # dimer interface; ~8%; mild
]

VARIANT_WEIGHTS = [0.25, 0.20, 0.15, 0.12, 0.10, 0.10, 0.08]

SEIZURE_TYPES = [
    "Febrile seizure (most common in Type I; triggered by proline-NMDA mechanism under fever; 40%)",
    "Focal cortical (proline NMDA agonism + GABA transport competition; 25%)",
    "Absence (mild-symptomatic phenotype; proline-mediated threshold lowering; 20%)",
    "GTCS (classic-symptomatic phenotype; 10%)",
    "Myoclonic (least common in Type I; mild mechanism; 5%)",
]


def _choose_pheno():
    r = _rng.random()
    cum = 0.0
    for ph, w in zip(PHENOTYPE_CLASSES, PHENO_WEIGHTS):
        cum += w
        if r < cum:
            return ph
    return PHENOTYPE_CLASSES[-1]


def _choose_variant(pheno):
    is_severe      = "Classic-Symptomatic" in pheno
    is_asymptomatic = "Asymptomatic" in pheno
    r = _rng.random()
    cum = 0.0
    for v, w in zip(VARIANTS, VARIANT_WEIGHTS):
        cum += w
        if r < cum:
            # Severe phenotype unlikely to carry mild variants
            if is_severe and v in ("p.Pro406Leu", "p.Gly372Ser", "p.Ala239Val"):
                return _rng.choice(["p.Arg185Trp", "p.Leu441Pro", "c.IVS4+1G>A"])
            # Asymptomatic mostly mild variants
            if is_asymptomatic and v in ("c.IVS4+1G>A", "p.Arg185Trp"):
                return _rng.choice(["p.Pro406Leu", "p.Gly372Ser", "p.Ala239Val"])
            return v
    return VARIANTS[0]


def _make_patient(idx):
    pheno           = _choose_pheno()
    is_classic      = "Classic-Symptomatic" in pheno
    is_mild         = "Mild-Symptomatic"     in pheno
    is_asymptomatic = "Asymptomatic"         in pheno

    # -------------------------------------------------------------------
    # Biomarkers — PRODH Type I profile
    # -------------------------------------------------------------------

    # Proline — ELEVATED but less than Type II (350–1000 µmol/L)
    if is_classic:
        proline_umol_l = round(_rng.uniform(800,  1000), 1)
    elif is_mild:
        proline_umol_l = round(_rng.uniform(550,  900),  1)
    else:  # asymptomatic
        proline_umol_l = round(_rng.uniform(350,  700),  1)

    # P5C — NORMAL in ALL patients (ALDH4A1 intact — key distinction from Type II)
    # P5C normal range <5 µmol/L; Type I: within normal (<5)
    p5c_umol_l = round(_rng.uniform(0.5, 4.8), 2)
    p5c_normal = True  # always True in Type I

    # PLP — NORMAL (no P5C-PLP inactivation; PLP pool intact)
    # Normal PLP plasma: 35–110 nmol/L
    plp_nmol_l = round(_rng.uniform(38, 108), 1)
    plp_normal = True  # always True in Type I

    # Key negatives — all NORMAL
    mma_normal        = True
    thcy_umol_l       = round(_rng.uniform(5, 14),   1)  # normal (<15)
    methionine_umol_l = round(_rng.uniform(20, 45),  1)  # NORMAL
    alpha_aasa_normal = True   # KEY NEGATIVE vs ALDH7A1/PDE
    pipecolic_normal  = True   # KEY NEGATIVE vs ALDH7A1/PDE
    lactate_normal    = True
    ammonia_normal    = True

    # Urine proline: elevated (mirrors plasma, overflow prolinuria)
    urine_proline_elevated = True

    # -------------------------------------------------------------------
    # Clinical features
    # -------------------------------------------------------------------
    # Seizure rates: Classic 55%, Mild 35%, Asymptomatic 5%
    seizures = _rng.random() < (0.55 if is_classic else (0.35 if is_mild else 0.05))

    # B6 trial: NOT routinely done (no indication) — some clinicians still trial it
    # We track whether it was trialled and whether it responded (should be near zero)
    b6_trial     = _rng.random() < (0.25 if is_classic else (0.15 if is_mild else 0.05))
    b6_responded = False  # No B6 response in Type I — PLP intact; no mechanism

    # IDD: Classic 50%, Mild 30%, Asymptomatic ~5%
    idd = _rng.random() < (0.50 if is_classic else (0.30 if is_mild else 0.05))

    # Drug-resistant epilepsy: <15% overall; Classic patients with seizures ~25%
    dre = (
        _rng.random() < (0.25 if is_classic else (0.10 if is_mild else 0.0))
        if seizures else False
    )

    # Psychiatric features (schizophrenia-like, bipolar-like) — UNIQUE to Type I
    # Classic 35%, Mild 20%, Asymptomatic 5%
    psychiatric = _rng.random() < (0.35 if is_classic else (0.20 if is_mild else 0.05))

    # ADHD-like behavioral
    behavioral = _rng.random() < (0.40 if is_classic else (0.25 if is_mild else 0.08))

    # NBS detection: harder than Type II (proline less markedly elevated)
    # Classic 45%, Mild 30%, Asymptomatic 35% (often incidentally flagged)
    nbs_detected = _rng.random() < (0.45 if is_classic else (0.30 if is_mild else 0.35))

    # Protein-restricted diet
    protein_restricted = _rng.random() < (0.55 if is_classic else (0.35 if is_mild else 0.15))

    # Seizure type
    seizure_type = _rng.choice(SEIZURE_TYPES) if seizures else None

    # Age at onset: Type I is older onset than Type II
    age_onset_mo = (
        _rng.randint(6,  36) if is_classic      else
        _rng.randint(12, 60) if is_mild          else
        None  # asymptomatic — no seizure onset
    )

    variant = _choose_variant(pheno)

    return {
        "id":                      f"PRODH-{idx:03d}",
        "phenotype":               pheno,
        "variant":                 variant,
        "age_onset_months":        age_onset_mo,
        # KEY ELEVATED BIOMARKER
        "proline_umol_l":          proline_umol_l,
        # KEY NORMALS — type-defining
        "p5c_umol_l":              p5c_umol_l,
        "p5c_normal":              p5c_normal,
        "plp_nmol_l":              plp_nmol_l,
        "plp_normal":              plp_normal,
        # OTHER KEY NEGATIVES
        "mma_normal":              mma_normal,
        "thcy_umol_l":             thcy_umol_l,
        "methionine_umol_l":       methionine_umol_l,
        "alpha_aasa_normal":       alpha_aasa_normal,
        "pipecolic_normal":        pipecolic_normal,
        "lactate_normal":          lactate_normal,
        "ammonia_normal":          ammonia_normal,
        "urine_proline_elevated":  urine_proline_elevated,
        # Clinical
        "seizures":                seizures,
        "seizure_type":            seizure_type,
        "dre":                     dre,
        "idd":                     idd,
        "psychiatric":             psychiatric,
        "behavioral":              behavioral,
        "b6_trial":                b6_trial,
        "b6_responded":            b6_responded,
        "protein_restricted":      protein_restricted,
        "nbs_detected":            nbs_detected,
    }


_PATIENTS = [_make_patient(i + 1) for i in range(40)]


# ---------------------------------------------------------------------------
# API response functions
# ---------------------------------------------------------------------------

def get_overview() -> dict:
    n = len(_PATIENTS)

    def avg(key):
        vals = [p[key] for p in _PATIENTS if isinstance(p.get(key), (int, float))]
        return round(sum(vals) / len(vals), 1) if vals else 0

    def pct(pred):
        return round(100 * sum(1 for p in _PATIENTS if pred(p)) / n, 1)

    pheno_dist = {}
    for p in _PATIENTS:
        short = p["phenotype"].split("(")[0].strip()
        pheno_dist[short] = pheno_dist.get(short, 0) + 1

    variant_dist = {}
    for p in _PATIENTS:
        v = p["variant"]
        variant_dist[v] = variant_dist.get(v, 0) + 1

    return {
        "dashboard_id":   "prodh",
        "title":          "PRODH Epilepsy Dashboard",
        "subtitle":       "Hyperprolinemia Type I — Proline Dehydrogenase Deficiency / Proline NMDA-Agonism / GABA-Transport Inhibition",
        "gene":           "PRODH",
        "disease_name":   "Hyperprolinemia Type I (Proline Oxidase / PRODH Deficiency)",
        "chromosome":     "19q13.2",
        "inheritance":    "Autosomal Recessive — biallelic LOF; some heterozygotes mildly elevated",
        "omim_gene":      "OMIM *606810",
        "omim_disease":   "OMIM #239500",
        "protein_size":   "601 aa; FAD-dependent flavoenzyme; inner mitochondrial membrane; monotopic",
        "prevalence":     "~200–400 cases worldwide (2026); may be underdiagnosed; many asymptomatic",
        "cohort_n":       n,
        "phenotype_distribution": pheno_dist,
        "variant_distribution":   variant_dist,
        "function": (
            "PRODH (Proline Oxidase/Dehydrogenase) catalyses: L-Proline + FAD → "
            "Delta-1-pyrroline-5-carboxylate (P5C) + FADH2. "
            "Step 1 of proline catabolism: Proline → [PRODH] → P5C → [ALDH4A1] → L-Glutamate → TCA/GABA."
        ),
        "mechanism": (
            "PRODH LOF → Proline accumulates (350–1000 µmol/L). P5C DOES NOT accumulate "
            "(ALDH4A1 is intact). PLP is NORMAL (no P5C-PLP adduct). "
            "Epileptogenic mechanisms (weaker than Type II): "
            "(1) Proline acts as partial NMDA receptor agonist → neuroexcitatory; "
            "(2) Proline competitively inhibits GABA transporters SLC6A1/GAT-1 and SLC6A11/GAT-3 "
            "→ disrupted inhibitory synchrony; "
            "(3) At high concentrations: mitochondrial membrane potential perturbation via FAD/FADH2 imbalance."
        ),
        "key_positive_features": (
            "Proline ELEVATED (350–1000 µmol/L; normal <260) — LESS than Type II (>1000–2200+) · "
            "Urine proline ELEVATED (prolinuria) · "
            "Psychiatric features: schizophrenia-like (15–25%), bipolar-like (10–15%) — UNIQUE to Type I"
        ),
        "key_negative_features": (
            "P5C NORMAL — MOST IMPORTANT DIFFERENTIAL from Type II (ALDH4A1) · "
            "PLP NORMAL — NO B6 indication or response · "
            "alpha-AASA NORMAL (KEY vs ALDH7A1/PDE where MARKEDLY ELEVATED) · "
            "Pipecolic acid NORMAL (KEY vs ALDH7A1 where elevated) · "
            "MMA NORMAL · tHcy NORMAL · Methionine NORMAL · Lactate NORMAL · Ammonia NORMAL"
        ),
        "nbs_primary":   "Plasma amino acids — proline ELEVATED (350–1000 µmol/L) — milder flag than Type II; some expanded NBS programs detect",
        "nbs_secondary": "Urine amino acids (prolinuria confirmation); P5C assay NOT needed (will be normal); plasma PLP (will be normal)",
        "kpis": {
            "avg_proline_umol_l":       avg("proline_umol_l"),
            "avg_p5c_umol_l":           avg("p5c_umol_l"),
            "avg_plp_nmol_l":           avg("plp_nmol_l"),
            "pct_seizures":             pct(lambda p: p["seizures"]),
            "pct_dre":                  pct(lambda p: p["dre"]),
            "pct_b6_trial":             pct(lambda p: p["b6_trial"]),
            "pct_b6_responded":         pct(lambda p: p["b6_responded"]),  # should be near 0
            "pct_idd":                  pct(lambda p: p["idd"]),
            "pct_psychiatric":          pct(lambda p: p["psychiatric"]),
            "pct_behavioral":           pct(lambda p: p["behavioral"]),
            "pct_nbs_detected":         pct(lambda p: p["nbs_detected"]),
            "pct_protein_restricted":   pct(lambda p: p["protein_restricted"]),
            "pct_p5c_normal":           pct(lambda p: p["p5c_normal"]),   # should be 100%
            "pct_plp_normal":           pct(lambda p: p["plp_normal"]),   # should be 100%
        },
        "biomarkers": [
            {
                "name":         "Proline (plasma)",
                "mean":         avg("proline_umol_l"),
                "unit":         "µmol/L",
                "normal_range": "< 260 µmol/L",
                "significance": "ELEVATED (350–1000) — primary diagnostic biomarker; LESS elevated than Type II (>1000–2200+)",
            },
            {
                "name":         "P5C / Delta-1-pyrroline-5-carboxylate (plasma)",
                "mean":         avg("p5c_umol_l"),
                "unit":         "µmol/L",
                "normal_range": "< 5 µmol/L",
                "significance": "NORMAL in Type I — MOST CRITICAL DIFFERENTIAL from Type II where P5C is ELEVATED-PATHOGNOMONIC",
            },
            {
                "name":         "PLP / Pyridoxal-5-phosphate (plasma)",
                "mean":         avg("plp_nmol_l"),
                "unit":         "nmol/L",
                "normal_range": "35–110 nmol/L",
                "significance": "NORMAL in Type I — no P5C-PLP inactivation; B6/pyridoxine has NO indication",
            },
            {
                "name":         "alpha-AASA (urine)",
                "mean":         None,
                "unit":         "mmol/mol creatinine",
                "normal_range": "< 1 mmol/mol creatinine",
                "significance": "NORMAL — KEY NEGATIVE vs ALDH7A1/PDE (antiquitin) where MARKEDLY ELEVATED (pathognomonic)",
            },
            {
                "name":         "Pipecolic acid (plasma)",
                "mean":         None,
                "unit":         "µmol/L",
                "normal_range": "< 5 µmol/L",
                "significance": "NORMAL — KEY NEGATIVE vs ALDH7A1/PDE where ELEVATED",
            },
            {
                "name":         "Total homocysteine (tHcy)",
                "mean":         avg("thcy_umol_l"),
                "unit":         "µmol/L",
                "normal_range": "< 15 µmol/L",
                "significance": "NORMAL — CBS and folate cycle intact; SAM-independent pathway",
            },
            {
                "name":         "MMA (methylmalonic acid)",
                "mean":         None,
                "unit":         "mmol/mol creatinine",
                "normal_range": "< 5 mmol/mol creatinine",
                "significance": "NORMAL — propionate/B12 pathway intact; KEY NEGATIVE vs MMUT/MMAA/MMAB/cblC",
            },
            {
                "name":         "Methionine (plasma)",
                "mean":         avg("methionine_umol_l"),
                "unit":         "µmol/L",
                "normal_range": "15–45 µmol/L",
                "significance": "NORMAL — SAM cycle unaffected; KEY NEGATIVE vs GNMT/AHCY/CBS",
            },
            {
                "name":         "Urine proline",
                "mean":         None,
                "unit":         "elevated (qualitative)",
                "normal_range": "trace / absent",
                "significance": "ELEVATED (prolinuria) — overflow from high plasma proline; confirms hyperprolinemia",
            },
        ],
        "seizure_types": [
            {
                "type":        "Febrile seizure",
                "description": "Most common in Type I; fever amplifies NMDA agonism of proline; triggers in 40% of seizure patients",
                "prevalence_in_seizure_patients": "~40%",
            },
            {
                "type":        "Focal cortical",
                "description": "Proline NMDA agonism + GABA transport inhibition (SLC6A1/SLC6A11); 25% of seizure patients",
                "prevalence_in_seizure_patients": "~25%",
            },
            {
                "type":        "Absence",
                "description": "Mild-symptomatic phenotype; proline-mediated seizure threshold lowering; 20%",
                "prevalence_in_seizure_patients": "~20%",
            },
            {
                "type":        "GTCS",
                "description": "Classic-symptomatic phenotype with higher proline load; 10%",
                "prevalence_in_seizure_patients": "~10%",
            },
            {
                "type":        "Myoclonic",
                "description": "Least common in Type I; mild mechanism without GABA collapse; 5%",
                "prevalence_in_seizure_patients": "~5%",
            },
        ],
        "key_statistics": {
            "seizure_prevalence_type1":     "25–35% (vs 60–80% in Type II ALDH4A1)",
            "dre_prevalence_type1":         "<15% (vs 25–40% in Type II)",
            "idd_prevalence_type1":         "30–40%; 40–50% cognitively normal",
            "psychiatric_prevalence_type1": "15–25% schizophrenia-like; 10–15% bipolar-like (UNIQUE to Type I)",
            "b6_response":                  "None — PLP intact; no P5C-PLP inactivation",
            "proline_range_type1":          "350–1000 µmol/L (vs >1000–2200+ in Type II)",
            "p5c_in_type1":                 "NORMAL (<5 µmol/L) — critical differential from Type II",
        },
        "type1_vs_type2": {
            "shared":   "Proline elevated; AR; proline catabolic pathway; mitochondrial",
            "PRODH":    "P5C NORMAL (ALDH4A1 intact); proline 350–1000; PLP NORMAL; no B6 effect; psychiatric prominent",
            "ALDH4A1":  "P5C ELEVATED (PATHOGNOMONIC); proline >1000; PLP LOW; B6 partial response (30–50%)",
            "epilepsy": "PRODH: 25–35% (NMDA + GABA transport, weak); ALDH4A1: 60–80% (P5C-PLP → GABA collapse)",
        },
        "prodh_and_schizophrenia": {
            "region":    "19q13.2; functional relationship with 22q11.2 deletion syndrome (DiGeorge/velocardiofacial syndrome)",
            "mechanism": "PRODH haploinsufficiency → mild proline elevation → NMDA receptor hypofunction → dopaminergic/glutamatergic dysregulation",
            "clinical":  "Schizophrenia-like 15–25%; bipolar-like 10–15%; ADHD-like 20–30% in biallelic LOF Type I",
            "note":      "This psychiatric overlap is UNIQUE to Type I and less prominent in Type II (ALDH4A1)",
        },
    }


def get_breakdown() -> dict:
    n = len(_PATIENTS)

    def pct(pred):
        return round(100 * sum(1 for p in _PATIENTS if pred(p)) / n, 1)

    def avg(key):
        vals = [p[key] for p in _PATIENTS if isinstance(p.get(key), (int, float))]
        return round(sum(vals) / len(vals), 1) if vals else 0

    # Phenotype distribution
    pheno_dist = {}
    for p in _PATIENTS:
        short = p["phenotype"].split("(")[0].strip()
        pheno_dist[short] = pheno_dist.get(short, 0) + 1

    # Variant counts
    var_counts = {}
    for p in _PATIENTS:
        v = p["variant"]
        var_counts[v] = var_counts.get(v, 0) + 1

    # Seizure type distribution
    sz_counts = {}
    for p in _PATIENTS:
        st = p.get("seizure_type")
        if st:
            short = st.split("(")[0].strip()
            sz_counts[short] = sz_counts.get(short, 0) + 1

    # Biomarker ranges by phenotype
    def biomarker_by_pheno(key):
        out = {}
        for short_label in ["Asymptomatic-Incidental", "Mild-Symptomatic", "Classic-Symptomatic"]:
            pts  = [p for p in _PATIENTS if short_label in p["phenotype"]]
            vals = [p[key] for p in pts if isinstance(p.get(key), (int, float))]
            if vals:
                out[short_label] = {
                    "min":  min(vals),
                    "max":  max(vals),
                    "mean": round(sum(vals) / len(vals), 1),
                    "n":    len(vals),
                }
        return out

    return {
        "cohort_n":           n,
        "phenotype_dist":     pheno_dist,
        "variant_dist":       var_counts,
        "seizure_type_dist":  sz_counts,
        "clinical_rates": {
            "pct_seizures":           pct(lambda p: p["seizures"]),
            "pct_dre":                pct(lambda p: p["dre"]),
            "pct_idd":                pct(lambda p: p["idd"]),
            "pct_psychiatric":        pct(lambda p: p["psychiatric"]),
            "pct_behavioral":         pct(lambda p: p["behavioral"]),
            "pct_b6_trial":           pct(lambda p: p["b6_trial"]),
            "pct_b6_responded":       pct(lambda p: p["b6_responded"]),
            "pct_nbs_detected":       pct(lambda p: p["nbs_detected"]),
            "pct_protein_restricted": pct(lambda p: p["protein_restricted"]),
            "pct_p5c_normal":         pct(lambda p: p["p5c_normal"]),    # expected 100%
            "pct_plp_normal":         pct(lambda p: p["plp_normal"]),    # expected 100%
            "pct_mma_normal":         pct(lambda p: p["mma_normal"]),    # expected 100%
            "pct_alpha_aasa_normal":  pct(lambda p: p["alpha_aasa_normal"]),  # expected 100%
            "pct_pipecolic_normal":   pct(lambda p: p["pipecolic_normal"]),   # expected 100%
        },
        "biomarker_ranges": {
            "proline_umol_l": biomarker_by_pheno("proline_umol_l"),
            "p5c_umol_l":     biomarker_by_pheno("p5c_umol_l"),   # all should be <5
            "plp_nmol_l":     biomarker_by_pheno("plp_nmol_l"),   # all should be 35–110
        },
        "avg_biomarkers": {
            "proline_umol_l":     avg("proline_umol_l"),
            "p5c_umol_l":         avg("p5c_umol_l"),
            "plp_nmol_l":         avg("plp_nmol_l"),
            "thcy_umol_l":        avg("thcy_umol_l"),
            "methionine_umol_l":  avg("methionine_umol_l"),
        },
        "patients": _PATIENTS,
        "key_differentials": [
            {
                "disease":     "ALDH4A1 Deficiency (Hyperprolinemia Type II)",
                "shared":      "Proline elevated; AR; proline catabolic pathway; seizures",
                "distinguish": "PRODH Type I: P5C NORMAL + PLP NORMAL + no B6 response; proline 350–1000; "
                               "ALDH4A1 Type II: P5C ELEVATED (PATHOGNOMONIC) + PLP LOW + B6 partial response (30–50%); proline >1000",
            },
            {
                "disease":     "ALDH7A1 / Antiquitin Deficiency (PDE-ALDH7A1)",
                "shared":      "Seizures; metabolic; B6-responsive (ALDH7A1 only)",
                "distinguish": "PRODH: proline ELEVATED + alpha-AASA NORMAL + pipecolic NORMAL + NO B6 response; "
                               "ALDH7A1: alpha-AASA MARKEDLY ELEVATED (PATHOGNOMONIC) + pipecolic HIGH + proline NORMAL + 85%+ B6 response",
            },
            {
                "disease":     "PNPO Deficiency (Pyridoxamine-5'-phosphate oxidase deficiency)",
                "shared":      "Seizures; metabolic; neonatal/infantile onset (PNPO)",
                "distinguish": "PRODH: proline ELEVATED + PLP NORMAL + NO B6 response; "
                               "PNPO: proline NORMAL + PLP LOW + responds to PLP only (not pyridoxine); severe neonatal epilepsy",
            },
            {
                "disease":     "22q11.2 Deletion Syndrome (DiGeorge/VCF syndrome)",
                "shared":      "Psychiatric features; mild proline elevation possible (PRODH haploinsufficiency in deletion region); seizures",
                "distinguish": "22q11.2 del: chromosomal micro-deletion confirmed by CMA/FISH; cardiac/palate defects; T-cell lymphopenia; "
                               "PRODH Type I: biallelic LOF only; normal chromosomes; higher proline (350–1000 vs 200–500 in 22q11.2 het)",
            },
            {
                "disease":     "Schizophrenia (idiopathic)",
                "shared":      "Psychosis-like features; dopaminergic/glutamatergic imbalance",
                "distinguish": "PRODH Type I: hyperprolinemia CONFIRMED on amino acids + biallelic PRODH variants; "
                               "Idiopathic schizophrenia: normal plasma proline; no metabolic basis",
            },
        ],
        "drug_risks": [
            {
                "drug":   "Valproate (VPA)",
                "risk":   "MODERATE RISK",
                "reason": "VPA inhibits PRODH in vitro → raises proline further in already-elevated pool; "
                          "carnitine depletion risk; less hazardous than in Type II (no PLP mechanism) "
                          "but monitoring recommended; consider alternative AEDs.",
            },
            {
                "drug":   "B6 antagonists (INH, cycloserine, D-penicillamine)",
                "risk":   "MODERATE RISK",
                "reason": "PLP is intact at baseline in Type I (less hazardous than Type II), but "
                          "prudent to avoid additional PLP interference; no absolute CI unless co-morbidity.",
            },
            {
                "drug":   "Phenylalanine-heavy protein loads / high-Phe formula",
                "risk":   "MODERATE RISK",
                "reason": "Phenylalanine competes with proline for shared amino acid transporters (LAT1/SLC7A5, "
                          "intestinal IMINO transporter); may reduce proline clearance; avoid excess.",
            },
            {
                "drug":   "Levetiracetam (LEV)",
                "risk":   "Level B SAFE",
                "reason": "No interaction with proline pathway or PLP; first-line AED in Type I",
            },
            {
                "drug":   "B6 / Pyridoxine / PLP",
                "risk":   "NOT INDICATED (unlike Type II)",
                "reason": "PLP is normal in Type I — there is NO P5C-PLP inactivation mechanism; "
                          "B6 supplementation has no seizure benefit and may be misleading diagnostically.",
            },
        ],
        "treatments": [
            {
                "treatment": "Proline-restricted diet",
                "level":     "Level B",
                "rationale": "Reduce proline intake: low-proline proteins; avoid collagen, gelatin, casein-rich foods. "
                             "Lowers plasma proline → less NMDA agonism, reduced GABA transport competition. "
                             "Moderate benefit; diet adherence challenging; dietitian supervision required.",
            },
            {
                "treatment": "Levetiracetam (LEV)",
                "level":     "Level B",
                "rationale": "First-line AED in Type I; no interaction with proline pathway or PLP; well-tolerated across age groups.",
            },
            {
                "treatment": "Standard AEDs (lamotrigine, oxcarbazepine, clonazepam)",
                "level":     "Level C",
                "rationale": "For seizures not controlled by LEV; clinical choice guided by seizure type and age; "
                             "VPA used cautiously with monitoring (PRODH inhibitor in vitro).",
            },
            {
                "treatment": "Psychiatric medication",
                "level":     "Level C",
                "rationale": "Second-generation antipsychotics for schizophrenia-like features; "
                             "mood stabilisers for bipolar-like phenotype; stimulants/atomoxetine for ADHD-like. "
                             "Clinical psychiatry consultation needed; metabolic monitoring.",
            },
            {
                "treatment": "Betaine (trimethylglycine)",
                "level":     "Level C — Experimental",
                "rationale": "May promote alternative methylation pathways affecting proline metabolism; "
                             "limited evidence; under study; not standard of care.",
            },
        ],
    }


def get_definitions() -> dict:
    return {
        "disease":      "Hyperprolinemia Type I (PRODH / Proline Oxidase / Proline Dehydrogenase Deficiency)",
        "gene_full":    "PRODH — Proline Dehydrogenase 1 (also PRODH1; Proline Oxidase; PO)",
        "omim_gene":    "OMIM *606810",
        "omim_disease": "OMIM #239500",
        "chromosome":   "19q13.2",
        "protein":      "601 aa; FAD-dependent flavoenzyme; inner mitochondrial membrane; monotopic; electron donor to ubiquinone (CoQ10)",
        "inheritance":  "Autosomal Recessive — biallelic loss-of-function required for full hyperprolinemia; some heterozygotes may show mild elevation",
        "pathway": (
            "Proline catabolism (Step 1 of 2): "
            "L-Proline + FAD → [PRODH] → Delta-1-pyrroline-5-carboxylate (P5C) + FADH2; "
            "P5C → [ALDH4A1 / P5CDH] → L-Glutamate → TCA cycle / GABA synthesis. "
            "In PRODH LOF: proline accumulates; P5C does NOT accumulate (ALDH4A1 intact). "
            "In ALDH4A1 LOF (Type II): P5C accumulates; PLP inactivated → GABA synthesis collapses."
        ),
        "biomarker_glossary": {
            "Proline (plasma)":
                "Primary amino acid substrate for PRODH; ELEVATED (350–1000 µmol/L) in Type I; "
                "normal <260 µmol/L. LESS elevated than Type II (>1000–2200+).",
            "P5C (Delta-1-pyrroline-5-carboxylate)":
                "Cyclic intermediate product of PRODH (Step 1) and substrate for ALDH4A1 (Step 2). "
                "NORMAL in Type I — ALDH4A1 is intact and converts P5C to glutamate normally. "
                "ELEVATED (PATHOGNOMONIC) only in Type II (ALDH4A1 deficiency). "
                "This NORMAL P5C is the single most important Type I vs Type II distinguishing biomarker.",
            "PLP (Pyridoxal-5-phosphate)":
                "Active form of vitamin B6; cofactor for GAD65/67 (GABA synthesis), CBS, ASAT, AADC etc. "
                "NORMAL in Type I — no P5C-PLP inactivation. LOW in Type II (secondary B6 deficiency). "
                "Normal PLP in Type I = NO B6/pyridoxine indication or expected response.",
            "alpha-AASA (alpha-aminoadipic semialdehyde)":
                "PATHOGNOMONIC biomarker of ALDH7A1/PDE (antiquitin deficiency). "
                "NORMAL in Type I PRODH deficiency — KEY NEGATIVE for ALDH7A1/PDE.",
            "Pipecolic acid":
                "Elevated in ALDH7A1/PDE via the piperideine-6-carboxylate (P6C) pathway. "
                "NORMAL in PRODH Type I — KEY NEGATIVE for ALDH7A1/PDE.",
            "PRODH":
                "Proline Oxidase/Dehydrogenase — Step 1 enzyme (proline → P5C). "
                "FAD-dependent; inner mitochondrial membrane; electrons donated to CoQ10/ubiquinone. "
                "LOF causes Hyperprolinemia Type I (#239500, *606810).",
            "ALDH4A1 / P5CDH":
                "Delta-1-pyrroline-5-carboxylate dehydrogenase — Step 2 enzyme (P5C → glutamate). "
                "INTACT in Type I → P5C is normally converted; does not accumulate. "
                "LOF causes Hyperprolinemia Type II (#239510, *606811).",
            "GAD65/GAD67":
                "Glutamate Decarboxylases — PLP-dependent enzymes that synthesize GABA. "
                "FUNCTIONAL in Type I (PLP is normal). Impaired only in Type II (PLP depleted by P5C-PLP adduct).",
            "SLC6A1 / GAT-1":
                "GABA transporter 1 — re-uptakes synaptic GABA. Competitively inhibited by high proline. "
                "Mechanism contributing to disrupted inhibitory synchrony in Type I.",
            "SLC6A11 / GAT-3":
                "GABA transporter 3 — astrocytic GABA re-uptake. Also competitively inhibited by proline.",
            "22q11.2 deletion / DiGeorge syndrome":
                "Chromosomal micro-deletion including PRODH regulatory region at 22q11.2. "
                "PRODH haploinsufficiency in this deletion → mild proline elevation (200–500 µmol/L). "
                "Associated with psychiatric susceptibility. PRODH Type I (biallelic LOF) has higher "
                "proline (350–1000) and is distinct from chromosomal 22q11.2 deletion.",
        },
        "key_concepts": [
            "P5C is NORMAL in Type I — the single fastest biomarker to distinguish Type I (PRODH) from Type II (ALDH4A1)",
            "PLP is NORMAL in Type I — B6/pyridoxine has NO indication, NO expected benefit, and is NOT analogous to Type II",
            "Seizure prevalence is MILDER (25–35%) than Type II (60–80%) because GABA synthesis (GAD65/67) is INTACT",
            "Psychiatric features (schizophrenia-like, bipolar-like) are UNIQUE to Type I — NMDA receptor hypofunction mechanism",
            "Proline range 350–1000 µmol/L — less than Type II (>1000–2200+); prolinuria present",
            "alpha-AASA and pipecolic acid are NORMAL — critical negatives against ALDH7A1/PDE",
            "VPA is a PRODH inhibitor in vitro — use with monitoring; preferred to avoid in Type I",
            "NBS: partially detected; proline elevation milder than Type II; many asymptomatic cases missed",
            "Betaine is experimental only; no standard metabolic rescue therapy as effective as B6 is in Type II",
            "40% of Type I patients are asymptomatic — incidental NBS or family screening detection",
        ],
        "variants_glossary": {
            "p.Pro406Leu":
                "Mitochondrial targeting region; most common worldwide (~25%); mild phenotype; "
                "associated with 22q11.2-like psychiatric susceptibility in heterozygotes",
            "p.Arg185Trp":
                "FAD binding domain; ~20%; moderate-severe; impairs FAD cofactor assembly → reduced electron transfer",
            "p.Leu441Pro":
                "Active site; ~15%; moderate; proline binding pocket disrupted; partial residual activity",
            "p.Ala239Val":
                "Substrate binding domain; ~12%; mild-moderate; reduced proline affinity (Km shift)",
            "p.Arg564Cys":
                "Membrane anchoring domain; ~10%; moderate; disrupts inner mitochondrial membrane insertion",
            "c.IVS4+1G>A":
                "Splice-null; ~10%; most severe within Type I; exon 4 skipped; truncated inactive protein; "
                "still milder overall than Type II severe phenotypes due to absence of P5C-PLP mechanism",
            "p.Gly372Ser":
                "Dimer interface; ~8%; mild; disrupts homodimerisation; some residual monomer activity",
        },
        "normal_ranges": {
            "Proline (plasma)":        "< 260 µmol/L (Type I: 350–1000 µmol/L)",
            "P5C (plasma)":            "< 5 µmol/L (Type I: NORMAL <5 — ALDH4A1 intact)",
            "PLP (plasma)":            "35–110 nmol/L (Type I: NORMAL 35–110 — no P5C-PLP inactivation)",
            "alpha-AASA (urine)":      "< 1 mmol/mol creatinine (NORMAL in Type I; HIGH in ALDH7A1/PDE)",
            "Pipecolic acid (plasma)": "< 5 µmol/L (NORMAL in Type I; ELEVATED in ALDH7A1/PDE)",
            "MMA (urine)":             "< 5 mmol/mol creatinine (NORMAL in Type I)",
            "Homocysteine (total)":    "< 15 µmol/L (NORMAL in Type I; high in CBS/MTHFR deficiency)",
            "Methionine (plasma)":     "15–45 µmol/L (NORMAL in Type I)",
            "Urine proline":           "Trace/absent (ELEVATED / prolinuria in Type I)",
        },
        "drug_risks": {
            "VPA (valproate)":                            "MODERATE RISK — PRODH inhibitor in vitro; raises proline further; carnitine depletion",
            "B6 antagonists (INH, cycloserine)":          "MODERATE RISK — less hazardous than Type II (PLP baseline normal) but prudent to avoid",
            "Phenylalanine-heavy protein loads":          "MODERATE RISK — competes with proline for shared transporters",
            "Levetiracetam (LEV)":                        "Level B SAFE — no metabolic interaction; first-line",
            "B6 / Pyridoxine / PLP":                      "NOT INDICATED — PLP normal; no P5C-PLP mechanism; no seizure benefit",
            "High-proline foods (collagen, gelatin)":     "MODERATE RISK — increases proline load directly",
        },
        "treatments": {
            "Proline-restricted diet":     "Level B — primary metabolic intervention; reduce proline intake",
            "Levetiracetam (LEV)":         "Level B — first-line AED; no metabolic interaction",
            "Standard AEDs":               "Level C — for refractory seizures; VPA used cautiously",
            "Psychiatric medications":     "Level C — antipsychotics for schizophrenia-like; mood stabilisers for bipolar-like; stimulants for ADHD-like",
            "Betaine (experimental)":      "Level C — may support alternative proline metabolism; limited evidence",
            "NO B6 / pyridoxine":          "PLP is normal; this is NOT Type II; do NOT give B6 expecting seizure benefit",
        },
    }
