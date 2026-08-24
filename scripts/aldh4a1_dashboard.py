#!/usr/bin/env python3
"""ALDH4A1 (Hyperprolinemia Type II / P5C Dehydrogenase Deficiency) Epilepsy Dashboard.

ALDH4A1 encodes Delta-1-pyrroline-5-carboxylate dehydrogenase (P5CDH), the second
mitochondrial enzyme of the proline catabolic pathway.

PROLINE CATABOLIC PATHWAY:
  Proline
    → [PRODH / Proline oxidase, FAD-dependent, inner mitochondrial membrane]
  Delta-1-pyrroline-5-carboxylate (P5C)
    → [ALDH4A1 / P5CDH, NAD-dependent, mitochondrial matrix]
  L-Glutamate
    → (TCA via alpha-KG; gluconeogenesis; GABA synthesis)

ALDH4A1 ENZYMATIC FUNCTION:
  P5C (cyclic form) ← spontaneous → Glutamate-5-semialdehyde (GSA, open chain)
  ALDH4A1 catalyses: Glutamate-5-semialdehyde + NAD+ + H2O → L-Glutamate + NADH + H+
  (irreversible oxidation; NAD as hydride acceptor)

ALDH4A1 LOF — P5C ACCUMULATES:
  P5C cannot be converted to glutamate → P5C accumulates in mitochondria, plasma, urine

THE CRITICAL EPILEPTOGENIC MECHANISM — P5C INACTIVATES PLP:
  P5C (specifically its open-chain form GSA) forms a Schiff base with
  Pyridoxal-5'-phosphate (PLP) via the PLP formyl group → P5C-PLP adduct.
  This adduct is catalytically INACTIVE.
  Consequence: secondary functional deficiency of ALL PLP-dependent enzymes:
    → GAD65/GAD67 (glutamate → GABA): IMPAIRED → GABA synthesis ↓ → seizures
    → Cystathionine beta-synthase (CBS): IMPAIRED → mild tHcy elevation possible
    → Aspartate aminotransferase (ASAT): IMPAIRED
    → Alanine aminotransferase (ALAT): IMPAIRED
    → AADC (aromatic amino acid decarboxylase): IMPAIRED
    → PDXK (pyridoxal kinase): IMPAIRED
  This IS THE SAME MECHANISM by which P5C drives epilepsy in Hyperprolinemia Type II.
  Partial pyridoxine/PLP supplementation can RESTORE PLP function and reduce seizure burden.

DISTINGUISHING HYPERPROLINEMIA TYPE I vs TYPE II:
  Feature                   Type I (PRODH)              Type II (ALDH4A1)
  Proline (plasma)          Elevated (350–1000 µmol/L)  MARKEDLY ELEVATED (>1000, up to 2000+)
  P5C (plasma/urine)        NORMAL (P5CDH intact)       ELEVATED — PATHOGNOMONIC
  PLP inactivation          ABSENT                      PRESENT — P5C-PLP adduct
  Pyridoxine responsiveness NONE                        PARTIAL (30–50%) — PLP restoration
  Seizure severity          Mild (25–35%)               HIGHER (60–80%)
  Drug-resistant epilepsy   Rare (<15%)                 25–40%
  IDD severity              Mild-moderate (30–40%)      Moderate (50–70%)

KEY BIOMARKERS IN ALDH4A1 DEFICIENCY:
  ELEVATED (pathognomonic constellation):
    Proline: MARKEDLY ELEVATED (>1000–2000+ µmol/L; normal <260)
    P5C/GSA: ELEVATED in plasma and urine — most specific marker
    Urine hydroxyproline: mildly elevated (overflow from high proline hydroxylation flux)
    Urine proline: MARKEDLY ELEVATED (prolinuria)

  NORMAL (key negatives):
    MMA: NORMAL — propionate/B12 pathway intact
    Total homocysteine: NORMAL or borderline (<20) — CBS may be partially impaired by
      P5C-PLP but rarely reaches diagnostic HHcy threshold
    Pipecolic acid: NORMAL — KEY NEGATIVE vs ALDH7A1 (PDE) where pipecolic is elevated
    alpha-AASA: NORMAL — KEY NEGATIVE vs ALDH7A1 (PDE) where alpha-AASA is the
      pathognomonic biomarker (urinary biomarker of PDE-ALDH7A1)
    Lactate/pyruvate: NORMAL — no primary mitochondrial RC involvement
    Ammonia: NORMAL — no primary urea cycle involvement
    MeCbl / AdoCbl: NORMAL — cobalamin intact
    Methionine: NORMAL — no SAM cycle involvement
    Glycine: NORMAL — GCS (GLDC/AMT/GCSH) intact

  SECONDARY (secondary B6 deficiency consequences):
    Plasma PLP: LOW to VERY LOW (P5C-PLP adduct depletes active PLP pool)
    CSF PLP: LOW — correlates with seizure severity
    Plasma pyridoxal: LOW
    Urine xanthurenic acid: ELEVATED (tryptophan loading test if PLP-dependent kynurenase
      impaired — not routine but diagnostically useful)

ALDH4A1 vs ALDH7A1 (PDE-ALDH7A1 / Antiquitin) — CRITICAL DISTINCTION:
  ALDH4A1 (Type II HyPro):     proline → P5C → [BLOCKED] → glutamate
  ALDH7A1 (Antiquitin / PDE): alpha-AASA → [BLOCKED] → alpha-aminoadipic semialdehyde → ... → pipecolic acid
  Both cause PLP inactivation, but via DIFFERENT accumulating metabolites:
    ALDH4A1: P5C inactivates PLP
    ALDH7A1: P6C (1-piperideine-6-carboxylate) inactivates PLP
  Biomarker distinction:
    ALDH4A1: Proline MARKEDLY HIGH + P5C ELEVATED; alpha-AASA NORMAL; pipecolic NORMAL
    ALDH7A1: alpha-AASA MARKEDLY HIGH (PATHOGNOMONIC); pipecolic HIGH; proline NORMAL

NBS STATUS:
  Standard NBS: MISSED by most panels — proline not on standard MS/MS NBS in most jurisdictions
  Some expanded NBS programs flag proline >400 µmol/L; Type II (>1000) more likely detected
  P5C not routinely measured in NBS
  Diagnosis usually triggered by: refractory neonatal/infantile seizures + plasma amino acids

TREATMENTS:
  Pyridoxine (B6) / PLP supplementation — Level B:
    Rationale: restores active PLP pool by mass action, overcoming P5C-PLP inactivation
    30–50% seizure response; partial (unlike ALDH7A1 where >85% B6 response)
    Dose trial: B6 100–300 mg/day (IV pyridoxine during acute SE trial mandatory)
    PLP (pyridoxal-5'-phosphate) preferred for CNS delivery
  Proline-restricted diet — Level B:
    Reduces proline flux → less P5C generation
    Low-proline diet (avoid high-proline proteins: collagen, gelatin, casein)
    Difficult to sustain; moderate benefit
  LEV (levetiracetam) — Level B: first-line AED (no specific metabolic interaction)
  Benzodiazepines — Level B: crisis management
  ACTH — Level C: infantile spasms variant; limited data
  KD (ketogenic diet) — Level C: may help if PLP partial rescue insufficient

HIGH-RISK DRUGS:
  VPA (valproate) — HIGH RISK:
    Mechanism 1: VPA inhibits mitochondrial enzymes → worsens energy deficit
    Mechanism 2: VPA directly inhibits PRODH in vitro (raises proline → more P5C → more PLP inactivation)
    Mechanism 3: VPA secondary PLP depletion (well-established B6-interacting drug)
    TRIPLE additive hazard: avoid if possible; if unavoidable, supplement PLP aggressively
  High-protein diet (collagen-rich) — HIGH RISK: raises proline → more P5C
  B6 antagonists (INH, D-penicillamine) — ABSOLUTE CI: further deplete PLP
  Phenytoin — MODERATE RISK: competes for PLP-dependent enzymes at AADC step
  CBZ / OXC — MODERATE RISK: potential proline flux augmentation

INHERITANCE:
  Gene: ALDH4A1, 1p36.13, 563 amino acids (mitochondrial targeting sequence 22 aa + 541 aa mature)
  Protein: homodimeric, mitochondrial matrix
  Inheritance: Autosomal Recessive — biallelic LOF required
  ~100–200 cases worldwide (2026); ultrarare; may be underdiagnosed given NBS gaps
  De novo variants: rare (<5%); mostly inherited biallelic AR

OMIM:
  Gene:    *606811 (ALDH4A1)
  Disease: #239510 (Hyperprolinemia, Type II)
"""

import random

# Deterministic cohort — seed 141
_rng = random.Random(141)

# ---------------------------------------------------------------------------
# Phenotypic classes
# ---------------------------------------------------------------------------
PHENOTYPE_CLASSES = [
    "Classic-Severe (drug-resistant epilepsy / moderate-severe IDD / onset neonatal-infantile / P5C markedly elevated)",
    "Moderate (seizures partially controlled / moderate IDD / proline >1200 / partial B6 response)",
    "Mild-Attenuated (seizures mild-occasional / mild IDD or cognitive normal / proline 1000-1500 / good B6 response)",
]

PHENO_WEIGHTS = [0.45, 0.40, 0.15]

VARIANTS = [
    "p.Arg563Trp",    # ALDH active site; most common worldwide; ~25%; moderate-severe
    "p.Asp779Asn",    # NAD cofactor binding; ~18%; severe
    "p.Gly486Ser",    # cofactor domain; ~15%; moderate
    "p.Arg440His",    # substrate-binding groove; ~12%; moderate
    "p.Glu425Lys",    # catalytic glutamate residue; ~10%; severe
    "c.IVS9+1G>A",   # splice-null; ~10%; severe
    "p.Ala519Val",    # mild, partial function; ~10%; attenuated
]

VARIANT_WEIGHTS = [0.25, 0.18, 0.15, 0.12, 0.10, 0.10, 0.10]

SEIZURE_TYPES = [
    "Infantile spasms/West syndrome (P5C-PLP → GABA deficiency neonatal; 30%)",
    "Focal cortical (P5C-PLP inactivation → GABA deficit; 25%)",
    "GTCS (generalised; moderate-severe phenotype; 20%)",
    "Myoclonic (PLP-dependent GAD impairment; 15%)",
    "Absence (mild-moderate phenotype; 10%)",
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
    is_severe = "Classic-Severe" in pheno
    is_mild   = "Mild-Attenuated" in pheno
    r = _rng.random()
    cum = 0.0
    for v, w in zip(VARIANTS, VARIANT_WEIGHTS):
        cum += w
        if r < cum:
            if is_mild and v in ("p.Asp779Asn", "c.IVS9+1G>A", "p.Glu425Lys"):
                return _rng.choice(["p.Ala519Val", "p.Gly486Ser"])
            if is_severe and v == "p.Ala519Val":
                return _rng.choice(["p.Arg563Trp", "p.Asp779Asn", "c.IVS9+1G>A"])
            return v
    return VARIANTS[0]


def _make_patient(idx):
    pheno      = _choose_pheno()
    is_severe  = "Classic-Severe" in pheno
    is_mod     = "Moderate"       in pheno
    is_mild    = "Mild-Attenuated" in pheno

    # Proline — MARKEDLY ELEVATED in ALL patients (Type II hallmark)
    if is_severe:
        proline_umol_l   = round(_rng.uniform(1400, 2200), 1)
        p5c_umol_l       = round(_rng.uniform(60,   200),  1)  # elevated
        plp_nmol_l       = round(_rng.uniform(5,    18),   1)  # very low (<20)
    elif is_mod:
        proline_umol_l   = round(_rng.uniform(1000, 1600), 1)
        p5c_umol_l       = round(_rng.uniform(30,   90),   1)
        plp_nmol_l       = round(_rng.uniform(15,   35),   1)  # low
    else:  # mild
        proline_umol_l   = round(_rng.uniform(800,  1300), 1)
        p5c_umol_l       = round(_rng.uniform(15,   50),   1)
        plp_nmol_l       = round(_rng.uniform(30,   55),   1)  # borderline low

    # Normal reference: PLP 35-110 nmol/L; P5C <5 µmol/L; proline <260 µmol/L

    # Key negatives — all NORMAL
    mma_normal        = True
    thcy_umol_l       = round(_rng.uniform(6,  18),  1)   # normal or borderline
    methionine_umol_l = round(_rng.uniform(22, 48),  1)   # NORMAL
    pipecolic_normal  = True   # KEY NEGATIVE vs ALDH7A1
    alpha_aasa_normal = True   # KEY NEGATIVE vs ALDH7A1
    lactate_normal    = True
    ammonia_normal    = True

    # Urine proline: elevated (mirrors plasma)
    urine_proline_elevated = True

    # Clinical features
    seizures   = _rng.random() < (0.90 if is_severe else (0.75 if is_mod else 0.45))
    b6_trial   = True  # mandatory in all
    b6_respond = _rng.random() < (0.20 if is_severe else (0.50 if is_mod else 0.75))
    idd        = _rng.random() < (0.92 if is_severe else (0.72 if is_mod else 0.30))
    dre        = _rng.random() < (0.65 if is_severe else (0.25 if is_mod else 0.05)) if seizures else False
    behavioral = _rng.random() < (0.70 if is_severe else (0.50 if is_mod else 0.20))
    psychiatric = _rng.random() < (0.45 if is_severe else (0.30 if is_mod else 0.10))
    nbs_detected = _rng.random() < (0.50 if is_severe else (0.35 if is_mod else 0.25))
    protein_restricted = _rng.random() < (0.70 if is_severe else (0.50 if is_mod else 0.30))

    seizure_type = _rng.choice(SEIZURE_TYPES) if seizures else None

    age_onset_mo = (
        _rng.randint(0,  6) if is_severe else
        _rng.randint(3, 24) if is_mod    else
        _rng.randint(6, 48)
    )

    variant = _choose_variant(pheno)

    return {
        "id":                      f"ALDH4A1-{idx:03d}",
        "phenotype":               pheno,
        "variant":                 variant,
        "age_onset_months":        age_onset_mo,
        # KEY ELEVATED BIOMARKERS
        "proline_umol_l":          proline_umol_l,
        "p5c_umol_l":              p5c_umol_l,
        "plp_nmol_l":              plp_nmol_l,
        # KEY NEGATIVES (all normal)
        "mma_normal":              mma_normal,
        "homocysteine_umol_l":     thcy_umol_l,
        "methionine_umol_l":       methionine_umol_l,
        "pipecolic_normal":        pipecolic_normal,
        "alpha_aasa_normal":       alpha_aasa_normal,
        "lactate_normal":          lactate_normal,
        "ammonia_normal":          ammonia_normal,
        "urine_proline_elevated":  urine_proline_elevated,
        # Clinical
        "seizures":                seizures,
        "seizure_type":            seizure_type,
        "drug_resistant":          dre,
        "b6_trial":                b6_trial,
        "b6_responsive":           b6_respond,
        "idd":                     idd,
        "behavioral":              behavioral,
        "psychiatric":             psychiatric,
        "nbs_detected":            nbs_detected,
        "protein_restricted":      protein_restricted,
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

    return {
        "dashboard_id":   "aldh4a1",
        "title":          "ALDH4A1 Epilepsy Dashboard",
        "subtitle":       "Hyperprolinemia Type II — P5C Dehydrogenase Deficiency / P5C-PLP Inactivation / Secondary B6 Deficiency",
        "gene":           "ALDH4A1",
        "disease_name":   "Hyperprolinemia Type II (P5C Dehydrogenase / ALDH4A1 Deficiency)",
        "chromosome":     "1p36.13",
        "inheritance":    "Autosomal Recessive — biallelic LOF; AR",
        "omim_gene":      "OMIM *606811",
        "omim_disease":   "OMIM #239510",
        "protein_size":   "563 aa; NAD+-dependent; homodimeric; mitochondrial matrix",
        "prevalence":     "~100–200 cases worldwide (2026); ultrarare; likely underdiagnosed given NBS gaps",
        "cohort_n":       n,
        "phenotype_distribution": pheno_dist,
        "function": (
            "ALDH4A1 (P5CDH) catalyses: Glutamate-5-semialdehyde (GSA/P5C open-chain) + NAD+ + H2O "
            "→ L-Glutamate + NADH. Step 2 of proline catabolism: Proline → P5C [PRODH] → Glutamate [ALDH4A1]."
        ),
        "mechanism": (
            "ALDH4A1 LOF → P5C (Delta-1-pyrroline-5-carboxylate) accumulates → P5C forms Schiff-base adduct "
            "with PLP (pyridoxal-5-phosphate) → P5C-PLP adduct is catalytically INACTIVE → secondary functional "
            "PLP deficiency → GAD65/GAD67 impaired → GABA synthesis ↓ → seizures. "
            "This is mechanistically analogous to P6C inactivating PLP in ALDH7A1/PDE."
        ),
        "key_positive_features": (
            "Proline MARKEDLY ELEVATED (>1000–2000+ µmol/L; normal <260) + "
            "P5C ELEVATED (plasma/urine; PATHOGNOMONIC) + "
            "Plasma PLP LOW (secondary B6 deficiency) + "
            "PARTIAL pyridoxine/PLP responsiveness (30–50% seizure reduction)"
        ),
        "key_negative_features": (
            "alpha-AASA NORMAL (KEY vs ALDH7A1/PDE where MARKEDLY ELEVATED) · "
            "Pipecolic acid NORMAL (KEY vs ALDH7A1 where elevated) · "
            "MMA NORMAL · tHcy NORMAL–borderline · Methionine NORMAL · "
            "Lactate NORMAL · Ammonia NORMAL"
        ),
        "nbs_primary":  "Plasma amino acids — proline MARKEDLY ELEVATED (>1000 µmol/L) triggers reflex testing",
        "nbs_secondary": "P5C plasma/urine assay (most specific); plasma PLP; urine organic acids for P5C derivatives",
        "kpis": {
            "avg_proline_umol_l":      avg("proline_umol_l"),
            "avg_p5c_umol_l":          avg("p5c_umol_l"),
            "avg_plp_nmol_l":          avg("plp_nmol_l"),
            "pct_seizures":            pct(lambda p: p["seizures"]),
            "pct_drug_resistant":      pct(lambda p: p["drug_resistant"]),
            "pct_b6_responsive":       pct(lambda p: p["b6_responsive"]),
            "pct_idd":                 pct(lambda p: p["idd"]),
            "pct_nbs_detected":        pct(lambda p: p["nbs_detected"]),
            "pct_protein_restricted":  pct(lambda p: p["protein_restricted"]),
        },
        "plp_inactivation_mechanism": {
            "step1": "ALDH4A1 LOF → P5C cannot be oxidised to glutamate",
            "step2": "P5C (open-chain GSA form) accumulates in mitochondria and plasma",
            "step3": "P5C forms Schiff base with PLP formyl group → P5C-PLP adduct",
            "step4": "P5C-PLP adduct is catalytically inactive — functional PLP pool depleted",
            "step5": "GAD65/67 (PLP-dependent): GABA synthesis ↓ → excitation/inhibition imbalance → seizures",
            "step6": "PLP supplementation restores active PLP by mass action → partial seizure rescue",
        },
        "vs_aldh7a1_pde": {
            "shared":      "Both accumulate aldehyde metabolites that inactivate PLP via Schiff base",
            "ALDH4A1":     "P5C inactivates PLP; proline HIGH; alpha-AASA NORMAL; pipecolic NORMAL",
            "ALDH7A1_PDE": "P6C inactivates PLP; alpha-AASA HIGH (pathognomonic); pipecolic HIGH; proline NORMAL",
            "B6_response": "ALDH7A1: >85% respond; ALDH4A1: 30–50% partial response (P5C harder to deplete)",
        },
        "vs_prodh_type1": {
            "shared":   "Both: proline elevated; AR inheritance; proline catabolic pathway",
            "ALDH4A1":  "P5C ELEVATED (PATHOGNOMONIC); proline >1000; PLP low; B6 partially responsive",
            "PRODH":    "P5C NORMAL (ALDH4A1 intact converts P5C); proline 350–1000; PLP normal; B6 no effect",
            "epilepsy": "ALDH4A1: 60–80% (P5C-PLP mechanism); PRODH: 25–35% (proline NMDA interaction only)",
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
        for short_label in ["Classic-Severe", "Moderate", "Mild-Attenuated"]:
            pts = [p for p in _PATIENTS if short_label in p["phenotype"]]
            vals = [p[key] for p in pts if isinstance(p.get(key), (int, float))]
            if vals:
                out[short_label] = {
                    "min": min(vals),
                    "max": max(vals),
                    "mean": round(sum(vals) / len(vals), 1),
                    "n": len(vals),
                }
        return out

    return {
        "cohort_n":            n,
        "phenotype_dist":      pheno_dist,
        "variant_dist":        var_counts,
        "seizure_type_dist":   sz_counts,
        "clinical_rates": {
            "pct_seizures":            pct(lambda p: p["seizures"]),
            "pct_drug_resistant":      pct(lambda p: p["drug_resistant"]),
            "pct_b6_responsive":       pct(lambda p: p["b6_responsive"]),
            "pct_idd":                 pct(lambda p: p["idd"]),
            "pct_behavioral":          pct(lambda p: p["behavioral"]),
            "pct_psychiatric":         pct(lambda p: p["psychiatric"]),
            "pct_nbs_detected":        pct(lambda p: p["nbs_detected"]),
            "pct_protein_restricted":  pct(lambda p: p["protein_restricted"]),
            "pct_mma_normal":          pct(lambda p: p["mma_normal"]),
            "pct_pipecolic_normal":    pct(lambda p: p["pipecolic_normal"]),
            "pct_alpha_aasa_normal":   pct(lambda p: p["alpha_aasa_normal"]),
        },
        "biomarker_ranges": {
            "proline_umol_l": biomarker_by_pheno("proline_umol_l"),
            "p5c_umol_l":     biomarker_by_pheno("p5c_umol_l"),
            "plp_nmol_l":     biomarker_by_pheno("plp_nmol_l"),
        },
        "avg_biomarkers": {
            "proline_umol_l":     avg("proline_umol_l"),
            "p5c_umol_l":         avg("p5c_umol_l"),
            "plp_nmol_l":         avg("plp_nmol_l"),
            "homocysteine_umol_l": avg("homocysteine_umol_l"),
            "methionine_umol_l":  avg("methionine_umol_l"),
        },
        "patients": _PATIENTS,
        "key_differentials": [
            {
                "disease": "ALDH7A1 / Antiquitin Deficiency (PDE-ALDH7A1)",
                "shared":  "PLP inactivation by cyclic imine metabolite; seizures; B6 responsive",
                "distinguish": "ALDH4A1: proline ELEVATED + P5C elevated, alpha-AASA NORMAL; "
                               "ALDH7A1: alpha-AASA MARKEDLY elevated (pathognomonic), pipecolic HIGH, proline NORMAL",
            },
            {
                "disease": "PRODH Deficiency (Hyperprolinemia Type I)",
                "shared":  "Proline elevated; AR; proline catabolic pathway",
                "distinguish": "ALDH4A1: P5C elevated + PLP low + B6 responsive (partial); "
                               "PRODH: P5C NORMAL (ALDH4A1 intact), PLP normal, no B6 benefit",
            },
            {
                "disease": "PNPO Deficiency (Pyridoxamine-5-phosphate oxidase deficiency)",
                "shared":  "PLP deficiency → seizures → B6 responsive",
                "distinguish": "ALDH4A1: proline HIGH + P5C HIGH; PNPO: proline NORMAL; "
                               "PNPO: responds ONLY to PLP (not pyridoxine); ALDH4A1: responds to both",
            },
            {
                "disease": "CBS Deficiency (Classical Homocystinuria)",
                "shared":  "Seizures; metabolic disorder; AR",
                "distinguish": "ALDH4A1: proline HIGH + tHcy NORMAL; "
                               "CBS: tHcy 100–500 + methionine HIGH; proline NORMAL",
            },
        ],
        "drug_risks": [
            {
                "drug":   "Valproate (VPA)",
                "risk":   "HIGH RISK",
                "reason": "Triple mechanism: (1) mitochondrial inhibition worsens energy; "
                          "(2) PRODH inhibition in vitro → more proline → more P5C; "
                          "(3) established B6/PLP depletion drug. Avoid; supplement PLP if used.",
            },
            {
                "drug":   "B6 antagonists (INH, D-penicillamine, cycloserine)",
                "risk":   "ABSOLUTE CI",
                "reason": "Further deplete already-compromised PLP pool",
            },
            {
                "drug":   "Phenytoin",
                "risk":   "MODERATE RISK",
                "reason": "Competes at PLP-dependent AADC step; may worsen secondary B6 impairment",
            },
            {
                "drug":   "High-proline diet (collagen, gelatin, casein-rich)",
                "risk":   "HIGH RISK",
                "reason": "Increases proline flux → PRODH → more P5C → worsens PLP inactivation",
            },
            {
                "drug":   "Levetiracetam (LEV)",
                "risk":   "Level B SAFE",
                "reason": "No interaction with proline/PLP pathway; first-line AED",
            },
        ],
        "treatments": [
            {
                "treatment": "Pyridoxine (B6) / PLP trial",
                "level":     "Level B",
                "rationale": "Restores PLP by mass action despite P5C-PLP inactivation; 30–50% seizure response; "
                             "IV pyridoxine 100 mg during acute SE is mandatory trial",
            },
            {
                "treatment": "Proline-restricted diet",
                "level":     "Level B",
                "rationale": "Reduces proline → PRODH → P5C flux; low-proline proteins; avoid collagen/gelatin",
            },
            {
                "treatment": "Levetiracetam (LEV)",
                "level":     "Level B",
                "rationale": "First-line AED; no metabolic interaction with proline/PLP pathway",
            },
            {
                "treatment": "ACTH",
                "level":     "Level C",
                "rationale": "Consider for infantile spasms phenotype; limited data",
            },
            {
                "treatment": "Ketogenic diet",
                "level":     "Level C",
                "rationale": "May complement B6 partial rescue; no direct proline pathway mechanism",
            },
        ],
    }


def get_definitions() -> dict:
    return {
        "disease": "Hyperprolinemia Type II (ALDH4A1 / P5CDH Deficiency)",
        "gene_full": "ALDH4A1 — Aldehyde Dehydrogenase 4 Family Member A1 (also P5CDH: Delta-1-pyrroline-5-carboxylate dehydrogenase)",
        "omim_gene":    "OMIM *606811",
        "omim_disease": "OMIM #239510",
        "chromosome":   "1p36.13",
        "protein":      "563 aa; NAD+-dependent aldehyde dehydrogenase; homodimeric; mitochondrial matrix",
        "inheritance":  "Autosomal Recessive — biallelic loss-of-function",
        "pathway":      "Proline catabolism: Proline → [PRODH] → P5C → [ALDH4A1] → L-Glutamate → TCA/GABA",
        "biomarker_glossary": {
            "P5C":          "Delta-1-pyrroline-5-carboxylate — cyclic intermediate in proline catabolism; accumulates in Type II; PATHOGNOMONIC marker",
            "GSA":          "Glutamate-5-semialdehyde — open-chain tautomer of P5C; substrate for ALDH4A1; reacts with PLP",
            "PLP":          "Pyridoxal-5-phosphate — active form of vitamin B6; co-factor for GAD65/67, CBS, ASAT, AADC etc.; inactivated by P5C-PLP Schiff base",
            "Schiff base":  "Imine formed between aldehyde (PLP or P5C) and amino group — P5C-PLP Schiff base renders PLP catalytically inactive",
            "GAD65/GAD67":  "Glutamate Decarboxylases — PLP-dependent enzymes that synthesize GABA; impaired when PLP is inactivated by P5C → GABA deficiency → seizures",
            "PRODH":        "Proline Oxidase/Dehydrogenase — Step 1 enzyme (proline → P5C); LOF causes Hyperprolinemia Type I (less severe)",
            "alpha-AASA":   "Alpha-aminoadipic semialdehyde — PATHOGNOMONIC biomarker of ALDH7A1/PDE; NORMAL in ALDH4A1 — critical differential",
            "Pipecolic acid": "Elevated in ALDH7A1/PDE (via pipecoline-6-carboxylate/P6C pathway); NORMAL in ALDH4A1",
        },
        "key_concepts": [
            "P5C is PATHOGNOMONIC — the only IMD where P5C itself accumulates as the primary substrate",
            "PLP inactivation by P5C is the DIRECT epileptogenic mechanism — GABA synthesis collapses",
            "Mechanistically analogous to ALDH7A1/PDE (P6C inactivates PLP) but different metabolite and pathway",
            "B6/PLP supplementation PARTIALLY reverses seizures (30–50%) by mass-action PLP restoration",
            "alpha-AASA and pipecolic acid are NORMAL — the single fastest way to distinguish ALDH4A1 from ALDH7A1",
            "Proline >1000 µmol/L + P5C elevated + PLP low = diagnostic triad for Hyperprolinemia Type II",
            "VPA has TRIPLE additive hazard: avoid in ALDH4A1 whenever possible",
        ],
        "variants_glossary": {
            "p.Arg563Trp":  "ALDH active site; most common worldwide (~25%); moderate-severe; partial residual activity",
            "p.Asp779Asn":  "NAD cofactor binding pocket; ~18%; severe; enzyme cofactor assembly impaired",
            "p.Gly486Ser":  "Cofactor domain; ~15%; moderate; disrupts NAD binding geometry",
            "p.Arg440His":  "Substrate-binding groove; ~12%; moderate; GSA/P5C access impaired",
            "p.Glu425Lys":  "Catalytic glutamate (proton relay); ~10%; severe; null enzymatic activity",
            "c.IVS9+1G>A":  "Splice-null; ~10%; severe; exon 9 skipped; truncated inactive protein",
            "p.Ala519Val":  "Mild, partial function retained (~30%); ~10%; attenuated phenotype; good B6 response",
        },
        "normal_ranges": {
            "Proline (plasma)":       "< 260 µmol/L (Type II: >1000–2200+)",
            "P5C (plasma)":           "< 5 µmol/L (Type II: 15–200+)",
            "PLP (plasma)":           "35–110 nmol/L (Type II: 5–55 nmol/L — secondary deficiency)",
            "alpha-AASA (urine)":     "< 1 mmol/mol creatinine (NORMAL in Type II; HIGH in ALDH7A1)",
            "Pipecolic acid (plasma)": "< 5 µmol/L (NORMAL in Type II; HIGH in ALDH7A1)",
            "MMA (urine)":            "< 5 mmol/mol creatinine (NORMAL in Type II)",
            "Homocysteine (total)":   "< 15 µmol/L (NORMAL in Type II; borderline possible)",
        },
    }
