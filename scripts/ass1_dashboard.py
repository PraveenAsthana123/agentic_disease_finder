#!/usr/bin/env python3
"""ASS1 (Argininosuccinate Synthetase 1) Deficiency — Citrullinemia Type 1 Dashboard.

ASS1 encodes Argininosuccinate Synthetase 1, a cytoplasmic homotetrameric enzyme:
  L-Citrulline + L-Aspartate + ATP  →  L-Argininosuccinate + AMP + PPi
  UREA CYCLE STEP 3 OF 5 — condensation step; produces argininosuccinate

ASS1 DISEASE: Citrullinemia Type 1 (CTLN1) / Argininosuccinate Synthetase Deficiency
  OMIM Disease: #215700   Gene: ASS1, OMIM *603470
  Chromosome: 9q34.11
  Inheritance: Autosomal Recessive (AR)
  Protein: 412 aa; cytoplasmic homotetrameric enzyme; ubiquitous expression
  Prevalence: ~1:57,000 (second most common UCD after OTC; ~4× more common than CPS1)

MECHANISM — LOSS-OF-FUNCTION (step 3 block → citrulline CANNOT enter argininosuccinate):
  Normal ASS1: Citrulline + Aspartate + ATP → Argininosuccinate + AMP + PPi
  ASS1 LOF: Argininosuccinate CANNOT be formed → citrulline ACCUMULATES DRAMATICALLY
  Citrulline: VERY HIGH → CRITICALLY HIGH (>500 µmol/L classic; >2000 severe neonatal) — PATHOGNOMONIC
  OTC works normally → citrulline produced from ornithine + carbamoyl-P continues
  Carbamoyl-P is consumed by functional OTC → less overflow to pyrimidines → orotic NORMAL/MILD
  Ammonia: CRITICALLY ELEVATED — urea cycle blocked at step 3, NH3 cannot proceed to urea
  Arginine: CRITICALLY LOW (<10 µmol/L) — downstream product NOT made; arginine CONDITIONALLY ESSENTIAL

POSITION IN UREA CYCLE — ASS1 AS STEP 3 CONDENSATION ENZYME:
  NAGS: Glutamate + Acetyl-CoA → NAG [cofactor generator]
  Step 1: NH₃ + CO₂ + 2ATP → [CPS1, requires NAG] → Carbamoyl-P       (entry, mitochondrial)
  Step 2: Carbamoyl-P + Ornithine → [OTC] → Citrulline + Pi             (mitochondrial)
  Step 3: Citrulline + Aspartate + ATP → [ASS1] → Argininosuccinate     (cytoplasmic, BLOCKED)
  Step 4: Argininosuccinate → [ASL] → Arginine + Fumarate               (cytoplasmic)
  Step 5: Arginine → [ARG1] → Ornithine + Urea                         (cytoplasmic)

  ASS1 BLOCK CONSEQUENCES:
    Ornithine cycling INTERRUPTED — ornithine regeneration (from arginine via ARG1) stops
    Citrulline CANNOT re-enter argininosuccinate → accumulates massively
    Nitrogen STUCK as citrulline — cannot progress to urea → hyperammonemia
    Fumarate NOT produced — links to TCA cycle disrupted
    Arginine NOT produced → conditionally essential in CTLN1 (exogenous arginine required)

ASS1 BIOCHEMISTRY (LOF → citrulline critically high → step 3 block):
  Plasma citrulline:      VERY HIGH – CRITICALLY HIGH (>500 µmol/L classic; >2000 severe neonatal;
                          normal 15-35 µmol/L) — THE PATHOGNOMONIC BIOMARKER
  Plasma ammonia:         CRITICALLY ELEVATED (>500 µmol/L neonatal; >200 late-onset crisis; normal <50)
  Plasma arginine:        CRITICALLY LOW (<10 µmol/L; normal 60-120; not made downstream)
                          → CONDITIONALLY ESSENTIAL; exogenous arginine MANDATORY
  Urine orotic acid:      NORMAL to MILDLY ELEVATED (<10 µmol/mol Cr; much less than OTC)
                          OTC functions → carbamoyl-P consumed → less overflow to pyrimidines
                          KEY DISTINCTION from OTC (where orotic MARKEDLY HIGH >20)
  Plasma argininosuccinate: ABSENT (not produced — block AT THE ASS1 step)
  Plasma ornithine:       LOW-NORMAL (ornithine used by OTC but not regenerated from arginine)
  Plasma aspartate:       MILDLY ELEVATED (substrate of ASS1 cannot be consumed)
  Plasma glutamine:       ELEVATED (600-900 µmol/L; GS detoxifies ammonia)
  PLP (plasma):           NORMAL — KEY NEGATIVE vs PNPO/ALDH4A1
  alpha-AASA (urine):     NORMAL — KEY NEGATIVE vs ALDH7A1/PDE
  Pipecolic acid:         NORMAL — KEY NEGATIVE
  tHcy:                   NORMAL — KEY NEGATIVE vs CBS/MTHFR
  MMA:                    NORMAL — KEY NEGATIVE vs methylmalonic acidemia
  GABA (CSF):             NORMAL — KEY NEGATIVE vs ABAT (HIGH) and GAD1 (LOW)
  GHB (urine):            NORMAL — KEY NEGATIVE vs SSADH (ALDH5A1)
  Glycine:                NORMAL — KEY NEGATIVE vs NKH (GLDC/AMT/GCSH)
  GAA:                    NORMAL — KEY NEGATIVE vs GAMT

CITRULLINE — THE ASS1 HALLMARK BIOMARKER:
  Plasma citrulline >500 µmol/L: HIGHLY SUGGESTIVE of ASS1 deficiency (or ASL deficiency)
  Plasma citrulline >2000 µmol/L: PATHOGNOMONIC for severe classic ASS1 (neonatal)
  Citrulline <5 µmol/L → OTC, CPS1, or NAGS (OPPOSITE DIRECTION — no confusion)
  Citrulline 5-20 (borderline): further testing needed
  Citrulline 50-200 (moderate): consider partial ASS1, ASL, or other UCD
  Citrulline >500 in newborn screening: immediate action mandatory

ARGININE — CONDITIONALLY ESSENTIAL IN ASS1:
  Arginine is the end-product of the urea cycle — NOT produced in ASS1 deficiency
  Arginine becomes CONDITIONALLY ESSENTIAL: must be supplied exogenously
  High-dose arginine supplementation serves DUAL purpose:
    1. Replenishes the conditionally essential amino acid
    2. Drives ornithine regeneration via ARG1 → restores partial urea cycle cycling
    3. Reduces citrulline accumulation (competing for urea cycle intermediates)
  Arginine is NOT used to activate NAGS (unlike its role as NAGS allosteric activator)
  HIGH arginine supplementation (Level A) is PRIMARY therapy in ASS1 (unlike other UCDs)

ASS1 vs OTC — KEY DISTINCTION (citrulline direction):
  ASS1: citrulline VERY HIGH (OTC works, produces citrulline, ASS1 cannot consume it)
  OTC:  citrulline CRITICALLY LOW (OTC blocked, citrulline CANNOT be made)
  → OPPOSITE direction; no diagnostic confusion if citrulline measured
  Both: ammonia HIGH, arginine LOW
  OTC:  orotic HIGH; ASS1: orotic NORMAL/MILD (OTC functions in ASS1, less overflow)

ASS1 vs CPS1/NAGS — KEY DISTINCTION (citrulline direction):
  ASS1: citrulline VERY HIGH (OTC produces citrulline, but ASS1 cannot consume it)
  CPS1/NAGS: citrulline CRITICALLY LOW (no carbamoyl-P → no citrulline produced at all)
  → OPPOSITE direction; single citrulline level separates all three groups

ASS1 vs ASL — CLOSEST RELATIVE:
  Both have citrulline HIGH (though ASL typically lower 100-500 µmol/L vs ASS1 >500)
  Both are cytoplasmic urea cycle enzymes, AR, arginine CONDITIONALLY ESSENTIAL
  KEY DIFFERENCE: ASL → argininosuccinate ELEVATED (urine/plasma; accumulates, lyase blocked)
  ASS1: argininosuccinate ABSENT (cannot be made — ASS1 is blocked before ASL step)
  Argininosuccinate in urine = ASL, not ASS1

VPA AND ASS1:
  VPA inhibits NAGS → no NAG → CPS1 INACTIVE → carbamoyl-P not made → urea cycle DOUBLY blocked
  In ASS1: block at step 3 (ASS1); VPA adds block at step 1 (CPS1) → CATASTROPHIC
  ABSOLUTE CONTRAINDICATION in all UCD including ASS1 — multiple fatalities reported

EPILEPSY IN ASS1 DEFICIENCY (secondary to hyperammonemia):
  Overall seizure rate: ~50-60% (ammonia impairs GABA-A receptor + NMDA overactivation)
  Seizure types:
    GTCS: ~40% (modal in acute hyperammonemic crisis)
    Focal seizures: ~25% (focal cortical dysfunction)
    Myoclonic encephalopathy: ~18% (high-amplitude, multi-focal; ammonia-driven)
    Absence-like (metabolic): ~10% (metabolic encephalopathy correlate)
    Status epilepticus: ~20% (neurological emergency; neonatal worst)
  EEG: burst-suppression (neonatal severe), triphasic waves (ammonia encephalopathy),
       diffuse slowing (interictal), rhythmic bifrontal delta
  Drug-resistant epilepsy: 15-25% (correlates with hyperammonemic crisis severity)

NON-SEIZURE NEUROLOGIC FEATURES:
  Intellectual disability: 60-75% (proportional to peak ammonia and crisis duration)
  Cerebral oedema (acute): 45-55% neonatal-onset
  Behavioural/psychiatric: 20-30% (ADHD, executive dysfunction — late-onset)
  Protein aversion: 55-65% late-onset patients

NON-NEUROLOGIC FEATURES:
  Hepatomegaly: 25-40% (metabolic stress)
  Elevated transaminases: 20-35% (metabolic crisis)
  Hypercitrullinemia on NBS: ~1:57,000 detected asymptomatically
  Growth retardation: 20-35%

PHENOTYPE CLASSES:
  Classic Neonatal (50%): null ASS1; day 1-5; ammonia >1000; citrulline >2000; cerebral oedema; CRRT
  Late-Onset Partial (30%): residual 5-20% ASS1; episodic crises; protein aversion; better prognosis
  Mild/Asymptomatic NBS (20%): detected on newborn screening; citrulline 150-500; minimal disease

COMMON VARIANTS:
  p.Gly390Arg: European most common; 15%; classic severe neonatal
  p.Arg157Gln: Mediterranean; 12%; moderate; partial ASS1 activity
  p.Arg304Trp: 10%; classic severe; North American
  c.IVS1-2A>G: splice null; 8%; severe neonatal; no protein
  p.Ala118Thr: 7%; attenuated; NBS detected; late-onset
  p.Val347Met: 6%; moderate; episodic
  p.Trp179Arg: 5%; classic severe; catalytic core
  p.Arg265Cys: 4%; moderate; partial activity
"""

import random, math
from pathlib import Path

_SEED = 211   # unique seed for ASS1 cohort
_N    = 40

random.seed(_SEED)

# ──────────────────────────────────────────────────────────────────────────────
# PHENOTYPE GROUPS
# ──────────────────────────────────────────────────────────────────────────────
n_classic  = 20   # Classic Neonatal (50%) — null ASS1
n_late     = 12   # Late-Onset Partial (30%) — partial ASS1
n_mild     = 8    # Mild/Asymptomatic NBS (20%) — minimal disease

assert n_classic + n_late + n_mild == _N

PHENOTYPES = (
    ["Classic Neonatal"]      * n_classic +
    ["Late-Onset Partial"]    * n_late    +
    ["Mild/Asymptomatic NBS"] * n_mild
)
random.shuffle(PHENOTYPES)

# ──────────────────────────────────────────────────────────────────────────────
# PATIENT RECORD GENERATION
# ──────────────────────────────────────────────────────────────────────────────
def _gauss(mu, sd, lo, hi, digits=1):
    v = random.gauss(mu, sd)
    return round(max(lo, min(hi, v)), digits)

def _make_patient(i, phenotype):
    if phenotype == "Classic Neonatal":
        citrulline         = _gauss(2100, 350, 1200, 3200, 0)   # VERY HIGH — hallmark
        ammonia            = _gauss(920,  180, 600,  1600, 0)   # CRITICALLY HIGH
        arginine           = _gauss(6,    2,   2,    15,   1)   # CRITICALLY LOW
        orotic             = _gauss(4.5,  1.5, 1.5,  9.5,  1)   # NORMAL to mildly elevated
        ornithine          = _gauss(35,   8,   18,   55,   1)   # LOW-NORMAL (not regenerated)
        glutamine          = _gauss(860,  80,  650,  1100, 0)   # ELEVATED
        age_dx_d           = random.randint(1, 5)
        seizures           = random.random() < 0.78
        dre                = seizures and random.random() < 0.32
        se                 = seizures and random.random() < 0.28
        idd                = random.random() < 0.82
        ncg_response       = False   # ASS1 deficiency — NCG not specifically indicated
        peak_ammonia       = _gauss(980, 200, 600, 1600, 0)
    elif phenotype == "Late-Onset Partial":
        citrulline         = _gauss(680,  120, 400,  950,  0)
        ammonia            = _gauss(350,  90,  200,  600,  0)
        arginine           = _gauss(15,   5,   5,    30,   1)
        orotic             = _gauss(5.2,  1.8, 2.0,  10.5, 1)
        ornithine          = _gauss(42,   10,  22,   65,   1)
        glutamine          = _gauss(700,  70,  550,  900,  0)
        age_dx_d           = None
        seizures           = random.random() < 0.48
        dre                = seizures and random.random() < 0.20
        se                 = seizures and random.random() < 0.15
        idd                = random.random() < 0.58
        ncg_response       = False
        peak_ammonia       = _gauss(380, 80, 200, 600, 0)
    else:  # Mild/Asymptomatic NBS
        citrulline         = _gauss(270,  60,  150,  480,  0)
        ammonia            = _gauss(65,   20,  35,   130,  0)
        arginine           = _gauss(42,   12,  18,   70,   1)
        orotic             = _gauss(3.8,  1.2, 1.5,  7.5,  1)
        ornithine          = _gauss(52,   12,  30,   75,   1)
        glutamine          = _gauss(540,  60,  420,  680,  0)
        age_dx_d           = None
        seizures           = random.random() < 0.10
        dre                = False
        se                 = False
        idd                = random.random() < 0.15
        ncg_response       = False
        peak_ammonia       = _gauss(80, 25, 40, 150, 0)

    return {
        "id":                 i + 1,
        "phenotype":          phenotype,
        "plasma_citrulline":  citrulline,
        "plasma_ammonia":     ammonia,
        "plasma_arginine":    arginine,
        "urine_orotic_acid":  orotic,
        "plasma_ornithine":   ornithine,
        "plasma_glutamine":   glutamine,
        "plp":                _gauss(48, 8, 25, 75, 1),    # NORMAL
        "alpha_aasa":         _gauss(0.8, 0.3, 0.2, 2.0, 2),  # NORMAL
        "thcy":               _gauss(8.5, 2.0, 4.0, 15.0, 1),  # NORMAL
        "mma":                _gauss(0.12, 0.04, 0.04, 0.25, 3),  # NORMAL
        "gaba_csf":           _gauss(55, 12, 30, 90, 1),    # NORMAL
        "ghb_urine":          _gauss(1.8, 0.5, 0.8, 3.5, 2),  # NORMAL
        "age_dx_days":        age_dx_d,
        "seizures":           seizures,
        "dre":                dre,
        "status_epilepticus": se,
        "idd":                idd,
        "ncg_response":       ncg_response,
        "peak_ammonia":       peak_ammonia,
    }

PATIENTS = [_make_patient(i, PHENOTYPES[i]) for i in range(_N)]

# ──────────────────────────────────────────────────────────────────────────────
def get_overview():
    n_sz  = sum(1 for p in PATIENTS if p["seizures"])
    n_dre = sum(1 for p in PATIENTS if p["dre"])
    n_se  = sum(1 for p in PATIENTS if p["status_epilepticus"])
    n_idd = sum(1 for p in PATIENTS if p["idd"])

    def _avg(key): return round(sum(p[key] for p in PATIENTS) / _N, 1)

    # Phenotype distribution counts
    ph_counts = {}
    for p in PATIENTS:
        ph = p["phenotype"]
        ph_counts[ph] = ph_counts.get(ph, 0) + 1

    return {
        "title":       "ASS1 Deficiency (Citrullinemia Type 1 / CTLN1) — Epilepsy Dashboard",
        "subtitle":    (
            "ASS1 (Argininosuccinate Synthetase 1) catalyses urea cycle step 3: "
            "Citrulline + Aspartate + ATP → Argininosuccinate. "
            "LOF → citrulline CANNOT enter argininosuccinate → VERY HIGH citrulline (PATHOGNOMONIC). "
            "Arginine CONDITIONALLY ESSENTIAL — not produced downstream; high-dose arginine = PRIMARY therapy."
        ),
        "gene":        "ASS1",
        "chromosome":  "9q34.11",
        "omim_disease":"215700",
        "omim_gene":   "603470",
        "protein_size":"412 aa; cytoplasmic homotetrameric enzyme; ubiquitous expression",
        "cohort_n":    _N,
        "seed":        _SEED,
        "kpi": {
            "avg_plasma_citrulline_umol_l":  _avg("plasma_citrulline"),
            "avg_plasma_ammonia_umol_l":     _avg("plasma_ammonia"),
            "avg_plasma_arginine_umol_l":    _avg("plasma_arginine"),
            "avg_urine_orotic_acid_umol_mol":_avg("urine_orotic_acid"),
            "avg_plasma_ornithine_umol_l":   _avg("plasma_ornithine"),
            "avg_plasma_glutamine_umol_l":   _avg("plasma_glutamine"),
            "pct_seizures":          round(100 * n_sz  / _N),
            "pct_dre":               round(100 * n_dre / _N),
            "pct_status_epilepticus":round(100 * n_se  / _N),
            "pct_idd":               round(100 * n_idd / _N),
        },
        "phenotype_distribution": {
            ph: {"n": cnt, "pct": round(100 * cnt / _N)}
            for ph, cnt in ph_counts.items()
        },
        "key_positives": {
            "plasma_citrulline_CRITICALLY_HIGH": (
                "VERY HIGH – CRITICALLY HIGH: >500 µmol/L classic; >2000 neonatal. "
                "Normal 15-35 µmol/L. THE PATHOGNOMONIC HALLMARK BIOMARKER. "
                "Single most important test — citrulline level identifies ASS1 group immediately."
            ),
            "plasma_ammonia_CRITICALLY_HIGH": (
                "CRITICALLY ELEVATED: >500 µmol/L neonatal; >200 late-onset crisis. "
                "Urea cycle step 3 blocked → NH₃ cannot be incorporated into urea."
            ),
            "plasma_arginine_CRITICALLY_LOW": (
                "CRITICALLY LOW (<10 µmol/L neonatal; <20 late-onset). "
                "Arginine is the urea cycle end-product — NOT produced in ASS1 deficiency. "
                "Arginine becomes CONDITIONALLY ESSENTIAL; mandatory supplementation."
            ),
        },
        "biomarker_normals": {
            "urine_orotic_acid_NORMAL_to_MILD": (
                "NORMAL to MILDLY ELEVATED (<10 µmol/mol Cr). "
                "OTC functions in ASS1 → carbamoyl-P consumed → less overflow to pyrimidines. "
                "KEY DISTINCTION from OTC where orotic MARKEDLY HIGH (>20 µmol/mol Cr). "
                "Mild elevation possible but far less than OTC."
            ),
            "plp_NORMAL":         "NORMAL — ASS1 NOT PLP-dependent (KEY NEG vs PNPO/ALDH4A1)",
            "alpha_aasa_NORMAL":  "NORMAL — KEY NEGATIVE vs ALDH7A1/PDE",
            "pipecolic_NORMAL":   "NORMAL — KEY NEGATIVE vs PDE/peroxisomal disorders",
            "thcy_NORMAL":        "NORMAL — KEY NEGATIVE vs CBS/MTHFR/MTR",
            "mma_NORMAL":         "NORMAL — KEY NEGATIVE vs methylmalonic acidemia",
            "gaba_csf_NORMAL":    "NORMAL — KEY NEGATIVE vs ABAT (HIGH) and GAD1 (LOW)",
            "ghb_NORMAL":         "NORMAL — KEY NEGATIVE vs SSADH (ALDH5A1) where GHB dramatically HIGH",
            "glycine_NORMAL":     "NORMAL — KEY NEGATIVE vs NKH (GLDC/AMT/GCSH)",
            "gaa_NORMAL":         "NORMAL — KEY NEGATIVE vs GAMT",
            "argininosuccinate_ABSENT": (
                "ABSENT in plasma/urine — block is AT ASS1 step, so argininosuccinate CANNOT BE MADE. "
                "Contrast with ASL deficiency where argininosuccinate is MARKEDLY ELEVATED (key differentiator)."
            ),
        },
        "pathway_context": {
            "urea_cycle_position": "Step 3 of 5 (cytoplasmic condensation step)",
            "block_consequence":   "Citrulline cannot enter argininosuccinate → accumulates massively",
            "arginine_axis":       (
                "Arginine is ASS1 step 4-5 downstream product — NOT produced in deficiency. "
                "Arginine supplementation is PRIMARY therapy: replenishes conditionally essential AA, "
                "restores partial ornithine recycling via ARG1, and reduces citrulline accumulation."
            ),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
def get_breakdown():
    n_sz     = sum(1 for p in PATIENTS if p["seizures"])

    # Seizure types (of those with seizures, ≥1)
    if n_sz > 0:
        seizure_types = {
            "GTCS (acute hyperammonemic crisis)": {
                "n": round(0.40 * n_sz), "pct": 40,
                "detail": "Modal seizure type; acute onset with hyperammonemic crisis; tonic-clonic"
            },
            "Focal Seizures": {
                "n": round(0.25 * n_sz), "pct": 25,
                "detail": "Frontal-predominant focal cortical dysfunction from ammonia encephalopathy"
            },
            "Myoclonic Encephalopathy": {
                "n": round(0.18 * n_sz), "pct": 18,
                "detail": "High-amplitude multifocal myoclonus; ammonia-driven; neonatal especially"
            },
            "Absence-like (metabolic)": {
                "n": round(0.10 * n_sz), "pct": 10,
                "detail": "Metabolic encephalopathy correlate; responds to ammonia control"
            },
            "Status Epilepticus": {
                "n": round(0.20 * n_sz), "pct": 20,
                "detail": "Neurological emergency; classic neonatal worst risk; CRRT mandatory"
            },
        }
    else:
        seizure_types = {}

    triggers = {
        "Protein/Nitrogen Load": {
            "pct": 78, "detail": "Dietary protein excess → ammonia surge; ASS1 cannot process increased urea flux"
        },
        "Febrile Illness (Catabolic)": {
            "pct": 65, "detail": "Tissue catabolism releases nitrogen → ammonia rise in partial ASS1"
        },
        "Fasting / Starvation": {
            "pct": 52, "detail": "Protein catabolism; anti-catabolic measures mandatory during illness"
        },
        "Valproate Exposure": {
            "pct": 20, "detail": "ABSOLUTE CI — VPA inhibits NAGS → CPS1 off → second block added"
        },
        "Post-Surgical Stress": {
            "pct": 35, "detail": "Catabolism + anaesthetic agents; protein-free IV fluids mandatory"
        },
        "Childbirth / Postpartum": {
            "pct": 15, "detail": "Catabolic state; unmasking late-onset disease in females"
        },
    }

    treatments = {
        "Arginine (High-Dose, Oral/IV)": {
            "level": "A",
            "category": "PRIMARY — Conditionally Essential Amino Acid + Urea Cycle Driver",
            "mechanism": (
                "Arginine is NOT produced in ASS1 deficiency → CONDITIONALLY ESSENTIAL. "
                "High-dose arginine (200-800 mg/kg/day oral; 250-600 mg/kg/day IV acute): "
                "replenishes essential AA, restores ornithine cycling via ARG1, reduces citrulline. "
                "Different from other UCDs — arginine supplementation is PRIMARY, not just supportive."
            ),
            "efficacy_pct": 88,
        },
        "Sodium Benzoate + Phenylacetate (RAVICTI)": {
            "level": "A",
            "category": "Nitrogen Scavenging — Alternative Pathway",
            "mechanism": "Benzoate conjugates glycine → hippurate; phenylacetate conjugates glutamine → phenylacetylglutamine. Each excretes 2N. Standard UCD scavenger, Level A chronic + acute.",
            "efficacy_pct": 80,
        },
        "Low-Protein Diet (Protein Restriction)": {
            "level": "A",
            "category": "Dietary — Reduces Nitrogen Load",
            "mechanism": "Essential protein intake only; UCD formula supplementation for EAA; reduces ammonia substrate load.",
            "efficacy_pct": 72,
        },
        "Citrulline Restriction": {
            "level": "A",
            "category": "Dietary — Reduces Citrulline Substrate Load",
            "mechanism": "Limit dietary citrulline and its precursors; reduces accumulation at the ASS1 block.",
            "efficacy_pct": 65,
        },
        "IV Dextrose (10%, GIR 8-12)": {
            "level": "A",
            "category": "Acute Crisis — Anti-Catabolic",
            "mechanism": "Glucose drives protein anabolism, halts catabolism; provides calories without nitrogen.",
            "efficacy_pct": 75,
        },
        "CRRT / Haemodialysis": {
            "level": "A",
            "category": "Acute Crisis — Ammonia Removal",
            "mechanism": "For NH₃ >500 µmol/L; CRRT preferred in neonates; rapid ammonia clearance; mandatory in neonatal classic.",
            "efficacy_pct": 90,
        },
        "Liver Transplant": {
            "level": "A",
            "category": "Definitive Cure — Hepatic ASS1 Restored",
            "mechanism": "Orthotopic or auxiliary liver transplant restores full hepatic ASS1 activity. Curative for hyperammonemia; citrulline normalises. Arginine restriction can be relaxed post-transplant.",
            "efficacy_pct": 96,
        },
        "Levetiracetam (LEV)": {
            "level": "B",
            "category": "AED — First-Line (Seizures Secondary to Hyperammonemia)",
            "mechanism": "SV2A modulation; no hepatotoxicity; safe in all UCD; preferred first-line AED when seizures require treatment beyond ammonia control.",
            "efficacy_pct": 60,
        },
    }

    drug_risks = {
        "Valproate / Valproic Acid (VPA)": {
            "risk": "ABSOLUTE CI",
            "detail": (
                "VPA inhibits NAGS → no NAG → CPS1 inactive → urea cycle DOUBLY blocked (step 1 + step 3). "
                "In ASS1 deficiency: existing block at step 3; VPA adds complete block at step 1. "
                "CATASTROPHIC hyperammonemia. Multiple fatalities reported. NEVER use."
            ),
        },
        "Valpromide / Valproyl-Glycine": {
            "risk": "ABSOLUTE CI",
            "detail": "VPA prodrugs — converted to VPA in vivo. Same NAGS-inhibition mechanism. ABSOLUTE CI.",
        },
        "High-Protein Diet": {
            "risk": "ABSOLUTE CI",
            "detail": "Nitrogen overload; ASS1 blocked at step 3; ammonia cannot proceed to urea. Crisis risk.",
        },
        "L-Asparaginase": {
            "risk": "ABSOLUTE CI",
            "detail": "Causes hyperammonemia via asparagine depletion → excess ammonia → crisis on ASS1 background.",
        },
        "Systemic Glucocorticoids (High-Dose)": {
            "risk": "HIGH RISK",
            "detail": "Catabolic — increases protein catabolism → nitrogen load → ammonia rise. Avoid; if necessary, cover with protein-free glucose.",
        },
        "N2O (Nitrous Oxide)": {
            "risk": "HIGH RISK",
            "detail": "Inactivates methionine synthase → disrupts folate/methionine cycle; also direct nitrogen concern. Avoid in anaesthesia.",
        },
        "Prolonged Fasting": {
            "risk": "HIGH RISK",
            "detail": "Starvation catabolism → protein breakdown → ammonia surge. Anti-catabolic glucose infusion mandatory during illness/procedures.",
        },
        "Carbamazepine / Phenytoin": {
            "risk": "MODERATE RISK",
            "detail": "Hepatic enzyme induction; higher risk with impaired hepatic function. LEV preferred.",
        },
    }

    # Variant distribution
    variants = [
        {"name": "p.Gly390Arg",   "pct": 15, "domain": "Catalytic core (G390 conserved across ASS family)",     "severity": "Classic severe neonatal; null-like activity", "ncg_analogue": False},
        {"name": "p.Arg157Gln",   "pct": 12, "domain": "Aspartate-binding region (R157 contacts aspartate)",    "severity": "Moderate; partial ASS1 activity; late-onset",  "ncg_analogue": False},
        {"name": "p.Arg304Trp",   "pct": 10, "domain": "Catalytic site (R304 in ATP-binding region)",           "severity": "Classic severe; North American founder",        "ncg_analogue": False},
        {"name": "c.IVS1-2A>G",   "pct":  8, "domain": "Splice-site null (no full-length transcript)",          "severity": "Null; severe neonatal; no residual protein",    "ncg_analogue": False},
        {"name": "p.Ala118Thr",   "pct":  7, "domain": "N-terminal region (structural; distant from active)",   "severity": "Attenuated; NBS-detected; mild-late-onset",     "ncg_analogue": True},
        {"name": "p.Val347Met",   "pct":  6, "domain": "Tetramer interface (V347 contacts adjacent subunit)",   "severity": "Moderate; episodic; partial tetramer assembly", "ncg_analogue": True},
        {"name": "p.Trp179Arg",   "pct":  5, "domain": "Catalytic core (W179 essential for substrate binding)", "severity": "Classic severe; catalytic null",                "ncg_analogue": False},
        {"name": "p.Arg265Cys",   "pct":  4, "domain": "Citrulline-binding domain (R265 contacts citrulline)", "severity": "Moderate; partial citrulline binding retained",  "ncg_analogue": True},
    ]

    differentials = [
        {
            "disease": "OTC deficiency",
            "key_diff": (
                "Citrulline CRITICALLY LOW in OTC (<5 µmol/L) vs VERY HIGH in ASS1 (>500). "
                "OPPOSITE direction — single citrulline measurement distinguishes. "
                "Orotic HIGH in OTC (OTC blocked → carbamoyl-P overflows); NORMAL in ASS1 (OTC works)."
            ),
            "distinguishing": "Plasma citrulline direction + urine orotic acid",
        },
        {
            "disease": "CPS1 / NAGS deficiency",
            "key_diff": (
                "Citrulline CRITICALLY LOW in CPS1/NAGS (<5 µmol/L) — no citrulline produced at all. "
                "ASS1: citrulline VERY HIGH. Orotic NORMAL in both (CPS1/NAGS same as ASS1 in this regard). "
                "OPPOSITE citrulline direction; no confusion."
            ),
            "distinguishing": "Plasma citrulline direction",
        },
        {
            "disease": "ASL deficiency (Argininosuccinic Aciduria)",
            "key_diff": (
                "CLOSEST RELATIVE — both have citrulline HIGH; both AR; both arginine low. "
                "KEY DIFFERENCE: ASL → argininosuccinate VERY HIGH in urine/plasma (ASL enzyme cannot cleave it). "
                "ASS1 → argininosuccinate ABSENT (cannot be made; ASS1 blocks BEFORE ASL step). "
                "Argininosuccinate in urine = ASL, NOT ASS1."
            ),
            "distinguishing": "Urine/plasma argininosuccinate (HIGH in ASL, ABSENT in ASS1)",
        },
        {
            "disease": "Citrullinemia Type 2 (SLC25A13 / Citrin deficiency)",
            "key_diff": (
                "Citrin (CTLN2): adult-onset Japanese predominantly; citrulline elevated but typically 100-500 (lower than classic ASS1). "
                "Associated with hyperlipidaemia, fatty liver, recurrent pancreatitis. SLC25A13 variants; Japanese founder. "
                "Neonatal intrahepatic cholestasis (NICCD) in infancy. Gene panel distinguishes."
            ),
            "distinguishing": "Gene testing (ASS1 vs SLC25A13), ethnicity, adult-onset pattern, lipids",
        },
        {
            "disease": "HHH Syndrome (SLC25A15)",
            "key_diff": (
                "HHH: homocitrullinuria pathognomonic; ornithine VERY HIGH (ornithine cannot enter mitochondria); "
                "citrulline mildly elevated or normal (different mechanism). "
                "Orotic acid HIGH in HHH (similar to OTC). "
                "Spastic paraplegia prominent in HHH."
            ),
            "distinguishing": "Urine homocitrullinuria + ornithine (VERY HIGH in HHH, LOW-NORMAL in ASS1)",
        },
        {
            "disease": "GLUD1 GoF (HHS Syndrome)",
            "key_diff": (
                "GLUD1: hyperinsulinism + hyperammonemia (HHS). "
                "Citrulline NORMAL (no urea cycle enzyme block); ammonia 100-500 not >1000. "
                "Glucose LOW (hypoglycaemia) — characteristic feature absent in ASS1."
            ),
            "distinguishing": "Citrulline (NORMAL in GLUD1, VERY HIGH in ASS1), glucose, insulin",
        },
    ]

    # Phenotype detail
    ph_detail = {}
    for ph, (n, pct, desc, avg_nh3, avg_cit, sz_pct) in [
        ("Classic Neonatal",      (n_classic, 50,
            "Null/near-null ASS1 variants. Day 1-5 neonatal presentation with ammonia >1000 µmol/L, "
            "citrulline >2000 µmol/L, cerebral oedema, burst-suppression EEG. "
            "CRRT mandatory. Liver transplant curative if available. "
            "Highest mortality without immediate intervention.",
            980, 2100, 78)),
        ("Late-Onset Partial",    (n_late,  30,
            "Residual 5-20% ASS1 activity; episodic hyperammonemic crises triggered by "
            "illness, protein excess, or catabolic stress. Protein aversion characteristic. "
            "May present in childhood or adulthood. Arginine + nitrogen scavengers; diet restriction; "
            "liver transplant if inadequately controlled.",
            380, 680, 48)),
        ("Mild/Asymptomatic NBS", (n_mild,  20,
            "Detected on newborn screening; citrulline 150-500 µmol/L. "
            "Clinically silent or minimal symptoms. Residual ASS1 function substantial. "
            "Some may remain asymptomatic; monitoring essential as stress can unmask disease.",
            80, 270, 10)),
    ]:
        ph_detail[ph] = {
            "n": n, "pct": pct,
            "description": desc,
            "avg_peak_ammonia_umol_l": avg_nh3,
            "avg_plasma_citrulline_umol_l": avg_cit,
            "seizure_rate_pct": sz_pct,
        }

    return {
        "seizure_types":   seizure_types,
        "triggers":        triggers,
        "treatments":      treatments,
        "drug_risks":      drug_risks,
        "variants":        variants,
        "differentials":   differentials,
        "phenotype_detail":ph_detail,
    }


# ──────────────────────────────────────────────────────────────────────────────
def get_definitions():
    return {
        "disease": "ASS1 Deficiency — Citrullinemia Type 1 (CTLN1)",
        "omim_disease": "215700",
        "omim_gene":    "603470",
        "gene":         "ASS1",
        "chromosome":   "9q34.11",
        "inheritance":  "Autosomal Recessive (AR); biallelic LOF required; males and females equally affected",
        "prevalence":   (
            "~1:57,000; second most common UCD after OTC (~4× more common than CPS1); "
            "detected on newborn screening programmes worldwide. "
            "Mild/asymptomatic cases may represent up to 20% of detected cases."
        ),
        "enzyme":       (
            "Argininosuccinate Synthetase 1 (ASS1); 412 aa; cytoplasmic homotetrameric enzyme; "
            "ubiquitous expression (liver, kidney, brain, fibroblasts); "
            "condensation enzyme — forms argininosuccinate from citrulline + aspartate."
        ),
        "reaction":     (
            "L-Citrulline + L-Aspartate + ATP  →  L-Argininosuccinate + AMP + PPi  [Step 3 of urea cycle, cytoplasmic]"
        ),
        "pathway_role": (
            "ASS1 catalyses STEP 3 of the urea cycle — the cytoplasmic condensation step. "
            "Citrulline (exported from mitochondria after OTC step 2) combines with aspartate to form argininosuccinate. "
            "Argininosuccinate is then cleaved by ASL (step 4) to arginine + fumarate. "
            "Fumarate re-enters the TCA cycle (linking urea cycle to energy metabolism). "
            "Arginine is the urea cycle end-product (step 5 via ARG1); CONDITIONALLY ESSENTIAL in ASS1 deficiency."
        ),
        "mechanism_of_disease": (
            "ASS1 LOF → argininosuccinate CANNOT be formed from citrulline + aspartate. "
            "OTC (step 2) and CPS1 (step 1) continue to function → citrulline is continuously produced. "
            "Citrulline cannot exit the block → ACCUMULATES DRAMATICALLY (>1000 µmol/L neonatal, >500 late-onset). "
            "Urea cycle arrested at step 3 → NH₃ incorporated into citrulline but cannot proceed to urea. "
            "Ammonia CRITICALLY ELEVATED. Arginine downstream product NOT synthesised → CONDITIONALLY ESSENTIAL. "
            "Ornithine cycling broken (ornithine not regenerated from arginine via ARG1) → metabolic cycle collapse."
        ),
        "arginine_conditional_essential": (
            "Arginine is the end-product of the urea cycle (step 5, via ARG1: Arginine → Ornithine + Urea). "
            "In ASS1 deficiency, arginine cannot be produced endogenously (block at step 3). "
            "Arginine therefore becomes CONDITIONALLY ESSENTIAL (normally non-essential in healthy individuals). "
            "High-dose arginine supplementation is PRIMARY therapy in ASS1: "
            "  1. Replenishes the conditionally essential amino acid "
            "  2. Allows ARG1 to regenerate ornithine → partial urea cycle restoration "
            "  3. Reduces citrulline build-up via competitive dynamics "
            "Unlike NAGS where arginine activates NAGS allosterically, in ASS1 the mechanism is simply: "
            "provide what cannot be made downstream."
        ),
        "citrulline_pathognomonic": (
            "Plasma citrulline is THE pathognomonic biomarker for ASS1 deficiency. "
            "In classic neonatal: >2000 µmol/L (normal 15-35); in late-onset: 500-1000 µmol/L. "
            "Citrulline <5 µmol/L = OTC/CPS1/NAGS (OPPOSITE extreme — no confusion). "
            "Citrulline >500 = ASS1 or ASL deficiency (argininosuccinate distinguishes). "
            "Newborn screening citrulline level is the primary trigger for urgent metabolic evaluation."
        ),
        "asl_distinction": (
            "ASL (argininosuccinic aciduria) is the CLOSEST biochemical relative of ASS1 deficiency. "
            "Both: citrulline elevated, arginine low, AR inheritance, arginine conditionally essential. "
            "KEY DIFFERENCE: "
            "  ASL deficiency → argininosuccinate VERY HIGH in plasma/urine (ASL cannot cleave it). "
            "  ASS1 deficiency → argininosuccinate ABSENT (cannot be synthesised — block is BEFORE ASL step). "
            "Urine organic acids + plasma amino acids together identify which enzyme is affected. "
            "Gene panel mandatory (ASS1 + ASL sequenced together)."
        ),
        "vpa_mechanism": (
            "VPA inhibits NAGS (competitive inhibitor of NAGS active site) → no NAG → CPS1 INACTIVE (step 1 block). "
            "In ASS1 deficiency: existing block at step 3. VPA adds complete block at step 1. "
            "Result: DOUBLY BLOCKED urea cycle — NO carbamoyl-P made (step 1) AND no argininosuccinate made (step 3). "
            "Catastrophic hyperammonemia. Multiple fatalities. ABSOLUTE CI in ALL UCD."
        ),
        "key_biomarker_positives": {
            "plasma_citrulline":   "VERY HIGH–CRITICALLY HIGH (>500 µmol/L classic; >2000 neonatal; normal 15-35) — PATHOGNOMONIC HALLMARK",
            "plasma_ammonia":      "CRITICALLY ELEVATED (>500 µmol/L neonatal; >200 late-onset crisis; normal <50 µmol/L)",
            "plasma_arginine":     "CRITICALLY LOW (<10 µmol/L neonatal; <20 late-onset; normal 60-120) — CONDITIONALLY ESSENTIAL",
            "plasma_aspartate":    "MILDLY ELEVATED — ASS1 substrate, cannot be consumed at step 3",
            "plasma_glutamine":    "ELEVATED (600-900 µmol/L) — GS detoxifies ammonia → glutamine",
        },
        "key_biomarker_negatives": {
            "urine_orotic_acid":      "NORMAL to MILDLY ELEVATED (<10 µmol/mol Cr; OTC functions → less carbamoyl-P overflow; much less than OTC)",
            "argininosuccinate":      "ABSENT — cannot be synthesised (block AT ASS1 step); contrast with ASL where argininosuccinate VERY HIGH",
            "plp":                    "NORMAL — ASS1 not PLP-dependent (KEY NEG vs PNPO/ALDH4A1/OAT)",
            "alpha_aasa":             "NORMAL — KEY NEGATIVE vs ALDH7A1/PDE",
            "pipecolic":              "NORMAL — KEY NEGATIVE vs PDE/peroxisomal",
            "thcy":                   "NORMAL — KEY NEGATIVE vs CBS/MTHFR/MTR",
            "mma":                    "NORMAL — KEY NEGATIVE vs methylmalonic acidemia",
            "gaba_csf":               "NORMAL — KEY NEGATIVE vs ABAT (HIGH) and GAD1 (LOW)",
            "ghb_urine":              "NORMAL — KEY NEGATIVE vs SSADH (ALDH5A1)",
            "glycine":                "NORMAL — KEY NEGATIVE vs NKH (GLDC/AMT/GCSH)",
            "gaa":                    "NORMAL — KEY NEGATIVE vs GAMT",
            "homocitrullinuria":      "NORMAL — KEY NEGATIVE vs HHH syndrome (SLC25A15)",
        },
        "critical_ass1_vs_otc_distinction": (
            "Citrulline VERY HIGH in ASS1 (>500 µmol/L; OTC functions, produces citrulline) "
            "vs CRITICALLY LOW in OTC (<5 µmol/L; OTC blocked, citrulline CANNOT be made). "
            "OPPOSITE DIRECTION — single citrulline level is a complete differentiator. "
            "Orotic acid: HIGH in OTC (carbamoyl-P overflows); NORMAL-MILD in ASS1 (carbamoyl-P consumed by OTC)."
        ),
        "critical_ass1_vs_cps1_nags_distinction": (
            "Citrulline VERY HIGH in ASS1 vs CRITICALLY LOW in CPS1/NAGS — OPPOSITE extremes. "
            "In CPS1/NAGS: no carbamoyl-P → no citrulline produced → citrulline <5 µmol/L. "
            "In ASS1: OTC produces citrulline normally → citrulline cannot proceed → accumulates >500 µmol/L. "
            "Single citrulline measurement separates ASS1 from all proximal UCD (CPS1/NAGS/OTC)."
        ),
        "absolute_contraindications": {
            "valproate_vpa": (
                "ABSOLUTE CI — VPA inhibits NAGS → CPS1 INACTIVE → urea cycle DOUBLY blocked (step 1 + step 3). "
                "Catastrophic hyperammonemia. Multiple fatalities. NEVER use in any UCD."
            ),
            "vpa_prodrugs": "Valpromide, valproyl-glycine — ABSOLUTE CI. Converted to VPA in vivo → same mechanism.",
            "high_protein_diet": "ABSOLUTE CI — excess nitrogen; ASS1 cannot process into urea; ammonia crisis.",
            "l_asparaginase": "ABSOLUTE CI — induces hyperammonemia; catastrophic on ASS1 background.",
        },
        "first_line_treatments": {
            "arginine_high_dose":        "Arginine 200-800 mg/kg/day oral; 250-600 IV acute — Level A PRIMARY; CONDITIONALLY ESSENTIAL in ASS1",
            "nitrogen_scavengers":       "Sodium benzoate + phenylacetate (RAVICTI) — Level A chronic + acute",
            "low_protein_diet":          "Essential protein restriction; UCD formula — Level A",
            "citrulline_restriction":    "Reduce dietary citrulline load — Level A",
            "iv_dextrose_anti_catabolic":"IV dextrose 10% (GIR 8-12) — Level A acute crisis anti-catabolism",
            "crrt_hd":                   "CRRT/HD for NH₃ >500 µmol/L — Level A; CRRT preferred in neonates",
            "liver_transplant":          "Orthotopic liver transplant — Level A CURATIVE; citrulline normalises; arginine restriction relaxed",
            "aed":                       "Levetiracetam (LEV) — Level B first-line AED (no hepatotoxicity)",
        },
        "seizure_mechanism": (
            "Ammonia impairs GABA-A receptor gating (reduces inhibitory tone) and over-activates NMDA receptors. "
            "Net: reduced inhibition + increased excitation → seizure threshold lowered. "
            "Cerebral oedema (cytotoxic + vasogenic) → intracranial hypertension → secondary seizures. "
            "EEG: burst-suppression (severe neonatal), triphasic waves (ammonia encephalopathy), diffuse slowing. "
            "Seizure treatment: control ammonia FIRST (arginine + scavengers + CRRT); "
            "AED (LEV) for persistent seizures. NEVER VPA."
        ),
        "ar_inheritance_note": (
            "AR inheritance: BOTH parents obligate heterozygous carriers; recurrence risk 25% per pregnancy. "
            "Males and females equally affected — contrast with OTC (X-linked; males predominantly). "
            "Heterozygous ASS1 carriers: typically asymptomatic. "
            "Cascade carrier testing after index diagnosis. Living-related liver donation possible. "
            "NBS (newborn screening) citrulline elevation triggers urgent genetic/metabolic evaluation."
        ),
        "unique_features_vs_other_ucd": (
            "1. CITRULLINE CRITICALLY HIGH — the pathognomonic hallmark (opposite of CPS1/NAGS/OTC where citrulline LOW). "
            "2. ARGININE is CONDITIONALLY ESSENTIAL — high-dose arginine is PRIMARY therapy (unique among proximal UCDs). "
            "3. CYTOPLASMIC enzyme — only ASS1 and ASL are cytoplasmic; CPS1/OTC/NAGS are mitochondrial. "
            "4. ASL DISTINCTION: argininosuccinate absent in ASS1, very high in ASL — single metabolite separates them. "
            "5. OROTIC ACID NORMAL/MILD — OTC is intact; less carbamoyl-P overflow than OTC deficiency. "
            "6. SECOND MOST COMMON UCD — ~1:57,000; large NBS-detected population; many asymptomatic/mild cases."
        ),
    }
