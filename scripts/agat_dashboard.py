#!/usr/bin/env python3
"""AGAT Deficiency (L-arginine:glycine amidinotransferase Deficiency) Epilepsy Dashboard.

GATM encodes L-arginine:glycine amidinotransferase (AGAT), expressed predominantly in
the kidney (proximal tubule) and pancreas. AGAT catalyses the FIRST and rate-limiting step
of creatine biosynthesis.

CREATINE BIOSYNTHESIS PATHWAY (two-step):

  Step 1 — AGAT (kidney/pancreas):    [← AGAT DEFICIENCY BLOCKS HERE]
    L-Arginine + Glycine → L-Ornithine + Guanidinoacetate (GAA)

  Step 2 — GAMT (liver/pancreas):
    SAM + Guanidinoacetate (GAA) → SAH + Creatine

  Step 3 — SLC6A8 (ubiquitous):
    Creatine transported into muscle & brain via Na+/Cl- cotransporter

AGAT IS THE FIRST COMMITTED STEP OF CREATINE SYNTHESIS:
  - AGAT performs transamidination: transfers the amidino group from arginine to glycine
  - This is NOT a SAM-dependent reaction (unlike GAMT/GNMT) — no methyl donor required
  - The reaction produces guanidinoacetate (GAA) and ornithine simultaneously
  - Without AGAT activity, no GAA is made → no creatine can be synthesised (GAMT has no substrate)
  - Kidney proximal tubule is the primary site of GAA production in humans

AGAT LOF — SINGLE PATHOLOGY (unlike GAMT's dual pathology):

  AGAT CANNOT MAKE GAA:
    L-Arg + Gly → × AGAT BLOCKED → GAA = ABSENT / VERY LOW
    ↓
    No GAA substrate → GAMT cannot make creatine (blocked indirectly)
    ↓
    Creatine = ABSENT / VERY LOW (<5 µmol/L; normal 20–80)
    ↓
    Brain phosphocreatine buffer = ZERO → neuronal energy deficiency → seizures
    ↓
    Brain H-MRS: creatine peak (3.0 ppm) ABSENT — pathognomonic (same as GAMT/SLC6A8)

CRITICAL DIFFERENCE FROM GAMT — GAA IS LOW, NOT HIGH:
  GAMT deficiency: AGAT still active → GAA massively ELEVATED (50–300 µmol/L)
    → GAA is a potent convulsant (GABA-A inhibition + NMDA activation) → drug-resistant epilepsy
  AGAT deficiency: AGAT absent → GAA NOT PRODUCED → GAA VERY LOW / ABSENT (<0.5 µmol/L)
    → No GAA neurotoxicity → seizures are LESS SEVERE and LESS DRUG-RESISTANT than GAMT
    → Seizure mechanism: purely creatine energy deficit (not GAA epileptogenicity)
    → Better AED response than GAMT; metabolic treatment (creatine) alone often sufficient

METABOLIC SIGNATURE OF AGAT DEFICIENCY:
  Guanidinoacetate (GAA): VERY LOW / ABSENT (<0.5 µmol/L; normal 0.5–3) — PATHOGNOMONIC
  Creatine (plasma):      VERY LOW or ABSENT (<5 µmol/L; normal 20–80)
  Creatinine (plasma/urine): LOW (creatinine is creatine catabolite — source absent)
  Methionine:             NORMAL (SAM pathway not involved — unlike MAT1A/GNMT/AHCY/CBS)
  tHcy:                   NORMAL (remethylation pathway intact)
  SAM:                    NORMAL (AGAT reaction does NOT use SAM — unlike GAMT)
  MMA:                    NORMAL (propionate arm intact)
  MeCbl/AdoCbl:           NORMAL (cobalamin system intact)
  Arginine:               NORMAL or mildly elevated (AGAT no longer consuming arginine)
  Ornithine:              NORMAL or slightly LOW-NORMAL
                          (AGAT makes ornithine as a co-product; other sources [urea cycle] intact)
  Brain H-MRS:            Creatine peak (3.0 ppm) ABSENT — PATHOGNOMONIC (same as GAMT + SLC6A8)

CLINICAL HALLMARKS (milder than GAMT due to absent GAA neurotoxicity):
  (1) Guanidinoacetate (GAA) VERY LOW or absent — pathognomonic; opposite of GAMT
  (2) Creatine absent on brain H-MRS — same as GAMT and SLC6A8 (shared CCDS feature)
  (3) Intellectual disability (100%): moderate-profound; same creatine deficit as GAMT
  (4) Seizures (~50–60%): less frequent and less drug-resistant than GAMT (no GAA epileptogenicity)
  (5) Low urine creatinine: same as GAMT (creatine absent → no catabolite)
  (6) Methionine NORMAL: distinguishes from entire hypermethioninemia spectrum

AGAT vs GAMT — THE TWO CREATINE BIOSYNTHESIS ENZYME DEFECTS:
  Feature               AGAT deficiency              GAMT deficiency
  ─────────────────────────────────────────────────────────────────────
  GAA level             VERY LOW (<0.5 µmol/L)       MASSIVELY HIGH (50–300 µmol/L)
  GAA direction         ↓ (source blocked)           ↑↑↑ (clearance blocked)
  Creatine              Absent                       Absent
  Brain MRS             Creatine absent              Creatine absent
  Drug-resistant Sz     25–35% (milder)              60–80% (severe)
  GAA neurotoxicity     ABSENT (no GAA)              PRESENT (GABA-A + NMDA)
  Treatment             Creatine ONLY (Level A)      Creatine + Ornithine (both Level A)
  Ornithine needed?     NO                           YES (reduces GAA)
  Arginine restriction  NOT indicated                Level B (reduces AGAT substrate)
  NBS biomarker         Low GAA + low creatinine     High GAA (easier to detect)
  Severity              Moderate-severe              Severe (dual pathology)
  Prevalence            ~150 cases (rarest CCDS)     ~200–250 cases (most common CCDS)

INHERITANCE AND GENE:
  Gene: GATM (encodes AGAT enzyme; disease acronym = AGAT deficiency)
  Chromosome: 15q21.1
  Protein: 423 amino acids (includes N-terminal mitochondrial targeting sequence ~30 aa;
           mature protein ~393 aa after mitochondrial processing)
  Autosomal Recessive — biallelic LOF required
  ~150 patients reported worldwide (2026) — rarest of the three CCDS
  Enzyme localisation: Mitochondria of kidney proximal tubule + pancreas

OMIM:
  Gene:    *602360 (GATM)
  Disease: #612718 (Cerebral creatine deficiency syndrome 3, CCDS3 / AGAT deficiency)
"""

import random
import math

_SEED = 117
_rng = random.Random(_SEED)


# ---------------------------------------------------------------------------
# Phenotypic classes
# ---------------------------------------------------------------------------
PHENOTYPE_CLASSES = [
    "Classic (moderate-profound IDD + seizures in majority / GAA very low-absent / MODAL)",
    "Moderate (moderate IDD + mild seizures or none / GAA very low / creatine very low)",
    "Mild (mild cognitive delay + rare or no seizures / GAA detectable but low)",
]

PHENO_WEIGHTS = [0.55, 0.35, 0.10]  # 55% classic / 35% moderate / 10% mild

VARIANTS = [
    "p.Arg149Trp",      # active site adjacent; most common; severe; ~20%
    "p.Val166Gly",      # catalytic pocket; severe; ~18%
    "p.Ser119Leu",      # substrate-binding region; moderate; ~15%
    "p.Thr133Ile",      # moderate; ~12%
    "p.Glu93Lys",       # surface exposed; moderate; ~10%
    "p.Gln236Lys",      # distal active site; moderate-mild; ~10%
    "c.IVS2+1G>A",     # splice-null; severe; ~8%
    "p.Leu45Pro",       # N-terminal mitochondrial signal; severe processing defect; ~7%
]

VARIANT_WEIGHTS = [0.20, 0.18, 0.15, 0.12, 0.10, 0.10, 0.08, 0.07]

SEIZURE_TYPES = [
    "Focal seizures (frontal/temporal — creatine energy deficit, neuronal hyperexcitability)",
    "Generalized tonic-clonic (severe creatine depletion, generalised energy failure)",
    "Infantile spasms / West syndrome (early-onset severe creatine-deficient AGAT)",
    "Absence / myoclonic (partial energy deficit, cortical hyperexcitability)",
    "Febrile seizures (metabolic vulnerability to temperature / catabolic stress)",
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
    is_mild   = "Mild"     in pheno
    is_severe = "Classic"  in pheno
    r = _rng.random()
    cum = 0.0
    for v, w in zip(VARIANTS, VARIANT_WEIGHTS):
        cum += w
        if r < cum:
            if is_mild and v in ("p.Val166Gly", "c.IVS2+1G>A", "p.Leu45Pro"):
                return _rng.choice(["p.Ser119Leu", "p.Thr133Ile", "p.Gln236Lys"])
            if is_severe and v in ("p.Gln236Lys", "p.Glu93Lys"):
                return _rng.choice(["p.Arg149Trp", "p.Val166Gly", "c.IVS2+1G>A"])
            return v
    return VARIANTS[0]


def _make_patient(idx):
    pheno      = _choose_pheno()
    is_mild    = "Mild"     in pheno
    is_moderate = "Moderate" in pheno
    is_severe  = "Classic"  in pheno

    # GAA — VERY LOW in ALL AGAT deficiency; nearly absent in severe
    if is_severe:
        gaa_umol    = round(_rng.uniform(0.05, 0.40), 2)   # nearly undetectable; normal 0.5–3
        creatine_pl = round(_rng.uniform(0.5,  3.5),  1)   # plasma creatine; normal 20–80
        creatinine  = round(_rng.uniform(5,    25),   1)   # urine creatinine very low
        ck_u_l      = round(_rng.uniform(50,   180),  0)   # CK mild (creatine-deficiency)
    elif is_moderate:
        gaa_umol    = round(_rng.uniform(0.1,  0.80), 2)
        creatine_pl = round(_rng.uniform(2,    10),   1)
        creatinine  = round(_rng.uniform(15,   45),   1)
        ck_u_l      = round(_rng.uniform(35,   120),  0)
    else:  # mild
        gaa_umol    = round(_rng.uniform(0.3,  1.20), 2)
        creatine_pl = round(_rng.uniform(5,    20),   1)
        creatinine  = round(_rng.uniform(30,   70),   1)
        ck_u_l      = round(_rng.uniform(25,   80),   0)

    # Methionine: NORMAL (SAM pathway not involved in AGAT reaction)
    methionine   = round(_rng.uniform(18, 42), 1)   # normal range ~10–45 µmol/L
    # tHcy: NORMAL (remethylation intact)
    hcy_umol     = round(_rng.uniform(5,  14), 1)
    # SAM: NORMAL (AGAT does NOT use SAM)
    sam_umol     = round(_rng.uniform(60, 125), 1)  # normal range 60–120
    # Arginine: NORMAL or mildly elevated (AGAT no longer consuming arginine, but diet-dependent)
    arginine     = round(_rng.uniform(50, 120), 1)  # normal 60–120 µmol/L; may be slightly high

    # Clinical features
    # Less drug-resistant than GAMT because no GAA neurotoxicity
    seizures       = _rng.random() < (0.70 if is_severe else (0.45 if is_moderate else 0.15))
    drug_resistant = seizures and _rng.random() < (0.38 if is_severe else (0.22 if is_moderate else 0.06))
    idd            = _rng.random() < (0.99 if is_severe else (0.90 if is_moderate else 0.50))
    speech_absent  = _rng.random() < (0.65 if is_severe else (0.35 if is_moderate else 0.10))
    autism_like    = _rng.random() < (0.42 if is_severe else (0.25 if is_moderate else 0.10))
    hypotonia      = _rng.random() < (0.65 if is_severe else (0.45 if is_moderate else 0.20))
    on_creatine    = _rng.random() < (0.90 if is_severe else (0.80 if is_moderate else 0.50))
    nbs_detected   = _rng.random() < (0.60 if is_severe else (0.45 if is_moderate else 0.25))

    seizure_type = _rng.choice(SEIZURE_TYPES) if seizures else None

    age_onset_mo = (
        _rng.randint(4,   24) if is_severe  else
        _rng.randint(8,   42) if is_moderate else
        _rng.randint(18,  72)
    )

    variant = _choose_variant(pheno)

    return {
        "id":                  f"AGAT-{idx:03d}",
        "phenotype":           pheno,
        "variant":             variant,
        "age_onset_months":    age_onset_mo,
        # Biomarkers
        "gaa_umol_l":          gaa_umol,          # VERY LOW / ABSENT — pathognomonic (opposite of GAMT)
        "creatine_umol_l":     creatine_pl,        # VERY LOW / ABSENT
        "creatinine_umol_l":   creatinine,         # LOW (creatine absent → catabolite absent)
        "methionine_umol_l":   methionine,         # NORMAL — KEY NEGATIVE vs SAM-cycle disorders
        "homocysteine_umol_l": hcy_umol,           # NORMAL
        "sam_umol_l":          sam_umol,           # NORMAL (AGAT does not use SAM)
        "arginine_umol_l":     arginine,           # NORMAL or slightly high
        "ck_u_l":              ck_u_l,
        "mma_normal":          True,
        "mecbl_normal":        True,
        "adocbl_normal":       True,
        # Clinical
        "seizures":            seizures,
        "seizure_type":        seizure_type,
        "drug_resistant_sz":   drug_resistant,
        "idd":                 idd,
        "speech_absent":       speech_absent,
        "autism_like":         autism_like,
        "hypotonia":           hypotonia,
        "nbs_detected":        nbs_detected,
        "on_creatine_tx":      on_creatine,
    }


_PATIENTS = [_make_patient(i + 1) for i in range(40)]


# ---------------------------------------------------------------------------
# API response functions
# ---------------------------------------------------------------------------

def get_overview() -> dict:
    n = len(_PATIENTS)

    def avg(key):
        vals = [p[key] for p in _PATIENTS if isinstance(p.get(key), (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else 0

    def pct(pred):
        return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    pheno_dist = {}
    for p in _PATIENTS:
        ph = p["phenotype"].split(" (")[0]
        pheno_dist[ph] = pheno_dist.get(ph, 0) + 1

    return {
        "dashboard_id": "agat",
        "title": "AGAT Epilepsy Dashboard",
        "subtitle": (
            "L-arginine:glycine amidinotransferase Deficiency — "
            "Cerebral Creatine Deficiency Syndrome 3 (CCDS3) / "
            "GATM-423aa-Transamidination-First-Step-15q21.1-AR"
        ),
        "gene": "GATM",
        "disease_name": (
            "L-arginine:glycine amidinotransferase Deficiency "
            "(AGAT Deficiency / Cerebral Creatine Deficiency Syndrome 3 / CCDS3)"
        ),
        "chromosome": "15q21.1",
        "inheritance": "Autosomal Recessive — biallelic LOF",
        "omim_gene":    "OMIM *602360",
        "omim_disease": "OMIM #612718",
        "protein_size": (
            "423 aa including N-terminal mitochondrial targeting sequence (~30 aa); "
            "mature protein ~393 aa after mitochondrial processing; "
            "homodimeric; expressed in kidney proximal tubule and pancreas; "
            "transamidination enzyme (not SAM-dependent); "
            "catalytic mechanism: active site Cys performs nucleophilic attack, "
            "Asp and His assist proton transfer; "
            "substrate specificity: glycine (amidino acceptor) + arginine (amidino donor); "
            "reaction type: amidino group transfer (NOT methylation, unlike GAMT/GNMT)"
        ),
        "prevalence": (
            "~150 patients reported worldwide (2026); "
            "rarest of the three cerebral creatine deficiency syndromes (GAMT > AGAT > SLC6A8-female); "
            "likely underdiagnosed due to non-specific phenotype (IDD + epilepsy) "
            "and challenging NBS detection (GAA is very LOW rather than HIGH — less obvious biomarker). "
            "Estimated prevalence 1:500,000–1:2,000,000. "
            "AR inheritance: both sexes equally affected."
        ),
        "cohort_n": n,
        "phenotype_distribution": pheno_dist,
        "function": (
            "AGAT (L-arginine:glycine amidinotransferase, encoded by GATM) catalyses: "
            "L-Arginine + Glycine → L-Ornithine + Guanidinoacetate (GAA). "
            "This is the FIRST and rate-limiting step of creatine biosynthesis. "
            "AGAT is NOT a methyltransferase — it performs transamidination: "
            "the amidino group (–C(=NH)NH₂) is transferred from arginine to glycine "
            "via a ping-pong mechanism involving an active site cysteine. "
            "SAM is NOT consumed (unlike GAMT which uses SAM in Step 2). "
            "Kidney proximal tubule is the primary site of GAA production; "
            "the GAA produced in kidney is then secreted into blood and transported to the liver "
            "where GAMT methylates it to form creatine."
        ),
        "mechanism": (
            "AGAT LOF creates a SINGLE PATHOLOGICAL STATE — absent GAA production: "
            "L-Arginine + Glycine → × AGAT BLOCKED → GAA = ABSENT / VERY LOW (<0.5 µmol/L; normal 0.5–3). "
            "Without GAA substrate, GAMT (Step 2, liver) cannot produce creatine. "
            "Plasma creatine approaches zero (<5 µmol/L; normal 20–80). "
            "Brain H-MRS: creatine peak at 3.0 ppm is ABSENT — pathognomonic (same as GAMT and SLC6A8). "
            "Brain/muscle creatine depletion → no phosphocreatine buffer → "
            "neuronal ATP cannot be regenerated during burst firing → seizure threshold lowered. "
            "CRITICAL DIFFERENCE FROM GAMT: "
            "In GAMT deficiency, GAA massively accumulates (50–300 µmol/L) and is a potent "
            "direct convulsant (GABA-A inhibition + NMDA activation + ROS). "
            "In AGAT deficiency, GAA is NOT produced → no GAA neurotoxicity → "
            "seizures are LESS drug-resistant and often respond to AEDs alone (unlike GAMT). "
            "Creatine supplementation is the definitive treatment — no ornithine required (no GAA to reduce)."
        ),
        "key_positive_features": (
            "KEY POSITIVES (unique to AGAT / CCDS3 among inherited epilepsies): "
            "(1) Guanidinoacetate (GAA) VERY LOW or ABSENT (<0.5 µmol/L; normal 0.5–3). "
            "    Low GAA is the discriminating test — opposite of GAMT (GAA 50–300 µmol/L). "
            "(2) Creatine ABSENT or very low (<5 µmol/L; normal 20–80). "
            "(3) Brain H-MRS: creatine/phosphocreatine peak (3.0 ppm) ABSENT — pathognomonic. "
            "    Shared with GAMT and SLC6A8 but mechanism differs (source absent vs clearance blocked). "
            "(4) Urine creatinine VERY LOW (creatinine = creatine catabolite; source absent). "
            "(5) Intellectual disability (universal — 100%); severity proportional to creatine deficit. "
            "(6) Seizures milder than GAMT (~50–60%) and LESS drug-resistant (25–35%). "
            "KEY NEGATIVES (absent in AGAT — major differentiators): "
            "(1) GAA NOT elevated — KEY NEGATIVE vs GAMT. "
            "(2) Methionine NORMAL — KEY NEGATIVE vs MAT1A/GNMT/AHCY/CBS. "
            "(3) tHcy NORMAL — KEY NEGATIVE vs CBS/MTHFR/cblE/cblG. "
            "(4) SAM NORMAL — KEY NEGATIVE vs GNMT (SAM very high) and AHCY (SAM elevated). "
            "(5) MMA NORMAL — KEY NEGATIVE vs all MMA disorders. "
            "(6) No GAA neurotoxicity — absence of drug-resistant epilepsy feature (unlike GAMT). "
            "(7) Ectopia lentis, thromboembolism, movement disorder all ABSENT. "
            "(8) No ornithine treatment needed (no GAA accumulation to suppress)."
        ),
        "nbs_primary": (
            "Expanded NBS — plasma or urine guanidinoacetate (GAA) VERY LOW or absent "
            "(<0.5 µmol/L; normal 0.5–3). "
            "~40–60% of AGAT cases detected on expanded NBS programs measuring GAA. "
            "CRITICAL NBS CHALLENGE: low GAA is a 'negative' flag — less conspicuous than "
            "the massively elevated GAA in GAMT deficiency. Most standard NBS cut-offs flag "
            "high GAA (GAMT), but a very low GAA below the normal lower range is a subtler signal. "
            "Standard MS/MS NBS (amino acids + acylcarnitines): MISSES AGAT — "
            "GAA not routinely measured; creatine not measured on basic MS/MS. "
            "Second-tier: urine creatine/creatinine ratio (very low); plasma GAA by LC-MS/MS. "
            "NBS programs must include a GAA lower-limit cut-off to detect AGAT (not just upper-limit)."
        ),
        "nbs_secondary": (
            "Plasma guanidinoacetate (GAA): VERY LOW or ABSENT (<0.5 µmol/L; normal 0.5–3 µmol/L). "
            "Plasma creatine: VERY LOW or absent (<5 µmol/L; normal 20–80). "
            "Urine creatinine: CRITICALLY LOW (creatinine = creatine catabolite; creatine source absent). "
            "Urine GAA: very low / absent (same as plasma). "
            "Brain H-MRS (diagnostic standard): creatine/phosphocreatine peak at 3.0 ppm ABSENT. "
            "Plasma methionine: NORMAL (SAM pathway uninvolved). "
            "Plasma total homocysteine: NORMAL (<15 µmol/L). "
            "Plasma SAM: NORMAL (AGAT does not use SAM; unlike GAMT reaction). "
            "Plasma arginine: NORMAL or mildly elevated (AGAT no longer consuming arginine). "
            "Plasma ornithine: NORMAL or slightly low-normal (ornithine also made via urea cycle). "
            "Urine organic acids: MMA NORMAL; no characteristic organic acid elevation. "
            "Serum B12/folate: NORMAL (cobalamin/folate systems intact). "
            "CK: normal to mildly elevated (30–180 U/L) in ~30–40% — less prominent than CBS/AHCY. "
            "EEG: epileptiform discharges in ~60–70% with seizures. "
            "Brain MRI: usually normal or mild white matter changes; less severe than GAMT. "
            "GATM enzyme activity (kidney biopsy or fibroblasts): absent or severely reduced. "
            "Genetic confirmation: GATM sequencing (biallelic LOF variants)."
        ),
        "kpis": {
            "avg_gaa_umol_l":          avg("gaa_umol_l"),
            "avg_creatine_umol_l":     avg("creatine_umol_l"),
            "avg_creatinine_umol_l":   avg("creatinine_umol_l"),
            "avg_methionine_umol_l":   avg("methionine_umol_l"),
            "avg_homocysteine_umol_l": avg("homocysteine_umol_l"),
            "avg_ck_u_l":              avg("ck_u_l"),
            "pct_seizures":            pct(lambda p: p["seizures"]),
            "pct_drug_resistant":      pct(lambda p: p["drug_resistant_sz"]),
            "pct_idd":                 pct(lambda p: p["idd"]),
            "pct_speech_absent":       pct(lambda p: p["speech_absent"]),
            "pct_autism_like":         pct(lambda p: p["autism_like"]),
            "pct_nbs_detected":        pct(lambda p: p["nbs_detected"]),
        },
    }


def get_breakdown() -> dict:
    n = len(_PATIENTS)

    def pct(pred):
        return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    seizure_types = [
        {"type": "Focal seizures (frontal/temporal — creatine energy deficit)", "pct": 40},
        {"type": "Generalized tonic-clonic (severe creatine depletion, energy failure)", "pct": 28},
        {"type": "Infantile spasms / West syndrome (early-onset severe AGAT)", "pct": 18},
        {"type": "Absence / myoclonic (partial energy deficit, cortical hyperexcitability)", "pct": 9},
        {"type": "Febrile seizures (metabolic vulnerability to temperature / catabolic stress)", "pct": 5},
    ]

    metabolic_triggers = [
        {
            "trigger": "Creatine deficiency — chronic (untreated / non-adherent)",
            "pct": 92,
            "mechanism": (
                "Without exogenous creatine supplementation, brain and muscle remain "
                "phosphocreatine-depleted. Creatine monohydrate (300–400 mg/kg/day) is Level A. "
                "Stopping creatine leads to rapid recurrence of IDD progression and seizures. "
                "Unlike GAMT, no ornithine needed — only creatine supplementation required. "
                "Brain H-MRS creatine peak partially recovers on creatine treatment."
            ),
        },
        {
            "trigger": "Intercurrent illness / catabolic state / fasting",
            "pct": 68,
            "mechanism": (
                "Catabolism depletes muscle creatine reserves → worsens the energy deficit. "
                "Febrile illness, surgery, fasting increase metabolic demand on creatine/phosphocreatine. "
                "Maintain oral creatine during illness; IV glucose if oral route not possible. "
                "No arginine restriction or ornithine treatment required (no GAA pathway involved)."
            ),
        },
        {
            "trigger": "Valproate (VPA) — MODERATE RISK",
            "pct": 52,
            "mechanism": (
                "VPA inhibits creatine kinase in some reports and depletes carnitine. "
                "In AGAT deficiency, where creatine-kinase activity is already impaired "
                "by substrate deficiency, VPA may worsen the energy buffer deficit. "
                "LEV is preferred first-line AED; avoid VPA where possible. "
                "If VPA unavoidable in refractory cases, supplement carnitine and monitor CK."
            ),
        },
        {
            "trigger": "Metformin — MODERATE RISK",
            "pct": 38,
            "mechanism": (
                "Metformin inhibits mitochondrial complex I and may interfere with creatine "
                "transporter (SLC6A8) at high concentrations. In AGAT deficiency, any impairment "
                "of creatine uptake into cells worsens the already-deficient creatine state. "
                "Avoid metformin in AGAT patients with concurrent metabolic syndrome."
            ),
        },
        {
            "trigger": "High-protein diet (arginine-rich foods) — MINIMAL RISK in AGAT",
            "pct": 22,
            "mechanism": (
                "Unlike GAMT deficiency, where high arginine increases GAA production via intact AGAT, "
                "in AGAT deficiency the enzyme is absent — arginine cannot drive GAA production "
                "regardless of dietary intake. "
                "Arginine restriction is therefore NOT indicated in AGAT deficiency "
                "(unlike GAMT where arginine restriction is Level B). "
                "Moderate high-protein diet is generally acceptable in AGAT."
            ),
        },
        {
            "trigger": "Surgery / general anaesthesia — catabolic stress",
            "pct": 45,
            "mechanism": (
                "Perioperative fasting and catabolic state deplete creatine reserves. "
                "Maintain creatine supplementation preoperatively and postoperatively. "
                "N2O is NOT a contraindication in AGAT (cobalamin/MTR pathway intact). "
                "Discuss creatine dose adjustment with metabolic team before elective surgery."
            ),
        },
    ]

    treatments = [
        {
            "name": "Creatine Monohydrate",
            "level": "Level A — First-Line Primary Treatment (ONLY treatment required)",
            "note": (
                "300–400 mg/kg/day oral creatine monohydrate in divided doses. "
                "Provides exogenous creatine — bypasses the absent AGAT step entirely. "
                "Creatine enters circulation and is transported into brain/muscle by SLC6A8 "
                "(creatine transporter intact in AGAT deficiency — unlike SLC6A8 where Tx fails). "
                "Brain H-MRS creatine peak partially normalises within weeks-months of treatment. "
                "IDD and seizures improve significantly, especially with early/pre-symptomatic treatment. "
                "Must be continued indefinitely — stopping causes rapid relapse. "
                "Unlike GAMT, NO ornithine is needed (no GAA accumulation) — creatine alone is sufficient. "
                "Unlike SLC6A8, creatine supplementation IS effective (transporter intact)."
            ),
        },
        {
            "name": "LEV (Levetiracetam)",
            "level": "Level A — First-Line AED",
            "note": (
                "No interaction with creatine metabolism. "
                "No hepatotoxicity (important — avoid VPA where possible). "
                "First-line AED for AGAT epilepsy across all seizure types. "
                "Unlike GAMT, seizure control is BETTER with creatine + LEV — "
                "GAA neurotoxicity is absent, so AEDs are more effective once creatine is provided. "
                "Many AGAT patients achieve seizure freedom on creatine + LEV. "
                "Seizure monitoring: EEG every 6–12 months; titrate AED to response."
            ),
        },
        {
            "name": "ACTH / Vigabatrin",
            "level": "Level A — Infantile Spasms",
            "note": (
                "Standard IS/West syndrome protocol for AGAT presenting with infantile spasms. "
                "ACTH preferred; vigabatrin second-line (no retinal CI in AGAT — unlike peroxisomal ZSD). "
                "Must be combined with creatine supplementation — "
                "ACTH alone will not prevent relapse without metabolic correction. "
                "Earlier creatine treatment reduces IS occurrence vs untreated AGAT."
            ),
        },
        {
            "name": "Creatine Monohydrate — Dose Optimisation",
            "level": "Level B — Monitoring & Adjustment",
            "note": (
                "Brain H-MRS monitoring every 6–12 months to assess creatine peak recovery. "
                "Target: partial normalisation of 3.0 ppm creatine peak. "
                "Urine creatinine monitoring: rising creatinine = evidence of creatine catabolism "
                "and absorption. "
                "Dose adjustment based on weight (mg/kg/day recalculated with growth). "
                "Plasma creatine monitoring: target within low-normal range (may not reach 20–80 "
                "due to pre-existing cellular deficiency). "
                "Supplement with creatine + adequate fluid intake (avoid renal creatine loading concerns)."
            ),
        },
        {
            "name": "Ornithine Supplementation",
            "level": "NOT INDICATED — Key Distinction from GAMT",
            "note": (
                "Ornithine supplementation is NOT needed in AGAT deficiency. "
                "In GAMT deficiency, ornithine competes with glycine at AGAT → reduces GAA production. "
                "In AGAT deficiency, the enzyme is absent — ornithine has no AGAT to compete at. "
                "GAA is already very low / absent — there is no GAA accumulation to reduce. "
                "Administering ornithine in AGAT is pharmacologically redundant. "
                "This is the key treatment distinction from GAMT: "
                "AGAT: creatine ONLY (Level A). GAMT: creatine + ornithine (both Level A). "
                "Do not substitute GAMT protocols for AGAT management."
            ),
        },
        {
            "name": "Arginine Restriction",
            "level": "NOT INDICATED — Key Distinction from GAMT",
            "note": (
                "Arginine restriction is NOT indicated in AGAT deficiency. "
                "In GAMT deficiency, high dietary arginine drives intact AGAT → more GAA → worse epilepsy. "
                "In AGAT deficiency, AGAT is absent — arginine cannot produce GAA regardless of intake. "
                "Arginine restriction would deprive the patient of protein without clinical benefit. "
                "Normal protein intake is appropriate. "
                "Do NOT apply GAMT dietary protocols to AGAT patients."
            ),
        },
    ]

    drug_risks = [
        {
            "drug": "Valproate (VPA)",
            "risk": "MODERATE RISK — Prefer LEV",
            "mechanism": (
                "VPA inhibits creatine kinase; depletes carnitine. "
                "In creatine-deficient AGAT, VPA may further impair the energy buffer. "
                "Unlike GAMT, VPA does not worsen GAA (GAA already very low in AGAT). "
                "LEV is strongly preferred. If VPA unavoidable, supplement carnitine and monitor CK."
            ),
        },
        {
            "drug": "Metformin",
            "risk": "MODERATE RISK — Avoid",
            "mechanism": (
                "Mitochondrial complex I inhibition; potential SLC6A8 (creatine transporter) interference. "
                "Creatine uptake inhibition worsens already-deficient cellular creatine levels. "
                "Avoid in AGAT patients. "
                "Alternative hypoglycaemics preferred if diabetes develops."
            ),
        },
        {
            "drug": "N2O (Nitrous oxide)",
            "risk": "LOW RISK (cobalamin / MTR system intact)",
            "mechanism": (
                "AGAT deficiency does NOT affect the cobalamin methylation system. "
                "MTR (methionine synthase) and MTRR are intact. "
                "N2O is NOT an absolute contraindication in AGAT — unlike cblE/cblG/MTHFR. "
                "Use standard anaesthetic precautions. "
                "Note: perioperative fasting is a trigger (creatine depletion) — manage accordingly."
            ),
        },
        {
            "drug": "Ornithine supplements",
            "risk": "NOT INDICATED (not harmful, but pharmacologically redundant)",
            "mechanism": (
                "Ornithine reduces GAA by competing at AGAT — but AGAT is absent in AGAT deficiency. "
                "No benefit; unnecessary cost; not harmful per se. "
                "Do not apply GAMT treatment protocols (which include ornithine) to AGAT patients."
            ),
        },
        {
            "drug": "SAMe / SAM supplements",
            "risk": "NOT INDICATED (no SAM deficiency)",
            "mechanism": (
                "AGAT deficiency does NOT involve the SAM pathway. "
                "SAM is normal; no methylation deficit. "
                "SAMe supplements have no role in AGAT management."
            ),
        },
        {
            "drug": "Betaine / TMG",
            "risk": "NOT INDICATED (no homocysteine elevation)",
            "mechanism": (
                "tHcy is NORMAL in AGAT deficiency. "
                "Betaine (BHMT pathway) is used to lower homocysteine in CBS/MTHFR/cblE/cblG. "
                "No indication in AGAT."
            ),
        },
    ]

    # Variant breakdown
    variant_counts: dict[str, int] = {}
    for p in _PATIENTS:
        v = p["variant"]
        variant_counts[v] = variant_counts.get(v, 0) + 1

    variants_list = [
        {
            "variant": v,
            "count": variant_counts.get(v, 0),
            "pct": round(variant_counts.get(v, 0) / n * 100),
        }
        for v in VARIANTS
    ]

    # Patient sample (first 12)
    sample = [
        {
            "id":              p["id"],
            "phenotype":       p["phenotype"].split(" (")[0],
            "variant":         p["variant"],
            "gaa":             p["gaa_umol_l"],
            "creatine":        p["creatine_umol_l"],
            "creatinine":      p["creatinine_umol_l"],
            "methionine":      p["methionine_umol_l"],
            "homocysteine":    p["homocysteine_umol_l"],
            "ck":              p["ck_u_l"],
            "seizures":        p["seizures"],
            "drug_resistant":  p["drug_resistant_sz"],
            "nbs":             p["nbs_detected"],
        }
        for p in _PATIENTS[:12]
    ]

    return {
        "dashboard_id": "agat",
        "cohort_n": n,
        "kpi_pcts": {
            "pct_seizures":          pct(lambda p: p["seizures"]),
            "pct_drug_resistant":    pct(lambda p: p["drug_resistant_sz"]),
            "pct_idd":               pct(lambda p: p["idd"]),
            "pct_speech_absent":     pct(lambda p: p["speech_absent"]),
            "pct_autism_like":       pct(lambda p: p["autism_like"]),
            "pct_hypotonia":         pct(lambda p: p["hypotonia"]),
            "pct_nbs":               pct(lambda p: p["nbs_detected"]),
        },
        "phenotype_dist": [
            {"label": "Classic",   "pct": 55},
            {"label": "Moderate",  "pct": 35},
            {"label": "Mild",      "pct": 10},
        ],
        "seizure_types":      seizure_types,
        "metabolic_triggers": metabolic_triggers,
        "treatments":         treatments,
        "drug_risks":         drug_risks,
        "variants":           variants_list,
        "patient_sample":     sample,
        "biomarker_ranges": {
            "gaa_severe_umol_l":      "<0.5 (VERY LOW / ABSENT — PATHOGNOMONIC; normal 0.5–3)",
            "gaa_moderate_umol_l":    "0.1–0.8 (VERY LOW — normal 0.5–3)",
            "gaa_mild_umol_l":        "0.3–1.2 (LOW — borderline/below lower reference range)",
            "creatine_umol_l":        "<3–20 (normal 20–80) — VERY LOW in all phenotypes",
            "creatinine_umol_l":      "5–70 (normal 40–120) — LOW (creatine catabolite absent)",
            "methionine_umol_l":      "18–42 (NORMAL — KEY NEGATIVE vs all hypermethioninemia)",
            "homocysteine_umol_l":    "5–14 (NORMAL — KEY NEGATIVE vs CBS/MTHFR/cblE/cblG)",
            "sam_umol_l":             "60–125 (NORMAL — no SAM pathway involvement, KEY NEGATIVE vs GNMT)",
            "arginine_umol_l":        "50–120 (NORMAL or mildly high — AGAT no longer consuming arginine)",
            "mma":                    "NORMAL (KEY NEGATIVE vs all MMA disorders)",
            "ck_u_l":                 "25–180 (normal to mildly elevated — less than AHCY/CBS)",
            "brain_mrs":              "Creatine peak 3.0 ppm ABSENT — all phenotypes — PATHOGNOMONIC",
        },
    }


def get_definitions() -> dict:
    return {
        "dashboard_id": "agat",
        "gene_card": {
            "gene":       "GATM",
            "disease_acronym": "AGAT (L-arginine:glycine amidinotransferase)",
            "full_name":  "L-arginine:glycine amidinotransferase",
            "chromosome": "15q21.1",
            "size_aa":    423,
            "mature_aa":  "~393 aa (after mitochondrial targeting sequence cleavage)",
            "structure":  (
                "Homodimeric mitochondrial enzyme; expressed in kidney proximal tubule and pancreas; "
                "transamidination mechanism (NOT methylation, NOT SAM-dependent); "
                "active site Cys performs nucleophilic attack on arginine guanidino group; "
                "Asp and His assist proton transfer; "
                "substrate 1: L-arginine (amidino group donor); substrate 2: glycine (amidino acceptor); "
                "products: L-ornithine + guanidinoacetate (GAA); "
                "ping-pong bi-bi kinetic mechanism with covalent enzyme-amidino intermediate."
            ),
            "omim_gene":    "*602360",
            "omim_disease": "#612718",
            "ccds_number":  "CCDS3 (Cerebral Creatine Deficiency Syndrome 3)",
            "inheritance":  "Autosomal Recessive",
        },
        "key_concepts": [
            {
                "term": "AGAT Deficiency — Single Pathology (Absent GAA/Creatine), Unlike GAMT's Dual Pathology",
                "definition": (
                    "AGAT deficiency creates ONE pathological consequence: "
                    "no GAA is made → GAMT has no substrate → no creatine. "
                    "Brain H-MRS creatine peak absent + energy buffer depleted → IDD + seizures. "
                    "GAMT deficiency has TWO problems: (1) no creatine + (2) GAA massively accumulates. "
                    "The accumulated GAA in GAMT is a potent endogenous convulsant "
                    "(GABA-A inhibition + NMDA activation). "
                    "AGAT has NO GAA accumulation — GAA is absent, not elevated. "
                    "This is why AGAT seizures are LESS drug-resistant (25–35%) than GAMT (60–80%) — "
                    "without GAA neurotoxicity, AEDs can work once creatine is replaced. "
                    "Treatment for AGAT: creatine ONLY. Treatment for GAMT: creatine + ornithine."
                ),
            },
            {
                "term": "GAA Direction — The Single Fastest Discriminating Test Between AGAT and GAMT",
                "definition": (
                    "Both AGAT and GAMT deficiency produce absent brain creatine on H-MRS. "
                    "The biochemical discriminator is the DIRECTION of GAA: "
                    "AGAT deficiency: GAA VERY LOW / ABSENT (<0.5 µmol/L; normal 0.5–3). "
                    "  Reason: AGAT cannot make GAA from arginine + glycine. "
                    "GAMT deficiency: GAA MASSIVELY HIGH (50–300 µmol/L; normal <3). "
                    "  Reason: AGAT still makes GAA but GAMT cannot methylate it to creatine. "
                    "Therefore: plasma or urine GAA measurement immediately distinguishes AGAT from GAMT. "
                    "Low GAA + absent creatine = AGAT. High GAA + absent creatine = GAMT. "
                    "If both brain MRS and GAA are measured simultaneously, diagnosis is made within hours."
                ),
            },
            {
                "term": "Creatine Biosynthesis — AGAT as Rate-Limiting First Step",
                "definition": (
                    "Step 1 — AGAT (GATM gene, kidney/pancreas): "
                    "L-Arginine + Glycine → L-Ornithine + Guanidinoacetate (GAA). [RATE-LIMITING STEP] "
                    "Step 2 — GAMT (liver/pancreas): SAM + GAA → SAH + Creatine. "
                    "Step 3 — SLC6A8 (ubiquitous): Creatine transported into muscle/brain. "
                    "In cells: Creatine kinase: Creatine + ATP ⇌ Phosphocreatine + ADP. "
                    "AGAT is rate-limiting — a small reduction in AGAT activity substantially "
                    "reduces creatine biosynthesis. "
                    "All three CCDS (AGAT, GAMT, SLC6A8) cause absent brain creatine by different steps: "
                    "AGAT: no GAA made (first step). "
                    "GAMT: GAA made but not methylated (second step). "
                    "SLC6A8: creatine made but cannot enter cells (third step, X-linked)."
                ),
            },
            {
                "term": "Why Creatine Treatment Works in AGAT but NOT in SLC6A8",
                "definition": (
                    "AGAT deficiency: AGAT cannot make GAA → GAMT has no substrate → no creatine. "
                    "The creatine TRANSPORTER (SLC6A8) is intact in AGAT deficiency. "
                    "Therefore: exogenous creatine (oral supplement) enters the bloodstream and "
                    "SLC6A8 transports it normally into muscle and brain. "
                    "Brain H-MRS creatine peak recovers (partially) within weeks–months. "
                    "SLC6A8 deficiency (CCDS1, X-linked): the TRANSPORTER is absent. "
                    "Even though creatine can be synthesised normally, it cannot enter cells. "
                    "Oral creatine supplementation in SLC6A8 is largely ineffective "
                    "because the transporter needed to bring it into muscle/brain is absent. "
                    "This is the key pharmacological distinction: "
                    "AGAT and GAMT → creatine treatment highly effective; "
                    "SLC6A8 → creatine treatment largely ineffective."
                ),
            },
            {
                "term": "Why No Ornithine and No Arginine Restriction in AGAT (Unlike GAMT)",
                "definition": (
                    "GAMT deficiency requires ornithine supplementation (Level A) because: "
                    "AGAT (intact) + Arginine + Glycine → Ornithine + GAA. "
                    "Ornithine competes with glycine at intact AGAT → less GAA made → GAA decreases. "
                    "In AGAT deficiency: AGAT is ABSENT. "
                    "→ Ornithine has no AGAT to compete at — it cannot reduce GAA production. "
                    "→ Ornithine supplementation is pharmacologically redundant (not harmful, just useless). "
                    "Similarly, arginine restriction in GAMT reduces AGAT substrate → less GAA. "
                    "In AGAT: no AGAT activity → arginine restriction cannot reduce GAA (already absent). "
                    "→ Arginine restriction is NOT indicated in AGAT. "
                    "This is a critical protocol differentiation: "
                    "Never apply GAMT metabolic protocols (ornithine + arginine restriction) "
                    "to AGAT patients — only creatine is needed."
                ),
            },
            {
                "term": "Brain H-MRS — Shared CCDS Feature, Different Underlying Mechanism",
                "definition": (
                    "Proton H-MRS shows the SAME finding in all three CCDS: "
                    "creatine/phosphocreatine peak at 3.0 ppm ABSENT. "
                    "However, the underlying mechanism differs: "
                    "AGAT: no GAA made → no creatine synthesised → brain empty of creatine. "
                    "GAMT: GAA made but cannot be methylated → no creatine synthesised → same. "
                    "SLC6A8: creatine synthesised normally → but cannot enter brain → brain empty. "
                    "The H-MRS finding alone cannot distinguish AGAT from GAMT from SLC6A8. "
                    "Biochemistry is needed: plasma GAA distinguishes AGAT (low) from GAMT (high). "
                    "Urine creatine/creatinine ratio, genetics confirm SLC6A8. "
                    "H-MRS is the fastest flag — if creatine absent on MRS → check plasma GAA immediately."
                ),
            },
            {
                "term": "NBS Complexity — Low GAA vs High GAA as Opposing Biomarkers",
                "definition": (
                    "GAMT NBS: GAA ELEVATED (50–300 µmol/L) → upper-limit cut-off flags it easily. "
                    "AGAT NBS: GAA VERY LOW (<0.5 µmol/L) → requires a LOWER-LIMIT cut-off for GAA. "
                    "Standard NBS programs set upper-limit GAA cut-offs (for GAMT). "
                    "A GAA value below the lower reference range is a different — and less familiar — flag. "
                    "Many AGAT cases are missed at NBS because programs lack a low-GAA cut-off. "
                    "Second-tier: urine creatine/creatinine ratio (very low) is an accessible flag. "
                    "Plasma creatine on NBS (if measured) would flag both AGAT and GAMT. "
                    "Early detection is critical: pre-symptomatic AGAT treatment can prevent IDD."
                ),
            },
        ],
        "differential_diagnosis": [
            {
                "disorder": "GAMT Deficiency (CCDS2 — Guanidinoacetate N-methyltransferase)",
                "key_distinction": (
                    "GAMT: GAA MASSIVELY HIGH (50–300 µmol/L). "
                    "AGAT: GAA VERY LOW / ABSENT (<0.5 µmol/L). "
                    "Both: creatine absent, brain H-MRS creatine absent, urine creatinine low. "
                    "GAMT: drug-resistant epilepsy 60–80% (GAA is potent convulsant). "
                    "AGAT: drug-resistant epilepsy only 25–35% (no GAA neurotoxicity). "
                    "GAMT treatment: creatine + ornithine (both Level A). "
                    "AGAT treatment: creatine ONLY — no ornithine, no arginine restriction. "
                    "GAA direction is the single fastest discriminating test."
                ),
            },
            {
                "disorder": "SLC6A8 Deficiency (CCDS1 — Creatine Transporter Defect, X-linked)",
                "key_distinction": (
                    "SLC6A8: creatine CANNOT enter cells (transporter absent). "
                    "GAA: NORMAL in SLC6A8 (synthesis intact — AGAT + GAMT both working). "
                    "Plasma creatine: NORMAL or HIGH in SLC6A8 (cannot be transported → accumulates). "
                    "Urine creatine/creatinine ratio: ELEVATED in SLC6A8 (opposite of AGAT). "
                    "X-linked: males severely affected; carrier females may have cognitive symptoms. "
                    "Creatine supplementation: LARGELY INEFFECTIVE in SLC6A8 (transporter absent). "
                    "AGAT: creatine supplementation HIGHLY EFFECTIVE (SLC6A8 intact). "
                    "Brain MRS: creatine absent in both (same MRS, different cause)."
                ),
            },
            {
                "disorder": "GAMT vs AGAT — Clinical Phenotype Overlap",
                "key_distinction": (
                    "Both: IDD (100%), absent brain MRS creatine, low urine creatinine, "
                    "methionine normal, tHcy normal, MMA normal. "
                    "GAMT: movement disorder 35–50%, myopathy 30–40%, seizures 82% classic, "
                    "speech absent 75% classic. "
                    "AGAT: movement disorder less common, seizures ~70% classic (milder), "
                    "better speech outcomes with early treatment. "
                    "AGAT: typically milder overall due to absent GAA neurotoxicity. "
                    "Both respond to creatine — but GAMT additionally needs ornithine."
                ),
            },
            {
                "disorder": "Non-specific IDD + Epilepsy — Broad Differential",
                "key_distinction": (
                    "Brain H-MRS creatine peak absence is the key diagnostic clue. "
                    "Without H-MRS, AGAT presents as non-specific IDD + epilepsy "
                    "(overlaps with Angelman, Rett, GLDC/NKH, metabolic epilepsies). "
                    "Plasma/urine GAA + creatine + creatinine panel is a cheap first-tier test. "
                    "In AGAT: GAA absent, creatine low, creatinine low (all pointing to creatine defect). "
                    "In Angelman/Rett/SCN1A/CDKL5: all biomarkers normal; diagnosis by genetics. "
                    "AGAT is treatable — early diagnosis dramatically changes outcome."
                ),
            },
            {
                "disorder": "MAT1A / GNMT / AHCY / CBS Deficiency (hypermethioninemia/homocystinuria)",
                "key_distinction": (
                    "All hypermethioninemia/homocystinuria disorders: "
                    "methionine HIGH or tHcy HIGH or MMA elevated. "
                    "AGAT: methionine NORMAL, tHcy NORMAL, MMA NORMAL, SAM NORMAL. "
                    "AGAT unique: GAA very low + creatine absent — no other condition shares this. "
                    "Single plasma amino acid panel showing low GAA + low creatine + normal methionine "
                    "immediately excludes the entire hypermethioninemia/homocystinuria spectrum."
                ),
            },
        ],
        "treatment_summary": {
            "first_line": (
                "Creatine monohydrate 300–400 mg/kg/day (Level A — only treatment required) + "
                "LEV (first-line AED, Level A)"
            ),
            "absolute_ci": [],    # No absolute contraindications unique to AGAT
            "high_risk": [
                "Valproate — impairs creatine kinase; carnitine depletion; prefer LEV",
                "Metformin — potential creatine transporter (SLC6A8) interference",
            ],
            "not_indicated": [
                "Ornithine supplementation — no AGAT activity to inhibit; no GAA to reduce",
                "Arginine restriction — AGAT absent; arginine cannot drive GAA production",
                "SAMe / SAM — no SAM deficiency in AGAT",
                "Betaine / TMG — no homocysteine elevation (tHcy normal)",
                "Folinic acid — no methylfolate trap (folate/cobalamin system intact)",
            ],
        },
    }
