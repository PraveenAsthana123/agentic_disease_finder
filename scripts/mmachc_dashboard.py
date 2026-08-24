#!/usr/bin/env python3
"""MMACHC (Methylmalonic Acidemia with Homocystinuria, cblC type) Epilepsy Dashboard.

MMACHC encodes the cytoplasmic bifunctional decyanase/reductase that is the FIRST and
MOST UPSTREAM step in intracellular cobalamin processing. It converts ALL dietary cobalamin
forms (CN-Cbl, OH-Cbl, Me-Cbl, Ado-Cbl) into cob(I)alamin for distribution to:
  (A) Mitochondria: → MMAB → AdoCbl → MMUT pathway (methylmalonate catabolism)
  (B) Cytoplasm: → MMADHC → MS/MTR → MeCbl → Methionine synthase (homocysteine remethylation)

MMACHC LOF → BOTH pathways blocked → COMBINED MMA + Homocystinuria (cblC disease)
This is the HALLMARK of cblC: isolated MMA (MMUT/MMAA/MMAB) does NOT elevate homocysteine.

  MMACHC PROTEIN BIOLOGY:
    MMACHC protein: 282 aa; cytoplasmic; ubiquitous expression.
    Bifunctional: (1) Decyanase — removes CN from CN-Cbl using NADPH; (2) Reductase/dealkylase
    — removes upper axial ligand (Me or Ado) from alkyl-Cbls, converting all forms to cob(I)alamin.
    Result: cob(I)alamin pool → partitioned to MMAB (mitochondrial; → AdoCbl) and MMADHC
    (cytoplasmic; → methionine synthase pathway; → MeCbl).

  MMACHC LOF — BOTH ARMS BLOCKED:
    Arm A (MMA): cob(I)alamin not made → MMAB cannot synthesize AdoCbl →
      MMUT apoenzyme cofactor-empty → L-methylmalonyl-CoA accumulates → MMA ↑↑
    Arm B (HHcy): cob(I)alamin not made → MMADHC cannot supply MeCbl →
      Methionine synthase (MS/MTR) cannot convert Hcy→Methionine →
      Homocysteine ACCUMULATES (HHcy ↑↑) + Methionine LOW

  KEY DISTINCTION FROM ISOLATED MMA (MMUT/MMAA/MMAB):
    Isolated MMA: ONLY MMA elevated; Homocysteine NORMAL (<15 µmol/L)
    cblC (MMACHC): BOTH MMA elevated AND Homocysteine elevated (HHcy >30–300 µmol/L)
    → Homocysteine elevation is the PATHOGNOMONIC distinguishing feature for cblC

  BIOMARKERS:
    C3 (propionylcarnitine): ↑ — NBS PRIMARY (same as PA, MMUT, MMAA, MMAB)
    Methylmalonic acid (urine): ↑ 200–2,000 mmol/mol Cr (elevated but LOWER than MMUT mut0 5,000+)
    Total homocysteine (tHcy): ↑↑ >30–300 µmol/L (PATHOGNOMONIC for cblC — KEY POSITIVE)
    Methionine (plasma): ↓↓ 8–18 µmol/L (normal 20–40); methionine synthase blocked
    MeCbl (fibroblasts): ABSENT — cytoplasmic MeCbl synthesis blocked
    AdoCbl (fibroblasts): ABSENT — mitochondrial AdoCbl synthesis also blocked (via MMAB)
    Methylcitrate: mildly elevated secondary only (<30 µmol/mmol Cr; vs PA pathognomonic)
    Ammonia: ↑ 50–500 µmol/L (secondary — NAGS/CPS1 via MMA; lower than isolated MMA)
    Free carnitine: ↓ 8–30 µmol/L (secondary depletion)
    BCAA: NORMAL (KEY NEGATIVE vs DLD/MSUD)

  CLINICAL HALLMARKS UNIQUE TO cblC:
    Maculopathy (retinal pigment epitheliopathy): ~80% of early-onset cblC — PATHOGNOMONIC for cblC
    Hemolytic uremic syndrome (HUS) / microangiopathy: neonatal severe ~30%
    Pulmonary hypertension: neonatal severe ~20%
    Congenital heart disease: ~15%
    Late-onset psychiatric: c.482G>A allele — schizophrenia-like syndrome onset >4 years

  KEY VARIANTS IN MMACHC:
    - c.271dupA (p.Arg91Lys*/frameshift): most common worldwide ~40–50% of cblC alleles;
      null; exon 4 frameshift; neonatal/infantile severe; no MMACHC protein
    - c.394C>T (p.Arg132*): second most common null; severe; pan-ethnic; neonatal onset
    - c.482G>A (p.Arg161Gln): late-onset allele; partial MMACHC function; psychiatric/
      subacute cognitive onset (>4 years); only compound het with null gives late phenotype
    - c.347T>C (p.Leu116Pro): intermediate; early infantile onset; partial function
    - c.565C>A (p.Pro189Thr): mild; late childhood onset
    - c.271dupA/c.482G>A: most common compound het for late-onset cblC (null + partial)
    - IVS1-1G>A (c.1-1G>A): splice site null; severe; European
    - c.616C>T (p.Arg206*): nonsense null; severe; early onset

  TREATMENT HIGHLIGHTS:
    Hydroxocobalamin (OHCbl) 1–2 mg/day IM (Level A) — bypasses MMACHC; ALL patients trial;
      ~75% respond for HHcy (tHcy falls >50%); MMA response ~50–60%; lifelong IM preferred.
    Betaine (trimethylglycine) 100–200 mg/kg/day orally (Level A) — remethylates Hcy via
      BHMT (betaine-homocysteine methyltransferase; cobalamin-independent liver pathway);
      reduces HHcy even when OHCbl response is incomplete; mandatory co-treatment.
    Folinic acid / 5-methylTHF (Level B) — depleted due to methylene-THF trap;
      folate-methylation cycle impaired when methionine synthase blocked.
    L-Carnitine 50–100 mg/kg/day (Level B) — secondary depletion; less severe than isolated MMA.
    Methionine supplementation (Level C) — if plasma methionine persistently <15 µmol/L.
    Protein restriction (Level C/supportive) — less critical than isolated MMA (MMA lower);
      moderate natural protein restriction only.
    VPA — HIGH RISK / AVOID: severe carnitine depletion + metabolic stress; not the same
      direct enzyme-inhibition mechanism as isolated MMA, but seizure control risk high;
      LEV preferred as first-line AED.
    Fasting — HIGH RISK (MMA surge + HHcy worsens in catabolic state).

  OMIM: Gene *609831 · Disease #277400 (Methylmalonic Acidemia and Homocystinuria, cblC type)
  Chromosome: 1p34.1 · Inheritance: AR · Prevalence: ~1:50,000–100,000
  (Most common inborn error of intracellular cobalamin metabolism)
"""

from __future__ import annotations
import random

random.seed(47)   # reproducible synthetic cohort


# ── helpers ──────────────────────────────────────────────────────────────────

def _pid(i: int) -> str:
    return f"MMACHC-{i:03d}"


def _sex(i: int) -> str:
    return "M" if i % 2 == 0 else "F"


def _phenotype(i: int) -> str:
    """
    cblC Early-Onset Neonatal Severe ~45% (18 pts) — HUS, pulm HTN, severe MMA+HHcy,
    cblC Early-Onset Infantile ~35% (14 pts — MODAL) — maculopathy, infantile spasms,
    cblC Late-Onset Child/Adult ~15% (6 pts) — psychiatric/cognitive, c.482G>A,
    cblC Mild/Attenuated ~5% (2 pts).
    """
    if i <= 18:   return "cblC Early-Onset Neonatal Severe"
    if i <= 32:   return "cblC Early-Onset Infantile"
    if i <= 38:   return "cblC Late-Onset Child/Adult"
    return "cblC Mild/Attenuated"


def _genotype(i: int) -> str:
    variants = [
        "c.271dupA/c.271dupA",          # homozygous most common null; neonatal severe
        "c.271dupA/c.394C>T",           # two nulls; severe neonatal
        "c.271dupA/c.482G>A",           # null/late-onset allele; intermediate or late
        "c.394C>T/c.394C>T",            # homozygous null; severe
        "c.271dupA/IVS1-1G>A",          # null/splice null; severe
        "c.271dupA/c.347T>C",           # null/intermediate; infantile
        "c.482G>A/c.482G>A",            # homozygous late-onset allele; late psychiatric
        "c.347T>C/c.394C>T",            # intermediate/null; infantile-childhood
        "c.565C>A/c.271dupA",           # mild/null; late childhood
        "c.271dupA/c.616C>T",           # null/null; severe neonatal
    ]
    return variants[i % len(variants)]


def _mma_urine(pheno: str) -> float:
    """Methylmalonic acid urine mmol/mol Cr — lower than MMUT mut0; higher in neonatal."""
    if pheno == "cblC Early-Onset Neonatal Severe":
        return round(random.uniform(1000, 2000), 0)
    if pheno == "cblC Early-Onset Infantile":
        return round(random.uniform(300, 1200), 0)
    if pheno == "cblC Late-Onset Child/Adult":
        return round(random.uniform(150, 600), 0)
    return round(random.uniform(80, 300), 0)


def _plasma_mma(pheno: str) -> float:
    """Plasma MMA µmol/L."""
    if pheno == "cblC Early-Onset Neonatal Severe":
        return round(random.uniform(500, 1500), 0)
    if pheno == "cblC Early-Onset Infantile":
        return round(random.uniform(150, 600), 0)
    if pheno == "cblC Late-Onset Child/Adult":
        return round(random.uniform(80, 300), 0)
    return round(random.uniform(40, 150), 0)


def _homocysteine(pheno: str) -> float:
    """Total homocysteine µmol/L — ELEVATED (PATHOGNOMONIC for cblC)."""
    if pheno == "cblC Early-Onset Neonatal Severe":
        return round(random.uniform(150, 300), 0)
    if pheno == "cblC Early-Onset Infantile":
        return round(random.uniform(80, 200), 0)
    if pheno == "cblC Late-Onset Child/Adult":
        return round(random.uniform(50, 150), 0)
    return round(random.uniform(30, 80), 0)


def _methionine(pheno: str) -> float:
    """Plasma methionine µmol/L — LOW (methionine synthase blocked)."""
    if pheno == "cblC Early-Onset Neonatal Severe":
        return round(random.uniform(5, 12), 1)
    if pheno == "cblC Early-Onset Infantile":
        return round(random.uniform(8, 16), 1)
    if pheno == "cblC Late-Onset Child/Adult":
        return round(random.uniform(12, 22), 1)
    return round(random.uniform(15, 28), 1)


def _c3(pheno: str) -> float:
    if pheno == "cblC Early-Onset Neonatal Severe":
        return round(random.uniform(8, 30), 1)
    if pheno == "cblC Early-Onset Infantile":
        return round(random.uniform(4, 18), 1)
    if pheno == "cblC Late-Onset Child/Adult":
        return round(random.uniform(2, 10), 1)
    return round(random.uniform(2, 6), 1)


def _ammonia(pheno: str) -> int:
    if pheno == "cblC Early-Onset Neonatal Severe":
        return random.randint(150, 500)
    if pheno == "cblC Early-Onset Infantile":
        return random.randint(60, 200)
    if pheno == "cblC Late-Onset Child/Adult":
        return random.randint(25, 80)
    return random.randint(15, 50)


def _carnitine(pheno: str) -> float:
    if pheno == "cblC Early-Onset Neonatal Severe":
        return round(random.uniform(8, 20), 1)
    if pheno == "cblC Early-Onset Infantile":
        return round(random.uniform(12, 30), 1)
    if pheno == "cblC Late-Onset Child/Adult":
        return round(random.uniform(18, 40), 1)
    return round(random.uniform(22, 45), 1)


def _onset_months(pheno: str) -> int:
    if pheno == "cblC Early-Onset Neonatal Severe":
        return random.randint(0, 1)
    if pheno == "cblC Early-Onset Infantile":
        return random.randint(1, 12)
    if pheno == "cblC Late-Onset Child/Adult":
        return random.randint(48, 240)
    return random.randint(24, 180)


def _ohcbl_response(pheno: str) -> bool:
    """~75% HHcy response; MMA response ~55-60% overall in cblC."""
    if pheno == "cblC Early-Onset Neonatal Severe":
        return random.random() < 0.60
    if pheno == "cblC Early-Onset Infantile":
        return random.random() < 0.80
    if pheno == "cblC Late-Onset Child/Adult":
        return random.random() < 0.90
    return True


def _maculopathy(pheno: str) -> bool:
    """Retinal maculopathy ~80% early-onset; hallmark of cblC."""
    if pheno == "cblC Early-Onset Neonatal Severe":
        return random.random() < 0.85
    if pheno == "cblC Early-Onset Infantile":
        return random.random() < 0.80
    if pheno == "cblC Late-Onset Child/Adult":
        return random.random() < 0.30
    return random.random() < 0.15


def _seizures(pheno: str) -> bool:
    if pheno == "cblC Early-Onset Neonatal Severe":
        return random.random() < 0.85
    if pheno == "cblC Early-Onset Infantile":
        return random.random() < 0.65
    if pheno == "cblC Late-Onset Child/Adult":
        return random.random() < 0.25
    return random.random() < 0.10


def _psychiatric(pheno: str) -> bool:
    """Late-onset cblC has psychiatric/cognitive manifestation — c.482G>A hallmark."""
    if pheno == "cblC Late-Onset Child/Adult":
        return random.random() < 0.75
    if pheno == "cblC Mild/Attenuated":
        return random.random() < 0.40
    return False


def _aed(pheno: str, seizures: bool) -> str:
    if not seizures:
        return "None"
    aeds = ["LEV", "LEV + CLB", "LEV + VNS", "LEV monotherapy", "LEV + folinic acid"]
    return random.choice(aeds)


def _nbs(pheno: str) -> bool:
    if pheno == "cblC Early-Onset Neonatal Severe":
        return random.random() < 0.70
    return random.random() < 0.85


def _diag_months(pheno: str, onset: int) -> int:
    if pheno == "cblC Early-Onset Neonatal Severe":
        return onset + random.randint(0, 2)
    if pheno == "cblC Late-Onset Child/Adult":
        return onset + random.randint(6, 60)   # late-onset often misdiagnosed as psychiatric
    return onset + random.randint(1, 8)


# ── build cohort ──────────────────────────────────────────────────────────────

_PATIENTS = []
for _i in range(1, 41):
    ph  = _phenotype(_i)
    sz  = _seizures(ph)
    ons = _onset_months(ph)
    _PATIENTS.append({
        "patient_id":                _pid(_i),
        "sex":                       _sex(_i),
        "phenotype":                 ph,
        "age_at_onset_months":       ons,
        "age_at_diagnosis_months":   _diag_months(ph, ons),
        "genotype":                  _genotype(_i),
        "mma_urine_mmol_molCr":      _mma_urine(ph),
        "plasma_mma_umol_l":         _plasma_mma(ph),
        "total_homocysteine_umol_l": _homocysteine(ph),
        "methionine_umol_l":         _methionine(ph),
        "c3_umol_l":                 _c3(ph),
        "ammonia_umol_l":            _ammonia(ph),
        "free_carnitine_umol_l":     _carnitine(ph),
        "seizures":                  sz,
        "maculopathy":               _maculopathy(ph),
        "psychiatric_features":      _psychiatric(ph),
        "nbs_detected":              _nbs(ph),
        "ohcbl_response":            _ohcbl_response(ph),
        "aed":                       _aed(ph, sz),
    })


# ── API functions ─────────────────────────────────────────────────────────────

def get_overview() -> dict:
    n       = len(_PATIENTS)
    mma_v   = [p["mma_urine_mmol_molCr"] for p in _PATIENTS]
    hcy_v   = [p["total_homocysteine_umol_l"] for p in _PATIENTS]
    met_v   = [p["methionine_umol_l"] for p in _PATIENTS]
    c3_v    = [p["c3_umol_l"] for p in _PATIENTS]
    nh3_v   = [p["ammonia_umol_l"] for p in _PATIENTS]
    car_v   = [p["free_carnitine_umol_l"] for p in _PATIENTS]

    def pct(pred): return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    return {
        "gene": "MMACHC",
        "full_name": "Methylmalonyl-CoA/Homocysteine Cobalamin Processing Chaperone (cblC gene product)",
        "chromosome": "1p34.1",
        "inheritance": "AR",
        "omim_gene": "*609831",
        "omim_disease": "#277400",
        "protein_size": "282 aa; cytoplasmic; bifunctional decyanase/reductase; ubiquitous",
        "prevalence": "~1:50,000–100,000 (most common inborn error of intracellular cobalamin metabolism)",
        "nbs_primary": "C3 (propionylcarnitine) elevated — identical to PA, MMUT, MMAA, MMAB; triggers MMA workup",
        "nbs_secondary": "Urine MMA + tHcy (BOTH elevated — PATHOGNOMONIC cblC); methionine low; gene panel",
        "function": (
            "MMACHC (cblC) is the cytoplasmic bifunctional enzyme that performs STEP 1 of intracellular "
            "cobalamin processing: converts ALL dietary cobalamin forms (CN-Cbl, OH-Cbl, Me-Cbl, Ado-Cbl) "
            "to cob(I)alamin via decyanation (NADPH-dependent) and dealkylation. "
            "cob(I)alamin is then partitioned to: "
            "(A) MMAB (mitochondria) → AdoCbl → MMUT (MMA catabolism); "
            "(B) MMADHC (cytoplasm) → MeCbl → Methionine synthase (Hcy→Met remethylation)."
        ),
        "mechanism": (
            "MMACHC LOF → cob(I)alamin CANNOT BE PRODUCED from any dietary cobalamin form → "
            "BOTH downstream pathways blocked simultaneously: "
            "(A) No AdoCbl → MMUT cofactor-empty → L-methylmalonyl-CoA accumulates → MMA ↑↑ (200–2,000 mmol/mol Cr); "
            "(B) No MeCbl → Methionine synthase (MS/MTR) inactive → Homocysteine ACCUMULATES (HHcy >30–300 µmol/L) "
            "+ Methionine LOW (8–18 µmol/L; methionine synthase cannot run forward). "
            "This COMBINED MMA+HHcy is PATHOGNOMONIC for cblC — isolated MMA (MMUT/MMAA/MMAB) has NORMAL Hcy."
        ),
        "key_negative": (
            "BCAA NORMAL — KEY NEGATIVE vs DLD/MSUD; "
            "METHYLCITRATE mildly elevated only — NOT pathognomonic (vs PA where pathognomonic); "
            "KEY POSITIVE: HOMOCYSTEINE ELEVATED (>30 µmol/L) — the DEFINING cblC feature vs isolated MMA; "
            "METHIONINE LOW — methionine synthase blocked; "
            "MeCbl AND AdoCbl BOTH absent in fibroblasts (both arms of pathway blocked)."
        ),
        "cohort_n": n,
        "kpis": {
            "avg_mma_urine": round(sum(mma_v) / n),
            "avg_homocysteine_umol_l": round(sum(hcy_v) / n),
            "avg_methionine_umol_l": round(sum(met_v) / n, 1),
            "avg_c3_umol_l": round(sum(c3_v) / n, 1),
            "avg_ammonia_umol_l": round(sum(nh3_v) / n),
            "avg_free_carnitine": round(sum(car_v) / n, 1),
            "seizure_pct": pct(lambda p: p["seizures"]),
            "maculopathy_pct": pct(lambda p: p["maculopathy"]),
            "psychiatric_pct": pct(lambda p: p["psychiatric_features"]),
            "nbs_detected_pct": pct(lambda p: p["nbs_detected"]),
            "ohcbl_response_pct": pct(lambda p: p["ohcbl_response"]),
            "b12_responsive_pct": pct(lambda p: p["ohcbl_response"]),
        },
        "phenotype_distribution": [
            {"phenotype": "cblC Early-Onset Neonatal Severe (~45%) — HUS, pulm HTN, severe MMA+HHcy", "pct": 45},
            {"phenotype": "cblC Early-Onset Infantile (~35%) — maculopathy, IS, regression", "pct": 35},
            {"phenotype": "cblC Late-Onset Child/Adult (~15%) — psychiatric, c.482G>A", "pct": 15},
            {"phenotype": "cblC Mild/Attenuated (~5%)", "pct": 5},
        ],
        "mmachc_pathway": [
            {
                "step": "Step 1 — MMACHC (cblC): dietary cobalamin → cob(I)alamin ← BLOCK IN MMACHC LOF",
                "reaction": "CN-Cbl / OH-Cbl / Me-Cbl / Ado-Cbl + NADPH → cob(I)alamin [MMACHC decyanase/reductase]",
                "enzyme": "MMACHC (cblC; 282 aa; cytoplasmic; bifunctional decyanase + dealkylase)",
                "cofactor": "NADPH / flavin",
                "consequence_lof": "BLOCKED: cob(I)alamin CANNOT BE MADE from any dietary cobalamin → BOTH arms (A+B) downstream are cofactor-starved",
            },
            {
                "step": "Step 2A (mitochondria) — MMAB: cob(I)alamin → AdoCbl [BLOCKED secondary]",
                "reaction": "cob(I)alamin + ATP → AdoCbl + PPPi [MMAB adenosyltransferase]",
                "enzyme": "MMAB (cblB; 250 aa; mitochondrial homotrimer; 12q24.11)",
                "cofactor": "ATP",
                "consequence_lof": "BLOCKED secondary: no cob(I)alamin substrate → MMAB cannot make AdoCbl → MMAA/MMUT pathway starved",
            },
            {
                "step": "Step 2B (cytoplasm) — MMADHC: cob(I)alamin → MeCbl [BLOCKED secondary]",
                "reaction": "cob(I)alamin + 5-methylTHF → MeCbl [MMADHC; methionine synthase reductase pathway]",
                "enzyme": "MMADHC (cblD variant; cytoplasmic; 2q23.2)",
                "cofactor": "5-methylTHF",
                "consequence_lof": "BLOCKED secondary: no cob(I)alamin → MMADHC cannot supply MeCbl → methionine synthase inactive",
            },
            {
                "step": "Step 3A — MMUT: L-methylmalonyl-CoA → succinyl-CoA [BLOCKED; MMA accumulates]",
                "reaction": "L-methylmalonyl-CoA → succinyl-CoA (AdoCbl radical mechanism) [MMUT]",
                "enzyme": "MMUT (methylmalonyl-CoA mutase; 750 aa; TIM-barrel; 6p12.3; apoenzyme intact)",
                "cofactor": "AdoCbl (unavailable — secondary block via MMACHC → MMAB pathway)",
                "consequence_lof": "MMA ACCUMULATES: 200–2,000 mmol/mol Cr (PATHOGNOMONIC; lower than MMUT mut0 because MMUT apoenzyme itself is intact)",
            },
            {
                "step": "Step 3B — Methionine Synthase (MS/MTR): Hcy → Methionine [BLOCKED; HHcy accumulates]",
                "reaction": "Homocysteine + 5-methylTHF → Methionine + THF [Methionine Synthase / MTR; MeCbl cofactor]",
                "enzyme": "MTR (Methionine Synthase; 5-MTHF-homocysteine methyltransferase; 1q43; MeCbl-dependent)",
                "cofactor": "MeCbl (unavailable — secondary block via MMACHC → MMADHC pathway)",
                "consequence_lof": "Homocysteine ACCUMULATES (HHcy >30–300 µmol/L; PATHOGNOMONIC for cblC vs isolated MMA) + Methionine LOW (8–18 µmol/L)",
            },
        ],
        "mmachc_vs_isolated_mma": {
            "title": "MMACHC (cblC) vs Isolated MMA (MMUT/MMAA/MMAB) — Critical Differential",
            "note": (
                "The HALLMARK distinction: cblC causes COMBINED MMA + HHcy; isolated MMA causes ONLY MMA. "
                "Homocysteine is NORMAL in MMUT/MMAA/MMAB — ELEVATED (>30 µmol/L) in cblC. "
                "This single biomarker separates cblC from all isolated MMA subtypes. "
                "Gene panel (MMACHC/MMUT/MMAA/MMAB/MMADHC) mandatory — biochemistry distinguishes combined vs isolated."
            ),
            "comparison": [
                {"feature": "Homocysteine", "MMACHC_cblC": "ELEVATED >30–300 µmol/L (PATHOGNOMONIC)", "Isolated_MMA": "NORMAL <15 µmol/L (KEY NEGATIVE in MMUT/MMAA/MMAB)"},
                {"feature": "Methionine", "MMACHC_cblC": "LOW 8–18 µmol/L (methionine synthase blocked)", "Isolated_MMA": "NORMAL (methionine synthase unaffected)"},
                {"feature": "MMA (urine)", "MMACHC_cblC": "200–2,000 mmol/mol Cr (elevated)", "Isolated_MMA": "MMUT mut0: 2,000–10,000+; MMAA/MMAB: 200–5,000"},
                {"feature": "Maculopathy", "MMACHC_cblC": "~80% early-onset (PATHOGNOMONIC for cblC)", "Isolated_MMA": "ABSENT (no retinal involvement in MMUT/MMAA/MMAB)"},
                {"feature": "MeCbl (fibroblasts)", "MMACHC_cblC": "ABSENT — cytoplasmic pathway blocked", "Isolated_MMA": "NORMAL (cytoplasmic MeCbl arm intact)"},
                {"feature": "AdoCbl (fibroblasts)", "MMACHC_cblC": "ABSENT — mitochondrial pathway also blocked", "Isolated_MMA": "Absent in MMUT mut0; reduced in MMAA/MMAB"},
                {"feature": "OHCbl response", "MMACHC_cblC": "~75% HHcy response; ~55–60% MMA response", "Isolated_MMA": "MMAA ~60–80%; MMAB ~40–60%; MMUT ~20%"},
                {"feature": "Betaine (treatment)", "MMACHC_cblC": "Level A — MANDATORY (remethylates Hcy via BHMT)", "Isolated_MMA": "NOT used (no HHcy to treat)"},
                {"feature": "Pulmonary hypertension", "MMACHC_cblC": "~20% neonatal severe — endothelial HHcy damage", "Isolated_MMA": "NOT a feature"},
                {"feature": "Psychiatric late-onset", "MMACHC_cblC": "~75% late-onset (c.482G>A) — schizophrenia-like", "Isolated_MMA": "NOT a typical feature"},
            ],
        },
        "high_risk_situations": [
            {"situation": "VPA (Valproate)", "risk": "HIGH RISK / AVOID", "detail": "Severe carnitine depletion (VPA-carnitine conjugation) + metabolic stress + seizure-control failure; no direct MMACHC enzyme inhibition (unlike isolated MMA) but metabolic decompensation risk is high; LEV preferred"},
            {"situation": "Fasting / catabolism", "risk": "HIGH RISK", "detail": "BCAA catabolism → propionyl-CoA → MMA surge + HHcy worsens in catabolic state; IV glucose mandatory during illness; do NOT omit OHCbl/betaine"},
            {"situation": "Missed OHCbl + betaine", "risk": "HIGH RISK", "detail": "HHcy rebounds within days of omitting betaine or OHCbl; maculopathy progression risk; psychiatric crisis in late-onset; compliance monitoring essential"},
            {"situation": "Intercurrent infection", "risk": "HIGH RISK", "detail": "Catabolic → MMA surge + HHcy worsens; IV glucose protocol; continue OHCbl and betaine throughout illness"},
            {"situation": "High methionine load", "risk": "CAUTION", "detail": "Paradoxical: excess methionine → homocysteine via SAM/SAH pathway → HHcy worsens if remethylation is blocked; avoid high-methionine supplements"},
            {"situation": "Nitrous oxide anesthesia", "risk": "ABSOLUTE CI", "detail": "N2O irreversibly oxidizes MeCbl cofactor → inactivates methionine synthase → precipitates acute HHcy crisis + MMA spike; ABSOLUTE CI in ALL cblC patients"},
        ],
    }


def get_breakdown() -> dict:
    sample = _PATIENTS[:15]
    return {
        "biomarkers": [
            {"name": "C3 (Propionylcarnitine)", "normal": "<4 µmol/L", "mmachc_range": "2–30 µmol/L", "significance": "NBS PRIMARY — identical to PA, MMUT, MMAA, MMAB; cannot distinguish by NBS alone; MMA+HHcy workup mandatory on positive screen", "method": "NBS tandem MS/MS; plasma acylcarnitines"},
            {"name": "Methylmalonic Acid (urine)", "normal": "<5 mmol/mol Cr", "mmachc_range": "200–2,000 mmol/mol Cr", "significance": "ELEVATED (same category as isolated MMA) but LOWER than MMUT mut0 (5,000+) because MMUT apoenzyme is intact — only cofactor supply blocked; key distinction from isolated MMA is HHcy co-elevation", "method": "Urine organic acids (GC-MS)"},
            {"name": "Total Homocysteine (tHcy)", "normal": "<15 µmol/L", "mmachc_range": ">30–300 µmol/L (ELEVATED; PATHOGNOMONIC)", "significance": "THE DEFINING BIOMARKER of cblC vs isolated MMA — elevated in cblC because methionine synthase (MeCbl-dependent) is blocked; NORMAL in MMUT/MMAA/MMAB (isolated MMA; their cytoplasmic arm is intact); must draw SIMULTANEOUSLY with plasma MMA", "method": "Plasma tHcy (HPLC or immunoassay); simultaneous draw with plasma MMA mandatory"},
            {"name": "Methionine (plasma)", "normal": "20–40 µmol/L", "mmachc_range": "8–18 µmol/L (LOW)", "significance": "LOW because methionine synthase cannot produce methionine (no MeCbl cofactor); low methionine + high HHcy = DIAGNOSTIC PAIR for cblC and other combined MMA+HHcy disorders; distinguishes from isolated HHcy (CBS deficiency) where methionine is HIGH", "method": "Plasma amino acids (quantitative)"},
            {"name": "MeCbl (fibroblasts)", "normal": "Present (reference range lab-specific)", "mmachc_range": "ABSENT — cytoplasmic pathway blocked", "significance": "ABSENT: MMACHC cannot process cobalamin → no cob(I)alamin → MMADHC cannot supply MeCbl; distinguishes cblC from MMADHC (cblD) variants and isolated HHcy disorders", "method": "Fibroblast cobalamin uptake and distribution assay (specialized lab)"},
            {"name": "AdoCbl (fibroblasts)", "normal": "Present (reference range lab-specific)", "mmachc_range": "ABSENT — mitochondrial pathway also blocked", "significance": "ABSENT in cblC (unlike isolated MMA where only AdoCbl is absent); both MeCbl AND AdoCbl absent confirms block at MMACHC — upstream of the branch point; MMUT corrects with exogenous AdoCbl (apoenzyme intact)", "method": "Fibroblast cobalamin distribution assay"},
            {"name": "Methylcitrate (urine)", "normal": "<5 µmol/mmol Cr", "mmachc_range": "<30 µmol/mmol Cr (mild secondary)", "significance": "MILDLY elevated secondary only — NOT pathognomonic; pathognomonic in PA (50–2,000); KEY DIFFERENTIAL from PA", "method": "Urine organic acids (GC-MS)"},
            {"name": "Ammonia (plasma)", "normal": "<50 µmol/L", "mmachc_range": "50–500 µmol/L", "significance": "Secondary hyperammonemia via NAGS/CPS1 inhibition (MMA-mediated; same mechanism as isolated MMA); typically LOWER peak than MMUT mut0 or MMAB neonatal severe because MMA level is lower in cblC", "method": "Plasma ammonia (enzymatic)"},
            {"name": "Free Carnitine", "normal": "25–60 µmol/L", "mmachc_range": "8–30 µmol/L (depleted)", "significance": "DEPLETED secondary — propionylcarnitine excretion; less severe than MMUT mut0; betaine co-treatment reduces this by improving MMA control via HHcy pathway", "method": "Plasma acylcarnitines"},
            {"name": "BCAA (Leu/Ile/Val)", "normal": "Within reference range", "mmachc_range": "NORMAL", "significance": "NORMAL — KEY NEGATIVE vs DLD (combined elevation of BCAA + lactate + 2-HG distinguishes DLD); BCKDH and DLD complexes are intact in cblC", "method": "Plasma amino acids (quantitative)"},
            {"name": "OHCbl Trial Response (tHcy)", "normal": "N/A", "mmachc_range": "~75% show tHcy reduction >50% at Day 5–7", "significance": "OHCbl bypasses MMACHC by providing cob(II)alamin directly, which MMAB can partially use for AdoCbl and MMADHC for MeCbl synthesis; HHcy response rate ~75% (higher than MMA response ~55%); measure tHcy AND MMA at Day 0 and Day 7", "method": "tHcy + plasma MMA before/after 5–7 days OHCbl 1–2 mg/day IM"},
            {"name": "OHCbl Trial Response (MMA)", "normal": "N/A", "mmachc_range": "~55–60% show MMA reduction >50% at Day 5–7", "significance": "MMA response (55–60%) is lower than HHcy response (75%) in cblC because the MMA arm (mitochondrial) depends on OHCbl reaching MMAB; both should be measured to guide lifelong OHCbl dosing", "method": "Plasma MMA before/after OHCbl trial"},
        ],
        "key_variants": [
            {"variant": "c.271dupA", "cdna": "c.271dupA (p.Arg91Lys*/frameshift)", "domain": "Exon 4 — decyanase/reductase core", "severity": "Severe (null)", "note": "Most common MMACHC allele worldwide ~40–50% of all cblC alleles; exon 4 frameshift → premature stop at codon 91; no functional MMACHC protein; neonatal/infantile severe; founder in many populations"},
            {"variant": "p.Arg132*", "cdna": "c.394C>T", "domain": "Exon 4 — decyanase domain", "severity": "Severe (null)", "note": "Second most common null allele; nonsense in decyanase catalytic region; neonatal severe; pan-ethnic; no MMACHC function"},
            {"variant": "p.Arg161Gln", "cdna": "c.482G>A", "domain": "Exon 5 — reductase/dealkylase domain", "severity": "Mild (partial — late-onset allele)", "note": "Hallmark LATE-ONSET cblC allele; partial MMACHC function retained (Gln vs Arg — reduced but not zero activity); as compound het with null (c.271dupA/c.482G>A) → late-onset psychiatric phenotype (>4 years); homozygous c.482G>A → mild attenuated; critical for genetic counseling in adult psychiatry"},
            {"variant": "p.Leu116Pro", "cdna": "c.347T>C", "domain": "Exon 4 — decyanase helix", "severity": "Intermediate", "note": "Disrupts alpha-helix in decyanase domain; intermediate residual activity; early infantile onset; partial OHCbl response"},
            {"variant": "p.Pro189Thr", "cdna": "c.565C>A", "domain": "Exon 5 — reductase surface", "severity": "Mild", "note": "Surface residue; mild conformational change; residual MMACHC activity; late childhood onset; better prognosis with OHCbl"},
            {"variant": "c.271dupA/c.482G>A", "cdna": "Compound het", "domain": "Exon 4 null + exon 5 partial", "severity": "Intermediate-Late (genotype-phenotype)", "note": "Most clinically important compound het for late-onset cblC; one null allele + one partial (c.482G>A) → late-onset presentation (months to decades); c.482G>A dictates residual function; critical for late psychiatric diagnosis"},
            {"variant": "IVS1-1G>A", "cdna": "c.-1G>A (splice site)", "domain": "Intron 1 — splice acceptor", "severity": "Severe (null)", "note": "Splice site mutation; no functional MMACHC mRNA; null allele; neonatal severe; European populations"},
            {"variant": "p.Arg206*", "cdna": "c.616C>T", "domain": "Exon 5 — C-terminal reductase", "severity": "Severe (null)", "note": "Nonsense C-terminal truncation; null; neonatal-infantile severe; compound het with c.271dupA common"},
        ],
        "seizure_types": [
            {"type": "Neonatal encephalopathic seizures (MMA+HHcy crisis)", "pct": 70, "note": "Multifocal/generalized; EEG burst-suppression pattern; combined MMA+HHcy-driven encephalopathy; treat underlying metabolic crisis first — OHCbl IM + betaine + IV glucose; LEV adjunct"},
            {"type": "Infantile spasms (West syndrome)", "pct": 40, "note": "Early-onset cblC especially; hypsarrhythmia on EEG; ACTH or vigabatrin when metabolically stabilized; folinic acid may augment response in cblC (methylation cycle support); LEV adjunct"},
            {"type": "Focal cortical seizures", "pct": 30, "note": "Cortical injury from HHcy-related endothelial damage + MMA mitochondrial toxicity; LEV first-line; improve with metabolic treatment"},
            {"type": "Myoclonic seizures", "pct": 25, "note": "Metabolic-triggered; MMA/HHcy spikes; LEV first-line AED; correlate with metabolic control — improve with OHCbl+betaine compliance"},
            {"type": "Epileptic encephalopathy / status epilepticus", "pct": 20, "note": "Severe early-onset cases; continuous EEG mandatory in neonatal severe; IV BZD + IV glucose + OHCbl IM + betaine"},
            {"type": "Absence-like (late-onset)", "pct": 10, "note": "Late-onset cblC (c.482G>A); often co-existing with psychiatric symptoms; metabolic treatment reduces seizure burden; LEV preferred over VPA"},
        ],
        "metabolic_triggers": [
            {"trigger": "Fasting / prolonged NPO", "pct": 68, "mechanism": "BCAA catabolism → propionyl-CoA → MMA surge + HHcy worsens (remethylation impaired during catabolism); IV glucose + betaine + OHCbl mandatory"},
            {"trigger": "Intercurrent illness / fever", "pct": 60, "mechanism": "Catabolic state → combined MMA+HHcy surge; IV glucose protocol; continue OHCbl and betaine; maculopathy can acutely worsen"},
            {"trigger": "Missed OHCbl and/or betaine doses", "pct": 45, "mechanism": "HHcy rebounds within 48–72 hours of omitting betaine; MMA rebounds within 3–7 days of omitting OHCbl; dual compliance monitoring essential"},
            {"trigger": "Nitrous oxide anesthesia", "pct": 100, "mechanism": "ABSOLUTE CI: N2O irreversibly oxidizes Cbl(I)/Cbl(II) → inactivates methionine synthase → acute HHcy crisis + MMA spike; precipitates life-threatening decompensation in cblC"},
            {"trigger": "High methionine diet / supplements", "pct": 15, "mechanism": "Excess methionine → SAM/SAH pathway → excess homocysteine produced; combined with blocked remethylation → HHcy worsens"},
        ],
        "high_risk_drugs": [
            {"drug": "Nitrous oxide (N2O)", "risk": "ABSOLUTE CI", "mechanism": "Irreversibly oxidizes methionine synthase cofactor → acute methionine synthase failure → HHcy crisis + MMA spike: LIFE-THREATENING; use halogenated volatile agents only for anesthesia in cblC"},
            {"drug": "Valproate (VPA)", "risk": "HIGH RISK / AVOID", "mechanism": "Severe carnitine depletion (VPA-carnitine conjugation) + metabolic stress + worsens combined MMA+HHcy cascade; LEV (levetiracetam) preferred as first-line AED for ALL cblC seizures"},
            {"drug": "Methotrexate / antifolates", "risk": "HIGH RISK / AVOID", "mechanism": "Depletes 5-methylTHF → worsens methylation cycle impairment → HHcy rises further; folinic acid supplementation is already needed in cblC"},
            {"drug": "Fasting / NPO without supplementation", "risk": "HIGH RISK", "mechanism": "BCAA catabolism → MMA surge + HHcy worsens; always IV dextrose + betaine + OHCbl during illness or fasting"},
            {"drug": "High methionine supplements", "risk": "CAUTION", "mechanism": "Paradoxical HHcy worsening via SAM/SAH overflow when remethylation is blocked; monitor tHcy if methionine given"},
        ],
        "treatments": [
            {"treatment": "Hydroxocobalamin (OHCbl) 1–2 mg/day IM", "evidence": "Level A", "response_pct": 75, "note": "Trial ALL cblC patients; ~75% HHcy response (tHcy falls >50%); ~55–60% MMA response; OHCbl bypasses MMACHC by providing reduced cobalamin directly to MMAB (mitochondria) and MMADHC (cytoplasm) pathways; lifelong IM preferred; oral maintenance investigated in partial responders"},
            {"treatment": "Betaine (trimethylglycine) 100–200 mg/kg/day orally", "evidence": "Level A", "response_pct": 80, "note": "MANDATORY co-treatment with OHCbl; remethylates homocysteine via BHMT (betaine-homocysteine methyltransferase; cobalamin-independent liver enzyme); Betaine + OHCbl together achieve better HHcy control than either alone; prevents maculopathy progression; psychiatric symptom stabilization in late-onset"},
            {"treatment": "Folinic acid / 5-methylTHF", "evidence": "Level B", "response_pct": 55, "note": "5-methylTHF depleted in cblC because methylene-THF trap operates when methionine synthase is blocked; folinic acid bypasses the blocked folate methylation step; may improve seizure control in infantile spasms; dose: 0.5–5 mg/day"},
            {"treatment": "L-Carnitine 50–100 mg/kg/day", "evidence": "Level B", "response_pct": 70, "note": "Secondary depletion via propionylcarnitine excretion; less severe than isolated MMA because MMA is lower in cblC; supplement to prevent cardiac complications; betaine co-treatment indirectly reduces carnitine loss by improving MMA"},
            {"treatment": "Protein restriction (moderate)", "evidence": "Level C/Supportive", "response_pct": 50, "note": "Less critical than isolated MMA (MMA levels lower in cblC); moderate natural protein restriction (1.2–2.0 g/kg/day); no need for MMA-free formula unless MMA severely elevated; avoid high methionine"},
            {"treatment": "Methionine supplementation", "evidence": "Level C", "response_pct": 40, "note": "If plasma methionine persistently <15 µmol/L after OHCbl+betaine; supports protein synthesis; monitor tHcy to avoid worsening HHcy via SAM/SAH overflow"},
            {"treatment": "IV Glucose GIR 8–12 (acute crisis)", "evidence": "Level A", "response_pct": 80, "note": "Acute catabolic crisis first intervention; stops BCAA catabolism → MMA surge halted; continue OHCbl IM and betaine throughout; IV betaine if oral route unavailable"},
            {"treatment": "Ammonia scavengers (Na benzoate / Na phenylacetate)", "evidence": "Level B", "response_pct": 65, "note": "If NH3 >200 µmol/L; secondary hyperammonemia from MMA-NAGS/CPS1 inhibition; same protocol as isolated MMA"},
            {"treatment": "Nitrous oxide avoidance (anesthesia protocol)", "evidence": "Level A", "response_pct": 100, "note": "ABSOLUTE CI: N2O inactivates methionine synthase cofactor; all surgical teams must be informed; use halogenated volatile agents; IV OHCbl perioperatively; metabolic team present"},
            {"treatment": "VPA (Valproate)", "evidence": "HIGH RISK / AVOID", "response_pct": 0, "note": "HIGH RISK (not absolute CI as in isolated MMA) but should be AVOIDED: severe carnitine depletion + metabolic stress + seizure risk; LEV is first-line AED for all cblC seizure types; VPA alternatives: LEV, CLB, VNS"},
        ],
        "patient_sample": sample,
    }


def get_definitions() -> dict:
    return {
        "gene_card": {
            "Gene": "MMACHC (Methylmalonyl-CoA/Homocysteine Cobalamin Processing Chaperone; cblC gene product)",
            "Full Name": "Methylmalonyl-CoA/Homocysteine Cobalamin Chaperone (Complementation Group cblC)",
            "Chromosome": "1p34.1",
            "Inheritance": "Autosomal Recessive (AR)",
            "OMIM Gene": "*609831",
            "OMIM Disease": "#277400 (Methylmalonic Acidemia and Homocystinuria, cblC type)",
            "Protein Size": "282 amino acids; cytoplasmic; bifunctional decyanase + dealkylase/reductase; ubiquitous",
            "Structure": "Cytoplasmic monomer; TonB-like N-terminal domain; C-terminal reductase-like domain; NADPH-dependent",
            "Reaction": "Dietary Cbl (CN-Cbl/OH-Cbl/Me-Cbl/Ado-Cbl) + NADPH → cob(I)alamin + CN⁻/Me/Ado (decyanation/dealkylation)",
            "Pathway Position": "Step 1 — MOST UPSTREAM intracellular cobalamin processing enzyme; upstream of MMAB (arm A: MMA) and MMADHC/MS (arm B: HHcy)",
            "Prevalence": "~1:50,000–100,000; most common inborn error of intracellular cobalamin metabolism",
            "NBS": "C3 (propionylcarnitine) elevated — same trigger as isolated MMA; urine organic acids + tHcy MANDATORY",
            "Key Positive": "Urine MMA 200–2,000 mmol/mol Cr + tHcy >30 µmol/L (COMBINED elevation = cblC hallmark); maculopathy ~80% early-onset",
            "Key Negative": "BCAA NORMAL (vs DLD/MSUD); methylcitrate NOT pathognomonic (vs PA); N2O ABSOLUTE CI (methionine synthase inactivation)",
        },
        "key_concepts": [
            {
                "concept": "Why MMACHC causes BOTH MMA and HHcy — the upstream branch point",
                "explanation": (
                    "MMACHC is upstream of the branch point where cobalamin is partitioned to two pathways: "
                    "(A) mitochondrial AdoCbl pathway (→ MMAB → MMUT → MMA catabolism) and "
                    "(B) cytoplasmic MeCbl pathway (→ MMADHC → Methionine Synthase → Hcy→Met remethylation). "
                    "All dietary cobalamin forms must pass through MMACHC first — it is the GATEKEEPER. "
                    "MMACHC LOF → cob(I)alamin pool collapses → BOTH downstream arms are simultaneously "
                    "cofactor-starved: (A) no AdoCbl → MMUT cofactor-empty → MMA ↑↑; "
                    "(B) no MeCbl → methionine synthase inactive → Hcy accumulates (HHcy ↑↑) + Met LOW. "
                    "Isolated MMA diseases (MMUT/MMAA/MMAB) only affect arm A — arm B (cytoplasmic) "
                    "is intact → Hcy is NORMAL. cblC is COMBINED because the block is upstream of the branch."
                ),
            },
            {
                "concept": "Homocysteine elevation — the PATHOGNOMONIC cblC biomarker separating it from isolated MMA",
                "explanation": (
                    "Total homocysteine (tHcy) >30 µmol/L in the context of elevated MMA = cblC until proven otherwise. "
                    "This is the single most important differentiating biomarker: "
                    "• MMUT/MMAA/MMAB (isolated MMA): Hcy NORMAL (<15 µmol/L) — arm B (cytoplasmic MeCbl) is INTACT. "
                    "• MMACHC (cblC): Hcy ELEVATED (30–300 µmol/L) — arm B BLOCKED. "
                    "Draw tHcy simultaneously with plasma MMA — always do a combined metabolic screen. "
                    "NBS alone cannot distinguish cblC from isolated MMA (both elevate C3). "
                    "Second-tier tHcy on NBS dried blood spots is the key step. "
                    "Also measure methionine: LOW in cblC (methionine synthase blocked) → high Hcy + low Met = cblC diagnostic pair."
                ),
            },
            {
                "concept": "OHCbl mechanism in cblC — partial bypass of the blocked MMACHC step",
                "explanation": (
                    "Hydroxocobalamin (OHCbl) in pharmacological doses bypasses MMACHC by providing "
                    "cob(II)alamin/cob(I)alamin directly to downstream enzymes: "
                    "(A) MMAB can directly use cob(I)alamin (reduced OHCbl) → AdoCbl → MMUT → MMA ↓; "
                    "(B) MMADHC + MTR can use cob(II)alamin → MeCbl → methionine synthase → Hcy ↓. "
                    "HHcy response (~75%) is HIGHER than MMA response (~55–60%) because the cytoplasmic arm "
                    "has higher apparent affinity for the bypass substrate. "
                    "OHCbl does NOT restore MMACHC enzyme function — it provides a pharmacological cobalamin flood. "
                    "Betaine (Level A) MUST be co-administered: remethylates Hcy via BHMT (cobalamin-independent "
                    "liver pathway) → acts in parallel with OHCbl for HHcy control."
                ),
            },
            {
                "concept": "Maculopathy in cblC — pathognomonic retinal finding; mechanism and monitoring",
                "explanation": (
                    "Macular/retinal pigment epitheliopathy occurs in ~80% of early-onset cblC and is a "
                    "PATHOGNOMONIC clinical finding for cblC (absent in isolated MMA — MMUT/MMAA/MMAB). "
                    "Mechanism: HHcy causes endothelial dysfunction and oxidative stress in retinal vessels + "
                    "MMA-related mitochondrial dysfunction in RPE cells → photoreceptor degeneration. "
                    "Presentation: nystagmus, visual inattention, reduced ERG in infants; in late-onset may "
                    "be the presenting sign before neurocognitive symptoms. "
                    "Management: OHCbl + betaine arrest or slow progression; do NOT cure existing damage. "
                    "Monitoring: ophthalmology at diagnosis + every 6–12 months; ERG annually; "
                    "metabolic control is the primary preventive measure."
                ),
            },
            {
                "concept": "Late-onset cblC and the c.482G>A allele — psychiatric misdiagnosis risk",
                "explanation": (
                    "The c.482G>A (p.Arg161Gln) MMACHC allele retains partial enzymatic function → "
                    "when combined with a null allele (compound het c.271dupA/c.482G>A), residual MMACHC "
                    "activity maintains partial cobalamin processing into childhood/adulthood. "
                    "Late-onset cblC presentation (>4 years, often teenagers/adults): "
                    "schizophrenia-like psychosis, dementia, myelopathy, ataxia — often misdiagnosed as "
                    "primary psychiatric illness for years. "
                    "Clue: tHcy elevated + MMA elevated in a psychiatric patient = cblC until proven otherwise. "
                    "Treatment: OHCbl + betaine achieves dramatic psychiatric improvement if started early — "
                    "residual partial MMACHC function + pharmacological bypass is effective. "
                    "Genetic counseling: c.482G>A siblings of known cblC patients need proactive tHcy screening."
                ),
            },
            {
                "concept": "Betaine mechanism in cblC — cobalamin-independent HHcy remethylation",
                "explanation": (
                    "Betaine (trimethylglycine, TMG) remethylates homocysteine via the enzyme "
                    "betaine-homocysteine methyltransferase (BHMT; liver-specific), which is INDEPENDENT "
                    "of cobalamin. This is the critical rationale for betaine as Level A co-treatment: "
                    "• Reaction: Hcy + Betaine (trimethylglycine) → Methionine + Dimethylglycine [BHMT]. "
                    "• No cobalamin cofactor required → works even when MMACHC is completely absent. "
                    "• Betaine + OHCbl together reduce tHcy more than either alone: "
                    "   OHCbl: provides bypass cobalamin → partial MeCbl → methionine synthase partially restored. "
                    "   Betaine: independent BHMT remethylation → reduces Hcy regardless of OHCbl response. "
                    "• Dose: 100–200 mg/kg/day orally (powders/solutions); monitor tHcy monthly initially."
                ),
            },
            {
                "concept": "Nitrous oxide ABSOLUTE CI in cblC — mechanism and anesthesia protocol",
                "explanation": (
                    "Nitrous oxide (N2O) IRREVERSIBLY oxidizes the cobalt centre of cobalamin (Co+ → Co3+), "
                    "permanently inactivating methionine synthase (MTR) which requires Co+ (cob(I)alamin) as cofactor. "
                    "In cblC patients, methionine synthase is ALREADY compromised (no MeCbl supply). "
                    "N2O exposure → complete methionine synthase failure → acute HHcy crisis (tHcy can spike to "
                    "500–1000 µmol/L) + acute MMA spike (catabolism) → life-threatening neurological crisis. "
                    "Anesthesia protocol for cblC: "
                    "• NEVER use N2O (ABSOLUTE CI). "
                    "• Use halogenated volatile agents (sevoflurane, isoflurane) or IV propofol. "
                    "• Perioperative IV OHCbl 1 mg IM day of surgery and next day. "
                    "• IV glucose GIR 6–8 during NPO. "
                    "• Metabolic team co-management for any elective procedure. "
                    "• All surgical/anesthetic teams MUST be notified of N2O contraindication BEFORE any procedure."
                ),
            },
        ],
        "diagnostic_thresholds": [
            {"parameter": "Total Homocysteine (tHcy)", "threshold": ">30 µmol/L with elevated MMA = cblC; >100 µmol/L = severe cblC crisis", "action": "Send MMACHC gene sequencing; start OHCbl 1 mg IM immediately + betaine 100 mg/kg/day; simultaneous plasma methionine"},
            {"parameter": "Urine MMA + tHcy combined", "threshold": "MMA >200 mmol/mol Cr AND tHcy >30 µmol/L = COMBINED = cblC until proven otherwise", "action": "Gene panel: MMACHC/MMADHC/MMUT/MMAA/MMAB; ophthalmology referral (maculopathy); start OHCbl + betaine pending confirmation"},
            {"parameter": "Methionine (plasma)", "threshold": "<15 µmol/L in context of high Hcy = methionine synthase blocked = cblC or MMADHC", "action": "Confirm with tHcy + MMA + gene panel; consider methionine supplementation if <12 µmol/L"},
            {"parameter": "OHCbl trial (tHcy response)", "threshold": "tHcy reduction >50% at Day 5–7 = OHCbl-responsive", "action": "Lifelong OHCbl IM; confirmed + betaine combination; recheck tHcy monthly"},
            {"parameter": "OHCbl trial (MMA response)", "threshold": "MMA reduction >50% at Day 5–7 = MMA-responsive", "action": "Combine with tHcy response data to classify: dual responder / partial MMA-responder / HHcy-only responder"},
            {"parameter": "Ammonia", "threshold": "NH3 >200 µmol/L = start ammonia scavengers; >400 = HD/CRRT", "action": "IV Na benzoate 250 mg/kg; IV glucose GIR 8–12; OHCbl IM; betaine via NG/IV if oral unavailable"},
            {"parameter": "tHcy monitoring on treatment", "threshold": "Target tHcy <50 µmol/L (ideally <30 µmol/L) on OHCbl + betaine", "action": "Monthly tHcy in first year; 3-monthly if stable; adjust betaine dose to achieve target"},
        ],
        "differential_diagnosis": [
            {"disease": "MMUT (mut-type) — methylmalonyl-CoA mutase apoenzyme defect", "distinguishing": "Isolated MMA ONLY — Hcy NORMAL (<15 µmol/L); MMUT fibroblast activity ABSENT without AdoCbl (NOT correctable in mut0); no maculopathy; gene: MMUT (6p12.3)"},
            {"disease": "MMAA (cblA) — GTPase chaperone delivery defect", "distinguishing": "Isolated MMA ONLY — Hcy NORMAL; OHCbl STRONG response ~60–80%; MMAB enzyme NORMAL; MMAA (4p14); no maculopathy; no HHcy"},
            {"disease": "MMAB (cblB) — adenosyltransferase synthesis defect", "distinguishing": "Isolated MMA ONLY — Hcy NORMAL; OHCbl MODERATE response ~40–60%; MMAB enzyme ABSENT; MMAB (12q24.11); no maculopathy; no HHcy"},
            {"disease": "MMADHC (cblD) — combined or isolated MMA variant", "distinguishing": "cblD can cause: (A) isolated MMA (cblD-MMA variant) or (B) combined MMA+HHcy (cblD-combined) — depending on allele; MMADHC (2q23.2); MeCbl absent (B-type) or AdoCbl absent (A-type); gene panel distinguishes from MMACHC"},
            {"disease": "CBS deficiency (classical homocystinuria)", "distinguishing": "ELEVATED Hcy (same) BUT: methionine HIGH (vs LOW in cblC); NO MMA elevation; NO C3 on NBS; marfanoid habitus; lens dislocation; autosomal recessive CBS (21q22.3); treat with pyridoxine + betaine + methionine restriction"},
            {"disease": "PCCA/PCCB (Propionic Acidemia)", "distinguishing": "Methylcitrate PATHOGNOMONIC (50–2,000 µmol/mmol Cr) vs cblC absent/mild; NO MMA elevation (MMA is the cblC biomarker, not PA); NO Hcy elevation; no maculopathy; gene panel: PCCA (13q32.3) / PCCB (3q22.3)"},
            {"disease": "DLD (dihydrolipoamide dehydrogenase, E3)", "distinguishing": "COMBINED elevation: lactate + BCAA (Leu/Ile/Val) + 2-HGA + mild glycine; BCAA NORMAL in cblC (KEY NEGATIVE); DLD (7q31.1); no maculopathy"},
        ],
    }
