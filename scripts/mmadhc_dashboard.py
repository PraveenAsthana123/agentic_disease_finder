#!/usr/bin/env python3
"""MMADHC (Methylmalonic Acidemia with Homocystinuria, cblD type) Epilepsy Dashboard.

MMADHC encodes a 296-aa bifunctional cytoplasmic/mitochondrial DISTRIBUTOR protein that
operates DOWNSTREAM of MMACHC in the intracellular cobalamin pathway. MMADHC receives
cob(I)alamin from MMACHC and channels it to TWO SEPARATE ARMS:

  (A) Mitochondrial arm — C-terminus: MMADHC escorts cob(I)alamin into mitochondria
      → MMAB synthesizes AdoCbl → MMUT (methylmalonyl-CoA mutase) uses AdoCbl as cofactor
      → L-methylmalonyl-CoA → succinyl-CoA (MMA catabolism).

  (B) Cytoplasmic arm — N-terminus: MMADHC retains cob(I)alamin in cytoplasm
      → MeCbl synthesized → Methionine Synthase (MTR) uses MeCbl cofactor
      → Homocysteine → Methionine (Hcy remethylation).

MMADHC UNIQUE FEATURE — GENOTYPE PREDICTS BIOCHEMICAL SUBTYPE:
  cblD-MMA only (C-terminus variants): mitochondrial arm blocked → isolated MMA; Hcy NORMAL
  cblD-HHcy only (N-terminus variants): cytoplasmic arm blocked → isolated HHcy; MMA NORMAL
  cblD-Combined (central/global variants): BOTH arms blocked → combined MMA + HHcy (like cblC/MMACHC)

This is the ONLY cobalamin disorder where the allele location predicts which metabolic arm is blocked.

  MMADHC PROTEIN BIOLOGY:
    296 amino acids; cytoplasmic with mitochondrial-targeting sequence.
    N-terminal domain (cytoplasmic): mediates MeCbl branch — directs cob(I)alamin to methionine
    synthase reductase (MTRR) pathway for MeCbl production.
    C-terminal domain (mitochondrial-targeting signal): mediates AdoCbl branch — directs
    cob(I)alamin into mitochondria to MMAB for AdoCbl synthesis.
    MMADHC LOF pattern depends on which domain is damaged.

  cblD-MMA (C-terminus/mitochondrial variants):
    Only mitochondrial arm blocked; cytoplasmic arm (MeCbl/Hcy remethylation) INTACT.
    → MMA elevated; Homocysteine NORMAL.
    Biochemically identical to MMAA (cblA) / MMAB (cblB) in terms of metabolites.
    KEY DISTINCTION from MMAA/MMAB: MMADHC is the distributor; MMAA/MMAB are chaperone/synthase.
    AdoCbl absent in fibroblasts; MeCbl NORMAL in fibroblasts (cytoplasmic arm intact).

  cblD-HHcy (N-terminus/cytoplasmic variants):
    Only cytoplasmic arm blocked; mitochondrial arm (AdoCbl/MMA catabolism) INTACT.
    → Homocysteine elevated; MMA NORMAL (or minimally elevated).
    Biochemically similar to MTR (methionine synthase) or MTRR (methionine synthase reductase) deficiency.
    MeCbl absent in fibroblasts; AdoCbl NORMAL in fibroblasts (mitochondrial arm intact).
    KEY DISTINCTION: no C3 elevation on NBS (MMA not elevated).

  cblD-Combined (central domain variants):
    Both arms blocked.
    → Combined MMA + HHcy (phenotypically similar to MMACHC/cblC).
    Both MeCbl and AdoCbl absent in fibroblasts.
    Gene panel distinguishes from cblC (MMACHC).

  KEY BIOMARKERS BY SUBTYPE:
    cblD-MMA: C3↑, MMA↑ 200–3000 mmol/mol Cr, Hcy NORMAL (<15), MeCbl NORMAL, AdoCbl absent
    cblD-HHcy: C3 normal/borderline, MMA NORMAL, Hcy↑ 30–200 µmol/L, MeCbl absent, AdoCbl NORMAL
    cblD-Combined: C3↑, MMA↑, Hcy↑, both MeCbl and AdoCbl absent

  KEY VARIANTS IN MMADHC:
    - p.Arg132Gln (c.395G>A): C-terminus; cblD-MMA subtype; most common cblD-MMA variant
    - p.Trp135* (c.405G>A): C-terminus nonsense; null C-terminal → cblD-MMA; severe
    - p.Arg197Ter (c.589C>T): C-terminus; cblD-MMA variant; neonatal severe
    - p.Gly132Arg (N-terminus variant): cytoplasmic domain → cblD-HHcy; isolated HHcy
    - c.1A>G (start codon loss): near-null; cblD-Combined; loss of both domains
    - Large del exon 3–4: null; cblD-Combined depending on extent

  TREATMENT:
    OHCbl 1–2 mg/day IM (Level A): trial ALL subtypes; bypasses MMADHC.
      cblD-MMA: ~50–70% MMA response (similar to MMAB); HHcy not present.
      cblD-HHcy: ~60–80% HHcy response; MMA not elevated.
      cblD-Combined: ~55–70% HHcy response; ~40–60% MMA response.
    Betaine 100–200 mg/kg/day (Level A): ONLY if HHcy component present (cblD-HHcy or cblD-Combined).
      No betaine needed for cblD-MMA (Hcy is NORMAL).
    Low protein / MMA-free formula (Level A): for MMA component (cblD-MMA or cblD-Combined).
    L-Carnitine (Level A): secondary depletion in MMA-carrying subtypes.
    Folinic acid (Level B): if HHcy component present.
    VPA — ABSOLUTE CI for MMA subtypes; HIGH RISK for HHcy subtypes.
    Fasting — HIGH RISK in MMA-carrying subtypes (MMA surge).
    N2O — ABSOLUTE CI in HHcy subtypes (inactivates methionine synthase).

  OMIM: Gene *607269 · Disease #277410 (Methylmalonic Acidemia and Homocystinuria, cblD type)
  Chromosome: 2q23.2 · Inheritance: AR · Prevalence: ~1:500,000–1,000,000 (rare; <200 cases)
"""

from __future__ import annotations
import random

random.seed(53)   # reproducible synthetic cohort


# ── helpers ───────────────────────────────────────────────────────────────────

def _pid(i: int) -> str:
    return f"MMADHC-{i:03d}"


def _sex(i: int) -> str:
    return "M" if i % 2 == 0 else "F"


def _subtype(i: int) -> str:
    """
    cblD-MMA only (~40%) — C-terminus variants; isolated MMA; 16 pts
    cblD-HHcy only (~35%) — N-terminus variants; isolated HHcy; 14 pts
    cblD-Combined (~25%) — global/central variants; combined MMA+HHcy; 10 pts
    """
    if i <= 16:  return "cblD-MMA only"
    if i <= 30:  return "cblD-HHcy only"
    return "cblD-Combined"


def _phenotype(i: int, subtype: str) -> str:
    """Phenotypic severity within each subtype."""
    if subtype == "cblD-MMA only":
        opts = ["Neonatal Severe MMA", "Classic Infantile MMA", "Late-Onset Episodic MMA", "Mild Attenuated MMA"]
        weights = [0.25, 0.45, 0.20, 0.10]
    elif subtype == "cblD-HHcy only":
        opts = ["Neonatal Severe HHcy", "Infantile HHcy + Neurological", "Late-Onset HHcy + Psychiatric", "Mild Attenuated HHcy"]
        weights = [0.20, 0.45, 0.25, 0.10]
    else:
        opts = ["Neonatal Severe Combined", "Classic Infantile Combined", "Late-Onset Combined"]
        weights = [0.40, 0.45, 0.15]
    return random.choices(opts, weights=weights)[0]


def _genotype(i: int, subtype: str) -> str:
    if subtype == "cblD-MMA only":
        variants = [
            "p.Arg132Gln/p.Arg132Gln",       # C-terminus; most common cblD-MMA
            "p.Trp135*/p.Arg132Gln",           # C-terminus null/missense
            "p.Arg197Ter/p.Arg197Ter",         # C-terminus null; severe
            "p.Trp135*/p.Trp135*",             # C-terminus biallelic null; severe neonatal
            "p.Arg132Gln/p.Arg197Ter",         # C-terminus compound het
            "c.IVS5plus1G-A/p.Arg132Gln",     # C-terminus splice/missense
        ]
    elif subtype == "cblD-HHcy only":
        variants = [
            "p.Gly132Arg/p.Gly132Arg",        # N-terminus; isolated HHcy
            "p.Leu49*/p.Gly132Arg",            # N-terminus null/missense
            "p.Met1Val/p.Gly132Arg",           # Start codon/N-terminus; near-null
            "p.Gly132Arg/p.Ala58Val",          # N-terminus compound het
            "p.Leu49*/p.Leu49*",              # N-terminus biallelic null; severe
            "c.IVS2minus2A-G/p.Gly132Arg",   # N-terminus splice/missense
        ]
    else:
        variants = [
            "c.1A>G/c.1A>G",                  # start codon loss; near-null; combined
            "c.1A>G/p.Gly132Arg",             # near-null/N-terminus; combined
            "Large-del-ex3-4/p.Arg132Gln",    # null/C-terminus; combined
            "p.Asp174Asn/p.Arg197Ter",        # central/C-terminus; combined
        ]
    return random.choice(variants)


def _mma_urine(subtype: str, pheno: str) -> float:
    """MMA urine mmol/mol Cr — elevated only if MMA arm affected."""
    if "HHcy only" in subtype:
        return round(random.uniform(0, 10), 1)   # NORMAL
    if "Neonatal" in pheno:
        return round(random.uniform(800, 3000), 0)
    if "Infantile" in pheno or "Classic" in pheno:
        return round(random.uniform(200, 1500), 0)
    if "Late-Onset" in pheno or "Episodic" in pheno:
        return round(random.uniform(80, 500), 0)
    return round(random.uniform(40, 200), 0)


def _plasma_mma(subtype: str, pheno: str) -> float:
    if "HHcy only" in subtype:
        return round(random.uniform(0, 8), 1)
    if "Neonatal" in pheno:
        return round(random.uniform(300, 1200), 0)
    if "Infantile" in pheno or "Classic" in pheno:
        return round(random.uniform(80, 500), 0)
    return round(random.uniform(30, 180), 0)


def _homocysteine(subtype: str, pheno: str) -> float:
    """Total Hcy µmol/L — elevated only if HHcy arm affected."""
    if "MMA only" in subtype:
        return round(random.uniform(5, 14), 1)   # NORMAL
    if "Neonatal" in pheno:
        return round(random.uniform(100, 250), 0)
    if "Infantile" in pheno or "Classic" in pheno:
        return round(random.uniform(50, 180), 0)
    if "Late-Onset" in pheno or "Psychiatric" in pheno:
        return round(random.uniform(30, 120), 0)
    return round(random.uniform(20, 60), 0)


def _methionine(subtype: str, pheno: str) -> float:
    """Plasma methionine µmol/L — low if HHcy arm blocked."""
    if "MMA only" in subtype:
        return round(random.uniform(20, 40), 1)  # NORMAL
    if "Neonatal" in pheno:
        return round(random.uniform(5, 13), 1)
    if "Infantile" in pheno or "Classic" in pheno:
        return round(random.uniform(9, 18), 1)
    return round(random.uniform(12, 25), 1)


def _c3(subtype: str, pheno: str) -> float:
    if "HHcy only" in subtype:
        return round(random.uniform(0.5, 3.5), 1)  # normal/borderline
    if "Neonatal" in pheno:
        return round(random.uniform(6, 25), 1)
    if "Infantile" in pheno or "Classic" in pheno:
        return round(random.uniform(3, 15), 1)
    return round(random.uniform(1.5, 8), 1)


def _ammonia(subtype: str, pheno: str) -> int:
    if "HHcy only" in subtype:
        return random.randint(15, 60)
    if "Neonatal" in pheno:
        return random.randint(100, 450)
    if "Infantile" in pheno or "Classic" in pheno:
        return random.randint(40, 180)
    return random.randint(15, 70)


def _carnitine(subtype: str, pheno: str) -> float:
    if "HHcy only" in subtype:
        return round(random.uniform(20, 50), 1)  # near normal
    if "Neonatal" in pheno:
        return round(random.uniform(8, 22), 1)
    return round(random.uniform(12, 35), 1)


def _onset_months(subtype: str, pheno: str) -> int:
    if "Neonatal" in pheno:
        return random.randint(0, 1)
    if "Infantile" in pheno or "Classic" in pheno:
        return random.randint(1, 18)
    if "Late-Onset" in pheno or "Episodic" in pheno or "Psychiatric" in pheno:
        return random.randint(36, 300)
    return random.randint(12, 120)


def _ohcbl_response(subtype: str, pheno: str) -> bool:
    """Response rates vary by subtype."""
    if subtype == "cblD-MMA only":
        return random.random() < 0.60   # ~50–70%
    if subtype == "cblD-HHcy only":
        return random.random() < 0.72   # ~60–80%
    return random.random() < 0.58       # Combined ~55%


def _seizures(pheno: str, subtype: str) -> bool:
    if "Neonatal" in pheno:
        return random.random() < 0.80
    if "Infantile" in pheno or "Classic" in pheno:
        return random.random() < 0.55
    if "Psychiatric" in pheno or "Late-Onset" in pheno:
        return random.random() < 0.30
    return random.random() < 0.15


def _psychiatric(pheno: str) -> bool:
    if "Psychiatric" in pheno or "Late-Onset" in pheno:
        return random.random() < 0.65
    return random.random() < 0.10


def _aed(sz: bool) -> str:
    if not sz:
        return "None"
    return random.choice(["LEV", "LEV + CLB", "LEV monotherapy", "LEV + VNS", "LEV + folinic acid"])


def _nbs(subtype: str, pheno: str) -> bool:
    if "HHcy only" in subtype:
        return random.random() < 0.30   # C3 not reliably elevated → often NBS-missed
    if "Neonatal" in pheno:
        return random.random() < 0.72
    return random.random() < 0.82


def _diag_months(pheno: str, onset: int) -> int:
    if "Neonatal" in pheno:
        return onset + random.randint(0, 3)
    if "Psychiatric" in pheno or "Late-Onset" in pheno:
        return onset + random.randint(12, 96)  # late-onset often misdiagnosed
    return onset + random.randint(1, 12)


# ── build cohort ──────────────────────────────────────────────────────────────

_PATIENTS = []
for _i in range(1, 41):
    sub  = _subtype(_i)
    ph   = _phenotype(_i, sub)
    sz   = _seizures(ph, sub)
    ons  = _onset_months(sub, ph)
    _PATIENTS.append({
        "patient_id":                _pid(_i),
        "sex":                       _sex(_i),
        "subtype":                   sub,
        "phenotype":                 ph,
        "age_at_onset_months":       ons,
        "age_at_diagnosis_months":   _diag_months(ph, ons),
        "genotype":                  _genotype(_i, sub),
        "mma_urine_mmol_molCr":      _mma_urine(sub, ph),
        "plasma_mma_umol_l":         _plasma_mma(sub, ph),
        "total_homocysteine_umol_l": _homocysteine(sub, ph),
        "methionine_umol_l":         _methionine(sub, ph),
        "c3_umol_l":                 _c3(sub, ph),
        "ammonia_umol_l":            _ammonia(sub, ph),
        "free_carnitine_umol_l":     _carnitine(sub, ph),
        "seizures":                  sz,
        "psychiatric_features":      _psychiatric(ph),
        "nbs_detected":              _nbs(sub, ph),
        "ohcbl_response":            _ohcbl_response(sub, ph),
        "aed":                       _aed(sz),
    })


# ── API functions ─────────────────────────────────────────────────────────────

def get_overview() -> dict:
    n     = len(_PATIENTS)
    mma_v = [p["mma_urine_mmol_molCr"] for p in _PATIENTS]
    hcy_v = [p["total_homocysteine_umol_l"] for p in _PATIENTS]
    met_v = [p["methionine_umol_l"] for p in _PATIENTS]
    c3_v  = [p["c3_umol_l"] for p in _PATIENTS]
    nh3_v = [p["ammonia_umol_l"] for p in _PATIENTS]
    car_v = [p["free_carnitine_umol_l"] for p in _PATIENTS]

    def pct(pred): return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    mma_pts   = [p for p in _PATIENTS if p["subtype"] == "cblD-MMA only"]
    hhcy_pts  = [p for p in _PATIENTS if p["subtype"] == "cblD-HHcy only"]
    comb_pts  = [p for p in _PATIENTS if p["subtype"] == "cblD-Combined"]

    return {
        "gene": "MMADHC",
        "full_name": "Methylmalonic Acidemia and Homocystinuria type D gene (cblD gene product)",
        "chromosome": "2q23.2",
        "inheritance": "AR",
        "omim_gene": "*607269",
        "omim_disease": "#277410",
        "protein_size": "296 aa; cytoplasmic N-terminus + mitochondrial-targeting C-terminus; bifunctional distributor",
        "prevalence": "~1:500,000–1,000,000 (rare; <200 reported cases worldwide)",
        "nbs_primary": "C3 (propionylcarnitine) elevated ONLY in cblD-MMA or cblD-Combined subtypes; NORMAL/borderline in cblD-HHcy only — NBS misses this subtype",
        "nbs_secondary": "Urine MMA (elevated in MMA subtypes) + tHcy (elevated in HHcy subtypes); gene panel mandatory — biochemical subtype guides treatment",
        "function": (
            "MMADHC (cblD) is the cytoplasmic DISTRIBUTOR of cobalamin, acting DOWNSTREAM of MMACHC. "
            "MMACHC converts dietary cobalamin to cob(I)alamin; MMADHC then channels this cob(I)alamin to TWO ARMS: "
            "(A) C-terminus (mitochondrial arm): directs cob(I)alamin into mitochondria → MMAB → AdoCbl → MMUT → MMA catabolism. "
            "(B) N-terminus (cytoplasmic arm): retains cob(I)alamin in cytoplasm → MeCbl synthesis → "
            "Methionine Synthase (MTR) → Homocysteine → Methionine remethylation. "
            "UNIQUE: the affected MMADHC domain PREDICTS which metabolic arm is blocked."
        ),
        "mechanism": (
            "MMADHC LOF pattern depends on which domain is mutated: "
            "(1) C-terminus variants → mitochondrial arm blocked → isolated MMA (Hcy NORMAL — cytoplasmic arm intact); "
            "(2) N-terminus variants → cytoplasmic arm blocked → isolated HHcy (MMA NORMAL — mitochondrial arm intact); "
            "(3) Global/central variants → BOTH arms blocked → combined MMA + HHcy (similar to cblC/MMACHC). "
            "This genotype-phenotype relationship is UNIQUE to MMADHC — no other cobalamin gene partitions this way."
        ),
        "key_negative": (
            "BCAA NORMAL in all MMADHC subtypes — KEY NEGATIVE vs DLD/MSUD; "
            "cblD-HHcy ONLY: MMA NORMAL — KEY NEGATIVE vs isolated MMA (MMUT/MMAA/MMAB/MMADHC-MMA); "
            "cblD-MMA ONLY: Hcy NORMAL — KEY NEGATIVE vs cblC (MMACHC) and cblD-HHcy; "
            "NBS: cblD-HHcy MISSES NBS screening (C3 not elevated) — HIGH NBS MISS RATE for this subtype."
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
            "psychiatric_pct": pct(lambda p: p["psychiatric_features"]),
            "nbs_detected_pct": pct(lambda p: p["nbs_detected"]),
            "ohcbl_response_pct": pct(lambda p: p["ohcbl_response"]),
            "mma_subtype_pct": round(len(mma_pts) / n * 100),
            "hhcy_subtype_pct": round(len(hhcy_pts) / n * 100),
            "combined_subtype_pct": round(len(comb_pts) / n * 100),
        },
        "subtype_distribution": [
            {"subtype": "cblD-MMA only (~40%) — C-terminus variants; isolated MMA; Hcy NORMAL",
             "pct": 40, "nbs_detected": "Yes (C3 elevated)", "hcy": "NORMAL", "mma": "ELEVATED"},
            {"subtype": "cblD-HHcy only (~35%) — N-terminus variants; isolated HHcy; MMA NORMAL",
             "pct": 35, "nbs_detected": "MISSED by NBS (C3 normal)", "hcy": "ELEVATED", "mma": "NORMAL"},
            {"subtype": "cblD-Combined (~25%) — central/global variants; combined MMA+HHcy",
             "pct": 25, "nbs_detected": "Yes (C3 elevated)", "hcy": "ELEVATED", "mma": "ELEVATED"},
        ],
        "mmadhc_pathway": [
            {
                "step": "Step 1 — MMACHC (cblC): dietary cobalamin → cob(I)alamin [upstream of MMADHC]",
                "reaction": "CN-Cbl / OH-Cbl / Me-Cbl / Ado-Cbl + NADPH → cob(I)alamin [MMACHC decyanase/reductase]",
                "enzyme": "MMACHC (1p34.1; 282 aa; cytoplasmic; bifunctional decyanase + dealkylase)",
                "consequence_lof": "Upstream block — cblC disease; if MMACHC normal, cob(I)alamin supplied to MMADHC",
            },
            {
                "step": "Step 2 — MMADHC: cob(I)alamin DISTRIBUTION ← BLOCK IN MMADHC LOF",
                "reaction": "cob(I)alamin → (A) mitochondrial arm [C-terminus] OR (B) cytoplasmic arm [N-terminus]",
                "enzyme": "MMADHC (cblD; 296 aa; 2q23.2; N-terminus cytoplasmic + C-terminus mitochondrial)",
                "consequence_lof": "Domain-specific: C-terminus LOF → arm A (MMA) blocked; N-terminus LOF → arm B (HHcy) blocked; both → combined",
            },
            {
                "step": "Step 3A — MMAB → AdoCbl [ARM A: blocked in cblD-MMA or cblD-Combined]",
                "reaction": "cob(I)alamin + ATP → AdoCbl + PPPi [MMAB adenosyltransferase; mitochondrial]",
                "enzyme": "MMAB (cblB; 250 aa; 12q24.11; mitochondrial homotrimer)",
                "consequence_lof": "MMA accumulates (200–3,000 mmol/mol Cr) when arm A blocked; Hcy NORMAL if arm B intact",
            },
            {
                "step": "Step 3B — MeCbl → MTR (Methionine Synthase) [ARM B: blocked in cblD-HHcy or cblD-Combined]",
                "reaction": "cob(I)alamin + 5-methylTHF → MeCbl → MTR: Hcy + 5-methylTHF → Methionine + THF",
                "enzyme": "MTR / Methionine Synthase (MeCbl-dependent; 1q43); MTRR activates MTR",
                "consequence_lof": "HHcy accumulates (30–250 µmol/L); Methionine LOW; MMA NORMAL if arm A intact",
            },
            {
                "step": "Step 4 — MMUT: L-methylmalonyl-CoA → succinyl-CoA [ARM A downstream]",
                "reaction": "L-methylmalonyl-CoA → succinyl-CoA (AdoCbl radical mechanism) [MMUT; TCA cycle entry]",
                "enzyme": "MMUT (methylmalonyl-CoA mutase; 750 aa; 6p12.3; homodimer; TIM-barrel)",
                "consequence_lof": "MMA accumulates when arm A blocked at any level (MMADHC or MMAB or MMAA or MMUT itself)",
            },
        ],
        "high_risk_situations": [
            {"situation": "VPA (Valproate)",
             "risk": "ABSOLUTE CI (MMA subtypes) / HIGH RISK (HHcy subtypes)",
             "detail": "cblD-MMA / cblD-Combined: VPA-CoA directly inhibits mitochondrial cobalamin metabolism + carnitine depletion; FATAL in MMA. cblD-HHcy: HIGH RISK via carnitine depletion; LEV preferred for ALL subtypes"},
            {"situation": "Nitrous oxide (N2O)",
             "risk": "ABSOLUTE CI (HHcy subtypes) / caution (MMA-only subtype)",
             "detail": "cblD-HHcy / cblD-Combined: N2O irreversibly inactivates methionine synthase (MeCbl-dependent) → acute HHcy crisis; LIFE-THREATENING. cblD-MMA: relative caution (methionine synthase usually intact, but metabolic stress)"},
            {"situation": "Fasting / catabolism",
             "risk": "HIGH RISK (MMA subtypes) / moderate (HHcy subtypes)",
             "detail": "MMA subtypes: BCAA catabolism → propionyl-CoA → MMA surge; IV glucose mandatory. HHcy subtypes: catabolic state worsens Hcy remethylation deficit"},
            {"situation": "Missed OHCbl / betaine",
             "risk": "HIGH RISK",
             "detail": "HHcy rebounds within 48–72 hours; MMA rebounds within 3–7 days; subtype-targeted compliance monitoring essential"},
            {"situation": "NBS false-negative (cblD-HHcy)",
             "risk": "DIAGNOSTIC HAZARD",
             "detail": "cblD-HHcy (N-terminus variants) is MISSED by NBS because C3 is not elevated; diagnosis delayed until HHcy symptoms emerge; reflexive plasma Hcy on any neurological presentation in infants"},
        ],
    }


def get_breakdown() -> dict:
    sample = _PATIENTS[:15]
    return {
        "biomarkers": [
            {"name": "C3 (Propionylcarnitine)", "normal": "<4 µmol/L",
             "mmadhc_range": "cblD-MMA: 2–25 µmol/L↑; cblD-HHcy: 0.5–3.5 µmol/L NORMAL; cblD-Combined: 3–20 µmol/L↑",
             "significance": "NBS PRIMARY only in cblD-MMA and cblD-Combined subtypes. cblD-HHcy (N-terminus) has NORMAL C3 — HIGH NBS MISS RATE for this subtype. Biochemical subtype must be confirmed by urine MMA + tHcy SIMULTANEOUSLY.",
             "method": "NBS tandem MS/MS; plasma acylcarnitines"},
            {"name": "Methylmalonic Acid (urine)", "normal": "<5 mmol/mol Cr",
             "mmadhc_range": "cblD-MMA: 200–3,000 mmol/mol Cr↑; cblD-HHcy: NORMAL (<10); cblD-Combined: 150–2,000↑",
             "significance": "ELEVATED in cblD-MMA and cblD-Combined (mitochondrial arm blocked). NORMAL in cblD-HHcy (mitochondrial arm INTACT). This biomarker partitions the two primary subtypes.",
             "method": "Urine organic acids (GC-MS)"},
            {"name": "Total Homocysteine (tHcy)", "normal": "<15 µmol/L",
             "mmadhc_range": "cblD-HHcy: 30–250 µmol/L ELEVATED; cblD-MMA: NORMAL (<14 µmol/L); cblD-Combined: 30–200↑",
             "significance": "ELEVATED in cblD-HHcy and cblD-Combined (cytoplasmic arm blocked). NORMAL in cblD-MMA (cytoplasmic arm INTACT). Combined with MMA: determines subtype. Elevated tHcy + NORMAL MMA = cblD-HHcy or MTR/MTRR deficiency → gene panel mandatory.",
             "method": "Plasma tHcy (HPLC or immunoassay); simultaneous draw with plasma MMA"},
            {"name": "Methionine (plasma)", "normal": "20–40 µmol/L",
             "mmadhc_range": "cblD-HHcy: 5–20 µmol/L LOW; cblD-MMA: NORMAL; cblD-Combined: 8–22 LOW",
             "significance": "LOW only when cytoplasmic/HHcy arm is blocked (methionine synthase inactive). NORMAL in cblD-MMA (methionine synthase intact). Low methionine + high HHcy confirms methionine synthase pathway blockade.",
             "method": "Plasma amino acids (quantitative)"},
            {"name": "MeCbl (fibroblasts)", "normal": "Present",
             "mmadhc_range": "cblD-HHcy: ABSENT; cblD-MMA: NORMAL; cblD-Combined: ABSENT",
             "significance": "ABSENT when N-terminus (cytoplasmic arm) is blocked. NORMAL in cblD-MMA (distinguishes from cblC where BOTH MeCbl and AdoCbl are absent). Key fibroblast test for subtype classification.",
             "method": "Fibroblast cobalamin distribution assay (specialized lab)"},
            {"name": "AdoCbl (fibroblasts)", "normal": "Present",
             "mmadhc_range": "cblD-MMA: ABSENT; cblD-HHcy: NORMAL; cblD-Combined: ABSENT",
             "significance": "ABSENT when C-terminus (mitochondrial arm) is blocked. NORMAL in cblD-HHcy. Combined with MeCbl: fibroblast pattern diagnoses subtype before gene sequencing.",
             "method": "Fibroblast cobalamin distribution assay"},
            {"name": "Methylcitrate (urine)", "normal": "<5 µmol/mmol Cr",
             "mmadhc_range": "cblD-MMA/Combined: <50 µmol/mmol Cr (mild secondary); cblD-HHcy: NORMAL",
             "significance": "Mildly elevated secondary in MMA subtypes. NOT pathognomonic. KEY NEGATIVE vs PA (propionic acidemia) where methylcitrate is pathognomonic (50–2,000).",
             "method": "Urine organic acids (GC-MS)"},
            {"name": "Ammonia (plasma)", "normal": "<50 µmol/L",
             "mmadhc_range": "cblD-MMA Neonatal: 100–450 µmol/L; cblD-HHcy: 15–60 µmol/L (near normal)",
             "significance": "Secondary hyperammonemia via NAGS/CPS1 inhibition (MMA-mediated) in MMA-carrying subtypes. Low/normal in cblD-HHcy alone.",
             "method": "Plasma ammonia (enzymatic)"},
            {"name": "Free Carnitine", "normal": "25–60 µmol/L",
             "mmadhc_range": "cblD-MMA: 8–30 µmol/L depleted; cblD-HHcy: 20–50 µmol/L near-normal",
             "significance": "Depleted via propionylcarnitine excretion in MMA-carrying subtypes. Near-normal in cblD-HHcy alone.",
             "method": "Plasma acylcarnitines"},
            {"name": "BCAA (Leu/Ile/Val)", "normal": "Within reference range",
             "mmadhc_range": "NORMAL in ALL MMADHC subtypes",
             "significance": "NORMAL — KEY NEGATIVE vs DLD (combined BCAA + lactate + 2-HGA elevation) and MSUD (BCAA↑↑). BCKDH and DLD complexes are intact in all MMADHC subtypes.",
             "method": "Plasma amino acids (quantitative)"},
            {"name": "OHCbl Trial Response (MMA)", "normal": "N/A",
             "mmadhc_range": "cblD-MMA: ~50–70% response; cblD-Combined: ~40–60%",
             "significance": "OHCbl bypasses MMADHC C-terminus block by providing cob(I)alamin directly to MMAB. Response rate intermediate between MMAA (high) and MMUT mut0 (low). Measure MMA at Day 0 and Day 7.",
             "method": "Plasma MMA before/after 5–7 days OHCbl 1–2 mg/day IM"},
            {"name": "OHCbl Trial Response (tHcy)", "normal": "N/A",
             "mmadhc_range": "cblD-HHcy: ~60–80% response; cblD-Combined: ~55–70%",
             "significance": "OHCbl bypasses MMADHC N-terminus block by providing cob(I)alamin directly to the methionine synthase pathway. Higher HHcy response rate than MMA response.",
             "method": "tHcy before/after OHCbl trial"},
        ],
        "key_variants": [
            {"variant": "p.Arg132Gln", "cdna": "c.395G>A", "domain": "C-terminus — mitochondrial arm",
             "severity": "Moderate (partial C-terminus)", "subtype": "cblD-MMA only",
             "note": "Most commonly reported cblD-MMA variant; C-terminus missense; partial loss of mitochondrial arm; moderate MMA elevation; OHCbl responsive ~60–70%; infantile or episodic onset"},
            {"variant": "p.Trp135*", "cdna": "c.405G>A", "domain": "C-terminus — nonsense",
             "severity": "Severe (C-terminus null)", "subtype": "cblD-MMA only",
             "note": "Nonsense in C-terminal domain; no mitochondrial targeting; severe cblD-MMA; neonatal onset; poor OHCbl response; more frequent in European populations"},
            {"variant": "p.Arg197Ter", "cdna": "c.589C>T", "domain": "C-terminus — nonsense",
             "severity": "Severe (null)", "subtype": "cblD-MMA only",
             "note": "C-terminus truncation; severe neonatal MMA; biallelic null produces worst cblD-MMA phenotype; combined liver-kidney transplant may be needed if OHCbl non-responsive"},
            {"variant": "p.Gly132Arg", "cdna": "c.394G>C", "domain": "N-terminus — cytoplasmic domain",
             "severity": "Moderate-Severe (N-terminus)", "subtype": "cblD-HHcy only",
             "note": "Most commonly reported cblD-HHcy variant; cytoplasmic N-terminal domain missense; isolated HHcy; MMA NORMAL; OHCbl + betaine responsive ~70–80%; often late-onset or psychiatric"},
            {"variant": "p.Leu49*", "cdna": "c.147T>A", "domain": "N-terminus — early truncation",
             "severity": "Severe (N-terminus null)", "subtype": "cblD-HHcy only",
             "note": "Early N-terminus truncation; complete loss of cytoplasmic arm; isolated severe HHcy; neonatal onset; OHCbl + betaine mandatory; NBS MISSED (C3 normal)"},
            {"variant": "c.1A>G", "cdna": "c.1A>G (start codon loss)", "domain": "Global (N+C both lost)",
             "severity": "Severe (near-null)", "subtype": "cblD-Combined",
             "note": "Start codon loss → minimal/no MMADHC protein; both N-terminus and C-terminus lost → combined cblD; combined MMA+HHcy; clinically similar to cblC/MMACHC; gene panel essential to distinguish"},
            {"variant": "Large del exon 3–4", "cdna": "Exon 3–4 deletion (MLPA)", "domain": "Central/C-terminus",
             "severity": "Severe", "subtype": "cblD-Combined or cblD-MMA (extent-dependent)",
             "note": "Large deletion; extent of deletion determines subtype; MLPA required for detection; not detectable by sequencing alone"},
            {"variant": "p.Met1Val", "cdna": "c.1A>T (start codon)", "domain": "Global",
             "severity": "Moderate-Severe", "subtype": "cblD-HHcy or cblD-Combined",
             "note": "Start codon substitution; reduced MMADHC expression; residual protein if alternate start used; variable phenotype"},
        ],
        "seizure_types": [
            {"type": "Neonatal encephalopathic seizures (MMA/HHcy crisis)", "pct": 72,
             "note": "Multifocal/generalized; driven by MMA mitochondrial toxicity (MMA subtypes) or HHcy endothelial damage (HHcy subtypes) or both (Combined); treat metabolic crisis first — OHCbl IM ± betaine; LEV adjunct; avoid VPA"},
            {"type": "Infantile spasms (West syndrome)", "pct": 38,
             "note": "Early-onset subtypes; hypsarrhythmia; ACTH or vigabatrin when metabolically stabilized; folinic acid for HHcy subtypes (methylation cycle support); LEV adjunct"},
            {"type": "Focal cortical seizures", "pct": 32,
             "note": "Cortical injury from MMA mitochondrial toxicity or HHcy endothelial dysfunction; LEV first-line; improve with metabolic control"},
            {"type": "Myoclonic seizures", "pct": 28,
             "note": "Metabolic-triggered; MMA/HHcy spikes; LEV first-line; correlate with metabolic compliance"},
            {"type": "Absence-like (late-onset HHcy subtype)", "pct": 18,
             "note": "cblD-HHcy late-onset; often with psychiatric symptoms; metabolic treatment reduces burden; LEV preferred"},
            {"type": "Epileptic status / status epilepticus", "pct": 15,
             "note": "Neonatal severe subtypes; continuous EEG; IV BZD + IV glucose + OHCbl IM ± betaine"},
        ],
        "metabolic_triggers": [
            {"trigger": "Fasting / prolonged NPO", "pct": 65,
             "mechanism": "MMA subtypes: BCAA catabolism → propionyl-CoA → MMA surge; IV glucose mandatory. HHcy subtypes: catabolism worsens Hcy remethylation deficit"},
            {"trigger": "Intercurrent illness / fever", "pct": 58,
             "mechanism": "Catabolic state — MMA surge in MMA subtypes; HHcy worsens in HHcy subtypes; IV glucose ± betaine ± OHCbl throughout"},
            {"trigger": "Missed OHCbl ± betaine", "pct": 42,
             "mechanism": "HHcy rebounds within 48–72 hours; MMA rebounds within 3–7 days; subtype-specific monitoring needed"},
            {"trigger": "Nitrous oxide anesthesia (HHcy/Combined subtypes)", "pct": 100,
             "mechanism": "ABSOLUTE CI in cblD-HHcy and cblD-Combined: N2O oxidizes methionine synthase cofactor → acute HHcy crisis; relative risk in cblD-MMA"},
            {"trigger": "High protein intake (MMA subtypes)", "pct": 30,
             "mechanism": "Excess Ile/Val/Met/Thr → propionyl-CoA overload → MMA surge"},
        ],
        "high_risk_drugs": [
            {"drug": "Valproate (VPA)", "risk": "ABSOLUTE CI (MMA subtypes) / HIGH RISK (HHcy subtypes)",
             "mechanism": "cblD-MMA/Combined: VPA-CoA inhibits mitochondrial cobalamin metabolism + carnitine depletion + MMA surge = potentially FATAL. cblD-HHcy: HIGH RISK via carnitine depletion; LEV preferred for ALL MMADHC subtypes"},
            {"drug": "Nitrous oxide (N2O)", "risk": "ABSOLUTE CI (HHcy subtypes)",
             "mechanism": "Irreversibly oxidizes methionine synthase cofactor → complete methionine synthase failure → acute HHcy crisis; LIFE-THREATENING in cblD-HHcy and cblD-Combined"},
            {"drug": "Methotrexate / antifolates", "risk": "HIGH RISK (HHcy subtypes)",
             "mechanism": "Depletes 5-methylTHF → worsens methylation cycle impairment → HHcy rises; folinic acid supplementation already needed"},
            {"drug": "Fasting / NPO without IV glucose", "risk": "HIGH RISK (MMA subtypes)",
             "mechanism": "BCAA catabolism → MMA surge; IV dextrose + OHCbl mandatory during illness in MMA subtypes"},
        ],
        "treatments": [
            {"treatment": "Hydroxocobalamin (OHCbl) 1–2 mg/day IM", "evidence": "Level A",
             "response_pct": 63,
             "note": "Trial ALL MMADHC subtypes. Bypasses MMADHC block by providing cob(I)alamin directly. cblD-MMA: ~50–70% MMA response. cblD-HHcy: ~60–80% HHcy response. cblD-Combined: ~55% HHcy, ~45% MMA. Lifelong IM preferred; monitor tHcy + MMA at Day 7."},
            {"treatment": "Betaine (trimethylglycine) 100–200 mg/kg/day", "evidence": "Level A",
             "response_pct": 78,
             "note": "ONLY for subtypes with HHcy component (cblD-HHcy, cblD-Combined). NOT needed for cblD-MMA only. Remethylates Hcy via BHMT (cobalamin-independent) — same mechanism as in cblC. Level A when HHcy present."},
            {"treatment": "Low protein diet / MMA-free formula", "evidence": "Level A",
             "response_pct": 70,
             "note": "For cblD-MMA and cblD-Combined (MMA component). Low isoleucine/valine/methionine/threonine. MMA-free formula Level A. Less critical for cblD-HHcy (MMA not elevated)."},
            {"treatment": "L-Carnitine 100 mg/kg/day", "evidence": "Level A",
             "response_pct": 80,
             "note": "Secondary depletion via propionylcarnitine excretion (MMA subtypes). Near-normal in cblD-HHcy. Supplement all subtypes prophylactically."},
            {"treatment": "Folinic acid / 5-methylTHF", "evidence": "Level B",
             "response_pct": 52,
             "note": "For HHcy subtypes: 5-methylTHF depleted when methionine synthase is blocked. Bypasses blocked folate methylation. Dose 0.5–5 mg/day. May improve seizure control in infantile spasms."},
            {"treatment": "IV Glucose GIR 8–12 (acute crisis)", "evidence": "Level A",
             "response_pct": 82,
             "note": "First intervention in acute MMA crisis (MMA subtypes); stops BCAA catabolism → MMA surge halted. Continue OHCbl IM during acute phase. For HHcy subtypes: IV glucose reduces catabolism."},
            {"treatment": "Ammonia scavengers (Na benzoate)", "evidence": "Level B",
             "response_pct": 65,
             "note": "NH3 >200 µmol/L in MMA subtypes; secondary via MMA-NAGS/CPS1 inhibition. Same protocol as isolated MMA."},
            {"treatment": "VPA — AVOID / ABSOLUTE CI", "evidence": "AVOID",
             "response_pct": 0,
             "note": "ABSOLUTE CI for cblD-MMA and cblD-Combined. HIGH RISK for cblD-HHcy. LEV is first-line AED for ALL MMADHC seizure types."},
        ],
        "patient_sample": sample,
    }


def get_definitions() -> dict:
    return {
        "gene_card": {
            "Gene": "MMADHC (Methylmalonic Acidemia and Homocystinuria type D gene; cblD gene product)",
            "Full Name": "Methylmalonic Acidemia with Homocystinuria, cblD type (MMADHC; Complementation Group cblD)",
            "Chromosome": "2q23.2",
            "Inheritance": "Autosomal Recessive (AR)",
            "OMIM Gene": "*607269",
            "OMIM Disease": "#277410 (Methylmalonic Acidemia and Homocystinuria, cblD type)",
            "Protein Size": "296 amino acids; N-terminus cytoplasmic (MeCbl arm) + C-terminus mitochondrial-targeting (AdoCbl arm)",
            "Structure": "Bifunctional distributor: N-terminal cytoplasmic domain directs cobalamin to methionine synthase pathway; C-terminal mitochondrial-targeting sequence directs cobalamin to MMAB/MMUT pathway",
            "Reaction": "cob(I)alamin [from MMACHC] → N-terminus: cytoplasmic MeCbl pathway (Hcy remethylation) OR C-terminus: mitochondrial AdoCbl pathway (MMA catabolism)",
            "Pathway Position": "Step 2 — DOWNSTREAM of MMACHC; UPSTREAM of MMAB (arm A) and MTR/MTRR (arm B); the COBALAMIN DISTRIBUTOR at the branch point",
            "Prevalence": "~1:500,000–1,000,000; rare; <200 reported cases; likely underdiagnosed (cblD-HHcy misses NBS)",
            "NBS": "C3 elevated only in cblD-MMA and cblD-Combined; cblD-HHcy MISSES NBS — requires plasma tHcy reflexive test or clinical suspicion",
            "Key Positive (cblD-MMA)": "MMA↑ 200–3,000 mmol/mol Cr; Hcy NORMAL; C3↑; AdoCbl absent fibroblasts; MeCbl NORMAL fibroblasts",
            "Key Positive (cblD-HHcy)": "MMA NORMAL; Hcy↑ 30–250 µmol/L; Methionine LOW; C3 normal/borderline; MeCbl absent fibroblasts; AdoCbl NORMAL fibroblasts",
            "Key Positive (cblD-Combined)": "MMA↑ + Hcy↑; both MeCbl and AdoCbl absent; similar to cblC/MMACHC — gene panel mandatory",
            "Key Negative (all subtypes)": "BCAA NORMAL (vs DLD/MSUD); Biotinidase NORMAL (vs BTD); Biotin not effective (MMADHC not biotin-dependent)",
        },
        "key_concepts": [
            {
                "concept": "MMADHC as the cobalamin DISTRIBUTOR — the branch-point gatekeeper downstream of MMACHC",
                "explanation": (
                    "MMADHC sits DOWNSTREAM of MMACHC and is the DISTRIBUTOR of cob(I)alamin to two separate arms. "
                    "Think of MMACHC as the 'converter' (dietary cobalamin → active form) and MMADHC as the 'router' "
                    "(decides which pathway each active cobalamin molecule goes to). "
                    "N-terminus (cytoplasmic): retains cob(I)alamin in cytoplasm → MeCbl synthesis → MTR (methionine synthase). "
                    "C-terminus (mitochondrial): escorts cob(I)alamin into mitochondria → MMAB → AdoCbl → MMUT. "
                    "This architecture explains why MMADHC is the ONLY gene where domain location of the variant "
                    "PREDICTS which metabolic pathway is disrupted."
                ),
            },
            {
                "concept": "cblD subtype classification — why genotype predicts biochemistry in MMADHC",
                "explanation": (
                    "MMADHC has a modular domain structure: N-terminus handles the cytoplasmic (HHcy) arm; "
                    "C-terminus handles the mitochondrial (MMA) arm. "
                    "• C-terminus variants (e.g., p.Arg132Gln, p.Trp135*) → arm A (MMA) disrupted → "
                    "isolated MMA; Hcy NORMAL; MeCbl NORMAL in fibroblasts. "
                    "• N-terminus variants (e.g., p.Gly132Arg, p.Leu49*) → arm B (HHcy/MeCbl) disrupted → "
                    "isolated HHcy; MMA NORMAL; AdoCbl NORMAL in fibroblasts. "
                    "• Global variants (c.1A>G, large deletions) → both arms disrupted → combined MMA+HHcy. "
                    "This genotype-phenotype correlation is UNIQUE in cobalamin disorders. "
                    "For clinical management: check which arm is disrupted BEFORE starting treatment — "
                    "betaine is Level A for HHcy subtypes but NOT indicated for cblD-MMA only."
                ),
            },
            {
                "concept": "cblD-HHcy — the NBS-invisible cobalamin disorder",
                "explanation": (
                    "The cblD-HHcy subtype (N-terminus variants) has NORMAL or borderline C3 on NBS because "
                    "the MMA catabolism pathway is INTACT. Propionylcarnitine (C3) is NOT elevated. "
                    "This means cblD-HHcy patients are MISSED by standard NBS algorithms that rely on C3 elevation. "
                    "They present later with: neurological regression, hypotonia, megaloblastic anemia, "
                    "developmental delay — or in late-onset form with psychiatric symptoms. "
                    "Clue in lab: tHcy elevated + MMA NORMAL + Methionine LOW → "
                    "differential includes cblD-HHcy, MTR (methionine synthase) deficiency, MTRR (methionine "
                    "synthase reductase) deficiency → gene panel mandatory. "
                    "Second-tier tHcy on NBS dried blood spots (if available) would detect these; "
                    "some NBS programs now include reflexive tHcy — critical for cblD-HHcy detection."
                ),
            },
            {
                "concept": "Distinguishing MMADHC from MMACHC (cblC) — fibroblast cobalamin studies are key",
                "explanation": (
                    "cblD-Combined (both arms blocked) is biochemically similar to cblC (MMACHC): "
                    "both show combined MMA + HHcy + LOW methionine. Clinical differential matters for: "
                    "prognosis, variant-specific OHCbl response prediction, and recurrence risk. "
                    "Fibroblast cobalamin distribution: "
                    "• cblC (MMACHC): BOTH MeCbl and AdoCbl absent (upstream block — no cob(I)alamin produced). "
                    "• cblD-Combined (MMADHC): BOTH MeCbl and AdoCbl absent (downstream block — cob(I)alamin "
                    "produced but cannot be distributed) → same result by fibroblast test. "
                    "GENE PANEL is the definitive distinction: MMACHC vs MMADHC sequencing. "
                    "FUNCTIONAL ASSAY on fibroblasts: cblD cells complement cblC cells in mixed culture "
                    "(complementation groups); laboratory complementation test distinguishes the two. "
                    "Practical: always send MMACHC + MMADHC + MMUT + MMAA + MMAB panel simultaneously."
                ),
            },
            {
                "concept": "OHCbl mechanism in MMADHC — arm-specific bypass",
                "explanation": (
                    "Hydroxocobalamin (OHCbl) in pharmacological doses bypasses MMADHC by flooding "
                    "the system with cob(II)alamin/cob(I)alamin that can directly enter the downstream enzymes: "
                    "• Arm A bypass: cob(I)alamin directly enters mitochondria → MMAB → AdoCbl → MMUT → MMA ↓. "
                    "• Arm B bypass: cob(I)alamin → direct MeCbl synthesis via MTRR → MTR → Hcy ↓. "
                    "Arm-specific response prediction from subtype: "
                    "cblD-MMA: OHCbl primarily helps arm A → MMA response ~50–70%. "
                    "cblD-HHcy: OHCbl primarily helps arm B → HHcy response ~60–80%. "
                    "cblD-Combined: both arms benefit → intermediate response both biomarkers. "
                    "Betaine added ONLY for HHcy subtypes: independently remethylates Hcy via BHMT."
                ),
            },
            {
                "concept": "N2O ABSOLUTE CI in cblD-HHcy and cblD-Combined — anesthesia protocol",
                "explanation": (
                    "Nitrous oxide (N2O) IRREVERSIBLY oxidizes the methionine synthase cofactor (cob(I)alamin → Co3+), "
                    "permanently inactivating MTR. In cblD-HHcy and cblD-Combined subtypes, "
                    "methionine synthase is ALREADY compromised (no MeCbl supply from arm B). "
                    "N2O exposure → complete methionine synthase failure → acute HHcy crisis. "
                    "cblD-MMA only: methionine synthase is INTACT (arm B functional) → N2O is relative risk "
                    "(metabolic stress) rather than immediate crisis, but still avoid. "
                    "Anesthesia protocol for MMADHC patients: NEVER use N2O. "
                    "Use halogenated volatile agents (sevoflurane) or IV propofol. "
                    "Perioperative IV OHCbl 1 mg IM day of surgery and day after. "
                    "Subtype must be documented in allergy section of medical record: "
                    "'N2O ABSOLUTE CI — MMADHC (cblD) — HHcy subtype.'"
                ),
            },
        ],
        "diagnostic_thresholds": [
            {"parameter": "MMA (urine) + tHcy combined pattern",
             "threshold": "MMA↑ + Hcy NORMAL = cblD-MMA or isolated MMA (MMUT/MMAA/MMAB); MMA NORMAL + Hcy↑ = cblD-HHcy or MTR/MTRR; both↑ = cblD-Combined or cblC/MMACHC",
             "action": "Gene panel: MMADHC + MMACHC + MMUT + MMAA + MMAB + MTR + MTRR; fibroblast cobalamin studies; start OHCbl 1 mg IM immediately"},
            {"parameter": "Isolated HHcy + NORMAL MMA in infant",
             "threshold": "tHcy >30 µmol/L + MMA <10 mmol/mol Cr + Methionine <20 µmol/L",
             "action": "HIGH SUSPICION for cblD-HHcy or MTR/MTRR — NOT cblC/MMACHC (those have MMA elevated); start OHCbl + betaine immediately; fibroblast MeCbl assay; MMADHC/MTR/MTRR gene panel"},
            {"parameter": "OHCbl trial response (MMA subtype)",
             "threshold": "MMA reduction >50% at Day 7 = OHCbl-responsive cblD-MMA",
             "action": "Lifelong OHCbl IM; subtype confirmed; add low protein diet + L-carnitine; no betaine needed (Hcy normal)"},
            {"parameter": "OHCbl trial response (HHcy subtype)",
             "threshold": "tHcy reduction >50% at Day 7 = OHCbl-responsive cblD-HHcy",
             "action": "Lifelong OHCbl IM + betaine Level A; no protein restriction needed (MMA normal); folinic acid add if tHcy partially controlled"},
            {"parameter": "Ammonia (MMA subtypes)",
             "threshold": "NH3 >200 µmol/L: start ammonia scavengers; >400: HD/CRRT",
             "action": "IV Na benzoate 250 mg/kg; IV glucose GIR 8–12; OHCbl IM; HD/CRRT if NH3 >400 or rising"},
            {"parameter": "tHcy monitoring on OHCbl + betaine",
             "threshold": "Target tHcy <50 µmol/L (ideally <30) on therapy",
             "action": "Monthly tHcy in first year; adjust betaine dose; if not at target add folinic acid"},
        ],
        "differential_diagnosis": [
            {"disease": "MMACHC (cblC) — upstream MMACHC decyanase/reductase defect",
             "distinguishing": "cblC always causes COMBINED MMA + HHcy (no isolated MMA-only or HHcy-only); MMACHC block is upstream of branch point so both arms always affected; maculopathy ~80% in cblC (rare in MMADHC); gene: MMACHC (1p34.1)"},
            {"disease": "MMUT (mut-type) — methylmalonyl-CoA mutase apoenzyme",
             "distinguishing": "Isolated MMA only; Hcy NORMAL; no HHcy; MeCbl NORMAL in fibroblasts; MMUT enzyme ABSENT without AdoCbl; gene: MMUT (6p12.3)"},
            {"disease": "MMAA (cblA) — GTPase chaperone delivery defect",
             "distinguishing": "Isolated MMA; Hcy NORMAL; OHCbl STRONG response ~60–80%; MMAB enzyme NORMAL; gene: MMAA (4p14)"},
            {"disease": "MMAB (cblB) — adenosyltransferase synthesis defect",
             "distinguishing": "Isolated MMA; Hcy NORMAL; OHCbl MODERATE response ~40–60%; MMAB enzyme ABSENT; gene: MMAB (12q24.11)"},
            {"disease": "MTR (cblG) — methionine synthase deficiency",
             "distinguishing": "Isolated HHcy + NORMAL MMA (identical metabolite pattern to cblD-HHcy); AdoCbl NORMAL in fibroblasts; MeCbl absent; MTR enzyme ABSENT; gene: MTR (1q43); gene panel distinguishes"},
            {"disease": "MTRR (cblE) — methionine synthase reductase deficiency",
             "distinguishing": "Isolated HHcy + NORMAL MMA (identical to cblD-HHcy + MTR); MTRR needed to reactivate MTR; gene: MTRR (5p15.31); complementation group cblE; gene panel distinguishes"},
            {"disease": "CBS deficiency (classical homocystinuria)",
             "distinguishing": "ELEVATED Hcy BUT: Methionine HIGH (vs LOW in cblD-HHcy); NO MMA elevation; NO C3 on NBS; marfanoid habitus; lens dislocation; CBS (21q22.3); treat with pyridoxine + betaine + methionine restriction"},
            {"disease": "PCCA/PCCB (Propionic Acidemia)",
             "distinguishing": "Methylcitrate PATHOGNOMONIC (50–2,000) vs MMADHC absent/mild; NO MMA elevation (MMA is the MMADHC biomarker, not PA); Hcy NORMAL; gene panel: PCCA/PCCB"},
        ],
    }
