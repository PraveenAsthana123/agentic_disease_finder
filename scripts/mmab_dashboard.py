#!/usr/bin/env python3
"""MMAB (Methylmalonic Acidemia, cblB type / Cobalamin-B Deficiency) Epilepsy Dashboard.

MMAB encodes ATP:cob(I)alamin adenosyltransferase (Cobalamin B, cblB), the mitochondrial
enzyme that synthesizes adenosylcobalamin (AdoCbl) from cob(I)alamin and ATP.

  MMAB PROTEIN BIOLOGY:
    MMAB protein: 250 aa, homotrimeric, mitochondrial matrix.
    Function: ATP:cob(I)alamin adenosyltransferase — catalyzes the final step of AdoCbl
    biosynthesis: cob(I)alamin + ATP → AdoCbl + PPPi (tripolyphosphate).
    This is STEP 2 of the cobalamin-to-MMUT pathway:
      Step 1: MMACHC (cblC) — converts dietary cobalamins → cob(I)alamin
      Step 2: MMAB (cblB) — makes AdoCbl from cob(I)alamin + ATP  ← BLOCK IN MMAB LOF
      Step 3: MMAA (cblA) — delivers AdoCbl from MMAB to MMUT TIM-barrel
      Step 4: MMUT — uses AdoCbl to isomerize L-methylmalonyl-CoA → succinyl-CoA

  MMAB LOF → AdoCbl CANNOT BE SYNTHESIZED:
    MMUT apoenzyme is STRUCTURALLY INTACT (enzyme protein normal — same as MMAA)
    → MMUT cannot catalyze L-methylmalonyl-CoA → succinyl-CoA (no cofactor)
    → L-methylmalonyl-CoA ACCUMULATES → METHYLMALONIC ACID (MMA) ELEVATED
    Distinction from MMAA: MMAA can synthesize AdoCbl (MMAB is upstream — makes the cofactor);
    MMAA only delivers it. In MMAB LOF, no AdoCbl is made at all.

  KEY DISTINCTIONS FROM MMAA (cblA):
    In MMAA deficiency: MMAB synthesizes AdoCbl normally — DELIVERY to MMUT fails.
    In MMAB deficiency: AdoCbl SYNTHESIS fails — nothing to deliver (MMAA is intact).
    OHCbl response rate: MMAB ~40–60% MODERATE (lower than MMAA ~60–80%) because
    pharmacological OHCbl must overcome an upstream synthetic defect rather than a
    delivery defect — harder to saturate alternate synthesis routes without MMAB.
    Fibroblast assay: MMAB patients, like MMAA, show MMUT correction when AdoCbl
    is provided exogenously (MMUT apoenzyme INTACT in both).
    Gene panel: MMUT, MMAA, MMAB, MMACHC, MMADHC mandatory — biochemistry alone
    CANNOT reliably distinguish MMAB from MMAA or MMUT.

  PATHWAY UPSTREAM OF MMAB BLOCK:
    Propionyl-CoA (from Ile, Val, Met, Thr; odd-chain FA; cholesterol)
      → D-methylmalonyl-CoA [PCC: PCCA+PCCB]
      → L-methylmalonyl-CoA [MCEE: epimerase]
    COFACTOR SYNTHESIS PATHWAY (MMAB step):
      Dietary cobalamin → MMACHC → cob(I)alamin → MMAB + ATP → AdoCbl ← BLOCK IN MMAB
      AdoCbl (if made) → MMAA chaperone → MMUT apoenzyme ← cannot happen in MMAB LOF
    Succinyl-CoA → TCA (→ fumarate → malate → OAA)

  BIOMARKERS:
    C3 (propionylcarnitine): ↑ — NBS PRIMARY (same as PA/MMUT/MMAA)
    Methylmalonic acid (urine): ↑↑ 300–5,000 mmol/mol Cr (PATHOGNOMONIC; slightly higher
      range than MMAA at baseline because OHCbl response rate is lower)
    Methylmalonic acid (plasma): ↑ 100–2,500 µmol/L
    Methylcitrate: mildly elevated secondary <50 µmol/mmol Cr (NOT pathognomonic; vs PA)
    Ammonia: ↑ 100–1,000 µmol/L (secondary NAGS/CPS1 inhibition — same mechanism as MMUT/PA)
    Free carnitine: ↓ 8–30 µmol/L (secondary depletion)
    Homocysteine: NORMAL <15 µmol/L (KEY NEGATIVE vs cblC/MMACHC)
    BCAA: NORMAL (KEY NEGATIVE vs DLD/MSUD)
    OHCbl response: ~40–60% moderate (LOWER than MMAA ~60–80%; higher than MMUT ~20%)
    MMUT enzyme activity in fibroblasts: NORMAL when AdoCbl provided (same as MMAA)
    MMAB enzyme activity: ABSENT or severely reduced in fibroblasts/lymphocytes

  KEY VARIANTS IN MMAB:
    - p.Arg190His (c.569G>A): most common MMAB allele; adenosyltransferase active site;
      partial enzyme activity; moderate OHCbl response; European/worldwide
    - p.Arg190Cys (c.568C>T): same codon, different substitution; severe/non-responsive
    - p.Arg206Cys (c.616C>T): trimeric MMAB interface disruption; moderate-severe
    - p.Arg234Trp (c.700C>T): substrate (cob(I)alamin) binding channel; severe
    - p.Gly188Arg (c.562G>A): catalytic Gly adjacent to active site; severe neonatal
    - c.IVS7+1G>A: splice site null allele; severe
    - Large del exon5-6: null allele (~8% of MMAB alleles); MLPA needed
    - p.Trp202* (c.606G>A): nonsense null; severe non-responsive

  TREATMENT HIGHLIGHTS:
    Hydroxocobalamin OHCbl 1–2 mg/day IM (Level A — trial ALL patients; ~40–60% respond —
      MODERATE response, lower than MMAA cblA ~60–80%)
    Protein restriction — low Ile/Val/Met/Thr (Level A; lifelong)
    MMAB-free amino acid formula (Level A)
    L-Carnitine 100 mg/kg/day (Level A; secondary depletion)
    IV glucose GIR 8–12 + insulin (Level A; acute crisis)
    Ammonia scavengers (Level A if NH3 >200)
    Metronidazole (Level B; reduces gut propionate)
    Liver transplant (Level B; for OHCbl non-responders — needed more than MMAA, less than MMUT mut0)
    HD/CRRT (Level B; MMA >3,000 or NH3 >500)
    VPA — ABSOLUTE CI (inhibits cobalamin synthesis pathway at MMAB step + carnitine depletion; FATAL)
    Fasting — EXTREME HAZARD (BCAA catabolism surge)
    High protein — EXTREME HAZARD (Ile/Val/Met/Thr precursors)
    Missed OHCbl doses — HIGH RISK (MMA rebounds within days)

  OMIM: Gene *607568 · Disease #251110 (Methylmalonic Acidemia, cblB type)
  Chromosome: 12q24.11 · Inheritance: AR · Prevalence: ~1:100,000–250,000
  (~15–20% of cobalamin-responsive isolated MMA; less common than MMAA)
"""

from __future__ import annotations
import random

random.seed(47)   # reproducible synthetic cohort


# ── helpers ──────────────────────────────────────────────────────────────────

def _pid(i: int) -> str:
    return f"MMAB-{i:03d}"


def _sex(i: int) -> str:
    return "M" if i % 2 == 0 else "F"


def _phenotype(i: int) -> str:
    """
    cblB Neonatal Severe ~35% (14 patients) — more severe than cblA (null alleles prevalent),
    cblB Classic Infantile ~45% (18 patients — MODAL),
    cblB Attenuated Late Onset ~15% (6 patients),
    cblB OHCbl-Responsive Mild ~5% (2 patients).
    """
    if i <= 14:   return "cblB Neonatal Severe"
    if i <= 32:   return "cblB Classic Infantile"
    if i <= 38:   return "cblB Attenuated Late Onset"
    return "cblB OHCbl-Responsive Mild"


def _genotype(i: int) -> str:
    variants = [
        "c.569G>A/c.569G>A",         # p.Arg190His homozygous; most common; moderate OHCbl response
        "c.569G>A/c.568C>T",         # p.Arg190His / p.Arg190Cys compound het; variable response
        "c.568C>T/c.568C>T",         # p.Arg190Cys homozygous; severe non-responsive
        "c.569G>A/c.606G>A",         # p.Arg190His / p.Trp202* compound het; moderate-severe
        "c.IVS7+1G>A/c.568C>T",      # splice null / p.Arg190Cys; severe
        "c.562G>A/c.569G>A",         # p.Gly188Arg / p.Arg190His; severe
        "c.569G>A/del.exon5-6",      # p.Arg190His / large deletion; compound het; moderate
        "c.616C>T/c.616C>T",         # p.Arg206Cys homozygous; moderate-severe trimeric interface
        "c.700C>T/c.569G>A",         # p.Arg234Trp / p.Arg190His; severe; substrate channel
        "c.606G>A/del.exon5-6",      # p.Trp202* / large deletion; null; severe non-responsive
    ]
    return variants[i % len(variants)]


def _mma_urine(pheno: str) -> float:
    """Methylmalonic acid urine mmol/mol Cr — slightly higher than MMAA (lower OHCbl response rate)."""
    if pheno == "cblB Neonatal Severe":
        return round(random.uniform(3500, 5000), 0)
    if pheno == "cblB Classic Infantile":
        return round(random.uniform(600, 3000), 0)
    if pheno == "cblB Attenuated Late Onset":
        return round(random.uniform(300, 1000), 0)
    return round(random.uniform(150, 500), 0)


def _plasma_mma(pheno: str) -> float:
    """Plasma MMA µmol/L."""
    if pheno == "cblB Neonatal Severe":
        return round(random.uniform(1000, 2500), 0)
    if pheno == "cblB Classic Infantile":
        return round(random.uniform(200, 1000), 0)
    if pheno == "cblB Attenuated Late Onset":
        return round(random.uniform(100, 400), 0)
    return round(random.uniform(50, 200), 0)


def _c3(pheno: str) -> float:
    if pheno == "cblB Neonatal Severe":
        return round(random.uniform(15, 40), 1)
    if pheno == "cblB Classic Infantile":
        return round(random.uniform(5, 25), 1)
    if pheno == "cblB Attenuated Late Onset":
        return round(random.uniform(3, 14), 1)
    return round(random.uniform(2, 7), 1)


def _ammonia(pheno: str) -> int:
    if pheno == "cblB Neonatal Severe":
        return random.randint(400, 1000)
    if pheno == "cblB Classic Infantile":
        return random.randint(150, 500)
    if pheno == "cblB Attenuated Late Onset":
        return random.randint(50, 200)
    return random.randint(20, 100)


def _carnitine(pheno: str) -> float:
    if pheno == "cblB Neonatal Severe":
        return round(random.uniform(6, 16), 1)
    if pheno == "cblB Classic Infantile":
        return round(random.uniform(10, 25), 1)
    if pheno == "cblB Attenuated Late Onset":
        return round(random.uniform(18, 35), 1)
    return round(random.uniform(22, 40), 1)


def _egfr(pheno: str) -> int:
    """eGFR — CKD pattern intermediate between MMAA and MMUT (lower response rate than MMAA)."""
    if pheno == "cblB Neonatal Severe":
        return random.randint(25, 65)
    if pheno == "cblB Classic Infantile":
        return random.randint(45, 85)
    if pheno == "cblB Attenuated Late Onset":
        return random.randint(60, 100)
    return random.randint(70, 115)


def _onset_months(pheno: str) -> int:
    if pheno == "cblB Neonatal Severe":
        return random.randint(0, 1)
    if pheno == "cblB Classic Infantile":
        return random.randint(1, 18)
    if pheno == "cblB Attenuated Late Onset":
        return random.randint(12, 96)
    return random.randint(18, 200)


def _ohcbl_response(pheno: str) -> bool:
    """~40–60% moderate OHCbl response rate in MMAB (LOWER than MMAA ~60–80%)."""
    if pheno == "cblB Neonatal Severe":
        return random.random() < 0.18   # null alleles predominant
    if pheno == "cblB Classic Infantile":
        return random.random() < 0.55
    if pheno == "cblB Attenuated Late Onset":
        return random.random() < 0.80
    return True  # mild phenotype — typically B12 responsive


def _seizures(pheno: str) -> bool:
    if pheno == "cblB Neonatal Severe":
        return random.random() < 0.80
    if pheno == "cblB Classic Infantile":
        return random.random() < 0.60
    if pheno == "cblB Attenuated Late Onset":
        return random.random() < 0.30
    return random.random() < 0.15


def _cardiomyopathy(pheno: str) -> bool:
    """~20–25% cardiomyopathy — intermediate between MMAA (15–20%) and MMUT (25–35%)."""
    if pheno == "cblB Neonatal Severe":
        return random.random() < 0.35
    if pheno == "cblB Classic Infantile":
        return random.random() < 0.22
    return False


def _aed(pheno: str, seizures: bool) -> str:
    if not seizures:
        return "None"
    aeds = ["LEV", "LEV + CLB", "LEV + VNS", "LEV monotherapy"]
    return random.choice(aeds)


def _nbs(pheno: str) -> bool:
    if pheno == "cblB Neonatal Severe":
        return random.random() < 0.60   # ~40% missed pre-NBS era
    return random.random() < 0.82


def _diag_months(pheno: str, onset: int) -> int:
    if pheno == "cblB Neonatal Severe":
        return onset + random.randint(0, 3)
    return onset + random.randint(1, 12)


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
        "c3_umol_l":                 _c3(ph),
        "ammonia_umol_l":            _ammonia(ph),
        "free_carnitine_umol_l":     _carnitine(ph),
        "egfr_ml_min_1_73m2":        _egfr(ph),
        "seizures":                  sz,
        "cardiomyopathy":            _cardiomyopathy(ph),
        "nbs_detected":              _nbs(ph),
        "ohcbl_response":            _ohcbl_response(ph),
        "aed":                       _aed(ph, sz),
    })


# ── API functions ─────────────────────────────────────────────────────────────

def get_overview() -> dict:
    n         = len(_PATIENTS)
    mma_vals  = [p["mma_urine_mmol_molCr"] for p in _PATIENTS]
    c3_vals   = [p["c3_umol_l"] for p in _PATIENTS]
    nh3_vals  = [p["ammonia_umol_l"] for p in _PATIENTS]
    car_vals  = [p["free_carnitine_umol_l"] for p in _PATIENTS]
    egfr_vals = [p["egfr_ml_min_1_73m2"] for p in _PATIENTS]

    def pct(pred): return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    return {
        "gene": "MMAB",
        "full_name": "ATP:cob(I)alamin Adenosyltransferase (Cobalamin B / cblB)",
        "chromosome": "12q24.11",
        "inheritance": "AR",
        "omim_gene": "*607568",
        "omim_disease": "#251110",
        "protein_size": "250 aa; homotrimeric; mitochondrial matrix; adenosyltransferase",
        "prevalence": "~1:100,000–250,000 (~15–20% of cobalamin-responsive isolated MMA)",
        "nbs_primary": "C3 (propionylcarnitine) elevated — identical to PA, MMUT, and MMAA; triggers MMA workup",
        "nbs_secondary": "Urine MMA (PATHOGNOMONIC 300–5,000); plasma MMA; OHCbl trial; MMAB enzyme assay",
        "function": (
            "MMAB (cblB) is ATP:cob(I)alamin adenosyltransferase — the mitochondrial enzyme that "
            "synthesizes adenosylcobalamin (AdoCbl) by adenylating cob(I)alamin using ATP. "
            "Reaction: cob(I)alamin + ATP → AdoCbl + tripolyphosphate (PPPi). "
            "MMAB forms a homotrimer; the active site lies at the interface of adjacent monomers."
        ),
        "mechanism": (
            "MMAB LOF → AdoCbl CANNOT BE SYNTHESIZED from cob(I)alamin + ATP → MMUT apoenzyme "
            "receives no cofactor (MMAA chaperone has nothing to deliver) → L-methylmalonyl-CoA "
            "accumulates → MMA elevated (300–5,000 mmol/mol Cr). "
            "MMUT apoenzyme is STRUCTURALLY INTACT — adding AdoCbl exogenously corrects fibroblast "
            "enzyme activity (same as MMAA). Pharmacological OHCbl (1–2 mg/day IM) provides substrate "
            "for alternative AdoCbl synthesis routes → ~40–60% of MMAB patients respond "
            "(MODERATE, lower than MMAA ~60–80% because synthesis defect harder to overcome). "
            "Secondary: MMA inhibits CPS1 via NAGS → hyperammonemia; MMA-related CKD risk."
        ),
        "key_negative": (
            "HOMOCYSTEINE NORMAL (<15 µmol/L) — KEY NEGATIVE vs cblC (MMACHC, combined MMA+HHcy); "
            "BCAA NORMAL — KEY NEGATIVE vs DLD/MSUD; METHYLCITRATE mildly elevated secondary only "
            "(NOT pathognomonic — vs PA where methylcitrate is PATHOGNOMONIC); "
            "MMUT ENZYME INTACT in fibroblasts when AdoCbl provided — same as MMAA; "
            "MMAB ENZYME ACTIVITY ABSENT or severely reduced in fibroblasts — KEY DISTINCTION from MMAA."
        ),
        "cohort_n": n,
        "kpis": {
            "avg_mma_urine": round(sum(mma_vals) / n),
            "avg_c3_umol_l": round(sum(c3_vals) / n, 1),
            "avg_ammonia_umol_l": round(sum(nh3_vals) / n),
            "avg_free_carnitine": round(sum(car_vals) / n, 1),
            "avg_egfr": round(sum(egfr_vals) / n),
            "seizure_pct": pct(lambda p: p["seizures"]),
            "cardiomyopathy_pct": pct(lambda p: p["cardiomyopathy"]),
            "ckd_pct": pct(lambda p: p["egfr_ml_min_1_73m2"] < 60),
            "nbs_detected_pct": pct(lambda p: p["nbs_detected"]),
            "ohcbl_response_pct": pct(lambda p: p["ohcbl_response"]),
            "b12_responsive_pct": pct(lambda p: p["ohcbl_response"]),
        },
        "phenotype_distribution": [
            {"phenotype": "cblB Neonatal Severe (~35%) — more severe than cblA", "pct": 35},
            {"phenotype": "cblB Classic Infantile (~45%) — MODAL", "pct": 45},
            {"phenotype": "cblB Attenuated Late Onset (~15%)", "pct": 15},
            {"phenotype": "cblB OHCbl-Responsive Mild (~5%)", "pct": 5},
        ],
        "mmab_pathway": [
            {
                "step": "Step 1 — MMACHC (cblC): dietary cobalamin → cob(I)alamin",
                "reaction": "Cob(II)alamin / CN-Cbl / OH-Cbl + reductant → cob(I)alamin [MMACHC decyanylase]",
                "enzyme": "MMACHC (cblC chaperone; cytoplasmic)",
                "cofactor": "FAD/reductant",
                "consequence_mmab_lof": "INTACT in MMAB deficiency — cob(I)alamin is produced normally",
            },
            {
                "step": "Step 2 — MMAB (cblB): cob(I)alamin → AdoCbl + ATP ← BLOCK IN MMAB LOF",
                "reaction": "cob(I)alamin + ATP → AdoCbl + PPPi [MMAB homotrimer active site]",
                "enzyme": "MMAB (cblB; ATP:cob(I)alamin adenosyltransferase; 250 aa; 12q24.11)",
                "cofactor": "ATP",
                "consequence_mmab_lof": "BLOCKED: AdoCbl CANNOT BE SYNTHESIZED → MMAA chaperone has nothing to transfer → MMUT cofactor-empty",
            },
            {
                "step": "Step 3 — MMAA (cblA): GTP-dependent AdoCbl delivery from MMAB to MMUT",
                "reaction": "MMAA + GTP + AdoCbl-MMAB → MMUT-AdoCbl holoenzyme",
                "enzyme": "MMAA (cblA; G3E P-loop GTPase chaperone; 418 aa; 4p14)",
                "cofactor": "GTP",
                "consequence_mmab_lof": "BLOCKED (secondary — MMAA is intact but has no AdoCbl cargo to deliver from MMAB LOF)",
            },
            {
                "step": "Step 4 — MMUT holoenzyme: L-methylmalonyl-CoA → succinyl-CoA → TCA",
                "reaction": "L-methylmalonyl-CoA → succinyl-CoA (AdoCbl radical mechanism) [MMUT]",
                "enzyme": "MMUT (methylmalonyl-CoA mutase; 750 aa; TIM-barrel; INTACT apoenzyme in MMAB deficiency)",
                "cofactor": "AdoCbl (not available — MMAB cannot synthesize it)",
                "consequence_mmab_lof": "BLOCKED (secondary): MMUT apoenzyme is STRUCTURALLY NORMAL → L-methylmalonyl-CoA accumulates → MMA ↑↑",
            },
        ],
        "mmab_vs_mmaa": {
            "title": "MMAB (cblB) vs MMAA (cblA) — 10-Feature Critical Differential (Both Isolated MMA)",
            "note": (
                "MMAB and MMAA both cause isolated MMA without homocysteine elevation. "
                "Both have intact MMUT apoenzyme correctable by exogenous AdoCbl in fibroblasts. "
                "Key differences: MMAB has lower OHCbl response rate; more null alleles; "
                "gene panel (MMUT, MMAA, MMAB, MMACHC, MMADHC) is mandatory to distinguish."
            ),
            "comparison": [
                {"feature": "OHCbl Response", "MMAB_cblB": "MODERATE ~40–60% (lower than cblA)", "MMAA_cblA": "STRONG ~60–80% (HALLMARK)"},
                {"feature": "Block in Pathway", "MMAB_cblB": "AdoCbl SYNTHESIS blocked (cob(I)alamin+ATP→AdoCbl)", "MMAA_cblA": "AdoCbl DELIVERY blocked (MMAB→MMUT transfer)"},
                {"feature": "MMUT Fibroblast Activity", "MMAB_cblB": "NORMAL when AdoCbl provided (apoenzyme intact)", "MMAA_cblA": "NORMAL when AdoCbl provided (apoenzyme intact)"},
                {"feature": "MMAB Enzyme Activity", "MMAB_cblB": "ABSENT/severely reduced — primary defect", "MMAA_cblA": "NORMAL (MMAB is intact; only delivery fails)"},
                {"feature": "Neonatal Severe %", "MMAB_cblB": "~35% (null alleles more prevalent)", "MMAA_cblA": "~25% (p.Arg145His partial allele common)"},
                {"feature": "Classic Infantile % (modal)", "MMAB_cblB": "~45% (MODAL)", "MMAA_cblA": "~50% (MODAL)"},
                {"feature": "CKD severity", "MMAB_cblB": "Moderate (~22%); lower response → more CKD than MMAA", "MMAA_cblA": "Less severe (~18%); better OHCbl response"},
                {"feature": "Cardiomyopathy %", "MMAB_cblB": "~20–25%", "MMAA_cblA": "~15–20%"},
                {"feature": "Transplant need", "MMAB_cblB": "Level B; more needed than MMAA (lower OHCbl resp.)", "MMAA_cblA": "Rarely needed; OHCbl usually sufficient"},
                {"feature": "Gene (chromosome)", "MMAB_cblB": "MMAB — 12q24.11 (adenosyltransferase)", "MMAA_cblA": "MMAA — 4p14 (GTPase chaperone)"},
            ],
        },
        "high_risk_situations": [
            {"situation": "VPA (Valproate)", "risk": "ABSOLUTE CI", "detail": "VPA and its CoA-thioester metabolites directly inhibit MMAB adenosyltransferase activity (primary enzyme in MMAB deficiency) + deplete carnitine + worsen MMA: FATAL"},
            {"situation": "Fasting", "risk": "EXTREME HAZARD", "detail": "BCAA catabolism → propionyl-CoA → L-methylmalonyl-CoA → MMA surge within hours; IV glucose mandatory"},
            {"situation": "High-protein diet", "risk": "EXTREME HAZARD", "detail": "Ile/Val/Met/Thr directly drives propionyl-CoA → L-methylmalonyl-CoA → MMA accumulation"},
            {"situation": "Intercurrent illness", "risk": "EXTREME HAZARD", "detail": "Catabolic state → MMA surge + hyperammonemia; IV glucose protocol mandatory; do NOT withhold OHCbl"},
            {"situation": "Missed OHCbl doses", "risk": "HIGH RISK", "detail": "MMA rebounds within 3–7 days; compliance monitoring essential; OHCbl responders especially at risk for rapid relapse"},
            {"situation": "Metformin", "risk": "AVOID", "detail": "Complex I inhibitor — synergistic toxicity with MMA-related mitochondrial dysfunction → lactic acidosis risk"},
        ],
    }


def get_breakdown() -> dict:
    sample = _PATIENTS[:15]
    return {
        "biomarkers": [
            {"name": "C3 (Propionylcarnitine)", "normal": "<4 µmol/L", "mmab_range": "3–40 µmol/L", "significance": "NBS PRIMARY — identical to PA, MMUT, and MMAA; cannot distinguish by NBS alone; MMA workup mandatory on positive screen", "method": "NBS tandem MS/MS; plasma acylcarnitines"},
            {"name": "Methylmalonic Acid (urine)", "normal": "<5 mmol/mol Cr", "mmab_range": "300–5,000 mmol/mol Cr", "significance": "PATHOGNOMONIC — slightly higher range than MMAA (fewer responders → less MMA reduction); absent in PA (KEY DIFFERENTIAL from PA)", "method": "Urine organic acids (GC-MS)"},
            {"name": "Methylmalonic Acid (plasma)", "normal": "<0.4 µmol/L", "mmab_range": "100–2,500 µmol/L", "significance": "Diagnostic confirmation; monitor on OHCbl trial (fall >50% at Day 3–5 = responsive); higher than MMAA baseline due to lower response rate", "method": "Plasma MMA (LC-MS/MS or GC-MS)"},
            {"name": "Methylcitrate (urine)", "normal": "<5 µmol/mmol Cr", "mmab_range": "<50 µmol/mmol Cr (mild secondary)", "significance": "MILDLY elevated secondary only — NOT pathognomonic; pathognomonic in PA (50–2,000 µmol/mmol Cr); KEY DIFFERENTIAL from PA", "method": "Urine organic acids (GC-MS)"},
            {"name": "Ammonia (plasma)", "normal": "<50 µmol/L", "mmab_range": "100–1,000 µmol/L", "significance": "Secondary hyperammonemia via NAGS/CPS1 inhibition; higher peak in neonatal severe MMAB than MMAA (lower MMA correction); NH3 >300 = crisis", "method": "Plasma ammonia (enzymatic)"},
            {"name": "Free Carnitine", "normal": "25–60 µmol/L", "mmab_range": "6–30 µmol/L", "significance": "DEPLETED secondary — MMA-carnitine (propionylcarnitine) excreted; lower than MMAA in non-responders because MMA is not controlled by OHCbl", "method": "Plasma acylcarnitines"},
            {"name": "Homocysteine (total)", "normal": "<15 µmol/L", "mmab_range": "NORMAL <15 µmol/L", "significance": "NORMAL — KEY NEGATIVE vs cblC (MMACHC): MMAB causes isolated MMA only; cblC causes BOTH MMA + homocystinuria; gene panel mandatory", "method": "Plasma tHcy (HPLC or immunoassay)"},
            {"name": "MMUT Enzyme (fibroblasts + AdoCbl)", "normal": ">200 nmol/h/mg", "mmab_range": "NORMAL when AdoCbl provided in vitro", "significance": "KEY: MMUT apoenzyme INTACT in MMAB (same as MMAA) — corrects with exogenous AdoCbl; distinguishes from MMUT mut0 (which does NOT correct)", "method": "Fibroblast enzyme assay ± AdoCbl (metabolic genetics lab)"},
            {"name": "MMAB Enzyme Activity", "normal": ">5 nmol/h/mg protein", "mmab_range": "ABSENT or <1 nmol/h/mg (primary defect)", "significance": "KEY PRIMARY DEFECT — MMAB adenosyltransferase activity absent/severely reduced; distinguishes MMAB from MMAA (where MMAB activity is NORMAL)", "method": "Fibroblast/lymphocyte MMAB enzyme assay (specialized metabolic lab)"},
            {"name": "OHCbl Trial Response", "normal": "N/A", "mmab_range": "~40–60% show MMA reduction >50% at Day 3–5", "significance": "MODERATE response (LOWER than MMAA ~60–80%; HIGHER than MMUT ~20%); defines MMAB pharmacological responsiveness; measure plasma MMA at Day 0 and Day 5 of OHCbl 1–2 mg IM daily", "method": "Plasma MMA before/after 3–5 days OHCbl 1–2 mg/day IM"},
            {"name": "eGFR / Creatinine", "normal": ">90 mL/min/1.73m²", "mmab_range": "25–100 mL/min/1.73m²", "significance": "CKD more prevalent than MMAA (~22%) due to lower OHCbl response rate; similar mechanism (direct MMA renal tubular toxicity); progressive in non-responders", "method": "Serum creatinine; cystatin C; 6-monthly monitoring"},
            {"name": "C3/C2 Ratio", "normal": "<0.04", "mmab_range": ">0.08", "significance": "Secondary NBS flag; not disease-specific (same in PA, MMUT, MMAA); triggers urine organic acid workup", "method": "NBS tandem MS/MS"},
        ],
        "key_variants": [
            {"variant": "p.Arg190His", "cdna": "c.569G>A", "domain": "Adenosyltransferase active site", "severity": "Moderate (OHCbl partial response)", "note": "Most common MMAB allele worldwide; located at trimeric active site interface; partial adenosyltransferase activity retained; partial OHCbl response; European/worldwide"},
            {"variant": "p.Arg190Cys", "cdna": "c.568C>T", "domain": "Adenosyltransferase active site", "severity": "Severe (non-responsive)", "note": "Same codon as p.Arg190His; different substitution → abolishes active site geometry; non-responsive to OHCbl; neonatal severe presentation"},
            {"variant": "p.Arg206Cys", "cdna": "c.616C>T", "domain": "MMAB trimeric interface", "severity": "Moderate-Severe", "note": "Disrupts homotrimeric MMAB assembly; partially destabilizes trimeric active site; moderate-severe; variable OHCbl response"},
            {"variant": "p.Arg234Trp", "cdna": "c.700C>T", "domain": "Substrate (cob(I)alamin) binding channel", "severity": "Severe", "note": "Blocks cob(I)alamin access to active site; no product formed; non-responsive; neonatal-infantile severe"},
            {"variant": "p.Gly188Arg", "cdna": "c.562G>A", "domain": "Catalytic Gly adjacent to active site", "severity": "Severe (neonatal)", "note": "Gly188 structurally critical for active site geometry; steric clash with Arg → complete enzyme loss; neonatal severe; non-responsive"},
            {"variant": "c.IVS7+1G>A", "cdna": "Intron 7 splice site", "domain": "Splice disruption — null", "severity": "Severe (null)", "note": "Splice site mutation abolishes MMAB mRNA; null allele; no enzyme produced; severe neonatal; non-responsive"},
            {"variant": "Large del exon5-6", "cdna": "MLPA confirmed", "domain": "Adenosyltransferase core", "severity": "Severe (null)", "note": "~8% of MMAB alleles; genomic deletion removing catalytic core exons; null allele; MLPA/CNV array needed for detection; severe non-responsive"},
            {"variant": "p.Trp202*", "cdna": "c.606G>A", "domain": "Truncating null (trimeric core)", "severity": "Severe (null)", "note": "Nonsense mutation; truncated MMAB protein unstable/non-functional; null; severe neonatal"},
        ],
        "seizure_types": [
            {"type": "Hyperammonemic encephalopathy seizures", "pct": 65, "note": "Most common in neonatal severe; multifocal/generalized; EEG burst-suppression pattern; FIRST: correct MMA + NH3 with IV glucose + OHCbl; higher NH3 peaks than MMAA"},
            {"type": "Focal cortical seizures", "pct": 25, "note": "Basal ganglia infarct-related in non-responders; striatal injury during decompensation; more frequent than MMAA (lower OHCbl response → more basal ganglia damage episodes)"},
            {"type": "Infantile spasms (West syndrome)", "pct": 20, "note": "More frequent than MMAA (higher neonatal severe rate); OHCbl non-responsive cases at particular risk; ACTH or vigabatrin if metabolically stabilized; LEV adjunct"},
            {"type": "Myoclonic seizures", "pct": 25, "note": "Metabolic-triggered; correlate with MMA/NH3 spikes; improve with metabolic management; LEV first-line AED; avoid VPA"},
            {"type": "Absence seizures", "pct": 12, "note": "Late-onset attenuated phenotype; correlates with metabolic control; less common than encephalopathic types"},
            {"type": "Status epilepticus", "pct": 18, "note": "Metabolic SE during hyperammonemic crisis; higher rate than MMAA (more severe neonatal presentation); EEG mandatory; IV glucose + BZD + metabolic correction; avoid VPA"},
        ],
        "metabolic_triggers": [
            {"trigger": "Fasting / prolonged NPO", "pct": 72, "mechanism": "BCAA catabolism → propionyl-CoA → L-methylmalonyl-CoA → MMA surge; IV glucose mandatory; do NOT omit OHCbl during fasting procedures"},
            {"trigger": "Intercurrent infection / fever", "pct": 65, "mechanism": "Catabolic state; MMA and NH3 surge together; IV glucose protocol mandatory; MMAB patients particularly vulnerable (less metabolic reserve in non-responders)"},
            {"trigger": "Missed OHCbl doses", "pct": 40, "mechanism": "MMA rebounds within 3–7 days; especially hazardous in partial responders (who rely on OHCbl to maintain metabolic control)"},
            {"trigger": "High-protein meal / formula error", "pct": 42, "mechanism": "Ile/Val/Met/Thr → propionyl-CoA accumulation → L-methylmalonyl-CoA → MMA"},
            {"trigger": "Surgery / anesthesia", "pct": 32, "mechanism": "NPO + catabolic stress; IV glucose perioperatively; continue IM OHCbl; metabolic team co-management mandatory"},
        ],
        "high_risk_drugs": [
            {"drug": "Valproate (VPA)", "risk": "ABSOLUTE CI", "mechanism": "VPA CoA-thioester metabolites directly inhibit MMAB adenosyltransferase activity (the primary deficient enzyme in MMAB disease) + carnitine depletion + worsens MMA: FATAL — use LEV as first-line AED"},
            {"drug": "Fasting / NPO without IV glucose", "risk": "EXTREME HAZARD", "mechanism": "BCAA catabolism surge → MMA crisis; always IV dextrose during illness or fasting procedures"},
            {"drug": "High-protein diet / excess Ile/Val/Met/Thr", "risk": "EXTREME HAZARD", "mechanism": "Direct propionyl-CoA → L-methylmalonyl-CoA → MMA; avoid high-protein supplements"},
            {"drug": "Metformin", "risk": "AVOID", "mechanism": "Complex I inhibitor → synergistic mitochondrial toxicity with MMA → lactic acidosis risk"},
            {"drug": "Missed IM OHCbl doses", "risk": "HIGH RISK", "mechanism": "MMA rebounds within 3–7 days; OHCbl compliance monitoring and parent education mandatory"},
        ],
        "treatments": [
            {"treatment": "Hydroxocobalamin (OHCbl) 1–2 mg/day IM", "evidence": "Level A", "response_pct": 50, "note": "Trial ALL MMAB patients; ~40–60% respond (MMA reduction >50% at Day 3–5); MODERATE response — lower than MMAA cblA (~60–80%) because synthesis defect is harder to bypass than delivery defect; lifelong IM preferred"},
            {"treatment": "Protein restriction (low Ile/Val/Met/Thr)", "evidence": "Level A", "response_pct": 85, "note": "Target plasma Ile <40 µmol/L; Val <150 µmol/L; natural protein 0.8–1.8 g/kg/day + MMAB-free formula; lifelong"},
            {"treatment": "MMAB-free amino acid formula", "evidence": "Level A", "response_pct": 85, "note": "Provides protein without Ile/Val/Met/Thr propiogenic precursors; essential alongside protein restriction; lifelong"},
            {"treatment": "L-Carnitine 100 mg/kg/day", "evidence": "Level A", "response_pct": 88, "note": "Secondary depletion; MMA-carnitine (propionylcarnitine C3) renal excretion; more supplementation often needed in non-responders vs MMAA"},
            {"treatment": "IV Glucose GIR 8–12 + Insulin", "evidence": "Level A", "response_pct": 80, "note": "Acute crisis FIRST intervention; stops catabolism → MMA surge halted; STOP protein during acute phase; continue OHCbl IM"},
            {"treatment": "Ammonia Scavengers (Na benzoate / Na phenylacetate)", "evidence": "Level A", "response_pct": 72, "note": "If NH3 >200 µmol/L; IV Na benzoate 250 mg/kg loading; same protocol as MMAA/MMUT/PA secondary hyperammonemia"},
            {"treatment": "Metronidazole", "evidence": "Level B", "response_pct": 25, "note": "Reduces gut bacterial propionate 20–30%; short 2–4 week courses; avoidchronic use (neurotoxicity risk)"},
            {"treatment": "Liver transplant", "evidence": "Level B", "response_pct": 65, "note": "More commonly needed than MMAA (lower OHCbl response rate in MMAB); corrects hepatic MMA production; does not protect extra-hepatic MMAB-expressing tissues; reserved for OHCbl non-responders"},
            {"treatment": "HD / CRRT", "evidence": "Level B", "response_pct": 65, "note": "MMA >3,000 mmol/mol Cr or NH3 >500 µmol/L; MMA dialyzable; bridge to metabolic stabilization; consider in neonatal severe MMAB more readily than MMAA"},
            {"treatment": "VPA (Valproate)", "evidence": "ABSOLUTE CI", "response_pct": 0, "note": "ABSOLUTE CONTRAINDICATION: VPA inhibits MMAB enzyme (primary defect!) + carnitine depletion + worsens MMA: FATAL — use LEV as first-line AED for all seizures"},
        ],
        "patient_sample": sample,
    }


def get_definitions() -> dict:
    return {
        "gene_card": {
            "Gene": "MMAB (ATP:cob(I)alamin adenosyltransferase; cblB gene product)",
            "Full Name": "ATP:cob(I)alamin Adenosyltransferase (Cobalamin B / Complementation Group cblB)",
            "Chromosome": "12q24.11",
            "Inheritance": "Autosomal Recessive (AR)",
            "OMIM Gene": "*607568",
            "OMIM Disease": "#251110 (Methylmalonic Acidemia, cblB type)",
            "Protein Size": "250 amino acids; forms homotrimer; mitochondrial matrix localization",
            "Structure": "Homotrimeric adenosyltransferase; active site lies at interface of adjacent monomers; Arg190 is the key catalytic residue",
            "Reaction": "cob(I)alamin + ATP → AdoCbl + PPPi (adenosylation; cofactor synthesis)",
            "Pathway Position": "Step 2 of 4 in the cobalamin-to-MMUT pathway (after MMACHC; before MMAA; before MMUT)",
            "Prevalence": "~1:100,000–250,000; accounts for ~15–20% of cobalamin-responsive isolated MMA",
            "NBS": "C3 (propionylcarnitine) elevated — same trigger as MMAA; urine organic acids MANDATORY",
            "Key Positive": "Urine MMA 300–5,000 mmol/mol Cr (PATHOGNOMONIC); OHCbl response ~40–60% MODERATE",
            "Key Negative": "Homocysteine NORMAL (vs cblC); BCAA NORMAL; Methylcitrate NOT pathognomonic (vs PA); MMUT apoenzyme INTACT when AdoCbl provided; MMAB enzyme activity ABSENT (primary defect)",
        },
        "key_concepts": [
            {
                "concept": "Why MMAB causes MMA without MMUT or MMAA enzyme defect — synthesis vs delivery distinction",
                "explanation": (
                    "The cobalamin pathway to MMUT has four steps: (1) MMACHC makes cob(I)alamin; "
                    "(2) MMAB synthesizes AdoCbl from cob(I)alamin + ATP; "
                    "(3) MMAA delivers AdoCbl from MMAB to MMUT; (4) MMUT uses AdoCbl to isomerize "
                    "L-methylmalonyl-CoA → succinyl-CoA. MMAB LOF: AdoCbl cannot be MADE → "
                    "MMAA chaperone has nothing to deliver → MMUT apoenzyme is cofactor-empty "
                    "(STRUCTURALLY INTACT). Critical: adding exogenous AdoCbl to MMAB patient fibroblasts "
                    "CORRECTS MMUT enzyme activity (apoenzyme is fine). This is the same as MMAA — "
                    "both show MMUT correction with AdoCbl; both have isolated MMA without HHcy. "
                    "The DIFFERENCE: in MMAA, MMAB synthesizes AdoCbl normally (delivery fails); "
                    "in MMAB, the synthesis step itself fails. MMAB enzyme assay is the primary distinguisher."
                ),
            },
            {
                "concept": "Why MMAB OHCbl response is MODERATE (~40–60%), lower than MMAA (~60–80%)",
                "explanation": (
                    "Hydroxocobalamin (OHCbl) pharmacological therapy partially compensates for both "
                    "MMAA (delivery defect) and MMAB (synthesis defect) by flooding the cobalamin pathway. "
                    "In MMAA: high cobalamin flux increases AdoCbl production by MMAB → more cargo for "
                    "MMAA to attempt delivery via alternative routes → relatively efficient bypass. "
                    "In MMAB: the synthesis step IS the defect → high cobalamin substrate (cob(I)alamin) "
                    "accumulates but cannot be converted to AdoCbl without MMAB → alternative synthesis "
                    "routes are less efficient → lower fraction of patients achieve MMA reduction >50%. "
                    "Consequently: MMAB response ~40–60% MODERATE vs MMAA ~60–80% STRONG. "
                    "Both are more responsive than MMUT (~20%), because MMUT apoenzyme is intact in MMAB. "
                    "Trial OHCbl in ALL MMAB patients — predictors of response: p.Arg190His allele, "
                    "later onset phenotype, residual MMAB activity."
                ),
            },
            {
                "concept": "Differentiating MMAB (cblB) from MMAA (cblA) — the two cobalamin-responsive isolated MMA subtypes",
                "explanation": (
                    "MMAB and MMAA are biochemically nearly identical: both isolated MMA, normal Hcy, "
                    "normal BCAA, normal methylcitrate (relative to PA), and correctable MMUT fibroblast activity. "
                    "Key distinguishing features: "
                    "• OHCbl response: MMAB ~40–60% MODERATE; MMAA ~60–80% STRONG (HALLMARK of cblA). "
                    "• MMAB enzyme assay: ABSENT in MMAB; NORMAL in MMAA. "
                    "• Pathway position: MMAB is step 2 (synthesis); MMAA is step 3 (delivery). "
                    "• Phenotype: MMAB has ~35% neonatal severe (vs MMAA ~25%) due to higher null allele frequency. "
                    "• Gene panel: MMAB (12q24.11) vs MMAA (4p14) — mandatory NGS gene panel. "
                    "Biochemistry alone CANNOT reliably distinguish — gene panel is mandatory."
                ),
            },
            {
                "concept": "CKD in MMAB — intermediate severity between MMAA and MMUT",
                "explanation": (
                    "MMA is a direct renal proximal tubular toxin in all MMA disorders. "
                    "MMAB: ~22% CKD rate — INTERMEDIATE between MMAA (~18%, better OHCbl response → "
                    "lower chronic MMA) and MMUT mut0 (~35%, non-responsive → highest chronic MMA). "
                    "MMAB non-responders (~40–60%) accumulate MMA chronically → progressive CKD risk "
                    "similar to MMUT. MMAB OHCbl responders have lower chronic MMA → less CKD. "
                    "Monitoring: eGFR + urine protein every 6 months; nephrology co-management for eGFR <60. "
                    "Kidney transplant: may be needed in MMAB non-responders more than MMAA, "
                    "less than MMUT mut0 where combined liver-kidney Tx is standard."
                ),
            },
            {
                "concept": "VPA ABSOLUTE CI in MMAB — mechanism directly targeting the deficient enzyme",
                "explanation": (
                    "Valproate (VPA) is ABSOLUTELY CONTRAINDICATED in MMAB. In MMAB specifically, "
                    "VPA's metabolic toxicity is particularly severe because VPA-CoA thioester "
                    "metabolites directly inhibit MMAB adenosyltransferase — THE VERY ENZYME that is "
                    "deficient in MMAB disease. This means VPA does not just add insult to injury — "
                    "it directly worsens the primary enzymatic block. Additionally: "
                    "VPA massively depletes carnitine via VPA-carnitine conjugation → worsens existing "
                    "secondary carnitine depletion. VPA also inhibits mitochondrial β-oxidation. "
                    "Combination → MMA ↑↑↑ + carnitine depletion + hepatic failure: FATAL. "
                    "LEV (levetiracetam) is first-line AED for ALL seizure types in MMAB patients."
                ),
            },
            {
                "concept": "MMAB homotrimeric structure — why Arg190 is the critical active site residue",
                "explanation": (
                    "MMAB forms a homotrimer; the active site for ATP:cob(I)alamin adenosyltransferase "
                    "activity lies at the interface of ADJACENT monomers. Arg190 (most common variant: "
                    "p.Arg190His and p.Arg190Cys) is located at this trimeric active site and: "
                    "(1) coordinates the cob(I)alamin substrate for adenylation; "
                    "(2) participates in ATP binding and positioning for the adenyl transfer reaction; "
                    "(3) stabilizes the transition state for the nucleophilic attack of "
                    "cob(I)alamin on the 5′-carbon of ATP's adenosyl group. "
                    "p.Arg190His: maintains partial positive charge (His+) at physiological pH → "
                    "partial activity → moderate OHCbl response (~50%). "
                    "p.Arg190Cys: no charge → complete active site disruption → non-responsive → severe."
                ),
            },
            {
                "concept": "Secondary hyperammonemia in MMAB — NAGS/CPS1 mechanism shared with MMAA, MMUT, and PA",
                "explanation": (
                    "In MMAB, L-methylmalonyl-CoA accumulates (downstream of MMAB block). "
                    "MMA and methylmalonyl-CoA inhibit N-acetylglutamate synthase (NAGS) → "
                    "reduced NAG → CPS1 (carbamoyl phosphate synthetase 1) downregulation → "
                    "urea cycle impaired → secondary hyperammonemia. "
                    "This is NOT primary hyperammonemia — resolves when MMA is corrected. "
                    "MMAB neonatal severe has higher NH3 peaks than MMAA neonatal severe because "
                    "OHCbl response rate is lower → MMA less rapidly corrected → NH3 persists longer. "
                    "Treat: IV glucose (first) + ammonia scavengers (Na benzoate if NH3 >200) + OHCbl. "
                    "HD/CRRT if NH3 >500 µmol/L."
                ),
            },
        ],
        "diagnostic_thresholds": [
            {"parameter": "Urine MMA", "threshold": ">300 mmol/mol Cr — diagnostic of MMAB (and all MMA)", "action": "Confirm with plasma MMA; start OHCbl trial; send gene panel (MMUT/MMAA/MMAB/MMACHC/MMADHC)"},
            {"parameter": "OHCbl trial response", "threshold": ">50% MMA reduction at Day 3–5 = B12-responsive", "action": "Confirmed responder: lifelong OHCbl IM; oral cyanocobalamin as maintenance option"},
            {"parameter": "Plasma MMA", "threshold": ">100 µmol/L diagnostic; >1,000 µmol/L = crisis threshold", "action": "Monitor OHCbl response; monthly in first year; 3-monthly if stable"},
            {"parameter": "MMAB enzyme activity", "threshold": "<1 nmol/h/mg (reference >5); absent in severe alleles", "action": "Absent activity + intact MMUT (with AdoCbl) = MMAB diagnosis confirmed; gene panel for variant characterization"},
            {"parameter": "Ammonia", "threshold": "NH3 >200 µmol/L = start ammonia scavengers; >500 = HD/CRRT", "action": "IV Na benzoate 250 mg/kg; IV glucose GIR 8–12; stop protein; continue OHCbl IM"},
            {"parameter": "eGFR", "threshold": "<60 mL/min/1.73m² = CKD Stage 3+", "action": "Nephrology co-management; consider Tx evaluation; optimize MMA control to slow CKD progression"},
            {"parameter": "Free carnitine", "threshold": "<20 µmol/L = supplement; <10 = crisis-level depletion", "action": "L-carnitine 100–200 mg/kg/day orally; IV in crisis; monitor acylcarnitine ratio"},
        ],
        "differential_diagnosis": [
            {"disease": "MMAA (cblA) — GTPase chaperone defect", "distinguishing": "Isolated MMA identical biochemically; MMAA OHCbl response STRONGER (~60–80% vs MMAB ~40–60%); MMAB enzyme NORMAL in MMAA; MMAA (4p14) vs MMAB (12q24.11); gene panel mandatory to distinguish"},
            {"disease": "MMUT (mut0/mut-) — methylmalonyl-CoA mutase apoenzyme defect", "distinguishing": "Isolated MMA; MMUT fibroblast activity ABSENT without AdoCbl (NOT correctable in mut0); OHCbl response only in mut- ~20%; MMUT (6p12.3) by gene panel"},
            {"disease": "MMACHC (cblC) — combined MMA + homocystinuria", "distinguishing": "BOTH MMA elevated AND homocysteine elevated (HHcy >30 µmol/L) — KEY POSITIVE distinguishing from MMAB; MMACHC (1p34.1)"},
            {"disease": "MMADHC (cblD) — combined or isolated MMA variant", "distinguishing": "Can cause isolated MMA OR combined MMA+HHcy depending on allele; gene panel: MMADHC (2q23.2)"},
            {"disease": "PCCA/PCCB (Propionic Acidemia)", "distinguishing": "Methylcitrate PATHOGNOMONIC in PA (50–2,000 µmol/mmol Cr) vs MMAB absent/mild; NO MMA elevation (MMA is KEY NEGATIVE in PA); methylcitrate and methylmalonate are biochemically distinct"},
            {"disease": "SUCLA2 / SUCLG1 (succinyl-CoA ligase deficiency)", "distinguishing": "Elevated MMA + elevated methylmalonate + SENSORINEURAL HEARING LOSS + early-onset severe encephalopathy; no cobalamin response; SUCLA2 (13q14.2) / SUCLG1 (2p11.2)"},
            {"disease": "DLD (dihydrolipoamide dehydrogenase, E3 deficiency)", "distinguishing": "COMBINED elevation: lactate + BCAA (leu/ile/val) + 2-hydroxyglutarate + mild glycine; BCAA is NORMAL in MMAB (KEY NEGATIVE); DLD (7q31.1)"},
        ],
    }
