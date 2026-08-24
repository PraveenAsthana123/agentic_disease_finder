#!/usr/bin/env python3
"""MMUT (Methylmalonyl-CoA Mutase / Methylmalonic Acidemia) Epilepsy Dashboard.

MMUT encodes methylmalonyl-CoA mutase (MCM), a mitochondrial enzyme that converts
L-methylmalonyl-CoA to succinyl-CoA, feeding the TCA cycle.

  METHYLMALONYL-CoA MUTASE (MMUT) — ENZYME BIOLOGY:
    MMUT protein: 750 aa, homodimer, mitochondrial matrix. Each monomer carries one
    adenosylcobalamin (AdoCbl) cofactor in the N-terminal TIM-barrel domain.
    Catalytic mechanism: radical-based isomerization.
    Step: L-methylmalonyl-CoA ⟶ succinyl-CoA
    This step links propionyl-CoA catabolism (from Ile, Val, Met, Thr, odd-chain FA,
    cholesterol) to the TCA cycle via succinyl-CoA.

  UPSTREAM PATHWAY:
    Propionyl-CoA (from BCAA: Ile, Val, Met, Thr; odd-chain FA; cholesterol)
      → Propionyl-CoA + HCO3- → D-methylmalonyl-CoA [PCC: PCCA+PCCB]
      → D-methylmalonyl-CoA → L-methylmalonyl-CoA [MCEE: epimerase]
      → L-methylmalonyl-CoA → succinyl-CoA [MMUT + AdoCbl] ← BLOCK HERE
    Succinyl-CoA → TCA (→ fumarate → malate → OAA → citrate cycle)

  MMUT LOF → L-methylmalonyl-CoA ACCUMULATES → METHYLMALONIC ACID (MMA) ELEVATED:
    MMA in urine: 200–10,000+ mmol/mol Cr (PATHOGNOMONIC — absent in PA/PCCA/PCCB)
    Secondary propionyl-CoA buildup → MILD methylcitrate (secondary, LOW vs PA where it is PATHOGNOMONIC)
    MMA inhibits CPS1 via NAGS (same as PA) → SECONDARY HYPERAMMONEMIA
    MMA: mitochondrial respiratory chain inhibitor (complex I/III/V) → cardiomyopathy
    MMA: direct renal tubular toxicant → PROGRESSIVE CKD (UNIQUE to MMA, not PA)
    MMA: optic nerve → OPTIC ATROPHY (unique, ~25% late-diagnosed)

  PHENOTYPIC SUBTYPES (MMUT allele categories):
    mut0 (null, no residual activity): ~50% — Classic Neonatal Severe, MMA >5,000, NH3 >500,
      NO AdoCbl response; combined liver-kidney transplant best long-term outcome
    mut- (partial residual activity): ~30% — variable onset, may show partial B12 response
      if residual apoenzyme present; mut- alleles include p.Arg694Trp, p.Arg694Gln, mild missense
    AdoCbl-partial-responsive mut-: ~20% — later onset, OHCbl 1 mg/day IM trial shows MMA fall
      >50% from baseline; still needs dietary restriction lifelong

  KEY BIOMARKERS:
    C3 (propionylcarnitine): ↑ — NBS PRIMARY (same as PA/PCCA/PCCB; cannot distinguish by NBS alone)
    Methylmalonic acid (urine): ↑↑↑ PATHOGNOMONIC — 200–10,000+ mmol/mol Cr
    Methylmalonic acid (plasma): ↑↑ — 200–5,000 µmol/L
    Methylcitrate: mildly elevated secondary (ABSENT as PATHOGNOMONIC finding — KEY NEG vs PA)
    Ammonia: ↑ secondary hyperammonemia (same CPS1/NAGS mechanism as PA)
    Free carnitine: ↓ (secondary depletion; MMA-carnitine conjugate excreted)
    Homocysteine: NORMAL (KEY NEGATIVE vs cblC/MMACHC — distinguishes MMUT from cobalamin combined defects)
    Glycine: mildly ↑ (mitochondrial dysfunction secondary)
    Lactate: can be ↑ (respiratory chain inhibition)
    GFR: ↓ progressive CKD (UNIQUE feature of MMA — not seen in PA)
    Propionylglycine: present mildly (secondary propionyl-CoA spill)
    3-Hydroxypropionate: mild secondary (LOW vs PA)

  KEY NEGATIVES (distinguishing MMUT from look-alikes):
    - Methylcitrate NOT pathognomonic (vs PA/PCCA/PCCB — absent as PRIMARY marker)
    - Homocysteine NORMAL (vs cblC/MMACHC — combined MMA+HHcy in cblC, isolated MMA in MMUT)
    - BCAA NORMAL (vs DLD/MSUD — no BCKDH block in isolated MMA)
    - Biotinidase NORMAL (vs BTD)
    - C5-OH NORMAL (vs HLCS/BTD/MCD)
    - Biotin NOT effective (MMUT enzyme mutant — not a biotin disorder)
    - AdoCbl response: ONLY in mut- partial alleles (~20%); mut0 does NOT respond

  FOUNDER VARIANTS IN MMUT:
    - c.655A>T (p.Arg219*): mut0 null; pan-ethnic; most commonly identified mut0 allele
    - c.682C>T (p.Arg228*): mut0 null; pan-ethnic (often mis-labeled p.Arg228Ter)
    - c.2080C>T (p.Arg694Trp): mut- partial; European/Middle Eastern; AdoCbl-responsive subset
    - c.1106G>A (p.Arg369His): mut0; European; active site vicinity
    - c.2150G>A (p.Arg717Gln): mut-; intermediate activity; B12 partial response
    - p.Gly630Glu: mut0; pan-European; severe
    - Large deletions (MLPA): 5–8% of MMUT alleles; mut0; neonatal severe
    - c.1106G>C (p.Arg369Pro): mut0; Saudi Arabian/Middle Eastern; severe

  TREATMENT HIGHLIGHTS:
    Hydroxocobalamin OHCbl 1 mg/day IM (Level A — trial ALL patients 3–5 days; only mut- respond)
    Protein restriction — low Ile/Val/Met/Thr (Level A; lifelong)
    MMUT-free amino acid formula (Level A)
    L-Carnitine 100 mg/kg/day (Level A; secondary depletion + conjugate excretion)
    IV glucose GIR 8–12 + insulin (Level A; acute crisis first intervention)
    Ammonia scavengers (Na benzoate/Na phenylacetate, Level A if NH3 >200)
    Metronidazole (Level B; reduces gut bacterial propionate 20–30%)
    Kidney transplant (Level B; corrects CKD; reduces daily MMA but does NOT normalize — hepatic MMUT absent)
    Combined liver-kidney transplant (Level B; best for mut0 — corrects hepatic enzyme + CKD; MMA near-normalizes)
    HD/CRRT (Level B; NH3 >500 or MMA >5,000 acute)
    VPA — ABSOLUTE CI (inhibits mitochondrial MMUT cofactor metabolism; carnitine depletion; MMA↑↑; FATAL)
    Fasting — EXTREME HAZARD (BCAA catabolism → propionyl-CoA → methylmalonyl-CoA → MMA surge)
    High protein — EXTREME HAZARD
    Metformin — AVOID (Complex I inhibitor → MMA + lactic acidosis synergy)

  OMIM: Gene *609058 · Disease #251000 (Methylmalonic Acidemia, mut type)
  Chromosome: 6p12.3 · Inheritance: AR · Prevalence: ~1:50,000–80,000 (combined MMA subtypes)
"""

from __future__ import annotations
import random

random.seed(47)   # reproducible synthetic cohort


# ── helpers ──────────────────────────────────────────────────────────────────

def _pid(i: int) -> str:
    return f"MMUT-{i:03d}"


def _sex(i: int) -> str:
    return "M" if i % 2 == 0 else "F"


def _phenotype(i: int) -> str:
    r = i % 20
    if r < 10:  return "mut0 Classic Neonatal Severe"
    if r < 14:  return "mut- AdoCbl Partial Responsive"
    if r < 18:  return "mut- Attenuated Late Infantile"
    return "mut- Mild Late Onset"


def _genotype(i: int) -> str:
    variants = [
        "c.655A>T/c.655A>T",             # p.Arg219* homozygous mut0
        "c.682C>T/c.1106G>A",            # p.Arg228* / p.Arg369His compound het mut0
        "c.655A>T/c.682C>T",             # two null alleles, severe mut0
        "c.2080C>T/c.2080C>T",           # p.Arg694Trp homozygous mut-
        "c.655A>T/c.2080C>T",            # mut0/mut- compound het
        "c.1106G>A/c.1106G>A",           # mut0 homozygous European
        "c.2080C>T/c.2150G>A",           # two mut- alleles, partial
        "c.655A>T/c.2150G>A",            # mut0/mut- attenuated
        "del.exon1-3/c.655A>T",          # large deletion / null
        "c.1106G>A/c.2080C>T",           # mut0/mut- compound het
    ]
    return variants[i % len(variants)]


def _mma_urine(pheno: str) -> float:
    """Methylmalonic acid urine mmol/mol Cr."""
    if pheno == "mut0 Classic Neonatal Severe":
        return round(random.uniform(2000, 10000), 0)
    if pheno == "mut- AdoCbl Partial Responsive":
        return round(random.uniform(400, 1500), 0)
    if pheno == "mut- Attenuated Late Infantile":
        return round(random.uniform(200, 800), 0)
    return round(random.uniform(100, 400), 0)


def _c3(pheno: str) -> float:
    if pheno == "mut0 Classic Neonatal Severe":
        return round(random.uniform(10, 40), 1)
    if pheno == "mut- AdoCbl Partial Responsive":
        return round(random.uniform(4, 18), 1)
    if pheno == "mut- Attenuated Late Infantile":
        return round(random.uniform(3, 12), 1)
    return round(random.uniform(2, 8), 1)


def _ammonia(pheno: str) -> int:
    if pheno == "mut0 Classic Neonatal Severe":
        return random.randint(200, 1200)
    if pheno == "mut- AdoCbl Partial Responsive":
        return random.randint(60, 300)
    if pheno == "mut- Attenuated Late Infantile":
        return random.randint(40, 150)
    return random.randint(25, 80)


def _carnitine(pheno: str) -> float:
    if pheno == "mut0 Classic Neonatal Severe":
        return round(random.uniform(5, 18), 1)
    if pheno == "mut- AdoCbl Partial Responsive":
        return round(random.uniform(10, 28), 1)
    if pheno == "mut- Attenuated Late Infantile":
        return round(random.uniform(12, 30), 1)
    return round(random.uniform(18, 38), 1)


def _gfr(pheno: str, i: int) -> int:
    """eGFR — CKD progressive in MMA (unique)."""
    if pheno == "mut0 Classic Neonatal Severe":
        return random.randint(15, 55)
    if pheno == "mut- AdoCbl Partial Responsive":
        return random.randint(30, 75)
    if pheno == "mut- Attenuated Late Infantile":
        return random.randint(50, 90)
    return random.randint(70, 110)


def _onset_months(pheno: str) -> int:
    if pheno == "mut0 Classic Neonatal Severe":
        return random.randint(0, 1)
    if pheno == "mut- AdoCbl Partial Responsive":
        return random.randint(1, 12)
    if pheno == "mut- Attenuated Late Infantile":
        return random.randint(6, 36)
    return random.randint(18, 120)


# ── patient table ─────────────────────────────────────────────────────────────

def _patients():
    pts = []
    for i in range(1, 41):
        pheno = _phenotype(i)
        pts.append({
            "id": _pid(i),
            "sex": _sex(i),
            "phenotype": pheno,
            "onset_age_months": _onset_months(pheno),
            "mma_urine_mmol_molCr": int(_mma_urine(pheno)),
            "c3_umol_l": _c3(pheno),
            "ammonia_umol_l": _ammonia(pheno),
            "free_carnitine_umol_l": _carnitine(pheno),
            "egfr_ml_min_1_73m2": _gfr(pheno, i),
            "seizures": pheno in ("mut0 Classic Neonatal Severe", "mut- AdoCbl Partial Responsive") and random.random() > 0.3,
            "cardiomyopathy": pheno == "mut0 Classic Neonatal Severe" and random.random() > 0.6,
            "optic_atrophy": pheno in ("mut0 Classic Neonatal Severe", "mut- Attenuated Late Infantile") and random.random() > 0.7,
            "nbs_detected": random.random() > 0.15,
            "adobcl_response": pheno in ("mut- AdoCbl Partial Responsive",) and random.random() > 0.3,
            "genotype": _genotype(i),
        })
    return pts


_PATIENTS = _patients()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def get_overview() -> dict:
    n = len(_PATIENTS)
    mma_vals = [p["mma_urine_mmol_molCr"] for p in _PATIENTS]
    c3_vals  = [p["c3_umol_l"] for p in _PATIENTS]
    nh3_vals = [p["ammonia_umol_l"] for p in _PATIENTS]
    car_vals = [p["free_carnitine_umol_l"] for p in _PATIENTS]
    egfr_vals= [p["egfr_ml_min_1_73m2"] for p in _PATIENTS]

    def pct(pred): return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    return {
        "gene": "MMUT",
        "full_name": "Methylmalonyl-CoA Mutase",
        "chromosome": "6p12.3",
        "inheritance": "AR",
        "omim_gene": "*609058",
        "omim_disease": "#251000",
        "protein_size": "750 aa; mitochondrial homodimer; AdoCbl cofactor TIM-barrel",
        "prevalence": "~1:50,000–80,000 (all MMA subtypes combined; MMUT ~60% of MMA)",
        "nbs_primary": "C3 (propionylcarnitine) elevated — triggers MMA workup (urine OA mandatory)",
        "nbs_secondary": "C3/C2 ratio; urine methylmalonic acid (PATHOGNOMONIC); plasma MMA",
        "function": (
            "MMUT catalyzes the radical-based adenosylcobalamin (AdoCbl)-dependent isomerization of "
            "L-methylmalonyl-CoA → succinyl-CoA, connecting propionyl-CoA catabolism "
            "(from Ile, Val, Met, Thr, odd-chain FA, cholesterol) to the TCA cycle via succinyl-CoA."
        ),
        "mechanism": (
            "MMUT LOF → L-methylmalonyl-CoA cannot be isomerized to succinyl-CoA → MMA accumulates "
            "(PATHOGNOMONIC: urine MMA 200–10,000+ mmol/mol Cr) → MMA inhibits CPS1 (via NAGS) → "
            "secondary hyperammonemia; MMA inhibits mitochondrial respiratory chain → cardiomyopathy; "
            "MMA directly damages renal tubules → progressive CKD (unique to MMA, not PA)."
        ),
        "key_negative": (
            "METHYLCITRATE absent as pathognomonic marker (vs PA/PCCA/PCCB where methylcitrate is "
            "PATHOGNOMONIC 50–2000 µmol/mmol Cr); HOMOCYSTEINE NORMAL (vs cblC/MMACHC combined MMA+HHcy); "
            "BCAA NORMAL (vs DLD/MSUD); Biotinidase NORMAL (vs BTD); Biotin NOT effective."
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
            "optic_atrophy_pct": pct(lambda p: p["optic_atrophy"]),
            "ckd_pct": pct(lambda p: p["egfr_ml_min_1_73m2"] < 60),
            "nbs_detected_pct": pct(lambda p: p["nbs_detected"]),
            "adocbl_response_pct": pct(lambda p: p["adobcl_response"]),
            "transplant_pct": pct(lambda p: p["phenotype"] == "mut0 Classic Neonatal Severe" and random.random() > 0.6),
        },
        "phenotype_distribution": [
            {"phenotype": "mut0 Classic Neonatal Severe (~50%)", "pct": 50},
            {"phenotype": "mut- AdoCbl Partial Responsive (~20%)", "pct": 20},
            {"phenotype": "mut- Attenuated Late Infantile (~20%)", "pct": 20},
            {"phenotype": "mut- Mild Late Onset (~10%)", "pct": 10},
        ],
        "mmut_pathway": [
            {
                "step": "Step 1 — Propionyl-CoA → D-methylmalonyl-CoA [PCC: PCCA+PCCB]",
                "reaction": "Propionyl-CoA + HCO3⁻ + ATP → D-methylmalonyl-CoA + ADP + Pᵢ",
                "enzyme": "Propionyl-CoA Carboxylase (PCCA BC domain + PCCB CT domain)",
                "cofactor": "Biotin (PCCA-Lys669)",
                "consequence_mmut_lof": "This step is INTACT in MMUT deficiency — propionyl-CoA is carboxylated normally",
            },
            {
                "step": "Step 2 — D-methylmalonyl-CoA → L-methylmalonyl-CoA [MCEE]",
                "reaction": "D-methylmalonyl-CoA ⇌ L-methylmalonyl-CoA",
                "enzyme": "Methylmalonyl-CoA Epimerase (MCEE)",
                "cofactor": "None",
                "consequence_mmut_lof": "This step is INTACT — but substrate L-methylmalonyl-CoA builds up as MMUT is blocked downstream",
            },
            {
                "step": "Step 3 — L-methylmalonyl-CoA → succinyl-CoA [MMUT] ← BLOCK",
                "reaction": "L-methylmalonyl-CoA → succinyl-CoA (AdoCbl radical mechanism)",
                "enzyme": "Methylmalonyl-CoA Mutase (MMUT / MCM)",
                "cofactor": "Adenosylcobalamin (AdoCbl) — vitamin B12 active form",
                "consequence_mmut_lof": "BLOCKED: L-methylmalonyl-CoA ACCUMULATES → MMA PATHOGNOMONIC ↑↑↑; succinyl-CoA DEPLETED → TCA anaplerosis lost",
            },
            {
                "step": "Step 4 — Succinyl-CoA → TCA [succinyl-CoA synthetase / SUCLA2/SUCLG]",
                "reaction": "Succinyl-CoA + GDP/ADP + Pᵢ → succinate + GTP/ATP",
                "enzyme": "Succinyl-CoA synthetase (SUCLA2/SUCLG1)",
                "cofactor": "None",
                "consequence_mmut_lof": "DEPLETED: TCA cycle anaplerosis via succinyl-CoA is lost → energy failure in high-flux tissues",
            },
        ],
        "mmut_vs_pa": {
            "title": "MMUT (MMA) vs PA (PCCA/PCCB) — 12-Feature Critical Differential",
            "note": (
                "NBS: both elevate C3 (propionylcarnitine) — CANNOT distinguish by NBS alone. "
                "Urine organic acids and plasma MMA MANDATORY for differential. "
                "Gene panel NGS distinguishes all: MMUT, PCCA, PCCB."
            ),
            "comparison": [
                {"feature": "NBS Trigger", "MMA_MMUT": "C3 ↑ (same as PA)", "PA_PCCAPCCB": "C3 ↑ (identical — cannot distinguish)"},
                {"feature": "Methylmalonic Acid (urine)", "MMA_MMUT": "↑↑↑ PATHOGNOMONIC 200–10,000 mmol/mol Cr", "PA_PCCAPCCB": "ABSENT (KEY NEG in PA)"},
                {"feature": "Methylcitrate (urine)", "MMA_MMUT": "Mild secondary only (<50 µmol/mmol Cr)", "PA_PCCAPCCB": "↑↑↑ PATHOGNOMONIC 50–2,000 µmol/mmol Cr"},
                {"feature": "Propionylglycine", "MMA_MMUT": "Mild secondary", "PA_PCCAPCCB": "↑ prominent (50–800 µmol/mmol Cr)"},
                {"feature": "Homocysteine", "MMA_MMUT": "NORMAL (KEY NEG vs cblC)", "PA_PCCAPCCB": "NORMAL"},
                {"feature": "Renal Involvement (CKD)", "MMA_MMUT": "YES — PROGRESSIVE CKD (unique)", "PA_PCCAPCCB": "Rare/Absent"},
                {"feature": "Cardiomyopathy", "MMA_MMUT": "~25–35% (mut0)", "PA_PCCAPCCB": "30–50% (propionyl-CoA)"},
                {"feature": "Optic Atrophy", "MMA_MMUT": "~25% (if late diagnosed)", "PA_PCCAPCCB": "Rare"},
                {"feature": "B12 (OHCbl) Response", "MMA_MMUT": "~20% (mut- partial only)", "PA_PCCAPCCB": "NOT EFFECTIVE"},
                {"feature": "Biotin Response", "MMA_MMUT": "NOT EFFECTIVE", "PA_PCCAPCCB": "NOT EFFECTIVE"},
                {"feature": "Liver Transplant Effect", "MMA_MMUT": "Corrects hepatic MMUT; CKD persists (Kidney Tx also needed)", "PA_PCCAPCCB": "75% hepatic PCC; cardiac NOT protected"},
                {"feature": "Gene Panel", "MMA_MMUT": "MMUT mandatory (MMAA/MMAB/MMADHC if cbl disorder)", "PA_PCCAPCCB": "PCCA + PCCB (biochemically identical — both mandatory)"},
            ],
        },
        "high_risk_situations": [
            {"situation": "VPA (Valproate)", "risk": "ABSOLUTE CI", "detail": "VPA inhibits mitochondrial MMUT cofactor metabolism + depletes carnitine + worsens MMA accumulation: FATAL in MMA patients"},
            {"situation": "Fasting", "risk": "EXTREME HAZARD", "detail": "BCAA catabolism surge → propionyl-CoA → methylmalonyl-CoA → MMA ↑↑ → crisis within hours"},
            {"situation": "High-protein diet", "risk": "EXTREME HAZARD", "detail": "Ile/Val/Met/Thr precursors of propionyl-CoA — directly drives MMA accumulation"},
            {"situation": "Intercurrent illness", "risk": "EXTREME HAZARD", "detail": "Catabolism → MMA surge + hyperammonemia; IV glucose protocol mandatory; do NOT withhold carnitine"},
            {"situation": "Metformin", "risk": "AVOID", "detail": "Complex I inhibitor — synergistic with MMA mitochondrial toxicity → lactic acidosis risk"},
            {"situation": "Biotin supplementation", "risk": "NOT EFFECTIVE", "detail": "MMUT deficiency is NOT a biotin disorder; biotin trial is not indicated (unlike HLCS/BTD)"},
        ],
    }


def get_breakdown() -> dict:
    sample = _PATIENTS[:15]
    return {
        "biomarkers": [
            {"name": "C3 (Propionylcarnitine)", "normal": "<4 µmol/L", "mma_range": "3–45 µmol/L", "significance": "NBS PRIMARY trigger; IDENTICAL to PA — cannot distinguish MMUT from PCCA/PCCB by NBS", "method": "NBS tandem MS/MS; plasma acylcarnitines"},
            {"name": "Methylmalonic Acid (urine)", "normal": "<5 mmol/mol Cr", "mma_range": "200–10,000+ mmol/mol Cr", "significance": "PATHOGNOMONIC — ELEVATED in MMUT, ABSENT in PA (KEY NEGATIVE for PA)", "method": "Urine organic acids (GC-MS)"},
            {"name": "Methylmalonic Acid (plasma)", "normal": "<0.4 µmol/L", "mma_range": "200–5,000 µmol/L", "significance": "Diagnostic confirmation; correlates with renal function — rises as CKD worsens", "method": "Plasma MMA (LC-MS/MS or GC-MS)"},
            {"name": "Methylcitrate (urine)", "normal": "<5 µmol/mmol Cr", "mma_range": "<50 µmol/mmol Cr (mild secondary)", "significance": "MILDLY elevated secondary (NOT pathognomonic in MMA); PATHOGNOMONIC in PA — KEY DIFFERENTIAL", "method": "Urine organic acids (GC-MS)"},
            {"name": "Ammonia (plasma)", "normal": "<50 µmol/L", "mma_range": "60–1,200 µmol/L", "significance": "Secondary hyperammonemia via NAGS/CPS1 inhibition (same as PA); NH3 >500 = crisis", "method": "Plasma ammonia (enzymatic)"},
            {"name": "Free Carnitine", "normal": "25–60 µmol/L", "mma_range": "5–30 µmol/L", "significance": "DEPLETED secondary — MMA-carnitine (C3-carnitine) excreted renally; L-carnitine replacement mandatory", "method": "Plasma acylcarnitines"},
            {"name": "Homocysteine (total)", "normal": "<15 µmol/L", "mma_range": "NORMAL <15 µmol/L", "significance": "NORMAL — KEY NEGATIVE vs cblC (MMACHC) where MMA+HHcy both elevated; distinguishes MMUT from cobalamin processing defects", "method": "Plasma tHcy (HPLC or immunoassay)"},
            {"name": "eGFR / Creatinine", "normal": ">90 mL/min/1.73m²", "mma_range": "15–90 mL/min/1.73m²", "significance": "Progressive CKD — UNIQUE to MMA (not PA); MMA is a direct renal tubular toxin; monitor every 6 months", "method": "Serum creatinine (CKD-EPI); cystatin C"},
            {"name": "Lactate / Pyruvate", "normal": "L<2 mmol/L; L/P 10–20", "mma_range": "L 2–10 mmol/L; L/P may rise", "significance": "MMA inhibits respiratory chain → lactic acidosis; L/P ratio less markedly elevated than PDH deficiency", "method": "Plasma lactate + pyruvate (SIMULTANEOUS draw, iced)"},
            {"name": "C3/C2 Ratio", "normal": "<0.04", "mma_range": ">0.10", "significance": "Secondary NBS flag; calculated acylcarnitine ratio — NOT disease-specific (elevated in PA and MMA equally)", "method": "NBS tandem MS/MS"},
            {"name": "Propionylglycine (urine)", "normal": "<10 µmol/mmol Cr", "mma_range": "Mild 10–100 µmol/mmol Cr", "significance": "MILD secondary overflow (vs PA 50–800 µmol/mmol Cr PROMINENT); indicates secondary propionyl-CoA buildup", "method": "Urine organic acids (GC-MS)"},
        ],
        "key_variants": [
            {"variant": "p.Arg219*", "cdna": "c.655A>T", "domain": "TIM-barrel (radical SAM region)", "severity": "Severe (mut0)", "note": "Pan-ethnic null; most commonly identified mut0 allele globally; NO AdoCbl response"},
            {"variant": "p.Arg228*", "cdna": "c.682C>T", "domain": "TIM-barrel", "severity": "Severe (mut0)", "note": "Pan-ethnic null; often compound het with other mut0; NO B12 response"},
            {"variant": "p.Arg694Trp", "cdna": "c.2080C>T", "domain": "C-terminal AdoCbl-binding", "severity": "Moderate (mut-)", "note": "European/Middle Eastern; partial residual activity; AdoCbl trial often partially responsive"},
            {"variant": "p.Arg369His", "cdna": "c.1106G>A", "domain": "Active site adjacency", "severity": "Severe (mut0)", "note": "European; active-site vicinity; no residual activity; combined liver-kidney Tx indicated"},
            {"variant": "p.Arg717Gln", "cdna": "c.2150G>A", "domain": "AdoCbl-binding domain", "severity": "Moderate (mut-)", "note": "Intermediate activity; partial B12 response; later onset; renal monitoring mandatory"},
            {"variant": "p.Gly630Glu", "cdna": "c.1889G>A", "domain": "TIM-barrel core", "severity": "Severe (mut0)", "note": "Pan-European; hydrophobic core disruption; null; severe neonatal presentation"},
            {"variant": "Large del exon 1-3", "cdna": "MLPA confirmed", "domain": "N-terminal / entire gene", "severity": "Severe (mut0)", "note": "5–8% of MMUT alleles; mut0 null; always severe; MLPA/CNV array needed for detection"},
            {"variant": "p.Arg369Pro", "cdna": "c.1106G>C", "domain": "Active site adjacency", "severity": "Severe (mut0)", "note": "Saudi Arabian/Middle Eastern founder; splice + structural disruption; severe neonatal"},
        ],
        "seizure_types": [
            {"type": "Metabolic encephalopathy seizures", "pct": 65, "note": "Multifocal/generalized during hyperammonemic crisis; EEG: burst-suppression pattern in neonates"},
            {"type": "Focal cortical seizures", "pct": 30, "note": "Basal ganglia infarct-related; striatal injury during decompensation → focal epilepsy"},
            {"type": "Infantile spasms (West syndrome)", "pct": 18, "note": "Seen in mut0 with severe metabolic insults in infancy; ACTH preferred if noted"},
            {"type": "Myoclonic seizures", "pct": 22, "note": "Often metabolic-triggered; improve with metabolic correction; LEV first-line"},
            {"type": "Absence seizures", "pct": 12, "note": "Reported in attenuated mut- phenotype; less severe; metabolic management reduces frequency"},
            {"type": "Status epilepticus", "pct": 20, "note": "Metabolic SE during crisis; EEG-confirmed; IV glucose + BZD + metabolic correction priority"},
        ],
        "metabolic_triggers": [
            {"trigger": "Fasting / prolonged NPO", "pct": 75, "mechanism": "Muscle BCAA catabolism → propionyl-CoA → methylmalonyl-CoA → MMA surge within hours"},
            {"trigger": "Intercurrent infection / fever", "pct": 65, "mechanism": "Catabolic state drives BCAA release; MMA + NH3 both surge; IV glucose emergency protocol mandatory"},
            {"trigger": "High-protein meal / formula error", "pct": 45, "mechanism": "Ile/Val/Met/Thr directly precursor propionyl-CoA → methylmalonyl-CoA cascade"},
            {"trigger": "Surgery / anesthesia", "pct": 40, "mechanism": "NPO + catabolic stress; propofol (lipid vehicle) may worsen; IV glucose protocol intraoperatively"},
            {"trigger": "Renal decompensation (CKD progression)", "pct": 35, "mechanism": "CKD → MMA retention (renal clearance impaired) → chronic MMA ↑ → metabolic burden; kidney Tx indicated"},
        ],
        "high_risk_drugs": [
            {"drug": "Valproate (VPA)", "risk": "ABSOLUTE CI", "mechanism": "Inhibits mitochondrial MMUT cofactor AdoCbl metabolism; depletes carnitine; worsens MMA accumulation; FATAL"},
            {"drug": "Fasting / NPO without IV glucose", "risk": "EXTREME HAZARD", "mechanism": "BCAA catabolism surge within hours → MMA crisis; always provide IV dextrose during illness or procedure"},
            {"drug": "Metformin", "risk": "AVOID", "mechanism": "Complex I inhibitor — synergistic toxicity with MMA-related respiratory chain inhibition → lactic acidosis"},
            {"drug": "High-protein diet / excess Ile/Val/Met/Thr", "risk": "EXTREME HAZARD", "mechanism": "Directly drives propionyl-CoA → methylmalonyl-CoA → MMA accumulation cascade"},
            {"drug": "Biotin (high-dose supplementation)", "risk": "NOT EFFECTIVE", "mechanism": "MMUT deficiency is NOT a biotin disorder; biotin supplementation is not indicated (vs HLCS/BTD)"},
        ],
        "treatments": [
            {"treatment": "Hydroxocobalamin (OHCbl) 1 mg/day IM", "evidence": "Level A", "response_pct": 20, "note": "Trial ALL patients 3–5 days. ONLY mut- with residual enzyme respond (~20%); mut0 does NOT respond. Lifelong if responsive"},
            {"treatment": "Protein restriction — low Ile/Val/Met/Thr", "evidence": "Level A", "response_pct": 85, "note": "Target plasma Ile <40 µmol/L; Val <150 µmol/L; natural protein 0.8–1.5 g/kg/day supplemented with MMUT-free formula; lifelong"},
            {"treatment": "MMUT-free amino acid formula", "evidence": "Level A", "response_pct": 85, "note": "Provides AA without propiogenic precursors; essential protein intake without Ile/Val/Met/Thr; concurrent natural protein reduction"},
            {"treatment": "L-Carnitine 100 mg/kg/day", "evidence": "Level A", "response_pct": 90, "note": "Secondary depletion correction; promotes MMA-carnitine conjugate (propionylcarnitine C3) renal excretion; monitor acylcarnitine ratio"},
            {"treatment": "IV Glucose GIR 8–12 + Insulin", "evidence": "Level A", "response_pct": 80, "note": "Acute crisis FIRST intervention; suppresses catabolism → MMA surge halted; STOP protein during acute phase"},
            {"treatment": "Ammonia Scavengers (Na benzoate / Na phenylacetate)", "evidence": "Level A", "response_pct": 70, "note": "If NH3 >200 µmol/L; IV Na benzoate 250 mg/kg loading then infusion; Na phenylacetate if combination product available"},
            {"treatment": "Metronidazole", "evidence": "Level B", "response_pct": 25, "note": "Reduces gut bacterial propionate production 20–30%; short courses (2–4 weeks); avoid chronic use (neurotoxicity risk)"},
            {"treatment": "Kidney transplant", "evidence": "Level B", "response_pct": 60, "note": "Corrects CKD and reduces plasma MMA (renal clearance restored); but hepatic MMUT absent → MMA does NOT normalize; continued dietary restriction required"},
            {"treatment": "Combined liver-kidney transplant", "evidence": "Level B", "response_pct": 80, "note": "Best long-term outcome for mut0; liver provides MMUT enzyme (near-normalizes MMA) + kidney corrects CKD; complex surgery; referral to metabolic transplant centre mandatory"},
            {"treatment": "HD / CRRT", "evidence": "Level B", "response_pct": 65, "note": "MMA >5,000 mmol/mol Cr or NH3 >500 µmol/L; MMA is dialyzable; bridge to metabolic stabilization"},
            {"treatment": "VPA (Valproate)", "evidence": "ABSOLUTE CI", "response_pct": 0, "note": "ABSOLUTE CONTRAINDICATION: inhibits mitochondrial MMUT cofactor; depletes carnitine; worsens MMA; FATAL — use LEV as first-line AED instead"},
            {"treatment": "Biotin supplementation", "evidence": "NOT EFFECTIVE", "response_pct": 0, "note": "MMUT deficiency is NOT a biotin disorder; biotin does not help (unlike HLCS/BTD where biotin is Level A)"},
        ],
        "patient_sample": [
            {**p, "methylcitrate_mild": round(random.uniform(5, 45), 1)} for p in sample
        ],
    }


def get_definitions() -> dict:
    return {
        "gene_card": {
            "Gene": "MMUT (previously MUT)",
            "Full Name": "Methylmalonyl-CoA Mutase",
            "Chromosome": "6p12.3",
            "Inheritance": "Autosomal Recessive (AR)",
            "OMIM Gene": "*609058",
            "OMIM Disease": "#251000 (Methylmalonic Acidemia, mut type)",
            "Protein Size": "750 amino acids; mitochondrial targeting sequence cleaved in mature form",
            "Structure": "Homodimer; each monomer contains one AdoCbl cofactor in N-terminal TIM-barrel domain",
            "Cofactor": "Adenosylcobalamin (AdoCbl) — active form of vitamin B12; generated from cob(I)alamin by MMAB (cblB)",
            "Substrate": "L-methylmalonyl-CoA → succinyl-CoA (radical-based isomerization)",
            "Prevalence": "~1:50,000–80,000 (all MMA subtypes); MMUT accounts for ~60% of all MMA",
            "NBS": "C3 (propionylcarnitine) elevated — same as PA; urine organic acids MANDATORY for differential",
            "Key Positive": "Urine MMA 200–10,000+ mmol/mol Cr PATHOGNOMONIC",
            "Key Negative": "Methylcitrate NOT pathognomonic; Homocysteine NORMAL (vs cblC); BCAA NORMAL; Biotin NOT effective",
        },
        "key_concepts": [
            {
                "concept": "Why MMA and PA both elevate C3 on NBS — but are DIFFERENT diseases",
                "explanation": (
                    "Both PA (PCCA/PCCB) and MMA (MMUT) accumulate propionyl-CoA derivatives that are "
                    "carnitine-conjugated (propionylcarnitine = C3). In PA, the block is AT PCC (propionyl-CoA → "
                    "D-methylmalonyl-CoA), so propionyl-CoA itself accumulates. In MMA, PCC is INTACT — "
                    "propionyl-CoA IS converted to D-methylmalonyl-CoA (then to L-methylmalonyl-CoA), but the "
                    "BLOCK is downstream at MMUT (L-methylmalonyl-CoA → succinyl-CoA). Both diseases back up "
                    "propionyl-CoA derivatives. KEY DIFFERENTIAL: urine MMA elevated in MMUT, absent in PA; "
                    "methylcitrate elevated (PATHOGNOMONIC) in PA, mildly elevated secondary in MMUT."
                ),
            },
            {
                "concept": "AdoCbl cofactor dependency and why mut0 patients do NOT respond to B12",
                "explanation": (
                    "MMUT requires adenosylcobalamin (AdoCbl) as a covalently-bound cofactor in its TIM-barrel "
                    "active site. In mut0 MMUT alleles (null mutations — frameshifts, nonsense, large deletions), "
                    "NO enzyme protein is made. Adding AdoCbl has nothing to bind to — the enzyme does not respond. "
                    "In mut- alleles (partial loss — missense mutations with residual apoenzyme), the misfolded/reduced "
                    "enzyme CAN sometimes be partially stabilized by its AdoCbl cofactor at higher concentrations "
                    "(pharmacological OHCbl dosing). This is why OHCbl trial (1 mg IM/day × 3–5 days) is performed "
                    "for ALL patients but only ~20% respond (exclusively mut-)."
                ),
            },
            {
                "concept": "Why CKD is unique to MMA but NOT seen in PA",
                "explanation": (
                    "Methylmalonic acid (MMA) is a direct renal proximal tubular toxin. MMA accumulates to "
                    "200–10,000+ mmol/mol Cr in urine and causes chronic tubular injury, impaired ammoniagenesis, "
                    "tubular acidosis, and progressive glomerulosclerosis. In PA (PCCA/PCCB), MMA is ABSENT — "
                    "the block is proximal to methylmalonyl-CoA formation, so the renal toxin never accumulates. "
                    "CKD in MMA is progressive (all phenotypes) and often requires kidney transplant by teen/adult "
                    "years. Kidney transplant corrects CKD and improves MMA clearance, but does NOT cure MMA "
                    "(hepatic MMUT still absent). Combined liver-kidney transplant is the only approach that "
                    "near-normalizes MMA (liver provides MMUT enzyme)."
                ),
            },
            {
                "concept": "VPA ABSOLUTE CI — mechanism specific to MMA",
                "explanation": (
                    "Valproate (VPA) is ABSOLUTELY CONTRAINDICATED in MMA. VPA and its metabolites: "
                    "(1) inhibit MMUT enzyme cofactor (AdoCbl) availability by perturbing cobalamin trafficking; "
                    "(2) massively deplete carnitine (same as PA) — exacerbating the secondary carnitine depletion "
                    "already present in MMA; (3) VPA directly inhibits mitochondrial function (same hepatotoxicity "
                    "as POLG1 patients) — additive with the MMA-induced respiratory chain damage. Combined, VPA "
                    "in MMA patients = MMA ↑↑↑ + carnitine depletion + hepatic failure: FATAL. LEV (levetiracetam) "
                    "is the first-line AED for seizures in MMA."
                ),
            },
            {
                "concept": "Why liver transplant alone is insufficient for MMA (unlike MSUD/PA)",
                "explanation": (
                    "In MSUD and PA, liver transplant provides 80–75% of the relevant enzyme activity (BCKDH/PCC "
                    "are highly expressed in liver), substantially reducing metabolic crises. In MMA, MMUT is "
                    "expressed in MANY extra-hepatic tissues (kidney, heart, muscle, brain). Liver transplant alone "
                    "provides ~70% of hepatic MMUT enzyme but leaves significant extra-hepatic enzyme deficiency. "
                    "Critically, CKD continues post-liver transplant because renal MMUT is still absent. Combined "
                    "liver-kidney transplant provides both the enzyme (liver) and corrects CKD (kidney) and is "
                    "the best long-term option for mut0 patients. Plasma MMA near-normalizes after combined Tx "
                    "(vs remains elevated after liver Tx alone due to renal retention)."
                ),
            },
            {
                "concept": "Differentiating MMUT from cblA (MMAA) and cblB (MMAB) isolated MMA",
                "explanation": (
                    "Isolated MMA (without homocystinuria) can be caused by: (1) MMUT apoenzyme defect — "
                    "mut0/mut-; (2) MMAA (cblA) — cobalamin adenosyltransferase defect; (3) MMAB (cblB) — "
                    "ATP:cob(I)alamin adenosyltransferase defect. All three elevate MMA without elevated "
                    "homocysteine (distinguishes from cblC/MMACHC). MMAA/MMAB patients have INTACT MMUT enzyme "
                    "protein but cannot synthesize AdoCbl — these patients are MORE likely to show B12 response "
                    "(OHCbl trial strongly positive). MMUT mut- may show partial response; MMUT mut0 does NOT. "
                    "Differentiation: enzyme assays (MMUT activity in fibroblasts), AdoCbl response, and "
                    "gene panel (MMUT, MMAA, MMAB) are required."
                ),
            },
            {
                "concept": "Secondary hyperammonemia in MMA — same CPS1/NAGS mechanism as PA",
                "explanation": (
                    "Both MMA (MMUT) and PA (PCCA/PCCB) cause secondary hyperammonemia through inhibition of "
                    "the urea cycle via the same mechanism: accumulated metabolites (methylmalonyl-CoA in MMA, "
                    "propionyl-CoA in PA) competitively inhibit N-acetylglutamate synthase (NAGS). Reduced NAG "
                    "→ CPS1 downregulation → urea cycle impaired → NH3 accumulates. Management is identical: "
                    "IV glucose to halt catabolism, ammonia scavengers if NH3 >200 µmol/L, HD/CRRT if NH3 >500. "
                    "This is NOT primary hyperammonemia (urea cycle enzyme deficiency) — it resolves when the "
                    "underlying organic acid accumulation is corrected."
                ),
            },
        ],
        "diagnostic_thresholds": [
            {"parameter": "Urine MMA", "threshold": ">5 mmol/mol Cr (abnormal); >200 = MMA diagnosis; >5,000 = severe/mut0", "action": "Confirm plasma MMA + homocysteine; start MMUT workup; gene panel"},
            {"parameter": "Plasma MMA", "threshold": ">0.4 µmol/L (abnormal); >200 = significant; >1,000 = crisis/transplant consideration", "action": "Correlate with eGFR (MMA rises as CKD progresses); hydroxocobalamin trial if not yet done"},
            {"parameter": "Plasma Ammonia", "threshold": ">100 µmol/L (action); >200 = start scavengers; >500 = HD/CRRT + ICU", "action": "IV glucose + STOP protein; Na benzoate/phenylacetate if >200; HD/CRRT if >500"},
            {"parameter": "eGFR", "threshold": "<90 = monitor 6-monthly; <60 = CKD stage 3+; <30 = kidney transplant consideration", "action": "Nephrology co-management; avoid nephrotoxic drugs; kidney ± liver transplant planning"},
            {"parameter": "Free Carnitine", "threshold": "<25 µmol/L = insufficient; <15 = significant depletion; <10 = crisis risk", "action": "L-carnitine 100 mg/kg/day PO; IV during crisis; monitor acylcarnitine ratio"},
            {"parameter": "OHCbl Trial Response", "threshold": "MMA fall >50% after 3–5 days OHCbl 1 mg IM = responsive (mut-)", "action": "Lifelong OHCbl if responsive; combined liver-kidney Tx planning if mut0 non-responder"},
        ],
        "differential_diagnosis": [
            {"disease": "PA — PCCA (Propionic Acidemia Type A)", "distinguishing": "Methylcitrate PATHOGNOMONIC (50–2,000); NO methylmalonate; C3 elevated same as MMA — urine OA mandatory"},
            {"disease": "PA — PCCB (Propionic Acidemia Type B)", "distinguishing": "Methylcitrate PATHOGNOMONIC; NO methylmalonate; PCCA vs PCCB biochemically identical — gene panel distinguishes both from MMUT"},
            {"disease": "cblC — MMACHC (Combined MMA + Homocystinuria)", "distinguishing": "HOMOCYSTEINE ELEVATED (>100 µmol/L) + MMA elevated — MMUT has NORMAL homocysteine; cblC also causes retinopathy and hemolytic uremic syndrome"},
            {"disease": "cblA — MMAA (isolated MMA, AdoCbl synthesis defect)", "distinguishing": "MMA elevated WITHOUT homocystinuria (same as MMUT); MMAA patients typically show STRONG B12 response (vs MMUT mut0 no response); gene panel (MMAA vs MMUT)"},
            {"disease": "cblB — MMAB (isolated MMA, AdoCbl synthesis defect)", "distinguishing": "MMA elevated WITHOUT homocystinuria; MMAB strong B12 response; enzyme assay shows MMUT activity NORMAL (vs deficient in MMUT); gene panel"},
            {"disease": "DLD Deficiency (E3 subunit)", "distinguishing": "BCAA ↑ + 2-hydroxyglutarate ↑ + glycine ↑ + lactate ↑ (4-complex block); MMUT has NORMAL BCAA; DLD MMA is ABSENT"},
            {"disease": "SUCLA2 / SUCLG1 (Succinyl-CoA ligase defect)", "distinguishing": "MMA mildly elevated; mtDNA depletion syndrome features; deafness; cardiomyopathy; gene panel"},
        ],
    }
