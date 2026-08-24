#!/usr/bin/env python3
"""MMAA (Methylmalonic Acidemia, cblA type / Cobalamin-A) Epilepsy Dashboard.

MMAA encodes the Methylmalonyl-CoA Mutase-Associated GTPase (Cobalamin A Chaperone),
a mitochondrial matrix GTPase that delivers adenosylcobalamin (AdoCbl) from MMAB to MMUT.

  MMAA PROTEIN BIOLOGY:
    MMAA protein: 418 aa, mitochondrial matrix, G3E P-loop GTPase (G-protein chaperone).
    Function: GTP-dependent chaperone that catalyzes the delivery of AdoCbl from MMAB
    (ATP:cob(I)alamin adenosyltransferase) to the MMUT apoenzyme active site (TIM-barrel).
    Mechanism: MMAA binds MMAB (AdoCbl-loaded) → GTP hydrolysis by MMAA → conformational
    change releases AdoCbl → AdoCbl inserted into MMUT TIM-barrel → MMUT holoenzyme activated.
    Without MMAA: AdoCbl cannot be transferred → MMUT apoenzyme remains cofactor-empty.

  MMAA LOF → AdoCbl NOT inserted into MMUT TIM-barrel:
    MMUT apoenzyme is STRUCTURALLY INTACT (this is critical — MMUT protein itself is normal)
    → MMUT cannot catalyze L-methylmalonyl-CoA → succinyl-CoA (cofactor-empty)
    → L-methylmalonyl-CoA ACCUMULATES → METHYLMALONIC ACID (MMA) ELEVATED
    → Pharmacological OHCbl can partially overcome the chaperone defect (strongly B12-responsive)

  KEY DISTINCTION FROM MMUT:
    In MMUT deficiency: the MMUT APOENZYME is defective (mut0 = no enzyme; mut- = misfolded enzyme).
    In MMAA deficiency: the MMUT APOENZYME IS INTACT — the DELIVERY MECHANISM fails.
    Consequence: MMAA patients respond STRONGLY to hydroxocobalamin (OHCbl) in ~60–80% of cases
    (vs ~20% in MMUT, exclusively mut- partial alleles).
    In vitro: MMAA fibroblasts CORRECT when AdoCbl is provided (MMUT apoenzyme is functional).
    MMUT mut0 fibroblasts do NOT correct (apoenzyme absent/defunct).

  PATHWAY UPSTREAM OF MMAA BLOCK:
    Propionyl-CoA (from Ile, Val, Met, Thr; odd-chain FA; cholesterol)
      → Propionyl-CoA + HCO3- → D-methylmalonyl-CoA [PCC: PCCA+PCCB]
      → D-methylmalonyl-CoA → L-methylmalonyl-CoA [MCEE: epimerase]
    COFACTOR DELIVERY PATHWAY (MMAA step):
      cob(II)alamin → MMAB → AdoCbl-MMAB complex → MMAA (GTP) → MMUT apoenzyme ← BLOCK IN MMAA
      [MMAB: ATP:cob(I)alamin adenosyltransferase, cblB gene product]
      [MMAA: GTP-dependent chaperone; transfers AdoCbl from MMAB to MMUT TIM-barrel]
    Succinyl-CoA → TCA (→ fumarate → malate → OAA)

  BIOMARKERS:
    C3 (propionylcarnitine): ↑ — NBS PRIMARY (same as PA/MMUT)
    Methylmalonic acid (urine): ↑↑ 200–5,000 mmol/mol Cr (PATHOGNOMONIC; lower than MMUT mut0)
    Methylmalonic acid (plasma): ↑ 50–2,000 µmol/L
    Methylcitrate: mildly elevated secondary <50 µmol/mmol Cr (NOT pathognomonic; vs PA PATHOGNOMONIC)
    Ammonia: ↑ 50–800 µmol/L (secondary NAGS/CPS1 inhibition)
    Free carnitine: ↓ 8–35 µmol/L (secondary depletion)
    Homocysteine: NORMAL <15 µmol/L (KEY NEGATIVE vs cblC/MMACHC)
    BCAA: NORMAL (KEY NEGATIVE vs DLD/MSUD)
    B12 response: >60% show MMA reduction >50% on OHCbl trial (STRONGEST among isolated MMA)
    MMUT enzyme activity in fibroblasts: REDUCED but correctable by AdoCbl in vitro

  KEY VARIANTS IN MMAA:
    - p.Arg145His (c.434G>A): GTPase G3 motif; most common MMAA allele; partial GTPase; strong OHCbl response
    - p.Arg186Trp (c.556C>T): GTPase-MMAB interface; disrupts MMAA-MMAB interaction; severe
    - p.Gln210* (c.628C>T): Truncating null; severe; rarely OHCbl responsive
    - c.IVS6+1G>A: Splice null; absent MMAA mRNA; severe neonatal
    - p.Ile96Thr (c.287T>C): N-terminal mitochondrial targeting impaired; intermediate response
    - p.Gly199Asp: GTPase catalytic core; GTP hydrolysis impaired; moderate-severe
    - p.Pro91Leu: P-loop adjacent; partial GTPase activity; partial OHCbl response
    - Large deletion exon 1-3 (MLPA): null; ~5% of MMAA alleles

  TREATMENT HIGHLIGHTS:
    Hydroxocobalamin OHCbl 1–2 mg/day IM (Level A — trial ALL patients; ~60–80% respond)
    Cyanocobalamin oral (Level B — maintenance for confirmed responders)
    Protein restriction — low Ile/Val/Met/Thr (Level A; lifelong)
    MMAA-free amino acid formula (Level A)
    L-Carnitine 100 mg/kg/day (Level A; secondary depletion)
    IV glucose GIR 8–12 + insulin (Level A; acute crisis)
    Ammonia scavengers (Level A if NH3 >200)
    Metronidazole (Level B; reduces gut propionate)
    Liver transplant (Level B; for OHCbl non-responders — less commonly needed than MMUT)
    HD/CRRT (Level B; MMA >3,000 or NH3 >500)
    VPA — ABSOLUTE CI (inhibits cobalamin metabolism/MMAA-MMAB pathway + carnitine depletion; FATAL)
    Fasting — EXTREME HAZARD (BCAA catabolism surge)
    High protein — EXTREME HAZARD (Ile/Val/Met/Thr precursors)
    Missed OHCbl doses — HIGH RISK (MMA rebounds within days)

  OMIM: Gene *607481 · Disease #251100 (Methylmalonic Acidemia, cblA type)
  Chromosome: 4p14 · Inheritance: AR · Prevalence: ~1:100,000–200,000
  (~25–30% of cobalamin-responsive isolated MMA)
"""

from __future__ import annotations
import random

random.seed(42)   # reproducible synthetic cohort


# ── helpers ──────────────────────────────────────────────────────────────────

def _pid(i: int) -> str:
    return f"MMAA-{i:03d}"


def _sex(i: int) -> str:
    return "M" if i % 2 == 0 else "F"


def _phenotype(i: int) -> str:
    """
    cblA Neonatal Severe ~25% (10 patients),
    cblA Classic Infantile ~50% (20 patients — MODAL),
    cblA Attenuated Late Onset ~20% (8 patients),
    cblA Mild Residual ~5% (2 patients).
    """
    # 40 patients: indices 1-40
    if i <= 10:   return "cblA Neonatal Severe"
    if i <= 30:   return "cblA Classic Infantile"
    if i <= 38:   return "cblA Attenuated Late Onset"
    return "cblA Mild Residual"


def _genotype(i: int) -> str:
    variants = [
        "c.434G>A/c.434G>A",         # p.Arg145His homozygous; most common; OHCbl responsive
        "c.434G>A/c.628C>T",         # p.Arg145His / p.Gln210* compound het; variable response
        "c.628C>T/c.628C>T",         # p.Gln210* homozygous null; non-responsive
        "c.434G>A/c.556C>T",         # p.Arg145His / p.Arg186Trp; moderate-severe
        "c.IVS6+1G>A/c.628C>T",      # splice null / null; severe
        "c.287T>C/c.434G>A",         # p.Ile96Thr / p.Arg145His; intermediate
        "c.434G>A/del.exon1-3",      # p.Arg145His / large deletion; compound het
        "c.556C>T/c.556C>T",         # p.Arg186Trp homozygous; severe
        "c.287T>C/c.287T>C",         # p.Ile96Thr homozygous; moderate
        "c.628C>T/c.IVS6+1G>A",      # null/splice; severe non-responsive
    ]
    return variants[i % len(variants)]


def _mma_urine(pheno: str) -> float:
    """Methylmalonic acid urine mmol/mol Cr."""
    if pheno == "cblA Neonatal Severe":
        return round(random.uniform(3000, 5000), 0)
    if pheno == "cblA Classic Infantile":
        return round(random.uniform(500, 2500), 0)
    if pheno == "cblA Attenuated Late Onset":
        return round(random.uniform(200, 800), 0)
    return round(random.uniform(100, 400), 0)


def _plasma_mma(pheno: str) -> float:
    """Plasma MMA µmol/L."""
    if pheno == "cblA Neonatal Severe":
        return round(random.uniform(800, 2000), 0)
    if pheno == "cblA Classic Infantile":
        return round(random.uniform(100, 800), 0)
    if pheno == "cblA Attenuated Late Onset":
        return round(random.uniform(50, 300), 0)
    return round(random.uniform(20, 100), 0)


def _c3(pheno: str) -> float:
    if pheno == "cblA Neonatal Severe":
        return round(random.uniform(12, 35), 1)
    if pheno == "cblA Classic Infantile":
        return round(random.uniform(4, 20), 1)
    if pheno == "cblA Attenuated Late Onset":
        return round(random.uniform(3, 12), 1)
    return round(random.uniform(2, 6), 1)


def _ammonia(pheno: str) -> int:
    if pheno == "cblA Neonatal Severe":
        return random.randint(300, 800)
    if pheno == "cblA Classic Infantile":
        return random.randint(100, 400)
    if pheno == "cblA Attenuated Late Onset":
        return random.randint(50, 150)
    return random.randint(20, 80)


def _carnitine(pheno: str) -> float:
    if pheno == "cblA Neonatal Severe":
        return round(random.uniform(8, 18), 1)
    if pheno == "cblA Classic Infantile":
        return round(random.uniform(12, 28), 1)
    if pheno == "cblA Attenuated Late Onset":
        return round(random.uniform(18, 35), 1)
    return round(random.uniform(25, 40), 1)


def _gfr(pheno: str) -> int:
    """eGFR — CKD less severe in MMAA than MMUT (MMA lower with OHCbl treatment)."""
    if pheno == "cblA Neonatal Severe":
        return random.randint(30, 70)
    if pheno == "cblA Classic Infantile":
        return random.randint(50, 90)
    if pheno == "cblA Attenuated Late Onset":
        return random.randint(65, 100)
    return random.randint(75, 115)


def _onset_months(pheno: str) -> int:
    if pheno == "cblA Neonatal Severe":
        return random.randint(0, 1)
    if pheno == "cblA Classic Infantile":
        return random.randint(1, 12)
    if pheno == "cblA Attenuated Late Onset":
        return random.randint(12, 84)
    return random.randint(24, 180)


def _ohcbl_response(pheno: str) -> bool:
    """OHCbl response rate: Neonatal Severe low, Classic ~70%, Attenuated ~90%, Mild 100%."""
    if pheno == "cblA Neonatal Severe":
        return random.random() < 0.20   # null alleles; mostly non-responsive
    if pheno == "cblA Classic Infantile":
        return random.random() < 0.70   # ~70% of classic cblA respond
    if pheno == "cblA Attenuated Late Onset":
        return random.random() < 0.90   # strong responders
    return True  # cblA Mild Residual: 100% responsive


def _seizures(pheno: str) -> bool:
    if pheno == "cblA Neonatal Severe":
        return random.random() < 0.70
    if pheno == "cblA Classic Infantile":
        return random.random() < 0.55
    if pheno == "cblA Attenuated Late Onset":
        return random.random() < 0.35
    return random.random() < 0.20


def _cardiomyopathy(pheno: str) -> bool:
    if pheno == "cblA Neonatal Severe":
        return random.random() < 0.20
    if pheno == "cblA Classic Infantile":
        return random.random() < 0.15
    if pheno == "cblA Attenuated Late Onset":
        return random.random() < 0.05
    return False


# ── patient table ─────────────────────────────────────────────────────────────

def _patients():
    pts = []
    for i in range(1, 41):
        pheno = _phenotype(i)
        pts.append({
            "patient_id": _pid(i),
            "sex": _sex(i),
            "phenotype": pheno,
            "age_at_diagnosis_months": _onset_months(pheno),
            "mma_urine_mmol_molCr": int(_mma_urine(pheno)),
            "plasma_mma_umol_l": int(_plasma_mma(pheno)),
            "c3_umol_l": _c3(pheno),
            "ammonia_umol_l": _ammonia(pheno),
            "free_carnitine_umol_l": _carnitine(pheno),
            "egfr_ml_min_1_73m2": _gfr(pheno),
            "seizures": _seizures(pheno),
            "cardiomyopathy": _cardiomyopathy(pheno),
            "nbs_detected": random.random() > 0.12,
            "ohcbl_response": _ohcbl_response(pheno),
            "aed": random.choice(["None", "LEV", "VGB", "LEV+CLB", "VGB+LEV"]) if _seizures(pheno) else "None",
            "genotype": _genotype(i),
        })
    return pts


_PATIENTS = _patients()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def get_overview() -> dict:
    n = len(_PATIENTS)
    mma_vals  = [p["mma_urine_mmol_molCr"] for p in _PATIENTS]
    c3_vals   = [p["c3_umol_l"] for p in _PATIENTS]
    nh3_vals  = [p["ammonia_umol_l"] for p in _PATIENTS]
    car_vals  = [p["free_carnitine_umol_l"] for p in _PATIENTS]
    egfr_vals = [p["egfr_ml_min_1_73m2"] for p in _PATIENTS]

    def pct(pred): return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    return {
        "gene": "MMAA",
        "full_name": "Methylmalonyl-CoA Mutase-Associated GTPase (Cobalamin A Chaperone)",
        "chromosome": "4p14",
        "inheritance": "AR",
        "omim_gene": "*607481",
        "omim_disease": "#251100",
        "protein_size": "418 aa; mitochondrial matrix; G3E P-loop GTPase chaperone",
        "prevalence": "~1:100,000–200,000 (~25–30% of cobalamin-responsive isolated MMA)",
        "nbs_primary": "C3 (propionylcarnitine) elevated — identical to PA and MMUT; triggers MMA workup",
        "nbs_secondary": "Urine MMA (PATHOGNOMONIC 200–5,000); plasma MMA; C3/C2 ratio; OHCbl trial",
        "function": (
            "MMAA is a GTPase chaperone that catalyzes GTP-dependent delivery of adenosylcobalamin (AdoCbl) "
            "from MMAB (ATP:cob(I)alamin adenosyltransferase) to the MMUT apoenzyme active site (TIM-barrel). "
            "MMAA binds AdoCbl-loaded MMAB, undergoes GTP hydrolysis, and releases AdoCbl into MMUT."
        ),
        "mechanism": (
            "MMAA LOF → AdoCbl cannot be transferred to MMUT TIM-barrel → MMUT apoenzyme remains "
            "cofactor-empty (structurally INTACT but inactive) → L-methylmalonyl-CoA accumulates → "
            "MMA elevated (200–5,000 mmol/mol Cr). Because MMUT apoenzyme is INTACT, pharmacological "
            "OHCbl partially overcomes the delivery defect: >60% of MMAA patients respond (vs ~20% MMUT). "
            "Secondary: MMA inhibits CPS1 via NAGS → hyperammonemia; MMA less CKD than MMUT (lower MMA with B12 Rx)."
        ),
        "key_negative": (
            "HOMOCYSTEINE NORMAL (<15 µmol/L) — KEY NEGATIVE vs cblC (MMACHC, combined MMA+HHcy); "
            "BCAA NORMAL — KEY NEGATIVE vs DLD/MSUD; METHYLCITRATE mildly elevated secondary only "
            "(NOT pathognomonic — vs PA where methylcitrate is PATHOGNOMONIC); "
            "MMUT ENZYME INTACT in fibroblasts when AdoCbl provided — KEY DISTINCTION from MMUT apoenzyme defects."
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
            {"phenotype": "cblA Neonatal Severe (~25%)", "pct": 25},
            {"phenotype": "cblA Classic Infantile (~50%) — MODAL", "pct": 50},
            {"phenotype": "cblA Attenuated Late Onset (~20%)", "pct": 20},
            {"phenotype": "cblA Mild Residual (~5%)", "pct": 5},
        ],
        "mmaa_pathway": [
            {
                "step": "Step 1 — cob(II)alamin uptake → MMACHC (cblC) → cob(I)alamin",
                "reaction": "Cob(II)alamin + reductant → cob(I)alamin [MMACHC decyanylase/reductase]",
                "enzyme": "MMACHC (cblC chaperone)",
                "cofactor": "None",
                "consequence_mmaa_lof": "This step is INTACT in MMAA deficiency — cobalamin enters mitochondria normally",
            },
            {
                "step": "Step 2 — MMAB: cob(I)alamin → AdoCbl [ATP:cob(I)alamin adenosyltransferase]",
                "reaction": "cob(I)alamin + ATP → AdoCbl + PPPi [MMAB/cblB]",
                "enzyme": "MMAB (cblB; ATP:cob(I)alamin adenosyltransferase)",
                "cofactor": "ATP",
                "consequence_mmaa_lof": "MMAB can SYNTHESIZE AdoCbl — but MMAA cannot transfer it onward to MMUT",
            },
            {
                "step": "Step 3 — MMAA: GTP-dependent AdoCbl delivery from MMAB to MMUT ← BLOCK",
                "reaction": "MMAA + GTP + AdoCbl-MMAB → MMAA:GTP:MMAB + AdoCbl → MMUT-AdoCbl holoenzyme",
                "enzyme": "MMAA (cblA; G3E P-loop GTPase chaperone, 418 aa, 4p14)",
                "cofactor": "GTP (for MMAA conformational switch); AdoCbl (cargo)",
                "consequence_mmaa_lof": "BLOCKED: AdoCbl-loaded MMAB cannot transfer cofactor → MMUT apoenzyme cofactor-empty → INACTIVE",
            },
            {
                "step": "Step 4 — MMUT holoenzyme: L-methylmalonyl-CoA → succinyl-CoA → TCA",
                "reaction": "L-methylmalonyl-CoA → succinyl-CoA (AdoCbl radical mechanism) [MMUT]",
                "enzyme": "MMUT (methylmalonyl-CoA mutase; 750 aa; TIM-barrel; INTACT apoenzyme in MMAA deficiency)",
                "cofactor": "AdoCbl (delivered by MMAA — absent in MMAA LOF)",
                "consequence_mmaa_lof": "BLOCKED (secondary): MMUT apoenzyme is STRUCTURALLY NORMAL but cannot act → L-methylmalonyl-CoA accumulates → MMA ↑",
            },
        ],
        "mmaa_vs_mmut": {
            "title": "MMAA (cblA) vs MMUT — 10-Feature Critical Differential",
            "note": (
                "Both MMAA and MMUT cause ISOLATED MMA (without elevated homocysteine). "
                "C3 (propionylcarnitine) elevated in BOTH — NBS CANNOT distinguish. "
                "OHCbl trial response, fibroblast enzyme assay, and gene panel are mandatory. "
                "Gene panel: MMUT, MMAA, MMAB, MMACHC, MMADHC — all required for isolated MMA workup."
            ),
            "comparison": [
                {"feature": "OHCbl Response", "MMAA_cblA": "STRONG >60% (HALLMARK cblA)", "MMUT": "~20% (mut- partial only; mut0 NONE)"},
                {"feature": "MMUT Enzyme Activity (fibroblasts)", "MMAA_cblA": "NORMAL when AdoCbl provided (apoenzyme intact)", "MMUT": "DEFICIENT — mut0: absent; mut-: reduced"},
                {"feature": "In vitro AdoCbl correction", "MMAA_cblA": "YES — fibroblasts CORRECT with AdoCbl addition", "MMUT": "mut0: does NOT correct; mut-: partial"},
                {"feature": "Molecular defect", "MMAA_cblA": "AdoCbl DELIVERY chaperone defect (MMUT apoenzyme intact)", "MMUT": "MMUT APOENZYME defect (cofactor binding or catalysis)"},
                {"feature": "Severity", "MMAA_cblA": "Generally milder; infantile modal; OHCbl often normalizes", "MMUT": "mut0 most severe; neonatal; no B12 response"},
                {"feature": "CKD", "MMAA_cblA": "Less severe (~18%); lower MMA with OHCbl treatment", "MMUT": "Progressive CKD ~35% (unique to MMA, not PA)"},
                {"feature": "Cardiomyopathy", "MMAA_cblA": "~15–20% (lower than MMUT mut0)", "MMUT": "~25–35% (mut0; respiratory chain inhibition)"},
                {"feature": "Transplant Need", "MMAA_cblA": "Liver Tx rarely needed; OHCbl usually sufficient", "MMUT": "mut0: combined liver-kidney Tx often needed"},
                {"feature": "Neonatal presentation", "MMAA_cblA": "~25% neonatal severe; 50% infantile onset (modal)", "MMUT": "mut0 ~50% neonatal severe Day 1–5"},
                {"feature": "Gene (chromosome)", "MMAA_cblA": "MMAA — 4p14 (GTPase chaperone)", "MMUT": "MMUT — 6p12.3 (methylmalonyl-CoA mutase apoenzyme)"},
            ],
        },
        "high_risk_situations": [
            {"situation": "VPA (Valproate)", "risk": "ABSOLUTE CI", "detail": "VPA inhibits cobalamin metabolism in the MMAA-MMAB pathway (adenosylcobalamin synthesis/delivery); massively depletes carnitine; worsens MMA accumulation: FATAL"},
            {"situation": "Fasting", "risk": "EXTREME HAZARD", "detail": "BCAA catabolism surge → propionyl-CoA → L-methylmalonyl-CoA → MMA ↑↑ within hours; IV glucose mandatory"},
            {"situation": "High-protein diet", "risk": "EXTREME HAZARD", "detail": "Ile/Val/Met/Thr directly drives propionyl-CoA → L-methylmalonyl-CoA → MMA accumulation"},
            {"situation": "Intercurrent illness", "risk": "EXTREME HAZARD", "detail": "Catabolism → MMA surge + hyperammonemia; IV glucose protocol mandatory; do NOT withhold carnitine or OHCbl"},
            {"situation": "Missed OHCbl doses", "risk": "HIGH RISK", "detail": "MMA rebounds within 3–7 days of missed IM OHCbl; compliance monitoring and family education essential"},
            {"situation": "Metformin", "risk": "AVOID", "detail": "Complex I inhibitor — synergistic toxicity with MMA-related mitochondrial dysfunction → lactic acidosis risk"},
        ],
    }


def get_breakdown() -> dict:
    sample = _PATIENTS[:15]
    return {
        "biomarkers": [
            {"name": "C3 (Propionylcarnitine)", "normal": "<4 µmol/L", "mmaa_range": "3–35 µmol/L", "significance": "NBS PRIMARY trigger — IDENTICAL to PA and MMUT; cannot distinguish MMAA from PCCA/PCCB/MMUT by NBS alone", "method": "NBS tandem MS/MS; plasma acylcarnitines"},
            {"name": "Methylmalonic Acid (urine)", "normal": "<5 mmol/mol Cr", "mmaa_range": "200–5,000 mmol/mol Cr", "significance": "PATHOGNOMONIC — lower range than MMUT mut0 (which reaches 10,000+) due to partial OHCbl responsiveness; absent in PA", "method": "Urine organic acids (GC-MS)"},
            {"name": "Methylmalonic Acid (plasma)", "normal": "<0.4 µmol/L", "mmaa_range": "50–2,000 µmol/L", "significance": "Diagnostic confirmation; monitor on OHCbl trial (fall >50% = responsive); lower than MMUT mut0 at baseline", "method": "Plasma MMA (LC-MS/MS or GC-MS)"},
            {"name": "Methylcitrate (urine)", "normal": "<5 µmol/mmol Cr", "mmaa_range": "<50 µmol/mmol Cr (mild secondary)", "significance": "MILDLY elevated secondary only — NOT pathognomonic; pathognomonic in PA (50–2,000 µmol/mmol Cr) — KEY DIFFERENTIAL from PA", "method": "Urine organic acids (GC-MS)"},
            {"name": "Ammonia (plasma)", "normal": "<50 µmol/L", "mmaa_range": "50–800 µmol/L", "significance": "Secondary hyperammonemia via NAGS/CPS1 inhibition (same mechanism as PA/MMUT); NH3 >300 = crisis", "method": "Plasma ammonia (enzymatic)"},
            {"name": "Free Carnitine", "normal": "25–60 µmol/L", "mmaa_range": "8–35 µmol/L", "significance": "DEPLETED secondary — MMA-carnitine (propionylcarnitine C3) excreted; L-carnitine replacement mandatory; lower depletion if OHCbl responsive", "method": "Plasma acylcarnitines"},
            {"name": "Homocysteine (total)", "normal": "<15 µmol/L", "mmaa_range": "NORMAL <15 µmol/L", "significance": "NORMAL — KEY NEGATIVE vs cblC (MMACHC) where BOTH MMA and homocysteine are elevated; isolated MMA without HHcy = MMUT/MMAA/MMAB", "method": "Plasma tHcy (HPLC or immunoassay)"},
            {"name": "MMUT Enzyme Activity (fibroblasts)", "normal": ">200 nmol/h/mg", "mmaa_range": "REDUCED without AdoCbl; NORMAL when AdoCbl provided in vitro", "significance": "KEY DISTINCTION: MMAA fibroblasts CORRECT with exogenous AdoCbl (MMUT apoenzyme intact); MMUT mut0 do NOT correct", "method": "Fibroblast enzyme assay ± AdoCbl supplementation (metabolic genetics lab)"},
            {"name": "eGFR / Creatinine", "normal": ">90 mL/min/1.73m²", "mmaa_range": "30–100 mL/min/1.73m²", "significance": "CKD present but LESS SEVERE than MMUT (lower MMA with OHCbl treatment reduces renal tubular MMA toxicity)", "method": "Serum creatinine (CKD-EPI); cystatin C; 6-monthly monitoring"},
            {"name": "OHCbl Trial Response", "normal": "N/A", "mmaa_range": ">60% show MMA reduction >50% at 3–5 days", "significance": "STRONGEST OHCbl response rate among isolated MMA subtypes; defines cblA pharmacological responsiveness; measure at Day 0 and Day 5 of OHCbl 1–2 mg IM daily", "method": "Plasma MMA before/after 3–5 days OHCbl 1–2 mg/day IM"},
            {"name": "C3/C2 Ratio", "normal": "<0.04", "mmaa_range": ">0.08", "significance": "Secondary NBS flag; not disease-specific (same in PA and MMA); triggers organic acid workup", "method": "NBS tandem MS/MS"},
        ],
        "key_variants": [
            {"variant": "p.Arg145His", "cdna": "c.434G>A", "domain": "GTPase G3 motif", "severity": "Moderate (cblA responsive)", "note": "Most common MMAA allele globally; partial GTPase activity retained; strong OHCbl response; European/worldwide founder-like"},
            {"variant": "p.Arg186Trp", "cdna": "c.556C>T", "domain": "GTPase-MMAB interface", "severity": "Moderate-Severe", "note": "Disrupts MMAA-MMAB interaction; AdoCbl transfer severely impaired; weaker OHCbl response"},
            {"variant": "p.Gln210*", "cdna": "c.628C>T", "domain": "Truncating null", "severity": "Severe (non-responsive)", "note": "Nullizygous; no GTPase protein made; rarely responsive to OHCbl; neonatal severe presentation"},
            {"variant": "c.IVS6+1G>A", "cdna": "Intron 6 splice site", "domain": "Splice disruption — null", "severity": "Severe (null)", "note": "Absent MMAA mRNA; null allele; severe neonatal presentation; no OHCbl response"},
            {"variant": "p.Ile96Thr", "cdna": "c.287T>C", "domain": "N-terminal mitochondrial targeting", "severity": "Moderate", "note": "Impaired mitochondrial import; residual cytoplasmic MMAA protein; intermediate OHCbl response"},
            {"variant": "p.Gly199Asp", "cdna": "GTPase catalytic core", "domain": "GTPase catalytic core", "severity": "Moderate-Severe", "note": "GTP hydrolysis impaired; conformational switch for AdoCbl delivery defective; partial OHCbl response"},
            {"variant": "p.Pro91Leu", "cdna": "P-loop adjacent", "domain": "P-loop adjacent residue", "severity": "Moderate", "note": "Minor P-loop disruption; partial GTPase activity; OHCbl partially responsive"},
            {"variant": "Large del exon 1-3", "cdna": "MLPA confirmed", "domain": "N-terminal (entire chaperone)", "severity": "Severe (null)", "note": "~5% of MMAA alleles; genomic deletion; no GTPase protein; MLPA/CNV array needed for detection"},
        ],
        "seizure_types": [
            {"type": "Metabolic encephalopathy seizures", "pct": 55, "note": "Multifocal/generalized during hyperammonemic crisis; EEG: burst-suppression in neonates; first priority: correct MMA and NH3"},
            {"type": "Focal cortical seizures", "pct": 22, "note": "Rare basal ganglia infarct in non-responders; striatal injury during decompensation → focal epilepsy; less common than MMUT"},
            {"type": "Infantile spasms (West syndrome)", "pct": 15, "note": "Less common than MMUT; OHCbl-responsive cases may resolve spasms with metabolic correction alone; LEV preferred"},
            {"type": "Myoclonic seizures", "pct": 20, "note": "Metabolic-triggered; improve with OHCbl + metabolic management; LEV first-line AED"},
            {"type": "Absence seizures", "pct": 15, "note": "More frequent in attenuated late-onset phenotype; metabolic control reduces frequency"},
            {"type": "Status epilepticus", "pct": 12, "note": "Metabolic SE during hyperammonemic crisis; EEG-confirmed; IV glucose + BZD + metabolic correction; avoid VPA"},
        ],
        "metabolic_triggers": [
            {"trigger": "Fasting / prolonged NPO", "pct": 70, "mechanism": "BCAA catabolism → propionyl-CoA → L-methylmalonyl-CoA → MMA surge within hours; provide IV glucose and maintain OHCbl dosing"},
            {"trigger": "Intercurrent infection / fever", "pct": 60, "mechanism": "Catabolic state drives BCAA release; MMA and NH3 surge together; IV glucose protocol mandatory; do NOT delay OHCbl"},
            {"trigger": "Missed OHCbl doses", "pct": 45, "mechanism": "MMA rebounds within 3–7 days; compliance critical; plasma MMA rises back to pre-treatment levels without IM OHCbl"},
            {"trigger": "High-protein meal / formula error", "pct": 40, "mechanism": "Ile/Val/Met/Thr directly drive propionyl-CoA accumulation → L-methylmalonyl-CoA → MMA"},
            {"trigger": "Surgery / anesthesia", "pct": 30, "mechanism": "NPO + catabolic stress; IV glucose protocol perioperatively; continue IM OHCbl; metabolic team co-management mandatory"},
        ],
        "high_risk_drugs": [
            {"drug": "Valproate (VPA)", "risk": "ABSOLUTE CI", "mechanism": "Inhibits cobalamin metabolism in MMAA-MMAB adenosylcobalamin synthesis/delivery pathway; depletes carnitine; worsens MMA accumulation; FATAL in MMAA patients — use LEV as first-line AED"},
            {"drug": "Fasting / NPO without IV glucose", "risk": "EXTREME HAZARD", "mechanism": "BCAA catabolism surge → MMA crisis within hours; always provide IV dextrose during illness or fasting procedures"},
            {"drug": "High-protein diet / excess Ile/Val/Met/Thr", "risk": "EXTREME HAZARD", "mechanism": "Directly drives propionyl-CoA → L-methylmalonyl-CoA → MMA accumulation; avoid high-protein supplementation"},
            {"drug": "Metformin", "risk": "AVOID", "mechanism": "Complex I inhibitor — synergistic mitochondrial toxicity with MMA-related respiratory chain damage → lactic acidosis risk"},
            {"drug": "Missed IM OHCbl doses", "risk": "HIGH RISK", "mechanism": "MMA rebounds within 3–7 days of missed OHCbl; compliance monitoring and parent education mandatory; oral cyanocobalamin is less reliable backup"},
        ],
        "treatments": [
            {"treatment": "Hydroxocobalamin (OHCbl) 1–2 mg/day IM", "evidence": "Level A", "response_pct": 68, "note": "Trial ALL MMAA patients; >60–80% respond (MMA reduction >50% at Day 3–5); lifelong IM preferred over oral; STRONGER than MMUT (20%) — HALLMARK of cblA"},
            {"treatment": "Cyanocobalamin oral (maintenance)", "evidence": "Level B", "response_pct": 45, "note": "Alternative for confirmed OHCbl responders during stable maintenance; less reliable than IM; monitor plasma MMA every 3 months"},
            {"treatment": "Protein restriction (low Ile/Val/Met/Thr)", "evidence": "Level A", "response_pct": 85, "note": "Target plasma Ile <40 µmol/L; Val <150 µmol/L; natural protein 1.0–1.8 g/kg/day supplemented with MMAA-free formula; lifelong"},
            {"treatment": "MMAA-free amino acid formula", "evidence": "Level A", "response_pct": 85, "note": "Provides protein intake without Ile/Val/Met/Thr propiogenic precursors; essential alongside natural protein restriction; lifelong"},
            {"treatment": "L-Carnitine 100 mg/kg/day", "evidence": "Level A", "response_pct": 88, "note": "Secondary depletion; promotes MMA-carnitine conjugate (propionylcarnitine C3) renal excretion; monitor carnitine levels — may need less supplementation if OHCbl normalizes MMA"},
            {"treatment": "IV Glucose GIR 8–12 + Insulin", "evidence": "Level A", "response_pct": 80, "note": "Acute crisis FIRST intervention; suppresses catabolism → MMA surge halted; STOP protein during acute phase; continue OHCbl IM"},
            {"treatment": "Ammonia Scavengers (Na benzoate / Na phenylacetate)", "evidence": "Level A", "response_pct": 70, "note": "If NH3 >200 µmol/L; IV Na benzoate 250 mg/kg loading; same protocol as MMUT/PA secondary hyperammonemia"},
            {"treatment": "Metronidazole", "evidence": "Level B", "response_pct": 25, "note": "Reduces gut bacterial propionate 20–30%; short courses (2–4 weeks); avoid chronic use (neurotoxicity)"},
            {"treatment": "Liver transplant", "evidence": "Level B", "response_pct": 65, "note": "Less commonly needed than MMUT mut0 (OHCbl usually effective); reserved for confirmed OHCbl non-responders with null alleles; reduces but does not fully normalize MMA if extra-hepatic MMAA absent"},
            {"treatment": "HD / CRRT", "evidence": "Level B", "response_pct": 65, "note": "MMA >3,000 mmol/mol Cr or NH3 >500 µmol/L; MMA is dialyzable; bridge to metabolic stabilization; less frequently needed than MMUT mut0"},
            {"treatment": "VPA (Valproate)", "evidence": "ABSOLUTE CI", "response_pct": 0, "note": "ABSOLUTE CONTRAINDICATION: inhibits MMAA-MMAB cobalamin pathway; depletes carnitine; worsens MMA; FATAL — use LEV as first-line AED"},
        ],
        "patient_sample": [
            {**p, "methylcitrate_mild": round(random.uniform(3, 45), 1)} for p in sample
        ],
    }


def get_definitions() -> dict:
    return {
        "gene_card": {
            "Gene": "MMAA (Methylmalonyl-CoA Mutase-Associated GTPase; previously MMAA gene)",
            "Full Name": "Methylmalonyl-CoA Mutase-Associated GTPase (Cobalamin A Chaperone)",
            "Chromosome": "4p14",
            "Inheritance": "Autosomal Recessive (AR)",
            "OMIM Gene": "*607481",
            "OMIM Disease": "#251100 (Methylmalonic Acidemia, cblA type)",
            "Protein Size": "418 amino acids; mitochondrial matrix; mitochondrial targeting sequence cleaved in mature form",
            "Structure": "G3E P-loop GTPase (G-protein type chaperone); forms complex with MMAB and MMUT during AdoCbl transfer",
            "Cofactor Required": "GTP (for MMAA conformational switch during AdoCbl delivery)",
            "Substrate/Function": "Chaperone: transfers AdoCbl from MMAB (cblB) to MMUT apoenzyme TIM-barrel active site",
            "Prevalence": "~1:100,000–200,000; accounts for ~25–30% of cobalamin-responsive isolated MMA",
            "NBS": "C3 (propionylcarnitine) elevated — identical trigger to PA and MMUT; urine organic acids MANDATORY",
            "Key Positive": "Urine MMA 200–5,000 mmol/mol Cr (PATHOGNOMONIC); OHCbl response >60% (HALLMARK of cblA)",
            "Key Negative": "Homocysteine NORMAL (vs cblC); BCAA NORMAL; Methylcitrate NOT pathognomonic (vs PA); MMUT apoenzyme INTACT when AdoCbl provided",
        },
        "key_concepts": [
            {
                "concept": "Why MMAA causes MMA without MMUT enzyme defect — the chaperone distinction",
                "explanation": (
                    "MMUT requires adenosylcobalamin (AdoCbl) as a cofactor in its TIM-barrel active site. "
                    "This cofactor must be synthesized (MMAB/cblB: ATP:cob(I)alamin adenosyltransferase) "
                    "and then DELIVERED to MMUT by a GTPase chaperone (MMAA/cblA). In MMAA LOF: "
                    "AdoCbl is made by MMAB but CANNOT be transferred to MMUT → MMUT apoenzyme is cofactor-empty. "
                    "Critically, the MMUT APOENZYME ITSELF IS STRUCTURALLY INTACT. "
                    "This is why: (1) fibroblasts from MMAA patients show MMUT enzyme activity when AdoCbl is "
                    "provided exogenously; (2) pharmacological OHCbl (which floods the cobalamin pathway) "
                    "partially overcomes the MMAA delivery defect → >60% of MMAA patients respond to B12 therapy. "
                    "MMUT mut0 patients: the apoenzyme IS defective — adding AdoCbl externally does nothing "
                    "because there is no functional enzyme to bind it. This distinction is fundamental."
                ),
            },
            {
                "concept": "Why MMAA cblA patients have stronger OHCbl response than MMUT (~60–80% vs ~20%)",
                "explanation": (
                    "In MMAA deficiency, the block is in AdoCbl DELIVERY (the MMAA chaperone step). "
                    "When pharmacological doses of hydroxocobalamin (OHCbl 1–2 mg/day IM) are given: "
                    "the increased cobalamin flux partially saturates MMAB, which can then generate more AdoCbl; "
                    "some AdoCbl reaches MMUT via alternative (non-MMAA) pathways at high substrate concentration. "
                    "Because MMUT apoenzyme is INTACT in MMAA patients, any AdoCbl that reaches the TIM-barrel "
                    "is catalytically active → MMA drops. "
                    "In MMUT mut- (partial): the apoenzyme has residual structure and can sometimes bind "
                    "AdoCbl at high concentrations → modest response in ~20% of MMUT patients. "
                    "In MMUT mut0: the enzyme is absent → no binding possible → ZERO response regardless of B12 dose."
                ),
            },
            {
                "concept": "CKD in MMAA — less severe than MMUT, but still requires monitoring",
                "explanation": (
                    "Methylmalonic acid (MMA) is a direct renal proximal tubular toxin in all MMA disorders. "
                    "In MMAA, because >60% of patients respond to OHCbl (reducing MMA substantially), "
                    "the chronic renal MMA exposure is LOWER than in MMUT mut0 (where MMA cannot be reduced). "
                    "Consequently, MMAA patients develop CKD less frequently (~18%) and later than MMUT mut0 (~35%). "
                    "However, OHCbl non-responders (null MMAA alleles) still accumulate MMA → progressive CKD. "
                    "Monitoring: eGFR and urine protein every 6 months; nephrology co-management for eGFR <60. "
                    "Kidney transplant rarely needed in MMAA (unlike MMUT mut0 where combined liver-kidney Tx is standard)."
                ),
            },
            {
                "concept": "Differentiating MMAA (cblA) from MMAB (cblB) and MMUT — three causes of isolated MMA",
                "explanation": (
                    "Isolated MMA (without elevated homocysteine) has three primary genetic causes: "
                    "(1) MMUT defect — apoenzyme absent (mut0) or structurally defective (mut-); "
                    "(2) MMAA (cblA) defect — GTPase chaperone; AdoCbl delivery absent; MMUT apoenzyme INTACT; "
                    "(3) MMAB (cblB) defect — AdoCbl synthesis absent (ATP:cob(I)alamin adenosyltransferase); "
                    "MMUT apoenzyme INTACT. "
                    "Key differential markers: "
                    "• OHCbl response: MMAA ~60–80%; MMAB ~40–50%; MMUT ~20% (mut- only). "
                    "• Fibroblast enzyme assay: MMAA and MMAB show MMUT activity NORMAL when AdoCbl provided; "
                    "MMUT mut0 does NOT correct. "
                    "• Gene panel: MMUT, MMAA, MMAB, MMACHC, MMADHC — gene-level diagnosis mandatory. "
                    "Biochemistry alone CANNOT reliably distinguish MMAA from MMAB."
                ),
            },
            {
                "concept": "VPA ABSOLUTE CI in MMAA — mechanism specific to the cobalamin pathway",
                "explanation": (
                    "Valproate (VPA) is ABSOLUTELY CONTRAINDICATED in MMAA (and all MMA disorders). "
                    "In MMAA specifically: VPA and its CoA-thioester metabolites directly disrupt "
                    "the cobalamin processing and adenosylcobalamin delivery pathway — the very pathway MMAA "
                    "is needed to facilitate. VPA inhibits MMAB and perturbs mitochondrial cobalamin trafficking, "
                    "further reducing the already-impaired AdoCbl delivery to MMUT. Additionally: "
                    "VPA massively depletes carnitine (via VPA-carnitine conjugation and renal excretion) — "
                    "worsening the existing secondary carnitine depletion in MMAA. "
                    "The combination → MMA ↑↑↑ + carnitine depletion + hepatic mitochondrial failure: FATAL. "
                    "LEV (levetiracetam) is the first-line AED for all seizure types in MMAA."
                ),
            },
            {
                "concept": "Why MMAA is a G3E P-loop GTPase — the GTP-energy mechanism for chaperone action",
                "explanation": (
                    "MMAA belongs to the G3E subfamily of P-loop NTPases (same family as MeaB in bacteria). "
                    "GTP hydrolysis by MMAA drives a conformational switch that: "
                    "(1) allows MMAA to bind MMAB when AdoCbl is loaded onto MMAB; "
                    "(2) the MMAA:GTP:MMAB:AdoCbl complex forms a transient ternary complex with MMUT; "
                    "(3) GTP hydrolysis → GDP + Pi → MMAA conformational change releases MMAB; "
                    "(4) AdoCbl is simultaneously captured by the MMUT TIM-barrel. "
                    "The G3 motif (DXXG — includes Arg145 and surrounding residues) is the 'switch II' "
                    "GTPase motif. Arg145His (p.Arg145His — most common variant) disrupts this motif "
                    "but retains partial GTPase activity → partial AdoCbl transfer → strong OHCbl response."
                ),
            },
            {
                "concept": "Secondary hyperammonemia in MMAA — same NAGS/CPS1 mechanism as MMUT and PA",
                "explanation": (
                    "In MMAA, L-methylmalonyl-CoA accumulates (same as MMUT). Methylmalonyl-CoA and MMA "
                    "inhibit N-acetylglutamate synthase (NAGS) → reduced N-acetylglutamate (NAG) → "
                    "CPS1 (carbamoyl phosphate synthetase 1) downregulation → urea cycle impaired → "
                    "secondary hyperammonemia. Same mechanism as PA (propionyl-CoA → NAGS inhibition) and MMUT. "
                    "This is NOT primary hyperammonemia — it resolves when MMA is corrected (OHCbl + IV glucose). "
                    "Management: IV glucose first (halts catabolism → MMA falls → NAGS recovers); "
                    "Na benzoate/phenylacetate if NH3 >200 µmol/L; HD/CRRT if NH3 >500 (rare in MMAA vs MMUT)."
                ),
            },
        ],
        "diagnostic_thresholds": [
            {"parameter": "Urine MMA", "threshold": ">5 mmol/mol Cr (abnormal); >200 = MMA diagnosis; >3,000 = severe/null allele MMAA", "action": "Confirm plasma MMA + homocysteine; OHCbl trial 1–2 mg/day IM × 5 days; gene panel (MMUT, MMAA, MMAB, MMACHC, MMADHC)"},
            {"parameter": "OHCbl Trial Response", "threshold": "MMA reduction >50% at Day 3–5 of OHCbl 1–2 mg IM/day = RESPONSIVE (cblA); <50% = non-responder", "action": "Lifelong IM OHCbl if responsive; gene panel confirms MMAA; liver transplant planning if null alleles and non-responsive"},
            {"parameter": "Plasma MMA", "threshold": ">0.4 µmol/L (abnormal); >100 µmol/L = significant; >800 = severe / acute crisis risk", "action": "Confirm OHCbl response status; correlate with eGFR; acute: IV glucose + metabolic team"},
            {"parameter": "Plasma Ammonia", "threshold": ">100 µmol/L (action threshold); >200 = start scavengers; >500 = HD/CRRT + ICU (rare in MMAA)", "action": "IV glucose + STOP protein; Na benzoate 250 mg/kg; HD/CRRT if >500 (uncommon in OHCbl-responsive MMAA)"},
            {"parameter": "eGFR", "threshold": "<90 = monitor 6-monthly; <60 = CKD stage 3+ (nephrology co-management); <45 = escalate monitoring", "action": "Nephrology co-management; avoid nephrotoxic drugs; kidney transplant rarely needed in MMAA (unlike MMUT)"},
            {"parameter": "Free Carnitine", "threshold": "<25 µmol/L = insufficient; <15 = significant depletion; <10 = crisis risk", "action": "L-carnitine 100 mg/kg/day PO; IV during crisis; monitor acylcarnitine ratio (C3/free carnitine); may improve with OHCbl alone in responsive patients"},
        ],
        "differential_diagnosis": [
            {"disease": "MMUT (Methylmalonyl-CoA Mutase deficiency) — mut0/mut-", "distinguishing": "MMUT apoenzyme DEFICIENT (vs MMAA: apoenzyme INTACT); OHCbl response ONLY ~20% (vs MMAA >60%); mut0 neonatal severe; fibroblasts do NOT correct with AdoCbl (vs MMAA: DO correct)"},
            {"disease": "MMAB (cblB) — ATP:cob(I)alamin adenosyltransferase deficiency", "distinguishing": "Also isolated MMA without HHcy; MMUT apoenzyme intact (same as MMAA); OHCbl response ~40–50%; defect in AdoCbl SYNTHESIS (MMAB), not delivery (MMAA); biochemically identical to MMAA — gene panel mandatory"},
            {"disease": "cblC — MMACHC (Combined MMA + Homocystinuria)", "distinguishing": "HOMOCYSTEINE ELEVATED (>100 µmol/L) + MMA elevated — MMAA has NORMAL homocysteine; cblC also causes retinopathy, microangiopathy, hemolytic uremic syndrome"},
            {"disease": "PA — PCCA/PCCB (Propionic Acidemia)", "distinguishing": "Methylcitrate PATHOGNOMONIC (50–2,000 µmol/mmol Cr); NO methylmalonate; C3 elevated same as MMAA — urine OA mandatory; biotin NOT effective; no OHCbl response"},
            {"disease": "cblD (MMADHC) — isolated MMA or combined MMA+HHcy", "distinguishing": "MMADHC mutations cause either combined (MMA+HHcy) or isolated MMA depending on allele position; gene panel distinguishes; OHCbl response variable"},
            {"disease": "DLD Deficiency (E3 subunit — combined BCAA/pyruvate/2-OG)", "distinguishing": "BCAA ↑ (Leu/Ile/Val) + 2-hydroxyglutarate ↑ + glycine ↑ + lactate ↑; MMAA has NORMAL BCAA; DLD is a 4-complex block; MMA mildly elevated but not PATHOGNOMONIC"},
            {"disease": "SUCLA2 / SUCLG1 (Succinyl-CoA ligase defect)", "distinguishing": "MMA mildly elevated; mtDNA depletion syndrome features; deafness; cardiomyopathy; Leigh-like MRI; gene panel; OHCbl NOT effective"},
        ],
    }
