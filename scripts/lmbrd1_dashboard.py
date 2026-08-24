#!/usr/bin/env python3
"""LMBRD1 (Methylmalonic Acidemia with Homocystinuria, cblF type) Epilepsy Dashboard.

LMBRD1 encodes a 540-aa lysosomal membrane protein (9 TM helices) that functions as
the MOST UPSTREAM intracellular cobalamin transporter — it exports cobalamin from
the LYSOSOMAL LUMEN into the cytoplasm, where MMACHC can then process it.

COBALAMIN PATHWAY — LMBRD1 IS STEP 1 (MOST UPSTREAM):
  Step 0: Dietary cobalamin bound to transcobalamin II (TC2) → TC2-receptor (CD320) →
          receptor-mediated endocytosis → lysosome (cobalamin trapped in lysosome).
  Step 1: LMBRD1 (cblF) — exports cobalamin from lysosomal lumen → cytoplasm.
          ← BLOCK IN cblF: cobalamin CANNOT LEAVE the lysosome.
  Step 2: MMACHC (cblC; 1p34.1) — converts cobalamin to cob(I)alamin in cytoplasm.
          (MMACHC receives NO substrate when LMBRD1 is defective.)
  Step 3: MMADHC (cblD; 2q23.2) — distributes cob(I)alamin to BOTH arms.
  Step 4A: MMAB → AdoCbl → MMUT → MMA catabolism (mitochondrial arm).
  Step 4B: MTR/MTRR → MeCbl → Methionine Synthase → Hcy remethylation (cytoplasmic arm).

LMBRD1 LOF RESULT — BOTH DOWNSTREAM ARMS BLOCKED SIMULTANEOUSLY:
  → MMA ELEVATED (200–1,500 mmol/mol Cr): mitochondrial arm starved of AdoCbl.
  → tHcy ELEVATED (30–200 µmol/L): cytoplasmic arm starved of MeCbl.
  → Methionine LOW (8–20 µmol/L): Methionine Synthase (MTR) inactive.
  → MeCbl fibroblasts: ABSENT.
  → AdoCbl fibroblasts: ABSENT.
  → Biochemically: COMBINED MMA + HHcy — identical pattern to cblC/MMACHC.

UNIQUE cblF (LMBRD1) FEATURES vs cblC (MMACHC):
  1. MOST UPSTREAM block: cobalamin trapped INSIDE lysosomes (not cytoplasm).
  2. Vacuolated lymphocytes: occasionally present (lysosomal storage feature);
     ABSENT in cblC/MMACHC — fastest bedside distinction at blood smear.
  3. Stomatitis / oral ulcers: characteristic early sign in cblF;
     NOT seen in cblC — pathognomonic mucosal sensitivity.
  4. Onset: typically infantile (2–12 months); later than cblC (often neonatal).
  5. MMA levels: often LOWER than cblC (200–1,500 vs 200–2,000 mmol/mol Cr).
  6. OHCbl bypasses lysosomal trap via alternative cellular entry routes (IM);
     response: ~50–70% HHcy, ~40–60% MMA — slightly less than cblC.
  7. Cobalamin visible in lysosomal compartment on fibroblast EM — diagnostic.
  8. Gene panel: LMBRD1 + ABCD4 (cblJ) mandatory — both block lysosomal cobalamin export.

LMBRD1 PROTEIN BIOLOGY:
  540 amino acids; integral lysosomal membrane protein; 9 TM helices; LMBR1 domain.
  N-terminal domain faces the cytoplasm; C-terminal tail faces lysosomal lumen.
  Forms functional complex with ABCD4 (cblJ) at lysosomal membrane for export.
  LMBRD1 mutation disrupts the transporter pore or ABCD4 interaction → cobalamin stranded in lysosome.

LMBRD1 BIOMARKERS:
  cblF combined pattern (both MMA + HHcy elevated):
    C3 (propionylcarnitine): 3–20 µmol/L↑ (NBS PRIMARY)
    Urine MMA: 200–1,500 mmol/mol Cr↑ (LOWER than MMUT mut0)
    tHcy: 30–200 µmol/L↑ (PATHOGNOMONIC combined pattern with MMA)
    Methionine: 8–20 µmol/L LOW (methionine synthase inactive)
    MeCbl fibroblasts: ABSENT
    AdoCbl fibroblasts: ABSENT
    Methylcitrate: mildly secondary elevated (NOT pathognomonic vs PA)
    Vacuolated lymphocytes: ~25–35% patients (lysosomal storage; ABSENT in cblC)

KEY VARIANTS IN LMBRD1:
  c.1056delG (exon 9; p.Leu353fsX7): most common reported allele; European; null frameshift;
    early infantile severe; biallelic null → worst phenotype.
  p.Arg83Cys (c.247C>T): TM domain 2; partial function; infantile classic; moderate.
  p.Glu299Lys (c.895G>A): luminal loop 3; moderate severity; 2–8 month onset.
  p.Gly243Ser (c.727G>A): TM domain 6 boundary; moderate; infantile classic.
  c.IVS9+1G>A (splice site exon 9): null; severe early infantile.
  p.Thr275Ile (c.824C>T): TM domain 7; partial function; moderate attenuated.

TREATMENT:
  OHCbl 1–2 mg/day IM (Level A): bypasses lysosomal trap via alternative entry routes;
    ~50–70% HHcy response; ~40–60% MMA response; lifelong.
  Betaine (TMG) 100–200 mg/kg/day (Level A): MANDATORY for HHcy component;
    cobalamin-independent BHMT remethylation of Hcy.
  Folinic acid 0.5–5 mg/day (Level B): 5-methylTHF depleted; supports methylation cycle.
  L-Carnitine 100 mg/kg/day (Level A–B): secondary depletion via propionylcarnitine excretion.
  Protein restriction + MMA-free formula (Level A): for MMA component (Ile/Val/Met/Thr restriction).
  IV Glucose GIR 8–12 (Level A): acute MMA crisis; stop BCAA catabolism.

CONTRAINDICATIONS:
  N2O — ABSOLUTE CI: inactivates methionine synthase (same mechanism as cblC/MMACHC).
  VPA — HIGH RISK: carnitine depletion + metabolic stress → MMA surge; LEV first-line.
  Fasting — HIGH RISK: BCAA catabolism → MMA surge + HHcy worsens.

OMIM: Gene *612634 · Disease #277380 (Methylmalonic Aciduria and Homocystinuria, cblF type)
Chromosome: 6q13 · Inheritance: AR · Prevalence: ~1:1,000,000 (very rare; <50 reported cases)
"""

from __future__ import annotations
import random

random.seed(54)   # reproducible synthetic cohort


# ── helpers ───────────────────────────────────────────────────────────────────

def _pid(i: int) -> str:
    return f"LMBRD1-{i:03d}"


def _sex(i: int) -> str:
    return "M" if i % 2 == 0 else "F"


def _phenotype(i: int) -> str:
    """
    Early Infantile Severe (~40%) — <6 months onset, stomatitis, vacuolated lymphocytes, NH3↑
    Infantile Classic (~45%)      — 2–18 months onset, stomatitis + combined MMA+HHcy
    Late Infantile Attenuated (~15%) — 12–36 months onset, milder
    """
    if i < 16:
        return "Early Infantile Severe"
    elif i < 34:
        return "Infantile Classic"
    else:
        return "Late Infantile Attenuated"


def _genotype(i: int, pheno: str) -> str:
    pheno_variants = {
        "Early Infantile Severe": [
            "c.1056delG/c.1056delG (homozygous null; most severe)",
            "c.1056delG/c.IVS9+1G>A (compound het null; severe)",
            "c.1056delG/p.Arg83Cys (null/partial; severe)",
            "c.IVS9+1G>A/c.IVS9+1G>A (splice-null; severe)",
        ],
        "Infantile Classic": [
            "p.Arg83Cys/p.Glu299Lys (compound het; moderate-severe)",
            "c.1056delG/p.Gly243Ser (null/partial; moderate)",
            "p.Arg83Cys/p.Arg83Cys (homozygous partial; moderate)",
            "p.Gly243Ser/p.Glu299Lys (compound het; moderate)",
            "c.1056delG/p.Thr275Ile (null/partial; moderate)",
            "p.Glu299Lys/p.Glu299Lys (homozygous luminal-loop; moderate)",
        ],
        "Late Infantile Attenuated": [
            "p.Thr275Ile/p.Thr275Ile (homozygous; mild-moderate residual)",
            "p.Gly243Ser/p.Thr275Ile (compound het; attenuated)",
            "p.Arg83Cys/p.Thr275Ile (partial/partial; mild)",
        ],
    }
    opts = pheno_variants[pheno]
    return opts[i % len(opts)]


def _mma_urine(pheno: str) -> float:
    if pheno == "Early Infantile Severe":
        return round(random.uniform(400, 1500), 0)
    elif pheno == "Infantile Classic":
        return round(random.uniform(200, 800), 0)
    else:
        return round(random.uniform(100, 400), 0)


def _homocysteine(pheno: str) -> float:
    if pheno == "Early Infantile Severe":
        return round(random.uniform(60, 200), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(30, 120), 1)
    else:
        return round(random.uniform(20, 80), 1)


def _methionine(pheno: str) -> float:
    if pheno == "Early Infantile Severe":
        return round(random.uniform(8, 14), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(10, 18), 1)
    else:
        return round(random.uniform(14, 22), 1)


def _c3(pheno: str) -> float:
    if pheno == "Early Infantile Severe":
        return round(random.uniform(6, 20), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(3, 12), 1)
    else:
        return round(random.uniform(2, 7), 1)


def _ammonia(pheno: str) -> int:
    if pheno == "Early Infantile Severe":
        return random.randint(100, 450)
    elif pheno == "Infantile Classic":
        return random.randint(40, 120)
    else:
        return random.randint(20, 60)


def _carnitine(pheno: str) -> float:
    if pheno == "Early Infantile Severe":
        return round(random.uniform(8, 22), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(12, 30), 1)
    else:
        return round(random.uniform(18, 40), 1)


def _onset_months(pheno: str) -> int:
    if pheno == "Early Infantile Severe":
        return random.randint(0, 5)
    elif pheno == "Infantile Classic":
        return random.randint(2, 18)
    else:
        return random.randint(12, 36)


def _ohcbl_response(pheno: str) -> bool:
    rates = {"Early Infantile Severe": 0.55, "Infantile Classic": 0.65, "Late Infantile Attenuated": 0.80}
    return random.random() < rates[pheno]


def _seizures(pheno: str) -> bool:
    rates = {"Early Infantile Severe": 0.80, "Infantile Classic": 0.65, "Late Infantile Attenuated": 0.40}
    return random.random() < rates[pheno]


def _stomatitis(pheno: str) -> bool:
    rates = {"Early Infantile Severe": 0.75, "Infantile Classic": 0.65, "Late Infantile Attenuated": 0.35}
    return random.random() < rates[pheno]


def _vacuolated_lymphocytes(pheno: str) -> bool:
    rates = {"Early Infantile Severe": 0.45, "Infantile Classic": 0.25, "Late Infantile Attenuated": 0.10}
    return random.random() < rates[pheno]


def _nbs(pheno: str) -> bool:
    # C3 elevated in all cblF subtypes → NBS usually detected
    rates = {"Early Infantile Severe": 0.80, "Infantile Classic": 0.70, "Late Infantile Attenuated": 0.55}
    return random.random() < rates[pheno]


def _aed(sz: bool) -> str:
    if not sz:
        return "None"
    opts = ["LEV", "LEV + CLB", "LEV + ACTH (IS)", "LEV + CLN", "LEV + PB (neonate)"]
    return random.choice(opts)


def _diag_months(pheno: str, onset: int) -> int:
    delays = {"Early Infantile Severe": (1, 6), "Infantile Classic": (2, 12), "Late Infantile Attenuated": (4, 18)}
    lo, hi = delays[pheno]
    return onset + random.randint(lo, hi)


# ── build cohort ──────────────────────────────────────────────────────────────

def _build_cohort() -> list[dict]:
    pts = []
    for i in range(40):
        pheno = _phenotype(i)
        sz = _seizures(pheno)
        onset = _onset_months(pheno)
        pts.append({
            "patient_id": _pid(i),
            "sex": _sex(i),
            "phenotype": pheno,
            "genotype": _genotype(i, pheno),
            "onset_months": onset,
            "diagnosis_months": _diag_months(pheno, onset),
            "mma_urine_mmol_molCr": _mma_urine(pheno),
            "total_homocysteine_umol_l": _homocysteine(pheno),
            "methionine_umol_l": _methionine(pheno),
            "c3_umol_l": _c3(pheno),
            "ammonia_umol_l": _ammonia(pheno),
            "free_carnitine_umol_l": _carnitine(pheno),
            "mecbl_fibroblasts": "Absent",   # both absent in cblF (all cases)
            "adocbl_fibroblasts": "Absent",  # both absent in cblF (all cases)
            "seizures": sz,
            "stomatitis": _stomatitis(pheno),
            "vacuolated_lymphocytes": _vacuolated_lymphocytes(pheno),
            "nbs_detected": _nbs(pheno),
            "ohcbl_response": _ohcbl_response(pheno),
            "aed": _aed(sz),
        })
    return pts


_PATIENTS: list[dict] = _build_cohort()


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

    early_pts  = [p for p in _PATIENTS if p["phenotype"] == "Early Infantile Severe"]
    class_pts  = [p for p in _PATIENTS if p["phenotype"] == "Infantile Classic"]
    late_pts   = [p for p in _PATIENTS if p["phenotype"] == "Late Infantile Attenuated"]

    return {
        "gene": "LMBRD1",
        "full_name": "LMBR1 Domain Containing 1 — Lysosomal Cobalamin Transporter (cblF gene product)",
        "chromosome": "6q13",
        "inheritance": "AR",
        "omim_gene": "*612634",
        "omim_disease": "#277380",
        "protein_size": "540 aa; integral lysosomal membrane protein; 9 TM helices; LMBR1 domain; forms complex with ABCD4",
        "prevalence": "~1:1,000,000 (very rare; <50 reported cases worldwide; likely underdiagnosed)",
        "nbs_primary": "C3 (propionylcarnitine) elevated — cblF detected by NBS in ~70% of cases (all subtypes have combined MMA+HHcy → C3 elevated)",
        "nbs_secondary": "Urine MMA + plasma tHcy simultaneously mandatory; NBS reflex confirms; gene panel (LMBRD1 + ABCD4) distinguishes cblF from cblC/cblJ",
        "function": (
            "LMBRD1 (cblF) is the MOST UPSTREAM intracellular cobalamin transporter. "
            "After dietary cobalamin is endocytosed with transcobalamin II (TC2) via CD320 receptor → lysosome, "
            "LMBRD1 exports cobalamin from the lysosomal lumen into the cytoplasm. "
            "MMACHC (cblC) then processes this exported cobalamin into cob(I)alamin for both downstream arms. "
            "LMBRD1 LOF → cobalamin TRAPPED in lysosomes → MMACHC receives NO substrate → "
            "BOTH metabolic arms blocked simultaneously → Combined MMA + HHcy."
        ),
        "mechanism": (
            "LMBRD1 LOF traps all dietary cobalamin inside lysosomes. "
            "Neither the mitochondrial arm (AdoCbl → MMUT → MMA catabolism) "
            "nor the cytoplasmic arm (MeCbl → Methionine Synthase → Hcy remethylation) receives cobalamin substrate. "
            "Result: MMA ELEVATED + tHcy ELEVATED + Methionine LOW — identical pattern to cblC/MMACHC "
            "but with UNIQUE features: stomatitis, occasional vacuolated lymphocytes, "
            "lysosomal cobalamin accumulation visible on EM."
        ),
        "key_negative": (
            "BCAA NORMAL — KEY NEGATIVE vs DLD/MSUD; "
            "Methylcitrate only mildly elevated (secondary) — KEY NEGATIVE vs PA (propionic acidemia, PCCA/PCCB); "
            "Vacuolated lymphocytes: ABSENT in cblC/MMACHC — KEY POSITIVE in cblF when present (lysosomal storage); "
            "Stomatitis: ABSENT in cblC — KEY POSITIVE (pathognomonic mucosal sign when present); "
            "Biotinidase NORMAL — KEY NEGATIVE vs BTD."
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
            "stomatitis_pct": pct(lambda p: p["stomatitis"]),
            "vacuolated_lymphocyte_pct": pct(lambda p: p["vacuolated_lymphocytes"]),
            "nbs_detected_pct": pct(lambda p: p["nbs_detected"]),
            "ohcbl_response_pct": pct(lambda p: p["ohcbl_response"]),
            "early_infantile_pct": round(len(early_pts) / n * 100),
            "infantile_classic_pct": round(len(class_pts) / n * 100),
            "late_infantile_pct": round(len(late_pts) / n * 100),
        },
        "phenotype_distribution": [
            {"phenotype": "Early Infantile Severe (~40%)",
             "pct": 40, "onset": "0–6 months", "stomatitis": "75%", "vacuolated_lymphocytes": "45%",
             "mma": "400–1,500 mmol/mol Cr", "tHcy": "60–200 µmol/L",
             "note": "Neonatal/early infantile onset; highest MMA+HHcy; stomatitis often first sign; occasional vacuolated lymphocytes; NH3 100–450; combined metabolic crisis"},
            {"phenotype": "Infantile Classic (~45%)",
             "pct": 45, "onset": "2–18 months", "stomatitis": "65%", "vacuolated_lymphocytes": "25%",
             "mma": "200–800 mmol/mol Cr", "tHcy": "30–120 µmol/L",
             "note": "Modal phenotype; combined MMA+HHcy; stomatitis common; vacuolated lymphocytes in ~25%; maculopathy 25%; OHCbl + betaine responsive ~65%"},
            {"phenotype": "Late Infantile Attenuated (~15%)",
             "pct": 15, "onset": "12–36 months", "stomatitis": "35%", "vacuolated_lymphocytes": "10%",
             "mma": "100–400 mmol/mol Cr", "tHcy": "20–80 µmol/L",
             "note": "Later onset; milder MMA and HHcy; partial LMBRD1 function; OHCbl + betaine highly effective; better neurodevelopmental outcome"},
        ],
        "lmbrd1_pathway": [
            {
                "step": "Step 0 — Dietary cobalamin + TC2 → CD320 receptor → endocytosis → lysosome",
                "reaction": "Cobalamin-TC2 complex binds CD320 (transcobalamin receptor) → clathrin endocytosis → lysosomal lumen",
                "enzyme": "CD320 (transcobalamin receptor); TC2 degraded by lysosomal proteases; free cobalamin in lysosome",
                "consequence_lof": "Not applicable — this step is TC2/CD320-dependent, not LMBRD1",
            },
            {
                "step": "Step 1 — LMBRD1 (cblF): lysosomal cobalamin → cytoplasm ← BLOCK IN cblF",
                "reaction": "Free cobalamin (lysosomal lumen) → LMBRD1 transporter pore → cytoplasm [requires ABCD4 cooperation]",
                "enzyme": "LMBRD1 (cblF; 540 aa; 6q13; lysosomal membrane; 9 TM helices; LMBR1 domain)",
                "consequence_lof": "Cobalamin TRAPPED in lysosomes; MMACHC receives no substrate; both downstream arms blocked",
            },
            {
                "step": "Step 2 — MMACHC (cblC): cytoplasmic cobalamin → cob(I)alamin [receives NO substrate in cblF]",
                "reaction": "Cobalamin (CN/OH/Me/Ado forms) + NADPH → cob(I)alamin [MMACHC decyanase/reductase]",
                "enzyme": "MMACHC (cblC; 282 aa; 1p34.1; cytoplasmic; bifunctional)",
                "consequence_lof": "Cannot proceed — no cobalamin delivered from LMBRD1 in cblF",
            },
            {
                "step": "Step 3 — MMADHC (cblD): distributes cob(I)alamin to ARM A (mitochondrial) + ARM B (cytoplasmic)",
                "reaction": "cob(I)alamin → ARM A [C-terminus → MMAB → AdoCbl] + ARM B [N-terminus → MeCbl → MTR]",
                "enzyme": "MMADHC (cblD; 296 aa; 2q23.2; cytoplasmic + mitochondrial)",
                "consequence_lof": "Cannot proceed — no cob(I)alamin from MMACHC in cblF",
            },
            {
                "step": "Step 4A — MMAB → AdoCbl → MMUT → MMA catabolism [ARM A; BLOCKED in cblF]",
                "reaction": "cob(I)alamin + ATP → AdoCbl → MMUT: L-methylmalonyl-CoA → succinyl-CoA",
                "enzyme": "MMAB (cblB; 12q24.11) + MMAA (cblA; 4p14) + MMUT (6p12.3)",
                "consequence_lof": "MMA ACCUMULATES (200–1,500 mmol/mol Cr); C3 elevated on NBS",
            },
            {
                "step": "Step 4B — MeCbl → MTR → Hcy remethylation [ARM B; BLOCKED in cblF]",
                "reaction": "cob(I)alamin → MeCbl → MTR: Hcy + 5-methylTHF → Methionine + THF",
                "enzyme": "MTR (methionine synthase; 1q43) + MTRR (cblE; 5p15.3)",
                "consequence_lof": "tHcy ELEVATED; Methionine LOW; same as cblC/MMACHC combined block",
            },
        ],
        "high_risk_situations": [
            {"situation": "Nitrous oxide (N2O)",
             "risk": "ABSOLUTE CI",
             "detail": "N2O irreversibly oxidizes cob(I)alamin → cannot synthesize MeCbl → Methionine Synthase (MTR) completely inactivated → acute HHcy crisis + metabolic catastrophe. LIFE-THREATENING. Same absolute CI as cblC/MMACHC."},
            {"situation": "VPA (Valproate)",
             "risk": "HIGH RISK",
             "detail": "VPA-CoA inhibits mitochondrial cobalamin metabolism + carnitine depletion + MMA surge. HIGH RISK in combined MMA+HHcy (cblF). Use LEV first-line for ALL seizure types in LMBRD1/cblF."},
            {"situation": "Fasting / catabolism",
             "risk": "HIGH RISK",
             "detail": "BCAA catabolism → propionyl-CoA → MMA surge; catabolism also worsens HHcy remethylation deficit. IV glucose + OHCbl IM throughout illness. Never NPO without IV dextrose."},
            {"situation": "Missed OHCbl + betaine",
             "risk": "HIGH RISK",
             "detail": "tHcy rebounds within 48–72 h; MMA rebounds within 3–7 days. Both medications mandatory together — neither alone is sufficient."},
            {"situation": "Antifolates (methotrexate / trimethoprim)",
             "risk": "HIGH RISK",
             "detail": "Further depletes 5-methylTHF → worsens methionine synthase pathway → acute HHcy rise. Folinic acid supplementation active monitoring essential."},
        ],
    }


def get_breakdown() -> dict:
    sample = _PATIENTS[:15]
    return {
        "biomarkers": [
            {"name": "C3 (Propionylcarnitine)", "normal": "<4 µmol/L",
             "lmbrd1_range": "3–20 µmol/L ELEVATED (all cblF subtypes have combined MMA+HHcy → C3 elevated)",
             "significance": "NBS PRIMARY TRIGGER — detected in ~70% of cblF on standard NBS. Unlike cblD-HHcy only (MMADHC) which misses NBS, cblF has C3 elevated because MMA component always present. Must be followed by urine MMA + tHcy simultaneously.",
             "method": "NBS tandem MS/MS; plasma acylcarnitines"},
            {"name": "Methylmalonic Acid (urine)", "normal": "<5 mmol/mol Cr",
             "lmbrd1_range": "200–1,500 mmol/mol Cr (LOWER than MMUT mut0 5,000–10,000; similar or lower than cblC 200–2,000)",
             "significance": "ELEVATED in ALL cblF patients (mitochondrial arm completely blocked). Combined with tHcy elevation = combined MMA+HHcy pattern. MMA alone (without HHcy) would suggest MMUT/MMAA/MMAB (isolated MMA).",
             "method": "Urine organic acids (GC-MS)"},
            {"name": "Total Homocysteine (tHcy)", "normal": "<15 µmol/L",
             "lmbrd1_range": "30–200 µmol/L ELEVATED (ALL cblF patients have HHcy — cytoplasmic arm blocked)",
             "significance": "PATHOGNOMONIC for combined block when elevated WITH MMA. tHcy + MMA together = combined cobalamin disorder (cblC, cblD-Combined, cblF, cblJ). tHcy alone without MMA elevation = MTR/MTRR or cblD-HHcy only.",
             "method": "Plasma tHcy (HPLC or immunoassay); draw SIMULTANEOUSLY with plasma MMA"},
            {"name": "Methionine (plasma)", "normal": "20–40 µmol/L",
             "lmbrd1_range": "8–20 µmol/L LOW (ALL cblF patients — methionine synthase MTR inactive)",
             "significance": "LOW methionine confirms methionine synthase pathway blocked (MeCbl absent). Combined with high HHcy = methionine cycle failure. Low methionine + high HHcy + high MMA = lysosomal or cytoplasmic cobalamin processing disorder.",
             "method": "Plasma amino acids (quantitative)"},
            {"name": "MeCbl (fibroblasts)", "normal": "Present",
             "lmbrd1_range": "ABSENT in ALL cblF patients (cytoplasmic arm receives no substrate from LMBRD1)",
             "significance": "ABSENT — confirms cytoplasmic arm blockade. Both MeCbl and AdoCbl absent (same as cblC) — distinguishes from MMADHC subtypes where only one cobalamin is absent.",
             "method": "Fibroblast cobalamin distribution assay (specialized metabolic lab)"},
            {"name": "AdoCbl (fibroblasts)", "normal": "Present",
             "lmbrd1_range": "ABSENT in ALL cblF patients (mitochondrial arm receives no substrate from LMBRD1)",
             "significance": "ABSENT — confirms mitochondrial arm blockade. Both MeCbl and AdoCbl absent simultaneously = combined block upstream of MMACHC (cblF/LMBRD1 or cblJ/ABCD4) OR at MMACHC (cblC).",
             "method": "Fibroblast cobalamin distribution assay"},
            {"name": "Vacuolated Lymphocytes", "normal": "Absent",
             "lmbrd1_range": "Present in ~25–45% of cblF patients (lysosomal cobalamin storage marker)",
             "significance": "LYSOSOMAL STORAGE MARKER: vacuolated lymphocytes indicate lysosomal accumulation. KEY POSITIVE for cblF when present — ABSENT in cblC/MMACHC (cytoplasmic block, not lysosomal storage). Blood smear vacuolation + combined MMA+HHcy → high suspicion for LMBRD1 (cblF) or ABCD4 (cblJ).",
             "method": "Peripheral blood smear microscopy (Wright-Giemsa stain)"},
            {"name": "Stomatitis / Oral Ulcers", "normal": "Absent",
             "lmbrd1_range": "Present in ~65% of cblF (Infantile Classic); ~75% Early Infantile Severe",
             "significance": "CHARACTERISTIC cblF FEATURE: stomatitis and oral ulcers reported in first case series and most cblF patients. Mechanism unclear (lysosomal cobalamin in mucosal epithelium?). ABSENT in cblC/MMACHC and isolated MMA. Stomatitis + combined MMA+HHcy in an infant = cblF until proven otherwise.",
             "method": "Clinical examination; oral mucosa inspection at each visit"},
            {"name": "Methylcitrate (urine)", "normal": "<5 µmol/mmol Cr",
             "lmbrd1_range": "5–80 µmol/mmol Cr (mildly elevated; secondary; NOT pathognomonic)",
             "significance": "Mildly elevated secondary to MMA accumulation. KEY NEGATIVE vs propionic acidemia (PCCA/PCCB) where methylcitrate is PATHOGNOMONIC (50–2,000 µmol/mmol Cr). Low methylcitrate + high MMA + high HHcy = cobalamin disorder (not PA).",
             "method": "Urine organic acids (GC-MS)"},
            {"name": "Ammonia (plasma)", "normal": "<50 µmol/L",
             "lmbrd1_range": "Early Infantile Severe: 100–450 µmol/L; Infantile Classic: 40–120 µmol/L",
             "significance": "Secondary hyperammonemia via MMA-mediated NAGS/CPS1 inhibition (same mechanism as MMUT, PA). Lower than PA/MMUT because MMA levels are lower in cblF. Treat with ammonia scavengers if NH3 >200.",
             "method": "Plasma ammonia (enzymatic colorimetric)"},
            {"name": "OHCbl Trial Response (tHcy)", "normal": "N/A",
             "lmbrd1_range": "~50–70% HHcy response (IM OHCbl bypasses lysosomal trap via alternative cell entry)",
             "significance": "OHCbl given IM enters cells by endocytosis but also by diffusion/alternative routes that bypass CD320/lysosomal pathway → partially bypasses LMBRD1 block. Response rate slightly lower than cblC (75% HHcy). Measure tHcy Day 0 vs Day 7.",
             "method": "tHcy before/after 5–7 day OHCbl 1–2 mg/day IM trial"},
            {"name": "OHCbl Trial Response (MMA)", "normal": "N/A",
             "lmbrd1_range": "~40–60% MMA response (similar to cblB; less than cblA)",
             "significance": "Partially bypasses LMBRD1 block for MMA catabolism arm. Response rate similar to MMAB (cblB; ~40–60%). Lower than MMAA (cblA; ~60–80%) because lysosomal bypass less efficient.",
             "method": "Urine MMA before/after OHCbl trial"},
        ],
        "key_variants": [
            {"variant": "c.1056delG", "cdna": "c.1056delG (exon 9; frameshift)", "domain": "TM domain 9 region",
             "severity": "Severe (null; frameshift → premature stop p.Leu353fsX7)", "phenotype": "Early Infantile Severe",
             "note": "Most commonly reported LMBRD1 allele; European founder-like; null frameshift; biallelic = worst cblF phenotype; early infantile onset <3 months; NH3 200–450; combined severe MMA+HHcy; stomatitis + vacuolated lymphocytes"},
            {"variant": "p.Arg83Cys", "cdna": "c.247C>T", "domain": "TM domain 2",
             "severity": "Moderate (partial TM function; reduced but not absent export)", "phenotype": "Infantile Classic",
             "note": "TM domain 2 missense; partial LMBRD1 function; infantile classic onset 2–8 months; combined MMA+HHcy moderate; OHCbl + betaine responsive ~70%; stomatitis common"},
            {"variant": "p.Glu299Lys", "cdna": "c.895G>A", "domain": "Luminal loop 3",
             "severity": "Moderate (luminal loop; ABCD4 interaction impaired)", "phenotype": "Infantile Classic",
             "note": "Luminal loop 3 missense; disrupts cobalamin-binding or ABCD4 interaction; infantile classic; combined MMA+HHcy moderate; OHCbl+betaine response ~65%"},
            {"variant": "p.Gly243Ser", "cdna": "c.727G>A", "domain": "TM domain 6 boundary",
             "severity": "Moderate", "phenotype": "Infantile Classic / Late Infantile Attenuated",
             "note": "TM6 boundary missense; moderate residual export; infantile classic to attenuated onset; partial response to OHCbl + betaine; better neurodevelopmental outcome than null alleles"},
            {"variant": "c.IVS9+1G>A", "cdna": "c.IVS9+1G>A (splice site; exon 9)", "domain": "Global (exon 9 skipped)",
             "severity": "Severe (null; exon 9 skipping → in-frame deletion or NMD)", "phenotype": "Early Infantile Severe",
             "note": "Canonical splice site; exon 9 skipping → null LMBRD1; severe early infantile; similar to c.1056delG phenotype; biallelic null requires aggressive OHCbl + betaine"},
            {"variant": "p.Thr275Ile", "cdna": "c.824C>T", "domain": "TM domain 7",
             "severity": "Mild-Moderate (partial; most residual function)", "phenotype": "Late Infantile Attenuated",
             "note": "TM7 missense; highest residual LMBRD1 function of known variants; late infantile onset 12–36 months; milder MMA and HHcy; best OHCbl+betaine response (~80% HHcy); best neurodevelopmental prognosis"},
        ],
        "seizure_types": [
            {"type": "Early infantile / neonatal encephalopathic seizures (combined MMA+HHcy crisis)", "pct": 70,
             "note": "Multifocal; driven by simultaneous MMA mitochondrial toxicity + HHcy endothelial damage; treat metabolic crisis first: OHCbl IM + betaine + IV glucose; LEV adjunct; AVOID VPA; avoid N2O anesthesia"},
            {"type": "Infantile spasms (West syndrome)", "pct": 32,
             "note": "Early-onset subtypes; hypsarrhythmia on EEG; ACTH or high-dose prednisolone preferred; folinic acid adjunct (methylation cycle support); LEV; metabolic control reduces IS burden"},
            {"type": "Focal cortical seizures", "pct": 28,
             "note": "Cortical injury from combined MMA+HHcy toxicity; LEV first-line; improve with metabolic compliance; map with MRI for focal cortical abnormalities"},
            {"type": "Myoclonic seizures", "pct": 22,
             "note": "Metabolic-triggered by MMA/HHcy spikes; correlate seizure clusters with metabolic compliance; LEV first-line"},
            {"type": "Epileptic status / NCSE", "pct": 12,
             "note": "Early severe subtypes; IV BZD + IV glucose + OHCbl IM + betaine simultaneous; continuous EEG monitoring in neonates and early infants"},
        ],
        "metabolic_triggers": [
            {"trigger": "Nitrous oxide (N2O) — any exposure", "pct": 100,
             "mechanism": "ABSOLUTE CI: N2O irreversibly oxidizes cob(I)alamin → Methionine Synthase (MTR) inactivated → acute HHcy crisis + metabolic catastrophe. ALL cblF patients at risk (MTR already partially deficient). Notify anesthesiology at every surgery."},
            {"trigger": "Fasting / prolonged NPO", "pct": 72,
             "mechanism": "BCAA catabolism → propionyl-CoA → MMA surge; catabolism worsens HHcy remethylation deficit. IV glucose GIR 8–12 mandatory whenever NPO or illness. Emergency protocol."},
            {"trigger": "Intercurrent illness / fever", "pct": 62,
             "mechanism": "Catabolic crisis; MMA surge + HHcy worsens; IV glucose + OHCbl IM throughout illness; betaine maintained. Emergency sick-day protocol activation."},
            {"trigger": "Missed OHCbl + betaine", "pct": 48,
             "mechanism": "tHcy rebounds within 48–72 h; MMA rebounds within 3–7 days. Both medications mandatory simultaneously — neither adequate alone."},
            {"trigger": "High protein intake (Ile/Val/Met/Thr excess)", "pct": 32,
             "mechanism": "Excess propionyl-CoA precursors → MMA surge; protein restriction mandatory for MMA component."},
        ],
        "high_risk_drugs": [
            {"drug": "Nitrous oxide (N2O)", "risk": "ABSOLUTE CI",
             "mechanism": "Irreversibly oxidizes cobalamin cofactor → Methionine Synthase (MTR) failure → acute HHcy crisis. LIFE-THREATENING in all cblF patients. Same absolute CI as cblC/MMACHC."},
            {"drug": "Valproate (VPA)", "risk": "HIGH RISK",
             "mechanism": "VPA-CoA inhibits mitochondrial cobalamin metabolism + carnitine depletion + MMA surge. Use LEV first-line for ALL seizure types in LMBRD1/cblF."},
            {"drug": "Antifolates (methotrexate, trimethoprim)", "risk": "HIGH RISK",
             "mechanism": "Further deplete 5-methylTHF → worsen methionine synthase pathway failure → acute HHcy elevation. Folinic acid monitoring essential."},
            {"drug": "Fasting / NPO without IV glucose", "risk": "HIGH RISK",
             "mechanism": "Catabolism-driven MMA surge + HHcy worsening. IV dextrose + OHCbl mandatory throughout any illness."},
        ],
        "treatments": [
            {"treatment": "Hydroxocobalamin (OHCbl) 1–2 mg/day IM", "evidence": "Level A",
             "response_pct": 62,
             "note": "Trial ALL cblF patients. OHCbl given IM enters cells via alternative routes (not only CD320/lysosome) → partially bypasses LMBRD1 lysosomal trap → provides cob(I)alamin to MMACHC. ~50–70% HHcy response; ~40–60% MMA response. Lifelong IM preferred; monitor tHcy + MMA Day 7."},
            {"treatment": "Betaine (trimethylglycine / TMG) 100–200 mg/kg/day", "evidence": "Level A",
             "response_pct": 80,
             "note": "MANDATORY for HHcy component in ALL cblF patients. Betaine-homocysteine methyltransferase (BHMT) remethylates Hcy to Methionine — cobalamin-INDEPENDENT pathway. Level A for HHcy present. Same mechanism/evidence as cblC/MMACHC."},
            {"treatment": "Protein restriction + MMA-free formula", "evidence": "Level A",
             "response_pct": 72,
             "note": "Low isoleucine/valine/methionine/threonine for MMA component. MMA-free formula Level A. Protein restriction lowers propionyl-CoA precursor load → MMA levels fall."},
            {"treatment": "L-Carnitine 100 mg/kg/day", "evidence": "Level A–B",
             "response_pct": 82,
             "note": "Secondary carnitine depletion via propionylcarnitine excretion. Supplement all cblF patients. Oral supplementation Level A for confirmed depletion."},
            {"treatment": "Folinic acid 0.5–5 mg/day", "evidence": "Level B",
             "response_pct": 55,
             "note": "5-methylTHF depleted when methionine synthase is blocked. Folinic acid bypasses blocked folate cycle. Dose 0.5–5 mg/day. May improve seizure control (infantile spasms) and methylation status."},
            {"treatment": "IV Glucose GIR 8–12 (acute crisis)", "evidence": "Level A",
             "response_pct": 85,
             "note": "First intervention in acute catabolic crisis; stops BCAA catabolism → MMA surge halted; continue OHCbl IM + betaine throughout. Mandatory during intercurrent illness — never NPO without IV dextrose."},
            {"treatment": "Ammonia scavengers (Na benzoate / Na phenylacetate)", "evidence": "Level B",
             "response_pct": 65,
             "note": "NH3 >200 µmol/L; secondary via MMA-mediated NAGS/CPS1 inhibition. Same protocol as MMUT/PA."},
            {"treatment": "VPA — HIGH RISK / AVOID", "evidence": "AVOID",
             "response_pct": 0,
             "note": "HIGH RISK in all cblF patients (combined MMA+HHcy). VPA-CoA inhibits cobalamin metabolism + carnitine depletion. LEV is first-line AED for ALL seizure types in LMBRD1/cblF."},
        ],
        "patient_sample": sample,
    }


def get_definitions() -> dict:
    return {
        "gene_card": {
            "Gene": "LMBRD1 (LMBR1 Domain Containing 1; cblF gene product)",
            "Full Name": "Methylmalonic Aciduria and Homocystinuria, cblF type (LMBRD1; Complementation Group cblF)",
            "Chromosome": "6q13",
            "Inheritance": "Autosomal Recessive (AR)",
            "OMIM Gene": "*612634",
            "OMIM Disease": "#277380 (Methylmalonic Aciduria and Homocystinuria, cblF type)",
            "Protein Size": "540 amino acids; integral lysosomal membrane protein; 9 TM helices; LMBR1 domain",
            "Protein Function": "Lysosomal cobalamin transporter — exports free cobalamin from lysosomal lumen to cytoplasm; forms functional complex with ABCD4",
            "Pathway Position": "MOST UPSTREAM intracellular cobalamin processor — Step 1 after TC2/CD320-mediated endocytosis",
            "Prevalence": "~1:1,000,000 (very rare; <50 reported cases worldwide; likely underdiagnosed)",
        },
        "key_concepts": [
            {
                "concept": "LMBRD1 is the MOST UPSTREAM intracellular cobalamin transporter",
                "explanation": (
                    "After dietary cobalamin is endocytosed with transcobalamin II (TC2) via CD320 receptor, "
                    "it arrives in the lysosomal lumen where LMBRD1 must export it to the cytoplasm. "
                    "LMBRD1 is therefore the most upstream intracellular cobalamin handler — earlier than MMACHC, MMADHC, MMAB, MMAA, or MMUT. "
                    "LMBRD1 LOF blocks ALL downstream cobalamin processing simultaneously."
                ),
            },
            {
                "concept": "Cobalamin trapped in lysosomes — lysosomal storage feature distinguishes cblF from cblC",
                "explanation": (
                    "In cblC (MMACHC), cobalamin enters the cytoplasm normally but cannot be processed there. "
                    "In cblF (LMBRD1), cobalamin cannot LEAVE the lysosome → accumulates in lysosomal compartment. "
                    "This lysosomal accumulation produces vacuolated lymphocytes (lysosomal storage marker) "
                    "visible on blood smear in ~25–45% of cblF patients — ABSENT in cblC. "
                    "Electron microscopy of fibroblasts shows cobalamin in lysosomal compartment — diagnostic of cblF/cblJ."
                ),
            },
            {
                "concept": "Stomatitis / oral ulcers — pathognomonic cblF mucosal sign absent in cblC",
                "explanation": (
                    "Stomatitis and oral ulcers are a characteristic early clinical finding in cblF, "
                    "reported in the original case series and most subsequent cblF patients. "
                    "Mechanism: lysosomal cobalamin accumulation in rapidly dividing mucosal epithelial cells "
                    "may impair cell turnover. ABSENT in cblC (MMACHC). "
                    "Stomatitis + combined MMA+HHcy in an infant = cblF until proven otherwise; "
                    "gene panel (LMBRD1 + ABCD4) mandatory."
                ),
            },
            {
                "concept": "LMBRD1 + ABCD4 (cblJ) gene panel mandatory — both block lysosomal cobalamin export",
                "explanation": (
                    "LMBRD1 (cblF) and ABCD4 (cblJ) are the TWO genes known to block lysosomal cobalamin export. "
                    "LMBRD1 is the transporter pore; ABCD4 is the ATP-binding cassette ATPase that energizes export. "
                    "Clinically and biochemically IDENTICAL (combined MMA+HHcy; both MeCbl and AdoCbl absent; "
                    "lysosomal cobalamin on EM). Gene panel must test BOTH genes — cannot distinguish by biochemistry alone. "
                    "ABCD4 (cblJ) has been reported at 14q24.3."
                ),
            },
            {
                "concept": "OHCbl bypasses lysosomal trap via alternative cellular entry — partial response",
                "explanation": (
                    "IM OHCbl enters cells not only via CD320/lysosomal endocytosis but also via diffusion "
                    "and alternative uptake mechanisms. This partial alternative entry bypasses the LMBRD1 block, "
                    "delivering cob(I)alamin to MMACHC without lysosomal transit. "
                    "Response is ~50–70% HHcy, ~40–60% MMA — effective but slightly less than cblC (~75% HHcy) "
                    "because the bypass is less complete. Betaine MANDATORY to supplement the HHcy arm."
                ),
            },
            {
                "concept": "N2O absolute CI — same mechanism as cblC/MMACHC",
                "explanation": (
                    "Nitrous oxide (N2O) irreversibly oxidizes the cob(I)alamin cofactor → Methionine Synthase (MTR) "
                    "completely inactivated → acute severe HHcy crisis. "
                    "In cblF, methionine synthase is already compromised (MeCbl cannot be synthesized). "
                    "N2O exposure = methionine synthase failure → LIFE-THREATENING metabolic crisis. "
                    "Notify anesthesiology at every surgery: N2O ABSOLUTE CI. Use alternative anesthetics only."
                ),
            },
            {
                "concept": "Betaine MANDATORY — cobalamin-independent Hcy remethylation",
                "explanation": (
                    "Betaine activates BHMT (betaine-homocysteine methyltransferase), "
                    "which remethylates homocysteine to methionine using betaine as the methyl donor — "
                    "completely INDEPENDENT of cobalamin. This provides HHcy control "
                    "even when LMBRD1 block prevents cobalamin delivery to methionine synthase. "
                    "Betaine Level A alongside OHCbl for ALL cblF patients."
                ),
            },
            {
                "concept": "cblF onset vs cblC — typically later infantile, not neonatal",
                "explanation": (
                    "Most cblC (MMACHC) patients present neonatal (<1 month) or early infantile (<3 months). "
                    "cblF (LMBRD1) typically presents later: 2–18 months (Infantile Classic, modal 45%). "
                    "Later onset may reflect partial residual LMBRD1 function or lower MMA levels than cblC. "
                    "NBS C3 elevation still detects most cblF in neonatal period."
                ),
            },
        ],
        "diagnostic_thresholds": [
            {"parameter": "Urine MMA", "threshold": ">200 mmol/mol Cr",
             "action": "Urine organic acids + plasma tHcy simultaneously. If MMA elevated AND tHcy elevated → combined cobalamin disorder (cblC/cblF/cblJ/cblD-Combined). Gene panel immediately."},
            {"parameter": "Plasma tHcy", "threshold": ">30 µmol/L",
             "action": "Plasma amino acids (methionine) + urine MMA simultaneously. If tHcy elevated AND MMA elevated → combined block. If tHcy elevated alone (MMA normal) → MTR/MTRR/cblD-HHcy."},
            {"parameter": "Plasma methionine", "threshold": "<18 µmol/L (low)",
             "action": "Low methionine + high tHcy = methionine synthase pathway blocked. Start betaine + OHCbl immediately (Level A) while awaiting gene panel."},
            {"parameter": "NBS C3", "threshold": ">4 µmol/L",
             "action": "Reflex: plasma acylcarnitines + urine MMA + plasma tHcy + plasma amino acids. If combined MMA+HHcy → gene panel LMBRD1+ABCD4+MMACHC+MMADHC."},
            {"parameter": "Plasma NH3", "threshold": ">100 µmol/L",
             "action": "Secondary hyperammonemia via MMA-NAGS/CPS1 inhibition. IV glucose GIR 8–12 + OHCbl IM + betaine + ammonia scavengers if NH3 >200."},
            {"parameter": "Vacuolated lymphocytes", "threshold": "Any on blood smear",
             "action": "Present + combined MMA+HHcy = HIGH SUSPICION for cblF (LMBRD1) or cblJ (ABCD4). Gene panel LMBRD1+ABCD4 priority. EM fibroblasts for lysosomal cobalamin confirmation."},
            {"parameter": "Stomatitis + combined MMA+HHcy", "threshold": "Any stomatitis",
             "action": "Stomatitis in infant + combined MMA+HHcy = cblF until proven otherwise. Gene panel LMBRD1+ABCD4 priority over MMACHC if stomatitis present."},
            {"parameter": "OHCbl trial: tHcy", "threshold": ">50% reduction after 7 days",
             "action": "Confirms OHCbl responsiveness. Continue OHCbl lifelong IM. Add betaine for HHcy arm."},
        ],
        "differential_diagnosis": [
            {"disease": "cblC (MMACHC — 1p34.1)",
             "distinguishing": "Combined MMA+HHcy (same pattern as cblF). KEY DIFFERENCES: (1) vacuolated lymphocytes ABSENT in cblC — PRESENT in cblF (blood smear); (2) stomatitis ABSENT in cblC — PRESENT in cblF; (3) maculopathy more common in cblC (80%); (4) onset typically neonatal in cblC vs infantile in cblF. Gene panel confirms."},
            {"disease": "cblJ (ABCD4 — 14q24.3)",
             "distinguishing": "Biochemically IDENTICAL to cblF (combined MMA+HHcy; both MeCbl/AdoCbl absent; lysosomal cobalamin on EM). CANNOT be distinguished by biochemistry — gene panel (LMBRD1 + ABCD4) mandatory for both."},
            {"disease": "cblD-Combined (MMADHC — 2q23.2)",
             "distinguishing": "Combined MMA+HHcy (similar). KEY DIFFERENCES: (1) MMADHC is cytoplasmic (downstream of MMACHC); (2) no vacuolated lymphocytes in MMADHC; (3) no stomatitis; (4) NBS may miss cblD-HHcy-only subtype; (5) fibroblast cobalamin on EM normal in MMADHC (cytoplasmic block, not lysosomal). Gene panel confirms."},
            {"disease": "PA — Propionic Acidemia (PCCA/PCCB)",
             "distinguishing": "C3 elevated (same as cblF). KEY NEGATIVES in cblF vs PA: (1) tHcy normal in PA — ELEVATED in cblF; (2) methylcitrate PATHOGNOMONIC in PA (50–2,000) — only mildly elevated in cblF; (3) methylmalonic acid NORMAL in PA — ELEVATED in cblF."},
            {"disease": "Isolated MMA (MMUT/MMAA/MMAB)",
             "distinguishing": "MMA elevated (same as cblF). KEY NEGATIVES in cblF vs isolated MMA: (1) tHcy NORMAL in isolated MMA — ELEVATED in cblF; (2) methionine NORMAL in isolated MMA — LOW in cblF. The HHcy component distinguishes combined from isolated MMA."},
            {"disease": "CBS — Classical Homocystinuria (CBS — 21q22.3)",
             "distinguishing": "HHcy elevated (same as cblF). KEY NEGATIVES vs cblF: (1) MMA NORMAL in CBS — ELEVATED in cblF; (2) methionine HIGH in CBS (blocked transsulfuration) — LOW in cblF (blocked remethylation); (3) CBS onset later (childhood/adult); (4) marfanoid features in CBS absent in cblF."},
        ],
    }
