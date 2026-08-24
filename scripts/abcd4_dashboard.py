#!/usr/bin/env python3
"""ABCD4 (Methylmalonic Acidemia with Homocystinuria, cblJ type) Epilepsy Dashboard.

ABCD4 encodes a 606-aa lysosomal membrane ABC half-transporter that functions as the
ATPase MOTOR powering cobalamin export from the lysosomal lumen.

COBALAMIN PATHWAY — ABCD4 IS THE ATPase MOTOR OF STEP 1 (ALONGSIDE LMBRD1):
  Step 0: Dietary cobalamin bound to transcobalamin II (TC2) → TC2-receptor (CD320) →
          receptor-mediated endocytosis → lysosome (cobalamin trapped in lysosome).
  Step 1: ABCD4 (cblJ) + LMBRD1 (cblF) — together export cobalamin from lysosomal lumen → cytoplasm.
          LMBRD1 provides the TRANSPORTER PORE (9 TM helices); ABCD4 provides the ATPase MOTOR (NBD domain).
          ABCD4 hydrolyzes ATP → conformational change in LMBRD1 pore → cobalamin exported.
          ← BLOCK IN cblJ: ABCD4 LOF → LMBRD1 has no ATPase partner → cannot export cobalamin.
  Step 2: MMACHC (cblC; 1p34.1) — converts cobalamin to cob(I)alamin in cytoplasm.
          (MMACHC receives NO substrate when ABCD4/LMBRD1 complex is defective.)
  Step 3–4: Same as cblF — both MMA and HHcy arms blocked simultaneously.

ABCD4 LOF RESULT — BOTH DOWNSTREAM ARMS BLOCKED SIMULTANEOUSLY:
  → MMA ELEVATED (150–1,200 mmol/mol Cr): mitochondrial arm starved of AdoCbl.
  → tHcy ELEVATED (25–180 µmol/L): cytoplasmic arm starved of MeCbl.
  → Methionine LOW (8–22 µmol/L): Methionine Synthase (MTR) inactive.
  → MeCbl fibroblasts: ABSENT.
  → AdoCbl fibroblasts: ABSENT.
  → Biochemically: COMBINED MMA + HHcy — IDENTICAL to cblC/MMACHC and cblF/LMBRD1.

KEY DIFFERENCE: ABCD4 (cblJ) vs LMBRD1 (cblF):
  1. NO vacuolated lymphocytes in cblJ (blood smear NORMAL) — ABSENT unlike cblF where present ~25–45%.
     ABCD4 is the ATPase motor; cobalamin accumulation pattern in lysosome may differ from cblF pore defect.
  2. NO stomatitis in cblJ — oral mucosa normal. Stomatitis is cblF-specific (LMBRD1 pore defect).
  3. Cannot be distinguished from cblF (LMBRD1) by ANY biochemical test — gene panel MANDATORY.
  4. ABCD4 is unique among ABCD transporters (ABCD1/ABCD2/ABCD3 are peroxisomal; ABCD4 is LYSOSOMAL).
  5. MMA levels typically slightly lower in cblJ vs cblF (150–1,200 vs 200–1,500 mmol/mol Cr).

ABCD4 PROTEIN BIOLOGY:
  606 amino acids; lysosomal membrane ABC half-transporter; 6 TM helices + C-terminal NBD (nucleotide-binding domain).
  NBD contains Walker A (GKT motif; ATP binding), Walker B (DExx; ATP hydrolysis), LSGGQ signature, D-loop.
  ABCD4 forms functional heterodimer complex with LMBRD1 at lysosomal membrane.
  ATP hydrolysis by ABCD4 NBD powers cobalamin translocation through LMBRD1 pore.
  Unlike ABCD1 (peroxisomal VLCFA transporter), ABCD4 localizes to LYSOSOMAL membrane — unique ABCD member.
  ABCD4 N-terminal region interacts with LMBRD1 for complex assembly.

ABCD4 BIOMARKERS (identical to cblF/LMBRD1):
  C3 (propionylcarnitine): 3–18 µmol/L↑ (NBS PRIMARY)
  Urine MMA: 150–1,200 mmol/mol Cr↑ (slightly lower than cblF 200–1,500; LOWER than MMUT)
  tHcy: 25–180 µmol/L↑ (PATHOGNOMONIC combined pattern with MMA)
  Methionine: 8–22 µmol/L LOW (methionine synthase inactive)
  MeCbl fibroblasts: ABSENT
  AdoCbl fibroblasts: ABSENT
  Vacuolated lymphocytes: ABSENT in cblJ (KEY NEGATIVE vs cblF)
  Stomatitis: ABSENT in cblJ (KEY NEGATIVE vs cblF)

KEY VARIANTS IN ABCD4:
  p.Arg432Gln (c.1295G>A): Walker A NBD; most commonly reported allele; moderate;
    Arg432 is in the GKT Walker A ATP-binding motif; disrupts ATP binding → no ATPase activity.
  p.Tyr319Cys (c.956A>G): TM domain 6; moderate severity; infantile classic.
  c.IVS11+1G>A: canonical splice site null; severe; early infantile.
  p.Trp479Arg (c.1435T>C): NBD domain; severe; unresponsive to OHCbl.
  p.Arg519Gln (c.1556G>A): Walker B DExx motif; moderate-severe; disrupts ATP hydrolysis.
  p.Ala350Val: TM domain 6; moderate; infantile classic.

TREATMENT (identical to cblF/LMBRD1):
  OHCbl 1–2 mg/day IM (Level A): alternative cellular entry bypasses lysosomal trap;
    ~50–65% HHcy response; ~35–55% MMA response; slightly lower than cblF due to ATPase-deficient complex.
  Betaine (TMG) 100–200 mg/kg/day (Level A): MANDATORY for HHcy; cobalamin-independent BHMT pathway.
  Protein restriction + MMA-free formula (Level A): Ile/Val/Met/Thr restriction; MMA component.
  L-Carnitine 100 mg/kg/day (Level A–B): secondary depletion via propionylcarnitine excretion.
  Folinic acid 0.5–5 mg/day (Level B): 5-methylTHF depleted; methylation support.
  IV Glucose GIR 8–12 (Level A): acute MMA crisis; stop BCAA catabolism.

CONTRAINDICATIONS (identical to cblF/LMBRD1):
  N2O — ABSOLUTE CI: inactivates methionine synthase (already compromised in cblJ). LIFE-THREATENING.
  VPA — HIGH RISK: carnitine depletion + metabolic stress → MMA surge; LEV first-line.
  Fasting — HIGH RISK: BCAA catabolism → MMA surge + HHcy worsens.

OMIM: Gene *603214 · Disease #614857 (Methylmalonic Aciduria and Homocystinuria, cblJ type)
Chromosome: 14q24.3 · Inheritance: AR · Prevalence: ~1:2,000,000 (ultra-rare; <25 reported cases 2026)
"""

from __future__ import annotations
import random

random.seed(77)   # reproducible synthetic cohort — seed 77 for cblJ/ABCD4


# ── helpers ───────────────────────────────────────────────────────────────────

def _pid(i: int) -> str:
    return f"ABCD4-{i:03d}"


def _sex(i: int) -> str:
    return "M" if i % 2 == 0 else "F"


def _phenotype(i: int) -> str:
    """
    Early Infantile Severe (~35%)    — <6 months; highest MMA+HHcy; no vacuolated lymphocytes; no stomatitis
    Infantile Classic (~50%)         — 2–24 months; modal; combined MMA+HHcy; OHCbl+betaine partially responsive
    Late Infantile Attenuated (~15%) — 12–36 months; partial residual ABCD4 ATPase; milder
    """
    if i < 14:
        return "Early Infantile Severe"
    elif i < 34:
        return "Infantile Classic"
    else:
        return "Late Infantile Attenuated"


def _genotype(i: int, pheno: str) -> str:
    pheno_variants = {
        "Early Infantile Severe": [
            "c.IVS11+1G>A/c.IVS11+1G>A (homozygous splice null; severe)",
            "p.Trp479Arg/c.IVS11+1G>A (NBD/null compound het; severe)",
            "p.Trp479Arg/p.Trp479Arg (homozygous NBD null; non-responsive)",
            "c.IVS11+1G>A/p.Arg519Gln (null/Walker-B compound het; severe)",
        ],
        "Infantile Classic": [
            "p.Arg432Gln/p.Arg432Gln (homozygous Walker-A; most common; moderate)",
            "p.Arg432Gln/p.Tyr319Cys (Walker-A/TM6 compound het; moderate)",
            "p.Tyr319Cys/p.Tyr319Cys (homozygous TM6; moderate)",
            "p.Arg432Gln/p.Ala350Val (Walker-A/TM compound het; moderate)",
            "p.Arg519Gln/p.Arg432Gln (Walker-B/Walker-A compound het; moderate-severe)",
            "p.Ala350Val/p.Tyr319Cys (TM/TM compound het; moderate)",
        ],
        "Late Infantile Attenuated": [
            "p.Arg432Gln/p.Ala350Val (Walker-A/TM; attenuated; partial ATPase)",
            "p.Tyr319Cys/p.Ala350Val (TM/TM; attenuated; partial pore interaction)",
            "p.Arg432Gln/p.Arg432Gln (homozygous Walker-A; attenuated variant branch)",
        ],
    }
    opts = pheno_variants[pheno]
    return opts[i % len(opts)]


def _mma_urine(pheno: str) -> float:
    if pheno == "Early Infantile Severe":
        return round(random.uniform(350, 1200), 0)
    elif pheno == "Infantile Classic":
        return round(random.uniform(150, 700), 0)
    else:
        return round(random.uniform(80, 350), 0)


def _homocysteine(pheno: str) -> float:
    if pheno == "Early Infantile Severe":
        return round(random.uniform(55, 180), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(25, 110), 1)
    else:
        return round(random.uniform(18, 70), 1)


def _methionine(pheno: str) -> float:
    if pheno == "Early Infantile Severe":
        return round(random.uniform(8, 15), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(10, 20), 1)
    else:
        return round(random.uniform(14, 24), 1)


def _c3(pheno: str) -> float:
    if pheno == "Early Infantile Severe":
        return round(random.uniform(5, 18), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(3, 11), 1)
    else:
        return round(random.uniform(2, 6), 1)


def _ammonia(pheno: str) -> int:
    if pheno == "Early Infantile Severe":
        return random.randint(80, 400)
    elif pheno == "Infantile Classic":
        return random.randint(35, 110)
    else:
        return random.randint(18, 55)


def _carnitine(pheno: str) -> float:
    if pheno == "Early Infantile Severe":
        return round(random.uniform(9, 24), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(13, 32), 1)
    else:
        return round(random.uniform(20, 42), 1)


def _onset_months(pheno: str) -> int:
    if pheno == "Early Infantile Severe":
        return random.randint(0, 5)
    elif pheno == "Infantile Classic":
        return random.randint(2, 24)
    else:
        return random.randint(12, 36)


def _ohcbl_response(pheno: str) -> bool:
    rates = {"Early Infantile Severe": 0.50, "Infantile Classic": 0.60, "Late Infantile Attenuated": 0.75}
    return random.random() < rates[pheno]


def _seizures(pheno: str) -> bool:
    rates = {"Early Infantile Severe": 0.78, "Infantile Classic": 0.62, "Late Infantile Attenuated": 0.38}
    return random.random() < rates[pheno]


def _nbs(pheno: str) -> bool:
    # C3 elevated — cblJ detected similarly to cblF
    rates = {"Early Infantile Severe": 0.75, "Infantile Classic": 0.65, "Late Infantile Attenuated": 0.50}
    return random.random() < rates[pheno]


def _aed(sz: bool) -> str:
    if not sz:
        return "None"
    opts = ["LEV", "LEV + CLB", "LEV + ACTH (IS)", "LEV + CLN", "LEV + PB (neonate)"]
    return random.choice(opts)


def _diag_months(pheno: str, onset: int) -> int:
    delays = {"Early Infantile Severe": (1, 7), "Infantile Classic": (2, 14), "Late Infantile Attenuated": (4, 20)}
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
            "mecbl_fibroblasts": "Absent",    # absent in all cblJ
            "adocbl_fibroblasts": "Absent",   # absent in all cblJ
            "vacuolated_lymphocytes": False,  # ABSENT in cblJ (KEY NEGATIVE vs cblF)
            "stomatitis": False,              # ABSENT in cblJ (KEY NEGATIVE vs cblF)
            "seizures": sz,
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
        "gene": "ABCD4",
        "full_name": "ATP-Binding Cassette Subfamily D Member 4 — Lysosomal Cobalamin ATPase Motor (cblJ gene product)",
        "chromosome": "14q24.3",
        "inheritance": "AR",
        "omim_gene": "*603214",
        "omim_disease": "#614857",
        "protein_size": "606 aa; lysosomal membrane ABC half-transporter; 6 TM helices + NBD; Walker A/B ATPase motifs; heterodimer with LMBRD1",
        "prevalence": "~1:2,000,000 (ultra-rare; <25 reported cases worldwide 2026; likely underdiagnosed)",
        "nbs_primary": "C3 (propionylcarnitine) elevated — cblJ detected by NBS in ~65% (all have combined MMA+HHcy → C3 elevated); NBS misses ~35% (lower C3 in attenuated subtypes)",
        "nbs_secondary": "Urine MMA + plasma tHcy simultaneously mandatory; gene panel (ABCD4 + LMBRD1) distinguishes cblJ from cblF; cannot distinguish by biochemistry alone",
        "function": (
            "ABCD4 (cblJ) is the ATPase MOTOR of the lysosomal cobalamin export complex. "
            "It forms a functional heterodimer with LMBRD1 (cblF) at the lysosomal membrane. "
            "ABCD4's NBD domain (Walker A/B motifs) hydrolyzes ATP → conformational change → "
            "powers cobalamin translocation through the LMBRD1 transporter pore from lysosomal lumen to cytoplasm. "
            "ABCD4 LOF → LMBRD1 pore cannot operate without its ATPase driver → cobalamin TRAPPED in lysosomes → "
            "MMACHC receives NO substrate → BOTH metabolic arms blocked → Combined MMA + HHcy."
        ),
        "mechanism": (
            "ABCD4 LOF silences the ATPase motor of the ABCD4-LMBRD1 lysosomal export complex. "
            "Without ABCD4 ATPase activity, LMBRD1 cannot translocate cobalamin across the lysosomal membrane. "
            "Cobalamin accumulates in the lysosomal lumen — lysosomal cobalamin visible on fibroblast EM. "
            "Result: Combined MMA + HHcy + LOW methionine — biochemically IDENTICAL to cblC/MMACHC and cblF/LMBRD1. "
            "KEY DISTINCTION: unlike cblF, NO vacuolated lymphocytes and NO stomatitis in cblJ. "
            "Gene panel (ABCD4 + LMBRD1) is the ONLY way to distinguish cblJ from cblF."
        ),
        "key_negative": (
            "Vacuolated lymphocytes: ABSENT in cblJ — KEY NEGATIVE vs cblF (LMBRD1); "
            "Stomatitis: ABSENT in cblJ — KEY NEGATIVE vs cblF; "
            "BCAA NORMAL — KEY NEGATIVE vs DLD/MSUD; "
            "Methylcitrate only mildly elevated (secondary) — KEY NEGATIVE vs PA; "
            "Biotinidase NORMAL — KEY NEGATIVE vs BTD; "
            "Biochemically identical to cblF, cblC, cblD-Combined — gene panel mandatory."
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
            "vacuolated_lymphocyte_pct": 0,   # ABSENT in ALL cblJ
            "stomatitis_pct": 0,              # ABSENT in ALL cblJ
            "nbs_detected_pct": pct(lambda p: p["nbs_detected"]),
            "ohcbl_response_pct": pct(lambda p: p["ohcbl_response"]),
            "early_infantile_pct": round(len(early_pts) / n * 100),
            "infantile_classic_pct": round(len(class_pts) / n * 100),
            "late_infantile_pct": round(len(late_pts) / n * 100),
        },
        "phenotype_distribution": [
            {"phenotype": "Early Infantile Severe (~35%)",
             "pct": 35, "onset": "0–6 months", "vacuolated_lymphocytes": "0% (ABSENT)",
             "stomatitis": "0% (ABSENT)",
             "mma": "350–1,200 mmol/mol Cr", "tHcy": "55–180 µmol/L",
             "note": "Neonatal/early infantile onset; highest MMA+HHcy; NO vacuolated lymphocytes (unlike cblF); NO stomatitis; NH3 80–400; combined metabolic crisis; null/severe NBD alleles"},
            {"phenotype": "Infantile Classic (~50%)",
             "pct": 50, "onset": "2–24 months", "vacuolated_lymphocytes": "0% (ABSENT)",
             "stomatitis": "0% (ABSENT)",
             "mma": "150–700 mmol/mol Cr", "tHcy": "25–110 µmol/L",
             "note": "Modal phenotype; combined MMA+HHcy; blood smear NORMAL (no vacuolated lymphocytes); OHCbl + betaine ~60% responsive; p.Arg432Gln most common allele (Walker A NBD)"},
            {"phenotype": "Late Infantile Attenuated (~15%)",
             "pct": 15, "onset": "12–36 months", "vacuolated_lymphocytes": "0% (ABSENT)",
             "stomatitis": "0% (ABSENT)",
             "mma": "80–350 mmol/mol Cr", "tHcy": "18–70 µmol/L",
             "note": "Later onset; milder MMA and HHcy; partial residual ABCD4 ATPase; OHCbl + betaine highly effective; better outcome; partial Walker A/TM domain variants"},
        ],
        "abcd4_pathway": [
            {
                "step": "Step 0 — Dietary cobalamin + TC2 → CD320 receptor → endocytosis → lysosome",
                "reaction": "Cobalamin-TC2 binds CD320 → clathrin endocytosis → lysosomal lumen; TC2 proteolysed; free cobalamin in lysosome",
                "enzyme": "CD320 (transcobalamin receptor); not ABCD4-dependent",
                "consequence_lof": "Not applicable — this step is TC2/CD320-dependent, not ABCD4",
            },
            {
                "step": "Step 1 — ABCD4 (cblJ) + LMBRD1 (cblF): lysosomal cobalamin → cytoplasm ← BLOCK IN cblJ",
                "reaction": "ABCD4 NBD hydrolyzes ATP → conformational change → LMBRD1 pore opens → cobalamin exported from lysosome to cytoplasm",
                "enzyme": "ABCD4 (cblJ; 606 aa; 14q24.3; lysosomal membrane; NBD Walker A/B ATPase) + LMBRD1 (cblF; 6q13; transporter pore)",
                "consequence_lof": "ABCD4 ATPase absent → LMBRD1 pore cannot open → cobalamin TRAPPED in lysosomes; MMACHC receives no substrate; both downstream arms blocked",
            },
            {
                "step": "Step 2 — MMACHC (cblC): cytoplasmic cobalamin → cob(I)alamin [receives NO substrate in cblJ]",
                "reaction": "Cobalamin + NADPH → cob(I)alamin [MMACHC decyanase/reductase]",
                "enzyme": "MMACHC (cblC; 282 aa; 1p34.1; cytoplasmic)",
                "consequence_lof": "Cannot proceed — no cobalamin delivered from ABCD4-LMBRD1 complex in cblJ",
            },
            {
                "step": "Step 3 — MMADHC (cblD): distributes cob(I)alamin to ARM A (mitochondrial) + ARM B (cytoplasmic)",
                "reaction": "cob(I)alamin → ARM A [MMAB → AdoCbl] + ARM B [MeCbl → MTR]",
                "enzyme": "MMADHC (cblD; 296 aa; 2q23.2)",
                "consequence_lof": "Cannot proceed — no cob(I)alamin from MMACHC in cblJ",
            },
            {
                "step": "Step 4A — MMAB → AdoCbl → MMUT → MMA catabolism [ARM A; BLOCKED in cblJ]",
                "reaction": "cob(I)alamin + ATP → AdoCbl → L-methylmalonyl-CoA → succinyl-CoA",
                "enzyme": "MMAB (cblB; 12q24.11) + MMAA (cblA; 4p14) + MMUT (6p12.3)",
                "consequence_lof": "MMA ACCUMULATES (150–1,200 mmol/mol Cr); C3 elevated on NBS",
            },
            {
                "step": "Step 4B — MeCbl → MTR → Hcy remethylation [ARM B; BLOCKED in cblJ]",
                "reaction": "cob(I)alamin → MeCbl → MTR: Hcy + 5-methylTHF → Methionine + THF",
                "enzyme": "MTR (methionine synthase; 1q43) + MTRR (cblE; 5p15.3)",
                "consequence_lof": "tHcy ELEVATED; Methionine LOW; same mechanism as cblC/cblF combined block",
            },
        ],
        "high_risk_situations": [
            {"situation": "Nitrous oxide (N2O)",
             "risk": "ABSOLUTE CI",
             "detail": "N2O irreversibly oxidizes cob(I)alamin → cannot synthesize MeCbl → Methionine Synthase (MTR) completely inactivated → acute HHcy crisis + metabolic catastrophe. LIFE-THREATENING. Same absolute CI as cblC/cblF. Notify anesthesiology at every surgery."},
            {"situation": "VPA (Valproate)",
             "risk": "HIGH RISK",
             "detail": "VPA-CoA inhibits mitochondrial cobalamin metabolism + carnitine depletion + MMA surge. HIGH RISK in combined MMA+HHcy (cblJ). Use LEV first-line for ALL seizure types in ABCD4/cblJ."},
            {"situation": "Fasting / catabolism",
             "risk": "HIGH RISK",
             "detail": "BCAA catabolism → propionyl-CoA → MMA surge; catabolism worsens HHcy remethylation deficit. IV glucose + OHCbl IM throughout illness. Never NPO without IV dextrose."},
            {"situation": "Missed OHCbl + betaine",
             "risk": "HIGH RISK",
             "detail": "tHcy rebounds within 48–72 h; MMA rebounds within 3–7 days. Both medications mandatory simultaneously — neither alone is sufficient. Same protocol as cblF/LMBRD1."},
            {"situation": "Antifolates (methotrexate / trimethoprim)",
             "risk": "HIGH RISK",
             "detail": "Further deplete 5-methylTHF → worsen methionine synthase pathway failure → acute HHcy elevation. Folinic acid supplementation and monitoring essential."},
        ],
    }


def get_breakdown() -> dict:
    sample = _PATIENTS[:15]
    return {
        "biomarkers": [
            {"name": "C3 (Propionylcarnitine)", "normal": "<4 µmol/L",
             "abcd4_range": "3–18 µmol/L ELEVATED (all cblJ subtypes: combined MMA+HHcy → C3 elevated)",
             "significance": "NBS PRIMARY TRIGGER — detected in ~65% of cblJ on standard NBS. Slightly lower C3 than cblF due to somewhat lower MMA. Must be followed by urine MMA + tHcy simultaneously. Gene panel (ABCD4+LMBRD1) distinguishes from cblF.",
             "method": "NBS tandem MS/MS; plasma acylcarnitines"},
            {"name": "Methylmalonic Acid (urine)", "normal": "<5 mmol/mol Cr",
             "abcd4_range": "150–1,200 mmol/mol Cr (slightly lower than cblF 200–1,500; LOWER than MMUT 200–10,000)",
             "significance": "ELEVATED in ALL cblJ patients (both downstream arms blocked). MMA + tHcy elevated simultaneously = combined cobalamin disorder (cblC/cblF/cblJ/cblD-Combined). Gene panel confirms cblJ vs others.",
             "method": "Urine organic acids (GC-MS)"},
            {"name": "Total Homocysteine (tHcy)", "normal": "<15 µmol/L",
             "abcd4_range": "25–180 µmol/L ELEVATED (ALL cblJ patients — cytoplasmic arm completely blocked)",
             "significance": "PATHOGNOMONIC for combined block when elevated WITH MMA. tHcy + MMA together = combined cobalamin disorder (cblC, cblD-Combined, cblF, cblJ). Draw SIMULTANEOUSLY with MMA — do not separate in time.",
             "method": "Plasma tHcy (HPLC or immunoassay); simultaneous with MMA"},
            {"name": "Methionine (plasma)", "normal": "20–40 µmol/L",
             "abcd4_range": "8–22 µmol/L LOW (ALL cblJ — methionine synthase MTR inactive: no MeCbl)",
             "significance": "LOW methionine confirms methionine synthase pathway blocked. Low methionine + high tHcy + high MMA = lysosomal cobalamin processing disorder (cblF or cblJ). Gene panel distinguishes.",
             "method": "Plasma amino acids (quantitative)"},
            {"name": "MeCbl (fibroblasts)", "normal": "Present",
             "abcd4_range": "ABSENT in ALL cblJ patients (cytoplasmic arm receives no substrate)",
             "significance": "ABSENT — confirms cytoplasmic arm blockade. Both MeCbl and AdoCbl absent (same as cblC and cblF) — distinguishes from MMADHC subtypes where only one cobalamin arm is absent.",
             "method": "Fibroblast cobalamin distribution assay (specialized metabolic lab)"},
            {"name": "AdoCbl (fibroblasts)", "normal": "Present",
             "abcd4_range": "ABSENT in ALL cblJ patients (mitochondrial arm receives no substrate)",
             "significance": "ABSENT — confirms mitochondrial arm blockade. Both MeCbl and AdoCbl absent simultaneously = combined lysosomal block (cblJ/ABCD4 or cblF/LMBRD1) OR cytoplasmic block (cblC/MMACHC).",
             "method": "Fibroblast cobalamin distribution assay"},
            {"name": "Vacuolated Lymphocytes", "normal": "Absent",
             "abcd4_range": "ABSENT in cblJ (0% of ABCD4 patients — KEY NEGATIVE vs cblF)",
             "significance": "KEY NEGATIVE for cblJ: blood smear shows NORMAL lymphocytes. Unlike cblF (LMBRD1) where vacuolated lymphocytes appear in ~25–45% (lysosomal storage). Absent vacuolated lymphocytes + combined MMA+HHcy → still need gene panel; could be cblJ, cblC, or cblD-Combined.",
             "method": "Peripheral blood smear microscopy (Wright-Giemsa stain)"},
            {"name": "Stomatitis / Oral Ulcers", "normal": "Absent",
             "abcd4_range": "ABSENT in cblJ (0% of ABCD4 patients — KEY NEGATIVE vs cblF)",
             "significance": "KEY NEGATIVE for cblJ: oral mucosa normal. Stomatitis is specific to cblF (LMBRD1 pore defect causing lysosomal accumulation in mucosal epithelium). Absent stomatitis does NOT rule out cblJ — gene panel required.",
             "method": "Clinical examination; oral mucosa inspection"},
            {"name": "Lysosomal cobalamin (EM)", "normal": "Absent",
             "abcd4_range": "Present in lysosomal compartment (fibroblast electron microscopy) — same as cblF",
             "significance": "Cobalamin visible in lysosomal compartment on fibroblast EM — diagnostic of lysosomal export defect (cblJ or cblF). EM CANNOT distinguish cblJ from cblF — gene panel mandatory. Used to confirm lysosomal block vs cytoplasmic block (cblC).",
             "method": "Fibroblast electron microscopy + cobalamin histochemistry (specialized)"},
            {"name": "Methylcitrate (urine)", "normal": "<5 µmol/mmol Cr",
             "abcd4_range": "4–70 µmol/mmol Cr (mildly elevated; secondary; NOT pathognomonic)",
             "significance": "Mildly elevated secondary to MMA accumulation. KEY NEGATIVE vs propionic acidemia (PCCA/PCCB) where methylcitrate is PATHOGNOMONIC (50–2,000). Low methylcitrate + high MMA + high HHcy = cobalamin disorder (not PA).",
             "method": "Urine organic acids (GC-MS)"},
            {"name": "Ammonia (plasma)", "normal": "<50 µmol/L",
             "abcd4_range": "Early Infantile Severe: 80–400 µmol/L; Infantile Classic: 35–110 µmol/L",
             "significance": "Secondary hyperammonemia via MMA-mediated NAGS/CPS1 inhibition. Treat with ammonia scavengers if NH3 >200. Slightly lower than MMUT because MMA levels somewhat lower in cblJ.",
             "method": "Plasma ammonia (enzymatic colorimetric)"},
            {"name": "OHCbl Trial Response (tHcy)", "normal": "N/A",
             "abcd4_range": "~50–65% HHcy response (IM OHCbl bypasses lysosomal trap via alternative cell entry)",
             "significance": "OHCbl given IM enters cells via alternative routes (diffusion/other) that partially bypass the ABCD4-LMBRD1 lysosomal block. Slightly lower response than cblC (~75%) because bypass of both ATPase+pore is less efficient. Measure Day 0 vs Day 7 tHcy.",
             "method": "tHcy before/after 5–7 day OHCbl 1–2 mg/day IM trial"},
            {"name": "OHCbl Trial Response (MMA)", "normal": "N/A",
             "abcd4_range": "~35–55% MMA response (comparable to cblF; partial bypass)",
             "significance": "Partially bypasses ABCD4-LMBRD1 complex for MMA catabolism arm. Response similar to cblF (LMBRD1; ~40–60%). Betaine MANDATORY for HHcy arm regardless of OHCbl response.",
             "method": "Urine MMA before/after OHCbl trial"},
        ],
        "key_variants": [
            {"variant": "p.Arg432Gln", "cdna": "c.1295G>A", "domain": "Walker A NBD (GKT motif; ATP binding)",
             "severity": "Moderate (disrupts ATP binding; partial residual ATPase)", "phenotype": "Infantile Classic",
             "note": "Most commonly reported ABCD4 allele; Walker A motif Arg432 directly coordinates ATP phosphate groups; Gln substitution impairs ATP binding but retains partial ATPase → reduced but not absent cobalamin export; infantile classic onset 2–12 months; combined MMA+HHcy moderate; OHCbl + betaine responsive ~65%"},
            {"variant": "p.Tyr319Cys", "cdna": "c.956A>G", "domain": "TM domain 6",
             "severity": "Moderate (TM folding defect; partial LMBRD1 interaction)", "phenotype": "Infantile Classic",
             "note": "TM domain 6 missense; Cys may form aberrant disulfide or alter TM packing → impairs ABCD4-LMBRD1 heterodimer assembly; infantile classic; combined MMA+HHcy moderate; OHCbl + betaine response ~60%"},
            {"variant": "c.IVS11+1G>A", "cdna": "c.IVS11+1G>A (canonical splice site)", "domain": "Global (exon 11 skipped → NMD)",
             "severity": "Severe (null; exon 11 skipping → frameshift or NMD)", "phenotype": "Early Infantile Severe",
             "note": "Canonical +1 splice site; exon 11 skipping → frameshifted NBD → NMD → absent ABCD4 protein; biallelic null → worst cblJ phenotype; early infantile <3 months; severe MMA+HHcy; NH3 150–400; aggressive OHCbl + betaine needed"},
            {"variant": "p.Trp479Arg", "cdna": "c.1435T>C", "domain": "NBD domain (hydrophobic core)",
             "severity": "Severe (NBD misfolding; protein instability; non-responsive)", "phenotype": "Early Infantile Severe",
             "note": "NBD core Trp→Arg substitution → hydrophobic collapse → unstable ABCD4 NBD → no ATPase activity; non-responsive to OHCbl; biallelic = severe neonatal phenotype; earliest onset cases; high NH3; aggressive metabolic management"},
            {"variant": "p.Arg519Gln", "cdna": "c.1556G>A", "domain": "Walker B motif (DExx; ATP hydrolysis)",
             "severity": "Moderate-Severe (disrupts ATP hydrolysis; ATPase rate severely reduced)", "phenotype": "Infantile Classic to Early Severe",
             "note": "Walker B DExx motif Arg519 involved in ATP hydrolysis coordination; Gln substitution impairs hydrolysis step → binds ATP but cannot cleave → LMBRD1 pore arrested mid-cycle; moderate-severe phenotype; partial OHCbl response ~50%"},
            {"variant": "p.Ala350Val", "cdna": "c.1049C>T", "domain": "TM domain 6/7 boundary",
             "severity": "Mild-Moderate (partial TM packing; reduced ABCD4-LMBRD1 interaction)", "phenotype": "Infantile Classic / Late Infantile Attenuated",
             "note": "TM domain boundary; mild disruption of ABCD4-LMBRD1 heterodimer; partial residual ATPase activity; infantile classic to attenuated onset; best OHCbl+betaine response (~70–75% HHcy); best prognosis among cblJ variants"},
        ],
        "seizure_types": [
            {"type": "Early infantile / neonatal encephalopathic seizures (combined MMA+HHcy crisis)", "pct": 68,
             "note": "Multifocal; driven by simultaneous MMA mitochondrial toxicity + HHcy endothelial damage; treat metabolic crisis first: OHCbl IM + betaine + IV glucose; LEV adjunct; AVOID VPA; AVOID N2O anesthesia"},
            {"type": "Infantile spasms (West syndrome)", "pct": 30,
             "note": "Early-onset subtypes; hypsarrhythmia on EEG; ACTH or high-dose prednisolone preferred; folinic acid adjunct (methylation support); metabolic control reduces IS burden; LEV first-line"},
            {"type": "Focal cortical seizures", "pct": 26,
             "note": "Cortical injury from combined MMA+HHcy toxicity; LEV first-line; improve with metabolic compliance"},
            {"type": "Myoclonic seizures", "pct": 20,
             "note": "Metabolic-triggered by MMA/HHcy spikes; correlate seizure clusters with metabolic compliance; LEV first-line"},
            {"type": "Epileptic status / NCSE", "pct": 10,
             "note": "Early severe subtypes; IV BZD + IV glucose + OHCbl IM + betaine simultaneous; continuous EEG monitoring"},
        ],
        "metabolic_triggers": [
            {"trigger": "Nitrous oxide (N2O) — any exposure", "pct": 100,
             "mechanism": "ABSOLUTE CI: N2O irreversibly oxidizes cob(I)alamin → Methionine Synthase (MTR) inactivated → acute HHcy crisis + metabolic catastrophe. ALL cblJ patients at risk. Notify anesthesiology at every surgery."},
            {"trigger": "Fasting / prolonged NPO", "pct": 70,
             "mechanism": "BCAA catabolism → propionyl-CoA → MMA surge; catabolism worsens HHcy deficit. IV glucose GIR 8–12 mandatory whenever NPO or illness. Emergency protocol."},
            {"trigger": "Intercurrent illness / fever", "pct": 60,
             "mechanism": "Catabolic crisis; MMA surge + HHcy worsens; IV glucose + OHCbl IM throughout; betaine maintained."},
            {"trigger": "Missed OHCbl + betaine", "pct": 45,
             "mechanism": "tHcy rebounds within 48–72 h; MMA rebounds within 3–7 days. Both medications mandatory simultaneously."},
            {"trigger": "High protein intake (Ile/Val/Met/Thr excess)", "pct": 30,
             "mechanism": "Excess propionyl-CoA precursors → MMA surge; protein restriction mandatory."},
        ],
        "high_risk_drugs": [
            {"drug": "Nitrous oxide (N2O)", "risk": "ABSOLUTE CI",
             "mechanism": "Irreversibly oxidizes cobalamin cofactor → Methionine Synthase (MTR) failure → acute HHcy crisis. LIFE-THREATENING in all cblJ patients. Same absolute CI as cblC/cblF."},
            {"drug": "Valproate (VPA)", "risk": "HIGH RISK",
             "mechanism": "VPA-CoA inhibits mitochondrial cobalamin metabolism + carnitine depletion + MMA surge. Use LEV first-line for ALL seizure types in ABCD4/cblJ."},
            {"drug": "Antifolates (methotrexate, trimethoprim)", "risk": "HIGH RISK",
             "mechanism": "Further deplete 5-methylTHF → worsen methionine synthase pathway failure → acute HHcy elevation. Folinic acid monitoring essential."},
            {"drug": "Fasting / NPO without IV glucose", "risk": "HIGH RISK",
             "mechanism": "Catabolism-driven MMA surge + HHcy worsening. IV dextrose + OHCbl mandatory throughout any illness."},
        ],
        "treatments": [
            {"treatment": "Hydroxocobalamin (OHCbl) 1–2 mg/day IM", "evidence": "Level A",
             "response_pct": 58,
             "note": "Trial ALL cblJ patients. IM OHCbl enters cells via alternative routes (not only CD320/lysosomal) → partially bypasses ABCD4-LMBRD1 block → provides cob(I)alamin to MMACHC. ~50–65% HHcy response; ~35–55% MMA response. Lifelong IM preferred; monitor tHcy + MMA Day 7."},
            {"treatment": "Betaine (trimethylglycine / TMG) 100–200 mg/kg/day", "evidence": "Level A",
             "response_pct": 78,
             "note": "MANDATORY for HHcy component in ALL cblJ patients. Betaine-homocysteine methyltransferase (BHMT) remethylates Hcy to Methionine — cobalamin-INDEPENDENT pathway. Same mechanism/evidence as cblC/cblF. Level A."},
            {"treatment": "Protein restriction + MMA-free formula", "evidence": "Level A",
             "response_pct": 70,
             "note": "Low Ile/Val/Met/Thr for MMA component. MMA-free formula Level A. Protein restriction lowers propionyl-CoA precursor load → MMA levels fall. Same rationale as cblC/cblF/MMUT."},
            {"treatment": "L-Carnitine 100 mg/kg/day", "evidence": "Level A–B",
             "response_pct": 80,
             "note": "Secondary carnitine depletion via propionylcarnitine excretion. Supplement all cblJ patients. Oral supplementation Level A for confirmed depletion."},
            {"treatment": "Folinic acid 0.5–5 mg/day", "evidence": "Level B",
             "response_pct": 52,
             "note": "5-methylTHF depleted when methionine synthase is blocked. Folinic acid bypasses blocked folate cycle. Dose 0.5–5 mg/day. May improve seizure control (infantile spasms) and methylation status."},
            {"treatment": "IV Glucose GIR 8–12 (acute crisis)", "evidence": "Level A",
             "response_pct": 83,
             "note": "First intervention in acute catabolic crisis; stops BCAA catabolism → MMA surge halted; continue OHCbl IM + betaine throughout. Mandatory during intercurrent illness."},
            {"treatment": "Ammonia scavengers (Na benzoate / Na phenylacetate)", "evidence": "Level B",
             "response_pct": 62,
             "note": "NH3 >200 µmol/L; secondary via MMA-mediated NAGS/CPS1 inhibition. Same protocol as MMUT/PA/cblF."},
            {"treatment": "VPA — HIGH RISK / AVOID", "evidence": "AVOID",
             "response_pct": 0,
             "note": "HIGH RISK in all cblJ patients (combined MMA+HHcy). VPA-CoA inhibits cobalamin metabolism + carnitine depletion. LEV is first-line AED for ALL seizure types in ABCD4/cblJ."},
        ],
        "patient_sample": sample,
    }


def get_definitions() -> dict:
    return {
        "gene_card": {
            "Gene": "ABCD4 (ATP-Binding Cassette Subfamily D Member 4; cblJ gene product; also called MAHCJ)",
            "Full Name": "Methylmalonic Aciduria and Homocystinuria, cblJ type (ABCD4; Complementation Group cblJ)",
            "Chromosome": "14q24.3",
            "Inheritance": "Autosomal Recessive (AR)",
            "OMIM Gene": "*603214",
            "OMIM Disease": "#614857 (Methylmalonic Aciduria and Homocystinuria, cblJ type)",
            "Protein Size": "606 amino acids; lysosomal membrane ABC half-transporter; 6 TM helices + NBD; Walker A/B ATPase motifs",
            "Protein Function": "ATPase MOTOR of lysosomal cobalamin export complex — hydrolyzes ATP to power LMBRD1-mediated cobalamin translocation from lysosomal lumen to cytoplasm",
            "Pathway Position": "Lysosomal cobalamin export (Step 1 motor partner) — alongside LMBRD1; downstream of TC2/CD320; upstream of MMACHC",
            "ABCD Family Note": "UNIQUE among ABCD transporters: ABCD1/ABCD2/ABCD3 are PEROXISOMAL; ABCD4 is LYSOSOMAL — only ABCD member at lysosomal membrane",
            "Prevalence": "~1:2,000,000 (ultra-rare; <25 reported cases worldwide 2026; likely underdiagnosed)",
        },
        "key_concepts": [
            {
                "concept": "ABCD4 is the ATPase MOTOR of the lysosomal cobalamin export complex",
                "explanation": (
                    "ABCD4 forms a functional heterodimer with LMBRD1 at the lysosomal membrane. "
                    "LMBRD1 provides the cobalamin transporter pore (9 TM helices); "
                    "ABCD4 provides the ATPase motor (NBD domain with Walker A/B motifs). "
                    "ATP hydrolysis by ABCD4 drives conformational changes that open the LMBRD1 pore → cobalamin exported. "
                    "ABCD4 LOF = no ATPase activity = LMBRD1 pore cannot cycle → cobalamin trapped in lysosome."
                ),
            },
            {
                "concept": "Biochemically IDENTICAL to cblF (LMBRD1) — gene panel MANDATORY to distinguish",
                "explanation": (
                    "cblJ (ABCD4) and cblF (LMBRD1) block the SAME step (lysosomal cobalamin export) via different molecular mechanisms. "
                    "The result is biochemically IDENTICAL: combined MMA + HHcy, both MeCbl and AdoCbl absent, "
                    "lysosomal cobalamin on fibroblast EM. NO biochemical test can distinguish cblJ from cblF. "
                    "Gene panel testing BOTH ABCD4 and LMBRD1 is MANDATORY for any combined MMA+HHcy case with lysosomal EM features."
                ),
            },
            {
                "concept": "NO vacuolated lymphocytes in cblJ — KEY NEGATIVE vs cblF (LMBRD1)",
                "explanation": (
                    "Unlike cblF (LMBRD1 pore defect), cblJ (ABCD4 ATPase defect) does NOT produce vacuolated lymphocytes on blood smear. "
                    "The mechanistic basis is uncertain: LMBRD1 pore defect may cause a different lysosomal accumulation pattern "
                    "than ABCD4 ATPase defect, or LMBRD1 may play additional lysosomal membrane roles. "
                    "Absent vacuolated lymphocytes in combined MMA+HHcy patient → cblJ possible (but also cblC, cblD-Combined) "
                    "→ gene panel required. Present vacuolated lymphocytes → cblF (LMBRD1) more likely."
                ),
            },
            {
                "concept": "NO stomatitis in cblJ — KEY NEGATIVE vs cblF (LMBRD1)",
                "explanation": (
                    "Stomatitis and oral ulcers are characteristic of cblF (LMBRD1 pore defect; ~65% of patients). "
                    "In cblJ (ABCD4 ATPase defect), stomatitis has NOT been reported. "
                    "Mechanism of cblF stomatitis is postulated to involve LMBRD1 pore dysfunction in mucosal epithelial lysosomes; "
                    "ABCD4 ATPase defect may not produce the same epithelial lysosomal phenotype. "
                    "Clinical rule: stomatitis + combined MMA+HHcy → cblF first; absent stomatitis → still need gene panel."
                ),
            },
            {
                "concept": "ABCD4 is LYSOSOMAL — unique among ABCD subfamily D transporters",
                "explanation": (
                    "The ABCD family D members (ABCD1-4) are all membrane transporters, but: "
                    "ABCD1 (X-ALD gene), ABCD2, and ABCD3 (PMP70) are peroxisomal VLCFA importers. "
                    "ABCD4 is the ONLY ABCD member that localizes to LYSOSOMES — functionally and biochemically distinct from its paralogs. "
                    "ABCD4 mutations cause cobalamin metabolism disorder (combined MMA+HHcy), NOT adrenoleukodystrophy/peroxisomal VLCFA disease. "
                    "VLCFA levels are NORMAL in ABCD4 deficiency — KEY NEGATIVE vs ABCD1/X-ALD."
                ),
            },
            {
                "concept": "OHCbl bypasses lysosomal trap via alternative entry — partial response in cblJ",
                "explanation": (
                    "IM OHCbl enters cells not only via CD320/lysosomal endocytosis but also via diffusion "
                    "and alternative uptake mechanisms that deliver cobalamin to the cytoplasm without lysosomal transit. "
                    "This partial bypass of the ABCD4-LMBRD1 block provides cob(I)alamin to MMACHC. "
                    "Response: ~50–65% HHcy; ~35–55% MMA — effective but slightly less than cblC (~75%). "
                    "Betaine MANDATORY alongside OHCbl to supplement the HHcy arm cobalamin-independently."
                ),
            },
            {
                "concept": "N2O ABSOLUTE CI — same mechanism as cblC/cblF",
                "explanation": (
                    "Nitrous oxide (N2O) irreversibly oxidizes the cob(I)alamin cofactor → Methionine Synthase (MTR) "
                    "completely inactivated → acute severe HHcy crisis. "
                    "In cblJ, methionine synthase is already compromised (MeCbl cannot be synthesized). "
                    "N2O exposure = methionine synthase failure → LIFE-THREATENING metabolic crisis. "
                    "Notify anesthesiology at every surgery: N2O ABSOLUTE CI. Use alternative anesthetics (e.g., propofol/sevoflurane)."
                ),
            },
            {
                "concept": "Betaine MANDATORY — cobalamin-independent Hcy remethylation via BHMT",
                "explanation": (
                    "Betaine activates BHMT (betaine-homocysteine methyltransferase), which remethylates "
                    "homocysteine to methionine using betaine as methyl donor — completely cobalamin-INDEPENDENT. "
                    "This provides HHcy control even when ABCD4-LMBRD1 block prevents MeCbl synthesis. "
                    "Betaine Level A alongside OHCbl for ALL cblJ patients. Same evidence base as cblC/cblF."
                ),
            },
        ],
        "diagnostic_thresholds": [
            {"parameter": "Urine MMA", "threshold": ">150 mmol/mol Cr",
             "action": "Urine organic acids + plasma tHcy simultaneously. If MMA elevated AND tHcy elevated → combined cobalamin disorder (cblC/cblF/cblJ/cblD-Combined). Gene panel (ABCD4+LMBRD1+MMACHC+MMADHC) immediately."},
            {"parameter": "Plasma tHcy", "threshold": ">25 µmol/L",
             "action": "Plasma amino acids (methionine) + urine MMA simultaneously. If tHcy AND MMA elevated → combined block; gene panel immediately. Start OHCbl + betaine empirically."},
            {"parameter": "Plasma methionine", "threshold": "<18 µmol/L (low)",
             "action": "Low methionine + high tHcy = methionine synthase pathway blocked. Start betaine + OHCbl immediately (Level A) while awaiting gene panel. Combined with high MMA → lysosomal cobalamin disorder."},
            {"parameter": "NBS C3", "threshold": ">3 µmol/L",
             "action": "Reflex: plasma acylcarnitines + urine MMA + plasma tHcy + plasma amino acids. If combined MMA+HHcy → gene panel ABCD4+LMBRD1+MMACHC+MMADHC urgently."},
            {"parameter": "Plasma NH3", "threshold": ">100 µmol/L",
             "action": "Secondary hyperammonemia via MMA-NAGS/CPS1 inhibition. IV glucose GIR 8–12 + OHCbl IM + betaine + ammonia scavengers if NH3 >200."},
            {"parameter": "Blood smear — NO vacuolated lymphocytes", "threshold": "Normal smear + combined MMA+HHcy",
             "action": "Normal blood smear does NOT exclude cblJ. cblJ (ABCD4) has NO vacuolated lymphocytes. Gene panel required. Absent vacuolated lymphocytes → less likely cblF (LMBRD1) but cannot exclude cblC/cblJ/cblD-Combined."},
            {"parameter": "OHCbl trial: tHcy", "threshold": ">50% reduction after 7 days",
             "action": "Confirms OHCbl responsiveness. Continue OHCbl lifelong IM + betaine. Partial response still indicates treatment benefit."},
            {"parameter": "Fibroblast EM — lysosomal cobalamin", "threshold": "Present",
             "action": "Lysosomal cobalamin on EM = lysosomal export defect (cblJ or cblF). Cannot distinguish by EM — gene panel ABCD4+LMBRD1 mandatory. Different from cblC (cytoplasmic block; EM normal)."},
        ],
        "differential_diagnosis": [
            {"disease": "cblF (LMBRD1 — 6q13)",
             "distinguishing": "Biochemically IDENTICAL to cblJ (combined MMA+HHcy; both MeCbl/AdoCbl absent; lysosomal cobalamin on EM). ONLY distinguishing features: (1) vacuolated lymphocytes present in cblF (~25–45%) — ABSENT in cblJ; (2) stomatitis present in cblF (~65%) — ABSENT in cblJ. Gene panel (ABCD4+LMBRD1) mandatory — CANNOT be distinguished biochemically."},
            {"disease": "cblC (MMACHC — 1p34.1)",
             "distinguishing": "Combined MMA+HHcy (same pattern). KEY DIFFERENCES: (1) MMACHC is cytoplasmic (not lysosomal); lysosomal EM normal in cblC; (2) maculopathy more common in cblC (80%); (3) vacuolated lymphocytes ABSENT in cblC; (4) onset typically neonatal in cblC vs infantile in cblJ. Gene panel confirms."},
            {"disease": "cblD-Combined (MMADHC — 2q23.2)",
             "distinguishing": "Combined MMA+HHcy (similar). KEY DIFFERENCES: (1) MMADHC cytoplasmic (downstream of MMACHC); (2) no vacuolated lymphocytes; (3) no stomatitis; (4) NBS may miss cblD-HHcy-only subtype; (5) fibroblast EM normal in MMADHC. Gene panel confirms."},
            {"disease": "PA — Propionic Acidemia (PCCA/PCCB)",
             "distinguishing": "C3 elevated (same as cblJ). KEY NEGATIVES in cblJ vs PA: (1) tHcy NORMAL in PA — ELEVATED in cblJ; (2) methylcitrate PATHOGNOMONIC in PA — only mildly elevated secondary in cblJ; (3) MMA NORMAL in PA — ELEVATED in cblJ."},
            {"disease": "Isolated MMA (MMUT/MMAA/MMAB)",
             "distinguishing": "MMA elevated (same as cblJ). KEY NEGATIVES in cblJ vs isolated MMA: (1) tHcy NORMAL in isolated MMA — ELEVATED in cblJ; (2) methionine NORMAL — LOW in cblJ. The HHcy component distinguishes combined from isolated MMA."},
            {"disease": "ABCD1 — X-ALD (Xq28)",
             "distinguishing": "Both ABCD family members. KEY DIFFERENCES: (1) ABCD1 is PEROXISOMAL — VLCFA ELEVATED in X-ALD; VLCFA NORMAL in ABCD4/cblJ; (2) adrenal insufficiency in X-ALD (71%) — ABSENT in cblJ; (3) MMA/HHcy NORMAL in X-ALD — ELEVATED in cblJ; (4) X-linked dominant vs AR. Completely different pathway and biochemistry."},
            {"disease": "CBS — Classical Homocystinuria (CBS — 21q22.3)",
             "distinguishing": "HHcy elevated (same). KEY NEGATIVES vs cblJ: (1) MMA NORMAL in CBS — ELEVATED in cblJ; (2) methionine HIGH in CBS (blocked transsulfuration) — LOW in cblJ (blocked remethylation); (3) CBS onset childhood/adult; (4) marfanoid features in CBS absent in cblJ."},
        ],
    }
