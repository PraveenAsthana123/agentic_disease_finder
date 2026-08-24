#!/usr/bin/env python3
"""HCFC1 (Methylmalonic Acidemia with Homocystinuria, cblX type) Epilepsy Dashboard.

HCFC1 encodes Host Cell Factor C1, a 2035-aa transcriptional coactivator that directly
activates MMACHC (cblC gene) transcription — making HCFC1 the TRANSCRIPTIONAL MASTER
REGULATOR of intracellular cobalamin metabolism.

COBALAMIN PATHWAY — HCFC1 IS THE TRANSCRIPTIONAL ACTIVATOR OF MMACHC (Step 2):
  Step 0: Dietary cobalamin bound to TC2 → CD320 receptor → endocytosis → lysosome.
  Step 1: LMBRD1 (cblF) + ABCD4 (cblJ) export cobalamin from lysosome → cytoplasm.
  Step 2: MMACHC (cblC) — converts cobalamin to cob(I)alamin.
          ← HCFC1 controls this step indirectly: HCFC1 activates MMACHC transcription
             via THAP11 (Ronin) transcription factor recruitment.
          ← cblX BLOCK: HCFC1 LOF → MMACHC not expressed → MMACHC protein absent →
             cobalamin cannot be processed → BOTH metabolic arms blocked.
  Step 3: MMADHC distributes cob(I)alamin to MMAB (AdoCbl arm) and MTR (MeCbl arm).
  Step 4: AdoCbl cofactor for MMUT (MMA catabolism); MeCbl cofactor for MTR (Hcy remethylation).

HCFC1 LOF RESULT — IDENTICAL DOWNSTREAM CONSEQUENCE TO cblC (MMACHC LOF):
  → MMA ELEVATED (200–2,500 mmol/mol Cr): mitochondrial arm starved of AdoCbl.
  → tHcy ELEVATED (40–300 µmol/L): cytoplasmic arm starved of MeCbl.
  → Methionine LOW (6–18 µmol/L): Methionine Synthase (MTR) inactive.
  → MeCbl fibroblasts: ABSENT (MMACHC cannot process cobalamin).
  → AdoCbl fibroblasts: ABSENT.
  → Biochemical pattern: COMBINED MMA + HHcy — IDENTICAL to cblC/MMACHC.

KEY DIFFERENCE: HCFC1 (cblX) vs MMACHC (cblC):
  1. X-LINKED (Xq28): males severely affected; females mostly asymptomatic carriers.
     ALL other cobalamin disorders (cblA–cblG, cblJ) are AUTOSOMAL RECESSIVE.
  2. SEVERE INTELLECTUAL DISABILITY (100% of males): IQ typically <40; profound NDD.
     cblC has variable ID; late-onset cblC can have near-normal IQ.
  3. NO MACULOPATHY in cblX — cblC has maculopathy in 80%. KEY NEGATIVE vs cblC.
  4. THAP11/Ronin pathway involvement: HCFC1 recruits THAP11 → THAP11 binds MMACHC
     promoter → activates transcription. THAP11 variants ALSO cause cblX-like disease.
  5. More severe epilepsy profile: Ohtahara syndrome (30%), West syndrome/IS (55%),
     vs cblC more typically infantile spasms without early Ohtahara.
  6. MMACHC mRNA and protein dramatically REDUCED in fibroblasts — unlike cblC where
     MMACHC protein is structurally defective (present but non-functional).
  7. WES/WGS mandatory for diagnosis: biochemistry cannot distinguish cblX from cblC.

HCFC1 PROTEIN BIOLOGY:
  2035 amino acids; nuclear protein; transcriptional coactivator.
  Contains: β-propeller (Kelch-repeat) domain (AA 1–380; TC2-receptor binding);
  Basic domain (AA 384–415; gene activation); Pro-Glu repeat (PEST) region;
  HCF-Pro repeat domain; Fn3-like repeat domain; SAS-I binding motif; acidic TAD.
  HCFC1 is cleaved at HCF-Pro repeats into N-terminal (1100 aa) and C-terminal (~935 aa) fragments
  that remain non-covalently associated.
  THAP11 (Ronin) binds the N-terminal HCFC1 fragment via its THAP domain →
  recruits HCFC1 to MMACHC promoter → transcriptional activation.
  Also regulates other genes: cell-cycle progression (E2F), cytokinesis (CDCA6).

HCFC1 BIOMARKERS (identical to cblC/MMACHC):
  C3 (propionylcarnitine): 4–25 µmol/L↑ (NBS PRIMARY — elevated; cblX detected by NBS)
  Urine MMA: 200–2,500 mmol/mol Cr↑ (higher range than cblF/cblJ; closer to cblC/cblD-combined)
  tHcy: 40–300 µmol/L↑ (PATHOGNOMONIC combined pattern)
  Methionine: 6–18 µmol/L LOW (severe; methionine synthase blocked)
  MeCbl fibroblasts: ABSENT
  AdoCbl fibroblasts: ABSENT
  MMACHC mRNA: DRAMATICALLY REDUCED — DIAGNOSTIC KEY vs cblC (MMACHC protein reduced in cblX)
  MMACHC protein immunoblot: ABSENT or severely reduced — KEY DIAGNOSTIC in cblX vs cblC

KEY VARIANTS IN HCFC1:
  p.Pro190Leu (c.569C>T): β-propeller; most commonly reported; severe; X-linked hemizygous.
    Disrupts THAP11 binding interface → MMACHC not activated.
  p.Arg986Cys (c.2956C>T): HCF-Pro repeat; severe neonatal; Ohtahara syndrome.
    Disrupts cleavage and N/C fragment association.
  p.Ala115Val (c.344C>T): β-propeller; moderate-severe; infantile spasms.
    Partial THAP11 recruitment; residual MMACHC transcription ~15–20%.
  p.Gly130Asp (c.389G>A): β-propeller; severe; neonatal encephalopathy.
  c.340-1G>A (splice; IVS3-1G>A): null allele; severe; Ohtahara + WS.
  p.Arg583Gln (c.1748G>A): Basic domain; moderate; childhood onset; better prognosis.
  p.Ala603Thr (c.1807G>A): Basic domain; attenuated; adolescent onset (rare).

TREATMENT:
  OHCbl 1–2 mg/day IM (Level A): same as cblC; bypasses MMACHC block at post-lysosomal level.
    BUT: RESPONSE IS LESS COMPLETE THAN cblC — MMACHC protein is absent, not dysfunctional.
    ~40–60% HHcy response; ~25–45% MMA response. Lower than cblC (~75%/~55–60%).
  Betaine (TMG) 100–200 mg/kg/day (Level A): MANDATORY; BHMT-mediated cobalamin-independent HHcy control.
  Protein restriction + MMA-free formula (Level A): same rationale as cblC.
  L-Carnitine 100 mg/kg/day (Level A–B): secondary depletion.
  Folinic acid 0.5–5 mg/day (Level B): 5-methylTHF depleted.
  IV Glucose GIR 8–12 (Level A): acute metabolic crisis.

CONTRAINDICATIONS:
  N2O — ABSOLUTE CI: same as cblC; methionine synthase failure → LIFE-THREATENING.
  VPA — HIGH RISK: carnitine depletion + metabolic stress; LEV first-line ALL seizures.
  Fasting — HIGH RISK: catabolic MMA surge + worsening HHcy.

OMIM: Gene *300019 (HCFC1) · Disease #309541 (Methylmalonic Aciduria and Homocystinuria, cblX type)
Chromosome: Xq28 · Inheritance: X-LINKED (hemizygous males affected; carrier females unaffected)
Prevalence: ~1:500,000 (rare; ~60–80 reported males worldwide 2026; likely underdiagnosed; NBS misses some)
"""

from __future__ import annotations
import random

random.seed(55)   # reproducible synthetic cohort — seed 55 for cblX/HCFC1


# ── helpers ───────────────────────────────────────────────────────────────────

def _pid(i: int) -> str:
    return f"HCFC1-{i:03d}"


def _sex(i: int) -> str:
    # X-linked: predominantly males in cohort; include obligate carrier females (20%)
    return "F" if i % 5 == 4 else "M"


def _phenotype(i: int) -> str:
    """
    Ohtahara / Neonatal Severe (~30%): <1 month; burst-suppression; Ohtahara + WS progression
    West Syndrome / Infantile Classic (~55%): 2–12 months; infantile spasms + HHcy+MMA crisis
    Childhood Attenuated (~15%): 12–60 months; mild HHcy+MMA; less severe ID (partial HCFC1)
    """
    if i < 12:
        return "Ohtahara Neonatal Severe"
    elif i < 34:
        return "West Syndrome Infantile Classic"
    else:
        return "Childhood Attenuated"


def _genotype(i: int, pheno: str) -> str:
    pheno_variants = {
        "Ohtahara Neonatal Severe": [
            "p.Arg986Cys/hemizygous (HCF-Pro repeat; severe neonatal Ohtahara)",
            "c.340-1G>A/hemizygous (IVS3-1G>A splice null; severe Ohtahara + WS)",
            "p.Gly130Asp/hemizygous (β-propeller null; neonatal encephalopathy)",
            "p.Pro190Leu/hemizygous (β-propeller THAP11-binding; severe early-onset)",
        ],
        "West Syndrome Infantile Classic": [
            "p.Pro190Leu/hemizygous (β-propeller; most common HCFC1 allele; IS)",
            "p.Ala115Val/hemizygous (β-propeller; partial THAP11 recruitment; IS)",
            "p.Pro190Leu/hemizygous (β-propeller; IS + WS; absent MMACHC protein)",
            "p.Ala115Val/hemizygous (β-propeller; 15% residual MMACHC; IS)",
            "p.Pro190Leu/c.340-1G>A — carrier female (unaffected clinically)",
            "p.Pro190Leu/hemizygous (β-propeller; West syndrome; profound ID)",
        ],
        "Childhood Attenuated": [
            "p.Arg583Gln/hemizygous (Basic domain; moderate; childhood onset)",
            "p.Ala603Thr/hemizygous (Basic domain; attenuated; adolescent onset rare)",
            "p.Arg583Gln/hemizygous (Basic domain; partial THAP11 recruitment retained)",
        ],
    }
    opts = pheno_variants[pheno]
    return opts[i % len(opts)]


def _mma_urine(pheno: str) -> float:
    if pheno == "Ohtahara Neonatal Severe":
        return round(random.uniform(500, 2500), 0)
    elif pheno == "West Syndrome Infantile Classic":
        return round(random.uniform(200, 1200), 0)
    else:
        return round(random.uniform(80, 400), 0)


def _homocysteine(pheno: str) -> float:
    if pheno == "Ohtahara Neonatal Severe":
        return round(random.uniform(80, 300), 1)
    elif pheno == "West Syndrome Infantile Classic":
        return round(random.uniform(40, 180), 1)
    else:
        return round(random.uniform(20, 80), 1)


def _methionine(pheno: str) -> float:
    if pheno == "Ohtahara Neonatal Severe":
        return round(random.uniform(6, 12), 1)
    elif pheno == "West Syndrome Infantile Classic":
        return round(random.uniform(8, 16), 1)
    else:
        return round(random.uniform(12, 22), 1)


def _c3(pheno: str) -> float:
    if pheno == "Ohtahara Neonatal Severe":
        return round(random.uniform(8, 25), 1)
    elif pheno == "West Syndrome Infantile Classic":
        return round(random.uniform(4, 16), 1)
    else:
        return round(random.uniform(3, 9), 1)


def _ammonia(pheno: str) -> int:
    if pheno == "Ohtahara Neonatal Severe":
        return random.randint(100, 600)
    elif pheno == "West Syndrome Infantile Classic":
        return random.randint(40, 180)
    else:
        return random.randint(20, 70)


def _carnitine(pheno: str) -> float:
    if pheno == "Ohtahara Neonatal Severe":
        return round(random.uniform(7, 20), 1)
    elif pheno == "West Syndrome Infantile Classic":
        return round(random.uniform(10, 28), 1)
    else:
        return round(random.uniform(18, 40), 1)


def _onset_months(pheno: str) -> int:
    if pheno == "Ohtahara Neonatal Severe":
        return random.randint(0, 1)
    elif pheno == "West Syndrome Infantile Classic":
        return random.randint(2, 12)
    else:
        return random.randint(12, 60)


def _ohcbl_response(pheno: str) -> bool:
    rates = {"Ohtahara Neonatal Severe": 0.35, "West Syndrome Infantile Classic": 0.50, "Childhood Attenuated": 0.72}
    return random.random() < rates[pheno]


def _seizures(pheno: str) -> bool:
    rates = {"Ohtahara Neonatal Severe": 0.97, "West Syndrome Infantile Classic": 0.90, "Childhood Attenuated": 0.62}
    return random.random() < rates[pheno]


def _nbs(pheno: str) -> bool:
    rates = {"Ohtahara Neonatal Severe": 0.70, "West Syndrome Infantile Classic": 0.62, "Childhood Attenuated": 0.45}
    return random.random() < rates[pheno]


def _aed(sz: bool, pheno: str) -> str:
    if not sz:
        return "None"
    if pheno == "Ohtahara Neonatal Severe":
        opts = ["LEV + PB + ACTH (OHT progression to IS)", "LEV + ACTH + CLB (OS+WS transition)",
                "LEV + PB (neonatal OS; OHCbl + betaine commenced)", "LEV + ACTH (IS onset; OHCbl trial)"]
    elif pheno == "West Syndrome Infantile Classic":
        opts = ["LEV + ACTH (IS)", "LEV + VIG (IS)", "LEV + CLB", "LEV + ACTH + CLB"]
    else:
        opts = ["LEV", "LEV + CLB", "LEV + LTG"]
    return random.choice(opts)


def _diag_months(pheno: str, onset: int) -> int:
    delays = {"Ohtahara Neonatal Severe": (0, 3), "West Syndrome Infantile Classic": (1, 8),
              "Childhood Attenuated": (3, 18)}
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
            "mecbl_fibroblasts": "Absent",
            "adocbl_fibroblasts": "Absent",
            "mmachc_mrna_reduced": True,    # DIAGNOSTIC KEY: cblX has reduced MMACHC mRNA
            "mmachc_protein_absent": True,  # DIAGNOSTIC KEY: MMACHC protein absent vs cblC
            "maculopathy": False,           # ABSENT in cblX — KEY NEGATIVE vs cblC
            "seizures": sz,
            "nbs_detected": _nbs(pheno),
            "ohcbl_response": _ohcbl_response(pheno),
            "aed": _aed(sz, pheno),
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

    ohta_pts   = [p for p in _PATIENTS if p["phenotype"] == "Ohtahara Neonatal Severe"]
    west_pts   = [p for p in _PATIENTS if p["phenotype"] == "West Syndrome Infantile Classic"]
    child_pts  = [p for p in _PATIENTS if p["phenotype"] == "Childhood Attenuated"]

    return {
        "gene": "HCFC1",
        "full_name": "Host Cell Factor C1 — Transcriptional Coactivator / MMACHC Transcription Activator (cblX gene product)",
        "chromosome": "Xq28",
        "inheritance": "X-LINKED (hemizygous males affected; carrier females typically unaffected)",
        "omim_gene": "*300019",
        "omim_disease": "#309541",
        "protein_size": "2035 aa; nuclear transcriptional coactivator; β-propeller / Kelch-repeat domain (AA 1–380); HCF-Pro repeat cleavage domain; Basic domain; Fn3-like domain; acidic TAD",
        "prevalence": "~1:500,000 (rare; ~60–80 reported males worldwide 2026; likely underdiagnosed; X-linked so all cases male except rare skewed X-inactivation)",
        "nbs_primary": "C3 (propionylcarnitine) elevated — cblX detected by NBS in ~62% (combined MMA+HHcy → C3 elevated); NBS misses ~38% (lower C3 in attenuated subtypes and some neonates before metabolic crisis)",
        "nbs_secondary": "Urine MMA + plasma tHcy simultaneously mandatory; MMACHC mRNA quantification in fibroblasts distinguishes cblX (mRNA absent/reduced) from cblC (mRNA present, protein dysfunctional); WES/WGS for Xq28 HCFC1 variants mandatory",
        "function": (
            "HCFC1 is the TRANSCRIPTIONAL MASTER REGULATOR of MMACHC (cblC gene). "
            "HCFC1 forms a complex with THAP11 (Ronin) — THAP11 binds the MMACHC promoter "
            "via its THAP zinc-finger domain; HCFC1 N-terminal domain recruits THAP11 to the promoter "
            "and activates MMACHC transcription. "
            "HCFC1 LOF → THAP11 cannot activate MMACHC promoter → MMACHC mRNA absent → "
            "MMACHC protein absent → cobalamin processing blocked → COMBINED MMA + HHcy. "
            "HCFC1 is X-linked: the ONLY X-linked gene in the entire intracellular cobalamin pathway."
        ),
        "mechanism": (
            "HCFC1 LOF silences MMACHC transcription via disruption of HCFC1–THAP11 complex. "
            "Unlike cblC (MMACHC structural variants → dysfunctional protein), "
            "cblX has NO MMACHC protein at all — the gene is not transcribed. "
            "This predicts more severe response to OHCbl therapy (no enzymatic scaffold available). "
            "Result: biochemically IDENTICAL to cblC — combined MMA + HHcy + LOW methionine — "
            "but mechanistically different: transcriptional (not enzymatic) defect. "
            "KEY DISTINCTION: MMACHC mRNA/protein dramatically reduced (absent) in fibroblasts → DIAGNOSTIC."
        ),
        "key_negative": (
            "Maculopathy: ABSENT in cblX — KEY NEGATIVE vs cblC (MMACHC; ~80% maculopathy); "
            "HCFC1 is X-linked: ONLY X-linked cobalamin disorder — ALL other cbl disorders are AR; "
            "MMACHC mRNA present in cblC (defective protein) — ABSENT in cblX (no transcription); "
            "Biochemically identical to cblC — WES/WGS and fibroblast MMACHC mRNA quantification mandatory."
        ),
        "cohort_n": n,
        "kpis": {
            "avg_mma_urine": round(sum(mma_v) / n),
            "avg_homocysteine_umol_l": round(sum(hcy_v) / n),
            "avg_methionine_umol_l": round(sum(met_v) / n, 1),
            "avg_c3_umol_l": round(sum(c3_v) / n, 1),
            "avg_ammonia_umol_l": round(sum(nh3_v) / n),
            "avg_free_carnitine_umol_l": round(sum(car_v) / n, 1),
            "pct_seizures": pct(lambda p: p["seizures"]),
            "pct_nbs_detected": pct(lambda p: p["nbs_detected"]),
            "pct_ohcbl_response": pct(lambda p: p["ohcbl_response"]),
            "pct_maculopathy": 0,  # ABSENT in cblX
            "pct_mmachc_protein_absent": 100,  # all cblX: MMACHC protein absent
            "pct_male": pct(lambda p: p["sex"] == "M"),
        },
        "phenotype_distribution": {
            "ohtahara_neonatal_severe": {"n": len(ohta_pts), "pct": round(len(ohta_pts) / n * 100)},
            "west_syndrome_infantile_classic": {"n": len(west_pts), "pct": round(len(west_pts) / n * 100)},
            "childhood_attenuated": {"n": len(child_pts), "pct": round(len(child_pts) / n * 100)},
        },
    }


def get_breakdown() -> dict:
    n = len(_PATIENTS)

    def pct(pred): return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    seizure_types = [
        {"type": "Infantile Spasms (West Syndrome)", "pct": pct(lambda p: p["seizures"] and "West" in p["phenotype"])},
        {"type": "Ohtahara / Burst-Suppression (neonatal)", "pct": pct(lambda p: p["seizures"] and "Ohtahara" in p["phenotype"])},
        {"type": "Focal seizures (attenuated / childhood)", "pct": pct(lambda p: p["seizures"] and "Childhood" in p["phenotype"])},
        {"type": "Myoclonic / tonic (any phenotype)", "pct": 42},
        {"type": "Generalized tonic-clonic", "pct": 28},
    ]

    metabolic_triggers = [
        {"trigger": "N2O exposure (anesthesia)", "pct": 100,
         "mechanism": "Irreversibly oxidizes MTR cobalt → methionine synthase failure → acute severe HHcy; LIFE-THREATENING."},
        {"trigger": "Fasting / catabolism", "pct": 78,
         "mechanism": "BCAA catabolism → propionyl-CoA → MMA surge; fasting also worsens HHcy via decreased methionine intake."},
        {"trigger": "Intercurrent illness / fever", "pct": 65,
         "mechanism": "Catabolic crisis in all organic acidemias; IV glucose mandatory."},
        {"trigger": "Missed OHCbl + betaine doses", "pct": 55,
         "mechanism": "tHcy rebounds rapidly (~48 h); MMA rebounds within 3–7 days. Both treatments mandatory."},
        {"trigger": "High protein / excess BCAA", "pct": 35,
         "mechanism": "Propionyl-CoA overload → MMA surge; protein restriction mandatory."},
    ]

    sample_size = min(6, n)
    sample = [{
        "id": p["patient_id"],
        "sex": p["sex"],
        "phenotype": p["phenotype"][:22],
        "genotype": p["genotype"][:40],
        "onset_mo": p["onset_months"],
        "diag_mo": p["diagnosis_months"],
        "mma": p["mma_urine_mmol_molCr"],
        "hcy": p["total_homocysteine_umol_l"],
        "met": p["methionine_umol_l"],
        "c3": p["c3_umol_l"],
        "nh3": p["ammonia_umol_l"],
        "car": p["free_carnitine_umol_l"],
        "sz": p["seizures"],
        "nbs": p["nbs_detected"],
        "ohcbl_resp": p["ohcbl_response"],
        "aed": p["aed"],
        "maculopathy": p["maculopathy"],
        "mmachc_absent": p["mmachc_protein_absent"],
    } for p in _PATIENTS[:sample_size]]

    return {
        "seizure_type_distribution": seizure_types,
        "metabolic_triggers": metabolic_triggers,
        "high_risk_drugs": [
            {"drug": "Nitrous oxide (N2O)", "risk": "ABSOLUTE CI",
             "mechanism": "Irreversibly oxidizes cobalt cofactor → MTR failure → acute severe HHcy. LIFE-THREATENING in ALL cblX patients. Notify anesthesiology at every procedure."},
            {"drug": "Valproate (VPA)", "risk": "HIGH RISK",
             "mechanism": "VPA-CoA inhibits mitochondrial cobalamin metabolism + carnitine depletion → MMA surge. LEV first-line for ALL seizure types in HCFC1/cblX."},
            {"drug": "Antifolates (methotrexate, trimethoprim)", "risk": "HIGH RISK",
             "mechanism": "5-methylTHF depletion worsens methionine synthase pathway failure → acute HHcy. Folinic acid monitoring essential."},
            {"drug": "Fasting / NPO without IV glucose", "risk": "HIGH RISK",
             "mechanism": "Catabolism drives MMA surge + HHcy worsening. IV dextrose GIR 8–12 mandatory throughout any illness."},
        ],
        "treatments": [
            {"treatment": "Hydroxocobalamin (OHCbl) 1–2 mg/day IM", "evidence": "Level A",
             "response_pct": 48,
             "note": "Trial ALL cblX patients. IM OHCbl enters cells via alternative routes → provides cofactor to MTR and MMUT directly. BUT: MMACHC protein ABSENT (not just dysfunctional) → response is LESS than cblC (~40–60% HHcy; ~25–45% MMA vs cblC ~75%/~55%). Lifelong IM mandatory; monitor tHcy + MMA at Day 7 and monthly."},
            {"treatment": "Betaine (TMG) 100–200 mg/kg/day", "evidence": "Level A",
             "response_pct": 72,
             "note": "MANDATORY for HHcy in ALL cblX patients. BHMT remethylates Hcy → Methionine (cobalamin-INDEPENDENT). Level A. Critical because OHCbl response is partial. Continue throughout life regardless of apparent HHcy control."},
            {"treatment": "Protein restriction + MMA-free formula", "evidence": "Level A",
             "response_pct": 65,
             "note": "Low Ile/Val/Met/Thr restricts propionyl-CoA precursors → MMA falls. MMA-free amino acid formula provides nitrogen source. Same Level A evidence as cblC."},
            {"treatment": "L-Carnitine 100 mg/kg/day", "evidence": "Level A–B",
             "response_pct": 76,
             "note": "Secondary carnitine depletion via propionylcarnitine excretion. Supplement all cblX patients. Oral Level A for confirmed depletion; IV in crisis."},
            {"treatment": "Folinic acid 0.5–5 mg/day", "evidence": "Level B",
             "response_pct": 48,
             "note": "5-methylTHF depleted when methionine synthase is blocked. Folinic acid bypasses blocked folate cycle. Dose 0.5–5 mg/day. May improve seizure control and methylation."},
            {"treatment": "IV Glucose GIR 8–12 (acute crisis)", "evidence": "Level A",
             "response_pct": 84,
             "note": "First intervention in acute metabolic or catabolic crisis. Stops BCAA catabolism → MMA surge halted. Continue OHCbl IM + betaine throughout. Mandatory during any illness or NPO episode."},
            {"treatment": "ACTH / Vigabatrin (infantile spasms)", "evidence": "Level A (IS)",
             "response_pct": 52,
             "note": "ACTH or vigabatrin for West syndrome / infantile spasms. Combined with OHCbl + betaine. IS response in cblX is generally less complete than metabolic-cause IS because HCFC1 also has non-metabolic neurodevelopmental roles."},
            {"treatment": "VPA — HIGH RISK / AVOID", "evidence": "AVOID",
             "response_pct": 0,
             "note": "HIGH RISK in all cblX patients. VPA-CoA inhibits cobalamin metabolism + carnitine depletion → MMA surge. LEV is first-line AED for ALL seizure types."},
        ],
        "patient_sample": sample,
    }


def get_definitions() -> dict:
    return {
        "gene_card": {
            "Gene": "HCFC1 (Host Cell Factor C1; cblX gene product; also HCFC, HCF, HCF-1, CFF, VCAF, MRX3)",
            "Full Name": "Methylmalonic Aciduria and Homocystinuria, cblX type (HCFC1; Complementation Group cblX)",
            "Chromosome": "Xq28",
            "Inheritance": "X-LINKED — the ONLY X-linked gene in intracellular cobalamin metabolism; ALL others (cblA–cblG, cblJ) are autosomal recessive",
            "OMIM Gene": "*300019",
            "OMIM Disease": "#309541 (Methylmalonic Aciduria and Homocystinuria, cblX type)",
            "Protein Size": "2035 amino acids; nuclear transcriptional coactivator; β-propeller (Kelch-repeat, AA 1–380) + HCF-Pro repeats (cleavage) + Basic domain + Fn3-like repeats + acidic TAD",
            "Protein Function": "Transcriptional coactivator that recruits THAP11 (Ronin) to the MMACHC promoter → activates MMACHC transcription → enables intracellular cobalamin processing",
            "Pathway Position": "Transcriptional regulation of MMACHC (Step 2) — indirectly controls ALL downstream cobalamin metabolism by controlling MMACHC expression",
            "X-Linked Note": "HCFC1 is at Xq28 (same region as MECP2/Rett syndrome, IKBKG/NEMO). Carrier females: random X-inactivation → usually unaffected; rarely skewed X-inactivation → mild phenotype",
            "Prevalence": "~1:500,000 (rare; ~60–80 reported males worldwide 2026; likely underdiagnosed)",
        },
        "key_concepts": [
            {
                "concept": "HCFC1 is the TRANSCRIPTIONAL REGULATOR of MMACHC — the cblX mechanism",
                "explanation": (
                    "HCFC1 does not directly process cobalamin — it controls MMACHC gene expression. "
                    "HCFC1 recruits THAP11 (Ronin zinc-finger transcription factor) to the MMACHC promoter. "
                    "THAP11 binds HCFC1 N-terminal β-propeller domain → forms activating complex → "
                    "MMACHC transcription initiated. "
                    "HCFC1 LOF → THAP11-HCFC1 complex cannot form → MMACHC promoter inactive → "
                    "MMACHC mRNA absent → MMACHC protein absent → cobalamin processing blocked → "
                    "COMBINED MMA + HHcy (identical to cblC at the biochemical level)."
                ),
            },
            {
                "concept": "MMACHC mRNA/protein ABSENT in cblX — DIAGNOSTIC KEY vs cblC",
                "explanation": (
                    "In cblC (MMACHC structural variants): MMACHC mRNA is PRESENT (the gene is transcribed), "
                    "but MMACHC protein is structurally defective — present but non-functional. "
                    "In cblX (HCFC1 LOF): MMACHC mRNA is ABSENT or dramatically reduced — the gene is NOT "
                    "being transcribed → MMACHC protein completely absent. "
                    "Fibroblast MMACHC mRNA quantification (qRT-PCR) + MMACHC protein immunoblot → "
                    "DIAGNOSTIC test to distinguish cblX from cblC before WES/WGS results. "
                    "This is the ONLY way to distinguish cblX from cblC in a fibroblast assay."
                ),
            },
            {
                "concept": "X-linked — ONLY X-linked gene in intracellular cobalamin metabolism",
                "explanation": (
                    "Every other gene in the intracellular cobalamin pathway (cblA/MMAA, cblB/MMAB, "
                    "cblC/MMACHC, cblD/MMADHC, cblE/MTRR, cblF/LMBRD1, cblG/MTR, cblJ/ABCD4) is "
                    "autosomal recessive. "
                    "HCFC1 (cblX) is the SOLE X-linked disorder in this pathway. "
                    "Clinical implication: (1) ALL affected patients in the series are male; "
                    "(2) carrier mothers are typically unaffected — important for family counseling; "
                    "(3) NBS may miss cblX in females due to X-inactivation mosaicism; "
                    "(4) recurrence risk for brothers is 50%; sisters have 50% chance of being carriers."
                ),
            },
            {
                "concept": "NO maculopathy in cblX — KEY NEGATIVE vs cblC (80% maculopathy)",
                "explanation": (
                    "Maculopathy (optic atrophy, retinal degeneration) is a hallmark of cblC (MMACHC structural defects) "
                    "occurring in ~80% of patients. "
                    "In cblX (HCFC1 LOF), maculopathy has NOT been reported. "
                    "The mechanistic basis is uncertain: possibly HCFC1 in retinal pigment epithelium has non-MMACHC "
                    "transcriptional roles that protect the retina differently, or maculopathy requires specific "
                    "MMACHC partial-function alleles (common in cblC) vs complete absence (cblX). "
                    "Clinical rule: combined MMA+HHcy + maculopathy → cblC strongly. "
                    "Combined MMA+HHcy + NO maculopathy + male + severe ID + reduced MMACHC mRNA → suspect cblX."
                ),
            },
            {
                "concept": "Ohtahara Syndrome and severe epilepsy in cblX",
                "explanation": (
                    "Ohtahara syndrome (early infantile epileptic encephalopathy, EIEE) is overrepresented in cblX (~30%) "
                    "compared to cblC (<5%). "
                    "HCFC1 has transcriptional targets beyond MMACHC — it regulates cell-cycle genes (E2F), "
                    "cytokinesis, and other neural differentiation genes. "
                    "The severe epilepsy in cblX may therefore reflect BOTH: "
                    "(1) metabolic epilepsy from combined MMA+HHcy and methylation deficiency; AND "
                    "(2) primary neurodevelopmental disruption from loss of HCFC1 non-metabolic targets. "
                    "This explains why OHCbl + betaine metabolic treatment alone is insufficient for seizure control "
                    "in many cblX patients — additional AED optimization is required."
                ),
            },
            {
                "concept": "OHCbl response LESS complete than cblC — because NO MMACHC protein scaffold",
                "explanation": (
                    "In cblC: OHCbl bypasses dietary cobalamin processing → provides cofactor directly to MTR/MMUT. "
                    "Even with dysfunctional MMACHC, some residual processing occurs → ~75% HHcy response. "
                    "In cblX: MMACHC protein is COMPLETELY ABSENT (not just dysfunctional). "
                    "OHCbl can still deliver cofactor to MTR/MMUT via alternative MMACHC-independent pathways "
                    "(partial bypass), but without any MMACHC scaffold, response is ~40–60% HHcy; ~25–45% MMA. "
                    "Betaine is therefore ESPECIALLY CRITICAL in cblX to compensate for reduced OHCbl efficacy."
                ),
            },
            {
                "concept": "THAP11 variants also cause cblX-like disease — HCFC1-THAP11 axis",
                "explanation": (
                    "THAP11 (Ronin) is the DNA-binding partner that HCFC1 recruits to the MMACHC promoter. "
                    "Germline loss-of-function variants in THAP11 cause combined MMA+HHcy biochemically identical "
                    "to cblX/HCFC1 — because THAP11 cannot activate MMACHC transcription without HCFC1. "
                    "THAP11 is autosomal (16q22.1) — so THAP11 deficiency is AR, not X-linked. "
                    "Clinical implication: if WES shows combined MMA+HHcy + reduced MMACHC mRNA + no HCFC1 variant, "
                    "sequence THAP11 next — the HCFC1-THAP11 axis is the regulatory module for MMACHC. "
                    "Both HCFC1 and THAP11 cause 'cblX complementation group' by biochemical complementation studies."
                ),
            },
            {
                "concept": "N2O ABSOLUTE CI — same mechanism as all combined MMA+HHcy disorders",
                "explanation": (
                    "Nitrous oxide (N2O) irreversibly oxidizes the cob(I)alamin cofactor of MTR "
                    "→ Methionine Synthase permanently inactivated → acute severe HHcy crisis. "
                    "In cblX, MTR is already partially compromised (MeCbl cannot be synthesized normally). "
                    "N2O exposure in a cblX patient = complete MTR failure → LIFE-THREATENING metabolic crisis. "
                    "Notify anesthesiology and ICU at every procedure: N2O ABSOLUTE CONTRAINDICATION in cblX. "
                    "Alternative anesthetics: propofol, sevoflurane (no cobalamin cofactor interaction)."
                ),
            },
        ],
        "diagnostic_thresholds": [
            {"parameter": "Urine MMA", "threshold": ">200 mmol/mol Cr",
             "action": "Urine organic acids + plasma tHcy simultaneously. If MMA elevated AND tHcy elevated → combined cobalamin disorder. WES/WGS including HCFC1 (Xq28) mandatory."},
            {"parameter": "Plasma tHcy", "threshold": ">40 µmol/L",
             "action": "Plasma amino acids (methionine) + urine MMA simultaneously. Combined MMA+HHcy → gene panel. Start OHCbl + betaine empirically while awaiting genetics."},
            {"parameter": "MMACHC mRNA (fibroblasts)", "threshold": "Dramatically reduced / absent",
             "action": "MMACHC mRNA absent + combined MMA+HHcy + male patient → cblX (HCFC1). Absent mRNA distinguishes cblX from cblC where mRNA is present. WES for HCFC1 (Xq28) variant confirmation."},
            {"parameter": "MMACHC protein (immunoblot)", "threshold": "Absent (cblX) vs present dysfunctional (cblC)",
             "action": "MMACHC protein absent on fibroblast immunoblot in male with combined MMA+HHcy → cblX strongly suspected. cblC has detectable but dysfunctional MMACHC protein."},
            {"parameter": "NBS C3", "threshold": ">4 µmol/L",
             "action": "Reflex: acylcarnitines + urine MMA + plasma tHcy + amino acids. Combined MMA+HHcy → gene panel including HCFC1/THAP11/MMACHC/MMADHC/LMBRD1/ABCD4."},
            {"parameter": "Plasma NH3", "threshold": ">100 µmol/L",
             "action": "Secondary hyperammonemia via MMA-NAGS/CPS1 inhibition. IV glucose GIR 8–12 + OHCbl IM + betaine + ammonia scavengers if NH3 >200."},
            {"parameter": "EEG: burst-suppression pattern", "threshold": "Any neonatal burst-suppression",
             "action": "Ohtahara syndrome + combined MMA+HHcy in male → cblX/HCFC1 high priority differential. WES/WGS urgently. Start OHCbl + betaine + ACTH empirically."},
            {"parameter": "OHCbl trial: tHcy", "threshold": "<50% reduction after 7 days (lower than cblC)",
             "action": "Partial response expected in cblX (MMACHC protein absent). Continue OHCbl regardless. Maximize betaine dose. Partial response still meaningfully reduces HHcy burden."},
        ],
        "differential_diagnosis": [
            {"disease": "cblC (MMACHC — 1p34.1)",
             "distinguishing": "Biochemically IDENTICAL (combined MMA+HHcy). KEY DIFFERENCES: (1) MMACHC mRNA PRESENT in cblC (protein dysfunctional) — ABSENT in cblX; (2) maculopathy ~80% in cblC — ABSENT in cblX; (3) cblC is AR (both sexes); cblX is X-linked (males only); (4) OHCbl response better in cblC (~75% HHcy) than cblX (~40–60%); (5) late-onset psychiatric cblC (c.482G>A allele) not described in cblX. MMACHC mRNA + gene panel distinguishes."},
            {"disease": "cblF (LMBRD1 — 6q13)",
             "distinguishing": "Combined MMA+HHcy (same). KEY DIFFERENCES: (1) LMBRD1 is lysosomal membrane pore (transport defect); cblX is transcriptional; (2) vacuolated lymphocytes 25–45% in cblF — ABSENT in cblX; (3) stomatitis ~65% in cblF — ABSENT in cblX; (4) MMACHC mRNA present in cblF (MMACHC gene normal); (5) cblF is AR; cblX is X-linked."},
            {"disease": "cblJ (ABCD4 — 14q24.3)",
             "distinguishing": "Combined MMA+HHcy. KEY DIFFERENCES: (1) ABCD4 is lysosomal ATPase motor; cblX is transcriptional; (2) NO vacuolated lymphocytes in both (similar); (3) MMACHC mRNA present in cblJ; (4) cblJ is AR; cblX is X-linked; (5) MMA range slightly different."},
            {"disease": "THAP11 deficiency (16q22.1)",
             "distinguishing": "Biochemically IDENTICAL to cblX (reduced MMACHC mRNA + combined MMA+HHcy). ONLY distinguishing feature: gene. THAP11 is AR (16q22.1); HCFC1/cblX is X-linked (Xq28). Both cause cblX complementation group by biochemical complementation. WES/WGS of both HCFC1 and THAP11 mandatory."},
            {"disease": "cblD-Combined (MMADHC — 2q23.2)",
             "distinguishing": "Combined MMA+HHcy (same). KEY DIFFERENCES: (1) MMADHC is downstream of MMACHC (cytoplasmic distributor); MMACHC mRNA/protein present in cblD; (2) no vacuolated lymphocytes, no stomatitis (same); (3) cblD-HHcy-only misses NBS (C3 normal) — cblX always elevated C3; (4) cblD is AR; cblX is X-linked."},
            {"disease": "CBS — Classical Homocystinuria (21q22.3)",
             "distinguishing": "HHcy elevated (same). KEY DIFFERENCES: (1) MMA NORMAL in CBS — ELEVATED in cblX; (2) methionine HIGH in CBS (blocked transsulfuration) — LOW in cblX (blocked remethylation); (3) CBS is AR, later onset, marfanoid; (4) no severe ID or infantile spasms typical of cblX."},
            {"disease": "Ohtahara Syndrome — other causes (STXBP1, KCNQ2, ARX)",
             "distinguishing": "Burst-suppression neonatal epilepsy. KEY DIFFERENCES: metabolic workup distinguishes: combined MMA+HHcy present only in cblX-Ohtahara. Other Ohtahara causes have NORMAL metabolic profile. If Ohtahara + combined MMA+HHcy + male → cblX first differential."},
        ],
    }
