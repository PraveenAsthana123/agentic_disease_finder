#!/usr/bin/env python3
"""MTRR (Methionine Synthase Reductase Deficiency / Homocystinuria-Megaloblastic Anemia cblE type) Epilepsy Dashboard.

MTRR encodes 5-methyltetrahydrofolate-homocysteine methyltransferase reductase, a 698-aa
FMN+FAD diflavin reductase at 5p15.31 that REACTIVATES Methionine Synthase (MTR/cblG) by
reducing its cob(II)alamin cofactor back to the active cob(I)alamin form.

COBALAMIN PATHWAY — MTRR REACTIVATES MTR (Methionine Synthase) at the CYTOPLASMIC STEP:
  Step 0: Dietary cobalamin bound to TC2 → CD320 receptor → endocytosis → lysosome.
  Step 1: LMBRD1 (cblF) + ABCD4 (cblJ) export cobalamin from lysosome → cytoplasm.
  Step 2: MMACHC (cblC) converts cobalamin → cob(I)alamin.
  Step 3: MMADHC distributes cob(I)alamin:
          ARM A (C-terminus, mitochondrial) → MMAB → AdoCbl → MMUT (MMA catabolism)
          ARM B (N-terminus, cytoplasmic)   → MTR → MeCbl (Hcy remethylation → Methionine)
  Step 4a: MMUT uses AdoCbl to catabolize methylmalonyl-CoA → succinyl-CoA (MMA arm).
  Step 4b: MTR (cblG) uses MeCbl to remethylate Hcy → Methionine (HHcy arm).
          ← MTRR acts HERE: MTRR reactivates MTR by transferring electrons from NADPH
             via FAD → FMN → cob(II)alamin (inactive MTR cofactor) → cob(I)alamin (active).
          ← cblE BLOCK: MTRR LOF → MTR slowly accumulates inactive cob(II)alamin → MTR fails
             → Hcy cannot be remethylated → tHcy ELEVATED → Methionine LOW.
             THE MMA ARM (Step 4a) IS COMPLETELY INTACT — MMAB, MMAA, MMUT unaffected.

MTRR LOF RESULT — ISOLATED HHcy (MMA arm intact — KEY NEGATIVE):
  → tHcy ELEVATED (40–200 µmol/L): MTR inactive → Hcy accumulates — ISOLATED, NO MMA.
  → MMA NORMAL: KEY NEGATIVE — MMUT/AdoCbl arm entirely unaffected by MTRR LOF.
  → Methionine LOW (5–20 µmol/L): MTR cannot remethylate Hcy → Methionine not synthesized.
  → MeCbl ABSENT in fibroblasts: MTR cannot be reactivated → cannot synthesize MeCbl.
  → AdoCbl NORMAL in fibroblasts: KEY POSITIVE — MMAB arm independent of MTR/MTRR entirely.
  → C3 NORMAL on NBS: MMA arm intact → C3 not elevated → NBS INVISIBLE for cblE (HIGH MISS RATE).
  → Megaloblastic anemia: MTR failure → 5-methylTHF cannot donate methyl group to THF →
    5-methylTHF trapped (methylfolate trap) → functional THF depletion → impaired DNA synthesis.
  → Folate paradox: serum folate HIGH but functionally unavailable (methylfolate trap).

METHYLFOLATE TRAP (central in cblE — more prominent than in combined disorders):
  Normal: 5-methylTHF → MTR → THF + Methionine (THF re-enters folate cycle for DNA synthesis).
  cblE: MTR inactive → 5-methylTHF cannot donate methyl group → 5-methylTHF accumulates
        → serum folate appears high (lab artifact) → but functionally ZERO THF available
        → megaloblastic anemia + impaired thymidylate synthesis.
  Folinic acid (5-formyl-THF) BYPASSES this block by entering the folate cycle as THF directly
  → more important in cblE (and cblG) than in combined disorders (cblC, cblX) where folate
  trap is secondary to global cobalamin processing failure.

MTRR PROTEIN BIOLOGY:
  698 amino acids; cytoplasmic FMN+FAD diflavin reductase; member of cytochrome P450 reductase
  (CPR) superfamily — same structural family as POR (P450 oxidoreductase) and NOS (nitric oxide
  synthase) reductase domains.
  FMN-binding N-terminal domain (AA 1–175): receives electrons from FAD → transfers to cob(II)alamin.
  FAD-binding C-terminal domain (AA 176–698): accepts electrons from NADPH → passes to FMN.
  Electron flow: NADPH → FAD (C-terminal) → FMN (N-terminal) → cob(II)alamin on MTR → cob(I)alamin.
  MTRR physically docks on MTR (cblG) via the MTRR FMN domain to reductively reactivate it.
  Common polymorphism: p.Ile22Met (c.66A>G, rs1801394) — NOT pathogenic; very common in general
  population (~25% allele frequency); erroneously listed in old literature as cblE; does NOT cause
  cblE disease; must not be confused with true cblE pathogenic variants.

MTRR BIOMARKERS:
  Plasma tHcy: 40–200 µmol/L ELEVATED (isolated — MMA NORMAL KEY NEGATIVE)
  Urine MMA: NORMAL (<5 mmol/mol Cr) — the critical distinguishing feature from cblC/combined
  Plasma methionine: 5–20 µmol/L LOW (MTR inactive → no remethylation → Methionine depleted)
  Serum folate: HIGH (folate trap — falsely elevated due to trapped 5-methylTHF)
  MeCbl fibroblasts: ABSENT (MTR cannot be reactivated → cannot use MeCbl)
  AdoCbl fibroblasts: NORMAL — KEY POSITIVE (MMAB arm independent of MTRR)
  C3 (propionylcarnitine): NORMAL — KEY NEGATIVE (no MMA → C3 not elevated → NBS misses cblE)
  CBC: megaloblastic anemia — macrocytic + hypersegmented neutrophils; elevated MCV

KEY VARIANTS IN MTRR (cblE pathogenic — NOT including I22M polymorphism):
  p.Ser175Leu (c.524C>T): FMN-binding domain at domain junction (AA 175); most common European cblE
    allele; disrupts FMN cofactor coordination → electron transfer to cob(II)alamin impossible;
    moderate-severe phenotype; intermediate OHCbl response.
  p.Arg415Gln (c.1244G>A): FAD-binding C-terminal domain; severe; poor OHCbl response; FAD
    binding disrupted → electron input to MTRR chain completely blocked → null-like.
  p.Arg174Gln (c.521G>A): FMN-binding domain; adjacent to Ser175; moderate; some residual
    electron transfer; better folinic acid + betaine response than Arg415Gln.
  c.903+469T>C: deep intronic; creates cryptic pseudoexon insertion; UK/Irish founder variant;
    splice-type mechanism — pseudoexon inserts premature stop codon; severe; NMD of mRNA.
  p.Arg415* (c.1243C>T): nonsense — null allele; FAD domain truncation; severe; no functional
    MTRR protein; extremely poor OHCbl response.
  p.Thr353Ile (c.1058C>T): FAD-binding domain; moderate-severe; FAD positioning affected;
    intermediate response to OHCbl + betaine combination.

TREATMENT:
  OHCbl 1–2 mg/day IM (Level A): bypasses MTRR requirement — provides cobalamin cofactor in
    excess such that alternative MTRR-independent pathways can synthesize sufficient MeCbl for MTR.
    Response: 60–75% HHcy reduction — BETTER than cblX (~40–60%), SIMILAR to cblC (~75%).
    Does not restore MTRR enzyme itself; exogenous cobalamin in high concentration partially
    compensates for impaired MTR reactivation by mass-action effect on cob(II)→cob(I) equilibrium.
  Betaine (TMG) 100–200 mg/kg/day (Level A): BHMT-mediated cobalamin-independent Hcy remethylation
    → Hcy → Methionine via betaine-homocysteine methyltransferase (BHMT), bypassing MTR completely.
    MANDATORY — essential complement to OHCbl.
  Folinic acid 0.5–5 mg/day (Level A — MORE IMPORTANT IN cblE THAN IN COMBINED DISORDERS):
    5-methylTHF is TRAPPED (methylfolate trap) because MTR cannot accept methyl group.
    Folinic acid (5-formyl-THF) bypasses the methylfolate trap by entering THF pool directly
    → relieves functional folate deficiency → corrects megaloblastic anemia + DNA synthesis.
    This is Level A in cblE (and cblG) because methylfolate trap is the CENTRAL pathological
    mechanism. In cblC/cblX, folate trap is secondary; in cblE it is primary.
  Folic acid (Level B): some evidence for megaloblastic anemia; folinic acid preferred because
    conversion of folic acid → dihydrofolate → THF → 5,10-methyleneTHF → 5-methylTHF still
    leads back to the blocked step; folinic acid bypasses completely.
  Protein restriction: NOT needed (MMA arm intact — no propionyl-CoA accumulation).
  L-Carnitine 50–100 mg/kg/day (Level B): mild secondary carnitine depletion possible but less
    severe than cblC since no MMA-driven propionylcarnitine excretion.
  VPA: HIGH RISK — both carnitine depletion AND folate depletion are worsened; LEV first-line ALL.
  N2O: ABSOLUTE CI — MTR is ALREADY COMPROMISED in cblE; N2O oxidizes MTR cofactor → complete
    MTR failure → acute severe HHcy crisis → LIFE-THREATENING.
  Fasting: MODERATE RISK — less severe than cblC (no MMA arm surge) but worsens HHcy via
    decreased dietary methionine and increased protein catabolism → Hcy accumulation.

DISTINGUISHING FEATURES (vs other HHcy disorders):
  vs cblG (MTR): IDENTICAL biochemistry — same isolated HHcy, same MMA normal, same methylfolate
    trap, same low methionine. ONLY difference: MTRR (reductase absent) vs MTR (synthase absent).
    Fibroblast cobalamin coenzyme synthesis + complementation assay distinguishes.
    cblE fibroblasts complement cblG fibroblasts (different complementation groups).
  vs cblC (MMACHC): cblC has COMBINED MMA+HHcy — MMA ELEVATED in cblC, NORMAL in cblE. AdoCbl
    ABSENT in cblC fibroblasts — NORMAL in cblE. NBS detects cblC (C3 elevated) — misses cblE (C3 normal).
  vs CBS (cystathionine β-synthase): HHcy elevated in both — METHIONINE HIGH in CBS (blocked
    transsulfuration → methionine accumulates) vs METHIONINE LOW in cblE (blocked remethylation).
    MMA normal in both. No megaloblastic anemia in CBS. CBS responds to pyridoxine (B6).
  vs cblD-HHcy (MMADHC N-terminus): IDENTICAL biochemistry — isolated HHcy, normal MMA.
    Gene panel (MMADHC vs MTRR) distinguishes — no clinical/biochemical difference between subtypes.
  vs MTHFR deficiency: HHcy elevated in both; MTHFR has NORMAL serum folate (not elevated) and
    NO megaloblastic anemia (thymidylate synthesis preserved via MTHFR-independent THF routes).
    MMA normal in both. MTHFR → low 5-methylTHF; cblE → trapped 5-methylTHF (high serum folate).

OMIM: Gene *602568 (MTRR) · Disease #236270 (Homocystinuria-megaloblastic anemia, cblE complementation type)
Chromosome: 5p15.31 · Inheritance: AUTOSOMAL RECESSIVE (biallelic LOF)
Prevalence: ~1:300,000–500,000 (rare; <200 cases worldwide 2026; likely underdiagnosed;
NBS misses cblE completely because C3 is normal — all diagnoses made clinically after symptom onset)
"""

from __future__ import annotations
import random

random.seed(58)   # reproducible synthetic cohort — seed 58 for cblE/MTRR


# ── helpers ───────────────────────────────────────────────────────────────────

def _pid(i: int) -> str:
    return f"MTRR-{i:03d}"


def _sex(i: int) -> str:
    # Autosomal recessive: both sexes equally affected — alternating male/female
    return "M" if i % 2 == 0 else "F"


def _phenotype(i: int) -> str:
    """
    Neonatal Severe (~35%): <1 month; megaloblastic anemia, severe HHcy, hypotonia, neonatal seizures
    Infantile Classic (~45%): 1–18 months; developmental delay, hypotonia, HHcy, anemia, infantile spasms — MODAL
    Late-Onset Attenuated (~20%): 18 months–adult; cognitive impairment, psychiatric features, SCD, partial activity
    """
    if i < 14:
        return "Neonatal Severe"
    elif i < 32:
        return "Infantile Classic"
    else:
        return "Late-Onset Attenuated"


def _genotype(i: int, pheno: str) -> str:
    pheno_variants = {
        "Neonatal Severe": [
            "p.Arg415*/hemizygous (nonsense; FAD domain null; no MTRR protein; severe)",
            "p.Arg415Gln/p.Arg415Gln (c.1244G>A; homozygous; FAD domain; poor OHCbl response)",
            "c.903+469T>C/p.Arg415* (UK/Irish pseudoexon founder + null; compound het; NMD + truncation)",
            "p.Arg415Gln/c.903+469T>C (FAD domain + deep intronic pseudoexon; compound het; severe)",
        ],
        "Infantile Classic": [
            "p.Ser175Leu/p.Ser175Leu (c.524C>T; homozygous; FMN domain; most common European cblE)",
            "p.Ser175Leu/p.Arg415Gln (c.524C>T + c.1244G>A; compound het; moderate-severe)",
            "p.Ser175Leu/c.903+469T>C (FMN domain + pseudoexon; compound het; infantile classic)",
            "p.Thr353Ile/p.Arg415Gln (c.1058C>T + c.1244G>A; compound het; FAD domain; IS)",
            "p.Ser175Leu/p.Arg174Gln (c.524C>T + c.521G>A; FMN domain compound het; moderate)",
            "p.Thr353Ile/p.Ser175Leu (c.1058C>T + c.524C>T; compound het; moderate-severe IS)",
        ],
        "Late-Onset Attenuated": [
            "p.Arg174Gln/p.Ser175Leu (c.521G>A + c.524C>T; FMN domain; partial activity retained)",
            "p.Arg174Gln/p.Arg174Gln (c.521G>A; homozygous; FMN domain; best prognosis)",
            "p.Thr353Ile/p.Arg174Gln (c.1058C>T + c.521G>A; compound het; attenuated; partial MTRR activity)",
        ],
    }
    opts = pheno_variants[pheno]
    return opts[i % len(opts)]


def _mma_urine(pheno: str) -> float:
    # MMA NORMAL in ALL cblE patients — MMA arm (MMUT/AdoCbl) is completely intact
    # Report as normal (<5 mmol/mol Cr) across all phenotypes
    return round(random.uniform(0.5, 4.5), 1)


def _homocysteine(pheno: str) -> float:
    if pheno == "Neonatal Severe":
        return round(random.uniform(80, 200), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(40, 150), 1)
    else:
        return round(random.uniform(25, 90), 1)


def _methionine(pheno: str) -> float:
    if pheno == "Neonatal Severe":
        return round(random.uniform(5, 10), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(7, 15), 1)
    else:
        return round(random.uniform(10, 20), 1)


def _c3(pheno: str) -> float:
    # C3 NORMAL in cblE — NBS invisible — all phenotypes have normal C3
    return round(random.uniform(0.5, 2.8), 1)


def _serum_folate(pheno: str) -> float:
    # Folate HIGH (methylfolate trap) — serum folate elevated despite functional deficiency
    if pheno == "Neonatal Severe":
        return round(random.uniform(25, 60), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(18, 50), 1)
    else:
        return round(random.uniform(15, 40), 1)


def _mcv(pheno: str) -> int:
    # MCV elevated — megaloblastic anemia (macrocytosis)
    if pheno == "Neonatal Severe":
        return random.randint(100, 125)
    elif pheno == "Infantile Classic":
        return random.randint(98, 118)
    else:
        return random.randint(95, 110)


def _hemoglobin(pheno: str) -> float:
    # Hemoglobin low — megaloblastic anemia
    if pheno == "Neonatal Severe":
        return round(random.uniform(4.5, 8.0), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(6.0, 10.0), 1)
    else:
        return round(random.uniform(8.0, 12.5), 1)


def _free_carnitine(pheno: str) -> float:
    # Mild secondary depletion possible; less severe than cblC (no MMA arm)
    if pheno == "Neonatal Severe":
        return round(random.uniform(18, 35), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(22, 42), 1)
    else:
        return round(random.uniform(28, 52), 1)


def _onset_months(pheno: str) -> int:
    if pheno == "Neonatal Severe":
        return random.randint(0, 1)
    elif pheno == "Infantile Classic":
        return random.randint(1, 18)
    else:
        return random.randint(18, 120)


def _ohcbl_response(pheno: str) -> bool:
    # OHCbl response: 60–75% overall — BETTER than cblX, similar to cblC
    rates = {"Neonatal Severe": 0.52, "Infantile Classic": 0.68, "Late-Onset Attenuated": 0.82}
    return random.random() < rates[pheno]


def _seizures(pheno: str) -> bool:
    # ~60% overall; predominantly myoclonic + infantile spasms (early-onset); focal (late-onset)
    rates = {"Neonatal Severe": 0.78, "Infantile Classic": 0.65, "Late-Onset Attenuated": 0.35}
    return random.random() < rates[pheno]


def _nbs(pheno: str) -> bool:
    # NBS INVISIBLE for cblE — C3 normal; almost never detected by NBS
    rates = {"Neonatal Severe": 0.04, "Infantile Classic": 0.03, "Late-Onset Attenuated": 0.02}
    return random.random() < rates[pheno]


def _megaloblastic_anemia(pheno: str) -> bool:
    # Megaloblastic anemia: hallmark of cblE — methylfolate trap → impaired DNA synthesis
    rates = {"Neonatal Severe": 0.95, "Infantile Classic": 0.88, "Late-Onset Attenuated": 0.70}
    return random.random() < rates[pheno]


def _aed(sz: bool, pheno: str) -> str:
    if not sz:
        return "None"
    if pheno == "Neonatal Severe":
        opts = [
            "LEV + PB (neonatal myoclonic; OHCbl + folinic acid commenced)",
            "LEV + PB + folinic acid rescue (neonatal seizures; megaloblastic crisis)",
            "LEV + CLB (neonatal myoclonic; betaine + folinic acid Level A)",
            "LEV + PB (neonatal HHcy seizures; OHCbl IM initiated)",
        ]
    elif pheno == "Infantile Classic":
        opts = [
            "LEV + ACTH (infantile spasms; OHCbl + betaine + folinic acid)",
            "LEV + VIG (IS; folinic acid Level A added)",
            "LEV + CLB (myoclonic; betaine + folinic acid maintained)",
            "LEV + ACTH + CLB (IS cluster; metabolic + AED combined approach)",
        ]
    else:
        opts = [
            "LEV (focal cognitive seizures; late-onset cblE)",
            "LEV + LTG (focal seizures; psychiatric features; adult onset)",
            "LEV + CLB (cognitive impairment + seizures; SCD workup)",
        ]
    return random.choice(opts)


def _diag_months(pheno: str, onset: int) -> int:
    delays = {
        "Neonatal Severe": (0, 3),
        "Infantile Classic": (2, 12),
        "Late-Onset Attenuated": (6, 36),
    }
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
            "mma_urine_mmol_molCr": _mma_urine(pheno),       # NORMAL in ALL cblE — MMA arm intact
            "total_homocysteine_umol_l": _homocysteine(pheno),
            "methionine_umol_l": _methionine(pheno),
            "c3_umol_l": _c3(pheno),                          # NORMAL — NBS invisible
            "serum_folate_nmol_l": _serum_folate(pheno),      # HIGH (methylfolate trap)
            "mcv_fl": _mcv(pheno),                            # ELEVATED — megaloblastic
            "hemoglobin_g_dl": _hemoglobin(pheno),            # LOW — megaloblastic anemia
            "free_carnitine_umol_l": _free_carnitine(pheno),
            "mecbl_fibroblasts": "Absent",    # MTR cannot be reactivated → MeCbl not synthesized
            "adocbl_fibroblasts": "Normal",   # KEY POSITIVE — MMAB arm intact; MMUT unaffected
            "mma_normal": True,               # KEY NEGATIVE: MMA arm completely unaffected
            "adocbl_normal": True,            # KEY POSITIVE: AdoCbl normal in fibroblasts
            "c3_normal": True,                # KEY NEGATIVE: NBS invisible (C3 not elevated)
            "megaloblastic_anemia": _megaloblastic_anemia(pheno),
            "seizures": sz,
            "nbs_detected": _nbs(pheno),      # Near-zero NBS detection — C3 normal
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
    fol_v = [p["serum_folate_nmol_l"] for p in _PATIENTS]
    hgb_v = [p["hemoglobin_g_dl"] for p in _PATIENTS]
    car_v = [p["free_carnitine_umol_l"] for p in _PATIENTS]

    def pct(pred): return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    neo_pts  = [p for p in _PATIENTS if p["phenotype"] == "Neonatal Severe"]
    inf_pts  = [p for p in _PATIENTS if p["phenotype"] == "Infantile Classic"]
    late_pts = [p for p in _PATIENTS if p["phenotype"] == "Late-Onset Attenuated"]

    return {
        "gene": "MTRR",
        "full_name": (
            "5-methyltetrahydrofolate-homocysteine methyltransferase reductase — "
            "MTR Reactivation Enzyme / FMN+FAD Diflavin Reductase (cblE gene product)"
        ),
        "chromosome": "5p15.31",
        "inheritance": "AUTOSOMAL RECESSIVE (biallelic LOF; both sexes equally affected)",
        "omim_gene": "*602568",
        "omim_disease": "#236270",
        "protein_size": (
            "698 aa; FMN+FAD diflavin reductase; cytochrome P450 reductase (CPR) superfamily; "
            "FMN-binding N-terminal domain (AA 1–175) + FAD-binding C-terminal domain (AA 176–698); "
            "electron flow: NADPH → FAD → FMN → cob(II)alamin on MTR → cob(I)alamin (active MTR)"
        ),
        "prevalence": (
            "~1:300,000–500,000 (rare; <200 cases worldwide 2026; likely underdiagnosed; "
            "NBS misses ALL cblE — C3 normal; all diagnoses after symptom onset)"
        ),
        "nbs_primary": (
            "NOT DETECTED by standard NBS — C3 normal (MMA arm intact); "
            "cblE is among the most commonly missed inborn errors by NBS; "
            "diagnosed after presentation with megaloblastic anemia + HHcy or neurological symptoms"
        ),
        "nbs_secondary": (
            "Plasma tHcy + plasma amino acids (methionine low) + serum folate (high — methylfolate trap) + "
            "CBC with MCV (megaloblastic); urine MMA NORMAL (KEY NEGATIVE vs cblC); "
            "fibroblast cobalamin synthesis: MeCbl absent + AdoCbl normal = cblE or cblG pattern; "
            "complementation assay distinguishes cblE (MTRR) from cblG (MTR); "
            "WES/WGS for MTRR (5p15.31) biallelic variants mandatory"
        ),
        "function": (
            "MTRR reactivates Methionine Synthase (MTR/cblG) by reducing the inactive cob(II)alamin "
            "cofactor of MTR back to the active cob(I)alamin form, using electrons from NADPH via "
            "FAD → FMN → cob(II)alamin. Without MTRR, MTR slowly and irreversibly accumulates the "
            "inactive cob(II)alamin form and cannot remethylate homocysteine. "
            "MTRR acts ONLY on the MeCbl/MTR arm (Step 4b) — the AdoCbl/MMUT arm (Step 4a) is "
            "completely independent of MTRR. This produces ISOLATED HHcy with NORMAL MMA."
        ),
        "mechanism": (
            "MTRR LOF → MTR cofactor (cob(II)alamin) cannot be reactivated → "
            "MTR accumulates inactive cob(II)alamin form → Hcy cannot be remethylated → "
            "tHcy ELEVATED (isolated — MMA NORMAL KEY NEGATIVE). "
            "Simultaneously: MTR inactivity traps 5-methylTHF (cannot donate methyl group) → "
            "methylfolate trap → functional THF depletion → megaloblastic anemia. "
            "Serum folate appears HIGH (methylfolate trap — trapped 5-methylTHF measured by folate assay) "
            "despite functional folate deficiency — the 'folate paradox' of cblE. "
            "KEY POSITIVE: AdoCbl NORMAL in fibroblasts (MMAB arm entirely unaffected by MTRR LOF)."
        ),
        "key_negative": (
            "MMA NORMAL: KEY NEGATIVE vs cblC/cblX (combined disorders) — MMA arm (MMUT/AdoCbl) intact; "
            "C3 NORMAL: KEY NEGATIVE for NBS — cblE is NBS-INVISIBLE (C3 not elevated); "
            "AdoCbl NORMAL in fibroblasts: KEY POSITIVE differentiating cblE from cblC; "
            "Methionine LOW (not high as in CBS) — remethylation arm blocked, not transsulfuration; "
            "Protein restriction NOT needed (no MMA, no propionyl-CoA overload); "
            "I22M (rs1801394) — common polymorphism (~25% allele freq) is NOT pathogenic — NOT cblE."
        ),
        "cohort_n": n,
        "kpis": {
            "avg_mma_urine_mmol_molCr": round(sum(mma_v) / n, 1),     # should be ~2 (NORMAL)
            "avg_homocysteine_umol_l": round(sum(hcy_v) / n, 1),
            "avg_methionine_umol_l": round(sum(met_v) / n, 1),
            "avg_c3_umol_l": round(sum(c3_v) / n, 1),                 # should be ~1.5 (NORMAL)
            "avg_serum_folate_nmol_l": round(sum(fol_v) / n, 1),      # HIGH (methylfolate trap)
            "avg_hemoglobin_g_dl": round(sum(hgb_v) / n, 1),
            "avg_free_carnitine_umol_l": round(sum(car_v) / n, 1),
            "pct_seizures": pct(lambda p: p["seizures"]),
            "pct_megaloblastic_anemia": pct(lambda p: p["megaloblastic_anemia"]),
            "pct_nbs_detected": pct(lambda p: p["nbs_detected"]),      # near-zero expected
            "pct_ohcbl_response": pct(lambda p: p["ohcbl_response"]),
            "pct_mma_normal": 100,       # ALL cblE: MMA arm intact
            "pct_adocbl_normal": 100,    # ALL cblE: AdoCbl fibroblasts normal
            "pct_c3_normal": 100,        # ALL cblE: NBS invisible
        },
        "phenotype_distribution": {
            "neonatal_severe": {"n": len(neo_pts), "pct": round(len(neo_pts) / n * 100)},
            "infantile_classic": {"n": len(inf_pts), "pct": round(len(inf_pts) / n * 100)},
            "late_onset_attenuated": {"n": len(late_pts), "pct": round(len(late_pts) / n * 100)},
        },
    }


def get_breakdown() -> dict:
    n = len(_PATIENTS)

    def pct(pred): return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    seizure_types = [
        {"type": "Myoclonic seizures (neonatal/infantile)", "pct": pct(
            lambda p: p["seizures"] and p["phenotype"] in ("Neonatal Severe", "Infantile Classic"))},
        {"type": "Infantile Spasms (West Syndrome pattern)", "pct": pct(
            lambda p: p["seizures"] and p["phenotype"] == "Infantile Classic")},
        {"type": "Focal seizures (late-onset / cognitive)", "pct": pct(
            lambda p: p["seizures"] and p["phenotype"] == "Late-Onset Attenuated")},
        {"type": "Generalized tonic-clonic", "pct": 22},
        {"type": "Absence seizures (late-onset)", "pct": 12},
    ]

    metabolic_triggers = [
        {"trigger": "N2O exposure (anesthesia)", "pct": 100,
         "mechanism": (
             "Irreversibly oxidizes the cob(I)alamin cofactor of MTR → "
             "MTR already compromised in cblE (MTRR absent) → N2O causes COMPLETE MTR failure → "
             "acute severe HHcy crisis. LIFE-THREATENING. "
             "Notify anesthesiology at EVERY procedure."
         )},
        {"trigger": "Folate-depleting drugs (methotrexate, trimethoprim, phenytoin)", "pct": 88,
         "mechanism": (
             "Dihydrofolate reductase inhibition further depletes THF pool already depleted by "
             "methylfolate trap → acute megaloblastic crisis + worsening HHcy. "
             "Folinic acid rescue mandatory. Never give antifolates without folinic acid co-administration."
         )},
        {"trigger": "Missed OHCbl + betaine + folinic acid doses", "pct": 72,
         "mechanism": (
             "tHcy rebounds rapidly (~48 h after missed betaine/OHCbl); "
             "megaloblastic anemia worsens within days after missed folinic acid; "
             "all THREE treatments are mandatory and synergistic — none may be omitted."
         )},
        {"trigger": "Intercurrent illness / fever (catabolism)", "pct": 58,
         "mechanism": (
             "Protein catabolism → Hcy generation exceeds BHMT + partial OHCbl-rescued MTR capacity → "
             "tHcy surge. IV glucose helpful; less MMA surge risk than cblC (no MMA arm) "
             "but HHcy crisis remains serious."
         )},
        {"trigger": "Fasting / inadequate dietary methionine", "pct": 45,
         "mechanism": (
             "Decreased dietary methionine input + catabolic Hcy release → "
             "worsens HHcy when MTR/MTRR axis is compromised. "
             "Moderate risk (less severe than cblC — no MMA arm); maintain regular protein intake."
         )},
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
        "folate": p["serum_folate_nmol_l"],
        "mcv": p["mcv_fl"],
        "hgb": p["hemoglobin_g_dl"],
        "car": p["free_carnitine_umol_l"],
        "mma_normal": p["mma_normal"],
        "adocbl_normal": p["adocbl_normal"],
        "megaloblastic": p["megaloblastic_anemia"],
        "sz": p["seizures"],
        "nbs": p["nbs_detected"],
        "ohcbl_resp": p["ohcbl_response"],
        "aed": p["aed"],
    } for p in _PATIENTS[:sample_size]]

    return {
        "seizure_type_distribution": seizure_types,
        "metabolic_triggers": metabolic_triggers,
        "high_risk_drugs": [
            {
                "drug": "Nitrous oxide (N2O)",
                "risk": "ABSOLUTE CI",
                "mechanism": (
                    "Irreversibly oxidizes cobalt cofactor of MTR → complete Methionine Synthase failure. "
                    "In cblE, MTR is ALREADY dependent on compromised MTRR for reactivation — "
                    "N2O pushes already-struggling MTR into complete inactivation → "
                    "acute catastrophic HHcy crisis. LIFE-THREATENING. "
                    "All anesthesia providers must be notified: N2O ABSOLUTE CONTRAINDICATION in ALL cblE patients."
                ),
            },
            {
                "drug": "Valproate (VPA)",
                "risk": "HIGH RISK",
                "mechanism": (
                    "VPA causes both carnitine depletion (secondary to VPA-carnitine conjugation) "
                    "AND folate depletion (VPA is a folate antagonist) — worsens both metabolic arms of cblE. "
                    "In a patient already requiring folinic acid (Level A), VPA-driven folate depletion "
                    "risks acute megaloblastic crisis. LEV is first-line for ALL seizure types in MTRR/cblE."
                ),
            },
            {
                "drug": "Antifolates (methotrexate, trimethoprim, pyrimethamine, phenytoin)",
                "risk": "HIGH RISK",
                "mechanism": (
                    "Any antifolate drug depletes available THF pool further in a patient already "
                    "suffering from methylfolate trap (functional THF depletion). "
                    "Risk of acute megaloblastic anemia crisis + worsening HHcy. "
                    "If antifolate unavoidable: mandatory folinic acid co-administration and tHcy monitoring."
                ),
            },
            {
                "drug": "Fasting / NPO without adequate nutrition",
                "risk": "MODERATE RISK",
                "mechanism": (
                    "Fasting increases protein catabolism → Hcy release exceeds BHMT + compromised MTR capacity. "
                    "Less severe than cblC (no MMA arm surge) but tHcy elevation during fasting "
                    "still risks neurological deterioration. "
                    "Maintain regular protein intake; IV dextrose during illness or NPO periods."
                ),
            },
        ],
        "treatments": [
            {
                "treatment": "Hydroxocobalamin (OHCbl) 1–2 mg/day IM",
                "evidence": "Level A",
                "response_pct": 68,
                "note": (
                    "Provides cobalamin cofactor in excess such that alternative MTRR-independent pathways "
                    "can partially synthesize MeCbl for MTR. Response: 60–75% HHcy reduction — "
                    "BETTER than cblX (~40–60%), similar to cblC (~75%). "
                    "Does not restore MTRR enzyme itself; high-dose OHCbl partially compensates via "
                    "mass-action cob(II)→cob(I) equilibrium shift. "
                    "Lifelong IM mandatory; monitor tHcy + methionine at Day 7 and monthly."
                ),
            },
            {
                "treatment": "Betaine (TMG) 100–200 mg/kg/day",
                "evidence": "Level A",
                "response_pct": 74,
                "note": (
                    "BHMT-mediated cobalamin-independent Hcy remethylation (Hcy → Methionine). "
                    "MANDATORY — essential synergistic partner to OHCbl. "
                    "BHMT operates entirely independently of MTR/MTRR axis — bypasses the block. "
                    "Continue throughout life regardless of apparent HHcy control."
                ),
            },
            {
                "treatment": "Folinic acid 0.5–5 mg/day",
                "evidence": "Level A",
                "note": (
                    "LEVEL A in cblE — MORE IMPORTANT than in combined disorders (cblC, cblX). "
                    "Methylfolate trap is the CENTRAL pathological mechanism in cblE: MTR inactive → "
                    "5-methylTHF cannot donate its methyl group → 5-methylTHF accumulates → "
                    "functional THF depletion → megaloblastic anemia. "
                    "Folinic acid (5-formyl-THF) bypasses the methylfolate trap completely by "
                    "entering the THF pool directly — corrects anemia + DNA synthesis impairment. "
                    "In cblC/cblX: folate trap is secondary; in cblE it is PRIMARY — folinic acid is Level A here."
                ),
                "response_pct": 82,
            },
            {
                "treatment": "Folic acid (supplemental)",
                "evidence": "Level B",
                "response_pct": 55,
                "note": (
                    "Some evidence for megaloblastic anemia correction; however, folinic acid is preferred. "
                    "Folic acid must first be converted folic → dihydrofolate → THF → 5,10-methyleneTHF → "
                    "5-methylTHF → then trapped again at the MTR block. "
                    "Folinic acid enters the cycle as 5-formyl-THF, directly bypassing the blocked step. "
                    "Use folinic acid as primary; folic acid as adjunct if folinic acid unavailable."
                ),
            },
            {
                "treatment": "L-Carnitine 50–100 mg/kg/day",
                "evidence": "Level B",
                "response_pct": 60,
                "note": (
                    "Mild secondary carnitine depletion possible in cblE (less severe than cblC "
                    "since no MMA-driven propionylcarnitine excretion). "
                    "Monitor free carnitine; supplement if low. "
                    "Level B — not as critical as in combined MMA+HHcy disorders."
                ),
            },
            {
                "treatment": "IV Glucose / Dextrose (acute crisis)",
                "evidence": "Level A",
                "response_pct": 78,
                "note": (
                    "Acute metabolic crisis management: IV dextrose to suppress catabolism + "
                    "Hcy release. Continue OHCbl IM + betaine + folinic acid during crisis. "
                    "Less MMA-crisis risk than cblC but tHcy crisis from catabolism is real. "
                    "Mandatory during any illness, surgery, or NPO episode."
                ),
            },
            {
                "treatment": "ACTH / Vigabatrin (infantile spasms)",
                "evidence": "Level A (IS)",
                "response_pct": 58,
                "note": (
                    "ACTH or vigabatrin for infantile spasms pattern (West syndrome). "
                    "Combined with OHCbl + betaine + folinic acid. "
                    "IS in cblE is primarily metabolic (HHcy + functional folate deficiency) — "
                    "metabolic correction often improves seizure control significantly."
                ),
            },
            {
                "treatment": "VPA — HIGH RISK / AVOID",
                "evidence": "AVOID",
                "response_pct": 0,
                "note": (
                    "HIGH RISK: VPA depletes both carnitine AND folate — worsens BOTH metabolic "
                    "vulnerabilities of cblE (folate depletion worsens methylfolate trap; "
                    "carnitine depletion in patients already on folinic acid regimen). "
                    "LEV is unconditionally first-line AED for ALL seizure types in MTRR/cblE."
                ),
            },
        ],
        "patient_sample": sample,
    }


def get_definitions() -> dict:
    return {
        "gene_card": {
            "Gene": (
                "MTRR (5-methyltetrahydrofolate-homocysteine methyltransferase reductase; "
                "cblE gene product; also CXORF3, MSR)"
            ),
            "Full Name": (
                "Homocystinuria-megaloblastic anemia, cblE complementation type "
                "(MTRR; Complementation Group cblE)"
            ),
            "Chromosome": "5p15.31",
            "Inheritance": "AUTOSOMAL RECESSIVE — biallelic LOF; both sexes equally affected",
            "OMIM Gene": "*602568",
            "OMIM Disease": "#236270 (Homocystinuria-megaloblastic anemia, cblE complementation type)",
            "Protein Size": (
                "698 amino acids; FMN+FAD diflavin reductase; cytochrome P450 reductase superfamily; "
                "FMN-binding N-terminal domain (AA 1–175) + FAD-binding C-terminal domain (AA 176–698)"
            ),
            "Protein Function": (
                "Reactivates Methionine Synthase (MTR) by transferring electrons from NADPH "
                "via FAD→FMN→cob(II)alamin on MTR → cob(I)alamin (active MTR cofactor); "
                "acts ONLY on MeCbl/MTR arm — AdoCbl/MMUT arm is completely independent"
            ),
            "Pathway Position": (
                "Step 4b reactivation — MTRR reactivates MTR after each catalytic cycle; "
                "without MTRR, MTR slowly inactivates → isolated HHcy + methylfolate trap"
            ),
            "Key Warning": (
                "I22M polymorphism (c.66A>G, rs1801394, ~25% allele frequency) is NOT pathogenic — "
                "NOT cblE; do not report as disease-causing; only confirmed pathogenic variants constitute cblE"
            ),
            "Prevalence": (
                "~1:300,000–500,000 (rare; <200 cases worldwide 2026; "
                "likely underdiagnosed due to NBS invisibility)"
            ),
        },
        "key_concepts": [
            {
                "concept": "MTRR reactivates MTR — the cblE mechanism",
                "explanation": (
                    "MTR (Methionine Synthase, cblG gene) catalyzes: "
                    "5-methylTHF + Hcy → THF + Methionine, using cob(I)alamin as cofactor. "
                    "During each catalytic cycle, cob(I)alamin is occasionally oxidized to "
                    "inactive cob(II)alamin (spontaneous oxidation ~1 in 2000 cycles). "
                    "MTRR reactivates MTR by docking on the MTR enzyme and transferring "
                    "electrons: NADPH → FAD (MTRR C-terminal) → FMN (MTRR N-terminal) → "
                    "cob(II)alamin on MTR → cob(I)alamin (active MTR). "
                    "MTRR LOF → MTR accumulates inactive cob(II)alamin → "
                    "Hcy cannot be remethylated → ISOLATED HHcy. "
                    "Crucially: the MMAB/AdoCbl/MMUT arm is completely independent of MTRR → "
                    "MMA NORMAL (KEY NEGATIVE vs combined disorders)."
                ),
            },
            {
                "concept": "Methylfolate trap — central mechanism in cblE (Level A for folinic acid)",
                "explanation": (
                    "MTR uses 5-methylTHF as the methyl donor (5-methylTHF → MTR → THF + Methionine). "
                    "When MTR is inactive (MTRR LOF), 5-methylTHF cannot donate its methyl group → "
                    "5-methylTHF ACCUMULATES (methylfolate trap). "
                    "Consequences: (1) THF pool depleted → impaired thymidylate synthesis → "
                    "megaloblastic anemia (macrocytic, hypersegmented neutrophils). "
                    "(2) Serum folate appears HIGH (methylfolate trap creates high 5-methylTHF) — "
                    "the 'folate paradox': folate assay measures all folates including trapped "
                    "5-methylTHF → serum folate HIGH despite functional folate deficiency. "
                    "Folinic acid (5-formyl-THF) bypasses the trap by entering THF cycle directly "
                    "→ Level A evidence in cblE (more important than in cblC/cblX)."
                ),
            },
            {
                "concept": "MMA NORMAL — KEY NEGATIVE: MMA arm completely intact in cblE",
                "explanation": (
                    "MTRR acts exclusively on MTR (MeCbl arm, Step 4b). "
                    "The AdoCbl arm (Step 4a: MMADHC → MMAB → AdoCbl → MMUT → MMA catabolism) "
                    "is entirely upstream of MTRR and completely unaffected. "
                    "Clinical implication: urine MMA NORMAL (<5 mmol/mol Cr) in ALL cblE patients. "
                    "This is the PRIMARY distinguishing feature from cblC (MMACHC) and cblX (HCFC1) "
                    "where combined MMA+HHcy is the hallmark. "
                    "If a patient has ELEVATED MMA + HHcy → NOT cblE → look for cblC, cblX, cblD-combined. "
                    "Protein restriction is also NOT needed in cblE (no propionyl-CoA overload)."
                ),
            },
            {
                "concept": "AdoCbl NORMAL in fibroblasts — KEY POSITIVE differentiating cblE from cblC",
                "explanation": (
                    "In cblC (MMACHC LOF): MMACHC cannot process cobalamin → "
                    "BOTH MeCbl AND AdoCbl absent in fibroblasts. "
                    "In cblE (MTRR LOF): MeCbl ABSENT (MTR cannot be reactivated) BUT "
                    "AdoCbl NORMAL (MMAB arm unaffected). "
                    "Fibroblast cobalamin coenzyme synthesis assay: "
                    "MeCbl absent + AdoCbl normal = cblE or cblG pattern. "
                    "Distinguishes from cblC (both absent) and is the KEY biochemical "
                    "test before genetics results return. "
                    "Complementation assay then separates cblE (MTRR) from cblG (MTR)."
                ),
            },
            {
                "concept": "NBS INVISIBLE — cblE is the most commonly missed cobalamin disorder by NBS",
                "explanation": (
                    "Standard NBS measures C3 (propionylcarnitine) as the primary cobalamin disorder screen. "
                    "C3 is elevated when the MMA arm is blocked (cblA, cblB, cblC, cblX, cblD-combined). "
                    "In cblE: MMA arm is COMPLETELY INTACT → C3 is NORMAL → NBS is NEGATIVE. "
                    "ALL cblE patients who are diagnosed clinically present after symptom onset: "
                    "megaloblastic anemia in infancy, developmental regression, or in late-onset cases "
                    "cognitive decline or psychiatric features in older children/adults. "
                    "Countries with expanded NBS including homocysteine still miss some cblE "
                    "because presentation is variable. Clinical awareness essential."
                ),
            },
            {
                "concept": "OHCbl response 60–75% — better than cblX, similar to cblC",
                "explanation": (
                    "OHCbl IM provides cobalamin cofactor in excess — high-dose OHCbl partially "
                    "compensates for MTRR absence via mass-action shift of cob(II)→cob(I) equilibrium: "
                    "at very high cofactor concentrations, even without MTRR-mediated reactivation, "
                    "some spontaneous reduction occurs sufficient to partially sustain MTR activity. "
                    "Additionally, alternative MTRR-independent minor reactivation pathways may contribute. "
                    "Response: 60–75% HHcy reduction. Better than cblX (~40–60% — no MMACHC protein scaffold). "
                    "Important: betaine is MANDATORY alongside OHCbl — "
                    "BHMT pathway provides cobalamin-independent HHcy control even when OHCbl response is partial."
                ),
            },
            {
                "concept": "cblE vs cblG — biochemically identical, different genes",
                "explanation": (
                    "cblE (MTRR LOF) and cblG (MTR LOF) produce IDENTICAL biochemistry: "
                    "isolated HHcy, normal MMA, low methionine, methylfolate trap, megaloblastic anemia, "
                    "MeCbl absent + AdoCbl normal in fibroblasts. "
                    "The ONLY distinction is the gene defect: "
                    "cblE = MTRR (reductase absent — MTR enzyme present but cannot be reactivated); "
                    "cblG = MTR (synthase absent — MTR enzyme not made). "
                    "Fibroblast complementation assay: cblE + cblG fibroblasts fused → "
                    "complementation occurs (different complementation groups) → distinguishes. "
                    "WES/WGS (MTRR 5p15.31 vs MTR 1q43) conclusively separates the two. "
                    "Treatment is the same for both (OHCbl + betaine + folinic acid Level A)."
                ),
            },
            {
                "concept": "I22M polymorphism (rs1801394) is NOT pathogenic — not cblE",
                "explanation": (
                    "p.Ile22Met (c.66A>G, rs1801394) is a common MTRR variant with allele frequency "
                    "~25% in European populations (~6% homozygosity). "
                    "Early literature erroneously associated I22M with neural tube defects and "
                    "mild hyperhomocysteinemia — subsequent large studies showed it is a "
                    "population polymorphism without clinical significance in isolation. "
                    "I22M must NOT be reported as a pathogenic cblE variant. "
                    "True cblE variants (p.Ser175Leu, p.Arg415Gln, p.Arg174Gln, c.903+469T>C, "
                    "p.Arg415*, p.Thr353Ile) are rare and cause complete loss of MTRR function. "
                    "In WES reports: I22M in trans with a true pathogenic variant does not count as "
                    "compound heterozygosity unless the second allele is independently confirmed pathogenic."
                ),
            },
            {
                "concept": "N2O ABSOLUTE CI — MTR already compromised, N2O causes complete MTR failure",
                "explanation": (
                    "Nitrous oxide (N2O) irreversibly oxidizes the cob(I)alamin cofactor of MTR "
                    "to cob(III)alamin (inactive, cannot be reactivated by MTRR). "
                    "In a healthy person, MTRR eventually reactivates MTR after N2O clearance. "
                    "In cblE: MTRR is ABSENT → even partial cob(III)alamin oxidation by N2O "
                    "cannot be reversed → MTR is permanently inactivated → "
                    "acute catastrophic HHcy crisis → severe neurological deterioration. "
                    "N2O ABSOLUTE CONTRAINDICATION in ALL cblE patients — "
                    "notify anesthesiology, ICU, and emergency teams at every encounter. "
                    "Alternative anesthetics: propofol, sevoflurane, isoflurane (no cobalamin cofactor interaction)."
                ),
            },
        ],
        "diagnostic_thresholds": [
            {
                "parameter": "Plasma tHcy",
                "threshold": ">40 µmol/L (isolated — MMA normal)",
                "action": (
                    "Isolated HHcy + normal MMA → cblE (MTRR), cblG (MTR), or cblD-HHcy (MMADHC N-term) differential. "
                    "Check plasma methionine (low → cblE/cblG/cblD-HHcy; high → CBS). "
                    "Check serum folate (high = methylfolate trap → cblE/cblG). "
                    "Start OHCbl + betaine + folinic acid empirically while awaiting genetics."
                ),
            },
            {
                "parameter": "Urine MMA",
                "threshold": "NORMAL (<5 mmol/mol Cr) in cblE — KEY NEGATIVE",
                "action": (
                    "MMA normal + HHcy elevated → ISOLATED HHcy pattern. Excludes cblC, cblX, cblD-combined. "
                    "Differential: cblE (MTRR), cblG (MTR), cblD-HHcy-only (MMADHC N-term), CBS. "
                    "If MMA elevated + HHcy elevated → combined disorder (cblC, cblX, cblF, cblJ, cblD-combined)."
                ),
            },
            {
                "parameter": "Serum folate",
                "threshold": "ELEVATED (>25 nmol/L) despite clinical folate deficiency",
                "action": (
                    "High serum folate + megaloblastic anemia + HHcy = methylfolate trap = cblE or cblG pattern. "
                    "This is the 'folate paradox': serum folate HIGH but functionally depleted. "
                    "Do NOT withhold folinic acid because serum folate is high — the functional deficiency is real. "
                    "Start folinic acid Level A immediately."
                ),
            },
            {
                "parameter": "MCV (mean corpuscular volume)",
                "threshold": ">98 fL (macrocytosis)",
                "action": (
                    "Macrocytic megaloblastic anemia + elevated tHcy + low methionine + high serum folate = cblE/cblG. "
                    "Review blood smear for hypersegmented neutrophils + macro-ovalocytes (megaloblastic morphology). "
                    "Folinic acid + OHCbl commenced; folate trap must be relieved — monitor CBC weekly."
                ),
            },
            {
                "parameter": "MeCbl fibroblasts",
                "threshold": "ABSENT (with AdoCbl NORMAL)",
                "action": (
                    "MeCbl absent + AdoCbl normal = cblE (MTRR) or cblG (MTR) fibroblast pattern. "
                    "Complementation assay with known cblE and cblG fibroblasts distinguishes. "
                    "WES/WGS of MTRR (5p15.31) + MTR (1q43) confirms gene. "
                    "Compare: cblC → MeCbl AND AdoCbl both absent (MMACHC upstream of both arms)."
                ),
            },
            {
                "parameter": "NBS C3",
                "threshold": "NORMAL (<3.5 µmol/L) — cblE NOT detected by NBS",
                "action": (
                    "Normal NBS C3 does NOT exclude cblE. cblE is NBS-INVISIBLE. "
                    "Any infant or child with megaloblastic anemia + unexplained neurological "
                    "regression + high serum folate needs plasma tHcy measured urgently — "
                    "do not be reassured by normal NBS result."
                ),
            },
            {
                "parameter": "OHCbl trial: tHcy",
                "threshold": "60–75% reduction expected after 7 days",
                "action": (
                    "Better response than cblX (~40–60%); similar to cblC (~75%). "
                    "If <60% HHcy reduction at day 7: maximize betaine dose; add folinic acid; "
                    "review adherence. Continue OHCbl regardless — partial response is still beneficial. "
                    "Monitor methionine concurrently — rising methionine confirms MTR activation."
                ),
            },
            {
                "parameter": "EEG: infantile spasms pattern (hypsarrhythmia)",
                "threshold": "Any infant with IS + megaloblastic anemia + HHcy",
                "action": (
                    "Infantile spasms + megaloblastic anemia + elevated tHcy + normal MMA = cblE/cblG. "
                    "Metabolic correction with OHCbl + betaine + folinic acid often reduces IS burden. "
                    "Add ACTH or vigabatrin for IS control per standard protocol. "
                    "LEV first-line AED; never VPA (folate-depleting in already folate-deficient patient)."
                ),
            },
        ],
        "differential_diagnosis": [
            {
                "disease": "cblG (MTR — 1q43)",
                "distinguishing": (
                    "Biochemically IDENTICAL to cblE — isolated HHcy, normal MMA, low methionine, "
                    "methylfolate trap, megaloblastic anemia, MeCbl absent + AdoCbl normal in fibroblasts. "
                    "ONLY distinction: gene defect — cblG = MTR absent (synthase absent); "
                    "cblE = MTRR absent (reductase absent, MTR enzyme present). "
                    "Fibroblast complementation assay (cblE + cblG fuse → complement = different groups) "
                    "or WES/WGS (MTR 1q43 vs MTRR 5p15.31) distinguishes. "
                    "Treatment identical: OHCbl + betaine + folinic acid Level A for both."
                ),
            },
            {
                "disease": "cblC (MMACHC — 1p34.1)",
                "distinguishing": (
                    "KEY DIFFERENCE: cblC has COMBINED MMA+HHcy — MMA ELEVATED in cblC, NORMAL in cblE. "
                    "AdoCbl ABSENT in cblC fibroblasts (MMACHC blocks both arms) — "
                    "NORMAL in cblE (only MeCbl arm affected). "
                    "NBS detects cblC (C3 elevated) — misses cblE (C3 normal). "
                    "Maculopathy in cblC (~80%) — absent in cblE. "
                    "Folinic acid is Level B in cblC (secondary methylfolate trap) — "
                    "Level A in cblE (methylfolate trap is PRIMARY mechanism)."
                ),
            },
            {
                "disease": "CBS — Classical Homocystinuria (21q22.3)",
                "distinguishing": (
                    "KEY DIFFERENCE: methionine HIGH in CBS (transsulfuration blocked → Met accumulates) "
                    "vs methionine LOW in cblE (remethylation blocked → Met not synthesized). "
                    "MMA normal in both (neither affects MMA arm). "
                    "CBS: NO megaloblastic anemia (THF cycle intact in CBS); "
                    "serum folate NORMAL in CBS (no methylfolate trap); "
                    "CBS responds to pyridoxine (B6) in ~50% of patients (cofactor of CBS enzyme); "
                    "cblE does not respond to B6. "
                    "Marfanoid habitus, lens dislocation in CBS — absent in cblE. "
                    "MTHFR deficiency: also HHcy + normal MMA but low folate (not high as in cblE)."
                ),
            },
            {
                "disease": "cblD-HHcy-only (MMADHC N-terminus — 2q23.2)",
                "distinguishing": (
                    "IDENTICAL biochemistry: isolated HHcy, normal MMA, low methionine. "
                    "MecbL absent + AdoCbl normal in fibroblasts — same as cblE. "
                    "Methylfolate trap may occur in cblD-HHcy too (secondary). "
                    "ONLY distinction: gene — MMADHC (2q23.2) N-terminus variants vs MTRR (5p15.31). "
                    "Gene panel distinguishes; no clinical or biochemical differentiation possible. "
                    "Both: OHCbl + betaine Level A; folinic acid."
                ),
            },
            {
                "disease": "MTHFR deficiency (1p36.3)",
                "distinguishing": (
                    "HHcy elevated + MMA normal (same as cblE). "
                    "KEY DIFFERENCES: serum folate NORMAL or LOW in MTHFR (5-methylTHF LOW because "
                    "MTHFR cannot make it) vs HIGH in cblE (5-methylTHF trapped = HIGH). "
                    "NO megaloblastic anemia in MTHFR (thymidylate synthesis via MTHFR-independent "
                    "THF routes is preserved); megaloblastic anemia = hallmark of cblE. "
                    "MTHFR responds to betaine + riboflavin; does NOT respond to OHCbl alone "
                    "(cobalamin not the primary defect in MTHFR). "
                    "MCV: macrocytic in cblE; MCV typically normal in MTHFR."
                ),
            },
            {
                "disease": "cblX (HCFC1 — Xq28)",
                "distinguishing": (
                    "KEY DIFFERENCE: cblX has COMBINED MMA+HHcy (MMACHC not expressed → both arms blocked). "
                    "cblE has ISOLATED HHcy (MMA normal — only MeCbl arm affected). "
                    "cblX is X-linked (males only); cblE is AR (both sexes equally). "
                    "OHCbl response worse in cblX (~40–60%) vs cblE (~60–75%). "
                    "Folinic acid Level B in cblX (secondary folate trap); Level A in cblE (primary). "
                    "NBS detects cblX (C3 elevated); misses cblE (C3 normal)."
                ),
            },
            {
                "disease": "Megaloblastic anemia — Vitamin B12 nutritional deficiency",
                "distinguishing": (
                    "Nutritional B12 deficiency (vegan/vegetarian diet, maternal deficiency) causes "
                    "HHcy + megaloblastic anemia — same as cblE. "
                    "KEY DIFFERENCES: (1) serum B12 LOW in nutritional deficiency; "
                    "NORMAL or HIGH in cblE (cobalamin uptake from diet is normal). "
                    "(2) Responds rapidly and completely to IM B12 replacement in nutritional deficiency; "
                    "cblE: partial response only (MTRR enzyme absent — B12 supplementation alone insufficient). "
                    "(3) Fibroblast cobalamin synthesis: NORMAL in nutritional B12 deficiency; "
                    "MeCbl absent in cblE fibroblasts. "
                    "WES/WGS distinguishes if needed after B12 supplementation trial."
                ),
            },
        ],
    }
