#!/usr/bin/env python3
"""MTR (Methionine Synthase Deficiency / Homocystinuria-Megaloblastic Anemia cblG type) Epilepsy Dashboard.

MTR encodes 5-methyltetrahydrofolate-homocysteine methyltransferase, a 1265-aa cytoplasmic
methionine synthase at 1q43 that directly catalyzes the remethylation of homocysteine (Hcy)
to methionine, using methylcobalamin (MeCbl) as the essential cofactor and 5-methylTHF as
the methyl donor. MTR is the effector enzyme of the cytoplasmic cobalamin arm; its
reactivation partner MTRR (cblE) provides the electrons required to keep MTR's cofactor active.

COBALAMIN PATHWAY — MTR IS THE TERMINAL CYTOPLASMIC ENZYME (Step 4b):
  Step 0: Dietary cobalamin bound to TC2 → CD320 receptor → endocytosis → lysosome.
  Step 1: LMBRD1 (cblF) + ABCD4 (cblJ) export cobalamin from lysosome → cytoplasm.
  Step 2: MMACHC (cblC) converts cobalamin → cob(I)alamin (universal intermediate).
  Step 3: MMADHC distributes cob(I)alamin:
          ARM A (C-terminus, mitochondrial) → MMAB → AdoCbl → MMUT (MMA catabolism).
          ARM B (N-terminus, cytoplasmic)   → MTR → MeCbl (Hcy remethylation → Methionine).
  Step 4a: MMUT uses AdoCbl to catabolize methylmalonyl-CoA → succinyl-CoA (MMA arm).
  Step 4b: MTR (cblG) uses MeCbl to remethylate Hcy → Methionine (HHcy arm).
           ← cblG BLOCK: MTR LOF → enzyme absent → Hcy cannot be remethylated →
              tHcy ELEVATED → Methionine LOW. MMA ARM (Step 4a) COMPLETELY INTACT.
           ← MTRR relationship: MTRR (cblE) reactivates MTR after each catalytic cycle
              by reducing cob(II)alamin back to cob(I)alamin. In cblG, MTR enzyme itself is
              ABSENT — MTRR is present and functional but has nothing to reactivate.

MTR PROTEIN BIOLOGY:
  1265 amino acids; cytoplasmic methionine synthase; ~139 kDa; cobalamin-dependent enzyme.
  N-terminal cobalamin-binding Rossmann fold domain (AA 1–348):
    Binds methylcobalamin (MeCbl) as cofactor; His759 is the axial cobalt ligand that
    coordinates the lower face of the corrin ring; mutations here reduce cobalamin affinity
    and destabilize the cofactor-enzyme interaction.
  Central catalytic β/α-barrel domain (AA 349–700):
    Contains the active site for methyl transfer from 5-methylTHF to Hcy;
    5-methylTHF docks here and donates its methyl group to cob(I)alamin → MeCbl;
    MeCbl then donates methyl to Hcy → Methionine; catalytic residues Asp757, His759, Cys759.
  C-terminal activation domain (AA 701–1265):
    Docking site for MTRR (cblE reductase); MTRR docks on this domain to deliver electrons
    from NADPH via FAD→FMN→cob(II)alamin on MTR → cob(I)alamin (reactivation).
    Variants here (most common: p.Asp919Gly) disrupt MTRR docking → MTR cannot be reactivated
    even though MTRR is present — functionally similar phenotype to cblE.

MTR LOF RESULT — ISOLATED HHcy (MMA arm intact — KEY NEGATIVE):
  → tHcy ELEVATED (40–200 µmol/L): MTR synthase absent → Hcy cannot be remethylated
    → ISOLATED HHcy. NO MMA. Same biochemical profile as cblE (MTRR).
  → MMA NORMAL: KEY NEGATIVE — MMUT/AdoCbl arm entirely unaffected by MTR LOF.
    MMAB, MMAA, MMUT act upstream and independently of MTR.
  → Methionine LOW (5–18 µmol/L): MTR cannot synthesize methionine from Hcy →
    methionine depleted → downstream SAM/methylation reactions impaired.
  → MeCbl ABSENT in fibroblasts: MTR enzyme absent → cannot incorporate cobalamin as MeCbl
    cofactor; fibroblast assay shows absent MeCbl synthesis.
  → AdoCbl NORMAL in fibroblasts: KEY POSITIVE — MMAB arm (AdoCbl synthesis) is independent
    of MTR; MMAB and MMUT are completely unaffected by MTR LOF.
  → C3 NORMAL on NBS: MMA arm intact → C3 not elevated → NBS INVISIBLE for cblG.
    Same NBS invisibility as cblE — all cblG diagnoses made clinically after symptom onset.
  → Megaloblastic anemia: methylfolate trap — MTR absent → 5-methylTHF cannot donate methyl
    group to become THF → 5-methylTHF trapped → functional THF depletion → impaired DNA
    synthesis → megaloblastic anemia (macrocytic, hypersegmented neutrophils, elevated MCV).
  → Folate paradox: serum folate HIGH (methylfolate trap — 5-methylTHF accumulates in plasma)
    despite functional folate deficiency. Identical to cblE.

METHYLFOLATE TRAP (central in cblG — Level A for folinic acid, same as cblE):
  Normal: 5-methylTHF → MTR → THF + Methionine (THF re-enters folate cycle for DNA synthesis).
  cblG: MTR enzyme ABSENT → 5-methylTHF cannot donate methyl group → 5-methylTHF accumulates
        → serum folate appears high (lab artifact) → but functionally ZERO THF available
        → megaloblastic anemia + impaired thymidylate synthesis.
  The methylfolate trap in cblG is mechanistically identical to cblE (MTRR LOF) — in both cases
  the terminal enzyme (MTR) that consumes 5-methylTHF is non-functional.
  Folinic acid (5-formyl-THF) BYPASSES this block by entering the folate cycle as THF directly
  → more important in cblG (and cblE) than in combined disorders (cblC, cblX) where folate
  trap is secondary to global cobalamin processing failure.

KEY BIOCHEMICAL DIFFERENCE: cblG vs cblE:
  cblG (MTR LOF): The SYNTHASE itself (MTR enzyme) is absent. MTRR is present and functional
    but has nothing to reactivate — no MTR enzyme substrate for MTRR to act upon.
  cblE (MTRR LOF): The REDUCTASE (MTRR enzyme) is absent. MTR enzyme is present but cannot be
    reactivated after cofactor oxidation → MTR progressively inactivates.
  OHCbl response: Slightly LESS effective in cblG (~50–65% HHcy reduction) vs cblE (~60–75%)
    because in cblG there is NO MTR enzyme at all — exogenous cobalamin cannot be incorporated
    into an absent enzyme, whereas in cblE the MTR enzyme is present and some spontaneous
    mass-action reactivation of cob(II)→cob(I) can occur.
  Complementation assay DISTINGUISHES: cblE fibroblasts + cblG fibroblasts fused → complement
    each other (different complementation groups) → confirms they are distinct genetic entities.
  Treatment: Identical for both — OHCbl + betaine + folinic acid Level A.

MTRR RELATIONSHIP (normal physiology):
  After each catalytic cycle of MTR, cob(I)alamin is oxidized ~1/2000 cycles to cob(II)alamin.
  MTRR (cblE gene product) reductively reactivates MTR: NADPH → FAD → FMN → cob(II)alamin
  on MTR → cob(I)alamin. In cblG: MTR is ABSENT → MTRR has no substrate → MTRR sits idle.
  Exogenous OHCbl cannot rescue absent MTR (no enzyme to deliver cobalamin to) but at very high
  concentrations may provide mass-action partial rescue via trace-level alternative methyltransferases.

MTR BIOMARKERS:
  Plasma tHcy: 40–200 µmol/L ELEVATED (isolated — MMA NORMAL KEY NEGATIVE)
  Urine MMA: NORMAL (<5 mmol/mol Cr) — same critical distinguishing feature as cblE vs cblC
  Plasma methionine: 5–18 µmol/L LOW (MTR absent → no methionine synthesis from Hcy)
  Serum folate: HIGH (methylfolate trap — 5-methylTHF accumulates; assay reads falsely elevated)
  MeCbl fibroblasts: ABSENT (MTR enzyme absent → cannot incorporate MeCbl cofactor)
  AdoCbl fibroblasts: NORMAL — KEY POSITIVE (MMAB arm independent of MTR)
  C3 (propionylcarnitine): NORMAL — KEY NEGATIVE (no MMA → C3 not elevated → NBS misses cblG)
  CBC: megaloblastic anemia — macrocytic + hypersegmented neutrophils; elevated MCV

KEY VARIANTS IN MTR (cblG pathogenic):
  p.Asp919Gly (c.2756A>G): C-terminal activation domain (AA 919); most common cblG allele;
    moderate-severe phenotype; impairs MTRR docking → MTR cannot be reactivated;
    MTR enzyme present but cofactor progressively oxidized without MTRR rescue.
  p.Pro1173Leu (c.3518C>T): C-terminal activation domain (AA 1173); severe null-like;
    complete disruption of activation domain; neonatal encephalopathy; poor OHCbl response.
  p.Arg553His (c.1658G>A): catalytic β/α-barrel domain (AA 553); moderate; some residual MTR
    activity; infantile spasms phenotype; intermediate OHCbl response.
  p.Thr289Ile (c.866C>T): cobalamin-binding Rossmann fold domain (AA 289); moderate;
    reduced cobalamin binding affinity → unstable cofactor-enzyme interaction.
  c.IVS6+1G>A (splice site): null allele; severe NMD of mRNA; neonatal encephalopathy;
    no functional MTR mRNA; complete loss of function.
  p.Arg585Gln (c.1754G>A): activation domain (AA 585); moderate-attenuated; partial MTRR
    docking retained in some conformations; best prognosis among cblG alleles; late-onset phenotype.

OMIM: Gene *156570 (MTR) · Disease #250940 (Homocystinuria-megaloblastic anemia, cblG complementation type)
Chromosome: 1q43 · Inheritance: AUTOSOMAL RECESSIVE (biallelic LOF; both sexes equally affected)
Prevalence: <100 cases worldwide 2026 (rarer than cblE; NBS misses ALL cblG — C3 normal;
all diagnoses made clinically after symptom onset; likely underdiagnosed)
"""

from __future__ import annotations
import random

random.seed(59)   # reproducible synthetic cohort — seed 59 for cblG/MTR


# ── helpers ───────────────────────────────────────────────────────────────────

def _pid(i: int) -> str:
    return f"MTR-{i:03d}"


def _sex(i: int) -> str:
    # Autosomal recessive: both sexes equally affected — alternating male/female
    return "M" if i % 2 == 0 else "F"


def _phenotype(i: int) -> str:
    """
    Early-Onset Severe (~30%): <3 months; neonatal encephalopathy, megaloblastic anemia,
      severe HHcy, neonatal seizures.
    Infantile Classic (~50%): 3–18 months; developmental delay, hypotonia, HHcy, infantile spasms — MODAL.
    Late-Onset Attenuated (~20%): 18 months–adult; cognitive impairment, psychiatric features,
      SCD risk, partial activity (e.g. p.Arg585Gln).
    """
    if i < 12:
        return "Early-Onset Severe"
    elif i < 32:
        return "Infantile Classic"
    else:
        return "Late-Onset Attenuated"


def _genotype(i: int, pheno: str) -> str:
    pheno_variants = {
        "Early-Onset Severe": [
            "p.Pro1173Leu/p.Pro1173Leu (c.3518C>T; homozygous; C-terminal activation domain null; neonatal encephalopathy)",
            "c.IVS6+1G>A/p.Pro1173Leu (splice null + activation domain null; compound het; NMD + truncation)",
            "c.IVS6+1G>A/c.IVS6+1G>A (homozygous splice null; severe NMD; no functional MTR mRNA; neonatal)",
            "p.Pro1173Leu/p.Asp919Gly (c.3518C>T + c.2756A>G; compound het; activation domain; severe)",
        ],
        "Infantile Classic": [
            "p.Asp919Gly/p.Asp919Gly (c.2756A>G; homozygous; most common cblG allele; MTRR-docking defect; moderate-severe)",
            "p.Asp919Gly/p.Arg553His (c.2756A>G + c.1658G>A; compound het; activation + catalytic domain; IS)",
            "p.Asp919Gly/p.Thr289Ile (c.2756A>G + c.866C>T; compound het; activation + Rossmann fold; moderate)",
            "p.Arg553His/p.Thr289Ile (c.1658G>A + c.866C>T; compound het; catalytic + cobalamin-binding; IS)",
            "p.Asp919Gly/c.IVS6+1G>A (most common allele + splice null; compound het; moderate-severe IS)",
            "p.Arg553His/p.Pro1173Leu (c.1658G>A + c.3518C>T; compound het; catalytic + null; moderate-severe)",
        ],
        "Late-Onset Attenuated": [
            "p.Arg585Gln/p.Asp919Gly (c.1754G>A + c.2756A>G; compound het; partial MTRR docking retained; attenuated)",
            "p.Arg585Gln/p.Arg585Gln (c.1754G>A; homozygous; best prognosis cblG; partial activation domain activity)",
            "p.Arg585Gln/p.Thr289Ile (c.1754G>A + c.866C>T; compound het; partial activity; late cognitive phenotype)",
            "p.Thr289Ile/p.Thr289Ile (c.866C>T; homozygous Rossmann fold; reduced cobalamin affinity; adult-onset)",
        ],
    }
    opts = pheno_variants[pheno]
    return opts[i % len(opts)]


def _mma_urine(pheno: str) -> float:
    # MMA NORMAL in ALL cblG patients — MMA arm (MMUT/AdoCbl) is completely intact
    # Report as normal (<5 mmol/mol Cr) across all phenotypes
    return round(random.uniform(0.4, 4.5), 1)


def _homocysteine(pheno: str) -> float:
    if pheno == "Early-Onset Severe":
        return round(random.uniform(80, 200), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(40, 155), 1)
    else:
        return round(random.uniform(25, 90), 1)


def _methionine(pheno: str) -> float:
    # MTR absent → methionine not synthesized from Hcy → LOW
    # Slightly lower range than cblE because no MTR enzyme at all
    if pheno == "Early-Onset Severe":
        return round(random.uniform(5, 9), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(7, 14), 1)
    else:
        return round(random.uniform(9, 18), 1)


def _c3(pheno: str) -> float:
    # C3 NORMAL in cblG — NBS invisible — all phenotypes have normal C3
    return round(random.uniform(0.4, 2.9), 1)


def _serum_folate(pheno: str) -> float:
    # Folate HIGH (methylfolate trap) — serum folate elevated despite functional deficiency
    if pheno == "Early-Onset Severe":
        return round(random.uniform(26, 62), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(18, 52), 1)
    else:
        return round(random.uniform(15, 42), 1)


def _mcv(pheno: str) -> int:
    # MCV elevated — megaloblastic anemia (macrocytosis)
    if pheno == "Early-Onset Severe":
        return random.randint(100, 128)
    elif pheno == "Infantile Classic":
        return random.randint(98, 120)
    else:
        return random.randint(95, 112)


def _hemoglobin(pheno: str) -> float:
    # Hemoglobin low — megaloblastic anemia; slightly more severe than cblE due to complete MTR absence
    if pheno == "Early-Onset Severe":
        return round(random.uniform(4.0, 7.5), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(5.5, 9.5), 1)
    else:
        return round(random.uniform(7.5, 12.0), 1)


def _free_carnitine(pheno: str) -> float:
    # Mild secondary depletion possible; less severe than cblC (no MMA arm)
    if pheno == "Early-Onset Severe":
        return round(random.uniform(17, 34), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(21, 42), 1)
    else:
        return round(random.uniform(27, 52), 1)


def _onset_months(pheno: str) -> int:
    if pheno == "Early-Onset Severe":
        return random.randint(0, 3)
    elif pheno == "Infantile Classic":
        return random.randint(3, 18)
    else:
        return random.randint(18, 144)


def _ohcbl_response(pheno: str) -> bool:
    # OHCbl response in cblG: ~50–65% HHcy reduction — SLIGHTLY LESS than cblE (60–75%)
    # because MTR enzyme itself is absent; exogenous cobalamin cannot reactivate absent enzyme
    rates = {"Early-Onset Severe": 0.42, "Infantile Classic": 0.60, "Late-Onset Attenuated": 0.75}
    return random.random() < rates[pheno]


def _seizures(pheno: str) -> bool:
    # ~60% overall; neonatal seizures (early-onset); infantile spasms (infantile); focal (late)
    rates = {"Early-Onset Severe": 0.82, "Infantile Classic": 0.62, "Late-Onset Attenuated": 0.32}
    return random.random() < rates[pheno]


def _nbs(pheno: str) -> bool:
    # NBS INVISIBLE for cblG — C3 normal; almost never detected by NBS
    rates = {"Early-Onset Severe": 0.03, "Infantile Classic": 0.03, "Late-Onset Attenuated": 0.02}
    return random.random() < rates[pheno]


def _megaloblastic_anemia(pheno: str) -> bool:
    # Megaloblastic anemia: hallmark of cblG — methylfolate trap → impaired DNA synthesis
    # Slightly higher rate than cblE because complete enzyme absence = more severe trap
    rates = {"Early-Onset Severe": 0.97, "Infantile Classic": 0.90, "Late-Onset Attenuated": 0.72}
    return random.random() < rates[pheno]


def _aed(sz: bool, pheno: str) -> str:
    if not sz:
        return "None"
    if pheno == "Early-Onset Severe":
        opts = [
            "LEV + PB (neonatal encephalopathy seizures; OHCbl + folinic acid commenced urgently)",
            "LEV + PB + folinic acid rescue (neonatal megaloblastic crisis + seizures)",
            "LEV + CLB (neonatal myoclonic; betaine + folinic acid Level A initiated)",
            "LEV + PB (neonatal HHcy seizures; OHCbl IM started; metabolic correction priority)",
        ]
    elif pheno == "Infantile Classic":
        opts = [
            "LEV + ACTH (infantile spasms; OHCbl + betaine + folinic acid synergistic)",
            "LEV + VIG (IS; folinic acid Level A added; methylfolate trap primary mechanism)",
            "LEV + CLB (myoclonic; betaine + folinic acid maintained; OHCbl IM lifelong)",
            "LEV + ACTH + CLB (IS cluster; metabolic + AED combined; folinic acid mandatory)",
        ]
    else:
        opts = [
            "LEV (focal cognitive seizures; late-onset cblG; OHCbl + betaine continued)",
            "LEV + LTG (focal seizures; psychiatric features; adult-onset MTR deficiency)",
            "LEV + CLB (cognitive impairment + seizures; SCD workup; betaine + folinic acid)",
        ]
    return random.choice(opts)


def _diag_months(pheno: str, onset: int) -> int:
    delays = {
        "Early-Onset Severe": (0, 3),
        "Infantile Classic": (2, 14),
        "Late-Onset Attenuated": (6, 48),
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
            "mma_urine_mmol_molCr": _mma_urine(pheno),        # NORMAL in ALL cblG — MMA arm intact
            "total_homocysteine_umol_l": _homocysteine(pheno),
            "methionine_umol_l": _methionine(pheno),
            "c3_umol_l": _c3(pheno),                           # NORMAL — NBS invisible
            "serum_folate_nmol_l": _serum_folate(pheno),       # HIGH (methylfolate trap)
            "mcv_fl": _mcv(pheno),                             # ELEVATED — megaloblastic
            "hemoglobin_g_dl": _hemoglobin(pheno),             # LOW — megaloblastic anemia
            "free_carnitine_umol_l": _free_carnitine(pheno),
            "mecbl_fibroblasts": "Absent",    # MTR enzyme absent → cannot incorporate MeCbl cofactor
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

    neo_pts  = [p for p in _PATIENTS if p["phenotype"] == "Early-Onset Severe"]
    inf_pts  = [p for p in _PATIENTS if p["phenotype"] == "Infantile Classic"]
    late_pts = [p for p in _PATIENTS if p["phenotype"] == "Late-Onset Attenuated"]

    return {
        "gene": "MTR",
        "full_name": (
            "5-methyltetrahydrofolate-homocysteine methyltransferase — "
            "Methionine Synthase / Cytoplasmic Cobalamin-Dependent Remethylase (cblG gene product)"
        ),
        "chromosome": "1q43",
        "inheritance": "AUTOSOMAL RECESSIVE (biallelic LOF; both sexes equally affected)",
        "omim_gene": "*156570",
        "omim_disease": "#250940",
        "protein_size": (
            "1265 aa; ~139 kDa; cytoplasmic methionine synthase; cobalamin-dependent; "
            "N-terminal cobalamin-binding Rossmann fold domain (AA 1–348, His759 axial cobalt ligand) + "
            "central catalytic β/α-barrel domain (AA 349–700, methyl transfer active site) + "
            "C-terminal activation domain (AA 701–1265, MTRR docking site for cofactor reactivation)"
        ),
        "prevalence": (
            "<100 cases worldwide 2026 (rarer than cblE/MTRR; likely underdiagnosed; "
            "NBS misses ALL cblG — C3 normal; all diagnoses after symptom onset)"
        ),
        "nbs_primary": (
            "NOT DETECTED by standard NBS — C3 normal (MMA arm intact); "
            "cblG shares NBS invisibility with cblE (MTRR) — both are isolated HHcy disorders; "
            "diagnosed after presentation with megaloblastic anemia + HHcy or neurological symptoms"
        ),
        "nbs_secondary": (
            "Plasma tHcy + plasma amino acids (methionine low) + serum folate (high — methylfolate trap) + "
            "CBC with MCV (megaloblastic); urine MMA NORMAL (KEY NEGATIVE vs cblC/combined disorders); "
            "fibroblast cobalamin synthesis: MeCbl absent + AdoCbl normal = cblG or cblE pattern; "
            "complementation assay distinguishes cblG (MTR) from cblE (MTRR) — different complementation groups; "
            "WES/WGS for MTR (1q43) biallelic variants mandatory"
        ),
        "function": (
            "MTR directly catalyzes remethylation of homocysteine to methionine: "
            "5-methylTHF + Hcy → THF + Methionine, using MeCbl as the essential methyl-transfer cofactor. "
            "MTR is the terminal enzyme of the cytoplasmic cobalamin arm (Step 4b); "
            "its reactivation is maintained by MTRR (cblE), which reduces the cob(II)alamin cofactor "
            "back to active cob(I)alamin. In cblG, the MTR enzyme itself is absent — "
            "MTRR is intact but has no substrate to reactivate. "
            "MTR acts ONLY on the MeCbl/HHcy arm — the AdoCbl/MMUT arm (Step 4a) is "
            "completely independent of MTR. This produces ISOLATED HHcy with NORMAL MMA."
        ),
        "mechanism": (
            "MTR LOF → methionine synthase enzyme absent → Hcy cannot be remethylated → "
            "tHcy ELEVATED (isolated — MMA NORMAL KEY NEGATIVE). "
            "Simultaneously: MTR absence traps 5-methylTHF (no enzyme to accept methyl donation) → "
            "methylfolate trap → functional THF depletion → megaloblastic anemia. "
            "Serum folate appears HIGH (methylfolate trap — trapped 5-methylTHF measured by folate assay) "
            "despite functional folate deficiency — the 'folate paradox' of cblG (identical to cblE). "
            "OHCbl response SLIGHTLY LESS in cblG (~50–65%) than cblE (~60–75%) because no MTR enzyme "
            "to deliver exogenous cobalamin to; mass-action rescue requires at least some enzyme scaffold. "
            "KEY POSITIVE: AdoCbl NORMAL in fibroblasts (MMAB arm entirely unaffected by MTR LOF). "
            "MTRR (cblE) is present and functional in cblG but has no MTR substrate to reactivate."
        ),
        "key_negative": (
            "MMA NORMAL: KEY NEGATIVE vs cblC/cblX (combined disorders) — MMA arm (MMUT/AdoCbl) intact; "
            "C3 NORMAL: KEY NEGATIVE for NBS — cblG is NBS-INVISIBLE (C3 not elevated); "
            "AdoCbl NORMAL in fibroblasts: KEY POSITIVE differentiating cblG from cblC; "
            "Methionine LOW (not high as in CBS) — remethylation arm absent, not transsulfuration; "
            "Protein restriction NOT needed (no MMA, no propionyl-CoA overload — same as cblE); "
            "MTRR NORMAL in cblG — the reductase is functional, the synthase itself is absent."
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
            "pct_mma_normal": 100,       # ALL cblG: MMA arm intact
            "pct_adocbl_normal": 100,    # ALL cblG: AdoCbl fibroblasts normal
            "pct_c3_normal": 100,        # ALL cblG: NBS invisible
        },
        "phenotype_distribution": {
            "early_onset_severe": {"n": len(neo_pts), "pct": round(len(neo_pts) / n * 100)},
            "infantile_classic": {"n": len(inf_pts), "pct": round(len(inf_pts) / n * 100)},
            "late_onset_attenuated": {"n": len(late_pts), "pct": round(len(late_pts) / n * 100)},
        },
    }


def get_breakdown() -> dict:
    n = len(_PATIENTS)

    def pct(pred): return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    seizure_types = [
        {"type": "Neonatal seizures (encephalopathy, early-onset severe)", "pct": pct(
            lambda p: p["seizures"] and p["phenotype"] == "Early-Onset Severe")},
        {"type": "Infantile Spasms (West Syndrome pattern)", "pct": pct(
            lambda p: p["seizures"] and p["phenotype"] == "Infantile Classic")},
        {"type": "Myoclonic seizures (infantile / early-onset)", "pct": pct(
            lambda p: p["seizures"] and p["phenotype"] in ("Early-Onset Severe", "Infantile Classic"))},
        {"type": "Focal seizures (late-onset / cognitive decline)", "pct": pct(
            lambda p: p["seizures"] and p["phenotype"] == "Late-Onset Attenuated")},
        {"type": "Generalized tonic-clonic", "pct": 18},
        {"type": "Absence seizures (late-onset)", "pct": 10},
    ]

    metabolic_triggers = [
        {"trigger": "N2O exposure (anesthesia)", "pct": 100,
         "mechanism": (
             "Irreversibly oxidizes the cob(I)alamin cofactor of any residual MTR activity → "
             "in cblG, MTR enzyme is ABSENT — any trace alternative methyltransferase activity "
             "using cobalamin is also permanently blocked by N2O → acute catastrophic HHcy crisis. "
             "LIFE-THREATENING. Notify anesthesiology at EVERY procedure without exception."
         )},
        {"trigger": "Folate-depleting drugs (methotrexate, trimethoprim, phenytoin)", "pct": 88,
         "mechanism": (
             "Dihydrofolate reductase inhibition further depletes THF pool already depleted by "
             "methylfolate trap (5-methylTHF trapped by absent MTR) → acute megaloblastic crisis + "
             "worsening HHcy. Folinic acid rescue mandatory. "
             "Never give antifolates without folinic acid co-administration in cblG."
         )},
        {"trigger": "Missed OHCbl + betaine + folinic acid doses", "pct": 74,
         "mechanism": (
             "tHcy rebounds rapidly (~48 h after missed betaine/OHCbl); "
             "megaloblastic anemia worsens within days after missed folinic acid; "
             "all THREE treatments are mandatory and synergistic — none may be omitted. "
             "OHCbl response in cblG is partial (~50–65%) so betaine becomes especially critical."
         )},
        {"trigger": "Intercurrent illness / fever (catabolism)", "pct": 60,
         "mechanism": (
             "Protein catabolism → Hcy generation exceeds BHMT capacity → "
             "tHcy surge (MTR absent, no remethylation possible without OHCbl mass-action rescue). "
             "IV glucose helpful; less MMA surge risk than cblC (no MMA arm) but "
             "HHcy crisis remains serious and can cause acute neurological deterioration."
         )},
        {"trigger": "Fasting / inadequate dietary methionine", "pct": 48,
         "mechanism": (
             "Decreased dietary methionine input + catabolic Hcy release → "
             "worsens HHcy when MTR axis is absent (no remethylation enzyme). "
             "Moderate risk (less severe than cblC — no MMA arm); maintain regular protein intake; "
             "IV dextrose during illness or prolonged NPO periods."
         )},
    ]

    sample_size = min(6, n)
    sample = [{
        "id": p["patient_id"],
        "sex": p["sex"],
        "phenotype": p["phenotype"][:24],
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
                    "Irreversibly oxidizes cobalt cofactor of cobalamin-dependent enzymes. "
                    "In cblG, MTR enzyme is ABSENT — MTRR is present but has no MTR to reactivate. "
                    "N2O exposure destroys any trace alternative cobalamin-dependent methyltransferase "
                    "activity and permanently blocks all residual HHcy remethylation capacity. "
                    "Acute catastrophic HHcy crisis → severe neurological deterioration → LIFE-THREATENING. "
                    "All anesthesia, ICU, and emergency providers must be notified: "
                    "N2O ABSOLUTE CONTRAINDICATION in ALL MTR/cblG patients. "
                    "Alternatives: propofol, sevoflurane, isoflurane (no cobalamin cofactor interaction)."
                ),
            },
            {
                "drug": "Valproate (VPA)",
                "risk": "HIGH RISK",
                "mechanism": (
                    "VPA causes both carnitine depletion (VPA-carnitine conjugation → secondary carnitine "
                    "deficiency) AND folate depletion (VPA is a folate antagonist — inhibits folate "
                    "absorption and increases folate catabolism). "
                    "In a cblG patient already requiring folinic acid (Level A) to relieve methylfolate trap, "
                    "VPA-driven folate depletion risks acute megaloblastic crisis and worsening HHcy. "
                    "LEV is unconditionally first-line AED for ALL seizure types in MTR/cblG."
                ),
            },
            {
                "drug": "Antifolates (methotrexate, trimethoprim, pyrimethamine, phenytoin)",
                "risk": "HIGH RISK",
                "mechanism": (
                    "Any antifolate drug depletes the already critically low THF pool in a patient "
                    "suffering from methylfolate trap (functional THF depletion because MTR absent). "
                    "Risk of acute megaloblastic anemia crisis + worsening HHcy. "
                    "If antifolate unavoidable (e.g. serious infection requiring trimethoprim): "
                    "mandatory folinic acid co-administration + urgent tHcy and CBC monitoring."
                ),
            },
            {
                "drug": "Fasting / NPO without adequate nutrition",
                "risk": "MODERATE RISK",
                "mechanism": (
                    "Fasting increases protein catabolism → Hcy release; with MTR absent, "
                    "remethylation capacity relies entirely on BHMT (betaine) + partial OHCbl effect. "
                    "Less severe than cblC (no MMA arm surge) but tHcy elevation during extended "
                    "fasting still risks neurological deterioration. "
                    "Maintain regular protein intake; IV dextrose + continue betaine + OHCbl during illness."
                ),
            },
        ],
        "treatments": [
            {
                "treatment": "Hydroxocobalamin (OHCbl) 1–2 mg/day IM",
                "evidence": "Level A",
                "response_pct": 60,
                "note": (
                    "Provides cobalamin cofactor; at high concentrations, mass-action effects and "
                    "alternative minor methyltransferases provide partial rescue. "
                    "Response: ~50–65% HHcy reduction — SLIGHTLY LESS than cblE (~60–75%) because "
                    "MTR enzyme itself is absent in cblG; OHCbl cannot reactivate an absent enzyme, "
                    "whereas in cblE the MTR enzyme is present and mass-action shift of cob(II)→cob(I) "
                    "can partially sustain existing MTR molecules. "
                    "In cblG, betaine becomes especially critical as the primary HHcy control. "
                    "Lifelong IM mandatory; monitor tHcy + methionine at Day 7 and monthly."
                ),
            },
            {
                "treatment": "Betaine (TMG) 100–200 mg/kg/day",
                "evidence": "Level A",
                "response_pct": 76,
                "note": (
                    "BHMT-mediated cobalamin-independent Hcy remethylation (Hcy → Methionine via BHMT). "
                    "MANDATORY — in cblG, betaine is particularly critical because OHCbl response is "
                    "slightly less complete than cblE; BHMT completely bypasses the absent MTR enzyme. "
                    "Continue throughout life regardless of apparent HHcy control. "
                    "BHMT operates entirely independently of the MTR/MTRR axis."
                ),
            },
            {
                "treatment": "Folinic acid 0.5–5 mg/day",
                "evidence": "Level A",
                "response_pct": 83,
                "note": (
                    "LEVEL A in cblG — methylfolate trap is the PRIMARY pathological mechanism, "
                    "identical to cblE. MTR enzyme absent → 5-methylTHF cannot donate its methyl group → "
                    "5-methylTHF accumulates → functional THF depletion → megaloblastic anemia. "
                    "Folinic acid (5-formyl-THF) bypasses the methylfolate trap completely by "
                    "entering the THF pool directly — corrects anemia + DNA synthesis impairment. "
                    "In cblC/cblX: folate trap is secondary; in cblG (and cblE) it is PRIMARY → Level A here. "
                    "Do NOT withhold because serum folate appears high — the functional deficiency is real."
                ),
            },
            {
                "treatment": "Folic acid (supplemental)",
                "evidence": "Level B",
                "response_pct": 52,
                "note": (
                    "Some evidence for megaloblastic anemia correction; however, folinic acid is strongly "
                    "preferred. Folic acid must be converted: folic → dihydrofolate → THF → 5,10-methyleneTHF → "
                    "5-methylTHF → then TRAPPED again at the absent MTR step. "
                    "Folinic acid (5-formyl-THF) enters the cycle directly, bypassing the blocked step. "
                    "Use folinic acid as primary agent; folic acid as adjunct only if folinic acid unavailable."
                ),
            },
            {
                "treatment": "L-Carnitine 50–100 mg/kg/day",
                "evidence": "Level B",
                "response_pct": 58,
                "note": (
                    "Mild secondary carnitine depletion possible in cblG (less severe than cblC "
                    "since no MMA-driven propionylcarnitine excretion). "
                    "Monitor free carnitine; supplement if low. "
                    "Level B — not as critical as in combined MMA+HHcy disorders."
                ),
            },
            {
                "treatment": "IV Glucose / Dextrose (acute crisis)",
                "evidence": "Level A",
                "response_pct": 76,
                "note": (
                    "Acute metabolic crisis management: IV dextrose to suppress protein catabolism + "
                    "Hcy release. Continue OHCbl IM + betaine + folinic acid during crisis. "
                    "Less MMA-crisis risk than cblC but tHcy crisis from catabolism is real in cblG. "
                    "Mandatory during any illness, surgery, or NPO episode."
                ),
            },
            {
                "treatment": "ACTH / Vigabatrin (infantile spasms)",
                "evidence": "Level A (IS)",
                "response_pct": 56,
                "note": (
                    "ACTH or vigabatrin for infantile spasms pattern (West syndrome). "
                    "Combined with OHCbl + betaine + folinic acid Level A. "
                    "IS in cblG is primarily metabolic (HHcy + functional folate deficiency) — "
                    "metabolic correction often significantly improves seizure control. "
                    "LEV first-line AED; never VPA (folate-depleting in already folate-deficient patient)."
                ),
            },
            {
                "treatment": "VPA — HIGH RISK / AVOID",
                "evidence": "AVOID",
                "response_pct": 0,
                "note": (
                    "HIGH RISK: VPA depletes both carnitine AND folate — worsens BOTH metabolic "
                    "vulnerabilities of cblG (folate depletion worsens methylfolate trap; "
                    "carnitine depletion in patients already on folinic acid + OHCbl regimen). "
                    "LEV is unconditionally first-line AED for ALL seizure types in MTR/cblG."
                ),
            },
        ],
        "patient_sample": sample,
    }


def get_definitions() -> dict:
    return {
        "gene_card": {
            "Gene": (
                "MTR (5-methyltetrahydrofolate-homocysteine methyltransferase; "
                "cblG gene product; Methionine Synthase; also MS)"
            ),
            "Full Name": (
                "Homocystinuria-megaloblastic anemia, cblG complementation type "
                "(MTR; Complementation Group cblG)"
            ),
            "Chromosome": "1q43",
            "Inheritance": "AUTOSOMAL RECESSIVE — biallelic LOF; both sexes equally affected",
            "OMIM Gene": "*156570",
            "OMIM Disease": "#250940 (Homocystinuria-megaloblastic anemia, cblG complementation type)",
            "Protein Size": (
                "1265 amino acids; ~139 kDa; cytoplasmic methionine synthase; cobalamin-dependent; "
                "N-terminal cobalamin-binding Rossmann fold domain (AA 1–348; His759 axial Co ligand) + "
                "central catalytic β/α-barrel domain (AA 349–700; methyl transfer active site) + "
                "C-terminal activation domain (AA 701–1265; MTRR docking site)"
            ),
            "Protein Function": (
                "Catalyzes: 5-methylTHF + Hcy → THF + Methionine, using MeCbl as methyl-transfer cofactor; "
                "terminal enzyme of cytoplasmic cobalamin arm (Step 4b); "
                "requires MTRR (cblE) to reactivate its cofactor after oxidation; "
                "acts ONLY on MeCbl/Hcy arm — AdoCbl/MMUT arm is completely independent"
            ),
            "Pathway Position": (
                "Step 4b — terminal cytoplasmic cobalamin enzyme; "
                "downstream of MMADHC ARM B distribution; "
                "without MTR, Hcy cannot be remethylated → isolated HHcy + methylfolate trap"
            ),
            "Key Role of MTRR": (
                "MTRR (cblE) reactivates MTR by reducing cob(II)alamin on MTR back to cob(I)alamin; "
                "in cblG, MTR enzyme is ABSENT — MTRR is present and functional but has no substrate; "
                "this distinguishes cblG mechanistically from cblE (reductase absent, synthase present)"
            ),
            "Prevalence": (
                "<100 cases worldwide 2026 (rarer than cblE; "
                "likely underdiagnosed due to NBS invisibility)"
            ),
        },
        "key_concepts": [
            {
                "concept": "MTR is the terminal synthase — the cblG mechanism",
                "explanation": (
                    "MTR (Methionine Synthase) catalyzes: "
                    "5-methylTHF + Hcy → THF + Methionine, using cob(I)alamin as the active cofactor. "
                    "During each catalytic cycle, cob(I)alamin is occasionally oxidized to "
                    "inactive cob(II)alamin (~1 in 2000 cycles). MTRR (cblE gene product) "
                    "reductively reactivates MTR by delivering electrons: NADPH → FAD → FMN → "
                    "cob(II)alamin on MTR → cob(I)alamin. "
                    "cblG LOF → MTR enzyme ABSENT → no remethylation at all → "
                    "Hcy cannot be converted to methionine → ISOLATED HHcy (MMA NORMAL KEY NEGATIVE). "
                    "In cblG, MTRR is present and functional but has no MTR enzyme to reactivate — "
                    "MTRR sits idle. This contrasts with cblE (MTRR absent, MTR present but inactive)."
                ),
            },
            {
                "concept": "Isolated HHcy (MMA NORMAL) — same biochemical pattern as cblE",
                "explanation": (
                    "MTR acts exclusively on the MeCbl/Hcy remethylation arm (Step 4b). "
                    "The AdoCbl arm (Step 4a: MMADHC → MMAB → AdoCbl → MMUT → MMA catabolism) "
                    "is entirely upstream of and independent of MTR. "
                    "Clinical implication: urine MMA NORMAL (<5 mmol/mol Cr) in ALL cblG patients. "
                    "This is the PRIMARY distinguishing feature from cblC (MMACHC), cblX (HCFC1), "
                    "cblD-combined (MMADHC C-terminus), and other combined MMA+HHcy disorders. "
                    "If a patient has ELEVATED MMA + HHcy → NOT cblG → look for combined disorders. "
                    "Protein restriction also NOT needed in cblG (no propionyl-CoA overload — same as cblE)."
                ),
            },
            {
                "concept": "Methylfolate trap — primary mechanism in cblG (Level A for folinic acid)",
                "explanation": (
                    "MTR uses 5-methylTHF as the methyl donor (5-methylTHF → MTR → THF + Methionine). "
                    "When MTR is absent (cblG LOF), 5-methylTHF cannot donate its methyl group → "
                    "5-methylTHF ACCUMULATES (methylfolate trap). "
                    "Consequences: (1) THF pool depleted → impaired thymidylate synthesis → "
                    "megaloblastic anemia (macrocytic, hypersegmented neutrophils, elevated MCV). "
                    "(2) Serum folate appears HIGH (methylfolate trap — folate assay measures all folates "
                    "including trapped 5-methylTHF) → 'folate paradox': folate HIGH but functionally zero. "
                    "Folinic acid (5-formyl-THF) bypasses the trap by entering THF cycle directly "
                    "→ Level A evidence in cblG (same as cblE — methylfolate trap is PRIMARY mechanism here). "
                    "In cblC/cblX, folate trap is secondary; in cblG/cblE it is primary."
                ),
            },
            {
                "concept": "NBS INVISIBLE — cblG missed by newborn screening",
                "explanation": (
                    "Standard NBS measures C3 (propionylcarnitine) as the primary cobalamin disorder screen. "
                    "C3 is elevated when the MMA arm is blocked (cblA, cblB, cblC, cblX, cblD-combined). "
                    "In cblG: MMA arm is COMPLETELY INTACT → C3 is NORMAL → NBS is NEGATIVE. "
                    "All cblG patients who are diagnosed clinically present after symptom onset: "
                    "megaloblastic anemia in early infancy, developmental regression, neonatal encephalopathy, "
                    "or in late-onset cases cognitive decline or psychiatric features. "
                    "Countries with expanded NBS including homocysteine can detect some cases but "
                    "cblG is ultra-rare (<100 worldwide) — clinical awareness is essential. "
                    "Never be reassured by normal NBS in a child with megaloblastic anemia + HHcy."
                ),
            },
            {
                "concept": "cblG vs cblE — biochemically IDENTICAL; gene panel + complementation distinguishes",
                "explanation": (
                    "cblG (MTR LOF) and cblE (MTRR LOF) produce IDENTICAL biochemistry: "
                    "isolated HHcy, normal MMA, low methionine, methylfolate trap, megaloblastic anemia, "
                    "MeCbl absent + AdoCbl normal in fibroblasts, high serum folate, NBS-invisible. "
                    "The mechanism differs fundamentally: "
                    "cblG = MTR (synthase) absent → no remethylation enzyme; MTRR is functional but idle. "
                    "cblE = MTRR (reductase) absent → MTR enzyme present but cofactor cannot be reactivated. "
                    "Fibroblast complementation assay: cblE + cblG fibroblasts fused → "
                    "complementation occurs (different complementation groups) → biochemically distinguishes. "
                    "WES/WGS (MTR 1q43 vs MTRR 5p15.31) conclusively separates the two. "
                    "Treatment is the same for both (OHCbl + betaine + folinic acid Level A)."
                ),
            },
            {
                "concept": "OHCbl response less complete in cblG (~50–65%) vs cblE (~60–75%)",
                "explanation": (
                    "In cblE (MTRR absent): MTR enzyme is PRESENT — high-dose OHCbl partially compensates "
                    "by mass-action shift of cob(II)→cob(I) equilibrium at very high cofactor concentrations, "
                    "maintaining partial activity of existing MTR molecules. Response ~60–75%. "
                    "In cblG (MTR absent): MTR enzyme is ABSENT — there is no enzyme scaffold to deliver "
                    "exogenous cobalamin to; OHCbl cannot restore absent enzyme function. "
                    "Any response in cblG derives from minor alternative methyltransferases + "
                    "high betaine-driven BHMT activity. Response ~50–65%. "
                    "This makes betaine (TMG) even more critical in cblG than in cblE — "
                    "BHMT is the primary controllable HHcy bypass in cblG."
                ),
            },
            {
                "concept": "MTRR requirement — MTRR is intact in cblG but has no substrate",
                "explanation": (
                    "MTRR (cblE gene product) reactivates MTR by delivering electrons to MTR's cofactor. "
                    "In normal physiology: MTRR docks on the C-terminal activation domain of MTR "
                    "and periodically reduces cob(II)alamin → cob(I)alamin to sustain MTR activity. "
                    "In cblG (MTR LOF): the MTRR enzyme is fully intact and functional — but there is "
                    "no MTR enzyme for MTRR to dock on. MTRR has no substrate. "
                    "Contrast: in cblE (MTRR LOF), MTR enzyme is present but accumulates inactive "
                    "cob(II)alamin because no MTRR reductase is available to reactivate it. "
                    "Both produce the same end-state (non-functional MTR axis) but via different mechanisms. "
                    "The C-terminal activation domain variants (p.Asp919Gly, p.Pro1173Leu) in cblG "
                    "disrupt the MTRR docking site — so even though MTRR is intact, it cannot dock → "
                    "cofactor slowly oxidizes → functionally similar to cblE."
                ),
            },
            {
                "concept": "No protein restriction needed (MMA arm intact) — same as cblE",
                "explanation": (
                    "Protein restriction is indicated in disorders where the MMA arm is blocked, "
                    "leading to propionyl-CoA / methylmalonyl-CoA accumulation (cblA, cblB, cblC, MMUT). "
                    "In cblG: MTR LOF affects only the Hcy remethylation arm (Step 4b). "
                    "The AdoCbl/MMUT arm (Step 4a) is completely intact — MMAB, MMAA, MMUT all normal. "
                    "Protein catabolism does NOT cause MMA accumulation in cblG. "
                    "Protein restriction would be HARMFUL: dietary methionine is the main methionine "
                    "source in cblG (since de novo synthesis via MTR is absent) — "
                    "restricting protein would worsen methionine deficiency and HHcy indirectly. "
                    "Maintain normal or slightly increased dietary methionine in cblG."
                ),
            },
            {
                "concept": "Folinic acid Level A — methylfolate trap is primary (not secondary) in cblG",
                "explanation": (
                    "In combined disorders (cblC, cblX): the primary defect is failure to process "
                    "cobalamin globally → both MMA and HHcy elevated; methylfolate trap occurs but "
                    "is secondary to the cobalamin processing failure → folinic acid is Level B. "
                    "In cblG (and cblE): the primary defect IS the methylfolate trap — "
                    "5-methylTHF accumulates because the enzyme (MTR) that consumes it is absent. "
                    "This is not a secondary consequence — it is the central pathological mechanism. "
                    "Folinic acid Level A means: start immediately, maintain lifelong, do not taper "
                    "even when HHcy is controlled with betaine + OHCbl (trap persists regardless). "
                    "Monitor MCV and hemoglobin alongside tHcy to track both disease axes independently."
                ),
            },
        ],
        "diagnostic_thresholds": [
            {
                "parameter": "Plasma tHcy",
                "threshold": ">40 µmol/L (isolated — MMA normal)",
                "action": (
                    "Isolated HHcy + normal MMA → cblG (MTR), cblE (MTRR), or cblD-HHcy (MMADHC N-term) differential. "
                    "Check plasma methionine (low → cblG/cblE/cblD-HHcy; high → CBS). "
                    "Check serum folate (high = methylfolate trap → cblG/cblE). "
                    "Start OHCbl + betaine + folinic acid empirically while awaiting genetics."
                ),
            },
            {
                "parameter": "Urine MMA",
                "threshold": "NORMAL (<5 mmol/mol Cr) in cblG — KEY NEGATIVE",
                "action": (
                    "MMA normal + HHcy elevated → ISOLATED HHcy pattern. Excludes cblC, cblX, cblD-combined. "
                    "Differential: cblG (MTR), cblE (MTRR), cblD-HHcy-only (MMADHC N-term), CBS. "
                    "If MMA elevated + HHcy elevated → combined disorder (cblC, cblX, cblF, cblJ, cblD-combined)."
                ),
            },
            {
                "parameter": "Serum folate",
                "threshold": "ELEVATED (>25 nmol/L) despite clinical folate deficiency",
                "action": (
                    "High serum folate + megaloblastic anemia + HHcy = methylfolate trap = cblG or cblE pattern. "
                    "This is the 'folate paradox': serum folate HIGH but functionally depleted. "
                    "Do NOT withhold folinic acid because serum folate is high — functional deficiency is real. "
                    "Start folinic acid Level A immediately. Distinguish from MTHFR (serum folate low, not high)."
                ),
            },
            {
                "parameter": "MCV (mean corpuscular volume)",
                "threshold": ">98 fL (macrocytosis)",
                "action": (
                    "Macrocytic megaloblastic anemia + elevated tHcy + low methionine + high serum folate = cblG/cblE. "
                    "Review blood smear for hypersegmented neutrophils + macro-ovalocytes (megaloblastic morphology). "
                    "Folinic acid + OHCbl commenced; methylfolate trap must be relieved — monitor CBC weekly."
                ),
            },
            {
                "parameter": "MeCbl fibroblasts",
                "threshold": "ABSENT (with AdoCbl NORMAL)",
                "action": (
                    "MeCbl absent + AdoCbl normal = cblG (MTR) or cblE (MTRR) fibroblast pattern. "
                    "Complementation assay with known cblE and cblG fibroblasts distinguishes. "
                    "WES/WGS of MTR (1q43) + MTRR (5p15.31) confirms gene. "
                    "Compare: cblC → MeCbl AND AdoCbl both absent (MMACHC upstream of both arms)."
                ),
            },
            {
                "parameter": "NBS C3",
                "threshold": "NORMAL (<3.5 µmol/L) — cblG NOT detected by NBS",
                "action": (
                    "Normal NBS C3 does NOT exclude cblG. cblG is NBS-INVISIBLE. "
                    "Any infant with megaloblastic anemia + unexplained neurological regression + "
                    "high serum folate needs plasma tHcy measured urgently — "
                    "do not be reassured by normal NBS result. cblG is ultra-rare but treatable."
                ),
            },
            {
                "parameter": "OHCbl trial: tHcy",
                "threshold": "~50–65% reduction expected after 7 days (less than cblE ~60–75%)",
                "action": (
                    "Slightly less OHCbl response than cblE (MTR enzyme itself absent in cblG). "
                    "If <50% HHcy reduction at day 7: maximize betaine dose (primary bypass in cblG); "
                    "confirm folinic acid Level A; review adherence. "
                    "Continue OHCbl regardless — partial response still beneficial. "
                    "Monitor methionine concurrently — rising methionine confirms some residual rescue."
                ),
            },
            {
                "parameter": "EEG: infantile spasms pattern (hypsarrhythmia)",
                "threshold": "Any infant with IS + megaloblastic anemia + HHcy",
                "action": (
                    "Infantile spasms + megaloblastic anemia + elevated tHcy + normal MMA = cblG/cblE. "
                    "Metabolic correction with OHCbl + betaine + folinic acid often reduces IS burden. "
                    "Add ACTH or vigabatrin for IS control per standard protocol. "
                    "LEV first-line AED; never VPA (folate-depleting in already folate-deficient patient)."
                ),
            },
        ],
        "differential_diagnosis": [
            {
                "disease": "cblE (MTRR — 5p15.31)",
                "distinguishing": (
                    "Biochemically IDENTICAL to cblG — isolated HHcy, normal MMA, low methionine, "
                    "methylfolate trap, megaloblastic anemia, MeCbl absent + AdoCbl normal in fibroblasts, "
                    "high serum folate, NBS invisible. "
                    "Mechanistic difference: cblE = MTRR (reductase) absent, MTR enzyme present but inactive; "
                    "cblG = MTR (synthase) absent, MTRR functional but no substrate. "
                    "OHCbl response slightly better in cblE (~60–75%) vs cblG (~50–65%). "
                    "Fibroblast complementation assay (cblE + cblG fuse → complement = different groups) "
                    "or WES/WGS (MTRR 5p15.31 vs MTR 1q43) is the ONLY way to distinguish. "
                    "Treatment identical: OHCbl + betaine + folinic acid Level A for both."
                ),
            },
            {
                "disease": "cblC (MMACHC — 1p34.1)",
                "distinguishing": (
                    "KEY DIFFERENCE: cblC has COMBINED MMA+HHcy — MMA ELEVATED in cblC, NORMAL in cblG. "
                    "AdoCbl ABSENT in cblC fibroblasts (MMACHC blocks both arms) — "
                    "NORMAL in cblG (only MeCbl arm affected). "
                    "NBS detects cblC (C3 elevated) — misses cblG (C3 normal). "
                    "Maculopathy in cblC (~80%) — absent in cblG. "
                    "Folinic acid is Level B in cblC (secondary methylfolate trap) — "
                    "Level A in cblG (methylfolate trap is PRIMARY mechanism)."
                ),
            },
            {
                "disease": "CBS — Classical Homocystinuria (21q22.3)",
                "distinguishing": (
                    "KEY DIFFERENCE: methionine HIGH in CBS (transsulfuration blocked → Met accumulates) "
                    "vs methionine LOW in cblG (remethylation enzyme absent → Met not synthesized). "
                    "MMA normal in both (neither affects MMA arm). "
                    "CBS: NO megaloblastic anemia (THF cycle intact in CBS); "
                    "serum folate NORMAL in CBS (no methylfolate trap); "
                    "CBS responds to pyridoxine (B6) in ~50% of patients; "
                    "cblG does not respond to B6. "
                    "Marfanoid habitus, lens dislocation in CBS — absent in cblG."
                ),
            },
            {
                "disease": "cblD-HHcy-only (MMADHC N-terminus — 2q23.2)",
                "distinguishing": (
                    "IDENTICAL biochemistry: isolated HHcy, normal MMA, low methionine. "
                    "MeCbl absent + AdoCbl normal in fibroblasts — same as cblG. "
                    "Methylfolate trap may occur in cblD-HHcy too (secondary to MTR arm block). "
                    "ONLY distinction: gene — MMADHC (2q23.2) N-terminus variants vs MTR (1q43). "
                    "Gene panel distinguishes; no clinical or biochemical differentiation possible. "
                    "Both: OHCbl + betaine Level A; folinic acid."
                ),
            },
            {
                "disease": "MTHFR deficiency (1p36.3)",
                "distinguishing": (
                    "HHcy elevated + MMA normal (same as cblG). "
                    "KEY DIFFERENCES: serum folate NORMAL or LOW in MTHFR (MTHFR cannot make 5-methylTHF → "
                    "5-methylTHF LOW or absent) vs HIGH in cblG (5-methylTHF trapped = HIGH). "
                    "NO megaloblastic anemia in MTHFR (thymidylate synthesis via MTHFR-independent "
                    "THF routes is preserved); megaloblastic anemia = hallmark of cblG. "
                    "MTHFR responds to betaine + riboflavin; does NOT respond to OHCbl alone "
                    "(cobalamin not the primary defect in MTHFR). "
                    "MCV: macrocytic in cblG; MCV typically normal in MTHFR."
                ),
            },
            {
                "disease": "cblX (HCFC1 — Xq28)",
                "distinguishing": (
                    "KEY DIFFERENCE: cblX has COMBINED MMA+HHcy (MMACHC not expressed → both arms blocked). "
                    "cblG has ISOLATED HHcy (MMA normal — only MeCbl arm affected). "
                    "cblX is X-linked (males only); cblG is AR (both sexes equally). "
                    "OHCbl response worse in cblX (~40–60%) vs cblG (~50–65%). "
                    "Folinic acid Level B in cblX (secondary folate trap); Level A in cblG (primary). "
                    "NBS detects cblX (C3 elevated); misses cblG (C3 normal)."
                ),
            },
            {
                "disease": "Megaloblastic anemia — Vitamin B12 nutritional deficiency",
                "distinguishing": (
                    "Nutritional B12 deficiency (vegan/vegetarian diet, maternal deficiency) causes "
                    "HHcy + megaloblastic anemia — same as cblG. "
                    "KEY DIFFERENCES: (1) serum B12 LOW in nutritional deficiency; "
                    "NORMAL or HIGH in cblG (cobalamin uptake from diet is normal; defect is in MTR enzyme). "
                    "(2) Responds rapidly and completely to IM B12 replacement in nutritional deficiency; "
                    "cblG: only partial response (MTR enzyme absent — B12 supplementation alone insufficient). "
                    "(3) Fibroblast cobalamin synthesis: NORMAL in nutritional B12 deficiency; "
                    "MeCbl absent in cblG fibroblasts. "
                    "WES/WGS distinguishes if needed after B12 supplementation trial."
                ),
            },
            {
                "disease": "MTHFR deficiency vs cblG — serum folate is the key differentiator",
                "distinguishing": (
                    "Both MTHFR and cblG present with HHcy + normal MMA. "
                    "In MTHFR deficiency: MTHFR cannot produce 5-methylTHF → 5-methylTHF is LOW → "
                    "serum folate is LOW or normal (less 5-methylTHF = less total measured folate). "
                    "In cblG: MTR absent → 5-methylTHF ACCUMULATES (methylfolate trap) → "
                    "serum folate HIGH despite functional deficiency. "
                    "Megaloblastic anemia is present in cblG (methylfolate trap → THF depletion); "
                    "in MTHFR it is typically absent (thymidylate synthesis routes partially preserved). "
                    "Betaine + riboflavin for MTHFR; OHCbl + betaine + folinic acid for cblG."
                ),
            },
        ],
    }
