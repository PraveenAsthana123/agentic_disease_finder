#!/usr/bin/env python3
"""MTHFR (Severe MTHFR Deficiency / Homocystinuria-Methylenetetrahydrofolate Reductase Deficiency) Epilepsy Dashboard.

MTHFR encodes Methylenetetrahydrofolate Reductase, a 656-aa cytoplasmic flavoprotein (FAD cofactor)
at 1p36.3 that catalyzes the NADPH-dependent reduction of 5,10-methyleneTHF → 5-methylTHF.
5-methylTHF is the ONLY methyl donor for MTR (methionine synthase / cblG), which remethylates
homocysteine (Hcy) → methionine. MTHFR is UPSTREAM of MTR and the entire cobalamin axis —
the cobalamin system (LMBRD1, ABCD4, MMACHC, MMADHC, MMAB, MMUT, MTRR, MTR) is STRUCTURALLY INTACT
in MTHFR deficiency; the defect is in the supply of the methyl-group substrate for MTR.

REMETHYLATION PATHWAY — MTHFR IS THE UPSTREAM FOLATE ENZYME (Step 0 in cytoplasmic arm):
  Folate cycle → DHFR → DHF → THF → SHMT → 5,10-methyleneTHF
                                                    │
                                             MTHFR (FAD, NADPH)
                                                    ↓
                                            5-methylTHF (methylfolate)
                                                    │
                                   MTR (cobalamin-dependent synthase, cblG)
                                                    │
                                   Hcy → Methionine (remethylation complete)
                                                    │
                           cblG BLOCK ← MTR absent → ISOLATED HHcy
                           MTHFR BLOCK ← MTHFR absent → NO 5-methylTHF MADE →
                                         MTR has NO methyl donor → same ISOLATED HHcy
                                         but via upstream deprivation of substrate

  AdoCbl arm (MMUT / MMA catabolism): COMPLETELY INDEPENDENT of MTHFR.
  Cobalamin processing (LMBRD1, ABCD4, MMACHC, MMADHC, MMAB, MMUT, MTRR, MTR): all INTACT.
  KEY: MeCbl in fibroblasts is NORMAL in MTHFR — because MeCbl synthesis does not depend on
  5-methylTHF being produced; the MTR enzyme is intact and can incorporate MeCbl normally.

MTHFR PROTEIN BIOLOGY:
  656 amino acids; cytoplasmic flavoprotein; ~74 kDa (monomer); active as dimer/octamer.
  FAD cofactor (riboflavin-derived): essential for NADPH electron transfer to reduce
  5,10-methyleneTHF to 5-methylTHF; FAD is non-covalently bound.
  N-terminal catalytic domain (AA 1–340): contains the active site; FAD binding site;
    substrate binding for 5,10-methyleneTHF; NADPH-binding site; most pathogenic variants cluster here.
  C-terminal regulatory domain (AA 341–656): SAM (S-adenosylmethionine) allosteric binding site;
    SAM inhibits MTHFR activity (product-feedback inhibition via downstream methionine); phosphorylation
    at Ser391 modulates activity. The regulatory domain is critical for homeostatic control —
    when methionine is high → SAM high → MTHFR inhibited → less 5-methylTHF produced.
  Common polymorphisms (NOT severe deficiency alleles):
    p.Ala222Val (c.665C>T, "C677T"): thermolabile variant; 50-70% residual activity; common in all
      populations (allele freq 0.3–0.4 in Europeans); associated with mild HHcy, not severe deficiency.
    p.Glu429Ala (c.1286A>C, "A1298C"): reduces activity mildly; compound C677T/A1298C can cause
      moderate HHcy but not severe MTHFR deficiency.
  Severe deficiency alleles: <1–5% residual MTHFR activity; biallelic LOF (frameshift, nonsense,
    missense in catalytic core, splice variants).

MTHFR LOF RESULT — ISOLATED HHcy, NORMAL MMA (MMA arm intact — KEY NEGATIVE):
  → 5-methylTHF NOT PRODUCED: MTHFR cannot reduce 5,10-methyleneTHF → 5-methylTHF.
    MTR has no methyl donor → Hcy cannot be remethylated → tHcy ELEVATED (50–300 µmol/L).
    Range often HIGHER than cblE/cblG because: in cblE/cblG, the folate cycle can still
    produce 5-methylTHF (MTHFR intact) and some serum 5-methylTHF exists; in MTHFR deficiency,
    no 5-methylTHF is made at all — MTR lacks substrate entirely.
  → Methionine LOW (5–15 µmol/L): MTR starved of methyl donor → methionine not synthesized.
  → MMA NORMAL: KEY NEGATIVE — MMUT/AdoCbl arm completely intact in MTHFR deficiency.
    MTHFR does not touch the AdoCbl axis.
  → MeCbl NORMAL in fibroblasts: KEY POSITIVE DIFFERENCE vs cblE/cblG/cblD-HHcy.
    The cobalamin system is entirely intact — LMBRD1, ABCD4, MMACHC, MMADHC all function normally.
    MTR enzyme is present; MTRR is intact; MTR CAN incorporate cobalamin → MeCbl is made normally.
    The problem is: when MTR tries to use MeCbl to remethylate Hcy, it needs 5-methylTHF as co-substrate —
    which is absent in MTHFR deficiency → MTR stalls (no methyl donor) → Hcy not remethylated.
    But the MeCbl-loading step (MTRR reactivation, MMADHC delivery) proceeds normally.
  → AdoCbl NORMAL in fibroblasts: same as MeCbl — whole cobalamin system intact.
  → C3 NORMAL on NBS: MMA arm intact → C3 not elevated → NBS INVISIBLE for MTHFR.
  → Serum folate LOW or NORMAL (KEY DISTINCTION vs cblE/cblG):
    In cblE/cblG: methylfolate TRAP — 5-methylTHF is produced normally by MTHFR but CANNOT
      be consumed by non-functional MTR → 5-methylTHF ACCUMULATES → serum folate HIGH.
    In MTHFR deficiency: MTHFR cannot PRODUCE 5-methylTHF → 5-methylTHF is LOW or absent →
      serum folate is LOW or NORMAL. The substrate (5,10-methyleneTHF) may accumulate instead.
    This is the MOST IMPORTANT distinguishing feature between MTHFR and cblE/cblG.
  → NO megaloblastic anemia (typical): In cblE/cblG, the methylfolate trap depletes THF →
    thymidylate synthesis impaired → megaloblastic anemia.
    In MTHFR deficiency: THF is not being trapped as 5-methylTHF; 5,10-methyleneTHF accumulates
    and can still feed thymidylate synthase (TYMS) directly → DNA synthesis partially preserved →
    megaloblastic anemia is ABSENT or MILD in most MTHFR patients (KEY NEGATIVE vs cblE/cblG).
  → Leukoencephalopathy PROMINENT: white matter disease more severe in MTHFR than cblE/cblG.
    Mechanism: methionine deficiency → SAM deficiency → myelin maintenance impaired;
    HHcy directly toxic to white matter endothelium; longer duration of HHcy (NBS invisible,
    severe MTHFR not detected early) → cumulative white matter damage.
  → Riboflavin (B2/FAD) cofactor dependence: MTHFR uses FAD; B2 deficiency worsens MTHFR activity;
    riboflavin supplementation Level A in MTHFR deficiency (stabilizes FAD-binding + thermolabile variants).

CRITICAL BIOCHEMICAL DIFFERENCES — MTHFR vs cblE/cblG (THE EXAM QUESTION):
  Feature                          MTHFR            cblE (MTRR)       cblG (MTR)
  tHcy                             ELEVATED          ELEVATED           ELEVATED
  MMA                              NORMAL            NORMAL             NORMAL
  Methionine                       LOW               LOW                LOW
  Serum folate                     LOW/NORMAL        HIGH               HIGH
  Megaloblastic anemia             ABSENT/RARE       90%                80%
  MeCbl fibroblasts                NORMAL ← KEY      ABSENT             ABSENT
  AdoCbl fibroblasts               NORMAL            NORMAL             NORMAL
  C3 / NBS                         NORMAL            NORMAL             NORMAL
  Riboflavin                       Level A ← KEY     Not indicated      Not indicated
  OHCbl response                   Level B ← KEY     Level A            Level A
  Folinic acid                     Level B ← KEY     Level A            Level A
  Methylfolate (5-methylTHF)       Bypasses block    Cannot help        Cannot help
  Betaine                          Level A (both)    Level A            Level A
  Methionine supplementation       Level A ← KEY     Not needed         Not needed
  White matter disease             PROMINENT ← KEY   Mild               Moderate
  Locus                            1p36.3            5p15.31            1q43

MTHFR BIOMARKERS:
  Plasma tHcy: 50–300 µmol/L ELEVATED — higher than cblE/cblG (40–200) because no 5-methylTHF at all
  Urine MMA: NORMAL (<5 mmol/mol Cr) — KEY NEGATIVE (MMA arm intact)
  Plasma methionine: 5–15 µmol/L LOW
  Serum folate: LOW or NORMAL (5–20 nmol/L) — KEY DISTINCTION vs cblE/cblG (where folate is HIGH)
  MeCbl fibroblasts: NORMAL — KEY POSITIVE difference vs cblE/cblG (cobalamin system fully intact)
  AdoCbl fibroblasts: NORMAL
  C3: NORMAL — NBS INVISIBLE
  CBC/MCV: typically NORMAL (no megaloblastic anemia); hemoglobin normal or mildly low
  Brain MRI: periventricular leukoencephalopathy (PROMINENT) — more severe than cblE/cblG

KEY VARIANTS IN MTHFR (severe deficiency pathogenic — NOT the common polymorphisms):
  p.Arg249Gln (c.746G>A): most common severe European allele; catalytic domain;
    severe — <1% residual activity; infantile to neonatal onset.
  p.Ile278Thr: catalytic domain; moderate-severe; 2–5% residual activity;
    infantile classic phenotype.
  p.Gly364Ala: FAD-binding region of catalytic domain; severe; FAD cofactor not retained;
    neonatal or early infantile.
  p.Leu432Phe: regulatory domain; moderate; partial SAM regulation retained; late-onset attenuated.
  c.1027T>A: exonic; severe null allele; early infantile; no mRNA detected by NMD.
  p.Ala222Val (C677T) alone: NOT pathogenic alone; thermolabile; 50-70% residual activity;
    only clinically significant as homozygous in B2-deficient individuals or compound with severe allele.
  p.Glu429Ala (A1298C) alone: NOT pathogenic alone; mild reduction; only relevant in combination.

OMIM: Gene *607093 (MTHFR) · Disease #236250 (Homocystinuria due to MTHFR deficiency)
Chromosome: 1p36.3 · Inheritance: AUTOSOMAL RECESSIVE (biallelic LOF; both sexes equally affected)
Prevalence: Severe deficiency — rare (estimated 1/200,000–400,000 births for severe form;
common polymorphisms C677T/A1298C are very frequent but cause mild HHcy, not severe deficiency)
"""

from __future__ import annotations
import random

random.seed(83)   # reproducible synthetic cohort — seed 83 for MTHFR


# ── helpers ───────────────────────────────────────────────────────────────────

def _pid(i: int) -> str:
    return f"MTHFR-{i:03d}"


def _sex(i: int) -> str:
    # Autosomal recessive: both sexes equally affected — alternating male/female
    return "M" if i % 2 == 0 else "F"


def _phenotype(i: int) -> str:
    """
    Neonatal Severe (~25%): null/null genotypes, MTHFR activity <1%; neonatal seizures,
      encephalopathy, severe HHcy (>150 µmol/L), white matter collapse.
    Infantile Classic (~50%): 1-5% residual MTHFR activity; developmental regression,
      hypotonia, leukoencephalopathy, infantile spasms — MODAL.
    Late-Onset Attenuated (~20%): 5-20% residual MTHFR activity; cognitive decline,
      psychiatric features, thromboembolism, schizophrenia-like presentation.
    Adult-Onset Psychiatric (~5%): >20% residual MTHFR activity; extreme tHcy,
      schizophrenia, stroke, late cognitive decline.
    """
    if i < 10:
        return "Neonatal Severe"
    elif i < 30:
        return "Infantile Classic"
    elif i < 38:
        return "Late-Onset Attenuated"
    else:
        return "Adult-Onset Psychiatric"


def _genotype(i: int, pheno: str) -> str:
    pheno_variants = {
        "Neonatal Severe": [
            "p.Gly364Ala/p.Gly364Ala (homozygous; FAD-binding catalytic domain null; <1% activity; neonatal encephalopathy)",
            "c.1027T>A/p.Gly364Ala (null exonic + FAD-binding null; compound het; NMD + zero FAD retention; neonatal)",
            "c.1027T>A/c.1027T>A (homozygous null; no mRNA by NMD; zero MTHFR protein; neonatal seizures)",
            "p.Arg249Gln/p.Gly364Ala (c.746G>A + FAD-binding null; compound het; catalytic + FAD domain; severe)",
            "p.Arg249Gln/c.1027T>A (most common severe allele + null; compound het; <1% activity; neonatal)",
        ],
        "Infantile Classic": [
            "p.Arg249Gln/p.Arg249Gln (c.746G>A; homozygous; most common severe European allele; <1% activity; IS)",
            "p.Arg249Gln/p.Ile278Thr (c.746G>A + Ile278Thr; compound het; catalytic domain; 2% residual; IS)",
            "p.Ile278Thr/p.Ile278Thr (homozygous; catalytic domain; 2-5% residual activity; infantile classic)",
            "p.Arg249Gln/p.Leu432Phe (catalytic + regulatory domain; compound het; moderate-severe; infantile)",
            "p.Ile278Thr/p.Gly364Ala (catalytic + FAD-binding; compound het; <2% activity; infantile spasms)",
            "p.Arg249Gln/c.IVS4+1G>A (most common allele + splice null; compound het; severe infantile)",
            "c.IVS4+1G>A/p.Ile278Thr (splice null + catalytic; compound het; <3% activity; IS pattern)",
        ],
        "Late-Onset Attenuated": [
            "p.Leu432Phe/p.Arg249Gln (regulatory + catalytic; compound het; partial SAM regulation; attenuated)",
            "p.Leu432Phe/p.Leu432Phe (homozygous regulatory domain; partial SAM regulation; 5-20% activity)",
            "p.Leu432Phe/p.Ile278Thr (regulatory + catalytic; compound het; moderate; late cognitive phenotype)",
            "p.Ala222Val/p.Arg249Gln (C677T thermolabile + severe allele; compound het; moderate HHcy; late-onset)",
        ],
        "Adult-Onset Psychiatric": [
            "p.Leu432Phe/p.Ala222Val (regulatory + C677T thermolabile; compound het; >20% activity; psychiatric)",
            "p.Ala222Val/p.Ile278Thr (C677T + catalytic; compound het; B2-responsive; adult schizophrenia-like)",
        ],
    }
    opts = pheno_variants[pheno]
    return opts[i % len(opts)]


def _mma_urine(pheno: str) -> float:
    # MMA NORMAL in ALL MTHFR patients — MMA arm (MMUT/AdoCbl) is completely intact
    return round(random.uniform(0.5, 4.0), 1)


def _homocysteine(pheno: str) -> float:
    # tHcy range HIGHER than cblE/cblG (50-300) because no 5-methylTHF substrate at all
    if pheno == "Neonatal Severe":
        return round(random.uniform(150, 300), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(80, 220), 1)
    elif pheno == "Late-Onset Attenuated":
        return round(random.uniform(50, 130), 1)
    else:  # Adult-Onset Psychiatric
        return round(random.uniform(50, 110), 1)


def _methionine(pheno: str) -> float:
    # Methionine LOW — MTR substrate-starved; 5-15 µmol/L range
    if pheno == "Neonatal Severe":
        return round(random.uniform(5, 8), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(6, 11), 1)
    elif pheno == "Late-Onset Attenuated":
        return round(random.uniform(8, 14), 1)
    else:  # Adult-Onset Psychiatric
        return round(random.uniform(9, 15), 1)


def _c3(pheno: str) -> float:
    # C3 NORMAL in MTHFR — NBS invisible — all phenotypes have normal C3
    return round(random.uniform(0.8, 2.5), 1)


def _serum_folate(pheno: str) -> float:
    # Folate LOW or NORMAL — KEY DISTINCTION vs cblE/cblG (where folate is HIGH)
    # MTHFR cannot produce 5-methylTHF → 5-methylTHF is low → total folate low-normal
    if pheno == "Neonatal Severe":
        return round(random.uniform(5, 12), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(6, 16), 1)
    elif pheno == "Late-Onset Attenuated":
        return round(random.uniform(8, 20), 1)
    else:  # Adult-Onset Psychiatric
        return round(random.uniform(10, 20), 1)


def _hemoglobin(pheno: str) -> float:
    # Hemoglobin typically NORMAL or mildly low — NO megaloblastic anemia in MTHFR
    # 5,10-methyleneTHF can still feed TYMS directly → thymidylate synthesis preserved
    if pheno == "Neonatal Severe":
        return round(random.uniform(9.5, 12.5), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(10.0, 13.5), 1)
    elif pheno == "Late-Onset Attenuated":
        return round(random.uniform(11.0, 14.0), 1)
    else:  # Adult-Onset Psychiatric
        return round(random.uniform(11.5, 14.5), 1)


def _mcv(pheno: str) -> int:
    # MCV typically NORMAL in MTHFR — no megaloblastic anemia (KEY NEGATIVE vs cblE/cblG)
    if pheno == "Neonatal Severe":
        return random.randint(80, 92)
    elif pheno == "Infantile Classic":
        return random.randint(80, 94)
    elif pheno == "Late-Onset Attenuated":
        return random.randint(82, 95)
    else:  # Adult-Onset Psychiatric
        return random.randint(83, 96)


def _free_carnitine(pheno: str) -> float:
    # Mild secondary depletion possible; no MMA arm → less severe than combined disorders
    if pheno == "Neonatal Severe":
        return round(random.uniform(25, 38), 1)
    elif pheno == "Infantile Classic":
        return round(random.uniform(28, 45), 1)
    elif pheno == "Late-Onset Attenuated":
        return round(random.uniform(30, 52), 1)
    else:  # Adult-Onset Psychiatric
        return round(random.uniform(32, 55), 1)


def _onset_months(pheno: str) -> int:
    if pheno == "Neonatal Severe":
        return random.randint(0, 1)
    elif pheno == "Infantile Classic":
        return random.randint(1, 18)
    elif pheno == "Late-Onset Attenuated":
        return random.randint(18, 120)
    else:  # Adult-Onset Psychiatric
        return random.randint(120, 360)


def _white_matter_disease(pheno: str) -> bool:
    # Leukoencephalopathy — PROMINENT in MTHFR, more than cblE/cblG
    rates = {
        "Neonatal Severe": 0.97,
        "Infantile Classic": 0.88,
        "Late-Onset Attenuated": 0.70,
        "Adult-Onset Psychiatric": 0.55,
    }
    return random.random() < rates[pheno]


def _thromboembolism(pheno: str) -> bool:
    # HHcy is procoagulant — 25% overall; higher in late-onset where HHcy is chronic
    rates = {
        "Neonatal Severe": 0.10,
        "Infantile Classic": 0.20,
        "Late-Onset Attenuated": 0.40,
        "Adult-Onset Psychiatric": 0.45,
    }
    return random.random() < rates[pheno]


def _psychiatric(pheno: str) -> bool:
    # Schizophrenia-like psychiatric features — more common in adult-onset
    rates = {
        "Neonatal Severe": 0.05,
        "Infantile Classic": 0.12,
        "Late-Onset Attenuated": 0.35,
        "Adult-Onset Psychiatric": 0.90,
    }
    return random.random() < rates[pheno]


def _megaloblastic_anemia(pheno: str) -> bool:
    # Megaloblastic anemia ABSENT/RARE in MTHFR — KEY NEGATIVE vs cblE/cblG
    # Thymidylate synthesis preserved via 5,10-methyleneTHF → TYMS direct pathway
    rates = {
        "Neonatal Severe": 0.12,
        "Infantile Classic": 0.08,
        "Late-Onset Attenuated": 0.05,
        "Adult-Onset Psychiatric": 0.04,
    }
    return random.random() < rates[pheno]


def _ohcbl_response(pheno: str) -> bool:
    # OHCbl response LESS important in MTHFR — Level B (cobalamin system intact)
    # MTR can use MeCbl fine, but it has no 5-methylTHF methyl donor to work with
    # High-dose OHCbl does not bypass the upstream MTHFR block
    rates = {
        "Neonatal Severe": 0.25,
        "Infantile Classic": 0.35,
        "Late-Onset Attenuated": 0.45,
        "Adult-Onset Psychiatric": 0.50,
    }
    return random.random() < rates[pheno]


def _betaine_response(pheno: str) -> bool:
    # Betaine (TMG) PRIMARY treatment in MTHFR — Level A; BHMT bypasses blocked MTR
    rates = {
        "Neonatal Severe": 0.70,
        "Infantile Classic": 0.82,
        "Late-Onset Attenuated": 0.90,
        "Adult-Onset Psychiatric": 0.88,
    }
    return random.random() < rates[pheno]


def _riboflavin_response(pheno: str) -> bool:
    # Riboflavin B2 Level A — MTHFR is FAD-dependent; B2 stabilizes MTHFR
    # Even severe variants may have partial partial response to B2 stabilization
    rates = {
        "Neonatal Severe": 0.30,
        "Infantile Classic": 0.52,
        "Late-Onset Attenuated": 0.72,
        "Adult-Onset Psychiatric": 0.78,
    }
    return random.random() < rates[pheno]


def _seizures(pheno: str) -> bool:
    # Seizure prevalence: infantile spasms, myoclonic, GTCS, focal, absence
    rates = {
        "Neonatal Severe": 0.90,
        "Infantile Classic": 0.65,
        "Late-Onset Attenuated": 0.30,
        "Adult-Onset Psychiatric": 0.20,
    }
    return random.random() < rates[pheno]


def _nbs(pheno: str) -> bool:
    # NBS INVISIBLE for MTHFR — C3 normal (MMA arm intact)
    rates = {
        "Neonatal Severe": 0.03,
        "Infantile Classic": 0.03,
        "Late-Onset Attenuated": 0.02,
        "Adult-Onset Psychiatric": 0.01,
    }
    return random.random() < rates[pheno]


def _peripheral_neuropathy(pheno: str) -> bool:
    # Peripheral neuropathy ~40% overall; more in late-onset (chronic HHcy + methionine depletion)
    rates = {
        "Neonatal Severe": 0.20,
        "Infantile Classic": 0.35,
        "Late-Onset Attenuated": 0.55,
        "Adult-Onset Psychiatric": 0.50,
    }
    return random.random() < rates[pheno]


def _aed(sz: bool, pheno: str) -> str:
    if not sz:
        return "None"
    if pheno == "Neonatal Severe":
        opts = [
            "LEV + PB (neonatal encephalopathy seizures; betaine + riboflavin commenced urgently; methionine supplement)",
            "LEV + PB + riboflavin (neonatal MTHFR seizures; FAD-dependent enzyme; betaine primary HHcy control)",
            "LEV + CLB (neonatal myoclonic; betaine + riboflavin B2 Level A initiated; methionine supplement started)",
            "LEV + PB (neonatal HHcy encephalopathy; white matter disease on MRI; metabolic correction priority)",
        ]
    elif pheno == "Infantile Classic":
        opts = [
            "LEV + ACTH (infantile spasms; betaine + riboflavin + methionine supplement synergistic)",
            "LEV + VIG (IS pattern; betaine primary HHcy treatment; riboflavin Level A; methionine supplement)",
            "LEV + CLB (myoclonic + leukoencephalopathy; riboflavin + betaine maintained; no VPA)",
            "LEV + ACTH + CLB (IS cluster; metabolic + AED combined; betaine primary; riboflavin mandatory)",
        ]
    elif pheno == "Late-Onset Attenuated":
        opts = [
            "LEV (focal cognitive seizures; late-onset MTHFR; betaine + riboflavin + methionine continued)",
            "LEV + LTG (focal seizures + psychiatric features; adult MTHFR; betaine + B2 supplementation)",
            "LEV + CLB (cognitive impairment + seizures; thromboembolism workup; betaine + riboflavin)",
        ]
    else:  # Adult-Onset Psychiatric
        opts = [
            "LEV + antipsychotic (adult-onset MTHFR; schizophrenia-like + seizures; betaine critical)",
            "LEV + LTG (adult GTCS; psychiatric features dominant; betaine + riboflavin + methionine)",
        ]
    return random.choice(opts)


def _diag_months(pheno: str, onset: int) -> int:
    delays = {
        "Neonatal Severe": (0, 2),
        "Infantile Classic": (3, 18),
        "Late-Onset Attenuated": (12, 60),
        "Adult-Onset Psychiatric": (24, 120),
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
            "mma_urine_mmol_molCr": _mma_urine(pheno),           # NORMAL in ALL MTHFR — MMA arm intact
            "total_homocysteine_umol_l": _homocysteine(pheno),    # HIGHER range than cblE/cblG
            "methionine_umol_l": _methionine(pheno),
            "c3_umol_l": _c3(pheno),                               # NORMAL — NBS invisible
            "serum_folate_nmol_l": _serum_folate(pheno),           # LOW/NORMAL — KEY vs cblE/cblG HIGH
            "mcv_fl": _mcv(pheno),                                 # NORMAL — no megaloblastic anemia
            "hemoglobin_g_dl": _hemoglobin(pheno),                 # NORMAL or mildly low
            "free_carnitine_umol_l": _free_carnitine(pheno),
            "mecbl_fibroblasts": "Normal",   # KEY POSITIVE: cobalamin system fully intact
            "adocbl_fibroblasts": "Normal",  # KEY POSITIVE: AdoCbl arm intact
            "mma_normal": True,              # KEY NEGATIVE: MMA arm completely unaffected
            "adocbl_normal": True,           # KEY POSITIVE: AdoCbl normal in fibroblasts
            "c3_normal": True,               # KEY NEGATIVE: NBS invisible (C3 not elevated)
            "megaloblastic_anemia": _megaloblastic_anemia(pheno),  # ABSENT/RARE — KEY vs cblE/cblG
            "white_matter_disease": _white_matter_disease(pheno),  # PROMINENT in MTHFR
            "thromboembolism": _thromboembolism(pheno),
            "psychiatric_features": _psychiatric(pheno),
            "peripheral_neuropathy": _peripheral_neuropathy(pheno),
            "seizures": sz,
            "nbs_detected": _nbs(pheno),       # Near-zero NBS detection — C3 normal
            "ohcbl_response": _ohcbl_response(pheno),   # Level B — less important
            "betaine_response": _betaine_response(pheno),
            "riboflavin_response": _riboflavin_response(pheno),
            "aed": _aed(sz, pheno),
        })
    return pts


_PATIENTS: list[dict] = _build_cohort()


# ── API functions ─────────────────────────────────────────────────────────────

def get_overview() -> dict:
    n      = len(_PATIENTS)
    mma_v  = [p["mma_urine_mmol_molCr"] for p in _PATIENTS]
    hcy_v  = [p["total_homocysteine_umol_l"] for p in _PATIENTS]
    met_v  = [p["methionine_umol_l"] for p in _PATIENTS]
    c3_v   = [p["c3_umol_l"] for p in _PATIENTS]
    fol_v  = [p["serum_folate_nmol_l"] for p in _PATIENTS]
    hgb_v  = [p["hemoglobin_g_dl"] for p in _PATIENTS]
    car_v  = [p["free_carnitine_umol_l"] for p in _PATIENTS]

    def pct(pred): return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    neo_pts   = [p for p in _PATIENTS if p["phenotype"] == "Neonatal Severe"]
    inf_pts   = [p for p in _PATIENTS if p["phenotype"] == "Infantile Classic"]
    late_pts  = [p for p in _PATIENTS if p["phenotype"] == "Late-Onset Attenuated"]
    adult_pts = [p for p in _PATIENTS if p["phenotype"] == "Adult-Onset Psychiatric"]

    return {
        "gene": "MTHFR",
        "full_name": (
            "Methylenetetrahydrofolate Reductase — "
            "Severe MTHFR Deficiency / Homocystinuria due to MTHFR deficiency (NOT the common C677T/A1298C polymorphisms)"
        ),
        "chromosome": "1p36.3",
        "inheritance": "AUTOSOMAL RECESSIVE (biallelic LOF; both sexes equally affected)",
        "omim_gene": "*607093",
        "omim_disease": "#236250",
        "protein_size": (
            "656 aa; ~74 kDa (monomer, active as dimer/octamer); cytoplasmic flavoprotein (FAD cofactor); "
            "N-terminal catalytic domain (AA 1–340: FAD binding, NADPH binding, 5,10-methyleneTHF substrate site; "
            "most pathogenic variants cluster here) + "
            "C-terminal regulatory domain (AA 341–656: SAM allosteric inhibition site, Ser391 phosphorylation)"
        ),
        "prevalence": (
            "Severe deficiency: estimated 1/200,000–400,000 births for severe form (biallelic null/severe alleles); "
            "common polymorphisms C677T/A1298C are very frequent (C677T allele freq ~0.3–0.4 in Europeans) but "
            "cause MILD HHcy only — NOT severe deficiency; likely underdiagnosed due to NBS invisibility"
        ),
        "nbs_primary": (
            "NOT DETECTED by standard NBS — C3 normal (MMA arm entirely intact in MTHFR deficiency); "
            "MTHFR deficiency shares NBS invisibility with cblE/cblG — all isolated HHcy disorders are NBS-invisible; "
            "diagnosed after clinical presentation: neonatal seizures/encephalopathy, white matter disease, "
            "developmental regression, or late-onset psychiatric/thrombotic features"
        ),
        "nbs_secondary": (
            "Plasma tHcy (ELEVATED, often higher than cblE/cblG — 50–300 µmol/L) + "
            "plasma methionine (LOW 5–15 µmol/L) + "
            "serum folate (LOW or NORMAL — KEY DISTINCTION from cblE/cblG where folate is HIGH) + "
            "MMA urine NORMAL (KEY NEGATIVE vs combined disorders) + "
            "MeCbl fibroblasts NORMAL (KEY POSITIVE DIFFERENCE vs cblE/cblG where MeCbl absent) + "
            "brain MRI (periventricular leukoencephalopathy — PROMINENT) + "
            "MTHFR enzyme activity in fibroblasts/leukocytes (<5% confirms severe deficiency) + "
            "WES/WGS for MTHFR (1p36.3) biallelic variants mandatory"
        ),
        "function": (
            "MTHFR catalyzes the NADPH-dependent reduction of 5,10-methyleneTHF to 5-methylTHF (methylfolate), "
            "using FAD (riboflavin-derived) as the electron-transfer cofactor. "
            "5-methylTHF is the ONLY methyl donor for MTR (methionine synthase), which uses it to "
            "remethylate homocysteine → methionine. "
            "MTHFR is therefore the UPSTREAM gatekeeper of the remethylation arm: "
            "without MTHFR activity, MTR has no substrate → Hcy cannot be remethylated → isolated HHcy. "
            "The cobalamin system (LMBRD1, ABCD4, MMACHC, MMADHC, MMAB, MMUT, MTRR, MTR) "
            "is STRUCTURALLY INTACT in MTHFR deficiency. "
            "The MMA arm (MMUT/AdoCbl) is completely independent of MTHFR. "
            "SAM (the downstream product of methionine + ATP) allosterically INHIBITS MTHFR — "
            "product-feedback regulation that maintains homocysteine homeostasis."
        ),
        "mechanism": (
            "MTHFR LOF → 5-methylTHF NOT PRODUCED → MTR (methionine synthase) has no methyl donor → "
            "Hcy cannot be remethylated → tHcy ELEVATED (50–300 µmol/L — higher range than cblE/cblG). "
            "MMA NORMAL (KEY NEGATIVE — MMA arm unaffected). "
            "MeCbl NORMAL in fibroblasts (KEY POSITIVE DIFFERENCE vs cblE/cblG — entire cobalamin system intact). "
            "Serum folate LOW or NORMAL (KEY DISTINCTION: MTHFR cannot MAKE 5-methylTHF → it is not trapped; "
            "in cblE/cblG the methylfolate TRAP accumulates 5-methylTHF → folate appears HIGH). "
            "NO megaloblastic anemia (KEY NEGATIVE vs cblE/cblG 80–90%): "
            "5,10-methyleneTHF accumulates and can still feed thymidylate synthase (TYMS) directly → "
            "thymidylate synthesis relatively preserved → no macrocytic anemia. "
            "White matter disease PROMINENT (leukoencephalopathy): methionine deficiency → SAM deficiency → "
            "impaired myelin maintenance + HHcy direct vascular/white-matter toxicity. "
            "Riboflavin (B2) Level A: MTHFR uses FAD; B2 supplementation stabilizes MTHFR protein "
            "and may restore partial activity even in severe variants. "
            "Betaine MANDATORY Level A: BHMT-mediated HHcy remethylation entirely bypasses MTHFR. "
            "Methionine supplementation Level A: methionine cannot be synthesized when MTR is substrate-starved."
        ),
        "key_negative": (
            "MMA NORMAL: KEY NEGATIVE vs combined disorders (cblC/cblX/cblD-combined) — MMA arm intact; "
            "C3 NORMAL: KEY NEGATIVE for NBS — MTHFR is NBS-INVISIBLE; "
            "MeCbl NORMAL in fibroblasts: KEY POSITIVE DIFFERENCE vs cblE/cblG (where MeCbl is absent); "
            "NO megaloblastic anemia (KEY NEGATIVE vs cblE 90% / cblG 80%): thymidylate synthesis preserved; "
            "Serum folate LOW/NORMAL (not high as in cblE/cblG where methylfolate trap elevates folate); "
            "Protein restriction NOT needed (no MMA arm problem — same as cblE/cblG); "
            "N2O HIGH RISK (not absolute CI as in cblE/cblG — MTR cobalamin is intact; but already "
            "methionine-depleted so any further MTR impairment is dangerous)."
        ),
        "cohort_n": n,
        "kpis": {
            "avg_mma_urine_mmol_molCr": round(sum(mma_v) / n, 1),     # should be ~2 (NORMAL)
            "avg_homocysteine_umol_l": round(sum(hcy_v) / n, 1),       # higher than cblE/cblG
            "avg_methionine_umol_l": round(sum(met_v) / n, 1),
            "avg_c3_umol_l": round(sum(c3_v) / n, 1),                  # NORMAL
            "avg_serum_folate_nmol_l": round(sum(fol_v) / n, 1),       # LOW/NORMAL (not high)
            "avg_hemoglobin_g_dl": round(sum(hgb_v) / n, 1),           # higher than cblE/cblG
            "avg_free_carnitine_umol_l": round(sum(car_v) / n, 1),
            "pct_seizures": pct(lambda p: p["seizures"]),
            "pct_white_matter_disease": pct(lambda p: p["white_matter_disease"]),
            "pct_thromboembolism": pct(lambda p: p["thromboembolism"]),
            "pct_psychiatric_features": pct(lambda p: p["psychiatric_features"]),
            "pct_peripheral_neuropathy": pct(lambda p: p["peripheral_neuropathy"]),
            "pct_megaloblastic_anemia": pct(lambda p: p["megaloblastic_anemia"]),  # near-zero expected
            "pct_nbs_detected": pct(lambda p: p["nbs_detected"]),        # near-zero expected
            "pct_ohcbl_response": pct(lambda p: p["ohcbl_response"]),    # lower — Level B
            "pct_betaine_response": pct(lambda p: p["betaine_response"]),
            "pct_riboflavin_response": pct(lambda p: p["riboflavin_response"]),
            "pct_mma_normal": 100,        # ALL MTHFR: MMA arm intact
            "pct_mecbl_normal": 100,      # ALL MTHFR: cobalamin system intact (KEY vs cblE/cblG)
            "pct_adocbl_normal": 100,     # ALL MTHFR: AdoCbl normal
            "pct_c3_normal": 100,         # ALL MTHFR: NBS invisible
        },
        "phenotype_distribution": {
            "neonatal_severe": {"n": len(neo_pts), "pct": round(len(neo_pts) / n * 100)},
            "infantile_classic": {"n": len(inf_pts), "pct": round(len(inf_pts) / n * 100)},
            "late_onset_attenuated": {"n": len(late_pts), "pct": round(len(late_pts) / n * 100)},
            "adult_onset_psychiatric": {"n": len(adult_pts), "pct": round(len(adult_pts) / n * 100)},
        },
    }


def get_breakdown() -> dict:
    n = len(_PATIENTS)

    def pct(pred): return round(sum(1 for p in _PATIENTS if pred(p)) / n * 100)

    seizure_types = [
        {"type": "Neonatal seizures (encephalopathy, neonatal severe)", "pct": pct(
            lambda p: p["seizures"] and p["phenotype"] == "Neonatal Severe")},
        {"type": "Infantile Spasms (West Syndrome pattern)", "pct": pct(
            lambda p: p["seizures"] and p["phenotype"] == "Infantile Classic")},
        {"type": "Myoclonic seizures (infantile / neonatal)", "pct": pct(
            lambda p: p["seizures"] and p["phenotype"] in ("Neonatal Severe", "Infantile Classic"))},
        {"type": "Focal seizures (late-onset / cognitive decline)", "pct": pct(
            lambda p: p["seizures"] and p["phenotype"] in ("Late-Onset Attenuated", "Adult-Onset Psychiatric"))},
        {"type": "Generalized tonic-clonic", "pct": 20},
        {"type": "Absence seizures (late-onset)", "pct": 8},
    ]

    metabolic_triggers = [
        {"trigger": "Antifolates (methotrexate, trimethoprim, DHFR inhibitors)", "pct": 92,
         "mechanism": (
             "DHFR inhibition blocks folic acid → dihydrofolate → tetrahydrofolate conversion, "
             "depleting the THF/5,10-methyleneTHF pool that MTHFR acts on. "
             "In MTHFR deficiency, MTHFR cannot produce 5-methylTHF at all — any additional "
             "upstream folate depletion by antifolates completely collapses the folate supply, "
             "worsening tHcy acutely and increasing white matter injury risk. "
             "HIGH RISK. If antifolates unavoidable: mandatory folinic acid and close tHcy monitoring."
         )},
        {"trigger": "Missed betaine + methionine supplement doses", "pct": 88,
         "mechanism": (
             "Betaine (TMG) is the PRIMARY HHcy control in MTHFR — BHMT bypasses the MTHFR block. "
             "tHcy rebounds rapidly (~24–48 h after missed betaine in severe MTHFR). "
             "Methionine supplement is the primary methionine source (MTR substrate-starved in MTHFR). "
             "Both are mandatory and synergistic — neither may be omitted. "
             "Riboflavin B2 must also be maintained (FAD-dependent MTHFR stability)."
         )},
        {"trigger": "Riboflavin (B2) deficiency (dietary or intercurrent illness)", "pct": 72,
         "mechanism": (
             "MTHFR is FAD-dependent — riboflavin deficiency worsens MTHFR enzyme activity. "
             "In severe MTHFR deficiency, B2 supplementation can stabilize even null-like variants "
             "by stabilizing residual FAD binding. B2 depletion from illness, diarrhea, or "
             "valproate use (VPA depletes riboflavin) can acutely worsen tHcy. "
             "VPA HIGH RISK specifically because it depletes riboflavin (FAD-MTHFR cofactor)."
         )},
        {"trigger": "N2O exposure (anesthesia)", "pct": 68,
         "mechanism": (
             "N2O irreversibly oxidizes cob(I)alamin on MTR → cob(III)alamin (inactive). "
             "In MTHFR deficiency, MTR cobalamin is INTACT — but MTR is ALREADY substrate-starved "
             "(no 5-methylTHF methyl donor). N2O adds a second block (MTR cofactor destroyed) → "
             "acute double-depletion: no methyl donor (MTHFR blocked) + MTR cofactor destroyed. "
             "Result: acute catastrophic HHcy crisis. HIGH RISK (not absolute CI as in cblE/cblG "
             "because MTR cobalamin is initially intact — but still dangerous in MTHFR). "
             "Notify anesthesiology; consider alternative agents; give OHCbl peri-operatively."
         )},
        {"trigger": "Intercurrent illness / fever (catabolism)", "pct": 58,
         "mechanism": (
             "Protein catabolism increases Hcy load; with MTHFR block, BHMT (betaine-driven) is "
             "the only major remethylation route available. "
             "Catabolic illness can overwhelm BHMT capacity → acute tHcy surge. "
             "Less MMA surge risk than cblC (no MMA arm) but acute HHcy can worsen leukoencephalopathy. "
             "Continue betaine + riboflavin + methionine supplement during illness; IV glucose helpful."
         )},
        {"trigger": "Fasting / inadequate dietary protein and methionine", "pct": 45,
         "mechanism": (
             "Dietary methionine is critical in MTHFR deficiency — de novo synthesis via MTR is blocked. "
             "Fasting → protein catabolism → Hcy release; simultaneously reduced methionine intake. "
             "MODERATE RISK. Maintain regular protein + methionine intake. "
             "IV dextrose + continue betaine during NPO episodes."
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
        "mecbl_normal": True,     # KEY POSITIVE DIFFERENCE vs cblE/cblG
        "adocbl_normal": p["adocbl_normal"],
        "megaloblastic": p["megaloblastic_anemia"],   # near-zero expected
        "wmd": p["white_matter_disease"],
        "thrombo": p["thromboembolism"],
        "psychiatric": p["psychiatric_features"],
        "periph_neuropathy": p["peripheral_neuropathy"],
        "sz": p["seizures"],
        "nbs": p["nbs_detected"],
        "ohcbl_resp": p["ohcbl_response"],
        "betaine_resp": p["betaine_response"],
        "riboflavin_resp": p["riboflavin_response"],
        "aed": p["aed"],
    } for p in _PATIENTS[:sample_size]]

    return {
        "seizure_type_distribution": seizure_types,
        "metabolic_triggers": metabolic_triggers,
        "high_risk_drugs": [
            {
                "drug": "Valproate (VPA)",
                "risk": "HIGH RISK",
                "mechanism": (
                    "VPA depletes RIBOFLAVIN (B2) — the FAD cofactor of MTHFR. "
                    "In MTHFR deficiency, riboflavin supplementation is Level A precisely because "
                    "MTHFR is FAD-dependent; VPA-driven riboflavin depletion worsens MTHFR instability "
                    "and reduces any residual MTHFR activity. "
                    "VPA also depletes carnitine and has antifolic properties. "
                    "LEV is unconditionally first-line AED for ALL seizure types in MTHFR deficiency. "
                    "If VPA used historically, transition to LEV and add riboflavin loading."
                ),
            },
            {
                "drug": "Antifolates (methotrexate, trimethoprim, pyrimethamine, phenytoin)",
                "risk": "HIGH RISK",
                "mechanism": (
                    "DHFR inhibitors block upstream folate conversion: folic acid → DHF → THF → "
                    "5,10-methyleneTHF. In MTHFR deficiency, MTHFR cannot convert the already-low "
                    "5,10-methyleneTHF to 5-methylTHF; antifolates deplete the 5,10-methyleneTHF "
                    "pool that MTHFR acts on → acute tHcy crisis + worsening white matter injury. "
                    "If antifolate unavoidable: mandatory folinic acid co-administration + "
                    "urgent tHcy and betaine monitoring. Never give antifolates without folinic acid in MTHFR."
                ),
            },
            {
                "drug": "Nitrous oxide (N2O)",
                "risk": "HIGH RISK (not absolute CI as in cblE/cblG, but dangerous)",
                "mechanism": (
                    "N2O irreversibly inactivates the cob(I)alamin cofactor of MTR by oxidation. "
                    "In MTHFR deficiency, the cobalamin system is INTACT (unlike cblE/cblG where "
                    "MTR is already non-functional). However, MTR in MTHFR patients is already "
                    "substrate-deprived (no 5-methylTHF from blocked MTHFR). "
                    "N2O adds a second block (MTR cofactor destroyed) on top of the MTHFR block → "
                    "acute catastrophic HHcy. HIGH RISK but not absolute CI (MTR cobalamin starts intact). "
                    "Notify anesthesiology; peri-operative OHCbl IM recommended; prefer propofol/sevoflurane. "
                    "In cblE/cblG: ABSOLUTE CI (MTR already non-functional). "
                    "In MTHFR: HIGH RISK (MTR starts intact but doubly vulnerable)."
                ),
            },
            {
                "drug": "Fasting / NPO without adequate nutrition and betaine",
                "risk": "MODERATE RISK",
                "mechanism": (
                    "Dietary methionine is the primary methionine source in MTHFR (de novo synthesis "
                    "via MTR is functionally blocked — no 5-methylTHF available). "
                    "Fasting reduces methionine intake and increases protein catabolism → Hcy surge. "
                    "With BHMT (betaine) as the only major HHcy buffer, maintaining betaine "
                    "during fasting/illness is critical. IV dextrose + betaine + methionine supplement."
                ),
            },
        ],
        "treatments": [
            {
                "treatment": "Betaine (TMG) 100–200 mg/kg/day",
                "evidence": "Level A",
                "response_pct": 82,
                "note": (
                    "PRIMARY HHcy treatment in MTHFR deficiency — BHMT (betaine:homocysteine methyltransferase) "
                    "remethylates Hcy → methionine entirely bypassing the blocked MTHFR → MTR → Methionine pathway. "
                    "MANDATORY — betaine is the most effective single treatment in MTHFR. "
                    "Continue throughout life regardless of apparent HHcy control. "
                    "BHMT operates completely independently of MTHFR, MTR, and the cobalamin system. "
                    "Monitor tHcy + methionine at Day 7 and monthly; target tHcy <50 µmol/L."
                ),
            },
            {
                "treatment": "Methionine supplementation (oral L-methionine) 10–30 mg/kg/day",
                "evidence": "Level A",
                "response_pct": 78,
                "note": (
                    "UNIQUE to MTHFR — NOT needed in cblE/cblG (those disorders produce adequate methionine "
                    "when OHCbl works; the betaine BHMT route provides methionine too). "
                    "In MTHFR: de novo methionine synthesis via MTR is blocked (no 5-methylTHF substrate) → "
                    "methionine must come from diet + betaine-BHMT route. "
                    "Methionine supplementation directly corrects methionine deficiency → improves SAM → "
                    "improves myelination, methylation reactions, cognitive function. "
                    "Monitor plasma methionine — target 20–40 µmol/L (avoid methionine toxicity >100 µmol/L)."
                ),
            },
            {
                "treatment": "Riboflavin (B2) 5–20 mg/day",
                "evidence": "Level A",
                "response_pct": 62,
                "note": (
                    "UNIQUE to MTHFR — NOT used in cblE/cblG (those enzymes do not depend on FAD). "
                    "MTHFR is a flavoprotein — FAD (riboflavin-derived) is essential for MTHFR catalysis. "
                    "B2 supplementation: (1) stabilizes the FAD-MTHFR interaction — even in severe variants, "
                    "some residual FAD binding may provide marginal activity; "
                    "(2) thermolabile variants (C677T) are strongly B2-responsive — B2 supplementation can "
                    "normalize their activity at physiological temperature; "
                    "(3) even in null/near-null variants, riboflavin loading rarely worsens and may help. "
                    "Monitor serum riboflavin or erythrocyte glutathione reductase (EGR) activity coefficient."
                ),
            },
            {
                "treatment": "Folinic acid 0.5–5 mg/day",
                "evidence": "Level B",
                "response_pct": 55,
                "note": (
                    "Level B in MTHFR (DIFFERENT from Level A in cblE/cblG). "
                    "In cblE/cblG: methylfolate TRAP is the PRIMARY mechanism → folinic acid Level A "
                    "to relieve functional THF depletion. "
                    "In MTHFR: no methylfolate trap (5-methylTHF is not accumulating — it is not being made). "
                    "THF cycle is less critically depleted in MTHFR (5,10-methyleneTHF accumulates → can "
                    "still feed thymidylate synthase TYMS → no megaloblastic anemia → less urgent need). "
                    "Folinic acid helps replenish the THF pool when dietary folate is low; Level B. "
                    "Some centers use 5-methylTHF (methylfolate) directly as it bypasses MTHFR entirely — "
                    "this is rational and increasingly used alongside betaine as the backbone."
                ),
            },
            {
                "treatment": "5-Methyltetrahydrofolate (methylfolate / 5-MTHF) supplementation",
                "evidence": "Level B (emerging, rational bypass)",
                "response_pct": 58,
                "note": (
                    "The PRODUCT of MTHFR — providing it exogenously directly bypasses the MTHFR block. "
                    "5-methylTHF (methylfolate) is the methyl donor for MTR — providing it allows "
                    "MTR to remethylate Hcy even when MTHFR is absent. "
                    "Mechanistically rational and used in some centers. "
                    "Folinic acid (5-formyl-THF) is more widely available and metabolically converted "
                    "to 5-methylTHF via several steps; 5-methylTHF bypasses MTHFR more directly. "
                    "Often used combined with betaine; not yet Level A in published guidelines. "
                    "Distinguish from the misleading 'methylfolate supplementation' for the common "
                    "C677T polymorphism — in SEVERE deficiency, doses and monitoring are very different."
                ),
            },
            {
                "treatment": "Hydroxocobalamin (OHCbl) 1–2 mg/day IM",
                "evidence": "Level B",
                "response_pct": 38,
                "note": (
                    "Level B in MTHFR — DIFFERENT from Level A in cblE/cblG. "
                    "In cblE/cblG: cobalamin is the direct treatment (the defect is in cobalamin cofactor "
                    "maintenance — MTR cofactor). OHCbl is primary. "
                    "In MTHFR: the entire cobalamin system is INTACT. MTR is present, MTRR is intact, "
                    "MeCbl loading proceeds normally. The problem is upstream substrate deprivation "
                    "(no 5-methylTHF). OHCbl does not provide the missing 5-methylTHF substrate. "
                    "OHCbl is used adjunctively to ensure optimal MTR cobalamin saturation, "
                    "especially if there is any concern about cobalamin adequacy. "
                    "Betaine (Level A) is far more effective than OHCbl in MTHFR."
                ),
            },
            {
                "treatment": "L-Carnitine 50–100 mg/kg/day",
                "evidence": "Level B",
                "response_pct": 50,
                "note": (
                    "Mild secondary carnitine depletion possible (no MMA arm — less severe than cblC/combined). "
                    "Monitor free carnitine; supplement if low. "
                    "VPA-induced carnitine depletion is an additional risk if VPA was previously used. "
                    "Level B — not as critical as in combined MMA+HHcy disorders."
                ),
            },
            {
                "treatment": "ACTH / Vigabatrin (infantile spasms)",
                "evidence": "Level A (IS)",
                "response_pct": 52,
                "note": (
                    "ACTH or vigabatrin for infantile spasms pattern (West syndrome). "
                    "Combined with betaine + riboflavin + methionine supplement Level A. "
                    "IS in MTHFR is primarily metabolic (HHcy + methionine deficiency → white matter injury). "
                    "Metabolic correction improves seizure burden significantly when started early. "
                    "LEV first-line AED; never VPA (riboflavin-depleting in FAD-dependent MTHFR patient)."
                ),
            },
            {
                "treatment": "VPA — HIGH RISK / AVOID",
                "evidence": "AVOID",
                "response_pct": 0,
                "note": (
                    "HIGH RISK: VPA depletes riboflavin (FAD) — directly worsens MTHFR enzyme stability "
                    "in an already FAD-dependent, FAD-vulnerable enzyme deficiency. "
                    "VPA also depletes carnitine and has antifolic properties. "
                    "This is mechanistically distinct from why VPA is avoided in cblE/cblG "
                    "(where VPA depletes folate in methylfolate-trap patients). "
                    "In MTHFR: VPA depletes RIBOFLAVIN specifically — the FAD cofactor of MTHFR itself. "
                    "LEV is unconditionally first-line AED for ALL seizure types in MTHFR deficiency."
                ),
            },
        ],
        "patient_sample": sample,
    }


def get_definitions() -> dict:
    return {
        "gene_card": {
            "Gene": (
                "MTHFR (Methylenetetrahydrofolate Reductase; cytoplasmic flavoprotein; FAD cofactor)"
            ),
            "Full Name": (
                "Homocystinuria due to MTHFR deficiency / Severe MTHFR Deficiency "
                "(distinct from common C677T/A1298C polymorphisms)"
            ),
            "Chromosome": "1p36.3",
            "Inheritance": "AUTOSOMAL RECESSIVE — biallelic LOF; both sexes equally affected",
            "OMIM Gene": "*607093",
            "OMIM Disease": "#236250 (Homocystinuria due to MTHFR deficiency)",
            "Protein Size": (
                "656 amino acids; ~74 kDa (monomer); active as dimer/octamer; cytoplasmic flavoprotein; "
                "N-terminal catalytic domain (AA 1–340: FAD binding, NADPH binding, 5,10-methyleneTHF substrate; "
                "most pathogenic variants cluster here) + "
                "C-terminal regulatory domain (AA 341–656: SAM allosteric inhibition site, Ser391 phosphorylation)"
            ),
            "Protein Function": (
                "Catalyzes: 5,10-methyleneTHF + NADPH + FAD → 5-methylTHF (irreversible reduction); "
                "5-methylTHF is the ONLY methyl donor for MTR (methionine synthase); "
                "MTHFR is therefore the upstream gatekeeper of the entire remethylation axis; "
                "the cobalamin system (MMACHC, MTR, MTRR, etc.) is INTACT in MTHFR deficiency"
            ),
            "Pathway Position": (
                "Upstream of MTR (cblG) and MTRR (cblE) in the remethylation axis; "
                "converts 5,10-methyleneTHF → 5-methylTHF (folate cycle step); "
                "without MTHFR, MTR has no methyl donor → isolated HHcy + methionine depletion; "
                "MMA arm (MMUT/AdoCbl) is completely independent of MTHFR"
            ),
            "FAD Cofactor": (
                "MTHFR requires FAD (flavin adenine dinucleotide, riboflavin-derived) as the "
                "electron-transfer cofactor for NADPH-dependent reduction of 5,10-methyleneTHF; "
                "FAD deficiency (riboflavin deficiency) worsens MTHFR activity; "
                "riboflavin (B2) supplementation Level A in severe MTHFR deficiency — "
                "stabilizes FAD binding and may restore partial activity; "
                "thermolabile C677T variant is highly B2-responsive (distinct from severe deficiency)"
            ),
            "SAM Regulation": (
                "SAM (S-adenosylmethionine, from methionine + ATP) allosterically INHIBITS MTHFR "
                "via the C-terminal regulatory domain — product-feedback homeostasis; "
                "in MTHFR deficiency: methionine is LOW → SAM is LOW → MTHFR inhibition relieved → "
                "any residual MTHFR activity is maximally stimulated (no SAM brake); "
                "this is why betaine (raising methionine via BHMT) must be dose-titrated carefully "
                "to avoid SAM excess that might inappropriately suppress residual MTHFR activity"
            ),
            "Prevalence": (
                "Severe deficiency (biallelic LOF): ~1/200,000–400,000 births; "
                "common polymorphisms (C677T, A1298C) are frequent but NOT severe deficiency"
            ),
        },
        "key_concepts": [
            {
                "concept": "MTHFR is UPSTREAM of the cobalamin axis — the entire cobalamin system is intact",
                "explanation": (
                    "MTHFR does not touch cobalamin at all. Its product (5-methylTHF) is the methyl donor "
                    "for MTR, but the cobalamin processing pathway (LMBRD1, ABCD4, MMACHC, MMADHC, MMAB, "
                    "MMUT, MTRR, and MTR itself) is STRUCTURALLY INTACT in MTHFR deficiency. "
                    "MTR enzyme is present; MTRR is functional; MeCbl can be made and loaded onto MTR "
                    "(MTR has cobalamin — it just has no 5-methylTHF methyl donor to work with). "
                    "This is why MeCbl in fibroblasts is NORMAL in MTHFR — the cobalamin loading step "
                    "proceeds normally. Compare: in cblE/cblG, MeCbl is ABSENT because the cobalamin "
                    "maintenance axis (MTRR in cblE; MTR itself in cblG) is defective. "
                    "This makes MeCbl fibroblast assay the single most powerful test to distinguish "
                    "MTHFR from cblE/cblG biochemically."
                ),
            },
            {
                "concept": "Serum folate LOW or NORMAL in MTHFR — not high as in cblE/cblG (KEY DISTINCTION)",
                "explanation": (
                    "In cblE/cblG: the methylfolate TRAP — MTHFR is intact and produces 5-methylTHF normally, "
                    "but MTR (absent or inactive) cannot consume it → 5-methylTHF ACCUMULATES → "
                    "serum folate appears HIGH (methylfolate trap; lab measures all folates including trapped "
                    "5-methylTHF). This is the 'folate paradox': high folate, yet megaloblastic anemia. "
                    "In MTHFR deficiency: MTHFR cannot PRODUCE 5-methylTHF → 5-methylTHF is low or absent → "
                    "serum folate is LOW or NORMAL (less 5-methylTHF to measure). "
                    "The substrate (5,10-methyleneTHF) may accumulate instead (it feeds TYMS → thymidylate). "
                    "Clinical implication: a patient with HHcy + normal MMA + LOW or NORMAL serum folate "
                    "should prompt MTHFR deficiency evaluation; "
                    "HIGH serum folate should prompt cblE/cblG evaluation. "
                    "This single test narrows the differential dramatically."
                ),
            },
            {
                "concept": "No megaloblastic anemia in MTHFR — KEY NEGATIVE vs cblE (90%) and cblG (80%)",
                "explanation": (
                    "In cblE/cblG: methylfolate trap → 5-methylTHF accumulates → THF pool depleted → "
                    "thymidylate synthase (TYMS) starved → impaired dTMP synthesis → megaloblastic anemia. "
                    "In MTHFR deficiency: MTHFR cannot reduce 5,10-methyleneTHF → 5-methylTHF. "
                    "This means 5,10-methyleneTHF ACCUMULATES (substrate of MTHFR piles up). "
                    "TYMS uses 5,10-methyleneTHF directly as its substrate (5,10-methylene-THF + dUMP → THF + dTMP). "
                    "With 5,10-methyleneTHF accumulating in MTHFR deficiency, TYMS is not starved → "
                    "thymidylate synthesis is preserved → NO megaloblastic anemia (or only mild). "
                    "This explains why MTHFR patients do NOT have macrocytic anemia typically — "
                    "the folate cycle shunts toward thymidylate synthesis instead of remethylation. "
                    "Practical: normal MCV in MTHFR is a red flag that distinguishes it from cblE/cblG."
                ),
            },
            {
                "concept": "Leukoencephalopathy PROMINENT in MTHFR — more severe than cblE/cblG",
                "explanation": (
                    "White matter disease is the hallmark radiological feature of MTHFR deficiency, "
                    "more prominent than in cblE/cblG. "
                    "Mechanisms: (1) Methionine deficiency → SAM deficiency → impaired methylation reactions → "
                    "defective myelin synthesis and maintenance (myelin basic protein methylation; phospholipid "
                    "methylation; epigenetic methylation of myelin gene promoters — all SAM-dependent). "
                    "(2) HHcy is directly toxic to white matter endothelium and oligodendrocytes — "
                    "HHcy levels in MTHFR are often higher (50–300 µmol/L) than cblE/cblG (40–200 µmol/L). "
                    "(3) Late/no NBS detection → prolonged HHcy before treatment → cumulative white matter injury. "
                    "Brain MRI: periventricular leukoencephalopathy, subcortical white matter changes, "
                    "posterior > anterior white matter. Severity correlates with tHcy and treatment delay. "
                    "Leukoencephalopathy does NOT distinguish MTHFR from the rest clinically — "
                    "it is a non-specific consequence of HHcy — but is more severe in MTHFR."
                ),
            },
            {
                "concept": "Riboflavin (B2/FAD) Level A — MTHFR-specific (not needed in cblE/cblG)",
                "explanation": (
                    "MTHFR is a flavoprotein: it binds FAD non-covalently as the electron-transfer cofactor "
                    "for the NADPH-dependent reduction of 5,10-methyleneTHF → 5-methylTHF. "
                    "Riboflavin (B2) → riboflavin → FAD (via RFK + FLAD1); B2 deficiency reduces FAD → "
                    "MTHFR loses its cofactor → MTHFR activity falls further. "
                    "B2 supplementation: (1) maximizes FAD supply to stabilize residual MTHFR; "
                    "(2) thermolabile variants (C677T, A1298C) are highly B2-responsive — FAD stabilizes "
                    "the thermolabile conformation at physiological temperature; "
                    "(3) even in severe LOF variants, B2 rarely worsens and may provide marginal benefit. "
                    "The common C677T polymorphism (A222V) is a prototype B2-responsive MTHFR variant — "
                    "this knowledge transfers to severe deficiency: B2 is always indicated. "
                    "VPA HIGH RISK in MTHFR specifically because VPA depletes riboflavin (B2/FAD)."
                ),
            },
            {
                "concept": "Methionine supplementation Level A — MTHFR-specific (not needed in cblE/cblG)",
                "explanation": (
                    "In MTHFR deficiency: MTR is substrate-starved (no 5-methylTHF) → methionine synthesis "
                    "from Hcy is severely impaired → plasma methionine is LOW (5–15 µmol/L). "
                    "Methionine deficiency → SAM deficiency → global methylation impairment → "
                    "white matter disease + neurocognitive effects. "
                    "Direct oral methionine supplementation bypasses the MTHFR → MTR → methionine route "
                    "and directly corrects methionine deficiency. "
                    "In cblE (MTRR LOF): betaine-BHMT and OHCbl together provide adequate methionine; "
                    "direct methionine supplementation is generally not needed. "
                    "In cblG (MTR LOF): similar — betaine BHMT provides methionine; direct supplementation "
                    "usually not first-line unless betaine inadequate. "
                    "In MTHFR: the metabolic blockade is more upstream and more severe — betaine + methionine "
                    "supplement together are the backbone treatment. "
                    "Monitor plasma methionine: target 20–40 µmol/L; avoid >100 (methionine toxicity)."
                ),
            },
            {
                "concept": "NBS INVISIBLE — MTHFR deficiency missed by newborn screening",
                "explanation": (
                    "Standard NBS measures C3 (propionylcarnitine) for cobalamin/MMA disorders. "
                    "In MTHFR: MMA arm is COMPLETELY INTACT → C3 is NORMAL → NBS is NEGATIVE. "
                    "No expanded panel currently screens for MTHFR-severe-deficiency reliably. "
                    "Some countries include plasma tHcy in NBS panels — this would detect MTHFR "
                    "but is not universally adopted. "
                    "All MTHFR patients diagnosed clinically: neonatal encephalopathy, white matter disease "
                    "on MRI, developmental regression, or late-onset psychiatric/thrombotic features. "
                    "Never assume normal NBS excludes MTHFR in a child with HHcy + white matter disease. "
                    "The common C677T/A1298C polymorphisms are irrelevant to the severe deficiency NBS issue — "
                    "they would never require NBS; only biallelic severe alleles cause the disease."
                ),
            },
            {
                "concept": "N2O HIGH RISK (not absolute CI) — different from cblE/cblG",
                "explanation": (
                    "In cblE/cblG: MTR enzyme is already non-functional → N2O cannot make it 'more broken'. "
                    "The risk from N2O in cblE/cblG comes from the fact that N2O destroys any residual "
                    "partial MTR activity and any cobalamin-dependent rescue from betaine-adjacent pathways → "
                    "hence ABSOLUTE CONTRAINDICATION. "
                    "In MTHFR: MTR enzyme is INTACT and its cobalamin cofactor starts in the ACTIVE form. "
                    "N2O exposure inactivates the cob(I)alamin cofactor on MTR → cob(III)alamin (inactive). "
                    "Now the patient has: (1) MTHFR blocked (no 5-methylTHF) AND (2) MTR cofactor destroyed. "
                    "Double block → acute catastrophic HHcy. "
                    "HIGH RISK — not absolute CI (because preoperative OHCbl + betaine + methylfolate can "
                    "partially protect) but essentially should be treated as absolute CI in practice. "
                    "Peri-operative: OHCbl IM + betaine maintained + continue methionine supplement. "
                    "Anesthesia note: prefer propofol/sevoflurane; if N2O unavoidable, give high-dose OHCbl."
                ),
            },
            {
                "concept": "Betaine (TMG) — PRIMARY treatment in MTHFR (same Level A as cblE/cblG but more critical)",
                "explanation": (
                    "BHMT (betaine:homocysteine methyltransferase) catalyzes: Hcy + betaine → Met + dimethylglycine. "
                    "This reaction is completely independent of MTHFR, MTR, MTRR, and cobalamin. "
                    "In MTHFR deficiency: betaine-BHMT is the PRIMARY (and most effective) route "
                    "to simultaneously reduce tHcy AND increase methionine. "
                    "In cblE/cblG: betaine is Level A but complements OHCbl (which is primary). "
                    "In MTHFR: betaine is MORE critical because OHCbl (Level B) provides minimal benefit "
                    "(the cobalamin system is intact — adding more cobalamin does not bypass MTHFR block). "
                    "Betaine dose: 100–200 mg/kg/day in children; 3–9 g/day in adults. "
                    "Monitor: tHcy (target <50 µmol/L) + methionine (avoid >100 µmol/L — methionine toxicity)."
                ),
            },
            {
                "concept": "Differential: MTHFR vs CBS (CBS methionine HIGH; MTHFR methionine LOW)",
                "explanation": (
                    "Both MTHFR deficiency and CBS (cystathionine beta-synthase, classical homocystinuria) "
                    "present with elevated tHcy + normal MMA + normal C3. "
                    "KEY DISTINCTION: methionine is the differentiator. "
                    "CBS LOF: transsulfuration of methionine → cysteine is blocked → "
                    "methionine ACCUMULATES → plasma methionine HIGH (>100 µmol/L in classic CBS). "
                    "MTHFR deficiency: remethylation of Hcy → methionine is blocked upstream → "
                    "methionine not synthesized → plasma methionine LOW (5–15 µmol/L). "
                    "Other CBS-specific features absent in MTHFR: marfanoid habitus, ectopia lentis, "
                    "pyridoxine (B6) responsiveness (~50% of CBS), osteoporosis, downward lens dislocation. "
                    "Both have thromboembolism risk from HHcy. Both are NBS-invisible by C3. "
                    "Plasma amino acid profile is diagnostic: methionine high in CBS, low in MTHFR."
                ),
            },
        ],
        "diagnostic_thresholds": [
            {
                "parameter": "Plasma tHcy",
                "threshold": ">50 µmol/L (isolated — MMA normal; often >100 in severe)",
                "action": (
                    "Isolated HHcy + normal MMA → cblE/cblG/MTHFR/cblD-HHcy/CBS differential. "
                    "Higher tHcy (>100 µmol/L) in MTHFR vs cblE/cblG (typically 40–200). "
                    "Check methionine (LOW in MTHFR/cblE/cblG; HIGH in CBS — KEY distinction). "
                    "Check serum folate (LOW/NORMAL in MTHFR; HIGH in cblE/cblG — KEY distinction). "
                    "Start betaine + methionine supplement empirically while awaiting genetics."
                ),
            },
            {
                "parameter": "Urine MMA",
                "threshold": "NORMAL (<5 mmol/mol Cr) in MTHFR — KEY NEGATIVE",
                "action": (
                    "Normal MMA + elevated HHcy → isolated HHcy pattern. "
                    "Excludes cblC, cblX, cblD-combined, cblA, cblB, MMUT. "
                    "Differential: MTHFR, cblE, cblG, cblD-HHcy-only, CBS. "
                    "Proceed to serum folate + MeCbl fibroblasts to sub-classify."
                ),
            },
            {
                "parameter": "Serum folate",
                "threshold": "LOW or NORMAL (5–20 nmol/L) in MTHFR — KEY vs cblE/cblG",
                "action": (
                    "LOW or NORMAL serum folate + HHcy + normal MMA → MTHFR deficiency (not cblE/cblG). "
                    "HIGH serum folate + HHcy + normal MMA → cblE/cblG (methylfolate trap). "
                    "This single test narrows the differential between MTHFR and cblE/cblG dramatically. "
                    "Do NOT conclude 'normal folate, no treatment needed' — MTHFR still needs betaine + B2 + methionine."
                ),
            },
            {
                "parameter": "MeCbl fibroblasts",
                "threshold": "NORMAL in MTHFR (KEY POSITIVE — whole cobalamin system intact)",
                "action": (
                    "MeCbl NORMAL + AdoCbl NORMAL + HHcy elevated + MMA normal = MTHFR deficiency. "
                    "Compare: cblE/cblG → MeCbl ABSENT + AdoCbl NORMAL. "
                    "Compare: cblC → BOTH MeCbl and AdoCbl ABSENT. "
                    "MeCbl normal in fibroblasts effectively rules out cblE, cblG, cblD-HHcy-only "
                    "and strongly supports MTHFR (or CBS, which also has normal fibroblast cobalamin). "
                    "Confirm with MTHFR enzyme activity (leukocytes or fibroblasts) + WES/WGS."
                ),
            },
            {
                "parameter": "MTHFR enzyme activity",
                "threshold": "<5% residual activity in leukocytes or fibroblasts — severe deficiency",
                "action": (
                    "Enzyme activity confirms severity of deficiency and guides prognosis. "
                    "<1%: neonatal severe phenotype expected; urgent betaine + methionine + riboflavin. "
                    "1–5%: infantile classic; aggressive treatment. "
                    "5–20%: late-onset attenuated; may have significant B2 response. "
                    ">20%: adult psychiatric phenotype; B2 + betaine primary. "
                    "Note: common C677T homozygous typically 30–50% activity — NOT severe deficiency."
                ),
            },
            {
                "parameter": "MCV (mean corpuscular volume)",
                "threshold": "NORMAL (80–96 fL) in MTHFR — KEY NEGATIVE vs cblE/cblG",
                "action": (
                    "Normal MCV + HHcy + normal MMA + low/normal folate → MTHFR differential (not cblE/cblG). "
                    "Macrocytic anemia (MCV >98 fL) + HHcy + high serum folate → cblE/cblG (methylfolate trap). "
                    "Always check blood smear: no hypersegmented neutrophils in MTHFR "
                    "(distinguish from nutritional B12 deficiency which also has normal MeCbl in fibroblasts)."
                ),
            },
            {
                "parameter": "Brain MRI",
                "threshold": "Periventricular leukoencephalopathy (PROMINENT in MTHFR)",
                "action": (
                    "Leukoencephalopathy + HHcy + normal MMA + normal MCV = MTHFR pattern. "
                    "White matter disease more prominent in MTHFR than cblE/cblG due to "
                    "higher HHcy burden and methionine/SAM deficiency. "
                    "MRI severity correlates with treatment delay — underscores need for early metabolic control. "
                    "Initiate betaine + riboflavin + methionine supplement urgently; "
                    "white matter damage is partially reversible with metabolic correction."
                ),
            },
            {
                "parameter": "OHCbl trial response",
                "threshold": "POOR response to OHCbl alone (Level B in MTHFR) — different from cblE/cblG",
                "action": (
                    "If tHcy does not fall significantly after 7 days of OHCbl alone → suspect MTHFR "
                    "(not cblE/cblG where OHCbl is Level A and typically shows 50–75% HHcy reduction). "
                    "In MTHFR: add betaine + methionine + riboflavin; betaine will be the primary responder. "
                    "Monitor: tHcy reduction with betaine (target >50% at 2 weeks); "
                    "methionine levels (target 20–40 µmol/L); riboflavin levels (EGR assay)."
                ),
            },
            {
                "parameter": "NBS C3",
                "threshold": "NORMAL (<3.5 µmol/L) — MTHFR NOT detected by NBS",
                "action": (
                    "Normal NBS C3 does NOT exclude MTHFR. MTHFR deficiency is NBS-INVISIBLE. "
                    "Any infant with unexplained leukoencephalopathy + HHcy + normal MCV "
                    "needs plasma tHcy + serum folate + MeCbl fibroblasts measured urgently. "
                    "Do not be reassured by normal NBS in a neonate with white matter disease."
                ),
            },
        ],
        "differential_diagnosis": [
            {
                "disease": "cblE (MTRR — 5p15.31)",
                "distinguishing": (
                    "KEY DIFFERENCES from MTHFR: "
                    "(1) MeCbl ABSENT in cblE fibroblasts vs NORMAL in MTHFR — "
                    "this is THE single most powerful distinguishing test. "
                    "(2) Serum folate HIGH in cblE (methylfolate trap: 5-methylTHF accumulates because "
                    "MTR cannot consume it — MTHFR is intact) vs LOW/NORMAL in MTHFR. "
                    "(3) Megaloblastic anemia in ~90% of cblE vs ABSENT/RARE in MTHFR. "
                    "(4) OHCbl Level A in cblE; Level B in MTHFR. "
                    "(5) Riboflavin not indicated in cblE; Level A in MTHFR. "
                    "Similarities: both have isolated HHcy, normal MMA, low methionine, NBS-invisible. "
                    "Treatment: betaine Level A for both; folinic acid Level A for cblE, Level B for MTHFR."
                ),
            },
            {
                "disease": "cblG (MTR — 1q43)",
                "distinguishing": (
                    "KEY DIFFERENCES from MTHFR: "
                    "(1) MeCbl ABSENT in cblG fibroblasts vs NORMAL in MTHFR — same as cblE distinction. "
                    "(2) Serum folate HIGH in cblG (methylfolate trap — 5-methylTHF accumulates "
                    "because absent MTR cannot consume it; MTHFR is intact in cblG) "
                    "vs LOW/NORMAL in MTHFR. "
                    "(3) Megaloblastic anemia in ~80% of cblG vs ABSENT/RARE in MTHFR. "
                    "(4) OHCbl Level A in cblG; Level B in MTHFR. "
                    "Mechanistically: in cblG, MTHFR is intact and produces 5-methylTHF normally — "
                    "the defect is in MTR using it. In MTHFR: MTR is intact but starved of substrate. "
                    "Gene: MTR 1q43 vs MTHFR 1p36.3. Both are AR."
                ),
            },
            {
                "disease": "CBS — Classical Homocystinuria (21q22.3)",
                "distinguishing": (
                    "KEY DIFFERENCE: methionine HIGH in CBS (>100 µmol/L; transsulfuration blocked → "
                    "methionine accumulates) vs methionine LOW in MTHFR (5–15 µmol/L; "
                    "remethylation blocked upstream → methionine not synthesized from Hcy). "
                    "Both have isolated HHcy + normal MMA. "
                    "CBS-specific features absent in MTHFR: marfanoid habitus, ectopia lentis (downward), "
                    "osteoporosis, pyridoxine (B6) responsiveness (~50% of CBS). "
                    "Serum folate: NORMAL in CBS (no folate cycle involvement) vs LOW/NORMAL in MTHFR. "
                    "MeCbl fibroblasts: NORMAL in both CBS and MTHFR (cobalamin intact in both). "
                    "Plasma amino acid methionine level is the rapid differentiating test."
                ),
            },
            {
                "disease": "cblC (MMACHC — 1p34.1)",
                "distinguishing": (
                    "KEY DIFFERENCE: cblC has COMBINED MMA+HHcy — MMA ELEVATED in cblC, NORMAL in MTHFR. "
                    "AdoCbl ABSENT in cblC fibroblasts; NORMAL in MTHFR (cobalamin system intact). "
                    "MeCbl ABSENT in cblC; NORMAL in MTHFR. "
                    "NBS detects cblC (C3 elevated); misses MTHFR (C3 normal). "
                    "Maculopathy common in cblC (~80%); absent in MTHFR. "
                    "Serum folate: secondary methylfolate trap elevation in cblC (secondary); "
                    "LOW/NORMAL in MTHFR. "
                    "Megaloblastic anemia: prominent in cblC (global cobalamin failure); "
                    "absent/rare in MTHFR."
                ),
            },
            {
                "disease": "cblX (HCFC1 — Xq28)",
                "distinguishing": (
                    "KEY DIFFERENCE: cblX has COMBINED MMA+HHcy — MMA ELEVATED; MTHFR has NORMAL MMA. "
                    "cblX is X-linked (males); MTHFR is AR (both sexes). "
                    "Both MeCbl and AdoCbl absent in cblX; both NORMAL in MTHFR. "
                    "NBS detects cblX (C3 elevated); misses MTHFR (C3 normal). "
                    "Leukoencephalopathy present in both but mechanistically different."
                ),
            },
            {
                "disease": "Nutritional B12 deficiency (vegan/maternal)",
                "distinguishing": (
                    "Nutritional B12 deficiency → reduced cobalamin → MTR cobalamin-depleted → HHcy + "
                    "megaloblastic anemia (methylfolate trap from MTR dysfunction). "
                    "KEY DIFFERENCES: "
                    "(1) Serum B12 LOW in nutritional deficiency; NORMAL or HIGH in MTHFR "
                    "(cobalamin absorption intact). "
                    "(2) MeCbl in fibroblasts: NORMAL in both (in nutritional deficiency, cobalamin "
                    "is just low — the MTHFR/cobalamin processing system is intact; add cobalamin → normal). "
                    "(3) Response to IM B12 replacement: COMPLETE and RAPID in nutritional deficiency; "
                    "POOR in MTHFR (betaine is what works in MTHFR). "
                    "(4) Serum folate: HIGH in nutritional B12 deficiency (methylfolate trap from "
                    "cobalamin-depleted MTR — same mechanism as cblE/cblG) vs LOW/NORMAL in MTHFR. "
                    "(5) Maternal B12 deficiency context — key clinical history."
                ),
            },
            {
                "disease": "cblD-HHcy-only (MMADHC N-terminus — 2q23.2)",
                "distinguishing": (
                    "cblD-HHcy-only: MMADHC N-terminal variants block cytoplasmic cobalamin distribution "
                    "to MTR → isolated HHcy + MeCbl absent + AdoCbl normal in fibroblasts (same as cblE/cblG). "
                    "KEY DIFFERENCE from MTHFR: MeCbl ABSENT in cblD-HHcy fibroblasts vs NORMAL in MTHFR. "
                    "Serum folate: HIGH in cblD-HHcy (methylfolate trap — MTR blocked) vs LOW/NORMAL in MTHFR. "
                    "Both have isolated HHcy, normal MMA, NBS-invisible. "
                    "Gene panel (MMADHC vs MTHFR) required for definitive distinction."
                ),
            },
            {
                "disease": "MTHFR thermolabile polymorphism (C677T / A1298C — NOT severe deficiency)",
                "distinguishing": (
                    "C677T (p.Ala222Val) homozygous: 50–70% residual MTHFR activity; mild HHcy "
                    "(typically 15–30 µmol/L, rarely >50); B2-responsive; no epilepsy, no white matter disease. "
                    "A1298C (p.Glu429Ala) homozygous: mild activity reduction only; alone almost no phenotype. "
                    "Compound C677T/A1298C: moderate HHcy (20–40 µmol/L typically). "
                    "SEVERE MTHFR DEFICIENCY: tHcy 50–300 µmol/L; enzyme activity <5%; "
                    "epilepsy, leukoencephalopathy, neonatal encephalopathy; biallelic severe LOF alleles. "
                    "The common polymorphisms are public health variants, not rare metabolic diseases. "
                    "Clinically: polymorphisms do not cause the severe neurological disease; "
                    "severe alleles (p.Arg249Gln, p.Gly364Ala, c.1027T>A etc.) do. "
                    "Both respond to riboflavin; only severe deficiency needs betaine + methionine supplement."
                ),
            },
            {
                "disease": "MTHFR vs cblE/cblG — serum folate and MeCbl are the two key differentiators",
                "distinguishing": (
                    "All three — MTHFR, cblE, cblG — present with isolated HHcy + normal MMA + "
                    "low methionine + NBS-invisible. "
                    "Two tests separate them: "
                    "(1) Serum folate: HIGH in cblE/cblG (methylfolate trap — 5-methylTHF accumulates "
                    "because MTR cannot consume it, but MTHFR produces it normally). "
                    "LOW or NORMAL in MTHFR (5-methylTHF not produced). "
                    "(2) MeCbl in fibroblasts: ABSENT in cblE/cblG (cobalamin axis is the defect). "
                    "NORMAL in MTHFR (cobalamin axis is intact — the defect is upstream in folate). "
                    "Additional distinguishers: "
                    "Megaloblastic anemia — yes in cblE/cblG (methylfolate trap deplete THF); "
                    "no in MTHFR (5,10-methyleneTHF feeds TYMS directly). "
                    "Riboflavin — Level A in MTHFR (FAD-dependent); not indicated in cblE/cblG. "
                    "OHCbl — Level A in cblE/cblG; Level B in MTHFR. "
                    "Methionine supplement — Level A in MTHFR; generally not needed in cblE/cblG."
                ),
            },
        ],
    }


# ── entry point (for quick local verification) ───────────────────────────────

if __name__ == "__main__":
    import json
    overview = get_overview()
    print(f"Gene: {overview['gene']}  |  N={overview['cohort_n']}")
    print(f"Avg tHcy: {overview['kpis']['avg_homocysteine_umol_l']} µmol/L  "
          f"(expected higher than cblE/cblG 40–200 range)")
    print(f"Avg serum folate: {overview['kpis']['avg_serum_folate_nmol_l']} nmol/L  "
          f"(expected LOW/NORMAL — KEY DISTINCTION from cblE/cblG where HIGH)")
    print(f"Avg methionine: {overview['kpis']['avg_methionine_umol_l']} µmol/L")
    print(f"Avg hemoglobin: {overview['kpis']['avg_hemoglobin_g_dl']} g/dL  "
          f"(expected normal — no megaloblastic anemia)")
    print(f"% seizures: {overview['kpis']['pct_seizures']}%")
    print(f"% white matter disease: {overview['kpis']['pct_white_matter_disease']}%")
    print(f"% megaloblastic anemia: {overview['kpis']['pct_megaloblastic_anemia']}%  "
          f"(expected ~5–10% — KEY NEGATIVE vs cblE/cblG 80–90%)")
    print(f"% MeCbl normal: {overview['kpis']['pct_mecbl_normal']}%  "
          f"(expected 100% — KEY POSITIVE DIFFERENCE vs cblE/cblG 0%)")
    print(f"% MMA normal: {overview['kpis']['pct_mma_normal']}%  (expected 100%)")
    print(f"% NBS detected: {overview['kpis']['pct_nbs_detected']}%  (expected ~0%)")
    print(f"Phenotypes: {overview['phenotype_distribution']}")
    print("\nBreakdown treatment count:", len(get_breakdown()["treatments"]))
    print("Definitions differential dx count:", len(get_definitions()["differential_diagnosis"]))
    print("\nAll OK.")
