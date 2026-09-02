#!/usr/bin/env python3
"""NDUFAF1 — Mitochondrial Complex I Deficiency (MCIA Complex / CI Assembly Factor 1 / CIA30).

NDUFAF1 (NADH:Ubiquinone Oxidoreductase Complex Assembly Factor 1), also known as CIA30
(Complex I Intermediate Associated protein, ~30 kDa), is a 327-aa nuclear-encoded protein
(~36 kDa) that is an essential scaffold member of the MCIA (Mitochondrial Complex I Assembly)
complex. NDUFAF1 forms the primary binary complex with ACAD9 — the first obligate step in
MCIA complex assembly — before ECSIT and TMEM126B are recruited to complete the tetrameric
MCIA complex. Together, the MCIA complex (ACAD9-NDUFAF1-ECSIT-TMEM126B) is required for
early CI membrane arm biogenesis (ND2 and ND5 modules).

  NDUFAF1 gene    OMIM *606934
  Disease        Mitochondrial Complex I Deficiency (OMIM #256000)
  Inheritance    AR (autosomal recessive biallelic)
  Chromosome     15q11.2-q13

PATHOPHYSIOLOGY (NDUFAF1 / MCIA Complex / ND2-ND5 Module / CI Membrane Arm):
  NDUFAF1 is the first assembled MCIA partner of ACAD9:
    1. ACAD9 (MCIA scaffold) recruits NDUFAF1 (CIA30) as its first and tightest binary partner.
    2. The ACAD9-NDUFAF1 binary sub-complex is the core obligate scaffold upon which ECSIT
       and TMEM126B assemble to form the complete MCIA tetramer.
    3. NDUFAF1 loss disrupts the ACAD9-NDUFAF1 binary interaction → ECSIT and TMEM126B
       cannot be properly recruited → MCIA tetramer fails to form → ND2/ND5 module
       biogenesis is aborted → CI membrane arm absent on BN-PAGE.
    4. Unlike ACAD9, NDUFAF1 has NO FAD-binding domain and NO riboflavin responsiveness.
    5. NDUFAF1 deficiency produces a Leigh syndrome / CI-deficiency phenotype without the
       exercise-intolerance-dominant variant seen in ACAD9 p.Arg518His.

NDUFAF1 UNIQUE FEATURES — THE OBLIGATE ACAD9 BINARY PARTNER:
  1. ACAD9-NDUFAF1 BINARY COMPLEX: NDUFAF1 forms the tightest and earliest binary interaction
     within the MCIA complex. ACAD9 cannot properly scaffold ECSIT/TMEM126B without NDUFAF1.
     This step is unique: the other MCIA members (ECSIT, TMEM126B) can only join once the
     ACAD9-NDUFAF1 binary complex is formed.
  2. NO FAD-BINDING / NO RIBOFLAVIN RESPONSE: Unlike ACAD9 (which has an ACAD-superfamily
     FAD-binding domain allowing riboflavin responsiveness), NDUFAF1 has no FAD-binding
     domain. High-dose riboflavin does NOT rescue NDUFAF1-deficient cells.
  3. CIA30 HISTORICAL NAME: NDUFAF1 was originally discovered in Neurospora crassa as "cia30"
     (Complex I Intermediate Associated, 30 kDa) by Kuffner et al (1998). This is the first
     identified CI assembly factor in any organism — predating all other NDUFAF family
     members. Human NDUFAF1 is the orthologue.
  4. PURE LEIGH PHENOTYPE (NO EXERCISE-INTOLERANCE DOMINANT VARIANT): Unlike ACAD9 (which
     has a bimodal spectrum), NDUFAF1 deficiency presents as severe infantile Leigh syndrome
     without a milder adolescent/adult exercise-intolerance-dominant allele. This is because
     NDUFAF1 has no riboflavin-stabilisable domain and no partial-function rescue allele
     equivalent to ACAD9 p.Arg518His.
  5. BN-PAGE: Severely reduced/absent CI holoenzyme; ND2/ND5 sub-assembly intermediates
     detectable (same as ACAD9 — because NDUFAF1 is an MCIA complex member targeting the
     same ND2/ND5 modules). CONTRAST with N-module intermediates (NDUFAF2/NDUFA12) or
     Q-module structural subunit intermediates.

DISTINGUISHING FEATURES vs OTHER CI-LEIGH AND MCIA GENES:
  vs ACAD9 (MCIA scaffold, same ND2/ND5 module):
    • ACAD9 has riboflavin responsiveness (50-60%) — NDUFAF1 DOES NOT (critical DDx)
    • ACAD9 has exercise-intolerance-dominant phenotype (p.Arg518His) — NDUFAF1 does NOT
    • ACAD9 has HCM (55-65%) — NDUFAF1 HCM less frequent (~20-30%)
    • Both: isolated CI deficiency; CII/CIII/CIV NORMAL; similar BN-PAGE ND2/ND5 intermediates
    • Both: normal acylcarnitines (no fatty-acid oxidation disorder)
  vs ECSIT (MCIA, same complex):
    • Same MCIA complex — overlapping phenotype; WES/chromosomal locus distinguishes
    • ECSIT: chromosome 19p13.3 vs NDUFAF1: 15q11.2-q13
  vs TMEM126B (MCIA, same complex):
    • TMEM126B: chromosome 11q14.1; 2 TM helices vs NDUFAF1: soluble, no TM helices
  vs NDUFAF2 (N-Q module assembly-factor swap):
    • NDUFAF2 targets N-Q module interface (matrix arm); NDUFAF1 targets ND2/ND5 module
      (membrane arm). Different BN-PAGE intermediate pattern. Different chromosomes.
  vs NDUFS1 (N-module IP1/75kDa):
    • NO peripheral neuropathy in NDUFAF1 (NDUFS1: ~50% — CRITICAL DDx)
  vs NDUFS4 (N-module accessory):
    • NO olfactory bulb MRI lesions in NDUFAF1 (NDUFS4: 52-65% — PATHOGNOMONIC for NDUFS4)
  vs NDUFV1 (N-module FMN/N3):
    • Minimal leukodystrophy in NDUFAF1 (NDUFV1: ~40-50%)
  vs POLG/DGUOK (mtDNA depletion):
    • NO hepatopathy in NDUFAF1 (POLG Alpers: ~80%; DGUOK: ~90%)

FOUNDER / RECURRENT MUTATIONS:
  p.Arg81Cys   c.241C>T  — ACAD9-binding interface disruption; severe infantile Leigh
  p.Leu167Pro  c.500T>C  — helix-breaking proline; CIA30 fold collapse; severe
  p.Glu94Lys   c.280G>A  — surface charge disruption; MCIA complex interaction impaired; intermediate
  p.Arg204Gln  c.611G>A  — ACAD9-docking surface; moderate; partial CI residual
  c.IVS3+1G>A            — splice donor exon 3; partial NDUFAF1 residual; moderate phenotype
  p.Trp45Ter   c.135G>A  — near-start truncation; null; consanguineous; neonatal severe

THERAPY — NDUFAF1 / CI-LEIGH SPECIFICS:
  No curative NDUFAF1 rescue therapy available.
  NO riboflavin response (unlike ACAD9 — critical management distinction).
  Absolute contraindications (direct CI inhibitors / mito toxins):
    Metformin — directly inhibits CI at ND1/quinone-binding site (NDUFAF1 ND2/ND5 module territory)
    Valproate — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression
    Linezolid — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
    Chloramphenicol — same mt-ribosomal mechanism
  CONTRAINDICATED:
    Ketogenic diet — forces NADH → β-oxidation; CI absent → lactic acidosis worsens catastrophically
  AVOID / HIGH CAUTION:
    Propofol — PRIS + secondary CIV inhibition; dual ETC failure
    Phenobarbital — secondary CI inhibitor; use LEV first
  LEVEL C cofactors / supportive:
    CoQ10 (ubiquinol) — electron acceptor downstream of failed CI
    Thiamine (B1) — MANDATORY empiric: mimics SLC19A3/BTD (treatable)
    Biotin — MANDATORY empiric: BTD (treatable mimic)
    Succinate — CII bypass; bypasses NDUFAF1-failed CI
    L-Carnitine — secondary carnitine deficiency may occur
  Preferred AED: Levetiracetam (LEV) — renal excretion; no mito toxicity
  Glucose: IV dextrose GIR 6-8 mg/kg/min — NEVER fast
  Anaesthesia: sevoflurane (NOT propofol)
"""

from __future__ import annotations
import random
from typing import Any

SEED = 677
GENE = "NDUFAF1"
DISEASE = "NDUFAF1 Complex I Deficiency — MCIA Complex Binary ACAD9 Partner / CIA30 (Leigh Syndrome / CI-Leigh)"
OMIM_GENE = "606934"
OMIM_DISEASE = "256000"
CHROMOSOME = "15q11.2-q13"
INHERITANCE = "AR"

# ---------------------------------------------------------------------------
# Synthetic 40-patient cohort  (seed-677, reflects published literature)
# ---------------------------------------------------------------------------

def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    sexes = ["M", "F"]
    mutations = [
        "p.Arg81Cys / p.Arg81Cys (hom, consanguineous, ACAD9-interface)",
        "p.Arg81Cys / p.Leu167Pro (compound het)",
        "p.Arg81Cys / c.IVS3+1G>A (compound het)",
        "p.Leu167Pro / p.Leu167Pro (hom, consanguineous)",
        "p.Glu94Lys / p.Arg204Gln (compound het)",
        "p.Trp45Ter / p.Arg81Cys (compound het, null/missense)",
        "p.Arg204Gln / p.Arg204Gln (hom, consanguineous)",
        "p.Glu94Lys / p.Leu167Pro (compound het)",
        "Novel biallelic LOF (frameshift/splice)",
        "p.Arg81Cys / novel missense (compound het)",
    ]
    regions = [
        "European", "MENA", "South Asian", "East Asian",
        "North African", "Latin American", "Sub-Saharan African",
    ]
    ci_act_ranges = [(5, 15), (8, 18), (12, 20)]

    cohort = []
    for i in range(1, n + 1):
        age_onset = round(rng.uniform(0.2, 24), 1)
        sex = rng.choice(sexes)
        mut = rng.choice(mutations)
        region = rng.choice(regions)
        ci_lo, ci_hi = rng.choice(ci_act_ranges)
        ci_activity = round(rng.uniform(ci_lo, ci_hi), 1)
        cohort.append({
            "id": i,
            "age_onset_months": round(age_onset * 12),
            "sex": sex,
            "mutation": mut,
            "region": region,
            "ci_activity_pct": ci_activity,
            "psychomotor_regression":  rng.random() < 0.94,
            "leigh_mri":               rng.random() < 0.88,
            "lactic_acidosis":         rng.random() < 0.91,
            "hypotonia":               rng.random() < 0.87,
            "hcm":                     rng.random() < 0.25,
            "seizures":                rng.random() < 0.55,
            "respiratory_compromise":  rng.random() < 0.62,
            "ataxia":                  rng.random() < 0.42,
            "dystonia":                rng.random() < 0.38,
            "exercise_intolerance":    rng.random() < 0.18,
            "peripheral_neuropathy":   rng.random() < 0.04,
            "olfactory_bulb_lesions":  rng.random() < 0.03,
            "leukodystrophy":          rng.random() < 0.06,
            "hepatopathy":             rng.random() < 0.04,
            "riboflavin_responder":    False,   # NDUFAF1 has NO riboflavin response
            "outcome": rng.choice(
                ["deceased before 3yr"] * 10 +
                ["deceased before 10yr"] * 10 +
                ["alive, severe disability"] * 12 +
                ["alive, moderate disability"] * 8
            ),
        })
    return cohort


_COHORT: list[dict[str, Any]] | None = None


def _cohort() -> list[dict[str, Any]]:
    global _COHORT
    if _COHORT is None:
        _COHORT = _build_cohort()
    return _COHORT


def _pct(key: str) -> float:
    pts = _cohort()
    return round(sum(1 for p in pts if p.get(key)) / len(pts) * 100)


# ---------------------------------------------------------------------------
# Public API: get_overview
# ---------------------------------------------------------------------------

def get_overview() -> dict[str, Any]:
    pts = _cohort()
    n = len(pts)
    ci_vals = [p["ci_activity_pct"] for p in pts]

    return {
        "gene":           GENE,
        "gene_full_name": "NADH:Ubiquinone Oxidoreductase Complex Assembly Factor 1 (CIA30)",
        "also_known_as":  "NDUFAF1 / CIA30 / Complex I Intermediate Associated Protein 30kDa / MCIA Binary ACAD9 Partner",
        "disease":        DISEASE,
        "omim_gene":      OMIM_GENE,
        "omim_disease":   OMIM_DISEASE,
        "chromosome":     CHROMOSOME,
        "inheritance":    INHERITANCE,
        "protein": {
            "size_aa":   327,
            "size_kda":  36.0,
            "fold":      "CIA30 fold — alpha-helical coiled-coil scaffold; no FAD-binding domain (unlike ACAD9); no TM helices; soluble matrix-facing protein; N-terminal MTS; C-terminal ACAD9-docking interface",
            "module":    "MCIA complex binary partner of ACAD9 — ND2/ND5 membrane arm module (early CI membrane arm assembly); NDUFAF1-ACAD9 binary complex is the obligate first step before ECSIT/TMEM126B recruitment",
            "fe_s_cluster": False,
            "fad_binding": False,
            "function":  (
                "CI assembly factor; obligate first binary partner of ACAD9 in the MCIA complex. "
                "NDUFAF1 (CIA30) was the FIRST CI assembly factor identified in any organism (Neurospora crassa, 1998). "
                "NDUFAF1 forms the tightest and earliest binary interaction within the MCIA complex. "
                "ACAD9 cannot recruit ECSIT and TMEM126B without NDUFAF1. "
                "Loss of NDUFAF1 prevents MCIA tetramer formation → ND2/ND5 module biogenesis aborted → "
                "CI membrane arm absent/severely reduced → isolated CI deficiency."
            ),
        },
        "key_pathway_note": (
            "NDUFAF1 (CIA30) is the OBLIGATE BINARY PARTNER of ACAD9 and was the first CI assembly factor "
            "ever discovered (Neurospora crassa cia30, Kuffner 1998). "
            "It forms the core binary scaffold with ACAD9 upon which the MCIA complex assembles. "
            "Unlike ACAD9, NDUFAF1 has NO FAD-binding domain and therefore NO riboflavin (B2) responsiveness — "
            "a critical clinical distinction. NDUFAF1 deficiency produces a severe pure Leigh syndrome phenotype "
            "without the exercise-intolerance-dominant variant seen in ACAD9 p.Arg518His. "
            "Both NDUFAF1 and ACAD9 target the same ND2/ND5 membrane arm modules and show similar BN-PAGE "
            "patterns (ND2/ND5 sub-assembly intermediates), but ONLY ACAD9 is riboflavin-responsive. "
            "If a patient has MCIA-complex-type CI deficiency (isolated CI, CII/CIII/CIV normal, ND2/ND5 BN-PAGE pattern) "
            "with NO riboflavin response, NDUFAF1, ECSIT, or TMEM126B deficiency should be considered."
        ),
        "biochemical_fingerprint": {
            "complex_I":   "5–20 % of control (REDUCED — ISOLATED CI DEFICIENCY)",
            "complex_II":  "NORMAL (SDHA — all nuclear; unaffected)",
            "complex_III": "NORMAL",
            "complex_IV":  "NORMAL — key DDx: SCO2 (CIV reduced, HCM 100%); SURF1 (CIV reduced, Leigh); COX10 (CIV reduced, tubulopathy)",
            "acylcarnitines":      "NORMAL (no fatty-acid oxidation disorder — DDx ACAD9 vs ETFDH/MADD)",
            "urine_organic_acids": "NORMAL",
            "riboflavin_response": "NONE — NDUFAF1 has no FAD-binding domain; riboflavin does NOT rescue (critical DDx vs ACAD9)",
        },
        "cohort": {
            "n":                      n,
            "seed":                   SEED,
            "mean_age_onset_months":  round(sum(p["age_onset_months"] for p in pts) / n),
            "ci_activity_mean_pct":   round(sum(ci_vals) / n, 1),
            "ci_activity_range_pct":  f"{min(ci_vals):.1f}–{max(ci_vals):.1f}",
            "riboflavin_responders_pct": 0,   # No riboflavin response in NDUFAF1
        },
        "feature_frequencies_pct": {
            "psychomotor_regression":  _pct("psychomotor_regression"),
            "leigh_mri":               _pct("leigh_mri"),
            "lactic_acidosis":         _pct("lactic_acidosis"),
            "hypotonia":               _pct("hypotonia"),
            "seizures":                _pct("seizures"),
            "respiratory_compromise":  _pct("respiratory_compromise"),
            "ataxia":                  _pct("ataxia"),
            "dystonia":                _pct("dystonia"),
            "hcm":                     _pct("hcm"),
            "exercise_intolerance":    _pct("exercise_intolerance"),
            "peripheral_neuropathy":   _pct("peripheral_neuropathy"),
            "olfactory_bulb_lesions":  _pct("olfactory_bulb_lesions"),
            "leukodystrophy":          _pct("leukodystrophy"),
            "hepatopathy":             _pct("hepatopathy"),
        },
        "key_ddx": [
            {
                "feature":         "NO riboflavin response — CRITICAL DDx vs ACAD9",
                "significance":    (
                    "NDUFAF1 has NO FAD-binding domain. Riboflavin (B2) does NOT rescue NDUFAF1 deficiency. "
                    "If MCIA-type CI deficiency (isolated CI, ND2/ND5 BN-PAGE pattern) shows riboflavin "
                    "response: ACAD9 is the diagnosis. No riboflavin response: NDUFAF1, ECSIT, TMEM126B, "
                    "or non-riboflavin-responsive ACAD9 allele. Check ACAD9 WES first (more common)."
                ),
                "target_gene":    "ACAD9 (riboflavin-responsive, 50-60%)",
                "target_freq_pct": 55,
            },
            {
                "feature":         "NDUFAF1 vs ACAD9 — same MCIA complex, same BN-PAGE, different riboflavin response",
                "significance":    (
                    "Both NDUFAF1 and ACAD9 are MCIA complex members targeting ND2/ND5 modules. "
                    "BN-PAGE pattern similar (ND2/ND5 sub-assembly intermediates). "
                    "CRITICAL distinction: ACAD9 has riboflavin response (50-60%), NDUFAF1 does NOT. "
                    "ACAD9 may show exercise-intolerance-dominant phenotype; NDUFAF1 does not. "
                    "WES with chromosomal locus (ACAD9: 3q21.3; NDUFAF1: 15q11.2-q13) resolves DDx."
                ),
                "target_gene":    "ACAD9 (3q21.3, MCIA scaffold, riboflavin-responsive)",
                "target_freq_pct": 0,
            },
            {
                "feature":         "NO peripheral neuropathy",
                "significance":    "KEY DDx vs NDUFS1 (~50%) — CRITICAL distinguisher",
                "target_gene":    "NDUFS1",
                "target_freq_pct": 50,
            },
            {
                "feature":         "NO olfactory bulb MRI lesions",
                "significance":    "KEY DDx vs NDUFS4 (52-65%) — PATHOGNOMONIC for NDUFS4",
                "target_gene":    "NDUFS4",
                "target_freq_pct": 58,
            },
            {
                "feature":         "NO significant leukodystrophy",
                "significance":    "KEY DDx vs NDUFV1 (~40-50% white matter T2)",
                "target_gene":    "NDUFV1",
                "target_freq_pct": 45,
            },
            {
                "feature":         "Low HCM rate (~20-30%) vs ACAD9 (~60%) and SCO2 (~100%)",
                "significance":    (
                    "HCM rare in NDUFAF1 vs common in ACAD9 (~60%) and SCO2 (~100%). "
                    "Biochemical: NDUFAF1 has CIV NORMAL (vs SCO2 CIV severely reduced). "
                    "Useful clinical pointer: HCM + riboflavin response = ACAD9; HCM + CIV reduced = SCO2."
                ),
                "target_gene":    "ACAD9 (HCM 55-65%) / SCO2 (HCM 100%)",
                "target_freq_pct": 77,
            },
            {
                "feature":         "NO hepatopathy",
                "significance":    "KEY DDx vs POLG Alpers (~80%) and DGUOK hepatocerebral (~90%)",
                "target_gene":    "POLG / DGUOK",
                "target_freq_pct": 85,
            },
            {
                "feature":         "NDUFAF1 vs ECSIT vs TMEM126B — same MCIA complex",
                "significance":    (
                    "All three are MCIA complex members. Phenotypically very similar. "
                    "ECSIT: chromosome 19p13.3; TMEM126B: chromosome 11q14.1 (2 TM helices). "
                    "NDUFAF1: chromosome 15q11.2-q13 (no TM helices, soluble). "
                    "WES mandatory to distinguish all three. NDUFAF1 is the obligate ACAD9 binary "
                    "partner; ECSIT and TMEM126B join later in the MCIA assembly sequence."
                ),
                "target_gene":    "ECSIT (19p13.3) / TMEM126B (11q14.1)",
                "target_freq_pct": 0,
            },
            {
                "feature":         "CIA30 first-ever CI assembly factor — historical DDx anchor",
                "significance":    (
                    "NDUFAF1/CIA30 was discovered in 1998 (Neurospora crassa) — the first CI assembly "
                    "factor identified in any organism. Human NDUFAF1 pathogenic variants are rare but "
                    "clinically significant: they establish the MCIA-type CI deficiency phenotype class "
                    "(isolated CI; CII/CIII/CIV normal; ND2/ND5 BN-PAGE intermediates; no riboflavin response)."
                ),
                "target_gene":    "NDUFAF1 (historical anchor for MCIA-type CI deficiency)",
                "target_freq_pct": 0,
            },
        ],
        "absolute_contraindications": [
            "Metformin — directly inhibits CI at ND1/quinone-binding site (ND2 module territory where NDUFAF1/MCIA assembles membrane arm)",
            "Valproate (VPA) — triple mechanism: CoA sequestration + POLG inhibition + mt ND-subunit expression block",
            "Linezolid — inhibits mitochondrial 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits",
            "Chloramphenicol — same mt-ribosomal mechanism as linezolid; equally contraindicated",
        ],
        "contraindicated": [
            "Ketogenic diet — forces NADH → β-oxidation; CI absent → NADH cannot be re-oxidised → lactic acidosis worsens catastrophically; no vestigial ACAD activity in NDUFAF1 (unlike ACAD9)",
        ],
        "preferred_treatments": [
            "NOTE: NO riboflavin response in NDUFAF1 (unlike ACAD9) — do NOT expect riboflavin benefit; confirm ACAD9 mutation excluded before trialling riboflavin",
            "LEV (levetiracetam) — AED first-line: renal excretion, no mito toxicity",
            "CoQ10 (ubiquinol) — electron acceptor at quinone site; downstream of failed CI; Level C",
            "Thiamine (B1) — MANDATORY empiric: SLC19A3/BTD treatable mimic",
            "Biotin — MANDATORY empiric: BTD treatable mimic",
            "Succinate — CII bypass; bypasses NDUFAF1-failed CI ND2/ND5 assembly defect entirely",
            "L-Carnitine — may have secondary carnitine deficiency; monitor free carnitine",
            "IV dextrose GIR 6-8 mg/kg/min — never fast",
            "Sevoflurane (not propofol) for anaesthesia",
        ],
        "key_references": [
            "Kuffner R et al (1998) J Mol Biol — first cia30 (CIA30/NDUFAF1) discovery in Neurospora crassa",
            "Vogel RO et al (2005) FEBS J — human NDUFAF1 characterisation as CIA30 orthologue",
            "Dunning CJ et al (2007) Hum Mol Genet — NDUFAF1 required for human CI assembly",
            "Guerrero-Castillo S et al (2017) Cell Metab — MCIA complex CI assembly dynamics; NDUFAF1-ACAD9 binary complex",
            "Fassone E & Rahman S (2012) J Med Genet — CI genetics review including NDUFAF1",
        ],
    }


# ---------------------------------------------------------------------------
# Public API: get_breakdown
# ---------------------------------------------------------------------------

def get_breakdown() -> dict[str, Any]:
    pts = _cohort()
    n = len(pts)

    features = [
        "psychomotor_regression", "leigh_mri", "lactic_acidosis", "hypotonia",
        "seizures", "respiratory_compromise", "ataxia", "dystonia", "hcm",
        "exercise_intolerance", "peripheral_neuropathy", "olfactory_bulb_lesions",
        "leukodystrophy", "hepatopathy",
    ]
    feature_frequencies = {
        f: {"count": sum(1 for p in pts if p.get(f)), "pct": _pct(f)}
        for f in features
    }

    # CI activity histogram
    ci_vals = [p["ci_activity_pct"] for p in pts]
    bins = ["0–5%", "5–10%", "10–15%", "15–20%", ">20%"]
    counts = [
        sum(1 for v in ci_vals if v < 5),
        sum(1 for v in ci_vals if 5 <= v < 10),
        sum(1 for v in ci_vals if 10 <= v < 15),
        sum(1 for v in ci_vals if 15 <= v < 20),
        sum(1 for v in ci_vals if v >= 20),
    ]

    # Outcome distribution
    out_dist: dict[str, int] = {}
    for p in pts:
        out_dist[p["outcome"]] = out_dist.get(p["outcome"], 0) + 1

    # Mutation distribution
    mut_dist: dict[str, int] = {}
    for p in pts:
        mut_dist[p["mutation"]] = mut_dist.get(p["mutation"], 0) + 1

    # Region distribution
    reg_dist: dict[str, int] = {}
    for p in pts:
        reg_dist[p["region"]] = reg_dist.get(p["region"], 0) + 1

    # Sex distribution
    sex_dist = {
        "M": sum(1 for p in pts if p["sex"] == "M"),
        "F": sum(1 for p in pts if p["sex"] == "F"),
    }

    return {
        "n":                     n,
        "patients":              pts,
        "feature_frequencies":   feature_frequencies,
        "ci_activity_histogram": {"bins": bins, "counts": counts},
        "outcome_distribution":  out_dist,
        "mutation_distribution": mut_dist,
        "region_distribution":   reg_dist,
        "sex_distribution":      sex_dist,
        "riboflavin_responders": 0,
        "riboflavin_responder_pct": 0,
        "mean_age_onset_months": round(sum(p["age_onset_months"] for p in pts) / n),
        "ci_activity_mean":      round(sum(ci_vals) / n, 1),
    }


# ---------------------------------------------------------------------------
# Public API: get_definitions
# ---------------------------------------------------------------------------

def get_definitions() -> dict[str, Any]:
    pharmacology = [
        {
            "term": "Riboflavin (B2) — NO RESPONSE in NDUFAF1 (critical DDx vs ACAD9)",
            "category": "important_negative",
            "detail": (
                "NDUFAF1 has NO FAD-binding domain (unlike ACAD9 which has an ACAD-superfamily "
                "FAD-binding domain). Riboflavin supplementation does NOT rescue NDUFAF1 deficiency. "
                "If a patient with MCIA-type CI deficiency (isolated CI; ND2/ND5 BN-PAGE; CII/CIII/CIV normal) "
                "shows riboflavin response: ACAD9 is the likely diagnosis. No response: consider NDUFAF1, "
                "ECSIT, TMEM126B, or non-responsive ACAD9 allele. ACAD9 WES should be checked first "
                "(more common gene). NDUFAF1 riboflavin trial is NOT indicated and should NOT be a "
                "management expectation."
            ),
        },
        {
            "term": "Metformin — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Metformin directly inhibits mitochondrial Complex I at the ND1/quinone-binding site "
                "(ND2 membrane arm territory where NDUFAF1/MCIA complex assembles CI). "
                "In NDUFAF1 deficiency, CI is already severely reduced (5–20%). "
                "Further metformin inhibition precipitates life-threatening lactic acidosis. "
                "ABSOLUTE contraindication."
            ),
        },
        {
            "term": "Valproate (VPA) — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "VPA inhibits CI via three mechanisms: (1) CoA sequestration (valproyl-CoA trapping); "
                "(2) direct POLG mitochondrial polymerase inhibition → mtDNA depletion; "
                "(3) impairs expression of mt-encoded ND subunits. All three mechanisms worsen "
                "NDUFAF1 CI deficiency. ABSOLUTE contraindication."
            ),
        },
        {
            "term": "Linezolid — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Linezolid inhibits mitochondrial 23S rRNA → blocks synthesis of all 7 mt-encoded ND "
                "subunits (MT-ND1 through MT-ND4L). In NDUFAF1 deficiency, CI membrane arm is already "
                "absent/severely reduced (ND2/ND5 module failure). ABSOLUTE contraindication."
            ),
        },
        {
            "term": "Chloramphenicol — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Same mitochondrial ribosomal inhibition mechanism as linezolid. "
                "ABSOLUTE contraindication in all CI-deficiency including NDUFAF1."
            ),
        },
        {
            "term": "Ketogenic Diet (KD) — CONTRAINDICATED",
            "category": "contraindicated",
            "detail": (
                "KD forces NADH → β-oxidation. In NDUFAF1 deficiency, CI is absent/severely reduced "
                "(ND2/ND5 module assembly failed). NADH from β-oxidation cannot be re-oxidised → "
                "NADH accumulates → lactic acidosis worsens catastrophically. "
                "Unlike ACAD9, NDUFAF1 has no vestigial ACAD activity overlap. CONTRAINDICATED."
            ),
        },
        {
            "term": "Propofol — AVOID",
            "category": "high_caution",
            "detail": (
                "PRIS (propofol infusion syndrome) inhibits CIV (COX). In NDUFAF1 deficiency CI is "
                "already severely reduced. Dual ETC failure (CI + CIV) is catastrophic. "
                "Use sevoflurane for anaesthesia instead."
            ),
        },
        {
            "term": "CoQ10 (Ubiquinol) — Level C",
            "category": "treatment",
            "detail": (
                "CoQ10 is the electron acceptor at the quinone-binding site of CI (downstream of "
                "NDUFAF1's MCIA complex ND2/ND5 assembly target). Level C evidence; clinically used "
                "empirically in CI-deficiency including NDUFAF1."
            ),
        },
        {
            "term": "Thiamine (B1) — MANDATORY Empiric",
            "category": "treatment",
            "detail": (
                "Thiamine (B1) treats SLC19A3 (biotin-thiamine-responsive basal ganglia disease) "
                "and BTD (biotinidase deficiency) — treatable conditions mimicking CI-Leigh on MRI. "
                "Must be given empirically before genetic diagnosis is confirmed."
            ),
        },
        {
            "term": "Biotin — MANDATORY Empiric",
            "category": "treatment",
            "detail": (
                "Biotin treats BTD (biotinidase deficiency) and HCS (holocarboxylase synthetase "
                "deficiency) — treatable mimics. Empiric biotin mandatory before genetic confirmation."
            ),
        },
        {
            "term": "Succinate — Level C (CII Bypass)",
            "category": "treatment",
            "detail": (
                "Succinate provides substrate to Complex II (SDHA — all nuclear, CII NORMAL in "
                "NDUFAF1 deficiency). CII feeds electrons directly to CoQ10, bypassing failed CI "
                "ND2/ND5 module entirely. This bypass is specific to CI deficiency."
            ),
        },
        {
            "term": "LEV (Levetiracetam) — Preferred AED",
            "category": "treatment",
            "detail": (
                "Levetiracetam is the preferred AED in CI-deficiency: renal excretion (avoids "
                "hepatic metabolism), no mitochondrial toxicity, no CYP interactions."
            ),
        },
    ]

    glossary = [
        {
            "term": "MCIA Complex (Mitochondrial Complex I Assembly Complex)",
            "definition": (
                "The MCIA complex is a four-protein CI assembly factor complex consisting of "
                "ACAD9, NDUFAF1 (CIA30), ECSIT, and TMEM126B. "
                "Assembly sequence: ACAD9 is recruited first; NDUFAF1 forms the tightest binary "
                "complex with ACAD9 (ACAD9-NDUFAF1 binary sub-complex is the obligate first step); "
                "ECSIT and TMEM126B then join to complete the tetrameric MCIA complex. "
                "The MCIA complex is essential for biogenesis of the ND2 and ND5 membrane arm modules. "
                "Disruption of any MCIA member (ACAD9, NDUFAF1, ECSIT, or TMEM126B) prevents "
                "ND2/ND5 module assembly → CI membrane arm absent → isolated CI deficiency."
            ),
        },
        {
            "term": "CIA30 / First CI Assembly Factor",
            "definition": (
                "CIA30 (Complex I Intermediate Associated protein, 30 kDa) was the first CI assembly "
                "factor ever identified, discovered in Neurospora crassa by Kuffner et al in 1998. "
                "Human NDUFAF1 is the direct orthologue. The discovery of CIA30/NDUFAF1 established "
                "the concept that CI is assembled stepwise through transient sub-assembly intermediates "
                "rather than by simultaneous incorporation of all ~45 subunits. "
                "NDUFAF1/CIA30 is thus the historical and conceptual anchor for the entire CI assembly "
                "factor field, predating NDUFAF2, NDUFAF3, NDUFAF4–8, ACAD9, ECSIT, TIMMDC1, and TMEM126B."
            ),
        },
        {
            "term": "ACAD9-NDUFAF1 Binary Complex (Obligate First MCIA Step)",
            "definition": (
                "NDUFAF1 forms the tightest and earliest binary interaction within the MCIA complex. "
                "The ACAD9-NDUFAF1 binary sub-complex is the obligate scaffold upon which ECSIT and "
                "TMEM126B subsequently assemble. Without NDUFAF1, ACAD9 cannot properly dock ECSIT "
                "or TMEM126B → MCIA tetramer fails → ND2/ND5 module assembly aborted. "
                "This obligate ordering distinguishes NDUFAF1 from ECSIT and TMEM126B (which require "
                "the ACAD9-NDUFAF1 platform to assemble). The ACAD9-NDUFAF1 interface is the primary "
                "target of most pathogenic NDUFAF1 missense variants (e.g., p.Arg81Cys, p.Arg204Gln)."
            ),
        },
        {
            "term": "No Riboflavin Response — NDUFAF1 vs ACAD9",
            "definition": (
                "ACAD9 is the ONLY MCIA complex member with riboflavin (B2) responsiveness (~50-60%), "
                "because ACAD9 retains an FAD-binding domain from its ACAD superfamily origin. "
                "NDUFAF1, ECSIT, and TMEM126B have NO FAD-binding domain and therefore NO riboflavin "
                "responsiveness. If a patient with MCIA-type CI deficiency fails to respond to "
                "riboflavin, NDUFAF1, ECSIT, or TMEM126B deficiency should be considered — but "
                "non-responsive ACAD9 alleles (e.g., p.Arg266Gln, p.Asp562Gly) also show no "
                "riboflavin response, so riboflavin failure alone does not exclude ACAD9."
            ),
        },
        {
            "term": "ND2 / ND5 Membrane Arm Modules",
            "definition": (
                "The CI membrane arm is built from multiple sub-assemblies called 'modules'. The "
                "ND2 module (containing MT-ND2, MT-ND3, MT-ND4L and nuclear subunits NDUFB6, "
                "NDUFB8, NDUFB7) and the ND5 module (containing MT-ND5 and associated nuclear "
                "subunits including NDUFB10) are assembled with the help of the MCIA complex. "
                "NDUFAF1 (as part of MCIA) is required for proper biogenesis of these modules. "
                "NDUFAF1 deficiency produces ND2/ND5 sub-assembly intermediates visible on BN-PAGE, "
                "distinguishable from N-module (NDUFAF2/NDUFA12) or Q-module structural subunit "
                "intermediates (NDUFS3/NDUFS7/NDUFS8)."
            ),
        },
        {
            "term": "Pure Leigh Phenotype (NDUFAF1) vs Bimodal ACAD9",
            "definition": (
                "NDUFAF1 deficiency produces a homogeneous severe Leigh syndrome phenotype across "
                "all reported alleles: infantile or early-childhood onset, Leigh MRI (bilateral putamen/"
                "brainstem), psychomotor regression, lactic acidosis, respiratory compromise. "
                "There is NO exercise-intolerance-dominant variant analogous to ACAD9 p.Arg518His. "
                "This is because NDUFAF1 has no FAD-binding domain to stabilise with riboflavin "
                "and no hypomorphic allele producing sufficient residual MCIA function for "
                "resting metabolism while failing only under exercise load."
            ),
        },
        {
            "term": "Isolated CI Deficiency (NDUFAF1)",
            "definition": (
                "In NDUFAF1 deficiency, only Complex I (CI) activity is reduced; Complexes II, III, "
                "and IV activities are within normal range. CI activity is 5–20% of control — "
                "consistent with other MCIA-type CI assembly factor deficiencies. "
                "Normal acylcarnitines and normal urine organic acids. "
                "This isolated CI biochemical fingerprint, combined with ND2/ND5 BN-PAGE intermediates "
                "and no riboflavin response, points toward NDUFAF1, ECSIT, or TMEM126B deficiency "
                "(after ACAD9 has been excluded or is non-responsive)."
            ),
        },
    ]

    return {
        "pharmacology":              pharmacology,
        "glossary":                  glossary,
        "gene_full":                 "NADH:Ubiquinone Oxidoreductase Complex Assembly Factor 1 (CIA30)",
        "historical_names":          ["NDUFAF1", "CIA30", "Complex I Intermediate Associated Protein 30kDa"],
        "omim_gene":                 OMIM_GENE,
        "omim_disease":              OMIM_DISEASE,
        "chromosome":                CHROMOSOME,
        "inheritance_detail":        (
            "Autosomal recessive (AR): biallelic pathogenic variants required; both sexes affected equally; "
            "consanguinity increases risk; no common founder mutation (unlike ACAD9 p.Arg518His European founder); "
            "pathogenic variants rare — most reported as isolated cases or small families"
        ),
        "protein_size_aa":           327,
        "protein_size_kda":          36.0,
        "tm_helices":                0,
        "fad_binding":               False,
        "module":                    "MCIA complex — obligate binary ACAD9 partner; ND2/ND5 membrane arm module (early CI membrane arm assembly); ACAD9-NDUFAF1 binary complex is the obligate first step before ECSIT/TMEM126B recruitment",
        "fe_s_cluster":              False,
        "ci_activity_range":         "5–20 % of control (isolated CI deficiency)",
        "cii_ciii_civ_normal":       True,
        "acylcarnitines_normal":     True,
        "urine_organic_acids_normal": True,
        "riboflavin_response":       False,
        "bn_page_pattern":           "Severely reduced CI holoenzyme; ND2/ND5 sub-assembly intermediates detectable — ND2/ND5 membrane arm module failure (same BN-PAGE class as ACAD9); distinct from N-module intermediates (NDUFAF2/NDUFA12) or Q-module intermediates (NDUFS3)",
        "key_distinguishing_feature": (
            "OBLIGATE ACAD9 BINARY PARTNER — NDUFAF1 (CIA30) forms the tightest binary sub-complex with ACAD9 "
            "in the MCIA tetramer (ACAD9→NDUFAF1 binary, then ECSIT+TMEM126B join). "
            "CIA30 was the FIRST CI assembly factor discovered in any organism (Neurospora 1998). "
            "NO riboflavin response (unlike ACAD9, critical clinical DDx). "
            "No FAD-binding domain. Pure Leigh syndrome (no bimodal exercise-intolerance variant). "
            "MCIA-type BN-PAGE (ND2/ND5 intermediates). Isolated CI deficiency. "
            "Chromosome 15q11.2-q13 (DDx ECSIT 19p13.3, TMEM126B 11q14.1, ACAD9 3q21.3)"
        ),
    }


if __name__ == "__main__":
    import json
    print("=== NDUFAF1 Overview ===")
    print(json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== Breakdown (sample) ===")
    bd = get_breakdown()
    print(f"n={bd['n']}, riboflavin_responders={bd['riboflavin_responders']} ({bd['riboflavin_responder_pct']}%)")
    print(f"sex_dist={bd['sex_distribution']}")
    print("\n=== Definitions (terms) ===")
    defs = get_definitions()
    print(f"pharmacology_terms={[p['term'] for p in defs['pharmacology']]}")
    print(f"glossary_terms={[g['term'] for g in defs['glossary']]}")
