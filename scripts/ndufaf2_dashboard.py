#!/usr/bin/env python3
"""NDUFAF2 — Leigh Syndrome Isolated Complex I Deficiency (CI Assembly Factor 2 / NDUFA12L).

NDUFAF2 (NADH:Ubiquinone Oxidoreductase Complex Assembly Factor 2, formerly NDUFA12L)
is a ~137-aa nuclear-encoded CI assembly factor (~12.5 kDa) that is a structural
PARALOG of the mature CI subunit NDUFA12 (B17.2). NDUFAF2 transiently occupies the
NDUFA12 binding site at the N-Q module interface during CI assembly and is subsequently
displaced when NDUFA12 is incorporated into the mature holoenzyme — the "assembly-factor
swap" or "chaperone-subunit exchange" mechanism unique to CI maturation.

  NDUFAF2 gene    OMIM *609653
  Disease         Leigh Syndrome (OMIM #256000) / MC1DN25 (OMIM #619777)
  Inheritance     AR (autosomal recessive biallelic)
  Chromosome      5q12.1

PATHOPHYSIOLOGY (Complex I / N-Q Module / NDUFAF2 / NDUFA12L / Assembly-Factor Swap):
  NDUFAF2 is the assembly chaperone that occupies the NDUFA12 (B17.2) binding site
  on the N-Q module sub-assembly intermediate during CI biogenesis. During CI assembly:
    1. NDUFAF2 is incorporated early, stabilising the N-Q module sub-assembly (N-module
       plus partial Q-module, still lacking full membrane-arm integration).
    2. When mature CI assembly reaches the point of N-Q module finalisation, NDUFAF2 is
       actively displaced from its binding site and NDUFA12 (B17.2) is incorporated
       in its place.
    3. This assembly-factor swap is the only documented CI assembly mechanism in which
       a mature structural subunit directly REPLACES a specific assembly factor from the
       same structural binding site.

  In NDUFAF2 deficiency, CI assembly stalls at the N-Q module sub-assembly stage
  (similar to NDUFA12 deficiency, because NDUFAF2 and NDUFA12 bind the same site):
    • CI holocomplex severely reduced or absent on BN-PAGE
    • N-Q module sub-assembly intermediates may be detectable on BN-PAGE
      (stalled assembly intermediate — NDUFAF2 deficiency stalls BEFORE the
      NDUFAF2→NDUFA12 swap can occur)
    • Isolated CI deficiency 5–20%; CII/CIII/CIV NORMAL

  UNIQUE MOLECULAR SIGNATURE — NDUFAF2 AS CI-ASSEMBLY-FACTOR-SWAP CHAPERONE:
    NDUFAF2 is the ONLY documented CI assembly factor that is a direct structural
    paralog of a mature CI subunit (NDUFA12) and occupies the same binding site.
    All other CI assembly factors (NDUFAF1, NDUFAF3-8, ACAD9, ECSIT, TIMMDC1, etc.)
    occupy distinct, non-subunit binding sites. NDUFAF2 uniquely mimics a mature CI
    subunit in structure (paralog), temporarily stands in during N-Q module assembly,
    and is then displaced by the true mature subunit NDUFA12. This makes NDUFAF2
    deficiency phenotypically and biochemically nearly indistinguishable from NDUFA12
    deficiency — the critical distinction requires molecular genetics (WES).

DISTINGUISHING FEATURES vs OTHER CI-LEIGH GENES:
  vs NDUFA12 (B17.2, N-Q interface, 12q22 — the mature subunit paralog of NDUFAF2):
    • NDUFAF2 (5q12.1) is the ASSEMBLY FACTOR; NDUFA12 (12q22) is the mature subunit.
    • Both produce near-identical phenotype (Leigh MRI, isolated CI 5–20%, N-Q sub-assembly
      intermediates on BN-PAGE, AR). Molecular distinction by WES only (5q12.1 vs 12q22).
    • NDUFAF2 mutations → swap CANNOT be initiated (assembly stalls early, before swap).
    • NDUFA12 mutations → swap cannot be COMPLETED (NDUFAF2 displaced but NDUFA12 cannot
      properly occupy the site). Same stalled phenotype, different molecular origin.
  vs NDUFS1 (N-module IP1/75kDa):
    • NO peripheral neuropathy in NDUFAF2 (NDUFS1: ~50% — CRITICAL DDx)
  vs NDUFS4 (N-module accessory):
    • NO olfactory bulb MRI lesions in NDUFAF2 (NDUFS4: 52–65% — PATHOGNOMONIC for NDUFS4)
  vs NDUFV1 (N-module FMN/N3):
    • NO leukodystrophy / white matter T2 signal in NDUFAF2 (NDUFV1: ~40–50%)
  vs NDUFV2 (N-module N1b-2Fe2S) / SCO2 (CIV):
    • NO hypertrophic cardiomyopathy (NDUFV2: ~80%; SCO2: ~100%)
  vs POLG/DGUOK (mtDNA depletion):
    • NO hepatopathy in NDUFAF2 (POLG Alpers: ~80%; DGUOK: ~90%)

FOUNDER / RECURRENT MUTATIONS:
  c.IVS1+3A>G   — splice donor intron 1; partial CI residual (~10–20%); Leigh syndrome;
                   first reported Ogilvie 2005 NatGenet
  p.Arg45Gln  c.134G>A  — N-Q binding surface disruption; affects NDUFA12 displacement
                           step; severe infantile Leigh
  p.Leu67Pro  c.200T>C  — helix-breaking proline; alpha-helix collapse; severe
  p.Ala94Val  c.281C>T  — hydrophobic core packing; assembly intermediate accumulates;
                           intermediate severity
  p.Glu5Ter   c.13G>T   — near start; early stop; null; consanguineous; severe neonatal

THERAPY — NDUFAF2 / CI-LEIGH SPECIFICS:
  No targeted NDUFAF2 rescue therapy is clinically available.
  Management follows the CI-Leigh supportive protocol:
    ABSOLUTE contraindications (direct CI inhibitors / mito toxins):
      Metformin — directly inhibits CI at ND1/quinone-binding site (N-Q module territory)
      Valproate  — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression block
      Linezolid  — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
      Chloramphenicol — same ribosomal mechanism as linezolid
    CONTRAINDICATED:
      Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH
    AVOID / HIGH CAUTION:
      Propofol — PRIS + secondary CIV inhibition adds a second ETC bottleneck
      Phenobarbital — secondary CI inhibitor; use LEV first
    LEVEL C cofactors (standard CI supportive):
      Riboflavin (B2) — CI-specific; FMN prosthetic group at NDUFV1 (upstream N-module)
      CoQ10 (ubiquinol) — electron acceptor at quinone site
      Thiamine (B1) — MANDATORY empiric: mimics SLC19A3/BTD (treatable)
      Biotin — MANDATORY empiric: mimics BTD (treatable)
      Succinate — CII bypass; bypasses NDUFAF2-failed CI entirely
      L-Carnitine — energy metabolism support
    Preferred AED: Levetiracetam (LEV) — renal excretion; no mito toxicity
    Glucose: IV dextrose GIR 6–8 mg/kg/min — NEVER fast
    Anaesthesia: sevoflurane (NOT propofol)
"""

from __future__ import annotations
import random
from typing import Any

SEED = 673
GENE = "NDUFAF2"
DISEASE = "NDUFAF2 Leigh Syndrome — Isolated Complex I Deficiency (CI Assembly Factor 2 / NDUFA12L / Assembly-Factor-Swap Chaperone)"
OMIM_GENE = "609653"
OMIM_DISEASE = "256000"
OMIM_DISEASE_SPECIFIC = "619777"
CHROMOSOME = "5q12.1"
INHERITANCE = "AR"

# ---------------------------------------------------------------------------
# Synthetic 40-patient cohort  (seed-673, reflects published literature)
# ---------------------------------------------------------------------------

def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    sexes = ["M", "F"]
    mutations = [
        "c.IVS1+3A>G / c.IVS1+3A>G (hom, consanguineous)",
        "c.IVS1+3A>G / p.Arg45Gln (compound het)",
        "p.Arg45Gln / p.Leu67Pro (compound het)",
        "c.IVS1+3A>G / p.Ala94Val (compound het)",
        "p.Glu5Ter / c.IVS1+3A>G (compound het)",
        "p.Leu67Pro / p.Ala94Val (compound het)",
        "p.Arg45Gln / p.Arg45Gln (hom, consanguineous)",
        "Novel biallelic LOF (frameshift/splice)",
        "c.IVS1+3A>G / novel missense (compound het)",
        "p.Ala94Val / p.Ala94Val (hom, consanguineous)",
    ]
    regions = [
        "MENA", "South Asian", "European", "East Asian",
        "North African", "Latin American", "Sub-Saharan African",
    ]
    ci_act_ranges = [(5, 10), (8, 15), (10, 20)]

    cohort = []
    for i in range(1, n + 1):
        age_onset = round(rng.uniform(0.2, 18), 1)
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
            "leigh_mri":              rng.random() < 0.84,
            "lactic_acidosis":        rng.random() < 0.87,
            "hypotonia":              rng.random() < 0.85,
            "psychomotor_regression": rng.random() < 0.95,
            "respiratory_compromise": rng.random() < 0.43,
            "seizures":               rng.random() < 0.48,
            "ataxia":                 rng.random() < 0.40,
            "dystonia":               rng.random() < 0.35,
            "hcm":                    rng.random() < 0.05,
            "peripheral_neuropathy":  rng.random() < 0.04,
            "olfactory_bulb_lesions": rng.random() < 0.03,
            "leukodystrophy":         rng.random() < 0.04,
            "hepatopathy":            rng.random() < 0.03,
            "outcome": rng.choice(
                ["deceased before 3yr"] * 12 +
                ["deceased before 10yr"] * 9 +
                ["alive, severe disability"] * 12 +
                ["alive, moderate disability"] * 7
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
        "gene_full_name": "NADH:Ubiquinone Oxidoreductase Complex Assembly Factor 2",
        "also_known_as":  "NDUFA12L (NDUFA12-like) / CI Assembly Factor 2 / Assembly-Factor-Swap Chaperone",
        "disease":        DISEASE,
        "omim_gene":      OMIM_GENE,
        "omim_disease":   OMIM_DISEASE,
        "omim_disease_specific": OMIM_DISEASE_SPECIFIC,
        "chromosome":     CHROMOSOME,
        "inheritance":    INHERITANCE,
        "protein": {
            "size_aa_precursor": 137,
            "size_aa_mature":    107,
            "size_kda":          12.5,
            "fold":              "NDUFA12-paralog fold — globular alpha/beta domain structurally homologous to NDUFA12 (B17.2); no canonical TM helix; peripheral N-Q module binding via protein-protein contacts at the N-Q interface",
            "module":            "N-Q module assembly intermediate — transiently occupies the NDUFA12 (B17.2) binding site at the N-Q interface during CI biogenesis; displaced and replaced by NDUFA12 upon mature CI holoenzyme formation",
            "fe_s_cluster":      False,
            "zinc_finger":       False,
            "function":          "CI assembly chaperone; transiently occupies the NDUFA12 (B17.2) binding site on the N-Q module sub-assembly intermediate; stabilises N-Q module during CI biogenesis; is the only CI assembly factor that is a direct structural paralog of the mature CI subunit it is displaced by (NDUFA12); loss → assembly stalls before the NDUFAF2-to-NDUFA12 swap can occur → N-Q module intermediate accumulates → CI holocomplex absent/severely reduced on BN-PAGE.",
        },
        "key_pathway_note": (
            "NDUFAF2 (NDUFA12L) is the ONLY documented CI assembly factor that is a direct "
            "structural PARALOG of a mature CI subunit. During CI biogenesis, NDUFAF2 transiently "
            "occupies the NDUFA12 (B17.2) binding site at the N-Q module interface — stabilising "
            "the nascent N-Q sub-assembly — and is subsequently displaced when NDUFA12 is "
            "incorporated. This 'assembly-factor swap' or 'chaperone-subunit exchange' is unique "
            "in the CI assembly pathway. In NDUFAF2 deficiency, the swap CANNOT be INITIATED "
            "(NDUFAF2 is missing from the start → N-Q module sub-assembly destabilised → assembly "
            "stalls → CI absent). In NDUFA12 deficiency (12q22), the swap cannot be COMPLETED "
            "(NDUFAF2 present but NDUFA12 cannot properly occupy the site → same stalled outcome). "
            "Both produce near-identical phenotype. Only WES (5q12.1 vs 12q22) distinguishes them. "
            "Biochemical fingerprint: isolated CI deficiency 5–20%; CII/CIII/CIV NORMAL."
        ),
        "biochemical_fingerprint": {
            "complex_I":   "5–20 % of control (SEVERELY REDUCED — ISOLATED CI DEFICIENCY)",
            "complex_II":  "NORMAL (SDHA — all nuclear; unaffected)",
            "complex_III": "NORMAL",
            "complex_IV":  "NORMAL — key DDx: SURF1 / SCO2 / COX10 / COX15",
        },
        "cohort": {
            "n":                       n,
            "seed":                    SEED,
            "mean_age_onset_months":   round(sum(p["age_onset_months"] for p in pts) / n),
            "ci_activity_mean_pct":    round(sum(ci_vals) / n, 1),
            "ci_activity_range_pct":   f"{min(ci_vals):.1f}–{max(ci_vals):.1f}",
        },
        "feature_frequencies_pct": {
            "psychomotor_regression":   _pct("psychomotor_regression"),
            "lactic_acidosis":          _pct("lactic_acidosis"),
            "hypotonia":                _pct("hypotonia"),
            "leigh_mri":                _pct("leigh_mri"),
            "seizures":                 _pct("seizures"),
            "respiratory_compromise":   _pct("respiratory_compromise"),
            "ataxia":                   _pct("ataxia"),
            "dystonia":                 _pct("dystonia"),
            "hcm":                      _pct("hcm"),
            "peripheral_neuropathy":    _pct("peripheral_neuropathy"),
            "olfactory_bulb_lesions":   _pct("olfactory_bulb_lesions"),
            "leukodystrophy":           _pct("leukodystrophy"),
            "hepatopathy":              _pct("hepatopathy"),
        },
        "key_ddx": [
            {
                "feature":        "NDUFAF2 vs NDUFA12 — near-identical phenotype — CRITICAL DDx REQUIRES WES",
                "significance":   "NDUFAF2 (5q12.1, assembly factor) and NDUFA12 (12q22, mature subunit) produce nearly identical Leigh syndrome + isolated CI deficiency + N-Q sub-assembly intermediates on BN-PAGE. Clinical, biochemical, and neuroimaging features CANNOT distinguish them. WES with locus discrimination (5q12.1 vs 12q22) is MANDATORY. This is the most critical and closest DDx pair in all CI-Leigh — two structurally related proteins from different chromosomes that mimic each other completely at the phenotypic level.",
                "target_gene":    "NDUFA12 (B17.2, N-Q interface, 12q22 — mature subunit paralog)",
                "target_freq_pct": 0,
            },
            {
                "feature":        "NO peripheral neuropathy",
                "significance":   "KEY DDx vs NDUFS1 (~50%) — CRITICAL distinguisher",
                "target_gene":    "NDUFS1",
                "target_freq_pct": 50,
            },
            {
                "feature":        "NO olfactory bulb MRI lesions",
                "significance":   "KEY DDx vs NDUFS4 (52–65%) — PATHOGNOMONIC for NDUFS4",
                "target_gene":    "NDUFS4",
                "target_freq_pct": 58,
            },
            {
                "feature":        "NO leukodystrophy / white matter T2 signal",
                "significance":   "KEY DDx vs NDUFV1 (~40–50%) — CRITICAL",
                "target_gene":    "NDUFV1",
                "target_freq_pct": 45,
            },
            {
                "feature":        "NO hypertrophic cardiomyopathy (HCM)",
                "significance":   "KEY DDx vs NDUFV2 (~80%) and SCO2 (~100%)",
                "target_gene":    "NDUFV2 / SCO2",
                "target_freq_pct": 80,
            },
            {
                "feature":        "NO hepatopathy",
                "significance":   "KEY DDx vs POLG Alpers (~80%) and DGUOK hepatocerebral (~90%)",
                "target_gene":    "POLG / DGUOK",
                "target_freq_pct": 85,
            },
            {
                "feature":        "CIV (COX) NORMAL",
                "significance":   "KEY DDx vs SURF1 / SCO2 / COX10 / COX15 (isolated CIV deficiency)",
                "target_gene":    "SURF1 / SCO2 / COX10 / COX15",
                "target_freq_pct": 100,
            },
        ],
        "absolute_contraindications": [
            "Metformin — directly inhibits CI at ND1/quinone-binding site (N-Q module territory where NDUFAF2 chaperones assembly)",
            "Valproate (VPA) — triple mechanism: CoA sequestration + POLG inhibition + ND-subunit expression block (all 7 mt-encoded ND subunits)",
            "Linezolid — inhibits mitochondrial 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits (MT-ND1 through MT-ND6 + MT-ND4L)",
            "Chloramphenicol — same mt-ribosomal mechanism as linezolid; equally contraindicated",
        ],
        "contraindicated": [
            "Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH (NDUFAF2 assembly-factor defect → CI absent; NADH accumulates → worsens lactic acidosis)",
        ],
        "preferred_treatments": [
            "LEV (levetiracetam) — AED first-line: renal excretion, no mito toxicity",
            "Riboflavin (B2) — CI-specific cofactor; FMN at NDUFV1 (N-module, upstream of N-Q interface)",
            "CoQ10 (ubiquinol) — electron acceptor at quinone site; downstream of failed CI",
            "Thiamine (B1) — MANDATORY empiric: SLC19A3/BTD treatable mimic",
            "Biotin — MANDATORY empiric: BTD treatable mimic",
            "Succinate — CII bypass; bypasses NDUFAF2-failed CI N-Q assembly defect entirely",
            "L-Carnitine — energy metabolism support",
            "IV dextrose GIR 6–8 mg/kg/min — never fast",
            "Sevoflurane (not propofol) for anaesthesia",
        ],
        "key_references": [
            "Ogilvie JM et al (2005) NatGenet — first NDUFAF2 mutations in Leigh syndrome",
            "Berger I et al (2008) — additional NDUFAF2 patients, clinical series",
            "Stroud DA et al (2016) Nature — CI assembly pathway NDUFAF2→NDUFA12 swap",
            "Guerrero-Castillo S et al (2017) CellMetab — CI assembly intermediates NDUFAF2",
            "Fassone E & Rahman S (2012) JMedGenet — CI genetics review",
        ],
    }


# ---------------------------------------------------------------------------
# Public API: get_breakdown
# ---------------------------------------------------------------------------

def get_breakdown() -> dict[str, Any]:
    pts = _cohort()
    n = len(pts)

    features = [
        "leigh_mri", "lactic_acidosis", "hypotonia", "psychomotor_regression",
        "respiratory_compromise", "seizures", "ataxia", "dystonia", "hcm",
        "peripheral_neuropathy", "olfactory_bulb_lesions", "leukodystrophy", "hepatopathy",
    ]
    feature_frequencies = {
        f: {"count": sum(1 for p in pts if p.get(f)), "pct": _pct(f)}
        for f in features
    }

    outcome_counts: dict[str, int] = {}
    for p in pts:
        outcome_counts[p["outcome"]] = outcome_counts.get(p["outcome"], 0) + 1

    mutation_counts: dict[str, int] = {}
    for p in pts:
        mutation_counts[p["mutation"]] = mutation_counts.get(p["mutation"], 0) + 1

    region_counts: dict[str, int] = {}
    for p in pts:
        region_counts[p["region"]] = region_counts.get(p["region"], 0) + 1

    return {
        "n": n,
        "patients": pts,
        "feature_frequencies": feature_frequencies,
        "outcome_distribution": outcome_counts,
        "mutation_distribution": mutation_counts,
        "region_distribution": region_counts,
        "sex_distribution": {
            "M": sum(1 for p in pts if p["sex"] == "M"),
            "F": sum(1 for p in pts if p["sex"] == "F"),
        },
        "ci_activity_histogram": {
            "bins": ["5–8%", "8–12%", "12–16%", "16–20%"],
            "counts": [
                sum(1 for p in pts if 5 <= p["ci_activity_pct"] < 8),
                sum(1 for p in pts if 8 <= p["ci_activity_pct"] < 12),
                sum(1 for p in pts if 12 <= p["ci_activity_pct"] < 16),
                sum(1 for p in pts if 16 <= p["ci_activity_pct"] <= 20),
            ],
        },
    }


# ---------------------------------------------------------------------------
# Public API: get_definitions
# ---------------------------------------------------------------------------

def get_definitions() -> dict[str, Any]:
    pharmacology = [
        {
            "term": "Metformin — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Metformin directly inhibits CI at the ND1/quinone-binding interface — "
                "within the N-Q module territory where NDUFAF2 chaperones assembly of the "
                "N-Q sub-complex. In NDUFAF2 deficiency CI activity is already 5–20%. "
                "Metformin precipitates fatal lactic crisis. NEVER use in any CI-Leigh."
            ),
        },
        {
            "term": "Valproate (VPA) — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "VPA: (1) sequesters CoA (mitochondrial beta-oxidation arrest); "
                "(2) inhibits POLG (mtDNA depletion affecting all 7 mt-encoded ND subunit genes); "
                "(3) suppresses expression of mt-encoded ND subunits. Triple mechanism "
                "makes VPA uniquely dangerous in NDUFAF2 deficiency."
            ),
        },
        {
            "term": "Linezolid — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Linezolid inhibits the mitochondrial large subunit (23S) rRNA → blocks "
                "synthesis of ALL 7 mt-encoded CI subunits (MT-ND1 through MT-ND6 + MT-ND4L). "
                "In NDUFAF2 deficiency, CI assembly is already stalled. Eliminating mt-ND "
                "subunit synthesis completely abolishes any residual CI. ABSOLUTE contraindication."
            ),
        },
        {
            "term": "Chloramphenicol — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Same mitochondrial ribosomal inhibition mechanism as linezolid. "
                "Blocks synthesis of all mt-encoded ND subunits. "
                "ABSOLUTE contraindication in all CI-Leigh."
            ),
        },
        {
            "term": "Ketogenic Diet (KD) — CONTRAINDICATED",
            "category": "contraindicated",
            "detail": (
                "KD forces NADH → β-oxidation pathway. In NDUFAF2 deficiency, CI is absent "
                "(N-Q module assembly stalled, no functional CI holoenzyme). NADH generated "
                "by β-oxidation cannot be re-oxidised by absent CI → NADH accumulates → "
                "lactic acidosis worsens catastrophically."
            ),
        },
        {
            "term": "Propofol — AVOID",
            "category": "high_caution",
            "detail": (
                "PRIS (propofol infusion syndrome) inhibits CIV (COX). In NDUFAF2 deficiency "
                "CI is already absent. Dual ETC failure (CI + CIV) is catastrophic. "
                "Use sevoflurane for anaesthesia instead."
            ),
        },
        {
            "term": "Phenobarbital — HIGH CAUTION",
            "category": "high_caution",
            "detail": (
                "Phenobarbital at high doses is a secondary CI inhibitor. In NDUFAF2 "
                "deficiency where CI is already severely deficient (5–20%), further "
                "inhibition may precipitate metabolic crisis. Use LEV preferentially."
            ),
        },
        {
            "term": "Riboflavin (B2) — Level C",
            "category": "treatment",
            "detail": (
                "Riboflavin is the precursor to FMN (flavin mononucleotide), the prosthetic "
                "group of NDUFV1 in the N-module — the entry point of electron transfer in CI. "
                "Level C evidence; clinically used empirically in CI-Leigh."
            ),
        },
        {
            "term": "CoQ10 (Ubiquinol) — Level C",
            "category": "treatment",
            "detail": (
                "CoQ10 is the electron acceptor at the quinone-binding site of CI (downstream "
                "of NDUFAF2's N-Q module assembly target). Level C evidence."
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
                "deficiency) — treatable mimics of CI-Leigh syndrome. Empiric biotin is mandatory "
                "before genetic diagnosis is confirmed."
            ),
        },
        {
            "term": "Succinate — Level C (CII Bypass)",
            "category": "treatment",
            "detail": (
                "Succinate provides substrate to Complex II (SDHA — all nuclear, CII NORMAL in "
                "NDUFAF2 deficiency). CII feeds electrons directly to CoQ10, bypassing failed CI "
                "entirely. This bypass is specific to CI deficiency."
            ),
        },
        {
            "term": "LEV (Levetiracetam) — Preferred AED",
            "category": "treatment",
            "detail": (
                "Levetiracetam is the preferred AED in CI-Leigh: renal excretion (avoids hepatic "
                "metabolism), no mitochondrial toxicity, no CYP interactions. First-line for "
                "seizures in NDUFAF2 Leigh syndrome."
            ),
        },
    ]

    glossary = [
        {
            "term": "Assembly-Factor Swap (NDUFAF2 → NDUFA12 Exchange)",
            "definition": (
                "The assembly-factor swap is the CI biogenesis step in which NDUFAF2 (NDUFA12L), "
                "the assembly chaperone occupying the NDUFA12 (B17.2) binding site on the N-Q "
                "module sub-assembly, is actively displaced and replaced by the mature structural "
                "subunit NDUFA12. This is the only documented example in CI assembly where a mature "
                "subunit directly replaces its own structural paralog from the same binding site. "
                "NDUFAF2 deficiency prevents the swap from being initiated; NDUFA12 deficiency "
                "prevents the swap from being completed. Both stall CI assembly at the N-Q intermediate."
            ),
        },
        {
            "term": "N-Q Module Sub-Assembly Intermediate",
            "definition": (
                "The N-Q module sub-assembly intermediate is a stalled CI assembly product detected "
                "by BN-PAGE in NDUFAF2 and NDUFA12 deficiencies. It comprises the N-module (FMN, "
                "NDUFV1/V2/V3, NDUFS1/3/5/8, NDUFA7/8) plus partial Q-module components assembled "
                "but not yet fully integrated into the CI membrane arm. This sub-assembly is "
                "characteristic of N-Q interface failure — a finding that biochemically narrows "
                "the DDx to NDUFAF2 and NDUFA12 among all CI-Leigh genes."
            ),
        },
        {
            "term": "NDUFA12L (NDUFA12-Like) / Structural Paralog",
            "definition": (
                "NDUFAF2 is historically called NDUFA12L (NDUFA12-like) because its 3D structure "
                "is highly homologous to NDUFA12 (B17.2). Both proteins adopt a similar "
                "alpha/beta globular fold and bind the same site on the N-Q module. NDUFAF2 is "
                "the assembly factor (transient, displaced); NDUFA12 is the structural subunit "
                "(permanent, incorporated into mature CI). This structural paralog relationship "
                "is unique among CI assembly factors — all other assembly factors (NDUFAF1, "
                "NDUFAF3-8, ACAD9, ECSIT, TIMMDC1, TMEM126B) occupy distinct sites not "
                "corresponding to any mature CI structural subunit position."
            ),
        },
        {
            "term": "BN-PAGE (Blue Native PAGE)",
            "definition": (
                "Blue native polyacrylamide gel electrophoresis separates intact mitochondrial "
                "respiratory chain complexes. In NDUFAF2 deficiency: absent or severely reduced CI "
                "holoenzyme band; N-Q module sub-assembly intermediates may be detectable (stalled "
                "assembly products at the N-Q module stage). Pattern similar to NDUFA12 deficiency "
                "— near-identical on BN-PAGE; distinction requires WES."
            ),
        },
        {
            "term": "Isolated CI Deficiency",
            "definition": (
                "Only Complex I (CI) activity is reduced; Complexes II, III, and IV activities are "
                "within normal range. This is the biochemical fingerprint of NDUFAF2 deficiency. "
                "CIV NORMAL distinguishes from SURF1/SCO2/COX10/COX15 (isolated CIV). "
                "Multi-complex involvement would suggest mtDNA depletion (POLG, DGUOK, MPV17) "
                "or a mtDNA mutation (MELAS, NARP, KSS)."
            ),
        },
        {
            "term": "Leigh Syndrome (OMIM #256000)",
            "definition": (
                "Leigh syndrome (subacute necrotising encephalopathy, OMIM #256000) is the most "
                "common mitochondrial disease presentation in childhood. Hallmark: bilateral "
                "symmetric T2-hyperintense lesions in basal ganglia (putamen, globus pallidus, "
                "caudate) and/or brainstem on MRI, with elevated lactate. NDUFAF2 deficiency "
                "causes isolated CI Leigh syndrome; also categorised as MC1DN25 (OMIM #619777)."
            ),
        },
    ]

    return {
        "pharmacology":                 pharmacology,
        "glossary":                     glossary,
        "gene_full":                    "NADH:Ubiquinone Oxidoreductase Complex Assembly Factor 2",
        "historical_names":             ["NDUFA12L", "CI Assembly Factor 2", "CIA Assembly Factor 2"],
        "omim_gene":                    OMIM_GENE,
        "omim_disease":                 OMIM_DISEASE,
        "omim_disease_specific":        OMIM_DISEASE_SPECIFIC,
        "chromosome":                   CHROMOSOME,
        "inheritance_detail":           "Autosomal recessive (AR): biallelic pathogenic variants required; both sexes affected equally; consanguinity increases risk; carrier parents asymptomatic",
        "protein_size_precursor_aa":    137,
        "protein_size_mature_aa":       107,
        "protein_size_kda":             12.5,
        "tm_helices":                   0,
        "module":                       "N-Q module assembly intermediate — NDUFA12 binding site (transient, displaced upon mature CI formation)",
        "fe_s_cluster":                 False,
        "ci_activity_range":            "5–20 % of control",
        "cii_ciii_civ_normal":          True,
        "bn_page_pattern":              "Absent CI holoenzyme; N-Q module sub-assembly intermediate may be detectable — stalled before NDUFAF2→NDUFA12 swap; near-identical to NDUFA12 deficiency BN-PAGE",
        "key_distinguishing_feature":   "ONLY CI assembly factor that is a direct structural PARALOG of the mature CI subunit it is displaced by (NDUFA12 / B17.2); occupies same N-Q interface binding site as NDUFA12; stalled assembly prevents NDUFAF2-to-NDUFA12 swap; near-identical phenotype to NDUFA12 deficiency — WES mandatory to distinguish 5q12.1 (NDUFAF2) from 12q22 (NDUFA12)",
    }


if __name__ == "__main__":
    import json
    print("=== NDUFAF2 Overview ===")
    print(json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== Breakdown (sample) ===")
    bd = get_breakdown()
    print(f"n={bd['n']}, features={list(bd['feature_frequencies'].keys())}")
    print(f"sex_dist={bd['sex_distribution']}")
    print("\n=== Definitions (terms) ===")
    defs = get_definitions()
    print(f"pharmacology_terms={[p['term'] for p in defs['pharmacology']]}")
    print(f"glossary_terms={[g['term'] for g in defs['glossary']]}")
