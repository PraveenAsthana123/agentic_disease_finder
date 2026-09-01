#!/usr/bin/env python3
"""NDUFS6 — Leigh Syndrome Isolated Complex I Deficiency (Q-Module Zinc-Finger Subunit / CXXC Motif / Quinone-Arm Junction).

NDUFS6 (NADH:Ubiquinone Oxidoreductase Core Subunit S6) is a 124-aa nuclear-encoded
subunit of Complex I located in the Q-module (quinone-arm), at the interface between
the hydrophilic peripheral arm and the membrane arm.  It carries a zinc-binding CXXC
tetracysteine motif — NOT an iron-sulfur cluster — that coordinates a structural zinc
ion essential for Q-module assembly and quinone-site integrity.

  NDUFS6 gene      OMIM *603848
  Disease          Leigh Syndrome (OMIM #256000) / CI Deficiency Nuclear Type 3 (OMIM #618225)
  Inheritance      AR (autosomal recessive biallelic)
  Chromosome       5p15.33

PATHOPHYSIOLOGY (Complex I / Q-module zinc-binding / NDUFS6):
  NDUFS6 resides in the Q-module (quinone-reduction module) at the junction between
  the peripheral arm and the proximal membrane arm.  Its CXXC zinc-finger motif
  (Cys-X-X-Cys tetracysteine) coordinates a structural Zn²⁺ ion essential for
  Q-module assembly and correct positioning of the quinone-reduction site.

  The Fe-S electron relay (entirely in the N-module and Q-module, upstream of NDUFS6):
    NDUFV1  (N3, [4Fe-4S]) ← FMN primary NADH acceptor (N-module)
    NDUFV2  (N1b, [2Fe-2S]) ← second relay step
    NDUFS7  (N4, [4Fe-4S]) ← third relay (Q-module)
    NDUFS8  (N6a/N6b, dual [4Fe-4S]) ← fourth/fifth relay (TYKY, Q-module)
    NDUFS1  (N5, [4Fe-4S]) ← peripheral arm relay
    NDUFS2  (N2, [4Fe-4S]) ← TERMINAL relay → ubiquinone reduction

  NDUFS6's unique role:
    1. NDUFS6 does NOT carry an Fe-S cluster; its Zn²⁺ is structural.
    2. The CXXC zinc finger stabilises the NDUFS2-containing Q-module sub-assembly,
       keeping the terminal N2 Fe-S cluster properly oriented toward ubiquinone.
    3. Loss of NDUFS6 disrupts Q-module architecture → CI assembly stalls.
       Sub-assembly intermediates visible on BN-PAGE (Q-module + membrane arm
       partially separated) — pattern similar to other structural/assembly defects
       (NDUFS3, NDUFS4, NDUFS5) rather than the cleaner absent-CI seen with direct
       Fe-S relay loss (NDUFS7, NDUFS8).
    4. Net result: isolated CI deficiency 5–20%; CII/CIII/CIV NORMAL.

  NDUFS6 CXXC zinc-finger — unique mechanistic point:
    • The structural Zn²⁺ is coordinated by four Cys residues (CXXC × 2).
    • Missense mutations disrupting CXXC motif residues abolish zinc binding →
      Q-module collapse even in proteins that fold partially correctly.
    • This mechanistic uniqueness makes NDUFS6 the only CI subunit with a bona fide
      zinc-finger structural domain (cf. NDUFS7/NDUFS8 iron-sulfur, NDUFV1/NDUFV2 FMN/Fe-S).

  Biochemical signature (identical to all CI-Leigh nuclear subunit mutations):
    Complex I: 5–20% of control (severely reduced) — ISOLATED CI DEFICIENCY
    Complex II (SDHA): NORMAL
    Complex III: NORMAL
    Complex IV (COX): NORMAL — KEY DDx SURF1/SCO2/COX10/COX15

DISTINGUISHING FEATURES vs OTHER CI-LEIGH SUBUNITS:
  vs NDUFS1 (IP1/75kDa, peripheral arm N-module):
    • NO peripheral neuropathy in NDUFS6 (NDUFS1: ~50% — CRITICAL DDx)
  vs NDUFS4 (N-module accessory, 5q11.2):
    • NO olfactory bulb MRI lesions in NDUFS6 (NDUFS4: 52–65% — PATHOGNOMONIC for NDUFS4)
  vs NDUFV1 (N-module FMN/N3):
    • NO leukodystrophy / white matter T2 signal (NDUFV1: ~40–50%)
  vs NDUFV2 (N-module N1b-2Fe2S) / SCO2 (CIV assembly):
    • NO hypertrophic cardiomyopathy (NDUFV2: ~80%; SCO2: ~100%)
  vs NDUFS7 / NDUFS8 (direct Fe-S relay subunits):
    • BN-PAGE: NDUFS6 shows Q-module sub-assembly intermediates (zinc-finger collapse)
      vs NDUFS7/NDUFS8 cleaner absent CI (direct relay block)
  vs NDUFA2 / NDUFB3 (structural junction / membrane arm):
    • NDUFS6 is Q-module zinc-finger (NOT N-module junction or PP-module scaffolding)
      — BN-PAGE pattern may overlap but Zn²⁺-binding mechanism is distinct
  vs POLG/DGUOK (mtDNA depletion syndromes):
    • NO hepatopathy in NDUFS6 (POLG: ~80%; DGUOK: ~90%)
  vs GRACILE (BCS1L/Complex III):
    • NO iron overload; NO aminoaciduria; NO neonatal cholestasis
  vs SURF1/SCO2/COX10/COX15 (COX deficiency):
    • COX activity NORMAL in NDUFS6 — biochemical fingerprint distinction

FOUNDER / RECURRENT MUTATIONS:
  c.IVS3+5G>A  (splice donor intron 3; partial CI residual; moderate course)
  p.Tyr119Cys  c.356A>G  — CXXC Zn-finger motif disruption; abolishes Zn²⁺ binding; severe
  p.Arg108Ter  c.322C>T  — null; homozygous in consanguineous; severe neonatal
  p.Ala122Val  c.365C>T  — near C-terminus; partial function; milder/moderate
  p.Arg14Gln   c.41G>A   — signal peptide region; neonatal severe presentation

THERAPY — NDUFS6 / CI-LEIGH SPECIFICS:
  No targeted NDUFS6 zinc-finger restoration is clinically available.
  Management follows the CI-Leigh supportive protocol:
    ABSOLUTE contraindications (direct CI inhibitors / mito toxins):
      Metformin — directly inhibits complex I at ND1/quinone-binding site
      Valproate  — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression
      Linezolid  — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
      Chloramphenicol — same ribosomal mechanism as linezolid
    CONTRAINDICATED:
      Ketogenic diet — forces NADH → β-oxidation; Q-module already collapsed
    AVOID / HIGH CAUTION:
      Propofol — PRIS + secondary CIV inhibition adds a second ETC bottleneck
      Phenobarbital — secondary CI inhibitor; use LEV first
    LEVEL C cofactors (standard CI supportive):
      Riboflavin (B2) — CI-specific; FMN prosthetic group upstream of NDUFS6
      CoQ10 (ubiquinol) — electron acceptor at quinone site (NDUFS6 module)
      Thiamine (B1) — MANDATORY empiric: mimics SLC19A3/BTD (treatable)
      Biotin — MANDATORY empiric: mimics BTD (treatable)
      Succinate — CII bypass; bypasses NDUFS6-failed CI Q-module entirely
      L-Carnitine — energy metabolism support
    Preferred AED: Levetiracetam (LEV) — renal excretion; no mito toxicity
    Glucose: IV dextrose GIR 6–8 mg/kg/min — NEVER fast
    Anaesthesia: sevoflurane (NOT propofol)
"""

import random
from typing import Any

SEED = 629
GENE = "NDUFS6"
DISEASE = "NDUFS6 Leigh Syndrome — Isolated Complex I Deficiency (Q-Module Zinc-Finger Subunit / CXXC Motif / Quinone-Arm Junction)"
OMIM_GENE = "603848"
OMIM_DISEASE = "256000"
OMIM_DISEASE_CI = "618225"
CHROMOSOME = "5p15.33"
INHERITANCE = "AR"

# ---------------------------------------------------------------------------
# Synthetic 40-patient cohort  (seed-629, reflects published literature)
# ---------------------------------------------------------------------------

def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    sexes = ["M", "F"]
    mutations = [
        "c.IVS3+5G>A / c.IVS3+5G>A (hom splice)",
        "p.Tyr119Cys / c.IVS3+5G>A (compound het)",
        "p.Arg108Ter / c.IVS3+5G>A (compound het)",
        "p.Tyr119Cys / p.Arg108Ter (compound het)",
        "p.Ala122Val / c.IVS3+5G>A (compound het)",
        "p.Arg14Gln / c.IVS3+5G>A (compound het)",
        "p.Tyr119Cys / p.Ala122Val (compound het)",
        "p.Arg108Ter / p.Arg14Gln (compound het)",
        "Novel biallelic LOF",
        "p.Tyr119Cys / novel missense (compound het)",
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
            "leigh_mri":              rng.random() < 0.82,
            "lactic_acidosis":        rng.random() < 0.88,
            "hypotonia":              rng.random() < 0.87,
            "psychomotor_regression": rng.random() < 0.95,
            "respiratory_compromise": rng.random() < 0.50,
            "seizures":               rng.random() < 0.48,
            "ataxia":                 rng.random() < 0.42,
            "dystonia":               rng.random() < 0.38,
            "hcm":                    rng.random() < 0.05,
            "peripheral_neuropathy":  rng.random() < 0.04,
            "olfactory_bulb_lesions": rng.random() < 0.03,
            "leukodystrophy":         rng.random() < 0.04,
            "hepatopathy":            rng.random() < 0.04,
            "outcome": rng.choice(
                ["deceased before 3yr"] * 14 +
                ["deceased before 10yr"] * 8 +
                ["alive, severe disability"] * 11 +
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
        "gene_full_name": "NADH:Ubiquinone Oxidoreductase Core Subunit S6",
        "also_known_as":  "AQDQ subunit (CI-TYKY-proximal zinc-finger)",
        "disease":        DISEASE,
        "omim_gene":      OMIM_GENE,
        "omim_disease":   OMIM_DISEASE,
        "omim_disease_ci": OMIM_DISEASE_CI,
        "chromosome":     CHROMOSOME,
        "inheritance":    INHERITANCE,
        "protein": {
            "size_aa_precursor": 124,
            "size_aa_mature":    94,
            "size_kda":          13.5,
            "fold":              "Zinc-finger (CXXC × 2 tetracysteine motif)",
            "module":            "Q-module (quinone-arm, NDUFS2-N2 cluster vicinity)",
            "fe_s_cluster":      False,
            "zinc_finger":       True,
            "function":          "Structural zinc-finger stabilising Q-module assembly and quinone-site integrity; Zn²⁺ coordinated by dual CXXC motif",
        },
        "key_pathway_note": (
            "NDUFS6 occupies the Q-module of Complex I, adjacent to NDUFS2 (N2 cluster, "
            "terminal Fe-S relay → ubiquinone).  Its dual CXXC zinc-finger coordinates a "
            "structural Zn²⁺ ion that anchors Q-module sub-assembly.  Loss abolishes Zn²⁺ "
            "binding → Q-module collapses → CI assembly failure.  NDUFS6 carries NO "
            "iron-sulfur cluster of its own — it is a structural zinc-finger, not a relay "
            "subunit.  BN-PAGE: Q-module sub-assembly intermediates (not clean absent-CI). "
            "Unique mechanistic footprint: the CXXC zinc-finger is found in no other CI "
            "structural subunit, making genotype-directed zinc supplementation a theoretical "
            "(but unproven) investigational concept."
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
                "feature":        "Q-module sub-assembly intermediates on BN-PAGE (zinc-finger collapse)",
                "significance":   "Structural zinc-finger defect — overlaps NDUFS3/NDUFS4/NDUFS5 assembly patterns; DISTINCT from Fe-S relay block (NDUFS7/NDUFS8 cleaner absent CI)",
                "target_gene":    "NDUFS7 / NDUFS8",
                "target_freq_pct": 0,
            },
            {
                "feature":        "CXXC zinc-finger NDUFS6 — unique mechanistic class",
                "significance":   "Only CI structural subunit with dual-CXXC Zn²⁺ motif; no other CI-Leigh gene shares this mechanism — confirmed by NDUFS6 WES/panel",
                "target_gene":    "all other CI-Leigh subunits",
                "target_freq_pct": 0,
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
            "Metformin — directly inhibits CI at ND1/quinone-binding site (same module as NDUFS6 Q-module)",
            "Valproate (VPA) — triple mechanism: CoA sequestration + POLG inhibition + ND-subunit expression block",
            "Linezolid — inhibits mitochondrial 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits",
            "Chloramphenicol — same mt-ribosomal mechanism as linezolid",
        ],
        "contraindicated": [
            "Ketogenic diet — forces NADH → β-oxidation through a Q-module that has collapsed (Zn²⁺ lost)",
        ],
        "preferred_treatments": [
            "LEV (levetiracetam) — AED first-line: renal excretion, no mito toxicity",
            "Riboflavin (B2) — CI-specific cofactor; FMN upstream of NDUFS6 Q-module",
            "CoQ10 (ubiquinol) — electron acceptor at the quinone site (NDUFS6 module territory)",
            "Thiamine (B1) — MANDATORY empiric: SLC19A3/BTD treatable mimic",
            "Biotin — MANDATORY empiric: BTD treatable mimic",
            "Succinate — CII bypass; bypasses NDUFS6-failed CI Q-module entirely",
            "L-Carnitine — energy metabolism support",
            "IV dextrose GIR 6–8 mg/kg/min — never fast",
            "Sevoflurane (not propofol) for anaesthesia",
        ],
        "key_references": [
            "Kirby DM et al. 2004 Ann Neurol — Early NDUFS6 patients: CI deficiency with Leigh syndrome and neonatal onset",
            "Tucker EJ et al. 2011 NatGenet — NDUFS6 mutations: Q-module zinc-finger and CI assembly failure",
            "Fassone E & Rahman S 2012 JMedGenet — CI genetics review (nuclear-encoded subunits)",
            "Sazanov LA 2015 NatRevMolCellBiol — CI cryo-EM structure: NDUFS6 CXXC zinc-finger in Q-module",
            "Stroud DA et al. 2016 Nature — CI assembly pathway mapping: NDUFS6 Q-module positioning",
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
                "Metformin directly inhibits CI at the ND1/quinone-binding interface. "
                "In NDUFS6 deficiency the Q-module has collapsed (Zn²⁺ lost from CXXC motif); "
                "CI activity is already 5–20%. Metformin precipitates fatal lactic crisis. "
                "Note: the quinone-binding site (ND1/NDUFS2-N2 interface) is the SAME "
                "module as NDUFS6 — making metformin uniquely dangerous here."
            ),
        },
        {
            "term": "Valproate (VPA) — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Three independent mechanisms: (1) CoA sequestration → energy deficit; "
                "(2) POLG inhibition → mtDNA depletion (worsens mt-ND subunit availability); "
                "(3) direct ND-subunit expression block. In NDUFS6 deficiency all three "
                "compound an already critically reduced CI."
            ),
        },
        {
            "term": "Linezolid — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Inhibits mitochondrial 23S rRNA (mt-ribosome) → blocks synthesis of ALL "
                "seven mt-encoded ND subunits (ND1–ND6, ND4L). In NDUFS6 deficiency the "
                "nuclear Q-module subunit is absent; removing all mt ND subunits too "
                "annihilates any residual CI. Fatal."
            ),
        },
        {
            "term": "Chloramphenicol — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Same mitochondrial ribosomal inhibition mechanism as linezolid. "
                "Blocks synthesis of all mt-encoded ND subunits. Never use in CI-Leigh."
            ),
        },
        {
            "term": "Ketogenic diet — CONTRAINDICATED",
            "category": "contraindicated",
            "detail": (
                "KD forces energy production through fatty-acid β-oxidation, generating NADH "
                "that must be re-oxidised via CI. In NDUFS6 deficiency the Q-module has "
                "collapsed → CI cannot reoxidise the NADH surge → fatal metabolic crisis."
            ),
        },
        {
            "term": "Propofol — AVOID",
            "category": "caution",
            "detail": (
                "PRIS (propofol infusion syndrome) inhibits CIV, creating a SECOND ETC "
                "bottleneck downstream of the already-failed CI. Combined CI+CIV failure "
                "is catastrophic. Sevoflurane is preferred for all anaesthesia."
            ),
        },
        {
            "term": "Phenobarbital — HIGH CAUTION",
            "category": "caution",
            "detail": (
                "Secondary CI inhibitor — adds to the primary NDUFS6-driven CI deficit. "
                "Use LEV (levetiracetam) first for seizure control. Reserve phenobarbital "
                "for refractory seizures with close metabolic monitoring."
            ),
        },
        {
            "term": "Succinate — Level C (CII bypass)",
            "category": "treatment",
            "detail": (
                "Succinate feeds directly into Complex II (SDHA, fully nuclear), bypassing "
                "the NDUFS6-failed Q-module of CI entirely. Electrons enter the ETC at "
                "ubiquinone via CII → CIII → CIV, generating ATP without CI. "
                "Level C evidence; standard adjunct in CI-Leigh."
            ),
        },
        {
            "term": "LEV (Levetiracetam) — Preferred AED",
            "category": "treatment",
            "detail": (
                "Renal excretion; no hepatic metabolism; no mito toxicity; no CI inhibition. "
                "First-line AED in all CI-Leigh syndromes. SV2A mechanism unrelated to ETC."
            ),
        },
    ]

    gene_concepts = [
        {
            "term": "NDUFS6 — Q-Module Zinc-Finger Subunit",
            "category": "gene_concept",
            "detail": (
                "NDUFS6 (NADH:Ubiquinone Oxidoreductase Core Subunit S6) is a 124-aa "
                "nuclear-encoded protein (~94 aa mature, ~13.5 kDa) located in the Q-module "
                "(quinone-reduction module) of Complex I, adjacent to NDUFS2 (N2 terminal "
                "Fe-S cluster). Its CXXC × 2 tetracysteine zinc-finger coordinates a "
                "structural Zn²⁺ essential for Q-module architecture. Chromosome 5p15.33. "
                "OMIM Gene *603848."
            ),
        },
        {
            "term": "CXXC Zinc-Finger Motif — Unique CI Structural Element",
            "category": "gene_concept",
            "detail": (
                "The dual CXXC tetracysteine motif (Cys-X-X-Cys repeated) in NDUFS6 "
                "coordinates a structural zinc ion — NOT an iron-sulfur cluster. This "
                "makes NDUFS6 mechanistically unique among CI structural subunits. "
                "Missense variants disrupting any CXXC cysteine abolish zinc binding → "
                "Q-module collapse even when the polypeptide is partially folded. "
                "Zinc supplementation is theoretically interesting but unproven clinically."
            ),
        },
        {
            "term": "Q-Module Architecture",
            "category": "gene_concept",
            "detail": (
                "The Q-module (quinone module) contains the ubiquinone reduction site and "
                "the terminal Fe-S relay cluster N2 (NDUFS2). NDUFS6, NDUFS7, NDUFS8, "
                "NDUFS2 and NDUFS3 form the Q-module core. NDUFS6 provides structural "
                "zinc-finger scaffolding; NDUFS7/NDUFS8 provide Fe-S relay steps. "
                "Loss of NDUFS6 → Q-module disassembly → CI assembly blocked before "
                "holocomplex can form."
            ),
        },
        {
            "term": "BN-PAGE Pattern in NDUFS6 Deficiency",
            "category": "gene_concept",
            "detail": (
                "Blue-native PAGE shows Q-module sub-assembly intermediates in NDUFS6 LOF — "
                "the Q-module partially separates from the rest of the arm. This overlaps "
                "with other structural/assembly-defect patterns (NDUFS3, NDUFS4, NDUFS5, "
                "NDUFA2). Contrast: NDUFS7/NDUFS8 (direct Fe-S relay block) show a "
                "cleaner absent-CI band with fewer intermediates."
            ),
        },
        {
            "term": "OMIM *603848 / #256000 / #618225",
            "category": "gene_concept",
            "detail": (
                "NDUFS6 gene: OMIM *603848. Primary disease: Leigh Syndrome OMIM #256000. "
                "CI Deficiency Nuclear Type 3: OMIM #618225. "
                "Inheritance: AR biallelic LOF. Chromosome: 5p15.33."
            ),
        },
    ]

    disease_concepts = [
        {
            "term": "Leigh Syndrome — CI-Leigh Nuclear Subunit Series",
            "category": "disease_concept",
            "detail": (
                "Leigh syndrome (OMIM #256000) is a severe mitochondrial encephalopathy "
                "characterised by bilateral brainstem/basal ganglia lesions on MRI, "
                "psychomotor regression, hypotonia, and lactic acidosis. Over 50 nuclear "
                "CI subunit genes can cause CI-Leigh. NDUFS6 is one of the less common "
                "CI nuclear subunit causes."
            ),
        },
        {
            "term": "Isolated CI Deficiency — Nuclear Type 3 (NDUFS6)",
            "category": "disease_concept",
            "detail": (
                "NDUFS6 LOF produces isolated Complex I deficiency (5–20% residual), with "
                "CII, CIII, CIV all normal — the 'biochemical fingerprint' of CI-Leigh. "
                "CI Nuclear Type 3 (OMIM #618225) specifically names NDUFS6 as the causal "
                "gene. Enzyme histochemistry: COX normal (CIV), SDH normal (CII). "
                "Respiratory chain analysis is essential before WES."
            ),
        },
        {
            "term": "NDUFS6 Genotype–Phenotype Correlations",
            "category": "disease_concept",
            "detail": (
                "Null alleles (p.Arg108Ter, frameshift): severe neonatal/early infantile onset, "
                "CI <10%, rapid deterioration. CXXC-disrupting missense (p.Tyr119Cys): "
                "severe, Zn²⁺ binding abolished. Hypomorphic splice (c.IVS3+5G>A): "
                "partial CI residual (10–20%), moderate course with episodic crises. "
                "Near-C-terminal missense (p.Ala122Val): milder, partial Q-module assembly."
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "Prescribing Safety Pocket Card — NDUFS6 Leigh",
            "category": "prescribing_safety",
            "detail": (
                "ABSOLUTE CI    : Metformin · Valproate · Linezolid · Chloramphenicol\n"
                "CONTRAINDICATED: Ketogenic diet (NADH cannot be re-oxidised — Q-module collapsed)\n"
                "AVOID          : Propofol (PRIS + CIV block → dual ETC failure)\n"
                "HIGH CAUTION   : Phenobarbital (secondary CI inhibitor; use LEV first)\n"
                "PREFERRED AED  : LEV (levetiracetam) — renal, no mito toxicity\n"
                "ANAESTHESIA    : Sevoflurane (NOT propofol)\n"
                "GLUCOSE        : IV dextrose GIR 6–8 mg/kg/min — NEVER fast\n"
                "COFACTORS      : Riboflavin B2 · CoQ10 (ubiquinol) · Thiamine B1* · Biotin* · Succinate · Carnitine\n"
                "                 (* MANDATORY empiric before genetics: rules out SLC19A3/BTD)\n"
                "MECHANISM NOTE : NDUFS6 CXXC zinc-finger in Q-module; quinone-site is same module as metformin ND1 target"
            ),
        },
    ]

    return {
        "pharmacology":       pharmacology,
        "gene_concepts":      gene_concepts,
        "disease_concepts":   disease_concepts,
        "prescribing_safety": prescribing_safety,
    }
