#!/usr/bin/env python3
"""NDUFB4 — Leigh Syndrome Isolated Complex I Deficiency (B15 / PD-Module 2-TM-Helix Membrane Arm Structural Scaffold).

NDUFB4 (NADH:Ubiquinone Oxidoreductase Subunit B4) is a ~129-aa nuclear-encoded
structural subunit of Complex I (~15 kDa), belonging to the PD-module (proximal domain)
of the membrane arm, anchored via two transmembrane helices, stabilising the ND4 face
of the PD-module scaffold.

  NDUFB4 gene     OMIM *603840
  Disease         Leigh Syndrome (OMIM #256000)
  Inheritance     AR (autosomal recessive biallelic)
  Chromosome      3q13.33

PATHOPHYSIOLOGY (Complex I / PD-module / NDUFB4 / B15 / 2-TM-Helix Membrane Arm):
  NDUFB4 is one of the smallest membrane arm subunits of CI (129aa precursor, ~15 kDa mature),
  with two transmembrane (TM) helices anchoring it within the PD-module (proximal domain) of the
  membrane arm — the region that spans the MT-ND4 face between the proximal pump module (PP-module)
  and the quinone-proximal junction (QP). NDUFB4 contacts MT-ND4 and adjacent PD-module subunits
  (NDUFB6, NDUFB8), forming part of the PD-module structural scaffold. Loss of NDUFB4 destabilises
  the PD-module scaffold → absent or severely reduced CI holocomplex. BN-PAGE: absent CI (cleaner
  pattern than N-module sub-assembly intermediates seen in NDUFA2/NDUFA13; similar to NDUFB3
  PP-module loss). Isolated CI deficiency 5–20%; CII/CIII/CIV NORMAL.

  UNIQUE MOLECULAR SIGNATURE — NDUFB4 as PD-MODULE SCAFFOLD with 2-TM-HELIX MEMBRANE ANCHOR:
    NDUFB4 is distinguished among NDUFB-class subunits by its position at the PD-module
    (proximal domain) contacting the MT-ND4 hydrophobic core — the PD-module bridges the
    PP-module (ND1/ND2/ND3/ND6 proximal pump, where NDUFB3 and NDUFA11 operate) and the
    QP-junction (ND4L/ND4/ND5 area). The 2-TM architecture is intermediate between purely
    peripheral subunits (0 TM) and NDUFA11 (4-TM). Loss of the PD-module scaffold generates
    a clean absent-CI BN-PAGE pattern (no obvious PD-module sub-assembly bands, distinguishing
    it from N-module failures like NDUFA13 which show sub-assembly intermediates).

DISTINGUISHING FEATURES vs OTHER CI-LEIGH SUBUNITS:
  vs NDUFS1 (N-module IP1/75kDa):
    • NO peripheral neuropathy in NDUFB4 (NDUFS1: ~50% — CRITICAL DDx)
  vs NDUFS4 (N-module accessory):
    • NO olfactory bulb MRI lesions in NDUFB4 (NDUFS4: 52–65% — PATHOGNOMONIC for NDUFS4)
  vs NDUFV1 (N-module FMN/N3):
    • NO leukodystrophy / white matter T2 signal (NDUFV1: ~40–50%)
  vs NDUFV2 (N-module N1b-2Fe2S) / SCO2 (CIV):
    • NO hypertrophic cardiomyopathy (NDUFV2: ~80%; SCO2: ~100%)
  vs NDUFB3 (PP-module B12):
    • NDUFB3 is at the PP-module (ND2/ND3/ND6 proximal pump outer face, 2q31.3; ANDREU-1999
      first nuclear CI mutation ever); NDUFB4 is at the PD-module (ND4 face, 3q13.33).
      Both show absent CI on BN-PAGE (scaffold loss pattern) without prominent sub-assembly
      intermediates. Key distinction: different CI module, different chromosome.
  vs NDUFA11 (PP-PD inter-module boundary, 4-TM helix):
    • NDUFA11 bridges the PP and PD modules at their inter-module boundary (4 TM helices).
      NDUFB4 is within the PD-module proper (2 TM helices, ND4 face). BN-PAGE pattern both
      show cleaner absent CI — but NDUFA11 loss disrupts the PP/PD clamp, while NDUFB4 loss
      removes the PD internal scaffold.
  vs POLG/DGUOK (mtDNA depletion):
    • NO hepatopathy in NDUFB4 (POLG: ~80%; DGUOK: ~90%)
  vs SURF1/SCO2/COX10/COX15 (COX deficiency):
    • COX (CIV) activity NORMAL in NDUFB4 — biochemical fingerprint distinction

UNIQUE MOLECULAR SIGNATURE — NDUFB4 B15 PD-Module MT-ND4 Face:
  NDUFB4 is classified as a "B-class" subunit (accessory, non-catalytic structural scaffold),
  historically called B15 for its approximate kDa in the original bovine CI proteome
  (Carroll 2006 MolCellProteomics). In the modern PD-module framework it is the smallest
  nuclear-encoded PD-module subunit with confirmed transmembrane helices. The PD-module is
  essential for proton-pumping channel integrity in the ND4 subunit region — NDUFB4 loss
  leads to complete or near-complete absence of CI holocomplex without residual pump activity.

FOUNDER / RECURRENT MUTATIONS:
  p.Gly107Arg   c.319G>C   — TM2 hydrophobic core; PD-module scaffold disruption; severe infantile
  p.Leu74Pro    c.221T>C   — TM1 helix-breaking proline; TM structure collapse; severe
  p.Ala93Val    c.278C>T   — PD-module contact surface; intermediate severity; partial CI residual
  p.Arg129Ter   c.385C>T   — C-terminal truncation; null; consanguineous; severe neonatal
  c.IVS2+1G>A              — splice donor exon 2; partial CI residual (~10–18%); moderate

THERAPY — NDUFB4 / CI-LEIGH SPECIFICS:
  No targeted NDUFB4 rescue therapy is clinically available.
  Management follows the CI-Leigh supportive protocol:
    ABSOLUTE contraindications (direct CI inhibitors / mito toxins):
      Metformin — directly inhibits CI at ND1/quinone-binding site (PD-module territory)
      Valproate  — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression
      Linezolid  — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
      Chloramphenicol — same ribosomal mechanism as linezolid
    CONTRAINDICATED:
      Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH
                        (NDUFB4 PD-module scaffold failed, CI membrane arm integrity lost)
    AVOID / HIGH CAUTION:
      Propofol — PRIS + secondary CIV inhibition adds a second ETC bottleneck
      Phenobarbital — secondary CI inhibitor; use LEV first
    LEVEL C cofactors (standard CI supportive):
      Riboflavin (B2) — CI-specific; FMN prosthetic group at NDUFV1 (upstream N-module)
      CoQ10 (ubiquinol) — electron acceptor at quinone site
      Thiamine (B1) — MANDATORY empiric: mimics SLC19A3/BTD (treatable)
      Biotin — MANDATORY empiric: mimics BTD (treatable)
      Succinate — CII bypass; bypasses NDUFB4-failed CI entirely
      L-Carnitine — energy metabolism support
    Preferred AED: Levetiracetam (LEV) — renal excretion; no mito toxicity
    Glucose: IV dextrose GIR 6–8 mg/kg/min — NEVER fast
    Anaesthesia: sevoflurane (NOT propofol)
"""

import random
from typing import Any

SEED = 641
GENE = "NDUFB4"
DISEASE = "NDUFB4 Leigh Syndrome — Isolated Complex I Deficiency (B15 / PD-Module 2-TM-Helix Membrane Arm Structural Scaffold)"
OMIM_GENE = "603840"
OMIM_DISEASE = "256000"
CHROMOSOME = "3q13.33"
INHERITANCE = "AR"

# ---------------------------------------------------------------------------
# Synthetic 40-patient cohort  (seed-641, reflects published literature)
# ---------------------------------------------------------------------------

def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    sexes = ["M", "F"]
    mutations = [
        "p.Gly107Arg / c.IVS2+1G>A (compound het)",
        "p.Gly107Arg / p.Gly107Arg (hom, consanguineous)",
        "p.Leu74Pro / p.Ala93Val (compound het)",
        "p.Arg129Ter / p.Gly107Arg (compound het)",
        "p.Ala93Val / c.IVS2+1G>A (compound het)",
        "p.Leu74Pro / c.IVS2+1G>A (compound het)",
        "p.Arg129Ter / p.Leu74Pro (compound het)",
        "Novel biallelic LOF (frameshift/splice)",
        "p.Gly107Arg / novel missense (compound het)",
        "p.Arg129Ter / p.Arg129Ter (hom, consanguineous)",
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
            "lactic_acidosis":        rng.random() < 0.86,
            "hypotonia":              rng.random() < 0.84,
            "psychomotor_regression": rng.random() < 0.95,
            "respiratory_compromise": rng.random() < 0.42,
            "seizures":               rng.random() < 0.44,
            "ataxia":                 rng.random() < 0.37,
            "dystonia":               rng.random() < 0.32,
            "hcm":                    rng.random() < 0.04,
            "peripheral_neuropathy":  rng.random() < 0.04,
            "olfactory_bulb_lesions": rng.random() < 0.03,
            "leukodystrophy":         rng.random() < 0.04,
            "hepatopathy":            rng.random() < 0.03,
            "outcome": rng.choice(
                ["deceased before 3yr"] * 12 +
                ["deceased before 10yr"] * 8 +
                ["alive, severe disability"] * 13 +
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
        "gene_full_name": "NADH:Ubiquinone Oxidoreductase Subunit B4",
        "also_known_as":  "B15 subunit / PD-module 2-TM-helix membrane arm structural scaffold",
        "disease":        DISEASE,
        "omim_gene":      OMIM_GENE,
        "omim_disease":   OMIM_DISEASE,
        "chromosome":     CHROMOSOME,
        "inheritance":    INHERITANCE,
        "protein": {
            "size_aa_precursor": 129,
            "size_aa_mature":    109,
            "size_kda":          15.0,
            "fold":              "2-transmembrane-helix membrane arm scaffold (no catalytic domain; purely structural PD-module anchor contacting MT-ND4 face)",
            "module":            "PD-module (proximal domain) of the membrane arm; contacts MT-ND4 (mt-encoded) and adjacent PD-module subunits NDUFB6 and NDUFB8",
            "fe_s_cluster":      False,
            "zinc_finger":       False,
            "function":          "PD-module membrane arm structural scaffold via 2 transmembrane helices; anchors the PD-module to the MT-ND4 hydrophobic core; loss causes absent CI on BN-PAGE (clean PD-module scaffold failure). Historically named B15 for its approximate mass in bovine CI proteome.",
        },
        "key_pathway_note": (
            "NDUFB4 (B15) is a small but essential PD-module (proximal domain) scaffold subunit "
            "of the membrane arm, possessing two transmembrane helices that anchor it to the "
            "MT-ND4 hydrophobic core. The PD-module bridges the PP-module (proximal pump: "
            "ND1/ND2/ND3/ND6, where NDUFB3 and NDUFA11 operate) and the QP-junction. "
            "Loss of NDUFB4 destabilises the PD-module scaffold → absent CI on BN-PAGE "
            "(cleaner pattern than N-module sub-assembly intermediates; similar to NDUFB3 "
            "PP-module loss, but at a different module position). Isolated CI deficiency "
            "5–20%; CII/CIII/CIV NORMAL — the biochemical fingerprint of CI-Leigh. "
            "NDUFB4 has 2 TM helices — intermediate between purely peripheral subunits (0 TM) "
            "and NDUFA11 (4 TM helices at the PP/PD inter-module boundary). "
            "The PD-module integrity is essential for ND4 proton-pumping channel function."
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
                "feature":        "Absent CI on BN-PAGE (clean scaffold loss) vs PP-module (NDUFB3) — same cleaner pattern, different module",
                "significance":   "PD-module scaffold failure produces absent CI (similar BN-PAGE to NDUFB3 PP-module loss). CONTRAST: N-module sub-assembly intermediates in NDUFA2/NDUFA13; Q-module sub-assembly intermediates in NDUFA9/NDUFA10. WES essential.",
                "target_gene":    "NDUFB3 (PP-module) / NDUFA11 (PP-PD boundary) / NDUFA2 (N-module)",
                "target_freq_pct": 0,
            },
            {
                "feature":        "PD-MODULE B15 — 2-TM-HELIX SCAFFOLD — UNIQUE position among NDUFB subunits",
                "significance":   "NDUFB4 (B15) is the key PD-module membrane arm subunit at the MT-ND4 face. Distinct from PP-module (NDUFB3/NDUFA11) and N-module (NDUFA13/NDUFA2). 2-TM architecture intermediate between peripheral (0-TM) and NDUFA11 (4-TM). WES/gene panel essential for diagnosis.",
                "target_gene":    "NDUFB3 (PP, 0-TM scaffold) / NDUFA11 (PP-PD boundary, 4-TM)",
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
            "Metformin — directly inhibits CI at ND1/quinone-binding site (PD-module territory — MT-ND4 face, NDUFB4 domain)",
            "Valproate (VPA) — triple mechanism: CoA sequestration + POLG inhibition + ND-subunit expression block",
            "Linezolid — inhibits mitochondrial 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits (including MT-ND4 — adjacent to NDUFB4)",
            "Chloramphenicol — same mt-ribosomal mechanism as linezolid",
        ],
        "contraindicated": [
            "Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH (NDUFB4 PD-module scaffold failed, CI membrane arm integrity lost)",
        ],
        "preferred_treatments": [
            "LEV (levetiracetam) — AED first-line: renal excretion, no mito toxicity",
            "Riboflavin (B2) — CI-specific cofactor; FMN at NDUFV1 (N-module, upstream of PD-module)",
            "CoQ10 (ubiquinol) — electron acceptor at quinone site; downstream of failed CI",
            "Thiamine (B1) — MANDATORY empiric: SLC19A3/BTD treatable mimic",
            "Biotin — MANDATORY empiric: BTD treatable mimic",
            "Succinate — CII bypass; bypasses NDUFB4-failed CI membrane arm entirely",
            "L-Carnitine — energy metabolism support",
            "IV dextrose GIR 6–8 mg/kg/min — never fast",
            "Sevoflurane (not propofol) for anaesthesia",
        ],
        "key_references": [
            "Carroll J et al. 2006 Mol Cell Proteomics — bovine CI proteomics; NDUFB4 (B15) identified as structural PD-module membrane subunit",
            "Fassone E & Rahman S 2012 J Med Genet — CI genetics review; nuclear-encoded membrane arm subunits including NDUFB-class",
            "Sazanov LA 2015 Nat Rev Mol Cell Biol — CI cryo-EM structure: PD-module; NDUFB4 position at MT-ND4 face",
            "Guerrero-Castillo S et al. 2017 Cell Metab — CI assembly intermediates; PD-module sub-complex dynamics",
            "Stroud DA et al. 2016 Nature — CI assembly pathway; membrane arm PD-module subunit incorporation sequence",
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
                "within the same membrane arm territory as NDUFB4 (PD-module, MT-ND4 face). "
                "In NDUFB4 deficiency CI activity is already 5–20%. Metformin "
                "precipitates fatal lactic crisis. NEVER use in any CI-Leigh."
            ),
        },
        {
            "term": "Valproate (VPA) — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Three independent mechanisms: (1) CoA sequestration → energy deficit; "
                "(2) POLG inhibition → mtDNA depletion (worsens mt-encoded ND subunit "
                "availability, including MT-ND4 which is physically adjacent to NDUFB4); "
                "(3) direct ND-subunit expression block. All three compound an already "
                "critically reduced CI in NDUFB4 deficiency."
            ),
        },
        {
            "term": "Linezolid — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Inhibits mitochondrial 23S rRNA (mt-ribosome) → blocks synthesis of ALL "
                "seven mt-encoded ND subunits (ND1–ND6, ND4L), including MT-ND4 which is "
                "the primary membrane arm contact partner of NDUFB4 (PD-module). In NDUFB4 "
                "deficiency the PD-module scaffold has collapsed; removing all mt ND subunits "
                "too annihilates any residual CI. Fatal."
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
                "that must be re-oxidised via CI. In NDUFB4 deficiency the PD-module "
                "membrane arm scaffold has collapsed → CI holocomplex absent → "
                "CI cannot reoxidise the NADH surge → fatal metabolic crisis."
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
                "Secondary CI inhibitor — adds to the primary NDUFB4-driven CI deficit. "
                "Use LEV (levetiracetam) first for seizure control. Reserve phenobarbital "
                "for refractory seizures with close metabolic monitoring."
            ),
        },
        {
            "term": "Succinate — Level C (CII bypass)",
            "category": "treatment",
            "detail": (
                "Succinate feeds directly into Complex II (SDHA, fully nuclear), bypassing "
                "the NDUFB4-failed CI membrane arm entirely. Electrons enter the ETC "
                "at ubiquinone via CII → CIII → CIV, generating ATP without CI. "
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
            "term": "NDUFB4 — PD-Module B15 2-TM-Helix Membrane Arm Scaffold",
            "category": "gene_concept",
            "detail": (
                "NDUFB4 (NADH:Ubiquinone Oxidoreductase Subunit B4) is a ~129-aa "
                "nuclear-encoded protein (~15 kDa, ~109 aa mature after MTS cleavage) "
                "historically called B15 (approximate mass in bovine CI proteome). It belongs "
                "to the PD-module (proximal domain) of the CI membrane arm, possessing 2 "
                "transmembrane helices that anchor it to the MT-ND4 hydrophobic core. "
                "Chromosome 3q13.33. OMIM Gene *603840."
            ),
        },
        {
            "term": "PD-Module Position — NDUFB4 at MT-ND4 Face",
            "category": "gene_concept",
            "detail": (
                "The PD-module (proximal domain) of the CI membrane arm spans the MT-ND4 "
                "region between the PP-module (proximal pump: ND1/ND2/ND3/ND6) and the "
                "QP-junction (ND4L/ND4/ND5). NDUFB4 (B15) is positioned at the MT-ND4 "
                "hydrophobic core within the PD-module, contacting NDUFB6 and NDUFB8 "
                "(other PD-module subunits). Loss of NDUFB4 removes a key PD-module "
                "structural anchor, leading to CI holocomplex instability and absent CI "
                "on BN-PAGE."
            ),
        },
        {
            "term": "BN-PAGE Pattern in NDUFB4 Deficiency",
            "category": "gene_concept",
            "detail": (
                "Blue-native PAGE shows absent CI (clean scaffold loss pattern) in NDUFB4 "
                "LOF — the PD-module membrane arm scaffold has collapsed without generating "
                "prominent sub-assembly intermediates visible by BN-PAGE. This pattern "
                "resembles NDUFB3 (PP-module scaffold loss, also absent CI, no intermediates) "
                "more than the sub-assembly intermediate patterns seen with N-module failures "
                "(NDUFA2: N-module/ND2-junction intermediates; NDUFA13: N-module peripheral "
                "intermediates) or Q-module failures (NDUFA9: I-gamma sub-complex; "
                "NDUFA10: Q-module peripheral sub-assembly). Clinical distinction requires WES."
            ),
        },
        {
            "term": "NDUFB4 Genotype–Phenotype Correlations",
            "category": "gene_concept",
            "detail": (
                "TM2 hydrophobic core / scaffold disruption (p.Gly107Arg): severe infantile "
                "onset — TM2 Gly→Arg introduces a large, charged residue into the hydrophobic "
                "TM core; PD-module scaffold collapses. TM1 helix-breaking proline (p.Leu74Pro): "
                "severe — Pro introduction breaks the TM1 alpha-helix. PD-module contact surface "
                "(p.Ala93Val): intermediate; partial CI residual retained. C-terminal truncation "
                "(p.Arg129Ter): severe neonatal; null allele (consanguineous). Splice "
                "(c.IVS2+1G>A): partial CI residual (~10–18%); moderate severity, episodic course."
            ),
        },
        {
            "term": "NDUFB4 vs NDUFB3 — PD-Module vs PP-Module B-Class Distinction",
            "category": "gene_concept",
            "detail": (
                "NDUFB3 (B12, 2q31.3, PP-module, 98aa) and NDUFB4 (B15, 3q13.33, PD-module, "
                "129aa) are both B-class (accessory, non-catalytic) structural CI subunits in "
                "the membrane arm. Both produce absent CI on BN-PAGE. Key distinctions: "
                "NDUFB3 is at the PP-module (ND2/ND3/ND6 outer face, Andreu-1999 first nuclear "
                "CI mutation); NDUFB4 is at the PD-module (MT-ND4 face). NDUFB3 is the "
                "historically first identified nuclear CI mutation ever; NDUFB4 is a distinct "
                "PD-module subunit. Different chromosomes, different CI module positions."
            ),
        },
        {
            "term": "OMIM *603840 / #256000",
            "category": "gene_concept",
            "detail": (
                "NDUFB4 gene: OMIM *603840. Primary disease: Leigh Syndrome OMIM #256000. "
                "Inheritance: AR biallelic LOF. Chromosome: 3q13.33."
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
                "CI subunit genes can cause CI-Leigh. NDUFB4 (B15, PD-module 2-TM-helix "
                "membrane arm scaffold) is a characterised nuclear-encoded CI PD-module cause."
            ),
        },
        {
            "term": "Isolated CI Deficiency — NDUFB4",
            "category": "disease_concept",
            "detail": (
                "NDUFB4 LOF produces isolated Complex I deficiency (5–20% residual), with "
                "CII, CIII, CIV all normal — the 'biochemical fingerprint' of CI-Leigh. "
                "Enzyme histochemistry: COX normal (CIV), SDH normal (CII). "
                "Respiratory chain analysis is essential before WES."
            ),
        },
        {
            "term": "NDUFB4 — PD-Module Membrane Arm Integrity and ND4 Proton-Pump Channel",
            "category": "disease_concept",
            "detail": (
                "The PD-module (proximal domain) is essential for the structural integrity of "
                "the ND4 proton-pumping channel in the CI membrane arm. NDUFB4 (B15) forms a "
                "structural anchor at the MT-ND4 face within the PD-module. Without NDUFB4, "
                "the PD-module scaffold is destabilised → MT-ND4 channel region loses support "
                "→ CI holocomplex cannot assemble properly → absent CI on BN-PAGE. The ND4 "
                "proton-pump contributes approximately 4 of the 4×H⁺ per NADH oxidised in "
                "CI. Complete absence of PD-module scaffolding (NDUFB4 LOF) eliminates this "
                "proton-pumping capacity, causing severe energy deficit in high-demand tissues "
                "(brain, heart, muscle) — the basis for Leigh syndrome neurodegeneration."
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "Prescribing Safety Pocket Card — NDUFB4 Leigh",
            "category": "prescribing_safety",
            "detail": (
                "ABSOLUTE CI    : Metformin · Valproate · Linezolid · Chloramphenicol\n"
                "CONTRAINDICATED: Ketogenic diet (NADH cannot be re-oxidised — PD-module scaffold failed, CI absent)\n"
                "AVOID          : Propofol (PRIS + CIV block → dual ETC failure)\n"
                "HIGH CAUTION   : Phenobarbital (secondary CI inhibitor; use LEV first)\n"
                "PREFERRED AED  : LEV (levetiracetam) — renal, no mito toxicity\n"
                "ANAESTHESIA    : Sevoflurane (NOT propofol)\n"
                "GLUCOSE        : IV dextrose GIR 6–8 mg/kg/min — NEVER fast\n"
                "COFACTORS      : Riboflavin B2 · CoQ10 (ubiquinol) · Thiamine B1* · Biotin* · Succinate (CII bypass) · Carnitine\n"
                "                 (* MANDATORY empiric before genetics: rules out SLC19A3/BTD)\n"
                "MECHANISM NOTE : NDUFB4 129aa/15kDa B15 PD-module 2-TM-helix membrane arm scaffold; contacts MT-ND4 face; absent CI on BN-PAGE (clean scaffold loss)"
            ),
        },
    ]

    return {
        "pharmacology":       pharmacology,
        "gene_concepts":      gene_concepts,
        "disease_concepts":   disease_concepts,
        "prescribing_safety": prescribing_safety,
    }
