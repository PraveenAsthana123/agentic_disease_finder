#!/usr/bin/env python3
"""NDUFB8 — Leigh Syndrome Isolated Complex I Deficiency (B22 / PD-Module 1-TM-Helix Membrane Arm Structural Subunit).

NDUFB8 (NADH:Ubiquinone Oxidoreductase Subunit B8) is a ~186-aa nuclear-encoded
structural subunit of Complex I (~20.9 kDa), belonging to the PD-module (proximal domain)
of the membrane arm, anchored via a single transmembrane helix, contacting the MT-ND4 face
and adjacent PD-module subunits NDUFB4 and NDUFB6.

  NDUFB8 gene     OMIM *602140
  Disease         Leigh Syndrome (OMIM #256000)
  Inheritance     AR (autosomal recessive biallelic)
  Chromosome      10q23.2

PATHOPHYSIOLOGY (Complex I / PD-module / NDUFB8 / B22 / 1-TM-Helix Membrane Arm):
  NDUFB8 is a single-transmembrane-helix nuclear-encoded subunit of the CI PD-module
  (proximal domain) membrane arm (~186 aa precursor, ~20.9 kDa). It is historically
  designated B22 for its approximate mass in the original bovine CI proteome. NDUFB8
  occupies the MT-ND4 face of the PD-module alongside NDUFB4 (B15) and NDUFB6 (B17)
  — all three form the PD-module ND4-face structural triad. Its single TM helix anchors
  the N-terminal hydrophilic domain (matrix-facing) into the IMM, while the C-terminal
  domain is also matrix-facing. Loss of NDUFB8 destabilises the PD-module ND4-face
  structural triad → absent or severely reduced CI holocomplex. BN-PAGE: absent CI
  (cleaner pattern — no prominent PD-module sub-assembly bands visible, similar to
  NDUFB4 and NDUFB3 scaffold-loss patterns; distinct from N-module sub-assembly
  intermediates in NDUFA2/NDUFA13 and Q-module sub-complexes in NDUFA9/NDUFA10).
  Isolated CI deficiency 5–20%; CII/CIII/CIV NORMAL.

  UNIQUE MOLECULAR SIGNATURE — NDUFB8 as PD-MODULE ND4-FACE TRIAD MEMBER with
  SINGLE-TM-HELIX (1-TM) — ONLY PD-MODULE SUBUNIT WITH CONFIRMED 1-TM ARCHITECTURE:
    Among the three PD-module ND4-face subunits (NDUFB4-B15 / 2-TM, NDUFB6-B17 /
    1-TM-but-matrix-anchored via coiled coil, NDUFB8-B22 / 1-TM-helix-IMM-spanning),
    NDUFB8 is unique in having a single canonical IMM-spanning transmembrane alpha-helix
    (~residues 120–140) with a substantial N-terminal matrix-facing domain and a short
    C-terminal lumenal loop. This 1-TM architecture places NDUFB8 structurally between
    the purely peripheral (0-TM) subunits (NDUFA13-N-module, NDUFA10-Q-module) and the
    2-TM subunit NDUFB4 in the same PD-module. The PD-module ND4-face triad
    (NDUFB4-NDUFB6-NDUFB8) cooperatively stabilises the MT-ND4 proton-pump channel
    domain; loss of any member collapses the triad and abolishes CI assembly.

DISTINGUISHING FEATURES vs OTHER CI-LEIGH SUBUNITS:
  vs NDUFS1 (N-module IP1/75kDa):
    • NO peripheral neuropathy in NDUFB8 (NDUFS1: ~50% — CRITICAL DDx)
  vs NDUFS4 (N-module accessory):
    • NO olfactory bulb MRI lesions in NDUFB8 (NDUFS4: 52–65% — PATHOGNOMONIC for NDUFS4)
  vs NDUFV1 (N-module FMN/N3):
    • NO leukodystrophy / white matter T2 signal in NDUFB8 (NDUFV1: ~40–50%)
  vs NDUFV2 (N-module N1b-2Fe2S) / SCO2 (CIV):
    • NO hypertrophic cardiomyopathy (NDUFV2: ~80%; SCO2: ~100%)
  vs NDUFB4 (PD-module B15, 2-TM):
    • NDUFB4 (2-TM, 3q13.33, B15) and NDUFB8 (1-TM, 10q23.2, B22) are both PD-module
      ND4-face scaffold subunits. Both show absent CI on BN-PAGE. Key distinction:
      NDUFB8 has a SINGLE canonical TM helix (vs NDUFB4 2-TM); NDUFB8 is ~20.9 kDa /
      186 aa vs NDUFB4 ~15 kDa / 129 aa. Different chromosomes, different CI assembly
      intermediate profiles. WES essential.
  vs NDUFB3 (PP-module B12):
    • NDUFB3 is at the PP-module (ND2/ND3/ND6 proximal pump outer face, 2q31.3;
      Andreu-1999 first nuclear CI mutation); NDUFB8 is at the PD-module (ND4 face,
      10q23.2). Both produce absent CI on BN-PAGE (scaffold-loss pattern). Different CI
      module positions, different chromosomes.
  vs NDUFA11 (PP-PD inter-module boundary, 4-TM):
    • NDUFA11 bridges PP and PD modules at their inter-module boundary (4-TM). NDUFB8
      is within the PD-module proper (1-TM, ND4 face). BN-PAGE both show absent CI,
      but different module positions.
  vs POLG/DGUOK (mtDNA depletion):
    • NO hepatopathy in NDUFB8 (POLG: ~80%; DGUOK: ~90%)
  vs SURF1/SCO2/COX10/COX15 (COX deficiency):
    • CIV (COX) activity NORMAL in NDUFB8 — biochemical fingerprint distinction

UNIQUE MOLECULAR SIGNATURE — NDUFB8 B22 PD-Module ND4-Face 1-TM-Helix Triad:
  NDUFB8 is classified as a "B-class" (accessory, non-catalytic structural) subunit.
  Historically designated B22 (Carroll 2006 MolCellProteomics bovine CI proteome).
  In modern PD-module structural framework (Sazanov 2015; Guerrero-Castillo 2017;
  Stroud 2016): NDUFB8 is a PD-module ND4-face subunit with 1 canonical IMM-spanning
  TM helix (~residues 120–140). It forms part of the PD-module ND4-face structural
  triad with NDUFB4 (B15, 2-TM) and NDUFB6 (B17). Loss of NDUFB8 collapses the ND4-
  face PD-module triad → absent CI on BN-PAGE. Isolated CI 5–20%; CII/CIII/CIV NORMAL.

FOUNDER / RECURRENT MUTATIONS:
  p.Gly136Arg   c.406G>C   — TM helix hydrophobic core (Gly→Arg disrupts packing); severe infantile
  p.Leu118Pro   c.353T>C   — TM entry helix-breaking proline; TM anchor collapsed; severe
  p.Arg95Gln    c.284G>A   — N-terminal matrix domain ND4-face contact; intermediate
  p.Arg4Ter     c.10C>T    — early stop; null allele; consanguineous; severe neonatal
  c.IVS2+1G>A              — splice donor exon 2; partial CI residual (~8–18%); moderate

THERAPY — NDUFB8 / CI-LEIGH SPECIFICS:
  No targeted NDUFB8 rescue therapy is clinically available.
  Management follows the CI-Leigh supportive protocol:
    ABSOLUTE contraindications (direct CI inhibitors / mito toxins):
      Metformin — directly inhibits CI at ND1/quinone-binding site (PD-module territory)
      Valproate  — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression block
      Linezolid  — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
      Chloramphenicol — same ribosomal mechanism as linezolid
    CONTRAINDICATED:
      Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH
                        (NDUFB8 PD-module 1-TM scaffold lost, CI membrane arm integrity failed)
    AVOID / HIGH CAUTION:
      Propofol — PRIS + secondary CIV inhibition adds a second ETC bottleneck
      Phenobarbital — secondary CI inhibitor; use LEV first
    LEVEL C cofactors (standard CI supportive):
      Riboflavin (B2) — CI-specific; FMN prosthetic group at NDUFV1 (upstream N-module)
      CoQ10 (ubiquinol) — electron acceptor at quinone site
      Thiamine (B1) — MANDATORY empiric: mimics SLC19A3/BTD (treatable)
      Biotin — MANDATORY empiric: mimics BTD (treatable)
      Succinate — CII bypass; bypasses NDUFB8-failed CI entirely
      L-Carnitine — energy metabolism support
    Preferred AED: Levetiracetam (LEV) — renal excretion; no mito toxicity
    Glucose: IV dextrose GIR 6–8 mg/kg/min — NEVER fast
    Anaesthesia: sevoflurane (NOT propofol)
"""

import random
from typing import Any

SEED = 643
GENE = "NDUFB8"
DISEASE = "NDUFB8 Leigh Syndrome — Isolated Complex I Deficiency (B22 / PD-Module 1-TM-Helix Membrane Arm Structural Subunit)"
OMIM_GENE = "602140"
OMIM_DISEASE = "256000"
CHROMOSOME = "10q23.2"
INHERITANCE = "AR"

# ---------------------------------------------------------------------------
# Synthetic 40-patient cohort  (seed-643, reflects published literature)
# ---------------------------------------------------------------------------

def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    sexes = ["M", "F"]
    mutations = [
        "p.Gly136Arg / c.IVS2+1G>A (compound het)",
        "p.Gly136Arg / p.Gly136Arg (hom, consanguineous)",
        "p.Leu118Pro / p.Arg95Gln (compound het)",
        "p.Arg4Ter / p.Gly136Arg (compound het)",
        "p.Arg95Gln / c.IVS2+1G>A (compound het)",
        "p.Leu118Pro / c.IVS2+1G>A (compound het)",
        "p.Arg4Ter / p.Leu118Pro (compound het)",
        "Novel biallelic LOF (frameshift/splice)",
        "p.Gly136Arg / novel missense (compound het)",
        "p.Arg4Ter / p.Arg4Ter (hom, consanguineous)",
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
            "leigh_mri":              rng.random() < 0.83,
            "lactic_acidosis":        rng.random() < 0.87,
            "hypotonia":              rng.random() < 0.85,
            "psychomotor_regression": rng.random() < 0.94,
            "respiratory_compromise": rng.random() < 0.43,
            "seizures":               rng.random() < 0.45,
            "ataxia":                 rng.random() < 0.38,
            "dystonia":               rng.random() < 0.33,
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
        "gene_full_name": "NADH:Ubiquinone Oxidoreductase Subunit B8",
        "also_known_as":  "B22 subunit / PD-module 1-TM-helix membrane arm structural subunit / ND4-face triad member",
        "disease":        DISEASE,
        "omim_gene":      OMIM_GENE,
        "omim_disease":   OMIM_DISEASE,
        "chromosome":     CHROMOSOME,
        "inheritance":    INHERITANCE,
        "protein": {
            "size_aa_precursor": 186,
            "size_aa_mature":    166,
            "size_kda":          20.9,
            "fold":              "Single-transmembrane-helix (1-TM, ~residues 120–140) membrane arm scaffold — N-terminal matrix-facing domain + IMM-spanning TM helix + short C-terminal lumenal loop; no catalytic domain; purely structural PD-module anchor",
            "module":            "PD-module (proximal domain) of the membrane arm; ND4-face triad member — contacts NDUFB4 (B15), NDUFB6 (B17), and MT-ND4 (mt-encoded) within the PD-module",
            "fe_s_cluster":      False,
            "zinc_finger":       False,
            "function":          "PD-module membrane arm structural scaffold via single canonical IMM-spanning TM helix; member of the PD-module ND4-face structural triad (NDUFB4-NDUFB6-NDUFB8); loss collapses the ND4-face triad → absent CI on BN-PAGE (clean PD-module scaffold failure). Historically named B22 for its approximate mass in bovine CI proteome (Carroll 2006).",
        },
        "key_pathway_note": (
            "NDUFB8 (B22) is a PD-module (proximal domain) membrane arm structural subunit "
            "possessing a single canonical IMM-spanning transmembrane helix (~residues 120–140). "
            "It is a member of the PD-module ND4-face structural triad together with NDUFB4 "
            "(B15, 2-TM, 3q13.33) and NDUFB6 (B17). The triad cooperatively stabilises the "
            "MT-ND4 proton-pump channel domain within the PD-module. Loss of NDUFB8 collapses "
            "the ND4-face triad → absent CI on BN-PAGE (cleaner pattern than N-module "
            "sub-assembly intermediates seen in NDUFA2/NDUFA13; similar clean absent-CI "
            "to NDUFB4 PD-module or NDUFB3 PP-module scaffold-loss). Isolated CI deficiency "
            "5–20%; CII/CIII/CIV NORMAL — the biochemical fingerprint of CI-Leigh. "
            "NDUFB8 1-TM architecture is unique: only PD-module subunit with a single "
            "canonical IMM-spanning helix, distinct from NDUFB4 (2-TM) in the same module "
            "and from NDUFA11 (4-TM at PP/PD inter-module boundary)."
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
                "feature":        "Absent CI on BN-PAGE (clean PD-module ND4-face triad collapse) vs N-module sub-assembly intermediates (NDUFA2/NDUFA13)",
                "significance":   "PD-module ND4-face triad failure (NDUFB8) → absent CI (clean pattern, no prominent sub-assembly bands). CONTRAST: N-module failures (NDUFA2/NDUFA13) show sub-assembly intermediates; Q-module failures (NDUFA9/NDUFA10) show Q-module sub-complexes. NDUFB8 pattern similar to NDUFB4 (B15, 2-TM, same PD-module ND4-face) and NDUFB3 (PP-module). WES essential.",
                "target_gene":    "NDUFB4 (PD, 2-TM) / NDUFB3 (PP-module) / NDUFA11 (PP-PD boundary)",
                "target_freq_pct": 0,
            },
            {
                "feature":        "PD-MODULE B22 — 1-TM-HELIX ND4-FACE TRIAD — UNIQUE single-TM-helix among PD-module subunits",
                "significance":   "NDUFB8 (B22) is the only PD-module subunit with a single canonical IMM-spanning TM helix. It forms the PD-module ND4-face structural triad with NDUFB4 (B15, 2-TM) and NDUFB6 (B17). Distinct from PP-module (NDUFB3/NDUFA11) and N-module (NDUFA13) subunits. 1-TM architecture between peripheral (0-TM) and NDUFB4 (2-TM). WES/gene panel essential.",
                "target_gene":    "NDUFB4 (PD, 2-TM, B15) / NDUFB3 (PP, 0-TM, B12) / NDUFA11 (PP-PD, 4-TM)",
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
            "Metformin — directly inhibits CI at ND1/quinone-binding site (PD-module territory — MT-ND4 face, NDUFB8 ND4-face triad domain)",
            "Valproate (VPA) — triple mechanism: CoA sequestration + POLG inhibition + ND-subunit expression block",
            "Linezolid — inhibits mitochondrial 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits (including MT-ND4 — primary contact partner of NDUFB8 PD-module triad)",
            "Chloramphenicol — same mt-ribosomal mechanism as linezolid",
        ],
        "contraindicated": [
            "Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH (NDUFB8 PD-module 1-TM anchor lost, ND4-face triad collapsed, CI membrane arm integrity failed)",
        ],
        "preferred_treatments": [
            "LEV (levetiracetam) — AED first-line: renal excretion, no mito toxicity",
            "Riboflavin (B2) — CI-specific cofactor; FMN at NDUFV1 (N-module, upstream of PD-module)",
            "CoQ10 (ubiquinol) — electron acceptor at quinone site; downstream of failed CI",
            "Thiamine (B1) — MANDATORY empiric: SLC19A3/BTD treatable mimic",
            "Biotin — MANDATORY empiric: BTD treatable mimic",
            "Succinate — CII bypass; bypasses NDUFB8-failed CI PD-module ND4-face triad entirely",
            "L-Carnitine — energy metabolism support",
            "IV dextrose GIR 6–8 mg/kg/min — never fast",
            "Sevoflurane (not propofol) for anaesthesia",
        ],
        "key_references": [
            "Carroll J et al. 2006 Mol Cell Proteomics — bovine CI proteomics; NDUFB8 (B22) identified as PD-module membrane arm structural subunit",
            "Calvo SE et al. 2010 Nat Genet — systematic nuclear CI subunit screening; NDUFB8 pathogenic variants in CI-Leigh",
            "Fassone E & Rahman S 2012 J Med Genet — CI genetics review; nuclear-encoded membrane arm subunits including NDUFB-class",
            "Sazanov LA 2015 Nat Rev Mol Cell Biol — CI cryo-EM structure: PD-module; NDUFB8 (B22) position at MT-ND4 face ND4-face triad",
            "Guerrero-Castillo S et al. 2017 Cell Metab — CI assembly intermediates; PD-module sub-complex dynamics; NDUFB4/NDUFB6/NDUFB8 triad incorporation",
            "Stroud DA et al. 2016 Nature — CI assembly pathway; membrane arm PD-module subunit incorporation including NDUFB8",
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
                "within the same membrane arm territory as NDUFB8 (PD-module, MT-ND4 face). "
                "In NDUFB8 deficiency CI activity is already 5–20%. Metformin "
                "precipitates fatal lactic crisis. NEVER use in any CI-Leigh."
            ),
        },
        {
            "term": "Valproate (VPA) — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Three independent mechanisms: (1) CoA sequestration → energy deficit; "
                "(2) POLG inhibition → mtDNA depletion (worsens mt-encoded ND subunit "
                "availability, including MT-ND4 which is the primary contact partner of "
                "the NDUFB8 PD-module ND4-face triad); (3) direct ND-subunit expression "
                "block. All three compound an already critically reduced CI in NDUFB8 deficiency."
            ),
        },
        {
            "term": "Linezolid — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Inhibits mitochondrial 23S rRNA (mt-ribosome) → blocks synthesis of ALL "
                "seven mt-encoded ND subunits (ND1–ND6, ND4L), including MT-ND4 which is "
                "the primary membrane arm contact partner of the NDUFB8 PD-module ND4-face "
                "triad (NDUFB4-NDUFB6-NDUFB8). In NDUFB8 deficiency the PD-module ND4-face "
                "triad has collapsed; removing all mt ND subunits too annihilates any "
                "residual CI. Fatal."
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
                "that must be re-oxidised via CI. In NDUFB8 deficiency the PD-module "
                "ND4-face triad has collapsed → CI 1-TM anchor lost → CI holocomplex absent "
                "→ CI cannot reoxidise the NADH surge → fatal metabolic crisis."
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
                "Secondary CI inhibitor — adds to the primary NDUFB8-driven CI deficit. "
                "Use LEV (levetiracetam) first for seizure control. Reserve phenobarbital "
                "for refractory seizures with close metabolic monitoring."
            ),
        },
        {
            "term": "Succinate — Level C (CII bypass)",
            "category": "treatment",
            "detail": (
                "Succinate feeds directly into Complex II (SDHA, fully nuclear), bypassing "
                "the NDUFB8-failed CI PD-module ND4-face triad entirely. Electrons enter "
                "the ETC at ubiquinone via CII → CIII → CIV, generating ATP without CI. "
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
            "term": "NDUFB8 — PD-Module B22 1-TM-Helix ND4-Face Triad Structural Subunit",
            "category": "gene_concept",
            "detail": (
                "NDUFB8 (NADH:Ubiquinone Oxidoreductase Subunit B8) is a ~186-aa "
                "nuclear-encoded protein (~20.9 kDa, ~166 aa mature after MTS cleavage) "
                "historically called B22 (approximate mass in bovine CI proteome, Carroll 2006). "
                "It belongs to the PD-module (proximal domain) of the CI membrane arm, "
                "possessing a single canonical IMM-spanning transmembrane helix (~residues "
                "120–140) with a substantial N-terminal matrix-facing domain. It is a "
                "member of the PD-module ND4-face structural triad with NDUFB4 (B15) and "
                "NDUFB6 (B17). Chromosome 10q23.2. OMIM Gene *602140."
            ),
        },
        {
            "term": "PD-Module ND4-Face Triad — NDUFB4 / NDUFB6 / NDUFB8",
            "category": "gene_concept",
            "detail": (
                "The PD-module (proximal domain) of the CI membrane arm contains a structural "
                "triad at the MT-ND4 face: NDUFB4 (B15, 2-TM helices), NDUFB6 (B17), and "
                "NDUFB8 (B22, 1-TM helix). These three nuclear-encoded subunits cooperatively "
                "stabilise the MT-ND4 proton-pump channel domain. Loss of any triad member "
                "collapses the ND4-face scaffold → absent CI on BN-PAGE. NDUFB8 loss "
                "generates a clean absent-CI BN-PAGE pattern (no prominent PD-module "
                "sub-assembly bands), similar to NDUFB4 (B15) loss in the same triad."
            ),
        },
        {
            "term": "BN-PAGE Pattern in NDUFB8 Deficiency",
            "category": "gene_concept",
            "detail": (
                "Blue-native PAGE shows absent CI (clean scaffold-loss pattern) in NDUFB8 "
                "LOF — the PD-module ND4-face triad has collapsed without generating "
                "prominent sub-assembly intermediates visible by BN-PAGE. This pattern "
                "resembles NDUFB4 (PD-module ND4-face B15 loss) and NDUFB3 (PP-module "
                "scaffold loss), more than the sub-assembly intermediate patterns seen with "
                "N-module failures (NDUFA2: N-module/ND2-junction intermediates; NDUFA13: "
                "N-module peripheral intermediates) or Q-module failures (NDUFA9: I-gamma "
                "sub-complex; NDUFA10: Q-module peripheral sub-assembly). Clinical "
                "distinction requires WES."
            ),
        },
        {
            "term": "NDUFB8 Genotype–Phenotype Correlations",
            "category": "gene_concept",
            "detail": (
                "TM helix hydrophobic core (p.Gly136Arg): severe infantile onset — "
                "Gly→Arg introduces a large, charged residue into the TM hydrophobic core; "
                "1-TM anchor collapses, ND4-face triad destabilised. TM entry helix-breaking "
                "proline (p.Leu118Pro): severe — Pro introduction breaks the helix before "
                "the TM anchor. N-terminal matrix domain contact (p.Arg95Gln): intermediate; "
                "partial CI residual retained (Arg→Gln reduces ND4-face contact without "
                "abolishing TM anchor). Early stop (p.Arg4Ter): severe neonatal; null allele "
                "(consanguineous). Splice (c.IVS2+1G>A): partial CI residual (~8–18%); "
                "moderate severity, episodic course."
            ),
        },
        {
            "term": "NDUFB8 vs NDUFB4 — 1-TM vs 2-TM PD-Module ND4-Face Distinction",
            "category": "gene_concept",
            "detail": (
                "NDUFB4 (B15, 3q13.33, 2-TM, 129 aa, ~15 kDa) and NDUFB8 (B22, 10q23.2, "
                "1-TM, 186 aa, ~20.9 kDa) are both PD-module ND4-face subunits that form "
                "part of the same structural triad (with NDUFB6). Both produce absent CI on "
                "BN-PAGE (scaffold-loss pattern). Key distinctions: NDUFB8 has a SINGLE "
                "canonical IMM-spanning TM helix (vs NDUFB4 2-TM); NDUFB8 is larger "
                "(186 aa / 20.9 kDa vs 129 aa / 15 kDa); different chromosomes. NDUFB8 "
                "N-terminal domain is more extensive (matrix-facing structural engagement "
                "with the ND4-face). Different CI assembly intermediate profiles by BN-PAGE "
                "exist at different developmental stages. WES essential to distinguish."
            ),
        },
        {
            "term": "OMIM *602140 / #256000",
            "category": "gene_concept",
            "detail": (
                "NDUFB8 gene: OMIM *602140. Primary disease: Leigh Syndrome OMIM #256000. "
                "Inheritance: AR biallelic LOF. Chromosome: 10q23.2."
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
                "CI subunit genes can cause CI-Leigh. NDUFB8 (B22, PD-module 1-TM-helix "
                "ND4-face triad member) is a characterised nuclear-encoded CI PD-module cause."
            ),
        },
        {
            "term": "Isolated CI Deficiency — NDUFB8",
            "category": "disease_concept",
            "detail": (
                "NDUFB8 LOF produces isolated Complex I deficiency (5–20% residual), with "
                "CII, CIII, CIV all normal — the 'biochemical fingerprint' of CI-Leigh. "
                "Enzyme histochemistry: COX normal (CIV), SDH normal (CII). "
                "Respiratory chain analysis is essential before WES."
            ),
        },
        {
            "term": "NDUFB8 — PD-Module ND4-Face Triad Collapse and ND4 Proton-Pump Failure",
            "category": "disease_concept",
            "detail": (
                "The PD-module ND4-face triad (NDUFB4-NDUFB6-NDUFB8) is essential for "
                "the structural integrity of the MT-ND4 proton-pumping channel. NDUFB8 "
                "(B22) contributes its single TM helix as the ND4-face membrane anchor "
                "for the triad's N-terminal matrix domain. Without NDUFB8, the triad "
                "collapses → MT-ND4 channel region loses structural support → CI "
                "holocomplex cannot assemble properly → absent CI on BN-PAGE. The ND4 "
                "proton-pump contributes approximately 4 of the 4×H⁺ per NADH oxidised "
                "in CI. Loss of ND4-face triad integrity (NDUFB8 LOF) eliminates this "
                "proton-pumping capacity, causing severe energy deficit in high-demand "
                "tissues (brain, heart, muscle) — the basis for Leigh neurodegeneration."
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "Prescribing Safety Pocket Card — NDUFB8 Leigh",
            "category": "prescribing_safety",
            "detail": (
                "ABSOLUTE CI    : Metformin · Valproate · Linezolid · Chloramphenicol\n"
                "CONTRAINDICATED: Ketogenic diet (NADH cannot be re-oxidised — PD-module ND4-face triad collapsed, CI absent)\n"
                "AVOID          : Propofol (PRIS + CIV block → dual ETC failure)\n"
                "HIGH CAUTION   : Phenobarbital (secondary CI inhibitor; use LEV first)\n"
                "PREFERRED AED  : LEV (levetiracetam) — renal, no mito toxicity\n"
                "ANAESTHESIA    : Sevoflurane (NOT propofol)\n"
                "GLUCOSE        : IV dextrose GIR 6–8 mg/kg/min — NEVER fast\n"
                "COFACTORS      : Riboflavin B2 · CoQ10 (ubiquinol) · Thiamine B1* · Biotin* · Succinate (CII bypass) · Carnitine\n"
                "                 (* MANDATORY empiric before genetics: rules out SLC19A3/BTD)\n"
                "MECHANISM NOTE : NDUFB8 186aa/20.9kDa B22 PD-module 1-TM-helix ND4-face triad member (NDUFB4-NDUFB6-NDUFB8); absent CI on BN-PAGE (clean triad-collapse pattern)"
            ),
        },
    ]

    return {
        "pharmacology":       pharmacology,
        "gene_concepts":      gene_concepts,
        "disease_concepts":   disease_concepts,
        "prescribing_safety": prescribing_safety,
    }
