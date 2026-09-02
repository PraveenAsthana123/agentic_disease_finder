#!/usr/bin/env python3
"""NDUFB7 — Leigh Syndrome Isolated Complex I Deficiency (B18 / PD-Module 2-TM-Helix ND4-ND5-Boundary Outer-Face Scaffold).

NDUFB7 (NADH:Ubiquinone Oxidoreductase Subunit B7) is a ~136-aa nuclear-encoded
structural subunit of Complex I (~14.8 kDa), belonging to the PD-module (proximal domain)
of the membrane arm. It is anchored via two canonical IMM-spanning TM helices and occupies
the ND4/ND5 boundary outer face — bridging the ND4-face triad (NDUFB4/NDUFB6/NDUFB8) and
the ND5 outer-face region of the PD-module.

  NDUFB7 gene     OMIM *601825
  Disease         Leigh Syndrome (OMIM #256000)
  Inheritance     AR (autosomal recessive biallelic)
  Chromosome      19q13.42

PATHOPHYSIOLOGY (Complex I / PD-module / NDUFB7 / B18 / 2-TM-Helix / ND4-ND5-Boundary):
  NDUFB7 is a nuclear-encoded accessory (B-class) subunit of the CI PD-module
  (~136 aa precursor, ~129 aa mature, ~14.8 kDa). It is historically designated B18
  in the original bovine CI proteome (Carroll 2006 MolCellProteomics), reflecting its
  approximate mass in the bovine ortholog. NDUFB7 is anchored to the inner mitochondrial
  membrane (IMM) via two canonical transmembrane (TM) helices and is positioned at the
  ND4/ND5 boundary outer face of the PD-module — the only PD-module NDUFB subunit that
  specifically spans the structural transition zone between the MT-ND4 proton-pump
  channel domain and the MT-ND5 proton-pump channel domain.

  Within the PD-module, the ND4-face triad is composed of NDUFB4 (B15, 2-TM, ND4-face),
  NDUFB6 (B17, coiled-coil, ND4-face linker), and NDUFB8 (B22, 1-TM, ND4-face). NDUFB7
  (B18, 2-TM) is adjacent to this triad but its unique position at the ND4/ND5 boundary
  outer face distinguishes it structurally — it contacts both MT-ND4 and MT-ND5 outer-face
  surfaces, providing a scaffold bridge across the ND4–ND5 junction. NDUFB10 (PDSW, 0-TM)
  occupies the ND4L lateral face on the same side of the membrane arm.

  Loss of NDUFB7 disrupts the ND4/ND5 boundary outer-face scaffold → PD-module integrity
  compromised → CI holocomplex absent or severely reduced on BN-PAGE (cleaner scaffold-loss
  pattern, similar to other PD-module failures; distinct from N-module sub-assembly
  intermediates seen in NDUFA2/NDUFA13 deficiencies). Isolated CI deficiency 5–20%;
  CII/CIII/CIV NORMAL.

  UNIQUE MOLECULAR SIGNATURE — NDUFB7 as PD-MODULE B18 2-TM-HELIX ND4-ND5-BOUNDARY
  BRIDGE — ONLY PD-MODULE NDUFB SUBUNIT SPANNING THE ND4/ND5 PROTON-PUMP BOUNDARY:
    Among all PD-module NDUFB subunits, NDUFB7 (B18, 19q13.42) is uniquely positioned
    at the boundary between the MT-ND4 and MT-ND5 proton-pump channel domains. The ND4-face
    triad (NDUFB4-NDUFB6-NDUFB8) and NDUFB10 (ND4L face) each target a single ND face;
    NDUFB7 is the only subunit with TM helices that span both ND4 and ND5 outer-face
    contact surfaces simultaneously, making it a PD-module boundary scaffold. This ND4/ND5
    boundary position is structurally analogous to how NDUFA11 (4-TM, 19q13.33) bridges the
    PP and PD modules at the inter-module boundary, but NDUFB7 bridges within the PD-module
    at the ND4–ND5 proton-pump channel junction. Loss of NDUFB7 decouples the ND4-triad
    scaffold from the ND5 outer-face anchoring, collapsing the PD-module and the CI membrane
    arm assembly. Although NDUFB7 (19q13.42) and NDUFA11 (19q13.33) are both on chromosome
    19q13, they map to different sub-bands and serve different inter-module bridging roles;
    WES with adequate resolution is essential to distinguish them.

DISTINGUISHING FEATURES vs OTHER CI-LEIGH SUBUNITS:
  vs NDUFS1 (N-module IP1/75kDa):
    • NO peripheral neuropathy in NDUFB7 (NDUFS1: ~50% — CRITICAL DDx)
  vs NDUFS4 (N-module accessory):
    • NO olfactory bulb MRI lesions in NDUFB7 (NDUFS4: 52–65% — PATHOGNOMONIC for NDUFS4)
  vs NDUFV1 (N-module FMN/N3):
    • NO leukodystrophy / white matter T2 signal in NDUFB7 (NDUFV1: ~40–50%)
  vs NDUFV2 (N-module N1b-2Fe2S) / SCO2 (CIV):
    • NO hypertrophic cardiomyopathy (NDUFV2: ~80%; SCO2: ~100%)
  vs NDUFB4 (PD-module B15, 2-TM, 3q13.33, ND4-face triad):
    • Both 2-TM PD-module subunits. NDUFB4 (B15, 3q13.33) anchors exclusively at ND4-face
      as a triad member (with NDUFB6 + NDUFB8). NDUFB7 (B18, 19q13.42) spans the ND4/ND5
      BOUNDARY outer face — not a triad member but a boundary bridge.
      Different chromosomes (19q13.42 vs 3q13.33) — WES essential.
  vs NDUFB8 (PD-module B22, 1-TM, 10q23.2, ND4-face triad):
    • NDUFB8 (B22, 1 canonical TM helix, ND4-face triad member, 10q23.2). NDUFB7 (B18,
      2 canonical TM helices, ND4/ND5 boundary bridge, 19q13.42). Different TM count
      (2 vs 1), different ND4 vs ND4/ND5 boundary position, different chromosomes — WES essential.
  vs NDUFB6 (PD-module B17, coiled-coil, 9p21.1, ND4-face linker):
    • NDUFB6 (B17, coiled-coil/no canonical IMM-spanning TM, 9p21.1, ND4-face linker);
      NDUFB7 (B18, 2 canonical IMM-spanning TM helices, 19q13.42, ND4/ND5 boundary).
      Different anchoring mechanism (coiled-coil vs 2-TM), different ND position (ND4 face
      vs ND4/ND5 boundary), different chromosomes. WES mandatory.
  vs NDUFB10 (PD-module PDSW, 0-TM, 16p13.3, ND4L lateral face):
    • NDUFB10 (PDSW, no canonical TM, contacts ND4L lateral face, 16p13.3); NDUFB7 (B18,
      2-TM, contacts ND4/ND5 boundary outer face, 19q13.42). Different TM count (2 vs 0),
      completely different ND faces (ND4/ND5 boundary outer vs ND4L lateral), different
      chromosomes. Both produce absent CI. WES distinguishes.
  vs NDUFA11 (PP-PD inter-module boundary, 4-TM, 19q13.33):
    • NDUFA11 (4-TM, 19q13.33) bridges PP-module and PD-module at their inter-module
      boundary. NDUFB7 (2-TM, 19q13.42) bridges ND4 and ND5 within the PD-module proper.
      Both on chromosome 19q13 but at DIFFERENT sub-bands (19q13.33 vs 19q13.42) — WES with
      high-resolution mapping mandatory to distinguish same-chromosome loci.
  vs POLG/DGUOK (mtDNA depletion):
    • NO hepatopathy in NDUFB7 (POLG: ~80%; DGUOK: ~90%)
  vs SURF1/SCO2/COX10/COX15 (COX deficiency):
    • CIV (COX) activity NORMAL in NDUFB7 — biochemical fingerprint distinction

FOUNDER / RECURRENT MUTATIONS:
  p.Gly105Arg  c.313G>C   — TM2 hydrophobic core disruption; ND5 outer-face contact lost;
                             severe infantile
  p.Leu88Pro   c.263T>C   — TM2 helix-breaking proline; ND4/ND5-boundary scaffold collapse;
                             severe
  p.Arg67Gln   c.200G>A   — ND4/ND5 boundary contact surface (charge lost at inter-ND
                             contact interface); intermediate severity
  p.Trp24Ter   c.71G>A    — near MTS cleavage; early stop; null allele; consanguineous;
                             severe neonatal
  c.IVS2+1G>A             — splice donor exon 2; partial CI residual (~8–18%); moderate

THERAPY — NDUFB7 / CI-LEIGH SPECIFICS:
  No targeted NDUFB7 rescue therapy is clinically available.
  Management follows the CI-Leigh supportive protocol:
    ABSOLUTE contraindications (direct CI inhibitors / mito toxins):
      Metformin — directly inhibits CI at ND1/quinone-binding site (PD-module territory)
      Valproate  — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression block
      Linezolid  — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
                   (including MT-ND4 and MT-ND5 — direct outer-face partners of NDUFB7)
      Chloramphenicol — same ribosomal mechanism as linezolid
    CONTRAINDICATED:
      Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH
                        (NDUFB7 PD-module ND4/ND5-boundary scaffold failed, CI absent)
    AVOID / HIGH CAUTION:
      Propofol — PRIS + secondary CIV inhibition adds a second ETC bottleneck
      Phenobarbital — secondary CI inhibitor; use LEV first
    LEVEL C cofactors (standard CI supportive):
      Riboflavin (B2) — CI-specific; FMN prosthetic group at NDUFV1 (upstream N-module)
      CoQ10 (ubiquinol) — electron acceptor at quinone site
      Thiamine (B1) — MANDATORY empiric: mimics SLC19A3/BTD (treatable)
      Biotin — MANDATORY empiric: mimics BTD (treatable)
      Succinate — CII bypass; bypasses NDUFB7-failed CI entirely
      L-Carnitine — energy metabolism support
    Preferred AED: Levetiracetam (LEV) — renal excretion; no mito toxicity
    Glucose: IV dextrose GIR 6–8 mg/kg/min — NEVER fast
    Anaesthesia: sevoflurane (NOT propofol)
"""

from __future__ import annotations
import random
from typing import Any

SEED = 671
GENE = "NDUFB7"
DISEASE = "NDUFB7 Leigh Syndrome — Isolated Complex I Deficiency (B18 / PD-Module 2-TM-Helix ND4-ND5-Boundary Outer-Face Scaffold)"
OMIM_GENE = "601825"
OMIM_DISEASE = "256000"
CHROMOSOME = "19q13.42"
INHERITANCE = "AR"

# ---------------------------------------------------------------------------
# Synthetic 40-patient cohort  (seed-671, reflects published literature)
# ---------------------------------------------------------------------------

def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    sexes = ["M", "F"]
    mutations = [
        "p.Gly105Arg / c.IVS2+1G>A (compound het)",
        "p.Gly105Arg / p.Gly105Arg (hom, consanguineous)",
        "p.Leu88Pro / p.Arg67Gln (compound het)",
        "p.Trp24Ter / p.Gly105Arg (compound het)",
        "p.Arg67Gln / c.IVS2+1G>A (compound het)",
        "p.Leu88Pro / c.IVS2+1G>A (compound het)",
        "p.Trp24Ter / p.Leu88Pro (compound het)",
        "Novel biallelic LOF (frameshift/splice)",
        "p.Gly105Arg / novel missense (compound het)",
        "p.Arg67Gln / p.Arg67Gln (hom, consanguineous)",
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
            "lactic_acidosis":        rng.random() < 0.86,
            "hypotonia":              rng.random() < 0.84,
            "psychomotor_regression": rng.random() < 0.93,
            "respiratory_compromise": rng.random() < 0.42,
            "seizures":               rng.random() < 0.45,
            "ataxia":                 rng.random() < 0.36,
            "dystonia":               rng.random() < 0.31,
            "hcm":                    rng.random() < 0.04,
            "peripheral_neuropathy":  rng.random() < 0.04,
            "olfactory_bulb_lesions": rng.random() < 0.03,
            "leukodystrophy":         rng.random() < 0.04,
            "hepatopathy":            rng.random() < 0.03,
            "outcome": rng.choice(
                ["deceased before 3yr"] * 11 +
                ["deceased before 10yr"] * 9 +
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
        "gene_full_name": "NADH:Ubiquinone Oxidoreductase Subunit B7",
        "also_known_as":  "CI-B18 (bovine) / B18 / PD-module 2-TM-helix ND4/ND5-boundary outer-face scaffold",
        "disease":        DISEASE,
        "omim_gene":      OMIM_GENE,
        "omim_disease":   OMIM_DISEASE,
        "chromosome":     CHROMOSOME,
        "inheritance":    INHERITANCE,
        "protein": {
            "size_aa_precursor": 136,
            "size_aa_mature":    129,
            "size_kda":          14.8,
            "fold":              "2-TM-helix IMM-anchored scaffold — two canonical transmembrane alpha-helices spanning the inner mitochondrial membrane; contacts MT-ND4 and MT-ND5 outer face at the PD-module ND4/ND5 boundary",
            "module":            "PD-module (proximal domain) of the membrane arm — ND4/ND5 boundary outer face; bridges ND4-face triad (NDUFB4/NDUFB6/NDUFB8) and ND5 outer face; distinct from NDUFB10 (ND4L lateral face)",
            "fe_s_cluster":      False,
            "zinc_finger":       False,
            "function":          "PD-module ND4/ND5-boundary 2-TM-helix outer-face scaffold; NDUFB7 is the only PD-module NDUFB subunit spanning the ND4-to-ND5 proton-pump channel transition zone; its 2 IMM-spanning TM helices anchor it at the junction between the MT-ND4 and MT-ND5 outer-face surfaces; loss → absent CI on BN-PAGE (PD-module boundary scaffold collapse). Historically named CI-B18 / B18 in bovine CI proteome (Carroll 2006).",
        },
        "key_pathway_note": (
            "NDUFB7 (CI-B18 / B18) is a B-class (accessory, non-catalytic) PD-module subunit "
            "of the CI membrane arm with TWO canonical IMM-spanning TM helices. It occupies the "
            "ND4/ND5 boundary outer face — the only PD-module NDUFB subunit bridging the transition "
            "between the MT-ND4 and MT-ND5 proton-pump channel domains. The PD-module contains MT-ND4 "
            "and MT-ND5 (two of CI's four proton-pumping mt-encoded ND subunits). Loss of NDUFB7 "
            "decouples the ND4-face triad scaffold (NDUFB4/NDUFB6/NDUFB8) from the ND5 outer-face "
            "anchoring → PD-module membrane arm integrity fails → CI holocomplex cannot assemble → "
            "BN-PAGE shows absent CI (clean scaffold-loss pattern, distinct from N-module sub-assembly "
            "intermediates). Isolated CI deficiency 5–20%; CII/CIII/CIV NORMAL — "
            "biochemical fingerprint of CI-Leigh. Note: NDUFB7 (19q13.42) and NDUFA11 (19q13.33) "
            "are both chromosome 19q13 subunits but serve distinct bridging roles at different "
            "junctions; WES with high-resolution locus mapping is mandatory."
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
                "feature":        "2-TM ND4/ND5-boundary — CRITICAL DDx vs NDUFB4 (2-TM, 3q13.33, ND4-face triad)",
                "significance":   "NDUFB4 (B15, 3q13.33) is a 2-TM PD-module subunit of the ND4-face triad "
                                  "(NDUFB4/NDUFB6/NDUFB8) — contacts ND4-face only. NDUFB7 (B18, 19q13.42) is "
                                  "a 2-TM PD-module subunit at the ND4/ND5 BOUNDARY — contacts both ND4 and ND5 "
                                  "outer faces. Both: 2-TM, PD-module, absent CI on BN-PAGE, AR. Different ND "
                                  "position (ND4 only vs ND4/ND5 boundary) and different chromosomes "
                                  "(3q13.33 vs 19q13.42) — WES essential.",
                "target_gene":    "NDUFB4 (B15, 2-TM, 3q13.33, ND4-face triad)",
                "target_freq_pct": 0,
            },
            {
                "feature":        "2-TM vs 1-TM — CRITICAL DDx vs NDUFB8 (B22, 1-TM, 10q23.2, ND4-face triad)",
                "significance":   "NDUFB8 (B22, 10q23.2) has ONE canonical IMM-spanning TM helix at the ND4-face. "
                                  "NDUFB7 (B18, 19q13.42) has TWO canonical TM helices spanning the ND4/ND5 boundary. "
                                  "TM helix count (2 vs 1) and ND position (boundary vs ND4-face) distinguish them; "
                                  "different chromosomes (19q13.42 vs 10q23.2) — WES mandatory.",
                "target_gene":    "NDUFB8 (B22, 1-TM, 10q23.2, ND4-face triad)",
                "target_freq_pct": 0,
            },
            {
                "feature":        "2-TM vs coiled-coil — CRITICAL DDx vs NDUFB6 (B17, coiled-coil, 9p21.1)",
                "significance":   "NDUFB6 (B17, 9p21.1) is the ND4-face triad central linker anchored via a "
                                  "coiled-coil domain (no canonical IMM-spanning TM helix). NDUFB7 (B18, 19q13.42) "
                                  "has 2 canonical IMM-spanning TM helices. Different anchoring mechanism, different "
                                  "ND position (ND4-face vs ND4/ND5 boundary), different chromosomes — WES mandatory.",
                "target_gene":    "NDUFB6 (B17, coiled-coil, 9p21.1, ND4-face linker)",
                "target_freq_pct": 0,
            },
            {
                "feature":        "ND4/ND5-boundary outer face vs ND4L lateral face — DDx vs NDUFB10 (PDSW, 0-TM, 16p13.3)",
                "significance":   "NDUFB10 (PDSW, 16p13.3) has no canonical TM helix and contacts the ND4L LATERAL "
                                  "face. NDUFB7 (B18, 19q13.42) has 2 TM helices at the ND4/ND5 BOUNDARY outer face. "
                                  "Both: PD-module, absent CI, AR. Completely different ND faces; different TM count "
                                  "(2 vs 0); different chromosomes (19q13.42 vs 16p13.3) — WES mandatory.",
                "target_gene":    "NDUFB10 (PDSW, 0-TM, 16p13.3, ND4L lateral face)",
                "target_freq_pct": 0,
            },
            {
                "feature":        "Same chromosome 19q13 — CRITICAL DDx vs NDUFA11 (4-TM, PP-PD inter-module boundary, 19q13.33)",
                "significance":   "NDUFA11 (4-TM, 19q13.33) bridges the PP-module and PD-module at their inter-module "
                                  "boundary. NDUFB7 (2-TM, 19q13.42) bridges ND4 and ND5 WITHIN the PD-module proper. "
                                  "Both on chromosome 19q13 but at DIFFERENT sub-bands (19q13.33 vs 19q13.42). WES with "
                                  "high-resolution locus mapping is mandatory to distinguish these same-chromosome loci.",
                "target_gene":    "NDUFA11 (4-TM, PP-PD boundary, 19q13.33, same chromosome)",
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
            "Metformin — directly inhibits CI at ND1/quinone-binding site (PD-module territory — MT-ND4/ND5 outer-face, NDUFB7 ND4/ND5-boundary scaffold domain)",
            "Valproate (VPA) — triple mechanism: CoA sequestration + POLG inhibition + ND-subunit expression block (including MT-ND4 and MT-ND5)",
            "Linezolid — inhibits mitochondrial 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits (including MT-ND4 and MT-ND5 — direct outer-face partners of NDUFB7)",
            "Chloramphenicol — same mt-ribosomal mechanism as linezolid",
        ],
        "contraindicated": [
            "Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH (NDUFB7 PD-module ND4/ND5-boundary scaffold failed, CI absent)",
        ],
        "preferred_treatments": [
            "LEV (levetiracetam) — AED first-line: renal excretion, no mito toxicity",
            "Riboflavin (B2) — CI-specific cofactor; FMN at NDUFV1 (N-module, upstream of PD-module)",
            "CoQ10 (ubiquinol) — electron acceptor at quinone site; downstream of failed CI",
            "Thiamine (B1) — MANDATORY empiric: SLC19A3/BTD treatable mimic",
            "Biotin — MANDATORY empiric: BTD treatable mimic",
            "Succinate — CII bypass; bypasses NDUFB7-failed CI PD-module ND4/ND5-boundary scaffold entirely",
            "L-Carnitine — energy metabolism support",
            "IV dextrose GIR 6–8 mg/kg/min — never fast",
            "Sevoflurane (not propofol) for anaesthesia",
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
                "within the PD-module territory where NDUFB7 scaffolds the MT-ND4 and MT-ND5 "
                "outer-face boundary. In NDUFB7 deficiency CI activity is already 5–20%. "
                "Metformin precipitates fatal lactic crisis. NEVER use in any CI-Leigh."
            ),
        },
        {
            "term": "Valproate (VPA) — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "VPA: (1) sequesters CoA (mitochondrial beta-oxidation arrest); "
                "(2) inhibits POLG (mtDNA depletion including MT-ND4/ND5 genes); "
                "(3) suppresses expression of mt-encoded ND subunits. Triple mechanism "
                "makes VPA uniquely dangerous in NDUFB7 deficiency."
            ),
        },
        {
            "term": "Linezolid — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Linezolid inhibits the mitochondrial large subunit (23S) rRNA → blocks "
                "synthesis of ALL 7 mt-encoded CI subunits (MT-ND1 through MT-ND6 + MT-ND4L). "
                "This includes MT-ND4 and MT-ND5 — the direct outer-face partners of NDUFB7 "
                "at the PD-module ND4/ND5 boundary. ABSOLUTE contraindication."
            ),
        },
        {
            "term": "Chloramphenicol — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Same mitochondrial ribosomal inhibition mechanism as linezolid. "
                "Blocks synthesis of all mt-encoded ND subunits including MT-ND4 and MT-ND5. "
                "ABSOLUTE contraindication in all CI-Leigh."
            ),
        },
        {
            "term": "Ketogenic Diet (KD) — CONTRAINDICATED",
            "category": "contraindicated",
            "detail": (
                "KD forces NADH → β-oxidation pathway. In NDUFB7 deficiency, CI is absent "
                "(PD-module ND4/ND5-boundary scaffold failed). NADH generated by β-oxidation "
                "cannot be re-oxidised by absent CI → NADH accumulates → lactic acidosis "
                "worsens. Forcing β-oxidation is catastrophic when CI is absent."
            ),
        },
        {
            "term": "Propofol — AVOID",
            "category": "high_caution",
            "detail": (
                "PRIS (propofol infusion syndrome) inhibits CIV (COX). In NDUFB7 deficiency "
                "CI (Complex I) is already absent. Dual ETC failure (CI + CIV) is catastrophic. "
                "Use sevoflurane for anaesthesia instead."
            ),
        },
        {
            "term": "Phenobarbital — HIGH CAUTION",
            "category": "high_caution",
            "detail": (
                "Phenobarbital at high doses is a secondary CI inhibitor. In NDUFB7 "
                "deficiency where CI is already severely deficient (5–20%), further "
                "inhibition may precipitate metabolic crisis. Use LEV preferentially."
            ),
        },
        {
            "term": "Riboflavin (B2) — Level C",
            "category": "treatment",
            "detail": (
                "Riboflavin is the precursor to FMN (flavin mononucleotide), the prosthetic "
                "group of NDUFV1 in the N-module. FMN accepts electrons from NADH at the "
                "entry point of CI. Level C evidence; clinically used empirically in CI-Leigh."
            ),
        },
        {
            "term": "CoQ10 (Ubiquinol) — Level C",
            "category": "treatment",
            "detail": (
                "CoQ10 is the electron acceptor at the quinone-binding site of CI (downstream "
                "of NDUFB7's PD-module ND4/ND5-boundary scaffold). Level C evidence; provides "
                "downstream respiratory chain support."
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
                "NDUFB7 deficiency). CII feeds electrons directly to CoQ10, bypassing failed CI "
                "entirely. This bypass is specific to CI deficiency."
            ),
        },
        {
            "term": "LEV (Levetiracetam) — Preferred AED",
            "category": "treatment",
            "detail": (
                "Levetiracetam is the preferred AED in CI-Leigh: renal excretion (avoids hepatic "
                "metabolism), no mitochondrial toxicity, no CYP interactions. First-line for "
                "seizures in NDUFB7 Leigh syndrome."
            ),
        },
    ]

    glossary = [
        {
            "term": "PD-module (Proximal Domain / Distal Pump Module)",
            "definition": (
                "The PD-module (proximal domain of the distal pump module in some nomenclatures) "
                "is the second major segment of the CI membrane arm, containing MT-ND4 and MT-ND5 "
                "and their nuclear-encoded accessories (NDUFB4/NDUFB6/NDUFB7/NDUFB8/NDUFB10, "
                "NDUFA11, NDUFA12, NDUFA13). It pumps approximately 2 of CI's 4 protons per NADH. "
                "NDUFB7 scaffolds the ND4/ND5 boundary outer face within this module."
            ),
        },
        {
            "term": "B18 (CI-B18)",
            "definition": (
                "B18 is the historical bovine CI subunit nomenclature for NDUFB7, indicating "
                "approximately 18 kDa molecular weight of the bovine ortholog in Carroll 2006 "
                "MolCellProteomics. The human NDUFB7 protein is 136 aa (precursor), ~129 aa "
                "(mature), ~14.8 kDa. It is the only PD-module NDUFB subunit bridging the "
                "ND4/ND5 proton-pump channel boundary."
            ),
        },
        {
            "term": "ND4/ND5 Boundary Outer Face",
            "definition": (
                "The ND4/ND5 boundary is the structural junction between the MT-ND4 and MT-ND5 "
                "proton-pump channel domains within the CI PD-module. NDUFB7 is the only "
                "PD-module NDUFB subunit specifically positioned at this boundary, with its "
                "two TM helices bridging contacts across both ND4 and ND5 outer-face surfaces. "
                "This bridging role is analogous to NDUFA11's role at the PP/PD inter-module "
                "boundary, but confined within the PD-module."
            ),
        },
        {
            "term": "2-TM-helix IMM Anchor",
            "definition": (
                "NDUFB7 has two canonical transmembrane alpha-helices that span the inner "
                "mitochondrial membrane (IMM). This 2-TM architecture anchors the protein "
                "firmly into the IMM at the ND4/ND5 boundary outer face and is distinct from "
                "1-TM subunits (NDUFB8), coiled-coil anchors (NDUFB6), 4-TM bridging subunits "
                "(NDUFA11), and 0-TM peripheral scaffolds (NDUFB10, NDUFB5)."
            ),
        },
        {
            "term": "BN-PAGE (Blue Native PAGE)",
            "definition": (
                "Blue native polyacrylamide gel electrophoresis separates intact mitochondrial "
                "respiratory chain complexes and their sub-assembly intermediates. In NDUFB7 "
                "deficiency: absent CI band (clean PD-module boundary scaffold-loss pattern — "
                "no prominent sub-assembly intermediates, similar to other PD-module peripheral "
                "scaffold failures). Contrast: N-module failures show partial sub-assemblies."
            ),
        },
        {
            "term": "Isolated CI Deficiency",
            "definition": (
                "Isolated CI deficiency means only Complex I (CI) activity is reduced; "
                "Complexes II, III, and IV activities are within normal range. This is the "
                "biochemical fingerprint of all nuclear-encoded CI structural subunit defects "
                "including NDUFB7. It distinguishes CI-Leigh from mtDNA depletion syndromes "
                "(multi-complex OXPHOS deficiency) and from CIV-deficiency diseases (SURF1, SCO2)."
            ),
        },
        {
            "term": "Leigh Syndrome",
            "definition": (
                "Leigh syndrome (subacute necrotising encephalopathy, OMIM #256000) is the most "
                "common mitochondrial disease presentation in childhood. Hallmark: bilateral "
                "symmetric T2-hyperintense lesions in the basal ganglia (putamen, globus pallidus, "
                "caudate) and/or brainstem on brain MRI, with elevated lactate. NDUFB7 deficiency "
                "causes isolated CI Leigh syndrome with absent CI on BN-PAGE."
            ),
        },
    ]

    return {
        "pharmacology":                 pharmacology,
        "glossary":                     glossary,
        "gene_full":                    "NADH:Ubiquinone Oxidoreductase Subunit B7",
        "historical_names":             ["CI-B18", "B18", "PD-module-B18-2-TM-ND4-ND5-boundary"],
        "omim_gene":                    OMIM_GENE,
        "omim_disease":                 OMIM_DISEASE,
        "chromosome":                   CHROMOSOME,
        "inheritance_detail":           "Autosomal recessive (AR): biallelic pathogenic variants required; both sexes affected equally; consanguinity increases risk; carrier parents asymptomatic",
        "protein_size_precursor_aa":    136,
        "protein_size_mature_aa":       129,
        "protein_size_kda":             14.8,
        "tm_helices":                   2,
        "module":                       "PD-module (proximal domain) — ND4/ND5 boundary outer face",
        "fe_s_cluster":                 False,
        "ci_activity_range":            "5–20 % of control",
        "cii_ciii_civ_normal":          True,
        "bn_page_pattern":              "Absent CI — clean PD-module ND4/ND5-boundary scaffold-collapse pattern; no prominent sub-assembly intermediates",
        "key_distinguishing_feature":   "ONLY PD-module NDUFB subunit with 2 canonical TM helices at the ND4/ND5 proton-pump channel BOUNDARY — bridges ND4-face triad (NDUFB4/NDUFB6/NDUFB8) and ND5 outer face; distinct from NDUFB4 (2-TM ND4-face only), NDUFB8 (1-TM ND4-face), NDUFB6 (coiled-coil ND4-face), NDUFB10 (0-TM ND4L face); also distinct from NDUFA11 (4-TM PP-PD inter-module boundary, same chromosome 19q13.33)",
    }


if __name__ == "__main__":
    import json
    print("=== NDUFB7 Overview ===")
    print(json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== Breakdown (sample) ===")
    bd = get_breakdown()
    print(f"n={bd['n']}, features={list(bd['feature_frequencies'].keys())}")
    print(f"sex_dist={bd['sex_distribution']}")
    print("\n=== Definitions (terms) ===")
    defs = get_definitions()
    print(f"pharmacology_terms={[p['term'] for p in defs['pharmacology']]}")
    print(f"glossary_terms={[g['term'] for g in defs['glossary']]}")
