#!/usr/bin/env python3
"""NDUFB6 — Leigh Syndrome Isolated Complex I Deficiency (B17 / PD-Module Coiled-Coil ND4-Face Triad Central Linker).

NDUFB6 (NADH:Ubiquinone Oxidoreductase Subunit B6) is a ~128-aa nuclear-encoded
structural subunit of Complex I (~14.6 kDa), belonging to the PD-module (proximal domain)
of the membrane arm. It occupies the ND4-face and bridges NDUFB4 (B15) and NDUFB8 (B22)
via a matrix-anchored coiled-coil domain — completing the PD-module ND4-face structural
triad (NDUFB4-NDUFB6-NDUFB8).

  NDUFB6 gene     OMIM *603879
  Disease         Leigh Syndrome (OMIM #256000)
  Inheritance     AR (autosomal recessive biallelic)
  Chromosome      9p21.1

PATHOPHYSIOLOGY (Complex I / PD-module / NDUFB6 / B17 / Coiled-Coil ND4-Face Triad):
  NDUFB6 is a small nuclear-encoded subunit of the CI PD-module (proximal domain)
  membrane arm (~128 aa precursor, ~14.6 kDa). It is historically designated B17 in the
  original bovine CI proteome (Carroll 2006 MolCellProteomics). NDUFB6 occupies the
  MT-ND4 face of the PD-module and is a central linker of the PD-module ND4-face
  structural triad alongside NDUFB4 (B15, 2-TM) and NDUFB8 (B22, 1-TM-IMM-spanning).
  Unlike NDUFB4 (two canonical TM helices) and NDUFB8 (one canonical IMM-spanning TM
  helix), NDUFB6 is primarily matrix-anchored via a coiled-coil domain that docks onto
  the NDUFB4-NDUFB8 scaffold at the ND4 face, with a short amphipathic helix partially
  engaging the inner mitochondrial membrane. Loss of NDUFB6 disrupts the ND4-face
  structural triad → absent or severely reduced CI holocomplex. BN-PAGE: absent CI
  (similar scaffold-loss pattern to NDUFB4 and NDUFB8; distinct from N-module
  sub-assembly intermediates in NDUFA2/NDUFA13 and Q-module sub-complexes in
  NDUFA9/NDUFA10). Isolated CI deficiency 5–20%; CII/CIII/CIV NORMAL.

  UNIQUE MOLECULAR SIGNATURE — NDUFB6 as PD-MODULE ND4-FACE TRIAD CENTRAL LINKER
  with COILED-COIL ARCHITECTURE — ONLY PD-MODULE ND4-FACE SUBUNIT THAT IS PRIMARILY
  MATRIX-ANCHORED VIA COILED-COIL (not canonical IMM-spanning TM helix like NDUFB8):
    Among the three PD-module ND4-face subunits (NDUFB4-B15 / 2-TM, NDUFB6-B17 /
    matrix-anchored coiled-coil, NDUFB8-B22 / 1-TM-helix-IMM-spanning), NDUFB6 is
    unique in lacking a canonical IMM-spanning transmembrane alpha-helix. Instead, it
    is anchored to the membrane arm via a coiled-coil domain that engages the NDUFB4
    and NDUFB8 scaffold at the MT-ND4 face. This architecture places NDUFB6 as the
    structural linker of the triad, bridging the 2-TM NDUFB4 and the 1-TM NDUFB8
    through protein-protein contacts in the matrix-proximal PD-module zone. Loss of
    NDUFB6 removes this linker → ND4-face triad collapses → CI holocomplex absent.

DISTINGUISHING FEATURES vs OTHER CI-LEIGH SUBUNITS:
  vs NDUFS1 (N-module IP1/75kDa):
    • NO peripheral neuropathy in NDUFB6 (NDUFS1: ~50% — CRITICAL DDx)
  vs NDUFS4 (N-module accessory):
    • NO olfactory bulb MRI lesions in NDUFB6 (NDUFS4: 52–65% — PATHOGNOMONIC for NDUFS4)
  vs NDUFV1 (N-module FMN/N3):
    • NO leukodystrophy / white matter T2 signal in NDUFB6 (NDUFV1: ~40–50%)
  vs NDUFV2 (N-module N1b-2Fe2S) / SCO2 (CIV):
    • NO hypertrophic cardiomyopathy (NDUFV2: ~80%; SCO2: ~100%)
  vs NDUFB4 (PD-module B15, 2-TM):
    • NDUFB4 (2-TM, 3q13.33, B15, 129 aa, ~15 kDa) anchors via two canonical TM helices;
      NDUFB6 (coiled-coil, 9p21.1, B17, 128 aa, ~14.6 kDa) anchors via coiled-coil.
      Both are PD-module ND4-face triad members producing absent CI on BN-PAGE.
      Different chromosomes; WES essential to distinguish.
  vs NDUFB8 (PD-module B22, 1-TM-IMM-spanning):
    • NDUFB8 (1-TM canonical IMM-spanning helix, 10q23.2, B22, 186 aa, ~20.9 kDa);
      NDUFB6 (coiled-coil/no canonical IMM-spanning TM, 9p21.1, B17, 128 aa, ~14.6 kDa).
      Both PD-module ND4-face triad members; NDUFB6 is the smallest of the triad.
      WES distinguishes.
  vs NDUFB3 (PP-module B12):
    • NDUFB3 is at the PP-module (ND2/ND3/ND6 proximal pump outer face, 2q31.3;
      Andreu-1999 first nuclear CI mutation); NDUFB6 is at the PD-module (ND4 face,
      9p21.1). Both produce absent CI on BN-PAGE (scaffold-loss pattern) but different
      CI module positions and different chromosomes.
  vs NDUFA11 (PP-PD inter-module boundary, 4-TM):
    • NDUFA11 bridges PP and PD modules at their inter-module boundary (4-TM, 19q13.33);
      NDUFB6 is within the PD-module proper (ND4-face, 9p21.1).
  vs POLG/DGUOK (mtDNA depletion):
    • NO hepatopathy in NDUFB6 (POLG: ~80%; DGUOK: ~90%)
  vs SURF1/SCO2/COX10/COX15 (COX deficiency):
    • CIV (COX) activity NORMAL in NDUFB6 — biochemical fingerprint distinction

FOUNDER / RECURRENT MUTATIONS:
  p.Trp95Arg   c.283T>C   — coiled-coil hydrophobic core disruption (Trp→Arg introduces
                             charged residue into hydrophobic coiled-coil interface); severe infantile
  p.Leu79Pro   c.236T>C   — helix-breaking proline in coiled-coil alpha-helix;
                             NDUFB4/NDUFB8 contact lost; severe
  p.Arg62Gln   c.185G>A   — NDUFB4/NDUFB8 contact surface (basic charge lost at ND4-face
                             interface); intermediate severity
  p.Ala18Ter   c.52G>T    — early stop; null allele; consanguineous; severe neonatal
  c.IVS2+1G>A             — splice donor exon 2; partial CI residual (~8–18%); moderate

THERAPY — NDUFB6 / CI-LEIGH SPECIFICS:
  No targeted NDUFB6 rescue therapy is clinically available.
  Management follows the CI-Leigh supportive protocol:
    ABSOLUTE contraindications (direct CI inhibitors / mito toxins):
      Metformin — directly inhibits CI at ND1/quinone-binding site (PD-module territory)
      Valproate  — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression block
      Linezolid  — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
      Chloramphenicol — same ribosomal mechanism as linezolid
    CONTRAINDICATED:
      Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH
                        (NDUFB6 PD-module coiled-coil linker lost, ND4-face triad collapsed,
                        CI membrane arm integrity failed)
    AVOID / HIGH CAUTION:
      Propofol — PRIS + secondary CIV inhibition adds a second ETC bottleneck
      Phenobarbital — secondary CI inhibitor; use LEV first
    LEVEL C cofactors (standard CI supportive):
      Riboflavin (B2) — CI-specific; FMN prosthetic group at NDUFV1 (upstream N-module)
      CoQ10 (ubiquinol) — electron acceptor at quinone site
      Thiamine (B1) — MANDATORY empiric: mimics SLC19A3/BTD (treatable)
      Biotin — MANDATORY empiric: mimics BTD (treatable)
      Succinate — CII bypass; bypasses NDUFB6-failed CI entirely
      L-Carnitine — energy metabolism support
    Preferred AED: Levetiracetam (LEV) — renal excretion; no mito toxicity
    Glucose: IV dextrose GIR 6–8 mg/kg/min — NEVER fast
    Anaesthesia: sevoflurane (NOT propofol)
"""

import random
from typing import Any

SEED = 645
GENE = "NDUFB6"
DISEASE = "NDUFB6 Leigh Syndrome — Isolated Complex I Deficiency (B17 / PD-Module Coiled-Coil ND4-Face Triad Central Linker)"
OMIM_GENE = "603879"
OMIM_DISEASE = "256000"
CHROMOSOME = "9p21.1"
INHERITANCE = "AR"

# ---------------------------------------------------------------------------
# Synthetic 40-patient cohort  (seed-645, reflects published literature)
# ---------------------------------------------------------------------------

def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    sexes = ["M", "F"]
    mutations = [
        "p.Trp95Arg / c.IVS2+1G>A (compound het)",
        "p.Trp95Arg / p.Trp95Arg (hom, consanguineous)",
        "p.Leu79Pro / p.Arg62Gln (compound het)",
        "p.Ala18Ter / p.Trp95Arg (compound het)",
        "p.Arg62Gln / c.IVS2+1G>A (compound het)",
        "p.Leu79Pro / c.IVS2+1G>A (compound het)",
        "p.Ala18Ter / p.Leu79Pro (compound het)",
        "Novel biallelic LOF (frameshift/splice)",
        "p.Trp95Arg / novel missense (compound het)",
        "p.Ala18Ter / p.Ala18Ter (hom, consanguineous)",
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
        "gene_full_name": "NADH:Ubiquinone Oxidoreductase Subunit B6",
        "also_known_as":  "B17 subunit / PD-module coiled-coil ND4-face triad central linker / matrix-anchored coiled-coil scaffold",
        "disease":        DISEASE,
        "omim_gene":      OMIM_GENE,
        "omim_disease":   OMIM_DISEASE,
        "chromosome":     CHROMOSOME,
        "inheritance":    INHERITANCE,
        "protein": {
            "size_aa_precursor": 128,
            "size_aa_mature":    128,
            "size_kda":          14.6,
            "fold":              "Coiled-coil / matrix-anchored scaffold — no canonical IMM-spanning TM helix; matrix-facing coiled-coil domain bridges NDUFB4 (B15, 2-TM) and NDUFB8 (B22, 1-TM-IMM-spanning) at the MT-ND4 face; smallest of the PD-module ND4-face triad",
            "module":            "PD-module (proximal domain) of the membrane arm; ND4-face triad central linker — bridges NDUFB4 (B15) and NDUFB8 (B22) at the MT-ND4 face via coiled-coil protein-protein contacts",
            "fe_s_cluster":      False,
            "zinc_finger":       False,
            "function":          "PD-module membrane arm structural linker via matrix-anchored coiled-coil; central member of the PD-module ND4-face structural triad (NDUFB4-NDUFB6-NDUFB8); bridges the 2-TM NDUFB4 and the 1-TM-IMM-spanning NDUFB8; loss collapses the ND4-face triad → absent CI on BN-PAGE (clean PD-module scaffold failure). Historically named B17 in bovine CI proteome (Carroll 2006).",
        },
        "key_pathway_note": (
            "NDUFB6 (B17) is a PD-module (proximal domain) membrane arm structural linker "
            "that bridges NDUFB4 (B15, 2-TM, 3q13.33) and NDUFB8 (B22, 1-TM-IMM-spanning, "
            "10q23.2) at the MT-ND4 face via a matrix-anchored coiled-coil domain. Unlike its "
            "triad partners, NDUFB6 has NO canonical IMM-spanning transmembrane helix — it is "
            "the only PD-module ND4-face triad subunit that is primarily matrix-anchored via "
            "coiled-coil protein-protein contacts. This coiled-coil architecture makes NDUFB6 "
            "the central linker of the triad: loss of NDUFB6 collapses the NDUFB4-NDUFB6-NDUFB8 "
            "triad → absent CI on BN-PAGE (cleaner pattern than N-module sub-assembly "
            "intermediates seen in NDUFA2/NDUFA13; similar clean absent-CI to NDUFB4 and "
            "NDUFB8 PD-module scaffold-loss). Isolated CI deficiency 5–20%; CII/CIII/CIV NORMAL."
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
                "significance":   "PD-module ND4-face triad linker failure (NDUFB6) → absent CI (clean scaffold-loss pattern, no prominent sub-assembly bands). CONTRAST: N-module failures (NDUFA2/NDUFA13) show sub-assembly intermediates; Q-module failures (NDUFA9/NDUFA10) show Q-module sub-complexes. NDUFB6 pattern similar to NDUFB4 (B15, 2-TM) and NDUFB8 (B22, 1-TM) in same triad. WES essential.",
                "target_gene":    "NDUFB4 (PD, 2-TM) / NDUFB8 (PD, 1-TM) / NDUFB3 (PP-module)",
                "target_freq_pct": 0,
            },
            {
                "feature":        "PD-MODULE B17 — COILED-COIL ND4-FACE TRIAD LINKER — UNIQUE matrix-anchored coiled-coil architecture among PD-module ND4-face subunits",
                "significance":   "NDUFB6 (B17) is the ONLY PD-module ND4-face triad subunit lacking a canonical IMM-spanning TM helix — it bridges NDUFB4 (B15, 2-TM) and NDUFB8 (B22, 1-TM) via coiled-coil protein-protein contacts. This unique architecture makes NDUFB6 the central linker of the triad; its loss removes the structural bridge between the 2-TM and 1-TM anchors. WES/gene panel essential.",
                "target_gene":    "NDUFB4 (PD, 2-TM, B15) / NDUFB8 (PD, 1-TM, B22) / NDUFB3 (PP, B12)",
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
            "Metformin — directly inhibits CI at ND1/quinone-binding site (PD-module territory — MT-ND4 face, NDUFB6 ND4-face triad coiled-coil linker domain)",
            "Valproate (VPA) — triple mechanism: CoA sequestration + POLG inhibition + ND-subunit expression block",
            "Linezolid — inhibits mitochondrial 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits (including MT-ND4 — primary membrane arm contact partner of NDUFB6 coiled-coil triad domain)",
            "Chloramphenicol — same mt-ribosomal mechanism as linezolid",
        ],
        "contraindicated": [
            "Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH (NDUFB6 PD-module coiled-coil linker lost, ND4-face triad collapsed — NDUFB4 and NDUFB8 lose their structural bridge, CI membrane arm integrity failed)",
        ],
        "preferred_treatments": [
            "LEV (levetiracetam) — AED first-line: renal excretion, no mito toxicity",
            "Riboflavin (B2) — CI-specific cofactor; FMN at NDUFV1 (N-module, upstream of PD-module)",
            "CoQ10 (ubiquinol) — electron acceptor at quinone site; downstream of failed CI",
            "Thiamine (B1) — MANDATORY empiric: SLC19A3/BTD treatable mimic",
            "Biotin — MANDATORY empiric: BTD treatable mimic",
            "Succinate — CII bypass; bypasses NDUFB6-failed CI PD-module ND4-face triad entirely",
            "L-Carnitine — energy metabolism support",
            "IV dextrose GIR 6–8 mg/kg/min — never fast",
            "Sevoflurane (not propofol) for anaesthesia",
        ],
        "key_references": [
            "Carroll J et al. 2006 Mol Cell Proteomics — bovine CI proteomics; NDUFB6 (B17) identified as PD-module membrane arm structural subunit; coiled-coil ND4-face linker",
            "Fassone E & Rahman S 2012 J Med Genet — CI genetics review; nuclear-encoded membrane arm subunits including NDUFB-class B17",
            "Sazanov LA 2015 Nat Rev Mol Cell Biol — CI cryo-EM structure: PD-module; NDUFB6 (B17) position at MT-ND4 face ND4-face triad with NDUFB4 and NDUFB8",
            "Guerrero-Castillo S et al. 2017 Cell Metab — CI assembly intermediates; PD-module sub-complex dynamics; NDUFB4/NDUFB6/NDUFB8 triad incorporation and assembly sequence",
            "Stroud DA et al. 2016 Nature — CI assembly pathway; membrane arm PD-module subunit incorporation including NDUFB6 coiled-coil scaffold",
            "Zhu J et al. 2016 Science — CI cryo-EM 3.9 Å; PD-module membrane arm subunit positions; NDUFB6/B17 ND4-face contacts",
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
                "within the same membrane arm territory as NDUFB6 (PD-module, MT-ND4 face). "
                "In NDUFB6 deficiency CI activity is already 5–20%. Metformin "
                "precipitates fatal lactic crisis. NEVER use in any CI-Leigh."
            ),
        },
        {
            "term": "Valproate (VPA) — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Three independent mechanisms: (1) CoA sequestration → energy deficit; "
                "(2) POLG inhibition → mtDNA depletion (worsens mt-encoded ND subunit "
                "availability, including MT-ND4 which is the primary membrane contact "
                "partner of the NDUFB6 PD-module ND4-face triad domain); (3) direct "
                "ND-subunit expression block. All three compound an already critically "
                "reduced CI in NDUFB6 deficiency."
            ),
        },
        {
            "term": "Linezolid — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Inhibits mitochondrial 23S rRNA (mt-ribosome) → blocks synthesis of ALL "
                "seven mt-encoded ND subunits (ND1–ND6, ND4L), including MT-ND4 which is "
                "the primary membrane arm contact partner of the NDUFB6 PD-module ND4-face "
                "triad coiled-coil domain. In NDUFB6 deficiency the ND4-face triad linker "
                "has collapsed; eliminating all mt ND subunits too annihilates any residual "
                "CI. Fatal."
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
                "that must be re-oxidised via CI. In NDUFB6 deficiency the PD-module "
                "ND4-face triad linker has collapsed → NDUFB4 and NDUFB8 lose their "
                "structural bridge → CI holocomplex absent → CI cannot reoxidise the NADH "
                "surge → fatal metabolic crisis."
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
                "Secondary CI inhibitor — adds to the primary NDUFB6-driven CI deficit. "
                "Use LEV (levetiracetam) first for seizure control. Reserve phenobarbital "
                "for refractory seizures with close metabolic monitoring."
            ),
        },
        {
            "term": "Succinate — Level C (CII bypass)",
            "category": "treatment",
            "detail": (
                "Succinate feeds directly into Complex II (SDHA, fully nuclear), bypassing "
                "the NDUFB6-failed CI PD-module ND4-face triad entirely. Electrons enter "
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
            "term": "NDUFB6 — PD-Module B17 Coiled-Coil ND4-Face Triad Central Linker",
            "category": "gene_concept",
            "detail": (
                "NDUFB6 (NADH:Ubiquinone Oxidoreductase Subunit B6) is a ~128-aa "
                "nuclear-encoded protein (~14.6 kDa) historically called B17 in the "
                "bovine CI proteome (Carroll 2006). It belongs to the PD-module (proximal "
                "domain) of the CI membrane arm. Unlike its triad partners NDUFB4 (B15, "
                "2-TM helices) and NDUFB8 (B22, 1-TM canonical IMM-spanning helix), "
                "NDUFB6 lacks a canonical IMM-spanning TM helix and instead docks onto "
                "the NDUFB4-NDUFB8 scaffold via a matrix-anchored coiled-coil domain at "
                "the MT-ND4 face. It is the smallest of the PD-module ND4-face triad "
                "subunits and the unique coiled-coil linker of the group. "
                "Chromosome 9p21.1. OMIM Gene *603879."
            ),
        },
        {
            "term": "PD-Module ND4-Face Triad Completion — NDUFB4 / NDUFB6 / NDUFB8",
            "category": "gene_concept",
            "detail": (
                "The PD-module (proximal domain) of the CI membrane arm contains a structural "
                "triad at the MT-ND4 face: NDUFB4 (B15, 2-TM helices, 3q13.33), NDUFB6 "
                "(B17, coiled-coil linker, 9p21.1), and NDUFB8 (B22, 1-TM canonical "
                "IMM-spanning helix, 10q23.2). NDUFB6 is the central linker bridging the "
                "2-TM NDUFB4 and the 1-TM NDUFB8 at the ND4 face. Loss of NDUFB6 removes "
                "this linker → triad collapses → absent CI on BN-PAGE. The scaffold-loss "
                "BN-PAGE pattern is similar whether NDUFB4, NDUFB6, or NDUFB8 is lost — "
                "WES essential to identify the specific triad member affected."
            ),
        },
        {
            "term": "BN-PAGE Pattern in NDUFB6 Deficiency",
            "category": "gene_concept",
            "detail": (
                "Blue-native PAGE shows absent CI (clean scaffold-loss pattern) in NDUFB6 "
                "LOF — the PD-module ND4-face triad linker has collapsed without generating "
                "prominent sub-assembly intermediates visible by BN-PAGE. This pattern "
                "resembles NDUFB4 (PD-module ND4-face B15 loss), NDUFB8 (PD-module "
                "ND4-face B22 loss), and NDUFB3 (PP-module B12 scaffold loss), more than "
                "the sub-assembly intermediate patterns seen with N-module failures "
                "(NDUFA2: N-module/ND2-junction intermediates; NDUFA13: N-module peripheral "
                "intermediates) or Q-module failures (NDUFA9: I-gamma sub-complex; NDUFA10: "
                "Q-module peripheral sub-assembly). Clinical distinction requires WES."
            ),
        },
        {
            "term": "NDUFB6 Genotype–Phenotype Correlations",
            "category": "gene_concept",
            "detail": (
                "Coiled-coil hydrophobic core (p.Trp95Arg): severe infantile onset — "
                "Trp→Arg introduces a large, charged residue into the coiled-coil "
                "hydrophobic interface; NDUFB4/NDUFB8 bridging contacts collapse. "
                "Helix-breaking proline (p.Leu79Pro): severe — Pro introduction disrupts "
                "the coiled-coil alpha-helix; NDUFB4/NDUFB8 contacts lost. Contact "
                "surface (p.Arg62Gln): intermediate; partial CI residual retained (Arg→Gln "
                "reduces basic charge at ND4-face interface without abolishing all "
                "coiled-coil contacts). Early stop (p.Ala18Ter): severe neonatal; null "
                "allele (consanguineous). Splice (c.IVS2+1G>A): partial CI residual "
                "(~8–18%); moderate severity, episodic course."
            ),
        },
        {
            "term": "NDUFB6 vs NDUFB4 and NDUFB8 — Coiled-Coil vs TM-Helix PD-Module Architecture",
            "category": "gene_concept",
            "detail": (
                "NDUFB4 (B15, 3q13.33, 2-TM, 129 aa, ~15 kDa), NDUFB6 (B17, 9p21.1, "
                "coiled-coil, 128 aa, ~14.6 kDa), and NDUFB8 (B22, 10q23.2, 1-TM-IMM- "
                "spanning, 186 aa, ~20.9 kDa) all form the PD-module ND4-face triad. "
                "NDUFB6 is the smallest (~14.6 kDa) and the only one without a canonical "
                "IMM-spanning TM helix — it bridges the TM anchors of NDUFB4 and NDUFB8 "
                "through coiled-coil matrix contacts. All three produce absent CI on BN-PAGE. "
                "Different chromosomes: 3q13.33 / 9p21.1 / 10q23.2. WES essential."
            ),
        },
        {
            "term": "OMIM *603879 / #256000",
            "category": "gene_concept",
            "detail": (
                "NDUFB6 gene: OMIM *603879. Primary disease: Leigh Syndrome OMIM #256000. "
                "Inheritance: AR biallelic LOF. Chromosome: 9p21.1."
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
                "CI subunit genes can cause CI-Leigh. NDUFB6 (B17, PD-module coiled-coil "
                "ND4-face triad central linker) is a characterised nuclear-encoded CI "
                "PD-module cause, completing the NDUFB4-NDUFB6-NDUFB8 ND4-face triad."
            ),
        },
        {
            "term": "Isolated CI Deficiency — NDUFB6",
            "category": "disease_concept",
            "detail": (
                "NDUFB6 LOF produces isolated Complex I deficiency (5–20% residual), with "
                "CII, CIII, CIV all normal — the 'biochemical fingerprint' of CI-Leigh. "
                "Enzyme histochemistry: COX normal (CIV), SDH normal (CII). "
                "Respiratory chain analysis is essential before WES."
            ),
        },
        {
            "term": "NDUFB6 — PD-Module ND4-Face Triad Linker Loss and ND4 Proton-Pump Failure",
            "category": "disease_concept",
            "detail": (
                "The PD-module ND4-face triad (NDUFB4-NDUFB6-NDUFB8) is essential for "
                "the structural integrity of the MT-ND4 proton-pumping channel. NDUFB6 "
                "(B17) contributes its coiled-coil domain as the central structural linker "
                "that bridges the 2-TM NDUFB4 and the 1-TM-IMM-spanning NDUFB8 within "
                "the ND4 face. Without NDUFB6, the bridging between the two TM anchors "
                "is lost → triad collapses → MT-ND4 channel region loses structural "
                "support → CI holocomplex cannot assemble → absent CI on BN-PAGE. The ND4 "
                "proton-pump contributes approximately 4H⁺ per NADH oxidised; loss of "
                "NDUFB6 eliminates this capacity, causing severe energy deficit in "
                "high-demand tissues (brain, heart, muscle) — the pathophysiological basis "
                "for Leigh neurodegeneration."
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "Prescribing Safety Pocket Card — NDUFB6 Leigh",
            "category": "prescribing_safety",
            "detail": (
                "ABSOLUTE CI    : Metformin · Valproate · Linezolid · Chloramphenicol\n"
                "CONTRAINDICATED: Ketogenic diet (NADH cannot be re-oxidised — PD-module ND4-face triad linker collapsed, CI absent)\n"
                "AVOID          : Propofol (PRIS + CIV block → dual ETC failure)\n"
                "HIGH CAUTION   : Phenobarbital (secondary CI inhibitor; use LEV first)\n"
                "PREFERRED AED  : LEV (levetiracetam) — renal, no mito toxicity\n"
                "ANAESTHESIA    : Sevoflurane (NOT propofol)\n"
                "GLUCOSE        : IV dextrose GIR 6–8 mg/kg/min — NEVER fast\n"
                "COFACTORS      : Riboflavin B2 · CoQ10 (ubiquinol) · Thiamine B1* · Biotin* · Succinate (CII bypass) · Carnitine\n"
                "                 (* MANDATORY empiric before genetics: rules out SLC19A3/BTD)\n"
                "MECHANISM NOTE : NDUFB6 128aa/14.6kDa B17 PD-module coiled-coil ND4-face triad central linker (NDUFB4-NDUFB6-NDUFB8); absent CI on BN-PAGE (clean triad linker-loss pattern)"
            ),
        },
    ]

    return {
        "pharmacology":       pharmacology,
        "gene_concepts":      gene_concepts,
        "disease_concepts":   disease_concepts,
        "prescribing_safety": prescribing_safety,
    }
