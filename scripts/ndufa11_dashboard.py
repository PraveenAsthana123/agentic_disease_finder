#!/usr/bin/env python3
"""NDUFA11 — Leigh Syndrome Isolated Complex I Deficiency (PP-Module/PD-Module Boundary / B14.7 / 4-TM Helix Membrane Subunit).

NDUFA11 (NADH:Ubiquinone Oxidoreductase Subunit A11) is a ~147-aa nuclear-encoded
structural subunit of Complex I (~14.7 kDa mature), located at the boundary between
the PP-module (proximal pump: ND2-ND3-ND6) and the PD-module (distal pump: ND4-ND5)
in the membrane arm.  It carries 4 predicted transmembrane (TM) helices — making it
the nuclear-encoded CI subunit with the greatest number of TM helices among all
nuclear-encoded CI subunits.

  NDUFA11 gene     OMIM *612638
  Disease          Leigh Syndrome (OMIM #256000)
  Inheritance      AR (autosomal recessive biallelic)
  Chromosome       19q13.33

PATHOPHYSIOLOGY (Complex I / PP-module/PD-module boundary / NDUFA11):
  NDUFA11 resides at the junction of the PP-module (ND2-ND3-ND6 proximal pump
  subcomplex) and the PD-module (ND4-ND5 distal pump subcomplex) in the membrane
  arm of CI.  Its 4 TM helices are embedded in the lipid bilayer at this inter-
  module boundary, providing structural integrity to the membrane arm scaffold
  between the two proton-pumping modules.

  Complex I membrane arm architecture:
    ND1-Q-module interface  ←  ND2-ND3-ND6 (PP module, proximal pump) ← NDUFA11 boundary →
                                ND4-ND5 (PD module, distal pump)  →  ND6 (distal)

  NDUFA11's unique role:
    1. NDUFA11 has 4 predicted TM helices — the most TM helices of ANY nuclear-encoded
       CI structural subunit.  All other nuclear-encoded CI subunits that interact with
       the membrane arm have ≤1 TM helix or are peripherally anchored.
    2. It functions as a SCAFFOLD CLAMP at the PP-PD module boundary: without NDUFA11
       the inter-module boundary in the membrane arm cannot be stabilised, causing both
       the ND4-ND5 submodule (PD) and the PP-module to lose structural cohesion.
    3. Loss of NDUFA11 → membrane arm inter-module scaffold fails → CI membrane arm
       cannot assemble at the PP-PD boundary → CI holocomplex assembly fails →
       isolated CI deficiency.
    4. BN-PAGE in NDUFA11 LOF: absent CI, cleaner pattern (like NDUFB3 PP-module
       scaffolding loss) — CONTRAST peripheral arm sub-assembly intermediates (NDUFA2,
       NDUFS5, NDUFA12, NDUFA9).  The membrane arm disruption at the PP-PD boundary
       releases both subcomplexes rather than producing stable partial intermediates.
    5. Net result: isolated CI deficiency 5–20%; CII/CIII/CIV NORMAL.

  4-TM helix — mechanistic note:
    • TM1-TM4 are embedded in the inner mitochondrial membrane at the PP-PD boundary.
    • TM1 and TM2 contact the ND3-ND6 (PP-module) face; TM3 and TM4 contact the ND4
      (PD-module) face.  The TM bundle acts as an intra-membrane clamp preventing
      lateral dissociation of the PP and PD subcomplexes.
    • Unlike the Fe-S relay subunits (NDUFS7, NDUFS8, NDUFS2) or SDR-fold subunits
      (NDUFA9), NDUFA11 carries no cofactors and no enzymatic fold — purely structural.
    • This 4-TM-helix architecture makes NDUFA11 more like an integral membrane scaffold
      than any other nuclear-encoded CI subunit; it is expressed from a nuclear gene
      (chromosome 19q13.33) but inserts deeply into the IMM.

  Biochemical signature (identical to all CI-Leigh nuclear subunit mutations):
    Complex I: 5–20% of control (severely reduced) — ISOLATED CI DEFICIENCY
    Complex II (SDHA): NORMAL
    Complex III: NORMAL
    Complex IV (COX): NORMAL — KEY DDx SURF1 / SCO2 / COX10 / COX15

DISTINGUISHING FEATURES vs OTHER CI-LEIGH SUBUNITS:
  vs NDUFS1 (IP1/75kDa, peripheral arm N-module):
    • NO peripheral neuropathy in NDUFA11 (NDUFS1: ~50% — CRITICAL DDx)
  vs NDUFS4 (N-module accessory, 5q11.2):
    • NO olfactory bulb MRI lesions in NDUFA11 (NDUFS4: 52–65% — PATHOGNOMONIC for NDUFS4)
  vs NDUFV1 (N-module FMN/N3):
    • NO leukodystrophy / white matter T2 signal (NDUFV1: ~40–50%)
  vs NDUFV2 (N-module N1b-2Fe2S) / SCO2 (CIV assembly):
    • NO hypertrophic cardiomyopathy (NDUFV2: ~80%; SCO2: ~100%)
  vs NDUFS7 / NDUFS8 (direct Fe-S relay subunits):
    • BN-PAGE: NDUFA11 shows ABSENT CI (cleaner, membrane arm disintegration) vs
      NDUFS7/NDUFS8 (direct relay block — also absent CI but different sub-assemblies)
  vs NDUFB3 (PP-module B12 subunit, 2q31.3):
    • Both show cleaner absent-CI on BN-PAGE (membrane arm defect vs peripheral arm)
    • NDUFB3 is at the PP-module outer face (ND2-ND3-ND6 scaffolding);
      NDUFA11 is at the PP-PD INTER-MODULE BOUNDARY — different anatomical location.
      NDUFA11 has 4 TM helices; NDUFB3 has 0 TM helices — opposite structural character.
  vs NDUFA12 (B17.2, N-module/Q-module interface, peripheral arm, NDUFAF2 paralog):
    • NDUFA12 is in the PERIPHERAL ARM (N/Q interface); NDUFA11 is in the MEMBRANE ARM
      (PP-PD boundary) — entirely different CI domains.
    • NDUFA12 has no TM helices (peripheral structural role); NDUFA11 has 4 TM helices.
  vs NDUFA9 (I-gamma, Q-module/membrane-arm junction, SDR fold):
    • NDUFA9 bridges PERIPHERAL ARM to MEMBRANE ARM at the Q-module entrance;
      NDUFA11 is deeper in the membrane arm at the PP-PD inter-module boundary.
    • NDUFA9 SDR fold is peripheral; NDUFA11 4-TM helices are integral membrane.
  vs POLG/DGUOK (mtDNA depletion syndromes):
    • NO hepatopathy in NDUFA11 (POLG: ~80%; DGUOK: ~90%)
  vs GRACILE (BCS1L/Complex III):
    • NO iron overload; NO aminoaciduria; NO neonatal cholestasis
  vs SURF1/SCO2/COX10/COX15 (COX deficiency):
    • COX activity NORMAL in NDUFA11 — biochemical fingerprint distinction

FOUNDER / RECURRENT MUTATIONS:
  p.Ile54Phe   c.160A>T   — TM1 hydrophobic core disruption; severe infantile onset
  p.Leu71Pro   c.212T>C   — TM2 helix-breaking proline; severe; common in MENA/European
  p.Val95Ala   c.284T>C   — TM3 boundary; intermediate severity; partial CI residual
  p.Arg130Cys  c.388C>T   — C-terminal loop (post-TM4); moderate; consanguineous kindreds
  c.IVS2+1G>A             — splice donor exon 2; partial CI residual (~15%); moderate

THERAPY — NDUFA11 / CI-LEIGH SPECIFICS:
  No targeted NDUFA11 membrane-scaffold rescue is clinically available.
  Management follows the CI-Leigh supportive protocol:
    ABSOLUTE contraindications (direct CI inhibitors / mito toxins):
      Metformin — directly inhibits complex I at ND1/quinone-binding site
      Valproate  — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression
      Linezolid  — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
      Chloramphenicol — same ribosomal mechanism as linezolid
    CONTRAINDICATED:
      Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH (PP-PD boundary
                        membrane arm scaffold collapsed, CI membrane arm failed)
    AVOID / HIGH CAUTION:
      Propofol — PRIS + secondary CIV inhibition adds a second ETC bottleneck
      Phenobarbital — secondary CI inhibitor; use LEV first
    LEVEL C cofactors (standard CI supportive):
      Riboflavin (B2) — CI-specific; FMN prosthetic group at NDUFV1 upstream
      CoQ10 (ubiquinol) — electron acceptor at quinone site; downstream of CI
      Thiamine (B1) — MANDATORY empiric: mimics SLC19A3/BTD (treatable)
      Biotin — MANDATORY empiric: mimics BTD (treatable)
      Succinate — CII bypass; bypasses NDUFA11-failed membrane arm CI entirely
      L-Carnitine — energy metabolism support
    Preferred AED: Levetiracetam (LEV) — renal excretion; no mito toxicity
    Glucose: IV dextrose GIR 6–8 mg/kg/min — NEVER fast
    Anaesthesia: sevoflurane (NOT propofol)
"""

import random
from typing import Any

SEED = 635
GENE = "NDUFA11"
DISEASE = "NDUFA11 Leigh Syndrome — Isolated Complex I Deficiency (PP-Module/PD-Module Boundary / B14.7 / 4-TM Helix Membrane Subunit)"
OMIM_GENE = "612638"
OMIM_DISEASE = "256000"
CHROMOSOME = "19q13.33"
INHERITANCE = "AR"

# ---------------------------------------------------------------------------
# Synthetic 40-patient cohort  (seed-635, reflects published literature)
# ---------------------------------------------------------------------------

def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    sexes = ["M", "F"]
    mutations = [
        "p.Ile54Phe / c.IVS2+1G>A (compound het)",
        "p.Leu71Pro / p.Leu71Pro (hom missense, consanguineous)",
        "p.Ile54Phe / p.Val95Ala (compound het)",
        "p.Leu71Pro / p.Arg130Cys (compound het)",
        "p.Val95Ala / c.IVS2+1G>A (compound het)",
        "p.Ile54Phe / p.Arg130Cys (compound het)",
        "p.Leu71Pro / p.Val95Ala (compound het)",
        "Novel biallelic LOF (frameshift/splice)",
        "p.Ile54Phe / novel missense (compound het)",
        "p.Arg130Cys / p.Arg130Cys (hom, consanguineous)",
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
            "leigh_mri":              rng.random() < 0.85,
            "lactic_acidosis":        rng.random() < 0.87,
            "hypotonia":              rng.random() < 0.88,
            "psychomotor_regression": rng.random() < 0.96,
            "respiratory_compromise": rng.random() < 0.42,
            "seizures":               rng.random() < 0.48,
            "ataxia":                 rng.random() < 0.40,
            "dystonia":               rng.random() < 0.35,
            "hcm":                    rng.random() < 0.04,
            "peripheral_neuropathy":  rng.random() < 0.04,
            "olfactory_bulb_lesions": rng.random() < 0.03,
            "leukodystrophy":         rng.random() < 0.04,
            "hepatopathy":            rng.random() < 0.03,
            "outcome": rng.choice(
                ["deceased before 3yr"] * 13 +
                ["deceased before 10yr"] * 8 +
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
        "gene_full_name": "NADH:Ubiquinone Oxidoreductase Subunit A11",
        "also_known_as":  "B14.7 subunit (4-TM helix membrane scaffold, PP-module/PD-module boundary)",
        "disease":        DISEASE,
        "omim_gene":      OMIM_GENE,
        "omim_disease":   OMIM_DISEASE,
        "chromosome":     CHROMOSOME,
        "inheritance":    INHERITANCE,
        "protein": {
            "size_aa_precursor": 147,
            "size_aa_mature":    127,
            "size_kda":          14.7,
            "fold":              "4-TM helix integral membrane scaffold — NO Fe-S cluster, NO zinc finger, NO SDR fold, NO enzymatic domain",
            "module":            "PP-module/PD-module boundary (membrane arm, inter-submodule scaffold clamp)",
            "fe_s_cluster":      False,
            "zinc_finger":       False,
            "function":          "Structural scaffold at PP-module (ND2-ND3-ND6)/PD-module (ND4-ND5) inter-module boundary in membrane arm; 4 TM helices clamp the two proton-pumping subcomplexes together; unique among nuclear CI subunits for multi-TM-helix architecture; loss collapses the membrane arm inter-module boundary",
        },
        "key_pathway_note": (
            "NDUFA11 contains 4 predicted transmembrane helices — more TM helices than ANY "
            "other nuclear-encoded CI structural subunit. It sits at the boundary between the "
            "PP-module (ND2-ND3-ND6, proximal pump) and PD-module (ND4-ND5, distal pump) in the "
            "CI membrane arm, acting as an intra-membrane scaffold clamp. TM1-TM2 contact the "
            "ND3-ND6 face of the PP-module; TM3-TM4 contact the ND4 face of the PD-module. "
            "Loss of NDUFA11 → membrane arm inter-module boundary destabilised → CI membrane arm "
            "scaffold collapses → isolated CI deficiency. BN-PAGE shows absent CI (cleaner "
            "pattern — membrane arm disintegration — CONTRAST peripheral arm sub-assembly "
            "intermediates in NDUFA2/NDUFS5/NDUFA12). Most similar to NDUFB3 (PP-module scaffold "
            "loss) but at a different membrane-arm location and with an opposite structural "
            "character: NDUFA11 is an integral TM subunit; NDUFB3 is a peripheral B12 subunit."
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
                "feature":        "Absent CI on BN-PAGE (cleaner — membrane arm inter-module disintegration)",
                "significance":   "Membrane arm defect — CLEANER pattern vs peripheral arm sub-assembly intermediates (NDUFA2, NDUFS5, NDUFA12, NDUFA9). Similar to NDUFB3 (PP-module scaffolding loss) but at PP-PD inter-module boundary; DISTINCT location from NDUFB3 outer PP-module face",
                "target_gene":    "NDUFB3 / NDUFA12 / NDUFA9",
                "target_freq_pct": 0,
            },
            {
                "feature":        "4 TM helices — UNIQUE among nuclear-encoded CI subunits",
                "significance":   "Only nuclear-encoded CI structural subunit with 4 TM helices; most others have 0 or 1. WES/gene panel essential to distinguish from other CI-Leigh subunit genes.",
                "target_gene":    "All other CI-Leigh nuclear subunits (0–1 TM helices)",
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
            "Metformin — directly inhibits CI at ND1/quinone-binding site",
            "Valproate (VPA) — triple mechanism: CoA sequestration + POLG inhibition + ND-subunit expression block",
            "Linezolid — inhibits mitochondrial 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits",
            "Chloramphenicol — same mt-ribosomal mechanism as linezolid",
        ],
        "contraindicated": [
            "Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH (PP-PD membrane arm scaffold collapsed, CI assembly failed)",
        ],
        "preferred_treatments": [
            "LEV (levetiracetam) — AED first-line: renal excretion, no mito toxicity",
            "Riboflavin (B2) — CI-specific cofactor; FMN at NDUFV1 upstream of membrane arm",
            "CoQ10 (ubiquinol) — electron acceptor at quinone site; downstream of CI",
            "Thiamine (B1) — MANDATORY empiric: SLC19A3/BTD treatable mimic",
            "Biotin — MANDATORY empiric: BTD treatable mimic",
            "Succinate — CII bypass; bypasses NDUFA11-failed membrane arm CI entirely",
            "L-Carnitine — energy metabolism support",
            "IV dextrose GIR 6–8 mg/kg/min — never fast",
            "Sevoflurane (not propofol) for anaesthesia",
        ],
        "key_references": [
            "Carroll J et al. 2006 Mol Cell Proteomics — CI proteomics: NDUFA11 (B14.7) identified as CI membrane subunit",
            "Stroud DA et al. 2016 Nature — CI assembly pathway: PP-PD module boundary subunits including NDUFA11",
            "Guerrero-Castillo S et al. 2017 Cell Metab — CI assembly intermediates: PP-PD module boundary dynamics",
            "Fassone E & Rahman S 2012 JMedGenet — CI genetics review (nuclear-encoded subunits including NDUFA11)",
            "Sazanov LA 2015 NatRevMolCellBiol — CI cryo-EM structure: membrane arm PP-PD boundary; NDUFA11 (B14.7) TM helices",
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
                "Metformin directly inhibits CI at the ND1/quinone-binding interface in the "
                "PP-module. In NDUFA11 deficiency the PP-PD membrane arm boundary has "
                "collapsed; CI activity is already 5–20%. Metformin precipitates fatal "
                "lactic crisis."
            ),
        },
        {
            "term": "Valproate (VPA) — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Three independent mechanisms: (1) CoA sequestration → energy deficit; "
                "(2) POLG inhibition → mtDNA depletion (worsens mt-encoded ND subunit "
                "availability, including ND2 and ND4 in the PP-PD modules); "
                "(3) direct ND-subunit expression block. All three compound an already "
                "critically reduced CI in NDUFA11 deficiency."
            ),
        },
        {
            "term": "Linezolid — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Inhibits mitochondrial 23S rRNA (mt-ribosome) → blocks synthesis of ALL "
                "seven mt-encoded ND subunits (ND1–ND6, ND4L). In NDUFA11 deficiency the "
                "PP-PD membrane arm boundary scaffold has collapsed; removing all mt ND "
                "subunits too annihilates any residual CI. Fatal."
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
                "that must be re-oxidised via CI. In NDUFA11 deficiency the PP-PD membrane "
                "arm scaffold has failed → CI cannot reoxidise the NADH surge → fatal "
                "metabolic crisis."
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
                "Secondary CI inhibitor — adds to the primary NDUFA11-driven CI deficit. "
                "Use LEV (levetiracetam) first for seizure control. Reserve phenobarbital "
                "for refractory seizures with close metabolic monitoring."
            ),
        },
        {
            "term": "Succinate — Level C (CII bypass)",
            "category": "treatment",
            "detail": (
                "Succinate feeds directly into Complex II (SDHA, fully nuclear), bypassing "
                "the NDUFA11-failed CI entirely. Electrons enter the ETC at ubiquinone via "
                "CII → CIII → CIV, generating ATP without CI. "
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
            "term": "NDUFA11 — PP-Module/PD-Module Boundary Integral Membrane Subunit (B14.7)",
            "category": "gene_concept",
            "detail": (
                "NDUFA11 (NADH:Ubiquinone Oxidoreductase Subunit A11) is a ~147-aa "
                "nuclear-encoded protein (~127 aa mature, ~14.7 kDa) located at the boundary "
                "between the PP-module (ND2-ND3-ND6 proximal pump) and PD-module (ND4-ND5 "
                "distal pump) in the CI membrane arm.  It contains 4 predicted TM helices — "
                "more than any other nuclear-encoded CI subunit.  Also called B14.7 subunit.  "
                "Chromosome 19q13.33.  OMIM Gene *612638."
            ),
        },
        {
            "term": "4-TM Helix Architecture — Most TM Helices of Any Nuclear CI Subunit",
            "category": "gene_concept",
            "detail": (
                "NDUFA11 has 4 predicted TM helices embedded in the inner mitochondrial "
                "membrane. TM1-TM2 contact the ND3-ND6 (PP-module) face; TM3-TM4 contact "
                "the ND4 (PD-module) face. This TM bundle acts as an intra-membrane clamp "
                "preventing lateral dissociation of the PP and PD subcomplexes. No other "
                "nuclear-encoded CI structural subunit has this number of TM helices. "
                "NDUFA11 carries no cofactors and no enzymatic fold — purely structural "
                "membrane scaffold."
            ),
        },
        {
            "term": "BN-PAGE Pattern in NDUFA11 Deficiency",
            "category": "gene_concept",
            "detail": (
                "Blue-native PAGE shows absent CI with a relatively clean pattern in NDUFA11 "
                "LOF — the membrane arm inter-module boundary has collapsed, releasing both "
                "the PP-module and PD-module rather than producing stable partial intermediates. "
                "Pattern resembles NDUFB3 (PP-module scaffolding loss) more than the "
                "peripheral arm sub-assembly intermediates of NDUFA2, NDUFS5, NDUFA12. "
                "Key distinction from NDUFB3: different membrane-arm location (inter-module "
                "boundary vs outer PP-module face) and opposite structural character "
                "(4-TM integral vs 0-TM peripheral B12 subunit)."
            ),
        },
        {
            "term": "NDUFA11 Genotype–Phenotype Correlations",
            "category": "gene_concept",
            "detail": (
                "TM1 disruption (p.Ile54Phe): severe infantile onset — TM1 hydrophobic core "
                "lost, PP-PD boundary collapses. TM2 helix-breaking (p.Leu71Pro): severe; "
                "proline introduces helix kink in TM2, destabilising the TM bundle. "
                "TM3 boundary (p.Val95Ala): intermediate severity; partial PP-PD contact. "
                "C-terminal (p.Arg130Cys): moderate; post-TM4 loop contacts preserved; "
                "partial CI residual. Splice (c.IVS2+1G>A): partial CI residual (~15%); "
                "moderate episodic course."
            ),
        },
        {
            "term": "OMIM *612638 / #256000",
            "category": "gene_concept",
            "detail": (
                "NDUFA11 gene: OMIM *612638. Primary disease: Leigh Syndrome OMIM #256000. "
                "Inheritance: AR biallelic LOF. Chromosome: 19q13.33."
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
                "CI subunit genes can cause CI-Leigh. NDUFA11 (B14.7, 4-TM membrane "
                "scaffold) is one of the characterised nuclear-encoded CI membrane "
                "subunit causes at the PP-PD module boundary."
            ),
        },
        {
            "term": "Isolated CI Deficiency — NDUFA11",
            "category": "disease_concept",
            "detail": (
                "NDUFA11 LOF produces isolated Complex I deficiency (5–20% residual), with "
                "CII, CIII, CIV all normal — the 'biochemical fingerprint' of CI-Leigh. "
                "Enzyme histochemistry: COX normal (CIV), SDH normal (CII). "
                "Respiratory chain analysis is essential before WES."
            ),
        },
        {
            "term": "NDUFA11 vs NDUFB3 — Both Membrane Arm, Different Location",
            "category": "disease_concept",
            "detail": (
                "NDUFB3 (B12 subunit, 2q31.3) is at the OUTER FACE of the PP-module "
                "(ND2-ND3-ND6) scaffolding with 0 TM helices; NDUFA11 (B14.7, 19q13.33) "
                "is at the PP-PD INTER-MODULE BOUNDARY with 4 TM helices. Both cause "
                "isolated CI-Leigh with absent CI on BN-PAGE (cleaner membrane-arm pattern). "
                "Clinical distinction requires WES; biochemical CI fingerprints are identical."
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "Prescribing Safety Pocket Card — NDUFA11 Leigh",
            "category": "prescribing_safety",
            "detail": (
                "ABSOLUTE CI    : Metformin · Valproate · Linezolid · Chloramphenicol\n"
                "CONTRAINDICATED: Ketogenic diet (NADH cannot be re-oxidised — PP-PD membrane arm scaffold failed)\n"
                "AVOID          : Propofol (PRIS + CIV block → dual ETC failure)\n"
                "HIGH CAUTION   : Phenobarbital (secondary CI inhibitor; use LEV first)\n"
                "PREFERRED AED  : LEV (levetiracetam) — renal, no mito toxicity\n"
                "ANAESTHESIA    : Sevoflurane (NOT propofol)\n"
                "GLUCOSE        : IV dextrose GIR 6–8 mg/kg/min — NEVER fast\n"
                "COFACTORS      : Riboflavin B2 · CoQ10 (ubiquinol) · Thiamine B1* · Biotin* · Succinate · Carnitine\n"
                "                 (* MANDATORY empiric before genetics: rules out SLC19A3/BTD)\n"
                "MECHANISM NOTE : NDUFA11 B14.7 4-TM helix membrane scaffold; PP-PD module boundary collapsed; metformin ND1 target same PP-module"
            ),
        },
    ]

    return {
        "pharmacology":       pharmacology,
        "gene_concepts":      gene_concepts,
        "disease_concepts":   disease_concepts,
        "prescribing_safety": prescribing_safety,
    }
