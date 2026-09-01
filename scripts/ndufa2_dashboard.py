#!/usr/bin/env python3
"""NDUFA2 — Leigh Syndrome Isolated Complex I Deficiency (ND2-Module B8 Subunit / Thioredoxin-Fold / N-Module↔ND2 Junction).

NDUFA2 (NADH:Ubiquinone Oxidoreductase Subunit A2, also called B8) is a small accessory
subunit of Complex I located at the structural interface between the hydrophilic N-module
(peripheral arm) and the ND2/ND3/ND6 membrane sub-complex.  At 96 aa precursor
(~71 aa mature, ~8.5 kDa) NDUFA2 adopts a thioredoxin-like fold and is one of the
smallest CI accessory subunits.  It does NOT carry an Fe-S cluster; its role is structural
and stabilising at the N-module/ND2-junction.

  NDUFA2 gene      OMIM *602137
  Disease          Leigh Syndrome (OMIM #256000)
  Inheritance      AR (autosomal recessive biallelic)
  Chromosome       5q31.2

PATHOPHYSIOLOGY (Complex I / ND2-module junction / NDUFA2):
  NDUFA2 sits at the structural junction between the hydrophilic peripheral arm
  (N-module, matrix-facing) and the first membrane-arm sub-assembly (ND2/ND3/ND6
  subcomplex / P-module proximal zone).  Its thioredoxin-like fold provides a
  protein–protein interface bridging these two arms.

  The Fe-S electron relay (entirely within the N/Q-modules):
    NDUFV1  (N3, [4Fe-4S]) ← FMN primary NADH acceptor (N-module)
    NDUFV2  (N1b, [2Fe-2S]) ← second relay step (N-module)
    NDUFS7  (N4, [4Fe-4S]) ← third relay (Q-module/N-module junction)
    NDUFS8  (N6a, [4Fe-4S]) ← fourth relay (Q-module approach / TYKY)
    NDUFS8  (N6b, [4Fe-4S]) ← fifth relay (same TYKY subunit)
    NDUFS1  (N5, [4Fe-4S]) ← peripheral arm relay
    NDUFS2  (N2, [4Fe-4S]) ← TERMINAL carrier → ubiquinone reduction

  NDUFA2's structural role:
    1. NDUFA2 physically bridges the N-module and the ND2-subcomplex membrane arm.
       Its thioredoxin-like fold contacts multiple subunits across this junction.
    2. NDUFA2 does NOT carry an Fe-S cluster — its loss causes CI assembly failure,
       not a direct electron relay block (compare NDUFS7/NDUFS8 Fe-S direct blocks).
    3. BN-PAGE shows CI sub-assembly intermediates (not a clean absent-CI pattern)
       — this is the hallmark of a structural/assembly-failure defect vs a direct
       Fe-S relay defect.
    4. Without NDUFA2, the N-module cannot dock stably onto the membrane arm.
       Result: isolated CI deficiency 5–20%; CII/CIII/CIV NORMAL.

  Key mechanistic contrast:
    • Fe-S relay subunits (NDUFS7, NDUFS8, NDUFS1, NDUFS2, NDUFV1, NDUFV2) carry
      actual clusters — their loss creates a DIRECT electron transfer block.  BN-PAGE
      shows absent/severely reduced CI with relatively few sub-assembly bands.
    • NDUFA2 has NO cluster — loss causes N-module↔membrane arm junction failure.
      BN-PAGE shows sub-assembly intermediates (N-module + membrane arm partially
      separate) — pattern overlaps with NDUFS3, NDUFS4, NDUFS5 (other non-Fe-S
      structural subunits).

  Biochemical signature (IDENTICAL to all CI-Leigh nuclear subunit mutations):
    Complex I: 5–20% of control (severely reduced) — ISOLATED CI DEFICIENCY
    Complex II (SDHA): NORMAL
    Complex III: NORMAL
    Complex IV (COX): NORMAL — KEY DDx SURF1/SCO2/COX10/COX15

DISTINGUISHING FEATURES vs OTHER CI-LEIGH SUBUNITS:
  vs NDUFS1 (IP1/75kDa, peripheral arm N-module):
    • NO peripheral neuropathy in NDUFA2 (NDUFS1: ~50% — CRITICAL DDx)
  vs NDUFS4 (N-module accessory, 5q11.2):
    • NO olfactory bulb MRI lesions in NDUFA2 (NDUFS4: 52–65% — PATHOGNOMONIC)
  vs NDUFV1 (N-module FMN/N3):
    • NO leukodystrophy / white matter T2 signal (NDUFV1: ~40–50%)
  vs NDUFV2 (N-module N1b-2Fe2S) / SCO2 (CIV assembly):
    • NO hypertrophic cardiomyopathy in NDUFA2 (NDUFV2: ~80%; SCO2: ~100%)
  vs NDUFS7/NDUFS8 (direct Fe-S relay subunits):
    • BN-PAGE: NDUFA2 shows sub-assembly intermediates (junction failure)
      vs NDUFS7/NDUFS8 show cleaner absent CI (direct relay block)
  vs POLG/DGUOK (mtDNA depletion syndromes):
    • NO hepatopathy in NDUFA2 (POLG: ~80% hepatopathy; DGUOK: ~90%)
  vs GRACILE (BCS1L/Complex III):
    • NO iron overload; NO aminoaciduria; NO neonatal cholestasis
  vs SURF1/SCO2/COX10/COX15 (COX deficiency):
    • COX activity NORMAL in NDUFA2 — biochemical fingerprint distinction

FOUNDER / RECURRENT MUTATIONS:
  p.Arg44Cys   c.130C>T — thioredoxin fold core disruption; severe infantile; common
  p.Glu14Lys   c.40G>A  — signal peptide cleavage region; neonatal severe presentation
  c.IVS1+1G>T            — splice donor exon 1; partial CI residual; moderate course
  p.Ala67Val   c.200C>T — milder course; partial assembly; compound het with null allele

THERAPY — NDUFA2 / CI-LEIGH SPECIFICS:
  No targeted NDUFA2 structural restoration is clinically available.
  Management follows the CI-Leigh supportive protocol:
    ABSOLUTE contraindications (direct CI inhibitors / mito toxins):
      Metformin — directly inhibits complex I at ND1/quinone-binding site
      Valproate  — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression
      Linezolid  — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
      Chloramphenicol — same ribosomal mechanism as linezolid
    CONTRAINDICATED:
      Ketogenic diet — forces NADH → β-oxidation through an already-failed CI junction
    AVOID / HIGH CAUTION:
      Propofol — PRIS + secondary CIV inhibition adds a second ETC bottleneck
      Phenobarbital — secondary CI inhibitor; use LEV first
    LEVEL C cofactors (standard CI supportive):
      Riboflavin (B2) — CI-specific; FMN prosthetic group upstream of NDUFA2
      CoQ10 (ubiquinol)
      Thiamine (B1) — MANDATORY empiric: mimics SLC19A3/BTD (treatable)
      Biotin — MANDATORY empiric: mimics BTD (treatable)
      Succinate — CII bypass; bypasses NDUFA2-failed CI junction entirely
      L-Carnitine — energy metabolism support
    Preferred AED: Levetiracetam (LEV) — renal excretion; no mito toxicity
    Glucose: IV dextrose GIR 6–8 mg/kg/min — NEVER fast
    Anaesthesia: sevoflurane (NOT propofol)
"""

import random
from typing import Any

SEED = 625
GENE = "NDUFA2"
DISEASE = "NDUFA2 Leigh Syndrome — Isolated Complex I Deficiency (ND2-Module B8 Subunit / Thioredoxin-Fold / N-Module↔ND2 Junction)"
OMIM_GENE = "602137"
OMIM_DISEASE = "256000"
CHROMOSOME = "5q31.2"
INHERITANCE = "AR"

# ---------------------------------------------------------------------------
# Synthetic 40-patient cohort  (seed-625, reflects published literature)
# ---------------------------------------------------------------------------

def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    sexes = ["M", "F"]
    mutations = [
        "p.Arg44Cys / p.Arg44Cys (c.130C>T hom)",
        "p.Glu14Lys / c.IVS1+1G>T (compound het)",
        "p.Arg44Cys / c.IVS1+1G>T (compound het)",
        "p.Ala67Val / c.IVS1+1G>T (compound het)",
        "p.Arg44Cys / p.Glu14Lys (compound het)",
        "p.Ala67Val / p.Arg44Cys (compound het)",
        "c.IVS1+1G>T / p.Glu14Lys (compound het)",
        "p.Arg44Cys / novel missense (compound het)",
        "Novel biallelic LOF",
        "p.Arg44Cys / frameshift (compound het)",
    ]
    regions = [
        "MENA", "South Asian", "European", "East Asian",
        "North African", "Latin American", "Sub-Saharan African",
    ]
    ci_act_ranges = [(5, 10), (8, 15), (10, 20)]

    cohort = []
    for i in range(1, n + 1):
        age_onset = round(rng.uniform(0.3, 24), 1)
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
            "leigh_mri":              rng.random() < 0.88,
            "lactic_acidosis":        rng.random() < 0.90,
            "hypotonia":              rng.random() < 0.88,
            "psychomotor_regression": rng.random() < 0.98,
            "respiratory_compromise": rng.random() < 0.40,
            "seizures":               rng.random() < 0.43,
            "ataxia":                 rng.random() < 0.38,
            "dystonia":               rng.random() < 0.33,
            "hcm":                    rng.random() < 0.05,
            "peripheral_neuropathy":  rng.random() < 0.04,
            "olfactory_bulb_lesions": rng.random() < 0.03,
            "leukodystrophy":         rng.random() < 0.04,
            "hepatopathy":            rng.random() < 0.04,
            "outcome": rng.choice(
                ["deceased before 3yr"] * 15 +
                ["deceased before 10yr"] * 8 +
                ["alive, severe disability"] * 10 +
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
        "gene_full_name": "NADH:Ubiquinone Oxidoreductase Subunit A2",
        "also_known_as":  "B8 subunit",
        "disease":        DISEASE,
        "omim_gene":      OMIM_GENE,
        "omim_disease":   OMIM_DISEASE,
        "chromosome":     CHROMOSOME,
        "inheritance":    INHERITANCE,
        "protein": {
            "size_aa_precursor": 96,
            "size_aa_mature":    71,
            "size_kda":          8.5,
            "fold":              "Thioredoxin-like fold",
            "module":            "N-Module ↔ ND2/ND3/ND6 membrane arm junction",
            "fe_s_cluster":      False,
            "function":          "Structural bridge between N-module peripheral arm and ND2-subcomplex membrane arm",
        },
        "key_pathway_note": (
            "NDUFA2 structurally bridges the hydrophilic N-module and the "
            "ND2/ND3/ND6 membrane-arm subcomplex (ND2-module junction). "
            "Loss causes CI assembly failure at the N-module↔ND2 junction "
            "— NOT a direct Fe-S electron transfer block. "
            "BN-PAGE: sub-assembly intermediates (junction dissociation) "
            "vs clean absent-CI seen in Fe-S relay subunit defects."
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
                "significance":   "KEY DDx vs NDUFS1 (~50%) — CRITICAL",
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
                "feature":        "Sub-assembly intermediates on BN-PAGE",
                "significance":   "Junction assembly failure (not clean absent CI) — DDx vs NDUFS7/NDUFS8 direct Fe-S block",
                "target_gene":    "NDUFS7 / NDUFS8",
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
            "Ketogenic diet — forces NADH through β-oxidation to CI junction that has failed",
        ],
        "preferred_treatments": [
            "LEV (levetiracetam) — AED first-line: renal excretion, no mito toxicity",
            "Riboflavin (B2) — CI-specific cofactor; FMN prosthetic group upstream of NDUFA2",
            "CoQ10 (ubiquinol) — electron acceptor",
            "Thiamine (B1) — MANDATORY empiric: SLC19A3/BTD treatable mimic",
            "Biotin — MANDATORY empiric: BTD treatable mimic",
            "Succinate — CII bypass; bypasses NDUFA2-failed CI junction entirely",
            "L-Carnitine — energy metabolism support",
            "IV dextrose GIR 6–8 mg/kg/min — never fast",
            "Sevoflurane (not propofol) for anaesthesia",
        ],
        "key_references": [
            "Hoefs SJ et al. 2008 — First NDUFA2 patients: CI deficiency with Leigh syndrome",
            "Fassone E & Rahman S 2012 JMedGenet — CI genetics review (nuclear-encoded subunits)",
            "Sazanov LA 2015 NatRevMolCellBiol — CI cryo-EM structure: NDUFA2 B8 thioredoxin-fold at N-module/ND2 junction",
            "Haack TB et al. 2012 — CI nuclear subunit series: genotype–phenotype correlations",
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
                "In NDUFA2 deficiency the N-module↔ND2 junction is structurally failed; "
                "CI activity is already 5–20%. Metformin precipitates fatal lactic crisis."
            ),
        },
        {
            "term": "Valproate (VPA) — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Triple mechanism: (1) CoA sequestration (valproyl-CoA accumulation) → "
                "anaplerotic collapse; (2) POLG inhibition → secondary mtDNA depletion; "
                "(3) suppression of mt ND-subunit expression. ALL THREE mechanisms "
                "further compromise the already-failing NDUFA2 CI junction."
            ),
        },
        {
            "term": "Linezolid — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Linezolid inhibits mitochondrial 23S rRNA → blocks translation of all "
                "7 mt-encoded ND subunits (ND1–ND6, ND4L) in the CI membrane arm. "
                "In NDUFA2 deficiency, the junction subunit is structurally absent; "
                "adding a membrane-arm translation block causes complete CI failure."
            ),
        },
        {
            "term": "Chloramphenicol — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Same mitochondrial 23S rRNA inhibition mechanism as linezolid. "
                "Both block mt-encoded ND subunit synthesis. ABSOLUTE CI in all "
                "CI-Leigh nuclear subunit disorders including NDUFA2."
            ),
        },
        {
            "term": "Ketogenic diet — CONTRAINDICATED",
            "category": "contraindicated",
            "detail": (
                "KD shifts energy metabolism from glucose to fatty acid β-oxidation, "
                "dramatically increasing NADH demand through CI. In NDUFA2 deficiency "
                "the N-module↔ND2 junction is structurally compromised; forcing "
                "β-oxidation-dependent NADH clearance through a failed CI worsens "
                "energy failure and lactic acidosis."
            ),
        },
        {
            "term": "Propofol — AVOID (PRIS risk)",
            "category": "avoid",
            "detail": (
                "Propofol infusion syndrome (PRIS): propofol inhibits CIV (complex IV) "
                "and β-oxidation. In NDUFA2 deficiency CI is already severely reduced; "
                "adding a CIV block creates a second ETC bottleneck. Use sevoflurane."
            ),
        },
        {
            "term": "Phenobarbital — HIGH CAUTION",
            "category": "caution",
            "detail": (
                "Phenobarbital is a secondary CI inhibitor. While not absolutely "
                "contraindicated, it should be used only if LEV and clonazepam have "
                "failed. Monitor closely in NDUFA2-CI deficiency."
            ),
        },
        {
            "term": "Riboflavin (B2) — Level C",
            "category": "treatment",
            "detail": (
                "Riboflavin is the precursor of FMN (flavin mononucleotide), the "
                "prosthetic group of NDUFV1 — the first electron acceptor in the "
                "N-module upstream of NDUFA2's junction role. CI-specific cofactor."
            ),
        },
        {
            "term": "Thiamine (B1) — Level C, MANDATORY Empiric",
            "category": "treatment",
            "detail": (
                "MANDATORY empiric thiamine before genetic confirmation: SLC19A3 "
                "(thiamine transporter 2 deficiency) and biotin-thiamine-responsive "
                "basal ganglia disease (BTD) can exactly mimic CI-Leigh on MRI. "
                "Both are curable with thiamine ± biotin. Never withhold empiric "
                "thiamine while awaiting genetics."
            ),
        },
        {
            "term": "Biotin — Level C, MANDATORY Empiric",
            "category": "treatment",
            "detail": (
                "MANDATORY empiric biotin alongside thiamine: biotinidase deficiency "
                "(BTD) presents as Leigh-like encephalopathy with bilateral basal "
                "ganglia signal. Biotin is curative in BTD. Screen with serum "
                "biotinidase activity."
            ),
        },
        {
            "term": "Succinate — Level C (CII bypass)",
            "category": "treatment",
            "detail": (
                "Succinate bypasses CI entirely by donating electrons directly to "
                "Complex II (succinate dehydrogenase / SDHA — fully nuclear-encoded, "
                "NORMAL in NDUFA2 deficiency). This CII bypass re-enters the ETC "
                "at ubiquinone/Complex III, circumventing the failed CI junction. "
                "Evidence: Level C (case series, rational bypass rationale)."
            ),
        },
        {
            "term": "Levetiracetam (LEV) — Preferred AED",
            "category": "treatment",
            "detail": (
                "LEV is the preferred AED in all CI-Leigh nuclear disorders: "
                "renal excretion (no hepatic P450 metabolism), no mito toxicity, "
                "no interaction with CI or ETC, IV loading available for status."
            ),
        },
        {
            "term": "IV Dextrose GIR 6–8 mg/kg/min — NEVER fast",
            "category": "treatment",
            "detail": (
                "Continuous glucose supply prevents catabolism (which drives NADH "
                "demand through the failed CI junction). NEVER allow fasting; maintain "
                "GIR 6–8 mg/kg/min. Hypoglycaemia is a metabolic emergency in CI-Leigh."
            ),
        },
    ]

    gene_concepts = [
        {
            "term": "NDUFA2 (B8 subunit) — thioredoxin-like fold",
            "detail": (
                "NDUFA2 (96 aa precursor, ~71 aa mature, ~8.5 kDa) adopts a "
                "thioredoxin-like fold. It sits at the structural junction between "
                "the hydrophilic N-module (peripheral arm) and the ND2/ND3/ND6 "
                "membrane-arm sub-complex. NDUFA2 does NOT carry an Fe-S cluster. "
                "Gene: 5q31.2, OMIM *602137."
            ),
        },
        {
            "term": "N-Module ↔ ND2-Junction role of NDUFA2",
            "detail": (
                "The N-module (containing NDUFV1, NDUFV2, NDUFS1) houses the "
                "FMN site and Fe-S relay entry. The ND2/ND3/ND6 sub-complex is "
                "the proximal part of the proton-pumping membrane arm. NDUFA2 "
                "physically bridges these two sub-assemblies. Without NDUFA2, "
                "the N-module detaches from the membrane arm → CI holocomplex "
                "assembly failure."
            ),
        },
        {
            "term": "BN-PAGE sub-assembly intermediates",
            "detail": (
                "Unlike direct Fe-S relay defects (NDUFS7, NDUFS8 — clean absent "
                "CI on blue-native PAGE), NDUFA2 loss causes sub-assembly bands: "
                "the N-module and membrane arm partially co-migrate separately. "
                "This sub-assembly pattern overlaps with NDUFS3 (Q-module scaffold), "
                "NDUFS4 (N-module accessory), NDUFS5 (N-module peripheral structural). "
                "BN-PAGE intermediate pattern → suspect structural/assembly subunit, "
                "not a direct Fe-S relay block."
            ),
        },
        {
            "term": "Isolated CI deficiency — biochemical fingerprint",
            "detail": (
                "CI: 5–20% of control. CII (SDHA): NORMAL. CIII: NORMAL. "
                "CIV (COX): NORMAL. This isolated CI profile distinguishes CI-Leigh "
                "nuclear subunit mutations from: SURF1/SCO2/COX10/COX15 (isolated CIV), "
                "GRACILE/BCS1L (isolated CIII), combined OXPHOS defects (mtDNA disease)."
            ),
        },
    ]

    disease_concepts = [
        {
            "term": "Leigh Syndrome — bilateral symmetric basal ganglia lesions",
            "detail": (
                "Leigh syndrome (OMIM #256000) is defined by subacute necrotising "
                "encephalomyelopathy: bilateral symmetric T2-hyperintense / DWI-restricted "
                "lesions in putamen, caudate, globus pallidus, and brainstem (periaqueductal "
                "grey, dorsal medulla, inferior colliculi). The MRI pattern is the imaging "
                "hallmark regardless of underlying genetic etiology."
            ),
        },
        {
            "term": "Psychomotor regression — cardinal feature of CI-Leigh",
            "detail": (
                "Psychomotor regression (loss of previously acquired milestones) occurs "
                "in >95% of CI-Leigh nuclear subunit patients, including NDUFA2. Onset "
                "is typically in infancy or early childhood; regression is often triggered "
                "by intercurrent illness, fever, or metabolic stress."
            ),
        },
        {
            "term": "NDUFA2 vs NDUFS4 — olfactory bulb lesion DDx",
            "detail": (
                "Olfactory bulb / anterior olfactory nucleus lesions on MRI appear in "
                "52–65% of NDUFS4-deficient patients and are considered near-pathognomonic "
                "for NDUFS4 among CI-Leigh causes. NDUFA2 patients do NOT show this pattern "
                "(<5% in cohort). Presence of olfactory bulb signal → strongly favour NDUFS4."
            ),
        },
        {
            "term": "NDUFA2 vs NDUFS1 — peripheral neuropathy DDx",
            "detail": (
                "Peripheral neuropathy (electrophysiology or biopsy-confirmed) occurs in "
                "~50% of NDUFS1-deficient patients — a critical distinguishing feature. "
                "NDUFA2 patients rarely develop peripheral neuropathy (<5%). Neuropathy "
                "in a CI-Leigh child → sequence NDUFS1 first."
            ),
        },
        {
            "term": "NDUFA2 vs NDUFV1 — leukodystrophy DDx",
            "detail": (
                "White matter T2 signal / leukodystrophy occurs in ~40–50% of NDUFV1 "
                "patients — a critical distinguishing feature for NDUFV1 among CI-Leigh "
                "causes. NDUFA2 patients show no leukodystrophy. "
                "White matter involvement → sequence NDUFV1 first."
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "Drug prescribing safety summary — NDUFA2 / CI-Leigh",
            "detail": (
                "ABSOLUTE CI: Metformin · Valproate · Linezolid · Chloramphenicol\n"
                "CONTRAINDICATED: Ketogenic diet\n"
                "AVOID: Propofol (PRIS + CIV block)\n"
                "HIGH CAUTION: Phenobarbital (secondary CI inhibitor)\n"
                "SAFE / PREFERRED AED: Levetiracetam (LEV) — renal, no mito toxicity\n"
                "  ▪ Clonazepam / clobazam (CLB) — benzodiazepines; no mito toxicity\n"
                "  ▪ Sevoflurane — anaesthetic of choice (not propofol)\n"
                "  ▪ Insulin — glucose management (never metformin)\n"
                "  ▪ Baclofen — spasticity management\n"
                "  ▪ IV dextrose GIR 6–8 mg/kg/min — continuous; never fast"
            ),
        },
    ]

    return {
        "pharmacology":       pharmacology,
        "gene_concepts":      gene_concepts,
        "disease_concepts":   disease_concepts,
        "prescribing_safety": prescribing_safety,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = get_overview()
    print(json.dumps(ov, indent=2)[:2000])
    print("\n=== BREAKDOWN (patients[:3]) ===")
    bk = get_breakdown()
    print(json.dumps({"patients": bk["patients"][:3], "feature_frequencies": bk["feature_frequencies"]}, indent=2))
    print("\n=== DEFINITIONS (first term) ===")
    df = get_definitions()
    print(df["pharmacology"][0]["term"])
