#!/usr/bin/env python3
"""NDUFA10 — Leigh Syndrome Isolated Complex I Deficiency (42kDa / NDP-Kinase-Like Rossmann-Fold / PINK1-Phosphorylation-Target / Q-Module-Associated Peripheral Subunit).

NDUFA10 (NADH:Ubiquinone Oxidoreductase Subunit A10) is a ~335-aa nuclear-encoded
structural subunit of Complex I (~38.5 kDa), associated with the Q-module, contacting
NDUFS2 and NDUFS3 on the quinone-module scaffold.

  NDUFA10 gene     OMIM *603835
  Disease          Leigh Syndrome (OMIM #256000)
  Inheritance      AR (autosomal recessive biallelic)
  Chromosome       2q37.3

PATHOPHYSIOLOGY (Complex I / Q-module-associated / NDUFA10 / Kinase-Like Rossmann Fold):
  NDUFA10 contains a nucleoside diphosphate (NDP) kinase-like domain (Rossmann fold)
  that is CATALYTICALLY INACTIVE — this Rossmann fold is repurposed for structural
  scaffolding within the Q-module periphery (contacts NDUFS2-PSST and NDUFS3-30kDa).

  UNIQUE MOLECULAR SIGNATURE — NDUFA10 as PINK1 phosphorylation target:
    NDUFA10 is the ONLY CI structural subunit that is a direct phosphorylation substrate
    of PINK1 (PTEN-Induced Kinase 1), the Parkinson's disease gene on chromosome 1p36.
    PINK1 phosphorylates NDUFA10 at Ser250 within the Rossmann NDP-kinase domain.
    Phosphorylation promotes CI activity and Q-module stability. In PINK1-deficient cells
    (Parkinson's disease model), Ser250 dephosphorylation destabilises NDUFA10 → reduced CI
    activity → this is the mechanistic link between PINK1-Parkinsonism and CI deficiency.
    In NDUFA10 biallelic LOF (Leigh), the protein is absent — both the structural scaffold
    AND the PINK1-responsive CI regulation are lost simultaneously.

  Loss of NDUFA10 → Q-module peripheral scaffolding fails → BN-PAGE shows Q-module
  sub-assembly intermediates (partial Q-module + peripheral arm visible — similar to
  NDUFA9 I-gamma junction pattern). CONTRAST: cleaner absent CI in NDUFS7/NDUFS8
  (direct Fe-S relay), NDUFB3 (PP-module scaffold), and NDUFA11 (PP-PD inter-module boundary).
  Isolated CI deficiency 5–20%; CII/CIII/CIV NORMAL.

DISTINGUISHING FEATURES vs OTHER CI-LEIGH SUBUNITS:
  vs NDUFS1 (IP1/75kDa, peripheral arm N-module):
    • NO peripheral neuropathy in NDUFA10 (NDUFS1: ~50% — CRITICAL DDx)
  vs NDUFS4 (N-module accessory, 5q11.2):
    • NO olfactory bulb MRI lesions in NDUFA10 (NDUFS4: 52–65% — PATHOGNOMONIC for NDUFS4)
  vs NDUFV1 (N-module FMN/N3):
    • NO leukodystrophy / white matter T2 signal (NDUFV1: ~40–50%)
  vs NDUFV2 (N-module N1b-2Fe2S) / SCO2 (CIV assembly):
    • NO hypertrophic cardiomyopathy (NDUFV2: ~80%; SCO2: ~100%)
  vs NDUFA13 (GRIM-19, N-module peripheral stabiliser):
    • NDUFA13 contacts the NDUFV1-NDUFV2 face of the N-module; NDUFA10 contacts NDUFS2-NDUFS3
      in the Q-module peripheral scaffold — different module, different face.
    • NDUFA13 has GRIM-19 apoptosis + STAT3 inhibitor dual function; NDUFA10 has PINK1
      phosphorylation regulatory role — both are unique, but in entirely different pathways.
  vs NDUFA9 (I-gamma, Q-module/membrane arm junction, SDR fold):
    • NDUFA9 is at the Q-module/membrane arm junction (I-gamma sub-assembly checkpoint);
      NDUFA10 contacts NDUFS2/NDUFS3 on the Q-module peripheral scaffold — overlapping
      Q-module territory with different sub-complex intermediates on BN-PAGE.
    • NDUFA9 has an SDR (short-chain dehydrogenase reductase) fold repurposed for structure;
      NDUFA10 has an NDP-kinase-like Rossmann fold repurposed for structure — both are
      "repurposed catalytic fold" subunits but in different enzymological lineages.
  vs POLG/DGUOK (mtDNA depletion syndromes):
    • NO hepatopathy in NDUFA10 (POLG: ~80%; DGUOK: ~90%)
  vs SURF1/SCO2/COX10/COX15 (COX deficiency):
    • COX activity NORMAL in NDUFA10 — biochemical fingerprint distinction

UNIQUE MOLECULAR SIGNATURE — NDUFA10 NDP-Kinase-Like Rossmann Fold:
  NDUFA10 is the ONLY CI structural subunit with an NDP (nucleoside diphosphate) kinase-like
  Rossmann-fold domain. The Rossmann fold is catalytically INACTIVE (no phosphotransferase
  activity), repurposed entirely for structural Q-module scaffold function. This is distinct
  from NDUFA9 (SDR Rossmann-fold, also catalytically inactive) — both have repurposed
  metabolic enzyme folds, but in different enzyme families (NDP kinase vs short-chain
  dehydrogenase). NDUFA10 Rossmann fold contains the PINK1-phosphorylation site (Ser250)
  which is the only known kinase-responsive CI phosphorylation site at the Q-module.

PINK1-Parkinsonism / NDUFA10 Mechanistic Bridge:
  PINK1 (PTEN-induced kinase 1, 1p36) causes autosomal recessive Parkinson's disease
  (PARK6, OMIM #605909) when biallelic LOF. PINK1 phosphorylates NDUFA10 at Ser250 in the
  Rossmann NDP-kinase domain, promoting CI stability. In PINK1-deficient Drosophila and
  human models, Ser250 dephosphorylation of NDUFA10 → reduced CI activity. This positions
  NDUFA10 as the DIRECT structural CI bridge between the PINK1-Parkinsonism pathway and CI
  deficiency — a unique mechanistic overlap not shared by any other CI-Leigh subunit.

FOUNDER / RECURRENT MUTATIONS:
  p.Arg108Trp   c.322C>T   — Rossmann NDP-kinase domain core; NDUFS2/NDUFS3 contact; severe infantile
  p.Gly271Val   c.812G>T   — kinase-like alpha-helix subdomain; intermediate severity
  p.Thr314Ile   c.941C>T   — C-terminal structural region; moderate, partial CI residual
  p.Pro31Ser    c.91C>T    — near mitochondrial targeting sequence; severe neonatal
  c.IVS4+1G>A              — splice donor intron 4; partial CI residual (~12–20%); intermediate

THERAPY — NDUFA10 / CI-LEIGH SPECIFICS:
  No targeted NDUFA10 Rossmann-fold rescue is clinically available.
  Management follows the CI-Leigh supportive protocol:
    ABSOLUTE contraindications (direct CI inhibitors / mito toxins):
      Metformin — directly inhibits CI at ND1/quinone-binding site (same Q-module territory)
      Valproate  — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression
      Linezolid  — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
      Chloramphenicol — same ribosomal mechanism as linezolid
    CONTRAINDICATED:
      Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH
                        (NDUFA10 Q-module peripheral scaffold failed, CI Q-module assembly disrupted)
    AVOID / HIGH CAUTION:
      Propofol — PRIS + secondary CIV inhibition adds a second ETC bottleneck
      Phenobarbital — secondary CI inhibitor; use LEV first
    LEVEL C cofactors (standard CI supportive):
      Riboflavin (B2) — CI-specific; FMN prosthetic group at NDUFV1
      CoQ10 (ubiquinol) — electron acceptor at quinone site; close to Q-module (NDUFA10 territory)
      Thiamine (B1) — MANDATORY empiric: mimics SLC19A3/BTD (treatable)
      Biotin — MANDATORY empiric: mimics BTD (treatable)
      Succinate — CII bypass; bypasses NDUFA10-failed Q-module CI scaffold entirely
      L-Carnitine — energy metabolism support
    Preferred AED: Levetiracetam (LEV) — renal excretion; no mito toxicity
    Glucose: IV dextrose GIR 6–8 mg/kg/min — NEVER fast
    Anaesthesia: sevoflurane (NOT propofol)
"""

import random
from typing import Any

SEED = 639
GENE = "NDUFA10"
DISEASE = "NDUFA10 Leigh Syndrome — Isolated Complex I Deficiency (42kDa / NDP-Kinase-Like Rossmann-Fold / PINK1-Phosphorylation-Target / Q-Module-Associated Peripheral Subunit)"
OMIM_GENE = "603835"
OMIM_DISEASE = "256000"
CHROMOSOME = "2q37.3"
INHERITANCE = "AR"

# ---------------------------------------------------------------------------
# Synthetic 40-patient cohort  (seed-639, reflects published literature)
# ---------------------------------------------------------------------------

def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    sexes = ["M", "F"]
    mutations = [
        "p.Arg108Trp / c.IVS4+1G>A (compound het)",
        "p.Arg108Trp / p.Arg108Trp (hom, consanguineous)",
        "p.Gly271Val / p.Thr314Ile (compound het)",
        "p.Pro31Ser / p.Arg108Trp (compound het)",
        "p.Thr314Ile / c.IVS4+1G>A (compound het)",
        "p.Gly271Val / c.IVS4+1G>A (compound het)",
        "p.Pro31Ser / p.Gly271Val (compound het)",
        "Novel biallelic LOF (frameshift/splice)",
        "p.Arg108Trp / novel missense (compound het)",
        "p.Thr314Ile / p.Thr314Ile (hom, consanguineous)",
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
            "hypotonia":              rng.random() < 0.86,
            "psychomotor_regression": rng.random() < 0.95,
            "respiratory_compromise": rng.random() < 0.43,
            "seizures":               rng.random() < 0.46,
            "ataxia":                 rng.random() < 0.39,
            "dystonia":               rng.random() < 0.34,
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
        "gene_full_name": "NADH:Ubiquinone Oxidoreductase Subunit A10",
        "also_known_as":  "42kDa subunit / NDP-kinase-like Rossmann-fold Q-module peripheral scaffold / PINK1 phosphorylation target (Ser250)",
        "disease":        DISEASE,
        "omim_gene":      OMIM_GENE,
        "omim_disease":   OMIM_DISEASE,
        "chromosome":     CHROMOSOME,
        "inheritance":    INHERITANCE,
        "protein": {
            "size_aa_precursor": 335,
            "size_aa_mature":    300,
            "size_kda":          38.5,
            "fold":              "NDP-kinase-like Rossmann-fold (catalytically INACTIVE — repurposed for Q-module peripheral structural scaffolding); PINK1 phosphorylation substrate at Ser250",
            "module":            "Q-module peripheral scaffold; contacts NDUFS2 (49kDa, PSST) and NDUFS3 (30kDa) in the Q-module sub-complex",
            "fe_s_cluster":      False,
            "zinc_finger":       False,
            "function":          "Q-module peripheral structural scaffold via inactive NDP-kinase Rossmann fold; contacts NDUFS2 and NDUFS3; loss causes Q-module sub-assembly intermediates on BN-PAGE. UNIQUE: ONLY CI subunit phosphorylated by PINK1 (Parkinson's disease kinase) at Ser250 — mechanistic CI bridge between PINK1-Parkinsonism and isolated CI deficiency.",
        },
        "key_pathway_note": (
            "NDUFA10 is the ONLY nuclear-encoded CI structural subunit that is a direct "
            "phosphorylation target of PINK1 (PTEN-Induced Kinase 1), the autosomal recessive "
            "Parkinson's disease gene. PINK1 phosphorylates NDUFA10 at Ser250 in the "
            "NDP-kinase-like Rossmann-fold domain, promoting CI activity and Q-module stability. "
            "In PINK1-deficient Parkinson's disease models, Ser250 dephosphorylation reduces CI "
            "activity — positioning NDUFA10 as the direct structural bridge between the "
            "PINK1-Parkinsonism pathway and CI deficiency. In NDUFA10 biallelic LOF (Leigh), "
            "the protein is absent — both the Q-module scaffold AND the PINK1-responsive "
            "regulatory mechanism are lost. BN-PAGE shows Q-module sub-assembly intermediates "
            "(partial Q-module + peripheral arm visible — similar to NDUFA9 I-gamma pattern). "
            "CONTRAST: cleaner absent CI in NDUFS7/NDUFS8 (direct Fe-S relay), NDUFB3 "
            "(PP-module scaffold), NDUFA11 (PP-PD boundary). "
            "Isolated CI deficiency 5–20%; CII/CIII/CIV NORMAL."
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
                "feature":        "Q-module sub-assembly intermediates on BN-PAGE (partial Q-module visible)",
                "significance":   "Q-module peripheral scaffold failure — sub-assembly intermediates visible (similar to NDUFA9 I-gamma pattern). CONTRAST cleaner absent CI in NDUFS7/NDUFS8 (Fe-S relay), NDUFB3 (PP-module scaffold), NDUFA11 (PP-PD boundary).",
                "target_gene":    "NDUFS7 / NDUFS8 / NDUFB3 / NDUFA11",
                "target_freq_pct": 0,
            },
            {
                "feature":        "NDP-KINASE-LIKE ROSSMANN FOLD — UNIQUE among all nuclear CI subunits",
                "significance":   "Only CI structural subunit with NDP-kinase-like domain (catalytically inactive) AND PINK1-phosphorylation at Ser250. Distinct from NDUFA9 (SDR Rossmann fold) — both repurpose metabolic folds structurally but in different enzyme families. WES/gene panel essential.",
                "target_gene":    "NDUFA9 (SDR fold) / All other CI-Leigh subunits",
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
            "Metformin — directly inhibits CI at ND1/quinone-binding site (same Q-module territory as NDUFA10)",
            "Valproate (VPA) — triple mechanism: CoA sequestration + POLG inhibition + ND-subunit expression block",
            "Linezolid — inhibits mitochondrial 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits",
            "Chloramphenicol — same mt-ribosomal mechanism as linezolid",
        ],
        "contraindicated": [
            "Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH (NDUFA10 Q-module peripheral scaffold failed, CI Q-module assembly disrupted)",
        ],
        "preferred_treatments": [
            "LEV (levetiracetam) — AED first-line: renal excretion, no mito toxicity",
            "Riboflavin (B2) — CI-specific cofactor; FMN at NDUFV1 (upstream of NDUFA10 Q-module scaffold)",
            "CoQ10 (ubiquinol) — electron acceptor at quinone site; same Q-module territory as NDUFA10",
            "Thiamine (B1) — MANDATORY empiric: SLC19A3/BTD treatable mimic",
            "Biotin — MANDATORY empiric: BTD treatable mimic",
            "Succinate — CII bypass; bypasses NDUFA10-failed Q-module CI scaffold entirely",
            "L-Carnitine — energy metabolism support",
            "IV dextrose GIR 6–8 mg/kg/min — never fast",
            "Sevoflurane (not propofol) for anaesthesia",
        ],
        "key_references": [
            "Kohda M et al. 2016 Nat Genet — comprehensive mtDNA + nuclear CI gene screening; NDUFA10 biallelic variants in Leigh syndrome",
            "Morais VA et al. 2009 EMBO Rep — PINK1 phosphorylates NDUFA10 at Ser250; promotes CI activity; Drosophila and human cell evidence",
            "Fassone E & Rahman S 2012 JMedGenet — CI genetics review (nuclear-encoded subunits including NDUFA10)",
            "Sazanov LA 2015 NatRevMolCellBiol — CI cryo-EM structure: Q-module; NDUFA10 peripheral scaffold contacts NDUFS2/NDUFS3",
            "Guerrero-Castillo S et al. 2017 CellMetab — CI assembly intermediates; Q-module sub-complex dynamics; NDUFA10 sub-assembly",
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
                "the same Q-module territory where NDUFA10 scaffolds NDUFS2/NDUFS3. "
                "In NDUFA10 deficiency CI activity is already 5–20%. Metformin "
                "precipitates fatal lactic crisis. NEVER use in any CI-Leigh."
            ),
        },
        {
            "term": "Valproate (VPA) — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Three independent mechanisms: (1) CoA sequestration → energy deficit; "
                "(2) POLG inhibition → mtDNA depletion (worsens mt-encoded ND subunit "
                "availability, all 7 mtDNA-encoded ND subunits affected); "
                "(3) direct ND-subunit expression block. All three compound an already "
                "critically reduced CI in NDUFA10 deficiency."
            ),
        },
        {
            "term": "Linezolid — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Inhibits mitochondrial 23S rRNA (mt-ribosome) → blocks synthesis of ALL "
                "seven mt-encoded ND subunits (ND1–ND6, ND4L). In NDUFA10 deficiency the "
                "Q-module peripheral scaffold has collapsed; removing all mt ND subunits "
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
                "that must be re-oxidised via CI. In NDUFA10 deficiency the Q-module "
                "peripheral scaffold has collapsed → CI Q-module assembly disrupted → "
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
                "Secondary CI inhibitor — adds to the primary NDUFA10-driven CI deficit. "
                "Use LEV (levetiracetam) first for seizure control. Reserve phenobarbital "
                "for refractory seizures with close metabolic monitoring."
            ),
        },
        {
            "term": "Succinate — Level C (CII bypass)",
            "category": "treatment",
            "detail": (
                "Succinate feeds directly into Complex II (SDHA, fully nuclear), bypassing "
                "the NDUFA10-failed CI Q-module scaffold entirely. Electrons enter the ETC "
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
            "term": "NDUFA10 — NDP-Kinase-Like Rossmann Fold Q-Module Peripheral Scaffold",
            "category": "gene_concept",
            "detail": (
                "NDUFA10 (NADH:Ubiquinone Oxidoreductase Subunit A10) is a ~335-aa "
                "nuclear-encoded protein (~38.5 kDa, ~300 aa mature after MTS cleavage) "
                "containing a nucleoside diphosphate (NDP) kinase-like Rossmann fold "
                "that is catalytically INACTIVE. The domain is repurposed for Q-module "
                "peripheral structural scaffolding, contacting NDUFS2 (49kDa/PSST) and "
                "NDUFS3 (30kDa) in the Q-module sub-complex. "
                "Chromosome 2q37.3. OMIM Gene *603835."
            ),
        },
        {
            "term": "PINK1 Phosphorylation at Ser250 — NDUFA10 Unique Regulatory Link",
            "category": "gene_concept",
            "detail": (
                "NDUFA10 is the ONLY CI structural subunit that is a direct phosphorylation "
                "substrate of PINK1 (PTEN-Induced Kinase 1, 1p36), the Parkinson's disease "
                "gene (PARK6, OMIM #605909). PINK1 phosphorylates Ser250 in the Rossmann "
                "NDP-kinase domain, promoting CI activity and Q-module stability. In "
                "PINK1-deficient cells (Parkinson's disease model), Ser250 dephosphorylation "
                "destabilises NDUFA10 → reduced CI activity. In NDUFA10 biallelic LOF "
                "(Leigh), BOTH the Q-module scaffold AND the PINK1-responsive CI regulation "
                "are absent simultaneously — a mechanistic overlap unique among CI-Leigh genes."
            ),
        },
        {
            "term": "BN-PAGE Pattern in NDUFA10 Deficiency",
            "category": "gene_concept",
            "detail": (
                "Blue-native PAGE shows Q-module sub-assembly intermediates (partial Q-module "
                "+ peripheral arm assemblies visible) in NDUFA10 LOF — the Q-module "
                "peripheral scaffold has collapsed, releasing partial Q-module sub-complexes. "
                "Pattern resembles NDUFA9 (I-gamma, also a Q-module-associated subunit with "
                "repurposed Rossmann fold) more than the cleaner absent CI in NDUFS7/NDUFS8 "
                "(direct Fe-S relay block), NDUFB3 (PP-module scaffolding loss), or NDUFA11 "
                "(PP-PD membrane arm inter-module boundary)."
            ),
        },
        {
            "term": "NDUFA10 Genotype–Phenotype Correlations",
            "category": "gene_concept",
            "detail": (
                "Rossmann NDP-kinase domain core / NDUFS2-NDUFS3 contact (p.Arg108Trp): "
                "severe infantile onset — scaffold contact lost, Q-module assembly collapses. "
                "Kinase-like alpha-helix subdomain (p.Gly271Val): intermediate; partial CI "
                "residual retained. C-terminal structural region (p.Thr314Ile): moderate; "
                "partial CI residual, consanguineous kindreds. Near MTS (p.Pro31Ser): severe "
                "neonatal; mitochondrial targeting disrupted. Splice (c.IVS4+1G>A): partial "
                "CI residual (~12–20%); intermediate severity, episodic course."
            ),
        },
        {
            "term": "NDP-Kinase vs SDR Fold — NDUFA10 vs NDUFA9 Repurposed Scaffolds",
            "category": "gene_concept",
            "detail": (
                "NDUFA10 (NDP-kinase-like Rossmann fold, Q-module peripheral, 2q37.3) and "
                "NDUFA9 (SDR/short-chain dehydrogenase Rossmann fold, Q-module/membrane arm "
                "junction, 12q24.31) are both 'repurposed catalytic fold' CI subunits. "
                "Both contain Rossmann folds from different enzyme families, both are "
                "catalytically inactive, both contribute to Q-module assembly checkpoints. "
                "Key distinction: NDUFA10 is at the Q-module peripheral scaffold (NDUFS2/"
                "NDUFS3 contacts) while NDUFA9 is at the Q-module/membrane arm junction "
                "(I-gamma sub-assembly). BN-PAGE both show Q-module intermediates, but "
                "sub-complex composition differs. Only NDUFA10 has PINK1 phosphorylation "
                "regulatory link. Clinical distinction requires WES."
            ),
        },
        {
            "term": "OMIM *603835 / #256000",
            "category": "gene_concept",
            "detail": (
                "NDUFA10 gene: OMIM *603835. Primary disease: Leigh Syndrome OMIM #256000. "
                "Inheritance: AR biallelic LOF. Chromosome: 2q37.3."
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
                "CI subunit genes can cause CI-Leigh. NDUFA10 (NDP-kinase-like Rossmann-fold, "
                "Q-module peripheral scaffold, PINK1 phosphorylation target) is one of the "
                "characterised nuclear-encoded CI Q-module-associated peripheral subunit causes."
            ),
        },
        {
            "term": "Isolated CI Deficiency — NDUFA10",
            "category": "disease_concept",
            "detail": (
                "NDUFA10 LOF produces isolated Complex I deficiency (5–20% residual), with "
                "CII, CIII, CIV all normal — the 'biochemical fingerprint' of CI-Leigh. "
                "Enzyme histochemistry: COX normal (CIV), SDH normal (CII). "
                "Respiratory chain analysis is essential before WES."
            ),
        },
        {
            "term": "NDUFA10 — PINK1-Parkinsonism / CI-Leigh Mechanistic Bridge",
            "category": "disease_concept",
            "detail": (
                "The PINK1-NDUFA10 phosphorylation axis connects two major neurological "
                "disease categories: Parkinson's disease (PINK1 LOF → Ser250 "
                "dephosphorylation → reduced CI activity) and Leigh syndrome (NDUFA10 "
                "biallelic LOF → absent Q-module scaffold → isolated CI deficiency). "
                "This mechanistic bridge makes NDUFA10 a unique gene spanning infantile "
                "mitochondrial encephalopathy (Leigh) and adult neurodegeneration "
                "(PINK1-Parkinson's) via the same protein domain (Rossmann fold Ser250 site)."
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "Prescribing Safety Pocket Card — NDUFA10 Leigh",
            "category": "prescribing_safety",
            "detail": (
                "ABSOLUTE CI    : Metformin · Valproate · Linezolid · Chloramphenicol\n"
                "CONTRAINDICATED: Ketogenic diet (NADH cannot be re-oxidised — Q-module peripheral scaffold failed)\n"
                "AVOID          : Propofol (PRIS + CIV block → dual ETC failure)\n"
                "HIGH CAUTION   : Phenobarbital (secondary CI inhibitor; use LEV first)\n"
                "PREFERRED AED  : LEV (levetiracetam) — renal, no mito toxicity\n"
                "ANAESTHESIA    : Sevoflurane (NOT propofol)\n"
                "GLUCOSE        : IV dextrose GIR 6–8 mg/kg/min — NEVER fast\n"
                "COFACTORS      : Riboflavin B2 · CoQ10 (ubiquinol, Q-module territory) · Thiamine B1* · Biotin* · Succinate (CII bypass) · Carnitine\n"
                "                 (* MANDATORY empiric before genetics: rules out SLC19A3/BTD)\n"
                "MECHANISM NOTE : NDUFA10 42kDa NDP-kinase-like Rossmann-fold Q-module peripheral scaffold; contacts NDUFS2 + NDUFS3; PINK1 phosphorylation target Ser250; Q-module sub-assembly intermediates on BN-PAGE"
            ),
        },
    ]

    return {
        "pharmacology":       pharmacology,
        "gene_concepts":      gene_concepts,
        "disease_concepts":   disease_concepts,
        "prescribing_safety": prescribing_safety,
    }
