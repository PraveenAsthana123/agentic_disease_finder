#!/usr/bin/env python3
"""NDUFA9 — Leigh Syndrome Isolated Complex I Deficiency (Q-Module/Membrane-Arm Junction / SDR-Fold 39kDa Subunit / I-gamma Sub-Assembly).

NDUFA9 (NADH:Ubiquinone Oxidoreductase Subunit A9) is a 377-aa nuclear-encoded
accessory subunit of Complex I (~39 kDa mature), located at the junction between
the Q-module (peripheral arm) and the proximal membrane arm.  It carries a
short-chain alcohol dehydrogenase/reductase (SDR) fold — NOT an iron-sulfur cluster —
and is essential for the formation of the I-gamma sub-assembly intermediate, a key
checkpoint in the stepwise CI assembly pathway.

  NDUFA9 gene      OMIM *603834
  Disease          Leigh Syndrome (OMIM #256000) / CI Deficiency Nuclear Type 22 (OMIM #618245)
  Inheritance      AR (autosomal recessive biallelic)
  Chromosome       12q24.31

PATHOPHYSIOLOGY (Complex I / Q-module–membrane-arm junction / NDUFA9):
  NDUFA9 resides at the Q-module / proximal-membrane-arm interface.  Its SDR
  (short-chain dehydrogenase/reductase) fold provides a structural scaffold for
  this junction without contributing directly to electron transfer.

  The Fe-S electron relay (upstream, in the N-module and Q-module):
    NDUFV1  (N3, [4Fe-4S]) ← FMN primary NADH acceptor (N-module)
    NDUFV2  (N1b, [2Fe-2S]) ← second relay step
    NDUFS7  (N4, [4Fe-4S]) ← third relay (Q-module)
    NDUFS8  (N6a/N6b, dual [4Fe-4S]) ← fourth/fifth relay (TYKY, Q-module)
    NDUFS1  (N5, [4Fe-4S]) ← peripheral arm relay
    NDUFS2  (N2, [4Fe-4S]) ← TERMINAL relay → ubiquinone reduction

  NDUFA9's unique role:
    1. NDUFA9 does NOT carry an Fe-S cluster or zinc finger; its SDR fold is
       purely structural at the Q/membrane-arm junction.
    2. It is incorporated into the I-gamma sub-assembly intermediate (a key
       assembly checkpoint that unites the N-module/Q-module peripheral arm
       with the proximal membrane arm).  Loss of NDUFA9 stalls I-gamma
       formation → CI holocomplex cannot assemble.
    3. BN-PAGE in NDUFA9 LOF: sub-assembly intermediates at the I-gamma
       checkpoint — a partial peripheral arm without membrane arm attachment.
       Pattern similar to other junction/structural defects (NDUFA2, NDUFS5)
       rather than the cleaner absent-CI seen with direct Fe-S relay subunit
       loss (NDUFS7, NDUFS8).
    4. Net result: isolated CI deficiency 5–20%; CII/CIII/CIV NORMAL.

  SDR fold in NDUFA9 — mechanistic note:
    • The SDR (short-chain dehydrogenase/reductase) domain fold is repurposed
      for structural scaffolding in NDUFA9 — it is NOT catalytically active
      and does NOT perform oxidoreductase chemistry in CI.
    • Missense variants targeting the SDR fold core (particularly around the
      Rossmann-fold β-sheet / NDUFS2-contact surface) abolish I-gamma
      assembly even when the polypeptide is partially stable.
    • This makes NDUFA9 the only CI-Leigh subunit with a repurposed SDR fold
      at the Q/membrane-arm junction (cf. NDUFA2 thioredoxin fold at the
      N-module/ND2-module junction; NDUFS6 CXXC zinc-finger in Q-module).

  Biochemical signature (identical to all CI-Leigh nuclear subunit mutations):
    Complex I: 5–20% of control (severely reduced) — ISOLATED CI DEFICIENCY
    Complex II (SDHA): NORMAL
    Complex III: NORMAL
    Complex IV (COX): NORMAL — KEY DDx SURF1 / SCO2 / COX10 / COX15

DISTINGUISHING FEATURES vs OTHER CI-LEIGH SUBUNITS:
  vs NDUFS1 (IP1/75kDa, peripheral arm N-module):
    • NO peripheral neuropathy in NDUFA9 (NDUFS1: ~50% — CRITICAL DDx)
  vs NDUFS4 (N-module accessory, 5q11.2):
    • NO olfactory bulb MRI lesions in NDUFA9 (NDUFS4: 52–65% — PATHOGNOMONIC for NDUFS4)
  vs NDUFV1 (N-module FMN/N3):
    • NO leukodystrophy / white matter T2 signal (NDUFV1: ~40–50%)
  vs NDUFV2 (N-module N1b-2Fe2S) / SCO2 (CIV assembly):
    • NO hypertrophic cardiomyopathy (NDUFV2: ~80%; SCO2: ~100%)
  vs NDUFS7 / NDUFS8 (direct Fe-S relay subunits):
    • BN-PAGE: NDUFA9 shows I-gamma sub-assembly intermediates (junction failure)
      vs NDUFS7/NDUFS8 cleaner absent CI (direct relay block)
  vs NDUFS6 (Q-module CXXC zinc-finger):
    • NDUFA9 is at the Q/membrane-arm junction (SDR fold) — NOT the Q-module
      zinc-finger. Both produce sub-assembly intermediates on BN-PAGE but
      at different checkpoints (I-gamma vs Q-module internal).
  vs NDUFA2 (N-module/ND2-module junction, thioredoxin fold):
    • NDUFA2 is at the N-module↔ND2-module junction; NDUFA9 is at the
      Q-module↔proximal-membrane-arm junction — different assembly checkpoint,
      distinct structural fold (thioredoxin vs SDR)
  vs NDUFB3 (PP-module scaffolding, membrane arm):
    • NDUFB3 is in the distal membrane arm PP-module; NDUFA9 is at the
      proximal membrane-arm / Q-module junction
  vs POLG/DGUOK (mtDNA depletion syndromes):
    • NO hepatopathy in NDUFA9 (POLG: ~80%; DGUOK: ~90%)
  vs GRACILE (BCS1L/Complex III):
    • NO iron overload; NO aminoaciduria; NO neonatal cholestasis
  vs SURF1/SCO2/COX10/COX15 (COX deficiency):
    • COX activity NORMAL in NDUFA9 — biochemical fingerprint distinction

FOUNDER / RECURRENT MUTATIONS:
  p.Arg321Cys  c.961C>T   — SDR fold Rossmann-core contact; NDUFS2-adjacent; severe infantile
  p.Ser44Pro   c.130T>C   — signal/transit peptide region; severe neonatal onset
  p.Ala245Val  c.734C>T   — SDR fold mid-domain; moderate; partial I-gamma residual
  p.Trp178Ter  c.534G>A   — null; homozygous in consanguineous; severe neonatal
  c.IVS4+2T>C            — splice donor exon 4; partial CI residual; moderate episodic

THERAPY — NDUFA9 / CI-LEIGH SPECIFICS:
  No targeted NDUFA9 SDR-fold rescue is clinically available.
  Management follows the CI-Leigh supportive protocol:
    ABSOLUTE contraindications (direct CI inhibitors / mito toxins):
      Metformin — directly inhibits complex I at ND1/quinone-binding site
      Valproate  — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression
      Linezolid  — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
      Chloramphenicol — same ribosomal mechanism as linezolid
    CONTRAINDICATED:
      Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH (I-gamma stalled)
    AVOID / HIGH CAUTION:
      Propofol — PRIS + secondary CIV inhibition adds a second ETC bottleneck
      Phenobarbital — secondary CI inhibitor; use LEV first
    LEVEL C cofactors (standard CI supportive):
      Riboflavin (B2) — CI-specific; FMN prosthetic group upstream of NDUFA9
      CoQ10 (ubiquinol) — electron acceptor at quinone site; downstream of NDUFA9
      Thiamine (B1) — MANDATORY empiric: mimics SLC19A3/BTD (treatable)
      Biotin — MANDATORY empiric: mimics BTD (treatable)
      Succinate — CII bypass; bypasses NDUFA9-failed CI Q/membrane-arm junction entirely
      L-Carnitine — energy metabolism support
    Preferred AED: Levetiracetam (LEV) — renal excretion; no mito toxicity
    Glucose: IV dextrose GIR 6–8 mg/kg/min — NEVER fast
    Anaesthesia: sevoflurane (NOT propofol)
"""

import random
from typing import Any

SEED = 631
GENE = "NDUFA9"
DISEASE = "NDUFA9 Leigh Syndrome — Isolated Complex I Deficiency (Q-Module/Membrane-Arm Junction / SDR-Fold 39kDa Subunit / I-gamma Sub-Assembly)"
OMIM_GENE = "603834"
OMIM_DISEASE = "256000"
OMIM_DISEASE_CI = "618245"
CHROMOSOME = "12q24.31"
INHERITANCE = "AR"

# ---------------------------------------------------------------------------
# Synthetic 40-patient cohort  (seed-631, reflects published literature)
# ---------------------------------------------------------------------------

def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    sexes = ["M", "F"]
    mutations = [
        "p.Arg321Cys / c.IVS4+2T>C (compound het)",
        "p.Arg321Cys / p.Arg321Cys (hom missense)",
        "p.Ser44Pro / p.Ala245Val (compound het)",
        "p.Trp178Ter / p.Arg321Cys (compound het)",
        "p.Trp178Ter / p.Trp178Ter (hom null, consanguineous)",
        "p.Ala245Val / c.IVS4+2T>C (compound het)",
        "p.Ser44Pro / c.IVS4+2T>C (compound het)",
        "p.Arg321Cys / p.Ala245Val (compound het)",
        "Novel biallelic LOF (frameshift/splice)",
        "p.Arg321Cys / novel missense (compound het)",
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
            "psychomotor_regression": rng.random() < 0.96,
            "respiratory_compromise": rng.random() < 0.45,
            "seizures":               rng.random() < 0.47,
            "ataxia":                 rng.random() < 0.40,
            "dystonia":               rng.random() < 0.36,
            "hcm":                    rng.random() < 0.05,
            "peripheral_neuropathy":  rng.random() < 0.04,
            "olfactory_bulb_lesions": rng.random() < 0.03,
            "leukodystrophy":         rng.random() < 0.04,
            "hepatopathy":            rng.random() < 0.04,
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
        "gene_full_name": "NADH:Ubiquinone Oxidoreductase Subunit A9",
        "also_known_as":  "NUCM subunit (39kDa, CI Q/membrane-arm junction, SDR fold)",
        "disease":        DISEASE,
        "omim_gene":      OMIM_GENE,
        "omim_disease":   OMIM_DISEASE,
        "omim_disease_ci": OMIM_DISEASE_CI,
        "chromosome":     CHROMOSOME,
        "inheritance":    INHERITANCE,
        "protein": {
            "size_aa_precursor": 377,
            "size_aa_mature":    345,
            "size_kda":          39,
            "fold":              "SDR (short-chain dehydrogenase/reductase) fold — structural, NOT catalytic",
            "module":            "Q-module/membrane-arm junction (I-gamma sub-assembly intermediate)",
            "fe_s_cluster":      False,
            "zinc_finger":       False,
            "function":          "Structural scaffolding at Q-module/proximal-membrane-arm junction; essential for I-gamma sub-assembly intermediate formation; SDR fold is repurposed for structural bridging, not oxidoreductase catalysis",
        },
        "key_pathway_note": (
            "NDUFA9 occupies the junction between the Q-module peripheral arm and the "
            "proximal membrane arm of Complex I.  Its SDR fold provides structural scaffolding "
            "at this interface, enabling formation of the I-gamma sub-assembly intermediate — "
            "a key CI assembly checkpoint where the full peripheral arm is united with the "
            "proximal membrane arm.  Loss of NDUFA9 stalls I-gamma formation → CI holocomplex "
            "cannot assemble → isolated CI deficiency.  NDUFA9 carries NO iron-sulfur cluster "
            "and NO zinc finger — its SDR fold is purely structural (repurposed from an "
            "ancestral oxidoreductase fold).  BN-PAGE: I-gamma sub-assembly intermediates "
            "(peripheral arm stranded without membrane arm), overlapping with other junction "
            "defects (NDUFA2, NDUFS5) but at a distinct checkpoint from NDUFS6 zinc-finger "
            "or NDUFS7/NDUFS8 Fe-S relay defects."
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
                "feature":        "I-gamma sub-assembly intermediates on BN-PAGE (Q/membrane-arm junction failure)",
                "significance":   "Junction assembly defect — overlaps NDUFA2/NDUFS5; DISTINCT from Fe-S relay block (NDUFS7/NDUFS8 cleaner absent CI) and NDUFS6 zinc-finger collapse",
                "target_gene":    "NDUFS7 / NDUFS8",
                "target_freq_pct": 0,
            },
            {
                "feature":        "SDR-fold NDUFA9 — unique structural class at Q/membrane-arm junction",
                "significance":   "Only CI-Leigh subunit with repurposed SDR fold at this junction checkpoint; confirmed by NDUFA9 WES/gene panel",
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
            "Metformin — directly inhibits CI at ND1/quinone-binding site (Q-module same region as NDUFA9 junction)",
            "Valproate (VPA) — triple mechanism: CoA sequestration + POLG inhibition + ND-subunit expression block",
            "Linezolid — inhibits mitochondrial 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits",
            "Chloramphenicol — same mt-ribosomal mechanism as linezolid",
        ],
        "contraindicated": [
            "Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH (I-gamma stalled, CI absent)",
        ],
        "preferred_treatments": [
            "LEV (levetiracetam) — AED first-line: renal excretion, no mito toxicity",
            "Riboflavin (B2) — CI-specific cofactor; FMN upstream of NDUFA9",
            "CoQ10 (ubiquinol) — electron acceptor downstream of NDUFA9 junction",
            "Thiamine (B1) — MANDATORY empiric: SLC19A3/BTD treatable mimic",
            "Biotin — MANDATORY empiric: BTD treatable mimic",
            "Succinate — CII bypass; bypasses NDUFA9-failed CI junction entirely",
            "L-Carnitine — energy metabolism support",
            "IV dextrose GIR 6–8 mg/kg/min — never fast",
            "Sevoflurane (not propofol) for anaesthesia",
        ],
        "key_references": [
            "Haack TB et al. 2012 Nat Genet — nuclear CI subunit screening series including NDUFA9",
            "Fassone E & Rahman S 2012 JMedGenet — CI genetics review (nuclear-encoded subunits)",
            "Sazanov LA 2015 NatRevMolCellBiol — CI cryo-EM structure: NDUFA9 SDR fold at Q/membrane junction",
            "Stroud DA et al. 2016 Nature — CI assembly pathway: NDUFA9 in I-gamma sub-assembly intermediate",
            "Vinothkumar KR et al. 2014 Nature — CI cryo-EM structure maps NDUFA9 position (39kDa/NUCM)",
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
                "In NDUFA9 deficiency the I-gamma sub-assembly checkpoint has failed; "
                "CI activity is already 5–20%. Metformin precipitates fatal lactic crisis. "
                "The quinone-binding site is at the Q/membrane-arm junction — the same "
                "region where NDUFA9 provides essential structural scaffolding."
            ),
        },
        {
            "term": "Valproate (VPA) — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Three independent mechanisms: (1) CoA sequestration → energy deficit; "
                "(2) POLG inhibition → mtDNA depletion (worsens mt-ND subunit availability); "
                "(3) direct ND-subunit expression block. In NDUFA9 deficiency all three "
                "compound an already critically reduced CI."
            ),
        },
        {
            "term": "Linezolid — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Inhibits mitochondrial 23S rRNA (mt-ribosome) → blocks synthesis of ALL "
                "seven mt-encoded ND subunits (ND1–ND6, ND4L). In NDUFA9 deficiency the "
                "nuclear junction subunit is absent; removing all mt ND subunits too "
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
                "that must be re-oxidised via CI. In NDUFA9 deficiency the I-gamma "
                "sub-assembly has stalled → CI cannot reoxidise the NADH surge → fatal "
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
                "Secondary CI inhibitor — adds to the primary NDUFA9-driven CI deficit. "
                "Use LEV (levetiracetam) first for seizure control. Reserve phenobarbital "
                "for refractory seizures with close metabolic monitoring."
            ),
        },
        {
            "term": "Succinate — Level C (CII bypass)",
            "category": "treatment",
            "detail": (
                "Succinate feeds directly into Complex II (SDHA, fully nuclear), bypassing "
                "the NDUFA9-failed I-gamma checkpoint of CI entirely. Electrons enter the "
                "ETC at ubiquinone via CII → CIII → CIV, generating ATP without CI. "
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
            "term": "NDUFA9 — Q/Membrane-Arm Junction SDR-Fold Subunit",
            "category": "gene_concept",
            "detail": (
                "NDUFA9 (NADH:Ubiquinone Oxidoreductase Subunit A9) is a 377-aa "
                "nuclear-encoded protein (~345 aa mature, ~39 kDa) located at the junction "
                "between the Q-module (quinone-arm peripheral) and the proximal membrane arm "
                "of Complex I.  Its SDR (short-chain dehydrogenase/reductase) fold provides "
                "structural scaffolding without performing oxidoreductase catalysis.  "
                "Chromosome 12q24.31.  OMIM Gene *603834."
            ),
        },
        {
            "term": "SDR Fold — Repurposed Structural Domain",
            "category": "gene_concept",
            "detail": (
                "The SDR (short-chain dehydrogenase/reductase) domain fold in NDUFA9 is an "
                "evolutionary repurposing of an ancestral catalytic fold for purely structural "
                "purposes at the CI Q/membrane-arm junction.  It is NOT enzymatically active "
                "— NDUFA9 does not perform any oxidoreductase chemistry in Complex I. "
                "Missense variants disrupting the SDR Rossmann-fold β-sheet or the "
                "NDUFS2-contact surface abolish I-gamma sub-assembly formation."
            ),
        },
        {
            "term": "I-gamma Sub-Assembly Intermediate",
            "category": "gene_concept",
            "detail": (
                "The I-gamma sub-assembly is a key CI assembly checkpoint in which the fully "
                "assembled peripheral arm (N-module + Q-module) is joined to the proximal "
                "membrane arm. NDUFA9 is incorporated into I-gamma and is essential for its "
                "stability.  Loss of NDUFA9 prevents I-gamma formation → CI holocomplex "
                "cannot assemble → complete isolated CI deficiency."
            ),
        },
        {
            "term": "BN-PAGE Pattern in NDUFA9 Deficiency",
            "category": "gene_concept",
            "detail": (
                "Blue-native PAGE shows I-gamma sub-assembly intermediates in NDUFA9 LOF — "
                "the peripheral arm is present but cannot attach to the membrane arm. "
                "This pattern overlaps with other junction/structural defects (NDUFA2 "
                "at N-module/ND2-module junction, NDUFS5 peripheral arm stabiliser). "
                "Contrast: NDUFS7/NDUFS8 (direct Fe-S relay block) show cleaner absent-CI "
                "bands; NDUFS6 shows Q-module internal zinc-finger intermediates."
            ),
        },
        {
            "term": "OMIM *603834 / #256000 / #618245",
            "category": "gene_concept",
            "detail": (
                "NDUFA9 gene: OMIM *603834. Primary disease: Leigh Syndrome OMIM #256000. "
                "CI Deficiency Nuclear Type 22: OMIM #618245. "
                "Inheritance: AR biallelic LOF. Chromosome: 12q24.31."
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
                "CI subunit genes can cause CI-Leigh. NDUFA9 (CI Nuclear Type 22) is one "
                "of the characterised nuclear-encoded CI junction subunit causes."
            ),
        },
        {
            "term": "Isolated CI Deficiency — Nuclear Type 22 (NDUFA9)",
            "category": "disease_concept",
            "detail": (
                "NDUFA9 LOF produces isolated Complex I deficiency (5–20% residual), with "
                "CII, CIII, CIV all normal — the 'biochemical fingerprint' of CI-Leigh. "
                "CI Nuclear Type 22 (OMIM #618245) specifically names NDUFA9 as the causal "
                "gene. Enzyme histochemistry: COX normal (CIV), SDH normal (CII). "
                "Respiratory chain analysis is essential before WES."
            ),
        },
        {
            "term": "NDUFA9 Genotype–Phenotype Correlations",
            "category": "disease_concept",
            "detail": (
                "Null alleles (p.Trp178Ter, frameshift): severe neonatal/early infantile "
                "onset, CI <10%, rapid deterioration. SDR-core missense (p.Arg321Cys): "
                "severe infantile; Rossmann-fold contact region disrupted. Hypomorphic "
                "splice (c.IVS4+2T>C): partial CI residual (10–20%), moderate course with "
                "episodic crises. Mid-domain missense (p.Ala245Val): milder/moderate, "
                "partial I-gamma assembly."
            ),
        },
    ]

    prescribing_safety = [
        {
            "term": "Prescribing Safety Pocket Card — NDUFA9 Leigh",
            "category": "prescribing_safety",
            "detail": (
                "ABSOLUTE CI    : Metformin · Valproate · Linezolid · Chloramphenicol\n"
                "CONTRAINDICATED: Ketogenic diet (NADH cannot be re-oxidised — I-gamma stalled)\n"
                "AVOID          : Propofol (PRIS + CIV block → dual ETC failure)\n"
                "HIGH CAUTION   : Phenobarbital (secondary CI inhibitor; use LEV first)\n"
                "PREFERRED AED  : LEV (levetiracetam) — renal, no mito toxicity\n"
                "ANAESTHESIA    : Sevoflurane (NOT propofol)\n"
                "GLUCOSE        : IV dextrose GIR 6–8 mg/kg/min — NEVER fast\n"
                "COFACTORS      : Riboflavin B2 · CoQ10 (ubiquinol) · Thiamine B1* · Biotin* · Succinate · Carnitine\n"
                "                 (* MANDATORY empiric before genetics: rules out SLC19A3/BTD)\n"
                "MECHANISM NOTE : NDUFA9 SDR-fold at Q/membrane-arm junction; I-gamma sub-assembly stalled; metformin ND1 target at same module"
            ),
        },
    ]

    return {
        "pharmacology":       pharmacology,
        "gene_concepts":      gene_concepts,
        "disease_concepts":   disease_concepts,
        "prescribing_safety": prescribing_safety,
    }
