#!/usr/bin/env python3
"""NDUFB3 — Leigh Syndrome Isolated Complex I Deficiency (PP-Module B12 Subunit / Proximal Membrane Arm / First Nuclear CI Mutation).

NDUFB3 (NADH:Ubiquinone Oxidoreductase Subunit B3, also called B12 in bovine nomenclature)
is a small accessory subunit of Complex I located in the proximal pump (PP) module of the
membrane arm.  At 98 aa precursor (~11 kDa), NDUFB3 is structurally essential for assembly
of the PP-module and the membrane arm as a whole.  It carries NO Fe-S cluster; its role is
purely structural within the proximal membrane arm sub-complex.

  NDUFB3 gene      OMIM *603839
  Disease          Leigh Syndrome (OMIM #256000)
  Inheritance      AR (autosomal recessive biallelic)
  Chromosome       2q31.3

HISTORICAL NOTE — FIRST NUCLEAR CI MUTATION EVER IDENTIFIED:
  Andreu et al. 1999 (NatGenet) identified p.Trp22Arg in NDUFB3 as the first nuclear-encoded
  CI subunit mutation ever reported in humans.  This discovery established the paradigm that
  nuclear-encoded CI subunits could cause Leigh syndrome and CI deficiency.  Prior to 1999,
  CI defects were attributed only to mitochondrial DNA mutations.  NDUFB3 thus has singular
  historical importance in the genetics of mitochondrial disease.

PATHOPHYSIOLOGY (Complex I / PP-module / NDUFB3):
  The membrane arm of Complex I is divided into three sub-modules:
    PP-module (proximal pump):  ND2/ND3/ND6 + accessory subunits including NDUFB3/B12
    PD-module (distal pump):    ND4/ND5 + NDUFB6/B15/B16/NDUFA10
    N-module (peripheral arm):  NDUFV1/NDUFV2/NDUFS1/NDUFS2/NDUFS3/NDUFS4/NDUFS5/NDUFS7/NDUFS8

  NDUFB3 (B12) anchors to the outer face of the PP-module.  It contacts the ND2/ND3
  sub-complex directly and is required for stable membrane arm assembly.  Without NDUFB3,
  the PP-module is structurally compromised and the full 1-MDa CI holocomplex cannot form.

  Mechanistic contrast with N-module structural subunits (NDUFA2, NDUFS3, NDUFS5):
    • N-module structural failures (NDUFA2/NDUFS3/NDUFS4/NDUFS5): BN-PAGE shows
      sub-assembly INTERMEDIATES (N-module partially separated from membrane arm).
      Pattern: visible accumulation of partial sub-assemblies.
    • PP-module failure (NDUFB3): BN-PAGE shows ABSENT or near-absent CI with FEWER
      sub-assembly intermediates — the PP-module scaffolding loss disrupts the whole
      membrane arm, leaving little stable intermediate accumulation.
      Pattern: cleaner absent CI, more similar to Fe-S relay defects (NDUFS7/NDUFS8)
      but WITHOUT the upstream electron relay block.

  The Fe-S electron relay (entirely within the N/Q-modules — UNAFFECTED in NDUFB3):
    NDUFV1  (N3, [4Fe-4S]) ← FMN primary NADH acceptor
    NDUFV2  (N1b, [2Fe-2S]) ← second relay step
    NDUFS7  (N4, [4Fe-4S]) ← third relay (Q-module junction)
    NDUFS8  (N6a/N6b, [4Fe-4S]) ← fourth/fifth relay (TYKY)
    NDUFS1  (N5, [4Fe-4S]) ← peripheral arm relay
    NDUFS2  (N2, [4Fe-4S]) ← terminal carrier → ubiquinone

  In NDUFB3 deficiency: NO Fe-S cluster defect.  The relay is intact but CI cannot
  assemble, so NADH→ubiquinone transfer fails by structural absence, not relay block.

  Biochemical signature (IDENTICAL to all CI-Leigh nuclear subunit mutations):
    Complex I: 5–20% of control (severely reduced) — ISOLATED CI DEFICIENCY
    Complex II (SDHA): NORMAL
    Complex III: NORMAL
    Complex IV (COX): NORMAL — KEY DDx SURF1/SCO2/COX10/COX15

DISTINGUISHING FEATURES vs OTHER CI-LEIGH SUBUNITS:
  vs NDUFS1 (IP1/75kDa, N-module, N5 Fe-S):
    • NO peripheral neuropathy in NDUFB3 (NDUFS1: ~50% — CRITICAL DDx)
    • NDUFS1 axonal neuropathy reflects distal axon vulnerability to N5 Fe-S block
  vs NDUFS4 (AQDQ, 18kDa, N-module accessory, 5q11.2):
    • NO olfactory bulb MRI lesions in NDUFB3 (NDUFS4: 52–65% — PATHOGNOMONIC)
    • NDUFS4 olfactory bulb tropism is unique; NDUFB3 does not share this
  vs NDUFV1 (51kDa, N-module, FMN/N3):
    • NO leukodystrophy in NDUFB3 (NDUFV1: ~40–50%)
    • NDUFV1 white matter T2 signal reflects profound N-module FMN-relay block
  vs NDUFV2 (24kDa, N-module N1b-2Fe2S) / SCO2 (CIV assembly):
    • NO hypertrophic cardiomyopathy in NDUFB3 (NDUFV2: ~80%; SCO2: ~100%)
  vs NDUFA2 (B8, 8.5kDa, N-module/ND2 junction):
    • NDUFA2: BN-PAGE sub-assembly intermediates (N-module↔ND2 junction dissociation)
    • NDUFB3: BN-PAGE cleaner absent CI (PP-module scaffolding loss, fewer intermediates)
    • NDUFA2: thioredoxin fold N-module bridge; NDUFB3: membrane arm PP-module anchor
  vs NDUFS3 (30kDa, Q-module structural):
    • NDUFS3 sub-assembly intermediates (N/Q-module junction); NDUFB3 membrane arm absent CI
  vs POLG/DGUOK (mtDNA depletion syndromes):
    • NO hepatopathy in NDUFB3 (POLG: ~80%; DGUOK: ~90%)
  vs GRACILE (BCS1L/Complex III):
    • NO iron overload; NO aminoaciduria; NO neonatal cholestasis
  vs SURF1/SCO2/COX10/COX15 (COX deficiency):
    • COX activity NORMAL — biochemical fingerprint distinction
  vs NDUFS2 (49kDa, Q-module, N2 Fe-S terminal):
    • NO severe neonatal form in NDUFB3 comparable to NDUFS2 N2-block severity

HISTORICAL SIGNIFICANCE:
  NDUFB3 p.Trp22Arg (c.64T>C) discovered by Andreu et al. 1999 was the FIRST nuclear-encoded
  CI mutation in human disease.  It established that:
    1. Nuclear genes encoding CI structural subunits cause Leigh syndrome
    2. Isolated CI deficiency can result from PP-module structural failure
    3. PP-module subunits are essential for CI holocomplex assembly
  This opened the entire field of nuclear CI genetics (now >50 disease genes).

FOUNDER / RECURRENT MUTATIONS:
  p.Trp22Arg   c.64T>C  — PP-module contact region; severe infantile; index mutation 1999
  p.Val85Leu   c.253G>T — C-terminal structural region; European; moderate course
  p.Arg17Gln   c.50G>A  — N-terminal membrane anchor; milder partial CI residual
  p.Thr4Pro    c.10A>C  — signal peptide/N-terminal; severe neonatal; near-null residual

THERAPY — NDUFB3 / CI-LEIGH SPECIFICS:
  No targeted NDUFB3 PP-module restoration is clinically available.
  Management follows the CI-Leigh supportive protocol:
    ABSOLUTE contraindications (direct CI inhibitors / mito toxins):
      Metformin — directly inhibits CI at ND1/quinone-binding site in PP-module
      Valproate  — triple: CoA sequestration, POLG inhibition, ND-subunit expression
      Linezolid  — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
      Chloramphenicol — same ribosomal mechanism as linezolid
    CONTRAINDICATED:
      Ketogenic diet — forces NADH → β-oxidation through already-absent CI
    AVOID / HIGH CAUTION:
      Propofol — PRIS + secondary CIV inhibition adds a second ETC bottleneck
      Phenobarbital — secondary CI inhibitor; use LEV first
    LEVEL C cofactors (standard CI supportive):
      Riboflavin (B2) — CI-specific; FMN upstream of NDUFB3 PP-module
      CoQ10 (ubiquinol)
      Thiamine (B1) — MANDATORY empiric: mimics SLC19A3/BTD (treatable)
      Biotin — MANDATORY empiric: mimics BTD (treatable)
      Succinate — CII bypass; bypasses absent NDUFB3-CI entirely
      Carnitine
    PREFERRED AED: Levetiracetam (renal excretion, no mitochondrial toxicity)
    ANAESTHESIA: Sevoflurane preferred over propofol
    GLUCOSE: IV dextrose GIR 6–8 mg/kg/min — NEVER fast (CI energy failure)
"""

from __future__ import annotations
import random
from functools import lru_cache
from typing import Any

# ── Identity ──────────────────────────────────────────────────────────────────
GENE       = "NDUFB3"
DISEASE    = ("NDUFB3 Leigh Syndrome — Isolated Complex I Deficiency "
              "(PP-Module B12 Subunit / Proximal Membrane Arm / First Nuclear CI Mutation)")
OMIM_GENE  = "603839"
OMIM_DISEASE = "256000"
CHROMOSOME = "2q31.3"
SEED       = 627


# ── Cohort synthesis ──────────────────────────────────────────────────────────
def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    regions = ["Europe", "Middle East", "North America", "South Asia", "East Asia", "Latin America"]
    outcomes = ["deceased_infancy", "deceased_early_childhood", "severe_impairment",
                "moderate_impairment", "mild_impairment"]
    outcome_weights = [0.30, 0.28, 0.22, 0.13, 0.07]
    mutations = ["p.Trp22Arg", "p.Val85Leu", "p.Arg17Gln", "p.Thr4Pro",
                 "c.IVS2+1G>T (splice)", "other/compound-het"]
    mut_weights = [0.38, 0.20, 0.16, 0.10, 0.09, 0.07]

    def pick(seq, weights=None):
        if weights:
            return rng.choices(seq, weights=weights, k=1)[0]
        return rng.choice(seq)

    patients = []
    for i in range(1, n + 1):
        ci = round(rng.uniform(4.0, 19.5), 1)
        onset = rng.randint(1, 14)
        sex = pick(["M", "F"])
        region = pick(regions)
        outcome = pick(outcomes, outcome_weights)
        mut = pick(mutations, mut_weights)
        # Feature probabilities (NDUFB3-specific)
        leigh_mri = rng.random() < 0.82
        lactic_acid = rng.random() < 0.88
        hypotonia = rng.random() < 0.87
        psychomotor_reg = rng.random() < 0.92
        seizures = rng.random() < 0.45
        resp_comp = rng.random() < 0.35
        ataxia = rng.random() < 0.38
        dystonia = rng.random() < 0.32
        hcm = rng.random() < 0.03            # very low — DDx NDUFV2/SCO2
        periph_neuropathy = rng.random() < 0.02  # very low — DDx NDUFS1 50%
        olf_bulb = rng.random() < 0.01       # DDx NDUFS4 58%
        leukodystrophy = rng.random() < 0.03  # DDx NDUFV1 45%
        hepatopathy = rng.random() < 0.01    # DDx POLG/DGUOK
        patients.append({
            "id": i, "sex": sex, "age_onset_months": onset, "region": region,
            "ci_activity_pct": ci, "mutation": mut, "outcome": outcome,
            "leigh_mri": leigh_mri, "lactic_acidosis": lactic_acid,
            "hypotonia": hypotonia, "psychomotor_regression": psychomotor_reg,
            "seizures": seizures, "respiratory_compromise": resp_comp,
            "ataxia": ataxia, "dystonia": dystonia, "hcm": hcm,
            "peripheral_neuropathy": periph_neuropathy,
            "olfactory_bulb_lesions": olf_bulb,
            "leukodystrophy": leukodystrophy,
            "hepatopathy": hepatopathy,
        })
    return patients


@lru_cache(maxsize=1)
def _cohort() -> list[dict[str, Any]]:
    return _build_cohort(40)


def _pct(key: str) -> float:
    pts = _cohort()
    return round(100 * sum(1 for p in pts if p.get(key)) / len(pts), 1)


# ── Public API ────────────────────────────────────────────────────────────────
def get_overview() -> dict[str, Any]:
    pts = _cohort()
    n = len(pts)
    ci_vals = [p["ci_activity_pct"] for p in pts]
    ci_mean = round(sum(ci_vals) / n, 1)
    ci_min  = round(min(ci_vals), 1)
    ci_max  = round(max(ci_vals), 1)

    return {
        "gene":         GENE,
        "also_known_as": "B12 (bovine nomenclature)",
        "disease":      DISEASE,
        "omim_gene":    OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "chromosome":   CHROMOSOME,
        "inheritance":  "AR (autosomal recessive, biallelic)",
        "module":       "PP-module (Proximal Pump, Membrane Arm)",
        "historical_note": (
            "NDUFB3 p.Trp22Arg (1999) was the FIRST nuclear-encoded Complex I subunit "
            "mutation identified in human disease — it established the entire field of "
            "nuclear CI genetics (now >50 disease genes)."
        ),
        "protein": {
            "size_aa":   98,
            "size_kda":  11.0,
            "fold":      "Small membrane arm accessory (PP-module)",
            "module":    "PP-module — proximal pump, membrane arm ND2/ND3/ND6 sub-complex",
            "function":  "Structural scaffold for PP-module; required for CI membrane arm assembly",
            "fe_s_cluster": "None (structural membrane arm subunit, not electron relay)",
        },
        "key_pathway_note": (
            "NDUFB3 (B12) anchors to the outer face of the PP (proximal pump) module of the "
            "Complex I membrane arm.  It contacts the ND2/ND3 sub-complex and is required "
            "for stable PP-module assembly.  WITHOUT NDUFB3, the full 1-MDa CI holocomplex "
            "cannot form.\n\n"
            "BN-PAGE distinguishing pattern: ABSENT CI with fewer sub-assembly intermediates "
            "(PP-module scaffolding loss → whole membrane arm destabilised) — CONTRAST with "
            "N-module structural failures (NDUFA2/NDUFS3/NDUFS5) which show clear sub-assembly "
            "intermediates (N-module/membrane arm partially separated)."
        ),
        "biochemical_fingerprint": {
            "Complex I":        "5–20% of control — SEVERELY REDUCED (ISOLATED)",
            "Complex II":       "NORMAL (SDHA intact)",
            "Complex III":      "NORMAL",
            "Complex IV (COX)": "NORMAL — KEY DDx SURF1/SCO2/COX10/COX15",
        },
        "feature_frequencies_pct": {
            "psychomotor_regression":  _pct("psychomotor_regression"),
            "hypotonia":               _pct("hypotonia"),
            "lactic_acidosis":         _pct("lactic_acidosis"),
            "leigh_mri":               _pct("leigh_mri"),
            "seizures":                _pct("seizures"),
            "respiratory_compromise":  _pct("respiratory_compromise"),
            "ataxia":                  _pct("ataxia"),
            "dystonia":                _pct("dystonia"),
            "hcm":                     _pct("hcm"),
            "peripheral_neuropathy":   _pct("peripheral_neuropathy"),
            "olfactory_bulb_lesions":  _pct("olfactory_bulb_lesions"),
            "leukodystrophy":          _pct("leukodystrophy"),
            "hepatopathy":             _pct("hepatopathy"),
        },
        "key_ddx": [
            {"feature": "NO Peripheral Neuropathy", "significance": "DDx NDUFS1 ~50% axonal neuropathy (N5 Fe-S block, distal axon vulnerable) — CRITICAL"},
            {"feature": "NO Olfactory Bulb MRI Lesions", "significance": "DDx NDUFS4 52–65% — PATHOGNOMONIC for NDUFS4 only; absent in NDUFB3"},
            {"feature": "NO Leukodystrophy", "significance": "DDx NDUFV1 ~40–50% white matter T2 signal (profound N-module FMN-relay block)"},
            {"feature": "NO HCM (<5%)", "significance": "DDx NDUFV2 ~80% HCM (N1b-relay) and SCO2 ~100% HCM (CIV) — CRITICAL cardiac DDx"},
            {"feature": "NO Hepatopathy", "significance": "DDx POLG ~80%, DGUOK ~90% (mtDNA depletion) — liver disease absent in CI nuclear subunit defects"},
            {"feature": "BN-PAGE: Absent CI (cleaner pattern)", "significance": "vs N-module structural failures (NDUFA2/NDUFS3/NDUFS5): sub-assembly intermediates. NDUFB3: PP-module scaffolding loss → cleaner absent CI"},
            {"feature": "COX NORMAL", "significance": "DDx SURF1/SCO2/COX10/COX15 all cause COX deficiency — isolated CI biochemical fingerprint"},
            {"feature": "NO Iron Overload / Aminoaciduria", "significance": "DDx GRACILE (BCS1L, CIII) — no haematological or renal tubular features"},
            {"feature": "Historical: FIRST nuclear CI mutation", "significance": "Andreu 1999 NatGenet p.Trp22Arg — only gene with this status; informs PP-module role"},
        ],
        "absolute_contraindications": [
            "Metformin — directly inhibits CI at ND1/quinone site in PP-module (same module as NDUFB3); worsens already-absent CI",
            "Valproate (VPA) — triple mechanism: CoA sequestration + POLG inhibition + suppression of mt-encoded ND subunit expression",
            "Linezolid — inhibits mitochondrial 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits (ND1–ND6/ND4L); CI cannot be assembled",
            "Chloramphenicol — same mitoribosome inhibition mechanism as linezolid",
        ],
        "contraindicated": [
            "Ketogenic diet — forces NADH production through β-oxidation, dramatically increasing NADH demand through already-absent CI; OXPHOS failure worsened",
        ],
        "preferred_treatments": [
            "Riboflavin (B2) — CI-specific cofactor; FMN prosthetic group upstream of NDUFB3 PP-module",
            "CoQ10 (ubiquinol preferred) — electron acceptor; partially bypasses CI blockade",
            "Thiamine (B1) — MANDATORY empiric before genetics result; rules out SLC19A3/BTD (treatable Leigh mimic)",
            "Biotin — MANDATORY empiric before genetics result; rules out BTD (treatable Leigh mimic)",
            "Succinate — CII bypass; bypasses absent NDUFB3-CI membrane arm entirely",
            "Carnitine — supports fatty acid metabolism; prevents secondary carnitine depletion",
            "Levetiracetam (LEV) — preferred AED: renal excretion, no mitochondrial toxicity",
            "IV dextrose GIR 6–8 mg/kg/min — maintain glucose supply; NEVER fast (CI energy failure → acute decompensation)",
            "Sevoflurane (NOT propofol) — preferred anaesthetic; propofol PRIS risk amplified in CI deficiency",
        ],
        "key_references": [
            "Andreu AL et al. 1999 NatGenet — FIRST nuclear CI mutation: NDUFB3 p.Trp22Arg in CI-Leigh",
            "Fassone E & Rahman S 2012 JMedGenet — CI genetics review; nuclear subunit mutations",
            "Sazanov LA 2015 NatRevMolCellBiol — CI cryo-EM structure; PP-module / NDUFB3 B12 location",
            "Carroll J et al. 2006 MolCellProteomics — CI proteomics; NDUFB3 PP-module assignment",
            "Haack TB et al. 2012 — CI nuclear subunit series; PP-module structural defects",
        ],
        "cohort": {
            "n":                     n,
            "ci_activity_mean_pct":  ci_mean,
            "ci_activity_range_pct": f"{ci_min}–{ci_max}%",
            "seed":                  SEED,
        },
    }


def get_breakdown() -> dict[str, Any]:
    pts = _cohort()
    n = len(pts)

    def freq(key):
        count = sum(1 for p in pts if p.get(key))
        return {"count": count, "pct": round(100 * count / n, 1)}

    features = [
        "psychomotor_regression", "hypotonia", "lactic_acidosis", "leigh_mri",
        "seizures", "respiratory_compromise", "ataxia", "dystonia", "hcm",
        "peripheral_neuropathy", "olfactory_bulb_lesions", "leukodystrophy", "hepatopathy",
    ]

    # Outcome distribution
    from collections import Counter
    oc = Counter(p["outcome"] for p in pts)
    # Mutation distribution
    mc = Counter(p["mutation"] for p in pts)
    # Region distribution
    rc = Counter(p["region"] for p in pts)
    # Sex distribution
    sc = Counter(p["sex"] for p in pts)

    # CI activity histogram
    bins  = ["<7%", "7–10%", "10–13%", "13–16%", "16–20%"]
    edges = [0, 7, 10, 13, 16, 20]
    counts = [0] * 5
    for p in pts:
        v = p["ci_activity_pct"]
        for idx in range(len(bins)):
            if edges[idx] <= v < edges[idx + 1]:
                counts[idx] += 1
                break

    return {
        "gene": GENE,
        "n":    n,
        "feature_frequencies": {k: freq(k) for k in features},
        "outcome_distribution":  dict(oc),
        "mutation_distribution": dict(mc),
        "region_distribution":   dict(rc),
        "sex_distribution":      dict(sc),
        "ci_activity_histogram": {"bins": bins, "counts": counts},
        "patients": pts,
    }


def get_definitions() -> dict[str, Any]:
    return {
        "gene": GENE,
        "pharmacology": [
            {
                "term": "Metformin (ABSOLUTE CI in NDUFB3)",
                "detail": (
                    "Biguanide antidiabetic that inhibits Complex I directly at the ND1/quinone-binding site "
                    "within the PP-module — the same module disrupted by NDUFB3 loss.  In NDUFB3 deficiency "
                    "CI is already structurally absent; metformin would eliminate any residual CI activity, "
                    "precipitate fatal lactic acidosis, and worsen energy failure."
                ),
            },
            {
                "term": "Valproate/VPA (ABSOLUTE CI in NDUFB3)",
                "detail": (
                    "Triple mitochondrial mechanism: (1) CoA sequestration via valproyl-CoA formation depletes "
                    "mitochondrial CoA pool; (2) direct POLG inhibition impairs mtDNA replication (NDUFB3 CI "
                    "already lacks 7 mt-encoded ND subunits to work with); (3) suppresses transcription of "
                    "mt-encoded ND subunits.  In NDUFB3 CI deficiency, VPA causes fatal decompensation and "
                    "hepatotoxicity."
                ),
            },
            {
                "term": "Linezolid (ABSOLUTE CI)",
                "detail": (
                    "Oxazolidinone antibiotic that inhibits the mitochondrial 23S rRNA, blocking synthesis of "
                    "all 7 mt-encoded ND subunits (ND1, ND2, ND3, ND4, ND4L, ND5, ND6).  In NDUFB3 deficiency "
                    "the PP-module (ND2/ND3/ND6-containing) cannot form without these subunits; linezolid removes "
                    "any possibility of residual CI assembly."
                ),
            },
            {
                "term": "Propofol (AVOID — PRIS)",
                "detail": (
                    "Propofol infusion syndrome (PRIS) involves secondary CIV inhibition.  In NDUFB3 CI deficiency "
                    "ETC is already severely compromised at CI; adding CIV inhibition creates a second uncrossable "
                    "bottleneck across the entire ETC.  Sevoflurane is preferred."
                ),
            },
            {
                "term": "Ketogenic Diet (CONTRAINDICATED)",
                "detail": (
                    "KD forces cellular energy metabolism toward fatty acid β-oxidation, which generates NADH "
                    "that must be re-oxidised through CI.  In NDUFB3 deficiency CI is structurally absent, "
                    "creating fatal NADH accumulation and lactic acidosis."
                ),
            },
            {
                "term": "Phenobarbital (HIGH CAUTION)",
                "detail": (
                    "Secondary CI inhibitor; in CI-deficient patients phenobarbital further reduces residual CI "
                    "activity.  LEV (levetiracetam) is preferred for seizure management: renal excretion, "
                    "no mitochondrial toxicity."
                ),
            },
            {
                "term": "Succinate (CII bypass)",
                "detail": (
                    "Succinate feeds Complex II (SDHA), bypassing the NADH→CI step entirely and delivering "
                    "electrons directly to ubiquinone/CIII.  In NDUFB3 CI deficiency this bypass allows "
                    "partial ETC flux to continue despite absent CI."
                ),
            },
        ],
        "gene_concepts": [
            {
                "term": "NDUFB3 / B12",
                "detail": (
                    "NADH:Ubiquinone Oxidoreductase Subunit B3 (human); homologous to bovine B12 subunit.  "
                    "98-aa precursor, ~11 kDa.  Located in the PP (proximal pump) module of the Complex I "
                    "membrane arm.  No Fe-S cluster.  Gene at 2q31.3, OMIM *603839."
                ),
            },
            {
                "term": "PP-module (Proximal Pump)",
                "detail": (
                    "The proximal portion of the Complex I membrane arm, containing mt-encoded subunits ND2, ND3, "
                    "ND6, and associated nuclear-encoded accessory subunits including NDUFB3/B12.  The PP-module "
                    "drives the first proton-pumping step of CI.  NDUFB3 anchors to the outer face of this "
                    "sub-complex; its loss destabilises the entire PP-module."
                ),
            },
            {
                "term": "First Nuclear CI Mutation (Historical)",
                "detail": (
                    "Andreu et al. 1999 (NatGenet) — NDUFB3 p.Trp22Arg was the FIRST mutation identified in a "
                    "nuclear-encoded CI subunit gene causing human disease.  Prior to 1999, all CI mutations "
                    "were attributed to mtDNA.  This discovery founded the field of nuclear CI genetics, "
                    "now encompassing >50 disease genes and hundreds of pathogenic variants."
                ),
            },
            {
                "term": "BN-PAGE Assembly Pattern — PP-module failure",
                "detail": (
                    "Blue-native PAGE of NDUFB3-deficient patient muscle/fibroblasts shows ABSENT or near-absent "
                    "CI band with fewer sub-assembly intermediates — PP-module scaffolding loss destabilises the "
                    "whole membrane arm leaving little stable partial assembly.\n"
                    "CONTRAST: N-module structural failures (NDUFA2/NDUFS3/NDUFS5) accumulate N-module "
                    "sub-assembly intermediates that appear as distinct bands below the full CI band.\n"
                    "CONTRAST: Fe-S relay defects (NDUFS7/NDUFS8) also show clean absent CI but with a "
                    "direct electron relay block (not assembly failure)."
                ),
            },
            {
                "term": "p.Trp22Arg (c.64T>C) — index mutation",
                "detail": (
                    "The most common NDUFB3 pathogenic variant.  Tryptophan-22 is in the PP-module contact "
                    "region; the Arg substitution disrupts hydrophobic contacts with ND2/ND3, destabilising "
                    "NDUFB3 integration.  Severe infantile CI deficiency with Leigh syndrome.  "
                    "First identified by Andreu et al. 1999."
                ),
            },
        ],
        "disease_concepts": [
            {
                "term": "Leigh Syndrome (OMIM #256000)",
                "detail": (
                    "Subacute necrotising encephalomyelopathy characterised by bilateral symmetric MRI lesions "
                    "in basal ganglia (putamen, caudate), brainstem periaqueductal grey, and thalami.  "
                    "Pathologically: spongiform degeneration, astrogliosis, capillary proliferation.  "
                    "Can result from defects in >75 genes (CI through CV, PDH, PC, BT deficiency).  "
                    "CI subunit mutations are the most common cause of nuclear-genetic Leigh syndrome."
                ),
            },
            {
                "term": "Isolated Complex I Deficiency",
                "detail": (
                    "Reduction of Complex I enzymatic activity to 5–20% of control with normal CII, CIII, CIV.  "
                    "The biochemical fingerprint of all nuclear CI subunit mutations including NDUFB3.  "
                    "Must be confirmed by spectrophotometric assay in muscle or fibroblasts."
                ),
            },
        ],
        "prescribing_safety": [
            {
                "term": "NDUFB3 Prescribing Safety Pocket Card",
                "detail": (
                    "ABSOLUTE CI  : Metformin · Valproate (VPA) · Linezolid · Chloramphenicol\n"
                    "CONTRAINDICATED: Ketogenic diet\n"
                    "AVOID        : Propofol (PRIS + secondary CIV block)\n"
                    "HIGH CAUTION : Phenobarbital (secondary CI inhibitor)\n"
                    "PREFERRED AED: LEV (levetiracetam) — renal excretion, no mito toxicity\n"
                    "ANAESTHESIA  : Sevoflurane (NOT propofol)\n"
                    "GLUCOSE      : IV dextrose GIR 6–8 mg/kg/min — NEVER fast\n"
                    "COFACTORS    : Riboflavin B2 · CoQ10 · Thiamine B1* · Biotin* · Succinate · Carnitine\n"
                    "               (* MANDATORY empiric before genetics: rules out SLC19A3/BTD)"
                ),
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN ===")
    bk = get_breakdown()
    print(f"N={bk['n']}, features={list(bk['feature_frequencies'].keys())[:5]}")
    print("\n=== DEFINITIONS ===")
    df = get_definitions()
    print(f"sections={list(df.keys())}")
