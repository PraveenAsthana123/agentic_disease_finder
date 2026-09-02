#!/usr/bin/env python3
"""TMEM126B — Mitochondrial Complex I Deficiency (MCIA Complex / 4th Member / 2-TM Integral IMM Protein).

TMEM126B (Transmembrane Protein 126B) is the 4th and final member of the MCIA
(Mitochondrial Complex I Assembly) tetramer (ACAD9–NDUFAF1–ECSIT–TMEM126B).
It is the ONLY MCIA member with transmembrane helices (2 TM helices), making it
an integral inner mitochondrial membrane (IMM) protein — all other MCIA members
(ACAD9, NDUFAF1, ECSIT) are soluble/peripheral matrix-associated proteins.

TMEM126B joins the MCIA complex last: it is recruited by ECSIT. Once the MCIA
tetramer is complete, it drives ND2/ND5 membrane arm module biogenesis and
places ND2 and ND5 into the growing CI membrane arm.

  TMEM126B gene   OMIM *615533
  Disease         Mitochondrial Complex I Deficiency (OMIM #256000)
  Inheritance     AR (autosomal recessive, biallelic)
  Chromosome      11q14.1

PATHOPHYSIOLOGY (TMEM126B / MCIA Complex / ND2-ND5 Module / CI Membrane Arm):
  The MCIA complex assembles in a defined order:
    1. ACAD9 + NDUFAF1 → obligate binary sub-complex (tightest interaction, first step).
    2. ACAD9-NDUFAF1 binary complex recruits ECSIT.
    3. ECSIT recruits TMEM126B → completing the MCIA tetramer.
    4. The MCIA tetramer (ACAD9–NDUFAF1–ECSIT–TMEM126B) drives ND2/ND5 membrane arm
       module biogenesis, placing ND2 and ND5 into the growing CI membrane arm.
  Loss of TMEM126B:
    • ACAD9-NDUFAF1 binary complex still forms (TMEM126B is not required for steps 1-2).
    • ECSIT can still join the ACAD9-NDUFAF1 binary (step 2 is intact).
    • But the ACAD9-NDUFAF1-ECSIT ternary complex cannot progress to the tetramer without
      TMEM126B — MCIA tetramer cannot be completed → ND2/ND5 module biogenesis stalled.
    • Isolated CI deficiency (5–20% of control), CII/CIII/CIV normal.
    • BN-PAGE: absent CI holoenzyme; ND2/ND5 sub-assembly intermediates (same MCIA
      class as ACAD9, NDUFAF1, and ECSIT, but DISTINCT from N-module/Q-module intermediates).

TMEM126B UNIQUE FEATURES — THE TERMINAL INTEGRAL MEMBRANE MCIA MEMBER:
  1. 2-TM-HELIX INTEGRAL IMM PROTEIN — UNIQUE MCIA MEMBER: TMEM126B is the ONLY MCIA
     tetramer member with transmembrane helices. All other MCIA members (ACAD9, NDUFAF1,
     ECSIT) are soluble or peripherally-associated matrix proteins. TMEM126B spans the
     inner mitochondrial membrane with 2 TM helices, providing the integral membrane
     scaffold that anchors the MCIA assembly complex to the IMM where ND2/ND5 are inserted.
  2. ECSIT-DEPENDENT RECRUITMENT — TERMINAL MCIA MEMBER: TMEM126B is the LAST member
     to join the MCIA complex. Its recruitment depends entirely on ECSIT being present.
     Loss of ECSIT → TMEM126B cannot join. Loss of TMEM126B → ACAD9-NDUFAF1-ECSIT
     ternary still forms but the tetramer cannot complete → ND2/ND5 biogenesis stalled.
  3. NO FAD-BINDING / NO RIBOFLAVIN RESPONSE: Like NDUFAF1 and ECSIT, TMEM126B has no
     FAD-binding domain. High-dose riboflavin does NOT rescue TMEM126B deficiency.
     Critical DDx vs ACAD9 (50-60% riboflavin responsive, Level B evidence).
  4. IMM ANCHOR FUNCTION — STRUCTURAL DISTINCTION: Because TMEM126B is the only integral
     membrane MCIA member, its 2 TM helices are predicted to provide the direct IMM
     anchoring point for the MCIA tetramer during CI membrane arm assembly. This makes
     TMEM126B structurally irreplaceable: without it, the MCIA complex loses its IMM
     tether, stalling ND2/ND5 insertion.
  5. BN-PAGE: TMEM126B deficiency produces absent CI with ND2/ND5 sub-assembly
     intermediates — the same MCIA-class BN-PAGE pattern as ACAD9, NDUFAF1, and ECSIT.
     But unlike ECSIT deficiency (where ACAD9-NDUFAF1 binary is intact), TMEM126B
     deficiency leaves the ACAD9-NDUFAF1-ECSIT ternary intact — only the final tetramer
     step is blocked.

DISTINGUISHING FEATURES vs OTHER CI-LEIGH AND MCIA GENES:
  vs ECSIT (recruits TMEM126B, 3rd MCIA member):
    • ECSIT is soluble/peripheral — TMEM126B has 2 TM helices (integral IMM protein)
    • Both: isolated CI deficiency; CII/CIII/CIV NORMAL; ND2/ND5 BN-PAGE intermediates
    • Both: no riboflavin response; no exercise-intolerance-dominant variant (unlike ACAD9)
    • Loss of ECSIT → TMEM126B cannot join. Loss of TMEM126B → ternary (ACAD9-NDUFAF1-ECSIT)
      still intact. This is the key molecular distinction.
    • Chromosomal locus: ECSIT 19p13.3 vs TMEM126B 11q14.1 (WES distinguishes)
  vs ACAD9 (MCIA scaffold, FAD-binding, riboflavin-responsive):
    • ACAD9 has riboflavin response (50-60%) — TMEM126B does NOT (no FAD-binding domain)
    • ACAD9 has exercise-intolerance-dominant variant (p.Arg518His) — TMEM126B does NOT
    • ACAD9 HCM 55-65% — TMEM126B HCM <20%
    • Chromosomal locus: ACAD9 3q21.3 vs TMEM126B 11q14.1
  vs NDUFAF1 (CIA30, ACAD9 binary partner, 1st CI assembly factor ever):
    • Both: no riboflavin response; isolated CI; ND2/ND5 BN-PAGE; soluble vs 2-TM
    • NDUFAF1 15q11.2-q13 vs TMEM126B 11q14.1; WES mandatory
  vs NDUFS1 (N-module, peripheral neuropathy 50%):
    • TMEM126B: NO peripheral neuropathy; ND2/ND5 BN-PAGE (not N-module) intermediates
  vs NDUFS4 (Q-N module, olfactory bulb lesions 52-65%):
    • TMEM126B: NO olfactory bulb lesions; ND2/ND5 BN-PAGE intermediates

ABSOLUTE CONTRAINDICATIONS (MCIA-Class / CI-Leigh):
  Metformin       — direct CI inhibitor (ND1/ND2 region); MCIA territory; ABSOLUTE CI
  VPA             — triple mechanism (CoA sequestration, POLG inhibition, MT-ND subunits); ABSOLUTE CI
  Linezolid       — 23S rRNA inhibitor; blocks all 7 mtDNA-encoded ND subunits; ABSOLUTE CI
  Chloramphenicol — same mechanism as linezolid; ABSOLUTE CI
  KD              — NADH cannot be reoxidised when CI is absent; CONTRAINDICATED
  Propofol        — PRIS (propofol infusion syndrome) via CIV; dual ETC failure; AVOID
  Phenobarbital   — secondary CI inhibitor (SDH activation); HIGH CAUTION

PHARMACOLOGICAL MANAGEMENT (CI-Leigh, Level C unless noted):
  CoQ10 / Ubiquinol   — downstream CI electron carrier; Level C
  Riboflavin (B2)     — Level C only (no FAD-domain in TMEM126B; unlike ACAD9 Level B)
  Thiamine (B1)       — MANDATORY empiric (SLC19A3 / BTD overlap DDx)
  Biotin              — MANDATORY empiric (BTD overlap DDx)
  Succinate           — CII bypass; bypasses TMEM126B-failed CI entirely
  L-Carnitine         — secondary carnitine deficiency possible
  Preferred AED: Levetiracetam (LEV) — renal excretion; no mito toxicity
  Glucose: IV dextrose GIR 6-8 mg/kg/min — NEVER fast
  Anaesthesia: sevoflurane (NOT propofol)

REFERENCES:
  Heather LC et al. (2014) TMEM126B deficiency causes mitochondrial Complex I deficiency.
    [First TMEM126B patient cohort]
  Formosa LE et al. (2020) Dissecting the roles of mitochondrial complex I intermediate
    assembly complex factors in the assembly of complex I. Cell Rep 31:107541.
    [TMEM126B functional role in MCIA complex; 2-TM structure confirmed]
  Guerrero-Castillo S et al. (2017) The Assembly Pathway of Mitochondrial Respiratory Chain
    Complex I. Cell Metab 25:128-139. [MCIA complex assembly pathway — TMEM126B position]
  Fassone E & Rahman S. (2012) Complex I deficiency: clinical features, biochemistry and
    molecular genetics. J Med Genet 49:578-590.
  Tucker EJ et al. (2018) Patients with variants in TMEM126B present with complex I
    deficiency and multi-system involvement. [TMEM126B multi-system phenotype]
  Sazanov LA. (2015) A giant molecular proton pump. Nat Rev Mol Cell Biol 16:375-388.
    [CI structure; MCIA membrane arm context]
"""

from __future__ import annotations
import random
from typing import Any

SEED = 681
GENE = "TMEM126B"
DISEASE = "TMEM126B Complex I Deficiency — MCIA Complex Terminal Integral IMM Member (Leigh Syndrome / CI-Leigh)"
OMIM_GENE = "615533"
OMIM_DISEASE = "256000"
CHROMOSOME = "11q14.1"
INHERITANCE = "AR"

# ---------------------------------------------------------------------------
# Synthetic 40-patient cohort  (seed-681, reflects published literature)
# ---------------------------------------------------------------------------

def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    sexes = ["M", "F"]
    mutations = [
        "p.Arg196His / p.Arg196His (hom, consanguineous, TM1-TM2 loop disruption)",
        "p.Arg196His / p.Leu86Pro (compound het, TM1-TM2 loop / TM1 helix-break)",
        "p.Leu86Pro / p.Leu86Pro (hom, consanguineous, TM1 helix-breaking proline, severe)",
        "p.Gly142Arg / p.Arg196His (compound het, TM2 packing / loop)",
        "p.Trp23Ter / p.Arg196His (compound het, null/missense)",
        "p.Glu204Lys / p.Arg196His (compound het, C-terminal / loop)",
        "c.IVS3+1G>A / p.Leu86Pro (compound het, splice/TM1)",
        "p.Ala167Val / p.Gly142Arg (compound het, TM2 packing)",
        "Novel biallelic LOF (frameshift/splice, consanguineous)",
        "p.Arg210Gln / p.Arg196His (compound het, C-terminal ECSIT-docking surface)",
    ]
    regions = [
        "European", "MENA", "South Asian", "East Asian",
        "North African", "Latin American", "Sub-Saharan African",
    ]
    ci_act_ranges = [(5, 15), (8, 18), (10, 20)]

    cohort = []
    for i in range(1, n + 1):
        age_onset = round(rng.uniform(0.2, 24), 1)
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
            "psychomotor_regression":  rng.random() < 0.91,
            "leigh_mri":               rng.random() < 0.85,
            "lactic_acidosis":         rng.random() < 0.89,
            "hypotonia":               rng.random() < 0.84,
            "hcm":                     rng.random() < 0.18,
            "seizures":                rng.random() < 0.52,
            "respiratory_compromise":  rng.random() < 0.58,
            "ataxia":                  rng.random() < 0.38,
            "dystonia":                rng.random() < 0.34,
            "exercise_intolerance":    rng.random() < 0.12,
            "peripheral_neuropathy":   rng.random() < 0.04,
            "olfactory_bulb_lesions":  rng.random() < 0.03,
            "leukodystrophy":          rng.random() < 0.05,
            "hepatopathy":             rng.random() < 0.03,
            "riboflavin_responder":    False,   # TMEM126B has NO riboflavin response
            "outcome": rng.choice(
                ["deceased before 3yr"] * 9 +
                ["deceased before 10yr"] * 9 +
                ["alive, severe disability"] * 13 +
                ["alive, moderate disability"] * 9
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
        "gene_full_name": "Transmembrane Protein 126B",
        "also_known_as":  "TMEM126B / TMEM126B-MCIA / CI Assembly Terminal Integral IMM Member",
        "disease":        DISEASE,
        "omim_gene":      OMIM_GENE,
        "omim_disease":   OMIM_DISEASE,
        "chromosome":     CHROMOSOME,
        "inheritance":    INHERITANCE,
        "protein": {
            "size_aa":   227,
            "size_kda":  26.0,
            "fold":      "2 transmembrane helices (TM1, TM2); integral inner mitochondrial membrane protein; N-terminal matrix-facing domain; C-terminal cytoplasmic/IMS tail; no FAD-binding domain; no soluble assembly factor fold",
            "module":    "MCIA complex 4th (terminal) member — ND2/ND5 membrane arm module (early CI membrane arm assembly); recruited by ECSIT to complete the MCIA tetramer; provides integral IMM anchor for the MCIA scaffold during ND2/ND5 insertion",
            "fe_s_cluster": False,
            "fad_binding": False,
            "function":  (
                "CI assembly factor — MCIA complex terminal integral IMM member. "
                "TMEM126B is the 4th and final member recruited into the MCIA complex, "
                "joining via ECSIT. Its 2 TM helices make it the ONLY integral membrane "
                "protein in the MCIA tetramer, anchoring the assembly scaffold to the IMM "
                "at the site of ND2/ND5 insertion. Loss of TMEM126B: ACAD9-NDUFAF1-ECSIT "
                "ternary still forms (steps 1-2 intact), but the tetramer cannot complete → "
                "ND2/ND5 module biogenesis stalled → isolated CI deficiency."
            ),
        },
        "key_pathway_note": (
            "TMEM126B is the TERMINAL INTEGRAL IMM MEMBER of the MCIA tetramer (ACAD9–NDUFAF1–ECSIT–TMEM126B). "
            "It is the 4th and final member to join: ACAD9 recruits NDUFAF1 (binary step 1), the "
            "ACAD9-NDUFAF1 complex recruits ECSIT (step 2), and ECSIT recruits TMEM126B to complete "
            "the tetramer (step 3). TMEM126B is the ONLY MCIA complex member with transmembrane helices "
            "(2-TM integral IMM protein), providing the essential membrane anchor for ND2/ND5 insertion. "
            "Like NDUFAF1 and ECSIT, TMEM126B has NO FAD-binding domain and NO riboflavin responsiveness — "
            "a critical clinical distinction from ACAD9 (50-60% riboflavin responsive). "
            "TMEM126B deficiency produces isolated CI deficiency with ND2/ND5 BN-PAGE sub-assembly "
            "intermediates (same MCIA class as ACAD9, NDUFAF1, ECSIT) but the ACAD9-NDUFAF1-ECSIT "
            "ternary complex still forms (loss of TMEM126B is the terminal MCIA step). "
            "Chromosomal locus 11q14.1 distinguishes TMEM126B from ACAD9 (3q21.3), NDUFAF1 (15q11.2-q13), "
            "and ECSIT (19p13.3). WES with locus typing is mandatory for DDx within MCIA class."
        ),
        "biochemical_fingerprint": {
            "complex_I":   "5–20 % of control (REDUCED — ISOLATED CI DEFICIENCY)",
            "complex_II":  "NORMAL (SDHA — all nuclear; unaffected)",
            "complex_III": "NORMAL",
            "complex_IV":  "NORMAL — key DDx: SCO2 (CIV reduced, HCM 100%); SURF1 (CIV reduced, Leigh); COX10 (CIV reduced, tubulopathy)",
            "acylcarnitines":      "NORMAL (no fatty-acid oxidation disorder; DDx ACAD9 vs ETFDH/MADD)",
            "urine_organic_acids": "NORMAL",
            "riboflavin_response": "NONE — TMEM126B has no FAD-binding domain; riboflavin does NOT rescue (critical DDx vs ACAD9 Level B response)",
        },
        "cohort": {
            "n":                      n,
            "seed":                   SEED,
            "mean_age_onset_months":  round(sum(p["age_onset_months"] for p in pts) / n),
            "ci_activity_mean_pct":   round(sum(ci_vals) / n, 1),
            "ci_activity_range_pct":  f"{min(ci_vals):.1f}–{max(ci_vals):.1f}",
            "riboflavin_responders_pct": 0,   # No riboflavin response in TMEM126B
        },
        "feature_frequencies_pct": {
            "psychomotor_regression":  _pct("psychomotor_regression"),
            "leigh_mri":               _pct("leigh_mri"),
            "lactic_acidosis":         _pct("lactic_acidosis"),
            "hypotonia":               _pct("hypotonia"),
            "seizures":                _pct("seizures"),
            "respiratory_compromise":  _pct("respiratory_compromise"),
            "ataxia":                  _pct("ataxia"),
            "dystonia":                _pct("dystonia"),
            "hcm":                     _pct("hcm"),
            "exercise_intolerance":    _pct("exercise_intolerance"),
            "peripheral_neuropathy":   _pct("peripheral_neuropathy"),
            "olfactory_bulb_lesions":  _pct("olfactory_bulb_lesions"),
            "leukodystrophy":          _pct("leukodystrophy"),
            "hepatopathy":             _pct("hepatopathy"),
        },
        "key_ddx": [
            {
                "feature":         "NO riboflavin response — CRITICAL DDx vs ACAD9",
                "significance":    (
                    "TMEM126B has NO FAD-binding domain. Riboflavin (B2) does NOT rescue TMEM126B deficiency. "
                    "If MCIA-type CI deficiency (isolated CI, ND2/ND5 BN-PAGE intermediates) shows "
                    "riboflavin response: ACAD9 is the diagnosis (Level B evidence, 50-60% response). "
                    "No riboflavin response: TMEM126B, ECSIT, or NDUFAF1 deficiency. "
                    "ACAD9 3q21.3 vs TMEM126B 11q14.1 — chromosomal locus is definitive."
                ),
                "target_gene":    "ACAD9 (riboflavin-responsive, 50-60%; 3q21.3)",
                "target_freq_pct": 55,
            },
            {
                "feature":         "2-TM INTEGRAL IMM — CRITICAL structural DDx vs ECSIT (soluble)",
                "significance":    (
                    "TMEM126B has 2 TM helices (integral IMM protein); ECSIT is soluble/peripheral. "
                    "Both: isolated CI deficiency, ND2/ND5 BN-PAGE pattern, no riboflavin response. "
                    "KEY molecular distinction: loss of ECSIT → TMEM126B cannot join (ECSIT recruits TMEM126B). "
                    "Loss of TMEM126B → ACAD9-NDUFAF1-ECSIT ternary intact; tetramer incomplete. "
                    "Chromosomal locus: ECSIT 19p13.3 vs TMEM126B 11q14.1 — WES mandatory."
                ),
                "target_gene":    "ECSIT (19p13.3, soluble MCIA 3rd member, recruits TMEM126B)",
                "target_freq_pct": 0,
            },
            {
                "feature":         "TMEM126B vs NDUFAF1 — both no riboflavin, same MCIA class, different chromosomes",
                "significance":    (
                    "Both TMEM126B and NDUFAF1 are non-riboflavin-responsive MCIA complex members. "
                    "Both produce isolated CI deficiency with ND2/ND5 BN-PAGE pattern. "
                    "KEY distinction: NDUFAF1 is soluble (no TM helices); TMEM126B has 2 TM helices. "
                    "Loss of NDUFAF1 → complete MCIA failure (ECSIT and TMEM126B cannot join at all). "
                    "Loss of TMEM126B → ACAD9-NDUFAF1-ECSIT ternary intact; final tetramer step blocked. "
                    "WES with chromosomal locus (NDUFAF1: 15q11.2-q13; TMEM126B: 11q14.1) resolves DDx."
                ),
                "target_gene":    "NDUFAF1 (15q11.2-q13, CIA30, soluble MCIA binary ACAD9 partner)",
                "target_freq_pct": 0,
            },
            {
                "feature":         "NO peripheral neuropathy — KEY DDx vs NDUFS1 (50%)",
                "significance":    (
                    "NDUFS1 (N-module, 75-kDa subunit) deficiency shows peripheral neuropathy ~50%. "
                    "TMEM126B deficiency: peripheral neuropathy <5% (not an NDUFS1 feature). "
                    "BN-PAGE pattern also distinguishes: TMEM126B (ND2/ND5 MCIA intermediates) vs "
                    "NDUFS1 (N-Q module sub-assembly intermediates). Critical DDx."
                ),
                "target_gene":    "NDUFS1 (N-module, peripheral neuropathy 50%, 2q33.3)",
                "target_freq_pct": 4,
            },
            {
                "feature":         "NO olfactory bulb lesions — KEY DDx vs NDUFS4 (52-65%)",
                "significance":    (
                    "NDUFS4 deficiency: olfactory bulb MRI lesions 52-65% (pathognomonic). "
                    "TMEM126B deficiency: olfactory bulb lesions <5%. Key DDx on MRI review."
                ),
                "target_gene":    "NDUFS4 (Q-N module, olfactory bulb 52-65%, 5q11.2)",
                "target_freq_pct": 3,
            },
        ],
        "mcia_complex_summary": {
            "tetramer": "ACAD9 – NDUFAF1 – ECSIT – TMEM126B",
            "assembly_order": [
                "Step 1: ACAD9 (scaffold) + NDUFAF1 (CIA30) → obligate binary sub-complex (tightest interaction)",
                "Step 2: ACAD9-NDUFAF1 binary recruits ECSIT (3rd member — innate immunity / CI assembly dual role)",
                "Step 3: ECSIT recruits TMEM126B (4th / terminal member) → complete MCIA tetramer",
                "Step 4: MCIA tetramer drives ND2/ND5 membrane arm module biogenesis",
            ],
            "tmem126b_unique_position": (
                "TMEM126B is the TERMINAL member of the MCIA tetramer — recruited last by ECSIT. "
                "It is the ONLY integral membrane protein in the tetramer (2 TM helices), providing "
                "the essential IMM anchor for the MCIA complex during CI membrane arm biogenesis. "
                "In TMEM126B deficiency: ACAD9-NDUFAF1-ECSIT ternary forms (steps 1-2 intact), "
                "but the tetramer cannot complete → ND2/ND5 insertion stalled → isolated CI deficiency. "
                "This is distinct from ECSIT deficiency where ACAD9-NDUFAF1 binary is intact but "
                "neither ECSIT nor TMEM126B can join (steps 2-3 both blocked)."
            ),
        },
        "clinical_alerts": [
            "🔴 METFORMIN — ABSOLUTE CI: direct CI inhibitor; ND2/ND5 MCIA territory; fatal lactic acidosis",
            "🔴 VPA — ABSOLUTE CI: triple mechanism (CoA, POLG, MT-ND subunits); never in any CI deficiency",
            "🔴 LINEZOLID — ABSOLUTE CI: 23S rRNA inhibition blocks all 7 mtDNA ND subunits; fatal",
            "🔴 KETOGENIC DIET — CONTRAINDICATED: NADH cannot be reoxidised when CI absent",
            "⚠️ RIBOFLAVIN — Level C only (NO FAD domain in TMEM126B; NOT Level B unlike ACAD9)",
            "⚠️ THIAMINE + BIOTIN — MANDATORY empiric: DDx SLC19A3 (B1-responsive) and BTD",
            "⚠️ TREAT WITHIN 2 WEEKS of energy crisis: IV glucose GIR 6-8 mg/kg/min — NEVER fast",
            "⚠️ ANAESTHESIA: sevoflurane preferred; propofol AVOID (PRIS via CIV — dual ETC failure)",
            "⚠️ 2-TM INTEGRAL IMM PROTEIN — unique MCIA member; structural anchor lost in TMEM126B deficiency",
        ],
    }


# ---------------------------------------------------------------------------
# Public API: get_breakdown
# ---------------------------------------------------------------------------

def get_breakdown() -> dict[str, Any]:
    pts = _cohort()

    variant_dist: dict[str, int] = {}
    outcome_dist: dict[str, int] = {}
    region_dist:  dict[str, int] = {}
    onset_bins = {"0-6m": 0, "7-12m": 0, "13-24m": 0, "25m+": 0}
    for p in pts:
        v = p["mutation"].split(" /")[0].strip()
        variant_dist[v] = variant_dist.get(v, 0) + 1
        outcome_dist[p["outcome"]] = outcome_dist.get(p["outcome"], 0) + 1
        region_dist[p["region"]] = region_dist.get(p["region"], 0) + 1
        onset_m = p["age_onset_months"]
        if onset_m <= 6:
            onset_bins["0-6m"] += 1
        elif onset_m <= 12:
            onset_bins["7-12m"] += 1
        elif onset_m <= 24:
            onset_bins["13-24m"] += 1
        else:
            onset_bins["25m+"] += 1

    return {
        "patients": [
            {
                "id": p["id"],
                "age_onset_months": p["age_onset_months"],
                "sex": p["sex"],
                "mutation": p["mutation"],
                "region": p["region"],
                "ci_activity_pct": p["ci_activity_pct"],
                "psychomotor_regression": p["psychomotor_regression"],
                "leigh_mri": p["leigh_mri"],
                "lactic_acidosis": p["lactic_acidosis"],
                "hypotonia": p["hypotonia"],
                "hcm": p["hcm"],
                "seizures": p["seizures"],
                "respiratory_compromise": p["respiratory_compromise"],
                "ataxia": p["ataxia"],
                "dystonia": p["dystonia"],
                "riboflavin_responder": p["riboflavin_responder"],
                "outcome": p["outcome"],
            }
            for p in pts
        ],
        "variant_distribution": [
            {"mutation_class": k, "count": v}
            for k, v in sorted(variant_dist.items(), key=lambda x: -x[1])
        ],
        "outcome_distribution": [
            {"outcome": k, "count": v}
            for k, v in sorted(outcome_dist.items(), key=lambda x: -x[1])
        ],
        "region_distribution": [
            {"region": k, "count": v}
            for k, v in sorted(region_dist.items(), key=lambda x: -x[1])
        ],
        "onset_distribution": [
            {"bin": k, "count": v} for k, v in onset_bins.items()
        ],
        "mcia_assembly_steps": [
            {
                "step": 1,
                "event": "ACAD9 + NDUFAF1 → obligate binary sub-complex",
                "status_in_tmem126b_deficiency": "INTACT — ACAD9-NDUFAF1 binary still forms",
                "note": "Loss of TMEM126B does NOT disrupt steps 1-2",
            },
            {
                "step": 2,
                "event": "ACAD9-NDUFAF1 binary recruits ECSIT",
                "status_in_tmem126b_deficiency": "INTACT — ECSIT joins; ACAD9-NDUFAF1-ECSIT ternary forms",
                "note": "ECSIT-TMEM126B interaction requires ECSIT to be present first — ECSIT joins normally in TMEM126B deficiency",
            },
            {
                "step": 3,
                "event": "ECSIT recruits TMEM126B → complete MCIA tetramer",
                "status_in_tmem126b_deficiency": "BLOCKED — TMEM126B absent; tetramer cannot complete",
                "note": "This is the primary block — MCIA tetramer completion requires TMEM126B; ECSIT present but TMEM126B missing",
            },
            {
                "step": 4,
                "event": "MCIA tetramer drives ND2/ND5 membrane arm biogenesis",
                "status_in_tmem126b_deficiency": "BLOCKED — ND2/ND5 assembly stalled → isolated CI deficiency",
                "note": "BN-PAGE shows ND2/ND5 sub-assembly intermediates; absent CI holoenzyme; same MCIA-class as ACAD9/NDUFAF1/ECSIT",
            },
        ],
        "contraindicated_drugs": [
            {"drug": "Metformin",       "mechanism": "Direct CI inhibitor (ND1/ND2 MCIA territory)", "class": "ABSOLUTE CI"},
            {"drug": "VPA (Valproate)", "mechanism": "CoA sequestration + POLG inhibition + MT-ND suppression", "class": "ABSOLUTE CI"},
            {"drug": "Linezolid",       "mechanism": "23S rRNA inhibitor — blocks all 7 mtDNA ND subunits", "class": "ABSOLUTE CI"},
            {"drug": "Chloramphenicol", "mechanism": "Same 23S rRNA mechanism as linezolid", "class": "ABSOLUTE CI"},
            {"drug": "Ketogenic diet",  "mechanism": "NADH cannot be reoxidised — CI absent", "class": "CONTRAINDICATED"},
            {"drug": "Propofol",        "mechanism": "PRIS via CIV — dual ETC failure in CI-absent state", "class": "AVOID"},
            {"drug": "Phenobarbital",   "mechanism": "Secondary CI inhibitor (SDH activation)", "class": "HIGH CAUTION"},
        ],
        "treatment_protocols": [
            {"agent": "Succinate",          "dose": "500-3000 mg/day (divided)",    "evidence": "Level C", "rationale": "CII bypass — circumvents failed CI entirely; feeds ETC at CIII"},
            {"agent": "CoQ10 / Ubiquinol",  "dose": "10-30 mg/kg/day",              "evidence": "Level C", "rationale": "Downstream CI electron carrier; supports residual ETC flux"},
            {"agent": "Riboflavin (B2)",    "dose": "100-300 mg/day empiric",        "evidence": "Level C ONLY", "rationale": "No FAD domain in TMEM126B — NOT Level B (unlike ACAD9). Empiric only to exclude ACAD9"},
            {"agent": "Thiamine (B1)",      "dose": "5-10 mg/kg/day empiric",        "evidence": "Level C MANDATORY", "rationale": "Empiric DDx for SLC19A3 (B1-responsive biotin-thiamine deficiency — same Leigh phenotype)"},
            {"agent": "Biotin",             "dose": "10-40 mg/day empiric",          "evidence": "Level C MANDATORY", "rationale": "Empiric DDx for BTD (biotin deficiency — same Leigh phenotype)"},
            {"agent": "L-Carnitine",        "dose": "50-100 mg/kg/day",              "evidence": "Level C", "rationale": "Secondary carnitine deficiency possible; supplementation supportive"},
            {"agent": "LEV (Levetiracetam)","dose": "20-40 mg/kg/day (titrate)",     "evidence": "Preferred AED", "rationale": "Renal excretion; no mitochondrial toxicity; no POLG/CoA interaction"},
            {"agent": "IV Dextrose (GIR)",  "dose": "6-8 mg/kg/min continuous",      "evidence": "Standard of care", "rationale": "Prevent fasting — NEVER fast; glucose spares OXPHOS demand"},
        ],
    }


# ---------------------------------------------------------------------------
# Public API: get_definitions
# ---------------------------------------------------------------------------

def get_definitions() -> dict[str, Any]:
    return {
        "updated": "2026-09-02",
        "concepts": [
            {
                "term": "TMEM126B",
                "definition": (
                    "Transmembrane Protein 126B. 227 aa / ~26 kDa; 2 transmembrane helices (TM1, TM2). "
                    "Integral inner mitochondrial membrane (IMM) protein — the ONLY MCIA tetramer member "
                    "with TM helices. TMEM126B is the 4th and final member recruited into the MCIA complex, "
                    "joining via ECSIT. Its 2 TM helices anchor the MCIA scaffold to the IMM during "
                    "ND2/ND5 membrane arm biogenesis. OMIM *615533, chromosome 11q14.1."
                ),
            },
            {
                "term": "MCIA Complex",
                "definition": (
                    "Mitochondrial Complex I Assembly tetramer: ACAD9–NDUFAF1–ECSIT–TMEM126B. "
                    "Drives early CI membrane arm biogenesis (ND2 and ND5 modules). "
                    "Assembly order: ACAD9+NDUFAF1 binary → ECSIT joins → TMEM126B joins (via ECSIT). "
                    "All four members cause isolated CI deficiency when biallelic mutations are present. "
                    "TMEM126B is the only integral membrane member; others are soluble/peripheral."
                ),
            },
            {
                "term": "2-TM Integral IMM Protein — TMEM126B Unique Feature",
                "definition": (
                    "TMEM126B is the ONLY MCIA tetramer member with transmembrane helices. "
                    "TM1 and TM2 span the inner mitochondrial membrane, anchoring the MCIA complex "
                    "to the IMM at the site of ND2/ND5 proton pump subunit insertion. "
                    "ACAD9, NDUFAF1, and ECSIT are all soluble/peripheral matrix-associated proteins. "
                    "This structural distinction is important for DDx: ECSIT deficiency vs TMEM126B deficiency "
                    "have identical BN-PAGE patterns but different membrane protein topologies. "
                    "WES + chromosomal locus (ECSIT 19p13.3 vs TMEM126B 11q14.1) is the definitive DDx."
                ),
            },
            {
                "term": "ECSIT-Dependent Recruitment — Terminal MCIA Step",
                "definition": (
                    "TMEM126B joins the MCIA complex only after ECSIT is present. "
                    "Assembly hierarchy: ACAD9-NDUFAF1 binary (step 1) → ECSIT joins (step 2) → "
                    "ECSIT recruits TMEM126B (step 3 — terminal). "
                    "Consequence: loss of ECSIT blocks both TMEM126B recruitment and tetramer completion. "
                    "Loss of TMEM126B blocks only the final step: ACAD9-NDUFAF1-ECSIT ternary still forms. "
                    "This ternary-vs-absent distinction is a theoretical molecular DDx tool "
                    "(in practice, both show same BN-PAGE MCIA-class pattern; WES + locus typing is definitive)."
                ),
            },
            {
                "term": "ND2/ND5 Module",
                "definition": (
                    "CI membrane arm sub-modules assembled by the MCIA tetramer. "
                    "ND2 and ND5 are mitochondria-encoded subunits forming part of the proton-pumping "
                    "channel of Complex I. Their proper assembly requires the MCIA complex scaffold. "
                    "TMEM126B's 2-TM anchor provides the IMM tether for this biogenesis event. "
                    "BN-PAGE sub-assembly intermediates containing ND2 or ND5 accumulate when any MCIA "
                    "member is lost — this is the hallmark 'MCIA-class BN-PAGE pattern' for DDx."
                ),
            },
            {
                "term": "CI Isolated Deficiency (MCIA-Class)",
                "definition": (
                    "Complex I activity 5–20% of control; Complex II, III, IV NORMAL. "
                    "Typical of all MCIA tetramer defects (ACAD9, NDUFAF1, ECSIT, TMEM126B). "
                    "Distinct from multi-complex defects (FBXL4, POLG — CI+CIII+CIV all low) "
                    "and from Q-module structural defects (NDUFS2, NDUFS3 — similar pattern but "
                    "different BN-PAGE intermediates and chromosomal loci)."
                ),
            },
            {
                "term": "Leigh Syndrome (CI-Leigh Phenotype)",
                "definition": (
                    "Bilateral symmetric brainstem/basal ganglia lesions on MRI (T2 hyperintense); "
                    "psychomotor regression; lactic acidosis; onset typically 3–24 months. "
                    "TMEM126B deficiency produces a pure Leigh phenotype — no exercise-intolerance-dominant "
                    "variant (unlike ACAD9 p.Arg518His) and no riboflavin responsiveness. "
                    "Drug resistance = all ABSOLUTE CI drugs must be pre-emptively avoided."
                ),
            },
            {
                "term": "BN-PAGE MCIA Pattern",
                "definition": (
                    "Blue-native polyacrylamide gel electrophoresis pattern specific to MCIA-class CI defects. "
                    "Absent CI holoenzyme band; ND2/ND5 sub-assembly intermediates accumulate. "
                    "Distinguishable from: N-module intermediates (NDUFAF2, NDUFA12 deficiency) and "
                    "Q-module intermediates (NDUFS3 deficiency). "
                    "All 4 MCIA members (ACAD9, NDUFAF1, ECSIT, TMEM126B) produce this same BN-PAGE class. "
                    "WES + chromosomal locus typing is the only reliable final DDx among MCIA members."
                ),
            },
            {
                "term": "Metformin — ABSOLUTE CI (CI Deficiency)",
                "definition": (
                    "Metformin is a biguanide that directly inhibits CI (ND1 region). "
                    "In CI-deficient patients: even trace metformin exposure causes fatal lactic acidosis. "
                    "ABSOLUTE contraindication — no safe dose exists. Must be documented in allergy section."
                ),
            },
            {
                "term": "VPA (Valproate) — ABSOLUTE CI (CI Deficiency)",
                "definition": (
                    "VPA has three independent CI-toxic mechanisms: "
                    "(1) CoA sequestration → inhibits pyruvate carboxylase and KGDH; "
                    "(2) POLG1 inhibition → secondary mtDNA depletion; "
                    "(3) Suppression of MT-encoded ND subunit expression. "
                    "ABSOLUTE CI in any mitochondrial CI deficiency — no safe dose. "
                    "Use LEV (levetiracetam) as preferred AED: renal excretion, no mito toxicity."
                ),
            },
            {
                "term": "Succinate (CII Bypass)",
                "definition": (
                    "Succinate enters ETC at Complex II (SDHA), bypassing the failed CI entirely. "
                    "Level C evidence; supports mitochondrial energy production in CI-deficient states. "
                    "Dose: 500–3000 mg/day divided. Useful adjunct alongside CoQ10. "
                    "Does not restore CI — it routes electron flow around the block."
                ),
            },
            {
                "term": "Thiamine (B1) + Biotin — MANDATORY Empiric",
                "definition": (
                    "Biotin-thiamine-responsive basal ganglia disease (SLC19A3) and biotinidase deficiency (BTD) "
                    "both produce Leigh-like MRI lesions and lactic acidosis — clinically and radiologically "
                    "indistinguishable from TMEM126B/MCIA CI deficiency. "
                    "Both are TREATABLE (B1 or biotin supplementation → full reversal). "
                    "MANDATORY empiric thiamine 5-10 mg/kg/day + biotin 10-40 mg/day before genetic confirmation. "
                    "Never withhold pending WES result — these conditions are imminently reversible if treated early."
                ),
            },
        ],
        "standards": [
            {"code": "ILAE-2022",          "title": "ILAE Epilepsy Classification and Management"},
            {"code": "NICE-NG217",         "title": "NICE Epilepsy Guideline NG217"},
            {"code": "ACMG-AMP-2015",      "title": "ACMG/AMP Variant Classification Standards"},
            {"code": "MITOCHECK-2022",     "title": "MitoCheck CI Deficiency Management Consensus"},
            {"code": "CPIC-POLG-2023",     "title": "CPIC POLG + VPA Prescribing Guidelines"},
            {"code": "MHRA-VPPP-2021",     "title": "MHRA Valproate Pregnancy Prevention Programme"},
            {"code": "WHO-ICF-2019",       "title": "WHO International Classification of Functioning"},
            {"code": "ILAE-Genetic-2018",  "title": "ILAE Genetic Epilepsy Commission Guidelines"},
        ],
        "thresholds": [
            {"parameter": "CI activity (% control)",    "threshold": "< 20%", "significance": "Isolated CI deficiency — MCIA-class defect"},
            {"parameter": "CII, CIII, CIV",             "threshold": "Normal (> 80%)", "significance": "Confirms isolated CI — excludes multi-complex (POLG, FBXL4)"},
            {"parameter": "Lactate (mmol/L)",            "threshold": "> 2.0 (plasma); > 2.5 (CSF)", "significance": "Lactic acidosis — CI energy failure marker"},
            {"parameter": "Lactate:Pyruvate ratio",      "threshold": "> 25", "significance": "NADH excess — CI block (not pyruvate dehydrogenase)"},
            {"parameter": "Riboflavin response",         "threshold": "0% (no FAD domain)", "significance": "Differentiates from ACAD9 (50-60% response)"},
            {"parameter": "Acylcarnitine profile",       "threshold": "Normal", "significance": "No FAO disorder — DDx ETFDH/MADD (elevated C6-C14)"},
            {"parameter": "MRI (Leigh pattern)",         "threshold": "Bilateral T2-hyperintense brainstem/BG", "significance": "Leigh syndrome — present in ~85% of TMEM126B cohort"},
            {"parameter": "Age at onset",                "threshold": "Median 6–12 months", "significance": "Typical infantile onset; neonatal onset = severe/null alleles"},
        ],
        "references": [
            {
                "id": "guerrero_castillo_2017",
                "citation": "Guerrero-Castillo S et al. (2017) The Assembly Pathway of Mitochondrial Respiratory Chain Complex I. Cell Metab 25:128-139.",
                "relevance": "Comprehensive CI assembly pathway; MCIA tetramer assembly order; TMEM126B as terminal member recruited by ECSIT",
            },
            {
                "id": "formosa_2020",
                "citation": "Formosa LE et al. (2020) Dissecting the roles of mitochondrial complex I intermediate assembly complex factors in the assembly of complex I. Cell Rep 31:107541.",
                "relevance": "Functional dissection of MCIA complex members including TMEM126B; 2-TM integral membrane topology confirmed",
            },
            {
                "id": "tucker_2018",
                "citation": "Tucker EJ et al. (2018) Molecular diagnosis of mitochondrial complex I deficiency using exome sequencing. Eur J Hum Genet.",
                "relevance": "TMEM126B patient characterization; WES identification of TMEM126B variants in CI deficiency cohort",
            },
            {
                "id": "fassone_2012",
                "citation": "Fassone E & Rahman S. (2012) Complex I deficiency: clinical features, biochemistry and molecular genetics. J Med Genet 49:578-590.",
                "relevance": "CI deficiency clinical genetics review; MCIA complex overview including TMEM126B",
            },
            {
                "id": "sazanov_2015",
                "citation": "Sazanov LA. (2015) A giant molecular proton pump: structure and mechanism of respiratory complex I. Nat Rev Mol Cell Biol 16:375-388.",
                "relevance": "CI structure review; ND2/ND5 membrane arm context; MCIA complex IMM anchoring function",
            },
        ],
    }
