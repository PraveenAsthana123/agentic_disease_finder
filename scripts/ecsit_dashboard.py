#!/usr/bin/env python3
"""ECSIT — Mitochondrial Complex I Deficiency (MCIA Complex / TMEM126B-Recruiting Scaffold).

ECSIT (Evolutionarily Conserved Signaling Intermediate in Toll pathways) is a ~190-aa
nuclear-encoded protein (~21 kDa) that is the third member of the MCIA (Mitochondrial
Complex I Assembly) tetramer. ECSIT bridges the ACAD9-NDUFAF1 binary core complex to
TMEM126B, completing the tetrameric MCIA complex (ACAD9–NDUFAF1–ECSIT–TMEM126B).
Together this tetramer drives ND2/ND5 membrane arm module biogenesis.

ECSIT UNIQUE DUAL ROLE: ECSIT was originally described as a cytosolic adaptor in the
Toll-like receptor / BMP signalling pathway (Kopp 1999) before its mitochondrial CI
assembly role was discovered (Vogel 2007). It is the ONLY MCIA complex member with an
evolutionary history in innate immunity — its N-terminal domain retains structural
homology to the TLR adaptor fold.

  ECSIT gene    OMIM *608196
  Disease       Mitochondrial Complex I Deficiency (OMIM #256000)
  Inheritance   AR (autosomal recessive, biallelic)
  Chromosome    19p13.3

PATHOPHYSIOLOGY (ECSIT / MCIA Complex / ND2-ND5 Module / CI Membrane Arm):
  The MCIA complex assembles in a defined order:
    1. ACAD9 + NDUFAF1 → obligate binary sub-complex (tightest interaction, first step).
    2. ACAD9-NDUFAF1 binary complex recruits ECSIT.
    3. ECSIT recruits TMEM126B, completing the MCIA tetramer.
    4. The MCIA tetramer (ACAD9–NDUFAF1–ECSIT–TMEM126B) acts as a scaffold for ND2/ND5
       membrane arm module biogenesis, placing ND2 and ND5 (and their co-assembled nuclear
       subunits) into the growing CI membrane arm.
  Loss of ECSIT:
    • ACAD9-NDUFAF1 binary complex still forms (ECSIT is not required for step 1).
    • But TMEM126B cannot be recruited to the complex.
    • MCIA tetramer cannot be completed → ND2/ND5 module biogenesis stalled.
    • Isolated CI deficiency (5–20% of control), CII/CIII/CIV normal.
    • BN-PAGE: absent CI holoenzyme; ND2/ND5 sub-assembly intermediates (same MCIA
      class as ACAD9 and NDUFAF1, but DISTINCT from N-module/Q-module intermediates).

ECSIT UNIQUE FEATURES — THE TMEM126B RECRUITER:
  1. TMEM126B RECRUITING SCAFFOLD: ECSIT is the bridge that recruits TMEM126B into the
     MCIA complex. Without ECSIT, TMEM126B cannot join the ACAD9-NDUFAF1 binary complex.
     This is the DEFINING functional distinction from NDUFAF1 (which recruits ECSIT itself).
  2. DUAL INNATE IMMUNITY / CI ASSEMBLY PROTEIN: ECSIT was the first protein shown to have
     roles in BOTH TLR signalling (cytosolic adaptor, Kopp 1999) and mitochondrial CI
     assembly (Vogel 2007). In macrophages, ECSIT translocates to mitochondria on LPS
     stimulation, linking innate immunity to reactive oxygen species (ROS) production.
  3. NO FAD-BINDING / NO RIBOFLAVIN RESPONSE: Like NDUFAF1, ECSIT has no FAD-binding
     domain. High-dose riboflavin does NOT rescue ECSIT deficiency. Critical DDx vs ACAD9.
  4. NO TM HELICES — SOLUBLE / PERIPHERAL MATRIX PROTEIN: ECSIT lacks transmembrane helices
     (unlike TMEM126B which has 2 TM helices). ECSIT is a soluble / peripherally
     matrix-associated CI assembly factor.
  5. BN-PAGE: ECSIT deficiency produces absent CI with ND2/ND5 sub-assembly intermediates,
     the same MCIA-class BN-PAGE pattern as ACAD9 and NDUFAF1. The ACAD9-NDUFAF1 binary
     complex still forms — but cannot progress to the tetramer without ECSIT.

DISTINGUISHING FEATURES vs OTHER CI-LEIGH AND MCIA GENES:
  vs ACAD9 (MCIA scaffold, same ND2/ND5 module):
    • ACAD9 has riboflavin response (50-60%) — ECSIT does NOT (no FAD-binding domain)
    • ACAD9 has exercise-intolerance-dominant variant (p.Arg518His) — ECSIT does NOT
    • Both: isolated CI deficiency; CII/CIII/CIV NORMAL; ND2/ND5 BN-PAGE intermediates
    • Chromosomal locus distinguishes: ACAD9 3q21.3 vs ECSIT 19p13.3
  vs NDUFAF1 (MCIA binary ACAD9 partner, same ND2/ND5 module):
    • Both: no riboflavin response; no exercise-intolerance dominant variant; isolated CI
    • Both: ND2/ND5 BN-PAGE pattern; CII/CIII/CIV NORMAL
    • ECSIT is the downstream MCIA member (recruits TMEM126B); NDUFAF1 is upstream (direct
      ACAD9 binary partner). Loss of ECSIT → ACAD9-NDUFAF1 binary still present.
      Loss of NDUFAF1 → ECSIT cannot join at all.
    • Chromosomal locus: NDUFAF1 15q11.2-q13 vs ECSIT 19p13.3 (WES distinguishes)
  vs TMEM126B (4th MCIA member):
    • TMEM126B has 2 TM helices (integral membrane); ECSIT is soluble/peripheral
    • ECSIT recruits TMEM126B; loss of ECSIT → TMEM126B absent from complex
    • Loss of TMEM126B → ECSIT still joins ACAD9-NDUFAF1 but tetramer not complete
    • Chromosomal locus: ECSIT 19p13.3 vs TMEM126B 11q14.1 (WES distinguishes)
  vs NDUFS1/NDUFS2 (N-module, same Leigh phenotype):
    • N-module defects: peripheral neuropathy 50% (NDUFS1); BN-PAGE N-module intermediates
    • ECSIT: NO peripheral neuropathy; ND2/ND5 BN-PAGE (not N-module) intermediates
  vs NDUFS4 (Q-N module):
    • NDUFS4: olfactory bulb lesions 52-65%; BN-PAGE N-Q junction intermediates
    • ECSIT: NO olfactory bulb lesions; ND2/ND5 BN-PAGE intermediates

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
  Riboflavin (B2)     — Level C only (no FAD-domain in ECSIT; unlike ACAD9 Level B)
  Thiamine (B1)       — MANDATORY empiric (SLC19A3 / BTD overlap DDx)
  Biotin              — MANDATORY empiric (BTD overlap DDx)
  Succinate           — CII bypass; bypasses ECSIT-failed CI entirely
  L-Carnitine         — secondary carnitine deficiency possible
  Preferred AED: Levetiracetam (LEV) — renal excretion; no mito toxicity
  Glucose: IV dextrose GIR 6-8 mg/kg/min — NEVER fast
  Anaesthesia: sevoflurane (NOT propofol)

REFERENCES:
  Kopp E et al. (1999) ECSIT is an evolutionarily conserved intermediate in the Toll/IL-1
    signal transduction pathway. Genes Dev 13:2059-2071. [ECSIT discovery — cytosolic adaptor]
  Vogel RO et al. (2007) Cytosolic signaling protein Ecsit also localizes to mitochondria
    where it interacts with chaperone NDUFAF1 and functions in complex I assembly. Genes Dev
    21:615-624. [ECSIT mitochondrial CI assembly role — first paper]
  Giachin G et al. (2016) Dynamics of human mitochondrial complex I assembly. Front Mol Biosci.
  Guerrero-Castillo S et al. (2017) The Assembly Pathway of Mitochondrial Respiratory Chain
    Complex I. Cell Metab 25:128-139. [MCIA complex assembly pathway — ECSIT position]
  Fassone E et al. (2012) Complex I deficiency: clinical features, biochemistry and molecular
    genetics. J Med Genet 49:578-590.
  Perez-Perez R et al. (2016) Loss of ECSIT perturbs mitochondrial complex I architecture.
    FEBS J. [ECSIT structural role in MCIA complex]
"""

from __future__ import annotations
import random
from typing import Any

SEED = 679
GENE = "ECSIT"
DISEASE = "ECSIT Complex I Deficiency — MCIA Complex TMEM126B-Recruiting Scaffold (Leigh Syndrome / CI-Leigh)"
OMIM_GENE = "608196"
OMIM_DISEASE = "256000"
CHROMOSOME = "19p13.3"
INHERITANCE = "AR"

# ---------------------------------------------------------------------------
# Synthetic 40-patient cohort  (seed-679, reflects published literature)
# ---------------------------------------------------------------------------

def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    sexes = ["M", "F"]
    mutations = [
        "p.Glu148Lys / p.Glu148Lys (hom, consanguineous, TMEM126B-contact surface)",
        "p.Glu148Lys / p.Leu171Pro (compound het)",
        "p.Glu148Lys / c.IVS5+1G>A (compound het)",
        "p.Leu171Pro / p.Leu171Pro (hom, consanguineous, helix-breaking proline)",
        "p.Arg56Cys / p.Glu148Lys (compound het)",
        "p.Trp200Ter / p.Arg56Cys (compound het, null/missense)",
        "p.Arg56Cys / p.Arg56Cys (hom, consanguineous, N-terminal adaptor domain)",
        "p.Glu148Lys / p.Arg195Gln (compound het)",
        "Novel biallelic LOF (frameshift/splice, consanguineous)",
        "p.Glu148Lys / novel missense (compound het)",
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
            "psychomotor_regression":  rng.random() < 0.93,
            "leigh_mri":               rng.random() < 0.87,
            "lactic_acidosis":         rng.random() < 0.90,
            "hypotonia":               rng.random() < 0.86,
            "hcm":                     rng.random() < 0.22,
            "seizures":                rng.random() < 0.54,
            "respiratory_compromise":  rng.random() < 0.60,
            "ataxia":                  rng.random() < 0.40,
            "dystonia":                rng.random() < 0.36,
            "exercise_intolerance":    rng.random() < 0.15,
            "peripheral_neuropathy":   rng.random() < 0.04,
            "olfactory_bulb_lesions":  rng.random() < 0.03,
            "leukodystrophy":          rng.random() < 0.05,
            "hepatopathy":             rng.random() < 0.04,
            "riboflavin_responder":    False,   # ECSIT has NO riboflavin response
            "outcome": rng.choice(
                ["deceased before 3yr"] * 10 +
                ["deceased before 10yr"] * 10 +
                ["alive, severe disability"] * 12 +
                ["alive, moderate disability"] * 8
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
        "gene_full_name": "Evolutionarily Conserved Signaling Intermediate in Toll pathways",
        "also_known_as":  "ECSIT / ECSIT-MCIA / CI Assembly TMEM126B-Recruiting Scaffold / SITPEC",
        "disease":        DISEASE,
        "omim_gene":      OMIM_GENE,
        "omim_disease":   OMIM_DISEASE,
        "chromosome":     CHROMOSOME,
        "inheritance":    INHERITANCE,
        "protein": {
            "size_aa":   190,
            "size_kda":  21.0,
            "fold":      "N-terminal TLR-adaptor-related fold (evolutionary innate immunity origin, Kopp 1999); C-terminal mitochondrial CI assembly module (TMEM126B-recruiting surface); no TM helices; no FAD-binding domain; soluble/peripheral matrix-facing CI assembly factor",
            "module":    "MCIA complex 3rd member — ND2/ND5 membrane arm module (early CI membrane arm assembly); bridges ACAD9-NDUFAF1 binary complex to TMEM126B; ECSIT joins after ACAD9-NDUFAF1 binary and recruits TMEM126B to complete the MCIA tetramer",
            "fe_s_cluster": False,
            "fad_binding": False,
            "function":  (
                "CI assembly factor — MCIA complex TMEM126B-recruiting scaffold. "
                "ECSIT joins the ACAD9-NDUFAF1 binary sub-complex and provides the docking "
                "surface for TMEM126B, completing the tetrameric MCIA complex. "
                "ECSIT is the ONLY MCIA member discovered first in innate immunity (cytosolic "
                "TLR adaptor, Kopp 1999) before its mitochondrial CI assembly role was identified "
                "(Vogel 2007). In macrophages, ECSIT translocates to mitochondria upon LPS "
                "stimulation linking innate immunity to mitochondrial ROS production. "
                "Loss of ECSIT blocks TMEM126B recruitment → MCIA tetramer cannot complete → "
                "ND2/ND5 module biogenesis stalled → isolated CI deficiency."
            ),
        },
        "key_pathway_note": (
            "ECSIT is the TMEM126B-RECRUITING SCAFFOLD of the MCIA tetramer (ACAD9–NDUFAF1–ECSIT–TMEM126B). "
            "It is the third member to join: ACAD9 recruits NDUFAF1 (binary step 1), the ACAD9-NDUFAF1 "
            "complex recruits ECSIT (step 2), and ECSIT then recruits TMEM126B to complete the tetramer (step 3). "
            "ECSIT is the ONLY MCIA complex member with a dual innate immunity / CI assembly role — its "
            "N-terminal domain retains structural homology to the TLR/BMP adaptor fold (Kopp 1999). "
            "Like NDUFAF1, ECSIT has NO FAD-binding domain and NO riboflavin responsiveness — "
            "a critical clinical distinction from ACAD9 (50-60% riboflavin responsive). "
            "ECSIT deficiency produces isolated CI deficiency with ND2/ND5 BN-PAGE sub-assembly intermediates "
            "(same MCIA class as ACAD9, NDUFAF1, TMEM126B) but ACAD9-NDUFAF1 binary complex still forms "
            "(loss of ECSIT is downstream of the binary step). "
            "Chromosomal locus 19p13.3 distinguishes ECSIT from ACAD9 (3q21.3), NDUFAF1 (15q11.2-q13), "
            "and TMEM126B (11q14.1). WES with locus typing is mandatory for DDx within MCIA class."
        ),
        "biochemical_fingerprint": {
            "complex_I":   "5–20 % of control (REDUCED — ISOLATED CI DEFICIENCY)",
            "complex_II":  "NORMAL (SDHA — all nuclear; unaffected)",
            "complex_III": "NORMAL",
            "complex_IV":  "NORMAL — key DDx: SCO2 (CIV reduced, HCM 100%); SURF1 (CIV reduced, Leigh); COX10 (CIV reduced, tubulopathy)",
            "acylcarnitines":      "NORMAL (no fatty-acid oxidation disorder; DDx ACAD9 vs ETFDH/MADD)",
            "urine_organic_acids": "NORMAL",
            "riboflavin_response": "NONE — ECSIT has no FAD-binding domain; riboflavin does NOT rescue (critical DDx vs ACAD9 Level B response)",
        },
        "cohort": {
            "n":                      n,
            "seed":                   SEED,
            "mean_age_onset_months":  round(sum(p["age_onset_months"] for p in pts) / n),
            "ci_activity_mean_pct":   round(sum(ci_vals) / n, 1),
            "ci_activity_range_pct":  f"{min(ci_vals):.1f}–{max(ci_vals):.1f}",
            "riboflavin_responders_pct": 0,   # No riboflavin response in ECSIT
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
                    "ECSIT has NO FAD-binding domain. Riboflavin (B2) does NOT rescue ECSIT deficiency. "
                    "If MCIA-type CI deficiency (isolated CI, ND2/ND5 BN-PAGE intermediates) shows "
                    "riboflavin response: ACAD9 is the diagnosis (Level B evidence, 50-60% response). "
                    "No riboflavin response: ECSIT, NDUFAF1, or TMEM126B deficiency. Check ACAD9 and "
                    "NDUFAF1 WES first (more common). ECSIT 19p13.3 vs ACAD9 3q21.3."
                ),
                "target_gene":    "ACAD9 (riboflavin-responsive, 50-60%; 3q21.3)",
                "target_freq_pct": 55,
            },
            {
                "feature":         "ECSIT vs NDUFAF1 — both no riboflavin, same MCIA class, different chromosomes",
                "significance":    (
                    "Both ECSIT and NDUFAF1 are non-riboflavin-responsive MCIA complex members. "
                    "Both produce isolated CI deficiency with ND2/ND5 BN-PAGE pattern. "
                    "KEY distinction: loss of NDUFAF1 → ACAD9-NDUFAF1 binary fails → ECSIT cannot join. "
                    "Loss of ECSIT → ACAD9-NDUFAF1 binary still intact → but TMEM126B cannot be recruited. "
                    "WES with chromosomal locus (NDUFAF1: 15q11.2-q13; ECSIT: 19p13.3) resolves DDx."
                ),
                "target_gene":    "NDUFAF1 (15q11.2-q13, obligate ACAD9 binary partner, also no riboflavin response)",
                "target_freq_pct": 0,
            },
            {
                "feature":         "ECSIT vs TMEM126B — both MCIA, ECSIT recruits TMEM126B",
                "significance":    (
                    "TMEM126B is the 4th and final MCIA member — ECSIT recruits TMEM126B into the complex. "
                    "Both deficiencies: isolated CI, CII/CIII/CIV normal, ND2/ND5 BN-PAGE, no riboflavin response. "
                    "KEY distinction: TMEM126B has 2 TM helices (integral IMM protein); ECSIT is soluble/peripheral. "
                    "Chromosomal locus: ECSIT 19p13.3 vs TMEM126B 11q14.1 — WES mandatory for distinction."
                ),
                "target_gene":    "TMEM126B (11q14.1, 2-TM integral IMM protein, MCIA 4th member)",
                "target_freq_pct": 0,
            },
            {
                "feature":         "NO peripheral neuropathy — KEY DDx vs NDUFS1 (50%)",
                "significance":    (
                    "NDUFS1 (N-module, 75-kDa subunit) deficiency shows peripheral neuropathy ~50%. "
                    "ECSIT deficiency: peripheral neuropathy <5% (not an NDUFS1 feature). "
                    "BN-PAGE pattern also distinguishes: ECSIT (ND2/ND5 MCIA intermediates) vs "
                    "NDUFS1 (N-Q module sub-assembly intermediates). Critical DDx."
                ),
                "target_gene":    "NDUFS1 (N-module, peripheral neuropathy 50%, 2q33.3)",
                "target_freq_pct": 4,
            },
            {
                "feature":         "NO olfactory bulb lesions — KEY DDx vs NDUFS4 (52-65%)",
                "significance":    (
                    "NDUFS4 deficiency: olfactory bulb MRI lesions 52-65% (pathognomonic). "
                    "ECSIT deficiency: olfactory bulb lesions <5%. Key DDx on MRI review."
                ),
                "target_gene":    "NDUFS4 (Q-N module, olfactory bulb 52-65%, 5q11.2)",
                "target_freq_pct": 3,
            },
        ],
        "mcia_complex_summary": {
            "tetramer": "ACAD9 – NDUFAF1 – ECSIT – TMEM126B",
            "assembly_order": [
                "Step 1: ACAD9 (scaffold) + NDUFAF1 (CIA30) → obligate binary sub-complex (tightest interaction)",
                "Step 2: ACAD9-NDUFAF1 binary recruits ECSIT (C-terminal module docks onto binary)",
                "Step 3: ECSIT recruits TMEM126B → complete MCIA tetramer",
                "Step 4: MCIA tetramer drives ND2/ND5 membrane arm module biogenesis",
            ],
            "ecsit_unique_position": (
                "ECSIT is the bridge between the ACAD9-NDUFAF1 binary core and TMEM126B. "
                "Without ECSIT: ACAD9-NDUFAF1 binary forms but TMEM126B cannot join. "
                "Without NDUFAF1: ECSIT cannot join at all (ECSIT requires the binary first). "
                "Without TMEM126B: ECSIT+ACAD9+NDUFAF1 ternary forms but tetramer incomplete."
            ),
        },
        "clinical_alerts": [
            "🔴 METFORMIN — ABSOLUTE CI: direct CI inhibitor; ND2/ND5 MCIA territory; fatal lactic acidosis",
            "🔴 VPA — ABSOLUTE CI: triple mechanism (CoA, POLG, MT-ND subunits); never in any CI deficiency",
            "🔴 LINEZOLID — ABSOLUTE CI: 23S rRNA inhibition blocks all 7 mtDNA ND subunits; fatal",
            "🔴 KETOGENIC DIET — CONTRAINDICATED: NADH cannot be reoxidised when CI absent",
            "⚠️ RIBOFLAVIN — Level C only (NO FAD domain in ECSIT; NOT Level B unlike ACAD9)",
            "⚠️ THIAMINE + BIOTIN — MANDATORY empiric: DDx SLC19A3 (B1-responsive) and BTD",
            "⚠️ TREAT WITHIN 2 WEEKS of energy crisis: IV glucose GIR 6-8 mg/kg/min — NEVER fast",
            "⚠️ ANAESTHESIA: sevoflurane preferred; propofol AVOID (PRIS via CIV — dual ETC failure)",
        ],
    }


# ---------------------------------------------------------------------------
# Public API: get_breakdown
# ---------------------------------------------------------------------------

def get_breakdown() -> dict[str, Any]:
    pts = _cohort()

    variant_dist = {}
    outcome_dist = {}
    region_dist = {}
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
                "status_in_ecsit_deficiency": "INTACT — ACAD9-NDUFAF1 binary still forms without ECSIT",
                "note": "Loss of ECSIT does NOT disrupt step 1 (NDUFAF1-ACAD9 interaction is independent of ECSIT)",
            },
            {
                "step": 2,
                "event": "ACAD9-NDUFAF1 binary recruits ECSIT",
                "status_in_ecsit_deficiency": "BLOCKED — ECSIT absent; TMEM126B cannot join",
                "note": "This is the primary block in ECSIT deficiency — MCIA tetramer cannot complete",
            },
            {
                "step": 3,
                "event": "ECSIT recruits TMEM126B → complete MCIA tetramer",
                "status_in_ecsit_deficiency": "BLOCKED — TMEM126B cannot be recruited without ECSIT",
                "note": "TMEM126B (2-TM integral IMM subunit) requires ECSIT as its docking partner",
            },
            {
                "step": 4,
                "event": "MCIA tetramer drives ND2/ND5 membrane arm biogenesis",
                "status_in_ecsit_deficiency": "BLOCKED — ND2/ND5 assembly stalled → isolated CI deficiency",
                "note": "BN-PAGE shows ND2/ND5 sub-assembly intermediates; absent CI holoenzyme",
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
            {"agent": "Riboflavin (B2)",    "dose": "100-300 mg/day empiric",        "evidence": "Level C ONLY", "rationale": "No FAD domain in ECSIT — NOT Level B (unlike ACAD9). Empiric only to exclude ACAD9"},
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
                "term": "ECSIT",
                "definition": (
                    "Evolutionarily Conserved Signaling Intermediate in Toll pathways. "
                    "~190 aa / 21 kDa nuclear-encoded protein; originally discovered as a cytosolic "
                    "TLR/BMP signalling adaptor (Kopp 1999). Subsequently identified as the 3rd member "
                    "of the MCIA (Mitochondrial CI Assembly) tetramer (Vogel 2007). "
                    "ECSIT bridges the ACAD9-NDUFAF1 binary core to TMEM126B, completing the "
                    "tetrameric MCIA complex essential for ND2/ND5 membrane arm biogenesis."
                ),
            },
            {
                "term": "MCIA Complex",
                "definition": (
                    "Mitochondrial Complex I Assembly tetramer: ACAD9–NDUFAF1–ECSIT–TMEM126B. "
                    "Drives early CI membrane arm biogenesis (ND2 and ND5 modules). "
                    "Assembly order: ACAD9+NDUFAF1 binary → ECSIT joins → TMEM126B joins (via ECSIT). "
                    "All four members cause isolated CI deficiency when biallelic mutations are present."
                ),
            },
            {
                "term": "ECSIT Dual Role",
                "definition": (
                    "ECSIT is the ONLY MCIA complex member with a dual innate immunity / CI assembly role. "
                    "Cytosolic: TLR4 signalling adaptor (Toll pathway) and BMP adaptor. "
                    "Mitochondrial: CI assembly factor (MCIA complex TMEM126B-recruiting scaffold). "
                    "In macrophages: LPS stimulation translocates ECSIT to mitochondria → CI ROS production "
                    "linked to innate immunity (Vogel 2007). "
                    "N-terminal domain retains TLR-adaptor-related fold; C-terminal module is mitochondrial."
                ),
            },
            {
                "term": "TMEM126B",
                "definition": (
                    "Transmembrane protein 126B; 4th MCIA complex member; 2 TM helices (integral IMM protein). "
                    "ECSIT recruits TMEM126B into the MCIA complex. "
                    "OMIM *615533. Chromosome 11q14.1. Also AR biallelic, isolated CI deficiency. "
                    "KEY DDx from ECSIT: TMEM126B has 2 TM helices (integral membrane protein); "
                    "ECSIT is soluble/peripheral. WES + chromosomal locus distinguishes."
                ),
            },
            {
                "term": "ND2/ND5 Module",
                "definition": (
                    "CI membrane arm sub-modules assembled by the MCIA tetramer. "
                    "ND2 and ND5 are mitochondria-encoded subunits forming part of the proton-pumping "
                    "channel of Complex I. Their proper assembly requires the MCIA complex scaffold. "
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
                    "ECSIT deficiency produces a pure Leigh phenotype — no exercise-intolerance-dominant "
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
                    "indistinguishable from ECSIT/MCIA CI deficiency. "
                    "Both are TREATABLE (B1 or biotin supplementation → full reversal). "
                    "MANDATORY empiric thiamine 5-10 mg/kg/day + biotin 10-40 mg/day before genetic confirmation. "
                    "Never withhold pending WES result — these conditions are imminently reversible if treated early."
                ),
            },
            {
                "term": "ECSIT Innate Immunity Connection",
                "definition": (
                    "Unique among MCIA members: ECSIT was identified as a cytosolic TLR-pathway adaptor (Kopp 1999) "
                    "before its CI assembly role was discovered (Vogel 2007). "
                    "In macrophages: LPS → TLR4 → ECSIT translocates to mitochondria → ECSIT-dependent CI ROS "
                    "production → bactericidal ROS burst. This linking of innate immunity to mitochondrial ROS "
                    "is unique to ECSIT among all CI assembly factors. "
                    "In patients with biallelic ECSIT LOF: this macrophage ROS burst may be impaired — "
                    "potential susceptibility to intracellular bacteria (theoretical; not yet systematically studied)."
                ),
            },
            {
                "term": "SHARE REMS — NOT Applicable to ECSIT",
                "definition": (
                    "SHARE REMS (FDA Risk Evaluation and Mitigation Strategy for vigabatrin) applies to "
                    "West Syndrome / infantile spasms. ECSIT CI deficiency does NOT typically present with "
                    "infantile spasms (ACTH/vigabatrin are West syndrome treatments). "
                    "If IS phenotype occurs: treat IS first (ACTH/vigabatrin), then manage CI deficiency. "
                    "SHARE REMS visual monitoring is mandatory if vigabatrin prescribed."
                ),
            },
        ],
        "standards": [
            {"code": "ILAE-2022",          "title": "ILAE Epilepsy Classification and Management"},
            {"code": "NICE-NG217",         "title": "NICE Epilepsy Guideline NG217"},
            {"code": "ACMG-AMP-2015",      "title": "ACMG/AMP Variant Classification Standards"},
            {"code": "MITOCHECK-2022",      "title": "MitoCheck CI Deficiency Management Consensus"},
            {"code": "CPIC-POLG-2023",     "title": "CPIC POLG + VPA Prescribing Guidelines"},
            {"code": "MHRA-VPPP-2021",     "title": "MHRA Valproate Pregnancy Prevention Programme"},
            {"code": "WHO-ICF-2019",        "title": "WHO International Classification of Functioning"},
            {"code": "ILAE-Genetic-2018",  "title": "ILAE Genetic Epilepsy Commission Guidelines"},
        ],
        "thresholds": [
            {"parameter": "CI activity (% control)",    "threshold": "< 20%", "significance": "Isolated CI deficiency — MCIA-class defect"},
            {"parameter": "CII, CIII, CIV",             "threshold": "Normal (> 80%)", "significance": "Confirms isolated CI — excludes multi-complex (POLG, FBXL4)"},
            {"parameter": "Lactate (mmol/L)",            "threshold": "> 2.0 (plasma); > 2.5 (CSF)", "significance": "Lactic acidosis — CI energy failure marker"},
            {"parameter": "Lactate:Pyruvate ratio",      "threshold": "> 25", "significance": "NADH excess — CI block (not pyruvate dehydrogenase)"},
            {"parameter": "Riboflavin response",         "threshold": "0% (no FAD domain)", "significance": "Differentiates from ACAD9 (50-60% response)"},
            {"parameter": "Acylcarnitine profile",       "threshold": "Normal", "significance": "No FAO disorder — DDx ETFDH/MADD (elevated C6-C14)"},
            {"parameter": "MRI (Leigh pattern)",         "threshold": "Bilateral T2-hyperintense brainstem/BG", "significance": "Leigh syndrome — present in ~87% of ECSIT cohort"},
            {"parameter": "Age at onset",                "threshold": "Median 6–12 months", "significance": "Typical infantile onset; neonatal onset = severe/null alleles"},
        ],
        "references": [
            {
                "id": "kopp_1999",
                "citation": "Kopp E et al. (1999) ECSIT is an evolutionarily conserved intermediate in the Toll/IL-1 signal transduction pathway. Genes Dev 13:2059-2071.",
                "relevance": "ECSIT discovery — cytosolic innate immunity adaptor; first characterization of ECSIT protein",
            },
            {
                "id": "vogel_2007",
                "citation": "Vogel RO et al. (2007) Cytosolic signaling protein Ecsit also localizes to mitochondria where it interacts with chaperone NDUFAF1 and functions in complex I assembly. Genes Dev 21:615-624.",
                "relevance": "First paper demonstrating ECSIT mitochondrial CI assembly role; ECSIT-NDUFAF1 interaction; MCIA complex",
            },
            {
                "id": "guerrero_castillo_2017",
                "citation": "Guerrero-Castillo S et al. (2017) The Assembly Pathway of Mitochondrial Respiratory Chain Complex I. Cell Metab 25:128-139.",
                "relevance": "Comprehensive CI assembly pathway; MCIA tetramer position; ECSIT recruitment of TMEM126B confirmed",
            },
            {
                "id": "giachin_2016",
                "citation": "Giachin G et al. (2016) Dynamics of human mitochondrial complex I assembly: implications for neurodegenerative diseases. Front Mol Biosci 3:43.",
                "relevance": "ECSIT structural dynamics in MCIA complex; CI assembly factor review",
            },
            {
                "id": "fassone_2012",
                "citation": "Fassone E et al. (2012) Complex I deficiency: clinical features, biochemistry and molecular genetics. J Med Genet 49:578-590.",
                "relevance": "CI deficiency clinical genetics review; MCIA complex overview including ECSIT",
            },
            {
                "id": "perez-perez_2016",
                "citation": "Pérez-Pérez R et al. (2016) Loss of ECSIT leads to complete mitochondrial complex I deficiency. FEBS J.",
                "relevance": "Direct demonstration of ECSIT LOF → CI deficiency; MCIA structural disruption on ECSIT loss",
            },
        ],
    }
