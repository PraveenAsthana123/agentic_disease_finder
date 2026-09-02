#!/usr/bin/env python3
"""ACAD9 — Complex I Deficiency (MCIA Complex / Riboflavin-Responsive CI Assembly Factor).

ACAD9 (Acyl-CoA Dehydrogenase Family, Member 9) is a 621-aa nuclear-encoded protein
(~70 kDa) that is the central scaffold of the MCIA (Mitochondrial Complex I Assembly)
complex. Despite belonging to the ACAD (acyl-CoA dehydrogenase) protein superfamily, ACAD9
has largely lost conventional fatty-acid β-oxidation enzymatic activity and instead serves
as a CI-specific assembly factor required for ND2/ND5 module biogenesis in the CI membrane
arm. ACAD9 forms a stable complex with NDUFAF1 (CIA30), ECSIT, and TMEM126B — the four
core members of the MCIA complex — which is essential for early membrane-arm CI assembly.

  ACAD9 gene    OMIM *611103
  Disease        Mitochondrial Complex I Deficiency (OMIM #256000)
  Inheritance    AR (autosomal recessive biallelic)
  Chromosome     3q21.3

PATHOPHYSIOLOGY (ACAD9 / MCIA Complex / ND2-ND5 Module / CI Membrane Arm):
  The MCIA complex is required for the early assembly of the CI membrane arm, specifically
  for biogenesis of the ND2 and ND5 modules:
    1. ACAD9 acts as the central scaffold protein around which the MCIA complex assembles.
    2. NDUFAF1 (CIA30) forms a tight binary complex with ACAD9 first, then recruits
       ECSIT and TMEM126B to complete the MCIA tetramer.
    3. The MCIA complex is required for incorporation of mt-encoded subunits ND2 and ND5
       (and their associated nuclear subunits NDUFB6, NDUFB8, NDUFB7, etc.) into the
       nascent CI membrane arm — the ND2 module and ND5 module assembly stages.
    4. ACAD9 deficiency disrupts MCIA complex integrity → ND2/ND5 module assembly fails
       → CI membrane arm cannot be assembled → CI holoenzyme absent or severely reduced
       on BN-PAGE, with ND2/ND5 sub-assembly intermediates sometimes detectable.

  ACAD9 UNIQUE FEATURES — THE ONLY RIBOFLAVIN-RESPONSIVE CI ASSEMBLY FACTOR:
    1. RIBOFLAVIN (B2) RESPONSE (~50–60% of patients):
       ACAD9 retains a FAD (flavin adenine dinucleotide) binding domain from its ACAD
       superfamily origin. High-dose riboflavin (B2 = riboflavin precursor → FMN → FAD)
       appears to stabilise mutant ACAD9 protein and partially restore MCIA complex
       integrity. Clinically, ~50–60% of ACAD9 patients show measurable improvement in
       CI activity (↑), exercise tolerance (↑), and/or HCM regression on riboflavin
       100–300 mg/day. This response rate is unique — no other CI assembly factor
       (NDUFAF1, NDUFAF2–8, ECSIT, TIMMDC1, TMEM126B) has established riboflavin
       responsiveness.
    2. EXERCISE INTOLERANCE AS DOMINANT / FIRST PRESENTATION:
       In adolescent/adult-onset ACAD9 deficiency (especially p.Arg518His), exercise
       intolerance and skeletal myopathy may be the SOLE presenting feature — Leigh MRI
       may be absent. This bimodal severity spectrum (Leigh-like infantile vs.
       exercise-intolerance-dominant adult/adolescent) is characteristic of ACAD9.
    3. FAD-BINDING DOMAIN: ACAD9 belongs to the ACAD superfamily (same protein family
       as VLCAD, LCAD, MCAD, SCAD, ETFB). Unlike its relatives, ACAD9 does NOT
       efficiently catalyse conventional fatty acid β-oxidation — it has evolved toward
       CI assembly chaperone function. This FAD-binding domain is the molecular basis of
       riboflavin responsiveness.
    4. MCIA COMPLEX ANCHOR: ACAD9 is the central, first-recruited member of the MCIA
       complex. Loss of ACAD9 prevents MCIA complex formation entirely (not just its
       enzymatic activity).
    5. BIMODAL SEVERITY SPECTRUM:
       • Leigh-like severe infantile: p.Arg266Gln (homozygous); p.Asp562Gly → Leigh MRI,
         hypotonia, lactic acidosis, HCM, early childhood death.
       • Exercise-intolerance-dominant: p.Arg518His (most common European founder) →
         exercise intolerance, skeletal myopathy, mild-moderate HCM; Leigh MRI less
         frequent; responds well to riboflavin.

DISTINGUISHING FEATURES vs OTHER CI-LEIGH GENES:
  vs NDUFS1 (N-module IP1/75kDa):
    • NO peripheral neuropathy in ACAD9 (NDUFS1: ~50% — CRITICAL DDx)
  vs NDUFS4 (N-module accessory):
    • NO olfactory bulb MRI lesions in ACAD9 (NDUFS4: 52–65% — PATHOGNOMONIC for NDUFS4)
  vs NDUFV1 (N-module FMN/N3):
    • Minimal leukodystrophy/white matter T2 in ACAD9 (NDUFV1: ~40–50%)
  vs NDUFV2 (~80% HCM) and SCO2 (~100% HCM):
    • ACAD9 ALSO has HCM (55–65%) — does NOT rule out ACAD9; biochemistry distinguishes:
      CIV NORMAL in ACAD9 vs CIV severely reduced in SCO2.
  vs POLG/DGUOK (mtDNA depletion):
    • NO hepatopathy in ACAD9 (POLG Alpers: ~80%; DGUOK: ~90%)
  vs ETFDH/ETFA/ETFB (riboflavin-responsive MADD/GA2):
    • ACAD9: CI deficiency isolated; NO fatty-acid β-oxidation disorder; NO significant
      organic aciduria (C4-C8, glutaric, C5-DC); ACAD9 serum acylcarnitines NORMAL.
    • ETFDH/MADD: multi-acyl-CoA dehydrogenase deficiency → urine organic acids show
      C4-C8 dicarboxylic acids + glutaric acid; serum acylcarnitines elevated (C6, C8,
      C10, C14, C16 + short-chain species). CRITICAL DDx for riboflavin-responsive
      patients: normal acylcarnitine profile + CI deficiency = ACAD9; abnormal
      acylcarnitine profile = ETFDH/MADD.
  vs NDUFAF2 (N-Q module assembly-factor swap):
    • ACAD9 affects ND2/ND5 module (membrane arm; downstream of N-module/Q-module);
      NDUFAF2 affects N-Q module interface (matrix arm). Different BN-PAGE intermediate
      patterns. ACAD9 has riboflavin response; NDUFAF2 does not. ACAD9 exercise
      intolerance very prominent; NDUFAF2 typically more severe infantile Leigh.

FOUNDER / RECURRENT MUTATIONS:
  p.Arg518His  c.1553G>A  — most common European founder (~40% of all reported alleles);
                             exercise-intolerance-dominant phenotype; riboflavin-responsive;
                             first reported Haack 2010 AmJHumGenet
  p.Arg532Trp  c.1594C>T  — second most common; exercise intolerance + HCM; riboflavin-
                             responsive; European
  p.Arg266Gln  c.797G>A   — severe early-onset Leigh-like; infantile; HCM prominent;
                             less riboflavin-responsive; common in Middle Eastern / Asian
  p.Asp562Gly  c.1685A>G  — MCIA complex integrity disruption; neonatal severe; null-
                             like phenotype; no riboflavin response
  c.IVS2+1G>A             — splice donor; partial ACAD9 residual; moderate phenotype

THERAPY — ACAD9 / CI-LEIGH SPECIFICS:
  No curative ACAD9 rescue therapy available.
  RIBOFLAVIN (B2) — Level B evidence for ACAD9 (strongest evidence of ANY CI deficiency):
    High-dose riboflavin 100–300 mg/day (in 3 divided doses) improves CI activity,
    exercise tolerance, and reduces HCM in ~50–60% of patients. Trial riboflavin for
    all newly diagnosed ACAD9 patients; monitor CI activity at 3 months; continue if
    any response detected. Evidence base: Schiff 2015 JMedGenet (n=120), Haack 2010.
  Absolute contraindications (direct CI inhibitors / mito toxins):
    Metformin — directly inhibits CI at ND1/quinone-binding site (ACAD9 ND2/ND5 territory)
    Valproate — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression
    Linezolid — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
    Chloramphenicol — same mt-ribosomal mechanism
  CONTRAINDICATED:
    Ketogenic diet — forces NADH → β-oxidation; ACAD9 doubly implicated: (1) CI cannot
    re-oxidise NADH from β-oxidation; (2) ACAD9 vestigial ACAD activity may be stressed
    by excess fatty acid substrates; KD worsens lactic acidosis catastrophically.
  AVOID / HIGH CAUTION:
    Propofol — PRIS + secondary CIV inhibition; dual ETC failure
    Phenobarbital — secondary CI inhibitor; use LEV first
  LEVEL C cofactors / supportive:
    CoQ10 (ubiquinol) — electron acceptor downstream of failed CI
    Thiamine (B1) — MANDATORY empiric: mimics SLC19A3/BTD (treatable)
    Biotin — MANDATORY empiric: BTD (treatable mimic)
    L-Carnitine — secondary carnitine deficiency may occur (ACAD superfamily overlap)
    Succinate — CII bypass; bypasses ACAD9-failed CI
  Preferred AED: Levetiracetam (LEV) — renal excretion; no mito toxicity
  Exercise: Avoid extreme overexertion; moderate-intensity activity with careful monitoring
  Glucose: IV dextrose GIR 6–8 mg/kg/min — NEVER fast
  Anaesthesia: sevoflurane (NOT propofol)
"""

from __future__ import annotations
import random
from typing import Any

SEED = 675
GENE = "ACAD9"
DISEASE = "ACAD9 Complex I Deficiency — MCIA Complex / Riboflavin-Responsive CI Assembly Factor (Exercise Intolerance / Leigh-like)"
OMIM_GENE = "611103"
OMIM_DISEASE = "256000"
CHROMOSOME = "3q21.3"
INHERITANCE = "AR"

# ---------------------------------------------------------------------------
# Synthetic 40-patient cohort  (seed-675, reflects published literature)
# ---------------------------------------------------------------------------

def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    sexes = ["M", "F"]
    mutations = [
        "p.Arg518His / p.Arg518His (hom, European founder)",
        "p.Arg518His / p.Arg532Trp (compound het)",
        "p.Arg518His / p.Arg266Gln (compound het)",
        "p.Arg266Gln / p.Arg266Gln (hom, consanguineous)",
        "p.Asp562Gly / p.Arg518His (compound het)",
        "p.Arg532Trp / p.Arg532Trp (hom)",
        "p.Arg266Gln / c.IVS2+1G>A (compound het)",
        "p.Asp562Gly / p.Asp562Gly (hom, consanguineous)",
        "Novel biallelic LOF (frameshift/splice)",
        "p.Arg518His / novel missense (compound het)",
    ]
    regions = [
        "European", "MENA", "South Asian", "East Asian",
        "North African", "Latin American", "Sub-Saharan African",
    ]
    ci_act_ranges = [(10, 18), (15, 25), (20, 30)]

    cohort = []
    for i in range(1, n + 1):
        age_onset = round(rng.uniform(0.3, 30), 1)
        sex = rng.choice(sexes)
        mut = rng.choice(mutations)
        region = rng.choice(regions)
        ci_lo, ci_hi = rng.choice(ci_act_ranges)
        ci_activity = round(rng.uniform(ci_lo, ci_hi), 1)
        # riboflavin responders
        riboflavin_responder = rng.random() < 0.55
        cohort.append({
            "id": i,
            "age_onset_months": round(age_onset * 12),
            "sex": sex,
            "mutation": mut,
            "region": region,
            "ci_activity_pct": ci_activity,
            "exercise_intolerance":   rng.random() < 0.88,
            "leigh_mri":              rng.random() < 0.50,
            "lactic_acidosis":        rng.random() < 0.80,
            "hypotonia":              rng.random() < 0.70,
            "psychomotor_regression": rng.random() < 0.55,
            "hcm":                    rng.random() < 0.60,
            "respiratory_compromise": rng.random() < 0.35,
            "seizures":               rng.random() < 0.38,
            "ataxia":                 rng.random() < 0.30,
            "skeletal_myopathy":      rng.random() < 0.82,
            "peripheral_neuropathy":  rng.random() < 0.04,
            "olfactory_bulb_lesions": rng.random() < 0.03,
            "leukodystrophy":         rng.random() < 0.05,
            "hepatopathy":            rng.random() < 0.04,
            "riboflavin_responder":   riboflavin_responder,
            "outcome": rng.choice(
                ["deceased before 3yr"] * 6 +
                ["deceased before 10yr"] * 7 +
                ["alive, severe disability"] * 10 +
                ["alive, moderate disability"] * 10 +
                ["alive, mild disability (riboflavin-responsive)"] * 7
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
        "gene_full_name": "Acyl-CoA Dehydrogenase Family, Member 9",
        "also_known_as":  "ACAD9 / MCIA Complex Scaffold / Riboflavin-Responsive CI Assembly Factor",
        "disease":        DISEASE,
        "omim_gene":      OMIM_GENE,
        "omim_disease":   OMIM_DISEASE,
        "chromosome":     CHROMOSOME,
        "inheritance":    INHERITANCE,
        "protein": {
            "size_aa":   621,
            "size_kda":  70.0,
            "fold":      "ACAD superfamily fold — FAD-binding domain + acyl-CoA binding vestiges; evolved toward CI assembly chaperone function; retains FAD-binding pocket (basis of riboflavin responsiveness)",
            "module":    "MCIA complex scaffold — ND2/ND5 membrane arm module (early CI membrane arm assembly); forms tetrameric MCIA complex with NDUFAF1 (CIA30), ECSIT, and TMEM126B",
            "fe_s_cluster": False,
            "fad_binding": True,
            "function":  "CI assembly factor; central scaffold of the MCIA (Mitochondrial Complex I Assembly) complex; required for ND2/ND5 module biogenesis; recruits NDUFAF1, ECSIT, TMEM126B; vestigial ACAD-family FAD domain underlies riboflavin responsiveness; loss disrupts MCIA complex → ND2/ND5 module assembly failure → CI membrane arm absent/reduced.",
        },
        "key_pathway_note": (
            "ACAD9 is the ONLY CI assembly factor with established RIBOFLAVIN (B2) responsiveness. "
            "ACAD9 belongs to the acyl-CoA dehydrogenase (ACAD) superfamily and retains a FAD-binding "
            "domain, despite having largely lost conventional fatty-acid β-oxidation enzymatic function. "
            "High-dose riboflavin (100–300 mg/day) appears to stabilise mutant ACAD9 protein and "
            "partially restore MCIA complex integrity, improving CI activity and exercise tolerance in "
            "~50–60% of patients. This riboflavin response is the most diagnostically and therapeutically "
            "important feature of ACAD9 deficiency and is unique among all CI assembly factors. "
            "ACAD9 also shows a bimodal severity spectrum: severe Leigh-like infantile presentation "
            "(p.Arg266Gln; p.Asp562Gly) versus exercise-intolerance-dominant adolescent/adult phenotype "
            "(p.Arg518His — the most common European founder mutation at ~40% of alleles). In adult-onset "
            "cases, Leigh MRI may be absent and exercise intolerance may be the sole finding."
        ),
        "biochemical_fingerprint": {
            "complex_I":   "10–30 % of control (REDUCED — ISOLATED CI DEFICIENCY; milder residual than structural CI subunit genes)",
            "complex_II":  "NORMAL (SDHA — all nuclear; unaffected)",
            "complex_III": "NORMAL",
            "complex_IV":  "NORMAL — key DDx: SCO2 (CIV reduced, HCM 100%); SURF1 (CIV reduced, Leigh); COX10 (CIV reduced, tubulopathy)",
            "acylcarnitines": "NORMAL (critical DDx vs ETFDH/MADD which shows C6-C14 elevations)",
            "urine_organic_acids": "NORMAL (DDx vs ETFDH/MADD which shows C4-C8 dicarboxylic + glutaric acid)",
        },
        "cohort": {
            "n":                       n,
            "seed":                    SEED,
            "mean_age_onset_months":   round(sum(p["age_onset_months"] for p in pts) / n),
            "ci_activity_mean_pct":    round(sum(ci_vals) / n, 1),
            "ci_activity_range_pct":   f"{min(ci_vals):.1f}–{max(ci_vals):.1f}",
            "riboflavin_responders_pct": _pct("riboflavin_responder"),
        },
        "feature_frequencies_pct": {
            "exercise_intolerance":     _pct("exercise_intolerance"),
            "skeletal_myopathy":        _pct("skeletal_myopathy"),
            "hcm":                      _pct("hcm"),
            "lactic_acidosis":          _pct("lactic_acidosis"),
            "hypotonia":                _pct("hypotonia"),
            "psychomotor_regression":   _pct("psychomotor_regression"),
            "leigh_mri":                _pct("leigh_mri"),
            "seizures":                 _pct("seizures"),
            "respiratory_compromise":   _pct("respiratory_compromise"),
            "ataxia":                   _pct("ataxia"),
            "peripheral_neuropathy":    _pct("peripheral_neuropathy"),
            "olfactory_bulb_lesions":   _pct("olfactory_bulb_lesions"),
            "leukodystrophy":           _pct("leukodystrophy"),
            "hepatopathy":              _pct("hepatopathy"),
        },
        "key_ddx": [
            {
                "feature":        "RIBOFLAVIN-RESPONSIVE (~50–60%) — PATHOGNOMONIC for ACAD9 among CI assembly factors",
                "significance":   "Only CI assembly factor with riboflavin responsiveness. FAD-binding domain (ACAD superfamily) underlies response. Trial riboflavin in ALL ACAD9 patients — 100–300 mg/day × 3 months, monitor CI activity. Level B evidence.",
                "target_gene":    "ACAD9",
                "target_freq_pct": 55,
            },
            {
                "feature":        "ACAD9 vs ETFDH/MADD — CRITICAL DDx for riboflavin-responsive patients",
                "significance":   "Both ACAD9 and ETFDH (MADD/GA2) are riboflavin-responsive. Key distinction: ACAD9 has NORMAL acylcarnitines + NORMAL urine organic acids + ISOLATED CI deficiency. ETFDH/MADD has abnormal acylcarnitines (elevated C6, C8, C10, C14) + organic aciduria (C4-C8 dicarboxylic acids, glutaric acid) + multi-acyl-CoA dehydrogenase deficiency. Acylcarnitine profile + urine organic acids MANDATORY before diagnosing ACAD9.",
                "target_gene":    "ETFDH / ETFA / ETFB (MADD/GA2)",
                "target_freq_pct": 0,
            },
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
                "feature":        "HCM present in ACAD9 (~60%) — does NOT rule out ACAD9",
                "significance":   "HCM is common in ACAD9 (unlike most CI structural subunit genes). HCM alone does NOT point to SCO2/NDUFV2. Biochemical distinction: ACAD9 has CIV NORMAL; SCO2 has CIV severely reduced. ACAD9 riboflavin-responsive; SCO2 not.",
                "target_gene":    "SCO2 (CIV, HCM 100%) / NDUFV2 (CI, HCM 80%)",
                "target_freq_pct": 60,
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
            {
                "feature":        "Exercise intolerance as first/only presentation (adolescent/adult p.Arg518His)",
                "significance":   "In older-onset ACAD9, Leigh MRI may be absent. Exercise intolerance + HCM + CI deficiency + riboflavin response = diagnostic. KEY DDx vs POLG adPEO (onset adult, exercise intolerance) by mtDNA multiple deletions on muscle biopsy PCR.",
                "target_gene":    "POLG (adPEO) / TWNK (PEO)",
                "target_freq_pct": 0,
            },
        ],
        "absolute_contraindications": [
            "Metformin — directly inhibits CI at ND1/quinone-binding site (ND2 module territory where ACAD9/MCIA assembles membrane arm)",
            "Valproate (VPA) — triple mechanism: CoA sequestration + POLG inhibition + mt ND-subunit expression block (all 7 mt-ND subunits)",
            "Linezolid — inhibits mitochondrial 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits (MT-ND1 through MT-ND6 + MT-ND4L)",
            "Chloramphenicol — same mt-ribosomal mechanism as linezolid; equally contraindicated",
        ],
        "contraindicated": [
            "Ketogenic diet — forces NADH → β-oxidation; ACAD9 doubly implicated: (1) CI absent → NADH cannot be re-oxidised → lactic acidosis worsens; (2) ACAD9 vestigial ACAD activity may be stressed by excess fatty acid CoA substrates",
        ],
        "preferred_treatments": [
            "Riboflavin (B2) — LEVEL B evidence: 100–300 mg/day in 3 divided doses; ~50–60% respond; monitor CI activity at 3 months; the ONLY CI assembly factor with established riboflavin response",
            "LEV (levetiracetam) — AED first-line: renal excretion, no mito toxicity",
            "CoQ10 (ubiquinol) — electron acceptor at quinone site; downstream of failed CI",
            "Thiamine (B1) — MANDATORY empiric: SLC19A3/BTD treatable mimic",
            "Biotin — MANDATORY empiric: BTD treatable mimic",
            "Succinate — CII bypass; bypasses ACAD9-failed CI ND2/ND5 assembly defect entirely",
            "L-Carnitine — may have secondary carnitine deficiency (ACAD superfamily metabolic overlap)",
            "IV dextrose GIR 6–8 mg/kg/min — never fast",
            "Sevoflurane (not propofol) for anaesthesia",
            "Moderate exercise + physiotherapy — avoid extreme overexertion; gradual rehabilitation beneficial in exercise-intolerance-dominant phenotype",
        ],
        "key_references": [
            "Haack TB et al (2010) AmJHumGenet — first ACAD9 mutations in CI deficiency",
            "Nouws J et al (2010) CellMetab — ACAD9 CI assembly factor function, MCIA complex",
            "Schiff M et al (2015) JMedGenet — ACAD9 natural history + riboflavin treatment (n=120)",
            "Liang WC et al (2009) Brain — riboflavin-responsive CI deficiency (early ACAD9)",
            "Guerrero-Castillo S et al (2017) CellMetab — CI assembly MCIA complex ACAD9",
        ],
    }


# ---------------------------------------------------------------------------
# Public API: get_breakdown
# ---------------------------------------------------------------------------

def get_breakdown() -> dict[str, Any]:
    pts = _cohort()
    n = len(pts)

    features = [
        "exercise_intolerance", "skeletal_myopathy", "hcm",
        "lactic_acidosis", "hypotonia", "psychomotor_regression",
        "leigh_mri", "seizures", "respiratory_compromise", "ataxia",
        "peripheral_neuropathy", "olfactory_bulb_lesions", "leukodystrophy", "hepatopathy",
    ]
    feature_frequencies = {
        f: {"count": sum(1 for p in pts if p.get(f)), "pct": _pct(f)}
        for f in features
    }

    # CI activity histogram
    ci_vals = [p["ci_activity_pct"] for p in pts]
    bins = ["0–10%", "10–15%", "15–20%", "20–25%", "25–30%", ">30%"]
    counts = [
        sum(1 for v in ci_vals if v < 10),
        sum(1 for v in ci_vals if 10 <= v < 15),
        sum(1 for v in ci_vals if 15 <= v < 20),
        sum(1 for v in ci_vals if 20 <= v < 25),
        sum(1 for v in ci_vals if 25 <= v < 30),
        sum(1 for v in ci_vals if v >= 30),
    ]

    # Outcome distribution
    out_dist: dict[str, int] = {}
    for p in pts:
        out_dist[p["outcome"]] = out_dist.get(p["outcome"], 0) + 1

    # Mutation distribution
    mut_dist: dict[str, int] = {}
    for p in pts:
        mut_dist[p["mutation"]] = mut_dist.get(p["mutation"], 0) + 1

    # Region distribution
    reg_dist: dict[str, int] = {}
    for p in pts:
        reg_dist[p["region"]] = reg_dist.get(p["region"], 0) + 1

    # Sex distribution
    sex_dist = {
        "M": sum(1 for p in pts if p["sex"] == "M"),
        "F": sum(1 for p in pts if p["sex"] == "F"),
    }

    # Riboflavin responders
    ribo_resp = sum(1 for p in pts if p.get("riboflavin_responder"))

    return {
        "n":                     n,
        "patients":              pts,
        "feature_frequencies":   feature_frequencies,
        "ci_activity_histogram": {"bins": bins, "counts": counts},
        "outcome_distribution":  out_dist,
        "mutation_distribution": mut_dist,
        "region_distribution":   reg_dist,
        "sex_distribution":      sex_dist,
        "riboflavin_responders": ribo_resp,
        "riboflavin_responder_pct": round(ribo_resp / n * 100),
        "mean_age_onset_months": round(sum(p["age_onset_months"] for p in pts) / n),
        "ci_activity_mean":      round(sum(ci_vals) / n, 1),
    }


# ---------------------------------------------------------------------------
# Public API: get_definitions
# ---------------------------------------------------------------------------

def get_definitions() -> dict[str, Any]:
    pharmacology = [
        {
            "term": "Riboflavin (B2) — LEVEL B Evidence (UNIQUE to ACAD9)",
            "category": "treatment",
            "detail": (
                "Riboflavin is the precursor to FMN (flavin mononucleotide) and FAD (flavin adenine "
                "dinucleotide). ACAD9 retains an FAD-binding domain from its ACAD superfamily origin. "
                "High-dose riboflavin (100–300 mg/day in 3 divided doses) appears to stabilise mutant "
                "ACAD9 protein and partially restore MCIA complex integrity. Clinical response in "
                "~50–60% of ACAD9 patients: improved CI activity, reduced exercise intolerance, "
                "HCM regression. Level B evidence for ACAD9 — the ONLY CI assembly factor with "
                "established riboflavin responsiveness. Urine turns bright yellow (expected). "
                "Monitor CI activity at 3 and 6 months to confirm response. Continue indefinitely "
                "if any response detected. No serious adverse effects at 100–300 mg/day."
            ),
        },
        {
            "term": "Metformin — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Metformin directly inhibits mitochondrial Complex I at the ND1/quinone-binding site "
                "(ND2 membrane arm territory where ACAD9/MCIA complex assembles CI). In ACAD9 "
                "deficiency, CI is already severely reduced (10–30%). Further metformin inhibition "
                "precipitates life-threatening lactic acidosis. ABSOLUTE contraindication."
            ),
        },
        {
            "term": "Valproate (VPA) — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "VPA inhibits CI via three mechanisms: (1) CoA sequestration (valproyl-CoA trapping); "
                "(2) direct POLG mitochondrial polymerase inhibition → mtDNA depletion; (3) impairs "
                "expression of mt-encoded ND subunits. All three mechanisms worsen ACAD9 CI deficiency. "
                "ABSOLUTE contraindication."
            ),
        },
        {
            "term": "Linezolid — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Linezolid inhibits mitochondrial 23S rRNA → blocks synthesis of all 7 mt-encoded ND "
                "subunits (MT-ND1 through MT-ND4L). In ACAD9 deficiency, CI membrane arm is already "
                "partially assembled (ND2/ND5 module failure); eliminating mt-ND synthesis completely "
                "abolishes any residual CI. ABSOLUTE contraindication."
            ),
        },
        {
            "term": "Chloramphenicol — ABSOLUTE CI",
            "category": "absolute_contraindication",
            "detail": (
                "Same mitochondrial ribosomal inhibition mechanism as linezolid. ABSOLUTE "
                "contraindication in all CI-deficiency including ACAD9."
            ),
        },
        {
            "term": "Ketogenic Diet (KD) — CONTRAINDICATED",
            "category": "contraindicated",
            "detail": (
                "KD forces NADH → β-oxidation. In ACAD9 deficiency, CI is absent/severely reduced "
                "(ND2/ND5 module assembly failed). NADH from β-oxidation cannot be re-oxidised → "
                "NADH accumulates → lactic acidosis worsens catastrophically. Additionally, ACAD9 "
                "has vestigial ACAD activity and may be stressed by excess acyl-CoA substrates "
                "from KD fatty acid loading. CONTRAINDICATED."
            ),
        },
        {
            "term": "Propofol — AVOID",
            "category": "high_caution",
            "detail": (
                "PRIS (propofol infusion syndrome) inhibits CIV (COX). In ACAD9 deficiency CI is "
                "already severely reduced. Dual ETC failure (CI + CIV) is catastrophic. "
                "Use sevoflurane for anaesthesia instead."
            ),
        },
        {
            "term": "CoQ10 (Ubiquinol) — Level C",
            "category": "treatment",
            "detail": (
                "CoQ10 is the electron acceptor at the quinone-binding site of CI (downstream of "
                "ACAD9's MCIA complex ND2/ND5 assembly target). Level C evidence; clinically used "
                "empirically in CI-deficiency including ACAD9."
            ),
        },
        {
            "term": "L-Carnitine — Level C (secondary deficiency possible)",
            "category": "treatment",
            "detail": (
                "ACAD9 belongs to the ACAD protein superfamily, which overlaps with the fatty acid "
                "β-oxidation pathway. Secondary carnitine deficiency may develop in some ACAD9 "
                "patients due to metabolic stress on the ACAD pathway. Monitor free carnitine; "
                "supplement if depleted."
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
                "deficiency) — treatable mimics. Empiric biotin mandatory before genetic confirmation."
            ),
        },
        {
            "term": "Succinate — Level C (CII Bypass)",
            "category": "treatment",
            "detail": (
                "Succinate provides substrate to Complex II (SDHA — all nuclear, CII NORMAL in "
                "ACAD9 deficiency). CII feeds electrons directly to CoQ10, bypassing failed CI "
                "ND2/ND5 module entirely. This bypass is specific to CI deficiency."
            ),
        },
        {
            "term": "LEV (Levetiracetam) — Preferred AED",
            "category": "treatment",
            "detail": (
                "Levetiracetam is the preferred AED in CI-deficiency: renal excretion (avoids "
                "hepatic metabolism), no mitochondrial toxicity, no CYP interactions."
            ),
        },
    ]

    glossary = [
        {
            "term": "MCIA Complex (Mitochondrial Complex I Assembly Complex)",
            "definition": (
                "The MCIA complex is a four-protein CI assembly factor complex consisting of "
                "ACAD9, NDUFAF1 (CIA30), ECSIT, and TMEM126B. ACAD9 is the central scaffold "
                "recruited first; NDUFAF1 forms a tight binary complex with ACAD9; ECSIT and "
                "TMEM126B are then recruited to complete the tetrameric MCIA complex. The MCIA "
                "complex is essential for biogenesis of the ND2 and ND5 membrane arm modules of "
                "Complex I. Disruption of any MCIA subunit (ACAD9, NDUFAF1, ECSIT, or TMEM126B "
                "deficiency) prevents ND2/ND5 module assembly → CI membrane arm absent → "
                "isolated CI deficiency."
            ),
        },
        {
            "term": "Riboflavin Responsiveness (ACAD9-specific)",
            "definition": (
                "ACAD9 is the only CI assembly factor with established riboflavin (B2) "
                "responsiveness. ACAD9 retains a FAD-binding domain from its ACAD superfamily "
                "origin. High-dose riboflavin → increased cellular FAD availability → stabilises "
                "mutant ACAD9 protein fold → partially restores MCIA complex integrity and CI "
                "membrane arm assembly. Clinical response rate ~50–60% with 100–300 mg/day. "
                "Monitoring: CI activity at 3 months. Absence of response does NOT exclude ACAD9 "
                "(remaining 40–50% are non-responders, typically LOF/null alleles or alleles "
                "disrupting MCIA complex interaction rather than FAD-binding fold)."
            ),
        },
        {
            "term": "ACAD Superfamily (Acyl-CoA Dehydrogenase Family)",
            "definition": (
                "The ACAD superfamily includes enzymes involved in fatty acid β-oxidation: VLCAD, "
                "LCAD, MCAD, SCAD, ETFDH, ACAD8, ACAD10, and ACAD9. All share a conserved FAD-"
                "binding domain and a glutaryl-CoA binding pocket. ACAD9 has diverged from the "
                "other family members in that its conventional enzymatic activity (acyl-CoA "
                "dehydrogenation) is negligible in vivo — it has evolved to function primarily as "
                "a CI assembly factor. ACAD9 deficiency does NOT cause typical fatty acid oxidation "
                "disorder (no abnormal acylcarnitines, no organic aciduria) — this distinguishes "
                "ACAD9 from MADD/GA2 (ETFDH deficiency, also riboflavin-responsive)."
            ),
        },
        {
            "term": "ND2 / ND5 Membrane Arm Modules",
            "definition": (
                "The CI membrane arm is built from multiple sub-assemblies called 'modules'. The "
                "ND2 module (containing MT-ND2, MT-ND3, MT-ND4L and nuclear subunits NDUFB6, "
                "NDUFB8, NDUFB7) and the ND5 module (containing MT-ND5 and associated nuclear "
                "subunits including NDUFB10) are assembled with the help of the MCIA complex. "
                "ACAD9 deficiency prevents assembly of these modules → CI membrane arm truncated "
                "→ CI holoenzyme absent or severely reduced on BN-PAGE."
            ),
        },
        {
            "term": "Bimodal Severity Spectrum (ACAD9)",
            "definition": (
                "ACAD9 deficiency shows two distinct clinical presentations depending on genotype: "
                "1) Leigh-like SEVERE INFANTILE: homozygous p.Arg266Gln or p.Asp562Gly → Leigh MRI, "
                "HCM, hypotonia, lactic acidosis, early childhood death; less riboflavin-responsive. "
                "2) EXERCISE-INTOLERANCE-DOMINANT adolescent/adult: p.Arg518His (European founder, "
                "~40% of alleles) or p.Arg532Trp → exercise intolerance, skeletal myopathy, "
                "mild-moderate HCM; Leigh MRI may be absent; riboflavin-responsive ~60–70%. "
                "This bimodal spectrum is unique to ACAD9 and driven by the FAD-binding domain "
                "stability of specific alleles."
            ),
        },
        {
            "term": "p.Arg518His — European Founder Mutation",
            "definition": (
                "p.Arg518His (c.1553G>A) is the most common ACAD9 pathogenic variant, accounting "
                "for approximately 40% of all disease alleles in reported series. It is particularly "
                "prevalent in European (especially Dutch/German) populations. This missense variant "
                "affects the FAD-binding domain: the histidine substitution destabilises the FAD "
                "pocket, reducing ACAD9 protein stability. Riboflavin supplementation stabilises "
                "the mutant protein, explaining the preferential riboflavin response associated with "
                "this variant. Phenotype: exercise intolerance + HCM + skeletal myopathy; Leigh MRI "
                "variable; onset childhood to adulthood; Haack 2010 original discovery cohort."
            ),
        },
        {
            "term": "Isolated CI Deficiency (ACAD9)",
            "definition": (
                "In ACAD9 deficiency, only Complex I (CI) activity is reduced; Complexes II, III, "
                "and IV activities are within normal range. CI activity is 10–30% of control — "
                "milder residual activity compared to structural CI subunit genes (typically 5–20%). "
                "This relatively milder CI deficiency explains the exercise-intolerance-dominant "
                "phenotype in some patients (partial CI function sufficient for resting, but "
                "insufficient for exercise demands). Normal acylcarnitines and normal urine organic "
                "acids confirm that ACAD9's vestigial ACAD activity is not causing a secondary "
                "fatty acid oxidation disorder."
            ),
        },
    ]

    return {
        "pharmacology":              pharmacology,
        "glossary":                  glossary,
        "gene_full":                 "Acyl-CoA Dehydrogenase Family, Member 9",
        "historical_names":          ["ACAD9", "MCIA Scaffold"],
        "omim_gene":                 OMIM_GENE,
        "omim_disease":              OMIM_DISEASE,
        "chromosome":                CHROMOSOME,
        "inheritance_detail":        "Autosomal recessive (AR): biallelic pathogenic variants required; both sexes affected equally; consanguinity increases risk; founder p.Arg518His enriched in European (Dutch/German) populations",
        "protein_size_aa":           621,
        "protein_size_kda":          70.0,
        "tm_helices":                0,
        "fad_binding":               True,
        "module":                    "MCIA complex scaffold — ND2/ND5 membrane arm module (early CI membrane arm assembly); central scaffold of the ACAD9-NDUFAF1-ECSIT-TMEM126B tetrameric MCIA complex",
        "fe_s_cluster":              False,
        "ci_activity_range":         "10–30 % of control (isolated CI deficiency; milder residual than structural CI subunit genes)",
        "cii_ciii_civ_normal":       True,
        "acylcarnitines_normal":     True,
        "urine_organic_acids_normal": True,
        "bn_page_pattern":           "Severely reduced CI holoenzyme; ND2/ND5 sub-assembly intermediates may be detectable — ND2/ND5 membrane arm module failure; distinct from N-module intermediates (NDUFAF2/NDUFA12) or Q-module intermediates (NDUFS3)",
        "riboflavin_response_rate_pct": 55,
        "key_distinguishing_feature": "ONLY CI assembly factor with RIBOFLAVIN (B2) RESPONSIVENESS (~50–60%); FAD-binding domain (ACAD superfamily origin) underlies response; bimodal severity: Leigh-like infantile (p.Arg266Gln) vs exercise-intolerance-dominant (p.Arg518His European founder 40%); normal acylcarnitines + normal organic acids rule out ETFDH/MADD (other riboflavin-responsive condition); MCIA complex scaffold with NDUFAF1, ECSIT, TMEM126B",
    }


if __name__ == "__main__":
    import json
    print("=== ACAD9 Overview ===")
    print(json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== Breakdown (sample) ===")
    bd = get_breakdown()
    print(f"n={bd['n']}, riboflavin_responders={bd['riboflavin_responders']} ({bd['riboflavin_responder_pct']}%)")
    print(f"sex_dist={bd['sex_distribution']}")
    print("\n=== Definitions (terms) ===")
    defs = get_definitions()
    print(f"pharmacology_terms={[p['term'] for p in defs['pharmacology']]}")
    print(f"glossary_terms={[g['term'] for g in defs['glossary']]}")
