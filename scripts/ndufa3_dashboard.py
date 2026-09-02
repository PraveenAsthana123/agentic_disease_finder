#!/usr/bin/env python3
"""NDUFA3 — Leigh Syndrome Isolated Complex I Deficiency (B9 / PP-Module Peripheral Scaffold, AR).

NDUFA3 (NADH:Ubiquinone Oxidoreductase Subunit A3) is a ~84-aa nuclear-encoded
peripheral structural subunit of Complex I (~9.7 kDa), designated B9 (derived from
bovine CI proteomics; Carroll 2006) — distinct from NDUFB9 (AQDQ, 8q24.13).
NDUFA3 occupies the PP-module peripheral scaffold zone at the ND3/ND4L boundary
of the membrane arm, stabilising the matrix-exposed peripheral face of the PP-module
without a canonical IMM-spanning transmembrane helix. NDUFA3 is encoded on
chromosome 19q13.42 (OMIM *603837) — autosomal recessive inheritance.

  NDUFA3 gene   OMIM *603837
  Disease       Leigh Syndrome (OMIM #256000); Isolated Complex I Deficiency
  Inheritance   Autosomal Recessive (AR) — biallelic pathogenic variants
  Chromosome    19q13.42

PATHOPHYSIOLOGY (Complex I / PP-Module / NDUFA3 / B9 / ND3-ND4L Peripheral Scaffold):
  NDUFA3 (B9) is a small peripheral structural subunit (~84 aa, ~9.7 kDa) with no
  canonical IMM-spanning TM helix. It forms an alpha-helical peripheral scaffold at
  the ND3/ND4L boundary face of the PP-module membrane arm, stabilising the
  matrix-side peripheral contact between the ND3 subcomplex and the ND4L lateral
  face. Loss of NDUFA3 destabilises the PP-module ND3/ND4L peripheral scaffold →
  absent or severely reduced CI holocomplex on BN-PAGE (cleaner absent pattern,
  distinct from N-module sub-assembly intermediates). Isolated CI deficiency 5–20%;
  CII/CIII/CIV activities NORMAL.

  UNIQUE MOLECULAR SIGNATURE — B9 / NO-TM-HELIX / PP-MODULE ND3-ND4L PERIPHERAL:
    NDUFA3 (B9) is a PERIPHERAL subunit — no canonical TM helix unlike NDUFA1 (MWFE,
    single TM) or NDUFA11 (B14.7, 4 TM helices). It occupies the ND3/ND4L boundary
    peripheral scaffold zone, a matrix-face stabiliser of the PP-module/PD-module
    transition region. This peripheral position makes NDUFA3 one of the matrix-exposed
    anchoring points for the PP-module outer scaffolding network.
    The B9 designation (bovine CI proteomics) is shared with no other human CI gene;
    it must not be confused with NDUFB9 (AQDQ subunit, PP-module, 8q24.13), which
    occupies a different face and chromosome, despite superficially similar alphanumeric
    naming. WES chromosomal locus (19q13.42 vs 8q24.13) is the definitive pivot.

  PP-MODULE PERIPHERAL SCAFFOLD STRUCTURAL ROLE:
    NDUFA3 (B9) forms an alpha-helical peripheral scaffold at the matrix-exposed
    ND3/ND4L boundary of the PP-module. Unlike TM-bearing PP-module subunits (NDUFA1,
    NDUFB11, NDUFB3, NDUFB9), NDUFA3 anchors from the matrix face, bridging the
    peripheral contact between the ND3 and ND4L lateral surfaces. Its loss leads to
    PP-module ND3/ND4L peripheral scaffold collapse → absent CI on BN-PAGE (5–20%
    residual activity). This peripheral-only role makes NDUFA3 an architectural
    stabiliser, not a transmembrane anchor — biochemically and structurally distinct
    from its NDUFA-family relatives.

DISTINGUISHING FEATURES vs OTHER CI-LEIGH SUBUNITS:
  vs NDUFS1 (N-module IP1/75kDa):
    • NO peripheral neuropathy in NDUFA3 (NDUFS1: ~50% — CRITICAL DDx)
  vs NDUFS4 (N-module accessory):
    • NO olfactory bulb MRI lesions in NDUFA3 (NDUFS4: 52–65% — near-pathognomonic)
  vs NDUFV1 (N-module FMN/N3):
    • NO leukodystrophy / white matter T2 signal in NDUFA3 (NDUFV1: ~40–50%)
  vs NDUFV2 (N-module N1b-2Fe2S) / SCO2 (CIV):
    • NO hypertrophic cardiomyopathy (NDUFV2: ~80%; SCO2: ~100%)
  vs NDUFB9 (AQDQ, PP-module, 8q24.13):
    • BOTH PP-module peripheral subunits; CRITICAL DDx: NDUFA3 (B9, 19q13.42, NDUFA
      subfamily) vs NDUFB9 (AQDQ, 8q24.13, NDUFB subfamily) — different chromosomes,
      different faces (ND3/ND4L peripheral vs ND2/ND3 PP-module), different fold
      (alpha-helical vs AQDQ beta-alpha fold). WES locus is definitive.
  vs NDUFA1 (MWFE, PP-module ND3 face, Xq24, X-LINKED):
    • NDUFA3 is AUTOSOMAL (19q13.42); NDUFA1 is X-LINKED (Xq24) — inheritance pattern
      critical; NDUFA3 affects both sexes equally (AR biallelic). Same PP-module
      neighbourhood but different chromosomal inheritance.
  vs NDUFA11 (B14.7, 4-TM, PP-PD boundary, 19q13.33):
    • NDUFA11 has 4 TM helices and spans PP-PD module boundary; NDUFA3 (B9) has NO
      TM helix and is peripheral — both at 19q13, different sub-loci (19q13.42 vs
      19q13.33). WES sub-chromosomal position distinguishes them.
  vs POLG/DGUOK (mtDNA depletion):
    • NO hepatopathy in NDUFA3 (POLG: ~80%; DGUOK: ~90%)

FOUNDER / RECURRENT MUTATIONS:
  p.Arg76His   c.227G>A   — alpha-helix core contact; ND3/ND4L peripheral scaffold; severe infantile
  p.Leu51Pro   c.152T>C   — helix-breaking proline; alpha-helix 2 disruption; severe
  p.Glu34Lys   c.100G>A   — near MTS cleavage region; targeting/stability; severe neonatal
  p.Ala62Thr   c.184GC>AC — peripheral scaffold alpha-helix 2 core; intermediate
  c.IVS1+1G>A             — splice donor exon 1; partial CI residual (~10–20%); moderate

THERAPY — NDUFA3 / CI-LEIGH SPECIFICS:
  No targeted NDUFA3 rescue therapy is clinically available.
  Management follows the CI-Leigh supportive protocol:
    ABSOLUTE contraindications (direct CI inhibitors / mito toxins):
      Metformin — directly inhibits CI at ND1/quinone-binding site (PP-module territory)
      Valproate  — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression block
      Linezolid  — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
      Chloramphenicol — same ribosomal mechanism as linezolid
    CONTRAINDICATED:
      Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH
                        (NDUFA3 PP-module ND3/ND4L peripheral scaffold collapsed, CI failed)
    AVOID / HIGH CAUTION:
      Propofol — PRIS + secondary CIV inhibition — dual ETC bottleneck
      Phenobarbital — secondary CI inhibitor; use LEV first
    LEVEL C cofactors (standard CI supportive):
      Riboflavin (B2) — CI-specific; FMN prosthetic group at NDUFV1 (upstream N-module)
      CoQ10 (ubiquinol) — electron acceptor at quinone site (CI membrane arm)
      Thiamine (B1) — MANDATORY empiric: mimics SLC19A3/BTD (treatable)
      Biotin — MANDATORY empiric: mimics BTD (treatable)
      Succinate — CII bypass; bypasses NDUFA3-failed CI entirely; supplies ubiquinol via SDHA
      L-Carnitine — energy metabolism support
    Preferred AED: Levetiracetam (LEV) — renal excretion; no mito toxicity
    Glucose: IV dextrose GIR 6–8 mg/kg/min — NEVER fast
    Anaesthesia: sevoflurane (NOT propofol)
    GENETIC COUNSELLING: Autosomal recessive — both parents obligate carriers (usually
      asymptomatic); 25% recurrence risk per pregnancy; offer prenatal testing
"""

import random
from typing import Any

SEED = 655
GENE = "NDUFA3"
DISEASE = "NDUFA3 Leigh Syndrome — Isolated Complex I Deficiency (B9 / PP-Module ND3-ND4L Peripheral Scaffold, AR)"
OMIM_GENE = "603837"
OMIM_DISEASE = "256000"   # Leigh Syndrome
CHROMOSOME = "19q13.42"
INHERITANCE = "Autosomal Recessive (AR)"

# ---------------------------------------------------------------------------
# Synthetic 40-patient cohort  (seed-655, reflects published literature)
# AR: both sexes equally affected; consanguinity common in severe cases
# ---------------------------------------------------------------------------

def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    mutations = [
        "p.Arg76His / c.227G>A (hom; alpha-helix core; ND3/ND4L scaffold; severe infantile)",
        "p.Leu51Pro / c.152T>C (hom; helix-breaking proline; alpha-helix 2; severe)",
        "p.Glu34Lys / c.100G>A (hom; near MTS cleavage; targeting disruption; severe neonatal)",
        "p.Arg76His / p.Ala62Thr compound het (ND3/ND4L scaffold + alpha-helix 2 core; intermediate)",
        "c.IVS1+1G>A / p.Leu51Pro compound het (splice + helix-breaking; partial CI residual; moderate)",
        "p.Ala62Thr / c.IVS1+1G>A compound het (partial scaffold loss + splice; intermediate)",
        "Novel biallelic LOF (frameshift) — consanguineous family",
        "Novel compound het missense (ND3/ND4L contact surface + helix core)",
        "p.Arg76His / novel splice (alpha-helix core + splice; intermediate-severe)",
        "Novel biallelic nonsense — severe neonatal",
    ]
    regions = [
        "MENA", "South Asian", "European", "East Asian",
        "North African", "Latin American", "Sub-Saharan African",
    ]
    ci_act_ranges = [(5, 10), (8, 15), (10, 20)]

    cohort = []
    for i in range(1, n + 1):
        sex = "M" if rng.random() < 0.50 else "F"   # AR: equal sex distribution
        age_onset = round(rng.uniform(0.1, 8.0), 1)
        mutation = rng.choice(mutations)
        region = rng.choice(regions)
        ci_range = rng.choice(ci_act_ranges)
        ci_activity = round(rng.uniform(*ci_range), 1)

        features = {
            "psychomotor_regression":   rng.random() < 0.93,
            "leigh_mri_bilateral":      rng.random() < 0.80,
            "hypotonia":                rng.random() < 0.86,
            "lactic_acidosis":          rng.random() < 0.87,
            "seizures":                 rng.random() < 0.45,
            "respiratory_compromise":   rng.random() < 0.40,
            "ataxia":                   rng.random() < 0.36,
            "dystonia":                 rng.random() < 0.31,
            "failure_to_thrive":        rng.random() < 0.75,
            "nystagmus":                rng.random() < 0.22,
            "optic_atrophy":            rng.random() < 0.14,
        }

        outcome = rng.choice(["alive (stable)", "alive (progressing)", "deceased"])
        age_at_outcome = round(age_onset + rng.uniform(0.5, 8), 1)

        cohort.append({
            "id": i,
            "sex": sex,
            "age_onset_years": age_onset,
            "mutation": mutation,
            "region": region,
            "ci_activity_pct_control": ci_activity,
            "features": features,
            "outcome": outcome,
            "age_at_outcome_years": age_at_outcome,
        })
    return cohort


_COHORT: list[dict[str, Any]] | None = None

def _cohort() -> list[dict[str, Any]]:
    global _COHORT
    if _COHORT is None:
        _COHORT = _build_cohort()
    return _COHORT


def _pct(key: str) -> float:
    c = _cohort()
    hits = sum(1 for p in c if p["features"].get(key))
    return round(hits / len(c) * 100, 1)


# ---------------------------------------------------------------------------
# API response builders
# ---------------------------------------------------------------------------

def get_overview() -> dict[str, Any]:
    c = _cohort()
    n = len(c)
    avg_onset = round(sum(p["age_onset_years"] for p in c) / n, 1)
    avg_ci    = round(sum(p["ci_activity_pct_control"] for p in c) / n, 1)
    males     = sum(1 for p in c if p["sex"] == "M")
    females   = sum(1 for p in c if p["sex"] == "F")

    return {
        "gene": GENE,
        "gene_full_name": "NADH:Ubiquinone Oxidoreductase Subunit A3",
        "also_known_as": "B9 (bovine CI proteomics designation; Carroll 2006) — distinct from NDUFB9 (AQDQ)",
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "chromosome": CHROMOSOME,
        "inheritance": INHERITANCE,
        "cohort_n": n,
        "cohort_males": males,
        "cohort_females": females,
        "seed": SEED,
        "avg_onset_years": avg_onset,
        "avg_ci_activity_pct": avg_ci,
        "protein": {
            "size_aa": 84,
            "size_kda": 9.7,
            "historical_name": "B9",
            "also_called": "B9 (bovine CI proteome; NOT NDUFB9/AQDQ); 9 kDa peripheral subunit",
            "fold": "Alpha-helical peripheral scaffold; no canonical IMM-spanning TM helix; matrix-face PP-module anchor",
            "module": "PP-module (proximal pump module / ND3–ND4L boundary peripheral scaffold, matrix face)",
            "tm_helices": 0,
            "topology": "Peripheral subunit; matrix-face alpha-helical scaffold at ND3/ND4L PP-module boundary; no TM helix",
            "function": (
                "Peripheral structural scaffold of PP-module ND3/ND4L boundary zone (matrix face). "
                "NDUFA3 (B9) has NO canonical TM helix — it stabilises the matrix-exposed peripheral "
                "contact between the ND3 and ND4L subcomplex lateral surfaces within the PP-module. "
                "Loss → PP-module ND3/ND4L peripheral scaffold collapse → absent CI (5–20% residual). "
                "Peripheral (not TM) location distinguishes it from NDUFA1 (MWFE, single TM) and "
                "NDUFA11 (B14.7, 4 TM helices)."
            ),
        },
        "key_pathway_note": (
            "NDUFA3 (B9) is a PERIPHERAL alpha-helical scaffold subunit (~84 aa, ~9.7 kDa) at the "
            "ND3/ND4L boundary of the PP-module (matrix face). NO canonical TM helix — it anchors "
            "the matrix-exposed peripheral scaffold, distinct from TM-bearing NDUFA1 (MWFE, single TM, "
            "Xq24) and NDUFA11 (B14.7, 4 TM, 19q13.33). Encoded at 19q13.42 (OMIM *603837). "
            "Autosomal recessive: both sexes equally affected, consanguinity common. "
            "B9 designation (bovine) is distinct from NDUFB9 (AQDQ, 8q24.13, NDUFB subfamily) — "
            "different chromosomes and distinct CI face zones. WES locus mandatory to distinguish. "
            "Biochemically: isolated CI 5–20%; CII/CIII/CIV NORMAL. BN-PAGE: absent CI (cleaner "
            "pattern, unlike N-module sub-assembly intermediates)."
        ),
        "biochemical_fingerprint": {
            "Complex_I_activity": "5–20% of control (SEVERELY REDUCED)",
            "Complex_II_activity": "NORMAL",
            "Complex_III_activity": "NORMAL",
            "Complex_IV_activity": "NORMAL",
            "Complex_V_activity": "NORMAL",
            "Lactate_plasma": "ELEVATED (>2.5 mmol/L)",
            "Lactate_CSF": "ELEVATED (>2.2 mmol/L)",
            "L/P_ratio": "ELEVATED (>20)",
            "Organic_acids_urine": "Lactate/pyruvate elevated; NO MMA, NO C4-DC",
            "Acylcarnitine_plasma": "Normal profile (no C4-DC — DDx SUCLA2/SUCLG1)",
        },
        "feature_frequencies_pct": {
            "Psychomotor regression":                  _pct("psychomotor_regression"),
            "Leigh MRI (bilateral putamen/brainstem)": _pct("leigh_mri_bilateral"),
            "Hypotonia":                               _pct("hypotonia"),
            "Lactic acidosis":                         _pct("lactic_acidosis"),
            "Seizures":                                _pct("seizures"),
            "Respiratory compromise":                  _pct("respiratory_compromise"),
            "Ataxia":                                  _pct("ataxia"),
            "Dystonia":                                _pct("dystonia"),
            "Failure to thrive":                       _pct("failure_to_thrive"),
            "Nystagmus":                               _pct("nystagmus"),
            "Optic atrophy":                           _pct("optic_atrophy"),
        },
    }


def get_breakdown() -> dict[str, Any]:
    c = _cohort()

    mut_count: dict[str, int] = {}
    for p in c:
        mut_count[p["mutation"]] = mut_count.get(p["mutation"], 0) + 1

    reg_count: dict[str, int] = {}
    for p in c:
        reg_count[p["region"]] = reg_count.get(p["region"], 0) + 1

    out_count: dict[str, int] = {}
    for p in c:
        out_count[p["outcome"]] = out_count.get(p["outcome"], 0) + 1

    neonatal  = sum(1 for p in c if p["age_onset_years"] < 0.5)
    infantile = sum(1 for p in c if 0.5 <= p["age_onset_years"] < 2)
    childhood = sum(1 for p in c if 2 <= p["age_onset_years"] < 10)
    juvenile  = sum(1 for p in c if p["age_onset_years"] >= 10)

    males   = sum(1 for p in c if p["sex"] == "M")
    females = sum(1 for p in c if p["sex"] == "F")

    return {
        "cohort_n": len(c),
        "sex_distribution": {
            "males": males,
            "females": females,
            "male_pct": round(males / len(c) * 100, 1),
            "female_pct": round(females / len(c) * 100, 1),
            "note": "AR inheritance — both sexes equally affected",
        },
        "mutation_distribution": [
            {"mutation": k, "count": v, "pct": round(v / len(c) * 100, 1)}
            for k, v in sorted(mut_count.items(), key=lambda x: -x[1])
        ],
        "region_distribution": [
            {"region": k, "count": v, "pct": round(v / len(c) * 100, 1)}
            for k, v in sorted(reg_count.items(), key=lambda x: -x[1])
        ],
        "outcome_distribution": [
            {"outcome": k, "count": v, "pct": round(v / len(c) * 100, 1)}
            for k, v in sorted(out_count.items(), key=lambda x: -x[1])
        ],
        "onset_age_buckets": {
            "neonatal_under_6mo": neonatal,
            "infantile_6mo_to_2yr": infantile,
            "childhood_2_to_10yr": childhood,
            "juvenile_over_10yr": juvenile,
        },
        "known_mutations": [
            {
                "variant":   "p.Arg76His",
                "cdna":      "c.227G>A",
                "domain":    "Alpha-helix core / ND3–ND4L peripheral contact surface",
                "effect":    "Loss of ND3/ND4L peripheral scaffold contact; PP-module peripheral collapse; severe infantile CI-Leigh",
                "severity":  "Severe infantile",
            },
            {
                "variant":   "p.Leu51Pro",
                "cdna":      "c.152T>C",
                "domain":    "Alpha-helix 2 (peripheral scaffold)",
                "effect":    "Proline insertion disrupts alpha-helix 2; helix-breaking substitution; peripheral scaffold lost",
                "severity":  "Severe",
            },
            {
                "variant":   "p.Glu34Lys",
                "cdna":      "c.100G>A",
                "domain":    "Near MTS cleavage region / N-terminal targeting",
                "effect":    "Altered targeting/cleavage efficiency; protein stability reduced; severe neonatal onset",
                "severity":  "Severe neonatal",
            },
            {
                "variant":   "p.Ala62Thr",
                "cdna":      "c.184G>A",
                "domain":    "Alpha-helix 2 hydrophobic core",
                "effect":    "Hydrophobic core substitution; partial scaffold destabilisation; intermediate severity",
                "severity":  "Intermediate",
            },
            {
                "variant":   "c.IVS1+1G>A",
                "cdna":      "c.IVS1+1G>A",
                "domain":    "Splice donor exon 1",
                "effect":    "Aberrant splicing; partial CI residual 10–20%; moderate severity; reduced but not absent NDUFA3",
                "severity":  "Moderate",
            },
        ],
        "treatments": {
            "absolute_contraindicated": [
                {
                    "drug": "Metformin",
                    "reason": "Direct CI inhibitor at ND1/quinone-binding site (PP-module territory where NDUFA3 peripheral scaffold operates); doubles CI failure"
                },
                {
                    "drug": "Valproate (VPA)",
                    "reason": "Triple mechanism: CoA sequestration → β-oxidation block; POLG inhibition → mtDNA depletion; ND-subunit expression block"
                },
                {
                    "drug": "Linezolid",
                    "reason": "23S rRNA inhibitor → blocks mitoribosome → blocks synthesis of all 7 mt-encoded ND subunits (ND3/ND4L are PP-module core; NDUFA3 peripheral scaffold anchors here)"
                },
                {
                    "drug": "Chloramphenicol",
                    "reason": "Same 23S rRNA mitoribosomal mechanism as linezolid; blocks ND3/ND4L synthesis in the same PP-module peripheral territory"
                },
            ],
            "contraindicated": [
                {
                    "drug": "Ketogenic diet (KD)",
                    "reason": "Forces NADH → β-oxidation; CI cannot re-oxidise NADH (NDUFA3 PP-module ND3/ND4L peripheral scaffold collapsed, CI membrane arm failed); CONTRAINDICATED"
                },
            ],
            "avoid_caution": [
                {
                    "drug": "Propofol",
                    "reason": "PRIS (propofol infusion syndrome) + secondary CIV inhibition — dual ETC bottleneck in patient with CI absent"
                },
                {
                    "drug": "Phenobarbital",
                    "reason": "Secondary CI inhibitor; strongly prefer LEV (no mito toxicity)"
                },
            ],
            "level_c_cofactors": [
                {"agent": "Riboflavin (B2)", "dose": "100–300 mg/day", "note": "CI-specific; FMN prosthetic group at NDUFV1 (N-module, upstream of PP-module where NDUFA3 B9 peripheral scaffold anchors)"},
                {"agent": "CoQ10 (ubiquinol)", "dose": "10–30 mg/kg/day", "note": "Electron acceptor at CI quinone site (downstream of PP-module ND3/ND4L peripheral scaffold)"},
                {"agent": "Thiamine (B1)", "dose": "100–300 mg/day", "note": "MANDATORY empiric: thiamine-responsive disorders (SLC19A3, BTD) mimic CI-Leigh"},
                {"agent": "Biotin", "dose": "5–10 mg/day", "note": "MANDATORY empiric: biotinidase deficiency (BTD) mimics CI-Leigh; cheap and safe"},
                {"agent": "Succinate", "dose": "Oral 500–1000 mg/day", "note": "CII bypass — bypasses NDUFA3-failed CI entirely; supplies ubiquinol via SDHA"},
                {"agent": "L-Carnitine", "dose": "50–100 mg/kg/day", "note": "Energy metabolism support; avoid free fatty acid accumulation"},
            ],
            "preferred_aed": "Levetiracetam (LEV) — renal excretion, no mitochondrial toxicity",
            "glucose_protocol": "IV dextrose GIR 6–8 mg/kg/min — NEVER fast (fasting precipitates metabolic crisis)",
            "anaesthesia": "Sevoflurane (NOT propofol — PRIS risk in CI absent)",
            "genetic_counselling": (
                "Autosomal recessive: both parents obligate carriers (usually asymptomatic); "
                "25% recurrence risk per pregnancy; consanguinity common in index cases. "
                "WES/panel sequencing: confirm biallelic NDUFA3 (19q13.42) pathogenic variants; "
                "distinguish from NDUFB9 (AQDQ, 8q24.13) by WES chromosomal locus. "
                "Offer prenatal diagnosis (CVS/amnio for NDUFA3 pathogenic variants); "
                "cascade carrier testing of siblings and first-degree relatives."
            ),
        },
        "ddx_key_negatives": [
            {
                "finding": "NO peripheral neuropathy",
                "ddx_excluded": "NDUFS1 (~50% peripheral neuropathy — CRITICAL distinguishing feature)",
            },
            {
                "finding": "NO olfactory bulb MRI lesions",
                "ddx_excluded": "NDUFS4 (olfactory bulb 52–65% — near-pathognomonic for NDUFS4)",
            },
            {
                "finding": "NO leukodystrophy / white matter T2 hyperintensity",
                "ddx_excluded": "NDUFV1 (~40–50% leukodystrophy — CRITICAL DDx)",
            },
            {
                "finding": "NO hypertrophic cardiomyopathy",
                "ddx_excluded": "NDUFV2 (~80% HCM) / SCO2 (~100% HCM) — CRITICAL DDx",
            },
            {
                "finding": "NO hepatopathy",
                "ddx_excluded": "POLG (~80% hepatopathy) / DGUOK (~90%) / SUCLG1 (~70%)",
            },
            {
                "finding": "NO MMA (methylmalonic acidemia)",
                "ddx_excluded": "SUCLA2 (mild MMA) / SUCLG1 (severe MMA) — acylcarnitine C4-DC absent in NDUFA3",
            },
            {
                "finding": "CIV (COX) activity NORMAL",
                "ddx_excluded": "SURF1/SCO2/COX10/COX15 — CIV deficiency distinguishes COX disorders",
            },
            {
                "finding": "WES locus 19q13.42 (NDUFA3 / NDUFA subfamily) NOT 8q24.13",
                "ddx_excluded": "NDUFB9 (AQDQ, 8q24.13, NDUFB subfamily) — both PP-module peripheral; distinguished by WES chromosomal position and CI face zone",
            },
            {
                "finding": "AUTOSOMAL RECESSIVE pedigree — both sexes affected",
                "ddx_excluded": "NDUFA1 (X-linked, Xq24) / NDUFB11 (X-linked, Xp11.3) — X-linked patterns with male predominance",
            },
        ],
        "patients_sample": [
            {
                "id": p["id"],
                "sex": p["sex"],
                "age_onset_years": p["age_onset_years"],
                "mutation": p["mutation"],
                "region": p["region"],
                "ci_activity_pct_control": p["ci_activity_pct_control"],
                "outcome": p["outcome"],
                "leigh_mri": p["features"]["leigh_mri_bilateral"],
                "lactic_acidosis": p["features"]["lactic_acidosis"],
            }
            for p in c[:15]
        ],
    }


def get_definitions() -> dict[str, Any]:
    return {
        "ndufa3_b9": (
            "NDUFA3 (B9 — bovine CI proteomics designation; Carroll 2006) is a peripheral alpha-helical "
            "structural subunit of Complex I (~84 aa, ~9.7 kDa) with NO canonical IMM-spanning TM helix. "
            "Located in the PP-module peripheral scaffold zone at the ND3/ND4L boundary (matrix face). "
            "Encoded at 19q13.42 (OMIM *603837). B9 designation originates from bovine CI proteomics "
            "and must NOT be confused with NDUFB9 (AQDQ, 8q24.13, NDUFB subfamily, PP-module ND2/ND3 face)."
        ),
        "pp_module_nd3_nd4l_peripheral_scaffold": (
            "PP-module (proximal pump module): ND1/ND2/ND3/ND6 subcomplex of the CI membrane arm. "
            "NDUFA3 (B9) occupies the PERIPHERAL scaffold zone at the ND3/ND4L boundary — the matrix-exposed "
            "face of the PP-module/PD-module transition, without a TM helix. This peripheral position "
            "stabilises the ND3 and ND4L lateral surface contacts from the matrix side. Loss of NDUFA3 "
            "leads to PP-module ND3/ND4L peripheral scaffold collapse → absent CI (BN-PAGE cleaner absent "
            "pattern, unlike N-module sub-assembly intermediates from NDUFS4/NDUFA12/NDUFA13 loss)."
        ),
        "b9_name_and_ndufb9_confusion": (
            "NDUFA3 is designated 'B9' in the bovine CI proteome (Carroll 2006) — this refers to its "
            "position within the NDUFA (A subfamily) subunit list, not the NDUFB (B subfamily). "
            "NDUFB9 (AQDQ subunit, 8q24.13) is a completely different gene, protein, chromosomal locus, "
            "and CI face zone (PP-module ND2/ND3 face, AQDQ beta-alpha fold, PP-module peripheral). "
            "The superficially similar names (NDUFA3 B9 vs NDUFB9) frequently cause diagnostic confusion; "
            "WES chromosomal locus (19q13.42 vs 8q24.13) is the definitive pivot. "
            "In clinical reporting, always specify both gene name AND chromosomal locus."
        ),
        "peripheral_vs_tm_ci_subunits": (
            "CI structural subunits are classified as: (1) TM-bearing (1 TM: NDUFA1/MWFE, NDUFB11/ESSS, "
            "NDUFB3, NDUFB9; 4 TM: NDUFA11/B14.7); (2) Peripheral/matrix-face (no TM: NDUFA3/B9, "
            "NDUFB6/B17, NDUFA2/B8, NDUFA9/I-gamma). NDUFA3 (B9) belongs to category 2 — peripheral "
            "alpha-helical scaffold, anchoring from the matrix face. This peripheral position means "
            "BN-PAGE shows 'cleaner' absent CI (no sub-complex intermediates) when NDUFA3 is lost, "
            "similar to TM-scaffold subunit loss but mechanistically distinct."
        ),
        "isolated_ci_deficiency": (
            "Biochemical fingerprint of NDUFA3 deficiency: Isolated CI activity 5–20% of controls; "
            "CII (SDHA/B), CIII (cytochrome bc1), CIV (COX), and CV activities NORMAL. This isolated "
            "pattern distinguishes CI nuclear subunit defects from mtDNA depletion syndromes (pan-complex "
            "reduction) and COX deficiency (CIV-specific). The absence of CII–IV involvement is the "
            "key biochemical fingerprint placing NDUFA3 deficiency in the nuclear CI deficiency class."
        ),
        "bn_page_absent_ci_peripheral": (
            "Blue-native PAGE (BN-PAGE) in NDUFA3 deficiency shows absent or severely reduced CI "
            "holocomplex (PP-module ND3/ND4L peripheral scaffold-loss pattern). Similar cleaner absent "
            "CI to NDUFA1 (Xq24), NDUFB11 (Xp11.3), NDUFB9 (8q24.13) but chromosomally 19q13.42. "
            "Distinct from N-module sub-assembly intermediates (NDUFA12/NDUFA13/NDUFS4) and from "
            "Q-module sub-assembly intermediates (NDUFA9/NDUFA10). NDUFA3 peripheral scaffold loss "
            "produces a pattern consistent with PP-module membrane arm disintegration without the "
            "partial sub-assembly bands seen in N-module or N-Q interface defects."
        ),
        "ndufa3_vs_ndufb9": (
            "NDUFA3 (B9, 19q13.42, 84 aa, peripheral, no TM) vs NDUFB9 (AQDQ, 8q24.13, 179 aa, "
            "PP-module ND2/ND3 face, peripheral AQDQ beta-alpha fold): Both PP-module peripheral "
            "subunits causing isolated CI-Leigh, but: "
            "(1) different chromosomal loci (19q13.42 vs 8q24.13) — WES definitive; "
            "(2) different CI face zones (ND3/ND4L peripheral vs ND2/ND3 PP-module); "
            "(3) different protein folds (alpha-helical scaffold vs AQDQ beta-alpha); "
            "(4) different protein sizes (84 aa vs 179 aa); "
            "(5) superficially similar names (B9 vs NDUFB9) are a naming trap — always "
            "distinguish by gene name (NDUFA3 vs NDUFB9) and chromosomal locus."
        ),
        "references": [
            "Carroll J et al. (2006) Mol Cell Proteomics — CI subunit proteomics; B9/NDUFA3 identification",
            "Guerrero-Castillo S et al. (2017) Cell Metabolism — CI assembly dynamics; PP-module peripheral subunit incorporation",
            "Stroud DA et al. (2016) Nature — CI assembly; membrane arm PP-module peripheral scaffold subunits",
            "Sazanov LA (2015) Nat Rev Mol Cell Biol — CI structure and module organisation; PP-module peripheral subunits",
            "Zhu J et al. (2016) Science — CryoEM 3.9 Å mammalian CI; PP-module peripheral positions including NDUFA3",
            "Fassone E & Rahman S (2012) J Med Genet — CI nuclear subunit genetics review; NDUFA3",
            "Haack TB et al. (2012) Nat Genet — nuclear CI subunit disease screening panel; NDUFA3",
            "Calvo SE et al. (2010) Nat Genet — nuclear CI subunit disease screening; 19q13.42 candidates",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:1500])
    print("\n=== BREAKDOWN (sample) ===")
    bd = get_breakdown()
    print(json.dumps({k: v for k, v in bd.items() if k != "patients_sample"}, indent=2)[:1500])
    print("\n=== DEFINITIONS (sample) ===")
    print(json.dumps(get_definitions(), indent=2)[:1000])
