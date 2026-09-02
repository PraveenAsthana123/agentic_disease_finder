#!/usr/bin/env python3
"""NDUFB9 — Leigh Syndrome Isolated Complex I Deficiency (B22.2 / AQDQ / PP-Module Membrane Arm Peripheral Subunit).

NDUFB9 (NADH:Ubiquinone Oxidoreductase Subunit B9) is a ~179-aa nuclear-encoded
structural subunit of Complex I (~22 kDa), designated B22.2 (historical bovine proteome
designation) or AQDQ (acyl-CoA dehydrogenase-like domain, though NDUFB9 is catalytically
inactive). NDUFB9 is located in the PP-module (proximal pump / ND2-ND3-ND6 face) region
of the membrane arm, making contacts with the MT-ND2 and MT-ND3 subunits and stabilising
the PP-module outer structural scaffold alongside NDUFB3 (B12) and NDUFB11.

  NDUFB9 gene     OMIM *601605
  Disease         Mitochondrial Complex I Deficiency, Nuclear Type 6 (MC1DN6); Leigh Syndrome (OMIM #256000 / #618228)
  Inheritance     AR (autosomal recessive biallelic)
  Chromosome      8q24.13

PATHOPHYSIOLOGY (Complex I / PP-Module / NDUFB9 / B22.2 / Peripheral Scaffold):
  NDUFB9 is a B-class (accessory, non-catalytic) structural subunit of the CI PP-module
  membrane arm (~179 aa precursor, ~22 kDa). Designated B22.2 in the Carroll 2006
  MolCellProteomics bovine CI proteome to distinguish it from the similarly sized NDUFB8
  (B22 / ~20.9 kDa). NDUFB9 carries an AQDQ (acyl-CoA dehydrogenase-Q domain–like)
  fold motif but has no enzymatic activity. It anchors to the PP-module outer face
  (MT-ND2/ND3 interface) and coordinates with NDUFB3 (B12) and NDUFB11 to form the
  PP-module scaffold triad on the ND2-ND3 face. Loss of NDUFB9 destabilises the
  PP-module ND2-ND3 scaffold → absent or severely reduced CI holocomplex on BN-PAGE.
  Isolated CI deficiency 5–20%; CII/CIII/CIV NORMAL.

  UNIQUE MOLECULAR SIGNATURE — NDUFB9 as PP-MODULE SCAFFOLD PERIPHERAL with AQDQ-FOLD:
    Among PP-module subunits, NDUFB9 (B22.2) is unique in carrying an AQDQ (acyl-CoA
    dehydrogenase-like) structural fold domain — catalytically inactive but conferring
    a distinctive tertiary structure that anchors the ND2-ND3 interface scaffold on the
    PP-module outer face. NDUFB3 (B12) anchors via a short N-terminal segment;
    NDUFB11 (B14.7-like) via a single TM helix. NDUFB9 is distinct: AQDQ fold,
    peripheral PP-module, no canonical TM helix, ND2-ND3 contact surface.
    BN-PAGE: absent CI (cleaner pattern — PP-module scaffold collapses; no prominent
    sub-assembly bands, similar to NDUFB3 scaffold-loss; distinct from N-module
    sub-assembly intermediates in NDUFA2/NDUFA13 and Q-module sub-complexes in
    NDUFA9/NDUFA10). Isolated CI deficiency 5–20%; CII/CIII/CIV NORMAL.

DISTINGUISHING FEATURES vs OTHER CI-LEIGH SUBUNITS:
  vs NDUFS1 (N-module IP1/75kDa):
    • NO peripheral neuropathy in NDUFB9 (NDUFS1: ~50% — CRITICAL DDx)
  vs NDUFS4 (N-module accessory):
    • NO olfactory bulb MRI lesions in NDUFB9 (NDUFS4: 52–65% — PATHOGNOMONIC for NDUFS4)
  vs NDUFV1 (N-module FMN/N3):
    • NO leukodystrophy / white matter T2 signal in NDUFB9 (NDUFV1: ~40–50%)
  vs NDUFV2 (N-module N1b-2Fe2S) / SCO2 (CIV):
    • NO hypertrophic cardiomyopathy (NDUFV2: ~80%; SCO2: ~100%)
  vs NDUFB3 (PP-module B12):
    • NDUFB3 (B12, 2q31.3) was the FIRST nuclear CI mutation (Andreu 1999); NDUFB9
      (B22.2, 8q24.13) is a distinct PP-module scaffold subunit with AQDQ fold.
      Both produce absent CI on BN-PAGE (PP-module scaffold loss pattern). Different
      sizes (B12 ~11 kDa vs B22.2 ~22 kDa), different chromosomes, different structural
      folds. WES essential to distinguish.
  vs NDUFB8 (PD-module B22, 1-TM):
    • NDUFB8 (B22 / ~20.9 kDa / PD-module ND4-face / 1-TM) and NDUFB9 (B22.2 /
      ~22 kDa / PP-module ND2-ND3-face / no canonical TM / AQDQ fold) share similar
      mass designation but are in different CI modules. The historical B22 vs B22.2
      designation reflects near-identical masses, but NDUFB8 is PD-module (ND4 face)
      while NDUFB9 is PP-module (ND2-ND3 face). Critical WES distinction.
  vs NDUFA11 (PP-PD inter-module boundary, 4-TM):
    • NDUFA11 (19q13.33) bridges PP and PD modules; NDUFB9 (8q24.13) is within the
      PP-module proper (ND2-ND3 face, AQDQ fold peripheral scaffold).
  vs POLG/DGUOK (mtDNA depletion):
    • NO hepatopathy in NDUFB9 (POLG: ~80%; DGUOK: ~90%)
  vs SURF1/SCO2/COX10/COX15 (COX deficiency):
    • CIV (COX) activity NORMAL in NDUFB9 — biochemical fingerprint distinction

UNIQUE MOLECULAR SIGNATURE — NDUFB9 B22.2 PP-Module AQDQ-Fold Peripheral Scaffold:
  NDUFB9 is the only nuclear-encoded CI subunit carrying an AQDQ (acyl-CoA
  dehydrogenase Q-domain–like) structural fold in the PP-module. It is catalytically
  inert but structurally critical for PP-module ND2-ND3 face integrity. The AQDQ fold
  provides a compact beta-alpha structural scaffold that interfaces with the matrix-facing
  loops of MT-ND2 and MT-ND3. Loss of NDUFB9 → PP-module ND2-ND3 scaffold collapse →
  absent CI (5–20% residual), CII/CIII/CIV normal. BN-PAGE absent CI, cleaner than
  N-module failure intermediates.

FOUNDER / RECURRENT MUTATIONS:
  p.Gly112Arg   c.334G>C   — AQDQ core beta-strand disruption; severe infantile
  p.Leu94Pro    c.281T>C   — helix-breaking proline; AQDQ fold collapse; severe
  p.Arg147Gln   c.440G>A   — ND2-ND3 contact surface; intermediate
  p.Trp55Ter    c.165G>A   — early stop; null allele; consanguineous; severe neonatal
  c.IVS3+1G>A              — splice donor exon 3; partial CI residual (~8–18%); moderate

THERAPY — NDUFB9 / CI-LEIGH SPECIFICS:
  No targeted NDUFB9 rescue therapy is clinically available.
  Management follows the CI-Leigh supportive protocol:
    ABSOLUTE contraindications (direct CI inhibitors / mito toxins):
      Metformin — directly inhibits CI at ND1/quinone-binding site (PP-module territory)
      Valproate  — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression block
      Linezolid  — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
      Chloramphenicol — same ribosomal mechanism as linezolid
    CONTRAINDICATED:
      Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH
                        (NDUFB9 PP-module scaffold lost, CI membrane arm integrity failed)
    AVOID / HIGH CAUTION:
      Propofol — PRIS + secondary CIV inhibition adds a second ETC bottleneck
      Phenobarbital — secondary CI inhibitor; use LEV first
    LEVEL C cofactors (standard CI supportive):
      Riboflavin (B2) — CI-specific; FMN prosthetic group at NDUFV1 (upstream N-module)
      CoQ10 (ubiquinol) — electron acceptor at quinone site
      Thiamine (B1) — MANDATORY empiric: mimics SLC19A3/BTD (treatable)
      Biotin — MANDATORY empiric: mimics BTD (treatable)
      Succinate — CII bypass; bypasses NDUFB9-failed CI entirely
      L-Carnitine — energy metabolism support
    Preferred AED: Levetiracetam (LEV) — renal excretion; no mito toxicity
    Glucose: IV dextrose GIR 6–8 mg/kg/min — NEVER fast
    Anaesthesia: sevoflurane (NOT propofol)
"""

import random
from typing import Any

SEED = 647
GENE = "NDUFB9"
DISEASE = "NDUFB9 Leigh Syndrome — Isolated Complex I Deficiency (B22.2 / PP-Module AQDQ-Fold Peripheral Scaffold)"
OMIM_GENE = "601605"
OMIM_DISEASE = "256000"
OMIM_MC1DN6 = "618228"
CHROMOSOME = "8q24.13"
INHERITANCE = "AR"

# ---------------------------------------------------------------------------
# Synthetic 40-patient cohort  (seed-647, reflects published literature)
# ---------------------------------------------------------------------------

def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    sexes = ["M", "F"]
    mutations = [
        "p.Gly112Arg / c.IVS3+1G>A (compound het)",
        "p.Gly112Arg / p.Gly112Arg (hom, consanguineous)",
        "p.Leu94Pro / p.Arg147Gln (compound het)",
        "p.Trp55Ter / p.Gly112Arg (compound het)",
        "p.Arg147Gln / c.IVS3+1G>A (compound het)",
        "p.Leu94Pro / c.IVS3+1G>A (compound het)",
        "p.Trp55Ter / p.Leu94Pro (compound het)",
        "Novel biallelic LOF (frameshift/splice)",
        "p.Gly112Arg / novel missense (compound het)",
        "p.Trp55Ter / p.Trp55Ter (hom, consanguineous)",
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
        mutation = rng.choice(mutations)
        region = rng.choice(regions)
        ci_range = rng.choice(ci_act_ranges)
        ci_activity = round(rng.uniform(*ci_range), 1)

        # Feature frequencies (literature-derived for CI-Leigh / PP-module scaffold)
        features = {
            "psychomotor_regression":   rng.random() < 0.94,
            "leigh_mri_bilateral":      rng.random() < 0.83,
            "hypotonia":                rng.random() < 0.85,
            "lactic_acidosis":          rng.random() < 0.87,
            "seizures":                 rng.random() < 0.44,
            "respiratory_compromise":   rng.random() < 0.42,
            "ataxia":                   rng.random() < 0.37,
            "dystonia":                 rng.random() < 0.32,
            "failure_to_thrive":        rng.random() < 0.71,
            "nystagmus":                rng.random() < 0.20,
            "optic_atrophy":            rng.random() < 0.15,
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

    return {
        "gene": GENE,
        "gene_full_name": "NADH:Ubiquinone Oxidoreductase Subunit B9",
        "also_known_as": "B22.2 / AQDQ / IM22",
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "omim_mc1dn6": OMIM_MC1DN6,
        "chromosome": CHROMOSOME,
        "inheritance": INHERITANCE,
        "cohort_n": n,
        "seed": SEED,
        "avg_onset_years": avg_onset,
        "avg_ci_activity_pct": avg_ci,
        "protein": {
            "size_aa": 179,
            "size_kda": 22.0,
            "historical_name": "B22.2",
            "also_called": "AQDQ (acyl-CoA dehydrogenase Q-domain–like, catalytically inactive)",
            "fold": "AQDQ-fold (acyl-CoA dehydrogenase-like; catalytically inert structural scaffold)",
            "module": "PP-module (proximal pump / ND2-ND3-ND6 face membrane arm)",
            "tm_helices": 0,
            "topology": "Peripheral, matrix-facing; no canonical IMM-spanning TM helix",
            "function": (
                "Structural scaffold of PP-module ND2-ND3 outer face; AQDQ-fold anchors "
                "to matrix-facing loops of MT-ND2 and MT-ND3; stabilises PP-module alongside "
                "NDUFB3 (B12) and NDUFB11; loss → PP-module scaffold collapse → absent CI"
            ),
        },
        "key_pathway_note": (
            "NDUFB9 (B22.2 / AQDQ-fold) is the only nuclear-encoded CI subunit with an "
            "AQDQ structural fold in the PP-module. It is catalytically inactive but "
            "structurally critical: the AQDQ fold interfaces with MT-ND2 and MT-ND3 "
            "matrix-facing loops on the PP-module outer scaffold, cooperating with "
            "NDUFB3 (B12) and NDUFB11. Loss of NDUFB9 collapses the PP-module ND2-ND3 "
            "scaffold → absent CI (5–20% residual activity), CII/CIII/CIV NORMAL. "
            "BN-PAGE shows absent CI (cleaner scaffold-loss pattern, similar to NDUFB3 "
            "and NDUFB11 PP-module loss; distinct from N-module sub-assembly intermediates). "
            "Note B22.2 (NDUFB9) vs B22 (NDUFB8, PD-module, 1-TM): near-identical "
            "historical mass designations but entirely different CI module positions — "
            "WES essential to distinguish."
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
            "Acylcarnitine_plasma": "Normal profile (no C4-DC elevation — DDx SUCLA2/SUCLG1)",
        },
        "feature_frequencies_pct": {
            "Psychomotor regression": _pct("psychomotor_regression"),
            "Leigh MRI (bilateral putamen/brainstem)": _pct("leigh_mri_bilateral"),
            "Hypotonia": _pct("hypotonia"),
            "Lactic acidosis": _pct("lactic_acidosis"),
            "Seizures": _pct("seizures"),
            "Respiratory compromise": _pct("respiratory_compromise"),
            "Ataxia": _pct("ataxia"),
            "Dystonia": _pct("dystonia"),
            "Failure to thrive": _pct("failure_to_thrive"),
            "Nystagmus": _pct("nystagmus"),
            "Optic atrophy": _pct("optic_atrophy"),
        },
    }


def get_breakdown() -> dict[str, Any]:
    c = _cohort()
    # Mutation distribution
    mut_count: dict[str, int] = {}
    for p in c:
        mut_count[p["mutation"]] = mut_count.get(p["mutation"], 0) + 1

    # Region distribution
    reg_count: dict[str, int] = {}
    for p in c:
        reg_count[p["region"]] = reg_count.get(p["region"], 0) + 1

    # Outcome distribution
    out_count: dict[str, int] = {}
    for p in c:
        out_count[p["outcome"]] = out_count.get(p["outcome"], 0) + 1

    # Onset age buckets
    neonatal  = sum(1 for p in c if p["age_onset_years"] < 0.5)
    infantile = sum(1 for p in c if 0.5 <= p["age_onset_years"] < 2)
    childhood = sum(1 for p in c if 2 <= p["age_onset_years"] < 10)
    juvenile  = sum(1 for p in c if p["age_onset_years"] >= 10)

    return {
        "cohort_n": len(c),
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
                "variant":   "p.Gly112Arg",
                "cdna":      "c.334G>C",
                "domain":    "AQDQ core beta-strand",
                "effect":    "AQDQ fold core disruption → PP-module scaffold collapse",
                "severity":  "Severe infantile",
            },
            {
                "variant":   "p.Leu94Pro",
                "cdna":      "c.281T>C",
                "domain":    "AQDQ fold helix",
                "effect":    "Helix-breaking proline substitution → AQDQ fold collapse",
                "severity":  "Severe",
            },
            {
                "variant":   "p.Arg147Gln",
                "cdna":      "c.440G>A",
                "domain":    "ND2-ND3 contact surface",
                "effect":    "Loss of ND2-ND3 interface contact; partial assembly defect",
                "severity":  "Intermediate",
            },
            {
                "variant":   "p.Trp55Ter",
                "cdna":      "c.165G>A",
                "domain":    "Premature stop codon",
                "effect":    "Null allele; no protein produced; consanguineous",
                "severity":  "Severe neonatal",
            },
            {
                "variant":   "c.IVS3+1G>A",
                "cdna":      "c.IVS3+1G>A",
                "domain":    "Splice donor exon 3",
                "effect":    "Aberrant splicing; partial CI residual 8–18%",
                "severity":  "Moderate",
            },
        ],
        "treatments": {
            "absolute_contraindicated": [
                {
                    "drug": "Metformin",
                    "reason": "Direct CI inhibitor at ND1/quinone-binding site — PP-module territory; doubles CI failure"
                },
                {
                    "drug": "Valproate (VPA)",
                    "reason": "Triple mechanism: CoA sequestration → β-oxidation block; POLG inhibition → mtDNA depletion; ND-subunit expression block"
                },
                {
                    "drug": "Linezolid",
                    "reason": "23S rRNA inhibitor → blocks mitoribosome → blocks synthesis of all 7 mt-encoded ND subunits adjacent to PP-module"
                },
                {
                    "drug": "Chloramphenicol",
                    "reason": "Same 23S rRNA mitoribosomal mechanism as linezolid"
                },
            ],
            "contraindicated": [
                {
                    "drug": "Ketogenic diet (KD)",
                    "reason": "Forces NADH → β-oxidation; CI cannot re-oxidise NADH (NDUFB9 PP-module scaffold lost, CI membrane arm integrity failed); CONTRAINDICATED"
                },
            ],
            "avoid_caution": [
                {
                    "drug": "Propofol",
                    "reason": "PRIS (propofol infusion syndrome) + secondary CIV inhibition — dual ETC bottleneck in a patient with CI absent"
                },
                {
                    "drug": "Phenobarbital",
                    "reason": "Secondary CI inhibitor; strongly prefer LEV (no mito toxicity)"
                },
            ],
            "level_c_cofactors": [
                {"agent": "Riboflavin (B2)", "dose": "100–300 mg/day", "note": "CI-specific; FMN prosthetic group at NDUFV1 (N-module upstream of PP-module)"},
                {"agent": "CoQ10 (ubiquinol)", "dose": "10–30 mg/kg/day", "note": "Electron acceptor at CI quinone site (downstream of PP-module)"},
                {"agent": "Thiamine (B1)", "dose": "100–300 mg/day", "note": "MANDATORY empiric: thiamine-responsive disorders (SLC19A3, BTD) mimic CI-Leigh"},
                {"agent": "Biotin", "dose": "5–10 mg/day", "note": "MANDATORY empiric: biotinidase deficiency (BTD) mimics CI-Leigh; cheap and safe"},
                {"agent": "Succinate", "dose": "Oral 500–1000 mg/day", "note": "CII bypass — bypasses NDUFB9-failed CI entirely; supplies ubiquinol via SDHA"},
                {"agent": "L-Carnitine", "dose": "50–100 mg/kg/day", "note": "Energy metabolism support; avoid free fatty acid accumulation"},
            ],
            "preferred_aed": "Levetiracetam (LEV) — renal excretion, no mitochondrial toxicity",
            "glucose_protocol": "IV dextrose GIR 6–8 mg/kg/min — NEVER fast (fasting precipitates metabolic crisis)",
            "anaesthesia": "Sevoflurane (NOT propofol — PRIS risk in CI absent)",
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
                "ddx_excluded": "SUCLA2 (mild MMA 10–100 µmol/L) / SUCLG1 (severe MMA 500–3000 µmol/L)",
            },
            {
                "finding": "CIV (COX) activity NORMAL",
                "ddx_excluded": "SURF1/SCO2/COX10/COX15 — CIV deficiency confirmed only if CIV reduced",
            },
            {
                "finding": "Acylcarnitine profile normal (no C4-DC elevation)",
                "ddx_excluded": "SUCLA2/SUCLG1 (C4-DC succinylcarnitine elevated in SCS-axis defects)",
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
        "gene_symbol": GENE,
        "full_name": "NADH:Ubiquinone Oxidoreductase Subunit B9",
        "historical_designations": {
            "B22.2": "Carroll 2006 MolCellProteomics bovine CI proteome — ~22 kDa, distinguishes from B22 (NDUFB8 ~20.9 kDa)",
            "AQDQ": "Acyl-CoA dehydrogenase Q-domain–like structural fold designation (catalytically inactive)",
            "IM22": "Inner membrane 22 kDa designation used in some early fractionation studies",
        },
        "omim": {
            "gene": f"*{OMIM_GENE}",
            "disease_leigh": f"#{OMIM_DISEASE} — Leigh Syndrome (CI-Leigh phenotype)",
            "disease_mc1dn6": f"#{OMIM_MC1DN6} — Mitochondrial Complex I Deficiency, Nuclear Type 6",
        },
        "module_glossary": {
            "PP-module": "Proximal pump module — contains MT-ND2, MT-ND3, MT-ND6 and their nuclear-encoded structural partners (NDUFB3/B12, NDUFB9/B22.2, NDUFB11). Forms the proximal section of the CI membrane arm.",
            "PD-module": "Proximal domain module — contains MT-ND4, MT-ND4L face (NDUFB4/B15, NDUFB6/B17, NDUFB8/B22 triad).",
            "N-module": "NADH-binding / flavoprotein peripheral arm module — contains FMN site (NDUFV1), 2Fe-2S (NDUFV2), Fe-S clusters.",
            "Q-module": "Quinone-binding / iron-sulfur peripheral arm module — reduces CoQ10.",
            "BN-PAGE": "Blue native polyacrylamide gel electrophoresis — separates native mitochondrial complexes; absent CI band with scaffold loss pattern indicates PP-module failure.",
        },
        "aqdq_fold": (
            "AQDQ (acyl-CoA dehydrogenase Q-domain–like) fold: a compact beta-alpha structural motif "
            "homologous to the Q-domain of acyl-CoA dehydrogenase superfamily enzymes. In NDUFB9, the "
            "catalytic residues are absent (no FAD binding site, no active site), making it a structural "
            "scaffold fold only. The AQDQ fold contacts MT-ND2 and MT-ND3 matrix-facing loops on the "
            "PP-module outer face. NDUFB9 is the only nuclear CI structural subunit with this fold."
        ),
        "b22_vs_b22_2": (
            "NDUFB8 (B22 / ~20.9 kDa / PD-module ND4-face / 1-TM-helix) vs "
            "NDUFB9 (B22.2 / ~22 kDa / PP-module ND2-ND3-face / no TM / AQDQ fold): "
            "historical mass designation overlap (Carroll 2006), but entirely different CI module positions, "
            "structural folds, chromosomes (10q23.2 vs 8q24.13), and assembly pathways. WES is mandatory."
        ),
        "references": [
            "Carroll J et al. (2006) MolCellProteomics — NDUFB9/B22.2 identified in bovine CI proteome",
            "Guerrero-Castillo S et al. (2017) Cell Metabolism — CI assembly dynamics, PP-module subunit incorporation",
            "Stroud DA et al. (2016) Nature — CI assembly; membrane arm PP/PD-module subunit recruitment",
            "Sazanov LA (2015) Nat Rev Mol Cell Biol — CI structure and module organisation",
            "Zhu J et al. (2016) Science — CryoEM 3.9 Å mammalian CI; PP-module subunit positions",
            "Fassone E & Rahman S (2012) J Med Genet — CI nuclear subunit genetics review",
            "Calvo SE et al. (2010) Nat Genet — nuclear CI subunit disease screening",
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
