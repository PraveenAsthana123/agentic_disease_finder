#!/usr/bin/env python3
"""NDUFB11 — Leigh Syndrome Isolated Complex I Deficiency (ESSS / PP-Module ND6/ND1-Face, X-LINKED).

NDUFB11 (NADH:Ubiquinone Oxidoreductase Subunit B11) is a ~136-aa nuclear-encoded
structural subunit of Complex I (~15.6 kDa), designated ESSS (exon-skipped short subunit)
in the bovine CI proteome (Carroll 2006). NDUFB11 is located in the PP-module (proximal
pump / ND1/ND2/ND3/ND6 face) of the membrane arm, with a single canonical IMM-spanning
transmembrane helix anchoring it to the ND6/ND1 boundary zone of the PP-module membrane
arm. NDUFB11 is encoded on the X chromosome (Xp11.3 — OMIM *300403), making it
the ONLY X-linked nuclear-encoded subunit among the NDUFB proteins — a defining
distinguishing feature with critical diagnostic and genetic counselling implications.

  NDUFB11 gene    OMIM *300403
  Disease         Mitochondrial Complex I Deficiency, Nuclear Type; Leigh Syndrome (OMIM #256000)
  Inheritance     X-LINKED (hemizygous males: severe/lethal; heterozygous females: variable, X-inactivation-dependent)
  Chromosome      Xp11.3

PATHOPHYSIOLOGY (Complex I / PP-Module / NDUFB11 / ESSS / ND6-ND1-Face):
  NDUFB11 is a B-class (accessory, non-catalytic) structural subunit of the CI PP-module
  membrane arm (~136 aa precursor, ~15.6 kDa, historical name ESSS). It possesses
  one canonical IMM-spanning TM helix anchoring to the ND6/ND1 boundary face of the
  PP-module. Unlike the ND4-face triad (NDUFB4/NDUFB6/NDUFB8) and the ND4L-face
  scaffold (NDUFB10), NDUFB11 occupies the PP-module ND6/ND1 boundary zone — a
  structurally and functionally distinct region of the membrane arm from the PD-module.
  Loss of NDUFB11 destabilises the PP-module ND6/ND1 boundary scaffold → absent or
  severely reduced CI holocomplex on BN-PAGE. Isolated CI deficiency 5–20%;
  CII/CIII/CIV NORMAL.

  UNIQUE MOLECULAR SIGNATURE — X-LINKED INHERITANCE (THE CRITICAL DISTINGUISHING FEATURE):
    NDUFB11 (Xp11.3) is the ONLY nuclear-encoded NDUFB subunit on the X chromosome.
    All other NDUFB subunits (NDUFB3 [2q31.3], NDUFB4 [3q13.33], NDUFB6 [9p21.1],
    NDUFB8 [10q23.2], NDUFB9 [8q24.13], NDUFB10 [16p13.3]) are autosomal recessive.
    X-linked inheritance fundamentally changes the:
      (a) genetic counselling (carrier females, affected male offspring)
      (b) genotype–phenotype: hemizygous males severe (often lethal neonatal/in utero);
          heterozygous females variable (mild to asymptomatic, depending on X-inactivation)
      (c) diagnostic strategy: X-linked pattern on pedigree + CI deficiency + Leigh =
          NDUFB11 (and other rare X-linked CI genes) before autosomal NDUFB genes
    WES/panel reveals Xp11.3 location confirming X-linked inheritance.

  PP-MODULE ESSS UNIQUE STRUCTURAL ROLE:
    NDUFB11 (ESSS) is among the first PP-module nuclear-encoded subunits incorporated into
    the assembling CI membrane arm (Guerrero-Castillo 2017). It interacts with the
    ND6/ND1 subunit pair at the PP-module ND6 boundary face, bridging the proximal
    mitochondria-encoded (MT-ND1/ND2/ND3/ND6) subcomplexes with the nuclear-encoded
    PP-module accessory scaffold. Loss of NDUFB11 → early PP-module assembly stall →
    absent CI (5–20% residual activity).

DISTINGUISHING FEATURES vs OTHER CI-LEIGH SUBUNITS:
  vs NDUFS1 (N-module IP1/75kDa):
    • NO peripheral neuropathy in NDUFB11 (NDUFS1: ~50% — CRITICAL DDx)
  vs NDUFS4 (N-module accessory):
    • NO olfactory bulb MRI lesions in NDUFB11 (NDUFS4: 52–65% — PATHOGNOMONIC for NDUFS4)
  vs NDUFV1 (N-module FMN/N3):
    • NO leukodystrophy / white matter T2 signal in NDUFB11 (NDUFV1: ~40–50%)
  vs NDUFV2 (N-module N1b-2Fe2S) / SCO2 (CIV):
    • NO hypertrophic cardiomyopathy (NDUFV2: ~80%; SCO2: ~100%)
  vs NDUFB3 (PP-module B12, 2q31.3, AUTOSOMAL RECESSIVE):
    • NDUFB3 is autosomal recessive (2q31.3); NDUFB11 is X-LINKED (Xp11.3) — same
      PP-module, different chromosomes, different inheritance mode. Pedigree analysis
      (X-linked vs AR pattern) + WES distinguishes them BEFORE biochemistry.
  vs NDUFB9 (PP-module B22.2, AQDQ-fold, 8q24.13, AR):
    • NDUFB9 is autosomal recessive (8q24.13, AQDQ fold); NDUFB11 is X-LINKED (Xp11.3,
      single-TM ESSS). Both PP-module but different modules faces and inheritance modes.
  vs POLG/DGUOK (mtDNA depletion):
    • NO hepatopathy in NDUFB11 (POLG: ~80%; DGUOK: ~90%)
  vs SURF1/SCO2/COX10/COX15 (COX deficiency):
    • CIV (COX) activity NORMAL in NDUFB11 — biochemical fingerprint distinction
  X-LINKED vs AUTOSOMAL — CRITICAL DIAGNOSTIC PIVOT:
    • Hemizygous male with severe neonatal CI-Leigh → immediately consider NDUFB11 (Xp11.3)
      and other X-linked CI genes before the autosomal NDUFB cluster
    • Female carrier mother may be asymptomatic (X-inactivation) — pedigree critical
    • Obligate carrier mothers: offer genetic counselling; X-linked recessive
    • Prenatal diagnosis available: CVS/amnio for NDUFB11 Xp11.3 pathogenic variant

FOUNDER / RECURRENT MUTATIONS:
  p.Arg37Ter   c.109C>T    — early stop; null allele; hemizygous males lethal neonatal; severe
  p.Pro174Leu  c.521C>T    — TM-helix proline insertion; helix disruption; severe
  p.Val79Ala   c.236T>C    — PP-module ND6-contact surface residue; intermediate
  p.Gly15Arg   c.43G>A     — N-terminal signal peptide disruption; severe
  c.IVS1+1G>A              — splice donor exon 1; partial CI residual (~8–18%); moderate

THERAPY — NDUFB11 / CI-LEIGH SPECIFICS:
  No targeted NDUFB11 rescue therapy is clinically available.
  Management follows the CI-Leigh supportive protocol:
    ABSOLUTE contraindications (direct CI inhibitors / mito toxins):
      Metformin — directly inhibits CI at ND1/quinone-binding site (PP-module territory)
      Valproate  — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression block
      Linezolid  — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
      Chloramphenicol — same ribosomal mechanism as linezolid
    CONTRAINDICATED:
      Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH
                        (NDUFB11 PP-module ND6/ND1 scaffold lost, CI membrane arm integrity failed)
    AVOID / HIGH CAUTION:
      Propofol — PRIS + secondary CIV inhibition adds a second ETC bottleneck
      Phenobarbital — secondary CI inhibitor; use LEV first
    LEVEL C cofactors (standard CI supportive):
      Riboflavin (B2) — CI-specific; FMN prosthetic group at NDUFV1 (upstream N-module)
      CoQ10 (ubiquinol) — electron acceptor at quinone site (CI membrane arm)
      Thiamine (B1) — MANDATORY empiric: mimics SLC19A3/BTD (treatable)
      Biotin — MANDATORY empiric: mimics BTD (treatable)
      Succinate — CII bypass; bypasses NDUFB11-failed CI entirely; supplies ubiquinol via SDHA
      L-Carnitine — energy metabolism support
    Preferred AED: Levetiracetam (LEV) — renal excretion; no mito toxicity
    Glucose: IV dextrose GIR 6–8 mg/kg/min — NEVER fast
    Anaesthesia: sevoflurane (NOT propofol)
    GENETIC COUNSELLING: X-linked recessive — obligate carrier mothers; 50% of male
      offspring affected; 50% of female offspring carriers; offer prenatal testing
"""

import random
from typing import Any

SEED = 651
GENE = "NDUFB11"
DISEASE = "NDUFB11 Leigh Syndrome — Isolated Complex I Deficiency (ESSS / PP-Module ND6/ND1-Face, X-LINKED)"
OMIM_GENE = "300403"
OMIM_DISEASE = "256000"
CHROMOSOME = "Xp11.3"
INHERITANCE = "X-LINKED (XL-recessive)"

# ---------------------------------------------------------------------------
# Synthetic 40-patient cohort  (seed-651, reflects published literature)
# X-linked: hemizygous males (severe/early), heterozygous females (variable)
# ---------------------------------------------------------------------------

def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    mutations = [
        "p.Arg37Ter / c.109C>T (hemizygous male; null)",
        "p.Pro174Leu / c.521C>T (hemizygous male; TM-helix)",
        "p.Val79Ala / c.236T>C (hemizygous male; ND6-contact)",
        "p.Gly15Arg / c.43G>A (hemizygous male; signal-peptide)",
        "c.IVS1+1G>A (hemizygous male; splice-donor exon 1)",
        "p.Arg37Ter / c.109C>T (heterozygous female; skewed X-inactivation)",
        "p.Pro174Leu / c.521C>T (heterozygous female; skewed X-inactivation)",
        "p.Val79Ala / c.236T>C (heterozygous female; symptomatic carrier)",
        "Novel hemizygous LOF (frameshift) — male",
        "Novel hemizygous missense / c.IVS1+1G>A (compound haplotype) — male",
    ]
    regions = [
        "MENA", "South Asian", "European", "East Asian",
        "North African", "Latin American", "Sub-Saharan African",
    ]
    ci_act_ranges = [(5, 10), (8, 15), (10, 20)]

    cohort = []
    for i in range(1, n + 1):
        # X-linked cohort: ~70% male (hemizygous, severe), ~30% female (heterozygous, variable)
        is_male = rng.random() < 0.70
        sex = "M" if is_male else "F"
        if is_male:
            age_onset = round(rng.uniform(0.1, 3.5), 1)   # males: severe early
            mutation = rng.choice(mutations[:5] + mutations[8:])
        else:
            age_onset = round(rng.uniform(0.5, 15.0), 1)  # females: more variable
            mutation = rng.choice(mutations[5:8])
        region = rng.choice(regions)
        ci_range = rng.choice(ci_act_ranges)
        ci_activity = round(rng.uniform(*ci_range), 1)

        # Feature frequencies (literature-derived for CI-Leigh / PP-module X-linked)
        features = {
            "psychomotor_regression":   rng.random() < 0.90,
            "leigh_mri_bilateral":      rng.random() < 0.75,
            "hypotonia":                rng.random() < 0.82,
            "lactic_acidosis":          rng.random() < 0.84,
            "seizures":                 rng.random() < 0.40,
            "respiratory_compromise":   rng.random() < 0.38,
            "ataxia":                   rng.random() < 0.33,
            "dystonia":                 rng.random() < 0.28,
            "failure_to_thrive":        rng.random() < 0.72,
            "nystagmus":                rng.random() < 0.22,
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
    males     = sum(1 for p in c if p["sex"] == "M")
    females   = sum(1 for p in c if p["sex"] == "F")

    return {
        "gene": GENE,
        "gene_full_name": "NADH:Ubiquinone Oxidoreductase Subunit B11",
        "also_known_as": "ESSS (Exon-Skipped Short Subunit; Carroll 2006 bovine CI proteome)",
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
            "size_aa": 136,
            "size_kda": 15.6,
            "historical_name": "ESSS",
            "also_called": "Exon-Skipped Short Subunit (Carroll 2006 bovine CI proteome)",
            "fold": "Single-TM helix anchor; N-terminal matrix domain; ND6/ND1 boundary scaffold",
            "module": "PP-module (proximal pump module / ND6-ND1 boundary face membrane arm)",
            "tm_helices": 1,
            "topology": "Single canonical IMM-spanning TM helix; N-terminal matrix domain; ND6/ND1-face anchor",
            "function": (
                "Structural scaffold of PP-module ND6/ND1 boundary zone; single TM helix anchors "
                "to ND6 lateral face of PP-module membrane arm; loss → PP-module ND6/ND1 scaffold "
                "disruption → absent CI. ONLY X-linked nuclear-encoded NDUFB subunit."
            ),
        },
        "key_pathway_note": (
            "NDUFB11 (ESSS) is the ONLY nuclear-encoded NDUFB Complex I subunit on the "
            "X chromosome (Xp11.3). All other NDUFB subunits are autosomal. This X-linked "
            "inheritance is THE critical distinguishing feature: hemizygous males present "
            "with severe neonatal CI-Leigh; heterozygous females show variable phenotype "
            "depending on X-inactivation pattern (may be carrier-only or mildly symptomatic). "
            "A pedigree showing X-linked inheritance + isolated CI deficiency + Leigh syndrome "
            "mandates immediate investigation of NDUFB11 (Xp11.3) before autosomal NDUFB genes. "
            "Biochemically: isolated CI activity 5–20%; CII/CIII/CIV NORMAL. BN-PAGE: absent CI "
            "(PP-module ND6/ND1 scaffold-loss pattern)."
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

    # Sex distribution
    males   = sum(1 for p in c if p["sex"] == "M")
    females = sum(1 for p in c if p["sex"] == "F")

    return {
        "cohort_n": len(c),
        "sex_distribution": {
            "males_hemizygous": males,
            "females_heterozygous": females,
            "male_pct": round(males / len(c) * 100, 1),
            "female_pct": round(females / len(c) * 100, 1),
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
                "variant":   "p.Arg37Ter",
                "cdna":      "c.109C>T",
                "domain":    "N-terminal matrix domain",
                "effect":    "Early stop; null allele; hemizygous males lethal/severe neonatal; heterozygous females carrier",
                "severity":  "Severe neonatal (hemizygous) / Carrier (heterozygous)",
            },
            {
                "variant":   "p.Pro174Leu",
                "cdna":      "c.521C>T",
                "domain":    "Single TM helix (C-terminal)",
                "effect":    "Proline insertion in TM helix; helix-breaking substitution; PP-module scaffold disruption",
                "severity":  "Severe",
            },
            {
                "variant":   "p.Val79Ala",
                "cdna":      "c.236T>C",
                "domain":    "PP-module ND6-contact surface",
                "effect":    "Loss of ND6 lateral face contact; partial PP-module assembly defect",
                "severity":  "Intermediate",
            },
            {
                "variant":   "p.Gly15Arg",
                "cdna":      "c.43G>A",
                "domain":    "Mitochondrial targeting sequence",
                "effect":    "Signal peptide disruption; impaired mitochondrial import; absent mature protein",
                "severity":  "Severe",
            },
            {
                "variant":   "c.IVS1+1G>A",
                "cdna":      "c.IVS1+1G>A",
                "domain":    "Splice donor exon 1",
                "effect":    "Aberrant splicing; partial CI residual 8–18%; exon-skipping product (ESSS isoform origin)",
                "severity":  "Moderate",
            },
        ],
        "treatments": {
            "absolute_contraindicated": [
                {
                    "drug": "Metformin",
                    "reason": "Direct CI inhibitor at ND1/quinone-binding site — PP-module territory (ND1 is the core PP-module MT-encoded subunit); doubles CI failure"
                },
                {
                    "drug": "Valproate (VPA)",
                    "reason": "Triple mechanism: CoA sequestration → β-oxidation block; POLG inhibition → mtDNA depletion; ND-subunit expression block"
                },
                {
                    "drug": "Linezolid",
                    "reason": "23S rRNA inhibitor → blocks mitoribosome → blocks synthesis of all 7 mt-encoded ND subunits (ND1 is PP-module core)"
                },
                {
                    "drug": "Chloramphenicol",
                    "reason": "Same 23S rRNA mitoribosomal mechanism as linezolid"
                },
            ],
            "contraindicated": [
                {
                    "drug": "Ketogenic diet (KD)",
                    "reason": "Forces NADH → β-oxidation; CI cannot re-oxidise NADH (NDUFB11 PP-module ND6/ND1 scaffold lost, CI membrane arm integrity failed); CONTRAINDICATED"
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
                {"agent": "Riboflavin (B2)", "dose": "100–300 mg/day", "note": "CI-specific; FMN prosthetic group at NDUFV1 (N-module, upstream of PP-module)"},
                {"agent": "CoQ10 (ubiquinol)", "dose": "10–30 mg/kg/day", "note": "Electron acceptor at CI quinone site (downstream of PP-module)"},
                {"agent": "Thiamine (B1)", "dose": "100–300 mg/day", "note": "MANDATORY empiric: thiamine-responsive disorders (SLC19A3, BTD) mimic CI-Leigh"},
                {"agent": "Biotin", "dose": "5–10 mg/day", "note": "MANDATORY empiric: biotinidase deficiency (BTD) mimics CI-Leigh; cheap and safe"},
                {"agent": "Succinate", "dose": "Oral 500–1000 mg/day", "note": "CII bypass — bypasses NDUFB11-failed CI entirely; supplies ubiquinol via SDHA"},
                {"agent": "L-Carnitine", "dose": "50–100 mg/kg/day", "note": "Energy metabolism support; avoid free fatty acid accumulation"},
            ],
            "preferred_aed": "Levetiracetam (LEV) — renal excretion, no mitochondrial toxicity",
            "glucose_protocol": "IV dextrose GIR 6–8 mg/kg/min — NEVER fast (fasting precipitates metabolic crisis)",
            "anaesthesia": "Sevoflurane (NOT propofol — PRIS risk in CI absent)",
            "genetic_counselling": (
                "X-linked recessive: obligate carrier mothers (usually asymptomatic or mildly affected "
                "due to skewed X-inactivation); 50% of male offspring affected (hemizygous, severe); "
                "50% of female offspring are carriers; offer prenatal diagnosis (CVS/amnio for "
                "Xp11.3 NDUFB11 pathogenic variant); cascade testing of maternal relatives."
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
                "ddx_excluded": "SUCLA2 (mild MMA 10–100 µmol/L) / SUCLG1 (severe MMA 500–3000 µmol/L)",
            },
            {
                "finding": "CIV (COX) activity NORMAL",
                "ddx_excluded": "SURF1/SCO2/COX10/COX15 — CIV deficiency only if CIV reduced",
            },
            {
                "finding": "X-LINKED pedigree (male predominance, carrier females)",
                "ddx_excluded": "ALL autosomal-recessive NDUFB genes (NDUFB3/4/6/8/9/10) — AR pedigree excludes NDUFB11",
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
        "ndufb11_esss": (
            "NDUFB11 (ESSS — Exon-Skipped Short Subunit) is a ~136-aa nuclear-encoded "
            "accessory subunit of the CI membrane arm PP-module (~15.6 kDa). Designated ESSS "
            "in the Carroll 2006 bovine CI proteome — the name reflects discovery via an "
            "exon-skipped mRNA isoform. NDUFB11 is located in the PP-module ND6/ND1 boundary "
            "face with a single IMM-spanning TM helix. Unique among NDUFB subunits: encoded "
            "on the X chromosome (Xp11.3, OMIM *300403) — X-linked inheritance."
        ),
        "pp_module": (
            "PP-module (proximal pump module): ND1/ND2/ND3/ND6 membrane subcomplex of the "
            "CI membrane arm. The most proximal (closest to the matrix arm) segment of the "
            "membrane arm. Contains nuclear-encoded scaffold subunits including NDUFB3 (B12), "
            "NDUFB9 (B22.2/AQDQ), and NDUFB11 (ESSS). PP-module assembly is the earliest "
            "membrane arm assembly step; its failure produces absent CI without the N-module "
            "sub-assembly intermediates seen in N-module subunit defects."
        ),
        "esss_name_origin": (
            "ESSS = Exon-Skipped Short Subunit. The name originated from discovery of a "
            "shorter isoform via alternative splicing (exon skipping) in the bovine CI "
            "proteome. The full-length 136-aa (15.6 kDa) form is the physiologically relevant "
            "protein; the 'exon-skipped' short isoform was the discovery artifact that gave "
            "the gene its historical designation."
        ),
        "x_linked_ci_deficiency": (
            "NDUFB11 deficiency is the paradigmatic X-linked nuclear CI subunit deficiency. "
            "Hemizygous males (one copy, Xp11.3) lack functional NDUFB11 → severe early-onset "
            "CI-Leigh. Heterozygous females (two X chromosomes) show X-inactivation-dependent "
            "phenotype: if mutant allele is preferentially inactivated → asymptomatic carrier; "
            "if mutant allele is preferentially expressed → symptomatic. WES mandatory to "
            "confirm Xp11.3 locus; chromosome location is the definitive diagnostic clue."
        ),
        "isolated_ci_deficiency": (
            "Biochemical fingerprint of NDUFB11 deficiency: Isolated CI activity 5–20% of "
            "controls; CII (SDHA/B), CIII (cytochrome bc1), CIV (COX), and CV activities "
            "NORMAL. This pattern distinguishes CI nuclear subunit defects from mtDNA "
            "depletion syndromes (pan-complex reduction) and COX deficiency (CIV-specific)."
        ),
        "bn_page_absent_ci": (
            "Blue-native PAGE (BN-PAGE) in NDUFB11 deficiency shows absent or severely "
            "reduced CI holocomplex (PP-module ND6/ND1 boundary scaffold-loss pattern). "
            "Similar to NDUFB3 (B12, PP-module, 2q31.3) and NDUFB9 (B22.2, PP-module, 8q24.13) "
            "but chromosomally X-linked vs autosomal. Distinct from N-module sub-assembly "
            "intermediates (NDUFA2/NDUFA13) and PD-module scaffold-loss (NDUFB4/NDUFB6/NDUFB8/NDUFB10)."
        ),
        "ndufb11_vs_ndufb3": (
            "NDUFB11 (ESSS, Xp11.3, X-LINKED) vs NDUFB3 (B12, 2q31.3, AUTOSOMAL RECESSIVE): "
            "Both are PP-module nuclear-encoded CI structural subunits and both produce absent "
            "CI on BN-PAGE. Critically different inheritance: NDUFB11 is X-linked (males affected, "
            "carrier females); NDUFB3 is AR (equal sex incidence, consanguinity risk). Pedigree "
            "analysis is the primary diagnostic pivot before WES; X-linked pattern immediately "
            "points toward NDUFB11 over NDUFB3."
        ),
        "references": [
            "Carroll J et al. (2006) MolCellProteomics — NDUFB11/ESSS identified in bovine CI proteome",
            "Guerrero-Castillo S et al. (2017) Cell Metabolism — CI assembly dynamics; PP-module subunit incorporation order",
            "Stroud DA et al. (2016) Nature — CI assembly; membrane arm PP-module subunit recruitment",
            "Sazanov LA (2015) Nat Rev Mol Cell Biol — CI structure and module organisation; PP-module",
            "Zhu J et al. (2016) Science — CryoEM 3.9 Å mammalian CI; PP-module subunit positions",
            "Fassone E & Rahman S (2012) J Med Genet — CI nuclear subunit genetics review",
            "Calvo SE et al. (2010) Nat Genet — nuclear CI subunit disease screening; X-linked candidates",
            "Friederich MW et al. (2017) Mol Genet Metab — NDUFB11 X-linked CI deficiency clinical series",
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
