#!/usr/bin/env python3
"""NDUFA1 — Leigh Syndrome Isolated Complex I Deficiency (MWFE / PP-Module ND3-Face, X-LINKED).

NDUFA1 (NADH:Ubiquinone Oxidoreductase Subunit A1) is a ~70-aa nuclear-encoded
structural subunit of Complex I (~8.0 kDa), designated MWFE (derived from the
N-terminal amino acid sequence Met-Trp-Phe-Glu) in the bovine CI proteome.
NDUFA1 is located in the PP-module (proximal pump / ND2/ND3 face) of the membrane
arm, with a single canonical IMM-spanning transmembrane helix anchoring to the ND3
lateral face of the PP-module. NDUFA1 is encoded on the X chromosome (Xq24 —
OMIM *300078), making it one of only two X-linked nuclear-encoded CI subunits
(alongside NDUFB11 at Xp11.3) — a defining distinguishing feature with critical
diagnostic and genetic counselling implications.

  NDUFA1 gene   OMIM *300078
  Disease       Mitochondrial Complex I Deficiency, Nuclear Type 2 (MC1DN2, OMIM #618224);
                Leigh Syndrome (OMIM #256000)
  Inheritance   X-LINKED (hemizygous males: severe/lethal; heterozygous females: variable)
  Chromosome    Xq24

PATHOPHYSIOLOGY (Complex I / PP-Module / NDUFA1 / MWFE / ND3-Face):
  NDUFA1 is the smallest nuclear-encoded CI subunit at ~70 aa (~8 kDa). Designated
  MWFE from its N-terminal tetrapeptide Met-Trp-Phe-Glu in the bovine CI proteome.
  It possesses a single canonical IMM-spanning TM helix anchoring to the ND3 lateral
  face of the PP-module membrane arm. NDUFA1 occupies the ND3 peripheral scaffold
  region — a structurally distinct zone from NDUFB11 (ESSS, ND6/ND1 boundary) and
  the ND4-face PD-module triad (NDUFB4/NDUFB6/NDUFB8). Loss of NDUFA1 destabilises
  the PP-module ND3 scaffold → absent or severely reduced CI holocomplex on BN-PAGE.
  Isolated CI deficiency 5–20%; CII/CIII/CIV NORMAL.

  UNIQUE MOLECULAR SIGNATURE — SMALLEST X-LINKED CI SUBUNIT / ND3 ANCHOR:
    NDUFA1 (Xq24) is among the SMALLEST nuclear-encoded CI subunits (70 aa, 8 kDa)
    and is X-linked. Only NDUFB11 (Xp11.3) shares X-linked inheritance among
    nuclear-encoded CI subunits in the NDUFB and NDUFA families.
    X-linked inheritance fundamentally changes the:
      (a) genetic counselling (carrier females, affected male offspring)
      (b) genotype–phenotype: hemizygous males severe (often lethal neonatal/infantile);
          heterozygous females variable (mild to asymptomatic, X-inactivation-dependent)
      (c) diagnostic strategy: X-linked pattern on pedigree + CI deficiency + Leigh =
          NDUFA1 (Xq24) alongside NDUFB11 (Xp11.3) before autosomal CI genes
    WES/panel reveals Xq24 location confirming X-linked inheritance.

  PP-MODULE MWFE UNIQUE STRUCTURAL ROLE:
    NDUFA1 (MWFE) anchors to the ND3 lateral face of the PP-module membrane arm.
    Despite its tiny size (70 aa), it is incorporated early in CI assembly (Guerrero-
    Castillo 2017) and stabilises the ND3 peripheral scaffold. Loss of NDUFA1 →
    PP-module ND3 peripheral destabilisation → absent CI (5–20% residual activity).
    The single TM helix of NDUFA1 tethers the N-terminal matrix loop to the ND3
    mitochondrial inner membrane face; without this tether, ND3 subcomplex assembly
    is compromised.

DISTINGUISHING FEATURES vs OTHER CI-LEIGH SUBUNITS:
  vs NDUFS1 (N-module IP1/75kDa):
    • NO peripheral neuropathy in NDUFA1 (NDUFS1: ~50% — CRITICAL DDx)
  vs NDUFS4 (N-module accessory):
    • NO olfactory bulb MRI lesions in NDUFA1 (NDUFS4: 52–65% — PATHOGNOMONIC for NDUFS4)
  vs NDUFV1 (N-module FMN/N3):
    • NO leukodystrophy / white matter T2 signal in NDUFA1 (NDUFV1: ~40–50%)
  vs NDUFV2 (N-module N1b-2Fe2S) / SCO2 (CIV):
    • NO hypertrophic cardiomyopathy (NDUFV2: ~80%; SCO2: ~100%)
  vs NDUFB11 (PP-module ESSS, Xp11.3, X-LINKED):
    • BOTH are X-linked; CRITICAL to distinguish by WES (Xq24 vs Xp11.3)
    • NDUFB11 anchors at ND6/ND1 boundary (PP-module); NDUFA1 anchors at ND3 lateral face (PP-module)
    • NDUFB11 is 136 aa; NDUFA1 is 70 aa — smallest X-linked CI subunit
    • Pedigree is identical (X-linked), WES locus (Xq24 vs Xp11.3) is the definitive pivot
  vs NDUFB3 (PP-module B12, 2q31.3, AR):
    • NDUFB3 is autosomal recessive; NDUFA1 is X-LINKED — same PP-module territory,
      different chromosomes (X vs 2q), different inheritance. Pedigree analysis key.
  vs POLG/DGUOK (mtDNA depletion):
    • NO hepatopathy in NDUFA1 (POLG: ~80%; DGUOK: ~90%)
  vs SURF1/SCO2/COX10/COX15 (COX deficiency):
    • CIV (COX) activity NORMAL in NDUFA1 — biochemical fingerprint distinction
  NDUFA1 vs NDUFB11 — SECOND X-LINKED NUCLEAR CI SUBUNIT PAIR:
    • Hemizygous male with severe neonatal/infantile CI-Leigh and X-linked pedigree →
      simultaneously consider NDUFA1 (Xq24) AND NDUFB11 (Xp11.3); both X-linked PP-module
    • WES localisation to X chromosome is definitive; Xq24 = NDUFA1, Xp11.3 = NDUFB11
    • Both produce isolated CI deficiency 5–20%; BN-PAGE absent CI
    • Carrier females in both: variable phenotype per X-inactivation pattern

FOUNDER / RECURRENT MUTATIONS:
  p.Gly32Arg   c.94G>A    — ND3-contact surface glycine; ND3-face PP-module disruption; severe infantile
  p.Arg38Cys   c.112C>T   — inter-helix loop ND3 interface; partial CI residual; intermediate
  p.Leu52Pro   c.155T>C   — TM helix proline insertion; helix-breaking; severe
  p.Trp10Ter   c.30G>A    — early stop; null allele; hemizygous males neonatal lethal
  c.IVS1+1G>A             — splice donor exon 1; partial CI residual (~10–20%); moderate

THERAPY — NDUFA1 / CI-LEIGH SPECIFICS:
  No targeted NDUFA1 rescue therapy is clinically available.
  Management follows the CI-Leigh supportive protocol:
    ABSOLUTE contraindications (direct CI inhibitors / mito toxins):
      Metformin — directly inhibits CI at ND1/quinone-binding site (PP-module territory)
      Valproate  — triple mechanism: CoA sequestration, POLG inhibition, ND-subunit expression block
      Linezolid  — inhibits 23S rRNA → blocks synthesis of all 7 mt-encoded ND subunits
      Chloramphenicol — same ribosomal mechanism as linezolid
    CONTRAINDICATED:
      Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH
                        (NDUFA1 PP-module ND3 scaffold lost, CI membrane arm integrity failed)
    AVOID / HIGH CAUTION:
      Propofol — PRIS + secondary CIV inhibition adds a second ETC bottleneck
      Phenobarbital — secondary CI inhibitor; use LEV first
    LEVEL C cofactors (standard CI supportive):
      Riboflavin (B2) — CI-specific; FMN prosthetic group at NDUFV1 (upstream N-module)
      CoQ10 (ubiquinol) — electron acceptor at quinone site (CI membrane arm)
      Thiamine (B1) — MANDATORY empiric: mimics SLC19A3/BTD (treatable)
      Biotin — MANDATORY empiric: mimics BTD (treatable)
      Succinate — CII bypass; bypasses NDUFA1-failed CI entirely; supplies ubiquinol via SDHA
      L-Carnitine — energy metabolism support
    Preferred AED: Levetiracetam (LEV) — renal excretion; no mito toxicity
    Glucose: IV dextrose GIR 6–8 mg/kg/min — NEVER fast
    Anaesthesia: sevoflurane (NOT propofol)
    GENETIC COUNSELLING: X-linked recessive — obligate carrier mothers; 50% of male
      offspring affected; 50% of female offspring carriers; offer prenatal testing
"""

import random
from typing import Any

SEED = 653
GENE = "NDUFA1"
DISEASE = "NDUFA1 Leigh Syndrome — Isolated Complex I Deficiency (MWFE / PP-Module ND3-Face, X-LINKED)"
OMIM_GENE = "300078"
OMIM_DISEASE = "618224"   # MC1DN2; also overlaps Leigh OMIM #256000
OMIM_DISEASE_LEIGH = "256000"
CHROMOSOME = "Xq24"
INHERITANCE = "X-LINKED (XL-recessive)"

# ---------------------------------------------------------------------------
# Synthetic 40-patient cohort  (seed-653, reflects published literature)
# X-linked: hemizygous males (severe/early), heterozygous females (variable)
# ---------------------------------------------------------------------------

def _build_cohort(n: int = 40) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    mutations = [
        "p.Gly32Arg / c.94G>A (hemizygous male; ND3-contact; severe infantile)",
        "p.Leu52Pro / c.155T>C (hemizygous male; TM-helix proline; severe)",
        "p.Trp10Ter / c.30G>A (hemizygous male; null; neonatal lethal)",
        "p.Arg38Cys / c.112C>T (hemizygous male; ND3-interface; intermediate)",
        "c.IVS1+1G>A (hemizygous male; splice-donor exon 1; partial CI residual)",
        "p.Gly32Arg / c.94G>A (heterozygous female; skewed X-inactivation; symptomatic)",
        "p.Arg38Cys / c.112C>T (heterozygous female; variable X-inactivation; mild)",
        "p.Leu52Pro / c.155T>C (heterozygous female; skewed X-inactivation; carrier-symptomatic)",
        "Novel hemizygous LOF (frameshift) — male",
        "Novel hemizygous missense ND3-face / splice — male",
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
            age_onset = round(rng.uniform(0.1, 4.0), 1)    # males: severe early
            mutation = rng.choice(mutations[:5] + mutations[8:])
        else:
            age_onset = round(rng.uniform(0.5, 18.0), 1)   # females: more variable
            mutation = rng.choice(mutations[5:8])
        region = rng.choice(regions)
        ci_range = rng.choice(ci_act_ranges)
        ci_activity = round(rng.uniform(*ci_range), 1)

        # Feature frequencies (literature-derived for CI-Leigh / PP-module X-linked)
        features = {
            "psychomotor_regression":   rng.random() < 0.91,
            "leigh_mri_bilateral":      rng.random() < 0.76,
            "hypotonia":                rng.random() < 0.83,
            "lactic_acidosis":          rng.random() < 0.85,
            "seizures":                 rng.random() < 0.42,
            "respiratory_compromise":   rng.random() < 0.39,
            "ataxia":                   rng.random() < 0.34,
            "dystonia":                 rng.random() < 0.29,
            "failure_to_thrive":        rng.random() < 0.73,
            "nystagmus":                rng.random() < 0.24,
            "optic_atrophy":            rng.random() < 0.16,
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
        "gene_full_name": "NADH:Ubiquinone Oxidoreductase Subunit A1",
        "also_known_as": "MWFE (Met-Trp-Phe-Glu N-terminal sequence; bovine CI proteome)",
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "omim_disease_leigh": OMIM_DISEASE_LEIGH,
        "chromosome": CHROMOSOME,
        "inheritance": INHERITANCE,
        "cohort_n": n,
        "cohort_males": males,
        "cohort_females": females,
        "seed": SEED,
        "avg_onset_years": avg_onset,
        "avg_ci_activity_pct": avg_ci,
        "protein": {
            "size_aa": 70,
            "size_kda": 8.0,
            "historical_name": "MWFE",
            "also_called": "Met-Trp-Phe-Glu (N-terminal tetrapeptide, bovine CI proteome designation)",
            "fold": "Single-TM helix anchor; N-terminal matrix loop; ND3 lateral face tether — smallest nuclear CI subunit",
            "module": "PP-module (proximal pump module / ND3 lateral face membrane arm)",
            "tm_helices": 1,
            "topology": "Single canonical IMM-spanning TM helix; N-terminal matrix loop; ND3 peripheral anchor",
            "function": (
                "Structural scaffold of PP-module ND3 lateral face; single TM helix anchors "
                "to ND3 subcomplex lateral face of PP-module; N-terminal matrix loop stabilises "
                "ND3 peripheral scaffold. Loss → PP-module ND3 peripheral destabilisation → "
                "absent CI. SMALLEST nuclear-encoded CI subunit. X-linked (Xq24)."
            ),
        },
        "key_pathway_note": (
            "NDUFA1 (MWFE) is among the SMALLEST nuclear-encoded Complex I subunits (70 aa, 8 kDa) "
            "and is X-linked (Xq24, OMIM *300078). Only NDUFB11 (Xp11.3) shares X-linked inheritance "
            "among the major nuclear-encoded CI subunit families. NDUFA1 anchors to the ND3 lateral "
            "face of the PP-module via a single TM helix — a distinct zone from NDUFB11 (ND6/ND1 "
            "boundary) within the same PP-module. X-linked CI-Leigh in a hemizygous male mandates "
            "simultaneous consideration of NDUFA1 (Xq24) and NDUFB11 (Xp11.3); WES locus is the "
            "definitive pivot. Biochemically: isolated CI 5–20%; CII/CIII/CIV NORMAL. BN-PAGE: absent CI."
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
                "variant":   "p.Gly32Arg",
                "cdna":      "c.94G>A",
                "domain":    "ND3-contact surface N-terminal loop",
                "effect":    "Loss of ND3 lateral face contact; PP-module ND3 scaffold disruption; severe infantile CI-Leigh",
                "severity":  "Severe infantile",
            },
            {
                "variant":   "p.Leu52Pro",
                "cdna":      "c.155T>C",
                "domain":    "Single TM helix",
                "effect":    "Proline insertion in TM helix; helix-breaking substitution; PP-module ND3 anchor lost",
                "severity":  "Severe",
            },
            {
                "variant":   "p.Trp10Ter",
                "cdna":      "c.30G>A",
                "domain":    "N-terminal matrix loop (MWFE W10)",
                "effect":    "Early stop; null allele; hemizygous males neonatal lethal; no functional protein produced",
                "severity":  "Severe neonatal / lethal",
            },
            {
                "variant":   "p.Arg38Cys",
                "cdna":      "c.112C>T",
                "domain":    "Inter-helix loop / ND3 interface",
                "effect":    "Loss of ND3 interface contact; partial CI residual (~12–18%); intermediate severity",
                "severity":  "Intermediate",
            },
            {
                "variant":   "c.IVS1+1G>A",
                "cdna":      "c.IVS1+1G>A",
                "domain":    "Splice donor exon 1",
                "effect":    "Aberrant splicing; partial CI residual 10–20%; moderate severity; reduced but not absent NDUFA1",
                "severity":  "Moderate",
            },
        ],
        "treatments": {
            "absolute_contraindicated": [
                {
                    "drug": "Metformin",
                    "reason": "Direct CI inhibitor at ND1/quinone-binding site — PP-module territory (ND1 is the core PP-module MT-encoded subunit NDUFA1 anchors near); doubles CI failure"
                },
                {
                    "drug": "Valproate (VPA)",
                    "reason": "Triple mechanism: CoA sequestration → β-oxidation block; POLG inhibition → mtDNA depletion; ND-subunit expression block"
                },
                {
                    "drug": "Linezolid",
                    "reason": "23S rRNA inhibitor → blocks mitoribosome → blocks synthesis of all 7 mt-encoded ND subunits (ND3 is PP-module core; NDUFA1 anchors at ND3 face)"
                },
                {
                    "drug": "Chloramphenicol",
                    "reason": "Same 23S rRNA mitoribosomal mechanism as linezolid; blocks ND3 synthesis in the same PP-module territory"
                },
            ],
            "contraindicated": [
                {
                    "drug": "Ketogenic diet (KD)",
                    "reason": "Forces NADH → β-oxidation; CI cannot re-oxidise NADH (NDUFA1 PP-module ND3 scaffold lost, CI membrane arm failed); CONTRAINDICATED"
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
                {"agent": "Riboflavin (B2)", "dose": "100–300 mg/day", "note": "CI-specific; FMN prosthetic group at NDUFV1 (N-module, upstream of PP-module where NDUFA1 anchors)"},
                {"agent": "CoQ10 (ubiquinol)", "dose": "10–30 mg/kg/day", "note": "Electron acceptor at CI quinone site (downstream of PP-module ND3 face)"},
                {"agent": "Thiamine (B1)", "dose": "100–300 mg/day", "note": "MANDATORY empiric: thiamine-responsive disorders (SLC19A3, BTD) mimic CI-Leigh"},
                {"agent": "Biotin", "dose": "5–10 mg/day", "note": "MANDATORY empiric: biotinidase deficiency (BTD) mimics CI-Leigh; cheap and safe"},
                {"agent": "Succinate", "dose": "Oral 500–1000 mg/day", "note": "CII bypass — bypasses NDUFA1-failed CI entirely; supplies ubiquinol via SDHA"},
                {"agent": "L-Carnitine", "dose": "50–100 mg/kg/day", "note": "Energy metabolism support; avoid free fatty acid accumulation"},
            ],
            "preferred_aed": "Levetiracetam (LEV) — renal excretion, no mitochondrial toxicity",
            "glucose_protocol": "IV dextrose GIR 6–8 mg/kg/min — NEVER fast (fasting precipitates metabolic crisis)",
            "anaesthesia": "Sevoflurane (NOT propofol — PRIS risk in CI absent)",
            "genetic_counselling": (
                "X-linked recessive: obligate carrier mothers (usually asymptomatic or mildly affected "
                "due to skewed X-inactivation); 50% of male offspring affected (hemizygous, severe); "
                "50% of female offspring are carriers; offer prenatal diagnosis (CVS/amnio for "
                "Xq24 NDUFA1 pathogenic variant); cascade testing of maternal relatives. "
                "Distinguish from NDUFB11 (Xp11.3) — both X-linked CI-Leigh; WES locus is key."
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
                "finding": "X-LINKED pedigree — WES locus Xq24 (not Xp11.3)",
                "ddx_excluded": "NDUFB11 (Xp11.3) — both X-linked PP-module; distinguished by WES chromosomal locus Xq24 vs Xp11.3",
            },
            {
                "finding": "X-LINKED pedigree (male predominance, carrier females)",
                "ddx_excluded": "ALL autosomal-recessive CI genes (NDUFB3/4/6/8/9/10, NDUFS1-8, NDUFV1/2 etc.) — AR pedigree excludes NDUFA1",
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
        "ndufa1_mwfe": (
            "NDUFA1 (MWFE — Met-Trp-Phe-Glu N-terminal tetrapeptide) is the smallest nuclear-encoded "
            "CI subunit at ~70 aa (~8 kDa). Located in the PP-module ND3 lateral face of the CI "
            "membrane arm. X-linked (Xq24, OMIM *300078). The MWFE designation comes from the bovine "
            "CI proteome where the N-terminal peptide sequence Met-Trp-Phe-Glu identified this tiny "
            "but essential structural anchor subunit."
        ),
        "pp_module_nd3_face": (
            "PP-module (proximal pump module): ND1/ND2/ND3/ND6 membrane subcomplex of the CI membrane "
            "arm. NDUFA1 (MWFE) anchors to the ND3 lateral face via its single TM helix — a distinct "
            "zone from NDUFB11 (ESSS, ND6/ND1 boundary). The PP-module is the most proximal (matrix-arm "
            "proximal) segment of the membrane arm and contains multiple nuclear scaffold subunits "
            "(NDUFB3, NDUFB9, NDUFB11, NDUFA1) that stabilise the mt-encoded ND1/ND2/ND3/ND6 core."
        ),
        "mwfe_name_origin": (
            "MWFE = Met-Trp-Phe-Glu. The name originated from the N-terminal amino acid sequence "
            "of the bovine CI subunit as identified in early proteomics studies. The gene encoding "
            "MWFE in humans is NDUFA1 (OMIM *300078, Xq24). The protein is 70 aa in human; the "
            "bovine homolog gave the traditional MWFE designation."
        ),
        "x_linked_ci_deficiency_ndufa1": (
            "NDUFA1 deficiency is the second characterised X-linked nuclear CI subunit deficiency "
            "alongside NDUFB11. Hemizygous males (one X copy, Xq24) lack functional NDUFA1 → "
            "severe early-onset CI-Leigh. Heterozygous females show X-inactivation-dependent "
            "phenotype. WES mandatory to confirm Xq24 locus; distinguish from NDUFB11 (Xp11.3) "
            "by chromosomal position. Both produce isolated CI deficiency 5–20%."
        ),
        "isolated_ci_deficiency": (
            "Biochemical fingerprint of NDUFA1 deficiency: Isolated CI activity 5–20% of controls; "
            "CII (SDHA/B), CIII (cytochrome bc1), CIV (COX), and CV activities NORMAL. This pattern "
            "distinguishes CI nuclear subunit defects from mtDNA depletion syndromes (pan-complex "
            "reduction) and COX deficiency (CIV-specific)."
        ),
        "bn_page_absent_ci": (
            "Blue-native PAGE (BN-PAGE) in NDUFA1 deficiency shows absent or severely reduced CI "
            "holocomplex (PP-module ND3 peripheral scaffold-loss pattern). Similar to NDUFB11 (PP-module "
            "ND6/ND1 boundary, Xp11.3) and NDUFB3 (PP-module B12, 2q31.3) but chromosomally Xq24. "
            "Distinct from N-module sub-assembly intermediates (NDUFA12/NDUFA13) and PD-module "
            "scaffold-loss (NDUFB4/NDUFB6/NDUFB8/NDUFB10)."
        ),
        "ndufa1_vs_ndufb11": (
            "NDUFA1 (MWFE, Xq24, 70 aa) vs NDUFB11 (ESSS, Xp11.3, 136 aa): Both X-linked PP-module "
            "CI structural subunits; both produce absent CI on BN-PAGE. Key differences: "
            "(1) chromosomal locus Xq24 vs Xp11.3 — WES is definitive; "
            "(2) ND3 face (NDUFA1) vs ND6/ND1 boundary (NDUFB11) within the same PP-module; "
            "(3) protein size 70 aa vs 136 aa (NDUFA1 is smallest nuclear CI subunit); "
            "(4) historical name MWFE vs ESSS. Pedigree is identical (X-linked in both). "
            "Simultaneous consideration of both is appropriate in hemizygous males with X-linked CI-Leigh."
        ),
        "mc1dn2": (
            "Mitochondrial Complex I Deficiency, Nuclear Type 2 (MC1DN2, OMIM #618224): the NDUFA1 "
            "disease entry in OMIM. Characterised by severe isolated CI deficiency (5–20% of controls), "
            "X-linked inheritance, Leigh syndrome MRI, lactic acidosis, and early-onset psychomotor "
            "regression. Overlaps phenotypically with other CI-Leigh subtypes but distinguished by "
            "X-linked pedigree and Xq24 WES locus."
        ),
        "references": [
            "Hoefs SJ et al. (2008) J Med Genet — NDUFA1 mutations in X-linked Leigh syndrome; clinical characterisation",
            "Berger I et al. (2008) — NDUFA1 X-linked CI deficiency; two families",
            "Guerrero-Castillo S et al. (2017) Cell Metabolism — CI assembly dynamics; PP-module subunit incorporation",
            "Stroud DA et al. (2016) Nature — CI assembly; membrane arm PP-module subunit recruitment",
            "Sazanov LA (2015) Nat Rev Mol Cell Biol — CI structure and module organisation; PP-module",
            "Zhu J et al. (2016) Science — CryoEM 3.9 Å mammalian CI; PP-module subunit positions including MWFE/NDUFA1",
            "Fassone E & Rahman S (2012) J Med Genet — CI nuclear subunit genetics review",
            "Calvo SE et al. (2010) Nat Genet — nuclear CI subunit disease screening; X-linked candidates",
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
