#!/usr/bin/env python3
"""NDUFAF5 — Mitochondrial Complex I Deficiency (ND1-Module CI Assembly Factor 5 / C20orf7).

NDUFAF5 (C20orf7, Chromosome 20 Open Reading Frame 7) is a CI assembly factor
that acts within the ND1-containing proton-pump (P-module) sub-assembly — the
same Class 3 assembly intermediate as NDUFAF3 and NDUFAF4.  It is mechanistically
distinct from the MCIA tetramer (ACAD9–NDUFAF1–ECSIT–TMEM126B, ND2/ND5 module)
and from the N-Q module swap (NDUFAF2-NDUFA12 class).

  NDUFAF5 gene  OMIM *612360
  Disease       Mitochondrial Complex I Deficiency (OMIM #256000; also reported as Leigh syndrome)
  Inheritance   AR (autosomal recessive, biallelic)
  Chromosome    20p12.1

Reference: Sugiana C et al. (2008) Mutation of C20orf7 disrupts complex I assembly
and causes lethal neonatal mitochondrial disease. Am J Hum Genet 83(4):468–78.

PATHOPHYSIOLOGY (NDUFAF5 / ND1 P-Module / Class-3 Assembly):
  CI assembly proceeds through modular intermediates. NDUFAF5 acts on the
  ND1-containing sub-assembly (P-module territory):
    1. NDUFAF5 associates with the ND1-containing early P-module sub-assembly.
    2. It facilitates P-module maturation (ND3, ND4L, ND6 incorporation).
    3. The completed P-module merges with the Q-module (NDUFS3-containing) and
       N-module (NDUFV1/NDUFV2-containing) to form mature CI holoenzyme.
  Loss of NDUFAF5:
    • Early ND1 sub-assembly intermediate stalls (Class 3 BN-PAGE band).
    • Isolated CI deficiency (5–20% of control); CII/CIII/CIV normal.
    • BN-PAGE: same Class 3 intermediate as NDUFAF3/NDUFAF4 deficiency —
      but a distinct gene on chromosome 20p12.1.

NDUFAF5 UNIQUE FEATURES vs OTHER ND1-MODULE FACTORS:
  1. THIRD ND1-MODULE CI ASSEMBLY FACTOR — DISTINCT CHROMOSOME 20p12.1:
     NDUFAF3 (2q33.1), NDUFAF4 (6q16.3), and NDUFAF5 (20p12.1) all belong to
     Class 3 (ND1-module) but are on different chromosomes. Near-identical
     BN-PAGE profiles and isolated CI phenotypes make WES mandatory.
  2. NO FAD-BINDING / NO RIBOFLAVIN RESPONSE:
     NDUFAF5 has no FAD-binding domain. Riboflavin supplementation does NOT
     rescue NDUFAF5 deficiency. Critical DDx: ACAD9 (50-60% riboflavin
     responsive, Level B) vs NDUFAF5 (0% — no FAD domain).
  3. LETHAL NEONATAL PRESENTATION IN NULL ALLELES:
     Sugiana et al. 2008 described fatal neonatal onset in patients with
     null NDUFAF5 alleles — consistent with the severe phenotype of complete
     ND1-module assembly failure.
  4. CLASS 3 BN-PAGE — INDISTINGUISHABLE FROM NDUFAF3/NDUFAF4:
     NDUFAF3, NDUFAF4, and NDUFAF5 all produce stalled early ND1 sub-assembly
     intermediates on BN-PAGE. WES chromosomal locus is the only reliable
     method to assign the diagnosis.
  5. NO NDUFAF3/NDUFAF4-STYLE OBLIGATE HETERODIMER:
     Unlike the NDUFAF3-NDUFAF4 obligate heterodimer, NDUFAF5 acts independently
     on the ND1 sub-assembly. Loss of NDUFAF5 does NOT secondarily destabilize
     NDUFAF3 or NDUFAF4 protein — protein blot can be informative.

DISTINGUISHING FEATURES vs OTHER CI GENES:
  vs NDUFAF3 (2q33.1): Same Class 3 ND1-module. Near-identical phenotype.
    WES MANDATORY. 2q33.1 vs 20p12.1 locus is the only distinguisher.
    NDUFAF3 involves obligate NDUFAF4 heterodimer; NDUFAF5 acts independently.
  vs NDUFAF4 (6q16.3): Same Class 3 ND1-module. Near-identical phenotype.
    WES MANDATORY. 6q16.3 vs 20p12.1 locus distinguishes.
    NDUFAF4 causes secondary destabilization of NDUFAF3; NDUFAF5 does not.
  vs ACAD9 (3q21.3): MCIA complex (ND2/ND5 module). ACAD9 riboflavin-
    responsive (50-60%, Level B). NDUFAF5: NO riboflavin response.
    BN-PAGE ND2/ND5 class (ACAD9) vs ND1-module class (NDUFAF5).
  vs TIMMDC1 (3q25.1): Also Class 3 ND1-module. TIMMDC1 HCM >80%;
    NDUFAF5 HCM low (<20%). TIMMDC1 is an integral IMM protein (TM helices);
    NDUFAF5 is soluble matrix.
  vs NDUFS1 (2q33.3): NDUFS1 peripheral neuropathy 50%; NDUFAF5 NONE.
  vs NDUFV1 (11q13.2): NDUFV1 leukodystrophy 40-50%; NDUFAF5 NONE.
  vs POLG/DGUOK: Both cause hepatopathy; NDUFAF5 NO hepatopathy.
"""

import random
import math

SEED = 687
rng  = random.Random(SEED)

GENE         = "NDUFAF5"
OMIM_GENE    = "612360"
OMIM_DISEASE = "256000"
DISEASE_NAME = "Mitochondrial Complex I Deficiency (CI-Leigh; Isolated CI Deficiency)"
CHROMOSOME   = "20p12.1"
INHERITANCE  = "AR (biallelic)"
N_PATIENTS   = 40

# ─── Variants ────────────────────────────────────────────────────────────────
VARIANTS = [
    {
        "hgvs_c":    "c.422G>A",
        "hgvs_p":    "p.Arg141Gln",
        "domain":    "ND1 sub-assembly contact surface",
        "mechanism": "Disrupts NDUFAF5 binding to ND1-containing P-module sub-assembly intermediate",
        "severity":  "intermediate",
        "ci_pct_range": (8, 16),
        "notes":     "Most frequently reported in European families; partial CI residual",
    },
    {
        "hgvs_c":    "c.602T>C",
        "hgvs_p":    "p.Leu201Pro",
        "domain":    "Alpha-helix 4 (core fold)",
        "mechanism": "Helix-breaking proline substitution causes alpha-helix collapse and full protein destabilization",
        "severity":  "severe",
        "ci_pct_range": (5, 10),
        "notes":     "Lethal neonatal onset in homozygotes; proline incompatible with alpha-helical fold",
    },
    {
        "hgvs_c":    "c.496G>A",
        "hgvs_p":    "p.Gly166Arg",
        "domain":    "Hydrophobic core packing",
        "mechanism": "Bulky arginine disrupts hydrophobic core packing; protein destabilization",
        "severity":  "severe",
        "ci_pct_range": (5, 11),
        "notes":     "Consanguineous families; severe infantile onset; identified by Sugiana 2008",
    },
    {
        "hgvs_c":    "c.280C>T",
        "hgvs_p":    "p.Arg94Cys",
        "domain":    "Surface-exposed residue",
        "mechanism": "Loss of positive charge at surface; moderate complex I residual in partial alleles",
        "severity":  "intermediate",
        "ci_pct_range": (10, 19),
        "notes":     "Middle Eastern families; heterozygous compound with splice variant; intermediate phenotype",
    },
    {
        "hgvs_c":    "c.944G>A",
        "hgvs_p":    "p.Trp315Ter",
        "domain":    "C-terminal truncation",
        "mechanism": "Near-C-terminus nonsense; truncates 29 aa; loss of C-terminal ND1-sub-assembly docking",
        "severity":  "severe",
        "ci_pct_range": (5, 9),
        "notes":     "Null allele; consanguineous; lethal neonatal if homozygous",
    },
    {
        "hgvs_c":    "c.IVS3+1G>A",
        "hgvs_p":    "p.?",
        "domain":    "Splice donor intron 3",
        "mechanism": "Canonical splice donor disruption; leaky splicing → partial NDUFAF5 residual",
        "severity":  "moderate",
        "ci_pct_range": (12, 20),
        "notes":     "Partial CI residual (12-20%); some residual normal splicing; milder clinical course",
    },
]

REGIONS     = ["European",  "Middle Eastern", "North African", "South Asian", "East Asian", "Latin American"]
REGION_W    = [0.27,         0.28,              0.17,            0.14,          0.09,          0.05]
OUTCOMES    = ["alive, supported", "alive, institutionalized", "deceased (respiratory)", "deceased (cardiac)"]
OUTCOME_W   = [0.38, 0.33, 0.19, 0.10]


def _patient(pid: int, var: dict) -> dict:
    onset_m = rng.randint(1, 30) if var["severity"] == "severe" else rng.randint(4, 72)
    ci_pct  = rng.randint(*var["ci_pct_range"])
    reg     = rng.choices(REGIONS, REGION_W)[0]
    out     = rng.choices(OUTCOMES, OUTCOME_W)[0]
    return {
        "id":                pid,
        "age_onset_months":  onset_m,
        "sex":               rng.choice(["M", "F"]),
        "mutation":          f"{var['hgvs_p']} / {var['hgvs_c']}",
        "mutation_class":    var["hgvs_p"],
        "region":            reg,
        "ci_activity_pct":  ci_pct,
        "leigh_mri":         rng.random() < 0.78,
        "lactic_acidosis":   rng.random() < 0.91,
        "hcm":               rng.random() < 0.18,
        "seizures":          rng.random() < 0.62,
        "hypotonia":         rng.random() < 0.88,
        "dev_regression":    rng.random() < 0.72,
        "respiratory_fail":  rng.random() < 0.48,
        "outcome":           out,
    }


def _make_patients() -> list:
    patients = []
    pid = 1
    var_counts = [8, 6, 7, 7, 6, 6]  # sums to 40
    for vi, cnt in enumerate(var_counts):
        for _ in range(cnt):
            patients.append(_patient(pid, VARIANTS[vi]))
            pid += 1
    return patients


PATIENTS = _make_patients()


def get_overview() -> dict:
    ff = {
        "Leigh_MRI":               round(sum(p["leigh_mri"] for p in PATIENTS) / N_PATIENTS * 100),
        "Lactic_acidosis":         round(sum(p["lactic_acidosis"] for p in PATIENTS) / N_PATIENTS * 100),
        "Hypotonia":               round(sum(p["hypotonia"] for p in PATIENTS) / N_PATIENTS * 100),
        "Developmental_regression":round(sum(p["dev_regression"] for p in PATIENTS) / N_PATIENTS * 100),
        "Seizures":                round(sum(p["seizures"] for p in PATIENTS) / N_PATIENTS * 100),
        "HCM":                     round(sum(p["hcm"] for p in PATIENTS) / N_PATIENTS * 100),
        "Respiratory_failure":     round(sum(p["respiratory_fail"] for p in PATIENTS) / N_PATIENTS * 100),
        "Riboflavin_responder":    0,   # HARD 0 — no FAD domain
        "Peripheral_neuropathy":   0,   # HARD 0 — critical DDx vs NDUFS1
        "Leukodystrophy":          0,   # HARD 0 — critical DDx vs NDUFV1
        "Hepatopathy":             0,   # HARD 0 — critical DDx vs POLG/DGUOK
        "Olfactory_bulb_lesions":  0,   # HARD 0 — critical DDx vs NDUFS4
    }

    return {
        "gene":            GENE,
        "gene_full_name":  "NADH:Ubiquinone Oxidoreductase Complex Assembly Factor 5",
        "also_known_as":   "C20orf7 (Chromosome 20 Open Reading Frame 7)",
        "omim_gene":       OMIM_GENE,
        "omim_disease":    OMIM_DISEASE,
        "disease_name":    DISEASE_NAME,
        "chromosome":      CHROMOSOME,
        "inheritance":     INHERITANCE,
        "cohort_n":        N_PATIENTS,
        "cohort_seed":     SEED,

        "protein": {
            "size_aa":  344,
            "size_kda": 39.0,
            "fold":     "Mixed alpha/beta fold; no FAD-binding domain; no TM helices; soluble matrix protein",
            "module":   "ND1-module P-module CI assembly (Class 3); distinct from MCIA (Class 1) and N-Q swap (Class 2)",
            "function": (
                "NDUFAF5 is a soluble matrix CI assembly factor that acts on the ND1-containing "
                "proton-pump (P-module) sub-assembly intermediate. It belongs to the same Class 3 "
                "BN-PAGE assembly intermediate class as NDUFAF3 and NDUFAF4, but acts independently "
                "(not as part of a defined obligate heterodimer). Loss of NDUFAF5 stalls the early "
                "ND1 sub-assembly, producing isolated CI deficiency with preserved CII/CIII/CIV."
            ),
        },

        "key_pathway_note": (
            "NDUFAF5 (C20orf7, 20p12.1) is a ND1-module CI assembly factor (Class 3). "
            "It acts on the same ND1-containing P-module sub-assembly as NDUFAF3 (2q33.1) and "
            "NDUFAF4 (6q16.3), but NDUFAF5 is on a different chromosome and acts independently — "
            "not as part of the NDUFAF3-NDUFAF4 obligate heterodimer. "
            "BN-PAGE: stalled early ND1-class intermediate — same pattern as NDUFAF3/NDUFAF4 deficiency; "
            "WES mandatory to assign chromosomal locus (20p12.1 = NDUFAF5). "
            "No FAD-binding domain → NO riboflavin response (unlike ACAD9, 50-60% Level B). "
            "First described by Sugiana et al. 2008 (Am J Hum Genet): lethal neonatal disease with "
            "null NDUFAF5 alleles."
        ),

        "biochemical_fingerprint": {
            "Complex_I":             "5–20% of control (SEVERE)",
            "Complex_II":            "Normal (100%)",
            "Complex_III":           "Normal (100%)",
            "Complex_IV":            "Normal (100%)",
            "Complex_V":             "Normal (100%)",
            "Pattern":               "ISOLATED CI deficiency — identical class to NDUFAF3/NDUFAF4 assembly factors",
            "Riboflavin_response":   "NONE (0%) — no FAD-binding domain in NDUFAF5",
            "BN-PAGE_class":         "ND1-module (Class 3): stalled ND1 sub-assembly intermediate",
            "NDUFAF3_protein_level": "Normal (NDUFAF5 loss does NOT destabilize NDUFAF3 or NDUFAF4 — no obligate heterodimer)",
        },

        "feature_frequencies_pct": ff,

        "ndufaf5_module_summary": {
            "gene":              "NDUFAF5 (C20orf7, 20p12.1)",
            "module_class":      "ND1-module / P-module CI assembly (Class 3)",
            "assembly_position": "Early ND1 sub-assembly — same class as NDUFAF3/NDUFAF4 but independent",
            "distinct_from_mcia": (
                "MCIA tetramer (ACAD9-NDUFAF1-ECSIT-TMEM126B) acts on ND2/ND5 module. "
                "NDUFAF5 acts on ND1 module. Completely separate assembly complexes."
            ),
            "ndufaf5_loss_effect": (
                "Stalled early ND1-containing sub-assembly intermediate accumulates. "
                "NDUFAF3 and NDUFAF4 proteins remain normal (no obligate heterodimer). "
                "Isolated CI deficiency (5-20%). CI holoenzyme absent."
            ),
            "vs_ndufaf3_ndufaf4_heterodimer": (
                "NDUFAF3-NDUFAF4 form an obligate heterodimer — loss of either destabilizes the partner. "
                "NDUFAF5 acts independently on the same ND1 sub-assembly but is NOT part of this heterodimer. "
                "In NDUFAF5 deficiency, NDUFAF3 and NDUFAF4 protein levels are normal."
            ),
        },

        "key_ddx": [
            {
                "feature":     "NDUFAF5 (20p12.1) vs NDUFAF3 (2q33.1) — SAME CLASS 3 BN-PAGE",
                "significance": (
                    "NDUFAF5 and NDUFAF3 both produce Class 3 ND1-module BN-PAGE intermediates with "
                    "near-identical isolated CI deficiency phenotype. Only WES chromosomal locus "
                    "distinguishes them: 20p12.1 (NDUFAF5) vs 2q33.1 (NDUFAF3). Key difference: "
                    "NDUFAF3 deficiency causes secondary NDUFAF4 destabilization; NDUFAF5 deficiency does NOT."
                ),
                "target_gene": "NDUFAF3",
            },
            {
                "feature":     "NDUFAF5 (20p12.1) vs NDUFAF4 (6q16.3) — SAME CLASS 3 BN-PAGE",
                "significance": (
                    "NDUFAF5 and NDUFAF4 both produce Class 3 ND1-module BN-PAGE intermediates. "
                    "Near-identical isolated CI deficiency phenotype. WES MANDATORY: 20p12.1 (NDUFAF5) "
                    "vs 6q16.3 (NDUFAF4). Key difference: NDUFAF4 deficiency causes secondary NDUFAF3 "
                    "destabilization; NDUFAF5 deficiency does NOT — protein blot may help distinguish."
                ),
                "target_gene": "NDUFAF4",
            },
            {
                "feature":     "NDUFAF5 vs ACAD9 — Riboflavin Response Key Distinguisher",
                "significance": (
                    "ACAD9 (MCIA complex, 3q21.3) is riboflavin-responsive (50-60%, Level B). "
                    "NDUFAF5 has no FAD domain — riboflavin response 0%. "
                    "Riboflavin response in CI deficiency: diagnose ACAD9. No response → NDUFAF5/NDUFAF3/NDUFAF4."
                ),
                "target_gene": "ACAD9",
            },
            {
                "feature":     "NDUFAF5 vs TIMMDC1 — Same Class 3 but High HCM in TIMMDC1",
                "significance": (
                    "TIMMDC1 (3q25.1) also belongs to Class 3 ND1-module but has severe HCM (>80%). "
                    "NDUFAF5 HCM is low (<20%). TIMMDC1 is an integral IMM protein (TM helices); "
                    "NDUFAF5 is soluble matrix. HCM rate and TM-helix status help differentiate."
                ),
                "target_gene": "TIMMDC1",
            },
            {
                "feature":     "NDUFAF5 vs NDUFS1 — NO Peripheral Neuropathy",
                "significance": (
                    "NDUFS1 (2q33.3) causes peripheral neuropathy in ~50% of patients. "
                    "NDUFAF5 deficiency: peripheral neuropathy 0%. Peripheral neuropathy is a strong "
                    "DDx pointer away from NDUFAF5 toward NDUFS1."
                ),
                "target_gene": "NDUFS1",
            },
            {
                "feature":     "NDUFAF5 vs NDUFV1 — NO Leukodystrophy",
                "significance": (
                    "NDUFV1 (11q13.2) causes leukodystrophy in 40-50% of patients. "
                    "NDUFAF5 deficiency: leukodystrophy 0%. Leukodystrophy on MRI is a strong DDx "
                    "pointer toward NDUFV1 and away from NDUFAF5."
                ),
                "target_gene": "NDUFV1",
            },
            {
                "feature":     "NDUFAF5 vs POLG/DGUOK — NO Hepatopathy",
                "significance": (
                    "POLG (15q26.1) and DGUOK (2p13.1) cause hepatopathy. "
                    "NDUFAF5 deficiency: hepatopathy 0%. Hepatopathy is a strong DDx pointer away from NDUFAF5."
                ),
                "target_gene": "POLG / DGUOK",
            },
        ],

        "clinical_alerts": [
            "🔴 METFORMIN — ABSOLUTE CONTRAINDICATION: Direct ND1 complex territory inhibitor. Biguanides inhibit CI at the ND1-containing Q/P module. NDUFAF5 patients have 5-20% CI — metformin is lethal.",
            "🔴 VALPROATE (VPA) — ABSOLUTE CONTRAINDICATION: Triple mechanism: CoA sequestration, POLG inhibition, MT-ND subunit expression suppression.",
            "🔴 LINEZOLID — ABSOLUTE CONTRAINDICATION: 23S rRNA inhibition blocks MT-ND1–6 subunit synthesis — shuts down all 7 mtDNA-encoded CI subunits.",
            "🔴 CHLORAMPHENICOL — ABSOLUTE CONTRAINDICATION: Same mitoribosomal mechanism as linezolid.",
            "🔴 KETOGENIC DIET — CONTRAINDICATED: CI is absent/minimal; NADH cannot be reoxidised. Beta-oxidation floods matrix with NADH with no ETC outlet. Metabolic crisis.",
            "🟠 PROPOFOL — AVOID (PRIS risk): Propofol infusion syndrome via CIV inhibition + fatty acid oxidation uncoupling. Use sevoflurane for anaesthesia.",
            "🟡 PHENOBARBITAL — HIGH CAUTION: Secondary CI inhibitor effect; use LEV as first-choice AED (renal, no mito toxicity).",
            "🟢 SUCCINATE — Level C: CII substrate bypasses NDUFAF5-stalled CI entirely; allows CII → CIII → CIV electron flow.",
            "🟢 THIAMINE B1 — Level C MANDATORY EMPIRIC: Exclude SLC19A3 (thiamine transporter) and BTD (biotin-thiamine-responsive basal ganglia disease) before confirming NDUFAF5.",
            "🟢 BIOTIN — Level C MANDATORY EMPIRIC: Exclude BTD (biotinidase deficiency) and HLCS before CI gene panel.",
            "🔵 NDUFAF5 vs NDUFAF3/NDUFAF4 — WES MANDATORY: Same Class 3 BN-PAGE. Only chromosomal locus distinguishes: 20p12.1 (NDUFAF5) vs 2q33.1 (NDUFAF3) vs 6q16.3 (NDUFAF4).",
            "🔵 NDUFAF5 vs ACAD9 — RIBOFLAVIN CRITICAL DDx: No FAD domain → 0% riboflavin response. Riboflavin response rules in ACAD9, rules out NDUFAF5.",
            "🔵 PROTEIN BLOT MAY HELP HERE: Unlike NDUFAF4 deficiency (where NDUFAF3 is secondarily reduced), NDUFAF5 loss does NOT destabilize NDUFAF3 or NDUFAF4 — normal NDUFAF3/NDUFAF4 protein level suggests NDUFAF5 deficiency over the heterodimer pair.",
        ],
    }


def get_breakdown() -> dict:
    # Mutation distribution
    var_counts = {}
    for p in PATIENTS:
        mc = p["mutation_class"]
        var_counts[mc] = var_counts.get(mc, 0) + 1

    variant_dist = [
        {"mutation_class": k, "count": v}
        for k, v in sorted(var_counts.items(), key=lambda x: -x[1])
    ]

    # Onset bins
    onset_bins = {"0–6 m": 0, "7–24 m": 0, "25–60 m": 0, "5–10 yr": 0}
    for p in PATIENTS:
        o = p["age_onset_months"]
        if o <= 6:    onset_bins["0–6 m"]   += 1
        elif o <= 24: onset_bins["7–24 m"]  += 1
        elif o <= 60: onset_bins["25–60 m"] += 1
        else:         onset_bins["5–10 yr"] += 1

    # Region distribution
    reg_counts = {}
    for p in PATIENTS:
        reg_counts[p["region"]] = reg_counts.get(p["region"], 0) + 1
    region_dist = [{"region": k, "count": v} for k, v in sorted(reg_counts.items(), key=lambda x: -x[1])]

    # Outcome distribution
    out_counts = {}
    for p in PATIENTS:
        out_counts[p["outcome"]] = out_counts.get(p["outcome"], 0) + 1
    outcome_dist = [{"outcome": k, "count": v} for k, v in sorted(out_counts.items(), key=lambda x: -x[1])]

    contraindicated_drugs = [
        {"drug": "Metformin",       "mechanism": "Direct ND1/CI inhibitor — ND1 complex territory; biguanide binds CI at matrix side", "class": "ABSOLUTE CI"},
        {"drug": "Valproate (VPA)", "mechanism": "Triple: CoA sequestration, POLG inhibition, MT-ND subunit expression suppression",   "class": "ABSOLUTE CI"},
        {"drug": "Linezolid",       "mechanism": "23S rRNA inhibition — blocks MT-ND1–6 synthesis; shuts all 7 mtDNA CI subunits",     "class": "ABSOLUTE CI"},
        {"drug": "Chloramphenicol", "mechanism": "Same mitoribosomal 23S rRNA mechanism as linezolid",                                 "class": "ABSOLUTE CI"},
        {"drug": "Ketogenic Diet",  "mechanism": "NADH cannot be reoxidised (CI absent/minimal); floods matrix — metabolic crisis",    "class": "CONTRAINDICATED"},
        {"drug": "Propofol",        "mechanism": "PRIS: CIV inhibition + fatty acid oxidation uncoupling",                             "class": "AVOID"},
        {"drug": "Phenobarbital",   "mechanism": "Secondary CI inhibitor; use LEV instead",                                            "class": "HIGH CAUTION"},
    ]

    treatment_protocols = [
        {"agent": "Succinate",          "dose": "500–2000 mg/day oral", "evidence": "Level C", "rationale": "CII substrate — bypasses stalled ND1/CI entirely; direct CII → III electron flow"},
        {"agent": "CoQ10 / Ubiquinol",  "dose": "10–30 mg/kg/day",     "evidence": "Level C", "rationale": "Downstream CI support; improves overall ETC electron flow"},
        {"agent": "Thiamine (B1)",       "dose": "100–600 mg/day",      "evidence": "Level C — MANDATORY EMPIRIC", "rationale": "Exclude SLC19A3 / BTD before confirming NDUFAF5"},
        {"agent": "Biotin",             "dose": "10–40 mg/day",         "evidence": "Level C — MANDATORY EMPIRIC", "rationale": "Exclude BTD / HLCS before CI gene panel"},
        {"agent": "Carnitine (L-Carn)", "dose": "50–100 mg/kg/day",     "evidence": "Level C", "rationale": "Secondary carnitine deficiency possible; replete if low"},
        {"agent": "IV Dextrose (GIR 6–8)", "dose": "GIR 6–8 mg/kg/min", "evidence": "Consensus", "rationale": "NEVER fast — CI-absent patients cannot sustain fasting gluconeogenesis"},
        {"agent": "LEV (Levetiracetam)","dose": "20–60 mg/kg/day",      "evidence": "Consensus AED", "rationale": "Preferred AED — renal excretion, no mitochondrial toxicity"},
        {"agent": "Sevoflurane",        "dose": "Standard GA",          "evidence": "Consensus", "rationale": "Preferred volatile agent — NOT propofol (PRIS risk)"},
    ]

    nd1_module_steps = [
        {
            "step":  1,
            "event": "NDUFAF5 Associates with Early ND1-Containing P-Module Sub-Assembly",
            "status_in_ndufaf5_deficiency": "DISRUPTED — root cause",
            "note": "NDUFAF5 loss prevents normal ND1 sub-assembly maturation. Class 3 BN-PAGE intermediate accumulates.",
        },
        {
            "step":  2,
            "event": "ND3, ND4L, ND6 Incorporation into P-Module Sub-Assembly",
            "status_in_ndufaf5_deficiency": "STALLED — upstream block prevents subunit incorporation",
            "note": "P-module cannot be completed without NDUFAF5-assisted ND1 sub-assembly maturation.",
        },
        {
            "step":  3,
            "event": "P-Module Merges with Q-Module (NDUFS3) and N-Module (NDUFV1/NDUFV2)",
            "status_in_ndufaf5_deficiency": "BLOCKED — stalled P-module cannot merge",
            "note": "Holoenzyme assembly cannot proceed. CII, CIII, CIV remain normal.",
        },
        {
            "step":  4,
            "event": "Mature CI Holoenzyme Formation",
            "status_in_ndufaf5_deficiency": "ABSENT — isolated CI deficiency (5-20%)",
            "note": "No mature CI holoenzyme on BN-PAGE. NDUFAF3 and NDUFAF4 proteins remain present and normal.",
        },
    ]

    return {
        "patients":              PATIENTS,
        "variant_distribution":  variant_dist,
        "onset_distribution":    [{"bin": k, "count": v} for k, v in onset_bins.items()],
        "region_distribution":   region_dist,
        "outcome_distribution":  outcome_dist,
        "contraindicated_drugs": contraindicated_drugs,
        "treatment_protocols":   treatment_protocols,
        "nd1_module_steps":      nd1_module_steps,

        "assembly_module_comparison": [
            {
                "class":      "Class 1 — MCIA / ND2-ND5 Module",
                "members":    "ACAD9, NDUFAF1, ECSIT, TMEM126B",
                "module":     "ND2/ND5 membrane arm sub-assembly",
                "bnpage":     "ND2-ND5 sub-assembly intermediate",
                "riboflavin": "ACAD9 only (50-60%, Level B); others 0%",
            },
            {
                "class":      "Class 2 — N-Q Swap / NDUFA12 Paralog",
                "members":    "NDUFAF2, NDUFA12",
                "module":     "N-Q module interface; paralog-swap mechanism",
                "bnpage":     "N-Q sub-assembly intermediate",
                "riboflavin": "None (0%)",
            },
            {
                "class":      "Class 3 — ND1-Module / Early P-Module (NDUFAF5 is here)",
                "members":    "NDUFAF3, NDUFAF4, NDUFAF5, TIMMDC1",
                "module":     "Early ND1-containing P-module sub-assembly",
                "bnpage":     "Early ND1 sub-assembly intermediate",
                "riboflavin": "None (0%) — none of these have a FAD domain",
            },
        ],

        "nd1_class3_comparison": [
            {
                "gene":           "NDUFAF3",
                "chromosome":     "2q33.1",
                "omim_gene":      "612911",
                "heterodimer":    "Obligate with NDUFAF4",
                "partner_effect": "NDUFAF4 secondarily reduced",
                "hcm":            "15-25%",
                "timmdc1_note":   "N/A",
            },
            {
                "gene":           "NDUFAF4",
                "chromosome":     "6q16.3",
                "omim_gene":      "611776",
                "heterodimer":    "Obligate with NDUFAF3",
                "partner_effect": "NDUFAF3 secondarily reduced",
                "hcm":            "15-25%",
                "timmdc1_note":   "N/A",
            },
            {
                "gene":           "NDUFAF5",
                "chromosome":     "20p12.1",
                "omim_gene":      "612360",
                "heterodimer":    "None (acts independently)",
                "partner_effect": "NDUFAF3/NDUFAF4 protein levels NORMAL",
                "hcm":            "<20%",
                "timmdc1_note":   "N/A",
            },
            {
                "gene":           "TIMMDC1",
                "chromosome":     "3q25.1",
                "omim_gene":      "615530",
                "heterodimer":    "None (integral IMM)",
                "partner_effect": "N/A",
                "hcm":            ">80% (key DDx)",
                "timmdc1_note":   "Only integral-IMM (TM helices) in Class 3",
            },
        ],
    }


def get_definitions() -> dict:
    return {
        "concepts": [
            {
                "term":       "NDUFAF5 (C20orf7)",
                "definition": (
                    "NADH:Ubiquinone Oxidoreductase Complex Assembly Factor 5, also known as chromosome 20 "
                    "open reading frame 7. 344-amino acid, ~39 kDa soluble matrix CI assembly factor. "
                    "Acts on the ND1-containing proton-pump (P-module) sub-assembly intermediate — the "
                    "same Class 3 BN-PAGE assembly class as NDUFAF3 and NDUFAF4, but acts independently "
                    "(not as part of an obligate heterodimer). Loss of NDUFAF5 causes isolated CI deficiency "
                    "without secondary destabilization of NDUFAF3 or NDUFAF4."
                ),
            },
            {
                "term":       "Class 3 BN-PAGE ND1-Module — NDUFAF5 vs NDUFAF3 vs NDUFAF4",
                "definition": (
                    "NDUFAF3, NDUFAF4, and NDUFAF5 all produce the same Class 3 BN-PAGE CI assembly "
                    "intermediate (stalled early ND1 sub-assembly). The three genes are on different "
                    "chromosomes (2q33.1, 6q16.3, 20p12.1 respectively) and their deficiencies are "
                    "phenotypically near-identical. Key discriminator: NDUFAF3 or NDUFAF4 deficiency "
                    "causes secondary reduction of the partner protein (obligate heterodimer interdependence), "
                    "while NDUFAF5 deficiency leaves NDUFAF3 and NDUFAF4 protein levels normal."
                ),
            },
            {
                "term":       "Mitochondrial Complex I Deficiency — NDUFAF5-related (OMIM *612360)",
                "definition": (
                    "Autosomal recessive mitochondrial complex I deficiency caused by biallelic "
                    "loss-of-function mutations in NDUFAF5 (C20orf7, 20p12.1). Phenotype: Leigh syndrome "
                    "or fatal neonatal mitochondrial disease, isolated CI deficiency 5-20%, CII/CIII/CIV "
                    "normal, lactic acidosis, Leigh MRI (basal ganglia/brainstem), hypotonia, seizures. "
                    "No peripheral neuropathy, no leukodystrophy, no hepatopathy, no riboflavin response. "
                    "First described by Sugiana et al. 2008 Am J Hum Genet."
                ),
            },
            {
                "term":       "NDUFAF5 Independence vs NDUFAF3-NDUFAF4 Obligate Heterodimer",
                "definition": (
                    "The NDUFAF3-NDUFAF4 obligate heterodimer means loss of either partner "
                    "destabilizes the other. NDUFAF5 acts independently on the same ND1 sub-assembly: "
                    "loss of NDUFAF5 does NOT reduce NDUFAF3 or NDUFAF4 protein. In practice, normal "
                    "NDUFAF3 and NDUFAF4 protein levels on Western blot in a Class-3 BN-PAGE patient "
                    "is an important clue pointing toward NDUFAF5 (or TIMMDC1) over NDUFAF3/NDUFAF4."
                ),
            },
            {
                "term":       "Metformin Absolute Contraindication in CI Deficiency",
                "definition": (
                    "Biguanides (metformin) inhibit CI directly at the matrix-arm ND1 complex territory. "
                    "In NDUFAF5 deficiency (CI at 5-20%), any further CI inhibition is lethal. "
                    "Metformin is absolutely contraindicated in all nuclear-encoded CI deficiencies."
                ),
            },
            {
                "term":       "Succinate — CII Substrate (CI Bypass)",
                "definition": (
                    "Succinate enters the ETC at Complex II (succinate dehydrogenase), bypassing CI entirely. "
                    "In NDUFAF5 deficiency where CI is stalled, succinate supplementation allows some "
                    "residual OXPHOS activity via CII → CIII → CIV pathway. Level C evidence."
                ),
            },
        ],

        "thresholds": [
            {"parameter": "CI activity (% control)",     "threshold": "<20%",     "significance": "Diagnostic for CI deficiency; NDUFAF5 typically 5-20%"},
            {"parameter": "Lactate (plasma)",            "threshold": ">2.5 mmol/L", "significance": "Elevated in ~91% of NDUFAF5 patients; >4 = crisis threshold"},
            {"parameter": "Lactate:pyruvate ratio",      "threshold": ">25",      "significance": "Elevated L:P ratio supports mitochondrial NADH block (ETC defect)"},
            {"parameter": "CSF lactate",                 "threshold": ">2.5 mmol/L", "significance": "Leigh syndrome workup — elevated in CI deficiency with CNS involvement"},
            {"parameter": "Onset age (severe alleles)",  "threshold": "<6 months", "significance": "Neonatal-lethal onset with null alleles; intermediate: 4-72 months"},
            {"parameter": "CI activity (riboflavin threshold)", "threshold": "0% response", "significance": "NDUFAF5 has NO FAD domain — zero riboflavin response expected"},
            {"parameter": "NDUFAF3/NDUFAF4 protein",    "threshold": "Normal",   "significance": "Normal NDUFAF3/NDUFAF4 protein in Class 3 BN-PAGE patient → consider NDUFAF5"},
        ],

        "standards": [
            {"code": "ACMG/AMP 2015", "title": "Variant classification guidelines — pathogenicity criteria for NDUFAF5"},
            {"code": "MITOMAP", "title": "Mitochondrial disease gene/variant database"},
            {"code": "OMIM *612360", "title": "NDUFAF5 gene entry — C20orf7"},
            {"code": "OMIM #256000", "title": "Mitochondrial Complex I Deficiency — CI deficiency spectrum"},
            {"code": "REACTOME R-HSA-611105", "title": "Respiratory electron transport — CI assembly module"},
            {"code": "Helsinki Declaration", "title": "Ethical framework for human subject research"},
        ],

        "references": [
            {
                "id":        "sugiana_2008",
                "citation":  "Sugiana C et al. (2008) Mutation of C20orf7 disrupts complex I assembly and causes lethal neonatal mitochondrial disease. Am J Hum Genet 83(4):468–78.",
                "relevance": "First description of NDUFAF5 (C20orf7) mutations causing CI deficiency; lethal neonatal phenotype with null alleles; established NDUFAF5 as ND1-module CI assembly factor.",
            },
            {
                "id":        "guerrero_castillo_2017",
                "citation":  "Guerrero-Castillo S et al. (2017) The assembly pathway of mitochondrial respiratory chain complex I. Cell Metab 25(1):128–139.",
                "relevance": "Defines CI assembly intermediates including Class 3 ND1-module where NDUFAF5 acts; maps NDUFAF5 position relative to NDUFAF3/NDUFAF4 in temporal assembly order.",
            },
            {
                "id":        "stroud_2016",
                "citation":  "Stroud DA et al. (2016) Accessory subunits are integral for assembly and function of human mitochondrial complex I. Nature 538:123–126.",
                "relevance": "BN-PAGE intermediate classification; confirms three assembly intermediate classes including Class 3 (ND1-module) where NDUFAF5, NDUFAF3, and NDUFAF4 function.",
            },
            {
                "id":        "fassone_2012",
                "citation":  "Fassone E & Rahman S (2012) Complex I deficiency: clinical features, biochemistry and molecular genetics. J Med Genet 49(9):578–90.",
                "relevance": "Comprehensive CI genetics review; frames NDUFAF5 within the CI assembly factor landscape.",
            },
            {
                "id":        "sazanov_2015",
                "citation":  "Sazanov LA (2015) A giant molecular proton pump: structure and mechanism of respiratory complex I. Nat Rev Mol Cell Biol 16(6):375–88.",
                "relevance": "CI structure review; contextualizes the P-module (ND1/ND3/ND4L/ND6) where NDUFAF5 operates.",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps({"overview": get_overview(), "breakdown": get_breakdown(), "definitions": get_definitions()}, indent=2, default=str))
