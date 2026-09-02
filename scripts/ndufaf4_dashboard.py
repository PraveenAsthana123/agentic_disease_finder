#!/usr/bin/env python3
"""NDUFAF4 — Mitochondrial Complex I Deficiency (Early ND1-Module Assembly / NDUFAF3 Obligate Heterodimer).

NDUFAF4 (C6orf66, Chromosome 6 Open Reading Frame 66) is a CI assembly factor
that forms an obligate heterodimer with NDUFAF3 (C3orf60). Together they constitute
the earliest committed CI assembly sub-complex, acting on the ND1-containing
proton-pump (P-module) sub-assembly intermediate — a fundamentally different
module from the MCIA complex (ACAD9–NDUFAF1–ECSIT–TMEM126B), which handles the
ND2/ND5 module.

  NDUFAF4 gene  OMIM *611776
  Disease       Mitochondrial Complex I Deficiency, Nuclear Type 20 (MC1DN20) OMIM #618245
  Inheritance   AR (autosomal recessive, biallelic)
  Chromosome    6q16.3

PATHOPHYSIOLOGY (NDUFAF4 / Early ND1-Module / NDUFAF3-NDUFAF4 Heterodimer):
  CI assembly proceeds through modular intermediates. NDUFAF3-NDUFAF4 heterodimer
  acts at the ND1-containing sub-assembly:
    1. NDUFAF4 + NDUFAF3 → obligate early heterodimer (tightest earliest interaction
       in ND1-module assembly; loss of NDUFAF4 destabilizes NDUFAF3 protein).
    2. Heterodimer associates with ND1-containing IMM sub-assembly intermediate.
    3. Additional P-module subunits (ND3, ND4L, ND6) are incorporated.
    4. P-module sub-assembly merges with Q-module (NDUFS3-containing) and N-module
       (NDUFV1/NDUFV2-containing) to form mature CI holoenzyme.
  Loss of NDUFAF4:
    • NDUFAF3 loses its obligate partner → NDUFAF3 protein is secondarily destabilized.
    • NDUFAF3-NDUFAF4 heterodimer cannot assemble → early ND1-module assembly stalled.
    • Isolated CI deficiency (5–20% of control), CII/CIII/CIV normal.
    • BN-PAGE: stalled early ND1-containing sub-assembly intermediate — DISTINCT from
      MCIA complex (ND2/ND5 class) and N-Q module (NDUFAF2-NDUFA12 class).

NDUFAF4 UNIQUE FEATURES — NDUFAF3-NDUFAF4 OBLIGATE HETERODIMER PARTNER / EARLIEST ND1-MODULE:
  1. NDUFAF4 LOSS DESTABILIZES NDUFAF3 — OBLIGATE PARTNER INTERDEPENDENCE:
     NDUFAF4 forms a tight obligate heterodimer with NDUFAF3 (C3orf60, 2q33.1).
     Loss of NDUFAF4 causes secondary destabilization of NDUFAF3 protein (the reciprocal
     of NDUFAF3 deficiency where NDUFAF4 is secondarily depleted). Both genes must be
     analysed when either partner shows reduced protein on Western blot.
  2. EARLY P-MODULE / ND1 TERRITORY — DISTINCT FROM MCIA (ND2/ND5):
     NDUFAF3-NDUFAF4 operates on the ND1 proton-pump sub-assembly, while the MCIA
     tetramer operates on ND2/ND5. BN-PAGE intermediate class differs: ND1-module
     class (NDUFAF3, NDUFAF4, TIMMDC1) vs ND2/ND5 class (MCIA members).
  3. NO FAD-BINDING / NO RIBOFLAVIN RESPONSE:
     NDUFAF4 has no FAD-binding domain. High-dose riboflavin does NOT rescue NDUFAF4
     deficiency. Critical DDx: ACAD9 (50-60% riboflavin responsive, Level B) vs
     NDUFAF4 (0% — no FAD domain, no riboflavin response).
  4. CHROMOSOME 6q16.3 — WES MANDATORY TO DISTINGUISH FROM NDUFAF3 (2q33.1):
     NDUFAF4 (6q16.3) and NDUFAF3 (2q33.1) are obligate heterodimer partners with
     near-identical CI deficiency phenotype. Only WES with chromosomal locus resolution
     distinguishes them: 6q16.3 = NDUFAF4, 2q33.1 = NDUFAF3.
  5. BN-PAGE ND1-MODULE INTERMEDIATE — THIRD CLASS DISTINCT FROM MCIA AND N-Q:
     Three distinct BN-PAGE CI assembly intermediate classes exist:
       Class 1 (MCIA / ND2-ND5): ACAD9, NDUFAF1, ECSIT, TMEM126B
       Class 2 (N-Q swap): NDUFAF2, NDUFA12
       Class 3 (ND1-module): NDUFAF3, NDUFAF4, TIMMDC1
     NDUFAF4 belongs exclusively to Class 3. NDUFAF3 protein level is ALSO secondarily
     reduced in NDUFAF4 patients — protein blot cannot distinguish the two; WES mandatory.

DISTINGUISHING FEATURES vs OTHER CI-LEIGH AND CI-ASSEMBLY GENES:
  vs NDUFAF3 (2q33.1): NEAR-IDENTICAL PHENOTYPE — obligate heterodimer partner.
    WES MANDATORY. Both AR, both Leigh, both isolated CI 5-20%, both no riboflavin.
    Only chromosomal locus distinguishes: 6q16.3 (NDUFAF4) vs 2q33.1 (NDUFAF3).
    NDUFAF4 loss also destabilizes NDUFAF3 protein — protein level may be low for both.
  vs ACAD9 (3q21.3): MCIA complex member (ND2/ND5 module) vs NDUFAF4 (ND1 module).
    ACAD9 riboflavin-responsive (50-60%, Level B); NDUFAF4 NO riboflavin response.
    ACAD9 has bimodal severity (exercise-intolerance dominant pArg518His); NDUFAF4 pure Leigh.
    BN-PAGE: ND2/ND5 class (ACAD9) vs ND1-module class (NDUFAF4).
  vs NDUFAF2 (5q12.1): N-Q module class (NDUFAF2 is an NDUFA12 paralog/swap).
    NDUFAF2 stalls at N-Q interface; NDUFAF4 stalls at early ND1 module.
    Both: no riboflavin, Leigh, isolated CI 5-20%. BN-PAGE class distinguishes.
  vs TIMMDC1 (same ND1-module class, 3q25.1): Both ND1-class, but TIMMDC1 has
    severe cardiomyopathy (>80%) while NDUFAF4 HCM is low (15-25%).
  vs NDUFS1 (2q33.3): NDUFS1 causes peripheral neuropathy (50%); NDUFAF4 NO neuropathy.
  vs NDUFV1 (11q13.2): NDUFV1 causes leukodystrophy (40-50%); NDUFAF4 NO leukodystrophy.
  vs POLG/DGUOK: Both cause hepatopathy; NDUFAF4 NO hepatopathy — key exclusion.
"""

import random
import math

SEED = 685
rng  = random.Random(SEED)

GENE         = "NDUFAF4"
OMIM_GENE    = "611776"
OMIM_DISEASE = "618245"
DISEASE_NAME = "Mitochondrial Complex I Deficiency, Nuclear Type 20 (MC1DN20)"
CHROMOSOME   = "6q16.3"
INHERITANCE  = "AR (biallelic)"
N_PATIENTS   = 40

# ─── Variants ────────────────────────────────────────────────────────────────
VARIANTS = [
    {
        "hgvs_c":    "c.130C>T",
        "hgvs_p":    "p.Arg44Trp",
        "domain":    "NDUFAF3-contact surface",
        "mechanism": "Disrupts obligate NDUFAF3-NDUFAF4 heterodimer interface; secondary NDUFAF3 destabilization",
        "severity":  "intermediate",
        "ci_pct_range": (8, 15),
        "notes":     "Most reported pathogenic variant; NDUFAF3 protein level secondarily reduced",
    },
    {
        "hgvs_c":    "c.179T>C",
        "hgvs_p":    "p.Leu60Pro",
        "domain":    "Alpha-helix 2 (core fold)",
        "mechanism": "Helix-breaking proline substitution causes alpha-helix collapse; full protein destabilization",
        "severity":  "severe",
        "ci_pct_range": (5, 10),
        "notes":     "Severe neonatal presentation; proline incompatible with alpha-helical fold",
    },
    {
        "hgvs_c":    "c.262G>A",
        "hgvs_p":    "p.Ala88Thr",
        "domain":    "Hydrophobic core",
        "mechanism": "Hydrophobic-to-polar substitution disrupts core packing; partial destabilization",
        "severity":  "intermediate",
        "ci_pct_range": (10, 18),
        "notes":     "Consanguineous families; some residual CI activity; intermediate phenotype",
    },
    {
        "hgvs_c":    "c.375G>A",
        "hgvs_p":    "p.Trp125Ter",
        "domain":    "C-terminal NDUFAF3-binding region",
        "mechanism": "Near-end nonsense — truncates last 5 aa; NDUFAF3 docking surface lost",
        "severity":  "severe",
        "ci_pct_range": (5, 9),
        "notes":     "Null allele; consanguineous; severe neonatal-onset; NDUFAF3 protein also absent",
    },
    {
        "hgvs_c":    "c.IVS2+1G>A",
        "hgvs_p":    "p.?",
        "domain":    "Splice donor intron 2",
        "mechanism": "Canonical splice donor disruption; leaky splicing → partial NDUFAF4 residual",
        "severity":  "moderate",
        "ci_pct_range": (12, 20),
        "notes":     "Partial CI residual (12-20%); clinical syndrome milder; school-age onset possible",
    },
]

REGIONS     = ["European",  "Middle Eastern", "North African", "South Asian", "East Asian", "Latin American"]
REGION_W    = [0.28,         0.27,              0.18,            0.14,          0.08,          0.05]
OUTCOMES    = ["alive, supported", "alive, institutionalized", "deceased (respiratory)", "deceased (cardiac)"]
OUTCOME_W   = [0.40, 0.32, 0.18, 0.10]


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
        "leigh_mri":         rng.random() < 0.75,
        "lactic_acidosis":   rng.random() < 0.92,
        "hcm":               rng.random() < 0.20,
        "seizures":          rng.random() < 0.60,
        "hypotonia":         rng.random() < 0.90,
        "dev_regression":    rng.random() < 0.70,
        "respiratory_fail":  rng.random() < 0.50,
        "outcome":           out,
    }


def _make_patients() -> list:
    patients = []
    pid = 1
    var_counts = [8, 7, 9, 8, 8]  # sums to 40
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
        "gene_full_name":  "NADH:Ubiquinone Oxidoreductase Complex Assembly Factor 4",
        "also_known_as":   "C6orf66 (Chromosome 6 Open Reading Frame 66)",
        "omim_gene":       OMIM_GENE,
        "omim_disease":    OMIM_DISEASE,
        "disease_name":    DISEASE_NAME,
        "chromosome":      CHROMOSOME,
        "inheritance":     INHERITANCE,
        "cohort_n":        N_PATIENTS,
        "cohort_seed":     SEED,

        "protein": {
            "size_aa":  130,
            "size_kda": 14.5,
            "fold":     "Beta-sheet / alpha-helix mixed fold; no canonical enzyme domain",
            "module":   "Early ND1-module P-module CI assembly; NDUFAF3-NDUFAF4 early heterodimer complex",
            "function": (
                "Forms obligate heterodimer with NDUFAF3 (C3orf60). The NDUFAF4-NDUFAF3 heterodimer "
                "constitutes the earliest committed CI assembly sub-complex, acting on the ND1-containing "
                "proton-pump (P-module) sub-assembly intermediate upstream of the MCIA tetramer (ND2/ND5 module). "
                "NDUFAF4 loss causes secondary destabilization of its obligate partner NDUFAF3."
            ),
        },

        "key_pathway_note": (
            "NDUFAF4 (C6orf66, 6q16.3) forms an obligate heterodimer with NDUFAF3 (C3orf60, 2q33.1). "
            "This heterodimer is the EARLIEST committed CI assembly complex, acting on ND1-containing "
            "sub-assemblies (P-module territory). Loss of NDUFAF4 causes secondary destabilization of NDUFAF3. "
            "BN-PAGE shows stalled early ND1-class intermediates — a DISTINCT pattern from MCIA complex "
            "(ND2/ND5 class: ACAD9, NDUFAF1, ECSIT, TMEM126B) and N-Q module (NDUFAF2, NDUFA12). "
            "No FAD-binding domain → NO riboflavin response (unlike ACAD9, 50-60% Level B). "
            "CRITICAL DDx: NDUFAF4 (6q16.3) vs NDUFAF3 (2q33.1) — near-identical phenotype, only WES "
            "chromosomal locus distinguishes the two obligate heterodimer partners."
        ),

        "biochemical_fingerprint": {
            "Complex_I":             "5–20% of control (SEVERE)",
            "Complex_II":            "Normal (100%)",
            "Complex_III":           "Normal (100%)",
            "Complex_IV":            "Normal (100%)",
            "Complex_V":             "Normal (100%)",
            "Pattern":               "ISOLATED CI deficiency — identical to all NDUFAF-class assembly factors",
            "Riboflavin_response":   "NONE (0%) — no FAD-binding domain in NDUFAF4",
            "BN-PAGE_class":         "ND1-module (Class 3): stalled ND1 sub-assembly intermediate",
            "NDUFAF3_protein_level": "Secondarily reduced — NDUFAF4 loss destabilizes obligate partner NDUFAF3",
        },

        "feature_frequencies_pct": ff,

        "ndufaf4_ndufaf3_module_summary": {
            "heterodimer":       "NDUFAF4 (6q16.3) + NDUFAF3 (2q33.1)",
            "module_class":      "Early ND1-module / P-module CI assembly",
            "assembly_position": "Earliest committed CI assembly step (upstream of MCIA tetramer)",
            "distinct_from_mcia": (
                "MCIA tetramer (ACAD9-NDUFAF1-ECSIT-TMEM126B) acts on ND2/ND5 module. "
                "NDUFAF4-NDUFAF3 acts on ND1 module. Completely separate assembly complexes."
            ),
            "ndufaf4_loss_effect": (
                "NDUFAF3 protein secondarily destabilized (loses obligate partner). "
                "Early ND1 sub-assembly intermediate accumulates. CI holoenzyme absent."
            ),
        },

        "key_ddx": [
            {
                "feature":     "NDUFAF4 (6q16.3) vs NDUFAF3 (2q33.1) — NEAR-IDENTICAL PHENOTYPE",
                "significance": (
                    "NDUFAF4 and NDUFAF3 are obligate heterodimer partners with nearly identical CI deficiency "
                    "phenotype (both Leigh, both isolated CI 5-20%, both no riboflavin response, both AR). "
                    "ONLY chromosomal locus distinguishes them on WES: 6q16.3 (NDUFAF4) vs 2q33.1 (NDUFAF3). "
                    "Both genes must always be sequenced when either is suspected. NDUFAF3 protein level is "
                    "secondarily reduced in NDUFAF4 deficiency — protein blot cannot distinguish the two."
                ),
                "target_gene": "NDUFAF3",
            },
            {
                "feature":     "NDUFAF4 vs ACAD9 — Riboflavin Response Key Distinguisher",
                "significance": (
                    "ACAD9 (MCIA complex, 3q21.3) is riboflavin-responsive (50-60%, Level B) due to FAD-binding domain. "
                    "NDUFAF4 has no FAD domain — riboflavin response 0%. If CI deficiency responds to riboflavin: "
                    "diagnose ACAD9. No response favours NDUFAF4 / NDUFAF3 / NDUFAF2 / other assembly factors."
                ),
                "target_gene": "ACAD9",
            },
            {
                "feature":     "NDUFAF4 vs NDUFV1 — NO Leukodystrophy",
                "significance": (
                    "NDUFV1 (CI subunit, 11q13.2) causes leukodystrophy in 40-50% of patients. "
                    "NDUFAF4 deficiency: leukodystrophy 0%. If leukodystrophy present on MRI: "
                    "sequence NDUFV1 before NDUFAF4."
                ),
                "target_gene": "NDUFV1",
            },
            {
                "feature":     "NDUFAF4 vs NDUFS1 — NO Peripheral Neuropathy",
                "significance": (
                    "NDUFS1 (2q33.3) causes peripheral neuropathy in ~50% of patients. "
                    "NDUFAF4 deficiency: peripheral neuropathy 0%. Peripheral neuropathy on presentation "
                    "is a strong DDx pointer away from NDUFAF4 toward NDUFS1."
                ),
                "target_gene": "NDUFS1",
            },
            {
                "feature":     "NDUFAF4 vs POLG/DGUOK — NO Hepatopathy",
                "significance": (
                    "POLG (15q26.1) and DGUOK (2p13.1) cause hepatopathy (liver failure, hepatic mtDNA depletion). "
                    "NDUFAF4 deficiency: hepatopathy 0%. Hepatopathy on presentation is a strong DDx pointer "
                    "away from NDUFAF4."
                ),
                "target_gene": "POLG / DGUOK",
            },
        ],

        "clinical_alerts": [
            "🔴 METFORMIN — ABSOLUTE CONTRAINDICATION: Direct ND1 complex territory inhibitor. Biguanides inhibit CI at the ND1-containing Q/P module. NDUFAF4 patients have 5-20% CI — metformin is lethal.",
            "🔴 VALPROATE (VPA) — ABSOLUTE CONTRAINDICATION: Triple mechanism: CoA sequestration, POLG inhibition, MT-ND subunit expression suppression.",
            "🔴 LINEZOLID — ABSOLUTE CONTRAINDICATION: 23S rRNA inhibition blocks MT-ND1–6 subunit synthesis — shuts down all 7 mtDNA-encoded CI subunits.",
            "🔴 CHLORAMPHENICOL — ABSOLUTE CONTRAINDICATION: Same mitoribosomal mechanism as linezolid.",
            "🔴 KETOGENIC DIET — CONTRAINDICATED: CI is absent/minimal; NADH cannot be reoxidised. Beta-oxidation floods matrix with NADH with no ETC outlet. Metabolic crisis.",
            "🟠 PROPOFOL — AVOID (PRIS risk): Propofol infusion syndrome via CIV inhibition + fatty acid oxidation uncoupling. Use sevoflurane for anaesthesia.",
            "🟡 PHENOBARBITAL — HIGH CAUTION: Secondary CI inhibitor effect; use LEV as first-choice AED (renal, no mito toxicity).",
            "🟢 SUCCINATE — Level C: CII substrate bypasses NDUFAF4-stalled CI entirely; bypasses ND1 module block.",
            "🟢 THIAMINE B1 — Level C MANDATORY EMPIRIC: Exclude SLC19A3 (thiamine transporter) and BTD (biotin-thiamine-responsive basal ganglia disease) before confirming NDUFAF4.",
            "🟢 BIOTIN — Level C MANDATORY EMPIRIC: Exclude BTD (biotinidase deficiency) and HLCS before CI gene sequencing.",
            "🔵 NDUFAF4 (6q16.3) vs NDUFAF3 (2q33.1) — WES MANDATORY: Near-identical phenotype; NDUFAF3 protein secondarily reduced in NDUFAF4 patients — protein blot unreliable; chromosomal locus only distinguisher.",
            "🔵 NDUFAF4 vs ACAD9 — RIBOFLAVIN CRITICAL DDx: No FAD domain → 0% riboflavin response. Riboflavin response rules in ACAD9, rules out NDUFAF4.",
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
        {"agent": "Thiamine (B1)",       "dose": "100–600 mg/day",      "evidence": "Level C — MANDATORY EMPIRIC", "rationale": "Exclude SLC19A3 / BTD before confirming NDUFAF4"},
        {"agent": "Biotin",             "dose": "10–40 mg/day",         "evidence": "Level C — MANDATORY EMPIRIC", "rationale": "Exclude BTD / HLCS before CI gene panel"},
        {"agent": "Carnitine (L-Carn)", "dose": "50–100 mg/kg/day",     "evidence": "Level C", "rationale": "Secondary carnitine deficiency possible; replete if low"},
        {"agent": "IV Dextrose (GIR 6–8)", "dose": "GIR 6–8 mg/kg/min", "evidence": "Consensus", "rationale": "NEVER fast — CI-absent patients cannot sustain fasting gluconeogenesis"},
        {"agent": "LEV (Levetiracetam)","dose": "20–60 mg/kg/day",      "evidence": "Consensus AED", "rationale": "Preferred AED — renal excretion, no mitochondrial toxicity"},
        {"agent": "Sevoflurane",        "dose": "Standard GA",          "evidence": "Consensus", "rationale": "Preferred volatile agent — NOT propofol (PRIS risk)"},
    ]

    nd1_module_steps = [
        {
            "step":  1,
            "event": "NDUFAF4 + NDUFAF3 → Obligate Early Heterodimer",
            "status_in_ndufaf4_deficiency": "DISRUPTED — root cause",
            "note": "NDUFAF4 loss prevents heterodimer formation; NDUFAF3 is secondarily destabilized and degraded.",
        },
        {
            "step":  2,
            "event": "NDUFAF4-NDUFAF3 Heterodimer Associates with ND1-Containing Sub-Assembly",
            "status_in_ndufaf4_deficiency": "STALLED — early ND1 intermediate accumulates",
            "note": "ND1 sub-assembly intermediate (Class 3 BN-PAGE band) accumulates without the NDUFAF4-NDUFAF3 chaperone.",
        },
        {
            "step":  3,
            "event": "ND3, ND4L, ND6 Incorporation into P-Module Sub-Assembly",
            "status_in_ndufaf4_deficiency": "BLOCKED — upstream stall prevents this step",
            "note": "P-module cannot be completed; additional proton-pump subunits cannot be incorporated.",
        },
        {
            "step":  4,
            "event": "P-Module Merges with Q-Module and N-Module → CI Holoenzyme",
            "status_in_ndufaf4_deficiency": "ABSENT — CI holoenzyme not formed",
            "note": "No mature CI holoenzyme on BN-PAGE. Isolated CI deficiency (5-20%). CII/CIII/CIV normal.",
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
                "bnpage":     "ND2-ND5 sub-assembly intermediate (absent CI holoenzyme)",
                "riboflavin": "ACAD9 only (50-60%, Level B); others 0%",
            },
            {
                "class":      "Class 2 — N-Q Swap / NDUFA12 Paralog",
                "members":    "NDUFAF2, NDUFA12",
                "module":     "N-Q module interface; paralog-swap mechanism",
                "bnpage":     "N-Q sub-assembly intermediate",
                "riboflavin": "None (0%) — no FAD domain",
            },
            {
                "class":      "Class 3 — ND1-Module / Early P-Module",
                "members":    "NDUFAF4, NDUFAF3, TIMMDC1",
                "module":     "Early ND1-containing P-module sub-assembly",
                "bnpage":     "Early ND1 sub-assembly intermediate",
                "riboflavin": "None (0%)",
            },
        ],
    }


def get_definitions() -> dict:
    return {
        "concepts": [
            {
                "term":       "NDUFAF4 (C6orf66)",
                "definition": (
                    "NADH:Ubiquinone Oxidoreductase Complex Assembly Factor 4, also known as chromosome 6 "
                    "open reading frame 66. 130-amino acid, 14.5 kDa CI assembly chaperone. Forms an obligate "
                    "heterodimer with NDUFAF3 (C3orf60, 2q33.1). The NDUFAF4-NDUFAF3 heterodimer is the "
                    "earliest committed CI assembly complex, acting on ND1-containing P-module sub-assemblies. "
                    "Loss of NDUFAF4 causes secondary destabilization of NDUFAF3 protein."
                ),
            },
            {
                "term":       "NDUFAF4-NDUFAF3 Obligate Heterodimer",
                "definition": (
                    "NDUFAF4 and NDUFAF3 form a tight obligate heterodimer. Neither protein is stable without "
                    "the other. Loss of NDUFAF4 causes secondary destabilization of NDUFAF3 protein. This "
                    "heterodimer constitutes the earliest committed CI assembly sub-complex, analogous in "
                    "concept to the ACAD9-NDUFAF1 binary sub-complex in the MCIA tetramer, but acting on a "
                    "completely different module (ND1/P-module vs ND2/ND5-module). Both NDUFAF4 and NDUFAF3 "
                    "protein levels are reduced in either partner's deficiency — WES is mandatory to distinguish."
                ),
            },
            {
                "term":       "Mitochondrial Complex I Deficiency, Nuclear Type 20 (MC1DN20)",
                "definition": (
                    "OMIM #618245. Autosomal recessive mitochondrial complex I deficiency caused by biallelic "
                    "loss-of-function mutations in NDUFAF4. Phenotype: Leigh syndrome, isolated CI deficiency "
                    "5-20%, CII/CIII/CIV normal, lactic acidosis, Leigh MRI (basal ganglia/brainstem), "
                    "hypotonia, seizures. No peripheral neuropathy, no leukodystrophy, no hepatopathy. "
                    "No riboflavin response. Near-identical to MC1DN19 (NDUFAF3) — WES mandatory to distinguish."
                ),
            },
            {
                "term":       "BN-PAGE ND1-Module Class (Class 3) — NDUFAF4 vs NDUFAF3",
                "definition": (
                    "Both NDUFAF4 and NDUFAF3 deficiency produce Class 3 BN-PAGE CI assembly intermediates "
                    "(early ND1 sub-assembly stalled). The pattern is indistinguishable between the two. "
                    "This is because the two proteins form a single obligate heterodimer — loss of either "
                    "blocks the same step. NDUFAF4 deficiency also shows reduced NDUFAF3 protein on Western "
                    "blot, mimicking primary NDUFAF3 deficiency. WES with chromosomal locus (6q16.3 vs 2q33.1) "
                    "is the only reliable method to assign the diagnosis to NDUFAF4 vs NDUFAF3."
                ),
            },
            {
                "term":       "Chromosomal DDx: 6q16.3 (NDUFAF4) vs 2q33.1 (NDUFAF3)",
                "definition": (
                    "NDUFAF4 (6q16.3) and NDUFAF3 (2q33.1) are obligate heterodimer partners on different "
                    "chromosomes. Both cause near-identical Leigh syndrome with isolated CI deficiency, no "
                    "riboflavin response, and secondary reduction of the partner protein. WES with variant "
                    "calling at chromosomal locus level is the only method to definitively assign the causal gene."
                ),
            },
            {
                "term":       "Metformin Absolute Contraindication in CI Deficiency",
                "definition": (
                    "Biguanides (metformin) inhibit CI directly at the matrix-arm ND1 complex territory. "
                    "In NDUFAF4 deficiency (CI at 5-20%), any further CI inhibition is lethal. "
                    "Metformin is absolutely contraindicated in all nuclear-encoded CI deficiencies."
                ),
            },
            {
                "term":       "Succinate — CII Substrate (CI Bypass)",
                "definition": (
                    "Succinate enters the ETC at Complex II (succinate dehydrogenase), bypassing CI entirely. "
                    "In NDUFAF4 deficiency where CI is stalled, succinate supplementation allows some residual "
                    "OXPHOS activity via CII → CIII → CIV pathway. Level C evidence."
                ),
            },
        ],

        "thresholds": [
            {"parameter": "CI activity (% control)",    "threshold": "<20%",    "significance": "Diagnostic for CI deficiency; NDUFAF4 typically 5-20%"},
            {"parameter": "Lactate (plasma)",           "threshold": ">2.5 mmol/L", "significance": "Elevated in 92% of NDUFAF4 patients; >4 = crisis threshold"},
            {"parameter": "Lactate:pyruvate ratio",     "threshold": ">25",     "significance": "Elevated L:P ratio supports mitochondrial NADH block (ETC defect)"},
            {"parameter": "CSF lactate",                "threshold": ">2.5 mmol/L", "significance": "Leigh syndrome workup — elevated in CI deficiency with CNS involvement"},
            {"parameter": "Onset age (severe alleles)", "threshold": "<6 months", "significance": "Neonatal-onset: biallelic null or helix-breaking alleles; intermediate: 4-72 months"},
            {"parameter": "CI activity (riboflavin threshold)", "threshold": "0% response", "significance": "NDUFAF4 has NO FAD domain — zero riboflavin response expected"},
        ],

        "standards": [
            {"code": "ACMG/AMP 2015", "title": "Variant classification guidelines — pathogenicity criteria for NDUFAF4"},
            {"code": "MITOMAP", "title": "Mitochondrial disease gene/variant database"},
            {"code": "OMIM *611776", "title": "NDUFAF4 gene entry — C6orf66"},
            {"code": "OMIM #618245", "title": "Mitochondrial Complex I Deficiency, Nuclear Type 20 (MC1DN20)"},
            {"code": "REACTOME R-HSA-611105", "title": "Respiratory electron transport — CI assembly module"},
            {"code": "Helsinki Declaration", "title": "Ethical framework for human subject research"},
        ],

        "references": [
            {
                "id":        "saada_2009",
                "citation":  "Saada A et al. (2009) Mutations in NDUFAF3 (C3orf60), encoding a mitochondrial complex I assembly protein, cause fatal neonatal heart failure. Am J Hum Genet 84(6):718–27.",
                "relevance": "First report of NDUFAF3/NDUFAF4 heterodimer requirement; NDUFAF4 mutations identified in same study as obligate partner of NDUFAF3.",
            },
            {
                "id":        "guerrero_castillo_2017",
                "citation":  "Guerrero-Castillo S et al. (2017) The assembly pathway of mitochondrial respiratory chain complex I. Cell Metab 25(1):128–139.",
                "relevance": "Defines CI assembly intermediates including ND1-module (Class 3) where NDUFAF4-NDUFAF3 acts; maps NDUFAF4 within temporal assembly order.",
            },
            {
                "id":        "stroud_2016",
                "citation":  "Stroud DA et al. (2016) Accessory subunits are integral for assembly and function of human mitochondrial complex I. Nature 538:123–126.",
                "relevance": "BN-PAGE intermediate classification; confirms three assembly intermediate classes including Class 3 (ND1-module) where NDUFAF4 functions.",
            },
            {
                "id":        "fassone_2012",
                "citation":  "Fassone E & Rahman S (2012) Complex I deficiency: clinical features, biochemistry and molecular genetics. J Med Genet 49(9):578–90.",
                "relevance": "Comprehensive CI genetics review; frames NDUFAF4 within the CI assembly factor landscape alongside NDUFAF3.",
            },
            {
                "id":        "sazanov_2015",
                "citation":  "Sazanov LA (2015) A giant molecular proton pump: structure and mechanism of respiratory complex I. Nat Rev Mol Cell Biol 16(6):375–88.",
                "relevance": "CI structure review; contextualizes P-module (ND1/ND3/ND4L/ND6) where NDUFAF4-NDUFAF3 heterodimer operates.",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps({"overview": get_overview(), "breakdown": get_breakdown(), "definitions": get_definitions()}, indent=2, default=str))
