#!/usr/bin/env python3
"""NDUFAF3 — Mitochondrial Complex I Deficiency (Early ND1-Module Assembly / NDUFAF4 Obligate Heterodimer).

NDUFAF3 (C3orf60, Chromosome 3 Open Reading Frame 60) is a CI assembly factor
that forms an obligate heterodimer with NDUFAF4 (C6orf66). Together they constitute
the earliest committed CI assembly sub-complex, acting on the ND1-containing
proton-pump (P-module) sub-assembly intermediate — a fundamentally different
module from the MCIA complex (ACAD9–NDUFAF1–ECSIT–TMEM126B), which handles the
ND2/ND5 module.

  NDUFAF3 gene  OMIM *612911
  Disease       Mitochondrial Complex I Deficiency, Nuclear Type 19 (MC1DN19) OMIM #618241
  Inheritance   AR (autosomal recessive, biallelic)
  Chromosome    2q33.1

PATHOPHYSIOLOGY (NDUFAF3 / Early ND1-Module / NDUFAF3-NDUFAF4 Heterodimer):
  CI assembly proceeds through modular intermediates. NDUFAF3-NDUFAF4 heterodimer
  acts at the ND1-containing sub-assembly:
    1. NDUFAF3 + NDUFAF4 → obligate early heterodimer (tightest earliest interaction
       in ND1-module assembly, analogous to ACAD9-NDUFAF1 binary in MCIA complex).
    2. Heterodimer associates with ND1-containing IMM sub-assembly intermediate.
    3. Additional P-module subunits (ND3, ND4L, ND6) are incorporated.
    4. P-module sub-assembly merges with Q-module (NDUFS3-containing) and N-module
       (NDUFV1/NDUFV2-containing) to form mature CI holoenzyme.
  Loss of NDUFAF3:
    • NDUFAF4 loses its obligate partner → NDUFAF4 is destabilized.
    • NDUFAF3-NDUFAF4 heterodimer cannot assemble → early ND1-module assembly stalled.
    • Isolated CI deficiency (5–20% of control), CII/CIII/CIV normal.
    • BN-PAGE: stalled early ND1-containing sub-assembly intermediate — DISTINCT from
      MCIA complex (ND2/ND5 class) and N-Q module (NDUFAF2-NDUFA12 class).

NDUFAF3 UNIQUE FEATURES — EARLIEST COMMITTED CI ASSEMBLY HETERODIMER PARTNER:
  1. NDUFAF3-NDUFAF4 OBLIGATE HETERODIMER — UNIQUE EARLIEST ND1-MODULE COMPLEX:
     NDUFAF3 forms a tight obligate heterodimer with NDUFAF4 (C6orf66, 6q16.3).
     This is the EARLIEST committed CI assembly sub-complex, functioning upstream of
     MCIA tetramer (ACAD9-NDUFAF1-ECSIT-TMEM126B) which handles the ND2/ND5 module.
     NDUFAF3 deficiency causes destabilization of NDUFAF4 because the partner is absent.
  2. EARLY P-MODULE / ND1 TERRITORY — DISTINCT FROM MCIA (ND2/ND5):
     NDUFAF3-NDUFAF4 operates on the ND1 proton-pump sub-assembly, while the MCIA
     tetramer operates on ND2/ND5. BN-PAGE intermediate class differs: ND1-module
     class (NDUFAF3, NDUFAF4, TIMMDC1) vs ND2/ND5 class (MCIA members).
  3. NO FAD-BINDING / NO RIBOFLAVIN RESPONSE:
     NDUFAF3 has no FAD-binding domain. High-dose riboflavin does NOT rescue NDUFAF3
     deficiency. Critical DDx: ACAD9 (50-60% riboflavin responsive, Level B) vs
     NDUFAF3 (0% — no FAD domain, no riboflavin response).
  4. CHROMOSOME 2q33.1 — WES MANDATORY TO DISTINGUISH FROM NDUFAF4 (6q16.3):
     NDUFAF3 (2q33.1) and NDUFAF4 (6q16.3) are obligate heterodimer partners with
     near-identical CI deficiency phenotype. Only WES with chromosomal locus resolution
     distinguishes them: 2q33.1 = NDUFAF3, 6q16.3 = NDUFAF4.
  5. BN-PAGE ND1-MODULE INTERMEDIATE — THIRD CLASS DISTINCT FROM MCIA AND N-Q:
     Three distinct BN-PAGE CI assembly intermediate classes exist:
       Class 1 (MCIA / ND2-ND5): ACAD9, NDUFAF1, ECSIT, TMEM126B
       Class 2 (N-Q swap): NDUFAF2, NDUFA12
       Class 3 (ND1-module): NDUFAF3, NDUFAF4, TIMMDC1
     NDUFAF3 belongs exclusively to Class 3.

DISTINGUISHING FEATURES vs OTHER CI-LEIGH AND CI-ASSEMBLY GENES:
  vs NDUFAF4 (6q16.3): NEAR-IDENTICAL PHENOTYPE — obligate heterodimer partner.
    WES MANDATORY. Both AR, both Leigh, both isolated CI 5-20%, both no riboflavin.
    Only chromosomal locus distinguishes: 2q33.1 (NDUFAF3) vs 6q16.3 (NDUFAF4).
    NDUFAF3 loss also destabilizes NDUFAF4 protein — protein level may be low for both.
  vs ACAD9 (3q21.3): MCIA complex member (ND2/ND5 module) vs NDUFAF3 (ND1 module).
    ACAD9 riboflavin-responsive (50-60%, Level B); NDUFAF3 NO riboflavin response.
    ACAD9 has bimodal severity (exercise-intolerance dominant pArg518His); NDUFAF3 pure Leigh.
    BN-PAGE: ND2/ND5 class (ACAD9) vs ND1-module class (NDUFAF3).
  vs NDUFAF2 (5q12.1): N-Q module class (NDUFAF2 is an NDUFA12 paralog/swap).
    NDUFAF2 stalls at N-Q interface; NDUFAF3 stalls at early ND1 module.
    Both: no riboflavin, Leigh, isolated CI 5-20%. BN-PAGE class distinguishes.
  vs TIMMDC1 (same ND1-module class, 3q25.1): Both ND1-class, but TIMMDC1 has
    severe cardiomyopathy (>80%) while NDUFAF3 HCM is low (15-25%).
  vs NDUFS1 (2q33.3, SAME CHROMOSOME ARM):
    NDUFS1 (2q33.3) is at the same chromosomal arm as NDUFAF3 (2q33.1) — CRITICAL
    chromosomal DDx. NDUFS1 causes peripheral neuropathy (50%); NDUFAF3 NO neuropathy.
  vs NDUFV1 (11q13.2): NDUFV1 causes leukodystrophy (40-50%); NDUFAF3 NO leukodystrophy.
  vs POLG/DGUOK: Both cause hepatopathy; NDUFAF3 NO hepatopathy — key exclusion.
"""

import random
import math

SEED = 683
rng  = random.Random(SEED)

GENE         = "NDUFAF3"
OMIM_GENE    = "612911"
OMIM_DISEASE = "618241"
DISEASE_NAME = "Mitochondrial Complex I Deficiency, Nuclear Type 19 (MC1DN19)"
CHROMOSOME   = "2q33.1"
INHERITANCE  = "AR (biallelic)"
N_PATIENTS   = 40

# ─── Variants ────────────────────────────────────────────────────────────────
VARIANTS = [
    {
        "hgvs_c":    "c.74G>A",
        "hgvs_p":    "p.Arg25Gln",
        "domain":    "NDUFAF4-contact surface",
        "mechanism": "Disrupts obligate NDUFAF3-NDUFAF4 heterodimer interface; destabilizes both partners",
        "severity":  "intermediate",
        "ci_pct_range": (8, 15),
        "notes":     "Most common pathogenic variant; NDUFAF4 protein level secondarily reduced",
    },
    {
        "hgvs_c":    "c.224T>C",
        "hgvs_p":    "p.Leu75Pro",
        "domain":    "Alpha-helix 3 (core fold)",
        "mechanism": "Helix-breaking proline substitution causes alpha-helix collapse; full protein destabilization",
        "severity":  "severe",
        "ci_pct_range": (5, 10),
        "notes":     "Severe neonatal presentation; proline incompatible with alpha-helical fold",
    },
    {
        "hgvs_c":    "c.157G>A",
        "hgvs_p":    "p.Ala53Thr",
        "domain":    "Hydrophobic core",
        "mechanism": "Hydrophobic-to-polar substitution disrupts core packing; partial destabilization",
        "severity":  "intermediate",
        "ci_pct_range": (10, 18),
        "notes":     "Consanguineous families; some residual CI activity; intermediate phenotype",
    },
    {
        "hgvs_c":    "c.326G>A",
        "hgvs_p":    "p.Trp109Ter",
        "domain":    "C-terminal NDUFAF4-binding region",
        "mechanism": "Near-end nonsense — truncates last 31 aa; NDUFAF4 docking surface lost",
        "severity":  "severe",
        "ci_pct_range": (5, 9),
        "notes":     "Null allele; consanguineous; severe neonatal-onset",
    },
    {
        "hgvs_c":    "c.IVS3+1G>A",
        "hgvs_p":    "p.?",
        "domain":    "Splice donor intron 3",
        "mechanism": "Canonical splice donor disruption; leaky splicing → partial NDUFAF3 residual",
        "severity":  "moderate",
        "ci_pct_range": (12, 20),
        "notes":     "Partial CI residual (12-20%); clinical syndrome milder; school-age onset possible",
    },
]

REGIONS     = ["European",  "Middle Eastern", "North African", "South Asian", "East Asian", "Latin American"]
REGION_W    = [0.30,         0.25,              0.18,            0.14,          0.08,          0.05]
OUTCOMES    = ["alive, supported", "alive, institutionalized", "deceased (respiratory)", "deceased (cardiac)"]
OUTCOME_W   = [0.42, 0.30, 0.18, 0.10]


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
        "Leigh_MRI":              round(sum(p["leigh_mri"] for p in PATIENTS) / N_PATIENTS * 100),
        "Lactic_acidosis":        round(sum(p["lactic_acidosis"] for p in PATIENTS) / N_PATIENTS * 100),
        "Hypotonia":              round(sum(p["hypotonia"] for p in PATIENTS) / N_PATIENTS * 100),
        "Developmental_regression":round(sum(p["dev_regression"] for p in PATIENTS) / N_PATIENTS * 100),
        "Seizures":               round(sum(p["seizures"] for p in PATIENTS) / N_PATIENTS * 100),
        "HCM":                    round(sum(p["hcm"] for p in PATIENTS) / N_PATIENTS * 100),
        "Respiratory_failure":    round(sum(p["respiratory_fail"] for p in PATIENTS) / N_PATIENTS * 100),
        "Riboflavin_responder":   0,   # HARD 0 — no FAD domain
        "Peripheral_neuropathy":  0,   # HARD 0 — critical DDx vs NDUFS1
        "Leukodystrophy":         0,   # HARD 0 — critical DDx vs NDUFV1
        "Hepatopathy":            0,   # HARD 0 — critical DDx vs POLG/DGUOK
        "Olfactory_bulb_lesions": 0,   # HARD 0 — critical DDx vs NDUFS4
    }

    return {
        "gene":            GENE,
        "gene_full_name":  "NADH:Ubiquinone Oxidoreductase Complex Assembly Factor 3",
        "also_known_as":   "C3orf60 (Chromosome 3 Open Reading Frame 60)",
        "omim_gene":       OMIM_GENE,
        "omim_disease":    OMIM_DISEASE,
        "disease_name":    DISEASE_NAME,
        "chromosome":      CHROMOSOME,
        "inheritance":     INHERITANCE,
        "cohort_n":        N_PATIENTS,
        "cohort_seed":     SEED,

        "protein": {
            "size_aa":  139,
            "size_kda": 16.1,
            "fold":     "Beta-sheet / alpha-helix mixed fold; no canonical enzyme domain",
            "module":   "Early ND1-module P-module CI assembly; NDUFAF3-NDUFAF4 early heterodimer complex",
            "function": (
                "Forms obligate heterodimer with NDUFAF4 (C6orf66). The NDUFAF3-NDUFAF4 heterodimer "
                "constitutes the earliest committed CI assembly sub-complex, acting on the ND1-containing "
                "proton-pump (P-module) sub-assembly intermediate upstream of the MCIA tetramer (ND2/ND5 module)."
            ),
        },

        "key_pathway_note": (
            "NDUFAF3 (C3orf60, 2q33.1) forms an obligate heterodimer with NDUFAF4 (C6orf66, 6q16.3). "
            "This heterodimer is the EARLIEST committed CI assembly complex, acting on ND1-containing "
            "sub-assemblies (P-module territory). Loss of NDUFAF3 destabilizes NDUFAF4 protein levels. "
            "BN-PAGE shows stalled early ND1-class intermediates — a DISTINCT pattern from MCIA complex "
            "(ND2/ND5 class: ACAD9, NDUFAF1, ECSIT, TMEM126B) and N-Q module (NDUFAF2, NDUFA12). "
            "No FAD-binding domain → NO riboflavin response (unlike ACAD9, 50-60% Level B). "
            "CRITICAL chromosomal DDx: NDUFAF3 (2q33.1) vs NDUFS1 (2q33.3) — SAME chromosome arm, "
            "different genes — NDUFS1 causes 50% peripheral neuropathy, NDUFAF3 NONE."
        ),

        "biochemical_fingerprint": {
            "Complex_I":             "5–20% of control (SEVERE)",
            "Complex_II":            "Normal (100%)",
            "Complex_III":           "Normal (100%)",
            "Complex_IV":            "Normal (100%)",
            "Complex_V":             "Normal (100%)",
            "Pattern":               "ISOLATED CI deficiency — identical to all NDUFAF-class assembly factors",
            "Riboflavin_response":   "NONE (0%) — no FAD-binding domain in NDUFAF3",
            "BN-PAGE_class":         "ND1-module (Class 3): stalled ND1 sub-assembly intermediate",
            "NDUFAF4_protein_level": "Secondarily reduced — NDUFAF3 loss destabilizes obligate partner",
        },

        "feature_frequencies_pct": ff,

        "ndufaf3_ndufaf4_module_summary": {
            "heterodimer":       "NDUFAF3 (2q33.1) + NDUFAF4 (6q16.3)",
            "module_class":      "Early ND1-module / P-module CI assembly",
            "assembly_position": "Earliest committed CI assembly step (upstream of MCIA tetramer)",
            "distinct_from_mcia": (
                "MCIA tetramer (ACAD9-NDUFAF1-ECSIT-TMEM126B) acts on ND2/ND5 module. "
                "NDUFAF3-NDUFAF4 acts on ND1 module. Completely separate assembly complexes."
            ),
            "ndufaf3_loss_effect": (
                "NDUFAF4 protein secondarily destabilized (loses obligate partner). "
                "Early ND1 sub-assembly intermediate accumulates. CI holoenzyme absent."
            ),
        },

        "key_ddx": [
            {
                "feature":     "NDUFAF3 (2q33.1) vs NDUFAF4 (6q16.3) — NEAR-IDENTICAL PHENOTYPE",
                "significance": (
                    "NDUFAF3 and NDUFAF4 are obligate heterodimer partners with nearly identical CI deficiency "
                    "phenotype (both Leigh, both isolated CI 5-20%, both no riboflavin response, both AR). "
                    "ONLY chromosomal locus distinguishes them on WES: 2q33.1 (NDUFAF3) vs 6q16.3 (NDUFAF4). "
                    "Both genes must always be sequenced when either is suspected."
                ),
                "target_gene": "NDUFAF4",
            },
            {
                "feature":     "NDUFAF3 (2q33.1) vs NDUFS1 (2q33.3) — SAME CHROMOSOME ARM",
                "significance": (
                    "NDUFAF3 (2q33.1) and NDUFS1 (2q33.3) map to the same chromosomal arm. "
                    "NDUFS1 causes peripheral neuropathy (50%) — a hallmark absent in NDUFAF3 deficiency. "
                    "Key clinical DDx: no peripheral neuropathy → favour NDUFAF3 over NDUFS1."
                ),
                "target_gene": "NDUFS1",
            },
            {
                "feature":     "NDUFAF3 vs ACAD9 — Riboflavin Response Key Distinguisher",
                "significance": (
                    "ACAD9 (MCIA complex, 3q21.3) is riboflavin-responsive (50-60%, Level B) due to FAD-binding domain. "
                    "NDUFAF3 has no FAD domain — riboflavin response 0%. If CI deficiency responds to riboflavin: "
                    "diagnose ACAD9. No response favours NDUFAF3 / NDUFAF4 / NDUFAF2 / other assembly factors."
                ),
                "target_gene": "ACAD9",
            },
            {
                "feature":     "NDUFAF3 vs NDUFV1 — NO Leukodystrophy",
                "significance": (
                    "NDUFV1 (CI subunit, 11q13.2) causes leukodystrophy in 40-50% of patients. "
                    "NDUFAF3 deficiency: leukodystrophy 0%. If leukodystrophy present on MRI: "
                    "sequence NDUFV1 before NDUFAF3."
                ),
                "target_gene": "NDUFV1",
            },
            {
                "feature":     "NDUFAF3 vs POLG/DGUOK — NO Hepatopathy",
                "significance": (
                    "POLG (15q26.1) and DGUOK (2p13.1) cause hepatopathy (liver failure, hepatic mtDNA depletion). "
                    "NDUFAF3 deficiency: hepatopathy 0%. Hepatopathy on presentation is a strong DDx pointer "
                    "away from NDUFAF3."
                ),
                "target_gene": "POLG / DGUOK",
            },
        ],

        "clinical_alerts": [
            "🔴 METFORMIN — ABSOLUTE CONTRAINDICATION: Direct ND1 complex territory inhibitor. Biguanides inhibit CI at the ND1-containing Q/P module. NDUFAF3 patients have 5-20% CI — metformin is lethal.",
            "🔴 VALPROATE (VPA) — ABSOLUTE CONTRAINDICATION: Triple mechanism: CoA sequestration, POLG inhibition, MT-ND subunit expression suppression.",
            "🔴 LINEZOLID — ABSOLUTE CONTRAINDICATION: 23S rRNA inhibition blocks MT-ND1–6 subunit synthesis — shuts down all 7 mtDNA-encoded CI subunits.",
            "🔴 CHLORAMPHENICOL — ABSOLUTE CONTRAINDICATION: Same mitoribosomal mechanism as linezolid.",
            "🔴 KETOGENIC DIET — CONTRAINDICATED: CI is absent/minimal; NADH cannot be reoxidised. Beta-oxidation floods matrix with NADH with no ETC outlet. Metabolic crisis.",
            "🟠 PROPOFOL — AVOID (PRIS risk): Propofol infusion syndrome via CIV inhibition + fatty acid oxidation uncoupling. Use sevoflurane for anaesthesia.",
            "🟡 PHENOBARBITAL — HIGH CAUTION: Secondary CI inhibitor effect; use LEV as first-choice AED (renal, no mito toxicity).",
            "🟢 SUCCINATE — Level C: CII substrate bypasses NDUFAF3-stalled CI entirely; bypasses ND1 module block.",
            "🟢 THIAMINE B1 — Level C MANDATORY EMPIRIC: Exclude SLC19A3 (thiamine transporter) and BTD (biotin-thiamine-responsive basal ganglia disease) before confirming NDUFAF3.",
            "🟢 BIOTIN — Level C MANDATORY EMPIRIC: Exclude BTD (biotinidase deficiency) and HLCS before CI gene sequencing.",
            "🔵 NDUFAF3 (2q33.1) vs NDUFS1 (2q33.3) — SAME ARM: Critical DDx. NDUFS1 causes peripheral neuropathy (50%); NDUFAF3 causes NONE.",
            "🔵 NDUFAF3 vs NDUFAF4 — WES MANDATORY: Near-identical phenotype; chromosomal locus 2q33.1 vs 6q16.3 only distinguisher. Always sequence both.",
        ],
    }


def get_breakdown() -> dict:
    # Mutation distribution
    var_names = [v["hgvs_p"] for v in VARIANTS]
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
        if o <= 6:   onset_bins["0–6 m"]   += 1
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
        {"drug": "Propofol",        "mechanism": "PRIS: CIV inhibition + fatty acid oxidation uncoupling",                            "class": "AVOID"},
        {"drug": "Phenobarbital",   "mechanism": "Secondary CI inhibitor; use LEV instead",                                            "class": "HIGH CAUTION"},
    ]

    treatment_protocols = [
        {"agent": "Succinate",          "dose": "500–2000 mg/day oral", "evidence": "Level C", "rationale": "CII substrate — bypasses stalled ND1/CI entirely; direct CII → III electron flow"},
        {"agent": "CoQ10 / Ubiquinol",  "dose": "10–30 mg/kg/day",     "evidence": "Level C", "rationale": "Downstream CI support; improves overall ETC electron flow"},
        {"agent": "Thiamine (B1)",       "dose": "100–600 mg/day",      "evidence": "Level C — MANDATORY EMPIRIC", "rationale": "Exclude SLC19A3 / BTD before confirming NDUFAF3"},
        {"agent": "Biotin",             "dose": "10–40 mg/day",         "evidence": "Level C — MANDATORY EMPIRIC", "rationale": "Exclude BTD / HLCS before CI gene panel"},
        {"agent": "Carnitine (L-Carn)", "dose": "50–100 mg/kg/day",     "evidence": "Level C", "rationale": "Secondary carnitine deficiency possible; replete if low"},
        {"agent": "Ribose",             "dose": "5 g TID",              "evidence": "Level C (limited)", "rationale": "Nucleotide pool support; experimental in CI assembly defects"},
        {"agent": "IV Dextrose (GIR 6–8)", "dose": "GIR 6–8 mg/kg/min", "evidence": "Consensus", "rationale": "NEVER fast — CI-absent patients cannot sustain fasting gluconeogenesis"},
        {"agent": "LEV (Levetiracetam)","dose": "20–60 mg/kg/day",      "evidence": "Consensus AED", "rationale": "Preferred AED — renal excretion, no mitochondrial toxicity"},
        {"agent": "Sevoflurane",        "dose": "Standard GA",          "evidence": "Consensus", "rationale": "Preferred volatile agent — NOT propofol (PRIS risk)"},
    ]

    nd1_module_steps = [
        {
            "step":  1,
            "event": "NDUFAF3 + NDUFAF4 → Obligate Early Heterodimer",
            "status_in_ndufaf3_deficiency": "DISRUPTED — root cause",
            "note": "NDUFAF3 loss prevents heterodimer formation; NDUFAF4 is secondarily destabilized and degraded.",
        },
        {
            "step":  2,
            "event": "NDUFAF3-NDUFAF4 Heterodimer Associates with ND1-Containing Sub-Assembly",
            "status_in_ndufaf3_deficiency": "STALLED — early ND1 intermediate accumulates",
            "note": "ND1 sub-assembly intermediate (Class 3 BN-PAGE band) accumulates without NDUFAF3-NDUFAF4 chaperone.",
        },
        {
            "step":  3,
            "event": "ND3, ND4L, ND6 Incorporation into P-Module Sub-Assembly",
            "status_in_ndufaf3_deficiency": "BLOCKED — upstream stall prevents this step",
            "note": "P-module cannot be completed; additional proton-pump subunits cannot be incorporated.",
        },
        {
            "step":  4,
            "event": "P-Module Merges with Q-Module and N-Module → CI Holoenzyme",
            "status_in_ndufaf3_deficiency": "ABSENT — CI holoenzyme not formed",
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
                "class":    "Class 1 — MCIA / ND2-ND5 Module",
                "members":  "ACAD9, NDUFAF1, ECSIT, TMEM126B",
                "module":   "ND2/ND5 membrane arm sub-assembly",
                "bnpage":   "ND2-ND5 sub-assembly intermediate (absent CI holoenzyme)",
                "riboflavin": "ACAD9 only (50-60%, Level B); others 0%",
            },
            {
                "class":    "Class 2 — N-Q Swap / NDUFA12 Paralog",
                "members":  "NDUFAF2, NDUFA12",
                "module":   "N-Q module interface; paralog-swap mechanism",
                "bnpage":   "N-Q sub-assembly intermediate",
                "riboflavin": "None (0%) — no FAD domain",
            },
            {
                "class":    "Class 3 — ND1-Module / Early P-Module",
                "members":  "NDUFAF3, NDUFAF4, TIMMDC1",
                "module":   "Early ND1-containing P-module sub-assembly",
                "bnpage":   "Early ND1 sub-assembly intermediate",
                "riboflavin": "None (0%)",
            },
        ],
    }


def get_definitions() -> dict:
    return {
        "concepts": [
            {
                "term":       "NDUFAF3 (C3orf60)",
                "definition": (
                    "NADH:Ubiquinone Oxidoreductase Complex Assembly Factor 3, also known as chromosome 3 "
                    "open reading frame 60. 139-amino acid, 16.1 kDa CI assembly chaperone. Forms an obligate "
                    "heterodimer with NDUFAF4 (C6orf66, 6q16.3). The NDUFAF3-NDUFAF4 heterodimer is the "
                    "earliest committed CI assembly complex, acting on ND1-containing P-module sub-assemblies."
                ),
            },
            {
                "term":       "NDUFAF3-NDUFAF4 Obligate Heterodimer",
                "definition": (
                    "NDUFAF3 and NDUFAF4 form a tight obligate heterodimer. Neither protein is stable without "
                    "the other. Loss of NDUFAF3 causes secondary destabilization of NDUFAF4 protein. This "
                    "heterodimer constitutes the earliest committed CI assembly sub-complex, analogous in "
                    "concept to the ACAD9-NDUFAF1 binary sub-complex in the MCIA tetramer, but acting on a "
                    "completely different module (ND1/P-module vs ND2/ND5-module)."
                ),
            },
            {
                "term":       "Early ND1-Module / P-Module CI Assembly",
                "definition": (
                    "The initial step of CI membrane arm assembly involves building the ND1-containing "
                    "proton-pump (P-module) sub-assembly. NDUFAF3-NDUFAF4 heterodimer chaperones this step, "
                    "distinct from the MCIA tetramer which handles the ND2/ND5 module. TIMMDC1 (3q25.1) is "
                    "another assembly factor in this ND1-class. BN-PAGE from NDUFAF3 patients shows "
                    "Class 3 intermediates (early ND1 sub-assembly)."
                ),
            },
            {
                "term":       "Mitochondrial Complex I Deficiency, Nuclear Type 19 (MC1DN19)",
                "definition": (
                    "OMIM #618241. Autosomal recessive mitochondrial complex I deficiency caused by biallelic "
                    "loss-of-function mutations in NDUFAF3. Phenotype: Leigh syndrome, isolated CI deficiency "
                    "5-20%, CII/CIII/CIV normal, lactic acidosis, Leigh MRI (basal ganglia/brainstem), "
                    "hypotonia, seizures. No peripheral neuropathy, no leukodystrophy, no hepatopathy. "
                    "No riboflavin response."
                ),
            },
            {
                "term":       "BN-PAGE ND1-Module Class (Class 3) vs MCIA Class (Class 1) vs N-Q Class (Class 2)",
                "definition": (
                    "Blue-native PAGE of CI assembly intermediates identifies distinct accumulating sub-assemblies: "
                    "Class 1 (MCIA — ACAD9, NDUFAF1, ECSIT, TMEM126B): ND2/ND5 sub-assembly intermediate. "
                    "Class 2 (N-Q swap — NDUFAF2, NDUFA12): N-Q interface intermediate. "
                    "Class 3 (ND1-module — NDUFAF3, NDUFAF4, TIMMDC1): early ND1 sub-assembly. "
                    "NDUFAF3 deficiency shows Class 3 pattern. This cannot be determined from enzymatic "
                    "assay alone — BN-PAGE required for assembly-class determination."
                ),
            },
            {
                "term":       "Chromosomal Arm DDx: 2q33.1 (NDUFAF3) vs 2q33.3 (NDUFS1)",
                "definition": (
                    "NDUFAF3 (2q33.1) and NDUFS1 (2q33.3) are located on the same chromosomal arm at 2q33. "
                    "NDUFS1 (CI structural subunit, ferredoxin reductase domain, N-module) causes peripheral "
                    "neuropathy in ~50% of patients — a feature entirely absent in NDUFAF3 deficiency. "
                    "Clinical DDx: no peripheral neuropathy favours NDUFAF3; peripheral neuropathy favours NDUFS1."
                ),
            },
            {
                "term":       "Metformin Absolute Contraindication in CI Deficiency",
                "definition": (
                    "Biguanides (metformin) inhibit CI directly at the matrix-arm ND1 complex territory. "
                    "In NDUFAF3 deficiency (CI at 5-20%), any further CI inhibition is lethal. "
                    "Metformin is absolutely contraindicated in all nuclear-encoded CI deficiencies."
                ),
            },
            {
                "term":       "Succinate — CII Substrate (CI Bypass)",
                "definition": (
                    "Succinate enters the ETC at Complex II (succinate dehydrogenase), bypassing CI entirely. "
                    "In NDUFAF3 deficiency where CI is stalled, succinate supplementation allows some residual "
                    "OXPHOS activity via CII → CIII → CIV pathway. Level C evidence."
                ),
            },
        ],

        "thresholds": [
            {"parameter": "CI activity (% control)",    "threshold": "<20%",    "significance": "Diagnostic for CI deficiency; NDUFAF3 typically 5-20%"},
            {"parameter": "Lactate (plasma)",           "threshold": ">2.5 mmol/L", "significance": "Elevated in 92% of NDUFAF3 patients; >4 = crisis threshold"},
            {"parameter": "Lactate:pyruvate ratio",     "threshold": ">25",     "significance": "Elevated L:P ratio supports mitochondrial NADH block (ETC defect)"},
            {"parameter": "CSF lactate",                "threshold": ">2.5 mmol/L", "significance": "Leigh syndrome workup — elevated in CI deficiency with CNS involvement"},
            {"parameter": "Onset age (severe alleles)", "threshold": "<6 months", "significance": "Neonatal-onset: biallelic null or TM1-TM2 severe alleles; intermediate: 4-72 months"},
            {"parameter": "CI activity (riboflavin threshold)", "threshold": "0% response", "significance": "NDUFAF3 has NO FAD domain — zero riboflavin response expected"},
        ],

        "standards": [
            {"code": "ACMG/AMP 2015", "title": "Variant classification guidelines — pathogenicity criteria for NDUFAF3"},
            {"code": "MITOMAP", "title": "Mitochondrial disease gene/variant database"},
            {"code": "OMIM *612911", "title": "NDUFAF3 gene entry — C3orf60"},
            {"code": "OMIM #618241", "title": "Mitochondrial Complex I Deficiency, Nuclear Type 19 (MC1DN19)"},
            {"code": "REACTOME R-HSA-611105", "title": "Respiratory electron transport — CI assembly module"},
            {"code": "Helsinki Declaration", "title": "Ethical framework for human subject research"},
        ],

        "references": [
            {
                "id":        "saada_2009",
                "citation":  "Saada A et al. (2009) Mutations in NDUFAF3 (C3orf60), encoding a mitochondrial complex I assembly protein, cause fatal neonatal heart failure. Am J Hum Genet 84(6):718–27.",
                "relevance": "First report of NDUFAF3 mutations causing fatal neonatal CI deficiency; established NDUFAF3-NDUFAF4 heterodimer requirement.",
            },
            {
                "id":        "guerrero_castillo_2017",
                "citation":  "Guerrero-Castillo S et al. (2017) The assembly pathway of mitochondrial respiratory chain complex I. Cell Metab 25(1):128–139.",
                "relevance": "Defines CI assembly intermediates including ND1-module (Class 3) where NDUFAF3-NDUFAF4 acts; maps NDUFAF3 within the temporal assembly order.",
            },
            {
                "id":        "sanchez_caballero_2016",
                "citation":  "Sánchez-Caballero L et al. (2016) Mutations in complex I assembly factor TMEM126B result in muscle weakness and isolated complex I deficiency. Am J Hum Genet 99(1):208–216.",
                "relevance": "Comparative CI assembly factor data; contextualizes NDUFAF3-NDUFAF4 ND1 class vs MCIA tetramer ND2/ND5 class.",
            },
            {
                "id":        "stroud_2016",
                "citation":  "Stroud DA et al. (2016) Accessory subunits are integral for assembly and function of human mitochondrial complex I. Nature 538:123–126.",
                "relevance": "BN-PAGE intermediate classification; confirms three assembly intermediate classes including Class 3 (ND1-module).",
            },
            {
                "id":        "fassone_2012",
                "citation":  "Fassone E & Rahman S (2012) Complex I deficiency: clinical features, biochemistry and molecular genetics. J Med Genet 49(9):578–90.",
                "relevance": "Comprehensive CI genetics review; frames NDUFAF3 within the CI assembly factor landscape.",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps({"overview": get_overview(), "breakdown": get_breakdown(), "definitions": get_definitions()}, indent=2, default=str))
