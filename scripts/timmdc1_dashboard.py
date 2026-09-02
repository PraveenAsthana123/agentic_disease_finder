#!/usr/bin/env python3
"""TIMMDC1 — Mitochondrial Complex I Deficiency (ND1-Module CI Assembly Factor / Integral IMM).

TIMMDC1 (Translocase of Inner Mitochondrial Membrane Domain Containing 1) is a CI
assembly factor that acts within the ND1-containing proton-pump (P-module) sub-assembly —
the same Class 3 assembly intermediate as NDUFAF3, NDUFAF4, and NDUFAF5.

TIMMDC1 is UNIQUE among all Class-3 ND1-module factors: it is the ONLY integral inner
mitochondrial membrane (IMM) protein in this class. NDUFAF3, NDUFAF4, and NDUFAF5 are
all soluble matrix proteins without TM helices. TIMMDC1's two TM helices anchor it to the
IMM and position its matrix-exposed loop domain to contact the ND1-containing sub-assembly.

  TIMMDC1 gene  OMIM *615530
  Disease       Mitochondrial Complex I Deficiency (OMIM #256000)
  Inheritance   AR (autosomal recessive, biallelic)
  Chromosome    3q25.1

Reference: Stroud DA et al. (2016) Accessory subunits are integral for assembly and function
of human mitochondrial complex I. Nature 538:123–126. (CI assembly intermediate class mapping)
Reference: Guerrero-Castillo S et al. (2017) The assembly pathway of mitochondrial respiratory
chain complex I. Cell Metab 25(1):128–139. (placed TIMMDC1 in ND1-module Class-3 assembly)

PATHOPHYSIOLOGY (TIMMDC1 / ND1-Module / Class-3 / Integral IMM Anchor):
  CI assembly proceeds through modular intermediates. TIMMDC1 acts on the
  ND1-containing sub-assembly (P-module territory) as an integral-IMM scaffold:
    1. TIMMDC1 is anchored to the IMM via its 2 TM helices.
    2. Its matrix-exposed loop domain contacts the ND1-containing early P-module
       sub-assembly intermediate.
    3. TIMMDC1 facilitates P-module sub-assembly maturation at the IMM surface.
    4. Loss of TIMMDC1 stalls the ND1-containing intermediate (Class 3 BN-PAGE band).
    5. Isolated CI deficiency (5–20%); CII/CIII/CIV normal.
  Key HCM mechanism: ND1-module class-3 factors with IMM-integrated scaffold support
  are critical for cardiomyocyte CI biogenesis; the IMM-anchored role of TIMMDC1 makes
  it disproportionately important in cardiac mitochondria → HCM rate >80%.

TIMMDC1 UNIQUE FEATURES vs OTHER ND1-MODULE CLASS-3 FACTORS:
  1. ONLY INTEGRAL IMM PROTEIN IN CLASS 3:
     NDUFAF3 (2q33.1), NDUFAF4 (6q16.3), NDUFAF5 (20p12.1) are all SOLUBLE matrix
     proteins (zero TM helices). TIMMDC1 (3q25.1) is the SOLE Class-3 member with
     TM helices — anchored to the IMM as an integral membrane scaffold.
  2. HIGHEST HCM RATE IN CLASS 3 (>80%):
     TIMMDC1 deficiency causes hypertrophic cardiomyopathy in >80% of patients.
     NDUFAF3 and NDUFAF4: 15–25%. NDUFAF5: <20%. This is the most important single
     clinical distinguishing feature within the Class-3 group.
  3. CLASS 3 BN-PAGE — SAME AS NDUFAF3/NDUFAF4/NDUFAF5:
     TIMMDC1, NDUFAF3, NDUFAF4, and NDUFAF5 all produce stalled early ND1 sub-assembly
     intermediates on BN-PAGE. WES chromosomal locus (3q25.1) is mandatory to confirm.
  4. NO FAD-BINDING / NO RIBOFLAVIN RESPONSE:
     TIMMDC1 has no FAD-binding domain. Riboflavin does NOT rescue TIMMDC1 deficiency.
     Critical DDx: ACAD9 (50-60% riboflavin-responsive) vs TIMMDC1 (0%).
  5. TM-HELIX ARCHITECTURE — DISTINCT PROTEIN TOPOLOGY:
     The 2-TM topology of TIMMDC1 means mutations in TM helices tend to disrupt both
     membrane anchoring AND the matrix-side ND1-sub-assembly contact loop — compound
     mechanism unlike missense disruptions in the soluble NDUFAF3/4/5 factors.

DISTINGUISHING FEATURES vs OTHER CI GENES:
  vs NDUFAF3 (2q33.1): Both Class 3 ND1-module. Near-identical BN-PAGE.
    KEY DIFFERENCE: TIMMDC1 HCM >80% vs NDUFAF3 HCM 15-25%.
    TIMMDC1: integral IMM (2 TM helices). NDUFAF3: soluble matrix.
    NDUFAF3 deficiency destabilizes NDUFAF4 partner; TIMMDC1 does NOT.
  vs NDUFAF4 (6q16.3): Both Class 3 ND1-module. Near-identical BN-PAGE.
    KEY DIFFERENCE: TIMMDC1 HCM >80% vs NDUFAF4 HCM 15-25%.
    TIMMDC1: integral IMM. NDUFAF4: soluble matrix. WES mandatory: 3q25.1 vs 6q16.3.
  vs NDUFAF5 (20p12.1): Both Class 3 ND1-module. Near-identical BN-PAGE.
    KEY DIFFERENCE: TIMMDC1 HCM >80% vs NDUFAF5 HCM <20%.
    TIMMDC1: integral IMM. NDUFAF5: soluble matrix. WES mandatory: 3q25.1 vs 20p12.1.
  vs ACAD9 (3q21.3): Same chromosome 3 — CRITICAL. ACAD9 (3q21.3) vs TIMMDC1 (3q25.1).
    MCIA class (ND2/ND5) vs ND1-module class (TIMMDC1). ACAD9 riboflavin-responsive;
    TIMMDC1 0% response. ACAD9 HCM 55-65%; TIMMDC1 HCM >80% — overlapping HCM rates
    make riboflavin response the key clinical distinguisher on same chromosome 3.
  vs NDUFS1 (2q33.3): NDUFS1 peripheral neuropathy 50%; TIMMDC1 NONE.
  vs NDUFV1 (11q13.2): NDUFV1 leukodystrophy 40-50%; TIMMDC1 NONE.
  vs POLG/DGUOK: Both cause hepatopathy; TIMMDC1 NO hepatopathy.
"""

import random
import math

SEED = 689
rng  = random.Random(SEED)

GENE         = "TIMMDC1"
OMIM_GENE    = "615530"
OMIM_DISEASE = "256000"
DISEASE_NAME = "Mitochondrial Complex I Deficiency (CI Deficiency; Leigh Syndrome spectrum)"
CHROMOSOME   = "3q25.1"
INHERITANCE  = "AR (biallelic)"
N_PATIENTS   = 40

# ─── Variants ────────────────────────────────────────────────────────────────
VARIANTS = [
    {
        "hgvs_c":    "c.244G>C",
        "hgvs_p":    "p.Gly82Arg",
        "domain":    "Transmembrane helix 1 (TM1) — hydrophobic core",
        "mechanism": "Glycine-to-arginine disrupts TM1 alpha-helix hydrophobic packing; prevents IMM anchoring",
        "severity":  "severe",
        "ci_pct_range": (5, 12),
        "notes":     "Severe neonatal onset; TM1 glycine is critical for helix flexibility and IMM insertion",
    },
    {
        "hgvs_c":    "c.446T>C",
        "hgvs_p":    "p.Leu149Pro",
        "domain":    "Transmembrane helix 2 (TM2) — helix-breaking proline",
        "mechanism": "Proline introduces rigid kink in TM2 alpha-helix; IMM anchoring and ND1-contact loop mispositioned",
        "severity":  "severe",
        "ci_pct_range": (5, 11),
        "notes":     "Lethal infantile onset; proline incompatible with TM2 alpha-helical fold",
    },
    {
        "hgvs_c":    "c.602G>A",
        "hgvs_p":    "p.Arg201Gln",
        "domain":    "Matrix-exposed loop — ND1 sub-assembly contact surface",
        "mechanism": "Loss of R201 positive charge disrupts electrostatic contact with ND1-containing sub-assembly",
        "severity":  "intermediate",
        "ci_pct_range": (9, 18),
        "notes":     "Partial CI residual; intermediate phenotype; most commonly found in Middle Eastern families",
    },
    {
        "hgvs_c":    "c.768G>A",
        "hgvs_p":    "p.Trp256Ter",
        "domain":    "C-terminal truncation (last residue)",
        "mechanism": "Nonsense at final residue; full-length C-terminus required for ND1-contact loop stability",
        "severity":  "severe",
        "ci_pct_range": (5, 10),
        "notes":     "Null-equivalent allele; consanguineous families; neonatal lethal if homozygous",
    },
    {
        "hgvs_c":    "c.IVS4+1G>A",
        "hgvs_p":    "p.?",
        "domain":    "Splice donor intron 4",
        "mechanism": "Canonical splice donor disruption; leaky splicing → partial TIMMDC1 protein residual",
        "severity":  "moderate",
        "ci_pct_range": (12, 20),
        "notes":     "Partial CI residual (12-20%); some residual normal splicing; milder clinical course; 3–12 months onset",
    },
    {
        "hgvs_c":    "c.335C>T",
        "hgvs_p":    "p.Ala112Val",
        "domain":    "TM1-TM2 interhelical loop",
        "mechanism": "Alanine-to-valine in interhelical loop; disrupts packing between TM1 and TM2 in the IMM bilayer",
        "severity":  "intermediate",
        "ci_pct_range": (10, 19),
        "notes":     "European founder variant in some populations; intermediate phenotype; some residual CI activity",
    },
]

REGIONS  = ["European",  "Middle Eastern", "North African", "South Asian", "East Asian", "Latin American"]
REGION_W = [0.28,         0.26,              0.18,            0.15,          0.08,          0.05]
OUTCOMES = ["alive, supported", "alive, institutionalized", "deceased (cardiac)", "deceased (respiratory)"]
OUTCOME_W = [0.33, 0.28, 0.24, 0.15]   # HCM-driven mortality shifts toward cardiac


def _patient(pid: int, var: dict) -> dict:
    onset_m = rng.randint(1, 24) if var["severity"] == "severe" else rng.randint(3, 72)
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
        "leigh_mri":         rng.random() < 0.72,
        "lactic_acidosis":   rng.random() < 0.92,
        "hcm":               rng.random() < 0.83,    # HIGH HCM — key distinguishing feature
        "seizures":          rng.random() < 0.58,
        "hypotonia":         rng.random() < 0.88,
        "dev_regression":    rng.random() < 0.75,
        "respiratory_fail":  rng.random() < 0.52,
        "outcome":           out,
    }


def _make_patients() -> list:
    patients = []
    pid = 1
    var_counts = [8, 7, 8, 6, 6, 5]   # sums to 40
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
        "HCM":                     round(sum(p["hcm"] for p in PATIENTS) / N_PATIENTS * 100),
        "Seizures":                round(sum(p["seizures"] for p in PATIENTS) / N_PATIENTS * 100),
        "Respiratory_failure":     round(sum(p["respiratory_fail"] for p in PATIENTS) / N_PATIENTS * 100),
        "Riboflavin_responder":    0,   # HARD 0 — no FAD domain
        "Peripheral_neuropathy":   0,   # HARD 0 — critical DDx vs NDUFS1
        "Leukodystrophy":          0,   # HARD 0 — critical DDx vs NDUFV1
        "Hepatopathy":             0,   # HARD 0 — critical DDx vs POLG/DGUOK
        "Olfactory_bulb_lesions":  0,   # HARD 0 — critical DDx vs NDUFS4
    }

    return {
        "gene":            GENE,
        "gene_full_name":  "Translocase of Inner Mitochondrial Membrane Domain Containing 1",
        "also_known_as":   "C3orf58 (Chromosome 3 Open Reading Frame 58)",
        "omim_gene":       OMIM_GENE,
        "omim_disease":    OMIM_DISEASE,
        "disease_name":    DISEASE_NAME,
        "chromosome":      CHROMOSOME,
        "inheritance":     INHERITANCE,
        "cohort_n":        N_PATIENTS,
        "cohort_seed":     SEED,

        "protein": {
            "size_aa":  256,
            "size_kda": 29.0,
            "fold":     "2 transmembrane (TM) helices; integral inner mitochondrial membrane protein; matrix-exposed loop contacts ND1-containing sub-assembly",
            "module":   "ND1-module P-module CI assembly (Class 3); ONLY integral-IMM member of Class 3 (NDUFAF3/4/5 are soluble matrix)",
            "function": (
                "TIMMDC1 is an integral inner mitochondrial membrane CI assembly factor anchored via "
                "two TM helices. Its matrix-exposed loop domain contacts the ND1-containing proton-pump "
                "(P-module) sub-assembly intermediate. TIMMDC1 belongs to the same Class 3 BN-PAGE "
                "assembly intermediate class as NDUFAF3/NDUFAF4/NDUFAF5 but is the ONLY integral-IMM "
                "member — all other Class-3 factors are soluble matrix proteins. Loss of TIMMDC1 stalls "
                "the ND1-containing sub-assembly, causing isolated CI deficiency with preserved CII/CIII/CIV. "
                "The IMM-anchored scaffold role is critical for cardiomyocyte CI biogenesis — HCM rate >80%."
            ),
        },

        "key_pathway_note": (
            "TIMMDC1 (C3orf58, 3q25.1) is the fourth Class-3 ND1-module CI assembly factor. "
            "It belongs to the same BN-PAGE assembly class as NDUFAF3 (2q33.1), NDUFAF4 (6q16.3), "
            "and NDUFAF5 (20p12.1), but is uniquely an integral IMM protein (2 TM helices). "
            "All other Class-3 factors are soluble matrix proteins. TIMMDC1's IMM-anchored position "
            "makes it the dominant ND1-module factor for cardiomyocyte CI assembly → HCM >80% "
            "(vs 15–25% for NDUFAF3/4 and <20% for NDUFAF5). "
            "High HCM + Class-3 BN-PAGE + isolated CI + 3q25.1 locus = TIMMDC1 fingerprint. "
            "Same chromosome 3 as ACAD9 (3q21.3): WES essential — different arm, different class. "
            "No FAD-binding domain → NO riboflavin response (unlike ACAD9, 50-60% Level B)."
        ),

        "biochemical_fingerprint": {
            "Complex_I":             "5–20% of control (SEVERE isolated deficiency)",
            "Complex_II":            "Normal (100%)",
            "Complex_III":           "Normal (100%)",
            "Complex_IV":            "Normal (100%)",
            "Complex_V":             "Normal (100%)",
            "Pattern":               "ISOLATED CI deficiency — Class 3 ND1-module; same BN-PAGE as NDUFAF3/4/5",
            "Riboflavin_response":   "NONE (0%) — no FAD-binding domain in TIMMDC1",
            "BN-PAGE_class":         "ND1-module (Class 3): stalled ND1 sub-assembly intermediate at IMM surface",
            "Integral_IMM_status":   "TIMMDC1 has 2 TM helices — only integral-IMM Class-3 member; NDUFAF3/4/5 are soluble",
            "HCM_rate":              "HIGH >80% — highest HCM rate in Class-3 group; key distinguishing feature",
        },

        "feature_frequencies_pct": ff,

        "timmdc1_module_summary": {
            "gene":              "TIMMDC1 (C3orf58, 3q25.1)",
            "module_class":      "ND1-module / P-module CI assembly (Class 3) — integral IMM anchor",
            "assembly_position": "Early ND1 sub-assembly at the IMM surface — matrix loop contacts ND1 intermediate",
            "integral_imm_unique": (
                "TIMMDC1 is the ONLY Class-3 ND1-module factor with TM helices. "
                "NDUFAF3, NDUFAF4, and NDUFAF5 are all soluble matrix proteins (zero TM helices). "
                "TIMMDC1's 2-TM topology anchors it to the IMM as a scaffold while its matrix-exposed "
                "loop domain contacts the ND1 sub-assembly intermediate."
            ),
            "hcm_mechanism": (
                "The IMM-anchored scaffold role of TIMMDC1 is critical for cardiomyocyte CI biogenesis. "
                "Cardiac mitochondria are dense and require tight IMM-surface coordination of CI assembly. "
                "Loss of the only IMM-anchored Class-3 factor impairs ND1-module assembly specifically "
                "in cardiac tissue → HCM >80% (highest HCM rate in the Class-3 group)."
            ),
            "timmdc1_loss_effect": (
                "Stalled early ND1-containing sub-assembly intermediate accumulates (Class 3 BN-PAGE). "
                "Isolated CI deficiency (5-20%). CI holoenzyme absent. "
                "HCM >80% due to cardiac CI assembly dependency on IMM-anchored scaffold. "
                "NDUFAF3 and NDUFAF4 proteins may be secondarily reduced in some patients "
                "(unlike NDUFAF5 which leaves them normal)."
            ),
        },

        "key_ddx": [
            {
                "feature":     "TIMMDC1 (3q25.1) vs NDUFAF3 (2q33.1) — HCM Key Distinguisher in Class 3",
                "significance": (
                    "TIMMDC1 and NDUFAF3 both produce Class 3 ND1-module BN-PAGE intermediates with "
                    "isolated CI deficiency. KEY CLINICAL DIFFERENCE: TIMMDC1 HCM >80% vs NDUFAF3 HCM 15-25%. "
                    "If HCM is severe and prominent in a Class-3 BN-PAGE patient → TIMMDC1. "
                    "WES mandatory: 3q25.1 (TIMMDC1) vs 2q33.1 (NDUFAF3). "
                    "TIMMDC1: integral IMM; NDUFAF3: soluble matrix. "
                    "NDUFAF3 deficiency destabilizes NDUFAF4; TIMMDC1 does not cause obligate partner loss."
                ),
                "target_gene": "NDUFAF3",
            },
            {
                "feature":     "TIMMDC1 (3q25.1) vs NDUFAF4 (6q16.3) — HCM Key Distinguisher in Class 3",
                "significance": (
                    "TIMMDC1 and NDUFAF4 both produce Class 3 BN-PAGE intermediates. "
                    "KEY CLINICAL DIFFERENCE: TIMMDC1 HCM >80% vs NDUFAF4 HCM 15-25%. "
                    "TIMMDC1: integral IMM (2 TM helices). NDUFAF4: soluble matrix. "
                    "WES mandatory: 3q25.1 vs 6q16.3. "
                    "HCM severity is the strongest single clinical discriminator in this comparison."
                ),
                "target_gene": "NDUFAF4",
            },
            {
                "feature":     "TIMMDC1 (3q25.1) vs NDUFAF5 (20p12.1) — HCM Rate Critical Distinguisher",
                "significance": (
                    "TIMMDC1 and NDUFAF5 both produce Class 3 BN-PAGE intermediates, both act independently "
                    "(not obligate heterodimers). KEY DIFFERENCE: TIMMDC1 HCM >80% vs NDUFAF5 HCM <20%. "
                    "TIMMDC1: integral IMM (2 TM helices). NDUFAF5: soluble matrix (no TM helices). "
                    "WES mandatory: 3q25.1 vs 20p12.1. HCM rate and TM-helix status differentiate."
                ),
                "target_gene": "NDUFAF5",
            },
            {
                "feature":     "TIMMDC1 (3q25.1) vs ACAD9 (3q21.3) — SAME CHROMOSOME, Different Class",
                "significance": (
                    "TIMMDC1 (3q25.1) and ACAD9 (3q21.3) are on the SAME chromosome 3 — CRITICAL DDx. "
                    "ACAD9: MCIA complex (ND2/ND5 module, Class 1 BN-PAGE). TIMMDC1: ND1-module (Class 3). "
                    "ACAD9: riboflavin-responsive (50-60%, Level B). TIMMDC1: 0% riboflavin response. "
                    "HCM overlap: ACAD9 55-65%, TIMMDC1 >80% — riboflavin response is the KEY distinguisher "
                    "on chromosome 3 (different bands, 3q21.3 vs 3q25.1, WES essential)."
                ),
                "target_gene": "ACAD9",
            },
            {
                "feature":     "TIMMDC1 vs NDUFS1 — NO Peripheral Neuropathy",
                "significance": (
                    "NDUFS1 (2q33.3) causes peripheral neuropathy in ~50% of patients. "
                    "TIMMDC1 deficiency: peripheral neuropathy 0%. Peripheral neuropathy rules out TIMMDC1 "
                    "and points to NDUFS1 or other N-module genes."
                ),
                "target_gene": "NDUFS1",
            },
            {
                "feature":     "TIMMDC1 vs NDUFV1 — NO Leukodystrophy",
                "significance": (
                    "NDUFV1 (11q13.2) causes leukodystrophy in 40-50% of patients. "
                    "TIMMDC1: leukodystrophy 0%. Leukodystrophy on MRI points away from TIMMDC1 toward NDUFV1."
                ),
                "target_gene": "NDUFV1",
            },
            {
                "feature":     "TIMMDC1 vs POLG/DGUOK — NO Hepatopathy",
                "significance": (
                    "POLG (15q26.1) and DGUOK (2p13.1) cause hepatopathy. "
                    "TIMMDC1 deficiency: hepatopathy 0%. Hepatopathy is a strong pointer away from TIMMDC1."
                ),
                "target_gene": "POLG / DGUOK",
            },
        ],

        "clinical_alerts": [
            "🔴 METFORMIN — ABSOLUTE CONTRAINDICATION: Direct CI inhibitor at ND1-complex territory. TIMMDC1 patients have 5-20% CI — metformin biguanide inhibition is lethal.",
            "🔴 VALPROATE (VPA) — ABSOLUTE CONTRAINDICATION: Triple mechanism: CoA sequestration, POLG inhibition, MT-ND subunit expression suppression.",
            "🔴 LINEZOLID — ABSOLUTE CONTRAINDICATION: 23S rRNA inhibition blocks MT-ND1–6 synthesis — shuts down all 7 mtDNA-encoded CI subunits.",
            "🔴 CHLORAMPHENICOL — ABSOLUTE CONTRAINDICATION: Same mitoribosomal 23S rRNA mechanism as linezolid.",
            "🔴 KETOGENIC DIET — CONTRAINDICATED: CI is absent/minimal; NADH cannot be reoxidised via ETC. Beta-oxidation floods matrix with NADH — metabolic crisis.",
            "🟠 PROPOFOL — AVOID (PRIS risk): Propofol infusion syndrome via CIV inhibition + fatty acid oxidation uncoupling. Use sevoflurane for anaesthesia.",
            "🟡 PHENOBARBITAL — HIGH CAUTION: Secondary CI inhibitor effect; use LEV as first-choice AED (renal excretion, no mitochondrial toxicity).",
            "🟢 SUCCINATE — Level C: CII substrate bypasses stalled CI entirely; allows CII → CIII → CIV electron flow.",
            "🟢 THIAMINE B1 — Level C MANDATORY EMPIRIC: Exclude SLC19A3 (thiamine transporter) and BTD (biotin-thiamine-responsive basal ganglia disease) before confirming TIMMDC1.",
            "🟢 BIOTIN — Level C MANDATORY EMPIRIC: Exclude BTD (biotinidase deficiency) and HLCS before CI gene panel.",
            "🔵 HCM >80% IN TIMMDC1: Highest HCM rate in Class-3 ND1-module group. Echocardiography at diagnosis and 6-monthly. Beta-blocker early if obstructive HCM. Digoxin AVOID (positive inotrope — worsens LVOT obstruction).",
            "🔵 TIMMDC1 (3q25.1) vs ACAD9 (3q21.3) — SAME CHROMOSOME: WES is mandatory. Riboflavin response distinguishes on chromosome 3 (ACAD9 responsive, TIMMDC1 0%).",
            "🔵 TIMMDC1 vs NDUFAF3/NDUFAF4/NDUFAF5 — WES MANDATORY: Same Class 3 BN-PAGE. HCM rate is the strongest pre-WES clinical clue (>80% in TIMMDC1 vs 15-25% in others).",
            "🔵 INTEGRAL IMM PROTEIN: Unlike soluble NDUFAF3/4/5, TIMMDC1 has 2 TM helices. TM-region mutations impair both IMM anchoring AND matrix-loop ND1-contact — compound mechanistic impact.",
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
        {"drug": "Metformin",       "mechanism": "Direct CI inhibitor at ND1-complex territory; biguanide binds CI at matrix side", "class": "ABSOLUTE CI"},
        {"drug": "Valproate (VPA)", "mechanism": "Triple: CoA sequestration, POLG inhibition, MT-ND subunit expression suppression", "class": "ABSOLUTE CI"},
        {"drug": "Linezolid",       "mechanism": "23S rRNA inhibition — blocks MT-ND1–6 synthesis; shuts all 7 mtDNA CI subunits",  "class": "ABSOLUTE CI"},
        {"drug": "Chloramphenicol", "mechanism": "Same mitoribosomal 23S rRNA mechanism as linezolid",                              "class": "ABSOLUTE CI"},
        {"drug": "Ketogenic Diet",  "mechanism": "NADH cannot be reoxidised (CI absent/minimal); floods matrix — metabolic crisis", "class": "CONTRAINDICATED"},
        {"drug": "Propofol",        "mechanism": "PRIS: CIV inhibition + fatty acid oxidation uncoupling",                          "class": "AVOID"},
        {"drug": "Phenobarbital",   "mechanism": "Secondary CI inhibitor; use LEV instead (renal, no mito toxicity)",               "class": "HIGH CAUTION"},
        {"drug": "Digoxin",         "mechanism": "Positive inotrope worsens LVOT obstruction in HCM — AVOID in obstructive HCM",   "class": "AVOID in HCM"},
    ]

    treatment_protocols = [
        {"agent": "Succinate",           "dose": "500–2000 mg/day oral",  "evidence": "Level C", "rationale": "CII substrate bypasses TIMMDC1-stalled CI; direct CII → CIII → CIV electron flow"},
        {"agent": "CoQ10 / Ubiquinol",   "dose": "10–30 mg/kg/day",       "evidence": "Level C", "rationale": "Downstream CI support; improves ETC electron flow at CIII–CIV"},
        {"agent": "Thiamine (B1)",        "dose": "100–600 mg/day",        "evidence": "Level C — MANDATORY EMPIRIC", "rationale": "Exclude SLC19A3 / BTD before confirming TIMMDC1"},
        {"agent": "Biotin",              "dose": "10–40 mg/day",           "evidence": "Level C — MANDATORY EMPIRIC", "rationale": "Exclude BTD / HLCS before CI gene panel"},
        {"agent": "Carnitine (L-Carn)",  "dose": "50–100 mg/kg/day",      "evidence": "Level C", "rationale": "Secondary carnitine deficiency possible; replete if low"},
        {"agent": "IV Dextrose (GIR 6–8)", "dose": "GIR 6–8 mg/kg/min",  "evidence": "Consensus", "rationale": "NEVER fast — CI-absent patients cannot sustain fasting gluconeogenesis"},
        {"agent": "Beta-blocker (HCM)",  "dose": "Atenolol / metoprolol",  "evidence": "Expert consensus — HCM", "rationale": "HCM >80%: beta-blocker for obstructive HCM symptom management; avoid digoxin"},
        {"agent": "LEV (Levetiracetam)", "dose": "20–60 mg/kg/day",       "evidence": "Consensus AED", "rationale": "Preferred AED — renal excretion, no mitochondrial toxicity"},
        {"agent": "Sevoflurane",         "dose": "Standard GA",            "evidence": "Consensus", "rationale": "Preferred volatile agent — NOT propofol (PRIS risk)"},
    ]

    nd1_module_steps = [
        {
            "step":  1,
            "event": "TIMMDC1 Anchors to IMM via 2 TM Helices — Integral Membrane Scaffold",
            "status_in_timmdc1_deficiency": "DISRUPTED — root cause",
            "note": "TIMMDC1 TM-helix mutations prevent IMM anchoring. Matrix loop cannot contact ND1 sub-assembly. Class 3 BN-PAGE intermediate accumulates.",
        },
        {
            "step":  2,
            "event": "Matrix-Loop Domain Contacts ND1-Containing Early P-Module Sub-Assembly",
            "status_in_timmdc1_deficiency": "ABSENT — IMM-anchor loss displaces matrix loop",
            "note": "Without IMM anchoring, the TIMMDC1 matrix loop cannot stably contact the ND1 sub-assembly intermediate.",
        },
        {
            "step":  3,
            "event": "ND3, ND4L, ND6 Incorporation into P-Module Sub-Assembly",
            "status_in_timmdc1_deficiency": "STALLED — upstream ND1-anchor block prevents subunit incorporation",
            "note": "P-module cannot complete without TIMMDC1-assisted IMM-anchored ND1 sub-assembly maturation.",
        },
        {
            "step":  4,
            "event": "Mature CI Holoenzyme Formation",
            "status_in_timmdc1_deficiency": "ABSENT — isolated CI deficiency (5-20%); HCM >80% from cardiac CI failure",
            "note": "No mature CI holoenzyme. Cardiomyocytes are particularly dependent on IMM-anchored TIMMDC1 → HCM >80%.",
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
                "class":      "Class 3 — ND1-Module / Early P-Module (TIMMDC1 is here)",
                "members":    "NDUFAF3, NDUFAF4, NDUFAF5, TIMMDC1",
                "module":     "Early ND1-containing P-module sub-assembly",
                "bnpage":     "Early ND1 sub-assembly intermediate",
                "riboflavin": "None (0%) — none have FAD domain; TIMMDC1 is only integral-IMM member",
            },
        ],

        "nd1_class3_comparison": [
            {
                "gene":           "NDUFAF3",
                "chromosome":     "2q33.1",
                "omim_gene":      "612911",
                "tm_helices":     "0 (soluble matrix)",
                "heterodimer":    "Obligate with NDUFAF4",
                "partner_effect": "NDUFAF4 secondarily reduced",
                "hcm":            "15-25%",
                "integral_imm":   "No",
            },
            {
                "gene":           "NDUFAF4",
                "chromosome":     "6q16.3",
                "omim_gene":      "611776",
                "tm_helices":     "0 (soluble matrix)",
                "heterodimer":    "Obligate with NDUFAF3",
                "partner_effect": "NDUFAF3 secondarily reduced",
                "hcm":            "15-25%",
                "integral_imm":   "No",
            },
            {
                "gene":           "NDUFAF5",
                "chromosome":     "20p12.1",
                "omim_gene":      "612360",
                "tm_helices":     "0 (soluble matrix)",
                "heterodimer":    "None (independent)",
                "partner_effect": "NDUFAF3/NDUFAF4 protein NORMAL",
                "hcm":            "<20%",
                "integral_imm":   "No",
            },
            {
                "gene":           "TIMMDC1",
                "chromosome":     "3q25.1",
                "omim_gene":      "615530",
                "tm_helices":     "2 TM helices (integral IMM)",
                "heterodimer":    "None (integral IMM scaffold)",
                "partner_effect": "N/A (IMM-anchored, not soluble partner)",
                "hcm":            ">80% (KEY DDx)",
                "integral_imm":   "Yes — ONLY integral-IMM Class-3 member",
            },
        ],
    }


def get_definitions() -> dict:
    return {
        "concepts": [
            {
                "term":       "TIMMDC1 (C3orf58) — Integral IMM ND1-Module Assembly Factor",
                "definition": (
                    "Translocase of Inner Mitochondrial Membrane Domain Containing 1, also known as "
                    "chromosome 3 open reading frame 58. ~256-amino acid, ~29 kDa integral inner "
                    "mitochondrial membrane CI assembly factor. Contains 2 TM helices that anchor "
                    "TIMMDC1 to the IMM, with a matrix-exposed loop domain that contacts the "
                    "ND1-containing P-module sub-assembly intermediate. TIMMDC1 is the ONLY Class-3 "
                    "ND1-module CI assembly factor with TM helices — NDUFAF3, NDUFAF4, and NDUFAF5 "
                    "are all soluble matrix proteins. Loss of TIMMDC1 causes isolated CI deficiency "
                    "with the highest HCM rate (>80%) of any Class-3 factor."
                ),
            },
            {
                "term":       "TIMMDC1 HCM >80% — IMM-Anchored Cardiac CI Assembly Dependency",
                "definition": (
                    "TIMMDC1 deficiency causes hypertrophic cardiomyopathy in >80% of patients — "
                    "the highest HCM rate of any Class-3 ND1-module gene (NDUFAF3/4: 15-25%, NDUFAF5: <20%). "
                    "Cardiomyocytes are dense in mitochondria and critically dependent on tightly regulated "
                    "IMM-surface CI assembly. The loss of the only IMM-anchored Class-3 scaffold factor "
                    "disproportionately impairs ND1-module assembly in cardiac tissue. "
                    "HCM in TIMMDC1 deficiency is often the presenting or dominant feature. "
                    "Echocardiography at diagnosis and 6-monthly surveillance are mandatory."
                ),
            },
            {
                "term":       "Class 3 BN-PAGE ND1-Module — TIMMDC1 vs NDUFAF3 vs NDUFAF4 vs NDUFAF5",
                "definition": (
                    "All four Class-3 genes (TIMMDC1, NDUFAF3, NDUFAF4, NDUFAF5) produce the same "
                    "Class 3 BN-PAGE CI assembly intermediate (stalled early ND1 sub-assembly). "
                    "WES chromosomal locus is mandatory to distinguish them: 3q25.1 (TIMMDC1), "
                    "2q33.1 (NDUFAF3), 6q16.3 (NDUFAF4), 20p12.1 (NDUFAF5). "
                    "Pre-WES clinical clue: HCM >80% is a strong pointer to TIMMDC1; "
                    "HCM 15-25% points to NDUFAF3/NDUFAF4; HCM <20% points to NDUFAF5."
                ),
            },
            {
                "term":       "TIMMDC1 (3q25.1) vs ACAD9 (3q21.3) — Same Chromosome, Different Class",
                "definition": (
                    "TIMMDC1 and ACAD9 are both on chromosome 3 (3q25.1 vs 3q21.3). "
                    "ACAD9 is in the MCIA complex (Class 1, ND2/ND5 module) and is riboflavin-responsive "
                    "(50-60%, Level B evidence). TIMMDC1 is Class-3 (ND1-module) and has 0% riboflavin "
                    "response (no FAD domain). Both have HCM (ACAD9: 55-65%; TIMMDC1: >80%). "
                    "On chromosome 3, riboflavin response is the key pre-WES clinical distinguisher: "
                    "response → ACAD9; no response → TIMMDC1 (or NDUFAF3/NDUFAF5 on other chromosomes)."
                ),
            },
            {
                "term":       "Integral IMM Protein — 2 TM Helices in TIMMDC1",
                "definition": (
                    "TIMMDC1's two transmembrane helices anchor it to the inner mitochondrial membrane "
                    "bilayer. Mutations in TM helices (e.g., p.Gly82Arg in TM1, p.Leu149Pro in TM2) "
                    "have a compound effect: they disrupt both the IMM-anchoring itself AND the "
                    "positioning of the matrix-exposed loop domain that contacts the ND1 sub-assembly. "
                    "TM-region mutations therefore tend to cause more severe phenotypes than equivalent "
                    "soluble-loop missense mutations, reflecting the dual-mechanism disruption."
                ),
            },
            {
                "term":       "Metformin Absolute Contraindication in CI Deficiency",
                "definition": (
                    "Biguanides (metformin) inhibit CI directly at the matrix-arm ND1 complex territory. "
                    "In TIMMDC1 deficiency (CI at 5-20%), any further CI inhibition is lethal. "
                    "Metformin is absolutely contraindicated in all nuclear-encoded CI deficiencies."
                ),
            },
            {
                "term":       "Succinate — CII Substrate (CI Bypass)",
                "definition": (
                    "Succinate enters the ETC at Complex II (succinate dehydrogenase), bypassing CI entirely. "
                    "In TIMMDC1 deficiency where CI is stalled, succinate supplementation allows residual "
                    "OXPHOS activity via CII → CIII → CIV pathway. Level C evidence."
                ),
            },
        ],

        "thresholds": [
            {"parameter": "CI activity (% control)",         "threshold": "<20%",         "significance": "Diagnostic for CI deficiency; TIMMDC1 typically 5-20%"},
            {"parameter": "Lactate (plasma)",                "threshold": ">2.5 mmol/L",  "significance": "Elevated in ~92% of TIMMDC1 patients; >4 = crisis threshold"},
            {"parameter": "Lactate:pyruvate ratio",          "threshold": ">25",           "significance": "Elevated L:P supports mitochondrial NADH block (ETC defect)"},
            {"parameter": "CSF lactate",                     "threshold": ">2.5 mmol/L",  "significance": "Leigh syndrome workup — elevated in CI deficiency with CNS involvement"},
            {"parameter": "HCM (echocardiography)",          "threshold": ">80% of cases","significance": "TIMMDC1: highest HCM rate in Class-3 group — echo at diagnosis + 6-monthly"},
            {"parameter": "Onset age (severe TM alleles)",   "threshold": "<24 months",   "significance": "TM-helix mutations (p.Gly82Arg, p.Leu149Pro): neonatal-infantile onset"},
            {"parameter": "CI activity (riboflavin trial)",  "threshold": "0% response",  "significance": "TIMMDC1 has NO FAD domain — zero riboflavin response expected; ACAD9 if responsive"},
        ],

        "standards": [
            {"code": "ACMG/AMP 2015",       "title": "Variant classification guidelines — pathogenicity criteria for TIMMDC1"},
            {"code": "MITOMAP",             "title": "Mitochondrial disease gene/variant database"},
            {"code": "OMIM *615530",        "title": "TIMMDC1 gene entry — C3orf58"},
            {"code": "OMIM #256000",        "title": "Mitochondrial Complex I Deficiency — CI deficiency spectrum"},
            {"code": "REACTOME R-HSA-611105","title": "Respiratory electron transport — CI assembly module"},
            {"code": "AHA/ACC 2020 HCM",    "title": "HCM management guidelines — beta-blocker, surveillance, avoid positive inotropes"},
            {"code": "Helsinki Declaration", "title": "Ethical framework for human subject research"},
        ],

        "references": [
            {
                "id":        "stroud_2016",
                "citation":  "Stroud DA et al. (2016) Accessory subunits are integral for assembly and function of human mitochondrial complex I. Nature 538:123–126.",
                "relevance": "BN-PAGE intermediate classification; mapped TIMMDC1 within CI assembly factor landscape; confirmed Class-3 ND1-module position of TIMMDC1.",
            },
            {
                "id":        "guerrero_castillo_2017",
                "citation":  "Guerrero-Castillo S et al. (2017) The assembly pathway of mitochondrial respiratory chain complex I. Cell Metab 25(1):128–139.",
                "relevance": "Defines CI assembly intermediates including Class 3 ND1-module; maps TIMMDC1 temporal position relative to NDUFAF3/NDUFAF4/NDUFAF5 in ND1-module assembly.",
            },
            {
                "id":        "pagliarini_2008",
                "citation":  "Pagliarini DJ et al. (2008) A mitochondrial protein compendium elucidates complex I disease biology. Cell 134(1):112–23.",
                "relevance": "Comprehensive mitochondrial proteome map; TIMMDC1 identified as CI-associated integral IMM protein; foundational reference for TIMMDC1 mitochondrial localization.",
            },
            {
                "id":        "fassone_2012",
                "citation":  "Fassone E & Rahman S (2012) Complex I deficiency: clinical features, biochemistry and molecular genetics. J Med Genet 49(9):578–90.",
                "relevance": "Comprehensive CI genetics review; frames TIMMDC1 within CI assembly factor landscape; clinical features of Class-3 ND1-module deficiencies.",
            },
            {
                "id":        "sazanov_2015",
                "citation":  "Sazanov LA (2015) A giant molecular proton pump: structure and mechanism of respiratory complex I. Nat Rev Mol Cell Biol 16(6):375–88.",
                "relevance": "CI structure review; contextualizes the P-module (ND1/ND3/ND4L/ND6) where TIMMDC1 operates as an integral-IMM scaffold.",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps({"overview": get_overview(), "breakdown": get_breakdown(), "definitions": get_definitions()}, indent=2, default=str))
