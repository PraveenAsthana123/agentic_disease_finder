#!/usr/bin/env python3
"""FOXRED1 — Mitochondrial Complex I Deficiency (N-Module CI Assembly Chaperone / FAD-Oxidoreductase).

FOXRED1 (FAD-dependent oxidoreductase domain-containing protein 1) is a CI-specific
assembly chaperone that acts on the N-module (NADH dehydrogenase module) sub-assembly.
It contains a FAD-binding oxidoreductase domain — but unlike ACAD9 (MCIA/ND2-ND5 module,
riboflavin-responsive 50–60%), FOXRED1 deficiency is NOT riboflavin-responsive. This is the
single most critical clinical distinguishing feature between FOXRED1 and ACAD9.

  FOXRED1 gene  OMIM *613622
  Disease       Mitochondrial Complex I Deficiency (OMIM #256000)
  Inheritance   AR (autosomal recessive, biallelic)
  Chromosome    11q24.2

Reference: Fassone E et al. (2010) FOXRED1, encoding an FAD-dependent oxidoreductase
complex-I-specific molecular chaperone, is mutated in infantile-onset mitochondrial
encephalopathy. Hum Mol Genet 19(24):4837–47. (first FOXRED1 disease gene identification)
Reference: Guerrero-Castillo S et al. (2017) The assembly pathway of mitochondrial respiratory
chain complex I. Cell Metab 25(1):128–139. (CI assembly intermediate mapping)
Reference: Stroud DA et al. (2016) Accessory subunits are integral for assembly and function
of human mitochondrial complex I. Nature 538:123–126. (CI assembly class framework)

PATHOPHYSIOLOGY (FOXRED1 / N-Module / FAD-Oxidoreductase Chaperone):
  CI assembly proceeds through modular intermediates. FOXRED1 acts as a chaperone
  specifically for the N-module (NADH dehydrogenase, matrix arm tip):
    1. FOXRED1 binds early N-module sub-assembly intermediates via its FAD domain.
    2. Its oxidoreductase activity (or chaperone scaffolding) facilitates correct folding
       of NDUFV1/NDUFV2/NDUFS1 N-module core subunits.
    3. Loss of FOXRED1 stalls N-module sub-assembly; holoenzyme CI cannot form.
    4. Isolated CI deficiency (5–20%); CII/CIII/CIV normal — N-module block is upstream.
    5. NO riboflavin response: although FOXRED1 is FAD-binding, riboflavin supplementation
       does NOT rescue CI assembly in FOXRED1 deficiency. Critical DDx vs ACAD9 (50-60%).

FOXRED1 UNIQUE FEATURES vs OTHER CI ASSEMBLY FACTORS:
  1. FAD-BINDING OXIDOREDUCTASE DOMAIN — BUT NO RIBOFLAVIN RESPONSE:
     FOXRED1 contains a bona fide FAD-binding oxidoreductase domain. This makes it
     superficially similar to ACAD9 (also FAD-binding). CRITICAL DIFFERENCE: ACAD9
     deficiency is riboflavin-responsive (50-60%, Level B evidence). FOXRED1 deficiency
     has ZERO riboflavin response. This is the KEY clinical distinguisher.
  2. N-MODULE SPECIFIC CHAPERONE — DISTINCT FROM MCIA (CLASS 1):
     ACAD9, NDUFAF1, ECSIT, TMEM126B all act on the MCIA/ND2-ND5 module (Class 1).
     FOXRED1 acts on the N-module (NADH dehydrogenase module, matrix arm tip).
     These are completely different assembly modules with different BN-PAGE intermediates.
  3. SOLUBLE MATRIX PROTEIN — NO TM HELICES:
     FOXRED1 is a soluble matrix protein (like NDUFAF3/4/5/NDUFAF1/ECSIT).
     Unlike TIMMDC1 (2 TM helices) or TMEM126B (2 TM helices), FOXRED1 is not IMM-anchored.
  4. NO HCM — KEY DDx vs TIMMDC1 AND NDUFV2/SCO2:
     FOXRED1 deficiency: HCM ~10% (low, similar to most N-module genes).
     Critical DDx: TIMMDC1 >80%, SCO2 ~65%, NDUFV2 ~60% — all high HCM.
     Low HCM in an N-module CI deficiency patient points toward FOXRED1.
  5. 11q24.2 — SAME CHROMOSOME 11 as NDUFV1 (11q13.2):
     NDUFV1 (11q13.2) causes leukodystrophy 40-50%. FOXRED1 (11q24.2): NO leukodystrophy.
     Different chromosome arm (p vs q), different band — WES essential on chromosome 11.

DISTINGUISHING FEATURES vs OTHER CI GENES:
  vs ACAD9 (3q21.3): Both FAD-binding domain. ACAD9 riboflavin-responsive (50-60%);
    FOXRED1 0% riboflavin response. ACAD9: MCIA/ND2-ND5 module (Class 1). FOXRED1: N-module.
    ACAD9 HCM 55-65%; FOXRED1 HCM ~10%. Riboflavin response is KEY distinguisher.
  vs NDUFV1 (11q13.2): SAME chromosome 11. NDUFV1: N-module structural subunit, causes
    leukodystrophy 40-50%. FOXRED1 (11q24.2): N-module chaperone, NO leukodystrophy.
    WES mandatory: 11q24.2 (FOXRED1) vs 11q13.2 (NDUFV1). Different locus on same chromosome.
  vs NDUFAF2 (5q12.1): Both N-module area assembly factors. NDUFAF2 is an NDUFA12 paralog
    (assembly-factor-swap chaperone). FOXRED1 is an FAD-oxidoreductase chaperone.
    Different mechanism; different chromosomal locus. WES distinguishes.
  vs NDUFS1 (2q33.3): NDUFS1 peripheral neuropathy ~50%; FOXRED1: 0%.
  vs NDUFV1 (11q13.2): NDUFV1 leukodystrophy 40-50%; FOXRED1: 0%.
  vs POLG/DGUOK: Both cause hepatopathy; FOXRED1 NO hepatopathy.
  vs TIMMDC1 (3q25.1): TIMMDC1 HCM >80%, integral IMM, ND1-module Class 3.
    FOXRED1 HCM ~10%, soluble matrix, N-module. Completely different assembly class.
"""

import random
import math

SEED = 691
rng  = random.Random(SEED)

GENE         = "FOXRED1"
OMIM_GENE    = "613622"
OMIM_DISEASE = "256000"
DISEASE_NAME = "Mitochondrial Complex I Deficiency (CI Deficiency; Leigh Syndrome spectrum)"
CHROMOSOME   = "11q24.2"
INHERITANCE  = "AR (biallelic)"
N_PATIENTS   = 40

# ─── Variants ────────────────────────────────────────────────────────────────
VARIANTS = [
    {
        "hgvs_c":    "c.787T>C",
        "hgvs_p":    "p.Trp263Arg",
        "domain":    "FAD-binding oxidoreductase domain — tryptophan structural core",
        "mechanism": "Tryptophan-to-arginine disrupts oxidoreductase domain core packing; chaperone scaffold collapses",
        "severity":  "severe",
        "ci_pct_range": (5, 12),
        "notes":     "Fassone 2010 index patient variant; severe infantile encephalopathy",
    },
    {
        "hgvs_c":    "c.1223G>A",
        "hgvs_p":    "p.Arg408Gln",
        "domain":    "FAD-binding interface — cofactor-coordinating arginine",
        "mechanism": "Arginine-to-glutamine weakens FAD cofactor coordination; oxidoreductase activity lost",
        "severity":  "severe",
        "ci_pct_range": (6, 14),
        "notes":     "FAD-binding interface mutation; chaperone function dependent on cofactor occupancy",
    },
    {
        "hgvs_c":    "c.1070T>C",
        "hgvs_p":    "p.Leu357Pro",
        "domain":    "Alpha-helix within oxidoreductase fold",
        "mechanism": "Helix-breaking proline disrupts alpha-helix continuity in core fold; protein misfolding",
        "severity":  "severe",
        "ci_pct_range": (5, 11),
        "notes":     "Proline-induced helix breaking — common severe mechanism across CI assembly factors",
    },
    {
        "hgvs_c":    "c.860C>T",
        "hgvs_p":    "p.Ala287Val",
        "domain":    "Oxidoreductase domain — hydrophobic core packing",
        "mechanism": "Alanine-to-valine adds steric bulk; disrupts tight core packing in oxidoreductase fold",
        "severity":  "intermediate",
        "ci_pct_range": (12, 22),
        "notes":     "Partial CI activity retained; later-onset infantile presentation",
    },
    {
        "hgvs_c":    "c.583G>A",
        "hgvs_p":    "p.Glu195Lys",
        "domain":    "Surface charge residue — N-module binding interface",
        "mechanism": "Glutamate-to-lysine inverts surface charge; disrupts electrostatic contact with N-module sub-assembly",
        "severity":  "intermediate",
        "ci_pct_range": (14, 24),
        "notes":     "Surface interaction disruption; partial N-module chaperone activity may persist",
    },
    {
        "hgvs_c":    "c.1054+1G>A",
        "hgvs_p":    "p.splice_donor_intron7",
        "domain":    "Splice donor — intron 7",
        "mechanism": "Splice donor disruption → intron 7 retention or exon 7 skipping → reading frame shifted/truncated",
        "severity":  "moderate",
        "ci_pct_range": (18, 30),
        "notes":     "Fassone 2010 second allele in compound het; partial normal transcript may persist (moderate residual CI)",
    },
]


def _variant(rng_):
    weights = [0.28, 0.20, 0.18, 0.16, 0.10, 0.08]
    v = rng_.choices(VARIANTS, weights=weights)[0]
    ci = rng_.uniform(*v["ci_pct_range"])
    return v, ci


def _make_patient(pid, rng_):
    v1, ci1 = _variant(rng_)
    v2, ci2 = _variant(rng_)
    ci_act  = round((ci1 + ci2) / 2 + rng_.gauss(0, 1.5), 1)
    ci_act  = max(3.0, min(32.0, ci_act))

    sev = v1["severity"]
    sev_score = {"severe": 3, "intermediate": 2, "moderate": 1}[sev]

    onset_mo = max(1, round(rng_.gauss(
        {"severe": 3, "intermediate": 9, "moderate": 18}[sev], 3
    )))
    outcome = rng_.choices(
        ["alive_stable", "alive_disabled", "deceased"],
        weights=[0.25, 0.50, 0.25]
    )[0]

    leigh_mri    = ci_act < 20 and rng_.random() < 0.72
    lactic_ac    = rng_.random() < 0.88
    hypotonia    = rng_.random() < 0.92
    dev_regr     = rng_.random() < 0.80
    hcm          = rng_.random() < 0.10   # ~10% — LOW, key DDx vs TIMMDC1/SCO2
    seizures     = rng_.random() < 0.58
    resp_fail    = rng_.random() < 0.48
    # HARD ZEROs — critical DDx
    riboflavin_r = False
    periph_nrp   = False
    leukodystro  = False
    hepatopathy  = False
    olfact_bulb  = False

    sex  = rng_.choice(["M", "F"])
    fam  = "consanguineous" if rng_.random() < 0.42 else "non-consanguineous"

    return {
        "id":                pid,
        "sex":               sex,
        "onset_age_months":  onset_mo,
        "family":            fam,
        "allele1":           v1["hgvs_p"],
        "allele2":           v2["hgvs_p"],
        "ci_activity_pct":   ci_act,
        "severity_allele1":  v1["severity"],
        "outcome":           outcome,
        "leigh_mri":         leigh_mri,
        "lactic_acidosis":   lactic_ac,
        "hypotonia":         hypotonia,
        "dev_regression":    dev_regr,
        "hcm":               hcm,
        "seizures":          seizures,
        "respiratory_fail":  resp_fail,
        "riboflavin_resp":   riboflavin_r,
        "peripheral_nrp":    periph_nrp,
        "leukodystrophy":    leukodystro,
        "hepatopathy":       hepatopathy,
        "olfactory_bulb":    olfact_bulb,
    }


PATIENTS = [_make_patient(i + 1, rng) for i in range(N_PATIENTS)]


# ─── get_overview ─────────────────────────────────────────────────────────────
def get_overview() -> dict:
    ff = {
        "Leigh_MRI":               round(sum(p["leigh_mri"] for p in PATIENTS) / N_PATIENTS * 100),
        "Lactic_acidosis":         round(sum(p["lactic_acidosis"] for p in PATIENTS) / N_PATIENTS * 100),
        "Hypotonia":               round(sum(p["hypotonia"] for p in PATIENTS) / N_PATIENTS * 100),
        "Developmental_regression":round(sum(p["dev_regression"] for p in PATIENTS) / N_PATIENTS * 100),
        "HCM":                     round(sum(p["hcm"] for p in PATIENTS) / N_PATIENTS * 100),
        "Seizures":                round(sum(p["seizures"] for p in PATIENTS) / N_PATIENTS * 100),
        "Respiratory_failure":     round(sum(p["respiratory_fail"] for p in PATIENTS) / N_PATIENTS * 100),
        "Riboflavin_responder":    0,   # HARD 0 — NO riboflavin response (key DDx vs ACAD9)
        "Peripheral_neuropathy":   0,   # HARD 0 — critical DDx vs NDUFS1
        "Leukodystrophy":          0,   # HARD 0 — critical DDx vs NDUFV1 (same chr 11)
        "Hepatopathy":             0,   # HARD 0 — critical DDx vs POLG/DGUOK
        "Olfactory_bulb_lesions":  0,   # HARD 0 — critical DDx vs NDUFS4
    }

    return {
        "gene":            GENE,
        "gene_full_name":  "FAD-dependent oxidoreductase domain-containing protein 1",
        "also_known_as":   "H17 (historical designator from initial sequence analysis)",
        "omim_gene":       OMIM_GENE,
        "omim_disease":    OMIM_DISEASE,
        "disease_name":    DISEASE_NAME,
        "chromosome":      CHROMOSOME,
        "inheritance":     INHERITANCE,
        "cohort_n":        N_PATIENTS,
        "cohort_seed":     SEED,

        "protein": {
            "size_aa":  486,
            "size_kda": 56.1,
            "fold":     "FAD-dependent oxidoreductase domain; soluble matrix protein; no TM helices",
            "module":   "N-module (NADH dehydrogenase module) CI assembly chaperone; distinct from MCIA/ND2-ND5 module (ACAD9/NDUFAF1/ECSIT/TMEM126B)",
            "function": (
                "FOXRED1 is an N-module-specific CI assembly chaperone. It contains a FAD-binding "
                "oxidoreductase domain that facilitates correct folding and assembly of N-module "
                "core subunits (NDUFV1, NDUFV2, NDUFS1). Loss of FOXRED1 stalls N-module "
                "sub-assembly, preventing holoenzyme CI formation and causing isolated CI deficiency. "
                "Despite containing a FAD-binding domain (like ACAD9), FOXRED1 deficiency is NOT "
                "riboflavin-responsive — the critical clinical distinguisher from ACAD9."
            ),
        },

        "key_pathway_note": (
            "FOXRED1 (11q24.2) is an N-module CI assembly chaperone with a FAD-binding "
            "oxidoreductase domain. It acts on the N-module (NADH dehydrogenase module — matrix arm tip), "
            "completely distinct from the MCIA complex (ACAD9/NDUFAF1/ECSIT/TMEM126B, ND2-ND5 module) "
            "and the ND1-module group (NDUFAF3/4/5/TIMMDC1). "
            "FAD-binding domain DOES NOT confer riboflavin responsiveness: FOXRED1 = 0% riboflavin response. "
            "This is the single most critical distinguishing feature from ACAD9 (50-60% riboflavin-responsive). "
            "Same chromosome 11 as NDUFV1 (11q13.2) — but different locus (11q24.2), "
            "different function (chaperone vs structural subunit), NO leukodystrophy. WES mandatory."
        ),

        "biochemical_fingerprint": {
            "Complex_I":             "5–20% of control (SEVERE isolated deficiency)",
            "Complex_II":            "Normal (100%)",
            "Complex_III":           "Normal (100%)",
            "Complex_IV":            "Normal (100%)",
            "Complex_V":             "Normal (100%)",
            "Pattern":               "ISOLATED CI deficiency — N-module chaperone block; N-module sub-assembly stalled",
            "Riboflavin_response":   "NONE (0%) — FAD domain present but riboflavin does NOT rescue FOXRED1 CI deficiency",
            "BN-PAGE_class":         "N-module sub-assembly intermediates stalled; holoenzyme CI absent/severely reduced",
            "Soluble_matrix_status": "FOXRED1 is soluble matrix (no TM helices); distinct from TIMMDC1/TMEM126B integral-IMM",
            "HCM_rate":              "LOW ~10% — critical DDx vs TIMMDC1 (>80%), SCO2 (~65%), NDUFV2 (~60%)",
        },

        "feature_frequencies_pct": ff,

        "foxred1_module_summary": {
            "gene":              "FOXRED1 (H17, 11q24.2)",
            "module_class":      "N-module (NADH dehydrogenase) specific CI assembly chaperone — FAD-oxidoreductase",
            "assembly_position": "N-module sub-assembly (matrix arm tip) — earliest NADH-oxidizing module",
            "fad_domain_unique": (
                "FOXRED1 contains a FAD-binding oxidoreductase domain — making it superficially "
                "similar to ACAD9 (also FAD-binding). CRITICAL DIFFERENCE: ACAD9 deficiency is "
                "riboflavin-responsive (50-60%, Level B); FOXRED1 deficiency has zero riboflavin "
                "response. The FAD domain in FOXRED1 serves a chaperone scaffolding/oxidoreductase "
                "function that cannot be rescued by riboflavin supplementation."
            ),
            "n_module_chaperone_role": (
                "FOXRED1 acts as a dedicated chaperone for the N-module — the NADH dehydrogenase "
                "module containing NDUFV1, NDUFV2, NDUFS1, and associated Fe-S subunits. "
                "This is the matrix arm tip of CI and the first module to receive electrons from NADH. "
                "Loss of FOXRED1 stalls N-module folding/assembly, preventing the N-sub-assembly "
                "from joining the growing holoenzyme. The block is upstream of the Q-module junction."
            ),
            "foxred1_loss_effect": (
                "N-module sub-assembly intermediates stall. Holoenzyme CI is absent/severely reduced. "
                "Isolated CI deficiency (5-20%). CII/CIII/CIV normal — block is N-module specific. "
                "No riboflavin response. BN-PAGE: N-module sub-assembly intermediates without "
                "holoenzyme formation (different pattern from MCIA-class Class 1 or ND1-module Class 3)."
            ),
        },

        "key_ddx": [
            {
                "feature":     "FOXRED1 (11q24.2) vs ACAD9 (3q21.3) — FAD-BINDING BUT NO RIBOFLAVIN RESPONSE",
                "significance": (
                    "Both FOXRED1 and ACAD9 have FAD-binding domains. CRITICAL DIFFERENCE: "
                    "ACAD9 deficiency is riboflavin-responsive (50-60%, Level B evidence) — "
                    "riboflavin is a first-line treatment that can dramatically improve CI activity in ACAD9. "
                    "FOXRED1 deficiency has ZERO riboflavin response — supplementation does not rescue CI. "
                    "ACAD9: MCIA complex (ND2-ND5 module, Class 1 BN-PAGE). FOXRED1: N-module chaperone. "
                    "ACAD9 HCM 55-65%; FOXRED1 HCM ~10%. Riboflavin response is the KEY clinical test. "
                    "WES mandatory: 3q21.3 (ACAD9) vs 11q24.2 (FOXRED1). Different chromosomes."
                ),
                "target_gene": "ACAD9",
            },
            {
                "feature":     "FOXRED1 (11q24.2) vs NDUFV1 (11q13.2) — SAME CHROMOSOME 11 — NO LEUKODYSTROPHY",
                "significance": (
                    "FOXRED1 (11q24.2) and NDUFV1 (11q13.2) are on the SAME chromosome 11 — CRITICAL DDx. "
                    "NDUFV1 is an N-module STRUCTURAL subunit (51kDa, NADH-binding, Fe-S cluster N3). "
                    "NDUFV1 deficiency: leukodystrophy 40-50%, characteristic progressive white matter disease. "
                    "FOXRED1 deficiency: N-module CHAPERONE, NO leukodystrophy. "
                    "WES mandatory: 11q13.2 (NDUFV1, long arm band 13.2) vs 11q24.2 (FOXRED1, long arm band 24.2). "
                    "Leukodystrophy on MRI is a strong pointer away from FOXRED1 toward NDUFV1."
                ),
                "target_gene": "NDUFV1",
            },
            {
                "feature":     "FOXRED1 vs NDUFAF2 (5q12.1) — Both N-Module Area, Different Mechanism",
                "significance": (
                    "NDUFAF2 is an assembly-factor-swap chaperone (NDUFA12 paralog that is displaced "
                    "by mature NDUFA12 subunit during final N-Q module assembly). FOXRED1 is an "
                    "FAD-oxidoreductase chaperone for early N-module folding. Different biochemical "
                    "mechanisms, different BN-PAGE intermediates, different chromosomal loci "
                    "(5q12.1 vs 11q24.2). WES distinguishes definitively."
                ),
                "target_gene": "NDUFAF2",
            },
            {
                "feature":     "FOXRED1 vs NDUFS1 (2q33.3) — NO Peripheral Neuropathy",
                "significance": (
                    "NDUFS1 (2q33.3) causes peripheral neuropathy in ~50% of patients — a hallmark "
                    "clinical feature. FOXRED1 deficiency: peripheral neuropathy 0%. Peripheral "
                    "neuropathy in a CI-deficiency patient rules out FOXRED1 and points to NDUFS1."
                ),
                "target_gene": "NDUFS1",
            },
            {
                "feature":     "FOXRED1 vs NDUFS4 (5q11.2) — NO Olfactory Bulb Lesions",
                "significance": (
                    "NDUFS4 deficiency causes pathognomonic bilateral olfactory bulb lesions on MRI "
                    "(52-65%). FOXRED1: olfactory bulb lesions 0%. Olfactory bulb MRI lesions "
                    "point away from FOXRED1 strongly."
                ),
                "target_gene": "NDUFS4",
            },
            {
                "feature":     "FOXRED1 vs TIMMDC1 (3q25.1) — Low HCM vs High HCM, Different Module",
                "significance": (
                    "TIMMDC1 deficiency: HCM >80% (highest in CI assembly factors). "
                    "FOXRED1 deficiency: HCM ~10% (low). If HCM is prominent, TIMMDC1 is more likely. "
                    "TIMMDC1: ND1-module (Class 3), integral IMM (2 TM helices). "
                    "FOXRED1: N-module chaperone, soluble matrix. Completely different assembly modules."
                ),
                "target_gene": "TIMMDC1",
            },
            {
                "feature":     "FOXRED1 vs POLG/DGUOK — NO Hepatopathy",
                "significance": (
                    "POLG (15q26.1) and DGUOK (2p13.1) cause hepatopathy. "
                    "FOXRED1 deficiency: hepatopathy 0%. Hepatopathy points away from FOXRED1."
                ),
                "target_gene": "POLG / DGUOK",
            },
        ],

        "clinical_alerts": [
            "🔴 METFORMIN — ABSOLUTE CONTRAINDICATION: Direct CI inhibitor. FOXRED1 patients have 5-20% CI; metformin biguanide inhibition is immediately life-threatening.",
            "🔴 VALPROATE (VPA) — ABSOLUTE CONTRAINDICATION: Triple mechanism: CoA sequestration, POLG inhibition, MT-ND subunit expression suppression. Causes further CI collapse.",
            "🔴 LINEZOLID — ABSOLUTE CONTRAINDICATION: 23S rRNA inhibition blocks all 7 mtDNA-encoded CI subunits (MT-ND1–6). No CI rescue possible.",
            "🔴 CHLORAMPHENICOL — ABSOLUTE CONTRAINDICATION: Same mitoribosomal 23S rRNA mechanism as linezolid.",
            "🔴 KETOGENIC DIET — CONTRAINDICATED: CI absent/minimal; NADH cannot be reoxidised via ETC. Beta-oxidation generates NADH that cannot be cleared — metabolic crisis.",
            "🟠 PROPOFOL — AVOID (PRIS risk): Propofol infusion syndrome via CIV inhibition + fatty acid oxidation uncoupling. Use sevoflurane for anaesthesia.",
            "🟡 PHENOBARBITAL — HIGH CAUTION: Secondary CI inhibitor effect; prefer LEV (levetiracetam) as first-choice AED — renal excretion, no mitochondrial toxicity.",
            "🟡 RIBOFLAVIN — NOT INDICATED FOR FOXRED1: FOXRED1 has a FAD-binding domain but riboflavin does NOT rescue CI deficiency in FOXRED1. Do not confuse with ACAD9 (riboflavin Level B).",
            "🟢 SUCCINATE — Level C: CII substrate bypasses stalled CI entirely; allows CII → CIII → CIV electron flow; partial ATP rescue.",
            "🟢 CoQ10 (Ubiquinol) — Level C: Antioxidant + electron carrier support; standard add-on.",
            "🟢 THIAMINE B1 — Level C MANDATORY EMPIRIC: Exclude SLC19A3 (thiamine transporter) and BTD (biotin-thiamine-responsive basal ganglia disease) BEFORE confirming FOXRED1.",
            "🟢 BIOTIN — Level C MANDATORY EMPIRIC: Exclude BTD (biotinidase deficiency) and HLCS BEFORE CI gene panel.",
            "🟢 CARNITINE — Level C: Supplement if secondary carnitine deficiency documented.",
            "🔵 FAD-BINDING ≠ RIBOFLAVIN RESPONSE: FOXRED1 has an FAD-binding oxidoreductase domain. CRITICAL: do NOT treat as riboflavin-responsive without WES confirmation. ACAD9 is the riboflavin-responsive CI factor.",
            "🔵 FOXRED1 (11q24.2) vs NDUFV1 (11q13.2) — SAME CHROMOSOME 11: WES mandatory. Leukodystrophy on MRI points strongly to NDUFV1, not FOXRED1.",
            "🔵 LOW HCM (~10%): Echocardiography at diagnosis. High HCM (>50%) in a CI-deficiency patient points away from FOXRED1 toward TIMMDC1 (>80%), SCO2 (~65%), or NDUFV2 (~60%).",
        ],
    }


# ─── get_breakdown ────────────────────────────────────────────────────────────
def get_breakdown() -> dict:
    variant_counts: dict[str, int] = {}
    for p in PATIENTS:
        for a in [p["allele1"], p["allele2"]]:
            variant_counts[a] = variant_counts.get(a, 0) + 1

    ci_vals    = [p["ci_activity_pct"] for p in PATIENTS]
    onset_vals = [p["onset_age_months"] for p in PATIENTS]

    ci_bands = {"<10%": 0, "10–15%": 0, "15–20%": 0, ">20%": 0}
    for c in ci_vals:
        if   c < 10:  ci_bands["<10%"]  += 1
        elif c < 15:  ci_bands["10–15%"] += 1
        elif c < 20:  ci_bands["15–20%"] += 1
        else:          ci_bands[">20%"]   += 1

    onset_bands = {"0–3 mo": 0, "4–12 mo": 0, "13–24 mo": 0, ">24 mo": 0}
    for o in onset_vals:
        if   o <= 3:   onset_bands["0–3 mo"]   += 1
        elif o <= 12:  onset_bands["4–12 mo"]   += 1
        elif o <= 24:  onset_bands["13–24 mo"]  += 1
        else:           onset_bands[">24 mo"]   += 1

    outcomes = {"alive_stable": 0, "alive_disabled": 0, "deceased": 0}
    for p in PATIENTS:
        outcomes[p["outcome"]] += 1

    severe_alleles = [v["hgvs_p"] for v in VARIANTS if v["severity"] == "severe"]
    top_severe = sum(
        1 for p in PATIENTS
        if p["allele1"] in severe_alleles or p["allele2"] in severe_alleles
    )

    consang = sum(1 for p in PATIENTS if p["family"] == "consanguineous")

    return {
        "cohort_n":           N_PATIENTS,
        "ci_activity_stats": {
            "mean":   round(sum(ci_vals) / N_PATIENTS, 1),
            "min":    round(min(ci_vals), 1),
            "max":    round(max(ci_vals), 1),
            "bands":  ci_bands,
        },
        "onset_stats": {
            "mean_months": round(sum(onset_vals) / N_PATIENTS, 1),
            "bands":       onset_bands,
        },
        "variant_frequency": dict(sorted(variant_counts.items(), key=lambda x: -x[1])[:8]),
        "sex_distribution":  {
            "M": sum(1 for p in PATIENTS if p["sex"] == "M"),
            "F": sum(1 for p in PATIENTS if p["sex"] == "F"),
        },
        "outcome_distribution":    outcomes,
        "consanguineous_pct":      round(consang / N_PATIENTS * 100),
        "pct_with_severe_allele":  round(top_severe / N_PATIENTS * 100),
        "key_n_module_features": {
            "FAD_domain_no_riboflavin_response": True,
            "N_module_specific_chaperone":       True,
            "soluble_matrix_no_TM_helices":      True,
            "isolated_CI_deficiency_only":       True,
            "MCIA_class_completely_different":   True,
            "HCM_rate_low":                      round(sum(p["hcm"] for p in PATIENTS) / N_PATIENTS * 100),
        },
        "treatment_summary": {
            "absolute_ci": ["Metformin", "Valproate (VPA)", "Linezolid", "Chloramphenicol", "Ketogenic Diet"],
            "avoid":       ["Propofol (PRIS)", "Phenobarbital (high caution)"],
            "do_not_use":  ["Riboflavin (NOT indicated — no response in FOXRED1 despite FAD domain)"],
            "level_c":     ["Succinate (CII bypass)", "CoQ10-Ubiquinol", "Thiamine B1 (MANDATORY empiric)", "Biotin (MANDATORY empiric)", "Carnitine"],
            "preferred_aed": "LEV (levetiracetam) — renal excretion, no mitochondrial toxicity",
            "anaesthesia": "Sevoflurane (NOT propofol — PRIS risk)",
        },
        "variant_table": [
            {
                "hgvs_c":    v["hgvs_c"],
                "hgvs_p":    v["hgvs_p"],
                "domain":    v["domain"],
                "mechanism": v["mechanism"],
                "severity":  v["severity"],
                "ci_range":  f"{v['ci_pct_range'][0]}–{v['ci_pct_range'][1]}%",
                "notes":     v["notes"],
            }
            for v in VARIANTS
        ],
        "ddx_matrix": [
            {
                "comparator":     "ACAD9 (3q21.3)",
                "foxred1":        "FAD-binding; 0% riboflavin response; N-module; HCM ~10%",
                "comparator_val": "FAD-binding; riboflavin-responsive 50-60%; MCIA/ND2-ND5; HCM 55-65%",
                "key_test":       "Riboflavin trial + WES locus (11q24.2 vs 3q21.3)",
            },
            {
                "comparator":     "NDUFV1 (11q13.2) — same chr 11",
                "foxred1":        "Chaperone; no leukodystrophy; 11q24.2",
                "comparator_val": "Structural N-module subunit; leukodystrophy 40-50%; 11q13.2",
                "key_test":       "MRI leukodystrophy + WES locus (different band on chr 11)",
            },
            {
                "comparator":     "NDUFAF2 (5q12.1)",
                "foxred1":        "FAD-oxidoreductase chaperone; N-module early folding",
                "comparator_val": "NDUFA12-paralog assembly-swap; N-Q module interface",
                "key_test":       "WES (different chromosomes)",
            },
            {
                "comparator":     "TIMMDC1 (3q25.1)",
                "foxred1":        "N-module; soluble matrix; HCM ~10%",
                "comparator_val": "ND1-module (Class 3); integral IMM; HCM >80%",
                "key_test":       "HCM rate + BN-PAGE class + WES locus",
            },
            {
                "comparator":     "NDUFS1 (2q33.3)",
                "foxred1":        "NO peripheral neuropathy",
                "comparator_val": "Peripheral neuropathy ~50%",
                "key_test":       "Clinical neurophysiology (nerve conduction study)",
            },
        ],
    }


# ─── get_definitions ──────────────────────────────────────────────────────────
def get_definitions() -> dict:
    return {
        "gene_definition": (
            "FOXRED1 (FAD-dependent oxidoreductase domain-containing protein 1; also H17) encodes "
            "a 486-amino-acid, 56.1 kDa soluble mitochondrial matrix protein. It contains a bona "
            "fide FAD-binding oxidoreductase domain and functions as a dedicated CI assembly "
            "chaperone for the N-module (NADH dehydrogenase module, matrix arm tip). FOXRED1 is "
            "required for correct folding and assembly of N-module core subunits including NDUFV1, "
            "NDUFV2, and NDUFS1. Loss-of-function variants cause isolated CI deficiency with "
            "Leigh syndrome spectrum. Despite its FAD domain, riboflavin supplementation does "
            "NOT rescue CI activity — critical DDx from ACAD9."
        ),
        "disease_definition": (
            "Mitochondrial Complex I Deficiency (CI Deficiency; OMIM #256000) due to FOXRED1 "
            "bi-allelic pathogenic variants presents as infantile-onset Leigh syndrome with "
            "profound isolated CI deficiency (5-20% of control), preserved CII/CIII/CIV, "
            "Leigh MRI (~72%), lactic acidosis (~88%), hypotonia (~92%), seizures (~58%), and "
            "low HCM rate (~10%). The key distinguishing metabolic feature is the complete "
            "absence of riboflavin response despite the presence of a FAD-binding domain in FOXRED1."
        ),
        "inheritance_definition": (
            "Autosomal recessive (AR). Biallelic pathogenic FOXRED1 variants required. "
            "Both sexes equally affected. Consanguinity is a common background (~40%). "
            "Carrier parents are clinically unaffected. Genetic counselling: 25% recurrence risk per pregnancy."
        ),
        "module_definitions": [
            {
                "term":       "N-Module (NADH Dehydrogenase Module)",
                "definition": (
                    "The N-module is the matrix arm tip of CI containing the NADH-binding and "
                    "primary Fe-S electron transfer subunits: NDUFV1 (51kDa, NADH-binding, N3 cluster), "
                    "NDUFV2 (24kDa, N1a/N1b cluster), NDUFV3, NDUFS1 (75kDa, N1b/N3/N4/N5 clusters). "
                    "The N-module is the first to receive electrons from NADH. FOXRED1 is an "
                    "N-module-specific chaperone — distinct from MCIA (ACAD9/NDUFAF1/ECSIT/TMEM126B) "
                    "which acts on the ND2-ND5 membrane arm, and from ND1-module factors "
                    "(NDUFAF3/4/5/TIMMDC1) which act on the ND1-containing P-module."
                ),
            },
            {
                "term":       "FAD-Binding Without Riboflavin Response",
                "definition": (
                    "FOXRED1 contains an FAD-binding oxidoreductase domain. FAD (flavin adenine "
                    "dinucleotide) is a riboflavin (vitamin B2) derivative. In ACAD9 deficiency, "
                    "the FAD-binding ACAD superfamily domain is partially stabilized by riboflavin "
                    "supplementation, rescuing CI assembly (50-60%, Level B). In FOXRED1 deficiency, "
                    "the FAD domain is used for a different oxidoreductase chaperone function and "
                    "riboflavin supplementation does NOT rescue CI assembly. This difference is "
                    "mechanistically important: in ACAD9, riboflavin stabilizes the enzyme's "
                    "catalytic fold; in FOXRED1, the block is in chaperone-client interaction "
                    "that riboflavin cannot overcome."
                ),
            },
            {
                "term":       "CI Assembly Chaperone vs Structural Subunit",
                "definition": (
                    "CI assembly chaperones (FOXRED1, NDUFAF1, ACAD9, ECSIT, TMEM126B, NDUFAF2-8) "
                    "are transiently associated with CI sub-assembly intermediates but are NOT "
                    "present in the mature holoenzyme. They guide folding, prevent aggregation, "
                    "and coordinate module assembly. Structural subunits (NDUFV1, NDUFV2, NDUFS1, etc.) "
                    "are integral components of the mature CI. FOXRED1 is a chaperone — absent from "
                    "mature CI — unlike NDUFV1 (structural N-module subunit on same chromosome 11)."
                ),
            },
        ],
        "clinical_thresholds": [
            {"parameter": "CI enzyme activity",                 "threshold": "<20% of control",    "significance": "Diagnostic criterion for severe CI deficiency in muscle biopsy"},
            {"parameter": "Lactate (plasma)",                   "threshold": ">2.5 mmol/L",        "significance": "Elevated; confirms mitochondrial dysfunction (non-specific)"},
            {"parameter": "Lactate:pyruvate ratio",             "threshold": ">25",                "significance": "Elevated L:P supports NADH-block at CI (ETC defect rather than PDH)"},
            {"parameter": "CSF lactate",                        "threshold": ">2.5 mmol/L",        "significance": "Leigh syndrome workup — elevated in CI deficiency with CNS involvement"},
            {"parameter": "Riboflavin trial response",          "threshold": "0% (none expected)", "significance": "FOXRED1 is FAD-binding but NOT riboflavin-responsive; 0% response confirms FOXRED1; any response favors ACAD9"},
            {"parameter": "HCM (echocardiography)",            "threshold": "~10% of cases",      "significance": "Low HCM — high HCM (>50%) points away from FOXRED1 toward TIMMDC1/SCO2"},
            {"parameter": "MRI leukodystrophy",                 "threshold": "0% expected",        "significance": "Leukodystrophy on MRI is a strong pointer away from FOXRED1 toward NDUFV1 (same chr 11)"},
            {"parameter": "Onset age (severe splice/missense)", "threshold": "<6 months",          "significance": "Biallelic severe alleles (p.Trp263Arg, p.Leu357Pro): neonatal-to-early-infantile"},
        ],
        "standards": [
            {"code": "ACMG/AMP 2015",         "title": "Variant classification guidelines — pathogenicity criteria for FOXRED1"},
            {"code": "MITOMAP",               "title": "Mitochondrial disease gene/variant database"},
            {"code": "OMIM *613622",          "title": "FOXRED1 gene entry — FAD-dependent oxidoreductase domain-containing protein 1"},
            {"code": "OMIM #256000",          "title": "Mitochondrial Complex I Deficiency — CI deficiency spectrum"},
            {"code": "REACTOME R-HSA-611105", "title": "Respiratory electron transport — CI assembly module"},
            {"code": "Helsinki Declaration",  "title": "Ethical framework for human subject research"},
        ],
        "references": [
            {
                "id":        "fassone_2010",
                "citation":  "Fassone E et al. (2010) FOXRED1, encoding an FAD-dependent oxidoreductase complex-I-specific molecular chaperone, is mutated in infantile-onset mitochondrial encephalopathy. Hum Mol Genet 19(24):4837–47.",
                "relevance": "First identification of FOXRED1 as a disease gene; compound heterozygote (c.1054+1G>A splice + p.Trp263Arg); establishes FOXRED1 as a CI-specific N-module chaperone; key reference for all subsequent FOXRED1 clinical work.",
            },
            {
                "id":        "guerrero_castillo_2017",
                "citation":  "Guerrero-Castillo S et al. (2017) The assembly pathway of mitochondrial respiratory chain complex I. Cell Metab 25(1):128–139.",
                "relevance": "Defines CI assembly intermediates and temporal order; contextualizes FOXRED1 within the N-module assembly pathway; maps FOXRED1 chaperone action relative to other CI assembly factors.",
            },
            {
                "id":        "stroud_2016",
                "citation":  "Stroud DA et al. (2016) Accessory subunits are integral for assembly and function of human mitochondrial complex I. Nature 538:123–126.",
                "relevance": "BN-PAGE intermediate classification framework; provides the three-class assembly model (Class 1/2/3); FOXRED1 acts upstream of Class 2 N-Q module consolidation.",
            },
            {
                "id":        "fassone_2012",
                "citation":  "Fassone E & Rahman S (2012) Complex I deficiency: clinical features, biochemistry and molecular genetics. J Med Genet 49(9):578–90.",
                "relevance": "Comprehensive CI genetics review; frames FOXRED1 within CI assembly factor landscape; clinical features and DDx of N-module chaperone deficiencies.",
            },
            {
                "id":        "sazanov_2015",
                "citation":  "Sazanov LA (2015) A giant molecular proton pump: structure and mechanism of respiratory complex I. Nat Rev Mol Cell Biol 16(6):375–88.",
                "relevance": "CI structure review; contextualizes the N-module where FOXRED1 acts as a chaperone; NADH-binding site, Fe-S clusters, and electron transfer chain within the N-module.",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps({"overview": get_overview(), "breakdown": get_breakdown(), "definitions": get_definitions()}, indent=2, default=str))
