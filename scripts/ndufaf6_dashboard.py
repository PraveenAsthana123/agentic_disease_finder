#!/usr/bin/env python3
"""NDUFAF6 (C8orf38) — Mitochondrial Complex I Deficiency MC1DN26 (Late-Stage CI Assembly 2OG-Fe(II)-Dioxygenase).

NDUFAF6 (NADH:Ubiquinone Oxidoreductase Complex Assembly Factor 6; also known as C8orf38) is a
2-oxoglutarate Fe(II)-dependent dioxygenase superfamily member dedicated to late-stage CI assembly.
It is the ONLY CI assembly factor with established 2OG-Fe(II) oxygenase/hydroxylase enzymatic activity,
catalyzing post-translational hydroxylation of a CI subunit (NDUFS7/PSST subunit leucyl/asparaginyl
modification) required for Q-module maturation.

  NDUFAF6 gene   OMIM *612392
  Disease        Mitochondrial Complex I Deficiency MC1DN26 (OMIM #614924)
  Inheritance    AR (autosomal recessive, biallelic)
  Chromosome     8q22.1

Reference: McKenzie M et al. (2011) Mutations in the gene encoding C8orf38 block complex I assembly
by inhibiting production of the mitochondria-encoded subunit ND1. Am J Hum Genet 89(5):711–8.
(First NDUFAF6/C8orf38 CI assembly factor identification)
Reference: Guerrero-Castillo S et al. (2017) The assembly pathway of mitochondrial respiratory
chain complex I. Cell Metab 25(1):128–139. (CI assembly intermediate mapping; late-stage context)

PATHOPHYSIOLOGY (NDUFAF6 / Late-Stage CI Assembly / 2OG-Dioxygenase):
  NDUFAF6/C8orf38 is a 2-oxoglutarate Fe(II)-dependent dioxygenase that acts at a late
  stage of CI assembly — specifically the Q-module maturation step:
    1. NDUFAF6 uses 2-oxoglutarate (2OG) as co-substrate and Fe(II) as cofactor to
       catalyze hydroxylation of a CI subunit (NDUFS7/PSST leucyl or asparaginyl residue).
    2. This post-translational modification is required for proper Q-module maturation —
       specifically the insertion of NDUFS7 (PSST subunit) into the Q-module of CI.
    3. Without NDUFAF6-mediated hydroxylation, Q-module maturation stalls; late-stage
       CI assembly intermediates accumulate on BN-PAGE; holoenzyme CI cannot form.
    4. Isolated CI deficiency (5–20%); CII/CIII/CIV normal — block is late-stage Q-module.
    5. No FAD domain, no TM helices — soluble matrix protein with 2OG-dioxygenase fold.

NDUFAF6 UNIQUE FEATURES vs OTHER CI ASSEMBLY FACTORS:
  1. ONLY CI ASSEMBLY FACTOR WITH 2OG-Fe(II) OXYGENASE/HYDROXYLASE ENZYMATIC ACTIVITY:
     NDUFAF6 is the sole known CI assembly factor that catalyzes a post-translational
     modification (hydroxylation) of a CI subunit. All other CI assembly factors act as
     chaperones, scaffolds, or structural assembly factors — not as hydroxylases/oxygenases.
     This is unique among all known CI assembly factors.
  2. POST-TRANSLATIONAL CI SUBUNIT MODIFICATION — NDUFS7/PSST TARGET:
     The hydroxylation target is NDUFS7 (PSST subunit, Q-module), specifically a leucyl
     or asparaginyl residue. This modification is required for NDUFS7 integration into
     the Q-module and for late-stage CI assembly to proceed.
  3. LATE-STAGE CI ASSEMBLY Q-MODULE CONTEXT:
     NDUFAF6 acts at a late stage distinct from: N-module (FOXRED1/NUBPL), ND2-ND5/MCIA
     (ACAD9/NDUFAF1/ECSIT/TMEM126B), and ND1-module/Class3 (NDUFAF3/4/5/TIMMDC1).
     BN-PAGE shows Q-module/late-stage assembly intermediates stalled.
  4. NO RIBOFLAVIN RESPONSE (0%): No FAD domain. Riboflavin supplementation CANNOT
     rescue the 2OG-dioxygenase hydroxylation defect. Critical DDx vs ACAD9 (50-60%).
  5. LOW HCM (<5-10%): Very low HCM compared to TIMMDC1 (>80%), NDUFV2 (80%),
     ACAD9 (55-65%). Important DDx marker.
  6. 8q22.1 — UNIQUE CHROMOSOMAL LOCUS for late-stage Q-module CI assembly.

DISTINGUISHING FEATURES vs OTHER CI GENES:
  vs ACAD9 (3q21.3): ACAD9 = MCIA/ND2-ND5 (Class 1); riboflavin-responsive 50-60%.
    NDUFAF6 = 2OG-dioxygenase; 0% riboflavin response. Different chromosomes. Key test: riboflavin trial.
  vs NDUFAF5 (20p12.1): Both no-FAD, no riboflavin response. Different chromosomes.
    WES mandatory. NDUFAF5 = ND1 module; NDUFAF6 = Q-module late-stage. BN-PAGE differs.
  vs FOXRED1 (11q24.2): FOXRED1 = N-module FAD-oxidoreductase chaperone. NDUFAF6 = 2OG-dioxygenase.
    Different modules, different biochemistry.
  vs NUBPL (14q12): NUBPL = N-module [4Fe-4S] delivery. NDUFAF6 = Q-module 2OG-hydroxylase.
    Different modules, different biochemistry.
  vs TIMMDC1 (3q25.1): TIMMDC1 HCM >80%; NDUFAF6 HCM <5-10%. Critical DDx marker.
  vs NDUFV2 (11q): NDUFV2 HCM ~80%; NDUFAF6 HCM <10%. Important DDx.
  vs NDUFV1 (11q13.2): NDUFV1 = leukodystrophy 40-50%. NDUFAF6: NO leukodystrophy.
  vs NDUFS1 (2q33.3): NDUFS1 peripheral neuropathy ~50%. NDUFAF6: 0%.
  vs POLG/DGUOK: NDUFAF6: NO hepatopathy.
"""

import random
import math

SEED = 695
rng  = random.Random(SEED)

GENE         = "NDUFAF6"
OMIM_GENE    = "612392"
OMIM_DISEASE = "614924"
DISEASE_NAME = "Mitochondrial Complex I Deficiency MC1DN26 (OMIM #614924)"
CHROMOSOME   = "8q22.1"
INHERITANCE  = "AR (biallelic)"
N_PATIENTS   = 40

# ─── Variants ────────────────────────────────────────────────────────────────
VARIANTS = [
    {
        "hgvs_c":    "c.253C>T",
        "hgvs_p":    "p.Arg85Trp",
        "domain":    "Hydroxylase active site — 2OG-Fe(II) dioxygenase catalytic core",
        "mechanism": "Arginine-to-tryptophan at the hydroxylase active site disrupts Fe(II) coordination and 2-oxoglutarate binding; oxygenase catalytic activity abolished; NDUFS7/PSST hydroxylation fails",
        "severity":  "severe",
        "ci_pct_range": (5, 12),
        "notes":     "Severe infantile onset; direct disruption of 2OG-dioxygenase catalytic residue; no residual hydroxylase activity; Q-module maturation stalled",
    },
    {
        "hgvs_c":    "c.638T>C",
        "hgvs_p":    "p.Leu213Pro",
        "domain":    "Alpha-helix — 2OG-dioxygenase fold structural domain",
        "mechanism": "Helix-breaking proline substitution disrupts alpha-helix continuity in the 2OG-dioxygenase fold; protein misfolding; loss of NDUFS7 hydroxylation activity",
        "severity":  "severe",
        "ci_pct_range": (5, 12),
        "notes":     "Severe infantile onset; helix-breaking proline — common severe mechanism across CI assembly factors; NDUFAF6 protein unstable; no residual 2OG-dioxygenase activity",
    },
    {
        "hgvs_c":    "c.175G>C",
        "hgvs_p":    "p.Gly59Arg",
        "domain":    "2OG-dioxygenase core fold — conserved glycine in jelly-roll beta-barrel",
        "mechanism": "Glycine-to-arginine in the conserved jelly-roll beta-barrel core of the 2OG-dioxygenase fold; steric clash; protein fold disrupted; loss of NDUFAF6 stability and enzymatic function",
        "severity":  "severe",
        "ci_pct_range": (5, 12),
        "notes":     "Severe infantile onset; glycine-to-arginine is a severe steric clash; core fold collapse; no residual hydroxylase activity; CI activity 5-12% of controls",
    },
    {
        "hgvs_c":    "c.860C>T",
        "hgvs_p":    "p.Ala287Val",
        "domain":    "Protein core packing — C-terminal dioxygenase domain",
        "mechanism": "Alanine-to-valine disrupts hydrophobic core packing in the C-terminal domain; partial protein misfolding; residual NDUFAF6 activity retained; intermediate CI deficiency",
        "severity":  "intermediate",
        "ci_pct_range": (14, 22),
        "notes":     "Intermediate onset (3-18 months); partial residual 2OG-dioxygenase activity; partial NDUFS7 hydroxylation; compound het with severe allele gives intermediate phenotype",
    },
    {
        "hgvs_c":    "c.IVS5+1G>A",
        "hgvs_p":    "p.splice_donor_intron5",
        "domain":    "Splice donor — intron 5",
        "mechanism": "Splice donor disruption → intron 5 retention or exon 5 skipping → reading frame shift/truncation; partial normal transcript may persist via cryptic splice site; hypomorphic allele",
        "severity":  "moderate",
        "ci_pct_range": (18, 28),
        "notes":     "Moderate; partial normal NDUFAF6 transcript retained if cryptic splice site activated; partial 2OG-dioxygenase activity preserved; variable CI residual 18-28%; later infantile onset",
    },
    {
        "hgvs_c":    "c.960G>A",
        "hgvs_p":    "p.Trp320Ter",
        "domain":    "C-terminal truncation — 2OG-dioxygenase substrate-binding domain",
        "mechanism": "Premature termination codon truncates the C-terminal substrate-binding domain of the 2OG-dioxygenase; C-terminal region required for NDUFS7/PSST substrate recognition; null allele",
        "severity":  "severe",
        "ci_pct_range": (4, 11),
        "notes":     "Null allele; neonatal-onset severe CI deficiency; C-terminal truncation abolishes NDUFS7 substrate binding; compound het with moderate allele gives intermediate phenotype",
    },
]


def _variant(rng_):
    # severe variants ~55% together (split 3 ways ~18.3% each); splice moderate ~18%; intermediate ~15%; null ~12%
    weights = [0.185, 0.185, 0.18, 0.15, 0.18, 0.12]
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
        {"severe": 3, "intermediate": 10, "moderate": 20}[sev], 3
    )))
    outcome = rng_.choices(
        ["alive_stable", "alive_disabled", "deceased"],
        weights=[0.22, 0.53, 0.25]
    )[0]

    leigh_mri    = ci_act < 20 and rng_.random() < 0.74
    lactic_ac    = rng_.random() < 0.85
    hypotonia    = rng_.random() < 0.90
    dev_regr     = rng_.random() < 0.75
    hcm          = rng_.random() < 0.08   # Low <5-10% — critical DDx vs TIMMDC1 (>80%), NDUFV2 (80%)
    seizures     = rng_.random() < 0.54
    resp_fail    = rng_.random() < 0.44
    # HARD ZEROs — critical DDx
    riboflavin_r = False   # NO riboflavin response — NDUFAF6 has no FAD domain
    periph_nrp   = False   # NO peripheral neuropathy — critical DDx vs NDUFS1
    leukodystro  = False   # NO leukodystrophy — critical DDx vs NDUFV1
    hepatopathy  = False   # NO hepatopathy — critical DDx vs POLG/DGUOK
    olfact_bulb  = False   # NO olfactory bulb lesions — critical DDx vs NDUFS4

    sex  = rng_.choice(["M", "F"])
    fam  = "consanguineous" if rng_.random() < 0.36 else "non-consanguineous"

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
        "Riboflavin_responder":    0,   # HARD 0 — NO riboflavin response (no FAD domain in NDUFAF6)
        "Peripheral_neuropathy":   0,   # HARD 0 — critical DDx vs NDUFS1
        "Leukodystrophy":          0,   # HARD 0 — critical DDx vs NDUFV1
        "Hepatopathy":             0,   # HARD 0 — critical DDx vs POLG/DGUOK
        "Olfactory_bulb_lesions":  0,   # HARD 0 — critical DDx vs NDUFS4
    }

    return {
        "gene":            GENE,
        "gene_full_name":  "NADH:Ubiquinone Oxidoreductase Complex Assembly Factor 6 (C8orf38 — 2OG-Fe(II)-dependent dioxygenase superfamily; late-stage CI assembly hydroxylase)",
        "also_known_as":   "C8orf38 (chromosome 8 open reading frame 38); 2-oxoglutarate-dependent dioxygenase CI assembly factor",
        "omim_gene":       OMIM_GENE,
        "omim_disease":    OMIM_DISEASE,
        "disease_name":    DISEASE_NAME,
        "chromosome":      CHROMOSOME,
        "inheritance":     INHERITANCE,
        "cohort_n":        N_PATIENTS,
        "cohort_seed":     SEED,

        "protein": {
            "size_aa":  369,
            "size_kda": 42.0,
            "fold":     "2-oxoglutarate Fe(II)-dependent dioxygenase superfamily (jelly-roll beta-barrel core); soluble matrix protein; no TM helices; no FAD domain",
            "module":   "Late-stage CI assembly — Q-module maturation; post-translational hydroxylation of NDUFS7/PSST subunit; distinct from N-module (FOXRED1/NUBPL), MCIA/ND2-ND5 (ACAD9/NDUFAF1/ECSIT/TMEM126B), and ND1-module (NDUFAF3/4/5/TIMMDC1)",
            "function": (
                "NDUFAF6/C8orf38 is the ONLY CI assembly factor with established 2OG-Fe(II) oxygenase/"
                "hydroxylase enzymatic activity. It catalyzes post-translational hydroxylation of the "
                "NDUFS7/PSST subunit (Q-module) using 2-oxoglutarate as co-substrate and Fe(II) as cofactor. "
                "This hydroxylation is required for Q-module maturation and late-stage CI assembly. "
                "Loss of NDUFAF6 stalls late-stage CI assembly; Q-module sub-assembly intermediates "
                "accumulate; holoenzyme CI cannot form, causing isolated CI deficiency (5-20%). "
                "No FAD domain — riboflavin supplementation cannot rescue the 2OG-dioxygenase defect."
            ),
        },

        "key_pathway_note": (
            "NDUFAF6/C8orf38 (8q22.1) is the ONLY CI assembly factor with confirmed 2OG-Fe(II) "
            "oxygenase/hydroxylase enzymatic activity. It acts at a late stage of CI assembly — "
            "Q-module maturation — by hydroxylating the NDUFS7/PSST subunit (post-translational modification). "
            "This is mechanistically unique: all other CI assembly factors (FOXRED1, NUBPL, ACAD9, NDUFAF1, "
            "ECSIT, TMEM126B, NDUFAF3/4/5, TIMMDC1) are chaperones, scaffolds, or structural factors — "
            "none catalyze a post-translational modification of a CI subunit. "
            "NO riboflavin response (no FAD domain). NO peripheral neuropathy. NO leukodystrophy. "
            "Low HCM (<5-10%) — important DDx vs TIMMDC1 (>80%), NDUFV2 (80%), ACAD9 (55-65%). "
            "NDUFAF6 8q22.1 vs ACAD9 3q21.3 — riboflavin response the key distinguisher. "
            "NDUFAF6 8q22.1 vs NDUFAF5 20p12.1 — both no-FAD, no riboflavin; WES mandatory."
        ),

        "biochemical_fingerprint": {
            "Complex_I":              "5–20% of control (SEVERE isolated deficiency)",
            "Complex_II":             "Normal (100%)",
            "Complex_III":            "Normal (100%)",
            "Complex_IV":             "Normal (100%)",
            "Complex_V":              "Normal (100%)",
            "Pattern":                "ISOLATED CI deficiency — late-stage Q-module assembly block; NDUFS7/PSST hydroxylation absent; Q-module maturation stalled",
            "Riboflavin_response":    "NONE (0%) — NDUFAF6 is a 2OG-dioxygenase with NO FAD domain; riboflavin cannot rescue hydroxylation defect",
            "BN-PAGE_class":          "Late-stage Q-module/assembly intermediates stalled; different from N-module (FOXRED1/NUBPL) and MCIA/ND2-ND5 class (ACAD9); holoenzyme CI absent/severely reduced",
            "2OG_dioxygenase_unique": "NDUFAF6 is the ONLY CI assembly factor with 2OG-Fe(II) oxygenase activity — post-translational hydroxylation of NDUFS7/PSST subunit; no other CI factor catalyzes a subunit PTM",
            "HCM_rate":               "LOW <5-10% — critical DDx vs TIMMDC1 (>80%), NDUFV2 (~80%), ACAD9 (55-65%), SCO2 (65%)",
            "Late_stage_assembly":    "Q-module maturation step — after N-module and ND1-module assembly; NDUFS7/PSST hydroxylation required for late CI maturation",
        },

        "feature_frequencies_pct": ff,

        "ndufaf6_module_summary": {
            "gene":              "NDUFAF6 (C8orf38, 8q22.1)",
            "module_class":      "Late-stage CI assembly — 2OG-Fe(II)-dependent dioxygenase; Q-module maturation factor; post-translational CI subunit hydroxylase",
            "assembly_position": "Late-stage Q-module maturation — NDUFS7/PSST subunit hydroxylation; after N-module and ND1-module assembly; before final holoenzyme CI integration",
            "unique_2og_dioxygenase": (
                "NDUFAF6 is the ONLY CI assembly factor with confirmed 2-oxoglutarate Fe(II)-dependent "
                "oxygenase/hydroxylase enzymatic activity. Using 2OG as co-substrate and Fe(II) as cofactor, "
                "NDUFAF6 hydroxylates a leucyl or asparaginyl residue of NDUFS7/PSST (Q-module subunit). "
                "This post-translational modification is required for NDUFS7 to integrate properly into "
                "the Q-module and for late-stage CI assembly to proceed to holoenzyme formation. "
                "All other CI assembly factors act as chaperones, scaffolds, Fe-S delivery factors, "
                "or structural assembly factors — none catalyze a covalent PTM of a CI subunit."
            ),
            "ndufaf6_vs_foxred1": (
                "FOXRED1 is an FAD-oxidoreductase protein chaperone that facilitates N-module "
                "protein folding (non-covalent assistance). NDUFAF6 is a 2OG-dioxygenase that "
                "covalently modifies (hydroxylates) NDUFS7/PSST (Q-module). Different modules "
                "(N-module vs Q-module), different biochemistry (chaperone vs hydroxylase), "
                "different chromosomes (11q24.2 vs 8q22.1). Both have 0% riboflavin response "
                "(FOXRED1 has FAD domain but no response; NDUFAF6 has no FAD domain at all)."
            ),
            "ndufaf6_vs_nubpl": (
                "NUBPL delivers [4Fe-4S] clusters to N-module Fe-S subunits (NDUFS1, NDUFV1) — "
                "a cofactor insertion step. NDUFAF6 hydroxylates NDUFS7/PSST (Q-module) — "
                "a covalent PTM. Different modules (N-module vs Q-module), completely different "
                "biochemistry ([4Fe-4S] delivery vs 2OG-dioxygenase hydroxylation). "
                "Both have 0% riboflavin response. WES mandatory: 14q12 (NUBPL) vs 8q22.1 (NDUFAF6)."
            ),
            "ndufaf6_vs_acad9": (
                "ACAD9 is an FAD-binding MCIA scaffold (ND2-ND5/Class1) — riboflavin-responsive 50-60% (Level B). "
                "NDUFAF6 is a 2OG-dioxygenase (Q-module late stage) — 0% riboflavin response (no FAD domain). "
                "Riboflavin trial is the KEY clinical distinguisher. Different modules, different chromosomes "
                "(3q21.3 vs 8q22.1). ACAD9 HCM 55-65%; NDUFAF6 HCM <5-10%."
            ),
        },

        "key_ddx": [
            {
                "feature":     "NDUFAF6 (8q22.1) vs ACAD9 (3q21.3) — NO RIBOFLAVIN RESPONSE — Critical DDx",
                "significance": (
                    "ACAD9 deficiency is riboflavin-responsive (50-60%, Level B evidence) — riboflavin "
                    "is first-line treatment. NDUFAF6 deficiency has ZERO riboflavin response — "
                    "NDUFAF6 is a 2OG-dioxygenase with NO FAD domain; riboflavin cannot rescue "
                    "the NDUFS7/PSST hydroxylation defect. ACAD9: MCIA/ND2-ND5 (Class 1). "
                    "NDUFAF6: 2OG-dioxygenase, Q-module late stage. ACAD9 HCM 55-65%; NDUFAF6 HCM <5-10%. "
                    "Riboflavin trial + WES (3q21.3 vs 8q22.1) is the mandatory discriminator."
                ),
                "target_gene": "ACAD9",
            },
            {
                "feature":     "NDUFAF6 (8q22.1) vs NDUFAF5 (20p12.1) — Both No-FAD, No-Riboflavin — WES Mandatory",
                "significance": (
                    "Both NDUFAF6 and NDUFAF5 have no FAD domain and 0% riboflavin response. "
                    "KEY DIFFERENCES: (1) NDUFAF5 = ND1-module/Class3. NDUFAF6 = 2OG-dioxygenase/Q-module late stage. "
                    "Different assembly modules and BN-PAGE stalling patterns. "
                    "(2) Different chromosomes: 8q22.1 (NDUFAF6) vs 20p12.1 (NDUFAF5). "
                    "WES is the mandatory discriminator when riboflavin response is absent."
                ),
                "target_gene": "NDUFAF5",
            },
            {
                "feature":     "NDUFAF6 vs TIMMDC1 (3q25.1) — HCM <5-10% vs HCM >80%",
                "significance": (
                    "TIMMDC1 deficiency: HCM >80% (highest in CI assembly factors; integral IMM, ND1-module Class 3). "
                    "NDUFAF6 deficiency: HCM <5-10% (very low). Prominent HCM (>60%) in a CI patient points "
                    "strongly toward TIMMDC1 and away from NDUFAF6. "
                    "Completely different assembly modules: NDUFAF6 2OG-dioxygenase/Q-module vs TIMMDC1 integral-IMM/ND1-module."
                ),
                "target_gene": "TIMMDC1",
            },
            {
                "feature":     "NDUFAF6 vs NDUFV1 (11q13.2) — NO Leukodystrophy",
                "significance": (
                    "NDUFV1 deficiency: leukodystrophy 40-50%. NDUFAF6: 0% leukodystrophy. "
                    "Leukodystrophy on MRI strongly points away from NDUFAF6 toward NDUFV1 or other white matter diseases. "
                    "WES mandatory: 8q22.1 (NDUFAF6) vs 11q13.2 (NDUFV1)."
                ),
                "target_gene": "NDUFV1",
            },
            {
                "feature":     "NDUFAF6 vs NDUFS1 (2q33.3) — NO Peripheral Neuropathy",
                "significance": (
                    "NDUFS1 deficiency causes peripheral neuropathy in ~50% of patients — a hallmark feature. "
                    "NDUFAF6 deficiency: peripheral neuropathy 0%. Peripheral neuropathy in a CI patient "
                    "points away from NDUFAF6 toward NDUFS1. WES mandatory: 2q33.3 vs 8q22.1."
                ),
                "target_gene": "NDUFS1",
            },
            {
                "feature":     "NDUFAF6 vs NDUFS4 (5q11.2) — NO Olfactory Bulb Lesions",
                "significance": (
                    "NDUFS4 deficiency causes pathognomonic bilateral olfactory bulb lesions on MRI (52-65%). "
                    "NDUFAF6 deficiency: olfactory bulb lesions 0%. Olfactory bulb MRI lesions "
                    "point away from NDUFAF6 strongly toward NDUFS4."
                ),
                "target_gene": "NDUFS4",
            },
            {
                "feature":     "NDUFAF6 vs POLG/DGUOK — NO Hepatopathy",
                "significance": (
                    "POLG (15q26.1) and DGUOK (2p13.1) cause hepatopathy. "
                    "NDUFAF6 deficiency: hepatopathy 0%. Hepatopathy points away from NDUFAF6."
                ),
                "target_gene": "POLG / DGUOK",
            },
            {
                "feature":     "NDUFAF6 vs FOXRED1 (11q24.2) / NUBPL (14q12) — 2OG-Dioxygenase vs N-Module Chaperones",
                "significance": (
                    "FOXRED1 (FAD-oxidoreductase chaperone, N-module) and NUBPL ([4Fe-4S] delivery, N-module) "
                    "are N-module factors. NDUFAF6 acts on a different module (Q-module, late stage) via a "
                    "completely different biochemical mechanism (2OG-dependent hydroxylation). "
                    "BN-PAGE stalling patterns differ. All three have 0% riboflavin response. "
                    "WES is mandatory: 11q24.2 / 14q12 vs 8q22.1."
                ),
                "target_gene": "FOXRED1 / NUBPL",
            },
        ],

        "clinical_alerts": [
            "🔴 METFORMIN — ABSOLUTE CONTRAINDICATION: Direct CI inhibitor. NDUFAF6 patients have 5-20% CI; metformin biguanide inhibition is immediately life-threatening.",
            "🔴 VALPROATE (VPA) — ABSOLUTE CONTRAINDICATION: Triple mechanism: CoA sequestration, POLG inhibition, MT-ND subunit expression suppression. Causes further CI collapse.",
            "🔴 LINEZOLID — ABSOLUTE CONTRAINDICATION: 23S rRNA inhibition blocks all 7 mtDNA-encoded CI subunits (MT-ND1–6). No CI rescue possible.",
            "🔴 CHLORAMPHENICOL — ABSOLUTE CONTRAINDICATION: Same mitoribosomal 23S rRNA mechanism as linezolid.",
            "🔴 KETOGENIC DIET — CONTRAINDICATED: CI absent/minimal; NADH cannot be reoxidised via ETC. Beta-oxidation generates NADH that cannot be cleared — metabolic crisis.",
            "🟠 PROPOFOL — AVOID (PRIS risk): Propofol infusion syndrome via CIV inhibition + fatty acid oxidation uncoupling. Use sevoflurane for anaesthesia.",
            "🟡 PHENOBARBITAL — HIGH CAUTION: Secondary CI inhibitor effect; prefer LEV (levetiracetam) as first-choice AED — renal excretion, no mitochondrial toxicity.",
            "🟡 RIBOFLAVIN — NOT INDICATED FOR NDUFAF6: NDUFAF6 is a 2OG-Fe(II)-dependent dioxygenase with NO FAD domain; riboflavin CANNOT rescue the NDUFS7/PSST hydroxylation defect. Do not treat as ACAD9 (riboflavin Level B). Critical DDx distinction.",
            "🟢 SUCCINATE — Level C: CII substrate bypasses stalled CI entirely; allows CII → CIII → CIV electron flow; partial ATP rescue.",
            "🟢 CoQ10 (Ubiquinol) — Level C: Antioxidant + electron carrier support; standard add-on.",
            "🟢 THIAMINE B1 — Level C MANDATORY EMPIRIC: Exclude SLC19A3 (thiamine transporter) and BTD (biotin-thiamine-responsive basal ganglia disease) BEFORE confirming NDUFAF6.",
            "🟢 BIOTIN — Level C MANDATORY EMPIRIC: Exclude BTD (biotinidase deficiency) and HLCS BEFORE CI gene panel.",
            "🟢 CARNITINE — Level C: Supplement if secondary carnitine deficiency documented.",
            "🔵 NDUFAF6 (8q22.1) — ONLY CI Assembly Factor with 2OG-Fe(II) Oxygenase/Hydroxylase Activity: All other CI assembly factors are chaperones/scaffolds. NDUFAF6 uniquely catalyzes post-translational hydroxylation of NDUFS7/PSST — a covalent PTM of a CI subunit. This biochemical distinction is clinically and mechanistically unique.",
            "🔵 NO RIBOFLAVIN RESPONSE — Critical DDx: Any riboflavin response favors ACAD9 (50-60%, Level B). NDUFAF6 = 0% response. Riboflavin trial is the first-line clinical discriminator before WES results are available.",
            "🔵 LOW HCM (<5-10%): Very low HCM rate — echocardiography at diagnosis. HCM <10% points away from TIMMDC1 (>80%), NDUFV2 (80%), ACAD9 (55-65%), SCO2 (65%). HCM rate is a critical DDx marker.",
            "🔵 NDUFAF6 8q22.1 vs NDUFAF5 20p12.1: Both no-FAD, no riboflavin response. WES mandatory to distinguish — different chromosomes; different assembly modules (Q-module vs ND1-module).",
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
        "patients":           PATIENTS,
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
        "key_2og_dioxygenase_features": {
            "Only_CI_assembly_factor_with_2OG_dioxygenase_activity": True,
            "Post_translational_hydroxylation_NDUFS7_PSST":          True,
            "Q_module_late_stage_CI_assembly":                        True,
            "No_FAD_domain_no_riboflavin_response":                   True,
            "Soluble_matrix_no_TM_helices":                           True,
            "Isolated_CI_deficiency_only":                            True,
            "HCM_rate_very_low_pct":                                  round(sum(p["hcm"] for p in PATIENTS) / N_PATIENTS * 100),
            "Jelly_roll_beta_barrel_2OG_dioxygenase_fold":            True,
        },
        "treatment_summary": {
            "absolute_ci": ["Metformin", "Valproate (VPA)", "Linezolid", "Chloramphenicol", "Ketogenic Diet"],
            "avoid":       ["Propofol (PRIS)", "Phenobarbital (high caution)"],
            "do_not_use":  ["Riboflavin (NOT indicated — NDUFAF6 is a 2OG-dioxygenase with no FAD domain; riboflavin cannot rescue NDUFS7/PSST hydroxylation defect; critical DDx vs ACAD9)"],
            "level_c":     ["Succinate (CII bypass)", "CoQ10-Ubiquinol", "Thiamine B1 (MANDATORY empiric)", "Biotin (MANDATORY empiric)", "Carnitine"],
            "preferred_aed": "LEV (levetiracetam) — renal excretion, no mitochondrial toxicity",
            "anaesthesia": "Sevoflurane (NOT propofol — PRIS risk)",
            "diagnostic_priority": "2OG-dioxygenase activity assay; WES 8q22.1 locus; riboflavin trial (expected 0% response distinguishes from ACAD9)",
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
                "ndufaf6":        "No FAD domain; 0% riboflavin response; 2OG-dioxygenase; Q-module; HCM <5-10%",
                "comparator_val": "FAD-binding; riboflavin-responsive 50-60% (Level B); MCIA/ND2-ND5; HCM 55-65%",
                "key_test":       "Riboflavin trial + WES locus (8q22.1 vs 3q21.3)",
            },
            {
                "comparator":     "NDUFAF5 (20p12.1)",
                "ndufaf6":        "2OG-dioxygenase; Q-module late stage; 8q22.1; hydroxylase activity",
                "comparator_val": "No FAD, no 2OG activity; ND1-module/Class3; 20p12.1; scaffold function",
                "key_test":       "WES locus (8q22.1 vs 20p12.1) + BN-PAGE stalling pattern",
            },
            {
                "comparator":     "TIMMDC1 (3q25.1)",
                "ndufaf6":        "Soluble matrix; HCM <5-10%; Q-module late stage",
                "comparator_val": "Integral IMM (2 TM helices); HCM >80%; ND1-module/Class3",
                "key_test":       "HCM rate + echocardiography + BN-PAGE class + WES locus",
            },
            {
                "comparator":     "FOXRED1 (11q24.2)",
                "ndufaf6":        "2OG-dioxygenase; Q-module; 8q22.1; no FAD domain",
                "comparator_val": "FAD-oxidoreductase chaperone; N-module; 11q24.2; FAD domain (no response)",
                "key_test":       "BN-PAGE assembly module + WES locus (8q22.1 vs 11q24.2)",
            },
            {
                "comparator":     "NDUFS1 (2q33.3)",
                "ndufaf6":        "NO peripheral neuropathy; 2OG-dioxygenase assembly factor",
                "comparator_val": "Peripheral neuropathy ~50%; structural N-module subunit",
                "key_test":       "Clinical neurophysiology (nerve conduction study)",
            },
            {
                "comparator":     "NDUFV1 (11q13.2)",
                "ndufaf6":        "NO leukodystrophy; soluble assembly factor; 8q22.1",
                "comparator_val": "Leukodystrophy 40-50%; FMN-binding structural N-module subunit; 11q13.2",
                "key_test":       "MRI leukodystrophy + WES locus",
            },
        ],
        "ndufaf6_module_summary": {
            "gene":              "NDUFAF6 (C8orf38, 8q22.1)",
            "module_class":      "Late-stage CI assembly — 2OG-Fe(II)-dependent dioxygenase; Q-module maturation hydroxylase; post-translational CI subunit modifier",
            "assembly_position": "Late-stage Q-module maturation — NDUFS7/PSST subunit hydroxylation; after N-module and ND1-module assembly; before final holoenzyme CI integration",
            "unique_2og_dioxygenase": (
                "NDUFAF6 is the ONLY CI assembly factor with confirmed 2-oxoglutarate Fe(II)-dependent "
                "oxygenase/hydroxylase enzymatic activity. Using 2OG as co-substrate and Fe(II) as cofactor, "
                "NDUFAF6 hydroxylates a leucyl or asparaginyl residue of NDUFS7/PSST (Q-module subunit). "
                "This post-translational modification is required for NDUFS7 to integrate properly into "
                "the Q-module and for late-stage CI assembly to proceed to holoenzyme formation. "
                "All other CI assembly factors act as chaperones, scaffolds, Fe-S delivery factors, "
                "or structural assembly factors — none catalyze a covalent PTM of a CI subunit."
            ),
            "ndufaf6_vs_foxred1": (
                "FOXRED1 is an FAD-oxidoreductase protein chaperone that facilitates N-module "
                "protein folding (non-covalent assistance). NDUFAF6 is a 2OG-dioxygenase that "
                "covalently modifies (hydroxylates) NDUFS7/PSST (Q-module). Different modules "
                "(N-module vs Q-module), different biochemistry (chaperone vs hydroxylase), "
                "different chromosomes (11q24.2 vs 8q22.1). Both have 0% riboflavin response."
            ),
            "ndufaf6_vs_nubpl": (
                "NUBPL delivers [4Fe-4S] clusters to N-module Fe-S subunits (NDUFS1, NDUFV1) — "
                "a cofactor insertion step. NDUFAF6 hydroxylates NDUFS7/PSST (Q-module) — "
                "a covalent PTM. Different modules (N-module vs Q-module), completely different "
                "biochemistry ([4Fe-4S] delivery vs 2OG-dioxygenase hydroxylation). "
                "Both have 0% riboflavin response. WES mandatory: 14q12 (NUBPL) vs 8q22.1 (NDUFAF6)."
            ),
            "ndufaf6_vs_acad9": (
                "ACAD9 is an FAD-binding MCIA scaffold (ND2-ND5/Class1) — riboflavin-responsive 50-60% (Level B). "
                "NDUFAF6 is a 2OG-dioxygenase (Q-module late stage) — 0% riboflavin response (no FAD domain). "
                "Riboflavin trial is the KEY clinical distinguisher. Different modules, different chromosomes "
                "(3q21.3 vs 8q22.1). ACAD9 HCM 55-65%; NDUFAF6 HCM <5-10%."
            ),
        },
    }


# ─── get_definitions ──────────────────────────────────────────────────────────
def get_definitions() -> dict:
    return {
        "gene_definition": (
            "NDUFAF6 (NADH:Ubiquinone Oxidoreductase Complex Assembly Factor 6; also known as C8orf38; "
            "OMIM *612392) encodes a 369-amino-acid, ~42 kDa soluble mitochondrial matrix protein "
            "belonging to the 2-oxoglutarate Fe(II)-dependent dioxygenase superfamily. "
            "NDUFAF6 is the ONLY CI assembly factor with established enzymatic activity — "
            "a 2OG-dependent hydroxylase that catalyzes post-translational modification (hydroxylation) "
            "of the NDUFS7/PSST subunit (Q-module), required for late-stage CI assembly and Q-module "
            "maturation. Loss-of-function variants cause isolated CI deficiency MC1DN26. "
            "No FAD domain, no TM helices — soluble matrix protein. No riboflavin response (0%)."
        ),
        "disease_definition": (
            "Mitochondrial Complex I Deficiency MC1DN26 (OMIM #614924) due to NDUFAF6 bi-allelic "
            "pathogenic variants presents as infantile-onset Leigh syndrome with profound isolated "
            "CI deficiency (5-20% of control), preserved CII/CIII/CIV, Leigh MRI (~72%), lactic "
            "acidosis (~85%), hypotonia (~90%), seizures (~54%), and very low HCM rate (<5-10%). "
            "BN-PAGE shows late-stage Q-module assembly intermediates stalled. "
            "No riboflavin response — critical DDx vs ACAD9 (50-60% riboflavin-responsive). "
            "McKenzie 2011 (AJHG) first identified NDUFAF6/C8orf38 as a CI assembly factor."
        ),
        "inheritance_definition": (
            "Autosomal recessive (AR). Biallelic pathogenic NDUFAF6 variants required. "
            "Both sexes equally affected. Consanguinity observed in ~36% of reported families. "
            "Carrier parents are clinically unaffected. Genetic counselling: 25% recurrence risk per pregnancy."
        ),
        "module_definitions": [
            {
                "term":       "NDUFAF6 as 2OG-Fe(II)-Dependent Dioxygenase — Unique CI Assembly Mechanism",
                "definition": (
                    "2-oxoglutarate (2OG) Fe(II)-dependent dioxygenases form a large superfamily that "
                    "catalyze oxidative reactions using 2OG as co-substrate and molecular oxygen, with Fe(II) "
                    "at the active site. The characteristic jelly-roll beta-barrel fold coordinates Fe(II) "
                    "via a HXD...H facial triad motif. NDUFAF6 uses this fold to hydroxylate a specific "
                    "leucyl or asparaginyl residue of the NDUFS7/PSST subunit of CI. This hydroxylation is "
                    "the only known post-translational modification of a CI subunit by a dedicated assembly "
                    "factor, and is required for Q-module maturation. p.Arg85Trp disrupts the Fe(II) "
                    "coordination/2OG-binding active site — the most direct pathogenic mechanism."
                ),
            },
            {
                "term":       "Q-Module Late-Stage CI Assembly and NDUFS7/PSST Hydroxylation",
                "definition": (
                    "The Q-module (quinone-reduction module) of CI contains NDUFS7 (PSST, 20kDa) and NDUFS2 "
                    "as the core quinone-binding and reduction subunits, positioned at the Q-channel interface "
                    "between the hydrophilic matrix arm and the membrane arm. NDUFAF6-mediated hydroxylation "
                    "of NDUFS7/PSST at a leucyl or asparaginyl residue is required for NDUFS7 to integrate "
                    "into the Q-module during late-stage CI assembly. Without this modification, Q-module "
                    "maturation stalls, and holoenzyme CI cannot form. This late-stage block is distinct from "
                    "N-module (FOXRED1/NUBPL) and ND1-module (NDUFAF3/4/5/TIMMDC1) assembly defects."
                ),
            },
            {
                "term":       "NDUFAF6 BN-PAGE — Late-Stage Q-Module Assembly Intermediates",
                "definition": (
                    "Blue-native PAGE (BN-PAGE) in NDUFAF6 patient fibroblasts/muscle shows accumulation "
                    "of late-stage CI assembly intermediates consistent with Q-module maturation stall. "
                    "This differs from: (1) N-module stalling (FOXRED1/NUBPL — smaller N-module intermediates), "
                    "(2) MCIA/ND2-ND5 stalling (ACAD9/NDUFAF1/ECSIT/TMEM126B — Class 1 intermediates), "
                    "and (3) ND1-module/Class3 stalling (NDUFAF3/4/5/TIMMDC1). "
                    "The Q-module/late-stage intermediate pattern is a biochemical fingerprint that, "
                    "combined with no riboflavin response and absent HCM, narrows the differential to "
                    "the NDUFAF6 (or other late-stage Q-module factors) category."
                ),
            },
            {
                "term":       "NDUFAF6 vs FOXRED1 — 2OG-Dioxygenase vs FAD-Oxidoreductase — Same No-Riboflavin Class, Different Biochemistry",
                "definition": (
                    "Both NDUFAF6 and FOXRED1 have 0% riboflavin response, but for different reasons: "
                    "FOXRED1 has an FAD-binding domain (H17 FAD-oxidoreductase) but cannot convert FAD "
                    "supplementation into improved CI assembly (chaperone function, not redox catalysis). "
                    "NDUFAF6 has NO FAD domain at all — it is a 2OG-dioxygenase. "
                    "FOXRED1 acts on the N-module (protein folding chaperone); NDUFAF6 acts on the Q-module "
                    "(NDUFS7 hydroxylation, late stage). Different chromosomes: 11q24.2 vs 8q22.1. "
                    "WES is mandatory to distinguish when both present as isolated CI deficiency with no riboflavin response."
                ),
            },
            {
                "term":       "NDUFAF6 vs ACAD9 — 2OG-Dioxygenase vs FAD-Scaffold — The Riboflavin Test",
                "definition": (
                    "ACAD9 deficiency is riboflavin-responsive (50-60%, Level B evidence). NDUFAF6 deficiency "
                    "has 0% riboflavin response. The riboflavin trial is the single most important clinical "
                    "discriminator before WES results are available: "
                    "(1) If riboflavin response is observed → ACAD9 (MCIA/ND2-ND5/Class1). "
                    "(2) If no riboflavin response → continue WES workup (NDUFAF6, NDUFAF5, TIMMDC1, "
                    "FOXRED1, NUBPL, and other non-FAD CI assembly factors). "
                    "ACAD9 HCM 55-65%; NDUFAF6 HCM <5-10% — HCM rate reinforces the distinction. "
                    "Different chromosomes: 3q21.3 vs 8q22.1."
                ),
            },
            {
                "term":       "NDUFAF6 vs NDUFAF5 — Both No-FAD, Both No-Riboflavin — WES Mandatory",
                "definition": (
                    "NDUFAF6 and NDUFAF5 share: no FAD domain, 0% riboflavin response, soluble matrix localization, "
                    "isolated CI deficiency. KEY DISTINCTIONS: "
                    "(1) NDUFAF6 = 2OG-dioxygenase / Q-module maturation. NDUFAF5 = scaffold function / ND1-module Class3. "
                    "Different biochemistry and different CI assembly stages. "
                    "(2) BN-PAGE: NDUFAF6 shows late-stage Q-module stalling. NDUFAF5 shows ND1-module/Class3 stalling. "
                    "(3) Chromosomes: 8q22.1 vs 20p12.1. WES is the definitive discriminator."
                ),
            },
            {
                "term":       "Low HCM Rate in NDUFAF6 — Critical DDx Marker",
                "definition": (
                    "NDUFAF6 deficiency shows very low HCM (<5-10%) — among the lowest of all CI assembly factors. "
                    "Comparison: TIMMDC1 >80%, NDUFV2 ~80%, ACAD9 55-65%, SCO2 65%, NUBPL ~25%, "
                    "FOXRED1 ~10%, NDUFAF6 <5-10%. "
                    "Very low HCM in a CI-deficiency patient with no riboflavin response points toward "
                    "NDUFAF6 (or other non-FAD, low-HCM CI assembly factors like NDUFAF5, FOXRED1). "
                    "Conversely, high HCM (>60%) in a CI patient points strongly away from NDUFAF6."
                ),
            },
            {
                "term":       "Absolute Contraindications in CI Deficiency — NDUFAF6",
                "definition": (
                    "Five absolute contraindications apply to ALL CI deficiency including NDUFAF6: "
                    "(1) METFORMIN — direct CI inhibitor; biguanide directly blocks complex I. "
                    "(2) VALPROATE (VPA) — triple mechanism: CoA trapping, POLG toxicity, MT-ND subunit suppression. "
                    "(3) LINEZOLID — 23S rRNA inhibitor blocks all 7 mtDNA-encoded ND subunits. "
                    "(4) CHLORAMPHENICOL — same mitoribosomal mechanism as linezolid. "
                    "(5) KETOGENIC DIET — CI absent; NADH cannot be reoxidized; NADH accumulation → metabolic crisis. "
                    "PROPOFOL: AVOID (PRIS risk — CIV inhibition + FA oxidation uncoupling). "
                    "Use sevoflurane for anaesthesia."
                ),
            },
            {
                "term":       "Supportive Treatments — NDUFAF6 CI Deficiency",
                "definition": (
                    "Supportive treatments follow standard CI deficiency protocols (all Level C evidence): "
                    "(1) SUCCINATE — CII substrate bypasses stalled CI; allows CII→CIII→CIV electron flow. "
                    "(2) CoQ10/UBIQUINOL — antioxidant + electron carrier support. "
                    "(3) THIAMINE B1 — MANDATORY empiric to exclude SLC19A3/BTD before confirming NDUFAF6. "
                    "(4) BIOTIN — MANDATORY empiric to exclude BTD/HLCS before CI gene panel. "
                    "(5) CARNITINE — supplement if secondary carnitine deficiency documented. "
                    "(6) LEV (levetiracetam) — preferred AED (renal excretion, no mitochondrial toxicity). "
                    "(7) IV Dextrose GIR 6-8 mg/kg/min — never fast; glucose prevents metabolic decompensation. "
                    "RIBOFLAVIN: NOT INDICATED (NDUFAF6 has no FAD domain; riboflavin cannot rescue 2OG-dioxygenase defect)."
                ),
            },
            {
                "term":       "McKenzie 2011 — First NDUFAF6/C8orf38 CI Assembly Factor Identification",
                "definition": (
                    "McKenzie M et al. (2011) identified C8orf38 (NDUFAF6) as a CI assembly factor required "
                    "for production of the mitochondria-encoded subunit ND1 and for CI holoenzyme assembly. "
                    "This foundational paper established: (1) NDUFAF6/C8orf38 is a genuine CI assembly factor; "
                    "(2) Loss-of-function mutations in NDUFAF6 cause isolated CI deficiency; "
                    "(3) The gene maps to chromosome 8q22.1; (4) Encoded protein has 2OG-dioxygenase fold. "
                    "Subsequent studies (Guerrero-Castillo 2017) contextualized NDUFAF6 within the late-stage "
                    "CI assembly pathway and Q-module maturation."
                ),
            },
            {
                "term":       "NDUFAF6 Chromosomal Locus 8q22.1 — WES Diagnostic Context",
                "definition": (
                    "NDUFAF6 maps to chromosome 8q22.1. Key chromosomal DDx: "
                    "NDUFAF6 (8q22.1) vs ACAD9 (3q21.3) — different chromosomes; riboflavin response key test. "
                    "NDUFAF6 (8q22.1) vs NDUFAF5 (20p12.1) — different chromosomes; both no riboflavin. "
                    "NDUFAF6 (8q22.1) vs FOXRED1 (11q24.2) — different chromosomes; both no riboflavin. "
                    "NDUFAF6 (8q22.1) vs NUBPL (14q12) — different chromosomes; both no riboflavin. "
                    "NDUFAF6 (8q22.1) vs TIMMDC1 (3q25.1) — different chromosomes; HCM key DDx. "
                    "WES with chromosomal locus verification is mandatory when riboflavin response is absent."
                ),
            },
            {
                "term":       "2OG-Dioxygenase Active Site — p.Arg85Trp and p.Gly59Arg Mechanisms",
                "definition": (
                    "The 2OG-Fe(II)-dependent dioxygenase active site contains a conserved HXD...H facial triad "
                    "that coordinates Fe(II) in octahedral geometry, with 2OG occupying two coordination sites "
                    "and the substrate hydroxylation site positioned for oxygen activation. "
                    "p.Arg85Trp (c.253C>T): Arg85 is a conserved active-site residue involved in 2OG coordination "
                    "or substrate positioning; Trp substitution disrupts the active-site geometry — Fe(II) binding "
                    "or 2OG binding is impaired; catalytic activity abolished; severe CI deficiency. "
                    "p.Gly59Arg (c.175G>C): Gly59 is in the conserved jelly-roll beta-barrel core; "
                    "Arg substitution introduces steric clash that disrupts the fold; protein unstable."
                ),
            },
            {
                "term":       "Helix-Breaking Proline — p.Leu213Pro Mechanism",
                "definition": (
                    "p.Leu213Pro (c.638T>C) introduces a proline at position 213 within an alpha-helix of "
                    "the 2OG-dioxygenase fold. Proline lacks the NH backbone hydrogen bond donor and introduces "
                    "a rigid kink — alpha-helices cannot accommodate proline in most positions without severe "
                    "distortion. In NDUFAF6, this helix is part of the structural framework supporting the "
                    "2OG-dioxygenase active site architecture. Helix collapse leads to protein misfolding, "
                    "NDUFAF6 degradation, and loss of NDUFS7/PSST hydroxylation. "
                    "Helix-breaking proline is a recurrent severe pathogenic mechanism across CI assembly factors "
                    "(p.Leu213Pro in NDUFAF6; p.Leu104Pro in NUBPL; p.Leu357Pro in FOXRED1; p.Leu149Pro in TIMMDC1)."
                ),
            },
            {
                "term":       "Intermediate Variant p.Ala287Val — Partial 2OG-Dioxygenase Activity",
                "definition": (
                    "p.Ala287Val (c.860C>T) in the C-terminal domain of NDUFAF6 disrupts hydrophobic core packing. "
                    "The Ala→Val substitution adds a methyl group that perturbs local tertiary structure without "
                    "completely unfolding the protein. Residual NDUFAF6 protein with partial 2OG-dioxygenase "
                    "activity is retained. This explains the intermediate severity: CI activity 14-22% (higher "
                    "than severe alleles 5-12%), later onset (3-18 months), and better prognosis than biallelic "
                    "severe variants. Compound heterozygosity with a severe allele produces intermediate CI deficiency."
                ),
            },
            {
                "term":       "Splice Donor Variant cIVS5+1G>A — Partial Normal Transcript",
                "definition": (
                    "cIVS5+1G>A disrupts the canonical splice donor at intron 5 of NDUFAF6. "
                    "Consequences depend on cryptic splice site activation: (1) Complete intron 5 retention → "
                    "frameshift/NMD; (2) Activation of a nearby cryptic splice site → partial intron 5 retention "
                    "with some exon inclusion preserved → hypomorphic NDUFAF6 mRNA/protein. "
                    "Partial normal transcript explains the moderate CI residual (18-28%), later infantile onset, "
                    "and variable clinical severity. This allele is analogous to c.815-27T>C in NUBPL — "
                    "a splice-affecting variant that allows partial NDUFAF6 function and modified disease course."
                ),
            },
        ],
        "clinical_thresholds": [
            {"parameter": "CI enzyme activity",             "threshold": "<20% of control",    "significance": "Diagnostic criterion for severe CI deficiency in muscle biopsy; NDUFAF6 typically 5-20%"},
            {"parameter": "Lactate (plasma)",               "threshold": ">2.5 mmol/L",        "significance": "Elevated; confirms mitochondrial dysfunction (non-specific)"},
            {"parameter": "Lactate:pyruvate ratio",         "threshold": ">25",                "significance": "Elevated L:P supports NADH-block at CI (ETC defect rather than PDH or pyruvate metabolism)"},
            {"parameter": "CSF lactate",                    "threshold": ">2.5 mmol/L",        "significance": "Leigh syndrome workup — elevated in CI deficiency with CNS involvement"},
            {"parameter": "Riboflavin trial response",      "threshold": "0% (none expected)", "significance": "NDUFAF6 has NO FAD domain; 0% response expected. Any response strongly favors ACAD9 (FAD domain, Level B 50-60%)"},
            {"parameter": "HCM (echocardiography)",         "threshold": "<5-10% of cases",    "significance": "Very low HCM — critical DDx marker: TIMMDC1 >80%, NDUFV2 ~80%, ACAD9 55-65% all have much higher HCM rates"},
            {"parameter": "BN-PAGE CI intermediates",       "threshold": "Late-stage Q-module stalling", "significance": "Q-module maturation intermediates distinguish NDUFAF6 from N-module (FOXRED1/NUBPL) and MCIA (ACAD9) stalling patterns"},
            {"parameter": "2OG-dioxygenase activity assay", "threshold": "Reduced/absent",     "significance": "Functional confirmation of NDUFAF6 2OG-hydroxylase defect; available in specialized biochemistry labs"},
            {"parameter": "NDUFS7/PSST hydroxylation",      "threshold": "Absent",             "significance": "Direct substrate modification assay; confirms NDUFAF6 enzymatic defect at the NDUFS7 target residue"},
            {"parameter": "Onset age (severe alleles)",     "threshold": "<6 months",          "significance": "Biallelic severe alleles (p.Arg85Trp, p.Leu213Pro, p.Gly59Arg, p.Trp320Ter): neonatal-to-early-infantile onset"},
            {"parameter": "Onset age (moderate/inter.)",    "threshold": "3–18 months",        "significance": "p.Ala287Val (intermediate) or cIVS5+1G>A compound het allows later onset with partial 2OG-dioxygenase activity"},
            {"parameter": "WES chromosomal locus",          "threshold": "8q22.1",             "significance": "NDUFAF6 maps to 8q22.1; WES locus verification is mandatory when riboflavin response is absent to distinguish from NDUFAF5 (20p12.1), ACAD9 (3q21.3), FOXRED1 (11q24.2)"},
        ],
        "standards": [
            {"code": "ACMG/AMP 2015",         "title": "Variant classification guidelines — pathogenicity criteria for NDUFAF6"},
            {"code": "MITOMAP",               "title": "Mitochondrial disease gene/variant database"},
            {"code": "OMIM *612392",          "title": "NDUFAF6 gene entry — NADH:Ubiquinone Oxidoreductase Complex Assembly Factor 6 (C8orf38)"},
            {"code": "OMIM #614924",          "title": "Mitochondrial Complex I Deficiency MC1DN26 — NDUFAF6-related"},
            {"code": "REACTOME R-HSA-611105", "title": "Respiratory electron transport — CI Q-module assembly and late-stage CI maturation"},
            {"code": "Helsinki Declaration",  "title": "Ethical framework for human subject research"},
        ],
        "references": [
            {
                "id":        "mckenzie_2011",
                "citation":  "McKenzie M et al. (2011) Mutations in the gene encoding C8orf38 block complex I assembly by inhibiting production of the mitochondria-encoded subunit ND1. Am J Hum Genet 89(5):711–8.",
                "relevance": "First identification of NDUFAF6/C8orf38 as a CI assembly factor; establishes 8q22.1 locus; demonstrates that NDUFAF6 loss causes isolated CI deficiency with Leigh syndrome; maps NDUFAF6 to late-stage CI assembly pathway involving ND1 production and Q-module maturation.",
            },
            {
                "id":        "guerrero_castillo_2017",
                "citation":  "Guerrero-Castillo S et al. (2017) The assembly pathway of mitochondrial respiratory chain complex I. Cell Metab 25(1):128–139.",
                "relevance": "Comprehensive CI assembly intermediate map; contextualizes NDUFAF6 within the late-stage CI assembly pathway; defines temporal order of CI assembly factors; BN-PAGE intermediate classes (N-module, MCIA/ND2-ND5, ND1-module/Class3) provide framework for NDUFAF6 Q-module stalling.",
            },
            {
                "id":        "stroud_2016",
                "citation":  "Stroud DA et al. (2016) Accessory subunits are integral for assembly and function of human mitochondrial complex I. Nature 538(7623):123–6.",
                "relevance": "Defines CI assembly classes (Class 1 MCIA, Class 2 N-Q, Class 3 ND1-module); BN-PAGE stalling pattern reference for distinguishing NDUFAF6 Q-module/late-stage stall from Class 1/2/3 patterns; key reference for NDUFAF6 DDx vs ACAD9, NDUFAF5, TIMMDC1.",
            },
            {
                "id":        "lim_2016",
                "citation":  "Lim SC et al. (2016) A founder mutation in PET100 causes complex IV deficiency in Lebanese individuals with Leigh syndrome. Am J Hum Genet [comparative reference for isolated respiratory chain complex deficiency DDx framework].",
                "relevance": "Provides comparative framework for isolated respiratory chain complex deficiency diagnosis; illustrates importance of WES locus identification and biochemical fingerprinting in the Leigh syndrome DDx — same principles apply to NDUDAF6 CI deficiency.",
            },
            {
                "id":        "fassone_2012",
                "citation":  "Fassone E & Rahman S (2012) Complex I deficiency: clinical features, biochemistry and molecular genetics. J Med Genet 49(9):578–90.",
                "relevance": "Comprehensive review of CI deficiency genetics, biochemistry, and clinical features; places NDUFAF6 within the broader landscape of CI assembly factors; documents phenotypic spectrum (Leigh syndrome, lactic acidosis, HCM) and DDx framework for all known CI assembly factor deficiencies.",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps({"overview": get_overview(), "breakdown": get_breakdown(), "definitions": get_definitions()}, indent=2, default=str))
