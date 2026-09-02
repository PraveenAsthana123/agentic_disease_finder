#!/usr/bin/env python3
"""NUBPL (IND1) — Mitochondrial Complex I Deficiency (N-Module [4Fe-4S] Cluster Assembly Factor).

NUBPL (Nucleotide-binding protein-like; also known as IND1) is a [4Fe-4S] cluster assembly
factor dedicated exclusively to the N-module of respiratory complex I. It is the only
CI-specific iron-sulfur cluster delivery protein known, delivering [4Fe-4S] clusters to
the N-module Fe-S subunits (N1b, N3, N4, N5, N6a, N6b, N7 positions in NDUFS1/NDUFV1/NDUFV2).

  NUBPL gene   OMIM *613621
  Disease      Mitochondrial Complex I Deficiency (OMIM #256000)
  Inheritance  AR (autosomal recessive, biallelic)
  Chromosome   14q12

Reference: Calvo SE et al. (2010) High-throughput, pooled sequencing identifies mutations in
NUBPL and FOXRED1 in human complex I deficiency. Nat Genet 42(10):851–8.
(First NUBPL disease gene identification; p.Gly56Arg European founder + c.815-27T>C branch-point)
Reference: Kevelam SH et al. (2013) NUBPL mutations in patients with complex I deficiency
and a distinct MRI pattern. Neurology 80(17):1577–83.
Reference: Guerrero-Castillo S et al. (2017) The assembly pathway of mitochondrial respiratory
chain complex I. Cell Metab 25(1):128–139. (CI assembly intermediate mapping; N-module context)

PATHOPHYSIOLOGY (NUBPL / N-Module / Fe-S Cluster Delivery):
  NUBPL/IND1 is a CIA (cytosolic iron-sulfur protein assembly) machinery homolog
  (related to yeast Ind1p) that is mitochondrially targeted and CI-specific:
    1. NUBPL acquires [4Fe-4S] clusters from the mitochondrial Fe-S cluster biosynthesis
       machinery (ISC assembly, ISCU scaffold, NFS1/LYRM4 desulfurase complex).
    2. NUBPL delivers [4Fe-4S] clusters to the N-module subunits of CI:
       — NDUFS1 (75kDa): receives N1b, N3, N4, N5 clusters
       — NDUFV1 (51kDa): receives N3 cluster (shared with NDUFS1)
       — NDUFV2 (24kDa): receives N1a cluster (note: N1a is [2Fe-2S], not NUBPL-dependent)
       — NDUFS8, NDUFS7: N2 and proximal Fe-S clusters (Q-module/N-module boundary)
    3. Without [4Fe-4S] clusters, N-module subunits cannot fold/assemble properly;
       N-module sub-assembly stalls; holoenzyme CI cannot form.
    4. Isolated CI deficiency (5–20%); CII/CIII/CIV normal — block is N-module specific.
    5. P-loop Walker A GTPase domain: GTPase/ATPase hydrolysis is required for Fe-S transfer.
       p.Gly56Arg (European founder) disrupts the Walker A motif; partial residual activity.

NUBPL UNIQUE FEATURES vs OTHER CI ASSEMBLY FACTORS:
  1. ONLY CI-SPECIFIC [4Fe-4S] CLUSTER ASSEMBLY FACTOR:
     NUBPL/IND1 is the sole known factor that specifically delivers [4Fe-4S] clusters
     to the N-module of CI. No other CI assembly factor has this function. General
     mitochondrial Fe-S cluster assembly (ISC pathway: ISCU, NFS1) feeds into NUBPL.
  2. P-LOOP GTPASE/WALKER A DOMAIN — UNIQUE AMONG CI ASSEMBLY FACTORS:
     NUBPL contains a P-loop NTPase (Walker A motif) domain. The European founder variant
     p.Gly56Arg disrupts the Walker A GxGxxG sequence, impairing nucleotide hydrolysis
     that drives Fe-S cluster transfer to CI N-module subunits.
  3. N-MODULE — SAME AS FOXRED1, DIFFERENT MECHANISM:
     Both NUBPL and FOXRED1 act on the N-module. NUBPL delivers [4Fe-4S] clusters
     (structural electron-transfer cofactors); FOXRED1 is an FAD-oxidoreductase chaperone
     for N-module protein folding. Different biochemical steps; different BN-PAGE stalling
     pattern. Neither has riboflavin response (NUBPL: no FAD domain; FOXRED1: FAD but no response).
  4. EUROPEAN FOUNDER ALLELE (p.Gly56Arg) — DIAGNOSTIC SHORTCUT:
     p.Gly56Arg (c.166G>A) is a European founder allele (Scottish/British enrichment),
     found in ~70% of published NUBPL patients typically in compound heterozygosity
     with a second loss-of-function allele. A European patient with CI deficiency +
     p.Gly56Arg heterozygous should prompt urgent search for second NUBPL allele.
  5. DEEP INTRONIC BRANCH-POINT VARIANT (c.815-27T>C):
     The second allele in the original Calvo 2010 patients was a deep intronic variant
     (c.815-27T>C) affecting a branch point — causing aberrant splicing with partial
     normal transcript retained. This explains intermediate severity in compound hets.
  6. HCM ~25% — INTERMEDIATE BETWEEN FOXRED1 (~10%) AND TIMMDC1 (>80%):
     NUBPL HCM rate (~25%) is higher than FOXRED1 (~10%) but much lower than TIMMDC1 (>80%).
     Fe-S cluster deficiency in the N-module creates a different cardiomyocyte vulnerability
     profile than ND1-module integral-IMM factors.
  7. 14q12 — UNIQUE CHROMOSOMAL LOCUS for N-module CI assembly factors.

DISTINGUISHING FEATURES vs OTHER CI GENES:
  vs FOXRED1 (11q24.2): Both act on N-module. FOXRED1 = FAD-oxidoreductase chaperone.
    NUBPL = [4Fe-4S] cluster delivery. Different mechanisms, different BN-PAGE sub-steps.
    Both have 0% riboflavin response. FOXRED1 HCM ~10%; NUBPL HCM ~25%.
    NUBPL p.Gly56Arg is an European founder allele — diagnostic clue absent in FOXRED1.
    WES mandatory: 14q12 (NUBPL) vs 11q24.2 (FOXRED1).
  vs ACAD9 (3q21.3): ACAD9 = MCIA/ND2-ND5 module (Class 1); riboflavin-responsive 50-60%.
    NUBPL = N-module; 0% riboflavin response. ACAD9 HCM 55-65%; NUBPL ~25%.
    Riboflavin response is the KEY clinical distinguisher.
  vs NDUFV1 (11q13.2): NDUFV1 = structural N-module subunit; leukodystrophy 40-50%.
    NUBPL = N-module Fe-S delivery factor; NO leukodystrophy. WES mandatory.
  vs NDUFS1 (2q33.3): NDUFS1 peripheral neuropathy ~50%. NUBPL: 0%. Fe-S delivery
    to NDUFS1 is NUBPL-dependent but NDUFS1 loss/mutation has different phenotype.
  vs NDUFAF2 (5q12.1): NDUFAF2 = assembly-swap chaperone (NDUFA12 paralog).
    NUBPL = Fe-S cluster delivery. Different mechanisms, both N-module area.
  vs TIMMDC1 (3q25.1): TIMMDC1 HCM >80%; NUBPL ~25%. TIMMDC1 = ND1-module integral-IMM.
    NUBPL = N-module soluble matrix. Completely different CI module.
"""

import random
import math

SEED = 693
rng  = random.Random(SEED)

GENE         = "NUBPL"
OMIM_GENE    = "613621"
OMIM_DISEASE = "256000"
DISEASE_NAME = "Mitochondrial Complex I Deficiency (CI Deficiency; Leigh Syndrome spectrum)"
CHROMOSOME   = "14q12"
INHERITANCE  = "AR (biallelic)"
N_PATIENTS   = 40

# ─── Variants ────────────────────────────────────────────────────────────────
VARIANTS = [
    {
        "hgvs_c":    "c.166G>A",
        "hgvs_p":    "p.Gly56Arg",
        "domain":    "Walker A P-loop (GxGxxG motif) — GTPase/ATPase nucleotide-binding domain",
        "mechanism": "Glycine-to-arginine disrupts Walker A P-loop motif; nucleotide hydrolysis impaired; [4Fe-4S] transfer to N-module stalled",
        "severity":  "intermediate",
        "ci_pct_range": (12, 22),
        "notes":     "European founder allele (Scottish/British enrichment); ~70% of published NUBPL patients; typically compound het with second allele (c.815-27T>C or other); partial residual GTPase activity",
    },
    {
        "hgvs_c":    "c.311T>C",
        "hgvs_p":    "p.Leu104Pro",
        "domain":    "Alpha-helix within Fe-S scaffold domain",
        "mechanism": "Helix-breaking proline disrupts alpha-helix continuity in Fe-S cluster scaffold domain; protein misfolding; [4Fe-4S] cluster cannot be stably bound",
        "severity":  "severe",
        "ci_pct_range": (5, 12),
        "notes":     "Severe infantile onset; helix-breaking proline — common severe mechanism across CI assembly factors; no residual Fe-S delivery",
    },
    {
        "hgvs_c":    "c.815-27T>C",
        "hgvs_p":    "p.splice_branch_point_intron9",
        "domain":    "Deep intronic branch point — intron 9",
        "mechanism": "Branch point mutation causes aberrant splicing; partial exon 9 skipping + partial normal transcript retained → hypomorphic allele",
        "severity":  "moderate",
        "ci_pct_range": (16, 28),
        "notes":     "Calvo 2010 second allele; deep intronic — would be missed by exome capture without deep intronic analysis; partial normal transcript explains intermediate severity",
    },
    {
        "hgvs_c":    "c.693+1G>A",
        "hgvs_p":    "p.splice_donor_intron8",
        "domain":    "Splice donor — intron 8",
        "mechanism": "Splice donor disruption → intron 8 retention or exon 8 skipping → reading frame shifted/truncated; loss of C-terminal Fe-S delivery domain",
        "severity":  "moderate",
        "ci_pct_range": (18, 30),
        "notes":     "Splice donor loss; partial normal transcript may persist if cryptic splice site activated (moderate residual CI in some patients)",
    },
    {
        "hgvs_c":    "c.595C>T",
        "hgvs_p":    "p.Arg199Cys",
        "domain":    "Fe-S cluster insertion interface — recipient-subunit contact surface",
        "mechanism": "Arginine-to-cysteine disrupts electrostatic contact with CI N-module subunit binding surface; [4Fe-4S] cluster transfer to NDUFS1/NDUFV1 fails",
        "severity":  "severe",
        "ci_pct_range": (5, 13),
        "notes":     "Critical Fe-S insertion interface residue; direct disruption of N-module subunit contact; severe infantile encephalopathy",
    },
    {
        "hgvs_c":    "c.720G>A",
        "hgvs_p":    "p.Trp240Ter",
        "domain":    "C-terminal truncation — Fe-S delivery scaffold",
        "mechanism": "Premature termination codon truncates C-terminal Fe-S delivery scaffold domain; C-terminal region required for [4Fe-4S] cluster stabilization and transfer to CI",
        "severity":  "severe",
        "ci_pct_range": (4, 11),
        "notes":     "Null allele; neonatal-onset severe CI deficiency; compound heterozygosity with milder allele gives intermediate phenotype",
    },
]


def _variant(rng_):
    weights = [0.35, 0.18, 0.18, 0.12, 0.10, 0.07]
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

    leigh_mri    = ci_act < 20 and rng_.random() < 0.76
    lactic_ac    = rng_.random() < 0.86
    hypotonia    = rng_.random() < 0.91
    dev_regr     = rng_.random() < 0.76
    hcm          = rng_.random() < 0.25   # ~25% — intermediate, higher than FOXRED1 (~10%), lower than TIMMDC1 (>80%)
    seizures     = rng_.random() < 0.55
    resp_fail    = rng_.random() < 0.46
    # HARD ZEROs — critical DDx
    riboflavin_r = False   # NO riboflavin response — NUBPL has no FAD domain
    periph_nrp   = False   # NO peripheral neuropathy — critical DDx vs NDUFS1
    leukodystro  = False   # NO leukodystrophy — critical DDx vs NDUFV1
    hepatopathy  = False   # NO hepatopathy — critical DDx vs POLG/DGUOK
    olfact_bulb  = False   # NO olfactory bulb lesions — critical DDx vs NDUFS4

    sex  = rng_.choice(["M", "F"])
    fam  = "consanguineous" if rng_.random() < 0.38 else "non-consanguineous"

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
        "Riboflavin_responder":    0,   # HARD 0 — NO riboflavin response (no FAD domain in NUBPL)
        "Peripheral_neuropathy":   0,   # HARD 0 — critical DDx vs NDUFS1
        "Leukodystrophy":          0,   # HARD 0 — critical DDx vs NDUFV1
        "Hepatopathy":             0,   # HARD 0 — critical DDx vs POLG/DGUOK
        "Olfactory_bulb_lesions":  0,   # HARD 0 — critical DDx vs NDUFS4
    }

    return {
        "gene":            GENE,
        "gene_full_name":  "Nucleotide-binding protein-like (IND1 — CI-specific [4Fe-4S] cluster assembly factor)",
        "also_known_as":   "IND1 (iron-sulfur protein required for NADH dehydrogenase, human homolog of yeast Ind1p)",
        "omim_gene":       OMIM_GENE,
        "omim_disease":    OMIM_DISEASE,
        "disease_name":    DISEASE_NAME,
        "chromosome":      CHROMOSOME,
        "inheritance":     INHERITANCE,
        "cohort_n":        N_PATIENTS,
        "cohort_seed":     SEED,

        "protein": {
            "size_aa":  301,
            "size_kda": 34.0,
            "fold":     "P-loop NTPase (Walker A GxGxxG motif) / Mrp/NBP35-type Fe-S cluster scaffold; soluble matrix protein; no TM helices",
            "module":   "N-module (NADH dehydrogenase module) CI-specific [4Fe-4S] cluster delivery factor; distinct from FOXRED1 (N-module FAD-oxidoreductase chaperone) and MCIA/ND2-ND5 (ACAD9/NDUFAF1/ECSIT/TMEM126B)",
            "function": (
                "NUBPL/IND1 is the only known CI-specific [4Fe-4S] cluster assembly factor. "
                "It delivers [4Fe-4S] clusters to N-module Fe-S subunits of respiratory complex I: "
                "NDUFS1 (N1b, N3, N4, N5 clusters), NDUFV1 (N3 cluster). The P-loop Walker A "
                "GTPase domain (GxGxxG motif) hydrolyzes nucleotides to drive cluster transfer. "
                "Loss of NUBPL stalls N-module [4Fe-4S] cluster insertion; N-module sub-assembly "
                "cannot proceed and holoenzyme CI cannot form, causing isolated CI deficiency. "
                "p.Gly56Arg (European founder) disrupts the Walker A motif with partial residual activity."
            ),
        },

        "key_pathway_note": (
            "NUBPL/IND1 (14q12) is the only CI-specific [4Fe-4S] cluster delivery factor. "
            "It acts on the N-module (NADH dehydrogenase module — matrix arm tip), "
            "delivering [4Fe-4S] clusters to NDUFS1 (N1b/N3/N4/N5) and NDUFV1 (N3) subunits. "
            "This is mechanistically distinct from FOXRED1 (same N-module, FAD-oxidoreductase chaperone), "
            "MCIA complex (ACAD9/NDUFAF1/ECSIT/TMEM126B, ND2-ND5 module), and ND1-module factors "
            "(NDUFAF3/4/5, TIMMDC1). NO riboflavin response (NUBPL has no FAD domain). "
            "European founder allele p.Gly56Arg (c.166G>A) is found in ~70% of NUBPL patients — "
            "a key diagnostic clue in European CI-deficiency patients. "
            "HCM ~25% — intermediate between FOXRED1 (~10%) and TIMMDC1 (>80%)."
        ),

        "biochemical_fingerprint": {
            "Complex_I":              "5–20% of control (SEVERE isolated deficiency)",
            "Complex_II":             "Normal (100%)",
            "Complex_III":            "Normal (100%)",
            "Complex_IV":             "Normal (100%)",
            "Complex_V":              "Normal (100%)",
            "Pattern":                "ISOLATED CI deficiency — N-module [4Fe-4S] cluster delivery block; N-module Fe-S subunits unassembled",
            "Riboflavin_response":    "NONE (0%) — NUBPL has no FAD domain; riboflavin cannot rescue [4Fe-4S] delivery defect",
            "BN-PAGE_class":          "N-module sub-assembly intermediates stalled; [4Fe-4S]-depleted N-module subunits cannot assemble; holoenzyme CI absent/severely reduced",
            "Fe-S_delivery_defect":   "N-module [4Fe-4S] clusters (N1b, N3, N4, N5) absent or depleted in NDUFS1/NDUFV1 — unique fingerprint of NUBPL deficiency",
            "HCM_rate":               "INTERMEDIATE ~25% — between FOXRED1 (~10%) and TIMMDC1 (>80%); higher than most N-module chaperones",
            "European_founder":       "p.Gly56Arg (c.166G>A) in ~70% of NUBPL patients — Walker A P-loop founder allele; diagnostic shortcut",
        },

        "feature_frequencies_pct": ff,

        "nubpl_module_summary": {
            "gene":              "NUBPL (IND1, 14q12)",
            "module_class":      "N-module (NADH dehydrogenase) CI-specific [4Fe-4S] cluster delivery factor — P-loop GTPase/Mrp-type scaffold",
            "assembly_position": "N-module Fe-S cluster insertion (before holoenzyme assembly) — delivers N1b, N3, N4, N5 clusters to NDUFS1 and N3 to NDUFV1",
            "fes_delivery_unique": (
                "NUBPL is the ONLY known CI-specific [4Fe-4S] cluster delivery factor. "
                "General mitochondrial Fe-S cluster biosynthesis (ISC pathway: ISCU, NFS1/LYRM4) "
                "produces [4Fe-4S] clusters, but NUBPL specifically delivers these to CI N-module "
                "subunits. The Walker A P-loop GTPase activity is required for cluster transfer. "
                "p.Gly56Arg disrupts Walker A motif (GxGxxG) — the most critical catalytic element."
            ),
            "european_founder_role": (
                "p.Gly56Arg (c.166G>A) is a European founder allele (enriched in Scottish/British "
                "populations). It is found in ~70% of published NUBPL patients, almost always "
                "in compound heterozygosity with a second allele (c.815-27T>C deep intronic, "
                "or other LOF variant). The p.Gly56Arg allele retains partial Walker A activity "
                "(hypomorphic), explaining why compound hets can survive to later infantile onset. "
                "Any European CI-deficiency patient with heterozygous p.Gly56Arg should undergo "
                "deep intronic NUBPL sequencing to find the second allele."
            ),
            "nubpl_vs_foxred1_same_module": (
                "Both NUBPL and FOXRED1 act on the N-module but at different steps. "
                "NUBPL delivers [4Fe-4S] clusters (cofactors for electron transfer: N1b/N3/N4/N5). "
                "FOXRED1 is an FAD-oxidoreductase chaperone facilitating N-module protein folding. "
                "These are sequential and independent steps. Neither gene has riboflavin response. "
                "FOXRED1 HCM ~10%; NUBPL HCM ~25%. WES (14q12 vs 11q24.2) is mandatory."
            ),
        },

        "key_ddx": [
            {
                "feature":     "NUBPL (14q12) vs FOXRED1 (11q24.2) — Same N-Module, Different Mechanism, Higher HCM in NUBPL",
                "significance": (
                    "Both NUBPL and FOXRED1 are N-module CI assembly factors causing isolated CI deficiency "
                    "with Leigh syndrome. KEY DIFFERENCES: "
                    "(1) NUBPL = [4Fe-4S] cluster delivery (P-loop GTPase/Fe-S carrier). "
                    "FOXRED1 = FAD-oxidoreductase protein folding chaperone. Different biochemical steps. "
                    "(2) NUBPL HCM ~25%; FOXRED1 HCM ~10%. Higher HCM in NUBPL points toward NUBPL. "
                    "(3) NUBPL p.Gly56Arg European founder (~70% of patients) — absent in FOXRED1. "
                    "(4) BN-PAGE stalling pattern differs: NUBPL stalls at Fe-S cluster insertion; "
                    "FOXRED1 stalls at chaperone-facilitated protein folding step. "
                    "WES mandatory: 14q12 (NUBPL) vs 11q24.2 (FOXRED1). Different chromosomes."
                ),
                "target_gene": "FOXRED1",
            },
            {
                "feature":     "NUBPL (14q12) vs ACAD9 (3q21.3) — NO RIBOFLAVIN RESPONSE — Critical DDx",
                "significance": (
                    "ACAD9 deficiency is riboflavin-responsive (50-60%, Level B evidence) — riboflavin "
                    "is a first-line treatment that can dramatically improve CI activity. "
                    "NUBPL deficiency has ZERO riboflavin response — NUBPL has no FAD domain and "
                    "riboflavin supplementation cannot rescue [4Fe-4S] cluster delivery defect. "
                    "ACAD9: MCIA complex (ND2-ND5 module, Class 1 BN-PAGE). NUBPL: N-module Fe-S delivery. "
                    "ACAD9 HCM 55-65%; NUBPL HCM ~25%. "
                    "Riboflavin response trial + WES (3q21.3 vs 14q12) is the mandatory discriminator."
                ),
                "target_gene": "ACAD9",
            },
            {
                "feature":     "NUBPL (14q12) vs NDUFV1 (11q13.2) — NO Leukodystrophy",
                "significance": (
                    "NDUFV1 is the N-module structural subunit (51kDa, FMN-binding, NADH oxidation, N3 cluster). "
                    "NDUFV1 deficiency: leukodystrophy 40-50%, progressive white matter disease. "
                    "NUBPL deficiency: N-module Fe-S delivery factor; NO leukodystrophy. "
                    "Both act on N-module. WES mandatory: 14q12 (NUBPL) vs 11q13.2 (NDUFV1). "
                    "Leukodystrophy on MRI is a strong pointer away from NUBPL toward NDUFV1."
                ),
                "target_gene": "NDUFV1",
            },
            {
                "feature":     "NUBPL vs NDUFS1 (2q33.3) — NO Peripheral Neuropathy",
                "significance": (
                    "NDUFS1 is the 75kDa N-module subunit (Fe-S clusters N1b/N3/N4/N5 — NUBPL delivers these). "
                    "NDUFS1 deficiency causes peripheral neuropathy in ~50% of patients — a hallmark feature. "
                    "NUBPL deficiency: peripheral neuropathy 0%. Peripheral neuropathy in a CI patient "
                    "points away from NUBPL toward NDUFS1. WES mandatory: 2q33.3 vs 14q12."
                ),
                "target_gene": "NDUFS1",
            },
            {
                "feature":     "NUBPL vs NDUFS4 (5q11.2) — NO Olfactory Bulb Lesions",
                "significance": (
                    "NDUFS4 deficiency causes pathognomonic bilateral olfactory bulb lesions on MRI (52-65%). "
                    "NUBPL deficiency: olfactory bulb lesions 0%. Olfactory bulb MRI lesions "
                    "point away from NUBPL strongly toward NDUFS4."
                ),
                "target_gene": "NDUFS4",
            },
            {
                "feature":     "NUBPL vs TIMMDC1 (3q25.1) — HCM ~25% vs HCM >80%",
                "significance": (
                    "TIMMDC1 deficiency: HCM >80% (highest in CI assembly factors; integral IMM, ND1-module Class 3). "
                    "NUBPL deficiency: HCM ~25% (intermediate). Prominent HCM (>60%) in a CI patient points toward TIMMDC1. "
                    "NUBPL ~25% HCM is intermediate — not the dominant clinical feature. "
                    "Completely different assembly modules: NUBPL N-module Fe-S delivery vs TIMMDC1 ND1-module integral-IMM."
                ),
                "target_gene": "TIMMDC1",
            },
            {
                "feature":     "NUBPL vs POLG/DGUOK — NO Hepatopathy",
                "significance": (
                    "POLG (15q26.1) and DGUOK (2p13.1) cause hepatopathy. "
                    "NUBPL deficiency: hepatopathy 0%. Hepatopathy points away from NUBPL."
                ),
                "target_gene": "POLG / DGUOK",
            },
        ],

        "clinical_alerts": [
            "🔴 METFORMIN — ABSOLUTE CONTRAINDICATION: Direct CI inhibitor. NUBPL patients have 5-20% CI; metformin biguanide inhibition is immediately life-threatening.",
            "🔴 VALPROATE (VPA) — ABSOLUTE CONTRAINDICATION: Triple mechanism: CoA sequestration, POLG inhibition, MT-ND subunit expression suppression. Causes further CI collapse.",
            "🔴 LINEZOLID — ABSOLUTE CONTRAINDICATION: 23S rRNA inhibition blocks all 7 mtDNA-encoded CI subunits (MT-ND1–6). No CI rescue possible.",
            "🔴 CHLORAMPHENICOL — ABSOLUTE CONTRAINDICATION: Same mitoribosomal 23S rRNA mechanism as linezolid.",
            "🔴 KETOGENIC DIET — CONTRAINDICATED: CI absent/minimal; NADH cannot be reoxidised via ETC. Beta-oxidation generates NADH that cannot be cleared — metabolic crisis.",
            "🟠 PROPOFOL — AVOID (PRIS risk): Propofol infusion syndrome via CIV inhibition + fatty acid oxidation uncoupling. Use sevoflurane for anaesthesia.",
            "🟡 PHENOBARBITAL — HIGH CAUTION: Secondary CI inhibitor effect; prefer LEV (levetiracetam) as first-choice AED — renal excretion, no mitochondrial toxicity.",
            "🟡 RIBOFLAVIN — NOT INDICATED FOR NUBPL: NUBPL has no FAD domain; riboflavin CANNOT rescue [4Fe-4S] cluster delivery defect. Do not treat as ACAD9 (riboflavin Level B).",
            "🟢 SUCCINATE — Level C: CII substrate bypasses stalled CI entirely; allows CII → CIII → CIV electron flow; partial ATP rescue.",
            "🟢 CoQ10 (Ubiquinol) — Level C: Antioxidant + electron carrier support; standard add-on.",
            "🟢 THIAMINE B1 — Level C MANDATORY EMPIRIC: Exclude SLC19A3 (thiamine transporter) and BTD (biotin-thiamine-responsive basal ganglia disease) BEFORE confirming NUBPL.",
            "🟢 BIOTIN — Level C MANDATORY EMPIRIC: Exclude BTD (biotinidase deficiency) and HLCS BEFORE CI gene panel.",
            "🟢 CARNITINE — Level C: Supplement if secondary carnitine deficiency documented.",
            "🔵 EUROPEAN FOUNDER ALLELE p.Gly56Arg (c.166G>A): ~70% of NUBPL patients carry this allele. Heterozygous p.Gly56Arg in a European CI-deficiency patient is a strong NUBPL pointer — deep intronic sequencing (c.815-27T>C branch point) mandatory to find second allele.",
            "🔵 NUBPL (14q12) vs FOXRED1 (11q24.2) — SAME N-MODULE, DIFFERENT MECHANISM: Both N-module factors, both 0% riboflavin response. NUBPL HCM ~25%; FOXRED1 HCM ~10%. WES mandatory to distinguish.",
            "🔵 HCM (~25%): Echocardiography at diagnosis. HCM rate intermediate — higher than FOXRED1, lower than TIMMDC1. Cardiology follow-up warranted.",
            "🔵 DEEP INTRONIC SECOND ALLELE: c.815-27T>C (branch point intron 9) is missed by standard exome sequencing. If p.Gly56Arg heterozygous with CI deficiency, request NUBPL RNA studies or targeted deep intronic sequencing.",
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

    # Gly56Arg carrier count (European founder)
    gly56arg_carriers = sum(
        1 for p in PATIENTS
        if "Gly56Arg" in p["allele1"] or "Gly56Arg" in p["allele2"]
    )

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
        "gly56arg_carrier_pct":    round(gly56arg_carriers / N_PATIENTS * 100),
        "key_n_module_fes_features": {
            "Only_CI_specific_4Fe4S_delivery_factor": True,
            "P_loop_Walker_A_GTPase_domain":          True,
            "European_founder_pGly56Arg_70pct":       True,
            "No_FAD_domain_no_riboflavin_response":   True,
            "Soluble_matrix_no_TM_helices":           True,
            "Isolated_CI_deficiency_only":            True,
            "HCM_rate_intermediate_pct":              round(sum(p["hcm"] for p in PATIENTS) / N_PATIENTS * 100),
            "Deep_intronic_c815_27TC_second_allele":  True,
        },
        "treatment_summary": {
            "absolute_ci": ["Metformin", "Valproate (VPA)", "Linezolid", "Chloramphenicol", "Ketogenic Diet"],
            "avoid":       ["Propofol (PRIS)", "Phenobarbital (high caution)"],
            "do_not_use":  ["Riboflavin (NOT indicated — NUBPL has no FAD domain; riboflavin cannot rescue Fe-S delivery defect)"],
            "level_c":     ["Succinate (CII bypass)", "CoQ10-Ubiquinol", "Thiamine B1 (MANDATORY empiric)", "Biotin (MANDATORY empiric)", "Carnitine"],
            "preferred_aed": "LEV (levetiracetam) — renal excretion, no mitochondrial toxicity",
            "anaesthesia": "Sevoflurane (NOT propofol — PRIS risk)",
            "diagnostic_priority": "Deep intronic NUBPL sequencing if heterozygous p.Gly56Arg + CI deficiency in European patient",
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
                "comparator":     "FOXRED1 (11q24.2) — same N-module",
                "nubpl":          "[4Fe-4S] delivery; HCM ~25%; p.Gly56Arg founder; P-loop GTPase",
                "comparator_val": "FAD-oxidoreductase chaperone; HCM ~10%; no founder allele; soluble",
                "key_test":       "HCM rate + p.Gly56Arg status + WES locus (14q12 vs 11q24.2)",
            },
            {
                "comparator":     "ACAD9 (3q21.3)",
                "nubpl":          "No FAD domain; 0% riboflavin response; N-module Fe-S delivery; HCM ~25%",
                "comparator_val": "FAD-binding; riboflavin-responsive 50-60% (Level B); MCIA/ND2-ND5; HCM 55-65%",
                "key_test":       "Riboflavin trial + WES locus (14q12 vs 3q21.3)",
            },
            {
                "comparator":     "NDUFV1 (11q13.2)",
                "nubpl":          "Fe-S delivery factor; no leukodystrophy; 14q12",
                "comparator_val": "Structural N-module subunit (FMN-binding); leukodystrophy 40-50%; 11q13.2",
                "key_test":       "MRI leukodystrophy + WES locus",
            },
            {
                "comparator":     "TIMMDC1 (3q25.1)",
                "nubpl":          "N-module; soluble matrix; HCM ~25%",
                "comparator_val": "ND1-module (Class 3); integral IMM; HCM >80%",
                "key_test":       "HCM rate + BN-PAGE class + WES locus",
            },
            {
                "comparator":     "NDUFS1 (2q33.3)",
                "nubpl":          "NO peripheral neuropathy",
                "comparator_val": "Peripheral neuropathy ~50%",
                "key_test":       "Clinical neurophysiology (nerve conduction study)",
            },
        ],
    }


# ─── get_definitions ──────────────────────────────────────────────────────────
def get_definitions() -> dict:
    return {
        "gene_definition": (
            "NUBPL (Nucleotide-binding protein-like; also known as IND1, OMIM *613621) encodes "
            "a 301-amino-acid, ~34 kDa soluble mitochondrial matrix protein. It is the human "
            "homolog of yeast Ind1p (iron-sulfur protein required for NADH dehydrogenase, part of "
            "CIA-like machinery). NUBPL/IND1 is the only known CI-specific [4Fe-4S] cluster "
            "delivery factor. It contains a P-loop NTPase (Walker A GxGxxG motif) domain whose "
            "GTPase activity drives [4Fe-4S] cluster transfer to N-module subunits NDUFS1 "
            "(N1b, N3, N4, N5 clusters) and NDUFV1 (N3 cluster). Loss-of-function variants "
            "cause isolated CI deficiency with Leigh syndrome spectrum. The European founder "
            "allele p.Gly56Arg (c.166G>A) is found in ~70% of published NUBPL patients."
        ),
        "disease_definition": (
            "Mitochondrial Complex I Deficiency (CI Deficiency; OMIM #256000) due to NUBPL "
            "bi-allelic pathogenic variants presents as infantile-onset Leigh syndrome with "
            "profound isolated CI deficiency (5-20% of control), preserved CII/CIII/CIV, "
            "Leigh MRI (~75%), lactic acidosis (~86%), hypotonia (~91%), seizures (~55%), and "
            "intermediate HCM rate (~25%). The p.Gly56Arg European founder allele is found in "
            "~70% of patients. A deep intronic second allele (c.815-27T>C branch point, intron 9) "
            "is missed by standard exome sequencing — RNA studies or targeted deep intronic "
            "sequencing are required. No riboflavin response."
        ),
        "inheritance_definition": (
            "Autosomal recessive (AR). Biallelic pathogenic NUBPL variants required. "
            "Both sexes equally affected. Consanguinity is a background (~38%). "
            "European founder allele p.Gly56Arg has enriched carrier frequency in Scottish/British populations. "
            "Carrier parents are clinically unaffected. Genetic counselling: 25% recurrence risk per pregnancy."
        ),
        "module_definitions": [
            {
                "term":       "N-Module [4Fe-4S] Cluster Delivery by NUBPL/IND1",
                "definition": (
                    "The N-module is the matrix arm tip of CI (NADH dehydrogenase module) containing "
                    "NDUFS1 (75kDa, N1b/N3/N4/N5 clusters), NDUFV1 (51kDa, FMN-binding, N3 cluster), "
                    "NDUFV2 (24kDa, N1a cluster), NDUFV3, and NDUFS6. [4Fe-4S] clusters are essential "
                    "electron transfer cofactors at N1b, N3, N4, N5, N6a, N6b, N7 positions. "
                    "NUBPL/IND1 delivers [4Fe-4S] clusters from the mitochondrial ISC assembly "
                    "machinery (ISCU/NFS1/LYRM4) specifically to CI N-module subunits. "
                    "Without functional NUBPL, N-module subunits lack their [4Fe-4S] cofactors, "
                    "cannot fold/assemble, and holoenzyme CI cannot form. This is different from "
                    "FOXRED1 (same N-module but acts as FAD-oxidoreductase protein chaperone) — "
                    "sequential, independent steps in N-module biogenesis."
                ),
            },
            {
                "term":       "Walker A P-Loop GTPase/NTPase Domain (NUBPL Catalytic Core)",
                "definition": (
                    "The Walker A motif (consensus GxGxxGK[T/S]) forms the P-loop that binds the "
                    "beta-phosphate of nucleoside triphosphates. Hydrolysis of GTP/ATP provides "
                    "the energy to drive [4Fe-4S] cluster transfer from NUBPL to acceptor subunits "
                    "(NDUFS1, NDUFV1). p.Gly56Arg disrupts the first glycine of the Walker A GxGxxG "
                    "sequence — arginine adds steric bulk that distorts the P-loop geometry and "
                    "impairs nucleotide coordination. Partial activity is retained (hypomorphic), "
                    "explaining the intermediate severity of the European founder allele. "
                    "Complete Walker A disruption (null alleles) causes severe/neonatal-lethal CI deficiency."
                ),
            },
            {
                "term":       "European Founder Allele p.Gly56Arg and Deep Intronic c.815-27T>C",
                "definition": (
                    "p.Gly56Arg (c.166G>A, exon 2) is a European founder allele enriched in "
                    "Scottish/British populations, found in ~70% of published NUBPL patients. "
                    "It is almost always in compound heterozygosity with a second allele. "
                    "The second allele in the original Calvo 2010 cohort was c.815-27T>C — a deep "
                    "intronic branch-point variant in intron 9, 27 nucleotides upstream of exon 10. "
                    "This branch-point mutation causes aberrant splicing with partial exon skipping "
                    "and reduced levels of normal mRNA. It is invisible to standard exome sequencing "
                    "and requires RNA analysis (RT-PCR, RNAseq) or targeted deep intronic sequencing "
                    "for detection. Clinicians suspecting NUBPL in a European patient with p.Gly56Arg "
                    "heterozygosity MUST request deep intronic sequencing."
                ),
            },
        ],
        "clinical_thresholds": [
            {"parameter": "CI enzyme activity",                 "threshold": "<20% of control",    "significance": "Diagnostic criterion for severe CI deficiency in muscle biopsy"},
            {"parameter": "Lactate (plasma)",                   "threshold": ">2.5 mmol/L",        "significance": "Elevated; confirms mitochondrial dysfunction (non-specific)"},
            {"parameter": "Lactate:pyruvate ratio",             "threshold": ">25",                "significance": "Elevated L:P supports NADH-block at CI (ETC defect rather than PDH)"},
            {"parameter": "CSF lactate",                        "threshold": ">2.5 mmol/L",        "significance": "Leigh syndrome workup — elevated in CI deficiency with CNS involvement"},
            {"parameter": "Riboflavin trial response",          "threshold": "0% (none expected)", "significance": "NUBPL has NO FAD domain; 0% response is expected. Any response favors ACAD9 (FAD domain, Level B)"},
            {"parameter": "HCM (echocardiography)",             "threshold": "~25% of cases",      "significance": "Intermediate HCM — higher than FOXRED1 (~10%), lower than TIMMDC1 (>80%). Cardiology follow-up needed"},
            {"parameter": "p.Gly56Arg heterozygosity screen",  "threshold": "Positive = suspect NUBPL", "significance": "European founder allele; ~70% of NUBPL patients; confirms NUBPL as leading diagnosis in European CI-deficiency patients"},
            {"parameter": "Deep intronic sequencing (NUBPL)",  "threshold": "c.815-27T>C",         "significance": "Branch-point intron 9 variant — missed by exome; must request if p.Gly56Arg heterozygous and CI deficiency"},
            {"parameter": "Onset age (severe alleles)",        "threshold": "<6 months",           "significance": "Biallelic severe alleles (p.Leu104Pro, p.Trp240Ter): neonatal-to-early-infantile onset"},
        ],
        "standards": [
            {"code": "ACMG/AMP 2015",         "title": "Variant classification guidelines — pathogenicity criteria for NUBPL"},
            {"code": "MITOMAP",               "title": "Mitochondrial disease gene/variant database"},
            {"code": "OMIM *613621",          "title": "NUBPL gene entry — Nucleotide-binding protein-like (IND1)"},
            {"code": "OMIM #256000",          "title": "Mitochondrial Complex I Deficiency — CI deficiency spectrum"},
            {"code": "REACTOME R-HSA-611105", "title": "Respiratory electron transport — CI [4Fe-4S] cluster assembly and N-module"},
            {"code": "Helsinki Declaration",  "title": "Ethical framework for human subject research"},
        ],
        "references": [
            {
                "id":        "calvo_2010",
                "citation":  "Calvo SE et al. (2010) High-throughput, pooled sequencing identifies mutations in NUBPL and FOXRED1 in human complex I deficiency. Nat Genet 42(10):851–8.",
                "relevance": "First identification of NUBPL (and FOXRED1) as CI disease genes; simultaneous discovery by high-throughput sequencing; establishes p.Gly56Arg European founder and c.815-27T>C deep intronic branch-point as the canonical NUBPL variant combination.",
            },
            {
                "id":        "kevelam_2013",
                "citation":  "Kevelam SH et al. (2013) NUBPL mutations in patients with complex I deficiency and a distinct MRI pattern. Neurology 80(17):1577–83.",
                "relevance": "Detailed NUBPL clinical series; characterizes the NUBPL MRI pattern (Leigh-like + variable cerebellar); confirms p.Gly56Arg European founder frequency; defines HCM and clinical spectrum; key clinical DDx reference.",
            },
            {
                "id":        "guerrero_castillo_2017",
                "citation":  "Guerrero-Castillo S et al. (2017) The assembly pathway of mitochondrial respiratory chain complex I. Cell Metab 25(1):128–139.",
                "relevance": "Defines CI assembly intermediates and temporal order; contextualizes NUBPL Fe-S cluster delivery within the N-module assembly pathway; maps relationship between NUBPL action and holoenzyme CI formation.",
            },
            {
                "id":        "lim_2016",
                "citation":  "Lim SC et al. (2016) A founder mutation in PET100 causes complex IV deficiency in Lebanese individuals with Leigh syndrome. Am J Hum Genet [comparative reference for founder allele patterns in mitochondrial disease].",
                "relevance": "Founder allele diagnostic framework; comparison study for p.Gly56Arg European founder strategy in NUBPL (used for context).",
            },
            {
                "id":        "sheftel_2009",
                "citation":  "Sheftel AD et al. (2009) Human ind1, an iron-sulfur cluster assembly factor for respiratory complex I. Mol Cell Biol 29(22):6059–73.",
                "relevance": "Functional characterization of human IND1/NUBPL; demonstrates [4Fe-4S] cluster delivery to CI N-module subunits; establishes Walker A P-loop GTPase mechanism; names the protein IND1 (human) as CI-specific Fe-S assembly factor.",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps({"overview": get_overview(), "breakdown": get_breakdown(), "definitions": get_definitions()}, indent=2, default=str))
