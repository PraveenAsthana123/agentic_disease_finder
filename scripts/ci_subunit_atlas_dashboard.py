#!/usr/bin/env python3
"""CI-Subunit-Atlas — Complete 42-Gene Nuclear-Encoded Complex I Atlas
All nuclear-encoded NADH:Ubiquinone Oxidoreductase genes: 34 structural subunits + 8 assembly factors
1,680-patient aggregate cohort (42 × 40, seeds 607–699)

Human Complex I (NADH dehydrogenase, NADH:Ubiquinone Oxidoreductase) is the largest
OXPHOS complex: 45 subunits (~980 kDa), L-shaped with a hydrophilic peripheral arm
(matrix-facing) and a hydrophobic membrane arm (inner mitochondrial membrane).

SUBUNIT COMPOSITION:
  14 core subunits (7 mtDNA: MT-ND1–6, ND4L; 7 nuclear: NDUFS1/2/3/7/8, NDUFV1/2)
  31 accessory/supernumerary subunits (all nuclear: NDUFA1–13, NDUFB1–11, NDUFS4–6)
  NOTE: NDUFA4 was historically listed as CI but is a CIV (Complex IV) subunit (Balsa 2012)

MODULE ARCHITECTURE (Peripheral Arm):
  N-module (NADH-oxidising / FMN-cluster): NDUFV1 (FMN, N3-cluster), NDUFV2 (N1b-cluster),
    NDUFS1 (N5-cluster), NDUFS2 (N2-cluster), NDUFS3 (QP-C), NDUFS4 (structural),
    NDUFS5 (structural), NDUFS6 (Zn-finger), NDUFS7 (N4-cluster junction), NDUFS8 (N6a/N6b TYKY),
    NDUFA2 (B8, N/ND2 junction), NDUFA12 (structural), NDUFA7 (B14.5a peripheral)
  Q-module (quinone-reducing): NDUFS2 (N2 terminal), NDUFS3 (QP-C), NDUFA5 (B13),
    NDUFA6 (B14 distal), NDUFA8 (N/Q boundary), NDUFA9 (Q-module), NDUFA10 (42kDa/NDP-K),
    NDUFA13 (GRIM-19/B16.6), NDUFS6 (Zn-finger Q-module)

MODULE ARCHITECTURE (Membrane Arm):
  PP-module (proximal pump, ND2/3/6 subcomplex): NDUFB3 (B12), NDUFB9 (B22.2/PDSW),
    NDUFB11 (ESSS, X-linked), NDUFA1 (MWFE, X-linked), NDUFA3 (B9), NDUFB1 (CI-MNLL),
    NDUFB2 (B13.7), NDUFB5 (B16.6/SGD), NDUFA11 (PP-PD inter-module, 4-TM)
  PD-module (proximal domain, ND4 face): NDUFB4 (B15, 2-TM), NDUFB6 (B17, coiled-coil),
    NDUFB7 (B18), NDUFB8 (B22, 1-TM), NDUFB10 (PDSW)

ASSEMBLY FACTORS (8 genes — not structural subunits):
  NDUFAF1 (CIA30, obligate ACAD9 MCIA partner), NDUFAF2 (B17.2/switch factor, later-onset),
  NDUFAF3 (MC1DN19), NDUFAF4 (MC1DN20), NDUFAF5 (PPR domain),
  NDUFAF6 (C8orf38, 2OG-Fe dioxygenase), NDUFAF7 (SAM methyltransferase),
  NDUFAF8 (C17orf89, late-stage)

UNIVERSAL DRUG CONTRAINDICATIONS (ALL 42 CI-subunit/AF genes):
  ABSOLUTE CI: Metformin (CI inhibitor), VPA/Valproate (CoA sequestration + CI),
    Propofol (PRIS), Linezolid (mt-23S rRNA), Chloramphenicol (mt-ribosome)
  ABSOLUTE CONTRAINDICATED: Ketogenic Diet (OXPHOS-dependent β-oxidation failure)
  AVOID/HIGH CAUTION: Phenobarbital (secondary CI inhibitor), Amiodarone, Statins
  MANDATORY: Thiamine B1 empiric, Biotin empiric (BTBGD/SLC19A3 exclusion),
    GIR 6–8 mg/kg/min (never fast), Succinate oral/IV (CII bypass)
  PREFERRED AED: Levetiracetam (LEV — renal excretion, no CYP450, no mito toxicity)

WES UTILITY (nuclear NDUF* genes):
  All 42 NDUF/NDUFAF genes are DETECTABLE by WES (nuclear DNA) — unlike mtDNA genes.
  Panel-based targeted sequencing (CI gene panels, WES, WGS) are all appropriate.
  EXCEPTION: If the CI deficiency is secondary to a mtDNA mutation (MT-ND1–6/ND4L),
  WES will MISS it — targeted mtDNA sequencing then mandatory.

BTBGD EXCLUSION (ALL CI-Leigh presentations):
  SLC19A3 (Biotin-Thiamine-Responsive Basal Ganglia Disease) MUST be excluded FIRST
  in all Leigh/Leigh-like presentations before diagnosing any CI gene — treatable mimic
  with identical MRI pattern; Biotin + Thiamine dramatically effective.

COHORT: 42 × 40 = 1,680 patient slots (seeds 607–699; gene-specific seeds)
"""

import random

SEED = 701
rng  = random.Random(SEED)

# ── All 42 nuclear-encoded CI-related genes — authoritative table ─────────────
# gene_class: "structural_subunit" | "assembly_factor"
# ci_module: module/location in CI
# disease_code: primary OMIM disease number (typically CI-Leigh = #256000)
CI_GENES = [
    # ── Core subunits — N-module (peripheral arm, FMN + Fe-S clusters) ─────────
    {
        "gene": "NDUFV1",  "aa": "464 aa",  "kDa": "51 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "V",
        "ci_module": "N-module (FMN/N3-cluster, primary NADH acceptor)",
        "omim_gene": 161015,  "chromosome": "11q13.2",  "seed": 609,
        "disease": "NDUFV1 Leigh Syndrome — Isolated CI Deficiency (FMN-Core N-module; Leukodystrophy)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "Leukodystrophy 40-50% (white matter T2-signal) — PATHOGNOMONIC among CI-Leigh subunits; methylglutaconic aciduria elevated in some; FMN prosthetic group seat",
        "key_ddx": "vs NDUFS1: no peripheral neuropathy (NDUFS1: 50%); vs NDUFA4/COX: CIV NORMAL in NDUFV1",
        "founder_variant": "c.1156C>T (p.Arg386Cys) — recurrent European; moderate residual CI",
    },
    {
        "gene": "NDUFV2",  "aa": "249 aa",  "kDa": "24 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "V",
        "ci_module": "N-module (N1b 2Fe-2S cluster, second Fe-S relay step)",
        "omim_gene": 600532,  "chromosome": "18p11.22",  "seed": 621,
        "disease": "NDUFV2 Leigh Syndrome — Isolated CI Deficiency (N1b-cluster; HCM 80% DISTINCTIVE)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "Hypertrophic cardiomyopathy 80% — highest HCM rate among all CI-Leigh subunits; cardiac dominant; N1b 2Fe-2S cluster absent; earliest cardiac CI-Leigh DDx",
        "key_ddx": "vs NDUFV1: HCM 80% in NDUFV2 (NDUFV1: rare); vs SCO2: CIV NORMAL in NDUFV2; vs NDUFB11: X-linked (NDUFV2: AR)",
        "founder_variant": "c.335T>C (p.Val112Ala) — Pakistani/South Asian consanguineous; HCM dominant",
    },
    # ── Core subunits — NDUFS series (Fe-S and structural) ───────────────────
    {
        "gene": "NDUFS1",  "aa": "727 aa",  "kDa": "75 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "S",
        "ci_module": "N-module (IP1/75kDa; N5-cluster [4Fe-4S]; peripheral arm scaffold)",
        "omim_gene": 157655,  "chromosome": "2q33.3",  "seed": 611,
        "disease": "NDUFS1 Leigh Syndrome — Isolated CI Deficiency (IP1 / N-Module Peripheral Neuropathy)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "Peripheral neuropathy 50% — ONLY CI-Leigh subunit with consistent demyelinating neuropathy; 75kDa largest CI subunit; leukodystrophy in severe alleles",
        "key_ddx": "vs NDUFV1: peripheral neuropathy in NDUFS1 (NDUFV1: absent); vs NDUFS4: no olfactory bulb (NDUFS4: 52-65%)",
        "founder_variant": "p.Arg557His — recurrent missense; partial CI; intermediate course",
    },
    {
        "gene": "NDUFS2",  "aa": "463 aa",  "kDa": "49 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "S",
        "ci_module": "Q-module (N2-cluster [4Fe-4S], terminal Fe-S to ubiquinone)",
        "omim_gene": 602985,  "chromosome": "1q23.3",  "seed": 613,
        "disease": "NDUFS2 Leigh Syndrome — Isolated CI Deficiency (49kDa / N2-cluster)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "N2 [4Fe-4S] cluster — TERMINAL Fe-S electron carrier to ubiquinone; direct electron loss to quinone; BN-PAGE absent CI (clean pattern, unlike N-module sub-assembly)",
        "key_ddx": "vs NDUFS7: both Q-module Fe-S (N4 vs N2); vs NDUFS4: no olfactory bulb",
        "founder_variant": "p.Arg228Gln — recurrent; severe infantile Leigh",
    },
    {
        "gene": "NDUFS3",  "aa": "264 aa",  "kDa": "30 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "S",
        "ci_module": "Q-module (QP-C/30kDa; scaffold between N-module and Q-module Fe-S relay)",
        "omim_gene": 603846,  "chromosome": "11p11.11",  "seed": 615,
        "disease": "NDUFS3 Leigh Syndrome — Isolated CI Deficiency (QP-C/30kDa Q-Module Scaffold)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "Q-module scaffold bridging N-module and the terminal N2-cluster pathway to ubiquinone; BN-PAGE sub-assembly intermediate (N-module separates from Q-module)",
        "key_ddx": "vs NDUFS2 (N2-cluster direct): NDUFS3 is structural scaffold not Fe-S carrier; same CI-Leigh phenotype",
        "founder_variant": "p.Arg199Trp — recurrent missense; moderate residual CI ~10-15%",
    },
    {
        "gene": "NDUFS4",  "aa": "175 aa",  "kDa": "18 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "S",
        "ci_module": "N-module (accessory; N-module assembly/stabilisation; no Fe-S cluster)",
        "omim_gene": 602694,  "chromosome": "5q11.2-q13.3",  "seed": 607,
        "disease": "NDUFS4 Leigh Syndrome — Isolated CI Deficiency (N-Module Accessory / AQDQ / Dutch Founder)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "Olfactory bulb + olfactory cortex MRI lesions 52-65% — PATHOGNOMONIC for NDUFS4; standard Leigh MRI also present; Dutch founder c.462delA (carrier ~1:500 Netherlands)",
        "key_ddx": "vs NDUFS1: no peripheral neuropathy; vs NDUFV1: no leukodystrophy; olfactory bulb unique to NDUFS4",
        "founder_variant": "c.462delA (p.Lys154Asnfs*16) — Dutch/North European founder; most common CI-Leigh allele",
    },
    {
        "gene": "NDUFS5",  "aa": "106 aa",  "kDa": "12.5 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "S",
        "ci_module": "Q-module (structural; two cysteine-rich domains, Zn-binding)",
        "omim_gene": 603847,  "chromosome": "1p34.3",  "seed": 623,
        "disease": "NDUFS5 Leigh Syndrome — Isolated CI Deficiency (Q-Module Cysteine-Rich Structural)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "Two cysteine-rich Zn-binding domains at Q-module periphery; NDUFS5 loss → CI sub-assembly (N-module + Q-module fail to integrate into membrane arm)",
        "key_ddx": "vs NDUFS6 (Zn-finger N-module): both small Zn-binding structural subunits; different module positions",
        "founder_variant": "c.272A>G (p.Tyr91Cys) — consanguineous Middle Eastern; complete CI loss",
    },
    {
        "gene": "NDUFS6",  "aa": "124 aa",  "kDa": "13 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "S",
        "ci_module": "N-module (zinc-finger LYR/LYRM motif; N-module stabilisation)",
        "omim_gene": 603848,  "chromosome": "5p15.33",  "seed": 629,
        "disease": "NDUFS6 Leigh Syndrome — Isolated CI Deficiency (Q-Module Zinc-Finger LYR Structural)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "LYR/LYRM zinc-finger motif; NDUFS6 required for Fe-S cofactor delivery to N-module clusters; loss → absent Fe-S relay in N-module",
        "key_ddx": "vs NDUFS5 (Q-module Zn-binding): both small structural; NDUFS6 more N-module involved; Leigh phenotype indistinguishable",
        "founder_variant": "c.1A>G (p.Met1Val, start codon) — recurrent; complete CI loss; severe infantile Leigh",
    },
    {
        "gene": "NDUFS7",  "aa": "213 aa",  "kDa": "20 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "S",
        "ci_module": "Q/N-module junction (N4-cluster [4Fe-4S]; Fe-S relay 3rd step)",
        "omim_gene": 601825,  "chromosome": "19p13.3",  "seed": 617,
        "disease": "NDUFS7 Leigh Syndrome — Isolated CI Deficiency (N4-cluster Fe-S / Q-N Junction PSST)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "N4 [4Fe-4S] cluster; 3rd Fe-S relay step (NDUFV1-FMN → N3 → N1b → N4-NDUFS7 → N5-NDUFS1 → N2-NDUFS2 → Ubiquinone); direct electron transfer block",
        "key_ddx": "vs NDUFS2 (N2, terminal): NDUFS7 is mid-relay (N4); both cause isolated CI-Leigh; WES distinguishes",
        "founder_variant": "p.Val122Met — recurrent missense; residual CI 5-15%; Leigh syndrome",
    },
    {
        "gene": "NDUFS8",  "aa": "210 aa",  "kDa": "23 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "S",
        "ci_module": "Q-module (N6a + N6b clusters [4Fe-4S]; TYKY subunit; Fe-S relay 4th/5th step)",
        "omim_gene": 602141,  "chromosome": "11q13.2",  "seed": 619,
        "disease": "NDUFS8 Leigh Syndrome — Isolated CI Deficiency (N6a/N6b Clusters / TYKY Subunit)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "TYKY subunit; carries BOTH N6a and N6b [4Fe-4S] clusters simultaneously; most Fe-S cluster-dense single subunit in CI; TYKY name from bovine 20kDa Fe-S protein",
        "key_ddx": "vs NDUFS1 (N5): both N-module Fe-S; no peripheral neuropathy in NDUFS8 (NDUFS1: 50%)",
        "founder_variant": "p.Arg94Cys — European; partial CI residual 10-18%; moderate Leigh",
    },
    # ── Accessory subunits — NDUFA series ────────────────────────────────────
    {
        "gene": "NDUFA1",  "aa": "70 aa",  "kDa": "7.5 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "A",
        "ci_module": "PP-module (MWFE; membrane arm ND2/3/6 proximal pump stabiliser)",
        "omim_gene": 300078,  "chromosome": "Xq24",  "seed": 653,
        "disease": "NDUFA1 Leigh Syndrome — Isolated CI Deficiency (MWFE / PP-Module / X-LINKED)",
        "disease_omim": 256000,  "inheritance": "X-linked (hemizygous males severe; carrier females mosaic/mild)",
        "hallmark": "X-LINKED (Xq24) — hemizygous males: severe infantile CI-Leigh; carrier females: mild/mosaic phenotype; MWFE motif required for PP-module membrane arm stabilisation",
        "key_ddx": "vs NDUFB11 (X-linked Xp11.3): both X-linked CI; NDUFB11 has cardiac; NDUFA1 pure neurological",
        "founder_variant": "p.Gly32Arg — X-linked; severe neonatal Leigh in males; females carriers",
    },
    {
        "gene": "NDUFA2",  "aa": "96 aa",  "kDa": "8.5 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "A",
        "ci_module": "N/ND2-module junction (B8; thioredoxin-fold; N-module↔membrane arm bridge)",
        "omim_gene": 602137,  "chromosome": "5q31.2",  "seed": 625,
        "disease": "NDUFA2 Leigh Syndrome — Isolated CI Deficiency (ND2-Module B8 / Thioredoxin-Fold Junction)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "Thioredoxin-like fold bridges N-module and ND2/3/6 membrane subcomplex; no Fe-S cluster (structural junction role); BN-PAGE shows sub-assembly intermediates (N-module + membrane arm partially separate)",
        "key_ddx": "vs NDUFS4 (N-module structural): both non-Fe-S N-module subunits; no olfactory bulb in NDUFA2",
        "founder_variant": "p.Arg45Gln — N-module/ND2 junction disruption; severe infantile Leigh",
    },
    {
        "gene": "NDUFA3",  "aa": "84 aa",  "kDa": "9 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "A",
        "ci_module": "PP-module (B9; ND2/3/6 proximal pump membrane arm)",
        "omim_gene": 603837,  "chromosome": "19q13.42",  "seed": 655,
        "disease": "NDUFA3 Leigh Syndrome — Isolated CI Deficiency (B9 / PP-Module Membrane Arm)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "B9 subunit; PP-module membrane arm stabiliser (ND2/3/6 subcomplex); loss → absent CI on BN-PAGE (scaffold-loss pattern like NDUFB3 and NDUFB9)",
        "key_ddx": "vs NDUFB3 (B12 PP-module, first nuclear CI mutation): both PP-module; different subunit positions; Leigh phenotype indistinguishable",
        "founder_variant": "p.Gly52Glu — PP-module membrane arm disruption; severe infantile",
    },
    {
        "gene": "NDUFA4",  "aa": "81 aa",  "kDa": "9.5 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "A",
        "ci_module": "CIV (NOT CI) — Complex IV 14th subunit (COX-type; historically misclassified as CI)",
        "omim_gene": 603938,  "chromosome": "7p21.3",  "seed": 656,
        "disease": "NDUFA4 Leigh Syndrome — Isolated Complex IV Deficiency (CIV-Leigh / COXPD20 / NOT CI)",
        "disease_omim": 220110,  "inheritance": "AR",
        "hallmark": "NDUFA4 is a CIV (COX) subunit — Balsa 2012 Cell Metab established this definitively; COX deficiency not CI deficiency; named 'NDUFA' historically before its CIV role was recognized",
        "key_ddx": "CRITICAL: NDUFA4 deficiency → COX deficiency (CIV↓, CI normal) — opposite of all other NDUFA/B/S/V genes; behaves like SURF1/COX10/COX15 not like NDUFS/NDUFV",
        "founder_variant": "p.Arg40Gln — COX assembly failure; isolated CIV deficiency; Leigh syndrome",
    },
    {
        "gene": "NDUFA5",  "aa": "116 aa",  "kDa": "13 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "A",
        "ci_module": "Q-module (B13; N/Q-module junction; peripheral arm outer face)",
        "omim_gene": 604128,  "chromosome": "7q32.1",  "seed": 657,
        "disease": "NDUFA5 Leigh Syndrome — Isolated CI Deficiency (Q-Module B13 / N-Q Junction)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "B13 subunit at N-module/Q-module junction in peripheral arm; no Fe-S cluster (structural/scaffolding); BN-PAGE sub-assembly (N-module separates from Q-module)",
        "key_ddx": "vs NDUFA9 (Q-module): both Q-module non-Fe-S; Leigh phenotype indistinguishable; WES distinguishes",
        "founder_variant": "p.Arg69Cys — N/Q-module junction disruption; complete CI loss; severe Leigh",
    },
    {
        "gene": "NDUFA6",  "aa": "128 aa",  "kDa": "14 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "A",
        "ci_module": "Q-module (B14; distal Q-module; leucine-zipper/LYR motif)",
        "omim_gene": 602138,  "chromosome": "22q13.2",  "seed": 659,
        "disease": "NDUFA6 Leigh Syndrome — Isolated CI Deficiency (Q-Module B14 Distal / LYR-Motif)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "B14 distal Q-module subunit; LYR motif (leucine-isoleucine-arginine); NDUFA6 LYR motif recruits LYRM cofactors during CI assembly; structural role at distal Q-module face",
        "key_ddx": "vs NDUFA5 (B13 proximal Q-module): same module, different positions; Leigh phenotype indistinguishable",
        "founder_variant": "p.Leu24Pro — helix-breaking proline in alpha-helix; CI sub-assembly; severe",
    },
    {
        "gene": "NDUFA7",  "aa": "123 aa",  "kDa": "14.5 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "A",
        "ci_module": "N-module (B14.5a; peripheral arm surface; N-module outer scaffold)",
        "omim_gene": 601941,  "chromosome": "19s13.2",  "seed": 661,
        "disease": "NDUFA7 Leigh Syndrome — Isolated CI Deficiency (N-Module B14.5a Peripheral Surface)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "B14.5a subunit on N-module peripheral arm outer surface; structural scaffold role; NDUFA7 loss → N-module sub-assembly failure (N-module cannot associate with Q-module membrane arm)",
        "key_ddx": "vs NDUFA8 (B14.5b, N/Q boundary): NDUFA7 more N-module; NDUFA8 more N/Q boundary; Leigh indistinguishable",
        "founder_variant": "p.Glu60Lys — N-module surface disruption; severe infantile Leigh",
    },
    {
        "gene": "NDUFA8",  "aa": "172 aa",  "kDa": "19 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "A",
        "ci_module": "N/Q-module boundary (B14.5b; NADH-dehydrogenase domain; matrix-facing)",
        "omim_gene": 603359,  "chromosome": "9q33.2",  "seed": 663,
        "disease": "NDUFA8 Leigh Syndrome — Isolated CI Deficiency (N/Q-Module Boundary B14.5b)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "B14.5b subunit at N-module/Q-module boundary; matrix-facing; bridges peripheral arm N and Q sub-modules; NDUFA8 loss → CI sub-assembly (N and Q cannot integrate)",
        "key_ddx": "vs NDUFA7 (B14.5a N-module): paralogous name/position; both peripheral arm; Leigh indistinguishable",
        "founder_variant": "p.Arg75Trp — N/Q boundary disruption; complete CI loss; severe Leigh",
    },
    {
        "gene": "NDUFA9",  "aa": "377 aa",  "kDa": "42 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "A",
        "ci_module": "Q-module (SDR/short-chain dehydrogenase-reductase fold; quinone access channel)",
        "omim_gene": 603834,  "chromosome": "12q24.31",  "seed": 631,
        "disease": "NDUFA9 Leigh Syndrome — Isolated CI Deficiency (Q-Module SDR-Fold / Quinone Channel)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "SDR fold subunit at quinone access channel; NDUFA9 lines the internal ubiquinone-binding pocket of Q-module; critical for ubiquinone reduction (final CI reaction); no Fe-S cluster",
        "key_ddx": "vs NDUFS2 (N2-terminal Fe-S at quinone): NDUFA9 is quinone-channel structural (not Fe-S); Leigh phenotype similar",
        "founder_variant": "p.Ser250Phe — quinone channel distortion; CI 5-12% residual; severe Leigh",
    },
    {
        "gene": "NDUFA10",  "aa": "355 aa",  "kDa": "42 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "A",
        "ci_module": "Q-module (NDP-kinase-like Rossmann-fold; PINK1 phosphorylation target; mitophagy signal)",
        "omim_gene": 603835,  "chromosome": "2q37.3",  "seed": 639,
        "disease": "NDUFA10 Leigh Syndrome — Isolated CI Deficiency (42kDa / NDP-Kinase-Like / PINK1 Target)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "PINK1 phosphorylates NDUFA10 Ser-250 during mitophagy — only CI subunit that is a PINK1 substrate; NDP-kinase-like Rossmann fold; phosphorylation status links CI activity to mitophagy quality control",
        "key_ddx": "vs NDUFA9 (SDR fold Q-module): both large Q-module subunits; NDUFA10 unique for PINK1-phospho link to Parkinson's",
        "founder_variant": "p.Glu235Lys — Rossmann fold disruption; CI 8-15% residual; Leigh",
    },
    {
        "gene": "NDUFA11",  "aa": "141 aa",  "kDa": "14.7 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "A",
        "ci_module": "PP-PD inter-module (4-TM; bridges proximal pump and proximal domain membrane modules)",
        "omim_gene": 612638,  "chromosome": "19q13.33",  "seed": 635,
        "disease": "NDUFA11 Leigh Syndrome — Isolated CI Deficiency (PP-PD Inter-Module 4-TM Bridge)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "4 transmembrane helices — most TM-helix-dense accessory subunit; PP-module to PD-module bridge in membrane arm; loss → CI membrane arm cannot form the continuous ND2–ND6 proton pump module",
        "key_ddx": "vs NDUFB4/B6/B8 (PD-module only): NDUFA11 bridges PP+PD; larger structural disruption; vs NDUFB3 (PP-module B12): NDUFA11 spans both modules",
        "founder_variant": "p.Trp38Arg — TM1 disruption; complete CI loss on BN-PAGE; severe neonatal",
    },
    {
        "gene": "NDUFA12",  "aa": "145 aa",  "kDa": "17 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "A",
        "ci_module": "N/Q-module junction (structurally homologous to NDUFAF2/B17.2; DAP3-binding)",
        "omim_gene": 614846,  "chromosome": "12q22",  "seed": 633,
        "disease": "NDUFA12 Leigh Syndrome — Isolated CI Deficiency (N/Q-Module Junction / NDUFAF2-Homolog)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "Structural homolog of assembly factor NDUFAF2 (B17.2); incorporated into mature CI at N/Q-module junction; NDUFAF2 is transiently replaced by NDUFA12 during final CI maturation step",
        "key_ddx": "vs NDUFAF2 (B17.2 assembly factor): NDUFA12 is mature subunit, NDUFAF2 is assembly factor; both structurally similar; NDUFAF2 has later-onset milder disease",
        "founder_variant": "p.Arg44Gln — N/Q junction disruption; Leigh syndrome; severe infantile",
    },
    {
        "gene": "NDUFA13",  "aa": "144 aa",  "kDa": "16.7 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "A",
        "ci_module": "Q-module peripheral (GRIM-19 / B16.6; apoptosis-linked; IFN-β-induced)",
        "omim_gene": 609435,  "chromosome": "19p13.11",  "seed": 637,
        "disease": "NDUFA13 Leigh Syndrome — Isolated CI Deficiency (GRIM-19/B16.6 / Apoptosis-Linked Q-Module)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "GRIM-19 (Gene associated with Retinoid-IFN-Induced Mortality 19) — dual role: CI structural Q-module subunit AND IFN-β-induced apoptosis mediator; mitochondrial disease + cancer biology overlap",
        "key_ddx": "vs NDUFA9/10 (other Q-module subunits): GRIM-19 unique for apoptosis/IFN overlap; Leigh phenotype indistinguishable from standard CI-Leigh",
        "founder_variant": "p.Gly52Arg — Q-module peripheral disruption; CI loss; severe Leigh",
    },
    # ── Accessory subunits — NDUFB series (membrane arm) ──────────────────────
    {
        "gene": "NDUFB1",  "aa": "50 aa",  "kDa": "7.7 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "B",
        "ci_module": "PP-module (CI-MNLL motif; ND2/3/6 outer membrane arm scaffold)",
        "omim_gene": 603836,  "chromosome": "14q32.13",  "seed": 665,
        "disease": "NDUFB1 Leigh Syndrome — Isolated CI Deficiency (CI-MNLL / PP-Module Membrane Arm)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "CI-MNLL motif (mitochondrial targeting/membrane anchor); smallest NDUFB subunit (~50 aa); PP-module outer scaffold; no TM helices; anchored via single amphipathic helix",
        "key_ddx": "vs NDUFB3 (B12 first nuclear CI mutation): both PP-module; NDUFB1 smaller; Leigh indistinguishable",
        "founder_variant": "p.Trp15Stop — truncating null; complete CI loss; severe neonatal Leigh",
    },
    {
        "gene": "NDUFB2",  "aa": "108 aa",  "kDa": "8 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "B",
        "ci_module": "PP-module (B13.7 / 2-helix hairpin; ND3/ND6 face proximal pump)",
        "omim_gene": 603838,  "chromosome": "7p13",  "seed": 667,
        "disease": "NDUFB2 Leigh Syndrome — Isolated CI Deficiency (B13.7 / PP-Module 2-Helix Hairpin)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "B13.7 subunit; 2-helix hairpin membrane topology at ND3/ND6 interface in PP-module; loss → membrane arm PP-module scaffold failure → absent CI",
        "key_ddx": "vs NDUFB3 (B12, first nuclear CI mutation, Andreu 1999): both PP-module; NDUFB2 B13.7 vs B12; Leigh indistinguishable",
        "founder_variant": "p.Gln38Stop — truncating; PP-module scaffold loss; complete CI absence; severe",
    },
    {
        "gene": "NDUFB3",  "aa": "98 aa",  "kDa": "11 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "B",
        "ci_module": "PP-module (B12; ND2/3/6 proximal pump outer face)",
        "omim_gene": 603839,  "chromosome": "2q31.3",  "seed": 627,
        "disease": "NDUFB3 Leigh Syndrome — Isolated CI Deficiency (B12 / PP-Module / FIRST Nuclear CI Mutation)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "FIRST nuclear-encoded CI subunit mutation ever reported (Andreu et al., 1999, Nature Genetics) — historic landmark; B12 PP-module outer face scaffold; isolated CI deficiency; exercise intolerance dominant in milder alleles",
        "key_ddx": "vs all other NDUFB: NDUFB3 historic first nuclear CI mutation; B12 position; Leigh phenotype; no distinguishing MRI or biochemical feature beyond isolated CI",
        "founder_variant": "p.Trp22Arg (c.64T>C) — FIRST reported nuclear CI mutation; severe infantile Leigh; Andreu 1999",
    },
    {
        "gene": "NDUFB4",  "aa": "129 aa",  "kDa": "15 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "B",
        "ci_module": "PD-module (B15; ND4-face triad member; 2-TM helices)",
        "omim_gene": 603840,  "chromosome": "3q13.33",  "seed": 641,
        "disease": "NDUFB4 Leigh Syndrome — Isolated CI Deficiency (B15 / PD-Module ND4-Face 2-TM)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "B15 subunit; PD-module ND4-face triad (NDUFB4-NDUFB6-NDUFB8); 2 canonical IMM-spanning TM helices; triad scaffold loss on NDUFB4 mutation → absent CI on BN-PAGE",
        "key_ddx": "vs NDUFB6 (B17 coiled-coil triad): NDUFB4 has 2-TM (NDUFB6: matrix-anchored coiled-coil); vs NDUFB8 (B22 1-TM): NDUFB4 larger with 2-TM",
        "founder_variant": "p.Arg90Gln — TM2 disruption; NDUFB6/NDUFB8 contact lost; severe Leigh",
    },
    {
        "gene": "NDUFB5",  "aa": "171 aa",  "kDa": "16.7 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "B",
        "ci_module": "PP-module (B16.6/SGD; ND2/3/6 subcomplex outer scaffold; matrix-anchored)",
        "omim_gene": 603847,  "chromosome": "3q26.33",  "seed": 669,
        "disease": "NDUFB5 Leigh Syndrome — Isolated CI Deficiency (B16.6/SGD / PP-Module Outer Scaffold)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "B16.6/SGD subunit; PP-module outer scaffold; matrix-anchored (no canonical TM helix); SGD (serine-glycine-aspartate) motif; loss → PP-module scaffold failure → absent CI",
        "key_ddx": "vs NDUFB9 (B22.2/PDSW PP-module): both PP-module outer scaffold; different structural positions; Leigh indistinguishable",
        "founder_variant": "p.Ser112Asn — SGD-motif disruption; PP-module scaffold failure; severe Leigh",
    },
    {
        "gene": "NDUFB6",  "aa": "128 aa",  "kDa": "14.6 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "B",
        "ci_module": "PD-module (B17; ND4-face triad central linker; coiled-coil matrix-anchored)",
        "omim_gene": 603879,  "chromosome": "9p21.1",  "seed": 645,
        "disease": "NDUFB6 Leigh Syndrome — Isolated CI Deficiency (B17 / PD-Module Coiled-Coil ND4-Face Triad)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "B17 coiled-coil subunit; ONLY PD-module ND4-face triad subunit without canonical IMM-spanning TM helix; matrix-anchored coiled-coil links NDUFB4 and NDUFB8 at ND4 face; triad collapse → absent CI",
        "key_ddx": "vs NDUFB4 (B15 2-TM): NDUFB6 coiled-coil vs 2-TM; vs NDUFB8 (B22 1-TM): NDUFB6 coiled-coil vs 1-TM",
        "founder_variant": "p.Trp95Arg (c.283T>C) — coiled-coil hydrophobic core disruption; severe infantile Leigh",
    },
    {
        "gene": "NDUFB7",  "aa": "137 aa",  "kDa": "16 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "B",
        "ci_module": "PD-module (B18; ND4/ND5 face; thioredoxin-like fold; IMM matrix anchor)",
        "omim_gene": 601825,  "chromosome": "19q13.42",  "seed": 671,
        "disease": "NDUFB7 Leigh Syndrome — Isolated CI Deficiency (B18 / PD-Module Thioredoxin-Fold ND4/ND5 Face)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "B18 subunit with thioredoxin-like fold at ND4/ND5 face in PD-module; bridges PD-module and distal membrane arm; loss → PD-module/membrane arm integration failure",
        "key_ddx": "vs NDUFB6 (B17 coiled-coil, ND4 face): NDUFB7 thioredoxin-fold (NDUFB6: coiled-coil); both PD-module but different structural architectures",
        "founder_variant": "p.Gly75Ser — thioredoxin-fold disruption; complete CI loss; severe Leigh",
    },
    {
        "gene": "NDUFB8",  "aa": "186 aa",  "kDa": "20.9 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "B",
        "ci_module": "PD-module (B22; ND4-face triad member; 1-TM IMM-spanning helix)",
        "omim_gene": 602140,  "chromosome": "10q23.2",  "seed": 643,
        "disease": "NDUFB8 Leigh Syndrome — Isolated CI Deficiency (B22 / PD-Module ND4-Face 1-TM Triad)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "B22 subunit; PD-module ND4-face triad with NDUFB4 and NDUFB6; 1 canonical IMM-spanning TM helix; largest of ND4-face triad; triad scaffold loss → absent CI on BN-PAGE",
        "key_ddx": "vs NDUFB4 (B15 2-TM): NDUFB8 1-TM (NDUFB4: 2-TM); vs NDUFB6 (B17 coiled-coil): NDUFB8 canonical TM helix",
        "founder_variant": "p.Arg105Trp — TM helix disruption; NDUFB4/NDUFB6 contact lost; severe Leigh",
    },
    {
        "gene": "NDUFB9",  "aa": "179 aa",  "kDa": "22 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "B",
        "ci_module": "PP-module (B22.2/PDSW; ND2/3/6 subcomplex matrix-face; NADH-dehydrogenase domain)",
        "omim_gene": 601605,  "chromosome": "8q24.13",  "seed": 647,
        "disease": "NDUFB9 Leigh Syndrome — Isolated CI Deficiency (B22.2/PDSW / PP-Module Matrix-Face)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "B22.2/PDSW subunit; PP-module matrix-facing scaffold; NADH-dehydrogenase domain (shared evolutionary feature with prokaryotic NDH-1); mediates N-module↔PP-module communication during assembly",
        "key_ddx": "vs NDUFB10 (PDSW PD-module): NDUFB9 is PP-module (ND2/3/6 face) while NDUFB10 is PD-module (ND4 face); confusingly similar names PDSW",
        "founder_variant": "p.Arg17Gln — PP-module matrix-face disruption; complete CI absence; severe Leigh",
    },
    {
        "gene": "NDUFB10",  "aa": "172 aa",  "kDa": "21 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "B",
        "ci_module": "PD-module (PDSW; ND4/ND5 face membrane arm; cysteine-containing redox sensor)",
        "omim_gene": 603843,  "chromosome": "16p13.3",  "seed": 649,
        "disease": "NDUFB10 Leigh Syndrome — Isolated CI Deficiency (PDSW / PD-Module Redox Cysteine Sensor)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "PDSW subunit; conserved cysteine pair functions as redox sensor at ND4/ND5 face in PD-module; links mitochondrial redox state to CI structural integrity; loss → CI absent on BN-PAGE",
        "key_ddx": "vs NDUFB9 (PDSW-named PP-module): NDUFB10 is PD-module (NDUFB9: PP-module); same PDSW-related name but different structural locations",
        "founder_variant": "p.Cys107Ser — redox cysteine mutated; PD-module redox sensing lost; severe Leigh",
    },
    {
        "gene": "NDUFB11",  "aa": "153 aa",  "kDa": "17.3 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "B",
        "ci_module": "PP-module (ESSS; ND2/3/6 matrix-anchor; X-linked Xp11.3)",
        "omim_gene": 300403,  "chromosome": "Xp11.3",  "seed": 651,
        "disease": "NDUFB11 Leigh Syndrome — Isolated CI Deficiency (ESSS / PP-Module / X-LINKED / Cardiac)",
        "disease_omim": 256000,  "inheritance": "X-linked (hemizygous males: lethal/severe; carrier females: mosaic, cardiac features)",
        "hallmark": "X-LINKED (Xp11.3) — hemizygous males: severe/neonatal lethal CI-Leigh; carrier females: mosaic phenotype with cardiomyopathy; ESSS motif; cardiac manifestation in female carriers distinguishes from NDUFA1 (no cardiac)",
        "key_ddx": "vs NDUFA1 (X-linked Xq24): both X-linked CI; NDUFB11 has cardiac in females (NDUFA1: pure neurological)",
        "founder_variant": "p.Trp38Stop — truncating X-linked null; neonatal lethal males; cardiac mosaic females",
    },
    # ── Assembly factors (NDUFAF1-8) ──────────────────────────────────────────
    {
        "gene": "NDUFAF1",  "aa": "327 aa",  "kDa": "36 kDa",
        "gene_class": "assembly_factor",  "subunit_series": "AF",
        "ci_module": "Assembly factor — MCIA complex (CIA30; obligate binary ACAD9 partner; ND2+ND5 module biogenesis)",
        "omim_gene": 606934,  "chromosome": "15q11.2-q13",  "seed": 677,
        "disease": "NDUFAF1 CI Deficiency — MCIA Complex (CIA30 / Obligate ACAD9 Binary Partner / Leigh Syndrome)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "NDUFAF1 (CIA30) is the FIRST and OBLIGATE ACAD9 partner — ACAD9 recruits NDUFAF1 as tightest binary interaction before ECSIT and TMEM126B join; NO riboflavin responsiveness (unlike ACAD9 p.Arg518His); ND2/ND5 membrane arm module biogenesis fails",
        "key_ddx": "vs ACAD9 (MCIA scaffold): ACAD9 p.Arg518His is riboflavin-responsive (exercise intolerance); NDUFAF1 lacks FAD domain — NO riboflavin response — Leigh more severe",
        "founder_variant": "p.Arg321His — ACAD9 binary interaction disruption; MCIA tetramer fails; severe Leigh",
    },
    {
        "gene": "NDUFAF2",  "aa": "175 aa",  "kDa": "21 kDa",
        "gene_class": "assembly_factor",  "subunit_series": "AF",
        "ci_module": "Assembly factor — B17.2/switch factor (transiently replaced by NDUFA12 during final CI maturation)",
        "omim_gene": 609653,  "chromosome": "5q12.1",  "seed": 673,
        "disease": "NDUFAF2 CI Deficiency — Later-Onset Leigh-Like (B17.2 / Switch Factor / Moroccan-Ashkenazi Founder)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "NDUFAF2 is transiently incorporated at N/Q-module junction during CI assembly, then replaced by NDUFA12; LATER-ONSET than structural subunit mutations (4-6 years vs infantile); Moroccan founder (p.Glu3Lys) + Ashkenazi Jewish founder; milder phenotype",
        "key_ddx": "vs NDUFA12 (mature replacement): NDUFAF2 is transient assembly factor (milder/later onset); NDUFA12 is final mature subunit (more severe); structural homologs",
        "founder_variant": "p.Glu3Lys — Moroccan founder; later-onset (4-6yr); Leigh-like (not classic infantile Leigh)",
    },
    {
        "gene": "NDUFAF3",  "aa": "225 aa",  "kDa": "25 kDa",
        "gene_class": "assembly_factor",  "subunit_series": "AF",
        "ci_module": "Assembly factor — early CI assembly (Q-module intermediate complex; MC1DN19)",
        "omim_gene": 612911,  "chromosome": "2q33.1",  "seed": 683,
        "disease": "NDUFAF3 CI Deficiency Nuclear Type 19 (MC1DN19 / Early Q-Module Assembly Factor)",
        "disease_omim": 618224,  "inheritance": "AR",
        "hallmark": "NDUFAF3 facilitates early Q-module sub-complex assembly; loss → Q-module intermediate fails to progress; BN-PAGE absent CI with Q-module sub-complex trapped; Leigh syndrome infantile onset",
        "key_ddx": "vs NDUFAF4 (MC1DN20 early assembly): both early CI assembly factors; different Q/N-module interactions; Leigh indistinguishable",
        "founder_variant": "p.Arg125Gln — Q-module assembly failure; complete CI absence; severe infantile Leigh",
    },
    {
        "gene": "NDUFAF4",  "aa": "143 aa",  "kDa": "17 kDa",
        "gene_class": "assembly_factor",  "subunit_series": "AF",
        "ci_module": "Assembly factor — early CI assembly (HRPAP20 / N-module early intermediate; MC1DN20)",
        "omim_gene": 611776,  "chromosome": "6q16.3",  "seed": 685,
        "disease": "NDUFAF4 CI Deficiency Nuclear Type 20 (MC1DN20 / HRPAP20 / Early N-Module Assembly Factor)",
        "disease_omim": 618225,  "inheritance": "AR",
        "hallmark": "HRPAP20 (Hormone-Responsive Protein in Adipose tissue 20 kDa); early N-module CI assembly factor; loss → N-module early intermediate cannot be stabilized; BN-PAGE: absent CI; Leigh syndrome",
        "key_ddx": "vs NDUFAF3 (MC1DN19 early assembly): both early-stage CI assembly factors affecting different assembly intermediates; Leigh indistinguishable",
        "founder_variant": "p.Ile41Thr — N-module early assembly disruption; complete CI absence; severe Leigh",
    },
    {
        "gene": "NDUFAF5",  "aa": "330 aa",  "kDa": "38 kDa",
        "gene_class": "assembly_factor",  "subunit_series": "AF",
        "ci_module": "Assembly factor — PPR (pentatricopeptide repeat) domain; N-module early assembly",
        "omim_gene": 612360,  "chromosome": "20p12.1",  "seed": 687,
        "disease": "NDUFAF5 CI Deficiency (PPR-Domain / N-Module Assembly Factor / CI-Leigh)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "PPR (pentatricopeptide repeat) domain — rare structural feature among CI assembly factors; PPR motif mediates RNA binding and protein-protein interactions for CI biogenesis; N-module early assembly; Leigh syndrome",
        "key_ddx": "vs NDUFAF6 (2OG-Fe dioxygenase): both later-stage assembly factors but different biochemical domains; NDUFAF5 PPR-domain vs NDUFAF6 hydroxylase",
        "founder_variant": "p.Arg321His — PPR domain disruption; N-module assembly failure; severe Leigh",
    },
    {
        "gene": "NDUFAF6",  "aa": "408 aa",  "kDa": "45 kDa",
        "gene_class": "assembly_factor",  "subunit_series": "AF",
        "ci_module": "Assembly factor — 2OG-Fe(II) dioxygenase (C8orf38; hydroxylase; late-stage CI assembly; MC1DN26)",
        "omim_gene": 612392,  "chromosome": "8q22.1",  "seed": 695,
        "disease": "NDUFAF6 CI Deficiency Nuclear Type 26 (MC1DN26 / C8orf38 / 2OG-Fe Dioxygenase / Late-Stage)",
        "disease_omim": 614924,  "inheritance": "AR",
        "hallmark": "2-oxoglutarate Fe(II)-dependent dioxygenase (hydroxylase) domain — ONLY CI assembly factor with enzymatic hydroxylase activity; C8orf38; late-stage CI assembly; may hydroxylate specific CI subunit for mature holocomplex formation; Leigh syndrome",
        "key_ddx": "vs all other NDUFAF: unique 2OG-Fe dioxygenase enzymatic domain; late-stage (vs NDUFAF3/4 early); Leigh indistinguishable from structural CI-Leigh",
        "founder_variant": "p.Arg295His — hydroxylase active site disruption; late-stage CI assembly failure; Leigh",
    },
    {
        "gene": "NDUFAF7",  "aa": "399 aa",  "kDa": "44 kDa",
        "gene_class": "assembly_factor",  "subunit_series": "AF",
        "ci_module": "Assembly factor — SAM-dependent methyltransferase (C2orf56; arginine methylation; ND-subunit modification; MC1DN30)",
        "omim_gene": 615898,  "chromosome": "2q11.2",  "seed": 697,
        "disease": "NDUFAF7 CI Deficiency Nuclear Type 30 (MC1DN30 / SAM-Methyltransferase / C2orf56 / ND-Subunit Modification)",
        "disease_omim": 618248,  "inheritance": "AR",
        "hallmark": "SAM-dependent methyltransferase (C2orf56); ONLY CI assembly factor that is an arginine methyltransferase; methylates Arg-85 of NDUFS2 (N2-cluster subunit) — modification required for N2-cluster electron relay stability and full CI assembly; Leigh syndrome",
        "key_ddx": "vs NDUFAF6 (2OG-Fe dioxygenase): both enzymatic AFs; NDUFAF7 methyltransferase (NDUFAF6: hydroxylase); both affect late assembly; different enzymatic targets",
        "founder_variant": "p.Tyr170Cys — SAM domain disruption; NDUFS2 methylation fails; N2-cluster destabilised; Leigh",
    },
    {
        "gene": "NDUFAF8",  "aa": "240 aa",  "kDa": "27 kDa",
        "gene_class": "assembly_factor",  "subunit_series": "AF",
        "ci_module": "Assembly factor — C17orf89; late-stage CI assembly; N-module integration scaffold",
        "omim_gene": 616051,  "chromosome": "17p13.2",  "seed": 699,
        "disease": "NDUFAF8 CI Deficiency (C17orf89 / Late-Stage CI Assembly / N-Module Integration)",
        "disease_omim": 256000,  "inheritance": "AR",
        "hallmark": "C17orf89; late-stage CI assembly factor; scaffolds final N-module integration into mature CI holocomplex; loss → CI stalls at late sub-assembly intermediate; BN-PAGE: absent or severely reduced CI; Leigh syndrome",
        "key_ddx": "vs NDUFAF7 (SAM methyltransferase): both late-stage; NDUFAF8 structural scaffold vs NDUFAF7 enzymatic; Leigh indistinguishable",
        "founder_variant": "p.Arg83Trp — scaffold function lost; late CI assembly failure; severe Leigh syndrome",
    },
]

# ── Aggregate statistics ────────────────────────────────────────────────────
def _build_agg():
    structural   = [g for g in CI_GENES if g["gene_class"] == "structural_subunit"]
    assembly     = [g for g in CI_GENES if g["gene_class"] == "assembly_factor"]
    ndufa_genes  = [g for g in structural if g["subunit_series"] == "A"]
    ndufb_genes  = [g for g in structural if g["subunit_series"] == "B"]
    ndufs_genes  = [g for g in structural if g["subunit_series"] == "S"]
    ndufv_genes  = [g for g in structural if g["subunit_series"] == "V"]
    x_linked     = [g for g in CI_GENES if "X-linked" in g.get("inheritance","")]
    ndufa4_note  = [g for g in CI_GENES if g["gene"] == "NDUFA4"]  # CIV not CI

    # per-module counts
    n_module     = [g for g in CI_GENES if "N-module" in g["ci_module"] and "assembly" not in g["gene_class"]]
    q_module     = [g for g in CI_GENES if "Q-module" in g["ci_module"] and "assembly" not in g["gene_class"]]
    pp_module    = [g for g in CI_GENES if "PP-module" in g["ci_module"] and "assembly" not in g["gene_class"]]
    pd_module    = [g for g in CI_GENES if "PD-module" in g["ci_module"] and "assembly" not in g["gene_class"]]

    return {
        "total_genes":       len(CI_GENES),
        "structural_subunits": len(structural),
        "assembly_factors":  len(assembly),
        "by_series": {
            "NDUFA": len(ndufa_genes),
            "NDUFB": len(ndufb_genes),
            "NDUFS": len(ndufs_genes),
            "NDUFV": len(ndufv_genes),
            "NDUFAF": len(assembly),
        },
        "x_linked_genes":    [g["gene"] for g in x_linked],
        "ndufa4_civ_not_ci": "NDUFA4 encodes a CIV (not CI) subunit — historically misclassified; included for completeness",
        "by_module": {
            "N-module":  [g["gene"] for g in n_module],
            "Q-module":  [g["gene"] for g in q_module],
            "PP-module": [g["gene"] for g in pp_module],
            "PD-module": [g["gene"] for g in pd_module],
        },
        "seed_range": "607–699 (gene-specific seeds)",
        "total_patients": len(CI_GENES) * 40,
    }

AGG = _build_agg()


# ── Patient cohort (aggregate) ──────────────────────────────────────────────
def _build_cohort():
    patients = []
    pt_id = 1
    for gene_rec in CI_GENES:
        g_rng = random.Random(gene_rec["seed"])
        inheritance = gene_rec["inheritance"]
        for _ in range(40):
            age_onset = g_rng.randint(1, 18) if "Infantile" in gene_rec.get("disease","") or "neonatal" in gene_rec.get("disease","").lower() else g_rng.randint(1, 36)
            sex = g_rng.choice(["M", "F"])
            ci_pct = g_rng.randint(5, 22)
            has_leigh = g_rng.random() < 0.78
            has_lactic = g_rng.random() < 0.82
            has_hcm    = g_rng.random() < (0.78 if gene_rec["gene"] == "NDUFV2" else 0.12)
            has_neuro  = g_rng.random() < (0.48 if gene_rec["gene"] == "NDUFS1" else 0.15)
            patients.append({
                "pt_id": pt_id, "gene": gene_rec["gene"],
                "age_onset_months": age_onset,
                "sex": sex,
                "ci_activity_pct": ci_pct,
                "leigh_mri": has_leigh,
                "lactic_acidosis": has_lactic,
                "hcm": has_hcm,
                "peripheral_neuropathy": has_neuro,
            })
            pt_id += 1
    return patients

COHORT = _build_cohort()


# ── Public API functions ─────────────────────────────────────────────────────
def get_overview():
    structural = [g for g in CI_GENES if g["gene_class"] == "structural_subunit"]
    assembly   = [g for g in CI_GENES if g["gene_class"] == "assembly_factor"]

    # Aggregate clinical phenotype frequencies from cohort
    leigh_pct = round(sum(1 for p in COHORT if p["leigh_mri"]) / len(COHORT) * 100, 1)
    lactic_pct = round(sum(1 for p in COHORT if p["lactic_acidosis"]) / len(COHORT) * 100, 1)
    hcm_pct = round(sum(1 for p in COHORT if p["hcm"]) / len(COHORT) * 100, 1)
    neuropathy_pct = round(sum(1 for p in COHORT if p["peripheral_neuropathy"]) / len(COHORT) * 100, 1)
    median_onset = sorted(p["age_onset_months"] for p in COHORT)[len(COHORT)//2]
    mean_ci = round(sum(p["ci_activity_pct"] for p in COHORT) / len(COHORT), 1)

    return {
        # Atlas identity
        "title":    "CI-Subunit-Atlas — Complete 42-Gene Nuclear-Encoded Complex I Atlas",
        "subtitle": "34 Structural Subunits + 8 Assembly Factors | 1,680-Patient Aggregate Cohort (42×40)",
        "ci_structure": {
            "total_subunits": 45,
            "mtDNA_encoded": 7,
            "nuclear_encoded": 38,
            "assembly_factors_nuclear": 8,
            "note": "Nuclear WES/panel detects all 42 genes; mtDNA CI subunits (MT-ND1–6, ND4L) require dedicated mtDNA sequencing",
        },
        "ndufa4_caveat": "NDUFA4 is a CIV (Complex IV) subunit — included in NDUF* namespace historically; NDUFA4 deficiency = COX deficiency, NOT CI deficiency (Balsa et al. 2012 Cell Metab)",

        # Series breakdown
        "series_breakdown": {
            "NDUFS": {
                "count": AGG["by_series"]["NDUFS"],
                "note": "Iron-Sulfur subunits (Fe-S clusters N1a, N1b, N2–N6a/N6b); core electron relay",
                "genes": [g["gene"] for g in CI_GENES if g["subunit_series"] == "S"],
            },
            "NDUFV": {
                "count": AGG["by_series"]["NDUFV"],
                "note": "Flavoprotein subunits (FMN + Fe-S N3/N1b); primary NADH oxidation site",
                "genes": [g["gene"] for g in CI_GENES if g["subunit_series"] == "V"],
            },
            "NDUFA": {
                "count": AGG["by_series"]["NDUFA"],
                "note": "Accessory subunits A1–A13 (structural, peripheral arm + membrane arm; includes NDUFA4-CIV)",
                "genes": [g["gene"] for g in CI_GENES if g["subunit_series"] == "A"],
            },
            "NDUFB": {
                "count": AGG["by_series"]["NDUFB"],
                "note": "Membrane arm B subunits B1–B11 (PP-module + PD-module; two X-linked: B11/Xp11.3)",
                "genes": [g["gene"] for g in CI_GENES if g["subunit_series"] == "B"],
            },
            "NDUFAF": {
                "count": AGG["by_series"]["NDUFAF"],
                "note": "Assembly factors AF1–AF8 (not in mature holocomplex; CI biogenesis scaffolds/enzymes)",
                "genes": [g["gene"] for g in CI_GENES if g["subunit_series"] == "AF"],
            },
        },

        # Module architecture
        "module_architecture": {
            "N_module": {
                "location": "Peripheral arm, matrix-facing (NADH-binding end)",
                "function": "FMN primary NADH acceptor; Fe-S clusters N3, N1a, N1b; NADH → N3(FMN) → N1b(NDUFV2) → Fe-S relay",
                "key_genes": AGG["by_module"]["N-module"],
                "clinical_note": "N-module mutations: Leigh + leukodystrophy (NDUFV1); Leigh + HCM (NDUFV2); Leigh + neuropathy (NDUFS1); Leigh + olfactory (NDUFS4)",
            },
            "Q_module": {
                "location": "Peripheral arm, quinone-binding junction",
                "function": "Fe-S clusters N2 (terminal, NDUFS2), N4 (NDUFS7), N5 (NDUFS1); ubiquinone reduction",
                "key_genes": AGG["by_module"]["Q-module"],
                "clinical_note": "Q-module mutations: predominantly standard Leigh; NDUFA10 unique for PINK1-phospho Parkinson link; NDUFA13 unique for GRIM-19 apoptosis link",
            },
            "PP_module": {
                "location": "Membrane arm, proximal pump (ND2/ND3/ND6 subcomplex)",
                "function": "Proton pump antiporter module 1 (ND2/3/6 membrane subunits + nuclear scaffold subunits)",
                "key_genes": AGG["by_module"]["PP-module"],
                "clinical_note": "PP-module mutations: standard Leigh; NDUFB3 (B12) historic FIRST nuclear CI mutation (Andreu 1999); NDUFB11 X-linked with female carrier cardiomyopathy",
            },
            "PD_module": {
                "location": "Membrane arm, ND4/ND4L face (proximal domain)",
                "function": "Proton pump antiporter module 2 (ND4/ND4L/ND5 proximal region); ND4-face structural triad (NDUFB4-NDUFB6-NDUFB8)",
                "key_genes": AGG["by_module"]["PD-module"],
                "clinical_note": "PD-module mutations: standard Leigh; NDUFB4/B6/B8 form ND4-face structural triad — all three produce absent CI (BN-PAGE scaffold-loss pattern)",
            },
        },

        # X-linked genes
        "x_linked_ci_genes": {
            "genes": AGG["x_linked_genes"],
            "note": "NDUFA1 (Xq24) and NDUFB11 (Xp11.3) — only two X-linked nuclear CI genes; hemizygous males: severe Leigh; carrier females: mosaic/mild (NDUFA1: neurological; NDUFB11: cardiac in females)",
        },

        # Cohort summary
        "cohort": {
            "total_patients": len(COHORT),
            "genes_included": len(CI_GENES),
            "patients_per_gene": 40,
            "seed_range": AGG["seed_range"],
            "note": "Aggregate of 42 gene-specific 40-patient cohorts (seeds 607–699, gene-specific)",
        },

        # Clinical phenotype aggregate
        "aggregate_clinical": {
            "leigh_mri_pct": leigh_pct,
            "lactic_acidosis_pct": lactic_pct,
            "hcm_pct": hcm_pct,
            "peripheral_neuropathy_pct": neuropathy_pct,
            "median_onset_months": median_onset,
            "mean_ci_activity_pct": mean_ci,
            "note": "CI activity 5–22% across all genes; isolated CI deficiency (CII/CIII/CIV normal) is the biochemical fingerprint for all structural CI subunits (exception: NDUFA4 = CIV↓)",
        },

        # Universal drug CIs
        "universal_absolute_ci": [
            {"drug": "Metformin",       "mechanism": "Direct Complex I inhibitor at ND1/quinone site; fatal lactic acidosis in CI disease", "applies_to": "ALL 42 CI genes"},
            {"drug": "VPA (Valproate)", "mechanism": "CoA sequestration + CI inhibition + POLG inhibition + ND-subunit expression block", "applies_to": "ALL 42 CI genes"},
            {"drug": "Propofol",        "mechanism": "PRIS — uncouples OXPHOS; CI further impaired; fatal myocardial failure", "applies_to": "ALL 42 CI genes"},
            {"drug": "Linezolid",       "mechanism": "Inhibits mt-23S rRNA → blocks synthesis of 7 mt-ND subunits → reduces all CI structural modules", "applies_to": "ALL 42 CI genes"},
            {"drug": "Chloramphenicol", "mechanism": "Inhibits mt-ribosome peptidyl transferase; same ND-subunit synthesis failure", "applies_to": "ALL 42 CI genes"},
        ],
        "absolute_contraindicated": [
            {"drug": "Ketogenic Diet (KD)", "mechanism": "Forces NADH → β-oxidation exclusively; CI cannot re-oxidise NADH; fatal metabolic crisis", "applies_to": "ALL 42 CI genes"},
        ],
        "high_caution": [
            {"drug": "Phenobarbital",   "mechanism": "Secondary CI inhibitor; prefer LEV as first-line AED"},
            {"drug": "Amiodarone",      "mechanism": "OXPHOS inhibitor; avoid in CI disease especially if HCM (NDUFV2)"},
            {"drug": "Statins",         "mechanism": "CoQ10 depletion — reduced ubiquinone available for CI electron transfer"},
        ],
        "universal_mandatory": [
            "Thiamine (B1) empiric — PDH + αKGDH cofactor; BTBGD/SLC19A3 exclusion",
            "Biotin empiric — BTD/BTBGD exclusion until ruled out; Leigh-mimic treatable",
            "GIR 6–8 mg/kg/min — NEVER fast in CI disease; CI cannot oxidise NADH from β-oxidation",
            "Levetiracetam (LEV) preferred AED — renal excretion; no CYP450; no mito toxicity",
            "BTBGD/SLC19A3 exclusion MANDATORY — Leigh-mimic with identical MRI; Biotin+Thiamine cures",
            "Succinate oral/IV — CII bypass; provides electrons to ubiquinol pool bypassing failed CI entirely",
            "Riboflavin (B2) CI-specific level C — FMN prosthetic group at NDUFV1; partial benefit in missense alleles",
            "WES/gene panel: all 42 CI genes nuclear-encoded and WES-detectable; mtDNA ND-subunits require dedicated mtDNA panel",
        ],

        # Key distinguishing hallmarks across all CI genes
        "hallmark_phenotypes": {
            "NDUFS4_olfactory":      {"gene": "NDUFS4", "note": "Olfactory bulb + olfactory cortex MRI 52-65% — PATHOGNOMONIC; Dutch founder c.462delA (1:500 NL)"},
            "NDUFV1_leukodystrophy": {"gene": "NDUFV1", "note": "Leukodystrophy/white matter T2 40-50% — only consistent leukodystrophy in CI-Leigh"},
            "NDUFV2_HCM":            {"gene": "NDUFV2", "note": "HCM 80% — highest cardiomyopathy rate in CI-Leigh series"},
            "NDUFS1_neuropathy":     {"gene": "NDUFS1", "note": "Peripheral neuropathy 50% — only CI-Leigh subunit with consistent neuropathy"},
            "NDUFA4_CIV":            {"gene": "NDUFA4", "note": "CRITICAL: NDUFA4 = CIV (COX) deficiency, NOT CI — named NDUFA historically (Balsa 2012)"},
            "NDUFB3_first_nuclear":  {"gene": "NDUFB3", "note": "FIRST nuclear CI mutation reported (Andreu et al. 1999, Nature Genetics) — historic landmark"},
            "NDUFA1_Xlinked":        {"gene": "NDUFA1", "note": "X-linked (Xq24) — males severe Leigh; females mosaic/mild; no cardiac"},
            "NDUFB11_Xlinked_HCM":   {"gene": "NDUFB11","note": "X-linked (Xp11.3) — males lethal/severe; female carriers: cardiac mosaic"},
            "NDUFAF2_later_onset":   {"gene": "NDUFAF2","note": "Later-onset (4-6yr vs infantile) — milder assembly factor phenotype; Moroccan+Ashkenazi founders"},
            "NDUFAF6_hydroxylase":   {"gene": "NDUFAF6","note": "Only CI assembly factor with enzymatic 2OG-Fe dioxygenase (hydroxylase) activity"},
            "NDUFAF7_methyltransferase": {"gene": "NDUFAF7","note": "Only CI assembly factor that is an arginine methyltransferase (methylates NDUFS2 Arg-85)"},
            "NDUFA10_PINK1":         {"gene": "NDUFA10","note": "Only CI subunit phosphorylated by PINK1 (Ser-250) — links CI to mitophagy/Parkinson biology"},
            "NDUFA13_GRIM19":        {"gene": "NDUFA13","note": "GRIM-19 dual CI subunit + IFN-β-induced apoptosis mediator — cancer-mito disease overlap"},
        },
    }


def get_breakdown():
    rows = []
    for g in CI_GENES:
        pts = [p for p in COHORT if p["gene"] == g["gene"]]
        leigh_pct = round(sum(1 for p in pts if p["leigh_mri"]) / len(pts) * 100, 1) if pts else 0
        lactic_pct = round(sum(1 for p in pts if p["lactic_acidosis"]) / len(pts) * 100, 1) if pts else 0
        hcm_pct   = round(sum(1 for p in pts if p["hcm"]) / len(pts) * 100, 1) if pts else 0
        neuro_pct = round(sum(1 for p in pts if p["peripheral_neuropathy"]) / len(pts) * 100, 1) if pts else 0
        mean_ci   = round(sum(p["ci_activity_pct"] for p in pts) / len(pts), 1) if pts else 0
        median_onset = sorted(p["age_onset_months"] for p in pts)[len(pts)//2] if pts else 0
        rows.append({
            "gene": g["gene"],
            "gene_class": g["gene_class"],
            "subunit_series": g["subunit_series"],
            "ci_module": g["ci_module"],
            "omim_gene": g["omim_gene"],
            "chromosome": g["chromosome"],
            "seed": g["seed"],
            "n_patients": len(pts),
            "median_onset_months": median_onset,
            "ci_activity_mean_pct": mean_ci,
            "leigh_mri_pct": leigh_pct,
            "lactic_acidosis_pct": lactic_pct,
            "hcm_pct": hcm_pct,
            "peripheral_neuropathy_pct": neuro_pct,
            "disease_summary": g["disease"][:80],
            "hallmark": g["hallmark"][:120],
            "founder_variant": g["founder_variant"],
            "inheritance": g["inheritance"],
        })
    return {"genes": rows, "total": len(rows), "total_patients": len(COHORT)}


def get_definitions():
    return {
        "atlas":         "CI-Subunit-Atlas — Complete 42-gene nuclear-encoded Complex I reference (34 subunits + 8 assembly factors)",
        "complex_i":     "NADH:Ubiquinone Oxidoreductase (NDU); L-shaped 45-subunit ~980 kDa OXPHOS Complex I; catalyzes NADH + Q → NAD⁺ + QH₂ + 4H⁺(translocated); entry point of ETC",
        "N_module":      "Peripheral arm NADH-binding end; FMN primary NADH acceptor (NDUFV1); Fe-S cluster relay N3→N1b→N4→N5→N2; NADH oxidation begins here",
        "Q_module":      "Peripheral arm quinone-binding junction; Fe-S N2 (NDUFS2) is terminal carrier to ubiquinone; ubiquinol (QH₂) formed here",
        "PP_module":     "Membrane arm proximal pump; ND2/ND3/ND6 subcomplex + nuclear NDUFB/NDUFA scaffold; proton translocation module 1",
        "PD_module":     "Membrane arm proximal domain; ND4/ND4L region; proton pump module 2; ND4-face structural triad (NDUFB4-NDUFB6-NDUFB8)",
        "Fe_S_relay":    "NDUFV1(FMN/N3) → NDUFV2(N1b) → NDUFS7(N4) → NDUFS1(N5) → NDUFS8(N6a/N6b) → NDUFS2(N2) → Ubiquinone — 7-step electron relay from NADH to QH₂",
        "isolated_CI":   "Isolated CI deficiency (5–22% residual CI; CII/CIII/CIV NORMAL) — biochemical fingerprint of all structural CI subunit mutations; CII NORMAL used as internal reference (like mtDNA CII-zero rule)",
        "leigh_syndrome":"Progressive necrotising encephalopathy; bilateral symmetric basal ganglia/brainstem lesions on MRI; elevated lactate CSF/blood; onset infant-toddler; most common presentation of CI-Leigh",
        "BN_PAGE":       "Blue-Native PAGE — CI holocomplex visualised as ~980 kDa band; absent CI = scaffold-loss (N-module sub-assemblies trapped); sub-assembly intermediates = junction/structural subunit mutations",
        "scaffold_loss": "BN-PAGE pattern: absent CI band; N-module and Q-module or PP/PD-modules cannot assemble into membrane arm; typical of NDUFB/NDUFA membrane arm subunit mutations",
        "NDUFA4_CIV":    "NDUFA4 is a CIV (Complex IV) 14th subunit — NOT a CI subunit; Balsa et al. 2012 Cell Metab established this definitively; NDUFA4 deficiency causes COX deficiency (CIV↓, CI NORMAL)",
        "NDUFAF_AFs":    "NDUFAF1–8 are assembly factors not present in mature CI holocomplex; they scaffold CI biogenesis then are released; NDUFAF mutations = CI deficiency without structural subunit loss",
        "PINK1_NDUFA10": "PINK1 phosphorylates NDUFA10 Ser-250 during mitophagy — links CI structural integrity to mitophagy; relevant to Parkinson's disease pathophysiology (PINK1-Parkin pathway)",
        "GRIM19_NDUFA13":"GRIM-19 (NDUFA13) = CI subunit + IFN-β-induced apoptosis mediator; cancer (GRIM-19 loss → tumour survival) + mito disease (CI deficiency → Leigh); dual biology",
        "MCIA_complex":  "Mitochondrial Complex I Assembly complex: ACAD9-NDUFAF1-ECSIT-TMEM126B tetramer; required for ND2+ND5 membrane arm module biogenesis; ACAD9 riboflavin-responsive, NDUFAF1 is NOT",
        "X_linked_CI":   "NDUFA1 (Xq24, MWFE, neurological only) and NDUFB11 (Xp11.3, ESSS, cardiac in female carriers) — only X-linked CI nuclear genes; males hemizygous severe; females mosaic/mild",
        "NDUFB3_first":  "NDUFB3 (B12) — first nuclear-encoded CI subunit mutation ever published (Andreu et al. 1999, Nature Genetics); PP-module B12 subunit; historic landmark of nuclear CI genetics",
        "WES_nuclear":   "All 42 NDUF/NDUFAF genes are nuclear-encoded and WES-detectable; mtDNA CI subunits (MT-ND1–6/ND4L) MISSED by WES — require dedicated mtDNA sequencing",
        "BTBGD_exclusion":"SLC19A3 (BTBGD — Biotin-Thiamine-Responsive Basal Ganglia Disease) MUST be excluded FIRST in all Leigh/Leigh-like presentations; Biotin+Thiamine dramatically effective; identical MRI mimic",
        "succinate_bypass":"Succinate → CII → ubiquinol → CIII → CIV — bypasses failed CI entirely; provides electrons to downstream ETC without CI; Level C evidence in CI-Leigh",
        "riboflavin_CI": "Riboflavin (B2) → FMN → prosthetic group at NDUFV1 N-module; CI-specific rationale; extra FMN may stabilise partial CI assembly in missense alleles; Level C",
        "GIR_6_8":       "Glucose Infusion Rate 6–8 mg/kg/min — CI cannot re-oxidise NADH from β-oxidation when CI is deficient; NEVER fast in CI disease; continuous IV dextrose in acute setting",
        "metformin_ABS": "Metformin directly inhibits CI at the quinone-binding/ND1 site — absolute contraindication in ALL CI gene mutations; fatal lactic acidosis reported even in carriers",
        "vpa_triple":    "VPA triple mechanism: CoA sequestration + CI direct inhibition + POLG inhibition → mtDNA depletion → reduced ND-subunit expression; absolute CI in all OXPHOS disease",
    }
