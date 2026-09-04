#!/usr/bin/env python3
"""CIII-Subunit-Atlas — Complete 15-Gene Nuclear-Encoded Complex III (bc1 Complex) Atlas
9 structural subunits + 6 assembly factors (all nuclear-encoded)
600-patient aggregate cohort (15 × 40, seeds 710–724)

Complex III (ubiquinol-cytochrome c reductase / bc1 complex) facts:
  - 11 total subunits: 1 mtDNA-encoded (MT-CYB) + 10 nuclear-encoded structural subunits
  - Functions as a HOMODIMER (CIII₂) within the inner mitochondrial membrane
  - Q cycle: accepts electrons from ubiquinol (QH₂) produced by CI/CII/FAD-linked enzymes,
    transfers electrons to cytochrome c via the mobile Rieske [2Fe-2S] head domain
  - UQCRFS1 (RISP) is the only mobile subunit — its [2Fe-2S] head pivots between
    the Qo site (near MT-CYB) and the Cyt c1 (CYC1) domain during each Q-cycle step
  - 6 nuclear-encoded assembly factors orchestrate ordered CIII₂ biogenesis
  - MT-CYB (only mtDNA CIII subunit) — covered in MT-CYB dashboard separately
  - CII ALWAYS NORMAL in isolated CIII deficiency (CII = internal reference control)

ATLAS SCOPE (15 nuclear genes):
  Structural subunits (9 nuclear): UQCRC1, UQCRC2, CYC1, UQCRFS1, UQCRB, UQCRQ,
                                    UQCR10, UQCR11, UQCRH
  Assembly factors (6 nuclear):    BCS1L, TTC19, LYRM7, UQCC1, UQCC2, UQCC3
  mtDNA subunit (not in atlas):    MT-CYB (OMIM 516020) — covered in MT-Genome-Atlas + MT-CYB dashboard

PHENOTYPIC SPECTRUM:
  - GRACILE syndrome (BCS1L p.S78G Finnish founder) — most severe CIII phenotype
  - Bjornstad syndrome (BCS1L) — pili torti + SNHL WITHOUT metabolic crisis
  - Progressive neurodegeneration (TTC19) — adult/childhood cerebellar + white matter
  - Infantile Leigh/Leigh-like (UQCRFS1, UQCRC1, UQCC1, UQCC3)
  - Hypertrophic cardiomyopathy (UQCC2, UQCRC2)
  - Isolated leukoencephalopathy (UQCRC1)
  - Hypoglycaemia-prominent (UQCRB, UQCC3)
  - Pontine hypoplasia / brainstem (UQCRC2 subset)

BIOCHEMICAL FINGERPRINT (Isolated CIII Deficiency):
  - CIII (rotenone-resistant NADH-Cyt c reductase) markedly reduced
  - CI normal in pure structural subunit mutations; low in UQCRC1/UQCRC2 (supercomplex SC I+III₂)
  - CII ALWAYS NORMAL — internal reference (no mtDNA CII subunits)
  - CIV usually normal in isolated CIII; low in UQCRC1 (SC I+III₂+IV disrupted)
  - BN-PAGE: sub-complex CIII₂ (pre-CIII intermediate) accumulates in BCS1L, LYRM7, TTC19 (RISP insertion failure)

COHORT: 15 × 40 = 600 patient slots (seeds 710–724; gene-specific seeds)
"""

import random

SEED = 725
rng  = random.Random(SEED)

# ── All 15 nuclear-encoded CIII-related genes — authoritative table ───────────
# gene_class: "structural_subunit" | "assembly_factor"
CIII_GENES = [
    # ── Nuclear Structural Subunits ───────────────────────────────────────────
    {
        "gene": "UQCRC1",  "aa": "480 aa",  "kDa": "52.6 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "Core1",
        "ciii_module": "Core protein 1 (matrix arm) — MPP-α analogue; scaffolds CIII matrix face; CI+CIII supercomplex (SC I+III₂) contact surface",
        "omim_gene": 191329,  "chromosome": "3p21.31",  "seed": 710,
        "phenotype": "Isolated CIII Deficiency + Leukoencephalopathy (combined CI+CIII via SC)",
        "disease": "CIII Deficiency — Nuclear Type 5: leukoencephalopathy, combined CI+CIII due to SC I+III₂ destabilisation",
        "disease_omim": 616111,  "inheritance": "AR",
        "hallmark": "COMBINED CI+CIII deficiency via supercomplex (SC I+III₂) destabilisation — NOT isolated CIII; white matter T2 changes predominate; hepatopathy 45%",
        "key_ddx": "vs UQCRC2: Core 2 → more cardiac; UQCRC1 → more white matter; vs GRACILE (BCS1L): multi-organ vs white matter; vs primary CI deficiency: check CIII too",
        "founder_variant": "p.Arg183Gln (UK/European recurrent); p.Ala359Thr (splice-site hotspot)",
        "leigh_mri_rate": 0.30, "leuko_rate": 0.65, "cardiac_rate": 0.25,
        "hepatopathy_rate": 0.45, "gracile_rate": 0.00, "bjornstad_rate": 0.00,
        "ci_also_low": True,  "ciii_activity_mean": 12.0, "ciii_activity_sd": 5.0,
    },
    {
        "gene": "UQCRC2",  "aa": "453 aa",  "kDa": "48.3 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "Core2",
        "ciii_module": "Core protein 2 (matrix arm) — MPP-β analogue; processes MT-CYB presequence; CI+CIII supercomplex contact; pontine scaffold",
        "omim_gene": 602453,  "chromosome": "16p12.2",  "seed": 711,
        "phenotype": "Isolated CIII Deficiency + Early-onset HCM / Pontine Hypoplasia",
        "disease": "CIII Deficiency — Nuclear Type 6: hypertrophic cardiomyopathy, pontine hypoplasia, combined CI+CIII via SC I+III₂",
        "disease_omim": 615501,  "inheritance": "AR",
        "hallmark": "Prominent HCM (55%) distinguishes from other CIII nuclear defects; pontine hypoplasia on MRI (40%) — unique CIII structural hallmark; combined CI+CIII via SC destabilisation",
        "key_ddx": "vs UQCRC1: Core 1 → leukoencephalopathy > cardiac; UQCRC2 → cardiac + pontine; vs UQCC2 (most cardiac AF): UQCC2 = pure cardiac, UQCRC2 = cardiac + neurological",
        "founder_variant": "p.Pro429Leu (Middle Eastern recurrent); p.Gly374Asp",
        "leigh_mri_rate": 0.40, "leuko_rate": 0.30, "cardiac_rate": 0.55,
        "hepatopathy_rate": 0.20, "gracile_rate": 0.00, "bjornstad_rate": 0.00,
        "ci_also_low": True,  "ciii_activity_mean": 14.0, "ciii_activity_sd": 5.0,
    },
    {
        "gene": "CYC1",  "aa": "325 aa",  "kDa": "35.4 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "Cyt-c1",
        "ciii_module": "Cytochrome c1 — haem c binding; terminal electron acceptor within CIII; transfers electrons to free cytochrome c (Cyt c); fixed subunit at CIII IMS face",
        "omim_gene": 123980,  "chromosome": "8q24.13",  "seed": 712,
        "phenotype": "Isolated CIII Deficiency with lactic acidaemia and encephalomyopathy",
        "disease": "CIII Deficiency — CYC1-related: lactic acidaemia, encephalomyopathy, exercise intolerance, Leigh-like MRI",
        "disease_omim": 615453,  "inheritance": "AR",
        "hallmark": "ISOLATED CIII deficiency (CI normal, CII normal, CIV normal); haem c attachment by HCCS (not by CYC1 itself); Cyt c1 loss → CIII complex instability; exercise intolerance common early presentation",
        "key_ddx": "vs UQCRFS1 (RISP): both isolated CIII; RISP → pre-CIII on BN-PAGE; CYC1 → no pre-CIII accumulation; vs MT-CYB mtDNA: CYC1 = nuclear WES-detectable; maternal vs AR inheritance",
        "founder_variant": "p.Gly270Asp (European); p.Arg217Ter (truncating, severe)",
        "leigh_mri_rate": 0.55, "leuko_rate": 0.20, "cardiac_rate": 0.15,
        "hepatopathy_rate": 0.20, "gracile_rate": 0.00, "bjornstad_rate": 0.00,
        "ci_also_low": False,  "ciii_activity_mean": 10.0, "ciii_activity_sd": 4.0,
    },
    {
        "gene": "UQCRFS1",  "aa": "274 aa",  "kDa": "29.6 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "RISP",
        "ciii_module": "Rieske iron-sulfur protein (RISP/ISP) — mobile [2Fe-2S] head domain pivots Qo→Cyt-c1 site during Q-cycle; only mobile CIII subunit; inserted last by BCS1L",
        "omim_gene": 191327,  "chromosome": "19q12",  "seed": 713,
        "phenotype": "Isolated CIII Deficiency — neurodegeneration, progressive; BN-PAGE pre-CIII accumulation",
        "disease": "CIII Deficiency Nuclear Type 2 — UQCRFS1: progressive neurodegeneration, lactic acidosis, pre-CIII₂ sub-complex on BN-PAGE",
        "disease_omim": 615157,  "inheritance": "AR",
        "hallmark": "PRE-CIII SUB-COMPLEX on BN-PAGE (pathognomonic for RISP insertion failure — same as BCS1L, LYRM7); RISP is the last subunit inserted (BCS1L-dependent); progressive neurodegeneration not acute Leigh",
        "key_ddx": "vs BCS1L: BCS1L = GRACILE/Bjornstad (clinically different); vs LYRM7: LYRM7 = earlier assembly step; both show pre-CIII BN-PAGE — distinguish by WES; vs MT-CYB (mtDNA) mutations",
        "founder_variant": "p.Arg56Gln (Mediterranean founder); p.Tyr155Cys (severe early-onset)",
        "leigh_mri_rate": 0.40, "leuko_rate": 0.35, "cardiac_rate": 0.10,
        "hepatopathy_rate": 0.15, "gracile_rate": 0.00, "bjornstad_rate": 0.00,
        "ci_also_low": False,  "ciii_activity_mean": 8.0, "ciii_activity_sd": 4.0,
    },
    {
        "gene": "UQCRB",  "aa": "109 aa",  "kDa": "13.4 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "Sub6-Qi",
        "ciii_module": "Subunit 6 / QP-C — peripheral matrix-face; Qi site proximity; contacts UQCRC1+UQCRC2; stabilises matrix arm–membrane junction",
        "omim_gene": 191339,  "chromosome": "8q22.1",  "seed": 714,
        "phenotype": "Isolated CIII Deficiency — hypoglycaemia, lactic acidosis (Haut 2003)",
        "disease": "CIII Deficiency — UQCRB: hypoglycaemia, lactic acidosis, hepatopathy, infantile onset; Haut 2003 first UQCRB report",
        "disease_omim": 615453,  "inheritance": "AR",
        "hallmark": "HYPOGLYCAEMIA (65%) most prominent metabolic feature — Qi site defect impairs gluconeogenesis via NADH/lactate; Haut-2003 landmark (HumGenet) first nuclear CIII structural subunit mutation; BN-PAGE: sub-complexes partially present",
        "key_ddx": "vs UQCRC1/UQCRC2: those cause combined CI+CIII; UQCRB = isolated CIII; vs UQCC3 (also hypoglycaemia-prominent AF): UQCC3 more severe neonatal; UQCRB = structural subunit",
        "founder_variant": "p.Lys74Asnfs (c.221_222delAA, Exon5 frameshift — Haut 2003 French infant); p.Gly43Ser; p.Arg85Trp",
        "leigh_mri_rate": 0.30, "leuko_rate": 0.15, "cardiac_rate": 0.10,
        "hepatopathy_rate": 0.55, "gracile_rate": 0.00, "bjornstad_rate": 0.00,
        "ci_also_low": False,  "ciii_activity_mean": 13.0, "ciii_activity_sd": 6.0,
    },
    {
        "gene": "UQCRQ",  "aa": "82 aa",  "kDa": "9.7 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "Sub7-Qo",
        "ciii_module": "Subunit 7 / QCR7 — peripheral intermembrane space face; Qo site proximity; stabilises RISP head in Qo position; required for CIII dimer interface",
        "omim_gene": 612080,  "chromosome": "5q31.1",  "seed": 715,
        "phenotype": "Isolated CIII Deficiency — severe early-onset encephalomyopathy, dystonia, optic atrophy",
        "disease": "CIII Deficiency — UQCRQ: severe early-onset encephalomyopathy, dystonia 75%, optic atrophy 42%, CIII 8-22% residual",
        "disease_omim": 615453,  "inheritance": "AR",
        "hallmark": "DYSTONIA (75%) + OPTIC ATROPHY (42%) distinguish from other CIII nuclear subunit defects; CIII 8-22% residual (surprisingly severe despite peripheral subunit); supercomplex destabilisation contributes; similar to GRACILE in severity but different multi-organ pattern",
        "key_ddx": "vs UQCRB: UQCRB = hypoglycaemia; UQCRQ = dystonia + optic atrophy; vs BCS1L GRACILE: GRACILE = cholestasis + iron overload + neonatal; UQCRQ = neurological dominant",
        "founder_variant": "p.Trp45Ter (severe, UK); p.Gly57Asp (CIII Qo-site helix breaker, Italian)",
        "leigh_mri_rate": 0.50, "leuko_rate": 0.25, "cardiac_rate": 0.15,
        "hepatopathy_rate": 0.25, "gracile_rate": 0.00, "bjornstad_rate": 0.00,
        "ci_also_low": False,  "ciii_activity_mean": 15.0, "ciii_activity_sd": 6.0,
    },
    {
        "gene": "UQCR10",  "aa": "63 aa",  "kDa": "7.5 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "Sub10",
        "ciii_module": "Subunit 10 — membrane-spanning; stabilises CIII dimer interface; required for final CIII₂ homodimerisation; 22q13.1",
        "omim_gene": 609006,  "chromosome": "22q13.1",  "seed": 716,
        "phenotype": "Isolated CIII Deficiency — severe infantile lactic acidosis, multi-organ failure",
        "disease": "CIII Deficiency — UQCR10: rare; severe infantile CIII deficiency, lactic acidosis, encephalopathy",
        "disease_omim": 615453,  "inheritance": "AR",
        "hallmark": "RARE (< 10 families worldwide); severe infantile CIII deficiency; dimer interface destabilisation → CIII₂ fails to form; BN-PAGE: CIII monomer instead of dimer",
        "key_ddx": "vs UQCR11: adjacent in assembly pathway; both rare; vs BCS1L (GRACILE): cholestasis + iron overload distinguish BCS1L; UQCR10 = isolated CIII without GRACILE features",
        "founder_variant": "p.Leu42Pro (first reported family, severe neonatal); c.IVS2+1G>A splice",
        "leigh_mri_rate": 0.45, "leuko_rate": 0.20, "cardiac_rate": 0.20,
        "hepatopathy_rate": 0.30, "gracile_rate": 0.00, "bjornstad_rate": 0.00,
        "ci_also_low": False,  "ciii_activity_mean": 9.0, "ciii_activity_sd": 4.0,
    },
    {
        "gene": "UQCR11",  "aa": "56 aa",  "kDa": "6.5 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "Sub11",
        "ciii_module": "Subunit 11 — smallest structural CIII subunit; membrane-spanning; stabilises dimer interface with UQCR10; 19q13.41",
        "omim_gene": 617145,  "chromosome": "19q13.41",  "seed": 717,
        "phenotype": "Isolated CIII Deficiency — severe infantile-onset, encephalomyopathy",
        "disease": "CIII Deficiency — UQCR11: rare; severe infantile CIII deficiency, lactic acidaemia, hypotonia",
        "disease_omim": 615453,  "inheritance": "AR",
        "hallmark": "SMALLEST CIII structural subunit (6.5 kDa, 56 aa); extremely rare mutations; severe CIII deficiency despite peripheral positioning; stabilises UQCR10 at dimer interface",
        "key_ddx": "vs UQCR10: structurally adjacent, clinically similar; WES distinguishes; vs UQCRH (hinge protein on chromosome 1p33): UQCRH = IMS face, UQCR11 = dimer interface",
        "founder_variant": "p.Arg34Ter (truncating, Israeli family); p.Gly29Val (missense, Belgian)",
        "leigh_mri_rate": 0.40, "leuko_rate": 0.20, "cardiac_rate": 0.15,
        "hepatopathy_rate": 0.25, "gracile_rate": 0.00, "bjornstad_rate": 0.00,
        "ci_also_low": False,  "ciii_activity_mean": 10.0, "ciii_activity_sd": 5.0,
    },
    {
        "gene": "UQCRH",  "aa": "91 aa",  "kDa": "9.5 kDa",
        "gene_class": "structural_subunit",  "subunit_series": "Hinge",
        "ciii_module": "Hinge protein (subunit 9 / QCR9) — IMS face; contacts Cyt c (CYC1) and Cyt c electron acceptor; facilitates rapid Cyt c turnover at CIII IMS exit; 1p33",
        "omim_gene": 136100,  "chromosome": "1p33",  "seed": 718,
        "phenotype": "Isolated CIII Deficiency — lactic acidosis, exercise intolerance, encephalomyopathy",
        "disease": "CIII Deficiency — UQCRH: rare; lactic acidosis, exercise intolerance, Leigh-like MRI; IMS electron-transfer impaired",
        "disease_omim": 615453,  "inheritance": "AR",
        "hallmark": "HINGE PROTEIN facilitates rapid Cyt c electron exit from CIII IMS face; loss → Cyt c bottleneck → CIII backup → lactic acidosis; IMS localisation (unlike matrix-face Core 1/2); exercise intolerance early sign",
        "key_ddx": "vs CYC1: CYC1 = haem c1 electron input; UQCRH = Cyt c exit facilitation; both IMS face; clinical overlap; vs UQCR10/11: dimer interface (membrane) vs hinge (IMS face)",
        "founder_variant": "p.Phe42Leu (French founder); p.Glu71Ter (truncating, Turkish)",
        "leigh_mri_rate": 0.45, "leuko_rate": 0.20, "cardiac_rate": 0.10,
        "hepatopathy_rate": 0.15, "gracile_rate": 0.00, "bjornstad_rate": 0.00,
        "ci_also_low": False,  "ciii_activity_mean": 11.0, "ciii_activity_sd": 5.0,
    },
    # ── Assembly Factors ──────────────────────────────────────────────────────
    {
        "gene": "BCS1L",  "aa": "419 aa",  "kDa": "47.5 kDa",
        "gene_class": "assembly_factor",  "subunit_series": "AF-BCS1L",
        "ciii_module": "BCS1 homologue (AAA+ ATPase) — inserts UQCRFS1 (RISP) [2Fe-2S] domain into pre-CIII scaffold (the critical last assembly step: pre-CIII → CIII₂)",
        "omim_gene": 603647,  "chromosome": "2q35",  "seed": 719,
        "phenotype": "GRACILE syndrome / Bjornstad syndrome (same gene, phenotype-genotype spectrum)",
        "disease": "GRACILE Syndrome (OMIM 603358) — Growth restriction, Aminoaciduria, Cholestasis, Iron overload, Lactic acidosis, Early death; OR Bjornstad Syndrome (OMIM 262000) — Pili torti + SNHL",
        "disease_omim": 603358,  "inheritance": "AR",
        "hallmark": "WIDEST PHENOTYPIC RANGE of all CIII genes: p.Ser78Gly (Finnish founder) = GRACILE (neonatal multi-organ failure, <5% survival >1yr); milder missense = Bjornstad (hair + hearing, NO metabolic crisis); BN-PAGE: pre-CIII sub-complex (RISP not yet inserted)",
        "key_ddx": "GRACILE vs UQCRFS1/LYRM7: all show pre-CIII on BN-PAGE; GRACILE = cholestasis + iron overload + aminoaciduria = diagnostic triad; Bjornstad = pili torti + SNHL without metabolic crisis",
        "founder_variant": "p.Ser78Gly (c.232A>G, Finnish founder — 1/36 carrier freq in Finland) → GRACILE; milder alleles p.Arg144Gln, p.Gln302Arg → Bjornstad",
        "leigh_mri_rate": 0.15, "leuko_rate": 0.10, "cardiac_rate": 0.10,
        "hepatopathy_rate": 0.88, "gracile_rate": 0.55, "bjornstad_rate": 0.30,
        "ci_also_low": False,  "ciii_activity_mean": 7.0, "ciii_activity_sd": 3.0,
    },
    {
        "gene": "TTC19",  "aa": "380 aa",  "kDa": "44 kDa",
        "gene_class": "assembly_factor",  "subunit_series": "AF-TTC19",
        "ciii_module": "Tetratricopeptide repeat protein 19 — promotes UQCRFS1 maturation AFTER BCS1L insertion; required for CIII supercomplex (SC III₂+IV) formation; 17p12",
        "omim_gene": 613814,  "chromosome": "17p12",  "seed": 720,
        "phenotype": "Progressive neurodegeneration — childhood/adult onset; cerebellar atrophy; psychiatric features",
        "disease": "CIII Deficiency Nuclear Type 4 (OMIM 615159) — progressive neurodegeneration, cerebellar atrophy, psychiatric features (psychosis, depression), white matter changes",
        "disease_omim": 615159,  "inheritance": "AR",
        "hallmark": "LATER ONSET than other CIII AFs (median 4-6 years; adult cases reported); PSYCHIATRIC FEATURES (40% — psychosis, depression) distinguish from acute Leigh; cerebellar atrophy + white matter on MRI; NOT GRACILE (no cholestasis/iron/aminoaciduria); BN-PAGE: pre-CIII accumulates",
        "key_ddx": "vs BCS1L: BCS1L acute neonatal/infant; TTC19 = progressive childhood/adult; vs LYRM7: LYRM7 = acute infantile; TTC19 = progressive; psychiatric features unique to TTC19 among CIII genes",
        "founder_variant": "p.Lys319Ter (truncating, Spanish/Portuguese); p.Glu381Ter; p.Ala101Thr (Turkish founder)",
        "leigh_mri_rate": 0.25, "leuko_rate": 0.60, "cardiac_rate": 0.05,
        "hepatopathy_rate": 0.10, "gracile_rate": 0.00, "bjornstad_rate": 0.00,
        "ci_also_low": False,  "ciii_activity_mean": 18.0, "ciii_activity_sd": 8.0,
    },
    {
        "gene": "LYRM7",  "aa": "121 aa",  "kDa": "14 kDa",
        "gene_class": "assembly_factor",  "subunit_series": "AF-LYRM7",
        "ciii_module": "LYRM motif chaperone (MZM1L homologue) — stabilises UQCRFS1 [2Fe-2S] domain BEFORE BCS1L-dependent insertion into pre-CIII; earliest RISP-specific assembly step; 5q31",
        "omim_gene": 615831,  "chromosome": "5q31",  "seed": 721,
        "phenotype": "Isolated CIII Deficiency — acute severe infantile onset; encephalopathy; lactic acidosis",
        "disease": "CIII Deficiency Nuclear Type 8 (OMIM 615838) — acute infantile CIII deficiency, lactic acidosis, encephalopathy, pre-CIII₂ sub-complex",
        "disease_omim": 615838,  "inheritance": "AR",
        "hallmark": "EARLIEST RISP ASSEMBLY STEP — LYRM7 acts before BCS1L; BN-PAGE: pre-CIII (same pattern as BCS1L and TTC19); ACUTE SEVERE INFANTILE onset unlike TTC19 (progressive); LYRM motif (LYR tripeptide) shared with SDHAF1 (CII) — LYRM proteins are a class of mitochondrial Fe-S chaperones",
        "key_ddx": "vs BCS1L: BCS1L = GRACILE/Bjornstad (clinical clue: cholestasis/pili torti); LYRM7 = acute infantile without GRACILE features; vs TTC19: TTC19 = progressive/psychiatric; LYRM7 = acute infant",
        "founder_variant": "p.Arg45Gln (first family — Dallabona 2015 AJHG); p.Cys95Ser (Turkish founder); p.Leu76Pro",
        "leigh_mri_rate": 0.55, "leuko_rate": 0.20, "cardiac_rate": 0.10,
        "hepatopathy_rate": 0.20, "gracile_rate": 0.00, "bjornstad_rate": 0.00,
        "ci_also_low": False,  "ciii_activity_mean": 8.0, "ciii_activity_sd": 3.0,
    },
    {
        "gene": "UQCC1",  "aa": "131 aa",  "kDa": "14.5 kDa",
        "gene_class": "assembly_factor",  "subunit_series": "AF-UQCC1",
        "ciii_module": "Ubiquinol-cytochrome c reductase complex assembly factor 1 (BRAWNIN) — early CIII assembly: scaffolds newly synthesised MT-CYB within the inner membrane; works with UQCC2 (MT-CYB module); 20p13",
        "omim_gene": 616047,  "chromosome": "20p13",  "seed": 722,
        "phenotype": "Isolated CIII Deficiency — severe neonatal/infantile onset; lactic acidosis; encephalomyopathy",
        "disease": "CIII Deficiency — UQCC1/BRAWNIN: severe neonatal CIII deficiency, lactic acidosis, hypotonia, encephalomyopathy; BN-PAGE: no assembled CIII₂",
        "disease_omim": 616111,  "inheritance": "AR",
        "hallmark": "EARLIEST CIII ASSEMBLY STEP — UQCC1 scaffolds MT-CYB before RISP-related steps; without UQCC1, MT-CYB cannot be stably incorporated → no CIII₂ on BN-PAGE (unlike pre-CIII in BCS1L/LYRM7); severe neonatal onset; UQCC1 interacts with UQCC2 (MT-CYB module)",
        "key_ddx": "vs UQCC2: both MT-CYB module, both cause severe CIII; UQCC2 → more cardiac; UQCC1 → more neurological; vs BCS1L: UQCC1 = no pre-CIII (MT-CYB not assembled); BCS1L = pre-CIII present (MT-CYB assembled, RISP missing)",
        "founder_variant": "p.Leu98Pro (Chinese founder — Tang 2020 AJHG); p.Trp107Ter (truncating European)",
        "leigh_mri_rate": 0.55, "leuko_rate": 0.25, "cardiac_rate": 0.20,
        "hepatopathy_rate": 0.30, "gracile_rate": 0.00, "bjornstad_rate": 0.00,
        "ci_also_low": False,  "ciii_activity_mean": 6.0, "ciii_activity_sd": 3.0,
    },
    {
        "gene": "UQCC2",  "aa": "259 aa",  "kDa": "29 kDa",
        "gene_class": "assembly_factor",  "subunit_series": "AF-UQCC2",
        "ciii_module": "Ubiquinol-cytochrome c reductase complex assembly factor 2 — stabilises newly synthesised MT-CYB; works with UQCC1 in the MT-CYB scaffolding module; 6p21.1",
        "omim_gene": 614461,  "chromosome": "6p21.1",  "seed": 723,
        "phenotype": "Isolated CIII Deficiency — infantile cardiomyopathy dominant; highest cardiac rate of all CIII genes",
        "disease": "CIII Deficiency Nuclear Type 6 (OMIM 615178) — infantile cardiomyopathy (80%), lactic acidosis, CIII deficiency; Gilbertson 2019 landmark paper",
        "disease_omim": 615178,  "inheritance": "AR",
        "hallmark": "HIGHEST CARDIAC RATE (80%) of ALL CIII genes — makes UQCC2 the most cardiomyopathy-associated CIII assembly factor; distinguishes from BCS1L (cholestasis/hair) and TTC19 (neurological); UQCC2 loss → MT-CYB instability → CIII assembly block at earliest MT-CYB step",
        "key_ddx": "vs UQCRC2 (Core 2 structural): both cause cardiomyopathy; UQCRC2 = combined CI+CIII + pontine hypoplasia; UQCC2 = isolated CIII + pure cardiac; vs UQCC1: UQCC1 = more neurological, less cardiac",
        "founder_variant": "p.Arg134Ter (Australian consanguineous — Gilbertson 2019 J Inherit Metab Dis); p.Tyr161Asp; p.Leu183Pro",
        "leigh_mri_rate": 0.20, "leuko_rate": 0.10, "cardiac_rate": 0.80,
        "hepatopathy_rate": 0.15, "gracile_rate": 0.00, "bjornstad_rate": 0.00,
        "ci_also_low": False,  "ciii_activity_mean": 9.0, "ciii_activity_sd": 4.0,
    },
    {
        "gene": "UQCC3",  "aa": "112 aa",  "kDa": "12.5 kDa",
        "gene_class": "assembly_factor",  "subunit_series": "AF-UQCC3",
        "ciii_module": "Ubiquinol-cytochrome c reductase complex assembly factor 3 — late-stage CIII assembly; stabilises final CIII₂ homodimer formation; promotes supercomplex (SC III₂+IV) integration; 11q12.1",
        "omim_gene": 616652,  "chromosome": "11q12.1",  "seed": 724,
        "phenotype": "Isolated CIII Deficiency — neonatal lactic acidosis, hypoglycaemia, respiratory failure",
        "disease": "CIII Deficiency Nuclear Type 9 (OMIM 616111) — neonatal lactic acidosis, hypoglycaemia (60%), respiratory distress; some improve with age",
        "disease_omim": 616111,  "inheritance": "AR",
        "hallmark": "HYPOGLYCAEMIA (60%) alongside neonatal lactic acidosis — distinguishes from most other CIII AFs; LATE-STAGE assembly factor → CIII₂ dimer and SC III₂+IV fail; BN-PAGE: CIII monomer without dimer; some patients improve with metabolic support (neonatal lactic acidosis may stabilise)",
        "key_ddx": "vs UQCRB (structural subunit also hypoglycaemia-prominent): UQCRB older infant onset; UQCC3 = neonatal; vs BCS1L GRACILE: GRACILE = cholestasis + iron + aminoaciduria; UQCC3 = hypoglycaemia without GRACILE triad",
        "founder_variant": "p.Glu30Ter (truncating, severe neonatal — Wanschers 2014 AJHG); p.Leu79Pro (missense, milder); p.Gly75Arg",
        "leigh_mri_rate": 0.35, "leuko_rate": 0.15, "cardiac_rate": 0.20,
        "hepatopathy_rate": 0.35, "gracile_rate": 0.00, "bjornstad_rate": 0.00,
        "ci_also_low": False,  "ciii_activity_mean": 11.0, "ciii_activity_sd": 4.0,
    },
]

# ── Patient cohort generation ─────────────────────────────────────────────────
def _gen_patient(gene_info, idx):
    rg = random.Random(gene_info["seed"] * 1000 + idx)
    gene = gene_info["gene"]

    # Onset model
    if gene_info["gracile_rate"] > 0.4:
        # BCS1L — GRACILE vs Bjornstad split
        if rg.random() < gene_info["gracile_rate"]:
            age_onset_months = int(rg.gauss(0.5, 0.3)); age_onset_months = max(0, age_onset_months)
            phenotype = "GRACILE"
        elif rg.random() < (gene_info["bjornstad_rate"] / (1 - gene_info["gracile_rate"] + 0.01)):
            age_onset_months = int(rg.gauss(24, 12)); age_onset_months = max(3, age_onset_months)
            phenotype = "Bjornstad"
        else:
            age_onset_months = int(rg.gauss(6, 4)); age_onset_months = max(0, age_onset_months)
            phenotype = "CIII-Leigh"
    elif gene_info["gene"] == "TTC19":
        # Progressive — later onset
        if rg.random() < 0.40:
            age_onset_months = int(rg.gauss(48, 24)); age_onset_months = max(6, age_onset_months)
        else:
            age_onset_months = int(rg.gauss(120, 60)); age_onset_months = max(24, age_onset_months)
        phenotype = "Progressive-Neuro"
    elif gene_info["leigh_mri_rate"] > 0.40:
        # Leigh-dominant
        age_onset_months = int(rg.gauss(8, 6)); age_onset_months = max(1, age_onset_months)
        phenotype = "CIII-Leigh"
    else:
        # Mixed / encephalomyopathy
        age_onset_months = int(rg.gauss(6, 5)); age_onset_months = max(0, age_onset_months)
        phenotype = "Encephalomyopathy"

    # CIII enzyme activity (% of mean normal)
    ciii_pct = max(2.0, min(80.0,
                             rg.gauss(gene_info["ciii_activity_mean"], gene_info["ciii_activity_sd"])))

    # CI low (Core 1/Core 2 supercomplex destabilisation)
    ci_also_low = gene_info["ci_also_low"] and rg.random() < 0.75

    # Clinical features
    leigh_mri    = rg.random() < gene_info["leigh_mri_rate"]
    leuko        = rg.random() < gene_info["leuko_rate"]
    cardiomyop   = rg.random() < gene_info["cardiac_rate"]
    hepatopathy  = rg.random() < gene_info["hepatopathy_rate"]
    gracile_full = (phenotype == "GRACILE") and rg.random() < 0.85
    bjornstad    = (phenotype == "Bjornstad")
    pili_torti   = bjornstad and rg.random() < 0.90
    snhl         = bjornstad and rg.random() < 0.80
    psychosis    = (gene == "TTC19") and rg.random() < 0.40
    iron_overload= gracile_full and rg.random() < 0.78
    aminoaciduria= gracile_full and rg.random() < 0.82
    cholestasis  = gracile_full and rg.random() < 0.88
    hypoglycemia = gene in ("UQCRB", "UQCC3") and rg.random() < 0.60
    lactic_ac    = rg.random() < (0.95 if phenotype in ("GRACILE","CIII-Leigh") else 0.65)
    hypotonia    = rg.random() < 0.70
    dystonia     = (gene == "UQCRQ") and rg.random() < 0.75
    optic_atr    = (gene == "UQCRQ") and rg.random() < 0.42
    cerebellar   = (gene == "TTC19") and rg.random() < 0.70
    pre_ciii_bnpage = gene in ("UQCRFS1", "BCS1L", "TTC19", "LYRM7") and rg.random() < 0.88

    return {
        "patient_id":          f"{gene}-{idx:03d}",
        "gene":                gene,
        "gene_class":          gene_info["gene_class"],
        "phenotype":           phenotype,
        "age_onset_months":    age_onset_months,
        "ciii_activity_pct":  round(ciii_pct, 1),
        "ci_also_low":         ci_also_low,
        "leigh_mri":           leigh_mri,
        "leukoencephalopathy": leuko,
        "cardiomyopathy":      cardiomyop,
        "hepatopathy":         hepatopathy,
        "gracile_syndrome":    gracile_full,
        "bjornstad_syndrome":  bjornstad,
        "pili_torti":          pili_torti,
        "snhl":                snhl,
        "psychosis":           psychosis,
        "iron_overload":       iron_overload,
        "aminoaciduria":       aminoaciduria,
        "cholestasis":         cholestasis,
        "hypoglycaemia":       hypoglycemia,
        "lactic_acidosis":     lactic_ac,
        "hypotonia":           hypotonia,
        "dystonia":            dystonia,
        "optic_atrophy":       optic_atr,
        "cerebellar_atrophy":  cerebellar,
        "pre_ciii_bnpage":     pre_ciii_bnpage,
    }


COHORT = []
for g in CIII_GENES:
    for i in range(40):
        COHORT.append(_gen_patient(g, i))


def get_overview():
    n_genes      = len(CIII_GENES)
    n_structural = sum(1 for g in CIII_GENES if g["gene_class"] == "structural_subunit")
    n_af         = sum(1 for g in CIII_GENES if g["gene_class"] == "assembly_factor")
    n_patients   = len(COHORT)

    gracile_pts  = [p for p in COHORT if p["gracile_syndrome"]]
    bjornstad_pts= [p for p in COHORT if p["bjornstad_syndrome"]]
    leigh_pts    = [p for p in COHORT if p["leigh_mri"]]
    leuko_pts    = [p for p in COHORT if p["leukoencephalopathy"]]
    cardiac_pts  = [p for p in COHORT if p["cardiomyopathy"]]
    hepato_pts   = [p for p in COHORT if p["hepatopathy"]]
    lactic_pts   = [p for p in COHORT if p["lactic_acidosis"]]
    pre_ciii_pts = [p for p in COHORT if p["pre_ciii_bnpage"]]
    ci_low_pts   = [p for p in COHORT if p["ci_also_low"]]

    return {
        "atlas": "CIII-Subunit-Atlas — Complete 15-Gene Nuclear-Encoded Complex III Reference",
        "complex_iii": {
            "full_name":                "Ubiquinol-Cytochrome c Reductase (bc1 complex) / Complex III",
            "subunits_total":           11,
            "subunits_nuclear":         10,
            "subunits_mtDNA":           1,
            "mtDNA_subunit":            "MT-CYB (cytochrome b) — only mtDNA-encoded CIII subunit; covered in MT-Genome-Atlas + MT-CYB dashboard",
            "assembly_factors":         6,
            "total_nuclear_genes":      n_genes,
            "functional_dimer":         "CIII₂ — obligate homodimer in vivo; each monomer 11 subunits; ~480 kDa per monomer",
            "function_q_cycle":         "Oxidises ubiquinol (QH₂) → ubiquinone (Q); transfers 2 electrons to 2× cytochrome c; pumps 4 H⁺/2e⁻ across IMM (Q cycle)",
            "risp_mobile":              "UQCRFS1 (RISP) is the ONLY mobile CIII subunit — [2Fe-2S] head pivots between Qo (MT-CYB) and CYC1 domains during each Q-cycle step",
            "supercomplex":             "CIII₂ assembles into respirasome supercomplexes: SC I+III₂ and SC I+III₂+IV; UQCRC1/UQCRC2 mutations → combined CI+CIII via SC destabilisation",
        },
        "atlas_scope": {
            "nuclear_structural_subunits": n_structural,
            "nuclear_assembly_factors":    n_af,
            "total_nuclear_genes":         n_genes,
            "mtDNA_subunit_note":          "MT-CYB (1 mtDNA subunit) excluded from this nuclear atlas; WES MISSES MT-CYB — requires dedicated mtDNA panel",
            "total_patients":              n_patients,
            "patients_per_gene":           40,
            "seeds":                       "710–724 (gene-specific)",
        },
        "biochemical_fingerprint": {
            "ciii_deficiency":     "CIII (rotenone-resistant NADH-Cyt c reductase) markedly reduced — hallmark of all 15 genes in this atlas",
            "cii_always_normal":   "CII ALWAYS NORMAL — no mtDNA-encoded CII subunits; CII = internal reference (same rule as for all non-CII OXPHOS defects)",
            "ci_also_low":         "CI also low in UQCRC1/UQCRC2 mutations — supercomplex (SC I+III₂) destabilisation; NOT true isolated CIII",
            "civ_usually_normal":  "CIV usually normal in isolated CIII deficiency; low in UQCRC1 (SC I+III₂+IV)",
            "pre_ciii_bnpage":     "Pre-CIII₂ sub-complex on BN-PAGE = pathognomonic for RISP-insertion failure: BCS1L, LYRM7, TTC19, UQCRFS1 mutations",
        },
        "phenotypic_spectrum": {
            "GRACILE": {
                "gene": "BCS1L",
                "features": "Growth restriction + Aminoaciduria + Cholestasis + Iron overload + Lactic acidosis + Early death",
                "onset": "Neonatal / early infantile",
                "survival": "<5% survive past 1 year (p.Ser78Gly homozygotes)",
                "founder": "p.Ser78Gly — Finnish founder, carrier freq 1:36 in Finland",
            },
            "Bjornstad": {
                "gene": "BCS1L",
                "features": "Pili torti (twisted hair) + SNHL — WITHOUT metabolic crisis or lactic acidosis",
                "onset": "Childhood (hair changes noticed 1-3 years)",
                "note": "Same gene as GRACILE — milder BCS1L alleles; normal lifespan",
            },
            "Progressive_Neurodegeneration": {
                "gene": "TTC19",
                "features": "Cerebellar atrophy + white matter changes + psychiatric features (psychosis/depression 40%)",
                "onset": "Childhood to adult (median 4-6 years; adult cases reported)",
                "note": "Only CIII gene causing prominent psychiatric features",
            },
            "Leigh_like": {
                "genes": "UQCRFS1, CYC1, UQCRQ, LYRM7, UQCC1",
                "features": "Basal ganglia T2 changes, lactic acidosis, hypotonia, regression",
                "onset": "Infantile",
            },
            "Cardiomyopathy_dominant": {
                "genes": "UQCC2 (80%), UQCRC2 (55%)",
                "features": "Hypertrophic cardiomyopathy predominant; UQCC2 = isolated CIII; UQCRC2 = combined CI+CIII + pontine",
            },
        },
        "aggregate_clinical": {
            "gracile_pct":      round(len(gracile_pts) / n_patients * 100, 1),
            "bjornstad_pct":    round(len(bjornstad_pts) / n_patients * 100, 1),
            "leigh_mri_pct":    round(len(leigh_pts) / n_patients * 100, 1),
            "leuko_pct":        round(len(leuko_pts) / n_patients * 100, 1),
            "cardiac_pct":      round(len(cardiac_pts) / n_patients * 100, 1),
            "hepatopathy_pct":  round(len(hepato_pts) / n_patients * 100, 1),
            "lactic_ac_pct":    round(len(lactic_pts) / n_patients * 100, 1),
            "pre_ciii_bnpage_pct": round(len(pre_ciii_pts) / n_patients * 100, 1),
            "ci_also_low_pct":  round(len(ci_low_pts) / n_patients * 100, 1),
            "mean_ciii_activity_pct": round(
                sum(p["ciii_activity_pct"] for p in COHORT) / n_patients, 1),
        },
        "drug_contraindications": {
            "absolute_ci_all_15_genes": [
                {"drug": "VPA / Valproate",     "mechanism": "CoA sequestration → impairs OXPHOS substrate supply; secondary CIII uncoupling; mito toxicity"},
                {"drug": "Metformin",           "mechanism": "CI inhibition → NADH accumulates → impairs QH₂ supply to CIII; worsen lactic acidosis"},
                {"drug": "Propofol",            "mechanism": "PRIS (Propofol Infusion Syndrome) — OXPHOS inhibition including CIII"},
                {"drug": "Linezolid",           "mechanism": "Inhibits mitoribosome → depletes MT-CYB → CIII assembly failure (all CIII genes affected)"},
                {"drug": "Chloramphenicol",     "mechanism": "Mitoribosome inhibition → secondary OXPHOS deficiency including CIII"},
            ],
            "mandatory_workup": [
                "BTBGD/SLC19A3 MANDATORY exclusion — Leigh/leukoencephalopathy mimic; Biotin+Thiamine responsive (life-saving)",
                "Biotin + Thiamine empiric before CIII diagnosis is confirmed (BTBGD exclusion)",
                "GIR 6-8 mg/kg/min — support glucose oxidation; avoid fasting; mandatory in GRACILE/UQCC3 hypoglycaemia",
                "Levetiracetam (LEV) preferred AED — renal clearance, no CYP450, no mito toxicity",
                "Riboflavin (B2) — FAD/FMN source; theoretical benefit (CIII electron chain requires FMN-linked CI activity); Level C",
                "CoQ10 supplementation — theoretical; maintain QH₂ supply to CIII; Level C",
            ],
            "gracile_specific": [
                "GIR 10-12 mg/kg/min (hypoglycaemia + cholestasis management)",
                "Ursodeoxycholic acid (UDCA) — cholestasis management in BCS1L-GRACILE",
                "Iron chelation (deferoxamine) — iron overload in GRACILE (ferritin >1000 μg/L)",
                "TPN / enteral support — growth restriction, severe anabolism failure",
            ],
            "bjornstad_specific": [
                "Hearing aids / cochlear implant assessment — SNHL management",
                "Trichology monitoring — pili torti surveillance",
                "Standard CIII mito contraindications apply even without metabolic crisis",
            ],
        },
        "wes_utility": {
            "nuclear_genes_detectable": "All 15 nuclear CIII genes (UQCRC1/UQCRC2/CYC1/UQCRFS1/UQCRB/UQCRQ/UQCR10/UQCR11/UQCRH + BCS1L/TTC19/LYRM7/UQCC1/UQCC2/UQCC3) — WES detects all",
            "mtDNA_missed":             "MT-CYB (only mtDNA CIII subunit) — WES MISSES; requires dedicated mtDNA panel or long-read sequencing",
            "panel_note":               "Targeted mitochondrial disease gene panels preferred for clinical diagnosis; WES comprehensive but may miss mtDNA + large deletions",
            "bnpage_value":             "Blue-native PAGE (BN-PAGE) distinguishes pre-CIII (BCS1L/LYRM7/TTC19/UQCRFS1) vs absent CIII (UQCC1/UQCC2) vs monomeric CIII (UQCR10/UQCC3)",
        },
    }


def get_breakdown():
    rows = []
    for g in CIII_GENES:
        pts = [p for p in COHORT if p["gene"] == g["gene"]]
        gracile_pct  = round(sum(1 for p in pts if p["gracile_syndrome"]) / len(pts) * 100, 1)
        bjorn_pct    = round(sum(1 for p in pts if p["bjornstad_syndrome"]) / len(pts) * 100, 1)
        leigh_pct    = round(sum(1 for p in pts if p["leigh_mri"]) / len(pts) * 100, 1)
        leuko_pct    = round(sum(1 for p in pts if p["leukoencephalopathy"]) / len(pts) * 100, 1)
        cardiac_pct  = round(sum(1 for p in pts if p["cardiomyopathy"]) / len(pts) * 100, 1)
        hepato_pct   = round(sum(1 for p in pts if p["hepatopathy"]) / len(pts) * 100, 1)
        lactic_pct   = round(sum(1 for p in pts if p["lactic_acidosis"]) / len(pts) * 100, 1)
        hypo_pct     = round(sum(1 for p in pts if p["hypoglycaemia"]) / len(pts) * 100, 1)
        pre_ciii_pct = round(sum(1 for p in pts if p["pre_ciii_bnpage"]) / len(pts) * 100, 1)
        ci_low_pct   = round(sum(1 for p in pts if p["ci_also_low"]) / len(pts) * 100, 1)
        mean_ciii    = round(sum(p["ciii_activity_pct"] for p in pts) / len(pts), 1)
        median_onset = sorted(p["age_onset_months"] for p in pts)[len(pts)//2]

        rows.append({
            "gene":              g["gene"],
            "gene_class":        g["gene_class"],
            "subunit_series":    g["subunit_series"],
            "ciii_module":       g["ciii_module"],
            "omim_gene":         g["omim_gene"],
            "disease_omim":      g["disease_omim"],
            "chromosome":        g["chromosome"],
            "seed":              g["seed"],
            "n_patients":        len(pts),
            "phenotype":         g["phenotype"],
            "inheritance":       g["inheritance"],
            "ci_also_low":       g["ci_also_low"],
            "median_onset_months":    median_onset,
            "ciii_activity_mean_pct": mean_ciii,
            "gracile_pct":       gracile_pct,
            "bjornstad_pct":     bjorn_pct,
            "leigh_mri_pct":     leigh_pct,
            "leuko_pct":         leuko_pct,
            "cardiac_pct":       cardiac_pct,
            "hepatopathy_pct":   hepato_pct,
            "lactic_acidosis_pct": lactic_pct,
            "hypoglycaemia_pct": hypo_pct,
            "pre_ciii_bnpage_pct": pre_ciii_pct,
            "ci_low_pct":        ci_low_pct,
            "disease_summary":   g["disease"][:95],
            "hallmark":          g["hallmark"][:125],
            "founder_variant":   g["founder_variant"],
        })
    return {"genes": rows, "total": len(rows), "total_patients": len(COHORT)}


def get_definitions():
    return {
        "atlas":             "CIII-Subunit-Atlas — Complete 15-gene nuclear-encoded Complex III reference (9 structural subunits + 6 assembly factors)",
        "complex_iii":       "Ubiquinol-cytochrome c reductase (bc1 complex) — obligate CIII₂ homodimer; 11 subunits per monomer (1 mtDNA: MT-CYB + 10 nuclear); Q-cycle electron transfer from QH₂ to Cyt c; pumps 4 H⁺/2e⁻ across IMM",
        "MT_CYB_note":       "MT-CYB (cytochrome b, 380 aa, mtDNA) = ONLY mtDNA-encoded CIII subunit; WES MISSES — covered in MT-Genome-Atlas and MT-CYB expert dashboard; maternal inheritance",
        "CII_always_normal": "CII ALWAYS NORMAL in isolated CIII deficiency — no mtDNA-encoded CII subunits exist; CII is internal biochemical reference for all OXPHOS panels",
        "Q_cycle":           "Q cycle (Mitchell 1975): QH₂ oxidised at Qo site (near MT-CYB) → 2e⁻ bifurcated: 1e⁻ via RISP→CYC1→Cyt c (high potential); 1e⁻ via MT-CYB bL→bH heme→Qi site→Q→QH₂ recycled; net: 4H⁺ pumped per 2e⁻ transferred",
        "RISP_UQCRFS1":      "UQCRFS1 (Rieske ISP/RISP, 19q12) — mobile [2Fe-2S] head pivots between Qo and CYC1 face; only mobile CIII subunit; inserted LAST by BCS1L; LYRM7→BCS1L→TTC19 ordered assembly cascade",
        "pre_CIII_bnpage":   "Pre-CIII₂ sub-complex on blue-native PAGE = pathognomonic for RISP-insertion failure: seen in BCS1L, LYRM7, TTC19, UQCRFS1 mutations; MT-CYB-containing scaffold present but RISP absent",
        "supercomplex":      "Respirasomes: SC I+III₂ and SC I+III₂+IV — CIII₂ core of both; UQCRC1/UQCRC2 (Core 1/2) form the SC contact interface → mutations cause combined CI+CIII deficiency via SC destabilisation",
        "GRACILE":           "GRACILE syndrome (BCS1L p.Ser78Gly, Finnish founder): Growth restriction + Aminoaciduria + Cholestasis + Iron overload + Lactic acidosis + Early death; <5% survival >1 year; 1:36 carrier freq in Finland; CIII 3-8% residual",
        "Bjornstad":         "Bjornstad syndrome (BCS1L milder alleles): Pili torti (twisted hair shafts) + Sensorineural hearing loss WITHOUT metabolic crisis; normal lifespan; same gene as GRACILE — genotype-phenotype distinction",
        "UQCRC1_Core1":      "UQCRC1 (Core 1, 480 aa, 3p21.31) — matrix arm scaffold; MPP-α analogue; SC I+III₂ contact; mutations → combined CI+CIII (SC destabilisation) + leukoencephalopathy 65% + hepatopathy 45%",
        "UQCRC2_Core2":      "UQCRC2 (Core 2, 453 aa, 16p12.2) — matrix arm; MPP-β analogue; processes MT-CYB presequence; SC I+III₂ contact; mutations → combined CI+CIII + HCM 55% + pontine hypoplasia 40% — most cardiac structural subunit",
        "CYC1_c1":           "CYC1 (Cyt c1, 325 aa, 8q24.13) — haem c binding; fixed IMS face; accepts e⁻ from RISP; transfers to free Cyt c; mutations → isolated CIII; no pre-CIII on BN-PAGE (unlike RISP pathway AFs)",
        "UQCRB_Sub6":        "UQCRB (Subunit 6, QP-C, 109 aa, 8q22.1) — peripheral matrix-face; Qi site proximity; hypoglycaemia 65% (gluconeogenesis disruption); Haut 2003 (HumGenet) first nuclear CIII structural subunit mutation reported",
        "UQCRQ_Sub7":        "UQCRQ (Subunit 7, QCR7, 82 aa, 5q31.1) — peripheral IMS face; Qo site proximity; dystonia 75% + optic atrophy 42% = distinguishing CIII phenotype; encephalomyopathy dominant",
        "UQCR10_Sub10":      "UQCR10 (Subunit 10, 63 aa, 22q13.1) — membrane-spanning dimer interface; CIII₂ homodimerisation; rare; BN-PAGE: CIII monomer (dimer fails); severe infantile",
        "UQCR11_Sub11":      "UQCR11 (Subunit 11, 56 aa, 19q13.41) — smallest structural CIII subunit; dimer interface with UQCR10; rare; severe infantile",
        "UQCRH_Hinge":       "UQCRH (Hinge/Subunit 9, QCR9, 91 aa, 1p33) — IMS face; facilitates rapid Cyt c turnover at CIII exit; mutations → isolated CIII; exercise intolerance early sign",
        "BCS1L_AF":          "BCS1L (419 aa, AAA+ ATPase, 2q35) — inserts UQCRFS1 [2Fe-2S] domain into pre-CIII (final RISP insertion step); mutations → GRACILE (p.Ser78Gly, Finnish) OR Bjornstad syndrome (milder alleles); widest CIII phenotypic range",
        "TTC19_AF":          "TTC19 (380 aa, TPR domain, 17p12) — UQCRFS1 maturation after BCS1L; SC III₂+IV assembly; mutations → progressive neurodegeneration, cerebellar atrophy, psychiatric features (psychosis/depression) — unique among CIII AFs; later onset (childhood-adult)",
        "LYRM7_AF":          "LYRM7/MZM1L (121 aa, LYRM motif, 5q31) — earliest RISP chaperone; stabilises UQCRFS1 before BCS1L step; Dallabona 2015 AJHG; acute severe infantile (unlike TTC19); LYRM motif shared with SDHAF1 (CII AF)",
        "UQCC1_AF":          "UQCC1/BRAWNIN (131 aa, 20p13) — scaffolds newly synthesised MT-CYB with UQCC2; earliest CIII assembly step; mutations → no assembled CIII₂ on BN-PAGE (unlike pre-CIII seen in RISP pathway); severe neonatal/infantile",
        "UQCC2_AF":          "UQCC2 (259 aa, 6p21.1) — MT-CYB stabilisation module with UQCC1; mutations → isolated CIII + HCM 80% (HIGHEST cardiac rate of all CIII genes); Gilbertson 2019 landmark; UQCC2 = most cardiomyopathy-associated CIII gene",
        "UQCC3_AF":          "UQCC3 (112 aa, 11q12.1) — late-stage CIII₂ dimer + SC III₂+IV; hypoglycaemia 60% + neonatal lactic acidosis; BN-PAGE: CIII monomer (no dimer); Wanschers 2014 AJHG; some patients improve with metabolic support",
        "vpa_ci":            "VPA absolute CI in ALL CIII disorders — CoA sequestration + OXPHOS uncoupling + mitochondrial toxicity; substitutes: levetiracetam (LEV) preferred AED",
        "metformin_ci":      "Metformin absolute CI — CI inhibition → NADH accumulation → impairs QH₂ generation → worsen CIII substrate starvation and lactic acidosis",
        "BTBGD_exclusion":   "SLC19A3 (BTBGD) MANDATORY exclusion before diagnosing CIII-Leigh or leukoencephalopathy — Leigh/white matter mimic treatable with Biotin+Thiamine (life-saving dramatic response)",
        "WES_nuclear_CIII":  "All 15 nuclear CIII genes in this atlas are WES-detectable; MT-CYB (mtDNA) is NOT WES-detectable — requires dedicated mtDNA panel; BN-PAGE essential functional complement to WES for CIII",
    }
