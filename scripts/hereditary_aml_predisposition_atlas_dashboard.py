#!/usr/bin/env python3
"""Hereditary-AML-Predisposition-Atlas -- Complete 8-Gene Reference
Acute Myeloid Leukaemia (AML) germline predisposition.

CEBPA    (C/EBPα; 358aa; 19q13.11; AD biallelic LOF;
          Familial AML / CEBPA-FPD (OMIM 601626);
          AML 40-60% lifetime; biallelic N+C-terminal mutations PATHOGNOMONIC;
          FAB M1/M2/M4; aza-venetoclax; NOT allograft after CR1 unless persistent biallelic;
          seed SEED_BASE+0).
DDX41    (DDX41; 622aa; 5q35.3; AD LOF;
          Familial AML/MDS (OMIM 608170);
          late onset 55-65yr HALLMARK; R525H somatic second-hit 50%;
          HMA+venetoclax preferred; donor DDX41 testing MANDATORY;
          seed SEED_BASE+1).
RUNX1    (RUNX1; 453aa; 21q22.12; AD LOF;
          FPD-AML / Familial Platelet Disorder with AML (OMIM 601399);
          thrombocytopenia PATHOGNOMONIC; AML 30-50% lifetime;
          sibling donors CI; azacitidine preferred;
          seed SEED_BASE+2).
GATA2    (GATA2; 480aa; 3q21.3; AD LOF;
          GATA2 Deficiency / MonoMAC / Emberger (OMIM 137295);
          AML 30-70x HIGHEST; NK lymphopenia PATHOGNOMONIC;
          HSCT curative do not delay; EBV CMV q3M mandatory;
          seed SEED_BASE+3).
TP53     (p53; 393aa; 17p13.1; AD LOF;
          Li-Fraumeni Syndrome (OMIM 151623);
          AML 10-20x; complex karyotype PATHOGNOMONIC;
          AVOID RADIATION ABSOLUTELY; WB-MRI Toronto annually;
          venetoclax-decitabine preferred;
          seed SEED_BASE+4).
ANKRD26  (ANKRD26; 1710aa; 10p12.1; AD LOF;
          THC2 / Thrombocytopenia 2 (OMIM 188000);
          5UTR promoter mutations; standard sequencing MISSES 5UTR;
          panel MANDATORY; AML/MDS 5-10% lifetime;
          seed SEED_BASE+5).
ETV6     (ETV6; 452aa; 12p13.2; AD LOF;
          THC5 / Thrombocytopenia 5 (OMIM 600618);
          thrombocytopenia; AML/MDS 10-15%;
          avoid NSAIDs; avoid high-dose MTX;
          ETV6-RUNX1 fusion absent in germline;
          seed SEED_BASE+6).
SAMD9L   (SAMD9L; 1589aa; 7q21.2; AD GOF;
          ATAD / Ataxia-Pancytopenia (OMIM 617053);
          pediatric AML; monosomy 7 80% PATHOGNOMONIC;
          cerebellar ataxia PATHOGNOMONIC; reversion mosaicism;
          HSCT corrects AML not ataxia; adrenal sick-day rule;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3662-3669)
"""
import random

SEED_BASE = 3662

ATLAS_GENES = [
    {
        "gene": "CEBPA",
        "protein": (
            "C/EBPa -- 19q13.11 AD biallelic LOF -- 358aa -- "
            "C/EBPa-42kDa-bZIP-Transcription-Factor-"
            "TAD1-TAD2-bZIP-"
            "CEBPA-FPD-AML-40-60pct-Lifetime-Biallelic-N+C-terminal-PATHOGNOMONIC-"
            "CEBPA-OMIM-601626"
        ),
        "locus": "19q13.11",
        "protein_size": (
            "358 aa / ~42 kDa / 19q13.11 CEBPA AML molecular context: "
            "STRUCTURE: "
            "  358 aa / ~42 kDa; basic leucine zipper (bZIP) transcription factor; myeloid differentiation master regulator; "
            "  Transactivation domain 1 (TAD1, aa 1-117): p42 isoform only; transcriptional activation; "
            "  Transactivation domain 2 (TAD2, aa 127-200): both p42 and p30 isoforms; CREB-binding protein interaction; "
            "  Basic DNA-binding domain (aa 291-318): sequence-specific C/EBP site recognition; "
            "  Leucine zipper (aa 318-358): homodimerisation and heterodimerisation; "
            "  N-terminal frameshift (germline LOF) -> p30 truncation -> dominant-negative over WT p42; "
            "  C-terminal in-frame mutation (somatic second hit) -> bZIP disruption -> biallelic LOF; "
            "CEBPA-FPD-AML CONTEXT: "
            "  Germline N-terminal frameshift (exon 1): familial CEBPA FPD-AML; ~0.5% of AML overall; "
            "  Biallelic N-terminal (germline) + C-terminal (somatic): PATHOGNOMONIC for CEBPA-FPD-AML; "
            "  AML risk 40-60% lifetime -- highest penetrance of any AML predisposition gene; "
            "  FAB M1/M2/M4 predominant; CEBPA-biallelic excellent CR rate ~80%; "
            "  NOT allograft in CR1 unless persistent biallelic second-hit or MRD-positive; "
            "  Azacitidine + venetoclax preferred for R/R -- HOX pathway sensitivity; "
            "CLINICAL RULES FOR CEBPA-AML: "
            "  Biallelic N+C mutation at AML diagnosis: PATHOGNOMONIC for germline predisposition; "
            "  Germline N-terminal CEBPA: test ALL first-degree relatives -- 50% carrier risk; "
            "  Sibling donors must be tested: CEBPA germline -> donor AML risk elevated; "
            "  NOT allograft after CR1 unless: persistent biallelic clones, MRD+, or adverse karyotype; "
            "  Annual CBC from age 18 in carriers; morphology at AML diagnosis mandatory; "
            "KEY VARIANTS: "
            "  N-terminal frameshifts: 4-bp (c.68_71dup), c.310_311ins, c.1072_1076del (most common families); "
            "  C-terminal in-frame somatic second hits: p.Arg297*, bZIP missense; "
            "SURVEILLANCE: "
            "  Annual CBC + morphology from age 18; "
            "  Bone marrow biopsy if cytopenias; MRD monitoring post-CR; "
            "  Family cascade germline CEBPA testing; "
            "TARGETED Rx: "
            "  Aza+venetoclax preferred R/R; standard 7+3 induction adequate CR1; "
            "  HOX pathway sensitivity: menin inhibitors (revumenib) emerging; "
            "  Allo-HSCT defer to CR2 or MRD-positive CR1; "
            "  Sibling donor: CEBPA germline testing MANDATORY before donation"
        ),
        "syndrome": "CEBPA-FPD / Familial AML",
        "inheritance": "AD LOF (biallelic N+C hits)",
        "aml_risk": "AML 40-60% lifetime; HIGHEST PENETRANCE AML predisposition gene",
        "pathognomonic": "Biallelic-N+C-terminal-CEBPA-PATHOGNOMONIC; FAB-M1-M2-M4-CEBPA-dominant",
        "key_rule": "NOT-allograft-CR1-unless-persistent-biallelic; aza-venetoclax-preferred-R/R; sibling-donor-CEBPA-test-MANDATORY",
        "key_avoid": "Avoid allograft in CR1 unless MRD+ or persistent biallelic; never use untested sibling donor",
        "surveillance": "Annual CBC from age 18; BM biopsy if cytopenias; MRD monitoring post-CR; family cascade testing",
        "targeted_rx": "Aza+venetoclax R/R; 7+3 induction CR1; menin inhibitors emerging; HSCT defer to CR2",
    },
    {
        "gene": "DDX41",
        "protein": (
            "DDX41 -- 5q35.3 AD LOF -- 622aa -- "
            "DDX41-70kDa-DEAD-Box-RNA-Helicase-"
            "DEAD-box-RecA-"
            "Familial-AML-MDS-Late-Onset-55-65yr-HALLMARK-R525H-50pct-"
            "DDX41-OMIM-608170"
        ),
        "locus": "5q35.3",
        "protein_size": (
            "622 aa / ~70 kDa / 5q35.3 DDX41 AML molecular context: "
            "STRUCTURE: "
            "  622 aa / ~70 kDa; DEAD-box RNA helicase; innate immunity / R-loop resolution; "
            "  N-terminal intrinsically disordered region (aa 1-280): low complexity; protein-protein interactions; "
            "  DEAD-box helicase core - RecA1 domain (aa 281-435): ATP binding; Walker A/B motifs; "
            "  RecA2 domain (aa 436-580): RNA binding; RNA unwinding; "
            "  C-terminal SPRY domain (aa 581-622): protein interactions; "
            "  DDX41 LOF -> impaired R-loop resolution -> genomic instability -> AML; "
            "  DDX41 LOF -> impaired innate DNA sensing -> tumour immune evasion; "
            "DDX41 FAMILIAL AML CONTEXT: "
            "  Germline DDX41 LOF: ~1.5-3% of AML overall; late onset 55-65yr HALLMARK; "
            "  Somatic R525H second hit in ~50% of DDX41-germline AML (biallelic in RecA2 domain); "
            "  R525H: recurrent somatic hotspot; RecA2 RNA-binding domain disruption; "
            "  AML FAB M0/M1/M2; MDS preceding AML in 30-40%; "
            "  HMA (azacitidine/decitabine) + venetoclax preferred -- HOX/R-loop sensitivity; "
            "  DONOR DDX41 TESTING MANDATORY: germline heterozygous donors transmit predisposition; "
            "CLINICAL RULES FOR DDX41-AML: "
            "  Late onset (55-65yr) + AML + MDS history -> DDX41 germline testing MANDATORY; "
            "  Donor testing: ALL related donors must be germline-tested before HSCT; "
            "  HMA+venetoclax preferred over intensive chemotherapy in older DDX41-AML; "
            "  R525H somatic in AML without prior germline -> test germline DDX41 (germline first); "
            "  MRD-guided therapy: DDX41-AML high CR rate (~75%) with HMA+ven; "
            "KEY VARIANTS: "
            "  Germline LOF: p.Tyr259* (most common in East Asian); p.Met1? (initiation codon); "
            "  c.1580delG (p.Gly527Alafs frameshift); c.415G>T (p.Glu139*); "
            "  Somatic R525H (c.1574G>A): RecA2 hotspot; ~50% of DDX41-germline AML; "
            "SURVEILLANCE: "
            "  Annual CBC from DDX41 germline diagnosis; BM biopsy if cytopenias; "
            "  Screen for MDS 3-yearly from age 50; "
            "  Family cascade germline testing; donor testing mandatory; "
            "TARGETED Rx: "
            "  HMA+venetoclax (azacitidine or decitabine + venetoclax) preferred 1st line older patients; "
            "  Allo-HSCT with DDX41-negative donor; "
            "  Intensive 7+3 feasible younger patients; "
            "  Menin inhibitors (revumenib, ziftomenib) emerging for R/R DDX41-AML"
        ),
        "syndrome": "Familial AML/MDS -- DDX41",
        "inheritance": "AD LOF heterozygous",
        "aml_risk": "AML ~3% lifetime; late onset 55-65yr HALLMARK; R525H somatic second-hit 50%",
        "pathognomonic": "Late-onset-55-65yr-AML-HALLMARK; R525H-somatic-second-hit-50pct; DDX41-germline-first",
        "key_rule": "DONOR-DDX41-TESTING-MANDATORY; HMA-venetoclax-preferred; late-onset-HALLMARK-55-65yr",
        "key_avoid": "Never use untested related donor; do not use DDX41-carrier sibling as HSCT donor",
        "surveillance": "Annual CBC; BM biopsy if cytopenias; 3-yearly MDS screen from age 50; cascade testing",
        "targeted_rx": "HMA+venetoclax preferred; allo-HSCT DDX41-negative donor; menin inhibitors emerging R/R",
    },
    {
        "gene": "RUNX1",
        "protein": (
            "RUNX1 -- 21q22.12 AD LOF -- 453aa -- "
            "RUNX1-51kDa-Runt-Domain-Transcription-Factor-"
            "Runt-domain-TAD-VWRPY-"
            "FPD-AML-Thrombocytopenia-PATHOGNOMONIC-AML-30-50pct-Sibling-Donors-CI-"
            "RUNX1-OMIM-601399"
        ),
        "locus": "21q22.12",
        "protein_size": (
            "453 aa / ~51 kDa / 21q22.12 RUNX1 AML molecular context: "
            "STRUCTURE: "
            "  453 aa / ~51 kDa; Runt-domain transcription factor; alpha subunit of CBF complex; "
            "  Runt homology domain (RHD, aa 50-177): DNA binding (TGTGGT core) and CBFb heterodimerisation; "
            "  Transactivation domain (TAD, aa 291-371): co-activator interactions; "
            "  VWRPY inhibitory domain (aa 449-453): Groucho/TLE co-repressor binding; "
            "  Nuclear localisation signal (aa 167-183); "
            "  RUNX1-CBFb: master myeloid/lymphoid differentiation regulator; "
            "  RUNX1 LOF -> blocked myeloid differentiation -> AML predisposition; "
            "FPD-AML CONTEXT: "
            "  Germline RUNX1 LOF (FPD/AML): thrombocytopenia + AML 30-50% lifetime; "
            "  Thrombocytopenia PATHOGNOMONIC: platelet count 40-80x10^9/L with functional defects; "
            "  AML onset typically 20-40yr; MDS precedes in ~30%; "
            "  Sibling donors: PATHOGNOMONIC CONTRAINDICATION -- RUNX1 germline -> sibling donor risk; "
            "  Azacitidine preferred over intensive chemotherapy -- RUNX1-AML sensitive; "
            "  Somatic RUNX1 mutation (acquired): additional hit accelerating AML; "
            "CLINICAL RULES FOR RUNX1-AML: "
            "  Thrombocytopenia + family AML history -> RUNX1 germline testing MANDATORY; "
            "  SIBLING DONORS CI -- all first-degree relatives must be tested before donation; "
            "  Azacitidine preferred as bridge and maintenance; "
            "  Platelet transfusion threshold: <10 or <20 if bleeding; "
            "  Annual BM morphology from age 18 in carriers with thrombocytopenia; "
            "KEY VARIANTS: "
            "  Runt domain: missense p.Arg139Gln, p.Asp171Asn, p.Arg204Gln (DNA-contact disruption); "
            "  LOF truncating throughout; "
            "  p.Pro290Ser: TAD disruption; "
            "SURVEILLANCE: "
            "  Annual CBC + morphology from diagnosis; BM biopsy if blasts increase; "
            "  Flow cytometry platelet function testing; family cascade testing; "
            "  Donor testing MANDATORY before any HSCT; "
            "TARGETED Rx: "
            "  Azacitidine preferred for MDS/AML RUNX1-mutant; "
            "  Venetoclax combination emerging; "
            "  Allo-HSCT unrelated donor preferred (siblings must be tested); "
            "  Menin inhibitors (revumenib): active in RUNX1-mutant AML"
        ),
        "syndrome": "FPD-AML / Familial Platelet Disorder with AML",
        "inheritance": "AD LOF heterozygous",
        "aml_risk": "AML 30-50% lifetime; MDS precedes 30%; thrombocytopenia PATHOGNOMONIC onset childhood",
        "pathognomonic": "Thrombocytopenia-40-80x10-9-PATHOGNOMONIC; FPD-family-history; sibling-donors-CI-ABSOLUTELY",
        "key_rule": "SIBLING-DONORS-CI; azacitidine-preferred; thrombocytopenia-PATHOGNOMONIC; cascade-test-MANDATORY",
        "key_avoid": "NEVER use untested sibling as donor; avoid platelet-toxic NSAIDs in carriers",
        "surveillance": "Annual CBC from age 18; BM biopsy if blasts increase; flow platelet function; cascade testing",
        "targeted_rx": "Azacitidine preferred; venetoclax combination; allo-HSCT unrelated donor; menin inhibitors R/R",
    },
    {
        "gene": "GATA2",
        "protein": (
            "GATA2 -- 3q21.3 AD LOF -- 480aa -- "
            "GATA2-50kDa-Zinc-Finger-Transcription-Factor-"
            "ZF1-ZF2-NLS-"
            "MonoMAC-Emberger-AML-30-70x-HIGHEST-NK-Lymphopenia-PATHOGNOMONIC-HSCT-Curative-"
            "GATA2-OMIM-137295"
        ),
        "locus": "3q21.3",
        "protein_size": (
            "480 aa / ~50 kDa / 3q21.3 GATA2 AML molecular context: "
            "STRUCTURE: "
            "  480 aa / ~50 kDa; GATA-family zinc-finger transcription factor; haematopoietic master regulator; "
            "  N-terminal zinc finger (NZF/ZF1, aa 200-230): protein-protein interactions; "
            "  C-terminal zinc finger (CZF/ZF2, aa 317-340): DNA binding (WGATAR motif); "
            "  Nuclear localisation signal (NLS, aa 330-340); "
            "  C-terminal domain (aa 340-480): transactivation; "
            "  GATA2 LOF -> HSC exhaustion -> MDS/AML + immune collapse; "
            "GATA2-AML CONTEXT: "
            "  GATA2 deficiency: haploinsufficiency -> MDS (30-70x risk) + AML + immune deficiency; "
            "  NK lymphopenia PATHOGNOMONIC: NK cells <0.05x10^9/L diagnostic; "
            "  MonoMAC syndrome: Monocytopenia + MAC infections (NTM, EBV); "
            "  Emberger syndrome: lymphoedema + MDS/AML; "
            "  HSCT CURATIVE -- do not delay; MDS progression to AML in 30-40% without HSCT; "
            "  EBV + CMV monitoring quarterly MANDATORY due to NK/T immune deficiency; "
            "CLINICAL RULES FOR GATA2-AML: "
            "  NK lymphopenia (<0.05x10^9/L) PATHOGNOMONIC -- test GATA2 in any unexplained monocytopenia + NK deficiency; "
            "  HSCT curative -- do not wait for AML transformation; refer at MDS stage; "
            "  EBV + CMV q3M MANDATORY; NTM prophylaxis azithromycin; "
            "  Donor: related donors must be GATA2-tested; "
            "  HPV vaccination before immune collapse (before HSCT); "
            "KEY VARIANTS: "
            "  C-terminal ZF2: p.Thr354Met (most common Europe/US); p.Arg361His; p.Leu359Val; "
            "  Intronic: c.1017+1G>T (splice donor ZF2 exon); "
            "  Large deletions: 3q21 deletion; "
            "SURVEILLANCE: "
            "  Annual CBC + NK cell count + monocyte count; "
            "  EBV + CMV PCR q3M; NTM chest CT annually; "
            "  Lymphoedema assessment; hearing assessment; BM annually from GATA2 diagnosis; "
            "TARGETED Rx: "
            "  Allo-HSCT curative (do not delay past MDS-EB2 or AML); "
            "  Azacitidine bridge to HSCT; "
            "  NTM: azithromycin prophylaxis; "
            "  HPV + all inactivated vaccines before HSCT"
        ),
        "syndrome": "GATA2 Deficiency / MonoMAC / Emberger",
        "inheritance": "AD LOF heterozygous",
        "aml_risk": "AML/MDS 30-70x HIGHEST; NK lymphopenia PATHOGNOMONIC; HSCT curative do not delay",
        "pathognomonic": "NK-lymphopenia-<0.05x10-9-PATHOGNOMONIC; MonoMAC-NTM-EBV-monocytopenia; Emberger-lymphoedema",
        "key_rule": "HSCT-CURATIVE-DO-NOT-DELAY; EBV-CMV-Q3M-MANDATORY; NK-lymphopenia-PATHOGNOMONIC",
        "key_avoid": "Do not delay HSCT waiting for AML; never use GATA2-carrier related donor",
        "surveillance": "Annual CBC + NK + monocyte; EBV CMV q3M; NTM CT annually; BM annually; HPV pre-HSCT",
        "targeted_rx": "Allo-HSCT curative; azacitidine bridge; NTM azithromycin prophylaxis; HPV vaccination",
    },
    {
        "gene": "TP53",
        "protein": (
            "p53 -- 17p13.1 AD LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-"
            "TAD-DBD-OD-"
            "LFS-AML-10-20x-Complex-Karyotype-PATHOGNOMONIC-AVOID-RADIATION-ABSOLUTELY-Venetoclax-Decitabine-"
            "TP53-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / ~43 kDa / 17p13.1 TP53 AML molecular context: "
            "STRUCTURE: "
            "  393 aa / ~43 kDa; transcription factor / tumour suppressor; "
            "  N-terminal transactivation domain 1 (TAD1, aa 1-40); TAD2 (aa 40-67): MDM2 binding; "
            "  Proline-rich domain (aa 68-98): apoptosis signalling; "
            "  DNA-binding domain (DBD, aa 102-292): sequence-specific binding; "
            "  Nuclear localisation signal (aa 316-325); "
            "  Tetramerisation domain (OD, aa 323-356): functional tetramer; "
            "  C-terminal regulatory domain (aa 364-393): ubiquitination; "
            "  TP53 LOF -> failed G1 checkpoint -> genomic instability; complex karyotype; "
            "TP53 LFS-AML CONTEXT: "
            "  LFS germline TP53: AML 10-20x; complex karyotype AML PATHOGNOMONIC; "
            "  Complex karyotype (>=3 unrelated): WORST PROGNOSIS AML -- median OS <6m without targeted Rx; "
            "  Venetoclax+decitabine preferred: TP53-mutant AML sensitive via p53-independent BCL2 pathway; "
            "  APR-246 (eprenetapopt): mutant p53 refolding -- Phase III trials; "
            "  AVOID RADIATION ABSOLUTELY in LFS; "
            "  WB-MRI Toronto protocol annually (sarcoma, breast, brain, adrenocortical); "
            "CLINICAL RULES FOR TP53-AML: "
            "  AVOID RADIATION ABSOLUTELY: even low-dose RT -> secondary malignancies in LFS; "
            "  Complex karyotype at AML diagnosis -> germline TP53 testing MANDATORY; "
            "  Venetoclax+decitabine preferred over 7+3 in TP53-mutant AML; "
            "  WB-MRI Toronto annually from diagnosis; "
            "  Alkylating conditioning: FLU-BU preferred over TBI in LFS-AML HSCT; "
            "KEY VARIANTS: "
            "  DBD hotspots: p.Arg175His, p.Gly245Ser, p.Arg248Trp/Gln, p.Arg249Ser, p.Arg273His/Cys; "
            "  Dominant-negative: mutant inhibits WT tetramer; "
            "SURVEILLANCE: "
            "  WB-MRI annually (Toronto protocol); CBC annually from age 18; "
            "  Brain MRI annually; colonoscopy 25yr or 5yr before earliest family cancer; "
            "TARGETED Rx: "
            "  Venetoclax+decitabine or venetoclax+azacitidine backbone TP53-mutant AML; "
            "  Eprenetapopt (APR-246) mutant p53 refolding; "
            "  FLU-BU conditioning (avoid TBI/RT in LFS); "
            "  AVOID alkylating-heavy regimens if RT exposure history"
        ),
        "syndrome": "Li-Fraumeni Syndrome",
        "inheritance": "AD LOF heterozygous",
        "aml_risk": "LFS AML 10-20x; complex karyotype AML PATHOGNOMONIC; worst prognosis without targeted Rx",
        "pathognomonic": "Complex-karyotype-AML-PATHOGNOMONIC-LFS; AVOID-RADIATION-ABSOLUTELY; WB-MRI-Toronto-annually",
        "key_rule": "AVOID-RADIATION-ABSOLUTELY; venetoclax-decitabine-preferred; complex-karyotype-PATHOGNOMONIC; WB-MRI-Toronto",
        "key_avoid": "AVOID RT absolutely; do not use TBI conditioning in LFS-AML HSCT; avoid alkylating-heavy regimens",
        "surveillance": "WB-MRI Toronto annually; CBC annually; brain MRI annually; colonoscopy 25yr; germline testing complex karyotype AML",
        "targeted_rx": "Venetoclax+decitabine backbone; eprenetapopt APR-246 trials; FLU-BU conditioning (no TBI); menin inhibitors",
    },
    {
        "gene": "ANKRD26",
        "protein": (
            "ANKRD26 -- 10p12.1 AD LOF -- 1710aa -- "
            "ANKRD26-190kDa-Ankyrin-Repeat-Domain-"
            "ANK-repeats-coiled-coil-"
            "THC2-5UTR-Promoter-Standard-Sequencing-MISSES-5UTR-Panel-MANDATORY-AML-MDS-5-10pct-"
            "ANKRD26-OMIM-188000"
        ),
        "locus": "10p12.1",
        "protein_size": (
            "1710 aa / ~190 kDa / 10p12.1 ANKRD26 AML molecular context: "
            "STRUCTURE: "
            "  1710 aa / ~190 kDa; ankyrin repeat domain protein; megakaryocyte development regulator; "
            "  N-terminal ankyrin (ANK) repeats (aa 1-800): protein-protein interactions; TPM3/TPM4 binding; "
            "  Central scaffold (aa 800-1400): structural; signalling scaffold; "
            "  C-terminal coiled-coil domain (aa 1400-1710): dimerisation; "
            "  ANKRD26 regulates MAPK signalling in megakaryopoiesis; "
            "  ANKRD26 5UTR LOF -> derepression of expression -> dysregulated thrombopoiesis -> thrombocytopenia + AML; "
            "ANKRD26-THC2-AML CONTEXT: "
            "  THC2 (Thrombocytopenia 2): ANKRD26 5UTR promoter mutations (not coding mutations); "
            "  Standard sequencing (Sanger / coding-only panel) MISSES 5UTR -- diagnostic pitfall; "
            "  5UTR-SPECIFIC PANEL MANDATORY: ANKRD26 pathogenic variants are exclusively in 5UTR TATA-box region; "
            "  AML/MDS lifetime risk 5-10%; moderate penetrance; "
            "  Thrombocytopenia 40-100x10^9/L with normal platelet function; "
            "  AML FAB M1/M2/M4; azacitidine+venetoclax sensitive; "
            "CLINICAL RULES FOR ANKRD26-AML: "
            "  STANDARD SEQUENCING MISSES -- 5UTR panel MANDATORY for ANKRD26; "
            "  Thrombocytopenia + family AML history -> ANKRD26 5UTR targeted sequencing first; "
            "  Sibling donor testing: ANKRD26 germline; "
            "  Annual CBC from diagnosis; BM biopsy if blasts; "
            "KEY VARIANTS: "
            "  5UTR variants: c.-128G>A (most common), c.-127A>T, c.-124C>T, c.-126G>T (TATA-box region); "
            "  All in 5UTR -- NOT detected by standard exome/panel coding-region only; "
            "SURVEILLANCE: "
            "  Annual CBC + BM morphology if cytopenias or blast increase; "
            "  5UTR-specific ANKRD26 panel for all first-degree relatives; "
            "TARGETED Rx: "
            "  Azacitidine+venetoclax for AML/MDS; "
            "  Eltrombopag: emerging for thrombocytopenia (caution -- may promote AML); "
            "  Allo-HSCT for AML transformation; "
            "  AVOID eltrombopag if MDS features present"
        ),
        "syndrome": "THC2 / Thrombocytopenia 2 -- ANKRD26",
        "inheritance": "AD LOF (5UTR promoter)",
        "aml_risk": "AML/MDS 5-10% lifetime; thrombocytopenia 40-100x10^9/L; 5UTR mutations exclusive",
        "pathognomonic": "5UTR-promoter-mutations-ONLY-standard-sequencing-MISSES; thrombocytopenia-ANKRD26",
        "key_rule": "5UTR-PANEL-MANDATORY; STANDARD-SEQUENCING-MISSES; aza-venetoclax-AML-preferred",
        "key_avoid": "Never rely on standard coding-region panel/Sanger for ANKRD26; 5UTR targeted test required",
        "surveillance": "Annual CBC; BM biopsy if cytopenias; 5UTR-specific panel all first-degree relatives",
        "targeted_rx": "Aza+venetoclax for AML/MDS; allo-HSCT at transformation; avoid eltrombopag if MDS features",
    },
    {
        "gene": "ETV6",
        "protein": (
            "ETV6 -- 12p13.2 AD LOF -- 452aa -- "
            "ETV6-57kDa-ETS-Transcription-Factor-"
            "PNT-DBD-"
            "THC5-Thrombocytopenia-AML-MDS-10-15pct-Avoid-NSAIDs-Avoid-HighDose-MTX-ETV6-RUNX1-Absent-Germline-"
            "ETV6-OMIM-600618"
        ),
        "locus": "12p13.2",
        "protein_size": (
            "452 aa / ~57 kDa / 12p13.2 ETV6 AML molecular context: "
            "STRUCTURE: "
            "  452 aa / ~57 kDa; ETS-family transcription factor; haematopoietic development regulator; "
            "  PNT (pointed) domain (aa 42-130): homotypic polymerisation; "
            "  Linker region (aa 130-336): regulatory; "
            "  ETS DNA-binding domain (DBD, aa 338-420): GGA(A/T) core binding; "
            "  C-terminal inhibitory domain (aa 421-452): auto-inhibitory ETS; "
            "  ETV6-RUNX1 t(12;21): SOMATIC childhood B-ALL (TEL-AML1) -- ABSENT in germline ETV6 LOF; "
            "  ETV6 germline LOF -> impaired megakaryopoiesis -> thrombocytopenia + AML/MDS; "
            "THC5-ETV6-AML CONTEXT: "
            "  THC5 (Thrombocytopenia 5): ETV6 germline LOF; thrombocytopenia + AML/MDS 10-15%; "
            "  ETV6-RUNX1 fusion: SOMATIC t(12;21) in childhood B-ALL -- ABSENT in germline ETV6 FPD; "
            "  AVOID NSAIDs: platelet function impaired; bleeding risk; "
            "  AVOID HIGH-DOSE METHOTREXATE: ETV6 regulates folate pathway; toxicity enhanced; "
            "  AML FAB M1/M2; azacitidine+venetoclax preferred; "
            "CLINICAL RULES FOR ETV6-AML: "
            "  ETV6-RUNX1 t(12;21) fusion ABSENT in germline ETV6 predisposition; "
            "  AVOID NSAIDs (aspirin, ibuprofen) in ETV6 carriers -- platelet dysfunction; "
            "  AVOID HIGH-DOSE MTX: ETV6 regulates DHFR; enhanced folate antagonism toxicity; "
            "  Annual CBC; BM biopsy if blasts; cascade family testing; "
            "KEY VARIANTS: "
            "  ETS domain: p.Arg369Gln, p.Arg418Gln (DNA-contact disruption -- most common); "
            "  Truncating: p.Gln213* (PNT-linker junction); "
            "  p.Pro214Leu: PNT domain disruption; "
            "SURVEILLANCE: "
            "  Annual CBC + platelet function; BM annually if thrombocytopenia; "
            "  Family cascade ETV6 germline testing; "
            "TARGETED Rx: "
            "  Azacitidine+venetoclax for AML/MDS; "
            "  Allo-HSCT for AML transformation; "
            "  AVOID high-dose MTX regimens (use reduced dose); "
            "  Standard-dose MTX tolerable with monitoring; eltrombopag emerging for thrombocytopenia"
        ),
        "syndrome": "THC5 / Thrombocytopenia 5 -- ETV6",
        "inheritance": "AD LOF heterozygous",
        "aml_risk": "AML/MDS 10-15% lifetime; thrombocytopenia; ETV6-RUNX1 fusion ABSENT in germline",
        "pathognomonic": "ETV6-RUNX1-fusion-ABSENT-germline (somatic-only); THC5-thrombocytopenia; NSAIDs-AVOID",
        "key_rule": "AVOID-NSAIDs; AVOID-HIGH-DOSE-MTX; ETV6-RUNX1-fusion-ABSENT-germline; aza-venetoclax-preferred",
        "key_avoid": "NSAIDs AVOID (platelet dysfunction); high-dose MTX AVOID (enhanced toxicity ETV6 DHFR axis)",
        "surveillance": "Annual CBC + platelet function; BM annually if thrombocytopenia; family cascade testing",
        "targeted_rx": "Aza+venetoclax AML/MDS; allo-HSCT transformation; avoid high-dose MTX; eltrombopag emerging",
    },
    {
        "gene": "SAMD9L",
        "protein": (
            "SAMD9L -- 7q21.2 AD GOF -- 1589aa -- "
            "SAMD9L-175kDa-Antiviral-Restriction-Factor-"
            "SAMD9L-SAM-"
            "ATAD-Pediatric-AML-Monosomy7-80pct-PATHOGNOMONIC-Cerebellar-Ataxia-PATHOGNOMONIC-"
            "HSCT-Corrects-AML-NOT-Ataxia-Adrenal-Sick-Day-Rule-"
            "SAMD9L-OMIM-617053"
        ),
        "locus": "7q21.2",
        "protein_size": (
            "1589 aa / ~175 kDa / 7q21.2 SAMD9L AML molecular context: "
            "STRUCTURE: "
            "  1589 aa / ~175 kDa; antiviral restriction factor; translation inhibitor; "
            "  SAM (Sterile Alpha Motif) domain (aa 1-70): protein oligomerisation; "
            "  SAMD9L central scaffold (aa 71-1200): antiviral restriction; ribosome interaction; "
            "  C-terminal domain (aa 1200-1589): cellular signalling; "
            "  SAMD9L GOF -> enhanced translation inhibition -> haematopoietic cell death -> cytopenias + AML; "
            "  Reversion mosaicism: haematopoietic cells select UPD7 (monosomy 7) to ESCAPE SAMD9L GOF; "
            "SAMD9L-ATAD-AML CONTEXT: "
            "  ATAD (Ataxia-Thrombocytopenia-Anaemia-myeloDysplasia): SAMD9L GOF AD; "
            "  Pediatric onset; monosomy 7 in 80% of SAMD9L-ATAD bone marrows PATHOGNOMONIC; "
            "  Cerebellar ataxia PATHOGNOMONIC: progressive from early childhood; "
            "  Monosomy 7 mechanism: UPD7/monosomy 7 loses WT SAMD9L allele -> selective advantage; "
            "  Reversion mosaicism: somatic reversion mutation can rescue haematopoiesis (track by allele balance); "
            "  HSCT CORRECTS AML/MDS but NOT cerebellar ataxia -- informed consent MANDATORY; "
            "  Adrenal insufficiency: sick-day rule 10x hydrocortisone dose for illness/surgery; "
            "CLINICAL RULES FOR SAMD9L-AML: "
            "  Monosomy 7 + cerebellar ataxia in child -> SAMD9L germline testing MANDATORY; "
            "  HSCT corrects haematological disease but NOT ataxia -- families must be counselled explicitly; "
            "  Adrenal sick-day rule: hydrocortisone 10x dose during fever/surgery (adrenal insufficiency); "
            "  Reversion mosaicism monitoring: allele balance in peripheral blood q6M; "
            "  Cerebellar surveillance: annual brain MRI; physio; occupational therapy; "
            "KEY VARIANTS: "
            "  GOF gain-of-function missense: p.Thr852Met (most common); p.Arg986Cys; p.Leu1014Pro; "
            "  p.Ala699Val: attenuated phenotype; "
            "SURVEILLANCE: "
            "  Annual CBC + BM morphology; monosomy 7 FISH q6M; reversion mosaicism allele balance; "
            "  Annual brain MRI cerebellum; physiotherapy; adrenal function testing; "
            "  Hydrocortisone sick-day rule: 10x dose fever >38.5 or surgical stress; "
            "TARGETED Rx: "
            "  Allo-HSCT for AML/MDS (corrects haematological); "
            "  Azacitidine bridge to HSCT; "
            "  Hydrocortisone replacement if adrenal insufficiency confirmed; "
            "  No targeted GOF-SAMD9L therapy yet -- HSCT standard"
        ),
        "syndrome": "ATAD / Ataxia-Pancytopenia -- SAMD9L GOF",
        "inheritance": "AD GOF heterozygous",
        "aml_risk": "Pediatric AML; monosomy 7 80% PATHOGNOMONIC; HSCT corrects AML not ataxia",
        "pathognomonic": "Monosomy-7-80pct-PATHOGNOMONIC; Cerebellar-ataxia-PATHOGNOMONIC; reversion-mosaicism",
        "key_rule": "HSCT-CORRECTS-AML-NOT-ATAXIA; adrenal-sick-day-rule-10x; monosomy-7-80pct-PATHOGNOMONIC",
        "key_avoid": "Never reassure family that HSCT corrects ataxia; do not miss adrenal sick-day rule",
        "surveillance": "Annual CBC + BM; monosomy 7 FISH q6M; reversion mosaicism q6M; brain MRI annually; adrenal testing",
        "targeted_rx": "Allo-HSCT corrects AML (not ataxia); azacitidine bridge; hydrocortisone if adrenal insufficient",
    },
]

KEY_CLINICAL_DISTINCTIONS = [
    "CEBPA: biallelic-N+C-terminal-PATHOGNOMONIC; AML-40-60pct-HIGHEST-PENETRANCE; NOT-allograft-CR1-unless-biallelic-persistent; sibling-donor-CEBPA-test-MANDATORY; aza-venetoclax-R/R",
    "DDX41: late-onset-55-65yr-HALLMARK; R525H-somatic-second-hit-50pct; DONOR-DDX41-TESTING-MANDATORY; HMA-venetoclax-preferred; menin-inhibitors-emerging",
    "RUNX1: thrombocytopenia-PATHOGNOMONIC; FPD-AML-30-50pct-lifetime; SIBLING-DONORS-CI-ABSOLUTELY; azacitidine-preferred; menin-inhibitors-active",
    "GATA2: AML-MDS-30-70x-HIGHEST; NK-lymphopenia-PATHOGNOMONIC; HSCT-CURATIVE-DO-NOT-DELAY; EBV-CMV-Q3M-MANDATORY; MonoMAC-NTM",
    "TP53: LFS-AML-10-20x; complex-karyotype-PATHOGNOMONIC; AVOID-RADIATION-ABSOLUTELY; venetoclax-decitabine-preferred; WB-MRI-Toronto; eprenetapopt-APR246",
    "ANKRD26: 5UTR-PANEL-MANDATORY; standard-sequencing-MISSES-5UTR; AML-MDS-5-10pct; THC2-thrombocytopenia; aza-venetoclax",
    "ETV6: AVOID-NSAIDs; AVOID-HIGH-DOSE-MTX; AML-MDS-10-15pct; ETV6-RUNX1-fusion-ABSENT-germline; THC5-thrombocytopenia",
    "SAMD9L: monosomy-7-80pct-PATHOGNOMONIC; cerebellar-ataxia-PATHOGNOMONIC; HSCT-CORRECTS-AML-NOT-ATAXIA; adrenal-sick-day-rule-10x; reversion-mosaicism",
]


def _make_cohort(gene_idx: int, seed: int) -> dict:
    """Generate a 40-patient AML cohort for one predisposition gene."""
    rng = random.Random(seed)
    g = ATLAS_GENES[gene_idx]

    # Age distributions per gene
    age_params = {
        0: (35, 12),   # CEBPA -- familial AML; earlier onset 20-50yr
        1: (60, 7),    # DDX41 -- late onset 55-65yr HALLMARK
        2: (30, 10),   # RUNX1 -- FPD-AML; 20-40yr onset
        3: (28, 14),   # GATA2 -- young adult variable
        4: (38, 14),   # TP53 -- LFS; earlier onset possible
        5: (45, 12),   # ANKRD26 -- moderate penetrance; adult
        6: (42, 13),   # ETV6 -- adult onset; variable
        7: (14, 6),    # SAMD9L -- pediatric AML
    }
    mu, sd = age_params[gene_idx]
    ages = [max(1, min(80, int(rng.gauss(mu, sd)))) for _ in range(40)]

    # ELN risk distribution
    # ELN 2022: favourable / intermediate / adverse
    eln_dist = {
        0: [0.70, 0.22, 0.08],  # CEBPA: biallelic favourable
        1: [0.30, 0.45, 0.25],  # DDX41: variable; R525H adverse
        2: [0.25, 0.40, 0.35],  # RUNX1: intermediate/adverse
        3: [0.15, 0.40, 0.45],  # GATA2: complex karyotype adverse
        4: [0.05, 0.10, 0.85],  # TP53: complex karyotype WORST
        5: [0.35, 0.45, 0.20],  # ANKRD26: moderate risk
        6: [0.30, 0.42, 0.28],  # ETV6: variable
        7: [0.10, 0.30, 0.60],  # SAMD9L: monosomy 7 adverse
    }
    bins = eln_dist[gene_idx]
    eln_fav = int(round(40 * bins[0]))
    eln_int = int(round(40 * bins[1]))
    eln_adv = 40 - eln_fav - eln_int

    # Cytogenetics relevant to AML
    cyto_dist = {
        0: {"normal_karyotype": 20, "t(8;21)": 6, "other": 14},
        1: {"normal_karyotype": 22, "complex_karyotype": 4, "del7q": 6, "other": 8},
        2: {"normal_karyotype": 16, "complex_karyotype": 6, "del5q": 4, "other": 14},
        3: {"monosomy_7": 18, "complex_karyotype": 12, "del3q": 4, "normal_karyotype": 6},
        4: {"complex_karyotype": 28, "del17p": 8, "normal_karyotype": 2, "other": 2},
        5: {"normal_karyotype": 22, "del7q": 6, "del5q": 4, "other": 8},
        6: {"normal_karyotype": 20, "del12p": 8, "del7q": 4, "other": 8},
        7: {"monosomy_7": 32, "del7q": 4, "normal_karyotype": 2, "complex_karyotype": 2},
    }
    cyto = cyto_dist[gene_idx]

    # Treatment first-line
    tx_intensive = {0: 28, 1: 12, 2: 20, 3: 16, 4: 8, 5: 18, 6: 20, 7: 22}[gene_idx]
    tx_hma_ven = {0: 8, 1: 22, 2: 12, 3: 14, 4: 24, 5: 16, 6: 12, 7: 10}[gene_idx]
    tx_hsct_upfront = {0: 2, 1: 4, 2: 6, 3: 8, 4: 6, 5: 4, 6: 6, 7: 8}[gene_idx]
    tx_other = 40 - tx_intensive - tx_hma_ven - tx_hsct_upfront
    tx_other = max(0, tx_other)

    # Outcomes
    cr_rate = {0: 0.82, 1: 0.75, 2: 0.68, 3: 0.62, 4: 0.42, 5: 0.65, 6: 0.67, 7: 0.58}[gene_idx]
    os_2yr = {0: 0.72, 1: 0.65, 2: 0.58, 3: 0.52, 4: 0.28, 5: 0.60, 6: 0.62, 7: 0.50}[gene_idx]

    monosomy7_count = cyto.get("monosomy_7", 0)
    complex_karyotype_count = cyto.get("complex_karyotype", 0)

    return {
        "gene": g["gene"],
        "locus": g["locus"],
        "inheritance": g["inheritance"],
        "n": 40,
        "seed": seed,
        "ages": ages,
        "mean_age": round(sum(ages) / 40, 1),
        "eln_favourable": eln_fav,
        "eln_intermediate": eln_int,
        "eln_adverse": eln_adv,
        "cytogenetics": cyto,
        "monosomy7_count": monosomy7_count,
        "complex_karyotype_count": complex_karyotype_count,
        "tx_intensive": tx_intensive,
        "tx_hma_ven": tx_hma_ven,
        "tx_hsct_upfront": tx_hsct_upfront,
        "tx_other": tx_other,
        "cr_rate": cr_rate,
        "os_2yr": os_2yr,
        "aml_risk": g["aml_risk"],
        "pathognomonic": g["pathognomonic"],
        "key_rule": g["key_rule"],
        "key_avoid": g["key_avoid"],
        "targeted_rx": g["targeted_rx"],
    }


def generate_overview() -> dict:
    cohorts = [_make_cohort(i, SEED_BASE + i) for i in range(8)]

    total_patients = sum(c["n"] for c in cohorts)
    mean_age_overall = round(sum(c["mean_age"] * c["n"] for c in cohorts) / total_patients, 1)

    total_eln_fav = sum(c["eln_favourable"] for c in cohorts)
    total_eln_int = sum(c["eln_intermediate"] for c in cohorts)
    total_eln_adv = sum(c["eln_adverse"] for c in cohorts)
    total_intensive = sum(c["tx_intensive"] for c in cohorts)
    total_hma_ven = sum(c["tx_hma_ven"] for c in cohorts)
    total_hsct_upfront = sum(c["tx_hsct_upfront"] for c in cohorts)
    total_monosomy7 = sum(c["monosomy7_count"] for c in cohorts)
    total_complex_karyotype = sum(c["complex_karyotype_count"] for c in cohorts)

    avg_cr = round(sum(c["cr_rate"] for c in cohorts) / 8 * 100, 1)
    avg_os2yr = round(sum(c["os_2yr"] for c in cohorts) / 8 * 100, 1)

    per_gene = [
        {
            "gene": c["gene"],
            "locus": c["locus"],
            "n": c["n"],
            "mean_age": c["mean_age"],
            "eln_fav_pct": round(c["eln_favourable"] / 40 * 100),
            "eln_adv_pct": round(c["eln_adverse"] / 40 * 100),
            "intensive_pct": round(c["tx_intensive"] / 40 * 100),
            "hma_ven_pct": round(c["tx_hma_ven"] / 40 * 100),
            "cr_pct": round(c["cr_rate"] * 100, 1),
            "os_2yr_pct": round(c["os_2yr"] * 100, 1),
            "monosomy7_count": c["monosomy7_count"],
            "complex_karyotype_count": c["complex_karyotype_count"],
            "aml_risk": c["aml_risk"],
            "key_rule": c["key_rule"],
            "targeted_rx": c["targeted_rx"],
        }
        for c in cohorts
    ]

    return {
        "atlas": "Hereditary-AML-Predisposition-Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "n_genes": 8,
        "total_patients": total_patients,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "mean_age_overall": mean_age_overall,
        "eln_distribution": {
            "eln_favourable": total_eln_fav,
            "eln_intermediate": total_eln_int,
            "eln_adverse": total_eln_adv,
            "eln_favourable_pct": round(total_eln_fav / total_patients * 100),
            "eln_adverse_pct": round(total_eln_adv / total_patients * 100),
        },
        "cytogenetic_distribution": {
            "monosomy_7": total_monosomy7,
            "complex_karyotype": total_complex_karyotype,
        },
        "treatment_distribution": {
            "intensive_7plus3": total_intensive,
            "hma_venetoclax": total_hma_ven,
            "hsct_upfront": total_hsct_upfront,
        },
        "outcomes": {
            "avg_cr_pct": avg_cr,
            "avg_os_2yr_pct": avg_os2yr,
        },
        "key_clinical_distinctions": KEY_CLINICAL_DISTINCTIONS,
        "per_gene": per_gene,
    }


def generate_breakdown() -> dict:
    cohorts = [_make_cohort(i, SEED_BASE + i) for i in range(8)]
    gene_data = {c["gene"]: c for c in cohorts}
    return {
        "atlas": "Hereditary-AML-Predisposition-Atlas",
        "gene_data": gene_data,
        "atlas_genes": [
            {
                "gene": g["gene"],
                "locus": g["locus"],
                "protein": g["protein"],
                "syndrome": g["syndrome"],
                "inheritance": g["inheritance"],
                "aml_risk": g["aml_risk"],
                "pathognomonic": g["pathognomonic"],
                "key_rule": g["key_rule"],
                "key_avoid": g["key_avoid"],
                "surveillance": g["surveillance"],
                "targeted_rx": g["targeted_rx"],
            }
            for g in ATLAS_GENES
        ],
    }


def generate_definitions() -> dict:
    defs = []
    for g in ATLAS_GENES:
        defs.append({
            "term": f"{g['gene']} -- {g['syndrome']}",
            "definition": (
                f"{g['locus']} {g['inheritance']} -- "
                f"{g['aml_risk']} -- "
                f"Pathognomonic: {g['pathognomonic']} -- "
                f"Key rule: {g['key_rule']} -- "
                f"Avoid: {g['key_avoid']} -- "
                f"Surveillance: {g['surveillance']} -- "
                f"Targeted Rx: {g['targeted_rx']}"
            ),
        })
        defs.append({
            "term": f"{g['gene']} molecular size",
            "definition": g["protein_size"],
        })
    defs.append({
        "term": "KEY CLINICAL DISTINCTIONS -- Hereditary AML Predisposition 8-Gene Atlas",
        "definition": " | ".join(KEY_CLINICAL_DISTINCTIONS),
    })
    return {"atlas": "Hereditary-AML-Predisposition-Atlas", "definitions": defs}


if __name__ == "__main__":
    import json
    result = generate_overview()
    print(json.dumps(result, indent=2))
