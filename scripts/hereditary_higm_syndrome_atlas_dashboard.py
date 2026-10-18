#!/usr/bin/env python3
"""Hereditary-Hyper-IgM-Syndrome-Atlas -- Complete 8-Gene Reference
Hyper-IgM (HIGM) & Class Switch Recombination (CSR) Deficiency --
CD40LG-CD40-AICDA-UNG-IKBKG-MSH5-RBCK1-NFKB2
(CD40LG/HIGM1-XLR-CD40L-Most-Common-65-70pct-Pneumocystis-Infancy-PATHOGNOMONIC,
 CD40/HIGM3-AR-Both-Sexes-CD40-Absent-B-Cells-Flow-DDx-HIGM1,
 AICDA/HIGM2-AR-Germinal-Center-Hyperplasia-Lymphadenopathy-PATHOGNOMONIC,
 UNG/HIGM5-AR-Milder-Normal-LN-Architecture-DDx-AICDA,
 IKBKG/NEMO-EDA-ID-XLR-Ectodermal-Dysplasia-Triad-PATHOGNOMONIC-BCG-ABSOLUTELY-CI,
 MSH5/CSR-Deficiency-AR-MHC-III-Region-6p21.33-Normal-LN,
 RBCK1/HOIL1-LUBAC-Polyglucosan-Myopathy-IBD-Elevated-IgM-PATHOGNOMONIC,
 NFKB2/CVID3-AD-LOF-Pituitary-Stalk-Thickening-CDI-Hypopituitarism-PATHOGNOMONIC):

CD40LG  (CD40 Ligand / TNFSF5 / CD154; 261aa; Xq26.3; X-linked recessive;
         Hyper-IgM Syndrome Type 1 (HIGM1) — MOST COMMON (65–70% of all HIGM):
           CD40LG encodes CD154/CD40L, a type II transmembrane protein of the TNF superfamily;
           expressed on activated CD4+ T cells; binds CD40 on B cells → triggers class switch
           recombination (CSR) from IgM → IgG, IgA, IgE, and somatic hypermutation (SHM);
           CD40L–CD40 interaction also required for: (1) monocyte/macrophage activation (IFN-γ
           synergy → Pneumocystis killing), (2) dendritic cell maturation, (3) T cell priming
           by APCs, (4) neutrophil survival signalling;
         CD40LG LOF → absent or non-functional CD40L on activated T cells →
           (1) NO CSR: B cells cannot switch from IgM → IgG/IgA/IgE → normal/elevated IgM,
             absent IgG, absent IgA, absent IgE (HIGM phenotype);
           (2) NO MONOCYTE ACTIVATION: Pneumocystis jirovecii not killed by alveolar macrophages
             without CD40L–CD40 costimulation → Pneumocystis pneumonia at 6–12 months in UNTREATED
             MALES = PATHOGNOMONIC (first infection in an infant boy with low IgG/IgA = HIGM1
             until proven otherwise);
           (3) CRYPTOSPORIDIUM: CD40L required for intestinal immunity; Cryptosporidium parvum
             → hepatobiliary disease (HIGM1 also shares cryptosporidiosis risk with BLS-II);
           (4) CYCLIC/CHRONIC NEUTROPENIA: ~50% of HIGM1 patients; severe neutropenia episodes
             (ANC <500/µL) → bacterial infections; G-CSF may help but does not correct immune defect;
         DIAGNOSTIC: CD40L on activated T cells absent by flow cytometry (stimulate PBMCs with PMA/
           ionomycin for 4h → anti-CD154 antibody → NEGATIVE on CD3+CD4+ cells) = PATHOGNOMONIC;
           serum IgG < 200 mg/dL, IgA undetectable, IgM normal or elevated (200–800 mg/dL);
           IVIG monthly mandatory; TMP-SMX Pneumocystis prophylaxis; G-CSF for neutropenia;
           HSCT curative: eliminates T cell defect → CSR restored;
         seed SEED_BASE+0).

CD40    (CD40 / TNFRSF5; 277aa; 20q13.12; autosomal recessive;
         Hyper-IgM Syndrome Type 3 (HIGM3) — RECEPTOR SIDE OF CD40L–CD40 AXIS:
           CD40 is a TNF receptor superfamily member on B cells, monocytes, and dendritic cells;
           CD40L (on activated T cells) binds CD40 → recruits TRAF2/3/5/6 → NF-κB + MAPK
           activation → AID induction → CSR begins; CD40 also signals for B cell survival,
           memory formation, and germinal centre (GC) entry;
           CD40 LOF → identical signal failure to HIGM1 (receptor side absent = same downstream null);
         DISTINGUISHING FEATURES FROM HIGM1:
           (1) BOTH SEXES AFFECTED: CD40 gene on chromosome 20 (not X-linked); females equally
             affected — HIGM in a FEMALE is NOT HIGM1 (XLR); always consider HIGM3 in girls with HIGM;
           (2) FLOW DDx: CD40L (CD154) on activated T cells IS PRESENT AND NORMAL in HIGM3 →
             send CD40L flow AND CD40 flow simultaneously; CD40 absent on B cells (flow: anti-CD40
             antibody on CD19+ cells = NEGATIVE) = HIGM3 DIAGNOSTIC;
           (3) Identical clinical phenotype to HIGM1: Pneumocystis risk, neutropenia, Cryptosporidium;
         IMPORTANT: CD40 absent on B cells + normal CD40L on T cells = HIGM3; both absent = check
           separately (CD40L activation-induced, CD40 constitutive);
         seed SEED_BASE+1).

AICDA   (Activation-Induced Cytidine Deaminase / AID; 198aa; 12p13.31; autosomal recessive;
         Hyper-IgM Syndrome Type 2 (HIGM2) — CSR ENZYME DEFICIENCY:
           AICDA encodes AID, a DNA deaminase that converts cytosine → uracil in the switch regions
           and variable regions of immunoglobulin genes; AID is INDUCED inside germinal centre
           B cells after B cell activation + T cell help (CD40L–CD40 + IL-4/IL-13);
           AID deaminates C→U in switch region DNA → creates mismatches → repaired by UNG (uracil
           removal) → creates DSBs → resolved by NHEJ → switch recombination;
           AID also creates C→U in V regions → somatic hypermutation (SHM) → affinity maturation;
         AICDA LOF → NO CSR (IgM stays), NO SHM → absent IgG/IgA/IgE + no memory B cells;
         PATHOGNOMONIC DISTINCTION FROM HIGM1/3:
           GERMINAL CENTER HYPERPLASIA / MASSIVE LYMPHADENOPATHY: AID-deficient B cells can ENTER
           germinal centres (CD40L–CD40 intact, BCR signalling intact) but CANNOT COMPLETE CSR/SHM
           → GC B cells accumulate → MASSIVE GC hyperplasia → FLORID LYMPHADENOPATHY (massive
           cervical, axillary, inguinal nodes), SPLENOMEGALY, HEPATOMEGALY, MESENTERIC ADENOPATHY;
           this lymphadenopathy is ABSENT in HIGM1/3 (where CD40L or CD40 failure prevents GC entry
           entirely); AID-deficient lymphadenopathy mimics lymphoma — biopsy shows GC hyperplasia
           without malignancy; a child with HIGM phenotype + massive lymphadenopathy = HIGM2 until
           proven otherwise;
         MEMORY B CELLS PRESENT: AID-deficient B cells undergo GC entry + BCR diversification
           despite no CSR → memory IgM+ B cells persist; unlike HIGM1/3 where memory B cells absent;
         MILDER INFECTIONS: CD40L–CD40 axis intact → monocyte activation preserved → Pneumocystis
           risk lower than HIGM1/3; neutropenia ABSENT (CD40L signalling to granulopoiesis intact);
         seed SEED_BASE+2).

UNG     (Uracil-DNA Glycosylase 2 / UNG2; 313aa; 12q24.11; autosomal recessive;
         Hyper-IgM Syndrome Type 5 (HIGM5) — BASE EXCISION REPAIR STEP IN CSR:
           UNG encodes two isoforms: UNG1 (mitochondrial, from P1 promoter) and UNG2 (nuclear, from
           P2 promoter); UNG2 is expressed in cycling B cells and removes uracil from DNA;
           in CSR: AID converts C→U in switch region → UNG2 removes the uracil → creates abasic
           (AP) site → AP endonuclease nicks DNA → single-strand nicks become DSBs → NHEJ → CSR;
           without UNG2, AID-generated U:G mismatches accumulate but are NOT resolved to DSBs
           → CSR fails; SHM also partially impaired (UNG2-independent routes partially compensate);
         UNG LOF → absent IgG/IgA/IgE; IgM normal/elevated; NO CSR; CSR-deficient HIGM5;
         DISTINGUISHES FROM AICDA/HIGM2:
           NORMAL LYMPH NODE ARCHITECTURE: UNG2-deficient B cells can undergo BCR activation + GC
           entry (like AID-deficient) but AID is PRESENT in UNG2 deficiency; AID deaminates C→U
           BUT uracil is not removed → AID-generated mismatches repaired by MISMATCH REPAIR (MMR)
           pathway (MSH2/MSH6-dominated in V regions) → partial SHM preserved → B cells do NOT
           massively accumulate in GCs because MMR alternative pathway partially resolves AID marks;
           NORMAL LYMPH NODES = key distinguisher from HIGM2/AICDA (where GC hyperplasia prominent);
         VERY RARE: < 20 patients reported globally (2026); fewer reports than AICDA;
         UNG2-specific (nuclear) isoform alone required for CSR; UNG1 (mitochondrial) intact →
           mitochondrial DNA maintenance normal; gene panel for HIGM should include UNG when
           AICDA-negative in AR HIGM phenotype without lymphadenopathy;
         seed SEED_BASE+3).

IKBKG   (NF-κB Essential Modulator / NEMO; 419aa; Xq28; X-linked (LOF: IP in females; EDA-ID in males);
         NF-κB Essential Modulator Deficiency — Ectodermal Dysplasia with Immunodeficiency (EDA-ID):
           IKBKG encodes NEMO (NF-κB Essential Modulator / IKKγ), the regulatory subunit of the
           IκB kinase (IKK) complex (IKKα + IKKβ + IKKγ/NEMO); NEMO is required for IKK complex
           assembly and activation by TRAF6 (downstream of CD40, TLRs, IL-1R, RANK) and TRAF2
           (downstream of TNF-R1/R2); IKK phosphorylates IκBα → IκBα ubiquitination → proteasomal
           degradation → NF-κB nuclear translocation → gene expression (AID, cytokines, survival);
         IKBKG LOF → CANNOT activate NF-κB → multiple downstream defects:
           (1) NO CSR: CD40 → CD40L → TRAF6 → NEMO → IKK → NF-κB → AID; without NEMO, AID not
             induced → NO CSR → HIGM phenotype (absent IgG/IgA/IgE, elevated/normal IgM);
           (2) INNATE IMMUNITY DEFECT: TLR/IL-1R/TNFR signalling to NF-κB blocked → impaired
             innate response to bacteria (Streptococcus, Haemophilus, Mycobacteria);
           (3) BCG ABSOLUTELY CONTRAINDICATED: mycobacterial susceptibility;
         ECTODERMAL DYSPLASIA TRIAD — PATHOGNOMONIC FOR NEMO-EDA-ID:
           (a) HYPOHIDROSIS/ANHIDROSIS: sweat gland development requires NEMO/NF-κB (EDAR–EDARADD
             pathway); absent/sparse sweating → hyperthermia in hot environments (critical in neonates);
           (b) CONICAL OR ABSENT INCISORS: tooth morphogenesis (especially upper lateral incisors)
             requires ectodysplasin A signalling via EDAR→EDARADD→NEMO→NF-κB; conical pegged
             incisors or hypodontia on dental panoramic X-ray;
           (c) SPARSE/FINE HAIR: hair follicle morphogenesis requires same pathway; sparse, fine,
             hypopigmented scalp and body hair at birth;
           PATHOGNOMONIC COMBINATION: ectodermal dysplasia (anhidrosis + conical teeth + sparse hair)
             + recurrent bacterial + mycobacterial infections + HIGM pattern = NEMO-EDA-ID;
         GENETICS: IKBKG at Xq28 (X-linked); NULL mutations → LETHAL IN UTERO IN MALES (NF-κB
           null is embryonically lethal in mice; human null males die in utero); HYPOMORPHIC
           missense mutations (e.g. p.Gln315Glu, p.Ala288Gly) → survive as EDA-ID males;
           HETEROZYGOUS FEMALES: incontinentia pigmenti (IP), a skin blistering disorder with
           characteristic 4-stage skin lesions (vesicles → verrucous → hyperpigmented whorls →
           hypopigmented streaks following Blaschko lines); IP is an ectodermal dysplasia but
           NOT EDA-ID; only hemizygous males (with hypomorphic mutations) get EDA-ID phenotype;
         seed SEED_BASE+4).

MSH5    (MutS Homolog 5; 833aa; 6p21.33; autosomal recessive;
         Class Switch Recombination Deficiency (HIGM4-like) — MHC CLASS III REGION:
           MSH5 encodes a MutS homolog (MMR superfamily) that forms heterodimers with MSH4;
           MSH4–MSH5 complex is required for meiotic recombination (Holliday junction resolution)
           and for class switch recombination in B cells (binds mismatch structures in switch regions
           created by AID→UNG → facilitates DSB formation for CSR);
           MSH5 gene maps to 6p21.33, within the MHC class III region (~80 kb centromeric of C4A/C4B);
           MSH5 is expressed in GC B cells during CSR and in meiotic cells during gametogenesis;
         MSH5 LOF → absent IgG/IgA/IgE; IgM normal/elevated; NO CSR;
         PHENOTYPE DISTINCT FROM AICDA/UNG:
           NORMAL LYMPH NODE ARCHITECTURE: no GC hyperplasia (unlike AICDA/HIGM2); MSH5 acts
           DOWNSTREAM of AID+UNG → GC B cells can initiate AID-mediated deamination but cannot
           complete switch recombination DSBs → cells do not massively accumulate in GCs;
           PNEUMOCYSTIS + SINOPULMONARY: CD40L–CD40 intact → monocyte activation partially preserved;
           infections milder than HIGM1/3 but recurrent;
           MALE INFERTILITY: MSH5 (meiosis) → male azoospermia/severe oligospermia as a co-phenotype;
           LYNCH SYNDROME OVERLAP (some kindreds): 6p21.33 MHC-III MSH5 is near other MMR genes;
           rare reports of microsatellite instability in MSH5-deficient kindreds; colorectal cancer
           risk uncertain; genetic counselling re: MSI-H tumours warranted;
         seed SEED_BASE+5).

RBCK1   (HOIL-1L / RNF54 / RBCK1; 380aa; 6p25.2; autosomal recessive;
         LUBAC HOIL-1L Deficiency — Polyglucosan Myopathy + IBD + HIGM:
           RBCK1 encodes HOIL-1L (Heme-Oxidised IRP2 Ubiquitin Ligase 1-Like), one of three subunits
           of the Linear Ubiquitin Chain Assembly Complex (LUBAC): HOIL-1L (RBCK1) + HOIP (RNF31) +
           SHARPIN (SIPL1); LUBAC adds linear (Met1-linked) ubiquitin chains to NEMO/IKKγ (and
           RIPK1/RIPK2/IRAK1) → stabilises IKK complex on receptor complexes → optimal NF-κB
           activation; HOIL-1L itself has a unique E3-ligase-like domain that mono-ubiquitinates HOIP
           to limit excessive LUBAC activity (auto-regulatory);
         RBCK1 LOF → LUBAC function impaired → reduced linear ubiquitination of NEMO →
           suboptimal NF-κB activation across multiple receptor pathways (TNFR, IL-1R, TLRs, NOD2);
         PATHOGNOMONIC COMBINATION (extremely rare, <10 families 2026):
           (1) POLYGLUCOSAN STORAGE MYOPATHY: HOIL-1L appears to regulate glycogen metabolism
             via ubiquitination of glycogen-associated proteins; RBCK1 LOF → abnormal polyglucosan
             (unbranched amylose-like glycogen) deposition in skeletal muscle + liver + cardiac muscle;
             PATHOGNOMONIC: muscle biopsy with PAS staining → PAS-positive polyglucosan inclusions;
             elevated CK + hepatomegaly + cardiomyopathy;
           (2) INFLAMMATORY BOWEL DISEASE: LUBAC required for NOD2 signalling (IBD-relevant) + TNF
             signalling in intestinal epithelium; RBCK1 LOF → colitis + Crohn's-like IBD + perianal disease;
           (3) ELEVATED IgM + ABSENT IgG/IgA: LUBAC → NEMO → NF-κB → CD40L–CD40 signalling
             impaired → incomplete CSR → HIGM pattern; IVIG mandatory;
           (4) INVASIVE BACTERIAL INFECTIONS: Streptococcus, Haemophilus, Pneumococcus; less
             mycobacterial susceptibility than NEMO-EDA-ID (LUBAC-NEMO axis partially preserved);
         DDx from NEMO-EDA-ID: NO ectodermal dysplasia; NO conical teeth; NO hypohidrosis;
           polyglucosan myopathy is ABSENT in NEMO-EDA-ID; IBD prominent in RBCK1 but not NEMO;
         seed SEED_BASE+6).

NFKB2   (Nuclear Factor Kappa-B Subunit 2 / p100/p52; 968aa; 10q24.32; autosomal dominant (haploinsufficiency);
         NF-κB2 Haploinsufficiency — CVID3 with Pituitary Stalk Thickening:
           NFKB2 encodes p100 (inactive precursor) and p52 (active, after processing); NF-κB2/p52
           forms the non-canonical NF-κB heterodimer (p52 + RelB) activated by BAFF, CD40L, LTβR,
           RANK, NIK (NF-κB inducing kinase) pathway → required for B cell maturation, follicular
           dendritic cell formation, secondary lymphoid organogenesis, and central tolerance in thymus;
           p100 also acts as an IκB-like inhibitor of non-canonical NF-κB targets (dual function:
           precursor + inhibitor); haploinsufficiency or C-terminal truncating mutations →
           dominant negative p100 accumulation → blocks non-canonical pathway;
         NFKB2 LOF → combined defect:
           (1) B CELL MATURATION FAILURE: non-canonical NF-κB required for transitional → marginal
             zone and follicular B cell development → reduced B cells + absent memory B cells +
             absent class-switched antibodies → CVID phenotype (absent IgG/IgA/IgE, low IgM);
           (2) AUTOIMMUNE ENDOCRINE INVOLVEMENT — PITUITARY PATHOLOGY PATHOGNOMONIC:
             PITUITARY STALK THICKENING on MRI + central diabetes insipidus (CDI) + growth hormone
             deficiency + ACTH deficiency (secondary adrenal insufficiency) in ~50% of mutation carriers;
             PATHOGNOMONIC COMBINATION: CVID phenotype + pituitary stalk lesion/thickening = NFKB2
             haploinsufficiency until proven otherwise (DDx: sarcoidosis, Langerhans cell histiocytosis,
             germinoma — all excluded by genetics);
             hypopituitarism can precede, coincide with, or follow CVID diagnosis;
           (3) AUTOIMMUNITY: thyroiditis, adrenalitis, cytopenias;
         DISTINGUISHES FROM NFKB1: NFKB1 haploinsufficiency causes CVID + IBD/colitis + splenomegaly
           but NO pituitary involvement; NFKB2 = CVID + PITUITARY; key: pituitary MRI at diagnosis
           + annually (even if asymptomatic CDI); check morning cortisol (ACTH deficiency);
         IVIG mandatory (CVID phenotype); hydrocortisone for ACTH deficiency; DDAVP for CDI;
         seed SEED_BASE+7).
"""

import random

SEED_BASE = 3942
GENES     = ["CD40LG", "CD40", "AICDA", "UNG", "IKBKG", "MSH5", "RBCK1", "NFKB2"]

GENE_META = {
    "CD40LG": {
        "full_name": "CD40 Ligand (CD154 / TNFSF5)",
        "aa": 261, "locus": "Xq26.3", "kda": 29, "inh": "X-linked recessive (males affected)",
        "syndrome": "HIGM1 — Hyper-IgM Syndrome Type 1; most common (65–70% of HIGM)",
        "mim_gene": "300386", "mim_disease": "308230",
        "iuis_category": "Predominantly antibody deficiency — HIGM1 / IUIS 2022 Category 3",
    },
    "CD40": {
        "full_name": "CD40 (TNFRSF5)",
        "aa": 277, "locus": "20q13.12", "kda": 30, "inh": "Autosomal recessive",
        "syndrome": "HIGM3 — Hyper-IgM Syndrome Type 3; both sexes; receptor side of CD40L–CD40 axis",
        "mim_gene": "109535", "mim_disease": "606843",
        "iuis_category": "Predominantly antibody deficiency — HIGM3 / IUIS 2022 Category 3",
    },
    "AICDA": {
        "full_name": "Activation-Induced Cytidine Deaminase (AID)",
        "aa": 198, "locus": "12p13.31", "kda": 24, "inh": "Autosomal recessive",
        "syndrome": "HIGM2 — Hyper-IgM Syndrome Type 2; germinal center hyperplasia / lymphadenopathy PATHOGNOMONIC",
        "mim_gene": "605257", "mim_disease": "605258",
        "iuis_category": "Predominantly antibody deficiency — HIGM2 (CSR enzyme defect) / IUIS 2022 Category 3",
    },
    "UNG": {
        "full_name": "Uracil-DNA Glycosylase 2 (UNG2 nuclear isoform)",
        "aa": 313, "locus": "12q24.11", "kda": 36, "inh": "Autosomal recessive",
        "syndrome": "HIGM5 — Hyper-IgM Syndrome Type 5; milder; normal lymph node architecture (DDx from AICDA)",
        "mim_gene": "191525", "mim_disease": "608106",
        "iuis_category": "Predominantly antibody deficiency — HIGM5 (BER step in CSR) / IUIS 2022 Category 3",
    },
    "IKBKG": {
        "full_name": "NF-κB Essential Modulator (NEMO / IKKγ)",
        "aa": 419, "locus": "Xq28", "kda": 48, "inh": "X-linked (hypomorphic → EDA-ID in males; null → IP in females)",
        "syndrome": "EDA-ID — Ectodermal Dysplasia with Immunodeficiency; ectodermal dysplasia triad PATHOGNOMONIC",
        "mim_gene": "300248", "mim_disease": "300291",
        "iuis_category": "Combined immunodeficiency with ectodermal features — NF-κB signalling / IUIS 2022 Category 2",
    },
    "MSH5": {
        "full_name": "MutS Homolog 5 (MSH4-MSH5 heterodimer)",
        "aa": 833, "locus": "6p21.33", "kda": 93, "inh": "Autosomal recessive",
        "syndrome": "CSR Deficiency (HIGM4-like) — MHC class III region; male infertility co-phenotype; normal LN",
        "mim_gene": "603382", "mim_disease": "None (CSR-deficiency phenotype, no OMIM disease entry 2026)",
        "iuis_category": "Predominantly antibody deficiency — CSR deficiency (post-AID/UNG) / IUIS 2022 Category 3",
    },
    "RBCK1": {
        "full_name": "RanBP-type and C3HC4-type Zinc Finger 1 / HOIL-1L (LUBAC subunit)",
        "aa": 380, "locus": "6p25.2", "kda": 42, "inh": "Autosomal recessive",
        "syndrome": "LUBAC HOIL-1L Deficiency — polyglucosan myopathy + IBD + elevated IgM (HIGM-like) PATHOGNOMONIC",
        "mim_gene": "610924", "mim_disease": "615895",
        "iuis_category": "Combined immunodeficiency with autoinflammatory/metabolic features / IUIS 2022 Category 2",
    },
    "NFKB2": {
        "full_name": "Nuclear Factor Kappa-B Subunit 2 (p100/p52)",
        "aa": 968, "locus": "10q24.32", "kda": 110, "inh": "Autosomal dominant (haploinsufficiency / DN truncation)",
        "syndrome": "NF-κB2 Haploinsufficiency / CVID3 — pituitary stalk thickening + CDI PATHOGNOMONIC",
        "mim_gene": "164012", "mim_disease": "615577",
        "iuis_category": "Predominantly antibody deficiency — CVID3 non-canonical NF-κB / IUIS 2022 Category 3",
    },
}

DEFINITIONS = [
    {"term": "Hyper-IgM Syndrome (HIGM)", "definition": "Group of primary immunodeficiencies characterised by normal or elevated serum IgM with absent or severely reduced IgG, IgA, and IgE due to failure of class switch recombination (CSR); five defined genetic types: HIGM1 (CD40LG, XLR), HIGM2 (AICDA, AR), HIGM3 (CD40, AR), HIGM5 (UNG, AR); phenotypically similar: absent IgG/IgA + Pneumocystis/Cryptosporidium susceptibility; DISTINGUISHING FEATURES: lymphadenopathy (HIGM2 only), neutropenia (~50% HIGM1/3), sex (only males in HIGM1 due to XLR); CD40L flow on activated T cells is the single most informative first test"},
    {"term": "Class Switch Recombination (CSR)", "definition": "Intrachromosomal deletion recombination event that replaces the immunoglobulin Cμ exon with a downstream constant region (Cγ, Cα, or Cε) → B cell switches from IgM secretion to IgG, IgA, or IgE; requires: (1) AID (deaminates C→U in switch regions), (2) UNG2 (removes U → AP site), (3) APE1 (nicks at AP site), (4) NHEJ complex (joins DSBs), (5) MSH4/5 or MSH2/6 (stabilise switch region intermediates); failure at any step → HIGM phenotype; CD40L–CD40 interaction is the trigger for AID induction in germinal centres"},
    {"term": "CD40L (CD154) Flow Cytometry — HIGM1 Diagnostic Test", "definition": "CD40L expressed on activated CD4+ T cells (not resting); diagnostic test: stimulate PBMCs with PMA + ionomycin (4 hours) → stain with anti-CD154 (CD40L) antibody → absent staining on CD3+CD4+ cells = HIGM1 DIAGNOSTIC; test must be performed on freshly activated cells (not resting PBMCs); FALSE NEGATIVE if T cells not properly activated; CD40 on B cells (constitutively expressed) tested separately with anti-CD40 antibody → absent = HIGM3"},
    {"term": "Pneumocystis jirovecii Pneumonia (PCP) — HIGM1/3 PATHOGNOMONIC", "definition": "Opportunistic infection by Pneumocystis jirovecii (reclassified from carinii); requires CD4+ T cell + macrophage cooperation for clearance (CD40L on T cells activates macrophages + dendritic cells); in HIGM1/3: CD40L–CD40 axis broken → macrophage activation failure → Pneumocystis cannot be cleared; PATHOGNOMONIC: PCP in an infant BOY (< 1 year) = HIGM1 until proven otherwise; presentation: progressive dyspnoea, hypoxaemia, bilateral interstitial infiltrates on CXR/CT; diagnose with BAL (silver stain/PCR); treat: TMP-SMX (high dose); prophylaxis: lifelong TMP-SMX; TMP-SMX should START AT BIRTH in boys with family history of HIGM1"},
    {"term": "Germinal Center (GC) Hyperplasia — HIGM2/AICDA Pathognomonic", "definition": "In AICDA deficiency: CD40L–CD40 interaction intact → B cells enter GCs normally → AID absent → CSR fails → B cells ACCUMULATE in GCs (cannot complete programme) → progressive GC hyperplasia → massive lymphadenopathy (cervical > inguinal > axillary > mesenteric); PATHOGNOMONIC for HIGM2: massive, hard cervical nodes + splenomegaly + hepatomegaly in a child with HIGM phenotype = AICDA until proven otherwise; biopsy shows expanded GCs with lack of immunoglobulin class diversity; ABSENT in HIGM1/3 (where B cells cannot even enter GC, because CD40L–CD40 signal for GC entry is missing)"},
    {"term": "Ectodermal Dysplasia Triad — NEMO-EDA-ID Pathognomonic", "definition": "Three ectodermal features caused by NEMO/IKBKG deficiency in X-linked EDA-ID: (1) HYPOHIDROSIS/ANHIDROSIS: absence or reduction of sweat glands → hyperthermia during exercise/fever; (2) CONICAL OR ABSENT TEETH: especially upper lateral incisors — pegged, conical, or absent; dental panoramic X-ray at diagnosis; (3) SPARSE/FINE HAIR: hypotrichosis; scalp + body hair sparse, fine, pale; PATHOGNOMONIC COMBINATION: any infant BOY with recurrent infections (especially mycobacterial + Haemophilus + Streptococcus) + these ectodermal features = NEMO-EDA-ID; ectodermal dysplasia can be subtle in infancy — actively examine for all three"},
    {"term": "Cyclic Neutropenia in HIGM1", "definition": "~50% of HIGM1 (CD40LG deficiency) patients develop cyclic or chronic neutropenia (ANC < 500/µL, nadir every 21 days); mechanism: CD40L–CD40 signalling required for neutrophil survival signalling from monocytes (via G-CSF/Fas-L pathway); absent CD40L → shortened neutrophil half-life → cyclic nadir; CLINICAL IMPACT: neutropenic fevers, bacterial infections (Staphylococcus, gram-negatives) between PCP and Cryptosporidium risks; MANAGEMENT: G-CSF (filgrastim) shortens neutropenic nadirs but does NOT correct the underlying immune defect; HSCT corrects neutropenia (donor myeloid cells have functioning CD40L from T cell help)"},
    {"term": "Polyglucosan Myopathy — RBCK1/HOIL-1L PATHOGNOMONIC", "definition": "Polyglucosan = abnormal, poorly branched glycogen polymer (long linear chains, similar to amylose); accumulates in RBCK1 deficiency (LUBAC dysfunction) in muscle, liver, and cardiac tissue; MUSCLE BIOPSY: PAS (periodic acid-Schiff) staining → PAS-positive inclusions in muscle fibres; EM: filamentous polyglucosan bodies in subsarcolemmal position; laboratory: elevated CK (muscle); imaging: hepatomegaly on ultrasound; CARDIAC INVOLVEMENT: cardiomyopathy (echocardiogram at diagnosis + annually); polyglucosan myopathy in a patient with HIGM-like antibody deficiency + IBD = RBCK1 deficiency until proven otherwise; NO other HIGM gene causes polyglucosan storage"},
    {"term": "Pituitary Stalk Thickening — NFKB2 Pathognomonic", "definition": "MRI finding of enlarged pituitary stalk (normal < 3 mm at infundibulum, < 2 mm at insertion); in NFKB2 deficiency: autoimmune/inflammatory process involving pituitary stalk + posterior pituitary → central diabetes insipidus (CDI: polyuria + polydipsia + dilute urine + hypernatraemia) + ACTH deficiency (secondary adrenal insufficiency) + GH deficiency; PATHOGNOMONIC COMBINATION: CVID phenotype + pituitary stalk lesion = NFKB2 until proven otherwise; DDx with sarcoidosis, LCH, germinoma, craniopharyngioma — all exclude by genetics; MANAGEMENT: DDAVP for CDI; hydrocortisone replacement (ACTH deficiency — morning cortisol mandatory); GH replacement if growth failure"},
    {"term": "Cryptosporidiosis in HIGM — Shared Feature with BLS-II", "definition": "Cryptosporidium parvum hepatobiliary disease occurs in BOTH HIGM1 (CD40LG) and BLS-II (MHC Class II deficiency) because BOTH conditions impair CD4+ T cell function required for intestinal immunity; in HIGM1: CD40L required for macrophage activation + Cryptosporidium killing; in BLS-II: absent MHC-II → CD4+ lymphopenia → Cryptosporidium not cleared; PRESENTATION in HIGM: progressive biliary dilation, cholangiopathy, elevated ALP/GGT, cholestasis → biliary cirrhosis; HSCT is the only treatment that eliminates the underlying immune defect; antiparasitic (nitazoxanide + paromomycin) temporises but does not cure without immune reconstitution"},
    {"term": "LUBAC (Linear Ubiquitin Chain Assembly Complex)", "definition": "Three-subunit E3 ubiquitin ligase complex: HOIP (RNF31, catalytic) + HOIL-1L (RBCK1, regulatory/structural) + SHARPIN (SIPL1, adaptor); LUBAC adds Met1-linked (linear) ubiquitin chains to NEMO (IKKγ), RIPK1, RIPK2, IRAK1 → stabilises these on receptor signalling complexes → optimal NF-κB activation; linear ubiquitin (not K48 or K63) is LUBAC's unique product; HOIL-1L has RING-IBR-RING domain → can mono-ubiquitinate HOIP to auto-regulate LUBAC; RBCK1/HOIL-1L deficiency → reduced LUBAC assembly → suboptimal linear ubiquitination of NEMO → impaired IKK complex stability → reduced NF-κB activation across TNFR, IL-1R, TLR, CD40, NOD2 signalling pathways"},
    {"term": "Non-Canonical NF-κB Pathway (NFKB2/RelB)", "definition": "Activated by BAFF-R, CD40, LTβR, RANK, TWEAK-R; NIK (MAP3K14) → IKKα homodimer → phosphorylates p100 (NFKB2) C-terminal → p100 ubiquitination → proteasomal processing → p52 + RelB dimer; p52/RelB drives: secondary lymphoid organ development (splenic B cell zones, lymph node follicles, HEV formation), B cell survival (BAFF-R → B cell maturation), and central tolerance in thymus; NFKB2 LOF (haploinsufficiency or dominant-negative C-terminal truncation) → defective non-canonical NF-κB → absent follicular B cell maturation → antibody deficiency + susceptibility to pituitary autoimmunity"},
    {"term": "HIGM vs CVID — Diagnostic Distinction", "definition": "HIGM: defined by elevated/normal IgM + absent IgG/IgA; key mechanisms: CD40L–CD40 axis defect (HIGM1/3), CSR enzyme defect (HIGM2/5); CVID: defined by reduced IgG + ≥1 of IgA/IgM; multiple mechanisms including NFKB1, NFKB2, LRBA, PIK3CD, PIK3R1, TACI; PRACTICAL DISTINCTION: measure serum IgM first: elevated IgM in antibody deficiency = HIGM; low IgM in antibody deficiency = typical CVID; OVERLAP: NFKB2 deficiency causes CVID-like (low IgM, low IgG, low IgA) not classic HIGM; RBCK1 shows elevated IgM (HIGM-like); genetic testing resolves overlap; CD40L flow cytometry is the most impactful single test in a male with HIGM phenotype"},
    {"term": "BCG Contraindication — HIGM and NEMO-EDA-ID", "definition": "BCG (Bacille Calmette-Guérin) live attenuated vaccine absolutely contraindicated in HIGM1, HIGM3, NEMO-EDA-ID, and MSH5 CSR deficiency; mechanisms: (1) HIGM1/3: CD40L–CD40 disruption → macrophage activation failure → BCG mycobacteraemia/dissemination; (2) NEMO-EDA-ID: NF-κB signalling blocked in macrophages → BCG uncontrolled multiplication; clinical: BCG-itis (local suppurative lymphadenopathy) progressing to BCG-osis (disseminated BCG, osteomyelitis, hepatic granulomata) → life-threatening; BCG-osis may be the PRESENTING EVENT in undiagnosed HIGM; management: rifampicin + isoniazid ± ethambutol (BCG-osis requires ≥6 months anti-TB therapy)"},
    {"term": "IVIG in HIGM — Target Trough and Monitoring", "definition": "Intravenous immunoglobulin (IVIG) is mandatory in HIGM1/3/5 and NFKB2 deficiency to prevent bacterial infections (IgG replacement); target trough IgG: ≥ 500–800 mg/dL (typically 600–800 mg/dL); frequency: IVIG every 3–4 weeks; IMPORTANT: IVIG does NOT correct: (1) Pneumocystis risk (PCP prophylaxis TMP-SMX always mandatory regardless of IgG trough), (2) CD40L signalling defect (macrophage activation failure), (3) Cryptosporidium susceptibility, (4) neutropenia; subcutaneous IgG (SCIG) an alternative; monthly IgG trough monitoring; breakthrough infections despite adequate trough → increase dose or frequency"},
]


def _gene_cohort(gene: str, seed: int) -> dict:
    rng = random.Random(seed)
    n = 40
    meta = GENE_META[gene]

    if gene == "CD40LG":
        mean_onset_months = rng.gauss(5, 2)
        mean_dx_lag = rng.gauss(1.5, 0.8)
        pcp_n              = rng.randint(28, 35)   # 70–87%: Pneumocystis PATHOGNOMONIC under 1yr
        neutropenia_n      = rng.randint(18, 24)   # ~50% cyclic/chronic neutropenia
        cryptosporidium_n  = rng.randint(16, 24)   # 40–60%
        ivig_n             = rng.randint(38, 40)   # near-universal: IVIG lifelong mandatory
        hsct_n             = rng.randint(18, 26)   # ~50% proceed to HSCT
        outcome_good_n     = rng.randint(22, 30)
        cd40l_absent_n     = rng.randint(38, 40)   # diagnostic: CD40L absent on activated T cells
        pathognomonic = "PNEUMOCYSTIS JIROVECII IN INFANT BOY PATHOGNOMONIC: PCP at < 1 year in a male + low IgG/IgA + normal/elevated IgM = HIGM1 (CD40LG) until proven otherwise; CD40L flow cytometry (activated T cells) ABSENT = diagnostic; cyclic neutropenia in ~50% (ANC <500/µL every 21d); Cryptosporidium hepatobiliary disease; HSCT curative (corrects T cell CD40L defect); IVIG lifelong; TMP-SMX prophylaxis mandatory from birth if family history known; BCG ABSOLUTELY CONTRAINDICATED"

    elif gene == "CD40":
        mean_onset_months = rng.gauss(6, 2.5)
        mean_dx_lag = rng.gauss(2.2, 1.0)
        pcp_n              = rng.randint(26, 33)
        neutropenia_n      = rng.randint(16, 22)   # ~40–55%: similar to HIGM1
        cryptosporidium_n  = rng.randint(16, 22)
        ivig_n             = rng.randint(38, 40)
        hsct_n             = rng.randint(18, 26)
        outcome_good_n     = rng.randint(20, 28)
        cd40_absent_n      = rng.randint(38, 40)   # CD40 absent on B cells = diagnostic
        pathognomonic = "BOTH SEXES AFFECTED (NOT X-LINKED): HIGM in a FEMALE = HIGM3 (CD40) most likely (not HIGM1); CD40 absent on B cells (flow: anti-CD40 on CD19+ cells NEGATIVE) + CD40L PRESENT on activated T cells = HIGM3 DIAGNOSTIC KEY DDx FROM HIGM1; identical clinical phenotype to HIGM1: PCP, neutropenia, Cryptosporidium; always send CD40L flow AND CD40 B-cell flow simultaneously in any child with HIGM phenotype; HSCT curative; IVIG mandatory; TMP-SMX prophylaxis"

    elif gene == "AICDA":
        mean_onset_months = rng.gauss(10, 4)       # later onset: lymphadenopathy noticed 1–3yr
        mean_dx_lag = rng.gauss(3.5, 1.5)
        pcp_n              = rng.randint(8, 16)    # 20–40%: lower PCP than HIGM1/3 (CD40L–CD40 intact)
        neutropenia_n      = rng.randint(1, 4)     # neutropenia ABSENT: CD40L signalling to granulopoiesis intact
        cryptosporidium_n  = rng.randint(6, 12)    # 15–30%: lower than HIGM1/3
        ivig_n             = rng.randint(36, 40)
        hsct_n             = rng.randint(6, 14)    # HSCT less common in HIGM2 (milder)
        outcome_good_n     = rng.randint(26, 34)
        lymphadenopathy_n  = rng.randint(36, 40)   # PATHOGNOMONIC: massive LN + splenomegaly
        pathognomonic = "MASSIVE LYMPHADENOPATHY / GERMINAL CENTER HYPERPLASIA PATHOGNOMONIC: HIGM phenotype + florid cervical/axillary/inguinal lymphadenopathy + splenomegaly = AICDA (HIGM2); GC hyperplasia without malignancy on biopsy; MEMORY B CELLS PRESENT (unlike HIGM1/3); NEUTROPENIA ABSENT (CD40L axis intact); LOWER PCP/CRYPTOSPORIDIUM RISK than HIGM1/3; AICDA encodes AID (the CSR deaminase enzyme); IVIG mandatory; TMP-SMX prophylaxis for PCP; HSCT not routinely needed in mild cases but considered for severe/progressive disease; lymphoma risk (GC B cell accumulation) — annual surveillance"

    elif gene == "UNG":
        mean_onset_months = rng.gauss(14, 5)       # latest onset among classical HIGM genes
        mean_dx_lag = rng.gauss(5.0, 2.0)          # long delay: very rare
        pcp_n              = rng.randint(6, 12)    # 15–30%: lower than HIGM1/3
        neutropenia_n      = rng.randint(0, 3)     # neutropenia absent
        cryptosporidium_n  = rng.randint(4, 10)
        ivig_n             = rng.randint(35, 40)
        hsct_n             = rng.randint(2, 8)     # HSCT rarely needed: milder phenotype
        outcome_good_n     = rng.randint(28, 36)   # best outcome: milder disease
        normal_ln_n        = rng.randint(36, 40)   # normal lymph nodes = key DDx from AICDA
        pathognomonic = "NORMAL LYMPH NODE ARCHITECTURE DDx FROM AICDA: UNG5/HIGM5 has HIGM phenotype (absent IgG/IgA) WITHOUT lymphadenopathy/GC hyperplasia (unlike AICDA/HIGM2); UNG2 nuclear isoform removes AID-generated uracils from switch regions; VERY RARE (<20 patients globally 2026); mildest classical HIGM phenotype; normal lymph nodes distinguishes from AICDA; IVIG mandatory; TMP-SMX prophylaxis; monitor for lymphoma (GC dysregulation, though less than AICDA); gene panel must include UNG when AICDA-negative in AR HIGM without lymphadenopathy"

    elif gene == "IKBKG":
        mean_onset_months = rng.gauss(3, 1.5)      # early infancy: ectodermal + infections
        mean_dx_lag = rng.gauss(3.0, 1.5)
        pcp_n              = rng.randint(16, 24)   # 40–60%: Pneumocystis risk from HIGM component
        eda_triad_n        = rng.randint(36, 40)   # ectodermal dysplasia triad PATHOGNOMONIC
        mycobacterial_n    = rng.randint(18, 26)   # 45–65%: BCG-itis/osis, atypical mycobacteria
        ivig_n             = rng.randint(36, 40)
        hsct_n             = rng.randint(14, 22)   # HSCT corrects immune defect; does not fix ectoderm
        outcome_good_n     = rng.randint(18, 26)
        bcg_osis_n         = rng.randint(14, 20)   # 35–50%: BCG-itis/osis if BCG given
        pathognomonic = "ECTODERMAL DYSPLASIA TRIAD PATHOGNOMONIC: anhidrosis/hypohidrosis + conical/absent incisors + sparse/fine hair = NEMO-EDA-ID (IKBKG hypomorphic); NF-kB signalling blocked across TLR/IL-1R/TNFR/CD40 axes → HIGM phenotype + mycobacterial + bacterial susceptibility; BCG ABSOLUTELY CONTRAINDICATED (BCG-osis 35-50% if BCG given); males ONLY with hypomorphic mutations (null mutations → lethal in utero; females null → incontinentia pigmenti); HSCT corrects immune defect but ectodermal features persist; DENTAL PANORAMIC X-RAY at diagnosis for conical incisors; HOT-ENVIRONMENT PRECAUTIONS (anhidrosis → hyperthermia risk)"

    elif gene == "MSH5":
        mean_onset_months = rng.gauss(18, 6)       # later: infections may be less severe
        mean_dx_lag = rng.gauss(6.0, 2.5)
        pcp_n              = rng.randint(10, 18)
        neutropenia_n      = rng.randint(2, 6)
        cryptosporidium_n  = rng.randint(6, 12)
        ivig_n             = rng.randint(34, 40)
        hsct_n             = rng.randint(4, 12)
        outcome_good_n     = rng.randint(24, 32)
        infertility_n      = rng.randint(16, 22)   # male infertility co-phenotype (meiosis role)
        pathognomonic = "MHC CLASS III REGION 6p21.33: MSH5 mapped in MHC-III (near C4A/C4B, close to TAP1/TAP2/TAPBP at 6p21.32); NORMAL LYMPH NODES (unlike AICDA); MSH5-MSH4 heterodimer stabilises Holliday junctions in meiosis + switch region intermediates in CSR; MALE INFERTILITY (azoospermia/oligospermia) co-phenotype from meiotic recombination failure; LYNCH SYNDROME OVERLAP: rare kindreds with MSI-H tumors; CSR deficiency milder than HIGM1/3; IVIG mandatory; TMP-SMX prophylaxis; genomic deletion involving MSH5 may also affect adjacent MHC genes"

    elif gene == "RBCK1":
        mean_onset_months = rng.gauss(8, 4)
        mean_dx_lag = rng.gauss(8.0, 3.0)          # very long delay: rare, unusual phenotype
        pcp_n              = rng.randint(8, 14)
        polyglucosan_n     = rng.randint(36, 40)   # PATHOGNOMONIC: polyglucosan myopathy
        ibd_n              = rng.randint(28, 36)   # IBD in 70–90%
        ivig_n             = rng.randint(32, 38)
        hsct_n             = rng.randint(4, 10)    # very few: rare disease
        outcome_good_n     = rng.randint(14, 22)   # poor prognosis: multiorgan
        cardiomyopathy_n   = rng.randint(16, 22)
        pathognomonic = "POLYGLUCOSAN MYOPATHY + IBD + ELEVATED IgM PATHOGNOMONIC for RBCK1/HOIL-1L: MUSCLE BIOPSY with PAS staining → PAS-positive polyglucosan inclusions = RBCK1 until proven otherwise; LUBAC complex (HOIL-1L+HOIP+SHARPIN) required for linear ubiquitination of NEMO → NF-kB; glycogen metabolism dysregulation from LUBAC → polyglucosan accumulation; IBD (Crohn's-like) from NOD2/TNFR signalling impairment; cardiomyopathy (echo at diagnosis + annually); EXTREMELY RARE (<10 families 2026); no ectodermal dysplasia (DDx from NEMO-EDA-ID); IVIG for antibody deficiency; IBD management (steroids + biologics); no targeted therapy 2026"

    else:  # NFKB2
        mean_onset_months = rng.gauss(60, 24)      # later onset: often childhood/adolescence
        mean_dx_lag = rng.gauss(5.0, 2.5)          # pituitary findings may antedate CVID
        pcp_n              = rng.randint(4, 10)
        pituitary_n        = rng.randint(18, 26)   # ~50%: CDI + pituitary stalk thickening
        ivig_n             = rng.randint(36, 40)
        hsct_n             = rng.randint(2, 6)     # HSCT rarely: AD-LOF (mosaic benefit unclear)
        outcome_good_n     = rng.randint(22, 30)
        adrenal_insuff_n   = rng.randint(12, 20)   # ACTH deficiency → secondary adrenal insuff
        pathognomonic = "PITUITARY STALK THICKENING + CDI + CVID PHENOTYPE PATHOGNOMONIC for NFKB2: MRI pituitary + stalk MANDATORY at diagnosis; central diabetes insipidus (polyuria/polydipsia/hypernatraemia/dilute urine) + anterior hypopituitarism (ACTH deficiency → adrenal crisis risk; GH deficiency → growth failure) = NFKB2 non-canonical NF-kB haploinsufficiency; DISTINGUISHES FROM NFKB1 (no pituitary in NFKB1); pituitary stalk lesion may antedate CVID by years; DDx sarcoidosis/LCH/germinoma/craniopharyngioma → excluded by NFKB2 genetics; IVIG lifelong; DDAVP for CDI; hydrocortisone for ACTH deficiency (morning cortisol MANDATORY annually); AD-LOF: half-normal NF-kB2 activity is insufficient"

    # Build return dict
    base = {
        "gene": gene,
        "n": n,
        "locus": meta["locus"],
        "aa": meta["aa"],
        "kda": meta["kda"],
        "inheritance": meta["inh"],
        "syndrome": meta["syndrome"],
        "mean_onset_months": round(abs(mean_onset_months), 1),
        "mean_dx_lag_years": round(abs(mean_dx_lag), 2),
        "ivig_n": ivig_n,
        "ivig_pct": round(ivig_n / n * 100, 1),
        "hsct_n": hsct_n,
        "hsct_pct": round(hsct_n / n * 100, 1),
        "outcome_good_n": outcome_good_n,
        "outcome_good_pct": round(outcome_good_n / n * 100, 1),
        "pathognomonic": pathognomonic,
        "mim_gene": meta["mim_gene"],
        "mim_disease": meta["mim_disease"],
        "seed": seed,
    }

    if gene in ("CD40LG",):
        base.update({
            "pcp_n": pcp_n, "pcp_pct": round(pcp_n/n*100, 1),
            "neutropenia_n": neutropenia_n, "neutropenia_pct": round(neutropenia_n/n*100, 1),
            "cryptosporidium_n": cryptosporidium_n, "cryptosporidium_pct": round(cryptosporidium_n/n*100, 1),
            "cd40l_absent_n": cd40l_absent_n, "cd40l_absent_pct": round(cd40l_absent_n/n*100, 1),
        })
    elif gene == "CD40":
        base.update({
            "pcp_n": pcp_n, "pcp_pct": round(pcp_n/n*100, 1),
            "neutropenia_n": neutropenia_n, "neutropenia_pct": round(neutropenia_n/n*100, 1),
            "cryptosporidium_n": cryptosporidium_n, "cryptosporidium_pct": round(cryptosporidium_n/n*100, 1),
            "cd40_absent_n": cd40_absent_n, "cd40_absent_pct": round(cd40_absent_n/n*100, 1),
        })
    elif gene == "AICDA":
        base.update({
            "pcp_n": pcp_n, "pcp_pct": round(pcp_n/n*100, 1),
            "neutropenia_n": neutropenia_n, "neutropenia_pct": round(neutropenia_n/n*100, 1),
            "lymphadenopathy_n": lymphadenopathy_n, "lymphadenopathy_pct": round(lymphadenopathy_n/n*100, 1),
            "cryptosporidium_n": cryptosporidium_n, "cryptosporidium_pct": round(cryptosporidium_n/n*100, 1),
        })
    elif gene == "UNG":
        base.update({
            "pcp_n": pcp_n, "pcp_pct": round(pcp_n/n*100, 1),
            "neutropenia_n": neutropenia_n, "neutropenia_pct": round(neutropenia_n/n*100, 1),
            "normal_ln_n": normal_ln_n, "normal_ln_pct": round(normal_ln_n/n*100, 1),
        })
    elif gene == "IKBKG":
        base.update({
            "pcp_n": pcp_n, "pcp_pct": round(pcp_n/n*100, 1),
            "eda_triad_n": eda_triad_n, "eda_triad_pct": round(eda_triad_n/n*100, 1),
            "mycobacterial_n": mycobacterial_n, "mycobacterial_pct": round(mycobacterial_n/n*100, 1),
            "bcg_osis_n": bcg_osis_n, "bcg_osis_pct": round(bcg_osis_n/n*100, 1),
        })
    elif gene == "MSH5":
        base.update({
            "pcp_n": pcp_n, "pcp_pct": round(pcp_n/n*100, 1),
            "neutropenia_n": neutropenia_n, "neutropenia_pct": round(neutropenia_n/n*100, 1),
            "infertility_n": infertility_n, "infertility_pct": round(infertility_n/n*100, 1),
            "cryptosporidium_n": cryptosporidium_n, "cryptosporidium_pct": round(cryptosporidium_n/n*100, 1),
        })
    elif gene == "RBCK1":
        base.update({
            "pcp_n": pcp_n, "pcp_pct": round(pcp_n/n*100, 1),
            "polyglucosan_n": polyglucosan_n, "polyglucosan_pct": round(polyglucosan_n/n*100, 1),
            "ibd_n": ibd_n, "ibd_pct": round(ibd_n/n*100, 1),
            "cardiomyopathy_n": cardiomyopathy_n, "cardiomyopathy_pct": round(cardiomyopathy_n/n*100, 1),
        })
    elif gene == "NFKB2":
        base.update({
            "pcp_n": pcp_n, "pcp_pct": round(pcp_n/n*100, 1),
            "pituitary_n": pituitary_n, "pituitary_pct": round(pituitary_n/n*100, 1),
            "adrenal_insuff_n": adrenal_insuff_n, "adrenal_insuff_pct": round(adrenal_insuff_n/n*100, 1),
        })

    return base


def generate_overview() -> dict:
    cohorts = [_gene_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    total_n = sum(c["n"] for c in cohorts)
    total_hsct = sum(c["hsct_n"] for c in cohorts)
    cd40_axis = [c for c in cohorts if c["gene"] in ("CD40LG", "CD40")]
    csr_enzyme = [c for c in cohorts if c["gene"] in ("AICDA", "UNG", "MSH5")]
    nfkb_pathway = [c for c in cohorts if c["gene"] in ("IKBKG", "RBCK1", "NFKB2")]
    return {
        "atlas": "Hereditary-Hyper-IgM-Syndrome-Atlas",
        "subtitle": (
            "Hyper-IgM (HIGM) & Class Switch Recombination Deficiency — "
            "CD40LG-CD40-AICDA-UNG-IKBKG-MSH5-RBCK1-NFKB2 "
            "(IUIS 2022: Predominantly Antibody Deficiency + Combined + Autoinflammatory categories; "
            "8-gene reference spanning CD40 axis, CSR enzymes, NF-κB pathway)"
        ),
        "total_patients": total_n,
        "genes_covered": GENES,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_stats": {
            "total_hsct_pct": round(total_hsct / total_n * 100, 1),
            "cd40_axis_genes": ["CD40LG", "CD40"],
            "csr_enzyme_genes": ["AICDA", "UNG", "MSH5"],
            "nfkb_pathway_genes": ["IKBKG", "RBCK1", "NFKB2"],
            "cd40_axis_n": sum(c["n"] for c in cd40_axis),
            "csr_enzyme_n": sum(c["n"] for c in csr_enzyme),
            "nfkb_pathway_n": sum(c["n"] for c in nfkb_pathway),
            "cd40lgx_pcp_pct": round(
                sum(c.get("pcp_n", 0) for c in cd40_axis) /
                max(sum(c["n"] for c in cd40_axis), 1) * 100, 1),
            "aicda_lymphadenopathy_pct": round(
                cohorts[2].get("lymphadenopathy_n", 0) / cohorts[2]["n"] * 100, 1),
            "ikbkg_eda_triad_pct": round(
                cohorts[4].get("eda_triad_n", 0) / cohorts[4]["n"] * 100, 1),
            "nfkb2_pituitary_pct": round(
                cohorts[7].get("pituitary_n", 0) / cohorts[7]["n"] * 100, 1),
        },
        "cohort_summaries": [
            {k: v for k, v in c.items() if k != "pathognomonic"}
            for c in cohorts
        ],
        "cascade_testing_concepts": [
            "ANY CHILD with RECURRENT INFECTIONS + HIGM pattern (elevated/normal IgM + absent IgG/IgA): FIRST TEST = CD40L flow cytometry (stimulate PBMCs with PMA/ionomycin → anti-CD154 on CD3+CD4+ cells); CD40L absent = HIGM1 (CD40LG); CD40L present but CD40 absent on B cells = HIGM3 (CD40); both present = go to CSR enzyme panel (AICDA, UNG, MSH5, IKBKG gene sequencing)",
            "HIGM IN A FEMALE: X-linked HIGM1 (CD40LG) CANNOT affect females; female with HIGM phenotype → HIGM3 (CD40, AR) most likely; then HIGM2 (AICDA, AR) — lymphadenopathy present?; IKBKG hypomorphic (ectodermal dysplasia?); NFKB2 AD-LOF (pituitary stalk on MRI?); systematic approach: CD40 flow on B cells first",
            "HIGM + MASSIVE LYMPHADENOPATHY/SPLENOMEGALY: GERMINAL CENTER HYPERPLASIA = AICDA (HIGM2) PATHOGNOMONIC; this distinguishes AICDA from all other HIGM genes (CD40LG/CD40 deficiency PREVENTS GC entry, so no lymphadenopathy); biopsy shows GC hyperplasia without malignancy; AICDA encodes the AID deaminase enzyme; milder infections than HIGM1/3; TMP-SMX prophylaxis; IVIG; lymphoma surveillance (annual lymph node imaging in adults)",
            "HIGM + ECTODERMAL DYSPLASIA (anhidrosis + conical teeth + sparse hair) = NEMO-EDA-ID (IKBKG hypomorphic); BCG-itis/osis if BCG given → life-threatening; IKBKG sequencing FIRST in males with HIGM + any ectodermal features; females with IKBKG null mutations → incontinentia pigmenti (NOT EDA-ID); hot-environment precautions from birth; HSCT corrects immune defect; ectodermal features persist post-HSCT",
            "HIGM + POLYGLUCOSAN MYOPATHY + IBD = RBCK1/HOIL-1L DEFICIENCY (LUBAC): muscle biopsy PAS staining → PAS-positive inclusions = diagnostic; extremely rare (<10 families 2026); cardiomyopathy surveillance (echo annually); LUBAC (HOIL-1L+HOIP+SHARPIN) required for linear ubiquitin on NEMO → NF-κB; no ectodermal dysplasia (DDx from NEMO-EDA-ID)",
            "HIGM/CVID + PITUITARY STALK THICKENING = NFKB2 HAPLOINSUFFICIENCY: pituitary MRI MANDATORY in any patient with CVID + CDI/polyuria or unexplained hypopituitarism; morning cortisol (ACTH deficiency → adrenal crisis risk); DDAVP for CDI; DISTINGUISHES FROM NFKB1 (no pituitary in NFKB1); non-canonical NF-κB pathway (NFKB2/RelB) required for B cell maturation; AD-LOF (one allele sufficient for disease)",
            "BCG ABSOLUTELY CONTRAINDICATED in ALL 8 GENES: HIGM1/3 (CD40L–CD40 deficiency → macrophage activation failure → BCG dissemination); NEMO-EDA-ID (NF-κB blocked in macrophages → BCG uncontrolled); MSH5 (CD40 axis intact but mycobacterial susceptibility from CSR-deficiency immune dysregulation); BCG-osis presentation: suppurative regional lymphadenopathy → dissemination to liver, bone, CNS; treat BCG-osis: rifampicin + isoniazid ± ethambutol ≥ 6 months",
            "TMP-SMX (TRIMETHOPRIM-SULFAMETHOXAZOLE) PROPHYLAXIS MANDATORY IN ALL HIGM TYPES: Pneumocystis jirovecii kills children with CD40L–CD40 deficiency even with adequate IVIG (because PCP risk is from macrophage activation failure, not IgG level); prophylaxis from diagnosis until HSCT (or lifelong if no HSCT); dose: 5 mg/kg/day TMP component; alternative: atovaquone; dapsone (G6PD normal); pentamidine monthly inhaled (if sulfa-allergic)",
        ],
        "iuis_category": "IUIS 2022: Category 3 (Predominantly Antibody Deficiency) + Category 2 (Combined T/B Cell Immunodeficiency with syndromic features — IKBKG/RBCK1) — Hyper-IgM Syndromes and CSR Deficiency",
    }


def generate_breakdown() -> dict:
    cohorts = [_gene_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    return {
        "atlas": "Hereditary-Hyper-IgM-Syndrome-Atlas",
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "gene_cohorts": cohorts,
        "diagnostic_groups": {
            "cd40_axis_defects": {
                "genes": ["CD40LG", "CD40"],
                "mechanism": (
                    "CD40 ligand (on T cells)–CD40 (on B cells) interaction failure → "
                    "NO AID induction → NO CSR → absent IgG/IgA/IgE + elevated/normal IgM; "
                    "ALSO: impaired macrophage/DC activation → Pneumocystis + Cryptosporidium"
                ),
                "shared_feature": "Identical clinical phenotype; distinguished by sex (HIGM1 males only) and flow cytometry (CD40L vs CD40 absent)",
                "key_ddx": "HIGM1 (CD40LG, XLR, males only): CD40L absent on activated T cells; HIGM3 (CD40, AR, both sexes): CD40 absent on B cells + CD40L PRESENT",
            },
            "csr_enzyme_defects": {
                "genes": ["AICDA", "UNG", "MSH5"],
                "mechanism": (
                    "CSR enzyme/factor failure downstream of CD40L–CD40 → AID deaminase (AICDA) "
                    "or uracil removal (UNG) or switch region stabilisation (MSH5) defective → "
                    "NO switch recombination → absent IgG/IgA/IgE"
                ),
                "shared_feature": "CD40L–CD40 axis INTACT; B cells CAN enter germinal centres (partially or fully); CD40L flow NORMAL; no neutropenia",
                "key_ddx": "AICDA: GC hyperplasia/lymphadenopathy PATHOGNOMONIC; UNG: normal LN (DDx from AICDA); MSH5: normal LN + male infertility co-phenotype",
            },
            "nfkb_pathway_defects": {
                "genes": ["IKBKG", "RBCK1", "NFKB2"],
                "mechanism": (
                    "NF-κB pathway defects impairing AID induction/B cell development: "
                    "IKBKG/NEMO (IKK complex) → canonical NF-κB; "
                    "RBCK1/HOIL-1L (LUBAC) → linear ubiquitin on NEMO → IKK activation; "
                    "NFKB2 (p100/p52) → non-canonical NF-κB → B cell maturation"
                ),
                "shared_feature": "Broader NF-κB impairment than pure CD40 axis or CSR enzyme defects; syndromic features present (ectodermal dysplasia, polyglucosan myopathy, pituitary stalk)",
                "key_ddx": "IKBKG: EDA-ID triad + mycobacterial; RBCK1: polyglucosan myopathy + IBD; NFKB2: pituitary stalk thickening + CDI",
            },
        },
        "pathognomonic_features": {
            "CD40LG_pcp_infant_male": (
                "PNEUMOCYSTIS JIROVECII in an INFANT BOY (< 1 year) with elevated IgM + absent IgG/IgA = "
                "HIGM1 (CD40LG) PATHOGNOMONIC; CD40L flow on activated T cells IMMEDIATELY"
            ),
            "CD40_both_sexes": (
                "HIGM in a FEMALE = NOT HIGM1 (X-linked); CD40 (AR) most likely → "
                "CD40 flow cytometry: absent CD40 on B cells + NORMAL CD40L on T cells = HIGM3 DIAGNOSTIC"
            ),
            "AICDA_lymphadenopathy": (
                "MASSIVE LYMPHADENOPATHY / GC HYPERPLASIA + HIGM phenotype = AICDA (HIGM2) PATHOGNOMONIC; "
                "biopsy shows GC expansion without malignancy; absent in HIGM1/3 where GC entry prevented"
            ),
            "UNG_normal_LN": (
                "NORMAL LYMPH NODE ARCHITECTURE + HIGM phenotype (absent IgG/IgA + elevated IgM) = "
                "UNG (HIGM5) key DDx from AICDA; very rare; milder disease"
            ),
            "IKBKG_eda_triad": (
                "ECTODERMAL DYSPLASIA (anhidrosis + conical/absent incisors + sparse hair) + "
                "recurrent bacterial/mycobacterial infections = NEMO-EDA-ID (IKBKG hypomorphic) PATHOGNOMONIC"
            ),
            "RBCK1_polyglucosan": (
                "POLYGLUCOSAN INCLUSIONS on muscle biopsy (PAS+) + HIGM-like antibody deficiency + IBD = "
                "RBCK1/HOIL-1L (LUBAC) deficiency PATHOGNOMONIC — no other HIGM gene causes polyglucosan storage"
            ),
            "NFKB2_pituitary": (
                "PITUITARY STALK THICKENING + CDI + hypopituitarism + CVID phenotype = "
                "NFKB2 haploinsufficiency PATHOGNOMONIC; pituitary MRI MANDATORY at CVID diagnosis"
            ),
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Hyper-IgM-Syndrome-Atlas",
        "definitions": DEFINITIONS,
        "gene_metadata": GENE_META,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "key_distinguishers": {
            "HIGM1_vs_HIGM3": (
                "HIGM1 (CD40LG, XLR): MALES ONLY; CD40L absent on activated T cells (flow); "
                "HIGM3 (CD40, AR): BOTH SEXES; CD40 absent on B cells + CD40L PRESENT; "
                "SINGLE KEY DDx: sex + flow cytometry (CD40L on T cells + CD40 on B cells simultaneously)"
            ),
            "HIGM1_vs_HIGM2": (
                "HIGM1 (CD40LG): NO lymphadenopathy (GC entry prevented by absent CD40L–CD40); neutropenia ~50%; "
                "HIGM2 (AICDA): MASSIVE LYMPHADENOPATHY (GC hyperplasia — B cells enter GC but cannot complete CSR); NO neutropenia; "
                "key: lymphadenopathy = HIGM2; neutropenia = HIGM1; both can have PCP but HIGM1 > HIGM2 PCP risk"
            ),
            "HIGM2_AICDA_vs_HIGM5_UNG": (
                "Both AR, both CSR enzyme defects (downstream of CD40L–CD40); "
                "AICDA: LYMPHADENOPATHY / GC HYPERPLASIA PATHOGNOMONIC; larger lymph nodes; memory IgM+ B cells present; "
                "UNG: NORMAL LYMPH NODES (key distinguisher); milder phenotype; UNG acts downstream of AID; very rare"
            ),
            "IKBKG_vs_RBCK1_NF-kB_defects": (
                "Both X-linked or AR NF-κB pathway defects with HIGM-like pattern; "
                "IKBKG/NEMO: ECTODERMAL DYSPLASIA (anhidrosis + conical teeth + sparse hair) + BCG risk; X-linked hypomorphic males; "
                "RBCK1/HOIL-1L: POLYGLUCOSAN MYOPATHY + IBD + NO ectodermal dysplasia; AR; both sexes"
            ),
            "NFKB1_vs_NFKB2": (
                "BOTH AD-LOF causing CVID-like antibody deficiency; "
                "NFKB1: canonical NF-κB (p50/p65); CVID + IBD/colitis + splenomegaly; NO pituitary involvement; "
                "NFKB2: non-canonical NF-κB (p100/p52); CVID + PITUITARY STALK THICKENING + CDI + hypopituitarism; "
                "KEY: pituitary MRI in CVID (absent in NFKB1, present in NFKB2)"
            ),
            "HIGM_vs_XLA_vs_CVID": (
                "XLA (BTK): ABSENT B CELLS (pre-B cell development failure); IgM also absent; B cell count ZERO; "
                "HIGM: B cells PRESENT; IgM normal/elevated; only IgG/IgA absent; "
                "CVID: B cells present; ALL antibody classes reduced (including IgM); typically adult onset; "
                "KEY: serum IgM level distinguishes HIGM (elevated/normal) from XLA (absent) from CVID (low)"
            ),
        },
        "diagnostic_pathway": [
            "Step 1: Serum immunoglobulins (IgG, IgA, IgM, IgE) — ELEVATED or normal IgM + absent IgG/IgA = HIGM pattern; proceed immediately to CD40L flow cytometry without waiting for genetic panel",
            "Step 2: CD40L (CD154) flow cytometry — stimulate PBMCs with PMA + ionomycin (4h) → anti-CD154 antibody on CD3+CD4+ T cells; CD40L ABSENT = HIGM1 (CD40LG); send CD40LG sequencing + start IVIG + TMP-SMX IMMEDIATELY",
            "Step 3: If CD40L PRESENT (HIGM1 excluded) → CD40 flow on B cells (anti-CD40 on CD19+); CD40 ABSENT = HIGM3; CD40 PRESENT = proceed to CSR enzyme panel (AICDA, UNG) + NF-kB pathway genes (IKBKG, RBCK1, NFKB2)",
            "Step 4: Physical examination — ectodermal dysplasia (anhidrosis + conical teeth + sparse hair) → IKBKG/NEMO EDA-ID; polyglucosan myopathy features (elevated CK, weakness) + IBD → RBCK1/HOIL-1L",
            "Step 5: Lymphadenopathy/splenomegaly → AICDA (HIGM2) most likely; biopsy for GC hyperplasia; AICDA sequencing; no lymphadenopathy → UNG or MSH5",
            "Step 6: CVID-like phenotype (low IgM + low IgG/IgA) → pituitary MRI (NFKB2 screen); NFKB2 sequencing if stalk thickening or CDI; morning cortisol (ACTH deficiency)",
            "Step 7: ALL PATIENTS REGARDLESS OF GENE — BCG ABSOLUTELY CONTRAINDICATED; TMP-SMX prophylaxis from diagnosis; IVIG monthly; no live vaccines until immune reconstitution",
            "Step 8: HSCT evaluation — indicated for HIGM1/3 (corrects T cell CD40L defect); IKBKG (corrects immune defect, ectodermal features persist); less clear for CSR enzyme defects (HIGM2/5/MSH5) — consider in progressive disease; NOT standard for NFKB2 AD-LOF",
        ],
    }
