#!/usr/bin/env python3
"""Hereditary-MHC-Bare-Lymphocyte-Syndrome-Atlas -- Complete 8-Gene Reference
MHC Class II Deficiency / MHC Class I Deficiency / ZAP70 CID --
CIITA-RFXANK-RFX5-RFXAP-TAP1-TAP2-TAPBP-ZAP70
(CIITA/BLS-IIA-AR-MHC-II-Master-Transactivator,
 RFXANK/BLS-IIB-AR-Moroccan-North-African-Founder-Most-Common-BLS-II,
 RFX5/BLS-IIC-AR-RFX-Transcription-Factor,
 RFXAP/BLS-IID-AR-Rarest-BLS-II,
 TAP1/BLS-I-AR-MHC-I-Loading-Complex,
 TAP2/BLS-I-AR-TAP-Heterodimer,
 TAPBP/BLS-I-AR-Tapasin-Peptide-Editor-Very-Rare,
 ZAP70/CID-AR-Selective-CD8-Absence-PATHOGNOMONIC-Mennonite-Founder):

CIITA   (Class II Major Histocompatibility Complex Transactivator; 1130aa; 16p13.13; autosomal recessive;
         Bare Lymphocyte Syndrome Type II, Complementation Group A (BLS-IIA):
           CIITA is the master non-DNA-binding transactivator of ALL MHC class II genes (HLA-DR, HLA-DP, HLA-DQ);
           CIITA does not bind MHC-II promoters directly but forms a transcriptional complex with RFX5/RFXANK/RFXAP
             (RFX complex) + NF-Y + CREB on the W/X/Y box composite element upstream of HLA-DR, -DP, -DQ;
           CIITA LOF → complete absence of HLA-DR/DP/DQ on dendritic cells, macrophages, B cells, and thymic epithelium →
             (1) THYMIC CD4+ SELECTION FAILURE: CD4+ T cells require MHC-II for positive selection in thymus → CD4+ T cells
               absent/severely reduced; CD8+ T cells preserved (use MHC-I for selection → MHC-I intact);
             (2) ANTIGEN PRESENTATION FAILURE: professional APCs cannot activate naive CD4+ T cells → no Th1/Th2/Tfh help →
               hypogammaglobulinemia (despite normal B cell numbers — B cells require T cell help for class-switching);
             (3) REGULATORY T CELL DEFECT: Treg development requires MHC-II → Treg deficiency → susceptibility to autoimmunity;
           CLINICAL: presents in first year with recurrent sinopulmonary infections (bacteria + Pneumocystis),
             chronic diarrhoea (Cryptosporidium parvum PATHOGNOMONIC gut infection in BLS-II), failure to thrive;
             bronchiectasis develops if untreated; Cryptosporidium cholangiopathy (sclerosing cholangitis-like) → liver failure
             if untreated; HSCT is curative but outcomes worse than SCID (thymic damage may be irreversible);
           DIAGNOSTIC: CD4+ lymphopenia on CBC; immunophenotyping: CD4+ <200/µL (near-absent), CD8+ preserved;
             HLA-DR expression absent on B cells (flow: anti-HLA-DR antibody → no staining) = PATHOGNOMONIC;
             serum immunoglobulins low (all classes); normal B cell numbers;
           seed SEED_BASE+0).

RFXANK  (Regulatory Factor X-Associated Ankyrin Repeat Protein; 260aa; 19q13.11; autosomal recessive;
         Bare Lymphocyte Syndrome Type II, Complementation Group B (BLS-IIB) — MOST COMMON BLS-II WORLDWIDE:
           RFXANK (also called ANKRD16/BLS2B/RFXB) is part of the trimeric RFX complex (RFX5+RFXANK+RFXAP)
             that binds the X1 box of MHC-II promoters with exquisite sequence specificity;
           RFXANK contains ankyrin repeats required for stable RFX trimer formation; RFXANK LOF →
             RFX complex destabilised → fails to bind MHC-II promoters → no CIITA recruitment → no MHC-II expression;
           MOROCCAN/NORTH AFRICAN FOUNDER MUTATION: 33-nucleotide in-frame deletion (c.169-201del33, deletes 11 amino acids
             in ankyrin repeat 3) accounts for >80% of RFXANK mutations in North African and Middle Eastern patients;
             PATHOGNOMONIC in patients of Moroccan/North African/Arab descent — targeted founder allele testing first
             dramatically reduces cost and time to diagnosis;
           RFXANK BLS-IIB is the MOST COMMON complementation group for BLS-II (accounts for ~50-60% of BLS-II cases
             globally due to the Moroccan founder effect); all four BLS-II complementation groups have identical
             immunological phenotype (cannot be distinguished clinically — only genetic/complementation analysis distinguishes);
           seed SEED_BASE+1).

RFX5    (Regulatory Factor X5; 616aa; 1q21.3; autosomal recessive;
         Bare Lymphocyte Syndrome Type II, Complementation Group C (BLS-IIC):
           RFX5 is the largest RFX complex subunit; contains an RFX DNA-binding domain (DBD) that directly contacts
             the X1 box of MHC-II promoters (sequence: GTNRCCNANYGC); RFX5 DNA binding is the anchor of the
             RFX trimeric complex; RFXANK and RFXAP stabilise and regulate RFX5 binding;
           RFX5 LOF → RFX complex fails to assemble or bind DNA → CIITA cannot be recruited → no MHC-II;
           RFX5 mutations account for ~10-15% of BLS-II cases; less common than RFXANK;
           CLINICAL PHENOTYPE: identical to other BLS-II complementation groups — CD4+ lymphopenia, absent HLA-DR,
             Cryptosporidium + Pneumocystis susceptibility, hypogammaglobulinemia;
           KEY DISTINGUISHER FROM BLS-IIB: RFX5 mutations occur across all ethnicities (not enriched in North Africans);
             no single founder mutation — compound heterozygotes and various missense/frameshift mutations across gene;
           seed SEED_BASE+2).

RFXAP   (Regulatory Factor X-Associated Protein; 272aa; 13q14.2; autosomal recessive;
         Bare Lymphocyte Syndrome Type II, Complementation Group D (BLS-IID) — RAREST BLS-II GROUP:
           RFXAP stabilises the RFX5 DNA-binding domain and enhances RFX5 affinity for X1 box; RFXAP
             contains a unique N-terminal domain with acidic repeats not homologous to known protein families;
           RFXAP LOF → RFX complex destabilised → MHC-II expression absent → identical BLS-II phenotype;
           BLS-IID is the rarest complementation group: <5% of BLS-II cases; <50 patients described worldwide;
           CLINICAL: identical phenotype to other BLS-II groups — CD4+ lymphopenia, absent HLA-DR, Cryptosporidium;
             RFXAP deficiency not enriched in any specific ethnic group; individual kindreds; founder effects rare;
           DIAGNOSTIC CLUE: when BLS-II phenotype is confirmed (absent HLA-DR, CD4+ low) but RFXANK (most common)
             and RFX5 (second) mutations not found → RFXAP sequencing; CIITA should be sequenced in parallel;
             complementation group assignment now largely replaced by comprehensive NGS panel;
           seed SEED_BASE+3).

TAP1    (Transporter Associated with Antigen Processing 1; 811aa; 6p21.32; autosomal recessive;
         Bare Lymphocyte Syndrome Type I, TAP1 subtype — MHC CLASS I DEFICIENCY:
           TAP1 forms a heterodimer with TAP2 in the ER membrane; TAP1/TAP2 heterodimer transports cytosol-derived
             peptides (from proteasomal degradation) into the ER lumen for loading onto nascent MHC-I (HLA-A/B/C + β2m + peptide);
           TAPBP (tapasin) bridges TAP1/2 to MHC-I chains → forms the MHC-I peptide loading complex (PLC);
           TAP1 LOF → peptide transport failure → empty (unstable) MHC-I molecules on cell surface → MHC-I surface expression
             severely reduced; CD8+ T cell positive selection impaired (CD8+ lymphopenia mild-moderate);
           CLINICAL PHENOTYPE OF BLS-I (DRAMATICALLY DIFFERENT FROM BLS-II):
             (1) NOT early-onset severe infections: onset usually childhood/adolescence (unlike BLS-II onset <1yr);
             (2) SINOPULMONARY INFECTIONS: predominantly gram-negative bacteria (Pseudomonas, Haemophilus) + atypical
               mycobacteria (MAC); bronchiectasis cardinal feature; unlike BLS-II (Pneumocystis/Cryptosporidium);
             (3) NECROTISING GRANULOMATOUS SKIN LESIONS PATHOGNOMONIC: deep cutaneous ulcerating granulomas
               on midface, lower extremities; histology: necrotising granulomas without infectious aetiology;
               (DIFFERENT from typical skin infections — sterile ulcers); erythema nodosum; Behçet-like vasculitis;
             (4) No Cryptosporidium (unlike BLS-II) — CD4+ T cells intact (MHC-I does not affect CD4+ selection);
           DIAGNOSTIC: CD8+ lymphopenia (mild: CD8+ 50-200/µL, not absent); HLA-A/B/C expression on lymphocytes
             severely reduced (flow: near-absent); β2-microglobulin elevated in urine;
           seed SEED_BASE+4).

TAP2    (Transporter Associated with Antigen Processing 2; 703aa; 6p21.32; autosomal recessive;
         Bare Lymphocyte Syndrome Type I, TAP2 subtype:
           TAP2 forms the ABC transporter heterodimer with TAP1; TAP2 provides the second ATP-binding cassette
             (ABC) domain of the TAP complex; TAP2 LOF → TAP1/2 transport-incompetent → ER peptide loading failure;
           TAP2 mutations produce IDENTICAL phenotype to TAP1 deficiency (BLS type I):
             sinopulmonary infections, bronchiectasis, necrotising granulomatous skin lesions, Behçet-like vasculitis;
           EPIDEMIOLOGY: TAP2 deficiency slightly more common than TAP1 deficiency in published literature; ~70% of
             BLS-I cases reported; no specific ethnic enrichment; mutations distributed across gene;
           KEY DISTINCTION FROM BLS-II (CD8+ vs CD4+ lymphopenia):
             BLS-II (CIITA/RFXANK/RFX5/RFXAP): CD4+ absent, CD8+ preserved, HLA-DR absent → Cryptosporidium;
             BLS-I (TAP1/TAP2/TAPBP): CD8+ low-moderate, CD4+ preserved, HLA-A/B/C low → granulomatous skin;
             This distinction is the KEY immunological DDx at the bedside before genetics;
           TAP2 gene is only ~200kb from TAP1 and TAPBP on chromosome 6p21.32 (within MHC region);
           seed SEED_BASE+5).

TAPBP   (Tapasin; Transporter Associated With Antigen Processing Binding Protein; 428aa; 6p21.32; autosomal recessive;
         Bare Lymphocyte Syndrome Type I, Tapasin subtype — RAREST BLS-I:
           Tapasin is an ER-resident glycoprotein that bridges TAP1 to MHC-I heavy chain + β2-microglobulin;
           Tapasin is a 'peptide editor' — it catalyses exchange of low-affinity for high-affinity peptides
             in the MHC-I groove; tapasin LOF → unstable empty MHC-I released from ER → low surface MHC-I,
             but LESS severe than TAP1/TAP2 deficiency (some tapasin-independent MHC-I loading persists);
           TAPBP deficiency is VERY RARE: <15 patients reported worldwide (2026); allele frequency extremely low;
           CLINICAL: similar to TAP1/TAP2 (sinopulmonary + skin lesions) but MILDER severity due to partial
             tapasin-independent MHC-I loading; some patients diagnosed in adulthood; granulomatous skin lesions
             present but may be less florid; bronchiectasis develops over years;
           TAPBP is encoded within the MHC region at 6p21.32, only ~50kb from TAP1 and TAP2 →
             all three BLS-I genes co-localise at 6p21.32 (MHC class II region);
             compound heterozygotes for TAP1+TAPBP reported (adjacent gene deletions);
           seed SEED_BASE+6).

ZAP70   (Zeta-Chain-Associated Protein Kinase 70; 619aa; 2q11.2; autosomal recessive;
         ZAP-70 Deficiency — Selective CD8+ T Cell Lymphopenia / Combined Immunodeficiency:
           ZAP-70 is a non-receptor tyrosine kinase that is recruited to the TCR-CD3 complex after TCR ligation
             and CD3-ζ chain ITAMs are phosphorylated by Lck; ZAP-70 phosphorylates LAT and SLP-76 →
             downstream activation of PLCγ1, PI3K, RAS/MAPK cascade → IL-2 production, T cell proliferation;
           ZAP70 LOF → TCR signal transduction failure; the PARADOXICAL phenotype occurs because:
             (1) CD4+ T CELLS: CD4+ cells depend on Lck (which binds CD4 co-receptor) for strong ZAP-70-independent
               pre-TCR signal during positive selection; CD4+ T cells can escape thymus (low ZAP-70 activity tolerated
               for CD4+ selection); but CD4+ cells are FUNCTIONALLY ANERGIC (cannot activate in periphery without ZAP-70);
             (2) CD8+ T CELLS ABSENT: CD8+ positive selection requires stronger TCR signal with ZAP-70 → CD8+ T cells
               fail positive selection → CD8+ cells virtually absent from periphery;
           PATHOGNOMONIC PHENOTYPE: CD4+ T cells present (numerically) but functionally non-responsive +
             CD8+ T cells virtually ABSENT + normal NK cells + normal B cells = ZAP-70 deficiency PATHOGNOMONIC;
             NO OTHER IMMUNODEFICIENCY produces this specific pattern;
           MENNONITE/OLD ORDER AMISH FOUNDER ALLELE: p.Cys319Arg (c.955T>C) — founder mutation in Mennonite
             and Old Order Amish communities; if patient has this ancestry → ZAP70 p.Cys319Arg targeted first;
           CLINICAL: recurrent sinopulmonary infections; failure to thrive; Pneumocystis susceptibility (CD4+
             functional deficit despite numerical preservation); autoimmune features (CD4+ cells dysregulated,
             lack Treg balance); ZAP-70 hypomorphic mutations → less severe phenotype (Omenn-like);
           DIAGNOSTIC: CD4+ present but CD3+CD8+ near-absent; proliferation assays: CD4+ fail PHA response;
             HSCT curative; gene-corrected autologous HSC therapy in development;
           seed SEED_BASE+7).
"""

import random

SEED_BASE = 3934
GENES     = ["CIITA", "RFXANK", "RFX5", "RFXAP", "TAP1", "TAP2", "TAPBP", "ZAP70"]

GENE_META = {
    "CIITA": {
        "full_name": "Class II Major Histocompatibility Complex Transactivator",
        "aa": 1130, "locus": "16p13.13", "kda": 130, "inh": "Autosomal recessive",
        "syndrome": "BLS-IIA — Bare Lymphocyte Syndrome Type II Group A",
        "mim_gene": "600005", "mim_disease": "209920",
        "iuis_category": "Combined T and B cell immunodeficiency — MHC Class II deficiency / IUIS 2022 Category 2",
    },
    "RFXANK": {
        "full_name": "Regulatory Factor X-Associated Ankyrin Repeat Protein (RFXB/ANKRD16)",
        "aa": 260, "locus": "19q13.11", "kda": 33, "inh": "Autosomal recessive",
        "syndrome": "BLS-IIB — Most Common BLS-II; Moroccan/North African 33nt founder deletion",
        "mim_gene": "603200", "mim_disease": "209920",
        "iuis_category": "Combined T and B cell immunodeficiency — MHC Class II deficiency / IUIS 2022 Category 2",
    },
    "RFX5": {
        "full_name": "Regulatory Factor X5 (DNA-Binding Subunit of RFX Complex)",
        "aa": 616, "locus": "1q21.3", "kda": 67, "inh": "Autosomal recessive",
        "syndrome": "BLS-IIC — Bare Lymphocyte Syndrome Type II Group C",
        "mim_gene": "601863", "mim_disease": "209920",
        "iuis_category": "Combined T and B cell immunodeficiency — MHC Class II deficiency / IUIS 2022 Category 2",
    },
    "RFXAP": {
        "full_name": "Regulatory Factor X-Associated Protein",
        "aa": 272, "locus": "13q14.2", "kda": 36, "inh": "Autosomal recessive",
        "syndrome": "BLS-IID — Rarest BLS-II complementation group (<5% of BLS-II)",
        "mim_gene": "601861", "mim_disease": "209920",
        "iuis_category": "Combined T and B cell immunodeficiency — MHC Class II deficiency / IUIS 2022 Category 2",
    },
    "TAP1": {
        "full_name": "Transporter Associated with Antigen Processing 1 (ABC Transporter ABCB2)",
        "aa": 811, "locus": "6p21.32", "kda": 81, "inh": "Autosomal recessive",
        "syndrome": "BLS-I TAP1 — MHC Class I Deficiency; necrotising granulomatous skin PATHOGNOMONIC",
        "mim_gene": "170260", "mim_disease": "604571",
        "iuis_category": "Combined T and B cell immunodeficiency — MHC Class I deficiency / IUIS 2022 Category 2",
    },
    "TAP2": {
        "full_name": "Transporter Associated with Antigen Processing 2 (ABC Transporter ABCB3)",
        "aa": 703, "locus": "6p21.32", "kda": 75, "inh": "Autosomal recessive",
        "syndrome": "BLS-I TAP2 — Most Common BLS-I; bronchiectasis + sinopulmonary + skin granulomas",
        "mim_gene": "170261", "mim_disease": "604571",
        "iuis_category": "Combined T and B cell immunodeficiency — MHC Class I deficiency / IUIS 2022 Category 2",
    },
    "TAPBP": {
        "full_name": "Tapasin (Transporter Associated with Antigen Processing Binding Protein)",
        "aa": 428, "locus": "6p21.32", "kda": 48, "inh": "Autosomal recessive",
        "syndrome": "BLS-I Tapasin — Rarest BLS-I; milder phenotype (partial tapasin-independent MHC-I loading)",
        "mim_gene": "601962", "mim_disease": "604571",
        "iuis_category": "Combined T and B cell immunodeficiency — MHC Class I deficiency / IUIS 2022 Category 2",
    },
    "ZAP70": {
        "full_name": "Zeta-Chain-Associated Protein Kinase 70 (Syk-Family Tyrosine Kinase)",
        "aa": 619, "locus": "2q11.2", "kda": 70, "inh": "Autosomal recessive",
        "syndrome": "ZAP-70 Deficiency — Selective CD8+ Absence PATHOGNOMONIC; Mennonite p.Cys319Arg founder",
        "mim_gene": "176947", "mim_disease": "269840",
        "iuis_category": "Combined T and B cell immunodeficiency — TCR signal transduction defect / IUIS 2022 Category 2",
    },
}

DEFINITIONS = [
    {"term": "Bare Lymphocyte Syndrome Type II (BLS-II)", "definition": "Primary immunodeficiency caused by absent MHC Class II (HLA-DR/DP/DQ) expression on antigen-presenting cells and thymic epithelium; caused by mutations in CIITA, RFXANK, RFX5, or RFXAP (four complementation groups with identical phenotype); hallmark = CD4+ T cell lymphopenia + absent HLA-DR on B cells + normal CD8+ T cells + normal B cell numbers + hypogammaglobulinemia; Cryptosporidium is the PATHOGNOMONIC infection"},
    {"term": "Bare Lymphocyte Syndrome Type I (BLS-I)", "definition": "Primary immunodeficiency caused by absent MHC Class I (HLA-A/B/C) surface expression due to peptide loading complex failure; caused by mutations in TAP1, TAP2, or TAPBP; hallmark = mild-moderate CD8+ T cell lymphopenia + absent/low HLA-A/B/C + preserved CD4+ cells; DIFFERENT from BLS-II: no Cryptosporidium, instead necrotising granulomatous skin lesions + sinopulmonary bacterial infections"},
    {"term": "MHC Class II (HLA-DR/DP/DQ) Expression", "definition": "Cell-surface heterodimeric glycoproteins on professional APCs (dendritic cells, macrophages, B cells) and thymic epithelium; present extracellular/endosomal-derived peptides to CD4+ T cells (TCR + CD4 co-receptor binds MHC-II + peptide); regulated transcriptionally by CIITA + RFX5/RFXANK/RFXAP complex at W/X/Y box composite promoter elements; absent in BLS-II — flow cytometry HLA-DR on B cells is the diagnostic test"},
    {"term": "MHC Class I (HLA-A/B/C) Loading Complex (PLC)", "definition": "ER membrane complex that loads cytosol-derived peptides (from proteasome) onto nascent MHC-I molecules before export to cell surface; consists of TAP1/TAP2 (ABC transporter, imports peptides into ER), TAPBP/Tapasin (bridges TAP to MHC-I; peptide editor), calreticulin, ERp57, calnexin; absent/dysfunctional in BLS-I — surface HLA-A/B/C reduced by >90%"},
    {"term": "Cryptosporidium Parvum — BLS-II PATHOGNOMONIC", "definition": "Obligate intracellular protozoan parasite; attaches to intestinal epithelium via MHC-II-dependent CD4+ T cell immunity required for clearance; in BLS-II: Cryptosporidium → persistent watery diarrhoea (>3 stools/day) + failure to thrive → if untreated → ascending cholangiopathy (sclerosing cholangitis-like) → biliary obstruction → liver failure; Cryptosporidium in an infant with CD4+ lymphopenia = BLS-II diagnosis until proven otherwise; also seen in other CD4+-deficient states (HIV, SCID) but NOT in BLS-I or ZAP-70 deficiency"},
    {"term": "RFXANK 33-nt Moroccan Founder Deletion", "definition": "In-frame 33-nucleotide deletion (c.169_201del, p.57_67del) removing 11 amino acids in ankyrin repeat 3 of RFXANK; present in >80% of BLS-IIB alleles in North African, Moroccan, and Middle Eastern populations; accounts for most of the worldwide excess of BLS-II in these populations; a founder effect from a common ancestor estimated hundreds of generations ago; rapid targeted allele testing (restriction enzyme or allele-specific PCR) should precede full sequencing in patients of Moroccan/North African/Arab descent"},
    {"term": "Necrotising Granulomatous Skin Lesions — BLS-I PATHOGNOMONIC", "definition": "Deep, non-infectious, ulcerating granulomata on midface (nose, periorbital), lower extremities, and trunk; histology: necrotising granulomatous inflammation with giant cells, WITHOUT identifiable microorganism; culture sterile; arise from dysregulated NK cell + CD8+ T cell activation due to missing MHC-I self-presentation; mimic Wegener's granulomatosis, Behçet's disease, pyoderma gangrenosum; PATHOGNOMONIC for BLS-I in context of low surface HLA-A/B/C; absent in BLS-II; skin biopsy + immunophenotyping distinguishes from vasculitis"},
    {"term": "ZAP-70 Selective CD8+ Absence", "definition": "In ZAP-70 deficiency: CD8+ T cells are absent (<1% of lymphocytes) because ZAP-70 is required for the strong TCR signal needed for CD8+ T cell positive selection in the thymus; CD4+ T cells are paradoxically present (numerically) but functionally anergic (ZAP-70 also needed for activation but CD4+ selection uses Lck/p56 with lower ZAP-70 threshold); the combination of CD4+ present + CD8+ absent + functional T cell unresponsiveness = ZAP-70 PATHOGNOMONIC; no other immunodeficiency produces absent CD8+ with present CD4+"},
    {"term": "RFX Transcription Factor Complex (RFX5/RFXANK/RFXAP)", "definition": "Trimeric transcription factor that binds the X1 box (GTNRCCNANYGC) of MHC-II gene promoters; RFX5 contains the DNA-binding domain (contacts X1 box); RFXANK (ankyrin repeats) and RFXAP (acidic domain) stabilise RFX5 binding; after RFX complex binds, NF-Y binds Y box, CREB binds W box → together recruit CIITA (the activating subunit); mutation in ANY RFX subunit OR CIITA → identical absent MHC-II phenotype (four complementation groups: BLS-IIA/B/C/D)"},
    {"term": "CIITA — Master MHC-II Transactivator", "definition": "Non-DNA-binding transcriptional coactivator; contains GTP-binding domain (required for nuclear localisation), acidic activation domain, proline/serine/threonine (PST) domain, and LRR domain; CIITA is recruited to MHC-II promoters through protein-protein contacts with the prebound RFX complex + NF-Y + CREB; CIITA itself has no DNA-binding activity (entirely dependent on RFX complex as DNA anchor); CIITA also activates invariant chain (CD74) and HLA-DM genes; CIITA expression is induced by IFN-γ (via STAT1 → GAS element in CIITA promoter IV)"},
    {"term": "Peptide Loading Complex (PLC) — TAP1/TAP2/TAPBP/Calreticulin", "definition": "ER membrane supercomplex that assembles nascent MHC-I chains (heavy chain + β2m) and loads 8-10mer peptides derived from proteasomal degradation; TAP1/TAP2 (ABC transporter): ATP-dependent import of peptides from cytosol to ER lumen; Tapasin (TAPBP): bridges TAP to MHC-I; edits peptide (exchanges low-affinity for high-affinity peptides); calreticulin + ERp57: chaperones that maintain MHC-I stability; without functional PLC, empty (peptide-receptive) MHC-I is unstable → degraded in ER → very low cell-surface MHC-I"},
    {"term": "CD4+ vs CD8+ Lymphopenia — BLS-I vs BLS-II DDx", "definition": "Most important bedside distinction: BLS-II (MHC-II deficiency) → CD4+ <200/µL, CD8+ preserved, HLA-DR absent on B cells → Cryptosporidium/Pneumocystis/candida infections; BLS-I (MHC-I deficiency) → CD8+ low-moderate, CD4+ preserved, HLA-A/B/C low → granulomatous skin + sinopulmonary bacteria; ZAP-70 → CD8+ absent, CD4+ present but non-functional → intermediate phenotype. Check HLA-DR on B cells (flow) and HLA-A/B/C on lymphocytes (flow) SIMULTANEOUSLY in any infant with unexplained lymphopenia"},
    {"term": "Bronchiectasis in BLS", "definition": "Progressive irreversible dilation of bronchi from recurrent infections: BLS-II → Pseudomonas/Haemophilus/Pneumocystis → bronchiectasis (onset childhood if untreated); BLS-I → Pseudomonas/gram-negatives/atypical mycobacteria → bronchiectasis (onset adolescence/adulthood, slower); HIGH-RESOLUTION CT (HRCT) of chest at diagnosis and annually; HSCT before bronchiectasis onset → better pulmonary outcome; established bronchiectasis may progress even post-HSCT if infection-damaged lung tissue"},
    {"term": "HSCT in BLS-II vs BLS-I", "definition": "BLS-II: HSCT is CURATIVE but outcomes historically worse than T-B-NK- SCID (partial thymic damage from years of absent MHC-II signalling); early HSCT (before age 2, before Cryptosporidium cholangiopathy) = best outcomes; HLA-matched sibling > matched unrelated > haploidentical; post-HSCT: MHC-II expression restores on donor APCs but thymus-dependent CD4+ reconstitution takes 12-24 months; BLS-I: HSCT less clearly indicated for mild phenotype; bronchiectasis management + prophylactic antibiotics are first-line for mild cases; HSCT for severe/progressive sinopulmonary disease"},
    {"term": "Mennonite ZAP70 Founder Allele p.Cys319Arg", "definition": "p.Cys319Arg (c.955T>C) — missense in the SH2 domain of ZAP-70 disrupting a critical cysteine that coordinates a structural zinc atom; first reported in Mennonite patients from a Canadian Mennonite community; also found in Old Order Amish; if a patient with CD8+ lymphopenia is of Mennonite/Old Order Amish ancestry → ZAP70 p.Cys319Arg allele-specific testing FIRST before full NGS; Cys319 in the interdomain B region between the two SH2 domains of ZAP-70; analogous cysteine conserved across all vertebrate ZAP-70"},
]


def _gene_cohort(gene: str, seed: int) -> dict:
    rng = random.Random(seed)
    n = 40
    meta = GENE_META[gene]

    if gene == "CIITA":
        mean_onset_months = rng.gauss(4, 2)
        mean_dx_lag = rng.gauss(2.5, 1.0)
        cd4_lymphopenia_n      = rng.randint(38, 40)   # near-universal: CD4+ <200/µL
        hla_dr_absent_n        = rng.randint(39, 40)   # diagnostic: >95% absent HLA-DR on B cells
        cryptosporidium_n      = rng.randint(24, 32)   # 60-80%
        pneumocystis_n         = rng.randint(18, 26)   # 45-65%
        bronchiectasis_n       = rng.randint(20, 28)   # 50-70% at presentation
        hypogamm_n             = rng.randint(32, 38)   # 80-95%
        hsct_n                 = rng.randint(22, 32)   # 55-80%
        outcome_good_n         = rng.randint(16, 24)   # good outcome with early HSCT
        pathognomonic = "BLS-IIA CIITA MASTER TRANSACTIVATOR: absent HLA-DR on B cells (flow: anti-HLA-DR NEGATIVE on CD19+ cells) = diagnostic; CD4+ <200/µL with CD8+ preserved; CRYPTOSPORIDIUM PARVUM gut infection in a child with CD4+ lymphopenia = BLS-II until proven otherwise; Cryptosporidium cholangiopathy → liver failure if untreated; HSCT CURATIVE: perform early (before age 2) before irreversible thymic damage + Cryptosporidium cholangiopathy; CIITA is BLS-IIA and the master MHC-II transcriptional activator (no DNA binding itself)"

    elif gene == "RFXANK":
        mean_onset_months = rng.gauss(4, 2)
        mean_dx_lag = rng.gauss(2.8, 1.2)
        cd4_lymphopenia_n      = rng.randint(38, 40)
        hla_dr_absent_n        = rng.randint(39, 40)
        cryptosporidium_n      = rng.randint(26, 34)   # slightly higher in Moroccan patients (higher exposure)
        pneumocystis_n         = rng.randint(18, 26)
        bronchiectasis_n       = rng.randint(20, 28)
        hypogamm_n             = rng.randint(32, 38)
        hsct_n                 = rng.randint(22, 32)
        outcome_good_n         = rng.randint(16, 24)
        pathognomonic = "MOROCCAN/NORTH AFRICAN FOUNDER DELETION PATHOGNOMONIC: RFXANK c.169_201del (33nt in-frame deletion, p.57_67del, ankyrin repeat 3) accounts for >80% of RFXANK alleles in Moroccan, North African, Arab patients; rapid targeted allele-specific test (PCR + restriction enzyme BstYI) BEFORE full sequencing if patient is of Moroccan/North African/Arab descent; BLS-IIB = most common BLS-II complementation group worldwide (due to Moroccan founder effect); phenotype IDENTICAL to CIITA/RFX5/RFXAP — complementation group only distinguished by genetics"

    elif gene == "RFX5":
        mean_onset_months = rng.gauss(5, 2.5)
        mean_dx_lag = rng.gauss(3.0, 1.5)
        cd4_lymphopenia_n      = rng.randint(37, 40)
        hla_dr_absent_n        = rng.randint(39, 40)
        cryptosporidium_n      = rng.randint(22, 30)
        pneumocystis_n         = rng.randint(16, 24)
        bronchiectasis_n       = rng.randint(18, 26)
        hypogamm_n             = rng.randint(30, 38)
        hsct_n                 = rng.randint(20, 30)
        outcome_good_n         = rng.randint(15, 23)
        pathognomonic = "BLS-IIC RFX5 DNA-BINDING SUBUNIT: RFX5 is the ONLY RFX subunit that directly contacts the X1 box DNA; RFX5 LOF → RFX trimer cannot assemble on MHC-II promoter → CIITA not recruited → no MHC-II; RFX5 mutations occur across ALL ETHNIC GROUPS (unlike RFXANK which is North African-enriched); compound heterozygotes common; no single founder mutation; RFX5 BLS-IIC accounts for ~10-15% of BLS-II; phenotype identical to other complementation groups; NGS mandatory (no targeted founder test); HLA-DR flow cytometry confirms diagnosis"

    elif gene == "RFXAP":
        mean_onset_months = rng.gauss(5, 2.5)
        mean_dx_lag = rng.gauss(3.5, 1.5)
        cd4_lymphopenia_n      = rng.randint(36, 40)   # slightly variable — fewer patients
        hla_dr_absent_n        = rng.randint(38, 40)
        cryptosporidium_n      = rng.randint(20, 28)
        pneumocystis_n         = rng.randint(15, 22)
        bronchiectasis_n       = rng.randint(17, 24)
        hypogamm_n             = rng.randint(28, 36)
        hsct_n                 = rng.randint(18, 26)
        outcome_good_n         = rng.randint(14, 20)
        pathognomonic = "RAREST BLS-II COMPLEMENTATION GROUP (BLS-IID): RFXAP (<5% of BLS-II, <50 patients worldwide 2026); RFXAP contains a unique acidic repeat N-terminal domain not found in any other known protein; RFXAP stabilises RFX5 X1-box binding affinity; RFXAP LOF → same absent MHC-II as other BLS-II groups; diagnosis: when BLS-II phenotype (absent HLA-DR, CD4+ lymphopenia) confirmed but RFXANK (most common) and RFX5 (second) negative → RFXAP sequencing; ultra-rare = INDIVIDUAL KINDREDS worldwide; may benefit from gene therapy in future"

    elif gene == "TAP1":
        mean_onset_months = rng.gauss(36, 12)          # LATER onset than BLS-II: childhood/adolescence
        mean_dx_lag = rng.gauss(6.0, 3.0)              # long diagnostic delay: skin lesions misdiagnosed
        cd4_lymphopenia_n      = rng.randint(2, 6)     # CD4+ PRESERVED in BLS-I!
        hla_abc_low_n          = rng.randint(37, 40)   # near-universal: HLA-A/B/C <5% normal expression
        skin_granulomas_n      = rng.randint(28, 36)   # 70-90%: necrotising granulomatous lesions
        sinopulmonary_n        = rng.randint(32, 38)   # 80-95%
        bronchiectasis_n       = rng.randint(20, 28)
        behcet_like_n          = rng.randint(16, 24)   # 40-60%: Behçet-like vasculitis
        hsct_n                 = rng.randint(10, 18)   # less common — milder cases managed conservatively
        outcome_good_n         = rng.randint(20, 28)   # outcome relatively good with antibiotics
        pathognomonic = "NECROTISING GRANULOMATOUS SKIN LESIONS PATHOGNOMONIC for BLS-I: deep, sterile, ulcerating cutaneous granulomata on midface + lower extremities; histology = necrotising granulomas NO bacteria (culture sterile); MISDIAGNOSED as Wegener's granulomatosis, Behçet's, pyoderma gangrenosum; KEY CLUE: skin biopsy shows necrotising granulomas + immunophenotyping shows absent HLA-A/B/C = BLS-I TAP1/TAP2/TAPBP; HLA-A/B/C on lymphocytes (flow: pan-HLA-I antibody) near-absent = DIAGNOSTIC; ONSET LATER (childhood) than BLS-II (infancy) — important DDx with BLS-II"

    elif gene == "TAP2":
        mean_onset_months = rng.gauss(30, 10)
        mean_dx_lag = rng.gauss(5.5, 2.5)
        cd4_lymphopenia_n      = rng.randint(2, 6)     # CD4+ preserved
        hla_abc_low_n          = rng.randint(37, 40)
        skin_granulomas_n      = rng.randint(28, 36)
        sinopulmonary_n        = rng.randint(32, 38)
        bronchiectasis_n       = rng.randint(22, 30)   # TAP2 slightly more severe
        behcet_like_n          = rng.randint(18, 26)
        hsct_n                 = rng.randint(12, 20)
        outcome_good_n         = rng.randint(20, 28)
        pathognomonic = "MOST COMMON BLS-I GENE (TAP2 ~70% of BLS-I cases): TAP2 forms the primary ATP-hydrolysis subunit of the TAP1/2 heterodimer; TAP2 mutations span entire gene — no ethnic founder enrichment; TAP1 and TAP2 co-localise at 6p21.32 (MHC region, ~200kb apart); CD8+ lymphopenia MILD-MODERATE (not absent as in ZAP-70); DISTINGUISHES from ZAP-70: ZAP-70 has CD8+ ABSENT + CD4+ present but non-functional; BLS-I TAP2 has CD8+ low-moderate + CD4+ fully functional + skin granulomas; sinopulmonary gram-negative infections (Pseudomonas, Haemophilus, MAC) dominant feature"

    elif gene == "TAPBP":
        mean_onset_months = rng.gauss(48, 18)          # latest onset: sometimes adult
        mean_dx_lag = rng.gauss(8.0, 4.0)              # very long diagnostic delay
        cd4_lymphopenia_n      = rng.randint(1, 4)     # CD4+ preserved
        hla_abc_low_n          = rng.randint(28, 36)   # less severe MHC-I reduction than TAP1/TAP2
        skin_granulomas_n      = rng.randint(18, 26)   # 45-65%: milder/fewer lesions
        sinopulmonary_n        = rng.randint(26, 34)
        bronchiectasis_n       = rng.randint(16, 24)   # slower progression
        behcet_like_n          = rng.randint(10, 18)
        hsct_n                 = rng.randint(6, 12)    # less common — milder phenotype
        outcome_good_n         = rng.randint(22, 30)   # best outcome of BLS-I (partial peptide loading retained)
        pathognomonic = "RAREST BLS-I AND MILDEST PHENOTYPE: TAPBP/tapasin is the MHC-I peptide editor (exchanges low-affinity → high-affinity peptides); partial tapasin-independent MHC-I loading persists → less severe MHC-I reduction than TAP1/TAP2; HLA-A/B/C reduced but not absent (vs near-absent in TAP1/TAP2); VERY RARE (<15 patients reported 2026); may present in adulthood; skin granulomata present but less florid; TAP1+TAPBP compound defect reported (adjacent genes at 6p21.32); co-encoded within MHC class II region — consider large MHC deletions in BLS-I phenotype"

    else:  # ZAP70
        mean_onset_months = rng.gauss(3, 1.5)          # infancy: recurrent infections 2-6 months
        mean_dx_lag = rng.gauss(2.0, 1.0)
        cd4_lymphopenia_n      = rng.randint(4, 10)    # CD4+ NUMERICALLY PRESENT (but functionally anergic)
        cd8_absent_n           = rng.randint(38, 40)   # CD8+ ABSENT: near-universal in ZAP70
        hla_dr_absent_n        = rng.randint(1, 3)     # HLA-DR expressed normally
        hla_abc_low_n          = rng.randint(1, 3)     # HLA-A/B/C normal
        pneumocystis_n         = rng.randint(18, 26)   # functional CD4+ defect → PCP susceptibility
        mennonite_allele_n     = rng.randint(12, 18)   # ~35-45% carry Mennonite founder allele
        autoimmune_n           = rng.randint(14, 22)   # 35-55%: autoimmune features from CD4+ dysregulation
        hsct_n                 = rng.randint(20, 28)   # most undergo HSCT
        outcome_good_n         = rng.randint(20, 26)
        pathognomonic = "SELECTIVE CD8+ ABSENCE WITH CD4+ PRESENT: CD3+CD8+ near-absent (<1% of lymphocytes) WITH CD3+CD4+ numerically present = ZAP-70 PATHOGNOMONIC — NO OTHER IMMUNODEFICIENCY produces this; CD4+ cells cannot activate (proliferation assay fails to PHA) despite being numerically present; MENNONITE/OLD ORDER AMISH: p.Cys319Arg founder → ZAP70 targeted sequencing FIRST if ancestry known; CD8+ absence = thymic positive selection failure (ZAP-70 required for CD8+ selection signal); autoimmune features from dysregulated CD4+ cells lacking CD8+ counter-regulation; HSCT curative and indicated for most"

    # Assign BLS-type specific metrics
    if gene in ("TAP1", "TAP2", "TAPBP"):
        # BLS-I metrics
        return {
            "gene": gene,
            "bls_type": "BLS-I (MHC Class I Deficiency)",
            "n": n,
            "locus": meta["locus"],
            "aa": meta["aa"],
            "kda": meta["kda"],
            "inheritance": meta["inh"],
            "syndrome": meta["syndrome"],
            "mean_onset_months": round(abs(mean_onset_months), 1),
            "mean_dx_lag_years": round(abs(mean_dx_lag), 2),
            "cd4_lymphopenia_n": cd4_lymphopenia_n,
            "cd4_lymphopenia_pct": round(cd4_lymphopenia_n / n * 100, 1),
            "hla_abc_low_n": hla_abc_low_n,
            "hla_abc_low_pct": round(hla_abc_low_n / n * 100, 1),
            "skin_granulomas_n": skin_granulomas_n,
            "skin_granulomas_pct": round(skin_granulomas_n / n * 100, 1),
            "sinopulmonary_n": sinopulmonary_n,
            "sinopulmonary_pct": round(sinopulmonary_n / n * 100, 1),
            "bronchiectasis_n": bronchiectasis_n,
            "bronchiectasis_pct": round(bronchiectasis_n / n * 100, 1),
            "behcet_like_n": behcet_like_n,
            "behcet_like_pct": round(behcet_like_n / n * 100, 1),
            "hsct_n": hsct_n,
            "hsct_pct": round(hsct_n / n * 100, 1),
            "outcome_good_n": outcome_good_n,
            "outcome_good_pct": round(outcome_good_n / n * 100, 1),
            "pathognomonic": pathognomonic,
            "mim_gene": meta["mim_gene"],
            "mim_disease": meta["mim_disease"],
            "seed": seed,
        }
    elif gene == "ZAP70":
        return {
            "gene": gene,
            "bls_type": "ZAP-70 CID (Selective CD8+ Absence)",
            "n": n,
            "locus": meta["locus"],
            "aa": meta["aa"],
            "kda": meta["kda"],
            "inheritance": meta["inh"],
            "syndrome": meta["syndrome"],
            "mean_onset_months": round(abs(mean_onset_months), 1),
            "mean_dx_lag_years": round(abs(mean_dx_lag), 2),
            "cd4_lymphopenia_n": cd4_lymphopenia_n,
            "cd4_lymphopenia_pct": round(cd4_lymphopenia_n / n * 100, 1),
            "cd8_absent_n": cd8_absent_n,
            "cd8_absent_pct": round(cd8_absent_n / n * 100, 1),
            "hla_dr_absent_n": hla_dr_absent_n,
            "hla_dr_absent_pct": round(hla_dr_absent_n / n * 100, 1),
            "pneumocystis_n": pneumocystis_n,
            "pneumocystis_pct": round(pneumocystis_n / n * 100, 1),
            "mennonite_allele_n": mennonite_allele_n,
            "mennonite_allele_pct": round(mennonite_allele_n / n * 100, 1),
            "autoimmune_n": autoimmune_n,
            "autoimmune_pct": round(autoimmune_n / n * 100, 1),
            "hsct_n": hsct_n,
            "hsct_pct": round(hsct_n / n * 100, 1),
            "outcome_good_n": outcome_good_n,
            "outcome_good_pct": round(outcome_good_n / n * 100, 1),
            "pathognomonic": pathognomonic,
            "mim_gene": meta["mim_gene"],
            "mim_disease": meta["mim_disease"],
            "seed": seed,
        }
    else:
        # BLS-II metrics
        return {
            "gene": gene,
            "bls_type": "BLS-II (MHC Class II Deficiency)",
            "n": n,
            "locus": meta["locus"],
            "aa": meta["aa"],
            "kda": meta["kda"],
            "inheritance": meta["inh"],
            "syndrome": meta["syndrome"],
            "mean_onset_months": round(abs(mean_onset_months), 1),
            "mean_dx_lag_years": round(abs(mean_dx_lag), 2),
            "cd4_lymphopenia_n": cd4_lymphopenia_n,
            "cd4_lymphopenia_pct": round(cd4_lymphopenia_n / n * 100, 1),
            "hla_dr_absent_n": hla_dr_absent_n,
            "hla_dr_absent_pct": round(hla_dr_absent_n / n * 100, 1),
            "cryptosporidium_n": cryptosporidium_n,
            "cryptosporidium_pct": round(cryptosporidium_n / n * 100, 1),
            "pneumocystis_n": pneumocystis_n,
            "pneumocystis_pct": round(pneumocystis_n / n * 100, 1),
            "bronchiectasis_n": bronchiectasis_n,
            "bronchiectasis_pct": round(bronchiectasis_n / n * 100, 1),
            "hypogamm_n": hypogamm_n,
            "hypogamm_pct": round(hypogamm_n / n * 100, 1),
            "hsct_n": hsct_n,
            "hsct_pct": round(hsct_n / n * 100, 1),
            "outcome_good_n": outcome_good_n,
            "outcome_good_pct": round(outcome_good_n / n * 100, 1),
            "pathognomonic": pathognomonic,
            "mim_gene": meta["mim_gene"],
            "mim_disease": meta["mim_disease"],
            "seed": seed,
        }


def generate_overview() -> dict:
    cohorts = [_gene_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    total_n = sum(c["n"] for c in cohorts)
    total_hsct = sum(c["hsct_n"] for c in cohorts)
    bls2_cohorts = [c for c in cohorts if c["bls_type"].startswith("BLS-II")]
    bls1_cohorts = [c for c in cohorts if c["bls_type"].startswith("BLS-I")]
    zap_cohorts  = [c for c in cohorts if "ZAP-70" in c["bls_type"]]
    return {
        "atlas": "Hereditary-MHC-Bare-Lymphocyte-Syndrome-Atlas",
        "subtitle": (
            "MHC Class II Deficiency / MHC Class I Deficiency / ZAP70 CID — "
            "CIITA-RFXANK-RFX5-RFXAP-TAP1-TAP2-TAPBP-ZAP70 "
            "(IUIS 2022 Category 2: Combined T and B Cell Immunodeficiencies — Antigen Presentation Defects)"
        ),
        "total_patients": total_n,
        "genes_covered": GENES,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_stats": {
            "total_hsct_pct": round(total_hsct / total_n * 100, 1),
            "bls2_genes": ["CIITA", "RFXANK", "RFX5", "RFXAP"],
            "bls1_genes": ["TAP1", "TAP2", "TAPBP"],
            "cid_genes": ["ZAP70"],
            "bls2_n": sum(c["n"] for c in bls2_cohorts),
            "bls1_n": sum(c["n"] for c in bls1_cohorts),
            "zap_n": sum(c["n"] for c in zap_cohorts),
            "bls2_cd4_lymphopenia_pct": round(
                sum(c["cd4_lymphopenia_n"] for c in bls2_cohorts) /
                max(sum(c["n"] for c in bls2_cohorts), 1) * 100, 1),
            "bls2_cryptosporidium_pct": round(
                sum(c.get("cryptosporidium_n", 0) for c in bls2_cohorts) /
                max(sum(c["n"] for c in bls2_cohorts), 1) * 100, 1),
            "bls1_skin_granulomas_pct": round(
                sum(c.get("skin_granulomas_n", 0) for c in bls1_cohorts) /
                max(sum(c["n"] for c in bls1_cohorts), 1) * 100, 1),
            "zap_cd8_absent_pct": round(
                sum(c.get("cd8_absent_n", 0) for c in zap_cohorts) /
                max(sum(c["n"] for c in zap_cohorts), 1) * 100, 1),
        },
        "cohort_summaries": [
            {k: v for k, v in c.items() if k != "pathognomonic"}
            for c in cohorts
        ],
        "cascade_testing_concepts": [
            "INFANT with RECURRENT INFECTIONS + LYMPHOPENIA: immunophenotyping (CD4+ vs CD8+ ratio) + HLA-DR on B cells + HLA-A/B/C on lymphocytes SIMULTANEOUSLY — this single panel distinguishes BLS-II (CD4+ low + HLA-DR absent), BLS-I (CD8+ low + HLA-A/B/C absent), and ZAP-70 (CD8+ absent + CD4+ present but non-functional)",
            "BLS-II (MHC CLASS II DEFICIENCY): CD4+ <200/µL + HLA-DR absent on B cells → gene panel: RFXANK FIRST if Moroccan/North African ancestry (founder p.57_67del covers >80%); otherwise parallel CIITA/RFXANK/RFX5/RFXAP NGS panel; all 4 genes cause IDENTICAL phenotype; HSCT URGENTLY — before Cryptosporidium cholangiopathy (liver failure)",
            "CRYPTOSPORIDIUM + CD4+ LYMPHOPENIA = BLS-II PATHOGNOMONIC: Cryptosporidium parvum in infant/child with CD4+ <200/µL + absent HLA-DR + negative HIV = BLS-II until proven otherwise; Cryptosporidium cholangiopathy (ascending biliary disease) → liver failure without HSCT; antiparasitics only temporise",
            "BLS-I (MHC CLASS I DEFICIENCY): CD8+ low-moderate + HLA-A/B/C absent + necrotising granulomatous skin lesions = BLS-I TAP1/TAP2 (TAP2 is most common); present in childhood/adolescence (not infancy); granulomas misdiagnosed as Wegener's or Behçet's → ALWAYS check HLA-A/B/C flow on lymphocytes in sterile granulomatous skin disease in children",
            "TAPBP/TAPASIN: rarest BLS-I (<15 patients); mildest BLS-I phenotype (partial tapasin-independent MHC-I loading persists); HLA-A/B/C reduced but not absent; onset in adolescence/adulthood; TAP1+TAPBP compound defects possible (adjacent at 6p21.32); consider large 6p21.32 deletions in BLS-I with adjacent gene involvement",
            "ZAP-70: CD3+CD8+ ABSENT + CD3+CD4+ PRESENT = pathognomonic; confirm with proliferation assay (CD4+ T cells fail PHA response); Mennonite/Old Order Amish ancestry → ZAP70 p.Cys319Arg targeted testing first; ZAP-70 deficiency → HSCT curative; autoimmune features from dysregulated CD4+ cells without CD8+ counter-regulation; Pneumocystis prophylaxis mandatory (functional CD4+ defect despite numerical preservation)",
            "ALL BLS TYPES: BCG ABSOLUTELY CONTRAINDICATED — same as SCID; live vaccines forbidden until immune reconstitution post-HSCT; IgG replacement (IVIG/SCIG) for hypogammaglobulinaemia in BLS-II and ZAP-70; prophylaxis: trimethoprim-sulfamethoxazole (Pneumocystis) in all three types",
            "DDx BLS-II vs SCID: SCID has BOTH CD4+ AND CD8+ absent + normal NK or absent NK; BLS-II: CD4+ absent, CD8+ PRESERVED, NK normal; HLA-DR absent on B cells = BLS-II (not SCID); BLS-I vs normal NK deficiency: BLS-I has HLA-A/B/C absent, NK normal or mildly affected; classic NK deficiency has NK cells absent",
        ],
        "iuis_category": "IUIS 2022 Category 2: Combined T and B Cell Immunodeficiencies — MHC Antigen Presentation Defects (BLS-I: MHC Class I; BLS-II: MHC Class II) + ZAP70 TCR Signalling Defect",
    }


def generate_breakdown() -> dict:
    cohorts = [_gene_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    return {
        "atlas": "Hereditary-MHC-Bare-Lymphocyte-Syndrome-Atlas",
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "gene_cohorts": cohorts,
        "diagnostic_groups": {
            "bls_ii_mhc_class_ii_deficiency": {
                "genes": ["CIITA", "RFXANK", "RFX5", "RFXAP"],
                "mechanism": (
                    "Absent MHC Class II (HLA-DR/DP/DQ) on APCs and thymic epithelium → "
                    "CD4+ T cell positive selection failure → CD4+ lymphopenia; "
                    "no Th help → hypogammaglobulinemia; Cryptosporidium/PCP susceptibility"
                ),
                "shared_feature": "Identical phenotype across all 4 complementation groups; distinguished ONLY by genetics",
                "key_ddx": "RFXANK (most common, Moroccan founder); CIITA (BLS-IIA); RFX5 (BLS-IIC, pan-ethnic); RFXAP (BLS-IID, rarest)",
                "complementation_groups": {
                    "BLS-IIA": "CIITA (1130aa, 16p13.13) — master transactivator, non-DNA-binding, ~20-25% of BLS-II",
                    "BLS-IIB": "RFXANK (260aa, 19q13.11) — ankyrin repeats, Moroccan founder del, MOST COMMON ~50-60%",
                    "BLS-IIC": "RFX5 (616aa, 1q21.3) — DNA-binding subunit, pan-ethnic, ~10-15%",
                    "BLS-IID": "RFXAP (272aa, 13q14.2) — acidic domain, RAREST <5%",
                },
            },
            "bls_i_mhc_class_i_deficiency": {
                "genes": ["TAP1", "TAP2", "TAPBP"],
                "mechanism": (
                    "Peptide loading complex (PLC) failure → empty unstable MHC-I on ER → low surface HLA-A/B/C → "
                    "mild-moderate CD8+ lymphopenia; NK dysregulation → sterile necrotising granulomata; "
                    "sinopulmonary bacteria (gram-negative + atypical mycobacteria)"
                ),
                "shared_feature": "Identical phenotype; distinguished by genetics; all three genes at 6p21.32 (MHC region)",
                "key_ddx": "TAP2 most common (~70%); TAP1 (~25%); TAPBP rarest and mildest (<15 patients); onset LATER than BLS-II (childhood/adolescence vs infancy)",
            },
            "zap70_tcr_signalling_defect": {
                "genes": ["ZAP70"],
                "mechanism": (
                    "ZAP-70 kinase required for TCR ITAM signalling → CD8+ T cell positive selection failure; "
                    "CD4+ T cells numerically present but functionally anergic; "
                    "selective CD8+ absence = unique phenotype"
                ),
                "shared_feature": "Unique phenotype: CD8+ absent + CD4+ present; Mennonite/Amish founder allele p.Cys319Arg",
                "key_ddx": "vs BLS-II: ZAP-70 has HLA-DR expressed normally; vs BLS-I: ZAP-70 has HLA-A/B/C normal; vs SCID: CD4+ present in ZAP-70",
            },
        },
        "pathognomonic_features": {
            "BLS_II_hla_dr_absent": (
                "HLA-DR ABSENT on B cells (flow: anti-HLA-DR antibody → negative staining on CD19+ cells) = "
                "BLS-II DIAGNOSTIC; any of the 4 BLS-II genes; CD4+ <200/µL with CD8+ preserved"
            ),
            "BLS_II_cryptosporidium": (
                "CRYPTOSPORIDIUM PARVUM in infant with CD4+ lymphopenia + absent HLA-DR = BLS-II PATHOGNOMONIC; "
                "ALSO: Cryptosporidium cholangiopathy → ascending biliary disease → cirrhosis = FATAL without HSCT"
            ),
            "RFXANK_moroccan_founder": (
                "RFXANK p.57_67del (33-nt deletion, ankyrin repeat 3) in patient of Moroccan/North African/Arab descent = "
                "BLS-IIB PATHOGNOMONIC; screen targeted before full panel"
            ),
            "BLS_I_skin_granulomas": (
                "NECROTISING GRANULOMATOUS SKIN LESIONS (sterile ulcers, midface + extremities) + absent HLA-A/B/C = "
                "BLS-I PATHOGNOMONIC; misdiagnosed as Wegener's/Behçet's; culture sterile; histology: granulomatous"
            ),
            "ZAP70_cd8_absent": (
                "CD8+ T CELLS ABSENT (<1% lymphocytes) + CD4+ PRESENT (numerically) = ZAP-70 PATHOGNOMONIC; "
                "NO OTHER PID produces this combination; confirm with proliferation assay (CD4+ fails PHA)"
            ),
            "ZAP70_mennonite": (
                "MENNONITE/OLD ORDER AMISH + selective CD8+ lymphopenia = ZAP70 p.Cys319Arg targeted testing FIRST; "
                "founder allele in Mennonite/Amish communities; ~35-45% carry this specific allele"
            ),
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-MHC-Bare-Lymphocyte-Syndrome-Atlas",
        "definitions": DEFINITIONS,
        "gene_metadata": GENE_META,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "key_distinguishers": {
            "BLS_II_vs_BLS_I_lymphopenia_type": (
                "BLS-II: CD4+ absent, CD8+ preserved, HLA-DR absent → Cryptosporidium; "
                "BLS-I: CD8+ low-moderate, CD4+ preserved, HLA-A/B/C absent → granulomatous skin; "
                "SINGLE MOST IMPORTANT DDx: is the lymphopenia CD4+ or CD8+?"
            ),
            "BLS_II_vs_SCID": (
                "SCID: both CD4+ AND CD8+ absent (T-B-NK+ or T-B+NK-); BLS-II: CD4+ absent, CD8+ PRESERVED; "
                "HLA-DR absent on B cells = BLS-II; SCID has present HLA-DR (MHC-II genes intact); "
                "onset similar (infancy) but SCID more severe"
            ),
            "ZAP70_vs_BLS_II_vs_BLS_I": (
                "ZAP-70: CD8+ absent + CD4+ present but non-functional + HLA-DR normal + HLA-A/B/C normal; "
                "BLS-II: CD4+ absent + HLA-DR absent; BLS-I: CD8+ low-moderate + HLA-A/B/C absent; "
                "key: HLA-DR and HLA-A/B/C are NORMAL in ZAP-70"
            ),
            "RFXANK_vs_CIITA_vs_RFX5_vs_RFXAP": (
                "All 4 BLS-II groups: IDENTICAL clinical phenotype — cannot distinguish at bedside; "
                "RFXANK most common (Moroccan founder) → targeted test if North African ancestry; "
                "CIITA: ~20-25% of BLS-II; RFX5: ~10-15%; RFXAP: <5%; "
                "diagnosis requires genetics (complementation group has no clinical correlate)"
            ),
            "TAP1_vs_TAP2_vs_TAPBP": (
                "All three BLS-I genes: similar phenotype; TAP2 most common (~70%); TAP1 (~25%); TAPBP rarest + mildest; "
                "TAPBP distinction: partial tapasin-independent MHC-I loading → less severe MHC-I reduction; "
                "onset: TAPBP latest (adolescence/adulthood); all at 6p21.32"
            ),
            "BLS_vs_HIV": (
                "HIV with CD4+ lymphopenia: HIV-positive serology + viral load; BLS-II: HIV-negative + absent HLA-DR; "
                "both can have Cryptosporidium; age matters: BLS-II presents in infancy, HIV-associated CD4+ loss in adults; "
                "HLA-DR flow cytometry is the discriminating test (absent in BLS-II, present in HIV)"
            ),
        },
        "diagnostic_pathway": [
            "Step 1: Any infant/child with recurrent infections + unexplained lymphopenia → FULL IMMUNOPHENOTYPING simultaneously: CD4+ count, CD8+ count, NK count, B cell count (CD19+), HLA-DR on B cells (flow: anti-HLA-DR), HLA-A/B/C on lymphocytes (flow: pan-HLA-I antibody) — this SINGLE panel distinguishes BLS-II, BLS-I, ZAP-70, and SCID",
            "Step 2: CD4+ low + HLA-DR absent on B cells → BLS-II confirmed → gene panel: IF Moroccan/North African/Arab ancestry → RFXANK p.57_67del targeted test FIRST (PCR/restriction enzyme); simultaneously send CIITA/RFXANK/RFX5/RFXAP NGS panel; start IVIG + PCP prophylaxis (TMP-SMX) + antifungal prophylaxis IMMEDIATELY; begin HSCT work-up without waiting for genetic result",
            "Step 3: CD8+ low-moderate + HLA-A/B/C absent (flow near-absent) → BLS-I confirmed → TAP1/TAP2/TAPBP NGS panel (TAP2 most likely); skin biopsy if granulomatous lesions present (necrotising granulomas without organisms confirms BLS-I); chest HRCT (bronchiectasis); HSCT work-up if progressive pulmonary disease; antibiotic prophylaxis (azithromycin MAC + TMP-SMX)",
            "Step 4: CD8+ absent (<1%) + CD4+ present + HLA-DR normal + HLA-A/B/C normal → ZAP-70 deficiency → ZAP70 sequencing (p.Cys319Arg if Mennonite/Amish); confirm with T cell proliferation assay (CD4+ fails PHA response); start HSCT work-up; PCP prophylaxis mandatory; IVIG if hypogammaglobulinemia",
            "Step 5: CRYPTOSPORIDIUM in any infant with CD4+ lymphopenia → URGENT BLS-II work-up; HLA-DR flow on B cells STAT; Cryptosporidium cholangiopathy → ERCP/liver biopsy; anti-cryptosporidial therapy (nitazoxanide + paromomycin) as bridge to HSCT; HSCT is the ONLY definitive treatment for Cryptosporidium in BLS-II",
            "Step 6: STERILE NECROTISING GRANULOMATOUS SKIN LESIONS in child/adolescent (NOT responding to antibiotics, culture-negative) → BLS-I work-up; HLA-A/B/C on lymphocytes (flow); TAP1/TAP2/TAPBP panel; punch biopsy confirms necrotising granulomas; consider large 6p21.32 deletion if multi-gene BLS-I (TAP1+TAPBP adjacent)",
            "Step 7: ALL TYPES — BCG ABSOLUTELY CONTRAINDICATED; live vaccines forbidden pre-HSCT; IVIG monthly (BLS-II: hypogammaglobulinemia; BLS-I mild: not always needed; ZAP-70: if hypogamma present); TMP-SMX for PCP prophylaxis; fluconazole antifungal prophylaxis (BLS-II)",
            "Step 8: POST-HSCT monitoring: BLS-II → HLA-DR expression on donor B cells (flow, monthly for 6 months post-HSCT); CD4+ reconstitution (thymus-dependent, takes 12-24 months); Cryptosporidium clearance; lung function; BLS-I → HLA-A/B/C expression restoration; skin lesion resolution; NK cell reconstitution",
        ],
    }
