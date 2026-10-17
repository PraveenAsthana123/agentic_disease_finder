#!/usr/bin/env python3
"""Hereditary-Follicular-Lymphoma-Predisposition-Atlas -- Complete 8-Gene Reference
CREBBP   (CREB-binding protein; 2442aa; 16p13.3; AD LOF;
          Rubinstein-Taybi Syndrome 1 (OMIM 180849);
          FL 60-70% somatic driver; HAT domain loss; mocetinostat HDACi;
          seed SEED_BASE+0).
EZH2     (EZH2 histone methyltransferase; 746aa; 7q36.1; AD GOF;
          Weaver Syndrome (OMIM 615502);
          FL 80-90% somatic Y641F/Y641N GOF; H3K27me3 hypermethylation;
          Tazemetostat FDA2020; seed SEED_BASE+1).
TNFRSF14 (HVEM; 283aa; 1p36.32; AD LOF;
          1p36 deletion syndrome (OMIM 607872);
          FL 40-68% somatic; BTLA/CD160 checkpoint loss; poor prognosis;
          seed SEED_BASE+2).
KMT2D    (MLL4 methyltransferase; 5537aa; 12q13.12; AD LOF;
          Kabuki Syndrome 1 (OMIM 147920);
          FL 60-80% somatic; H3K4me1 enhancer loss; B-cell lymphoma driver;
          tazemetostat + lenalidomide; seed SEED_BASE+3).
RRAGC    (RagC GTPase; 424aa; 1p34.3; AD GOF;
          No named syndrome — RASopathy-like mTOR;
          FL 15-17% relapsed/refractory; mTORC1 hyperactivation;
          Everolimus active; seed SEED_BASE+4).
MAP2K1   (MEK1; 393aa; 15q22.31; AD GOF;
          Cardio-facio-cutaneous Syndrome (OMIM 115150);
          FL 10-15% relapsed/transformed; RAS-MAPK pathway;
          Cobimetinib/Trametinib MEKi; seed SEED_BASE+5).
TP53     (p53; 393aa; 17p13.1; AD LOF;
          Li-Fraumeni Syndrome (OMIM 151623);
          FL→DLBCL histological transformation 40-50%; AVOID RADIATION ABSOLUTELY;
          WB-MRI Toronto; APR-246/venetoclax; seed SEED_BASE+6).
ATM      (ATM kinase; 3056aa; 11q22.3; AD LOF;
          Ataxia-Telangiectasia heterozygous (OMIM 208900);
          NHL 4x; FL 2-3x; del11q somatic 35%; RADIOSENSITIVITY ABSOLUTE;
          ceralasertib ATRi; seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3590-3597)
"""
import random

SEED_BASE = 3590

ATLAS_GENES = [
    {
        "gene": "CREBBP",
        "protein": (
            "CREBBP -- 16p13.3 AD LOF -- 2442aa -- "
            "KAT3A-CREB-Binding-Protein-265kDa-"
            "HAT-Domain-Bromodomain-KIX-Domain-"
            "RTS1-Rubinstein-Taybi-FL-60-70pct-Mocetinostat-OMIM-180849"
        ),
        "locus": "16p13.3",
        "protein_size": (
            "2442 aa / ~265 kDa / 16p13.3 CREBBP FL molecular context: "
            "STRUCTURE: "
            "  2442 aa / ~265 kDa; lysine acetyltransferase (KAT3A); "
            "  KIX domain (aa 586-672): CREB-binding, AML target; "
            "  Bromodomain (aa 1079-1196): recognises H3K27ac; "
            "  HAT catalytic domain (aa 1197-1459): acetylates H3K27; "
            "  SID domain: NCOA coactivator binding; "
            "  LOF: loss of H3K27 acetylation → gene silencing at enhancers; "
            "  Germline heterozygous LOF = RTS1; somatic biallelic = FL driver; "
            "FL MECHANISM: "
            "  CREBBP HAT LOF → H3K27ac loss at BCL-2/BCL-6 enhancers → "
            "  derepression of lymphoma-promoting genes; "
            "  60-70% of FL carry somatic CREBBP HAT mutations (exon 30-31 hotspot); "
            "  Germline CREBBP LOF (RTS1) → lymphoma predisposition — FL documented; "
            "  HAT mutations accelerate FL progression and transformation; "
            "CLINICAL RULES FOR CREBBP-FL: "
            "  Mocetinostat (HDAC1/2/3 inhibitor) restores H3K27ac; "
            "  Tazemetostat (EZH2i) synergistic — EZH2 overactivated when CREBBP LOF; "
            "  Azacitidine (DNMTi) + mocetinostat epigenetic combo active; "
            "  R-CHOP standard; epigenetic maintenance after CR emerging; "
            "  RTS1 features at lymphoma presentation: broad thumbs/toes, short stature, ID; "
            "  Panel testing for germline CREBBP when FL onset <40 yr; "
            "KEY VARIANTS: "
            "  HAT domain: Y1450C, K1455E, D1399G, E1441K (hotspot exon 30-31); "
            "  Truncating: frameshift/stopgain upstream of HAT; "
            "  KIX domain: R1446C (AML-associated); "
            "SURVEILLANCE: "
            "  Annual PET/CT post-CR; bone marrow biopsy at relapse; "
            "  Cascade testing first-degree relatives; "
            "  Screen for concurrent breast/prostate cancer (CREBBP germline); "
        ),
        "syndrome": "RTS1-Rubinstein-Taybi-Syndrome-1",
        "inheritance": "AD LOF heterozygous",
        "fl_risk": "FL 60-70% somatic HAT driver; germline predisposition; younger onset <40yr; transformation risk",
        "pathognomonic": (
            "FL + broad thumbs/toes + short stature + intellectual disability = RTS1 germline CREBBP; "
            "CREBBP somatic HAT mutation in 60-70% FL (IHC: H3K27ac loss in tumour cells); "
            "Somatic CREBBP detected in FL → test germline in young patients (<40 yr)"
        ),
        "key_avoid": (
            "Standard R-CHOP alone insufficient for CREBBP-FL (HAT LOF drives epigenetic progression); "
            "Avoid HDACi class I/II non-selectively (toxicity); use HDAC1/2 selective mocetinostat; "
            "Avoid withholding epigenetic therapy at relapse"
        ),
        "key_rule": (
            "Mocetinostat (HDAC1/2/3) restores H3K27ac in CREBBP LOF FL — clinical trial priority; "
            "Tazemetostat synergistic (CREBBP LOF → EZH2 dominant → tazemetostat active); "
            "Azacitidine + lenalidomide + rituximab (R2-aza) for relapsed CREBBP-FL; "
            "Germline CREBBP: RTS1 multi-system surveillance; broad-thumbs sign at clinical exam"
        ),
        "surveillance": (
            "Annual PET/CT post-CR; BM biopsy at relapse; "
            "Germline panel if FL onset <40 yr or family history lymphoma; "
            "RTS1 co-features: ID assessment, cardiac evaluation (30%), broad thumbs/toes sign; "
            "CREBBP somatic → germline testing of proband + first-degree relatives"
        ),
        "targeted_rx": (
            "Mocetinostat HDAC1/2/3i (Phase 2 active); tazemetostat EZH2i FDA2020 synergistic; "
            "Azacitidine + lenalidomide + rituximab; R-CHOP ± obinutuzumab; "
            "APR-246 if concurrent TP53 co-mutation; "
            "epigenetic maintenance post-CR emerging evidence"
        ),
    },
    {
        "gene": "EZH2",
        "protein": (
            "EZH2 -- 7q36.1 AD GOF -- 746aa -- "
            "KMT6-PRC2-Catalytic-Subunit-85kDa-"
            "SET-Domain-H3K27me3-Silencer-"
            "Weaver-Syndrome-FL-80-90pct-Tazemetostat-FDA2020-OMIM-615502"
        ),
        "locus": "7q36.1",
        "protein_size": (
            "746 aa / ~85 kDa / 7q36.1 EZH2 FL molecular context: "
            "STRUCTURE: "
            "  746 aa / ~85 kDa; catalytic subunit of PRC2 complex; "
            "  SET domain (aa 612-726): H3K27 trimethylase; "
            "  Y641 residue (aa 641): substrate specificity switch — GOF shifts activity to H3K27me3; "
            "  WD repeat binding: SUZ12/EED interaction; "
            "  EED binding domain: PRC2 assembly; "
            "  GOF Y641F/Y641N/Y641S/Y641H: enhanced H3K27me3 → gene silencing; "
            "  A677G/A687V: alternative GOF hotspots; "
            "FL GOF MECHANISM: "
            "  EZH2 Y641 GOF → excess H3K27me3 → silencing of Polycomb target genes "
            "  (CDKN2A, SMAD1, IRF4, BCL-6 regulators) → B-cell differentiation block; "
            "  80-90% of FL Grade 1-3A carry EZH2 Y641 somatic GOF; "
            "  Germline EZH2 (Weaver) → lymphoma predisposition documented; "
            "  EZH2 GOF + CREBBP LOF = most common combination (50% FL); "
            "TAZEMETOSTAT FDA2020: "
            "  First EZH2 inhibitor FDA approved for relapsed/refractory FL with EZH2 mutation; "
            "  Tazemetostat 800 mg BID; objective response 69% (EZH2-mutant); "
            "  Response 34% in EZH2-WT FL (PRC2-dependent); "
            "  Weaver germline: tazemetostat first-line consideration; "
            "KEY VARIANTS: "
            "  Y641F (most common 50%), Y641N (30%), Y641S, Y641H, Y641C; "
            "  A677G (~8% FL), A687V (~3% FL); "
            "  Weaver germline: c.553C>T (R185C), p.Arg685His, truncating NSD; "
            "SURVEILLANCE: "
            "  Annual PET/CT; EZH2 mutation testing at diagnosis mandatory for FL; "
            "  EZH2 WT FL → still consider tazemetostat (PRC2-dependent effect); "
        ),
        "syndrome": "Weaver-Syndrome-EZH2-Overgrowth",
        "inheritance": "AD GOF heterozygous",
        "fl_risk": "FL 80-90% somatic Y641 GOF; Grade 1-3A MOST COMMON FL driver; tazemetostat FDA2020 approved",
        "pathognomonic": (
            "EZH2 Y641/A677/A687 mutation in FL = MOST COMMON somatic mutation 80-90%; "
            "Tazemetostat FDA2020 ONLY indicated for EZH2-mutant FL (ORR 69%); "
            "Weaver germline: overgrowth + facial dysmorphology + FL presentation = germline EZH2 test"
        ),
        "key_avoid": (
            "Withholding EZH2 mutation testing at FL diagnosis (changes treatment decision); "
            "Tazemetostat in EZH2-WT FL: lower ORR 34% (still some activity via PRC2 dependence); "
            "Avoid conventional therapy alone without adding tazemetostat in EZH2 GOF relapsed FL"
        ),
        "key_rule": (
            "EZH2 mutation test MANDATORY at FL diagnosis (impacts treatment choice); "
            "Tazemetostat 800 mg BID FDA2020 standard for relapsed/refractory EZH2-mutant FL; "
            "Obinutuzumab + tazemetostat (SYMPHONY trial) frontline emerging; "
            "Weaver germline: multi-system surveillance (overgrowth, cardiac, joints, lymphoma)"
        ),
        "surveillance": (
            "EZH2 mutation testing at diagnosis mandatory; annual PET/CT post-CR; "
            "Weaver features: overgrowth assessment, facial dysmorphology, joint laxity; "
            "Germline EZH2 if FL onset <35 yr or Weaver features; "
            "BM biopsy at progression to assess grade transformation"
        ),
        "targeted_rx": (
            "Tazemetostat 800 mg BID FDA2020 (EZH2-mutant relapsed/refractory FL); "
            "Obinutuzumab + tazemetostat SYMPHONY frontline; "
            "R-CHOP + tazemetostat maintenance; "
            "Venetoclax + tazemetostat emerging (BCL-2 co-dysregulation)"
        ),
    },
    {
        "gene": "TNFRSF14",
        "protein": (
            "TNFRSF14 -- 1p36.32 AD LOF -- 283aa -- "
            "HVEM-LIGHTR-TNF-Receptor-31kDa-"
            "Cysteine-Rich-Domains-BTLA-CD160-BTLA-Checkpoint-"
            "FL-40-68pct-Poor-Prognosis-OMIM-607872"
        ),
        "locus": "1p36.32",
        "protein_size": (
            "283 aa / ~31 kDa / 1p36.32 TNFRSF14 FL molecular context: "
            "STRUCTURE: "
            "  283 aa / ~31 kDa; TNF receptor superfamily member 14; "
            "  Herpesvirus entry mediator (HVEM); "
            "  3 cysteine-rich domains (CRD1-3) in extracellular region; "
            "  CRD1: BTLA ligand binding; CRD2-3: LIGHT/LTα binding; "
            "  Transmembrane + short cytoplasmic tail; "
            "  Signals via TRAF2/5 (pro-survival) and BTLA (inhibitory checkpoint); "
            "  LOF: loss of BTLA/CD160 inhibitory signalling → immune evasion; "
            "FL LOF MECHANISM: "
            "  TNFRSF14 LOF in FL → loss of BTLA-mediated T-cell inhibition → "
            "  tumour microenvironment (TME) T-cell dysfunction; "
            "  40-68% somatic biallelic LOF in FL; "
            "  1p36 deletion (monosomy 1p36): TNFRSF14 in minimal deleted region; "
            "  Associated with poor prognosis, higher transformation risk; "
            "  CD4+PD-1+ TFH cells accumulate in TNFRSF14-LOF FL TME; "
            "CLINICAL RULES: "
            "  Obinutuzumab preferred over rituximab (enhanced ADCC in TME-disrupted FL); "
            "  PD-1/PD-L1 checkpoint blockade rationale (nivolumab/pembrolizumab); "
            "  FLIPI2 score: 1p36/TNFRSF14 LOF → higher FLIPI2 category; "
            "  Lenalidomide + rituximab (R2): restores TME (lenalidomide modulates T-cell function); "
            "KEY VARIANTS: "
            "  CRD1/BTLA-binding: R54C, N62D, Y60C; "
            "  Truncating: most common in FL (frameshift/stopgain); "
            "  1p36 monosomy detected by FISH (large deletion > single nucleotide); "
            "SURVEILLANCE: "
            "  Annual PET/CT; FISH 1p36 at diagnosis; FLIPI2 calculation mandatory; "
            "  BM trephine at diagnosis (BM involvement 60-70% FL); "
            "  Transformation surveillance PET/CT q6M if FLIPI2 high; "
        ),
        "syndrome": "1p36-Deletion-Syndrome-TNFRSF14-LOF",
        "inheritance": "AD LOF heterozygous / 1p36 monosomy",
        "fl_risk": "FL 40-68% somatic HIGHEST somatic FL driver; 1p36 deletion; poor prognosis; transformation risk",
        "pathognomonic": (
            "1p36 FISH deletion in FL + TNFRSF14 LOF IHC/sequencing = HIGHEST-RISK FL (poor prognosis); "
            "CD4+PD-1+ TFH cell accumulation in FL TME = TNFRSF14 LOF signature; "
            "FLIPI2 ≥3 + 1p36 deletion = obinutuzumab preferred over rituximab"
        ),
        "key_avoid": (
            "Rituximab monotherapy maintenance (inferior to obinutuzumab in TNFRSF14-LOF FL); "
            "Delay in FISH 1p36 testing at diagnosis (changes prognosis + treatment intensity); "
            "Withholding lenalidomide maintenance (R2 active in TNFRSF14-LOF TME)"
        ),
        "key_rule": (
            "TNFRSF14 LOF = HIGHEST-RISK FL marker → obinutuzumab preferred over rituximab; "
            "1p36 FISH mandatory at diagnosis; FLIPI2 calculation mandatory; "
            "Lenalidomide + rituximab R2 active (TME T-cell restoration); "
            "PD-1 checkpoint blockade rationale (nivolumab FL trial CHECKMATE-140 active)"
        ),
        "surveillance": (
            "1p36 FISH at diagnosis mandatory; FLIPI2 calculation; annual PET/CT; "
            "BM trephine at diagnosis; transformation screening PET/CT q6M if FLIPI2 ≥3; "
            "Cascade testing family if germline 1p36 deletion syndrome features"
        ),
        "targeted_rx": (
            "Obinutuzumab + CHOP (G-CHOP) preferred in TNFRSF14-LOF high-risk FL; "
            "Lenalidomide + rituximab (R2) maintenance; "
            "PD-1 inhibitor nivolumab/pembrolizumab (CHECKMATE-140 FL cohort); "
            "HSCT consolidation for transformed FL"
        ),
    },
    {
        "gene": "KMT2D",
        "protein": (
            "KMT2D -- 12q13.12 AD LOF -- 5537aa -- "
            "MLL4-COMPASS-LIKE-Methyltransferase-500kDa-"
            "H3K4me1-Enhancer-Methylation-SET-Domain-"
            "Kabuki1-Syndrome-FL-60-80pct-Tazemetostat-OMIM-147920"
        ),
        "locus": "12q13.12",
        "protein_size": (
            "5537 aa / ~500 kDa / 12q13.12 KMT2D FL molecular context: "
            "STRUCTURE: "
            "  5537 aa / ~500 kDa; lysine methyltransferase 2D (MLL4); "
            "  SET domain (C-terminal): H3K4 monomethylation at enhancers; "
            "  PHD fingers: histone binding; LXXLL motif: nuclear receptor interaction; "
            "  COMPASS-like complex: KMT2D + KDM6A + NCOA6 + PAGR1 + ASH2L + WDR5; "
            "  LOF: H3K4me1 loss at active enhancers → B-cell gene expression failure; "
            "  Germline heterozygous LOF = Kabuki Syndrome 1; "
            "FL LOF MECHANISM: "
            "  KMT2D LOF → H3K4me1 loss at BCL-6/BLIMP-1/IRF4 enhancers → "
            "  B-cell differentiation block at germinal centre stage → FL; "
            "  60-80% of FL carry somatic KMT2D LOF (most commonly truncating); "
            "  Germline KMT2D (Kabuki1) → FL/DLBCL predisposition documented; "
            "  KMT2D + CREBBP combined = 50-60% of all FL (dual epigenetic hit); "
            "CLINICAL RULES: "
            "  Tazemetostat active (KMT2D LOF → EZH2 unopposed → H3K27me3 excess); "
            "  Azacitidine + lenalidomide + rituximab (R2-aza) for KMT2D-FL; "
            "  Kabuki features at FL presentation: short stature, broad nasal tip, "
            "  persistent fingertip pads, intellectual disability, heart defects (30%); "
            "  Germline panel if FL onset <40 yr or Kabuki features; "
            "KEY VARIANTS: "
            "  Truncating: frameshift/stopgain throughout gene (most common); "
            "  SET domain: missense A4765T, Y4751N; "
            "  PHD fingers: missense; "
            "SURVEILLANCE: "
            "  Annual PET/CT post-CR; BM biopsy at relapse; "
            "  Kabuki multi-system: cardiac (annual echo), growth chart, renal USS; "
            "  Germline testing: family cascade; "
        ),
        "syndrome": "Kabuki-Syndrome-1-KMT2D-LOF",
        "inheritance": "AD LOF heterozygous",
        "fl_risk": "FL 60-80% somatic LOF; Kabuki1 germline predisposition; H3K4me1 enhancer loss; tazemetostat active",
        "pathognomonic": (
            "FL + Kabuki features (broad nasal tip, persistent fingertip pads, short stature) = KMT2D germline; "
            "KMT2D + CREBBP dual LOF in FL = most common epigenetic hit combination (50-60%); "
            "H3K4me1 IHC loss in tumour cells + H3K27me3 gain = KMT2D/EZH2 co-hit FL"
        ),
        "key_avoid": (
            "Standard R-CHOP without epigenetic maintenance in KMT2D-FL (early relapse likely); "
            "Avoid withholding tazemetostat in KMT2D LOF FL (epigenetically driven disease); "
            "Avoid delaying germline testing in young FL with Kabuki features"
        ),
        "key_rule": (
            "Tazemetostat active in KMT2D LOF FL (EZH2 unopposed → PRC2 dependence); "
            "Lenalidomide + rituximab R2 ± azacitidine for KMT2D-FL; "
            "Kabuki1 multi-system surveillance: cardiac echo, renal USS, growth, ID; "
            "Germline panel mandatory if FL onset <40 yr + Kabuki features"
        ),
        "surveillance": (
            "Annual PET/CT; Kabuki multi-system surveillance (cardiac echo, renal USS, ID assessment); "
            "Germline cascade testing; BM biopsy at relapse to assess grade transformation; "
            "KMT2D somatic testing at FL diagnosis to guide epigenetic therapy"
        ),
        "targeted_rx": (
            "Tazemetostat 800 mg BID (EZH2i active in KMT2D LOF); "
            "Lenalidomide + rituximab R2 ± azacitidine; "
            "Azacitidine + mocetinostat epigenetic combo; "
            "R-CHOP or G-CHOP + tazemetostat maintenance"
        ),
    },
    {
        "gene": "RRAGC",
        "protein": (
            "RRAGC -- 1p34.3 AD GOF -- 424aa -- "
            "RagC-GTPase-mTORC1-Activator-46kDa-"
            "Rag-GTPase-Complex-C-Subunit-"
            "FL-15-17pct-Relapsed-Everolimus-mTOR-OMIM-615754"
        ),
        "locus": "1p34.3",
        "protein_size": (
            "424 aa / ~46 kDa / 1p34.3 RRAGC FL molecular context: "
            "STRUCTURE: "
            "  424 aa / ~46 kDa; Ras-related GTP-binding protein C (RagC/RRAGA-D dimer); "
            "  GTP-binding domain (switch I/II regions); "
            "  Forms heterodimer with RRAGA/B at lysosomal surface; "
            "  Recruits mTORC1 to lysosome in response to amino acids; "
            "  GOF: mTORC1 constitutively active independent of amino acid sensing; "
            "FL GOF MECHANISM: "
            "  RRAGC GOF → lysosomal mTORC1 hyperactivation → "
            "  BCL-2/BCL-6 translational upregulation → cell survival; "
            "  15-17% of relapsed/refractory FL carry somatic RRAGC GOF; "
            "  Mutually exclusive with MTOR direct GOF (same pathway, different node); "
            "  RRAGC GOF correlates with histological grade 3B transformation risk; "
            "CLINICAL RULES: "
            "  Everolimus (mTORC1 inhibitor) active in RRAGC GOF FL; "
            "  Idelalisib (PI3Kδ) + everolimus synergistic (PI3K-mTOR axis); "
            "  Temsirolimus (mTORC1) active; "
            "  RRAGC GOF confers relative resistance to standard R-CHOP; "
            "KEY VARIANTS: "
            "  T90N (most common 40%); S75F (25%); Q120L (15%); W115L (10%); "
            "  All localise to GTP-binding switch regions → constitutive GTP binding; "
            "SURVEILLANCE: "
            "  PET/CT at 6M post-CR (high transformation risk); "
            "  RRAGC mutation testing at FL diagnosis emerging; "
            "  Grade transformation biopsy at any progression; "
        ),
        "syndrome": "RRAGC-GOF-mTOR-Hyperactivation-FL",
        "inheritance": "AD GOF heterozygous (primarily somatic; germline reported)",
        "fl_risk": "FL 15-17% relapsed/refractory somatic GOF; mTORC1 hyperactivation; grade 3B transformation; everolimus active",
        "pathognomonic": (
            "Relapsed/refractory FL + RRAGC GOF mutation = mTORC1 pathway dependency → everolimus active; "
            "RRAGC GOF correlates with grade 3B FL (40-60% RRAGC-mutant FL transform to grade 3B within 5yr); "
            "Lysosomal mTOR hyperactivation biomarker: p-S6K1 IHC elevation in RRAGC GOF FL"
        ),
        "key_avoid": (
            "Standard salvage R-ICE/R-DHAP alone without mTOR inhibition in RRAGC GOF FL; "
            "Delay in grade assessment biopsy at relapse (RRAGC GOF high transformation risk); "
            "PI3Kδ monotherapy without adding mTOR inhibitor (incomplete pathway suppression)"
        ),
        "key_rule": (
            "Everolimus (mTORC1 inhibitor) standard for RRAGC GOF relapsed FL; "
            "RRAGC mutation testing at diagnosis and relapse to guide mTOR inhibitor use; "
            "Idelalisib + everolimus combo (PI3K-mTOR); temsirolimus active; "
            "Grade biopsy mandatory at any progression (high transform risk)"
        ),
        "surveillance": (
            "PET/CT at 6M post-CR; grade biopsy at progression; "
            "RRAGC mutation testing at diagnosis and relapse; "
            "p-S6K1/p-4EBP1 IHC as mTORC1 activation biomarker; "
            "Cascade testing if germline GOF confirmed"
        ),
        "targeted_rx": (
            "Everolimus 10 mg QD (mTORC1 inhibitor); "
            "Temsirolimus + rituximab; idelalisib + everolimus PI3K-mTOR combo; "
            "R-bendamustine + everolimus for relapsed; "
            "HSCT consolidation after CR2 for transformed FL"
        ),
    },
    {
        "gene": "MAP2K1",
        "protein": (
            "MAP2K1 -- 15q22.31 AD GOF -- 393aa -- "
            "MEK1-MAPKK1-Dual-Kinase-44kDa-"
            "RAS-MAPK-Pathway-ERK1-2-Activator-"
            "CFC-Syndrome-FL-10-15pct-Cobimetinib-OMIM-115150"
        ),
        "locus": "15q22.31",
        "protein_size": (
            "393 aa / ~44 kDa / 15q22.31 MAP2K1 FL molecular context: "
            "STRUCTURE: "
            "  393 aa / ~44 kDa; dual-specificity kinase MEK1; "
            "  Negative regulatory region (NRR) aa 44-51: autoinhibitory helix; "
            "  Kinase domain (aa 66-361); "
            "  GOF: NRR deletion/mutation → constitutive ERK1/2 phosphorylation; "
            "  Forms homodimer with MAP2K2 (MEK2); "
            "  Signals downstream of BRAF/RAF; "
            "FL GOF MECHANISM: "
            "  MAP2K1 GOF → constitutive ERK1/2 activation → cyclin D1/D3 upregulation → "
            "  G1/S override in FL cells; "
            "  10-15% of relapsed/transformed FL carry somatic MAP2K1 GOF; "
            "  RAS-MAPK activation alternative to PIK3CA/AKT in FL progression; "
            "  MAP2K1 GOF correlates with ibrutinib/chemotherapy resistance; "
            "  Germline MAP2K1 GOF (CFC): FL predisposition case reports; "
            "CLINICAL RULES: "
            "  Cobimetinib (MEK1/2 inhibitor) first-line for MAP2K1 GOF FL; "
            "  Trametinib (MEK1/2) active; selumetinib (MEK1/2) third option; "
            "  CFC germline features: heart defect, ectodermal, facial dysmorphology; "
            "  BRAF-MEK-ERK pathway parallel to BCL-2 in transformed FL; "
            "KEY VARIANTS: "
            "  NRR deletion: Δ98-104 (most common 35%); K57E (25%); Q56P (20%); "
            "  Kinase domain activation: E102K (10%); F53L (10%); "
            "SURVEILLANCE: "
            "  PET/CT at relapse with biopsy for MAP2K1 testing; "
            "  CFC germline: cardiac echo, ectodermal assessment, annual FL surveillance; "
            "  ERK phosphorylation (p-ERK IHC) as biomarker of MAP2K1 GOF FL; "
        ),
        "syndrome": "CFC-Cardio-Facio-Cutaneous-Syndrome-MAP2K1",
        "inheritance": "AD GOF heterozygous",
        "fl_risk": "FL 10-15% relapsed/transformed somatic GOF; RAS-MAPK activation; cobimetinib/trametinib MEKi active",
        "pathognomonic": (
            "Relapsed/transformed FL + MAP2K1 GOF = RAS-MAPK dependency → cobimetinib active; "
            "CFC germline at FL presentation: heart defect + sparse curly hair + FL = MAP2K1 germline; "
            "p-ERK IHC positivity in FL biopsy = MAPK pathway activation (MAP2K1/BRAF/KRAS)"
        ),
        "key_avoid": (
            "Standard salvage R-ICE/R-DHAP without MEK inhibitor in MAP2K1 GOF FL; "
            "Ibrutinib monotherapy in MAP2K1 GOF FL (RAS-MAPK bypass of BTK signalling); "
            "Delay in biopsy MAP2K1 testing at transformation (impacts salvage choice)"
        ),
        "key_rule": (
            "Cobimetinib 60 mg OD (21 days on/7 days off) for MAP2K1 GOF relapsed FL; "
            "Trametinib 2 mg QD alternative MEKi; BRAF mutation co-test (V600E: add vemurafenib); "
            "CFC germline: cardiac echo, dermatology, multi-system surveillance; "
            "p-ERK IHC biomarker confirms MAPK activation → MEKi benefit"
        ),
        "surveillance": (
            "PET/CT at relapse + biopsy for MAP2K1 testing; p-ERK IHC at relapse; "
            "CFC germline: cardiac echo, dermatology, ophthalmology; "
            "BRAF co-mutation testing (COBRAF = vemurafenib + cobimetinib); "
            "Family cascade testing for CFC features"
        ),
        "targeted_rx": (
            "Cobimetinib 60 mg OD MEKi; trametinib 2 mg QD; selumetinib; "
            "Vemurafenib + cobimetinib if BRAF V600E co-mutation; "
            "R-bendamustine + cobimetinib emerging; "
            "PI3Ki + MEKi dual pathway blockade"
        ),
    },
    {
        "gene": "TP53",
        "protein": (
            "TP53 -- 17p13.1 AD LOF -- 393aa -- "
            "p53-43kDa-Tumour-Suppressor-"
            "LFS-Li-Fraumeni-FL-DLBCL-Transformation-"
            "AVOID-RADIATION-ABSOLUTELY-OMIM-151623"
        ),
        "locus": "17p13.1",
        "protein_size": (
            "393 aa / ~43 kDa / 17p13.1 TP53 FL molecular context: "
            "STRUCTURE: "
            "  393 aa / ~43 kDa; tumour suppressor; tetramer; "
            "  N-terminal transactivation domain (aa 1-67); proline-rich region; "
            "  DNA-binding domain (aa 100-290) — 90% of all mutations; "
            "  Tetramerisation domain (aa 323-356); C-terminal reg domain; "
            "FL TRANSFORMATION MECHANISM: "
            "  Germline TP53 LOF → impaired G1/DNA damage checkpoint; "
            "  FL is INDOLENT in most but TP53 germline → histological transformation to DLBCL 40-50%; "
            "  10-year transformation risk: general FL 3% per year (~27%); TP53-germline FL: 40-50%; "
            "  TP53 mutation at transformation = ELN adverse equivalent; complex karyotype 55%; "
            "  APR-246 (eprenetapopt) reactivates mutant p53 → apoptosis; "
            "  Venetoclax: BCL-2 upregulated in TP53-mutant transformed FL; "
            "CLINICAL RULES FOR TP53-FL: "
            "  AVOID RADIATION ABSOLUTELY — LFS → even involved-field RT triggers second malignancies; "
            "  WB-MRI Toronto protocol: Annual WB-MRI + brain MRI; "
            "  Early HSCT consideration in CR1 for TP53-FL (high transformation risk); "
            "  Watchful waiting INAPPROPRIATE in TP53-FL (grade progression accelerated); "
            "  Venetoclax + R-CHOP for transformed FL with TP53 mutation; "
            "KEY VARIANTS: "
            "  Dominant-negative: R175H, G245S, R248W, R248Q, R273H; "
            "  LOF truncating: frameshift/stopgain/splice; "
            "  R337H Brazilian founder (0.3% south Brazil); "
            "SURVEILLANCE: "
            "  Annual WB-MRI + brain MRI + abdominal US; CBC q6M; "
            "  NO PET-CT (radiation exposure); MRI preferred imaging; "
            "  Cascade testing ALL first-degree relatives MANDATORY; "
        ),
        "syndrome": "LFS-Li-Fraumeni-Syndrome",
        "inheritance": "AD LOF heterozygous",
        "fl_risk": "FL→DLBCL transformation 40-50%; AVOID RADIATION ABSOLUTELY; watchful waiting CONTRAINDICATED; WB-MRI Toronto",
        "pathognomonic": (
            "FL + LFS personal/family history (sarcoma/brain/breast/adrenocortical carcinoma) = TP53 germline mandatory; "
            "FL transformation to DLBCL in TP53-germline: complex karyotype 55% + del17p; "
            "p53 IHC: >50% nuclear staining OR null pattern = TP53 mutation in tumour"
        ),
        "key_avoid": (
            "AVOID RADIATION ABSOLUTELY (even involved-field low-dose RT in LFS); "
            "Watchful waiting/watch-and-wait strategy in TP53-germline FL (transformation risk too high); "
            "Delay in HSCT consideration: TP53-FL transforming within 5yr → early CR1 HSCT planning"
        ),
        "key_rule": (
            "AVOID RADIATION ABSOLUTELY — WB-MRI Toronto (annual WB-MRI + brain MRI); "
            "Initiate treatment EARLY in TP53-FL (no watchful waiting); "
            "APR-246 eprenetapopt + venetoclax for TP53-mutant transformed FL; "
            "HSCT consolidation in CR1 (reduce transformation risk); cascade testing ALL relatives"
        ),
        "surveillance": (
            "Annual WB-MRI + brain MRI (NO PET-CT); abdominal US + CBC q6M; "
            "LFS multi-tumour surveillance: annual breast MRI (females), annual colonoscopy ≥25yr; "
            "Grade transformation biopsy at any progression; "
            "Cascade testing ALL first-degree relatives MANDATORY"
        ),
        "targeted_rx": (
            "APR-246 eprenetapopt (reactivates mut-p53) + venetoclax for transformed FL; "
            "R-CHOP + venetoclax for TP53-LOF transformed FL; "
            "Allo-SCT consolidation in CR2 (avoid RT); "
            "MDM2 inhibitor (nutlin/APG-115) + rituximab emerging in WT residual fl"
        ),
    },
    {
        "gene": "ATM",
        "protein": (
            "ATM -- 11q22.3 AD LOF -- 3056aa -- "
            "ATM-PIK3-Kinase-350kDa-"
            "DNA-Damage-Sensor-DSB-Repair-"
            "A-T-RADIOSENSITIVITY-ABSOLUTE-FL-2-3x-OMIM-208900"
        ),
        "locus": "11q22.3",
        "protein_size": (
            "393 aa / ~350 kDa / 11q22.3 ATM FL molecular context: "
            "STRUCTURE: "
            "  3056 aa / ~350 kDa; PI3K-related kinase; "
            "  FAT domain (aa 1966-2480); kinase domain (aa 2712-2962); "
            "  FATC domain (aa 2962-3056): regulatory; "
            "  Activated by MRN complex (MRE11-RAD50-NBS1) at DSBs; "
            "  Phosphorylates H2AX (γH2AX), CHEK2, BRCA1, p53, MDM2; "
            "  LOF: impaired DSB repair → accumulation of chromosomal breaks; "
            "FL MECHANISM: "
            "  Germline ATM heterozygous LOF → NHL 4x lifetime; FL 2-3x; "
            "  del11q22.3 somatic in 35% ATM-FL (biallelic tumour loss); "
            "  RADIOSENSITIVITY ABSOLUTE — germline ATM carriers cannot repair RT-induced DSBs; "
            "  ATM-LOF FL: higher grade transformation risk (del11q accelerates); "
            "  Ceralasertib (ATR inhibitor): synthetic lethality in ATM-null FL; "
            "CLINICAL RULES FOR ATM-FL: "
            "  RADIOSENSITIVITY ABSOLUTE — NO RADIATION EVER (involved field, ISRT, consolidation RT); "
            "  NO TBI-based HSCT conditioning (use RIC: fludarabine + cyclophosphamide); "
            "  Ceralasertib AZD6738 (ATRi): synthetic lethality — ATM-null → ATR-dependent; "
            "  PARP inhibitors (olaparib): active in ATM-null FL (HRD-like BER defect); "
            "  Screen female carriers for breast/ovarian cancer concurrently; "
            "KEY VARIANTS: "
            "  Truncating (frameshift/stopgain): most FL-associated germline variants; "
            "  p.Ser49Cys (kinase-dead missense); p.Lys2759Asn (FATC domain); "
            "  c.7271T>G (p.Val2424Gly): breast cancer-associated; "
            "SURVEILLANCE: "
            "  Annual chest MRI (no mammography for female carriers — AVOID RT); "
            "  del11q22.3 FISH at FL diagnosis; CBC q6M; "
            "  IgA level (A-T biallelic → IgA deficiency — monitor in biallelic); "
            "  Cascade testing ALL first-degree relatives; "
        ),
        "syndrome": "A-T-Ataxia-Telangiectasia-Heterozygous",
        "inheritance": "AD LOF heterozygous",
        "fl_risk": "NHL 4x; FL 2-3x; RADIOSENSITIVITY ABSOLUTE; del11q 35%; ceralasertib ATRi synthetic lethality",
        "pathognomonic": (
            "del11q22.3 FISH in FL + ATM LOF germline = RADIOSENSITIVITY ABSOLUTE — NO RT EVER; "
            "A-T heterozygous + FL + IgA low (biallelic concern) = full ATM sequencing + del11q FISH; "
            "γH2AX accumulation after low-dose RT in vitro = ATM deficiency functional biomarker"
        ),
        "key_avoid": (
            "ANY RADIATION (involved field, ISRT, consolidation RT, TBI): RADIOSENSITIVITY ABSOLUTE; "
            "Mammography in female ATM carriers (use breast MRI ONLY); "
            "TBI-based HSCT conditioning (use RIC fludarabine-cyclophosphamide)"
        ),
        "key_rule": (
            "RADIOSENSITIVITY ABSOLUTE — NO RT EVER; del11q22.3 FISH mandatory at FL diagnosis; "
            "Ceralasertib ATRi (AZD6738) synthetic lethality in ATM-null FL (Phase 2 active); "
            "PARP inhibitors active (olaparib: Phase 2 ATM-deficient NHL); "
            "RIC-HSCT (fludarabine-cyclophosphamide) only — no TBI; breast MRI females"
        ),
        "surveillance": (
            "del11q22.3 FISH at FL diagnosis; annual breast MRI females (NO mammography); "
            "CBC q6M; IgA level if biallelic concern; "
            "Cascade testing ALL first-degree relatives MANDATORY; "
            "Grade transformation biopsy at any progression"
        ),
        "targeted_rx": (
            "Ceralasertib AZD6738 ATRi (synthetic lethality ATM-null) Phase 2 FL; "
            "Olaparib PARP inhibitor (Phase 2 ATM-deficient NHL); "
            "R-bendamustine + ceralasertib; "
            "RIC-HSCT (fludarabine-cyclophosphamide) — NO TBI; "
            "Screen concurrent breast/ovarian cancer"
        ),
    },
]


def _make_patient(gene: str, seed: int) -> dict:
    rng = random.Random(seed)

    g = next(gd for gd in ATLAS_GENES if gd["gene"] == gene)

    # FL-specific parameters by gene
    fl_params = {
        "CREBBP": dict(age_mu=52, age_sd=14,  male_prob=0.48, ezh2_mut_prob=0.50,
                       bm_involved_prob=0.65, transform_prob=0.18, flipi_hi_prob=0.40,
                       cr_rate=0.72, five_yr_os=0.80),
        "EZH2":   dict(age_mu=58, age_sd=12,  male_prob=0.50, ezh2_mut_prob=0.90,
                       bm_involved_prob=0.60, transform_prob=0.12, flipi_hi_prob=0.35,
                       cr_rate=0.75, five_yr_os=0.82),
        "TNFRSF14": dict(age_mu=55, age_sd=13, male_prob=0.48, ezh2_mut_prob=0.35,
                         bm_involved_prob=0.70, transform_prob=0.25, flipi_hi_prob=0.55,
                         cr_rate=0.65, five_yr_os=0.72),
        "KMT2D":  dict(age_mu=50, age_sd=14,  male_prob=0.50, ezh2_mut_prob=0.55,
                       bm_involved_prob=0.62, transform_prob=0.20, flipi_hi_prob=0.42,
                       cr_rate=0.70, five_yr_os=0.78),
        "RRAGC":  dict(age_mu=57, age_sd=12,  male_prob=0.52, ezh2_mut_prob=0.30,
                       bm_involved_prob=0.55, transform_prob=0.35, flipi_hi_prob=0.55,
                       cr_rate=0.62, five_yr_os=0.68),
        "MAP2K1": dict(age_mu=56, age_sd=12,  male_prob=0.52, ezh2_mut_prob=0.25,
                       bm_involved_prob=0.50, transform_prob=0.30, flipi_hi_prob=0.48,
                       cr_rate=0.65, five_yr_os=0.70),
        "TP53":   dict(age_mu=38, age_sd=14,  male_prob=0.50, ezh2_mut_prob=0.20,
                       bm_involved_prob=0.55, transform_prob=0.45, flipi_hi_prob=0.55,
                       cr_rate=0.60, five_yr_os=0.62),
        "ATM":    dict(age_mu=57, age_sd=12,  male_prob=0.50, ezh2_mut_prob=0.25,
                       bm_involved_prob=0.65, transform_prob=0.22, flipi_hi_prob=0.42,
                       cr_rate=0.70, five_yr_os=0.75),
    }
    p = fl_params[gene]

    # Age at FL diagnosis
    age_raw = rng.gauss(p["age_mu"], p["age_sd"])
    if gene == "TP53":
        age = max(18, min(65, round(age_raw, 1)))
    else:
        age = max(28, min(82, round(age_raw, 1)))

    is_male             = rng.random() < p["male_prob"]
    ezh2_mutated        = rng.random() < p["ezh2_mut_prob"]
    bm_involved         = rng.random() < p["bm_involved_prob"]
    transformed         = rng.random() < p["transform_prob"]
    flipi_high          = rng.random() < p["flipi_hi_prob"]

    # FL grade (1-2, 3A, 3B)
    if gene in ("EZH2", "CREBBP", "KMT2D"):
        grade_weights = [55, 35, 10]  # Grade 1-2 / 3A / 3B
    elif gene in ("RRAGC", "MAP2K1", "TP53"):
        grade_weights = [35, 35, 30]  # More 3B
    else:
        grade_weights = [50, 35, 15]
    fl_grade = rng.choices(["Grade 1-2", "Grade 3A", "Grade 3B"], weights=grade_weights, k=1)[0]

    # FLIPI score
    if flipi_high:
        flipi = rng.choices([3, 4, 5], weights=[40, 35, 25], k=1)[0]
    else:
        flipi = rng.choices([0, 1, 2], weights=[20, 45, 35], k=1)[0]

    # Ann Arbor stage (FL is usually advanced stage)
    stage_probs = [0.05, 0.12, 0.30, 0.53]
    stage_roll = rng.random()
    stage = 4
    cumulative = 0.0
    for s, sp in enumerate(stage_probs, 1):
        cumulative += sp
        if stage_roll < cumulative:
            stage = s
            break

    # Concurrent breast cancer (females)
    concurrent_breast = (not is_male) and (gene in ("ATM", "TP53", "BRCA2")) and rng.random() < 0.15

    # IVIG required (immunodeficiency-related)
    ivig_required = (gene in ("TNFRSF14",)) and rng.random() < 0.08

    # LDH
    ldh_elevated = rng.random() < (0.60 if (transformed or fl_grade == "Grade 3B") else 0.30)

    # Response to first-line R-CHOP/G-CHOP
    cr_base = p["cr_rate"]
    if transformed:
        cr_base -= 0.12
    if flipi >= 4:
        cr_base -= 0.05
    cr_base = max(0.45, min(0.90, cr_base))

    response = rng.choices(
        ["CR", "PR", "SD", "PD"],
        weights=[cr_base * 100, 18, 7, (1 - cr_base) * 100 - 25],
        k=1
    )[0]

    # 5-yr OS
    os_base = p["five_yr_os"]
    if transformed:
        os_base -= 0.12
    if flipi >= 4:
        os_base -= 0.08
    if fl_grade == "Grade 3B":
        os_base -= 0.05
    os_base = max(0.30, min(0.95, os_base))

    five_yr_os_achieved = rng.random() < os_base
    outcome = "Alive in remission" if (five_yr_os_achieved and response == "CR") else \
              "Alive with disease" if five_yr_os_achieved else "Deceased"

    return {
        "gene":               gene,
        "seed":               seed,
        "sex":                "M" if is_male else "F",
        "age_at_diagnosis":   age,
        "fl_grade":           fl_grade,
        "stage":              stage,
        "flipi":              flipi,
        "ezh2_mutated":       ezh2_mutated,
        "bm_involved":        bm_involved,
        "transformed":        transformed,
        "ldh_elevated":       ldh_elevated,
        "concurrent_breast":  concurrent_breast,
        "ivig_required":      ivig_required,
        "response":           response,
        "five_yr_os":         five_yr_os_achieved,
        "outcome":            outcome,
        "locus":              g["locus"],
        "inheritance":        g["inheritance"],
        "key_rule":           g["key_rule"],
    }


def generate_overview() -> dict:
    patients = []
    for i, gd in enumerate(ATLAS_GENES):
        for j in range(40):
            patients.append(_make_patient(gd["gene"], SEED_BASE + i + j * 8))

    gene_counts = {}
    for p in patients:
        gene_counts.setdefault(p["gene"], 0)
        gene_counts[p["gene"]] += 1

    total           = len(patients)
    cr_n            = sum(1 for p in patients if p["response"] == "CR")
    ezh2_mut_n      = sum(1 for p in patients if p["ezh2_mutated"])
    bm_involved_n   = sum(1 for p in patients if p["bm_involved"])
    transformed_n   = sum(1 for p in patients if p["transformed"])
    ldh_elevated_n  = sum(1 for p in patients if p["ldh_elevated"])
    five_yr_alive   = sum(1 for p in patients if p["five_yr_os"])
    males           = sum(1 for p in patients if p["sex"] == "M")
    grade3b_n       = sum(1 for p in patients if p["fl_grade"] == "Grade 3B")
    flipi_hi_n      = sum(1 for p in patients if p["flipi"] >= 3)
    stage4_n        = sum(1 for p in patients if p["stage"] == 4)

    avg_age  = round(sum(p["age_at_diagnosis"] for p in patients) / total, 1)
    avg_flipi = round(sum(p["flipi"] for p in patients) / total, 1)

    gene_ages = {}
    for gd in ATLAS_GENES:
        pts = [p for p in patients if p["gene"] == gd["gene"]]
        gene_ages[gd["gene"]] = round(sum(p["age_at_diagnosis"] for p in pts) / len(pts), 1)

    gene_summary = []
    for gd in ATLAS_GENES:
        pts = [p for p in patients if p["gene"] == gd["gene"]]
        gene_summary.append({
            "gene":          gd["gene"],
            "syndrome":      gd["syndrome"],
            "locus":         gd["locus"],
            "inheritance":   gd["inheritance"],
            "n":             len(pts),
            "mean_age_dx":   gene_ages[gd["gene"]],
            "fl_risk":       gd["fl_risk"],
            "pathognomonic": gd["pathognomonic"],
            "key_rule":      gd["key_rule"],
            "key_avoid":     gd["key_avoid"],
            "targeted_rx":   gd["targeted_rx"],
        })

    return {
        "atlas":            "Hereditary-Follicular-Lymphoma-Predisposition-Atlas",
        "seed_range":       f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients":   total,
        "total_genes":      8,
        "gene_counts":      gene_counts,
        "cr_n":             cr_n,
        "ezh2_mutated":     ezh2_mut_n,
        "bm_involved":      bm_involved_n,
        "transformed":      transformed_n,
        "ldh_elevated":     ldh_elevated_n,
        "five_yr_alive":    five_yr_alive,
        "males":            males,
        "grade3b":          grade3b_n,
        "flipi_high":       flipi_hi_n,
        "stage4":           stage4_n,
        "avg_age":          avg_age,
        "avg_flipi":        avg_flipi,
        "gene_summary":     gene_summary,
        "clinical_pearl": (
            "FL-GERMLINE: germline predisposition determines BOTH transformation risk AND treatment approach; "
            "EZH2-GOF→tazemetostat-FDA2020-800mg-BID (ORR 69%); CREBBP/KMT2D-LOF→tazemetostat+mocetinostat-epigenetic-combo; "
            "TP53-AVOID-RADIATION-ABSOLUTELY (LFS — no watchful waiting); ATM-RADIOSENSITIVITY-ABSOLUTE (no RT, no TBI); "
            "TNFRSF14-LOF→obinutuzumab-preferred-over-rituximab (poor-TME-prognosis); "
            "RRAGC-GOF→everolimus-mTOR; MAP2K1-GOF→cobimetinib-MEKi"
        ),
    }


def generate_breakdown() -> dict:
    all_patients = []
    for i, gd in enumerate(ATLAS_GENES):
        for j in range(40):
            all_patients.append(_make_patient(gd["gene"], SEED_BASE + i + j * 8))

    gene_data = {}
    for gd in ATLAS_GENES:
        pts = [p for p in all_patients if p["gene"] == gd["gene"]]

        cr_n         = sum(1 for p in pts if p["response"] == "CR")
        ezh2_n       = sum(1 for p in pts if p["ezh2_mutated"])
        bm_n         = sum(1 for p in pts if p["bm_involved"])
        transform_n  = sum(1 for p in pts if p["transformed"])
        ldh_n        = sum(1 for p in pts if p["ldh_elevated"])
        grade3b_n    = sum(1 for p in pts if p["fl_grade"] == "Grade 3B")
        flipi_hi_n   = sum(1 for p in pts if p["flipi"] >= 3)
        stage4_n     = sum(1 for p in pts if p["stage"] == 4)
        alive5_n     = sum(1 for p in pts if p["five_yr_os"])
        concurrent_b = sum(1 for p in pts if p["concurrent_breast"])

        avg_age_dx = round(sum(p["age_at_diagnosis"] for p in pts) / len(pts), 1)
        avg_flipi   = round(sum(p["flipi"] for p in pts) / len(pts), 1)

        patients_list = [{
            "seed":             p["seed"],
            "sex":              p["sex"],
            "age_at_diagnosis": p["age_at_diagnosis"],
            "fl_grade":         p["fl_grade"],
            "stage":            p["stage"],
            "flipi":            p["flipi"],
            "ezh2_mutated":     p["ezh2_mutated"],
            "bm_involved":      p["bm_involved"],
            "transformed":      p["transformed"],
            "ldh_elevated":     p["ldh_elevated"],
            "response":         p["response"],
            "outcome":          p["outcome"],
        } for p in pts]

        gene_data[gd["gene"]] = {
            "gene":            gd["gene"],
            "syndrome":        gd["syndrome"],
            "locus":           gd["locus"],
            "inheritance":     gd["inheritance"],
            "n":               len(pts),
            "cr_n":            cr_n,
            "ezh2_mutated_n":  ezh2_n,
            "bm_involved_n":   bm_n,
            "transformed_n":   transform_n,
            "ldh_elevated_n":  ldh_n,
            "grade3b_n":       grade3b_n,
            "flipi_high_n":    flipi_hi_n,
            "stage4_n":        stage4_n,
            "five_yr_alive_n": alive5_n,
            "concurrent_breast_n": concurrent_b,
            "avg_age_dx":      avg_age_dx,
            "avg_flipi":       avg_flipi,
            "fl_risk":         gd["fl_risk"],
            "pathognomonic":   gd["pathognomonic"],
            "key_rule":        gd["key_rule"],
            "key_avoid":       gd["key_avoid"],
            "targeted_rx":     gd["targeted_rx"],
            "patients":        patients_list[:20],
        }

    return {
        "atlas":      "Hereditary-Follicular-Lymphoma-Predisposition-Atlas",
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "gene_data":  gene_data,
    }


def generate_definitions() -> dict:
    defs = {}
    for gd in ATLAS_GENES:
        defs[gd["gene"]] = {
            "gene":          gd["gene"],
            "locus":         gd["locus"],
            "protein_size":  gd["protein_size"],
            "syndrome":      gd["syndrome"],
            "inheritance":   gd["inheritance"],
            "fl_risk":       gd["fl_risk"],
            "pathognomonic": gd["pathognomonic"],
            "key_avoid":     gd["key_avoid"],
            "key_rule":      gd["key_rule"],
            "surveillance":  gd["surveillance"],
            "targeted_rx":   gd["targeted_rx"],
        }

    key_clinical_distinctions = [
        "CREBBP-FL HAT-LOF mocetinostat: 60-70% FL carry somatic CREBBP HAT mutations (exon 30-31 hotspot); "
        "germline RTS1 (broad thumbs/toes + short stature + ID) → FL predisposition; "
        "mocetinostat HDAC1/2/3 restores H3K27ac; tazemetostat synergistic (CREBBP LOF → EZH2 dominant); "
        "azacitidine + lenalidomide + rituximab R2-aza for relapsed CREBBP-FL",

        "EZH2 GOF tazemetostat FDA2020 — ONLY approved EZH2 inhibitor for FL: Y641F/Y641N GOF in 80-90% FL; "
        "tazemetostat 800 mg BID oral ORR 69% (EZH2-mutant) vs 34% (EZH2-WT); "
        "EZH2 mutation testing MANDATORY at FL diagnosis; "
        "obinutuzumab + tazemetostat SYMPHONY trial frontline; Weaver germline: multi-system surveillance",

        "TNFRSF14 LOF obinutuzumab preferred — BTLA/CD160 checkpoint loss: 40-68% somatic FL; "
        "1p36 FISH deletion in 35-40% FL; poor prognosis marker (FLIPI2 elevation); "
        "obinutuzumab preferred over rituximab (enhanced ADCC in TME-disrupted FL); "
        "lenalidomide + rituximab R2 restores TME; PD-1 checkpoint rationale (nivolumab FL cohort); "
        "1p36 FISH mandatory at diagnosis",

        "KMT2D LOF tazemetostat + epigenetic combo: 60-80% somatic; Kabuki1 germline (broad nasal tip, "
        "persistent fingertip pads, short stature, ID, heart defects 30%); "
        "H3K4me1 loss → EZH2 unopposed → tazemetostat active; "
        "azacitidine + lenalidomide + rituximab for relapsed KMT2D-FL; "
        "cardiac echo mandatory in Kabuki1 (heart defects 30%)",

        "RRAGC GOF everolimus — mTORC1 hyperactivation: 15-17% relapsed/refractory FL; "
        "constitutive lysosomal mTORC1 activation → BCL-2/BCL-6 translation; "
        "everolimus 10 mg QD active; temsirolimus + rituximab; idelalisib + everolimus PI3K-mTOR combo; "
        "grade 3B transformation risk 40-60% at 5yr in RRAGC GOF FL; "
        "p-S6K1/p-4EBP1 IHC as biomarker; biopsy mandatory at any progression",

        "MAP2K1 GOF cobimetinib MEKi — RAS-MAPK activation: 10-15% relapsed/transformed FL; "
        "NRR deletion (Δ98-104) most common; constitutive ERK1/2 phosphorylation; "
        "cobimetinib 60 mg OD (21/7 schedule); trametinib 2 mg QD; BRAF V600E co-test (add vemurafenib); "
        "CFC germline: heart defect + sparse curly hair + FL; p-ERK IHC confirms MAPK activation; "
        "standard R-ICE insufficient alone without MEKi in MAP2K1 GOF",

        "TP53-FL AVOID RADIATION ABSOLUTELY + no watchful waiting: LFS → transformation to DLBCL 40-50%; "
        "10yr transform risk general FL 27%; TP53-germline FL 40-50% — no watchful waiting appropriate; "
        "APR-246 eprenetapopt + venetoclax for TP53-mutant transformed FL; "
        "WB-MRI Toronto annual (NO PET-CT); allo-SCT CR1 consolidation preferred; "
        "cascade testing ALL first-degree relatives MANDATORY",

        "ATM-FL RADIOSENSITIVITY ABSOLUTE — no RT ever, no TBI: NHL 4x; FL 2-3x; del11q22.3 FISH 35%; "
        "ceralasertib AZD6738 ATRi synthetic lethality in ATM-null FL (Phase 2 active); "
        "olaparib PARP inhibitor active (ATM-deficient NHL Phase 2); "
        "NO involved-field RT, NO ISRT, NO consolidation RT, NO TBI-based HSCT; "
        "RIC-HSCT only (fludarabine-cyclophosphamide); breast MRI females (NO mammography)",
    ]

    defs_array = [{
        "term":       f"{gd['gene']} — {gd['syndrome']}",
        "definition": (
            f"LOCUS: {gd['locus']} | INHERITANCE: {gd['inheritance']}\n"
            f"FL RISK: {gd['fl_risk']}\n"
            f"PATHOGNOMONIC: {gd['pathognomonic']}\n"
            f"KEY RULE: {gd['key_rule']}\n"
            f"KEY AVOID: {gd['key_avoid']}\n"
            f"SURVEILLANCE: {gd['surveillance']}\n"
            f"TARGETED Rx: {gd['targeted_rx']}"
        ),
    } for gd in ATLAS_GENES]

    return {
        "atlas":                     "Hereditary-Follicular-Lymphoma-Predisposition-Atlas",
        "seed_range":                f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions":               defs_array,
        "gene_defs":                 defs,
        "key_clinical_distinctions": key_clinical_distinctions,
    }
