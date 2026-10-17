#!/usr/bin/env python3
"""Hereditary-NK-T-Cell-Lymphoma-Predisposition-Atlas -- Complete 8-Gene Reference
Extranodal NK/T-Cell Lymphoma, Nasal Type and related EBV-driven T/NK
lymphoproliferative disorders — germline predisposition.

SH2D1A  (SAP; 128aa; Xq25; XLR LOF;
          X-linked Lymphoproliferative Disease Type 1 (XLP1/OMIM 308240);
          EBV-NK-T-lymphoma 50-80x HIGHEST; iNKT-cell absent PATHOGNOMONIC;
          HSCT MANDATORY BEFORE EBV exposure; seed SEED_BASE+0).
XIAP    (XIAP/BIRC4; 497aa; Xq25; XLR LOF;
          X-linked Lymphoproliferative Disease Type 2 (XLP2/OMIM 300635);
          EBV-NK-T-lymphoma + IBD-HLH combination PATHOGNOMONIC;
          NOT EBV-exclusive; Ruxolitinib+HSCT; seed SEED_BASE+1).
CD27    (CD27; 260aa; 12p13.31; AR LOF;
          CD27 Deficiency (OMIM 615122);
          EBV-NK-T-LPD 30-50x HIGHEST; HLH 45pct; HSCT MANDATORY;
          Rituximab INEFFECTIVE NK-cell lymphoma; seed SEED_BASE+2).
ITK     (ITK; 620aa; 5q33.3; AR LOF;
          ITK Deficiency / Tec-Kinase (OMIM 186973);
          EBV-CD4-NK-T-LPD 90pct; HLH 50pct; HSCT MANDATORY BEFORE EBV;
          seed SEED_BASE+3).
GATA2   (GATA2; 480aa; 3q21.3; AD LOF;
          GATA2 Deficiency / MonoMAC / Emberger (OMIM 137295);
          NK-lymphopenia PATHOGNOMONIC; NK-T-lymphoma EBV-CMV-Q3M;
          HSCT Curative; seed SEED_BASE+4).
PIK3CD  (PI3K-delta; 1044aa; 1p36.22; AD GOF;
          Activated PI3K-delta Syndrome Type 1 (APDS1/OMIM 615513);
          EBV-NK-T-lymphoma 10-15pct; Leniolisib FDA2023;
          Live-vaccines CI; seed SEED_BASE+5).
STAT5B  (STAT5B; 787aa; 17q21.2; AD GOF;
          STAT5B GOF Syndrome / NK-LGL Leukemia (OMIM 604260);
          NK-LGL-NK-T-lymphoma; Cytopenia HALLMARK; Ruxolitinib JAKi;
          Methotrexate AVOID; seed SEED_BASE+6).
RASGRP1 (RASGRP1; 795aa; 15q14; AR LOF;
          RASGRP1 Deficiency (OMIM 603962);
          EBV-T-NK-LPD; RAS-MAPK dysregulation; HSCT beneficial;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3654-3661)
"""
import random

SEED_BASE = 3654

ATLAS_GENES = [
    {
        "gene": "SH2D1A",
        "protein": (
            "SAP -- Xq25 XLR LOF -- 128aa -- "
            "SAP-15kDa-SLAM-Associated-Adaptor-"
            "SH2-Domain-"
            "XLP1-Duncan-EBV-NK-T-Lymphoma-50-80x-HIGHEST-"
            "SH2D1A-OMIM-308240"
        ),
        "locus": "Xq25",
        "protein_size": (
            "128 aa / ~15 kDa / Xq25 SH2D1A NK/T lymphoma molecular context: "
            "STRUCTURE: "
            "  128 aa / ~15 kDa; SH2 domain only adaptor protein; signalling scaffold; "
            "  SH2 domain (aa 1-103): SLAM-family receptor tyrosine phosphopeptide binding; "
            "  C-terminal extension (aa 103-128): regulates binding affinity; "
            "  SAP binds SLAM (SLAMF1), 2B4 (SLAMF4), NTB-A (SLAMF6), CRACC (SLAMF7): ITK/Fyn recruitment; "
            "  SAP LOF → uninhibited SLAM signalling → impaired NKT cell development; "
            "  iNKT cells (invariant NKT): absolutely ABSENT in XLP1 — PATHOGNOMONIC; "
            "  EBV-specific cytotoxic T-cell and NK-cell responses severely impaired; "
            "  EBV → uncontrolled B-cell infection → NK/T-LPD and lymphoma; "
            "XLP1-NK/T LYMPHOMA CONTEXT: "
            "  XLP1 (Duncan disease): X-linked; males ONLY (X-linked recessive); "
            "  EBV-triggered NK/T-lymphoproliferative disease: 50-80x HIGHEST risk; "
            "  Manifestations: fulminant EBV/infectious mononucleosis 60-70%; HLH 70%; NK/T-lymphoma 30%; "
            "  SAP-deficient iNKT ABSENT — PATHOGNOMONIC flow cytometry finding; "
            "  HSCT MANDATORY before EBV exposure (ideally pre-school age); "
            "  Post-EBV: immunoglobulin deficiency (hypogammaglobulinemia) 25%; "
            "CLINICAL RULES FOR SH2D1A-NK/T LYMPHOMA: "
            "  HSCT MANDATORY — only curative option; pre-EBV timing critical (improves survival); "
            "  iNKT ABSENT: flow cytometry iNKT (Valpha24+Vbeta11+) — PATHOGNOMONIC diagnostic; "
            "  ALL MALE FAMILY MEMBERS: genetic testing at birth before EBV; "
            "  Live EBV vaccine: ABSOLUTELY CI in XLP1; "
            "  Acyclovir/rituximab: bridging before HSCT — NOT curative; "
            "  HLH protocol: HLH-2004 (etoposide+dex+ciclosporin) if HLH triggered; "
            "KEY VARIANTS: "
            "  Missense SH2: pCys42Arg, pThr53Ile, pGln99Pro (pY-binding surface); "
            "  Truncating: early frameshift/nonsense — null phenotype; "
            "  Splicing: IVS1+1G>A, IVS2+1G>T; "
            "  Large deletions detected by MLPA; "
            "SURVEILLANCE: "
            "  EBV serology monthly until HSCT or through pre-school years; "
            "  Annual CBC + NK-cell count; iNKT flow cytometry at diagnosis; "
            "  All male family members: SH2D1A genetic testing at birth; "
            "TARGETED Rx: "
            "  HSCT (allo-SCT): only curative — RIC preferred; "
            "  HLH-2004 bridging: etoposide+dexamethasone+ciclosporin; "
            "  Rituximab: B-cell debulking pre-HSCT (not curative); "
            "  Acyclovir: antiviral prophylaxis pre-HSCT; "
            "  L-asparaginase-based regimen (SMILE/DDGP) for NK/T-lymphoma diagnosis"
        ),
        "syndrome": "X-linked Lymphoproliferative Disease Type 1 (XLP1/Duncan disease)",
        "inheritance": "XLR LOF; males only",
        "nkt_risk": "EBV-NK-T-lymphoma 50-80x HIGHEST; iNKT absent PATHOGNOMONIC; HLH 70pct",
        "pathognomonic": "iNKT-cells-ABSENT-flow-cytometry-PATHOGNOMONIC; EBV-triggered-fulminant-IM",
        "key_rule": "HSCT-MANDATORY-BEFORE-EBV; iNKT-ABSENT-DIAGNOSTIC; ALL-MALES-TEST-AT-BIRTH",
        "key_avoid": "Live EBV vaccine CI; rituximab alone NOT curative; avoid CHOP without L-asparaginase",
        "surveillance": "EBV serology monthly until HSCT; CBC+NK-cell count annual; male family members genetic test",
        "targeted_rx": "HSCT allo-SCT RIC (curative); HLH-2004 etoposide bridge; acyclovir; L-asp-based NK/T-Rx",
    },
    {
        "gene": "XIAP",
        "protein": (
            "XIAP/BIRC4 -- Xq25 XLR LOF -- 497aa -- "
            "XIAP-56kDa-BIR1-BIR2-BIR3-RING-"
            "Inhibitor-Apoptosis-NOD2-RIPK2-"
            "XLP2-IBD-HLH-NK-T-Lymphoma-PATHOGNOMONIC-"
            "XIAP-OMIM-300635"
        ),
        "locus": "Xq25",
        "protein_size": (
            "497 aa / ~56 kDa / Xq25 XIAP NK/T lymphoma molecular context: "
            "STRUCTURE: "
            "  497 aa / ~56 kDa; inhibitor of apoptosis protein (IAP) family; "
            "  BIR1 domain (aa 22-89): TRAF2 interaction; NF-kB pathway; "
            "  BIR2 domain (aa 163-230): caspase-3/7 inhibition; "
            "  BIR3 domain (aa 258-329): caspase-9 inhibition; Smac/DIABLO binding; "
            "  RING domain (aa 447-490): E3 ubiquitin ligase; RIPK2 ubiquitination; "
            "  XIAP LOF → dysregulated NOD2/RIPK2 signalling → intestinal inflammation; "
            "  XIAP LOF → impaired NK-cell cytotoxicity; TNF-driven NK-cell apoptosis; "
            "  XIAP LOF → susceptibility to EBV-driven lymphoproliferation (NOT EBV exclusive); "
            "XLP2-NK/T LYMPHOMA CONTEXT: "
            "  XLP2 (XIAP/BIRC4 deficiency): X-linked; males ONLY; "
            "  DISTINCT from XLP1: NOT predominantly EBV-triggered; "
            "  IBD+HLH COMBINATION: PATHOGNOMONIC (Crohn's-like IBD + recurrent splenomegaly); "
            "  NK/T-lymphoproliferative disease: 15-25% of XLP2 patients; "
            "  NK-cell cytotoxicity impaired; CD107a normal (unlike FHL); "
            "  Ruxolitinib (JAKi): controls inflammation + bridges to HSCT; "
            "CLINICAL RULES FOR XIAP-NK/T LYMPHOMA: "
            "  NOT EBV-exclusive — colitis + splenomegaly without EBV = diagnostic clue; "
            "  IBD: Crohn's-like colitis in XLP2 — standard IBD Rx insufficient; "
            "  Ruxolitinib (JAK1/2i): controls HLH+IBD inflammation bridge to HSCT; "
            "  HSCT: corrects HLH and IBD — MANDATORY for severe disease; "
            "  Anakinra (IL-1Ra): used for refractory HLH bridge; "
            "  SMAC mimetics: BV6, birinapant — experimental in XIAP-deficient lymphoma; "
            "KEY VARIANTS: "
            "  BIR1: pSer65Pro (NOD2 signalling impaired); "
            "  BIR2: pArg162Gln (caspase inhibition lost); "
            "  Truncating: frameshift throughout — severe; "
            "  RING: pCys450Arg (E3 ligase lost); "
            "SURVEILLANCE: "
            "  CBC + LFTs quarterly; EBV+CMV monitoring monthly; "
            "  GI endoscopy if IBD symptoms; ferritin trend for HLH monitoring; "
            "  Male family members: XIAP genetic testing; "
            "TARGETED Rx: "
            "  Ruxolitinib (JAK1/2i): HLH+IBD control — bridge to HSCT; "
            "  Anakinra IL-1Ra: refractory HLH bridge; "
            "  HSCT allo-SCT: curative for HLH+IBD+lymphoma; "
            "  SMAC mimetics: experimental; "
            "  L-asparaginase-based regimen (SMILE/DDGP) for NK/T-lymphoma"
        ),
        "syndrome": "X-linked Lymphoproliferative Disease Type 2 (XLP2/XIAP deficiency)",
        "inheritance": "XLR LOF; males only",
        "nkt_risk": "NK-T-lymphoma 15-25pct; IBD-HLH COMBINATION PATHOGNOMONIC; NOT-EBV-exclusive",
        "pathognomonic": "IBD+HLH-COMBINATION-PATHOGNOMONIC; colitis+splenomegaly-without-EBV",
        "key_rule": "NOT-EBV-EXCLUSIVE; Ruxolitinib-bridge-HSCT; IBD-colitis-PATHOGNOMONIC",
        "key_avoid": "Standard IBD Rx insufficient XLP2; don't assume EBV trigger needed for HLH",
        "surveillance": "CBC+LFTs quarterly; EBV+CMV monthly; GI endoscopy if IBD; ferritin HLH trend",
        "targeted_rx": "Ruxolitinib JAKi bridge; anakinra IL-1Ra; HSCT curative; SMAC mimetics experimental",
    },
    {
        "gene": "CD27",
        "protein": (
            "CD27 -- 12p13.31 AR LOF -- 260aa -- "
            "CD27-29kDa-TNF-Receptor-TNFRSF7-"
            "CRD-TM-Cytoplasmic-TRAF2-"
            "EBV-NK-T-LPD-30-50x-HIGHEST-"
            "CD27-OMIM-186711"
        ),
        "locus": "12p13.31",
        "protein_size": (
            "260 aa / ~29 kDa / 12p13.31 CD27 NK/T lymphoma molecular context: "
            "STRUCTURE: "
            "  260 aa / ~29 kDa; type I transmembrane glycoprotein; TNF receptor superfamily 7; "
            "  Signal peptide (aa 1-19); "
            "  Cysteine-rich domain (CRD, aa 20-57): CD70 (CD27L) binding; "
            "  Transmembrane domain (aa 183-205); "
            "  Cytoplasmic tail (aa 206-260): TRAF2/TRAF5 binding; NF-kB/MAPK activation; "
            "  CD27+CD70 costimulation: T-cell and NK-cell survival, memory, and cytotoxicity; "
            "  CD27 LOF → loss of T-cell/NK-cell costimulation → impaired EBV-specific immunity; "
            "CD27 DEFICIENCY-NK/T LYMPHOMA CONTEXT: "
            "  CD27 deficiency (OMIM 615122): AR biallelic LOF; "
            "  EBV-driven NK/T-cell lymphoproliferative disease: 30-50x HIGHEST risk; "
            "  HLH: 45% of CD27-deficient patients; "
            "  Manifestations: splenomegaly, lymphadenopathy, NK/T-LPD, HLH; "
            "  Hypogammaglobulinemia: 60% — IVIG required; "
            "  Rituximab: INEFFECTIVE for NK-cell lymphoma (NK cells CD20 negative); "
            "  HSCT MANDATORY — curative; "
            "CLINICAL RULES FOR CD27-NK/T LYMPHOMA: "
            "  HSCT MANDATORY — curative for NK/T-LPD; "
            "  Rituximab INEFFECTIVE: NK-cell lymphoma is CD20-negative; "
            "  IVIG: mandatory if IgG <4 g/L (bridge and maintenance); "
            "  HLH-2004 protocol: etoposide+dex+ciclosporin for HLH episodes; "
            "  EBV monitoring: PCR monthly until HSCT; "
            "  Acyclovir prophylaxis: 20 mg/kg/day until HSCT; "
            "KEY VARIANTS: "
            "  pCys to Ser: CRD disulfide disruption; most common; "
            "  Frameshift/truncating: complete LOF; "
            "  Splice-site: exon skipping; "
            "SURVEILLANCE: "
            "  EBV PCR monthly; CBC + Ig subsets monthly; ferritin for HLH; "
            "  Echocardiogram: lymphoma staging; "
            "  Sibling testing: AR — both parents typically carriers; "
            "TARGETED Rx: "
            "  HSCT allo-SCT: curative; "
            "  HLH-2004 etoposide: bridge to HSCT; "
            "  IVIG IgG<4: mandatory replacement; "
            "  Acyclovir 20mg/kg/day EBV prophylaxis; "
            "  L-asparaginase-based regimen (SMILE/DDGP) for NK/T-lymphoma; "
            "  AVOID rituximab alone: ineffective NK-cell disease"
        ),
        "syndrome": "CD27 Deficiency — EBV susceptibility",
        "inheritance": "AR LOF; biallelic",
        "nkt_risk": "EBV-NK-T-LPD 30-50x HIGHEST; HLH 45pct; Rituximab-INEFFECTIVE",
        "pathognomonic": "EBV-driven-NK-T-LPD-30-50x; splenomegaly+hypogamma+EBV-susceptibility",
        "key_rule": "HSCT-MANDATORY; Rituximab-INEFFECTIVE-NK-CD20-negative; IVIG-mandatory-IgG<4",
        "key_avoid": "Rituximab alone ineffective NK-cell disease (CD20 negative); delay HSCT risks HLH",
        "surveillance": "EBV PCR monthly; CBC+Ig monthly; ferritin HLH trend; sibling cascade testing",
        "targeted_rx": "HSCT curative; HLH-2004 bridge; IVIG IgG<4; acyclovir prophylaxis; L-asp NK/T-Rx",
    },
    {
        "gene": "ITK",
        "protein": (
            "ITK -- 5q33.3 AR LOF -- 620aa -- "
            "ITK-72kDa-Tec-Kinase-"
            "PH-TH-SH3-SH2-KD-"
            "EBV-CD4-NK-T-LPD-90pct-HLH-50pct-"
            "ITK-OMIM-186973"
        ),
        "locus": "5q33.3",
        "protein_size": (
            "620 aa / ~72 kDa / 5q33.3 ITK NK/T lymphoma molecular context: "
            "STRUCTURE: "
            "  620 aa / ~72 kDa; non-receptor tyrosine kinase; Tec family; "
            "  PH domain (aa 1-100): PIP3 binding; membrane recruitment; "
            "  TH domain (aa 101-158): Btk motif + proline-rich; "
            "  SH3 domain (aa 159-214): proline-rich peptide binding; "
            "  SH2 domain (aa 215-309): pTyr recognition; ZAP70 interaction; "
            "  Kinase domain (KD, aa 323-619): catalytic; PLCg1 phosphorylation; "
            "  ITK LOF → impaired TCR/NKT-cell signalling → PLCg1 not activated → Ca2+ flux absent; "
            "  ITK LOF → selective EBV susceptibility: CD4+ T-cell and NK/T-cell dysregulation; "
            "ITK DEFICIENCY-NK/T LYMPHOMA CONTEXT: "
            "  ITK deficiency: AR biallelic LOF; rare primary immunodeficiency; "
            "  EBV-driven CD4+ T-cell and NK/T-cell LPD: 90% of cases involve NK/T-cell compartment; "
            "  HLH: 50% of ITK-deficient patients triggered by EBV; "
            "  EBV+ T-cell and NK/T-cell proliferation: systemic EBV+ T/NK-LPD; "
            "  HSCT MANDATORY before EBV exposure — pre-school timing critical; "
            "  Rituximab INEFFECTIVE: T-cell and NK-cell disease; "
            "CLINICAL RULES FOR ITK-NK/T LYMPHOMA: "
            "  HSCT MANDATORY before EBV: prophylactic pre-exposure timing; "
            "  EBV-seropositive trigger: urgent HSCT escalation; "
            "  Rituximab INEFFECTIVE: T/NK-cell disease (not B-cell); "
            "  HLH-2004 protocol: bridging before HSCT; "
            "  Immunoglobulin replacement: if hypogammaglobulinemia; "
            "  L-asparaginase-based chemotherapy for NK/T-lymphoma (SMILE/DDGP); "
            "  Radiation: ABSOLUTE CI in established NK/T-lymphoma pre-HSCT; "
            "KEY VARIANTS: "
            "  PH domain: pGly13Arg (PIP3 binding impaired); "
            "  KD: pArg477Trp (catalytic dead); "
            "  Truncating: complete LOF severe; "
            "SURVEILLANCE: "
            "  EBV PCR monthly; CBC + T/NK-cell immunophenotyping quarterly; "
            "  Ferritin for HLH monitoring; "
            "  All siblings: ITK genetic testing immediately; "
            "TARGETED Rx: "
            "  HSCT allo-SCT curative; pre-EBV timing critical; "
            "  HLH-2004 etoposide bridge; "
            "  L-asparaginase-based (SMILE/DDGP) for NK/T-lymphoma; "
            "  IVIG if IgG <4 g/L; "
            "  AVOID rituximab: T/NK-cell disease"
        ),
        "syndrome": "ITK Deficiency — EBV-susceptibility NK/T-cell LPD",
        "inheritance": "AR LOF; biallelic",
        "nkt_risk": "EBV-NK-T-LPD 90pct NK/T involvement; HLH 50pct; systemic-EBV-T-NK-LPD",
        "pathognomonic": "EBV-triggered-NK-T-LPD-90pct; systemic-EBV-CD4-T-NK-proliferation",
        "key_rule": "HSCT-MANDATORY-BEFORE-EBV; Rituximab-INEFFECTIVE-T-NK-cell; HLH-50pct-bridge-MANDATORY",
        "key_avoid": "Rituximab ineffective T/NK disease; delay HSCT until post-EBV worsens prognosis",
        "surveillance": "EBV PCR monthly; CBC+T-NK immunophenotyping quarterly; ferritin HLH trend",
        "targeted_rx": "HSCT curative; HLH-2004 bridge; L-asp SMILE/DDGP; IVIG IgG<4; AVOID rituximab",
    },
    {
        "gene": "GATA2",
        "protein": (
            "GATA2 -- 3q21.3 AD LOF -- 480aa -- "
            "GATA2-50kDa-Zinc-Finger-TF-"
            "NTD-ZnF1-ZnF2-CTD-"
            "NK-Lymphopenia-PATHOGNOMONIC-NK-T-Lymphoma-EBV-CMV-"
            "GATA2-OMIM-137295"
        ),
        "locus": "3q21.3",
        "protein_size": (
            "480 aa / ~50 kDa / 3q21.3 GATA2 NK/T lymphoma molecular context: "
            "STRUCTURE: "
            "  480 aa / ~50 kDa; zinc finger transcription factor; GATA family member; "
            "  N-terminal transactivation domain (NTD, aa 1-200); "
            "  Zinc finger 1 (ZnF1, aa 201-228): protein-protein interaction; FOG binding; "
            "  Zinc finger 2 (ZnF2, aa 301-325): DNA binding (GATA motif recognition); "
            "  C-terminal domain (CTD, aa 326-480): transactivation; "
            "  GATA2 governs NK-cell, DC, and monocyte specification in bone marrow; "
            "  GATA2 LOF → NK-cell lymphopenia (NK cells <0.05 × 10⁹/L); "
            "  GATA2 LOF → susceptibility to EBV/CMV/HPV/NTM infections; "
            "  NK-cell and NK/T-cell lymphoma susceptibility through impaired NK surveillance; "
            "GATA2 DEFICIENCY-NK/T LYMPHOMA CONTEXT: "
            "  GATA2 deficiency (MonoMAC/Emberger syndrome): AD haploinsufficiency; "
            "  NK-lymphopenia: hallmark — flow cytometry NK <0.05 × 10⁹/L PATHOGNOMONIC; "
            "  NK/T-cell lymphoma: NK-cell surveillance failure → EBV/CMV-driven NK-T-LPD; "
            "  MDS transformation risk: 30-70x highest among all GATA2-deficiency manifestations; "
            "  EBV+CMV quarterly monitoring MANDATORY; "
            "  HSCT curative for MDS and NK/T-LPD; "
            "CLINICAL RULES FOR GATA2-NK/T LYMPHOMA: "
            "  NK-lymphopenia: flow cytometry NK <0.05 × 10⁹/L — PATHOGNOMONIC; "
            "  EBV PCR quarterly monitoring: mandatory from diagnosis; "
            "  CMV: quarterly monitoring; ganciclovir prophylaxis if CMV+ or immunosuppressed; "
            "  HPV vaccination: mandatory, avoid cervical/anal Ca; "
            "  NTM prophylaxis: azithromycin if CD4 <200; "
            "  HSCT: curative for MDS and NK/T-LPD — do NOT delay if MDS/NK-T-lymphoma develops; "
            "  Lymphedema (Emberger): compression stockings; "
            "KEY VARIANTS: "
            "  ZnF2 missense: pThr354Met, pCys382Arg (DNA binding impaired); "
            "  ZnF1 missense: pArg204Gln, pPhe217Ile (FOG binding impaired); "
            "  Truncating throughout NTD+ZnF2: complete haploinsufficiency; "
            "  Large deletions at 3q21 — MLPA recommended; "
            "SURVEILLANCE: "
            "  EBV PCR quarterly; CMV PCR quarterly; HPV genotyping annually; "
            "  CBC + NK-cell flow cytometry quarterly; MDS screening: BM biopsy annually; "
            "  Lymphedema: annual doppler; "
            "TARGETED Rx: "
            "  HSCT allo-SCT: curative (MDS+NK/T-LPD); do NOT delay; "
            "  Ganciclovir/valganciclovir: CMV prophylaxis; "
            "  HPV vaccination; azithromycin NTM; "
            "  G-CSF: bridging for neutropenia pre-HSCT; "
            "  L-asparaginase-based (SMILE/DDGP) for NK/T-lymphoma"
        ),
        "syndrome": "GATA2 Deficiency / MonoMAC / Emberger Syndrome",
        "inheritance": "AD LOF heterozygous",
        "nkt_risk": "NK-lymphopenia PATHOGNOMONIC; NK-T-lymphoma EBV-CMV-driven; MDS 30-70x also",
        "pathognomonic": "NK-lymphopenia-<0.05-PATHOGNOMONIC; Emberger-lymphedema; MonoMAC-NTM-HPV-EBV",
        "key_rule": "NK-LYMPHOPENIA-PATHOGNOMONIC; EBV-CMV-Q3M-MANDATORY; HSCT-CURATIVE-DO-NOT-DELAY",
        "key_avoid": "Delay HSCT worsens MDS and NK/T-LPD prognosis; NTM untreated fatal",
        "surveillance": "EBV+CMV PCR quarterly; NK-cell flow Q3M; BM biopsy annual MDS screen; HPV annual",
        "targeted_rx": "HSCT curative; ganciclovir CMV; HPV vaccine; G-CSF bridge; L-asp NK/T-Rx",
    },
    {
        "gene": "PIK3CD",
        "protein": (
            "PI3K-delta -- 1p36.22 AD GOF -- 1044aa -- "
            "PI3K-delta-119kDa-Class-IA-PI3K-"
            "ABD-RBD-C2-Helical-KD-"
            "APDS1-EBV-NK-T-Lymphoma-10-15pct-Leniolisib-FDA2023-"
            "PIK3CD-OMIM-615513"
        ),
        "locus": "1p36.22",
        "protein_size": (
            "1044 aa / ~119 kDa / 1p36.22 PIK3CD NK/T lymphoma molecular context: "
            "STRUCTURE: "
            "  1044 aa / ~119 kDa; catalytic subunit of Class IA PI3K; "
            "  ABD (adapter binding domain, aa 1-108): p85 regulatory subunit binding; "
            "  RBD (Ras binding domain, aa 148-225): Ras/GTPase interaction; "
            "  C2 domain (aa 226-408): membrane binding; "
            "  Helical domain (aa 409-511): structural scaffold; "
            "  Kinase domain (KD, aa 512-1044): catalytic; PIP3 production; "
            "  GOF activating variants → constitutive PI3K-delta activation → PIP3 accumulation; "
            "  PI3K-delta hyperactivation → FOXO1 exclusion → B-cell mTOR→ senescence; "
            "  PI3K-delta GOF → impaired EBV CTL surveillance → EBV-driven NK/T-LPD; "
            "APDS1-NK/T LYMPHOMA CONTEXT: "
            "  APDS1 (Activated PI3K-delta Syndrome Type 1): AD gain-of-function; "
            "  EBV-driven NK/T-cell lymphoma: 10-15% of APDS1 patients; "
            "  EBV+CMV co-infection susceptibility: NK-cell cytotoxicity impaired; "
            "  Leniolisib (PI3K-delta inhibitor): FDA2023 — first approved PI3K-delta inhibitor; "
            "  Live vaccines: CI absolutely (OPV, varicella, MMR, yellow fever, BCG); "
            "  Acyclovir: mandatory prophylaxis given EBV/CMV susceptibility; "
            "CLINICAL RULES FOR PIK3CD-NK/T LYMPHOMA: "
            "  Leniolisib FDA2023: 70 mg twice daily — first-line PI3K-delta inhibitor; "
            "  Live vaccines CI: OPV, varicella, MMR, BCG, yellow fever — absolutely CI; "
            "  Acyclovir prophylaxis: 10 mg/kg/day mandatory from diagnosis; "
            "  IVIG: if IgG <4 g/L — common in APDS1; "
            "  Sirolimus (mTOR inhibitor): alternative; less selective; "
            "  Idelalisib (PI3K-delta inhibitor): older agent — hepatotoxicity risk; "
            "  L-asparaginase-based for confirmed NK/T-lymphoma; "
            "KEY VARIANTS: "
            "  Most common: pGlu1021Lys (KD domain — helical activating variant); "
            "  pThr442Pro, pTyr602Cys (alternative activating); "
            "  Less common KD: pMet1015Ile, pGly981Arg; "
            "SURVEILLANCE: "
            "  EBV+CMV PCR quarterly; Ig levels quarterly; "
            "  LFTs quarterly (leniolisib hepatotoxicity); "
            "  CBC + lymphocyte subsets quarterly; "
            "TARGETED Rx: "
            "  Leniolisib 70mg BID (FDA2023 approved); "
            "  IVIG if IgG <4 g/L; "
            "  Acyclovir 10mg/kg/day prophylaxis; "
            "  L-asparaginase-based (SMILE/DDGP) for confirmed NK/T-lymphoma; "
            "  HSCT: reserved for treatment-refractory NK/T-LPD"
        ),
        "syndrome": "Activated PI3K-delta Syndrome Type 1 (APDS1)",
        "inheritance": "AD GOF heterozygous",
        "nkt_risk": "EBV-NK-T-lymphoma 10-15pct; EBV+CMV-susceptibility; impaired NK-CTL-surveillance",
        "pathognomonic": "APDS1-senescent-B-cells-T-cell-EBV+CMV-susceptibility; recurrent-sinopulmonary",
        "key_rule": "Leniolisib-FDA2023-FIRST-LINE; LIVE-VACCINES-CI-ABSOLUTELY; Acyclovir-MANDATORY",
        "key_avoid": "Live vaccines (OPV/varicella/MMR/BCG/YF) absolutely CI; idelalisib older hepatotoxic",
        "surveillance": "EBV+CMV PCR Q3M; Ig quarterly; LFTs quarterly leniolisib; CBC+lymph subsets Q3M",
        "targeted_rx": "Leniolisib 70mg BID FDA2023; IVIG IgG<4; acyclovir prophylaxis; L-asp NK/T-Rx",
    },
    {
        "gene": "STAT5B",
        "protein": (
            "STAT5B -- 17q21.2 AD GOF -- 787aa -- "
            "STAT5B-92kDa-JAK-STAT-TF-"
            "Coiled-coil-DBD-Linker-SH2-TAD-"
            "NK-LGL-NK-T-Lymphoma-Cytopenia-HALLMARK-Ruxolitinib-"
            "STAT5B-OMIM-604260"
        ),
        "locus": "17q21.2",
        "protein_size": (
            "787 aa / ~92 kDa / 17q21.2 STAT5B NK/T lymphoma molecular context: "
            "STRUCTURE: "
            "  787 aa / ~92 kDa; signal transducer and activator of transcription 5B; "
            "  N-terminal domain (aa 1-122): oligomerisation; "
            "  Coiled-coil domain (aa 123-318): nuclear export; "
            "  DNA-binding domain (DBD, aa 319-483): GAS element binding (TTCNNNGAA); "
            "  Linker domain (aa 484-567); "
            "  SH2 domain (aa 568-656): pTyr-JAK binding; STAT5B activation; "
            "  Transactivation domain (TAD, aa 657-787): pTyr694 site; transcription; "
            "  STAT5B GOF → constitutive pTyr694 activation → NK-cell proliferation; "
            "  STAT5B GOF → IL-2/IL-15-driven NK-LGL expansion; "
            "  STAT5B GOF → NK/T-cell lymphoma via sustained proliferative signalling; "
            "STAT5B GOF-NK/T LYMPHOMA CONTEXT: "
            "  STAT5B GOF (gain-of-function) mutations: AD; lymphoproliferative predisposition; "
            "  NK-LGL (large granular lymphocyte) leukemia: most common manifestation — cytopenia HALLMARK; "
            "  NK/T-cell lymphoma: 15-25% of STAT5B GOF patients with progressive disease; "
            "  Cytopenia (cytopenias): hallmark — neutropenia, anemia, thrombocytopenia; "
            "  Ruxolitinib (JAK1/2i): FDA2023 approved for STAT3-GOF; active in STAT5B GOF; "
            "  Methotrexate AVOID: NK-LGL reactivation risk; "
            "CLINICAL RULES FOR STAT5B-NK/T LYMPHOMA: "
            "  Ruxolitinib JAKi: FIRST LINE for NK-LGL/NK-T-lymphoma; cytopenia control; "
            "  Methotrexate AVOID: NK-LGL reactivation and progression risk; "
            "  Cytopenia HALLMARK: monitor CBC weekly during treatment; "
            "  Cyclosporin A: alternative immunosuppressive for NK-LGL; "
            "  STAT5B sequencing (germline): mandatory family cascade testing; "
            "  HSCT: for transformation/progressive disease refractory to JAKi; "
            "KEY VARIANTS: "
            "  pTyr665Phe (most common GOF — SH2 domain, constitutive activation); "
            "  pAsp719Glu (GOF — reduces dephosphorylation); "
            "  pArg659Gln (TAD activation); "
            "  GOF mechanism: reduced pTyr dephosphorylation by SHP1/SHP2; "
            "SURVEILLANCE: "
            "  CBC weekly during active treatment; monthly in remission; "
            "  LGL count: peripheral blood LGL >2 × 10⁹/L diagnostic; "
            "  STAT5B germline testing: all first-degree relatives; "
            "TARGETED Rx: "
            "  Ruxolitinib (JAK1/2i): 10-20 mg BID — NK-LGL/NK-T-lymphoma; "
            "  Cyclosporin A (CsA): alternative immunosuppressive; "
            "  Low-dose methotrexate: AVOID (NK-LGL reactivation); "
            "  Cyclophosphamide: for aggressive NK/T-lymphoma; "
            "  HSCT: transformation/JAKi-refractory disease"
        ),
        "syndrome": "STAT5B GOF Syndrome / NK-LGL Leukemia",
        "inheritance": "AD GOF heterozygous",
        "nkt_risk": "NK-LGL NK-T-lymphoma; cytopenia HALLMARK; Ruxolitinib-active; 15-25pct NK/T-lymphoma",
        "pathognomonic": "NK-LGL-expansion-cytopenia-HALLMARK; LGL>2x10-9-L-DIAGNOSTIC; STAT5B-GOF-germline",
        "key_rule": "Ruxolitinib-FIRST-LINE-NK-LGL; CYTOPENIA-HALLMARK-monitor-CBC; METHOTREXATE-AVOID",
        "key_avoid": "Methotrexate (NK-LGL reactivation); avoid JAK inhibitor holiday without taper",
        "surveillance": "CBC weekly active treatment; LGL count peripheral blood; STAT5B family testing",
        "targeted_rx": "Ruxolitinib 10-20mg BID first-line; cyclosporin A alternative; HSCT refractory",
    },
    {
        "gene": "RASGRP1",
        "protein": (
            "RASGRP1 -- 15q14 AR LOF -- 795aa -- "
            "RASGRP1-90kDa-RAS-GRP-"
            "REM-CDC25-EF-CRD-DAG-"
            "EBV-T-NK-LPD-RAS-MAPK-dysregulation-HSCT-beneficial-"
            "RASGRP1-OMIM-603962"
        ),
        "locus": "15q14",
        "protein_size": (
            "795 aa / ~90 kDa / 15q14 RASGRP1 NK/T lymphoma molecular context: "
            "STRUCTURE: "
            "  795 aa / ~90 kDa; RAS guanyl nucleotide-releasing protein 1; "
            "  REM domain (aa 1-50): regulatory; "
            "  CDC25 (catalytic) homology domain (aa 51-337): GEF activity for RAS/RAP; "
            "  EF-hand calcium-binding motifs (aa 502-537): Ca2+-regulated activation; "
            "  Cysteine-rich domain (CRD, aa 546-619): DAG/phorbol ester binding; "
            "  DAG (diacylglycerol) binding: membrane translocation; "
            "  RASGRP1 LOF → impaired T-cell and NK-cell TCR/NCR signalling; "
            "  RASGRP1 LOF → RAS-MAPK dysregulation → T-cell selection defects; "
            "  EBV-susceptibility: impaired cytotoxic T/NK responses → EBV-driven NK-T-LPD; "
            "RASGRP1 DEFICIENCY-NK/T LYMPHOMA CONTEXT: "
            "  RASGRP1 deficiency: AR biallelic LOF; rare primary immunodeficiency; "
            "  EBV-driven T/NK-LPD: established susceptibility via impaired T/NK cytotoxicity; "
            "  Autoimmune features: SLE-like autoimmunity, cytopenias (immune dysregulation); "
            "  T-cell lymphopenia: CD4+ and CD8+ reduced; "
            "  NK-cell dysfunction: reduced cytotoxicity despite normal NK count; "
            "  HSCT beneficial: corrects immune dysregulation and NK-T-LPD risk; "
            "  Ruxolitinib (JAK1/2i): controls inflammatory cytokine storm; "
            "CLINICAL RULES FOR RASGRP1-NK/T LYMPHOMA: "
            "  HSCT beneficial: corrects NK-cell dysfunction and T-cell dysregulation; "
            "  Ruxolitinib: cytokine storm control + NK-T-LPD inflammation; "
            "  Autoimmune surveillance: ANA, dsDNA, complement annually; "
            "  EBV PCR: monthly monitoring; "
            "  Hydroxychloroquine: SLE-like autoimmune features; "
            "  L-asparaginase-based for confirmed NK/T-lymphoma; "
            "KEY VARIANTS: "
            "  Frameshift/truncating: CDC25 domain disruption; "
            "  pAsp248Gly (CDC25 catalytic residue): ablates GEF activity; "
            "  pArg519Trp (EF-hand Ca2+ binding impaired); "
            "  Compound heterozygous: partial LOF combinations; "
            "SURVEILLANCE: "
            "  EBV PCR monthly; CBC monthly; ANA/dsDNA annually; "
            "  NK-cell functional assay (NK cytotoxicity): annually; "
            "  T/NK-cell immunophenotyping: quarterly; "
            "TARGETED Rx: "
            "  HSCT allo-SCT: corrects immune dysregulation + NK-T-LPD; "
            "  Ruxolitinib JAKi: cytokine storm + NK-T-LPD inflammation; "
            "  Hydroxychloroquine: SLE-like features; "
            "  L-asparaginase-based (SMILE/DDGP) for confirmed NK/T-lymphoma; "
            "  Etoposide: HLH bridging; "
            "  IVIG: if IgG <4 g/L"
        ),
        "syndrome": "RASGRP1 Deficiency — EBV-susceptibility NK/T-LPD",
        "inheritance": "AR LOF; biallelic",
        "nkt_risk": "EBV-T-NK-LPD; RAS-MAPK-dysregulation; autoimmune-SLE-like; HSCT-beneficial",
        "pathognomonic": "EBV-T-NK-LPD-RAS-MAPK; SLE-like-autoimmunity-NK-cell-dysfunction; rare",
        "key_rule": "HSCT-BENEFICIAL; Ruxolitinib-cytokine-storm; EBV-PCR-monthly; autoimmune-surveillance",
        "key_avoid": "Delay HSCT when NK-T-LPD established worsens prognosis",
        "surveillance": "EBV PCR monthly; CBC monthly; ANA/dsDNA annual; NK cytotoxicity assay annual",
        "targeted_rx": "HSCT corrects immune dysregulation; ruxolitinib JAKi; hydroxychloroquine; L-asp NK/T-Rx",
    },
]

KEY_CLINICAL_DISTINCTIONS = [
    "SH2D1A: XLP1 MALES-ONLY; iNKT-ABSENT-PATHOGNOMONIC; EBV-NK-T-LPD-50-80x-HIGHEST; HSCT-MANDATORY-BEFORE-EBV",
    "XIAP: XLP2 MALES-ONLY; NOT-EBV-EXCLUSIVE; IBD+HLH-COMBINATION-PATHOGNOMONIC; Ruxolitinib-bridge-HSCT",
    "CD27: AR; EBV-NK-T-LPD-30-50x-HIGHEST; Rituximab-INEFFECTIVE-NK-CD20-negative; IVIG-mandatory-IgG<4",
    "ITK: AR; EBV-NK-T-LPD-90pct; HLH-50pct; Rituximab-INEFFECTIVE-T-NK-cell; HSCT-MANDATORY-BEFORE-EBV",
    "GATA2: AD; NK-LYMPHOPENIA-PATHOGNOMONIC; EBV-CMV-Q3M-MANDATORY; HSCT-CURATIVE-DO-NOT-DELAY",
    "PIK3CD: AD-GOF; APDS1; EBV-NK-T-lymphoma-10-15pct; Leniolisib-FDA2023-FIRST-LINE; LIVE-VACCINES-CI-ABSOLUTELY",
    "STAT5B: AD-GOF; NK-LGL-NK-T-lymphoma; CYTOPENIA-HALLMARK; Ruxolitinib-JAKi-FIRST-LINE; METHOTREXATE-AVOID",
    "RASGRP1: AR; EBV-T-NK-LPD; RAS-MAPK-dysregulation; SLE-like-autoimmunity; HSCT-BENEFICIAL",
]


def _make_cohort(gene_idx: int, seed: int) -> dict:
    """Generate a 40-patient NK/T-cell lymphoma cohort for one predisposition gene."""
    rng = random.Random(seed)
    g = ATLAS_GENES[gene_idx]

    age_params = {
        0: (18, 8),   # SH2D1A — XLP1 childhood/young adult EBV-triggered
        1: (22, 10),  # XIAP — XLP2 variable onset
        2: (20, 9),   # CD27 — AR early onset EBV susceptibility
        3: (16, 8),   # ITK — AR early childhood
        4: (35, 12),  # GATA2 — AD variable later onset
        5: (28, 10),  # PIK3CD — APDS1 childhood/young adult
        6: (52, 12),  # STAT5B — NK-LGL adult onset
        7: (25, 10),  # RASGRP1 — AR variable onset
    }
    mu, sd = age_params[gene_idx]
    ages = [max(2, min(75, int(rng.gauss(mu, sd)))) for _ in range(40)]

    # NK/T lymphoma Ann Arbor staging (I-II localised vs III-IV disseminated)
    stage_dist = {
        0: [0.20, 0.25, 0.25, 0.30],  # SH2D1A — XLP1 disseminated
        1: [0.25, 0.30, 0.25, 0.20],  # XIAP — XLP2 variable
        2: [0.15, 0.25, 0.30, 0.30],  # CD27 — disseminated EBV
        3: [0.20, 0.25, 0.25, 0.30],  # ITK — disseminated EBV
        4: [0.30, 0.30, 0.25, 0.15],  # GATA2 — variable
        5: [0.35, 0.30, 0.20, 0.15],  # PIK3CD — more localised initially
        6: [0.40, 0.35, 0.15, 0.10],  # STAT5B — NK-LGL leukemic phase
        7: [0.25, 0.30, 0.25, 0.20],  # RASGRP1 — variable
    }
    bins = stage_dist[gene_idx]
    stage_i = int(round(40 * bins[0]))
    stage_ii = int(round(40 * bins[1]))
    stage_iii = int(round(40 * bins[2]))
    stage_iv = 40 - stage_i - stage_ii - stage_iii

    # EBV status (EBER positivity)
    eber_pos = {
        0: 38, 1: 28, 2: 36, 3: 38, 4: 30, 5: 26, 6: 14, 7: 30
    }[gene_idx]
    eber_neg = 40 - eber_pos

    # Treatment first-line
    tx_l_asp = {0: 28, 1: 20, 2: 26, 3: 28, 4: 22, 5: 18, 6: 10, 7: 22}[gene_idx]
    tx_ruxolitinib = {0: 0, 1: 12, 2: 0, 3: 0, 4: 4, 5: 6, 6: 22, 7: 8}[gene_idx]
    tx_hsct = {0: 30, 1: 24, 2: 28, 3: 30, 4: 24, 5: 12, 6: 10, 7: 18}[gene_idx]

    # Outcomes
    cr_rate = {0: 0.40, 1: 0.52, 2: 0.45, 3: 0.42, 4: 0.55, 5: 0.60, 6: 0.65, 7: 0.50}[gene_idx]
    os_2yr = {0: 0.48, 1: 0.60, 2: 0.52, 3: 0.50, 4: 0.65, 5: 0.70, 6: 0.72, 7: 0.58}[gene_idx]

    hlh_count = {0: 28, 1: 20, 2: 18, 3: 20, 4: 10, 5: 8, 6: 4, 7: 14}[gene_idx]
    hsct_performed = {0: 30, 1: 24, 2: 26, 3: 28, 4: 22, 5: 10, 6: 8, 7: 16}[gene_idx]

    return {
        "gene": g["gene"],
        "locus": g["locus"],
        "inheritance": g["inheritance"],
        "n": 40,
        "seed": seed,
        "ages": ages,
        "mean_age": round(sum(ages) / 40, 1),
        "stage_i": stage_i,
        "stage_ii": stage_ii,
        "stage_iii": stage_iii,
        "stage_iv": stage_iv,
        "eber_pos": eber_pos,
        "eber_neg": eber_neg,
        "tx_l_asp": tx_l_asp,
        "tx_ruxolitinib": tx_ruxolitinib,
        "tx_hsct": tx_hsct,
        "cr_rate": cr_rate,
        "os_2yr": os_2yr,
        "hlh_count": hlh_count,
        "hsct_performed": hsct_performed,
        "nkt_risk": g["nkt_risk"],
        "pathognomonic": g["pathognomonic"],
        "key_rule": g["key_rule"],
        "key_avoid": g["key_avoid"],
        "targeted_rx": g["targeted_rx"],
    }


def generate_overview() -> dict:
    cohorts = [_make_cohort(i, SEED_BASE + i) for i in range(8)]

    total_patients = sum(c["n"] for c in cohorts)
    mean_age_overall = round(sum(c["mean_age"] * c["n"] for c in cohorts) / total_patients, 1)

    total_stage_i = sum(c["stage_i"] for c in cohorts)
    total_stage_ii = sum(c["stage_ii"] for c in cohorts)
    total_stage_iii = sum(c["stage_iii"] for c in cohorts)
    total_stage_iv = sum(c["stage_iv"] for c in cohorts)
    total_eber_pos = sum(c["eber_pos"] for c in cohorts)
    total_l_asp = sum(c["tx_l_asp"] for c in cohorts)
    total_ruxolitinib = sum(c["tx_ruxolitinib"] for c in cohorts)
    total_hsct = sum(c["tx_hsct"] for c in cohorts)
    total_hlh = sum(c["hlh_count"] for c in cohorts)
    total_hsct_done = sum(c["hsct_performed"] for c in cohorts)

    avg_cr = round(sum(c["cr_rate"] for c in cohorts) / 8 * 100, 1)
    avg_os2yr = round(sum(c["os_2yr"] for c in cohorts) / 8 * 100, 1)

    per_gene = [
        {
            "gene": c["gene"],
            "locus": c["locus"],
            "n": c["n"],
            "mean_age": c["mean_age"],
            "stage_iv_pct": round(c["stage_iv"] / 40 * 100),
            "eber_pos_pct": round(c["eber_pos"] / 40 * 100),
            "l_asp_pct": round(c["tx_l_asp"] / 40 * 100),
            "hsct_pct": round(c["tx_hsct"] / 40 * 100),
            "cr_pct": round(c["cr_rate"] * 100, 1),
            "os_2yr_pct": round(c["os_2yr"] * 100, 1),
            "hlh_count": c["hlh_count"],
            "nkt_risk": c["nkt_risk"],
            "key_rule": c["key_rule"],
            "targeted_rx": c["targeted_rx"],
        }
        for c in cohorts
    ]

    return {
        "atlas": "Hereditary-NK-T-Cell-Lymphoma-Predisposition-Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "n_genes": 8,
        "total_patients": total_patients,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "mean_age_overall": mean_age_overall,
        "stage_distribution": {
            "stage_i": total_stage_i,
            "stage_ii": total_stage_ii,
            "stage_iii": total_stage_iii,
            "stage_iv": total_stage_iv,
            "localised_pct": round((total_stage_i + total_stage_ii) / total_patients * 100),
            "disseminated_pct": round((total_stage_iii + total_stage_iv) / total_patients * 100),
        },
        "ebv_distribution": {
            "eber_pos": total_eber_pos,
            "eber_neg": total_patients - total_eber_pos,
            "eber_pos_pct": round(total_eber_pos / total_patients * 100),
        },
        "treatment_distribution": {
            "l_asparaginase": total_l_asp,
            "ruxolitinib": total_ruxolitinib,
            "hsct": total_hsct,
        },
        "outcomes": {
            "avg_cr_pct": avg_cr,
            "avg_os_2yr_pct": avg_os2yr,
            "total_hlh": total_hlh,
            "hlh_pct": round(total_hlh / total_patients * 100, 1),
            "total_hsct_done": total_hsct_done,
            "hsct_done_pct": round(total_hsct_done / total_patients * 100, 1),
        },
        "key_clinical_distinctions": KEY_CLINICAL_DISTINCTIONS,
        "per_gene": per_gene,
    }


def generate_breakdown() -> dict:
    cohorts = [_make_cohort(i, SEED_BASE + i) for i in range(8)]
    gene_data = {c["gene"]: c for c in cohorts}
    return {
        "atlas": "Hereditary-NK-T-Cell-Lymphoma-Predisposition-Atlas",
        "gene_data": gene_data,
        "atlas_genes": [
            {
                "gene": g["gene"],
                "locus": g["locus"],
                "protein": g["protein"],
                "syndrome": g["syndrome"],
                "inheritance": g["inheritance"],
                "nkt_risk": g["nkt_risk"],
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
            "term": f"{g['gene']} — {g['syndrome']}",
            "definition": (
                f"{g['locus']} {g['inheritance']} — "
                f"{g['nkt_risk']} — "
                f"Pathognomonic: {g['pathognomonic']} — "
                f"Key rule: {g['key_rule']} — "
                f"Avoid: {g['key_avoid']} — "
                f"Surveillance: {g['surveillance']} — "
                f"Targeted Rx: {g['targeted_rx']}"
            ),
        })
        defs.append({
            "term": f"{g['gene']} molecular size",
            "definition": g["protein_size"],
        })
    defs.append({
        "term": "KEY CLINICAL DISTINCTIONS — Hereditary NK/T-Cell Lymphoma Predisposition 8-Gene Atlas",
        "definition": " | ".join(KEY_CLINICAL_DISTINCTIONS),
    })
    return {"atlas": "Hereditary-NK-T-Cell-Lymphoma-Predisposition-Atlas", "definitions": defs}
