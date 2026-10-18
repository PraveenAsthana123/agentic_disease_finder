#!/usr/bin/env python3
"""Hereditary-EBV-XLP-NK-T-Cell-Susceptibility-Atlas -- Complete 8-Gene Reference
EBV Susceptibility / X-Linked Lymphoproliferative / NK-T Cell Deficiency Diseases
(SH2D1A/XLP1-NKT-Absent-PATHOGNOMONIC-EBV-HLH-Fatal-SAP-Absent-Flow-Rituximab-HSCT,
 XIAP/XLP2-EBV-IBD-Dual-Phenotype-PATHOGNOMONIC-XIAP-Absent-Flow-Iloprost,
 ITK/EBV-CID-CD4-Lymphopenia-T-Cell-Activation-Defect-5q33.3,
 MAGT1/XMEN-Syndrome-XLR-CD4-Glycosylation-Mg2plus-Defect-EBV-Neoplasia,
 CD27/EBV-Lymphoproliferation-HLH-NK-Cytotoxicity-Normal-DDx-XLP1-XIAP,
 CD70/EBV-T-Cell-Exhaustion-PATHOGNOMONIC-Lymphoma-Very-High-Risk,
 TNFRSF9/4-1BB-EBV-HLH-Lymphoma-NK-CD8-Dysfunction,
 RASGRP1/CID-EBV-RAS-MAPK-Activation-Defect-Autoimmunity-Overlap):

SH2D1A (Signalling lymphocytic activation molecule (SLAM)-associated protein / SAP / XLP1; 128aa; Xq25; XLR;
         SH2D1A encodes SAP (SLAM-associated protein), a small SH2 domain-only adaptor expressed in T, NKT, and NK cells;
         SAP links SLAM family receptors (SLAMF1/CD150, SLAMF4/2B4, SLAMF6/Ly108) to downstream signalling → cytotoxicity + antibody switch;
         XLP1 (X-linked lymphoproliferative disease type 1): MOST COMMON EBV susceptibility PID;
         PATHOGNOMONIC BIOMARKER: NKT cells (CD3+Vα24+Vβ11+, invariant NKT) = ABSENT in flow cytometry — zero NKT cells;
         NKT cells are almost uniquely SAP-dependent for development and survival → total NKT cell absence in all XLP1 patients;
         EBV triggers: FATAL HAEMOPHAGOCYTIC LYMPHOHISTIOCYTOSIS (HLH) — untreated mortality >70%; median age at HLH = 5-8y;
         Other XLP1 features: dysgammaglobulinemia (Ig isotype class-switch defect — absent IgG subclasses), lymphoma (Burkitt-type most common);
         SAP absent on intracellular flow cytometry → DIAGNOSTIC TEST: SAP protein flow cytometry + SH2D1A sequencing;
         Treatment: Rituximab (anti-CD20) for EBV-HLH to deplete EBV-infected B cells — IMMEDIATELY; then HSCT for definitive cure;
         HLH diagnostic criteria: ferritin >10,000, haemophagocytosis, splenomegaly, cytopenia, NK dysfunction, elevated sIL-2Rα;
         seed SEED_BASE+0).
XIAP   (X-linked inhibitor of apoptosis protein / BIRC4 / XLP2; 497aa; Xq25; XLR;
         XIAP encodes the X-linked inhibitor of apoptosis protein, the largest IAP family member with three BIR domains + RING domain;
         XIAP normally inhibits caspase-3, caspase-7, and caspase-9 → prevents apoptosis; BIR2 inhibits caspase-3; BIR3 inhibits caspase-9;
         XIAP also activates NF-κB via RIPK2/NOD2 signalling → critical for intestinal epithelial homeostasis;
         XLP2 (X-linked lymphoproliferative disease type 2): second most common XLP PID;
         PATHOGNOMONIC DUAL PHENOTYPE: EBV-triggered HLH + INFLAMMATORY BOWEL DISEASE — this combination is PATHOGNOMONIC for XIAP deficiency;
         IBD in XIAP: severe early-onset IBD (Crohn's-like, perianal disease, fistulae) — NOD2/RIPK2 pathway loss in gut epithelium;
         XIAP absent on intracellular flow cytometry → DIAGNOSTIC TEST: XIAP protein flow cytometry (XIAP absent in XLP2 males);
         Unlike XLP1: NKT cells MAY BE PRESENT (partial NKT preservation distinguishes XLP2 from XLP1 on flow);
         Treatment: Iloprost (prostacyclin analogue) for cytopenias and splenomegaly in XIAP deficiency; HSCT curative;
         Perforin and SAP normal (DDx HLH subtypes); XIAP function absent = XLP2; SAP absent = XLP1;
         seed SEED_BASE+1).
ITK    (IL-2-inducible T-cell kinase; 620aa; 5q33.3; AR;
         ITK encodes IL-2-inducible T-cell kinase (ITK), a Tec family tyrosine kinase expressed in T cells, NKT cells, and mast cells;
         ITK is activated by T-cell receptor (TCR) + CD28 co-stimulation → phosphorylates PLCγ1 → activates NFAT + AP-1 + NF-κB;
         ITK is CRITICAL for NKT cell development and iNKT cell maturation (like SAP but via distinct kinase pathway);
         ITK DEFICIENCY: autosomal recessive CID with EBV susceptibility;
         CLINICAL: EBV-driven lymphoproliferative disease, Burkitt lymphoma, and HLH in childhood;
         CD4 T-cell lymphopenia (CD4 count reduced to 200-500 cells/μL) — NKT cells absent or severely reduced;
         Hypogammaglobulinemia with class-switch defect similar to XLP1;
         ITK deficiency is FEMALE-PRESENTING CID (no X-linkage — AR gene on 5q33.3) unlike XLP1/XLP2;
         Functional ITK assay: reduced phospho-PLCγ1 after TCR stimulation → diagnostic;
         Importantly distinguishes from XLP1: SAP normal, XIAP normal; ITK by sequencing/functional assay;
         Treatment: HSCT for EBV-driven lymphoproliferation; rituximab for acute EBV-HLH; IVIG for hypogammaglobulinemia;
         seed SEED_BASE+2).
MAGT1  (Magnesium transporter 1 / XMEN; 335aa; Xq21.1; XLR;
         MAGT1 encodes Magnesium transporter 1, an ER-localised Mg2+ channel required for free cytosolic Mg2+ flux after TCR activation;
         MAGT1 also functions as an oligosaccharyltransferase subunit → N-glycosylation of CD4, NKG2D, and other cell surface proteins;
         XMEN SYNDROME (X-linked immunodeficiency with Magnesium defect, EBV infection, and Neoplasia):
         CD4 T-CELL LYMPHOPENIA (CD4 <300 cells/μL) — MOST PROMINENT FEATURE; CD8 T cells near-normal;
         NKG2D (natural killer cell group 2D receptor) ABSENT on NK cells and CD8 T cells → impaired cytotoxicity (Mg2+-dependent glycosylation);
         EBV-driven lymphoproliferative disease: chronic active EBV infection, EBV lymphoma, Hodgkin lymphoma;
         UNIQUE BIOCHEMICAL CLUE: free intracellular Mg2+ level REDUCED in T cells (measurable with fluorescent Mg2+ probes);
         Mg2+ SUPPLEMENTATION (dietary/oral) can partially restore NKG2D expression → Level B evidence — pragmatic initial step;
         Distinguishes from XLP1: SAP present, NKT cells may be preserved, CD4 count very low (vs. XLP1 where CD4 often normal-ish but NKT absent);
         Treatment: Mg2+ supplementation; HSCT for severe EBV lymphoproliferation; antiviral (ganciclovir/valganciclovir) suppression;
         seed SEED_BASE+3).
CD27   (CD27 antigen / TNFRSF7; 260aa; 12p13.31; AR;
         CD27 encodes the TNF receptor superfamily member CD27, expressed on T cells, B cells, and NK cells;
         CD27 signalling through its ligand CD70 → CD27/CD70 co-stimulatory axis drives T-cell activation, B-cell differentiation, and NK cytotoxicity;
         CD70 (gene CD70) is the cognate ligand → bi-directional requirement: loss of either CD27 or CD70 impairs the full axis;
         CD27 DEFICIENCY: autosomal recessive EBV-susceptibility PID;
         CLINICAL: EBV-induced lymphoproliferative disease, HLH, aplastic anaemia, and Burkitt lymphoma;
         NK cytotoxicity is NORMAL in CD27 deficiency (distinguishes from XLP1 and XIAP where NK cytotoxicity often impaired);
         CD27 absent on T/B/NK cell surfaces → DIAGNOSTIC by flow cytometry (CD27 negative);
         Low IgG, absent switched memory B cells (CD27+IgD-) → CVID-like hypogammaglobulinemia;
         EBV-specific T-cell response defective (CD27/CD70 co-stimulation required for EBV-specific CD8 expansion);
         IVIG for hypogammaglobulinemia; rituximab for EBV-HLH; HSCT for recurrent severe EBV disease;
         seed SEED_BASE+4).
CD70   (CD70 antigen / CD27 ligand / TNFSF7; 193aa; 19p13.3; AR;
         CD70 encodes the TNF superfamily ligand CD70, the cognate ligand for CD27; transiently expressed on activated T cells, B cells, and dendritic cells;
         CD70-CD27 axis: critical for EBV-specific T-cell expansion, memory T-cell formation, and B-cell isotype switch;
         CD70 DEFICIENCY: autosomal recessive EBV-susceptibility PID (phenocopy of CD27 deficiency);
         PATHOGNOMONIC: EBV-SPECIFIC T-CELL EXHAUSTION — CD8 T cells fail to expand properly against EBV → chronic EBV viraemia;
         Progressive loss of CD27 expression on T cells due to chronic overstimulation (ligand absent → no signal → exhaustion);
         Very high lymphoma risk: EBV-associated lymphoma (Hodgkin and non-Hodgkin) because CD8 T cells cannot eliminate EBV-infected B cells;
         CD70 absent on activated T cells → flow cytometry after anti-CD3 stimulation;
         Distinguishes from CD27 deficiency: CD70 gene 19p13.3 vs CD27 gene 12p13.31 — requires sequencing panel;
         IVIG, rituximab, and HSCT similar indications to CD27 deficiency;
         Prophylactic antiviral therapy (valganciclovir) used to suppress EBV viraemia pending HSCT;
         seed SEED_BASE+5).
TNFRSF9 (4-1BB / CD137; 255aa; 1p36.23; AR;
         TNFRSF9 encodes 4-1BB (also known as CD137), a co-stimulatory TNF receptor expressed on activated T cells and NK cells;
         4-1BB signalling through its ligand 4-1BBL (TNFSF9) → activates NF-κB + PI3K → CD8 T-cell expansion, survival, and memory;
         4-1BB is critical for EBV-specific CD8 T-cell survival and long-term cytotoxic memory;
         4-1BB DEFICIENCY: autosomal recessive EBV susceptibility PID;
         CLINICAL: EBV-driven lymphoproliferative disease, Hodgkin lymphoma, HLH;
         NK cells and CD8 T cells show REDUCED CYTOTOXICITY against EBV-infected cells (4-1BB co-stimulation required for full NK/CD8 activation);
         4-1BB absent on activated T/NK surfaces → flow cytometry with anti-CD137 staining post-stimulation;
         Low switched memory B cells, hypogammaglobulinemia in a subset;
         Rare — only a few families reported; allelic series includes homozygous nonsense and compound heterozygous missense;
         Importantly: 4-1BB agonist antibodies (e.g. urelumab, utomilumab) are CONTRAINDICATED in 4-1BB deficiency (no receptor to target);
         HSCT curative; rituximab for acute EBV-HLH; IVIG for hypogammaglobulinemia;
         seed SEED_BASE+6).
RASGRP1 (RAS guanyl-releasing protein 1; 797aa; 15q14; AR;
         RASGRP1 encodes RAS guanyl-nucleotide-releasing protein 1, a DAG-regulated guanine nucleotide exchange factor (GEF) that activates RAS;
         RASGRP1 is activated downstream of TCR → PLCγ1 → DAG → RASGRP1 → activates H-RAS → ERK/MAPK signalling cascade;
         ERK/MAPK pathway essential for T-cell proliferation, positive thymic selection, and effector function;
         RASGRP1 DEFICIENCY: autosomal recessive combined immunodeficiency with EBV susceptibility and autoimmunity;
         COMBINED IMMUNODEFICIENCY FEATURES: CD4+ T-cell lymphopenia, reduced memory T cells, impaired T-cell proliferation to PHA;
         EBV SUSCEPTIBILITY: EBV-driven lymphoproliferation and HLH due to defective cytotoxic T-cell expansion;
         AUTOIMMUNITY OVERLAP: unique feature — autoimmune haemolytic anaemia, lupus-like manifestations, anti-nuclear antibodies;
         Functional assay: absent phospho-ERK after PMA/PHA stimulation → diagnostic;
         Reduced NK cell cytotoxicity (ERK-dependent for perforin degranulation);
         DISTINGUISHES FROM XLP1/XLP2: SAP normal, XIAP normal, ITK normal → WES/gene panel for RASGRP1;
         Treatment: HSCT for severe immunodeficiency; IVIG; careful immunosuppression for autoimmune manifestations (avoid agents triggering EBV);
         seed SEED_BASE+7).
"""

import random
import math

ATLAS_NAME   = "Hereditary-EBV-XLP-NK-T-Cell-Susceptibility-Atlas"
SEED_BASE    = 3990   # seeds 3990-3997 (8 genes × 40 patients each)
N_PER_GENE   = 40
TOTAL_N      = N_PER_GENE * 8

GENES = ["SH2D1A", "XIAP", "ITK", "MAGT1", "CD27", "CD70", "TNFRSF9", "RASGRP1"]

GENE_DATA = {
    "SH2D1A": {
        "full_name":             "Signalling lymphocytic activation molecule (SLAM)-associated protein (SAP / XLP1)",
        "aa":                    128,
        "locus":                 "Xq25",
        "kda":                   "15",
        "inheritance":           "XLR (X-linked recessive; males affected; female carriers asymptomatic)",
        "syndrome":              "XLP1 — X-linked lymphoproliferative disease type 1 (most common EBV-susceptibility PID)",
        "pathway":               "TCR-SLAM-SAP axis absent → NKT cell development fails completely → EBV-infected B cells unchecked → fatal HLH",
        "onset_range":           (1, 96),    # early childhood EBV HLH
        "ebv_hlh_pct":           0.70,       # EBV-triggered HLH
        "lymphoma_pct":          0.35,       # Burkitt lymphoma most common
        "dysgammaglobulinemia_pct": 0.40,    # Ig class-switch defect
        "nkt_absent_pct":        0.99,       # NKT cells absent — near-universal
        "sap_absent_flow_pct":   0.99,       # SAP absent on intracellular flow
        "rituximab_pct":         0.65,       # Rituximab for EBV-HLH
        "hsct_pct":              0.55,       # HSCT curative
        "ferritin_10k_pct":      0.68,       # HLH ferritin >10,000
        "pathognomonic":         "NKT CELLS (CD3+Vα24+Vβ11+) ABSENT on flow cytometry — PATHOGNOMONIC; SAP absent on intracellular flow; EBV-triggered fatal HLH in childhood; NKT absence almost exclusive to XLP1",
        "key_drug":              "Rituximab (anti-CD20) IMMEDIATELY for EBV-HLH to deplete EBV-infected B cells; followed by HSCT for cure; dexamethasone + etoposide for HLH (HLH-2004 protocol)",
    },
    "XIAP": {
        "full_name":             "X-linked inhibitor of apoptosis protein (BIRC4 / XIAP / XLP2)",
        "aa":                    497,
        "locus":                 "Xq25",
        "kda":                   "56",
        "inheritance":           "XLR (X-linked recessive; Xq25 — same locus region as SH2D1A)",
        "syndrome":              "XLP2 — X-linked lymphoproliferative disease type 2 (EBV-HLH + IBD dual phenotype PATHOGNOMONIC)",
        "pathway":               "XIAP absent → caspase-3/9 uninhibited → enhanced apoptosis + NF-κB/NOD2 signalling lost → gut homeostasis fails → IBD + HLH",
        "onset_range":           (2, 120),
        "ebv_hlh_pct":           0.65,
        "lymphoma_pct":          0.20,
        "ibd_pct":               0.45,       # IBD (Crohn's-like) — PATHOGNOMONIC dual
        "xiap_absent_flow_pct":  0.99,       # XIAP absent on intracellular flow
        "nkt_partial_pct":       0.50,       # NKT partially present (unlike XLP1)
        "iloprost_pct":          0.40,       # Iloprost for cytopenias
        "hsct_pct":              0.50,
        "ferritin_10k_pct":      0.60,
        "pathognomonic":         "EBV-HLH + INFLAMMATORY BOWEL DISEASE DUAL PHENOTYPE PATHOGNOMONIC; XIAP absent on intracellular flow cytometry; NKT cells MAY BE PRESENT (distinguishes from XLP1 on flow)",
        "key_drug":              "Iloprost (prostacyclin analogue) for cytopenias and splenomegaly; HSCT curative; rituximab for EBV-HLH; anti-TNF therapy for IBD component (with EBV monitoring)",
    },
    "ITK": {
        "full_name":             "IL-2-inducible T-cell kinase (ITK; Tec kinase family)",
        "aa":                    620,
        "locus":                 "5q33.3",
        "kda":                   "72",
        "inheritance":           "AR (autosomal recessive; 5q33.3; both sexes affected — distinguishes from XLP1/XLP2/MAGT1)",
        "syndrome":              "ITK Deficiency — EBV-susceptibility CID with CD4 lymphopenia and class-switch defect",
        "pathway":               "ITK absent → PLCγ1 phosphorylation defective → NFAT/AP-1/NF-κB reduced → CD4 T-cell and NKT cell development impaired",
        "onset_range":           (3, 144),
        "ebv_hlh_pct":           0.55,
        "lymphoma_pct":          0.40,       # Burkitt lymphoma
        "cd4_lymphopenia_pct":   0.80,       # CD4 count 200-500
        "nkt_absent_pct":        0.70,       # NKT reduced/absent
        "hypo_gamma_pct":        0.60,
        "rituximab_pct":         0.55,
        "hsct_pct":              0.45,
        "ferritin_10k_pct":      0.52,
        "pathognomonic":         "FEMALE CID WITH EBV-SUSCEPTIBILITY (AR 5q33.3 — no sex bias unlike XLP1/XLP2/MAGT1); CD4 lymphopenia + NKT reduced; SAP PRESENT (DDx XLP1); ITK functional assay: reduced phospho-PLCγ1 after TCR stimulation",
        "key_drug":              "HSCT for EBV-driven lymphoproliferation; rituximab for acute EBV-HLH; IVIG for hypogammaglobulinemia; ganciclovir for EBV viraemia",
    },
    "MAGT1": {
        "full_name":             "Magnesium transporter 1 (MAGT1; XMEN syndrome gene)",
        "aa":                    335,
        "locus":                 "Xq21.1",
        "kda":                   "38",
        "inheritance":           "XLR (X-linked recessive; Xq21.1; males affected)",
        "syndrome":              "XMEN Syndrome — X-linked immunodeficiency with Magnesium defect, EBV infection, and Neoplasia",
        "pathway":               "MAGT1 absent → ER Mg2+ flux impaired after TCR → NKG2D glycosylation fails → NK/CD8 cytotoxicity reduced + CD4 lymphopenia",
        "onset_range":           (3, 180),
        "ebv_hlh_pct":           0.45,
        "lymphoma_pct":          0.45,       # Hodgkin and NHL
        "cd4_lymphopenia_pct":   0.90,       # CD4 very low <300 — MOST PROMINENT
        "nkg2d_absent_pct":      0.85,       # NKG2D absent on NK/CD8 cells
        "mg_reduced_pct":        0.90,       # free intracellular Mg2+ reduced
        "mg_supplement_pct":     0.70,       # Mg2+ supplementation used
        "rituximab_pct":         0.40,
        "hsct_pct":              0.40,
        "ferritin_10k_pct":      0.42,
        "pathognomonic":         "CD4 LYMPHOPENIA (<300/μL) VERY LOW — MOST PROMINENT FEATURE; NKG2D ABSENT on NK and CD8 T cells (Mg2+-dependent glycosylation); FREE INTRACELLULAR Mg2+ REDUCED in T cells (diagnostic biochemical); Mg2+ supplementation may partially restore NKG2D",
        "key_drug":              "Mg2+ supplementation (oral/dietary) to partially restore NKG2D expression; antivirals (valganciclovir) for EBV suppression; HSCT for severe EBV lymphoproliferation; IVIG for hypogammaglobulinemia",
    },
    "CD27": {
        "full_name":             "CD27 antigen (TNFRSF7; CD27 deficiency)",
        "aa":                    260,
        "locus":                 "12p13.31",
        "kda":                   "29",
        "inheritance":           "AR (autosomal recessive; 12p13.31; both sexes equally)",
        "syndrome":              "CD27 Deficiency — EBV-susceptibility PID with lymphoproliferation and HLH",
        "pathway":               "CD27-CD70 co-stimulatory axis absent → EBV-specific T-cell expansion defective → lymphoproliferation + HLH",
        "onset_range":           (5, 120),
        "ebv_hlh_pct":           0.60,
        "lymphoma_pct":          0.30,
        "aplastic_anemia_pct":   0.20,       # EBV-triggered aplastic anaemia
        "cd27_absent_flow_pct":  0.99,       # CD27 absent on lymphocyte surface
        "nk_cytotox_normal_pct": 0.75,       # NK cytotoxicity normal (DDx XLP1)
        "switched_mem_absent_pct": 0.85,     # absent switched memory B cells
        "hypo_gamma_pct":        0.55,
        "rituximab_pct":         0.55,
        "hsct_pct":              0.45,
        "ferritin_10k_pct":      0.58,
        "pathognomonic":         "CD27 ABSENT on T/B/NK surfaces (flow cytometry diagnostic); NK CYTOTOXICITY NORMAL (DDx from XLP1/XIAP where NK cytotoxicity often impaired); ABSENT SWITCHED MEMORY B CELLS (CD27+IgD- = zero on flow)",
        "key_drug":              "IVIG for hypogammaglobulinemia; rituximab for EBV-HLH; HSCT for recurrent severe EBV disease; prophylactic antivirals (valganciclovir)",
    },
    "CD70": {
        "full_name":             "CD70 antigen (TNFSF7; CD27 ligand; CD70 deficiency)",
        "aa":                    193,
        "locus":                 "19p13.3",
        "kda":                   "29",
        "inheritance":           "AR (autosomal recessive; 19p13.3; both sexes equally)",
        "syndrome":              "CD70 Deficiency — EBV-susceptibility PID with T-cell exhaustion PATHOGNOMONIC and very high lymphoma risk",
        "pathway":               "CD70-CD27 axis absent (ligand side) → EBV-specific CD8 T cells fail to expand → chronic EBV viraemia → progressive T-cell exhaustion → lymphoma",
        "onset_range":           (4, 144),
        "ebv_hlh_pct":           0.50,
        "lymphoma_pct":          0.55,       # VERY HIGH lymphoma risk
        "t_cell_exhaustion_pct": 0.85,       # EBV-specific T-cell exhaustion PATHOGNOMONIC
        "cd70_absent_stim_pct":  0.99,       # CD70 absent after stimulation
        "chronic_ebv_pct":       0.80,       # chronic EBV viraemia
        "switched_mem_absent_pct": 0.70,
        "hypo_gamma_pct":        0.50,
        "rituximab_pct":         0.45,
        "hsct_pct":              0.55,
        "ferritin_10k_pct":      0.48,
        "pathognomonic":         "EBV-SPECIFIC T-CELL EXHAUSTION PATHOGNOMONIC — CD8 T cells fail to expand against EBV; VERY HIGH LYMPHOMA RISK (Hodgkin + non-Hodgkin); CD70 ABSENT on activated T cells (post-stimulation flow); CHRONIC EBV VIRAEMIA due to impaired CD8 clearance",
        "key_drug":              "Prophylactic antiviral (valganciclovir) for EBV viraemia suppression pending HSCT; HSCT for EBV lymphoproliferation/lymphoma; rituximab for EBV-HLH; IVIG",
    },
    "TNFRSF9": {
        "full_name":             "Tumour necrosis factor receptor superfamily member 9 (4-1BB / CD137)",
        "aa":                    255,
        "locus":                 "1p36.23",
        "kda":                   "27",
        "inheritance":           "AR (autosomal recessive; 1p36.23; both sexes equally)",
        "syndrome":              "4-1BB (CD137) Deficiency — EBV susceptibility PID with NK and CD8 T-cell co-stimulation defect",
        "pathway":               "4-1BB absent → 4-1BBL co-stimulation of NK and CD8 T cells lost → EBV-specific CD8 survival impaired → lymphoproliferation + HLH",
        "onset_range":           (4, 120),
        "ebv_hlh_pct":           0.60,
        "lymphoma_pct":          0.35,       # Hodgkin lymphoma
        "nk_cytotox_reduced_pct": 0.65,      # NK cytotoxicity reduced
        "cd8_expansion_defect_pct": 0.75,    # EBV-specific CD8 expansion defect
        "cd137_absent_flow_pct": 0.99,       # 4-1BB absent on activated T/NK
        "hypo_gamma_pct":        0.40,
        "rituximab_pct":         0.55,
        "hsct_pct":              0.50,
        "ferritin_10k_pct":      0.58,
        "pathognomonic":         "4-1BB (CD137) ABSENT on activated T/NK cells (post-stimulation flow); EBV-SPECIFIC CD8 T-CELL EXPANSION DEFECT; NK cytotoxicity reduced; NOTE: 4-1BB agonist antibodies (urelumab/utomilumab) CONTRAINDICATED — no receptor to target",
        "key_drug":              "HSCT curative; rituximab for acute EBV-HLH; IVIG for hypogammaglobulinemia; etoposide+dexamethasone for HLH; 4-1BB agonist antibodies CONTRAINDICATED",
    },
    "RASGRP1": {
        "full_name":             "RAS guanyl-releasing protein 1 (RASGRP1; DAG-regulated RAS/MAPK GEF)",
        "aa":                    797,
        "locus":                 "15q14",
        "kda":                   "90",
        "inheritance":           "AR (autosomal recessive; 15q14; both sexes equally)",
        "syndrome":              "RASGRP1 Deficiency — combined immunodeficiency with EBV susceptibility AND autoimmunity overlap",
        "pathway":               "TCR → PLCγ1 → DAG → RASGRP1 → H-RAS/ERK absent → CD4 T-cell lymphopenia + NK cytotoxicity reduced + autoimmunity from impaired apoptosis",
        "onset_range":           (6, 144),
        "ebv_hlh_pct":           0.45,
        "lymphoma_pct":          0.25,
        "cd4_lymphopenia_pct":   0.70,       # CD4 reduced
        "autoimmunity_pct":      0.65,       # autoimmunity UNIQUE overlap — AIHA, lupus-like, ANA
        "erk_phospho_absent_pct": 0.90,      # absent pERK after stimulation
        "nk_cytotox_reduced_pct": 0.60,
        "hypo_gamma_pct":        0.50,
        "rituximab_pct":         0.40,
        "hsct_pct":              0.45,
        "ferritin_10k_pct":      0.43,
        "pathognomonic":         "AUTOIMMUNITY OVERLAP UNIQUE (AIHA, lupus-like, ANA positive) — distinguishes from pure EBV-susceptibility PIDs; ABSENT PHOSPHO-ERK after PMA/PHA stimulation (functional assay); CID + EBV susceptibility + autoimmunity combination = RASGRP1 signature",
        "key_drug":              "HSCT for severe CID/EBV; IVIG; rituximab for EBV-HLH; careful immunosuppression for autoimmune (avoid EBV reactivation triggers); hydroxychloroquine for lupus-like manifestations",
    },
}


def _rng(seed):
    r = random.Random(seed)
    return r


def _aggregate_gene(gene, idx):
    gd = GENE_DATA[gene]
    seed = SEED_BASE + idx
    rng = _rng(seed)

    n = N_PER_GENE
    cohort = []
    for i in range(n):
        s = _rng(seed * 1000 + i)
        age_m = s.randint(*gd["onset_range"])
        sex = "M" if gd["inheritance"].startswith("XLR") else s.choice(["M", "F"])
        cohort.append({
            "age_onset_months": age_m,
            "sex":               sex,
        })

    male_n = sum(1 for p in cohort if p["sex"] == "M")

    def _p(key):
        base = gd.get(key, 0.0)
        noise = rng.gauss(0, 0.03)
        return round(max(0.0, min(1.0, base + noise)), 3)

    return {
        "gene":                    gene,
        "full_name":               gd["full_name"],
        "aa":                      gd["aa"],
        "locus":                   gd["locus"],
        "kda":                     gd["kda"],
        "inheritance":             gd["inheritance"],
        "syndrome":                gd["syndrome"],
        "n":                       n,
        "male_n":                  male_n,
        "female_n":                n - male_n,
        "median_onset_months":     sorted(p["age_onset_months"] for p in cohort)[n // 2],
        "ebv_hlh_pct":             _p("ebv_hlh_pct"),
        "lymphoma_pct":            _p("lymphoma_pct"),
        "rituximab_pct":           _p("rituximab_pct"),
        "hsct_pct":                _p("hsct_pct"),
        "ferritin_10k_pct":        _p("ferritin_10k_pct"),
        "pathognomonic":           gd["pathognomonic"],
        "key_drug":                gd["key_drug"],
        "seed":                    seed,
    }


def generate_overview():
    cohorts = [_aggregate_gene(g, i) for i, g in enumerate(GENES)]

    total_ebv_hlh    = round(sum(c["ebv_hlh_pct"]    for c in cohorts) / len(cohorts), 3)
    total_lymphoma   = round(sum(c["lymphoma_pct"]   for c in cohorts) / len(cohorts), 3)
    total_rituximab  = round(sum(c["rituximab_pct"]  for c in cohorts) / len(cohorts), 3)
    total_hsct       = round(sum(c["hsct_pct"]       for c in cohorts) / len(cohorts), 3)
    total_ferritin   = round(sum(c["ferritin_10k_pct"] for c in cohorts) / len(cohorts), 3)

    return {
        "atlas":    ATLAS_NAME,
        "genes":    GENES,
        "total_n":  TOTAL_N,
        "seeds":    f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_stats": {
            "ebv_hlh_pct":     total_ebv_hlh,
            "lymphoma_pct":    total_lymphoma,
            "rituximab_pct":   total_rituximab,
            "hsct_pct":        total_hsct,
            "ferritin_10k_pct": total_ferritin,
        },
        "key_facts": [
            "SH2D1A/XLP1 (X-linked Lymphoproliferative Disease 1): NKT CELLS (CD3+Vα24+Vβ11+) ABSENT on flow = PATHOGNOMONIC; SAP absent on intracellular flow; FATAL EBV-HLH in childhood (mortality >70% untreated); Rituximab IMMEDIATELY for EBV-HLH + HSCT curative",
            "XIAP/XLP2 (X-linked Lymphoproliferative Disease 2): EBV-HLH + INFLAMMATORY BOWEL DISEASE DUAL PHENOTYPE PATHOGNOMONIC; XIAP absent on intracellular flow; NKT cells MAY BE PRESENT (DDx XLP1); Iloprost for cytopenias; HSCT curative",
            "ITK Deficiency (AR 5q33.3): FEMALE CID WITH EBV-SUSCEPTIBILITY (AR — both sexes); CD4 lymphopenia 200-500/μL; NKT cells reduced/absent; SAP PRESENT (DDx XLP1); phospho-PLCγ1 defect on TCR stimulation diagnostic",
            "MAGT1/XMEN Syndrome (XLR Xq21.1): CD4 LYMPHOPENIA VERY LOW (<300/μL) MOST PROMINENT; NKG2D ABSENT on NK/CD8 cells; FREE INTRACELLULAR Mg2+ REDUCED diagnostic; Mg2+ supplementation may partially restore NKG2D (Level B)",
            "CD27 Deficiency (AR 12p13.31): CD27 ABSENT on T/B/NK surface (flow); NK CYTOTOXICITY NORMAL (DDx XLP1/XIAP); absent switched memory B cells (CD27+IgD- = zero); IVIG + rituximab + HSCT",
            "CD70 Deficiency (AR 19p13.3): EBV-SPECIFIC T-CELL EXHAUSTION PATHOGNOMONIC; VERY HIGH LYMPHOMA RISK (Hodgkin + NHL); CD70 absent after stimulation; CHRONIC EBV VIRAEMIA; prophylactic valganciclovir pending HSCT",
            "TNFRSF9/4-1BB Deficiency (AR 1p36.23): 4-1BB ABSENT on activated T/NK cells; EBV-specific CD8 expansion defect; NOTE: 4-1BB AGONIST ANTIBODIES (urelumab/utomilumab) CONTRAINDICATED — no receptor to target",
            "RASGRP1 Deficiency (AR 15q14): AUTOIMMUNITY OVERLAP UNIQUE (AIHA + lupus-like + ANA positive); ABSENT PHOSPHO-ERK after PMA/PHA stimulation; CID + EBV susceptibility + autoimmunity = RASGRP1 signature",
        ],
        "xlr_group":    ["SH2D1A", "XIAP", "MAGT1"],
        "ar_group":     ["ITK", "CD27", "CD70", "TNFRSF9", "RASGRP1"],
        "nkt_absent_group":  ["SH2D1A", "ITK"],
        "ebv_hlh_group":     ["SH2D1A", "XIAP", "ITK", "MAGT1", "CD27", "CD70", "TNFRSF9", "RASGRP1"],
        "cohort_summaries":  cohorts,
    }


def generate_breakdown():
    cohorts = [_aggregate_gene(g, i) for i, g in enumerate(GENES)]

    diagnostic_groups = {
        "xlr_xlp1_xlp2": {
            "genes":          ["SH2D1A", "XIAP"],
            "mechanism":      "X-linked lymphoproliferative (XLP): X-linked SAP or XIAP deficiency → EBV-infected B cell unchecked → fatal HLH in childhood boys",
            "shared_feature": "X-linked recessive; males only; EBV-triggered HLH; absent/reduced NK cytotoxicity; intracellular flow diagnostic; HSCT curative",
            "key_ddx":        "XLP1 (SH2D1A/SAP): NKT cells ABSENT; SAP absent on flow; dysgammaglobulinemia. XLP2 (XIAP): NKT cells may be PRESENT; XIAP absent on flow; IBD = PATHOGNOMONIC dual feature distinguishes XLP2.",
        },
        "xlr_xmen_magt1": {
            "genes":          ["MAGT1"],
            "mechanism":      "XMEN syndrome: XLR MAGT1 → ER Mg2+ flux absent → NKG2D glycosylation fails → NK/CD8 cytotoxicity impaired + CD4 lymphopenia",
            "shared_feature": "X-linked; CD4 very low (<300/μL); NKG2D absent on NK/CD8; free intracellular Mg2+ reduced; EBV neoplasia + Hodgkin lymphoma",
            "key_ddx":        "MAGT1 vs XLP1: CD4 very low in MAGT1 (vs near-normal in XLP1); NKG2D absent (MAGT1-specific); SAP present in MAGT1; Mg2+ supplementation specific to MAGT1.",
        },
        "ar_tec_kinase_itk": {
            "genes":          ["ITK"],
            "mechanism":      "ITK absent → PLCγ1 not phosphorylated after TCR → NFAT/AP-1 not activated → CD4 T-cell and NKT cell development impaired",
            "shared_feature": "AR; both sexes; CD4 lymphopenia; NKT reduced/absent; EBV lymphoproliferation; phospho-PLCγ1 functional assay diagnostic",
            "key_ddx":        "ITK vs XLP1/XLP2: ITK is AR (female presenting). SAP normal in ITK deficiency. ITK vs RASGRP1: both AR non-XLP CID; ITK: phospho-PLCγ1 defect; RASGRP1: phospho-ERK defect (downstream of ITK in RAS pathway).",
        },
        "ar_cd27_cd70_costimulatory": {
            "genes":          ["CD27", "CD70"],
            "mechanism":      "CD27-CD70 co-stimulatory axis absent (either receptor or ligand side) → EBV-specific T-cell expansion fails → chronic EBV viraemia → lymphoma/HLH",
            "shared_feature": "AR; both sexes; EBV-driven lymphoproliferation; absent switched memory B cells; CVID-like hypogammaglobulinemia",
            "key_ddx":        "CD27 vs CD70: CD27 absent on surface (CD27 deficiency); CD70 absent after stimulation (CD70 deficiency). CD27: NK cytotoxicity normal. CD70: T-cell exhaustion and very high lymphoma risk more prominent. Gene panel or WES required to distinguish.",
        },
        "ar_tnfrsf9_4_1bb": {
            "genes":          ["TNFRSF9"],
            "mechanism":      "4-1BB absent → 4-1BBL co-stimulation of NK and CD8 T cells absent → EBV-specific CD8 expansion and survival impaired → lymphoproliferation + HLH",
            "shared_feature": "AR; 4-1BB absent on activated T/NK cells; EBV-specific CD8 expansion defect; NK cytotoxicity reduced; rare",
            "key_ddx":        "TNFRSF9 vs CD27/CD70: 4-1BB targets CD8 and NK memory/survival (vs CD27/CD70 which targets CD4 and B-cell switch). 4-1BB agonists CONTRAINDICATED (no receptor). Very rare — WES required.",
        },
        "ar_rasgrp1_autoimmunity": {
            "genes":          ["RASGRP1"],
            "mechanism":      "RASGRP1 absent → RAS/ERK pathway absent after TCR → T-cell proliferation and positive selection impaired + NK perforin degranulation reduced + autoimmunity from impaired peripheral tolerance",
            "shared_feature": "AR; CID + EBV susceptibility + autoimmunity overlap; phospho-ERK absent; unique among EBV-susceptibility PIDs",
            "key_ddx":        "RASGRP1 vs all other EBV PIDs: AUTOIMMUNITY unique feature (AIHA, ANA, lupus-like). SAP/XIAP/MAGT1/CD27/CD70/4-1BB all normal. Functional assay: absent phospho-ERK after PMA/PHA. RASGRP1 on WES.",
        },
    }

    pathognomonic_features = {
        "xlp1_nkt_absent":                  "SH2D1A/XLP1: NKT CELLS (CD3+Vα24+Vβ11+) ABSOLUTELY ABSENT on flow cytometry — PATHOGNOMONIC biomarker; NKT cells are uniquely SAP-dependent; zero NKT = virtually diagnostic of XLP1 in male with EBV-HLH; confirm with SAP intracellular flow + SH2D1A sequencing",
        "xlp2_ebd_ibd_dual":                "XIAP/XLP2: EBV-HLH + INFLAMMATORY BOWEL DISEASE DUAL PHENOTYPE = PATHOGNOMONIC combination; IBD in XLP2 is severe early-onset Crohn's-like disease (fistulae, perianal); mechanism: NOD2/RIPK2/NF-κB in gut lost; XIAP absent on intracellular flow",
        "magt1_nkg2d_mg_low":               "MAGT1/XMEN: NKG2D ABSENT on NK and CD8 T cells (Mg2+-dependent N-glycosylation); FREE INTRACELLULAR Mg2+ REDUCED in T cells (measurable by fluorescent Mg2+ probe); CD4 <300/μL very low; Mg2+ supplementation (oral/dietary) may partially restore NKG2D = Level B specific intervention",
        "cd70_t_cell_exhaustion":           "CD70: EBV-SPECIFIC T-CELL EXHAUSTION PATHOGNOMONIC — chronic EBV viraemia because CD8 T cells cannot expand without CD70-CD27 co-stimulation → progressive exhaustion markers (PD-1+, TIM-3+, LAG-3+ on EBV-specific CD8 cells); VERY HIGH LYMPHOMA RISK (Hodgkin and NHL)",
        "rasgrp1_autoimmunity_unique":      "RASGRP1: AUTOIMMUNITY OVERLAP IS UNIQUE among EBV-susceptibility PIDs — autoimmune haemolytic anaemia, ANA positive, lupus-like manifestations; absent phospho-ERK after PMA/PHA stimulation = functional diagnostic; distinguish from primary autoimmune disease by concurrent CID + EBV susceptibility",
        "sap_xiap_flow_diagnostic_axis":    "XLP1 vs XLP2 FLOW CYTOMETRY DIAGNOSTIC AXIS: SAP intracellular flow (XLP1: absent; XLP2: present) + NKT cell count (XLP1: zero; XLP2: may be present) + XIAP intracellular flow (XLP2: absent; XLP1: present). Three-marker panel distinguishes XLP1 vs XLP2 BEFORE genetic sequencing.",
        "ebv_ferritin_10k_hlh":             "ALL 8 GENES: EBV-triggered HLH — ferritin >10,000 μg/L HIGHLY SUPPORTIVE; HLH-2004 diagnostic criteria: ferritin, splenomegaly, cytopenia, haemophagocytosis on BM biopsy, NK dysfunction (NK flow cytotoxicity assay), elevated sIL-2Rα (CD25), triglycerides elevated + fibrinogen low. RITUXIMAB + DEXAMETHASONE/ETOPOSIDE + CSAS urgently",
        "hsct_curative_all_genes":          "HSCT IS CURATIVE FOR ALL 8 GENES in this atlas when performed before irreversible organ damage. Conditioning: RIC (reduced-intensity conditioning) used for XLP1/XLP2. Matched related or unrelated donor; haplo-identical acceptable. Timing: AFTER achieving HLH remission with rituximab + HLH-2004 protocol",
        "tnfrsf9_4_1bb_agonist_ci":         "TNFRSF9/4-1BB deficiency: 4-1BB AGONIST ANTIBODIES (urelumab, utomilumab) ARE ABSOLUTELY CONTRAINDICATED — patient has no 4-1BB receptor to engage; these therapies are designed to stimulate 4-1BB co-stimulation which is absent in this disease; wrong diagnosis or misuse would be an adverse event with no benefit",
        "ebv_rituximab_emergency":          "ALL EBV-SUSCEPTIBILITY PIDs: Rituximab (anti-CD20 375 mg/m²) as EMERGENCY intervention for EBV-HLH — depletes EBV-infected B cells (main reservoir); reduces EBV viral load rapidly; window of opportunity before multi-organ failure; combine with dexamethasone + etoposide for HLH cytokine storm control",
    }

    return {
        "gene_cohorts":           cohorts,
        "diagnostic_groups":      diagnostic_groups,
        "pathognomonic_features": pathognomonic_features,
        "cascade_testing_order": [
            "Step 1: Male child with EBV-HLH → screen IMMEDIATELY: SAP intracellular flow + XIAP intracellular flow + NKT cell count (CD3+Vα24+Vβ11+) + serum ferritin; do NOT wait for genetics before rituximab if HLH criteria met",
            "Step 2: SAP ABSENT on flow → XLP1 (SH2D1A); confirm NKT absent; start SH2D1A sequencing; rituximab + HLH-2004 protocol IMMEDIATELY; HSCT referral",
            "Step 3: XIAP ABSENT on flow → XLP2 (BIRC4); NKT may be present; check for IBD history; start rituximab + iloprost; HSCT referral; BIRC4 sequencing",
            "Step 4: Female with EBV-CID or any sex with EBV-susceptibility + CD4 lymphopenia → add ITK functional assay (phospho-PLCγ1) + MAGT1 panel + CD27/CD70 flow",
            "Step 5: CD4 <300/μL + NKG2D absent on NK/CD8 → MAGT1/XMEN syndrome; confirm Mg2+ level; start Mg2+ supplementation + valganciclovir; MAGT1 sequencing",
            "Step 6: CD27 absent on surface → CD27 deficiency; check switched memory B cells (CD27+IgD-); start IVIG + rituximab; CD27 gene sequencing",
            "Step 7: CD70 absent after PHA stimulation → CD70 deficiency; assess EBV viral load + lymphoma staging (CT/PET); prophylactic valganciclovir + HSCT referral; CD70 sequencing",
            "Step 8: 4-1BB (CD137) absent on activated T/NK → TNFRSF9 deficiency; HSCT referral; NOTE: do NOT use 4-1BB agonist antibodies; TNFRSF9 sequencing",
            "Step 9: Autoimmunity (AIHA/ANA) + CID + EBV → RASGRP1 functional assay (phospho-ERK absent after PMA/PHA); RASGRP1 sequencing; WES if panel negative",
            "Step 10: All above normal + EBV-susceptibility phenotype → extended WES/genome sequencing for novel genes; refer PID specialist centre",
        ],
    }


def generate_definitions():
    return {
        "definitions": [
            {"term": "XLP1 (SH2D1A / SAP Deficiency)", "definition": "XLR SH2D1A (Xq25; 128aa). SAP (SLAM-associated protein) links SLAM receptors to downstream signalling in T, NKT, and NK cells. XLP1: most common EBV-susceptibility PID. PATHOGNOMONIC: NKT cells (CD3+Vα24+Vβ11+) ABSENT on flow cytometry — near-universal finding. SAP absent on intracellular flow cytometry — diagnostic. EBV triggers fatal HLH (mortality >70% untreated). Other features: dysgammaglobulinemia (Ig class-switch defect), Burkitt lymphoma. Treatment: Rituximab IMMEDIATELY for EBV-HLH (deplete EBV-infected B cells); then HSCT curative. HLH-2004 protocol (dexamethasone + etoposide + cyclosporin)."},
            {"term": "XLP2 (XIAP / BIRC4 Deficiency)", "definition": "XLR BIRC4 (Xq25; 497aa). XIAP (X-linked inhibitor of apoptosis) inhibits caspase-3/9 and activates NF-κB via RIPK2/NOD2 for gut homeostasis. XLP2: second most common XLP PID. PATHOGNOMONIC DUAL PHENOTYPE: EBV-triggered HLH + INFLAMMATORY BOWEL DISEASE — this combination is unique to XIAP deficiency. XIAP absent on intracellular flow cytometry — diagnostic. NKT cells MAY BE PRESENT (distinguishes from XLP1). IBD: severe early-onset Crohn's-like (fistulae, perianal disease). Treatment: Iloprost (prostacyclin analogue) for cytopenias; HSCT curative; anti-TNF for IBD with EBV monitoring; rituximab for EBV-HLH."},
            {"term": "ITK Deficiency", "definition": "AR ITK (5q33.3; 620aa). ITK (IL-2-inducible T-cell kinase) is a Tec family kinase that phosphorylates PLCγ1 downstream of TCR. ITK deficiency: AR CID with EBV susceptibility. AR on 5q33.3 → BOTH SEXES affected (distinguishes from XLP1/XLP2/MAGT1 which are XLR). CD4 lymphopenia (200-500/μL); NKT cells absent or severely reduced. EBV-driven Burkitt lymphoma and HLH in childhood. SAP PRESENT (key DDx from XLP1). Functional assay: absent phospho-PLCγ1 after TCR stimulation. Treatment: HSCT for EBV lymphoproliferation; rituximab for EBV-HLH; IVIG for hypogammaglobulinemia."},
            {"term": "MAGT1 / XMEN Syndrome", "definition": "XLR MAGT1 (Xq21.1; 335aa). MAGT1 is an ER Mg2+ channel AND oligosaccharyltransferase subunit required for N-glycosylation. XMEN syndrome = X-linked immunodeficiency with Magnesium defect, EBV infection, and Neoplasia. CD4 LYMPHOPENIA (<300/μL) = MOST PROMINENT FEATURE. NKG2D ABSENT on NK and CD8 T cells (Mg2+-dependent N-glycosylation of NKG2D fails). FREE INTRACELLULAR Mg2+ REDUCED in T cells — unique biochemical diagnostic. Mg2+ SUPPLEMENTATION may partially restore NKG2D (Level B evidence). EBV lymphoma (Hodgkin and NHL). Treatment: Mg2+ supplementation; valganciclovir for EBV suppression; HSCT for severe lymphoproliferation."},
            {"term": "CD27 Deficiency", "definition": "AR CD27/TNFRSF7 (12p13.31; 260aa). CD27 = TNF receptor expressed on T, B, and NK cells. CD27-CD70 co-stimulatory axis drives T-cell activation and B-cell isotype switch. CD27 ABSENT on T/B/NK surface → DIAGNOSTIC by flow cytometry. NK CYTOTOXICITY NORMAL in CD27 deficiency (DDx from XLP1/XIAP where NK cytotoxicity impaired). Absent switched memory B cells (CD27+IgD- = zero). CVID-like hypogammaglobulinemia. EBV-induced HLH, aplastic anaemia, lymphoma. Treatment: IVIG for hypogammaglobulinemia; rituximab for EBV-HLH; HSCT for recurrent severe EBV disease; prophylactic valganciclovir."},
            {"term": "CD70 Deficiency", "definition": "AR CD70/TNFSF7 (19p13.3; 193aa). CD70 = TNF superfamily ligand for CD27; transiently expressed on activated T/B cells and DCs. CD70-CD27 axis (ligand side): CD70 absent → EBV-specific CD8 T cells cannot be properly co-stimulated → fail to expand → chronic EBV viraemia. EBV-SPECIFIC T-CELL EXHAUSTION PATHOGNOMONIC — progressive exhaustion (PD-1+, TIM-3+ on EBV-specific CD8). VERY HIGH LYMPHOMA RISK (Hodgkin and non-Hodgkin) due to impaired CD8 immune surveillance. CD70 absent on activated T cells (post-stimulation flow). Treatment: prophylactic valganciclovir for EBV suppression; HSCT for EBV lymphoproliferation/lymphoma; IVIG."},
            {"term": "4-1BB (TNFRSF9/CD137) Deficiency", "definition": "AR TNFRSF9 (1p36.23; 255aa). 4-1BB (CD137) = co-stimulatory TNF receptor on activated T cells and NK cells; 4-1BBL co-stimulation drives CD8 T-cell expansion, survival, and memory formation. 4-1BB absent → EBV-specific CD8 expansion and long-term memory impaired. 4-1BB ABSENT on activated T/NK cells (post-stimulation flow). NK cytotoxicity reduced. EBV-driven Hodgkin lymphoma and HLH. CRITICAL: 4-1BB AGONIST ANTIBODIES (urelumab, utomilumab — cancer immunotherapy agents) ARE ABSOLUTELY CONTRAINDICATED — no receptor to target. Treatment: HSCT; rituximab for EBV-HLH; IVIG. Very rare."},
            {"term": "RASGRP1 Deficiency", "definition": "AR RASGRP1 (15q14; 797aa). RASGRP1 = DAG-regulated RAS GEF that activates H-RAS/ERK downstream of TCR → PLCγ1 → DAG. ERK/MAPK pathway essential for T-cell proliferation and NK perforin degranulation. UNIQUE FEATURE: AUTOIMMUNITY OVERLAP — autoimmune haemolytic anaemia, ANA positive, lupus-like manifestations alongside CID (distinguishes from all other EBV-susceptibility PIDs). CD4 lymphopenia. EBV-susceptibility with lymphoproliferation. ABSENT PHOSPHO-ERK after PMA/PHA stimulation = functional diagnostic. Treatment: HSCT for severe CID/EBV; IVIG; rituximab for EBV-HLH; hydroxychloroquine for lupus-like component; careful immunosuppression to avoid EBV reactivation."},
            {"term": "EBV-HLH Diagnostic Criteria and Treatment", "definition": "HLH-2004 DIAGNOSTIC CRITERIA (5 of 8): fever, splenomegaly, cytopenia ≥2 cell lines, hypertriglyceridaemia (>3.0 mmol/L) or hypofibrinogenaemia (<1.5 g/L), haemophagocytosis on BM/spleen/LN biopsy, low/absent NK cell activity (flow cytometry), ferritin >500 μg/L (>10,000 highly specific for HLH), elevated soluble CD25 (sIL-2Rα) >2400 U/mL. TREATMENT FOR EBV-HLH IN PID: Rituximab (anti-CD20) 375 mg/m² EMERGENTLY to deplete EBV-infected B cells; Dexamethasone 10 mg/m²/day; Etoposide 150 mg/m²; Cyclosporin; then HSCT after remission. Do NOT delay rituximab pending genetics in suspected XLP1/XLP2."},
            {"term": "Intracellular Flow Cytometry for XLP Diagnosis", "definition": "DIAGNOSTIC PROTOCOL for XLP1 vs XLP2: SAP intracellular flow (permeabilise PBMCs with saponin/Fix-Perm; anti-SAP antibody; XLP1: SAP absent — DIAGNOSTIC; XLP2: SAP present). XIAP intracellular flow (same protocol; anti-XIAP antibody; XLP2: XIAP absent — DIAGNOSTIC; XLP1: XIAP present). NKT cell count (surface staining; CD3+CD1d-αGalCer tetramer+ or Vα24-Jα18 TCR stain; XLP1: zero NKT cells — PATHOGNOMONIC; XLP2: variable). Three-marker panel distinguishes XLP1 from XLP2 BEFORE genetic sequencing. Confirm with SH2D1A or BIRC4 gene sequencing."},
            {"term": "HSCT in EBV-Susceptibility PID", "definition": "HSCT IS CURATIVE FOR ALL 8 GENES in this atlas. KEY PRINCIPLES: (1) Achieve EBV-HLH remission FIRST with rituximab + HLH-2004 before conditioning; (2) RIC (reduced-intensity conditioning with fludarabine/cyclophosphamide/alemtuzumab or ATG) preferred over MAC for XLP1/XLP2 (organ toxicity concern post-HLH); (3) Donor: matched related (MRD) or matched unrelated (MUD) preferred; haplo-identical acceptable with PTCy; (4) GvHD prophylaxis standard; (5) EBV monitoring post-HSCT (weekly EBV PCR); (6) Outcome: >80% long-term disease-free survival when HSCT performed in remission before multiple organ failure."},
            {"term": "NK/NKT Cell Function Testing in EBV-PID Workup", "definition": "FLOW CYTOMETRY PANEL FOR EBV-SUSCEPTIBILITY PID: CD3+Vα24+Vβ11+ (iNKT cells) → absent in XLP1/ITK. CD3-CD16+CD56+ (NK cells) count + NKG2D expression (absent in MAGT1/XMEN). NK cytotoxicity assay (K562 targets): absent/reduced in XLP1, XIAP, RASGRP1; NORMAL in CD27 deficiency. SAP intracellular (absent XLP1). XIAP intracellular (absent XLP2). CD27 surface (absent CD27 deficiency). 4-1BB (CD137) surface post-stimulation (absent TNFRSF9 deficiency). CD70 on activated T cells post-stimulation (absent CD70 deficiency). Perforin intracellular (absent in PRF1/familial HLH — different diagnosis). Functional assays: phospho-PLCγ1 (ITK deficiency), phospho-ERK (RASGRP1 deficiency)."},
        ],
        "key_distinguishers": {
            "sex_pattern":              "XLR (males only): SH2D1A (XLP1), XIAP (XLP2), MAGT1 (XMEN). AR (both sexes): ITK, CD27, CD70, TNFRSF9, RASGRP1.",
            "nkt_absent":               "NKT cells ABSENT: SH2D1A/XLP1 (most prominent, near-universal), ITK (severely reduced). NKT cells may be PRESENT: XIAP/XLP2, MAGT1, CD27, CD70, TNFRSF9, RASGRP1.",
            "ibd_dual_phenotype":       "XIAP/XLP2 ONLY: EBV-HLH + IBD dual phenotype PATHOGNOMONIC — no other EBV-susceptibility PID has IBD as a hallmark feature.",
            "autoimmunity_overlap":     "RASGRP1 ONLY: autoimmunity overlap (AIHA, ANA, lupus-like) alongside CID + EBV — unique among all 8 genes.",
            "low_cd4":                  "Most prominent CD4 lymphopenia: MAGT1 (<300/μL most severe), ITK (200-500/μL), RASGRP1 (reduced). Relatively preserved CD4: SH2D1A (NKT absent but CD4 often near-normal), XIAP, CD27, CD70, TNFRSF9.",
            "flow_diagnostic_axes":     "SAP absent = XLP1. XIAP absent = XLP2. NKG2D absent = MAGT1. CD27 absent = CD27 deficiency. CD70 absent (post-stimulation) = CD70 deficiency. 4-1BB absent (post-stimulation) = TNFRSF9. All above present + EBV-susceptibility → ITK or RASGRP1 (functional assays required).",
            "4_1bb_agonist_ci":         "TNFRSF9/4-1BB deficiency: ABSOLUTELY CONTRAINDICATED: urelumab, utomilumab, any 4-1BB agonist antibody.",
            "mg_supplementation_unique": "MAGT1/XMEN ONLY: Mg2+ supplementation is a specific, unique intervention (Level B) to partially restore NKG2D expression. No other EBV-susceptibility PID benefits from Mg2+ supplementation.",
        },
        "treatment_matrix": {
            "rituximab":                "ALL 8 genes for EBV-HLH: Rituximab (anti-CD20; 375 mg/m²) EMERGENTLY to deplete EBV-infected B cells; reduces EBV viral load; window before multi-organ failure.",
            "hsct":                     "ALL 8 genes: HSCT curative. RIC conditioning preferred for XLP1/XLP2 (post-HLH organ protection). MRD/MUD donor; haplo acceptable. Achieve remission first.",
            "iloprost":                 "XIAP/XLP2 SPECIFICALLY: Iloprost (prostacyclin analogue) for cytopenias and splenomegaly; reduces apoptosis-driven cell death in XIAP-deficient cells.",
            "mg_supplementation":       "MAGT1/XMEN SPECIFICALLY: Mg2+ supplementation (dietary/oral) to partially restore NKG2D glycosylation and expression (Level B — pragmatic step before HSCT).",
            "valganciclovir":           "CD70 + CD27 + TNFRSF9 + MAGT1: prophylactic valganciclovir for chronic EBV viraemia suppression pending HSCT; monitor EBV viral load monthly.",
            "ivig":                     "CD27 + CD70 + ITK + MAGT1 + TNFRSF9 + RASGRP1: IVIG for hypogammaglobulinemia and class-switch defect. 400-600 mg/kg every 3-4 weeks.",
            "hlh_protocol":             "ALL 8 genes for EBV-HLH: HLH-2004 protocol (dexamethasone 10 mg/m²/day + etoposide 150 mg/m²/wk + cyclosporin) alongside rituximab for EBV B-cell clearance.",
            "hydroxychloroquine":       "RASGRP1 specifically: hydroxychloroquine for lupus-like autoimmune manifestations; use cautiously — EBV monitoring still required.",
        },
    }


if __name__ == "__main__":
    import json
    ov = generate_overview()
    bk = generate_breakdown()
    df = generate_definitions()
    print(f"Overview genes: {ov['genes']}")
    print(f"Total N: {ov['total_n']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Gene cohorts: {len(bk['gene_cohorts'])}")
    print(f"Definitions: {len(df['definitions'])}")
    print("OK")
