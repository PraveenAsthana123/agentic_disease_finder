#!/usr/bin/env python3
"""Hereditary-Primary-Immunodeficiency-Atlas — Complete 8-Gene Hereditary Primary Immunodeficiency Atlas
(IL2RG · ADA · RAG1 · BTK · AIRE · LRBA · CTLA4 · PIK3CD).

IL2RG    (Common Gamma Chain; 369 aa; Xq13.1; XLR;
          X-linked SCID (SCID-X1);
          T-B+NK- IMMUNOPHENOTYPE PATHOGNOMONIC — absent T + NK, normal but non-functional B;
          Most common SCID in Western countries (45-50%);
          NO live vaccines EVER — BCG before diagnosis = BCG-osis (disseminated fatal);
          Gene therapy: lentiviral IL2RG (OTL-101 / CARTEYVA) — FDA curative;
          TREC assay (newborn screen) detects absent T cells from birth;
          seed SEED_BASE+0).
ADA      (Adenosine Deaminase; 363 aa; 20q13.12; AR;
          ADA-SCID;
          T-B-NK- IMMUNOPHENOTYPE — COMPLETE ABSENCE ALL LYMPHOCYTES (worst of all SCIDs);
          Rib-costal junction cupping ("rachitic rosary") on CXR PATHOGNOMONIC;
          PEG-ADA enzyme replacement (ADAGEN/REVCOVI) — bridge, non-curative;
          Gene therapy: Strimvelis (GSK) — FDA approved, potentially curative;
          seed SEED_BASE+1).
RAG1     (Recombination Activating Gene 1; 1043 aa; 11p13; AR;
          RAG1-SCID / Omenn Syndrome;
          Complete null: T-B-NK+ SCID; Hypomorphic: Omenn Syndrome;
          ERYTHRODERMA + ELEVATED IgE + EOSINOPHILIA + ABSENT NORMAL IMMUNOGLOBULINS PATHOGNOMONIC (Omenn);
          RAG1/RAG2 both on 11p13 head-to-head — gene panel tests both simultaneously;
          seed SEED_BASE+2).
BTK      (Bruton Tyrosine Kinase; 659 aa; Xq22.1; XLR;
          X-linked Agammaglobulinemia (XLA / Bruton);
          PERIPHERAL B CELLS <1% + ALL IMMUNOGLOBULIN ISOTYPES VIRTUALLY ZERO PATHOGNOMONIC;
          ENTEROVIRAL ENCEPHALITIS — chronic CNS infection (ECHO/poliovirus) life-threatening;
          IVIG lifelong (every 3-4 weeks); NO live vaccines (OPV strictly contraindicated);
          seed SEED_BASE+3).
AIRE     (Autoimmune Regulator; 552 aa; 21q22.3; AR;
          Autoimmune Polyendocrinopathy-Candidiasis-Ectodermal Dystrophy (APECED / APS-1);
          MUCOCUTANEOUS CANDIDIASIS + HYPOPARATHYROIDISM + ADRENAL INSUFFICIENCY TRIAD PATHOGNOMONIC;
          Anti-IFN-omega antibodies = diagnostic biomarker (>95% sensitivity/specificity);
          Finnish founder p.Arg257Ter (1:25,000 Finland); ADRENAL CRISIS = emergency;
          seed SEED_BASE+4).
LRBA     (LPS-Responsive Beige-Like Anchor; 2863 aa; 4q31.3; AR;
          LRBA Deficiency;
          HYPOGAMMAGLOBULINEMIA + AUTOIMMUNITY COMBINED PATHOGNOMONIC DDx from CVID;
          CTLA4-Ig (ABATACEPT) highly effective — reverses autoimmunity within weeks;
          IBD-like enteropathy prominent (50-70%); CTLA4 flow on T-regs = functional screen;
          seed SEED_BASE+5).
CTLA4    (Cytotoxic T-Lymphocyte Antigen 4; 223 aa; 2q33.2; AD haploinsufficiency;
          CTLA4 Haploinsufficiency (CTLA4-H);
          SPLENOMEGALY + LYMPHADENOPATHY + AUTOIMMUNE CYTOPENIAS + HYPOGAMMAGLOBULINEMIA + LYMPHOCYTIC INFILTRATION PATHOGNOMONIC;
          ABATACEPT highly effective (provides functional CTLA4 replacement);
          Distinguished from LRBA: AD one-hit vs AR; LRBA flow shows absent LRBA protein;
          seed SEED_BASE+6).
PIK3CD   (Phosphatidylinositol-4,5-bisphosphate 3-kinase catalytic delta; 1044 aa; 1p36.22; AD GOF;
          Activated PI3Kd Syndrome (APDS / APDS1);
          LYMPHOPROLIFERATION + EBV/CMV SUSCEPTIBILITY + RECURRENT RESPIRATORY INFECTIONS + HYPOGAMMAGLOBULINEMIA PATHOGNOMONIC;
          LENIOLISIB (Joenja) FDA approved 2023 — first targeted PI3Kd inhibitor for APDS;
          EBV-driven lymphoma risk up to 20% — annual EBV viral load monitoring MANDATORY;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2310-2317).
"""

import random

SEED_BASE = 2310

IMMUNO_GENES = [
    # -- IL2RG — SCID-X1 -------------------------------------------------------------------
    {
        "gene": "IL2RG",
        "alt_name": (
            "IL2RG (IL2RG-369aa-Xq13.1 / XLR — X-linked-SCID-SCID-X1 — "
            "T-B+NK--IMMUNOPHENOTYPE-PATHOGNOMONIC — "
            "Most-Common-SCID-Western-Countries-45-50pct — "
            "NO-LIVE-VACCINES-EVER-BCG-Before-Dx-BCG-osis-Fatal)"
        ),
        "protein": (
            "IL2RG -- Xq13.1 XLR -- IL2RG-369aa -- "
            "Common-Gamma-Chain-gc-42kDa-Shared-Receptor-Subunit-IL-2-IL-4-IL-7-IL-9-IL-15-IL-21 -- "
            "IL-7-Signal-Required-T-Cell-Development-Thymic-Progenitor-Maturation -- "
            "IL-15-Signal-Required-NK-Cell-Development-Differentiation -- "
            "ABSENT-T-CELLS-ABSENT-NK-CELLS-NORMAL-B-CELLS-BUT-NON-FUNCTIONAL-T-B+NK- -- "
            "SCID-X1-MOST-COMMON-SCID-WESTERN-45-50pct-All-SCID-Males-Affected -- "
            "TREC-ASSAY-NEWBORN-SCREEN-Absent-T-Cell-Receptor-Excision-Circles-DETECTED -- "
            "NO-LIVE-VACCINES-BCG-GIVEN-BEFORE-DIAGNOSIS-BCG-OSIS-DISSEMINATED-FATAL -- "
            "GENE-THERAPY-LENTIVIRAL-IL2RG-OTL-101-CARTEYVA-FDA-Curative -- "
            "MATERNAL-ENGRAFTMENT-May-Partially-Mask-Disease-Oligoclonal-T-Cells -- "
            "OMIM-Gene-308380-Disease-SCIDX1-300400"
        ),
        "locus": "Xq13.1",
        "protein_size": "369 aa / 42 kDa",
        "inheritance": (
            "XLR; hemizygous IL2RG mutations in males; carrier females (heterozygous) unaffected; "
            "de novo mutations ~50%; all affected individuals are male (XLR); "
            "lyonisation in carrier females: skewed X-inactivation of IL2RG-carrying X → "
            "selective advantage for WT-IL2RG lymphocytes in carriers (diagnostic clue); "
            "female SCID-X1 possible if Turner (45X) + IL2RG mutation on single X"
        ),
        "immunodeficiency_category": "SCID — X-linked SCID (IL2RG, gamma-c chain; T-B+NK- phenotype; most common SCID Western countries)",
        "pathognomonic": (
            "T-B+NK- IMMUNOPHENOTYPE: ABSENT T CELLS + ABSENT NK CELLS + NORMAL BUT NON-FUNCTIONAL B CELLS; "
            "T cell absence from birth (TREC assay = zero); "
            "B cells present but non-functional (no IL-2/IL-4/IL-7/IL-9/IL-15/IL-21 signalling — all require gamma-c chain); "
            "recurrent/severe infections by 3-6 months as maternal IgG wanes; "
            "BCG vaccine given before diagnosis = BCG-osis (disseminated BCG) — PATHOGNOMONIC IATROGENIC EMERGENCY"
        ),
        "treatment": (
            "HAEMATOPOIETIC STEM CELL TRANSPLANTATION (HSCT) — preferred curative treatment; "
            "MATCHED SIBLING DONOR (MSD): best outcomes (>90% survival); "
            "T-cell depleted matched unrelated donor (MUD): also effective (>75% modern series); "
            "GENE THERAPY: lentiviral IL2RG (OTL-101 / CARTEYVA — FDA approved): "
            "curative in absence of matched donor; multicentre trials show immune reconstitution; "
            "IVIG: bridge pre-HSCT/gene therapy for infections; "
            "NO LIVE VACCINES EVER — BCG, MMR, varicella absolutely contraindicated; "
            "prophylactic antibiotics (TMP-SMX) + antifungals (fluconazole) pre-HSCT; "
            "NEWBORN SCREEN: TREC assay detects SCID before symptoms — early HSCT improves outcomes; "
            "WITHOUT HSCT: median survival < 1 year (overwhelming infections by 6 months)"
        ),
        "key_features": [
            "T-B+NK- IMMUNOPHENOTYPE — absent T + NK cells, normal but non-functional B cells — PATHOGNOMONIC",
            "MOST COMMON SCID in Western countries: 45-50% of all SCID cases; exclusively males (XLR)",
            "TREC ASSAY (newborn screen): detects absent T cells from birth — enables pre-symptomatic diagnosis",
            "BCG BEFORE DIAGNOSIS = BCG-OSIS: disseminated mycobacterial infection — iatrogenic emergency; fatal without HSCT",
            "gamma-c chain (IL2RG) shared by IL-2/IL-4/IL-7/IL-9/IL-15/IL-21 receptors — ALL T/NK development blocked",
            "MATERNAL ENGRAFTMENT: maternal T cells may partially engraft → oligoclonal T cells — can mask disease transiently",
        ],
        "monitoring": [
            "Newborn TREC screen: absent TREC = SCID referral within 24-48hr; confirm with lymphocyte phenotyping",
            "Lymphocyte subsets (CD3/CD4/CD8/CD16-56/CD19) at diagnosis: confirm T-B+NK- phenotype",
            "IVIG trough levels (target >600-800 mg/dL) monthly until immune reconstitution post-HSCT",
            "Post-HSCT immune reconstitution: T cell counts + TREC + mitogen proliferation monthly × 12 months",
            "Viral surveillance post-HSCT: CMV/EBV/adenovirus PCR weekly × 6 months (pre-engraftment risk)",
            "BCG site inspection if vaccinated before diagnosis: disseminated BCG = urgent infectious diseases referral",
        ],
        "key_ddx": [
            "ADA-SCID (T-B-NK- — ALL lymphocytes absent; dATP/ATP ratio diagnostic; rib cupping on CXR)",
            "RAG1/RAG2-SCID (T-B-NK+ — both T and B absent, NK preserved; V(D)J recombination failure)",
            "JAK3-SCID (T-B+NK- phenotype identical to IL2RG but JAK3 gene; AR; genetic panel distinguishes)",
            "MHC class II deficiency (CD4 T cell lymphopenia but NOT NK deficiency; AR; CIITA/RFX mutations)",
        ],
        "immunophenotype": "T-B+NK-",
        "sex_restriction": "male_only",
        "autoimmune_risk": False,
        "lymphoproliferation_risk": False,
        "enteropathy_risk": False,
        "scid_type": True,
        "agammaglobulinemia": False,
        "severity_options": [
            "Classic SCID-X1 (absent T+NK, birth onset, BCG complication if vaccinated before diagnosis)",
            "SCID-X1 post-gene-therapy (partial immune reconstitution, ongoing monitoring)",
        ],
        "complication_options": ["Recurrent pneumonia", "Failure to thrive", "BCG-osis", "Viral dissemination", "Candidiasis", "CMV pneumonitis"],
        "immunoglobulin_options": ["normal_non_functional", "severely_low", "low"],
        "treatments_used": ["HSCT (curative)", "Gene therapy (OTL-101/CARTEYVA)", "IVIG (bridge)", "TMP-SMX prophylaxis", "Antifungal prophylaxis", "TREC newborn screen"],
    },
    # -- ADA — ADA-SCID --------------------------------------------------------------------
    {
        "gene": "ADA",
        "alt_name": (
            "ADA (ADA-363aa-20q13.12 / AR — ADA-SCID — "
            "T-B-NK--IMMUNOPHENOTYPE-COMPLETE-ABSENCE-ALL-LYMPHOCYTES-Worst-SCID — "
            "RIB-COSTAL-CUPPING-Rachitic-Rosary-CXR-PATHOGNOMONIC — "
            "Gene-Therapy-Strimvelis-PEG-ADA-Enzyme-Replacement)"
        ),
        "protein": (
            "ADA -- 20q13.12 AR -- ADA-363aa -- "
            "Adenosine-Deaminase-41kDa-Purine-Salvage-Enzyme-Converts-Adenosine-Deoxyadenosine-to-Inosine -- "
            "ADA-Deficiency-Toxic-dATP-deoxyAdenosine-Accumulation-Lymphocyte-Apoptosis -- "
            "T-B-NK-IMMUNOPHENOTYPE-COMPLETE-ABSENCE-ALL-LYMPHOCYTES-Worst-All-SCIDs -- "
            "RIB-COSTAL-JUNCTION-CUPPING-Rachitic-Rosary-Like-CXR-PATHOGNOMONIC-ADA-SCID -- "
            "NEONATAL-COMPLETE-LYMPHOPENIA-Absolute-Lymphocyte-Count-Near-Zero -- "
            "HYPOGAMMAGLOBULINEMIA-All-Isotypes-Absent-B-Cells-Non-Functional-Apoptosis -- "
            "PEG-ADA-ADAGEN-REVCOVI-Enzyme-Replacement-Bridge-Non-Curative -- "
            "STRIMVELIS-GSK-Ex-Vivo-Retroviral-ADA-Gene-Therapy-FDA-Approved-Potentially-Curative -- "
            "PARTIAL-ADA-DEFICIENCY-Late-Onset-Adult-Milder-Some-Adults -- "
            "OMIM-Gene-608958-Disease-ADA-SCID-102700"
        ),
        "locus": "20q13.12",
        "protein_size": "363 aa / 41 kDa",
        "inheritance": (
            "AR; biallelic ADA mutations; ~15% of all SCID; "
            "ADA: purine salvage enzyme — deoxyadenosine → inosine; "
            "deficiency → deoxyadenosine/dATP accumulate → toxic to lymphocytes (all lineages) → T-B-NK- phenotype; "
            "RBC dATP:ATP ratio elevated = biochemical diagnostic marker; "
            "partial ADA deficiency (hypomorphic mutations) → late-onset/milder adult presentation; "
            "prenatal diagnosis: ADA activity in CVS/amniotic cells"
        ),
        "immunodeficiency_category": "SCID — ADA-SCID (ADA, purine toxicity; T-B-NK- phenotype; all lymphocytes absent — worst of all SCIDs)",
        "pathognomonic": (
            "T-B-NK- IMMUNOPHENOTYPE — COMPLETE ABSENCE OF ALL LYMPHOCYTES (worst of all SCIDs); "
            "RIB-COSTAL JUNCTION CUPPING ('rachitic rosary') on chest X-ray PATHOGNOMONIC for ADA-SCID "
            "(dATP accumulation disrupts chondrocyte metabolism → metaphyseal flaring); "
            "ABSOLUTE LYMPHOPENIA at birth (total lymphocyte count often <500 cells/µL); "
            "COMPLETE HYPOGAMMAGLOBULINEMIA (all isotypes); "
            "RBC dATP:ATP ratio elevated + absent bone marrow ADA activity = DEFINITIVE diagnostic"
        ),
        "treatment": (
            "PEG-ADA ENZYME REPLACEMENT (ADAGEN/REVCOVI): "
            "pegylated bovine ADA; IM weekly; reduces toxic metabolites + improves immunity; "
            "BRIDGE THERAPY ONLY — not curative; lymphocyte counts improve partially; "
            "GENE THERAPY — Strimvelis (GSK): ex-vivo retroviral ADA transduction of CD34+ HSCs; "
            "FDA approved; complete immune reconstitution reported; curative in many; "
            "Lentiviral ADA gene therapy (EFS-ADA): newer, improved safety profile; clinical trials; "
            "ALLOGENEIC HSCT: curative if MSD available; less preferred than gene therapy in absence of MSD; "
            "prophylactic TMP-SMX + antifungal pre-treatment; IVIG bridge; "
            "NO LIVE VACCINES EVER (pre-immune reconstitution); "
            "DIETARY: low adenosine diet of limited benefit — molecular/cellular therapy required"
        ),
        "key_features": [
            "T-B-NK- IMMUNOPHENOTYPE — ALL lymphocyte lineages absent (T, B, NK) — worst of all SCIDs",
            "RIB-COSTAL JUNCTION CUPPING ('rachitic rosary') on CXR — PATHOGNOMONIC for ADA-SCID",
            "ABSOLUTE LYMPHOPENIA at birth: total lymphocyte count near zero even without newborn screen",
            "COMPLETE HYPOGAMMAGLOBULINEMIA — all isotypes (IgG/IgM/IgA/IgE) absent",
            "RBC dATP:ATP RATIO elevated + absent ADA activity in bone marrow = definitive diagnostic",
            "PARTIAL ADA DEFICIENCY: late-onset adult form (hypomorphic mutations) — milder, often undiagnosed",
        ],
        "monitoring": [
            "Newborn TREC screen: absent TREC → SCID workup; lymphocyte subset phenotyping confirms T-B-NK-",
            "RBC dATP:ATP ratio + plasma deoxyadenosine: biochemical monitoring on PEG-ADA therapy",
            "PEG-ADA trough plasma ADA activity weekly × 1 month, then monthly (target range)",
            "Post-gene-therapy: T-cell TREC + ADA gene marking (PCR) + lymphocyte subsets monthly × 12 months",
            "Skeletal survey or CXR at diagnosis: rib-costal cupping confirms ADA-SCID",
            "Bone marrow aspirate for ADA activity if diagnosis uncertain (definitive if near zero)",
        ],
        "key_ddx": [
            "IL2RG-SCID-X1 (T-B+NK- — NK absent too, but B cells normal count; XLR males only; gamma-c chain)",
            "RAG1/RAG2-SCID (T-B-NK+ — NK cells preserved; no cupping on CXR; V(D)J failure not purine toxicity)",
            "Reticular dysgenesis (AK2 — T-B-NK- + myeloid failure + sensorineural deafness; ADA activity normal)",
            "Nutritional lymphopenia (severe malnutrition — low lymphocytes but NOT absent; recovers with nutrition)",
        ],
        "immunophenotype": "T-B-NK-",
        "sex_restriction": "both",
        "autoimmune_risk": False,
        "lymphoproliferation_risk": False,
        "enteropathy_risk": False,
        "scid_type": True,
        "agammaglobulinemia": False,
        "severity_options": [
            "Classic ADA-SCID (neonatal T-B-NK-, rib cupping, near-zero lymphocytes at birth)",
            "Partial ADA deficiency (late-onset adult, hypomorphic mutations, milder course)",
        ],
        "complication_options": ["Pneumocystis pneumonia", "CMV pneumonitis", "Failure to thrive", "Recurrent infections", "Skeletal abnormalities", "Candidiasis"],
        "immunoglobulin_options": ["undetectable", "severely_low"],
        "treatments_used": ["PEG-ADA enzyme replacement", "Gene therapy (Strimvelis)", "HSCT (curative)", "IVIG (bridge)", "TMP-SMX prophylaxis", "Antifungal prophylaxis"],
    },
    # -- RAG1 — RAG1-SCID / Omenn Syndrome -------------------------------------------------
    {
        "gene": "RAG1",
        "alt_name": (
            "RAG1 (RAG1-1043aa-11p13 / AR — RAG1-SCID-Omenn-Syndrome — "
            "T-B-NK+-SCID-Complete-Null-V-D-J-Recombination-Failure — "
            "OMENN-ERYTHRODERMA+ELEVATED-IgE+EOSINOPHILIA+ABSENT-NORMAL-IMMUNOGLOBULINS-PATHOGNOMONIC — "
            "RAG1-RAG2-Both-11p13-Head-to-Head-Divergent-Promoter-Gene-Panel-Tests-Both)"
        ),
        "protein": (
            "RAG1 -- 11p13 AR -- RAG1-1043aa -- "
            "Recombination-Activating-Gene-1-119kDa-V-D-J-Recombination-Endonuclease-Component -- "
            "RAG1-RAG2-Heterodimer-Initiates-DNA-Double-Strand-Breaks-V-D-J-Segments-TCR-BCR -- "
            "COMPLETE-NULL-RAG1-NO-TCR-NO-BCR-GENERATED-T-B-ABSENT-NK-PRESERVED-T-B-NK+ -- "
            "HYPOMORPHIC-RAG1-Partial-V-D-J-Autoreactive-T-Clones-Expand-Omenn-Syndrome -- "
            "OMENN-SYNDROME-ERYTHRODERMA-Alopecia-Eosinophilia-Elevated-IgE-Hepatosplenomegaly -- "
            "RAG1-RAG2-Both-Chromosome-11p13-Head-to-Head-Divergent-Promoter-Panel-Tests-Both -- "
            "Leaky-SCID-Hypomorphic-May-Present-Later-With-Autoimmunity-Lymphoproliferation -- "
            "HSCT-CURATIVE-All-RAG1-Phenotypes-Including-Omenn -- "
            "OMIM-Gene-RAG1-179615-Disease-SCID-601457-Omenn-603554"
        ),
        "locus": "11p13",
        "protein_size": "1043 aa / 119 kDa",
        "inheritance": (
            "AR; biallelic RAG1 mutations; "
            "RAG1 and RAG2 are head-to-head on chromosome 11p13 with divergent promoters; "
            "gene panels test both RAG1 and RAG2 simultaneously; "
            "complete null mutations → classic T-B-NK+ SCID; "
            "hypomorphic mutations → Omenn syndrome (partial V(D)J activity, autoreactive T clones); "
            "genotype-phenotype: null = severe SCID; hypomorphic = Omenn/leaky SCID; "
            "late-onset/atypical: hypomorphic RAG1 may present in adolescents/adults"
        ),
        "immunodeficiency_category": "SCID — RAG1-SCID / Omenn Syndrome (RAG1, V(D)J recombination failure; T-B-NK+ or Omenn phenotype)",
        "pathognomonic": (
            "COMPLETE RAG1 NULL: T-B-NK+ SCID — T and B cells completely absent, NK cells preserved; "
            "OMENN SYNDROME (hypomorphic RAG1): "
            "ERYTHRODERMA (universal red scaly rash) + ELEVATED IgE (paradoxically very high) + "
            "EOSINOPHILIA + ABSENT NORMAL IMMUNOGLOBULINS (despite elevated IgE — oligoclonal autoreactive only) + "
            "ALOPECIA + HEPATOSPLENOMEGALY = PATHOGNOMONIC COMBINATION; "
            "Omenn must be distinguished from SCID with graft-versus-host from maternal engraftment"
        ),
        "treatment": (
            "HAEMATOPOIETIC STEM CELL TRANSPLANTATION (HSCT) — CURATIVE for all RAG1 phenotypes; "
            "pre-HSCT immunosuppression for Omenn syndrome: cyclosporin A + corticosteroids "
            "(reduces autoreactive T-cell expansion before HSCT conditioning); "
            "IVIG: replace immunoglobulins (absent in both SCID and Omenn); "
            "prophylactic TMP-SMX + antifungal pre-HSCT; "
            "NO LIVE VACCINES at any time pre-immune reconstitution; "
            "Omenn: skin care critical — emollients + topical steroids (erythroderma); "
            "gene therapy trials: RAG1 lentiviral vector — in development, not yet FDA-approved; "
            "leaky/atypical RAG1 (adult presentation): may require immunosuppression first, then HSCT"
        ),
        "key_features": [
            "COMPLETE NULL: T-B-NK+ SCID — T and B absent (NK cells preserved — distinguishes from IL2RG/ADA)",
            "OMENN SYNDROME (hypomorphic): ERYTHRODERMA + ELEVATED IgE + EOSINOPHILIA + HEPATOSPLENOMEGALY — PATHOGNOMONIC",
            "RAG1 and RAG2 both on chromosome 11p13 (head-to-head) — gene panel tests both simultaneously",
            "V(D)J RECOMBINATION FAILURE — no TCR or BCR generated from germline DNA → no T or B cells",
            "ABSENT NORMAL IMMUNOGLOBULINS despite elevated IgE in Omenn (IgE = oligoclonal autoreactive, not protective)",
            "LEAKY SCID: hypomorphic RAG1 → partial V(D)J → autoreactive T clones → Omenn phenotype",
        ],
        "monitoring": [
            "TREC + KREC assay at birth: absent both = classic SCID; lymphocyte phenotyping (T-B-NK+)",
            "Omenn: skin biopsy if erythroderma — histology shows T-cell infiltration (not infection)",
            "IgE level + eosinophil count: very elevated in Omenn (paradox); IgG/IgM/IgA absent",
            "HLA typing + donor search URGENTLY at diagnosis (goal: HSCT before infection)",
            "Post-HSCT: TREC + KREC + lymphocyte subset reconstitution monthly × 12 months",
            "Maternal chimerism PCR: rule out engrafted maternal T cells mimicking Omenn syndrome",
        ],
        "key_ddx": [
            "IL2RG-SCID-X1 (T-B+NK- — NK absent; B cells normal count; males only; gamma-c chain)",
            "ADA-SCID (T-B-NK- — all absent including NK; rib cupping CXR; RBC dATP:ATP diagnostic)",
            "Omenn vs maternal GvHD (maternal T cells engrafted in SCID — PCR chimerism distinguishes)",
            "Netherton syndrome (erythroderma + eosinophilia + elevated IgE — SPINK5 mutation; not immunodeficient)",
        ],
        "immunophenotype": "T-B-NK+",
        "sex_restriction": "both",
        "autoimmune_risk": True,
        "lymphoproliferation_risk": False,
        "enteropathy_risk": False,
        "scid_type": True,
        "agammaglobulinemia": False,
        "severity_options": [
            "Classic RAG1-SCID (complete null, T-B-NK+, absent all T/B cells, severe infections neonatal)",
            "Omenn Syndrome (hypomorphic RAG1, erythroderma + elevated IgE + eosinophilia — pathognomonic)",
        ],
        "complication_options": ["Erythroderma (Omenn)", "Eosinophilia", "Hepatosplenomegaly", "CMV infection", "Candidiasis", "Failure to thrive"],
        "immunoglobulin_options": ["undetectable", "severely_low"],
        "treatments_used": ["HSCT (curative)", "Cyclosporin (Omenn pre-HSCT)", "IVIG (replacement)", "TMP-SMX prophylaxis", "Skin emollients (Omenn)", "Antifungal prophylaxis"],
    },
    # -- BTK — X-linked Agammaglobulinemia (XLA / Bruton) ----------------------------------
    {
        "gene": "BTK",
        "alt_name": (
            "BTK (BTK-659aa-Xq22.1 / XLR — X-linked-Agammaglobulinemia-XLA-Bruton — "
            "PERIPHERAL-B-CELLS<1pct+ALL-IMMUNOGLOBULIN-ISOTYPES-ZERO-PATHOGNOMONIC — "
            "ENTEROVIRAL-ENCEPHALITIS-Chronic-CNS-ECHO-Poliovirus-Life-Threatening — "
            "IVIG-Lifelong-NO-Live-Vaccines-OPV-Strictly-Contraindicated)"
        ),
        "protein": (
            "BTK -- Xq22.1 XLR -- BTK-659aa -- "
            "Bruton-Tyrosine-Kinase-76kDa-TEC-Family-Non-Receptor-Tyrosine-Kinase -- "
            "Pre-B-Cell-Receptor-BCR-Signalling-Pro-B-to-Pre-B-Maturation-Checkpoint -- "
            "BTK-LOF-B-Cell-Maturation-Block-at-Pro-B-Stage-Absent-Circulating-B-Cells -- "
            "PERIPHERAL-B-CELLS<1pct-Normal-5-15pct-PATHOGNOMONIC-Flow-Cytometry-CD19 -- "
            "ALL-IMMUNOGLOBULIN-ISOTYPES-VIRTUALLY-ZERO-IgG-IgM-IgA-IgE-Absent -- "
            "ABSENT-TONSILS-ADENOIDS-Clinically-Lymphoid-Tissue-Absent -- "
            "PRESENTS-AGE-6-18-MONTHS-Maternal-IgG-Wanes-Encapsulated-Bacterial-Infections -- "
            "ENTEROVIRAL-ENCEPHALITIS-ECHO-Poliovirus-Chronic-CNS-Infection-Life-Threatening -- "
            "IVIG-EVERY-3-4-WEEKS-LIFELONG-Trough-IgG-Target-600-800-mg-dL -- "
            "OMIM-Gene-300300-Disease-XLA-300755"
        ),
        "locus": "Xq22.1",
        "protein_size": "659 aa / 76 kDa",
        "inheritance": (
            "XLR; hemizygous BTK mutations in males; carrier females typically unaffected; "
            "de novo mutations ~30%; all affected are male (XLR); "
            "lyonisation in carrier females: skewed X-inactivation at B-cell level (diagnostic clue in carriers); "
            "females with XLA phenotype: BTK mutation on single X in Turner syndrome"
        ),
        "immunodeficiency_category": "Agammaglobulinemia — X-linked Agammaglobulinemia (BTK; B cell maturation block; T-B-NK+ with T and NK normal)",
        "pathognomonic": (
            "PERIPHERAL B CELLS <1% (normal 5-15%) PATHOGNOMONIC on flow cytometry (CD19/CD20); "
            "ALL IMMUNOGLOBULIN ISOTYPES VIRTUALLY ZERO (IgG/IgM/IgA/IgE); "
            "ABSENT TONSILS AND ADENOIDS on physical examination; "
            "PRESENTS AGE 6-18 MONTHS when maternal IgG wanes: recurrent encapsulated bacterial infections; "
            "ENTEROVIRAL ENCEPHALITIS: chronic CNS infection (echovirus/poliovirus) — progressive, life-threatening; "
            "T cells and NK cells NORMAL (distinguishes from SCID)"
        ),
        "treatment": (
            "IVIG LIFELONG — every 3-4 weeks; target trough IgG >600-800 mg/dL; "
            "SC-Ig (subcutaneous immunoglobulin) self-administered weekly — preferred by many patients; "
            "DOSE ESCALATION if recurrent infections despite trough >600 mg/dL (target >1000 mg/dL); "
            "NO LIVE VACCINES EVER — OPV (oral poliovirus vaccine) STRICTLY CONTRAINDICATED; "
            "attenuated poliovirus replication unchecked in XLA → paralytic polio risk; "
            "ENTEROVIRAL ENCEPHALITIS TREATMENT: "
            "IVIG + pleconaril (compassionate use antiviral) + intrathecal IVIG (severe CNS disease); "
            "ARTICULAR SEPTIC ARTHRITIS vs XLA ARTHRITIS: culture joint fluid — XLA-associated arthritis is sterile; "
            "GIARDIA: metronidazole/tinidazole (recurrent — treat empirically if chronic diarrhoea); "
            "prophylactic antibiotics (amoxicillin/azithromycin) for sinopulmonary disease prevention"
        ),
        "key_features": [
            "PERIPHERAL B CELLS <1% — absent circulating B cells on flow cytometry (CD19/CD20) — PATHOGNOMONIC",
            "ALL IMMUNOGLOBULIN ISOTYPES ZERO — IgG/IgM/IgA/IgE virtually undetectable",
            "ABSENT TONSILS AND ADENOIDS on physical exam — no lymphoid tissue development",
            "PRESENTS AGE 6-18 MONTHS: maternal IgG wanes → recurrent encapsulated bacterial infections (pneumococcus/H. influenzae/Giardia)",
            "ENTEROVIRAL ENCEPHALITIS: chronic CNS ECHO/poliovirus infection — life-threatening; IVIG + pleconaril + intrathecal Ig",
            "XLA ARTHRITIS: non-infectious, sterile joint inflammation — distinguish from septic arthritis by culture",
        ],
        "monitoring": [
            "IgG trough levels: every 3-4 months once stable on IVIG (target >600-800 mg/dL)",
            "B cell percentage (CD19/CD20): confirm <1% at diagnosis; monitoring during therapy",
            "Pulmonary function tests annually: sinopulmonary disease → bronchiectasis surveillance",
            "HRCT chest: if recurrent/chronic sinopulmonary infections — bronchiectasis assessment",
            "Enteroviral CSF PCR: any new neurological symptoms in XLA → urgent CSF analysis (echo/poliovirus)",
            "Audiometry annually: chronic sinusitis/otitis media → conductive hearing loss risk",
        ],
        "key_ddx": [
            "CVID (Common Variable Immunodeficiency — onset 20-30yr; B cells present but dysfunctional; can affect females)",
            "ADA-SCID (T-B-NK- — NK absent; T cells absent; much younger onset; rib cupping; not agammaglobulinemia alone)",
            "Transient hypogammaglobulinemia of infancy (THI — IgG low 6-18 months but B cells NORMAL count; self-resolves)",
            "Good syndrome (thymoma + agammaglobulinemia — adult onset; BTK normal; thymoma on CT chest)",
        ],
        "immunophenotype": "T+B-NK+",
        "sex_restriction": "male_only",
        "autoimmune_risk": False,
        "lymphoproliferation_risk": False,
        "enteropathy_risk": True,
        "scid_type": False,
        "agammaglobulinemia": True,
        "severity_options": [
            "Classic XLA (absent B cells/Ig, onset 6-18 months, recurrent bacterial infections)",
            "XLA with enteroviral encephalitis (chronic CNS echovirus — severe, life-threatening complication)",
        ],
        "complication_options": ["Recurrent pneumonia", "Sinusitis", "Bronchiectasis", "Enteroviral encephalitis", "Giardia diarrhoea", "Septic arthritis"],
        "immunoglobulin_options": ["undetectable", "undetectable"],
        "treatments_used": ["IVIG lifelong", "SC-Ig self-administration", "Pleconaril (enteroviral)", "Prophylactic antibiotics", "Metronidazole (Giardia)", "No live vaccines"],
    },
    # -- AIRE — APECED / APS-1 -------------------------------------------------------------
    {
        "gene": "AIRE",
        "alt_name": (
            "AIRE (AIRE-552aa-21q22.3 / AR — APECED-APS-1-Autoimmune-Polyendocrinopathy — "
            "MUCOCUTANEOUS-CANDIDIASIS+HYPOPARATHYROIDISM+ADRENAL-INSUFFICIENCY-TRIAD-PATHOGNOMONIC — "
            "Anti-IFN-omega-Antibodies-Diagnostic-Biomarker-95pct-Sensitivity-Specificity — "
            "Finnish-Founder-p.Arg257Ter-1-25000-Finland-ADRENAL-CRISIS-Emergency)"
        ),
        "protein": (
            "AIRE -- 21q22.3 AR -- AIRE-552aa -- "
            "Autoimmune-Regulator-58kDa-Transcription-Factor-Thymic-Medullary-Epithelial-Cells -- "
            "AIRE-Drives-Ectopic-Expression-Peripheral-Self-Antigens-Thymic-mTEC -- "
            "Negative-Selection-Autoreactive-T-Cells-AIRE-Maintains-Central-Tolerance -- "
            "AIRE-LOF-Loss-Negative-Selection-Autoreactive-T-Cells-Escape-Thymus -- "
            "APECED-TRIAD-MUCOCUTANEOUS-CANDIDIASIS-FIRST-age-1-5yr-Before-Endocrinopathies -- "
            "HYPOPARATHYROIDISM-Hypocalcaemia-Tetany-Mandatory-Calcium-Calcitriol -- "
            "ADRENAL-INSUFFICIENCY-Addison-Hydrocortisone-Stress-Dosing-MANDATORY -- "
            "Anti-IFN-omega-ANTIBODIES-Diagnostic-Biomarker-Greater-95pct-Sens-Spec -- "
            "FINNISH-FOUNDER-p.Arg257Ter-Exon6-1-25000-Finland-Sardinian-p.Arg139Ter -- "
            "OMIM-Gene-607358-Disease-APECED-240300"
        ),
        "locus": "21q22.3",
        "protein_size": "552 aa / 58 kDa",
        "inheritance": (
            "AR; biallelic AIRE mutations; "
            "Finnish founder: p.Arg257Ter (exon 6) — prevalence 1:25,000 Finland (1:100 carrier); "
            "Sardinian founder: p.Arg139Ter; Iranian Jewish founder: multiple; "
            "AIRE expressed in thymic medullary epithelial cells (mTEC): drives ectopic self-antigen expression → "
            "clonal deletion of autoreactive T cells (central tolerance); "
            "LOF → autoreactive T cells escape thymus → multi-organ autoimmunity"
        ),
        "immunodeficiency_category": "Immune Dysregulation — APECED/APS-1 (AIRE, central tolerance failure; polyendocrinopathy + candidiasis triad)",
        "pathognomonic": (
            "APECED TRIAD (2 of 3 = diagnosis; all 3 = PATHOGNOMONIC): "
            "(1) MUCOCUTANEOUS CANDIDIASIS — chronic oral/nail/vaginal, FIRST component, appears age 1-5yr; "
            "(2) HYPOPARATHYROIDISM — hypocalcaemia + tetany + seizures (calcium monitoring MANDATORY); "
            "(3) ADRENAL INSUFFICIENCY (Addison disease) — anti-21-hydroxylase antibodies; "
            "ANTI-IFN-OMEGA (IFN-ω) ANTIBODIES = DIAGNOSTIC BIOMARKER — >95% sensitivity/specificity; "
            "CANDIDIASIS APPEARS FIRST (age 1-5yr) — before endocrinopathies develop"
        ),
        "treatment": (
            "MUCOCUTANEOUS CANDIDIASIS: fluconazole/itraconazole oral (long-term antifungal); "
            "resistance surveillance — azole resistance (Candida albicans) with prolonged use; "
            "HYPOPARATHYROIDISM: calcium carbonate + calcitriol (active vitamin D) — LIFELONG; "
            "target serum calcium 2.0-2.1 mmol/L (lower end normal, avoids hypercalciuria/stones); "
            "PTH-replacement (recombinant hPTH 1-34/1-84) in refractory cases; "
            "ADRENAL INSUFFICIENCY: hydrocortisone + fludrocortisone; "
            "SICK-DAY RULE: 3x hydrocortisone dose during intercurrent illness/surgery — MANDATORY; "
            "ADRENAL CRISIS = EMERGENCY: IV hydrocortisone 100mg stat + fluid resuscitation + monitoring; "
            "ADDITIONAL COMPONENTS screening: anti-21-OHase (adrenal), anti-insulin (T1DM), anti-GAD, "
            "anti-TPO/anti-Tg (thyroid), anti-ovarian (premature ovarian failure), anti-IFN-ω; "
            "Vitamin B12 / cobalamin: pernicious anaemia component in 30-40%"
        ),
        "key_features": [
            "APECED TRIAD PATHOGNOMONIC: Mucocutaneous Candidiasis + Hypoparathyroidism + Adrenal Insufficiency",
            "CANDIDIASIS FIRST — appears age 1-5yr (before endocrinopathies); chronic oral/nail/vaginal Candida",
            "ANTI-IFN-OMEGA (IFN-ω) ANTIBODIES — diagnostic biomarker >95% sensitivity/specificity (blood test)",
            "FINNISH FOUNDER: p.Arg257Ter (exon 6) — prevalence 1:25,000 Finland; Sardinian: p.Arg139Ter",
            "ADRENAL CRISIS = LIFE-THREATENING EMERGENCY: stress-dosing hydrocortisone mandatory; sick-day rule",
            "AIRE maintains central tolerance: ectopic self-antigen expression in thymus → negative selection",
        ],
        "monitoring": [
            "Anti-IFN-ω antibodies at diagnosis: confirmatory biomarker (>95% sensitivity)",
            "Annual: anti-21-hydroxylase (adrenal), anti-parathyroid, anti-insulin, anti-GAD, anti-TPO, anti-ovarian",
            "Serum calcium + PTH + calcitriol: 3-monthly (hypoparathyroidism monitoring, avoid hypercalciuria)",
            "Cortisol stimulation test annually: assess adrenal reserve before frank insufficiency",
            "B12 + pernicious anaemia antibodies: every 2yr (pernicious anaemia 30-40% lifetime risk)",
            "Ophthalmology: keratoconjunctivitis component (20-30% APECED) — annual slit-lamp",
        ],
        "key_ddx": [
            "LRBA deficiency (hypogammaglobulinemia + autoimmunity; CTLA4 flow reduced; responds to abatacept; no candidiasis first)",
            "DiGeorge syndrome (22q11.2 del — thymic aplasia + hypoparathyroidism + cardiac; NOT candidiasis first; NOT AIRE)",
            "Polyglandular autoimmune syndrome type 2 (APS-2 — Addison + T1DM + thyroid; NO candidiasis; HLA-DR3/4 assoc; adults)",
            "Chronic mucocutaneous candidiasis (CMC) other causes (IL-17/STAT1-GOF/IL-17RA/F mutations — no endocrinopathy triad)",
        ],
        "immunophenotype": "T+B+NK+ (dysregulated)",
        "sex_restriction": "both",
        "autoimmune_risk": True,
        "lymphoproliferation_risk": False,
        "enteropathy_risk": True,
        "scid_type": False,
        "agammaglobulinemia": False,
        "severity_options": [
            "Full APECED triad (candidiasis + hypoparathyroidism + adrenal insufficiency — classic)",
            "Partial APECED (2 of 3 triad components + anti-IFN-ω positive, further components developing)",
        ],
        "complication_options": ["Adrenal crisis", "Hypocalcaemic tetany", "Oral candidiasis", "Premature ovarian failure", "Type 1 diabetes", "Pernicious anaemia"],
        "immunoglobulin_options": ["normal", "low"],
        "treatments_used": ["Fluconazole/itraconazole", "Calcium + calcitriol", "Hydrocortisone + fludrocortisone", "Sick-day rule (3x dose)", "Anti-IFN-ω screening", "Annual autoantibody panel"],
    },
    # -- LRBA — LRBA Deficiency ------------------------------------------------------------
    {
        "gene": "LRBA",
        "alt_name": (
            "LRBA (LRBA-2863aa-4q31.3 / AR — LRBA-Deficiency — "
            "HYPOGAMMAGLOBULINEMIA+AUTOIMMUNITY-COMBINED-PATHOGNOMONIC-DDx-CVID — "
            "CTLA4-Ig-ABATACEPT-Highly-Effective-Reverses-Autoimmunity-Weeks — "
            "IBD-Like-Enteropathy-50-70pct-Sirolimus-Alternative-HSCT-Severe)"
        ),
        "protein": (
            "LRBA -- 4q31.3 AR -- LRBA-2863aa -- "
            "Lipopolysaccharide-Responsive-Beige-Like-Anchor-Protein-319kDa-BEACH-Domain-WD40 -- "
            "LRBA-Regulates-CTLA4-Intracellular-Trafficking-Endosomal-Recycling-Cell-Surface -- "
            "LRBA-LOF-CTLA4-Fails-Recycle-Cell-Surface-T-Reg-Dysfunction-Effector-Dysregulation -- "
            "HYPOGAMMAGLOBULINEMIA-Low-IgG-IgA-IgM-CVID-Like-But-More-Autoimmunity -- "
            "AUTOIMMUNITY-Cytopenias-AIHA-ITP-Enteropathy-Hepatitis-Lung-Disease-Combined -- "
            "IBD-LIKE-ENTEROPATHY-50-70pct-Prominent-Villous-Atrophy-Colonoscopic-Ulceration -- "
            "ABATACEPT-CTLA4-Ig-HIGHLY-EFFECTIVE-Reverses-Autoimmunity-Weeks-Pathognomonic-Response -- "
            "CTLA4-FLOW-CYTOMETRY-Reduced-T-Reg-Cells-Functional-Screening -- "
            "IVIG-Hypogammaglobulinemia-Component-Sirolimus-Alternative-HSCT-Severe -- "
            "OMIM-Gene-606453-Disease-CVID8-614700"
        ),
        "locus": "4q31.3",
        "protein_size": "2863 aa / 319 kDa",
        "inheritance": (
            "AR; biallelic LRBA mutations; "
            "LRBA regulates CTLA4 recycling from lysosomes back to cell surface; "
            "LRBA loss → CTLA4 degraded in lysosomes → reduced surface CTLA4 on T-reg cells → "
            "T-reg dysfunction → effector T cell dysregulation → autoimmunity + hypogammaglobulinemia; "
            "phenotype mimics CVID but autoimmunity far more prominent; "
            "founder mutations reported in Iranian and Turkish populations"
        ),
        "immunodeficiency_category": "Immune Dysregulation — LRBA Deficiency (AR; CTLA4 trafficking failure; hypogammaglobulinemia + autoimmunity combined)",
        "pathognomonic": (
            "HYPOGAMMAGLOBULINEMIA + AUTOIMMUNITY COMBINED = LRBA PATHOGNOMONIC DDx FROM CVID; "
            "CVID: hypogammaglobulinemia but usually LESS autoimmunity; "
            "LRBA: hypogammaglobulinemia + PROMINENT AUTOIMMUNITY (cytopenias, enteropathy, hepatitis, lung disease); "
            "IBD-LIKE ENTEROPATHY prominent (50-70% of LRBA patients); "
            "CTLA4-Ig (ABATACEPT) DRAMATICALLY EFFECTIVE — reversal of autoimmunity within weeks; "
            "dramatic abatacept response SUPPORTS LRBA/CTLA4 diagnosis (diagnostic-therapeutic)"
        ),
        "treatment": (
            "ABATACEPT (CTLA4-Ig fusion protein): FIRST-LINE for autoimmune manifestations; "
            "IV abatacept monthly → dramatic reversal of cytopenias/enteropathy/hepatitis within weeks; "
            "subcutaneous abatacept weekly available; "
            "IVIG: for hypogammaglobulinemia component (maintain IgG trough >600-800 mg/dL); "
            "ENTEROPATHY: abatacept + IVIG; biological agents (vedolizumab, infliximab) second-line; "
            "SIROLIMUS (rapamycin): mTOR inhibitor — alternative if abatacept unavailable; "
            "HAEMATOPOIETIC STEM CELL TRANSPLANTATION (HSCT): curative in severe/refractory cases; "
            "CTLA4 expression monitoring by flow cytometry: reduced T-reg CTLA4 = functional screening; "
            "prophylactic antibiotics/antifungals for recurrent infections; "
            "PULMONARY: GLILD (granulomatous-lymphocytic ILD) — abatacept + IVIG first; rituximab second-line"
        ),
        "key_features": [
            "HYPOGAMMAGLOBULINEMIA + AUTOIMMUNITY COMBINED — key DDx from CVID (CVID has less autoimmunity)",
            "IBD-LIKE ENTEROPATHY — villous atrophy + colonoscopic ulceration prominent in 50-70% LRBA patients",
            "ABATACEPT (CTLA4-Ig) HIGHLY EFFECTIVE — reversal of autoimmunity within weeks (diagnostic-therapeutic)",
            "CTLA4 FLOW CYTOMETRY — reduced surface CTLA4 on T-regulatory cells = functional screening",
            "LRBA regulates CTLA4 endosomal recycling — LRBA loss → CTLA4 degraded, not recycled to surface",
            "DISTINGUISHES FROM CTLA4-H: LRBA is AR (biallelic); CTLA4-H is AD (heterozygous); LRBA protein absent on flow",
        ],
        "monitoring": [
            "LRBA protein expression by flow cytometry (intracellular staining): absent in LRBA deficiency",
            "CTLA4 surface expression on CD4+FoxP3+ T-reg cells: reduced in both LRBA and CTLA4-H",
            "IgG trough monthly on IVIG therapy (target >600-800 mg/dL); immunoglobulin panel at diagnosis",
            "Abatacept response assessment: CBC + hepatic enzymes + calprotectin monthly (first 3 months)",
            "Pulmonary: HRCT chest + PFTs 6-monthly (GLILD surveillance)",
            "Endoscopy: if persistent diarrhoea despite abatacept — assess enteropathy degree",
        ],
        "key_ddx": [
            "CVID (Common Variable ID — hypogammaglobulinemia; B cells present but dysfunctional; LESS autoimmunity; adults)",
            "CTLA4 Haploinsufficiency (AD; heterozygous; similar phenotype; CTLA4 flow reduced; LRBA protein normal)",
            "PIK3CD-APDS (lymphoproliferation + EBV/CMV; hypogammaglobulinemia; PI3K pathway — leniolisib responsive)",
            "RAG1 Omenn (erythroderma + eosinophilia + IgE — different clinical context; younger onset; SCID range)",
        ],
        "immunophenotype": "T+B+NK+ (dysregulated, reduced T-regs)",
        "sex_restriction": "both",
        "autoimmune_risk": True,
        "lymphoproliferation_risk": True,
        "enteropathy_risk": True,
        "scid_type": False,
        "agammaglobulinemia": False,
        "severity_options": [
            "LRBA with prominent enteropathy (IBD-like colitis + villous atrophy + cytopenias — most common)",
            "LRBA with pulmonary GLILD (granulomatous-lymphocytic ILD + hypogammaglobulinemia + organomegaly)",
        ],
        "complication_options": ["AIHA", "ITP", "Enteropathy (IBD-like)", "Pulmonary GLILD", "Hepatitis", "Recurrent sinopulmonary infections"],
        "immunoglobulin_options": ["severely_low", "low"],
        "treatments_used": ["Abatacept (CTLA4-Ig)", "IVIG replacement", "Sirolimus (alternative)", "HSCT (severe/refractory)", "Vedolizumab (enteropathy)", "Prophylactic antibiotics"],
    },
    # -- CTLA4 — CTLA4 Haploinsufficiency --------------------------------------------------
    {
        "gene": "CTLA4",
        "alt_name": (
            "CTLA4 (CTLA4-223aa-2q33.2 / AD-Haploinsufficiency — CTLA4-Haploinsufficiency-CTLA4-H — "
            "SPLENOMEGALY+LYMPHADENOPATHY+AUTOIMMUNE-CYTOPENIAS+HYPOGAMMA+LYMPHOCYTIC-INFILTRATION-PATHOGNOMONIC — "
            "ABATACEPT-Highly-Effective-Functional-CTLA4-Replacement — "
            "Distinguished-LRBA-AD-vs-AR-LRBA-Flow-Absent-LRBA-Protein)"
        ),
        "protein": (
            "CTLA4 -- 2q33.2 AD-haploinsufficiency -- CTLA4-223aa -- "
            "Cytotoxic-T-Lymphocyte-Antigen-4-25kDa-Checkpoint-Inhibitor-B7-CD80-CD86-Outcompetes-CD28 -- "
            "CTLA4-Constitutively-Expressed-T-Regulatory-Cells-Suppresses-T-Cell-Activation -- "
            "CTLA4-Outcompetes-CD28-For-B7-CD80-CD86-On-APC-Suppresses-T-Cell-Activation -- "
            "CTLA4-Haploinsufficiency-Heterozygous-LOF-AD-Insufficient-CTLA4-Surface-Expression -- "
            "T-Cell-Dysregulation-Multi-Organ-Autoimmunity-Lymphocytic-Infiltration -- "
            "SPLENOMEGALY-LYMPHADENOPATHY-AUTOIMMUNE-CYTOPENIAS-HYPOGAMMAGLOBULINEMIA-GLILD -- "
            "LYMPHOCYTIC-INTERSTITIAL-LUNG-GLILD-Sarcoid-Like-CT-CHECK-CTLA4-Before-Immunosuppression -- "
            "ABATACEPT-CTLA4-Ig-HIGHLY-EFFECTIVE-Provides-Functional-CTLA4-Replacement -- "
            "DISTINGUISHED-LRBA-CTLA4H-AD-One-Hit-LRBA-AR-Both-Hits-LRBA-Protein-Absent-LRBA -- "
            "OMIM-Gene-123890-Disease-CTLA4-Haploinsufficiency-616100"
        ),
        "locus": "2q33.2",
        "protein_size": "223 aa / 25 kDa",
        "inheritance": (
            "AD haploinsufficiency; heterozygous LOF CTLA4 mutations; "
            "50% penetrance (not all carriers manifest disease); "
            "family history of autoimmunity/immunodeficiency in AD pattern; "
            "CTLA4 constitutively expressed on T-regulatory cells; "
            "heterozygous LOF → reduced surface CTLA4 → insufficient T-cell checkpoint → "
            "multi-organ T-cell infiltration and autoimmunity"
        ),
        "immunodeficiency_category": "Immune Dysregulation — CTLA4 Haploinsufficiency (AD; heterozygous CTLA4 LOF; multi-organ autoimmunity + hypogammaglobulinemia)",
        "pathognomonic": (
            "SPLENOMEGALY + LYMPHADENOPATHY + AUTOIMMUNE CYTOPENIAS (AIHA/ITP/neutropenia) + "
            "HYPOGAMMAGLOBULINEMIA + LYMPHOCYTIC INFILTRATION OF LUNGS/GUT/BRAIN = PATHOGNOMONIC PATTERN; "
            "PARADOX: recurrent infections despite splenomegaly (hypogammaglobulinemia + lymphocyte dysfunction); "
            "GLILD (granulomatous-lymphocytic interstitial lung disease) resembles sarcoid on CT — "
            "CHECK CTLA4 GENE BEFORE IMMUNOSUPPRESSING WITH STEROIDS; "
            "ABATACEPT response: dramatic = diagnostic signal supporting CTLA4/LRBA"
        ),
        "treatment": (
            "ABATACEPT (CTLA4-Ig fusion protein): HIGHLY EFFECTIVE — provides functional CTLA4 replacement; "
            "IV monthly or SC weekly; dramatic reversal of cytopenias/lung disease/organomegaly; "
            "IVIG: for hypogammaglobulinemia component (target IgG trough >600-800 mg/dL); "
            "CORTICOSTEROIDS: used acutely for autoimmune cytopenias (AIHA/ITP); steroid-sparing preferred; "
            "RITUXIMAB: B-cell depletion for AIHA/ITP refractory to abatacept; "
            "PULMONARY GLILD: abatacept first-line; rituximab second-line; azathioprine/mycophenolate adjunct; "
            "HAEMATOPOIETIC STEM CELL TRANSPLANTATION (HSCT): curative for severe/progressive cases; "
            "SIROLIMUS (rapamycin): alternative mTOR-based immunosuppression; "
            "CTLA4 flow cytometry: monitor T-reg CTLA4 expression pre/post-abatacept"
        ),
        "key_features": [
            "SPLENOMEGALY + LYMPHADENOPATHY + AUTOIMMUNE CYTOPENIAS + HYPOGAMMAGLOBULINEMIA — PATHOGNOMONIC pattern",
            "LYMPHOCYTIC INTERSTITIAL LUNG DISEASE (GLILD) — sarcoid-like on CT; check CTLA4 before steroids alone",
            "PARADOX: recurrent infections + splenomegaly (hypogamma + immune dysregulation coexist)",
            "ABATACEPT (CTLA4-Ig) HIGHLY EFFECTIVE — provides exogenous functional CTLA4; dramatic response = diagnosis",
            "CTLA4 FLOW CYTOMETRY: reduced surface CTLA4 on T-regulatory cells (CD4+FoxP3+)",
            "DISTINGUISHED FROM LRBA: CTLA4-H is AD (one-hit); LRBA is AR (two-hit); LRBA flow: absent LRBA protein",
        ],
        "monitoring": [
            "CTLA4 surface expression by flow cytometry (CD4+FoxP3+ T-regs): reduced vs normal (screening + monitoring)",
            "LRBA protein intracellular flow: normal in CTLA4-H (distinguishes from LRBA deficiency)",
            "HRCT chest 6-monthly (GLILD surveillance): new nodules/ground-glass = abatacept escalation",
            "CBC + reticulocytes + direct Coombs monthly (autoimmune cytopenias monitoring on abatacept)",
            "IgG trough monthly on IVIG; immunoglobulin panel annually",
            "Brain MRI if neurological symptoms: CTLA4-H lymphocytic infiltration of CNS reported",
        ],
        "key_ddx": [
            "LRBA deficiency (AR biallelic; phenotype similar; LRBA protein absent on flow; both respond to abatacept)",
            "PIK3CD-APDS (lymphoproliferation + EBV/CMV susceptibility; GOF not LOF; leniolisib targeted therapy)",
            "Sarcoidosis (GLILD mimics sarcoid CT; CTLA4-H check before diagnosing idiopathic sarcoid in young patients)",
            "CVID (Common Variable ID — hypogammaglobulinemia; LESS autoimmunity; CTLA4/LRBA genetic panel key)",
        ],
        "immunophenotype": "T+B+NK+ (dysregulated, reduced T-reg function)",
        "sex_restriction": "both",
        "autoimmune_risk": True,
        "lymphoproliferation_risk": True,
        "enteropathy_risk": True,
        "scid_type": False,
        "agammaglobulinemia": False,
        "severity_options": [
            "CTLA4-H with pulmonary GLILD (sarcoid-like CT + cytopenias + hypogammaglobulinemia — most common severe)",
            "CTLA4-H with autoimmune cytopenias (AIHA/ITP + splenomegaly + lymphadenopathy — common moderate)",
        ],
        "complication_options": ["AIHA", "ITP", "Pulmonary GLILD", "Enteropathy", "Splenomegaly", "Recurrent infections"],
        "immunoglobulin_options": ["severely_low", "low"],
        "treatments_used": ["Abatacept (CTLA4-Ig)", "IVIG replacement", "Rituximab (cytopenias)", "Corticosteroids (acute)", "Sirolimus (alternative)", "HSCT (severe)"],
    },
    # -- PIK3CD — APDS / APDS1 ------------------------------------------------------------
    {
        "gene": "PIK3CD",
        "alt_name": (
            "PIK3CD (PIK3CD-1044aa-1p36.22 / AD-GOF — Activated-PI3Kd-Syndrome-APDS-APDS1 — "
            "LYMPHOPROLIFERATION+EBV-CMV-SUSCEPTIBILITY+RECURRENT-RESPIRATORY-INFECTIONS+HYPOGAMMAGLOBULINEMIA-PATHOGNOMONIC — "
            "LENIOLISIB-Joenja-FDA-2023-First-Targeted-PI3Kd-Inhibitor-APDS — "
            "EBV-Driven-Lymphoma-Risk-20pct-Annual-EBV-Viral-Load-MANDATORY)"
        ),
        "protein": (
            "PIK3CD -- 1p36.22 AD-GOF -- PIK3CD-1044aa -- "
            "PI3K-Catalytic-Subunit-Delta-119kDa-Lymphocyte-Expressed-Lipid-Kinase -- "
            "PIK3CD-GOF-Constitutive-PI3K-AKT-mTOR-Pathway-Activation-Lymphocytes -- "
            "Terminal-Lymphocyte-Differentiation-Failure-Senescent-Effector-Memory-T-Cell-Accumulation -- "
            "LYMPHOPROLIFERATION-Hepatosplenomegaly-Lymphadenopathy-EBV-CMV-Susceptibility -- "
            "EBV-DRIVEN-LYMPHOMA-RISK-UP-TO-20pct-Annual-EBV-Viral-Load-MANDATORY -- "
            "RECURRENT-SINOPULMONARY-INFECTIONS-Bronchiectasis-2nd-Decade-Hypogammaglobulinemia -- "
            "LENIOLISIB-JOENJA-FDA-APPROVED-2023-Selective-PI3Kd-Inhibitor-First-Targeted-APDS -- "
            "PIK3R1-GOF-APDS2-Distinct-Gene-Similar-Phenotype-Both-PI3Kd-Inhibitor -- "
            "RAPAMYCIN-SIROLIMUS-mTOR-Inhibitor-Alternative-Prior-Leniolisib-Availability -- "
            "OMIM-Gene-602839-Disease-APDS1-615513"
        ),
        "locus": "1p36.22",
        "protein_size": "1044 aa / 119 kDa",
        "inheritance": (
            "AD gain-of-function (GOF); heterozygous PIK3CD activating mutations; "
            "variable penetrance; de novo mutations common; "
            "PI3Kδ expressed in lymphocytes: normally activated transiently by antigen receptor signalling; "
            "GOF → constitutive PI3K-AKT-mTOR activation → terminal differentiation failure → "
            "senescent effector memory T cells accumulate + B cells fail to class-switch; "
            "PIK3R1 GOF → APDS2 (regulatory subunit; similar phenotype; both respond to PI3Kδ inhibitor)"
        ),
        "immunodeficiency_category": "Immune Dysregulation — APDS/APDS1 (PIK3CD GOF; constitutive PI3K-AKT-mTOR; lymphoproliferation + EBV/CMV susceptibility)",
        "pathognomonic": (
            "LYMPHOPROLIFERATION (hepatosplenomegaly + lymphadenopathy) + "
            "EBV/CMV SUSCEPTIBILITY (chronic active infection, lymphoma risk up to 20%) + "
            "RECURRENT RESPIRATORY TRACT INFECTIONS (sinopulmonary, bronchiectasis by 2nd decade) + "
            "HYPOGAMMAGLOBULINEMIA = APDS PATHOGNOMONIC COMBINATION; "
            "EBV-DRIVEN LYMPHOMA: up to 20% lifetime risk — annual EBV viral load monitoring MANDATORY; "
            "LENIOLISIB (Joenja) FDA approved 2023: first targeted PI3Kδ inhibitor for APDS"
        ),
        "treatment": (
            "LENIOLISIB (Joenja): selective PI3Kδ inhibitor — FDA approved March 2023; "
            "reduces lymphoproliferation + improves B cell class-switching + reduces infection frequency; "
            "dosing: 70mg oral BD (adults); "
            "RAPAMYCIN (sirolimus): mTOR inhibitor — reduces lymphoproliferation; "
            "pre-dated leniolisib availability; still used in resource-limited settings; "
            "IVIG: for hypogammaglobulinemia (target trough IgG >600-800 mg/dL); "
            "ANTIVIRAL PROPHYLAXIS: aciclovir (HSV/VZV) + valganciclovir (CMV high risk) long-term; "
            "EBV MONITORING: annual EBV DNA PCR in blood — if viral load rises → intensify antiviral; "
            "LYMPHOMA SURVEILLANCE: annual PET/CT if persistent lymphadenopathy; "
            "prophylactic antibiotics: azithromycin/amoxicillin for sinopulmonary disease prevention; "
            "HAEMATOPOIETIC STEM CELL TRANSPLANTATION (HSCT): curative in severe/refractory/lymphoma; "
            "PIK3R1 GOF (APDS2): same treatment approach as APDS1 (same PI3Kδ pathway)"
        ),
        "key_features": [
            "LYMPHOPROLIFERATION + EBV/CMV SUSCEPTIBILITY + RECURRENT RESPIRATORY INFECTIONS + HYPOGAMMAGLOBULINEMIA — PATHOGNOMONIC",
            "EBV-DRIVEN LYMPHOMA RISK: up to 20% lifetime — annual EBV viral load monitoring MANDATORY",
            "LENIOLISIB (Joenja): FDA approved 2023 — first targeted PI3Kδ inhibitor; reduces lymphoproliferation",
            "SENESCENT EFFECTOR MEMORY T CELLS: PI3K-AKT-mTOR constitutive → T cell terminal differentiation failure",
            "BRONCHIECTASIS by 2nd decade: recurrent sinopulmonary infections → progressive structural lung damage",
            "PIK3R1 GOF → APDS2 (distinct gene, similar phenotype) — both respond to PI3Kδ inhibitor (leniolisib)",
        ],
        "monitoring": [
            "Annual EBV DNA PCR (blood): lymphoma surveillance — rising viral load = intensify antiviral + imaging",
            "Annual PET/CT or CT neck/chest/abdomen/pelvis: lymphoma screening if persistent lymphadenopathy",
            "IgG trough monthly on IVIG; immunoglobulin panel annually",
            "Pulmonary function tests + HRCT chest: bronchiectasis monitoring (annually from age 10yr)",
            "Leniolisib: LFTs + CBC monthly first 3 months (hepatotoxicity monitoring); quarterly thereafter",
            "Flow cytometry: B cell subsets (switched memory B cells) + T cell senescence markers (CD57) — leniolisib response",
        ],
        "key_ddx": [
            "CTLA4 Haploinsufficiency (lymphoproliferation + hypogamma; AD; CTLA4 flow reduced; no PI3K pathway)",
            "LRBA deficiency (hypogamma + autoimmunity + enteropathy; AR; abatacept responsive; LRBA protein absent)",
            "EBV-driven lymphoma without underlying PID (no sinopulmonary infections; no hypogammaglobulinemia; no family history)",
            "CVID (hypogammaglobulinemia; B cells present dysfunctional; LESS lymphoproliferation; older onset; no EBV predilection)",
        ],
        "immunophenotype": "T+B+NK+ (dysregulated, senescent T cells, reduced switched-memory B cells)",
        "sex_restriction": "both",
        "autoimmune_risk": True,
        "lymphoproliferation_risk": True,
        "enteropathy_risk": True,
        "scid_type": False,
        "agammaglobulinemia": False,
        "severity_options": [
            "APDS with EBV lymphoproliferation (splenomegaly + EBV-positive lymphadenopathy + recurrent RTIs — most common)",
            "APDS with lymphoma (EBV-driven B cell lymphoma — leniolisib/HSCT required)",
        ],
        "complication_options": ["EBV lymphoproliferation", "Lymphoma (EBV-driven)", "Recurrent pneumonia", "Bronchiectasis", "CMV infection", "Splenomegaly"],
        "immunoglobulin_options": ["severely_low", "low"],
        "treatments_used": ["Leniolisib (Joenja)", "Sirolimus/rapamycin (alternative)", "IVIG replacement", "Valganciclovir prophylaxis", "Annual EBV PCR", "HSCT (severe/lymphoma)"],
    },
]


def _build_cohort() -> list:
    cohort = []
    for i, entry in enumerate(IMMUNO_GENES):
        seed = SEED_BASE + i
        rng = random.Random(seed)
        for j in range(40):
            gene = entry["gene"]
            severity = rng.choice(entry["severity_options"])
            complication = rng.sample(
                entry["complication_options"],
                k=min(3, len(entry["complication_options"]))
            )
            treatment = rng.choice(entry["treatments_used"])
            immunoglobulin = rng.choice(entry["immunoglobulin_options"])

            # Age ranges are disease-appropriate
            if gene in ("IL2RG", "ADA", "RAG1"):  # SCID — neonatal/infant
                age_dx = rng.uniform(0.1, 2.0)
                sex = "M"  # XLR for IL2RG, but ADA/RAG1 both — use rng for ADA/RAG1
                if gene in ("ADA", "RAG1"):
                    sex = rng.choice(["M", "F"])
            elif gene == "BTK":  # XLA — maternal IgG wanes
                age_dx = rng.uniform(0.5, 3.0)
                sex = "M"
            elif gene == "AIRE":  # APECED — candidiasis first age 1-5yr
                age_dx = rng.uniform(1.0, 8.0)
                sex = rng.choice(["M", "F"])
            else:  # LRBA, CTLA4, PIK3CD — childhood to adult
                age_dx = rng.uniform(2.0, 40.0)
                sex = rng.choice(["M", "F"])

            autoimmune_event = rng.random() < (0.65 if entry["autoimmune_risk"] else 0.04)
            lymphoproliferation_event = rng.random() < (0.55 if entry["lymphoproliferation_risk"] else 0.02)
            enteropathy_event = rng.random() < (0.55 if entry["enteropathy_risk"] else 0.03)
            fu_yrs = rng.uniform(0.5, 18.0)

            cohort.append({
                "patient_id": f"{gene}-{seed:04d}-{j+1:02d}",
                "gene": gene,
                "sex": sex,
                "severity": severity,
                "complications": complication,
                "treatment": treatment,
                "immunophenotype": entry["immunophenotype"],
                "immunoglobulin_level_category": immunoglobulin,
                "age_at_dx_yrs": round(age_dx, 1),
                "follow_up_yrs": round(fu_yrs, 1),
                "autoimmune_event": autoimmune_event,
                "lymphoproliferation_event": lymphoproliferation_event,
                "enteropathy_event": enteropathy_event,
                "scid_type": entry["scid_type"],
                "agammaglobulinemia": entry["agammaglobulinemia"],
                "seed": seed,
            })
    return cohort


def generate_overview() -> dict:
    cohort = _build_cohort()
    gene_counts = {}
    type_counts = {}
    scid_genes = []
    agamma_genes = []
    autoimmune_genes = []
    lymphoproliferation_genes = []
    gene_summary = {}

    for entry in IMMUNO_GENES:
        g = entry["gene"]
        pts = [p for p in cohort if p["gene"] == g]
        if entry["scid_type"]:
            scid_genes.append(g)
        if entry["agammaglobulinemia"]:
            agamma_genes.append(g)
        if entry["autoimmune_risk"]:
            autoimmune_genes.append(g)
        if entry["lymphoproliferation_risk"]:
            lymphoproliferation_genes.append(g)

        cat = entry["immunodeficiency_category"].split("—")[0].strip()
        type_counts[cat] = type_counts.get(cat, 0) + len(pts)
        gene_counts[g] = len(pts)
        gene_summary[g] = {
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "immunodeficiency_category": entry["immunodeficiency_category"],
            "pathognomonic": entry["pathognomonic"][:300],
            "immunophenotype": entry["immunophenotype"],
            "scid_type": entry["scid_type"],
            "agammaglobulinemia": entry["agammaglobulinemia"],
            "autoimmune_risk": entry["autoimmune_risk"],
            "lymphoproliferation_risk": entry["lymphoproliferation_risk"],
        }

    return {
        "title": "Hereditary-Primary-Immunodeficiency-Atlas — Complete 8-Gene Hereditary Primary Immunodeficiency Atlas",
        "n_genes": len(IMMUNO_GENES),
        "n_patients": len(cohort),
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "immunodeficiency_categories": {
            "SCID": ["IL2RG", "ADA", "RAG1"],
            "Agammaglobulinemia": ["BTK"],
            "Immune dysregulation": ["AIRE", "LRBA", "CTLA4", "PIK3CD"],
        },
        "inheritance_map": {
            "IL2RG": "XLR", "ADA": "AR", "RAG1": "AR", "BTK": "XLR",
            "AIRE": "AR", "LRBA": "AR", "CTLA4": "AD-haploinsufficiency", "PIK3CD": "AD-GOF",
        },
        "immunophenotype_map": {
            "IL2RG": "T-B+NK-",
            "ADA": "T-B-NK-",
            "RAG1": "T-B-NK+ (null) / Omenn (hypomorphic)",
            "BTK": "T+B-NK+",
            "AIRE": "T+B+NK+ (dysregulated)",
            "LRBA": "T+B+NK+ (dysregulated, reduced T-regs)",
            "CTLA4": "T+B+NK+ (dysregulated, reduced T-reg function)",
            "PIK3CD": "T+B+NK+ (dysregulated, senescent T cells)",
        },
        "key_clinical_pearls": [
            "IL2RG-SCID-X1: T-B+NK- IMMUNOPHENOTYPE PATHOGNOMONIC (absent T + NK; normal but non-functional B cells); MOST COMMON SCID WESTERN countries (45-50%); TREC assay (newborn screen) detects absent T cells at birth; BCG BEFORE DIAGNOSIS = BCG-OSIS DISSEMINATED FATAL; NO LIVE VACCINES EVER; gene therapy OTL-101/CARTEYVA FDA curative; MATERNAL ENGRAFTMENT may mask disease — look for oligoclonal T cells",
            "ADA-SCID: T-B-NK- IMMUNOPHENOTYPE — COMPLETE ABSENCE ALL LYMPHOCYTES (worst of all SCIDs); RIB-COSTAL JUNCTION CUPPING ('rachitic rosary') on CXR PATHOGNOMONIC; RBC dATP:ATP ratio elevated + absent ADA activity = definitive diagnostic; PEG-ADA (ADAGEN/REVCOVI) = enzyme replacement bridge (non-curative); STRIMVELIS gene therapy FDA approved — potentially curative; PARTIAL ADA DEFICIENCY = late-onset adult milder form",
            "RAG1-SCID/Omenn: COMPLETE NULL = T-B-NK+ SCID (NK preserved — distinguishes from IL2RG/ADA); HYPOMORPHIC = OMENN SYNDROME — ERYTHRODERMA + ELEVATED IgE + EOSINOPHILIA + HEPATOSPLENOMEGALY + ABSENT NORMAL IMMUNOGLOBULINS PATHOGNOMONIC; RAG1 + RAG2 both on 11p13 (head-to-head) — panel tests both simultaneously; HSCT curative all phenotypes; distinguish Omenn from maternal GvHD (PCR chimerism)",
            "BTK-XLA: PERIPHERAL B CELLS <1% PATHOGNOMONIC (flow cytometry CD19/CD20); ALL IMMUNOGLOBULIN ISOTYPES VIRTUALLY ZERO; ABSENT TONSILS/ADENOIDS on examination; PRESENTS AGE 6-18 MONTHS (maternal IgG wanes); ENTEROVIRAL ENCEPHALITIS (chronic ECHO/poliovirus CNS infection) LIFE-THREATENING — IVIG + pleconaril + intrathecal Ig; IVIG LIFELONG every 3-4 weeks (trough >600-800 mg/dL); NO LIVE VACCINES (OPV strictly contraindicated — poliovirus replication unchecked in XLA)",
            "AIRE-APECED: APECED TRIAD PATHOGNOMONIC — MUCOCUTANEOUS CANDIDIASIS (FIRST, age 1-5yr) + HYPOPARATHYROIDISM (hypocalcaemia/tetany) + ADRENAL INSUFFICIENCY (Addison); 2 of 3 = diagnosis; ALL 3 = PATHOGNOMONIC; ANTI-IFN-OMEGA ANTIBODIES diagnostic biomarker >95% sensitivity/specificity; FINNISH FOUNDER p.Arg257Ter 1:25,000; ADRENAL CRISIS = EMERGENCY (100mg IV hydrocortisone stat); SICK-DAY RULE (3x dose) MANDATORY; annual autoantibody panel (adrenal/thyroid/ovarian/GAD/insulin)",
            "LRBA/CTLA4/PIK3CD IMMUNE DYSREGULATION: LRBA (AR) + CTLA4-H (AD) — BOTH respond dramatically to ABATACEPT (CTLA4-Ig) within weeks; LRBA distinguished by absent LRBA protein on flow cytometry; CTLA4-H has GLILD (sarcoid-like CT lung) — CHECK CTLA4 BEFORE IMMUNOSUPPRESSING; PIK3CD-APDS — LENIOLISIB (Joenja) FDA 2023 first targeted PI3Kd inhibitor; EBV LYMPHOMA RISK 20% in APDS — ANNUAL EBV PCR MANDATORY; HYPOGAMMAGLOBULINEMIA + AUTOIMMUNITY = key DDx from CVID (CVID has less autoimmunity)",
        ],
        "clinical_emergency_flags": [
            "BCG-OSIS IN SCID: any infant receiving BCG vaccine who subsequently receives SCID diagnosis has active disseminated mycobacterial infection (BCG-osis) — URGENT infectious diseases + immunology; anti-mycobacterial therapy (isoniazid + rifampicin + ethambutol) IMMEDIATELY; BCG-osis is FATAL without both anti-mycobacterial therapy AND immune reconstitution (HSCT/gene therapy); TREC newborn screening prevents this by detecting SCID before BCG",
            "ADRENAL CRISIS IN APECED: any APECED patient with fever/vomiting/collapse — PRESUME ADRENAL CRISIS; IV hydrocortisone 100mg stat (adult; 2-4mg/kg paediatric) + 0.9% saline fluid resuscitation + IV dextrose (hypoglycaemia); do NOT wait for cortisol levels; sick-day rule must be rehearsed at every clinic visit; medical alert bracelet MANDATORY; IM hydrocortisone self-injection kit for home emergencies",
            "ENTEROVIRAL ENCEPHALITIS IN XLA: any BTK/XLA patient with new neurological symptoms (headache/seizures/personality change/deteriorating cognition) = ASSUME ENTEROVIRAL ENCEPHALITIS until proven otherwise; CSF enteroviral PCR STAT (ECHO/poliovirus/coxsackie); IVIG high-dose + pleconaril (compassionate use) + intrathecal IVIG if severe; progression is IRREVERSIBLE without early treatment; avoid OPV in ALL household contacts of XLA patients",
            "EBV LYMPHOMA SURVEILLANCE IN APDS: all PIK3CD-APDS patients require ANNUAL EBV DNA PCR in peripheral blood; rising EBV viral load = intensify antiviral (valganciclovir) + urgent PET/CT (lymphoma screening); up to 20% lifetime lymphoma risk; leniolisib reduces lymphoproliferation and may reduce lymphoma risk; do NOT attribute lymphadenopathy in APDS to infection alone without EBV monitoring and imaging",
            "HYPOCALCAEMIC TETANY IN APECED: APECED patients with hypoparathyroidism may develop acute hypocalcaemic tetany/seizures; IV calcium gluconate 10% (10mL over 10min) stat + calcium infusion maintenance; check serum calcium + magnesium (hypomagnesaemia impairs PTH secretion and worsens hypocalcaemia); long-term: calcitriol + calcium supplements titrated to serum Ca 2.0-2.1 mmol/L; recombinant PTH (teriparatide) if refractory",
        ],
        "gene_summary": gene_summary,
        "scid_genes": scid_genes,
        "agammaglobulinemia_genes": agamma_genes,
        "autoimmune_risk_genes": autoimmune_genes,
        "lymphoproliferation_risk_genes": lymphoproliferation_genes,
        "diagnostic_algorithm": {
            "Step_1": "Classify presentation: (A) Neonatal severe infections + absent lymphocytes → SCID screen (TREC assay); (B) Recurrent encapsulated bacterial infections from 6-18 months + absent B cells → XLA (BTK); (C) Candidiasis before age 5 + endocrinopathy → APECED (AIRE); (D) Hypogammaglobulinemia + autoimmunity → LRBA/CTLA4/PIK3CD panel",
            "Step_2": "Immunophenotyping (lymphocyte subsets): T-B+NK- = IL2RG/JAK3; T-B-NK- = ADA; T-B-NK+ = RAG1/RAG2; T+B-NK+ = BTK/XLA; T+B+NK+ dysregulated = AIRE/LRBA/CTLA4/PIK3CD",
            "Step_3": "Immunoglobulin levels: ALL isotypes undetectable (IgG/IgM/IgA/IgE near zero) → XLA (BTK) or SCID; Low IgG + autoimmunity → LRBA/CTLA4; Low IgG + lymphoproliferation + EBV → PIK3CD-APDS",
            "Step_4": "Functional biomarkers: TREC/KREC newborn assay (T/B naive cells); RBC dATP:ATP ratio (ADA-SCID); anti-IFN-ω antibodies (APECED >95% sensitivity); CTLA4 + LRBA flow cytometry (LRBA/CTLA4-H); PI3K pathway activation markers (APDS)",
            "Step_5": "Targeted therapy clue: ABATACEPT dramatic response within weeks → LRBA or CTLA4-H (both); LENIOLISIB response (PI3Kδ inhibitor) → PIK3CD-APDS; PEG-ADA improvement → ADA-SCID confirmed; IVIG correction of infection frequency → XLA/CVID spectrum",
            "Step_6": "Molecular confirmation: WES or targeted PID panel; XLR males (IL2RG/BTK): hemizygous mutation; AR (ADA/RAG1/AIRE/LRBA): biallelic mutations; AD (CTLA4/PIK3CD): heterozygous (CTLA4 LOF; PIK3CD GOF annotated separately)",
        },
        "type_distribution": dict(sorted(type_counts.items(), key=lambda x: -x[1])[:10]),
    }


def generate_breakdown() -> dict:
    cohort = _build_cohort()
    by_gene = {}
    for p in cohort:
        by_gene.setdefault(p["gene"], []).append(p)

    gene_breakdown = {}
    for entry in IMMUNO_GENES:
        g = entry["gene"]
        pts = by_gene.get(g, [])

        severity_dist = {}
        treatment_dist = {}
        complication_dist = {}
        ig_dist = {}
        for p in pts:
            sv = p["severity"]
            severity_dist[sv] = severity_dist.get(sv, 0) + 1
            tx = p["treatment"]
            treatment_dist[tx] = treatment_dist.get(tx, 0) + 1
            for comp in p["complications"]:
                complication_dist[comp] = complication_dist.get(comp, 0) + 1
            ig = p["immunoglobulin_level_category"]
            ig_dist[ig] = ig_dist.get(ig, 0) + 1

        avg_age = round(sum(p["age_at_dx_yrs"] for p in pts) / len(pts), 2) if pts else 0
        avg_fu = round(sum(p["follow_up_yrs"] for p in pts) / len(pts), 1) if pts else 0
        autoimmune_n = sum(1 for p in pts if p["autoimmune_event"])
        lymphoproliferation_n = sum(1 for p in pts if p["lymphoproliferation_event"])
        enteropathy_n = sum(1 for p in pts if p["enteropathy_event"])

        gene_breakdown[g] = {
            "gene": g,
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "immunodeficiency_category": entry["immunodeficiency_category"],
            "immunophenotype": entry["immunophenotype"],
            "n_patients": len(pts),
            "avg_age_at_dx_yrs": avg_age,
            "avg_follow_up_yrs": avg_fu,
            "autoimmune_event_pct": round(autoimmune_n / len(pts) * 100, 1) if pts else 0,
            "lymphoproliferation_event_pct": round(lymphoproliferation_n / len(pts) * 100, 1) if pts else 0,
            "enteropathy_event_pct": round(enteropathy_n / len(pts) * 100, 1) if pts else 0,
            "pathognomonic": entry["pathognomonic"],
            "treatment_highlight": entry["treatment"][:600],
            "key_features": entry["key_features"],
            "treatment_summary": entry["treatment"][:700],
            "monitoring": entry["monitoring"],
            "key_ddx": entry["key_ddx"],
            "scid_type": entry["scid_type"],
            "agammaglobulinemia": entry["agammaglobulinemia"],
            "autoimmune_risk": entry["autoimmune_risk"],
            "lymphoproliferation_risk": entry["lymphoproliferation_risk"],
            "enteropathy_risk": entry["enteropathy_risk"],
            "severity_distribution": dict(sorted(severity_dist.items(), key=lambda x: -x[1])),
            "complication_distribution": dict(sorted(complication_dist.items(), key=lambda x: -x[1])),
            "immunoglobulin_distribution": dict(sorted(ig_dist.items(), key=lambda x: -x[1])),
            "treatment_distribution": dict(sorted(treatment_dist.items(), key=lambda x: -x[1])),
            "patients_sample": pts[:5],
        }

    return {
        "title": "Hereditary-Primary-Immunodeficiency-Atlas — Per-Gene Breakdown",
        "n_genes": 8,
        "n_patients": len(cohort),
        "gene_breakdown": gene_breakdown,
        "clinical_emergency_flags": [
            "IL2RG/ADA/RAG1 — BCG-OSIS EMERGENCY: any SCID infant with prior BCG vaccination has active disseminated mycobacterial disease; URGENT: anti-mycobacterial triple therapy (isoniazid + rifampicin + ethambutol) + immunology + HSCT/gene-therapy planning; BCG-osis is fatal without both anti-mycobacterial treatment AND immune reconstitution; TREC newborn screening prevents this iatrogenic catastrophe",
            "AIRE — ADRENAL CRISIS: APECED patient with fever/vomiting/hypotension → IV hydrocortisone 100mg STAT (adult) / 2-4mg/kg (paediatric) + IV 0.9% saline + IV dextrose; do NOT await cortisol results; sick-day rule (3x dose during illness) must be rehearsed and medical alert bracelet worn; IM hydrocortisone self-injection kit essential for home emergency",
            "BTK — ENTEROVIRAL ENCEPHALITIS: any XLA patient with new neurological symptoms → CSF enteroviral PCR STAT (ECHO/poliovirus/coxsackie); IVIG high-dose IV + pleconaril compassionate use + intrathecal IVIG; progressive irreversible CNS damage without treatment; ALL household contacts must avoid OPV (attenuated poliovirus shed = risk to XLA patient)",
            "AIRE — HYPOCALCAEMIC TETANY/SEIZURES: APECED patient with perioral tingling/carpopedal spasm/seizure → IV calcium gluconate 10% (10mL over 10min) STAT + infusion; check Mg2+ (hypomagnesaemia impairs PTH secretion); long-term calcitriol + calcium target serum Ca 2.0-2.1 mmol/L; teriparatide for refractory hypoparathyroidism",
            "PIK3CD-APDS — EBV LYMPHOMA SURVEILLANCE: annual EBV DNA PCR mandatory; rising viral load → intensify antiviral (valganciclovir) + urgent PET/CT; lymphadenopathy in APDS is NOT benign — biopsy if persistent; leniolisib (Joenja) reduces lymphoproliferation; HSCT if lymphoma confirmed; up to 20% lifetime lymphoma risk if unmonitored",
        ],
    }


def generate_definitions() -> dict:
    return {
        "title": "Hereditary-Primary-Immunodeficiency-Atlas — Definitions & Glossary",
        "gene_entries": {
            entry["gene"]: {
                "full_name": entry["gene"],
                "protein_size": entry["protein_size"],
                "locus": entry["locus"],
                "inheritance": entry["inheritance"],
                "disease_name": entry["immunodeficiency_category"],
                "pathognomonic": entry["pathognomonic"],
                "key_features": entry["key_features"],
                "treatment": entry["treatment"],
                "key_ddx": entry["key_ddx"],
                "monitoring": entry["monitoring"],
            }
            for entry in IMMUNO_GENES
        },
        "immunology_glossary": {
            "SCID (Severe Combined Immunodeficiency)": (
                "SCID is a life-threatening primary immunodeficiency characterised by absent or profoundly deficient "
                "T-cell function (with variable B cell and NK cell involvement depending on the genetic cause). "
                "Classical SCID presents in the first 6 months of life with recurrent/severe infections (bacterial, "
                "viral, fungal, opportunistic), failure to thrive, and chronic diarrhoea as maternal IgG wanes. "
                "SCID is a paediatric emergency — UNTREATED MEDIAN SURVIVAL <1 YEAR. "
                "Immunophenotypes: T-B+NK- (IL2RG/JAK3); T-B-NK- (ADA); T-B-NK+ (RAG1/RAG2/Artemis/DNA-PKcs); "
                "T-B+NK+ (ZAP70, CD8 deficiency). TREC (T-cell receptor excision circles) newborn screening enables "
                "pre-symptomatic diagnosis — absent TREC = absent naive T cells = SCID until proven otherwise. "
                "TREATMENT: HSCT (curative) or gene therapy (increasingly available, curative in several forms)."
            ),
            "V(D)J recombination": (
                "The somatic DNA recombination process that generates the diverse T-cell receptor (TCR) and "
                "B-cell receptor (BCR) repertoires from germline gene segments. "
                "RAG1-RAG2 heterodimer recognises recombination signal sequences (RSS) flanking V, D, J gene "
                "segments and introduces DNA double-strand breaks (DSBs); NHEJ (non-homologous end joining) "
                "repairs these breaks introducing junctional diversity (N nucleotides). "
                "COMPLETE RAG1/RAG2 DEFICIENCY: no TCR or BCR generated → no T or B cells → SCID. "
                "HYPOMORPHIC RAG1/RAG2: partial V(D)J → limited autoreactive TCR clones → "
                "Omenn syndrome (erythroderma + eosinophilia + elevated IgE). "
                "TREC (T-cell receptor excision circles) are byproducts of V(D)J recombination → "
                "TREC assay in newborn screen measures T-cell neogenesis."
            ),
            "Omenn Syndrome": (
                "A specific clinical phenotype caused by hypomorphic (partial loss-of-function) mutations in RAG1, "
                "RAG2, Artemis, or other V(D)J recombination genes. Partial V(D)J activity generates a severely "
                "restricted, oligoclonal, autoreactive T-cell repertoire. These autoreactive T-cells expand and "
                "infiltrate skin, gut, and lymphoid organs. "
                "PATHOGNOMONIC COMBINATION: ERYTHRODERMA (universal red scaly rash — T-cell skin infiltration) + "
                "VERY ELEVATED IgE (paradox: oligoclonal Th2-skewed T cells drive IgE) + EOSINOPHILIA + "
                "ABSENT NORMAL IMMUNOGLOBULINS (despite elevated IgE — IgE is oligoclonal, not protective) + "
                "HEPATOSPLENOMEGALY + ALOPECIA. "
                "DISTINGUISHED FROM SCID: Omenn has T cells present but oligoclonal/autoreactive; "
                "from maternal GvHD: PCR chimerism differentiates (maternal vs endogenous T cells)."
            ),
            "T-regulatory cells (Tregs)": (
                "CD4+FoxP3+CD25+ T-regulatory cells are a specialised T-cell subset that suppress excessive immune "
                "activation and maintain self-tolerance in the periphery. "
                "Tregs constitutively express CTLA4 (at high levels) — the checkpoint receptor that competes with "
                "CD28 for B7 (CD80/CD86) ligands on APCs → suppresses effector T-cell activation. "
                "CTLA4 surface expression on Tregs is maintained by LRBA-mediated endosomal recycling "
                "(LRBA rescues CTLA4 from lysosomal degradation). "
                "LRBA deficiency OR CTLA4 haploinsufficiency → reduced surface CTLA4 on Tregs → "
                "Treg dysfunction → effector T-cell dysregulation → multi-organ autoimmunity. "
                "CTLA4 FLOW CYTOMETRY on CD4+FoxP3+ T cells = functional screening for LRBA/CTLA4-H."
            ),
            "CTLA4 checkpoint": (
                "CTLA4 (CD152) is an inhibitory co-receptor expressed on activated T cells and constitutively on "
                "T-regulatory cells. It outcompetes CD28 for shared B7 ligands (CD80/CD86) on APCs with 10-100x "
                "higher affinity → reduces CD28 co-stimulation → downregulates T-cell activation. "
                "CTLA4 HAPLOINSUFFICIENCY (heterozygous LOF, AD): one functioning CTLA4 allele insufficient → "
                "reduced surface CTLA4 → unchecked T-cell activation → multi-organ autoimmunity + "
                "lymphocytic infiltration (lungs = GLILD, gut = enteropathy, CNS). "
                "ABATACEPT (CTLA4-Ig): recombinant fusion protein (CTLA4 ectodomain + IgG Fc); "
                "provides exogenous soluble CTLA4 → restores B7 blockade → suppresses T-cell dysregulation. "
                "Dramatic response to abatacept is itself a diagnostic signal for CTLA4-H or LRBA deficiency."
            ),
            "PI3Kd pathway (PIK3CD)": (
                "PI3Kδ (phosphatidylinositol-4,5-bisphosphate 3-kinase, catalytic subunit delta) is a lipid kinase "
                "expressed predominantly in lymphocytes. It converts PIP2 to PIP3 → recruits AKT to plasma membrane "
                "→ activates AKT-mTOR-S6K pathway → promotes lymphocyte survival and differentiation. "
                "Normally activated transiently by BCR/TCR signalling. "
                "PIK3CD GAIN-OF-FUNCTION (AD-GOF): constitutive PI3K-AKT-mTOR activation → "
                "terminal lymphocyte differentiation failure (B cells fail to class-switch; "
                "T cells accumulate as senescent effector memory; NK cells dysfunctional) → "
                "hypogammaglobulinemia + recurrent infections + lymphoproliferation + EBV/CMV susceptibility. "
                "LENIOLISIB (Joenja): selective PI3Kδ inhibitor; FDA approved 2023; reduces PIP3 production → "
                "restores normal lymphocyte differentiation and reduces lymphoproliferation."
            ),
            "TREC/KREC newborn screening": (
                "T-cell Receptor Excision Circles (TRECs) and Kappa-deleting Recombination Excision Circles (KRECs) "
                "are circular DNA byproducts generated during V(D)J recombination in the thymus (T cells) and "
                "bone marrow (B cells) respectively. "
                "TRECs are generated with each new naive T cell produced in the thymus (signal joint TREC = sjTREC). "
                "Absent TREC = absent thymic output = absent naive T cells = SCID (IL2RG, ADA, RAG1) or "
                "other T-cell deficiency. "
                "TREC newborn screening: dried blood spot (Guthrie card) PCR quantifies sjTREC copies. "
                "Low TREC: <25 TRECs/µL (varies by lab) = refer for immunophenotyping. "
                "KREC: byproduct of BCR Igκ rearrangement → measures B-cell neogenesis; low KREC = XLA (BTK). "
                "Combined TREC+KREC detects SCID (T) + XLA/agammaglobulinemia (B) on single newborn screen."
            ),
        },
        "treatment_glossary": {
            "IVIG (Intravenous Immunoglobulin)": (
                "Polyclonal IgG pooled from thousands of healthy donors; "
                "indication: hypogammaglobulinemia in XLA, CVID, LRBA, CTLA4-H, SCID pre-HSCT; "
                "dosing: 400-600 mg/kg every 3-4 weeks IV; trough IgG target: >600-800 mg/dL (>1000 mg/dL if "
                "recurrent infections despite adequate trough); "
                "SC-Ig (subcutaneous immunoglobulin): self-administered at home weekly — preferred quality-of-life; "
                "mechanism: passive antibody replacement (opsonins, neutralising antibodies against encapsulated bacteria); "
                "does NOT restore T-cell or NK-cell function — supplemental only; "
                "MONITORING: trough IgG 3-monthly; infection frequency; pulmonary function annually (bronchiectasis)."
            ),
            "HSCT for SCID": (
                "Haematopoietic stem cell transplantation (HSCT) is the only established curative treatment for most "
                "SCID variants (IL2RG, ADA, RAG1/2) when gene therapy is unavailable or unsuitable. "
                "BEST OUTCOMES: matched sibling donor (MSD) pre-infection; survival >90% modern series; "
                "T-cell depleted MUD (matched unrelated donor): 70-85% survival; "
                "haplo-identical (parental) T-depleted: 60-75% (improving with post-transplant cyclophosphamide); "
                "CONDITIONING: SCID may not require myeloablative conditioning (MHC-matched thymic tolerance); "
                "ADA-SCID: non-myeloablative preferred (toxicity reduction); "
                "EARLY TRANSPLANT (before 3 months, pre-infection): best predictor of outcome in ALL SCID types; "
                "TREC newborn screening enables this pre-symptomatic HSCT window."
            ),
            "PEG-ADA (Enzyme Replacement Therapy for ADA-SCID)": (
                "Pegylated bovine adenosine deaminase (PEG-ADA; ADAGEN/REVCOVI); "
                "mechanism: exogenous ADA clears toxic deoxyadenosine/dATP from RBCs → reduces lymphocyte apoptosis; "
                "route: intramuscular injection weekly; "
                "bridge therapy — NOT curative; partial immune reconstitution; reduces infection frequency; "
                "lymphocyte counts improve but rarely to fully normal; "
                "monitoring: RBC dATP:ATP ratio + plasma ADA activity weekly → monthly; "
                "TACHYPHYLAXIS: anti-PEG-ADA IgG antibodies develop over time → reduced efficacy; "
                "REVCOVI (pegadricase) — newer formulation; improved enzyme stability; "
                "REPLACED BY GENE THERAPY where available (Strimvelis or lentiviral ADA) — curative."
            ),
            "Gene therapy for SCID": (
                "Ex-vivo autologous haematopoietic stem cell gene therapy: patient CD34+ HSCs collected, "
                "corrected with viral vector carrying functional gene, reinfused. "
                "IL2RG: OTL-101 / CARTEYVA (lentiviral vector) — FDA approved; curative; avoids insertional "
                "oncogenesis of earlier retroviral vectors; complete immune reconstitution in most; "
                "ADA-SCID: Strimvelis (GSK; retroviral ADA) — European EMA approved; complete immune reconstitution; "
                "lentiviral ADA (EFS-ADA): improved safety profile; clinical trials ongoing; "
                "RAG1/RAG2: lentiviral vectors in clinical development; NOT yet FDA-approved; "
                "ADVANTAGES over HSCT: no matched donor required; no GvHD risk; no graft failure; "
                "avoids conditioning-related toxicity in fragile SCID infants; "
                "MONITORING: gene marking (PCR); T-cell TREC; immune reconstitution monthly × 24 months."
            ),
            "Abatacept/CTLA4-Ig": (
                "Abatacept (Orencia) is a recombinant fusion protein: CTLA4 ectodomain + human IgG1 Fc region. "
                "Mechanism: soluble CTLA4 competes with CD28 for B7 (CD80/CD86) → blocks T-cell co-stimulation → "
                "suppresses effector T-cell dysregulation. "
                "LRBA deficiency and CTLA4 haploinsufficiency: reduced endogenous CTLA4 surface expression → "
                "abatacept provides exogenous functional CTLA4 replacement → reversal of autoimmunity within weeks. "
                "DOSING: IV abatacept 10mg/kg monthly (weight-based; <75kg/75-100kg/>100kg fixed); "
                "or SC abatacept 125mg weekly; "
                "RESPONSE: dramatic improvement in autoimmune cytopenias + enteropathy + pulmonary disease "
                "within 4-8 weeks = itself a diagnostic signal for CTLA4/LRBA; "
                "COMBINE WITH IVIG if hypogammaglobulinemia component; "
                "LONG-TERM: safety profile well established (rheumatoid arthritis experience); infections risk low."
            ),
            "Leniolisib (Joenja)": (
                "Leniolisib (Joenja; Pharming Group) is a selective, oral, small-molecule PI3Kδ inhibitor. "
                "FDA approved March 2023 for APDS/APDS1 (PIK3CD GOF) — first targeted therapy for APDS. "
                "Mechanism: blocks constitutive PI3K-AKT-mTOR signalling in lymphocytes → "
                "reduces senescent T-cell accumulation → improves B-cell class-switching → "
                "reduces lymphoproliferation and increases naive lymphocyte populations. "
                "DOSING: 70mg oral twice daily (adults); paediatric dosing trials ongoing; "
                "CLINICAL RESPONSE: reduced splenomegaly/lymphadenopathy + improved IgG + reduced infection frequency; "
                "MONITORING: LFTs monthly × 3 months (hepatotoxicity); then quarterly; "
                "CBC: watch for paradoxical cytopenia (rare); EBV PCR: leniolisib reduces but does not eliminate "
                "lymphoma risk — annual EBV monitoring continues; "
                "ALSO FOR APDS2 (PIK3R1 GOF): same pathway, same drug, same efficacy."
            ),
        },
        "diagnostic_tests": {
            "TREC assay (Newborn T-cell screening)": (
                "T-cell Receptor Excision Circles (TREC) assay on dried blood spot (Guthrie card) by real-time PCR; "
                "measures sjTREC (signal-joint TREC) — byproduct of TCR-alpha V(J) rearrangement in thymus; "
                "LOW TREC (<25 copies/µL, varies by lab): refer URGENTLY for lymphocyte phenotyping and SCID workup; "
                "ABSENT TREC = absent thymic T-cell output = SCID until excluded; "
                "causes: SCID (IL2RG/ADA/RAG1/RAG2/Artemis/others); DiGeorge (22q11.2); prematurity (thymic immaturity); "
                "COMBINED TREC+KREC: also detects XLA (absent KREC = absent B-cell neogenesis); "
                "TREC screening programs: standard in USA (all 50 states), UK, Europe, Canada — enables pre-BCG diagnosis of SCID."
            ),
            "Lymphocyte phenotyping by flow cytometry": (
                "Multi-parameter flow cytometric immunophenotyping: absolute counts and percentages of: "
                "T cells (CD3, CD4, CD8); B cells (CD19, CD20, IgM/IgD/IgG subclasses — switched memory); "
                "NK cells (CD16+CD56+); T-regulatory cells (CD4+CD25+FoxP3+); "
                "CTLA4 surface expression (CD4+FoxP3+ T-regs — reduced in LRBA/CTLA4-H); "
                "LRBA intracellular protein (LRBA-specific antibody — absent in LRBA deficiency); "
                "APDS/PIK3CD: p-AKT/p-S6K1 phospho-flow (PI3K pathway activation markers); "
                "BTK expression on monocytes (absent in XLA — BTK protein flow diagnostic); "
                "SCID immunophenotype panel: T-B+NK- (IL2RG); T-B-NK- (ADA); T-B-NK+ (RAG1); T+B-NK+ (BTK)."
            ),
            "V(D)J recombination PCR/NGS": (
                "Molecular assessment of TCR and BCR repertoire diversity: "
                "TREC + KREC quantification by PCR (new T and B cell neogenesis); "
                "TCR spectratyping (CDR3 length distribution): normal = Gaussian; SCID/Omenn = oligoclonal peaks; "
                "BCR NGS (immunoglobulin heavy chain sequencing): class-switched memory B cells (absent in APDS/CVID); "
                "V(D)J gene segment usage NGS: detects restricted clonal expansion in Omenn vs polyclonal reconstitution; "
                "RAG1/RAG2 molecular diagnosis: WES or Sanger sequencing of both genes (head-to-head on 11p13); "
                "Interpretation: biallelic null = complete SCID; hypomorphic = Omenn/leaky SCID; "
                "V(D)J gene therapy vector integration site analysis (IS-PCR): safety monitoring post-gene-therapy."
            ),
            "Anti-IFN-omega antibodies (AIRE/APECED)": (
                "Anti-interferon-omega (anti-IFN-ω) autoantibodies are PATHOGNOMONIC biomarkers for APECED/APS-1 "
                "(AIRE deficiency). Detected in >95% of APECED patients (sensitivity and specificity). "
                "Mechanism: AIRE deficiency → autoreactive T cells escape thymus → target IFN-ω producing cells → "
                "anti-IFN-ω antibodies generated (detectable years before clinical APECED manifestations). "
                "TEST: ELISA or luciferase immunoprecipitation system (LIPS) assay for anti-IFN-ω and anti-IFN-α2; "
                "CLINICAL USE: diagnostic when APECED suspected (chronic candidiasis + endocrinopathy); "
                "also useful in: COVID-19 severity prediction (anti-IFN-ω same antibody in COVID severe disease); "
                "ADDITIONAL APECED ANTIBODIES: anti-21-hydroxylase (adrenal), anti-IL-22 (candidiasis marker), "
                "anti-IL-17F, anti-parathyroid, anti-insulin, anti-GAD, anti-TPO — screen annually."
            ),
        },
    }


# Aliases for api_backend.py compatibility
def overview() -> dict:
    return generate_overview()


def breakdown() -> dict:
    return generate_breakdown()


def definitions() -> dict:
    return generate_definitions()


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(f"  Title: {ov['title']}")
    print(f"  Patients: {ov['n_patients']}  |  Genes: {ov['n_genes']}")
    print(f"  Seeds: {ov['seed_range']}")
    print("  Key pearls:")
    for p in ov["key_clinical_pearls"][:4]:
        print(f"    - {p[:120]}")
    print("\n=== BREAKDOWN (gene counts) ===")
    bk = generate_breakdown()
    for g, info in bk["gene_breakdown"].items():
        print(f"  {g}: {info['n_patients']} pts | Type: {info['immunodeficiency_category'][:70]}")
    print("\n=== DEFINITIONS (gene count) ===")
    df = generate_definitions()
    print(f"  Genes defined: {len(df['gene_entries'])}")
    print(f"  Immunology glossary entries: {len(df['immunology_glossary'])}")
    print(f"  Treatment glossary entries: {len(df['treatment_glossary'])}")
    print(f"  Diagnostic test entries: {len(df['diagnostic_tests'])}")
