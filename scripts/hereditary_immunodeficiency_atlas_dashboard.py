#!/usr/bin/env python3
"""Hereditary-Immunodeficiency-Atlas — Complete 8-Gene Primary Immunodeficiency Atlas
BTK     (Bruton tyrosine kinase; 638 aa; Xq22.1; XLR;
         X-linked Agammaglobulinemia (XLA); OMIM gene 300300; disease OMIM 300755;
         LOF -> no mature B cells -> absent all immunoglobulin classes;
         profound recurrent bacterial infections; live vaccines ABSOLUTELY CI;
         IVIG lifelong; no T-cell defect; seed SEED_BASE+0) .
ADA     (adenosine deaminase; 363 aa; 20q13.12; AR;
         ADA-SCID / SCID1; OMIM gene 608958; disease OMIM 102700;
         LOF -> dATP accumulates -> toxic to all lymphocytes -> T-B-NK absent;
         gene therapy (Strimvelis/PEG-ADA); HLA-identical HSCT curative; seed SEED_BASE+1) .
IL2RG   (common gamma chain gamma-c; 369 aa; Xq13.1; XLR;
         XSCID / SCIDX1; OMIM gene 308380; disease OMIM 300400;
         shared gamma-c for IL-2/-4/-7/-9/-15/-21 receptors;
         LOF -> T-B+NK- phenotype; gene therapy OTL-101 FDA2024;
         live vaccines ABSOLUTELY CI; seed SEED_BASE+2) .
RAG1    (recombination activating gene 1; 1043 aa; 11p13; AR;
         Omenn Syndrome / RAG1-SCID; OMIM gene 179615; disease OMIM 601457/267500;
         V(D)J recombination enzyme; hypomorphic -> Omenn; complete LOF -> SCID; seed SEED_BASE+3) .
WAS     (Wiskott-Aldrich Syndrome Protein; 502 aa; Xp11.23; XLR;
         Wiskott-Aldrich Syndrome; OMIM gene 300392; disease OMIM 301000;
         eczema + thrombocytopenia + immunodeficiency TRIAD PATHOGNOMONIC;
         small platelets PATHOGNOMONIC; splenectomy CI; HSCT curative; seed SEED_BASE+4) .
DOCK8   (dedicator of cytokinesis 8; 2099 aa; 9p24.3; AR;
         DOCK8 deficiency / HIES type 2; OMIM gene 611432; disease OMIM 243700;
         severe eczema + cutaneous viral infections PATHOGNOMONIC + elevated IgE;
         STAT3-HIES TYPE 1 is AD form -- KEY DDx; HSCT curative; seed SEED_BASE+5) .
TNFRSF13B (TACI; 293 aa; 17p11.2; AD/AR;
         CVID2; OMIM gene 604907; disease OMIM 240500;
         LOF -> hypogammaglobulinaemia IgG + IgA; IVIG/SCIG lifelong;
         granulomatous disease 10-20%; increased lymphoma risk; seed SEED_BASE+6) .
LRBA    (LPS-responsive beige-like anchor protein; 2863 aa; 4q31.3; AR;
         CVID8 / LRBA deficiency; OMIM gene 606453; disease OMIM 614700;
         regulates CTLA-4 recycling; LOF -> immune dysregulation;
         Abatacept DRAMATICALLY effective -- PATHOGNOMONIC TREATMENT RESPONSE; seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 1782-1789)
"""

import random

SEED_BASE = 1782

IMID_GENES = [
    # -- BTK -- X-linked Agammaglobulinemia (XLA) ----------------------------
    {
        "gene": "BTK",
        "protein": (
            "BTK -- Xq22.1 XLR -- Bruton-tyrosine-kinase-638aa -- "
            "X-linked-Agammaglobulinemia-XLA -- "
            "LOF->No-Mature-B-Cells->Absent-ALL-Immunoglobulin-Classes -- "
            "Recurrent-Encapsulated-Bacterial-Infections -- "
            "Live-Vaccines-ABSOLUTELY-CONTRAINDICATED -- "
            "IVIG-Lifelong -- No-T-Cell-Defect"
        ),
        "alias": (
            "BTK (Bruton tyrosine kinase); OMIM gene 300300; "
            "X-linked Agammaglobulinemia (XLA; Bruton disease) OMIM 300755. "
            "Xq22.1; 638 aa; ~76 kDa; X-linked recessive -- affects males, females are carriers. "
            "FUNCTION: BTK is a non-receptor tyrosine kinase of the Tec kinase family, "
            "essential for B-cell development and activation. "
            "BTK mediates signalling downstream of the pre-B-cell receptor (pre-BCR) and mature BCR: "
            "B-cell progenitor -> pre-B cell (requires pre-BCR signalling via BTK) -> "
            "BTK LOF -> signal fails -> B-cell maturation arrests at the pro-B to pre-B transition -> "
            "NO mature B cells in peripheral blood -> "
            "NO immunoglobulin production of any class (IgG, IgA, IgM, IgE, IgD all absent). "
            "BTK is also expressed in monocytes, macrophages, and platelets (but not T cells or NK cells). "
            "CLINICAL PHENOTYPE: "
            "Boys only (XLR); mothers are obligate carriers (usually asymptomatic). "
            "Recurrent sinopulmonary bacterial infections: Streptococcus pneumoniae, Haemophilus influenzae, "
            "Staphylococcus aureus, Pseudomonas aeruginosa. "
            "Onset: typically 6-18 months (maternal antibody wanes by 3-6 months -- protects early). "
            "PATHOGNOMONIC INFECTIONS: encapsulated bacteria (polysaccharide-capsulated) -- "
            "these require opsonisation by specific antibody for clearance; "
            "without antibodies -> repeated pneumonias, sinusitis, otitis media, meningitis, septicaemia. "
            "Giardia lamblia: intestinal infection common in XLA (IgA in gut normally clears Giardia). "
            "Enteroviral encephalitis: CRITICAL complication -- enteroviruses (echovirus, poliovirus) "
            "normally cleared by antibody; XLA patients can develop fatal chronic enteroviral "
            "meningoencephalitis; live polio vaccine ABSOLUTELY CONTRAINDICATED (OPV). "
            "ABSENT T-CELL DEFECT: BTK not expressed in T cells -- T-cell numbers and function NORMAL; "
            "no opportunistic infections (Pneumocystis, CMV, fungi) -- "
            "this distinguishes XLA from SCID (T-cell defects). "
            "LABORATORY DIAGNOSIS: "
            "Absent/very low serum IgG, IgA, IgM, IgE (<0.01 g/L typical); "
            "CD19+ B cells absent or <1% of lymphocytes (flow cytometry DIAGNOSTIC); "
            "T cells normal; NK cells normal; "
            "BTK protein expression absent on monocytes (flow cytometry for BTK protein); "
            "BTK gene sequencing confirms. "
            "TREATMENT: "
            "IVIG (intravenous immunoglobulin) or SCIG (subcutaneous): lifelong replacement; "
            "Target trough IgG >8 g/L (or higher with chronic lung disease); "
            "Dose: 400-600 mg/kg IV every 3-4 weeks or SCIG equivalent; "
            "Antibiotic prophylaxis during infections; "
            "LIVE VACCINES ABSOLUTELY CONTRAINDICATED: OPV, MMR, varicella, yellow fever, rotavirus -- "
            "live attenuated organisms can cause disease in immunodeficient patients; "
            "BCG: absolutely CI (disseminated BCG reported); "
            "HSCT: not routinely indicated (IVIG management effective); rare cases with malignancy. "
            "PROGNOSIS: with adequate IVIG replacement, normal lifespan is achievable; "
            "chronic lung disease (bronchiectasis) from recurrent pneumonias is the main long-term complication."
        ),
        "locus": "Xq22.1",
        "aa": 638,
        "kDa": 76,
        "omim_gene": "300300",
        "omim_disease": "300755",
        "inheritance": "XLR -- affects males; females are carriers",
        "gene_class": "Non-receptor Tec-family tyrosine kinase -- B-cell pre-BCR/BCR signalling",
        "key_alerts": [
            "BTK-LIVE-VACCINES-ABSOLUTELY-CI: ALL live attenuated vaccines are absolutely contraindicated in XLA -- oral polio vaccine (OPV) can cause vaccine-derived poliomyelitis; MMR, varicella, yellow fever, rotavirus, BCG all CI; family contacts should use inactivated polio vaccine (IPV) only",
            "BTK-ENTEROVIRAL-ENCEPHALITIS: Chronic enteroviral meningoencephalitis is a life-threatening complication of XLA -- echovirus and other enteroviruses cause progressive encephalitis; no effective treatment; high-dose IVIG may slow progression; polio immunisation of contacts is critical",
            "BTK-B-CELLS-ABSENT-T-CELLS-NORMAL: XLA is a PURE B-cell/antibody defect -- no T-cell or NK-cell defect; no risk of Pneumocystis, fungal, or viral (CMV, EBV) opportunistic infections; this distinguishes XLA from SCID; absence of CD19+ B cells on flow cytometry is the key laboratory marker",
            "BTK-IVIG-LIFELONG-TROUGH-TARGET: IVIG or SCIG replacement is lifelong and must achieve trough IgG >8 g/L (higher if bronchiectasis present); inadequate replacement -> recurrent sinopulmonary infections -> bronchiectasis -> progressive respiratory failure; trough IgG must be checked before every infusion",
            "BTK-GIARDIA-ANTIBIOTICS: Giardia lamblia causes chronic diarrhoea in XLA (normally cleared by secretory IgA in the gut); metronidazole or tinidazole treatment; recurrences common; consider longer courses",
            "BTK-MATERNAL-PROTECTION-6-18M: Maternal IgG is passively transferred transplacentally and protects XLA infants until 3-6 months of age; diagnosis typically delayed until 6-18 months when maternal antibody wanes and infections begin",
        ],
        "etiologies": [
            {"variant": "p.Arg525Gln (c.1574G>A)", "type": "missense LOF -- kinase domain", "frequency": "common", "severity": "severe XLA"},
            {"variant": "p.Leu511Pro (c.1532T>C)", "type": "missense LOF -- kinase domain", "frequency": "moderate", "severity": "severe"},
            {"variant": "Exon deletion/frameshift", "type": "LOF truncating", "frequency": "~30% of XLA", "severity": "severe"},
            {"variant": "Splice site mutations", "type": "LOF splice", "frequency": "~15% of XLA", "severity": "variable"},
            {"variant": "p.Cys154Arg (c.460T>C)", "type": "missense LOF -- SH2 domain", "frequency": "moderate", "severity": "severe"},
        ],
        "stats": {
            "incidence": "~1:190,000 male births",
            "igg_at_diagnosis": "<0.1 g/L typical",
            "cd19_b_cells": "<1% of lymphocytes",
            "bronchiectasis_risk": "~50% by adult age without good IgG control",
            "ivig_trough_target": ">8 g/L (>10 g/L with lung disease)",
        },
        "dx_delay_distribution": {
            "infant_6_12m": 30,
            "infant_12_24m": 45,
            "child_2_5y": 20,
            "late_5y_plus": 5,
        },
    },

    # -- ADA -- ADA-SCID (Adenosine Deaminase Deficiency) --------------------
    {
        "gene": "ADA",
        "protein": (
            "ADA -- 20q13.12 AR -- Adenosine-deaminase-363aa -- "
            "ADA-SCID-SCID1 -- "
            "dATP-Accumulates->Lymphocyte-Toxicity->T-B-NK-All-Absent -- "
            "Gene-Therapy-Strimvelis-EMA-Approved-OTL-101 -- "
            "PEG-ADA-Bridge-Therapy -- "
            "HLA-Identical-HSCT-Curative"
        ),
        "alias": (
            "ADA (adenosine deaminase); OMIM gene 608958; "
            "ADA-SCID (Severe Combined Immunodeficiency type 1) OMIM 102700. "
            "20q13.12; 363 aa; ~41 kDa; autosomal recessive -- biallelic LOF. "
            "FUNCTION: ADA is a ubiquitous enzyme of the purine salvage pathway: "
            "ADA catalyses the irreversible deamination of adenosine -> inosine and "
            "deoxyadenosine -> deoxyinosine. "
            "MECHANISM OF IMMUNODEFICIENCY: "
            "ADA LOF -> deoxyadenosine (dAdo) accumulates -> "
            "dAdo phosphorylated by deoxycytidine kinase -> dATP accumulates intracellularly; "
            "dATP is selectively toxic to lymphocytes (especially T cells) because: "
            "(1) dATP inhibits ribonucleotide reductase -> blocks DNA synthesis; "
            "(2) dATP triggers apoptosis via mitochondrial pathway; "
            "(3) lymphocytes have high deoxycytidine kinase and low 5-nucleotidase activity; "
            "result: T, B, AND NK cells ALL absent (T-B-NK- SCID -- pan-lymphopenia). "
            "ADA-SCID accounts for ~15% of all SCID cases. "
            "CLINICAL PHENOTYPE: "
            "Recurrent opportunistic and non-opportunistic infections from birth; "
            "Pneumocystis jirovecii pneumonia (PJP/PCP) -- hallmark of T-cell deficiency; "
            "CMV, EBV, adenovirus, fungal infections; "
            "Failure to thrive; "
            "Thymic shadow absent on chest X-ray; "
            "Skeletal dysplasia (costochondral junctions abnormal on X-ray) -- "
            "ADA also expressed in osteoblasts -> bony abnormalities unique to ADA-SCID; "
            "Neurological features: behavioural issues, deafness reported -- ADA expressed in nervous system; "
            "NBS (newborn screening): T-cell receptor excision circles (TRECs) -- absent in all SCID. "
            "DIAGNOSIS: "
            "ADA enzyme activity: measured in erythrocytes (ADA activity <1% of normal); "
            "dATP elevated in erythrocytes (metabolic marker); "
            "lymphopenia: absolute lymphocyte count <1000/uL (often <500/uL); "
            "T, B, NK cells all absent; ADA gene sequencing confirms. "
            "TREATMENT: "
            "HSCT (HLA-identical sibling or MUD): curative if available; "
            "Gene therapy: "
            "Strimvelis (EMA-approved 2016): ex vivo autologous HSC gamma-retroviral vector gene therapy "
            "for ADA-SCID; available at San Raffaele Hospital Milan; long-term immune reconstitution; "
            "OTL-101 (Orchard Therapeutics, lentiviral): FDA/EMA regulatory review pathway; "
            "PEG-ADA (pegylated bovine ADA; Adagen): enzyme replacement therapy; "
            "weekly SC injections; provides ADA activity; bridge to HSCT or gene therapy; "
            "does NOT fully reconstitute immunity but prevents metabolite accumulation; "
            "prophylaxis: TMP-SMX for PJP; antifungal; CMV monitoring."
        ),
        "locus": "20q13.12",
        "aa": 363,
        "kDa": 41,
        "omim_gene": "608958",
        "omim_disease": "102700",
        "inheritance": "AR -- biallelic loss-of-function",
        "gene_class": "Purine salvage pathway enzyme -- adenosine/deoxyadenosine deaminase",
        "key_alerts": [
            "ADA-SCID-T-B-NK-ALL-ABSENT: ADA-SCID is a pan-lymphopenic SCID -- T, B, AND NK cells are all absent; this distinguishes it from XLA (only B cells absent) and XSCID/IL2RG (T-NK absent, B present); dATP toxicity affects ALL lymphocyte lineages",
            "ADA-GENE-THERAPY-APPROVED: Strimvelis (EMA 2016) is the first approved gene therapy for ADA-SCID; ex vivo autologous HSC treatment with gamma-retroviral vector; available at specialist centres; OTL-101 (lentiviral) in regulatory pipeline; early referral to gene therapy centre is mandatory",
            "ADA-PEG-ADA-BRIDGE: PEG-ADA (Adagen) enzyme replacement is the bridge to HSCT or gene therapy; weekly SC injection restores ADA activity; does NOT fully reconstitute immunity; do not withhold prophylactic antibiotics (TMP-SMX) while on PEG-ADA",
            "ADA-SKELETAL-DYSPLASIA-CLUE: Costochondral junction abnormalities on chest X-ray are unique to ADA-SCID among all SCID causes -- a radiological clue that should prompt ADA enzyme assay",
            "ADA-NBS-TREC: ADA-SCID is detected by newborn screening (TREC assay) -- TRECs absent due to absent T cells; early pre-symptomatic diagnosis dramatically improves outcomes from gene therapy and HSCT",
        ],
        "etiologies": [
            {"variant": "p.Arg156Cys (c.466C>T)", "type": "missense LOF", "frequency": "common", "severity": "severe ADA-SCID"},
            {"variant": "p.Gln3Ter (c.7C>T)", "type": "nonsense LOF", "frequency": "moderate", "severity": "severe"},
            {"variant": "p.Asp8Asn (c.22G>A)", "type": "missense partial", "frequency": "late-onset phenotype", "severity": "mild/partial ADA deficiency"},
            {"variant": "Exon deletions (various)", "type": "LOF deletion", "frequency": "~20% of ADA-SCID", "severity": "severe"},
            {"variant": "p.Gly216Arg (c.646G>A)", "type": "missense LOF", "frequency": "moderate", "severity": "severe"},
        ],
        "stats": {
            "proportion_scid": "~15% of all SCID",
            "ada_enzyme_activity": "<1% of normal at diagnosis",
            "lymphocyte_count": "<500/uL typical",
            "gene_therapy_success": ">90% immune reconstitution with Strimvelis",
            "hsct_matched_sibling": "~85% survival",
        },
        "dx_delay_distribution": {
            "nbs_detected_0_1m": 25,
            "infant_1_3m": 45,
            "infant_3_6m": 20,
            "late_6m_plus": 10,
        },
    },

    # -- IL2RG -- X-linked SCID (XSCID) --------------------------------------
    {
        "gene": "IL2RG",
        "protein": (
            "IL2RG -- Xq13.1 XLR -- Common-gamma-chain-gammac-369aa -- "
            "X-linked-SCID-XSCID-SCIDX1 -- "
            "Shared-gammac-for-IL-2-IL-4-IL-7-IL-9-IL-15-IL-21-Receptors -- "
            "LOF->T-B+NK--Phenotype -- "
            "Gene-Therapy-OTL-101-FDA2024 -- "
            "HSCT-Curative -- Live-Vaccines-ABSOLUTELY-CI"
        ),
        "alias": (
            "IL2RG (interleukin-2 receptor subunit gamma; common gamma chain; gamma-c); "
            "OMIM gene 308380; X-linked SCID (XSCID; SCIDX1) OMIM 300400. "
            "Xq13.1; 369 aa; ~42 kDa; X-linked recessive -- affects males; females carriers. "
            "FUNCTION: IL2RG encodes the common gamma chain (gamma-c), a shared signalling subunit used by "
            "the receptors for IL-2, IL-4, IL-7, IL-9, IL-15, and IL-21. "
            "IL-7 receptor (IL-7Ra + gamma-c): essential for T-cell and NK-cell development in the thymus; "
            "IL-15 receptor (IL-15Ra + IL-2Rb + gamma-c): essential for NK-cell development and homeostasis; "
            "IL-2 receptor (IL-2Ra + IL-2Rb + gamma-c): T-cell proliferation and survival. "
            "LOF -> signalling failure through all gamma-c-dependent cytokine receptors -> "
            "T-cell development fails (IL-7 signalling absent) -> "
            "NK-cell development fails (IL-15 signalling absent) -> "
            "B cells are present (B-cell development is not gamma-c-dependent) but FUNCTIONLESS "
            "(no T-cell help -> no antibody production) -> "
            "XSCID IMMUNOPHENOTYPE: T-B+NK- (T cells absent, B cells present but non-functional, NK cells absent). "
            "MOST COMMON FORM OF SCID: XSCID accounts for ~45-50% of all SCID. "
            "CLINICAL PHENOTYPE: "
            "Profound susceptibility to ALL pathogens from birth (no T or NK cells). "
            "Recurrent respiratory infections (RSV, parainfluenza, adenovirus); "
            "PJP (Pneumocystis jirovecii pneumonia); Mucocutaneous candidiasis; "
            "CMV, EBV, adenovirus viral infections; Failure to thrive, chronic diarrhoea; "
            "GvHD from MATERNAL lymphocytes crossing the placenta -- XSCID infants cannot reject non-self lymphocytes; "
            "GvHD from blood transfusions: ALL blood products must be irradiated and CMV-negative. "
            "TREATMENT: "
            "HSCT: HLA-identical sibling: ~95% survival; MUD: ~70-80% survival; "
            "best outcomes when performed before 3 months of age (before infections); "
            "Gene therapy: OTL-101 (Lentigen, ex vivo lentiviral, autologous HSC) -- FDA approved 2024; "
            "earlier generation retroviral vectors associated with insertional oncogenesis (T-cell lymphoma); "
            "lentiviral vectors have improved safety profile; "
            "Blood products: MUST be irradiated + CMV-negative + leucodepleted -- "
            "non-irradiated blood -> transfusion-associated GvHD -> fatal; "
            "Live vaccines ABSOLUTELY CI: no immune system to contain live organisms."
        ),
        "locus": "Xq13.1",
        "aa": 369,
        "kDa": 42,
        "omim_gene": "308380",
        "omim_disease": "300400",
        "inheritance": "XLR -- affects males; females are carriers",
        "gene_class": "Cytokine receptor common gamma chain -- IL-2/-4/-7/-9/-15/-21 signalling subunit",
        "key_alerts": [
            "IL2RG-LIVE-VACCINES-ABSOLUTELY-CI: Live vaccines are absolutely contraindicated in XSCID -- BCG given before diagnosis can cause disseminated BCGosis (fatal); OPV causes vaccine-derived poliovirus infection; MMR, varicella all CI; any live vaccine in the NICU period must be withheld until SCID excluded",
            "IL2RG-IRRADIATED-BLOOD-MANDATORY: ALL blood transfusions in XSCID/SCID must be irradiated (and CMV-negative, leucodepleted) -- non-irradiated blood contains donor lymphocytes -> transfusion-associated GvHD -> fatal in an immune-deficient host; this is a critical emergency order",
            "IL2RG-MATERNAL-LYMPHOCYTE-GVHD: Maternal lymphocytes cross the placenta during pregnancy; XSCID infants cannot reject them -> maternal engraftment -> neonatal GvHD (rash, liver disease, failure to thrive); check for maternal T-cell chimaerism if unexplained GvHD features in a male neonate",
            "IL2RG-GENE-THERAPY-OTL-101: OTL-101 lentiviral gene therapy FDA-approved 2024 for XSCID; early gamma-retroviral vectors caused insertional oncogenesis (T-cell leukaemia); lentiviral vectors have dramatically improved safety; specialist centres only",
            "IL2RG-T-B-PLUS-NK-MINUS: XSCID immunophenotype is T-B+NK- -- B cells are present (gamma-c not needed for B-cell development) but completely non-functional without T-cell help; NK cells absent (require IL-15/gamma-c); do not be misled by a normal B-cell count",
        ],
        "etiologies": [
            {"variant": "p.Arg222Cys (c.664C>T)", "type": "missense LOF -- extracellular domain", "frequency": "common", "severity": "severe XSCID"},
            {"variant": "p.Tyr103Ter (c.309C>A)", "type": "nonsense LOF", "frequency": "moderate", "severity": "severe"},
            {"variant": "Exon deletions (various)", "type": "LOF deletion", "frequency": "~20% of XSCID", "severity": "severe"},
            {"variant": "Splice site variants", "type": "LOF splice", "frequency": "~15% of XSCID", "severity": "variable"},
            {"variant": "p.Trp237Ter (c.711G>A)", "type": "nonsense LOF -- cytoplasmic domain", "frequency": "less common", "severity": "severe"},
        ],
        "stats": {
            "proportion_scid": "~45-50% of all SCID (most common SCID)",
            "immunophenotype": "T-B+NK-",
            "hsct_matched_sibling_survival": "~95%",
            "hsct_before_3m_advantage": "Best outcomes -- pre-infection HSCT target",
            "gene_therapy_otl101": "FDA approved 2024 -- lentiviral ex vivo",
        },
        "dx_delay_distribution": {
            "nbs_detected_0_1m": 30,
            "infant_1_3m": 42,
            "infant_3_6m": 20,
            "late_6m_plus": 8,
        },
    },

    # -- RAG1 -- Omenn Syndrome / RAG1-SCID -----------------------------------
    {
        "gene": "RAG1",
        "protein": (
            "RAG1 -- 11p13 AR -- RAG1-1043aa -- "
            "Omenn-Syndrome-Hypomorphic / RAG1-SCID-Complete-LOF -- "
            "V-D-J-Recombination-Enzyme -- "
            "Omenn: Erythroderma+Eosinophilia+Hepatosplenomegaly+Elevated-IgE -- "
            "Complete-LOF->T-B--SCID-NK-Cells-Present"
        ),
        "alias": (
            "RAG1 (recombination activating gene 1); OMIM gene 179615; "
            "Omenn Syndrome OMIM 267500; RAG1-SCID (combined immunodeficiency) OMIM 601457. "
            "11p13; 1043 aa; ~119 kDa; autosomal recessive. "
            "FUNCTION: RAG1 (together with RAG2) forms the RAG recombinase complex, "
            "which is essential for V(D)J recombination -- the process by which "
            "T-cell receptors (TCR) and B-cell receptors (BCR/immunoglobulins) generate diversity. "
            "RAG1/RAG2 introduce DNA double-strand breaks at recombination signal sequences (RSS) -> "
            "DNA repair machinery joins V, D, J segments randomly -> "
            "generates the vast TCR and BCR diversity (>10^18 possible combinations). "
            "WITHOUT RAG1: No V(D)J recombination -> No functional TCR -> No T cells; "
            "No functional BCR -> No B cells -> T-B- SCID (NK cells present). "
            "DUAL PHENOTYPE -- COMPLETE vs HYPOMORPHIC RAG1 MUTATIONS: "
            "COMPLETE LOF (biallelic null mutations): "
            "No V(D)J recombination possible -> T-B-NK+ SCID; "
            "no T or B cells; opportunistic infections from birth; "
            "HYPOMORPHIC MUTATIONS (partial residual RAG1 activity): "
            "Omenn Syndrome -- the oligoclonal T-cell expansion syndrome: "
            "a few T cells escape thymic selection -> oligoclonal activated T cells "
            "that are autoreactive -> multi-organ infiltration -> "
            "Erythroderma (total body erythematous rash): lymphocytic skin infiltration; "
            "Eosinophilia (eosinophils >1500/uL) -- Th2 cytokine skewing; "
            "Hepatosplenomegaly (lymphocytic organ infiltration); "
            "Elevated IgE (Th2 bias -- IL-4/IL-13 driven); "
            "Absent IgG, IgA, IgM (B cells absent/dysfunctional); "
            "Lymphadenopathy. "
            "OMENN TRIAD: erythroderma + eosinophilia + hepatosplenomegaly. "
            "DIAGNOSIS: "
            "Flow cytometry: T-B-NK+ SCID (complete LOF) or "
            "oligoclonal T+ (Omenn -- TCR spectratyping shows restricted repertoire); "
            "RAG1 gene sequencing; elevated serum IgE in Omenn; absent IgG, IgA, IgM. "
            "TREATMENT: "
            "HSCT: the ONLY curative option for both SCID and Omenn; "
            "Omenn pre-transplant: immunosuppression (ciclosporin + steroids) to control "
            "the autoreactive T cells and skin disease before HSCT; "
            "PJP prophylaxis (TMP-SMX); antifungal; IVIG; irradiated blood products; live vaccines CI."
        ),
        "locus": "11p13",
        "aa": 1043,
        "kDa": 119,
        "omim_gene": "179615",
        "omim_disease": "601457",
        "inheritance": "AR -- biallelic (complete LOF or hypomorphic)",
        "gene_class": "V(D)J recombinase -- adaptive immune receptor diversity generator",
        "key_alerts": [
            "RAG1-OMENN-ERYTHRODERMA-PATHOGNOMONIC: Omenn syndrome triad -- total erythroderma + eosinophilia + hepatosplenomegaly in an infant = hypomorphic RAG1/RAG2 (or other SCID gene) until proven otherwise; IgE elevated (unique among SCID presentations); oligoclonal T cells on TCR spectratyping",
            "RAG1-COMPLETE-LOF-T-B-MINUS: Complete RAG1 biallelic null mutations -> T-B-NK+ SCID; no T or B cells at all; NK cells present (do not use V(D)J recombination); distinguish from XSCID (T-B+NK-) and ADA-SCID (T-B-NK-) by immunophenotype",
            "RAG1-OMENN-IMMUNOSUPPRESSION-PRE-HSCT: Omenn syndrome requires ciclosporin + corticosteroids BEFORE HSCT to control autoreactive oligoclonal T-cell activation and skin disease; untreated Omenn -> progressive organ damage; HSCT is the only cure",
            "RAG1-IVIG-MANDATORY: In both SCID and Omenn, IgG, IgA, IgM are absent or severely reduced; IVIG replacement mandatory until post-HSCT B-cell reconstitution; Omenn may have high IgE but this is non-functional allergen-reactive Ig",
        ],
        "etiologies": [
            {"variant": "p.Arg559Ser (c.1675C>A)", "type": "hypomorphic missense -- Omenn", "frequency": "common Omenn", "severity": "Omenn syndrome"},
            {"variant": "p.Arg229Gln (c.686G>A)", "type": "hypomorphic missense", "frequency": "moderate Omenn", "severity": "Omenn syndrome"},
            {"variant": "p.Ala444Val (c.1331C>T)", "type": "hypomorphic missense", "frequency": "Omenn", "severity": "Omenn"},
            {"variant": "Frameshift/nonsense (biallelic)", "type": "complete LOF", "frequency": "SCID phenotype", "severity": "severe T-B-NK+ SCID"},
            {"variant": "p.Ser401Asn + complete LOF (compound het)", "type": "compound het Omenn", "frequency": "variable", "severity": "Omenn or partial SCID"},
        ],
        "stats": {
            "proportion_scid": "~10-15% of all SCID (RAG1+RAG2 combined)",
            "omenn_frequency": "Hypomorphic mutations -> ~50% present as Omenn",
            "ige_omenn": "Typically >1000 IU/mL",
            "eosinophilia_omenn": ">1500 eosinophils/uL",
            "hsct_survival": "~70-80% with MUD or haploidentical",
        },
        "dx_delay_distribution": {
            "neonatal_0_1m": 20,
            "infant_1_3m": 35,
            "infant_3_6m": 30,
            "late_6m_plus": 15,
        },
    },

    # -- WAS -- Wiskott-Aldrich Syndrome --------------------------------------
    {
        "gene": "WAS",
        "protein": (
            "WAS -- Xp11.23 XLR -- WASP-502aa -- "
            "Wiskott-Aldrich-Syndrome -- "
            "Eczema+Thrombocytopenia+Immunodeficiency-TRIAD-PATHOGNOMONIC -- "
            "Small-Platelets-MPV-less-than-7fL-PATHOGNOMONIC -- "
            "Splenectomy-CONTRAINDICATED -- "
            "HSCT-Curative -- Gene-Therapy-OTL-103"
        ),
        "alias": (
            "WAS (Wiskott-Aldrich Syndrome Protein; WASP); OMIM gene 300392; "
            "Wiskott-Aldrich Syndrome (WAS) OMIM 301000. "
            "Xp11.23; 502 aa; ~53 kDa; X-linked recessive -- affects males; females are carriers. "
            "FUNCTION: WASP is a haematopoietic cell-specific intracellular signalling molecule "
            "that links surface receptors to the actin cytoskeleton. "
            "WASP activates the Arp2/3 complex -> branched actin polymerisation -> "
            "essential for: "
            "(1) T-cell immune synapse formation (WASP required for TCR-APC contact); "
            "(2) NK-cell cytotoxic synapse; "
            "(3) B-cell receptor signalling and antibody class switching; "
            "(4) Platelet formation from megakaryocytes -- absence -> small, poorly functional platelets -- "
            "MICROPLATELETS, MPV <7 fL; "
            "(5) Dendritic cell migration and antigen presentation. "
            "THE PATHOGNOMONIC TRIAD: "
            "1. ECZEMA: typically severe, atopic-like; starts in infancy; may be the first sign; "
            "mechanism: impaired T regulatory cell function -> Th2 skewing -> atopic disease. "
            "2. THROMBOCYTOPENIA: low platelet COUNT + small platelet SIZE (microplatelets MPV <7 fL); "
            "bleeding risk: petechiae, bruising, gastrointestinal bleeding, intracranial haemorrhage (ICH); "
            "ICH is the most feared complication (10-15% of untreated patients); "
            "MICROPLATELETS ARE PATHOGNOMONIC: MPV <7 fL distinguishes WAS from all other "
            "thrombocytopenias (ITP, Bernard-Soulier, MYH9 disorders -- all have LARGE platelets). "
            "3. IMMUNODEFICIENCY: "
            "Recurrent sinopulmonary bacterial infections; "
            "Opportunistic infections (PJP, CMV, herpes); "
            "Autoimmune disease (haemolytic anaemia, vasculitis, nephritis) -- 40-70%; "
            "Lymphoma (especially EBV-driven B-cell lymphoma) -- long-term risk 13-22%. "
            "SPLENECTOMY -- ABSOLUTELY CONTRAINDICATED: "
            "Splenectomy raises platelet count but removes the last line of phagocytic defence -> "
            "overwhelming post-splenectomy infection (OPSI) from encapsulated bacteria. "
            "TREATMENT: "
            "HSCT: only curative option; best outcomes before age 5; MSD/MUD acceptable; "
            "excellent outcomes (>90% survival) at experienced centres; "
            "Gene therapy: OTL-103 (Orchard Therapeutics, lentiviral, ex vivo) -- clinical trials; "
            "IVIG (for antibody deficiency); TMP-SMX (PJP prophylaxis); "
            "Acyclovir/antiviral prophylaxis; "
            "Platelet transfusion for bleeding (not routine -- alloimmunisation risk)."
        ),
        "locus": "Xp11.23",
        "aa": 502,
        "kDa": 53,
        "omim_gene": "300392",
        "omim_disease": "301000",
        "inheritance": "XLR -- affects males; females are carriers",
        "gene_class": "Haematopoietic actin cytoskeleton regulator -- Arp2/3 activator (WASP)",
        "key_alerts": [
            "WAS-TRIAD-PATHOGNOMONIC: Eczema + thrombocytopenia + recurrent infections in a male infant = Wiskott-Aldrich Syndrome until proven otherwise; the combination of all three is virtually pathognomonic; any one alone has a broad differential",
            "WAS-MICROPLATELETS-PATHOGNOMONIC: Mean platelet volume (MPV) <7 fL with thrombocytopenia is PATHOGNOMONIC for WAS -- all other thrombocytopenias causing concern (ITP, TTP, Bernard-Soulier) have large or normal platelets; always request MPV with platelet count in infant thrombocytopenia",
            "WAS-SPLENECTOMY-CONTRAINDICATED: Splenectomy is ABSOLUTELY CONTRAINDICATED in WAS -- it raises platelets temporarily but removes critical anti-bacterial phagocytic defence and risks fatal overwhelming post-splenectomy infection (OPSI); HSCT is the only appropriate curative intervention",
            "WAS-ICH-RISK: Intracranial haemorrhage (ICH) occurs in 10-15% of untreated WAS -- a leading cause of death; maintain platelet count >20 x 10^9/L target; avoid NSAIDs and aspirin; any severe headache or altered consciousness in WAS = emergency CT head",
            "WAS-LYMPHOMA-RISK: EBV-driven B-cell lymphoma risk is 13-22% by adult age in WAS; annual surveillance with EBV PCR (quantitative); any lymphadenopathy + EBV viraemia -> investigate urgently; HSCT before lymphoma is strongly preferred",
        ],
        "etiologies": [
            {"variant": "p.Arg86Cys (c.256C>T)", "type": "missense -- WH1 domain", "frequency": "common -- mild phenotype", "severity": "XLT (X-linked thrombocytopenia)"},
            {"variant": "p.Arg86His (c.257G>A)", "type": "missense -- WH1 domain", "frequency": "common -- variable", "severity": "variable WAS/XLT"},
            {"variant": "p.Ala47Thr (c.139G>A)", "type": "missense", "frequency": "moderate", "severity": "classical WAS"},
            {"variant": "Exon deletions/frameshift", "type": "LOF truncating", "frequency": "~30% WAS", "severity": "severe WAS"},
            {"variant": "Splice site mutations", "type": "LOF splice", "frequency": "~20% WAS", "severity": "variable"},
        ],
        "stats": {
            "incidence": "~1:100,000 male births",
            "platelet_count": "20-80 x 10^9/L typical",
            "mpv": "<7 fL (pathognomonic)",
            "ich_risk": "10-15% without definitive treatment",
            "autoimmune_risk": "40-70%",
            "lymphoma_risk": "13-22% by adult age",
        },
        "dx_delay_distribution": {
            "neonatal_0_1m": 35,
            "infant_1_6m": 40,
            "child_6m_2y": 20,
            "late_2y_plus": 5,
        },
    },

    # -- DOCK8 -- DOCK8 Deficiency / HIES Type 2 ------------------------------
    {
        "gene": "DOCK8",
        "protein": (
            "DOCK8 -- 9p24.3 AR -- DOCK8-2099aa -- "
            "DOCK8-Deficiency-HIES-Type-2-Hyper-IgE-Syndrome-2 -- "
            "Severe-Eczema+Cutaneous-Viral-Molluscum-HPV-PATHOGNOMONIC -- "
            "Elevated-IgE+Low-IgM -- "
            "STAT3-HIES-TYPE-1-AD-form-KEY-DDx -- "
            "HSCT-Curative"
        ),
        "alias": (
            "DOCK8 (dedicator of cytokinesis 8); OMIM gene 611432; "
            "Hyper-IgE Syndrome type 2 (HIES2; DOCK8 deficiency) OMIM 243700. "
            "9p24.3; 2099 aa; ~237 kDa; autosomal recessive -- biallelic LOF. "
            "FUNCTION: DOCK8 is a guanine nucleotide exchange factor (GEF) for CDC42 and RAC1, "
            "activating Rho-family GTPases that regulate the actin cytoskeleton. "
            "DOCK8 is essential for: "
            "(1) NK-cell and T-cell immune synapse formation and cytotoxic killing; "
            "(2) T-cell survival in peripheral non-lymphoid tissues (especially skin); "
            "(3) B-cell migration and survival in germinal centres; "
            "(4) NK-cell antiviral responses in peripheral tissues. "
            "DOCK8 LOF -> NK cells fail to form immunological synapses against virus-infected cells -> "
            "T cells fail to survive in peripheral tissues -> "
            "profound susceptibility to cutaneous viral infections. "
            "CLINICAL PHENOTYPE -- THE DOCK8 FINGERPRINT: "
            "SEVERE ECZEMA: recalcitrant atopic dermatitis; begins in infancy. "
            "CUTANEOUS VIRAL INFECTIONS -- PATHOGNOMONIC: "
            "Molluscum contagiosum: extensive, confluent, resistant to treatment -- "
            "hundreds of lesions; persistent despite standard treatments; "
            "Human papillomavirus (HPV): extensive warts (verrucae); anogenital HPV; "
            "HPV-associated squamous cell carcinoma risk; "
            "Herpes simplex virus (HSV): recurrent extensive herpetic lesions; eczema herpeticum; "
            "Varicella-zoster virus (VZV): severe primary chickenpox; recurrent zoster. "
            "THESE EXTENSIVE CUTANEOUS VIRAL INFECTIONS ARE THE PATHOGNOMONIC FEATURE: "
            "extensive Molluscum + HPV warts in a child with eczema = DOCK8 until proven otherwise. "
            "IMMUNOLOGICAL PARAMETERS: "
            "Elevated serum IgE: often >1000-10000 IU/mL; "
            "Low serum IgM: characteristic (unlike STAT3-HIES where IgM is typically normal); "
            "Variable IgG; Low NK-cell function; Low CD8+ T-cell counts. "
            "MALIGNANCY RISK: HPV-associated squamous cell carcinoma; EBV-associated lymphoma. "
            "STAT3-HIES (TYPE 1) -- KEY DDx: "
            "STAT3 GOF mutations -> HIES type 1 (autosomal DOMINANT): "
            "eczema + elevated IgE + recurrent pneumonias + skeletal abnormalities "
            "(hyperextensible joints, retained primary teeth, scoliosis) + coarse facies; "
            "Pneumatocele formation (hallmark of STAT3-HIES, NOT seen in DOCK8); "
            "DOCK8: AR, more severe T and NK defects, cutaneous viral infections dominant; "
            "STAT3-HIES: AD, skeletal/dental/pulmonary dominant. "
            "TREATMENT: "
            "HSCT: curative for DOCK8 deficiency; eliminates skin infections post-transplant; "
            "IFN-alpha: some benefit for Molluscum; Prophylactic antivirals (acyclovir) and antibiotics; "
            "HPV vaccination (before HPV exposure if possible); IVIG (antibody replacement)."
        ),
        "locus": "9p24.3",
        "aa": 2099,
        "kDa": 237,
        "omim_gene": "611432",
        "omim_disease": "243700",
        "inheritance": "AR -- biallelic loss-of-function",
        "gene_class": "Rho-GEF actin cytoskeleton regulator -- CDC42/RAC1 GEF in lymphocytes",
        "key_alerts": [
            "DOCK8-CUTANEOUS-VIRAL-INFECTIONS-PATHOGNOMONIC: Extensive Molluscum contagiosum + HPV warts in a child with severe eczema = DOCK8 deficiency until proven otherwise; hundreds of Molluscum lesions resistant to treatment; extensive anogenital HPV warts; this combination is the PATHOGNOMONIC fingerprint of DOCK8 deficiency",
            "DOCK8-VS-STAT3-HIES-DDx: DOCK8 deficiency (AR) and STAT3 HIES (AD) both cause eczema + elevated IgE but differ: DOCK8 has cutaneous viral infections + low IgM + T/NK dysfunction; STAT3-HIES has pneumatoceles + retained primary teeth + skeletal abnormalities + coarse facies -- perform STAT3 sequencing simultaneously",
            "DOCK8-HPV-CARCINOMA-RISK: Extensive HPV infection -> HPV-associated squamous cell carcinoma risk (anal, oropharyngeal, vulval/cervical); annual surveillance with gynaecological and dermatological examination; HPV vaccination should be offered early; HSCT may halt HPV-driven dysplasia progression",
            "DOCK8-LOW-IGM-CLUE: Low serum IgM is characteristic of DOCK8 deficiency and helps distinguish it from STAT3-HIES (where IgM is usually normal); combined low IgM + elevated IgE + eczema + cutaneous viral infections = DOCK8 panel sequencing mandatory",
            "DOCK8-HSCT-CURATIVE: HSCT is the only curative treatment; post-HSCT resolution of Molluscum contagiosum and HPV warts is dramatic and often complete; best outcomes with early HSCT before HPV-associated malignancy or organ damage develops",
        ],
        "etiologies": [
            {"variant": "Large exon deletions (genomic)", "type": "LOF deletion -- most common", "frequency": "~50% of DOCK8 deficiency", "severity": "severe"},
            {"variant": "Frameshift insertions/deletions", "type": "LOF frameshift", "frequency": "~25%", "severity": "severe"},
            {"variant": "Splice site mutations", "type": "LOF splice", "frequency": "~15%", "severity": "severe"},
            {"variant": "Nonsense mutations (various exons)", "type": "LOF nonsense", "frequency": "~10%", "severity": "severe"},
            {"variant": "p.Arg1749Ter (c.5245C>T)", "type": "nonsense LOF", "frequency": "reported", "severity": "severe"},
        ],
        "stats": {
            "proportion_hies": "AR form -- HIES type 2; STAT3-HIES is the AD type 1",
            "ige_range": "1000-100000 IU/mL",
            "igm_level": "Low -- characteristic DDx from STAT3-HIES",
            "molluscum_prevalence": ">90% of DOCK8 patients",
            "malignancy_risk": "Significant HPV-SCC and EBV-lymphoma risk",
        },
        "dx_delay_distribution": {
            "infant_0_12m": 15,
            "child_1_5y": 45,
            "child_5_10y": 30,
            "late_10y_plus": 10,
        },
    },

    # -- TNFRSF13B / TACI -- CVID2 --------------------------------------------
    {
        "gene": "TNFRSF13B",
        "protein": (
            "TNFRSF13B -- 17p11.2 AD/AR -- TACI-293aa -- "
            "CVID2-Common-Variable-Immunodeficiency-type-2 -- "
            "B-cell-survival-receptor-for-BAFF-and-APRIL -- "
            "LOF->Hypogammaglobulinaemia-IgG+IgA-predominantly -- "
            "IVIG-SCIG-Lifelong -- Granulomatous-Disease-10-20pct -- "
            "Autoimmune-Cytopenias -- Lymphoma-Risk-5-fold"
        ),
        "alias": (
            "TNFRSF13B (tumour necrosis factor receptor superfamily member 13B; TACI -- "
            "transmembrane activator and CAML interactor); OMIM gene 604907; "
            "Common Variable Immunodeficiency type 2 (CVID2) OMIM 240500. "
            "17p11.2; 293 aa; ~32 kDa; autosomal dominant (heterozygous LOF) or "
            "autosomal recessive (biallelic LOF -- typically more severe). "
            "FUNCTION: TACI is a receptor on B cells that binds two key survival/differentiation factors: "
            "BAFF (B-cell activating factor of the TNF family; BLyS) and APRIL "
            "(A proliferation-inducing ligand). "
            "TACI signalling is essential for: "
            "(1) Class-switch recombination (IgM -> IgG, IgA, IgE); "
            "(2) B-cell survival in the marginal zone and germinal centre; "
            "(3) Plasma cell differentiation and long-lived plasma cell maintenance. "
            "TACI LOF -> impaired BAFF/APRIL signalling -> "
            "failure of B-cell class-switching -> predominantly IgG and IgA deficiency -> "
            "hypogammaglobulinaemia. "
            "CVID -- THE CLINICAL SYNDROME: "
            "Most common symptomatic primary immunodeficiency in adults; "
            "Onset: typically second-fourth decade (bimodal: 5-10y and 20-40y); "
            "DIAGNOSTIC CRITERIA for CVID: "
            "Serum IgG <7 g/L (>2 SD below age-normal); "
            "One or both of IgA <0.07 g/L and IgM <0.40 g/L; "
            "Absent vaccine responses (pneumococcal/tetanus); "
            "Age >2 years; other causes excluded. "
            "TNFRSF13B is found in ~8-10% of CVID cases. "
            "CLINICAL COMPLICATIONS: "
            "Recurrent sinopulmonary bacterial infections: Streptococcus pneumoniae, H. influenzae; "
            "Giardia lamblia intestinal infection (secretory IgA deficiency); "
            "Granulomatous disease: sarcoid-like non-caseating granulomas in lung (GLILD), "
            "liver, spleen, lymph nodes -- in 10-20% of CVID; "
            "GLILD (granulomatous-lymphocytic interstitial lung disease): "
            "progressive pulmonary infiltrates + restrictive lung disease; "
            "CT chest: bilateral ground-glass opacities, nodules, hilar lymphadenopathy; "
            "Autoimmune cytopenias: AIHA, ITP, neutropenia -- in ~20-30% of CVID; "
            "Lymphoma risk: 5-fold increased risk versus general population; "
            "particularly MALT lymphoma and diffuse large B-cell lymphoma. "
            "LIVE VACCINES CI: no live vaccines -- antibody-deficient, impaired immune responses. "
            "TREATMENT: "
            "IVIG: 400-600 mg/kg q3-4 weeks; target trough >8 g/L; "
            "SCIG: equivalent subcutaneous alternative -- preferred by many patients; "
            "Granulomatous disease: rituximab +/- steroids; "
            "Autoimmune cytopenias: steroids, IVIG high-dose, rituximab."
        ),
        "locus": "17p11.2",
        "aa": 293,
        "kDa": 32,
        "omim_gene": "604907",
        "omim_disease": "240500",
        "inheritance": "AD (heterozygous LOF -- incomplete penetrance) or AR (biallelic -- more severe)",
        "gene_class": "TNF receptor superfamily -- BAFF/APRIL B-cell survival receptor",
        "key_alerts": [
            "TNFRSF13B-CVID-IVIG-LIFELONG: CVID requires lifelong IVIG or SCIG replacement -- not a temporary measure; target trough IgG >8 g/L (>10 g/L with bronchiectasis or GLILD); inadequate replacement -> progressive bronchiectasis; never stop replacement without specialist review",
            "TNFRSF13B-GLILD-PULMONARY: Granulomatous-lymphocytic interstitial lung disease (GLILD) in 10-20% of CVID -- sarcoid-like granulomas in lung, liver, spleen; CT chest: bilateral nodules, ground-glass, hilar lymphadenopathy; rituximab +/- steroids are treatment; annual lung function testing mandatory",
            "TNFRSF13B-AUTOIMMUNE-CYTOPENIAS: Autoimmune haemolytic anaemia and ITP occur in 20-30% of CVID -- paradoxically, immune dysregulation alongside immunodeficiency; AIHA + ITP workup (DAT, reticulocytes, platelet antibodies) at any cytopaenia",
            "TNFRSF13B-LYMPHOMA-SURVEILLANCE: 5-fold increased lymphoma risk in CVID -- annual examination for lymphadenopathy; LDH monitoring; PET-CT if suspicious lymphadenopathy; EBV PCR annually; any unexplained B symptoms -> urgent lymphoma workup",
            "TNFRSF13B-LIVE-VACCINES-CI: No live vaccines -- CVID patients cannot generate protective responses and live organisms may cause disease; all household contacts should use inactivated vaccines; annual inactivated influenza vaccine recommended",
        ],
        "etiologies": [
            {"variant": "p.Cys104Arg (c.310T>C)", "type": "LOF missense -- BAFF-R binding domain", "frequency": "most common CVID-associated TACI variant", "severity": "CVID -- variable penetrance"},
            {"variant": "p.Ala181Glu (c.542C>A)", "type": "LOF missense", "frequency": "common -- AD", "severity": "CVID"},
            {"variant": "p.Pro251Leu (c.752C>T)", "type": "missense -- partial LOF", "frequency": "moderate", "severity": "mild CVID"},
            {"variant": "Homozygous TACI LOF", "type": "AR biallelic", "frequency": "rare -- more severe", "severity": "severe CVID"},
            {"variant": "p.Arg202His (c.605G>A)", "type": "missense LOF", "frequency": "moderate", "severity": "CVID"},
        ],
        "stats": {
            "proportion_cvid": "~8-10% of CVID has TNFRSF13B variants",
            "igg_at_diagnosis": "<5 g/L typical",
            "granulomatous_disease": "10-20% of CVID",
            "autoimmune_cytopenias": "20-30% of CVID",
            "lymphoma_risk": "5-fold above general population",
        },
        "dx_delay_distribution": {
            "child_1_5y": 10,
            "child_5_15y": 25,
            "young_adult_15_30y": 45,
            "adult_30y_plus": 20,
        },
    },

    # -- LRBA -- LRBA Deficiency / CVID8 --------------------------------------
    {
        "gene": "LRBA",
        "protein": (
            "LRBA -- 4q31.3 AR -- LRBA-2863aa -- "
            "CVID8-LRBA-Deficiency-Immune-Dysregulation -- "
            "Regulates-CTLA-4-Recycling -- "
            "LOF->CTLA-4-Degraded->Uncontrolled-T-Cell-Activation -- "
            "AIHA+IBD+Interstitial-Lung-Disease-DOMINANT -- "
            "Abatacept-DRAMATICALLY-Effective-PATHOGNOMONIC-TREATMENT-RESPONSE -- "
            "IVIG-Abatacept-HSCT-Curative"
        ),
        "alias": (
            "LRBA (LPS-responsive beige-like anchor protein); OMIM gene 606453; "
            "Common Variable Immunodeficiency type 8 with autoimmunity (CVID8) OMIM 614700. "
            "4q31.3; 2863 aa; ~319 kDa; autosomal recessive -- biallelic LOF. "
            "FUNCTION: LRBA is a member of the BEACH (beige and Chediak-Higashi) domain-containing "
            "protein family involved in intracellular vesicle trafficking. "
            "KEY MOLECULAR FUNCTION: LRBA regulates the intracellular recycling of CTLA-4 "
            "(cytotoxic T-lymphocyte antigen 4; CD152). "
            "CTLA-4 BIOLOGY: "
            "CTLA-4 is the master negative regulator of T-cell activation: "
            "CTLA-4 competes with CD28 for binding to CD80/CD86 on antigen-presenting cells -> "
            "CTLA-4 binding inhibits T-cell activation (dominant-negative over CD28 co-stimulation); "
            "CTLA-4 is constitutively expressed on regulatory T cells (Tregs) -- "
            "Tregs use CTLA-4 to suppress effector T cells. "
            "LRBA-CTLA-4 RECYCLING: "
            "After CTLA-4 is internalised (endocytosis from cell surface) -> "
            "LRBA is required to recycle CTLA-4 from endosomes back to the cell surface; "
            "LRBA LOF -> CTLA-4 is not recycled -> directed to lysosomes -> degraded -> "
            "CTLA-4 surface expression dramatically reduced on T cells and Tregs. "
            "CONSEQUENCE: Without CTLA-4 -> T-cell activation unchecked -> "
            "effector T cells overactivated -> immune dysregulation -> "
            "autoimmune attack on multiple organs. "
            "CLINICAL PHENOTYPE: "
            "IMMUNE DYSREGULATION dominates the presentation (unlike other CVID): "
            "Autoimmune haemolytic anaemia (AIHA): Coombs-positive -- common presenting feature; "
            "Evans syndrome (AIHA + ITP simultaneously); "
            "Inflammatory bowel disease (IBD): Crohn-like or ulcerative colitis-like; "
            "Granulomatous-lymphocytic interstitial lung disease (GLILD); "
            "Autoimmune hepatitis; Arthritis; "
            "Hypogammaglobulinaemia (IgG low, IgA low): IVIG replacement needed; "
            "Lymphadenopathy, splenomegaly. "
            "TREATMENT -- THE ABATACEPT RESPONSE: "
            "Abatacept (CTLA-4 Ig fusion protein -- Orencia): "
            "Abatacept = CTLA-4 extracellular domain fused to IgG1 Fc; "
            "provides exogenous CTLA-4 function -> blocks CD80/CD86 -> suppresses overactivated T cells -> "
            "DRAMATICALLY effective in LRBA deficiency -- "
            "clinical response is rapid and striking (autoimmune features resolve, "
            "AIHA corrects, IBD improves, lung disease stabilises); "
            "this dramatic response to abatacept is PATHOGNOMONIC for LRBA deficiency "
            "(and CTLA-4 haploinsufficiency); "
            "IVIG: concurrent antibody replacement; "
            "HSCT: curative; considered for severe cases or abatacept-refractory."
        ),
        "locus": "4q31.3",
        "aa": 2863,
        "kDa": 319,
        "omim_gene": "606453",
        "omim_disease": "614700",
        "inheritance": "AR -- biallelic loss-of-function",
        "gene_class": "BEACH-domain vesicle trafficking protein -- CTLA-4 endosomal recycling regulator",
        "key_alerts": [
            "LRBA-ABATACEPT-PATHOGNOMONIC-RESPONSE: Dramatic, rapid response to abatacept (CTLA-4 Ig) is PATHOGNOMONIC for LRBA deficiency -- AIHA resolves, IBD improves, lung disease stabilises; this response distinguishes LRBA from other CVID; if abatacept-responsive immune dysregulation -> sequence LRBA immediately",
            "LRBA-IMMUNE-DYSREGULATION-DOMINATES: Unlike typical CVID, LRBA deficiency presents with IMMUNE DYSREGULATION as the dominant feature -- AIHA, Evans syndrome, IBD, hepatitis -- not just recurrent infections; the combination of hypogammaglobulinaemia + autoimmunity in a young patient = LRBA deficiency until proven otherwise",
            "LRBA-CTLA4-HAPLOINSUFFICIENCY-DDx: CTLA4 heterozygous LOF mutations cause a similar syndrome (CTLA-4 haploinsufficiency -- CHAI disease) -- also abatacept-responsive; sequence both LRBA and CTLA4 when abatacept-responsive immune dysregulation is found; LRBA is AR (biallelic); CTLA4 haploinsufficiency is AD (heterozygous)",
            "LRBA-IVIG-PLUS-ABATACEPT: LRBA deficiency requires BOTH IVIG (for antibody deficiency) AND abatacept (for immune dysregulation/CTLA-4 deficiency) -- one without the other is insufficient; do not use abatacept alone without addressing hypogammaglobulinaemia",
            "LRBA-HSCT-CURATIVE: HSCT is curative for LRBA deficiency and can replace the need for lifelong abatacept; considered when abatacept is insufficient or for young patients with a suitable donor; post-HSCT immune reconstitution restores LRBA function",
        ],
        "etiologies": [
            {"variant": "Large exon/multiexon deletions", "type": "LOF deletion -- most common", "frequency": "~40% of LRBA deficiency", "severity": "severe immune dysregulation"},
            {"variant": "Frameshift insertions/deletions", "type": "LOF frameshift", "frequency": "~25%", "severity": "severe"},
            {"variant": "Splice site mutations", "type": "LOF splice", "frequency": "~20%", "severity": "moderate-severe"},
            {"variant": "p.Gln2253Ter (c.6757C>T)", "type": "nonsense LOF", "frequency": "reported", "severity": "severe"},
            {"variant": "p.Leu3082Arg (c.9245T>G)", "type": "missense LOF -- BEACH domain", "frequency": "reported", "severity": "severe"},
        ],
        "stats": {
            "igg_at_diagnosis": "<5 g/L typical",
            "aiha_prevalence": "~60-70% of LRBA patients",
            "ibd_prevalence": "~40% of LRBA patients",
            "glild_prevalence": "~30% of LRBA patients",
            "abatacept_response": "Dramatic -- pathognomonic for CTLA-4 pathway defects",
        },
        "dx_delay_distribution": {
            "infant_0_12m": 20,
            "child_1_5y": 40,
            "child_5_15y": 30,
            "adult_15y_plus": 10,
        },
    },
]


def _generate_patients():
    """Generate 40 deterministic synthetic patients per gene using seeded RNG."""
    for idx, gene_data in enumerate(IMID_GENES):
        seed = SEED_BASE + idx
        rng = random.Random(seed)
        gene = gene_data["gene"]
        patients = []

        for i in range(40):
            if gene == "BTK":
                # XLR -- all males
                age_dx_months = rng.randint(6, 36)
                dx_delay_months = rng.randint(3, 24)
                igg_at_dx = round(rng.uniform(0.01, 0.15), 2)
                b_cells_pct = round(rng.uniform(0.0, 0.8), 1)
                presenting_infection = rng.choices(
                    ["pneumonia", "sinusitis", "meningitis", "otitis_media", "septicaemia", "giardiasis"],
                    weights=[35, 25, 10, 15, 10, 5]
                )[0]
                ivig_trough_achieved = round(rng.uniform(6.5, 12.0), 1)
                bronchiectasis = rng.random() < (0.3 if age_dx_months < 18 else 0.5)
                enteroviral_enc = rng.random() < 0.05
                patients.append({
                    "patient_id": f"BTK-{i+1:03d}",
                    "sex": "M",
                    "age_dx_months": age_dx_months,
                    "dx_delay_months": dx_delay_months,
                    "igg_at_dx_gL": igg_at_dx,
                    "cd19_b_cells_pct": b_cells_pct,
                    "presenting_infection": presenting_infection,
                    "ivig_trough_gL": ivig_trough_achieved,
                    "bronchiectasis": bronchiectasis,
                    "enteroviral_encephalitis": enteroviral_enc,
                    "live_vaccines_ci": True,
                    "gene": gene, "seed": seed,
                })

            elif gene == "ADA":
                sex = rng.choices(["M", "F"], weights=[50, 50])[0]
                dx_age_days = rng.randint(0, 90)
                lymphocyte_count = rng.randint(150, 800)
                ada_activity_pct = round(rng.uniform(0.1, 1.0), 1)
                datp_elevated = True
                skeletal_dysplasia = rng.random() < 0.50
                treatment = rng.choices(
                    ["gene_therapy_strimvelis", "hsct_msd", "hsct_mud", "peg_ada_bridge"],
                    weights=[25, 20, 35, 20]
                )[0]
                nbs_detected = rng.random() < 0.30
                patients.append({
                    "patient_id": f"ADA-{i+1:03d}",
                    "sex": sex,
                    "age_dx_days": dx_age_days,
                    "lymphocyte_count_per_uL": lymphocyte_count,
                    "ada_activity_pct_normal": ada_activity_pct,
                    "datp_elevated": datp_elevated,
                    "skeletal_dysplasia_xray": skeletal_dysplasia,
                    "nbs_detected": nbs_detected,
                    "treatment": treatment,
                    "gene": gene, "seed": seed,
                })

            elif gene == "IL2RG":
                # XLR -- all males
                dx_age_days = rng.randint(0, 90)
                immunophenotype = "T-B+NK-"
                presenting_infection = rng.choices(
                    ["pjp", "cmv", "failure_to_thrive", "rsv_severe", "candidiasis"],
                    weights=[30, 20, 20, 15, 15]
                )[0]
                maternal_gvhd = rng.random() < 0.20
                treatment = rng.choices(
                    ["hsct_msd", "hsct_mud", "hsct_haploidentical", "gene_therapy_otl101"],
                    weights=[20, 45, 20, 15]
                )[0]
                irradiated_blood_used = True
                nbs_detected = rng.random() < 0.35
                patients.append({
                    "patient_id": f"IL2RG-{i+1:03d}",
                    "sex": "M",
                    "age_dx_days": dx_age_days,
                    "immunophenotype": immunophenotype,
                    "presenting_infection": presenting_infection,
                    "maternal_lymphocyte_gvhd": maternal_gvhd,
                    "irradiated_blood_products": irradiated_blood_used,
                    "nbs_detected": nbs_detected,
                    "treatment": treatment,
                    "live_vaccines_ci": True,
                    "gene": gene, "seed": seed,
                })

            elif gene == "RAG1":
                sex = rng.choices(["M", "F"], weights=[50, 50])[0]
                phenotype = rng.choices(
                    ["omenn_syndrome", "scid_t_b_minus", "partial_scid"],
                    weights=[50, 35, 15]
                )[0]
                ige_level = round(rng.uniform(1200, 18000), 0) if phenotype == "omenn_syndrome" else round(rng.uniform(0.1, 2.0), 1)
                eosinophilia = rng.random() < 0.90 if phenotype == "omenn_syndrome" else rng.random() < 0.10
                erythroderma = (phenotype == "omenn_syndrome")
                dx_age_days = rng.randint(7, 120)
                treatment = rng.choices(
                    ["hsct_msd", "hsct_mud", "hsct_haploidentical"],
                    weights=[15, 50, 35]
                )[0]
                pre_hsct_immunosuppression = (phenotype == "omenn_syndrome")
                patients.append({
                    "patient_id": f"RAG1-{i+1:03d}",
                    "sex": sex,
                    "age_dx_days": dx_age_days,
                    "phenotype": phenotype,
                    "erythroderma": erythroderma,
                    "eosinophilia": eosinophilia,
                    "ige_iu_mL": ige_level,
                    "pre_hsct_immunosuppression": pre_hsct_immunosuppression,
                    "treatment": treatment,
                    "gene": gene, "seed": seed,
                })

            elif gene == "WAS":
                # XLR -- all males
                age_dx_months = rng.randint(0, 12)
                platelet_count = rng.randint(20, 80)
                mpv_fL = round(rng.uniform(4.0, 6.8), 1)
                eczema_severity = rng.choices(["mild", "moderate", "severe"], weights=[15, 35, 50])[0]
                ich_event = rng.random() < 0.10
                autoimmune = rng.random() < 0.45
                autoimmune_type = rng.choice(["aiha", "itp_additional", "vasculitis", "nephritis"]) if autoimmune else None
                lymphoma = rng.random() < 0.08
                treatment = rng.choices(
                    ["hsct_msd", "hsct_mud", "hsct_haploidentical", "gene_therapy_otl103"],
                    weights=[20, 45, 20, 15]
                )[0]
                patients.append({
                    "patient_id": f"WAS-{i+1:03d}",
                    "sex": "M",
                    "age_dx_months": age_dx_months,
                    "platelet_count_x10_9_L": platelet_count,
                    "mpv_fL": mpv_fL,
                    "eczema_severity": eczema_severity,
                    "ich_event": ich_event,
                    "autoimmune_complication": autoimmune,
                    "autoimmune_type": autoimmune_type,
                    "lymphoma": lymphoma,
                    "splenectomy_ci": True,
                    "splenectomy_performed": False,
                    "treatment": treatment,
                    "gene": gene, "seed": seed,
                })

            elif gene == "DOCK8":
                sex = rng.choices(["M", "F"], weights=[50, 50])[0]
                age_dx_years = rng.randint(1, 12)
                ige_level = round(rng.uniform(1000, 50000), 0)
                igm_low = True
                molluscum_extensive = rng.random() < 0.92
                hpv_warts = rng.random() < 0.75
                hsv_recurrent = rng.random() < 0.60
                hpv_scc = rng.random() < (0.10 if age_dx_years > 8 else 0.02)
                eczema_severity = rng.choices(["moderate", "severe"], weights=[30, 70])[0]
                stat3_excluded = True
                treatment = rng.choices(
                    ["hsct_mud", "hsct_haploidentical", "supportive_ivig_antivirals"],
                    weights=[50, 25, 25]
                )[0]
                patients.append({
                    "patient_id": f"DOCK8-{i+1:03d}",
                    "sex": sex,
                    "age_dx_years": age_dx_years,
                    "ige_iu_mL": ige_level,
                    "igm_low": igm_low,
                    "molluscum_extensive": molluscum_extensive,
                    "hpv_warts_extensive": hpv_warts,
                    "hsv_recurrent": hsv_recurrent,
                    "hpv_associated_scc": hpv_scc,
                    "eczema_severity": eczema_severity,
                    "stat3_hies_excluded": stat3_excluded,
                    "treatment": treatment,
                    "gene": gene, "seed": seed,
                })

            elif gene == "TNFRSF13B":
                sex = rng.choices(["M", "F"], weights=[45, 55])[0]
                age_dx_years = rng.randint(5, 55)
                igg_at_dx = round(rng.uniform(1.5, 5.5), 1)
                iga_at_dx = round(rng.uniform(0.01, 0.5), 2)
                vaccine_response = "absent"
                presenting_feature = rng.choices(
                    ["recurrent_sinopulmonary", "giardiasis", "autoimmune_cytopenia", "glild", "lymphoma_workup"],
                    weights=[45, 10, 20, 15, 10]
                )[0]
                granulomatous_disease = rng.random() < 0.15
                autoimmune_cytopenia = rng.random() < 0.25
                lymphoma_risk_monitoring = rng.random() < 0.08
                inheritance_pattern = rng.choices(["AD_het", "AR_biallelic"], weights=[80, 20])[0]
                patients.append({
                    "patient_id": f"TACI-{i+1:03d}",
                    "sex": sex,
                    "age_dx_years": age_dx_years,
                    "igg_at_dx_gL": igg_at_dx,
                    "iga_at_dx_gL": iga_at_dx,
                    "vaccine_response": vaccine_response,
                    "presenting_feature": presenting_feature,
                    "glild_granulomatous": granulomatous_disease,
                    "autoimmune_cytopenia": autoimmune_cytopenia,
                    "lymphoma_surveillance_flag": lymphoma_risk_monitoring,
                    "inheritance_pattern": inheritance_pattern,
                    "ivig_or_scig_lifelong": True,
                    "gene": gene, "seed": seed,
                })

            else:  # LRBA
                sex = rng.choices(["M", "F"], weights=[50, 50])[0]
                age_dx_years = rng.randint(1, 20)
                igg_at_dx = round(rng.uniform(1.0, 4.5), 1)
                aiha = rng.random() < 0.65
                ibd = rng.random() < 0.42
                glild = rng.random() < 0.30
                autoimmune_hepatitis = rng.random() < 0.20
                evans_syndrome = aiha and rng.random() < 0.40
                abatacept_response = rng.choices(
                    ["dramatic_response", "good_response", "partial_response"],
                    weights=[55, 30, 15]
                )[0]
                ctla4_surface_expression = round(rng.uniform(5, 25), 1)  # % of normal
                hsct_planned = rng.random() < 0.25
                patients.append({
                    "patient_id": f"LRBA-{i+1:03d}",
                    "sex": sex,
                    "age_dx_years": age_dx_years,
                    "igg_at_dx_gL": igg_at_dx,
                    "aiha": aiha,
                    "ibd": ibd,
                    "glild_lung": glild,
                    "autoimmune_hepatitis": autoimmune_hepatitis,
                    "evans_syndrome": evans_syndrome,
                    "ctla4_surface_expression_pct_normal": ctla4_surface_expression,
                    "abatacept_response": abatacept_response,
                    "hsct_planned_or_completed": hsct_planned,
                    "ivig_replacement": True,
                    "gene": gene, "seed": seed,
                })
        gene_data["patients"] = patients


_generate_patients()


def overview():
    all_genes_info = [
        {
            "gene": g["gene"],
            "locus": g["locus"],
            "aa": g["aa"],
            "n_patients": len(g["patients"]),
        }
        for g in IMID_GENES
    ]
    total = sum(len(g["patients"]) for g in IMID_GENES)
    return {
        "atlas": "Hereditary Immunodeficiency Atlas -- Complete 8-Gene Primary Immunodeficiency Atlas",
        "subtitle": (
            "BTK (XLA-Agammaglobulinaemia) . ADA (ADA-SCID-T-B-NK-) . IL2RG (XSCID-T-B+NK-) . "
            "RAG1 (Omenn/RAG1-SCID) . WAS (Wiskott-Aldrich-Triad) . "
            "DOCK8 (HIES2-Cutaneous-Viral) . TNFRSF13B (CVID2-TACI) . LRBA (CVID8-Abatacept) -- "
            "320 Patients (8x40, Seeds 1782-1789)"
        ),
        "total_patients": total,
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_stats": {
            "genes_covered": 8,
            "patients_per_gene": 40,
            "xlr_genes": 3,
            "ar_genes": 4,
            "ad_ar_genes": 1,
            "scid_genes": 3,
            "antibody_deficiency_genes": 2,
            "combined_immunodeficiency_genes": 3,
        },
        "genes": all_genes_info,
        "top_alerts": [
            "BTK-LIVE-VACCINES-ABSOLUTELY-CI: ALL live vaccines absolutely contraindicated in XLA -- OPV causes vaccine-derived poliomyelitis; BCG causes disseminated BCGosis; MMR, varicella, rotavirus all CI; enteroviral encephalitis is fatal XLA complication; IVIG trough >8 g/L is mandatory lifelong",
            "ADA-GENE-THERAPY-STRIMVELIS-APPROVED: ADA-SCID is the first SCID with EMA-approved gene therapy (Strimvelis 2016); dATP accumulation causes T-B-NK- pan-lymphopenia; NBS (TREC) detects pre-symptomatically; skeletal dysplasia on CXR is a unique ADA-SCID radiological clue",
            "IL2RG-IRRADIATED-BLOOD-MANDATORY-AND-LIVE-VACCINES-CI: All XSCID blood products must be irradiated + CMV-negative -- transfusion-associated GvHD is fatal; maternal lymphocyte GvHD also possible; OTL-101 lentiviral gene therapy FDA approved 2024; HSCT before 3 months gives best outcomes",
            "RAG1-OMENN-TRIAD-IMMUNOSUPPRESSION-BEFORE-HSCT: Omenn syndrome (erythroderma + eosinophilia + hepatosplenomegaly + elevated IgE) from hypomorphic RAG1/RAG2 requires ciclosporin + steroids before HSCT to control autoreactive T-cell activation; complete RAG1 LOF = T-B-NK+ SCID",
            "WAS-MICROPLATELETS-PATHOGNOMONIC-SPLENECTOMY-CI: MPV <7 fL + thrombocytopenia = WAS until proven otherwise -- pathognomonic; eczema + thrombocytopenia + infections triad; splenectomy absolutely contraindicated (fatal OPSI); ICH risk 10-15%; HSCT curative with >90% survival at specialist centres",
            "DOCK8-MOLLUSCUM-HPV-PATHOGNOMONIC-STAT3-DDx: Extensive Molluscum contagiosum + HPV warts + severe eczema = DOCK8 deficiency; elevated IgE + low IgM; HSCT curative; STAT3-HIES (AD) is KEY DDx -- pneumatoceles + skeletal abnormalities + retained primary teeth distinguish STAT3; sequence both",
            "TNFRSF13B-CVID-GLILD-LYMPHOMA-IVIG-LIFELONG: CVID requires lifelong IVIG/SCIG (trough >8 g/L); GLILD (granulomatous lung disease) in 10-20% -- rituximab treatment; autoimmune cytopenias 20-30%; 5-fold lymphoma risk -- annual surveillance; no live vaccines",
            "LRBA-ABATACEPT-PATHOGNOMONIC-RESPONSE-CTLA4: Dramatic abatacept (CTLA-4 Ig) response is pathognomonic for LRBA deficiency -- AIHA resolves, IBD improves, GLILD stabilises; immune dysregulation (AIHA + IBD + interstitial lung disease) dominates over infections; CTLA-4 surface expression reduced to 5-25% of normal; HSCT curative",
        ],
    }


def breakdown():
    result = []
    for idx, g in enumerate(IMID_GENES):
        result.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "alias": g["alias"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "omim_gene": g["omim_gene"],
            "omim_disease": g["omim_disease"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "key_alerts": g["key_alerts"],
            "etiologies": g["etiologies"],
            "stats": g["stats"],
            "dx_delay_distribution": g["dx_delay_distribution"],
            "computed": {
                "n_patients": len(g["patients"]),
                "seed": SEED_BASE + idx,
            },
            "sample_patients": g["patients"][:10],
        })
    return result


def definitions():
    return {
        "concepts": {
            "Primary Immunodeficiency -- Classification, Pathophysiology and the PIDD Spectrum": (
                "Primary immunodeficiency diseases (PIDs; also primary immune deficiency disorders, PIDDs) "
                "are a heterogeneous group of >450 monogenic disorders causing innate or adaptive immune "
                "failure. The International Union of Immunological Societies (IUIS) classifies PIDs into "
                "10 major categories: "
                "(1) Combined immunodeficiencies (T and B cell defects -- SCID, combined ID): "
                "ADA-SCID, IL2RG-XSCID, RAG1/RAG2-SCID, DCLRE1C (Artemis), JAK3, IL7R deficiencies; "
                "T-cell dysfunction -> opportunistic infections (PJP, CMV, fungal); "
                "(2) Predominantly antibody deficiencies: "
                "BTK-XLA (absent B cells), CVID (hypogammaglobulinaemia, TNFRSF13B/LRBA), "
                "IgA deficiency (most common PID -- 1:300), HIGM syndromes (CD40/CD40L); "
                "bacterial infections from encapsulated organisms; "
                "(3) Diseases of immune dysregulation: "
                "LRBA, CTLA4 haploinsufficiency, FOXP3-IPEX, XIAP, NLRC4 -- "
                "combined deficiency + autoimmunity; "
                "(4) Congenital defects of phagocyte number, function or both: "
                "Chronic granulomatous disease (CYBB/NCF1/2), Severe congenital neutropaenia, LAD; "
                "(5) Defects in intrinsic and innate immunity: TLR, IRAK, NEMO pathway defects; "
                "(6) Autoinflammatory disorders: Periodic fever syndromes, NLRP3, MVK, TNFRSF1A; "
                "(7) Complement deficiencies: C1q, C3, C5-C9 (N. meningitidis risk); "
                "(8) Phenocopies: STAT3 GOF, RAC2 GOF, CARD11 GOF -- somatic or de novo; "
                "(9) Bone marrow failure syndromes with immunological features; "
                "(10) SCID and other well-defined immunodeficiency syndromes. "
                "INCIDENCE: Overall PID ~1:2000 live births (if all forms included); "
                "SCID ~1:40,000-75,000 (NBS programs detect ~1:58,000); "
                "XLA ~1:190,000 male births; WAS ~1:100,000 male births; CVID ~1:25,000. "
                "TEN WARNING SIGNS OF PID (Jeffrey Modell Foundation): "
                ">=4 new ear infections in 1 year; >=2 serious sinus infections per year; "
                ">=2 months on antibiotics with little effect; >=2 pneumonias in 1 year; "
                "failure to thrive in an infant; recurrent deep skin/organ abscesses; "
                "persistent thrush or skin fungal infections; need for IV antibiotics to clear infections; "
                ">=2 deep-seated infections (meningitis, osteomyelitis, septicaemia); "
                "family history of PID."
            ),
            "HSCT in Primary Immunodeficiency -- Indications, Timing and Conditioning": (
                "Haematopoietic stem cell transplantation (HSCT) is curative for many PIDs by "
                "replacing the defective immune system with a donor-derived immune system. "
                "INDICATIONS: "
                "CURATIVE HSCT indicated in: ADA-SCID (if gene therapy unavailable), "
                "IL2RG-XSCID, RAG1/2-SCID, all other SCID forms; "
                "WAS (best outcomes <5 years), DOCK8 deficiency, LRBA deficiency (severe/refractory); "
                "Chronic granulomatous disease, LAD (severe). "
                "TIMING -- CRITICAL PRINCIPLE: "
                "HSCT before infection gives dramatically better outcomes -- "
                "SCID: HSCT in first 3-6 months of life (pre-infection) -> >90% survival; "
                "SCID with active infection: ~60-70% survival. "
                "Newborn screening (TREC assay) enables pre-symptomatic SCID HSCT. "
                "DONOR HIERARCHY: "
                "(1) HLA-identical sibling (MSD) -- best outcomes, lowest GvHD; "
                "(2) Matched unrelated donor (MUD, 10/10 HLA match): increasingly good outcomes; "
                "(3) Haploidentical family donor (parent, 5/6 match): "
                "T-cell depleted or post-transplant cyclophosphamide approaches; "
                "(4) Cord blood: limited cell dose but low GvHD. "
                "CONDITIONING REGIMENS: "
                "Myeloablative conditioning (MAC): busulfan + cyclophosphamide/fludarabine -- "
                "full donor engraftment; risk of toxicity; "
                "Reduced-intensity conditioning (RIC): lower toxicity, more mixed chimaerism; "
                "No conditioning (SCID only): T-depleted MSD SCID -- some donor T-cell engraftment. "
                "SPECIAL CONSIDERATIONS: "
                "Irradiated + CMV-negative + leucodepleted blood products mandatory for all SCID; "
                "Live vaccines CI post-HSCT until >2 years post-transplant with confirmed immunity; "
                "GvHD prophylaxis: ciclosporin +/- methotrexate +/- MMF; "
                "Engraftment monitoring: chimaerism studies at weeks 4, 8, 12, 6 months, annually."
            ),
            "Immunoglobulin Replacement Therapy -- IVIG vs SCIG, Dosing and Monitoring": (
                "Immunoglobulin replacement is lifelong therapy for antibody deficiency "
                "(XLA, CVID, WAS, IgG subclass deficiency). "
                "PREPARATIONS: "
                "IVIG (intravenous immunoglobulin): "
                "Pooled IgG from >=1000 donors; half-life ~21 days; "
                "Dose: 400-600 mg/kg every 3-4 weeks (IV infusion over 2-4 hours); "
                "Higher doses (600-800 mg/kg) for chronic lung disease or refractory infections. "
                "SCIG (subcutaneous immunoglobulin): "
                "Same product administered subcutaneously; "
                "Weekly or bi-weekly self-administration at home (via infusion pump); "
                "More stable IgG troughs (no peak-and-trough cycles); "
                "Preferred by many patients -- home administration, no IV access needed. "
                "Facilitated SCIG (fSCIG, hyaluronidase-facilitated): monthly. "
                "TROUGH TARGETS: "
                "Minimum: IgG trough >8 g/L; "
                "With chronic lung disease/bronchiectasis: IgG trough >10-12 g/L; "
                "Measure trough immediately before each infusion. "
                "MONITORING: "
                "Trough IgG: before every infusion (IVIG) or monthly (SCIG); "
                "Infection frequency: number of significant infections per year -- "
                "target <1 significant sinopulmonary infection per year; "
                "Pulmonary function: annual spirometry for all established antibody deficiency; "
                "CT chest (HRCT): every 3-5 years to screen for bronchiectasis/GLILD; "
                "Vaccine responses: pneumococcal polysaccharide (PPV23) + conjugate (PCV13) "
                "and tetanus toxoid responses -- absent responses confirm diagnosis. "
                "ADVERSE REACTIONS: "
                "Systemic reactions (fever, chills, headache, myalgia): 5-15% -- "
                "slow infusion rate, pre-medicate with paracetamol/antihistamine; "
                "Aseptic meningitis: rare, usually resolves with slowing infusion; "
                "Thrombosis risk (high IgG doses, immobility): use low-IgA preparations if IgA-deficient "
                "with anti-IgA antibodies (anaphylaxis risk)."
            ),
            "Gene Therapy in Primary Immunodeficiency -- ADA-SCID, XSCID, WAS": (
                "Gene therapy has transformed the treatment of several PIDs, offering curative potential "
                "without the immune risks of allogeneic HSCT. "
                "ADA-SCID -- FIRST APPROVED GENE THERAPY: "
                "Strimvelis (GSK/Orchard Therapeutics): "
                "Ex vivo autologous HSC transduction with gamma-retroviral vector carrying ADA cDNA; "
                "EMA approved 2016 -- the first approved HSC gene therapy; "
                "Autologous cells used -> no GvHD risk; no need for HLA-matched donor; "
                "Procedure: collect patient HSCs -> transduce ex vivo -> reinfuse after mild conditioning; "
                "Outcomes: >90% immune reconstitution; available at San Raffaele Hospital, Milan; "
                "OTL-101 (Orchard): lentiviral vector -- regulatory approval pathway (FDA/EMA); "
                "PEG-ADA remains the bridge enzyme replacement while awaiting GT or HSCT. "
                "XSCID (IL2RG) -- GENE THERAPY EVOLUTION: "
                "Early trials (1999-2002, Paris/London): gamma-retroviral vectors -> insertional oncogenesis "
                "-> 5 cases of T-cell acute lymphoblastic leukaemia from LMO2 insertion; "
                "Lentiviral vectors: self-inactivating (SIN) LV -> dramatically reduced insertional risk; "
                "OTL-101 (Lentigen/Orchard): SIN-LV IL2RG -- FDA approved 2024; "
                "clinical trials show T, B, NK reconstitution; good safety profile to date. "
                "WAS -- OTL-103: "
                "Ex vivo autologous HSC lentiviral transduction with WAS cDNA; "
                "OTL-103 (Orchard): clinical trials showing platelet recovery + immune reconstitution; "
                "earlier retroviral WAS trials: some insertional mutagenesis events -> lentiviral switch. "
                "GENERAL GENE THERAPY PRINCIPLES: "
                "Autologous GT avoids GvHD (no donor T cells) -- major advantage over allogeneic HSCT; "
                "Mild conditioning (busulfan-only) for HSC engraftment space -- less toxic than MAC; "
                "Long-term follow-up for insertional oncogenesis mandatory (20+ years); "
                "Restricted to specialist centres with HSC collection and manufacturing capabilities."
            ),
            "Live Vaccines -- Absolute Contraindications in Primary Immunodeficiency": (
                "Live attenuated vaccines contain replication-competent organisms that are attenuated "
                "(weakened) to not cause disease in immunocompetent hosts -- but CAN cause disease "
                "in immunocompromised patients who cannot control the organism. "
                "LIVE VACCINES CI IN ALL CELLULAR IMMUNODEFICIENCY (T-cell defects, SCID, WAS): "
                "BCG (Bacille Calmette-Guerin): mycobacterial vaccine -- can cause disseminated BCG "
                "disease (BCGosis) in SCID -> fatal; given in many countries at birth -- "
                "neonatal SCID may receive BCG before diagnosis; urgent isoniazid + rifampicin "
                "prophylaxis needed if BCG given to a subsequently diagnosed SCID; "
                "OPV (oral poliovirus vaccine -- Sabin): live polio virus strains -> "
                "vaccine-derived poliovirus infection in immune-deficient patients; "
                "XLA: enteroviral susceptibility -> OPV can cause poliomyelitis + encephalitis; "
                "IPV (inactivated polio vaccine -- Salk) is the SAFE alternative; "
                "MMR (measles-mumps-rubella): live viral vaccine -> measles pneumonia, "
                "giant cell pneumonia, measles encephalitis in T-cell-deficient patients; "
                "Varicella (VZV): varicella-zoster virus -> progressive varicella, visceral VZV disease; "
                "Yellow fever vaccine: live flavivirus -> viscerotropic/neurotropic disease; "
                "Rotavirus: live attenuated -> chronic diarrhoea in SCID patients; "
                "Intranasal influenza (FluMist/Fluenz): live attenuated influenza virus; "
                "Oral typhoid (Ty21a): live S. typhi. "
                "HOUSEHOLD CONTACTS: "
                "OPV given to a household contact of an immunocompromised patient -> "
                "vaccine-derived poliovirus shed in stool -> contact transmission -> CI; "
                "recommend IPV for household contacts; "
                "BCG given to a sibling should prompt isolation until shedding stops. "
                "INACTIVATED VACCINES SAFE (may give reduced responses): "
                "IPV, DTaP, Hib, PCV, PPV23, meningococcal ACWY/B, hepatitis A/B, "
                "HPV (inactivated), inactivated influenza, Japanese encephalitis (inactivated); "
                "responses may be absent (SCID, XLA) but still recommended (partial benefit possible). "
                "MEDICAL ALERT: every PID patient should carry/wear documentation of live vaccine CI."
            ),
        },
        "pharmacological_distinctions": [
            "IVIG (pooled IgG 400-600 mg/kg IV q3-4w) -- lifelong replacement for XLA, CVID (TNFRSF13B, LRBA), WAS; target trough IgG >8 g/L (>10-12 g/L with bronchiectasis or GLILD); always check trough immediately before infusion; reduce infusion rate if systemic reactions occur",
            "SCIG (subcutaneous IgG, equivalent monthly dose divided weekly) -- home-administered alternative to IVIG; more stable trough levels; preferred by patients for convenience; facilitated SCIG (with hyaluronidase) allows monthly administration; same target trough as IVIG",
            "PEG-ADA (pegylated bovine adenosine deaminase; Adagen; weekly SC) -- enzyme replacement for ADA-SCID; bridge to HSCT or gene therapy; restores ADA enzyme activity; reduces dATP toxicity; does NOT fully reconstitute lymphocyte counts; do not stop TMP-SMX prophylaxis while on PEG-ADA",
            "Abatacept (CTLA-4 Ig fusion protein; Orencia; SC weekly or IV monthly) -- PATHOGNOMONIC treatment for LRBA deficiency and CTLA-4 haploinsufficiency; provides exogenous CTLA-4 function; rapidly reverses AIHA, IBD, and GLILD; must be combined with IVIG; not a substitute for HSCT in severe cases",
            "TMP-SMX (trimethoprim-sulfamethoxazole; co-trimoxazole; daily or 3x/week) -- Pneumocystis jirovecii pneumonia (PJP) prophylaxis for all T-cell deficient patients (SCID, WAS, DOCK8); continue until post-HSCT immune reconstitution confirmed (CD4 >200/uL); alternative: atovaquone or dapsone",
            "Ciclosporin (5-8 mg/kg/day, target trough 100-200 ng/mL) -- pre-HSCT immunosuppression for Omenn syndrome (RAG1 hypomorphic); suppresses oligoclonal autoreactive T cells; combined with prednisolone; allows skin disease control before HSCT; monitor renal function",
            "Rituximab (anti-CD20 375 mg/m2 IV x4 doses) -- treatment for EBV-driven lymphoproliferation, AIHA, and granulomatous disease in CVID (TNFRSF13B) and LRBA; combined with ciclosporin or steroids for GLILD; monitor for hypogammaglobulinaemia worsening post-rituximab",
            "Acyclovir/valacyclovir (prophylactic dosing 400 mg BD or weight-adjusted) -- HSV and VZV prophylaxis in DOCK8 deficiency; reduces HSV/VZV reactivation frequency; does NOT prevent HPV or Molluscum contagiosum (different virus mechanisms); continue until post-HSCT immune reconstitution",
            "Sirolimus (mTOR inhibitor, target trough 5-10 ng/mL) -- for lymphoproliferation and granulomatous disease in LRBA and CVID; alternative to ciclosporin for immune dysregulation; monitor pneumonitis risk; significant immunosuppression -- infection screening before starting",
            "Interferon-alpha (IFN-alpha SC; off-label) -- some benefit for extensive Molluscum contagiosum in DOCK8 deficiency; anti-viral activity; limited evidence; bridge while awaiting HSCT; not curative for DOCK8 immune defect",
        ],
        "key_standards": [
            "ESID/AAAAI Primary Immunodeficiency Guidelines: all patients with PID should be managed at or in consultation with a specialist PID centre; genetic diagnosis mandatory for all suspected monogenic PID; newborn screening (TREC + KREC assay) recommended for early SCID and XLA detection; annual clinical review minimum",
            "Newborn Screening for SCID (TREC Assay): T-cell receptor excision circles (TRECs) absent in all SCID types; implemented in all US states, UK, many EU countries; pre-symptomatic HSCT before 3 months gives >90% survival vs ~70% with symptomatic presentation; all TREC-low results require urgent lymphocyte subset flow cytometry",
            "IVIG Trough Monitoring Protocol: trough IgG measured immediately before every IVIG infusion; target >8 g/L minimum; >10-12 g/L if chronic lung disease; annual chest CT (HRCT) for bronchiectasis; annual spirometry; annual infection count; adjust dose to achieve target -- not weight-based alone",
            "XLA Enteroviral Surveillance: all XLA patients should have annual review for neurological symptoms; brain MRI if any cognitive change or headache; enteroviral PCR (stool + CSF) if suspected encephalitis; high-dose IVIG for enteroviral encephalitis (limited evidence but first-line); no antiviral approved for treatment",
            "WAS Transplant Decision Protocol: HSCT indication for all classical WAS (score 3-5); optimal timing before age 5 years; MSD or 10/10 MUD preferred; haploidentical transplant with post-transplant cyclophosphamide (PTCy) acceptable; splenectomy absolutely contraindicated pre-HSCT; ICH surveillance: platelet count maintained >20 x 10^9/L target",
            "DOCK8 HSCT Timing and Surveillance: HSCT indicated for all DOCK8 deficiency; perform before HPV-associated malignancy develops; annual gynaecological/dermatological examination for HPV dysplasia; EBV PCR quarterly; HPV vaccination before HPV exposure ideally; post-HSCT Molluscum/HPV resolution is dramatic and confirms successful engraftment",
            "CVID Complication Monitoring (TNFRSF13B/LRBA): annual HRCT chest for GLILD (ground-glass, nodules, hilar lymphadenopathy); annual spirometry; annual full blood count for autoimmune cytopenias; LDH + EBV PCR annually for lymphoma surveillance; colonoscopy if IBD symptoms (LRBA); biopsy of granulomatous lesions to exclude lymphoma before immunosuppression",
            "LRBA/CTLA-4 Abatacept Protocol: diagnosis of LRBA or CTLA-4 haploinsufficiency -> start abatacept (CTLA-4 Ig) SC 125 mg weekly or IV 10 mg/kg monthly; response expected within 4-8 weeks; concurrent IVIG replacement; monitor AIHA (DAT, Hb), IBD (faecal calprotectin), lung (spirometry); sequence LRBA and CTLA4 together in all abatacept-responsive immune dysregulation",
        ],
    }
