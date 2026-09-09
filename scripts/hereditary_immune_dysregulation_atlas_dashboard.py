#!/usr/bin/env python3
"""Hereditary-Immune-Dysregulation-Atlas — Complete 8-Gene Immune Dysregulation Atlas
(FAS · FASLG · CASP10 · CASP8 · FOXP3 · WAS · DOCK8 · STAT3-GOF).

FAS      (TNFRSF6/APO-1/CD95; 335 aa; ~38 kDa; 10q23.31; AD (heterozygous);
          ALPS Type Ia — Autoimmune Lymphoproliferative Syndrome, most common form ~70%;
          CHRONIC NON-MALIGNANT LYMPHADENOPATHY + SPLENOMEGALY + AUTOIMMUNE CYTOPENIAS TRIAD;
          DOUBLE-NEGATIVE T CELLS (CD3+CD4-CD8-TCRαβ+) >1.5% of lymphocytes PATHOGNOMONIC;
          SIROLIMUS (mTOR inhibitor) most effective — ACNS 2020 standard;
          seed SEED_BASE+0).
FASLG    (FasL/CD95L/TNFSF6; 281 aa; ~37 kDa; 1q24.3; AD rare;
          ALPS Type Ib — FasL (Fas Ligand) deficiency;
          Same ALPS phenotype as FAS Ia but FasL cannot signal via Fas;
          MARKEDLY ELEVATED sFASL (soluble FasL >200 pg/mL) BIOMARKER;
          seed SEED_BASE+1).
CASP10   (Caspase-10; 521 aa; ~59 kDa; 2q33.1; AD;
          ALPS Type IIa — caspase cascade failure downstream of FAS;
          Identical clinical ALPS phenotype; distinguished by genetic testing only;
          SOMATIC REVERSION mosaicism documented — false negatives in germline testing;
          seed SEED_BASE+2).
CASP8    (Caspase-8; 479 aa; ~55 kDa; 2q33.1; AR;
          ALPS Type IIb + COMBINED IMMUNODEFICIENCY — unique dual role:
          apoptosis AND T/NK/B-cell activation BOTH impaired;
          RECURRENT BACTERIAL + HERPESVIRAL INFECTIONS superimposed on ALPS — PATHOGNOMONIC;
          seed SEED_BASE+3).
FOXP3    (Forkhead Box P3; 431 aa; ~47 kDa; Xp11.23; XLR;
          IPEX — Immune dysregulation, Polyendocrinopathy, Enteropathy, X-linked;
          NEONATAL DIABETES + INTRACTABLE DIARRHOEA + ECZEMA TRIAD PATHOGNOMONIC IN BOYS;
          FATAL WITHOUT HSCT in severe cases; Tacrolimus/sirolimus bridge;
          seed SEED_BASE+4).
WAS      (Wiskott-Aldrich Syndrome protein/WASp; 502 aa; ~53 kDa; Xp11.22; XLR;
          Wiskott-Aldrich Syndrome (WAS) — classic triad;
          MICROTHROMBOCYTOPENIA + ECZEMA + IMMUNODEFICIENCY TRIAD PATHOGNOMONIC;
          SMALL PLATELETS (<10 fL) on blood film PATHOGNOMONIC — KEY DDx from ITP;
          HSCT CURATIVE for all three; autoimmunity + lymphoma risk in older patients;
          seed SEED_BASE+5).
DOCK8    (Dedicator of Cytokinesis 8; 2099 aa; ~238 kDa; 9p24.3; AR;
          DOCK8 Deficiency — Hyper-IgE Syndrome type 2 (HIES2);
          ECZEMA + RECURRENT CUTANEOUS HERPESVIRAL INFECTIONS (disseminated HSV, molluscum) + VERY HIGH IgE;
          HSV DISSEMINATION PATHOGNOMONIC — distinguishes DOCK8 from STAT3-GOF HIES;
          CD8+ T-cell lymphopenia progressive; lymphoma risk 10-15%;
          seed SEED_BASE+6).
STAT3    (Signal Transducer and Activator of Transcription 3; 770 aa; ~92 kDa; 17q21.2; AD GOF;
          STAT3 Gain-of-Function — multi-system immune dysregulation;
          LYMPHOPROLIFERATION + MULTI-ORGAN AUTOIMMUNITY + SHORT STATURE + EARLY-ONSET T1DM/AIHA;
          DISTINCT FROM STAT3-LOF (Hyper-IgE type 1 HIES1) — opposite immune phenotype;
          JAK INHIBITORS (ruxolitinib/tofacitinib) HIGHLY EFFECTIVE — DRAMATIC RESPONSE;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2358-2365).
"""

import random

SEED_BASE = 2358

IMMUNE_DYSREGULATION_GENES = [
    # -- FAS -- ALPS Type Ia -------------------------------------------------------
    {
        "gene": "FAS",
        "alt_name": (
            "FAS (FAS-335aa-10q23.31 / AD-Heterozygous -- ALPS-Ia -- "
            "MOST-COMMON-ALPS-~70pct -- "
            "CHRONIC-NON-MALIGNANT-LYMPHADENOPATHY+SPLENOMEGALY+AUTOIMMUNE-CYTOPENIAS-TRIAD -- "
            "DOUBLE-NEGATIVE-T-CELLS-CD3+CD4-CD8-TCRab+->1.5pct-PATHOGNOMONIC -- "
            "SIROLIMUS-mTOR-ACNS2020-STANDARD-FIRST-LINE)"
        ),
        "protein": (
            "FAS -- 10q23.31 AD -- FAS-335aa -- "
            "TNFRSF6-APO-1-CD95-38kDa-Type-I-Transmembrane-TNF-Receptor-Superfamily -- "
            "Three-Extracellular-Cysteine-Rich-Domains-CRD1-CRD2-CRD3-Ligand-Binding -- "
            "Intracellular-Death-Domain-DD-Recruits-FADD-Adaptor-Protein -- "
            "FADD-Recruits-Procaspase-8-DISC-Death-Inducing-Signalling-Complex -- "
            "DISC-Activates-Caspase-8-Caspase-3-Apoptosis-Intrinsic-Extrinsic-Amplification -- "
            "ALPS-Heterozygous-LOF-Haploinsufficiency-Dominant-Negative-50pct-Residual-FAS -- "
            "DNT-Cells-Accumulate-Cannot-Be-Eliminated-Lymphoproliferation-Result -- "
            "OMIM-Gene-134637-Disease-ALPS-Ia-601859"
        ),
        "locus": "10q23.31",
        "protein_size": "335 aa / 38 kDa",
        "inheritance": (
            "AD (autosomal dominant); heterozygous LOF; haploinsufficiency or dominant-negative; "
            "~70% of all ALPS cases; most common hereditary cause; "
            "penetrance incomplete — same family members with same variant show variable expression; "
            "somatic FAS mutations also occur (somatic ALPS)"
        ),
        "disease_category": "ALPS Type Ia — Autoimmune Lymphoproliferative Syndrome (most common, ~70% of ALPS)",
        "disease_pathway": (
            "FAS (CD95) is the death receptor on lymphocytes essential for activation-induced cell death (AICD). "
            "FAS haploinsufficiency or dominant-negative -> AICD failure -> lymphocyte accumulation -> "
            "lymphadenopathy and splenomegaly. "
            "Accumulated DNT cells (CD3+CD4-CD8-TCRαβ+) are pathognomonic — these are antigen-experienced "
            "T cells that escaped apoptosis and lost co-receptor expression. "
            "Autoimmunity: autoantibodies from uncleared autoreactive B cells -> haemolytic anaemia, "
            "ITP, neutropenia."
        ),
        "pathognomonic": (
            "CHRONIC NON-MALIGNANT LYMPHADENOPATHY (>6 months, symmetrical, cervical/axillary/inguinal) + "
            "SPLENOMEGALY + AUTOIMMUNE CYTOPENIAS (AIHA, ITP, autoimmune neutropenia) TRIAD. "
            "DOUBLE-NEGATIVE T CELLS (CD3+CD4-CD8-TCRαβ+) >1.5% of lymphocytes (or >2.5% of CD3+) PATHOGNOMONIC — "
            "required for ALPS diagnosis per ACNS 2010 criteria. "
            "In vitro FAS-mediated apoptosis assay: impaired apoptosis confirms ALPS. "
            "B12 elevated (lymphocyte production marker). Lymphoma risk 10-50x elevated."
        ),
        "treatment": (
            "SIROLIMUS (rapamycin, mTOR inhibitor): 2-8 mg/m2/day — ACNS 2020 first-line; "
            "controls lymphoproliferation and autoimmunity; dramatically reduces DNT cells within weeks. "
            "MMF (mycophenolate mofetil): alternative; good for autoimmunity. "
            "Corticosteroids: short-term for acute autoimmune flares only (not long-term). "
            "HSCT: considered for refractory disease or lymphoma development. "
            "Lymphoma surveillance: annual PET-CT/CT in high-risk; avoid unnecessary biopsies. "
            "Prophylactic pneumocystis: co-trimoxazole if on immunosuppression."
        ),
        "key_features": [
            "Most common ALPS cause (~70% of all ALPS cases)",
            "Chronic non-malignant lymphadenopathy + splenomegaly + autoimmune cytopenias TRIAD",
            "DNT cells (CD3+CD4-CD8-TCRαβ+) >1.5% PATHOGNOMONIC — required for ALPS diagnosis",
            "Elevated vitamin B12 (lymphocyte turnover marker)",
            "Lymphoma risk 10-50x elevated (Hodgkin and NHL)",
            "Sirolimus (mTOR inhibitor) ACNS 2020 first-line — dramatic lymphoproliferation control",
        ],
        "key_ddx": (
            "Infectious mononucleosis (EBV): acute onset; positive Paul-Bunnell/EBV serology; DNT cells absent. "
            "Lymphoma: clonal; PET-avid; progressive; biopsy required; DNT cells absent. "
            "ITP (isolated): no lymphadenopathy/splenomegaly; no DNT cells; FAS normal. "
            "SLE: autoantibodies (dsDNA, Sm); complement low; DNT cells rare in SLE. "
            "CVID: hypogammaglobulinaemia; no DNT cells; recurrent infections dominant."
        ),
        "autoimmunity_risk": "HIGH — haemolytic anaemia, ITP, autoimmune neutropenia most common",
        "lymphoma_risk": "HIGH — 10-50x elevated (Hodgkin + NHL); lifetime surveillance mandatory",
        "sirolimus_response": "Excellent — first-line ACNS 2020; controls lymphoproliferation and cytopenias",
        "hsct_required": "Reserved for refractory/lymphoma",
        "attack_trigger_common": "Infections (EBV, CMV), intercurrent illness, puberty, variable",
        "onset_age": "Usually childhood (median 2-3 years); neonatal to adult onset reported",
    },
    # -- FASLG -- ALPS Type Ib -------------------------------------------------------
    {
        "gene": "FASLG",
        "alt_name": (
            "FASLG (FASLG-281aa-1q24.3 / AD-Rare -- ALPS-Ib -- "
            "FAS-LIGAND-DEFICIENCY -- "
            "SAME-ALPS-PHENOTYPE-AS-FAS-Ia-DNT-CELLS-ELEVATED -- "
            "ELEVATED-SOLUBLE-FASL-sFASL->200pg-mL-BIOMARKER-PATHOGNOMONIC -- "
            "FAS-APOPTOSIS-ASSAY-NORMAL-But-FASL-FUNCTIONAL-ASSAY-IMPAIRED)"
        ),
        "protein": (
            "FASLG -- 1q24.3 AD-Rare -- FASLG-281aa -- "
            "FasL-CD95L-TNFSF6-37kDa-Type-II-Transmembrane-TNF-Superfamily -- "
            "Homotrimeric-Type-II-Transmembrane-Expressed-Activated-T-NK-Cells -- "
            "Metalloprotease-ADAM10-Sheds-Soluble-sFasL-Can-Induce-Apoptosis -- "
            "FasL-Binds-FAS-CD95-on-Target-Lymphocytes-DISC-Assembly-Apoptosis -- "
            "FASLG-Variants-Cannot-Signal-FAS-AICD-Failure-DNT-Accumulation -- "
            "sFasL-Elevated->200-pg-mL-Biomarker-Supports-ALPS-Diagnosis -- "
            "OMIM-Gene-134638-Disease-ALPS-Ib"
        ),
        "locus": "1q24.3",
        "protein_size": "281 aa / 37 kDa",
        "inheritance": (
            "AD (autosomal dominant); rare (<5% of ALPS); heterozygous FASLG LOF; "
            "FAS pathway at ligand level; same ALPS phenotype as ALPS-Ia (FAS); "
            "FAS apoptosis assay normal (receptor intact) but FASLG functional assay impaired"
        ),
        "disease_category": "ALPS Type Ib — FasL Deficiency (rare; same clinical phenotype as ALPS-Ia)",
        "disease_pathway": (
            "FasL (FasLigand) is expressed on activated T and NK cells and binds FAS on target lymphocytes. "
            "FASLG variants -> defective FasL cannot bind FAS -> AICD failure at ligand level -> "
            "lymphocyte accumulation identical to FAS haploinsufficiency. "
            "Key distinction: FAS receptor is intact, so standard FAS-mediated apoptosis assay using "
            "anti-FAS antibody (Jo2 or CH-11) appears NORMAL — FASLG-specific functional assay "
            "or genetic sequencing required to distinguish from ALPS-Ia."
        ),
        "pathognomonic": (
            "SAME CLINICAL ALPS TRIAD AS FAS-Ia: lymphadenopathy + splenomegaly + autoimmune cytopenias. "
            "DNT CELLS >1.5% of lymphocytes PATHOGNOMONIC (same threshold as ALPS-Ia). "
            "ELEVATED SOLUBLE FASL (sFASL >200 pg/mL) — biomarker; elevated due to receptor resistance feedback. "
            "FAS-MEDIATED APOPTOSIS ASSAY NORMAL (FAS intact) — KEY DDx from ALPS-Ia. "
            "Confirmed by FASLG sequencing or FasL functional assay."
        ),
        "treatment": (
            "SAME AS ALPS-Ia: sirolimus first-line (ACNS 2020). "
            "MMF: alternative immunosuppression. "
            "Corticosteroids: acute autoimmune flares only. "
            "HSCT: refractory disease. "
            "sFasL monitoring: biomarker of disease activity."
        ),
        "key_features": [
            "Rare ALPS (<5% of all ALPS)",
            "Identical clinical phenotype to ALPS-Ia (FAS mutation)",
            "FAS apoptosis assay NORMAL — KEY DDx from ALPS-Ia",
            "Elevated sFasL (>200 pg/mL) biomarker",
            "DNT cells elevated same as ALPS-Ia",
            "Genetic testing required to distinguish from ALPS-Ia/IIa",
        ],
        "key_ddx": (
            "ALPS-Ia (FAS): FAS apoptosis assay impaired; FAS sequencing positive; sFasL may be normal. "
            "ALPS-IIa (CASP10): all FAS-pathway assays normal; only genetic testing distinguishes. "
            "Lymphoma: clonal; no DNT cells; FAS/FASLG germline normal. "
        ),
        "autoimmunity_risk": "HIGH — same as ALPS-Ia",
        "lymphoma_risk": "HIGH — same as ALPS-Ia",
        "sirolimus_response": "Excellent — same mechanism as ALPS-Ia",
        "hsct_required": "Reserved for refractory/lymphoma",
        "attack_trigger_common": "Same as ALPS-Ia: infections, intercurrent illness",
        "onset_age": "Childhood; same onset pattern as ALPS-Ia",
    },
    # -- CASP10 -- ALPS Type IIa -------------------------------------------------------
    {
        "gene": "CASP10",
        "alt_name": (
            "CASP10 (CASP10-521aa-2q33.1 / AD -- ALPS-IIa -- "
            "CASPASE-10-DEFICIENCY-Downstream-FAS-Pathway -- "
            "IDENTICAL-ALPS-PHENOTYPE-GENETIC-TESTING-ONLY-DISTINCTION -- "
            "SOMATIC-REVERSION-MOSAICISM-FALSE-NEGATIVES-Germline-Testing -- "
            "ALL-FAS-PATHWAY-ASSAYS-NORMAL-Only-Molecular-Testing-Distinguishes)"
        ),
        "protein": (
            "CASP10 -- 2q33.1 AD -- CASP10-521aa -- "
            "Caspase-10-Apical-Initiator-Caspase-Death-Effector-Domain-DED-x2 -- "
            "Recruited-to-DISC-By-FADD-Alongside-Caspase-8-Redundant-Partial-Functions -- "
            "CASP10-Activates-Downstream-Executioner-Caspases-3-6-7-Apoptosis -- "
            "ALPS-IIa-Variants-LOF-CASP10-Cannot-Complete-Apoptosis-Cascade -- "
            "Somatic-Reversion-Mosaicism-Documented-CASP10-ALPS-False-Negative-Germline -- "
            "2q33.1-Locus-CASP10-CASP8-Adjacent-Same-Chromosomal-Region -- "
            "OMIM-Gene-601762-Disease-ALPS-IIa"
        ),
        "locus": "2q33.1",
        "protein_size": "521 aa / 59 kDa",
        "inheritance": (
            "AD (autosomal dominant); heterozygous LOF; downstream FAS signalling; "
            "FAS and FASLG intact; DISC forms normally but caspase cascade fails; "
            "somatic reversion mosaicism documented — germline testing may miss somatic cases"
        ),
        "disease_category": "ALPS Type IIa — Caspase-10 Deficiency (identical clinical ALPS phenotype)",
        "disease_pathway": (
            "Caspase-10 is recruited to the DISC alongside caspase-8 and participates in apoptosis initiation. "
            "CASP10 LOF -> DISC assembled normally but caspase cascade activation impaired -> "
            "AICD failure despite intact FAS and FASLG. "
            "DNT cell accumulation occurs identically to ALPS-Ia/Ib. "
            "All FAS-pathway functional assays (FAS apoptosis, sFasL) appear normal — "
            "only caspase-10 specific functional assay or molecular sequencing confirms."
        ),
        "pathognomonic": (
            "SAME CLINICAL ALPS TRIAD: lymphadenopathy + splenomegaly + autoimmune cytopenias. "
            "DNT CELLS >1.5% PATHOGNOMONIC (same threshold). "
            "ALL STANDARD FAS-PATHWAY ASSAYS NORMAL — KEY distinguishing feature from ALPS-Ia/Ib. "
            "SOMATIC MOSAICISM: some patients have CASP10 variants detected only in DNT cells, "
            "not in peripheral blood granulocytes — deep sequencing may be required."
        ),
        "treatment": (
            "SAME AS ALPS-Ia/Ib: sirolimus (ACNS 2020 first-line). "
            "MMF: alternative. HSCT: refractory/lymphoma. "
            "Somatic mosaicism: consider deep sequencing of DNT cells if germline negative but ALPS phenotype."
        ),
        "key_features": [
            "ALPS-IIa: identical clinical phenotype to ALPS-Ia/Ib",
            "All FAS-pathway assays NORMAL — only molecular testing distinguishes",
            "Somatic reversion mosaicism documented — germline testing may miss",
            "Same treatment as ALPS-Ia: sirolimus first-line",
            "CASP10/CASP8 at same chromosomal locus 2q33.1",
            "Rare — requires NGS panel including CASP10 for ALPS diagnosis",
        ],
        "key_ddx": (
            "ALPS-Ia (FAS): FAS apoptosis assay impaired; FAS sequencing positive. "
            "ALPS-Ib (FASLG): sFasL elevated; FasL functional assay impaired. "
            "ALPS-IIb (CASP8): AR; COMBINED immunodeficiency + ALPS; recurrent infections. "
            "Somatic ALPS: no germline variant in any gene; somatic FAS/CASP10 variants in DNT cells."
        ),
        "autoimmunity_risk": "HIGH — same as ALPS-Ia",
        "lymphoma_risk": "HIGH — same as ALPS-Ia",
        "sirolimus_response": "Excellent — same mechanism",
        "hsct_required": "Reserved for refractory/lymphoma",
        "attack_trigger_common": "Infections, intercurrent illness",
        "onset_age": "Childhood; same onset pattern",
    },
    # -- CASP8 -- ALPS Type IIb + Combined Immunodeficiency -------------------------------------------------------
    {
        "gene": "CASP8",
        "alt_name": (
            "CASP8 (CASP8-479aa-2q33.1 / AR -- ALPS-IIb-PLUS-COMBINED-IMMUNODEFICIENCY -- "
            "UNIQUE-DUAL-ROLE-Apoptosis-AND-T-NK-B-Cell-Activation-BOTH-Impaired -- "
            "RECURRENT-BACTERIAL+HERPESVIRAL-INFECTIONS-SUPERIMPOSED-ON-ALPS-PATHOGNOMONIC -- "
            "AR-NOT-AD-Biallelic-Required-Severe-Phenotype)"
        ),
        "protein": (
            "CASP8 -- 2q33.1 AR -- CASP8-479aa -- "
            "Caspase-8-55kDa-Apical-Initiator-Death-Effector-Domain-DED-x2-Caspase-Domain -- "
            "DISC-Component-FAS-TRAIL-Receptor-Pathways-Extrinsic-Apoptosis -- "
            "DUAL-ROLE-UNIQUE-Caspase-8-Also-Required-Antigen-Receptor-Induced-Proliferation -- "
            "Naive-T-Cells-NK-Cells-Require-Caspase-8-For-Initial-Activation-Cleavage-NFkB-Step -- "
            "CASP8-Biallelic-LOF-Apoptosis-Fail-AND-T-NK-B-Activation-Fail-Combined-Defect -- "
            "AR-Inheritance-Only-Biallelic-Complete-LOF-Causes-Disease-Unlike-CASP10-AD -- "
            "OMIM-Gene-601763-Disease-ALPS-IIb-607271"
        ),
        "locus": "2q33.1",
        "protein_size": "479 aa / 55 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic LOF required; UNLIKE CASP10 which is AD; "
            "rare — very few families described worldwide; "
            "heterozygous CASP8 variants do NOT cause ALPS (unlike heterozygous FAS/CASP10)"
        ),
        "disease_category": "ALPS Type IIb + Combined Immunodeficiency — Caspase-8 Biallelic Deficiency",
        "disease_pathway": (
            "Caspase-8 has TWO essential immune functions: "
            "(1) Apoptosis: DISC component — cleaves and activates caspase-3/7 for lymphocyte death. "
            "(2) Lymphocyte activation: required for NFκB activation downstream of TCR/BCR/NK receptor — "
            "naive T cells, NK cells, and B cells cannot initiate activation without caspase-8. "
            "Biallelic CASP8 LOF -> AICD failure (ALPS phenotype: DNT cells, lymphoproliferation) "
            "AND activation failure (combined immunodeficiency: recurrent bacterial and herpesviral infections). "
            "This dual phenotype (lymphoproliferation + immunodeficiency together) is PATHOGNOMONIC of CASP8."
        ),
        "pathognomonic": (
            "ALPS TRIAD (lymphadenopathy + splenomegaly + autoimmune cytopenias) + "
            "COMBINED IMMUNODEFICIENCY (recurrent bacterial infections + disseminated herpesviral infections) "
            "COMBINATION IS PATHOGNOMONIC — no other ALPS type shows significant infection susceptibility. "
            "DNT CELLS elevated (ALPS feature). "
            "T-CELL PROLIFERATION ASSAY IMPAIRED (immunodeficiency feature). "
            "HERPESVIRUS SUSCEPTIBILITY (HSV, EBV, VZV) disproportionate — treat aggressively."
        ),
        "treatment": (
            "SIROLIMUS: controls lymphoproliferation component. "
            "IMMUNOGLOBULIN REPLACEMENT (IVIG/SCIG): for recurrent bacterial infections. "
            "PROPHYLACTIC ANTIVIRALS (aciclovir): for herpesviral infections — mandatory. "
            "Avoid live vaccines (combined immunodeficiency). "
            "HSCT: considered early given combined severity. "
            "Co-trimoxazole: PCP prophylaxis."
        ),
        "key_features": [
            "UNIQUE: ALPS + combined immunodeficiency simultaneously PATHOGNOMONIC",
            "AR (biallelic) — NOT AD like other ALPS genes",
            "Dual role: caspase-8 needed for both apoptosis AND lymphocyte activation",
            "Recurrent bacterial + herpesviral infections superimposed on ALPS lymphoproliferation",
            "T-cell proliferation assay impaired (unique among ALPS types)",
            "IVIG + antivirals mandatory alongside sirolimus",
        ],
        "key_ddx": (
            "ALPS-Ia/Ib/IIa: AD; NO significant infection susceptibility; activation assays normal. "
            "SCID: no lymphoproliferation/DNT cells; completely absent T cells. "
            "LRBA deficiency: ALPS-like but hypogammaglobulinaemia + enteropathy dominant; LRBA variant. "
            "IPEX (FOXP3): X-linked; enteropathy + T1DM + eczema; TREG absent."
        ),
        "autoimmunity_risk": "HIGH — ALPS component",
        "lymphoma_risk": "HIGH — same as other ALPS types",
        "sirolimus_response": "Partial — controls lymphoproliferation but not immunodeficiency",
        "hsct_required": "Yes — early consideration given combined severity",
        "attack_trigger_common": "Infections trigger both immune activation and autoimmune flares",
        "onset_age": "Early childhood; severe combined phenotype usually apparent in first years",
    },
    # -- FOXP3 -- IPEX -------------------------------------------------------
    {
        "gene": "FOXP3",
        "alt_name": (
            "FOXP3 (FOXP3-431aa-Xp11.23 / XLR -- IPEX -- "
            "NEONATAL-DIABETES+INTRACTABLE-ENTEROPATHY+ECZEMA-TRIAD-BOYS-PATHOGNOMONIC -- "
            "REGULATORY-T-CELLS-ABSENT-FOXP3-MASTER-REGULATOR-TREG -- "
            "FATAL-WITHOUT-HSCT-Tacrolimus-Sirolimus-Bridge -- "
            "TREG-ABSENT-FLOW-CD4+CD25+FoxP3+-TEST-MANDATORY)"
        ),
        "protein": (
            "FOXP3 -- Xp11.23 XLR -- FOXP3-431aa -- "
            "Forkhead-Box-P3-47kDa-Scurfin-Zinc-Finger-Leucine-Zipper-Forkhead-TF -- "
            "Master-Transcription-Factor-Regulatory-T-Cells-TREG-Differentiation-Maintenance -- "
            "FOXP3-Drives-CD25-CTLA4-IL-2Ralpha-Expression-In-TREG-Cells -- "
            "TREG-Suppress-Autoreactive-Effector-T-Cells-Peripheral-Tolerance-Maintenance -- "
            "Scurfy-Mouse-Foxp3-Null-Lethal-Multi-Organ-Autoimmunity-Model -- "
            "FOXP3-LOF-No-TREG-Cells-Autoreactive-T-Cells-Unrestrained -- "
            "Neonatal-Diabetes-Enteropathy-Eczema-Thyroiditis-Autoimmune-Multi-Organ -- "
            "OMIM-Gene-300292-Disease-IPEX-304790"
        ),
        "locus": "Xp11.23",
        "protein_size": "431 aa / 47 kDa",
        "inheritance": (
            "XLR (X-linked recessive); boys affected; carrier females usually asymptomatic; "
            "new mutations frequent; de novo mutations account for significant proportion; "
            "IPEX syndrome exclusively males in classic form; IPEX-like syndromes (IPEX-2) may affect females"
        ),
        "disease_category": "IPEX — Immune Dysregulation, Polyendocrinopathy, Enteropathy, X-linked",
        "disease_pathway": (
            "FOXP3 is the master transcription factor for regulatory T cells (TREG). "
            "FOXP3 LOF -> TREG absent or non-functional -> autoreactive effector T cells (Th1/Th2) "
            "cannot be suppressed -> multi-organ autoimmunity from birth. "
            "Target organs: pancreatic beta cells (neonatal T1DM), gut enterocytes "
            "(intractable villous-atrophy enteropathy), skin (severe eczema), thyroid, blood cells. "
            "IL-2 and CTLA-4 signalling downstream of FOXP3 also disrupted."
        ),
        "pathognomonic": (
            "NEONATAL T1DM (often first presentation, within weeks of birth) + "
            "INTRACTABLE SECRETORY DIARRHOEA (villous atrophy, malabsorption, failure to thrive) + "
            "SEVERE ECZEMA — TRIAD IN BOYS = IPEX PATHOGNOMONIC. "
            "TREG ABSENT: CD4+CD25+FoxP3+ cells <1% of CD4 T cells on flow cytometry. "
            "Other autoimmunity: thyroiditis, haemolytic anaemia, ITP, nephritis. "
            "FATAL WITHOUT TREATMENT in severe neonatal presentation — median survival without HSCT <1 year in severe."
        ),
        "treatment": (
            "HSCT CURATIVE: only definitive therapy; best outcomes with HLA-matched sibling or MUD <2 years age. "
            "BRIDGE TO HSCT: tacrolimus (calcineurin inhibitor) or sirolimus (mTOR inhibitor) — reduce T-cell activation. "
            "Neonatal T1DM: insulin mandatory from diagnosis. "
            "Enteropathy: elemental formula or PN; no standard diet tolerated. "
            "Eczema: topical corticosteroids/tacrolimus. "
            "GENE THERAPY: investigational FOXP3 gene therapy trials underway (2024+). "
            "Milder FOXP3 variants (partial function): may tolerate long-term sirolimus/tacrolimus without HSCT."
        ),
        "key_features": [
            "X-linked recessive — only boys affected classically",
            "NEONATAL T1DM (first presentation within weeks) + intractable enteropathy + severe eczema TRIAD",
            "TREG absent (CD4+CD25+FoxP3+ <1%) on flow cytometry — confirmatory test",
            "FATAL WITHOUT HSCT in severe neonatal cases; HSCT curative",
            "Tacrolimus/sirolimus bridge to HSCT",
            "Multi-organ autoimmunity: thyroid, blood, kidney",
        ],
        "key_ddx": (
            "Neonatal diabetes (isolated): no enteropathy/eczema; FOXP3 germline normal; NDM genes (KCNJ11/ABCC8). "
            "CMPA/cow's milk allergy: eosinophilic; resolves with elimination; no T1DM; FOXP3 normal. "
            "CASP8-ALPS: lymphoproliferation + infections; no neonatal diabetes; AR CASP8. "
            "WAS: microthrombocytopenia + eczema + infections; NO neonatal T1DM; TREG present."
        ),
        "autoimmunity_risk": "SEVERE — multi-organ from birth (T1DM, enteropathy, eczema, thyroid, blood)",
        "lymphoma_risk": "Low — TREG deficiency does not directly predispose to lymphoma",
        "sirolimus_response": "Partial bridge only — HSCT is curative",
        "hsct_required": "Yes — curative and recommended in severe disease",
        "attack_trigger_common": "Intrinsic autoimmunity from birth; infections worsen",
        "onset_age": "Neonatal to infancy; T1DM within weeks of birth in severe form",
    },
    # -- WAS -- Wiskott-Aldrich Syndrome -------------------------------------------------------
    {
        "gene": "WAS",
        "alt_name": (
            "WAS (WAS-502aa-Xp11.22 / XLR -- WISKOTT-ALDRICH-SYNDROME -- "
            "MICROTHROMBOCYTOPENIA+ECZEMA+IMMUNODEFICIENCY-CLASSIC-TRIAD-BOYS -- "
            "SMALL-PLATELETS-<10fL-Pathognomonic-KEY-DDx-From-ITP -- "
            "HSCT-CURATIVE-All-Three-Components -- "
            "AUTOIMMUNITY+LYMPHOMA-Risk-Older-Patients)"
        ),
        "protein": (
            "WAS -- Xp11.22 XLR -- WAS-502aa -- "
            "WASp-Wiskott-Aldrich-Syndrome-Protein-53kDa-Cytoplasmic-Signal-Transducer -- "
            "WH1-Domain-WASP-Interacting-Protein-GBD-CDC42-Rho-GTPase-Binding -- "
            "VCA-Verprolin-Cofilin-Acidic-Domain-ARP2-3-Complex-Actin-Polymerisation -- "
            "WASp-Links-CDC42-To-ARP2-3-Complex-Actin-Cytoskeleton-Branching -- "
            "T-Cell-IS-Immunological-Synapse-Formation-Requires-WASp-Actin-Remodelling -- "
            "Platelet-Size-Regulation-WASp-Required-Demarcation-Membrane-Normal-Platelet -- "
            "WAS-LOF-Small-Dysfunctional-Platelets-AND-T-NK-B-Cell-Dysfunction -- "
            "OMIM-Gene-300392-Disease-WAS-301000"
        ),
        "locus": "Xp11.22",
        "protein_size": "502 aa / 53 kDa",
        "inheritance": (
            "XLR (X-linked recessive); boys affected; carrier females usually asymptomatic with skewed X-inactivation; "
            "WAS score 1-5 (Zhu scoring): 1-2 = XLT (X-linked thrombocytopenia, mild); "
            "3-5 = classic WAS (full triad); GOF WAS variants cause XLN (X-linked neutropenia)"
        ),
        "disease_category": "Wiskott-Aldrich Syndrome (WAS) — microthrombocytopenia + eczema + combined immunodeficiency",
        "disease_pathway": (
            "WASp (WAS protein) links CDC42 GTPase signalling to ARP2/3 complex-mediated actin polymerisation. "
            "WASp is essential for: (1) actin remodelling at the immunological synapse (T-cell activation), "
            "(2) NK cell cytotoxic lysis, (3) platelet demarcation membrane formation (platelet sizing), "
            "(4) B-cell receptor signalling. "
            "WAS LOF -> small dysfunctional platelets (microthrombocytopenia) + "
            "T/NK/B cell dysfunction (combined immunodeficiency) + eczema (IgE-mediated, T-cell dysregulation). "
            "Older patients: autoimmunity (WASp also needed for TREG function) and lymphoma (NK surveillance)."
        ),
        "pathognomonic": (
            "MICROTHROMBOCYTOPENIA (platelets <70,000/μL) + ECZEMA (often severe, infected) + "
            "COMBINED IMMUNODEFICIENCY (recurrent bacterial + opportunistic infections) TRIAD IN BOYS. "
            "SMALL PLATELETS: MPV (mean platelet volume) <7 fL (often <5 fL) PATHOGNOMONIC — "
            "critical DDx from ITP where platelets are LARGE. "
            "WASp expression by flow cytometry on monocytes/T cells (WASp antibody staining): "
            "absent or markedly reduced in classic WAS. "
            "IgM low (early in disease); IgA and IgE elevated."
        ),
        "treatment": (
            "HSCT CURATIVE: all three triad components corrected; best outcomes <5 years age, HLA-matched donor. "
            "GENE THERAPY: Lentiviral WAS gene therapy (OTL-103) — EMA approved 2022 for WAS (Genethon); "
            "curative alternative if no HLA-matched donor. "
            "Pre-HSCT: IVIG (infection prophylaxis), prophylactic antibiotics (co-trimoxazole). "
            "Eczema: topical corticosteroids/tacrolimus; treat superinfections aggressively. "
            "Bleeding: avoid aspirin/NSAIDs (platelet dysfunction); platelet transfusion for severe bleeds. "
            "Splenectomy: NOT recommended (increases infection risk despite transiently improving platelet count). "
            "Autoimmunity surveillance in older patients with partial disease (XLT)."
        ),
        "key_features": [
            "X-linked recessive — only boys affected",
            "Microthrombocytopenia + eczema + combined immunodeficiency CLASSIC TRIAD",
            "SMALL PLATELETS (MPV <7 fL) PATHOGNOMONIC — KEY DDx from ITP (where MPV is HIGH)",
            "HSCT curative — corrects all three triad components",
            "Gene therapy (OTL-103) EMA 2022 — alternative if no matched donor",
            "Autoimmunity (haemolytic anaemia, nephritis) + lymphoma (10-15%) risk in older patients",
        ],
        "key_ddx": (
            "ITP: LARGE platelets (MPV high); no eczema; no immunodeficiency; WASp normal. "
            "IPEX (FOXP3): eczema + T1DM + enteropathy; NO thrombocytopenia; TREG absent; X-linked. "
            "DOCK8 deficiency: eczema + infections; NO thrombocytopenia; IgE very high; AR. "
            "DiGeorge syndrome: T-cell deficiency; calcium low; heart defects; no microthrombocytopenia."
        ),
        "autoimmunity_risk": "MODERATE-HIGH in older patients (haemolytic anaemia, nephritis, inflammatory bowel)",
        "lymphoma_risk": "HIGH (10-15%) — predominantly NHL; NK surveillance failure",
        "sirolimus_response": "Partial — for autoimmunity; not definitive",
        "hsct_required": "Yes — curative; gene therapy (OTL-103) alternative",
        "attack_trigger_common": "Infections trigger bleeding and immune activation",
        "onset_age": "Neonatal thrombocytopenia; eczema and infections first months of life",
    },
    # -- DOCK8 -- DOCK8 Deficiency / HIES2 -------------------------------------------------------
    {
        "gene": "DOCK8",
        "alt_name": (
            "DOCK8 (DOCK8-2099aa-9p24.3 / AR -- DOCK8-DEFICIENCY-HIES2 -- "
            "ECZEMA+RECURRENT-CUTANEOUS-HERPESVIRAL-INFECTIONS+VERY-HIGH-IgE -- "
            "DISSEMINATED-HSV-MOLLUSCUM-PATHOGNOMONIC-Distinguishes-From-STAT3-HIES -- "
            "CD8-T-CELL-LYMPHOPENIA-Progressive-LYMPHOMA-10-15pct -- "
            "HSCT-CURATIVE-NK-CD8-Restored)"
        ),
        "protein": (
            "DOCK8 -- 9p24.3 AR -- DOCK8-2099aa -- "
            "Dedicator-of-Cytokinesis-8-238kDa-DOCK-Superfamily-DHR1-DHR2-Domains -- "
            "DHR2-Domain-GEF-Activity-Activates-CDC42-Rac1-Rho-GTPases -- "
            "DOCK8-CDC42-ARP2-3-Actin-Polymerisation-In-CD8-T-Cells-NK-Cells -- "
            "Required-CD8-T-Cell-Long-Lived-Memory-Formation-Survival-In-Tissue -- "
            "Required-NK-Cell-Immune-Synapse-Cytotoxic-Granule-Polarisation -- "
            "DOCK8-LOF-CD8-T-Cells-Cannot-Survive-In-Peripheral-Tissues-Progress-Lymphopenia -- "
            "Eczema-TH2-Skewing-B-Cell-IgE-Class-Switch-Uncontrolled-IgE-Elevation -- "
            "OMIM-Gene-611432-Disease-DOCK8-Deficiency-243700"
        ),
        "locus": "9p24.3",
        "protein_size": "2099 aa / 238 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic LOF; large deletions common — MLPA recommended; "
            "consanguineous families enriched; sporadic in non-consanguineous; "
            "both DOCK8 alleles must be affected for disease"
        ),
        "disease_category": "DOCK8 Deficiency — Hyper-IgE Syndrome type 2 (HIES2); combined immunodeficiency",
        "disease_pathway": (
            "DOCK8 is a guanine nucleotide exchange factor (GEF) for CDC42 in lymphocytes. "
            "DOCK8 activates CDC42 -> actin cytoskeleton remodelling -> immunological synapse formation. "
            "DOCK8 specifically required for CD8+ T-cell and NK-cell survival in peripheral tissues "
            "('migration through tight spaces' — interstitial CD8 T cells cannot survive without DOCK8-actin remodelling). "
            "Progressive CD8 T-cell lymphopenia -> viral skin infection susceptibility (HSV dissemination, molluscum). "
            "TH2 skewing + uncontrolled IgE class switch -> very high IgE + eczema. "
            "NK cell cytotoxic failure -> EBV/viral lymphoma surveillance loss."
        ),
        "pathognomonic": (
            "ECZEMA (severe, infected) + RECURRENT CUTANEOUS HERPESVIRAL INFECTIONS "
            "(disseminated HSV, extensive molluscum contagiosum, recurrent HSV keratitis) + VERY HIGH IgE (often >2000 IU/mL). "
            "DISSEMINATED HSV / MOLLUSCUM CONTAGIOSUM PATHOGNOMONIC — KEY DDx from STAT3-HIES "
            "(STAT3-HIES has pneumatoceles/skeletal features; DOCK8 has herpesviral skin dissemination). "
            "CD8+ T-CELL LYMPHOPENIA: progressive decline over years (hallmark). "
            "NK CELLS reduced. "
            "LYMPHOMA RISK 10-15% (predominantly EBV-associated B-cell lymphoma, T-cell lymphoma)."
        ),
        "treatment": (
            "HSCT CURATIVE: corrects CD8 T-cell defect and NK cell function; "
            "reduces lymphoma risk; eczema may persist post-HSCT. "
            "PRE-HSCT: prophylactic aciclovir (herpesviral prevention MANDATORY); "
            "co-trimoxazole (PCP prevention); IVIG for bacterial infections. "
            "Eczema: aggressive topical management. "
            "Avoid live vaccines. "
            "Annual surveillance: full blood count, EBV PCR, imaging for lymphoma. "
            "MLPA for large DOCK8 deletions if sequencing negative but phenotype typical."
        ),
        "key_features": [
            "AR — biallelic LOF; large deletions common (MLPA mandatory if sequencing negative)",
            "Eczema + disseminated HSV/molluscum + very high IgE TRIAD",
            "DISSEMINATED HSV/MOLLUSCUM PATHOGNOMONIC — KEY DDx from STAT3-GOF HIES",
            "Progressive CD8+ T-cell lymphopenia (hallmark over years)",
            "Lymphoma risk 10-15% (EBV-associated; NK surveillance failure)",
            "HSCT curative; prophylactic aciclovir MANDATORY pre-HSCT",
        ],
        "key_ddx": (
            "STAT3-LOF (HIES1): AD; pneumatoceles (lung cysts); coarse facies; retained primary teeth; scoliosis; NO herpesviral dissemination; high IgE. "
            "STAT3-GOF: lymphoproliferation; short stature; T1DM; JAK inhibitors effective; DIFFERENT from DOCK8. "
            "WAS: microthrombocytopenia; NO CD8 lymphopenia; herpesviral less typical; X-linked. "
            "Atopic eczema (AD): no systemic immunodeficiency; normal CD8; IgE usually <2000; no herpesviral dissemination."
        ),
        "autoimmunity_risk": "LOW — TH2 skewing but not classical ALPS autoimmunity",
        "lymphoma_risk": "HIGH (10-15%) — EBV-associated B-cell + T-cell lymphoma; NK surveillance failure",
        "sirolimus_response": "Not primary treatment",
        "hsct_required": "Yes — curative recommendation",
        "attack_trigger_common": "Viral infections (HSV, VZV, EBV, molluscum) — primary triggers",
        "onset_age": "Infancy; eczema and recurrent infections from first year of life",
    },
    # -- STAT3-GOF -- STAT3 Gain-of-Function -------------------------------------------------------
    {
        "gene": "STAT3",
        "alt_name": (
            "STAT3 (STAT3-770aa-17q21.2 / AD-GOF -- STAT3-GAIN-OF-FUNCTION -- "
            "LYMPHOPROLIFERATION+MULTI-ORGAN-AUTOIMMUNITY+SHORT-STATURE+EARLY-T1DM-AIHA -- "
            "DISTINCT-FROM-STAT3-LOF-HIES1-OPPOSITE-PHENOTYPE -- "
            "JAK-INHIBITORS-Ruxolitinib-Tofacitinib-HIGHLY-EFFECTIVE-DRAMATIC-RESPONSE -- "
            "STAT3-GOF-Constitutive-JAK-STAT3-TREG-Suppression)"
        ),
        "protein": (
            "STAT3 -- 17q21.2 AD-GOF -- STAT3-770aa -- "
            "Signal-Transducer-Activator-Transcription-3-92kDa-SH2-Coiled-Coil-DNA-Binding -- "
            "Downstream-Cytokine-Receptors-JAK1-JAK2-TYK2-Phosphorylation-Y705-Activation -- "
            "STAT3-Drives-IL-6-IL-10-IL-21-IL-23-Th17-Response-Acute-Phase-Genes -- "
            "STAT3-Also-Induced-By-Growth-Hormone-EGF-Insulin-Multiple-Non-Immune-Pathways -- "
            "GOF-Variants-SH2-DNA-Binding-Constitutive-STAT3-Phosphorylation-Without-Ligand -- "
            "Constitutive-STAT3-Suppresses-TREG-Differentiation-Uncontrolled-Effector-T-Cells -- "
            "STAT3-GOF-Opposite-To-STAT3-LOF-HIES1-Which-Has-INCREASED-TREG-No-Lymphoproliferation -- "
            "OMIM-Gene-102582-Disease-STAT3-GOF-615952"
        ),
        "locus": "17q21.2",
        "protein_size": "770 aa / 92 kDa",
        "inheritance": (
            "AD (autosomal dominant) GOF; heterozygous gain-of-function; de novo mutations frequent; "
            "variants cluster in SH2 and DNA-binding domains; DISTINCT from STAT3 LOF (which causes HIES1); "
            "same gene, opposite immune phenotype — always clarify GOF vs LOF"
        ),
        "disease_category": "STAT3 Gain-of-Function — multi-system immune dysregulation with lymphoproliferation",
        "disease_pathway": (
            "Normal: STAT3 is transiently phosphorylated at Y705 by JAKs downstream of cytokine receptors. "
            "STAT3 GOF -> constitutive pY705-STAT3 without cytokine stimulation -> "
            "constitutive IL-10, IL-6, BCL2, MCL1 expression -> lymphocyte survival and proliferation. "
            "Key pathomechanism: constitutive STAT3 SUPPRESSES FOXP3 transcription -> TREG reduced -> "
            "autoreactive effector T cells unopposed -> multi-organ autoimmunity. "
            "Growth axis: STAT3 required for GH receptor signalling -> constitutive STAT3 paradoxically "
            "impairs GH response -> growth failure (IGF-1 low). "
            "Insulin signalling disruption: early-onset T1DM-like diabetes."
        ),
        "pathognomonic": (
            "LYMPHOPROLIFERATION (lymphadenopathy, splenomegaly) + "
            "MULTI-ORGAN AUTOIMMUNITY (T1DM, AIHA, ITP, thyroiditis, IBD-like enteropathy, glomerulonephritis) + "
            "SHORT STATURE (GH resistance, low IGF-1) = STAT3-GOF TRIAD. "
            "EARLY-ONSET T1DM (often before age 5 years, not neonatal unlike FOXP3) PATHOGNOMONIC FEATURE. "
            "TREG REDUCED on flow cytometry (CD4+CD25+FoxP3+ low). "
            "ELEVATED IFN-γ, IL-6, CXCL10 (inflammation markers). "
            "JAK INHIBITOR RESPONSE: dramatic improvement in autoimmunity and lymphoproliferation within weeks — "
            "confirms diagnosis if uncertain."
        ),
        "treatment": (
            "JAK INHIBITORS: ruxolitinib (JAK1/2 inhibitor) or tofacitinib (JAK1/3 inhibitor) — FIRST-LINE; "
            "DRAMATIC reduction in lymphoproliferation, autoimmunity, and cytokine storm within weeks. "
            "SIROLIMUS: adjunct for lymphoproliferation control. "
            "Diabetes: insulin; autoimmunity: organ-specific treatments. "
            "HSCT: considered for refractory cases or post-transformation. "
            "GH therapy: LOW BENEFIT (GH resistance from constitutive STAT3); manage growth expectations. "
            "AVOID long-term high-dose steroids (infections + growth suppression). "
            "Lymphoma surveillance: annual imaging."
        ),
        "key_features": [
            "AD GOF — heterozygous; DISTINCT from STAT3-LOF (HIES1, which has NO lymphoproliferation)",
            "Lymphoproliferation + multi-organ autoimmunity + short stature TRIAD",
            "Early-onset T1DM (before age 5) PATHOGNOMONIC feature",
            "TREG reduced (STAT3-GOF constitutively suppresses FOXP3)",
            "JAK inhibitors (ruxolitinib/tofacitinib) HIGHLY EFFECTIVE — dramatic response confirms diagnosis",
            "Lymphoma risk — surveillance mandatory",
        ],
        "key_ddx": (
            "STAT3-LOF (HIES1 Job syndrome): AD LOF; coarse facies; pneumatoceles; eczema; high IgE; NO lymphoproliferation; NO T1DM; OPPOSITE phenotype. "
            "DOCK8 (HIES2): AR; herpesviral skin infections; NO short stature; NO T1DM; CD8 lymphopenia progressive. "
            "FOXP3 (IPEX): XLR; neonatal diabetes; intractable enteropathy; TREG absent; NO lymphoproliferation. "
            "ALPS (FAS): DNT cells pathognomonic; normal TREG; no T1DM; no short stature."
        ),
        "autoimmunity_risk": "SEVERE — multi-organ from early childhood (T1DM, AIHA, ITP, thyroid, gut, kidney)",
        "lymphoma_risk": "HIGH — lymphoproliferation -> lymphoma risk; annual surveillance",
        "sirolimus_response": "Partial adjunct — JAK inhibitors are primary",
        "hsct_required": "Reserved for refractory/lymphoma transformation",
        "attack_trigger_common": "Infections trigger autoimmune flares; constitutive inflammation",
        "onset_age": "First 1-5 years of life; T1DM often before age 5; lymphoproliferation early childhood",
    },
]

PATIENTS_PER_GENE = 40


def _make_cohort(gene_entry, seed):
    rng = random.Random(seed)
    gene = gene_entry["gene"]
    patients = []
    for i in range(PATIENTS_PER_GENE):
        age_at_dx = round(rng.uniform(0.1, 18.0), 1)

        # gene-specific lymphoproliferation severity
        if gene in ("FAS", "FASLG", "CASP10", "CASP8"):
            lympho_score = round(rng.uniform(1.5, 4.5), 1)  # ALPS lympho
        elif gene == "STAT3":
            lympho_score = round(rng.uniform(2.0, 5.0), 1)  # STAT3-GOF severe lympho
        elif gene == "FOXP3":
            lympho_score = round(rng.uniform(0.5, 2.5), 1)  # IPEX organomegaly, not classic lympho
        else:
            lympho_score = round(rng.uniform(0.5, 2.0), 1)  # WAS/DOCK8 less prominent lympho

        # DNT cells (ALPS genes)
        if gene in ("FAS", "FASLG", "CASP10", "CASP8"):
            dnt_pct = round(rng.uniform(2.0, 18.0), 1)
        else:
            dnt_pct = round(rng.uniform(0.0, 1.4), 1)  # Below diagnostic threshold

        # Autoimmunity
        if gene == "FAS":
            autoimmunity = rng.random() < 0.65
        elif gene in ("FASLG", "CASP10"):
            autoimmunity = rng.random() < 0.60
        elif gene == "CASP8":
            autoimmunity = rng.random() < 0.55
        elif gene == "FOXP3":
            autoimmunity = True  # By definition
        elif gene == "STAT3":
            autoimmunity = rng.random() < 0.85
        elif gene == "WAS":
            autoimmunity = rng.random() < 0.45  # older patients
        else:  # DOCK8
            autoimmunity = rng.random() < 0.15

        # Lymphoma risk
        if gene in ("FAS", "FASLG", "CASP10", "CASP8", "WAS", "DOCK8", "STAT3"):
            lymphoma = rng.random() < 0.12
        else:  # FOXP3
            lymphoma = rng.random() < 0.02

        # Infection type
        if gene in ("FAS", "FASLG", "CASP10"):
            infection_type = rng.choice(["EBV", "CMV", "bacterial", "none", "none"])
        elif gene == "CASP8":
            infection_type = rng.choice(["HSV", "bacterial", "EBV", "VZV", "CMV"])
        elif gene == "FOXP3":
            infection_type = rng.choice(["candida", "bacterial", "CMV", "bacterial"])
        elif gene == "WAS":
            infection_type = rng.choice(["bacterial", "PCP", "VZV", "CMV", "bacterial"])
        elif gene == "DOCK8":
            infection_type = rng.choice(["HSV", "HSV", "molluscum", "VZV", "bacterial"])
        else:  # STAT3
            infection_type = rng.choice(["bacterial", "viral", "CMV", "bacterial", "none"])

        # HSCT received
        if gene in ("FOXP3", "WAS", "DOCK8", "CASP8"):
            hsct_done = rng.random() < 0.55
        elif gene in ("FAS", "FASLG", "CASP10"):
            hsct_done = rng.random() < 0.18
        else:  # STAT3
            hsct_done = rng.random() < 0.12

        # Sirolimus/JAK inhibitor use
        if gene in ("FAS", "FASLG", "CASP10", "CASP8"):
            sirolimus_use = rng.random() < 0.75
        elif gene == "STAT3":
            sirolimus_use = rng.random() < 0.85  # JAK inhibitors (counted here)
        else:
            sirolimus_use = rng.random() < 0.25

        # Outcome
        if gene in ("FAS", "FASLG", "CASP10"):
            outcome = rng.choice([
                "lymphoproliferation controlled on sirolimus",
                "lymphoproliferation controlled on sirolimus",
                "autoimmune cytopenia remission",
                "lymphoma detected",
                "partial response MMF",
            ])
        elif gene == "CASP8":
            outcome = rng.choice([
                "lymphoproliferation controlled with infections managed",
                "recurrent infections ongoing",
                "HSCT curative",
                "lymphoma detected",
                "partial response",
            ])
        elif gene == "FOXP3":
            outcome = rng.choice([
                "HSCT curative — diabetes persists",
                "HSCT curative — enteropathy resolved",
                "tacrolimus bridge pending HSCT",
                "severe neonatal death without HSCT",
                "partial response sirolimus",
            ])
        elif gene == "WAS":
            outcome = rng.choice([
                "HSCT curative — all three components",
                "gene therapy OTL-103",
                "partial response IVIG",
                "lymphoma detected",
                "bleeding complication",
            ])
        elif gene == "DOCK8":
            outcome = rng.choice([
                "HSCT curative",
                "recurrent HSV controlled aciclovir",
                "lymphoma detected",
                "CD8 lymphopenia progressive",
                "partial response IVIG",
            ])
        else:  # STAT3
            outcome = rng.choice([
                "JAK inhibitor dramatic response",
                "JAK inhibitor dramatic response",
                "sirolimus adjunct partial",
                "HSCT post-lymphoma",
                "ongoing multi-organ autoimmunity",
            ])

        patients.append({
            "patient_id": f"{gene}-{seed}-{i + 1:03d}",
            "age_at_diagnosis_years": age_at_dx,
            "lymphoproliferation_score": lympho_score,
            "dnt_pct_of_lymphocytes": dnt_pct,
            "autoimmunity": autoimmunity,
            "lymphoma": lymphoma,
            "dominant_infection_type": infection_type,
            "hsct_received": hsct_done,
            "sirolimus_or_jak_inhibitor": sirolimus_use,
            "outcome": outcome,
            "gene": gene,
        })
    return patients


def generate_overview():
    all_patients = []
    for idx, entry in enumerate(IMMUNE_DYSREGULATION_GENES):
        seed = SEED_BASE + idx
        all_patients.extend(_make_cohort(entry, seed))

    total = len(all_patients)
    autoimmunity_count = sum(1 for p in all_patients if p["autoimmunity"])
    lymphoma_count = sum(1 for p in all_patients if p["lymphoma"])
    hsct_count = sum(1 for p in all_patients if p["hsct_received"])
    sirolimus_count = sum(1 for p in all_patients if p["sirolimus_or_jak_inhibitor"])
    alps_genes = {"FAS", "FASLG", "CASP10", "CASP8"}
    dnt_elevated = sum(1 for p in all_patients if p["gene"] in alps_genes and p["dnt_pct_of_lymphocytes"] > 1.5)

    gene_summary = {}
    for idx, entry in enumerate(IMMUNE_DYSREGULATION_GENES):
        gene = entry["gene"]
        cohort = _make_cohort(entry, SEED_BASE + idx)
        gene_summary[gene] = {
            "gene": gene,
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "disease_category": entry["disease_category"],
            "pathognomonic": entry["pathognomonic"][:300],
            "n_patients": len(cohort),
            "autoimmunity_pct": round(100 * sum(1 for p in cohort if p["autoimmunity"]) / len(cohort), 1),
            "lymphoma_pct": round(100 * sum(1 for p in cohort if p["lymphoma"]) / len(cohort), 1),
            "hsct_pct": round(100 * sum(1 for p in cohort if p["hsct_received"]) / len(cohort), 1),
            "sirolimus_jak_pct": round(100 * sum(1 for p in cohort if p["sirolimus_or_jak_inhibitor"]) / len(cohort), 1),
            "avg_dnt_pct": round(sum(p["dnt_pct_of_lymphocytes"] for p in cohort) / len(cohort), 1),
            "avg_age_dx_years": round(sum(p["age_at_diagnosis_years"] for p in cohort) / len(cohort), 1),
        }

    return {
        "atlas": "Hereditary-Immune-Dysregulation-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Immune Dysregulation Reference -- ALPS/IPEX/WAS/DOCK8/STAT3-GOF",
        "genes_covered": [e["gene"] for e in IMMUNE_DYSREGULATION_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_metrics": {
            "autoimmunity_pct": round(100 * autoimmunity_count / total, 1),
            "lymphoma_pct": round(100 * lymphoma_count / total, 1),
            "hsct_received_pct": round(100 * hsct_count / total, 1),
            "sirolimus_or_jak_inhibitor_pct": round(100 * sirolimus_count / total, 1),
            "alps_dnt_elevated_pct": round(100 * dnt_elevated / (PATIENTS_PER_GENE * 4), 1),
        },
        "gene_summary": gene_summary,
    }


def generate_breakdown():
    breakdown = []
    for idx, entry in enumerate(IMMUNE_DYSREGULATION_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(entry, seed)
        breakdown.append({
            "gene": entry["gene"],
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"],
            "disease_category": entry["disease_category"],
            "disease_pathway": entry["disease_pathway"],
            "pathognomonic": entry["pathognomonic"],
            "treatment": entry["treatment"],
            "key_features": entry["key_features"],
            "key_ddx": entry["key_ddx"],
            "autoimmunity_risk": entry["autoimmunity_risk"],
            "lymphoma_risk": entry["lymphoma_risk"],
            "sirolimus_response": entry["sirolimus_response"],
            "hsct_required": entry["hsct_required"],
            "attack_trigger_common": entry["attack_trigger_common"],
            "onset_age": entry["onset_age"],
            "n_patients": len(cohort),
            "autoimmunity_pct": round(100 * sum(1 for p in cohort if p["autoimmunity"]) / len(cohort), 1),
            "lymphoma_pct": round(100 * sum(1 for p in cohort if p["lymphoma"]) / len(cohort), 1),
            "hsct_pct": round(100 * sum(1 for p in cohort if p["hsct_received"]) / len(cohort), 1),
            "sirolimus_jak_pct": round(100 * sum(1 for p in cohort if p["sirolimus_or_jak_inhibitor"]) / len(cohort), 1),
            "avg_dnt_pct": round(sum(p["dnt_pct_of_lymphocytes"] for p in cohort) / len(cohort), 1),
            "avg_age_dx_years": round(sum(p["age_at_diagnosis_years"] for p in cohort) / len(cohort), 1),
            "sample_patients": cohort[:3],
        })
    return {"gene_breakdowns": breakdown}


def generate_definitions():
    return {
        "gene_entries": {
            entry["gene"]: {
                "gene": entry["gene"],
                "full_name": entry["protein"].split(" --")[0].strip(),
                "locus": entry["locus"],
                "protein_size": entry["protein_size"],
                "inheritance": entry["inheritance"].split(";")[0].strip(),
                "disease_name": entry["disease_category"],
                "disease_pathway": entry["disease_pathway"],
                "pathognomonic": entry["pathognomonic"],
                "treatment": entry["treatment"][:400],
                "key_features": entry["key_features"],
                "key_ddx": entry["key_ddx"],
                "autoimmunity_risk": entry["autoimmunity_risk"],
                "lymphoma_risk": entry["lymphoma_risk"],
                "sirolimus_response": entry["sirolimus_response"],
                "hsct_required": entry["hsct_required"],
                "attack_trigger_common": entry["attack_trigger_common"],
                "onset_age": entry["onset_age"],
            }
            for entry in IMMUNE_DYSREGULATION_GENES
        },
        "immune_dysregulation_glossary": {
            "ALPS — Autoimmune Lymphoproliferative Syndrome": (
                "A disorder of lymphocyte apoptosis. Hallmark: accumulation of DNT cells "
                "(CD3+CD4-CD8-TCRαβ+) >1.5% of lymphocytes due to AICD failure. "
                "ACNS 2010 diagnostic criteria: required features — chronic non-malignant "
                "lymphadenopathy/splenomegaly + elevated DNT cells; supportive — elevated sFasL, "
                "elevated IL-10/B12/IgG, defective FAS apoptosis assay, germline FAS/FASLG/CASP10 variant. "
                "Sirolimus ACNS 2020 first-line. Lymphoma risk 10-50x elevated — annual surveillance mandatory."
            ),
            "DNT Cells — Diagnostic ALPS Biomarker": (
                "Double-negative T cells (CD3+CD4-CD8-TCRαβ+) are the pathognomonic ALPS biomarker. "
                ">1.5% of total lymphocytes (or >2.5% of CD3+ T cells) required for ALPS diagnosis. "
                "These cells are antigen-experienced T cells that survived activation-induced cell death, "
                "lost CD4/CD8 co-receptor, and continue to accumulate in lymph nodes and spleen. "
                "Important: DNT threshold is for PERIPHERAL BLOOD — elevated in lymph nodes even in early ALPS. "
                "DNT cells are NOT a feature of FOXP3/IPEX, WAS, DOCK8, or STAT3-GOF."
            ),
            "IPEX — FOXP3 Triad and HSCT": (
                "IPEX (FOXP3 LOF) presents in neonatal period with the triad: "
                "neonatal T1DM (within weeks of birth), intractable secretory enteropathy (watery diarrhoea, villous atrophy), "
                "severe atopic eczema in boys. "
                "TREG absent on flow cytometry (CD4+CD25+FoxP3+ <1% of CD4 T cells). "
                "FATAL within 1-2 years without treatment in severe cases. "
                "HSCT is the only curative therapy — correct donor match and early timing are critical. "
                "Bridge to HSCT: tacrolimus or sirolimus (4-6 months). "
                "Post-HSCT: T1DM persists (pancreatic beta cells already destroyed), enteropathy resolves."
            ),
            "WAS Small Platelets — Critical DDx from ITP": (
                "WAS microthrombocytopenia: platelet count low (<70,000/μL) + MPV SMALL (<7 fL, often <5 fL). "
                "ITP (immune thrombocytopenic purpura): platelet count low but MPV LARGE (reactive young platelets). "
                "Distinguishing WAS from ITP: MPV and platelet morphology on film are the CRITICAL first step "
                "BEFORE considering splenectomy (which is CONTRAINDICATED in WAS as it increases fatal infections). "
                "WASp expression by flow cytometry (anti-WASp antibody staining on monocytes/T cells) is the "
                "rapid confirmatory test; WAS gene sequencing for definitive diagnosis."
            ),
            "DOCK8 vs STAT3-LOF HIES — Key DDx": (
                "Both present with very high IgE and eczema but are distinct: "
                "DOCK8 (AR; 9p24.3): herpesviral skin dissemination (HSV, molluscum) PATHOGNOMONIC; "
                "progressive CD8 lymphopenia; no skeletal/dental features; no pneumatoceles; HSCT curative. "
                "STAT3-LOF/HIES1 (AD; 17q21.2): skeletal features (scoliosis, minimal trauma fractures), "
                "retained deciduous teeth, coarse facies, pneumatoceles after pneumonia (Staph aureus/Aspergillus); "
                "NO herpesviral dissemination; NO lymphoproliferation; HSCT NOT recommended. "
                "STAT3-GOF: lymphoproliferation + autoimmunity + short stature — entirely different phenotype "
                "from STAT3-LOF despite same gene."
            ),
            "STAT3 GOF vs LOF — Same Gene, Opposite Phenotypes": (
                "STAT3 is unusual: GOF and LOF variants in the same gene cause opposite immune phenotypes. "
                "STAT3-LOF (HIES1/Job syndrome): LOSS of STAT3 -> impaired Th17 (cannot fight Staph/Candida) "
                "-> high IgE + eczema + recurrent skin/lung infections + skeletal features. NO lymphoproliferation. "
                "STAT3-GOF: GAIN of STAT3 -> constitutive STAT3 -> suppresses FOXP3/TREG -> lymphoproliferation "
                "+ multi-organ autoimmunity + short stature + T1DM. NO skeletal/dental features. "
                "STAT3-GOF responds dramatically to JAK inhibitors (ruxolitinib) — STAT3-LOF does NOT respond "
                "to JAK inhibitors (different mechanism). Variant pathogenicity must specify GOF or LOF."
            ),
            "JAK Inhibitors in Immune Dysregulation": (
                "JAK inhibitors block JAK1/JAK2/JAK3/TYK2 -> prevent STAT3 phosphorylation -> "
                "reduce constitutive STAT3 activity in STAT3-GOF. "
                "Ruxolitinib (JAK1/2): most evidence in STAT3-GOF; also used in ALPS refractory cases. "
                "Tofacitinib (JAK1/3): alternative. "
                "Response in STAT3-GOF: dramatic within 2-4 weeks (lymphadenopathy reduction, "
                "autoimmunity improvement, inflammatory marker normalisation). "
                "Important: JAK inhibitors NOT effective in STAT3-LOF (different mechanism) and NOT "
                "first-line in ALPS-Ia/Ib/IIa/IIb (sirolimus is preferred)."
            ),
            "Sirolimus in ALPS — ACNS 2020 Standard": (
                "Sirolimus (rapamycin, mTOR inhibitor) is the ACNS 2020 first-line for ALPS. "
                "Mechanism: mTOR inhibition -> reduced lymphocyte proliferation and survival -> "
                "DNT cell reduction, lymphadenopathy/splenomegaly control, autoimmune cytopenia resolution. "
                "Target level: 5-15 ng/mL. Response within 2-4 weeks (faster than previous agents). "
                "Side effects: oral mucositis (dose-reduce, folate supplement), hyperlipidaemia, "
                "impaired wound healing, pulmonary toxicity (rare). "
                "Previously mycophenolate mofetil (MMF) was most-used — sirolimus now preferred due to "
                "superior DNT cell control and lymphoma risk reduction. "
                "IMPORTANT: sirolimus does NOT prevent lymphoma — annual surveillance CT remains mandatory."
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (gene 0 only) ===")
    bd = generate_breakdown()
    print(json.dumps(bd["gene_breakdowns"][0], indent=2)[:2000])
    print("\n=== DEFINITIONS (first key) ===")
    defn = generate_definitions()
    first_key = next(iter(defn["gene_entries"]))
    print(json.dumps(defn["gene_entries"][first_key], indent=2)[:1500])
