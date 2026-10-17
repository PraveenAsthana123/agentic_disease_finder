#!/usr/bin/env python3
"""Hereditary-Agammaglobulinemia-CVID-Atlas -- Complete 8-Gene Reference
B-Cell and Antibody Deficiency Primary Immunodeficiencies: B-cell maturation arrest and CVID by gene
(BTK/XLA-Most-Common-85pct-XL-Echovirus-Encephalitis-PATHOGNOMONIC-IVIG-Lifelong,
 IGHM/AR-Agamma1-Pre-BCR-Proximal-Female-Patients-Key-Distinction,
 IGLL1/AR-Agamma2-Lambda5-Surrogate-Light-Chain,
 BLNK/AR-Agamma4-SLP65-B-Cell-Linker,
 PIK3CD/AD-GOF-APDS1-Activated-PI3K-Delta-EBV-CMV-Leniolisib-FDA2023,
 PIK3R1/AR-APDS2-Regulatory-Subunit-p85alpha-Similar-APDS1,
 LRBA/AR-LRBA-Deficiency-CTLA4-Surface-Loss-Autoimmunity-Abatacept-HIGHLY-EFFECTIVE,
 NFKB1/AD-LOF-CVID7-Most-Common-Genetic-CVID-10pct-Splenomegaly-IBD-DISTINGUISHER):

BTK    (Bruton's tyrosine kinase; 659aa; Xq22.1; X-linked hemizygous males;
         X-linked Agammaglobulinemia (XLA) — MOST COMMON B-cell deficiency (85% of agammaglobulinemia);
         BTK is the tyrosine kinase essential for pre-B cell receptor (pre-BCR) signalling and
           mature B-cell activation — all B lymphocyte maturation is arrested at the pro-B cell stage
           without functional BTK;
         NO CIRCULATING B CELLS: peripheral blood B-lymphocyte count virtually zero (<1% of lymphocytes);
         Immunoglobulins ALL severely reduced: IgG, IgA, IgM all <200 mg/dL (often undetectable);
         ONSET: maternal IgG protects newborns — symptoms emerge at 6-9 months when maternal antibody
           wanes; recurrent bacterial sinopulmonary infections (H. influenzae, S. pneumoniae, Pseudomonas)
           and Giardia lamblia GI infections are the presenting features;
         PATHOGNOMONIC: ECHOVIRUS (enterovirus) MENINGOENCEPHALITIS — chronic CNS enteroviral infection
           that leads to progressive encephalopathy with skin/muscle involvement (DERMO-MATO-ENCEPHALITIS);
           occurs in untreated or inadequately replaced XLA; this CNS complication is UNIQUE to
           agammaglobulinemia among PIDs;
         TREATMENT: IVIG replacement LIFELONG (target IgG trough ≥800 mg/dL); prevent echovirus
           encephalitis — screen with stool PCR/CSF PCR in patients presenting with neurological symptoms;
         HISTORICAL: first PID described by Colonel Ogden Bruton in 1952 — discovery of 'agammaglobulinemia'
           in a boy with recurrent pneumococcal infections and absent gamma-globulin band on serum protein
           electrophoresis;
         X-LINKED: only MALES affected (hemizygous); if female with identical phenotype → sequence IGHM,
           IGLL1, BLNK first;
         seed SEED_BASE+0).
IGHM   (immunoglobulin heavy constant mu; 576aa; 14q32.33; AR;
         Agammaglobulinemia Type 1 (AGA1) — autosomal recessive agammaglobulinemia;
         IGHM encodes the IgM heavy chain (Cμ constant region); the pre-BCR complex requires:
           Igμ heavy chain + lambda5 (IGLL1) + VpreB1/VpreB2 surrogate light chain + Igα(CD79a)/Igβ(CD79b);
         Without functional Igμ heavy chain: pre-BCR cannot form → B-cell maturation arrested at pro-B
           to pre-B cell transition;
         PHENOTYPE IDENTICAL TO BTK-XLA: absent B cells, agammaglobulinemia, recurrent sinopulmonary
           infections, Giardia, echovirus risk;
         KEY DISTINGUISHER FROM BTK: AUTOSOMAL RECESSIVE — BOTH MALES AND FEMALES AFFECTED;
           if female with XLA-like phenotype: IGHM is the first gene to sequence;
         Second most common cause of autosomal recessive agammaglobulinemia (after BLNK-tied with IGLL1);
         Treatment: IVIG replacement lifelong — identical to BTK-XLA management;
         seed SEED_BASE+1).
IGLL1  (immunoglobulin lambda-like polypeptide 1 = lambda5; 214aa; 22q11.23; AR;
         Agammaglobulinemia Type 2 (AGA2) — autosomal recessive agammaglobulinemia;
         IGLL1 encodes lambda5, the surrogate light chain component of the pre-BCR along with
           VpreB1 and VpreB2 — the pre-BCR uses this 'fake' lambda-like light chain transiently
           during pro-B cell differentiation to test μ heavy chain folding before true light chain
           gene rearrangement;
         Without lambda5: the surrogate light chain half of pre-BCR is absent → pre-BCR assembly fails
           → B-cell maturation blocked at pro-B stage;
         PHENOTYPE: identical to IGHM and BTK-XLA (absent B cells, agammaglobulinemia, recurrent bacterial
           infections); AR inheritance (both sexes);
         SMALLEST protein in the agammaglobulinemia panel (214aa, 24.2kDa);
         Clinical severity may be slightly milder in some IGLL1 patients (partial in some alleles);
         Treatment: IVIG lifelong — same as BTK/IGHM;
         seed SEED_BASE+2).
BLNK   (B-cell linker protein = SLP-65; 456aa; 10q24.1; AR;
         Agammaglobulinemia Type 4 (AGA4) — autosomal recessive agammaglobulinemia;
         BLNK/SLP-65 is the scaffold adaptor protein that physically links BTK and PLCγ2 in the
           B-cell receptor signalling cascade: BCR activation → Lyn/Syk → phospho-BLNK → recruits
           BTK + PLCγ2 → PLC activation → Ca2+/PKC/MAPK → B-cell proliferation/differentiation;
         Without BLNK: the BTK-PLCγ2 connection is severed → downstream BCR signalling aborted →
           B-cell maturation arrest at pro-B stage (identical arrest point to BTK deficiency);
         PHENOTYPE IDENTICAL TO BTK-XLA/IGHM: absent B cells, agammaglobulinemia, bacterial sinopulmonary
           infections, Giardia, echovirus risk; AR inheritance (both sexes);
         FUNCTIONAL PHENOCOPY OF BTK: despite BLNK being downstream of BTK, loss of either produces
           identical clinical syndrome — mechanistic redundancy in the BTK→BLNK→PLCγ2 axis means
           disruption of any node causes complete B-cell maturation failure;
         Treatment: IVIG replacement lifelong; same as BTK/IGHM/IGLL1;
         seed SEED_BASE+3).
PIK3CD (phosphatidylinositol-4,5-bisphosphate 3-kinase catalytic subunit delta; 1044aa; 1p36.22; AD GOF;
         Activated PI3K-delta Syndrome type 1 (APDS1) / PASLI disease;
         PI3Kδ is the phosphoinositide 3-kinase catalytic subunit expressed almost exclusively in
           leukocytes; it converts PIP2 to PIP3 at the cell membrane; GOF PIK3CD variants (most
           commonly E1021K, E525K) → constitutively active PI3Kδ → PIP3 accumulation →
           hyperactivation of Akt/mTOR/S6K1 signalling;
         PHENOTYPE (distinct from classical agammaglobulinemia — B cells PRESENT but non-functional):
           (1) B cells present but with maturation defect: naive B cells accumulate, memory B cells
               reduced, class-switched memory B cells nearly absent; hypogammaglobulinaemia;
           (2) EBV AND CMV SUSCEPTIBILITY: profoundly impaired CD8+ T-cell killing (terminally
               differentiated/exhausted T-cell pool) → EBV-driven lymphoproliferation and
               lymphoma; CMV disease; this EBV susceptibility is a DISTINGUISHER from classical XLA;
           (3) SINOPULMONARY INFECTIONS + BRONCHIECTASIS: recurrent bacterial infections from
               antibody deficiency, progressing to bronchiectasis;
           (4) LYMPHOPROLIFERATION: lymphadenopathy, splenomegaly, EBV-driven lymphoma;
           (5) AUTOIMMUNITY: autoimmune cytopenias (ITP, AIHA);
         PRECISION TREATMENT: LENIOLISIB (PI3Kδ-specific inhibitor) FDA APPROVED 2023 for APDS1/2;
           also idelalisib (broader PI3K inhibitor, more toxicity); these drugs directly target the
           activated PI3Kδ pathway — disease-modifying, not just replacement;
         seed SEED_BASE+4).
PIK3R1 (phosphoinositide-3-kinase regulatory subunit 1 = p85α; 724aa; 5q13.1; AR (some AD);
         Activated PI3K-delta Syndrome type 2 (APDS2);
         PIK3R1 encodes the regulatory subunit p85α which, under normal circumstances, INHIBITS the
           catalytic subunit p110δ (encoded by PIK3CD) — p85α keeps PI3Kδ restrained in the resting
           state; LOF PIK3R1 → loss of p85α inhibition → p110δ constitutively active → same downstream
           PIP3/Akt/mTOR hyperactivation as APDS1;
         PHENOTYPE: APDS2 = phenotypically identical to APDS1 (same activated PI3Kδ pathway, different
           gene): sinopulmonary infections, bronchiectasis, B-cell maturation defect, EBV/CMV susceptibility,
           lymphoproliferation, autoimmunity;
         DISTINGUISHER FROM APDS1: AR inheritance (homozygous/compound heterozygous PIK3R1 LOF) vs
           APDS1 which is AD GOF PIK3CD; p85α also regulates other PI3K isoforms (p110α/β) so some
           APDS2 patients may have additional phenotypic features;
         TREATMENT: same as APDS1 — leniolisib/idelalisib (PI3Kδ inhibition addresses the excess PI3Kδ
           activity even though caused by loss of inhibitory subunit); IVIG for antibody deficiency;
         seed SEED_BASE+5).
LRBA   (LPS-responsive beige-like anchor protein; 2863aa; 4q31.3; AR;
         LRBA deficiency — CTLA4 surface loss, multi-organ autoimmunity, Abatacept curative;
         LRBA is a large scaffolding protein involved in intracellular vesicular trafficking; its key
           function relevant to immune deficiency: LRBA is required for recycling of CTLA4 from the
           lysosomal degradation pathway back to the cell surface — without LRBA, CTLA4 is rapidly
           sorted to lysosomes and degraded, dramatically reducing CTLA4 surface expression on T cells;
         MECHANISM OF DISEASE: CTLA4 surface expression on T regulatory (Treg) cells and activated
           T cells is required for trans-endocytosis of CD80/CD86 on antigen-presenting cells — this
           CTLA4 checkpoint normally prevents autoreactive T-cell activation; reduced CTLA4 → unchecked
           T-cell activation → immune dysregulation affecting multiple organs;
         PHENOTYPE (immune dysregulation + antibody deficiency):
           (1) AUTOIMMUNITY (DOMINANT feature): autoimmune cytopenias (ITP, AIHA, neutropenia),
               inflammatory bowel disease-like colitis, autoimmune hepatitis, interstitial lung disease
               (lymphoid interstitial pneumonia, LIP), thyroiditis, diabetes Type 1;
           (2) HYPOGAMMAGLOBULINAEMIA: CVID-like antibody deficiency — low IgG/IgA/IgM, poor vaccine
               responses, recurrent sinopulmonary infections;
           (3) LYMPHOPROLIFERATION: splenomegaly, lymphadenopathy, organomegaly; EBV-driven LPD;
           (4) OFTEN MISDIAGNOSED AS CVID for years before LRBA gene identified;
         PATHOGNOMONIC TREATMENT RESPONSE: ABATACEPT (CTLA4-Ig fusion protein, Orencia) — directly
           substitutes for the absent CTLA4 function → dramatic improvement in autoimmune manifestations
           (HIGHLY EFFECTIVE, often disease-modifying); recommended as first-line specific therapy;
           rapamycin (sirolimus) for lymphoproliferation; IVIG for antibody deficiency;
         LRBA is the LARGEST protein in this panel (2863aa, 319kDa);
         seed SEED_BASE+6).
NFKB1  (nuclear factor kappa-B p50/p105; 968aa; 4q24; AD LOF haploinsufficiency;
         CVID Type 7 (CVID7) — MOST COMMON DEFINED GENETIC CAUSE OF CVID (~10% of genetic CVID cases);
         NFKB1 encodes both NF-κB1 p50 (the mature transcription factor, generated by proteolytic
           processing) and its precursor p105 (which also functions as an IκB-like inhibitor);
         NF-κB1 (p50) forms homodimers or heterodimers with RelA (p65) — these transcription factor
           complexes drive expression of genes critical for B-cell class switching, germinal centre
           formation, and plasma cell differentiation;
         AD HAPLOINSUFFICIENCY: ONE non-functional allele is sufficient to cause disease — unlike AR
           agammaglobulinemia where both alleles must be lost; this explains VARIABLE EXPRESSIVITY and
           INCOMPLETE PENETRANCE within families (carriers may be asymptomatic or mildly affected);
         PHENOTYPE (later-onset CVID pattern, often adult-onset):
           (1) ANTIBODY DEFICIENCY: low IgG + IgA +/- IgM, poor vaccine responses; recurrent
               sinopulmonary bacterial infections; bronchiectasis from chronic infection;
           (2) SPLENOMEGALY: common (NF-κB1 normally restrains splenic B cell homeostasis) — prominent
               DISTINGUISHER among genetic CVID causes;
           (3) IBD-LIKE INFLAMMATORY COLITIS: Crohn's-like gastrointestinal disease with prominent
               granulomatous inflammation; NF-κB1 haploinsufficiency in gut epithelial cells impairs
               innate immune homeostasis — drives chronic gut inflammation; IBD-DISTINGUISHER;
           (4) AUTOIMMUNE CYTOPENIAS: ITP, AIHA;
           (5) LYMPHOPROLIFERATION: splenic marginal zone hyperplasia; lymphoma risk;
         TREATMENT: IVIG replacement; antibiotics for infections; anti-TNF/anti-integrin for IBD;
           corticosteroids for autoimmunity; bronchiectasis management; family cascade testing (AD);
         seed SEED_BASE+7).
"""
import random
import math

SEED_BASE = 3878

ATLAS_GENES = [
    {
        "name": "BTK",
        "full_name": "Bruton's tyrosine kinase",
        "aa": 659,
        "locus": "Xq22.1",
        "inheritance": "XL",
        "mechanism": "LOF",
        "syndrome": "XLA / X-linked-Agammaglobulinemia / B-Cells-Absent / Echovirus-Encephalitis-PATHOGNOMONIC / IVIG-Lifelong",
        "mw_kda": 76.3,
        "critical_rules": [
            "BTK-XL-MALES-ONLY-HEMIZYGOUS-FEMALES-CARRIER-ASYMPTOMATIC",
            "BTK-B-CELLS-ABSENT-LESS-THAN-1PCT-PERIPHERAL-LYMPHOCYTES",
            "BTK-ECHOVIRUS-MENINGOENCEPHALITIS-PATHOGNOMONIC-SCREEN-NEUROLOGICAL",
            "BTK-IVIG-TROUGH-GT800-LIFELONG-MANDATORY",
        ],
        "distinguishing_feature": "XLA most common (85% agammaglobulinemia); males only (XL); all B cells absent; ALL immunoglobulins undetectable; echovirus/enterovirus meningoencephalitis PATHOGNOMONIC; Bruton's (first PID described 1952); IVIG lifelong",
        "first_line_treatment": "IVIG replacement lifelong (target trough IgG ≥800 mg/dL; some centres ≥1000); subcutaneous IgG (SCIG) as home alternative; enterovirus/echovirus CNS surveillance (CSF PCR + stool PCR in neurological symptoms); inactivated vaccines only (no live vaccines — no MMR, varicella, oral polio); antibiotic prophylaxis for breakthrough infections",
        "pathognomonic": "Male infant at 6-9 months with recurrent bacterial sinopulmonary infections + absent B cells on flow cytometry + all immunoglobulins undetectable + Giardia = BTK-XLA. Echovirus encephalitis in male with agammaglobulinemia = PATHOGNOMONIC.",
        "onset_months_mean": 8.0,
        "penetrance_pct": 100,
        "b_cells_absent_pct": 100,
        "agammaglobulinemia_pct": 100,
        "sinopulmonary_pct": 92,
        "gi_infection_pct": 60,
        "enteroviral_pct": 35,
        "autoimmunity_pct": 12,
        "lymphoproliferation_pct": 8,
        "bronchiectasis_pct": 38,
        "ivig_responsive_pct": 92,
        "hsct_benefit_pct": 20,
        "dx_lag_yrs": 2.5,
    },
    {
        "name": "IGHM",
        "full_name": "Immunoglobulin heavy constant mu (Igμ heavy chain)",
        "aa": 576,
        "locus": "14q32.33",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "AR-Agammaglobulinemia-Type1 / Pre-BCR-Proximal / Female-Patients-Key-Distinction",
        "mw_kda": 65.0,
        "critical_rules": [
            "IGHM-AR-BOTH-MALES-AND-FEMALES-UNLIKE-BTK-XLA",
            "IGHM-B-CELLS-ABSENT-IDENTICAL-TO-BTK-PHENOTYPE",
            "IGHM-FIRST-GENE-SEQUENCE-IN-FEMALE-WITH-XLA-PHENOTYPE",
            "IGHM-IVIG-LIFELONG-IDENTICAL-MANAGEMENT-TO-BTK",
        ],
        "distinguishing_feature": "AR agammaglobulinemia Type 1; phenotype IDENTICAL to BTK-XLA (absent B cells, agammaglobulinemia, bacterial infections); KEY DISTINGUISHER: AR inheritance — affects FEMALES; if female with XLA-like phenotype → sequence IGHM first",
        "first_line_treatment": "IVIG replacement lifelong (identical to BTK-XLA management; target trough ≥800 mg/dL); enterovirus surveillance; inactivated vaccines only; antibiotic prophylaxis; genetic counselling for AR (sibling recurrence 25%, parents carriers)",
        "pathognomonic": "Female (or male) infant at 6-9 months with absent B cells + agammaglobulinemia + recurrent bacterial infections = AR agammaglobulinemia; IGHM is most common AR cause. In female with XLA-like phenotype: sequence IGHM before other AR genes.",
        "onset_months_mean": 8.5,
        "penetrance_pct": 100,
        "b_cells_absent_pct": 100,
        "agammaglobulinemia_pct": 100,
        "sinopulmonary_pct": 90,
        "gi_infection_pct": 55,
        "enteroviral_pct": 30,
        "autoimmunity_pct": 12,
        "lymphoproliferation_pct": 8,
        "bronchiectasis_pct": 35,
        "ivig_responsive_pct": 90,
        "hsct_benefit_pct": 20,
        "dx_lag_yrs": 3.0,
    },
    {
        "name": "IGLL1",
        "full_name": "Immunoglobulin lambda-like polypeptide 1 (lambda5 surrogate light chain)",
        "aa": 214,
        "locus": "22q11.23",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "AR-Agammaglobulinemia-Type2 / Lambda5-Surrogate-Light-Chain / Pre-BCR-Formation-Failed",
        "mw_kda": 24.2,
        "critical_rules": [
            "IGLL1-LAMBDA5-SURROGATE-LIGHT-CHAIN-PRE-BCR-ABSENT",
            "IGLL1-SMALLEST-PROTEIN-AGAMMAGLOBULINEMIA-PANEL-214AA",
            "IGLL1-B-CELLS-ABSENT-AGAMMAGLOBULINEMIA-IDENTICAL-XLA-PHENOTYPE",
            "IGLL1-AR-BOTH-SEXES-IVIG-LIFELONG",
        ],
        "distinguishing_feature": "AR agammaglobulinemia Type 2; lambda5 surrogate light chain component of pre-BCR; SMALLEST protein in panel (214aa, 24.2kDa); B-cell arrest at pro-B stage; phenotype identical to BTK/IGHM; AR inheritance (both sexes); milder severity in some alleles",
        "first_line_treatment": "IVIG replacement lifelong; identical to BTK/IGHM management; enterovirus surveillance; inactivated vaccines only; monitor for breakthrough bacterial infections",
        "pathognomonic": "AR agammaglobulinemia (absent B cells + agammaglobulinemia + recurrent bacterial infections) in male OR female with negative BTK/IGHM/BLNK sequencing → sequence IGLL1 (and VpreB1). Phenotype identical to XLA.",
        "onset_months_mean": 9.0,
        "penetrance_pct": 100,
        "b_cells_absent_pct": 100,
        "agammaglobulinemia_pct": 100,
        "sinopulmonary_pct": 85,
        "gi_infection_pct": 50,
        "enteroviral_pct": 25,
        "autoimmunity_pct": 10,
        "lymphoproliferation_pct": 7,
        "bronchiectasis_pct": 28,
        "ivig_responsive_pct": 88,
        "hsct_benefit_pct": 20,
        "dx_lag_yrs": 4.0,
    },
    {
        "name": "BLNK",
        "full_name": "B-cell linker protein (SLP-65)",
        "aa": 456,
        "locus": "10q24.1",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "AR-Agammaglobulinemia-Type4 / SLP65-Adaptor / Functional-Phenocopy-BTK / B-Cell-Linker",
        "mw_kda": 50.1,
        "critical_rules": [
            "BLNK-SLP65-LINKS-BTK-TO-PLCGAMMA2-B-CELL-SIGNALLING",
            "BLNK-FUNCTIONAL-PHENOCOPY-BTK-IDENTICAL-CLINICAL-SYNDROME",
            "BLNK-B-CELLS-ABSENT-AGAMMAGLOBULINEMIA-AR-BOTH-SEXES",
            "BLNK-IVIG-LIFELONG-SAME-MANAGEMENT-AS-BTK",
        ],
        "distinguishing_feature": "AR agammaglobulinemia Type 4; SLP-65/BLNK adaptor links BTK→PLCγ2 in BCR signalling; functional phenocopy of BTK (same B-cell arrest point despite being downstream); absent B cells, agammaglobulinemia; AR (both sexes); identical management to BTK/IGHM",
        "first_line_treatment": "IVIG lifelong (identical to BTK/IGHM/IGLL1); enterovirus CNS surveillance; inactivated vaccines only; antibiotic prophylaxis; genetic counselling AR inheritance",
        "pathognomonic": "AR agammaglobulinemia (absent B cells + agammaglobulinemia + recurrent bacterial infections) in male or female with negative BTK/IGHM/IGLL1 sequencing → sequence BLNK. Clinical phenotype indistinguishable from BTK-XLA.",
        "onset_months_mean": 9.5,
        "penetrance_pct": 100,
        "b_cells_absent_pct": 100,
        "agammaglobulinemia_pct": 100,
        "sinopulmonary_pct": 85,
        "gi_infection_pct": 50,
        "enteroviral_pct": 25,
        "autoimmunity_pct": 10,
        "lymphoproliferation_pct": 7,
        "bronchiectasis_pct": 28,
        "ivig_responsive_pct": 88,
        "hsct_benefit_pct": 20,
        "dx_lag_yrs": 4.5,
    },
    {
        "name": "PIK3CD",
        "full_name": "Phosphatidylinositol-4,5-bisphosphate 3-kinase catalytic subunit delta (p110δ)",
        "aa": 1044,
        "locus": "1p36.22",
        "inheritance": "AD",
        "mechanism": "GOF",
        "syndrome": "APDS1 / Activated-PI3K-Delta-Syndrome / EBV-CMV-Susceptibility / Leniolisib-FDA2023",
        "mw_kda": 118.9,
        "critical_rules": [
            "PIK3CD-GOF-B-CELLS-PRESENT-NOT-ABSENT-UNLIKE-XLA",
            "PIK3CD-EBV-CMV-SUSCEPTIBILITY-DISTINGUISHER-FROM-CLASSICAL-AGAMMAGLOBULINEMIA",
            "PIK3CD-LENIOLISIB-PI3K-DELTA-INHIBITOR-FDA2023-DISEASE-MODIFYING",
            "PIK3CD-LYMPHOPROLIFERATION-LYMPHOMA-RISK-EBV-DRIVEN",
        ],
        "distinguishing_feature": "AD-GOF APDS1; PI3Kδ constitutively active; B CELLS PRESENT (not absent — key distinguisher from classical agammaglobulinemia); antibody deficiency + EBV/CMV susceptibility + lymphoproliferation; leniolisib FDA2023 (PI3Kδ-specific inhibitor, disease-modifying); autoimmunity prominent",
        "first_line_treatment": "LENIOLISIB (FDA-approved 2023 for APDS1; specific PI3Kδ kinase inhibitor) — primary disease-modifying therapy; IVIG for antibody deficiency; antivirals (valganciclovir for CMV; acyclovir for HSV); rituximab for EBV-driven lymphoproliferation; rapamycin for lymphoproliferation; avoid live vaccines (impaired immune surveillance); consider HSCT for severe refractory disease",
        "pathognomonic": "Recurrent sinopulmonary infections + bronchiectasis + B CELLS PRESENT (not absent) + antibody deficiency + EBV lymphoproliferation/lymphoma + CMV disease = APDS1; sequence PIK3CD (E1021K, E525K hotspots). Leniolisib/idelalisib response confirms diagnosis.",
        "onset_months_mean": 36.0,
        "penetrance_pct": 90,
        "b_cells_absent_pct": 5,
        "agammaglobulinemia_pct": 55,
        "sinopulmonary_pct": 90,
        "gi_infection_pct": 40,
        "enteroviral_pct": 60,
        "autoimmunity_pct": 45,
        "lymphoproliferation_pct": 70,
        "bronchiectasis_pct": 55,
        "ivig_responsive_pct": 65,
        "hsct_benefit_pct": 55,
        "dx_lag_yrs": 6.0,
    },
    {
        "name": "PIK3R1",
        "full_name": "Phosphoinositide-3-kinase regulatory subunit 1 (p85α)",
        "aa": 724,
        "locus": "5q13.1",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "APDS2 / Regulatory-Subunit-p85alpha / PI3K-Delta-Constitutive-Same-APDS1 / Leniolisib-Responsive",
        "mw_kda": 83.6,
        "critical_rules": [
            "PIK3R1-AR-LOF-UNLIKE-PIK3CD-AD-GOF-SAME-DOWNSTREAM-PATHWAY",
            "PIK3R1-P85ALPHA-INHIBITS-P110-DELTA-LOSS-DISINHIBITS-PI3K",
            "PIK3R1-B-CELLS-PRESENT-ANTIBODY-DEFICIENCY-LIKE-APDS1",
            "PIK3R1-LENIOLISIB-TARGETS-OVERACTIVE-PI3K-REGARDLESS-OF-CAUSE",
        ],
        "distinguishing_feature": "AR APDS2; p85α normally inhibits p110δ (PI3Kδ); LOF p85α → PI3Kδ disinhibited → constitutively active (same as PIK3CD-GOF by different mechanism); AR inheritance distinguishes from APDS1; B cells present; sinopulmonary + EBV/CMV + lymphoproliferation; leniolisib addresses excess PI3Kδ activity",
        "first_line_treatment": "LENIOLISIB (addresses excess PI3Kδ activity caused by loss of p85α inhibition — same target as APDS1); IVIG for antibody deficiency; antivirals for EBV/CMV; rapamycin for lymphoproliferation; genetic counselling AR (parents carriers, siblings 25% risk); similar management to APDS1",
        "pathognomonic": "APDS phenotype (sinopulmonary infections + B-cell maturation defect + EBV/CMV + lymphoproliferation) + AR inheritance (parental consanguinity or heterozygous parents) + PIK3CD sequencing NEGATIVE → sequence PIK3R1 (APDS2). Same clinical features as APDS1 — molecular testing distinguishes.",
        "onset_months_mean": 42.0,
        "penetrance_pct": 85,
        "b_cells_absent_pct": 5,
        "agammaglobulinemia_pct": 50,
        "sinopulmonary_pct": 88,
        "gi_infection_pct": 38,
        "enteroviral_pct": 55,
        "autoimmunity_pct": 40,
        "lymphoproliferation_pct": 65,
        "bronchiectasis_pct": 50,
        "ivig_responsive_pct": 65,
        "hsct_benefit_pct": 55,
        "dx_lag_yrs": 7.0,
    },
    {
        "name": "LRBA",
        "full_name": "LPS-responsive beige-like anchor protein",
        "aa": 2863,
        "locus": "4q31.3",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "LRBA-Deficiency / CTLA4-Surface-Loss / Multi-Organ-Autoimmunity / Abatacept-HIGHLY-EFFECTIVE",
        "mw_kda": 319.2,
        "critical_rules": [
            "LRBA-CTLA4-SURFACE-EXPRESSION-REDUCED-NOT-ABSENT-CTLA4-GENE",
            "LRBA-AUTOIMMUNITY-DOMINANT-FEATURE-UNLIKE-CLASSICAL-AGAMMAGLOBULINEMIA",
            "LRBA-ABATACEPT-CTLA4-IG-HIGHLY-EFFECTIVE-FIRST-LINE-SPECIFIC",
            "LRBA-OFTEN-MISDIAGNOSED-AS-CVID-YEARS-BEFORE-LRBA-IDENTIFIED",
        ],
        "distinguishing_feature": "AR LRBA deficiency; CTLA4 surface expression reduced (LRBA required for CTLA4 recycling from lysosomes); AUTOIMMUNITY DOMINANT feature (ITP/AIHA/IBD/hepatitis/ILD/diabetes) + hypogammaglobulinaemia; LARGEST protein in panel (2863aa, 319kDa); Abatacept (CTLA4-Ig) HIGHLY EFFECTIVE disease-modifying; often misdiagnosed as CVID",
        "first_line_treatment": "ABATACEPT (Orencia; CTLA4-Ig fusion protein) — substitutes for absent CTLA4 surface checkpoint → highly effective for autoimmune manifestations (ITP, IBD, hepatitis, ILD); first-line specific therapy; IVIG for antibody deficiency; rapamycin/sirolimus for lymphoproliferation; rituximab for refractory autoimmune cytopenias; immunosuppression for IBD/hepatitis; HSCT curative for severe refractory cases",
        "pathognomonic": "CVID-like (antibody deficiency + recurrent infections) + MULTI-ORGAN AUTOIMMUNITY (especially ITP + IBD-like colitis + hepatitis) + splenomegaly/lymphadenopathy = LRBA deficiency. Check CTLA4 surface expression on T cells (reduced). Abatacept response is strongly diagnostic/therapeutic. Sequence LRBA.",
        "onset_months_mean": 24.0,
        "penetrance_pct": 95,
        "b_cells_absent_pct": 3,
        "agammaglobulinemia_pct": 65,
        "sinopulmonary_pct": 75,
        "gi_infection_pct": 50,
        "enteroviral_pct": 30,
        "autoimmunity_pct": 90,
        "lymphoproliferation_pct": 85,
        "bronchiectasis_pct": 35,
        "ivig_responsive_pct": 70,
        "hsct_benefit_pct": 65,
        "dx_lag_yrs": 5.0,
    },
    {
        "name": "NFKB1",
        "full_name": "Nuclear factor kappa-B subunit 1 (NF-κB1 p50/p105)",
        "aa": 968,
        "locus": "4q24",
        "inheritance": "AD",
        "mechanism": "LOF",
        "syndrome": "CVID-Type7 / Most-Common-Genetic-CVID-10pct / Haploinsufficiency / Splenomegaly-IBD-DISTINGUISHER",
        "mw_kda": 105.7,
        "critical_rules": [
            "NFKB1-MOST-COMMON-DEFINED-GENETIC-CAUSE-CVID-10PCT",
            "NFKB1-SPLENOMEGALY-AND-IBD-DISTINGUISHERS-AMONG-GENETIC-CVID",
            "NFKB1-AD-HAPLOINSUFFICIENCY-VARIABLE-EXPRESSIVITY-FAMILY-MEMBERS",
            "NFKB1-ADULT-ONSET-COMMON-UNLIKE-AGAMMAGLOBULINEMIA-WHICH-IS-INFANTILE",
        ],
        "distinguishing_feature": "AD haploinsufficiency CVID Type 7; MOST COMMON defined genetic cause of CVID (~10%); adult-onset pattern common (unlike infantile agammaglobulinemia); SPLENOMEGALY prominent DISTINGUISHER; IBD-like colitis (granulomatous) another key distinguisher; variable expressivity/incomplete penetrance within families; B cells present but dysfunctional",
        "first_line_treatment": "IVIG replacement for antibody deficiency; anti-TNF (infliximab) or anti-integrin (vedolizumab) for IBD-like colitis; corticosteroids for autoimmune cytopenias; bronchiectasis physiotherapy + antibiotics; splenomegaly surveillance; lymphoma surveillance; cascade family testing (AD, ~50% offspring risk); avoid live vaccines",
        "pathognomonic": "Adult-onset CVID (low IgG/IgA/IgM + recurrent sinopulmonary infections) + SPLENOMEGALY + IBD-like colitis + family history of CVID (AD) = NFKB1 haploinsufficiency. Sequence NFKB1 in all genetic CVID workup — most common finding at ~10% of genetic cases.",
        "onset_months_mean": 60.0,
        "penetrance_pct": 70,
        "b_cells_absent_pct": 2,
        "agammaglobulinemia_pct": 70,
        "sinopulmonary_pct": 85,
        "gi_infection_pct": 40,
        "enteroviral_pct": 20,
        "autoimmunity_pct": 50,
        "lymphoproliferation_pct": 80,
        "bronchiectasis_pct": 50,
        "ivig_responsive_pct": 78,
        "hsct_benefit_pct": 35,
        "dx_lag_yrs": 8.0,
    },
]


def _simulate_cohort(gene_data, seed):
    rng = random.Random(seed)
    n = 40
    rows = []
    for _ in range(n):
        row = {
            "b_cells_absent": rng.random() < gene_data["b_cells_absent_pct"] / 100,
            "agammaglobulinemia": rng.random() < gene_data["agammaglobulinemia_pct"] / 100,
            "sinopulmonary": rng.random() < gene_data["sinopulmonary_pct"] / 100,
            "gi_infection": rng.random() < gene_data["gi_infection_pct"] / 100,
            "enteroviral": rng.random() < gene_data["enteroviral_pct"] / 100,
            "autoimmunity": rng.random() < gene_data["autoimmunity_pct"] / 100,
            "lymphoproliferation": rng.random() < gene_data["lymphoproliferation_pct"] / 100,
            "bronchiectasis": rng.random() < gene_data["bronchiectasis_pct"] / 100,
            "ivig_responsive": rng.random() < gene_data["ivig_responsive_pct"] / 100,
            "hsct_benefit": rng.random() < gene_data["hsct_benefit_pct"] / 100,
            "onset_months": max(1.0, rng.gauss(gene_data["onset_months_mean"], gene_data["onset_months_mean"] * 0.35)),
            "dx_lag": max(0.5, rng.gauss(gene_data["dx_lag_yrs"], gene_data["dx_lag_yrs"] * 0.40)),
        }
        rows.append(row)
    return rows


def generate_overview():
    cohorts = []
    for i, g in enumerate(ATLAS_GENES):
        seed = SEED_BASE + i
        rows = _simulate_cohort(g, seed)
        n = len(rows)
        cohorts.append({
            "gene": g["name"],
            "n": n,
            "syndrome": g["syndrome"],
            "inheritance": g["inheritance"],
            "mechanism": g["mechanism"],
            "aa": g["aa"],
            "locus": g["locus"],
            "mw_kda": g["mw_kda"],
            "critical_rules": g["critical_rules"],
            "pathognomonic": g["pathognomonic"],
            "first_line_treatment": g["first_line_treatment"],
            "distinguishing_feature": g["distinguishing_feature"],
            "mean_onset_months": round(sum(r["onset_months"] for r in rows) / n, 1),
            "mean_dx_lag_yrs": round(sum(r["dx_lag"] for r in rows) / n, 1),
            "b_cells_absent_n": sum(1 for r in rows if r["b_cells_absent"]),
            "agammaglobulinemia_n": sum(1 for r in rows if r["agammaglobulinemia"]),
            "sinopulmonary_n": sum(1 for r in rows if r["sinopulmonary"]),
            "gi_infection_n": sum(1 for r in rows if r["gi_infection"]),
            "enteroviral_n": sum(1 for r in rows if r["enteroviral"]),
            "autoimmunity_n": sum(1 for r in rows if r["autoimmunity"]),
            "lymphoproliferation_n": sum(1 for r in rows if r["lymphoproliferation"]),
            "bronchiectasis_n": sum(1 for r in rows if r["bronchiectasis"]),
            "ivig_responsive_n": sum(1 for r in rows if r["ivig_responsive"]),
            "hsct_benefit_n": sum(1 for r in rows if r["hsct_benefit"]),
        })

    return {
        "atlas": "Hereditary-Agammaglobulinemia-CVID-Atlas",
        "subtitle": "B-Cell and Antibody Deficiency Primary Immunodeficiencies — Complete 8-Gene Reference",
        "total_patients": sum(c["n"] for c in cohorts),
        "genes": [c["gene"] for c in cohorts],
        "cohort_size": 40,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "cohorts": cohorts,
        "key_facts": {
            "BTK_XLA_MOST_COMMON_85PCT_XL": "BTK-XLA is the most common B-cell deficiency (85% of agammaglobulinemia). X-linked: only males affected. ALL B cells absent (<1% peripheral lymphocytes). ALL immunoglobulins undetectable. First PID described (Colonel Ogden Bruton, 1952). IVIG trough ≥800 mg/dL lifelong.",
            "ECHOVIRUS_ENCEPHALITIS_BTK_PATHOGNOMONIC": "Echovirus (enterovirus) meningoencephalitis is PATHOGNOMONIC for agammaglobulinemia (especially BTK-XLA). Chronic CNS enteroviral infection → progressive dermo-myo-encephalitis. Screen all agammaglobulinemia patients with neurological symptoms: CSF PCR + stool PCR for enteroviruses.",
            "FEMALE_XLA_PHENOTYPE_SEQUENCE_IGHM": "If FEMALE has XLA-like phenotype (absent B cells + agammaglobulinemia + bacterial infections): BTK gene is X-linked and will NOT be the cause. Sequence IGHM first (AR Agamma Type 1), then IGLL1, BLNK. IGHM is most common AR agammaglobulinemia gene.",
            "APDS1_PIK3CD_B_CELLS_PRESENT": "APDS1 (PIK3CD-GOF): B cells are PRESENT but functionally defective — naive B cells accumulate, memory B cells reduced. This is the KEY DISTINGUISHER from classical agammaglobulinemia (BTK/IGHM/IGLL1/BLNK) where B cells are absent. EBV/CMV susceptibility and lymphoproliferation are additional distinguishers.",
            "LENIOLISIB_FDA2023_APDS1_APDS2": "Leniolisib (JOENJA) is the first FDA-approved (2023) specific PI3Kδ inhibitor for APDS1 (PIK3CD-GOF) and APDS2 (PIK3R1-LOF). It directly targets the overactive PI3Kδ pathway — disease-modifying, not just replacement therapy. PI3Kδ mediates PIP3 accumulation that drives T-cell exhaustion and B-cell maturation failure.",
            "LRBA_ABATACEPT_DISEASE_MODIFYING": "LRBA deficiency: abatacept (CTLA4-Ig, Orencia) is HIGHLY EFFECTIVE and disease-modifying. LRBA is required to recycle CTLA4 from lysosomes to cell surface — without LRBA, CTLA4 is degraded. Abatacept substitutes for absent CTLA4 function. Autoimmunity (ITP, IBD, hepatitis, ILD) is the dominant clinical feature, not just antibody deficiency.",
            "NFKB1_MOST_COMMON_GENETIC_CVID": "NFKB1 haploinsufficiency is the MOST COMMON defined genetic cause of CVID (~10% of genetic CVID cases). AD with variable expressivity and incomplete penetrance. Adult-onset pattern common. SPLENOMEGALY and IBD-like granulomatous colitis are DISTINGUISHERS from other genetic CVID causes.",
            "AGAMMAGLOBULINEMIA_PANEL_ORDER": "When evaluating agammaglobulinemia (absent B cells + undetectable immunoglobulins): First test BTK (85% — if male); if female OR male BTK negative: IGHM → IGLL1 → BLNK. If B cells present with antibody deficiency: check PIK3CD, PIK3R1, LRBA, NFKB1 (CVID-spectrum). CVID genetic panel includes NFKB1, NFKB2, TNFRSF13B (TACI), CTLA4, LRBA, PIK3CD, PIK3R1.",
        },
        "diagnostic_approach": {
            "Step1_B_Cell_Count": "Flow cytometry: CD19+/CD20+ B cells. If <1% of peripheral lymphocytes: true agammaglobulinemia (BTK/IGHM/IGLL1/BLNK panel). If B cells present (>1%) with antibody deficiency: CVID-spectrum (PIK3CD/PIK3R1/LRBA/NFKB1/TACI/CTLA4 panel).",
            "Step2_Sex_and_Inheritance": "Male with absent B cells: BTK first (XLA, 85%); if BTK negative: IGHM/IGLL1/BLNK. Female with absent B cells: BTK is NOT the cause — sequence IGHM directly. Family history (AD pattern): NFKB1, PIK3CD, CTLA4, TNFRSF13B (TACI).",
            "Step3_Clinical_Distinguishers": "Pure antibody deficiency + absent B cells + bacterial sinopulmonary infections = agammaglobulinemia panel. Antibody deficiency + EBV/CMV + lymphoproliferation = APDS1/2. Antibody deficiency + autoimmunity (ITP/IBD/hepatitis) = LRBA/CTLA4. Antibody deficiency + splenomegaly + IBD = NFKB1.",
            "Step4_Echovirus_Surveillance": "All patients with agammaglobulinemia (especially BTK): stool PCR for enteroviruses + CSF PCR if any neurological symptoms (headache, cognitive change, muscle weakness, skin induration) — echovirus encephalitis is underdiagnosed and fatal if untreated; IVIG trough ≥800 mg/dL protects.",
            "Step5_CTLA4_Flow_Cytometry": "In suspected LRBA deficiency: measure CTLA4 surface expression on T cells by flow cytometry after stimulation — reduced CTLA4 surface expression (with normal CTLA4 gene/mRNA) = LRBA deficiency. CTLA4 gene defects also reduce CTLA4 surface expression but by different mechanism.",
        },
    }


def generate_breakdown():
    genes_data = []
    for i, g in enumerate(ATLAS_GENES):
        seed = SEED_BASE + i
        rows = _simulate_cohort(g, seed)
        n = len(rows)
        genes_data.append({
            "rank": i + 1,
            "gene": g["name"],
            "full_name": g["full_name"],
            "aa": g["aa"],
            "locus": g["locus"],
            "inheritance": g["inheritance"],
            "mechanism": g["mechanism"],
            "mw_kda": g["mw_kda"],
            "syndrome": g["syndrome"],
            "n": n,
            "mean_onset_months": round(sum(r["onset_months"] for r in rows) / n, 1),
            "mean_dx_lag_yrs": round(sum(r["dx_lag"] for r in rows) / n, 1),
            "b_cells_absent_n": sum(1 for r in rows if r["b_cells_absent"]),
            "agammaglobulinemia_n": sum(1 for r in rows if r["agammaglobulinemia"]),
            "sinopulmonary_n": sum(1 for r in rows if r["sinopulmonary"]),
            "gi_infection_n": sum(1 for r in rows if r["gi_infection"]),
            "enteroviral_n": sum(1 for r in rows if r["enteroviral"]),
            "autoimmunity_n": sum(1 for r in rows if r["autoimmunity"]),
            "lymphoproliferation_n": sum(1 for r in rows if r["lymphoproliferation"]),
            "bronchiectasis_n": sum(1 for r in rows if r["bronchiectasis"]),
            "ivig_responsive_n": sum(1 for r in rows if r["ivig_responsive"]),
            "hsct_benefit_n": sum(1 for r in rows if r["hsct_benefit"]),
            "pathognomonic": g["pathognomonic"],
            "first_line_treatment": g["first_line_treatment"],
            "distinguishing_feature": g["distinguishing_feature"],
            "critical_rules": g["critical_rules"],
        })
    return {"atlas": "Hereditary-Agammaglobulinemia-CVID-Atlas", "genes": genes_data,
            "total_patients": sum(g["n"] for g in genes_data)}


def generate_definitions():
    return {
        "atlas": "Hereditary-Agammaglobulinemia-CVID-Atlas",
        "definitions": {
            "BTK / XLA — Bruton's Agammaglobulinemia, X-linked, Echovirus Encephalitis PATHOGNOMONIC": (
                "BTK (Bruton's tyrosine kinase) encodes a TEC-family non-receptor tyrosine kinase expressed "
                "throughout B-cell development. BTK is activated downstream of the pre-BCR and mature BCR "
                "by Lyn and Syk kinases → BTK phosphorylates BLNK/SLP-65 → recruits and activates PLCγ2 "
                "→ DAG + IP3 → PKCβ + Ca2+ release → NF-κB + NFAT activation → B-cell survival and "
                "differentiation signals. Without BTK: B lymphocyte development is arrested at the "
                "pro-B cell stage — essentially no mature B cells exit the bone marrow. "
                "DISCOVERY: Colonel Ogden Bruton described the first PID in 1952 — a boy with recurrent "
                "pneumococcal infections and absent gamma-globulin band on serum protein electrophoresis "
                "(electrophoresis was new at the time). The disease is named Bruton's agammaglobulinemia. "
                "BTK gene was cloned in 1993 (Vetrie et al., Nature 1993). "
                "X-LINKED INHERITANCE: BTK is on chromosome Xq22.1 — hemizygous males have disease "
                "(one mutant copy = disease); heterozygous females are carriers (normal B cells by "
                "lyonisation — the wild-type X is preferentially used in B cells). Thus: females CANNOT "
                "have BTK-XLA (they have a normal BTK allele). "
                "PATHOGNOMONIC — ECHOVIRUS MENINGOENCEPHALITIS: Enteroviruses (echovirus, poliovirus, "
                "coxsackievirus) require antibody-mediated neutralisation for clearance from the CNS. "
                "Without IgG, echoviruses chronically infect the CNS → progressive encephalitis + "
                "skin/muscle involvement (dermatomyositis-like syndrome = dermo-myo-encephalitis). "
                "This is specific to agammaglobulinemia (BTK, IGHM, IGLL1, BLNK) — not seen in T-cell "
                "deficiencies. Screen with stool PCR + CSF PCR; IVIG trough ≥800 mg/dL prevents this. "
                "IVIG TREATMENT: human normal immunoglobulin (IVIG or SCIG) replacement is the mandatory "
                "lifelong treatment. Target trough ≥800 mg/dL (many centres target ≥1000 mg/dL in "
                "patients with chronic lung disease or recent infections)."
            ),
            "IGHM / IGLL1 / BLNK — Autosomal Recessive Agammaglobulinemia Panel": (
                "These three genes each cause autosomal recessive agammaglobulinemia by disrupting "
                "sequential steps in the pre-BCR assembly and signalling pathway: "
                "IGHM (Igμ heavy chain): the Ig heavy chain is the first component to be synthesised "
                "in B-cell development. Without the Igμ heavy chain, the pre-BCR complex cannot be "
                "assembled — B cells cannot progress beyond the pro-B stage. IGHM is at chromosome "
                "14q32.33 in the immunoglobulin heavy chain locus cluster. FIRST gene to sequence in "
                "females with XLA-like phenotype. "
                "IGLL1 (lambda5 surrogate light chain): The pre-BCR uses a 'fake' lambda-like light "
                "chain (lambda5/IGLL1 + VpreB) instead of a true light chain during differentiation — "
                "this allows pro-B cells to test Igμ heavy chain folding before light chain gene "
                "rearrangement. Without lambda5: pre-BCR assembly fails (the heavy chain alone cannot "
                "fold/signal correctly without the surrogate light chain partner). IGLL1 is at "
                "chromosome 22q11.23. Smallest protein in the panel at 214aa (24.2 kDa). "
                "BLNK (SLP-65, B-cell linker): BLNK is the scaffold adaptor downstream of BTK. "
                "Once the pre-BCR signals: Lyn + Syk activate → phospho-BLNK creates a docking "
                "platform that simultaneously binds activated BTK + PLCγ2 — this co-recruitment is "
                "essential for PLCγ2 activation. Without BLNK: BTK is present and active but cannot "
                "deliver its phosphorylation to PLCγ2 → signalling aborted → pro-B arrest. "
                "BLNK deficiency is therefore a FUNCTIONAL PHENOCOPY of BTK despite being downstream. "
                "All three AR agammaglobulinemia genes: phenotype identical to BTK-XLA; IVIG lifelong; "
                "AR (25% sibling recurrence); affects males AND females (critical distinction from BTK)."
            ),
            "PIK3CD / PIK3R1 — APDS1 and APDS2, Activated PI3K-delta Syndrome": (
                "The PI3K-delta enzyme complex consists of two subunits: "
                "PIK3CD (p110δ): the catalytic subunit that phosphorylates PIP2 to PIP3 at the plasma "
                "membrane; expressed selectively in haematopoietic cells. "
                "PIK3R1 (p85α): the regulatory subunit that normally INHIBITS p110δ in the resting state "
                "(prevents constitutive PI3K activity when no receptor stimulation is present). "
                "APDS1 (PIK3CD gain-of-function, AD): Common hotspot variants E1021K and E525K — these "
                "disrupt the p110δ-p85α interface, constitutively activating PI3Kδ. PIP3 accumulates "
                "→ Akt/mTOR/S6K1 are constitutively active. "
                "APDS2 (PIK3R1 loss-of-function, AR): loss of p85α inhibitory function → p110δ "
                "disinhibited → same PIP3/Akt/mTOR hyperactivation as APDS1 despite opposite "
                "genetic mechanism (GOF vs LOF, AD vs AR). "
                "DOWNSTREAM CONSEQUENCES (shared APDS1 and APDS2): "
                "(1) B-CELL MATURATION DEFECT: naive B cells accumulate (unable to differentiate into "
                "memory cells due to mTOR-driven anergy); class-switched memory B cells nearly absent; "
                "poor vaccine responses; hypogammaglobulinaemia. NOTE: B cells are PRESENT (unlike XLA). "
                "(2) T-CELL EXHAUSTION: CD8+ T cells terminally differentiated early (mTOR-driven "
                "premature senescence); impaired killing of EBV, CMV — viral lymphoproliferation. "
                "(3) BRONCHIECTASIS: from recurrent sinopulmonary infections + structural lung damage. "
                "(4) LYMPHOPROLIFERATION + LYMPHOMA: EBV-driven, as impaired T-cell surveillance. "
                "LENIOLISIB (JOENJA): first FDA-approved (April 2023) specific PI3Kδ inhibitor for "
                "APDS1 and APDS2. Targets the constitutively active PI3Kδ enzyme directly. "
                "Disease-modifying: reduces lymphoproliferation, improves B-cell maturation, reduces "
                "infections. Idelalisib is broader PI3K inhibitor (p110δ + p110α, more toxicity)."
            ),
            "LRBA Deficiency — CTLA4 Recycling Failure, Abatacept Substitutes": (
                "LRBA (Lipopolysaccharide-responsive beige-like anchor protein) is a BEACH-domain "
                "protein involved in vesicular trafficking — specifically, LRBA directs recycling "
                "endosomes to retrieve transmembrane proteins (particularly CTLA4) from late "
                "endosomes/multivesicular bodies back to the cell surface, rescuing them from "
                "lysosomal degradation. "
                "WITHOUT LRBA: CTLA4 that has been internalised from the T-cell surface (after "
                "CD80/CD86 engagement) cannot be recycled — it is diverted to lysosomes and "
                "degraded. CTLA4 surface expression is therefore dramatically reduced. "
                "(Note: the CTLA4 GENE itself is intact and CTLA4 mRNA/protein is normally produced "
                "in LRBA deficiency — the gene is not mutated; the problem is trafficking/recycling.) "
                "CTLA4 (CTLA4 surface) normally functions as: (1) T-cell co-inhibitory receptor "
                "(competes with CD28 for CD80/CD86 binding; higher affinity than CD28); (2) Treg "
                "effector mechanism — CTLA4 on Tregs trans-endocytoses CD80/CD86 from APCs, removing "
                "the co-stimulatory signal for nearby T effector cells. "
                "Without surface CTLA4: T-cell activation is unrestrained → multi-organ autoimmunity. "
                "PHENOTYPE: autoimmune cytopenias (ITP, AIHA, neutropenia); IBD-like colitis (with "
                "lymphocytic infiltrates and villous atrophy); autoimmune hepatitis; interstitial lung "
                "disease (LIP/follicular bronchiolitis); thyroiditis; Type 1 diabetes. "
                "PLUS antibody deficiency (CVID-like: low IgG/IgA/IgM, poor vaccine responses) + "
                "lymphoproliferation (organomegaly, lymphadenopathy, EBV-driven LPD). "
                "ABATACEPT (Orencia; CTLA4-Ig fusion protein): consists of CTLA4 extracellular domain "
                "fused to IgG1 Fc — binds CD80/CD86 on APCs, directly substituting for absent CTLA4 "
                "surface function → halts T-cell hyperactivation → dramatic improvement in autoimmune "
                "manifestations. Highly effective; subcutaneous abatacept preferred for long-term use. "
                "LRBA is the LARGEST protein in the agammaglobulinemia/CVID panel (2863aa, 319 kDa)."
            ),
            "NFKB1 / CVID7 — Most Common Genetic CVID, Haploinsufficiency, Splenomegaly + IBD": (
                "NFKB1 encodes two proteins from the same gene via proteolytic processing: "
                "NF-κB1 p105 (the full-length precursor, also functions as an IκB-like inhibitor "
                "sequestering RelA in the cytoplasm) and NF-κB1 p50 (the mature transcription "
                "factor subunit generated by proteasomal processing of the N-terminal portion). "
                "NF-κB1 p50 forms transcriptionally active heterodimers with RelA (p65) that drive "
                "expression of genes required for B-cell class-switch recombination (AID/AICDA), "
                "germinal centre reactions (BCL6, EZH2), and plasma cell differentiation (BLIMP1/PRDM1). "
                "HAPLOINSUFFICIENCY: ONE functional NFKB1 allele is not sufficient to drive normal "
                "B-cell differentiation responses — p50/p65 levels are reduced → impaired germinal "
                "centre reactions → class-switched memory B cells reduced → CVID phenotype. "
                "PREVALENCE: NFKB1 LOF variants are found in approximately 10% of patients with a "
                "defined genetic cause of CVID — making it the MOST FREQUENTLY identified CVID gene. "
                "PHENOTYPE DISTINGUISHERS from other genetic CVID causes: "
                "(1) SPLENOMEGALY: consistent and often prominent — NF-κB1 normally restrains splenic "
                "B-cell homeostasis; splenic marginal zone hyperplasia common. "
                "(2) IBD-LIKE COLITIS: Crohn's-like granulomatous colitis — NF-κB1 haploinsufficiency "
                "in gut epithelial cells impairs innate immune barrier function, driving chronic "
                "gut inflammation. Anti-TNF (infliximab) or anti-integrin (vedolizumab) for GI disease. "
                "VARIABLE EXPRESSIVITY: penetrance ~70%; some family members carrying the variant "
                "are entirely asymptomatic. This variability makes genetic counselling nuanced."
            ),
            "CVID vs Agammaglobulinemia — Clinical and Immunological Distinction": (
                "These two categories are often confused because both present with antibody deficiency "
                "and recurrent sinopulmonary infections, but they are mechanistically and clinically "
                "distinct: "
                "AGAMMAGLOBULINEMIA (BTK/IGHM/IGLL1/BLNK): "
                "B-cell maturation is ARRESTED — no B cells exit the bone marrow. "
                "Flow cytometry: <1% CD19+ B cells in peripheral blood (virtually absent). "
                "ALL immunoglobulin classes severely reduced (IgG typically <200 mg/dL, often "
                "undetectable). Onset: 6-9 months (when maternal IgG wanes). "
                "Infections: bacterial sinopulmonary (H. influenzae, S. pneumoniae, Pseudomonas, "
                "Staphylococcus) + Giardia lamblia + echovirus risk. "
                "No autoimmunity or lymphoproliferation (no B cells to drive either). "
                "CVID-SPECTRUM (PIK3CD/PIK3R1/LRBA/NFKB1/TNFRSF13B/CTLA4/NFKB2): "
                "B cells are PRESENT but dysfunctional — maturation, class switching, or memory "
                "differentiation is impaired, not blocked. "
                "Flow cytometry: B cells present (CD19+ >1% lymphocytes) but reduced memory/class-switched "
                "subsets; some forms (LRBA, PIK3CD) have near-normal total B-cell counts. "
                "IgG low-moderate (200-500 mg/dL typically); IgA and/or IgM may be variably reduced. "
                "Onset: typically later (childhood or adult-onset vs infantile agammaglobulinemia). "
                "Key CVID features: autoimmunity (LRBA, CTLA4, NFKB1) + lymphoproliferation "
                "(LRBA, PIK3CD, NFKB1) + EBV/CMV susceptibility (PIK3CD, PIK3R1) which are absent "
                "in pure agammaglobulinemia."
            ),
            "IVIG in Antibody Deficiency — Target Trough, Enterovirus Protection": (
                "IVIG (intravenous immunoglobulin) and SCIG (subcutaneous immunoglobulin) are the "
                "cornerstone treatment for all primary antibody deficiencies in this atlas. "
                "MECHANISM: passive transfer of pooled polyclonal IgG from thousands of donors, "
                "providing broad antibody coverage against bacterial and viral pathogens. "
                "TARGET TROUGH: For agammaglobulinemia (BTK, IGHM, IGLL1, BLNK): IgG trough ≥800 "
                "mg/dL (minimum); many centres target ≥1000 mg/dL especially in patients with "
                "bronchiectasis or prior serious infections. Higher troughs reduce infection rate. "
                "For CVID-spectrum: similar target, ≥800 mg/dL. "
                "ENTEROVIRUS/ECHOVIRUS PROTECTION: adequate IVIG trough (≥800 mg/dL) is the "
                "primary strategy to PREVENT echovirus meningoencephalitis. Human IVIG pools "
                "contain neutralising antibodies against all common enterovirus serotypes. "
                "Once echovirus CNS infection is established: IVIG provides some benefit "
                "(reduce viral load) but cannot reverse established CNS damage; intraventricular "
                "IVIG delivery has been attempted in cases with poor CNS penetration. "
                "DOSING: IVIG 400-600 mg/kg every 3-4 weeks, OR SCIG 100-150 mg/kg/week "
                "(subcutaneous home administration, more stable troughs). "
                "LIVE VACCINES CONTRAINDICATED: MMR, varicella, oral polio — these require "
                "functional B/T cells for clearance; passive IVIG antibodies also neutralise "
                "live vaccines rendering them ineffective. Use inactivated vaccines only."
            ),
            "PI3K-delta Inhibition — Leniolisib and the Activated PI3K-delta Syndrome": (
                "Leniolisib (JOENJA, Pharming Group) was approved by the FDA in April 2023 for "
                "adults and paediatric patients ≥12 years with APDS (both APDS1 and APDS2). "
                "It is the first disease-specific small molecule approved for a primary "
                "immunodeficiency. "
                "MECHANISM: Leniolisib is a highly selective PI3Kδ inhibitor (IC50 for PI3Kδ ~0.5 nM; "
                ">200-fold selective over PI3Kα/β/γ). It directly blocks the constitutively active "
                "PI3Kδ in APDS1 (PIK3CD-GOF) and also blocks the disinhibited PI3Kδ in APDS2 "
                "(PIK3R1-LOF, where p85α restraint is lost). "
                "CLINICAL EFFECTS (from registration trial): "
                "(1) Reduces lymphadenopathy and splenomegaly (lymphoproliferation regresses); "
                "(2) Improves naive B-cell differentiation into class-switched memory B cells; "
                "(3) Reduces transitional B-cell counts (hallmark of APDS); "
                "(4) Reduces infection frequency. "
                "IDELALISIB: broader PI3K inhibitor (p110δ primary + p110α secondary); available "
                "before leniolisib; more GI and hepatic toxicity due to broader inhibition. "
                "NOTE ON IVIG IN APDS: IVIG remains indicated for antibody deficiency component "
                "of APDS — leniolisib is disease-modifying but may not fully normalise antibody "
                "production, especially in established bronchiectasis."
            ),
            "ABATACEPT in LRBA Deficiency — Substituting the Absent CTLA4 Checkpoint": (
                "Abatacept (Orencia, BMS-188667) is a fusion protein of the extracellular domain "
                "of human CTLA4 fused to a modified IgG1 Fc. It was originally developed as "
                "a T-cell co-inhibitory drug for rheumatoid arthritis (blocks CD28 co-stimulation "
                "by binding CD80/CD86, competing with the CD28 receptor). "
                "IN LRBA DEFICIENCY: Abatacept is repurposed as a REPLACEMENT for absent CTLA4 — "
                "it directly performs the molecular function that CTLA4 should perform: binding and "
                "trans-endocytosing CD80/CD86 from antigen-presenting cells, removing the "
                "co-stimulatory signal for autoreactive T effector cells. "
                "RESULTS: dramatic improvement in autoimmune manifestations (ITP platelet count "
                "normalises, IBD colitis regresses, hepatitis resolves, ILD improves) in the "
                "majority of LRBA-deficient patients. Often called 'disease-modifying' because "
                "it targets the actual molecular defect. "
                "NOTE: CTLA4 haploinsufficiency (CTLA4 gene mutation) causes a phenotypically "
                "identical syndrome to LRBA deficiency — same mechanism (reduced CTLA4 surface "
                "expression → T-cell hyperactivation → autoimmunity + hypogammaglobulinaemia). "
                "Abatacept is also highly effective in CTLA4 haploinsufficiency (same substitute "
                "CTLA4 function rationale)."
            ),
        },
        "standards": [
            "BTK / XLA — Vetrie D et al. Nature 1993 (BTK cloning); Conley ME et al. J Clin Invest 1994",
            "IGHM agammaglobulinemia — Yel L et al. NEJM 1996",
            "IGLL1 agammaglobulinemia — Minegishi Y et al. Cell 1998",
            "BLNK / SLP-65 agammaglobulinemia — Minegishi Y et al. Immunity 1999",
            "PIK3CD APDS1 — Lucas CL et al. Nat Genet 2014; Angulo I et al. Science 2013",
            "PIK3R1 APDS2 — Deau MC et al. J Clin Invest 2014; Lucas CL et al. Nat Genet 2014",
            "LRBA deficiency — Lopez-Herrera G et al. J Allergy Clin Immunol 2012",
            "LRBA abatacept — Lo B et al. Science 2015; Kiykim A et al. J Clin Immunol 2015",
            "NFKB1 CVID — Fliegauf M et al. Am J Hum Genet 2015",
            "Leniolisib FDA approval 2023 — Rao VK et al. NEJM 2023",
            "Echovirus encephalitis in XLA — Misbah SA et al. QJM 1992; Winkelstein JA et al. Medicine 2006",
            "Primary immunodeficiency classification IUIS 2022; ESID diagnostic criteria CVID 2019",
            "IVIG dosing guidelines — ESID/IPOPI 2018; Orange JS et al. J Clin Immunol 2010",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== Overview ===")
    ov = generate_overview()
    print(f"Total patients: {ov['total_patients']}, genes: {ov['genes']}, seeds: {ov['seeds']}")
    print("=== Breakdown ===")
    bk = generate_breakdown()
    print(f"Genes: {[g['gene'] for g in bk['genes']]}")
    print("=== Definitions ===")
    df = generate_definitions()
    print(f"Definition keys: {list(df['definitions'].keys())[:3]}")
