#!/usr/bin/env python3
"""Hereditary-HIES-Atlas -- Complete 8-Gene Reference
Hyper-IgE Syndrome and Related Primary Immunodeficiencies: Th17/STAT3/cytokine signaling by gene
(STAT3/AD-HIES-Job-Syndrome-Th17-Absent-Skeletal-Retained-Primary-Teeth-PATHOGNOMONIC,
 DOCK8/AR-HIES-Type2-Viral-Infections-Atopy-Food-Allergy-NK-Defect-Lymphoma-Risk,
 PGM3/AR-HIES-Type3-Skeletal-Dysplasia-Eosinophilia-Neutropenia-Hypo-N-Glycosylation,
 CARD11/AD-GOF-BENTA-NF-kB-Th17-Absent-IgE-Elevated-CARD11-Gain-of-Function,
 IL6ST/AR-gp130-Deficiency-Combined-HIES-Hypereosinophilia-Skeletal-Coronary-Aneurysm,
 ZNF341/AR-STAT3-Transcriptional-Regulator-Milder-HIES-Skeletal-Retained-Teeth-Milder,
 IL21R/AR-Combined-Immunodeficiency-CID-Cryptosporidium-Liver-Disease-B-Cell-Loss,
 TYK2/AR-LOF-Mycobacterial-Viral-Candida-Combined-Cytokine-Signaling-IL12-IFN-alpha-IL23):

STAT3  (STAT3 transcription factor; 770aa; 17q21.2; AD dominant-negative;
         Hyper-IgE Syndrome Type 1 (AD-HIES) / Job's Syndrome — MOST COMMON HIES gene;
         STAT3 is the core transcription factor for Th17 differentiation — Th17 cells are virtually
           absent in AD-HIES, impairing IL-17A/F-mediated neutrophil recruitment to sites of infection;
         PATHOGNOMONIC FEATURES (the clinical triad): (1) eczema from birth; (2) recurrent skin/pulmonary
           Staph aureus abscesses ("cold abscesses" — without typical warm/red inflammation due to absent IL-17);
           (3) elevated serum IgE (>2000 IU/mL; often >10,000 IU/mL);
         CONNECTIVE TISSUE / SKELETAL: retained primary teeth (deciduous teeth do not shed normally —
           70-80% of AD-HIES patients); craniosynostosis; hyperextensible joints; scoliosis; tall stature;
           characteristic coarse facial features (frontal bossing, broad nasal bridge, prominent chin);
           These skeletal/connective tissue features are ABSENT in AR-HIES (DOCK8) — key diagnostic separator;
         PULMONARY: recurrent pneumonias → pneumatoceles (pneumatoceles are bronchiectatic cavities filled
           with air or fluid — PATHOGNOMONIC for severe disease); Aspergillus and gram-negative organisms
           colonise pre-existing pneumatoceles;
         NIH Score: ≥40 is diagnostic threshold for AD-HIES; NIH score incorporates all above features;
         STAT3 dominant-negative variants cluster in: DNA-binding domain, SH2 domain, transactivation domain;
         Treatment: prophylactic TMP-SMX + antifungal (itraconazole/voriconazole);
           antihistamines + topical steroids for eczema; HSCT does NOT reliably correct skeletal/CT features;
         seed SEED_BASE+0).
DOCK8  (dedicator of cytokinesis 8; 2099aa; 9p24.3; AR;
         Hyper-IgE Syndrome Type 2 (AR-HIES) — the AR form;
         DOCK8 is a Rho-GEF that activates Cdc42 — essential for NK cell cytotoxicity and T cell
           cytoskeletal reorganisation during immune synapse formation;
         PHENOTYPE DISTINGUISHERS FROM STAT3-HIES:
           (1) VIRAL infections dominate — recurrent HSV (severe/disseminated), HPV (widespread skin warts,
               risk of HPV-related squamous cancers), molluscum contagiosum (extensive), CMV;
           (2) FOOD ALLERGY — anaphylaxis to common foods (much more prominent than AD-HIES);
           (3) MALIGNANCY — B-cell lymphoma/Hodgkin lymphoma risk ~10-15% (NK cell defect + EBV);
           (4) NO skeletal/connective tissue features (no retained primary teeth, no pneumatoceles typical);
           (5) NK cell cytotoxicity severely impaired + CD4/CD8 T cell lymphopenia progressive;
         IgE very elevated (>2000 IU/mL) and eosinophilia present;
         Eczema often severe, beginning in infancy;
         HSCT CURATIVE for immunological phenotype — strongly recommended early in severe cases
           (before EBV-lymphoma or permanent organ damage);
         seed SEED_BASE+1).
PGM3   (phosphoglucomutase 3; 686aa; 6q14.1; AR;
         Hyper-IgE Syndrome Type 3 (AR-HIES) / Congenital Disorder of Glycosylation (CDG-PGM3);
         PGM3 converts N-acetylglucosamine-6-phosphate to N-acetylglucosamine-1-phosphate, required
           for UDP-GlcNAc synthesis — essential for O-GlcNAcylation and N-glycosylation of proteins;
         Reduced UDP-GlcNAc → defective O-GlcNAcylation → impaired STAT3 activation → Th17 absent → HIES;
         PATHOGNOMONIC: HIES phenotype + SKELETAL DYSPLASIA (short stature, spondyloepiphyseal dysplasia,
           abnormal vertebrae, joint laxity — more severe than STAT3-HIES skeletal involvement) + NEUTROPENIA
           (chronic neutropenia, unlike STAT3-HIES which has normal neutrophil count) + EOSINOPHILIA;
         Patients also have: ichthyosis-like skin changes (prominent skin manifestation);
         IgE elevated but often less extreme than STAT3-HIES;
         Neurological: some patients have mild cognitive impairment;
         Treatment: G-CSF for neutropenia; same prophylaxis as HIES; glucosamine supplementation under study;
         seed SEED_BASE+2).
CARD11 (caspase recruitment domain family member 11; 1154aa; 7p22.2; AD gain-of-function;
         BENTA disease (B-cell expansion with NF-κB and T-cell anergy) — GOF variants;
         CARD11 is the scaffolding protein for the CBM complex (CARD11-BCL10-MALT1) that transmits
           TCR/BCR signals to NF-κB activation;
         GOF CARD11 variants (typically in coiled-coil domain) cause CONSTITUTIVE NF-κB signalling
           in B cells → massive B-cell expansion → hypergammaglobulinaemia including very high IgE;
         PHENOTYPE: atopic dermatitis + elevated IgE + food allergy (resembles DOCK8-HIES) + B cell
           lymphocytosis (distinct feature — B cells clonally expanded) + T cell anergy;
         Th17 impaired (NF-κB hyperactivation paradoxically impairs Th17 differentiation);
         DISTINGUISHER: B-cell expansion (lymphocytosis of B cells) — not seen in STAT3 or DOCK8 HIES;
         Treatment: dupilumab for atopic dermatitis; no specific cure; avoid immunosuppression (risk of
           lymphoma with uncontrolled B-cell expansion); rapamycin under trial;
         Biallelic LOF CARD11 causes SCID-like combined immunodeficiency (opposite phenotype);
         seed SEED_BASE+3).
IL6ST  (interleukin-6 signal transducer / gp130; 918aa; 5q11.2; AR;
         gp130 deficiency — combined HIES + coronary artery aneurysm + hypereosinophilia;
         IL6ST encodes gp130, the shared signal transducer for IL-6, IL-11, IL-27, LIF, OSM,
           CT-1 — this shared receptor explains the broad multi-cytokine signalling loss;
         PHENOTYPE (all due to combined loss of IL-6/IL-11/IL-27 signalling):
           (1) HIES features: elevated IgE, eczema, bacterial infections, Th17 absent;
           (2) CORONARY ARTERY ANEURYSMS (Kawasaki-like) in infancy — IL-11/LIF signalling required
               for cardiac vessel integrity; PATHOGNOMONIC for gp130 among HIES genes;
           (3) HYPEREOSINOPHILIA — IL-27 suppresses eosinophils; loss → eosinophil excess;
           (4) SKELETAL: IL-11 signals through gp130 for bone homeostasis — craniosynostosis, retained
               teeth (similar to STAT3-HIES);
           (5) SHORT STATURE — growth hormone resistance (GH signals partially through gp130);
         Echo annually mandatory for coronary aneurysm surveillance;
         Very rare — <50 patients described; diagnosed by sequence of IL6ST gene;
         Treatment: similar to STAT3-HIES; aspirin for coronary aneurysms as per Kawasaki protocol;
         seed SEED_BASE+4).
ZNF341 (zinc finger protein 341; 844aa; 20q11.21; AR;
         Autosomal recessive HIES — ZNF341 is a transcriptional activator of the STAT3 gene itself;
         ZNF341 binds the STAT3 promoter and drives STAT3 transcription; without ZNF341, STAT3 levels
           are profoundly reduced → same downstream phenotype as dominant-negative STAT3 mutations;
         PHENOTYPE: essentially identical to STAT3-HIES but with AR (not AD) inheritance pattern;
           All cardinal STAT3-HIES features present: elevated IgE, eczema, recurrent cold abscesses,
           skeletal features (retained primary teeth, craniosynostosis, scoliosis);
         DISTINGUISHER FROM STAT3-HIES: AR inheritance — both parents carriers (consanguineous families
           enriched); siblings at risk; female carriers asymptomatic (unlike STAT3-HIES where de novo
           dominant-negative variants arise spontaneously in any sex);
         MOLECULAR: STAT3 protein level reduced (not mutated) — protein level measurement (not just
           sequencing) needed; standard NGS sequencing of STAT3 gene is NEGATIVE in ZNF341 deficiency;
         Treatment: same as STAT3-HIES (TMP-SMX + antifungal; skin care; NIH scoring);
         seed SEED_BASE+5).
IL21R  (interleukin-21 receptor; 538aa; 16p12.1; AR;
         IL-21R deficiency — cryptosporidium + liver disease + progressive B-cell loss;
         IL-21 signals through IL-21R + gamma-c chain (IL2RG); IL-21R is selectively important for
           B-cell differentiation, class-switching, germinal centre formation, and mucosal immunity;
         PHENOTYPE (distinct from classical HIES):
           (1) CRYPTOSPORIDIUM INFECTION (biliary cryptosporidiosis leading to progressive sclerosing
               cholangitis and liver failure) — PATHOGNOMONIC for IL-21R deficiency among PIDs;
           (2) PROGRESSIVE B-CELL LOSS — hypogammaglobulinaemia develops over years (initially present,
               then declining — unlike common CVID where B cells often present but non-functional);
           (3) ELEVATED IgE + EOSINOPHILIA — overlaps HIES spectrum superficially;
           (4) GI infections (Giardia, Cryptosporidium) prominent;
         IgE elevated but combined with hypogammaglobulinaemia distinguishes from pure HIES;
         Treatment: IVIG for hypogammaglobulinaemia; nitazoxanide for cryptosporidiosis (limited efficacy);
           HSCT curative if performed before irreversible liver damage;
         seed SEED_BASE+6).
TYK2   (tyrosine kinase 2; 1187aa; 19p13.2; AR loss-of-function;
         TYK2 deficiency — combined cytokine signalling loss IL-12/IFN-α/IL-23/IL-10;
         TYK2 is a JAK-family kinase that signals downstream of: IFN-α/β (type I IFN), IL-12, IL-23,
           IL-10, IL-27, IL-13;
         LOF TYK2 variants (biallelic) impair multiple cytokine pathways simultaneously:
           Loss of IFN-α signalling → severe viral infections (HSV, Varicella, CMV) + mycobacterial disease;
           Loss of IL-12 signalling → Mendelian susceptibility to mycobacterial disease (MSMD) + Salmonella;
           Loss of IL-23 signalling → Th17 absent → recurrent Staph/Candida/mucosal infections;
         IgE elevated (due to Th17 absence + IL-13 dysregulation);
         DISTINGUISHER FROM STAT3-HIES: prominent MYCOBACTERIAL + VIRAL infections;
           no skeletal features (no retained teeth, no pneumatoceles typical);
         P1104A missense variant is HYPOMORPHIC (reduced but not absent TYK2) — surprisingly PROTECTIVE
           against tuberculosis in some populations but increased risk HIES-like if homozygous;
         JAK1 inhibitors (ruxolitinib, baricitinib): benefit in severe inflammatory sequelae but caution
           with infection risk; bimekizumab (IL-17 blockade) under trial;
         seed SEED_BASE+7).
"""
import random
import math

SEED_BASE = 3870

ATLAS_GENES = [
    {
        "name": "STAT3",
        "full_name": "Signal transducer and activator of transcription 3",
        "aa": 770,
        "locus": "17q21.2",
        "inheritance": "AD",
        "mechanism": "DN",
        "syndrome": "AD-HIES Type1 / Job-Syndrome / Th17-Absent / Retained-Primary-Teeth-PATHOGNOMONIC",
        "mw_kda": 88.0,
        "critical_rules": [
            "STAT3-RETAINED-PRIMARY-TEETH-70-80PCT-PATHOGNOMONIC",
            "STAT3-TH17-ABSENT-COLD-ABSCESSES-NO-INFLAMMATION",
            "STAT3-PNEUMATOCELES-POSTINFECTIOUS-ASPERGILLUS-RISK",
            "STAT3-IgE-GT2000-OFTEN-GT10000-NIH-SCORE-GT40-DIAGNOSTIC",
        ],
        "distinguishing_feature": "AD-HIES most common; retained primary teeth 70-80% PATHOGNOMONIC; cold abscesses (no warmth/redness due to absent Th17); pneumatoceles; NIH score ≥40; no HSCT for skeletal features",
        "first_line_treatment": "TMP-SMX (bacterial prophylaxis) + itraconazole/voriconazole (fungal); antihistamines + topical steroids for eczema; NIH-HIES score follow-up; pulmonary surveillance; dental management retained teeth",
        "pathognomonic": "Child with eczema + recurrent 'cold' Staph abscesses (without typical warmth/redness) + IgE >2000 IU/mL + retained primary deciduous teeth = STAT3-HIES",
        "onset_yrs_mean": 1.0,
        "penetrance_pct": 100,
        "ige_gt2000_pct": 100,
        "eosinophilia_pct": 90,
        "cold_abscess_pct": 95,
        "pneumatoceles_pct": 65,
        "retained_teeth_pct": 75,
        "viral_infection_pct": 20,
        "food_allergy_pct": 25,
        "lymphoma_risk_pct": 5,
        "coronary_aneurysm_pct": 0,
        "hsct_benefit_pct": 40,
        "dx_lag_yrs": 2.5,
    },
    {
        "name": "DOCK8",
        "full_name": "Dedicator of cytokinesis 8",
        "aa": 2099,
        "locus": "9p24.3",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "AR-HIES Type2 / Viral-Infections / NK-Defect / Food-Allergy / Lymphoma-10-15pct",
        "mw_kda": 237.0,
        "critical_rules": [
            "DOCK8-VIRAL-HSV-HPV-MOLLUSCUM-DOMINANT-UNLIKE-STAT3",
            "DOCK8-NK-CYTOTOXICITY-SEVERELY-IMPAIRED",
            "DOCK8-LYMPHOMA-EBV-DRIVEN-10-15PCT",
            "DOCK8-HSCT-CURATIVE-RECOMMENDED-EARLY",
        ],
        "distinguishing_feature": "AR-HIES; VIRAL infections dominate (HSV/HPV/molluscum/CMV); food allergy prominent; B/NK cell defects; lymphoma risk 10-15% (EBV-driven); NO skeletal features; HSCT curative",
        "first_line_treatment": "TMP-SMX + antifungal prophylaxis; acyclovir for recurrent HSV; rituximab for EBV-lymphoproliferation; HSCT strongly recommended early (before lymphoma/organ damage); allergen avoidance",
        "pathognomonic": "Child with HIES (eczema + elevated IgE) + severe/recurrent VIRAL infections (HSV/HPV/molluscum) + food allergy + no retained primary teeth = DOCK8 deficiency",
        "onset_yrs_mean": 1.5,
        "penetrance_pct": 100,
        "ige_gt2000_pct": 95,
        "eosinophilia_pct": 85,
        "cold_abscess_pct": 50,
        "pneumatoceles_pct": 10,
        "retained_teeth_pct": 5,
        "viral_infection_pct": 95,
        "food_allergy_pct": 80,
        "lymphoma_risk_pct": 13,
        "coronary_aneurysm_pct": 0,
        "hsct_benefit_pct": 85,
        "dx_lag_yrs": 3.0,
    },
    {
        "name": "PGM3",
        "full_name": "Phosphoglucomutase 3",
        "aa": 686,
        "locus": "6q14.1",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "AR-HIES Type3 / Skeletal-Dysplasia / Neutropenia / Eosinophilia / CDG",
        "mw_kda": 77.0,
        "critical_rules": [
            "PGM3-SKELETAL-DYSPLASIA-SPONDYLOEPIPHYSEAL-PATHOGNOMONIC",
            "PGM3-NEUTROPENIA-CHRONIC-UNLIKE-STAT3-NORMAL-COUNT",
            "PGM3-UDP-GLCNAC-DEFICIENCY-O-GLCNACYLATION-DEFECT",
            "PGM3-GLUCOSAMINE-SUPPLEMENT-UNDER-INVESTIGATION",
        ],
        "distinguishing_feature": "AR-HIES; skeletal dysplasia PATHOGNOMONIC (spondyloepiphyseal + joint laxity); chronic NEUTROPENIA (unlike STAT3-HIES which has normal count); eosinophilia; CDG-type mechanism",
        "first_line_treatment": "G-CSF for neutropenia; TMP-SMX + antifungal prophylaxis; orthopaedic management for skeletal dysplasia; glucosamine supplementation (experimental); dermatology for ichthyotic skin changes",
        "pathognomonic": "HIES phenotype + spondyloepiphyseal skeletal dysplasia + chronic neutropenia + eosinophilia = PGM3 deficiency (test UDP-GlcNAc levels and sequence PGM3)",
        "onset_yrs_mean": 2.0,
        "penetrance_pct": 100,
        "ige_gt2000_pct": 80,
        "eosinophilia_pct": 92,
        "cold_abscess_pct": 60,
        "pneumatoceles_pct": 20,
        "retained_teeth_pct": 30,
        "viral_infection_pct": 35,
        "food_allergy_pct": 40,
        "lymphoma_risk_pct": 3,
        "coronary_aneurysm_pct": 0,
        "hsct_benefit_pct": 60,
        "dx_lag_yrs": 4.5,
    },
    {
        "name": "CARD11",
        "full_name": "Caspase recruitment domain family member 11",
        "aa": 1154,
        "locus": "7p22.2",
        "inheritance": "AD",
        "mechanism": "GOF",
        "syndrome": "AD-GOF-BENTA / NF-kB-Constitutive / B-Cell-Expansion / T-Cell-Anergy / IgE-Elevated",
        "mw_kda": 133.0,
        "critical_rules": [
            "CARD11-GOF-B-CELL-EXPANSION-LYMPHOCYTOSIS-DISTINGUISHER",
            "CARD11-NF-KB-CONSTITUTIVE-ACTIVATION",
            "CARD11-T-CELL-ANERGY-TH17-IMPAIRED",
            "CARD11-LOF-BIALLELIC-OPPOSITE-SCID-PHENOTYPE",
        ],
        "distinguishing_feature": "AD-GOF (BENTA disease); B-cell expansion PATHOGNOMONIC distinguisher from STAT3/DOCK8 HIES; NF-κB constitutive activation; T-cell anergy; IgE very elevated; risk of lymphoma with B-cell clonal expansion",
        "first_line_treatment": "Dupilumab for atopic dermatitis; avoid broad immunosuppression (lymphoma risk); rapamycin mTOR inhibition under trial; anti-IgE omalizumab for IgE-driven symptoms; lymphoma surveillance annually",
        "pathognomonic": "HIES-like (eczema + elevated IgE + food allergy) + B-cell LYMPHOCYTOSIS (B-cell counts elevated >10×10^9/L) + T-cell anergy = CARD11-GOF / BENTA",
        "onset_yrs_mean": 1.5,
        "penetrance_pct": 85,
        "ige_gt2000_pct": 90,
        "eosinophilia_pct": 75,
        "cold_abscess_pct": 30,
        "pneumatoceles_pct": 5,
        "retained_teeth_pct": 5,
        "viral_infection_pct": 40,
        "food_allergy_pct": 85,
        "lymphoma_risk_pct": 10,
        "coronary_aneurysm_pct": 0,
        "hsct_benefit_pct": 50,
        "dx_lag_yrs": 3.5,
    },
    {
        "name": "IL6ST",
        "full_name": "Interleukin-6 signal transducer (gp130)",
        "aa": 918,
        "locus": "5q11.2",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "AR-gp130-Deficiency / Coronary-Aneurysm-PATHOGNOMONIC / Hypereosinophilia / Combined-HIES-Skeletal",
        "mw_kda": 103.0,
        "critical_rules": [
            "IL6ST-CORONARY-ARTERY-ANEURYSM-KAWASAKI-LIKE-PATHOGNOMONIC",
            "IL6ST-COMBINED-IL6-IL11-IL27-LIF-SIGNALING-LOSS",
            "IL6ST-ECHO-ANNUALLY-MANDATORY-CORONARY-SURVEILLANCE",
            "IL6ST-SHORT-STATURE-GH-RESISTANCE-SKELETAL-CRANIOSYNOSTOSIS",
        ],
        "distinguishing_feature": "AR-gp130 deficiency; CORONARY ARTERY ANEURYSMS (Kawasaki-like) PATHOGNOMONIC; hypereosinophilia; skeletal features (craniosynostosis, retained teeth); short stature; multi-cytokine pathway loss (IL-6/IL-11/IL-27/LIF/OSM)",
        "first_line_treatment": "Same as STAT3-HIES (TMP-SMX + antifungal); aspirin/IVIG for coronary aneurysms (as per Kawasaki protocol); annual echocardiogram MANDATORY; IL-6 replacement under study; growth hormone therapy for short stature",
        "pathognomonic": "HIES features + CORONARY ARTERY ANEURYSM (echocardiogram) in infant/child + hypereosinophilia + skeletal features = gp130 (IL6ST) deficiency; sequence IL6ST",
        "onset_yrs_mean": 0.5,
        "penetrance_pct": 100,
        "ige_gt2000_pct": 90,
        "eosinophilia_pct": 98,
        "cold_abscess_pct": 70,
        "pneumatoceles_pct": 30,
        "retained_teeth_pct": 60,
        "viral_infection_pct": 30,
        "food_allergy_pct": 35,
        "lymphoma_risk_pct": 2,
        "coronary_aneurysm_pct": 80,
        "hsct_benefit_pct": 55,
        "dx_lag_yrs": 2.0,
    },
    {
        "name": "ZNF341",
        "full_name": "Zinc finger protein 341",
        "aa": 844,
        "locus": "20q11.21",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "AR-HIES / STAT3-Transcriptional-Activator / Milder-Skeletal / Consanguineous",
        "mw_kda": 95.0,
        "critical_rules": [
            "ZNF341-AR-INHERITANCE-DISTINGUISHES-FROM-STAT3-AD",
            "ZNF341-STAT3-PROTEIN-REDUCED-NOT-MUTATED-LEVEL-ASSAY-NEEDED",
            "ZNF341-NGS-STAT3-SEQUENCING-NEGATIVE-IN-ZNF341-DEFICIENCY",
            "ZNF341-CONSANGUINEOUS-FAMILIES-ENRICHED",
        ],
        "distinguishing_feature": "AR-HIES phenotypically identical to STAT3-HIES; AR inheritance (consanguineous); STAT3 protein level reduced (not mutated) — STAT3 sequencing negative; STAT3 protein level measurement needed",
        "first_line_treatment": "Identical to STAT3-HIES (TMP-SMX + antifungal + skin care + dental management); NIH-HIES scoring; genetic counselling for AR inheritance; sibling testing (25% risk); ZNF341 sequencing when STAT3 sequencing negative + classic HIES phenotype",
        "pathognomonic": "Classic STAT3-HIES phenotype (retained teeth + cold abscesses + elevated IgE) + AR inheritance (consanguineous parents) + STAT3 sequencing NEGATIVE → ZNF341 sequencing",
        "onset_yrs_mean": 1.5,
        "penetrance_pct": 100,
        "ige_gt2000_pct": 95,
        "eosinophilia_pct": 88,
        "cold_abscess_pct": 88,
        "pneumatoceles_pct": 50,
        "retained_teeth_pct": 70,
        "viral_infection_pct": 25,
        "food_allergy_pct": 30,
        "lymphoma_risk_pct": 4,
        "coronary_aneurysm_pct": 0,
        "hsct_benefit_pct": 45,
        "dx_lag_yrs": 5.0,
    },
    {
        "name": "IL21R",
        "full_name": "Interleukin-21 receptor",
        "aa": 538,
        "locus": "16p12.1",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "AR-IL21R-CID / Cryptosporidium-Liver-Disease-PATHOGNOMONIC / Progressive-B-Cell-Loss",
        "mw_kda": 59.0,
        "critical_rules": [
            "IL21R-CRYPTOSPORIDIUM-BILIARY-SCLEROSING-CHOLANGITIS-PATHOGNOMONIC",
            "IL21R-PROGRESSIVE-B-CELL-LOSS-HYPOGAMMAGLOBULINAEMIA-DEVELOPS",
            "IL21R-HSCT-BEFORE-IRREVERSIBLE-LIVER-DAMAGE",
            "IL21R-IVIG-MANDATORY-HYPOGAMMAGLOBULINAEMIA",
        ],
        "distinguishing_feature": "AR-IL21R; CRYPTOSPORIDIUM biliary/liver disease PATHOGNOMONIC; progressive B-cell loss → hypogammaglobulinaemia developing over years; overlaps HIES spectrum (IgE + eosinophilia) but distinct; HSCT curative before liver failure",
        "first_line_treatment": "IVIG replacement for hypogammaglobulinaemia; nitazoxanide for Cryptosporidium (limited efficacy); ursodeoxycholic acid for cholangitis; HSCT curative — URGENTLY recommend before irreversible liver/bile duct damage; liver function monitoring",
        "pathognomonic": "HIES-like features (IgE + eosinophilia) + Cryptosporidium biliary disease + progressive decline in B-cell numbers + hypogammaglobulinaemia = IL-21R deficiency",
        "onset_yrs_mean": 2.0,
        "penetrance_pct": 100,
        "ige_gt2000_pct": 70,
        "eosinophilia_pct": 80,
        "cold_abscess_pct": 35,
        "pneumatoceles_pct": 15,
        "retained_teeth_pct": 5,
        "viral_infection_pct": 60,
        "food_allergy_pct": 40,
        "lymphoma_risk_pct": 8,
        "coronary_aneurysm_pct": 0,
        "hsct_benefit_pct": 80,
        "dx_lag_yrs": 4.0,
    },
    {
        "name": "TYK2",
        "full_name": "Tyrosine kinase 2",
        "aa": 1187,
        "locus": "19p13.2",
        "inheritance": "AR",
        "mechanism": "LOF",
        "syndrome": "AR-TYK2-LOF / Combined-Cytokine-Loss-IL12-IFN-alpha-IL23 / Mycobacterial-Viral-Candida",
        "mw_kda": 134.0,
        "critical_rules": [
            "TYK2-MYCOBACTERIAL-MSMD-PLUS-VIRAL-INFECTIONS-COMBINED",
            "TYK2-P1104A-HYPOMORPHIC-VARIANT-CONTEXT-DEPENDENT",
            "TYK2-IFN-ALPHA-SIGNALING-LOSS-SEVERE-VIRAL-SUSCEPTIBILITY",
            "TYK2-IL23-LOSS-TH17-ABSENT-MUCOSAL-CANDIDA-STAPH",
        ],
        "distinguishing_feature": "AR-TYK2-LOF; combined IL-12/IFN-α/IL-23 signalling loss → mycobacterial (MSMD) + viral (HSV/VZV/CMV) + Candida infections; IgE elevated (IL-13 dysregulation); NO skeletal features; P1104A hypomorphic variant common",
        "first_line_treatment": "IFN-γ for mycobacterial infections (supplementing lost IL-12→IFN-γ axis); antifungals for Candida; acyclovir/antivirals for herpesvirus; TMP-SMX prophylaxis; ruxolitinib for severe inflammatory sequelae; avoid BCG vaccination (mycobacterial risk)",
        "pathognomonic": "HIES-like IgE elevation + MYCOBACTERIAL infections (BCG disease/NTM/TB) + severe viral infections (HSV/VZV/CMV) + Candida = TYK2 deficiency (sequence TYK2, especially exon 22 P1104A in European populations)",
        "onset_yrs_mean": 2.5,
        "penetrance_pct": 95,
        "ige_gt2000_pct": 75,
        "eosinophilia_pct": 65,
        "cold_abscess_pct": 45,
        "pneumatoceles_pct": 20,
        "retained_teeth_pct": 5,
        "viral_infection_pct": 90,
        "food_allergy_pct": 30,
        "lymphoma_risk_pct": 6,
        "coronary_aneurysm_pct": 0,
        "hsct_benefit_pct": 65,
        "dx_lag_yrs": 5.5,
    },
]


def _simulate_cohort(gene_data, seed):
    rng = random.Random(seed)
    n = 40
    rows = []
    for _ in range(n):
        row = {
            "ige_elevated": rng.random() < gene_data["ige_gt2000_pct"] / 100,
            "eosinophilia": rng.random() < gene_data["eosinophilia_pct"] / 100,
            "cold_abscess": rng.random() < gene_data["cold_abscess_pct"] / 100,
            "pneumatoceles": rng.random() < gene_data["pneumatoceles_pct"] / 100,
            "retained_teeth": rng.random() < gene_data["retained_teeth_pct"] / 100,
            "viral_infection": rng.random() < gene_data["viral_infection_pct"] / 100,
            "food_allergy": rng.random() < gene_data["food_allergy_pct"] / 100,
            "lymphoma": rng.random() < gene_data["lymphoma_risk_pct"] / 100,
            "coronary_aneurysm": rng.random() < gene_data["coronary_aneurysm_pct"] / 100,
            "hsct_benefit": rng.random() < gene_data["hsct_benefit_pct"] / 100,
            "onset_age": max(0.2, rng.gauss(gene_data["onset_yrs_mean"], gene_data["onset_yrs_mean"] * 0.4)),
            "dx_lag": max(0.5, rng.gauss(gene_data["dx_lag_yrs"], gene_data["dx_lag_yrs"] * 0.4)),
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
            "mean_onset_age": round(sum(r["onset_age"] for r in rows) / n, 1),
            "mean_dx_lag_yrs": round(sum(r["dx_lag"] for r in rows) / n, 1),
            "ige_elevated_n": sum(1 for r in rows if r["ige_elevated"]),
            "eosinophilia_n": sum(1 for r in rows if r["eosinophilia"]),
            "cold_abscess_n": sum(1 for r in rows if r["cold_abscess"]),
            "pneumatoceles_n": sum(1 for r in rows if r["pneumatoceles"]),
            "retained_teeth_n": sum(1 for r in rows if r["retained_teeth"]),
            "viral_infection_n": sum(1 for r in rows if r["viral_infection"]),
            "food_allergy_n": sum(1 for r in rows if r["food_allergy"]),
            "lymphoma_n": sum(1 for r in rows if r["lymphoma"]),
            "coronary_aneurysm_n": sum(1 for r in rows if r["coronary_aneurysm"]),
            "hsct_benefit_n": sum(1 for r in rows if r["hsct_benefit"]),
        })

    return {
        "atlas": "Hereditary-HIES-Atlas",
        "subtitle": "Hyper-IgE Syndrome & Related Primary Immunodeficiencies — Complete 8-Gene Reference",
        "total_patients": sum(c["n"] for c in cohorts),
        "genes": [c["gene"] for c in cohorts],
        "cohort_size": 40,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "cohorts": cohorts,
        "key_facts": {
            "STAT3_AD_HIES_MOST_COMMON": "STAT3 dominant-negative is the most common HIES gene. Cardinal triad: eczema + cold Staph abscesses (absent Th17 = no IL-17-driven neutrophil recruitment = no warmth/redness) + IgE >2000. NIH HIES score ≥40 = diagnostic.",
            "RETAINED_PRIMARY_TEETH_STAT3_ZNF341": "Retained deciduous (primary) teeth that do not shed: 70-80% of STAT3-HIES and ZNF341-HIES. PATHOGNOMONIC for STAT3-spectrum HIES. Absent in DOCK8 and TYK2-HIES — key diagnostic separator.",
            "DOCK8_VIRAL_NOT_BACTERIAL_DOMINANT": "DOCK8-AR-HIES is dominated by VIRAL infections (HSV/HPV/molluscum/CMV) + food allergy, unlike STAT3-HIES where bacterial cold abscesses dominate. NK cell cytotoxicity severely impaired. HSCT curative.",
            "IL6ST_CORONARY_ANEURYSM_PATHOGNOMONIC": "gp130 (IL6ST) deficiency: coronary artery aneurysms (Kawasaki-like) occurring in infancy = PATHOGNOMONIC distinguisher. Annual echocardiogram MANDATORY. Multiple cytokine pathways lost (IL-6/IL-11/IL-27/LIF).",
            "ZNF341_STAT3_SEQUENCING_NEGATIVE": "ZNF341 deficiency = AR phenocopy of STAT3-HIES. STAT3 gene sequencing is NEGATIVE (STAT3 protein level reduced, not mutated). If classic HIES + AR inheritance + STAT3 negative → sequence ZNF341.",
            "IL21R_CRYPTOSPORIDIUM_LIVER_URGENT": "IL-21R deficiency: Cryptosporidium biliary disease → progressive sclerosing cholangitis PATHOGNOMONIC. HSCT must occur BEFORE irreversible liver damage. IVIG for hypogammaglobulinaemia (B cells progressively lost).",
            "TYK2_MYCOBACTERIAL_VIRAL_COMBINED": "TYK2-LOF: combined loss of IL-12/IFN-α/IL-23 signalling → mycobacterial (MSMD) + severe viral + Candida infections. IgE elevated (IL-13 dysregulation). BCG vaccination ABSOLUTELY contraindicated.",
            "CARD11_GOF_B_CELL_EXPANSION": "CARD11-GOF (BENTA): B-cell LYMPHOCYTOSIS distinguishes from all other HIES. NF-κB constitutive activation. T-cell anergy. Lymphoma risk with uncontrolled B-cell clonal expansion. Avoid broad immunosuppression.",
        },
        "diagnostic_approach": {
            "Step1_Measure_IgE_Eosinophilia": "Serum IgE: >2000 IU/mL in most HIES. Eosinophilia (AEC >1500). Calculate NIH HIES score (score ≥40 = classic AD-HIES). Note: IgE >2000 is necessary but not sufficient — also seen in DOCK8, TYK2, CARD11.",
            "Step2_Distinguish_AD_vs_AR_HIES": "AD vs AR inheritance: STAT3 and CARD11-GOF are AD (spontaneous or family history); DOCK8/PGM3/IL6ST/ZNF341/IL21R/TYK2 are AR (consanguineous families, siblings). De novo AD STAT3 variants common.",
            "Step3_Viral_vs_Bacterial_Dominance": "If VIRAL infections dominate (HSV/HPV/molluscum) + food allergy: DOCK8 (AR). If bacterial cold abscesses + retained teeth + pneumatoceles: STAT3/ZNF341 (AD or AR). If mycobacterial + viral combined: TYK2.",
            "Step4_Additional_Features": "Skeletal dysplasia + neutropenia: PGM3. Coronary aneurysm: IL6ST. B-cell lymphocytosis: CARD11-GOF. Cryptosporidium + liver disease + B-cell decline: IL21R. NIH score: STAT3/ZNF341.",
            "Step5_Molecular_Testing_Pathway": "Panel sequencing: STAT3 (AD), ZNF341 (if STAT3 negative + AR + classic phenotype), DOCK8, PGM3, IL6ST, IL21R, TYK2, CARD11. STAT3 protein level measurement if sequencing negative + classic HIES (ZNF341 = reduced STAT3 protein). Flow: Th17 cells (absent in STAT3/ZNF341), NK cytotoxicity (impaired in DOCK8).",
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
            "mean_onset_age": round(sum(r["onset_age"] for r in rows) / n, 1),
            "mean_dx_lag_yrs": round(sum(r["dx_lag"] for r in rows) / n, 1),
            "ige_elevated_n": sum(1 for r in rows if r["ige_elevated"]),
            "eosinophilia_n": sum(1 for r in rows if r["eosinophilia"]),
            "cold_abscess_n": sum(1 for r in rows if r["cold_abscess"]),
            "pneumatoceles_n": sum(1 for r in rows if r["pneumatoceles"]),
            "retained_teeth_n": sum(1 for r in rows if r["retained_teeth"]),
            "viral_infection_n": sum(1 for r in rows if r["viral_infection"]),
            "food_allergy_n": sum(1 for r in rows if r["food_allergy"]),
            "lymphoma_n": sum(1 for r in rows if r["lymphoma"]),
            "coronary_aneurysm_n": sum(1 for r in rows if r["coronary_aneurysm"]),
            "hsct_benefit_n": sum(1 for r in rows if r["hsct_benefit"]),
            "pathognomonic": g["pathognomonic"],
            "first_line_treatment": g["first_line_treatment"],
            "distinguishing_feature": g["distinguishing_feature"],
            "critical_rules": g["critical_rules"],
        })
    return {"atlas": "Hereditary-HIES-Atlas", "genes": genes_data,
            "total_patients": sum(g["n"] for g in genes_data)}


def generate_definitions():
    return {
        "atlas": "Hereditary-HIES-Atlas",
        "definitions": {
            "STAT3 / AD-HIES / Job's Syndrome — Th17 Absent, Cold Abscesses, Retained Teeth": (
                "STAT3 encodes Signal Transducer and Activator of Transcription 3, the master transcription "
                "factor for Th17 cell differentiation from naive CD4+ T cells. Dominant-negative STAT3 "
                "variants (in SH2, DNA-binding, or transactivation domain) impair IL-6/IL-21/IL-23 signalling "
                "→ Th17 cells virtually absent → no IL-17A/F production → neutrophils not recruited to skin/lung "
                "infection sites → 'cold' abscesses (pus without warmth, redness, or pain — because IL-17 "
                "drives CXCL1/CXCL8 chemokines that recruit neutrophils acutely). "
                "The three cardinal diagnostic criteria are: (1) eczema from birth; (2) recurrent 'cold' "
                "Staph aureus abscesses; (3) serum IgE >2000 IU/mL (often >10,000-100,000 IU/mL). "
                "Non-immune connective tissue/skeletal features (STAT3 drives matrix metalloproteinases): "
                "retained primary deciduous teeth (70-80%; deciduous teeth fail to resorb), craniosynostosis, "
                "hyperextensible joints, scoliosis, coarse facial features (frontal bossing, broad nasal "
                "bridge, prominent mandible), tall stature. "
                "Pulmonary: post-infectious pneumatoceles (air-fluid cysts after Staph/gram-negative pneumonia) "
                "colonised by Aspergillus → invasive fungal disease. "
                "NIH HIES Score: ≥40 is diagnostic threshold (incorporates all clinical and laboratory features). "
                "Treatment: TMP-SMX + azole antifungal prophylaxis; topical steroids and antihistamines "
                "for eczema; dental management of retained primary teeth; pulmonary surveillance. "
                "HSCT: corrects immune phenotype but does NOT reverse skeletal/connective tissue features "
                "(not driven by haematopoietic cells) — limited benefit unless severe immunological disease."
            ),
            "DOCK8 / AR-HIES — Viral Infections, NK Defect, Lymphoma Risk, HSCT Curative": (
                "DOCK8 encodes a guanine nucleotide exchange factor (GEF) for Cdc42 in haematopoietic cells. "
                "DOCK8 is essential for NK cell cytotoxicity (immunological synapse formation), CD8+ T cell "
                "killing, and long-lived memory B cell survival (B cells lacking DOCK8 do not survive in "
                "lymphoid organs). "
                "PHENOTYPE DISTINGUISHERS from STAT3-HIES (critical differential): "
                "(1) VIRAL infections DOMINATE: severe recurrent HSV-1/2 (disseminated, multi-dermatomal), "
                "widespread HPV-related warts (risk of HPV-driven squamous cell carcinoma), extensive "
                "molluscum contagiosum refractory to treatment, CMV disease; "
                "(2) FOOD ALLERGY: anaphylaxis to common foods (milk, egg, nuts) in >70% — much more "
                "prominent than STAT3-HIES; "
                "(3) LYMPHOMA RISK: B-cell lymphoma/Hodgkin lymphoma in 10-15% (EBV-driven; NK cell defect "
                "+ impaired cytotoxic T cells = impaired EBV immune surveillance); "
                "(4) ABSENT skeletal/connective tissue features (no retained primary teeth, no pneumatoceles "
                "as typical STAT3 feature, no craniosynostosis) — key separator; "
                "(5) NK cytotoxicity severely impaired; CD4/CD8 T-cell lymphopenia progressive. "
                "HSCT is curative for the immunological phenotype (NK and T cell reconstitution) and is "
                "STRONGLY RECOMMENDED EARLY, before lymphoma development or end-organ damage. "
                "DOCK8 is the second most common HIES gene; largest gene in the HIES panel (2099aa; 9p24.3)."
            ),
            "PGM3 / AR-HIES Type3 — Skeletal Dysplasia + Neutropenia + CDG Mechanism": (
                "PGM3 encodes phosphoglucomutase-3, which catalyses the interconversion of "
                "N-acetylglucosamine-6-phosphate and N-acetylglucosamine-1-phosphate, the precursor for "
                "UDP-GlcNAc synthesis. UDP-GlcNAc is the sugar-nucleotide donor for O-GlcNAcylation and "
                "N-glycosylation of cytoplasmic, nuclear, and secretory proteins. "
                "Reduced UDP-GlcNAc → impaired STAT3 O-GlcNAcylation → reduced STAT3 activation → "
                "Th17 deficiency → HIES phenotype. This makes PGM3-HIES a congenital disorder of "
                "glycosylation (CDG) causing HIES. "
                "PATHOGNOMONIC FEATURES distinguishing PGM3 from STAT3-HIES: "
                "(1) SKELETAL DYSPLASIA: spondyloepiphyseal dysplasia (abnormal vertebral and epiphyseal "
                "development), short stature, joint laxity (more severe than STAT3 skeletal involvement); "
                "(2) CHRONIC NEUTROPENIA: STAT3-HIES has NORMAL neutrophil count — neutropenia is the "
                "biochemical clue that UDP-GlcNAc deficiency impairs granulopoiesis; "
                "(3) EOSINOPHILIA: often marked (AEC >3000); "
                "(4) ICHTHYOTIC SKIN CHANGES: dry, ichthyosis-like skin, in addition to eczema. "
                "Treatment: G-CSF for neutropenia; same prophylaxis as HIES; glucosamine supplementation "
                "(substrate for UDP-GlcNAc synthesis) under investigation as disease-modifying therapy."
            ),
            "CARD11 GOF / BENTA Disease — B-Cell Expansion, NF-κB, T-Cell Anergy": (
                "CARD11 (CARMA1) is the scaffolding protein of the CBM complex (CARD11-BCL10-MALT1) that "
                "couples T-cell receptor and B-cell receptor signals to canonical NF-κB activation. "
                "Gain-of-function CARD11 variants (typically in the coiled-coil domain, e.g., G116S, "
                "R35Q, G126D) cause CONSTITUTIVE NF-κB signalling in lymphocytes independent of antigen "
                "stimulation. In B cells: constitutive NF-κB → massive B-cell proliferation and expansion "
                "→ hypergammaglobulinaemia including very high IgE. In T cells: PARADOXICALLY impaired "
                "T-cell signalling → T-cell anergy (constitutive signalling exhausts the T cell). "
                "BENTA disease (B-cell expansion with NF-κB and T-cell anergy): named for the cardinal "
                "features. KEY DISTINGUISHER from all other HIES: B-CELL LYMPHOCYTOSIS (B cells >10×10^9/L). "
                "No other HIES gene causes this B-cell clonal expansion. "
                "Th17 deficiency present (as in STAT3-HIES) but mechanism: T-cell anergy prevents Th17 "
                "differentiation, not absent STAT3 transcriptional activation. "
                "OPPOSITE PHENOTYPE: biallelic LOF CARD11 variants cause SCID-like combined immune "
                "deficiency (no B or T cell signalling). Distinction: GOF = BENTA (gain), LOF = SCID-like. "
                "Treatment: dupilumab (IL-4Ra blockade) for atopic dermatitis; rapamycin (mTOR inhibition) "
                "under clinical trial; lymphoma surveillance annually; avoid broad immunosuppression."
            ),
            "IL6ST / gp130 Deficiency — Coronary Aneurysm PATHOGNOMONIC, Multi-Cytokine Loss": (
                "IL6ST encodes gp130 (interleukin-6 signal transducer), the common β-signal transducing "
                "receptor subunit shared by ALL IL-6 family cytokines: IL-6, IL-11, IL-27, LIF (leukaemia "
                "inhibitory factor), OSM (oncostatin M), CT-1, CNTF. When gp130 is lost, ALL these cytokines "
                "fail to signal, causing a multi-system phenotype. "
                "KEY PATHOGNOMONIC DISTINGUISHER from all other HIES: CORONARY ARTERY ANEURYSMS. "
                "IL-11 and LIF signal through gp130 to maintain coronary arterial wall integrity — without "
                "gp130, coronary arteries develop Kawasaki disease-like aneurysms in infancy. "
                "Annual echocardiogram is MANDATORY; aspirin and IVIG therapy as per modified Kawasaki protocol. "
                "Additional features: hypereosinophilia (IL-27 normally suppresses eosinophils — absent "
                "suppression → eosinophil excess); skeletal: craniosynostosis + retained teeth (IL-11 "
                "signals through gp130 for bone turnover — retained teeth as in STAT3-HIES); short stature "
                "(growth hormone resistance — GH partially signals through gp130-containing receptor complexes). "
                "HIES features (eczema, cold abscesses, elevated IgE, absent Th17) present: IL-27 through "
                "gp130 normally amplifies Th17 differentiation. "
                "Rarest HIES gene: <50 patients globally. Sequence IL6ST in HIES + coronary aneurysm."
            ),
            "ZNF341 / AR-HIES — STAT3 Promoter Activator, STAT3 Protein Reduced, Sequencing Negative": (
                "ZNF341 encodes zinc finger protein 341, a transcriptional activator that directly binds the "
                "STAT3 gene promoter and activates STAT3 transcription. ZNF341 was identified as a cause "
                "of HIES in 2018 (Beziat et al., NEJM 2018). "
                "Without ZNF341: STAT3 mRNA and protein levels are profoundly reduced (typically <20% of "
                "normal) — the STAT3 gene is intact and unmodified, but its transcription is impaired. "
                "RESULT: same downstream Th17 deficiency and clinical phenotype as STAT3 dominant-negative "
                "variants — eczema, cold abscesses, elevated IgE, retained primary teeth, skeletal features. "
                "CRITICAL MOLECULAR DIAGNOSTIC POINT: "
                "When clinician orders NGS sequencing of the STAT3 gene → NEGATIVE (no pathogenic STAT3 variant). "
                "If phenotype is classic STAT3-HIES + AR inheritance (consanguineous family) + STAT3 sequencing "
                "negative → STAT3 PROTEIN LEVEL MEASUREMENT is next step (will show markedly reduced "
                "STAT3 protein) → then ZNF341 sequencing → diagnostic. "
                "This sequencing trap is a known clinical pitfall: many ZNF341 patients are misdiagnosed "
                "as 'STAT3-negative HIES of unknown cause' when ZNF341 is not reflexively sequenced. "
                "Inheritance: strictly AR. Parents are obligate heterozygous carriers (phenotypically normal). "
                "Sibling recurrence risk: 25%."
            ),
            "IL21R / CID — Cryptosporidium Biliary Disease PATHOGNOMONIC, HSCT Before Liver Failure": (
                "IL-21R deficiency (IL21R biallelic LOF) causes combined immunodeficiency with a distinctive "
                "PATHOGNOMONIC presentation: Cryptosporidium parvum biliary infection leading to progressive "
                "primary sclerosing cholangitis (PSC) and liver failure. "
                "IL-21 signals through IL-21R + common γ-chain (IL2RG, shared with IL-2/IL-4/IL-7/IL-9/IL-15 "
                "receptors) to promote: B-cell germinal centre reactions and class switching; plasma cell "
                "differentiation; CD4+ T-cell Th17 and Tfh differentiation; NK cell maturation. "
                "PHENOTYPE: Cryptosporidium infection of the biliary tree → progressive biliary strictures "
                "and sclerosing cholangitis (bile duct destruction) → liver failure over years; this biliary "
                "cryptosporidiosis is also seen in CVID and CD40L deficiency but is MOST characteristic of "
                "IL-21R among HIES-spectrum diseases. "
                "PROGRESSIVE B-CELL LOSS: B-cell counts initially normal in early infancy but decline "
                "progressively → hypogammaglobulinaemia requires IVIG. Distinguishes from most HIES "
                "forms where B cells are preserved. "
                "HSCT URGENCY: Must perform before irreversible liver damage — cryptosporidium-driven "
                "biliary destruction is irreversible. Early HSCT achieves normal immune reconstitution "
                "and prevents further hepatic/biliary progression. "
                "IVIG replacement for hypogammaglobulinaemia; nitazoxanide for Cryptosporidium treatment "
                "(limited efficacy in immune deficiency; disease control, not cure)."
            ),
            "TYK2 / LOF — Combined IL-12/IFN-α/IL-23 Loss, Mycobacterial + Viral + Candida": (
                "TYK2 (Tyrosine Kinase 2) is a Janus kinase (JAK family member) that is the mandatory "
                "signalling partner for: (1) IFN-α/β receptor (IFNAR1/2) — type I interferon signalling → "
                "antiviral innate immunity; (2) IL-12 receptor (IL12RB1/2) → Th1 differentiation → "
                "mycobacterial killing; (3) IL-23 receptor (IL23R) → Th17 maintenance → antifungal/"
                "antibacterial mucosal immunity; (4) IL-10 receptor; (5) IL-13 receptor. "
                "COMBINED CYTOKINE LOSS: unlike most PID genes that affect one pathway, TYK2 LOF impairs "
                "multiple independent cytokine axes simultaneously. "
                "CLINICAL CONSEQUENCES: "
                "Loss of IL-12/Th1 signalling → Mendelian susceptibility to mycobacterial disease (MSMD): "
                "disseminated BCG disease after vaccination, NTM infections, TB at low bacillary load. "
                "BCG VACCINATION IS ABSOLUTELY CONTRAINDICATED in TYK2-deficient infants; "
                "Loss of IFN-α signalling → severe viral infections: HSV (disseminated), VZV (severe), "
                "CMV disease; "
                "Loss of IL-23 signalling → Th17 absent → mucosal Candida + Staph + HIES features "
                "(IgE elevated). "
                "P1104A VARIANT: relatively common hypomorphic TYK2 variant in European populations "
                "(minor allele frequency ~2-4% in northern Europeans) — reduces TYK2 kinase activity "
                "partially. Homozygous P1104A: milder HIES-like phenotype. Context-dependent: P1104A "
                "homozygous protective against auto-inflammatory diseases (reduced JAK2 signalling) but "
                "increases susceptibility to intracellular organisms."
            ),
            "NIH HIES Scoring System — Clinical Diagnostic Tool for AD-HIES (STAT3/ZNF341)": (
                "The NIH HIES (Hyper-IgE Syndrome) Score is a validated clinical scoring tool specifically "
                "designed to identify patients with AD-HIES (STAT3 or ZNF341 deficiency). "
                "Score ≥40 is the diagnostic threshold for likely AD-HIES. "
                "Scored domains (maximum points per domain): "
                "Serum IgE level (max 10): IgE >2000 IU/mL = 10 points; IgE 200-2000 = 5 points. "
                "Skin abscesses (max 10): ≥4 episodes = 10 points. "
                "Pneumonia (max 10): ≥3 episodes = 10 points. "
                "Lung abnormality — pneumatoceles or bronchiectasis (max 8): present = 8 points. "
                "Retained primary teeth (max 8): ≥2 retained teeth = 8 points. "
                "Scoliosis (max 4): >30 degrees = 4 points. "
                "Eczema (max 5): requiring treatment = 5 points. "
                "Characteristic face (max 3). "
                "High palate (max 2). "
                "Fracture (max 2). "
                "Midline abnormality (max 2). "
                "IMPORTANT: The NIH score was designed for STAT3-spectrum (AD-HIES). It does NOT reliably "
                "identify DOCK8-HIES (AR-HIES) where skeletal features are absent and viral infections "
                "dominate. DOCK8 patients often score <40 despite severe immunological disease. "
                "Always sequence both STAT3 and DOCK8 (and other HIES genes) regardless of NIH score "
                "if clinical suspicion is present."
            ),
            "HIES vs ATOPIC DERMATITIS — Distinguishing Severe Atopy from Primary Immunodeficiency": (
                "Severe atopic dermatitis + elevated IgE ≠ HIES. The key differentiators are: "
                "(1) INFECTION PATTERN: In simple atopy, Staph colonisation is present but systemic deep "
                "abscesses are rare. In HIES: visceral/pulmonary abscesses, pneumonias, pneumatoceles — "
                "infection burden beyond typical eczema. "
                "(2) SITE OF ABSCESSES: 'Cold' abscesses in HIES (minimal inflammation — no warmth/redness) "
                "vs typical warm/tender skin abscesses in eczema with secondary infection. "
                "(3) LEVEL OF IgE: Pure atopy rarely exceeds 5000 IU/mL; STAT3-HIES commonly >10,000-100,000. "
                "(4) NON-IMMUNE FEATURES: retained primary teeth, craniosynostosis, hyperextensible joints "
                "— absent in atopy. "
                "(5) VIRAL INFECTIONS: severe/recurrent HSV, molluscum, HPV warts → DOCK8. "
                "(6) FAMILY HISTORY: AD inheritance for STAT3-HIES; AR (consanguineous) for DOCK8/ZNF341. "
                "(7) FLOW CYTOMETRY: absent Th17 cells (IL-17A-producing CD4+ T cells) by intracellular "
                "cytokine staining after PMA+ionomycin stimulation. Normal in atopy; absent in STAT3-HIES. "
                "The NIH HIES score provides a structured approach to this differential."
            ),
            "HSCT in HIES — When, For Whom, and Limitations by Gene": (
                "HSCT (haematopoietic stem cell transplantation) recommendations differ significantly by HIES gene: "
                "DOCK8 (AR-HIES): STRONGLY RECOMMENDED and curative for immune phenotype. Reverse NK defect, "
                "T-cell lymphopenia, and food allergy. Prevents lymphoma. Perform before lymphoma or "
                "end-organ damage. Reduced-intensity conditioning (RIC) preferred. "
                "IL21R: URGENTLY INDICATED before irreversible liver/biliary damage from Cryptosporidium. "
                "TYK2: Indicated for severe disease; curative for immune phenotype. "
                "STAT3 (AD-HIES): HSCT corrects immunological abnormalities (Th17 reconstitution) and reduces "
                "infections, but does NOT correct skeletal/connective tissue features (these are driven by "
                "STAT3 activity in non-haematopoietic cells — osteoblasts, fibroblasts). Retained teeth and "
                "scoliosis persist post-HSCT. HSCT reserved for very severe, refractory disease. "
                "PGM3: corrects neutropenia and immune defects; skeletal dysplasia may persist. "
                "CARD11-GOF (BENTA): HSCT corrects B-cell expansion and T-cell anergy; prevents lymphoma. "
                "IL6ST: HSCT corrects immune phenotype; coronary aneurysm surveillance continues post-HSCT."
            ),
        },
        "standards": [
            "AD-HIES (STAT3) — Holland SM et al. NEJM 2007; Minegishi Y et al. Nature 2007",
            "DOCK8 deficiency — Zhang Q et al. J Exp Med 2009; Engelhardt KR et al. J Allergy Clin Immunol 2009",
            "PGM3 HIES — Zhang Y et al. NEJM 2014",
            "CARD11 GOF BENTA — Snow AL et al. J Exp Med 2012",
            "IL6ST (gp130) deficiency — Schwerd T et al. J Exp Med 2017",
            "ZNF341 HIES — Beziat V et al. NEJM 2018; Frey-Jakobs S et al. Sci Immunol 2018",
            "IL21R deficiency — Kotlarz D et al. NEJM 2013",
            "TYK2 deficiency — Minegishi Y et al. Immunity 2006; Kreins AY et al. J Exp Med 2015",
            "NIH HIES Score — Grimbacher B et al. J Immunol 1999",
            "HIES HSCT outcomes — Gatz SA et al. Haematologica 2011",
            "Primary immunodeficiency classification IUIS 2022",
            "DOCK8 HSCT outcomes — Ghosh S et al. J Allergy Clin Immunol 2012",
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
