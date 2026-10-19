#!/usr/bin/env python3
"""Hereditary-LAD-DADA2-Lectin-Complement-Granulocyte-Defect-Atlas -- Complete 8-Gene Reference
Leukocyte Adhesion Deficiency (LAD-I/III) + DADA2 Vasculopathy + Specific Granule Deficiency + CXCR2 SCN6 + Lectin Pathway (MBL2-MASP2-COLEC11)
(ITGB2/CD18-LAD-I-Absent-CD11-CD18-Delayed-Cord-Separation-PATHOGNOMONIC-Omphalitis-Periodontitis-HSCT-Curative,
 FERMT3/Kindlin-3-LAD-III-All-Integrin-Activation-Failure-Combined-Infectious-Bleeding-Glanzmann-Like-PATHOGNOMONIC-HSCT,
 ADA2/CECR1-DADA2-Vasculopathy-PAN-Like-Stroke-Livedo-PATHOGNOMONIC-Anti-TNF-CURATIVE-NOT-HSCT,
 CEBPE/C-EBPe-Specific-Granule-Deficiency-Bilobed-PMN-Nuclei-PATHOGNOMONIC-Absent-Lactoferrin-Gelatinase,
 CXCR2-SCN6-Chemotaxis-Failure-IL8-Receptor-Absent-Neutropenia-Skin-UTI-G-CSF-Effective,
 MBL2-Mannose-Binding-Lectin-Deficiency-Childhood-Infections-Partial-Immunodeficiency-Lectin-Pathway,
 MASP2-Lectin-Pathway-Effector-Absent-Pyogenic-Infections-RA-Like-CH50-AH50-Normal-MBL-Pathway-Absent,
 COLEC11/CL-K1-3MC-Syndrome-Craniosynostosis-Hearing-Loss-Cleft-Palate-PATHOGNOMONIC-Lectin-Complement):

ITGB2 (Integrin beta-2 / CD18; 769aa; 21q22.3; AR;
         ITGB2 encodes the common β2 integrin subunit (CD18) shared by CD11a/LFA-1 (αLβ2),
           CD11b/Mac-1 (αMβ2), CD11c/p150,95 (αXβ2), and CD11d (αDβ2);
         Without CD18: all four β2-integrin heterodimers fail surface expression → neutrophils
           cannot bind ICAM-1/ICAM-2 on activated endothelium → complete failure of leukocyte
           rolling-arrest-extravasation cascade;
         LEUKOCYTE ADHESION DEFICIENCY TYPE I (LAD-I): AR defect; most common LAD type (~65% of all LAD);
         PATHOGNOMONIC: DELAYED UMBILICAL CORD SEPARATION (>3-4 weeks) — periumbilical infection
           without adequate PMN infiltration → umbilical cord persists; omphalitis; this is the classic
           neonatal presentation; coincident LEUKOCYTOSIS (WBC 50-100k) WITHOUT PUS FORMATION
           (neutrophils cannot exit blood into tissues despite extreme numbers in circulation);
         RECURRENT INFECTIONS: skin (Staphylococcus aureus, Pseudomonas aeruginosa), mucosal
           (perirectal abscesses, perineal cellulitis), periodontitis (severe juvenile periodontitis
           destroying bone by age 20-30 in survivors), pneumonia;
         SEVERITY GRADES: Severe (<1% CD18 surface) = neonatal omphalitis, early death without HSCT;
           Moderate (1-10% CD18) = recurrent severe infections; Mild (>10% CD18) = periodontitis + mild infections;
         ABSENCE OF ICAM-1 BINDING: LFA-1 (αLβ2) is required for T-lymphocyte migration and NK cell killing —
           LAD-I patients also have impaired NK cell cytotoxicity and T-cell trafficking defects;
         DIAGNOSIS: CD18 surface expression by flow cytometry (CD18/CD11b staining) = absent or markedly reduced;
           CD18 absent = PATHOGNOMONIC; CD45 normal (to confirm leukocyte viability); ITGB2 sequencing;
         TREATMENT: prophylactic TMP-SMX + antifungal; G-CSF limited benefit in LAD-I
           (neutrophils present, cannot extravasate — G-CSF makes more non-functional PMNs);
           HSCT = ONLY CURATIVE TREATMENT (conditioning corrects CD18 expression in all hematopoietic cells);
           gene therapy: lentiviral ITGB2 ex vivo correction — clinical trials ongoing; seed SEED_BASE+0).
FERMT3 (Fermitin family homolog 3 / Kindlin-3; 667aa; 11q13.1; AR;
         FERMT3 encodes Kindlin-3, the hematopoietic-specific member of the Kindlin family;
         Kindlin-3 is an intracellular integrin co-activator: it binds the cytoplasmic tail of β1, β2,
           and β3 integrin subunits and cooperates with talin to convert integrins from LOW- to HIGH-affinity
           (inside-out activation);
         Without Kindlin-3: β2 integrins (CD18/CD11) cannot undergo inside-out activation →
           leukocyte adhesion deficiency phenotype (same as LAD-I functionally, but CD18 EXPRESSION IS NORMAL);
           β3 integrins (αIIbβ3 on platelets) cannot activate → Glanzmann thrombasthenia-like platelet defect;
         LEUKOCYTE ADHESION DEFICIENCY TYPE III (LAD-III): AR defect; much rarer than LAD-I;
         PATHOGNOMONIC: COMBINED INFECTIONS + BLEEDING — leukocyte adhesion failure (like LAD-I:
           delayed cord separation, omphalitis, periodontitis, recurrent skin/mucosal infections)
           PLUS PLATELET DYSFUNCTION BLEEDING (like Glanzmann thrombasthenia: mucosal bleeding,
           prolonged bleeding time, normal platelet count but absent ADP-induced aggregation);
         KEY DISTINCTION FROM LAD-I: CD18 SURFACE EXPRESSION NORMAL in LAD-III (ITGB2 gene intact,
           protein expressed) — but FUNCTION absent (β2 integrins cannot activate); requires
           activation assay (Mn2+ or activating antibody stimulation) + FERMT3 sequencing;
         MOLECULAR MECHANISM: kindlin-3 binds β-integrin NPXY/F motif → enables talin head to
           flip the integrin transmembrane domain → high-affinity extended conformation;
           without kindlin-3, integrins stay in low-affinity bent-closed conformation;
         DIAGNOSIS: normal CD18 flow cytometry (DDx LAD-I), then β2-integrin activation assay
           (CD11b upregulation after PMA/fMLP stimulation — absent in LAD-III), platelet aggregation
           (absent with ADP, collagen, thrombin despite normal GPIIb/IIIa), FERMT3 sequencing;
         TREATMENT: HSCT MANDATORY AND CURATIVE (only treatment); platelet transfusions for bleeding;
           prophylactic antibiotics; G-CSF not effective (same mechanism as LAD-I); seed SEED_BASE+1).
ADA2   (Adenosine deaminase 2 / CECR1; 511aa; 22q11.1; AR;
         ADA2 (previously CECR1) encodes Adenosine deaminase 2, a secreted enzyme that converts
           adenosine to inosine in the extracellular microenvironment;
         ADA2 also functions as a monocyte/macrophage growth factor promoting M2 (anti-inflammatory)
           macrophage polarization and vascular endothelial homeostasis;
         Without ADA2: accumulation of extracellular adenosine → hyperactivation of A2A adenosine
           receptors on monocytes → dysregulated M1 macrophage skewing → destructive vasculitis;
           also loss of ADA2's trophic function for vascular pericytes → endothelial fragility;
         DADA2 = DEFICIENCY OF ADA2: described in 2014 by Navon Elkan et al. and Zhou et al.
           (simultaneous discovery in PAN cohorts and immune deficiency cohorts);
         PATHOGNOMONIC: POLYARTERITIS NODOSA-LIKE VASCULOPATHY + EARLY-ONSET STROKE (< 10 years,
           often < 5 years) + LIVEDO RETICULARIS (reticular discoloration of skin from cutaneous vessel
           inflammation); this clinical triad in any child should prompt ADA2 testing immediately;
         NK CELL DEFICIENCY: NK cell lymphopenia (often < 5% of normal) — distinguishes DADA2 from
           classic PAN (which has normal NK cells); CD4/CD8 ratio inverted; hypogammaglobulinemia;
         SYSTEMIC INFLAMMATION: hepatosplenomegaly; pancytopenia from bone marrow involvement
           (stromal defect from macrophage dysregulation); elevated ESR, CRP, ferritin;
         TREATMENT — CRITICAL: TNF-INHIBITORS (etanercept 0.4 mg/kg twice weekly, adalimumab
           40 mg every 2 weeks) ARE CURATIVE FOR VASCULOPATHY — prevent stroke and organ damage;
           START ANTI-TNF IMMEDIATELY when DADA2 diagnosed to prevent irreversible CNS damage;
           HSCT is curative for severe cytopenias and refractory cases (not first-line for vasculopathy);
           IVIG for hypogammaglobulinemia; ADA2 enzyme replacement experimental;
         DDx: classic PAN (ANCA-negative, adult onset, no NK deficiency), Kawasaki disease
           (coronary arteritis, younger, self-limiting), Cogan syndrome (ocular + vestibular);
         GENETIC HETEROGENEITY: >60 variants known; founder variants in Georgian, Turkish, Japanese
           populations; p.G47R and p.Y453C common; seed SEED_BASE+2).
CEBPE  (CCAAT/enhancer-binding protein epsilon / C/EBPε; 317aa; 14q11.2; AR;
         CEBPE encodes C/EBPε, a myeloid-specific CCAAT/enhancer-binding transcription factor;
         C/EBPε is expressed exclusively in myeloid cells and drives transcription of genes encoding
           secondary (specific) granule proteins: lactoferrin, collagenase (MMP-8), vitamin B12-binding
           protein (transcobalamin I), gelatinase (MMP-9), and tertiary granule protein-encoding genes;
         Without C/EBPε: neutrophil secondary granule biogenesis COMPLETELY ABSENT; tertiary granules absent;
           primary (azurophil) granules with myeloperoxidase REMAIN INTACT (MPO is not C/EBPε dependent);
         SPECIFIC GRANULE DEFICIENCY (SGD): very rare AR primary neutrophil disorder;
         PATHOGNOMONIC: BILOBED PMN NUCLEI on blood film (failure of normal multilobed segmentation
           = the PMN nuclear hypersegmentation program requires intact specific granule signals;
           ABSENT LACTOFERRIN, VITAMIN B12-BINDING PROTEIN, GELATINASE in PMNs by granule protein assay
           (lactoferrin by ELISA in PMN lysate is absent) = confirmatory; NBT and DHR tests NORMAL
           (NADPH oxidase from primary granules intact — this distinguishes SGD from CGD);
         INFECTIONS: Staphylococcal abscesses (liver, lymph nodes, perirectal), Aspergillus pneumonia,
           gram-negative (Pseudomonas, Klebsiella) septicemia; same pathogens as CGD but NBT/DHR normal;
         DIFFERENTIAL from CGD: NBT/DHR NORMAL in SGD (not CGD); bilobed nuclei on blood film
           (not seen in CGD); lactoferrin absent by granule assay; CEBPE sequencing;
         DIAGNOSIS: blood smear (bilobed nuclei), PMN granule protein assay (lactoferrin absent),
           DHR flow (normal — unlike CGD), CEBPE sequencing;
         TREATMENT: prophylactic TMP-SMX (PCP + bacterial); antifungal prophylaxis (itraconazole/voriconazole);
           G-CSF variable benefit; HSCT for severe cases (corrects granule biogenesis); seed SEED_BASE+3).
CXCR2  (C-X-C chemokine receptor type 2 / IL-8 receptor B / CD182; 360aa; 2q35; AR;
         CXCR2 encodes the receptor for IL-8 (CXCL8) and related CXC chemokines (CXCL1, CXCL2,
           CXCL3, CXCL5, CXCL6 — the "GRO chemokines") on the surface of neutrophils;
         CXCR2 mediates both neutrophil MOBILIZATION from bone marrow (IL-8 gradient from marrow niches)
           and neutrophil CHEMOTAXIS toward sites of infection;
         Without CXCR2: neutrophils cannot exit bone marrow sinusoids (CXCR2 opposes CXCR4
           bone-marrow-retention signals) → PERIPHERAL NEUTROPENIA with NORMAL OR ELEVATED MARROW
           NEUTROPHIL RESERVE; also cannot migrate toward CXCL8 gradient at infection sites;
         SEVERE CONGENITAL NEUTROPENIA TYPE 6 (SCN6) / CXCR2 DEFICIENCY: very rare AR SCN variant;
           described in Middle Eastern and European families; ANC typically 0-500 cells/μL;
         INFLAMMATORY SKIN DISEASE: neutrophilic dermatosis (pyoderma gangrenosum-like or Sweet
           syndrome-like skin lesions) from dysregulated neutrophil retention and activation in dermis;
         RECURRENT URINARY TRACT INFECTIONS: CXCL1/CXCL8 drives PMN trafficking into urine;
           without CXCR2, uroepithelial PMN chemotaxis absent → recurrent E. coli UTIs;
         DIAGNOSIS: neutropenia + normal/elevated marrow myeloid reserve (rules out production failure);
           absent CXCR2 on PMN surface by flow (CD182 antibody); biallelic CXCR2 variants;
         TREATMENT: G-CSF EFFECTIVE (G-CSFR signaling bypasses CXCR2; mobilizes marrow reserves via
           a CXCR2-independent pathway) — G-CSF normalizes ANC and reduces infections;
           ruxolitinib (JAK1/2i) used for inflammatory skin disease; HSCT for refractory severe cases;
           seed SEED_BASE+4).
MBL2   (Mannose-binding lectin 2; 244aa; 10q11.22; AR (functional deficiency via structural variants);
         MBL2 encodes mannose-binding lectin (MBL), a collagen-domain lectin (collectin family) secreted
           by the liver into plasma as oligomers (dimers/trimers/tetramers/hexamers);
         MBL binds carbohydrate patterns on bacterial/fungal/viral surfaces (mannose, N-acetylglucosamine,
           fucose arrays in geometric patterns) → recruits MASP1 and MASP2 serine proteases → MASP2
           cleaves C4 and C2 → forms C3 convertase → LECTIN COMPLEMENT PATHWAY ACTIVATION;
         MBL DEFICIENCY: functional deficiency caused by structural variants at codons 52 (p.Arg52Cys),
           54 (p.Gly54Asp), 57 (p.Gly57Glu) → monomeric/unstable MBL fails to oligomerize →
           cannot activate MASP1/MASP2; promoter variants (H/L, X/Y) affect expression level;
         MBL2 deficiency is a PARTIAL IMMUNODEFICIENCY: alone insufficient to cause severe disease;
           clinically significant in synergy with agammaglobulinemia/CVID, during immunosuppression,
           in infancy (before maternal IgG wanes), or with neutropenia;
         INFECTIONS: recurrent otitis media, pneumococcal pneumonia, respiratory syncytial virus,
           recurrent sinopulmonary infections, opportunistic infections in immunosuppressed patients;
         CH50 and AH50: NORMAL (classical and alternative pathways intact); only MBL-pathway assay
           (mannan-coated plate complement activation) reduced; MBL serum level < 100 ng/mL;
         TREATMENT: prophylactic antibiotics in high-risk periods (infancy, immunosuppression);
           recombinant MBL replacement trials (rMBL2); IVIG (contains MBL) partial benefit;
           fresh frozen plasma as MBL source in acute settings; seed SEED_BASE+5).
MASP2  (MBL-associated serine protease 2; 686aa; 1p36.22; AR;
         MASP2 encodes MBL-associated serine protease 2, the primary effector enzyme of the lectin
           complement pathway; MASP2 is recruited to oligomeric MBL or ficolin (FCN1/FCN2/FCN3)
           pattern-recognition molecules after carbohydrate-binding;
         MASP2 is the SOLE enzyme responsible for cleaving C4 and C2 in the lectin complement pathway:
           MASP2 cleaves C4 → C4a + C4b; MASP2 cleaves C2 → C2a + C2b; C4b2a = C3 convertase;
         Without MASP2: MBL and ficolins CAN BIND their carbohydrate targets but CANNOT activate
           complement (no C4/C2 cleavage); lectin pathway complement activation = ZERO;
         Note: MASP1 activates MASP2 (MASP1 autocleavage/trans-cleavage required for MASP2 activation);
           MASP3 activates the alternative pathway; MASP2 deficiency specifically blocks lectin → C3
           convertase but leaves alternative and classical pathways intact;
         CLINICAL PRESENTATION: recurrent pyogenic bacterial infections (pneumococcal pneumonia,
           meningitis, otitis), RHEUMATOID ARTHRITIS-LIKE AUTOIMMUNE PRESENTATIONS (synovitis,
           complement-mediated immune complex clearance failure), inflammatory disease;
           very rare — fewer than 30 families described globally;
         DIAGNOSIS: serum MASP2 by ELISA (undetectable); MBL-mediated complement activation assay
           on mannan-coated plates (absent C4 deposition); CH50 normal (classical pathway intact);
           AH50 normal (alternative pathway intact); biallelic MASP2 variants;
         MASP2 INHIBITOR (narsoplimab/OMS721): APPROVED for HSCT-TMA (thrombotic microangiopathy) —
           opposite use: in MASP2 deficiency (LOF) there is already no activity; narsoplimab NOT indicated;
         TREATMENT: prophylactic antibiotics; plasma infusion (contains MASP2) for acute severe infections;
           IVIG as secondary support; antirheumatic therapy for inflammatory manifestations; seed SEED_BASE+6).
COLEC11 (Collectin subfamily member 11 / Collectin-K1 / CL-K1; 344aa; 2p25.3; AR;
          COLEC11 encodes Collectin-11 (CL-K1), a collagen-domain lectin secreted from liver and kidney;
          COLEC11 forms hetero-oligomers with Collectin-L1 (COLEC10) → activates lectin complement
            pathway through MASP1/MASP2 (independent of MBL2 but using same effector complex);
          COLEC11 is expressed during embryonic development and plays a role in guiding neural crest
            cell migration via complement pathway → loss causes structural developmental defects;
          3MC SYNDROME (Malpuech-Michels-Mingarelli-Carnevale syndrome): AR disease caused by
            COLEC11 mutations (3MC type 2) or MASP1 mutations (3MC type 1);
          PATHOGNOMONIC DEVELOPMENTAL FEATURES: craniosynostosis (premature skull suture fusion),
            posterior embryotoxon/Axenfeld anomaly (anterior segment eye defects), cleft lip/palate,
            intellectual disability/hearing loss, caudal dysgenesis, ptosis, blepharophimosis,
            hypertelorism, preauricular tags — this constellation = 3MC syndrome;
          IMMUNE DEFICIENCY COMPONENT: lectin pathway activation absent → recurrent childhood
            infections (pneumococcal, RSV, respiratory viruses); infection severity milder than
            MBL/MASP2 deficiency alone (COLEC11 is one of multiple MBL-independent lectin PRM);
          DIAGNOSIS: 3MC facial/structural features on examination; COLEC11 serum levels (reduced);
            lectin pathway complement activation (reduced); COLEC11 vs MASP1 sequencing (both cause 3MC);
            ophthalmology assessment (posterior embryotoxon — posterior corneal ring);
          TREATMENT: address structural anomalies (craniofacial surgery, hearing aids, cleft repair);
            prophylactic antibiotics for infections; no specific immune replacement; seed SEED_BASE+7).

ATLAS_NAME   = "Hereditary-LAD-DADA2-Lectin-Complement-Granulocyte-Defect-Atlas"
SEED_BASE    = 4006   # seeds 4006-4013 (8 genes × 40 patients each)
N_PER_GENE   = 40
TOTAL_N      = N_PER_GENE * 8
"""

import random
import math

ATLAS_NAME   = "Hereditary-LAD-DADA2-Lectin-Complement-Granulocyte-Defect-Atlas"
SEED_BASE    = 4006   # seeds 4006-4013 (8 genes × 40 patients each)
N_PER_GENE   = 40
TOTAL_N      = N_PER_GENE * 8

GENES = ["ITGB2", "FERMT3", "ADA2", "CEBPE", "CXCR2", "MBL2", "MASP2", "COLEC11"]

GENE_DATA = {
    "ITGB2": {
        "full_name":               "Integrin beta-2 / CD18 (ITGB2; Leukocyte Adhesion Deficiency type I / LAD-I)",
        "aa":                      769,
        "locus":                   "21q22.3",
        "kda":                     "84",
        "inheritance":             "AR (autosomal recessive; biallelic ITGB2 mutations; both sexes equally affected)",
        "syndrome":                "LAD-I — Leukocyte Adhesion Deficiency type I; CD18 absent; most common LAD (~65% of all LAD cases)",
        "pathway":                 "ITGB2 absent → all four β2-integrin heterodimers (CD11a-CD18/LFA-1, CD11b-CD18/Mac-1, CD11c-CD18, CD11d-CD18) fail surface expression → neutrophils cannot adhere to ICAM-1/ICAM-2 → no leukocyte extravasation into tissues",
        "onset_range":             (0, 6),      # neonatal to infancy (severe LAD-I)
        "cd18_absent_pct":         0.99,        # CD18 absent/markedly reduced by flow
        "delayed_cord_sep_pct":    0.85,        # delayed umbilical cord separation PATHOGNOMONIC
        "omphalitis_pct":          0.80,        # omphalitis
        "periodontitis_pct":       0.75,        # severe juvenile periodontitis in survivors
        "leukocytosis_pct":        0.92,        # leukocytosis 50-100k without pus
        "hsct_pct":                0.75,        # HSCT curative; only definitive tx
        "recurrent_infect_pct":    0.98,        # recurrent severe infections
        "pathognomonic":           "DELAYED UMBILICAL CORD SEPARATION (>3-4 weeks) + omphalitis PATHOGNOMONIC; LEUKOCYTOSIS 50-100k WITHOUT PUS FORMATION (neutrophils in blood, cannot exit into tissues); severe PERIODONTITIS destroying alveolar bone by 3rd decade; CD18 ABSENT on flow cytometry = confirmatory (CD18/CD11b staining near-zero); LAD-I vs LAD-III: ITGB2 protein ABSENT in LAD-I (CD18 undetectable); in LAD-III (FERMT3) CD18 expression NORMAL but function absent",
        "key_drug":                "HSCT is ONLY CURATIVE TREATMENT for severe LAD-I (corrects CD18 expression in all hematopoietic cells); prophylactic TMP-SMX (bacterial) + itraconazole/voriconazole (Aspergillus); G-CSF limited benefit (makes more non-functional non-adherent PMNs); lentiviral ITGB2 gene therapy in clinical trials; IVIG for secondary antibody deficiency",
    },
    "FERMT3": {
        "full_name":               "Fermitin family homolog 3 / Kindlin-3 (FERMT3; Leukocyte Adhesion Deficiency type III / LAD-III)",
        "aa":                      667,
        "locus":                   "11q13.1",
        "kda":                     "76",
        "inheritance":             "AR (autosomal recessive; biallelic FERMT3 mutations; very rare; most cases in consanguineous families)",
        "syndrome":                "LAD-III — Leukocyte Adhesion Deficiency type III (integrin activation defect); all β1/β2/β3 integrins fail inside-out activation → combined leukocyte adhesion failure + Glanzmann thrombasthenia-like platelet defect",
        "pathway":                 "FERMT3/Kindlin-3 absent → β2 integrins cannot undergo inside-out activation → same leukocyte adhesion failure as LAD-I + β3 integrins on platelets fail activation → platelet aggregation defect; CD18 EXPRESSION NORMAL (ITGB2 intact) but β2 integrin FUNCTION absent",
        "onset_range":             (0, 12),
        "cd18_absent_pct":         0.05,        # CD18 expression NORMAL in LAD-III (key DDx from LAD-I)
        "delayed_cord_sep_pct":    0.82,        # same as LAD-I
        "omphalitis_pct":          0.78,
        "periodontitis_pct":       0.70,
        "leukocytosis_pct":        0.90,        # leukocytosis WITHOUT pus
        "platelet_dysfunction_pct": 0.92,       # Glanzmann-like bleeding PATHOGNOMONIC
        "mucosal_bleeding_pct":    0.88,        # epistaxis, gingival bleeding
        "hsct_pct":                0.88,        # HSCT mandatory and curative
        "recurrent_infect_pct":    0.96,
        "pathognomonic":           "COMBINED INFECTIONS + BLEEDING PATHOGNOMONIC for LAD-III: same leukocyte adhesion failure as LAD-I (delayed cord separation, omphalitis, periodontitis, absent pus) PLUS platelet dysfunction bleeding (like Glanzmann thrombasthenia: mucosal bleeding, prolonged bleeding time, absent ADP aggregation, normal platelet count); CRITICAL DISTINCTION from LAD-I: CD18 surface expression NORMAL by flow cytometry (ITGB2 protein present, Kindlin-3-mediated activation absent); functional assay: β2 integrin activation (CD11b upregulation after PMA/fMLP stimulation) ABSENT despite normal CD18",
        "key_drug":                "HSCT MANDATORY AND CURATIVE (only treatment: reconstitutes Kindlin-3 in all hematopoietic cells, correcting both PMN adhesion and platelet activation); platelet transfusions for acute bleeding; prophylactic antibiotics; recombinant factor VIIa for life-threatening hemorrhage; G-CSF not effective",
    },
    "ADA2": {
        "full_name":               "Adenosine deaminase 2 / CECR1 (ADA2; DADA2 = Deficiency of ADA2)",
        "aa":                      511,
        "locus":                   "22q11.1",
        "kda":                     "59",
        "inheritance":             "AR (autosomal recessive; biallelic ADA2/CECR1 mutations; both sexes; consanguineous families; Turkish, Georgian, Middle Eastern founder variants)",
        "syndrome":                "DADA2 — Deficiency of ADA2; vasculopathy (PAN-like) + early stroke + livedo reticularis + NK cell deficiency; described 2014 (Navon Elkan et al. / Zhou et al.)",
        "pathway":                 "ADA2 absent → extracellular adenosine accumulates → A2A receptor hyperactivation on monocytes → M1 macrophage skewing + vascular endothelial injury → destructive vasculitis; loss of ADA2 trophic growth factor for vascular pericytes → endothelial fragility → vasculitis",
        "onset_range":             (6, 120),    # infancy to late childhood (stroke often < 5yr)
        "vasculopathy_pct":        0.85,        # PAN-like vasculopathy
        "stroke_pct":              0.65,        # early childhood stroke PATHOGNOMONIC
        "livedo_pct":              0.78,        # livedo reticularis PATHOGNOMONIC
        "nk_deficiency_pct":       0.82,        # NK cell lymphopenia
        "hepatosplenomegaly_pct":  0.70,        # organomegaly
        "hypogamma_pct":           0.55,        # hypogammaglobulinemia
        "anti_tnf_pct":            0.88,        # anti-TNF curative for vasculopathy
        "hsct_pct":                0.22,        # HSCT for severe cytopenias only (not first-line)
        "recurrent_infect_pct":    0.60,        # infections less prominent than classic PID
        "pathognomonic":           "PAN-LIKE VASCULOPATHY + EARLY STROKE (< 5-10 years) + LIVEDO RETICULARIS = DADA2 CLINICAL TRIAD PATHOGNOMONIC; NK CELL LYMPHOPENIA distinguishes from classic PAN (classic PAN has normal NK cells); start TNF-INHIBITORS IMMEDIATELY to prevent further strokes; CRITICAL DDx from Kawasaki (coronary artery aneurysms dominant, self-limited, no NK deficiency) and classic PAN (older onset, ANCA-variable, no NK lymphopenia); serum ADA2 enzyme activity (absent or markedly reduced) = confirmatory functional test; ADA2/CECR1 sequencing",
        "key_drug":                "TNF-INHIBITORS (etanercept 0.4 mg/kg SC twice weekly OR adalimumab 40 mg SC every 2 weeks) ARE CURATIVE FOR VASCULOPATHY — START IMMEDIATELY on diagnosis to prevent irreversible CNS injury; IVIG for hypogammaglobulinemia; HSCT for severe bone marrow failure (cytopenias) not responsive to anti-TNF; steroids for acute flares; aspirin/anticoagulation for stroke prevention; ADA2 enzyme replacement (pegylated ADA2) in experimental trials; do NOT delay anti-TNF",
    },
    "CEBPE": {
        "full_name":               "CCAAT/enhancer-binding protein epsilon / C/EBPε (CEBPE; Specific Granule Deficiency / SGD)",
        "aa":                      317,
        "locus":                   "14q11.2",
        "kda":                     "32",
        "inheritance":             "AR (autosomal recessive; biallelic CEBPE mutations; very rare; <30 families globally)",
        "syndrome":                "SGD — Specific Granule Deficiency; complete absence of neutrophil secondary and tertiary granules; primary (azurophil/MPO) granules intact",
        "pathway":                 "CEBPE absent → no transcription of secondary granule genes (lactoferrin/LTF, collagenase/MMP-8, vitamin B12-binding protein/transcobalamin-I) and tertiary granule genes (gelatinase/MMP-9) → neutrophils lack all antimicrobial functions from secondary/tertiary granules despite normal NADPH oxidase activity",
        "onset_range":             (2, 36),
        "bilobed_nuclei_pct":      0.98,        # PATHOGNOMONIC on blood film
        "absent_lactoferrin_pct":  0.99,        # confirmatory granule assay
        "nbt_normal_pct":          0.97,        # NBT/DHR NORMAL (not CGD) - DDx
        "staph_infect_pct":        0.88,        # Staphylococcus aureus abscesses
        "aspergillus_pct":         0.55,        # Aspergillus pulmonary infections
        "gram_neg_pct":            0.70,        # Pseudomonas/Klebsiella septicemia
        "hsct_pct":                0.45,        # HSCT for severe cases
        "recurrent_infect_pct":    0.95,
        "pathognomonic":           "BILOBED PMN NUCLEI on blood film PATHOGNOMONIC (failure of nuclear segmentation — C/EBPε required for terminal PMN differentiation program; bilobed nucleus = Pelger-Huet-like morphology in SGD); ABSENT LACTOFERRIN by granule protein assay (lactoferrin ELISA on PMN lysate = undetectable); NBT/DHR TESTS NORMAL (NADPH oxidase intact — key DDx from CGD where NBT/DHR absent); MPO staining PRESENT (primary granules intact); pathogens identical to CGD (Staphylococcus, Aspergillus, gram-negatives) but oxidase assay normal",
        "key_drug":                "PROPHYLACTIC TMP-SMX (PCP + bacterial prophylaxis) + ANTIFUNGAL PROPHYLAXIS (itraconazole/voriconazole for Aspergillus — same as CGD prophylaxis); G-CSF variable benefit (increases circulating neutrophil numbers but cannot restore absent secondary granule contents); HSCT curative for severe cases; IVIG if secondary antibody deficiency develops; CEBPE sequencing + granule protein assay (lactoferrin absent) = confirmatory",
    },
    "CXCR2": {
        "full_name":               "C-X-C chemokine receptor type 2 / IL-8 receptor B / CD182 (CXCR2; Severe Congenital Neutropenia type 6 / SCN6)",
        "aa":                      360,
        "locus":                   "2q35",
        "kda":                     "40",
        "inheritance":             "AR (autosomal recessive; biallelic CXCR2 mutations; very rare; Middle Eastern and European families; some consanguineous)",
        "syndrome":                "SCN6 — Severe Congenital Neutropenia type 6 / CXCR2 deficiency; neutrophil mobilization failure + chemotaxis defect + inflammatory skin disease + recurrent UTI",
        "pathway":                 "CXCR2 absent → PMNs cannot respond to CXCL8/IL-8, CXCL1-3, CXCL5-6 gradients → failure to exit bone marrow (CXCR2 opposes CXCR4 retention) → peripheral neutropenia + retained marrow pool; also no chemotaxis to infection sites",
        "onset_range":             (1, 24),
        "neutropenia_pct":         0.95,        # peripheral neutropenia
        "marrow_reserve_normal_pct": 0.88,      # normal marrow pool (exit failure not production)
        "skin_disease_pct":        0.75,        # neutrophilic dermatosis
        "uti_recurrent_pct":       0.70,        # recurrent UTI from absent uroepithelial PMN
        "gcsf_response_pct":       0.82,        # G-CSF effective (bypasses CXCR2)
        "hsct_pct":                0.20,        # HSCT for refractory cases
        "recurrent_infect_pct":    0.88,
        "pathognomonic":           "NEUTROPENIA + NORMAL/ELEVATED BONE MARROW MYELOID RESERVE (exit failure, not production failure — DDx from ELANE/HAX1 SCN where marrow is arrested); CXCR2 ABSENT on PMN surface by flow cytometry (CD182/CXCR2 staining); NEUTROPHILIC INFLAMMATORY SKIN DISEASE (pyoderma gangrenosum-like or Sweet syndrome lesions) — neutrophils present in skin but dysregulated without CXCR2 signaling; RECURRENT UTIs (uroepithelial PMN trafficking absent); G-CSF EFFECTIVE (mobilizes via CXCR2-independent G-CSFR pathway; normalizes ANC) — this differentiates SCN6 management from LAD-I (where G-CSF not useful)",
        "key_drug":                "G-CSF (filgrastim 5-10 mcg/kg/day) EFFECTIVE AND FIRST-LINE — ANC normalizes via G-CSFR, bypassing absent CXCR2; ruxolitinib (JAK1/2 inhibitor) for inflammatory neutrophilic skin disease; prophylactic TMP-SMX; HSCT for refractory cases unresponsive to G-CSF; CXCR2 sequencing + flow cytometry (CD182 absent) = confirmatory",
    },
    "MBL2": {
        "full_name":               "Mannose-binding lectin 2 (MBL2; MBL deficiency / Partial lectin pathway immunodeficiency)",
        "aa":                      244,
        "locus":                   "10q11.22",
        "kda":                     "26",
        "inheritance":             "AR (autosomal recessive; functional deficiency from structural variants at codons 52/54/57; promoter variants further reduce expression; common partial deficiency via compound heterozygosity)",
        "syndrome":                "MBL deficiency — reduced/absent plasma MBL; partial immunodeficiency with childhood infections; complement lectin pathway activation absent/reduced",
        "pathway":                 "MBL2 absent/reduced → plasma MBL monomers fail oligomerization → cannot bind MASP1/MASP2 → lectin complement pathway inactive; loss of opsonization function for pathogens with terminal mannose/GlcNAc arrays (bacteria, fungi, viruses)",
        "onset_range":             (3, 60),     # infancy to early childhood
        "mbl_absent_pct":          0.95,        # MBL serum < 100 ng/mL
        "lectin_pathway_absent_pct": 0.90,      # absent lectin pathway on mannan-coated plates
        "ch50_normal_pct":         0.98,        # CH50 normal (classical pathway intact)
        "ah50_normal_pct":         0.97,        # AH50 normal (alternative pathway intact)
        "childhood_infect_pct":    0.75,        # recurrent otitis/pneumonia in childhood
        "synergy_immunosupp_pct":  0.60,        # significant with concurrent immunosuppression
        "hsct_pct":                0.02,        # HSCT not used
        "recurrent_infect_pct":    0.70,
        "pathognomonic":           "PARTIAL IMMUNODEFICIENCY: MBL deficiency alone rarely causes severe disease in healthy adults but clinically significant in infancy (before maternal IgG wanes), during immunosuppression, or with concurrent humoral immunodeficiency; CH50 NORMAL (classical pathway intact) + AH50 NORMAL (alternative intact) — only MBL-SPECIFIC lectin pathway assay (mannan-coated plate C4 deposition) reduced/absent; MBL SERUM LEVEL < 100 ng/mL diagnostic; structural variants at codons 52, 54, 57 — homozygous or compound heterozygous structural variants cause profound deficiency; promoter haplotypes (H, L, X, Y) modify expression level",
        "key_drug":                "PROPHYLACTIC ANTIBIOTICS during high-risk periods (infancy, immunosuppression, concurrent humoral deficiency); RECOMBINANT MBL replacement (rMBL2) — clinical trials show reduced infection frequency; IVIG (natural source of MBL and complement opsonins); FRESH FROZEN PLASMA as MBL source for acute severe infections; avoid immunosuppression without antibiotic cover; no HSCT indicated",
    },
    "MASP2": {
        "full_name":               "MBL-associated serine protease 2 (MASP2; Lectin pathway complement effector deficiency)",
        "aa":                      686,
        "locus":                   "1p36.22",
        "kda":                     "76",
        "inheritance":             "AR (autosomal recessive; biallelic MASP2 mutations; very rare <30 families globally; some compound heterozygous)",
        "syndrome":                "MASP2 deficiency — complete lectin pathway complement effector absence; recurrent pyogenic infections + rheumatoid arthritis-like inflammation; CH50 and AH50 normal",
        "pathway":                 "MASP2 absent → MBL/ficolin binding to pathogens activates MASP1 but MASP2 cannot be cleaved → C4 and C2 not cleaved → no lectin-pathway C3 convertase (C4b2a) → lectin complement pathway inactive; MBL binding occurs but complement activation fails",
        "onset_range":             (6, 120),
        "masp2_absent_pct":        0.99,        # serum MASP2 absent
        "lectin_pathway_absent_pct": 0.99,      # mannan plate C4 deposition absent
        "ch50_normal_pct":         0.98,        # CH50 NORMAL
        "ah50_normal_pct":         0.97,        # AH50 NORMAL
        "pyogenic_pct":            0.78,        # pyogenic bacterial infections
        "ra_like_pct":             0.45,        # rheumatoid arthritis-like presentation
        "inflammatory_pct":        0.55,        # autoimmune/inflammatory
        "hsct_pct":                0.02,        # HSCT not used
        "recurrent_infect_pct":    0.72,
        "pathognomonic":           "MASP2 SERUM UNDETECTABLE by ELISA = confirmatory; LECTIN PATHWAY COMPLEMENT ABSENT (mannan-coated plate C4 deposition = zero) despite CH50 NORMAL (classical) and AH50 NORMAL (alternative); RHEUMATOID ARTHRITIS-LIKE PRESENTATIONS (synovitis, immune complex deposition without complement clearance) alongside pyogenic infections — this inflammatory + infectious dual presentation distinguishes MASP2 from MBL2 deficiency (MBL2 = mostly infections; MASP2 = infections + autoimmune); MASP INHIBITOR NARSOPLIMAB (OMS721) approved for HSCT-TMA — CONTRAINDICATED in MASP2 deficiency (already absent function)",
        "key_drug":                "PROPHYLACTIC ANTIBIOTICS (TMP-SMX + penicillin for encapsulated organisms); FRESH FROZEN PLASMA (contains MASP2) for acute severe infections; IVIG for secondary antibody support; antirheumatic therapy (hydroxychloroquine, DMARDs) for inflammatory presentations; narsoplimab/OMS721 CONTRAINDICATED (inhibits MASP2 — already absent); no HSCT indicated",
    },
    "COLEC11": {
        "full_name":               "Collectin subfamily member 11 / Collectin-K1 / CL-K1 (COLEC11; 3MC syndrome type 2 / Lectin complement + developmental defect)",
        "aa":                      344,
        "locus":                   "2p25.3",
        "kda":                     "38",
        "inheritance":             "AR (autosomal recessive; biallelic COLEC11 mutations; very rare; consanguineous families; multiple ethnic backgrounds)",
        "syndrome":                "3MC syndrome type 2 (Malpuech-Michels-Mingarelli-Carnevale syndrome) — COLEC11 mutations; craniofacial developmental defects + lectin complement pathway deficiency + recurrent childhood infections",
        "pathway":                 "COLEC11/CL-K1 absent → reduced lectin complement pathway activation through COLEC11-MASP1/MASP2 complexes; also loss of COLEC11 developmental guidance role (neural crest migration via complement) → craniofacial and caudal developmental anomalies (3MC structural phenotype)",
        "onset_range":             (0, 24),     # congenital/infantile structural + infections
        "craniosynostosis_pct":    0.70,        # craniofacial fusion
        "cleft_lip_palate_pct":    0.65,        # cleft lip/palate PATHOGNOMONIC
        "hearing_loss_pct":        0.60,        # sensorineural hearing loss
        "posterior_embryotoxon_pct": 0.55,      # posterior corneal ring (Axenfeld anomaly)
        "caudal_dysgenesis_pct":   0.35,        # sacral agenesis / caudal regression
        "lectin_pathway_reduced_pct": 0.85,     # reduced (not absent) lectin complement
        "childhood_infect_pct":    0.68,        # recurrent respiratory infections
        "hsct_pct":                0.02,        # HSCT not used
        "recurrent_infect_pct":    0.65,
        "pathognomonic":           "3MC SYNDROME CLINICAL FEATURES PATHOGNOMONIC: craniosynostosis (premature skull suture fusion), POSTERIOR EMBRYOTOXON / AXENFELD ANOMALY (whitish line at posterior corneal periphery = Schwalbe ring displacement — ophthalmology finding), cleft lip/palate, intellectual disability, sensorineural hearing loss, caudal dysgenesis (sacral agenesis, urogenital anomalies) — constellation defines 3MC syndrome; COLEC11 vs MASP1 sequencing distinguishes 3MC type 2 (COLEC11) from 3MC type 1 (MASP1); lectin complement pathway REDUCED (not absent) because MBL2/ficolin pathways partially compensate",
        "key_drug":                "Address STRUCTURAL ANOMALIES: craniofacial surgery for craniosynostosis; cleft repair; hearing aids/cochlear implant; ophthalmology follow-up for glaucoma risk (posterior embryotoxon); urogenital surgery for caudal defects; PROPHYLACTIC ANTIBIOTICS for infections; no specific immune replacement; COLEC11 and MASP1 sequencing for 3MC confirmation",
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
        sex = s.choice(["M", "F"])   # all AR → equal sex distribution
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
        "hsct_pct":                _p("hsct_pct"),
        "recurrent_infect_pct":    _p("recurrent_infect_pct"),
        "pathognomonic":           gd["pathognomonic"],
        "key_drug":                gd["key_drug"],
        "seed":                    seed,
    }


def generate_overview():
    cohorts = [_aggregate_gene(g, i) for i, g in enumerate(GENES)]

    total_hsct      = round(sum(c["hsct_pct"]      for c in cohorts) / len(cohorts), 3)
    total_infect    = round(sum(c["recurrent_infect_pct"] for c in cohorts) / len(cohorts), 3)

    return {
        "atlas":    ATLAS_NAME,
        "genes":    GENES,
        "total_n":  TOTAL_N,
        "seeds":    f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_stats": {
            "hsct_pct":                    total_hsct,
            "recurrent_infections_pct":    total_infect,
        },
        "key_facts": [
            "ITGB2/CD18 (AR 21q22.3): LAD-I — CD18 ABSENT on flow cytometry; DELAYED UMBILICAL CORD SEPARATION (>3-4 weeks) PATHOGNOMONIC; LEUKOCYTOSIS 50-100k WITHOUT PUS FORMATION; severe periodontitis in survivors; G-CSF NOT effective (PMNs non-adherent); HSCT = ONLY CURATIVE TREATMENT; gene therapy trials ongoing",
            "FERMT3/Kindlin-3 (AR 11q13.1): LAD-III — COMBINED INFECTIONS + BLEEDING PATHOGNOMONIC; CD18 EXPRESSION NORMAL (DDx LAD-I) but β2-integrin function absent; all β1/β2/β3 integrins fail inside-out activation → also Glanzmann-like platelet dysfunction; HSCT MANDATORY AND CURATIVE; functional integrin activation assay distinguishes from LAD-I",
            "ADA2/CECR1 (AR 22q11.1): DADA2 — POLYARTERITIS NODOSA-LIKE VASCULOPATHY + EARLY STROKE (<10yr) + LIVEDO RETICULARIS = PATHOGNOMONIC TRIAD; NK CELL LYMPHOPENIA distinguishes from classic PAN; TNF-INHIBITORS (etanercept/adalimumab) CURATIVE FOR VASCULOPATHY — START IMMEDIATELY; IVIG for hypogammaglobulinemia; HSCT only for refractory cytopenias",
            "CEBPE/C-EBPε (AR 14q11.2): Specific Granule Deficiency — BILOBED PMN NUCLEI on blood film PATHOGNOMONIC; ABSENT LACTOFERRIN on granule assay; NBT/DHR NORMAL (not CGD — primary granules intact); same pathogens as CGD (Staph, Aspergillus, gram-negatives) but oxidase normal; prophylactic TMP-SMX + antifungal; HSCT for severe cases",
            "CXCR2/CD182 (AR 2q35): SCN6 — NEUTROPENIA + NORMAL MARROW RESERVE (exit failure, not production failure); CXCR2 ABSENT on PMN flow; inflammatory skin disease; recurrent UTIs; G-CSF EFFECTIVE (bypasses absent CXCR2 via G-CSFR); ruxolitinib for skin disease; HSCT for refractory",
            "MBL2 (AR 10q11.22): MBL DEFICIENCY — CH50/AH50 NORMAL; only MBL-pathway (mannan plate) reduced; PARTIAL IMMUNODEFICIENCY significant in infancy/immunosuppression; recombinant MBL replacement trials; prophylactic antibiotics high-risk periods; IVIG as opsonin source",
            "MASP2 (AR 1p36.22): MASP2 DEFICIENCY — serum MASP2 undetectable; lectin pathway C4 deposition absent; CH50/AH50 NORMAL; pyogenic infections + RA-LIKE inflammation; narsoplimab CONTRAINDICATED (MASP2 already absent); FFP contains MASP2 for acute infections",
            "COLEC11/CL-K1 (AR 2p25.3): 3MC SYNDROME TYPE 2 — CRANIOSYNOSTOSIS + POSTERIOR EMBRYOTOXON (Axenfeld corneal ring) + CLEFT LIP/PALATE + HEARING LOSS = PATHOGNOMONIC developmental triad alongside lectin complement reduction; COLEC11 vs MASP1 sequencing for 3MC subtyping; address structural anomalies + prophylactic antibiotics",
        ],
        "lad_group":             ["ITGB2", "FERMT3"],
        "vasculopathy_group":    ["ADA2"],
        "granule_chemotaxis_group": ["CEBPE", "CXCR2"],
        "lectin_pathway_group":  ["MBL2", "MASP2", "COLEC11"],
        "hsct_mandatory_group":  ["ITGB2", "FERMT3"],
        "anti_tnf_group":        ["ADA2"],
        "gcsf_effective_group":  ["CXCR2"],
        "cohort_summaries":      cohorts,
    }


def generate_breakdown():
    cohorts = [_aggregate_gene(g, i) for i, g in enumerate(GENES)]

    diagnostic_groups = {
        "lad_integrin_adhesion_deficiency": {
            "genes":          ["ITGB2", "FERMT3"],
            "mechanism":      "Leukocyte integrin adhesion failure: ITGB2/LAD-I — CD18 subunit absent → all four β2-integrin heterodimers fail surface expression → zero PMN adhesion to ICAM-1/2; FERMT3/LAD-III — Kindlin-3 absent → β2 integrin inside-out activation fails (CD18 expressed but inactive) + β3 platelet integrin fails (Glanzmann-like bleeding)",
            "shared_feature": "Both AR; DELAYED UMBILICAL CORD SEPARATION; omphalitis; LEUKOCYTOSIS WITHOUT PUS FORMATION; severe periodontitis; recurrent skin/mucosal bacterial infections; HSCT CURATIVE AND MANDATORY for severe forms",
            "key_ddx":        "LAD-I (ITGB2): CD18 ABSENT by flow cytometry (CD18 near-zero); no platelet dysfunction; mutations in ITGB2 gene. LAD-III (FERMT3): CD18 EXPRESSION NORMAL by flow (DDx from LAD-I); β2-integrin ACTIVATION absent (CD11b upregulation assay after PMA/fMLP = negative); PLATELET DYSFUNCTION BLEEDING (prolonged bleeding time, absent ADP aggregation, normal platelet count — combined LAD-III = LAD + Glanzmann thrombasthenia phenotype). KEY DIAGNOSTIC STEP: always check CD18 flow first; if CD18 normal but LAD clinical phenotype → order FERMT3 sequencing and integrin activation assay.",
        },
        "dada2_vasculopathy": {
            "genes":          ["ADA2"],
            "mechanism":      "ADA2 (adenosine deaminase 2) absent → extracellular adenosine accumulates → monocyte hyperactivation via A2A receptor → M1 macrophage skewing → destructive vasculitis; ADA2 growth factor role for vascular pericytes lost → endothelial fragility; NK cell differentiation impaired (ADA2 supports NK/monocyte lineage)",
            "shared_feature": "AR; PAN-like vasculopathy; early-onset stroke; livedo reticularis; NK cell lymphopenia; hepatosplenomegaly; respond dramatically to TNF-inhibitors",
            "key_ddx":        "DADA2 vs Classic PAN: DADA2 = childhood onset (<10yr), NK lymphopenia, biallelic ADA2 mutations, anti-TNF CURATIVE; Classic PAN = adult onset, ANCA-variable, NK cells normal, treated with steroids + cyclophosphamide (not curative). DADA2 vs Kawasaki: Kawasaki = coronary artery aneurysms dominant, fever >5 days, self-limited, IVIG + aspirin responsive; DADA2 = recurrent strokes, NK deficiency, not fever-dominant, not self-limited, anti-TNF required. ADA2 enzyme activity (serum ADA2 < 0.1 nmol/hr/mL in DADA2) = confirmatory functional assay BEFORE sequencing in urgent cases.",
        },
        "granule_and_chemotaxis_defects": {
            "genes":          ["CEBPE", "CXCR2"],
            "mechanism":      "CEBPE/SGD: transcription factor for secondary/tertiary granule biogenesis absent → PMNs lack lactoferrin, gelatinase, collagenase granule proteins despite intact NADPH oxidase; CXCR2/SCN6: IL-8 receptor absent → PMNs trapped in bone marrow (exit failure) + cannot migrate toward infection site",
            "shared_feature": "Both AR; recurrent bacterial (Staphylococcus, gram-negative) and fungal (Aspergillus) infections; neutrophil functional defect distinguished from CGD by NBT/DHR NORMAL (both CEBPE and CXCR2)",
            "key_ddx":        "CEBPE (SGD): bilobed PMN nuclei on blood smear PATHOGNOMONIC; absent lactoferrin by granule assay; NBT/DHR NORMAL; peripheral neutrophil count normal or mild neutropenia; CEBPE sequencing. CXCR2 (SCN6): peripheral neutropenia (ANC 0-500); normal marrow myeloid reserve (exit failure); absent CXCR2 by flow (CD182 staining zero); inflammatory skin disease; G-CSF effective (CXCR2 vs ELANE/HAX1 SCN: all respond to G-CSF but CXCR2 has normal marrow); CXCR2 sequencing.",
        },
        "lectin_pathway_complement_deficiency": {
            "genes":          ["MBL2", "MASP2", "COLEC11"],
            "mechanism":      "Lectin complement pathway (MBL/ficolin → MASP1/MASP2 → C4/C2 → C3 convertase) interrupted at three levels: MBL2 = pattern recognition failure (no MBL to bind carbohydrates); MASP2 = effector enzyme absent (binding occurs but C4/C2 not cleaved); COLEC11 = alternative lectin PRM absent (CL-K1/CL-L1 pathway inactive + developmental guidance role lost → 3MC syndrome)",
            "shared_feature": "CH50 NORMAL (classical pathway intact), AH50 NORMAL (alternative pathway intact) in all three — lectin-specific assay (mannan plate C4/C3 deposition) reduced/absent; AR inheritance",
            "key_ddx":        "MBL2: structural variant codons 52/54/57 (not rare Mendelian mutation); partial immunodeficiency; promoter + structural variants needed for profound deficiency; often modifier of other immune defects. MASP2: rarer; pyogenic infections + RA-like inflammation; MASP2 serum absent by ELISA; narsoplimab CONTRAINDICATED in MASP2 LOF. COLEC11: 3MC SYNDROME developmental features dominant (craniosynostosis, posterior embryotoxon, cleft, hearing loss, caudal dysgenesis) alongside lectin pathway reduction; MASP1 mutations cause same 3MC type 1 (sequence both).",
        },
    }

    pathognomonic_features = {
        "delayed_cord_separation_lad": "DELAYED UMBILICAL CORD SEPARATION (> 3-4 weeks after birth) with omphalitis is the PATHOGNOMONIC neonatal sign of LAD-I (ITGB2) and LAD-III (FERMT3): periumbilical neutrophil infiltration fails because PMNs cannot adhere to inflamed umbilical stump vessels → cord persists and becomes infected without adequate pus; normal cord separation = 7-15 days; >3 weeks = investigate LAD; CD18 flow cytometry FIRST (LAD-I = absent; LAD-III = normal but functionally inactive)",
        "lad_leukocytosis_without_pus": "LAD-I/III PARADOX: EXTREME LEUKOCYTOSIS (WBC 50-100k cells/μL, mostly neutrophils) WITHOUT LOCAL PUS FORMATION; PMNs are present in blood but cannot exit into infected tissues (adhesion-failed PMNs recirculate); abscesses empty of PMNs (sterile-looking abscesses); periodontitis without typical PMN-rich gingival pus; the combination of leukocytosis + abscess without pus = LAD until proven otherwise",
        "lad3_combined_infections_bleeding": "LAD-III COMBINED PHENOTYPE PATHOGNOMONIC: Kindlin-3 is expressed in ALL hematopoietic cells — β2 integrins on PMNs (CD18 activation defect → LAD phenotype) AND β3 integrins on platelets (αIIbβ3 activation defect → Glanzmann-like platelet dysfunction); combined severe infections (from PMN) + mucosal bleeding (from platelet) = FERMT3/LAD-III; CD18 normal by flow (DDx from LAD-I is critical — LAD-III may be misdiagnosed if only CD18 expression checked without activation assay)",
        "dada2_vasculopathy_triad": "DADA2 TRIAD PATHOGNOMONIC: (1) POLYARTERITIS NODOSA-like vasculopathy (medium-vessel wall inflammation → aneurysms + stenoses); (2) EARLY-ONSET STROKE (hemorrhagic or ischemic, onset often before age 5 years) from cerebral vasculitis; (3) LIVEDO RETICULARIS (reticular/net-like skin discoloration from cutaneous vessel involvement); ANY CHILD with unexplained stroke + livedo reticularis → ADA2 testing IMMEDIATELY; NK cell lymphopenia confirms (classic PAN has normal NK); serum ADA2 enzyme activity is rapid functional screening",
        "sgd_bilobed_nuclei": "SPECIFIC GRANULE DEFICIENCY (CEBPE LOF) BLOOD FILM PATHOGNOMONIC: PMN BILOBED NUCLEI (Pelger-Huet-like appearance) — failure of terminal granulocyte nuclear segmentation program (C/EBPε drives nuclear hypersegmentation); ABSENT LACTOFERRIN in PMN granule assay (lactoferrin undetectable by ELISA in PMN lysate or supernatant); NBT test and DHR flow cytometry BOTH NORMAL (NADPH oxidase from primary granules = intact); MPO histochemistry POSITIVE (azurophil primary granules with MPO present); the combination of bilobed nuclei + absent specific granule proteins + normal NBT/DHR = SGD signature",
        "cxcr2_exit_failure": "CXCR2/SCN6 BONE MARROW EXIT FAILURE: peripheral neutropenia (ANC 0-500) WITH normal or increased bone marrow myeloid reserve (on bone marrow biopsy); this distinguishes SCN6 from production-failure SCN (ELANE, HAX1 where maturation arrest occurs); CXCR2 (CD182) absent on peripheral PMN surface by flow cytometry; CXCR2 is required for CXCL8/IL-8-mediated mobilization from marrow → without CXCR2, marrow retention via CXCR4 is unopposed; G-CSF overcomes retention by alternative pathway (G-CSF → G-CSFR → PI3K → CXCR4 downregulation) — G-CSF responsiveness is a distinguishing feature",
        "lectin_ch50_ah50_normal": "LECTIN PATHWAY DEFICIENCY (MBL2, MASP2, COLEC11) DIAGNOSTIC CLUE: CH50 (classical pathway = C1q-C4-C2-C3 cascade) NORMAL + AH50 (alternative pathway = factor D-B-properdin loop) NORMAL, but LECTIN-SPECIFIC assay (COMPLEMENT ACTIVATION ON MANNAN-COATED PLATE — measures C4/C3 deposition driven by MBL→MASP2) ABSENT OR MARKEDLY REDUCED; this pattern (normal CH50 + normal AH50 + absent lectin pathway) is the fingerprint of MBL2/MASP2/COLEC11 deficiency; many laboratories don't routinely test lectin pathway — must REQUEST specifically",
        "3mc_posterior_embryotoxon": "COLEC11/3MC SYNDROME OPHTHALMOLOGY FINDING: POSTERIOR EMBRYOTOXON (= Schwalbe's line prominence — the posterior corneal peripheral ring visible on slit-lamp as a white arcuate line near the limbus) is Axenfeld anomaly finding characteristic of 3MC syndrome; this combined with craniosynostosis, cleft lip/palate, sensorineural hearing loss, caudal dysgenesis = 3MC syndrome; 3MC = Malpuech-Michels-Mingarelli-Carnevale (4 physicians who described overlapping syndromes now unified); COLEC11 (3MC type 2) vs MASP1 (3MC type 1) — sequence both",
        "all_ar_no_sex_bias": "ALL 8 GENES IN THIS ATLAS ARE AUTOSOMAL RECESSIVE: equal sex distribution for all 8 diseases (no X-linked disease); consanguinity is a major risk factor (especially FERMT3, ADA2, CEBPE, CXCR2, COLEC11); male-only or female-only presentation should prompt reconsideration; LAD-I and LAD-III affect both sexes equally despite clinical similarity to X-linked CGD (which is predominantly male)",
    }

    return {
        "gene_cohorts":            cohorts,
        "diagnostic_groups":       diagnostic_groups,
        "pathognomonic_features":  pathognomonic_features,
        "cascade_testing_order": [
            "Step 1: Newborn with DELAYED UMBILICAL CORD SEPARATION (> 3-4 weeks) and/or omphalitis without adequate healing → IMMEDIATE CD18 flow cytometry (CD18/CD11b staining); CD18 absent/markedly reduced = LAD-I (ITGB2 sequencing); CD18 normal → proceed to Step 2",
            "Step 2: LAD clinical phenotype (delayed cord, omphalitis, leukocytosis without pus, periodontitis) but CD18 EXPRESSION NORMAL → ORDER FERMT3 SEQUENCING + β2-integrin activation assay (CD11b upregulation after PMA/fMLP) + platelet aggregation testing (absent = LAD-III); do NOT stop at CD18 flow if LAD is suspected",
            "Step 3: CHILD WITH EARLY STROKE (<10 years) + livedo reticularis OR PAN-like vasculopathy → ADA2 ENZYME ACTIVITY test (serum ADA2 < 0.1 nmol/hr/mL) IMMEDIATELY; NK cell count by flow (NK lymphopenia supports DADA2); ADA2/CECR1 sequencing; if DADA2 confirmed → START TNF-INHIBITOR IMMEDIATELY (do not wait for secondary workup)",
            "Step 4: Child with recurrent Staph/Aspergillus/gram-negative infections + NORMAL NBT/DHR test (NADPH oxidase intact — not CGD) → ORDER BLOOD FILM for bilobed PMN nuclei; if bilobed nuclei present → lactoferrin assay on PMN granule extract (absent = SGD) → CEBPE sequencing",
            "Step 5: Child with severe neutropenia (ANC < 500) + G-CSF responsive + absent CXCR2 on PMN flow (CD182 staining) → SCN6 (CXCR2 sequencing); bone marrow biopsy shows normal myeloid reserve (exit failure not maturation arrest — DDx from ELANE/HAX1 SCN); check for inflammatory skin disease (neutrophilic dermatosis) and recurrent UTIs",
            "Step 6: RECURRENT CHILDHOOD INFECTIONS + NORMAL CH50 + NORMAL AH50 → ORDER LECTIN PATHWAY COMPLEMENT ASSAY (mannan-coated plate C4/C3 deposition); if lectin pathway absent/reduced → MBL serum level (MBL2 deficiency = level < 100 ng/mL) + MASP2 serum ELISA (MASP2 deficiency = undetectable)",
            "Step 7: CHILD WITH DEVELOPMENTAL ANOMALIES (craniosynostosis, cleft lip/palate, hearing loss) + recurrent infections → OPHTHALMOLOGY ASSESSMENT (posterior embryotoxon/Axenfeld anomaly); if 3MC features → COLEC11 + MASP1 sequencing (3MC type 2 vs type 1); lectin pathway assay (reduced in both)",
            "Step 8: LAD-I (ITGB2) confirmed severe (< 1% CD18) → HSCT REFERRAL IMMEDIATELY; do not delay; use MSD > MUD; cord blood acceptable; conditioning with myeloablative regimen preferred for full chimerism and CD18 reconstitution; gene therapy if no suitable donor",
            "Step 9: LAD-III (FERMT3) confirmed → HSCT MANDATORY AND URGENT; platelet transfusions for acute bleeding; platelet function analyzer (PFA-100) confirms platelet dysfunction alongside PMN assays; counseling about combined PMN/platelet reconstitution after HSCT",
            "Step 10: DADA2 confirmed → TNF-INHIBITOR START; etanercept or adalimumab SC; monitor MRI brain every 6 months for new infarcts; IVIG for hypogammaglobulinemia; HSCT considered only for severe cytopenias unresponsive to anti-TNF; neurologist for stroke rehabilitation; ADA2 enzyme replacement in experimental trials",
        ],
    }


def generate_definitions():
    return {
        "definitions": [
            {"term": "Leukocyte Adhesion Deficiency (LAD)", "definition": "LAD is a group of rare AR primary immunodeficiencies defined by failure of neutrophil (and other leukocyte) adhesion to activated vascular endothelium, resulting in inability to extravasate into infected tissues. CLASSIC TRIAD: (1) delayed umbilical cord separation (>3-4 weeks) with omphalitis; (2) recurrent, life-threatening bacterial/fungal infections without adequate pus formation; (3) persistent leukocytosis (50-100k WBC) despite infection. Three types: LAD-I (ITGB2: CD18 absent), LAD-II (SLC35C1: fucosylation defect → sialyl-Lewis X absent), LAD-III (FERMT3: Kindlin-3 absent → integrin inside-out activation fails + Glanzmann platelet defect). All are AR; HSCT curative for severe LAD-I and LAD-III."},
            {"term": "ITGB2/LAD-I (Leukocyte Adhesion Deficiency type I)", "definition": "AR ITGB2 (21q22.3; 769aa). CD18/ITGB2 is the β2 integrin subunit shared by LFA-1 (αLβ2/CD11a-CD18), Mac-1 (αMβ2/CD11b-CD18), p150,95 (αXβ2/CD11c-CD18), and αDβ2 (CD11d-CD18). Without CD18, all four β2-integrin heterodimers fail surface expression → neutrophils cannot bind ICAM-1/ICAM-2 on inflamed endothelium → zero PMN extravasation. DIAGNOSIS: CD18 absent on PMN surface by flow cytometry. SEVERITY: Severe (<1% CD18) = neonatal omphalitis + early death without HSCT; Moderate (1-10%) = recurrent severe infections; Mild (>10%) = periodontitis-dominant. TREATMENT: HSCT = only curative therapy; gene therapy trials ongoing."},
            {"term": "FERMT3/LAD-III (Leukocyte Adhesion Deficiency type III)", "definition": "AR FERMT3 (11q13.1; 667aa). Kindlin-3 is the hematopoietic integrin co-activator that binds β1, β2, and β3 integrin cytoplasmic tails to enable inside-out activation (talin + kindlin cooperative mechanism converts low-affinity bent-closed integrin to high-affinity extended-open conformation). Without Kindlin-3: β2 integrins (PMN) cannot activate (→ LAD phenotype despite NORMAL CD18 EXPRESSION) AND β3 integrins (αIIbβ3 on platelets) cannot activate (→ Glanzmann thrombasthenia-like bleeding). KEY DISTINGUISHER: CD18 expression NORMAL by flow (DDx from LAD-I). Combined infections + bleeding = FERMT3 signature. HSCT mandatory."},
            {"term": "DADA2 (Deficiency of ADA2 / ADA2 Deficiency)", "definition": "AR ADA2/CECR1 (22q11.1; 511aa). ADA2 (Adenosine Deaminase 2) is a secreted enzyme that converts extracellular adenosine to inosine and acts as a growth factor for myeloid/vascular homeostasis. Without ADA2: adenosine accumulates → monocyte hyperactivation → M1 macrophage vasculitis (PAN-like). DADA2 described in 2014. CLINICAL TRIAD: PAN-like vasculopathy + early stroke + livedo reticularis. NK cell lymphopenia. TREATMENT: TNF-INHIBITORS (etanercept/adalimumab) are curative for vasculopathy — START IMMEDIATELY to prevent irreversible CNS damage. HSCT for refractory cytopenias only (not first-line for vasculopathy)."},
            {"term": "Specific Granule Deficiency (SGD / CEBPE Deficiency)", "definition": "AR CEBPE (14q11.2; 317aa). C/EBPε is the myeloid-specific transcription factor driving secondary (specific) granule biogenesis. SGD = complete absence of secondary and tertiary neutrophil granules; primary (azurophil/MPO) granules intact. PATHOGNOMONIC: bilobed PMN nuclei on blood smear + absent lactoferrin (granule protein assay) + NORMAL NBT/DHR (not CGD). Infections: identical to CGD pathogens (Staphylococcus, Aspergillus, gram-negatives) but OXIDASE TESTS NORMAL — the normal NBT/DHR is the key clue pointing to SGD vs CGD. Treatment: TMP-SMX + antifungal prophylaxis; HSCT for severe cases."},
            {"term": "CXCR2 Deficiency (SCN6 / Neutrophil Chemotaxis Failure)", "definition": "AR CXCR2 (2q35; 360aa). CXCR2 (IL-8 receptor B/CD182) mediates neutrophil mobilization from bone marrow and chemotaxis toward infection sites in response to CXCL8/IL-8 and related GRO chemokines (CXCL1-3, 5-6). Without CXCR2: neutrophils cannot exit bone marrow sinusoids (CXCR4 retention unopposed) → peripheral neutropenia with NORMAL MARROW RESERVE. TREATMENT: G-CSF EFFECTIVE (mobilizes via G-CSFR, bypassing absent CXCR2). Inflammatory neutrophilic skin disease treated with ruxolitinib. DIAGNOSIS: CD182 absent on PMN flow + biallelic CXCR2 variants."},
            {"term": "MBL Deficiency (MBL2 Deficiency / Lectin Pathway PR Deficiency)", "definition": "Functional MBL2 deficiency from AR structural variants at codons 52 (p.R52C), 54 (p.G54D), 57 (p.G57E) causing unstable monomers that fail oligomerization; promoter variants (H, L, X, Y haplotypes) reduce expression. MBL oligomers cannot form → MASP1/MASP2 not recruited → lectin pathway inactive. PARTIAL IMMUNODEFICIENCY: significant in infancy, during immunosuppression, or with concurrent humoral deficiency. CH50/AH50 NORMAL; lectin-specific assay on mannan-coated plates reduced. MBL serum < 100 ng/mL diagnostic. Recombinant MBL replacement trials ongoing."},
            {"term": "MASP2 Deficiency (MBL-Associated Serine Protease 2 Deficiency)", "definition": "AR MASP2 (1p36.22; 686aa). MASP2 is the SOLE EFFECTOR ENZYME of the lectin complement pathway: cleaves C4 → C4a + C4b and C2 → C2a + C2b, forming C3 convertase C4b2a. Without MASP2: MBL and ficolins can bind carbohydrate patterns but cannot activate complement (no C4/C2 cleavage). CH50 NORMAL; AH50 NORMAL; lectin pathway absent. Pyogenic infections + RA-like autoimmunity. MASP2 inhibitor narsoplimab (OMS721, approved for HSCT-TMA) is CONTRAINDICATED in MASP2 deficiency (opposite use case: inhibitor for overactive MASP2 vs deficiency LOF). Very rare (<30 families globally)."},
            {"term": "3MC Syndrome (COLEC11 / Collectin-11 Deficiency)", "definition": "AR COLEC11 (2p25.3; 344aa). Collectin-11 (CL-K1) activates lectin complement pathway and guides embryonic neural crest cell migration. 3MC syndrome (Malpuech-Michels-Mingarelli-Carnevale) caused by COLEC11 (type 2) or MASP1 (type 1) mutations. PATHOGNOMONIC DEVELOPMENTAL FEATURES: craniosynostosis + posterior embryotoxon (Axenfeld anomaly) + cleft lip/palate + sensorineural hearing loss + caudal dysgenesis. Immune component: reduced lectin pathway → recurrent childhood infections. Treatment: structural anomaly repair (craniofacial surgery, hearing aids) + prophylactic antibiotics. Ophthalmology assessment for glaucoma risk (posterior embryotoxon/Axenfeld)."},
            {"term": "Lectin Complement Pathway", "definition": "The lectin complement pathway is activated by pattern recognition molecules — MBL (MBL2), Ficolins (FCN1/FCN2/FCN3), and Collectins (CL-K1/COLEC11, CL-L1/COLEC10) — that bind carbohydrate arrays (mannose, GlcNAc, fucose, acetyl groups) on pathogen surfaces. After binding: MASP1 autoactivates → MASP1 trans-activates MASP2 → MASP2 cleaves C4 and C2 → C4b2a C3 convertase forms → C3 cleaved to C3a + C3b → opsonization + C5-C9 MAC formation. DIAGNOSTIC FINGERPRINT: CH50 normal (classical pathway), AH50 normal (alternative pathway), but LECTIN PATHWAY ASSAY on mannan-coated plates = absent (for MBL2, MASP2, COLEC11 deficiency)."},
            {"term": "CD18 Flow Cytometry — LAD Diagnosis", "definition": "DIAGNOSTIC PROTOCOL FOR LAD-I: Collect EDTA blood; prepare white blood cells; incubate with FITC-conjugated anti-CD18 (clone 6.7 or equivalent) + anti-CD11b (Mac-1) for neutrophil identification; gate on SSC-high granulocytes; NORMAL CD18 expression: >95% CD18+ granulocytes; LAD-I (ITGB2 LOF): < 2% CD18+ neutrophils (absent surface CD18) = PATHOGNOMONIC; DDx from LAD-III: CD18 NORMAL SURFACE EXPRESSION in LAD-III (FERMT3 LOF) — do not stop at CD18 flow if LAD-III suspected; also order β2-integrin activation assay (stimulated CD11b upregulation after PMA or fMLP) for LAD-III diagnosis."},
            {"term": "Cascade Testing — LAD/DADA2/Granule/Lectin Pathway", "definition": "STEP 1: Delayed cord separation/omphalitis → CD18 flow (absent = LAD-I ITGB2 sequencing; normal = LAD-III FERMT3 sequencing + β2-integrin activation assay). STEP 2: Child with stroke + livedo reticularis → ADA2 enzyme activity + NK cell count + ADA2/CECR1 sequencing → TNF-inhibitor immediately. STEP 3: Recurrent Staph/Aspergillus + NBT/DHR NORMAL → blood film (bilobed nuclei = SGD) → lactoferrin assay → CEBPE sequencing. STEP 4: Neutropenia + G-CSF responsive + normal marrow → CXCR2 flow (CD182) → CXCR2 sequencing. STEP 5: Recurrent infections + CH50 normal + AH50 normal → lectin pathway assay → MBL level + MASP2 ELISA → COLEC11/MASP1 if 3MC features."},
        ],
        "key_distinguishers": {
            "lad1_vs_lad3":           "LAD-I (ITGB2): CD18 ABSENT by flow (< 2% expression); no platelet dysfunction; ITGB2 mutations. LAD-III (FERMT3): CD18 EXPRESSION NORMAL by flow — the LAD-I/LAD-III DDx requires checking CD18 expression FIRST; if normal → functional activation assay (stimulated CD11b upregulation absent in LAD-III) + platelet aggregation (absent in LAD-III from β3 dysfunction); FERMT3 sequencing.",
            "dada2_vs_pan_kawasaki":  "DADA2: childhood stroke + livedo reticularis + NK lymphopenia + ADA2 enzyme absent + biallelic ADA2 mutations → anti-TNF curative. Classic PAN: adult onset, ANCA-variable, NK normal, steroids + cyclophosphamide. Kawasaki: fever > 5 days, coronary aneurysms, NK normal, self-limited, IVIG + aspirin.",
            "sgd_vs_cgd":             "SGD (CEBPE): NBT/DHR NORMAL (NADPH oxidase intact); bilobed PMN nuclei; absent lactoferrin; same pathogens as CGD. CGD: NBT/DHR ABSENT (NADPH oxidase deficient); normal PMN nuclear shape; normal lactoferrin. Both: Staphylococcus, Aspergillus, gram-negatives. KEY TEST: DHR flow cytometry FIRST — CGD = flat; SGD = normal histogram.",
            "cxcr2_vs_elane_scn":     "CXCR2 (SCN6): normal marrow myeloid reserve (exit failure); CXCR2 absent on PMN flow; inflammatory skin; G-CSF responsive. ELANE (SCN1): maturation arrest at promyelocyte/myelocyte on marrow biopsy; ELANE sequencing; cyclic neutropenia variant possible. HAX1 (Kostmann): maturation arrest; HAX1 sequencing; neurological features. All SCN types: G-CSF responsive.",
            "lectin_pathway_ddx":     "MBL2: serum MBL < 100 ng/mL + structural variant codons 52/54/57; partial immune deficiency; CH50/AH50 normal. MASP2: serum MASP2 absent by ELISA; lectin pathway C4 deposition absent; pyogenic infections + RA-like; narsoplimab contraindicated. COLEC11: 3MC facial/structural features; posterior embryotoxon; craniofacial developmental anomalies; COLEC11 + MASP1 sequencing.",
        },
        "treatment_matrix": {
            "hsct":               "LAD-I (severe): HSCT mandatory and curative (only definitive treatment; MSD > MUD; myeloablative preferred for full chimerism). LAD-III (FERMT3): HSCT mandatory and curative (corrects PMN Kindlin-3 AND platelet αIIbβ3 activation). CEBPE (severe SGD): HSCT curative. CXCR2 (refractory): HSCT option. ADA2, MBL2, MASP2, COLEC11: HSCT not indicated as primary therapy.",
            "anti_tnf":           "DADA2 (ADA2 deficiency): TNF-inhibitors (etanercept SC BIW or adalimumab SC every 2 weeks) CURATIVE for vasculopathy — START IMMEDIATELY when DADA2 confirmed; prevents progressive strokes and organ damage; lifelong therapy; monitor for TB/infections during anti-TNF.",
            "gcsf":               "CXCR2 (SCN6): G-CSF first-line and effective (bypasses absent CXCR2 via G-CSFR). LAD-I/III: G-CSF NOT effective (makes more non-adherent PMNs). SGD: variable G-CSF benefit. MBL2/MASP2/COLEC11: G-CSF not indicated.",
            "prophylactic_antibiotics": "LAD-I + LAD-III: TMP-SMX + antifungal prophylaxis (voriconazole/itraconazole); penicillin for Streptococcal cover. SGD: TMP-SMX + antifungal (same as CGD prophylaxis). CXCR2: TMP-SMX. MBL2/MASP2/COLEC11: antibiotics during high-risk periods or recurrent infections.",
            "masp2_inhibitor_contraindicated": "Narsoplimab (OMS721, MASP2 inhibitor) = CONTRAINDICATED in MASP2 DEFICIENCY (no MASP2 to inhibit); approved for HSCT-TMA where MASP2 is overactive; drug is for MASP2 GOF/overactivation states, not LOF. Similarly: never use MASP2 inhibitor in MBL2 or COLEC11 deficiency.",
        },
    }
