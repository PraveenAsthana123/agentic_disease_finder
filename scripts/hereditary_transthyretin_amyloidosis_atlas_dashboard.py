#!/usr/bin/env python3
"""Hereditary-Transthyretin-Amyloidosis-Atlas -- Complete 8-Gene Reference
Hereditary systemic amyloidoses (ATTR TTR, AApoAI APOA1, AFib FGA,
ALys LYZ, AGel GSN, AApoAII APOA2, AB2M B2M, PLCA OSMR):

TTR    (Transthyretin; 127aa; 18q12.1; AD/AR GOF amyloidogenic;
         Most-common hereditary amyloidosis; V30M polyneuropathy-dominant;
         V122I African-American cardiac-dominant; T60A Irish-founder;
         Tafamidis/Patisiran/Inotersen/Vutrisiran/Eplontersen FDA-approved;
         99mTc-DPD-scan PATHOGNOMONIC cardiac-ATTR;
         seed SEED_BASE+0).
APOA1  (Apolipoprotein A-I; 267aa; 11q23.3; AD LOF amyloidogenic;
         Multi-organ: renal+hepatic+cardiac+GI+peripheral-neuropathy;
         HDL-C markedly-reduced PATHOGNOMONIC;
         Fat-pad-biopsy Congo-red GOLD-STANDARD;
         seed SEED_BASE+1).
FGA    (Fibrinogen alpha chain; 866aa; 4q28.1; AD LOF amyloidogenic;
         Most-common hereditary RENAL amyloidosis;
         Glu526Val most-common variant; heavy-proteinuria->ESRD;
         NO peripheral-neuropathy distinguishes from TTR;
         Liver-transplant CURATIVE prevents progression;
         seed SEED_BASE+2).
LYZ    (Lysozyme; 148aa; 12q15; AD GOF amyloidogenic;
         Non-neuropathic systemic: liver+kidney+spleen+GI;
         Hepatomegaly+splenomegaly PATHOGNOMONIC combination;
         Asp67His D67H and Ile56Thr I56T UK-founders;
         seed SEED_BASE+3).
GSN    (Gelsolin; 782aa; 9q33.2; AD GOF amyloidogenic;
         Finnish-type/Meretoja-syndrome; D187N Finnish-founder;
         MERETOJA TRIAD: lattice-corneal-dystrophy-II + facial-nerve-palsy + cutis-laxa PATHOGNOMONIC;
         seed SEED_BASE+4).
APOA2  (Apolipoprotein A-II; 100aa; 1q23.3; AD LOF amyloidogenic;
         Stop-codon-Ter94 amyloidogenic-truncation;
         Renal-dominant: proteinuria->ESRD; late-onset 50-70yr;
         HDL reduced but less than APOA1;
         seed SEED_BASE+5).
B2M    (Beta-2 microglobulin; 119aa; 15q21.1; AR rare-hereditary / acquired-DRA;
         D76N hereditary systemic; dialysis-related AB2M;
         Carpal-tunnel PATHOGNOMONIC in chronic-dialysis >10yr;
         High-flux-dialysis/hemodiafiltration reduces AB2M;
         seed SEED_BASE+6).
OSMR   (Oncostatin-M receptor beta; 979aa; 5p13.1; AD GOF;
         Primary-localized-cutaneous-amyloidosis PLCA;
         Intensely-pruritic papular deposits interscapular/extremities PATHOGNOMONIC;
         NO systemic-organ involvement distinguishes from all-others;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3782-3789)
"""
import random

SEED_BASE = 3782

ATLAS_GENES = [
    {
        "gene": "TTR",
        "protein": (
            "TTR -- 18q12.1 AD amyloidogenic GOF -- 127aa -- "
            "Transthyretin-tetrameric-transport-protein-thyroid-hormone-retinol-"
            "V30M-neuropathy-dominant-Portuguese-Swedish-V122I-cardiac-African-American-"
            "T60A-Irish-founder-E89Q-South-Asian-cardiac-ATTR-OMIM-176300"
        ),
        "locus": "18q12.1",
        "protein_size": (
            "127 aa / ~14 kDa monomer (~56 kDa tetramer) / 18q12.1 TTR transthyretin molecular context: "
            "STRUCTURE: "
            "  127 aa / ~14 kDa; forms obligate homotetramer (4 x 14kDa = 56kDa); "
            "  T4 and retinol-RBP binding protein; synthesized primarily in liver; "
            "  Pathogenic variants destabilize tetramer -> dissociation -> monomer misfolding -> "
            "  beta-sheet polymerization -> amyloid fibril deposition; "
            "HEREDITARY ATTR (ATTRv): "
            "  >130 pathogenic variants identified; predominantly AD; "
            "  V30M (Val30Met, p.Val50Met): most common worldwide; Portuguese/Swedish founders; "
            "  V122I (Ile122Val, p.Ile142Val): 3-4% African Americans; cardiac-dominant; "
            "  T60A (Thr60Ala, p.Thr80Ala): Irish founder; mixed cardiac+neuropathic; "
            "  I68L (Ile68Leu): Appalachian USA; cardiac dominant; "
            "  E89Q (Glu89Gln): South Asian/Pakistani; cardiac dominant; "
            "CARDIAC ATTR (ATTRv+ATTRwt): "
            "  Symmetrical ventricular hypertrophy (septal+posterior equally); "
            "  Granular sparkling appearance on echo (NOT pathognomonic alone); "
            "  APICAL SPARING on longitudinal strain GLS = PATHOGNOMONIC for cardiac ATTR; "
            "  GLS ratio (apex:basal+mid) > 1 = PATHOGNOMONIC; "
            "  99mTc-DPD/PYP bone scintigraphy: Grade 2-3 cardiac uptake PATHOGNOMONIC (Perugini scale); "
            "  Serum/urine protein electrophoresis: EXCLUDE AL amyloid (plasma cell clone); "
            "NEUROPATHIC ATTR (ATTRv V30M): "
            "  Length-dependent small fiber > large fiber; "
            "  Pain/temperature loss feet first; autonomic failure (orthostasis, constipation, impotence); "
            "  Bilateral carpal tunnel syndrome EARLY MANIFESTATION; "
            "  Progressive to walking aid 10yr, wheelchair 15yr without treatment; "
            "PRECISION THERAPIES (FDA-APPROVED): "
            "  Tafamidis (FDA 2019): TTR tetramer stabilizer; slows cardiac ATTRv+ATTRwt; "
            "  Patisiran (FDA 2018, EMA 2018): RNAi hepatocyte targeting TTR mRNA; polyneuropathy; "
            "  Inotersen (FDA 2018, EMA 2018): ASO targeting TTR mRNA; polyneuropathy; "
            "  Vutrisiran (FDA 2022): RNAi; non-inferior to patisiran; less frequent dosing; "
            "  Eplontersen (FDA 2023): ASO; TTR reduction >90%; polyneuropathy+cardiac; "
            "  Liver transplant: curative for V30M polyneuropathy if early (prevents new deposits); "
            "DIAGNOSIS: "
            "  Serum/urine SPEP+IFE to exclude AL; "
            "  99mTc-DPD/PYP scan: Grade 2-3 = ATTR cardiac amyloidosis (without clonal plasma cells); "
            "  Endomyocardial biopsy: Congo red + apple-green birefringence (polarized); "
            "  TTR gene sequencing: distinguishes ATTRv from ATTRwt; "
            "  TTR tetramer stability assay"
        ),
        "syndrome": (
            "ATTR / hereditary-transthyretin-amyloidosis / "
            "V30M-polyneuropathy / V122I-cardiac-African-American / "
            "T60A-Irish / E89Q-South-Asian-cardiac"
        ),
        "inheritance": "AD (most variants) / rare AR",
        "key_therapy": "Tafamidis/Patisiran/Inotersen/Vutrisiran/Eplontersen FDA-approved",
        "pathognomonic": (
            "Apical-sparing-longitudinal-strain (GLS ratio apex:base >1) on echo PATHOGNOMONIC cardiac-ATTR; "
            "99mTc-DPD/PYP Grade-2-3 bone-scintigraphy PATHOGNOMONIC cardiac-ATTR without clonal plasma cells"
        ),
        "key_drugs_ci": [
            "DIGOXIN-AVOID-binds-amyloid-fibrils-unpredictable-toxicity",
            "AMIODARONE-AVOID-QTc-prolongation-conduction-disease",
            "VERAPAMIL-DILTIAZEM-AVOID-cardiac-ATTR-conduction-block",
            "SPIRONOLACTONE-AVOID-cardiac-ATTR-hypotension-renal-failure",
            "Tafamidis-CARDIAC-ATTRv+ATTRwt-FDA2019",
            "Patisiran/Vutrisiran-POLYNEUROPATHY-FDA2018/2022",
        ],
    },
    {
        "gene": "APOA1",
        "protein": (
            "APOA1 -- 11q23.3 AD amyloidogenic LOF -- 267aa -- "
            "Apolipoprotein-A-I-HDL-major-structural-protein-"
            "Arg173Pro-Iowa-Leu75Pro-Trp50Arg-multi-organ-"
            "renal-hepatic-cardiac-GI-peripheral-neuropathy-AApoAI-OMIM-107680"
        ),
        "locus": "11q23.3",
        "protein_size": (
            "267 aa / ~28 kDa / 11q23.3 APOA1 apolipoprotein A-I molecular context: "
            "STRUCTURE: "
            "  267 aa / ~28 kDa; major structural protein of HDL; "
            "  11 helix repeats; promotes cholesterol efflux via ABCA1; "
            "  Amyloidogenic variants destabilize helical structure -> "
            "  expose hydrophobic core -> fibrillation; "
            "  HDL assembly impaired -> marked HDL-C reduction PATHOGNOMONIC; "
            "HEREDITARY AApoAI AMYLOIDOSIS: "
            "  Variants: Arg173Pro (Iowa), Leu75Pro (Italian), Trp50Arg (various); "
            "  Gly26Arg (German), Ala26Thr; >20 pathogenic variants identified; "
            "MULTI-ORGAN INVOLVEMENT: "
            "  Renal: proteinuria -> nephrotic -> ESRD (most common presentation); "
            "  Hepatic: hepatomegaly, liver failure (hepatic variant); "
            "  Cardiac: conduction disease, HF; "
            "  Peripheral neuropathy (distinguishes from FGA/LYZ); "
            "  GI: malabsorption, diarrhea; "
            "  Skin: papular deposits (less common); "
            "HDL-C MARKEDLY REDUCED PATHOGNOMONIC: "
            "  Most AApoAI variants -> HDL-C near-zero or very low; "
            "  HDL-C <10 mg/dL with amyloidosis -> test APOA1 gene; "
            "  Differs from Tangier disease (ABCA1): Tangier = near-zero-HDL + "
            "  orange-tonsil-PATHOGNOMONIC (no amyloidosis); "
            "DIAGNOSIS: "
            "  Fat pad biopsy / rectal biopsy / renal biopsy: Congo red + apple-green birefringence; "
            "  Mass spectrometry fibril typing: differentiates AL/ATTR/AApoAI; "
            "  APOA1 gene sequencing; "
            "  HDL-C level: near-zero suggests APOA1 variant; "
            "TREATMENT: "
            "  No approved therapy; liver transplant for hepatic-dominant; "
            "  Combined liver+kidney transplant if both organs failing; "
            "  Renal replacement for ESRD; "
            "  HDL-raising NOT effective (source is defective APOA1)"
        ),
        "syndrome": (
            "AApoAI / hereditary-apolipoprotein-A-I-amyloidosis / "
            "multi-organ-renal-hepatic-cardiac / HDL-C-markedly-reduced-PATHOGNOMONIC"
        ),
        "inheritance": "AD LOF amyloidogenic",
        "key_therapy": "Liver transplant (hepatic-dominant); combined liver+kidney transplant",
        "pathognomonic": (
            "HDL-C markedly-reduced (near-zero) with systemic amyloidosis PATHOGNOMONIC AApoAI; "
            "Fat-pad-biopsy Congo-red apple-green birefringence GOLD-STANDARD"
        ),
        "key_drugs_ci": [
            "HDL-RAISING-AGENTS-INEFFECTIVE-defective-APOA1-source",
            "NIACIN-AVOID-ineffective-hepatotoxic",
            "Fat-pad-biopsy-Congo-red-GOLD-STANDARD-diagnosis",
            "Mass-spectrometry-fibril-typing-MANDATORY-exclude-AL",
        ],
    },
    {
        "gene": "FGA",
        "protein": (
            "FGA -- 4q28.1 AD amyloidogenic LOF -- 866aa -- "
            "Fibrinogen-alpha-chain-clotting-factor-I-A-chain-"
            "Glu526Val-E526V-most-common-renal-dominant-NO-neuropathy-"
            "liver-transplant-CURATIVE-AFib-OMIM-134820"
        ),
        "locus": "4q28.1",
        "protein_size": (
            "866 aa / ~95 kDa (Aalpha chain) / 4q28.1 FGA fibrinogen alpha chain molecular context: "
            "STRUCTURE: "
            "  866 aa / ~95 kDa; Fibrinogen = (Aalpha)2(Bbeta)2(gamma)2 hexamer; "
            "  Amyloidogenic variants in C-terminal fibrinogen-like domain (residues 500-600); "
            "  Truncated or misfolded C-terminal fragment deposits in renal glomeruli; "
            "  Fibrinogen synthesized exclusively in hepatocytes -> liver-transplant CURATIVE; "
            "HEREDITARY AFib AMYLOIDOSIS: "
            "  Most common hereditary RENAL amyloidosis; "
            "  Glu526Val (E526V): most common variant worldwide; "
            "  Arg554Leu (R554L): 2nd most common; "
            "  Val526Glu, Gly534Val, Gly538Glu also described; "
            "  All variants cluster in C-terminal fibrinogen-like domain; "
            "RENAL-DOMINANT PHENOTYPE: "
            "  Heavy proteinuria (nephrotic-range) is presenting feature; "
            "  Proteinuria -> nephrotic syndrome -> ESRD within 5-15yr; "
            "  No peripheral neuropathy (KEY DISTINGUISHER FROM TTR/APOA1); "
            "  No cardiac amyloidosis (KEY DISTINGUISHER); "
            "  Renal biopsy: glomerular amyloid deposits; "
            "  Normal serum fibrinogen or slightly elevated; "
            "  Fibrinogen coagulation function preserved (amyloidogenic fragment ≠ clotting fragment); "
            "LIVER TRANSPLANT CURATIVE: "
            "  Liver produces 100% of fibrinogen; "
            "  Transplant removes amyloidogenic FGA source; "
            "  Prevents further renal amyloid deposition; "
            "  Renal function stabilizes post-transplant; "
            "  Combined liver+kidney transplant if ESRD already present; "
            "DIAGNOSIS: "
            "  Renal biopsy: Congo red + apple-green birefringence + mass spectrometry fibril typing; "
            "  Mass spectrometry: AFib fibril type (contains fibrinogen peptides); "
            "  FGA gene sequencing; "
            "  Exclude AL: serum/urine SPEP+IFE MANDATORY; "
            "TREATMENT: "
            "  Liver transplant: CURATIVE - prevents progression; "
            "  TIMING CRITICAL: before ESRD for best renal outcome; "
            "  Renal replacement: bridging to transplant; "
            "  No approved medical therapy for AFib amyloidosis"
        ),
        "syndrome": (
            "AFib / hereditary-fibrinogen-amyloidosis / "
            "renal-dominant-ESRD / NO-neuropathy / Glu526Val-most-common / "
            "liver-transplant-CURATIVE"
        ),
        "inheritance": "AD LOF amyloidogenic",
        "key_therapy": "Liver transplant CURATIVE (removes amyloidogenic source)",
        "pathognomonic": (
            "Renal-dominant proteinuria->ESRD with NO peripheral neuropathy DISTINGUISHES AFib from TTR/APOA1; "
            "Liver transplant CURATIVE as fibrinogen produced exclusively in liver"
        ),
        "key_drugs_ci": [
            "LIVER-TRANSPLANT-CURATIVE-BEFORE-ESRD-optimal-timing",
            "MASS-SPECTROMETRY-FIBRIL-TYPING-MANDATORY-distinguishes-AL-from-AFib",
            "NO-PERIPHERAL-NEUROPATHY-distinguishes-FGA-from-TTR-APOA1",
            "RENAL-BIOPSY-MANDATORY-Congo-red-fibril-typing",
        ],
    },
    {
        "gene": "LYZ",
        "protein": (
            "LYZ -- 12q15 AD amyloidogenic GOF -- 148aa -- "
            "Lysozyme-muramidase-innate-immunity-enzyme-"
            "Asp67His-D67H-Ile56Thr-I56T-UK-founders-"
            "non-neuropathic-systemic-hepatomegaly-splenomegaly-PATHOGNOMONIC-ALys-OMIM-153450"
        ),
        "locus": "12q15",
        "protein_size": (
            "148 aa / ~14 kDa / 12q15 LYZ lysozyme molecular context: "
            "STRUCTURE: "
            "  148 aa / ~14 kDa; muramidase (hydrolyzes peptidoglycan); "
            "  Innate immunity: bactericidal in secretions/neutrophil granules; "
            "  Amyloidogenic variants (D67H, I56T) reduce thermodynamic stability; "
            "  Partially unfolded intermediate deposits as amyloid; "
            "  Produced in liver (hepatocytes) and macrophages; "
            "HEREDITARY ALys AMYLOIDOSIS: "
            "  D67H (Asp67His): most common; identified in UK (Welsh/English families); "
            "  I56T (Ile56Thr): 2nd variant; UK families; milder course; "
            "  Both reduce lysozyme thermal stability by ~12-15 degrees Celsius; "
            "NON-NEUROPATHIC SYSTEMIC PHENOTYPE: "
            "  NO peripheral neuropathy (KEY DISTINGUISHER from TTR/APOA1); "
            "  Liver: hepatomegaly PROMINENT, liver failure in severe cases; "
            "  Spleen: splenomegaly PROMINENT; "
            "  Hepatomegaly + splenomegaly PATHOGNOMONIC combination for ALys; "
            "  Kidney: proteinuria/ESRD (less dominant than FGA); "
            "  GI: diarrhea, malabsorption, autonomic dysfunction; "
            "  Skin: petechiae, purpura (rare); "
            "DIAGNOSIS: "
            "  Biopsy (fat pad, liver, kidney): Congo red + mass spectrometry fibril typing; "
            "  Serum lysozyme level: may be elevated (>200 mcg/mL) -- NOT pathognomonic; "
            "  LYZ gene sequencing; "
            "  Exclude AL MANDATORY: SPEP+IFE; "
            "TREATMENT: "
            "  No approved targeted therapy; "
            "  Liver transplant considered for progressive hepatic dominant; "
            "  Organ support: renal replacement, nutritional supplementation"
        ),
        "syndrome": (
            "ALys / hereditary-lysozyme-amyloidosis / "
            "non-neuropathic-systemic / hepatomegaly-splenomegaly-PATHOGNOMONIC / "
            "D67H-I56T-UK-founders"
        ),
        "inheritance": "AD GOF amyloidogenic",
        "key_therapy": "No approved therapy; liver transplant for progressive hepatic dominant",
        "pathognomonic": (
            "Hepatomegaly+splenomegaly PATHOGNOMONIC combination for ALys amyloidosis; "
            "NO peripheral neuropathy distinguishes from TTR/APOA1"
        ),
        "key_drugs_ci": [
            "NO-APPROVED-TARGETED-THERAPY-2026",
            "HEPATOMEGALY-SPLENOMEGALY-PATHOGNOMONIC-combination",
            "MASS-SPECTROMETRY-FIBRIL-TYPING-MANDATORY",
            "LIVER-TRANSPLANT-consider-progressive-hepatic-dominant",
        ],
    },
    {
        "gene": "GSN",
        "protein": (
            "GSN -- 9q33.2 AD amyloidogenic GOF -- 782aa -- "
            "Gelsolin-actin-cytoskeleton-regulator-"
            "D187N-Finnish-founder-D187Y-MERETOJA-TRIAD-"
            "lattice-corneal-dystrophy-II-facial-nerve-palsy-cutis-laxa-PATHOGNOMONIC-AGel-OMIM-137350"
        ),
        "locus": "9q33.2",
        "protein_size": (
            "782 aa / ~80 kDa / 9q33.2 GSN gelsolin molecular context: "
            "STRUCTURE: "
            "  782 aa / ~80 kDa; actin-binding cytoskeletal regulator; "
            "  6 homologous gelsolin domains (G1-G6); Ca2+-regulated; "
            "  D187N/D187Y: furin cleavage site exposed -> 8kDa C-terminal fragment -> amyloid; "
            "  Fragment accumulates in cornea, peripheral nerves, skin, meninges; "
            "MERETOJA SYNDROME (Finnish gelsolin amyloidosis): "
            "  D187N (Asp187Asn): Finnish founder; most common worldwide; "
            "  D187Y (Asp187Tyr): found in non-Finnish families; equivalent phenotype; "
            "MERETOJA TRIAD (ALL THREE = PATHOGNOMONIC): "
            "  1. Lattice corneal dystrophy TYPE II (not Type I Biber-Haab-Dimmer): "
            "     Bilateral criss-cross/lattice opacities in corneal stroma; "
            "     TYPE II vs TYPE I: Type I = TGFBI mutation, corneal only; "
            "     Type II (Meretoja) = GSN mutation, SYSTEMIC with cranial neuropathy+skin; "
            "     Progressive corneal clouding -> visual impairment; keratoplasty needed; "
            "  2. Cranial neuropathy (facial nerve VII most prominent): "
            "     Progressive bilateral facial nerve palsy; "
            "     Also trigeminal (V), glossopharyngeal (IX), vagal (X) later; "
            "     Bulbar symptoms: dysarthria, dysphagia; "
            "  3. Cutis laxa (skin laxity): "
            "     Progressive skin laxity (pendulous folds); "
            "     Aged appearance; premature cutaneous aging; "
            "ADDITIONAL FEATURES: "
            "  Peripheral neuropathy (length-dependent) -- later in course; "
            "  Renal amyloidosis: proteinuria, mild ESRD risk; "
            "  Carpal tunnel syndrome; "
            "DIAGNOSIS: "
            "  Clinical: Meretoja triad in Finnish/Northern European person; "
            "  Slit-lamp: lattice type II corneal opacities; "
            "  Biopsy: skin, conjunctiva, nerve -- Congo red + mass spectrometry; "
            "  GSN gene sequencing: D187N/D187Y; "
            "TREATMENT: "
            "  No disease-modifying therapy; "
            "  Keratoplasty (corneal graft) for corneal opacity; "
            "  Physical therapy for facial palsy; "
            "  Renal replacement if ESRD"
        ),
        "syndrome": (
            "AGel / Finnish-type-gelsolin-amyloidosis / Meretoja-syndrome / "
            "MERETOJA-TRIAD: lattice-corneal-dystrophy-II + facial-nerve-palsy + cutis-laxa PATHOGNOMONIC"
        ),
        "inheritance": "AD GOF amyloidogenic",
        "key_therapy": "No disease-modifying therapy; keratoplasty for corneal opacity",
        "pathognomonic": (
            "MERETOJA TRIAD: lattice-corneal-dystrophy TYPE II + bilateral facial-nerve-palsy + cutis-laxa "
            "ALL THREE TOGETHER = PATHOGNOMONIC AGel gelsolin amyloidosis"
        ),
        "key_drugs_ci": [
            "LATTICE-CORNEAL-DYSTROPHY-TYPE-II-distinguishes-from-Type-I-TGFBI",
            "KERATOPLASTY-REQUIRED-corneal-opacity",
            "FACIAL-NERVE-VII-MOST-COMMON-cranial-neuropathy",
            "NO-DISEASE-MODIFYING-THERAPY-2026",
        ],
    },
    {
        "gene": "APOA2",
        "protein": (
            "APOA2 -- 1q23.3 AD amyloidogenic LOF -- 100aa -- "
            "Apolipoprotein-A-II-HDL-structural-protein-"
            "stop-codon-Ter94-amyloidogenic-truncation-"
            "renal-dominant-late-onset-50-70yr-AApoAII-OMIM-107670"
        ),
        "locus": "1q23.3",
        "protein_size": (
            "100 aa / ~8.7 kDa monomer / 1q23.3 APOA2 apolipoprotein A-II molecular context: "
            "STRUCTURE: "
            "  100 aa / ~8.7 kDa; minor structural protein of HDL (after APOA1); "
            "  Antiparallel homodimer; associates with HDL surface; "
            "  PATHOGENIC MECHANISM: stop codon at position 94 (Ter94Stop) -> "
            "  6 aa truncated C-terminus -> abnormal C-terminal peptide -> amyloidogenic; "
            "  Truncated variant loses normal C-terminal processing; "
            "HEREDITARY AApoAII AMYLOIDOSIS: "
            "  Stop codon mutations: Ter94Stop most common (multiple ethnic groups); "
            "  Japanese, Italian, European families documented; "
            "  C-terminal 6 amino acids removed -> amyloid precursor; "
            "RENAL-DOMINANT PHENOTYPE: "
            "  Proteinuria -> nephrotic syndrome -> ESRD; "
            "  Late onset: typically 50-70yr (later than TTR/APOA1); "
            "  Milder and slower than FGA renal amyloidosis; "
            "  No peripheral neuropathy; "
            "  No cardiac amyloidosis in most cases; "
            "  HDL reduced but less dramatically than AApoAI (APOA1 still normal); "
            "DIAGNOSIS: "
            "  Renal biopsy: Congo red + apple-green birefringence; "
            "  Mass spectrometry fibril typing: AApoAII subtype; "
            "  APOA2 gene sequencing: Ter94Stop; "
            "  Exclude AL: SPEP+IFE MANDATORY; "
            "TREATMENT: "
            "  No approved targeted therapy; "
            "  Renal replacement for ESRD; "
            "  Liver transplant removes amyloidogenic source (theoretical but not standard)"
        ),
        "syndrome": (
            "AApoAII / hereditary-apolipoprotein-A-II-amyloidosis / "
            "renal-dominant / stop-codon-Ter94 / late-onset-50-70yr"
        ),
        "inheritance": "AD LOF amyloidogenic",
        "key_therapy": "No approved therapy; renal replacement for ESRD",
        "pathognomonic": (
            "Late-onset (50-70yr) renal amyloidosis with stop-codon-Ter94 APOA2 mutation; "
            "Milder course than FGA/APOA1 renal amyloidosis"
        ),
        "key_drugs_ci": [
            "NO-APPROVED-THERAPY-2026",
            "MASS-SPECTROMETRY-FIBRIL-TYPING-MANDATORY-distinguish-AL",
            "LATE-ONSET-50-70yr-distinguishes-from-TTR-early-onset",
            "STOP-CODON-Ter94-PATHOGENIC-MECHANISM",
        ],
    },
    {
        "gene": "B2M",
        "protein": (
            "B2M -- 15q21.1 AR-rare-hereditary / acquired-dialysis -- 119aa -- "
            "Beta-2-microglobulin-MHC-I-light-chain-"
            "D76N-hereditary-systemic / dialysis-related-AB2M-DRA-"
            "carpal-tunnel-PATHOGNOMONIC-chronic-dialysis->10yr-OMIM-109700"
        ),
        "locus": "15q21.1",
        "protein_size": (
            "119 aa / ~11.8 kDa / 15q21.1 B2M beta-2 microglobulin molecular context: "
            "STRUCTURE: "
            "  119 aa / ~11.8 kDa; non-covalent light chain of MHC class I; "
            "  Required for MHC I surface expression; shed from cell surface; "
            "  Normally cleared by kidneys (freely filtered); "
            "  DIALYSIS-RELATED AB2M: B2M accumulates when kidneys fail; "
            "  Retained >10yr dialysis -> reaches amyloidogenic concentration; "
            "  HIGH-FLUX MEMBRANES remove B2M better than low-flux; "
            "DIALYSIS-RELATED AMYLOIDOSIS (DRA/AB2M): "
            "  Most common AB2M presentation (NOT hereditary, dialysis-acquired); "
            "  Onset: typically >5-10yr of dialysis; "
            "  CARPAL TUNNEL SYNDROME: first/most common manifestation PATHOGNOMONIC; "
            "  Bilateral CTS in dialysis patient >5yr = test for AB2M; "
            "  Destructive arthropathy: shoulder, hip, spine (spondyloarthropathy); "
            "  Subchondral cysts on X-ray PATHOGNOMONIC in dialysis patient; "
            "  Spondyloarthropathy: disk space narrowing + erosive vertebral endplates; "
            "HEREDITARY AB2M (D76N VARIANT): "
            "  D76N (Asp76Asn): autosomal dominant (NOT dialysis-required); "
            "  Extremely rare: <20 families worldwide; "
            "  Systemic amyloidosis without dialysis; visceral organs affected; "
            "  Earlier onset than dialysis-related form; "
            "B2M LEVELS: "
            "  Normal serum B2M: 1-3 mg/L; "
            "  Pre-dialysis ESRD: 20-40 mg/L; "
            "  During dialysis (low-flux): >50 mg/L associated with DRA; "
            "  High-flux/hemodiafiltration: reduces B2M to <30 mg/L; "
            "TREATMENT: "
            "  Online hemodiafiltration (HDF): best B2M removal, reduces DRA progression; "
            "  High-flux membranes: superior to low-flux; "
            "  Renal transplant: CURATIVE for dialysis-related form (restores B2M clearance); "
            "  No pharmacological therapy; "
            "  Surgery for CTS: decompression, but amyloid recurs without transplant"
        ),
        "syndrome": (
            "AB2M / dialysis-related-amyloidosis-DRA / "
            "carpal-tunnel-PATHOGNOMONIC-chronic-dialysis / "
            "D76N-hereditary-systemic-rare"
        ),
        "inheritance": "AR hereditary (D76N, very rare) / acquired dialysis-related",
        "key_therapy": "Renal transplant CURATIVE for dialysis-related; high-flux/HDF membranes",
        "pathognomonic": (
            "Bilateral carpal tunnel syndrome in chronic dialysis patient >5-10yr PATHOGNOMONIC AB2M; "
            "Subchondral cysts on X-ray in dialysis patient PATHOGNOMONIC destructive arthropathy"
        ),
        "key_drugs_ci": [
            "RENAL-TRANSPLANT-CURATIVE-dialysis-related-AB2M",
            "ONLINE-HDF-BEST-B2M-REMOVAL",
            "HIGH-FLUX-MEMBRANES-superior-to-low-flux",
            "BILATERAL-CTS-DIALYSIS->10yr-TEST-B2M",
        ],
    },
    {
        "gene": "OSMR",
        "protein": (
            "OSMR -- 5p13.1 AD GOF -- 979aa -- "
            "Oncostatin-M-receptor-beta-IL-31-receptor-beta-"
            "Glu356Lys-Pro694Leu-Thr914Ile-"
            "primary-localized-cutaneous-amyloidosis-PLCA-"
            "pruritic-papular-deposits-interscapular-PATHOGNOMONIC-NO-systemic-organs-OMIM-615537"
        ),
        "locus": "5p13.1",
        "protein_size": (
            "979 aa / ~110 kDa / 5p13.1 OSMR oncostatin-M receptor beta molecular context: "
            "STRUCTURE: "
            "  979 aa / ~110 kDa; type I cytokine receptor; "
            "  Shared subunit of OSM-Rbeta complex (pairs with gp130) and IL-31 receptor (pairs with IL-31RA); "
            "  IL-31 signaling: major pruritogen pathway; "
            "  OSMR GOF variants -> enhanced IL-31 signaling -> keratinocyte stress -> "
            "  keratin intermediate filament aggregation -> amyloid deposition in dermis; "
            "PRIMARY LOCALIZED CUTANEOUS AMYLOIDOSIS (PLCA): "
            "  OSMR Glu356Lys: original variant; Chinese families; "
            "  OSMR Pro694Leu: associated with macular/lichen amyloidosis; "
            "  OSMR Thr914Ile; multiple variants in exons 7-8; "
            "  Autosomal dominant with variable expressivity; "
            "  Can also occur sporadically (IL31RA mutations also cause PLCA); "
            "CLINICAL FEATURES: "
            "  INTENSELY PRURITIC papular deposits: hallmark presentation; "
            "  Distribution: interscapular (upper back) PATHOGNOMONIC location; "
            "  Also: shin, forearms, extremities (less specific); "
            "  Skin: hyperpigmented pruritic papules and plaques; "
            "  Rippled/wavy pigmentation pattern (macular amyloidosis subtype); "
            "  Nodular lichen-type variant: larger papules/plaques; "
            "NO SYSTEMIC ORGAN INVOLVEMENT: "
            "  Absolutely NO cardiac amyloidosis; "
            "  Absolutely NO renal amyloidosis; "
            "  Absolutely NO neuropathy; "
            "  This distinguishes PLCA/OSMR from ALL other amyloidosis types; "
            "  Prognosis: excellent for survival; quality of life impacted by pruritus; "
            "AMYLOID TYPE: "
            "  Amyloid K (keratin-derived intermediate filaments); "
            "  NOT serum-derived amyloid (unlike TTR/APOA1/FGA/LYZ/GSN/APOA2/B2M); "
            "SKIN BIOPSY: "
            "  Amorphous eosinophilic deposits in papillary dermis; "
            "  Congo red positive + apple-green birefringence; "
            "  Cytokeratin immunostaining positive (confirms keratin origin); "
            "TREATMENT: "
            "  No curative therapy; "
            "  Topical corticosteroids: mild benefit for pruritus; "
            "  Topical tacrolimus/pimecrolimus: some benefit; "
            "  Antihistamines: limited benefit for itch; "
            "  Narrowband UVB phototherapy: reduces pruritus+pigmentation; "
            "  IL-31/IL-31RA pathway: nemolizumab (anti-IL-31RA) investigational; "
            "  Dermabrasion/CO2 laser: cosmetic, not curative"
        ),
        "syndrome": (
            "PLCA / primary-localized-cutaneous-amyloidosis / "
            "OSMR-GOF / interscapular-pruritic-papules-PATHOGNOMONIC / "
            "NO-systemic-organs-distinguishes-from-all-others"
        ),
        "inheritance": "AD GOF",
        "key_therapy": "No curative therapy; narrowband UVB; nemolizumab (IL-31RA) investigational",
        "pathognomonic": (
            "Intensely pruritic papular deposits on INTERSCAPULAR skin PATHOGNOMONIC PLCA/OSMR; "
            "NO systemic organ involvement DISTINGUISHES PLCA from ALL other hereditary amyloidoses"
        ),
        "key_drugs_ci": [
            "NO-SYSTEMIC-ORGAN-INVOLVEMENT-distinguishes-from-ALL-others",
            "NARROWBAND-UVB-REDUCES-PRURITUS",
            "NEMOLIZUMAB-IL31RA-INVESTIGATIONAL",
            "KERATIN-ORIGIN-amyloid-K-NOT-serum-derived",
        ],
    },
]

DEFINITIONS = {
    "ATTR": (
        "Amyloid transthyretin -- hereditary (ATTRv) or wildtype (ATTRwt); "
        "most common hereditary amyloidosis; "
        "V30M polyneuropathy, V122I cardiac African-American, T60A Irish-founder; "
        "FDA therapies: Tafamidis/Patisiran/Inotersen/Vutrisiran/Eplontersen; "
        "99mTc-DPD/PYP scan Grade 2-3 PATHOGNOMONIC cardiac ATTR"
    ),
    "ATTRv": (
        "Hereditary transthyretin amyloidosis -- TTR gene mutation; "
        "AD; >130 variants; V30M, V122I, T60A, I68L, E89Q commonest; "
        "Distinguished from ATTRwt by gene sequencing"
    ),
    "ATTRwt": (
        "Wild-type (senile/age-related) transthyretin amyloidosis -- "
        "no TTR gene mutation; male predominance >75yr; "
        "Cardiac HF + AF + bilateral CTS; not hereditary but clinically important"
    ),
    "AApoAI": (
        "Apolipoprotein A-I amyloidosis -- APOA1 mutation; "
        "multi-organ (renal+hepatic+cardiac+GI+neuropathic); "
        "HDL-C near-zero PATHOGNOMONIC; liver transplant for hepatic-dominant"
    ),
    "AFib": (
        "Fibrinogen alpha chain amyloidosis -- FGA mutation; "
        "most common hereditary renal amyloidosis; Glu526Val most common; "
        "NO neuropathy; liver transplant CURATIVE"
    ),
    "ALys": (
        "Lysozyme amyloidosis -- LYZ mutation; "
        "non-neuropathic systemic; D67H and I56T UK-founders; "
        "hepatomegaly+splenomegaly PATHOGNOMONIC combination"
    ),
    "AGel": (
        "Gelsolin amyloidosis -- GSN mutation; Finnish-type / Meretoja syndrome; "
        "MERETOJA TRIAD: lattice corneal dystrophy type II + facial nerve palsy + cutis laxa PATHOGNOMONIC; "
        "D187N Finnish founder"
    ),
    "AApoAII": (
        "Apolipoprotein A-II amyloidosis -- APOA2 stop codon Ter94; "
        "renal-dominant late-onset 50-70yr; mild course; "
        "no neuropathy or cardiac involvement"
    ),
    "AB2M": (
        "Beta-2 microglobulin amyloidosis -- dialysis-related (DRA) or rare hereditary D76N; "
        "bilateral carpal tunnel syndrome in dialysis patient >5-10yr PATHOGNOMONIC; "
        "high-flux HDF reduces B2M; renal transplant CURATIVE for DRA"
    ),
    "PLCA": (
        "Primary localized cutaneous amyloidosis -- OSMR GOF mutation; "
        "pruritic papular deposits interscapular PATHOGNOMONIC; "
        "NO systemic organs; amyloid K (keratin-derived); nemolizumab investigational"
    ),
    "DPD-scan": (
        "99mTc-DPD (diphosphono-propanodicarboxylic acid) bone scintigraphy; "
        "Grade 2-3 cardiac uptake (Perugini scale) PATHOGNOMONIC for cardiac ATTR "
        "(ATTRv + ATTRwt) without clonal plasma cells; "
        "If AL excluded by SPEP+IFE+FLC: Grade 2-3 DPD = non-biopsy diagnosis of ATTR cardiac amyloidosis"
    ),
    "Apical-sparing": (
        "Longitudinal strain pattern on speckle-tracking echocardiography; "
        "GLS ratio (apex / basal+mid) > 1.0 = PATHOGNOMONIC cardiac amyloidosis; "
        "Amyloid deposits spare apex (no coronary supply restriction); "
        "Normal LV: uniform strain gradient; amyloid: base+mid severely impaired, apex relatively preserved"
    ),
    "Congo-red": (
        "Histochemical stain for amyloid -- deposits appear apple-green under polarized light; "
        "Gold standard for tissue diagnosis; "
        "Applied to: fat pad biopsy, rectal biopsy, organ biopsy, bone marrow; "
        "Mass spectrometry fibril typing MANDATORY on Congo-red+ specimen to determine amyloid type"
    ),
    "Mass-spectrometry-fibril-typing": (
        "Laser capture microdissection + mass spectrometry on amyloid deposits; "
        "Definitively identifies amyloid fibril protein type (TTR, APOA1, FGA, LYZ, GSN, etc.); "
        "MANDATORY to distinguish AL (plasma cell origin) from hereditary (gene mutation) types; "
        "Replaces immunohistochemistry (prone to false-positive/negative); "
        "Performed at specialized centers (Mayo, UK national amyloid centre)"
    ),
    "Tafamidis": (
        "Transthyretin tetramer stabilizer; FDA 2019; oral; "
        "Slows progression of ATTRv+ATTRwt cardiac amyloidosis; "
        "ATTR-ACT trial: 30% reduction in mortality+CV hospitalization; "
        "TTR stabilization prevents tetramer dissociation and fibrillogenesis"
    ),
    "Patisiran": (
        "RNAi targeting TTR mRNA in hepatocytes; FDA 2018; IV q3wk; "
        "TTR reduction >80%; ATTRv polyneuropathy; "
        "APOLLO trial: significant improvement in neuropathy score; "
        "Lipid nanoparticle delivery to liver"
    ),
    "Vutrisiran": (
        "RNAi targeting TTR mRNA; FDA 2022; SC q3months (less frequent than patisiran); "
        "HELIOS-A trial: non-inferior to patisiran for polyneuropathy; "
        "Also cardiac benefit in HELIOS-B trial; "
        "Subcutaneous quarterly injection"
    ),
    "Inotersen": (
        "ASO (antisense oligonucleotide) targeting TTR mRNA; FDA 2018; SC weekly; "
        "ATTRv polyneuropathy; NEURO-TTR trial; "
        "Thrombocytopenia monitoring REQUIRED (platelet-depleting ASO class effect); "
        "Monitor platelets every 2 weeks"
    ),
    "Eplontersen": (
        "ASO targeting TTR mRNA conjugated to GalNAc (liver targeting); FDA 2023; "
        "NEURO-TTRansform trial: superiority over inotersen for polyneuropathy; "
        "Also cardiac benefit; subcutaneous monthly; "
        "TTR reduction >90%"
    ),
    "Meretoja-syndrome": (
        "Finnish-type gelsolin amyloidosis (AGel); GSN D187N/D187Y mutation; "
        "MERETOJA TRIAD: lattice corneal dystrophy type II + bilateral facial nerve palsy + cutis laxa; "
        "All three together PATHOGNOMONIC; Finnish/Northern European enriched"
    ),
    "Lattice-corneal-dystrophy": (
        "TYPE I: TGFBI mutation, corneal only, no systemic involvement; "
        "TYPE II (Meretoja/AGel): GSN mutation, systemic with cranial neuropathy+skin laxity PATHOGNOMONIC; "
        "Distinguishing Type I vs Type II critical: only Type II requires systemic workup"
    ),
    "Dialysis-related-amyloidosis": (
        "AB2M amyloidosis from beta-2 microglobulin accumulation in dialysis patients; "
        "Bilateral carpal tunnel syndrome onset PATHOGNOMONIC >5-10yr dialysis; "
        "High-flux hemodiafiltration reduces B2M; renal transplant CURATIVE; "
        "Subchondral cysts on X-ray PATHOGNOMONIC destructive arthropathy"
    ),
    "Liver-transplant-amyloidosis": (
        "CURATIVE source-removal strategy for liver-produced amyloid proteins: "
        "TTR (liver transplant = V30M polyneuropathy curative if early); "
        "FGA (AFib amyloidosis: liver transplant = CURATIVE prevents renal progression); "
        "Timing: before end-stage organ damage for best outcome; "
        "Combined liver+kidney if both organs failing"
    ),
    "cascade-testing-amyloidosis": (
        "All 1st-degree relatives of index case; "
        "AD variants (TTR/APOA1/FGA/LYZ/GSN/APOA2/OSMR): 50% inheritance risk; "
        "Pre-symptomatic identification enables early therapy (TTR) or transplant planning; "
        "V122I African-American cardiac: screen all relatives before cardiac symptoms (irreversible damage)"
    ),
}


def _make_patient(gene_idx: int, seed: int) -> dict:
    """Generate one synthetic patient record for the given gene index."""
    rng = random.Random(seed)
    gene = ATLAS_GENES[gene_idx]["gene"]

    if gene == "TTR":
        age_dx = rng.randint(35, 70)
        variants = ["V30M", "V122I", "T60A", "I68L", "E89Q", "V30M"]
        variant = variants[rng.randint(0, len(variants) - 1)]
        cardiac_n = 1 if variant in ("V122I", "T60A", "I68L", "E89Q") else rng.randint(0, 1)
        neuro_n = 1 if variant == "V30M" else rng.randint(0, 1)
        treatment = rng.choice(["Tafamidis", "Patisiran", "Vutrisiran", "Inotersen", "Eplontersen", "Liver_Tx", "None"])
        return {
            "gene": gene, "seed": seed, "age_dx": age_dx, "variant": variant,
            "cardiac": cardiac_n, "neuropathy": neuro_n, "renal": rng.randint(0, 1),
            "treatment": treatment, "dpd_positive": cardiac_n,
            "apical_sparing": cardiac_n, "carpal_tunnel": rng.randint(0, 1),
        }
    elif gene == "APOA1":
        age_dx = rng.randint(30, 65)
        variants = ["Arg173Pro", "Leu75Pro", "Trp50Arg", "Gly26Arg"]
        return {
            "gene": gene, "seed": seed, "age_dx": age_dx,
            "variant": variants[rng.randint(0, len(variants) - 1)],
            "renal": 1, "hepatic": rng.randint(0, 1), "cardiac": rng.randint(0, 1),
            "neuropathy": rng.randint(0, 1), "hdl_low": 1, "treatment": "Organ_support",
        }
    elif gene == "FGA":
        age_dx = rng.randint(40, 70)
        variants = ["Glu526Val", "Arg554Leu", "Val526Glu"]
        return {
            "gene": gene, "seed": seed, "age_dx": age_dx,
            "variant": variants[rng.randint(0, len(variants) - 1)],
            "renal": 1, "neuropathy": 0, "cardiac": 0,
            "liver_tx": rng.randint(0, 1), "esrd": rng.randint(0, 1),
            "treatment": rng.choice(["Liver_Tx", "Renal_Replacement", "Observation"]),
        }
    elif gene == "LYZ":
        age_dx = rng.randint(35, 60)
        variants = ["D67H", "I56T"]
        return {
            "gene": gene, "seed": seed, "age_dx": age_dx,
            "variant": variants[rng.randint(0, 1)],
            "renal": rng.randint(0, 1), "hepatomegaly": 1, "splenomegaly": 1,
            "neuropathy": 0, "cardiac": 0, "treatment": "Organ_support",
        }
    elif gene == "GSN":
        age_dx = rng.randint(30, 55)
        variants = ["D187N", "D187Y"]
        return {
            "gene": gene, "seed": seed, "age_dx": age_dx,
            "variant": variants[rng.randint(0, 1)],
            "corneal_lattice_II": 1, "facial_nerve_palsy": rng.randint(0, 1),
            "cutis_laxa": rng.randint(0, 1), "neuropathy": rng.randint(0, 1),
            "renal": rng.randint(0, 1), "keratoplasty": rng.randint(0, 1),
            "treatment": "Keratoplasty_symptomatic",
        }
    elif gene == "APOA2":
        age_dx = rng.randint(50, 75)
        return {
            "gene": gene, "seed": seed, "age_dx": age_dx,
            "variant": "Ter94Stop", "renal": 1, "neuropathy": 0,
            "cardiac": 0, "esrd": rng.randint(0, 1), "treatment": "Organ_support",
        }
    elif gene == "B2M":
        age_dx = rng.randint(40, 70)
        dialysis_duration = rng.randint(5, 25)
        return {
            "gene": gene, "seed": seed, "age_dx": age_dx,
            "variant": rng.choice(["D76N_hereditary", "Dialysis_acquired"]),
            "carpal_tunnel": 1, "destructive_arthropathy": rng.randint(0, 1),
            "dialysis_yr": dialysis_duration, "b2m_level_ugml": rng.randint(40, 120),
            "renal_tx": rng.randint(0, 1), "treatment": "High_flux_HDF",
        }
    else:  # OSMR
        age_dx = rng.randint(20, 50)
        return {
            "gene": gene, "seed": seed, "age_dx": age_dx,
            "variant": rng.choice(["Glu356Lys", "Pro694Leu", "Thr914Ile"]),
            "pruritic_papules": 1, "interscapular": 1, "cardiac": 0, "renal": 0,
            "neuropathy": 0, "treatment": rng.choice(["Topical_CS", "NB_UVB", "Antihistamine"]),
        }


def _cohort(gene_idx: int, n: int = 40) -> list:
    rng = random.Random(SEED_BASE + gene_idx)
    seeds = [rng.randint(10000, 99999) for _ in range(n)]
    return [_make_patient(gene_idx, s) for s in seeds]


def generate_overview() -> dict:
    all_cohorts = [_cohort(i) for i in range(8)]
    gene_names = [g["gene"] for g in ATLAS_GENES]

    per_gene_summary = {}
    for i, gene in enumerate(gene_names):
        cohort = all_cohorts[i]
        recs = cohort
        mean_age = round(sum(r["age_dx"] for r in recs) / len(recs), 1)
        cardiac_n = sum(r.get("cardiac", 0) for r in recs)
        neuropathy_n = sum(r.get("neuropathy", 0) for r in recs)
        renal_n = sum(r.get("renal", 0) for r in recs)
        dpd_n = sum(r.get("dpd_positive", 0) for r in recs)
        per_gene_summary[gene] = {
            "n": len(recs),
            "mean_age_dx": mean_age,
            "cardiac_n": cardiac_n,
            "neuropathy_n": neuropathy_n,
            "renal_n": renal_n,
            "dpd_positive_n": dpd_n,
        }

    return {
        "atlas": "Hereditary-Transthyretin-Amyloidosis-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Systemic Amyloidosis Reference — "
            "TTR-ATTR-V30M-V122I-Tafamidis-Patisiran | APOA1-AApoAI-HDL-low | "
            "FGA-AFib-Renal-dominant-Liver-Tx-CURATIVE | LYZ-ALys-Hepatomegaly-Splenomegaly | "
            "GSN-AGel-MERETOJA-TRIAD | APOA2-AApoAII-Renal-Ter94 | "
            "B2M-AB2M-DRA-CTS-Dialysis | OSMR-PLCA-Pruritic-Cutaneous-NO-Systemic"
        ),
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": 320,
        "seeds": "3782-3789",
        "seed_base": SEED_BASE,
        "per_gene_summary": per_gene_summary,
        "atlas_genes": [
            {
                "gene": g["gene"],
                "locus": g["locus"],
                "syndrome": g["syndrome"],
                "inheritance": g["inheritance"],
                "key_therapy": g["key_therapy"],
                "pathognomonic": g["pathognomonic"],
            }
            for g in ATLAS_GENES
        ],
        "key_clinical_themes": {
            "FDA_approved_TTR_therapies": [
                "Tafamidis-cardiac-ATTRv+ATTRwt",
                "Patisiran-polyneuropathy",
                "Inotersen-polyneuropathy-platelet-monitor",
                "Vutrisiran-polyneuropathy+cardiac",
                "Eplontersen-polyneuropathy+cardiac",
            ],
            "liver_transplant_curative": ["TTR-V30M-polyneuropathy-early", "FGA-AFib-renal-amyloidosis"],
            "pathognomonic_findings": {
                "TTR-cardiac": "Apical-sparing-longitudinal-strain + 99mTc-DPD-Grade2-3",
                "APOA1": "HDL-C-near-zero + multi-organ",
                "LYZ": "Hepatomegaly+splenomegaly-combination",
                "GSN": "MERETOJA-TRIAD-corneal-lattice-II+facial-VII+cutis-laxa",
                "B2M": "Bilateral-CTS-dialysis->10yr",
                "OSMR": "Interscapular-pruritic-papules-NO-systemic",
            },
        },
    }


def generate_breakdown() -> dict:
    all_cohorts = [_cohort(i) for i in range(8)]
    gene_names = [g["gene"] for g in ATLAS_GENES]

    gene_data = {}
    for i, gene in enumerate(gene_names):
        cohort = all_cohorts[i]
        mean_age = round(sum(r["age_dx"] for r in cohort) / len(cohort), 1)
        cardiac_n = sum(r.get("cardiac", 0) for r in cohort)
        neuropathy_n = sum(r.get("neuropathy", 0) for r in cohort)
        renal_n = sum(r.get("renal", 0) for r in cohort)
        hepatomegaly_n = sum(r.get("hepatomegaly", 0) for r in cohort)
        splenomegaly_n = sum(r.get("splenomegaly", 0) for r in cohort)
        corneal_n = sum(r.get("corneal_lattice_II", 0) for r in cohort)
        carpal_tunnel_n = sum(r.get("carpal_tunnel", 0) for r in cohort)
        dpd_n = sum(r.get("dpd_positive", 0) for r in cohort)
        cutaneous_n = sum(r.get("pruritic_papules", 0) for r in cohort)
        gene_data[gene] = {
            "n": 40,
            "mean_age_dx": mean_age,
            "cardiac_n": cardiac_n,
            "neuropathy_n": neuropathy_n,
            "renal_n": renal_n,
            "hepatomegaly_n": hepatomegaly_n,
            "splenomegaly_n": splenomegaly_n,
            "corneal_lattice_n": corneal_n,
            "carpal_tunnel_n": carpal_tunnel_n,
            "dpd_positive_n": dpd_n,
            "cutaneous_n": cutaneous_n,
            "systemic_organs": gene != "OSMR",
        }

    return {
        "atlas": "Hereditary-Transthyretin-Amyloidosis-Atlas",
        "gene_data": gene_data,
        "atlas_genes": ATLAS_GENES,
        "amyloid_type_map": {
            "TTR": "ATTR",
            "APOA1": "AApoAI",
            "FGA": "AFib",
            "LYZ": "ALys",
            "GSN": "AGel",
            "APOA2": "AApoAII",
            "B2M": "AB2M",
            "OSMR": "PLCA-Amyloid-K",
        },
        "treatment_curative": {
            "TTR": "Liver-transplant (V30M polyneuropathy early) + Tafamidis/RNAi/ASO",
            "APOA1": "Liver+kidney-transplant (hepatic/renal dominant)",
            "FGA": "Liver-transplant CURATIVE (removes amyloidogenic fibrinogen source)",
            "LYZ": "No curative; organ support",
            "GSN": "No curative; keratoplasty for corneal",
            "APOA2": "No curative; renal replacement",
            "B2M": "Renal-transplant CURATIVE (dialysis-related); high-flux HDF",
            "OSMR": "No curative; narrowband UVB",
        },
        "no_systemic_rule": {
            "note": "OSMR-PLCA has NO systemic organ involvement -- critical distinction",
            "pathognomonic_absence": ["cardiac", "renal", "neuropathy"],
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Transthyretin-Amyloidosis-Atlas",
        "definitions": DEFINITIONS,
        "critical_clinical_rules": {
            "TTR-APICAL-SPARING-PATHOGNOMONIC": (
                "Cardiac ATTR: Apical sparing on longitudinal strain (GLS); "
                "GLS ratio (apex / basal+mid segments) > 1.0 = PATHOGNOMONIC; "
                "Amyloid deposits concentrate in base+mid wall, spare apex; "
                "Combined with 99mTc-DPD Grade 2-3 = non-biopsy cardiac ATTR diagnosis"
            ),
            "TTR-DPD-PYP-SCAN-PATHOGNOMONIC": (
                "99mTc-DPD/PYP bone scintigraphy Grade 2-3 (Perugini scale) = PATHOGNOMONIC cardiac ATTR; "
                "PREREQUISITE: exclude clonal plasma cell disorder (AL amyloid) first; "
                "SPEP + serum IFE + urine IFE + serum FLC ratio MANDATORY before DPD interpretation; "
                "Grade 2-3 DPD + no clonal disease = non-biopsy ATTR cardiac diagnosis (class IIa ACC/AHA)"
            ),
            "TTR-AVOID-DIGOXIN-VERAPAMIL": (
                "Cardiac ATTR: DIGOXIN AVOID -- binds amyloid fibrils -> unpredictable toxicity, "
                "heart block, VT at therapeutic levels; "
                "VERAPAMIL/DILTIAZEM AVOID -- calcium channel blockade + amyloid conduction disease = "
                "complete heart block risk; "
                "AMIODARONE AVOID -- QTc prolongation + baseline conduction disease; "
                "SPIRONOLACTONE AVOID -- hypotension + renal failure in low-output amyloid heart"
            ),
            "FGA-LIVER-TRANSPLANT-CURATIVE": (
                "Fibrinogen alpha chain (FGA) amyloidosis = AFib; "
                "Liver transplant CURATIVE: liver produces 100% of fibrinogen -> transplant removes source; "
                "Prevents further renal amyloid deposition; renal function stabilizes; "
                "TIMING CRITICAL: transplant BEFORE ESRD for best renal outcome; "
                "Combined liver+kidney if ESRD already present; "
                "NO peripheral neuropathy = key distinguisher from TTR/APOA1"
            ),
            "GSN-MERETOJA-TRIAD-PATHOGNOMONIC": (
                "Gelsolin amyloidosis (AGel) = Meretoja syndrome; "
                "MERETOJA TRIAD ALL THREE REQUIRED FOR PATHOGNOMONIC: "
                "1. Lattice corneal dystrophy TYPE II (NOT Type I which is TGFBI, corneal only); "
                "2. Bilateral facial nerve (VII) palsy progressive; "
                "3. Cutis laxa (skin laxity, pendulous folds); "
                "Finnish/Northern European enriched; D187N founder; "
                "Keratoplasty for corneal opacity; no systemic disease-modifying therapy"
            ),
            "B2M-CTS-DIALYSIS-PATHOGNOMONIC": (
                "Beta-2 microglobulin amyloidosis (AB2M): "
                "Bilateral CTS (carpal tunnel syndrome) onset PATHOGNOMONIC in dialysis patient >5-10yr; "
                "Subchondral cysts on X-ray PATHOGNOMONIC destructive arthropathy in dialysis patients; "
                "Shoulder (Milwaukee shoulder) also common; "
                "HIGH-FLUX HEMODIAFILTRATION (online HDF): best B2M removal, slows DRA; "
                "RENAL TRANSPLANT: CURATIVE for dialysis-related form (restores renal B2M clearance)"
            ),
            "OSMR-PLCA-NO-SYSTEMIC-ORGANS": (
                "OSMR-PLCA (primary localized cutaneous amyloidosis): "
                "Intensely pruritic papular deposits INTERSCAPULAR location PATHOGNOMONIC; "
                "NO cardiac amyloidosis; NO renal amyloidosis; NO neuropathy -- ABSOLUTE RULE; "
                "Amyloid K (keratin intermediate filament origin) -- NOT serum-derived; "
                "This distinguishes PLCA from TTR/APOA1/FGA/LYZ/GSN/APOA2/B2M; "
                "Prognosis excellent for survival; narrowband UVB for pruritus"
            ),
            "EXCLUDE-AL-MANDATORY-ALL-AMYLOIDOSIS": (
                "For ANY systemic amyloidosis workup -- EXCLUDE AL (light chain) amyloid FIRST: "
                "1. Serum protein electrophoresis (SPEP) + immunofixation (IFE); "
                "2. Urine protein electrophoresis (UPEP) + IFE; "
                "3. Serum free light chain (FLC) ratio; "
                "4. Bone marrow biopsy if any clonal plasma cell signal; "
                "AL amyloidosis treated with chemotherapy (daratumumab+CyBorD); "
                "Hereditary amyloidosis treated very differently; "
                "NEVER proceed to liver transplant or targeted therapy without excluding AL"
            ),
            "MASS-SPECTROMETRY-FIBRIL-TYPING-MANDATORY": (
                "Congo-red positive biopsy requires MASS SPECTROMETRY fibril typing; "
                "Identifies specific amyloid protein (TTR/APOA1/FGA/LYZ/GSN/APOA2/B2M/light chain); "
                "Immunohistochemistry (IHC) panels: prone to false-positive and false-negative; "
                "MS fibril typing GOLD STANDARD for amyloid typing; "
                "Send to specialized amyloid centers (Mayo Clinic Amyloid Program, UK National)"
            ),
            "CASCADE-TESTING-HEREDITARY-AMYLOIDOSIS": (
                "All 1st-degree relatives of hereditary amyloidosis index case; "
                "AD variants (TTR/APOA1/FGA/LYZ/GSN/APOA2/OSMR): 50% inheritance risk; "
                "V122I African-American: screen ALL relatives -- 3-4% of African Americans carriers; "
                "Pre-symptomatic TTR detection: enables early tafamidis/RNAi/ASO therapy; "
                "FGA: early transplant before ESRD = best outcome; "
                "GSN: ophthalmology screening + corneal exam annually"
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== Hereditary Transthyretin Amyloidosis Atlas ===")
    ov = generate_overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print()
    br = generate_breakdown()
    for gene, d in br["gene_data"].items():
        print(
            f"  {gene}: n={d['n']}, age_dx={d['mean_age_dx']}yr, "
            f"cardiac={d['cardiac_n']}/40, neuro={d['neuropathy_n']}/40, "
            f"renal={d['renal_n']}/40"
        )
    print()
    df = generate_definitions()
    print(f"Definitions: {len(df['definitions'])} terms")
    print(f"Clinical rules: {len(df['critical_clinical_rules'])} rules")
    print("SCRIPT OK")
