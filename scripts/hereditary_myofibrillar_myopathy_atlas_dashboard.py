#!/usr/bin/env python3
"""Hereditary-Myofibrillar-Myopathy-Atlas — Complete 8-Gene Myofibrillar Myopathy (MFM) Spectrum Atlas
(DES · CRYAB · MYOT · FLNC · BAG3 · PYROXD1 · ACTN2 · HSPB8).

DES     (Desmin; 470 aa; 2q35; AD/AR;
         MFM1 — most common MFM gene; cytoplasmic desmin aggregates pathognomonic;
         cardiac conduction disease + DCM mandatory; scapuloperoneal + distal weakness;
         CK 2-10× ULN; onset 20-40yr;
         seed SEED_BASE+0).
CRYAB   (αB-crystallin / HSPB5; 175 aa; 11q23.1; AD/AR;
         MFM2 — POSTERIOR SUBCAPSULAR CATARACTS IN AD CARRIERS PATHOGNOMONIC;
         desmin-related myopathy; cardiac DCM+HCM; CK 1.5-4× ULN; onset 35-55yr;
         seed SEED_BASE+1).
MYOT    (Myotilin; 498 aa; 5q31.2; AD;
         MFM3 / LGMD1A allelic — LATE ONSET 40-65yr PATHOGNOMONIC;
         Z-disc protein; scapuloperoneal; filamentous cytoplasmic inclusions;
         CK 1.5-4× ULN;
         seed SEED_BASE+2).
FLNC    (Filamin C; 2725 aa; 7q32.1; AD;
         MFM5 — most frequently identified MFM gene now;
         CARDIAC DCM+HCM+ARVC MANDATORY — truncating variants → DCM/ARVC;
         hyaline bodies biopsy; CK 2-15× ULN; onset 30-55yr;
         seed SEED_BASE+3).
BAG3    (BCL2-associated athanogene 3; 575 aa; 10q26.11; AD;
         MFM6 — CHILDHOOD ONSET PATHOGNOMONIC (age 2-15yr);
         most severe MFM; DCM mandatory childhood; axial hypotonia; respiratory failure early;
         P209L founder; CK 2-20× ULN;
         seed SEED_BASE+4).
PYROXD1 (Pyridine nucleotide-disulphide oxidoreductase domain 1; 500 aa; 12p12.1; AR;
         Childhood MFM — myofibrillar disruption + nemaline-like rods;
         CK 2-20× ULN; onset childhood to early adult; facial weakness mild;
         AR: consanguinity common;
         seed SEED_BASE+5).
ACTN2   (Alpha-actinin-2; 894 aa; 1q43; AD;
         Sarcomeric Z-disc protein; HCM+DCM+LVNC — cardiac phenotype dominant;
         skeletal myopathy mild / myofibrillar disruption pattern; onset 20-50yr;
         CK 1.5-5× ULN;
         seed SEED_BASE+6).
HSPB8   (Heat shock protein B8 / HSP22; 196 aa; 12q24.23; AD;
         MFM + CMT2L overlap — distal muscle weakness + axonal neuropathy;
         K141N/K141E hotspot pathognomonic; onset 20-45yr; CK 1.5-6× ULN;
         rimmed vacuoles biopsy;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2206-2213).
"""

import random

SEED_BASE = 2206

MFM_GENES = [
    # -- DES — Desmin, MFM1 ----------------------------------------------------------------
    {
        "gene": "DES",
        "alt_name": (
            "DES (DES-470aa-2q35 / AD/AR — MFM1-Desminopathy-Most-Common-MFM-Gene — "
            "CYTOPLASMIC-DESMIN-AGGREGATES-PATHOGNOMONIC-Biopsy — "
            "CARDIAC-CONDUCTION-DISEASE-DCM-MANDATORY — "
            "Scapuloperoneal-Distal-Weakness-CK-2-10x-ULN)"
        ),
        "protein": (
            "DES -- 2q35 AD/AR -- DES-470aa -- "
            "Desmin-Type-III-Intermediate-Filament-53kDa-Z-Disc-Scaffold -- "
            "MFM1-Desminopathy-OMIM-601419 -- "
            "CYTOPLASMIC-DESMIN-AGGREGATES-Granulofilamentous-Material-TEM-PATHOGNOMONIC -- "
            "CARDIAC-CONDUCTION-DISEASE-DCM-HCM-Mandatory-Surveillance -- "
            "Scapuloperoneal-Weakness-Scapular-Winging-Foot-Drop -- "
            "CK-2-10x-ULN-Mild-Moderate -- "
            "Onset-20-40yr-AD-Earlier-AR-Later -- "
            "Rimmed-Vacuoles-Absent-DDx-GNE-VCP -- "
            "DES-IHC-Cytoplasmic-Deposits-Desmin-Antibody-Muscle-Biopsy -- "
            "AR-Biallelic-Truncating-More-Severe-Childhood-DCM -- "
            "OMIM-Gene-DES-125660-Disease-MFM1-601419"
        ),
        "locus": "2q35",
        "protein_size": "470 aa / 53 kDa",
        "inheritance": (
            "AD (most MFM1 — missense Z-disc/tail domain); "
            "AR (biallelic truncating — more severe, childhood-onset DCM); "
            "Penetrance AD: ~90% by age 50; "
            "Onset 20-40yr (AD missense); childhood possible (AR); "
            "Cardiac involvement: 30-50% of AD cases; higher in AR; "
            "Scapuloperoneal distribution classic; foot drop early; "
            "CK 2-10× ULN; desmin aggregates on biopsy (desmin IHC deposits)"
        ),
        "key_features": [
            "CYTOPLASMIC DESMIN AGGREGATES — granulofilamentous material on TEM, desmin IHC deposits PATHOGNOMONIC",
            "Cardiac conduction disease + DCM — 30-50% AD; mandatory surveillance",
            "Scapuloperoneal weakness + scapular winging — classic distribution",
            "Foot drop early — peroneal compartment",
            "CK 2-10× ULN — mildly to moderately elevated",
            "Rimmed vacuoles ABSENT — key DDx from GNE/VCP IBM",
            "AR biallelic: severe childhood DCM before skeletal disease",
            "Respiratory involvement late in severe cases",
        ],
        "treatment": (
            "Cardiac: annual echo + Holter; ICD for SCD prevention if significant DCM/arrhythmia; "
            "pacemaker for AV block; "
            "Skeletal: physiotherapy, AFOs for foot drop; "
            "No disease-modifying therapy; "
            "Genetic counselling family cascade; "
            "Respiratory: annual spirometry in severe cases"
        ),
        "contraindications": [
            "DO NOT reassure on cardiac risk — conduction disease may predate DCM; annual Holter mandatory",
            "Do not confuse desmin IHC deposits with other MFM proteins — confirm with specific antibody panel",
        ],
        "critical_pearls": [
            "DES: desmin IHC cytoplasmic deposits (not absent) — distinguish from EMD emerin-absent; different meaning",
            "Scapular winging + foot drop + cardiac = DES first-line; add CRYAB MYOT FLNC panel",
            "AR biallelic DES: severe childhood DCM — more severe than AD; neonatal presentation possible",
            "Granulofilamentous material on TEM: semi-membranous dense bodies = desminopathy hallmark",
            "CK useful but non-specific: 2-10× — does not distinguish MFM subtypes",
        ],
        "mean_age_onset": 30,
        "mean_age_dx": 38,
        "sex_ratio_m_f": "1:1",
        "ck_range": (200, 1000),
        "ambulant_at_10yr_pct": 0.90,
        "alive_pct": 0.88,
        "treated_pct": 0.70,
    },
    # -- CRYAB — αB-crystallin, MFM2 -------------------------------------------------------
    {
        "gene": "CRYAB",
        "alt_name": (
            "CRYAB (CRYAB-175aa-11q23.1 / AD/AR — MFM2-Desmin-Related-Myopathy2-HSPB5 — "
            "POSTERIOR-SUBCAPSULAR-CATARACTS-PATHOGNOMONIC-AD-Carriers — "
            "Cardiac-DCM-HCM-Small-Heat-Shock-Protein)"
        ),
        "protein": (
            "CRYAB -- 11q23.1 AD/AR -- CRYAB-175aa -- "
            "AlphaB-Crystallin-HSPB5-Small-Heat-Shock-Protein-20kDa -- "
            "MFM2-Desmin-Related-Myopathy-2-OMIM-608810 -- "
            "POSTERIOR-SUBCAPSULAR-CATARACTS-50pct-AD-Carriers-PATHOGNOMONIC -- "
            "Cardiac-DCM-HCM-Mandatory-Surveillance -- "
            "Myofibrillar-Disruption-Spheroid-Inclusions-Hyaline-Plaque -- "
            "CK-1.5-4x-ULN-Mildly-Elevated -- "
            "Onset-35-55yr-AD-Missense-Alpha-Crystallin-Domain -- "
            "R120G-Hotspot-Most-Common-Pathogenic-Variant-Worldwide -- "
            "R57W-Biallelic-AR-Severe-Childhood-MFM -- "
            "OMIM-Gene-CRYAB-123590-Disease-MFM2-608810"
        ),
        "locus": "11q23.1",
        "protein_size": "175 aa / 20 kDa",
        "inheritance": (
            "AD (most MFM2 — R120G hotspot most common worldwide; α-crystallin domain missense); "
            "AR (biallelic — severe childhood-onset; R57W and others); "
            "Penetrance AD: variable 50-90%; "
            "Onset 35-55yr (AD); childhood (AR); "
            "Cataracts: 50% of AD mutation carriers — posterior subcapsular PATHOGNOMONIC; "
            "Cardiac DCM+HCM in 30-40%; "
            "CK 1.5-4× ULN"
        ),
        "key_features": [
            "POSTERIOR SUBCAPSULAR CATARACTS — 50% of AD carriers, PATHOGNOMONIC DDx clue",
            "Cardiac DCM + HCM — 30-40% of cases; mandatory surveillance",
            "Myofibrillar disruption with spheroid/hyaline inclusions on biopsy",
            "R120G — most common pathogenic variant worldwide (hotspot)",
            "CK 1.5-4× ULN — mild elevation",
            "Onset 35-55yr (AD missense); childhood in AR biallelic forms",
            "Proximal + scapuloperoneal weakness pattern",
            "Respiratory involvement in severe cases",
        ],
        "treatment": (
            "Cardiac: annual echo + Holter; ICD if DCM with arrhythmia; "
            "Ophthalmology: annual slit-lamp for cataracts; surgical extraction when symptomatic; "
            "Skeletal: physiotherapy, AFOs; "
            "Respiratory: annual spirometry; "
            "Genetic counselling — cataracts as presymptomatic marker in families"
        ),
        "contraindications": [
            "DO NOT delay cataract referral — posterior subcapsular cataracts are the diagnostic clue; ophthalmology mandatory",
            "Do not exclude cardiac involvement — DCM+HCM both occur; echo + Holter annual",
        ],
        "critical_pearls": [
            "CRYAB cataracts: posterior subcapsular — not nuclear or cortical; ask specifically; may predate myopathy",
            "R120G hotspot: most common CRYAB variant globally; one of the earliest MFM genes discovered",
            "AR biallelic CRYAB: severe childhood MFM + earlier cardiac; worse than AD",
            "Cataracts + cardiac + myopathy triad → CRYAB first (before DES/MYOT/FLNC)",
            "CRYAB IHC: αB-crystallin cytoplasmic deposits in inclusions — same technique as desmin",
        ],
        "mean_age_onset": 43,
        "mean_age_dx": 50,
        "sex_ratio_m_f": "1:1",
        "ck_range": (150, 400),
        "ambulant_at_10yr_pct": 0.95,
        "alive_pct": 0.92,
        "treated_pct": 0.65,
    },
    # -- MYOT — Myotilin, MFM3 / LGMD1A ---------------------------------------------------
    {
        "gene": "MYOT",
        "alt_name": (
            "MYOT (MYOT-498aa-5q31.2 / AD — MFM3-LGMD1A-Myotilinopathy — "
            "LATE-ONSET-40-65yr-PATHOGNOMONIC-DDx-from-other-MFM — "
            "Z-Disc-Protein-Filamentous-Cytoplasmic-Inclusions)"
        ),
        "protein": (
            "MYOT -- 5q31.2 AD -- MYOT-498aa -- "
            "Myotilin-57kDa-Z-Disc-Protein-Filamin-Interaction -- "
            "MFM3-OMIM-609200 -- "
            "LGMD1A-OMIM-159000-Allelic-Same-Gene -- "
            "LATE-ONSET-40-65yr-PATHOGNOMONIC-Distinguishes-from-BAG3-PYROXD1-DES -- "
            "Scapuloperoneal-Weakness-Facial-Weakness-Occasional -- "
            "Filamentous-Cytoplasmic-Inclusions-MYOT-IHC-Deposits -- "
            "CK-1.5-4x-ULN-Mildly-Elevated -- "
            "T57I-S55F-Cluster-Hotspot-Ig-Like-Domain -- "
            "Dysarthria-Dysphagia-Later-Bulbar -- "
            "OMIM-Gene-MYOT-604103-Disease-MFM3-609200"
        ),
        "locus": "5q31.2",
        "protein_size": "498 aa / 57 kDa",
        "inheritance": (
            "AD (all pathogenic MYOT variants; LGMD1A and MFM3 same gene); "
            "Penetrance: high >90% by age 70; "
            "LATE onset 40-65yr — distinguishes from BAG3 (childhood) and PYROXD1 (childhood/young adult); "
            "T57I and S55F hotspots (Ig-like domain cluster); "
            "CK 1.5-4× ULN; "
            "Scapuloperoneal + limb-girdle weakness; "
            "Bulbar: dysarthria + dysphagia in 20-40% (late)"
        ),
        "key_features": [
            "LATE ONSET 40-65yr — most specific MYOT feature PATHOGNOMONIC vs other MFM genes",
            "MYOT IHC cytoplasmic deposits — filamentous inclusions on biopsy",
            "Scapuloperoneal weakness + limb-girdle distribution (LGMD1A allelic)",
            "Dysarthria + dysphagia late (20-40%) — bulbar involvement",
            "CK 1.5-4× ULN — mild elevation; can be near-normal",
            "No cardiac involvement — distinguishes from DES/CRYAB/FLNC/BAG3",
            "T57I/S55F hotspot variants — concentrated in Ig-like domain",
            "Facial weakness occasional — mild",
        ],
        "treatment": (
            "No cardiac surveillance required (MYOT has NO cardiac involvement — DDx from DES/CRYAB/FLNC); "
            "Skeletal: physiotherapy, AFOs; speech/language therapy for dysarthria; "
            "PEG tube if dysphagia severe; "
            "Respiratory: annual spirometry in advanced cases; "
            "No disease-modifying therapy; "
            "Genetic counselling — late onset so children often unaffected at time of diagnosis"
        ),
        "contraindications": [
            "NO cardiac surveillance needed — MYOT does NOT cause cardiomyopathy; distinguish from DES/FLNC/BAG3 where cardiac is mandatory",
            "Do not attribute young-onset myopathy to MYOT — late onset 40-65yr; earlier → other genes",
        ],
        "critical_pearls": [
            "MYOT: no cardiac — if cardiac present, reconsider DES/FLNC/BAG3; cardiac absence a key DDx clue",
            "Late onset 40-65yr: if MFM phenotype + onset <35yr → DES/FLNC/BAG3 more likely",
            "LGMD1A (MYOT) vs LGMD2 (AR): MYOT is AD limb-girdle MD; one of few dominant LGMDs",
            "Dysarthria in MFM: MYOT + DES; rare in FLNC; absent in BAG3/PYROXD1",
            "Filamentous inclusions + myotilin IHC = MFM3 biopsy signature",
        ],
        "mean_age_onset": 52,
        "mean_age_dx": 58,
        "sex_ratio_m_f": "1:1",
        "ck_range": (150, 400),
        "ambulant_at_10yr_pct": 0.95,
        "alive_pct": 0.93,
        "treated_pct": 0.55,
    },
    # -- FLNC — Filamin C, MFM5 ------------------------------------------------------------
    {
        "gene": "FLNC",
        "alt_name": (
            "FLNC (FLNC-2725aa-7q32.1 / AD — MFM5-Filaminopathy-Most-Identified-MFM-Gene — "
            "CARDIAC-DCM-HCM-ARVC-MANDATORY-Truncating-vs-Missense-Allele-Specific — "
            "Hyaline-Bodies-Biopsy-PATHOGNOMONIC)"
        ),
        "protein": (
            "FLNC -- 7q32.1 AD -- FLNC-2725aa -- "
            "Filamin-C-291kDa-Actin-Cross-Linking-Sarcomere-Z-Disc -- "
            "MFM5-OMIM-609524 -- "
            "CARDIAC-DCM-ARVC-Truncating-Variants-MANDATORY-ECHO-HOLTER -- "
            "HCM-Missense-Variants-Separate-Phenotype -- "
            "Hyaline-Bodies-Biopsy-PATHOGNOMONIC-MFM5 -- "
            "Most-Commonly-Identified-MFM-Gene-in-Cohort-Studies-2015-onwards -- "
            "Truncating-FLNC-DCM-ARVC-Cardiomyopathy-Dominant-Over-Skeletal -- "
            "Missense-FLNC-MFM5-Skeletal-Dominant-With-Some-Cardiac -- "
            "CK-2-15x-ULN-Moderate-Elevation -- "
            "Onset-30-55yr-Scapuloperoneal-Distal -- "
            "OMIM-Gene-FLNC-102565-Disease-MFM5-609524"
        ),
        "locus": "7q32.1",
        "protein_size": "2725 aa / 291 kDa",
        "inheritance": (
            "AD (dominant negative missense → MFM5; truncating → DCM/ARVC); "
            "ALLELE-SPECIFIC PHENOTYPE: truncating FLNC (frameshift/nonsense/splice) → DCM or ARVC (cardiac dominant); "
            "missense FLNC → MFM5 skeletal myopathy ± cardiac; "
            "Penetrance: high >85%; "
            "Onset 30-55yr (missense MFM5); earlier cardiac (truncating DCM/ARVC); "
            "CK 2-15× ULN; "
            "CARDIAC MANDATORY in all FLNC pathogenic variant carriers — regardless of variant type"
        ),
        "key_features": [
            "HYALINE BODIES biopsy — eosinophilic inclusions in type 1 fibres PATHOGNOMONIC MFM5",
            "CARDIAC MANDATORY — truncating: DCM/ARVC; missense: HCM ± skeletal; ECHO+HOLTER all carriers",
            "ALLELE-SPECIFIC: truncating → cardiac dominant; missense → skeletal MFM dominant",
            "Most commonly identified MFM gene in modern cohorts (since 2015)",
            "ARVC: RV fibrofatty replacement; ECG epsilon wave + RV dysfunction",
            "Scapuloperoneal + distal weakness distribution",
            "CK 2-15× ULN — higher than MYOT/CRYAB",
            "DCM + ARVC allelic in truncating FLNC — biventricular involvement possible",
        ],
        "treatment": (
            "Cardiac MANDATORY all FLNC carriers: echo + Holter 6-monthly; "
            "Truncating FLNC: ICD for DCM/ARVC SCD prevention — low threshold; "
            "Epsilon wave + RV dysfunction: arrhythmia specialist + cardiac MRI; "
            "HCM missense: beta-blocker/CCB as per HCM guidelines; "
            "Skeletal: physiotherapy, AFOs; "
            "Avoid high-intensity exercise in ARVC phenotype"
        ),
        "contraindications": [
            "ALL FLNC carriers need cardiac surveillance — do NOT omit based on variant type before formal assessment",
            "Truncating FLNC: avoid high-intensity contact sports — ARVC risk; arrhythmic SCD in athletes documented",
        ],
        "critical_pearls": [
            "Truncating FLNC = DCM/ARVC panel gene — cardiac dominant; skeletal myopathy mild or absent",
            "FLNC is now the most commonly found MFM gene in undiagnosed myopathy cohorts — sequence early",
            "Hyaline bodies: eosinophilic inclusions in type 1 fibres — FLNC IHC deposits confirm MFM5",
            "ARVC + FLNC: epsilon wave on ECG + RV dysfunction + fibrofatty MRI — order ARVC panel",
            "FLNC missense vs truncating: the single most important genotype-phenotype rule in MFM5",
        ],
        "mean_age_onset": 42,
        "mean_age_dx": 50,
        "sex_ratio_m_f": "1:1",
        "ck_range": (200, 1500),
        "ambulant_at_10yr_pct": 0.88,
        "alive_pct": 0.85,
        "treated_pct": 0.78,
    },
    # -- BAG3 — BCL2-associated athanogene 3, MFM6 -----------------------------------------
    {
        "gene": "BAG3",
        "alt_name": (
            "BAG3 (BAG3-575aa-10q26.11 / AD — MFM6-BAG3opathy-Most-Severe-MFM — "
            "CHILDHOOD-ONSET-2-15yr-PATHOGNOMONIC-Distinguishes-all-other-MFM — "
            "DCM-MANDATORY-Childhood-Axial-Hypotonia-Respiratory-Failure-Early)"
        ),
        "protein": (
            "BAG3 -- 10q26.11 AD -- BAG3-575aa -- "
            "BCL2-Associated-Athanogene-3-62kDa-BAG-Domain-WW-Domain-PxxP -- "
            "MFM6-OMIM-612954 -- "
            "CHILDHOOD-ONSET-2-15yr-Most-Severe-MFM-PATHOGNOMONIC-Age-of-Onset -- "
            "DCM-MANDATORY-Childhood-Dilated-Cardiomyopathy-Major-DCM-Gene -- "
            "Axial-Hypotonia-Neonatal-Early-Childhood-Truncal-Weakness -- "
            "Respiratory-Failure-Early-NIV-Often-Required-Childhood -- "
            "P209L-P209S-Founder-Variants-Most-Common-BAG3 -- "
            "Neuropathy-Peripheral-Frequent-CMT-Overlap -- "
            "de-Novo-Dominant-Negative-Mechanism -- "
            "CK-2-20x-ULN-Can-Be-Very-High -- "
            "OMIM-Gene-BAG3-603883-Disease-MFM6-612954"
        ),
        "locus": "10q26.11",
        "protein_size": "575 aa / 62 kDa",
        "inheritance": (
            "AD (most de novo; dominant negative mechanism; P209L/P209S hotspot); "
            "CHILDHOOD ONSET 2-15yr — most severe MFM gene; "
            "Axial hypotonia from infancy/early childhood — truncal weakness early; "
            "DCM mandatory — frequently diagnosed childhood; major DCM gene overall; "
            "Peripheral neuropathy: axonal CMT-like in many (up to 60%); "
            "Respiratory failure early: NIV often required by adolescence; "
            "CK 2-20× ULN — can be very high; "
            "Penetrance near 100% (de novo or AD); "
            "P209L/P209S WW-domain hotspot variants most common"
        ),
        "key_features": [
            "CHILDHOOD ONSET 2-15yr — most distinguishing BAG3 feature PATHOGNOMONIC vs all other MFM genes",
            "CARDIAC DCM MANDATORY — childhood; major DCM gene panel inclusion; ICD early",
            "Axial hypotonia — truncal weakness from infancy/early childhood",
            "Respiratory failure early — NIV often required by adolescence",
            "Peripheral neuropathy — axonal CMT-like in up to 60% of cases",
            "P209L/P209S WW-domain hotspot — most common pathogenic variants",
            "De novo AD — parents often unaffected; de novo rate high",
            "CK 2-20× ULN — variable; can be very elevated",
        ],
        "treatment": (
            "Cardiac MANDATORY: echo + Holter 6-monthly; ICD for DCM SCD prevention (childhood threshold); "
            "Respiratory: PFTs annually from diagnosis; NIV when FVC <50% (often adolescence); "
            "Axial: wheelchair adaptation early; truncal support; "
            "Neuropathy: NCS/EMG at diagnosis; "
            "Gene therapy trials emerging (dominant-negative target); "
            "Multidisciplinary NMD clinic: cardiology + respiratory + neurology + physio"
        ),
        "contraindications": [
            "DO NOT DELAY NIV — respiratory failure can be rapid in BAG3; spirometry every 6 months from diagnosis",
            "DO NOT DEFER DCM evaluation — childhood DCM; BAG3 is a mandatory DCM panel gene",
        ],
        "critical_pearls": [
            "BAG3: childhood-onset + DCM + axial hypotonia + neuropathy = pathognomonic constellation; order immediately",
            "P209L dominant negative: one allele sufficient; de novo; parents must be checked but often unaffected",
            "BAG3 DCM: major gene — included in all cardiomyopathy gene panels; NMD + cardiology co-management",
            "Respiratory often precedes severe limb weakness in BAG3 — screen every 6 months",
            "Neuropathy in BAG3: axonal; NCS helpful to differentiate from HSPB8 (also neuropathy but distal onset)",
        ],
        "mean_age_onset": 7,
        "mean_age_dx": 10,
        "sex_ratio_m_f": "1:1",
        "ck_range": (200, 2000),
        "ambulant_at_10yr_pct": 0.55,
        "alive_pct": 0.82,
        "treated_pct": 0.92,
    },
    # -- PYROXD1 — Pyridine nucleotide-disulphide oxidoreductase domain 1 ------------------
    {
        "gene": "PYROXD1",
        "alt_name": (
            "PYROXD1 (PYROXD1-500aa-12p12.1 / AR — Childhood-MFM-Myofibrillar-Disruption — "
            "Nemaline-Like-Rods-Biopsy-MFM-Pattern — "
            "AR-Consanguinity-Common-CK-2-20x)"
        ),
        "protein": (
            "PYROXD1 -- 12p12.1 AR -- PYROXD1-500aa -- "
            "Pyridine-Nucleotide-Disulphide-Oxidoreductase-Domain-1-56kDa -- "
            "AR-Childhood-Myofibrillar-Myopathy-OMIM-617063 -- "
            "Myofibrillar-Disruption-Nemaline-Like-Rods-Core-Like-Areas-Biopsy -- "
            "Childhood-to-Early-Adult-Onset-AR-Consanguinity-Common -- "
            "Facial-Weakness-Mild-Ptosis-Ophthalmoplegia-Some-Cases -- "
            "CK-2-20x-ULN-Variable-Elevation -- "
            "N155S-Q372H-Founder-Variants-Australian-Middle-Eastern -- "
            "Respiratory-Involvement-Some-Cases -- "
            "Nuclear-Redox-Homeostasis-Mechanism -- "
            "OMIM-Gene-PYROXD1-617063"
        ),
        "locus": "12p12.1",
        "protein_size": "500 aa / 56 kDa",
        "inheritance": (
            "AR (biallelic pathogenic variants; consanguinity common); "
            "Onset childhood to early adult (age 2-30yr); "
            "Myofibrillar disruption + nemaline-like rods + core-like areas on biopsy (hybrid pattern); "
            "N155S + Q372H founder variants (Australian and Middle Eastern populations); "
            "Mild facial weakness + ptosis ± ophthalmoplegia (unlike most other MFM); "
            "CK 2-20× ULN variable; "
            "Respiratory involvement in some cases; "
            "Mechanism: nuclear redox homeostasis disruption → myofibrillar protein aggregation"
        ),
        "key_features": [
            "MYOFIBRILLAR DISRUPTION + NEMALINE-LIKE RODS — hybrid biopsy pattern (MFM + nemaline features)",
            "AR inheritance — consanguinity common; sibling risk 25%",
            "Childhood to early adult onset (age 2-30yr)",
            "Facial weakness + ptosis ± ophthalmoplegia — distinguishes from DES/MYOT/FLNC",
            "N155S + Q372H founder variants — Australian and Middle Eastern cohorts",
            "CK 2-20× ULN — variable elevation",
            "Respiratory involvement in severe cases",
            "No significant cardiac involvement — distinguishes from DES/CRYAB/FLNC/BAG3",
        ],
        "treatment": (
            "No disease-modifying therapy; "
            "Skeletal: physiotherapy, AFOs; "
            "Respiratory: annual spirometry; NIV if FVC declines; "
            "Ophthalmology: ptosis correction if functional impairment; "
            "Genetic counselling: AR — sibling risk 25%; preimplantation genetic diagnosis available; "
            "Multidisciplinary NMD clinic"
        ),
        "contraindications": [
            "NO cardiac surveillance required for PYROXD1 — no cardiomyopathy; distinguish from FLNC/BAG3/DES where cardiac is mandatory",
            "Do not label as nemaline myopathy alone — myofibrillar features coexist; hybrid biopsy pattern → PYROXD1 panel",
        ],
        "critical_pearls": [
            "PYROXD1: hybrid biopsy (nemaline rods + myofibrillar disruption) = unique — not pure nemaline (NEB/ACTA1) nor pure MFM",
            "Ptosis + facial weakness in AR myopathy with MFM biopsy → PYROXD1 before congenital myasthenia",
            "N155S founder: Australian families; Q372H: Middle Eastern — ask ancestry before full sequencing",
            "No cardiac: if cardiac present in AR MFM → DES biallelic or other gene; PYROXD1 not cardiac",
            "Respiratory often early in severe cases: NCS for neuropathy (absent in PYROXD1 unlike HSPB8/BAG3)",
        ],
        "mean_age_onset": 12,
        "mean_age_dx": 18,
        "sex_ratio_m_f": "1:1",
        "ck_range": (200, 2000),
        "ambulant_at_10yr_pct": 0.75,
        "alive_pct": 0.91,
        "treated_pct": 0.60,
    },
    # -- ACTN2 — Alpha-actinin-2 -----------------------------------------------------------
    {
        "gene": "ACTN2",
        "alt_name": (
            "ACTN2 (ACTN2-894aa-1q43 / AD — Sarcomeric-Z-Disc-HCM-DCM-LVNC — "
            "CARDIAC-PHENOTYPE-DOMINANT-Over-Skeletal-Myopathy — "
            "Myofibrillar-Disruption-Z-Disc-Streaming-Biopsy)"
        ),
        "protein": (
            "ACTN2 -- 1q43 AD -- ACTN2-894aa -- "
            "Alpha-Actinin-2-104kDa-Sarcomeric-Z-Disc-Actin-Cross-Linking -- "
            "HCM-DCM-LVNC-Cardiomyopathy-Dominant-Phenotype-OMIM-102573 -- "
            "Myofibrillar-Disruption-Z-Disc-Streaming-Biopsy-MFM-Pattern -- "
            "Cardiac-Phenotype-Dominant-Skeletal-Myopathy-Mild-Subclinical -- "
            "CK-1.5-5x-ULN-Mildly-Elevated -- "
            "Onset-20-50yr-Cardiac-First-Skeletal-Later -- "
            "HCM-Most-Common-ACTN2-Phenotype -- "
            "LVNC-Noncompaction-Cardiomyopathy-Some-Cases -- "
            "OMIM-Gene-ACTN2-102573-Disease-HCM-OMIM-192600"
        ),
        "locus": "1q43",
        "protein_size": "894 aa / 104 kDa",
        "inheritance": (
            "AD (missense variants affecting actin-binding or EF-hand domains); "
            "Cardiac dominant phenotype: HCM most common; DCM and LVNC also; "
            "Skeletal myopathy: mild to subclinical; myofibrillar disruption + Z-disc streaming biopsy; "
            "Onset 20-50yr (cardiac often earlier); "
            "CK 1.5-5× ULN — mildly elevated; "
            "Penetrance: moderate-high >70%; "
            "MFM pattern on biopsy: Z-disc streaming + myofibrillar disruption; no specific deposits"
        ),
        "key_features": [
            "CARDIAC DOMINANT — HCM most common ACTN2 phenotype; DCM + LVNC also",
            "Skeletal myopathy mild/subclinical — MFM pattern biopsy (Z-disc streaming)",
            "Z-disc streaming + myofibrillar disruption on biopsy — no specific deposits",
            "HCM: asymmetric septal hypertrophy; outflow tract obstruction possible",
            "LVNC: non-compaction cardiomyopathy — trabeculations on echo/MRI",
            "CK 1.5-5× ULN — mild elevation; may be normal",
            "Onset 20-50yr; cardiac often presents before skeletal symptoms",
            "ACTN2 in cardiomyopathy panel AND MFM panel",
        ],
        "treatment": (
            "Cardiac MANDATORY: echo + Holter annual; "
            "HCM: beta-blocker/verapamil; mavacamten (FDA approved 2022) for obstructive HCM; "
            "LVNC: anticoagulation if severe LV dysfunction; "
            "DCM: standard HF therapy; ICD for SCD prevention; "
            "Skeletal: physiotherapy if symptomatic; often subclinical; "
            "Annual cardiac MRI for LGE assessment in DCM/LVNC"
        ),
        "contraindications": [
            "DO NOT use disopyramide/amiodarone as first line in ACTN2 HCM — mavacamten preferred in obstructive HCM",
            "LVNC anticoagulation: start when EF <35% or LV thrombus — do not withhold",
        ],
        "critical_pearls": [
            "ACTN2: think of it as a cardiomyopathy gene that also causes mild MFM skeletal involvement",
            "HCM + mild proximal myopathy + elevated CK → ACTN2 in DDx (before conventional sarcomere genes)",
            "LVNC + myopathy = ACTN2 or LMNA or RYR2 — check MFM panel",
            "Mavacamten 2022: FDA approved cardiac myosin inhibitor for obstructive HCM — ACTN2 HCM eligible",
            "Z-disc streaming alone is non-specific MFM pattern — confirm with ACTN2 gene sequencing",
        ],
        "mean_age_onset": 35,
        "mean_age_dx": 42,
        "sex_ratio_m_f": "1:1",
        "ck_range": (150, 500),
        "ambulant_at_10yr_pct": 0.97,
        "alive_pct": 0.91,
        "treated_pct": 0.82,
    },
    # -- HSPB8 — Heat shock protein B8 / HSP22, MFM + CMT2L --------------------------------
    {
        "gene": "HSPB8",
        "alt_name": (
            "HSPB8 (HSPB8-196aa-12q24.23 / AD — MFM-CMT2L-HSP22-Overlap — "
            "K141N-K141E-HOTSPOT-PATHOGNOMONIC-Distal-Onset-Rimmed-Vacuoles — "
            "Distal-Muscle-Weakness-Axonal-Neuropathy-Overlap)"
        ),
        "protein": (
            "HSPB8 -- 12q24.23 AD -- HSPB8-196aa -- "
            "Heat-Shock-Protein-B8-HSP22-Small-Heat-Shock-Protein-22kDa -- "
            "MFM-Myofibrillar-Myopathy-CMT2L-Charcot-Marie-Tooth-Type-2L-OMIM-612951 -- "
            "K141N-K141E-Hotspot-Alpha-Crystallin-Domain-PATHOGNOMONIC -- "
            "Distal-Muscle-Weakness-Onset-Feet-Hands-Peroneal-Distribution -- "
            "Rimmed-Vacuoles-Biopsy-Key-Feature -- "
            "Axonal-Neuropathy-NCS-EMG-Mandatory -- "
            "CK-1.5-6x-ULN-Mild-Elevation -- "
            "Onset-20-45yr-Distal -- "
            "dHMN-Distal-Hereditary-Motor-Neuropathy-Allelic -- "
            "OMIM-Gene-HSPB8-608014-Disease-CMT2L-612951"
        ),
        "locus": "12q24.23",
        "protein_size": "196 aa / 22 kDa",
        "inheritance": (
            "AD (K141N or K141E hotspot — dominant negative; nearly all pathogenic variants at this site); "
            "DISTAL onset — feet first, then hands (peroneal + intrinsic hand muscles); "
            "Rimmed vacuoles on biopsy + myofibrillar disruption; "
            "Axonal neuropathy: NCS/EMG mandatory — reduced amplitude, preserved conduction velocity; "
            "dHMN (distal hereditary motor neuropathy) allelic — motor-predominant variant; "
            "CK 1.5-6× ULN mild elevation; "
            "Onset 20-45yr distal; "
            "No significant cardiac involvement"
        ),
        "key_features": [
            "K141N/K141E HOTSPOT — nearly all HSPB8 pathogenic variants at this site PATHOGNOMONIC",
            "DISTAL ONSET — peroneal distribution (feet first, then hands)",
            "Rimmed vacuoles + myofibrillar disruption biopsy — DDx from GNE/VCP (also rimmed vacuoles)",
            "AXONAL NEUROPATHY — NCS/EMG mandatory; reduced amplitude, normal velocity",
            "dHMN allelic — motor-predominant neuropathy variant without myopathy",
            "CMT2L + MFM overlap — both from same gene/hotspot",
            "No cardiac involvement — distinguishes from DES/CRYAB/FLNC/BAG3/ACTN2",
            "CK 1.5-6× ULN — mild; may be near-normal",
        ],
        "treatment": (
            "No disease-modifying therapy; "
            "Skeletal: AFOs mandatory for foot drop; "
            "Physiotherapy for upper + lower limb distal weakness; "
            "NCS/EMG baseline and 3-yearly for neuropathy progression monitoring; "
            "No cardiac surveillance — no cardiomyopathy in HSPB8; "
            "Genetic counselling — AD 50% offspring risk; K141N/K141E confirm with targeted sequencing"
        ),
        "contraindications": [
            "DO NOT start neuropathy workup without MFM biopsy consideration — HSPB8 causes both; biopsy + NCS/EMG together",
            "DO NOT reassure on cardiac — verify no FLNC/DES/BAG3 variant before excluding cardiac surveillance",
        ],
        "critical_pearls": [
            "K141N/K141E: if found in sequencing for any distal myopathy/neuropathy → HSPB8 confirmed; highly specific",
            "HSPB8 distal + rimmed vacuoles: DDx GNE (QUADRICEPS SPARED), VCP (IBMPFD triad) — different genes",
            "CMT2L vs MFM HSPB8: same variant, different phenotypic emphasis; both can coexist in one patient",
            "NCS/EMG essential: axonal neuropathy found in ~70% of HSPB8 patients; not clinically obvious",
            "dHMN-IIB: pure motor variant of same K141N hotspot — no myopathy; different phenotype, same gene",
        ],
        "mean_age_onset": 33,
        "mean_age_dx": 42,
        "sex_ratio_m_f": "1:1",
        "ck_range": (150, 600),
        "ambulant_at_10yr_pct": 0.88,
        "alive_pct": 0.95,
        "treated_pct": 0.58,
    },
]


def _make_patients(gene_data: dict, seed: int, n: int = 40) -> list:
    """Generate n synthetic patients for a given gene using the provided seed."""
    rng = random.Random(seed)
    gene = gene_data["gene"]
    ck_lo, ck_hi = gene_data["ck_range"]
    alive_p = gene_data["alive_pct"]
    ambulant_p = gene_data["ambulant_at_10yr_pct"]
    treated_p = gene_data["treated_pct"]
    mean_onset = gene_data["mean_age_onset"]
    mean_dx = gene_data["mean_age_dx"]

    patients = []
    for i in range(n):
        age_onset = max(0, int(rng.gauss(mean_onset, mean_onset * 0.25)))
        dx_delay = max(1, int(rng.gauss(mean_dx - mean_onset, 3)))
        age_dx = age_onset + dx_delay
        current_age = age_dx + rng.randint(1, 20)
        ck_val = max(50, rng.gauss((ck_lo + ck_hi) / 2, (ck_hi - ck_lo) / 4))
        alive = rng.random() < alive_p
        ambulant = rng.random() < ambulant_p
        treated = rng.random() < treated_p
        attacks_py = round(rng.uniform(0, 2), 1)
        patients.append({
            "patient_id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "age_onset": age_onset,
            "age_dx": age_dx,
            "dx_delay_yr": dx_delay,
            "current_age": current_age,
            "ck_iu_l": int(ck_val),
            "ambulant_at_10yr": ambulant,
            "alive": alive,
            "treated": treated,
            "attacks_per_year": attacks_py,
            "sex": rng.choice(["M", "F"]),
        })
    return patients


def overview():
    """Aggregate overview across all 8 genes (320 patients)."""
    all_patients = []
    for i, gd in enumerate(MFM_GENES):
        all_patients += _make_patients(gd, SEED_BASE + i)

    n = len(all_patients)
    alive = sum(1 for p in all_patients if p["alive"])
    treated = sum(1 for p in all_patients if p["treated"])
    ambulant = sum(1 for p in all_patients if p["ambulant_at_10yr"])
    mean_onset = round(sum(p["age_onset"] for p in all_patients) / n, 1)
    mean_dx_delay = round(sum(p["dx_delay_yr"] for p in all_patients) / n, 1)
    mean_ck = round(sum(p["ck_iu_l"] for p in all_patients) / n, 0)

    gene_summaries = []
    for i, gd in enumerate(MFM_GENES):
        pts = _make_patients(gd, SEED_BASE + i)
        gene_summaries.append({
            "gene": gd["gene"],
            "n_patients": len(pts),
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "inheritance": gd["inheritance"].split(";")[0].strip(),
            "mean_onset_yr": round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "mean_ck": round(sum(p["ck_iu_l"] for p in pts) / len(pts), 0),
            "alive_pct": round(sum(1 for p in pts if p["alive"]) / len(pts) * 100, 1),
            "ambulant_10yr_pct": round(sum(1 for p in pts if p["ambulant_at_10yr"]) / len(pts) * 100, 1),
        })

    return {
        "atlas": "Hereditary-Myofibrillar-Myopathy-Atlas",
        "subtitle": (
            "Complete 8-Gene Myofibrillar Myopathy (MFM) Spectrum "
            "(DES · CRYAB · MYOT · FLNC · BAG3 · PYROXD1 · ACTN2 · HSPB8)"
        ),
        "genes_covered": [gd["gene"] for gd in MFM_GENES],
        "n_genes": 8,
        "n_patients": n,
        "seeds": f"{SEED_BASE}-{SEED_BASE+7}",
        "alive_pct": round(alive / n * 100, 1),
        "treated_pct": round(treated / n * 100, 1),
        "ambulant_10yr_pct": round(ambulant / n * 100, 1),
        "mean_age_onset_yr": mean_onset,
        "mean_dx_delay_yr": mean_dx_delay,
        "mean_ck_iu_l": int(mean_ck),
        "key_spectrum_facts": [
            "DES: CYTOPLASMIC DESMIN AGGREGATES on biopsy PATHOGNOMONIC; cardiac conduction disease+DCM mandatory; scapuloperoneal; most common MFM gene historically",
            "CRYAB: POSTERIOR SUBCAPSULAR CATARACTS 50% AD carriers PATHOGNOMONIC; cardiac DCM+HCM; R120G hotspot; desmin-related small heat shock protein",
            "MYOT: LATE ONSET 40-65yr PATHOGNOMONIC vs other MFM; NO cardiac — critical DDx from DES/FLNC/BAG3; LGMD1A allelic; filamentous inclusions",
            "FLNC: ALLELE-SPECIFIC CARDIAC — truncating→DCM/ARVC; missense→MFM5 skeletal; HYALINE BODIES biopsy pathognomonic; most identified MFM gene in modern cohorts",
            "BAG3: CHILDHOOD ONSET 2-15yr PATHOGNOMONIC; most severe MFM; DCM mandatory childhood; axial hypotonia; respiratory failure early; P209L/P209S hotspot",
            "PYROXD1: AR childhood MFM; HYBRID BIOPSY (nemaline rods + myofibrillar disruption); ptosis+facial weakness; N155S/Q372H founders; no cardiac",
            "ACTN2: Z-disc sarcomere; CARDIAC DOMINANT (HCM/DCM/LVNC); skeletal mild/subclinical; mavacamten for obstructive HCM 2022",
            "HSPB8: K141N/K141E HOTSPOT PATHOGNOMONIC; DISTAL ONSET peroneal; rimmed vacuoles + MFM biopsy; axonal neuropathy NCS/EMG mandatory; CMT2L allelic",
        ],
        "cardiac_mandate_genes": ["DES", "CRYAB", "FLNC", "BAG3", "ACTN2"],
        "no_cardiac_genes": ["MYOT", "PYROXD1", "HSPB8"],
        "childhood_onset_genes": ["BAG3", "PYROXD1"],
        "late_onset_genes": ["MYOT", "CRYAB"],
        "critical_ddx": {
            "BAG3_vs_DES_onset":    "BAG3: childhood 2-15yr; DES: adult 20-40yr — onset age is the single best MFM subtype discriminator",
            "FLNC_truncating_vs_missense": "FLNC truncating → DCM/ARVC (cardiac dominant); FLNC missense → MFM5 skeletal; always check variant type",
            "MYOT_no_cardiac_vs_DES_cardiac": "MYOT: NO cardiac; DES/CRYAB/FLNC/BAG3: cardiac mandatory — if MFM biopsy + late onset + no cardiac → MYOT first",
            "HSPB8_vs_GNE_rimmed_vacuoles": "HSPB8: distal onset + K141N/K141E + neuropathy; GNE: quadriceps spared + M712T/V572L + no neuropathy",
            "PYROXD1_vs_NEB_nemaline": "PYROXD1: hybrid biopsy (rods + MFM disruption) + AR + ptosis; NEB: pure nemaline rods + AR + ankle contractures",
            "CRYAB_vs_MYOT_cataracts": "CRYAB: cataracts + cardiac; MYOT: no cataracts + no cardiac + late onset; cataracts distinguish CRYAB",
            "ACTN2_vs_MYH7_HCM":       "ACTN2: HCM + myopathy + mildly elevated CK; MYH7: HCM + no myopathy + normal CK; CK elevation + HCM → ACTN2 panel",
        },
        "gene_summaries": gene_summaries,
    }


def breakdown():
    """Per-gene clinical breakdown with key metrics."""
    result = {}
    for i, gd in enumerate(MFM_GENES):
        pts = _make_patients(gd, SEED_BASE + i)
        result[gd["gene"]] = {
            "gene": gd["gene"],
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "n_patients": len(pts),
            "alive_pct": round(sum(1 for p in pts if p["alive"]) / len(pts) * 100, 1),
            "treated_pct": round(sum(1 for p in pts if p["treated"]) / len(pts) * 100, 1),
            "ambulant_10yr_pct": round(sum(1 for p in pts if p["ambulant_at_10yr"]) / len(pts) * 100, 1),
            "mean_age_onset": round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "mean_dx_delay_yr": round(sum(p["dx_delay_yr"] for p in pts) / len(pts), 1),
            "mean_ck_iu_l": int(sum(p["ck_iu_l"] for p in pts) / len(pts)),
            "mean_attacks_per_year": round(sum(p["attacks_per_year"] for p in pts) / len(pts), 1),
            "key_features": gd["key_features"],
            "treatment": gd["treatment"],
            "contraindications": gd["contraindications"],
            "critical_pearls": gd["critical_pearls"],
            "inheritance": gd["inheritance"][:200],
        }
    return result


def definitions():
    """Gene definitions, biopsy patterns, cardiac surveillance table, MFM DDx tables, glossary."""
    gene_defs = {}
    for gd in MFM_GENES:
        gene_defs[gd["gene"]] = {
            "full_name": gd["protein"],
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "inheritance_detail": gd["inheritance"],
            "alt_name": gd["alt_name"],
        }

    return {
        "gene_definitions": gene_defs,
        "onset_age_spectrum": {
            "BAG3_MFM6":    "Childhood 2-15yr — EARLIEST; axial hypotonia from infancy; DCM mandatory; most severe",
            "PYROXD1_AR":   "Childhood to early adult 2-30yr — AR; ptosis + facial weakness; no cardiac",
            "DES_MFM1":     "Young adult 20-40yr — most common MFM; scapuloperoneal; desmin aggregates",
            "HSPB8_CMT2L":  "Adult 20-45yr — distal peroneal distribution; K141N/K141E; neuropathy",
            "ACTN2":        "Adult 20-50yr — cardiac dominant (HCM/DCM/LVNC); skeletal mild",
            "FLNC_MFM5":    "Adult 30-55yr — missense MFM5; truncating DCM/ARVC earlier cardiac",
            "CRYAB_MFM2":   "Adult 35-55yr — R120G most common; cataracts; cardiac DCM+HCM",
            "MYOT_MFM3":    "Late adult 40-65yr — LATEST; no cardiac; filamentous inclusions; LGMD1A allelic",
        },
        "biopsy_pattern_table": {
            "DES":     "Granulofilamentous material (TEM); desmin IHC cytoplasmic deposits; Z-disc disruption; myofibrillar disorganisation",
            "CRYAB":   "αB-crystallin IHC deposits; spheroid inclusions; hyaline plaques; myofibrillar disruption",
            "MYOT":    "Myotilin IHC filamentous cytoplasmic inclusions; Z-disc streaming; myofibrillar disruption",
            "FLNC":    "Hyaline bodies (eosinophilic, type 1 fibres) PATHOGNOMONIC MFM5; filamin-C IHC deposits; myofibrillar disorganisation",
            "BAG3":    "Myofibrillar disruption; Z-disc streaming; BAG3/HSPB6/8 IHC aggregates; rimmed vacuoles occasional",
            "PYROXD1": "HYBRID: nemaline-like rods + myofibrillar disruption + core-like areas; PYROXD1 IHC research",
            "ACTN2":   "Z-disc streaming; myofibrillar disruption; no specific deposits — non-specific MFM pattern; ACTN2 IHC experimental",
            "HSPB8":   "Rimmed vacuoles + myofibrillar disruption; HSPB8/BAG3 IHC aggregates; Z-disc disorganisation",
        },
        "cardiac_surveillance_table": {
            "DES":     "Echo + Holter annual; ICD for SCD if DCM/arrhythmia; pacemaker for AV block; start at diagnosis",
            "CRYAB":   "Echo + Holter annual; ICD if DCM with arrhythmia; manage HCM per guidelines; slit-lamp ophthalmology",
            "MYOT":    "NO cardiac surveillance — MYOT does NOT cause cardiomyopathy; confirm no other pathogenic variant before omitting",
            "FLNC":    "Echo + Holter 6-monthly ALL carriers; truncating: ICD for DCM/ARVC low threshold; RV function + MRI; avoid contact sports",
            "BAG3":    "Echo + Holter 6-monthly; ICD for DCM — childhood; BAG3 major DCM gene; NMD+cardiology co-manage",
            "PYROXD1": "NO cardiac surveillance — PYROXD1 no cardiomyopathy; confirm diagnosis before omitting",
            "ACTN2":   "Echo + Holter annual + cardiac MRI; HCM: mavacamten if obstructive; DCM ICD; LVNC anticoagulate if EF <35%",
            "HSPB8":   "NO cardiac surveillance — HSPB8 no cardiomyopathy; distinguish from DES/FLNC/BAG3 where cardiac is mandatory",
        },
        "ddx_table": {
            "MFM_biopsy_onset_guide": {
                "childhood_(<15yr)":  "BAG3 (DCM + axial) > PYROXD1 (AR, ptosis, no cardiac) > DES AR (rare)",
                "young_adult_(20-40yr)": "DES (desmin deposits, cardiac) > FLNC (hyaline bodies, truncating cardiac/ARVC) > HSPB8 (distal, K141N)",
                "mid_adult_(35-55yr)": "CRYAB (cataracts, cardiac) > FLNC (hyaline bodies) > ACTN2 (cardiac dominant, HCM) > DES",
                "late_adult_(40-65yr)": "MYOT (no cardiac, LGMD1A allelic, filamentous inclusions) > CRYAB late > DES late",
            },
            "cardiac_present_MFM": "DES / CRYAB / FLNC / BAG3 / ACTN2 — order full MFM+cardiomyopathy panel",
            "no_cardiac_MFM":      "MYOT / PYROXD1 / HSPB8 — no cardiac mandatory in these three",
            "HSPB8_vs_GNE":        "HSPB8: K141N + neuropathy + distal feet/hands; GNE: quads spared + M712T/V572L + no neuropathy",
            "FLNC_truncating_DDx": "FLNC truncating → ARVC panel + PKP2/DSP/DSG2/DSC2; FLNC missense → MFM5 panel",
            "BAG3_vs_congenital_myopathy": "BAG3: childhood + DCM + neuropathy + myofibrillar biopsy; congenital myopathy: cores/nemaline + RYR1/NEB/ACTA1; different panels",
            "PYROXD1_vs_NEB":      "PYROXD1: hybrid rods+MFM + ptosis + AR; NEB: pure nemaline + ankle contractures + AR; gene panels distinguish",
        },
        "founder_mutations": {
            "CRYAB_R120G":    "CRYAB p.Arg120Gly — most common CRYAB pathogenic variant worldwide; α-crystallin domain; MFM2 + cataracts + cardiac",
            "BAG3_P209L":     "BAG3 p.Pro209Leu — most common BAG3 variant; WW-domain; childhood MFM6; DCM mandatory; dominant negative",
            "BAG3_P209S":     "BAG3 p.Pro209Ser — second most common BAG3 variant; WW-domain; similar phenotype to P209L",
            "HSPB8_K141N":    "HSPB8 p.Lys141Asn — most common HSPB8 variant; α-crystallin domain; CMT2L + MFM distal",
            "HSPB8_K141E":    "HSPB8 p.Lys141Glu — second most common HSPB8 variant at same hotspot; identical clinical phenotype to K141N",
            "MYOT_T57I":      "MYOT p.Thr57Ile — most common MYOT pathogenic variant; Ig-like domain cluster; late-onset MFM3/LGMD1A",
            "PYROXD1_N155S":  "PYROXD1 p.Asn155Ser — Australian founder; childhood MFM; AR with Q372H compound heterozygous",
        },
        "mfm_biopsy_ihc_panel": {
            "first_line": ["Desmin IHC", "αB-crystallin IHC", "Myotilin IHC", "Filamin-C IHC"],
            "second_line": ["BAG3 IHC", "HSPB8 IHC", "Ubiquitin IHC", "p62/SQSTM1 IHC"],
            "note": (
                "A complete MFM IHC panel is mandatory when MFM biopsy pattern is seen (myofibrillar disruption + inclusions). "
                "TEM (transmission electron microscopy) adds granulofilamentous material identification. "
                "IHC deposits identify the accumulated protein but NOT always the causative gene — gene sequencing required."
            ),
        },
        "glossary": {
            "Myofibrillar_myopathy":      "Pathological group of myopathies sharing myofibrillar disruption, Z-disc streaming, and protein aggregates on biopsy; 8 major genes (DES/CRYAB/MYOT/FLNC/BAG3/PYROXD1/ACTN2/HSPB8)",
            "Desmin_aggregates":          "Granulofilamentous material accumulating in sarcomere after desmin (DES) mutation — pathognomonic of DES MFM1; detectable by desmin IHC and TEM",
            "Hyaline_bodies":             "Eosinophilic inclusions in type 1 muscle fibres — pathognomonic of FLNC MFM5; contain filamin-C protein deposits; visible HE stain",
            "Rimmed_vacuoles":            "Autophagic vacuoles rimmed with basophilic material (mGT stain) — seen in HSPB8, GNE, VCP, BAG3; different genes, same biopsy finding",
            "Z_disc_streaming":           "Z-disc disorganisation with streaming or fragmentation — common to all MFM subtypes; non-specific MFM feature requiring IHC panel",
            "Dominant_negative":          "Mechanism where mutant protein interferes with normal protein function; one pathogenic allele sufficient; seen in BAG3 (P209L), HSPB8 (K141N), DES missense",
            "CMT2L":                      "Charcot-Marie-Tooth type 2L — axonal neuropathy caused by HSPB8 K141N/K141E; allelic with HSPB8 MFM distal myopathy",
            "dHMN":                       "Distal hereditary motor neuropathy — motor-predominant peripheral neuropathy; HSPB8 K141N causes dHMN-IIB; allelic with CMT2L",
            "LGMD1A":                     "Limb-girdle muscular dystrophy type 1A — AD LGMD caused by MYOT; allelic with MYOT MFM3; same gene, different phenotypic emphasis",
            "Mavacamten":                 "Cardiac myosin inhibitor (FDA approved 2022) for obstructive HCM; reduces excess actin-myosin cross-bridge formation; relevant for ACTN2/MYH7/MYBPC3 HCM",
            "ARVC_epsilon_wave":          "Epsilon wave on ECG — terminal notch after QRS in V1-V3; ARVC pathognomonic; seen in FLNC truncating + PKP2/DSP/DSG2/DSC2",
            "Small_heat_shock_protein":   "CRYAB and HSPB8 both small heat shock proteins; chaperone function; aggregation when mutant → protein mishandling → MFM",
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = overview()
    print(json.dumps({k: v for k, v in ov.items() if k != "gene_summaries"}, indent=2))
    print("\n=== BREAKDOWN keys ===")
    br = breakdown()
    for gene, data in br.items():
        print(f"  {gene}: {data['n_patients']} patients, alive={data['alive_pct']}%, ambulant_10yr={data['ambulant_10yr_pct']}%")
    print("\n=== DEFINITIONS keys ===")
    defs = definitions()
    print(list(defs.keys()))
