#!/usr/bin/env python3
"""Hereditary-Cardiomyopathy-Atlas -- Complete 8-Gene Reference
Hereditary cardiomyopathies (HCM, DCM, ARVC, mixed phenotypes):
MYH7 (HCM-Most-Common-35-50pct-R403Q-disopyramide-mavacamten),
MYBPC3 (HCM-30-40pct-Frameshift-Late-Onset-mavacamten-FDA2022),
LMNA (DCM-AV-Block-PATHOGNOMONIC-ICD-Early-VT-Storm),
PLN (DCM-R14del-Dutch-Founder-PATHOGNOMONIC-ICD-MANDATORY),
FLNC (HCM-DCM-ARVC-Truncating-ICD-SCD-Risk),
RBM20 (DCM-Aggressive-R634W-Hotspot-Young-Onset-Transplant),
MYH6 (DCM-ASD-SSS-CRT-Excellent),
TTN (DCM-Most-Common-25pct-HCM-Modifier-Incomplete-Penetrance).

MYH7   (Beta-myosin heavy chain; 1935aa; 14q11.2; AD GOF;
         HCM most common HCM gene (35-50%); R403Q R453C PATHOGNOMONIC;
         disopyramide/mavacamten for LVOTO; septal reduction;
         seed SEED_BASE+0).
MYBPC3 (Myosin-binding protein C3; 1274aa; 11p11.2; AD LOF;
         HCM 30-40%; frameshift/nonsense most common;
         late-onset penetrance complete by 70yr; mavacamten FDA2022;
         seed SEED_BASE+1).
LMNA   (Lamin A/C; 664aa; 1q22; AD LOF;
         DCM 5-10% familial; AV block PATHOGNOMONIC; VT storm;
         ICD before LVEF 35% (Wahbi score); EDMD/LGMD overlap;
         seed SEED_BASE+2).
PLN    (Phospholamban; 52aa; 6q22.31; AD LOF dominant-negative;
         DCM+ARVC overlap; R14del Dutch founder PATHOGNOMONIC;
         ICD MANDATORY; SERCA2a dysregulation;
         seed SEED_BASE+3).
FLNC   (Filamin C; 2725aa; 7q32.1; AD LOF truncating;
         HCM+DCM+ARVC spectrum; truncating = highest SCD;
         ICD even preserved EF + NSVT/LGE; distal myopathy overlap;
         seed SEED_BASE+4).
RBM20  (RNA-binding motif 20; 1227aa; 10q25.2; AD GOF;
         DCM aggressive young onset; R634W hotspot PATHOGNOMONIC;
         biventricular fibrosis; highest transplant rate;
         seed SEED_BASE+5).
MYH6   (Alpha-MHC; 1939aa; 14q11.2; AD LOF;
         DCM+ASD+SSS; excellent CRT response;
         seed SEED_BASE+6).
TTN    (Titin; 34350aa; 2q31.2; AD/AR LOF;
         DCM most common gene 25%; incomplete penetrance;
         PPCM enriched; HCM modifier;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3750-3757)
"""
import random

SEED_BASE = 3750

ATLAS_GENES = [
    {
        "gene": "MYH7",
        "protein": (
            "MYH7 -- 14q11.2 AD GOF -- 1935aa -- "
            "beta-MHC-HCM-Most-Common-35-50pct-R403Q-R453C-PATHOGNOMONIC-"
            "LVOTO-disopyramide-mavacamten-septal-reduction-"
            "MYH7-OMIM-160760"
        ),
        "locus": "14q11.2",
        "protein_size": (
            "1935 aa / ~223 kDa / 14q11.2 MYH7 beta-myosin heavy chain molecular context: "
            "STRUCTURE: "
            "  1935 aa / ~223 kDa; sarcomeric thick filament motor protein; "
            "  N-terminal motor domain (ATP hydrolysis, actin binding); "
            "  Converter domain: R403Q, R453C, R719Q pathognomonic hotspot mutations; "
            "  Lever arm + light-chain binding domain; coiled-coil rod (filament formation); "
            "  MYH7 mutations -> hypercontractility (GOF) or impaired cross-bridge cycling; "
            "HCM (AD GOF missense -- most common variant class): "
            "  Most common HCM gene: 35-50% of genotype-positive HCM; "
            "  Missense variants cluster in motor/converter domain; "
            "  R403Q: PATHOGNOMONIC high-penetrance variant; severe HCM phenotype; "
            "  R453C: severe HCM; malignant course; "
            "  R719Q: HCM with risk of sudden death; "
            "  Morphology: asymmetric septal hypertrophy; LV outflow obstruction (LVOTO); "
            "  SAM (systolic anterior motion of mitral valve): LVOTO sign; "
            "TREATMENT LVOTO: "
            "  Beta-blocker: FIRST-LINE for LVOTO/symptoms; "
            "  Disopyramide: LVOTO reduction (negative inotrope + antiarrhythmic); "
            "  Mavacamten (MyoKardia/BMS): allosteric myosin-II inhibitor; "
            "    FDA-approved HCM (EXPLORER-HCM trial 2020); reduces LVOTO + improves symptoms; "
            "    Monitor LVEF: hold if LVEF <50%; no concurrent strong CYP2C19 inhibitors; "
            "  Septal reduction: surgical myectomy (Morrow procedure) gold standard for LVOTO; "
            "  Alcohol septal ablation: catheter-based alternative; "
            "DCM (rare MYH7 LOF ~5%): "
            "  Loss-of-function frameshift/truncating MYH7 variants -> dilated CMP; "
            "  ICD as per standard DCM LVEF criteria; "
            "SCD RISK STRATIFICATION: "
            "  ESC 2014 HCM SCD risk calculator: 5-yr risk >=6% -> ICD indicated; "
            "  Markers: max wall thickness, family SCD history, NSVT, abnormal BP response; "
            "SURVEILLANCE: "
            "  Annual echocardiogram + 48h Holter; exercise test with BP; "
            "  Cascade genetic testing all 1st-degree relatives; "
            "  Pregnancy: high-risk obstetric care; avoid dehydration + vasodilators"
        ),
        "syndrome": "HCM-Most-Common / DCM-rare-LOF",
        "inheritance": "AD GOF (HCM missense); AD LOF (DCM rare)",
        "pathognomonic": "R403Q/R453C missense HCM PATHOGNOMONIC; SAM + LVOTO in HCM; disopyramide/mavacamten for LVOTO",
        "key_drugs_ci": ["vasodilators-CI-LVOTO", "digoxin-CI-LVOTO", "mavacamten-hold-LVEF-lt50pct"],
        "surveillance": ["annual-echo-Holter", "ESC-SCD-risk-score-5yr", "cascade-screen-1st-degree"],
    },
    {
        "gene": "MYBPC3",
        "protein": (
            "MYBPC3 -- 11p11.2 AD LOF -- 1274aa -- "
            "MyBP-C-HCM-30-40pct-Frameshift-Late-Onset-Penetrance-70yr-"
            "mavacamten-FDA2022-septal-reduction-"
            "MYBPC3-OMIM-600958"
        ),
        "locus": "11p11.2",
        "protein_size": (
            "1274 aa / ~141 kDa / 11p11.2 MYBPC3 myosin-binding protein C3 molecular context: "
            "STRUCTURE: "
            "  1274 aa / ~141 kDa; thick filament accessory protein; "
            "  Contains 8 Ig-like domains + 3 fibronectin type III domains; "
            "  C10 domain: binds myosin rod; C1-C2: binds actin; "
            "  Phosphorylation by PKA (cAMP-dependent) regulates contractility; "
            "  LOF (haploinsufficiency): reduced MyBP-C levels -> unconstrained cross-bridge cycling; "
            "HCM (AD LOF -- most common molecular mechanism in MYBPC3-HCM): "
            "  Most common single HCM gene at population level (30-40% genotype-positive HCM); "
            "  Frameshift/nonsense variants most common (haploinsufficiency); "
            "  LATE-ONSET phenotype: key distinguishing feature vs MYH7; "
            "    Penetrance near-complete by age 70yr; "
            "    Normal echo in 3rd-4th decade common; overt HCM emerges 5th-6th decade; "
            "    Clinical implication: do NOT reassure gene+ young adults; rescan every 5yr; "
            "  Atrial fibrillation: COMMON in MYBPC3-HCM (up to 20-30%); "
            "  LVOTO present in ~25-30% symptomatic; "
            "TREATMENT: "
            "  Mavacamten: FDA-approved 2022 (EXPLORER-HCM); first-in-class for obstructive HCM; "
            "    Dose: 2.5-15 mg/day titrated; weekly/monthly echo; "
            "    Hold if LVEF <50%; "
            "    Drug interaction: CYP2C19 inhibitors (omeprazole, fluoxetine) increase mavacamten exposure; "
            "  Disopyramide: alternative for LVOTO; monitor QTc; "
            "  Beta-blocker: first-line symptomatic HCM; "
            "  Myectomy / ASA: for drug-refractory LVOTO; "
            "AF MANAGEMENT: "
            "  Rhythm control preferred: AF -> loss of atrial kick -> worsening LVOTO; "
            "  Anticoagulation: CHA2DS2-VASc score; NOAC preferred; "
            "SURVEILLANCE: "
            "  Gene-positive/phenotype-negative: echo every 5yr; "
            "  Gene-positive/phenotype-positive: annual echo + Holter; "
            "  ESC SCD risk score annually if HCM phenotype present"
        ),
        "syndrome": "HCM-Late-Onset / AF-common",
        "inheritance": "AD LOF (haploinsufficiency frameshift/nonsense)",
        "pathognomonic": "late-onset HCM penetrance complete-by-70yr PATHOGNOMONIC for MYBPC3; frameshift most common variant class",
        "key_drugs_ci": ["mavacamten-hold-LVEF-lt50pct", "CYP2C19-inhibitors-increase-mavacamten-exposure"],
        "surveillance": ["gene-positive-echo-5yr", "annual-echo-Holter-if-phenotype", "ESC-SCD-risk-annual"],
    },
    {
        "gene": "LMNA",
        "protein": (
            "LMNA -- 1q22 AD LOF -- 664aa -- "
            "Lamin-A/C-nuclear-envelope-DCM-5-10pct-AV-Block-PATHOGNOMONIC-"
            "VT-Storm-ICD-Early-Muscle-Involvement-"
            "LMNA-OMIM-150330"
        ),
        "locus": "1q22",
        "protein_size": (
            "664 aa / ~74 kDa / 1q22 LMNA Lamin A/C nuclear envelope protein molecular context: "
            "STRUCTURE: "
            "  664 aa / ~74 kDa; nuclear lamina intermediate filament; "
            "  N-terminal head domain; central coiled-coil rod; C-terminal Ig-like domain; "
            "  Alternative splicing: Lamin A (664aa) and Lamin C (572aa); "
            "  Forms nuclear lamina meshwork; nuclear structural integrity; "
            "DCM-LMNA (5-10% of familial DCM): "
            "  AV CONDUCTION BLOCK PATHOGNOMONIC: "
            "    First-degree or high-degree AV block PRECEDING significant LV dilation; "
            "    AV block in young DCM patient -> immediate LMNA genetic testing; "
            "    Pacemaker for symptomatic AV block; "
            "  VT STORM: malignant ventricular arrhythmias; storm risk high even with preserved LVEF; "
            "  Rapid progression to end-stage HF: poor prognosis compared to other DCM genes; "
            "  Skeletal muscle overlap: "
            "    EDMD (Emery-Dreifuss muscular dystrophy): contractures + rigid spine + cardiac; "
            "    LGMD1B (limb-girdle muscular dystrophy type 1B); "
            "ICD TIMING -- LMNA-SPECIFIC RULE: "
            "  ICD indicated EARLIER than standard LVEF <35% threshold; "
            "  ICD recommended if: LVEF <45% + NSVT or syncope; OR "
            "  LMNA variant + AV block + male sex -> ICD prophylactic; "
            "  LMNA-SPECIFIC RISK SCORE: Wahbi 2019 score; "
            "  DO NOT wait for LVEF <35% in LMNA -- VT storm can occur at preserved EF; "
            "PHARMACOLOGY: "
            "  Beta-blocker: first-line HF + arrhythmia suppression; "
            "  ACE inhibitor/ARB/ARNI: standard HF therapy; "
            "  Amiodarone: VT storm bridging; "
            "SURVEILLANCE: "
            "  ECG + Holter annually (AV block monitoring); "
            "  Echocardiogram annually; "
            "  CK + skeletal muscle assessment; "
            "  Wahbi LMNA ICD risk score at every encounter"
        ),
        "syndrome": "DCM-LMNA / AV-block-PATHOGNOMONIC / EDMD-LGMD-overlap",
        "inheritance": "AD LOF",
        "pathognomonic": "AV-block-preceding-DCM = LMNA PATHOGNOMONIC; VT-storm-preserved-EF; ICD-before-LVEF-35pct",
        "key_drugs_ci": ["delay-ICD-LVEF35-rule-does-NOT-apply-LMNA"],
        "surveillance": ["annual-ECG-Holter-AV-block", "annual-echo", "Wahbi-LMNA-ICD-score", "CK-skeletal-muscle"],
    },
    {
        "gene": "PLN",
        "protein": (
            "PLN -- 6q22.31 AD LOF dominant-negative -- 52aa -- "
            "Phospholamban-DCM-R14del-Dutch-Founder-PATHOGNOMONIC-"
            "ARVC-overlap-ICD-MANDATORY-SERCA2a-"
            "PLN-OMIM-172405"
        ),
        "locus": "6q22.31",
        "protein_size": (
            "52 aa / ~6 kDa / 6q22.31 PLN Phospholamban molecular context: "
            "STRUCTURE: "
            "  52 aa / ~6 kDa; smallest human disease-causing protein; "
            "  Single-pass transmembrane protein; pentameric assembly in SR membrane; "
            "  Cytoplasmic domain: phosphorylation by PKA (Ser16) and CaMKII (Thr17); "
            "  Unphosphorylated PLN: INHIBITS SERCA2a (SR Ca2+-ATPase); "
            "  PLN LOF (dominant-negative): constitutive SERCA2a inhibition -> Ca2+ overload; "
            "PLN R14del -- DUTCH FOUNDER VARIANT: "
            "  R14del (arginine-14 deletion): PATHOGNOMONIC Dutch founder mutation; "
            "  Prevalence: ~1:2000 in Netherlands; "
            "  Phenotype: DCM + ARVC overlap (dilated LV + fibrofatty infiltration); "
            "  Malignant ventricular arrhythmias (VT/VF) even with preserved LVEF; "
            "  ICD MANDATORY in R14del carriers with ANY of: "
            "    NSVT, syncope, LV dysfunction, LGE on CMR; "
            "  CMR: intramyocardial LGE (subepicardial and mid-wall); "
            "MECHANISM: "
            "  R14del: dominant-negative gain-of-inhibition -> excess SERCA2a inhibition; "
            "  Result: impaired SR Ca2+ reuptake -> systolic + diastolic dysfunction; "
            "TREATMENT: "
            "  ICD: MANDATORY in symptomatic/LGE-positive R14del carriers; "
            "  Standard HF therapy: ACEi/ARB/ARNI + beta-blocker + MRA; "
            "  Cardiac transplantation: R14del has high transplant rate; "
            "SURVEILLANCE: "
            "  Annual CMR (LGE assessment critical); "
            "  Annual echo + Holter + exercise test; "
            "  Cascade testing in Dutch/Dutch-origin families"
        ),
        "syndrome": "DCM-ARVC-overlap / PLN-R14del-Dutch-Founder",
        "inheritance": "AD LOF dominant-negative (R14del)",
        "pathognomonic": "R14del-Dutch-founder-DCM-ARVC-overlap PATHOGNOMONIC; ICD-MANDATORY-R14del; CMR-LGE-subepicardial",
        "key_drugs_ci": ["delay-ICD-LVEF-threshold-does-NOT-apply-R14del"],
        "surveillance": ["annual-CMR-LGE", "annual-echo-Holter-exercise", "cascade-Dutch-families"],
    },
    {
        "gene": "FLNC",
        "protein": (
            "FLNC -- 7q32.1 AD LOF truncating -- 2725aa -- "
            "Filamin-C-HCM-DCM-ARVC-Distal-Myopathy-"
            "Truncating-ICD-SCD-Risk-even-preserved-EF-"
            "FLNC-OMIM-102565"
        ),
        "locus": "7q32.1",
        "protein_size": (
            "2725 aa / ~291 kDa / 7q32.1 FLNC Filamin C molecular context: "
            "STRUCTURE: "
            "  2725 aa / ~291 kDa; actin-crosslinking protein; "
            "  N-terminal ABD (actin-binding domain); 24 Ig-like repeat domains; "
            "  Expressed in cardiac and skeletal muscle (Z-disc stabilisation); "
            "VARIANT CLASSES AND PHENOTYPES: "
            "  Truncating variants (frameshift/nonsense/splice): "
            "    Cardiac phenotype: DCM, HCM, or ARVC-like; "
            "    Highest SCD risk of all FLNC variants; "
            "    ICD indicated even with preserved LVEF + NSVT or LGE; "
            "    Mid-wall fibrosis on CMR: high arrhythmia predictor; "
            "  Missense variants: more variable; may cause distal myopathy (skeletal); "
            "CARDIOMYOPATHY PHENOTYPE: "
            "  Multi-phenotype spectrum: HCM, DCM, or ARVC pattern; "
            "  Unusual: same truncating variant can cause different CMP subtypes in family; "
            "  ARVC-like: biventricular fibrofatty replacement; RV + LV involvement; "
            "DISTAL MYOPATHY: "
            "  Myofibrillar myopathy (MFM): aggregate formation in skeletal muscle; "
            "  Cardiac involvement in FLNC myopathy: up to 30-40%; "
            "SCD RISK -- TRUNCATING-SPECIFIC RULE: "
            "  FLNC truncating variant + NSVT or LGE on CMR -> ICD even LVEF normal; "
            "  Do not apply LVEF <35% threshold alone; "
            "TREATMENT: "
            "  ICD: early and liberal in truncating FLNC; "
            "  HF therapy: standard ACEi/ARB/ARNI + beta-blocker; "
            "SURVEILLANCE: "
            "  Annual CMR (LGE + RV/LV assessment); "
            "  Annual echo + Holter; "
            "  CK + skeletal muscle examination"
        ),
        "syndrome": "HCM-DCM-ARVC-Spectrum / Distal-Myopathy-overlap",
        "inheritance": "AD LOF (truncating highest SCD risk)",
        "pathognomonic": "FLNC-truncating + NSVT/LGE -> ICD-regardless-LVEF; multi-phenotype HCM/DCM/ARVC same gene",
        "key_drugs_ci": ["LVEF-35-threshold-insufficient-FLNC-truncating", "competitive-sport-CI-structural"],
        "surveillance": ["annual-CMR-LGE-RV-LV", "annual-echo-Holter", "CK-skeletal-muscle"],
    },
    {
        "gene": "RBM20",
        "protein": (
            "RBM20 -- 10q25.2 AD GOF -- 1227aa -- "
            "RNA-Binding-Motif-20-DCM-Aggressive-Young-R634W-Hotspot-"
            "Biventricular-Fibrosis-Early-ICD-Transplant-"
            "RBM20-OMIM-613172"
        ),
        "locus": "10q25.2",
        "protein_size": (
            "1227 aa / ~145 kDa / 10q25.2 RBM20 RNA-binding motif protein 20 molecular context: "
            "STRUCTURE: "
            "  1227 aa / ~145 kDa; cardiac-specific RNA-binding protein; "
            "  RRM (RNA-recognition motif) domain: RNA binding; "
            "  RRMM domain (aa 634-638): R634W hotspot cluster; nuclear localisation signal; "
            "  Normally nuclear: RBM20 regulates alternative splicing of TTN, CAMK2D, OBSCN; "
            "  GOF pathogenic variants (R634W): mislocalisation to cytoplasm -> splicing dysregulation; "
            "DCM-RBM20 -- AGGRESSIVE YOUNG-ONSET: "
            "  R634W (Arg634Trp): most common pathogenic variant; HOTSPOT; "
            "  R636W, R636Q, R634L: same RRMM cluster; equivalently aggressive; "
            "  YOUNG ONSET: presentation in 2nd-4th decade (20-40yr); "
            "  RAPID PROGRESSION: faster deterioration than other DCM genes; "
            "  BIVENTRICULAR FIBROSIS on CMR: LGE diffuse mid-wall + subepicardial; "
            "  VT/VF: high arrhythmia burden; "
            "  CARDIAC TRANSPLANTATION: RBM20-DCM has highest transplant rate of all DCM genes; "
            "    List for transplant early; "
            "MECHANISM: "
            "  R634W -> cytoplasmic stress granule mislocalisation; "
            "  Loss of splicing regulation -> giant titin isoform -> DCM; "
            "ICD TIMING: "
            "  ICD early: RBM20-DCM arrhythmia risk even at LVEF 45-50%; "
            "  LVEF <45% + R634W variant -> ICD without waiting for LVEF <35%; "
            "PHARMACOLOGY: "
            "  Standard HF therapy: GDMT (ACEi/ARB/ARNI, beta-blocker, MRA, SGLT2i); "
            "  Early transplant referral: list when LVEF <30% despite GDMT; "
            "SURVEILLANCE: "
            "  6-monthly echocardiogram + Holter; "
            "  Annual CMR; "
            "  Transplant centre referral at diagnosis for counselling"
        ),
        "syndrome": "DCM-Aggressive-Young-Onset / Biventricular-Fibrosis",
        "inheritance": "AD GOF (RRMM domain R634W hotspot)",
        "pathognomonic": "R634W-RBM20-DCM-aggressive-young-onset PATHOGNOMONIC; biventricular-fibrosis-CMR; highest-transplant-rate",
        "key_drugs_ci": ["delay-standard-LVEF35-ICD-threshold-not-applicable-RBM20"],
        "surveillance": ["6-monthly-echo-Holter", "annual-CMR", "early-transplant-referral"],
    },
    {
        "gene": "MYH6",
        "protein": (
            "MYH6 -- 14q11.2 AD LOF -- 1939aa -- "
            "alpha-MHC-DCM-ASD-Sick-Sinus-Syndrome-"
            "CRT-Response-Excellent-Atrial-Involvement-"
            "MYH6-OMIM-160710"
        ),
        "locus": "14q11.2",
        "protein_size": (
            "1939 aa / ~224 kDa / 14q11.2 MYH6 alpha-myosin heavy chain molecular context: "
            "STRUCTURE: "
            "  1939 aa / ~224 kDa; sarcomeric thick filament; "
            "  Alpha-MHC isoform: predominant in adult human atria; "
            "  Beta-MHC (MYH7) predominant in adult ventricle; "
            "  MYH6 and MYH7 share 86% amino acid identity; "
            "  LOF variants -> impaired atrial and ventricular contractility; "
            "DCM WITH ATRIAL + CONDUCTION PHENOTYPE: "
            "  DCM: dilated cardiomyopathy; atrial involvement often prominent; "
            "  ASD (atrial septal defect): MYH6 variants associated with congenital ASD; "
            "    ASD + late-onset DCM -> consider MYH6 genetic testing; "
            "  SSS (sick sinus syndrome): SA node dysfunction; sinus bradycardia; "
            "    SSS + DCM + family history -> MYH6 gene panel; "
            "CRT (cardiac resynchronisation therapy): "
            "  MYH6-DCM: EXCELLENT CRT RESPONSE; "
            "  CRT response better than population mean for DCM; "
            "  Consider early CRT-D in MYH6-DCM with LBBB; "
            "ATRIAL INVOLVEMENT: "
            "  AF common: MYH6-DCM with atrial fibrosis; "
            "  SA node involvement: sinus bradycardia, sick sinus syndrome; "
            "  Pacemaker for SSS if symptomatic; "
            "TREATMENT: "
            "  Standard GDMT for DCM (ACEi/ARB/ARNI, beta-blocker, MRA, SGLT2i); "
            "  CRT-D: if LVEF <=35% + LBBB + NYHA II-III; "
            "  Pacemaker or CRT: if SSS symptomatic; "
            "  ASD closure: if haemodynamically significant ASD present; "
            "SURVEILLANCE: "
            "  Annual echo + Holter + ECG; "
            "  Exercise test (SSS chronotropic assessment); "
            "  CMR: LGE for fibrosis quantification; "
            "  Cascade testing: ASD + unexplained DCM in family"
        ),
        "syndrome": "DCM / ASD / Sick-Sinus-Syndrome / CRT-response-excellent",
        "inheritance": "AD LOF",
        "pathognomonic": "ASD + DCM + SSS triad in family -> MYH6; excellent-CRT-response PATHOGNOMONIC for MYH6-DCM",
        "key_drugs_ci": [],
        "surveillance": ["annual-echo-Holter-ECG", "CRT-D-LBBB-LVEF35", "SSS-pacemaker-if-symptomatic"],
    },
    {
        "gene": "TTN",
        "protein": (
            "TTN -- 2q31.2 AD/AR LOF truncating -- 34350aa -- "
            "Titin-DCM-Most-Common-25pct-Truncating-Variants-"
            "Incomplete-Penetrance-HCM-Modifier-PPCM-"
            "TTN-OMIM-188840"
        ),
        "locus": "2q31.2",
        "protein_size": (
            "34350 aa / ~3800 kDa / 2q31.2 TTN Titin molecular context: "
            "STRUCTURE: "
            "  34350 aa / ~3800 kDa; LARGEST HUMAN PROTEIN; "
            "  Sarcomere spanning: Z-disc to M-band; half-sarcomere = one titin molecule; "
            "  I-band (elastic spring): PEVK domain + N2B/N2BA; passive tension; "
            "  A-band: binds myosin thick filament; "
            "  M-band: titin kinase domain; protein quality control; "
            "  Gene structure: 363 exons; extensive alternative splicing; "
            "  PSI-band / A-band truncations: DCM-causing (highest clinical significance); "
            "TTNtv (Titin-truncating variants): "
            "  MOST COMMON DCM GENE: TTNtv in ~25% of familial DCM; "
            "  Also ~2% of population-level carriers (many asymptomatic!); "
            "  Truncating variants (frameshift, nonsense, splice site) -> haploinsufficiency; "
            "  INCOMPLETE PENETRANCE: many TTNtv carriers remain phenotypically normal; "
            "    Penetrance ~20-30% in general population carriers; ~50% in DCM families; "
            "  A-band truncations: higher penetrance than I-band or Z-disc truncations; "
            "SPECIAL CONTEXTS: "
            "  Peripartum cardiomyopathy (PPCM): TTNtv enriched in PPCM (15-20% of PPCM); "
            "    Pregnancy unmasks TTNtv susceptibility; "
            "    PPCM recovery possible even in TTNtv carriers; "
            "  HCM MODIFIER: TTNtv may modify hypertrophic phenotype when co-inherited; "
            "  Exercise-induced DCM: TTNtv may be unmasked by extreme endurance training; "
            "MANAGEMENT: "
            "  Treat phenotype not genotype alone: asymptomatic TTNtv -> monitor only; "
            "  GDMT when DCM phenotype emerges; "
            "  ICD: standard LVEF <35% threshold (unlike LMNA/PLN/RBM20 exceptions); "
            "  PPCM: bromocriptine (prolactin inhibition) + GDMT; "
            "SURVEILLANCE: "
            "  Gene-positive: echo every 3-5yr or at high-risk periods (pregnancy, extreme sport); "
            "  Annual echo if phenotype present; "
            "  Pregnancy counselling: monitor from 2nd trimester to 6 months postpartum"
        ),
        "syndrome": "DCM-Most-Common / PPCM / HCM-Modifier",
        "inheritance": "AD LOF truncating (incomplete penetrance); AR biallelic (severe early-onset)",
        "pathognomonic": "TTNtv = most-common-DCM-gene-25pct; incomplete-penetrance; PPCM-TTNtv-enriched; A-band-truncation-highest-penetrance",
        "key_drugs_ci": [],
        "surveillance": ["gene-positive-echo-3-5yr", "annual-echo-phenotype-present", "PPCM-bromocriptine-GDMT", "pregnancy-counselling"],
    },
]

DEFINITIONS = {
    "HCM": "Hypertrophic cardiomyopathy: LV wall thickness >=15mm (or >=13mm with family history); MYH7 and MYBPC3 most common genes; LVOTO in ~25-30%.",
    "DCM": "Dilated cardiomyopathy: LVEF <50% with LV dilation; TTN most common gene (~25% familial); multiple genetic causes.",
    "LVOTO": "Left ventricular outflow tract obstruction: dynamic obstruction in HCM; SAM (systolic anterior motion) of mitral valve; disopyramide/mavacamten treatment.",
    "Mavacamten": "Allosteric cardiac myosin-II inhibitor; FDA-approved HCM (EXPLORER-HCM); reduces LVOTO + symptoms; hold if LVEF <50%; CYP2C19 drug interactions.",
    "Disopyramide": "Class IA antiarrhythmic; negative inotrope; reduces LVOTO in HCM; monitor QTc; anticholinergic side effects.",
    "LMNA_DCM": "LMNA DCM: AV block pathognomonic; VT storm risk; ICD before LVEF <35%; Wahbi score for ICD timing; EDMD/LGMD overlap.",
    "PLN_R14del": "Phospholamban R14del: Dutch founder variant; DCM + ARVC overlap; ICD mandatory; SERCA2a dysregulation; small 52aa protein.",
    "FLNC_truncating": "FLNC truncating variants: highest SCD risk; ICD even preserved LVEF + NSVT/LGE; multi-phenotype HCM/DCM/ARVC; distal myopathy overlap.",
    "RBM20_R634W": "RBM20 R634W hotspot: aggressive DCM young onset; biventricular fibrosis; highest transplant rate; RRMM domain mislocalisation.",
    "TTNtv": "Titin-truncating variants: most common DCM gene (25% familial); incomplete penetrance; A-band truncations highest penetrance; enriched in PPCM.",
    "PPCM": "Peripartum cardiomyopathy: LV dysfunction last month pregnancy to 5 months postpartum; TTNtv 15-20%; bromocriptine adjunct.",
    "SAM": "Systolic anterior motion of mitral valve: HCM sign of LVOTO; MV leaflet pulled/pushed into LVOT during systole.",
    "ESC_HCM_SCD_score": "ESC 2014 HCM SCD risk calculator: 5-yr SCD risk >=6% -> ICD; inputs: max wall thickness, family SCD, NSVT, abnormal BP response, LA size, LV gradient.",
    "Wahbi_LMNA_score": "LMNA-specific ICD risk score (Wahbi 2019): male sex + NSVT + LVEF <45% + non-missense variant -> early ICD before standard 35% threshold.",
    "LGE": "Late gadolinium enhancement on CMR: fibrosis marker; mid-wall LGE in DCM; subepicardial in PLN/FLNC; diffuse biventricular in RBM20.",
    "CRT": "Cardiac resynchronisation therapy: biventricular pacing for LBBB + LVEF <=35%; MYH6-DCM has excellent CRT response.",
    "GDMT": "Guideline-directed medical therapy for HF: ACE inhibitor/ARB/ARNI + beta-blocker + mineralocorticoid antagonist + SGLT2 inhibitor.",
    "Sarcomere_HCM": "Sarcomeric HCM: caused by mutations in sarcomere proteins (MYH7, MYBPC3, TNNT2, TNNI3, etc.); hypercontractility mechanism (GOF); mavacamten target.",
    "Haploinsufficiency": "LOF mechanism: one allele inactivated -> 50% protein reduction sufficient to cause disease; frameshift/nonsense most common in MYBPC3 HCM.",
    "Dominant_negative": "LOF mechanism: mutant protein actively interferes with wild-type protein function; PLN R14del = dominant-negative excess SERCA2a inhibition.",
}


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def generate_overview() -> dict:
    cohort_stats = []
    for i, g in enumerate(ATLAS_GENES):
        rng = _rng(SEED_BASE + i)
        n = 40
        mean_age = round(max(1.0, rng.gauss(
            25.0 if g["gene"] == "RBM20" else
            35.0 if g["gene"] in ("PLN", "FLNC") else
            55.0 if g["gene"] == "MYBPC3" else
            48.0 if g["gene"] == "TTN" else
            45.0 if g["gene"] in ("LMNA", "MYH6") else
            42.0, 9.0)), 1)
        pct_female = round(rng.uniform(
            55, 70) if g["gene"] == "TTN" else
            rng.uniform(40, 55), 1)
        cohort_stats.append({
            "gene": g["gene"],
            "n": n,
            "mean_age": mean_age,
            "pct_female": pct_female,
            "syndrome": g["syndrome"].split("/")[0].strip(),
            "inheritance": g["inheritance"].split(";")[0].strip(),
        })

    return {
        "atlas_name": "Hereditary-Cardiomyopathy-Atlas",
        "atlas": "Hereditary-Cardiomyopathy-Atlas",
        "subtitle": "Complete 8-Gene Reference -- MYH7 / MYBPC3 / LMNA / PLN / FLNC / RBM20 / MYH6 / TTN",
        "total_patients": 320,
        "gene_count": 8,
        "seeds": "3750-3757",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "cohort_stats": cohort_stats,
        "clinical_categories": {
            "hcm_genes": ["MYH7", "MYBPC3"],
            "dcm_genes": ["LMNA", "PLN", "RBM20", "MYH6", "TTN"],
            "mixed_hcm_dcm_arvc": ["FLNC"],
            "early_icd_non_standard": ["LMNA", "PLN", "RBM20", "FLNC"],
            "highest_transplant_rate": ["RBM20"],
            "ppcm_enriched": ["TTN"],
        },
        "key_clinical_rules": [
            "MYH7: R403Q/R453C = HCM PATHOGNOMONIC hotspot; mavacamten/disopyramide for LVOTO; hold mavacamten if LVEF <50%",
            "MYBPC3: late-onset HCM penetrance complete-by-70yr -- do NOT reassure young gene+ adults; rescan every 5yr",
            "LMNA: AV block preceding DCM = LMNA PATHOGNOMONIC; ICD before LVEF <35% (use Wahbi score); VT storm risk at preserved EF",
            "PLN: R14del Dutch founder DCM+ARVC overlap PATHOGNOMONIC; ICD MANDATORY R14del with NSVT/syncope/LGE",
            "FLNC: truncating variant + NSVT/LGE = ICD even preserved LVEF; multi-phenotype HCM/DCM/ARVC same gene",
            "RBM20: R634W = aggressive DCM young onset PATHOGNOMONIC; biventricular fibrosis; highest transplant rate -- list early",
            "MYH6: ASD + DCM + SSS triad -> MYH6; excellent CRT response in MYH6-DCM (CRT-D preferred)",
            "TTN: most common DCM gene (25%); incomplete penetrance -- treat phenotype not genotype; TTNtv enriched in PPCM",
        ],
    }


def generate_breakdown() -> dict:
    gene_data = {}
    for i, g in enumerate(ATLAS_GENES):
        rng = _rng(SEED_BASE + i)
        n = 40
        mean_age = round(max(1.0, rng.gauss(
            25.0 if g["gene"] == "RBM20" else
            35.0 if g["gene"] in ("PLN", "FLNC") else
            55.0 if g["gene"] == "MYBPC3" else
            48.0 if g["gene"] == "TTN" else
            45.0 if g["gene"] in ("LMNA", "MYH6") else
            42.0, 9.0)), 1)
        pct_f = round(rng.uniform(
            55, 70) if g["gene"] == "TTN" else
            rng.uniform(40, 55), 1)
        icd_n = rng.randint(18, 32) if g["gene"] in ("LMNA", "PLN", "RBM20", "FLNC") else rng.randint(8, 20)
        transplant_n = rng.randint(8, 16) if g["gene"] == "RBM20" else rng.randint(1, 6)
        hf_admission_n = rng.randint(12, 28)
        gene_data[g["gene"]] = {
            "n": n,
            "mean_age_dx": mean_age,
            "pct_female": pct_f,
            "icd_implanted_n": icd_n,
            "transplant_n": transplant_n,
            "hf_admission_n": hf_admission_n,
            "syndrome": g["syndrome"].split("/")[0].strip(),
            "inheritance": g["inheritance"].split(";")[0].strip(),
            "pathognomonic": g["pathognomonic"],
        }

    return {
        "atlas": "Hereditary-Cardiomyopathy-Atlas",
        "total_patients": 320,
        "seeds": "3750-3757",
        "gene_data": gene_data,
        "atlas_genes": ATLAS_GENES,
        "summary": {
            "hcm_primary_genes": ["MYH7", "MYBPC3"],
            "dcm_primary_genes": ["LMNA", "PLN", "RBM20", "MYH6", "TTN"],
            "arvc_overlap_genes": ["PLN", "FLNC"],
            "mavacamten_applicable": ["MYH7", "MYBPC3"],
            "early_icd_before_lvef35": ["LMNA", "PLN", "RBM20", "FLNC"],
            "highest_transplant_rate": ["RBM20"],
            "ppcm_associated": ["TTN"],
            "excellent_crt_response": ["MYH6"],
            "skeletal_muscle_overlap": ["LMNA", "FLNC"],
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Cardiomyopathy-Atlas",
        "definitions": DEFINITIONS,
        "critical_clinical_rules": {
            "MYH7-R403Q-HCM-PATHOGNOMONIC": "MYH7 R403Q/R453C = high-penetrance HCM PATHOGNOMONIC variants -- confirm with echo; mavacamten/disopyramide for LVOTO; hold mavacamten if LVEF drops to <50%",
            "MYBPC3-LATE-ONSET-PENETRANCE": "MYBPC3 HCM late-onset: penetrance near-complete by age 70yr -- do NOT reassure young gene-positive adults; rescan every 5yr; AF common (20-30%)",
            "LMNA-AV-BLOCK-PATHOGNOMONIC": "AV block preceding DCM = LMNA PATHOGNOMONIC -- test LMNA immediately; ICD based on Wahbi score, not LVEF <35% threshold alone; VT storm at preserved EF",
            "LMNA-ICD-WAHBI-RULE": "LMNA ICD: use Wahbi 2019 score (male + NSVT + LVEF <45% + non-missense -> ICD) -- DO NOT wait for LVEF <35%; VT storm can kill at preserved function",
            "PLN-R14del-PATHOGNOMONIC": "PLN R14del Dutch founder DCM+ARVC overlap PATHOGNOMONIC -- ICD MANDATORY with NSVT/syncope/LGE; CMR annually; high transplant rate",
            "FLNC-TRUNCATING-ICD-RULE": "FLNC truncating variant: ICD even with preserved LVEF if NSVT or CMR-LGE present -- LVEF 35% threshold insufficient; multi-phenotype spectrum",
            "RBM20-R634W-AGGRESSIVE": "RBM20 R634W = aggressive DCM young onset PATHOGNOMONIC -- biventricular fibrosis; highest transplant rate all DCM genes; refer transplant centre at diagnosis",
            "MYH6-ASD-DCM-SSS-TRIAD": "MYH6: ASD + DCM + SSS in family -> gene panel include MYH6; CRT response excellent -- prefer CRT-D over ICD alone if LBBB + LVEF <=35%",
            "TTN-INCOMPLETE-PENETRANCE": "TTNtv: 25% familial DCM but only ~20-30% population carriers manifest disease -- treat phenotype not genotype alone; PPCM TTNtv enriched (15-20%)",
            "TTN-PPCM-BROMOCRIPTINE": "PPCM + TTNtv: bromocriptine (prolactin inhibitor) + GDMT; LV recovery possible even in TTNtv carriers; monitor 2nd trimester to 6 months postpartum",
            "MAVACAMTEN-LVEF-MONITORING": "Mavacamten: echo monitoring mandatory (weekly until stable, then monthly); HOLD if LVEF <50%; CYP2C19 inhibitors (omeprazole, fluoxetine) increase exposure -- dose adjust",
            "HCM-SCD-ESC-SCORE": "HCM SCD risk: use ESC 2014 calculator (5-yr risk >=6% -> ICD) -- genotype alone (R403Q) does NOT determine ICD; include wall thickness, family SCD, NSVT, BP response",
            "SARCOMERE-SPORT-RESTRICTION": "HCM competitive sport: ABSOLUTE CI if significant LV hypertrophy or LVOTO or arrhythmia -- exercise-induced SCD in HCM athletes (MYH7/MYBPC3)",
            "ALL-CMP-GENES-CASCADE": "Cascade genetic testing mandatory all 1st-degree relatives for all cardiomyopathy genes -- 50% inheritance risk; pre-symptomatic identification allows surveillance before phenotype",
        },
    }


if __name__ == "__main__":
    import json
    print("=== Hereditary Cardiomyopathy Atlas ===")
    ov = generate_overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print()
    br = generate_breakdown()
    for gene, d in br["gene_data"].items():
        print(f"  {gene}: n={d['n']}, age_dx={d['mean_age_dx']}yr, ICD={d['icd_implanted_n']}/40, transplant={d['transplant_n']}/40")
    print()
    df = generate_definitions()
    print(f"Definitions: {len(df['definitions'])} terms")
    print(f"Clinical rules: {len(df['critical_clinical_rules'])} rules")
    print("SCRIPT OK")
