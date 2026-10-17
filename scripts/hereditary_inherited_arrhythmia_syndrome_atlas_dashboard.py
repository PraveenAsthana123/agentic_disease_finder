#!/usr/bin/env python3
"""Hereditary-Inherited-Arrhythmia-Syndrome-Atlas -- Complete 8-Gene Reference
Cardiac channelopathies and arrhythmogenic cardiomyopathies:
KCNQ1 (LQT1/JLNS/SQT2), KCNH2 (LQT2/SQT1), SCN5A (LQT3/Brugada/SSS/PCCD),
RYR2 (CPVT1/ARVC2), PKP2 (ARVC1-most-common), DSP (ARVC8/cardiocutaneous),
CACNA1C (LQT8/Timothy), HCN4 (SSS2/hereditary-bradycardia).

KCNQ1   (Potassium voltage-gated channel subfamily Q member 1; 676aa; 11p15.5; AD/AR LOF;
          Romano-Ward syndrome (AD LOF): LQT1; swim/exertion trigger PATHOGNOMONIC;
          Jervell-Lange-Nielsen syndrome (AR biallelic LOF): profound congenital deafness +
          severe LQT PATHOGNOMONIC — ALWAYS screen LQT for deafness;
          SQT2: short QT, gain-of-function rare;
          seed SEED_BASE+0).
KCNH2   (Potassium voltage-gated channel subfamily H member 2 / hERG; 1159aa; 7q36.1; AD LOF;
          LQT2: sudden auditory arousal trigger PATHOGNOMONIC (alarm/phone/doorbell);
          SQT1: short QT gain-of-function;
          MOST drug-induced LQT syndromes block hERG: avoid ALL QT-prolonging drugs;
          seed SEED_BASE+1).
SCN5A   (Sodium voltage-gated channel alpha subunit 5; 2016aa; 3p22.2; AD LOF/GOF;
          LQT3 (LOF): malignant nocturnal events; bradycardia-triggered; ECG sinus-pause + LQT;
          Brugada syndrome (LOF): fever trigger PATHOGNOMONIC; coved-type ST V1-V2 PATHOGNOMONIC;
          SSS (sinus node dysfunction); PCCD (progressive cardiac conduction disease);
          SCN5A alone = 5 distinct phenotypes — overlap syndromes common;
          seed SEED_BASE+2).
RYR2    (Ryanodine receptor 2; 4967aa; 1q43; AD GOF;
          CPVT1: exercise/emotion-triggered bidirectional ventricular tachycardia PATHOGNOMONIC;
          Normal resting ECG and echo — diagnosis missed without exercise stress test;
          Beta-blocker mandatory (nadolol preferred); flecainide adjunct;
          ARVC2: RYR2 GOF variant; overlap with structural ARVC;
          seed SEED_BASE+3).
PKP2    (Plakophilin-2; 897aa; 12p11.21; AD LOF;
          ARVC1: most common ARVC gene (11-43% ARVC); desmosomal protein;
          Epsilon wave PATHOGNOMONIC (terminal activation delay V1-V3);
          Task Force Criteria 2010: major=coved ARVC MRI, minor=epsilon wave;
          RV fatty replacement + fibrosis on MRI; avoid competitive sport MANDATORY;
          seed SEED_BASE+4).
DSP     (Desmoplakin; 2871aa; 6p24.3; AD LOF;
          ARVC8: biventricular / LV-dominant ARVC — LEFT-dominant variant PATHOGNOMONIC;
          Cardiocutaneous syndromes: wooly hair + palmoplantar keratoderma PATHOGNOMONIC Carvajal;
          LMNA-like malignant arrhythmias; DSP = highest sudden death risk among ARVC genes;
          DSP variants common in South Asian populations;
          seed SEED_BASE+5).
CACNA1C (Calcium voltage-gated channel subunit alpha1 C; 2221aa; 12p13.33; AD GOF;
          LQT8 / Timothy syndrome: EXTREMELY RARE (< 20 cases worldwide);
          2:1 AV block PATHOGNOMONIC; cutaneous syndactyly PATHOGNOMONIC;
          Multisystem: autism spectrum, hypoglycaemia, immune deficiency, HCM;
          De novo GOF G406R (exon 8A) accounts for > 90% Timothy syndrome;
          seed SEED_BASE+6).
HCN4    (Hyperpolarization activated cyclic nucleotide gated potassium channel 4; 1208aa; 15q24.1; AD LOF;
          SSS2: hereditary sick sinus syndrome type 2;
          Symptomatic bradycardia / chronotropic incompetence → pacemaker mandatory;
          Isolated non-compaction cardiomyopathy (LVNC) overlap;
          Brugada-like ST in some variants; ivermectin ABSOLUTE CI (HCN channel blocker);
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3742-3749)
"""
import random

SEED_BASE = 3742

ATLAS_GENES = [
    {
        "gene": "KCNQ1",
        "protein": (
            "KCNQ1 -- 11p15.5 AD/AR LOF -- 676aa -- "
            "IKs-potassium-channel-Kv7.1-LQT1-Romano-Ward-JLNS-SQT2-"
            "swim-exertion-trigger-PATHOGNOMONIC-deafness-biallelic-JLNS-"
            "KCNQ1-OMIM-607542"
        ),
        "locus": "11p15.5",
        "protein_size": (
            "676 aa / ~75 kDa / 11p15.5 KCNQ1 Kv7.1 alpha-subunit molecular context: "
            "STRUCTURE: "
            "  676 aa / ~75 kDa; voltage-gated K+ channel alpha-subunit; "
            "  S1-S6 transmembrane segments; S4 voltage sensor; S5-S6 pore-forming; "
            "  Assembles with KCNE1 (minK beta-subunit) → IKs (slow delayed rectifier) current; "
            "  IKs = critical repolarisation reserve in heart; "
            "  Biallelic LOF → severe IKs loss + deafness (endolymph homeostasis); "
            "LQT1 (Romano-Ward, AD heterozygous LOF): "
            "  Most common congenital LQTS (30-35% of all LQT); "
            "  SWIM / EXERTION as arrhythmia trigger PATHOGNOMONIC for LQT1; "
            "  Reason: IKs is adrenergically upregulated; exercise → sympathetic surge → "
            "    IKs needed for rapid repolarisation; LOF → QT fails to shorten → VF; "
            "  T-wave morphology: broad-based T-waves in LQT1; "
            "  Beta-blocker HIGHEST efficacy in LQT1 (vs LQT2/LQT3); "
            "  Penetrance ~30-50%; QTc may be normal at rest in gene+ patients; "
            "JLNS (Jervell-Lange-Nielsen, AR biallelic LOF): "
            "  Profound congenital bilateral sensorineural deafness PATHOGNOMONIC; "
            "  Severe QT prolongation (QTc often >500ms); first event often at age 3-5yr; "
            "  ALWAYS screen LQT paediatric patient for deafness (and vice versa); "
            "  Biallelic = complete IKs loss; highest LQTS mortality without ICD; "
            "  Cochlear implant + ICD both required; beta-blocker alone insufficient; "
            "SQT2 (rare GOF KCNQ1): "
            "  Short QT syndrome; increased IKs; QTc <340ms; AF + VF risk; "
            "SLC involvement: not applicable (purely cardiac/cochlear phenotype); "
            "PHARMACOLOGY: "
            "  Beta-blocker (nadolol/propranolol): FIRST-LINE — highest efficacy in LQT1; "
            "  ICD: JLNS, multiple breakthroughs on beta-blocker, cardiac arrest survivors; "
            "  Left cardiac sympathetic denervation (LCSD): refractory LQT1 adjunct; "
            "  Class IA (quinidine, procainamide): ABSOLUTE CI — prolong QT further; "
            "  Class IC (flecainide, propafenone): RELATIVE CI in LQT (not Brugada); "
            "  Class III (amiodarone, sotalol): sotalol = ABSOLUTE CI (QT-prolonging); "
            "SURVEILLANCE: "
            "  Annual 12-lead ECG + Holter; exercise test every 2-3 yr; "
            "  All first-degree relatives cascade-tested; swim avoidance if symptomatic; "
            "  Paediatric onset: cochlear screening if QTc > 460ms"
        ),
        "syndrome": "LQT1-Romano-Ward / JLNS-deafness-PATHOGNOMONIC / SQT2",
        "inheritance": "AD (LQT1); AR biallelic (JLNS); AD GOF (SQT2)",
        "pathognomonic": "swim/exertion trigger LQT1; deafness + severe-LQT = JLNS biallelic KCNQ1",
        "key_drugs_ci": ["quinidine-ABSOLUTE-CI", "sotalol-ABSOLUTE-CI", "all-QT-prolonging-ABSOLUTE-CI"],
        "surveillance": ["annual-ECG-Holter", "exercise-test-2-3yr", "cascade-screen-1st-degree"],
    },
    {
        "gene": "KCNH2",
        "protein": (
            "KCNH2 -- 7q36.1 AD LOF -- 1159aa -- "
            "IKr-hERG-potassium-channel-Kv11.1-LQT2-auditory-trigger-PATHOGNOMONIC-"
            "SQT1-drug-interactions-hERG-block-"
            "KCNH2-OMIM-152427"
        ),
        "locus": "7q36.1",
        "protein_size": (
            "1159 aa / ~127 kDa / 7q36.1 KCNH2 hERG Kv11.1 alpha-subunit molecular context: "
            "STRUCTURE: "
            "  1159 aa / ~127 kDa; voltage-gated K+ channel alpha-subunit; "
            "  Unique PAS domain (N-terminal): involved in channel gating and drug binding; "
            "  CNBHD (C-terminal): cyclic nucleotide binding homology domain; "
            "  Forms IKr (rapid delayed rectifier): critical for phase 3 repolarisation; "
            "  hERG channel = most common target of drug-induced QT prolongation; "
            "LQT2 (AD heterozygous LOF): "
            "  Second most common LQTS (25-30% of LQTS); "
            "  SUDDEN AUDITORY AROUSAL as trigger PATHOGNOMONIC for LQT2; "
            "    Phone ring, alarm clock, doorbell → sympathetic arousal → IKr demand; "
            "    LOF → QT fails to shorten on sudden arousal → VF/VT; "
            "  T-wave morphology: notched bifid T-waves in LQT2; "
            "  Beta-blocker: effective but less than LQT1; "
            "  Mexiletine: adjunct (partial LQT2-specific benefit); "
            "  Potassium supplementation: serum K+ > 4.5 mEq/L reduces IKr dependence; "
            "  ICD: appropriate in symptomatic/cardiac arrest LQT2; "
            "SQT1 (rare GOF KCNH2): "
            "  Short QT; N588K most common variant; QTc <320ms; prominent tall peaked T-waves; "
            "  Quinidine: preferred (IKr block) for SQT1; "
            "DRUG-INDUCED LQT: "
            "  hERG block = universal mechanism of drug-induced QTc prolongation; "
            "  CredibleMeds categories: Known/Conditional/Possible/Absent risk; "
            "  Antibiotics: azithromycin, fluoroquinolones, clarithromycin — avoid in LQT2; "
            "  Antipsychotics: haloperidol, quetiapine, ziprasidone — ABSOLUTE CI in LQT2; "
            "  Antiemetics: ondansetron, domperidone — avoid in LQT2; "
            "  Antifungals: fluconazole, voriconazole — avoid in LQT2; "
            "  SSRI: citalopram/escitalopram > 20mg/day — QT-prolonging dose; "
            "  Methadone: ABSOLUTE CI in LQT; "
            "SURVEILLANCE: "
            "  Annual ECG + Holter; alarm management (wake alarm = musical tone not beeping); "
            "  Cascade screen; potassium check electrolytes quarterly; "
            "  PREGNANCY: LQT2 women have HIGHEST postpartum arrhythmia risk (6 weeks postpartum)"
        ),
        "syndrome": "LQT2-Romano-Ward / SQT1",
        "inheritance": "AD LOF (LQT2); AD GOF (SQT1)",
        "pathognomonic": "sudden auditory arousal trigger (alarm/phone/doorbell) = LQT2 PATHOGNOMONIC",
        "key_drugs_ci": ["haloperidol-ABSOLUTE-CI", "azithromycin-AVOID", "ondansetron-AVOID", "methadone-ABSOLUTE-CI", "sotalol-ABSOLUTE-CI"],
        "surveillance": ["annual-ECG-Holter", "serum-K+-quarterly", "alarm-tone-modification", "postpartum-surveillance-6wk"],
    },
    {
        "gene": "SCN5A",
        "protein": (
            "SCN5A -- 3p22.2 AD LOF/GOF -- 2016aa -- "
            "Nav1.5-cardiac-sodium-channel-LQT3-Brugada-SSS-PCCD-"
            "fever-trigger-Brugada-PATHOGNOMONIC-coved-ST-V1-V2-PATHOGNOMONIC-"
            "SCN5A-OMIM-600163"
        ),
        "locus": "3p22.2",
        "protein_size": (
            "2016 aa / ~220 kDa / 3p22.2 SCN5A Nav1.5 alpha-subunit molecular context: "
            "STRUCTURE: "
            "  2016 aa / ~220 kDa; cardiac-specific Na+ channel alpha-subunit; "
            "  4 homologous domains (DI-DIV), each with 6 transmembrane segments; "
            "  S4 segments = voltage sensors; S5-S6 = pore-forming; "
            "  Beta-subunits (SCN1B-4B): modulate gating, surface expression; "
            "  Responsible for the rapid upstroke of the cardiac action potential (phase 0); "
            "FIVE DISTINCT SCN5A PHENOTYPES: "
            "  (1) LQT3 (late Na+ current GOF): nocturnal bradycardia-triggered events; "
            "      PATHOGNOMONIC: LQT3 events occur during sleep/rest (vs LQT1/LQT2 exertion); "
            "      Mexiletine: primary treatment (blocks late INa); ranolazine alternative; "
            "      Beta-blockers: less effective than in LQT1/LQT2; "
            "  (2) Brugada syndrome (Nav1.5 LOF): "
            "      Coved-type ST elevation V1-V3 PATHOGNOMONIC (Type 1 Brugada pattern); "
            "      Fever trigger PATHOGNOMONIC — fever accelerates channel inactivation; "
            "      Emergency: give antipyretics IMMEDIATELY in febrile BrS patients; "
            "      Sodium channel blockers ABSOLUTE CI (ajmaline, flecainide diagnostic challenge only); "
            "      Quinidine: only oral drug with evidence in symptomatic BrS; "
            "      ICD: indicated in symptomatic BrS (syncope/VF); "
            "      Male predominance (8:1); nocturnal events; "
            "  (3) SSS (sinus node dysfunction LOF): bradycardia, chronotropic incompetence; "
            "  (4) PCCD (progressive cardiac conduction disease): His-Purkinje fibrosis; "
            "      BBB → AV block → pacemaker required; "
            "  (5) DILATED CARDIOMYOPATHY: isolated/overlap with conduction disease; "
            "SCN5A OVERLAP SYNDROMES: "
            "  Single variant can cause BrS + LQT3 + SSS + PCCD (overlap syndrome); "
            "  Always assess for all 5 phenotypes when SCN5A variant found; "
            "PHARMACOLOGY: "
            "  LQT3: mexiletine FIRST-LINE; nadolol adjunct; ICD if breakthrough; "
            "  BrS: ALL Na-channel blockers ABSOLUTE CI for treatment (ajmaline/flecainide = diagnostic ONLY); "
            "    Quinidine (Ito blockade): only oral drug with evidence; "
            "    ICD: symptomatic BrS standard; epicardial VT ablation adjunct; "
            "  FEVER: antipyretics + hospitalization mandatory in BrS; "
            "  Avoid: propafenone, flecainide, encainide, lidocaine (can precipitate BrS storm)"
        ),
        "syndrome": "LQT3 / Brugada-syndrome / SSS / PCCD / DCM",
        "inheritance": "AD LOF (Brugada/SSS/PCCD); AD GOF (LQT3); AD (DCM)",
        "pathognomonic": "coved-ST-V1-V3 = Brugada Type1 PATHOGNOMONIC; fever = Brugada trigger PATHOGNOMONIC; sleep/rest = LQT3 events PATHOGNOMONIC",
        "key_drugs_ci": ["flecainide-ABSOLUTE-CI-Brugada", "propafenone-ABSOLUTE-CI-Brugada", "ajmaline-diagnostic-ONLY-not-treatment", "all-Na-channel-blockers-CI-Brugada"],
        "surveillance": ["fever-protocol-hospitalization", "annual-ECG-Holter", "ICD-symptomatic-Brugada"],
    },
    {
        "gene": "RYR2",
        "protein": (
            "RYR2 -- 1q43 AD GOF -- 4967aa -- "
            "cardiac-ryanodine-receptor-CPVT1-bidirectional-VT-PATHOGNOMONIC-"
            "exercise-emotion-trigger-normal-resting-ECG-PATHOGNOMONIC-"
            "RYR2-OMIM-180902"
        ),
        "locus": "1q43",
        "protein_size": (
            "4967 aa / ~560 kDa / 1q43 RYR2 cardiac ryanodine receptor molecular context: "
            "STRUCTURE: "
            "  4967 aa / ~560 kDa; largest human protein; "
            "  Homotetrameric channel: 4 x 560 kDa = ~2.2 MDa complex; "
            "  Regulates sarcoplasmic reticulum Ca2+ release in cardiomyocytes; "
            "  N-terminal, central, and C-terminal domains; hot-spot mutation clusters; "
            "  Calsequestrin (CASQ2), calmodulin, FKBP12.6 regulate channel; "
            "CPVT1 (catecholaminergic polymorphic VT): "
            "  EXERCISE or EMOTIONAL STRESS as trigger (NOT resting): PATHOGNOMONIC; "
            "  NORMAL resting ECG and normal echocardiogram: PATHOGNOMONIC (misdiagnosis risk); "
            "    Key differentiator from structural heart disease; "
            "  BIDIRECTIONAL VENTRICULAR TACHYCARDIA on exercise ECG: PATHOGNOMONIC; "
            "    Beat-to-beat 180° rotation of QRS axis during tachycardia; "
            "  Mechanism: adrenergic stimulation → hyperactive RYR2 → SR Ca2+ leak → "
            "    triggered activity → bidirectional VT → VF; "
            "  Exercise stress test: MANDATORY for diagnosis (resting ECG normal); "
            "    Exercise-induced VPB, couplets, non-sustained VT → full diagnostic; "
            "TREATMENT: "
            "  Beta-blocker (nadolol PREFERRED): FIRST-LINE; "
            "    Nadolol 1-2 mg/kg/day: superior to metoprolol in CPVT (less breakthrough); "
            "  Flecainide: ADJUNCT therapy; reduces SR Ca2+ spark frequency; "
            "    Flecainide + nadolol = combination standard of care; "
            "  ICD: cardiac arrest survivors; breakthrough VF on beta-blocker; "
            "    ICD NOTE: shocks can trigger adrenergic storm → VF → more shocks (storm); "
            "    Programme ICD with ATP (anti-tachycardia pacing) to avoid inappropriate shocks; "
            "  Left cardiac sympathetic denervation (LCSD): adjunct for refractory CPVT; "
            "ARVC2: "
            "  Subset of RYR2 variants → ARVC-like structural changes; "
            "  Overlap CPVT1 + structural ARVC phenotype; "
            "ABSOLUTE CONTRAINDICATIONS: "
            "  Isoproterenol (provocation): ABSOLUTE CI — can precipitate VF storm; "
            "  High-intensity competitive sport: ABSOLUTE CI in CPVT; "
            "SURVEILLANCE: "
            "  Annual exercise stress test; Holter 24h; "
            "  All first-degree relatives: cascade genetic testing + exercise ECG"
        ),
        "syndrome": "CPVT1 / ARVC2",
        "inheritance": "AD GOF",
        "pathognomonic": "bidirectional-VT-on-exercise PATHOGNOMONIC; normal-resting-ECG-echo + exercise-VT = CPVT PATHOGNOMONIC",
        "key_drugs_ci": ["isoproterenol-ABSOLUTE-CI-provocation", "competitive-sport-ABSOLUTE-CI"],
        "surveillance": ["annual-exercise-stress-test", "Holter-24h", "cascade-screen-exercise-ECG"],
    },
    {
        "gene": "PKP2",
        "protein": (
            "PKP2 -- 12p11.21 AD LOF -- 897aa -- "
            "Plakophilin-2-desmosomal-ARVC1-most-common-11-43pct-"
            "epsilon-wave-PATHOGNOMONIC-RV-fat-fibrosis-MRI-"
            "PKP2-OMIM-602861"
        ),
        "locus": "12p11.21",
        "protein_size": (
            "897 aa / ~97 kDa / 12p11.21 PKP2 Plakophilin-2 molecular context: "
            "STRUCTURE: "
            "  897 aa / ~97 kDa; desmosomal armadillo-repeat protein; "
            "  N-terminal head domain; armadillo-repeat domain (ARM repeats); "
            "  Component of desmosome: PKP2 links desmin cytoskeleton to desmosomal cadherins; "
            "  Expressed in cardiac intercalated discs and skin; "
            "  PKP2 LOF → desmosome disruption → cardiomyocyte death → fibrofatty replacement; "
            "ARVC1 (arrhythmogenic right ventricular cardiomyopathy type 1): "
            "  Most common ARVC gene (11-43% of ARVC cases); "
            "  Right ventricular predominance (vs DSP which is biventricular); "
            "  EPSILON WAVE PATHOGNOMONIC: terminal activation delay in V1-V3 on ECG; "
            "    Represents delayed depolarisation of residual cardiomyocytes in fibrofatty RV; "
            "    Sensitivity only 30%: absence does NOT exclude ARVC; "
            "  ARVC Task Force Criteria 2010: "
            "    Major structural: RV dilation + WMAs; RV fibrofatty replacement on biopsy; "
            "    Major ECG: epsilon wave; T-wave inversion V1-V3 (in absence of RBBB); "
            "    Major functional: exercise-inducible VT with LBBB morphology; "
            "  MRI: RV dilation, wall motion abnormality, intramyocardial fat (controversial); "
            "TRIGGERS / RISK FACTORS: "
            "  COMPETITIVE SPORT: ABSOLUTE CI — exercise accelerates fibrofatty progression; "
            "  PKP2 penetrance = incomplete; athletes at highest risk; "
            "  SCD in young athletes: PKP2 = leading genetic cause; "
            "TREATMENT: "
            "  Beta-blocker: first-line for ventricular arrhythmia suppression; "
            "  Antiarrhythmic: amiodarone or sotalol (note: sotalol CI in LQT but used in ARVC); "
            "  ICD: standard for ARVC with sustained VT/VF/syncope; "
            "  Catheter ablation: adjunct for recurrent VT (endocardial + epicardial approach); "
            "  Heart transplant: end-stage biventricular failure; "
            "SURVEILLANCE: "
            "  Annual ECG, Holter, echocardiogram, cardiac MRI; "
            "  Cascade testing all first-degree relatives; "
            "  Avoid dehydration (concentrates RV arrhythmia risk)"
        ),
        "syndrome": "ARVC1 / RV-dominant-arrhythmogenic-cardiomyopathy",
        "inheritance": "AD LOF (incomplete penetrance)",
        "pathognomonic": "epsilon wave V1-V3 PATHOGNOMONIC (but only 30% sensitivity); RV fat fibrosis MRI; LBBB-morphology VT",
        "key_drugs_ci": ["competitive-sport-ABSOLUTE-CI", "exercise-avoidance-mandatory"],
        "surveillance": ["annual-ECG-Holter-echo-MRI", "cascade-screen", "sport-restriction-MANDATORY"],
    },
    {
        "gene": "DSP",
        "protein": (
            "DSP -- 6p24.3 AD LOF -- 2871aa -- "
            "Desmoplakin-ARVC8-biventricular-LV-dominant-PATHOGNOMONIC-"
            "cardiocutaneous-wooly-hair-keratoderma-Carvajal-PATHOGNOMONIC-"
            "DSP-OMIM-125647"
        ),
        "locus": "6p24.3",
        "protein_size": (
            "2871 aa / ~332 kDa / 6p24.3 DSP Desmoplakin molecular context: "
            "STRUCTURE: "
            "  2871 aa / ~332 kDa; largest desmosomal protein; "
            "  N-terminal plakin domain (interacts with junction proteins PKP2, JUP); "
            "  Central coiled-coil rod domain (forms antiparallel dimer); "
            "  C-terminal plakin repeat domains (PRD A, B, C): bind desmin intermediate filaments; "
            "  Bridges inner desmosomal plaque to cytoskeleton; "
            "  Cardiac + epithelial expression (skin, GI tract); "
            "ARVC8 (DSP-related cardiomyopathy): "
            "  BIVENTRICULAR / LEFT-VENTRICULAR DOMINANT: PATHOGNOMONIC for DSP vs PKP2; "
            "    PKP2 = RV dominant; DSP = LV dominant or biventricular; "
            "    LV involvement at presentation differentiates DSP from classic ARVC; "
            "  Late gadolinium enhancement (LGE) on CMR: subepicardial LV inferior/lateral PATHOGNOMONIC; "
            "  HIGHEST SCD risk among desmosomal ARVC genes; "
            "  Aggressive arrhythmias despite relatively preserved LV function; "
            "CARDIOCUTANEOUS SYNDROMES: "
            "  Carvajal syndrome (AR biallelic DSP LOF): "
            "    WOOLY HAIR + PALMOPLANTAR KERATODERMA + dilated cardiomyopathy PATHOGNOMONIC; "
            "    Presentation in childhood; "
            "  Naxos disease-like overlap (SOME DSP variants): skin + hair + heart; "
            "  AD heterozygous DSP: pure cardiac (no skin); "
            "CLINICAL TRIAD for DSP SUSPICION: "
            "  Young patient + unexplained VT with LV involvement + family history of SCD; "
            "  NOT typical ARVC presentation — misdiagnosed as myocarditis or idiopathic VT; "
            "TREATMENT: "
            "  ICD: EARLY and liberal indication given high SCD risk; "
            "  Beta-blocker + amiodarone: adjunct; "
            "  Heart transplant: progression to end-stage CMP; "
            "  SPORT RESTRICTION: same as PKP2 — competitive sport ABSOLUTE CI; "
            "SURVEILLANCE: "
            "  Annual CMR (specifically assess LGE); annual ECG + Holter; "
            "  Skin examination (cardiocutaneous syndromes); "
            "  CASCADE in family: all 1st-degree relatives"
        ),
        "syndrome": "ARVC8-LV-dominant / Carvajal-cardiocutaneous-syndrome",
        "inheritance": "AD LOF (ARVC8); AR biallelic (Carvajal)",
        "pathognomonic": "LV-dominant-ARVC = DSP PATHOGNOMONIC (vs RV-dominant PKP2); wooly-hair+keratoderma+DCM = Carvajal PATHOGNOMONIC",
        "key_drugs_ci": ["competitive-sport-ABSOLUTE-CI"],
        "surveillance": ["annual-CMR-LGE", "annual-ECG-Holter", "cascade-screen", "sport-restriction-MANDATORY"],
    },
    {
        "gene": "CACNA1C",
        "protein": (
            "CACNA1C -- 12p13.33 AD GOF -- 2221aa -- "
            "Cav1.2-L-type-Ca-channel-LQT8-Timothy-syndrome-EXTREMELY-RARE-"
            "2:1-AV-block-PATHOGNOMONIC-syndactyly-PATHOGNOMONIC-autism-HCM-"
            "CACNA1C-OMIM-114205"
        ),
        "locus": "12p13.33",
        "protein_size": (
            "2221 aa / ~250 kDa / 12p13.33 CACNA1C Cav1.2 L-type calcium channel molecular context: "
            "STRUCTURE: "
            "  2221 aa / ~250 kDa; L-type voltage-gated Ca2+ channel alpha-1 subunit; "
            "  Four homologous domains (DI-DIV), each 6 transmembrane segments; "
            "  Major mediator of Ca2+ entry during cardiac action potential plateau (phase 2); "
            "  Also expressed in: brain, vascular smooth muscle, pancreas, immune cells; "
            "  CACNA1C alternative splicing → cardiac-specific and neuronal isoforms; "
            "TIMOTHY SYNDROME (LQT8): "
            "  EXTREMELY RARE: < 20 published cases worldwide; "
            "  De novo GOF mutation G406R (exon 8A) accounts for > 90% of cases; "
            "  QTc massively prolonged (600-700ms); "
            "  2:1 AV block PATHOGNOMONIC: long QT + 2:1 AV block in infant MANDATES CACNA1C; "
            "  CUTANEOUS SYNDACTYLY (webbing between fingers/toes) PATHOGNOMONIC; "
            "  MULTISYSTEM: "
            "    Autism spectrum disorder; "
            "    Severe hypoglycaemia (pancreatic beta-cell involvement); "
            "    Primary immune deficiency; "
            "    Hypertrophic cardiomyopathy (biventricular outflow obstruction); "
            "    Hepatic glycogen storage; "
            "    Distinctive facial features; "
            "DIFFERENTIAL DIAGNOSIS: "
            "  2:1 AV block in newborn + long QT → CACNA1C before anti-Ro antibody testing; "
            "  Avoid assuming immune-mediated AV block if QT very prolonged; "
            "MANAGEMENT: "
            "  ICD: standard (SCD risk very high from birth); "
            "  Verapamil: logical (Ca2+ channel blocker) but limited evidence; "
            "  Beta-blocker + mexiletine: adjunct; "
            "  AVOID ALL QT-PROLONGING DRUGS; "
            "  Multidisciplinary care (neurology, endocrinology, immunology, cardiology); "
            "PROGNOSIS: "
            "  Median survival without ICD < 2.5 years; most events in infancy; "
            "  Long-term neurodevelopmental impact severe"
        ),
        "syndrome": "LQT8-Timothy-syndrome / multisystem-GOF-Cav1.2",
        "inheritance": "AD GOF (de novo G406R > 90%)",
        "pathognomonic": "2:1-AV-block + syndactyly in infant = Timothy syndrome PATHOGNOMONIC; QTc 600-700ms PATHOGNOMONIC",
        "key_drugs_ci": ["all-QT-prolonging-ABSOLUTE-CI"],
        "surveillance": ["ICD-from-infancy", "multidisciplinary-metabolic-neurological-immune"],
    },
    {
        "gene": "HCN4",
        "protein": (
            "HCN4 -- 15q24.1 AD LOF -- 1208aa -- "
            "hyperpolarisation-activated-If-pacemaker-current-SSS2-"
            "hereditary-bradycardia-pacemaker-mandatory-LVNC-overlap-"
            "HCN4-OMIM-605206"
        ),
        "locus": "15q24.1",
        "protein_size": (
            "1208 aa / ~130 kDa / 15q24.1 HCN4 If-current pacemaker channel molecular context: "
            "STRUCTURE: "
            "  1208 aa / ~130 kDa; hyperpolarisation-activated cyclic nucleotide-gated channel; "
            "  6 transmembrane segments (S1-S6); S4 voltage sensor (activated on hyperpolarisation); "
            "  C-terminal CNBD (cyclic nucleotide binding domain): cAMP directly accelerates If; "
            "  Predominantly expressed in: SA node, AV node, Purkinje fibres; "
            "  HCN4 = principal subunit of If ('funny current') → cardiac pacemaking; "
            "  If = inward current on hyperpolarisation → spontaneous diastolic depolarisation; "
            "SSS2 (sick sinus syndrome type 2): "
            "  Hereditary bradycardia: symptomatic resting HR < 40 bpm; "
            "  Chronotropic incompetence: failure to increase HR with exercise; "
            "  Symptoms: syncope, presyncope, exercise intolerance, fatigue; "
            "  PACEMAKER IMPLANTATION MANDATORY for symptomatic SSS2 (no medical alternative); "
            "  Bradycardia drug therapy (atropine, isoprenaline): only temporary bridge; "
            "  Beta-blockers: RELATIVELY CONTRAINDICATED in HCN4 LOF (worsen bradycardia); "
            "LVNC OVERLAP: "
            "  HCN4 variants associated with left ventricular non-compaction (LVNC); "
            "  Not all HCN4 → LVNC; overlap phenotype screening required; "
            "BRUGADA-LIKE ST: "
            "  Some HCN4 variants → transient right precordial ST elevation Brugada-like pattern; "
            "  Sodium channel blockers: RELATIVE CI if Brugada-like ST present; "
            "IVERMECTIN ABSOLUTE CI: "
            "  Ivermectin is a direct HCN channel blocker (at anti-parasitic doses); "
            "  Profound iatrogenic bradycardia reported in patients with HCN4 haploinsufficiency; "
            "  ABSOLUTE CI in HCN4 LOF pathogenic variants; "
            "OTHER HCN4 ASSOCIATIONS: "
            "  Atrial fibrillation risk (SA node disease → AF); "
            "  Some variants: dilated cardiomyopathy (DCM) overlap; "
            "SURVEILLANCE: "
            "  Annual resting ECG + Holter + exercise test; "
            "  Echocardiogram annually (LVNC screen); "
            "  Pacemaker follow-up once implanted"
        ),
        "syndrome": "SSS2-hereditary-bradycardia / LVNC-overlap",
        "inheritance": "AD LOF",
        "pathognomonic": "hereditary-bradycardia HR<40-chronotropic-incompetence = HCN4 SSS2; pacemaker mandatory symptomatic",
        "key_drugs_ci": ["ivermectin-ABSOLUTE-CI-HCN-blocker", "beta-blockers-RELATIVE-CI-worsen-bradycardia"],
        "surveillance": ["annual-ECG-Holter-exercise-test", "echo-LVNC-screen", "pacemaker-follow-up"],
    },
]

DEFINITIONS = {
    "LQT1": "Long QT syndrome type 1 (KCNQ1 LOF): IKs reduction; swim/exertion trigger; Romano-Ward AD, JLNS AR with deafness.",
    "LQT2": "Long QT syndrome type 2 (KCNH2 LOF): IKr hERG reduction; auditory arousal trigger; most drug-induced QT via hERG block.",
    "LQT3": "Long QT syndrome type 3 (SCN5A GOF late INa): nocturnal bradycardia-triggered events; mexiletine first-line.",
    "Brugada_syndrome": "SCN5A LOF: coved ST V1-V3 PATHOGNOMONIC; fever trigger PATHOGNOMONIC; Na-channel blockers ABSOLUTE CI; quinidine treatment.",
    "CPVT1": "Catecholaminergic polymorphic VT (RYR2 GOF): exercise/emotion trigger; bidirectional VT PATHOGNOMONIC; normal resting ECG PATHOGNOMONIC.",
    "ARVC": "Arrhythmogenic right ventricular cardiomyopathy: fibrofatty replacement of myocardium; PKP2 most common gene; epsilon wave PATHOGNOMONIC.",
    "JLNS": "Jervell-Lange-Nielsen syndrome: AR biallelic KCNQ1 LOF; profound deafness + severe LQT PATHOGNOMONIC; ICD + cochlear implant.",
    "SQT": "Short QT syndrome: QTc <340ms; KCNH2/KCNQ1/SCN5A GOF; AF + VF risk; quinidine for SQT1.",
    "Timothy_syndrome": "LQT8 (CACNA1C GOF): EXTREMELY RARE; 2:1 AV block + syndactyly PATHOGNOMONIC; multisystem disorder; median survival without ICD <2.5yr.",
    "SSS": "Sick sinus syndrome: SA node dysfunction; HCN4 LOF → hereditary SSS2; pacemaker mandatory.",
    "LVNC": "Left ventricular non-compaction: failure of myocardial compaction; HCN4 overlap; prominent trabeculations.",
    "Bidirectional_VT": "Beat-to-beat 180-degree QRS axis alternation in tachycardia; PATHOGNOMONIC for CPVT1 (RYR2) or digoxin toxicity.",
    "Epsilon_wave": "Terminal activation delay V1-V3; PATHOGNOMONIC for ARVC; represents delayed RV depolarisation through fibrofatty tissue.",
    "Task_Force_2010": "ARVC diagnostic criteria: major/minor structural, ECG, arrhythmia, family history, and biopsy criteria.",
    "Carvajal_syndrome": "AR biallelic DSP LOF: wooly hair + palmoplantar keratoderma + dilated CMP PATHOGNOMONIC triad.",
    "IKs": "Slow delayed rectifier K+ current (KCNQ1+KCNE1): critical repolarisation reserve; adrenergically upregulated; LOF → LQT1.",
    "IKr": "Rapid delayed rectifier K+ current (KCNH2/hERG): rapid repolarisation; blocked by most QT-prolonging drugs.",
    "If": "Funny current (HCN4): inward current on hyperpolarisation; generates spontaneous diastolic depolarisation; cardiac pacemaking.",
    "PCCD": "Progressive cardiac conduction disease: His-Purkinje fibrosis → BBB → AV block; SCN5A LOF.",
    "LGE": "Late gadolinium enhancement on CMR: fibrosis marker; subepicardial LV inferior/lateral pattern PATHOGNOMONIC for DSP cardiomyopathy.",
}


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def generate_overview() -> dict:
    cohort_stats = []
    for i, g in enumerate(ATLAS_GENES):
        rng = _rng(SEED_BASE + i)
        n = 40
        mean_age = round(max(0.5, rng.gauss(
            1.5 if g["gene"] == "CACNA1C" else   # Timothy = infantile
            35.0 if g["gene"] in ("PKP2", "DSP") else
            28.0 if g["gene"] == "RYR2" else
            42.0 if g["gene"] == "HCN4" else
            32.0, 8.0)), 1)
        pct_female = round(rng.uniform(
            45, 55) if g["gene"] not in ("SCN5A",) else rng.uniform(20, 35), 1)
        cohort_stats.append({
            "gene": g["gene"],
            "n": n,
            "mean_age": mean_age,
            "pct_female": pct_female,
            "syndrome": g["syndrome"].split("/")[0].strip(),
            "inheritance": g["inheritance"].split(";")[0].strip(),
        })

    return {
        "atlas": "Hereditary-Inherited-Arrhythmia-Syndrome-Atlas",
        "subtitle": "Complete 8-Gene Reference — KCNQ1 / KCNH2 / SCN5A / RYR2 / PKP2 / DSP / CACNA1C / HCN4",
        "total_patients": 320,
        "gene_count": 8,
        "seeds": "3742-3749",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "cohort_stats": cohort_stats,
        "clinical_categories": {
            "long_qt_syndromes": ["KCNQ1", "KCNH2", "SCN5A", "CACNA1C"],
            "brugada_sodium_channel": ["SCN5A"],
            "cpvt_ryanodine": ["RYR2"],
            "arvc_desmosomal": ["PKP2", "DSP"],
            "bradycardia_pacemaker": ["HCN4"],
        },
        "key_clinical_rules": [
            "KCNQ1: swim/exertion trigger = LQT1 PATHOGNOMONIC; deafness + severe LQT = JLNS biallelic",
            "KCNH2: sudden auditory arousal (alarm/phone) = LQT2 PATHOGNOMONIC; avoid ALL QT-prolonging drugs",
            "SCN5A: fever = Brugada trigger PATHOGNOMONIC; Na-channel blockers = ABSOLUTE CI for treatment",
            "RYR2: bidirectional VT on exercise = CPVT PATHOGNOMONIC; normal resting ECG = CPVT PATHOGNOMONIC",
            "PKP2: epsilon wave V1-V3 = ARVC PATHOGNOMONIC; competitive sport = ABSOLUTE CI",
            "DSP: LV-dominant ARVC = DSP PATHOGNOMONIC; wooly hair + keratoderma = Carvajal PATHOGNOMONIC",
            "CACNA1C: 2:1 AV block + syndactyly in infant = Timothy syndrome PATHOGNOMONIC (< 20 cases worldwide)",
            "HCN4: hereditary bradycardia + chronotropic incompetence = SSS2; ivermectin = ABSOLUTE CI",
        ],
    }


def generate_breakdown() -> dict:
    gene_data = {}
    for i, g in enumerate(ATLAS_GENES):
        rng = _rng(SEED_BASE + i)
        n = 40
        mean_age = round(max(0.5, rng.gauss(
            1.5 if g["gene"] == "CACNA1C" else
            35.0 if g["gene"] in ("PKP2", "DSP") else
            28.0 if g["gene"] == "RYR2" else
            42.0 if g["gene"] == "HCN4" else
            32.0, 8.0)), 1)
        pct_f = round(rng.uniform(
            45, 55) if g["gene"] != "SCN5A" else rng.uniform(20, 35), 1)
        icd_n = rng.randint(15, 30) if g["gene"] in ("DSP", "CACNA1C", "RYR2") else rng.randint(8, 22)
        scd_n = rng.randint(3, 10)
        syncope_n = rng.randint(12, 28)
        gene_data[g["gene"]] = {
            "n": n,
            "mean_age_dx": mean_age,
            "pct_female": pct_f,
            "icd_implanted_n": icd_n,
            "scd_or_resuscitated_n": scd_n,
            "syncope_n": syncope_n,
            "syndrome": g["syndrome"].split("/")[0].strip(),
            "inheritance": g["inheritance"].split(";")[0].strip(),
            "pathognomonic": g["pathognomonic"],
        }

    return {
        "atlas": "Hereditary-Inherited-Arrhythmia-Syndrome-Atlas",
        "total_patients": 320,
        "seeds": "3742-3749",
        "gene_data": gene_data,
        "atlas_genes": ATLAS_GENES,
        "summary": {
            "lqt_genes": ["KCNQ1", "KCNH2", "SCN5A", "CACNA1C"],
            "arvc_genes": ["PKP2", "DSP"],
            "cpvt_genes": ["RYR2"],
            "bradycardia_genes": ["HCN4"],
            "sport_restriction_all": ["PKP2", "DSP", "RYR2"],
            "pacemaker_required": ["HCN4"],
            "ivermectin_absolute_ci": ["HCN4"],
            "na_channel_blockers_ci_brugada": ["SCN5A"],
            "fever_emergency_protocol": ["SCN5A"],
            "deafness_screen_mandatory": ["KCNQ1"],
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Inherited-Arrhythmia-Syndrome-Atlas",
        "definitions": DEFINITIONS,
        "critical_clinical_rules": {
            "KCNQ1-SWIM-LQT1-PATHOGNOMONIC": "Swim / exertion as arrhythmia trigger = LQT1 (KCNQ1) PATHOGNOMONIC — IKs is adrenergically upregulated; LOF → QT fails to shorten on exercise → VF",
            "KCNQ1-JLNS-DEAFNESS-MANDATORY": "Deafness + severe LQT = JLNS biallelic KCNQ1 — ALWAYS screen paediatric LQT for deafness; ICD + cochlear implant both required",
            "KCNH2-AUDITORY-TRIGGER-PATHOGNOMONIC": "Sudden auditory arousal (alarm/doorbell/phone) = LQT2 PATHOGNOMONIC — modify alarm tone; avoid all hERG-blocking drugs",
            "SCN5A-FEVER-BRUGADA-PATHOGNOMONIC": "Fever = Brugada trigger PATHOGNOMONIC — antipyretics IMMEDIATELY; hospitalise febrile Brugada patients; Na-channel blockers ABSOLUTE CI",
            "SCN5A-COVED-ST-PATHOGNOMONIC": "Coved-type ST elevation V1-V3 (Type 1 Brugada pattern) PATHOGNOMONIC — ajmaline/flecainide = diagnostic challenge ONLY, never treatment",
            "RYR2-BIDIRECTIONAL-VT-PATHOGNOMONIC": "Bidirectional VT on exercise = CPVT PATHOGNOMONIC — normal resting ECG = CPVT PATHOGNOMONIC; exercise stress test MANDATORY for diagnosis",
            "RYR2-ISOPROTERENOL-ABSOLUTE-CI": "Isoproterenol provocation = ABSOLUTE CI in RYR2/CPVT — can precipitate VF storm; programme ICD with ATP to avoid inappropriate shocks",
            "PKP2-EPSILON-WAVE-ARVC": "Epsilon wave V1-V3 = ARVC PATHOGNOMONIC (sensitivity 30% only); competitive sport = ABSOLUTE CI — PKP2 = most common ARVC gene (11-43%)",
            "DSP-LV-DOMINANT-PATHOGNOMONIC": "LV-dominant ARVC = DSP PATHOGNOMONIC — not typical ARVC presentation; CMR LGE subepicardial LV inferior/lateral PATHOGNOMONIC for DSP",
            "DSP-CARVAJAL-TRIAD": "Wooly hair + palmoplantar keratoderma + DCM = Carvajal syndrome PATHOGNOMONIC (AR biallelic DSP)",
            "CACNA1C-2_1-AV-BLOCK-SYNDACTYLY": "2:1 AV block + cutaneous syndactyly in infant = Timothy syndrome PATHOGNOMONIC — ICD from infancy; multidisciplinary care",
            "HCN4-IVERMECTIN-ABSOLUTE-CI": "Ivermectin = ABSOLUTE CI in HCN4 LOF — direct HCN channel blocker → profound iatrogenic bradycardia",
            "HCN4-PACEMAKER-MANDATORY": "Symptomatic hereditary bradycardia (HR <40, chronotropic incompetence) = pacemaker mandatory — no medical alternative; HCN4 SSS2",
            "ALL-ARVC-SPORT-ABSOLUTE-CI": "Competitive sport = ABSOLUTE CI for all desmosomal ARVC genes (PKP2, DSP) — exercise accelerates fibrofatty progression regardless of symptom status",
        },
    }
