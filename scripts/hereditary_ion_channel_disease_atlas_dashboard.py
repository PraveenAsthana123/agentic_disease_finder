#!/usr/bin/env python3
"""Hereditary-Ion-Channel-Disease-Atlas — Complete 8-Gene Hereditary Channelopathy Atlas.

KCNA1   (voltage-gated potassium channel Kv1.1; 495 aa; 12p13.32; AD;
         Episodic Ataxia Type 1 (EA1) + Myokymia — continuous muscle fibre activity;
         acetazolamide first-line; carbamazepine for myokymia;
         seed SEED_BASE+0).
KCNQ1   (voltage-gated potassium channel Kv7.1; 676 aa; 11p15.5; AD/AR;
         Long QT Syndrome Type 1 (LQT1) AD + Jervell and Lange-Nielsen Syndrome (JLNS) AR;
         SWIMMING ABSOLUTELY CONTRAINDICATED — LQT1 triggered by exertion/swimming;
         beta-blockers first-line; ICD for high-risk; JLNS = profound congenital SNHL + severe LQT;
         seed SEED_BASE+1).
KCNH2   (voltage-gated potassium channel hERG/Kv11.1; 1159 aa; 7q36.1; AD;
         Long QT Syndrome Type 2 (LQT2) — triggered by sudden auditory stimuli / arousal;
         beta-blockers; AVOID QT-prolonging drugs; potassium supplementation;
         ALARM CLOCK TRIGGER PATHOGNOMONIC — sudden noise → TdP;
         seed SEED_BASE+2).
SCN5A   (cardiac sodium channel Nav1.5; 2016 aa; 3p22.2; AD;
         Brugada Syndrome / LQT3 / Sick Sinus Syndrome / PCCD — broad allelic spectrum;
         ICD for Brugada with symptoms; quinidine for arrhythmia suppression;
         FEVER ABSOLUTELY CONTRAINDICATED — unmasks coved-type Brugada pattern;
         seed SEED_BASE+3).
RYR1    (ryanodine receptor 1; 5038 aa; 19q13.2; AD/AR;
         Malignant Hyperthermia (MH) AD GOF / Central Core Disease (CCD) AR LOF;
         DANTROLENE EMERGENCY MANDATORY — 2.5 mg/kg IV bolus; AVOID volatile anaesthetics + succinylcholine;
         MH alert bracelet MANDATORY; CCD: non-progressive myopathy, hip dislocation, scoliosis;
         seed SEED_BASE+4).
CACNA1S (L-type voltage-gated calcium channel alpha-1S; 1873 aa; 1q32.1; AD;
         Hypokalemic Periodic Paralysis Type 1 (HypoPP1) + MH susceptibility;
         dichlorphenamide / acetazolamide; AVOID high-carbohydrate meals + rest after exercise;
         PARADOXICAL DEPOLARISATION at low K+ — membrane paradox unique to HypoPP;
         seed SEED_BASE+5).
CLCN1   (voltage-gated chloride channel ClC-1; 988 aa; 7q34; AD/AR;
         Myotonia Congenita — Thomsen (AD) / Becker (AR) — warm-up phenomenon;
         mexiletine FIRST-LINE; carbamazepine; lamotrigine;
         BECKER FORM: transient weakness after myotonia — DISTINGUISH from paramyotonia;
         seed SEED_BASE+6).
KCNJ2   (inward-rectifier potassium channel Kir2.1; 427 aa; 17q24.3; AD;
         Andersen-Tawil Syndrome (ATS / LQT7) — TRIAD pathognomonic;
         TRIAD = periodic paralysis + cardiac arrhythmia (LQT7/VT) + dysmorphic features;
         acetazolamide for paralysis; ICD for malignant VT; AVOID triggers;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2030-2037).
"""

import random

SEED_BASE = 2030

IC_GENES = [
    # -- KCNA1 — Episodic Ataxia Type 1 / Myokymia (AD) --------------------------------
    {
        "gene": "KCNA1",
        "alt_name": "KCNA1 (Kv1.1 Voltage-Gated K+ Channel / AD — EA1 Episodic-Ataxia-Type-1 — Myokymia-Continuous-Muscle-Fibre-Activity — Acetazolamide-First-Line)",
        "protein": (
            "KCNA1 -- 12p13.32 AD -- KCNA1-495aa -- "
            "Episodic-Ataxia-Type-1-EA1-Kv1.1-Voltage-Gated-K-Channel -- "
            "Seconds-to-Minutes-Ataxia-Episodes-Stress-Exercise-Fever-Trigger -- "
            "Continuous-Muscle-Fibre-Activity-Myokymia-PATHOGNOMONIC -- "
            "Acetazolamide-First-Line-Carbamazepine-Myokymia"
        ),
        "locus": "12p13.32",
        "protein_size": "495 aa",
        "inheritance": "AD (autosomal dominant) — loss-of-function haploinsufficiency",
        "age_of_onset": (
            "EA1 onset: childhood to adolescence (2-15 yr typical); "
            "Episodes: seconds to minutes (DISTINGUISH from EA2 hours-long); "
            "Triggers: startle, exercise, emotional stress, fever, vestibular stimulation; "
            "Interictal: continuous myokymia — rippling muscle movement visible under skin; "
            "Myokymia on EMG: doublet/triplet discharges at 40-300 Hz (neuromyotonia pattern); "
            "Cerebellar: may develop mild persistent ataxia and nystagmus between attacks in adults; "
            "Cognitive: usually normal; "
            "Some families: temporal lobe epilepsy co-segregates (Kv1.1 expressed in hippocampus)"
        ),
        "key_biomarker": (
            "EMG: spontaneous continuous muscle fibre activity (CMFA) — doublets/triplets; "
            "neuromyotonia pattern between episodes even when asymptomatic — PATHOGNOMONIC; "
            "Serum CK: normal or mildly elevated; "
            "Brain MRI: normal (DISTINGUISH from EA2: cerebellar atrophy in EA2); "
            "Acetylcholine receptor antibodies: NEGATIVE (distinguish from acquired neuromyotonia = Isaac syndrome); "
            "EEG: usually normal; temporal lobe changes if co-segregating epilepsy; "
            "Gene panel: KCNA1 sequencing; V408A most common pathogenic variant (>50 kindreds); "
            "Family testing: AD — 50% risk; test first-degree relatives for myokymia on EMG"
        ),
        "pathognomonic": (
            "CONTINUOUS MYOKYMIA on EMG in a patient with episodic ataxia = EA1 until proven otherwise; "
            "Seconds-to-minutes duration distinguishes EA1 from EA2 (hours) and EA3/4/5 (other durations); "
            "No persistent cerebellar atrophy on MRI (versus EA2 which develops vermis atrophy); "
            "DISTINGUISH: Isaac syndrome (acquired neuromyotonia — anti-CASPR2/anti-VGKC — NOT genetic); "
            "DISTINGUISH: EA2 (CACNA1A — longer episodes, nystagmus, progressive cerebellar atrophy, responds acetazolamide better than EA1)"
        ),
        "treatment": (
            "Episodic ataxia: acetazolamide 125-250 mg BD — reduces episode frequency/severity; "
            "Myokymia/neuromyotonia: carbamazepine OR mexiletine OR phenytoin; "
            "Triggers: avoid sudden exercise, emotional stress where possible; "
            "Epilepsy (if co-segregating): standard AED — avoid voltage-gated channel blockers that worsen myokymia; "
            "Physical therapy for persistent ataxia; "
            "Genetic counselling: AD — 50% risk to offspring; "
            "AVOID: high-temperature environments (fever → episodes); antipyretics promptly; "
            "AVOID: potassium-depleting diuretics (worsens channel dysfunction)"
        ),
        "critical_flags": [
            "KCNA1-EA1-SECONDS-MINUTES-NOT-HOURS-EA2",
            "KCNA1-MYOKYMIA-CMFA-EMG-PATHOGNOMONIC",
            "KCNA1-ACETAZOLAMIDE-FIRST-LINE",
            "KCNA1-CARBAMAZEPINE-MYOKYMIA",
            "KCNA1-V408A-MOST-COMMON-VARIANT",
            "KCNA1-DISTINGUISH-ISAACS-SYNDROME-ACQUIRED",
            "KCNA1-TEMPORAL-LOBE-EPILEPSY-CO-SEGREGATION",
        ],
        "seed": SEED_BASE + 0,
    },
    # -- KCNQ1 — LQT1 / JLNS (AD/AR) --------------------------------------------------
    {
        "gene": "KCNQ1",
        "alt_name": "KCNQ1 (Kv7.1 IKs K+ Channel / AD-LQT1 / AR-JLNS — Swimming-ABSOLUTELY-CI — Beta-Blockers-First-Line — JLNS-Profound-SNHL-Plus-Severe-LQT)",
        "protein": (
            "KCNQ1 -- 11p15.5 AD-LQT1-AR-JLNS -- KCNQ1-676aa -- "
            "Long-QT-Syndrome-Type-1-IKs-Slow-Delayed-Rectifier-K-Channel -- "
            "SWIMMING-EXERCISE-TRIGGERED-TdP-PATHOGNOMONIC-LQT1 -- "
            "Jervell-Lange-Nielsen-Syndrome-JLNS-Biallelic-Profound-SNHL -- "
            "Beta-Blockers-First-Line-ICD-High-Risk-JLNS"
        ),
        "locus": "11p15.5",
        "protein_size": "676 aa",
        "inheritance": "AD (LQT1 — haploinsufficiency) / AR (JLNS — biallelic LOF + stria vascularis K+ recycling defect → profound SNHL)",
        "age_of_onset": (
            "LQT1: QTc usually 460-500 ms; events in childhood/adolescence; "
            "Triggers: physical exercise (especially swimming), emotional stress — adrenergic surge; "
            "Symptoms: syncope, palpitations, cardiac arrest — during/immediately after exercise; "
            "SCD risk: highest in children/young adults; "
            "JLNS: biallelic KCNQ1 — QTc >550 ms typical; profound congenital sensorineural hearing loss (SNHL) + severe LQT; "
            "JLNS events: spontaneous, exercise-triggered — extreme SCD risk without treatment; "
            "Asymp carriers: QTc prolongation only; "
            "Hormonal: LQT1 women — post-partum period highest risk"
        ),
        "key_biomarker": (
            "12-lead ECG: QTc prolongation (>440 ms men; >460 ms women); broad-based T wave (LQT1 pattern); "
            "Holter: QTc dynamics; pause-dependent QT prolongation less prominent than LQT2; "
            "Exercise test: QTc FAILS to shorten normally on exercise (LQT1 feature); "
            "Epinephrine QT stress test: LQT1 — QTc paradoxically lengthens at low-dose epi; "
            "Audiometry: normal in LQT1; profound bilateral SNHL in JLNS; "
            "Genetic: KCNQ1 panel; >600 variants known; A341V, G269S common pathogenic; "
            "Family cascade testing: 12-lead ECG + KCNQ1 genotyping; "
            "Schwartz score: QTc + symptoms + family history scoring"
        ),
        "pathognomonic": (
            "SWIMMING-triggered syncope in child with prolonged QTc = LQT1 until proven otherwise; "
            "Broad-based T wave on ECG (versus LQT2 notched T wave; LQT3 late-onset peaked T wave); "
            "JLNS triad: profound congenital SNHL + QTc >550 ms + cardiac events in childhood = biallelic KCNQ1; "
            "DISTINGUISH: LQT2 (KCNH2 — auditory trigger, notched T wave); LQT3 (SCN5A — rest/sleep trigger, peaked late T); "
            "DISTINGUISH: Romano-Ward vs JLNS — JLNS SNHL is the phenotypic marker of biallelic state"
        ),
        "treatment": (
            "LQT1: beta-blockers (nadolol preferred; propranolol/metoprolol) — highly effective in LQT1; "
            "ICD: prior SCD survivor, refractory symptoms on beta-blockers; "
            "SWIMMING ABSOLUTELY CONTRAINDICATED — highest-risk trigger for LQT1 SCD; "
            "Left cardiac sympathetic denervation (LCSD): alternative to ICD when ICD refused/contraindicated; "
            "JLNS: cochlear implants + beta-blockers + ICD (JLNS = highest LQT risk); "
            "AVOID: QT-prolonging drugs (CredibleMeds website), hypokalemia, hypomagnesemia; "
            "Post-partum monitoring: women with LQT1 at highest risk 3-9 months post-partum; "
            "Sport restriction: competitive sports contraindicated; recreational low-intensity sport with supervision"
        ),
        "critical_flags": [
            "KCNQ1-SWIMMING-ABSOLUTELY-CONTRAINDICATED-LQT1",
            "KCNQ1-BROAD-BASED-T-WAVE-ECG-LQT1",
            "KCNQ1-BETA-BLOCKERS-HIGHLY-EFFECTIVE-LQT1",
            "KCNQ1-JLNS-PROFOUND-SNHL-BIALLELIC",
            "KCNQ1-JLNS-QTc-ABOVE-550ms-EXTREME-RISK",
            "KCNQ1-EXERCISE-STRESS-TEST-QTc-FAILS-SHORTEN",
            "KCNQ1-POSTPARTUM-WOMEN-HIGHEST-RISK",
        ],
        "seed": SEED_BASE + 1,
    },
    # -- KCNH2 — LQT2 / hERG (AD) -------------------------------------------------------
    {
        "gene": "KCNH2",
        "alt_name": "KCNH2 (hERG Kv11.1 IKr K+ Channel / AD — LQT2 — Auditory-Trigger-Alarm-Clock-PATHOGNOMONIC — Notched-T-Wave — Avoid-QT-Drugs-Hypokalemia)",
        "protein": (
            "KCNH2 -- 7q36.1 AD -- KCNH2-1159aa -- "
            "Long-QT-Syndrome-Type-2-IKr-Rapid-Delayed-Rectifier-hERG -- "
            "SUDDEN-AUDITORY-STIMULUS-ALARM-CLOCK-TRIGGER-PATHOGNOMONIC -- "
            "Notched-Bifid-T-Wave-ECG-DISTINCTIVE-LQT2 -- "
            "Avoid-QT-Prolonging-Drugs-Hypokalemia-Hypomagnesemia"
        ),
        "locus": "7q36.1",
        "protein_size": "1159 aa",
        "inheritance": "AD (autosomal dominant) — loss-of-function haploinsufficiency (dominant-negative for many missense)",
        "age_of_onset": (
            "LQT2: events across all ages, including neonatal period; "
            "CHARACTERISTIC trigger: sudden auditory stimuli — alarm clock, phone ring, doorbell → TdP/VF; "
            "Also triggered by: emotional arousal, sleep, post-partum; "
            "QTc: usually 470-520 ms; "
            "Female sex: higher penetrance and event rate than males (adrenergic differences); "
            "ECG T wave: notched or bifid T wave — DISTINCTIVE for LQT2; "
            "SCD risk: 2nd highest of LQT subtypes after JLNS; "
            "Neonatal LQT2: 2:1 AV block in utero possible — foetal bradycardia screening indicated"
        ),
        "key_biomarker": (
            "12-lead ECG: QTc prolongation + NOTCHED/BIFID T wave (especially leads V4-V6) — LQT2 hallmark; "
            "Exercise test: QTc shortens on exercise but may be prolonged at rest/recovery; "
            "Epinephrine QT stress test: LQT2 — QTc increase in recovery phase (distinct from LQT1); "
            "Holter: QTc lengthening during sleep/rest; "
            "Genetic: KCNH2 sequencing — >300 pathogenic variants; A558V, R176W, N629D common; "
            "Drug history: MANDATORY — many common drugs (azithromycin, haloperidol, hydroxychloroquine) prolong QTc; "
            "Potassium: serum K+ level — hypokalemia greatly worsens LQT2; "
            "Family cascade: ECG + KCNH2 genotyping in first-degree relatives"
        ),
        "pathognomonic": (
            "ALARM CLOCK syncope — cardiac arrest triggered by alarm/phone ring = LQT2 until proven otherwise; "
            "Notched bifid T wave on ECG in leads V4-V6 — highly distinctive of LQT2; "
            "DISTINGUISH: LQT1 (broad T wave, swim trigger); LQT3 (peaked late T, rest/sleep trigger); "
            "DISTINGUISH: acquired LQT (drug-induced — always check CredibleMeds drug list); "
            "Dominant-negative missense variants (e.g. G628S) impair hERG trafficking — worse phenotype than LOF"
        ),
        "treatment": (
            "Beta-blockers: nadolol or propranolol — effective but LESS than LQT1; "
            "AVOID: QT-prolonging drugs — check CredibleMeds for every new prescription; "
            "AVOID: hypokalemia (diuretics, vomiting, diarrhoea) — potassium supplementation; "
            "AVOID: hypomagnesemia — magnesium supplementation; "
            "ICD: SCD survivors, symptomatic despite beta-blockers, QTc >500 ms in high-risk; "
            "PRACTICAL: remove alarm clock from bedside — switch to vibration/light alarm; "
            "Mexiletine: sodium channel blocker shortens QTc in LQT2 (via IKr-INa interactions); "
            "Post-partum monitoring: women at high risk; "
            "Genetic counselling: AD — 50% offspring risk; family cascade screening essential"
        ),
        "critical_flags": [
            "KCNH2-ALARM-CLOCK-AUDITORY-TRIGGER-PATHOGNOMONIC-LQT2",
            "KCNH2-NOTCHED-BIFID-T-WAVE-ECG-DISTINCTIVE",
            "KCNH2-AVOID-ALL-QT-PROLONGING-DRUGS-CREDIBLEMEDS",
            "KCNH2-AVOID-HYPOKALEMIA-POTASSIUM-SUPPLEMENTATION",
            "KCNH2-DOMINANT-NEGATIVE-TRAFFICKING-DEFECT-WORSE",
            "KCNH2-FEMALE-HIGHER-PENETRANCE-EVENTS",
            "KCNH2-NEONATAL-AV-BLOCK-FOETAL-SURVEILLANCE",
        ],
        "seed": SEED_BASE + 2,
    },
    # -- SCN5A — Brugada / LQT3 / SSS / PCCD (AD) -------------------------------------
    {
        "gene": "SCN5A",
        "alt_name": "SCN5A (Nav1.5 Cardiac Na+ Channel / AD — Brugada-Syndrome — LQT3 — SSS — PCCD — Fever-ABSOLUTELY-CI — Quinidine-Arrhythmia-Suppression)",
        "protein": (
            "KCNQ1 -- SCN5A -- 3p22.2 AD -- SCN5A-2016aa -- "
            "Brugada-Syndrome-GOF-Type3-INa-Loss / Long-QT-Type3-INa-GOF-Persistent / "
            "Sick-Sinus-Syndrome-SSS / Progressive-Cardiac-Conduction-Disease-PCCD -- "
            "FEVER-ABSOLUTELY-CONTRAINDICATED-Unmasks-Brugada -- "
            "COVED-TYPE-ST-ELEVATION-V1-V3-PATHOGNOMONIC-Brugada"
        ),
        "locus": "3p22.2",
        "protein_size": "2016 aa",
        "inheritance": "AD (autosomal dominant) — broad allelic series: LOF → Brugada/SSS/PCCD; GOF → LQT3",
        "age_of_onset": (
            "Brugada: predominantly adult males (5:1 male); events at rest/sleep; "
            "ECG: coved-type ST elevation ≥2 mm in ≥1 right precordial lead (V1/V2) at 2nd-4th ICS; "
            "Fever: unmasks Brugada pattern and precipitates VF — FEVER IS AN EMERGENCY IN BRUGADA; "
            "LQT3: late-onset peaked T wave at rest; events during sleep/bradycardia; "
            "SSS: sinus node dysfunction, bradycardia, chronotropic incompetence; "
            "PCCD: progressive PR + QRS widening → bundle branch block → complete AV block; "
            "Asian prevalence: Brugada higher in SE Asia (Sudden Unexpected Nocturnal Death Syndrome = Brugada); "
            "Sodium channel blocker unmasking: flecainide/ajmaline/procainamide → Brugada pattern diagnostic"
        ),
        "key_biomarker": (
            "ECG at rest: may be concealed — coved pattern only during fever/provocation; "
            "Sodium channel blocker challenge (flecainide/ajmaline): provokes type 1 pattern — DIAGNOSIS; "
            "Temperature monitoring: MANDATORY during febrile illness; "
            "HRV/rhythm: ventricular fibrillation during sleep/rest; "
            "For LQT3: QTc prolongation + late-onset peaked T wave; "
            "For SSS: Holter shows prolonged sinus pauses; "
            "For PCCD: ECG serial PR/QRS measurement; "
            "Electrophysiology study (EPS): Brugada risk stratification (VF inducibility); "
            "Genetic: SCN5A sequencing; R1232W, E1784K, delK1500 among known pathogenic; "
            "Family cascade: ECG + ajmaline provocation + SCN5A genotyping"
        ),
        "pathognomonic": (
            "COVED-TYPE ST elevation ≥2 mm in V1/V2/V3 = Brugada type 1 — pathognomonic; "
            "Spontaneous type 1 pattern = highest risk; drug-provoked type 1 = diagnostic; "
            "FEVER PRECIPITATING VF in known Brugada = medical emergency — cool patient + ICU; "
            "DISTINGUISH: right bundle branch block (no coved ST morphology); "
            "DISTINGUISH: arrhythmogenic RV cardiomyopathy (epsilon wave, fatty infiltration); "
            "SCN5A is the ONLY gene with Brugada + LQT3 + SSS + PCCD allelic spectrum"
        ),
        "treatment": (
            "Brugada symptomatic (VF/syncope): ICD — ONLY effective therapy proven to prevent SCD; "
            "Brugada arrhythmia storm: quinidine (Ito current blocker) suppresses arrhythmias; "
            "Brugada asymptomatic: monitor; ICD debated; avoid triggers; "
            "FEVER: treat aggressively with antipyretics; hospital admission; ICU during febrile illness with known Brugada; "
            "AVOID: sodium channel blockers (flecainide, propafenone, ajmaline, tricyclics, cocaine); "
            "AVOID: vagotonic drugs (neostigmine) — vagal tone worsens Brugada; "
            "LQT3: mexiletine shortens QTc; beta-blockers LESS effective; "
            "SSS/PCCD: pacemaker if symptomatic bradycardia; "
            "Brugada diet: avoid heavy carbohydrate meals (postprandial vagal surge)"
        ),
        "critical_flags": [
            "SCN5A-FEVER-ABSOLUTELY-CONTRAINDICATED-BRUGADA-VF",
            "SCN5A-COVED-ST-V1-V3-PATHOGNOMONIC-BRUGADA",
            "SCN5A-QUINIDINE-ARRHYTHMIA-STORM-SUPPRESSION",
            "SCN5A-ICD-ONLY-PROVEN-SCD-PREVENTION-BRUGADA",
            "SCN5A-AVOID-SODIUM-CHANNEL-BLOCKERS-FLECAINIDE",
            "SCN5A-SLEEP-REST-VF-TRIGGER-BRUGADA",
            "SCN5A-BROAD-ALLELIC-BRUGADA-LQT3-SSS-PCCD",
        ],
        "seed": SEED_BASE + 3,
    },
    # -- RYR1 — Malignant Hyperthermia / Central Core Disease (AD/AR) ------------------
    {
        "gene": "RYR1",
        "alt_name": "RYR1 (Ryanodine Receptor 1 Sarcoplasmic-Reticulum Ca2+ Release / AD-MH-GOF / AR-CCD-LOF — Dantrolene-EMERGENCY-MANDATORY — Avoid-Volatile-Anaesthetics-Succinylcholine — MH-Alert-Bracelet-MANDATORY)",
        "protein": (
            "RYR1 -- 19q13.2 AD-MH-GOF-AR-CCD-LOF -- RYR1-5038aa -- "
            "Ryanodine-Receptor-1-SR-Ca2-Release-Channel-Skeletal-Muscle -- "
            "MALIGNANT-HYPERTHERMIA-MH-LIFE-THREATENING-ANAESTHETIC-CRISIS -- "
            "DANTROLENE-2.5mgkg-IV-BOLUS-EMERGENCY-THEN-1mgkg-q6h -- "
            "Central-Core-Disease-CCD-Non-Progressive-Myopathy-Hip-Dislocation"
        ),
        "locus": "19q13.2",
        "protein_size": "5038 aa",
        "inheritance": "AD (MH susceptibility — GOF: hyperactivation of SR Ca2+ release) / AR biallelic (CCD/MMDO — LOF: non-progressive myopathy)",
        "age_of_onset": (
            "MH: triggered by exposure to volatile halogenated anaesthetics (halothane, sevoflurane, desflurane, isoflurane) or succinylcholine; "
            "MH crisis: rigidity + hyperthermia + acidosis + tachycardia + rhabdomyolysis — occurs intraoperatively; "
            "Mortality untreated: 70-80%; with dantrolene: <5%; "
            "Dantrolene must be immediately available in ALL ORs; "
            "CCD (biallelic LOF): congenital-onset non-progressive proximal muscle weakness; "
            "CCD features: hip dislocation at birth, scoliosis, delayed motor milestones; "
            "CCD biopsy: central cores on Gomori trichrome — mitochondria and oxidative enzyme-free zones; "
            "CCD patients ALSO at MH risk (RYR1 same gene — allelic)"
        ),
        "key_biomarker": (
            "MH diagnosis: Caffeine-Halothane Contracture Test (CHCT — gold standard, invasive) OR In Vitro Contracture Test (IVCT); "
            "Requires fresh muscle biopsy — cannot be done after MH episode (wait 3 months); "
            "Serum CK: chronically elevated in RYR1 carriers (500-1000 U/L at rest) — PATHOGNOMONIC clue; "
            "Resting CK elevated > 3-4× ULN in >90% of RYR1 MH-susceptible individuals; "
            "Muscle biopsy (CCD): central cores on H&E and Gomori trichrome; NADH-dehydrogenase staining shows absence; "
            "Genetic: RYR1 sequencing — >500 MH/CCD variants; C4958S, R2163H, G2434R among hotspots; "
            "ALL first-degree relatives: genetic testing + CK; "
            "Anaesthesia alert: MH alert documentation in medical records + bracelet"
        ),
        "pathognomonic": (
            "INTRAOPERATIVE HYPERTHERMIA + MUSCLE RIGIDITY + RAPIDLY RISING ETCO2 after volatile agent = MH CRISIS; "
            "Masseter spasm after succinylcholine = MH susceptibility until proven otherwise; "
            "Chronically elevated resting CK (3-10× ULN) in an otherwise well person = RYR1 variant until proven otherwise; "
            "Central cores on muscle biopsy = CCD diagnosis; "
            "DISTINGUISH: MH from NMS (neuroleptic malignant syndrome — slower onset, no trigger, dopamine antagonist); "
            "DISTINGUISH: serotonin syndrome (myoclonus, diarrhoea, not triggered by anaesthetics)"
        ),
        "treatment": (
            "MH CRISIS: (1) STOP volatile agent IMMEDIATELY; (2) switch to TIVA (propofol + fentanyl — SAFE); "
            "(3) DANTROLENE 2.5 mg/kg IV bolus — repeat q5-10 min to max 10 mg/kg; "
            "(4) Cool patient — cool IV fluids, ice packs, cooling blanket; "
            "(5) Treat acidosis + hyperkalemia + arrhythmias; (6) Monitor CK, K+, creatinine (rhabdomyolysis); "
            "Dantrolene maintenance: 1 mg/kg q6h for 24-48h post-crisis; "
            "MH PREVENTION: use TIVA (total IV anaesthesia — propofol, fentanyl, non-depolarising NMBAs); "
            "ABSOLUTELY AVOID: volatile agents (sevoflurane, halothane, desflurane, isoflurane), succinylcholine; "
            "CCD: physiotherapy; orthopaedic management (hip, scoliosis); avoid triggers; "
            "MH alert bracelet MANDATORY; medical alert documentation"
        ),
        "critical_flags": [
            "RYR1-DANTROLENE-2.5mgkg-IV-BOLUS-EMERGENCY-MH-CRISIS",
            "RYR1-VOLATILE-ANAESTHETICS-ABSOLUTELY-CONTRAINDICATED",
            "RYR1-SUCCINYLCHOLINE-ABSOLUTELY-CONTRAINDICATED",
            "RYR1-MH-ALERT-BRACELET-MEDICAL-RECORD-MANDATORY",
            "RYR1-TIVA-PROPOFOL-FENTANYL-SAFE-ALTERNATIVE",
            "RYR1-CK-CHRONICALLY-ELEVATED-REST-CLUE",
            "RYR1-CCD-CENTRAL-CORES-BIOPSY-PATHOGNOMONIC",
        ],
        "seed": SEED_BASE + 4,
    },
    # -- CACNA1S — HypoPP1 / MH susceptibility (AD) ------------------------------------
    {
        "gene": "CACNA1S",
        "alt_name": "CACNA1S (L-Type Ca2+ Channel Alpha-1S Dihydropyridine Receptor / AD — Hypokalemic-Periodic-Paralysis-Type-1 — MH-Susceptibility — Paradoxical-Depolarisation-Membrane-Paradox — Dichlorphenamide-First-Line)",
        "protein": (
            "CACNA1S -- 1q32.1 AD -- CACNA1S-1873aa -- "
            "L-Type-Voltage-Gated-Ca-Channel-Alpha1S-DHPR-T-Tubule-Triad -- "
            "Hypokalemic-Periodic-Paralysis-HypoPP-Type-1 -- "
            "PARADOXICAL-DEPOLARISATION-AT-LOW-K-MEMBRANE-PARADOX-UNIQUE -- "
            "Malignant-Hyperthermia-Susceptibility-MH-Concurrent"
        ),
        "locus": "1q32.1",
        "protein_size": "1873 aa",
        "inheritance": "AD (autosomal dominant) — gain-of-function gating-pore current creates paradoxical depolarisation",
        "age_of_onset": (
            "HypoPP1 onset: first or second decade; episodic attacks of weakness; "
            "Triggers: carbohydrate-rich meal, rest after exercise, cold exposure, alcohol, stress; "
            "During attack: serum K+ falls (shift into cells via insulin/catecholamine); "
            "Paradox: at low K+, normal channels HYPERPOLARISE (less excitable), but CACNA1S HypoPP1 channels DEPOLARISE → inexcitable; "
            "Weakness: proximal > distal; legs > arms; typically spares respiratory and ocular muscles; "
            "Duration: hours to days; "
            "Permanent myopathy: may develop after decades of attacks — vacuolar myopathy; "
            "MH susceptibility: R1086H and R1086C CACNA1S variants confer concurrent MH risk"
        ),
        "key_biomarker": (
            "Serum K+: LOW during attack (1.5-3.0 mmol/L) — DIAGNOSTIC; "
            "ECG during attack: U waves, ST depression, T flattening (hypokalemia); "
            "Glucose loading test: oral carbohydrate provokes attack (K+ shift by insulin); "
            "Exercise test: McManis protocol — compound muscle action potential drops post-exercise; "
            "Muscle biopsy: vacuolar myopathy in chronic disease; "
            "Thyroid function: MANDATORY — exclude thyrotoxic periodic paralysis (TPP); "
            "Genetic: CACNA1S sequencing — R1086H (most common HypoPP1), R1086C, R528H; "
            "24-hour urine K+: low in familial periodic paralysis (versus renal K+ wasting); "
            "MCAS/IVF test: identify MH susceptibility for anaesthetic planning"
        ),
        "pathognomonic": (
            "PARADOXICAL DEPOLARISATION: muscles become LESS excitable at LOW K+ (opposite of normal physiology) — unique to HypoPP1/HypoPP2; "
            "Episodic flaccid paralysis + hypokalemia + carbohydrate/rest trigger = HypoPP until proven otherwise; "
            "DISTINGUISH: HypoPP2 (KCNJ2 — Andersen-Tawil cardiac involvement, dysmorphic features); "
            "DISTINGUISH: Thyrotoxic periodic paralysis (TPP — same phenotype but thyroid-driven; predominantly Asian males); "
            "DISTINGUISH: HypoPP1 (CACNA1S) from HypoPP2 (SCN4A — NaV1.4, also AD); "
            "Acetazolamide can WORSEN HypoPP1 (unlike HypoPP2 where it helps) — genomic diagnosis essential before prescribing"
        ),
        "treatment": (
            "Acute attack: oral KCl 40-60 mmol (avoid glucose/saline IV — worsen K+ shift); "
            "IV KCl: only if unable to swallow or severe weakness; dilute; monitor ECG; "
            "Prevention: dichlorphenamide (FDA-approved, carbonic anhydrase inhibitor) OR acetazolamide; "
            "CAUTION: acetazolamide may WORSEN some CACNA1S variants — monitor; "
            "Diet: low carbohydrate, avoid large carbohydrate meals; "
            "Spironolactone: aldosterone antagonist, K+ sparing; "
            "Exercise: moderate-intensity preferred over intense then rest; "
            "MH precautions: TIVA protocol; dantrolene available; avoid volatile agents; "
            "Genetic counselling: AD — 50% risk; family anaesthetic alert records"
        ),
        "critical_flags": [
            "CACNA1S-PARADOXICAL-DEPOLARISATION-UNIQUE-MEMBRANE-PARADOX",
            "CACNA1S-DICHLORPHENAMIDE-FDA-APPROVED-FIRST-LINE",
            "CACNA1S-ACETAZOLAMIDE-MAY-WORSEN-GENOTYPE-CHECK",
            "CACNA1S-IV-GLUCOSE-SALINE-WORSEN-ATTACK-AVOID",
            "CACNA1S-MH-SUSCEPTIBILITY-R1086H-R1086C-TIVA",
            "CACNA1S-CARBOHYDRATE-REST-COLD-TRIGGERS",
            "CACNA1S-DISTINGUISH-THYROTOXIC-PP-THYROID-TFTs",
        ],
        "seed": SEED_BASE + 5,
    },
    # -- CLCN1 — Myotonia Congenita Thomsen/Becker (AD/AR) ----------------------------
    {
        "gene": "CLCN1",
        "alt_name": "CLCN1 (ClC-1 Voltage-Gated Chloride Channel / AD-Thomsen / AR-Becker — Mexiletine-First-Line — Warm-Up-Phenomenon-PATHOGNOMONIC — Becker-Transient-Weakness-Distinguishes-From-Paramyotonia)",
        "protein": (
            "CLCN1 -- 7q34 AD-Thomsen-AR-Becker -- CLCN1-988aa -- "
            "Voltage-Gated-Chloride-Channel-ClC-1-Skeletal-Muscle-T-Tubule -- "
            "Myotonia-Congenita-Impaired-Cl-Conductance-Prolonged-Action-Potentials -- "
            "WARM-UP-PHENOMENON-Myotonia-Decreases-Repeated-Exercise-PATHOGNOMONIC -- "
            "Mexiletine-Sodium-Channel-Blocker-First-Line"
        ),
        "locus": "7q34",
        "protein_size": "988 aa",
        "inheritance": "AD (Thomsen — haploinsufficiency + dominant negative; mild) / AR (Becker — biallelic LOF; more severe; transient weakness feature)",
        "age_of_onset": (
            "Thomsen (AD): infancy to childhood onset; lifelong; "
            "Becker (AR): onset 4-12 yr; slightly later than Thomsen; "
            "Myotonia: delayed muscle relaxation after voluntary contraction; "
            "Warm-up phenomenon: myotonia IMPROVES with repeated muscle use — PATHOGNOMONIC for myotonia congenita; "
            "Becker distinguishing feature: transient WEAKNESS before myotonia on first contraction (limbs 'give way'); "
            "Cold: worsens myotonia in both (DISTINGUISH: paramyotonia congenita — SCN4A — cold dramatically worsens); "
            "Percussion myotonia: sustained muscle contraction visible/palpable after percussion (e.g. thenar eminence); "
            "Hypertrophy: 'Herculean' or 'athletic' appearance — prominent muscle bulk from myotonic contractions"
        ),
        "key_biomarker": (
            "EMG: myotonic discharges — waxing-waning frequency and amplitude ('dive-bomber' sound); "
            "After repeated needle insertion, discharges DECREASE (warm-up on EMG); "
            "Serum CK: mildly elevated or normal; "
            "Muscle biopsy: non-specific; increased fibre size variation; central nuclei in Becker; "
            "Genetic: CLCN1 sequencing — >150 pathogenic variants; F413C, A531V, R894X common; "
            "Cold exposure test: brief immersion of hand in ice water — myotonia worsens but does NOT cause paralysis (unlike paramyotonia/HyperPP); "
            "Exercise test: repetitive grip; IMPROVES with exercise (warm-up) — diagnostic; "
            "Family cascade: AD in Thomsen; AR in Becker — carrier parents show no symptoms"
        ),
        "pathognomonic": (
            "WARM-UP PHENOMENON: myotonia that DECREASES with repeated exercise = myotonia congenita (CLCN1); "
            "DISTINGUISH: paramyotonia congenita (SCN4A) — myotonia WORSENS with repeated exercise (opposite); "
            "DISTINGUISH: myotonic dystrophy type 1 DM1 (DMPK) — multi-systemic, distal weakness, facial myotonia, cataracts, CTG repeat; "
            "Becker transient weakness: first contraction causes leg buckling/weakness → THEN myotonia — unique feature; "
            "Muscle hypertrophy in otherwise healthy young person with grip myotonia = CLCN1 until proven otherwise"
        ),
        "treatment": (
            "Mexiletine 150-200 mg TDS — FIRST-LINE sodium channel blocker; reduces myotonia significantly; "
            "Carbamazepine / phenytoin: alternatives; "
            "Lamotrigine: effective for myotonia congenita; "
            "Quinine: historically used; cardiac monitoring required (QTc); "
            "Avoid: beta-blockers — may worsen myotonia; "
            "Avoid: depolarising muscle relaxants (succinylcholine) — can trigger prolonged myotonic contraction intraoperatively; "
            "General anaesthesia warning: myotonic contractions during anaesthesia induction; non-depolarising NMBAs safe; "
            "Mexiletine cardiac monitoring: ECG at baseline and after dose changes (sodium channel effect on heart); "
            "Warming: warm environment reduces myotonia; "
            "Physiotherapy: not curative but improves function"
        ),
        "critical_flags": [
            "CLCN1-WARM-UP-PHENOMENON-PATHOGNOMONIC-MYOTONIA-CONGENITA",
            "CLCN1-MEXILETINE-FIRST-LINE",
            "CLCN1-DISTINGUISH-SCN4A-PARAMYOTONIA-WORSENS-EXERCISE",
            "CLCN1-BECKER-TRANSIENT-WEAKNESS-BEFORE-MYOTONIA",
            "CLCN1-AVOID-SUCCINYLCHOLINE-INTRAOP-MYOTONIA",
            "CLCN1-AVOID-BETA-BLOCKERS-WORSEN",
            "CLCN1-MUSCLE-HYPERTROPHY-HERCULEAN-APPEARANCE",
        ],
        "seed": SEED_BASE + 6,
    },
    # -- KCNJ2 — Andersen-Tawil Syndrome (ATS/LQT7) (AD) ----------------------------
    {
        "gene": "KCNJ2",
        "alt_name": "KCNJ2 (Kir2.1 Inward-Rectifier IK1 K+ Channel / AD — Andersen-Tawil-Syndrome-ATS-LQT7 — TRIAD-Pathognomonic — Periodic-Paralysis-Plus-Cardiac-Arrhythmia-Plus-Dysmorphic-Features)",
        "protein": (
            "KCNJ2 -- 17q24.3 AD -- KCNJ2-427aa -- "
            "Kir2.1-Inward-Rectifier-IK1-K-Channel-Cardiac-Muscle-Stabiliser -- "
            "Andersen-Tawil-Syndrome-ATS-LQT7 -- "
            "TRIAD-Periodic-Paralysis-Cardiac-Arrhythmia-Dysmorphic-Features-PATHOGNOMONIC -- "
            "Acetazolamide-Paralysis-ICD-Malignant-VT"
        ),
        "locus": "17q24.3",
        "protein_size": "427 aa",
        "inheritance": "AD (autosomal dominant) — loss-of-function haploinsufficiency; dominant-negative for some missense",
        "age_of_onset": (
            "ATS onset: childhood to young adulthood; "
            "TRIAD: (1) episodic periodic paralysis (hypokalemic or normokalemic); "
            "(2) cardiac arrhythmia — ventricular bigeminy, PVCs, VT, TdP — QTc prolonged + prominent U waves; "
            "(3) dysmorphic features — low-set ears, micrognathia, clinodactyly, syndactyly, short stature, scoliosis; "
            "Not all 3 features present in every patient (incomplete penetrance); "
            "Cardiac arrhythmia: may be ventricular bigeminy (most common) or malignant polymorphic VT/TdP; "
            "DISTINGUISH: ATS rarely causes malignant arrhythmia compared to LQT1/LQT2/LQT3; VT often 'benign-looking' bigeminy; "
            "Paralysis: HYPOKALEMIC (most) or NORMOKALEMIC (unlike other HypoPP syndromes); "
            "Dysmorphic: may be subtle — careful clinical examination required"
        ),
        "key_biomarker": (
            "ECG: QTc prolongation + LARGE PROMINENT U WAVES — characteristic of ATS/LQT7; "
            "QTc appears longer than it is if U waves measured as T waves — recheck QTU interval; "
            "Holter: ventricular bigeminy, PVCs — may be >10,000 ectopics/24h but haemodynamically tolerated; "
            "Serum K+: during attack (hypokalemia or normokalemia); "
            "Dysmorphic assessment: low-set ears, broad forehead, micrognathia, hand/foot anomalies; "
            "Genetic: KCNJ2 sequencing — R218Q, R218W, G300V, D71V among pathogenic; "
            "Exercise CMAP test: post-exercise CMAP decrement (as in HypoPP); "
            "Electrophysiology study: VT inducibility — guides ICD decision; "
            "Family examination: seek subtle dysmorphic features + ECG in all first-degree relatives"
        ),
        "pathognomonic": (
            "TRIAD: periodic paralysis + cardiac arrhythmia (bigeminy/PVCs/LQT) + dysmorphic features = ATS/KCNJ2; "
            "PROMINENT U WAVES on ECG in periodic paralysis patient = KCNJ2 until proven otherwise; "
            "QTU prolongation mistaken for extreme QTc — measurement artefact common in ATS; "
            "DISTINGUISH: CACNA1S HypoPP1 (no cardiac features, no dysmorphic); SCN4A HypoPP2 (no cardiac); "
            "DISTINGUISH: LQT1/LQT2/LQT3 (no periodic paralysis, no dysmorphic); "
            "Normokalemic periodic paralysis is very unusual — seen in ATS (and SCN4A); "
            "Ventricular bigeminy in a young person with dysmorphic features = KCNJ2 flag"
        ),
        "treatment": (
            "Paralysis: potassium supplementation (hypokalemic attacks); "
            "Acetazolamide: reduces paralysis frequency in ATS; "
            "Cardiac arrhythmia: beta-blockers (especially for bigeminy/PVCs); "
            "ICD: for documented malignant VT/VF or high-risk; "
            "Flecainide: suppresses PVCs/VT in ATS (Na channel blocker — effect on Kir2.1 downstream); "
            "AVOID: triggers — hypokalemia, intense exercise, carbohydrate loads; "
            "AVOID: QT-prolonging drugs; "
            "Genetic counselling: AD — 50% offspring; examine all relatives for subtle dysmorphic + ECG; "
            "Orthopaedic: manage scoliosis; dental (micrognathia); "
            "Cardiology review: annual ECG + Holter; ICD discussion in symptomatic arrhythmia patients"
        ),
        "critical_flags": [
            "KCNJ2-ATS-TRIAD-PERIODIC-PARALYSIS-CARDIAC-DYSMORPHIC-PATHOGNOMONIC",
            "KCNJ2-PROMINENT-U-WAVES-ECG-PATHOGNOMONIC",
            "KCNJ2-NORMOKALEMIC-PARALYSIS-UNUSUAL-KCNJ2-SCN4A",
            "KCNJ2-BIGEMINY-HAEMODYNAMICALLY-TOLERATED-USUALLY",
            "KCNJ2-QTU-PROLONGATION-MEASUREMENT-ARTEFACT",
            "KCNJ2-DYSMORPHIC-FEATURES-LOW-SET-EARS-MICROGNATHIA",
            "KCNJ2-FLECAINIDE-PVC-SUPPRESSION",
        ],
        "seed": SEED_BASE + 7,
    },
]


# ---------------------------------------------------------------------------
# Cohort generation
# ---------------------------------------------------------------------------

def _generate_cohort(gene_entry: dict) -> list:
    rng = random.Random(gene_entry["seed"])
    cohort = []
    gene = gene_entry["gene"]

    for i in range(40):
        age = rng.randint(3, 65)
        sex = rng.choice(["M", "F"])

        # Gene-specific clinical feature probabilities
        if gene == "KCNA1":
            cardiac_arrhythmia     = rng.random() < 0.05
            sudden_cardiac_death   = rng.random() < 0.02
            periodic_paralysis     = rng.random() < 0.05
            myotonia               = rng.random() < 0.10
            malignant_hyperthermia = rng.random() < 0.02
            muscle_weakness        = rng.random() < 0.20
            episodic_ataxia        = rng.random() < 0.98
            myokymia_emg           = rng.random() < 0.97
            warm_up_phenomenon     = rng.random() < 0.05
            drug_trigger_hazard    = rng.random() < 0.15
            dysmorphic_features    = rng.random() < 0.03
            snhl                   = rng.random() < 0.03
            fever_hazard           = rng.random() < 0.60
            swimming_restriction   = rng.random() < 0.10
            ck_elevated_rest       = rng.random() < 0.25

        elif gene == "KCNQ1":
            cardiac_arrhythmia     = rng.random() < 0.90
            sudden_cardiac_death   = rng.random() < 0.25
            periodic_paralysis     = rng.random() < 0.02
            myotonia               = rng.random() < 0.02
            malignant_hyperthermia = rng.random() < 0.02
            muscle_weakness        = rng.random() < 0.05
            episodic_ataxia        = rng.random() < 0.02
            myokymia_emg           = rng.random() < 0.02
            warm_up_phenomenon     = rng.random() < 0.02
            drug_trigger_hazard    = rng.random() < 0.85
            dysmorphic_features    = rng.random() < 0.05
            snhl                   = rng.random() < 0.30  # JLNS subset
            fever_hazard           = rng.random() < 0.15
            swimming_restriction   = rng.random() < 0.97
            ck_elevated_rest       = rng.random() < 0.05

        elif gene == "KCNH2":
            cardiac_arrhythmia     = rng.random() < 0.92
            sudden_cardiac_death   = rng.random() < 0.28
            periodic_paralysis     = rng.random() < 0.02
            myotonia               = rng.random() < 0.02
            malignant_hyperthermia = rng.random() < 0.02
            muscle_weakness        = rng.random() < 0.03
            episodic_ataxia        = rng.random() < 0.02
            myokymia_emg           = rng.random() < 0.02
            warm_up_phenomenon     = rng.random() < 0.02
            drug_trigger_hazard    = rng.random() < 0.95
            dysmorphic_features    = rng.random() < 0.03
            snhl                   = rng.random() < 0.04
            fever_hazard           = rng.random() < 0.20
            swimming_restriction   = rng.random() < 0.55
            ck_elevated_rest       = rng.random() < 0.04

        elif gene == "SCN5A":
            cardiac_arrhythmia     = rng.random() < 0.95
            sudden_cardiac_death   = rng.random() < 0.35
            periodic_paralysis     = rng.random() < 0.02
            myotonia               = rng.random() < 0.02
            malignant_hyperthermia = rng.random() < 0.02
            muscle_weakness        = rng.random() < 0.03
            episodic_ataxia        = rng.random() < 0.02
            myokymia_emg           = rng.random() < 0.02
            warm_up_phenomenon     = rng.random() < 0.02
            drug_trigger_hazard    = rng.random() < 0.90
            dysmorphic_features    = rng.random() < 0.03
            snhl                   = rng.random() < 0.03
            fever_hazard           = rng.random() < 0.98
            swimming_restriction   = rng.random() < 0.30
            ck_elevated_rest       = rng.random() < 0.05

        elif gene == "RYR1":
            cardiac_arrhythmia     = rng.random() < 0.10
            sudden_cardiac_death   = rng.random() < 0.08
            periodic_paralysis     = rng.random() < 0.05
            myotonia               = rng.random() < 0.08
            malignant_hyperthermia = rng.random() < 0.97
            muscle_weakness        = rng.random() < 0.80  # CCD subset
            episodic_ataxia        = rng.random() < 0.03
            myokymia_emg           = rng.random() < 0.05
            warm_up_phenomenon     = rng.random() < 0.05
            drug_trigger_hazard    = rng.random() < 0.98  # volatile agents
            dysmorphic_features    = rng.random() < 0.10
            snhl                   = rng.random() < 0.05
            fever_hazard           = rng.random() < 0.25
            swimming_restriction   = rng.random() < 0.10
            ck_elevated_rest       = rng.random() < 0.93

        elif gene == "CACNA1S":
            cardiac_arrhythmia     = rng.random() < 0.15
            sudden_cardiac_death   = rng.random() < 0.05
            periodic_paralysis     = rng.random() < 0.98
            myotonia               = rng.random() < 0.05
            malignant_hyperthermia = rng.random() < 0.30  # R1086H/C subset
            muscle_weakness        = rng.random() < 0.95
            episodic_ataxia        = rng.random() < 0.03
            myokymia_emg           = rng.random() < 0.03
            warm_up_phenomenon     = rng.random() < 0.03
            drug_trigger_hazard    = rng.random() < 0.85  # carbs, glucose
            dysmorphic_features    = rng.random() < 0.05
            snhl                   = rng.random() < 0.03
            fever_hazard           = rng.random() < 0.15
            swimming_restriction   = rng.random() < 0.20
            ck_elevated_rest       = rng.random() < 0.30

        elif gene == "CLCN1":
            cardiac_arrhythmia     = rng.random() < 0.05
            sudden_cardiac_death   = rng.random() < 0.02
            periodic_paralysis     = rng.random() < 0.08
            myotonia               = rng.random() < 0.98
            malignant_hyperthermia = rng.random() < 0.05
            muscle_weakness        = rng.random() < 0.40  # Becker transient weakness
            episodic_ataxia        = rng.random() < 0.03
            myokymia_emg           = rng.random() < 0.20
            warm_up_phenomenon     = rng.random() < 0.97
            drug_trigger_hazard    = rng.random() < 0.70  # succinylcholine
            dysmorphic_features    = rng.random() < 0.05
            snhl                   = rng.random() < 0.03
            fever_hazard           = rng.random() < 0.10
            swimming_restriction   = rng.random() < 0.08
            ck_elevated_rest       = rng.random() < 0.35

        elif gene == "KCNJ2":
            cardiac_arrhythmia     = rng.random() < 0.88
            sudden_cardiac_death   = rng.random() < 0.12
            periodic_paralysis     = rng.random() < 0.90
            myotonia               = rng.random() < 0.05
            malignant_hyperthermia = rng.random() < 0.03
            muscle_weakness        = rng.random() < 0.80
            episodic_ataxia        = rng.random() < 0.05
            myokymia_emg           = rng.random() < 0.05
            warm_up_phenomenon     = rng.random() < 0.05
            drug_trigger_hazard    = rng.random() < 0.70
            dysmorphic_features    = rng.random() < 0.85
            snhl                   = rng.random() < 0.05
            fever_hazard           = rng.random() < 0.15
            swimming_restriction   = rng.random() < 0.50
            ck_elevated_rest       = rng.random() < 0.20

        else:
            cardiac_arrhythmia = sudden_cardiac_death = periodic_paralysis = myotonia = False
            malignant_hyperthermia = muscle_weakness = episodic_ataxia = myokymia_emg = False
            warm_up_phenomenon = drug_trigger_hazard = dysmorphic_features = snhl = False
            fever_hazard = swimming_restriction = ck_elevated_rest = False

        cohort.append({
            "patient_id":            f"{gene}-{i+1:03d}",
            "age":                   age,
            "sex":                   sex,
            "gene":                  gene,
            "cardiac_arrhythmia":    cardiac_arrhythmia,
            "sudden_cardiac_death":  sudden_cardiac_death,
            "periodic_paralysis":    periodic_paralysis,
            "myotonia":              myotonia,
            "malignant_hyperthermia":malignant_hyperthermia,
            "muscle_weakness":       muscle_weakness,
            "episodic_ataxia":       episodic_ataxia,
            "myokymia_emg":          myokymia_emg,
            "warm_up_phenomenon":    warm_up_phenomenon,
            "drug_trigger_hazard":   drug_trigger_hazard,
            "dysmorphic_features":   dysmorphic_features,
            "snhl":                  snhl,
            "fever_hazard":          fever_hazard,
            "swimming_restriction":  swimming_restriction,
            "ck_elevated_rest":      ck_elevated_rest,
        })

    return cohort


# ---------------------------------------------------------------------------
# API functions
# ---------------------------------------------------------------------------

def overview() -> dict:
    all_cohorts = [_generate_cohort(g) for g in IC_GENES]
    all_pts = [p for c in all_cohorts for p in c]
    total = len(all_pts)

    def N(key): return sum(1 for p in all_pts if p[key])

    return {
        "atlas": "Hereditary-Ion-Channel-Disease-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Channelopathy Atlas: "
            "KCNA1 (EA1 episodic ataxia — myokymia CMFA — acetazolamide) + "
            "KCNQ1 (LQT1 — swimming ABSOLUTELY CI — JLNS profound SNHL biallelic) + "
            "KCNH2 (LQT2 — alarm-clock auditory trigger — notched T wave — avoid QT drugs) + "
            "SCN5A (Brugada — fever ABSOLUTELY CI — coved ST V1-V3 — quinidine) + "
            "RYR1 (MH — dantrolene emergency — volatile agents ABSOLUTELY CI — CCD central cores) + "
            "CACNA1S (HypoPP1 — paradoxical depolarisation — dichlorphenamide — MH risk R1086H) + "
            "CLCN1 (myotonia congenita — warm-up phenomenon — mexiletine first-line — Becker transient weakness) + "
            "KCNJ2 (ATS/LQT7 — triad paralysis/arrhythmia/dysmorphic — prominent U waves)"
        ),
        "genes": [g["gene"] for g in IC_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}–{SEED_BASE + len(IC_GENES) - 1}",
        "cardiac_arrhythmia_patients":     N("cardiac_arrhythmia"),
        "sudden_cardiac_death_patients":   N("sudden_cardiac_death"),
        "periodic_paralysis_patients":     N("periodic_paralysis"),
        "myotonia_patients":               N("myotonia"),
        "malignant_hyperthermia_patients": N("malignant_hyperthermia"),
        "muscle_weakness_patients":        N("muscle_weakness"),
        "episodic_ataxia_patients":        N("episodic_ataxia"),
        "myokymia_emg_patients":           N("myokymia_emg"),
        "warm_up_phenomenon_patients":     N("warm_up_phenomenon"),
        "drug_trigger_hazard_patients":    N("drug_trigger_hazard"),
        "dysmorphic_features_patients":    N("dysmorphic_features"),
        "snhl_patients":                   N("snhl"),
        "fever_hazard_patients":           N("fever_hazard"),
        "swimming_restriction_patients":   N("swimming_restriction"),
        "ck_elevated_rest_patients":       N("ck_elevated_rest"),
        "gene_patient_counts": {g["gene"]: 40 for g in IC_GENES},
        "pathway": (
            "Hereditary ion channel diseases — shared mechanism: "
            "germline variants in voltage-gated K+ channels (KCNA1, KCNQ1, KCNH2), Na+ channel (SCN5A), "
            "inward-rectifier K+ channel (KCNJ2), L-type Ca2+ channel (CACNA1S), "
            "sarcoplasmic reticulum Ca2+ release channel (RYR1), or Cl- channel (CLCN1) "
            "alter membrane excitability in heart, skeletal muscle, or neurons. "
            "LOF K+/Cl- channels → prolonged depolarisation → excitability excess (myotonia, LQT, epilepsy). "
            "GOF Na+ persistent current → prolonged depolarisation → LQT3 or hyperkalaemic paralysis. "
            "GOF Ca2+ release (RYR1) → uncontrolled Ca2+ release → malignant hyperthermia. "
            "Paradoxical depolarisation (CACNA1S HypoPP1): gating-pore current active at hyperpolarised VM → inexcitability."
        ),
        "key_clinical_insight": (
            "KCNA1: myokymia on EMG + seconds-duration ataxia episodes = EA1; acetazolamide + carbamazepine. "
            "KCNQ1: swimming absolutely contraindicated; broad-based T wave; JLNS = biallelic + profound SNHL. "
            "KCNH2: alarm clock trigger pathognomonic; notched bifid T wave; avoid ALL QT-prolonging drugs. "
            "SCN5A: fever absolutely contraindicated; coved ST V1-V3 pathognomonic Brugada; ICD + quinidine. "
            "RYR1: dantrolene 2.5 mg/kg IV emergency; volatile agents + succinylcholine absolutely CI; MH bracelet mandatory. "
            "CACNA1S: paradoxical depolarisation unique; dichlorphenamide FDA-approved; acetazolamide may worsen R1086H. "
            "CLCN1: warm-up phenomenon pathognomonic; mexiletine first-line; Becker transient weakness distinguishes. "
            "KCNJ2: ATS triad pathognomonic; prominent U waves; normokalemic PP unusual clue."
        ),
    }


def breakdown() -> dict:
    result = {}
    for gene_entry in IC_GENES:
        cohort = _generate_cohort(gene_entry)
        gene = gene_entry["gene"]

        def pct(key):
            return round(100 * sum(1 for p in cohort if p[key]) / len(cohort))

        result[gene] = {
            "gene":                       gene,
            "alt_name":                   gene_entry["alt_name"],
            "locus":                      gene_entry["locus"],
            "protein_size":               gene_entry["protein_size"],
            "inheritance":                gene_entry["inheritance"],
            "n_patients":                 len(cohort),
            "cardiac_arrhythmia_pct":     pct("cardiac_arrhythmia"),
            "sudden_cardiac_death_pct":   pct("sudden_cardiac_death"),
            "periodic_paralysis_pct":     pct("periodic_paralysis"),
            "myotonia_pct":               pct("myotonia"),
            "malignant_hyperthermia_pct": pct("malignant_hyperthermia"),
            "muscle_weakness_pct":        pct("muscle_weakness"),
            "episodic_ataxia_pct":        pct("episodic_ataxia"),
            "myokymia_emg_pct":           pct("myokymia_emg"),
            "warm_up_phenomenon_pct":     pct("warm_up_phenomenon"),
            "drug_trigger_hazard_pct":    pct("drug_trigger_hazard"),
            "dysmorphic_features_pct":    pct("dysmorphic_features"),
            "snhl_pct":                   pct("snhl"),
            "fever_hazard_pct":           pct("fever_hazard"),
            "swimming_restriction_pct":   pct("swimming_restriction"),
            "ck_elevated_rest_pct":       pct("ck_elevated_rest"),
            "age_of_onset":   gene_entry["age_of_onset"],
            "key_biomarker":  gene_entry["key_biomarker"],
            "pathognomonic":  gene_entry["pathognomonic"],
            "treatment":      gene_entry["treatment"],
            "critical_flags": gene_entry["critical_flags"],
            "seed":           gene_entry["seed"],
            "cohort_preview": cohort[:5],
        }
    return result


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Ion-Channel-Disease-Atlas",
        "pathway": "Voltage-Gated K+/Na+/Ca2+ Channels / Inward-Rectifier K+ / Cl- Channel / SR Ca2+ Release",
        "shared_mechanism": (
            "Hereditary ion channel diseases (channelopathies) result from germline variants that alter "
            "the gating, expression, trafficking, or ion selectivity of channels in heart, skeletal muscle, or brain. "
            "The final common pathway is dysregulated membrane potential: "
            "LOF in repolarising K+ channels (KCNQ1-IKs, KCNH2-IKr, KCNJ2-IK1) or gain-of-function persistent Na+ current (SCN5A-LQT3) "
            "prolong cardiac action potentials → QTc prolongation → risk of TdP/VF. "
            "LOF in skeletal muscle Cl- channel (CLCN1) → impaired repolarisation → myotonia. "
            "Gating-pore current in CACNA1S (HypoPP1) creates aberrant inward current at hyperpolarised voltages → paradoxical inexcitability at low K+. "
            "GOF in RYR1 SR Ca2+ release → uncontrolled Ca2+ flood → sustained contraction + heat = malignant hyperthermia. "
            "KCNA1 LOF (Kv1.1) → impaired peripheral nerve K+ repolarisation → continuous spontaneous firing = myokymia + episodic ataxia."
        ),
        "genes": {
            g["gene"]: {
                "full_name": g["alt_name"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "critical_flags": g["critical_flags"],
                "pathognomonic": g["pathognomonic"],
                "treatment_summary": g["treatment"],
            }
            for g in IC_GENES
        },
        "glossary": {
            "Episodic Ataxia Type 1 (EA1)": "KCNA1 LOF — seconds-to-minutes ataxia episodes triggered by startle/exercise; continuous myokymia (CMFA on EMG) between episodes; acetazolamide; DISTINGUISH EA2 (CACNA1A — hours-long, cerebellar atrophy)",
            "Myokymia (CMFA)": "Continuous muscle fibre activity — spontaneous doublet/triplet EMG discharges from peripheral nerve hyperexcitability; visible 'rippling' movement under skin; pathognomonic for KCNA1 (EA1); must exclude acquired neuromyotonia (anti-CASPR2)",
            "LQT1 (KCNQ1)": "IKs (slow delayed rectifier K+) LOF; broadened T wave; exertion/swimming triggers TdP; QTc 460-500 ms typical; highly responsive to beta-blockers; SWIMMING ABSOLUTELY CI",
            "JLNS (Jervell and Lange-Nielsen)": "Biallelic KCNQ1 — IKs completely absent; profound congenital SNHL (stria vascularis K+ recycling dependent on KCNQ1) + severe QTc prolongation (>550 ms); very high SCD risk; cochlear implants + ICD",
            "LQT2 (KCNH2/hERG)": "IKr (rapid delayed rectifier K+) LOF; notched/bifid T wave; auditory trigger (alarm clock) pathognomonic; QTc 470-520 ms; K+ and Mg2+ supplementation; avoid 1000+ QT-prolonging drugs (CredibleMeds)",
            "Brugada Syndrome (SCN5A)": "INa LOF → reduced depolarisation reserve in right ventricle; coved-type ST elevation V1-V3 pathognomonic; VF at rest/sleep; FEVER ABSOLUTELY CI (unmasks/precipitates VF); ICD only proven SCD prevention; quinidine for arrhythmia storm",
            "Long QT Type 3 (SCN5A GOF)": "SCN5A GOF → persistent late INa → prolonged repolarisation; peaked late T wave at rest; events during bradycardia/sleep; mexiletine shortens QTc; DISTINGUISH from Brugada (same gene — opposite allelic effect)",
            "Malignant Hyperthermia (RYR1 GOF)": "Inhalational anaesthetic or succinylcholine triggers uncontrolled SR Ca2+ release → sustained skeletal muscle contraction → hyperthermia + acidosis + rhabdomyolysis; DANTROLENE 2.5 mg/kg IV bolus is the antidote; mortality 70-80% untreated, <5% with dantrolene",
            "Central Core Disease (CCD)": "Biallelic RYR1 LOF — non-progressive congenital myopathy; central cores on NADH-stain muscle biopsy (mitochondria-free zones in fibre centres); hip dislocation, scoliosis; also at MH risk (same gene)",
            "Hypokalemic Periodic Paralysis Type 1 (HypoPP1)": "CACNA1S GOF gating-pore current → paradoxical depolarisation at low K+ → inexcitability; episodic flaccid paralysis; serum K+ low during attack; dichlorphenamide FDA-approved; AVOID IV glucose/saline during attack",
            "Paradoxical depolarisation": "In HypoPP1 (CACNA1S) and HypoPP2 (SCN4A): at low extracellular K+, normal physiology dictates hyperpolarisation, but pathological gating-pore current causes net DEPOLARISATION → channels become inexcitable → weakness; unique to HypoPP channelopathies",
            "Myotonia Congenita (CLCN1)": "ClC-1 Cl- channel LOF → prolonged action potentials in skeletal muscle T-tubules → myotonia; Thomsen AD (mild); Becker AR (more severe, transient weakness); WARM-UP PHENOMENON pathognomonic (myotonia improves with repeated use); mexiletine first-line",
            "Warm-up phenomenon": "Myotonia that DECREASES (improves) with repeated voluntary muscle contraction — pathognomonic for myotonia congenita (CLCN1); OPPOSITE to paramyotonia congenita (SCN4A) which WORSENS with repeated exercise and cold",
            "Andersen-Tawil Syndrome (ATS/LQT7)": "KCNJ2 (Kir2.1/IK1) LOF — TRIAD: episodic periodic paralysis + cardiac arrhythmia (bigeminy/PVCs/LQT) + dysmorphic features (micrognathia, low-set ears, clinodactyly); prominent U waves on ECG; normokalemic PP unusual clue",
            "Prominent U waves": "Large positive deflection after T wave on ECG; PATHOGNOMONIC clue for KCNJ2/ATS (LQT7); beware of measuring QTU instead of QTc — apparent extreme QTc may be QTU artefact",
            "Dantrolene": "Ryanodine receptor antagonist (RYR1 antagonist) — blocks SR Ca2+ release channel; specific antidote for malignant hyperthermia; dose 2.5 mg/kg IV bolus, repeat q5-10 min to max 10 mg/kg; maintenance 1 mg/kg q6h for 24-48h",
            "CredibleMeds QTDrugs": "Database of QT-prolonging drugs (crediblemeds.org); mandatory reference before prescribing to patients with KCNQ1/KCNH2/SCN5A/KCNJ2; >1000 drugs with various levels of risk",
            "TIVA (Total Intravenous Anaesthesia)": "Propofol + fentanyl + non-depolarising NMBAs — SAFE protocol for RYR1 MH-susceptible and CACNA1S R1086H patients; avoids volatile agents and succinylcholine that trigger MH",
            "Dichlorphenamide": "Carbonic anhydrase inhibitor; FDA-approved for hypokalemic periodic paralysis (CACNA1S HypoPP1 and SCN4A HypoPP2); mechanism: chronic mild metabolic acidosis shifts K+ out of cells; preferred over acetazolamide which may worsen CACNA1S variants",
            "Mexiletine": "Class IB sodium channel blocker; oral; first-line for myotonia congenita (CLCN1); also shortens QTc in LQT2 (KCNH2) and LQT3 (SCN5A); cardiac monitoring required; dose 150-200 mg TDS",
            "Caffeine-Halothane Contracture Test (CHCT)": "Gold standard for MH susceptibility diagnosis; requires fresh muscle biopsy (cannot be frozen); contracture threshold measured in caffeine and halothane; cannot be done post-crisis (wait 3 months); European IVCT also used",
        },
        "surveillance_protocols": {
            "KCNA1": "Annual neurology review; EMG for myokymia documentation; acetazolamide dose titration; temperature management during febrile illness; epilepsy monitoring if co-segregating; genetic counselling AD",
            "KCNQ1": "Annual ECG + QTc; Holter if symptomatic; exercise test (QTc shortening); beta-blocker adherence; swimming prohibition documented; JLNS — cochlear implant team; audiometry in all KCNQ1 family members; post-partum monitoring women",
            "KCNH2": "Annual ECG + QTc; Holter; serum K+/Mg2+; CredibleMeds drug review at every new prescription; auditory trigger management (silent alarm); beta-blocker adherence; electrophysiology if high-risk",
            "SCN5A": "Annual ECG; ajmaline provocation if concealed; fever emergency protocol documented; avoid drug list; ICD check if implanted; electrophysiology for risk stratification; family cascade ECG + provocation",
            "RYR1": "Anaesthetic alert in medical record + MH bracelet; CK baseline annually; genetic counselling for surgical procedures; CHCT/IVCT for family members before elective surgery; CCD patients — physiotherapy + orthopaedic follow-up",
            "CACNA1S": "Annual: serum K+, ECG, muscle strength; glucose loading test if diagnostic uncertainty; anaesthetic alert R1086H/C; dichlorphenamide dose review; thyroid function annually (exclude TPP); dietary counselling",
            "CLCN1": "Annual neurology; EMG warm-up demonstration; mexiletine efficacy/toxicity monitoring (ECG, LFTs); anaesthetic precautions documented; cold avoidance counselling; Becker — assess functional weakness separately from myotonia",
            "KCNJ2": "Annual: ECG + Holter (bigeminy burden); serum K+; muscle strength; dysmorphic feature documentation; ICD if malignant VT; acetazolamide for paralysis; orthopaedic review (scoliosis, micrognathia — orthodontics); family examination for subtle triad",
        },
    }


if __name__ == "__main__":
    import json
    ov = overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Cardiac arrhythmia patients: {ov['cardiac_arrhythmia_patients']}")
    print(f"Malignant hyperthermia patients: {ov['malignant_hyperthermia_patients']}")
    print(f"Myotonia patients: {ov['myotonia_patients']}")
    print(f"Periodic paralysis patients: {ov['periodic_paralysis_patients']}")
    print(f"Episodic ataxia patients: {ov['episodic_ataxia_patients']}")
