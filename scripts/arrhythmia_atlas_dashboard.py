#!/usr/bin/env python3
"""Arrhythmia-Atlas — Complete 8-Gene Inherited Cardiac Arrhythmia Atlas
KCNQ1  (Kv7.1; 580 aa; 11p15.5; LQT1 / Jervell-Lange-Nielsen; AD/AR;
         most common LQTS; IKs alpha subunit; swim-triggered syncope;
         beta-blocker highly effective ~80% SCD protection; JLN (AR biallelic) severe deafness+QT) ·
KCNH2  (hERG Kv11.1; 1159 aa; 7q36.1; LQT2; AD; IKr alpha subunit;
         auditory-triggered syncope; drug-induced LQT2 commonest cause;
         potassium supplementation + hERG-SAFE drugs; NO class III antiarrhythmics) ·
SCN5A  (Nav1.5; 2016 aa; 3p22.2; LQT3 + Brugada Syndrome + PCCD; AD;
         gain-of-function = LQT3; loss-of-function = Brugada/PCCD;
         mexiletine for LQT3; quinidine for Brugada (ICD second-line in select);
         fever ABSOLUTELY CONTRAINDICATED in Brugada — must pre-treat) ·
RYR2   (Ryanodine receptor 2; 4967 aa; 1q43; CPVT1 Catecholaminergic Polymorphic VT; AD;
         exercise/emotion-triggered bidirectional/polymorphic VT; SR calcium leak;
         nadolol PREFERRED over metoprolol; flecainide adjunct; ICDs can storm;
         swimming + high-emotion sport PROHIBITED; cognitive-behavioural approach for anger) ·
CASQ2  (Calsequestrin 2; 399 aa; 1p13.3-p11; CPVT2; AR; earlier-onset, more severe;
         same exercise/emotion trigger as CPVT1; biallelic — both copies needed;
         nadolol + flecainide mandatory; ICD very high risk of storm; consider sympathetic denervation) ·
HCN4   (Funny channel If; 1203 aa; 15q24.1; Familial Sinus Node Disease + Hereditary Bradycardia;
         AD; pacemaker current suppressed → sinus bradycardia/arrest/atrial standstill;
         ivabradine CAUTION (further suppresses If); pacemaker implantation often required;
         overlap with Brugada-like phenotype in some variants) ·
ANK2   (Ankyrin-B; 3952 aa; 4q25; LQT4 / Ankyrin-B Syndrome; AD;
         multimorphic — sinus node dysfunction + atrial flutter + VF + CPVT-like;
         beta-blocker + ICD; not captured by standard LQTS genetic panels without ANK2 inclusion) ·
KCNE1  (MinK; 129 aa; 21q22.12; LQT5 / Jervell-Lange-Nielsen Type 2; AD/AR;
         beta subunit of IKs (Kv7.1-MinK); females disproportionately affected;
         JLN2 (AR biallelic) = deafness + severe QT prolongation;
         beta-blocker first-line; sex-specific risk — same as KCNQ1 (both subunits of IKs))
320-patient aggregate cohort (8 x 40, seeds 1294-1301)

Inherited Cardiac Arrhythmia — Key Principles:
  - TRIGGER SPECIFICITY: KCNQ1(LQT1) = swimming/exertion; KCNH2(LQT2) = auditory/sudden noise;
    SCN5A(LQT3) = bradycardia/sleep/fever; RYR2/CASQ2(CPVT) = exercise/catecholamine surge.
  - DRUG-INDUCED LQT: hERG (IKr) channel is UNIQUELY VULNERABLE to drug block;
    KCNH2 carriers have HEIGHTENED risk from QT-prolonging medications;
    check CredibleMeds/Arizona CERT list for any new prescription.
  - ICD PARADOX IN CPVT: ICD shock → catecholamine surge → triggers more VT → electrical storm;
    beta-blocker MUST be maximised before ICD implantation; flecainide adjunct may prevent storm.
  - FEVER IN BRUGADA: fever actively unmasks/exacerbates Brugada pattern (Nav1.5 temperature-
    sensitive inactivation gate) → acetaminophen/NSAID immediately for any fever ≥38°C; anti-fever plan.
  - SEX DIFFERENCES: females have naturally longer QTc (normal ≤450ms vs male ≤440ms);
    KCNQ1 and KCNE1 females have higher event rates during menstrual cycle / hormone changes;
    QT-prolonging drugs riskier in females.
  - QUINIDINE IN BRUGADA: class IA sodium channel blocker + ITo blocker → restores J-point;
    quinidine is the only drug with Level C evidence for Brugada arrhythmia prevention;
    used when ICD declined/contraindicated, as bridge, or for supraventricular arrhythmia.

COHORT: 8 × 40 = 320 patient slots (seeds 1294-1301; gene-specific seeds)
"""

import random

SEED_BASE = 1294

ARRHYTHMIA_GENES = [
    # ── KCNQ1 — Long QT Syndrome Type 1 / Jervell-Lange-Nielsen ──────────────
    {
        "gene": "KCNQ1",
        "protein": "Voltage-Gated Potassium Channel Kv7.1 (KCNQ1)",
        "alias": (
            "KCNQ1 (OMIM gene 607542); LQT1 #192500; JLN1 #220400; 11p15.5; 580 aa; ~75 kDa; "
            "AD (LQT1) / AR biallelic (JLN1); most common LQTS gene (~35-40% of genotype-positive LQTS); "
            "alpha subunit of IKs (slow delayed rectifier potassium channel); "
            "IKs = KCNQ1 (alpha) + KCNE1 (MinK beta); Kv7.1 channel repolarises cardiac AP; "
            "LOF variants → prolonged cardiac AP → prolonged QTc; "
            "swim-triggered syncope/SCD = PATHOGNOMONIC trigger for LQT1; "
            "beta-blocker highly effective: ~80% reduction in SCD risk; "
            "JLN syndrome (biallelic): congenital sensorineural deafness + severe QT prolongation"
        ),
        "aa": "580 aa",
        "kDa": "~75 kDa",
        "locus": "11p15.5",
        "omim_gene": 607542,
        "omim_disease": 192500,
        "inheritance": "AD (LQT1) — one pathogenic variant; AR biallelic (Jervell-Lange-Nielsen Type 1) — deafness + severe QT; consanguinity increases JLN risk",
        "seed_offset": 0,
        "onset_range_y": (2.0, 50.0),
        "gene_class": (
            "KCNQ1 encodes the alpha subunit of the cardiac IKs channel (slow delayed rectifier K+ current). "
            "IKs = KCNQ1 (alpha subunit) + KCNE1 (MinK beta subunit) — both required for full function. "
            "IKs activates during the cardiac action potential plateau phase and contributes to "
            "phase 3 repolarisation, especially at high heart rates (rate-adaptive shortening of QT). "
            "KCNQ1 LOF → IKs reduction → prolonged action potential → prolonged QTc on ECG → "
            "torsades de pointes (TdP) → syncope/SCD. "
            "KEY PATHOPHYSIOLOGY OF TRIGGER: during exercise or swimming, sympathetic stimulation "
            "activates IKs via beta-adrenergic signalling; KCNQ1 LOF → IKs cannot respond adequately → "
            "QT paradoxically FAILS TO SHORTEN during increased heart rate → TdP at high HR. "
            "This explains why swimming is particularly dangerous in LQT1 (total body immersion in "
            "cold water triggers dive reflex + vagal tone + adrenergic surge simultaneously). "
            "BETA-BLOCKER MECHANISM: propranolol/nadolol reduce adrenergic stimulation → "
            "prevent failure of QT shortening → highly effective in LQT1 (better than LQT2/LQT3). "
            "JLN SYNDROME (biallelic KCNQ1 LOF): zero IKs → profound QT prolongation (QTc often >550ms); "
            "congenital sensorineural deafness (IKs also expressed in stria vascularis of cochlea). "
            "GENOTYPE-PHENOTYPE: transmembrane domain variants tend to more severe than C-terminal domain."
        ),
        "phenotype": (
            "LQT1 (AD): QTc >460ms females, >440ms males; "
            "exercise-triggered syncope (especially swimming); "
            "T-wave broad and blunted on ECG (LQT1-specific pattern); "
            "cardiac arrest risk higher in males until age 40, then equalises; "
            "SCD risk reduced to ~1-2% with appropriate therapy; penetrance ~40-60%. "
            "JLN1 (AR biallelic): congenital profound sensorineural deafness (bilateral); "
            "QTc typically 550-600ms; very high arrhythmia risk from infancy; "
            "cochlear implant for deafness; ICD likely required."
        ),
        "hallmark": (
            "SWIMMING-TRIGGERED SYNCOPE in QTc-prolonged patient = LQT1 until proven otherwise. "
            "T-WAVE MORPHOLOGY: LQT1 = broad-based blunted T-wave (vs LQT2 = notched bifid; LQT3 = late peaked T). "
            "BETA-BLOCKER SUPERIOR EFFICACY: in LQT1, beta-blockers achieve ~80% SCD prevention vs "
            "~50% in LQT2; explain to patient that adherence to beta-blocker is life-saving. "
            "JERVELL-LANGE-NIELSEN: deaf child + prolonged QTc = JLN until proven otherwise. "
            "Do NOT dismiss syncope during swimming as a near-drowning event — it IS the arrhythmia."
        ),
        "treatment_alerts": [
            "BETA-BLOCKER MANDATORY: nadolol (preferred — long half-life, no hepatic metabolism variation) or propranolol; dose titrate to resting HR 55-60 bpm; NEVER stop abruptly (rebound withdrawal arrhythmia).",
            "SWIMMING PROHIBITED: LQT1 patients must NOT swim unattended; competitive swimming ABSOLUTELY CONTRAINDICATED; supervised pool access may be considered in fully compliant beta-blocked patients on case-by-case basis.",
            "QT-PROLONGING DRUGS: use CredibleMeds (AZCERT) database for every new prescription; avoid ALL known QT-prolonging drugs (antihistamines, azole antifungals, fluoroquinolones, macrolides, antipsychotics, antiemetics).",
            "ICD INDICATIONS: SCD survivor; breakthrough cardiac arrest on adequate beta-blocker; JLN1 syndrome (high risk); symptomatic patients intolerant of or non-adherent to beta-blocker.",
            "MEXILETINE (LQT1): less benefit than in LQT3; not first-line; consider adjunct if QTc remains >500ms on beta-blocker.",
            "PREGNANCY/POSTPARTUM: QTc physiologically shortens in pregnancy but RISK RISES POSTPARTUM; continue beta-blocker throughout pregnancy and increase vigilance in first 9 months postpartum.",
            "FAMILY SCREENING: cascade genetic testing all first-degree relatives; ECG for all relatives; identify asymptomatic gene carriers before their first event.",
        ],
        "key_ddx": (
            "KCNH2 LQT2 (bifid notched T-wave; auditory trigger; drug-induced most common cause); "
            "SCN5A LQT3 (late-peaked T; nocturnal trigger; sodium-channel blocker mexiletine); "
            "KCNE1 LQT5 (same IKs pathway — beta subunit; clinically similar to LQT1; females worse); "
            "ANK2 LQT4 (multimorphic — sinus node + AF + VF; atypical LQTS); "
            "Acquired QT prolongation (electrolytes — hypokalaemia, hypomagnesaemia; drugs; bradycardia; hypothyroidism)."
        ),
    },
    # ── KCNH2 — Long QT Syndrome Type 2 / Drug-Induced LQT ──────────────────
    {
        "gene": "KCNH2",
        "protein": "hERG Channel Kv11.1 (KCNH2 / hERG)",
        "alias": (
            "KCNH2 (OMIM gene 152427); LQT2 #613688; 7q36.1; 1159 aa; ~127 kDa; AD; "
            "second most common LQTS gene (~25-30% of genotype-positive LQTS); "
            "alpha subunit of IKr (rapid delayed rectifier potassium channel); "
            "hERG channel = MOST drug-sensitive cardiac ion channel — unique inactivation gating; "
            "auditory-triggered syncope/SCD = KCNH2/LQT2 hallmark; "
            "BIFID NOTCHED T-WAVE = LQT2 ECG signature; "
            "drug-induced QT prolongation is almost always hERG block; "
            "potassium supplementation (K+ 4.5-5.0 mEq/L target) reduces QT; "
            "nadolol/propranolol; NO class III antiarrhythmics (amiodarone, sotalol AVOIDED)"
        ),
        "aa": "1159 aa",
        "kDa": "~127 kDa",
        "locus": "7q36.1",
        "omim_gene": 152427,
        "omim_disease": 613688,
        "inheritance": "AD (LQT2); de novo possible; reduced penetrance ~40-50%; females have higher event rates",
        "seed_offset": 1,
        "onset_range_y": (5.0, 55.0),
        "gene_class": (
            "KCNH2 encodes the human Ether-à-go-go Related Gene (hERG) channel, the alpha subunit "
            "of the cardiac IKr channel (rapid delayed rectifier potassium current). "
            "hERG UNIQUE PHARMACOLOGY: the channel has an unusually large inner vestibule and "
            "lacks the Pro-Val-Pro motif that protects other Kv channels from drug block → "
            "lipophilic and amphipathic drugs can enter the channel via the activation gate and "
            "become 'trapped' in the closed state — this explains why hERG block by drugs is so common. "
            "HUNDREDS of drugs block hERG: antihistamines (terfenadine, astemizole, loratadine), "
            "antipsychotics (haloperidol, droperidol, sertindole), antiemetics (domperidone, metoclopramide), "
            "fluoroquinolones (sparfloxacin, moxifloxacin), macrolides (erythromycin, clarithromycin), "
            "azole antifungals (fluconazole, ketoconazole), antiarrhythmics (sotalol, amiodarone). "
            "TRIGGER: auditory stimulus (sudden loud noise, alarm, telephone ringing) → adrenergic surge → "
            "combined IKr reduction + sympathetic acceleration → QT fails to shorten → TdP. "
            "POTASSIUM MODULATION: extracellular K+ concentration modulates hERG channel function — "
            "hypokalaemia dramatically worsens IKr and drug-induced block; "
            "supplementing K+ to 4.5-5.0 mEq/L reduces QT and drug-block sensitivity. "
            "FEMALE PREDOMINANCE: females have naturally less IKr reserve → same KCNH2 variant → "
            "longer QTc and higher event rate in females vs males."
        ),
        "phenotype": (
            "LQT2 (AD): QTc >460ms females, >440ms males; "
            "auditory-triggered syncope (alarm clock, doorbell, phone ringing); "
            "bifid notched T-wave on ECG (KCNH2-specific); "
            "females have higher arrhythmic event rates; "
            "drug-induced TdP in KCNH2 carriers from standard-dose medications; "
            "SCD risk reduced with beta-blocker + K+ maintenance + drug avoidance."
        ),
        "hallmark": (
            "BIFID NOTCHED T-WAVE: two distinct peaks to the T-wave on 12-lead ECG (lead II, V5, V6) — "
            "PATHOGNOMONIC ECG finding for LQT2; compare with LQT1 (broad blunted) and LQT3 (late peaked). "
            "AUDITORY TRIGGER: syncope/cardiac arrest provoked by sudden loud noise = LQT2. "
            "BEDSIDE PROTOCOL: patients must silence phone alarms and choose gradual wake-up alarms; "
            "telephone ringing in the night is highest-risk scenario. "
            "DRUG-INDUCED LQT: any QT prolongation on a new drug in a known KCNH2 carrier → "
            "STOP the drug immediately and monitor continuously until QTc normalises. "
            "POTASSIUM TARGET: maintain serum K+ 4.5-5.0 mEq/L (oral K+ supplementation)."
        ),
        "treatment_alerts": [
            "BETA-BLOCKER FIRST-LINE: nadolol or propranolol; effective but less so than in LQT1 (~50% SCD reduction vs ~80% in LQT1); use maximum tolerated dose.",
            "POTASSIUM SUPPLEMENTATION: maintain K+ 4.5-5.0 mEq/L; oral K+ supplements (slow-release KCl); potassium + magnesium (Mg2+ 0.4-0.8 mmol/kg/day) combination reduces TdP risk.",
            "AVOID ALL QT-PROLONGING DRUGS: check CredibleMeds/AZCERT for every new drug; class III antiarrhythmics (sotalol, dofetilide, amiodarone) are relatively CONTRAINDICATED in KCNH2 carriers (all block hERG).",
            "ICD INDICATIONS: SCD survivor; breakthrough cardiac arrest on beta-blocker; very long QTc >500ms with symptoms; JLN-equivalent (rare homozygous LQT2 — near-zero IKr).",
            "ACOUSTIC ENVIRONMENT MANAGEMENT: gradual-wake alarms; no sudden loud noises in bedroom; counsel family members to avoid startling patient from sleep; turn phone to vibrate at night.",
            "MEXILETINE ADJUNCT: reduces late Na+ current which can modestly shorten QT; consider in LQT2 if QTc remains >500ms (off-label, less data than in LQT3).",
            "PREGNANCY: QTc shortens in pregnancy but event risk rises postpartum; beta-blocker throughout and increased monitoring 3-6 months postpartum.",
        ],
        "key_ddx": (
            "KCNQ1 LQT1 (broad blunted T-wave; swimming trigger; beta-blocker more effective); "
            "SCN5A LQT3 (late peaked T; nocturnal trigger; mexiletine specific); "
            "Acquired LQT (drug history, electrolytes, structural heart disease; no KCNH2 variant); "
            "Short QT syndrome KCNH2 GOF (opposite — very short QT <320ms; symmetric peaked T; AF + VF)."
        ),
    },
    # ── SCN5A — LQT3 / Brugada Syndrome / PCCD ───────────────────────────────
    {
        "gene": "SCN5A",
        "protein": "Voltage-Gated Sodium Channel Nav1.5 (SCN5A)",
        "alias": (
            "SCN5A (OMIM gene 600163); LQT3 #603830 / Brugada Syndrome #601144 / PCCD #113900; "
            "3p22.2; 2016 aa; ~227 kDa; AD; "
            "cardiac sodium channel Nav1.5 — generates INa (fast sodium current); "
            "GAIN-OF-FUNCTION (GOF) = LQT3: channel fails to inactivate → persistent late INa → "
            "prolonged AP → QTc prolonged; nocturnal/bradycardia trigger; mexiletine HIGHLY effective; "
            "LOSS-OF-FUNCTION (LOF) = Brugada Syndrome: reduced INa → coved ST elevation V1-V2 → "
            "VF risk; fever ABSOLUTELY CONTRAINDICATED (unmasks/worsens Brugada); "
            "LOF = PCCD (Progressive Cardiac Conduction Disease): AV block, bundle branch block; "
            "one SCN5A gene can cause LQT3, Brugada, AND PCCD in same family (overlap syndrome)"
        ),
        "aa": "2016 aa",
        "kDa": "~227 kDa",
        "locus": "3p22.2",
        "omim_gene": 600163,
        "omim_disease": 601144,
        "inheritance": "AD; de novo common; reduced penetrance; highly variable expressivity even within families; male predominance in Brugada (testosterone modulates Nav1.5 expression); overlap phenotypes",
        "seed_offset": 2,
        "onset_range_y": (0.0, 60.0),
        "gene_class": (
            "SCN5A encodes the alpha subunit of the cardiac voltage-gated sodium channel Nav1.5, "
            "the primary ion channel responsible for the rapid upstroke (phase 0) of the cardiac action potential. "
            "Nav1.5 generates INa — a large inward sodium current that depolarises the cardiomyocyte "
            "within milliseconds and drives rapid conduction through working myocardium, Purkinje fibres, and AV node. "
            "THREE DISTINCT PHENOTYPES from SCN5A variants: "
            "1. LQT3 (GOF): defective inactivation → persistent late INa (INa-L) throughout plateau phase → "
            "action potential prolongation → QTc prolonged → TdP at slow heart rates; "
            "nocturnal/bradycardia trigger (opposite to LQT1/2); mexiletine blocks INa-L selectively. "
            "2. BRUGADA SYNDROME (LOF): reduced peak INa → reduced phase 1 notch/dome → "
            "ST elevation V1-V3 (coved type) → ventricular fibrillation; "
            "fever dramatically worsens Brugada (temperature accelerates Nav1.5 inactivation → more LOF); "
            "male predominance (testosterone downregulates Nav1.5 in right ventricular epicardium). "
            "3. PCCD (LOF): conduction system Nav1.5 → AV node + His-Purkinje conduction slowing → "
            "progressive AV block (may require pacemaker); first-degree → LBBB → complete AV block; "
            "often co-exists with Brugada pattern. "
            "OVERLAP SYNDROMES: same SCN5A family can have Brugada + PCCD + LQT3 (common!) — "
            "genotype cannot reliably predict which phenotype predominates."
        ),
        "phenotype": (
            "LQT3: QTc prolonged; nocturnal/bradycardia-triggered TdP → syncope/SCD; "
            "late-peaked asymmetric T-wave on ECG (QT appears long mainly due to late T); "
            "SCD risk at rest/sleep higher than LQT1/2 (less common events but higher fatality per event). "
            "Brugada Syndrome: spontaneous or drug/fever-induced coved ST elevation V1-V2 ≥2mm; "
            "VF episodes characteristically nocturnal (Asian males); "
            "RBBB morphology; may be asymptomatic for decades; "
            "fever unmasks in 30-40% of previously concealed Brugada patients. "
            "PCCD: progressive AV block (first-degree → Wenckebach → complete); "
            "LBBB/RBBB alternating; syncope from high-grade AV block; pacemaker required."
        ),
        "hallmark": (
            "BRUGADA: COVED ST ELEVATION ≥2mm in V1-V2 with RBBB morphology — "
            "Type 1 (spontaneous or drug-provoked) is DIAGNOSTIC; Type 2 (saddle-back) requires provocation. "
            "FEVER ABSOLUTE RULE in Brugada: temperature ≥38°C → administer antipyretic (paracetamol) IMMEDIATELY; "
            "do NOT allow fever to persist; hospital admission if fever uncontrolled; "
            "WRITTEN EMERGENCY PLAN given to ALL Brugada patients. "
            "LQT3 vs LQT1/2: LQT3 is uniquely sodium-channel driven → mexiletine (Na+ blocker) shortens QTc "
            "dramatically in LQT3 (use as diagnostic test: 200-300mg oral → measure QT at 2h). "
            "OVERLAP: check for PR prolongation and RBBB in any LQTS patient → may have co-existing PCCD/Brugada."
        ),
        "treatment_alerts": [
            "LQT3: MEXILETINE is FIRST-LINE adjunct — 200-400mg TDS; shortens QTc by 20-50ms in LQT3 (sodium channel blocker suppresses persistent late INa); monitor QRS widening (stop if QRS >130% baseline).",
            "LQT3 BETA-BLOCKER LESS EFFECTIVE: beta-blockers reduce adrenergic triggers but LQT3 trigger is bradycardia/sleep → rate-slowing may paradoxically worsen risk; use with caution; consider pacemaker to prevent bradycardia.",
            "BRUGADA ICD: symptomatic Brugada (VF survivor or sustained VT) = Class I ICD indication; asymptomatic Brugada with spontaneous Type 1 = Class IIa; asymptomatic Brugada drug-provoked only = controversial.",
            "BRUGADA QUINIDINE: class IA antiarrhythmic + ITo blocker; quinidine restores J-point and suppresses VF in Brugada; level C evidence; use when ICD declined/contraindicated; oral 600-1500mg/day.",
            "BRUGADA FEVER PROTOCOL: written action plan; paracetamol 1g at first sign of fever; present to ER if fever >38.5°C or unresponsive to antipyretic; avoid recreational drugs, cocaine, cannabis (all unmask Brugada).",
            "BRUGADA AVOID: sodium channel blockers (ajmaline, flecainide, propafenone) are used diagnostically but NOT therapeutically; beta-blockers ineffective and not indicated for Brugada arrhythmia.",
            "PCCD PACEMAKER: if symptomatic high-grade AV block or syncope from PCCD component → DDD pacemaker; monitor yearly ECG for conduction deterioration.",
        ],
        "key_ddx": (
            "Early repolarisation syndrome (ERS — J-point elevation V4-V6, inferior leads; less arrhythmic but overlap with Brugada); "
            "Acquired Brugada pattern (fever, cocaine, flecainide, tricyclic overdose — provoked pattern in non-SCN5A individuals); "
            "ARVC/ARVD (epsilon waves, RV dilation on MRI, desmosomal genes); "
            "KCNQ1/KCNH2 LQTS (T-wave morphology distinguishes; trigger pattern different); "
            "Ischemic RBBB + ST elevation (acute STEMI — coronary angiography distinguishes)."
        ),
    },
    # ── RYR2 — CPVT Type 1 ───────────────────────────────────────────────────
    {
        "gene": "RYR2",
        "protein": "Ryanodine Receptor 2 (RYR2 / Cardiac Ryanodine Receptor)",
        "alias": (
            "RYR2 (OMIM gene 180902); CPVT1 Catecholaminergic Polymorphic VT #604772; 1q43; "
            "4967 aa; ~560 kDa (largest human protein in clinical genetics); AD; "
            "cardiac SR calcium-release channel; RYR2 GOF → diastolic SR Ca2+ leak → "
            "delayed afterdepolarisations (DADs) → triggered activity → BIDIRECTIONAL VT → VF; "
            "trigger is ALWAYS catecholamine surge (exercise/emotion — NEVER at rest); "
            "nadolol PREFERRED over metoprolol (superior HR reduction + beta-1 selectivity avoidance); "
            "flecainide ADJUNCT (blocks open RYR2 channel); "
            "ICD shock → sympathetic storm → triggers more VT = PARADOXICAL WORSENING"
        ),
        "aa": "4967 aa",
        "kDa": "~560 kDa",
        "locus": "1q43",
        "omim_gene": 180902,
        "omim_disease": 604772,
        "inheritance": "AD; de novo variants common (~30%); penetrance 60-80%; highly variable — same variant can have very different phenotypes in different family members",
        "seed_offset": 3,
        "onset_range_y": (3.0, 40.0),
        "gene_class": (
            "RYR2 encodes the cardiac ryanodine receptor (RyR2), a massive homo-tetrameric Ca2+-release "
            "channel located in the sarcoplasmic reticulum (SR) membrane of cardiomyocytes. "
            "RyR2 mediates calcium-induced calcium release (CICR) — the trigger for cardiac contraction. "
            "CPVT MECHANISM: RYR2 GOF → pathological diastolic Ca2+ leak from SR → "
            "cytoplasmic [Ca2+] rises in diastole → NCX (sodium-calcium exchanger, NCX1) "
            "extruding Ca2+ generates inward Na+ current → delayed afterdepolarisations (DADs) → "
            "triggered action potentials → ventricular ectopy → bidirectional VT → VF. "
            "CATECHOLAMINE SENSITIVITY: PKA phosphorylates RYR2 at Ser2808/Ser2814 (via beta-adrenergic "
            "signalling) → increases channel open probability; GOF variants further sensitise channel → "
            "SR leak only occurs when PKA activity is high (exercise, emotion, catecholamines) → "
            "explains why CPVT NEVER occurs at rest in classic CPVT1. "
            "BIDIRECTIONAL VT: the pathognomonic arrhythmia — alternating QRS axis (+60° / -120°) "
            "from alternating fascicular origin; also polymorphic VT is seen. "
            "FLECAINIDE MECHANISM IN CPVT: flecainide is a class IC sodium channel blocker that "
            "also directly blocks RYR2 in the open state → reduces SR calcium leak → "
            "prevents DADs → prevents triggered activity; this mechanism is independent of Na+ channel block."
        ),
        "phenotype": (
            "CPVT1: exercise-triggered or emotional-stress-triggered syncope/SCD in young patients; "
            "NO structural heart disease; ECG at rest often NORMAL (no QT prolongation, no ST changes); "
            "bidirectional VT on exercise stress test (hallmark); "
            "polymorphic VT; onset age 3-20 years (median ~8 years); "
            "untreated 5-year mortality ~30%; "
            "ICD paradox — shocks worsen by triggering catecholamine surge → arrhythmia storm."
        ),
        "hallmark": (
            "BIDIRECTIONAL VT during exercise or emotional stress in a child/young adult with "
            "NORMAL RESTING ECG = CPVT1 until proven otherwise. "
            "EXERCISE STRESS TEST: CPVT characteristic progression — ventricular ectopy at low workload → "
            "bigeminy → bidirectional VT → polymorphic VT as HR rises above threshold. "
            "EMOTIONAL TRIGGER: emotional stress (anger, fright) equally potent as exercise — "
            "specifically counsel about anger management, competitive sport social stress. "
            "DRUG TEST: epinephrine infusion (0.05-0.3 μg/kg/min) provokes VT in CPVT patients in EP lab — "
            "diagnostic when exercise test equivocal (use with extreme caution + defibrillation ready)."
        ),
        "treatment_alerts": [
            "NADOLOL PREFERRED: non-selective beta-blocker (nadolol 1-2.5 mg/kg/day) preferred over selective beta-1-blockers (metoprolol); non-selective beta-blockade better suppresses adrenergic facilitation of RYR2 SR leak; titrate to prevent exercise-induced ectopy on ETT.",
            "FLECAINIDE ADJUNCT: 100-300mg/day in 2 doses; directly blocks open RYR2 channel; reduces VT burden by ~75% as adjunct to nadolol; now standard combination in high-risk CPVT1.",
            "ICD PARADOX: shock → pain + fear → catecholamine release → more DADs → VT storm; programme ICD with HIGH detection rates and long detection time to allow spontaneous termination; ALWAYS maximise beta-blocker before ICD; consider flecainide to reduce shock burden.",
            "LEFT CARDIAC SYMPATHETIC DENERVATION (LCSD): surgical/thoracoscopic denervation of left stellate ganglion; reduces NE release to heart by ~75%; effective when drugs fail or ICD storms recur; NOT a cure but highly effective adjunct.",
            "ACTIVITY RESTRICTION: competitive sports PROHIBITED; recreational swimming PROHIBITED (dive reflex + emotional tension); emotion-provoking environments (combat games, extreme sports) avoided; cognitive-behavioural therapy for anger management.",
            "SCREENING: exercise stress test all first-degree relatives; Holter monitor + genetic cascade testing; even asymptomatic gene-positive children require treatment.",
            "AVOID: epinephrine injection for dental procedures (use mepivacaine without vasoconstrictor); epinephrine in anaesthetic adrenaline; any catecholamine infusion triggers VT.",
        ],
        "key_ddx": (
            "CASQ2 CPVT2 (AR biallelic — more severe, earlier onset, same bidirectional VT phenotype; CASQ2 sequencing); "
            "Idiopathic Ventricular Fibrillation (no bidirectional VT; normal RYR2 gene); "
            "Hypertrophic Cardiomyopathy with VT (structural disease on ECHO/MRI; HCM genes); "
            "Long QT Syndrome with exercise TdP (QTc prolonged at rest; TdP not bidirectional); "
            "Andersen-Tawil Syndrome KCNJ2 (bidirectional VT + hypokalaemia + dysmorphic features; very long QU interval)."
        ),
    },
    # ── CASQ2 — CPVT Type 2 ──────────────────────────────────────────────────
    {
        "gene": "CASQ2",
        "protein": "Calsequestrin 2 (CASQ2 / Cardiac Calsequestrin)",
        "alias": (
            "CASQ2 (OMIM gene 114251); CPVT2 #611938; 1p13.3-p11; 399 aa; ~46 kDa; AR (biallelic); "
            "calsequestrin-2 is the primary SR calcium-storage protein in cardiomyocytes; "
            "CASQ2 buffers free SR Ca2+ and regulates RYR2 open probability via triadin-junctin interaction; "
            "CASQ2 LOF → SR Ca2+ buffering REDUCED → SR more 'hair-trigger' → same diastolic Ca2+ leak → "
            "bidirectional VT as in CPVT1 but often MORE SEVERE and EARLIER ONSET; "
            "biallelic mutations required (AR) → consanguinity increases risk; "
            "Bedouin and Middle Eastern founder variants; "
            "nadolol + flecainide mandatory; ICD high risk of storm in CASQ2 too"
        ),
        "aa": "399 aa",
        "kDa": "~46 kDa",
        "locus": "1p13.3-p11",
        "omim_gene": 114251,
        "omim_disease": 611938,
        "inheritance": "AR (biallelic); heterozygous carriers usually asymptomatic (some exceptions); consanguinity common in affected families; Bedouin and Yemenite Jewish founder variants",
        "seed_offset": 4,
        "onset_range_y": (1.0, 20.0),
        "gene_class": (
            "CASQ2 encodes calsequestrin-2, the major low-affinity high-capacity calcium-binding protein "
            "in the sarcoplasmic reticulum (SR) lumen of cardiomyocytes. "
            "CASQ2 FUNCTION: buffers SR Ca2+ (each molecule binds ~20-40 Ca2+ ions) → maintains "
            "high total SR Ca2+ stores without high free [Ca2+]; forms a polymer network in SR lumen; "
            "interacts with the RYR2 channel complex via triadin and junctin → when SR Ca2+ is high, "
            "CASQ2 signals RYR2 to reduce open probability (feedback inhibition). "
            "CASQ2 LOF → SR Ca2+ buffering impaired → free SR [Ca2+] elevated at rest → "
            "RYR2 open probability increased (CASQ2 no longer applying feedback inhibition) → "
            "diastolic SR Ca2+ leak → delayed afterdepolarisations → bidirectional VT. "
            "SEVERITY: CPVT2 (CASQ2) tends to be MORE SEVERE than CPVT1 (RYR2): "
            "earlier onset (infancy to early childhood); more frequent cardiac events; "
            "higher risk of SCD; biallelic state means ZERO residual CASQ2 function. "
            "TRIADIN INTERACTION: CASQ2 LOF disrupts the CASQ2-triadin-junctin-RYR2 macromolecular "
            "complex → triadin is destabilised → triadin-knockout phenotype also produces CPVT-like "
            "disease (TRDN gene — CPVT5)."
        ),
        "phenotype": (
            "CPVT2: severe exercise/catecholamine-triggered bidirectional and polymorphic VT; "
            "onset often earlier than CPVT1 (infancy to age 10 common); "
            "same trigger pattern (exercise + emotional stress) as CPVT1; "
            "normal resting ECG, normal QTc; "
            "high event rate despite therapy; SCD risk high if undertreated; "
            "defibrillation/AED required at school and home for severe cases."
        ),
        "hallmark": (
            "CPVT2 = CASQ2 AR BIALLELIC: confirm biallelic variants — heterozygous CASQ2 variant alone "
            "is usually benign; sequence both alleles. "
            "CONSANGUINITY HISTORY: Bedouin/Middle Eastern consanguineous family + bidirectional VT in child = "
            "CASQ2 CPVT2 until proven otherwise. "
            "SEVERITY vs CPVT1: CASQ2 patients often have more severe arrhythmia burden — higher VT frequency, "
            "earlier age at first event; require more aggressive management. "
            "TRIADIN MUTATION (TRDN CPVT5): similar phenotype to CASQ2 — check TRDN sequencing if CASQ2 negative."
        ),
        "treatment_alerts": [
            "NADOLOL + FLECAINIDE COMBINATION MANDATORY: CPVT2 more severe — do NOT wait; start both from diagnosis; titrate nadolol to maximum tolerated dose (exercise test monitoring).",
            "ICD RISK IN CPVT2: ICD storm risk very high; programme conservatively (high rate cut-off, long detection window); LCSD (left cardiac sympathetic denervation) strongly recommended as ICD adjunct in CPVT2.",
            "EMERGENCY AED: AED at home + school in all symptomatic CPVT2 children; family trained in CPR; school liaison for emergency management plan.",
            "GENE THERAPY TRIAL: RYR2-directed antisense oligonucleotide and AAV-CASQ2 trials in preclinical/early clinical development; enrol in trials if available.",
            "SIBLINGS: 25% risk for each sibling; exercise stress test + CASQ2 sequencing in all siblings; pre-symptomatic therapy if gene-positive.",
            "AVOID SAME AS CPVT1: catecholamines; epinephrine; competitive sport; swimming unattended; anger-provoking environments.",
            "SYMPATHETIC DENERVATION (LCSD): strongly recommended in CASQ2 CPVT2 with recurrent events or ICD storms; remove left stellate ganglion branches T1-T4 thoracoscopic; expert cardiac surgery centre.",
        ],
        "key_ddx": (
            "RYR2 CPVT1 (AD dominant inheritance; milder than CASQ2; single heterozygous variant sufficient); "
            "TRDN CPVT5 (triadin; AR; same phenotype; TRDN sequencing); "
            "Andersen-Tawil Syndrome KCNJ2 (bidirectional VT + periodic paralysis + dysmorphic features); "
            "Structurally abnormal heart mimicking CPVT (ARVC — desmosomal; HCM — MYH7/MYBPC3; exclude with MRI)."
        ),
    },
    # ── HCN4 — Familial Sinus Node Disease / Hereditary Bradycardia ──────────
    {
        "gene": "HCN4",
        "protein": "Hyperpolarisation-Activated Cyclic Nucleotide-Gated Channel 4 (HCN4)",
        "alias": (
            "HCN4 (OMIM gene 605206); Sick Sinus Syndrome 2 #163800; 15q24.1; 1203 aa; ~136 kDa; AD; "
            "pacemaker channel — generates If (funny current) in SA node and AV node; "
            "HCN4 LOF → reduced If → impaired automaticity → sinus bradycardia/pauses/arrest; "
            "ivabradine ABSOLUTELY CONTRAINDICATED (directly blocks HCN4 — further suppresses If); "
            "beta-blockers CONTRAINDICATED or use with extreme caution (further slow SAN); "
            "permanent pacemaker (DDD or DDDR) often required; "
            "some HCN4 variants cause atrial fibrillation + bradycardia (bradycardia-tachycardia syndrome); "
            "Brugada-like ST pattern in some HCN4 LOF patients (overlap)"
        ),
        "aa": "1203 aa",
        "kDa": "~136 kDa",
        "locus": "15q24.1",
        "omim_gene": 605206,
        "omim_disease": 163800,
        "inheritance": "AD; variable penetrance; de novo described; some families with partial penetrance where only some members clinically affected despite genotype",
        "seed_offset": 5,
        "onset_range_y": (5.0, 60.0),
        "gene_class": (
            "HCN4 encodes the hyperpolarisation-activated cyclic nucleotide-gated channel 4, "
            "the primary channel generating the cardiac pacemaker current (If — 'funny' current). "
            "If is an inward depolarising mixed Na+/K+ current activated by membrane hyperpolarisation "
            "(opposite to most voltage-gated channels) → responsible for spontaneous phase 4 "
            "depolarisation (pacemaker potential) in sinoatrial (SA) and atrioventricular (AV) node cells. "
            "HCN4 REGULATION: cyclic AMP (cAMP) binds the C-terminal CNBD (cyclic nucleotide-binding domain) → "
            "shifts activation curve positively → increases If → increases pacemaker rate; "
            "beta-adrenergic stimulation → cAMP increase → faster SA rate; "
            "vagal stimulation → cAMP decrease → slower SA rate. "
            "HCN4 LOF → reduced If → less spontaneous depolarisation → sinus bradycardia; "
            "inadequate rate response to exercise; sinus pauses; sinus arrest; junctional escape. "
            "ATRIAL FIBRILLATION ASSOCIATION: HCN4 variants associate with AF — mechanism unclear; "
            "possibly from atrial electrophysiological remodelling secondary to slow sinus rate or "
            "direct HCN4 expression in atrial myocardium. "
            "IVABRADINE MECHANISM: ivabradine directly blocks HCN4 channel → reduces If → slows HR; "
            "in HCN4 LOF patients, ivabradine would further suppress already-deficient pacemaker activity → "
            "ABSOLUTE CONTRAINDICATION."
        ),
        "phenotype": (
            "Familial Sick Sinus Syndrome: sinus bradycardia (resting HR 30-50 bpm typical); "
            "chronotropic incompetence (HR fails to increase with exercise); "
            "sinus pauses/arrests (may cause presyncope, syncope); "
            "junctional escape rhythm; "
            "bradycardia-tachycardia syndrome (alternating bradycardia + AF or atrial flutter); "
            "exertional dyspnea from chronotropic incompetence; "
            "AF episodes with long pauses on cardioversion. "
            "Brugada-like pattern: some HCN4 variants show V1-V2 J-point elevation pattern — "
            "distinguish from true Brugada by lack of fever inducibility."
        ),
        "hallmark": (
            "HEREDITARY SINUS BRADYCARDIA: documented resting HR <45 bpm in multiple family members + "
            "AD inheritance pattern + symptom onset before age 40 = HCN4 until proven otherwise. "
            "CHRONOTROPIC INCOMPETENCE on exercise test: HR fails to reach 85% of predicted maximum; "
            "subjective fatigue and dyspnea during modest exertion. "
            "IVABRADINE CONTRAINDICATED: ivabradine was developed specifically as an HCN4 blocker; "
            "giving it to an HCN4 LOF patient = compounding the molecular defect → asystole risk; "
            "MARK CLEARLY in drug allergy/contraindication field of patient record."
        ),
        "treatment_alerts": [
            "PACEMAKER INDICATION: symptomatic sinus bradycardia (syncope/presyncope), HR <40 bpm at rest, sinus pauses >3s, chronotropic incompetence with functional limitation → DDD pacemaker; rate-response (DDDR) for chronotropic incompetence.",
            "IVABRADINE ABSOLUTE CONTRAINDICATION: directly blocks HCN4; will worsen bradycardia in HCN4 LOF patients; note prominently in EPR; contraindication applies even if patient has concurrent AF (other indications for ivabradine are irrelevant here).",
            "BETA-BLOCKERS: generally AVOID or use with extreme caution; slow SA rate further; if required for concurrent AF/HCM, monitor HR closely and consider pacemaker backup first.",
            "AF MANAGEMENT: rhythm control (cardioversion + antiarrhythmic) preferred; avoid rate-slowing drugs without pacemaker backup; flecainide is relatively safe if no structural disease and no Brugada overlap.",
            "SPORTS PARTICIPATION: competitive sport with chronotropic incompetence may be limited; assess with formal exercise test; pacemaker enables more active lifestyle.",
            "HOLTER MONITORING: annual 24-48h Holter; look for asymptomatic long pauses (>3s) that may require pacemaker upgrade before symptoms develop.",
        ],
        "key_ddx": (
            "Acquired sinus node disease (fibrosis — older patients; inferior STEMI; radiation; drug-induced: beta-blockers, diltiazem, digoxin); "
            "Autonomic sinus bradycardia (athletic bradycardia — HR recovers with exercise; no structural/genetic cause); "
            "SCN5A PCCD (progressive conduction disease — AV block + bundle branch block more prominent; sinus node often later); "
            "Hypothyroidism (TSH elevated; reversible bradycardia)."
        ),
    },
    # ── ANK2 — LQT4 / Ankyrin-B Syndrome ────────────────────────────────────
    {
        "gene": "ANK2",
        "protein": "Ankyrin-B (ANK2 / ANKB)",
        "alias": (
            "ANK2 (OMIM gene 106410); LQT4 / Ankyrin-B Syndrome #600919; 4q25; "
            "3952 aa; ~441 kDa; AD; "
            "ankyrin-B is a scaffolding protein that anchors ion transporters to the "
            "cardiac T-tubule/SR membrane — specifically NCX1 (Na/Ca exchanger), "
            "Na/K-ATPase, and the InsP3 receptor; "
            "ANK2 LOF → NCX1 and NaK-ATPase mislocalised → altered [Ca2+]i → "
            "MULTIMORPHIC phenotype: sinus node dysfunction + AF + atrial flutter + "
            "exercise-induced VF + sudden death; "
            "CPVT-like episodes in some cases; QTc prolonged (but variable, not always prominent); "
            "not captured on standard 4-gene LQTS panels — ANK2 must be included"
        ),
        "aa": "3952 aa",
        "kDa": "~441 kDa",
        "locus": "4q25",
        "omim_gene": 106410,
        "omim_disease": 600919,
        "inheritance": "AD; significant intra-familial variability; same variant can cause SND in one family member and VF in another; penetrance 60-80%",
        "seed_offset": 6,
        "onset_range_y": (0.0, 50.0),
        "gene_class": (
            "ANK2 encodes Ankyrin-B, a member of the ankyrin adaptor protein family that coordinates "
            "the subcellular localisation of ion channels, transporters, and signalling molecules "
            "to specialised membrane domains in cardiomyocytes. "
            "ANK2 TARGETS in cardiomyocytes: "
            "(1) NCX1 (sodium-calcium exchanger, SLC8A1): anchors NCX1 to the T-tubule/SR junction → "
            "critical for Ca2+ extrusion and local SR refilling; "
            "(2) Na/K-ATPase (NKA, ATP1A1): anchors at lateral membrane for Na+ homeostasis; "
            "(3) Inositol 1,4,5-trisphosphate receptor (InsP3R): SR Ca2+ release in response to IP3. "
            "ANK2 LOF → NCX1 + NKA mislocalised → local [Na+] and [Ca2+] dysregulation → "
            "altered SR Ca2+ loading → diastolic Ca2+ oscillations → DADs → triggered arrhythmia. "
            "PHENOTYPIC SPECTRUM (multimorphic — same gene, different manifestations): "
            "Sinus node dysfunction (brady); "
            "Atrial fibrillation (most common — ~40% of ANK2 carriers); "
            "Atrial flutter; "
            "Exercise-induced VT/VF (CPVT-like — catecholamine-triggered); "
            "Sudden cardiac death (exercise or rest). "
            "CLINICAL DISTINCTION from other LQTS: ANK2 LQT4 does NOT always show obvious QTc prolongation; "
            "QTU abnormality or variable QT; captured by comprehensive cardiac gene panels, NOT by standard LQTS4-gene panels."
        ),
        "phenotype": (
            "Ankyrin-B syndrome: heterogeneous family phenotype; "
            "sinus bradycardia in some members; AF in others; VF in others; SCD; "
            "exercise-triggered VT/VF (CPVT-like) in a subset; "
            "QTc may be prolonged (hence LQT4 designation) but often only mildly so; "
            "can present as unexplained SCD in young adult or as late-onset AF; "
            "comprehensive arrhythmia workup needed."
        ),
        "hallmark": (
            "MULTIMORPHIC FAMILY PHENOTYPE: family with some members having sinus node disease, "
            "others AF, others VF/SCD — all from same ANK2 variant. "
            "NOT CAPTURED by 4-gene LQTS panels: ANK2 should be included in comprehensive "
            "cardiac arrhythmia gene panels (including LQTS, Brugada, CPVT, SND genes). "
            "EXERCISE VF in ANK2: catecholamine-triggered — may resemble CPVT; distinguish by RYR2/CASQ2 negative + ANK2 positive."
        ),
        "treatment_alerts": [
            "BETA-BLOCKER: for exercise-induced VT component; nadolol preferred; similar rationale to CPVT.",
            "ICD: any ANK2 patient with documented sustained VT/VF or SCD survivor requires ICD; low threshold for ICD given multimorphic sudden death risk.",
            "AF MANAGEMENT: rate control (cautious with bradycardia risk); rhythm control; anticoagulation per CHA2DS2-VASc (standard threshold — genetic arrhythmia does NOT confer AF score but stroke risk is real).",
            "PACEMAKER: if symptomatic sinus node dysfunction component (syncope from pauses).",
            "COMPREHENSIVE GENE PANEL: ensure ANK2 sequencing included in panel; do not accept negative result from limited LQTS1-4 panel if ANK2 not included.",
            "FAMILY CASCADE: multimorphic presentation means asymptomatic gene-positive relatives could be at VF risk; exercise stress test + Holter all first-degree relatives.",
        ],
        "key_ddx": (
            "CPVT1 RYR2 / CPVT2 CASQ2 (exercise VT bidirectional — RYR2/CASQ2 sequencing); "
            "Isolated AF (no SCD or VT or SND pattern; standard AF genes — KCNQ1 gain-of-function, SCN5A, PITX2); "
            "Brugada syndrome SCN5A (coved ST elevation V1-V2; fever provocation; no sinus node issue); "
            "HCN4 sinus node disease (pure bradycardia, no VF/AF typically)."
        ),
    },
    # ── KCNE1 — LQT5 / Jervell-Lange-Nielsen Type 2 ─────────────────────────
    {
        "gene": "KCNE1",
        "protein": "MinK Beta Subunit of IKs (KCNE1 / MinK)",
        "alias": (
            "KCNE1 (OMIM gene 176261); LQT5 #613695 / JLN2 #612347; 21q22.12; 129 aa; ~15 kDa; AD/AR; "
            "MinK (minimal K channel) is the essential beta subunit of IKs — "
            "KCNQ1 (alpha) + KCNE1 (MinK beta) together form the functional IKs channel; "
            "KCNE1 LOF → IKs reduced → same QT prolongation mechanism as KCNQ1 (LQT1); "
            "females DISPROPORTIONATELY affected (IKs reduction plus naturally longer QTc); "
            "JLN2 (biallelic KCNE1) = deafness + severe QT — same cochlear expression as KCNQ1; "
            "beta-blocker first-line; avoid QT-prolonging drugs; "
            "note: KCNE2 (MiRP1) is a different subunit — LQT6 (beta subunit of IKr)"
        ),
        "aa": "129 aa",
        "kDa": "~15 kDa",
        "locus": "21q22.12",
        "omim_gene": 176261,
        "omim_disease": 613695,
        "inheritance": "AD (LQT5) or AR biallelic (JLN2 — deafness + severe QT); sex hormone modulates IKs — females have higher QTc and event rate at equivalent KCNE1 haploinsufficiency",
        "seed_offset": 7,
        "onset_range_y": (2.0, 50.0),
        "gene_class": (
            "KCNE1 encodes MinK (minimal potassium channel, also called ISK), a single-transmembrane-domain "
            "accessory (beta) subunit that assembles with KCNQ1 alpha subunits to form the IKs channel. "
            "KCNE1 ROLE IN IKs: MinK dramatically alters gating kinetics of KCNQ1 — "
            "MinK slows IKs activation, shifts voltage dependence, increases single-channel conductance, "
            "and mediates beta-adrenergic regulation via PKA phosphorylation. "
            "Without MinK, KCNQ1 alone has much faster activation gating and is non-physiological. "
            "KCNE1 LOF → IKs channel dysfunctional → repolarisation prolonged → QTc prolonged. "
            "SEX DIFFERENCE MECHANISM: sex steroids regulate KCNQ1 + KCNE1 expression and trafficking; "
            "testosterone increases IKs expression → men have shorter QTc (more reserve); "
            "oestrogen has complex effects; progesterone reduces IKs; "
            "net result: females have naturally less IKs 'reserve' → same KCNE1 variant → "
            "higher QTc prolongation and higher event rate in females (particularly during "
            "luteal phase and after puberty). "
            "JLN2 (biallelic KCNE1): cochlea also expresses IKs (KCNQ1 + KCNE1) in stria vascularis; "
            "biallelic KCNE1 LOF → absent IKs in cochlea → endocochlear potential failure → "
            "congenital sensorineural deafness (same mechanism as JLN1 from biallelic KCNQ1). "
            "TRISOMY 21 ASSOCIATION: KCNE1 located at 21q22.12 (Down syndrome chromosome); "
            "trisomy 21 increases KCNE1 copy number → IKs increases → shorter QTc; "
            "trisomic chromosome may 'protect' against LQT5 from the extra KCNE1 copy."
        ),
        "phenotype": (
            "LQT5 (AD KCNE1 LOF): QTc prolonged >460ms females, >440ms males; "
            "syncope/SCD exertion or stress-triggered (similar to LQT1 — same IKs pathway); "
            "females have higher symptom rate per genotype; "
            "T-wave morphology similar to LQT1 (broad blunted). "
            "JLN2 (AR biallelic): congenital bilateral profound sensorineural deafness; "
            "very prolonged QTc (often >550ms); high-risk arrhythmia from infancy; "
            "cochlear implant for deafness management; ICD plus beta-blocker usually required."
        ),
        "hallmark": (
            "FEMALE-PREDOMINANT LQT: same IKs pathway as KCNQ1 (LQT1); females disproportionately affected; "
            "review family history for asymmetric sex distribution of symptoms. "
            "JLN2: deaf child + severe QTc prolongation + consanguinity → KCNE1 biallelic sequencing + KCNQ1 biallelic sequencing; "
            "both genes needed in JLN diagnostic panel. "
            "TRISOMY 21 NOTE: Down syndrome patients on 21q chromosome have extra KCNE1; "
            "their QTc tends shorter; this also means a KCNE1 LOF variant in a person with Down syndrome "
            "may be partially compensated — but do not rely on this."
        ),
        "treatment_alerts": [
            "BETA-BLOCKER FIRST-LINE: nadolol preferred; same efficacy as in LQT1 (~80% SCD risk reduction); long half-life prevents missed-dose arrhythmia; compliance critical.",
            "AVOID QT-PROLONGING DRUGS: IKs is the repolarisation reserve — any additional QT prolongation from drugs is particularly dangerous in KCNE1 LOF patients; use CredibleMeds list.",
            "FEMALES — HEIGHTENED VIGILANCE: puberty onwards → increased arrhythmia risk; oral contraceptive pill choice: norgestimate-based OCP does not worsen QT; avoid oestrogen-only preparations (some progestins help, some worsen).",
            "JLN2 MANAGEMENT: cochlear implant for deafness (improves quality of life dramatically); ICD for arrhythmia; beta-blocker maximally tolerated; cardiac genetics specialist coordination.",
            "HORMONAL CYCLE MONITORING: in premenopausal women with LQT5, symptom diary linked to menstrual cycle; Holter monitoring around luteal phase if symptoms correlate.",
            "ELECTROLYTE MANAGEMENT: maintain K+ >4.0 mEq/L and Mg2+ >0.8 mmol/L (IKs reduction compounded by hypokalaemia/hypomagnesaemia).",
        ],
        "key_ddx": (
            "KCNQ1 LQT1 (alpha subunit of same IKs channel; clinically similar; phenotype identical; distinguish by gene sequencing); "
            "KCNH2 LQT2 (IKr not IKs; bifid T-wave; auditory trigger; drug-induced most common); "
            "SCN5A LQT3 (sodium channel; nocturnal; mexiletine specific); "
            "JLN1 (biallelic KCNQ1) vs JLN2 (biallelic KCNE1): both deaf + severe QT; distinguish by sequencing."
        ),
    },
]


def _make_cohort(gene_dict: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    gene = gene_dict["gene"]
    onset_lo, onset_hi = gene_dict.get("onset_range_y", (0.0, 40.0))

    GENE_PROPS = {
        "KCNQ1": {
            "subtypes": [("LQT1 Symptomatic", 0.55), ("LQT1 Asymptomatic Gene-Positive", 0.30), ("JLN1 Biallelic", 0.15)],
            "treatments": [("Nadolol BB", 0.55), ("Propranolol BB", 0.25), ("Nadolol + ICD", 0.12), ("JLN1: Nadolol + CI + ICD", 0.08)],
            "outcomes": [("Event-free on BB", 0.60), ("Breakthrough syncope", 0.15), ("Pacemaker implanted", 0.08), ("ICD implanted", 0.12), ("SCD survivor", 0.05)],
        },
        "KCNH2": {
            "subtypes": [("LQT2 Symptomatic", 0.50), ("LQT2 Asymptomatic Gene-Positive", 0.28), ("Drug-Induced TdP on KCNH2", 0.22)],
            "treatments": [("Nadolol + K+ supplementation", 0.50), ("Propranolol + K+", 0.20), ("Nadolol + ICD", 0.18), ("Drug cessation + monitoring", 0.12)],
            "outcomes": [("Event-free", 0.55), ("Drug-induced TdP resolved", 0.18), ("Breakthrough syncope", 0.12), ("ICD shock", 0.10), ("SCD survivor", 0.05)],
        },
        "SCN5A": {
            "subtypes": [("Brugada Syndrome (LOF)", 0.45), ("LQT3 (GOF)", 0.30), ("PCCD Overlap", 0.15), ("Overlap: Brugada+PCCD", 0.10)],
            "treatments": [("Brugada: ICD", 0.30), ("Brugada: Quinidine", 0.18), ("LQT3: Mexiletine + BB", 0.28), ("PCCD: Pacemaker", 0.14), ("Asymptomatic surveillance", 0.10)],
            "outcomes": [("Event-free on therapy", 0.50), ("VF episode ICD-terminated", 0.18), ("Fever-induced Brugada event", 0.12), ("AV block paced", 0.12), ("SCD survivor", 0.08)],
        },
        "RYR2": {
            "subtypes": [("CPVT1 Symptomatic", 0.60), ("CPVT1 Asymptomatic Gene-Positive", 0.25), ("CPVT1 SCD Survivor", 0.15)],
            "treatments": [("Nadolol + Flecainide", 0.55), ("Nadolol alone", 0.18), ("Nadolol + Flecainide + ICD", 0.18), ("LCSD + meds", 0.09)],
            "outcomes": [("VT-free on combination therapy", 0.50), ("Breakthrough VT on ETT", 0.18), ("ICD storm", 0.08), ("LCSD performed", 0.12), ("SCD survivor", 0.12)],
        },
        "CASQ2": {
            "subtypes": [("CPVT2 Severe Biallelic", 0.70), ("CPVT2 Milder Biallelic", 0.20), ("Heterozygous Carrier (usually benign)", 0.10)],
            "treatments": [("Nadolol + Flecainide mandatory", 0.55), ("Nadolol + Flecainide + ICD", 0.28), ("LCSD + meds", 0.12), ("Supportive (asymptomatic carrier)", 0.05)],
            "outcomes": [("Partial VT reduction", 0.35), ("ICD storm", 0.15), ("LCSD benefit", 0.18), ("Stable on combination", 0.22), ("SCD/resuscitated", 0.10)],
        },
        "HCN4": {
            "subtypes": [("Sinus Bradycardia + Chronotropic Incompetence", 0.55), ("Bradycardia-Tachycardia (AF+SSS)", 0.30), ("Brugada-Overlap HCN4", 0.15)],
            "treatments": [("Pacemaker (DDD/DDDR)", 0.55), ("Pacemaker + rate-control AF", 0.25), ("Observation (asymptomatic)", 0.15), ("Antiarrhythmic + pacemaker", 0.05)],
            "outcomes": [("Symptom-free post-pacemaker", 0.58), ("AF persists requiring anticoagulation", 0.20), ("Chronotropic incompetence paced", 0.12), ("Brugada event (overlap)", 0.10)],
        },
        "ANK2": {
            "subtypes": [("Multimorphic: SND + AF + VF risk", 0.50), ("Isolated AF phenotype", 0.25), ("CPVT-like VT phenotype", 0.15), ("Sinus node only", 0.10)],
            "treatments": [("Beta-blocker + ICD", 0.45), ("Beta-blocker alone (AF, no VT)", 0.25), ("Pacemaker + BB", 0.18), ("Anticoagulation + rhythm control", 0.12)],
            "outcomes": [("ICD therapies delivered", 0.30), ("AF managed anticoagulated", 0.28), ("Event-free on BB", 0.25), ("Pacemaker implanted", 0.12), ("SCD survivor", 0.05)],
        },
        "KCNE1": {
            "subtypes": [("LQT5 Symptomatic Female", 0.45), ("LQT5 Asymptomatic Gene-Positive", 0.30), ("JLN2 Biallelic + Deaf", 0.15), ("LQT5 Symptomatic Male", 0.10)],
            "treatments": [("Nadolol", 0.55), ("Propranolol", 0.20), ("Nadolol + ICD", 0.15), ("JLN2: CI + Nadolol + ICD", 0.10)],
            "outcomes": [("Event-free on BB", 0.58), ("Breakthrough TdP", 0.15), ("ICD implanted", 0.14), ("JLN2 CI successful", 0.08), ("SCD survivor", 0.05)],
        },
    }

    props = GENE_PROPS.get(gene, {})
    subtypes = props.get("subtypes", [("Unknown", 1.0)])
    treatments = props.get("treatments", [("Supportive", 1.0)])
    outcomes = props.get("outcomes", [("Stable", 1.0)])

    def weighted_choice(choices):
        r = rng.random()
        cum = 0.0
        for name, prob in choices:
            cum += prob
            if r < cum:
                return name
        return choices[-1][0]

    patients = []
    for i in range(n):
        age_onset = round(rng.uniform(onset_lo, onset_hi), 1)
        age_current = round(age_onset + rng.uniform(1.0, 30.0), 1)
        age_current = min(age_current, 80.0)
        # For X-linked/sex-affected genes, weight sex appropriately
        if gene in ["KCNE1"]:
            sex = rng.choices(["F", "M"], weights=[0.70, 0.30])[0]
        elif gene in ["HCN4", "SCN5A"]:
            sex = rng.choices(["M", "F"], weights=[0.60, 0.40])[0]
        else:
            sex = rng.choice(["M", "F"])
        subtype = weighted_choice(subtypes)
        treatment = weighted_choice(treatments)
        outcome = weighted_choice(outcomes)
        qtc = round(rng.uniform(420, 590), 0)
        patients.append({
            "patient_id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "subtype": subtype,
            "age_onset_y": age_onset,
            "age_current_y": age_current,
            "sex": sex,
            "treatment": treatment,
            "outcome": outcome,
            "qtc_ms": int(qtc),
            "icd_implanted": "ICD" in treatment,
            "pacemaker_implanted": "Pacemaker" in treatment or "pacemaker" in treatment,
        })
    return patients


def get_overview() -> dict:
    all_patients = []
    gene_summaries = {}
    for g in ARRHYTHMIA_GENES:
        seed = SEED_BASE + g["seed_offset"]
        pts = _make_cohort(g, seed=seed, n=40)
        all_patients.extend(pts)
        n_icd = sum(1 for p in pts if p["icd_implanted"])
        n_pm = sum(1 for p in pts if p["pacemaker_implanted"])
        gene_summaries[g["gene"]] = {
            "n": len(pts),
            "icd_n": n_icd,
            "icd_pct": round(100 * n_icd / len(pts), 1),
            "pacemaker_n": n_pm,
            "pacemaker_pct": round(100 * n_pm / len(pts), 1),
            "locus": g["locus"],
            "inheritance": g["inheritance"].split(";")[0].strip(),
            "aa": g["aa"],
            "omim_disease": g["omim_disease"],
            "disease_short": {
                "KCNQ1": "LQT1 / JLN1 — IKs Alpha",
                "KCNH2": "LQT2 — IKr hERG",
                "SCN5A": "LQT3 + Brugada + PCCD — Nav1.5",
                "RYR2": "CPVT1 — SR Calcium Leak",
                "CASQ2": "CPVT2 — Calsequestrin AR",
                "HCN4": "Sick Sinus / Hereditary Bradycardia",
                "ANK2": "LQT4 / Ankyrin-B Syndrome",
                "KCNE1": "LQT5 / JLN2 — IKs Beta (MinK)",
            }.get(g["gene"], g["gene"]),
        }

    n_total = len(all_patients)
    n_icd = sum(1 for p in all_patients if p["icd_implanted"])
    n_pm = sum(1 for p in all_patients if p["pacemaker_implanted"])
    mean_qtc = round(sum(p["qtc_ms"] for p in all_patients) / n_total, 0)

    return {
        "atlas_name": "Arrhythmia-Atlas — Complete 8-Gene Inherited Cardiac Arrhythmia Atlas",
        "subtitle": "KCNQ1 · KCNH2 · SCN5A · RYR2 · CASQ2 · HCN4 · ANK2 · KCNE1",
        "n_patients": n_total,
        "n_genes": len(ARRHYTHMIA_GENES),
        "seeds": f"{SEED_BASE}-{SEED_BASE + len(ARRHYTHMIA_GENES) - 1}",
        "genes": [g["gene"] for g in ARRHYTHMIA_GENES],
        "gene_summaries": gene_summaries,
        "aggregate_clinical": {
            "icd_n": n_icd,
            "icd_pct": round(100 * n_icd / n_total, 1),
            "pacemaker_n": n_pm,
            "pacemaker_pct": round(100 * n_pm / n_total, 1),
            "mean_qtc_ms": int(mean_qtc),
        },
        "key_principles": [
            "TRIGGER SPECIFICITY: KCNQ1 swim, KCNH2 auditory, SCN5A nocturnal/fever, RYR2/CASQ2 exercise/emotion",
            "FEVER IN BRUGADA (SCN5A): immediately treat fever ≥38°C with paracetamol — written emergency plan every patient",
            "ICD PARADOX IN CPVT (RYR2/CASQ2): shock triggers catecholamine surge → VT storm; maximise beta-blocker first",
            "IVABRADINE CONTRAINDICATED in HCN4 LOF — directly blocks the deficient pacemaker channel",
            "DRUG-INDUCED QT: hERG (KCNH2/IKr) is the drug-sensitive channel; check CredibleMeds for every prescription",
            "FEMALE SEX RISK: KCNQ1 and KCNE1 (IKs) — females have less IKs reserve; higher QTc and event rates",
            "FLECAINIDE IN CPVT: blocks RYR2 open channel directly — add to nadolol for ~75% VT reduction",
            "ANK2 MULTIMORPHIC: same variant → SND in one family member, VF in another; include in comprehensive panels",
        ],
    }


def get_breakdown() -> dict:
    breakdown = {}
    for g in ARRHYTHMIA_GENES:
        seed = SEED_BASE + g["seed_offset"]
        pts = _make_cohort(g, seed=seed, n=40)
        from collections import Counter
        subtype_counts = dict(Counter(p["subtype"] for p in pts))
        treatment_counts = dict(Counter(p["treatment"] for p in pts))
        outcome_counts = dict(Counter(p["outcome"] for p in pts))
        ages_onset = [p["age_onset_y"] for p in pts]
        n_icd = sum(1 for p in pts if p["icd_implanted"])
        n_pm = sum(1 for p in pts if p["pacemaker_implanted"])
        breakdown[g["gene"]] = {
            "gene": g["gene"],
            "protein": g["protein"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "locus": g["locus"],
            "omim_gene": g["omim_gene"],
            "omim_disease": g["omim_disease"],
            "inheritance": g["inheritance"],
            "alias": g["alias"],
            "gene_class": g["gene_class"],
            "phenotype": g["phenotype"],
            "hallmark": g["hallmark"],
            "treatment_alerts": g["treatment_alerts"],
            "key_ddx": g["key_ddx"],
            "cohort_stats": {
                "n": len(pts),
                "seed": seed,
                "icd_n": n_icd,
                "icd_pct": round(100 * n_icd / len(pts), 1),
                "pacemaker_n": n_pm,
                "pacemaker_pct": round(100 * n_pm / len(pts), 1),
                "mean_age_onset_y": round(sum(ages_onset) / len(ages_onset), 1),
                "subtype_distribution": subtype_counts,
                "treatment_distribution": treatment_counts,
                "outcome_distribution": outcome_counts,
            },
        }
    return {"breakdown": breakdown}


def get_definitions() -> dict:
    return {
        "definitions": {
            "IKs_IKr_Distinction": (
                "IKs (slow delayed rectifier K+ current): KCNQ1 (alpha) + KCNE1 (MinK beta); "
                "activated slowly at positive potentials; critical for rate-adaptive QT shortening; "
                "upregulated by sympathetic (beta-adrenergic → cAMP → PKA phosphorylation of KCNQ1); "
                "deficiency = LQT1 (KCNQ1) or LQT5 (KCNE1); swim trigger; beta-blocker highly effective. "
                "IKr (rapid delayed rectifier K+ current): KCNH2/hERG (alpha) + KCNE2 (MiRP1 beta); "
                "activated faster than IKs; unique inward rectification from C-type inactivation; "
                "DRUG-SENSITIVE (hundreds of drugs block hERG via open-channel trapping); "
                "deficiency = LQT2 (KCNH2) or LQT6 (KCNE2); auditory trigger; potassium-sensitive."
            ),
            "Torsades_de_Pointes_TdP": (
                "Torsades de Pointes (TdP): a polymorphic ventricular tachycardia characterised by "
                "the QRS complex appearing to 'twist around' the isoelectric baseline — "
                "the amplitude and axis rotate in a sinusoidal fashion over 5-20 beats; "
                "French: 'twisting of the points (tips)'. "
                "INITIATING MECHANISM: early afterdepolarisation (EAD) during prolonged AP → "
                "triggered action potential → initiates TdP; "
                "maintaining mechanism: rotors or spiral waves in myocardium with heterogeneous repolarisation. "
                "PAUSE-DEPENDENT: TdP in LQTS is classically pause-dependent — "
                "a premature ventricular beat followed by a compensatory pause → "
                "long-short sequence ('long-short coupling') → EAD → TdP. "
                "MANAGEMENT: acute TdP in congenital LQTS: IV magnesium sulphate; "
                "temporary pacing if bradycardia-dependent; isoproterenol (accelerates HR, shortens QT); "
                "AVOID amiodarone (prolongs QT further); direct current cardioversion if persistent."
            ),
            "Bidirectional_VT_CPVT": (
                "Bidirectional VT: a ventricular tachycardia with alternating QRS axis "
                "(characteristically +60° and -120°, or alternating LBBB and RBBB morphology); "
                "beat-to-beat axis alternation visible in limb leads as alternating + and - QRS. "
                "MECHANISM: DAD-triggered alternating fascicular tachycardia — "
                "left posterior fascicle and left anterior fascicle alternately generate triggered beats "
                "when cytoplasmic [Ca2+] oscillates from SR calcium leak. "
                "PATHOGNOMONIC FOR CPVT: bidirectional VT induced by exercise or catecholamines "
                "in a patient with NORMAL resting ECG (no QT prolongation, no structural disease) "
                "= CPVT until proven otherwise. "
                "ALSO SEEN IN: Andersen-Tawil syndrome (KCNJ2), digitalis toxicity, "
                "rare cases of ischaemia — differentiate by context."
            ),
            "Brugada_Coved_Pattern": (
                "Brugada Pattern / Brugada Syndrome: "
                "ECG PATTERN (may be drug-provoked or spontaneous): "
                "Type 1 (Diagnostic): coved ST-segment elevation ≥2mm in ≥1 right precordial lead "
                "(V1 or V2 placed in standard or high intercostal position) with RBBB morphology; "
                "ST-segment descends with a convex upslope and negative T-wave. "
                "Type 2: saddle-back pattern (J-point ≥2mm, ST rises then falls, positive T) — "
                "requires Na+ channel blocker provocation or fever to unmask Type 1. "
                "BRUGADA SYNDROME DIAGNOSIS requires Type 1 pattern + clinical features "
                "(VF, family SCD history, syncope, inducibility at EP study). "
                "FEVER UNMASKING: fever shifts Nav1.5 inactivation gate → more LOF → "
                "unmasks Type 1 in previously concealed Brugada; applies to ~30-40% of Brugada patients."
            ),
            "Drug_Induced_QT_CredibleMeds": (
                "Drug-induced QT prolongation / TdP risk: "
                "CredibleMeds (AZCERT) database: crediblemeds.org — categorises drugs as: "
                "KNOWN RISK: associated with TdP when used as directed (azithromycin, haloperidol, "
                "sotalol, dofetilide, methadone, domperidone); "
                "CONDITIONAL RISK: TdP risk only under specific conditions (electrolyte abnormality, "
                "drug interactions, high dose) — amiodarone, fluconazole, metoclopramide; "
                "POSSIBLE RISK: limited evidence — check with specialist. "
                "MECHANISM: virtually all drugs that prolong QT do so by blocking hERG (IKr, KCNH2); "
                "KCNH2 carriers have reduced IKr reserve → standard drug doses cause excessive QT prolongation. "
                "CHECK before prescribing: antibiotics, antifungals, antipsychotics, "
                "antiemetics, antiarrhythmics, antihistamines, opioids (methadone)."
            ),
            "LCSD_Left_Cardiac_Sympathetic_Denervation": (
                "Left Cardiac Sympathetic Denervation (LCSD): "
                "Surgical technique: resect left stellate ganglion lower half + T2-T4 thoracic ganglia; "
                "performed thoracoscopically (minimally invasive) or via cervicothoracic incision. "
                "MECHANISM: removes left-sided cardiac noradrenaline release → reduces catecholamine-triggered "
                "arrhythmia → raises VF threshold; "
                "right sympathetic chain preserved → some adrenergic cardiac innervation maintained. "
                "INDICATIONS: CPVT (RYR2/CASQ2) with recurrent VT/VF despite maximal medication; "
                "LQTS with breakthrough events on beta-blocker + ICD; "
                "ICD storm in CPVT (preferred adjunct to reduce shock burden). "
                "EVIDENCE: LCSD reduces VT/VF burden by ~50-70% in CPVT; "
                "does not eliminate arrhythmia — combine with pharmacological therapy. "
                "Horner syndrome: mild permanent ptosis + anhidrosis on left face in ~20% — "
                "counsel pre-operatively."
            ),
            "JLN_Jervell_Lange_Nielsen": (
                "Jervell-Lange-Nielsen (JLN) Syndrome: "
                "Rare AR condition combining severe congenital sensorineural deafness + severe QT prolongation. "
                "GENETICS: "
                "JLN1: biallelic KCNQ1 pathogenic variants (both IKs alpha subunit copies lost); "
                "JLN2: biallelic KCNE1 pathogenic variants (both IKs MinK beta subunit copies lost). "
                "MOLECULAR MECHANISM: IKs expressed in stria vascularis of cochlea "
                "(KCNQ1 + KCNE1 maintain endocochlear potential); "
                "biallelic LOF → absent endocochlear potential → profound deafness. "
                "CLINICAL SEVERITY: QTc typically 550-600ms; arrhythmia events from infancy/early childhood; "
                "untreated mortality very high in first decade. "
                "MANAGEMENT: cochlear implant for deafness (highly effective); "
                "beta-blocker (nadolol) + ICD; genetic family cascade (heterozygous parents = LQT1 or LQT5). "
                "CONSANGUINITY: significantly increases JLN risk."
            ),
            "QTc_Correction_Fridericia": (
                "QT Correction Formulas: "
                "The QT interval varies with heart rate — must be corrected. "
                "BAZETT (QTc = QT / √RR): standard but overcorrects at high HR; still widely used. "
                "FRIDERICIA (QTcF = QT / RR^0.333): more accurate at higher HRs; preferred in drug trials. "
                "NORMAL THRESHOLDS: "
                "QTc ≤440ms in males (both Bazett and Fridericia); "
                "QTc ≤460ms in females (females naturally longer — progesterone/oestrogen effects on IKs/IKr). "
                "LQTS DIAGNOSTIC THRESHOLD: QTc ≥480ms in definite LQTS; "
                "QTc 460-479ms = 'borderline' — genetic testing warranted. "
                "MEASUREMENT: measure in leads II or V5 from QRS onset to end of T-wave; "
                "average 3-5 beats; exclude U-waves (common in hypokalaemia — NOT part of QT)."
            ),
        }
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = get_overview()
    print(f"Atlas: {ov['atlas_name']}")
    print(f"N patients: {ov['n_patients']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Aggregate: {json.dumps(ov['aggregate_clinical'], indent=2)}")
    print("\n=== BREAKDOWN (gene list) ===")
    bd = get_breakdown()
    for g, info in bd["breakdown"].items():
        print(f"  {g}: {info['cohort_stats']}")
    print("\n=== DEFINITIONS (keys) ===")
    df = get_definitions()
    for k in df["definitions"]:
        print(f"  - {k}")
