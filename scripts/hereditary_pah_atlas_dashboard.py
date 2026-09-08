#!/usr/bin/env python3
"""Hereditary-Pulmonary-Arterial-Hypertension-Atlas — Complete 8-Gene Hereditary PAH / PVOD Atlas
BMPR2   (Bone morphogenetic protein receptor type 2; 2037 aa; 2q33.1; AD;
         Most common hereditary PAH gene — 70-80% of HPAH;
         PENETRANCE only 20% (80% silent carriers) — genetic counselling essential;
         Female penetrance higher than male; ERA/PDE5i/prostacyclin;
         seed SEED_BASE+0) .
ACVRL1  (Activin receptor-like kinase 1 / ALK1; 503 aa; 12q13.13; AD;
         HHT2-associated PAH — hepatic AVM overlap complicates diagnosis;
         PAH in HHT2: distinguish from high-output failure from hepatic AVM;
         Avoid pulmonary vasodilators if hepatic AVM is the true cause;
         seed SEED_BASE+1) .
ENG     (Endoglin; 658 aa; 9q34.11; AD;
         HHT1-associated PAH — predominantly pulmonary AVM context;
         PAH rare in ENG-HHT1 vs ACVRL1-HHT2; right heart catheterisation mandatory;
         Bevacizumab (anti-VEGF) level B for HHT bleeding; not indicated for PAH;
         seed SEED_BASE+2) .
SMAD9   (SMAD family member 9 / MADH9; 530 aa; 13q13.3, AD;
         Rare HPAH — BMP-SMAD signalling downstream of BMPR2;
         Phenotype clinically similar to BMPR2-HPAH;
         Always test alongside BMPR2 on PAH gene panel;
         seed SEED_BASE+3) .
CAV1    (Caveolin-1; 178 aa; 7q31.2; AD;
         Rare HPAH; caveolae formation in pulmonary endothelium;
         Lipodystrophy association — congenital generalised lipodystrophy type 3;
         Low penetrance; female predominance;
         seed SEED_BASE+4) .
KCNK3   (K2P3.1 / TASK-1 potassium channel; 354 aa; 2p23.3; AD;
         GOF mechanism — UNUSUAL (most HPAH genes are LOF/haploinsufficiency);
         Reduced TASK-1 K+ current → sustained pulmonary vasoconstriction;
         Doxapram (TASK-1 activator) investigational;
         seed SEED_BASE+5) .
TBX4    (T-box transcription factor 4; 520 aa; 17q23.2; AD;
         ~20% of paediatric HPAH — DISTINCTIVE childhood-onset PAH;
         Small patella syndrome (musculoskeletal features absent in many);
         TBX4 most common gene in childhood HPAH after BMPR2;
         seed SEED_BASE+6) .
EIF2AK4 (Eukaryotic translation initiation factor 2-alpha kinase 4 / GCN2; 1649 aa; 15q15.1; AR;
         Pulmonary veno-occlusive disease / capillary haemangiomatosis (PVOD/PCH);
         VASODILATORS ABSOLUTELY CONTRAINDICATED — pulmonary oedema and death;
         Lung transplant is the ONLY curative treatment;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 1998-2005)
"""

import random

SEED_BASE = 1998

PAH_GENES = [
    # -- BMPR2 -- Most common hereditary PAH -------------------------------------------
    {
        "gene": "BMPR2",
        "alt_name": "BMPR2 (BMP Receptor type 2 / Hereditary PAH -- Most Common HPAH Gene 70-80%)",
        "protein": (
            "BMPR2 -- 2q33.1 AD -- BMPR2-2037aa -- "
            "Hereditary-PAH-Most-Common-70-80pct-HPAH -- "
            "PENETRANCE-20pct-Only-80pct-Silent-Carriers -- "
            "Female-Penetrance-Higher-Than-Male -- "
            "ERA-PDE5i-Prostacyclin-Triple-Therapy-Severe -- "
            "Right-Heart-Catheterisation-MANDATORY-Diagnosis"
        ),
        "locus": "2q33.1",
        "protein_size": "2037 aa",
        "inheritance": "AD (autosomal dominant) — penetrance only 20%",
        "age_of_onset": (
            "Young adult: most common presentation age 20-40yr; female predominance (F:M ~4:1 in HPAH); "
            "Penetrance: 80% of BMPR2 pathogenic variant carriers NEVER develop PAH; "
            "Symptoms: exertional dyspnoea → right heart failure → death if untreated; "
            "Median survival untreated: ~2.8yr from diagnosis; "
            "Paediatric BMPR2-PAH: less common; more aggressive; earlier transplant; "
            "Family history positive in ~50% HPAH — BMPR2 most common family cause; "
            "Triggers: OCP use, HIV, appetite suppressants — increase penetrance in carriers"
        ),
        "key_biomarker": (
            "Right heart catheterisation (RHC): mPAP ≥20 mmHg + PVR ≥3 WU + PAWP ≤15 mmHg — diagnostic PAH; "
            "NT-proBNP/BNP: severity + prognostication — target <300 pg/mL on therapy; "
            "6-minute walk distance (6MWD): functional assessment — target >440m low risk; "
            "Echo: estimated RVSP; RV dilation/dysfunction; TR velocity; pericardial effusion (poor prognostic sign); "
            "HRCT chest: excludes interstitial lung disease, PVOD pattern; "
            "Lung function: mild restriction common; DLCO reduced (severity marker); "
            "Molecular: BMPR2 panel sequencing + MLPA (large deletions 10%); "
            "Genetics: family cascade testing — 20% penetrance; inform all first-degree relatives"
        ),
        "pathognomonic": (
            "Young woman + exertional dyspnoea + elevated RVSP on echo + right heart failure = PAH workup; "
            "RHC: mPAP ≥20 mmHg + PVR ≥3 WU (WHO 2022 criteria) + PAWP ≤15 mmHg = PAH; "
            "BMPR2 pathogenic variant in HPAH family = presumptive HPAH unless other cause found; "
            "DISTINGUISH from PVOD (EIF2AK4): PVOD has HRCT pattern (centrilobular GGO + lymphadenopathy); "
            "DISTINGUISH from Group 2 (left heart): PAWP >15 mmHg; DISTINGUISH from Group 3 (lung): HRCT + spirometry; "
            "Vasodilator test (adenosine/inhaled NO) during RHC: positive if mPAP falls ≥10 mmHg to <40 mmHg — "
            "CCB responders (rare, ~10%) have dramatically better prognosis"
        ),
        "treatment": (
            "ERA: endothelin receptor antagonists (ambrisentan, macitentan, bosentan) — first-line; "
            "PDE5 inhibitor: sildenafil, tadalafil — first-line (often combined with ERA); "
            "Prostacyclin analogue: epoprostenol IV (most potent), inhaled iloprost, subcutaneous treprostinil; "
            "UPFRONT triple therapy: ERA + PDE5i + prostacyclin for high-risk PAH (RV failure, 6MWD <165m); "
            "Sotatercept: activin signalling ligand trap — FDA approved 2024 (STELLAR trial); "
            "Lung transplant: bilateral sequential — for refractory PAH; "
            "AVOID: pregnancy in BMPR2-PAH (maternal mortality 30-56%); high-altitude travel; heavy exercise; "
            "Genetics: cascade testing of all first-degree relatives (50% have variant; 20% penetrance)"
        ),
        "critical_flags": [
            "BMPR2-PENETRANCE-20pct-80pct-SILENT-CARRIERS",
            "BMPR2-RHC-MANDATORY-ECHO-NOT-SUFFICIENT-FOR-DIAGNOSIS",
            "BMPR2-PREGNANCY-MATERNAL-MORTALITY-30-56pct-AVOID",
            "BMPR2-SOTATERCEPT-FDA-2024-STELLAR-TRIAL",
            "BMPR2-MLPA-MANDATORY-10pct-LARGE-DELETIONS",
            "BMPR2-CASCADE-TESTING-FIRST-DEGREE-RELATIVES",
            "BMPR2-OCP-INCREASES-PENETRANCE-AVOID",
        ],
        "seed": SEED_BASE + 0,
    },
    # -- ACVRL1 -- HHT2-associated PAH -------------------------------------------------
    {
        "gene": "ACVRL1",
        "alt_name": "ACVRL1 (ALK1 / HHT2-Associated PAH -- Hepatic AVM Overlap Complicates Diagnosis)",
        "protein": (
            "ACVRL1 -- 12q13.13 AD -- ACVRL1-503aa -- "
            "HHT2-Associated-PAH-Hepatic-AVM-Overlap -- "
            "Distinguish-PAH-From-High-Output-Failure-Hepatic-AVM -- "
            "Avoid-Pulmonary-Vasodilators-If-Hepatic-AVM-True-Cause -- "
            "Epistaxis-Telangiectasia-HHT2-Plus-PAH"
        ),
        "locus": "12q13.13",
        "protein_size": "503 aa",
        "inheritance": "AD (autosomal dominant)",
        "age_of_onset": (
            "HHT2 features from childhood: recurrent epistaxis (nosebleeds) from telangiectasia; "
            "Hepatic AVM: most common AVM in ACVRL1-HHT2 (>80%) — may be asymptomatic initially; "
            "High-output cardiac failure: if hepatic AVM large — cardiomegaly, hyperdynamic circulation; "
            "PAH: occurs in ~2-5% of ACVRL1-HHT2 — investigate if unexplained dyspnoea; "
            "Adult: PAH diagnosis 20-60yr; hepatic disease may coexist; "
            "Screen all HHT2 patients for PAH with echocardiography — annual if symptomatic"
        ),
        "key_biomarker": (
            "Echocardiography: screen for elevated RVSP — distinguishes PAH from high-output failure; "
            "Hepatic Doppler USS: hepatic AVMs — hepatic arterioles to hepatic veins fistulae; "
            "Cardiac output (RHC): HIGH in hepatic AVM high-output failure; NORMAL in PAH; "
            "PAWP: elevated in high-output failure (ACVRL1 hepatic AVM); NORMAL in PAH; "
            "NT-proBNP: elevated in both — not discriminatory; "
            "RHC with cardiac output: mPAP + PVR + CO + PAWP — essential to differentiate; "
            "Molecular: ACVRL1 sequencing; MLPA for large deletions; ENG also test (10% conversion)"
        ),
        "pathognomonic": (
            "HHT2 + dyspnoea + elevated RVSP = MUST DISTINGUISH PAH vs hepatic AVM high-output failure; "
            "Key test: RHC — high CO + elevated PAWP = hepatic AVM (not PAH); normal CO + normal PAWP + high PVR = PAH; "
            "Hepatic AVM Doppler: hepatic arterialization of portal/hepatic veins — hyperdynamic flow; "
            "DISTINGUISH from BMPR2-PAH: ACVRL1-PAH has concurrent HHT features (epistaxis, telangiectasia, hepatic AVM); "
            "AVOID: pulmonary vasodilators (ERAs, PDE5i) if the true cause is hepatic AVM high-output failure — worsens it; "
            "Hepatic embolisation: ABSOLUTELY CONTRAINDICATED in ACVRL1 hepatic AVM — biliary ischaemia/necrosis risk"
        ),
        "treatment": (
            "If true PAH confirmed (RHC + normal CO + normal PAWP): treat as BMPR2-PAH with ERA/PDE5i/prostacyclin; "
            "If hepatic AVM high-output failure: manage heart failure + consider liver transplantation; "
            "Bevacizumab (anti-VEGF): level B evidence for HHT bleeding (epistaxis) — not for PAH; "
            "AVOID hepatic embolisation: biliary ischaemia risk — liver transplant if severe; "
            "HHT management: nasal packing + laser for epistaxis; iron supplementation; "
            "Multidisciplinary: pulmonology + hepatology + genetics + ENT; "
            "CASCADE testing: all first-degree relatives for HHT + PAH screening"
        ),
        "critical_flags": [
            "ACVRL1-DISTINGUISH-PAH-FROM-HIGH-OUTPUT-FAILURE-HEPATIC-AVM",
            "ACVRL1-AVOID-PULMONARY-VASODILATORS-IF-HEPATIC-AVM",
            "ACVRL1-HEPATIC-EMBOLISATION-ABSOLUTELY-CONTRAINDICATED",
            "ACVRL1-RHC-CO-PAWP-KEY-TO-DIAGNOSIS",
            "ACVRL1-ANNUAL-ECHO-SCREEN-ALL-HHT2-PATIENTS",
            "ACVRL1-BEVACIZUMAB-FOR-EPISTAXIS-NOT-PAH",
        ],
        "seed": SEED_BASE + 1,
    },
    # -- ENG -- HHT1-associated PAH ----------------------------------------------------
    {
        "gene": "ENG",
        "alt_name": "ENG (Endoglin / HHT1-Associated PAH -- Predominantly Pulmonary AVM Context)",
        "protein": (
            "ENG -- 9q34.11 AD -- ENG-658aa -- "
            "HHT1-Associated-PAH-Pulmonary-AVM-Predominant -- "
            "PAH-Rare-In-HHT1-vs-HHT2-ACVRL1 -- "
            "Right-Heart-Catheterisation-Mandatory-If-Dyspnoeic -- "
            "Pulmonary-AVM-Paradoxical-Embolism-Stroke-Abscess"
        ),
        "locus": "9q34.11",
        "protein_size": "658 aa",
        "inheritance": "AD (autosomal dominant)",
        "age_of_onset": (
            "HHT1 features: epistaxis from childhood; telangiectasia face/hands/mucosa by 2nd-3rd decade; "
            "Pulmonary AVM: 60-80% of ENG-HHT1 — MAIN pulmonary vascular complication; "
            "PAH: rare in HHT1 (<1%) — much less common than in ACVRL1-HHT2; "
            "Paradoxical embolism: pulmonary AVM allows paradoxical embolism → stroke + brain abscess; "
            "Screen: CT pulmonary angiography for PAVM at diagnosis + every 3-5 years; "
            "Cerebral AVM: 10-20% ENG — MRI brain at diagnosis mandatory"
        ),
        "key_biomarker": (
            "CT pulmonary angiography (CTPA): PAVM — feeding artery + nidal size (>3mm = treat); "
            "Echocardiography: bubble contrast study — intrapulmonary R-to-L shunt (agitated saline); "
            "Oxygen saturations: hypoxaemia from PAVM (R-to-L shunt) — SpO2 drop on exercise; "
            "Molecular: ENG sequencing + MLPA; "
            "If dyspnoeic: RHC to exclude PAH (rare but present in HHT1); "
            "Brain MRI: cerebral AVM screen — mandatory at diagnosis; "
            "Genetics: cascade testing of all first-degree relatives"
        ),
        "pathognomonic": (
            "Young adult + recurrent epistaxis + pulmonary AVM on CT + ENG variant = HHT1; "
            "Brain abscess in young adult without cardiac source = HHT1 pulmonary AVM until proven otherwise; "
            "Bubble echo: microbubbles appear LEFT heart chambers 3-8 beats after right = intrapulmonary shunt (PAVM); "
            "DISTINGUISH from ACVRL1-HHT2: ENG has MORE pulmonary AVM; ACVRL1 has MORE hepatic AVM + more PAH; "
            "DISTINGUISH PAH (rare) from hypoxaemia due to PAVM shunt: SpO2 low; A-a gradient; CTPA shows nidus; "
            "Antibiotic prophylaxis: MANDATORY before dental/invasive procedures (paradoxical embolism → brain abscess)"
        ),
        "treatment": (
            "PAVM embolisation: percutaneous transcatheter coil/vascular plug if feeding artery ≥3 mm; "
            "Antibiotic prophylaxis: MANDATORY pre-dental/procedural — bacterial paradoxical embolism risk; "
            "AVOID: air in IV lines (paradoxical air embolism through PAVM); "
            "AVOID: needle/Valsalva (increases R-to-L shunting); "
            "If PAH (rare): treat as BMPR2-PAH (ERA/PDE5i/prostacyclin); "
            "Epistaxis: nasal packing, laser, tranexamic acid; bevacizumab (anti-VEGF) level B; "
            "Cerebral AVM: neurosurgical/endovascular/radiosurgery assessment; "
            "HHT screening: PAVM CT every 3-5yr; cerebral MRI at diagnosis; liver USS"
        ),
        "critical_flags": [
            "ENG-PAVM-PARADOXICAL-EMBOLISM-BRAIN-ABSCESS",
            "ENG-ANTIBIOTIC-PROPHYLAXIS-MANDATORY-DENTAL-PROCEDURES",
            "ENG-PAH-RARE-IN-HHT1-VERSUS-ACVRL1-HHT2",
            "ENG-AVOID-AIR-IV-LINES-PARADOXICAL-AIR-EMBOLISM",
            "ENG-BUBBLE-ECHO-INTRAPULMONARY-SHUNT",
            "ENG-PAVM-EMBOLISE-IF-FEEDING-ARTERY-3mm",
            "ENG-CEREBRAL-AVM-MRI-MANDATORY-AT-DIAGNOSIS",
        ],
        "seed": SEED_BASE + 2,
    },
    # -- SMAD9 -- Rare HPAH -----------------------------------------------------------
    {
        "gene": "SMAD9",
        "alt_name": "SMAD9 (SMAD Family Member 9 / MADH9 -- Rare HPAH BMP-SMAD Downstream Signalling)",
        "protein": (
            "SMAD9 -- 13q13.3 AD -- SMAD9-530aa -- "
            "Rare-HPAH-BMP-SMAD-Pathway-Downstream-BMPR2 -- "
            "Phenotype-Clinically-Similar-To-BMPR2-HPAH -- "
            "Always-Test-On-PAH-Gene-Panel-With-BMPR2 -- "
            "SMAD1-SMAD5-SMAD9-BMP-R-SMADs"
        ),
        "locus": "13q13.3",
        "protein_size": "530 aa",
        "inheritance": "AD (autosomal dominant)",
        "age_of_onset": (
            "Young adult: phenotype clinically similar to BMPR2-PAH; "
            "SMAD9 (also called SMAD8 / MADH9) encodes a BMP-restricted SMAD (R-SMAD); "
            "Acts downstream of BMPR2 — haploinsufficiency → impaired BMP signalling → PAH; "
            "Female predominance similar to BMPR2; "
            "Lower penetrance than BMPR2 estimated; "
            "Clinically: exertional dyspnoea → right heart failure; same functional class trajectory"
        ),
        "key_biomarker": (
            "RHC: same criteria as BMPR2 — mPAP ≥20 + PVR ≥3 WU + PAWP ≤15 mmHg; "
            "NT-proBNP: prognostication; "
            "6MWD: functional status; "
            "Echo: RV dilation, TR velocity; "
            "Molecular: SMAD9 sequencing — included in all PAH gene panels; "
            "BMP signalling biomarkers: investigational (BMP9, BMP10 serum levels) — research tools"
        ),
        "pathognomonic": (
            "SMAD9-PAH: clinically indistinguishable from BMPR2-PAH; molecular diagnosis required; "
            "BMP pathway hierarchy: BMPR2 → SMAD1/5/9 → transcription (anti-proliferative in PAECs); "
            "LOF of any component = impaired BMP signalling = PAEC proliferation/apoptosis resistance; "
            "DISTINGUISH from KCNK3 (GOF mechanism): SMAD9 is LOF/haploinsufficiency; "
            "DISTINGUISH from EIF2AK4/PVOD: no PVOD features (GGO on HRCT, lymphadenopathy); "
            "Always include on PAH panel: BMPR2 + BMPR1B + ACVRL1 + ENG + SMAD9 + CAV1 + KCNK3 + TBX4 + EIF2AK4"
        ),
        "treatment": (
            "Same PAH treatment ladder as BMPR2-PAH: ERA + PDE5i ± prostacyclin; "
            "Sotatercept: activin pathway modulator — rational mechanism (rebalances BMP/activin); "
            "Lung transplant: refractory PAH; "
            "AVOID: pregnancy (same risk as BMPR2); high-altitude; heavy exercise; "
            "Genetic counselling: 50% inheritance risk; penetrance lower than BMPR2"
        ),
        "critical_flags": [
            "SMAD9-CLINICALLY-IDENTICAL-TO-BMPR2-PAH",
            "SMAD9-INCLUDE-ON-ALL-PAH-GENE-PANELS",
            "SMAD9-BMP-RSMAD-DOWNSTREAM-BMPR2",
            "SMAD9-SOTATERCEPT-RATIONAL-MECHANISM",
            "SMAD9-PENETRANCE-LOWER-THAN-BMPR2-ESTIMATE",
        ],
        "seed": SEED_BASE + 3,
    },
    # -- CAV1 -- Caveolin-1 rare PAH ---------------------------------------------------
    {
        "gene": "CAV1",
        "alt_name": "CAV1 (Caveolin-1 / Rare HPAH -- Lipodystrophy Association + Low Penetrance)",
        "protein": (
            "CAV1 -- 7q31.2 AD -- CAV1-178aa -- "
            "Rare-HPAH-Caveolae-Formation-Pulmonary-Endothelium -- "
            "Congenital-Generalised-Lipodystrophy-Type3-Association -- "
            "Female-Predominance-Low-Penetrance -- "
            "BMPR2-Interacts-With-Caveolae-Mechanistic-Link"
        ),
        "locus": "7q31.2",
        "protein_size": "178 aa",
        "inheritance": "AD (autosomal dominant); also AR for lipodystrophy",
        "age_of_onset": (
            "Rare: <2% of HPAH families; "
            "AD LOF: PAH + possible lipodystrophy phenotype; "
            "AR biallelic LOF: congenital generalised lipodystrophy type 3 (CGL3) — absent adipose tissue; "
            "Female predominance similar to BMPR2; young adult onset; "
            "Low penetrance: many CAV1 variant carriers are unaffected; "
            "Caveolae are flask-shaped plasma membrane invaginations — endocytosis + signalling hubs; "
            "Mechanistic: caveolin-1 scaffolds BMPR2 — loss disrupts BMP signalling in PAECs"
        ),
        "key_biomarker": (
            "RHC: mPAP ≥20 + PVR ≥3 WU + PAWP ≤15 mmHg — same PAH diagnosis; "
            "If lipodystrophy: metabolic evaluation (triglycerides, HbA1c, hepatic steatosis); "
            "Molecular: CAV1 sequencing; "
            "Echo: RV assessment; "
            "Clinical: look for lipodystrophy features (absent subcutaneous fat limbs/trunk) if biallelic suspected"
        ),
        "pathognomonic": (
            "PAH + generalised lipodystrophy = CAV1 (biallelic) or AGPAT2/BSCL2 (other lipodystrophy genes); "
            "Isolated PAH + CAV1 variant: rare — penetrance low; "
            "DISTINGUISH from BMPR2-PAH: CAV1 phenotype identical without lipodystrophy; molecular differentiates; "
            "DISTINGUISH from AR lipodystrophy (CGL): biallelic CAV1 → lipodystrophy; AD → PAH risk"
        ),
        "treatment": (
            "PAH: ERA + PDE5i + prostacyclin — same algorithm as BMPR2; "
            "If CGL3 (AR): metformin + leptin replacement (metreleptin) for insulin resistance; "
            "Lung transplant: refractory PAH; "
            "AVOID: pregnancy; "
            "Cascade testing: first-degree relatives"
        ),
        "critical_flags": [
            "CAV1-RARE-HPAH-LESS-THAN-2pct",
            "CAV1-AR-BIALLELIC-CONGENITAL-LIPODYSTROPHY-CGL3",
            "CAV1-LOW-PENETRANCE-FEMALE-PREDOMINANCE",
            "CAV1-CAVEOLAE-BMPR2-INTERACTION",
            "CAV1-METABOLIC-EVALUATION-IF-LIPODYSTROPHY",
        ],
        "seed": SEED_BASE + 4,
    },
    # -- KCNK3 -- TASK-1 GOF PAH -------------------------------------------------------
    {
        "gene": "KCNK3",
        "alt_name": "KCNK3 (TASK-1 K2P Channel / HPAH -- GOF Mechanism Unusual -- Doxapram Investigational)",
        "protein": (
            "KCNK3 -- 2p23.3 AD -- KCNK3-354aa -- "
            "K2P3.1-TASK1-Potassium-Channel-PAH -- "
            "GOF-Mechanism-Unusual-Most-HPAH-Are-LOF -- "
            "Reduced-TASK1-Current-Sustained-Pulmonary-Vasoconstriction -- "
            "Doxapram-TASK1-Activator-Investigational"
        ),
        "locus": "2p23.3",
        "protein_size": "354 aa",
        "inheritance": "AD (autosomal dominant) — GOF mechanism",
        "age_of_onset": (
            "Young to middle adult: similar age distribution to BMPR2-PAH; "
            "GOF mechanism: variants reduce TASK-1 channel outward K+ current (not LOF/haploinsufficiency); "
            "Consequence: sustained pulmonary arterial smooth muscle depolarisation → vasoconstriction → PAH; "
            "Lower penetrance than BMPR2; family clustering may be less obvious; "
            "Phenotype: clinically similar to BMPR2-PAH — exertional dyspnoea, RV failure"
        ),
        "key_biomarker": (
            "RHC: diagnostic — mPAP ≥20 + PVR ≥3 WU + PAWP ≤15 mmHg; "
            "Patch clamp electrophysiology: reduced TASK-1 current in patient-derived cells — research; "
            "Molecular: KCNK3 sequencing — included on PAH gene panels; "
            "Response to doxapram: TASK-1 activator (apnoea treatment drug) — vasodilatory in KCNK3-PAH (investigational)"
        ),
        "pathognomonic": (
            "KCNK3-PAH: clinically indistinguishable from BMPR2 without molecular diagnosis; "
            "GOF mechanism: distinguishes KCNK3 from all other HPAH genes (all LOF/haploinsufficiency); "
            "DISTINGUISH from SMAD9/CAV1: same PAH phenotype; different molecular mechanism; "
            "Doxapram rationale: TASK-1 activators theoretically reverse vasoconstriction — clinical trials ongoing; "
            "AVOID: standard TASK-1 blockers (e.g. bupivacaine — local anaesthetic) may worsen KCNK3-PAH"
        ),
        "treatment": (
            "ERA + PDE5i + prostacyclin: standard PAH algorithm applies; "
            "Doxapram: TASK-1 activator — investigational for KCNK3-PAH specifically; "
            "Sotatercept: PAH-approved; mechanism-agnostic; "
            "Lung transplant: refractory PAH; "
            "AVOID: bupivacaine (TASK-1 blocker) and other local anaesthetics that block TASK-1 channels; "
            "Genetic counselling: 50% inheritance risk"
        ),
        "critical_flags": [
            "KCNK3-GOF-UNUSUAL-MOST-HPAH-GENES-ARE-LOF",
            "KCNK3-DOXAPRAM-TASK1-ACTIVATOR-INVESTIGATIONAL",
            "KCNK3-AVOID-BUPIVACAINE-TASK1-BLOCKER",
            "KCNK3-REDUCED-K-CURRENT-VASOCONSTRICTION",
            "KCNK3-INCLUDE-ON-PAH-GENE-PANEL",
        ],
        "seed": SEED_BASE + 5,
    },
    # -- TBX4 -- Childhood HPAH --------------------------------------------------------
    {
        "gene": "TBX4",
        "alt_name": "TBX4 (T-Box Transcription Factor 4 / Paediatric PAH -- ~20% of Childhood HPAH)",
        "protein": (
            "TBX4 -- 17q23.2 AD -- TBX4-520aa -- "
            "Paediatric-PAH-DISTINCTIVE-Childhood-Onset -- "
            "20pct-Of-Childhood-HPAH-After-BMPR2 -- "
            "Small-Patella-Syndrome-Musculoskeletal-Features-Absent-Many -- "
            "Most-Common-Gene-Childhood-HPAH-After-BMPR2"
        ),
        "locus": "17q23.2",
        "protein_size": "520 aa",
        "inheritance": "AD (autosomal dominant)",
        "age_of_onset": (
            "Childhood/adolescence: DISTINCTIVE paediatric PAH — most common HPAH gene in children after BMPR2; "
            "~20% of childhood HPAH carries TBX4 variant; "
            "Small patella syndrome: haploinsufficiency of TBX4 → absent/small patellae + acinar dysplasia; "
            "Many TBX4-PAH patients: NO musculoskeletal features (variable penetrance of skeletal vs PAH); "
            "Acinar lung dysplasia: some TBX4 patients have parenchymal lung disease (acinar hypoplasia); "
            "PAH onset: infancy to young adulthood; variable severity"
        ),
        "key_biomarker": (
            "RHC: paediatric diagnostic criteria — mPAP >25 mmHg + PVR/PBF ratio for children; "
            "X-ray knee: absent/hypoplastic patella — small patella syndrome (supportive); "
            "Echo: RV pressure estimation; patent ductus arteriosus / ASD coexisting; "
            "HRCT chest: acinar dysplasia (ground glass) if lung parenchyma affected; "
            "Molecular: TBX4 sequencing — include on paediatric PAH panel; "
            "Lung function (if older child): restriction if acinar dysplasia"
        ),
        "pathognomonic": (
            "Childhood PAH + absent patella on X-ray = TBX4 until proven otherwise; "
            "Childhood PAH without skeletal features: TBX4 still most common after BMPR2 — MUST screen; "
            "DISTINGUISH from BMPR2-childhood-PAH: TBX4 more common in paediatric; "
            "DISTINGUISH from pulmonary hypertension from cardiac disease: PAWP normal; PVR elevated; "
            "Acinar dysplasia: TBX4 → hypoplastic peripheral airways → ventilation-perfusion mismatch; "
            "Family history: look for small patella syndrome in parents/relatives of TBX4-PAH child"
        ),
        "treatment": (
            "Paediatric PAH: ERA (bosentan/macitentan), PDE5i (sildenafil), prostacyclin (treprostinil); "
            "Dosing: paediatric weight-based; sildenafil has FDA/EMA approval for paediatric PAH; "
            "Bosentan: FDA approved for PAH age ≥1yr; LFT monitoring mandatory; "
            "Lung transplant: BILATERAL sequential; bridge with intravenous prostacyclin; "
            "Small patella: physiotherapy; patellar stabilisation; avoid high-impact sports; "
            "Genetic counselling: 50% inheritance; screen siblings with knee X-ray + echo"
        ),
        "critical_flags": [
            "TBX4-MOST-COMMON-CHILDHOOD-HPAH-GENE-AFTER-BMPR2",
            "TBX4-20pct-CHILDHOOD-HPAH",
            "TBX4-SMALL-PATELLA-ABSENT-PATELLA-X-RAY",
            "TBX4-MANY-NO-MUSCULOSKELETAL-FEATURES-VARIABLE",
            "TBX4-ACINAR-DYSPLASIA-LUNG-PARENCHYMA",
            "TBX4-SCREEN-PAEDIATRIC-PAH-PANEL-MANDATORY",
        ],
        "seed": SEED_BASE + 6,
    },
    # -- EIF2AK4 -- PVOD/PCH (AR) ------------------------------------------------------
    {
        "gene": "EIF2AK4",
        "alt_name": "EIF2AK4 (GCN2 Kinase / Pulmonary Veno-Occlusive Disease PVOD -- VASODILATORS CONTRAINDICATED)",
        "protein": (
            "EIF2AK4 -- 15q15.1 AR -- EIF2AK4-1649aa -- "
            "Pulmonary-Veno-Occlusive-Disease-PVOD-PCH -- "
            "VASODILATORS-ABSOLUTELY-CONTRAINDICATED-Pulmonary-Oedema-Death -- "
            "Lung-Transplant-ONLY-Curative-Treatment -- "
            "Biallelic-AR-LOF-EIF2AK4-Defines-Hereditary-PVOD"
        ),
        "locus": "15q15.1",
        "protein_size": "1649 aa",
        "inheritance": "AR (autosomal recessive) — biallelic LOF",
        "age_of_onset": (
            "Young to middle adult: biallelic EIF2AK4 LOF → PVOD; "
            "PVOD: pulmonary venous occlusion → post-capillary block → pulmonary oedema rather than arterial PAH; "
            "Presentation: dyspnoea + exertional desaturation + haemoptysis; "
            "PVOD/PCH is clinically misdiagnosed as Group 1 PAH — CRITICAL DISTINCTION; "
            "Occupational risk: organic solvent exposure (especially trichloroethylene) increases PVOD risk; "
            "Prognosis: very poor without transplant — rapid deterioration if vasodilators given"
        ),
        "key_biomarker": (
            "HRCT chest: PATHOGNOMONIC pattern — centrilobular ground-glass opacities + interlobular septal thickening + mediastinal lymphadenopathy; "
            "DLCO: markedly reduced (lower than arterial PAH); "
            "Bronchoalveolar lavage: haemosiderin-laden macrophages (occult alveolar haemorrhage); "
            "Echocardiography: elevated RVSP; pericardial effusion common; "
            "RHC: mPAP elevated + PVR elevated + PAWP NORMAL (confusing — venous block is post-capillary); "
            "Vasodilator test: provokes pulmonary oedema — ABSOLUTELY CONTRAINDICATED in PVOD; "
            "Molecular: EIF2AK4 biallelic sequencing — confirms hereditary PVOD/PCH; "
            "Lung biopsy: avoided if molecular confirmed — high surgical risk in severe PAH"
        ),
        "pathognomonic": (
            "HRCT: centrilobular GGO + interlobular septal thickening + mediastinal LN = PVOD PATHOGNOMONIC; "
            "Biallelic EIF2AK4 LOF = HEREDITARY PVOD/PCH (does NOT need biopsy for diagnosis); "
            "DISTINGUISH from arterial PAH (BMPR2 etc.): HRCT pattern; DLCO much lower; haemoptysis; BAL haemosiderin; "
            "ACUTE PULMONARY OEDEMA after vasodilator = PVOD (not arterial PAH) — triggered by any pulmonary vasodilator; "
            "DISTINGUISH from left heart Group 2: PAWP normal in PVOD (confusing — post-capillary venous, not cardiac); "
            "Organic solvent history: trichloroethylene/perchloroethylene exposure increases PVOD risk"
        ),
        "treatment": (
            "LUNG TRANSPLANT: ONLY curative treatment — bilateral sequential; list EARLY (rapid deterioration); "
            "AVOID ALL pulmonary vasodilators: ERA, PDE5i, prostacyclin — risk of fatal flash pulmonary oedema; "
            "Bridge to transplant: inhaled nitric oxide may be used cautiously in ICU only with haemodynamic monitoring; "
            "Diuretics: cautious use for fluid balance; "
            "Oxygen: supplemental O2 for desaturation; "
            "NEVER give: sildenafil / ambrisentan / iloprost / epoprostenol as outpatients — fatal oedema risk; "
            "Genetic counselling: AR — 25% recurrence risk; heterozygous carriers (parents): no increased PVOD risk; "
            "Organic solvent avoidance: workplace exposure cessation"
        ),
        "critical_flags": [
            "EIF2AK4-VASODILATORS-ABSOLUTELY-CONTRAINDICATED-PULMONARY-OEDEMA-DEATH",
            "EIF2AK4-LUNG-TRANSPLANT-ONLY-CURATIVE",
            "EIF2AK4-HRCT-CENTRILOBULAR-GGO-PATHOGNOMONIC",
            "EIF2AK4-BIALLELIC-LOF-CONFIRMS-HEREDITARY-PVOD-NO-BIOPSY",
            "EIF2AK4-ORGANIC-SOLVENT-TRICHLOROETHYLENE-RISK",
            "EIF2AK4-BAL-HAEMOSIDERIN-LADEN-MACROPHAGES",
            "EIF2AK4-LIST-TRANSPLANT-EARLY-RAPID-DETERIORATION",
        ],
        "seed": SEED_BASE + 7,
    },
]

# ── COHORT GENERATION ─────────────────────────────────────────────────────────

def _generate_cohort(gene_entry: dict, n: int = 40) -> list:
    rng = random.Random(gene_entry["seed"])
    gene = gene_entry["gene"]
    cohort = []
    for i in range(n):
        age = rng.randint(5, 75)
        sex = rng.choice(["M", "F"])

        # Gene-specific clinical probabilities (clinically grounded)
        if gene == "BMPR2":
            mPAP_elevated    = rng.random() < 0.95
            pvr_elevated     = rng.random() < 0.95
            right_hf         = rng.random() < 0.55
            six_mwd_low      = rng.random() < 0.60
            ntprobnp_high    = rng.random() < 0.70
            echo_rv_dilated  = rng.random() < 0.70
            vasodilator_test = rng.random() < 0.10  # positive responder ~10%
            triple_therapy   = rng.random() < 0.40
            transplant       = rng.random() < 0.20
            haemoptysis      = rng.random() < 0.10
            pvod_pattern_hrct= False
            hht_features     = False
            pavm             = False
            lipodystrophy    = False
            childhood_onset  = rng.random() < 0.10
            small_patella    = False
            pericardial_eff  = rng.random() < 0.15

        elif gene == "ACVRL1":
            mPAP_elevated    = rng.random() < 0.70
            pvr_elevated     = rng.random() < 0.70
            right_hf         = rng.random() < 0.35
            six_mwd_low      = rng.random() < 0.45
            ntprobnp_high    = rng.random() < 0.50
            echo_rv_dilated  = rng.random() < 0.50
            vasodilator_test = rng.random() < 0.05
            triple_therapy   = rng.random() < 0.20
            transplant       = rng.random() < 0.10
            haemoptysis      = rng.random() < 0.15
            pvod_pattern_hrct= False
            hht_features     = rng.random() < 0.95
            pavm             = rng.random() < 0.30
            lipodystrophy    = False
            childhood_onset  = rng.random() < 0.08
            small_patella    = False
            pericardial_eff  = rng.random() < 0.10

        elif gene == "ENG":
            mPAP_elevated    = rng.random() < 0.40
            pvr_elevated     = rng.random() < 0.40
            right_hf         = rng.random() < 0.20
            six_mwd_low      = rng.random() < 0.30
            ntprobnp_high    = rng.random() < 0.30
            echo_rv_dilated  = rng.random() < 0.30
            vasodilator_test = rng.random() < 0.05
            triple_therapy   = rng.random() < 0.10
            transplant       = rng.random() < 0.08
            haemoptysis      = rng.random() < 0.20
            pvod_pattern_hrct= False
            hht_features     = rng.random() < 0.98
            pavm             = rng.random() < 0.70
            lipodystrophy    = False
            childhood_onset  = rng.random() < 0.05
            small_patella    = False
            pericardial_eff  = rng.random() < 0.08

        elif gene == "SMAD9":
            mPAP_elevated    = rng.random() < 0.92
            pvr_elevated     = rng.random() < 0.92
            right_hf         = rng.random() < 0.50
            six_mwd_low      = rng.random() < 0.55
            ntprobnp_high    = rng.random() < 0.65
            echo_rv_dilated  = rng.random() < 0.65
            vasodilator_test = rng.random() < 0.08
            triple_therapy   = rng.random() < 0.35
            transplant       = rng.random() < 0.15
            haemoptysis      = rng.random() < 0.08
            pvod_pattern_hrct= False
            hht_features     = False
            pavm             = False
            lipodystrophy    = False
            childhood_onset  = rng.random() < 0.10
            small_patella    = False
            pericardial_eff  = rng.random() < 0.12

        elif gene == "CAV1":
            mPAP_elevated    = rng.random() < 0.88
            pvr_elevated     = rng.random() < 0.88
            right_hf         = rng.random() < 0.45
            six_mwd_low      = rng.random() < 0.50
            ntprobnp_high    = rng.random() < 0.60
            echo_rv_dilated  = rng.random() < 0.60
            vasodilator_test = rng.random() < 0.08
            triple_therapy   = rng.random() < 0.30
            transplant       = rng.random() < 0.12
            haemoptysis      = rng.random() < 0.08
            pvod_pattern_hrct= False
            hht_features     = False
            pavm             = False
            lipodystrophy    = rng.random() < 0.20
            childhood_onset  = rng.random() < 0.08
            small_patella    = False
            pericardial_eff  = rng.random() < 0.10

        elif gene == "KCNK3":
            mPAP_elevated    = rng.random() < 0.90
            pvr_elevated     = rng.random() < 0.90
            right_hf         = rng.random() < 0.50
            six_mwd_low      = rng.random() < 0.55
            ntprobnp_high    = rng.random() < 0.65
            echo_rv_dilated  = rng.random() < 0.65
            vasodilator_test = rng.random() < 0.08
            triple_therapy   = rng.random() < 0.35
            transplant       = rng.random() < 0.15
            haemoptysis      = rng.random() < 0.08
            pvod_pattern_hrct= False
            hht_features     = False
            pavm             = False
            lipodystrophy    = False
            childhood_onset  = rng.random() < 0.10
            small_patella    = False
            pericardial_eff  = rng.random() < 0.12

        elif gene == "TBX4":
            mPAP_elevated    = rng.random() < 0.92
            pvr_elevated     = rng.random() < 0.92
            right_hf         = rng.random() < 0.45
            six_mwd_low      = rng.random() < 0.50
            ntprobnp_high    = rng.random() < 0.60
            echo_rv_dilated  = rng.random() < 0.60
            vasodilator_test = rng.random() < 0.10
            triple_therapy   = rng.random() < 0.30
            transplant       = rng.random() < 0.20
            haemoptysis      = rng.random() < 0.10
            pvod_pattern_hrct= False
            hht_features     = False
            pavm             = False
            lipodystrophy    = False
            childhood_onset  = rng.random() < 0.75
            small_patella    = rng.random() < 0.55
            pericardial_eff  = rng.random() < 0.12

        else:  # EIF2AK4
            mPAP_elevated    = rng.random() < 0.95
            pvr_elevated     = rng.random() < 0.90
            right_hf         = rng.random() < 0.70
            six_mwd_low      = rng.random() < 0.80
            ntprobnp_high    = rng.random() < 0.85
            echo_rv_dilated  = rng.random() < 0.80
            vasodilator_test = False  # CONTRAINDICATED
            triple_therapy   = False  # CONTRAINDICATED
            transplant       = rng.random() < 0.55
            haemoptysis      = rng.random() < 0.55
            pvod_pattern_hrct= rng.random() < 0.90
            hht_features     = False
            pavm             = False
            lipodystrophy    = False
            childhood_onset  = rng.random() < 0.15
            small_patella    = False
            pericardial_eff  = rng.random() < 0.40

        cohort.append({
            "id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "age": age,
            "sex": sex,
            "mPAP_elevated": mPAP_elevated,
            "pvr_elevated": pvr_elevated,
            "right_heart_failure": right_hf,
            "six_mwd_low": six_mwd_low,
            "ntprobnp_high": ntprobnp_high,
            "echo_rv_dilated": echo_rv_dilated,
            "vasodilator_test_positive": vasodilator_test,
            "triple_therapy": triple_therapy,
            "transplant_listed_or_done": transplant,
            "haemoptysis": haemoptysis,
            "pvod_pattern_hrct": pvod_pattern_hrct,
            "hht_features": hht_features,
            "pavm": pavm,
            "lipodystrophy": lipodystrophy,
            "childhood_onset": childhood_onset,
            "small_patella": small_patella,
            "pericardial_effusion": pericardial_eff,
        })
    return cohort


def overview() -> dict:
    all_cohorts = [_generate_cohort(g) for g in PAH_GENES]
    total = sum(len(c) for c in all_cohorts)
    all_pts = [p for c in all_cohorts for p in c]

    def N(key): return sum(1 for p in all_pts if p[key])

    return {
        "atlas": "Hereditary-Pulmonary-Arterial-Hypertension-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary PAH / PVOD Atlas: "
            "BMPR2, ACVRL1, ENG, SMAD9, CAV1, KCNK3 (Group 1 PAH) + "
            "TBX4 (paediatric PAH) + EIF2AK4 (PVOD/PCH — vasodilators CONTRAINDICATED)"
        ),
        "genes": [g["gene"] for g in PAH_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE+7}",
        "mPAP_elevated_patients":       N("mPAP_elevated"),
        "pvr_elevated_patients":        N("pvr_elevated"),
        "right_heart_failure_patients": N("right_heart_failure"),
        "transplant_patients":          N("transplant_listed_or_done"),
        "haemoptysis_patients":         N("haemoptysis"),
        "pvod_pattern_patients":        N("pvod_pattern_hrct"),
        "hht_features_patients":        N("hht_features"),
        "pavm_patients":                N("pavm"),
        "childhood_onset_patients":     N("childhood_onset"),
        "small_patella_patients":       N("small_patella"),
        "lipodystrophy_patients":       N("lipodystrophy"),
        "triple_therapy_patients":      N("triple_therapy"),
        "vasodilator_test_positive_patients": N("vasodilator_test_positive"),
        "gene_patient_counts": {g["gene"]: len(_generate_cohort(g)) for g in PAH_GENES},
        "pathway": (
            "BMP-SMAD signalling maintains pulmonary arterial homeostasis: "
            "BMP9/10 → BMPR2 (+ ACVRL1 co-receptor) → SMAD1/5/9 → anti-proliferative transcription; "
            "CAV1 scaffolds BMPR2 in caveolae — loss disrupts signalling; "
            "KCNK3 (TASK-1) regulates pulmonary vascular tone via K+ conductance; "
            "TBX4 regulates lung development — haploinsufficiency → alveolar + vascular abnormalities; "
            "EIF2AK4 (GCN2) regulates integrated stress response — "
            "biallelic LOF → pulmonary venous endothelial stress → PVOD."
        ),
        "key_clinical_insight": (
            "BMPR2: most common HPAH (70-80%); penetrance ONLY 20% — cascade testing critical; pregnancy mortality 30-56% AVOID. "
            "ACVRL1: DISTINGUISH PAH from hepatic AVM high-output failure (RHC with CO mandatory); hepatic embolisation ABSOLUTELY CI. "
            "ENG: PAH RARE in HHT1 — mainly pulmonary AVM; antibiotic prophylaxis MANDATORY dental procedures. "
            "SMAD9: phenotypically identical to BMPR2-PAH; always include on PAH panel; sotatercept rational mechanism. "
            "CAV1: rare HPAH; AR biallelic = generalised lipodystrophy CGL3; low penetrance. "
            "KCNK3: GOF mechanism UNUSUAL among HPAH genes; doxapram (TASK-1 activator) investigational. "
            "TBX4: ~20% childhood HPAH; small patella diagnostic; absent musculoskeletal features does NOT exclude TBX4. "
            "EIF2AK4: PVOD/PCH — VASODILATORS ABSOLUTELY CONTRAINDICATED (flash pulmonary oedema); LUNG TRANSPLANT ONLY curative."
        ),
    }


def breakdown() -> dict:
    result = {}
    for gene_entry in PAH_GENES:
        cohort = _generate_cohort(gene_entry)
        gene = gene_entry["gene"]

        def pct(key):
            return round(100 * sum(1 for p in cohort if p[key]) / len(cohort))

        result[gene] = {
            "gene": gene,
            "alt_name": gene_entry["alt_name"],
            "locus": gene_entry["locus"],
            "protein_size": gene_entry["protein_size"],
            "inheritance": gene_entry["inheritance"],
            "n_patients": len(cohort),
            "mPAP_elevated_pct":            pct("mPAP_elevated"),
            "pvr_elevated_pct":             pct("pvr_elevated"),
            "right_heart_failure_pct":      pct("right_heart_failure"),
            "six_mwd_low_pct":              pct("six_mwd_low"),
            "ntprobnp_high_pct":            pct("ntprobnp_high"),
            "echo_rv_dilated_pct":          pct("echo_rv_dilated"),
            "vasodilator_test_positive_pct": pct("vasodilator_test_positive"),
            "triple_therapy_pct":           pct("triple_therapy"),
            "transplant_pct":               pct("transplant_listed_or_done"),
            "haemoptysis_pct":              pct("haemoptysis"),
            "pvod_pattern_hrct_pct":        pct("pvod_pattern_hrct"),
            "hht_features_pct":             pct("hht_features"),
            "pavm_pct":                     pct("pavm"),
            "lipodystrophy_pct":            pct("lipodystrophy"),
            "childhood_onset_pct":          pct("childhood_onset"),
            "small_patella_pct":            pct("small_patella"),
            "pericardial_effusion_pct":     pct("pericardial_effusion"),
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
        "atlas": "Hereditary-Pulmonary-Arterial-Hypertension-Atlas",
        "pathway": "BMP-SMAD Signalling / Pulmonary Vascular Homeostasis",
        "shared_mechanism": (
            "Group 1 Hereditary PAH (BMPR2, ACVRL1, ENG, SMAD9, CAV1, KCNK3, TBX4): "
            "impaired BMP-SMAD anti-proliferative signalling in pulmonary arterial endothelial cells (PAECs) + "
            "smooth muscle cells (PASMCs) → pathological remodelling → obliterative arteriopathy → "
            "elevated PVR → right ventricular failure. "
            "PVOD (EIF2AK4): distinct mechanism — pulmonary VENOUS occlusion (post-capillary), NOT arterial; "
            "clinically similar presentation but vasodilators provoke pulmonary oedema."
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
            for g in PAH_GENES
        },
        "glossary": {
            "Pulmonary arterial hypertension (PAH)": "mPAP ≥20 mmHg + PVR ≥3 WU + PAWP ≤15 mmHg on RHC (WHO 2022 criteria); Group 1 = arterial; distinct from Groups 2-5",
            "Right heart catheterisation (RHC)": "Gold standard for PAH diagnosis — measures mPAP, PAWP, PVR, CO; echocardiography is screening only, NOT diagnostic",
            "Pulmonary vascular resistance (PVR)": "= (mPAP - PAWP) / CO; units = Wood units (WU) or dyn·s·cm⁻⁵; ≥3 WU = elevated",
            "PAWP (pulmonary artery wedge pressure)": "Surrogate for left atrial pressure; ≤15 mmHg confirms pre-capillary PAH; >15 mmHg = post-capillary (left heart) cause",
            "PVOD (pulmonary veno-occlusive disease)": "Obliterative remodelling of pulmonary VEINS/venules; EIF2AK4 biallelic LOF = hereditary PVOD; vasodilators CONTRAINDICATED",
            "PCH (pulmonary capillary haemangiomatosis)": "Capillary proliferation in alveolar septa; closely related to PVOD; EIF2AK4 causes both PVOD and PCH (PVOD/PCH spectrum)",
            "BMP signalling": "Bone morphogenetic protein pathway; BMPs bind BMPRII + BMPRI (ALK1/ALK2/etc.) → SMAD1/5/9 phosphorylation → anti-proliferative transcription in PAECs",
            "Sotatercept": "Activin receptor ligand trap (ActRIIA-Fc fusion); rebalances BMP-activin ratio; FDA approved 2024 for Group 1 PAH (STELLAR trial)",
            "ERA (endothelin receptor antagonist)": "Ambrisentan (selective ETA), macitentan (dual ET-A/B), bosentan (dual): block endothelin-1 → vasodilation + anti-remodelling",
            "PDE5 inhibitor": "Sildenafil, tadalafil: block cGMP degradation → enhanced NO signalling → pulmonary vasodilation; approved for PAH",
            "Prostacyclin / prostacyclin analogues": "Epoprostenol IV (most potent, CONTINUOUS infusion), treprostinil (SC/IV/inhaled/oral), iloprost (inhaled); vasodilation + anti-platelet + anti-proliferative",
            "Vasodilator test": "Adenosine or inhaled NO during RHC; positive response (mPAP falls ≥10 to <40 mmHg) = CCB candidate; ABSOLUTELY CONTRAINDICATED in PVOD (EIF2AK4)",
            "6-minute walk distance (6MWD)": "PAH severity + prognostication marker; target >440m for low-risk; <165m = high-risk; decline on treatment = escalate",
            "NT-proBNP": "Right ventricular stress biomarker; target <300 pg/mL (low risk) on therapy; rising = deterioration",
            "HRCT PVOD pattern": "Centrilobular ground-glass opacities + interlobular septal thickening + mediastinal lymphadenopathy = PVOD PATHOGNOMONIC; absent in arterial PAH",
            "Hepatic AVM (HHT)": "Arteriovenous malformation in liver — common in ACVRL1-HHT2; causes high-output cardiac failure; distinguish from PAH by RHC (CO elevated, PAWP elevated)",
            "PAVM (pulmonary AVM)": "Pulmonary arteriovenous malformation — ENG-HHT1 predominantly; causes R-to-L shunt → hypoxaemia + paradoxical embolism → stroke/abscess",
            "Paradoxical embolism": "Embolus crosses from venous to arterial circulation via cardiac defect or PAVM; in HHT1 (ENG): bacteria/clot crosses PAVM → brain abscess/stroke",
            "Cascade genetic testing": "Systematic genetic testing of relatives of index case; BMPR2 20% penetrance means 80% of carriers clinically unaffected — cascade essential",
            "TASK-1 (KCNK3)": "Two-pore domain (K2P) background K+ channel; open at rest → stabilises resting membrane potential in PASMCs; GOF variants reduce current → depolarisation → vasoconstriction",
            "TBX4": "T-box transcription factor involved in lung morphogenesis; haploinsufficiency → acinar dysplasia + PAH; most common HPAH gene in children after BMPR2",
            "Small patella syndrome": "Haploinsufficiency of TBX4 → absent/hypoplastic patellae + tarsal abnormalities; ~50% of TBX4-PAH — many have NO musculoskeletal features",
            "EIF2AK4 (GCN2)": "Serine/threonine kinase — integrated stress response; biallelic LOF → impaired amino-acid deprivation response → pulmonary venous endothelial stress → PVOD",
        },
        "surveillance_protocols": {
            "BMPR2": "RHC at diagnosis; echo 6-monthly; 6MWD + NT-proBNP 3-monthly; cascade testing all relatives; AVOID pregnancy",
            "ACVRL1": "Annual echo (HHT + PAH screen); hepatic USS; RHC if dyspnoeic; ENT for epistaxis; cascade testing",
            "ENG": "CTPA at diagnosis (PAVM); bubble echo; annual echo; RHC if dyspnoea; brain MRI; antibiotic prophylaxis before dental",
            "SMAD9": "Same as BMPR2 — echo 6-monthly; 6MWD + NT-proBNP; cascade testing",
            "CAV1": "Echo 6-monthly; metabolic screen (lipodystrophy); cascade testing; RHC at diagnosis",
            "KCNK3": "Echo 6-monthly; 6MWD + NT-proBNP; AVOID bupivacaine; cascade testing; doxapram trial eligibility",
            "TBX4": "Paediatric PAH protocol; echo 3-monthly; 6MWD (if ambulant); knee X-ray at diagnosis; sibling screening",
            "EIF2AK4": "HRCT 6-monthly; DLCO 3-monthly; BAL if uncertain; LIST FOR TRANSPLANT EARLY; AVOID all vasodilators",
        },
    }


if __name__ == "__main__":
    import json
    ov = overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"mPAP elevated: {ov['mPAP_elevated_patients']}")
    print(f"PVOD pattern patients: {ov['pvod_pattern_patients']}")
    print(f"Transplant patients: {ov['transplant_patients']}")
    print(f"HHT features: {ov['hht_features_patients']}")
    print(f"Childhood onset: {ov['childhood_onset_patients']}")
    print("Breakdown gene keys:", list(breakdown().keys()))
