#!/usr/bin/env python3
"""Hereditary-Cardiac-Arrhythmia-Atlas — Complete 8-Gene Hereditary Cardiac Arrhythmia Atlas
KCNQ1  (KvLQT1/Kv7.1 IKs channel; 676 aa; 11p15.5; AD/AR;
         LQT1 Romano-Ward / JLNS — 30–35% of all LQTS; swimming triggers;
         beta-blockers 97% efficacy; AR biallelic = JLNS deafness;
         seed SEED_BASE+0) ·
KCNH2  (hERG/Kv11.1 IKr channel; 1159 aa; 7q36.1; AD;
         LQT2 — 25–30% of LQTS; sudden auditory arousal triggers;
         >200 QT-prolonging drugs; hypokalemia synergistic;
         seed SEED_BASE+1) ·
SCN5A  (Nav1.5 cardiac Na channel; 2016 aa; 3p22.2; AD;
         LQT3 (GOF) / Brugada Syndrome (LOF) / Lev-Lenègre CCS;
         mexiletine for LQT3; Na-channel blockers ABSOLUTE CI in BrS;
         seed SEED_BASE+2) ·
CALM1  (Calmodulin 1; 149 aa; 14q32.11; AD de novo;
         Calmodulinopathy — LQT14 + CPVT phenotype; lethal perinatal;
         flecainide adjunct; rarest but most severe arrhythmia syndrome;
         seed SEED_BASE+3) ·
RYR2   (Ryanodine receptor 2; 4967 aa; 1q43; AD;
         CPVT1 — catecholaminergic polymorphic VT; bidirectional VT PATHOGNOMONIC;
         exercise restriction MANDATORY; beta-blockers + flecainide; 60–70% of CPVT;
         seed SEED_BASE+4) ·
CASQ2  (Calsequestrin 2; 399 aa; 1p13.3; AR;
         CPVT2 — 5% of CPVT; biallelic; same treatment as CPVT1; SR Ca-buffering;
         seed SEED_BASE+5) ·
KCNJ2  (Kir2.1 IK1 channel; 427 aa; 17q24.3; AD;
         Andersen-Tawil Syndrome LQT7 — triad periodic paralysis + VT + dysmorphia PATHOGNOMONIC;
         quinidine/flecainide VT; acetazolamide paralysis; DO NOT miss triad;
         seed SEED_BASE+6) ·
HCN4   (HCN4 If-current pacemaker channel; 1203 aa; 15q24.1; AD;
         Sick Sinus Syndrome type 2 / sinus bradycardia; LV non-compaction overlap;
         pacemaker implantation; ivabradine CONTRAINDICATED (worsens SSS);
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1662–1669)
"""

import random

SEED_BASE = 1662

ARRHYTHMIA_GENES = [
    # ── KCNQ1 — LQT1 / Romano-Ward / JLNS ──────────────────────────────────
    {
        "gene": "KCNQ1",
        "protein": "KCNQ1 — LQT1 AD / JLNS AR — IKs Kv7.1 Channel — 30–35% of LQTS — Swimming Triggers — Beta-Blockers 97% Efficacy — JLNS Biallelic Deafness",
        "alias": (
            "KCNQ1 (KvLQT1); OMIM gene 607542; Romano-Ward LQTS type 1 OMIM 192500; "
            "Jervell and Lange-Nielsen syndrome (JLNS) OMIM 220400 (biallelic). "
            "11p15.5; 676 aa; ~75 kDa; AD (heterozygous LOF) — Romano-Ward; AR (biallelic LOF) — JLNS. "
            "FUNCTION: KCNQ1 encodes the α-subunit of the cardiac slow delayed rectifier K+ channel (IKs/Kv7.1). "
            "Assembles as tetramer (α4) with KCNE1 (minK, β-subunit) → IKs channel complex. "
            "IKs is the dominant repolarisation reserve during sympathetic activation: "
            "β-adrenergic → PKA phosphorylates KCNQ1 S27/S92 → increased IKs amplitude → faster repolarisation. "
            "LOF KCNQ1: IKs reduced → phase 3 repolarisation delayed → QTc prolongation → early afterdepolarisations → "
            "triggered activity → torsade de pointes (TdP) VT. "
            "EPIDEMIOLOGY: Most common LQTS subtype (30–35% of all LQTS). "
            "Prevalence: 1 in 2,000 (LQTS overall); LQT1 accounts for ~1/3. "
            "CLINICAL PHENOTYPE: Adrenergic/exercise triggers (vs LQT2 auditory, LQT3 sleep). "
            "SWIMMING IS THE MOST CHARACTERISTIC LQT1 TRIGGER: "
            "Cold water → sympathetic burst → IKs demand surge → vulnerable in LQT1; "
            "swimming syncope/cardiac arrest must trigger KCNQ1 germline testing immediately. "
            "Triggers: exercise (swimming > running), stress, emotion; less: auditory, sleep. "
            "ECG: Broad-based T wave (reduced IKs amplitude → smooth shoulder-to-peak transition). "
            "JLNS (biallelic): 0.5–1 per million; most severe LQTS (mean QTc ~550 ms); "
            "sensorineural hearing loss (SNHL) in both ears (KCNQ1 expressed in stria vascularis); "
            "cardiac events in 50% by age 3; ICD often required early. "
            "BETA-BLOCKER EFFICACY: Propranolol/nadolol — reduce adrenergic IKs demand. "
            "LQT1 responds best to beta-blockers among all LQTS subtypes: "
            "Moss 2000 (NEJM): beta-blockers reduce cardiac events 97% vs 38% (LQT2) in LQT1; "
            "nadolol preferred (non-selective, longer half-life, once-daily adherence). "
            "ICD INDICATIONS (HRS/EHRA/APHRS 2013, updated 2022): "
            "Symptomatic LQT1 on maximal beta-blockers; cardiac arrest survivors; "
            "QTc >500 ms + symptoms; JLNS with symptomatic events. "
            "LIFESTYLE: No competitive sports (Class I recommendation); avoid QT-prolonging drugs "
            "(www.crediblemeds.org — mandatory prescriber alert); no diuretics without electrolyte monitoring; "
            "no sudden temperature changes; "
            "SCHWARTZ-SCORE: QTc + clinical features used for diagnosis before genetic confirmation. "
            "GENETICS: Point mutations, small indels, splicing; A-domain mutations → trafficking defect; "
            "C-terminus IKs complex assembly mutations; large deletions by MLPA in ~5%. "
            "CASCADE TESTING: First-degree relatives mandatory — 50% asymptomatic carriers at lethal QTc risk."
        ),
        "locus": "11p15.5",
        "aa": 676,
        "kDa": 75,
        "omim_gene": "607542",
        "omim_disease": "Long QT Syndrome type 1 Romano-Ward (OMIM 192500); Jervell and Lange-Nielsen Syndrome 1 (OMIM 220400)",
        "inheritance": "AD heterozygous LOF (Romano-Ward LQT1); AR biallelic LOF (JLNS — profound SNHL + severe LQT)",
        "gene_class": "Kv7.1 IKs voltage-gated K+ channel α-subunit — slow delayed rectifier — adrenergic repolarisation reserve",
        "key_alerts": [
            "KCNQ1-SWIMMING-TRIGGER-CARDINAL: Swimming in cold water is the MOST CHARACTERISTIC LQT1 trigger (adrenergic burst + temperature stress) — any swimming syncope warrants KCNQ1 germline testing; competitive swimming PROHIBITED",
            "KCNQ1-BETA-BLOCKERS-97PCT: Propranolol/nadolol reduce LQT1 cardiac events by 97% (Moss 2000 NEJM) — highest efficacy of beta-blockers across all LQTS subtypes; nadolol preferred for once-daily adherence",
            "KCNQ1-JLNS-BIALLELIC-CRITICAL: Biallelic KCNQ1 = Jervell and Lange-Nielsen — congenital profound bilateral SNHL + mean QTc ~550 ms + 50% cardiac events by age 3; ICD required early; do NOT dismiss deafness as incidental in LQTS",
            "KCNQ1-QT-PROLONGING-DRUGS-FORBIDDEN: crediblemeds.org list — mandatory prescriber alert for all LQT1 carriers; antipsychotics/antiemetics/macrolides most common offenders; document in medical records",
            "KCNQ1-CASCADE-TESTING-MANDATORY: 50% of first-degree relatives are carriers; asymptomatic carriers can have QTc >480 ms; family cascade testing prevents sudden death in undiagnosed relatives",
        ],
        "etiologies": [
            "KCNQ1 LOF heterozygous → IKs channel haploinsufficiency → phase 3 repolarisation delayed → QTc prolongation → TdP",
            "KCNQ1 trafficking mutations (A-domain) → ER retention of Kv7.1 → reduced surface IKs → dominant-negative effect in some mutations",
            "Biallelic KCNQ1 LOF → complete IKs abolition → JLNS (severe LQT + SNHL via stria vascularis K+ recycling failure)",
            "Exercise/sympathetic activation → adrenergic PKA demand on IKs cannot be met (LOF) → TdP during peak heart rate",
        ],
        "stats": {
            "mean_dx_age": 22,
            "mean_dx_delay_months": 14,
            "lqts_prevalence_pct": 35,
            "beta_blocker_efficacy_pct": 97,
            "jlns_biallelic_pct": 5,
            "lifetime_event_risk_untreated_pct": 30,
        },
        "dx_delay_distribution": "6–24 months (syncope attributed to vasovagal; QTc measurement mandatory in all unexplained syncope)",
    },

    # ── KCNH2 — LQT2 / hERG ─────────────────────────────────────────────────
    {
        "gene": "KCNH2",
        "protein": "KCNH2 — LQT2 AD — hERG/Kv11.1 IKr Channel — 25–30% of LQTS — Auditory Arousal Triggers — >200 QT-Prolonging Drugs — Hypokalemia Synergistic",
        "alias": (
            "KCNH2 (hERG, ERG1, HERG); OMIM gene 152427; LQTS type 2 OMIM 613688. "
            "7q36.1; 1159 aa; ~127 kDa; AD missense/truncating LOF. "
            "FUNCTION: KCNH2 encodes the α-subunit of the cardiac rapid delayed rectifier K+ channel (IKr). "
            "IKr (hERG/Kv11.1) is the primary repolarising current in ventricular myocytes during phase 3. "
            "UNIQUE GATING PROPERTY: hERG channels have unusually rapid inactivation (C-type) → "
            "reduced current during plateau (protects from premature repolarisation) → "
            "rapid recovery from inactivation on repolarisation → large surge of IKr during phase 3 → "
            "most efficient repolarising current. "
            "This rapid inactivation makes hERG UNIQUELY SENSITIVE to drug block: "
            "the drug-binding site (Y652, F656) is exposed in the open-inactivated state — "
            "accessible to >200 structurally diverse drugs. "
            "LOF KCNH2: IKr reduced → delayed phase 3 repolarisation → QTc prolongation → TdP. "
            "EPIDEMIOLOGY: 2nd most common LQTS (25–30%); mean QTc ~480 ms (heterozygous). "
            "CLINICAL PHENOTYPE — AUDITORY AROUSAL TRIGGERS: "
            "Sudden sounds (alarm clock, telephone, doorbell) → sympathetic arousal → "
            "IKr demand (adrenergic) cannot be met → TdP; "
            "NOT exercise-predominant (unlike LQT1); "
            "ALSO sleep/rest events (less adrenergic than LQT1 triggers, but auditory-specific risk). "
            "ECG: Low-amplitude, notched, or bifid T wave with prominent U wave; QTu interval prolongation. "
            "DRUG-INDUCED LQTS: KCNH2 variants (even functional ones) are the #1 substrate for "
            "acquired/drug-induced LQTS. hERG block by drugs is synergistic with KCNH2 heterozygous LOF. "
            "Key offenders: Class IA antiarrhythmics (quinidine, procainamide, disopyramide — ABSOLUTE CI); "
            "Class III (sotalol, dofetilide, amiodarone — monitor); "
            "antipsychotics (haloperidol, thioridazine, ziprasidone); "
            "antiemetics (domperidone, ondansetron); macrolides (azithromycin); "
            "fluoroquinolones (moxifloxacin); antimalarials (hydroxychloroquine). "
            "HYPOKALEMIA SYNERGISTIC RISK: IKr amplitude depends on extracellular K+ (paradoxical — "
            "lower [K+]o → reduced IKr amplitude in hERG); "
            "loop diuretics → hypokalemia → compound LQT2 → TdP; "
            "K+ supplementation target: serum K+ 4.0–4.5 mEq/L; Mg2+ supplementation co-administered. "
            "ALARM CLOCKS: Remove auditory alarms where possible; vibrating alarm alternative; "
            "bedside telephone (move away from bed). "
            "BETA-BLOCKERS: Effective but less so than LQT1 (38% reduction vs 97%); "
            "still recommended as first-line pharmacotherapy. "
            "MEXILETINE IN LQT2: emerging data for LQT2 (IKs-independent mechanism of QT shortening — "
            "mexiletine blocks late INa, different mechanism from LQT3 but may benefit). "
            "GENETICS: Missense most common; C-terminus missense → trafficking defects; "
            "N-terminal PAS domain mutations → altered deactivation kinetics; "
            "MLPA for deletions; full gene sequencing + splicing analysis required."
        ),
        "locus": "7q36.1",
        "aa": 1159,
        "kDa": 127,
        "omim_gene": "152427",
        "omim_disease": "Long QT Syndrome type 2 (OMIM 613688); Short QT Syndrome type 1 GOF (OMIM 609620)",
        "inheritance": "AD heterozygous LOF (LQT2); AD GOF mutations cause Short QT syndrome (rare)",
        "gene_class": "hERG/Kv11.1 IKr voltage-gated K+ channel α-subunit — rapid delayed rectifier — primary ventricular repolarisation current",
        "key_alerts": [
            "KCNH2-AUDITORY-AROUSAL-TRIGGER: Sudden auditory stimuli (alarm, phone) are the CARDINAL LQT2 trigger — vibrating alarms MANDATORY; telephone by bedside removed; QTc monitoring with any new medication",
            "KCNH2-DRUG-TRIGGER-CRITICAL: >200 drugs block hERG — crediblemeds.org mandatory check before ANY new prescription; Class IA/III antiarrhythmics ABSOLUTE CI in LQT2; domperidone/ondansetron require ECG monitoring",
            "KCNH2-HYPOKALEMIA-SYNERGISTIC: Target serum K+ 4.0–4.5 mEq/L; Mg2+ supplementation; avoid loop diuretics without K+ replacement; hypokalemia + LQT2 = high TdP risk",
            "KCNH2-ACQUIRED-LQTS-SUBSTRATE: Heterozygous KCNH2 variants markedly increase drug-induced LQTS risk (reduced repolarisation reserve); document in drug alert system; anaesthesia/ICU protocols needed",
            "KCNH2-BETA-BLOCKER-LESS-EFFECTIVE: Beta-blockers 38% event reduction in LQT2 (vs 97% in LQT1) — higher residual risk; lower ICD threshold vs LQT1 for symptomatic carriers on beta-blockers",
        ],
        "etiologies": [
            "KCNH2 LOF missense → reduced IKr current density → phase 3 delayed → QTc prolongation → early afterdepolarisations → TdP",
            "KCNH2 trafficking mutations (C-terminus) → ER retention of hERG tetramer → <50% surface channel density → dominant-negative effect",
            "Drug blockade synergy with KCNH2 heterozygous LOF → additive IKr reduction → acquired LQTS on genetic substrate",
            "Hypokalemia → reduced [K+]o → paradoxical hERG IKr amplitude reduction → compound repolarisation failure in LQT2",
        ],
        "stats": {
            "mean_dx_age": 24,
            "mean_dx_delay_months": 18,
            "lqts_prevalence_pct": 28,
            "beta_blocker_efficacy_pct": 38,
            "drug_trigger_drugs_count": 200,
            "lifetime_event_risk_untreated_pct": 40,
        },
        "dx_delay_distribution": "12–30 months (drug-induced TdP may be the presenting event; often misattributed to structural heart disease)",
    },

    # ── SCN5A — LQT3 / Brugada Syndrome / Lev-Lenègre ──────────────────────
    {
        "gene": "SCN5A",
        "protein": "SCN5A — LQT3 AD-GOF / Brugada-Syndrome1 AD-LOF — Nav1.5 Cardiac Na Channel — 2016aa — Mexiletine-LQT3 — Na-Channel-Blockers-ABSOLUTE-CI-BrS — Fever-ABSOLUTE-CI-BrS",
        "alias": (
            "SCN5A; OMIM gene 600163; LQT3 OMIM 603830; Brugada Syndrome type 1 OMIM 601144; "
            "Conduction System Disease (Lev-Lenègre) OMIM 113900. "
            "3p22.2; 2016 aa; ~227 kDa; AD — multiple phenotypes depending on mutation direction. "
            "FUNCTION: SCN5A encodes the cardiac voltage-gated Na+ channel α-subunit (Nav1.5). "
            "Nav1.5 is responsible for: phase 0 rapid depolarisation (INa) — action potential upstroke; "
            "impulse conduction velocity (CV) in His-Purkinje system and ventricular myocardium. "
            "DUAL PHENOTYPES FROM OPPOSITE MUTATION DIRECTIONS: "
            "GOF mutations (persistent late INa, delayed inactivation) → LQT3. "
            "LOF mutations (reduced INa) → Brugada Syndrome / Lev-Lenègre (CCS disease). "
            "LQT3 (GOF — PERSISTENT LATE INa): "
            "Persistent late INa → prolonged depolarisation (plateau) → QTc prolongation → "
            "pause-dependent TdP (mainly at rest/sleep — bradycardia-dependent; different from LQT1/2). "
            "ECG in LQT3: late-onset T wave with a notch or biphasic morphology; 'short QT but tall T'. "
            "MEXILETINE (Class IB) FOR LQT3: "
            "Mexiletine is the ONLY approved oral drug specifically shortening QTc in LQT3: "
            "blocks persistent late INa (use-dependent Na channel blocker) → reduces QT prolongation "
            "without blocking peak INa significantly → shortens QTc by 20–40 ms; "
            "clinical data: Schwartz 1995 (Circulation), Mazzanti 2016 (JACC); "
            "response to mexiletine predicts SCN5A-specific LQT3 (vs other subtypes); "
            "ranolazine also blocks late INa but less LQT3-specific data than mexiletine; "
            "DO NOT use Class IA or IC agents for LQT3 (may paradoxically worsen late INa in some mutations). "
            "BRUGADA SYNDROME (LOF — REDUCED PEAK INa): "
            "Prevalence: 1 in 2,000; predominantly men (8:1 male:female — testosterone reduces INa reserve); "
            "Typical age of presentation: 30–50 years; "
            "SCD risk stratification: symptomatic (CA survivor or unexplained syncope) — ICD Category I; "
            "asymptomatic — lower risk (1–2%/year); risk stratification by EP study debated (Brugada-Risk score). "
            "BRUGADA ECG (TYPE 1): coved ST-segment elevation ≥2 mm in ≥1 right precordial lead (V1–V2) "
            "with negative T wave — SPONTANEOUS TYPE 1 IS DIAGNOSTIC. "
            "Sodium channel blockers UNMASK TYPE 1 ECG in concealed BrS: "
            "ajmaline IV, flecainide PO/IV, pilsicainide — diagnostic use ONLY; "
            "FLECAINIDE IS ABSOLUTELY CONTRAINDICATED AS TREATMENT in BrS "
            "(further INa reduction → VF risk). "
            "FEVER IN BRUGADA — ABSOLUTE EMERGENCY: Fever → INa further reduced → "
            "TYPE 1 ECG unmasked → VF storm even in previously asymptomatic BrS; "
            "treat ANY fever aggressively with antipyretics in BrS patients; "
            "ICU admission for febrile illness with core temp >38°C; "
            "wearable defibrillator vest during febrile illness for unimplanted BrS. "
            "QUINIDINE FOR BrS: quinidine (Ito blocker + INa blocker at toxic doses) — "
            "reduces VT/VF in BrS (Brugada 2004); used for electrical storms in BrS; "
            "isoproterenol IV for acute electrical storm (increases INa + ICa-L). "
            "LEV-LENÈGRE CARDIAC CONDUCTION DISEASE: SCN5A LOF → slow conduction velocity → "
            "progressive His-Purkinje disease → bundle branch blocks → AV block → pacemaker. "
            "GENETICS: >300 SCN5A mutations; missense most common; MLPA for exon deletions; "
            "overlapping phenotypes (one mutation can cause both BrS + CCS in same family); "
            "compound heterozygosity increases severity."
        ),
        "locus": "3p22.2",
        "aa": 2016,
        "kDa": 227,
        "omim_gene": "600163",
        "omim_disease": "Long QT Syndrome type 3 (OMIM 603830); Brugada Syndrome type 1 (OMIM 601144); Sick Sinus Syndrome type 1 (OMIM 608567); Conduction Disease Lev-Lenègre (OMIM 113900)",
        "inheritance": "AD — GOF mutations cause LQT3 (persistent late INa); LOF mutations cause Brugada/CCS/SSS1; rare overlap phenotypes in same family",
        "gene_class": "Nav1.5 cardiac voltage-gated Na+ channel α-subunit — phase 0 depolarisation — conduction velocity — most pleiotropic cardiac channelopathy gene",
        "key_alerts": [
            "SCN5A-NA-CHANNEL-BLOCKERS-ABSOLUTE-CI-BrS: Flecainide, ajmaline, pilsicainide, procainamide ABSOLUTELY CONTRAINDICATED as treatment in Brugada — further INa reduction → VF storm; Na-channel blockers used for DIAGNOSIS ONLY (under monitored conditions)",
            "SCN5A-FEVER-ABSOLUTE-EMERGENCY-BrS: Fever >38°C in Brugada = medical emergency; temperature-dependent Nav1.5 gating worsens INa reduction → Type 1 ECG + VF; treat fever aggressively with antipyretics; ICU admission for febrile illness in Brugada",
            "SCN5A-MEXILETINE-LQT3-SPECIFIC: Mexiletine shortens QTc by 20–40 ms in LQT3 by blocking persistent late INa; response to mexiletine CONFIRMS LQT3 SCN5A mechanism; not effective for LQT1/LQT2 (different mechanisms)",
            "SCN5A-MALE-8:1-BrS: Brugada predominantly male (testosterone increases Ito/Kv4.3, reducing repolarisation reserve that compensates reduced INa); female carriers may be asymptomatic but still transmit; males require more aggressive risk stratification",
            "SCN5A-DUAL-PHENOTYPE-SINGLE-FAMILY: One family can have both LQT3 and Brugada in different members depending on modifier genes; genetic testing of ALL relatives critical; do NOT assume same phenotype within family from one proband",
        ],
        "etiologies": [
            "SCN5A GOF (persistent late INa) → prolonged plateau phase → QTc prolongation → pause-dependent TdP at rest/sleep (LQT3)",
            "SCN5A LOF (reduced peak INa) → reduced conduction velocity + epicardial Ito-IKr imbalance → phase 2 reentry → VF in right ventricle (Brugada Syndrome)",
            "SCN5A LOF progressive → His-Purkinje conduction slowing → PR prolongation, BBB, AV block → progressive CCS (Lev-Lenègre)",
            "Environmental modifier (fever, drugs) + SCN5A LOF → unmasking of concealed Brugada ECG → arrhythmia threshold crossed",
        ],
        "stats": {
            "mean_dx_age": 28,
            "mean_dx_delay_months": 24,
            "brugada_prevalence_per_2000": 1,
            "brugada_male_predominance_ratio": "8:1",
            "mexiletine_qtc_shortening_ms": "20–40",
            "lifetime_event_risk_symptomatic_pct": 10,
        },
        "dx_delay_distribution": "18–36 months (Brugada often asymptomatic until SCD; LQT3 events nocturnal and may be missed until autopsy review)",
    },

    # ── CALM1 — Calmodulinopathy (LQT14 / CPVT-overlap) ────────────────────
    {
        "gene": "CALM1",
        "protein": "CALM1 — Calmodulinopathy AD-de-novo — LQT14+CPVT-Overlap — 149aa — 14q32.11 — Lethal-Perinatal — Flecainide-Adjunct — Rarest-Most-Severe",
        "alias": (
            "CALM1 (Calmodulin 1); OMIM gene 114180; Calmodulinopathy with LQT (LQTS14) OMIM 616247; "
            "Calmodulinopathy with CPVT OMIM 615441. "
            "14q32.11; 149 aa; ~17 kDa; AD (almost exclusively de novo — autosomal dominant de novo). "
            "FUNCTION: Calmodulin (CaM) is the universal intracellular calcium sensor: "
            "149 aa, 4 EF-hand Ca2+-binding motifs (paired: EF1/EF2 N-lobe, EF3/EF4 C-lobe); "
            "CaM regulates a vast array of Ca2+-dependent processes; "
            "in the heart, CaM modulates: "
            "(1) KCNH2 (hERG/IKr) — Ca2+-dependent inactivation via C-lobe EF3/4 binding; "
            "(2) RYR2 (ryanodine receptor 2) — calmodulin inhibits SR Ca2+ release at high Ca2+; "
            "(3) CaMKII (calmodulin kinase II) activation — phosphorylation of RYR2, PLN, INa. "
            "CALMODULINOPATHY MUTATIONS: Clustered at EF-hand Ca2+-binding residues (D96V, N98S, D130G, F142L, D96H, E141G): "
            "reduced Ca2+ affinity of CaM mutants → fails to inhibit RYR2 → uncontrolled SR Ca2+ release; "
            "fails to mediate KCNH2 Ca2+-dependent inactivation → IKr prolonged → QTc extreme prolongation; "
            "combined RYR2-CaM + KCNH2-CaM dysfunction explains MIXED LQT + CPVT phenotype. "
            "CLINICAL PHENOTYPE (MOST SEVERE CHANNELOPATHY PRESENTATION): "
            "Extreme QTc prolongation (mean QTc 600–650 ms — highest of any LQTS subtype); "
            "onset: perinatal/neonatal — ventricular tachycardia (VT), sinus arrest, cardiogenic shock in utero or day 1; "
            "INTRAUTERINE HYDROPS, neonatal VT storm, complete AV block — obstetric emergency; "
            "bidirectional VT (as in CPVT) on adrenergic stimulation — CPVT-like overlap; "
            "intellectual disability in survivors (secondary to hypoxic-ischaemic events); "
            "mortality without treatment: ~50% in first year. "
            "TREATMENT STRATEGY (COMPLEX): "
            "Beta-blockers (nadolol) — suppress adrenergic RYR2 dysregulation; "
            "FLECAINIDE — adjunct (class IC) targets RYR2 directly at cardiac ryanodine receptor (open-channel block of RYR2) "
            "+ mild KCNH2/IKr augmentation → dual benefit in calmodulinopathy; "
            "Verapamil — reduces Ca2+ influx → less RYR2 Ca2+ overload; "
            "ICD — mandatory in symptomatic survivors; "
            "LEFT CARDIAC SYMPATHETIC DENERVATION (LCSD) — surgical or EUS-guided stellate ganglion block "
            "for refractory VT storms; adjunct to ICD; "
            "Mexiletine — blocks late INa in some CALM1 mutations (SCN5A functional overlap via CaM-Nav1.5 interaction). "
            "GENETIC TESTING IMPERATIVE: CALM1, CALM2 (11p13), CALM3 (19q13.32) — "
            "all three CaM isoforms identical protein, different loci; "
            "WES/WGS recommended — panel sequencing may miss CALM2/3; "
            "de novo confirmation by parental testing; recurrence risk <1% (de novo). "
            "NOTE: Calmodulinopathy is rarer than LQT1/2/3/CPVT1 but is arguably the most severe "
            "heritable cardiac arrhythmia syndrome (fetal and neonatal presentation, extreme QTc, high mortality)."
        ),
        "locus": "14q32.11",
        "aa": 149,
        "kDa": 17,
        "omim_gene": "114180",
        "omim_disease": "Calmodulinopathy with Long QT (LQTS14, OMIM 616247); Calmodulinopathy with CPVT (OMIM 615441); Calmodulinopathy with idiopathic VF",
        "inheritance": "AD de novo (almost exclusively — parental testing negative; recurrence risk <1%); rarely inherited from low-penetrance mosaic parent",
        "gene_class": "Universal Ca2+-sensor EF-hand protein — regulates RYR2 (SR Ca release) + KCNH2 (IKr Ca-inactivation) + CaMKII — most severe heritable arrhythmia gene",
        "key_alerts": [
            "CALM1-FETAL-NEONATAL-LETHAL: Calmodulinopathy presents in utero (hydrops, fetal VT) or neonatal period (VT storm, complete AV block, cardiogenic shock) — obstetric/neonatal cardiology emergency; highest mortality of all heritable arrhythmia syndromes",
            "CALM1-EXTREME-QTC: Mean QTc 600–650 ms in CALM1 calmodulinopathy (highest of any LQTS subtype) — extreme QTc in neonate/infant with VT → WES mandatory; CALM1/CALM2/CALM3 all three loci must be tested",
            "CALM1-FLECAINIDE-RYR2-DUAL: Flecainide blocks RYR2 (open-channel) + augments IKr in calmodulinopathy — dual mechanism benefit; use as adjunct to beta-blockers; unlike CPVT1 where flecainide is purely RYR2-targeted",
            "CALM1-DE-NOVO-CONFIRM: Almost all CALM1 calmodulinopathy mutations are de novo — parental testing mandatory to confirm (rules out familial transmission and low-penetrance mosaicism); recurrence risk in siblings <1%",
            "CALM1-LCSD-REFRACTORY-VT: Left cardiac sympathetic denervation (LCSD) is adjunct for electrical storm refractory to beta-blocker + flecainide + ICD; EUS-guided stellate ganglion block for acute VT storm",
        ],
        "etiologies": [
            "CALM1 EF-hand mutation → reduced Ca2+ affinity → fails to inhibit RYR2 at high cytosolic Ca2+ → uncontrolled SR Ca2+ sparks → delayed afterdepolarisations → VT",
            "CALM1 mutation → fails to mediate KCNH2 Ca2+-dependent inactivation → IKr prolongation → extreme QTc → TdP",
            "CaM mutant → CaMKII activation dysregulated → hyperphosphorylation of RYR2 S2808/S2814 → open probability increased → Ca2+ leak → bidirectional VT",
            "CaM-Nav1.5 interaction disrupted in some CALM1 mutations → late INa increase → additive QTc prolongation",
        ],
        "stats": {
            "mean_dx_age": 4,
            "mean_dx_delay_months": 6,
            "mean_qtc_ms": 625,
            "neonatal_presentation_pct": 70,
            "de_novo_pct": 95,
            "mortality_without_treatment_pct": 50,
        },
        "dx_delay_distribution": "2–12 months (most present acutely in neonatal period; delay due to missed CALM locus testing — CALM2/CALM3 often not included in basic LQTS panels)",
    },

    # ── RYR2 — CPVT1 ────────────────────────────────────────────────────────
    {
        "gene": "RYR2",
        "protein": "RYR2 — CPVT1 AD — Ryanodine-Receptor-2 SR-Ca-Release — 4967aa — 1q43 — Bidirectional-VT-PATHOGNOMONIC — Exercise-Restriction-MANDATORY — Beta-Blockers+Flecainide",
        "alias": (
            "RYR2 (Cardiac Ryanodine Receptor 2); OMIM gene 180902; CPVT1 OMIM 604772. "
            "1q43; 4967 aa; ~560 kDa homotetrameric macrocomplex; AD missense GOF mutations (most). "
            "FUNCTION: RYR2 is the sarcoplasmic reticulum (SR) Ca2+ release channel in cardiomyocytes. "
            "Excitation-contraction (EC) coupling: "
            "Action potential → L-type Ca2+ channel (DHPR/Cav1.2) → Ca2+ entry → "
            "Ca2+-induced Ca2+ release (CICR) via RYR2 → cytosolic Ca2+ transient → "
            "troponin C activation → cross-bridge cycling → contraction. "
            "RYR2 is regulated by: "
            "Calmodulin (inhibitory at high [Ca2+]i); FKBP12.6 (stabilises closed state); "
            "PKA phosphorylation (S2808) — β-adrenergic → increased open probability; "
            "CaMKII phosphorylation (S2814) — exercise/stress → further opening; "
            "calsequestrin 2 (CASQ2) + triadin + junctin — SR Ca2+ buffering complex. "
            "CPVT1 GOF MECHANISM: RYR2 missense mutations destabilise closed state → "
            "diastolic Ca2+ leak (spontaneous SR Ca2+ release between beats) → "
            "cytosolic Ca2+ → NCX (Na+/Ca2+ exchanger) → inward current (INCX) → "
            "delayed afterdepolarisations (DADs) → triggered activity → VT; "
            "adrenergic activation (PKA, CaMKII phosphorylation of mutant RYR2) markedly worsens leak. "
            "BIDIRECTIONAL VENTRICULAR TACHYCARDIA (BiVT) — PATHOGNOMONIC: "
            "Alternating QRS morphology beat-to-beat (180° alternation of QRS axis) — "
            "from alternating fascicular/ventricular origins during DAD-triggered activity; "
            "ONLY CPVT (RYR2/CASQ2) and Andersen-Tawil syndrome (KCNJ2) produce BiVT reliably. "
            "CPVT DIAGNOSIS: "
            "(1) Exercise treadmill test (ETT): adrenergic stimulation → VT during or post-exercise; "
            "BiVT at heart rate threshold (usually 110–130 bpm) is diagnostic; "
            "(2) Epinephrine (adrenaline) bolus provocation — alternative to ETT in children; "
            "(3) Holter with exercise — ambulatory confirmation. "
            "NORMAL QTc at rest in CPVT (distinguishes from LQT subtypes). "
            "EXERCISE RESTRICTION — MANDATORY AND UNCONDITIONAL: "
            "No competitive sports; no moderate-high intensity exercise; "
            "SWIMMING, GYM CLASSES, SPORTS: PROHIBITED without ICD; "
            "exceptions only with ICD in situ AND documented stable pharmacotherapy response. "
            "PHARMACOTHERAPY: "
            "Nadolol/propranolol — non-selective beta-blockers first-line (beta-1 + beta-2 blockade); "
            "metoprolol INFERIOR (cardioselective — beta-2 still active → sympathetic escape); "
            "Flecainide — adjunct: CLASS IC Na-channel blocker ALSO blocks RYR2 directly at the open channel "
            "(Watanabe 2009, Nature Medicine) → reduces diastolic SR Ca2+ leak independent of INa block; "
            "flecainide + nadolol is the recommended combination for drug-refractory CPVT; "
            "Verapamil — Ca2+ channel blocker reduces Ca2+ entry → reduced SR Ca2+ load → less RYR2 opening; "
            "ICD — for refractory VT, cardiac arrest survivors; SHOCK CAN TRIGGER ADRENERGIC STORM → "
            "combine ICD with nadolol + flecainide; "
            "LCSD — Left Cardiac Sympathetic Denervation — for ICD storm/refractory CPVT. "
            "GENETICS: >300 RYR2 mutations; mostly missense in 4 hotspot domains (N-term, SPRY, central, C-term); "
            "rare autosomal recessive CPVT1 variants (compound heterozygous); CASQ2 mutations cause CPVT2."
        ),
        "locus": "1q43",
        "aa": 4967,
        "kDa": 560,
        "omim_gene": "180902",
        "omim_disease": "Catecholaminergic Polymorphic Ventricular Tachycardia type 1 CPVT1 (OMIM 604772); Arrhythmogenic Right Ventricular Cardiomyopathy type 2 ARVC2 (rare RYR2 mutations)",
        "inheritance": "AD GOF missense mutations (dominant-negative effect on RYR2 tetramer); rare AR biallelic (very severe early onset)",
        "gene_class": "Ryanodine Receptor 2 — SR Ca2+ release channel macrocomplex — EC coupling — GOF: diastolic Ca2+ leak → DADs → CPVT",
        "key_alerts": [
            "RYR2-BIDIRECTIONAL-VT-PATHOGNOMONIC: Bidirectional VT (alternating 180° QRS axis beat-to-beat) on exercise ECG = CPVT until proven otherwise; exercise treadmill test + RYR2 germline sequencing mandatory; normal QTc at rest does NOT exclude CPVT",
            "RYR2-EXERCISE-RESTRICTION-MANDATORY-UNCONDITIONAL: No competitive sports, no moderate-intensity exercise without ICD; swimming and gym prohibited; exercise restriction is the SINGLE most important intervention before pharmacotherapy is optimised",
            "RYR2-METOPROLOL-INFERIOR: Non-selective beta-blockers (nadolol/propranolol) mandatory — metoprolol is cardioselective (beta-1 only) → beta-2 adrenergic escape → catecholamine surge → CPVT VT; nadolol preferred once-daily adherence",
            "RYR2-FLECAINIDE-RYR2-DIRECT-BLOCK: Flecainide blocks RYR2 open-channel directly (independent of INa block) — reduces diastolic SR Ca2+ leak; add flecainide to nadolol for drug-refractory CPVT; Watanabe 2009 Nature Medicine mechanistic basis",
            "RYR2-ICD-SHOCK-TRIGGERS-VT-STORM: ICD shock pain/fear → catecholamine surge → more RYR2 VT → more shocks (storm); ALWAYS combine ICD with maximal pharmacotherapy (nadolol + flecainide) and LCSD if refractory; programme anti-tachycardia pacing (ATP) to avoid shocks where possible",
        ],
        "etiologies": [
            "RYR2 GOF missense → destabilised closed state → diastolic SR Ca2+ leak → NCX-INCX → DADs → triggered VT (adrenergic-dependent)",
            "PKA (β-adrenergic) phosphorylation of mutant RYR2 S2808 → further channel opening → CPVT threshold lowered during exercise",
            "CaMKII phosphorylation of mutant RYR2 S2814 → hyperphosphorylation-dependent Ca2+ leak → EAD/DAD alternation → BiVT morphology",
            "Calmodulin/FKBP12.6 dissociation from mutant RYR2 → loss of stabilising inhibition → spontaneous Ca2+ sparks → triggered activity",
        ],
        "stats": {
            "mean_dx_age": 12,
            "mean_dx_delay_months": 18,
            "cpvt_prevalence_pct": 70,
            "bidirectional_vt_sensitivity_pct": 85,
            "nadolol_efficacy_pct": 70,
            "nadolol_flecainide_combined_pct": 85,
        },
        "dx_delay_distribution": "12–24 months (exercise VT diagnosed as epilepsy/vasovagal; normal resting ECG/echo falsely reassures clinicians; ETT not ordered unless CPVT suspected)",
    },

    # ── CASQ2 — CPVT2 ───────────────────────────────────────────────────────
    {
        "gene": "CASQ2",
        "protein": "CASQ2 — CPVT2 AR — Calsequestrin-2 SR-Ca-Buffer — 399aa — 1p13.3 — 5pct-of-CPVT — Biallelic-LOF — Same-Treatment-CPVT1 — Severe-Pediatric",
        "alias": (
            "CASQ2 (Calsequestrin 2); OMIM gene 114251; CPVT2 OMIM 611938. "
            "1p13.3; 399 aa; ~46 kDa; AR biallelic LOF (recessive). "
            "FUNCTION: CASQ2 is the primary Ca2+-buffering protein of the junctional SR (jSR) in cardiomyocytes. "
            "Located within SR lumen, tethered by triadin-junctin scaffold adjacent to RYR2. "
            "Ca2+ buffering: CASQ2 polymer binds 40–50 Ca2+ ions per monomer with low affinity/high capacity → "
            "buffered Ca2+ storage prevents toxic free SR [Ca2+] spikes; "
            "CASQ2 also directly modulates RYR2 gating via triadin/junctin: "
            "high luminal Ca2+ (CASQ2 saturated) → dissociation of CASQ2-triadin complex → RYR2 activation; "
            "low luminal Ca2+ → CASQ2 reassociates with triadin → RYR2 inhibition. "
            "CPVT2 MECHANISM (LOF — recessive): "
            "Biallelic CASQ2 LOF → absent or misfolded CASQ2 → "
            "(1) Reduced SR Ca2+ buffering capacity → higher free [Ca2+]SR at rest → lower threshold for diastolic RYR2 opening; "
            "(2) Loss of RYR2-inhibitory signal from CASQ2-triadin complex → constitutively active RYR2; "
            "combined effects → identical downstream consequence to RYR2 GOF: "
            "diastolic Ca2+ leak → DADs → triggered VT on adrenergic stimulation. "
            "CLINICAL PRESENTATION: Essentially identical to CPVT1 (RYR2): "
            "exercise-induced, bidirectional VT, syncope, SCD in children/young adults; "
            "onset: 3–12 years (slightly younger than CPVT1 mean 12 years); "
            "CPVT2 accounts for ~5% of CPVT cases (vs RYR2 60–70%); "
            "rarest common cause but carries comparable lethality to CPVT1. "
            "Consanguinity common in CASQ2-CPVT2 — autosomal recessive: "
            "parents usually unaffected heterozygous carriers; "
            "D307H founder mutation in Bedouin families (Israel). "
            "TREATMENT — IDENTICAL TO CPVT1: "
            "Nadolol/propranolol (non-selective beta-blockers) first-line; "
            "METOPROLOL INFERIOR (same rationale as CPVT1); "
            "Flecainide adjunct (RYR2 direct block) — same mechanism applies regardless of CASQ2 LOF vs RYR2 GOF; "
            "Exercise restriction mandatory (same unconditional prohibition as CPVT1); "
            "ICD for cardiac arrest survivors / refractory VT; "
            "LCSD for electrical storm. "
            "KEY DISTINCTION FROM CPVT1 (RYR2): "
            "Inheritance is AUTOSOMAL RECESSIVE (not AD) — carrier parents unaffected; "
            "family cascade testing targets siblings (25% risk biallelic) not parents (usually unaffected); "
            "penetrance in biallelic carriers is ~100% (LOF is complete when both alleles lost). "
            "GENETIC TESTING: Full CASQ2 sequencing; MLPA for exon deletions; "
            "heterozygous CASQ2 carriers are generally NOT at CPVT risk (insufficient LOF for haploinsufficiency). "
            "CARDIAC MRI: no structural abnormality (normal echo/MRI — distinguishes from ARVC, HCM)."
        ),
        "locus": "1p13.3",
        "aa": 399,
        "kDa": 46,
        "omim_gene": "114251",
        "omim_disease": "Catecholaminergic Polymorphic Ventricular Tachycardia type 2 CPVT2 (OMIM 611938)",
        "inheritance": "AR biallelic LOF; heterozygous carriers generally unaffected; consanguinity common",
        "gene_class": "Calsequestrin 2 — junctional SR Ca2+ buffer — RYR2 regulatory scaffold — LOF: reduced SR buffering + constitutive RYR2 activation → CPVT2",
        "key_alerts": [
            "CASQ2-AUTOSOMAL-RECESSIVE-NOT-AD: CPVT2 is RECESSIVE — parents are carriers but unaffected; siblings have 25% risk biallelic; cascade testing targets siblings/extended family not parents; contrasts with CPVT1 (AD RYR2)",
            "CASQ2-TREATMENT-IDENTICAL-CPVT1: Nadolol + exercise restriction + flecainide adjunct = same protocol as RYR2-CPVT1; metoprolol still inferior; exercise prohibition still mandatory; ICD for symptomatic/refractory cases",
            "CASQ2-PEDIATRIC-YOUNGER-ONSET: Mean onset 3–8 years (younger than CPVT1 mean 12 years); bidirectional VT in a child with normal QTc/normal heart = CASQ2 or RYR2 — epinephrine provocation + full CPVT gene panel",
            "CASQ2-D307H-BEDOUIN-FOUNDER: p.Asp307His is a founder mutation in Bedouin and certain Middle Eastern families; consanguineous CPVT → test CASQ2 D307H first before full sequencing in at-risk ethnic groups",
            "CASQ2-HETEROZYGOUS-NOT-AT-RISK: Single heterozygous CASQ2 variant does NOT cause CPVT2 (recessive — haploinsufficiency not sufficient); do NOT overinterpret carrier status; require BOTH alleles LOF for CPVT2 phenotype",
        ],
        "etiologies": [
            "Biallelic CASQ2 LOF → absent SR Ca2+ buffer → higher free [Ca2+]SR → lower threshold for spontaneous RYR2 opening → diastolic Ca2+ leak → DADs → CPVT",
            "Loss of CASQ2-triadin inhibitory signal on RYR2 → constitutively increased RYR2 open probability → Ca2+ sparks at rest amplified by adrenergic stimulation",
            "Misfolded CASQ2 (missense) → ER retention → reduced SR lumen CASQ2 protein → functional haploinsufficiency biallelic",
            "Adrenergic activation (PKA-CaMKII) + CASQ2 LOF → SR Ca2+ overload + unrestrained RYR2 → bidirectional VT threshold reached at low heart rates",
        ],
        "stats": {
            "mean_dx_age": 8,
            "mean_dx_delay_months": 24,
            "cpvt_prevalence_pct": 5,
            "inheritance_ar_pct": 100,
            "consanguinity_pct": 40,
            "d307h_bedouin_founder": True,
        },
        "dx_delay_distribution": "18–36 months (recessive inheritance means family history negative; CPVT not suspected in young child with syncope and normal ECG/echo; CASQ2 often not included in first-pass arrhythmia panels)",
    },

    # ── KCNJ2 — Andersen-Tawil Syndrome / LQT7 ─────────────────────────────
    {
        "gene": "KCNJ2",
        "protein": "KCNJ2 — Andersen-Tawil-Syndrome LQT7 AD — Kir2.1 IK1-Channel — 427aa — 17q24.3 — Triad-Periodic-Paralysis+VT+Dysmorphia-PATHOGNOMONIC — Quinidine-Flecainide — Acetazolamide-Paralysis",
        "alias": (
            "KCNJ2 (Kir2.1, IRK1); OMIM gene 600681; Andersen-Tawil Syndrome type 1 (ATS1, LQT7) OMIM 170390. "
            "17q24.3; 427 aa; ~48 kDa; AD dominant-negative or LOF missense. "
            "FUNCTION: KCNJ2 encodes the inwardly rectifying K+ channel Kir2.1. "
            "IK1 (Kir2.x channels including Kir2.1) is critical for: "
            "(1) RESTING MEMBRANE POTENTIAL maintenance (~-90 mV in cardiomyocytes — Kir2.1 inward rectification keeps cells polarised at rest); "
            "(2) Phase 4 resting potential stability; "
            "(3) Phase 3 terminal repolarisation — IK1 activates during late repolarisation to bring Vm back to resting potential. "
            "In skeletal muscle: IK1 maintains resting potential → LOF → periodic paralysis. "
            "ATS PATHOPHYSIOLOGY: KCNJ2 LOF (dominant-negative — mutant subunits poison heterotetramer) → "
            "reduced IK1 → elevated resting Vm → reduced resting potential stability → "
            "arrhythmia in heart + periodic paralysis in skeletal muscle. "
            "ANDERSEN-TAWIL SYNDROME TRIAD (PATHOGNOMONIC — all three required for ATS): "
            "(1) PERIODIC PARALYSIS: episodic flaccid muscle weakness/paralysis — "
            "hypokalemic (most common), hyperkalemic, or normokalemic; "
            "triggered by rest after exercise, fasting, carbohydrate meal; "
            "(2) VENTRICULAR ARRHYTHMIAS: frequent PVCs/couplets; bidirectional VT (like CPVT); "
            "non-sustained VT; QTc often minimally prolonged (QU prolongation more prominent); "
            "(3) DISTINCTIVE FACIAL/SKELETAL DYSMORPHIA: low-set ears, hypertelorism, mandibular hypoplasia, "
            "broad forehead, clinodactyly, brachydactyly, 2nd/3rd toe syndactyly, short stature — "
            "subtle but PATHOGNOMONIC when complete triad present. "
            "BIDIRECTIONAL VT IN ATS: Morphologically identical to CPVT-BiVT but different mechanism: "
            "IK1 reduction (not RYR2 Ca2+ leak) → spontaneous phase 4 depolarisation → fascicular-triggered BiVT. "
            "ATS-RISK: Lower SCD risk than CPVT1/2 despite frequent VT — most BiVT in ATS non-sustained; "
            "SCD reported but rare compared to CPVT. "
            "TREATMENT — VENTRICULAR ARRHYTHMIAS: "
            "Quinidine: Class IA — reduces VT burden in ATS (IK1 reduction makes cells more quinidine-sensitive — paradoxically useful); "
            "Flecainide: adjunct; reduces PVC/BiVT burden; "
            "Beta-blockers: less effective than in CPVT (ATS VT not primarily adrenergic-triggered); "
            "ICD: for symptomatic sustained VT/VF; "
            "Note: Na-channel blocker therapy rationale in ATS is complex — must monitor carefully. "
            "TREATMENT — PERIODIC PARALYSIS: "
            "Acetazolamide (carbonic anhydrase inhibitor) — reduces paralysis frequency; "
            "Avoid: carbohydrate meals pre-exercise, cold exposure, rest after intense exercise; "
            "K+ supplementation for hypokalemic episodes. "
            "GENETICS: 66% of clinically defined ATS have KCNJ2 mutation; remainder KCNJ2-negative; "
            "dominant-negative mutations most common (C-terminus, selectivity filter); "
            "KCNJ2 R312C, G300V, D71V — common ATS mutations; penetrance variable even within family. "
            "SHORT QT SYNDROME type 3 (GOF KCNJ2 mutations) — distinct rare entity — increased IK1 → QTc <330 ms."
        ),
        "locus": "17q24.3",
        "aa": 427,
        "kDa": 48,
        "omim_gene": "600681",
        "omim_disease": "Andersen-Tawil Syndrome type 1 ATS1 / Long QT Syndrome type 7 LQT7 (OMIM 170390); Short QT Syndrome type 3 GOF (OMIM 609622)",
        "inheritance": "AD dominant-negative LOF; variable penetrance and expressivity — triad may be incomplete in mild carriers",
        "gene_class": "Kir2.1 IK1 inwardly rectifying K+ channel — resting membrane potential + phase 3 terminal repolarisation + skeletal muscle polarisation — LOF: ATS triad",
        "key_alerts": [
            "KCNJ2-ATS-TRIAD-PATHOGNOMONIC: Andersen-Tawil triad = periodic paralysis + ventricular arrhythmias + facial dysmorphia — THREE components = ATS1 until KCNJ2 testing negative; dysmorphia subtle but examine ears/jaw/digits in EVERY periodic paralysis/VT patient",
            "KCNJ2-BIDIRECTIONAL-VT-NOT-CPVT: ATS produces bidirectional VT (IK1-driven, phase 4 triggered) morphologically identical to CPVT — NORMAL RESTING QTc (or minimal prolongation) + normal RYR2/CASQ2 → test KCNJ2; ATS VT less lethal than CPVT1 BiVT",
            "KCNJ2-QUINIDINE-ATS-SPECIFIC: Quinidine reduces VT burden in ATS (complex paradoxical IK1-QIA relationship); beta-blockers less effective (VT not primarily adrenergic); distinguish from LQT1 where beta-blockers are first-line",
            "KCNJ2-ACETAZOLAMIDE-PARALYSIS: Acetazolamide reduces periodic paralysis frequency in ATS; avoid paralysis triggers (carb meals, rest after exercise, cold); supplement K+ for hypokalemic episodes; neurology co-management required",
            "KCNJ2-INCOMPLETE-TRIAD-COMMON: 66% of clinical ATS have KCNJ2 mutations; penetrance variable — triad incomplete in 20–30% of carriers; do not require all 3 components to test KCNJ2 if 2 of 3 present; family cascade testing reveals incomplete expressivity",
        ],
        "etiologies": [
            "KCNJ2 LOF dominant-negative → reduced IK1 density → elevated resting Vm → reduced resting potential stability → spontaneous phase 4 depolarisation → fascicular-triggered VT / BiVT",
            "Reduced IK1 in skeletal muscle → impaired resting Vm after exercise → Na+ channel recovery failure → transient Na+ influx → muscle depolarisation lock → periodic paralysis",
            "KCNJ2 dominant-negative tetramer poisoning → ≥1 mutant Kir2.1 subunit in tetrameric channel → dominant-negative loss of IK1 even at 50% heterozygous expression",
            "Hypokalemia → further IK1 reduction (IK1 amplitude K+-dependent) → compounds KCNJ2 LOF → paralysis trigger via lowered [K+]o",
        ],
        "stats": {
            "mean_dx_age": 14,
            "mean_dx_delay_months": 36,
            "triad_complete_pct": 75,
            "kcnj2_positive_in_clinical_ats_pct": 66,
            "scd_risk_vs_cpvt": "lower (most BiVT non-sustained)",
            "acetazolamide_paralysis_response_pct": 70,
        },
        "dx_delay_distribution": "24–60 months (triad misdiagnosed as 3 separate conditions; periodic paralysis referred to neurology, VT to cardiology, dysmorphia to genetics — integration of triad delayed)",
    },

    # ── HCN4 — Sick Sinus Syndrome type 2 ────────────────────────────────────
    {
        "gene": "HCN4",
        "protein": "HCN4 — Sick-Sinus-Syndrome2 AD-LOF — If-Current Pacemaker-Channel — 1203aa — 15q24.1 — Bradycardia-Pacemaker — LV-Noncompaction-Overlap — Ivabradine-CONTRAINDICATED-SSS",
        "alias": (
            "HCN4 (Hyperpolarisation-activated Cyclic Nucleotide-gated channel 4); OMIM gene 605206; "
            "Sick Sinus Syndrome type 2 (SSS2) OMIM 163800; Brugada-like with HCN4. "
            "15q24.1; 1203 aa; ~134 kDa; AD LOF. "
            "FUNCTION: HCN4 is the primary pacemaker channel of the sinoatrial (SA) node. "
            "If (funny current) is a mixed Na+/K+ current that activates on HYPERPOLARISATION "
            "(opposite to most voltage-gated channels — activated when cell repolarises after action potential). "
            "Mechanism of AUTOMATICITY via If: "
            "SA node action potential repolarises → -60 to -70 mV → HCN4 channels open → "
            "Na+ influx via If → slow phase 4 depolarisation (pacemaker potential) → threshold → "
            "new action potential → repeating cardiac cycle. "
            "cAMP REGULATION OF If: β-adrenergic → cAMP → binds HCN4 C-terminus CNBD → "
            "increases If (faster activation) → increased heart rate (chronotropy); "
            "vagal → reduced cAMP → reduced If → slower heart rate. "
            "HCN4 MUTATIONS — LOSS OF FUNCTION: "
            "LOF reduces If → slower phase 4 depolarisation → "
            "(1) Sinus bradycardia at rest (HR 30–50 bpm); "
            "(2) Chronotropic incompetence (inability to raise HR adequately with exercise); "
            "(3) Sinus pauses / sinus arrest; "
            "(4) AF — bradycardia-triggered atrial fibrillation (escape AF); "
            "(5) AV conduction disease (HCN4 in AVN). "
            "LV NON-COMPACTION (LVNC) ASSOCIATION: "
            "Subset of HCN4 mutations (particularly R550W and others affecting channel trafficking) → "
            "LVNC (failure of myocardial compaction during embryogenesis) + SSS; "
            "cardiac MRI mandatory to exclude LVNC in HCN4 LOF carriers (trabeculation/compaction ratio); "
            "LVNC + SSS + bradycardia = HCN4 mutation until proven otherwise. "
            "IVABRADINE CONTRAINDICATION IN HCN4-SSS: "
            "Ivabradine is an HCN4 (If) channel blocker — used for inappropriate sinus tachycardia / HFrEF; "
            "ADMINISTERING IVABRADINE TO HCN4-LOF PATIENTS = further If reduction → severe bradycardia/arrest; "
            "ABSOLUTE CONTRAINDICATION: do not use ivabradine in HCN4-SSS; "
            "distinguishes HCN4-SSS from IST (inappropriate sinus tachycardia where ivabradine IS indicated). "
            "TREATMENT: "
            "Pacemaker implantation — definitive treatment for symptomatic SSS; "
            "dual-chamber pacemaker (DDD) preserves AV synchrony; "
            "avoid negative chronotropes (beta-blockers, Ca-channel blockers) unless absolutely necessary; "
            "THEOPHYLLINE (weak — not standard) or isoproterenol for acute sinus arrest; "
            "GENETIC TESTING: full HCN4 sequencing + MLPA for deletions; "
            "cardiac MRI for LVNC in all HCN4 LOF carriers. "
            "BRUGADA-LIKE PHENOTYPE WITH HCN4: "
            "Rare HCN4 variants produce Brugada ECG pattern + SSS (If-independent sodium channel interaction); "
            "differentiate from SCN5A-Brugada by genetic testing and Na-channel blocker provocation."
        ),
        "locus": "15q24.1",
        "aa": 1203,
        "kDa": 134,
        "omim_gene": "605206",
        "omim_disease": "Sick Sinus Syndrome type 2 SSS2 (OMIM 163800); Left Ventricular Non-Compaction with SSS (OMIM 613120); Brugada-like syndrome with HCN4",
        "inheritance": "AD LOF; variable penetrance — some heterozygous carriers asymptomatic; penetrance increases with age",
        "gene_class": "HCN4 If hyperpolarisation-activated pacemaker channel — SA/AV node automaticity — cAMP-regulated chronotropy — LOF: SSS/bradycardia/LVNC",
        "key_alerts": [
            "HCN4-IVABRADINE-ABSOLUTE-CI: Ivabradine directly blocks HCN4/If channel — ABSOLUTELY CONTRAINDICATED in HCN4-SSS (further If reduction → severe bradycardia/arrest); ivabradine is only for IST/HFrEF with normal sinus function; document HCN4 LOF in drug allergy system",
            "HCN4-LVNC-CARDIAC-MRI-MANDATORY: HCN4 LOF subset causes left ventricular non-compaction (LVNC) + SSS — cardiac MRI compulsory in ALL HCN4 mutation carriers to exclude LVNC (trabeculation/compaction ratio >2.3); LVNC adds thromboembolic and VT risk",
            "HCN4-PACEMAKER-DEFINITIVE: Symptomatic SSS (syncope, presyncope, chronotropic incompetence limiting exercise) → pacemaker implantation is definitive treatment; dual-chamber DDD to preserve AV synchrony; avoid negative chronotropes",
            "HCN4-CHRONOTROPIC-INCOMPETENCE: HCN4-SSS patients cannot raise HR adequately with exercise (If-dependent accelerated phase 4) → exercise intolerance + reduced cardiopulmonary fitness; exercise ECG demonstrates flat HR response; distinguish from deconditioning",
            "HCN4-AF-BRADYCARDIA-TRIGGERED: Sinus arrest/bradycardia in HCN4-SSS → escape AF common (bradycardia-triggered AF); anticoagulation required if AF documented; rate control agents (beta-blockers, CCBs) worsen bradycardia — rhythm control preferred; pacemaker first",
        ],
        "etiologies": [
            "HCN4 LOF → reduced If amplitude → slower SA node phase 4 depolarisation → sinus bradycardia / chronotropic incompetence / sinus arrest",
            "HCN4 LOF → impaired cAMP-mediated If upregulation → inadequate heart rate increase on exercise → chronotropic incompetence despite adrenergic stimulation",
            "HCN4 LOF + embryonic LVNC: HCN4 expressed in embryonic myocardium regulates Ca2+ signalling during compaction; LOF → trabecular non-compaction → LVNC",
            "SSS-triggered bradycardia → atrial vulnerability window → AF initiation (bradycardia-dependent, not AF-substrate-dependent); escape pacemaker rhythm in AF",
        ],
        "stats": {
            "mean_dx_age": 40,
            "mean_dx_delay_months": 24,
            "lvnc_association_pct": 30,
            "af_risk_pct": 35,
            "pacemaker_implant_by_age_60_pct": 60,
            "ivabradine_ci": True,
        },
        "dx_delay_distribution": "18–36 months (bradycardia attributed to athletic training; LVNC found incidentally on echocardiogram without SSS correlation; HCN4 not included in routine arrhythmia panels until SSS + LVNC constellation triggers WES)",
    },
]


def _make_patients(gene_info: dict, seed: int):
    rng = random.Random(seed)
    pts = []
    mean_age = gene_info["stats"]["mean_dx_age"]
    mean_delay = gene_info["stats"]["mean_dx_delay_months"]
    gene = gene_info["gene"]
    for i in range(40):
        age = max(1, min(75, int(rng.gauss(mean_age, max(2, mean_age * 0.35)))))
        delay = max(1, int(rng.gauss(mean_delay, mean_delay * 0.4)))
        sex = rng.choice(["M", "F"])
        pts.append({
            "patient_id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "age_at_dx": age,
            "sex": sex,
            "dx_delay_months": delay,
            "seed": seed,
        })
    return pts


# pre-generate all patient cohorts
for _idx, _g in enumerate(ARRHYTHMIA_GENES):
    _g["patients"] = _make_patients(_g, SEED_BASE + _idx)


def get_overview():
    genes = []
    all_ages = []
    all_delays = []
    for g in ARRHYTHMIA_GENES:
        ages = [p["age_at_dx"] for p in g["patients"]]
        delays = [p["dx_delay_months"] for p in g["patients"]]
        all_ages.extend(ages)
        all_delays.extend(delays)
        genes.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "omim_gene": g["omim_gene"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "mean_dx_age": round(sum(ages) / len(ages), 1),
            "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
            "key_alerts": g["key_alerts"],
            "n_patients": len(g["patients"]),
        })
    return {
        "atlas": "Hereditary-Cardiac-Arrhythmia-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Cardiac Arrhythmia Atlas — Ion Channel Disease (LQT/Brugada/CPVT/SSS)",
        "seed_range": f"{SEED_BASE}–{SEED_BASE+7}",
        "total_patients": sum(len(g["patients"]) for g in ARRHYTHMIA_GENES),
        "aggregate_stats": {
            "mean_dx_age": round(sum(all_ages) / len(all_ages), 1),
            "mean_dx_delay_months": round(sum(all_delays) / len(all_delays), 1),
            "genes_covered": len(ARRHYTHMIA_GENES),
            "patients_per_gene": 40,
        },
        "genes": genes,
        "top_alerts": [
            "KCNQ1-SWIMMING-TRIGGER: Swimming syncope = LQT1 until proven otherwise — cold water + adrenergic surge; nadolol 97% protection; JLNS biallelic = deafness + extreme QTc + early ICD",
            "SCN5A-FEVER-ABSOLUTE-EMERGENCY: Fever in Brugada = medical emergency — ICU admission + aggressive antipyretics; Na-channel blockers (flecainide) ABSOLUTE CI as TREATMENT in Brugada (diagnostic use only under monitoring)",
            "RYR2-EXERCISE-RESTRICTION-MANDATORY: CPVT1 exercise restriction UNCONDITIONAL — no competitive sports; bidirectional VT at exercise = CPVT; metoprolol INFERIOR (cardioselective) — use nadolol; flecainide blocks RYR2 directly (Watanabe 2009)",
            "CALM1-NEONATAL-LETHAL: Calmodulinopathy presents in utero/neonatal — extreme QTc 600–650 ms; WES/WGS mandatory; flecainide dual RYR2+IKr benefit; rarest but most severe arrhythmia syndrome",
            "HCN4-IVABRADINE-ABSOLUTE-CI: Ivabradine blocks HCN4/If — ABSOLUTE CI in HCN4-SSS; cardiac MRI for LVNC in all HCN4 LOF; pacemaker is definitive treatment for symptomatic SSS",
            "KCNJ2-ATS-TRIAD-PATHOGNOMONIC: Andersen-Tawil triad = periodic paralysis + BiVT + facial dysmorphia; quinidine for VT; acetazolamide for paralysis; DO NOT treat as standard CPVT (different mechanism, lower lethality)",
            "KCNH2-DRUG-TRIGGER-200-DRUGS: >200 drugs block hERG — crediblemeds.org check mandatory before ANY new prescription in LQT2; hypokalemia synergistic risk; auditory alarm modification required",
            "CASQ2-RECESSIVE-CPVT2: CASQ2-CPVT2 is autosomal recessive — carrier parents unaffected; siblings at 25% risk biallelic; same treatment protocol as RYR2-CPVT1 (nadolol + flecainide + exercise restriction)",
        ],
    }


def get_breakdown():
    result = []
    for idx, g in enumerate(ARRHYTHMIA_GENES):
        ages = [p["age_at_dx"] for p in g["patients"]]
        delays = [p["dx_delay_months"] for p in g["patients"]]
        result.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "alias": g["alias"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "omim_gene": g["omim_gene"],
            "omim_disease": g["omim_disease"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "key_alerts": g["key_alerts"],
            "etiologies": g["etiologies"],
            "stats": g["stats"],
            "dx_delay_distribution": g["dx_delay_distribution"],
            "computed": {
                "mean_dx_age": round(sum(ages) / len(ages), 1),
                "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
                "n_patients": len(g["patients"]),
                "seed": SEED_BASE + idx,
            },
            "sample_patients": g["patients"][:10],
        })
    return result


def get_definitions():
    return {
        "concepts": {
            "QTc Prolongation — Mechanism, Measurement, and the Repolarisation Reserve Hypothesis": (
                "QTc prolongation reflects delayed ventricular repolarisation and predisposition to early afterdepolarisations (EADs) and TdP. "
                "MEASUREMENT: QT corrected for heart rate (Bazett: QTc = QT/√RR; Fridericia: QTc = QT/RR^(1/3)); "
                "Bazett overcorrects at high HR; Fridericia preferred at exercise. "
                "NORMAL: QTc <440 ms (male), <460 ms (female); "
                "BORDERLINE: 440–480 ms (male), 460–490 ms (female); "
                "PROLONGED: >480 ms; high risk: >500 ms. "
                "REPOLARISATION RESERVE HYPOTHESIS (Roden 1998): "
                "Cardiac repolarisation has multiple overlapping currents (IKr/IKs/IK1/Ito/ICa-L) providing reserve. "
                "In congenital LQTS: one current already reduced (LOF KCNQ1/KCNH2/SCN5A); "
                "additional insult (drug, hypokalaemia, bradycardia, fever) overwhelms remaining reserve → TdP. "
                "TORSADE DE POINTES (TdP): Polymorphic VT with twisting QRS around isoelectric line; "
                "initiated by EAD (from prolonged plateau → L-type Ca2+ window current reactivation) → triggered beat at long-short RR; "
                "often self-terminates but may degenerate to VF; pause-dependence is hallmark. "
                "CHANNEL-SPECIFIC QTc PATTERNS: "
                "LQT1 (KCNQ1): broad-based T wave, symmetric; "
                "LQT2 (KCNH2): low-amplitude, notched or bifid T wave; prominent U wave; "
                "LQT3 (SCN5A): late-onset T wave, long flat ST segment before T peak; "
                "LQT7 (KCNJ2): minimal QTc prolongation; QU prolongation more prominent; BiVT. "
                "SCHWARTZ SCORE: clinical score (QTc + symptoms + family history + ECG features) for LQTS diagnosis before genetic confirmation; "
                "score ≤1: low probability; 1.5–3: intermediate; ≥3.5: high probability — genetic testing in all intermediate/high."
            ),
            "Brugada Syndrome — Type 1 ECG, Sodium Channel Pharmacology, and Risk Stratification": (
                "Brugada Syndrome (BrS) is caused by reduced right ventricular outflow tract (RVOT) Na+ channel function → "
                "epicardial Ito-mediated phase 2 reentry → VF. "
                "TYPE 1 ECG (DIAGNOSTIC): Coved ST elevation ≥2 mm in ≥1 right precordial lead (V1, V2, V1–V2 positioned higher — 2nd/3rd ICS) "
                "with negative T wave. Spontaneous Type 1 is diagnostic; "
                "induced Type 1 (by ajmaline/flecainide/pilsicainide) requires documented conversion. "
                "TYPE 2 (SADDLE-BACK): ≥2 mm J-point elevation with saddle-back morphology — non-diagnostic; "
                "provocable to Type 1. "
                "SODIUM CHANNEL BLOCKER PROVOCATION — DIAGNOSTIC USE ONLY: "
                "Ajmaline (0.5–1 mg/kg IV over 10 min) — most sensitive/specific; used in Europe; "
                "Flecainide (2 mg/kg IV or 400 mg PO) — used in Asia; "
                "MUST be performed under continuous ECG monitoring with defibrillator available; "
                "stop at Type 1 conversion or PVC/VT. "
                "NA-CHANNEL BLOCKERS AS TREATMENT — ABSOLUTE CONTRAINDICATION: "
                "Flecainide, pilsicainide, propafenone, ajmaline, quinidine Class IA doses → WORSEN BrS → "
                "increase VF risk; do NOT use as antiarrhythmic treatment in BrS. "
                "FEVER MANAGEMENT IN BrS: "
                "Nav1.5 gating is temperature-sensitive (kinetics shift with temperature); "
                "fever → reduced peak INa + increased Ito → RVOT epicardial phase 2 reentry worsens; "
                "administer paracetamol/ibuprofen immediately for any fever in BrS patients; "
                "avoidance of high ambient temperature (saunas, hot baths); "
                "wearable defibrillator vest during febrile illness in unimplanted BrS. "
                "RISK STRATIFICATION: "
                "Symptomatic (CA survivor, unexplained syncope with Type 1 ECG) → ICD Class I; "
                "asymptomatic with spontaneous Type 1 → ICD debated (Brugada-Risk score); "
                "asymptomatic drug-induced Type 1 → low-risk; clinical/family history + EP study. "
                "QUINIDINE FOR BrS: blocks Ito (transient outward K+ current) → reduces phase 2 notch → "
                "restores epicardial dome → reduces VF; used for electrical storms and asymptomatic BrS with frequent Type 1 or high-risk features."
            ),
            "CPVT — Exercise Provocation, Bidirectional VT Mechanism, and Flecainide-RYR2 Pharmacology": (
                "Catecholaminergic Polymorphic VT (CPVT) is the most lethal inherited arrhythmia in children (1–3%/yr mortality untreated). "
                "PATHOPHYSIOLOGY: β-adrenergic stimulation + defective RYR2 (CPVT1) or CASQ2 (CPVT2) → "
                "diastolic SR Ca2+ leak → NCX-INCX (inward Na+/Ca2+ exchanger current) → DADs → triggered VT at NORMAL QTc. "
                "BIDIRECTIONAL VT MORPHOLOGY: QRS alternates axis beat-to-beat (~180°) — "
                "arises from alternating left and right bundle branch / fascicular foci activated by DADs; "
                "PATHOGNOMONIC for CPVT (and ATS/KCNJ2, different mechanism). "
                "EXERCISE TREADMILL TEST FOR DIAGNOSIS: "
                "Standard Bruce protocol: VT typically appears at 110–150 bpm heart rate threshold; "
                "BiVT appearance is diagnostic of CPVT (without structural heart disease); "
                "frequent PVCs → bigeminy → couplets → sustained BiVT with increasing workload; "
                "resolution with decreased workload (differentiates from ischaemic VT which worsens). "
                "FLECAINIDE RYR2 PHARMACOLOGY (Watanabe 2009, Nature Medicine): "
                "Flecainide (Class IC Na-channel blocker) blocks RYR2 in its open-channel configuration: "
                "binds to a site within the RYR2 channel pore → reduces open probability of RYR2 → "
                "less diastolic SR Ca2+ leak → fewer DADs → fewer triggered beats; "
                "this is INDEPENDENT of its INa-blocking action on sarcolemmal Nav1.5; "
                "flecainide + nadolol: complementary mechanisms → superior CPVT suppression to either alone; "
                "flecainide dose in CPVT: 100–150 mg BID (lower than antiarrhythmic use in AF); "
                "MONITOR: PR/QRS prolongation on ECG (INa block); structural disease must be excluded "
                "(flecainide proarrhythmic in ischaemic/structural heart disease — CAST trial). "
                "LCSD (LEFT CARDIAC SYMPATHETIC DENERVATION): "
                "Video-assisted thoracoscopic (VATS) removal of lower half of stellate ganglion + T2–T4 ganglia → "
                "reduces noradrenaline release to ventricle by 70–80%; "
                "reduces CPVT events significantly (Schwartz 2004, Mayo LCSD series); "
                "adjunct to pharmacotherapy in refractory CPVT; also used for LQT1/LQT2 refractory. "
                "ICD PROGRAMMING IN CPVT: "
                "Programme VT detection at conservative rate (>200 bpm for shocks if possible); "
                "maximise ATP; minimise unnecessary shocks (shock pain → catecholamine surge → more VT)."
            ),
            "HCN4 and If Current — Pacemaker Channel Biology and Sick Sinus Syndrome Treatment": (
                "If (funny current) is the primary pacemaker current in the sinoatrial (SA) node: "
                "BIOPHYSICS: Mixed Na+/K+ inward current; activates on HYPERPOLARISATION (below -40 to -50 mV); "
                "activation kinetics: slow (200–500 ms time constant at -65 mV) → generates slow phase 4 depolarisation; "
                "reversal potential: -10 to -30 mV (depolarising current relative to resting Vm). "
                "cAMP GATING: β-adrenergic → cAMP → binds HCN4 C-terminus CNBD domain → "
                "right-shifts activation curve (more If at any given Vm) → faster phase 4 → increased HR (positive chronotropy); "
                "vagal → reduced cAMP → left-shifts activation → reduced If → slower HR (negative chronotropy). "
                "HCN4 IN AV NODE: Contributes to AV nodal automaticity and conduction velocity; "
                "HCN4 LOF → PR interval prolongation in addition to sinus bradycardia. "
                "HCN4 IN EMBRYOGENESIS: Expressed in embryonic myocardium regulating Ca2+ handling and compaction; "
                "HCN4 null mice: embryonic lethality from cardiac compaction defects → basis for LVNC association in humans. "
                "SICK SINUS SYNDROME — CLINICAL FEATURES: "
                "Sinus bradycardia (HR <50 at rest); sinus pauses (>3 sec diagnostically significant); "
                "tachy-brady syndrome (alternating tachycardia/bradycardia); "
                "chronotropic incompetence (failure to reach 80% MPHR on ETT); "
                "syncope (sinus arrest or brady-induced AF with rapid ventricular response on termination). "
                "PACEMAKER THERAPY: "
                "DDD pacing (dual-chamber) preferred — preserves physiological AV synchrony and rate response; "
                "DDDR (rate-responsive) for chronotropic incompetence during exercise; "
                "minimise right ventricular pacing (pacing-induced CMP risk) — programming minimisation algorithms. "
                "IVABRADINE MECHANISM AND CI IN HCN4-SSS: "
                "Ivabradine is a selective HCN channel blocker (pure If blocker, no other cardiac effects); "
                "FDA/EMA approved for: (1) Heart failure with HFrEF + HR >70 bpm on max beta-blockers; "
                "(2) Inappropriate sinus tachycardia (IST); "
                "IN HCN4-SSS: further reducing already deficient If → sinus arrest → NEVER use; "
                "prescribers must check HCN4 testing before ivabradine in any patient with sinus abnormality."
            ),
        },
        "pharmacological_distinctions": [
            "Nadolol vs metoprolol in CPVT: Non-selective beta-blockers (nadolol, propranolol) block both β1 AND β2 adrenoceptors → full adrenergic RYR2 suppression. Metoprolol is cardioselective (β1 only) → β2-adrenergic escape → residual catecholaminergic stimulation → CPVT VT events on metoprolol that would be suppressed on nadolol; nadolol ALWAYS preferred for CPVT",
            "Mexiletine (LQT3) vs flecainide (CPVT/Calmodulinopathy): Mexiletine blocks persistent late INa (SCN5A GOF) → QTc shortening specific to LQT3; does NOT block RYR2 or augment IKr significantly. Flecainide blocks both Nav1.5 peak INa AND RYR2 (open-channel) AND augments IKr in calmodulinopathy → broader channelopathy utility; flecainide CONTRAINDICATED in Brugada/structural heart disease (CAST); mexiletine safe in Brugada context",
            "Quinidine in Brugada vs quinidine in ATS: In Brugada — quinidine blocks Ito (transient outward K+ current, Kv4.3) → restores epicardial dome → reduces phase 2 reentry → anti-VF; in ATS (KCNJ2) — complex interactions with IK1 / fascicular reentry → reduces BiVT burden by different mechanism; same drug, different channel targets, different arrhythmia mechanisms",
            "Ivabradine (If blocker) in IST vs CI in HCN4-SSS: Ivabradine is the therapeutic agent for REDUCING excessive sinus rate in IST/HFrEF (FDA-approved); CONTRAINDICATED in HCN4-LOF SSS — further If reduction → sinus arrest; prescribers must distinguish increased If (IST) from decreased If (HCN4-SSS) before prescribing ivabradine",
            "Flecainide CI in Brugada vs therapeutic in CPVT: Flecainide ABSOLUTELY CONTRAINDICATED as antiarrhythmic treatment in Brugada (further reduces peak INa → worsens Ito-INa imbalance → VF); conversely flecainide BENEFICIAL in CPVT (RYR2 open-channel block reduces diastolic Ca2+ leak); same drug, opposite outcomes in two channelopathies — MUST confirm diagnosis before prescribing",
        ],
        "key_standards": [
            "HRS/EHRA/APHRS/SOLAECE 2013 Expert Consensus — Diagnosis and Treatment of Inherited Primary Arrhythmia Syndromes: LQTS, CPVT, Brugada, SQTS, PCCD, ERS — classification, diagnostic criteria, ICD indications, pharmacotherapy recommendations",
            "ESC 2022 Guidelines on Ventricular Arrhythmias and Prevention of Sudden Cardiac Death (Zeppenfeld 2022, EHJ): updated CPVT/LQTS/Brugada evidence; Class I nadolol for CPVT; Class IIa flecainide adjunct for CPVT; LCSD recommendations",
            "AHA/ACC/HRS 2018 Guideline for Evaluation and Management of Syncope (Shen 2018, JACC): exercise treadmill test for CPVT workup; channelopathy workup in unexplained syncope young patients; pacemaker criteria in SSS",
            "Crediblemeds.org (Arizona CERT) QT drug risk categories: CONDX (conditional risk), Known/Possible/Conditional risk categories — mandatory prescriber reference for all LQT2/LQTS patients before any new medication",
            "Watanabe 2009 (Nature Medicine): Flecainide prevents catecholaminergic polymorphic ventricular tachycardia in mice and humans — mechanistic evidence for RYR2 open-channel block by flecainide; basis for clinical flecainide use in CPVT",
        ],
    }
