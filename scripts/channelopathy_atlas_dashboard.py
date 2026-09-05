#!/usr/bin/env python3
"""Channelopathy-Atlas — Complete 8-Gene Skeletal Muscle + Cardiac Channelopathy Atlas
SCN4A   (Nav1.4; 1836 aa; 17q23.3; HyperKPP/PMC/PAM/CMS-SCN4A; GOF/LOF; Mexiletine first-line myotonia) ·
CACNA1S (Cav1.1; 1873 aa; 1q32.1; HypoKPP type 1; MH susceptibility; Acetazolamide) ·
KCNJ2   (Kir2.1; 427 aa; 17q24.3; Andersen-Tawil ATS1/LQT7; bidirectional VT PATHOGNOMONIC; Flecainide) ·
CLCN1   (ClC-1; 988 aa; 7q34; Myotonia Congenita Thomsen/Becker; Mexiletine; QUININE/QUINIDINE ABSOLUTELY CI) ·
KCNQ1   (Kv7.1; 676 aa; 11p15.5; LQT1 AD + JLNS AR biallelic deafness; Swimming HIGH RISK; Beta-blockers) ·
KCNH2   (Kv11.1/HERG; 1159 aa; 7q36.1; LQT2; Sudden arousal trigger; massive drug-DDI list) ·
SCN5A   (Nav1.5; 2016 aa; 3p22.2; Brugada/LQT3/PCCD/SIDS; FEVER unmasks Brugada; Quinidine; avoid Na blockers) ·
RYR2    (5038 aa; 1q43; CPVT1/ARVC2; exercise-triggered VT; Flecainide + Beta-blockers; ICD adjunct NOT replacement)
320-patient aggregate cohort (8 × 40, seeds 1054–1061)
"""

import random

SEED_BASE = 1054

CHANNELOPATHY_GENES = [
    # ── SCN4A — Nav1.4 ────────────────────────────────────────────────────
    {
        "gene": "SCN4A", "protein": "Voltage-Gated Sodium Channel α1-Subunit (Nav1.4)",
        "alias": "SCN4A; OMIM gene 603967; 17q23.3; 1836 aa; HyperKPP (OMIM #170500), PMC (#168300), PAM, CMS-SCN4A; GOF → myotonia/periodic paralysis; LOF → CMS",
        "aa": "1836 aa", "kDa": "208 kDa",
        "channel_class": (
            "SCN4A encodes Nav1.4, the dominant voltage-gated sodium channel α-subunit in adult skeletal "
            "muscle. Nav1.4 governs the rapid depolarisation phase of the muscle action potential. "
            "GAIN-OF-FUNCTION (GOF) mutations → delayed or incomplete channel inactivation → "
            "persistent sodium influx → membrane hyperexcitability → MYOTONIA (repetitive involuntary "
            "discharges) and/or PERIODIC PARALYSIS (paradoxical depolarisation block). "
            "LOSS-OF-FUNCTION (LOF) mutations → reduced sodium current → neuromuscular junction failure "
            "→ CONGENITAL MYASTHENIC SYNDROME (CMS-SCN4A; also represented in CMS-Atlas). "
            "MEXILETINE MECHANISM: class Ib sodium channel blocker; binds inactivated Nav1.4 → "
            "reduces persistent INa → abolishes myotonic discharges. First-line for SCN4A myotonia. "
            "COLD triggers PMC (Paramyotonia Congenita) — channel inactivation further impaired at "
            "low temperatures → prolonged myotonia + weakness in cold. "
            "ACETAZOLAMIDE: carbonic anhydrase inhibitor → metabolic acidosis → reduces PP attacks "
            "(mechanism: acid-shifts resting membrane potential away from depolarisation block). "
            "KEY CLINICAL SUBTYPES: HyperKPP (attacks with HIGH or NORMAL K+; carb intake and rest "
            "trigger attacks); PMC (Eulenberg disease — cold-triggered myotonia + periodic paralysis); "
            "PAM (Potassium-Aggravated Myotonia — myotonia worsened by K+ ingestion, no PP)."
        ),
        "channel_group": "Voltage-Gated Sodium Channel / Nav1.4 / Skeletal Muscle",
        "channel_type": "HyperKPP / PMC / PAM / CMS-SCN4A (GOF myotonia/PP; LOF CMS)",
        "locus": "17q23.3", "omim_gene": 603967, "omim_disease": 170500,
        "inheritance": (
            "AD (autosomal dominant) for HyperKPP, PMC, PAM. High penetrance. "
            "LOF alleles causing CMS: AR or de novo. "
            "HyperKPP: p.Thr704Met (T704M) and p.Met1592Val (M1592V) most common (~70% HyperKPP). "
            "PMC: p.Arg1448Cys (R1448C/H), p.Thr1313Met — cold-sensitive inactivation defect. "
            "PAM: p.Val445Met, p.Ile1160Val — myotonia without PP; K+ aggravated. "
            "De novo mutations well documented. Family history often positive with variable expression."
        ),
        "phenotype": (
            "HYPERKPP ONSET: neonatal/infancy (floppiness) or childhood/adolescence. "
            "ATTACKS: weakness episodes lasting minutes to hours, triggered by rest after exercise, "
            "fasting, cold, K+-rich foods (bananas, orange juice). Serum K+ normal to HIGH during attack. "
            "MYOTONIA: grip myotonia and lid lag; typically mild in HyperKPP. "
            "PMC: COLD IS DIAGNOSTIC TRIGGER — hands stiffen in cold water; paradoxical myotonia "
            "(worsens with repeated contractions, opposite to warm-up in CLCN1). "
            "PAM: no periodic paralysis; sustained myotonia; K+ ingestion clearly worsens. "
            "LONG-TERM: permanent proximal myopathy develops in some HyperKPP patients (tubular aggregates on biopsy). "
            "EMG: myotonic discharges (wax/wane); cooling exacerbates in PMC. "
            "CK: mildly elevated (200–800 IU/L). "
            "CARDIAC: NOT a primary feature of GOF SCN4A (cardiac Nav1.5 = SCN5A). "
            "NOTE: SCN4A LOF → CMS (separate phenotype; see CMS-Atlas)."
        ),
        "disease": (
            "SCN4A GOF channelopathy spectrum: HyperKPP, PMC, PAM. "
            "TREATMENT: Mexiletine 150–300 mg TDS (first-line myotonia + prevents PP attacks). "
            "Acetazolamide 125–1000 mg/day for PP prevention (especially HyperKPP). "
            "Low-carbohydrate diet; avoid prolonged rest after exercise; warm clothing (PMC). "
            "Thiazide diuretics (hydrochlorothiazide) second-line PP prevention. "
            "AVOID triggers: prolonged rest post-exercise, fasting, cold (PMC), K+-rich diet (PAM)."
        ),
        "treatment_options": [
            "Mexiletine 150–300 mg TDS: FIRST-LINE for myotonia (SCN4A + CLCN1 — same drug, different mechanism); "
            "sodium channel blocker; reduces myotonic discharges; LFT monitoring not needed (unlike some antiepileptics)",
            "Acetazolamide 125–500 mg BD: PP prevention (HyperKPP, PMC); carbonic anhydrase inhibitor; "
            "monitor renal stones (calcium oxalate risk); ensure adequate hydration",
            "Hydrochlorothiazide 25–75 mg/day: second-line PP prevention if acetazolamide not tolerated; "
            "lowers serum K+ → reduces depolarisation block risk",
            "Avoid triggers: rest after exercise (HyperKPP), prolonged fasting, cold exposure (PMC), "
            "K+-rich foods/drinks (PAM), high-carbohydrate meals",
            "Warm clothing/heated environments: mandatory in PMC — cold prevents normal Nav1.4 inactivation",
            "Emergency PP attack: oral/IV glucose + carbohydrate (HyperKPP); inhaled salbutamol (rapid K+ shift)",
            "Genetic counselling: AD; 50% offspring risk; prenatal/preimplantation genetic diagnosis available",
            "Annual neuromuscular review: monitor for fixed proximal myopathy (long-term HyperKPP complication)",
        ],
        "key_ddx": [
            "HypoKPP type 1 (CACNA1S) — LOW K+ during attacks vs HyperKPP (normal-high K+); critical distinction",
            "HypoKPP type 2 (SCN4A LOF, different alleles) — rare; LOF causing KPP vs GOF causing myotonia",
            "CLCN1 Myotonia Congenita — chloride channel; warm-up phenomenon; NO periodic paralysis",
            "Andersen-Tawil (KCNJ2) — PP + arrhythmia + dysmorphic features; BIDIRECTIONAL VT",
            "Thyrotoxic periodic paralysis — exclude thyroid disease in all new-onset PP",
            "Schwartz-Jampel syndrome (HSPG2) — myotonia + skeletal dysplasia; continuous myotonic discharges",
        ],
        "onset_range_y": (0, 30),
        "cardiac_risk": False,
        "arrhythmia_risk": False,
        "myotonia": True,
        "periodic_paralysis": True,
        "mh_risk": False,
        "deafness_risk": False,
        "juvenile_onset": True,
        "bidirectional_vt": False,
        "ck_range": (150, 800),
        "attack_k_trend": "Normal-High",
        "first_line_drug": "Mexiletine",
        "critical_avoid": "Prolonged rest post-exercise; cold (PMC); K+-rich diet (PAM)",
    },
    # ── CACNA1S — Cav1.1 ──────────────────────────────────────────────────
    {
        "gene": "CACNA1S", "protein": "L-Type Voltage-Gated Calcium Channel α1S-Subunit (Cav1.1)",
        "alias": "CACNA1S; OMIM gene 114208; 1q32.1; 1873 aa; HypoKPP type 1 (OMIM #170400); MH susceptibility (MHS5); Cav1.1",
        "aa": "1873 aa", "kDa": "212 kDa",
        "channel_class": (
            "CACNA1S encodes Cav1.1 (L-type calcium channel α1S-subunit), the dominant dihydropyridine "
            "receptor (DHPR) in the skeletal muscle triad. Cav1.1 acts as the voltage sensor for "
            "excitation-contraction (EC) coupling: membrane depolarisation → Cav1.1 conformational change → "
            "mechanical activation of RYR1 → SR calcium release → muscle contraction. "
            "MECHANISM OF DISEASE: HypoKPP type 1 mutations (predominantly R528H, R1239H) impair the "
            "voltage-sensing S4 segment → aberrant gating pore currents (omega currents) → depolarising "
            "inward cation leak at negative resting potentials → paradoxical sustained depolarisation "
            "→ sodium channel inactivation → paralysis during hypokalaemia. "
            "MALIGNANT HYPERTHERMIA (MH): CACNA1S HypoKPP alleles confer MH susceptibility — "
            "volatile anaesthetic agents (halothane, isoflurane, sevoflurane, desflurane) and "
            "succinylcholine → uncontrolled RYR1 Ca2+ release → MH crisis (hyperthermia, rigidity, "
            "acidosis, rhabdomyolysis). MH is life-threatening without dantrolene. "
            "TRIGGERS (HypoKPP): high-carbohydrate meals, alcohol, rest after exercise, "
            "insulin infusion, cold, stress — all lower serum K+."
        ),
        "channel_group": "L-Type Calcium Channel / Cav1.1 / EC Coupling / DHPR",
        "channel_type": "HypoKPP Type 1 (AD) + MH Susceptibility (MHS5)",
        "locus": "1q32.1", "omim_gene": 114208, "omim_disease": 170400,
        "inheritance": (
            "AD (autosomal dominant). High penetrance but variable severity. "
            "Males more severely affected than females (sex-hormone modulation of K+ homeostasis). "
            "p.Arg528His (R528H): most common variant (~60% HypoKPP1); classic HypoKPP phenotype. "
            "p.Arg1239His (R1239H): second most common; similar phenotype. "
            "Both R528H and R1239H = MH-susceptible alleles — MH alert card MANDATORY. "
            "HypoKPP type 1 (~40% all HypoKPP) vs HypoKPP type 2 (SCN4A, ~60%)."
        ),
        "phenotype": (
            "ONSET: childhood to early adulthood (typically 10–20 years). "
            "ATTACKS: episodic flaccid limb weakness (often on waking after high-carb meal previous evening); "
            "SERUM K+ LOW during attack (typically 2.5–3.0 mmol/L; severe attacks <2.0 mmol/L). "
            "DURATION: hours to days (longer than HyperKPP). "
            "TRIGGERS: carbohydrate-rich meals, alcohol, rest after exercise, cold, infection, stress, "
            "insulin, corticosteroids. "
            "MYOTONIA: absent (distinguishes from SCN4A and CLCN1 disorders). "
            "PERMANENT MYOPATHY: proximal weakness between attacks develops in middle age (~30–50% of patients). "
            "MH: volatile anaesthetics/succinylcholine → uncontrolled Ca2+ release → MH crisis. "
            "FREQUENCY: attacks can be daily (severe) to monthly (mild). "
            "CK: mildly elevated interictally; markedly elevated during attack + rhabdomyolysis risk. "
            "CARDIAC: NOT a primary feature (RYR2 is cardiac SR Ca2+ channel)."
        ),
        "disease": (
            "CACNA1S HypoKPP type 1. Diagnosis: gene panel + serum K+ during attack (LOW) + EMG "
            "(myopathic; no myotonic discharges). "
            "ACUTE ATTACK: oral K+ supplementation 40–60 mmol (KCl) immediately; IV KCl if severe "
            "(cardiac monitoring mandatory during IV K+ replacement — risk of hyperkalaemia overshoot). "
            "PREVENTION: acetazolamide 125–1000 mg/day (most effective); low-carbohydrate diet; "
            "avoid alcohol; potassium-sparing diuretics (spironolactone) as adjunct. "
            "MH PRECAUTION: MH alert card; avoid volatile agents + succinylcholine for ALL surgery; "
            "dantrolene available in theatre; use total IV anaesthesia (propofol + remifentanil)."
        ),
        "treatment_options": [
            "ACUTE ATTACK — oral KCl 40–60 mmol stat: first-line for mild-moderate attack; "
            "monitor K+ every 2h; avoid IV unless patient cannot swallow or K+ <2.5 with ECG changes",
            "IV KCl infusion: only for severe attacks (K+ <2.5, cardiac changes, inability to swallow); "
            "MANDATORY cardiac monitoring; max rate 20 mmol/h; reassess K+ q1-2h; "
            "DANGER of hyperkalaemic rebound — do NOT overshoot",
            "Acetazolamide 125–500 mg BD: FIRST-LINE PREVENTION; reduces attack frequency by 70-80%; "
            "mechanism: mild metabolic acidosis + carbonic anhydrase inhibition; "
            "monitor for renal stones; ensure hydration",
            "Dichlorphenamide: carbonic anhydrase inhibitor; alternative to acetazolamide if intolerant",
            "Low-carbohydrate diet: reduces insulin-mediated K+ shift into cells; "
            "avoid high-glycaemic meals especially in evening",
            "Avoid alcohol, fasting, prolonged rest after exercise, cold environments",
            "MH ALERT CARD: MANDATORY for ALL CACNA1S patients; volatile anaesthetics + succinylcholine "
            "ABSOLUTELY AVOIDED — total IV anaesthesia (TIVA); dantrolene available in theatre",
            "Genetic counselling: AD; 50% familial risk; males more severely affected",
        ],
        "key_ddx": [
            "HypoKPP type 2 (SCN4A LOF) — genetically distinct; clinically similar; gene panel required",
            "HyperKPP (SCN4A GOF) — HIGH/normal K+ during attack vs HypoKPP (LOW K+) — CRITICAL distinction",
            "Andersen-Tawil syndrome (KCNJ2) — hypoKPP + arrhythmia + dysmorphic features",
            "Thyrotoxic periodic paralysis — secondary HypoKPP; exclude thyroid disease first",
            "Secondary HypoKPP (aldosteronism, diuretics, GI K+ loss) — exclude with serum/urine electrolytes",
            "Malignant hyperthermia (RYR1) — acute crisis; CACNA1S also MH-susceptible (same pathway)",
        ],
        "onset_range_y": (5, 30),
        "cardiac_risk": False,
        "arrhythmia_risk": False,
        "myotonia": False,
        "periodic_paralysis": True,
        "mh_risk": True,
        "deafness_risk": False,
        "juvenile_onset": True,
        "bidirectional_vt": False,
        "ck_range": (100, 600),
        "attack_k_trend": "Low",
        "first_line_drug": "Acetazolamide + oral KCl (attack)",
        "critical_avoid": "Volatile anaesthetics; succinylcholine; high-carb meals; alcohol",
    },
    # ── KCNJ2 — Kir2.1 / Andersen-Tawil ─────────────────────────────────
    {
        "gene": "KCNJ2", "protein": "Inward Rectifier Potassium Channel Kir2.1",
        "alias": "KCNJ2; OMIM gene 600681; 17q24.3; 427 aa; Andersen-Tawil Syndrome ATS1/LQT7 (OMIM #170390); BIDIRECTIONAL VT PATHOGNOMONIC; triad: PP + arrhythmia + dysmorphic",
        "aa": "427 aa", "kDa": "48 kDa",
        "channel_class": (
            "KCNJ2 encodes Kir2.1 (Inward Rectifier K+ Channel 2.1), the principal IK1 current "
            "channel in cardiac muscle and also expressed in skeletal muscle. IK1 current is critical for: "
            "(1) maintaining the resting membrane potential near E_K (−80 to −90 mV); "
            "(2) phase 3 repolarisation of the cardiac action potential; "
            "(3) preventing spontaneous depolarisation between beats. "
            "MECHANISM OF DISEASE: AD LOF KCNJ2 mutations → reduced IK1 → resting membrane "
            "depolarisation in skeletal muscle → periodic paralysis; "
            "in cardiac muscle → unstable resting potential → spontaneous triggered activity → "
            "VENTRICULAR ARRHYTHMIA. "
            "BIDIRECTIONAL VT: the pathognomonic arrhythmia of Andersen-Tawil syndrome (ATS1/LQT7) — "
            "alternating QRS axis beat-to-beat (180° axis shift). "
            "CRITICAL DISTINCTION: bidirectional VT is also the hallmark of CPVT (RYR2) but "
            "treatment is DIFFERENT — do NOT confuse ATS with CPVT. "
            "DYSMORPHIC FEATURES: TRIAD — (1) periodic paralysis (hypo- or hyperkalaemic); "
            "(2) ventricular arrhythmia (bidirectional VT, prolonged QU interval = LQT7); "
            "(3) dysmorphic features: low-set ears, wide-set eyes (hypertelorism), "
            "micrognathia, clinodactyly, short stature, scoliosis."
        ),
        "channel_group": "Inward Rectifier Potassium Channel / Kir2.1 / IK1 / Cardiac + Skeletal",
        "channel_type": "Andersen-Tawil Syndrome ATS1 / LQT7 (AD LOF — triad: PP + VT + dysmorphic)",
        "locus": "17q24.3", "omim_gene": 600681, "omim_disease": 170390,
        "inheritance": (
            "AD (autosomal dominant LOF). ~60% of ATS has KCNJ2 mutation. "
            "~40% ATS2 — unknown gene. "
            "De novo mutations common (~30–40% of KCNJ2-ATS). "
            "p.Arg218Trp, p.Arg67Trp, p.Gly300Val: established pathogenic variants. "
            "Haploinsufficiency + dominant-negative mechanisms. "
            "Highly variable expressivity within families (triad may be incomplete — arrhythmia alone, "
            "or PP alone — without all three features)."
        ),
        "phenotype": (
            "ONSET: childhood to early adulthood (typically 2–18 years for PP attacks). "
            "PERIODIC PARALYSIS: hypo- or hyperkalaemic (serum K+ variable — distinguishes from HypoKPP1/2). "
            "Duration: hours; triggers: exercise, rest, carbohydrate, stress. "
            "ARRHYTHMIA: bidirectional ventricular tachycardia (BidVT) — pathognomonic; "
            "prolonged QU/QT interval (LQT7); ventricular bigeminy; polymorphic VT. "
            "Arrhythmia risk: significant but SCD LESS COMMON than LQT1/LQT2 (better prognosis than CPVT). "
            "DYSMORPHIC FEATURES: low-set ears, micrognathia, clinodactyly, hypertelorism, short stature — "
            "present in ~80% of KCNJ2-ATS. "
            "KEY: bidirectional VT + dysmorphic features + PP = ATS; "
            "bidirectional VT WITHOUT dysmorphic features + exercise trigger = think CPVT (RYR2). "
            "CK: normal to mildly elevated."
        ),
        "disease": (
            "Andersen-Tawil Syndrome ATS1. Diagnosis: gene panel + ECG (BidVT, prolonged QU, "
            "U-wave prominence) + clinical triad. "
            "TREATMENT: Flecainide reduces VT burden in ATS1 (different from CPVT where it is also used). "
            "Beta-blockers: some benefit; less effective than in LQT1/LQT2. "
            "AVOID QT-prolonging drugs (crediblemeds.org; ATS = LQT7 — additive QT risk). "
            "Acetazolamide: reduces PP frequency in some. "
            "ICD: consider in severe arrhythmia; but SCD risk lower than CPVT/LQT2."
        ),
        "treatment_options": [
            "Flecainide 100–200 mg BD: reduces bidirectional VT burden in ATS1; "
            "antiarrhythmic class Ic; also used in CPVT but mechanism different in ATS",
            "Beta-blockers (propranolol, nadolol): modest arrhythmia suppression; "
            "less effective in ATS than LQT1; do NOT omit flecainide if VT burden high",
            "Avoid QT-prolonging drugs: check crediblemeds.org for ALL medications; "
            "ATS = LQT7 — additive QT prolongation risk; list includes many antibiotics, antiemetics, antihistamines",
            "Acetazolamide: reduces periodic paralysis frequency in some ATS patients",
            "ICD implantation: consider in patients with syncope, sustained VT, or frequent complex arrhythmias; "
            "SCD risk lower than CPVT/LQT2 but not negligible",
            "Potassium supplementation: maintain normokalemia (K+ 4.0–4.5 mmol/L) to reduce PP attacks",
            "Genetic counselling: AD; ~30-40% de novo; triad may be incomplete in relatives",
            "Annual cardiology review + Holter: arrhythmia burden monitoring; ECG family screening",
        ],
        "key_ddx": [
            "CPVT type 1 (RYR2): bidirectional VT with exercise — no dysmorphic features, no PP; "
            "different treatment (BB + flecainide; ICD adjunct NOT replacement in CPVT)",
            "LQT1 (KCNQ1): QT prolongation + arrhythmia — no PP, no dysmorphic features, no BidVT",
            "LQT2 (KCNH2): sudden arousal trigger; drug DDI; no PP, no dysmorphic",
            "HypoKPP type 1 (CACNA1S): PP with low K+ but NO arrhythmia, NO dysmorphic",
            "HyperKPP (SCN4A): PP + myotonia but NO cardiac features, NO dysmorphic",
            "Digitalis toxicity: bidirectional VT mimics ATS/CPVT — check digoxin level",
        ],
        "onset_range_y": (2, 25),
        "cardiac_risk": True,
        "arrhythmia_risk": True,
        "myotonia": False,
        "periodic_paralysis": True,
        "mh_risk": False,
        "deafness_risk": False,
        "juvenile_onset": True,
        "bidirectional_vt": True,
        "ck_range": (80, 300),
        "attack_k_trend": "Variable",
        "first_line_drug": "Flecainide",
        "critical_avoid": "QT-prolonging drugs; digitalis toxicity; confusing with CPVT",
    },
    # ── CLCN1 — ClC-1 / Myotonia Congenita ───────────────────────────────
    {
        "gene": "CLCN1", "protein": "Skeletal Muscle Chloride Channel (ClC-1)",
        "alias": "CLCN1; OMIM gene 118425; 7q34; 988 aa; Myotonia Congenita Thomsen AD (OMIM #160800) / Becker AR (OMIM #255700); MEXILETINE first-line; QUININE/QUINIDINE ABSOLUTELY CONTRAINDICATED",
        "aa": "988 aa", "kDa": "112 kDa",
        "channel_class": (
            "CLCN1 encodes ClC-1, the predominant chloride channel in adult skeletal muscle membrane. "
            "ClC-1 carries ~80% of the resting muscle membrane conductance (gCl). "
            "NORMAL FUNCTION: high gCl clamps membrane potential near E_Cl (≈ resting potential) → "
            "dampens post-action-potential depolarising afterpotentials → prevents repetitive firing. "
            "MECHANISM OF DISEASE: LOF CLCN1 mutations → reduced gCl → membrane becomes electrically "
            "unstable after action potential → repetitive involuntary discharges → MYOTONIA. "
            "THOMSEN (AD): heterozygous LOF → milder myotonia; dominant-negative mechanism. "
            "BECKER (AR): biallelic LOF → severe myotonia; transient weakness at start of movement "
            "(Becker-specific feature: brief paralysis then recovery with repeated contractions). "
            "WARM-UP PHENOMENON (Becker): myotonia improves with repeated movement (opposite of SCN4A-PMC "
            "where cold/repeated movement worsens). "
            "QUININE/QUINIDINE ABSOLUTELY CONTRAINDICATED: these drugs BLOCK ClC-1 → "
            "paradoxically WORSEN myotonia in CLCN1 patients (mechanism: quinidine is a ClC-1 blocker; "
            "historically used for other myotonias but catastrophically wrong for CLCN1). "
            "MEXILETINE: sodium channel blocker (NOT a ClC-1 blocker) → reduces persistent INa → "
            "reduces membrane hyperexcitability → effective for CLCN1 myotonia."
        ),
        "channel_group": "Chloride Channel / ClC-1 / Skeletal Muscle / gCl",
        "channel_type": "Myotonia Congenita: Thomsen (AD/milder) / Becker (AR/severe + warm-up phenomenon)",
        "locus": "7q34", "omim_gene": 118425, "omim_disease": 160800,
        "inheritance": (
            "Thomsen disease: AD (heterozygous dominant-negative LOF). Milder myotonia. "
            "Becker disease: AR (biallelic LOF — compound het or homozygous). Severe myotonia. "
            "Thomsen: onset infancy/early childhood; p.Gly190Ser (G190S) most common. "
            "Becker: onset childhood-adolescence; p.Phe413Cys, p.Tyr578Asp; transient paralysis episodes. "
            "CLCN1 mutations: >200 identified; LOF by reduced chloride conductance. "
            "Thomsen: 50% offspring risk. Becker: 25% risk if both parents are carriers; "
            "Becker carriers may have subtle myotonia (heterozygote expression). "
            "Frequency: Becker MC more common than Thomsen globally."
        ),
        "phenotype": (
            "THOMSEN ONSET: infancy/early childhood; difficulty releasing grip; facial stiffness. "
            "PHENOTYPE: grip myotonia (cannot release handshake); percussion myotonia (thenar eminence); "
            "lid lag (orbicularis oculi myotonia); slow facial expression changes. "
            "NO PERIODIC PARALYSIS in CLCN1 (key distinction from SCN4A + KCNJ2). "
            "NO CARDIAC INVOLVEMENT. "
            "BECKER (AR): similar but SEVERE; transient paralysis (starting movement → brief weakness "
            "before myotonia resolves = warm-up period); percussion myotonia striking. "
            "WARM-UP PHENOMENON (Becker): myotonia decreases with repeated contractions — the pathognomonic "
            "feature distinguishing from SCN4A-PMC (which worsens with cold/repetition). "
            "COLD WORSENS CLCN1 myotonia (as in many sodium channelopathies). "
            "EMG: myotonic discharges (dive-bomber sound) at rest and on percussion. "
            "CK: mildly elevated (100–500 IU/L). MUSCLE HYPERTROPHY: common in Becker (Herculean appearance)."
        ),
        "disease": (
            "CLCN1 Myotonia Congenita. Diagnosis: gene panel + EMG (myotonic discharges) + clinical. "
            "TREATMENT: Mexiletine 150–300 mg TDS FIRST-LINE (sodium channel stabiliser — reduces "
            "persistent INa that sustains myotonic discharges; effective for both Thomsen and Becker). "
            "QUININE/QUINIDINE ABSOLUTELY CONTRAINDICATED — blocks ClC-1, worsens myotonia paradoxically. "
            "Carbamazepine: second-line (sodium channel stabiliser, less evidence than mexiletine). "
            "Lamotrigine: third-line evidence. Tocilizumab: anecdotal. "
            "Warm environments; avoid cold triggers; occupational adaptations."
        ),
        "treatment_options": [
            "Mexiletine 150–300 mg TDS: FIRST-LINE — sodium channel blocker; reduces persistent INa "
            "→ dampens myotonic discharges; effective in BOTH Thomsen (AD) and Becker (AR); "
            "same drug as SCN4A myotonia but different mechanism of benefit",
            "QUININE/QUINIDINE: ABSOLUTELY CONTRAINDICATED — PARADOXICAL MYOTONIA WORSENING; "
            "ClC-1 blocker; historically used for 'cramps' — CATASTROPHIC in CLCN1; "
            "check ALL patient medications; stop immediately if encountered",
            "Carbamazepine 200–400 mg BD: second-line sodium channel stabiliser; less evidence than mexiletine; "
            "useful if mexiletine not tolerated; CYP450 interactions",
            "Lamotrigine 50–200 mg/day: third-line; sodium channel; slower titration needed",
            "Avoid cold: warm clothing, warm water when washing hands; myotonia significantly worsened by cold",
            "Warm-up exercises before demanding tasks (especially Becker patients with transient paralysis on initiation)",
            "Occupational therapy: grip aids, ergonomic adaptations; driving safety assessment",
            "Genetic counselling: Thomsen (AD, 50% risk); Becker (AR, 25% risk; carrier screening for partners)",
        ],
        "key_ddx": [
            "SCN4A PMC: cold-triggered myotonia + periodic paralysis; worsens with repetition (opposite warm-up)",
            "SCN4A HyperKPP: periodic paralysis + myotonia; K+ elevated during attack",
            "Myotonic Dystrophy DM1 (DMPK): systemic features (cataracts, cardiac, endocrine, cognitive); "
            "CTG repeat; ice-pick face — if systemic features present, exclude DM1 first",
            "Myotonic Dystrophy DM2 (CNBP/ZNF9): proximal > distal; CCTG repeat; less severe facial myotonia",
            "Schwartz-Jampel syndrome: myotonia + skeletal dysplasia (blepharospasm, short stature) — continuous",
            "Hyperkalaemic PP mimics (renal failure, medications) — check electrolytes",
        ],
        "onset_range_y": (0, 20),
        "cardiac_risk": False,
        "arrhythmia_risk": False,
        "myotonia": True,
        "periodic_paralysis": False,
        "mh_risk": False,
        "deafness_risk": False,
        "juvenile_onset": True,
        "bidirectional_vt": False,
        "ck_range": (100, 500),
        "attack_k_trend": "Not Applicable",
        "first_line_drug": "Mexiletine",
        "critical_avoid": "QUININE/QUINIDINE — absolutely contraindicated (paradoxical worsening)",
    },
    # ── KCNQ1 — Kv7.1 / LQT1 / JLNS ─────────────────────────────────────
    {
        "gene": "KCNQ1", "protein": "Voltage-Gated Potassium Channel Kv7.1 (IKs)",
        "alias": "KCNQ1; OMIM gene 607542; 11p15.5; 676 aa; LQT1 AD (OMIM #192500); JLNS AR biallelic (OMIM #220400) + sensorineural deafness; SWIMMING HIGH RISK; Beta-blockers mandatory",
        "aa": "676 aa", "kDa": "75 kDa",
        "channel_class": (
            "KCNQ1 encodes Kv7.1, the α-subunit of the slow delayed rectifier potassium channel (IKs), "
            "which assembles with KCNE1 (minK) β-subunit. IKs current mediates cardiac action potential "
            "phase 3 repolarisation reserve — critical during adrenergic stimulation (exercise, emotion) "
            "when heart rate increases demand on repolarisation. "
            "MECHANISM OF DISEASE (LQT1, AD LOF): heterozygous LOF → reduced IKs → impaired "
            "repolarisation → QT prolongation → TdP (Torsades de Pointes) → VF → SCD. "
            "CATECHOLAMINE SENSITIVITY: IKs is the dominant repolarisation current during sympathetic "
            "stimulation — LOF means EXERCISE and EMOTION dramatically reduce repolarisation reserve → "
            "SWIMMING and DIVING are HIGH-RISK (immersion + intense adrenergic surge + cold trigger). "
            "JLNS (Jervell-Lange-Nielsen Syndrome): BIALLELIC KCNQ1 LOF → severe LQT1 + "
            "SENSORINEURAL DEAFNESS (IKs essential for endolymph K+ recycling in cochlea). "
            "JLNS has longest QT and highest SCD risk of any LQT syndrome. "
            "CLINICAL RULE: ALL LQT patients → HEARING SCREEN MANDATORY (JLNS may be missed). "
            "KCNE1 mutations: same LQT1 phenotype (Romano-Ward without deafness — different gene, same channel)."
        ),
        "channel_group": "Slow Delayed Rectifier K+ Channel / Kv7.1-KCNE1 / IKs / Cardiac Repolarisation",
        "channel_type": "LQT1 (AD LOF) + JLNS (AR biallelic LOF + sensorineural deafness)",
        "locus": "11p15.5", "omim_gene": 607542, "omim_disease": 192500,
        "inheritance": (
            "LQT1: AD (autosomal dominant LOF). Most common LQT syndrome (~30–35% of all LQTS). "
            "JLNS: AR (biallelic LOF) → severe LQTS + sensorineural deafness. "
            "p.Gly168Arg, p.Ala341Val, p.Trp248Arg: established LQT1 pathogenic variants. "
            "Penetrance: ~25–40% of LQT1 mutation carriers have QTc >480ms; many have QTc 450–480ms. "
            "Females: longer QTc, higher symptom risk postpuberty. "
            "Males: higher pre-pubertal risk; relative risk equalises after puberty. "
            "De novo mutations documented. Population prevalence: ~1:5,000–10,000."
        ),
        "phenotype": (
            "QTc: typically 470–550ms (LQT1); >550ms in JLNS. "
            "TRIGGERS: EXERCISE (especially SWIMMING) and EMOTIONAL STRESS — catecholamine surge "
            "→ IKs demand exceeds reduced IKs supply → TdP. Swimming is THE highest-risk activity in LQT1. "
            "EVENTS: syncope (T-wave oversensing in ICD), VF, SCD. "
            "JLNS: profound bilateral sensorineural deafness from birth; extremely long QT; "
            "highest SCD risk of all LQTS; ICD often required despite beta-blockers. "
            "ECG: broad-based T-wave (LQT1 ECG morphology); prolonged QTc. "
            "RISK STRATIFICATION: QTc >500ms + prior events + male sex pre-puberty → high risk. "
            "ASYMPTOMATIC CARRIERS: up to 60% LQT1 gene carriers never have events — but STILL avoid "
            "swimming, QT-prolonging drugs, electrolyte abnormalities. "
            "HEARING: ALL LQT patients → hearing test (JLNS may present as 'deaf child with arrhythmia')."
        ),
        "disease": (
            "LQT1 Romano-Ward / JLNS. Diagnosis: gene panel + ECG (QTc, T-wave morphology) + "
            "exercise test (QTc paradoxical prolongation with exercise = LQT1 specific finding). "
            "TREATMENT: Beta-blockers MANDATORY first-line ALL LQT1 (nadolol or propranolol; "
            "reduces event rate by >60%). ICD for survivors of VF, JLNS, high-risk features. "
            "AVOID: QT-prolonging drugs; electrolyte abnormalities; SWIMMING without supervision. "
            "SWIMMING RESTRICTION: high-risk activity → pool supervisor, lifeguard, medical ID bracelet; "
            "competitive swimming prohibited; leisure swimming with supervision and beta-blockers."
        ),
        "treatment_options": [
            "Beta-blockers MANDATORY: nadolol 0.5–2 mg/kg/day (first choice — long-acting, non-selective); "
            "propranolol 2–4 mg/kg/day (alternative); bisoprolol acceptable; "
            "reduces event rate >60% in LQT1; do NOT stop abruptly (rebound arrhythmia risk)",
            "SWIMMING RESTRICTION: HIGH RISK — avoid competitive swimming; leisure swimming only with "
            "lifeguard present, beta-blocker optimised, medical ID bracelet; "
            "diving prohibited; cold water immersion extremely high risk",
            "ICD implantation: survivors of VF/cardiac arrest; JLNS (biallelic, severe QT, deaf); "
            "QTc >500ms + symptoms despite beta-blockers; high-risk features on risk calculator",
            "Mexiletine adjunct: reduces QTc in LQT1 by shortening action potential; "
            "useful when QTc remains >500ms on beta-blockers",
            "Avoid QT-prolonging drugs: check crediblemeds.org for EVERY medication; "
            "KCNQ1 and KCNH2 patients most vulnerable; include antibiotics, antiemetics, antidepressants",
            "Electrolyte monitoring: maintain K+ 4.0–4.5 mmol/L, Mg2+ 0.8–1.0 mmol/L; "
            "hypokalaemia + hypomagnesaemia worsen QT; IV electrolytes during illness/vomiting",
            "JLNS management: beta-blockers + ICD in most cases; profound deafness → hearing aids/CI; "
            "genetic counselling (AR — 25% sibling risk; heterozygous parents LQT1 themselves)",
            "Medical alert ID bracelet: QT syndrome; beta-blocker medication; emergency contact",
        ],
        "key_ddx": [
            "LQT2 (KCNH2): arousal trigger (alarm clock); T-wave bifid/notched; massive drug DDI list",
            "LQT3 (SCN5A): SCN5A GOF; bradycardia + sleep trigger; Brugada alleles also in SCN5A",
            "JLNS vs Waardenburg syndrome: deafness + pigmentation anomalies in Waardenburg (PAX3/MITF); "
            "no QT prolongation in Waardenburg — ECG discriminates",
            "Drug-induced LQTS (diLQTS): many QT-prolonging drugs; identify and stop offending agent",
            "LQT1 vs LQT5 (KCNE1/minK): same IKs channel; KCNE1 mutation = Jervell-Lange-Nielsen JLNS2 if biallelic",
            "Brugada syndrome (SCN5A): sodium channel; ST elevation V1-V3; fever unmasks; no QT prolongation",
        ],
        "onset_range_y": (0, 40),
        "cardiac_risk": True,
        "arrhythmia_risk": True,
        "myotonia": False,
        "periodic_paralysis": False,
        "mh_risk": False,
        "deafness_risk": True,
        "juvenile_onset": True,
        "bidirectional_vt": False,
        "ck_range": (70, 200),
        "attack_k_trend": "Not Applicable",
        "first_line_drug": "Beta-blockers (nadolol/propranolol)",
        "critical_avoid": "Swimming (high risk); QT-prolonging drugs; electrolyte abnormalities",
    },
    # ── KCNH2 — Kv11.1 / HERG / LQT2 ────────────────────────────────────
    {
        "gene": "KCNH2", "protein": "Rapid Delayed Rectifier K+ Channel Kv11.1 (hERG/IKr)",
        "alias": "KCNH2; hERG; OMIM gene 152427; 7q36.1; 1159 aa; LQT2 AD (OMIM #613688); SUDDEN AROUSAL trigger (alarm clock); massive drug-DDI list; HERG block by hundreds of drugs",
        "aa": "1159 aa", "kDa": "127 kDa",
        "channel_class": (
            "KCNH2 encodes Kv11.1 (hERG — human Ether-à-go-go Related Gene), the α-subunit of the "
            "rapid delayed rectifier potassium channel (IKr). IKr is the dominant repolarisation current "
            "during the late plateau and phase 3 of the cardiac action potential, "
            "operating across most physiological heart rates. "
            "MECHANISM OF DISEASE (LQT2, AD LOF): heterozygous LOF → reduced IKr → prolonged "
            "action potential → QT prolongation → TdP → VF → SCD. "
            "SUDDEN AROUSAL TRIGGER: the classic LQT2 trigger — sudden unexpected sound (alarm clock, "
            "doorbell, phone ringing, baby crying) during sleep or rest → sympathetic burst → "
            "demand for fast repolarisation CANNOT be met by reduced IKr → TdP. "
            "HYPOKALAEMIA CRITICAL SENSITIVITY: IKr amplitude is profoundly reduced by low extracellular K+ "
            "(K+ shifts hERG gating) → hypokalaemia dramatically worsens QT in LQT2. "
            "Electrolyte monitoring MANDATORY. "
            "MASSIVE DRUG-DDI LIST: hERG channel has a unique drug-binding site in its inner pore cavity "
            "(aromatic/hydrophobic residues accessible from the cytoplasm); HUNDREDS of structurally unrelated "
            "drugs block hERG → ACQUIRED LQT2. Drug classes: antipsychotics, macrolide antibiotics, "
            "fluoroquinolones, antiemetics, antifungals, antihistamines, antidepressants. "
            "Check crediblemeds.org for EVERY drug prescribed to LQT2 patients."
        ),
        "channel_group": "Rapid Delayed Rectifier K+ Channel / Kv11.1-hERG / IKr / Cardiac Repolarisation",
        "channel_type": "LQT2 (AD LOF) — Sudden Arousal Trigger; Drug-DDI Critical; Hypokalaemia Sensitiser",
        "locus": "7q36.1", "omim_gene": 152427, "omim_disease": 613688,
        "inheritance": (
            "AD (autosomal dominant LOF). Second most common LQT syndrome (~25–30% of all LQTS). "
            "p.Ala614Val, p.Gly628Ser, p.Asn470Asp: established pathogenic variants. "
            "Trafficking-deficient mutations (channels synthesised but not trafficked to membrane): "
            "p.Gly628Ser — common; rescue with elevated extracellular K+ or pharmacological chaperones. "
            "Penetrance: higher than LQT1 for arrhythmic events (~50–70% lifetime). "
            "Females: higher risk than males postpuberty. "
            "De novo mutations documented. Population prevalence: ~1:6,000–10,000."
        ),
        "phenotype": (
            "QTc: typically 480–560ms; can exceed 600ms. "
            "TRIGGERS: SUDDEN AROUSAL — auditory stimuli during sleep or rest (alarm clock, phone, "
            "doorbell, baby crying). Exercise is LESS common trigger than LQT1. "
            "EVENTS: syncope, VF, SCD — often nocturnal. "
            "ECG: BIFID or NOTCHED T-WAVE (pathognomonic LQT2 morphology; split T-wave peak); "
            "prolonged QTc. T-wave morphology differentiates LQT1 (broad-based) vs LQT2 (bifid/notched). "
            "HYPOKALAEMIA: profoundly worsens QTc; maintain K+ ≥4.0 mmol/L — "
            "IV KCl during vomiting/diarrhoea episodes is life-saving. "
            "DRUG INTERACTIONS: any new prescription must be checked against hERG-blocking drug list. "
            "RISK: highest SCD risk of the three major LQT syndromes (LQT1/2/3) in some analyses."
        ),
        "disease": (
            "LQT2 Romano-Ward. Diagnosis: gene panel + ECG (bifid T-wave, prolonged QTc) + drug history. "
            "TREATMENT: Beta-blockers first-line (less effective than in LQT1 but still mandatory). "
            "ICD for high-risk patients. Potassium supplementation (target K+ 4.0–4.5 mmol/L). "
            "ALARM MANAGEMENT: avoid sudden auditory stimuli — silent mode on phone at night; "
            "progressive alarm clock (gradual volume increase); medical alert bracelet. "
            "Drug review: stop ALL hERG-blocking drugs if any started recently → QTc response."
        ),
        "treatment_options": [
            "Beta-blockers: nadolol (preferred) or propranolol; MANDATORY first-line all LQT2; "
            "less effective than LQT1 (arousal trigger partly non-adrenergic) but still reduces events",
            "Potassium supplementation: target K+ 4.0–4.5 mmol/L; MANDATORY — hypokalaemia critically "
            "worsens QTc in LQT2 (hERG gating profoundly K+-dependent); oral KCl supplements; "
            "IV KCl during illness with vomiting/diarrhoea — potentially life-saving",
            "AVOID QT-prolonging drugs: MANDATORY — check crediblemeds.org for EVERY medication; "
            "drug classes to avoid: macrolides (azithromycin), fluoroquinolones, ondansetron, haloperidol, "
            "methadone, many antihistamines; patient must carry drug interaction card",
            "ICD implantation: VF survivors; QTc >500ms + prior syncope; high-risk features; "
            "women with QTc >500ms (female sex = high risk in LQT2)",
            "Alarm modification: silent phone at night; progressive/gentle alarms; reduce sudden auditory stimuli; "
            "doorbell/phone alert systems to avoid startle; especially important during sleep",
            "Mexiletine adjunct: shortens QTc in LQT2 (reduces late INa, indirect repolarisation improvement); "
            "add when QTc >500ms on beta-blockers",
            "Electrolyte monitoring: serum K+ + Mg2+ at each visit; supplement both; "
            "maintain Mg2+ 0.8–1.0 mmol/L (hypomagnesaemia worsens TdP risk)",
            "Genetic counselling: AD; 50% familial risk; drug interaction card for every family member tested positive",
        ],
        "key_ddx": [
            "LQT1 (KCNQ1): exercise/swimming trigger vs LQT2 (arousal trigger); T-wave morphology different",
            "LQT3 (SCN5A): sleep/bradycardia trigger; SCN5A GOF; T-wave late-onset peaked; mexiletine more effective",
            "Drug-induced LQTS: acquired hERG block; identify offending agent; usually reversible",
            "LQT5 (KCNE1) / LQT6 (KCNE2): β-subunit mutations affecting IKs/IKr; phenotypically similar",
            "Brugada syndrome: SCN5A; type 1 ST elevation V1-V3; fever trigger; no QT prolongation",
            "Andersen-Tawil (KCNJ2): bidirectional VT + PP + dysmorphic features; LQT7",
        ],
        "onset_range_y": (0, 45),
        "cardiac_risk": True,
        "arrhythmia_risk": True,
        "myotonia": False,
        "periodic_paralysis": False,
        "mh_risk": False,
        "deafness_risk": False,
        "juvenile_onset": True,
        "bidirectional_vt": False,
        "ck_range": (70, 200),
        "attack_k_trend": "Not Applicable",
        "first_line_drug": "Beta-blockers + K+ supplementation",
        "critical_avoid": "QT-prolonging drugs (MASSIVE list); hypokalaemia; sudden auditory stimuli (night)",
    },
    # ── SCN5A — Nav1.5 / Brugada / LQT3 ─────────────────────────────────
    {
        "gene": "SCN5A", "protein": "Cardiac Voltage-Gated Sodium Channel α-Subunit (Nav1.5)",
        "alias": "SCN5A; OMIM gene 600163; 3p22.2; 2016 aa; Brugada Syndrome BrS1 (OMIM #601144); LQT3 (OMIM #603830); PCCD; SIDS; FEVER unmasks Brugada; Quinidine for Brugada VT",
        "aa": "2016 aa", "kDa": "227 kDa",
        "channel_class": (
            "SCN5A encodes Nav1.5, the principal voltage-gated sodium channel α-subunit of adult "
            "cardiomyocytes (atrial + ventricular). Nav1.5 generates the rapid INa responsible for "
            "cardiac action potential phase 0 (rapid upstroke) and contributes a small persistent "
            "(late) INa during the plateau phase. "
            "MULTIPLE ALLELIC DISEASES from SCN5A mutations: "
            "(1) BRUGADA SYNDROME (BrS1, LOF/dominant-negative): reduced INa in epicardial RV → "
            "loss of AP dome in RVOT epicardium → Phase 2 re-entry → VF → SCD. "
            "Type 1 pattern (coved ST elevation ≥2mm in V1-V2) DIAGNOSTIC. "
            "FEVER UNMASKS BRUGADA: fever reduces Nav1.5 kinetics → exaggerates INa reduction → "
            "unmasking of Type 1 pattern in previously silent carriers. MANDATORY antipyretics "
            "(paracetamol/acetaminophen; NOT ibuprofen/aspirin which have cardiac effects). "
            "(2) LQT3 (GOF): persistent late INa (gain of inactivation failure) → prolonged AP → "
            "QT prolongation → TdP. Triggers: BRADYCARDIA + SLEEP (opposite of LQT1/2). "
            "(3) PCCD (Progressive Cardiac Conduction Disease): LOF → defective conduction system. "
            "(4) SIDS: SCN5A LOF in some sudden infant death cases. "
            "QUINIDINE FOR BRUGADA: Ito blocker → restores RVOT epicardial AP dome → "
            "eliminates Phase 2 re-entry; indicated for electrical storm or symptomatic Brugada "
            "when ICD inappropriate. "
            "AVOID SODIUM CHANNEL BLOCKERS IN BRUGADA: flecainide, procainamide, ajmaline (used "
            "DIAGNOSTICALLY) — these drugs block Nav1.5 → WORSEN Type 1 pattern → VF risk."
        ),
        "channel_group": "Voltage-Gated Cardiac Sodium Channel / Nav1.5 / INa / Cardiac AP Phase 0",
        "channel_type": "Brugada BrS1 (LOF) + LQT3 (GOF) + PCCD + SIDS (allelic SCN5A spectrum)",
        "locus": "3p22.2", "omim_gene": 600163, "omim_disease": 601144,
        "inheritance": (
            "Brugada BrS1: AD (LOF/dominant-negative). Most common identified Brugada gene (~20–30% of BrS). "
            "LQT3: AD (GOF — persistent late INa). "
            "PCCD: AD (LOF — conduction defect). "
            "De novo mutations documented for all subtypes. "
            "p.Arg1232Trp, p.Glu1784Lys, p.Phe1486Leu: established pathogenic variants. "
            "Penetrance: ~20–30% of SCN5A Brugada carriers are symptomatic; majority have ECG change only. "
            "Male predominance in Brugada (testosterone modulates Ito; SCD risk 8× higher in males). "
            "Asian populations higher prevalence (Southeast Asian males highest risk globally)."
        ),
        "phenotype": (
            "BRUGADA: Type 1 pattern (spontaneous or provoked) in V1-V2; "
            "syncope or SCD without structural heart disease, often during sleep/rest. "
            "FEVER UNMASKS: febrile illness converts silent/type 2/3 ECG to diagnostic type 1. "
            "FEVER MANAGEMENT: paracetamol (acetaminophen) mandatory at first sign of fever; "
            "avoid ibuprofen (cardiac Nav effects); hospital admission for fever >38.5°C in confirmed BrS. "
            "LQT3: QTc >480ms; late-onset peaked T-wave; events during bradycardia/sleep. "
            "PCCD: progressive AV block, bundle branch block, sinus node dysfunction. "
            "SIDS association: suspected in some infant sudden death families. "
            "SODIUM CHANNEL BLOCKER CHALLENGE: ajmaline/flecainide IV → unmasks type 1 if positive "
            "(DIAGNOSTIC use only — not for long-term treatment)."
        ),
        "disease": (
            "SCN5A channelopathy spectrum. Brugada: risk stratify (asymptomatic vs symptomatic; "
            "spontaneous vs drug-induced type 1). "
            "ICD: symptomatic Brugada (VF, syncope); consider in high-risk asymptomatic. "
            "Quinidine: Brugada with electrical storm or recurrent VF in ICD-ineligible patients; "
            "Ito blocker → restores AP dome → prevents Phase 2 re-entry. "
            "AVOID flecainide/procainamide/amiodarone in Brugada. "
            "Catheter ablation (RVOT epicardium): newer therapy for drug-refractory Brugada electrical storm."
        ),
        "treatment_options": [
            "FEVER MANAGEMENT — MANDATORY: paracetamol/acetaminophen at first sign of fever; "
            "target temperature <37.5°C; avoid ibuprofen; avoid aspirin in children; "
            "hospital admission for fever >38.5°C in confirmed BrS — fever can unmask fatal arrhythmia",
            "ICD implantation: first-line for SYMPTOMATIC Brugada (VF survivor, unexplained syncope); "
            "asymptomatic spontaneous type 1: risk stratify with EP study; "
            "asymptomatic drug-induced type 1 only: risk very low, ICD usually not indicated",
            "Quinidine 300–600 mg TDS (Ito blocker): Brugada electrical storm, VF recurrence with ICD, "
            "Brugada in infants/children, symptomatic short-coupled PVCs; "
            "NOT widely available; monitor QTc (prolongation risk)",
            "AVOID SODIUM CHANNEL BLOCKERS IN BRUGADA: flecainide, procainamide, ajmaline (therapeutic use); "
            "these drugs WORSEN type 1 pattern and risk VF; ajmaline/flecainide = DIAGNOSTIC PROVOCATION ONLY",
            "LQT3 treatment: beta-blockers (less effective than LQT1/2 — bradycardia is trigger); "
            "mexiletine MOST EFFECTIVE in LQT3 (reduces persistent late INa → shortens QT dramatically); "
            "ICD for high-risk LQT3 (QTc >500ms, events)",
            "PCCD treatment: pacemaker for symptomatic AV block or sinus node dysfunction; "
            "annual ECG + Holter for PCCD monitoring; avoid drugs that slow conduction",
            "Avoid drugs that unmask/worsen Brugada: cocaine, alcohol excess, tricyclic antidepressants, "
            "lithium, carbamazepine, phenytoin, propofol — all can unmask type 1 pattern",
            "Genetic counselling: AD; family cascade ECG screening; drug allergy card for Brugada; "
            "male relatives at higher risk (testosterone effect on Ito)",
        ],
        "key_ddx": [
            "ARVC (arrhythmogenic RV cardiomyopathy): structural RV disease on MRI/echo; "
            "epsilon waves; ARVC genes (PKP2, DSP, DSG2) vs SCN5A",
            "LQT1/LQT2 (KCNQ1/KCNH2): QT prolongation vs Brugada (ST elevation no QT); different triggers",
            "Early repolarisation syndrome: ST elevation in inferior + lateral leads; less V1-V2 coved pattern",
            "Benign RBBB: V1-V2 rsR' pattern without coved ST elevation ≥2mm",
            "Right heart abnormalities (PE, RV myocarditis): transient ST changes; clinical context",
            "Sodium channel blocker toxicity (drug-induced Brugada pattern): identify and stop drug",
        ],
        "onset_range_y": (0, 60),
        "cardiac_risk": True,
        "arrhythmia_risk": True,
        "myotonia": False,
        "periodic_paralysis": False,
        "mh_risk": False,
        "deafness_risk": False,
        "juvenile_onset": False,
        "bidirectional_vt": False,
        "ck_range": (70, 200),
        "attack_k_trend": "Not Applicable",
        "first_line_drug": "Quinidine (BrS); Mexiletine (LQT3); ICD (symptomatic)",
        "critical_avoid": "FEVER (unmasks Type 1 Brugada); sodium channel blockers (flecainide in BrS)",
    },
    # ── RYR2 — Ryanodine Receptor 2 / CPVT1 ──────────────────────────────
    {
        "gene": "RYR2", "protein": "Ryanodine Receptor 2 (RyR2) — Cardiac SR Ca2+ Release Channel",
        "alias": "RYR2; OMIM gene 180902; 1q43; 5038 aa; CPVT1 (OMIM #604772) + ARVC2; EXERCISE-TRIGGERED bidirectional/polymorphic VT; Flecainide + Beta-blockers; ICD adjunct NOT replacement",
        "aa": "5038 aa", "kDa": "560 kDa (homotetramer ~2.2 MDa)",
        "channel_class": (
            "RYR2 encodes the Ryanodine Receptor 2 (RyR2), a homotetrameric SR (sarcoplasmic reticulum) "
            "calcium release channel in cardiomyocytes. RyR2 is the primary Ca2+ release channel for "
            "cardiac EC coupling: membrane depolarisation → L-type Ca2+ channel (LTCC/Cav1.2) → "
            "Ca2+-induced Ca2+ release (CICR) via RyR2 → cytosolic Ca2+ transient → contraction. "
            "MECHANISM OF DISEASE (CPVT1, AD GOF): AD RYR2 gain-of-function mutations → "
            "channels sensitised to cytosolic Ca2+ activation → during ADRENERGIC STIMULATION "
            "(exercise, catecholamines) → SR Ca2+ overload + spontaneous Ca2+ sparks → "
            "DELAYED AFTERDEPOLARISATIONS (DADs) → triggered ventricular ectopy → "
            "BIDIRECTIONAL VENTRICULAR TACHYCARDIA → POLYMORPHIC VT → VF → SCD. "
            "EXERCISE IS THE PATHOGNOMONIC TRIGGER: arrhythmia occurs at predictable heart rate "
            "thresholds during exercise testing — diagnostic and therapeutic monitoring tool. "
            "FLECAINIDE MECHANISM IN CPVT: RyR2 block (independent of sodium channel block) → "
            "reduces spontaneous Ca2+ sparks → prevents DADs → suppresses triggered VT. "
            "ICD PARADOX IN CPVT: ICD shock → catecholamine surge (pain/fear) → MORE triggered "
            "activity → VT STORM; ICD should NEVER replace beta-blockers + flecainide; "
            "ICD is ADJUNCT for survivors only. "
            "ARVC2: some RYR2 mutations associated with ARVC phenotype (different from typical desmosomal ARVC)."
        ),
        "channel_group": "SR Calcium Release Channel / RyR2 / CICR / Cardiac EC Coupling",
        "channel_type": "CPVT1 (AD GOF — exercise VT) + ARVC2 (allelic); ICD adjunct NOT replacement for BB+flecainide",
        "locus": "1q43", "omim_gene": 180902, "omim_disease": 604772,
        "inheritance": (
            "AD (autosomal dominant GOF). CPVT1 most common CPVT (~50–60% of all CPVT). "
            "CPVT2 (AR, CASQ2): less common; biallelic calsequestrin-2 mutations. "
            "De novo RYR2 mutations: common in CPVT1 (~20–30%). "
            "p.Ser2246Leu, p.Arg2474Ser, p.Arg4497Cys: well-established pathogenic CPVT1 variants. "
            "Penetrance: high for exercise-triggered arrhythmia (~70–80% lifetime event rate without treatment). "
            "High lethality without diagnosis: ~30% of untreated CPVT patients die before age 30."
        ),
        "phenotype": (
            "ONSET: childhood-adolescence (typically 7–15 years; range 2–40 years). "
            "TRIGGERS: PHYSICAL EXERCISE and EMOTIONAL STRESS — catecholamine surge is the pathognomonic trigger. "
            "EVENTS: syncope during exercise (NOT at rest); if VT degenerates → SCD during sports. "
            "BIDIRECTIONAL VT: alternating 180° QRS axis beat-to-beat — PATHOGNOMONIC (also in ATS but "
            "ATS has dysmorphic features + PP; CPVT does NOT). "
            "POLYMORPHIC VT → VF if not cardioverted. "
            "RESTING ECG: NORMAL (no QT prolongation, no Brugada, no delta waves) — "
            "this makes CPVT particularly dangerous (no resting ECG clue). "
            "EXERCISE TEST: arrhythmia onset at predictable HR (usually >120–130 bpm); PVCs → BidVT → VT. "
            "EMOTIONAL STRESS: emotional triggers (sudden fright, anger) → adrenergic → VT. "
            "FAMILY HISTORY: unexplained drowning, sports-related SCD, young SCD — CPVT in relatives. "
            "CK: normal."
        ),
        "disease": (
            "CPVT1 — RYR2-related Catecholaminergic Polymorphic VT. Diagnosis: gene panel + "
            "exercise test (bidirectional/polymorphic VT at ≥120 bpm) + genetic testing. "
            "TREATMENT: Beta-blockers MANDATORY (nadolol preferred; maximally tolerated dose) + "
            "Flecainide (RyR2 blocker + NaCh blocker) in all patients — COMBINATION is standard. "
            "EXERCISE RESTRICTION MANDATORY: avoid competitive sports; supervised exercise only. "
            "ICD: adjunct in survivors of VF; NEVER instead of beta-blockers + flecainide "
            "(ICD shock = catecholamine surge = VT storm = death by ICD)."
        ),
        "treatment_options": [
            "Beta-blockers MANDATORY + MAXIMAL: nadolol (first choice — long-acting, non-selective) "
            "1–4 mg/kg/day; propranolol 2–6 mg/kg/day; suppress catecholamine-triggered arrhythmia; "
            "do NOT stop abruptly; titrate to maximal tolerated dose",
            "Flecainide MANDATORY (added to beta-blocker): 100–200 mg BD; "
            "DUAL MECHANISM: (1) RyR2 channel block → reduces Ca2+ sparks/DADs; "
            "(2) sodium channel block → reduces triggered activity; "
            "reduces VT burden 60–80% beyond beta-blockers alone; COMBINATION IS STANDARD OF CARE",
            "EXERCISE RESTRICTION MANDATORY: no competitive sports; no high-intensity exercise; "
            "supervised moderate activity only with beta-blockers optimised; "
            "avoid all catecholamine-surge activities (horror films, roller coasters, heated arguments)",
            "ICD ADJUNCT (NOT REPLACEMENT): survivors of VF/cardiac arrest or high-risk despite maximal "
            "medical therapy; ICD shock triggers catecholamine surge → VT storm — "
            "BETA-BLOCKERS + FLECAINIDE MUST BE MAXIMISED BEFORE AND AFTER ICD IMPLANT; "
            "ICDs do NOT replace medication in CPVT — they are the last resort add-on",
            "Left cardiac sympathetic denervation (LCSD): surgical option for refractory CPVT; "
            "reduces catecholamine effect at myocardium; adjunct to medications; "
            "evidence level B; consider when maximal medical therapy fails",
            "Avoid catecholamine excess: avoid decongestants (ephedrine, pseudoephedrine), "
            "stimulants (methylphenidate, amphetamine, cocaine), excessive caffeine",
            "Emergency management: IV beta-blocker (esmolol) for VT storm; "
            "IV flecainide if available; magnesium 2g IV for polymorphic VT; avoid isoproterenol",
            "Genetic counselling: AD; 50% offspring risk; de novo mutations ~20-30%; "
            "ALL first-degree relatives need exercise ECG testing regardless of resting ECG",
        ],
        "key_ddx": [
            "Andersen-Tawil syndrome (KCNJ2): bidirectional VT + PP + dysmorphic features — "
            "ATS dysmorphic features absent in CPVT; treatment different (flecainide in both but different context)",
            "Brugada syndrome (SCN5A): RV ST changes; fever trigger; no exercise trigger; resting ECG abnormal",
            "LQT1 (KCNQ1): QT prolongation on resting ECG (CPVT = normal resting ECG)",
            "ARVC (desmosomal): structural RV changes on MRI; epsilon waves; ARVC gene panel (PKP2, DSP)",
            "WPW (Wolff-Parkinson-White): delta wave on resting ECG; SVT not VT",
            "Digitalis toxicity: bidirectional VT mimics CPVT — check digoxin level; medication history",
        ],
        "onset_range_y": (2, 35),
        "cardiac_risk": True,
        "arrhythmia_risk": True,
        "myotonia": False,
        "periodic_paralysis": False,
        "mh_risk": False,
        "deafness_risk": False,
        "juvenile_onset": True,
        "bidirectional_vt": True,
        "ck_range": (70, 200),
        "attack_k_trend": "Not Applicable",
        "first_line_drug": "Beta-blockers (nadolol) + Flecainide (BOTH mandatory)",
        "critical_avoid": "ICD instead of (not in addition to) BB+flecainide; catecholamine-surge activities",
    },
]


def _gen_patients(gene_data: dict, seed: int) -> list:
    rng = random.Random(seed)
    gene = gene_data["gene"]
    patients = []
    ck_lo, ck_hi = gene_data["ck_range"]
    onset_lo, onset_hi = gene_data["onset_range_y"]

    for i in range(40):
        onset = round(rng.uniform(onset_lo, onset_hi), 1)
        ck_val = round(rng.uniform(ck_lo, ck_hi))

        # Severity based on gene characteristics
        r = rng.random()
        if gene in ("CLCN1", "SCN4A"):
            # Myotonia — mostly mild to moderate
            sev = "Mild" if r < 0.45 else ("Moderate" if r < 0.80 else "Severe")
        elif gene in ("RYR2", "KCNQ1", "KCNH2", "SCN5A"):
            # Cardiac arrhythmia genes — moderate-severe events
            sev = "Moderate" if r < 0.40 else ("Severe" if r < 0.70 else "Mild")
        elif gene == "KCNJ2":
            # ATS — variable
            sev = "Moderate" if r < 0.50 else ("Mild" if r < 0.75 else "Severe")
        else:
            # CACNA1S HypoKPP
            sev = "Moderate" if r < 0.45 else ("Mild" if r < 0.72 else "Severe")

        # Gene-specific features
        arrhythmia  = gene_data["arrhythmia_risk"] and rng.random() < 0.65
        myotonia    = gene_data["myotonia"] and rng.random() < 0.85
        pp_episodes = gene_data["periodic_paralysis"] and rng.random() < 0.75
        mh_risk     = gene_data["mh_risk"]
        deafness    = gene_data["deafness_risk"] and rng.random() < (0.15 if gene == "KCNQ1" else 0.0)
        bid_vt      = gene_data["bidirectional_vt"] and arrhythmia and rng.random() < 0.60
        icd         = arrhythmia and rng.random() < (0.50 if gene == "RYR2" else 0.35)
        beta_b      = gene_data["cardiac_risk"] and rng.random() < 0.88
        mexiletine  = gene in ("SCN4A", "CLCN1") and rng.random() < 0.78
        acetazol    = gene in ("CACNA1S", "SCN4A") and rng.random() < 0.65
        flecainide  = gene in ("RYR2", "KCNJ2") and rng.random() < (0.82 if gene == "RYR2" else 0.60)

        # Treatment summary
        if gene == "SCN4A":
            tx = "Mexiletine + acetazolamide (PP prevention)" if pp_episodes else "Mexiletine (myotonia)"
        elif gene == "CACNA1S":
            tx = "Acetazolamide + oral KCl supplementation; avoid volatile anaesthetics (MH)"
        elif gene == "KCNJ2":
            tx = "Flecainide + beta-blocker; avoid QT-prolonging drugs; K+ supplementation"
        elif gene == "CLCN1":
            tx = "Mexiletine (first-line); avoid quinine/quinidine ABSOLUTELY CONTRAINDICATED"
        elif gene == "KCNQ1":
            tx = "Nadolol (beta-blocker mandatory); ICD if high-risk; no swimming unsupervised"
        elif gene == "KCNH2":
            tx = "Beta-blocker + K+ supplement + drug DDI review; ICD if high-risk"
        elif gene == "SCN5A":
            tx = "ICD (symptomatic BrS); quinidine (Ito blocker); paracetamol when febrile; avoid Na blockers"
        elif gene == "RYR2":
            tx = "Nadolol + flecainide (BOTH mandatory); ICD adjunct; exercise restriction"
        else:
            tx = "Standard channelopathy management"

        pid = f"CHAN-{gene}-{seed}-{i+1:03d}"
        sex = rng.choice(["M", "F"])
        # Male predominance in Brugada
        if gene == "SCN5A":
            sex = "M" if rng.random() < 0.72 else "F"

        patients.append({
            "id": pid, "gene": gene, "sex": sex,
            "onset_age_y": onset,
            "severity": sev,
            "arrhythmia": arrhythmia,
            "myotonia": myotonia,
            "periodic_paralysis": pp_episodes,
            "mh_risk": mh_risk,
            "sensorineural_deafness": deafness,
            "bidirectional_vt": bid_vt,
            "icd_implanted": icd,
            "beta_blocker": beta_b,
            "mexiletine": mexiletine,
            "acetazolamide": acetazol,
            "flecainide": flecainide,
            "juvenile_onset": onset < 18,
            "ck_iu_l": ck_val,
            "current_treatment": tx,
            "channel_group": gene_data["channel_group"],
            "cardiac_risk": gene_data["cardiac_risk"],
        })
    return patients


def _gen_cohort() -> list:
    all_pts = []
    for idx, gene_data in enumerate(CHANNELOPATHY_GENES):
        seed = SEED_BASE + idx
        all_pts.extend(_gen_patients(gene_data, seed))
    return all_pts


def get_overview() -> dict:
    patients = _gen_cohort()
    n = len(patients)

    sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
    for p in patients:
        sev[p["severity"]] += 1

    arrhythmia_n  = sum(1 for p in patients if p["arrhythmia"])
    myotonia_n    = sum(1 for p in patients if p["myotonia"])
    pp_n          = sum(1 for p in patients if p["periodic_paralysis"])
    mh_n          = sum(1 for p in patients if p["mh_risk"])
    deaf_n        = sum(1 for p in patients if p["sensorineural_deafness"])
    bid_vt_n      = sum(1 for p in patients if p["bidirectional_vt"])
    icd_n         = sum(1 for p in patients if p["icd_implanted"])
    bb_n          = sum(1 for p in patients if p["beta_blocker"])
    mex_n         = sum(1 for p in patients if p["mexiletine"])
    flec_n        = sum(1 for p in patients if p["flecainide"])
    juv_n         = sum(1 for p in patients if p["juvenile_onset"])
    cardiac_n     = sum(1 for p in patients if p["cardiac_risk"])

    onsets = [p["onset_age_y"] for p in patients]
    mean_onset = round(sum(onsets) / len(onsets), 1)
    mean_ck = round(sum(p["ck_iu_l"] for p in patients) / n)

    return {
        "atlas": "Channelopathy-Atlas",
        "full_name": "Complete 8-Gene Skeletal Muscle + Cardiac Channelopathy Atlas",
        "subtitle": "SCN4A·CACNA1S·KCNJ2·CLCN1·KCNQ1·KCNH2·SCN5A·RYR2 — 320 patients (8×40, seeds 1054–1061)",
        "description": (
            "Comprehensive atlas of the 8 most clinically critical skeletal muscle and cardiac "
            "channelopathy genes. Covers sodium channels (SCN4A Nav1.4 — skeletal; SCN5A Nav1.5 — cardiac), "
            "calcium channels (CACNA1S Cav1.1 — EC coupling; RYR2 SR Ca2+ release), "
            "potassium channels (KCNQ1 Kv7.1/IKs LQT1; KCNH2 Kv11.1/HERG/IKr LQT2; KCNJ2 Kir2.1/IK1 ATS), "
            "and chloride channel (CLCN1 ClC-1 myotonia congenita). "
            "CRITICAL CLINICAL RULES: MEXILETINE is first-line for BOTH SCN4A and CLCN1 myotonia; "
            "QUININE/QUINIDINE ABSOLUTELY CONTRAINDICATED in CLCN1 (paradoxical worsening); "
            "FEVER unmasks SCN5A Brugada (paracetamol mandatory); SWIMMING is HIGH-RISK in KCNQ1 LQT1; "
            "SUDDEN AROUSAL is LQT2 (KCNH2) trigger; RYR2 CPVT requires BOTH flecainide + beta-blockers "
            "(ICD is adjunct, NOT replacement — shock causes VT storm); JLNS (biallelic KCNQ1) = deafness "
            "(hearing screen ALL LQT patients); ANDERSEN-TAWIL bidirectional VT is NOT CPVT — "
            "different gene (KCNJ2 vs RYR2), different treatment."
        ),
        "total_patients": n,
        "genes_covered": len(CHANNELOPATHY_GENES),
        "patients_per_gene": 40,
        "seed_range": "1054–1061",
        "gene_list": [g["gene"] for g in CHANNELOPATHY_GENES],
        "channel_category_breakdown": {
            "Skeletal Muscle Sodium Channel / Nav1.4 (HyperKPP/PMC/PAM — Mexiletine)": ["SCN4A"],
            "L-Type Calcium Channel / Cav1.1 (HypoKPP type 1 + MH susceptibility)": ["CACNA1S"],
            "Inward Rectifier K+ / Kir2.1 (Andersen-Tawil — Bidirectional VT + PP + Dysmorphic)": ["KCNJ2"],
            "Skeletal Muscle Cl− Channel / ClC-1 (Myotonia Congenita — Quinine ABSOLUTELY CI)": ["CLCN1"],
            "Slow Delayed Rectifier K+ / Kv7.1-IKs (LQT1 + JLNS Deafness — Swimming HIGH RISK)": ["KCNQ1"],
            "Rapid Delayed Rectifier K+ / Kv11.1-HERG-IKr (LQT2 — Arousal Trigger + Drug DDI)": ["KCNH2"],
            "Cardiac Sodium Channel / Nav1.5 (Brugada+LQT3+PCCD — Fever Unmasks)": ["SCN5A"],
            "SR Calcium Release / RyR2 (CPVT1 — Exercise VT — Flecainide+BB BOTH mandatory)": ["RYR2"],
        },
        "severity": {
            "mild_pct": round(100 * sev["Mild"] / n, 1),
            "moderate_pct": round(100 * sev["Moderate"] / n, 1),
            "severe_pct": round(100 * sev["Severe"] / n, 1),
        },
        "mean_onset_age_y": mean_onset,
        "mean_ck_iu_l": mean_ck,
        "kpis": [
            {"label": "Total Patients", "value": n, "color": "#1565c0"},
            {"label": "Genes Covered", "value": len(CHANNELOPATHY_GENES), "color": "#2e7d32"},
            {"label": "Patients/Gene", "value": 40, "color": "#6a1b9a"},
            {"label": "Cardiac Risk Genes", "value": 5, "color": "#b71c1c"},
            {"label": "Mean Onset (y)", "value": mean_onset, "color": "#e65100"},
            {"label": "Seeds", "value": "1054–1061", "color": "#37474f"},
        ],
        "clinical_features_prevalence": {
            "arrhythmia_pct":          round(100 * arrhythmia_n / n, 1),
            "myotonia_pct":            round(100 * myotonia_n / n, 1),
            "periodic_paralysis_pct":  round(100 * pp_n / n, 1),
            "mh_risk_pct":             round(100 * mh_n / n, 1),
            "sensorineural_deafness_pct": round(100 * deaf_n / n, 1),
            "bidirectional_vt_pct":    round(100 * bid_vt_n / n, 1),
            "icd_implanted_pct":       round(100 * icd_n / n, 1),
            "beta_blocker_pct":        round(100 * bb_n / n, 1),
            "mexiletine_pct":          round(100 * mex_n / n, 1),
            "flecainide_pct":          round(100 * flec_n / n, 1),
            "juvenile_onset_pct":      round(100 * juv_n / n, 1),
            "cardiac_risk_pct":        round(100 * cardiac_n / n, 1),
        },
        "key_teaching_points": [
            "MEXILETINE: first-line sodium channel stabiliser for BOTH SCN4A myotonia (Nav1.4 GOF) AND "
            "CLCN1 myotonia congenita (ClC-1 LOF) — same drug, different pathomechanism; "
            "reduces persistent INa → dampens membrane hyperexcitability in both channelopathies",
            "QUININE/QUINIDINE ABSOLUTELY CONTRAINDICATED IN CLCN1 myotonia congenita — "
            "blocks ClC-1 → PARADOXICAL WORSENING of myotonia; historically used for 'cramps' — "
            "catastrophically wrong in CLCN1; check ALL medications for any quinine-containing products",
            "FEVER UNMASKS SCN5A BRUGADA: fever reduces Nav1.5 kinetics → exaggerates INa reduction → "
            "diagnostic type 1 coved ST pattern in previously silent carriers; "
            "paracetamol/acetaminophen MANDATORY at first sign of fever; hospital admission if >38.5°C",
            "SWIMMING HIGH RISK IN KCNQ1 LQT1: IKs is the dominant repolarisation reserve during "
            "catecholamine stimulation; immersion + adrenergic surge → IKs demand cannot be met → TdP; "
            "competitive swimming PROHIBITED; leisure swimming with lifeguard/supervision only",
            "SUDDEN AROUSAL TRIGGER IN KCNH2 LQT2: alarm clock, phone ringing, baby crying during sleep "
            "→ sympathetic burst → IKr demand unmet → TdP; silent phone at night; progressive alarms; "
            "hypokalaemia dramatically worsens LQT2 (maintain K+ ≥4.0 mmol/L; IV KCl during vomiting)",
            "RYR2 CPVT: FLECAINIDE + BETA-BLOCKERS BOTH MANDATORY — flecainide blocks RyR2 (independent "
            "of Nav) reducing Ca2+ sparks + DADs; ICD shock = catecholamine surge = VT STORM; "
            "ICD is ADJUNCT for survivors ONLY, never instead of maximised medications",
            "JLNS (biallelic KCNQ1): sensorineural deafness + severe LQT; ALL LQT patients → "
            "HEARING SCREEN MANDATORY; JLNS may present as 'deaf child with arrhythmia' — "
            "KCNQ1 gene panel before attributing deafness to other causes",
            "ANDERSEN-TAWIL (KCNJ2) bidirectional VT ≠ CPVT (RYR2): ATS has TRIAD (PP + arrhythmia + "
            "dysmorphic); CPVT has no dysmorphic features; treatment overlaps (flecainide in both) "
            "but mechanisms and additional therapy differ — DO NOT CONFLATE",
            "QT-PROLONGING DRUGS: KCNQ1 and KCNH2 patients most vulnerable; check crediblemeds.org "
            "for EVERY prescription; hERG (KCNH2) blocked by hundreds of structurally unrelated drugs "
            "(macrolides, fluoroquinolones, antiemetics, antipsychotics, antihistamines, antifungals)",
            "CACNA1S MH SUSCEPTIBILITY: HypoKPP type 1 alleles (R528H, R1239H) confer MH risk; "
            "MH alert card MANDATORY; total IV anaesthesia (propofol + remifentanil) for ALL surgery; "
            "dantrolene available in theatre at all times",
        ],
        "drug_alerts": [
            "QUININE/QUINIDINE ABSOLUTELY CONTRAINDICATED IN CLCN1 MYOTONIA CONGENITA — "
            "paradoxical worsening; blocks ClC-1; check ALL medications; stop immediately if found",
            "FEVER IN SCN5A BRUGADA: paracetamol at first fever sign; NO ibuprofen/aspirin; "
            "hospital admission for fever >38.5°C; fever can trigger fatal VF within hours",
            "FLECAINIDE/PROCAINAMIDE AVOID IN BRUGADA (SCN5A): sodium channel blockers WORSEN "
            "type 1 pattern; diagnostic provocation use ONLY (never therapeutic in Brugada)",
            "HYPOKALAEMIA IN KCNH2 LQT2: critically worsens QTc; maintain K+ ≥4.0 mmol/L; "
            "IV KCl during vomiting/diarrhoea illness — potentially life-saving emergency action",
            "RYR2 CPVT ICD PARADOX: ICD shock → catecholamine surge → more triggered VT → storm; "
            "NEVER use ICD as monotherapy; beta-blockers + flecainide MUST be maximised",
            "CACNA1S VOLATILE ANAESTHETICS: MH-susceptible (R528H, R1239H alleles); "
            "succinylcholine ABSOLUTELY AVOIDED; total IV anaesthesia mandatory; MH alert card issued",
        ],
        "standards": [
            "crediblemeds.org: QT drug interaction database — MANDATORY for KCNQ1 + KCNH2 patients",
            "CPVT exercise test protocol: arrhythmia onset >120 bpm = diagnostic; "
            "treatment target = suppression of VT at equivalent heart rate",
            "LQT risk stratification: QTc >500ms + symptoms + female + family SCD = high risk",
            "Brugada: male + SE Asian + spontaneous type 1 pattern + syncope = highest risk",
            "MH: European MH Group (EMHG) protocol for CACNA1S/RYR1 surgical management",
        ],
    }


def get_breakdown() -> dict:
    patients = _gen_cohort()
    gene_profiles = []
    for gene_data in CHANNELOPATHY_GENES:
        gene_pts = [p for p in patients if p["gene"] == gene_data["gene"]]
        n = len(gene_pts)
        sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
        for p in gene_pts:
            sev[p["severity"]] += 1
        mean_ck_g = round(sum(p["ck_iu_l"] for p in gene_pts) / n)
        gene_profiles.append({
            "gene": gene_data["gene"],
            "protein": gene_data["protein"],
            "alias": gene_data["alias"],
            "locus": gene_data["locus"],
            "omim_gene": gene_data["omim_gene"],
            "omim_disease": gene_data["omim_disease"],
            "inheritance": gene_data["inheritance"],
            "channel_group": gene_data["channel_group"],
            "channel_type": gene_data["channel_type"],
            "aa": gene_data["aa"],
            "kDa": gene_data["kDa"],
            "channel_class": gene_data["channel_class"],
            "phenotype": gene_data["phenotype"],
            "disease": gene_data["disease"],
            "treatment_options": gene_data["treatment_options"],
            "key_ddx": gene_data["key_ddx"],
            "onset_range_y": list(gene_data["onset_range_y"]),
            "n_patients": n,
            "cardiac_risk": gene_data["cardiac_risk"],
            "arrhythmia_risk": gene_data["arrhythmia_risk"],
            "myotonia": gene_data["myotonia"],
            "periodic_paralysis": gene_data["periodic_paralysis"],
            "mh_risk": gene_data["mh_risk"],
            "deafness_risk": gene_data["deafness_risk"],
            "juvenile_onset": gene_data["juvenile_onset"],
            "bidirectional_vt": gene_data["bidirectional_vt"],
            "first_line_drug": gene_data["first_line_drug"],
            "critical_avoid": gene_data["critical_avoid"],
            "mean_ck_iu_l": mean_ck_g,
            "severity_distribution": {
                "mild_pct": round(100 * sev["Mild"] / n, 1),
                "moderate_pct": round(100 * sev["Moderate"] / n, 1),
                "severe_pct": round(100 * sev["Severe"] / n, 1),
            },
            "clinical_features": {
                "arrhythmia_pct":         round(100 * sum(1 for p in gene_pts if p["arrhythmia"]) / n, 1),
                "myotonia_pct":           round(100 * sum(1 for p in gene_pts if p["myotonia"]) / n, 1),
                "periodic_paralysis_pct": round(100 * sum(1 for p in gene_pts if p["periodic_paralysis"]) / n, 1),
                "bidirectional_vt_pct":   round(100 * sum(1 for p in gene_pts if p["bidirectional_vt"]) / n, 1),
                "icd_implanted_pct":      round(100 * sum(1 for p in gene_pts if p["icd_implanted"]) / n, 1),
                "beta_blocker_pct":       round(100 * sum(1 for p in gene_pts if p["beta_blocker"]) / n, 1),
                "mexiletine_pct":         round(100 * sum(1 for p in gene_pts if p["mexiletine"]) / n, 1),
                "flecainide_pct":         round(100 * sum(1 for p in gene_pts if p["flecainide"]) / n, 1),
                "juvenile_onset_pct":     round(100 * sum(1 for p in gene_pts if p["juvenile_onset"]) / n, 1),
            },
            "sample_patients": gene_pts[:3],
        })
    return {
        "atlas": "Channelopathy-Atlas",
        "genes": gene_profiles,
        "total_patients": len(patients),
        "pharmacology_matrix": {
            "mexiletine": ["SCN4A (HyperKPP/PMC/PAM myotonia)", "CLCN1 (Myotonia Congenita — Thomsen + Becker)"],
            "quinine_quinidine_CI": ["CLCN1 (ABSOLUTELY CONTRAINDICATED — paradoxical ClC-1 block)"],
            "acetazolamide": ["SCN4A (HyperKPP PP prevention)", "CACNA1S (HypoKPP type 1 prevention)", "KCNJ2 (ATS PP reduction)"],
            "beta_blockers_mandatory": ["KCNQ1 (LQT1)", "KCNH2 (LQT2)", "RYR2 (CPVT1)"],
            "flecainide_antiarrhythmic": ["RYR2 (CPVT1 — RyR2 block)", "KCNJ2 (ATS — VT suppression)"],
            "quinidine_antiarrhythmic": ["SCN5A (Brugada — Ito blocker; NOT Na channel blocker context)"],
            "icd_indicated": ["KCNQ1 (survivors + JLNS)", "KCNH2 (high-risk LQT2)", "SCN5A (symptomatic Brugada)", "RYR2 (survivors ONLY; adjunct)"],
            "sodium_blockers_AVOID_Brugada": ["SCN5A (flecainide, procainamide, ajmaline — worsen type 1 pattern)"],
            "volatile_anaesthetics_CI": ["CACNA1S (MH susceptibility R528H/R1239H)"],
            "qt_prolonging_drugs_CI": ["KCNQ1", "KCNH2", "KCNJ2 (LQT7)"],
        },
    }


def get_definitions() -> dict:
    return {
        "atlas": "Channelopathy-Atlas",
        "terms": [
            {
                "term": "Mexiletine — First-Line Sodium Channel Stabiliser for Both SCN4A and CLCN1 Myotonia",
                "definition": (
                    "Mexiletine is a class Ib oral sodium channel blocker used as first-line treatment for "
                    "myotonia in BOTH SCN4A channelopathy (HyperKPP, PMC, PAM) and CLCN1 myotonia congenita "
                    "(Thomsen and Becker disease). Despite treating both, the mechanism differs: "
                    "In SCN4A GOF myotonia: mexiletine binds inactivated Nav1.4 → reduces persistent INa "
                    "(the pathological current) → dampens repetitive firing. "
                    "In CLCN1 LOF myotonia: mexiletine reduces persistent INa (secondary membrane instability "
                    "from reduced gCl) → restores effective membrane damping. "
                    "DOSE: 150–300 mg TDS (start 150 mg BD, titrate). No routine LFT monitoring required. "
                    "SIDE EFFECTS: dose-dependent GI upset (nausea, diarrhoea — take with food); "
                    "dizziness; cardiac proarrhythmia at toxic doses (rare at therapeutic doses). "
                    "MONITORING: ECG at baseline (check QRS width — class I antiarrhythmic risk); "
                    "avoid in significant structural heart disease. "
                    "QUININE/QUINIDINE: completely different — ABSOLUTELY CONTRAINDICATED in CLCN1 "
                    "(paradoxical ClC-1 block worsens myotonia — DO NOT confuse with mexiletine)."
                ),
                "clinical_rule": "MEXILETINE = first-line BOTH SCN4A + CLCN1; QUININE/QUINIDINE = ABSOLUTELY CI in CLCN1",
            },
            {
                "term": "Quinine/Quinidine — Absolutely Contraindicated in CLCN1 Myotonia Congenita",
                "definition": (
                    "Quinine and quinidine BLOCK the ClC-1 chloride channel — the SAME channel that is "
                    "already dysfunctional (loss-of-function) in CLCN1 myotonia congenita. "
                    "Administration → further reduction of already-reduced gCl → PARADOXICAL WORSENING "
                    "of myotonia, potentially severe. "
                    "HISTORICAL CONTEXT: quinine was used for 'muscle cramps' and 'myotonia' before "
                    "genetic understanding — catastrophic in CLCN1; this remains a live medication error risk "
                    "because quinine is still prescribed for leg cramps (off-label) and malaria. "
                    "CLINICAL RULE: Any patient with CLCN1 myotonia congenita presenting acutely worsened "
                    "must have their medication list checked for quinine-containing products "
                    "(tonic water, antimalarials, herbal preparations). "
                    "QUINIDINE (antiarrhythmic): similarly contraindicated in CLCN1 — different from "
                    "its role as Ito blocker in Brugada (SCN5A) where it is therapeutic. "
                    "Quinidine is THERAPEUTIC in Brugada (not myotonia) and HARMFUL in CLCN1 — "
                    "two completely different contexts for the same drug."
                ),
                "clinical_rule": "STOP QUININE/QUINIDINE IMMEDIATELY if found in any CLCN1 patient — check all medications including OTC",
            },
            {
                "term": "SCN5A Brugada Syndrome — Fever Unmasks Type 1 Pattern",
                "definition": (
                    "Brugada syndrome type 1 ECG (coved ST elevation ≥2mm in V1-V2 with RBBB morphology) "
                    "may be concealed at rest in many mutation carriers — unmasked by specific triggers. "
                    "FEVER MECHANISM: elevated temperature → accelerated Nav1.5 channel inactivation → "
                    "exaggerates existing loss-of-function → RVOT epicardial AP dome loss → "
                    "Phase 2 re-entry → VF. Even fever of 38.0–38.5°C can unmask the pattern and trigger VF. "
                    "MANAGEMENT RULE: ALL SCN5A Brugada patients must: "
                    "(1) Carry a medical alert card specifying Brugada syndrome; "
                    "(2) Use paracetamol (acetaminophen) as antipyretic — NOT ibuprofen (cardiac Na effects), "
                    "NOT aspirin (Reye's risk in children); "
                    "(3) Seek medical attention for fever >38.5°C — consider inpatient monitoring; "
                    "(4) Preoperative notification — fever under general anaesthesia is a known trigger. "
                    "SODIUM CHANNEL BLOCKERS (flecainide, procainamide, ajmaline): WORSEN type 1 pattern "
                    "by further reducing INa — used ONLY as diagnostic PROVOCATION agents IV in cardiology lab, "
                    "NEVER therapeutically in diagnosed Brugada."
                ),
                "clinical_rule": "FEVER in Brugada = MEDICAL EMERGENCY; paracetamol only; no ibuprofen; hospital if >38.5°C",
            },
            {
                "term": "KCNQ1 LQT1 — Swimming as High-Risk Trigger",
                "definition": (
                    "LQT1 (KCNQ1 LOF) is the most exercise-sensitive LQTS. IKs (Kv7.1/KCNE1) is the "
                    "DOMINANT repolarisation reserve current during adrenergic stimulation — when heart rate "
                    "increases, IKs must scale proportionally to maintain repolarisation. "
                    "SWIMMING-SPECIFIC RISK: combination of (1) intense catecholamine surge (exercise), "
                    "(2) cold water (vagal → bradycardia → long diastole → long QT), "
                    "(3) water immersion (vagal tone altered), and (4) face immersion (diving reflex — "
                    "sudden vagal burst) → creates the highest-risk catecholamine+vagal alternation scenario "
                    "for LQT1 TdP. Up to 30% of LQT1 SCD events occur during or immediately after swimming. "
                    "MANAGEMENT: competitive swimming PROHIBITED; leisure swimming only with "
                    "trained lifeguard present, beta-blockers optimised, medical ID bracelet visible; "
                    "diving PROHIBITED; cold water swimming PROHIBITED. "
                    "EXERCISE STRESS TEST: LQT1 specific finding — QTc paradoxically PROLONGS with "
                    "exercise recovery (fails to shorten as expected) — confirms LQT1 diagnosis."
                ),
                "clinical_rule": "LQT1 + SWIMMING = HIGH RISK; competitive swimming PROHIBITED; leisure swimming needs lifeguard + beta-blockers",
            },
            {
                "term": "KCNH2 LQT2 — Sudden Arousal Trigger and Drug Interaction",
                "definition": (
                    "LQT2 (KCNH2 LOF) has two defining clinical features: "
                    "(1) SUDDEN AROUSAL TRIGGER: abrupt unexpected auditory stimuli (alarm clock, phone, "
                    "doorbell, baby crying, gunshot) during sleep or rest → rapid sympathetic burst → "
                    "IKr demand unmet → TdP. Most LQT2 events occur at night or at rest. "
                    "PRACTICAL MANAGEMENT: silent phone mode at night; progressive alarm clocks "
                    "(gradual volume increase); door chime replacement with vibrating devices. "
                    "(2) MASSIVE DRUG-DDI LIST: the hERG channel (Kv11.1/KCNH2) has a unique inner "
                    "pore cavity accessible from the cytoplasm with aromatic/hydrophobic binding sites; "
                    "hundreds of drugs from unrelated classes block hERG → acquired LQT2 → additive "
                    "QTc prolongation → TdP. "
                    "Drug classes that block hERG (not exhaustive): "
                    "Macrolide antibiotics (azithromycin, erythromycin, clarithromycin); "
                    "Fluoroquinolones (moxifloxacin > levofloxacin); "
                    "Antiemetics (ondansetron, domperidone, droperidol); "
                    "Antipsychotics (haloperidol, quetiapine, risperidone); "
                    "Antidepressants (citalopram, escitalopram at higher doses); "
                    "Antifungals (fluconazole, ketoconazole); antihistamines (astemizole — withdrawn); "
                    "Methadone; cocaine; hydroxychloroquine. "
                    "CHECK crediblemeds.org (CredibleMeds AZCERT database) for EVERY new drug."
                ),
                "clinical_rule": "LQT2: silent phone at night; check crediblemeds.org for EVERY drug; maintain K+ ≥4.0 mmol/L",
            },
            {
                "term": "RYR2 CPVT — ICD Paradox and Flecainide Mechanism",
                "definition": (
                    "Catecholaminergic Polymorphic VT (CPVT1, RYR2 GOF) is treated with the combination "
                    "of beta-blockers + flecainide. The ICD paradox is a critical clinical teaching point: "
                    "ICD SHOCK → PAIN + FEAR → CATECHOLAMINE SURGE → MORE RyR2-mediated DADs → "
                    "MORE triggered VT → VT STORM → ICD delivers more shocks → more catecholamines → "
                    "cycle of electrical storm → death by ICD-triggered VT storm. "
                    "ICD is therefore ADJUNCT for VF survivors ONLY — never monotherapy, "
                    "never instead of maximised medications. "
                    "FLECAINIDE MECHANISM IN CPVT (unique): "
                    "(1) Direct RyR2 channel block — reduces spontaneous Ca2+ sparks from SR → "
                    "reduces delayed afterdepolarisations (DADs); this is INDEPENDENT of sodium channel block. "
                    "(2) Sodium channel block — reduces triggered activity from DADs. "
                    "Both mechanisms synergise — combination with beta-blockers reduces VT by 60–80% "
                    "compared with beta-blockers alone. BOTH drugs are MANDATORY — not sequential. "
                    "EXERCISE TEST: used for diagnosis AND treatment monitoring — "
                    "goal is VT suppression at the heart rate that previously triggered VT."
                ),
                "clinical_rule": "CPVT: Flecainide + BB BOTH mandatory; ICD = adjunct for survivors ONLY; never instead of drugs",
            },
            {
                "term": "JLNS (Jervell-Lange-Nielsen Syndrome) — Deafness Hearing Screen Rule",
                "definition": (
                    "JLNS is the AR form of KCNQ1 channelopathy — biallelic loss-of-function → "
                    "severe LQT (QTc often >550ms, sometimes >600ms) + PROFOUND BILATERAL SENSORINEURAL "
                    "DEAFNESS from birth. "
                    "MECHANISM OF DEAFNESS: IKs (Kv7.1/KCNE1) is essential for K+ recycling in the "
                    "cochlea (endolymph homeostasis) — biallelic LOF → stria vascularis dysfunction → "
                    "absent endolymphatic K+ gradient → sensorineural hearing loss. "
                    "CLINICAL RULE: ALL patients with LQT syndrome (QTc >450ms) must have formal "
                    "audiological assessment — JLNS may present as 'congenitally deaf child with "
                    "unexplained syncope/seizures' where the underlying arrhythmia is missed. "
                    "Conversely, a deaf child with family history of sudden death → urgent ECG + "
                    "KCNQ1 gene panel. "
                    "JLNS MANAGEMENT: beta-blockers + ICD in most cases (SCD risk very high); "
                    "hearing rehabilitation (hearing aids or cochlear implant); "
                    "parents of JLNS child are BOTH heterozygous KCNQ1 carriers → LQT1 (Romano-Ward) — "
                    "both parents need ECG + beta-blocker consideration. "
                    "Heterozygous KCNQ1 carriers may also have JLNS2 (KCNE1 biallelic)."
                ),
                "clinical_rule": "ALL LQT patients → hearing screen; deaf child + syncope → urgent ECG + KCNQ1 panel",
            },
            {
                "term": "Andersen-Tawil vs CPVT — Bidirectional VT Differential",
                "definition": (
                    "Bidirectional ventricular tachycardia (BidVT) — alternating 180° QRS axis beat-to-beat — "
                    "occurs in two distinct genetic channelopathies with different treatment approaches: "
                    "ANDERSEN-TAWIL SYNDROME (ATS, KCNJ2 LOF, LQT7): "
                    "  - TRIAD: PP + ventricular arrhythmia + dysmorphic features "
                    "  - Dysmorphic features: low-set ears, micrognathia, clinodactyly, hypertelorism "
                    "  - PP: variable K+ (hypo- or hyperkalaemic) "
                    "  - SCD risk: LOWER than CPVT (arrhythmia is less malignant in ATS on average) "
                    "  - Treatment: Flecainide ± beta-blockers; acetazolamide for PP "
                    "CPVT TYPE 1 (RYR2 GOF): "
                    "  - No dysmorphic features "
                    "  - No periodic paralysis "
                    "  - Exercise/catecholamine PATHOGNOMONIC trigger "
                    "  - SCD risk: HIGH (30% mortality <30 years without treatment) "
                    "  - Treatment: Beta-blockers + flecainide BOTH mandatory; ICD adjunct "
                    "  - Normal resting ECG (no QT prolongation, no structural changes) "
                    "CLINICAL RULE: BidVT + dysmorphic + PP = ATS (KCNJ2); "
                    "BidVT + exercise trigger + no dysmorphic = CPVT (RYR2). "
                    "Digitalis toxicity = third cause of BidVT — check digoxin level first."
                ),
                "clinical_rule": "Bidirectional VT: ATS (KCNJ2) has dysmorphic+PP; CPVT (RYR2) has exercise trigger+normal resting ECG; check digoxin",
            },
        ],
    }
