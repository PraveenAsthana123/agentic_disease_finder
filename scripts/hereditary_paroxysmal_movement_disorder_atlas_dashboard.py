#!/usr/bin/env python3
"""Hereditary-Paroxysmal-Movement-Disorder-Atlas — Complete 8-Gene Paroxysmal Movement Disorder Atlas
(PKD-PRRT2 · PED-SLC2A1 · PNKD2-PNKD · EA1-KCNA1 ·
 EA2/SCA6/FHM1-CACNA1A · FHM2-ATP1A2 · ADCY5-Nocturnal · Hyperekplexia-SLC6A5).

PRRT2   (Proline-Rich Transmembrane Protein 2; 340 aa; 16p11.2; AD;
         PKD / BFIS / ICCA — most common paroxysmal movement disorder worldwide;
         MOVEMENT-TRIGGERED brief attacks <1 min, warning aura, NO LOC PATHOGNOMONIC;
         CARBAMAZEPINE >90% EFFECTIVE at low dose — dramatic response;
         seed SEED_BASE+0).
SLC2A1  (GLUT1 Glucose Transporter 1; 492 aa; 1p34.2; AD;
         GLUT1 Deficiency Syndrome / PED (Paroxysmal Exercise-Induced Dyskinesia);
         CSF GLUCOSE <45 mg/dL + LOW CSF:SERUM RATIO (<0.6) PATHOGNOMONIC;
         KETOGENIC DIET FIRST-LINE — seizures abate dramatically;
         seed SEED_BASE+1).
PNKD    (Paroxysmal Non-Kinesigenic Dyskinesia / MR-1; 385 aa; 2q35; AD;
         PNKD / BDC (Benign Dystonia with Choreoathetosis);
         CAFFEINE + ALCOHOL trigger PATHOGNOMONIC — minutes to 12-hour episodes;
         clonazepam partially effective; episodes NOT movement-triggered (key DDx PRRT2);
         seed SEED_BASE+2).
KCNA1   (Kv1.1 Voltage-Gated K+ Channel; 495 aa; 12p13.32; AD;
         EA1 (Episodic Ataxia Type 1);
         INTERICTAL MYOKYMIA (continuous rippling) PATHOGNOMONIC;
         episodes seconds-minutes triggered by startle/exercise;
         CARBAMAZEPINE + ACETAZOLAMIDE effective;
         seed SEED_BASE+3).
CACNA1A (Cav2.1 P/Q-Type Ca²⁺ Channel; 2510 aa; 19p13.13; AD;
         EA2 / FHM1 / SCA6 — allelic spectrum single gene three phenotypes;
         INTERICTAL NYSTAGMUS + episodes HOURS duration PATHOGNOMONIC for EA2;
         ACETAZOLAMIDE HIGHLY EFFECTIVE in EA2 — dramatic response;
         Progressive cerebellar atrophy in SCA6;
         seed SEED_BASE+4).
ATP1A2  (Na⁺/K⁺-ATPase α2; 1020 aa; 1q23.2; AD;
         FHM2 (Familial Hemiplegic Migraine Type 2);
         HEMIPLEGIC AURA + CONFUSION + PROLONGED WEAKNESS PATHOGNOMONIC;
         astrocytic glutamate/K⁺ clearance failure mechanism;
         VALPROATE + TOPIRAMATE + PREVENTIVES;
         seed SEED_BASE+5).
ADCY5   (Adenylyl Cyclase 5; 1261 aa; 3q21.3; AD gain-of-function;
         ADCY5-related movement disorder / Facial Dyskinesia with Choreiform Movements;
         NOCTURNAL ATTACKS from NREM SLEEP PATHOGNOMONIC — EEG normal (not epilepsy);
         CAFFEINE ABSOLUTELY CONTRAINDICATED — dramatically worsens;
         clonazepam + acetazolamide first-line;
         seed SEED_BASE+6).
SLC6A5  (GlyT2 Glycine Transporter 2; 797 aa; 11p15.1; AR;
         Hyperekplexia (Startle Disease) — GLYT2 type;
         EXAGGERATED STARTLE + NEONATAL HYPERTONIA + APNOEA PATHOGNOMONIC;
         CLONAZEPAM CURATIVE — dramatically reduces hyperekplexia;
         NOSE-TIPPING MANOEUVRE acute crisis intervention;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2166-2173).
"""

import random

SEED_BASE = 2166

PAROXYSMAL_GENES = [
    # -- PRRT2 — PKD / BFIS / ICCA ---------------------------------------------------
    {
        "gene": "PRRT2",
        "alt_name": (
            "PRRT2 (PRRT2-340aa-16p11.2 / AD — PKD-PRRT2-Most-Common-Paroxysmal-Movement-Disorder — "
            "MOVEMENT-TRIGGERED-Brief-<1-min-NO-LOC-PATHOGNOMONIC — "
            "CARBAMAZEPINE->90pct-Effective-Low-Dose-DRAMATIC-RESPONSE — "
            "16p11.2-DELETION-SYNDROME-Complicates-Genetic-Testing)"
        ),
        "protein": (
            "PRRT2 -- 16p11.2 AD -- PRRT2-340aa -- "
            "Proline-Rich-Transmembrane-Protein-2-Presynaptic-Membrane-SNAP25-Interactor -- "
            "PKD-OMIM-128200-Most-Common-Paroxysmal-Movement-Disorder-Worldwide -- "
            "Also-BFIS-Benign-Familial-Infantile-Seizures-OMIM-605751 -- "
            "Also-ICCA-Infantile-Convulsions-Choreoathetosis-Same-Gene -- "
            "MOVEMENT-TRIGGERED-Kinesigenic-Onset-ACTION-TRIGGERS-Attack-PATHOGNOMONIC -- "
            "Brief-Duration-<1-min-Typically-10-30-sec-Warning-Aura-Before-Attack -- "
            "No-LOC-No-EEG-Correlate-During-Attack-KEY-DDx-Epilepsy -- "
            "Choreoathetosis-Dystonia-Ballismus-Mix-During-Attack -- "
            "Carbamazepine-Very-Low-Dose-100-200mg-day->90pct-Effective-Dramatic -- "
            "16p11.2-Microdeletion-Syndrome-Complicates-NGS-CNV-Detection-Mandatory -- "
            "Truncating-Variants-Most-Common-Frameshift-Stop-Gained-LOF-Mechanism -- "
            "SNAP25-Binding-Presynaptic-Release-Modulation -- "
            "16p11.2"
        ),
        "locus": "16p11.2",
        "protein_size": "340 aa",
        "inheritance": (
            "AD (autosomal dominant); near-complete penetrance; "
            "truncating variants (frameshift/nonsense) >70% of cases; "
            "16p11.2 microdeletion detection requires CNV analysis (not just SNV); "
            "de novo in ~25% of PKD; BFIS/ICCA from same gene — family history may show mixed phenotypes; "
            "mosaicism reported."
        ),
        "key_features": [
            "PKD — movement-triggered paroxysmal dyskinesia, attack <1 min, warning aura, no LOC",
            "Carbamazepine >90% effective at very low doses (100-200 mg/day) — dramatic cessation",
            "No EEG correlate during attack — critical DDx from epileptic seizure",
            "BFIS/ICCA — same gene; family may show infantile seizures + later PKD",
            "16p11.2 deletion requires CNV analysis; SNV sequencing alone may miss",
            "DDx: SLC2A1 (exercise >5 min; low CSF glucose), PNKD (caffeine/alcohol, not movement)",
        ],
        "treatment": [
            "Carbamazepine 100-200 mg/day — FIRST-LINE >90% effective, very low doses sufficient",
            "Oxcarbazepine — alternative if CBZ intolerated",
            "Phenytoin — second-line option",
            "Spontaneous remission common in adulthood — trial of dose reduction after remission",
        ],
        "contraindications": [
            "Sodium channel blockers generally safe — do NOT use Ca-channel blockers as primary",
            "Avoid misdiagnosis as epilepsy requiring high-dose AED — low dose CBZ sufficient",
        ],
        "critical_pearls": [
            "CARBAMAZEPINE DOSE IS VERY LOW — many clinicians underdose or overdose; start 100 mg/day",
            "No LOC + movement-triggered + <1 min = PKD until proven otherwise",
            "Check for 16p11.2 deletion with chromosomal microarray if PRRT2 sequencing negative",
        ],
    },
    # -- SLC2A1 — GLUT1D / PED / Absence epilepsy ------------------------------------
    {
        "gene": "SLC2A1",
        "alt_name": (
            "SLC2A1 (SLC2A1-492aa-1p34.2 / AD — GLUT1-Deficiency-Syndrome-PED-Absence-Epilepsy — "
            "CSF-GLUCOSE-<45-mg/dL-CSF:SERUM-<0.6-PATHOGNOMONIC — "
            "KETOGENIC-DIET-FIRST-LINE-Seizures-Abate-Dramatically — "
            "EXERCISE->5-min-Triggers-Dyskinesia-KEY-DDx-PRRT2)"
        ),
        "protein": (
            "SLC2A1 -- 1p34.2 AD -- SLC2A1-492aa -- "
            "GLUT1-Glucose-Transporter-1-Facilitative-12-TM-Helix-Blood-Brain-Barrier -- "
            "GLUT1-Deficiency-Syndrome-G1D-OMIM-606777-De-Vivo-Disease -- "
            "PED-Paroxysmal-Exercise-Induced-Dyskinesia-PATHOGNOMONIC-Subtype -- "
            "Absence-Epilepsy-Early-Onset-Often-Misdiagnosed-Drug-Resistant -- "
            "Intellectual-Disability-Variable-Mild-Moderate -- "
            "CSF-Glucose-<45-mg/dL-ABSOLUTE-THRESHOLD-PATHOGNOMONIC -- "
            "CSF:Serum-Glucose-Ratio-<0.6-PATHOGNOMONIC-Normal->0.65 -- "
            "CSF-Lactate-Normal-KEY-DDx-Mitochondrial -- "
            "Ketogenic-Diet-FIRST-LINE-Provides-Ketone-Body-Fuel-Bypasses-GLUT1 -- "
            "Seizures-Abate-Dramatically-Within-Weeks-KD-Start -- "
            "Exercise->5min-Triggers-PED-vs-PRRT2-Any-Abrupt-Movement -- "
            "Fasting-Worsens-All-Symptoms-Never-Fast-GLUT1D -- "
            "Haploinsufficiency-Dominant-Negative-Rare-Mechanism -- "
            "1p34.2"
        ),
        "locus": "1p34.2",
        "protein_size": "492 aa",
        "inheritance": (
            "AD (autosomal dominant); haploinsufficiency mechanism; "
            "de novo in ~50% of cases; missense + deletion + frameshift variants; "
            "carrier parents may have mild phenotype; "
            "genetic testing: full gene + CNV analysis (deletions common); "
            "mosaic forms reported with milder phenotype."
        ),
        "key_features": [
            "CSF glucose <45 mg/dL + CSF:serum ratio <0.6 PATHOGNOMONIC — lumbar puncture MANDATORY",
            "PED — prolonged exercise (>5 min) triggers paroxysmal dyskinesia (DDx PRRT2: any movement)",
            "Early-onset absence epilepsy, often drug-resistant — KD diagnostic AND therapeutic",
            "CSF lactate NORMAL — critical DDx mitochondrial disease",
            "Fasting dramatically worsens all symptoms — NEVER fast GLUT1D patients",
            "Ketogenic diet: seizures abate within weeks; movement disorder improves over months",
        ],
        "treatment": [
            "Ketogenic diet — FIRST-LINE for all GLUT1D phenotypes (seizures, movement, cognition)",
            "Modified Atkins Diet — alternative if classical KD not tolerated",
            "Triheptanoin (C7 anaplerotic) — compassionate use for movement symptoms",
            "Avoid valproate — inhibits fatty acid oxidation, may worsen metabolic compensation",
            "AVOID FASTING — ensure continuous carbohydrate substrate even during illness",
            "Frequent small meals + complex carbohydrates — minimize glucose dips",
        ],
        "contraindications": [
            "VALPROATE — inhibits fatty acid oxidation, may worsen GLUT1D metabolic state",
            "FASTING for surgery/procedures — glucose infusion MANDATORY perioperatively",
            "Phenobarbital — may inhibit GLUT1 expression (theoretical concern)",
        ],
        "critical_pearls": [
            "LUMBAR PUNCTURE IS DIAGNOSTIC — always measure CSF glucose in drug-resistant childhood epilepsy",
            "Send CSF and serum glucose simultaneously — ratio is more reliable than absolute CSF value",
            "PED lasting hours after exercise is almost always GLUT1D, not PKD (too long for PRRT2)",
        ],
    },
    # -- PNKD — PNKD2 / BDC ----------------------------------------------------------
    {
        "gene": "PNKD",
        "alt_name": (
            "PNKD (PNKD-385aa-2q35 / AD — PNKD2-MR-1-Benign-Dystonia-Choreoathetosis — "
            "CAFFEINE+ALCOHOL-TRIGGER-PATHOGNOMONIC-Not-Movement-KEY-DDx-PRRT2 — "
            "Episodes-10-min-to-12-hr-KEY-DDx-EA2-Hours-Not-PRRT2-Seconds — "
            "Clonazepam-Partially-Effective)"
        ),
        "protein": (
            "PNKD -- 2q35 AD -- PNKD-385aa -- "
            "Myofibrillogenesis-Regulator-1-MR-1-Hydroxyacylglutathione-Hydrolase-Domain -- "
            "PNKD-Paroxysmal-Non-Kinesigenic-Dyskinesia-OMIM-118800 -- "
            "Also-BDC-Benign-Dystonia-with-Choreoathetosis-Older-Nomenclature -- "
            "CAFFEINE-TRIGGER-PATHOGNOMONIC-Coffee-Tea-Chocolate-Energy-Drinks -- "
            "ALCOHOL-TRIGGER-PATHOGNOMONIC-Even-Small-Amounts-< 1-Standard-Drink -- "
            "NOT-Movement-Triggered-KEY-DDx-PRRT2-PKD-Movement-Kinesigenic -- "
            "Episodes-Duration-10-min-to-12-hr-KEY-DDx-EA1-Seconds-EA2-Hours -- "
            "Choreoathetosis-Dystonia-Ballismus-During-Attack-No-LOC -- "
            "Clonazepam-GABA-A-Partially-Effective-50-70pct-Reduction -- "
            "Stress-Fatigue-Fever-Also-Trigger-Non-Specific -- "
            "Glu7Ala-and-Ala9Val-HOTSPOT-Missense-Variants-Most-Common -- "
            "Spontaneous-Remission-30pct-Adulthood -- "
            "2q35"
        ),
        "locus": "2q35",
        "protein_size": "385 aa",
        "inheritance": (
            "AD (autosomal dominant); high penetrance; "
            "Glu7Ala (c.20A>C) and Ala9Val (c.26C>T) — hotspot missense variants accounting for >70%; "
            "genetic testing should include these two hotspots first; "
            "de novo rare (<10%); familial most common."
        ),
        "key_features": [
            "CAFFEINE + ALCOHOL triggers PATHOGNOMONIC — document trigger diary",
            "NOT movement-triggered — key DDx from PKD/PRRT2",
            "Episodes 10 min to 12 hours — key DDx: EA2 (hours but nystagmus), PRRT2 (seconds)",
            "Choreoathetosis + dystonia during attack; NO LOC; return to baseline complete",
            "Glu7Ala and Ala9Val hotspot variants >70% of cases — targeted testing first",
            "Spontaneous remission in 30% by adulthood",
        ],
        "treatment": [
            "CAFFEINE AVOIDANCE — most important non-pharmacological intervention",
            "ALCOHOL AVOIDANCE — major trigger, dependence risk if used as self-treatment",
            "Clonazepam 0.5-2 mg — partially effective, 50-70% attack frequency reduction",
            "Oxazepam — alternative benzodiazepine if clonazepam not tolerated",
            "Carbamazepine — INEFFECTIVE in PNKD (key DDx distinction from PKD/PRRT2)",
            "Stress management + regular sleep schedule — reduce non-specific triggers",
        ],
        "contraindications": [
            "CARBAMAZEPINE INEFFECTIVE — if CBZ fails, reconsider PKD vs PNKD diagnosis",
            "Alcohol as self-treatment — dependence risk high, triggers attacks even in small amounts",
        ],
        "critical_pearls": [
            "CAFFEINE diary is DIAGNOSTIC — if patient avoids caffeine attacks drop dramatically",
            "CBZ fails in PNKD but works in PKD — response to CBZ is a diagnostic test",
            "Glu7Ala hotspot — test this first before full gene sequencing (>40% of PNKD cases)",
        ],
    },
    # -- KCNA1 — EA1 ------------------------------------------------------------------
    {
        "gene": "KCNA1",
        "alt_name": (
            "KCNA1 (KCNA1-495aa-12p13.32 / AD — EA1-Episodic-Ataxia-Type-1 — "
            "INTERICTAL-MYOKYMIA-Continuous-Rippling-PATHOGNOMONIC — "
            "Startle/Exercise-Triggered-Seconds-to-Minutes-KEY-DDx-EA2-Hours — "
            "Carbamazepine+Acetazolamide-Effective)"
        ),
        "protein": (
            "KCNA1 -- 12p13.32 AD -- KCNA1-495aa -- "
            "Kv1.1-Voltage-Gated-Potassium-Channel-Alpha-Subunit-6-TM-Helix-Homotetrameric -- "
            "EA1-Episodic-Ataxia-Type-1-OMIM-160120 -- "
            "Also-Partial-Epilepsy-with-Peripheral-Neuropathy-Allelic -- "
            "INTERICTAL-MYOKYMIA-PATHOGNOMONIC-Continuous-Rippling-Especially-Periorbital-Hands -- "
            "Attacks-Triggered-Startle-Sudden-Movement-Exercise-Fever -- "
            "Attack-Duration-Seconds-to-Minutes-KEY-DDx-EA2-Hours -- "
            "Ataxia-Dysarthria-Jerky-Limb-Movements-During-Attack -- "
            "NO-Nystagmus-Between-Attacks-KEY-DDx-EA2-Interictal-Nystagmus -- "
            "EMG-Myokymic-Discharges-Spontaneous-Groups-of-Potentials -- "
            "Carbamazepine-Effective-Reduces-Attack-Frequency -- "
            "Acetazolamide-Also-Effective-Alternative-First-Line -- "
            "Phenytoin-Third-Line-Option -- "
            "Haploinsufficiency-Loss-of-Function-Mechanism -- "
            "12p13.32"
        ),
        "locus": "12p13.32",
        "protein_size": "495 aa",
        "inheritance": (
            "AD (autosomal dominant); high penetrance; "
            "missense variants predominantly; LOF mechanism; "
            "EMG required to demonstrate myokymia — clinical exam may miss subtle cases; "
            "genetic testing: full KCNA1 sequencing."
        ),
        "key_features": [
            "INTERICTAL MYOKYMIA — continuous muscle rippling especially periorbital, hands PATHOGNOMONIC",
            "Startle/exercise triggered; attack seconds to minutes (DDx EA2: hours)",
            "NO interictal nystagmus (DDx EA2: interictal nystagmus pathognomonic for EA2)",
            "EMG shows spontaneous myokymic discharges — diagnostic investigation",
            "Ataxia + dysarthria + limb jerking during attack; complete resolution between",
            "Carbamazepine AND acetazolamide both effective first-line options",
        ],
        "treatment": [
            "Carbamazepine 200-400 mg/day — first-line, reduces attack frequency and myokymia",
            "Acetazolamide 250-1000 mg/day — alternative first-line",
            "Phenytoin — third-line",
            "Avoid trigger factors: sudden movement, startle, exercise, fever",
            "Treatment of myokymia improves both interictal rippling and attack frequency",
        ],
        "contraindications": [
            "Avoid 4-aminopyridine (4-AP) — worsens Kv1.1 dysfunction (reduces K+ channel activity)",
        ],
        "critical_pearls": [
            "EMG IS MANDATORY in EA workup — myokymia on EMG = EA1 until proven otherwise",
            "Periorbital myokymia (eye-lid flickering at rest) is the classic clinical observation",
            "EA1 vs EA2: startle-triggered + myokymia + seconds = EA1; nausea/vomiting + hours + nystagmus = EA2",
        ],
    },
    # -- CACNA1A — EA2 / SCA6 / FHM1 -------------------------------------------------
    {
        "gene": "CACNA1A",
        "alt_name": (
            "CACNA1A (CACNA1A-2510aa-19p13.13 / AD — EA2-SCA6-FHM1-ALLELIC-SPECTRUM — "
            "INTERICTAL-NYSTAGMUS-+Episodes-HOURS-Duration-PATHOGNOMONIC-EA2 — "
            "ACETAZOLAMIDE-HIGHLY-EFFECTIVE-EA2-Dramatic-Response — "
            "Progressive-Cerebellar-Atrophy-SCA6-CAG-Repeat-Expansion)"
        ),
        "protein": (
            "CACNA1A -- 19p13.13 AD -- CACNA1A-2510aa -- "
            "Cav2.1-P/Q-Type-Voltage-Gated-Ca2+-Channel-Alpha1-Subunit-24-TM-Segments -- "
            "EA2-Episodic-Ataxia-Type-2-OMIM-108500-Most-Common-Episodic-Ataxia -- "
            "SCA6-Spinocerebellar-Ataxia-6-OMIM-183086-CAG-Repeat-Expansion-C-Terminal -- "
            "FHM1-Familial-Hemiplegic-Migraine-Type-1-OMIM-141500-GoF-Mechanism -- "
            "INTERICTAL-NYSTAGMUS-PATHOGNOMONIC-EA2-Always-Present-Between-Attacks -- "
            "Episodes-Hours-PATHOGNOMONIC-Duration-12-72-hr-KEY-DDx-EA1-Seconds -- "
            "Nausea-Vomiting-Headache-Common-During-EA2-Attack-Migraine-Overlap -- "
            "ACETAZOLAMIDE-HIGHLY-EFFECTIVE-EA2->80pct-Attack-Frequency-Reduction -- "
            "Missense-GoF-FHM1-LOF-EA2-Truncating-LOF-EA2-CAG-Repeat-SCA6 -- "
            "SCA6-CAG-Repeat->19-Normal-<20-Pathogenic-Southern-Blot-Required -- "
            "SCA6-Progressive-Cerebellar-Atrophy-50yr-Onset-No-Treatment -- "
            "FHM1-Hemiplegia-Profound-Aura-ICU-Required-Acute-Severe-Attacks -- "
            "4-Aminopyridine-K+-Blocker-Also-Effective-EA2 -- "
            "19p13.13"
        ),
        "locus": "19p13.13",
        "protein_size": "2510 aa",
        "inheritance": (
            "AD (autosomal dominant); "
            "EA2: LOF missense/truncating; SCA6: CAG repeat expansion (19-33 repeats pathogenic); "
            "FHM1: GOF missense; "
            "genetic testing MUST include repeat expansion analysis for SCA6 (southern blot or specific repeat assay); "
            "standard NGS may miss CAG expansion; "
            "de novo rare in EA2; familial most cases."
        ),
        "key_features": [
            "INTERICTAL NYSTAGMUS PATHOGNOMONIC — always present between attacks in EA2",
            "Episodes HOURS duration (12-72 hr) — DDx EA1 (seconds), PNKD (hours but no nystagmus)",
            "Nausea/vomiting/headache during attack — migraine overlap common",
            "Acetazolamide >80% reduction in EA2 attack frequency — DRAMATIC response",
            "SCA6 from same gene: CAG repeat >19 — progressive cerebellar atrophy, onset ~50yr",
            "FHM1 from same gene: profound hemiplegic aura — ICU management may be needed acutely",
        ],
        "treatment": [
            "Acetazolamide 250-1000 mg/day — FIRST-LINE EA2, dramatic response",
            "4-Aminopyridine (4-AP) 10-20 mg TID — effective alternative, blocks Kv channels",
            "Topiramate — second-line EA2 if acetazolamide not tolerated",
            "Verapamil — calcium channel blocker, anecdotal EA2 benefit",
            "Flunarizine — alternative calcium channel blocker, EA2 use reported",
            "FHM1 acute: triptans CONTRAINDICATED in hemiplegic migraine (see below)",
        ],
        "contraindications": [
            "TRIPTANS CONTRAINDICATED in FHM1 hemiplegic migraine — vasoconstrictive risk",
            "Ergotamine CONTRAINDICATED in FHM — same vasoconstrictive mechanism",
            "Carbonic anhydrase inhibitors — monitor renal stones (acetazolamide long-term)",
        ],
        "critical_pearls": [
            "INTERICTAL NYSTAGMUS is the clinical key — if nystagmus is always present, think EA2/CACNA1A",
            "SCA6 CAG repeat MUST be explicitly ordered — standard NGS panel does not detect it",
            "Acetazolamide response is so dramatic in EA2 it can serve as a diagnostic trial",
        ],
    },
    # -- ATP1A2 — FHM2 ----------------------------------------------------------------
    {
        "gene": "ATP1A2",
        "alt_name": (
            "ATP1A2 (ATP1A2-1020aa-1q23.2 / AD — FHM2-Familial-Hemiplegic-Migraine-Type-2 — "
            "HEMIPLEGIC-AURA-+CONFUSION-+PROLONGED-WEAKNESS-PATHOGNOMONIC — "
            "Astrocytic-Glutamate-K+-Clearance-Failure-Mechanism — "
            "Valproate+Topiramate+Preventives-First-Line)"
        ),
        "protein": (
            "ATP1A2 -- 1q23.2 AD -- ATP1A2-1020aa -- "
            "Na+/K+-ATPase-Alpha-2-Subunit-Astrocyte-Predominant-10-TM-Helix -- "
            "FHM2-Familial-Hemiplegic-Migraine-Type-2-OMIM-602481 -- "
            "Also-Allelic-Benign-Familial-Infantile-Seizures-BFIS-Like-Phenotype -- "
            "HEMIPLEGIC-AURA-PATHOGNOMONIC-Focal-Motor-Weakness-Aura-Hours -- "
            "CONFUSION-ALTERED-CONSCIOUSNESS-Aura-Distinguishes-FHM-from-Typical-Migraine -- "
            "PROLONGED-WEAKNESS-Hours-to-Days-Hemiplegia-PATHOGNOMONIC -- "
            "Astrocyte-Na/K-ATPase-Failure-Extracellular-K+-Glutamate-Accumulate-CSD -- "
            "Cortical-Spreading-Depression-CSD-Propagation-Wave-Mechanism -- "
            "Basilar-Type-Features-Common-Dysarthria-Diplopia-Tinnitus-Ataxia -- "
            "ICU-Admission-Required-Severe-FHM2-Attacks-Confusion-Coma-Seizures -- "
            "Valproate-FIRST-LINE-Prevention-Broad-Spectrum-AED-CSD-Suppressor -- "
            "Topiramate-Second-Line-Prevention -- "
            "Flunarizine-Ca2+-Blocker-Prevention -- "
            "Missense-LOF-Most-Common-Rare-Truncating -- "
            "1q23.2"
        ),
        "locus": "1q23.2",
        "protein_size": "1020 aa",
        "inheritance": (
            "AD (autosomal dominant); high penetrance for migraine, variable for hemiplegia severity; "
            "missense LOF variants most common; "
            "de novo rare; familial most cases; "
            "phenotypic spectrum from typical migraine with aura to severe hemiplegic with coma."
        ),
        "key_features": [
            "HEMIPLEGIC AURA — focal motor weakness during aura, distinguishes from typical migraine",
            "CONFUSION/ALTERED CONSCIOUSNESS during severe attacks — may mimic stroke",
            "PROLONGED WEAKNESS hours to days — some episodes require ICU observation",
            "Basilar features common: dysarthria, diplopia, tinnitus, ataxia aura",
            "CSD (cortical spreading depression) mechanism — astrocytic K+/glutamate clearance failure",
            "FHM2 phenotype overlaps EA-like: episodic neurological dysfunction, paroxysmal",
        ],
        "treatment": [
            "Valproate — FIRST-LINE prevention; CSD suppressor; effective broadly",
            "Topiramate — second-line prevention; carbonic anhydrase + Na-channel mechanism",
            "Flunarizine — calcium channel blocker, prevention",
            "Verapamil — alternative calcium channel blocker",
            "Acute attack: rest in dark quiet room; IV fluids if prolonged; avoid triptans",
            "Severe attack with confusion/coma: ICU, IV valproate, neuroimaging to exclude haemorrhage",
        ],
        "contraindications": [
            "TRIPTANS ABSOLUTELY CONTRAINDICATED in hemiplegic migraine — vasoconstriction risk",
            "ERGOTAMINE CONTRAINDICATED — same vasoconstriction mechanism",
            "Valproate teratogenicity — contraceptive counselling in women of reproductive age mandatory",
        ],
        "critical_pearls": [
            "TRIPTANS KILL IN HEMIPLEGIC MIGRAINE — always ask about motor weakness before prescribing triptan",
            "MRI DWI may show transient changes during severe FHM attack — does NOT mean infarct",
            "Family history: ask if relatives have migraines with weakness/confusion — often labelled 'complicated migraine'",
        ],
    },
    # -- ADCY5 — Nocturnal Dyskinesia -------------------------------------------------
    {
        "gene": "ADCY5",
        "alt_name": (
            "ADCY5 (ADCY5-1261aa-3q21.3 / AD-GoF — ADCY5-Related-Movement-Disorder-Nocturnal-Dyskinesia — "
            "NOCTURNAL-ATTACKS-FROM-NREM-SLEEP-PATHOGNOMONIC-EEG-NORMAL-Not-Epilepsy — "
            "CAFFEINE-ABSOLUTELY-CONTRAINDICATED-Dramatically-Worsens — "
            "Clonazepam+Acetazolamide-First-Line)"
        ),
        "protein": (
            "ADCY5 -- 3q21.3 AD-GoF -- ADCY5-1261aa -- "
            "Adenylyl-Cyclase-5-12-TM-Helix-cAMP-Synthesiser-Striatal-Predominant -- "
            "ADCY5-Related-Movement-Disorder-OMIM-615352-Familial-Dyskinesia-Facial-Myokymia -- "
            "Gain-of-Function-Mechanism-Excess-cAMP-Striatal-Medium-Spiny-Neurons -- "
            "NOCTURNAL-ATTACKS-PATHOGNOMONIC-From-NREM-Sleep-Not-from-Wakefulness -- "
            "EEG-NORMAL-During-Nocturnal-Episode-KEY-DDx-Epilepsy-Frontal-Lobe -- "
            "Chorea-Dystonia-Myoclonus-Face-Limbs-During-Attack-Mixed-Phenomenology -- "
            "Facial-Myokymia-Interictal-DISTINCTIVE-Rippling-Face-Muscles -- "
            "CAFFEINE-ABSOLUTELY-CI-cAMP-Phosphodiesterase-Inhibition-Worsens -- "
            "Clonazepam-GABA-A-First-Line-Reduces-Attack-Frequency -- "
            "Acetazolamide-Second-Line-Mechanism-Unknown-But-Effective -- "
            "Caffeine-Avoidance-Major-Non-Pharmacological-Intervention -- "
            "Arg418Trp-and-Arg418Gln-HOTSPOT-Variants-Most-Common -- "
            "3q21.3"
        ),
        "locus": "3q21.3",
        "protein_size": "1261 aa",
        "inheritance": (
            "AD (autosomal dominant); GoF mechanism; "
            "Arg418Trp and Arg418Gln hotspot missense variants in most cases; "
            "de novo in majority (~80%); familial in ~20%; "
            "variable expressivity; interictal facial myokymia often subtle."
        ),
        "key_features": [
            "NOCTURNAL ATTACKS FROM NREM SLEEP PATHOGNOMONIC — patient wakes from sleep mid-attack",
            "EEG NORMAL during attack — critical DDx frontal lobe epilepsy (FLE)",
            "Facial myokymia interictal — subtle rippling face muscles, DISTINCTIVE interictal sign",
            "CAFFEINE ABSOLUTELY CONTRAINDICATED — dramatically worsens attack frequency and severity",
            "Mixed phenomenology: chorea + dystonia + myoclonus during attack",
            "Arg418Trp/Arg418Gln hotspot variants — target these first in diagnostic testing",
        ],
        "treatment": [
            "CAFFEINE AVOIDANCE — most important immediate intervention",
            "Clonazepam 0.5-2 mg at bedtime — FIRST-LINE reduces nocturnal attacks",
            "Acetazolamide 250-500 mg — second-line or adjunct",
            "Sodium channel blockers NOT effective (contrast PRRT2)",
            "Deep brain stimulation (GPi) — reported for severe refractory cases",
            "Caffeine elimination diet — removes all dietary caffeine (coffee, tea, chocolate, cola, medications)",
        ],
        "contraindications": [
            "CAFFEINE ABSOLUTELY CONTRAINDICATED — all forms (coffee, tea, chocolate, energy drinks, medications)",
            "Carbamazepine NOT effective (diagnostic distinction from PKD/PRRT2)",
        ],
        "critical_pearls": [
            "WAKING FROM SLEEP + EEG NORMAL = ADCY5 until proven otherwise (not FLE)",
            "Video-polysomnography distinguishes: ADCY5 attacks have no EEG correlate; FLE has ictal discharge",
            "Facial myokymia at clinic visit — ask patient/family to video interictal face for subtle rippling",
        ],
    },
    # -- SLC6A5 — Hyperekplexia (GlyT2) ----------------------------------------------
    {
        "gene": "SLC6A5",
        "alt_name": (
            "SLC6A5 (SLC6A5-797aa-11p15.1 / AR — Hyperekplexia-GLYT2-Startle-Disease — "
            "EXAGGERATED-STARTLE-+NEONATAL-HYPERTONIA-+APNOEA-PATHOGNOMONIC — "
            "CLONAZEPAM-CURATIVE-Dramatically-Reduces-Hyperekplexia — "
            "NOSE-TIPPING-MANOEUVRE-Acute-Crisis-Termination)"
        ),
        "protein": (
            "SLC6A5 -- 11p15.1 AR -- SLC6A5-797aa -- "
            "GlyT2-Glycine-Transporter-2-12-TM-Helix-Presynaptic-Glycine-Reuptake -- "
            "Hyperekplexia-GLYT2-Type-OMIM-614618-Startle-Disease -- "
            "Also-Major-Hyperekplexia-GLRA1-AD-AR-Most-Common-GLRB-GPHN-Alleles -- "
            "EXAGGERATED-STARTLE-PATHOGNOMONIC-Auditory-Tactile-Visual-Startle -- "
            "NEONATAL-HYPERTONIA-Generalised-Stiffness-Birth-DISTINCTIVE -- "
            "APNOEA-PATHOGNOMONIC-Post-Startle-Breath-Holding-Episodes-Dangerous -- "
            "NOSE-TIPPING-MANOEUVRE-ACUTE-CRISIS-Flexion-Head-Knees-Terminates-Episode -- "
            "Clonazepam-CURATIVE-GABA-A-Enhancement-Compensates-Glycine-Loss -- "
            "SIDS-Risk-Cardiorespiratory-Monitoring-Mandatory-First-Year -- "
            "Frog-Position-Sleeping-Prone-on-Parents-Chest-Neonatal-Period -- "
            "Startle-Episodes-Distinguish-from-Epilepsy-No-EEG-Correlate -- "
            "LOF-Presynaptic-GlyT2-Reduced-Glycine-Recycling-Inhibitory-Synapse-Depleted -- "
            "11p15.1"
        ),
        "locus": "11p15.1",
        "protein_size": "797 aa",
        "inheritance": (
            "AR (autosomal recessive) for SLC6A5/GLYT2 hyperekplexia; "
            "biallelic LOF variants; consanguinity enriches; "
            "NOTE: GLRA1 (glycine receptor alpha-1) is the most common hyperekplexia gene — AD and AR; "
            "genetic panel should include GLRA1, GLRB, SLC6A5, GPHN, SLC6A9; "
            "clinical phenotype identical across genes — genotype determines inheritance."
        ),
        "key_features": [
            "EXAGGERATED STARTLE PATHOGNOMONIC — auditory/tactile/visual stimuli elicit excessive whole-body startle",
            "NEONATAL HYPERTONIA — generalised stiffness at birth, may mimic spasticity",
            "POST-STARTLE APNOEA — breath-holding after startle episodes; SIDS risk — monitoring mandatory",
            "NOSE-TIPPING MANOEUVRE — firm pressure on nose + flex head to knees terminates acute episode",
            "Clonazepam CURATIVE — complete resolution of hyperekplexia in most cases",
            "NO EEG correlate during startle episodes — critical DDx epilepsy",
        ],
        "treatment": [
            "Clonazepam — FIRST-LINE, CURATIVE, 0.1 mg/kg/day in divided doses; dramatic response",
            "Neonatal apnoea monitoring — cardiorespiratory monitor MANDATORY first year",
            "Nose-tipping manoeuvre — all caregivers MUST be trained",
            "Prone 'frog position' during acute neonatal period — reduces startle episodes",
            "Physiotherapy — hypertonia management; walking delayed but usually achieved",
            "Phenobarbital — alternative/adjunct in neonates if clonazepam not immediately available",
        ],
        "contraindications": [
            "DO NOT apply painful stimuli to terminate episode — worsens startle response",
            "Avoid unnecessary startling stimuli in environment — door slams, unexpected touch",
        ],
        "critical_pearls": [
            "NOSE-TIPPING IS LIFE-SAVING — every family must demonstrate this manoeuvre before discharge",
            "Apnoea monitoring saves lives — install before discharge from NICU/neonatal ward",
            "Clonazepam response so dramatic it serves as diagnostic trial — typically within days",
        ],
    },
]


def _make_patients(gene_dict, seed):
    rng = random.Random(seed)
    gene = gene_dict["gene"]
    locus = gene_dict["locus"]
    inh = gene_dict["inheritance"]
    n = 40

    ad = "AR" not in inh.split(";")[0].upper()[:10]
    # age distribution varies by condition
    ages = [rng.randint(2, 45) for _ in range(n)]
    # attack frequency per month
    attack_freqs = [rng.randint(0, 30) for _ in range(n)]
    # treatment response
    treated = [rng.random() < 0.85 for _ in range(n)]
    # alive (almost all — paroxysmal disorders non-fatal generally)
    alive = [rng.random() > 0.01 for _ in range(n)]

    patients = []
    for i in range(n):
        patients.append({
            "id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "seed": seed,
            "age": ages[i],
            "sex": "F" if rng.random() < 0.5 else "M",
            "alive": alive[i],
            "treated": treated[i],
            "attacks_per_month": attack_freqs[i],
            "inheritance": "AD" if ad else "AR",
            "locus": locus,
        })
    return patients


def overview():
    all_patients = []
    for i, gd in enumerate(PAROXYSMAL_GENES):
        all_patients += _make_patients(gd, SEED_BASE + i)

    total = len(all_patients)
    alive = sum(1 for p in all_patients if p["alive"])
    female = sum(1 for p in all_patients if p["sex"] == "F")
    treated = sum(1 for p in all_patients if p["treated"])
    avg_age = round(sum(p["age"] for p in all_patients) / total, 1)
    avg_attacks = round(sum(p["attacks_per_month"] for p in all_patients) / total, 1)

    gene_summaries = {}
    for i, gd in enumerate(PAROXYSMAL_GENES):
        pts = _make_patients(gd, SEED_BASE + i)
        gene_summaries[gd["gene"]] = {
            "gene": gd["gene"],
            "alt_name": gd["alt_name"],
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "n_patients": len(pts),
            "alive_pct": round(100 * sum(1 for p in pts if p["alive"]) / len(pts), 1),
            "female_pct": round(100 * sum(1 for p in pts if p["sex"] == "F") / len(pts), 1),
            "treated_pct": round(100 * sum(1 for p in pts if p["treated"]) / len(pts), 1),
            "avg_age": round(sum(p["age"] for p in pts) / len(pts), 1),
            "avg_attacks_per_month": round(sum(p["attacks_per_month"] for p in pts) / len(pts), 1),
            "key_features": gd["key_features"],
            "treatment": gd["treatment"],
            "critical_pearls": gd["critical_pearls"],
        }

    return {
        "atlas": "Hereditary-Paroxysmal-Movement-Disorder-Atlas",
        "subtitle": "Complete 8-Gene Paroxysmal Movement Disorder Reference",
        "genes": [g["gene"] for g in PAROXYSMAL_GENES],
        "seeds": f"{SEED_BASE}-{SEED_BASE+7}",
        "total_patients": total,
        "alive_pct": round(100 * alive / total, 1),
        "female_pct": round(100 * female / total, 1),
        "treated_pct": round(100 * treated / total, 1),
        "avg_age": avg_age,
        "avg_attacks_per_month": avg_attacks,
        "gene_summaries": gene_summaries,
        "clinical_axioms": [
            "PRRT2/PKD: movement-triggered + <1 min + no LOC = carbamazepine at low dose, >90% effective",
            "SLC2A1/GLUT1D: low CSF glucose PATHOGNOMONIC — lumbar puncture is the diagnostic test",
            "PNKD: caffeine + alcohol triggers — avoidance alone dramatically reduces attacks",
            "KCNA1/EA1: interictal myokymia on EMG PATHOGNOMONIC — acetazolamide/carbamazepine",
            "CACNA1A/EA2: interictal nystagmus PATHOGNOMONIC — acetazolamide >80% reduction",
            "ATP1A2/FHM2: hemiplegic migraine — TRIPTANS ABSOLUTELY CONTRAINDICATED",
            "ADCY5: nocturnal NREM attacks + EEG normal — caffeine ABSOLUTELY CI, clonazepam first",
            "SLC6A5/Hyperekplexia: nose-tipping LIFE-SAVING — clonazepam curative",
        ],
    }


def breakdown():
    result = {}
    for i, gd in enumerate(PAROXYSMAL_GENES):
        pts = _make_patients(gd, SEED_BASE + i)
        n = len(pts)
        result[gd["gene"]] = {
            "gene": gd["gene"],
            "alt_name": gd["alt_name"],
            "protein": gd["protein"],
            "locus": gd["locus"],
            "protein_size": gd["protein_size"],
            "inheritance": gd["inheritance"],
            "n_patients": n,
            "alive_pct": round(100 * sum(1 for p in pts if p["alive"]) / n, 1),
            "female_pct": round(100 * sum(1 for p in pts if p["sex"] == "F") / n, 1),
            "treated_pct": round(100 * sum(1 for p in pts if p["treated"]) / n, 1),
            "avg_age": round(sum(p["age"] for p in pts) / n, 1),
            "avg_attacks_per_month": round(sum(p["attacks_per_month"] for p in pts) / n, 1),
            "key_features": gd["key_features"],
            "treatment": gd["treatment"],
            "contraindications": gd["contraindications"],
            "critical_pearls": gd["critical_pearls"],
            "seed": SEED_BASE + i,
        }
    return result


def definitions():
    return {
        "paroxysmal_disorder_classification": {
            "PKD": "Paroxysmal Kinesigenic Dyskinesia — PRRT2; movement-triggered; <1 min; carbamazepine curative",
            "PNKD": "Paroxysmal Non-Kinesigenic Dyskinesia — PNKD; caffeine/alcohol trigger; hours; clonazepam",
            "PED": "Paroxysmal Exercise-Induced Dyskinesia — SLC2A1; prolonged exercise; low CSF glucose",
            "EA1": "Episodic Ataxia Type 1 — KCNA1; startle triggered; seconds; myokymia; carbamazepine",
            "EA2": "Episodic Ataxia Type 2 — CACNA1A; hours; interictal nystagmus; acetazolamide",
            "SCA6": "Spinocerebellar Ataxia 6 — CACNA1A CAG repeat; progressive; onset ~50yr",
            "FHM1": "Familial Hemiplegic Migraine 1 — CACNA1A GOF; hemiplegic aura; triptans CI",
            "FHM2": "Familial Hemiplegic Migraine 2 — ATP1A2; astrocytic K+/glu failure; triptans CI",
            "ADCY5-RMD": "ADCY5-Related Movement Disorder — nocturnal NREM attacks; EEG normal; caffeine CI",
            "Hyperekplexia": "Startle Disease — GLRA1/SLC6A5/GLRB; exaggerated startle; clonazepam curative",
            "GLUT1D": "GLUT1 Deficiency Syndrome — SLC2A1; low CSF glucose; ketogenic diet",
            "BFIS": "Benign Familial Infantile Seizures — PRRT2; infantile seizures same gene as PKD",
            "ICCA": "Infantile Convulsions with Choreoathetosis — PRRT2; overlap BFIS+PKD phenotype",
        },
        "key_pharmacology": {
            "Carbamazepine_PKD": "Carbamazepine 100-200 mg/day — FIRST-LINE PKD (PRRT2), >90% effective at very low dose",
            "Acetazolamide_EA2": "Acetazolamide 250-1000 mg/day — FIRST-LINE EA2 (CACNA1A), >80% attack reduction",
            "Clonazepam_PNKD": "Clonazepam GABA-A — PNKD partially effective; Hyperekplexia CURATIVE",
            "KD_GLUT1D": "Ketogenic Diet — FIRST-LINE GLUT1D, seizures abate within weeks of start",
            "Valproate_FHM": "Valproate — FIRST-LINE FHM2 prevention; CSD suppressor; teratogenic",
            "4AP_EA2": "4-Aminopyridine — Kv channel blocker, effective EA2 alternative to acetazolamide",
            "Flunarizine_EA2": "Flunarizine — calcium channel blocker, EA2 prevention alternative",
        },
        "critical_contraindications": {
            "Triptans_FHM": "TRIPTANS ABSOLUTELY CONTRAINDICATED in FHM1 and FHM2 — hemiplegic migraine vasoconstrictive risk",
            "Caffeine_ADCY5": "CAFFEINE ABSOLUTELY CONTRAINDICATED in ADCY5 — dramatically worsens dyskinesia",
            "Caffeine_PNKD": "CAFFEINE trigger in PNKD — avoidance reduces attacks dramatically",
            "Valproate_GLUT1D": "VALPROATE potentially harmful in GLUT1D — inhibits fatty acid oxidation",
            "Fasting_GLUT1D": "FASTING ABSOLUTELY CONTRAINDICATED in GLUT1D — precipitates crisis",
            "4AP_EA1": "4-AMINOPYRIDINE WORSENS EA1 (KCNA1) — reduces Kv channel activity further",
            "CBZ_PNKD": "CARBAMAZEPINE INEFFECTIVE in PNKD — failure of CBZ diagnostic: favour PNKD over PKD",
            "Triptans_hemiplegic": "NEVER give triptans without asking about motor weakness aura — hemiplegic migraine",
        },
        "pathognomonic_signs": {
            "PRRT2_movement_trigger": "Movement-triggered attack <1 min with warning aura + NO LOC — PKD/PRRT2 pathognomonic",
            "SLC2A1_CSF_glucose": "CSF glucose <45 mg/dL + CSF:serum ratio <0.6 — GLUT1D PATHOGNOMONIC",
            "PNKD_caffeine_alcohol": "Caffeine + alcohol triggering dyskinesia — PNKD PATHOGNOMONIC",
            "KCNA1_myokymia": "Interictal myokymia on EMG + startle-triggered brief attacks — EA1/KCNA1 PATHOGNOMONIC",
            "CACNA1A_interictal_nystagmus": "Always-present interictal nystagmus + hours-duration attacks — EA2/CACNA1A PATHOGNOMONIC",
            "ATP1A2_hemiplegic_aura": "Hemiplegic aura + confusion + prolonged weakness in migraine — FHM2/ATP1A2 PATHOGNOMONIC",
            "ADCY5_nocturnal_NREM": "Attacks from NREM sleep + EEG normal — ADCY5 PATHOGNOMONIC",
            "SLC6A5_startle_apnoea": "Exaggerated startle + neonatal hypertonia + post-startle apnoea — Hyperekplexia PATHOGNOMONIC",
        },
        "ddx_table": {
            "PKD_vs_PNKD": "PKD: movement-triggered, seconds, CBZ effective; PNKD: caffeine/alcohol, hours, CBZ ineffective",
            "PKD_vs_PED": "PKD: any abrupt movement, <1 min; PED: prolonged exercise >5 min, longer duration",
            "EA1_vs_EA2": "EA1: seconds, myokymia, startle; EA2: hours, nausea, interictal nystagmus, acetazolamide",
            "ADCY5_vs_FLE": "ADCY5: EEG normal, NREM, caffeine worsens; FLE: ictal EEG discharge during episode",
            "Hyperekplexia_vs_epilepsy": "Hyperekplexia: startle-triggered, EEG normal, nose-tip terminates; epilepsy: spontaneous EEG correlate",
            "FHM1_vs_FHM2": "FHM1 CACNA1A (GOF): progressive cerebellar, SCA6 family; FHM2 ATP1A2: astrocytic mechanism",
            "GLUT1D_vs_mito": "GLUT1D: low CSF glucose, normal lactate; Mito: elevated lactate, normal CSF glucose",
        },
        "emergency_protocols": {
            "Hyperekplexia_apnoea": "NOSE-TIP MANOEUVRE: firm pressure on nose + flex head to knees — terminates episode immediately",
            "Hyperekplexia_neonatal": "All neonates with hyperekplexia: cardiorespiratory monitoring + clonazepam before discharge",
            "FHM_severe_attack": "Severe FHM with confusion/coma: IV fluids, IV valproate, neuroimaging, ICU; NO triptans",
            "GLUT1D_perioperative": "GLUT1D surgery: IV glucose infusion perioperatively — NO fasting protocol",
            "ADCY5_crisis": "ADCY5 status: clonazepam + remove all caffeine including IV medications",
        },
        "glossary": {
            "Paroxysmal": "Abrupt onset episodic neurological dysfunction with complete return to baseline",
            "Kinesigenic": "Triggered by sudden voluntary movement — hallmark of PKD/PRRT2",
            "Non-kinesigenic": "NOT triggered by movement — PNKD triggered by caffeine/alcohol",
            "Exercise-induced": "Triggered by prolonged (>5 min) sustained exercise — PED/GLUT1D",
            "Myokymia": "Spontaneous continuous rippling of muscle fibres — interictal sign EA1/KCNA1",
            "CSD": "Cortical spreading depression — slow wave of neuronal depolarisation underlying FHM/migraine aura",
            "Hyperekplexia": "Exaggerated startle response due to glycinergic inhibition failure",
            "Hypoglycorrhachia": "Low CSF glucose — below 45 mg/dL absolute or <0.6 ratio to serum",
            "Ketogenic_diet": "High-fat very-low-carbohydrate diet; provides ketone bodies as alternative brain fuel",
            "4-AP": "4-Aminopyridine — potassium channel blocker; effective in EA2; worsens EA1",
            "Acetazolamide": "Carbonic anhydrase inhibitor — first-line EA2; mechanism in cerebellar ataxia unclear",
            "Nos_tipping": "Nose-tipping manoeuvre — life-saving Vigevano manoeuvre for hyperekplexia acute episodes",
            "NREM": "Non-rapid eye movement sleep — ADCY5 attacks arise specifically from NREM",
            "GoF": "Gain-of-function variant — ADCY5; excess cAMP production in striatum",
            "BDC": "Benign Dystonia with Choreoathetosis — older name for PNKD",
            "Penetrance": "Proportion of genotype carriers who develop clinical phenotype",
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
        print(f"  {gene}: {data['n_patients']} patients, alive={data['alive_pct']}%")
    print("\n=== DEFINITIONS keys ===")
    defs = definitions()
    print(list(defs.keys()))
