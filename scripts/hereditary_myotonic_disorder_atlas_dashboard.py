#!/usr/bin/env python3
"""Hereditary-Myotonic-Disorder-Atlas — Complete 8-Gene Myotonic Disorder & Periodic Paralysis Atlas
(DM1-DMPK · DM2-CNBP · Myotonia-Congenita-CLCN1 · Paramyotonia-SCN4A ·
 HypoPP1-CACNA1S · Andersen-Tawil-KCNJ2 · MHS/Core-Myopathy-RYR1 · Brody-ATP2A1).

DMPK    (Dystrophia Myotonica Protein Kinase; 639 aa; 19q13.32; AD;
         Myotonic Dystrophy Type 1 (DM1/Steinert disease) — CTG trinucleotide repeat >50;
         ANAESTHESIA EXTREME RISK — depolarising NMB (succinylcholine) ABSOLUTELY CI,
         volatile anaesthetics EXTREME CAUTION, malignant hyperthermia-like crises;
         seed SEED_BASE+0).
CNBP    (CCHC-Type Zinc Finger Nucleic Acid Binding Protein; 347 aa; 3q21.3; AD;
         Myotonic Dystrophy Type 2 (DM2/PROMM) — CCTG tetranucleotide repeat in intron 1;
         PROXIMAL WEAKNESS (not distal as in DM1) PATHOGNOMONIC DDx;
         MUSCLE PAIN/STIFFNESS prominent — often misdiagnosed as fibromyalgia;
         seed SEED_BASE+1).
CLCN1   (Chloride Channel 1; 988 aa; 7q34; AD/AR;
         Myotonia Congenita — Thomsen (AD) and Becker (AR) types;
         WARM-UP PHENOMENON — stiffness improves with repeated contractions PATHOGNOMONIC;
         NO WEAKNESS, NO SYSTEMIC FEATURES — pure myotonia;
         seed SEED_BASE+2).
SCN4A   (Nav1.4 Voltage-Gated Na+ Channel; 1836 aa; 17q23.3; AD;
         Paramyotonia Congenita (PMC) + Hyperkalemic Periodic Paralysis Type 2 (HyperPP2);
         COLD WORSENS MYOTONIA PATHOGNOMONIC — paradoxical myotonia (warm-up fails);
         provocative test: exercise + cold water immersion;
         seed SEED_BASE+3).
CACNA1S (Cav1.1 L-type Voltage-Gated Ca2+ Channel; 1873 aa; 1q32.1; AD;
         Hypokalemic Periodic Paralysis Type 1 (HypoPP1) — most common hereditary periodic paralysis;
         CARBOHYDRATE + REST AFTER EXERCISE triggers paralysis PATHOGNOMONIC;
         ACETAZOLAMIDE FIRST-LINE plus KCl supplementation;
         seed SEED_BASE+4).
KCNJ2   (Kir2.1 Inward Rectifier K+ Channel; 427 aa; 17q24.3; AD;
         Andersen-Tawil Syndrome (ATS/LQT7) — TRIAD: episodic flaccid paralysis +
         cardiac arrhythmia (LQT/VT) + dysmorphic features PATHOGNOMONIC;
         FLECAINIDE for arrhythmia; ICD consideration mandatory;
         seed SEED_BASE+5).
RYR1    (Ryanodine Receptor 1; 5038 aa; 19q13.2; AR/AD;
         Malignant Hyperthermia Susceptibility (MHS1) + Central Core Disease + Multi-minicore;
         SUCCINYLCHOLINE + ALL VOLATILE ANAESTHETICS ABSOLUTELY CONTRAINDICATED;
         DANTROLENE — life-saving antidote, must be available in all theatres;
         seed SEED_BASE+6).
ATP2A1  (SERCA1 Sarco/Endoplasmic Reticulum Ca2+-ATPase 1; 994 aa; 16p11.2; AR;
         Brody Myopathy — exercise-induced impaired muscle relaxation;
         SILENT MYOTONIA — EMG SILENT (no electrical activity) despite mechanical stiffness PATHOGNOMONIC;
         clinically important DDx: CLCN1/SCN4A show EMG myotonic discharges;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2174-2181).
"""

import random

SEED_BASE = 2174

MYOTONIC_GENES = [
    # -- DMPK — Myotonic Dystrophy Type 1 (DM1/Steinert) ----------------------------
    {
        "gene": "DMPK",
        "alt_name": (
            "DMPK (DMPK-639aa-19q13.32 / AD — DM1-Steinert-Myotonic-Dystrophy-Type-1 — "
            "CTG-REPEAT->50-PATHOGNOMONIC-Most-Common-Adult-Muscular-Dystrophy — "
            "ANAESTHESIA-EXTREME-RISK-Succinylcholine-ABSOLUTELY-CI-Volatile-EXTREME-CAUTION — "
            "PACEMAKER-MANDATORY-Annual-Holter-Cardiac-Conduction-Disease)"
        ),
        "protein": (
            "DMPK -- 19q13.32 AD -- DMPK-639aa -- "
            "Dystrophia-Myotonica-Protein-Kinase-Ser-Thr-Kinase-Myosin-Heavy-Chain-Interaction -- "
            "DM1-Myotonic-Dystrophy-Type-1-OMIM-160900-Steinert-Disease -- "
            "CTG-Repeat-DMPK-3prime-UTR-Normal-<37-Premutation-38-49-Pathogenic->50 -- "
            "Anticipation-MATERNAL-Bias-Congenital-DM1-CTG->1000-Maternal-Only -- "
            "Congenital-DM1-Neonatal-Hypotonia-Respiratory-Failure-Clubfoot-ID -- "
            "Distal-Weakness-Ptosis-Temporalis-Wasting-Facial-Appearance-CLASSIC -- "
            "Myotonia-Hand-Grip-Release-Delayed-Percussion-Myotonia-CLINICAL -- "
            "Cardiac-Conduction-Defects-All-Patients-Annual-Holter-Pacemaker-If-HV>70ms -- "
            "Respiratory-FVC-Annual-NIV-When-FVC<50pct-Predictive-Mortality -- "
            "Cataracts-Posterior-Subcapsular-Slit-Lamp-Annual -- "
            "Diabetes-Mellitus-Insulin-Resistance-HbA1c-Annual -- "
            "Daytime-Somnolence-Modafinil-Methylphenidate -- "
            "Cognitive-Impairment-Executive-Function-Neuropsychology -- "
            "SUCCINYLCHOLINE-ABSOLUTELY-CI-Prolonged-Paralysis-Myotonic-Crisis -- "
            "VOLATILE-ANAESTHETICS-EXTREME-CAUTION-MH-Like-Crisis-Reported -- "
            "SUXAMETHONIUM-ABSOLUTE-CI-Emergency-Surgeons-Must-Know -- "
            "Mexiletine-Myotonia-Symptomatic-Not-Disease-Modifying -- "
            "19q13.32"
        ),
        "locus": "19q13.32",
        "protein_size": "639 aa",
        "inheritance": (
            "AD (autosomal dominant); CTG repeat in DMPK 3'UTR; anticipation (repeats increase each generation); "
            "maternal transmission bias for congenital DM1 (CTG >1000); "
            "normal <37; premutation 38-49; mildly affected 50-150; "
            "classic DM1 150-1000; congenital >1000; "
            "triplet repeat primed PCR + Southern blot for large expansions."
        ),
        "key_features": [
            "Distal weakness (ankle dorsiflexors, finger extensors) + myotonia + frontal baldness + cataracts",
            "Facial myopathy: ptosis, temporalis/masseter wasting, 'hatchet face' appearance",
            "Cardiac conduction: AV block + VT — annual Holter, pacemaker if HV interval >70 ms",
            "Respiratory: FVC annual; NIV when FVC <50% — major cause of DM1 mortality",
            "Myotonia: grip-release lag, percussion myotonia; worsens in cold; mexiletine symptomatic",
            "CTG anticipation: earlier onset and more severe in each generation; maternal congenital form",
        ],
        "treatment": [
            "Mexiletine 150-200 mg TID — symptomatic myotonia (level A evidence), NOT disease-modifying",
            "Annual Holter + ECG — pacemaker if HV >70 ms; ICD if LVEF <45%",
            "Annual FVC — NIV when FVC <50%; sleep study for central apnoea",
            "Annual slit-lamp cataracts — phacoemulsification when vision impaired",
            "Modafinil/methylphenidate — daytime somnolence (EDS common)",
            "Annual HbA1c — insulin sensitiser if T2DM develops",
        ],
        "contraindications": [
            "SUCCINYLCHOLINE ABSOLUTELY CONTRAINDICATED — prolonged depolarisation block, myotonic crisis",
            "VOLATILE ANAESTHETICS EXTREME CAUTION — MH-like events reported, TIVA preferred",
            "NEOSTIGMINE CAUTION — can trigger myotonic crisis; sugammadex preferred for reversal",
            "HIGH-DOSE CORTICOSTEROIDS — respiratory muscle weakness may worsen acutely",
            "Avoid ALL respiratory depressants — benzodiazepines, opioids with extreme caution",
            "CLASS IA/III ANTIARRHYTHMICS CAUTION — myotonic effect; mexiletine IB preferred",
        ],
        "critical_pearls": [
            "ANAESTHESIA ALERT CARD mandatory — carry documentation to every hospital visit",
            "CTG repeat size predicts severity — congenital DM1 almost always maternal, CTG >1000",
            "Cardiac death is #1 cause of mortality — never miss annual Holter",
            "Genetic counselling mandatory — anticipation means children more severely affected",
            "Congenital DM1: maternal CTG should be tested in all neonatal hypotonia with weak cry",
        ],
    },
    # -- CNBP — Myotonic Dystrophy Type 2 (DM2/PROMM) --------------------------------
    {
        "gene": "CNBP",
        "alt_name": (
            "CNBP (CNBP-347aa-3q21.3 / AD — DM2-PROMM-Myotonic-Dystrophy-Type-2 — "
            "CCTG-TETRANUCLEOTIDE-REPEAT-Intron-1-No-Congenital-Form-NO-Anticipation-DM2 — "
            "PROXIMAL-WEAKNESS-DDx-DM1-Distal — "
            "MUSCLE-PAIN-Myalgia-Stiffness-Often-Misdiagnosed-Fibromyalgia)"
        ),
        "protein": (
            "CNBP -- 3q21.3 AD -- CNBP-347aa -- "
            "CCHC-Type-Zinc-Finger-Nucleic-Acid-Binding-Protein-7-Zinc-Finger-Motifs-RNA-Binding -- "
            "DM2-Myotonic-Dystrophy-Type-2-OMIM-602668-PROMM-Proximal-Myotonic-Myopathy -- "
            "CCTG-Repeat-Intron-1-CNBP-Normal-<27-Pathogenic->75-Typical->5000 -- "
            "No-Congenital-Form-KEY-DDx-DM1-Maternal-Congenital-Never-DM2 -- "
            "No-Anticipation-Clinically-Significant-DDx-DM1-Each-Generation-Worse -- "
            "Proximal-Weakness-Hip-Flexors-Neck-Flexors-Not-Distal-CRITICAL-DDx-DM1 -- "
            "Myalgia-Muscle-Pain-Prominent-Often-Fibromyalgia-Misdiagnosis-Years -- "
            "Myotonia-Milder-Than-DM1-Hand-Grip-Percussion-Myotonia-Present -- "
            "Cardiac-Conduction-Same-Surveillance-As-DM1-Annual-Holter-Required -- "
            "Cataracts-Posterior-Subcapsular-Same-As-DM1-Mandatory-Slit-Lamp -- "
            "Insulin-Resistance-T2DM-Same-Frequency-As-DM1 -- "
            "CCTG-REPEAT-PRIMED-PCR-Followed-Southern-Blot-Confirmation -- "
            "3q21.3"
        ),
        "locus": "3q21.3",
        "protein_size": "347 aa",
        "inheritance": (
            "AD (autosomal dominant); CCTG tetranucleotide repeat expansion in CNBP intron 1; "
            "no clinically significant anticipation (contrast with DM1); "
            "NO congenital form — a neonate with DM phenotype has DM1 not DM2; "
            "repeat size >75 pathogenic; typical expansions >5000 repeats; "
            "Southern blot required for confirmation after repeat-primed PCR."
        ),
        "key_features": [
            "PROXIMAL weakness (hip, knee, neck flexors) — NOT distal, KEY DDx from DM1",
            "Muscle pain and stiffness prominent — frequently misdiagnosed as fibromyalgia for years",
            "Milder myotonia than DM1: grip-release lag, percussion myotonia, rarely disabling",
            "NO congenital form — any congenital myotonic dystrophy = DM1 until proven otherwise",
            "Cardiac surveillance mandatory: same as DM1 (Holter, pacemaker threshold same)",
            "Cataracts: same posterior subcapsular type as DM1; annual slit-lamp",
        ],
        "treatment": [
            "Mexiletine — symptomatic myotonia if disabling; similar evidence base to DM1",
            "Annual Holter + ECG — same surveillance as DM1; pacemaker threshold same",
            "Annual FVC — respiratory involvement less severe than DM1 but still monitor",
            "Pain management: nonsteroidal + physiotherapy + stretching for myalgia",
            "Annual HbA1c — insulin resistance/T2DM surveillance same as DM1",
            "Annual slit-lamp — cataracts similar frequency",
        ],
        "contraindications": [
            "SUCCINYLCHOLINE ABSOLUTELY CONTRAINDICATED — same risk as DM1",
            "VOLATILE ANAESTHETICS EXTREME CAUTION — same anaesthetic protocol as DM1",
            "AVOID labelling as fibromyalgia without genetic testing in proximal myopathy with pain",
            "CLASS IA/III ANTIARRHYTHMICS CAUTION — same as DM1",
        ],
        "critical_pearls": [
            "DM2 pain is real muscle disease — NOT psychosomatic fibromyalgia",
            "NO congenital DM2 — if neonatal, order DM1 CTG repeat first",
            "CK may be mildly elevated — not as high as LGMD; EMG myotonic discharges present",
            "Anaesthetic alert card mandatory — same risks as DM1",
            "Repeat-primed PCR + Southern blot: standard PCR may miss large expansions",
        ],
    },
    # -- CLCN1 — Myotonia Congenita (Thomsen AD / Becker AR) --------------------------
    {
        "gene": "CLCN1",
        "alt_name": (
            "CLCN1 (CLCN1-988aa-7q34 / AD-AR — Myotonia-Congenita-Thomsen-AD-Becker-AR — "
            "WARM-UP-PHENOMENON-PATHOGNOMONIC-Stiffness-Improves-Repeated-Contractions — "
            "NO-WEAKNESS-NO-SYSTEMIC-Features-Pure-Myotonia — "
            "Mexiletine-FIRST-LINE-Mexiletine-Level-A)"
        ),
        "protein": (
            "CLCN1 -- 7q34 AD/AR -- CLCN1-988aa -- "
            "Voltage-Gated-Chloride-Channel-1-Skeletal-Muscle-Homodimer-18-TM-Helices -- "
            "Thomsen-Disease-OMIM-160800-AD-LOF-Haploinsufficiency-Dominant-Negative -- "
            "Becker-Disease-OMIM-255700-AR-Biallelic-LOF-More-Severe-Transient-Weakness-After-Stiffness -- "
            "Warm-Up-Phenomenon-PATHOGNOMONIC-Stiffness-Worsens-Rest-Improves-10-15-Contractions -- "
            "No-Systemic-Features-Pure-Skeletal-Muscle-Disease-Normal-CK-or-Mildly-Elevated -- "
            "No-Cardiac-Involvement-KEY-DDx-DM1-DM2-No-Annual-Holter-Needed -- "
            "No-Cataracts-No-Diabetes-No-Cognitive-Decline -- "
            "EMG-Myotonic-Discharges-Dive-Bomber-Sound-CHARACTERISTIC -- "
            "Cold-Worsens-NOT-Paradoxical-Worsens-But-Warm-Up-Still-Works -- "
            "Mexiletine-FIRST-LINE-Level-A-RCT-Evidence -- "
            "Lamotrigine-Alternative-If-Mexiletine-Intolerant -- "
            "7q34"
        ),
        "locus": "7q34",
        "protein_size": "988 aa",
        "inheritance": (
            "AD (Thomsen) — haploinsufficiency + dominant-negative; "
            "AR (Becker) — biallelic LOF; more severe with transient weakness after prolonged stiffness; "
            "heterozygous carriers of Becker variants may have mild myotonia; "
            "penetrance high in both types; ClinVar has >250 pathogenic CLCN1 variants; "
            "targeted sequencing + CNV analysis (rare deletions)."
        ),
        "key_features": [
            "WARM-UP PHENOMENON: stiffness severe at rest, dramatically improves after 10-15 contractions — PATHOGNOMONIC",
            "NO systemic features: normal cardiac, no cataracts, no cognitive decline (DDx DM1/DM2)",
            "Becker type: brief transient weakness after prolonged stiffness — NOT in Thomsen",
            "EMG: dive-bomber myotonic discharges on needle EMG — diagnostic",
            "Cold worsens myotonia but warm-up phenomenon still present (DDx SCN4A/PMC)",
            "CK: normal or mildly elevated — not a marker of disease severity",
        ],
        "treatment": [
            "Mexiletine 150-200 mg TID — FIRST-LINE, level A RCT evidence (NCT01287156)",
            "Lamotrigine 150-300 mg/day — second-line if mexiletine intolerated",
            "Carbamazepine — third-line alternative; less evidence than mexiletine",
            "Physiotherapy — warm environment, avoid cold, gradual warm-up routines",
            "Lifestyle: early morning stiffness worst — allow warm-up time before activity",
        ],
        "contraindications": [
            "ACETAZOLAMIDE — generally ineffective in CLCN1; avoid (contrast with SCN4A/CACNA1S)",
            "CLASS IA ANTIARRHYTHMICS (quinidine) — proarrhythmic at doses needed for myotonia",
            "No anaesthetic restriction specific to CLCN1 (no depolarising NMB issue unlike DMPK)",
            "Avoid cold environments — worsens stiffness, plan activities accordingly",
        ],
        "critical_pearls": [
            "Pure myotonia + warm-up + no weakness + no systemic = CLCN1 until proven otherwise",
            "DO NOT mislabel as DM1 — no cardiac/respiratory surveillance burden in CLCN1",
            "Becker type more severe — transient weakness post-stiffness can suggest NMJ disease",
            "Mexiletine titration: start 150 mg TID, ECG at baseline (QTc prolongation rare)",
            "Children with myotonia: exclude DM1 CTG expansion first before CLCN1 panel",
        ],
    },
    # -- SCN4A — Paramyotonia Congenita + Hyperkalemic Periodic Paralysis Type 2 ------
    {
        "gene": "SCN4A",
        "alt_name": (
            "SCN4A (SCN4A-1836aa-17q23.3 / AD — Paramyotonia-Congenita-PMC-HyperPP2 — "
            "COLD-WORSENS-PARADOXICAL-MYOTONIA-Warm-Up-FAILS-PATHOGNOMONIC-DDx-CLCN1 — "
            "POTASSIUM-Triggers-Paralysis-Provocative-Test-Oral-KCl — "
            "Mexiletine-PMC-Acetazolamide-HyperPP2)"
        ),
        "protein": (
            "SCN4A -- 17q23.3 AD -- SCN4A-1836aa -- "
            "Nav1.4-Voltage-Gated-Na+-Channel-Alpha-Subunit-Skeletal-Muscle-4-Domain-24-TM-Helices -- "
            "Paramyotonia-Congenita-PMC-OMIM-168300-Eulenburg-Paramyotonia -- "
            "HyperPP2-Hyperkalemic-Periodic-Paralysis-Type-2-OMIM-170500 -- "
            "HypoPP2-Hypokalemic-Periodic-Paralysis-Type-2-OMIM-613345 -- "
            "Sodium-Channel-Myotonia-SCM-OMIM-608390 -- "
            "Paradoxical-Myotonia-PMC-COLD-WORSENS-Warm-Up-FAILS-PATHOGNOMONIC-DDx-CLCN1 -- "
            "Cold-Immersion-Test-5min-Cold-Water-Worsening-Myotonia-Diagnostic-PMC -- "
            "Potassium-Triggers-Paralysis-HyperPP-Oral-KCl-1mEq/kg-Provocative -- "
            "Sodium-Channel-Gain-Function-Slow-Inactivation-Defect -- "
            "p.R1448H-p.T1313M-p.R1448C-Most-Common-PMC-Hotspots -- "
            "p.T704M-p.M1592V-Most-Common-HyperPP2-Hotspots -- "
            "Mexiletine-FIRST-LINE-Paramyotonia-Sodium-Channel-Blocker -- "
            "Acetazolamide-FIRST-LINE-HyperPP2-Plus-Avoid-Potassium -- "
            "17q23.3"
        ),
        "locus": "17q23.3",
        "protein_size": "1836 aa",
        "inheritance": (
            "AD (autosomal dominant); gain-of-function variants in Nav1.4 alpha subunit; "
            "different variants cause different phenotypes (PMC vs HyperPP2 vs HypoPP2 vs SCM); "
            "genotype-phenotype correlation: R1448 variants → PMC; T704M → HyperPP2; "
            "family history may show phenotypic variability even with same variant; "
            "de novo variants reported in severe cases."
        ),
        "key_features": [
            "PARADOXICAL MYOTONIA: cold worsens myotonia AND warm-up does NOT help — PATHOGNOMONIC (DDx CLCN1)",
            "Cold-immersion test: 5 min in cold water provokes myotonia + possible weakness",
            "Potassium-triggered paralysis (HyperPP2): oral KCl 1 mEq/kg provocative test",
            "Can overlap PMC + HyperPP2 in same patient (allelic, variant-specific phenotype)",
            "Exercise warms muscle transiently then paralysis during REST after exercise (K+ release)",
            "EMG: myotonic discharges (dive-bomber); may show electrical silence during paralytic attacks",
        ],
        "treatment": [
            "Mexiletine 150-200 mg TID — FIRST-LINE for paramyotonia and SCM (sodium channel blocker)",
            "Acetazolamide — FIRST-LINE for HyperPP2 (avoid in PMC — can worsen)",
            "Dichlorphenamide — alternative carbonic anhydrase inhibitor for HyperPP2/HypoPP2",
            "Avoid cold triggers — warm environment; heated gloves in winter for PMC",
            "Avoid potassium-rich foods during HyperPP2 attacks; glucose + insulin if severe",
            "Provocative testing (cold-immersion/KCl) done under supervised hospital setting only",
        ],
        "contraindications": [
            "POTASSIUM SUPPLEMENTATION CONTRAINDICATED in HyperPP2 — worsens paralysis",
            "ACETAZOLAMIDE CONTRAINDICATED in pure PMC — may worsen myotonia (alkalosis mechanism)",
            "SUCCINYLCHOLINE CAUTION in SCN4A channelopathy — can trigger prolonged paralysis",
            "COLD ENVIRONMENTS: strict avoidance for PMC — cold is a definitive trigger",
            "STRENUOUS EXERCISE FOLLOWED BY REST: avoid in HyperPP2 — classic attack precipitant",
        ],
        "critical_pearls": [
            "COLD worsening + NO warm-up benefit = SCN4A (PMC) not CLCN1 — cold-immersion test",
            "Genotype determines treatment: confirm PMC vs HyperPP2 before choosing mexiletine vs acetazolamide",
            "HypoPP type 2: SCN4A, treated differently to HypoPP1 (CACNA1S) — same clinical phenotype",
            "Vacuolar myopathy on biopsy: chronic HyperPP — prompt treatment prevents permanent weakness",
            "EMG during attack: electrical silence = channel inactivation (Na+ channel trapped in open state)",
        ],
    },
    # -- CACNA1S — Hypokalemic Periodic Paralysis Type 1 (HypoPP1) --------------------
    {
        "gene": "CACNA1S",
        "alt_name": (
            "CACNA1S (CACNA1S-1873aa-1q32.1 / AD — HypoPP1-Hypokalemic-Periodic-Paralysis-Type-1 — "
            "CARBOHYDRATE+REST-After-Exercise-Triggers-PATHOGNOMONIC — "
            "ACETAZOLAMIDE-FIRST-LINE-Plus-Oral-KCl — "
            "Malignant-Hyperthermia-Susceptibility-MHS5-Same-Gene)"
        ),
        "protein": (
            "CACNA1S -- 1q32.1 AD -- CACNA1S-1873aa -- "
            "Cav1.1-L-Type-Voltage-Gated-Ca2+-Channel-Alpha1S-Skeletal-Muscle-Excitation-Contraction -- "
            "HypoPP1-Hypokalemic-Periodic-Paralysis-Type-1-OMIM-170400-Most-Common-Hereditary-PP -- "
            "MHS5-Malignant-Hyperthermia-Susceptibility-5-Allelic-Same-Gene -- "
            "Gating-Pore-Current-S4-Voltage-Sensor-Arginine-Substitution-Mechanism-Unique -- "
            "R1-R2-R3-Arginine-S4-Mutations-Most-Common-p.R528H-p.R1239H-p.R1239G -- "
            "Carbohydrate-Meal-Triggers-Insulin-Drives-K+-Into-Cells-Hypokalemia -- "
            "Rest-After-Exercise-Triggers-Exertional-K+-Uptake-Intracellular-Shift -- "
            "Attacks-2nd-3rd-Decade-Improve-After-50yr-Spontaneously -- "
            "Bulbar-Respiratory-Muscles-Usually-Spared-DDx-Myasthenia -- "
            "Serum-K+-LOW-During-Attack-<3.5-Often-<2.5 -- "
            "Acetazolamide-FIRST-LINE-Prevents-Attacks-Not-Replaces-KCl -- "
            "Oral-KCl-During-Attack-First-Choice-IV-If-Severe -- "
            "Thyroid-Exclude-TPP-Thyrotoxic-PP-First-Common-Cause -- "
            "1q32.1"
        ),
        "locus": "1q32.1",
        "protein_size": "1873 aa",
        "inheritance": (
            "AD (autosomal dominant); gain-of-function gating pore current in S4 voltage sensor; "
            "p.R528H and p.R1239H most common (>60% of CACNA1S HypoPP1); "
            "penetrance higher in males — males more severely affected; "
            "females may be subclinical carriers; "
            "allelic MHS5 — same gene, different variant class."
        ),
        "key_features": [
            "CARBOHYDRATE + REST AFTER EXERCISE triggers paralytic attack — PATHOGNOMONIC",
            "Serum K+ falls during attack (<3.5, often <2.5 mEq/L) — measure K+ at attack onset",
            "Bulbar and respiratory muscles usually spared — if affected, consider myasthenia DDx",
            "Attacks peak in 2nd-3rd decade; spontaneous improvement after age 50 years",
            "EXCLUDE thyrotoxic periodic paralysis first — check TSH (common acquires cause of HypoPP)",
            "Vacuolar myopathy on biopsy in chronic cases — prevent with acetazolamide",
        ],
        "treatment": [
            "Acetazolamide 125-250 mg BD — FIRST-LINE prevention (reduces attack frequency 50-70%)",
            "Oral KCl 40-60 mEq — acute attack treatment; prefer oral over IV if swallowing intact",
            "IV KCl (non-glucose saline) — severe attacks with weakness or K+ <2.5 mEq/L",
            "Dietary: low-carbohydrate, low-sodium, avoid alcohol, regular mild activity",
            "Avoid triggers: large carbohydrate meals, REST after strenuous exercise, alcohol",
            "Dichlorphenamide — alternative if acetazolamide intolerated",
        ],
        "contraindications": [
            "GLUCOSE/DEXTROSE IV SOLUTIONS CONTRAINDICATED — drive K+ intracellularly, worsen attack",
            "HIGH-CARBOHYDRATE MEALS: avoid large portions, especially evening carbohydrate load",
            "SPIRONOLACTONE CONTRAINDICATED — can precipitate or prolong attacks",
            "SUCCINYLCHOLINE CAUTION (allelic MHS5) — malignant hyperthermia susceptibility",
            "VOLATILE ANAESTHETICS CAUTION (MHS5) — dantrolene must be available",
            "ALCOHOL: triggers attacks — complete avoidance recommended",
        ],
        "critical_pearls": [
            "ALWAYS check TSH first — thyrotoxic PP is the common acquired cause of HypoPP",
            "IV glucose solutions will worsen the attack — use normal saline with KCl",
            "MHS5 overlap: when anaesthesia planned, MH precautions mandatory even without MH history",
            "Females may have milder phenotype but still carry and transmit mutation",
            "Vacuolar myopathy on biopsy = permanent weakness — start acetazolamide promptly",
        ],
    },
    # -- KCNJ2 — Andersen-Tawil Syndrome (ATS/LQT7) -----------------------------------
    {
        "gene": "KCNJ2",
        "alt_name": (
            "KCNJ2 (KCNJ2-427aa-17q24.3 / AD — Andersen-Tawil-Syndrome-ATS-LQT7 — "
            "TRIAD-Episodic-Flaccid-Paralysis+Cardiac-Arrhythmia-LQT+Dysmorphic-Features-PATHOGNOMONIC — "
            "FLECAINIDE-Arrhythmia-ICD-Mandatory-Consideration — "
            "NOT-Typical-LQTS-Management-VERY-DIFFERENT-Protocol)"
        ),
        "protein": (
            "KCNJ2 -- 17q24.3 AD -- KCNJ2-427aa -- "
            "Kir2.1-Inward-Rectifier-K+-Channel-2-TM-PIP2-Dependent-IK1-Current -- "
            "Andersen-Tawil-Syndrome-ATS-OMIM-170390-LQT7-Atkins-2003-Renaming -- "
            "TRIAD-Episodic-Flaccid-Paralysis-Cardiac-LQT-Ventricular-Arrhythmia-Dysmorphic-PATHOGNOMONIC -- "
            "Dysmorphic-Features-Hypertelorism-Mandibular-Hypoplasia-Clinodactyly-Low-Set-Ears -- "
            "Cardiac-Bidirectional-VT-PVCs-LQT-POLYMORPHIC-VT-NOT-Torsades-Typical -- "
            "Paralysis-Hypo-Normo-Hyperkalemic-All-3-Types-Seen-DISTINCTIVE -- "
            "LOF-Dominant-Negative-Mechanism-IK1-Current-Reduced -- "
            "FLECAINIDE-CLASS-IC-Bidirectional-VT-ATS-Specific-Evidence -- "
            "ICD-Considered-If-Sustained-VT-Syncope-Cardiac-Arrest-History -- "
            "QTc-Prolonged-Usually-Not-As-Long-As-LQT1-LQT2-BEWARE-Underestimation -- "
            "Beta-Blockers-Modest-Benefit-NOT-As-Effective-LQT1-LQT2 -- "
            "Acetazolamide-Paralysis-Prevention-Variable-Benefit -- "
            "17q24.3"
        ),
        "locus": "17q24.3",
        "protein_size": "427 aa",
        "inheritance": (
            "AD (autosomal dominant); loss-of-function dominant-negative mechanism on Kir2.1; "
            "penetrance variable — incomplete penetrance in some families; "
            "de novo variants in ~30% of cases; "
            "triad penetrance incomplete — not all three features present in every case; "
            "KCNJ2 and KCNJ5 allelic (KCNJ5 → Familial Primary Aldosteronism type 3 — different disease)."
        ),
        "key_features": [
            "TRIAD: episodic flaccid paralysis + cardiac arrhythmia (LQT/bidirectional VT) + dysmorphic features — PATHOGNOMONIC",
            "Cardiac: bidirectional VT, frequent PVCs, polymorphic VT — NOT typical torsades-de-pointes pattern",
            "Dysmorphic: hypertelorism, mandibular hypoplasia, clinodactyly, low-set ears",
            "Paralysis can occur with hypo-, normo-, OR hyper-kalemia — all three types reported",
            "QTc prolonged but often less marked than LQT1/LQT2 — NEVER underestimate arrhythmia risk",
            "Cardiac arrest/sudden death risk: ICD threshold lower than typical LQTS",
        ],
        "treatment": [
            "Flecainide — CLASS IC, specific evidence for ATS bidirectional VT suppression",
            "ICD implantation — if syncope, sustained VT, or cardiac arrest history",
            "Acetazolamide — paralysis prevention (variable benefit; use if other HypoPP treatments fail)",
            "Oral KCl or potassium-sparing diet — if hypokalemic attacks predominate",
            "Avoid QT-prolonging drugs (azithromycin, haloperidol, ondansetron) — prolonged QTc",
            "Annual cardiology review + Holter monitoring — arrhythmia burden tracking",
        ],
        "contraindications": [
            "ALL QT-PROLONGING MEDICATIONS: check QTc before prescribing any new drug (CredibleMeds)",
            "CLASS IA ANTIARRHYTHMICS (quinidine, procainamide) — worsen QT, use flecainide instead",
            "AMIODARONE CAUTION — QT prolongation; not first-line in ATS",
            "BETA-BLOCKERS ALONE: insufficient for bidirectional VT in ATS; add flecainide",
            "SUCCINYLCHOLINE CAUTION — hyperkalemia surge may trigger arrhythmia in ATS",
            "HIGH-INTENSITY EXERCISE — arrhythmia risk; restrict to moderate supervised activity",
        ],
        "critical_pearls": [
            "ATS arrhythmia is bidirectional VT — looks different from typical LQTS on ECG",
            "Triad is incomplete in many patients — dysmorphic features alone should prompt KCNJ2 testing",
            "QTc may be 'borderline' — U-waves merge with T-waves; measure QTU interval",
            "Flecainide is ATS-specific — use INSTEAD of amiodarone for sustained VT",
            "Genetic testing: KCNJ2 first; if negative consider KCNJ5 and CALM1-3 for overlap DDx",
        ],
    },
    # -- RYR1 — Malignant Hyperthermia Susceptibility + Core Myopathy ------------------
    {
        "gene": "RYR1",
        "alt_name": (
            "RYR1 (RYR1-5038aa-19q13.2 / AR-AD — MHS1-Malignant-Hyperthermia-Susceptibility-Core-Myopathy — "
            "SUCCINYLCHOLINE+ALL-VOLATILE-ANAESTHETICS-ABSOLUTELY-CI-MH-TRIGGER — "
            "DANTROLENE-Life-Saving-Antidote-Must-Be-Available-ALL-Theatres — "
            "KATP-Hotspots-p.R614C-p.R2163C-Ryanodine-Test-IVCT)"
        ),
        "protein": (
            "RYR1 -- 19q13.2 AR/AD -- RYR1-5038aa -- "
            "Ryanodine-Receptor-1-SR-Ca2+-Release-Channel-Homotetrameric-Largest-Known-Protein -- "
            "MHS1-Malignant-Hyperthermia-Susceptibility-OMIM-145600-AD-Most-Common-MHS -- "
            "CCD-Central-Core-Disease-OMIM-117000-AR-More-Severe-Bilateral-Symmetric-Weakness -- "
            "MmD-Multi-Minicore-Disease-OMIM-255320-AR -- "
            "King-Denborough-Syndrome-AD-Dysmorphic-MH-Susceptibility -- "
            "TRIGGERING-AGENTS-Succinylcholine-ABSOLUTE-CI-ALL-Volatile-Agents-ABSOLUTE-CI -- "
            "DANTROLENE-2.5mg/kg-IV-Bolus-Every-5min-Max-10mg/kg-LIFE-SAVING -- "
            "MH-CRISIS-Hyperthermia-Rigidity-Tachycardia-Hypercarbia-Acidosis-CK-Rise -- "
            "Cooling-Hyperventilate-100pct-O2-Dantrolene-Bicarbonate-Procainamide-If-Arrhythmia -- "
            "IVCT-In-Vitro-Contracture-Test-Gold-Standard-Diagnosis-MHS -- "
            "p.R614C-p.R2163C-p.T4826I-AD-MHS-Hotspots-EMHG-Classified-Pathogenic -- "
            "AR-Variants-Central-Core-Disease-Central-Cores-Biopsy-Absent-Mitochondria-Oxidative -- "
            "TIVA-Total-Intravenous-Anaesthesia-Propofol-Mandatory-RYR1-Patients -- "
            "Dantrolene-26-Vials-Minimum-Available-All-MHS-Theatres-Mandatory -- "
            "19q13.2"
        ),
        "locus": "19q13.2",
        "protein_size": "5038 aa",
        "inheritance": (
            "AD (MHS1): gain-of-function pathogenic variants in RYR1; penetrance near-complete; "
            "AR (Core Myopathy): biallelic LOF variants; more severe; neonatal hypotonia, scoliosis; "
            "same gene — allelic spectrum from MHS (AD, asymptomatic until anaesthesia) to CCD (AR, lifelong); "
            "EMHG (European Malignant Hyperthermia Group) classifies 31 pathogenic RYR1 variants; "
            "genetic diagnosis supplemented by IVCT (in vitro contracture test) — gold standard."
        ),
        "key_features": [
            "MH crisis: rapid hyperthermia + rigidity + tachycardia + hypercarbia + rhabdomyolysis triggered by volatile anaesthetics or succinylcholine",
            "DANTROLENE 2.5 mg/kg IV bolus repeated every 5 min (max 10 mg/kg) — LIFE-SAVING antidote",
            "TIVA (propofol-based) mandatory for all RYR1 patients — NO volatile agents ever",
            "Central Core Disease: proximal weakness + scoliosis + hip dysplasia + respiratory compromise",
            "CK elevated in core myopathy; may be normal in MHS-only patients",
            "Family testing: all first-degree relatives of MH survivor need IVCT or genetic testing",
        ],
        "treatment": [
            "MH CRISIS: Dantrolene 2.5 mg/kg IV bolus q5min max 10 mg/kg + cooling + hyperventilation O2",
            "Dantrolene maintenance post-crisis: 1 mg/kg q4-6h for 24-48h to prevent recurrence",
            "TIVA for ALL future anaesthetics: propofol infusion + nitrous oxide permitted + regional blocks",
            "Core Myopathy supportive: physiotherapy, scoliosis monitoring, respiratory monitoring",
            "Annual FVC in CCD: NIV if FVC <50%; nocturnal oximetry",
            "Orthopaedic: hip dysplasia screening, scoliosis surgical threshold",
        ],
        "contraindications": [
            "ALL VOLATILE ANAESTHETICS ABSOLUTELY CONTRAINDICATED: halothane, isoflurane, sevoflurane, desflurane",
            "SUCCINYLCHOLINE ABSOLUTELY CONTRAINDICATED: triggers MH + massive K+ release in myopathy",
            "NITROUS OXIDE: permitted (not an MH trigger — contrary to common belief)",
            "PROPOFOL INFUSION SYNDROME: high-dose propofol risk with concurrent mitochondrial disease",
            "STATIN MYOPATHY RISK INCREASED in CCD (underlying myopathy)",
            "ENSURE dantrolene 26 vials (700 mg) available before any RYR1 patient enters theatre",
        ],
        "critical_pearls": [
            "26 vials of dantrolene MUST be on-hand before RYR1 patient enters theatre — non-negotiable",
            "MH can occur on FIRST anaesthetic — no prior uneventful anaesthetic excludes susceptibility",
            "SUCCINYLCHOLINE triggers MH INDEPENDENTLY of anaesthetic agent — both triggers required in practice",
            "CCD and MHS can coexist in same patient: both risks simultaneously",
            "If MH suspected intraoperatively: STOP volatile, call for help, mix dantrolene IMMEDIATELY",
        ],
    },
    # -- ATP2A1 — Brody Myopathy (SERCA1 deficiency) ----------------------------------
    {
        "gene": "ATP2A1",
        "alt_name": (
            "ATP2A1 (ATP2A1-994aa-16p11.2 / AR — Brody-Myopathy-SERCA1-Deficiency — "
            "SILENT-MYOTONIA-EMG-SILENT-Despite-Mechanical-Stiffness-PATHOGNOMONIC-DDx-CLCN1-SCN4A — "
            "Exercise-Induced-Impaired-Muscle-Relaxation-Pain-Free — "
            "No-Systemic-Features-Pure-Skeletal-Muscle)"
        ),
        "protein": (
            "ATP2A1 -- 16p11.2 AR -- ATP2A1-994aa -- "
            "SERCA1-Sarco-Endoplasmic-Reticulum-Ca2+-ATPase-1-Fast-Twitch-Muscle-SR-Ca2+-Pump -- "
            "Brody-Myopathy-OMIM-601003-Exercise-Induced-Impaired-Relaxation-Ca2+-Accumulation-Cytosol -- "
            "SILENT-Myotonia-Mechanical-Stiffness-NO-EMG-Myotonic-Discharges-KEY-DDx -- "
            "EMG-SILENT-During-Stiffness-PATHOGNOMONIC-Not-CLCN1-Not-SCN4A-No-Dive-Bomber -- "
            "Exercise-Worsens-Stiffness-Prolonged-Relaxation-Especially-Fast-Twitch-Muscles -- "
            "Pain-Usually-Absent-Stiffness-Without-Cramping-DDx-Metabolic-Myopathies -- "
            "No-CK-Elevation-No-Systemic-Features-Pure-Skeletal-Muscle-SERCA1 -- "
            "Verapamil-Level-C-Anecdotal-Ca2+-Channel-Antagonist-Reduces-Cytosolic-Ca2+ -- "
            "Dantrolene-Theoretical-Reduces-RYR1-Ca2+-Release-Limited-Evidence -- "
            "No-FDA-Approved-Treatment-Symptomatic-Physiotherapy -- "
            "Rare-Disease-<50-Families-Reported-Worldwide -- "
            "16p11.2"
        ),
        "locus": "16p11.2",
        "protein_size": "994 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic LOF in ATP2A1 encoding SERCA1; "
            "very rare — fewer than 50 families reported worldwide; "
            "obligate heterozygous carriers asymptomatic; "
            "rare dominant variants reported (ATP2A1 dominant — Brody-like with AD inheritance); "
            "note: ATP2A2 (SERCA2) → Darier disease (skin); not the same gene."
        ),
        "key_features": [
            "SILENT MYOTONIA: EMG is electrically SILENT despite mechanical stiffness — PATHOGNOMONIC DDx CLCN1/SCN4A",
            "Exercise-induced delayed muscle relaxation: especially fast-twitch muscles (forearms, calves)",
            "Painless — stiffness WITHOUT cramping or myalgia (DDx exercise-induced cramps in metabolic myopathies)",
            "Normal CK, no systemic involvement, no cardiac involvement",
            "Worsens with rapid repetitive activity; slow relaxation time measurable on SFEMG",
            "EMG: NO myotonic discharges — the absence is diagnostic; muscle biopsy: SERCA1 IHC reduced",
        ],
        "treatment": [
            "Verapamil 40-80 mg TID — anecdotal benefit; reduces cytosolic Ca2+ accumulation (level C)",
            "Dantrolene — theoretical benefit reducing RyR1 Ca2+ release; limited clinical evidence",
            "Physiotherapy — paced activity, avoid abrupt maximal contractions, stretching",
            "Lifestyle: pace activity, avoid sudden intense exercise; allow rest periods",
            "No disease-modifying treatment; gene therapy in preclinical development",
        ],
        "contraindications": [
            "MEXILETINE INEFFECTIVE — does not address SERCA1 Ca2+ pump deficiency",
            "ACETAZOLAMIDE INEFFECTIVE — not a channelopathy mechanism",
            "AVOID MISDIAGNOSIS as CLCN1/SCN4A — EMG distinguishes (silent vs discharges)",
            "High-intensity interval training worsens symptoms — moderate paced exercise preferred",
        ],
        "critical_pearls": [
            "SILENT EMG during mechanical stiffness = Brody until proven otherwise",
            "EMG is the KEY DIAGNOSTIC discriminator — order EMG before gene panel",
            "ATP2A1 deficiency is SERCA1 — confirm with muscle biopsy IHC (anti-SERCA1 markedly reduced)",
            "Brody myopathy is ultra-rare: exclude all other causes of exercise myopathy first",
            "Do not confuse ATP2A1 (SERCA1, AR, Brody) with ATP2A2 (SERCA2, AD, Darier disease — skin)",
        ],
    },
]


def _make_patients(gene_data, seed, n=40):
    """Generate deterministic synthetic patient records for one gene."""
    rng = random.Random(seed)
    gene = gene_data["gene"]
    locus = gene_data["locus"]
    inh = gene_data["inheritance"]
    ad = "AR" not in inh.split(";")[0].upper()[:10]

    # Age at onset varies by condition
    if gene == "DMPK":
        ages = [rng.randint(10, 60) for _ in range(n)]
    elif gene == "CNBP":
        ages = [rng.randint(20, 65) for _ in range(n)]
    elif gene == "CLCN1":
        ages = [rng.randint(1, 40) for _ in range(n)]
    elif gene == "SCN4A":
        ages = [rng.randint(5, 45) for _ in range(n)]
    elif gene == "CACNA1S":
        ages = [rng.randint(10, 50) for _ in range(n)]
    elif gene == "KCNJ2":
        ages = [rng.randint(5, 40) for _ in range(n)]
    elif gene == "RYR1":
        ages = [rng.randint(0, 50) for _ in range(n)]
    else:  # ATP2A1
        ages = [rng.randint(10, 50) for _ in range(n)]

    # Attack frequency (episodes per month) — varies by disease
    if gene in ("CACNA1S", "KCNJ2", "SCN4A"):
        attack_freqs = [rng.randint(0, 12) for _ in range(n)]
    elif gene in ("CLCN1", "CNBP"):
        attack_freqs = [rng.randint(0, 4) for _ in range(n)]  # stiffness episodes counted
    else:
        attack_freqs = [rng.randint(0, 6) for _ in range(n)]

    # Treatment response — mexiletine/acetazolamide high response rates
    if gene == "CLCN1":
        treated_fraction = 0.90  # mexiletine very effective
    elif gene == "CACNA1S":
        treated_fraction = 0.80  # acetazolamide reduces attacks
    elif gene == "DMPK":
        treated_fraction = 0.75  # mexiletine for myotonia but disease still progresses
    elif gene == "RYR1":
        treated_fraction = 0.70  # TIVA protocol adherence
    else:
        treated_fraction = 0.65

    treated = [rng.random() < treated_fraction for _ in range(n)]

    # Alive — near complete for most conditions
    if gene in ("KCNJ2", "RYR1"):
        alive_fraction = 0.92  # arrhythmia/MH mortality risk
    elif gene == "DMPK":
        alive_fraction = 0.88  # cardiac/respiratory mortality
    else:
        alive_fraction = 0.97

    alive = [rng.random() < alive_fraction for _ in range(n)]

    # Sex distribution
    if gene == "CACNA1S":
        female_fraction = 0.45  # males more severely affected
    elif gene == "DMPK":
        female_fraction = 0.48
    else:
        female_fraction = 0.50

    patients = []
    for i in range(n):
        patients.append({
            "id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "seed": seed,
            "age": ages[i],
            "sex": "F" if rng.random() < female_fraction else "M",
            "alive": alive[i],
            "treated": treated[i],
            "attacks_per_month": attack_freqs[i],
            "inheritance": "AD" if ad else "AR",
            "locus": locus,
        })
    return patients


def overview():
    all_patients = []
    for i, gd in enumerate(MYOTONIC_GENES):
        all_patients += _make_patients(gd, SEED_BASE + i)

    total = len(all_patients)
    alive = sum(1 for p in all_patients if p["alive"])
    female = sum(1 for p in all_patients if p["sex"] == "F")
    treated = sum(1 for p in all_patients if p["treated"])
    avg_age = round(sum(p["age"] for p in all_patients) / total, 1)
    avg_attacks = round(sum(p["attacks_per_month"] for p in all_patients) / total, 1)

    gene_summaries = {}
    for i, gd in enumerate(MYOTONIC_GENES):
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
        "atlas": "Hereditary-Myotonic-Disorder-Atlas",
        "subtitle": "Complete 8-Gene Myotonic Disorder & Periodic Paralysis Reference",
        "genes": [g["gene"] for g in MYOTONIC_GENES],
        "seeds": f"{SEED_BASE}-{SEED_BASE+7}",
        "total_patients": total,
        "alive_pct": round(100 * alive / total, 1),
        "female_pct": round(100 * female / total, 1),
        "treated_pct": round(100 * treated / total, 1),
        "avg_age": avg_age,
        "avg_attacks_per_month": avg_attacks,
        "gene_summaries": gene_summaries,
        "clinical_axioms": [
            "DMPK/DM1: succinylcholine ABSOLUTELY CI + volatile anaesthetics EXTREME CAUTION — anaesthesia alert card mandatory",
            "CNBP/DM2: proximal (NOT distal) weakness + myalgia — often misdiagnosed as fibromyalgia for years",
            "CLCN1: warm-up phenomenon PATHOGNOMONIC — NO systemic features DDx DM1/DM2",
            "SCN4A/PMC: COLD worsens myotonia AND warm-up FAILS — PATHOGNOMONIC DDx CLCN1",
            "CACNA1S/HypoPP1: glucose/dextrose IV CONTRAINDICATED — normal saline + KCl only",
            "KCNJ2/ATS: TRIAD paralysis + bidirectional VT + dysmorphic — flecainide + ICD evaluation",
            "RYR1/MHS: ALL volatile anaesthetics + succinylcholine ABSOLUTELY CI — dantrolene 26 vials mandatory",
            "ATP2A1/Brody: EMG SILENT despite stiffness PATHOGNOMONIC — NOT CLCN1, NOT SCN4A",
        ],
    }


def breakdown():
    result = {}
    for i, gd in enumerate(MYOTONIC_GENES):
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
        "myotonic_disorder_classification": {
            "DM1": "Myotonic Dystrophy Type 1 — DMPK CTG repeat; multisystem; distal weakness; anticipation; succinylcholine CI",
            "DM2": "Myotonic Dystrophy Type 2 — CNBP CCTG repeat; proximal weakness; myalgia; no congenital form",
            "MC_Thomsen": "Myotonia Congenita Thomsen — CLCN1 AD; warm-up phenomenon; pure myotonia; no weakness",
            "MC_Becker": "Myotonia Congenita Becker — CLCN1 AR; warm-up + transient weakness; more severe than Thomsen",
            "PMC": "Paramyotonia Congenita — SCN4A AD; COLD worsens paradoxically; warm-up FAILS; mexiletine",
            "HyperPP": "Hyperkalemic Periodic Paralysis — SCN4A AD; K+ triggers; acetazolamide; avoid K+ supplements",
            "HypoPP1": "Hypokalemic Periodic Paralysis Type 1 — CACNA1S AD; carbohydrate+rest triggers; acetazolamide",
            "HypoPP2": "Hypokalemic Periodic Paralysis Type 2 — SCN4A AD; same phenotype as HypoPP1 different gene",
            "ATS": "Andersen-Tawil Syndrome — KCNJ2 AD; paralysis+LQT+dysmorphic triad; flecainide; ICD",
            "MHS": "Malignant Hyperthermia Susceptibility — RYR1 AD; triggered by volatile+succinylcholine; dantrolene",
            "CCD": "Central Core Disease — RYR1 AR; congenital myopathy; central cores biopsy; same gene as MHS",
            "Brody": "Brody Myopathy — ATP2A1 AR; exercise-induced stiffness; EMG SILENT; SERCA1 deficiency",
        },
        "key_pharmacology": {
            "Mexiletine_DM1_CLCN1": "Mexiletine 150-200 mg TID — FIRST-LINE DM1 myotonia and MC (CLCN1); level A RCT evidence for CLCN1",
            "Acetazolamide_HypoPP": "Acetazolamide 125-250 mg BD — FIRST-LINE HypoPP1 (CACNA1S) and HypoPP2 (SCN4A); reduces attacks 50-70%",
            "Dantrolene_MH": "Dantrolene 2.5 mg/kg IV bolus q5min (max 10 mg/kg) — SOLE LIFE-SAVING antidote for MH crisis",
            "Flecainide_ATS": "Flecainide CLASS IC — specific evidence for ATS (KCNJ2) bidirectional VT suppression",
            "Verapamil_Brody": "Verapamil 40-80 mg TID — anecdotal evidence for Brody myopathy (level C); reduces cytosolic Ca2+",
            "KCl_HypoPP": "Oral KCl 40-60 mEq — acute HypoPP attack; prefer oral over IV if swallowing intact",
            "Dichlorphenamide_PP": "Dichlorphenamide — alternative carbonic anhydrase inhibitor if acetazolamide intolerated",
            "Modafinil_DM1": "Modafinil/methylphenidate — daytime somnolence in DM1 (EDS prominent feature)",
        },
        "critical_contraindications": {
            "Succinylcholine_DM1": "SUCCINYLCHOLINE ABSOLUTELY CI in DM1 — prolonged depolarisation, myotonic crisis",
            "Succinylcholine_RYR1": "SUCCINYLCHOLINE ABSOLUTELY CI in RYR1/MHS — triggers malignant hyperthermia",
            "Volatile_anaesthetics_RYR1": "ALL VOLATILE ANAESTHETICS ABSOLUTELY CI in RYR1/MHS — isoflurane, sevoflurane, desflurane, halothane",
            "Glucose_HypoPP1": "GLUCOSE/DEXTROSE IV CONTRAINDICATED in HypoPP1 — drives K+ intracellularly, worsens attack",
            "Potassium_HyperPP": "POTASSIUM SUPPLEMENTS CONTRAINDICATED in HyperPP2 — worsens paralytic attacks",
            "Cold_SCN4A": "COLD ENVIRONMENTS in SCN4A/PMC — definitive trigger for paradoxical myotonia",
            "QT_KCNJ2": "ALL QT-PROLONGING DRUGS in KCNJ2/ATS — azithromycin, haloperidol, ondansetron, methadone",
            "Acetazolamide_PMC": "ACETAZOLAMIDE CONTRAINDICATED in pure PMC (SCN4A) — may worsen paramyotonia",
            "Carbohydrate_HypoPP": "HIGH-CARBOHYDRATE MEALS in HypoPP1 — triggers insulin-driven K+ intracellular shift",
        },
        "pathognomonic_signs": {
            "DMPK_distal_myotonia": "Distal weakness (ankle dorsiflexors) + grip myotonia + ptosis + 'hatchet face' — DM1/DMPK pathognomonic",
            "CNBP_proximal_pain": "PROXIMAL weakness + myalgia without distal involvement — DM2/CNBP DDx DM1 (distal)",
            "CLCN1_warm_up": "Stiffness severe at rest dramatically improves after 10-15 contractions — warm-up phenomenon PATHOGNOMONIC",
            "SCN4A_cold_paradox": "COLD worsens myotonia AND warm-up FAILS — paradoxical myotonia SCN4A/PMC PATHOGNOMONIC",
            "CACNA1S_carbohydrate": "Paralytic attacks triggered by CARBOHYDRATE MEAL + REST after exercise — HypoPP1 PATHOGNOMONIC",
            "KCNJ2_triad": "Episodic paralysis + bidirectional VT + hypertelorism/mandibular hypoplasia — ATS triad PATHOGNOMONIC",
            "RYR1_mh_crisis": "Rapid hyperthermia + rigidity + hypercarbia after volatile/succinylcholine — MH PATHOGNOMONIC",
            "ATP2A1_silent_emg": "Mechanical muscle stiffness with ELECTRICALLY SILENT EMG — Brody myopathy PATHOGNOMONIC",
        },
        "ddx_table": {
            "DM1_vs_DM2": "DM1: distal weakness, CTG, congenital form, anticipation, facial/distal; DM2: proximal, CCTG, no congenital, myalgia",
            "CLCN1_vs_SCN4A": "CLCN1: warm-up works, cold worsens but warm-up still helps; SCN4A/PMC: cold worsens AND warm-up FAILS",
            "CLCN1_vs_Brody": "CLCN1: EMG dive-bomber myotonic discharges; ATP2A1/Brody: EMG electrically SILENT",
            "HypoPP1_vs_HypoPP2": "HypoPP1: CACNA1S, R528H/R1239H; HypoPP2: SCN4A, same phenotype — gene panel required",
            "HyperPP_vs_HypoPP": "HyperPP: K+ triggers, K+ HIGH or normal during attack; HypoPP: K+ LOW during attack",
            "ATS_vs_LQT_typical": "ATS: bidirectional VT, flecainide preferred, triad with paralysis; LQT1/2: torsades, beta-blockers",
            "MH_vs_NMS": "MH: triggered by volatiles/succinylcholine, rigidity, hypercarbia, dantrolene curative; NMS: antipsychotics, gradual onset",
            "Brody_vs_metabolic_myopathy": "Brody: no pain, EMG silent; McArdle/GSD: cramp + myoglobinuria + CK elevation + ischaemic forearm test",
        },
        "anaesthesia_protocols": {
            "DM1_DM2_anaesthesia": (
                "TIVA MANDATORY: avoid succinylcholine (absolute), avoid volatile anaesthetics (extreme caution), "
                "sugammadex preferred over neostigmine, ICU monitoring post-op, avoid respiratory depressants"
            ),
            "RYR1_MHS_anaesthesia": (
                "TIVA MANDATORY: propofol infusion only, NO volatile agents (isoflurane/sevoflurane/desflurane/halothane), "
                "NO succinylcholine, 26 vials dantrolene pre-stocked, post-op monitoring 4-6 hours minimum"
            ),
            "MH_crisis_protocol": (
                "1. STOP volatile/succinylcholine immediately. "
                "2. Call for HELP + dantrolene. "
                "3. Dantrolene 2.5 mg/kg IV bolus q5min until rigidity resolves (max 10 mg/kg). "
                "4. Hyperventilate 100% O2. "
                "5. Active cooling (cold saline, ice packs). "
                "6. Bicarbonate for acidosis. "
                "7. Procainamide for arrhythmia (NOT lignocaine). "
                "8. ICU admission; dantrolene maintenance 1 mg/kg q6h × 24-48h."
            ),
            "CACNA1S_HypoPP_anaesthesia": (
                "Avoid glucose solutions; normal saline preferred; "
                "check K+ pre-op; MH precautions for allelic MHS5 overlap; "
                "monitor neuromuscular function post-reversal"
            ),
        },
        "emergency_protocols": {
            "MH_crisis": "STOP volatiles + DANTROLENE 2.5 mg/kg IV q5min + cooling + 100% O2 + ICU — call MH Helpline",
            "HypoPP_severe_attack": "IV KCl in normal saline (NOT glucose); monitor K+ 1-2 hourly; respiratory monitoring",
            "ATS_sustained_VT": "Flecainide 2 mg/kg IV if hemodynamically stable; DC cardioversion if unstable; cardiology + ICD evaluation",
            "DM1_respiratory_failure": "NIV if FVC <50%; intubation may trigger myotonic crisis — rapid sequence IF succinylcholine NOT used",
            "Hyperekplexia_style_note": "Note: Hyperekplexia (SLC6A5) managed in Paroxysmal Movement Disorder Atlas — refer there for nose-tipping protocol",
        },
        "glossary": {
            "Myotonia": "Delayed muscle relaxation after voluntary contraction — electrical or mechanical (Brody)",
            "Paramyotonia": "Myotonia worsened by cold and exercise; warm-up FAILS — SCN4A specific term",
            "Periodic_paralysis": "Episodic flaccid weakness due to transient electrical inexcitability of muscle membrane",
            "Warm_up_phenomenon": "Stiffness improves with repeated contractions — CLCN1 pathognomonic, absent in SCN4A/PMC",
            "Paradoxical_myotonia": "Myotonia worsening (not improving) with exercise — SCN4A/PMC hallmark",
            "Bidirectional_VT": "Beat-to-beat alternation of QRS axis — typical of ATS (KCNJ2), distinct from torsades",
            "CTG_repeat": "Cytosine-thymine-guanine trinucleotide expansion in DMPK 3'UTR — pathogenic >50 repeats",
            "CCTG_repeat": "Cytosine-cytosine-thymine-guanine tetranucleotide in CNBP intron 1 — DM2 cause",
            "Gating_pore_current": "Pathological leak current through S4 voltage sensor arginine substitution — CACNA1S HypoPP1 mechanism",
            "IVCT": "In Vitro Contracture Test — gold standard diagnosis for MH susceptibility; biopsy required",
            "TIVA": "Total Intravenous Anaesthesia — propofol-based; MANDATORY for DM1/DM2 and RYR1/MHS patients",
            "Dantrolene": "Ryanodine receptor antagonist — sole effective antidote for malignant hyperthermia",
            "SERCA1": "Sarco/Endoplasmic Reticulum Ca2+-ATPase 1 — encoded by ATP2A1; pumps Ca2+ back into SR after contraction",
            "IK1_current": "Inward rectifier K+ current — maintained by Kir2.1 (KCNJ2); stabilises resting membrane potential",
            "Anticipation": "Progressive worsening of repeat expansion and earlier onset in successive generations — DM1 feature",
            "CK": "Creatine kinase — elevated in myopathy; may be normal in channelopathies (CLCN1, CACNA1S)",
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
        print(f"  {gene}: {data['n_patients']} patients, alive={data['alive_pct']}%, treated={data['treated_pct']}%")
    print("\n=== DEFINITIONS keys ===")
    defs = definitions()
    print(list(defs.keys()))
