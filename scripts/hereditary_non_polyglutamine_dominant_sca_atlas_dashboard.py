#!/usr/bin/env python3
"""Hereditary-Non-Polyglutamine-Dominant-SCA-Atlas — Complete 8-Gene Non-Polyglutamine
Autosomal-Dominant Spinocerebellar Ataxia Atlas (SCA5/SCA19/SCA21/SCA42/SCA44/SCA45/SCA47/SCA48).

SPTBN2   (Spectrin beta-III; 2390 aa; 11q13.2; AD;
          SCA5 — "Lincoln Family Ataxia" — descends from Abraham Lincoln's maternal grandparents;
          PURE CEREBELLAR ataxia, very slow progression — can drive for 20+ yr after onset;
          Stabilises EAAT4 glutamate transporter on Purkinje cells;
          No pathognomonic sign — pure cerebellar on exam and MRI;
          Normal life expectancy; mild dysarthria; seed SEED_BASE+0).
CACNA1G  (Cav3.1 T-type voltage-gated Ca channel; 2107 aa; 17q21.33; AD;
          SCA42 — French/French-Canadian founder p.Arg1715His (c.5144G>A);
          PURE CEREBELLAR ataxia, adult onset 40s-50s, slow progression;
          T-type Ca channel gain-of-function alters Purkinje cell firing;
          Flunarizine (T-type Ca blocker) off-label trial in affected families;
          No pathognomonic sign; standard exome panels catch it; seed SEED_BASE+1).
KCND3    (Kv4.3 A-type potassium channel; 655 aa; 1p13.2; AD;
          SCA19/22 — COGNITIVE IMPAIRMENT + PSYCHIATRIC FEATURES EARLY PATHOGNOMONIC;
          Myoclonus 25% (multifocal, cortical) — distinguishes from pure cerebellar SCAs;
          Action tremor (hands) 40% before obvious ataxia;
          Dutch / Norwegian / Japanese ethnic clue;
          Slow progression; SARA scores low for years; seed SEED_BASE+2).
TMEM240  (TMEM240 transmembrane protein; 232 aa; 1p36.33; AD;
          SCA21 — COGNITIVE IMPAIRMENT PATHOGNOMONIC from CHILDHOOD (ADHD/dyslexia diagnosis);
          Ataxia onset 2nd-4th decade — years after first cognitive symptoms;
          Extrapyramidal features (tremor, rigidity) 30%;
          French family first described (Angioi 2015 Brain);
          Pure cerebellar cortical atrophy on MRI; seed SEED_BASE+3).
STUB1    (CHIP — C-terminus of Hsc70-Interacting Protein; E3 ubiquitin ligase; 303 aa; 16p13.3; AD;
          SCA48 — COGNITIVE IMPAIRMENT DOMINATES — misdiagnosed as FTD or early-onset dementia;
          Heterozygous = SCA48 (AD); homozygous/compound het = SCAR16 (AR, more severe);
          Parkinsonism 30% (rest tremor, rigidity, bradykinesia) — dopamine transporter SPECT abnormal;
          Psychiatric features (anxiety, depression, behavioural) 25%;
          Cerebellar ataxia often AFTER cognitive symptoms — neurogenetics referral delayed;
          Recently described (Corben 2019, Genis 2018); seed SEED_BASE+4).
GRM1     (mGluR1; metabotropic glutamate receptor 1; 1194 aa; 6q24.3; AD;
          SCA44 — ACTION TREMOR PROMINENT before cerebellar ataxia — misdiagnosed as Essential Tremor;
          Gq-coupled receptor: PKC signalling in Purkinje cells;
          Adult onset 4th-6th decade, very slow progression;
          Cerebellar atrophy on MRI, brainstem spared;
          Negative allosteric mGluR1 modulator trials ongoing (e.g. BAY-357); seed SEED_BASE+5).
FAT2     (FAT atypical cadherin 2; 4589 aa; 5q33.1; AD;
          SCA45 — adult onset pure cerebellar, slow progression;
          FAT2 protocadherin: cell-cell adhesion + planar cell polarity signalling;
          Few families worldwide; standard SCA gene panels capture it;
          No pathognomonic clinical feature; pure cerebellar on MRI;
          Normal life expectancy; mild dysarthria; seed SEED_BASE+6).
PUM1     (Pumilio 1 RNA-binding protein; 1186 aa; 1p35.2; AD;
          SCA47 — DEVELOPMENTAL DELAY / MILD-MODERATE INTELLECTUAL DISABILITY PATHOGNOMONIC;
          Non-progressive or slowly progressive cerebellar ataxia;
          Seizures 30% (infantile spasms or focal);
          Haploinsufficiency — PUM1 represses ATXN1/ATXN2/EAAT4 mRNAs;
          Childhood / early adult onset; cerebellar hypoplasia (not atrophy) on MRI;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2110-2117).
"""

import random

SEED_BASE = 2110

NONPOLYQ_DOMINANT_SCA_GENES = [
    # -- SPTBN2 — SCA5 -------------------------------------------------------
    {
        "gene": "SPTBN2",
        "alt_name": (
            "SPTBN2 (SPTBN2-2390aa-11q13.2 / AD — SCA5-Pure-Cerebellar-Lincoln-Family — "
            "Very-Slow-Progression-Drive-20yr-After-Onset — "
            "EAAT4-Stabilisation-Purkinje-Cell-Glutamate-Transport — "
            "Normal-Life-Expectancy-No-Pathognomonic-Sign)"
        ),
        "protein": (
            "SPTBN2 -- 11q13.2 AD -- SPTBN2-2390aa -- "
            "Spectrin-beta-III-Cytoskeletal-Scaffolding-Purkinje-Cell-Dendritic-Arbour -- "
            "EAAT4-Glutamate-Transporter-Stabilisation-Dendritic-Spine-Anchoring -- "
            "Ankyrin-Binding-Domain-Spectrin-Repeat-Cytoskeletal-Organisation -- "
            "SCA5-Pure-Cerebellar-Syndrome-Dysmetria-Dysarthria-Dysdiadochokinesia -- "
            "Loss-Of-Function-Purkinje-Cell-Glutamate-Excitotoxicity-Chronic -- "
            "Spectrin-Repeat-Missense-Most-Common-Also-Truncating-Mutations -- "
            "VERY-SLOW-PROGRESSION-Can-Drive-20-Plus-Years-After-Symptom-Onset -- "
            "MRI-Pure-Cerebellar-Cortical-Atrophy-Vermis-Hemispheres-Brainstem-Spared -- "
            "Lincoln-Family-Abraham-Lincoln-Maternal-Grandparent-Descent -- "
            "Normal-Life-Expectancy-Mild-Disability"
        ),
        "locus": "11q13.2",
        "protein_size": "2390 aa",
        "inheritance": (
            "AD (autosomal dominant); high penetrance; "
            "missense mutations in spectrin-repeat domain most common; "
            "also truncating variants; "
            "de novo mutations reported; "
            "anticipation NOT described"
        ),
        "age_of_onset": "30-40 years (range: 10-60); slow onset",
        "pathognomonic": (
            "No pathognomonic sign — PURE CEREBELLAR syndrome; "
            "gait ataxia + limb ataxia + dysarthria without extracerebellar features; "
            "MRI: cerebellar cortical atrophy sparing brainstem; "
            "VERY SLOW progression = SCA5 clinical clue; "
            "Lincoln family descent geographic clustering (North American)"
        ),
        "treatment": (
            "No disease-modifying therapy; "
            "RILUZOLE 50 mg BD (off-label neuroprotection, Level B evidence in SCAs generally); "
            "Physiotherapy: balance, gait retraining, fall prevention; "
            "SLT: dysarthria management; "
            "Occupational therapy: adaptive equipment when needed; "
            "Avoid sedating medications (worsens ataxia); "
            "Genetic counselling: 50% inheritance risk each child; "
            "EUROSCA / ClinicalTrials.gov registration"
        ),
        "key_biomarker": "Pure cerebellar + very slow progression + Lincoln descent → SPTBN2 first",
        "critical_flags": [
            "SPTBN2-PURE-CEREBELLAR-SCA5",
            "VERY-SLOW-PROGRESSION-DRIVE-20yr",
            "LINCOLN-FAMILY-NORTH-AMERICAN",
            "EAAT4-STABILISATION-MECHANISM",
            "NORMAL-LIFE-EXPECTANCY",
            "NO-PATHOGNOMONIC-SIGN-PURE-CEREBELLAR",
            "RILUZOLE-LEVEL-B-TRIAL",
            "MRI-CEREBELLAR-BRAINSTEM-SPARED",
        ],
        "cohort_seed": SEED_BASE + 0,
    },
    # -- CACNA1G — SCA42 -----------------------------------------------------
    {
        "gene": "CACNA1G",
        "alt_name": (
            "CACNA1G (CACNA1G-2107aa-17q21.33 / AD — SCA42-Cav3.1-T-type-Calcium-Channel — "
            "French-Founder-pArg1715His-Gain-Of-Function — "
            "Flunarizine-T-type-Blocker-Off-Label-Trial — "
            "Pure-Cerebellar-Adult-Onset-Slow-Progression)"
        ),
        "protein": (
            "CACNA1G -- 17q21.33 AD -- CACNA1G-2107aa -- "
            "Cav3.1-Low-Voltage-Activated-T-Type-Calcium-Channel-Alpha-1G-Subunit -- "
            "Thalamo-Purkinje-Rhythmogenesis-T-Type-Channel-Burst-Firing -- "
            "Gain-Of-Function-Heterozygous-Slowed-Channel-Inactivation-Prolonged-Ca-Influx -- "
            "Purkinje-Cell-Dendrite-Hyperexcitability-Parallel-Fibre-LTD-Impaired -- "
            "SCA42-Pure-Cerebellar-Ataxia-Dysarthria-Dysmetria-Adult-Onset -- "
            "pArg1715His-c5144GA-Alpha-1G-Voltage-Sensor-Domain-IV-French-Founder -- "
            "Also-pArg1279Cys-pGly1108Ser-Additional-Gain-Of-Function-Variants -- "
            "MRI-Cerebellar-Cortical-Atrophy-Vermis-Hemispheres -- "
            "Flunarizine-T-Type-Ca-Blocker-Off-Label-Electrophysiological-Rationale -- "
            "Slow-Progression-Normal-Life-Expectancy"
        ),
        "locus": "17q21.33",
        "protein_size": "2107 aa",
        "inheritance": (
            "AD (autosomal dominant); gain-of-function mechanism; "
            "French and French-Canadian founder: p.Arg1715His; "
            "CACNA1G also mutated in SCA-42N (non-coding) variant; "
            "high penetrance; "
            "de novo reported"
        ),
        "age_of_onset": "40-55 years; adult onset",
        "pathognomonic": (
            "No pathognomonic sign — PURE CEREBELLAR; "
            "French/French-Canadian ancestry + pure cerebellar ataxia = test CACNA1G; "
            "T-type calcium channel GOF mechanism unique; "
            "Flunarizine (T-type blocker) may stabilise — trial warranted in carriers; "
            "SARA: slow progression; no extracerebellar features"
        ),
        "treatment": (
            "FLUNARIZINE 10 mg/day (T-type Ca channel blocker; off-label based on GOF mechanism; "
            "used in French-Canadian kindred with anecdotal benefit); "
            "Physiotherapy and SLT as for all SCAs; "
            "Genetic counselling: 50% risk; "
            "SARA biannually; "
            "MRI brain at diagnosis + if acceleration of symptoms; "
            "Register in EudraCT / SCA registry"
        ),
        "key_biomarker": "French/French-Canadian + pure cerebellar adult ataxia → CACNA1G (p.Arg1715His)",
        "critical_flags": [
            "CACNA1G-GAIN-OF-FUNCTION-T-TYPE-Ca",
            "FRENCH-FOUNDER-pArg1715His",
            "FLUNARIZINE-T-TYPE-BLOCKER-TRIAL",
            "PURE-CEREBELLAR-ADULT-ONSET",
            "SLOW-PROGRESSION-NORMAL-LIFE",
            "SCA42-DISTINCT-FROM-EA2-CACNA1A",
            "MRI-CEREBELLAR-CORTICAL-ATROPHY",
            "CACNA1G-NOT-CACNA1A-DIFFERENT-GENE",
        ],
        "cohort_seed": SEED_BASE + 1,
    },
    # -- KCND3 — SCA19/22 ----------------------------------------------------
    {
        "gene": "KCND3",
        "alt_name": (
            "KCND3 (KCND3-655aa-1p13.2 / AD — SCA19-SCA22-Kv4.3-A-type-Potassium-Channel — "
            "Cognitive-Psychiatric-Features-EARLY-PATHOGNOMONIC — "
            "Myoclonus-25pct-Action-Tremor-40pct-Non-Pure-Cerebellar — "
            "Dutch-Norwegian-Japanese-Founder)"
        ),
        "protein": (
            "KCND3 -- 1p13.2 AD -- KCND3-655aa -- "
            "Kv4.3-A-type-Rapidly-Inactivating-K-Channel-Repolarisation-Purkinje-Neurons -- "
            "Loss-Of-Function-Impairs-Purkinje-Firing-Pattern-Repetitive-Discharge -- "
            "S4-Voltage-Sensor-Mutations-Most-Common-Dominant-Negative-Channel-Trafficking -- "
            "pVal406Ile-Dutch-Founder-Cognitive-Psychiatric-Dominant -- "
            "pArg293Cys-Japanese-Pure-Cerebellar-Variant -- "
            "COGNITIVE-IMPAIRMENT-40-50pct-Depression-Anxiety-Psychiatric-Features-Early -- "
            "MYOCLONUS-25pct-Multifocal-Cortical-Action-Sensitive -- "
            "Action-Tremor-40pct-Hands-Before-Obvious-Ataxia -- "
            "Hyporeflexia-Some-Peripheral-Neuropathy-Axonal-30pct -- "
            "MRI-Cerebellar-Cortical-Atrophy-Mild"
        ),
        "locus": "1p13.2",
        "protein_size": "655 aa",
        "inheritance": (
            "AD (autosomal dominant); loss-of-function + dominant-negative; "
            "SCA19 (Dutch/Norwegian) and SCA22 (Asian/Japanese) = same gene different founder; "
            "p.Val406Ile Dutch/Belgian/Norwegian founder; "
            "p.Arg293Cys Japanese and Chinese families; "
            "penetrance >95% in carriers"
        ),
        "age_of_onset": "30-55 years (range: 20-70); cognitive often first",
        "pathognomonic": (
            "COGNITIVE IMPAIRMENT + PSYCHIATRIC FEATURES BEFORE ATAXIA PATHOGNOMONIC; "
            "Depression, anxiety, subtle cognitive slowing in 4th-5th decade before gait ataxia; "
            "Myoclonus (multifocal, action-sensitive) in 25% — distinguishes from pure cerebellar SCAs; "
            "Action tremor (hands) in 40% preceding ataxia — misdiagnosed as ET; "
            "Dutch/Norwegian/Japanese ethnic clue"
        ),
        "treatment": (
            "LEVETIRACETAM 500-1500 mg/day for myoclonus (preferred; least hepatotoxicity); "
            "CLONAZEPAM 0.5-2 mg/day for myoclonus if LEV insufficient; "
            "PROPRANOLOL 40-120 mg/day for action tremor; "
            "SSRI/SNRI for depression/anxiety (psychiatric component); "
            "RILUZOLE off-label neuroprotection (Level B SCAs); "
            "SARA biannually; neuropsychological testing at diagnosis + 2-yearly; "
            "Physiotherapy and SLT; "
            "Genetic counselling: 50% risk per child"
        ),
        "key_biomarker": "Cognitive/psychiatric + myoclonus + ataxia (Dutch/Norwegian/Japanese) → KCND3",
        "critical_flags": [
            "KCND3-COGNITIVE-PSYCHIATRIC-EARLY-PATHOGNOMONIC",
            "MYOCLONUS-25pct-ACTION-SENSITIVE-DDx-PME",
            "ACTION-TREMOR-40pct-MISDIAGNOSED-ET",
            "DUTCH-NORWEGIAN-JAPANESE-FOUNDER",
            "LEV-FOR-MYOCLONUS-FIRST-LINE",
            "NOT-PURE-CEREBELLAR-NON-MOTOR-FEATURES",
            "HYPOREFLEXIA-PERIPHERAL-NEUROPATHY-30pct",
            "SSRI-SNRI-PSYCHIATRIC-COMPONENT",
        ],
        "cohort_seed": SEED_BASE + 2,
    },
    # -- TMEM240 — SCA21 -----------------------------------------------------
    {
        "gene": "TMEM240",
        "alt_name": (
            "TMEM240 (TMEM240-232aa-1p36.33 / AD — SCA21-Cognitive-Impairment-PATHOGNOMONIC-Childhood — "
            "ADHD-Dyslexia-Diagnosed-Before-Ataxia-Appears — "
            "French-Family-Angioi-2015-Brain — "
            "Extrapyramidal-Features-30pct)"
        ),
        "protein": (
            "TMEM240 -- 1p36.33 AD -- TMEM240-232aa -- "
            "TMEM240-Single-Pass-Transmembrane-Protein-Cerebellar-Function-Unknown -- "
            "Expressed-Purkinje-Cells-Cerebellar-Nuclei-Cortex -- "
            "pArg59Trp-c175CT-French-Founder-Mild-Misfolding -- "
            "COGNITIVE-IMPAIRMENT-CHILDHOOD-ADHD-Dyslexia-Learning-Difficulty-First-Presentation -- "
            "Ataxia-Onset-2nd-To-4th-Decade-After-Cognitive-Symptoms-Already-Established -- "
            "EXTRAPYRAMIDAL-FEATURES-30pct-Tremor-Rigidity-Bradykinesia -- "
            "MRI-Pure-Cerebellar-Cortical-Atrophy-Vermis-Predominant -- "
            "SARA-Slow-Progression-10-yr-Mean-Course -- "
            "Normal-Life-Expectancy"
        ),
        "locus": "1p36.33",
        "protein_size": "232 aa",
        "inheritance": (
            "AD (autosomal dominant); "
            "French family first description (Angioi 2015 Brain); "
            "p.Arg59Trp primary pathogenic variant in French kindred; "
            "high penetrance; "
            "small number of unrelated families now reported"
        ),
        "age_of_onset": "Cognitive: childhood/teens; Ataxia: 20-40 years",
        "pathognomonic": (
            "COGNITIVE IMPAIRMENT FROM CHILDHOOD PATHOGNOMONIC; "
            "Learning difficulties, ADHD symptoms, dyslexia years before ataxia onset; "
            "Ataxia appears 2nd-4th decade after cognitive issues already established; "
            "Extrapyramidal features (tremor, mild rigidity) in ~30% — not typical for SCA; "
            "French ancestry + childhood cognitive issues + later ataxia = test TMEM240"
        ),
        "treatment": (
            "No disease-modifying therapy; "
            "Neuropsychological assessment at diagnosis (baseline cognitive profile); "
            "Educational support: ADHD/dyslexia accommodations for school-age children at risk; "
            "Extrapyramidal: LEVODOPA trial if Parkinsonism features (modest response); "
            "RILUZOLE off-label neuroprotection (Level B evidence in SCAs); "
            "Physiotherapy and SLT as per all SCAs; "
            "SARA biannually; MoCA/neuropsychology 2-yearly; "
            "Genetic counselling: 50% risk; cognitive symptoms in children = at-risk status"
        ),
        "key_biomarker": "Childhood cognitive issues (ADHD/dyslexia) + later adult-onset ataxia + French ancestry → TMEM240",
        "critical_flags": [
            "TMEM240-COGNITIVE-CHILDHOOD-PATHOGNOMONIC",
            "ADHD-DYSLEXIA-FIRST-DIAGNOSIS-BEFORE-ATAXIA",
            "ATAXIA-DELAYED-AFTER-COGNITIVE-ONSET",
            "EXTRAPYRAMIDAL-30pct-TREMOR-RIGIDITY",
            "FRENCH-FOUNDER-pArg59Trp",
            "LEVODOPA-TRIAL-EXTRAPYRAMIDAL-COMPONENT",
            "EDUCATIONAL-SUPPORT-AT-RISK-CHILDREN",
            "PURE-CEREBELLAR-MRI-CORTICAL-ATROPHY",
        ],
        "cohort_seed": SEED_BASE + 3,
    },
    # -- STUB1 — SCA48 --------------------------------------------------------
    {
        "gene": "STUB1",
        "alt_name": (
            "STUB1 (STUB1-303aa-16p13.3 / AD — SCA48-CHIP-E3-Ubiquitin-Ligase-Co-Chaperone — "
            "COGNITIVE-IMPAIRMENT-DOMINATES-FTD-Mimic — "
            "Parkinsonism-30pct-DaT-SPECT-Abnormal — "
            "Heterozygous-SCA48-AD-vs-Homozygous-SCAR16-AR-Different-Phenotypes)"
        ),
        "protein": (
            "STUB1 -- 16p13.3 AD -- STUB1-303aa -- "
            "CHIP-C-Terminus-Hsc70-Interacting-Protein-E3-Ubiquitin-Ligase-Co-Chaperone -- "
            "U-Box-Domain-Ubiquitin-Ligase-Activity-Misfolded-Protein-Degradation -- "
            "TPR-Domain-Hsc70-Hsp90-Binding-Protein-Triage-Decision -- "
            "Heterozygous-LOF-SCA48-AD-haploinsufficiency -- "
            "Homozygous-Compound-Het-SCAR16-AR-More-Severe-Childhood-Onset -- "
            "COGNITIVE-IMPAIRMENT-DOMINANT-FTD-Mimic-Frontal-Executive-Memory-Prominent -- "
            "Parkinsonism-30pct-Resting-Tremor-Rigidity-DaT-SPECT-Reduced-Striatal-Uptake -- "
            "Psychiatric-25pct-Anxiety-Depression-Behavioural-Changes -- "
            "Cerebellar-Ataxia-Often-AFTER-Cognitive-Symptoms-Neurogenetics-Referral-Delayed -- "
            "CHIP-Substrate-ATXN1-ATXN3-Tau-Alpha-Synuclein-Proteostasis-Network -- "
            "Recently-Described-Corben-2019-Genis-2018-Frequently-Missed"
        ),
        "locus": "16p13.3",
        "protein_size": "303 aa",
        "inheritance": (
            "AD (autosomal dominant; heterozygous haploinsufficiency → SCA48); "
            "AR (autosomal recessive; homozygous/compound het → SCAR16, distinct more severe phenotype); "
            "SCA48 = heterozygous ONLY; "
            "SCAR16 = biallelic; "
            "recently described 2018-2019; penetrance >95% for heterozygous"
        ),
        "age_of_onset": "Cognitive: 50-60 years; Ataxia: 55-70 years; late onset",
        "pathognomonic": (
            "COGNITIVE IMPAIRMENT DOMINATES PHENOTYPE — PATHOGNOMONIC for SCA48; "
            "Frontal-executive dysfunction + memory impairment mimics FTD; "
            "Cerebellar ataxia appears AFTER cognitive symptoms — often misdiagnosed as FTD/early dementia; "
            "Parkinsonism (30%): rest tremor, rigidity, bradykinesia; DaT-SPECT reduced in Parkinsonism variant; "
            "STUB1 heterozygous (AD SCA48) vs biallelic (AR SCAR16) — critical genotype-phenotype distinction"
        ),
        "treatment": (
            "COGNITIVE: neuropsychological assessment; acetylcholinesterase inhibitors (donepezil/rivastigmine) "
            "if dementia features; behavioural management; carer support; "
            "PARKINSONISM: LEVODOPA trial (modest response typical; DaT-SPECT to guide); "
            "PSYCHIATRIC: SSRI/SNRI for anxiety/depression; antipsychotics AVOID unless essential (sedation worsens ataxia); "
            "CEREBELLAR: RILUZOLE off-label (Level B); "
            "Genetic counselling: CRITICAL — distinguish AD SCA48 (50% risk) from AR SCAR16 (25% sibling risk); "
            "DaT-SPECT if Parkinsonism suspected; "
            "Brain MRI: cerebellar + frontoparietal cortical atrophy; "
            "STUB1 genetic testing: standard exome captures it but cognitive presentation delays referral"
        ),
        "key_biomarker": "Late-onset cognitive decline + ataxia + Parkinsonism → STUB1 (confirm heterozygous SCA48 vs biallelic SCAR16)",
        "critical_flags": [
            "STUB1-COGNITIVE-DOMINATES-FTD-MIMIC-PATHOGNOMONIC",
            "HETEROZYGOUS-AD-SCA48-vs-BIALLELIC-AR-SCAR16",
            "PARKINSONISM-30pct-DaT-SPECT-ABNORMAL",
            "CEREBELLAR-ATAXIA-AFTER-COGNITIVE-SYMPTOMS",
            "PSYCHIATRIC-FEATURES-25pct",
            "RECENTLY-DESCRIBED-2018-2019-MISSED",
            "LEVODOPA-TRIAL-PARKINSONISM",
            "DONEPEZIL-RIVASTIGMINE-COGNITIVE",
        ],
        "cohort_seed": SEED_BASE + 4,
    },
    # -- GRM1 — SCA44 ---------------------------------------------------------
    {
        "gene": "GRM1",
        "alt_name": (
            "GRM1 (GRM1-1194aa-6q24.3 / AD — SCA44-mGluR1-Metabotropic-Glutamate-Receptor-1 — "
            "Action-Tremor-PROMINENT-Before-Cerebellar-Ataxia-Misdiagnosed-ET — "
            "Gq-PKC-Purkinje-Cell-LTD-Pathway — "
            "mGluR1-Negative-Allosteric-Modulator-Trial-Ongoing)"
        ),
        "protein": (
            "GRM1 -- 6q24.3 AD -- GRM1-1194aa -- "
            "mGluR1-Metabotropic-Glutamate-Receptor-1-Group-1-Gq-Coupled-7TM -- "
            "Purkinje-Cell-Dendritic-Spine-PKC-DAG-IP3-Signalling-LTD-Induction -- "
            "Parallel-Fibre-Climbing-Fibre-Coincidence-Detection-Motor-Learning -- "
            "Heterozygous-GOF-Constitutive-mGluR1-Activation-Purkinje-Cell-Excitotoxicity -- "
            "pAla168Val-French-Family-S1-Domain-Ligand-Binding-Loss -- "
            "ACTION-TREMOR-Hands-Prominent-Before-Cerebellar-Ataxia-ET-Misdiagnosis -- "
            "Adult-Onset-4th-6th-Decade-Slow-Progression -- "
            "Cerebellar-Atrophy-MRI-Brainstem-Spared -- "
            "Negative-Allosteric-Modulator-BAY-357-Preclinical-Trials -- "
            "Very-Rare-Few-Families-Worldwide"
        ),
        "locus": "6q24.3",
        "protein_size": "1194 aa",
        "inheritance": (
            "AD (autosomal dominant); loss-of-function + gain-of-function variants both reported; "
            "very rare — few families; "
            "p.Ala168Val described in French family; "
            "heterozygous; penetrance appears high"
        ),
        "age_of_onset": "40-65 years; action tremor often first",
        "pathognomonic": (
            "ACTION TREMOR PROMINENT BEFORE CEREBELLAR ATAXIA — misdiagnosed as Essential Tremor; "
            "Postural + kinetic tremor hands years before obvious ataxia; "
            "Eventually develops dysmetria, dysarthria, gait ataxia; "
            "mGluR1 involvement: PKC-LTD pathway impairment; "
            "No other pathognomonic sign; very rare gene — diagnosed on panel/exome"
        ),
        "treatment": (
            "ACTION TREMOR: PROPRANOLOL 40-120 mg/day; primidone 50-250 mg/day; "
            "DBS VIM if disabling refractory tremor; "
            "CEREBELLAR: RILUZOLE off-label; "
            "mGluR1 negative allosteric modulators (BAY-357, MPEP): experimental/trial only; "
            "SARA biannually; "
            "Physiotherapy and SLT; "
            "Genetic counselling: 50% risk; "
            "MRI at diagnosis (baseline cerebellar atrophy)"
        ),
        "key_biomarker": "Action tremor (ET misdiagnosis) + later cerebellar ataxia → GRM1 (mGluR1) panel/exome",
        "critical_flags": [
            "GRM1-ACTION-TREMOR-BEFORE-ATAXIA-PATHOGNOMONIC",
            "ET-MISDIAGNOSIS-TREMOR-FIRST",
            "mGluR1-Gq-PKC-LTD-MECHANISM",
            "mGluR1-NAM-TRIAL-ONGOING",
            "PROPRANOLOL-PRIMIDONE-TREMOR",
            "DBS-VIM-REFRACTORY-TREMOR",
            "VERY-RARE-FEW-FAMILIES",
            "EXOME-PANELS-CAPTURE-GRM1",
        ],
        "cohort_seed": SEED_BASE + 5,
    },
    # -- FAT2 — SCA45 ---------------------------------------------------------
    {
        "gene": "FAT2",
        "alt_name": (
            "FAT2 (FAT2-4589aa-5q33.1 / AD — SCA45-FAT-Atypical-Cadherin-2-Protocadherin — "
            "Adult-Onset-Pure-Cerebellar-Slow-Progression — "
            "Planar-Cell-Polarity-Purkinje-Cell-Axon-Guidance — "
            "Normal-Life-Expectancy-No-Pathognomonic-Sign)"
        ),
        "protein": (
            "FAT2 -- 5q33.1 AD -- FAT2-4589aa -- "
            "FAT-Atypical-Cadherin-2-Protocadherin-Large-Extracellular-Cadherin-Repeat-Domain -- "
            "Planar-Cell-Polarity-Wnt-Non-Canonical-Cerebellar-Organisation -- "
            "Purkinje-Cell-Axon-Guidance-Granule-Cell-Migration-Regulated -- "
            "Heterozygous-LOF-Missense-Truncating-Haploinsufficiency -- "
            "pArg2745Gln-First-Pathogenic-Variant-Brazilian-Family-Renaud-2017-JAMA-Neurol -- "
            "PURE-CEREBELLAR-Ataxia-Dysarthria-Adult-Onset-40s-50s -- "
            "Very-Slow-Progression-Normal-Life-Expectancy -- "
            "MRI-Cerebellar-Cortical-Atrophy-Pons-Spared -- "
            "Few-Families-Worldwide-Very-Rare"
        ),
        "locus": "5q33.1",
        "protein_size": "4589 aa",
        "inheritance": (
            "AD (autosomal dominant); haploinsufficiency; "
            "p.Arg2745Gln first pathogenic variant (Brazilian family); "
            "few families; high penetrance when pathogenic variant confirmed; "
            "de novo variants reported"
        ),
        "age_of_onset": "40-55 years; adult onset",
        "pathognomonic": (
            "No pathognomonic sign — PURE CEREBELLAR syndrome; "
            "Gait ataxia, limb dysmetria, dysarthria without extracerebellar features; "
            "Diagnosed on SCA gene panel/exome (clinically indistinguishable from SCA5/CACNA1G); "
            "MRI: cerebellar cortical atrophy, pons and brainstem spared; "
            "Slow progression, normal life expectancy"
        ),
        "treatment": (
            "No disease-modifying therapy; "
            "RILUZOLE off-label (Level B neuroprotection SCAs); "
            "Physiotherapy: balance, gait, fall prevention; "
            "SLT: dysarthria; "
            "Occupational therapy as needed; "
            "SARA biannually; MRI at diagnosis then if acceleration; "
            "Genetic counselling: 50% risk per child"
        ),
        "key_biomarker": "Adult pure cerebellar ataxia + Brazilian/South American ancestry → include FAT2 on panel",
        "critical_flags": [
            "FAT2-PURE-CEREBELLAR-SCA45",
            "ADULT-ONSET-SLOW-PROGRESSION",
            "PROTOCADHERIN-CELL-POLARITY-MECHANISM",
            "NORMAL-LIFE-EXPECTANCY",
            "NO-PATHOGNOMONIC-SIGN-PANEL-DIAGNOSIS",
            "BRAZILIAN-FOUNDER-pArg2745Gln",
            "MRI-CEREBELLAR-PONS-SPARED",
            "RILUZOLE-LEVEL-B-TRIAL",
        ],
        "cohort_seed": SEED_BASE + 6,
    },
    # -- PUM1 — SCA47 ---------------------------------------------------------
    {
        "gene": "PUM1",
        "alt_name": (
            "PUM1 (PUM1-1186aa-1p35.2 / AD — SCA47-Pumilio-1-RNA-Binding-Translational-Repressor — "
            "Developmental-Delay-Intellectual-Disability-PATHOGNOMONIC — "
            "Non-Progressive-Or-Slowly-Progressive-Cerebellar-Ataxia — "
            "Seizures-30pct-Cerebellar-Hypoplasia-Not-Atrophy)"
        ),
        "protein": (
            "PUM1 -- 1p35.2 AD -- PUM1-1186aa -- "
            "Pumilio-1-PUF-Domain-RNA-Binding-Protein-Translational-Repressor -- "
            "PUF-Domain-Binds-PRE-Pumilio-Recognition-Element-mRNA-3UTR -- "
            "Represses-ATXN1-ATXN2-EAAT4-mRNAs-Purkinje-Cell-Homeostasis -- "
            "Haploinsufficiency-Derepresses-ATXN1-SCA1-Modifier-Effect -- "
            "DEVELOPMENTAL-DELAY-INTELLECTUAL-DISABILITY-Mild-To-Moderate-PATHOGNOMONIC -- "
            "Non-Progressive-Or-Slowly-Progressive-Cerebellar-Ataxia -- "
            "Childhood-Onset-Gait-Ataxia-Hypotonia -- "
            "SEIZURES-30pct-Infantile-Spasms-Or-Focal -- "
            "MRI-Cerebellar-HYPOPLASIA-Not-Atrophy-Vermis-Prominent -- "
            "Rare-Haploinsufficiency-Mechanism-Unlike-Other-SCAs"
        ),
        "locus": "1p35.2",
        "protein_size": "1186 aa",
        "inheritance": (
            "AD (autosomal dominant); haploinsufficiency; "
            "rare; "
            "de novo variants reported; "
            "PUM1 represses ATXN1/ATXN2/EAAT4; "
            "heterozygous LOF sufficient for phenotype; "
            "penetrance high"
        ),
        "age_of_onset": "Childhood/early developmental; ataxia apparent by age 5-10",
        "pathognomonic": (
            "DEVELOPMENTAL DELAY / INTELLECTUAL DISABILITY PATHOGNOMONIC early — not typical for adult SCAs; "
            "Mild-moderate ID + childhood ataxia distinguishes from adult-onset SCAs; "
            "Seizures in 30% (infantile spasms, focal); "
            "MRI: cerebellar HYPOPLASIA (not progressive atrophy) — structural not degenerative; "
            "Non-progressive or very slowly progressive course"
        ),
        "treatment": (
            "SEIZURES: VIGABATRIN for infantile spasms (first-line); "
            "LEVETIRACETAM for focal seizures (first-line); "
            "VPA for broad spectrum (second-line); "
            "INTELLECTUAL DISABILITY: educational intervention; special needs support; "
            "ATAXIA: physiotherapy — balance, gait; adaptations; "
            "SLT: dysarthria + language delay; "
            "Occupational therapy; "
            "ACTH if infantile spasms not responding; "
            "EEG at diagnosis + if new seizure type; "
            "MRI brain (cerebellar hypoplasia); "
            "Genetic counselling: de novo vs inherited; 50% if inherited"
        ),
        "key_biomarker": "Childhood ataxia + ID + cerebellar hypoplasia on MRI ± seizures → PUM1 panel/exome",
        "critical_flags": [
            "PUM1-DEVELOPMENTAL-DELAY-ID-PATHOGNOMONIC",
            "CHILDHOOD-ONSET-NOT-ADULT-SCA",
            "CEREBELLAR-HYPOPLASIA-NOT-ATROPHY",
            "SEIZURES-30pct-INFANTILE-SPASMS",
            "NON-PROGRESSIVE-OR-SLOW",
            "ATXN1-ATXN2-REPRESSOR-MECHANISM",
            "VIGABATRIN-INFANTILE-SPASMS",
            "EDUCATIONAL-SUPPORT-MANDATORY",
        ],
        "cohort_seed": SEED_BASE + 7,
    },
]


def _generate_cohort(gene_entry):
    """Generate a 40-patient synthetic cohort for a non-polyQ dominant SCA gene."""
    rng = random.Random(gene_entry["cohort_seed"])
    gene = gene_entry["gene"]
    inh = gene_entry["inheritance"]
    onset_range = gene_entry["age_of_onset"]

    patients = []
    for i in range(1, 41):
        pid = f"{gene}-P{i:02d}"

        # Age of onset per gene
        if gene == "SPTBN2":
            onset = rng.randint(28, 52)
        elif gene == "CACNA1G":
            onset = rng.randint(38, 58)
        elif gene == "KCND3":
            onset = rng.randint(30, 58)
        elif gene == "TMEM240":
            onset = rng.randint(18, 42)
        elif gene == "STUB1":
            onset = rng.randint(48, 68)
        elif gene == "GRM1":
            onset = rng.randint(38, 65)
        elif gene == "FAT2":
            onset = rng.randint(38, 58)
        else:  # PUM1
            onset = rng.randint(2, 12)

        age = onset + rng.randint(2, 25)
        sara_score = round(rng.uniform(3.0, 28.0), 1)
        sex = rng.choice(["M", "F", "M", "F"])  # slight 50/50

        # Gene-specific clinical features
        if gene == "SPTBN2":
            cognitive_impairment = rng.random() < 0.05
            myoclonus = rng.random() < 0.02
            parkinsonism = rng.random() < 0.03
            retinal_involvement = False
            seizures = rng.random() < 0.02
            tremor_prominent = rng.random() < 0.10
            slow_progression = True
            childhood_id = False
        elif gene == "CACNA1G":
            cognitive_impairment = rng.random() < 0.08
            myoclonus = rng.random() < 0.03
            parkinsonism = rng.random() < 0.04
            retinal_involvement = False
            seizures = rng.random() < 0.02
            tremor_prominent = rng.random() < 0.15
            slow_progression = True
            childhood_id = False
        elif gene == "KCND3":
            cognitive_impairment = rng.random() < 0.45
            myoclonus = rng.random() < 0.25
            parkinsonism = rng.random() < 0.05
            retinal_involvement = False
            seizures = rng.random() < 0.05
            tremor_prominent = rng.random() < 0.40
            slow_progression = True
            childhood_id = False
        elif gene == "TMEM240":
            cognitive_impairment = rng.random() < 0.75
            myoclonus = rng.random() < 0.05
            parkinsonism = rng.random() < 0.30
            retinal_involvement = False
            seizures = rng.random() < 0.05
            tremor_prominent = rng.random() < 0.30
            slow_progression = True
            childhood_id = rng.random() < 0.70  # cognitive in childhood
        elif gene == "STUB1":
            cognitive_impairment = rng.random() < 0.92
            myoclonus = rng.random() < 0.05
            parkinsonism = rng.random() < 0.30
            retinal_involvement = False
            seizures = rng.random() < 0.05
            tremor_prominent = rng.random() < 0.20
            slow_progression = True
            childhood_id = False
        elif gene == "GRM1":
            cognitive_impairment = rng.random() < 0.10
            myoclonus = rng.random() < 0.05
            parkinsonism = rng.random() < 0.05
            retinal_involvement = False
            seizures = rng.random() < 0.02
            tremor_prominent = rng.random() < 0.85
            slow_progression = True
            childhood_id = False
        elif gene == "FAT2":
            cognitive_impairment = rng.random() < 0.05
            myoclonus = rng.random() < 0.02
            parkinsonism = rng.random() < 0.03
            retinal_involvement = False
            seizures = rng.random() < 0.02
            tremor_prominent = rng.random() < 0.08
            slow_progression = True
            childhood_id = False
        else:  # PUM1
            cognitive_impairment = rng.random() < 0.95
            myoclonus = rng.random() < 0.10
            parkinsonism = rng.random() < 0.02
            retinal_involvement = False
            seizures = rng.random() < 0.30
            tremor_prominent = rng.random() < 0.10
            slow_progression = rng.random() < 0.70
            childhood_id = True

        patients.append({
            "patient_id": pid,
            "gene": gene,
            "sex": sex,
            "onset_age": onset,
            "current_age": age,
            "sara_score": sara_score,
            "cognitive_impairment": cognitive_impairment,
            "myoclonus": myoclonus,
            "parkinsonism": parkinsonism,
            "tremor_prominent": tremor_prominent,
            "seizures": seizures,
            "childhood_id": childhood_id,
            "slow_progression": slow_progression,
        })
    return patients


# Pre-generate all cohorts
_ALL_COHORTS = {g["gene"]: _generate_cohort(g) for g in NONPOLYQ_DOMINANT_SCA_GENES}
_ALL_PATIENTS = [p for cohort in _ALL_COHORTS.values() for p in cohort]


def overview():
    """Aggregate 8-gene overview."""
    pts = _ALL_PATIENTS
    n = len(pts)
    return {
        "atlas": "Hereditary Non-Polyglutamine Dominant SCA Atlas",
        "subtitle": "8-Gene Non-Polyglutamine Autosomal-Dominant Spinocerebellar Ataxia Reference (SCA5/SCA19/SCA21/SCA42/SCA44/SCA45/SCA47/SCA48)",
        "genes": [g["gene"] for g in NONPOLYQ_DOMINANT_SCA_GENES],
        "total_patients": n,
        "seeds": f"{SEED_BASE}-{SEED_BASE+7}",
        # Key clinical flags
        "stub1_cognitive_ftd_mimic_patients": sum(1 for p in pts if p["gene"] == "STUB1" and p["cognitive_impairment"]),
        "kcnd3_cognitive_psychiatric_patients": sum(1 for p in pts if p["gene"] == "KCND3" and p["cognitive_impairment"]),
        "tmem240_childhood_cognitive_patients": sum(1 for p in pts if p["gene"] == "TMEM240" and p["childhood_id"]),
        "kcnd3_myoclonus_patients": sum(1 for p in pts if p["gene"] == "KCND3" and p["myoclonus"]),
        "pum1_seizure_patients": sum(1 for p in pts if p["gene"] == "PUM1" and p["seizures"]),
        "pum1_childhood_id_patients": sum(1 for p in pts if p["gene"] == "PUM1" and p["childhood_id"]),
        "stub1_parkinsonism_patients": sum(1 for p in pts if p["gene"] == "STUB1" and p["parkinsonism"]),
        "grm1_tremor_prominent_patients": sum(1 for p in pts if p["gene"] == "GRM1" and p["tremor_prominent"]),
        "kcnd3_tremor_patients": sum(1 for p in pts if p["gene"] == "KCND3" and p["tremor_prominent"]),
        "cognitive_any_gene_patients": sum(1 for p in pts if p["cognitive_impairment"]),
        "et_misdiagnosis_risk_patients": sum(1 for p in pts if p["gene"] in ("GRM1", "KCND3") and p["tremor_prominent"]),
        "ftd_misdiagnosis_risk_patients": sum(1 for p in pts if p["gene"] == "STUB1" and p["cognitive_impairment"]),
    }


def breakdown():
    """Per-gene breakdown data."""
    result = {}
    for g in NONPOLYQ_DOMINANT_SCA_GENES:
        gene = g["gene"]
        cohort = _ALL_COHORTS[gene]
        n = len(cohort)
        result[gene] = {
            "gene": gene,
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "age_of_onset": g["age_of_onset"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "key_biomarker": g["key_biomarker"],
            "critical_flags": g["critical_flags"],
            "patient_count": n,
            "mean_sara": round(sum(p["sara_score"] for p in cohort) / n, 1),
            "mean_onset_age": round(sum(p["onset_age"] for p in cohort) / n, 1),
            "cognitive_pct": round(100 * sum(1 for p in cohort if p["cognitive_impairment"]) / n, 1),
            "myoclonus_pct": round(100 * sum(1 for p in cohort if p["myoclonus"]) / n, 1),
            "parkinsonism_pct": round(100 * sum(1 for p in cohort if p["parkinsonism"]) / n, 1),
            "tremor_prominent_pct": round(100 * sum(1 for p in cohort if p["tremor_prominent"]) / n, 1),
            "seizures_pct": round(100 * sum(1 for p in cohort if p["seizures"]) / n, 1),
            "childhood_id_pct": round(100 * sum(1 for p in cohort if p["childhood_id"]) / n, 1),
        }
    return result


def definitions():
    """Gene definitions, glossary, and surveillance protocols."""
    return {
        "genes": {
            g["gene"]: g["protein"]
            for g in NONPOLYQ_DOMINANT_SCA_GENES
        },
        "glossary": {
            "Non-Polyglutamine SCA": (
                "Spinocerebellar ataxias caused by mechanisms other than polyglutamine (CAG) repeat expansion. "
                "Includes missense mutations in ion channels, cytoskeletal proteins, receptor subunits, and RNA-binding proteins. "
                "Usually detected on standard gene panels or WES (unlike some repeat-expansion SCAs). "
                "Typically slower progression than polyglutamine SCAs."
            ),
            "SCA48 / STUB1": (
                "Heterozygous STUB1 (CHIP) = SCA48 (AD), cognitive-dominant, late onset. "
                "Biallelic STUB1 = SCAR16 (AR), more severe, childhood onset. "
                "CRITICAL: genotype determines phenotype — report zygosity explicitly."
            ),
            "CHIP / STUB1": (
                "C-terminus of Hsc70-Interacting Protein: E3 ubiquitin ligase + co-chaperone. "
                "Triages misfolded proteins between refolding (Hsc70) and degradation (UPS). "
                "Substrates include tau, alpha-synuclein, ATXN1, ATXN3 — links SCA48 to neurodegeneration."
            ),
            "Kv4.3 / KCND3": (
                "Voltage-gated A-type potassium channel (rapidly inactivating). "
                "Repolarises Purkinje cells after action potentials. "
                "Loss-of-function → prolonged firing, excitotoxicity. "
                "SCA19 (Dutch/Norwegian) and SCA22 (Japanese/Asian) = same KCND3 gene."
            ),
            "mGluR1 / GRM1": (
                "Metabotropic glutamate receptor 1 — Gq-coupled 7TM receptor on Purkinje cells. "
                "Activated by parallel fibre glutamate; triggers PKC signalling + LTD. "
                "Critical for Purkinje cell motor learning and cerebellar gain control. "
                "Negative allosteric modulators (NAMs) are investigational treatments."
            ),
            "EAAT4 / SLC1A1": (
                "Glutamate transporter on Purkinje cell dendritic spines. "
                "Cleared from synapse by SPTBN2 (spectrin-βIII). "
                "SCA5 (SPTBN2 loss) → EAAT4 destabilisation → glutamate excitotoxicity."
            ),
            "PUM1 (Pumilio 1)": (
                "RNA-binding translational repressor — binds PRE sequences in 3'UTR. "
                "Represses ATXN1, ATXN2, EAAT4 mRNAs in Purkinje cells. "
                "Haploinsufficiency derepresses these targets — links to polyQ SCAs mechanistically."
            ),
            "Cav3.1 / CACNA1G": (
                "Low-voltage-activated T-type calcium channel alpha-1G subunit. "
                "Mediates burst firing and rhythmogenesis in Purkinje cells and thalamus. "
                "SCA42 gain-of-function → prolonged channel activation → Ca overload Purkinje cells."
            ),
            "FAT2 Protocadherin": (
                "Large cell adhesion molecule with 34 cadherin repeats + intracellular signalling domain. "
                "Regulates planar cell polarity and Wnt non-canonical signalling in cerebellum. "
                "Haploinsufficiency disrupts Purkinje axon guidance and cerebellar circuitry assembly."
            ),
            "SARA (Scale for Assessment and Rating of Ataxia)": (
                "0-40 scale: gait/stance/sitting/speech/finger-chase/nose-finger/fast alternating/heel-shin. "
                "Lower = better. Used biannually in SCA clinics. "
                "Non-polyQ SCAs typically show very slow SARA progression (0.5-1 pt/year)."
            ),
        },
        "surveillance_protocols": {
            "SPTBN2 (SCA5)": (
                "SPTBN2 sequencing (standard SCA gene panel includes it); "
                "SARA biannually; "
                "Physiotherapy: balance + gait programme, fall risk annually; "
                "SLT: dysarthria (baseline + if worsening); "
                "Occupational therapy when ATM/driving affected; "
                "MRI brain at diagnosis (cerebellar cortical atrophy pattern); "
                "Lincoln family — genetic counselling: 50% risk per child; "
                "Driving assessment: SCA5 often very slow — can drive for 20+ yr from onset; "
                "EUROSCA registry enrolment"
            ),
            "CACNA1G (SCA42)": (
                "CACNA1G sequencing; report specific variant (p.Arg1715His French founder vs other); "
                "FLUNARIZINE 10 mg/day trial if pathogenic GOF variant — monitor for sedation; "
                "SARA biannually; "
                "MRI brain (baseline + if symptom acceleration); "
                "French/French-Canadian ancestry: community cascade testing; "
                "Genetic counselling: 50% risk; "
                "Physiotherapy and SLT; "
                "Register in EudraCT / SCA42 international registry"
            ),
            "KCND3 (SCA19/22)": (
                "KCND3 sequencing; report ethnic founder (p.Val406Ile Dutch vs p.Arg293Cys Japanese); "
                "Neuropsychological assessment at diagnosis (cognitive profile); "
                "MoCA annually; neuropsychology 2-yearly; "
                "EEG if myoclonus suspected (cortical myoclonus pattern); "
                "LEVETIRACETAM if myoclonus; PROPRANOLOL if action tremor; "
                "SSRI/SNRI for depression/anxiety component; "
                "SARA biannually; "
                "Physiotherapy, SLT; "
                "Genetic counselling: 50% risk; cognitive features in at-risk relatives"
            ),
            "TMEM240 (SCA21)": (
                "TMEM240 sequencing; "
                "Neuropsychological assessment at diagnosis (ADHD/dyslexia/cognitive profile); "
                "School-age at-risk relatives: paediatric neuropsychology if symptoms; "
                "LEVODOPA trial if extrapyramidal features (modest response); "
                "SARA biannually once ataxia manifests; MoCA annually; "
                "MRI brain (cerebellar cortical atrophy); "
                "Genetic counselling: 50% risk; cognitive symptoms may precede ataxia by decades; "
                "Physiotherapy and SLT"
            ),
            "STUB1 (SCA48)": (
                "STUB1 sequencing; REPORT ZYGOSITY — heterozygous (AD SCA48) vs biallelic (AR SCAR16); "
                "Neuropsychological assessment at diagnosis (FTD-like cognitive profile); "
                "DaT-SPECT if Parkinsonism suspected (reduced striatal uptake); "
                "MRI brain (cerebellar + frontoparietal cortical atrophy); "
                "LEVODOPA trial for Parkinsonism (modest response); "
                "Donepezil/rivastigmine for dementia features; "
                "SSRI/SNRI for psychiatric component; "
                "Genetic counselling: AD SCA48 = 50% risk per child; AR SCAR16 = 25% sibling risk; "
                "Carer support: cognitive-dominant phenotype mimics FTD — specialist carer education; "
                "SARA biannually"
            ),
            "GRM1 (SCA44)": (
                "GRM1 sequencing; "
                "Neurological exam: document tremor type (action/postural) and onset vs ataxia; "
                "PROPRANOLOL 40-120 mg/day for action tremor; primidone if propranolol ineffective; "
                "DBS VIM referral if disabling refractory tremor; "
                "SARA biannually once ataxia manifests; "
                "MRI brain (cerebellar atrophy); "
                "mGluR1 NAM trial eligibility screening (if available); "
                "Physiotherapy and SLT; "
                "Genetic counselling: 50% risk"
            ),
            "FAT2 (SCA45)": (
                "FAT2 sequencing (standard SCA gene panel); "
                "SARA biannually; "
                "Physiotherapy: balance, gait, falls; SLT: dysarthria; "
                "Occupational therapy for ADL; "
                "MRI brain at diagnosis; "
                "Genetic counselling: 50% risk per child; "
                "RILUZOLE off-label (Level B neuroprotection); "
                "Normal life expectancy — communicate positive prognosis"
            ),
            "PUM1 (SCA47)": (
                "PUM1 sequencing; "
                "EEG at diagnosis (infantile spasms or focal seizure pattern); "
                "VIGABATRIN for infantile spasms; LEVETIRACETAM for focal; "
                "Developmental assessment: BSID or Griffiths at diagnosis; "
                "Educational support plan: special education / IEP if ID; "
                "Physiotherapy: motor skills, gait, balance; "
                "SLT: language delay + dysarthria; "
                "Occupational therapy for self-care; "
                "MRI brain (cerebellar hypoplasia — structural, not progressive atrophy); "
                "Genetic counselling: de novo vs inherited; 50% risk if inherited; "
                "ACTH if infantile spasms unresponsive to vigabatrin; "
                "Annual review: seizure control + developmental progress"
            ),
        },
    }


if __name__ == "__main__":
    import json
    ov = overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"STUB1 cognitive (FTD-mimic): {ov['stub1_cognitive_ftd_mimic_patients']}")
    print(f"KCND3 cognitive+psychiatric: {ov['kcnd3_cognitive_psychiatric_patients']}")
    print(f"TMEM240 childhood cognitive: {ov['tmem240_childhood_cognitive_patients']}")
    print(f"KCND3 myoclonus: {ov['kcnd3_myoclonus_patients']}")
    print(f"PUM1 seizures: {ov['pum1_seizure_patients']}")
    print(f"PUM1 childhood ID: {ov['pum1_childhood_id_patients']}")
    print(f"STUB1 parkinsonism: {ov['stub1_parkinsonism_patients']}")
    print(f"GRM1 tremor prominent: {ov['grm1_tremor_prominent_patients']}")
    print(f"ET misdiagnosis risk: {ov['et_misdiagnosis_risk_patients']}")
    print(f"FTD misdiagnosis risk: {ov['ftd_misdiagnosis_risk_patients']}")
