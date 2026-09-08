#!/usr/bin/env python3
"""Hereditary-Chorea-Atlas — Complete 8-Gene Hereditary Chorea Atlas.

HTT     (Huntingtin; 3144 aa; 4p16.3; AD;
          Huntington Disease (HD) — CAG≥36 pathogenic; CAG≥60 juvenile (<21 yr);
          tetrabenazine/deutetrabenazine FDA-approved; HD is THE model of polyglutamine disease;
          seed SEED_BASE+0).
VPS13A  (Vacuolar protein sorting 13A; 3174 aa; 9q21.2; AR;
          Chorea-Acanthocytosis (ChAc) — acanthocytes on peripheral blood film PATHOGNOMONIC;
          elevated serum CK; tongue/lip biting; feeding dystonia; McLeod overlap;
          seed SEED_BASE+1).
PANK2   (Pantothenate kinase 2; 570 aa; 20p13; AR;
          PKAN (Pantothenate Kinase-Associated Neurodegeneration) — NBIA1;
          "eye-of-the-tiger" sign on T2/FLAIR MRI PATHOGNOMONIC (GPi central hyperintensity);
          pantethine trial; iron chelation; seed SEED_BASE+2).
WDR45   (WD repeat domain 45 / WIPI4; 330 aa; Xp11.23; XL (X-linked dominant de novo);
          BPAN (Beta-propeller protein-associated neurodegeneration) — NBIA5;
          early epilepsy + intellectual disability → adult parkinsonism-dementia;
          predominantly females (de novo dominant); seed SEED_BASE+3).
FTL     (Ferritin light chain; 175 aa; 19q13.33; AD;
          Hereditary Ferritinopathy / Neuroferritinopathy (HF);
          LOW serum ferritin PARADOX (iron accumulates in brain NOT serum);
          adult-onset chorea + parkinsonism + dementia; seed SEED_BASE+4).
NKX2-1  (NK2 homeobox 1 / TTF-1; 401 aa; 14q13.3; AD;
          Benign Hereditary Chorea (BHC) — brain-lung-thyroid syndrome;
          childhood chorea + thyroid dysfunction (70%) + lung disease (54%);
          NOT progressive; good prognosis; seed SEED_BASE+5).
JPH3    (Junctophilin 3; 741 aa; 16q24.2; AD;
          Huntington Disease-Like 2 (HDL2) — CTG/CAG repeat 16q24.3;
          predominantly African ancestry; psychiatric features; acanthocytes in 50%;
          clinically indistinguishable from HD → test HTT first; seed SEED_BASE+6).
TBP     (TATA-binding protein; 339 aa; 6q27; AD;
          SCA17 / Huntington Disease-Like 4 (HDL4) — polyglutamine CAG repeat;
          broad phenotype: cerebellar ataxia + chorea + dementia + parkinsonism;
          CAG≥49 pathogenic; HD-like presentation with ataxia distinguishes;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2078-2085).
"""

import random

SEED_BASE = 2078

CHOREA_GENES = [
    # -- HTT — Huntington Disease ----------------------------------------------------
    {
        "gene": "HTT",
        "alt_name": (
            "HTT (HTT-3144aa-4p16.3 / AD — Huntington-Disease — "
            "CAG-Repeat-Expansion-PolyQ-IT15-Locus — CAG≥36-Pathogenic-CAG≥60-Juvenile — "
            "Tetrabenazine-Deutetrabenazine-FDA-Approved — Huntingtin-3144aa)"
        ),
        "protein": (
            "HTT -- 4p16.3 AD -- HTT-3144aa -- "
            "Huntingtin-Large-HEAT-Repeat-Scaffold-Protein-N-terminal-PolyQ-Tract -- "
            "Huntington-Disease-HD-CAG-Trinucleotide-Repeat-Expansion-IT15 -- "
            "CAG≥36-Pathogenic-CAG36-39-Reduced-Penetrance-CAG≥40-Full-Penetrance -- "
            "CAG≥60-Juvenile-Onset-<21yr-Rigidity-Dominant-NOT-Chorea -- "
            "Autosomal-Dominant-Toxic-Gain-of-Function-PolyQ-Aggregation -- "
            "Striatal-MSN-Degeneration-Caudate-Putamen-Selective-Neuronal-Loss -- "
            "Tetrabenazine-VMAT2-Inhibitor-FDA-Approved-Chorea-HTT-ONLY -- "
            "Deutetrabenazine-Valbenazine-Second-Generation-VMAT2-Inhibitors"
        ),
        "locus": "4p16.3",
        "protein_size": "3144 aa",
        "inheritance": (
            "AD (autosomal dominant) — toxic gain-of-function PolyQ aggregation; "
            "CAG≥36 pathogenic; CAG36-39 reduced penetrance; CAG≥40 fully penetrant; "
            "CAG≥60 juvenile onset (<21 yr) — rigid akinetic not choreic; "
            "Paternal transmission → anticipation (CAG expansion in male meiosis); "
            "Maternal transmission → relatively stable repeat; "
            "De novo expansions from intermediate alleles (CAG27-35) — counsel families; "
            "Prevalence: 5-10/100,000 Western populations; "
            "Presymptomatic testing: mandatory pre-test counselling (Huntington Protocol)"
        ),
        "pathognomonic": (
            "CHOREA — involuntary, unpredictable, flowing movements, worst trunk/limb/face; "
            "Cognitive decline (frontal-subcortical); psychiatric (depression, irritability, apathy) often FIRST; "
            "Motor impersistence: cannot sustain tongue protrusion (milkmaid's grip); "
            "CAUDATE ATROPHY on MRI — enlarged frontal horns (boxer's caudate); "
            "Gait disturbance: prancing/dancing; choking from dysphagia (late)"
        ),
        "treatment": (
            "**Tetrabenazine** (VMAT2 inhibitor, FDA 2008) — chorea-specific, HTT ONLY; start 12.5mg; "
            "**Deutetrabenazine** (FDA 2017) — smoother pharmacokinetics, preferred; "
            "**Valbenazine** (FDA 2023) — once-daily; "
            "**Antipsychotics** (olanzapine, risperidone) — psychiatric + chorea; "
            "**Multidisciplinary**: physiotherapy, speech/swallowing therapy, dietitian, psychiatry; "
            "**No disease-modifying** treatment approved (ASO/gene silencing trials ongoing)"
        ),
        "critical_flags": [
            "CAG≥36-PATHOGNOMONIC-MOLECULAR-DIAGNOSIS",
            "CAG≥60-JUVENILE-ONSET-RIGID-NOT-CHOREIC",
            "TETRABENAZINE-HTT-ONLY-APPROVED",
            "PRESYMPTOMATIC-TESTING-MANDATORY-COUNSELLING-PROTOCOL",
            "PATERNAL-TRANSMISSION-ANTICIPATION",
            "CAUDATE-ATROPHY-MRI-BOXER-CAUDATE",
            "PSYCHIATRIC-FEATURES-OFTEN-FIRST",
            "SUICIDE-RISK-HIGHEST-EARLY-MIDDLE-STAGE",
        ],
        "age_of_onset": "Adult: 30-50 yr (modal 40 yr); Juvenile: <21 yr (CAG≥60) — rigid akinetic; Presymptomatic: 15-20 yr before motor onset",
        "key_biomarker": "CAG repeat count (IT15 locus); Neurofilament light chain (NfL) elevated pre-symptomatically; Caudate atrophy on MRI",
        "seed_offset": 0,
    },

    # -- VPS13A — Chorea-Acanthocytosis ---------------------------------------------
    {
        "gene": "VPS13A",
        "alt_name": (
            "VPS13A (VPS13A-3174aa-9q21.2 / AR — Chorea-Acanthocytosis-ChAc — "
            "Acanthocytes-Peripheral-Blood-Film-PATHOGNOMONIC — Elevated-CK-PATHOGNOMONIC — "
            "Tongue-Lip-Biting-Feeding-Dystonia — McLeod-Overlap-XK-Gene)"
        ),
        "protein": (
            "VPS13A -- 9q21.2 AR -- VPS13A-3174aa -- "
            "Vacuolar-Protein-Sorting-13A-Large-Scaffold-Lipid-Transfer-Protein -- "
            "Membrane-Contact-Site-ER-Mitochondria-Golgi-Lipid-Homeostasis -- "
            "Chorea-Acanthocytosis-ChAc-Neuroacanthocytosis-Syndrome -- "
            "Biallelic-Loss-of-Function-AR-Autosomal-Recessive -- "
            "Striatum-Caudate-Putamen-Degeneration-Spiny-Neurons -- "
            "Acanthocytes-Spiculated-Erythrocytes-Peripheral-Blood-Film -- "
            "Elevated-CK-Creatine-Kinase-Serum-Marker -- "
            "Tongue-Lip-Self-Mutilation-Involuntary-Biting-Feeding-Dystonia-CLASSIC"
        ),
        "locus": "9q21.2",
        "protein_size": "3174 aa",
        "inheritance": (
            "AR (autosomal recessive) — biallelic loss-of-function VPS13A variants; "
            "Consanguinity increases risk; Prevalence: rare (~500 cases worldwide); "
            "Onset typically 20s-40s; "
            "McLeod syndrome (XK gene, X-linked) overlaps: acanthocytes + chorea + elevated CK; "
            "Distinguish: XK (X-linked, Kell antigen weak) vs VPS13A (AR, normal Kell); "
            "No anticipation; de novo rare"
        ),
        "pathognomonic": (
            "ACANTHOCYTES on peripheral blood film (spiculated red cells) — PATHOGNOMONIC (>3%); "
            "Elevated SERUM CK — PATHOGNOMONIC (myopathic despite no myopathy); "
            "Tongue/lip BITING — self-mutilation; involuntary tongue protrusion with biting; "
            "Feeding DYSTONIA — involuntary expulsion of food during eating; "
            "Orofacial dyskinesia + chorea + tics; Caudate atrophy MRI; "
            "SEIZURES in 50% (frontal lobe type)"
        ),
        "treatment": (
            "**No disease-modifying treatment** available; "
            "**Tetrabenazine/Deutetrabenazine** — chorea management (off-label); "
            "**Antipsychotics** (haloperidol low-dose) — chorea + behavioural; "
            "**Valproate** — seizure control; "
            "**Botulinum toxin** — orolingual dystonia, biting; dental protection; "
            "**Nutritional support** — feeding dystonia management; "
            "**Pacemaker** — cardiac arrhythmia surveillance (cardiomyopathy risk)"
        ),
        "critical_flags": [
            "ACANTHOCYTES-PERIPHERAL-BLOOD-FILM-PATHOGNOMONIC",
            "ELEVATED-CK-PATHOGNOMONIC-NOT-MYOPATHY",
            "TONGUE-LIP-BITING-SELF-MUTILATION",
            "FEEDING-DYSTONIA-FOOD-EXPULSION",
            "MCLOED-SYNDROME-DISTINGUISH-XK-KELL-ANTIGEN",
            "CARDIAC-SURVEILLANCE-CARDIOMYOPATHY",
            "SEIZURES-50pct-FRONTAL",
            "FRESH-BLOOD-FILM-AVOID-EDTA-ARTEFACT",
        ],
        "age_of_onset": "20-40 yr; rare childhood onset; insidious — behavioural/psychiatric years before chorea",
        "key_biomarker": "Acanthocytes on fresh peripheral blood film (avoid EDTA); Serum CK (elevated); VPS13A chorein protein absent on Western blot",
        "seed_offset": 1,
    },

    # -- PANK2 — PKAN (NBIA1) -------------------------------------------------------
    {
        "gene": "PANK2",
        "alt_name": (
            "PANK2 (PANK2-570aa-20p13 / AR — PKAN-NBIA1-Pantothenate-Kinase-Associated-Neurodegeneration — "
            "Eye-of-the-Tiger-Sign-T2-MRI-PATHOGNOMONIC — GPi-Central-Hyperintensity — "
            "Pantethine-Trial — Iron-Chelation-Deferiprone)"
        ),
        "protein": (
            "PANK2 -- 20p13 AR -- PANK2-570aa -- "
            "Pantothenate-Kinase-2-Mitochondrial-CoA-Biosynthesis-Rate-Limiting-Enzyme -- "
            "NBIA1-Neurodegeneration-with-Brain-Iron-Accumulation-Type-1 -- "
            "Biallelic-Loss-of-Function-AR-Autosomal-Recessive -- "
            "Globus-Pallidus-Interna-Iron-Accumulation-Selective -- "
            "Eye-of-the-Tiger-Sign-T2-FLAIR-Hypointense-GPi-Central-Hyperintensity -- "
            "Classic-PKAN-Childhood-Onset-<10yr-Rapid-Progression -- "
            "Atypical-PKAN-Later-Onset-Slower-Speech-Prominent -- "
            "Pantethine-Bypasses-PANK2-Block-CoA-Supplementation"
        ),
        "locus": "20p13",
        "protein_size": "570 aa",
        "inheritance": (
            "AR (autosomal recessive) — biallelic PANK2 variants; "
            "Classic PKAN: onset <10 yr, rapid progression, wheelchair by 15 yr; "
            "Atypical PKAN: onset 10-40 yr, slower, speech/psychiatric prominent; "
            "Prevalence: 1-3/1,000,000; "
            "Classic vs Atypical: distinguished by age/progression NOT genotype alone; "
            "Founder mutations in certain populations; parental consanguinity risk factor"
        ),
        "pathognomonic": (
            "EYE-OF-THE-TIGER sign on T2/FLAIR MRI — PATHOGNOMONIC: "
            "Bilateral GPi hypointensity (iron) with central hyperintensity (gliosis/oedema); "
            "Dystonia (prominent in classic) + chorea + spasticity + pigmentary retinopathy; "
            "Parkinsonism in atypical PKAN; "
            "Acanthocytes in 8% (less than VPS13A); "
            "Pigmentary RETINOPATHY in classic PKAN"
        ),
        "treatment": (
            "**Pantethine** (CoA precursor, bypasses PANK2 block) — limited evidence, trial warranted; "
            "**Deferiprone** (iron chelation, crosses BBB) — clinical trials; "
            "**GPi-DBS** — dystonia management, significant benefit in PKAN; "
            "**Baclofen/Botulinum toxin** — spasticity and focal dystonia; "
            "**Antiepileptics** — seizure management; "
            "**CRISPR/gene therapy** — preclinical stage; "
            "**No FDA-approved disease-modifying** treatment"
        ),
        "critical_flags": [
            "EYE-OF-THE-TIGER-T2-MRI-PATHOGNOMONIC",
            "GPi-IRON-ACCUMULATION-BILATERAL",
            "PANTETHINE-TRIAL-CoA-BYPASS",
            "GPi-DBS-DYSTONIA-BENEFIT",
            "DEFERIPRONE-IRON-CHELATION-TRIAL",
            "RETINOPATHY-OPHTHALMOLOGY-MANDATORY",
            "ACANTHOCYTES-ONLY-8pct-LESS-THAN-ChAc",
            "CLASSIC-CHILDHOOD-WHEELCHAIR-15yr",
        ],
        "age_of_onset": "Classic: <10 yr (mean 3-4 yr); Atypical: 10-40 yr; wheelchair by 15 yr (classic); slower in atypical",
        "key_biomarker": "Eye-of-the-tiger sign on T2 MRI (GPi); Serum pantothenate; PANK2 gene sequencing",
        "seed_offset": 2,
    },

    # -- WDR45 — BPAN (NBIA5) -------------------------------------------------------
    {
        "gene": "WDR45",
        "alt_name": (
            "WDR45 (WDR45-330aa-Xp11.23 / XL-Dominant-De-Novo — BPAN-NBIA5-Beta-Propeller-Protein-Associated-Neurodegeneration — "
            "Early-Epilepsy-ID-Then-Adult-Parkinsonism-Dementia — Predominantly-Females-De-Novo)"
        ),
        "protein": (
            "WDR45 -- Xp11.23 XL-dominant-de-novo -- WDR45-330aa -- "
            "WD-Repeat-Domain-45-WIPI4-Autophagy-PI3P-Binding-Propeller-Scaffold -- "
            "NBIA5-Neurodegeneration-with-Brain-Iron-Accumulation-Type-5 -- "
            "X-Linked-Dominant-De-Novo-Males-Typically-More-Severe -- "
            "Biphasic-Phenotype-Early-Epilepsy-ID-then-Adult-Parkinsonism-Dementia -- "
            "Substantia-Nigra-GPi-Iron-Accumulation-MRI-T2-Hypointensity -- "
            "Autophagy-Defect-Dysfunctional-Mitophagy-Iron-Accumulation -- "
            "DaTscan-Abnormal-Dopamine-Transporter-Parkinsonism-Phase"
        ),
        "locus": "Xp11.23",
        "protein_size": "330 aa",
        "inheritance": (
            "X-linked dominant — de novo WDR45 variants (almost exclusively); "
            "Predominantly females (males more severely affected, often lethal prenatally or severe ID); "
            "Gonadal mosaicism rare — recurrence risk low but not zero; "
            "Biphasic course: Phase 1 (childhood): epilepsy + global intellectual disability; "
            "Phase 2 (adolescence/adulthood): parkinsonism-dementia; "
            "Prevalence: rare (~60 cases reported); increasing with recognition"
        ),
        "pathognomonic": (
            "BIPHASIC COURSE: childhood epilepsy+ID → adult parkinsonism-dementia — PATHOGNOMONIC; "
            "T2 MRI: iron in SUBSTANTIA NIGRA + GPi (halo sign); "
            "DaTscan ABNORMAL (dopaminergic deficit — parkinsonism phase); "
            "Levodopa response PARTIAL in parkinsonism phase; "
            "Stereotyped hand movements; sleep disturbance prominent"
        ),
        "treatment": (
            "**Levodopa** — partial response in parkinsonism phase; "
            "**Antiepileptics** — valproate, levetiracetam for childhood epilepsy phase; "
            "**Supportive care** — physiotherapy, speech, OT; "
            "**Chelation** — deferiprone trials ongoing; "
            "**No approved disease-modifying** treatment; "
            "**Genetic counselling**: de novo — recurrence risk low; test parents; "
            "**DaTscan** — confirm dopaminergic parkinsonism in adult phase"
        ),
        "critical_flags": [
            "BIPHASIC-COURSE-EPILEPSY-ID-THEN-PARKINSONISM-PATHOGNOMONIC",
            "PREDOMINANTLY-FEMALES-DE-NOVO",
            "SUBSTANTIA-NIGRA-GPi-IRON-T2-HALO",
            "DATSCAN-ABNORMAL-PARKINSONISM-PHASE",
            "LEVODOPA-PARTIAL-RESPONSE",
            "AUTOPHAGY-DEFECT-MITOPHAGY",
            "MALES-SEVERE-OFTEN-LETHAL",
            "GONADAL-MOSAICISM-LOW-RECURRENCE",
        ],
        "age_of_onset": "Phase 1: 2-10 yr (epilepsy+ID); Phase 2: 20-40 yr (parkinsonism+dementia); progression variable",
        "key_biomarker": "T2 MRI iron in SN + GPi (halo pattern); DaTscan abnormal; WDR45 gene sequencing (Xp11.23)",
        "seed_offset": 3,
    },

    # -- FTL — Neuroferritinopathy --------------------------------------------------
    {
        "gene": "FTL",
        "alt_name": (
            "FTL (FTL-175aa-19q13.33 / AD — Hereditary-Ferritinopathy-Neuroferritinopathy-HF — "
            "LOW-Serum-Ferritin-PARADOX-Iron-Accumulates-Brain-NOT-Serum — "
            "Adult-Chorea-Parkinsonism-Dementia — Cysts-MRI-PATHOGNOMONIC)"
        ),
        "protein": (
            "FTL -- 19q13.33 AD -- FTL-175aa -- "
            "Ferritin-Light-Chain-24-Subunit-Shell-Iron-Storage-Nanocage -- "
            "Hereditary-Ferritinopathy-HF-Adult-Onset-Neurodegeneration -- "
            "AD-Autosomal-Dominant-Gain-of-Function-Mutant-Shell-Disruption -- "
            "Iron-Accumulates-Brain-Basal-Ganglia-Cerebellum-Brainstem -- "
            "LOW-Serum-Ferritin-PARADOX-Disrupted-Nanocage-Cannot-Store-Iron-Systemically -- "
            "Cavitary-Lesions-Cysts-MRI-T2-PATHOGNOMONIC-Basal-Ganglia -- "
            "Cerebellar-Dentate-Iron-Hemosiderin-Deposits"
        ),
        "locus": "19q13.33",
        "protein_size": "175 aa",
        "inheritance": (
            "AD (autosomal dominant) — gain-of-function FTL frameshift variants disrupting nanocage; "
            "c.460dupA (p.Ala97Thrfs) most common; UK founder mutation; "
            "Penetrance: high (>90%) by 55 yr; "
            "Prevalence: rare; UK + France + Japan clusters; "
            "Adult onset 35-55 yr; progressive; "
            "Serum ferritin LOW (paradox) despite brain iron overload"
        ),
        "pathognomonic": (
            "LOW SERUM FERRITIN paradox — PATHOGNOMONIC (iron in brain, not serum); "
            "CYSTS/CAVITARY LESIONS on T2 MRI (basal ganglia, thalamus) — PATHOGNOMONIC; "
            "Adult-onset chorea + parkinsonism + dystonia + dementia; "
            "Dysarthria prominent; cerebellar signs; "
            "Family history: similar adult-onset movement disorder + dementia; "
            "Iron deposits: low T2/T2* signal in caudate, putamen, globus pallidus"
        ),
        "treatment": (
            "**Iron chelation** — deferiprone (DFP), deferasirox — limited evidence but rationale strong; "
            "**Tetrabenazine** — chorea management; "
            "**Antipsychotics** (low-dose) — chorea + behavioural; "
            "**No disease-modifying** approved treatment; "
            "**Avoid iron supplementation** (worsens brain iron); "
            "**Do NOT prescribe ferritin for low serum ferritin** — treat the cause, not the number; "
            "**Multidisciplinary**: neurology, haematology, genetics"
        ),
        "critical_flags": [
            "LOW-SERUM-FERRITIN-PARADOX-PATHOGNOMONIC",
            "DO-NOT-SUPPLEMENT-IRON",
            "CYSTS-T2-MRI-BASAL-GANGLIA-PATHOGNOMONIC",
            "IRON-CHELATION-DEFERIPRONE-RATIONALE",
            "BRAIN-IRON-NOT-SERUM-IRON",
            "ADULT-ONSET-35-55yr",
            "UK-FRANCE-JAPAN-CLUSTERS-c460dupA",
            "CEREBELLAR-DENTATE-IRON-T2-STAR",
        ],
        "age_of_onset": "35-55 yr; progressive over 10-20 yr; death typically 60-70 yr",
        "key_biomarker": "Low serum ferritin (paradox); T2/T2* MRI iron deposits + cysts; FTL gene sequencing",
        "seed_offset": 4,
    },

    # -- NKX2-1 — Benign Hereditary Chorea ------------------------------------------
    {
        "gene": "NKX2-1",
        "alt_name": (
            "NKX2-1 (NKX2-1-401aa-14q13.3 / AD — Benign-Hereditary-Chorea-BHC — "
            "Brain-Lung-Thyroid-Syndrome — Childhood-Chorea-NOT-Progressive — "
            "Thyroid-Dysfunction-70pct — Lung-Disease-54pct)"
        ),
        "protein": (
            "NKX2-1 -- 14q13.3 AD -- NKX2-1-401aa -- "
            "NK2-Homeobox-1-TTF-1-Thyroid-Transcription-Factor-1-Homeodomain-TF -- "
            "Brain-Lung-Thyroid-Transcription-Factor-Organ-Development -- "
            "Benign-Hereditary-Chorea-BHC-Childhood-onset-Non-progressive -- "
            "AD-Autosomal-Dominant-Haploinsufficiency -- "
            "Thyroid-Dysfunction-Hypothyroidism-Hyperthyroidism-70pct -- "
            "Lung-Disease-Respiratory-Distress-Neonatal-ILD-54pct -- "
            "NOT-Progressive-NOT-Huntington-Good-Prognosis -- "
            "Brain-Surfactant-Protein-SP-B-SP-C-Thyroid-TTF1-Lung-NKX2-1"
        ),
        "locus": "14q13.3",
        "protein_size": "401 aa",
        "inheritance": (
            "AD (autosomal dominant) — haploinsufficiency NKX2-1 variants + deletions; "
            "14q13.3 chromosomal deletions detectable by CMA; "
            "Variable expressivity: brain-only, brain+thyroid, brain+lung+thyroid; "
            "Penetrance: high for chorea; thyroid/lung variable; "
            "De novo and familial; "
            "NOT progressive — distinguishes from Huntington; "
            "Prevalence: rare but likely underdiagnosed"
        ),
        "pathognomonic": (
            "CHILDHOOD CHOREA + THYROID DYSFUNCTION + LUNG DISEASE = PATHOGNOMONIC triad; "
            "Chorea: childhood onset, non-progressive, often improves in adulthood; "
            "Thyroid: hypothyroidism (most common), hyperthyroidism, thyroid dysgenesis (70%); "
            "Lung: neonatal respiratory distress, interstitial lung disease, pulmonary alveolar proteinosis (54%); "
            "Intellectual disability/learning difficulties in some; "
            "MRI: normal or mild abnormalities (NOT iron accumulation)"
        ),
        "treatment": (
            "**Levodopa** — excellent response in many BHC patients (trial mandatory); "
            "**Tetrabenazine** — if levodopa inadequate; "
            "**Thyroid hormone replacement** — hypothyroidism management; "
            "**Pulmonology** — lung disease management; "
            "**Good prognosis** — chorea often improves with age; "
            "**Genetic counselling** — CMA + sequencing; "
            "**Annual thyroid function tests** — lifelong; "
            "**Pulmonary function tests** — periodic surveillance"
        ),
        "critical_flags": [
            "BRAIN-LUNG-THYROID-TRIAD-PATHOGNOMONIC",
            "NOT-PROGRESSIVE-GOOD-PROGNOSIS",
            "THYROID-FUNCTION-ANNUAL-LIFELONG",
            "LEVODOPA-TRIAL-EXCELLENT-RESPONSE",
            "CMA-MANDATORY-14q13.3-DELETION",
            "NEONATAL-RESPIRATORY-DISTRESS-LUNG-DISEASE",
            "DISTINGUISH-FROM-HUNTINGTON-NOT-PROGRESSIVE",
            "INTELLECTUAL-DISABILITY-SOME-PATIENTS",
        ],
        "age_of_onset": "Infancy to early childhood (1-5 yr); non-progressive; may improve spontaneously in adolescence/adulthood",
        "key_biomarker": "Thyroid function tests (TFTs); Pulmonary function tests; NKX2-1 gene + CMA for 14q13.3 deletion",
        "seed_offset": 5,
    },

    # -- JPH3 — Huntington Disease-Like 2 -------------------------------------------
    {
        "gene": "JPH3",
        "alt_name": (
            "JPH3 (JPH3-741aa-16q24.2 / AD — Huntington-Disease-Like-2-HDL2 — "
            "CTG-CAG-Repeat-16q24.3 — Predominantly-African-Ancestry — "
            "Psychiatric-Features-Prominent — Acanthocytes-50pct — Test-HTT-FIRST)"
        ),
        "protein": (
            "JPH3 -- 16q24.2 AD -- JPH3-741aa -- "
            "Junctophilin-3-Membrane-Anchoring-Junctional-SR-Plasma-Membrane-Junction -- "
            "Huntington-Disease-Like-2-HDL2-CTG-CAG-Trinucleotide-Repeat-16q24.3 -- "
            "AD-Autosomal-Dominant-Repeat-Expansion-Toxic-RNA-Gain-of-Function -- "
            "CTG≥41-Pathogenic-Normal-≤28-Intermediate-29-40 -- "
            "Predominantly-African-Ancestry-Founder-Alleles -- "
            "Clinically-Indistinguishable-from-Huntington-Disease -- "
            "Acanthocytes-50pct-ChAc-Like -- "
            "Test-HTT-Repeat-FIRST-Then-JPH3-If-Negative-African-Ancestry"
        ),
        "locus": "16q24.2",
        "protein_size": "741 aa",
        "inheritance": (
            "AD (autosomal dominant) — CTG/CAG repeat expansion JPH3 gene (16q24.3); "
            "CTG≥41 pathogenic; CTG29-40 intermediate (risk of expansion); "
            "Paternal transmission → anticipation possible; "
            "Predominantly African ancestry (Africa, African-American, Afro-Caribbean); "
            "Prevalence in non-African populations: rare; "
            "Identical phenotype to HD → always exclude HTT first; "
            "Acanthocytes in 50% (differs from VPS13A: AR vs AD)"
        ),
        "pathognomonic": (
            "CLINICALLY INDISTINGUISHABLE FROM HUNTINGTON DISEASE; "
            "Chorea + cognitive decline + psychiatric (depression, irritability, apathy); "
            "AFRICAN ANCESTRY — key epidemiological clue; "
            "ACANTHOCYTES in 50% (higher than HD); "
            "Negative HTT repeat expansion + African ancestry → test JPH3; "
            "Caudate atrophy on MRI (same as HD); "
            "Age of onset 30-50 yr; progression similar to HD"
        ),
        "treatment": (
            "**Same as Huntington Disease**: tetrabenazine/deutetrabenazine for chorea; "
            "**Antipsychotics** — psychiatric symptoms; "
            "**Multidisciplinary** — same as HD; "
            "**No disease-modifying** treatment; "
            "**Genetic counselling** — repeat expansion; anticipation counselling; "
            "**Presymptomatic testing** — HD Protocol (counselling mandatory); "
            "**Screen for acanthocytes** — may guide diagnosis"
        ),
        "critical_flags": [
            "TEST-HTT-FIRST-BEFORE-JPH3",
            "AFRICAN-ANCESTRY-KEY-CLUE",
            "CLINICALLY-IDENTICAL-TO-HD",
            "ACANTHOCYTES-50pct-PERIPHERAL-BLOOD",
            "CTG≥41-PATHOGNOMONIC",
            "ANTICIPATION-PATERNAL-TRANSMISSION",
            "PRESYMPTOMATIC-TESTING-HD-PROTOCOL",
            "CAUDATE-ATROPHY-SAME-AS-HD",
        ],
        "age_of_onset": "30-50 yr; similar to Huntington disease; juvenile onset rare",
        "key_biomarker": "CTG/CAG repeat count (JPH3 locus); Negative HTT repeat; Acanthocytes (50%); Caudate atrophy MRI",
        "seed_offset": 6,
    },

    # -- TBP — SCA17 / Huntington Disease-Like 4 ------------------------------------
    {
        "gene": "TBP",
        "alt_name": (
            "TBP (TBP-339aa-6q27 / AD — SCA17-HDL4-TATA-Binding-Protein-CAG-Repeat-PolyQ — "
            "CAG≥49-Pathogenic — Ataxia-PLUS-Chorea-Dementia-Parkinsonism — "
            "HD-Like-Phenotype-WITH-Ataxia-KEY-Distinguisher)"
        ),
        "protein": (
            "TBP -- 6q27 AD -- TBP-339aa -- "
            "TATA-Binding-Protein-General-Transcription-Factor-Basal-RNA-Pol-II-Promoter -- "
            "SCA17-Spinocerebellar-Ataxia-17-HDL4-Huntington-Disease-Like-4 -- "
            "CAG-Repeat-Expansion-PolyQ-N-terminal-Tract-TBP-Exon-3 -- "
            "CAG≥49-Pathogenic-CAG44-48-Reduced-Penetrance-CAG≤43-Normal -- "
            "AD-Autosomal-Dominant-Toxic-PolyQ-Aggregation-Nuclear-Inclusions -- "
            "Cerebellar-Ataxia-PLUS-Chorea-PLUS-Dementia-PLUS-Parkinsonism -- "
            "Ataxia-Distinguishes-SCA17-from-Pure-HD -- "
            "Anticipation-Paternal-Transmission-Preferred"
        ),
        "locus": "6q27",
        "protein_size": "339 aa",
        "inheritance": (
            "AD (autosomal dominant) — CAG repeat expansion in TBP exon 3; "
            "CAG≥49 pathogenic; CAG44-48 reduced penetrance; CAG≤43 normal; "
            "Paternal transmission → anticipation (CAG expands); "
            "Broad phenotype: ataxia-dominant (SCA17) to HD-like (HDL4); "
            "Cerebellar atrophy on MRI (distinguishes from pure HD); "
            "Prevalence: rare; prevalence of CAG40-48 intermediate alleles ~1/1000"
        ),
        "pathognomonic": (
            "CHOREA + ATAXIA combined = KEY distinguisher from pure HD (ataxia absent in HD); "
            "Dementia (frontal > posterior); parkinsonism; epilepsy (30%); "
            "CEREBELLAR ATROPHY on MRI — PATHOGNOMONIC vs HD (caudate only); "
            "Psychiatric: depression, psychosis, personality change; "
            "Gait: combination of cerebellar + choreic — distinctive; "
            "Negative HTT + ataxia → TBP sequencing"
        ),
        "treatment": (
            "**Tetrabenazine/Deutetrabenazine** — chorea management; "
            "**Antipsychotics** — psychiatric + chorea; "
            "**Antiepileptics** — seizure management (30%); "
            "**Physiotherapy** — ataxia rehabilitation; "
            "**No disease-modifying** approved treatment; "
            "**Genetic counselling** — anticipation; presymptomatic testing (HD Protocol); "
            "**riluzole** — off-label for ataxia component (SCA evidence)"
        ),
        "critical_flags": [
            "ATAXIA-PLUS-CHOREA-KEY-DISTINGUISHER-FROM-HD",
            "CAG≥49-PATHOGNOMONIC",
            "CEREBELLAR-ATROPHY-MRI-NOT-CAUDATE-ONLY",
            "TEST-HTT-AND-JPH3-NEGATIVE-THEN-TBP",
            "ANTICIPATION-PATERNAL-TRANSMISSION",
            "EPILEPSY-30pct",
            "PSYCHIATRIC-PSYCHOSIS-DEPRESSION",
            "HD-LIKE-WITH-ATAXIA-CLINICAL-FLAG",
        ],
        "age_of_onset": "20-50 yr (variable); mean ~35 yr; broader range than HD; juvenile onset with very large repeats",
        "key_biomarker": "CAG repeat count (TBP locus); Cerebellar atrophy on MRI; Negative HTT and JPH3 repeats",
        "seed_offset": 7,
    },
]


def _generate_cohort():
    """Generate 8 × 40-patient cohort (320 total) with realistic clinical parameters."""
    all_patients = []
    for gene_data in CHOREA_GENES:
        gene = gene_data["gene"]
        rng = random.Random(SEED_BASE + gene_data["seed_offset"])
        for i in range(40):
            pid = f"{gene}-{SEED_BASE + gene_data['seed_offset']}-{i:03d}"

            # Gene-specific parameters
            if gene == "HTT":
                cag_repeat = rng.randint(36, 65)
                juvenile = cag_repeat >= 60
                age_onset = rng.randint(10, 20) if juvenile else rng.randint(28, 55)
                tetrabenazine = rng.random() < 0.65
                psychiatric_first = rng.random() < 0.60
                presymptomatic_test = rng.random() < 0.25
                caudate_atrophy = rng.random() < 0.80
                all_patients.append({
                    "patient_id": pid,
                    "gene": gene,
                    "cag_repeat": cag_repeat,
                    "juvenile": juvenile,
                    "age_onset": age_onset,
                    "tetrabenazine": tetrabenazine,
                    "psychiatric_first": psychiatric_first,
                    "presymptomatic_test": presymptomatic_test,
                    "caudate_atrophy": caudate_atrophy,
                })

            elif gene == "VPS13A":
                acanthocytes = rng.random() < 0.92
                ck_elevated = rng.random() < 0.90
                tongue_biting = rng.random() < 0.75
                feeding_dystonia = rng.random() < 0.65
                seizures = rng.random() < 0.52
                all_patients.append({
                    "patient_id": pid,
                    "gene": gene,
                    "acanthocytes": acanthocytes,
                    "ck_elevated": ck_elevated,
                    "tongue_biting": tongue_biting,
                    "feeding_dystonia": feeding_dystonia,
                    "seizures": seizures,
                })

            elif gene == "PANK2":
                eye_tiger = rng.random() < 0.95
                classic = rng.random() < 0.55
                age_onset = rng.randint(2, 8) if classic else rng.randint(10, 35)
                dystonia = rng.random() < 0.85
                retinopathy = rng.random() < (0.65 if classic else 0.15)
                gpi_dbs = rng.random() < 0.35
                pantethine_trial = rng.random() < 0.30
                all_patients.append({
                    "patient_id": pid,
                    "gene": gene,
                    "eye_tiger_mri": eye_tiger,
                    "classic_pkan": classic,
                    "age_onset": age_onset,
                    "dystonia": dystonia,
                    "retinopathy": retinopathy,
                    "gpi_dbs": gpi_dbs,
                    "pantethine_trial": pantethine_trial,
                })

            elif gene == "WDR45":
                female = rng.random() < 0.82
                biphasic = rng.random() < 0.90
                epilepsy_phase1 = rng.random() < 0.92
                parkinsonism_phase2 = rng.random() < 0.75
                levodopa_partial = rng.random() < 0.55
                datscan_abnormal = rng.random() < 0.78
                all_patients.append({
                    "patient_id": pid,
                    "gene": gene,
                    "female": female,
                    "biphasic_course": biphasic,
                    "epilepsy_phase1": epilepsy_phase1,
                    "parkinsonism_phase2": parkinsonism_phase2,
                    "levodopa_partial": levodopa_partial,
                    "datscan_abnormal": datscan_abnormal,
                })

            elif gene == "FTL":
                low_ferritin = rng.random() < 0.95
                cysts_mri = rng.random() < 0.85
                chorea = rng.random() < 0.88
                parkinsonism = rng.random() < 0.60
                dementia = rng.random() < 0.55
                iron_chelation = rng.random() < 0.40
                all_patients.append({
                    "patient_id": pid,
                    "gene": gene,
                    "low_serum_ferritin": low_ferritin,
                    "cysts_mri": cysts_mri,
                    "chorea": chorea,
                    "parkinsonism": parkinsonism,
                    "dementia": dementia,
                    "iron_chelation": iron_chelation,
                })

            elif gene == "NKX2-1":
                thyroid_dysfunction = rng.random() < 0.72
                lung_disease = rng.random() < 0.55
                levodopa_response = rng.random() < 0.65
                non_progressive = rng.random() < 0.90
                childhood_onset = rng.random() < 0.92
                intellectual_disability = rng.random() < 0.35
                all_patients.append({
                    "patient_id": pid,
                    "gene": gene,
                    "thyroid_dysfunction": thyroid_dysfunction,
                    "lung_disease": lung_disease,
                    "levodopa_response": levodopa_response,
                    "non_progressive": non_progressive,
                    "childhood_onset": childhood_onset,
                    "intellectual_disability": intellectual_disability,
                })

            elif gene == "JPH3":
                african_ancestry = rng.random() < 0.88
                acanthocytes = rng.random() < 0.52
                psychiatric_features = rng.random() < 0.70
                caudate_atrophy = rng.random() < 0.78
                htt_negative = True  # By definition (HDL2 only if HTT negative)
                all_patients.append({
                    "patient_id": pid,
                    "gene": gene,
                    "african_ancestry": african_ancestry,
                    "acanthocytes": acanthocytes,
                    "psychiatric_features": psychiatric_features,
                    "caudate_atrophy": caudate_atrophy,
                    "htt_repeat_negative": htt_negative,
                })

            elif gene == "TBP":
                ataxia_prominent = rng.random() < 0.80
                chorea_present = rng.random() < 0.75
                dementia = rng.random() < 0.65
                epilepsy = rng.random() < 0.32
                cerebellar_atrophy = rng.random() < 0.88
                psychiatric = rng.random() < 0.60
                cag_repeat = rng.randint(49, 70)
                all_patients.append({
                    "patient_id": pid,
                    "gene": gene,
                    "ataxia_prominent": ataxia_prominent,
                    "chorea_present": chorea_present,
                    "dementia": dementia,
                    "epilepsy": epilepsy,
                    "cerebellar_atrophy": cerebellar_atrophy,
                    "psychiatric": psychiatric,
                    "cag_repeat": cag_repeat,
                })

    return all_patients


def overview():
    """Return atlas-level aggregate overview."""
    patients = _generate_cohort()
    tetrabenazine = sum(1 for p in patients if p.get("tetrabenazine"))
    psychiatric_first = sum(1 for p in patients if p.get("psychiatric_first") or p.get("psychiatric_features"))
    acanthocytes = sum(1 for p in patients if p.get("acanthocytes"))
    eye_tiger = sum(1 for p in patients if p.get("eye_tiger_mri"))
    biphasic = sum(1 for p in patients if p.get("biphasic_course"))
    low_ferritin = sum(1 for p in patients if p.get("low_serum_ferritin"))
    thyroid_dysfunction = sum(1 for p in patients if p.get("thyroid_dysfunction"))
    african_ancestry = sum(1 for p in patients if p.get("african_ancestry"))
    cerebellar_atrophy = sum(1 for p in patients if p.get("cerebellar_atrophy"))
    tongue_biting = sum(1 for p in patients if p.get("tongue_biting"))
    return {
        "atlas": "Hereditary-Chorea-Atlas",
        "genes": [g["gene"] for g in CHOREA_GENES],
        "total_patients": len(patients),
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "tetrabenazine_patients": tetrabenazine,
        "psychiatric_first_patients": psychiatric_first,
        "acanthocytes_patients": acanthocytes,
        "eye_tiger_mri_patients": eye_tiger,
        "biphasic_bpan_patients": biphasic,
        "low_ferritin_paradox_patients": low_ferritin,
        "thyroid_dysfunction_bhc_patients": thyroid_dysfunction,
        "african_ancestry_hdl2_patients": african_ancestry,
        "cerebellar_atrophy_sca17_patients": cerebellar_atrophy,
        "tongue_biting_chac_patients": tongue_biting,
    }


def breakdown():
    """Return per-gene breakdown with clinical details."""
    patients = _generate_cohort()
    result = {}
    for gene_data in CHOREA_GENES:
        gene = gene_data["gene"]
        gene_patients = [p for p in patients if p["gene"] == gene]
        result[gene] = {
            "gene": gene,
            "alt_name": gene_data["alt_name"],
            "locus": gene_data["locus"],
            "protein_size": gene_data["protein_size"],
            "inheritance": gene_data["inheritance"],
            "patient_count": len(gene_patients),
            "pathognomonic": gene_data["pathognomonic"],
            "treatment": gene_data["treatment"],
            "critical_flags": gene_data["critical_flags"],
            "age_of_onset": gene_data["age_of_onset"],
            "key_biomarker": gene_data["key_biomarker"],
        }
    return result


def definitions():
    """Return gene definitions, glossary and treatment protocols."""
    return {
        "genes": {g["gene"]: g["protein"] for g in CHOREA_GENES},
        "glossary": {
            "Chorea": "Involuntary, rapid, unpredictable, flowing movements; shifts from body part to body part; resembles a dance (Greek: choreia); worsened by stress; reduced by concentration; absent during sleep; hallmark of HD but also seen in NBIA, BHC, HDL syndromes",
            "VMAT2 Inhibitor": "Vesicular monoamine transporter 2 inhibitor; depletes presynaptic dopamine/serotonin/norepinephrine; tetrabenazine (FDA 2008 HD chorea), deutetrabenazine (FDA 2017), valbenazine (FDA 2023); CONTRAINDICATED in primary dystonia (worsens it); approved specifically for HD chorea",
            "Neurodegeneration with Brain Iron Accumulation (NBIA)": "Group of genetic disorders with pathological iron deposition in basal ganglia; includes PKAN (PANK2), BPAN (WDR45), Neuroferritinopathy (FTL), MPAN, FAHN, COASY; all show T2 hypointensity in GPi/SN; key: eye-of-tiger (PKAN), halo sign (BPAN), cysts (FTL)",
            "HD Phenocopy Syndromes (HDL)": "Conditions clinically identical to HD but HTT repeat normal: HDL1 (PRNP), HDL2 (JPH3 — African), HDL3 (unknown), HDL4 (TBP/SCA17); workup: HTT first, then JPH3 (African ancestry), then TBP; collectively account for ~1% of HD-like presentations",
            "Eye-of-the-Tiger Sign": "T2/FLAIR MRI finding: bilateral GPi hypointensity (iron) with central area of hyperintensity (gliosis/neuronal loss/oedema); pathognomonic for PKAN (PANK2 mutations); not seen in other NBIA subtypes; confirmed on gradient echo/susceptibility-weighted imaging",
            "Acanthocytes": "Spiculated (thorny) erythrocytes on peripheral blood film; seen in neuroacanthocytosis syndromes: VPS13A (ChAc — >3% specific), JPH3 (HDL2 ~50%), McLeod syndrome (XK); count on fresh smear (EDTA causes false acanthocytosis artefact — always use fresh heparinised blood)",
            "Chorea-Acanthocytosis (ChAc)": "VPS13A AR disorder; neuroacanthocytosis + acanthocytes + elevated CK + self-mutilating orofacial dystonia; caudate atrophy; McLeod syndrome (XK, X-linked) overlaps — distinguish by Kell antigen typing; ChAc: normal Kell",
            "PKAN (Pantothenate Kinase-Associated Neurodegeneration)": "PANK2 AR; NBIA1; most common NBIA; eye-of-tiger MRI pathognomonic; Classic (onset <10 yr, rapid) vs Atypical (onset >10 yr, slower, speech prominent); pantethine bypasses PANK2 block; deferiprone chelation ongoing trials; GPi-DBS helps dystonia",
            "BPAN (Beta-propeller Protein-Associated Neurodegeneration)": "WDR45 X-linked dominant de novo; NBIA5; biphasic: Phase 1 childhood epilepsy+ID → Phase 2 adult parkinsonism+dementia; predominantly females; DaTscan abnormal in phase 2; autophagy defect (WIPI4); iron in SN+GPi on MRI",
            "Benign Hereditary Chorea (BHC)": "NKX2-1 AD; brain-lung-thyroid syndrome; childhood chorea NOT progressive (distinguishes from HD); thyroid dysfunction 70%, lung disease 54%; levodopa often dramatically effective; CMA essential (14q13.3 deletions); annual TFTs mandatory lifelong",
            "Serum Ferritin Paradox (Neuroferritinopathy)": "FTL AD; LOW serum ferritin despite brain iron overload; disrupted ferritin nanocage cannot sequester iron in circulation → iron accumulates in brain basal ganglia, cerebellum; do NOT supplement iron; chelation rationale strong; cysts on T2 MRI pathognomonic",
            "Presymptomatic Testing — HD Protocol": "Strict international guidelines for testing at-risk individuals before symptoms; mandatory: pre-test genetic counselling (minimum 2 sessions), psychological assessment, no economic coercion; result disclosure in person with counsellor present; applies to HD (HTT), HDL2 (JPH3), HDL4 (TBP); predictive testing for minors not recommended",
            "Anticipation in Polyglutamine Diseases": "CAG repeat expansions tend to increase in successive generations (especially paternal transmission); earlier onset + more severe disease in subsequent generations; documented in HTT, TBP (SCA17), JPH3; intermediate alleles (27-35 for HTT) can expand to pathogenic range in next generation — counsel carefully",
        },
        "surveillance_protocols": {
            "HTT (Huntington Disease)": "Molecular: triplet repeat PCR (CAG count); presymptomatic: HD Protocol counselling; symptomatic: neuropsychological battery (UHDRS); MRI caudate volume; NfL plasma (surrogate progression marker); tetrabenazine/deutetrabenazine for chorea; suicide risk screening every visit; multidisciplinary team: neurology + psychiatry + physiotherapy + SLT + dietitian + genetics; ENROLL-HD registry enrolment recommended",
            "VPS13A (Chorea-Acanthocytosis)": "Fresh peripheral blood film (heparinised — NOT EDTA) for acanthocytes; CK serum; chorein Western blot (absent in ChAc) if available; XK gene sequencing for McLeod overlap; Kell antigen typing; cardiac monitoring (cardiomyopathy risk); seizure management (EEG); nutritional assessment (feeding dystonia); botulinum toxin for orolingual dystonia; dental protection",
            "PANK2 (PKAN-NBIA1)": "T2/FLAIR MRI (eye-of-tiger pathognomonic); PANK2 sequencing; ophthalmology (retinopathy — annual in classic); pantethine trial (off-label, 300-900 mg/day); deferiprone trial (referral to NBIA specialist); GPi-DBS assessment if dystonia dominant; physiotherapy; NBIA UK/NORD specialist centre referral",
            "WDR45 (BPAN-NBIA5)": "WDR45 sequencing (X-linked — test female proband + male relatives); parental blood for de novo confirmation; MRI (SN + GPi T2 halo); EEG (phase 1 epilepsy); DaTscan when parkinsonism emerges; levodopa trial; antiepileptic therapy (VPA avoid — risk of hyperammonaemia); BPAN/NBIA specialist registry; deferiprone trial consideration",
            "FTL (Neuroferritinopathy)": "FTL sequencing; serum ferritin (LOW — paradox marker); iron studies; T2*/GRE MRI (iron + cysts basal ganglia); Do NOT supplement iron; iron chelation (deferiprone — specialist referral); tetrabenazine for chorea; family cascade testing; c.460dupA UK founder screen first if UK/European; UK Neuroferritinopathy specialist clinic referral",
            "NKX2-1 (Benign Hereditary Chorea)": "NKX2-1 sequencing + CMA (14q13.3 deletion detection mandatory); thyroid function tests (TFTs) — annual lifelong; thyroid ultrasound; lung function tests (PFT); HRCT chest if lung symptoms; levodopa trial (often dramatic response); neuropsychological assessment; brain MRI (usually normal); OMIM 600635",
            "JPH3 (Huntington Disease-Like 2)": "Step 1: HTT CAG repeat (always first); Step 2: JPH3 CTG repeat if HTT normal + African ancestry; Fresh peripheral blood film (acanthocytes ~50%); Brain MRI (caudate atrophy); HD Protocol presymptomatic counselling applies; psychiatry referral; same multidisciplinary approach as HD; XK gene if acanthocytes present (McLeod vs JPH3)",
            "TBP (SCA17/HDL4)": "TBP CAG repeat sequencing; Brain MRI (cerebellar atrophy — distinguishes from HD); EEG (epilepsy 30%); neuropsychological battery; presymptomatic testing as per HD Protocol; tetrabenazine for chorea; antiepileptics; ataxia rehabilitation (physiotherapy); HTT then JPH3 then TBP in diagnostic algorithm; riluzole (off-label for ataxia component)",
        },
    }


if __name__ == "__main__":
    import json
    ov = overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Tetrabenazine patients: {ov['tetrabenazine_patients']}")
    print(f"Acanthocytes patients: {ov['acanthocytes_patients']}")
    print(f"Eye-of-the-tiger MRI: {ov['eye_tiger_mri_patients']}")
    print(f"Biphasic BPAN patients: {ov['biphasic_bpan_patients']}")
    print(f"Low ferritin paradox: {ov['low_ferritin_paradox_patients']}")
    print(f"Thyroid dysfunction (BHC): {ov['thyroid_dysfunction_bhc_patients']}")
