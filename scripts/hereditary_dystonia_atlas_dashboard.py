#!/usr/bin/env python3
"""Hereditary-Dystonia-Atlas — Complete 8-Gene Hereditary Dystonia Atlas
(DYT-TOR1A/DYT1 · DYT-SGCE/DYT11 · DRD-GCH1/DYT5a · DRD-TH/DYT5b ·
 DYT-KMT2B/DYT28 · DYT-THAP1/DYT6 · ATP1A3-AHC/RDP · DYT-ANO3/DYT24).

TOR1A   (Torsin-1A; 332 aa; 9q34.11; AD;
         DYT-TOR1A / DYT1 — most common genetic generalised dystonia;
         GAG deletion (c.907_909delGAG) removes Glu302/303; pathognomonic variant;
         seed SEED_BASE+0).
SGCE    (ε-Sarcoglycan; 437 aa; 7q21.3; AD — paternal imprint;
         DYT-SGCE / Myoclonus-Dystonia (M-D) / DYT11;
         MYOCLONUS PREDOMINATES over dystonia — lightning-fast jerks;
         ALCOHOL RESPONSIVE (temporary; ethanol test has diagnostic utility);
         seed SEED_BASE+1).
GCH1    (GTP Cyclohydrolase I; 250 aa; 14q22.2; AD;
         DRD / DYT-GCH1 / DYT5a — Dopa-Responsive Dystonia (Segawa syndrome);
         L-DOPA CURATIVE — MUST try before botox in any childhood dystonia with diurnal variation;
         DIURNAL VARIATION (better morning, worse evening) PATHOGNOMONIC;
         seed SEED_BASE+2).
TH      (Tyrosine Hydroxylase; 528 aa; 11p15.5; AR;
         DRD / DYT-TH / DYT5b — severe infantile form of DRD;
         L-DOPA responsive at lower doses; infantile encephalopathy phenotype;
         seed SEED_BASE+3).
KMT2B   (Lysine Methyltransferase 2B; 3969 aa; 19q13.12; AD (90% de novo);
         DYT-KMT2B / DYT28 — childhood onset focal dystonia → generalises;
         INTELLECTUAL DISABILITY 40%; DBS-GPi HIGHLY EFFECTIVE even with ID;
         seed SEED_BASE+4).
THAP1   (THAP domain-containing protein 1; 213 aa; 8p21.3; AD;
         DYT-THAP1 / DYT6 — young adult onset;
         LARYNGEAL / CRANIAL INVOLVEMENT PATHOGNOMONIC;
         Ashkenazi Jewish founder enrichment;
         seed SEED_BASE+5).
ATP1A3  (Na+/K+-ATPase alpha-3; 1013 aa; 19q13.2; AD (de novo);
         AHC (Alternating Hemiplegia of Childhood) / CAPOS / RDP;
         FEVER = ABSOLUTE trigger for AHC attacks — FEVER PROTOCOL MANDATORY;
         FLUNARIZINE first-line for AHC episodes;
         seed SEED_BASE+6).
ANO3    (Anoctamin-3; 913 aa; 11p14.3; AD;
         DYT-ANO3 / DYT24 — adult onset craniocervical dystonia;
         CRANIOCERVICAL + TREMULOUS DYSTONIA PATHOGNOMONIC (tremor distinguishes from THAP1);
         Botulinum toxin first-line; DBS-GPi considered;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2158-2165).
"""

import random

SEED_BASE = 2158

DYSTONIA_GENES = [
    # -- TOR1A — DYT1 / DYT-TOR1A -----------------------------------------------
    {
        "gene": "TOR1A",
        "alt_name": (
            "TOR1A (TOR1A-332aa-9q34.11 / AD — DYT-TOR1A-DYT1-Oppenheim-Dystonia — "
            "GAG-DELETION-Glu302303-PATHOGNOMONIC-Single-Variant-95pct — "
            "MOST-COMMON-GENETIC-GENERALISED-DYSTONIA — "
            "DBS-GPi-HIGHLY-EFFECTIVE->70pct-Improvement — "
            "PENETRANCE-30-40pct-Incomplete)"
        ),
        "protein": (
            "TOR1A -- 9q34.11 AD -- TOR1A-332aa -- "
            "Torsin-1A-AAA+-ATPase-Superfamily-ER-Lumen-Nuclear-Envelope-Torsion-Function -- "
            "DYT-TOR1A-Oppenheim-Dystonia-DYT1-OMIM-128100 -- "
            "GAG-Deletion-c907_909delGAG-Removes-Glu302-or-Glu303-PATHOGNOMONIC-SINGLE-VARIANT-IN-95pct -- "
            "Most-Common-Genetic-Generalised-Dystonia-Worldwide-1-in-3000-Ashkenazi-Jewish -- "
            "Childhood-Onset-6-26yr-Lower-Limb-FIRST-PATHOGNOMONIC-Then-Spreads -- "
            "Incomplete-Penetrance-30-40pct-Most-Carriers-Unaffected -- "
            "DBS-GPi-Highly-Effective->70pct-Improvement-Even-Severe-Cases -- "
            "Trihexyphenidyl-HIGH-DOSE-Anticholinergic-Trial-MANDATORY-Before-DBS -- "
            "Torsin-1A-ER-Lumen-AAA-ATPase-LAP1-LULL1-Cofactors-Nuclear-Pore-Complex -- "
            "9q34.11"
        ),
        "locus": "9q34.11",
        "protein_size": "332 aa",
        "inheritance": (
            "AD (autosomal dominant); incomplete penetrance 30-40%; "
            "GAG deletion (c.907_909delGAG) = 95%+ of DYT-TOR1A worldwide; "
            "sequencing alone may miss complex indels — long-read sequencing in negative cases; "
            "de novo rare (<5%); familial mutation in most; "
            "founder effect Ashkenazi Jewish (1 in 2000-3000 carriers); "
            "heterozygous = risk; homozygous = lethal (embryonic); "
            "modifier genes proposed (HDAC) — environmental triggers uncertain"
        ),
        "age_of_onset": "Childhood 6-26 yr (median 12 yr); lower limb onset first",
        "pathognomonic": (
            "GAG DELETION in TOR1A is the PATHOGNOMONIC VARIANT — 95%+ of DYT-TOR1A; "
            "LOWER LIMB ONSET first (foot inversion/plantar flexion) then spreads to trunk/arms; "
            "SUSTAINED TORSIONAL MOVEMENTS distinguishing from tremor or myoclonus; "
            "Diurnal variation ABSENT (distinguishes from DRD/GCH1 — critical DDx); "
            "Normal cognitive function — cognition preserved in uncomplicated DYT1; "
            "MRI brain normal — structural neuroimaging does not confirm; "
            "Penetrance 30-40%: family members with same variant may be unaffected; "
            "Action-induced dystonia worsens with specific movement then spills over; "
            "Oromandibular and cranial involvement RARE in DYT1 (unlike THAP1 or KMT2B)"
        ),
        "treatment": (
            "FIRST-LINE PHARMACOLOGICAL: Trihexyphenidyl (benztropine/THP) — HIGH DOSE titrated; "
            "start 1-2 mg/day → target 20-30 mg/day in children; monitor for dry mouth, urinary retention; "
            "CHILDREN tolerate high doses better than adults; "
            "SECOND-LINE: Tetrabenazine (VMAT2 inhibitor) — monoamine depletion; "
            "risk depression and parkinsonism — monitor; "
            "Clonazepam / diazepam — adjunct for task-specific relief; "
            "Baclofen oral or INTRATHECAL for severe generalised; "
            "SURGERY — DBS-GPi: HIGHLY EFFECTIVE (>70% BFMDRS improvement); "
            "DBS-GPi first-line surgery for generalised DYT-TOR1A with functional impairment; "
            "Bilateral GPi targets; programming key to outcome; "
            "DBS efficacy better the younger the implant; "
            "BOTULINUM TOXIN: for focal involvement (cervical, writer's cramp) as adjunct; "
            "SENSORY TRICK (geste antagoniste): many patients have transient relief with touch — exploitable"
        ),
        "contraindications": (
            "AVOID DOPAMINE BLOCKERS (haloperidol, metoclopramide, chlorpromazine): "
            "may worsen DYT1 dystonia and precipitate acute dystonic reaction; "
            "CAUTION LEVODOPA — dystonia worsens in TOR1A (unlike GCH1 which responds); "
            "L-DOPA TRIAL IS NOT RECOMMENDED as primary treatment for DYT1; "
            "AVOID HIGH-DOSE ANTICHOLINERGICS IN ELDERLY: urinary retention, cognitive decline; "
            "TETRABENAZINE: avoid in depression — suicide risk; "
            "DO NOT MISTAKE FOR CONVERSION DISORDER: imaging normal but real organic condition; "
            "AVOID NECK SURGERY without neurological review (cervical dystonia may require DBS not surgery)"
        ),
        "monitoring": (
            "BFMDRS (Burke-Fahn-Marsden Dystonia Rating Scale): at diagnosis + 6-monthly; "
            "TRIHEXYPHENIDYL ESCALATION: monthly visits during titration; "
            "DBS PROGRAMMING: 3-monthly post-implant first year, then 6-monthly; "
            "OPHTHALMOLOGY: if high-dose anticholinergic (intraocular pressure); "
            "SWALLOWING: SLT assessment if oropharyngeal involvement; "
            "SCHOOL/OCCUPATIONAL THERAPY: for hand/limb involvement — adaptive aids; "
            "TOR1A GENETIC COUNSELLING: 50% offspring risk; penetrance 30-40% (most carriers unaffected); "
            "FAMILY CASCADE: offer testing to first-degree relatives — penetrance counselling; "
            "PSYCHIATRIC: depression in chronic dystonia — annual PHQ-9; "
            "DBS BATTERY: monitor impedance and battery life 6-monthly (10-15 year lifespan typical)"
        ),
        "lifecycle": [
            "Childhood (6-12 yr): foot inversion onset, misdiagnosis as orthopaedic common",
            "Adolescence (12-18 yr): spread to trunk/arms, school impact, THP trial",
            "Young adult (18-30 yr): DBS evaluation if functional impairment",
            "Adult (30-50 yr): DBS maintenance, battery replacement, programming adjustments",
            "Mid-adult (50+ yr): hardware maintenance, cognitive effects of anticholinergics monitored",
            "Family planning: 50% transmission risk; penetrance counselling; prenatal testing available",
        ],
        "concepts": [
            "DYT1 / DYT-TOR1A: most common genetic generalised dystonia worldwide",
            "GAG deletion: 3-nucleotide deletion removing Glu302 or Glu303 from Torsin-1A",
            "Penetrance 30-40%: most TOR1A carriers are unaffected (modifier genes)",
            "Lower limb onset: pathognomonic for DYT1 (foot/ankle first)",
            "DBS-GPi: >70% improvement in BFMDRS — gold-standard surgical treatment",
            "No diurnal variation: distinguishes DYT1 from DRD (GCH1/TH) — critical DDx",
            "Trihexyphenidyl: high-dose anticholinergic — tolerated better by children",
            "Normal MRI: structural neuroimaging cannot confirm or exclude DYT1",
            "Sensory trick (geste antagoniste): transient dystonia relief with touch",
            "Action-induced: dystonia worse with movement, may spill over to rest",
            "No cognitive involvement: pure motor disorder (unlike KMT2B 40% ID)",
            "Tetrabenazine caution: depression risk — baseline mood assessment required",
            "Ashkenazi founder: 1 in 2000-3000 Ashkenazi Jewish carry GAG deletion",
            "Bilateral DBS: both GPi electrodes — response better than unilateral",
            "Early DBS: younger age at implant → better dystonia response",
        ],
        "thresholds": [
            "BFMDRS >20: consider DBS-GPi referral if pharmacology inadequate",
            "Trihexyphenidyl dose >4 mg without effect in 4 weeks: escalate or switch strategy",
            "Age <7 years: avoid high-dose anticholinergics (cognitive effects in development)",
            "DBS battery voltage <2.8 V: plan replacement within 3 months",
            "Depression PHQ-9 >10: pause tetrabenazine; psychiatric referral",
            "Penetrance 30-40%: counsel families that most carriers will not develop dystonia",
        ],
        "standards": [
            "ESDA European Dystonia Consortium guidelines 2021",
            "EFNS/MDS-ES recommendations for DBS in dystonia",
            "BFMDRS Burke-Fahn-Marsden Dystonia Rating Scale (validated outcome measure)",
            "ACMG-AMP-2015 variant classification",
            "MDS Task Force dystonia classification 2013",
            "DBS programming guidelines — Neuromodulation Society",
        ],
        "etiologies": [
            {"type": "TOR1A GAG deletion het — DYT1 Generalised", "pct": 62},
            {"type": "TOR1A GAG deletion het — DYT1 Focal (reduced penetrance phenotype)", "pct": 18},
            {"type": "TOR1A Atypical/Missense — Non-GAG variant", "pct": 8},
            {"type": "TOR1A GAG deletion + modifier — Severe Early-Onset", "pct": 8},
            {"type": "Phenocopy (negative TOR1A, DYT1 clinical)", "pct": 4},
        ],
        "seizure_types": [
            {"type": "Generalised Dystonia (trunk + limbs)", "pct": 72},
            {"type": "Multifocal Dystonia (2-3 body regions)", "pct": 45},
            {"type": "Focal Limb Dystonia (single limb)", "pct": 32},
            {"type": "Axial Dystonia (trunk predominant)", "pct": 28},
            {"type": "Dystonic Storm (status dystonicus)", "pct": 5},
        ],
        "triggers": [
            {"trigger": "Voluntary Movement / Action", "pct": 95},
            {"trigger": "Stress / Anxiety", "pct": 68},
            {"trigger": "Fatigue", "pct": 62},
            {"trigger": "Sleep Deprivation", "pct": 48},
            {"trigger": "Caffeine / Stimulants", "pct": 35},
            {"trigger": "Intercurrent Illness (fever)", "pct": 28},
            {"trigger": "Hormonal Changes (menstrual)", "pct": 22},
            {"trigger": "Missed Medication", "pct": 45},
        ],
        "references": [
            "Ozelius LJ 1997 Nat Genet (TOR1A GAG deletion discovery)",
            "Albanese A 2013 Mov Disord (dystonia classification MDS)",
            "Kupsch A 2006 NEJM (DBS-GPi for DYT1 RCT)",
            "Vidailhet M 2005 NEJM (DBS bilateral GPi)",
            "Jankovic J 2013 Lancet Neurol (dystonia therapeutics review)",
            "Siokas V 2019 Front Neurol (TOR1A penetrance genetics)",
        ],
    },
    # -- SGCE — DYT11 / Myoclonus-Dystonia --------------------------------------
    {
        "gene": "SGCE",
        "alt_name": (
            "SGCE (SGCE-437aa-7q21.3 / AD-Paternal-Imprint — DYT-SGCE-DYT11-Myoclonus-Dystonia — "
            "MYOCLONUS-LIGHTNING-JERKS-PREDOMINATE-PATHOGNOMONIC — "
            "ALCOHOL-RESPONSIVE-Diagnostic-Not-Treatment — "
            "PATERNAL-IMPRINTING-Maternal-Allele-Silent)"
        ),
        "protein": (
            "SGCE -- 7q21.3 AD-paternal-imprint -- SGCE-437aa -- "
            "Epsilon-Sarcoglycan-Dystroglycan-Complex-Muscle-Neuronal-Scaffold -- "
            "DYT-SGCE-Myoclonus-Dystonia-DYT11-OMIM-159900 -- "
            "PATERNAL-IMPRINT-Maternal-SGCE-Allele-Silenced-Paternal-Allele-Expressed -- "
            "MYOCLONUS-PREDOMINATES-Lightning-Fast-Millisecond-Jerks-Arms-Neck-Trunk -- "
            "Dystonia-Cervical-Writer-Cramp-Milder-Component -- "
            "ALCOHOL-TRANSIENT-SUPPRESSION-Ethanol-1-drink-Relieves-Myoclonus-50-80pct -- "
            "Psychiatric-Comorbidity-OCD-30pct-Depression-50pct-Anxiety -- "
            "SSRI-Risk-Serotonin-Myoclonus-Aggravation-MONITOR -- "
            "Clonazepam-Alcohol-Same-GABA-Mechanism-Explains-Ethanol-Response -- "
            "7q21.3"
        ),
        "locus": "7q21.3",
        "protein_size": "437 aa",
        "inheritance": (
            "AD (autosomal dominant) with PATERNAL IMPRINTING; "
            "maternal SGCE allele is epigenetically silenced — only paternal copy expressed; "
            "therefore: mutation on PATERNAL chromosome → affected child; "
            "mutation on MATERNAL chromosome → UNAFFECTED (silenced); "
            "KEY: test parent of origin before counselling; "
            "50% offspring risk IF inherited from father; 0% if from mother; "
            "de novo mutations: 30-40% (apparent sporadic — maternal silencing may masquerade)"
        ),
        "age_of_onset": "Childhood 5-20 yr; myoclonus typically precedes dystonia",
        "pathognomonic": (
            "MYOCLONUS PREDOMINATES — lightning-fast (<100 ms) involuntary jerks of arms/neck/trunk; "
            "Dystonia (cervical or upper limb) is milder secondary feature; "
            "ALCOHOL RESPONSIVE: 1-2 units ethanol suppress myoclonus 50-80% temporarily; "
            "ethanol test has DIAGNOSTIC utility (ethanol = GABA-A potentiator = same mechanism as clonazepam); "
            "PATERNAL IMPRINTING: test parent of mutation to confirm paternal origin before counselling; "
            "Psychiatric comorbidities PATHOGNOMONIC in some families (OCD 30%, depression 50%); "
            "SSRI-aggravated myoclonus: SSRIs increase serotonin → worsen myoclonus in some patients; "
            "EEG cortical correlate absent (subcortical/cortical mixed origin, unlike cortical myoclonus)"
        ),
        "treatment": (
            "FIRST-LINE: Clonazepam — reduces myoclonus via GABA-A (same mechanism as alcohol); "
            "titrate 0.5-4 mg/day; sedation dose-limiting; "
            "LEVETIRACETAM — second-line anti-myoclonic; well tolerated; "
            "ALCOHOL: NOT a treatment — ethanol dependence risk is HIGH in this condition; "
            "educate patients explicitly: alcohol is diagnostic clue, NOT a therapeutic strategy; "
            "VALPROATE: some evidence for myoclonus suppression in M-D (Level C); "
            "avoid in females of reproductive age (teratogenicity); "
            "BOTULINUM TOXIN: for cervical dystonia or upper limb dystonia component; "
            "DBS-GPi or THALAMIC (Vim/VoA): effective for severe cases — bilateral stimulation; "
            "PSYCHIATRIC: treat OCD (CBT preferred; SSRI caution — monitor myoclonus worsening); "
            "Treat depression (avoid serotonergic if myoclonus worsens on SSRI); "
            "PHYSICAL THERAPY: proprioceptive exercises; OCCUPATIONAL THERAPY: task modification"
        ),
        "contraindications": (
            "AVOID ALCOHOL AS TREATMENT: ethanol dependence rate VERY HIGH in M-D (up to 30%); "
            "explicitly counsel: alcohol suppresses symptoms but causes addiction — do not self-medicate; "
            "SSRI CAUTION: serotonin excess worsens myoclonus in some SGCE patients; "
            "start low, monitor carefully; switch to SNRI or bupropion if worsening; "
            "CAUTION DOPAMINE BLOCKERS: may worsen dystonia; "
            "AVOID HIGH CLONAZEPAM ABRUPT CESSATION: seizure risk — taper only; "
            "VALPROATE in females: teratogenicity risk — contraceptive counselling; "
            "DO NOT COUNSEL AS 50% RISK without determining parent of origin — maternal mutation = 0% risk"
        ),
        "monitoring": (
            "FAHN-TOLOSA-MARIN (FTM) tremor rating scale for myoclonus severity monthly during titration; "
            "ALCOHOL USE: CAGE screening 6-monthly; addiction risk HIGH; "
            "PSYCHIATRIC: PHQ-9 depression + OCD-YBOCS 6-monthly; "
            "SSRI MONITORING: myoclonus frequency diary after SSRI initiation; "
            "GENETIC — PARENT OF ORIGIN: confirm paternal transmission before counselling; "
            "DBS programming: 3-monthly first year (thalamic or GPi); "
            "LIVER FUNCTION: if valproate used; "
            "SGCE FAMILY CASCADE: first-degree relatives; paternal imprinting counselling mandatory"
        ),
        "lifecycle": [
            "Childhood (5-12 yr): myoclonus appears first; school disruption; clonazepam initiation",
            "Adolescence (12-18 yr): alcohol discovery (self-medicate risk); psychiatric comorbidities emerge",
            "Young adult (18-30 yr): alcohol dependence vigilance; DBS evaluation if refractory",
            "Adult (30-50 yr): DBS maintenance; psychiatric treatment; family planning",
            "Reproductive age: valproate teratogenicity — switch before conception",
            "Family planning: paternal imprint counselling; parent-of-origin testing",
        ],
        "concepts": [
            "DYT-SGCE: myoclonus-dystonia, myoclonus PREDOMINATES over dystonia",
            "Paternal imprinting: maternal SGCE allele silenced — only paternal copy active",
            "Alcohol-responsive: ethanol suppresses myoclonus (GABA-A) — diagnostic not therapeutic",
            "Ethanol dependence: up to 30% of M-D patients develop alcohol use disorder",
            "SSRI caution: serotonin excess may worsen myoclonus — baseline and monitor",
            "Clonazepam: same mechanism as alcohol (GABA-A) — preferred pharmacotherapy",
            "Parent-of-origin testing: mandatory before genetic counselling — maternal = 0% risk",
            "OCD comorbidity: 30% of SGCE patients; CBT preferred over SSRI",
            "Depression comorbidity: 50% — high burden; treat but monitor serotonin-myoclonus",
            "DBS targets: GPi (dystonia predominant) vs Vim/VoA (myoclonus predominant)",
            "Subcortical origin: EEG cortical correlate typically absent in SGCE myoclonus",
            "No cognitive impairment in SGCE (unlike KMT2B which has 40% ID)",
            "Levetiracetam: well-tolerated second-line anti-myoclonic",
            "ε-Sarcoglycan: component of dystroglycan complex in brain and muscle",
            "De novo rate 30-40%: apparent sporadic cases often have new mutation",
        ],
        "thresholds": [
            "CAGE ≥2: alcohol use disorder — addictions referral before DBS evaluation",
            "OCD-YBOCS >16: OCD treatment required; prefer CBT over SSRI",
            "Clonazepam >4 mg/day without adequate response: DBS evaluation",
            "SSRI initiation: myoclonus diary for 4 weeks post-start",
            "Ethanol use >14 units/week: formal dependence assessment mandatory",
            "Valproate female of reproductive age: switch before conception planning",
        ],
        "standards": [
            "MDS Task Force Myoclonus-Dystonia Classification",
            "ESDA Myoclonus-Dystonia Management Guidelines",
            "Consensus statement SGCE imprinting (van der Salm 2012 Brain)",
            "ACMG-AMP-2015 variant classification",
            "WHO ICD-11 movement disorders classification",
            "CAGE alcohol screening validated tool",
        ],
        "etiologies": [
            {"type": "SGCE Nonsense/Frameshift — LOF Paternal — Classic M-D", "pct": 48},
            {"type": "SGCE Missense — Partial LOF Paternal", "pct": 25},
            {"type": "SGCE Large Deletion — Contiguous gene (7q21)", "pct": 12},
            {"type": "SGCE Maternal Mutation — Phenotypically Unaffected (imprinting)", "pct": 8},
            {"type": "SGCE Negative Phenocopy (other myoclonus-dystonia genes)", "pct": 7},
        ],
        "seizure_types": [
            {"type": "Myoclonus (lightning jerks arms/neck dominant)", "pct": 88},
            {"type": "Cervical Dystonia + Myoclonus", "pct": 58},
            {"type": "Upper Limb Dystonia (writer's cramp + jerks)", "pct": 42},
            {"type": "Trunk Myoclonus", "pct": 35},
            {"type": "Generalised Myoclonus-Dystonia", "pct": 18},
        ],
        "triggers": [
            {"trigger": "Voluntary Movement / Action", "pct": 82},
            {"trigger": "Stress / Anxiety", "pct": 75},
            {"trigger": "Fatigue", "pct": 68},
            {"trigger": "Missed Clonazepam Dose", "pct": 62},
            {"trigger": "Sleep Deprivation", "pct": 48},
            {"trigger": "SSRI Initiation", "pct": 30},
            {"trigger": "Caffeine", "pct": 25},
            {"trigger": "Alcohol Withdrawal", "pct": 20},
        ],
        "references": [
            "Zimprich A 2001 Nat Genet (SGCE Myoclonus-Dystonia discovery)",
            "Nardocci N 2008 Neurology (SGCE phenotype spectrum)",
            "Grunewald A 2008 Neurology (alcohol and M-D)",
            "van der Salm SM 2012 Brain (SGCE imprinting paternal)",
            "Mencacci NE 2015 Am J Hum Genet (M-D genetics update)",
            "Espay AJ 2018 Mov Disord (DBS in myoclonus-dystonia)",
        ],
    },
    # -- GCH1 — DRD / DYT5a / Segawa Syndrome -----------------------------------
    {
        "gene": "GCH1",
        "alt_name": (
            "GCH1 (GCH1-250aa-14q22.2 / AD — DRD-DYT-GCH1-DYT5a-Segawa-Syndrome — "
            "L-DOPA-CURATIVE-MUST-TRY-BEFORE-BOTOX-IN-CHILDHOOD-DYSTONIA — "
            "DIURNAL-VARIATION-BETTER-MORNING-WORSE-EVENING-PATHOGNOMONIC — "
            "TREATABLE-NEVER-MISS)"
        ),
        "protein": (
            "GCH1 -- 14q22.2 AD -- GCH1-250aa -- "
            "GTP-Cyclohydrolase-I-Tetrahydrobiopterin-Synthesis-First-Committed-Step -- "
            "BH4-Cofactor-Aromatic-Amino-Acid-Hydroxylases-TH-PAH-TPH -- "
            "DRD-Dopa-Responsive-Dystonia-DYT5a-Segawa-Syndrome-OMIM-128230 -- "
            "BH4-Deficiency-Reduces-TH-Activity-Reduces-Dopamine-Nigro-Striatal -- "
            "L-DOPA-COMPLETELY-CURATIVE-3-5-mg-kg-Day-Low-Dose-SUFFICIENT -- "
            "DIURNAL-VARIATION-PATHOGNOMONIC-Better-Morning-After-Rest-Worse-Evening -- "
            "Walk-On-Toes-Foot-Dystonia-Child-Misdiagnosed-As-Cerebral-Palsy -- "
            "DO-NOT-USE-BOTOX-BEFORE-L-DOPA-TRIAL-IN-CHILDHOOD-DYSTONIA -- "
            "Female-Predominant-4:1-Incomplete-Penetrance -- "
            "14q22.2"
        ),
        "locus": "14q22.2",
        "protein_size": "250 aa",
        "inheritance": (
            "AD (autosomal dominant); incomplete penetrance; "
            "female predominance 4:1 (females more severely affected); "
            "female penetrance ~85%; male penetrance ~40%; "
            "biallelic GCH1 = severe hyperphenylalaninaemia + infantile encephalopathy (rare); "
            "monoallelic = classic DRD (Segawa syndrome); "
            "point mutations: truncating > missense > deletions; "
            "CSF BH4 / neopterin analysis confirms biochemistry; "
            "L-DOPA trial has 100% diagnostic accuracy in GCH1 DRD"
        ),
        "age_of_onset": "Childhood 1-12 yr (typically 3-8 yr); rare adult onset",
        "pathognomonic": (
            "DIURNAL VARIATION — better in morning after sleep, progressively worse through day: PATHOGNOMONIC; "
            "Patient walks better in the morning; by evening can barely walk; "
            "FOOT DYSTONIA / EQUINOVARUS — walking on toes or foot turning in: presenting sign; "
            "MISDIAGNOSIS AS CEREBRAL PALSY: common, especially if birth history slightly abnormal; "
            "L-DOPA TRIAL: COMPLETE RESOLUTION with small doses (3-5 mg/kg/day) = diagnostic and curative; "
            "MUST TRY L-DOPA IN ANY CHILD WITH UNEXPLAINED DYSTONIA before botox or orthopaedic surgery; "
            "Hyperreflexia often present (dopamine deficit reduces D2 inhibition at spinal cord); "
            "Normal MRI brain; FDG-PET and DAT-SPECT normal (presynaptic intact, post-synaptic intact); "
            "Female predominance 4:1 (incomplete penetrance sex-modified)"
        ),
        "treatment": (
            "L-DOPA / CARBIDOPA: CURATIVE — complete resolution of dystonia; "
            "Start: levodopa 0.5-1 mg/kg/day in 3 divided doses (with carbidopa 1:4 ratio); "
            "Target: 3-5 mg/kg/day; typical final dose 50-300 mg levodopa/day (LOW compared to PD); "
            "RESPONSE EXPECTED WITHIN DAYS TO WEEKS — dramatic improvement; "
            "LIFELONG treatment required — stopping → relapse; "
            "NEVER INCREASES DOSE LIKE PARKINSON'S: GCH1-DRD responds to low dose permanently; "
            "Pramipexole/ropinirole (DA agonists): alternative if L-DOPA intolerant; "
            "DIETARY: no specific restrictions; "
            "NO BOTULINUM TOXIN needed if L-DOPA adequate; "
            "ORTHOPAEDIC SURGERY: CONTRAINDICATED before L-DOPA trial — foot surgery on dystonic foot is disaster; "
            "PHYSIOTHERAPY: adjunct during dose establishment; "
            "BH4 supplementation (sapropterin): for biallelic GCH1 / AR forms"
        ),
        "contraindications": (
            "BOTULINUM TOXIN BEFORE L-DOPA TRIAL: ABSOLUTE CI — always try L-DOPA first in childhood dystonia; "
            "ORTHOPAEDIC SURGERY BEFORE L-DOPA TRIAL: many children with GCH1-DRD have undergone "
            "unnecessary foot surgery; L-DOPA resolves equinovarus without surgery; "
            "DOPAMINE BLOCKERS (metoclopramide): antagonise treatment — avoid; "
            "STOPPING L-DOPA ABRUPTLY: gradual taper only — acute withdrawal may cause dystonic crisis; "
            "HIGH-DOSE L-DOPA (Parkinson doses): unnecessary and may cause dyskinesia — use low dose; "
            "VALPROATE: consider if seizures present but monitor L-DOPA interaction; "
            "DO NOT LABEL AS CEREBRAL PALSY without GCH1 testing in childhood dystonia with diurnal variation"
        ),
        "monitoring": (
            "RESPONSE TO L-DOPA: daily dystonia diary first 4 weeks; BFMDRS at 4 and 12 weeks; "
            "DOSE OPTIMISATION: monthly until stable; "
            "LONG-TERM L-DOPA: 6-monthly review; avoid dyskinesia (rare at DRD doses); "
            "DIURNAL VARIATION TRACKING: parent/patient diary to confirm response pattern; "
            "BLOOD PRESSURE: orthostatic hypotension on L-DOPA initiation; "
            "GCH1 FAMILY CASCADE: 50% risk to offspring; female relatives more likely symptomatic; "
            "PREGNANCY: L-DOPA in pregnancy — limited data but benefits outweigh risk; "
            "PSYCHIATRIC: anxiety common from years of misdiagnosis; counselling support; "
            "CSF BH4/NEOPTERIN: at diagnosis to confirm biochemistry (optional if molecular confirmed)"
        ),
        "lifecycle": [
            "Infancy-toddler (1-3 yr): toe-walking, foot dystonia noted, often misattributed",
            "Childhood (3-8 yr): classic presentation; diurnal variation observed; misdiagnosis common",
            "School age (8-12 yr): L-DOPA initiation → dramatic resolution; return to normal activity",
            "Adolescence (12-18 yr): stable on L-DOPA; normal neurological development continues",
            "Adulthood: lifelong L-DOPA; dose rarely needs escalation; full functional life",
            "Pregnancy: L-DOPA continues; counselling; 50% risk to children",
        ],
        "concepts": [
            "DRD: Dopa-Responsive Dystonia — GCH1 is commonest cause (AD form)",
            "GCH1: GTP cyclohydrolase I — first enzyme in BH4 (tetrahydrobiopterin) synthesis",
            "BH4: cofactor for TH (tyrosine hydroxylase) — BH4 deficiency → dopamine deficiency",
            "Diurnal variation: PATHOGNOMONIC — better morning, worse evening (dopamine depletes during day)",
            "L-DOPA curative: complete resolution at low doses (3-5 mg/kg/day)",
            "NEVER botox first: always try L-DOPA in childhood lower limb dystonia",
            "Misdiagnosis as CP: most common misdiagnosis — diurnal variation clue",
            "Female predominance 4:1: sex-modified penetrance",
            "Low dose forever: unlike Parkinson's, DRD doses remain low lifelong",
            "BH4 synthesis: GCH1 → BH4 → TH active → tyrosine → L-DOPA → dopamine",
            "FDG-PET/DAT normal: helps distinguish from early-onset Parkinson's",
            "Foot equinovarus: presenting sign — toe-walking, foot inversion on walking",
            "Orthopaedic surgery avoidance: L-DOPA resolves deformity without surgery",
            "Complete penetrance at 10 mg/kg/day: diagnostic trial dose",
            "Lifelong treatment: stopping = relapse; emphasise compliance",
        ],
        "thresholds": [
            "Diurnal variation + childhood dystonia: L-DOPA trial MANDATORY before any other intervention",
            "L-DOPA 5 mg/kg/day × 4 weeks: no response → reconsider diagnosis (but extend to 8 weeks)",
            "L-DOPA induced dyskinesia at DRD doses: reduce dose by 25% — very rare",
            "Female first-degree relative of GCH1 carrier: 85% lifetime penetrance risk",
            "Orthopaedic referral for dystonic foot: PAUSE — GCH1 testing first",
            "L-DOPA >10 mg/kg/day in child: unlikely DRD — reconsider diagnosis",
        ],
        "standards": [
            "EFNS/MDS-ES Guidelines for DRD management",
            "Ichinose H 1994 Nat Genet (GCH1 discovery)",
            "Furukawa Y 2002 Ann Neurol (DRD clinical guidelines)",
            "ACMG-AMP-2015 variant classification",
            "Kurian MA 2011 Lancet Neurol (DRD review)",
            "NICE-NG217 Movement Disorders",
        ],
        "etiologies": [
            {"type": "GCH1 Truncating (Nonsense/Frameshift) het — Classic DRD", "pct": 55},
            {"type": "GCH1 Missense het — DRD (variable severity)", "pct": 30},
            {"type": "GCH1 Large Deletion het — DRD", "pct": 8},
            {"type": "GCH1 Biallelic (AR) — Severe HPAenia + encephalopathy", "pct": 4},
            {"type": "GCH1 Phenocopy (TH/SPR/other BH4 enzyme)", "pct": 3},
        ],
        "seizure_types": [
            {"type": "Foot/Lower Limb Dystonia (equinovarus)", "pct": 88},
            {"type": "Multifocal Dystonia (lower + upper limbs)", "pct": 52},
            {"type": "Generalised Dystonia (severe, late presentation)", "pct": 18},
            {"type": "Cervical Dystonia (adult onset)", "pct": 15},
            {"type": "Parkinsonism Features (untreated adult)", "pct": 10},
        ],
        "triggers": [
            {"trigger": "Afternoon / Evening (diurnal worsening)", "pct": 98},
            {"trigger": "Exercise / Prolonged Walking", "pct": 85},
            {"trigger": "Stress", "pct": 62},
            {"trigger": "Missed L-DOPA Dose", "pct": 78},
            {"trigger": "Illness / Fever", "pct": 35},
            {"trigger": "Sleep Deprivation", "pct": 28},
            {"trigger": "Cold Weather", "pct": 22},
            {"trigger": "Caffeine (minor)", "pct": 15},
        ],
        "references": [
            "Ichinose H 1994 Nat Genet (GCH1 mutations in DRD)",
            "Segawa M 1976 Adv Neurol (original DRD description Segawa)",
            "Furukawa Y 2002 Ann Neurol (GCH1 spectrum)",
            "Kurian MA 2011 Lancet Neurol (DRD comprehensive review)",
            "Tadic V 2012 Neurology (DRD long-term outcomes)",
            "Charlesworth G 2013 Hum Mutat (GCH1 genotype-phenotype)",
        ],
    },
    # -- TH — DYT5b / AR-DRD ---------------------------------------------------
    {
        "gene": "TH",
        "alt_name": (
            "TH (TH-528aa-11p15.5 / AR — DRD-DYT-TH-DYT5b-Tyrosine-Hydroxylase-Deficiency — "
            "SEVERE-INFANTILE-ENCEPHALOPATHY-L-DOPA-Responsive-LOWER-DOSES — "
            "BIALLELIC-LOF-Reduces-Catecholamine-Synthesis-AR-DRD)"
        ),
        "protein": (
            "TH -- 11p15.5 AR -- TH-528aa -- "
            "Tyrosine-Hydroxylase-Rate-Limiting-Catecholamine-Synthesis-Enzyme -- "
            "BH4-Dependent-Aromatic-Amino-Acid-Hydroxylase-Converts-Tyrosine-to-L-DOPA -- "
            "DRD-Tyrosine-Hydroxylase-Deficiency-DYT5b-OMIM-605407 -- "
            "AR-Biallelic-More-Severe-Than-GCH1-AD-DRD -- "
            "Type-A-L-DOPA-Responsive-Milder-Compound-Het -- "
            "Type-B-L-DOPA-Poorly-Responsive-Truncating-Mutations-Severe-Neonatal -- "
            "Infantile-Encephalopathy-Hypotonia-Parkinsonian-Features-in-Infants -- "
            "HVA-5HIAA-Reduced-CSF-DIAGNOSTIC-PATHOGNOMONIC -- "
            "Pterin-Profile-Normal-unlike-GCH1-deficiency -- "
            "11p15.5"
        ),
        "locus": "11p15.5",
        "protein_size": "528 aa",
        "inheritance": (
            "AR (autosomal recessive); biallelic loss-of-function; "
            "compound heterozygous most common (missense + truncating = Type A, better response); "
            "truncating/truncating = Type B (severe, poor L-DOPA response); "
            "parents are obligate carriers — asymptomatic; "
            "25% recurrence risk to siblings; "
            "consanguinity in severe forms; "
            "TH is the rate-limiting enzyme in the catecholamine pathway"
        ),
        "age_of_onset": "Neonatal to infancy (1-6 months) — earlier than GCH1-DRD",
        "pathognomonic": (
            "INFANTILE HYPOTONIA + HYPOKINESIA + RIGIDITY (parkinsonian triad in infant); "
            "CSF: reduced HVA (homovanillic acid) + reduced 5-HIAA = PATHOGNOMONIC (catecholamine/serotonin deficiency); "
            "CSF pterins NORMAL (unlike GCH1-deficiency which has low BH4/neopterin); "
            "L-DOPA RESPONSIVE but at LOWER doses than GCH1-DRD; "
            "Type A (missense/mild): significant L-DOPA response; "
            "Type B (truncating/severe): poor or partial L-DOPA response; "
            "Oculogyric crises: episodic upward eye deviation + dystonic posturing — PATHOGNOMONIC in infants; "
            "Diurnal variation may be present but less striking than GCH1-DRD; "
            "Ptosis, miosis (Horner-like): autonomic catecholamine deficit"
        ),
        "treatment": (
            "L-DOPA/CARBIDOPA: first-line — lower dose than GCH1-DRD; "
            "Start 1 mg/kg/day levodopa; titrate slowly to 3-10 mg/kg/day; "
            "Type A: good response; Type B: partial response — augment with serotonin precursors; "
            "5-HTP (5-hydroxytryptophan): adjunct for serotonin deficiency (reduces oculogyric crises); "
            "CARBIDOPA: co-administer to reduce peripheral conversion; "
            "PYRIDOXINE (B6): some patients benefit — enzyme cofactor; "
            "MONOAMINE OXIDASE INHIBITORS: controversial — not standard; "
            "AVOID SUDDEN L-DOPA STOP: autonomic instability; "
            "RESPIRATORY SUPPORT: neonates may require ventilatory support; "
            "NASOGASTRIC FEEDING: early — poor suck/swallow reflex in neonatal-onset; "
            "PHYSIOTHERAPY: for hypertonia/dystonia management; "
            "OCCUPATIONAL THERAPY: motor skills"
        ),
        "contraindications": (
            "AVOID DOPAMINE BLOCKERS (metoclopramide, haloperidol): worsen dopamine deficiency — ABSOLUTE CI; "
            "AVOID SEROTONIN DEPLETING DRUGS: tetrabenazine in Type B may reduce residual monoamines; "
            "CAUTION HIGH-DOSE L-DOPA IN TYPE B: dyskinesia at low doses may occur; "
            "AVOID PYRIDOXINE MEGADOSE (>50 mg/day): potential peripheral neuropathy; "
            "DO NOT DIAGNOSE AS HYPOTONIC CP without catecholamine metabolite testing; "
            "AVOID ANTICHOLINERGICS IN INFANTS: cognitive and autonomic side effects; "
            "DO NOT STOP L-DOPA ABRUPTLY: severe rebound dystonia"
        ),
        "monitoring": (
            "CSF HVA + 5-HIAA: at diagnosis and 6-monthly for dose adjustment; "
            "L-DOPA PLASMA LEVELS: peak and trough during dose adjustment; "
            "OCULOGYRIC CRISIS FREQUENCY: diary; aim for elimination with adequate dosing; "
            "PTERIN PROFILE: once at diagnosis to exclude GCH1 (BH4 normal in TH-DRD); "
            "DEVELOPMENTAL MILESTONES: monthly in first 2 years; "
            "FEEDING/SWALLOWING: SLT 3-monthly; "
            "RESPIRATORY: sleep study 6-monthly for hypoventilation; "
            "TH FAMILY CASCADE: siblings 25% risk; prenatal testing available"
        ),
        "lifecycle": [
            "Neonatal (0-4 wk): hypotonia, poor feeding, oculogyric crises, NG tube",
            "Infancy (1-12 mo): L-DOPA initiation, oculogyric crisis control, developmental assessment",
            "Toddler (1-3 yr): L-DOPA optimisation, motor delay, physiotherapy",
            "Childhood (3-12 yr): stable treatment, school support, adaptive equipment",
            "Adolescence (12-18 yr): dose adjustment for weight, independence planning",
            "Adulthood: ongoing L-DOPA; variable outcome (Type A near-normal; Type B significant disability)",
        ],
        "concepts": [
            "TH: tyrosine hydroxylase — rate-limiting enzyme in catecholamine pathway",
            "AR-DRD (DYT5b): more severe than AD-DRD (GCH1/DYT5a) — biallelic required",
            "CSF HVA reduced: catecholamine deficiency biomarker",
            "CSF 5-HIAA reduced: serotonin deficiency — cofactor for oculogyric crisis management",
            "Pterin profile NORMAL: distinguishes TH-DRD from GCH1-deficiency (pterins reduced in GCH1)",
            "Type A vs B: compound het missense (A, better) vs truncating (B, worse response)",
            "Oculogyric crises: episodic upward eye deviation + dystonia — PATHOGNOMONIC in infants",
            "L-DOPA lower dose: TH-DRD uses lower L-DOPA than GCH1-DRD typically",
            "Infantile parkinsonism: hypokinesia + rigidity + hypotonia in neonate",
            "5-HTP adjunct: helps serotonin deficiency component",
            "Diurnal variation: present but less striking than GCH1-DRD",
            "Dopamine blockers ABSOLUTE CI: worsen already-depleted dopamine",
            "CSF diagnosis critical: cannot distinguish on clinical grounds from other hypotonic infants",
            "Catecholamine pathway: TH converts Tyr→L-DOPA; DOPA decarboxylase → dopamine",
            "Neonatal onset: earlier than GCH1-DRD which presents in childhood",
        ],
        "thresholds": [
            "CSF HVA <200 nmol/L: TH deficiency likely — confirm with TH sequencing",
            "Oculogyric crisis >3 per week: L-DOPA dose inadequate — titrate up",
            "L-DOPA >15 mg/kg/day in infant: consider alternative diagnosis or Type B",
            "Peak plasma L-DOPA <1200 ng/mL: under-absorption — formulation review",
            "Pterin profile: BH4 normal in TH-DRD; low in GCH1 — distinguishing test",
            "Type B: realistic outcome goal is reduction not elimination of motor deficits",
        ],
        "standards": [
            "Willemsen MA 2010 Neurology (TH deficiency clinical spectrum)",
            "Brun L 2010 J Inherit Metab Dis (TH deficiency treatment)",
            "Verbeek MM 2007 Clin Chim Acta (CSF monoamine metabolites)",
            "ACMG-AMP-2015 variant classification",
            "BIOPTERIN-GROUP European metabolic neurology standards",
            "SSIEM guidelines for inherited metabolic neurology",
        ],
        "etiologies": [
            {"type": "TH Missense/Missense biallelic — Type A (L-DOPA responsive)", "pct": 42},
            {"type": "TH Missense + Truncating — Type A intermediate", "pct": 28},
            {"type": "TH Truncating/Truncating — Type B (poor L-DOPA response)", "pct": 18},
            {"type": "TH Single Missense + VUS — Uncertain", "pct": 8},
            {"type": "TH Phenocopy (AADC/DOPA decarboxylase deficiency)", "pct": 4},
        ],
        "seizure_types": [
            {"type": "Oculogyric Crises (episodic upward eye deviation + dystonia)", "pct": 72},
            {"type": "Generalised Hypotonia + Dystonia (infantile)", "pct": 65},
            {"type": "Parkinsonian Features (rigidity + hypokinesia)", "pct": 55},
            {"type": "Focal Limb Dystonia", "pct": 30},
            {"type": "Axial Dystonia (truncal)", "pct": 25},
        ],
        "triggers": [
            {"trigger": "Missed L-DOPA Dose", "pct": 88},
            {"trigger": "Stress / Illness / Fever", "pct": 72},
            {"trigger": "Afternoon / Evening (diurnal)", "pct": 60},
            {"trigger": "Sleep Deprivation", "pct": 48},
            {"trigger": "Emotional Excitement", "pct": 35},
            {"trigger": "Cold Exposure", "pct": 22},
            {"trigger": "Dopamine Blocker Exposure", "pct": 18},
            {"trigger": "Fasting / Hypoglycaemia", "pct": 15},
        ],
        "references": [
            "Ludecke B 1995 Hum Genet (TH mutations DRD)",
            "Willemsen MA 2010 Neurology (TH deficiency spectrum 18 patients)",
            "Brun L 2010 J Inherit Metab Dis (TH management)",
            "Nardocci N 2003 Neurology (DRD types A and B)",
            "Verbeek MM 2007 Clin Chim Acta (CSF metabolomics)",
            "Kulak W 2019 Brain Dev (TH deficiency long-term outcomes)",
        ],
    },
    # -- KMT2B — DYT28 ----------------------------------------------------------
    {
        "gene": "KMT2B",
        "alt_name": (
            "KMT2B (KMT2B-3969aa-19q13.12 / AD-90pct-De-Novo — DYT-KMT2B-DYT28 — "
            "CHILDHOOD-FOCAL-DYSTONIA-GENERALISES-PATHOGNOMONIC — "
            "INTELLECTUAL-DISABILITY-40pct-PATHOGNOMONIC — "
            "DBS-GPi-HIGHLY-EFFECTIVE-EVEN-WITH-ID)"
        ),
        "protein": (
            "KMT2B -- 19q13.12 AD (90% de novo) -- KMT2B-3969aa -- "
            "Lysine-Methyltransferase-2B-Trithorax-Group-Histone-H3-Lys4-Methyltransferase -- "
            "SET-Domain-CXXC-PHD-Zinc-Finger-Chromatin-Remodelling-Transcriptional-Regulation -- "
            "DYT-KMT2B-Dystonia-28-OMIM-617284 -- "
            "Childhood-Onset-Focal-Dystonia-Foot-Leg-FIRST-Then-Generalises -- "
            "Intellectual-Disability-40pct-Microcephaly-Short-Stature-PATHOGNOMONIC-Constellation -- "
            "DBS-GPi-HIGHLY-EFFECTIVE->80pct-BFMDRS-Even-Cognitive-Impaired-Patients -- "
            "De-Novo-90pct-Frameshift-Nonsense-LOF-Haploinsufficiency -- "
            "Neuroophthalmological-Signs-30pct-Cataracts-Optic-Atrophy -- "
            "Facial-Dysmorphism-Mild-Low-Set-Ears-Bitemporal-Narrowing -- "
            "19q13.12"
        ),
        "locus": "19q13.12",
        "protein_size": "3969 aa",
        "inheritance": (
            "AD (autosomal dominant); 90% de novo; "
            "LOF haploinsufficiency (truncating > missense); "
            "familial cases reported (10%) with variable expressivity; "
            "large gene 3969 aa — requires comprehensive sequencing + CNV analysis; "
            "de novo rate high → negative family history does not exclude; "
            "GENETIC TESTING: WES/WGS preferred; small exon panels may miss large deletions"
        ),
        "age_of_onset": "Childhood 2-10 yr; focal foot/leg onset → generalisation over years",
        "pathognomonic": (
            "FOCAL LOWER LIMB DYSTONIA at onset (foot/ankle) → GENERALISES to all body regions over years; "
            "Generalisation trajectory: foot → trunk → upper limbs → cranial; "
            "INTELLECTUAL DISABILITY ~40% (mild-moderate): distinguishes KMT2B from TOR1A (no ID); "
            "MICROCEPHALY + SHORT STATURE constellation in syndromic cases; "
            "Ophthalmic: cataracts (30%), optic atrophy — PATHOGNOMONIC in KMT2B; "
            "Mild facial dysmorphism: bitemporal narrowing, low-set ears; "
            "DBS-GPi REMARKABLY EFFECTIVE (>80% BFMDRS improvement) even in patients with ID; "
            "DO NOT exclude DBS because of cognitive impairment — KMT2B responds well; "
            "MRI: small volume white matter changes in some — non-specific"
        ),
        "treatment": (
            "PHARMACOLOGICAL (limited benefit, supportive role): "
            "Trihexyphenidyl high-dose: trial first (same as DYT1); "
            "Clonazepam: adjunct for task-specific relief; "
            "Tetrabenazine: limited evidence in KMT2B; "
            "Botulinum toxin: focal symptom relief while awaiting DBS; "
            "SURGERY — DBS-GPi: TREATMENT OF CHOICE — bilateral GPi; "
            ">80% improvement in BFMDRS in several case series; "
            "DBS INDICATED EVEN WITH INTELLECTUAL DISABILITY — outcome equally good; "
            "Early DBS referral: do not wait for severe generalisation; "
            "OPHTHALMOLOGICAL REVIEW: cataract extraction when visually significant; "
            "EDUCATIONAL SUPPORT: 40% have ID — educational psychology assessment; "
            "SPEECH AND LANGUAGE THERAPY: oro-cranial involvement; "
            "PHYSIOTHERAPY: contracture prevention during titration waiting period"
        ),
        "contraindications": (
            "DO NOT EXCLUDE DBS DUE TO INTELLECTUAL DISABILITY: KMT2B responds remarkably well; "
            "AVOID HIGH-DOSE TRIHEXYPHENIDYL IN ID PATIENTS: cognitive worsening — lower target doses; "
            "CAUTION TETRABENAZINE IN DEPRESSION: KMT2B has psychiatric comorbidity; "
            "CAUTION BOTULINUM TOXIN GENERALISED DYSTONIA: insufficient for widespread involvement; "
            "AVOID DELAY IN DBS REFERRAL: early intervention → better outcome"
        ),
        "monitoring": [
            "BFMDRS: 6-monthly; pre-DBS baseline; 3-monthly post-implant first year",
            "OPHTHALMOLOGY: annual slit-lamp for cataracts; OCT for optic atrophy",
            "DEVELOPMENTAL ASSESSMENT: Griffiths/Bayley 6-monthly in childhood",
            "EDUCATIONAL PSYCHOLOGY: school-based support plan annual review",
            "HEAD CIRCUMFERENCE: 6-monthly in childhood (microcephaly tracking)",
            "KMT2B FAMILY: de novo 90% — low recurrence; parental testing to exclude germline mosaic",
            "DBS battery and impedance: 3-monthly first year, then 6-monthly",
            "PSYCHIATRIC: PHQ-9 annual; cognitive testing biennial",
        ],
        "lifecycle": [
            "Toddler (2-5 yr): focal foot dystonia onset; physiotherapy; THP trial",
            "Childhood (5-10 yr): generalisation; school placement; DBS evaluation",
            "Adolescence (10-18 yr): DBS implant (optimal timing); programming; education",
            "Young adult (18-30 yr): DBS maintenance; independent living support",
            "Adult (30+ yr): hardware management; ongoing support; cataract management",
            "Family planning: 90% de novo → low recurrence; germline mosaic testing of parents",
        ],
        "concepts": [
            "DYT28 / DYT-KMT2B: childhood generalised dystonia with ID — 2017 discovery",
            "KMT2B: histone methyltransferase (H3K4me3) — epigenetic chromatin regulation",
            "De novo 90%: most cases sporadic — no family history does not exclude",
            "Focal → generalised: invariable trajectory distinguishes from adult-onset focal dystonia",
            "ID 40%: cognitive impairment NOT a contraindication to DBS",
            "DBS highly effective: >80% BFMDRS — best surgical outcome of any inherited dystonia",
            "Cataracts 30%: ophthalmological surveillance mandatory",
            "Optic atrophy: rare but may cause visual impairment — OCT monitoring",
            "Short stature + microcephaly: syndromic KMT2B constellation",
            "Early DBS referral: do not await severe generalisation",
            "Trihexyphenidyl lower dose: tolerance reduced in ID patients",
            "LOF haploinsufficiency: one functioning copy insufficient for normal development",
            "3969 aa: one of largest human proteins — WGS preferred for full coverage",
            "Phenotype spectrum: pure dystonia to syndromic with ID/eye/growth",
            "Botox bridge: use while DBS evaluation/implant waiting period",
        ],
        "thresholds": [
            "BFMDRS >30 or rapid generalisation: urgent DBS referral",
            "IQ <70: educational support mandatory; DBS still indicated — counsel family",
            "Cataract visual acuity <6/18: ophthalmology referral for extraction",
            "DBS age: no lower limit established; case reports from age 5 yr",
            "Trihexyphenidyl >12 mg/day in ID patient: cognitive monitoring every visit",
            "Generalisation from focal to multifocal: reassess DBS urgency",
        ],
        "standards": [
            "Meyer E 2017 Nat Genet (KMT2B DYT28 discovery)",
            "Zech M 2017 Ann Neurol (KMT2B phenotype expansion)",
            "Cif L 2019 Lancet Neurol (DBS in inherited dystonia)",
            "ACMG-AMP-2015 variant classification",
            "ESDA European Dystonia Consortium guidelines",
            "MDS Task Force dystonia classification",
        ],
        "etiologies": [
            {"type": "KMT2B Truncating (Frameshift/Nonsense) de novo — Classic DYT28", "pct": 55},
            {"type": "KMT2B Missense de novo — DYT28 (variable severity)", "pct": 22},
            {"type": "KMT2B Large Deletion (CNV) — Syndromic DYT28 + ID", "pct": 12},
            {"type": "KMT2B Familial (inherited) — 10% of cases", "pct": 7},
            {"type": "KMT2B Phenocopy (other epigenetic dystonia genes)", "pct": 4},
        ],
        "seizure_types": [
            {"type": "Generalised Dystonia (trunk + all limbs)", "pct": 78},
            {"type": "Multifocal Dystonia (3+ body regions)", "pct": 55},
            {"type": "Focal Lower Limb (initial presentation)", "pct": 45},
            {"type": "Cranial-Cervical Dystonia (dysarthria/dysphagia)", "pct": 30},
            {"type": "Status Dystonicus (dystonic storm)", "pct": 8},
        ],
        "triggers": [
            {"trigger": "Voluntary Movement", "pct": 90},
            {"trigger": "Stress / Anxiety", "pct": 70},
            {"trigger": "Missed Medication", "pct": 60},
            {"trigger": "Fatigue", "pct": 65},
            {"trigger": "Illness / Fever", "pct": 45},
            {"trigger": "Sleep Deprivation", "pct": 40},
            {"trigger": "Emotional Excitement", "pct": 35},
            {"trigger": "Cold (peripheral stimulus)", "pct": 18},
        ],
        "references": [
            "Meyer E 2017 Nat Genet (KMT2B DYT28 discovery)",
            "Zech M 2017 Ann Neurol (KMT2B genotype-phenotype)",
            "Cif L 2019 Lancet Neurol (DBS DYT28)",
            "Carecchio M 2019 Mov Disord (DYT-KMT2B clinical series)",
            "Maudet A 2020 Neurology (KMT2B DBS outcomes)",
            "ESDA 2021 Dystonia Guidelines",
        ],
    },
    # -- THAP1 — DYT6 -----------------------------------------------------------
    {
        "gene": "THAP1",
        "alt_name": (
            "THAP1 (THAP1-213aa-8p21.3 / AD — DYT-THAP1-DYT6 — "
            "LARYNGEAL-CRANIAL-INVOLVEMENT-PATHOGNOMONIC-Young-Adult — "
            "ASHKENAZI-JEWISH-FOUNDER-ENRICHMENT — "
            "BOTULINUM-TOXIN-FOR-CERVICAL-DBS-FOR-SEVERE)"
        ),
        "protein": (
            "THAP1 -- 8p21.3 AD -- THAP1-213aa -- "
            "THAP-Domain-Containing-Protein-1-Zinc-Finger-BED-DNA-Binding-Transcription-Factor -- "
            "Proapoptotic-Target-Gene-Regulation-Endothelial-Cell-G1-S-Cell-Cycle -- "
            "DYT-THAP1-Dystonia-6-OMIM-602629 -- "
            "Young-Adult-Onset-12-50yr-Mixed-Focal-Segmental-Generalised -- "
            "LARYNGEAL-DYSTONIA-Hoarseness-Strained-Voice-PATHOGNOMONIC-THAP1 -- "
            "Cranio-Cervical-Involvement-Blepharospasm-Torticollis-Oromandibular -- "
            "Upper-Limb-Less-Common-Writers-Cramp-10-20pct -- "
            "ASHKENAZI-JEWISH-FOUNDER-C-terminal-mutations-p.Pro41 -- "
            "Incomplete-Penetrance-60pct -- "
            "8p21.3"
        ),
        "locus": "8p21.3",
        "protein_size": "213 aa",
        "inheritance": (
            "AD (autosomal dominant); incomplete penetrance ~60%; "
            "variable expressivity (focal to segmental); "
            "Ashkenazi Jewish founder enrichment (p.Pro41 region variants); "
            "THAP domain mutations (N-terminus): affects DNA binding; "
            "C-terminal mutations: variable; "
            "de novo: 5-10%; familial majority; "
            "GENETIC TESTING: THAP1 Sanger or NGS panel"
        ),
        "age_of_onset": "Young adult 12-50 yr; most 20-40 yr",
        "pathognomonic": (
            "LARYNGEAL DYSTONIA — hoarse, strained, strangled voice quality — PATHOGNOMONIC for THAP1; "
            "Voice involvement distinguishes THAP1 from DYT1 (rare cranial) and DYT-KMT2B; "
            "Cranio-cervical dystonia: blepharospasm, torticollis, oromandibular; "
            "Upper limb: writer's cramp 10-20% (less than TOR1A); "
            "Generalisation to lower limbs RARE (unlike KMT2B which generalises); "
            "Penetrance 60%: carriers may be asymptomatic; "
            "Ashkenazi enrichment: test THAP1 first in Ashkenazi Jewish with laryngeal dystonia; "
            "Task-specific onset: voice worse with speaking (strained); "
            "Whispering preserved initially (spasmodic dysphonia pattern)"
        ),
        "treatment": (
            "FIRST-LINE: BOTULINUM TOXIN — injected into affected muscles; "
            "LARYNGEAL (spasmodic dysphonia): botox into thyroarytenoid muscle via EMG guidance; "
            "CERVICAL DYSTONIA: botox into sternocleidomastoid / splenius capitis; "
            "BLEPHAROSPASM: botox periorbital; "
            "Repeat injections 3-monthly; effectiveness maintained long-term; "
            "PHARMACOLOGICAL: trihexyphenidyl (moderate benefit); clonazepam (adjunct); "
            "DBS-GPi: reserved for severe, generalised, botox-refractory cases; "
            "DBS LESS EFFECTIVE than in DYT1 for focal THAP1 — botox preferred for focal; "
            "VOICE THERAPY: SLT for compensatory strategies; "
            "PSYCHOLOGY: significant quality of life impact from voice change; "
            "OCCUPATIONAL THERAPY: if upper limb involvement"
        ),
        "contraindications": (
            "AVOID DOPAMINE BLOCKERS: worsen dystonia; "
            "CAUTION HIGH-DOSE TRIHEXYPHENIDYL: limited benefit in THAP1 vs DYT1; "
            "BOTOX LARYNGEAL: requires EMG guidance — ENT/neurologist with laryngeal botox expertise; "
            "AVOID GENERAL ANAESTHESIA without airway assessment in severe laryngeal dystonia; "
            "DO NOT COUNSEL 100% PENETRANCE: 60% penetrance — many carriers unaffected"
        ),
        "monitoring": [
            "VOICE RECORDING: baseline + 3-monthly post-botox (objective voice analysis)",
            "BOTULINUM TOXIN EFFECTS: symptom diary; re-injection at 12 weeks if wearing off",
            "BFMDRS / TWSTRS (cervical): 6-monthly",
            "SWALLOWING: modified barium swallow if dysphagia (oromandibular involvement)",
            "THAP1 FAMILY CASCADE: 50% offspring risk; penetrance 60%",
            "PSYCHIATRIC: quality of life scales (SF-36, DystoniaQoL); depression screening",
            "AIRWAY: ENT review if severe laryngeal involvement",
        ],
        "lifecycle": [
            "Young adult (12-25 yr): voice changes onset; ENT misdiagnosis common",
            "Adult (25-40 yr): established diagnosis; botox programme; cervical involvement",
            "Mid-adult (40-55 yr): stable or slow progression; DBS if refractory",
            "Later adult (55+ yr): maintenance; hardware if DBS implanted",
            "Ashkenazi carrier: 50% offspring risk with 60% penetrance — genetic counselling",
            "Family cascade: testing first-degree relatives (siblings/offspring of confirmed carriers)",
        ],
        "concepts": [
            "DYT6/DYT-THAP1: young-adult onset cranio-cervical and laryngeal dystonia",
            "Laryngeal dystonia: strained/strangled voice — PATHOGNOMONIC distinguishing feature",
            "THAP domain: zinc finger DNA-binding domain — transcription factor",
            "Spasmodic dysphonia: adductor type (voice breaks) in THAP1",
            "Botox thyroarytenoid: gold standard for laryngeal dystonia — 3-monthly",
            "Ashkenazi enrichment: test THAP1 first in Ashkenazi with voice/neck dystonia",
            "Penetrance 60%: lower than TOR1A; family counselling critical",
            "DBS less effective focal: botox preferred for cervical/laryngeal; DBS for generalised",
            "No cognitive impairment: pure motor disorder (unlike KMT2B)",
            "Voice misdiagnosed: often diagnosed as functional voice disorder or laryngitis",
            "Cranio-cervical spectrum: blepharospasm, oromandibular, torticollis all possible",
            "Upper limb rare: writer's cramp <20% (TOR1A >60%)",
            "Generalisation uncommon: THAP1 tends to remain cranio-cervical",
            "EMG-guided botox: mandatory for accurate laryngeal muscle injection",
            "Age 12-50 yr: younger onset than typical adult-onset focal dystonia",
        ],
        "thresholds": [
            "Laryngeal botox: re-injection at 12 weeks or when symptoms return >70% baseline",
            "Cervical TWSTRS >30: botox referral; >50: DBS evaluation",
            "Voice handicap index (VHI) >40: laryngeal botox priority",
            "DBS referral: BFMDRS >25 with inadequate botox response",
            "Penetrance 60%: 3 in 5 carriers will develop symptoms — counselling figure",
            "Airway assessment: FVC <60%: anaesthetic alert card",
        ],
        "standards": [
            "Fuchs T 2009 Nat Genet (THAP1 DYT6 discovery)",
            "Blanchard A 2011 Arch Neurol (THAP1 genotype-phenotype)",
            "ESDA laryngeal dystonia botox guidelines",
            "ACMG-AMP-2015 variant classification",
            "MDS Task Force spasmodic dysphonia consensus",
            "Botulinum toxin certification — NICE guidance NG217",
        ],
        "etiologies": [
            {"type": "THAP1 THAP-domain Missense — Cranio-Cervical DYT6", "pct": 50},
            {"type": "THAP1 C-terminal Truncating — Generalised DYT6", "pct": 22},
            {"type": "THAP1 Ashkenazi Founder Variant — Laryngeal Predominant", "pct": 15},
            {"type": "THAP1 Deep Intronic / Splice — Atypical", "pct": 8},
            {"type": "THAP1 Phenocopy (other cranio-cervical dystonia genes)", "pct": 5},
        ],
        "seizure_types": [
            {"type": "Laryngeal Dystonia (spasmodic dysphonia)", "pct": 75},
            {"type": "Cervical Dystonia (torticollis)", "pct": 65},
            {"type": "Blepharospasm", "pct": 42},
            {"type": "Oromandibular Dystonia", "pct": 28},
            {"type": "Upper Limb Dystonia (writer's cramp)", "pct": 18},
        ],
        "triggers": [
            {"trigger": "Speaking / Voice Use", "pct": 92},
            {"trigger": "Stress / Anxiety", "pct": 78},
            {"trigger": "Fatigue", "pct": 68},
            {"trigger": "Missed Botox Window (>12 weeks)", "pct": 62},
            {"trigger": "Sleep Deprivation", "pct": 42},
            {"trigger": "Caffeine", "pct": 30},
            {"trigger": "Intercurrent URTI (voice strain)", "pct": 25},
            {"trigger": "Missed Anticholinergic Dose", "pct": 35},
        ],
        "references": [
            "Fuchs T 2009 Nat Genet (THAP1 mutations in DYT6)",
            "Blanchard A 2011 Arch Neurol (THAP1 phenotype spectrum)",
            "Xiromerisiou G 2012 PLoS One (THAP1 Ashkenazi)",
            "Cif L 2019 Lancet Neurol (DBS inherited dystonia)",
            "Coubes P 2004 Neurology (laryngeal botox dystonia)",
            "Bressman S 2009 Neurology (DYT6 families NY Ashkenazi)",
        ],
    },
    # -- ATP1A3 — AHC / CAPOS / RDP -----------------------------------------------
    {
        "gene": "ATP1A3",
        "alt_name": (
            "ATP1A3 (ATP1A3-1013aa-19q13.2 / AD-De-Novo — AHC-Alternating-Hemiplegia-Of-Childhood — "
            "FEVER-ABSOLUTE-TRIGGER-AHC-ATTACKS-FEVER-PROTOCOL-MANDATORY — "
            "FLUNARIZINE-FIRST-LINE-AHC-Episodes — "
            "CAPOS-RDP-Rapid-Onset-Dystonia-Parkinsonism-Same-Gene)"
        ),
        "protein": (
            "ATP1A3 -- 19q13.2 AD (de novo) -- ATP1A3-1013aa -- "
            "Na-K-ATPase-Alpha-3-Subunit-Neuronal-Specific-Ion-Pump-Electrochemical-Gradient -- "
            "AHC-Alternating-Hemiplegia-Childhood-OMIM-614820 -- "
            "CAPOS-Cerebellar-Ataxia-Areflexia-Pes-Cavus-Optic-Atrophy-OMIM-601338 -- "
            "RDP-Rapid-Onset-Dystonia-Parkinsonism-OMIM-128235 -- "
            "p.D801N-Most-Common-AHC-Variant-WORLDWIDE -- "
            "p.E815K-More-Severe-AHC-Phenotype -- "
            "p.G947R-E945K-CAPOS -- "
            "p.D923N-I363N-RDP -- "
            "FEVER-ABSOLUTE-TRIGGER-LIFE-THREATENING-FEVER-PROTOCOL-MANDATORY -- "
            "Episodic-Hemiplegia-Both-Sides-Alternating-PATHOGNOMONIC -- "
            "19q13.2"
        ),
        "locus": "19q13.2",
        "protein_size": "1013 aa",
        "inheritance": (
            "AD (autosomal dominant); >95% de novo; "
            "three distinct allelic syndromes: "
            "(1) AHC: D801N (most common), E815K (severe); "
            "(2) CAPOS: G947R, E945K; "
            "(3) RDP: D923N, I363N (young adult, acute onset); "
            "genotype-phenotype correlation strong — variant determines syndrome; "
            "no familial recurrence typical (de novo); "
            "germline mosaicism in 1-2% (recurrence risk)"
        ),
        "age_of_onset": "AHC: neonatal/infancy (<18 months); CAPOS: childhood; RDP: young adult (sudden)",
        "pathognomonic": (
            "AHC — ALTERNATING HEMIPLEGIA: episodes affecting right then left side alternately PATHOGNOMONIC; "
            "FEVER = ABSOLUTE TRIGGER FOR AHC ATTACKS: even minor temperature elevation → prolonged hemiplegia; "
            "FEVER PROTOCOL MANDATORY: paracetamol, cooling, reduce fever IMMEDIATELY; "
            "AHC attacks also triggered by: water immersion, emotional upset, specific foods; "
            "Episodes resolve with SLEEP (pathognomonic — hemiplegia disappears when child wakes); "
            "CAPOS: acute cerebellar ataxia + areflexia + pes cavus + optic atrophy + SNHL — episodic; "
            "RDP: RAPID-ONSET (hours) dystonia + parkinsonism in young adult after physical/emotional stress; "
            "RDP: caudal-rostral distribution (legs > arms > face — PATHOGNOMONIC RDP direction); "
            "Nystagmus: present in AHC and CAPOS; "
            "D801N variant: typical AHC severity; E815K: severe AHC + worse epilepsy"
        ),
        "treatment": (
            "AHC — ACUTE ATTACKS: "
            "FLUNARIZINE (calcium channel blocker): first-line prevention; 2.5-10 mg/day; "
            "reduces attack frequency 40-60%; not curative; "
            "BENZODIAZEPINES (diazepam): for prolonged attacks — abort episode; "
            "FEVER: IMMEDIATE FEVER CONTROL (paracetamol + cooling) — MOST IMPORTANT MANAGEMENT; "
            "AVOID ALL TRIGGERS: water immersion, emotional stress, specific foods; "
            "INTER-ATTACK: physical + cognitive development support; "
            "STATUS HEMIPLEGICUS: IV diazepam; hospital protocol required; "
            "CAPOS: same trigger avoidance; vestibular therapy for balance; "
            "CAPOS SNHL: hearing aids early; cochlear implant if profound; "
            "RDP: no effective pharmacological treatment; DBS-GPi attempted with limited success; "
            "PHYSIOTHERAPY: all three syndromes; "
            "KETOGENIC DIET: some AHC benefit reported (Level C)"
        ),
        "contraindications": (
            "FEVER IN AHC: NEVER IGNORE — FEVER IS LIFE-THREATENING TRIGGER; "
            "AVOID FEVER EXPOSURE: no live attenuated vaccines during illness; "
            "WATER IMMERSION RESTRICTIONS: bathing supervised; no swimming alone; "
            "AVOID EMOTIONAL EXTREMES: excitement, fear — attack triggers; "
            "ACETAZOLAMIDE: reported to worsen some AHC — NOT routinely used; "
            "CAUTION SODIUM CHANNEL BLOCKERS: may worsen Na+/K+-ATPase dysfunction; "
            "AVOID GENERAL ANAESTHESIA WITHOUT NEUROLOGICAL ALERT: Na+/K+-ATPase dysfunction; "
            "DO NOT DISMISS HEMIPLEGIA AS TODD'S PARESIS without ATP1A3 testing in infant"
        ),
        "monitoring": [
            "AHC ATTACK DIARY: daily; record duration, side, trigger, temperature",
            "FEVER TEMPERATURE: threshold 37.5°C → activate fever protocol",
            "FLUNARIZINE DOSE: 3-monthly review during titration (weight-based)",
            "COGNITIVE DEVELOPMENT: Bayley / WISC 6-monthly — cognitive regression risk",
            "OPHTHALMOLOGY: optic atrophy (CAPOS); annual OCT",
            "AUDIOLOGY: SNHL in CAPOS — annual audiogram",
            "CARDIAC: arrhythmia reported in ATP1A3 — 12-lead ECG annually",
            "EEG: 6-monthly (AHC + epilepsy comorbidity 50%)",
            "ATP1A3 EMERGENCY LETTER: patient carries letter + protocol card at all times",
        ],
        "lifecycle": [
            "Neonatal (AHC): nystagmus, episodic floppiness, first hemiplegic episodes",
            "Infancy (AHC): fever protocol established; flunarizine initiation",
            "Childhood (AHC): cognitive/motor development; school support; attack diary",
            "Adolescence (AHC): independence; self-management of fever protocol",
            "Adult (AHC/RDP/CAPOS): transition; driving restrictions; employment",
            "Family planning: 95% de novo → low recurrence; rare germline mosaic counselling",
        ],
        "concepts": [
            "ATP1A3: Na+/K+-ATPase alpha-3 — neuronal electrogenic pump",
            "AHC: alternating hemiplegia of childhood — episodes both sides alternating",
            "Sleep resolution: hemiplegia resolves with sleep — PATHOGNOMONIC",
            "Fever absolute trigger: even mild temperature elevation → attack",
            "Fever protocol: immediate paracetamol + cooling — most important intervention",
            "D801N: most common AHC variant; E815K: more severe",
            "CAPOS: completely different phenotype (cerebellar + optic + SNHL) — same gene",
            "RDP: young adult acute dystonia-parkinsonism — hours onset",
            "RDP caudal-rostral: legs > arms > face — direction of spread pathognomonic",
            "Flunarizine: Ca2+ channel blocker — reduces AHC frequency",
            "Water immersion trigger: bathing supervised at all times in AHC",
            "Sleep resolves attack: clinically useful — induce nap for acute attack",
            "Allelic heterogeneity: 3 distinct syndromes from different ATP1A3 variants",
            "De novo 95%: most cases new mutation — negative family history expected",
            "Emergency letter: always carry fever protocol and diagnosis card",
        ],
        "thresholds": [
            "Temperature 37.5°C in AHC: activate fever protocol immediately",
            "Attack duration >2 hours: consider IV diazepam; hospital attendance",
            "Attack frequency >4/month on flunarizine: dose increase or KD trial",
            "CAPOS audiogram: pure tone average >40 dB: hearing aid fitting",
            "AHC ECG: QTc >450 ms: cardiology referral",
            "Cognitive regression: 2-point drop on standardised score: increase support",
        ],
        "standards": [
            "Heinzen EL 2012 Nat Genet (ATP1A3 AHC discovery)",
            "Rosewich H 2012 Nat Genet (ATP1A3 CAPOS)",
            "AHC of Childhood International Working Group consensus 2015",
            "ACMG-AMP-2015 variant classification",
            "Flunarizine dosing — European Paediatric Neurology guidelines",
            "CAPOS Management — Kagawa metabolic neurology consensus",
        ],
        "etiologies": [
            {"type": "ATP1A3 p.D801N — Classic AHC (alternating hemiplegia)", "pct": 45},
            {"type": "ATP1A3 p.E815K — Severe AHC + Epilepsy", "pct": 18},
            {"type": "ATP1A3 CAPOS variants (G947R/E945K) — CAPOS syndrome", "pct": 15},
            {"type": "ATP1A3 RDP variants (D923N/I363N) — Rapid-Onset Dystonia-Parkinsonism", "pct": 12},
            {"type": "ATP1A3 Other Missense — Atypical AHC/Overlap", "pct": 10},
        ],
        "seizure_types": [
            {"type": "Alternating Hemiplegia Episodes (AHC)", "pct": 88},
            {"type": "Epileptic Seizures (comorbid in AHC)", "pct": 52},
            {"type": "Generalised Dystonia (inter-attack baseline in severe)", "pct": 35},
            {"type": "Cerebellar Ataxia Episodes (CAPOS)", "pct": 22},
            {"type": "Acute Dystonia-Parkinsonism Onset (RDP)", "pct": 15},
        ],
        "triggers": [
            {"trigger": "FEVER (ANY TEMPERATURE ELEVATION)", "pct": 98},
            {"trigger": "Water Immersion (bathing / swimming)", "pct": 82},
            {"trigger": "Emotional Stress / Excitement", "pct": 75},
            {"trigger": "Fatigue / Sleep Deprivation", "pct": 65},
            {"trigger": "Specific Foods (chocolate, citrus — patient-specific)", "pct": 40},
            {"trigger": "Physical Exertion (RDP)", "pct": 35},
            {"trigger": "Bright Light / Flicker", "pct": 28},
            {"trigger": "Missed Flunarizine Dose", "pct": 45},
        ],
        "references": [
            "Heinzen EL 2012 Nat Genet (ATP1A3 AHC discovery)",
            "Rosewich H 2012 Nat Genet (ATP1A3 CAPOS)",
            "Mikati MA 2013 Neurology (AHC clinical management)",
            "Brashear A 1997 Neurology (RDP original description)",
            "Sweney MT 2015 Pediatr Neurol (AHC fever protocol)",
            "Dard R 2015 Dev Med Child Neurol (AHC flunarizine outcomes)",
        ],
    },
    # -- ANO3 — DYT24 -----------------------------------------------------------
    {
        "gene": "ANO3",
        "alt_name": (
            "ANO3 (ANO3-913aa-11p14.3 / AD — DYT-ANO3-DYT24 — "
            "CRANIOCERVICAL-TREMULOUS-DYSTONIA-PATHOGNOMONIC-Adult-Onset — "
            "BOTULINUM-TOXIN-FIRST-LINE — "
            "DBS-GPi-FOR-SEVERE)"
        ),
        "protein": (
            "ANO3 -- 11p14.3 AD -- ANO3-913aa -- "
            "Anoctamin-3-TMEM16-Family-Calcium-Activated-Chloride-Channel-Scramblase -- "
            "DYT-ANO3-Dystonia-24-OMIM-615034 -- "
            "Adult-Onset-Craniocervical-Dystonia-20-50yr -- "
            "TREMULOUS-DYSTONIA-Tremor-Prominent-Component-PATHOGNOMONIC-DISTINGUISHES-From-THAP1 -- "
            "Cervical-Dystonia-Head-Tremor-Blepharospasm-Laryngeal-Dysphonia-Variable -- "
            "Botulinum-Toxin-First-Line-Effective-Focal-Involvement -- "
            "DBS-GPi-Considered-Botox-Refractory-Generalised -- "
            "Incomplete-Penetrance-Variable-Expressivity -- "
            "TMEM16-Family-8-TM-Domains-Homodimer-Ca2+-Activated -- "
            "11p14.3"
        ),
        "locus": "11p14.3",
        "protein_size": "913 aa",
        "inheritance": (
            "AD (autosomal dominant); incomplete penetrance; "
            "variable expressivity within families; "
            "missense dominant (gain-of-function or dominant negative proposed); "
            "de novo: ~15-20%; familial majority; "
            "multiple families reported European and Asian populations; "
            "GENETIC TESTING: ANO3 targeted sequencing or NGS movement disorder panel"
        ),
        "age_of_onset": "Adult onset 20-50 yr (mean ~40 yr); rare childhood forms reported",
        "pathognomonic": (
            "CRANIOCERVICAL DYSTONIA WITH TREMULOUS COMPONENT — tremor distinguishes from THAP1; "
            "CERVICAL DYSTONIA predominant: head turned/tilted + tremulous head shaking; "
            "Tremor component in ANO3 more prominent than in other dystonia genes; "
            "BLEPHAROSPASM: involuntary eye closure — common in ANO3 craniocervical; "
            "Laryngeal involvement: dysphonia/spasmodic dysphonia (less prominent than THAP1); "
            "Upper limb dystonia: writer's cramp, focal arm dystonia — variable; "
            "Generalisation uncommon (craniocervical restricted in most); "
            "ANOCTAMIN-3 CLUE: calcium-activated chloride channel in neurons — basal ganglia; "
            "MRI normal; DAT-SPECT normal (not parkinsonism)"
        ),
        "treatment": (
            "FIRST-LINE: BOTULINUM TOXIN — effective for focal craniocervical involvement; "
            "CERVICAL: botox into sternocleidomastoid, splenius capitis, semispinalis (3-monthly); "
            "BLEPHAROSPASM: periorbital botox (orbicularis oculi); "
            "TREMULOUS component: may be partially responsive to botox or propranolol; "
            "PROPRANOLOL / PRIMIDONE: for tremulous dystonia component (trial Level C); "
            "TRIHEXYPHENIDYL: moderate benefit; "
            "CLONAZEPAM: adjunct for task-specific relief; "
            "DBS-GPi or DBS-Vim (tremulous): for severe, refractory cases; "
            "DBS-GPi preferred if dystonia predominant; DBS-Vim if tremor predominant; "
            "PHYSIOTHERAPY: head position support, neck exercises; "
            "SENSORY TRICK: patients often find specific touch relieves dystonia temporarily"
        ),
        "contraindications": (
            "AVOID DOPAMINE BLOCKERS: worsen dystonia (same as all hereditary dystonias); "
            "AVOID LABELLING AS ESSENTIAL TREMOR without ANO3 testing in familial tremulous dystonia; "
            "CAUTION PROPRANOLOL IN ASTHMA: tremorous component treatment; "
            "BOTOX OVERDOSE: dysphagia risk with excessive cervical injection; "
            "DO NOT ASSUME FUNCTIONAL: ANO3 dystonia can look unusual (tremulous makes it complex)"
        ),
        "monitoring": [
            "BFMDRS / TWSTRS-2: 6-monthly",
            "BOTULINUM TOXIN: symptom diary; re-injection at 12 weeks",
            "TREMOR (Fahn-Tolosa-Marin): 6-monthly — track tremor vs dystonia ratio",
            "VOICE (VHI): if laryngeal involved",
            "DBS PROGRAMMING: 3-monthly first year",
            "ANO3 FAMILY CASCADE: 50% offspring risk; penetrance incomplete",
            "PSYCHIATRIC: depression / anxiety — QoL scales 6-monthly",
            "DYSPHAGIA ASSESSMENT: SLT if oromandibular or laryngeal involvement",
        ],
        "lifecycle": [
            "Young adult (20-35 yr): cervical dystonia onset; misdiagnosed as cervical spondylosis",
            "Mid-adult (35-50 yr): established; botox programme; tremulous component prominent",
            "Adult (50-65 yr): DBS if refractory; maintenance",
            "Older adult (65+ yr): hardware considerations; DBS battery planning",
            "Family planning: 50% risk; penetrance incomplete counselling",
            "Family cascade: test symptomatic relatives; genetic counselling for carriers",
        ],
        "concepts": [
            "DYT-ANO3 / DYT24: adult-onset craniocervical tremulous dystonia",
            "Anoctamin-3 (ANO3): TMEM16 family Ca2+-activated chloride channel",
            "Tremulous dystonia: tremor component distinguishes from THAP1 (less tremulous)",
            "Craniocervical restriction: most cases do not generalise to limbs",
            "Botox first-line: focal craniocervical responds well",
            "DBS dual option: GPi (dystonia dominant) or Vim (tremor dominant)",
            "Blepharospasm common: part of craniocervical spectrum",
            "Propranolol adjunct: for tremulous component",
            "Sensory trick: transient relief with touch — exploitable for therapy",
            "Incomplete penetrance: family members with same variant may be unaffected",
            "DAT-SPECT normal: helps exclude early Parkinson's with dystonia",
            "Misdiagnosis risk: tremulous + neck pain → spondylosis or ET misdiagnosis",
            "Variable expressivity: same variant → blepharospasm vs cervical vs mixed",
            "Ca2+-activated Cl- channel: ANO3 dysfunction alters basal ganglia neuronal firing",
            "Movement disorder panel: ANO3 should be included in familial focal dystonia panels",
        ],
        "thresholds": [
            "Cervical TWSTRS >30: botox referral; >50: DBS evaluation",
            "Botox: re-injection if symptoms return >70% baseline at 12 weeks",
            "Tremor FTM >20 on tremor subscale: propranolol/primidone trial",
            "BFMDRS >25 botox-refractory: DBS referral",
            "Penetrance: 50-60% estimated — counsel accordingly",
            "DBS Vim vs GPi: if tremor FTM score > dystonia BFMDRS, favour Vim target",
        ],
        "standards": [
            "Charlesworth G 2012 Am J Hum Genet (ANO3 DYT24 discovery)",
            "Huang XJ 2018 Mov Disord (ANO3 genotype-phenotype expansion)",
            "ESDA craniocervical dystonia guidelines",
            "ACMG-AMP-2015 variant classification",
            "MDS Task Force dystonia classification 2013",
            "Botulinum toxin dystonia — ABTA standards",
        ],
        "etiologies": [
            {"type": "ANO3 Missense het — Craniocervical Tremulous Dystonia Classic", "pct": 55},
            {"type": "ANO3 Missense het — Blepharospasm Predominant", "pct": 22},
            {"type": "ANO3 Missense het — Cervical Dystonia + Writer's Cramp", "pct": 15},
            {"type": "ANO3 de novo — Early Onset Variant", "pct": 5},
            {"type": "ANO3 Phenocopy (other craniocervical dystonia genes)", "pct": 3},
        ],
        "seizure_types": [
            {"type": "Cervical Dystonia + Head Tremor", "pct": 88},
            {"type": "Blepharospasm", "pct": 55},
            {"type": "Laryngeal Dystonia (spasmodic dysphonia)", "pct": 35},
            {"type": "Upper Limb Dystonia (writer's cramp)", "pct": 25},
            {"type": "Oromandibular Dystonia", "pct": 20},
        ],
        "triggers": [
            {"trigger": "Head Movement / Specific Posture", "pct": 88},
            {"trigger": "Stress / Anxiety", "pct": 78},
            {"trigger": "Speaking / Voice Use", "pct": 58},
            {"trigger": "Fatigue", "pct": 68},
            {"trigger": "Missed Botox Window (>12 weeks)", "pct": 60},
            {"trigger": "Cold Weather", "pct": 35},
            {"trigger": "Sleep Deprivation", "pct": 45},
            {"trigger": "Caffeine", "pct": 28},
        ],
        "references": [
            "Charlesworth G 2012 Am J Hum Genet (ANO3 DYT24 discovery)",
            "Huang XJ 2018 Mov Disord (ANO3 clinical expansion)",
            "Norgren N 2011 Neurogenetics (ANO3 early report)",
            "Albanese A 2013 Mov Disord (dystonia classification)",
            "Jinnah HA 2017 Mov Disord (dystonia mechanisms)",
            "Cif L 2019 Lancet Neurol (DBS inherited dystonia)",
        ],
    },
]


def _make_cohort(gene_data, seed):
    rng = random.Random(seed)
    ages = [rng.randint(2, 70) for _ in range(40)]
    # female predominance in GCH1 (4:1), balanced otherwise
    female_bias = {"GCH1": 0.80}.get(gene_data["gene"], 0.50)
    sexes = ["F" if rng.random() < female_bias else "M" for _ in range(40)]
    # survival — all adult-onset generally have normal lifespan except ATP1A3 (AHC attacks)
    alive_prob = 0.55 if gene_data["gene"] == "ATP1A3" else 0.88
    alive = [rng.random() < alive_prob for _ in range(40)]
    etiol_types = [e["type"] for e in gene_data["etiologies"]]
    etiol_wts = [e["pct"] for e in gene_data["etiologies"]]
    etiols = rng.choices(etiol_types, weights=etiol_wts, k=40)
    sz_types = [s["type"] for s in gene_data["seizure_types"]]
    sz_wts = [s["pct"] for s in gene_data["seizure_types"]]
    szs = rng.choices(sz_types, weights=sz_wts, k=40)
    trig_types = [t["trigger"] for t in gene_data["triggers"]]
    trig_wts = [t["pct"] for t in gene_data["triggers"]]
    trigs = rng.choices(trig_types, weights=trig_wts, k=40)
    patients = []
    for i in range(40):
        patients.append({
            "id": f"{gene_data['gene']}-{seed}-{i+1:02d}",
            "gene": gene_data["gene"],
            "age": ages[i],
            "sex": sexes[i],
            "alive": alive[i],
            "etiology": etiols[i],
            "dystonia_type": szs[i],
            "trigger": trigs[i],
        })
    return patients


def _build_all():
    all_patients = []
    for idx, g in enumerate(DYSTONIA_GENES):
        seed = SEED_BASE + idx
        all_patients.extend(_make_cohort(g, seed))
    return all_patients


def overview():
    pts = _build_all()
    total = len(pts)
    alive_pct = round(100 * sum(1 for p in pts if p["alive"]) / total, 1)

    gene_summaries = {}
    for idx, g in enumerate(DYSTONIA_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(g, seed)
        gene_summaries[g["gene"]] = {
            "gene": g["gene"],
            "alt_name": g["alt_name"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"].split(";")[0].strip(),
            "n_patients": len(cohort),
            "alive_pct": round(100 * sum(1 for p in cohort if p["alive"]) / len(cohort), 1),
            "top_etiology": max(g["etiologies"], key=lambda e: e["pct"])["type"],
            "top_dystonia_type": max(g["seizure_types"], key=lambda s: s["pct"])["type"],
            "top_trigger": max(g["triggers"], key=lambda t: t["pct"])["trigger"],
            "pathognomonic_summary": g["pathognomonic"][:160] + "…",
        }

    from collections import Counter
    all_etiol = Counter(p["etiology"] for p in pts)
    all_dyst = Counter(p["dystonia_type"] for p in pts)

    return {
        "title": "Hereditary-Dystonia-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Dystonia Atlas — "
            "TOR1A (DYT1) · SGCE (DYT11/M-D) · GCH1 (DRD/DYT5a) · TH (DRD/DYT5b) · "
            "KMT2B (DYT28) · THAP1 (DYT6) · ATP1A3 (AHC/CAPOS/RDP) · ANO3 (DYT24)"
        ),
        "n_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE+7}",
        "alive_pct": alive_pct,
        "gene_summaries": gene_summaries,
        "etiology_distribution": dict(all_etiol.most_common(10)),
        "dystonia_type_distribution": dict(all_dyst.most_common(8)),
        "key_flags": [
            "GCH1-L-DOPA-CURATIVE-MUST-TRY-BEFORE-BOTOX-IN-CHILDHOOD-DYSTONIA",
            "GCH1-DIURNAL-VARIATION-PATHOGNOMONIC-BETTER-MORNING",
            "TOR1A-GAG-DELETION-MOST-COMMON-GENETIC-GENERALISED-DYSTONIA",
            "TOR1A-DBS-GPi->70pct-IMPROVEMENT",
            "SGCE-MYOCLONUS-PREDOMINATES-PATERNAL-IMPRINTING",
            "SGCE-ALCOHOL-RESPONSIVE-DIAGNOSTIC-NOT-TREATMENT",
            "SGCE-ALCOHOL-DEPENDENCE-RISK-30pct",
            "TH-AR-DRD-BIALLELIC-CSF-HVA-5HIAA-REDUCED-PATHOGNOMONIC",
            "KMT2B-DBS-GPi-HIGHLY-EFFECTIVE-EVEN-WITH-INTELLECTUAL-DISABILITY",
            "THAP1-LARYNGEAL-DYSTONIA-PATHOGNOMONIC",
            "ATP1A3-FEVER-ABSOLUTE-TRIGGER-FEVER-PROTOCOL-MANDATORY",
            "ATP1A3-SLEEP-RESOLVES-AHC-ATTACK-PATHOGNOMONIC",
            "ANO3-CRANIOCERVICAL-TREMULOUS-DYSTONIA-PATHOGNOMONIC",
            "L-DOPA-FIRST-IN-CHILDHOOD-DYSTONIA-ALWAYS",
        ],
    }


def breakdown():
    result = {}
    for idx, g in enumerate(DYSTONIA_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(g, seed)
        result[g["gene"]] = {
            "gene": g["gene"],
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "age_of_onset": g["age_of_onset"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "contraindications": g["contraindications"],
            "monitoring": g["monitoring"],
            "lifecycle": g["lifecycle"],
            "concepts": g["concepts"],
            "thresholds": g["thresholds"],
            "standards": g["standards"],
            "etiologies": g["etiologies"],
            "dystonia_types": g["seizure_types"],
            "triggers": g["triggers"],
            "references": g["references"],
            "n_patients": len(cohort),
            "alive_pct": round(100 * sum(1 for p in cohort if p["alive"]) / len(cohort), 1),
            "cohort": cohort,
        }
    return result


def definitions():
    return {
        "dystonia_classification": {
            "DYT1": "DYT-TOR1A — generalised, childhood onset, GAG deletion, DBS-GPi highly effective",
            "DYT5a": "DRD-GCH1 — dopa-responsive, AD, diurnal variation, L-DOPA curative",
            "DYT5b": "DRD-TH — AR, infantile, CSF HVA/5-HIAA reduced, L-DOPA responsive",
            "DYT6": "DYT-THAP1 — young adult, cranio-cervical + laryngeal, Ashkenazi enrichment",
            "DYT11": "DYT-SGCE — myoclonus-dystonia, paternal imprint, alcohol responsive",
            "DYT24": "DYT-ANO3 — adult craniocervical tremulous dystonia, botox first-line",
            "DYT28": "DYT-KMT2B — childhood generalised + ID, de novo, DBS highly effective",
            "AHC": "Alternating Hemiplegia of Childhood — ATP1A3, fever trigger absolute",
            "CAPOS": "Cerebellar Ataxia Areflexia Pes Cavus Optic Atrophy — ATP1A3 allelic to AHC",
            "RDP": "Rapid-Onset Dystonia-Parkinsonism — ATP1A3, young adult, acute hours onset",
        },
        "key_pharmacology": {
            "L-DOPA_DRD": "L-DOPA 3-5 mg/kg/day CURATIVE in GCH1-DRD — must try before any other intervention in childhood dystonia",
            "THP_DYT1": "Trihexyphenidyl HIGH DOSE (up to 30 mg/day) — children tolerate better than adults",
            "Botox_focal": "Botulinum toxin — first-line for focal dystonia (cervical, laryngeal, blepharospasm)",
            "Clonazepam_SGCE": "Clonazepam GABA-A — same mechanism as alcohol in SGCE; prevents ethanol dependence cycle",
            "Flunarizine_AHC": "Flunarizine (Ca2+ channel blocker) — first-line prevention AHC attacks (2.5-10 mg/day)",
            "DBS_GPi": "DBS-GPi — highly effective in DYT1 (>70%), KMT2B (>80%), variable in others",
        },
        "critical_contraindications": {
            "Botox_before_LDOPA": "NEVER inject botox in childhood lower limb dystonia without L-DOPA trial — GCH1 must be excluded",
            "Orthopaedic_DRD": "DO NOT perform foot surgery for equinovarus in child without GCH1 L-DOPA trial",
            "Dopamine_blockers": "AVOID haloperidol/metoclopramide in all hereditary dystonias — precipitate dystonic crisis",
            "Alcohol_SGCE": "ALCOHOL NOT A TREATMENT in SGCE — dependence risk very high (30%)",
            "Fever_AHC": "FEVER ABSOLUTE TRIGGER in ATP1A3-AHC — immediate paracetamol + cooling MANDATORY",
            "DBS_KMT2B_ID": "DO NOT EXCLUDE DBS because of intellectual disability in KMT2B — equally effective",
            "Valproate_female": "VALPROATE teratogenicity — contraceptive counselling in females of reproductive age",
        },
        "pathognomonic_signs": {
            "TOR1A_lower_limb": "Lower limb onset in childhood generalised dystonia — DYT1 pathognomonic presentation",
            "GCH1_diurnal": "Diurnal variation (better morning, worse evening) — DRD/GCH1 PATHOGNOMONIC",
            "SGCE_myoclonus": "Lightning-fast myoclonus predominating over dystonia — SGCE hallmark",
            "THAP1_laryngeal": "Strained/strangled voice in young adult dystonia — THAP1 pathognomonic",
            "KMT2B_focal_to_general": "Childhood focal lower limb → generalised + ID — KMT2B pathognomonic trajectory",
            "AHC_sleep_resolution": "Hemiplegia resolving completely with sleep — AHC/ATP1A3 pathognomonic",
            "AHC_alternating": "Episodes affecting alternating sides — AHC specific",
            "ANO3_tremulous": "Craniocervical dystonia with prominent tremulous component — ANO3 distinguishing feature",
        },
        "ddx_table": {
            "DYT1_vs_DRD": "DYT1: no diurnal variation, no L-DOPA response; DRD/GCH1: diurnal variation, L-DOPA curative",
            "GCH1_vs_TH": "GCH1 AD: pterin profile reduced; TH AR: CSF HVA/5-HIAA reduced, pterin normal",
            "SGCE_vs_cortical_myoclonus": "SGCE: no EEG cortical correlate; cortical myoclonus: EEG spike before jerk",
            "THAP1_vs_ANO3": "THAP1: laryngeal predominant, less tremor; ANO3: tremulous craniocervical",
            "AHC_vs_Todd": "AHC: alternating, sleep resolves, recurring; Todd's paresis: unilateral, post-seizure, single",
            "KMT2B_vs_DYT1": "KMT2B: ID 40%, generalises always, childhood; DYT1: no ID, same GAG deletion, variable",
        },
        "glossary": {
            "Dystonia": "Sustained or intermittent muscle contractions causing abnormal postures or repetitive movements",
            "DBS": "Deep Brain Stimulation — implanted neurostimulator delivering electrical pulses to basal ganglia targets",
            "GPi": "Globus Pallidus internus — primary DBS target for dystonia",
            "Vim": "Ventral intermediate nucleus of thalamus — DBS target for tremor-predominant conditions",
            "BFMDRS": "Burke-Fahn-Marsden Dystonia Rating Scale — validated clinical outcome measure",
            "TWSTRS": "Toronto Western Spasmodic Torticollis Rating Scale — cervical dystonia severity",
            "Geste_antagoniste": "Sensory trick — touching an area near dystonic muscle temporarily relieves dystonia",
            "Diurnal_variation": "Symptom variation through day — worse evening; key clue for DRD/GCH1",
            "Myoclonus": "Brief, shock-like involuntary muscle jerks (<100 ms) — different from dystonia",
            "Botulinum_toxin": "Neuromuscular blocking agent from Clostridium botulinum — focal dystonia treatment",
            "BH4": "Tetrahydrobiopterin — cofactor for aromatic amino acid hydroxylases including TH",
            "Penetrance": "Proportion of genotype carriers who develop clinical phenotype",
            "Imprinting": "Epigenetic silencing of one parental allele — SGCE is paternally expressed",
            "Status_dystonicus": "Life-threatening dystonic storm requiring intensive care — IV diazepam/sedation",
            "Flunarizine": "Calcium channel blocker — first-line for AHC prevention in ATP1A3",
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
