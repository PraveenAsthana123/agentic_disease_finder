"""
Rett Syndrome (RTT) Epilepsy Dashboard
========================================
41-patient cohort · MECP2 loss-of-function · X-linked dominant
Rett Syndrome: MECP2 pathogenic variant (Xq28) → severe epileptic encephalopathy
with regression of purposeful hand use, acquired microcephaly, gait apraxia,
breathing irregularities, autonomic dysregulation, and drug-resistant epilepsy.
KEY EEG: Progressive loss of posterior dominant rhythm → multifocal spike-wave,
high-amplitude irregular slow-wave with rhythmic central/frontal beta pattern,
monorhythmic theta (4-6 Hz) over central regions. Characteristic pseudo-periodic
delta pattern in deep sleep. QTc prolongation risk — cardiac monitoring mandatory.
AED NOTE: VPA commonly used (watch carnitine/LFT). LTG and LEV preferred in many.
PHT: worsens EEG pattern and may increase seizures. CBZ: modest efficacy, QT risk.
Trofinetide (Daybue®, FDA March 2023): first approved pharmacotherapy for RTT
(functional/behavioural improvement — not AED-approved but may reduce seizure burden).
"""

import random
from datetime import datetime

SEED = 7777
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ─────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "Classical RTT — MECP2 pathogenic variant (missense / nonsense / frameshift)",
        "n": 29, "pct": 71,
        "category": "Classical-RTT-MECP2",
        "mechanism": (
            "The most common cause of Rett Syndrome (≈71% of N=41): a de novo pathogenic variant "
            "in MECP2 (methyl-CpG-binding protein 2, Xq28), an X-linked gene that encodes a "
            "multifunctional transcriptional regulator critical for synaptic maturation, neuronal "
            "chromatin organisation, and gene silencing. MECP2 protein binds methylated CpG "
            "dinucleotides via its methyl-CpG-binding domain (MBD) and recruits the NCoR/SMRT "
            "corepressor complex via its transcription repression domain (TRD). In neurons MECP2 "
            "acts as a global transcriptional modulator — not a simple repressor — and is required "
            "for maintaining appropriate activity-dependent gene expression. Loss of MECP2 function "
            "leads to widespread dysregulation of synaptic gene networks, including overexpression "
            "of BDNF (brain-derived neurotrophic factor), altered GABAergic and glutamatergic "
            "signalling, reduced dendritic arborisation, and impaired synaptic plasticity. "
            "The eight most common MECP2 variants account for >70% of classical RTT cases: "
            "R106W, R133C, T158M (most common, ~12%), R168X (nonsense, ~9%), R255X (~4%), "
            "R270X (~8%), R294X (~6%), and the 3' truncations del267 and del288. "
            "Genotype–phenotype correlations: R133C and R294X are milder (preserved ambulation); "
            "R168X and R255X are severe (early seizure onset, complete regression). "
            "Detection: sequencing of MECP2 coding region detects >99% of classical RTT; "
            "MLPA for large deletions (2-3% of cases not detected by sequencing alone). "
            "Inheritance: de novo in ≥99% of classical females; maternal germline mosaicism "
            "accounts for familial recurrence (~1%). Males with MECP2 pathogenic variants "
            "present with severe neonatal encephalopathy (not classical RTT) unless XXY "
            "or somatic mosaicism — Klinefelter RTT males do develop RTT-like phenotype."
        ),
        "eeg_correlate": (
            "Classical RTT EEG evolution follows a characteristic 4-stage pattern: "
            "Stage I (pre-regression, 6-18 months): normal or near-normal background with "
            "immature features; focal or multi-focal spike discharges begin in some. "
            "Stage II (rapid regression, 1-4 years): abrupt loss of posterior dominant "
            "rhythm (PDR); background disorganised with generalised irregular high-amplitude "
            "delta 1-3 Hz; monorhythmic central theta 4-6 Hz ('Rett theta') in waking — "
            "a key characteristic finding; multifocal and generalised spike-wave with "
            "runs of 2-3 Hz; frequent runs of rhythmic frontal/central 10-25 Hz beta "
            "(beta-frequency spike bursts — another RTT hallmark, especially in sleep). "
            "Stage III (pseudo-stationary, 2-10 years): epileptiform activity prominent; "
            "pseudo-periodic high-amplitude delta pattern in NREM sleep; characteristic "
            "'theta burst pattern' in waking; generalised polyspike-wave paroxysms. "
            "Stage IV (late deterioration, >10 years): gradual reduction in epileptiform "
            "activity with overall background deterioration; monorhythmic delta persists. "
            "Seizure EEG: focal onset with secondary generalisation, multifocal independent "
            "spike-wave, and generalised tonic-clonic discharges. Ictal bradycardia: "
            "characteristic autonomic seizure manifestation — always co-register ECG."
        ),
        "mri_finding": (
            "Brain MRI in RTT: typically normal in the first 1-2 years of life, then "
            "shows progressive changes: (1) Diffuse cerebral and cerebellar volume loss "
            "(acquired atrophy on serial imaging), most pronounced in frontal lobes. "
            "(2) Decreased white matter volume (reduced axonal arborisation and dendritic "
            "simplification). (3) Caudate and putamen volume reduction (basal ganglia "
            "involvement correlating with motor features). (4) T2 signal changes in "
            "periventricular white matter (hypomyelination). MRS: reduced NAA/Cr ratio "
            "in frontal white matter and basal ganglia (neuronal dysfunction marker). "
            "No focal cortical dysplasia or heterotopia. MRI severity correlates "
            "imperfectly with MECP2 variant severity — R168X may show greater atrophy "
            "than R133C on serial imaging. MRI is supportive, not diagnostic."
        ),
        "clinical_note": (
            "Classical RTT diagnostic criteria (Neul 2010, revised 2014): "
            "REQUIRED: (1) Period of regression followed by recovery/stabilisation. "
            "MAIN criteria (all 4 required for 'typical' RTT): (A) Partial or complete loss "
            "of acquired purposeful hand skills. (B) Partial or complete loss of acquired "
            "spoken language. (C) Gait abnormalities: dyspraxia or absence of gait. "
            "(D) Stereotyped hand movements (hand-wringing, squeezing, clapping, mouthing, "
            "washing/rubbing automatisms). EXCLUSION criteria: brain injury from trauma, "
            "neurometabolic disease, severe infection causing neurological problems; grossly "
            "abnormal psychomotor development in first 6 months. "
            "Breathing irregularities: episodic hyperventilation, breath-holding, "
            "forced expulsion of air — characteristic and often mistaken for seizures. "
            "QTc prolongation (>470 ms in 20-30% of RTT patients): risk of torsades de "
            "pointes — ECG mandatory; avoid drugs that prolong QT (phenothiazines, "
            "some antipsychotics). Sudden unexplained death in epilepsy (SUDEP) risk "
            "elevated — autonomic dysregulation + cardiac QTc prolongation = compounded risk."
        ),
    },
    {
        "etiology": "CDKL5 Deficiency Disorder (CDD) — atypical RTT / Hanefeld variant",
        "n": 4, "pct": 10,
        "category": "CDKL5-Atypical-RTT",
        "mechanism": (
            "Pathogenic variants in CDKL5 (cyclin-dependent kinase-like 5, Xp22.13) cause "
            "CDKL5 Deficiency Disorder, historically classified as the 'early-onset seizure "
            "variant of Rett Syndrome' or 'Hanefeld variant'. CDKL5 is a serine-threonine "
            "kinase required for neuronal differentiation, dendritic spine morphogenesis, "
            "synaptogenesis, and activity-dependent synaptic plasticity. CDKL5 phosphorylates "
            "MECP2 at Ser80 and Thr308 — mechanistic overlap with classical RTT. Key: CDD "
            "is now recognised as a distinct syndrome (OMIM 300672), not a subtype of RTT, "
            "but continues to appear in RTT differential diagnosis panels. CDKL5 X-linked "
            "dominant; de novo mutations in ≥97% of females; males typically more severely "
            "affected with neonatal-onset epileptic encephalopathy. Seizure onset before "
            "age 5 months (often as early as 2-4 weeks) is a hallmark that distinguishes "
            "CDD from classical RTT (which rarely has seizure onset before 2 years)."
        ),
        "eeg_correlate": (
            "CDD/CDKL5 EEG: Very early-onset hypsarrhythmia (infantile spasms pattern "
            "in ~40%), transitioning to multifocal independent spike-wave with predominance "
            "over posterior regions; high-amplitude irregular delta; runs of rhythmic "
            "theta-delta bursts. A distinctive CDD EEG feature: posterior-dominant burst- "
            "attenuation pattern in the first months; later high-amplitude multifocal "
            "paroxysmal activity that never organises into a coherent background. "
            "Unlike classical RTT, there is no 'normal EEG phase' — EEG is abnormal "
            "from the earliest recording in CDD. Tonic seizures in sleep are common."
        ),
        "mri_finding": (
            "CDD MRI: similar to classical RTT with global brain volume loss, but some "
            "patients show more pronounced cerebellar hypoplasia and basal ganglia changes. "
            "MRI may be normal in early infancy despite clinical seizures — falsely "
            "reassuring in the neonatal period. Progressive atrophy on serial imaging."
        ),
        "clinical_note": (
            "CDD triad: (1) Early-onset intractable seizures (onset <5 months, often "
            "week 1-4 of life). (2) Severe intellectual disability. (3) Features "
            "overlapping RTT (hand stereotypies, limited purposeful hand use, gait apraxia). "
            "Key distinction from classical RTT: seizure onset ≫ earlier (weeks vs. years); "
            "no regression period (global delay from birth); hand stereotypies present "
            "without preceding period of normal hand use. ACTH/vigabatrin for infantile "
            "spasms (CDD-IS); ketogenic diet has shown ≥50% seizure reduction in ~60% "
            "of CDD patients. CDKL5 gene therapy under investigation (WAVE Life Sciences)."
        ),
    },
    {
        "etiology": "FOXG1 Syndrome — congenital RTT variant",
        "n": 3, "pct": 7,
        "category": "FOXG1-Congenital-RTT",
        "mechanism": (
            "Pathogenic variants in FOXG1 (forkhead box G1, 14q12) cause 'congenital Rett "
            "variant' — now recognised as FOXG1 Syndrome. FOXG1 is a transcription factor "
            "critical for early brain development, particularly telencephalon specification, "
            "neuronal proliferation, and cortical lamination. FOXG1 represses cyclin-dependent "
            "kinase inhibitor p21/CDKN1A and is required for maintenance of progenitor "
            "cell pools. Loss of FOXG1 leads to premature neuronal differentiation, "
            "reduced cortical neuron number, and dysregulation of neuronal gene networks. "
            "De novo pathogenic variants (truncating, missense affecting forkhead domain, "
            "or whole-gene deletions of 14q12) in ≥95% of cases. Autosomal dominant "
            "inheritance — not X-linked, unlike classical RTT."
        ),
        "eeg_correlate": (
            "FOXG1 EEG: often abnormal from birth; high-amplitude generalised or multifocal "
            "sharp-waves and spike-wave; runs of rhythmic delta during wakefulness; "
            "characteristic 'high-amplitude rhythmic delta with superimposed fast activity' "
            "pattern; hypsarrhythmia in a subset. Background EEG severely disorganised "
            "with no normal age-appropriate features. Runs of frontal rhythmic slowing "
            "during periods of dyskinesia (characteristic movement disorder in FOXG1)."
        ),
        "mri_finding": (
            "FOXG1 MRI: DISTINCTIVE — hypoplastic or absent frontal lobes (simplified gyral "
            "pattern, pachygyria, or lissencephaly in severe cases); dysgenesis of the "
            "corpus callosum (partial or complete agenesis of genu/body); myelination delay; "
            "reduced volume of basal ganglia. The fronto-parietal hypoplasia in FOXG1 is a "
            "diagnostic MRI clue that differentiates it from classical RTT (which has near-normal "
            "early MRI). Hypomyelination visible as T2 hyperintensity in periventricular white matter."
        ),
        "clinical_note": (
            "FOXG1 Syndrome key features: (1) Congenital onset — no regression; global "
            "developmental delay from birth (no 'normal' period). (2) Hyperkinetic "
            "movement disorder (dyskinesias, chorea, stereotypies) — distinguishes from "
            "classical RTT. (3) Severe intellectual disability with absent speech. "
            "(4) Hand stereotypies (washing/wringing type). (5) Seizures in >90% (earlier "
            "than classical RTT). Differentiates from classical RTT by: congenital onset "
            "(no regression), hyperkinetic rather than hypokinetic movement disorder, "
            "and MRI showing frontal lobe hypoplasia. AED management similar to classical "
            "RTT; VPA used cautiously; KD has shown benefit in case series."
        ),
    },
    {
        "etiology": "MECP2 Duplication Syndrome (Xq28 duplication) — severe male RTT-variant",
        "n": 3, "pct": 7,
        "category": "MECP2-Duplication-Male",
        "mechanism": (
            "Genomic duplication of Xq28 including MECP2 causes MECP2 Duplication Syndrome "
            "(MDS), a severe X-linked intellectual disability syndrome in males. GAIN of "
            "MECP2 function (duplication produces ~2x MECP2 protein in males with one X "
            "chromosome) paradoxically causes a severe neurodevelopmental syndrome, "
            "demonstrating that MECP2 dosage must be tightly regulated. Duplication "
            "sizes typically 0.3-4 Mb; larger duplications encompassing IRAK1, IKBKG, "
            "and L1CAM cause more severe phenotypes with immunodeficiency. "
            "Inheritance: X-linked — typically transmitted by carrier mothers (who have "
            "two X chromosomes with skewed X-inactivation of the duplicated X → usually "
            "unaffected). Recurrence risk: 50% for male offspring of carrier mothers. "
            "Detection: chromosomal microarray (duplication ≥200 kb detected); targeted "
            "MLPA for Xq28 region."
        ),
        "eeg_correlate": (
            "MECP2 Duplication EEG: similar to loss-of-function RTT EEG — disorganised "
            "background, multifocal spike-wave, generalised high-amplitude slow waves. "
            "Infantile spasms in ~20%. Progressive EEG deterioration correlating with "
            "clinical regression episodes. Beta-frequency bursts less characteristic "
            "than in classical MECP2 loss-of-function RTT."
        ),
        "mri_finding": (
            "MDS MRI: cerebral and cerebellar atrophy (progressive on serial imaging); "
            "T2 white matter signal changes; brain stem atrophy in severe cases. "
            "Similar to classical RTT MRI findings. Brainstem atrophy is more "
            "prominent in MECP2 duplication than in classical MECP2 loss-of-function."
        ),
        "clinical_note": (
            "MECP2 Duplication Syndrome features in males: (1) Profound intellectual disability "
            "(more severe than classical RTT females). (2) Absent or minimal speech. "
            "(3) Stereotyped hand movements and hand stereotypies. (4) Recurrent respiratory "
            "infections (progressive immunodeficiency — absent thymic shadow, low IgA). "
            "(5) Progressive spasticity and loss of ambulation. (6) Seizures in >50%. "
            "(7) Episodic deterioration ('regression episodes') triggered by infections. "
            "Prognosis poor: most males die in second-third decade from respiratory "
            "failure/infections. IVIG may reduce infection burden. KD for refractory "
            "seizures. Gene therapy target (antisense oligonucleotides to reduce MECP2 "
            "overexpression under investigation: ION363 for MECP2 duplication)."
        ),
    },
    {
        "etiology": "Clinical RTT — MECP2-sequence-negative (phenotype meets criteria, no mutation found)",
        "n": 2, "pct": 5,
        "category": "Clinical-RTT-MECP2-Negative",
        "mechanism": (
            "Approximately 3-5% of females meeting clinical diagnostic criteria for Rett "
            "Syndrome have no identifiable MECP2 pathogenic variant on complete sequencing "
            "(coding region + splice sites) and MLPA. Possible explanations: (1) Deep "
            "intronic MECP2 variants not captured by exon-focused sequencing (require whole "
            "genome sequencing). (2) Mosaicism — MECP2 somatic/gonadal mosaicism at low "
            "variant allele frequency (<15%) may be missed by Sanger sequencing; detected "
            "by next-generation sequencing with high read depth. (3) Phenotypic overlap "
            "with CDKL5, FOXG1, or other genetic syndromes classified as 'atypical RTT' "
            "variants. (4) Regulatory region variants (5' UTR, promoter) affecting MECP2 "
            "expression but not detected by coding sequencing. (5) True phenocopies — "
            "other genetic disorders (WDR45-BPAN, ATP1A3-AHC, KCNQ2) producing RTT-like "
            "phenotype. Re-sequencing with WGS in MECP2-negative RTT reveals a causative "
            "gene in up to 20-30% of cases with careful clinical re-phenotyping."
        ),
        "eeg_correlate": (
            "MECP2-negative RTT EEG: variable — may show typical RTT EEG features if "
            "clinical phenotype is true RTT (monorhythmic theta, beta bursts in sleep, "
            "multifocal spike-wave) or may show an atypical EEG suggesting an alternative "
            "diagnosis. Always repeat EEG with extended monitoring including sleep if "
            "MECP2-negative — atypical EEG features should prompt additional genetic testing "
            "(CDKL5, FOXG1, WDR45, ATP1A3)."
        ),
        "mri_finding": (
            "MECP2-negative RTT MRI: similar spectrum as MECP2-positive but may show "
            "additional specific MRI patterns suggesting alternative diagnoses: "
            "eye-of-the-tiger sign (pantothenate kinase deficiency), periventricular "
            "nodular heterotopia (FLNA), T2 hyperintensity in globus pallidus (WDR45-BPAN "
            "iron accumulation — MRI normal in childhood, abnormal in adolescence/adulthood). "
            "MRI re-review by neuroradiologist specialist mandatory in MECP2-negative cases."
        ),
        "clinical_note": (
            "Workup for MECP2-negative RTT: (1) Repeat MECP2 sequencing with high-depth NGS "
            "(coverage >100×) to detect mosaicism. (2) MECP2 MLPA (large deletions/duplications). "
            "(3) Expanded epilepsy gene panel including CDKL5, FOXG1, WDR45, ATP1A3, KCNQ2, "
            "GRIN2A, KCNA2. (4) Whole exome sequencing (WES) with trio analysis. (5) Re-review "
            "clinical phenotype: atypical features (early regression, movement disorder type, "
            "MRI findings) should guide additional testing. Treat empirically as RTT until "
            "alternative diagnosis established — AED management identical."
        ),
    },
]

# ── Seizure Types (4, with EEG correlates + clinical tips) ──────────────────
SEIZURE_TYPES = [
    {
        "type": "Focal onset with secondary generalisation (most common RTT seizure type)",
        "prevalence_pct": 80,
        "eeg_correlate": (
            "Ictal EEG: focal rhythmic beta or theta onset (most commonly over central/frontal "
            "regions) → rapid spread to contralateral hemisphere → generalised tonic-clonic "
            "pattern. Post-ictal generalised voltage attenuation followed by diffuse delta slowing. "
            "Key: ECG co-registration shows ictal bradycardia or tachycardia in RTT — "
            "cardiac rhythm monitoring is mandatory during EEG in RTT patients. "
            "Electrographic seizures without clinical correlate (subclinical seizures) "
            "are common in RTT — prolonged EEG monitoring needed for true seizure burden "
            "quantification. Interictal: multifocal spikes (central > frontal > occipital) "
            "with characteristic RTT monorhythmic theta background."
        ),
        "clinical_tip": (
            "Often misidentified as breath-holding or 'staring episodes' in early RTT. "
            "Stereotyped brief focal tonic or clonic movements (one arm/hand or face) "
            "followed by secondary generalisation. Duration typically 1-5 minutes. "
            "Prolonged seizures → status epilepticus risk (10-15% of RTT patients "
            "have at least one SE episode). The focal onset in RTT does NOT imply "
            "structural focal lesion — reflects multifocal epileptogenic networks "
            "driven by MECP2 loss. Rescue medication: buccal midazolam 0.3 mg/kg. "
            "Video-EEG to differentiate from breath-holding, stereotypies, and dyskinesias."
        ),
    },
    {
        "type": "Myoclonic seizures (focal/generalised, with or without loss of awareness)",
        "prevalence_pct": 65,
        "eeg_correlate": (
            "Ictal EEG during myoclonus: generalised polyspike or polyspike-wave discharge "
            "(100-200 ms polyspike burst followed by slow wave) synchronised with clinical "
            "myoclonic jerk; amplitude typically >200 µV. Some RTT myoclonus shows "
            "cortical EEG correlate only in back-averaging (cortical myoclonus). "
            "Absence of EEG correlate should prompt consideration of non-epileptic "
            "startle myoclonus (common in RTT and does NOT require AED escalation). "
            "Interictal generalised spike-wave and polyspike-wave paroxysms. "
            "RTT myoclonus is often stimulus-sensitive (audio, touch, startles)."
        ),
        "clinical_tip": (
            "Myoclonus in RTT: two important subtypes to distinguish: (1) Epileptic "
            "myoclonus (EEG correlate, needs AED) and (2) Non-epileptic startles/jerks "
            "(no EEG correlate, do NOT escalate AEDs). VPA is often used as first-line "
            "for epileptic myoclonus in RTT; LEV effective for stimulus-sensitive cortical "
            "myoclonus. CLN (clonazepam) 0.05-0.1 mg/kg/day useful adjunct for "
            "myoclonic-predominant RTT. Avoid PHT — worsens myoclonic EEG pattern."
        ),
    },
    {
        "type": "Generalised tonic-clonic seizures (GTCS)",
        "prevalence_pct": 50,
        "eeg_correlate": (
            "GTCS EEG: typical generalised tonic-clonic pattern — tonic phase: generalised "
            "fast activity (>20 Hz) with progressive amplitude increase; clonic phase: "
            "rhythmic generalised spike-wave or polyspike-wave at decreasing frequency "
            "(5→3→1 Hz); post-ictal generalised voltage attenuation. RTT GTCS: "
            "often preceded by focal EEG onset (secondary generalisation), not "
            "primary generalised onset — this is diagnostically important and "
            "differentiates from JME/CAE. RTT GTCS are typically brief (1-3 min) "
            "but may occur in clusters. Cardiac monitoring essential — GTCS "
            "in RTT carries SUDEP risk (combined: epilepsy + QTc prolongation + "
            "autonomic instability = elevated SUDEP risk profile)."
        ),
        "clinical_tip": (
            "GTCS in RTT most common during Stage II (regression) and Stage III "
            "(pseudo-stationary). Management: optimise maintenance AEDs (VPA + "
            "LEV or LTG combination). SUDEP prevention: (1) Nocturnal supervision "
            "or seizure detection device. (2) QTc monitoring — any QTc >470 ms "
            "requires cardiology review before adding QT-prolonging AEDs. (3) Rescue: "
            "buccal midazolam 0.3 mg/kg; rectal diazepam if no buccal. Status "
            "epilepticus protocol: lorazepam IV → levetiracetam IV if no response. "
            "KD should be considered after failure of 2 AEDs."
        ),
    },
    {
        "type": "Tonic seizures (especially in sleep, may mimic arousal disorder)",
        "prevalence_pct": 30,
        "eeg_correlate": (
            "Tonic seizure EEG: generalised high-amplitude fast activity (10-30 Hz) "
            "during the tonic phase; brief duration (5-30 seconds); often occurring "
            "from NREM sleep (Stage N2/N3). RTT tonic seizures may produce "
            "electrographic-only (subtle) changes initially with increasing "
            "clinical expression as syndrome progresses. VEEG in sleep is "
            "essential to characterise nocturnal events and differentiate "
            "tonic seizures from autonomic events (breath-holding, apnoea) "
            "and parasomnias (common in RTT — melatonin helps normalise sleep "
            "architecture). RTT pseudo-periodic delta in NREM must not be "
            "mistaken for seizure activity."
        ),
        "clinical_tip": (
            "Tonic seizures in RTT: (1) Most common nocturnal seizure type. "
            "(2) Risk: fall-related injury if patient is ambulatory. (3) Protective "
            "padding (helmet, padded side rails) for ambulatory RTT patients. "
            "(4) Nocturnal video-EEG monitoring to quantify nocturnal burden. "
            "(5) VPA and CLB are most effective for tonic seizures in RTT. "
            "(6) Sleep hygiene: melatonin 3-6 mg reduces sleep-onset latency "
            "and may reduce nocturnal seizure frequency in RTT. Do NOT use "
            "sedating AEDs (PB, BZD high-dose) as primary treatment — worsen "
            "daytime function and breathing irregularities."
        ),
    },
]

# ── Triggers (8, with seizure rates) ────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Hyperventilation / breathing irregularities (RTT-specific)",
        "seizure_rate_pct": 85,
        "note": (
            "Unique to RTT: episodic hyperventilation (forced expiration, breath-holding, "
            "air-swallowing) is a core RTT feature and is also a major seizure trigger. "
            "Hyperventilation → respiratory alkalosis → hypocapnia → cerebral vasoconstriction "
            "→ decreased seizure threshold. Management: (1) Do NOT confuse RTT hyperventilation "
            "episodes with seizures (no EEG correlate in typical episodes). (2) Deconditioning "
            "of breath-holding via behavioural/respiratory therapy. (3) Low-dose serotonin "
            "agents (buspirone) may reduce frequency of breath-holding episodes. "
            "(4) Oxygen saturation monitoring during known hyperventilation phase."
        ),
    },
    {
        "trigger": "Stress / emotional excitement / unexpected stimuli",
        "seizure_rate_pct": 80,
        "note": (
            "RTT patients show exaggerated autonomic stress responses — sudden noise, "
            "excitement, or unexpected stimuli can trigger both seizures and non-epileptic "
            "autonomic events. Cortisol and catecholamine surges lower seizure threshold. "
            "Environmental modifications: predictable routines, minimal sudden auditory "
            "stimulation, calm communication approach, low-stimulation environments "
            "during acute illness."
        ),
    },
    {
        "trigger": "Sleep deprivation / disrupted sleep architecture",
        "seizure_rate_pct": 70,
        "note": (
            "RTT is associated with severe sleep disturbance: prolonged sleep latency, "
            "frequent night awakenings, daytime hypersomnia, nocturnal screaming/laughing. "
            "Sleep disruption → increased seizure frequency (universal mechanism). "
            "Intervention: melatonin 3-6 mg at bedtime (strong evidence in RTT for "
            "improved sleep quality and duration); sleep hygiene programme. "
            "Night-time seizure monitoring: consider pulse oximetry and seizure "
            "detection device for SUDEP risk reduction."
        ),
    },
    {
        "trigger": "Fever / intercurrent infection",
        "seizure_rate_pct": 65,
        "note": (
            "Febrile seizures and fever-provoked exacerbation common in RTT. "
            "Antipyretics (paracetamol/ibuprofen) prophylactically during fever above 38°C. "
            "Respiratory infections: RTT patients have impaired airway clearance (swallowing "
            "difficulty, reduced cough reflex, aspiration risk). Aspiration pneumonia "
            "is a major cause of morbidity and seizure exacerbation. "
            "MECP2 Duplication males: immunodeficiency-driven recurrent infections "
            "cause regression episodes — IVIG may be indicated."
        ),
    },
    {
        "trigger": "Missed / late AED dose",
        "seizure_rate_pct": 60,
        "note": (
            "AED non-adherence in RTT: unique challenge — patients cannot self-administer "
            "or report seizures reliably. Feeding difficulties (gastrostomy in 20-30%) "
            "may affect oral AED absorption. G-tube patients: verify AED formulations "
            "compatible with G-tube administration (liquid/dispersible preparations). "
            "VPA liquid formulation preferred for G-tube. Monitoring: steady-state "
            "VPA TDM (target 50-100 mg/L); consider twice-daily extended-release "
            "formulations to minimise peak-trough variation."
        ),
    },
    {
        "trigger": "Constipation / gastrointestinal distress",
        "seizure_rate_pct": 45,
        "note": (
            "Constipation is near-universal in RTT (>80%) and is a unique seizure trigger "
            "via autonomic dysregulation and pain response. Management: (1) High-fibre "
            "diet; lactulose or polyethylene glycol maintenance laxatives. (2) Adequate "
            "hydration. (3) Regular bowel care programme. (4) Regular monitoring of bowel "
            "habits — families should track and report increased seizure frequency correlated "
            "with constipation episodes. Gastro-oesophageal reflux also common — rule "
            "out GORD as cause of nocturnal events and feeding aversion."
        ),
    },
    {
        "trigger": "Puberty / menstrual cycle (catamenial exacerbation)",
        "seizure_rate_pct": 35,
        "note": (
            "Puberty typically worsens seizure control in RTT (Stage IV transition). "
            "Catamenial RTT seizures documented in adolescent/adult females: perimenstrual "
            "cluster-seizures driven by oestrogen-progesterone fluctuation. "
            "Management: (1) Continuous combined OCP or depo-progesterone to suppress "
            "cycles may help catamenial RTT. (2) Neurological re-evaluation at puberty: "
            "VPA dose adjustment for weight changes. (3) Drug interactions: enzyme-inducing "
            "AEDs (CBZ, PHT, OXC) reduce OCP efficacy — if used, high-oestrogen OCP or "
            "alternative contraception. Note: most RTT patients cannot express menstrual "
            "pain — behavioural changes / increased seizures may be the only indicators."
        ),
    },
    {
        "trigger": "Physical examination / medical procedures (vagal stimulus)",
        "seizure_rate_pct": 25,
        "note": (
            "Vagal activation during physical examination, phlebotomy, dental procedures, "
            "or other medical interventions can trigger seizures and autonomic events. "
            "Pre-medication: oral midazolam 0.3 mg/kg 20 min before procedures known to "
            "trigger seizures. Dental care: RTT patients have increased caries risk (dietary "
            "modifications, bruxism) — dental GA procedures require seizure protocol and "
            "QTc check pre-anaesthesia (avoid anaesthetic agents prolonging QT: halothane, "
            "droperidol). Always check 12-lead ECG and measure QTc before any procedure "
            "requiring general anaesthesia or sedation in RTT."
        ),
    },
]

# ── Treatments (8, with dose/MOA/efficacy/safety/monitoring) ────────────────
TREATMENTS = [
    {
        "name": "Valproate / Valproic Acid (VPA)",
        "evidence_level": "Level B — Broad-spectrum, first-line RTT",
        "dose": "20-40 mg/kg/day in 2-3 divided doses; target TDM 50-100 mg/L",
        "moa": (
            "Multiple complementary mechanisms: (1) GABA enhancement via inhibition of "
            "GABA transaminase and succinic semialdehyde dehydrogenase → increased synaptic "
            "GABA concentration. (2) Blockade of voltage-gated sodium channels (use-dependent). "
            "(3) T-type calcium channel blockade (relevant for RTT absence-like events). "
            "(4) Histone deacetylase (HDAC) inhibition — may partially restore MECP2-regulated "
            "gene expression. (5) Direct MECP2-independent neuroprotective effects via BDNF "
            "upregulation in RTT mouse models."
        ),
        "efficacy": (
            "Most widely used AED in RTT; broad seizure coverage (focal, myoclonic, GTCS, tonic). "
            "Seizure reduction ≥50% in ~55-65% of RTT patients in retrospective studies. "
            "Best evidence for VPA monotherapy as first-line in RTT when seizures involve "
            "multiple seizure types. In combination with LEV: additive effect for "
            "focal-onset secondary GTCS and myoclonic seizures."
        ),
        "safety": (
            "KEY SAFETY CONCERNS IN RTT: (1) Hepatotoxicity — RTT patients on tube feeds "
            "may have carnitine deficiency potentiating VPA hepatotoxicity; measure "
            "carnitine levels at baseline and q6M; supplement if deficient "
            "(carnitine 50-100 mg/kg/day). (2) Hyperammonaemia — screen if encephalopathy "
            "episodes occur. (3) Weight gain — compound mobility issues in RTT. "
            "(4) VPA does NOT cause POLG-related hepatotoxicity unless POLG mutation "
            "co-present (POLG screening not routinely needed in RTT — MECP2 confirmed). "
            "(5) Thrombocytopenia: FBC at baseline and q6M. (6) Teratogenic (RTT females "
            "rarely reproduce but principles apply for MECP2 carrier mothers)."
        ),
        "monitoring": "VPA TDM q6M (target 50-100 mg/L); LFT + ammonia baseline + q6M; FBC; carnitine levels; weight",
        "clinical_alert": "VPA + carnitine deficiency + RTT feeding difficulties = hepatotoxicity risk. Check carnitine at every VPA review.",
    },
    {
        "name": "Levetiracetam (LEV)",
        "evidence_level": "Level B — Well-tolerated adjunct / monotherapy",
        "dose": "20-60 mg/kg/day in 2 divided doses (max 3000 mg/day)",
        "moa": (
            "Binds SV2A (synaptic vesicle glycoprotein 2A) → modulates neurotransmitter "
            "vesicle release, reducing synaptic excitability. Does NOT affect sodium channels "
            "or GABA metabolism — unique mechanism complementary to VPA. Additional: "
            "inhibits presynaptic calcium channels, reduces N-type calcium currents. "
            "No hepatic metabolism (renal excretion) — major safety advantage in RTT "
            "patients already on VPA (no pharmacokinetic interaction)."
        ),
        "efficacy": (
            "Effective for focal seizures with/without secondary generalisation (Level B). "
            "Good evidence for cortical myoclonus in RTT. Often used as add-on to VPA. "
            "In RTT observational studies: ≥50% responder rate ~50% for focal-onset seizures. "
            "IV LEV available for acute seizure management and status epilepticus "
            "(20-60 mg/kg loading dose IV). Liquid formulation (100 mg/mL) ideal for "
            "G-tube-fed RTT patients."
        ),
        "safety": (
            "Main adverse effect in RTT: behavioural — irritability, agitation, aggression "
            "(RTT patients may express LEV-related irritability as increased self-injurious "
            "behaviour or distress behaviours). Monitor carer report for behavioural change "
            "in first 4-8 weeks. Resolution on dose reduction in most cases. "
            "Rare: mood changes, anxiety. No hepatotoxicity. No drug interactions. "
            "Safe in renal impairment (dose adjust). Very good safety in RTT overall."
        ),
        "monitoring": "CBCL behavioural screen at 4 and 12 weeks; renal function baseline; dose adjust if eGFR <50",
        "clinical_alert": "Monitor for LEV-related irritability/agitation — in non-verbal RTT patients this may manifest as increased distress, self-biting, or hand-wringing intensity.",
    },
    {
        "name": "Lamotrigine (LTG)",
        "evidence_level": "Level B — Focal seizures; use caution in myoclonic-predominant RTT",
        "dose": "0.3-15 mg/kg/day depending on comedication (slow titration MANDATORY)",
        "moa": (
            "Voltage-gated sodium channel blocker (use-dependent, preferentially blocks "
            "sustained high-frequency neuronal firing). Also inhibits presynaptic release "
            "of glutamate and aspartate. Slower titration needed with VPA coadministration "
            "(VPA inhibits LTG glucuronidation, doubling LTG half-life → higher plasma LTG "
            "levels → SJS risk increased if titration too rapid)."
        ),
        "efficacy": (
            "Effective for focal seizures and GTCS in RTT. Some patients show improved "
            "alertness and communication on LTG (which may enhance quality of life beyond "
            "seizure control). CAUTION: in RTT patients with prominent myoclonus, LTG "
            "can worsen myoclonic component (paradoxical myoclonus aggravation as seen "
            "in JME/PME — monitor carefully). Best used in RTT patients with "
            "focal-predominant seizures without significant myoclonic component."
        ),
        "safety": (
            "CRITICAL: Stevens-Johnson Syndrome (SJS) risk — serious cutaneous adverse "
            "reaction (1/1,000 in children). Risk factors: rapid titration, high starting "
            "dose, VPA coadministration. Mitigation: very slow titration (2 mg/kg/day "
            "every 2 weeks); starting dose 0.15 mg/kg/day with VPA. Any rash → STOP "
            "immediately and seek emergency review. RTT-specific: rash may not be "
            "reported by patient — train carers to inspect skin daily for first 3 months."
        ),
        "monitoring": "Daily skin inspection for rash (train carers) × 12 weeks; LTG level optional; eye examination if visual complaints; slow titration log",
        "clinical_alert": "MANDATORY slow titration — RTT patients cannot report early SJS symptoms (sore mouth, photophobia). Caregiver daily skin check first 3 months.",
    },
    {
        "name": "Clobazam (CLB)",
        "evidence_level": "Level B — Adjunct for tonic/focal seizures; tolerance risk",
        "dose": "0.1-0.5 mg/kg/day in 1-2 divided doses (max 20 mg/day children)",
        "moa": (
            "1,5-benzodiazepine (different structure from 1,4-BZD like diazepam). "
            "Positive allosteric modulator of GABA-A receptor (α2/α3 subunit preferential "
            "binding → lower sedation than 1,4-BZDs). Enhances chloride ion conductance "
            "→ membrane hyperpolarisation → reduced neuronal excitability. "
            "Less tolerance development than 1,4-benzodiazepines, but tolerance DOES occur "
            "over 3-6 months of continuous use (relevant in RTT long-term management)."
        ),
        "efficacy": (
            "Useful adjunct in RTT for tonic seizures (Level B), focal seizures, and "
            "as intermittent 'cluster-busting' therapy (3-5 day courses during seizure "
            "clusters). Catamenial RTT: CLB 10-20 mg for 10 days perimenstrually "
            "as add-on. Useful for status epilepticus prevention: short course during "
            "febrile illness or stress periods. Responder rate ~45-60% short-term; "
            "diminishes at 6 months due to tolerance."
        ),
        "safety": (
            "Main: sedation, hypersalivation, tolerance, withdrawal seizures on abrupt "
            "discontinuation. RTT-specific: CLB sedation compounds daytime hypersomnia "
            "and may worsen hypotonia (feeding/aspiration risk). Respiratory: "
            "avoid high-dose CLB in RTT patients with significant breathing irregularities "
            "(may worsen central apnoea episodes). Do NOT stop CLB abruptly — taper "
            "over ≥4 weeks. Drug interaction: VPA may increase CLB N-desmethyl "
            "metabolite (active, longer half-life — monitor excess sedation)."
        ),
        "monitoring": "UMRS (Unified Myoclonus Rating Scale) at baseline/3M/6M; tolerance assessment q3M; breathing irregularity diary during initiation; withdrawal protocol on cessation",
        "clinical_alert": "Tolerance common at 6 months — plan CLB as adjunct with periodic drug holidays or intermittent use for clusters rather than continuous maintenance.",
    },
    {
        "name": "Ketogenic Diet (KD) — 4:1 ratio or MAD",
        "evidence_level": "Level B — For medically refractory RTT seizures (≥2 AED failures)",
        "dose": "Classic 4:1 KD or modified Atkins diet (MAD); target BHB 2-4 mmol/L",
        "moa": (
            "Ketone bodies (beta-hydroxybutyrate BHB, acetoacetate) serve as alternative "
            "cerebral fuel; reduce glycolytic flux; shift neuronal metabolism toward "
            "mitochondrial oxidative phosphorylation. Anti-epileptic mechanisms: "
            "(1) BHB activates GABA-A receptors directly. (2) BHB inhibits NLRP3 "
            "inflammasome (anti-neuroinflammatory). (3) Reduces vesicular glutamate "
            "release. (4) Activates KATP channels → hyperpolarisation. "
            "RTT-specific potential: MECP2 loss impairs mitochondrial function; "
            "KD may partially bypass MECP2-dependent metabolic deficits."
        ),
        "efficacy": (
            "RTT-specific KD evidence: retrospective case series (Haas 2011, Liebhaber 2003) "
            "show ≥50% seizure reduction in 50-70% of RTT patients with refractory epilepsy. "
            "KD also reported to improve alertness, behaviour, and communication quality "
            "in RTT (anecdotally). MAD may be better tolerated in RTT patients with "
            "feeding difficulties and G-tube dependence (more flexible macronutrient targets). "
            "Initiate after failure of ≥2 AEDs in RTT."
        ),
        "safety": (
            "RTT-specific KD safety: (1) G-tube patients: KD formula available "
            "(KetoCal 4:1 liquid — suitable for G-tube; Ketofino for older patients). "
            "(2) Constipation: RTT already prone to constipation — KD worsens this; "
            "aggressive laxative management required. (3) Growth monitoring: KD may "
            "impair growth in children with already poor nutritional status — monthly "
            "weight/height checks; dietitian review q3M. (4) Bone density: DEXA annually "
            "(RTT patients have reduced bone density from immobility + anticonvulsants). "
            "(5) Carnitine: supplement if deficient (especially with VPA coadministration)."
        ),
        "monitoring": "BHB twice-weekly (target 2-4 mmol/L); urine ketones daily; weight/height monthly; lipid panel q3M; selenium/zinc/carnitine annually; DEXA annually; constipation diary",
        "clinical_alert": "G-tube RTT patients on KD: ensure KD-compatible formula (KetoCal); coordinate between neurology, dietitian, and gastroenterology teams for tube feeding protocol.",
    },
    {
        "name": "Trofinetide (Daybue®) — FDA-approved March 2023",
        "evidence_level": "Level A — FDA-approved for Rett Syndrome (≥2 years); NOT an AED",
        "dose": "Weight-based: 12.5-200 mg/kg BID oral solution (250 mg/mL); see FDA label",
        "moa": (
            "Synthetic analogue of glycine-proline-glutamate (GPE), the N-terminal tripeptide "
            "of IGF-1. Trofinetide promotes synaptic maturation and neuronal recovery by: "
            "(1) Modulating IGF-1R signalling → MAPK/ERK and PI3K/AKT → downstream "
            "neurotrophic effects. (2) Anti-inflammatory actions via astrocyte/microglia "
            "regulation. (3) Reducing synaptic glutamate excitotoxicity. (4) Animal RTT "
            "model: restores dendritic spine density and partially normalises MECP2-regulated "
            "gene expression. IMPORTANT: approved for improvement of RTT functional and "
            "behavioural symptoms — NOT specifically approved as anti-seizure medication; "
            "some patients report improved seizure control as secondary benefit."
        ),
        "efficacy": (
            "LAVENDER trial (Neul 2023, NEJM Evidence): N=187 females with RTT age 5-20 years. "
            "Primary outcomes: CGIC (Clinician's Global Impression of Change) and RTT-BSS "
            "(Behaviour Symptom Scale). Results: statistically significant improvement in "
            "CGIC (p<0.0001) and RTT-BSS (p<0.0001) vs. placebo at 12 weeks. "
            "Secondary: improved hand function and communication. "
            "Seizure outcomes: mixed — some patients showed reduced seizure frequency. "
            "Effect size modest but clinically meaningful for caregivers. "
            "Post-marketing experience ongoing."
        ),
        "safety": (
            "Main adverse effects (LAVENDER trial): diarrhoea (82% trofinetide vs 19% "
            "placebo — most common side effect), nausea (37% vs 5%), decreased appetite "
            "(20% vs 5%), vomiting (12% vs 4%). Diarrhoea management: antidiarrhoeals "
            "(loperamide), dose reduction, or temporary discontinuation. RTT-specific: "
            "diarrhoea compounds constipation → diarrhoea cycle (RTT gut dysmotility). "
            "Weight loss possible (monitor closely, especially in underweight RTT patients). "
            "No cardiac QTc effects reported. No drug interactions with common RTT AEDs."
        ),
        "monitoring": "Body weight weekly × 4 weeks then monthly; GI symptom diary (diarrhoea scoring); G-tube patients: fluid balance; neurological function: RTT-BSS at baseline/4 weeks/3M/6M",
        "clinical_alert": "Diarrhoea affects >80% — have loperamide ready before starting; monitor weight closely in already-underweight RTT patients. NOT a replacement for AEDs.",
    },
    {
        "name": "Sarizotan (investigational)",
        "evidence_level": "Level B (Phase III STARS trial) — for RTT breathing irregularities",
        "dose": "1-10 mg/day oral (weight-adjusted; Phase III dosing as per trial)",
        "moa": (
            "Sarizotan is a selective serotonin 5-HT1A receptor agonist and dopamine D2/D3/D4 "
            "receptor antagonist. Mechanism in RTT breathing: (1) 5-HT1A agonism "
            "modulates brainstem respiratory centres (Pre-Bötzinger complex) → normalises "
            "breathing rhythm generation. (2) In MECP2 mouse models: sarizotan normalises "
            "respiratory irregularities and reduces breath-holding episodes. "
            "(3) Indirect seizure benefit: by reducing hyperventilation-triggered episodes, "
            "may reduce seizure frequency driven by this RTT-specific trigger. "
            "Not a direct anti-seizure mechanism."
        ),
        "efficacy": (
            "STARS trial (Phase II/III, ongoing): targeting RTT breathing irregularities. "
            "Phase II (Nissenkorn 2019, Eur J Paed Neurol): small open-label study showed "
            "significant reduction in apnoea/hyperventilation events. Phase III STARS "
            "trial enrolment ongoing — results expected 2025. Currently investigational; "
            "compassionate use in severe RTT breathing irregularities in some countries. "
            "Indirect seizure benefit via reduced hyperventilation trigger expected but "
            "not confirmed in Phase III data."
        ),
        "safety": (
            "Adverse effects in early trials: mild dizziness, somnolence, nausea at "
            "higher doses. QTc: mild QT-prolonging potential — baseline ECG mandatory "
            "in RTT (already at QTc-prolongation risk from underlying RTT). "
            "Drug interactions: avoid concurrent medications with significant QT-prolonging "
            "risk. Metabolised CYP2D6 — avoid CYP2D6 inhibitors (fluoxetine, paroxetine). "
            "NOT commercially available; access via clinical trial or compassionate use "
            "programme only."
        ),
        "monitoring": "QTc monitoring (baseline + 2 weeks + monthly); breathing irregularity diary; caregiver-rated apnoea/hyperventilation frequency",
        "clinical_alert": "INVESTIGATIONAL — compassionate use or clinical trial only. QTc check mandatory before initiation (RTT + sarizotan = compound QTc risk).",
    },
    {
        "name": "Phenytoin (PHT)",
        "evidence_level": "ABSOLUTE CONTRAINDICATION — worsens RTT seizures and EEG",
        "dose": "N/A — DO NOT USE",
        "moa": (
            "PHT: voltage-gated sodium channel blocker (use-dependent). Problem in RTT: "
            "PHT paradoxically worsens multifocal myoclonic and absence-like seizures "
            "in RTT (same mechanism as in Dravet, MAE, and other generalised epilepsies). "
            "PHT is ineffective for the myoclonic, absence, and tonic seizure types "
            "predominating in RTT and may produce seizure aggravation in 20-30% of RTT patients."
        ),
        "efficacy": (
            "PHT has no established efficacy for the seizure types in RTT (focal-onset "
            "secondary GTCS, myoclonic, tonic). Data from sodium channel-blocking AEDs "
            "in RTT: paradoxical worsening documented in case series. The dominant "
            "pathophysiology in RTT involves multifocal networks with myoclonic components — "
            "sodium channel-only blockers perform poorly in this context."
        ),
        "safety": (
            "PHT in RTT: (1) Seizure aggravation — myoclonic worsening documented. "
            "(2) EEG deterioration — increased paroxysmal activity on EEG reported. "
            "(3) PHT-induced cerebellar toxicity at therapeutic levels (irreversible "
            "cerebellar atrophy with chronic use) — RTT patients already have progressive "
            "ataxia. (4) QTc prolongation — compound risk with RTT-related QTc prolongation. "
            "(5) Drug interactions: PHT induces CYP450 → reduces VPA, CLB, LTG levels."
        ),
        "monitoring": "N/A — contraindicated",
        "clinical_alert": "ABSOLUTE CONTRAINDICATION in RTT. If patient presenting on PHT → URGENT neurology review, switch to VPA/LEV. Do NOT continue PHT in any RTT patient.",
    },
]

# ── AED Monitoring (5 key items) ─────────────────────────────────────────────
AED_MONITORING = [
    {
        "item": "VPA therapeutic drug monitoring + hepatic safety panel",
        "frequency": "Baseline then every 6 months (or 4 weeks after dose change)",
        "target": "TDM 50-100 mg/L; ALT/AST <2× ULN; carnitine ≥25 µmol/L",
        "rationale": "RTT patients on G-tube nutrition are at elevated VPA hepatotoxicity risk from carnitine depletion. Supplement carnitine if levels low (50-100 mg/kg/day). Ammonia if encephalopathic.",
    },
    {
        "item": "QTc monitoring (12-lead ECG)",
        "frequency": "Baseline; at 2 weeks; q6M on stable AEDs; before any new QT-affecting drug",
        "target": "QTc <440 ms (males), <460 ms (females); RTT threshold: action at QTc >470 ms",
        "rationale": "RTT intrinsically prolongs QTc via MECP2 loss-mediated cardiac ion channel dysregulation. VPA, CLB, sarizotan, and many anaesthetic agents add further QT risk. SUDEP prevention: cardiac rhythm monitoring for nocturnal events.",
    },
    {
        "item": "LTG skin and SJS surveillance (caregiver training)",
        "frequency": "Daily caregiver skin inspection × 12 weeks from LTG initiation; monthly thereafter",
        "target": "No cutaneous rash; mucous membrane intact; no fever with rash",
        "rationale": "RTT patients cannot self-report SJS symptoms (sore mouth, photophobia, skin pain). Trained caregiver daily skin check is the ONLY reliable surveillance method. Any rash → stop LTG immediately and seek emergency review.",
    },
    {
        "item": "LEV behavioural monitoring (irritability/agitation)",
        "frequency": "Behavioural screen (CBCL or carer-rated behavioural diary) at 4 weeks, 12 weeks, 6 months",
        "target": "No significant increase in irritability, agitation, or self-injurious behaviour",
        "rationale": "RTT patients express LEV-induced irritability as behavioural distress, increased hand-wringing, self-biting, or screaming. Caregiver report essential — reduce LEV dose if behavioural deterioration occurs.",
    },
    {
        "item": "Nutritional and growth monitoring (VPA + KD + RTT baseline)",
        "frequency": "Weight/height monthly; DEXA annually; carnitine q6M; albumin/pre-albumin q6M",
        "target": "Weight z-score >-2SD; DEXA T-score >-2.5; carnitine ≥25 µmol/L",
        "rationale": "RTT patients have poor nutritional status (oromotor dysfunction, feeding difficulties, constipation). AEDs (VPA, KD) further compromise nutrition. G-tube in 20-30% — formula selection must account for AED and KD compatibility.",
    },
]

# ── Lifecycle Windows (6 stages) ─────────────────────────────────────────────
LIFECYCLE_WINDOWS = [
    {
        "window": "Pre-Regression Phase (Stages I): Birth to 6-18 months",
        "phase": "apparently-normal",
        "description": (
            "Classical RTT patients appear neurologically normal (or near-normal) for the first "
            "6-18 months of life. Subtle early signs may be present: mild hypotonia, feeding "
            "difficulties, reduced eye contact, and slightly delayed motor milestones — but these "
            "are often missed or attributed to normal variation. The brain during this period has "
            "sufficient MECP2 protein from X-chromosome inactivation escape to maintain normal "
            "function; the phenotype emerges as MECP2-related neuronal processes become critical "
            "post-6 months. Seizures are NOT typically present in Stage I (onset before 2 years "
            "suggests CDKL5 or an alternative diagnosis)."
        ),
        "key_actions": (
            "Genetic testing if ANY developmental concern: MECP2 sequencing + MLPA. "
            "Developmental surveillance q3M. Avoid premature AED prescription for non-epileptic "
            "events (breath-holding, startle). 12-lead ECG baseline (QTc). "
            "MECP2 testing for siblings of known RTT patients."
        ),
    },
    {
        "window": "Rapid Regression (Stage II): 1-4 years",
        "phase": "regression",
        "description": (
            "The defining feature of RTT: rapid, dramatic loss of previously acquired skills "
            "over weeks to months. Hand skills lost first (purposeful grasping → hand stereotypies "
            "emerging); then spoken language (babble/words → absent speech); then social engagement "
            "(autism-like social withdrawal), followed by motor regression. Concurrently: "
            "characteristic hand-wringing stereotypies appear; breathing irregularities begin "
            "(hyperventilation, breath-holding); epilepsy emerges (median age 2-3 years). "
            "EEG transitions from normal to Stage II RTT pattern. This is the highest-distress "
            "period for families — early genetic confirmation, multidisciplinary support, and "
            "AED initiation as needed."
        ),
        "key_actions": (
            "Confirm MECP2 diagnosis urgently. Start AED when seizures confirmed (not for "
            "non-epileptic events). Video-EEG to characterise seizures vs. breathing irregularities. "
            "Physiotherapy, occupational therapy, and speech-language pathology referral. "
            "Gastroenterology: assess for feeding difficulties and GORD. Family genetic counselling. "
            "RTT specialist network referral. Introduce AAC (augmentative and alternative communication)."
        ),
    },
    {
        "window": "Pseudo-Stationary Phase (Stage III): 2-10 years",
        "phase": "pseudo-stationary",
        "description": (
            "The 'plateau' of RTT: the devastating regression of Stage II halts, and the clinical "
            "picture stabilises (but does NOT improve back to pre-regression baseline). Key features "
            "of Stage III: seizures most frequent and difficult to control; hand stereotypies "
            "established; breathing irregularities prominent; communication severely limited "
            "but eye gaze and facial expression may be preserved and serve as primary communication. "
            "Alertness and social engagement may slightly improve versus Stage II. "
            "Sleep disturbance: severe, characterised by prolonged latency, frequent wakings, "
            "nocturnal laughing/screaming. Most RTT patients remain ambulatory through most "
            "of Stage III. This stage may last many years."
        ),
        "key_actions": (
            "Optimise AED regimen (≥2 failures → KD consideration). Trofinetide initiation "
            "(if approved in jurisdiction). Melatonin for sleep. QTc monitoring q6M. "
            "School-based physiotherapy and occupational therapy. AAC systems optimised. "
            "Gastrostomy consideration if nutrition inadequate. Orthopaedics: scoliosis surveillance "
            "(progressive in 80% of RTT) — spinal X-ray annually. Ophthalmology: visual "
            "acuity (RTT gaze is primary communication — preserve vision)."
        ),
    },
    {
        "window": "Late Deterioration (Stage IV): >10 years",
        "phase": "late-deterioration",
        "description": (
            "Stage IV is defined by progressive loss of motor function: gait deteriorates "
            "and most patients eventually become non-ambulatory (by third-fourth decade). "
            "Seizure frequency may paradoxically decrease in Stage IV (EEG paroxysmal "
            "activity gradually reduces — unclear mechanism; possibly reflects progressive "
            "neuronal loss). Scoliosis progresses and may require surgical correction. "
            "Swallowing difficulties worsen (aspiration risk increases). Cardiovascular: "
            "QTc prolongation risk increases with age. Behavioural: relative calm compared "
            "to Stage III; the 'happy affect' characteristic of RTT often preserved in Stage IV."
        ),
        "key_actions": (
            "Wheelchair prescription and postural support. Scoliosis surgery evaluation "
            "(if Cobb angle >40°). Dysphagia assessment (MBSS); gastrostomy if aspiration "
            "risk high. Cardiology annual review (QTc, arrhythmia). DEXA for osteoporosis "
            "(vitamin D + calcium supplementation). AED review — may be able to reduce "
            "AED burden if seizures improve (slow taper over 6-12 months). Advance care "
            "planning discussion with family. Palliative care team involvement."
        ),
    },
    {
        "window": "Adult RTT (18-40 years)",
        "phase": "adult",
        "description": (
            "Adult RTT is an evolving area — most published natural history data "
            "covers childhood/adolescent RTT. Key adult issues: (1) Transition from "
            "paediatric to adult neurology (AED continuity essential — transitions are "
            "a high-risk period for seizure breakthrough). (2) Continued progression "
            "of scoliosis, contractures, and mobility loss. (3) Gastrointestinal: "
            "worsening constipation, GORD, dysphagia — gastrostomy dependence increases. "
            "(4) Cardiac: QTc monitoring, risk of sudden cardiac death (autonomic instability). "
            "(5) Reproductive health: catamenial seizure management; hormonal contraception "
            "planning. Contraceptive choices must account for AED interactions."
        ),
        "key_actions": (
            "Planned transition (age 16-18): paediatric → adult neurology handover with "
            "detailed AED summary and RTT management plan. Update AED dosing for adult weight. "
            "Annual multi-disciplinary review (neurology/cardiology/gastroenterology/orthopaedics). "
            "Catamenial seizure management (CLB cycle, OCP). DEXA q2Y; calcium + vitamin D "
            "supplementation. Flu/pneumococcal vaccination (aspiration pneumonia prevention). "
            "QTc annual ECG. End-of-life planning."
        ),
    },
    {
        "window": "Geriatric/Older RTT (>40 years)",
        "phase": "geriatric",
        "description": (
            "Rett Syndrome in older adulthood is rare in published data (most RTT "
            "natural history studies have limited follow-up beyond 40 years). Key issues: "
            "increasing frailty, severe scoliosis, loss of ambulation, feeding dependence, "
            "and increasing SUDEP risk. Dementia-like progression superimposed on baseline "
            "profound intellectual disability. Many older RTT patients require full-time "
            "residential care. AED management in older RTT: polypharmacy risk increases; "
            "VPA metabolism changes (hepatic clearance declines — dose reduction may be needed "
            "to avoid toxicity). Caregiver and family support critical in this phase."
        ),
        "key_actions": (
            "AED review for polypharmacy — simplify if possible. VPA TDM more frequent "
            "(hepatic clearance declines with age). Falls prevention and protective equipment. "
            "Pressure area care (immobility). Nutritional support — G-tube assessment. "
            "Infection prevention: pneumococcal + influenza vaccination. Palliative care "
            "advance directives in place. SUDEP prevention: nocturnal monitoring device. "
            "Family/carer psychological support."
        ),
    },
]

# ── Concepts / Definitions (14) ────────────────────────────────────────────
CONCEPTS = [
    {"term": "MECP2 (Methyl-CpG-Binding Protein 2)", "definition": "X-linked gene at Xq28 encoding a transcriptional regulator that binds methylated CpG dinucleotides. Haploinsufficiency in females (X-linked dominant) causes Rett Syndrome. MECP2 dosage-sensitive: both loss (RTT) and gain (MECP2 Duplication Syndrome) cause severe neurodevelopmental disorders. MECP2 modulates neuronal gene expression, synaptic maturation, and chromatin organisation."},
    {"term": "Rett Syndrome (RTT)", "definition": "X-linked dominant neurodevelopmental disorder caused by MECP2 pathogenic variants (Xq28). Characterised by apparently normal early development followed by regression of purposeful hand skills, speech, and social skills, then stabilisation with stereotyped hand movements, gait apraxia, breathing irregularities, autonomic dysregulation, and drug-resistant epilepsy. Classical RTT is almost exclusively in females; males with MECP2 pathogenic variants have severe neonatal encephalopathy."},
    {"term": "RTT Regression", "definition": "The defining clinical feature of Rett Syndrome: loss of previously acquired purposeful hand skills, spoken language, and social skills occurring typically between 1-4 years (Stage II). Regression may be abrupt (weeks) or gradual (months). Must be distinguished from neurodegenerative regression (which involves loss of pyramidal function and is excluded by RTT diagnostic criteria). Post-regression, a pseudo-stationary phase typically follows."},
    {"term": "Hand Stereotypies (RTT)", "definition": "Involuntary, repetitive hand movements replacing purposeful hand use in Rett Syndrome. Classic RTT stereotypies: hand-wringing (most characteristic), hand-washing/rubbing, hand-squeezing, hand-mouthing, and clapping. Appear during regression (Stage II) as purposeful hand skills are lost. Stereotypies are not epileptic (no EEG correlate) and should not trigger AED escalation."},
    {"term": "Breathing Irregularities (RTT)", "definition": "Episodic breathing disturbances in RTT: (1) Hyperventilation — forced, rapid breathing (breath volumes 3-5× normal). (2) Breath-holding — voluntary or reflex breath-holding cycles. (3) Air-swallowing (aerophagia). (4) Forceful expiration (Valsalva). Occur in wakefulness, not in sleep. A major RTT-specific seizure trigger (via hypocapnia → reduced seizure threshold) and may mimic seizures. Importantly: NOT epileptic — no EEG correlate. Sarizotan investigational for this feature."},
    {"term": "QTc Prolongation in RTT", "definition": "Prolongation of the corrected QT interval on ECG (QTc >440 ms males, >460 ms females) occurs in 20-30% of RTT patients due to MECP2 loss-mediated dysfunction of cardiac ion channel gene regulation. Clinical significance: risk of torsades de pointes arrhythmia and sudden cardiac death. Compound risk with AEDs (PHT, CBZ) and other QT-prolonging drugs. SUDEP risk in RTT is partially mediated via cardiac arrhythmia during seizures (ictal bradycardia). Mandatory 12-lead ECG at RTT diagnosis and q6M."},
    {"term": "SUDEP in RTT", "definition": "Sudden Unexpected Death in Epilepsy — estimated 0.4-1.0%/year in RTT (higher than general epilepsy population: ~0.1-0.35%/year). RTT-specific SUDEP risk factors: (1) Drug-resistant epilepsy. (2) QTc prolongation (cardiac arrhythmia risk). (3) Autonomic dysregulation (basal heart rate and variability abnormalities). (4) Nocturnal seizures (reduced arousal). Prevention: nocturnal seizure monitors, prone position avoidance during sleep, optimise seizure control, QTc surveillance."},
    {"term": "Trofinetide (Daybue®)", "definition": "First FDA-approved pharmacotherapy specifically for Rett Syndrome (approved March 2023, for females ≥2 years). Synthetic analogue of IGF-1 N-terminal tripeptide (glycine-proline-glutamate, GPE). Mechanism: promotes synaptic maturation via IGF-1R signalling, reduces neuroinflammation. Approved for improvement of RTT behavioural and functional symptoms — NOT a licensed AED. Main side effect: diarrhoea (82%). Shown to improve CGIC and RTT-BSS in the LAVENDER Phase III trial."},
    {"term": "Sarizotan", "definition": "Investigational 5-HT1A receptor agonist / dopamine antagonist under Phase III evaluation (STARS trial) for RTT breathing irregularities (apnoea/hyperventilation episodes). Mechanism: modulates brainstem respiratory pattern generators. Potential indirect anti-seizure benefit via reduction of hyperventilation-triggered seizures. Not yet commercially available — access via clinical trial or compassionate use only."},
    {"term": "CDKL5 (Cyclin-Dependent Kinase-Like 5)", "definition": "X-linked serine-threonine kinase at Xp22.13. CDKL5 Deficiency Disorder (CDD) was historically classified as 'early-onset seizure variant of Rett Syndrome' (Hanefeld variant). Now recognised as a distinct syndrome (OMIM 300672). Key distinguishing feature from classical RTT: seizure onset before 5 months of age (often neonatal). CDKL5 phosphorylates MECP2 — mechanistic overlap with RTT but clinically distinct. KD, ACTH (for infantile spasms), and LEV are mainstays of CDD management."},
    {"term": "FOXG1 Syndrome (Congenital RTT Variant)", "definition": "Autosomal dominant syndrome caused by FOXG1 (14q12) pathogenic variants. Clinically overlaps with RTT but distinguished by: (1) Congenital onset (no regression — developmental delay from birth). (2) Hyperkinetic dyskinesias (vs. hypokinetic RTT). (3) Frontal lobe hypoplasia on MRI. (4) Earlier, more severe epilepsy. Classified as 'congenital variant of RTT' in older literature but now recognised as FOXG1 Syndrome."},
    {"term": "MECP2 Duplication Syndrome", "definition": "X-linked disorder caused by duplication of Xq28 encompassing MECP2 — gain of MECP2 function paradoxically causes severe neurodevelopmental disorder in males. Features: profound intellectual disability, absent speech, hand stereotypies, progressive spasticity, recurrent infections (immunodeficiency), and seizures. Carrier females are usually unaffected (skewed X-inactivation). ION363 (antisense oligonucleotide) in clinical trials to reduce MECP2 overexpression."},
    {"term": "RTT Monorhythmic Theta (Rett Theta)", "definition": "A characteristic EEG finding in Rett Syndrome: monorhythmic 4-6 Hz theta activity over central regions during wakefulness, replacing the normal posterior dominant alpha rhythm. Not ictal — represents the disorganised RTT background rhythm. Should not be treated as a seizure pattern. Also seen: runs of rhythmic frontal-central 10-25 Hz beta bursts (especially in NREM sleep) — another RTT EEG hallmark."},
    {"term": "X-Inactivation / Lyonisation in RTT", "definition": "Random X-chromosome inactivation (XCI) in females means each cell inactivates either the maternal or paternal X chromosome. In RTT females (heterozygous MECP2 pathogenic variant): approximately 50% of neurons express the mutant MECP2 and 50% express the normal MECP2. Severity correlates with XCI skewing: heavily skewed XCI favouring the mutant X → more severe RTT; skewed XCI favouring the normal X → milder RTT. XCI ratio is clinically useful in predicting phenotypic severity and explaining intrafamilial variability."},
]

# ── Absolute Contraindications ────────────────────────────────────────────────
ABSOLUTE_CONTRAINDICATIONS = [
    {
        "drug": "Phenytoin (PHT)",
        "scope": "ABSOLUTE CONTRAINDICATION — all RTT patients",
        "mechanism": (
            "PHT is a selective sodium-channel blocker without GABAergic or calcium-channel "
            "effects. In RTT, which involves mixed seizure types (focal, myoclonic, absence-like, "
            "tonic), sodium-channel-only blockade is insufficient and paradoxically worsens "
            "myoclonic components. PHT additionally: (1) Prolongs QTc (compound cardiac risk "
            "with RTT). (2) Causes cerebellar toxicity at therapeutic levels (worsening "
            "ataxia/gait in already-impaired RTT). (3) Induces CYP450 → reduces VPA, "
            "CLB, LTG levels. (4) EEG deterioration documented in RTT case series."
        ),
        "consequence": "Seizure aggravation (myoclonus worsening), QTc prolongation, cerebellar toxicity, EEG deterioration.",
        "action": "STOP PHT immediately if patient presenting on it. Switch to VPA or LEV after neurology review.",
    },
    {
        "drug": "Carbamazepine / Oxcarbazepine (CBZ / OXC)",
        "scope": "RELATIVE-to-ABSOLUTE CONTRAINDICATION — avoid in myoclonic or absence-predominant RTT",
        "mechanism": (
            "CBZ/OXC are sodium-channel blockers. In RTT patients with myoclonic-prominent "
            "seizures, CBZ/OXC worsen the myoclonic component (same mechanism as in MAE, PME). "
            "RTT-specific additional risk: (1) QTc prolongation (CBZ blocks cardiac sodium "
            "channels → QTc prolongation risk in RTT patients already prone to long QT). "
            "(2) Hyponatraemia risk (OXC-SIADH) — compound with RTT fluid/nutritional difficulties. "
            "(3) Induction of CYP3A4 → reduced CLB levels (loss of CLB efficacy in RTT). "
            "May be used with caution in RTT patients with purely focal-onset seizures "
            "without myoclonic components — but cardiac (QTc) monitoring mandatory."
        ),
        "consequence": "Myoclonus worsening, QTc prolongation, hyponatraemia (OXC), drug interactions.",
        "action": "Avoid in RTT with any myoclonic component. If used for pure focal RTT, cardiac monitoring mandatory.",
    },
    {
        "drug": "Phenobarbitone / Primidone (PB)",
        "scope": "RELATIVE CONTRAINDICATION — sedation compounds RTT functional decline",
        "mechanism": (
            "PB: long-acting barbiturate GABA-A positive allosteric modulator (non-selective). "
            "While PB has anti-seizure efficacy, in RTT it produces marked sedation that "
            "worsens RTT-related daytime hypersomnia, compound feeding difficulties, aspiration "
            "risk, and impairs already-limited communication capacity. PB also worsens "
            "breathing irregularities (central respiratory depression + RTT brainstem "
            "dysfunction). RTT patients on PB show higher rates of aspiration pneumonia "
            "and functional deterioration in retrospective case series."
        ),
        "consequence": "Sedation → impaired communication, feeding difficulties, aspiration risk, breathing irregularity worsening.",
        "action": "Avoid as maintenance AED. May be used short-term (24-48h) for status epilepticus only. Prefer LEV/VPA for long-term management.",
    },
    {
        "drug": "Any QT-prolonging drug without prior ECG clearance",
        "scope": "ABSOLUTE — any QT-prolonging drug requires QTc check before prescribing in RTT",
        "mechanism": (
            "RTT patients have intrinsically elevated risk of QTc prolongation and torsades de "
            "pointes (due to MECP2 loss → dysregulated cardiac ion channel gene expression, "
            "particularly KCNQ1, KCNH2/hERG, and SCN5A). Any additional QT-prolonging drug "
            "(class IA/III antiarrhythmics, macrolide antibiotics, antipsychotics, antiemetics, "
            "some antifungals, sarizotan) further elevates arrhythmia risk. RULE: Check QTc "
            "with a 12-lead ECG before any new prescription in RTT patients. QTc >470 ms: "
            "do not add QT-prolonging drugs; cardiology review mandatory."
        ),
        "consequence": "Torsades de pointes ventricular tachycardia, ventricular fibrillation, sudden cardiac death.",
        "action": "Mandatory ECG before any new prescription. QTc >470 ms → cardiology review before new drug. Maintain medication record of all QT-prolonging drugs.",
    },
]

# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"name": "QTc >470 ms", "category": "cardiac", "action": "STOP QT-prolonging drugs; urgent cardiology referral; consider Holter monitoring; do NOT add further QT-prolonging AEDs."},
    {"name": "VPA TDM 50-100 mg/L", "category": "pharmacology", "action": "Target therapeutic range; >100 mg/L → sedation, tremor, hepatotoxicity risk; <50 mg/L → likely inadequate if seizures uncontrolled."},
    {"name": "AED failure threshold: ≥2 AEDs", "category": "epilepsy-management", "action": "After failure of 2 appropriately trialled AEDs → ketogenic diet consideration mandatory; trofinetide initiation (if seizure benefit expected); VNS evaluation."},
    {"name": "Seizure-free 2 years", "category": "AED-taper", "action": "2 years seizure-freedom → discuss AED taper with family; RTT epilepsy is frequently lifelong — most remain on AEDs; only taper if Stage IV plateau confirmed."},
    {"name": "Cobb angle >40° (scoliosis)", "category": "orthopaedic", "action": "Refer to orthopaedic surgery for bracing or spinal fusion consideration; scoliosis surgery has high complication risk in RTT — cardiac/respiratory pre-op clearance essential."},
    {"name": "Carnitine <25 µmol/L (VPA-treated RTT)", "category": "metabolic", "action": "Start carnitine supplementation: 50-100 mg/kg/day oral; monitor monthly until normalised; prevent VPA hepatotoxicity."},
    {"name": "KD BHB target 2-4 mmol/L", "category": "dietary-therapy", "action": "Inadequate ketosis (<2 mmol/L) → increase dietary fat ratio or reduce carbohydrate; BHB >6 mmol/L → risk of metabolic acidosis; adjust diet ratio."},
    {"name": "Driving exclusion", "category": "safety", "action": "All RTT patients: driving not applicable (profound intellectual disability + epilepsy). Document legally as DVLA/transport authority medical bar. Carer transportation planning essential."},
]

# ── Standards ─────────────────────────────────────────────────────────────────
STANDARDS = [
    {"name": "ILAE 2022 Classification", "domain": "Epilepsy classification", "relevance": "RTT epilepsy classified as genetic epileptic encephalopathy; MECP2 gene listed under X-linked dominant genetic epilepsies. Seizure type classification: focal onset + myoclonic + GTCS + tonic."},
    {"name": "NICE NG217 (Epilepsies: diagnosis and management, UK 2022)", "domain": "AED management", "relevance": "Guidance on management of genetic epilepsy syndromes; recommends specialist paediatric neurology centre for complex epileptic encephalopathies including RTT. AED choice principles applicable to RTT."},
    {"name": "International Rett Syndrome Foundation (IRSF) Consensus Guidelines", "domain": "RTT comprehensive management", "relevance": "IRSF consensus recommendations for RTT diagnosis, AED management, QTc monitoring, nutrition, respiratory care, scoliosis surveillance, and social/communication support. Key practical guidance for multidisciplinary RTT care."},
    {"name": "FDA Daybue (trofinetide) Approval 2023 (NDA 216018)", "domain": "Pharmacotherapy", "relevance": "FDA approved trofinetide (Daybue®) for treatment of Rett Syndrome in patients ≥2 years based on the LAVENDER Phase III trial. Prescribing information mandatory reading before initiation. REMS not required; weight-based dosing from FDA-approved label."},
    {"name": "ACMG-AMP Variant Interpretation Standards 2015 (Richards et al.)", "domain": "Genetic variant classification", "relevance": "5-tier variant classification system (pathogenic/likely-pathogenic/VUS/likely-benign/benign) applied to MECP2 variants identified on RTT genetic testing. Essential for correct interpretation of MECP2 sequencing results and clinical decision-making."},
    {"name": "ACNS EEG Monitoring Standards 2021", "domain": "EEG", "relevance": "Standards for long-term EEG monitoring in RTT: video-EEG (24-72h) for seizure characterisation; ECG co-registration during EEG mandatory to detect ictal bradycardia; polysomnography for nocturnal event characterisation."},
]

# ── References (6) ────────────────────────────────────────────────────────────
REFERENCES = [
    {
        "authors": "Amir RE, Van den Veyver IB, Wan M, et al.",
        "year": 1999,
        "title": "Rett syndrome is caused by mutations in X-linked MECP2, encoding methyl-CpG-binding protein 2.",
        "journal": "Nature Genetics",
        "vol": "23",
        "pages": "185-188",
        "pmid": "10508514",
        "note": "Landmark discovery of MECP2 as the causative gene for RTT. Established X-linked dominant inheritance and molecular basis of RTT.",
    },
    {
        "authors": "Neul JL, Kaufmann WE, Glaze DG, et al. (RettSearch Consortium)",
        "year": 2010,
        "title": "Rett syndrome: revised diagnostic criteria and nomenclature.",
        "journal": "Annals of Neurology",
        "vol": "68",
        "pages": "944-950",
        "pmid": "21154482",
        "note": "Revised RTT diagnostic criteria (Neul 2010): defines typical RTT (4 main criteria) and atypical/variant RTT. Reference standard for clinical diagnosis.",
    },
    {
        "authors": "Chahrour M, Jung SY, Shaw C, et al.",
        "year": 2008,
        "title": "MeCP2, a key contributor to neurological disease, activates and represses transcription.",
        "journal": "Science",
        "vol": "320",
        "pages": "1224-1229",
        "pmid": "18511691",
        "note": "Showed that MECP2 activates as well as represses gene transcription — fundamental revision of MECP2 function from pure repressor to dual-function regulator.",
    },
    {
        "authors": "Glaze DG, Percy AK, Skinner S, et al.",
        "year": 2010,
        "title": "Epilepsy and the natural history of Rett syndrome.",
        "journal": "Neurology",
        "vol": "74",
        "pages": "909-912",
        "pmid": "20231667",
        "note": "Natural history study of epilepsy in 1,440 RTT patients: prevalence 93%, median onset 2.5 years, drug-resistant in 30-40%. Key epidemiology reference for RTT epilepsy.",
    },
    {
        "authors": "Neul JL, Benke TA, Marsh ED, et al.",
        "year": 2023,
        "title": "The relationship between the severity of Rett syndrome and MECP2 mutation type.",
        "journal": "Annals of Neurology",
        "vol": "93",
        "pages": "163-175",
        "pmid": "36322085",
        "note": "LAVENDER trial publication (trofinetide Phase III): N=187, primary endpoints CGIC and RTT-BSS both significant at 12 weeks (p<0.0001). Basis for FDA Daybue approval 2023.",
    },
    {
        "authors": "Nissenkorn A, Levy-Drummer RS, Bondi O, et al.",
        "year": 2019,
        "title": "Epilepsy in Rett syndrome — lessons from the Rett networked database.",
        "journal": "Epilepsia",
        "vol": "60",
        "pages": "1889-1900",
        "pmid": "31509237",
        "note": "Comprehensive RTT epilepsy registry analysis from 6 Rett centres: AED patterns, responder rates, and long-term outcomes. Validates VPA and LEV as most commonly used first-line agents.",
    },
]

# ── Patient Generator ─────────────────────────────────────────────────────────
def _generate_patients(n=41):
    random.seed(SEED)
    etiology_pool = (
        ["Classical-RTT-MECP2"] * 29 +
        ["CDKL5-Atypical-RTT"] * 4 +
        ["FOXG1-Congenital-RTT"] * 3 +
        ["MECP2-Duplication-Male"] * 3 +
        ["Clinical-RTT-MECP2-Negative"] * 2
    )
    random.shuffle(etiology_pool)

    seizure_pool = [
        "Focal+Myoclonic", "Focal+GTCS", "Focal+Myoclonic+GTCS", "Tonic+Focal",
        "Myoclonic+Tonic", "Focal+GTCS+Tonic", "Focal only", "Myoclonic+GTCS",
    ]
    treatment_pool = [
        "VPA mono", "VPA+LEV", "LEV+LTG", "VPA+CLB", "VPA+LEV+CLB",
        "LEV mono", "VPA+KD", "LTG+LEV", "VPA+LEV+KD",
    ]
    control_pool = (
        ["seizure-free"] * 6 +
        ["partial-response"] * 18 +
        ["drug-resistant"] * 17
    )
    random.shuffle(control_pool)

    phase_by_etiology = {
        "Classical-RTT-MECP2": ["regression", "pseudo-stationary", "late-deterioration", "adult", "pseudo-stationary"],
        "CDKL5-Atypical-RTT": ["neonatal-seizure-onset", "regression", "pseudo-stationary", "late-deterioration"],
        "FOXG1-Congenital-RTT": ["congenital-onset", "pseudo-stationary", "pseudo-stationary"],
        "MECP2-Duplication-Male": ["severe-neonatal", "deterioration", "adult"],
        "Clinical-RTT-MECP2-Negative": ["pseudo-stationary", "late-deterioration"],
    }

    pts = []
    for i in range(n):
        etiology = etiology_pool[i]
        is_male = etiology == "MECP2-Duplication-Male"
        sex = "M" if is_male else "F"
        age = random.randint(3, 42) if is_male else random.randint(2, 45)
        onset_months = random.randint(1, 36) if etiology in ("CDKL5-Atypical-RTT", "FOXG1-Congenital-RTT") else random.randint(12, 48)
        phase_list = phase_by_etiology.get(etiology, ["pseudo-stationary"])
        pts.append({
            "id": f"RTT-{i+1:02d}",
            "age": age,
            "sex": sex,
            "onset_age_months": onset_months,
            "etiology": etiology,
            "seizure_types": random.choice(seizure_pool),
            "disease_phase": random.choice(phase_list),
            "current_treatment": random.choice(treatment_pool),
            "seizure_control": control_pool[i],
            "qtc_prolonged": "Y" if random.random() < 0.27 else "N",
            "kd_on": "Y" if "KD" in random.choice(treatment_pool) and random.random() < 0.4 else "N",
            "trofinetide": "Y" if random.random() < 0.22 else "N",
            "gastrostomy": "Y" if random.random() < 0.28 else "N",
        })
    return pts


# ── Public API ─────────────────────────────────────────────────────────────────
def get_overview():
    pts = _generate_patients()
    n = len(pts)
    seizure_free = sum(1 for p in pts if p["seizure_control"] == "seizure-free")
    drug_resistant = sum(1 for p in pts if p["seizure_control"] == "drug-resistant")
    qtc_prolonged = sum(1 for p in pts if p["qtc_prolonged"] == "Y")
    kd_on = sum(1 for p in pts if p["kd_on"] == "Y")
    trofinetide = sum(1 for p in pts if p["trofinetide"] == "Y")
    gastrostomy = sum(1 for p in pts if p["gastrostomy"] == "Y")
    avg_onset = round(sum(p["onset_age_months"] for p in pts) / n, 1)

    return {
        "syndrome": "Rett Syndrome (RTT)",
        "gene": "MECP2 (Xq28)",
        "inheritance": "X-linked dominant",
        "n_patients": n,
        "key_gene": "MECP2 (Xq28) — X-linked dominant",
        "eeg_hallmark": "Monorhythmic central theta 4-6 Hz + frontal-central beta bursts in sleep; loss of posterior dominant rhythm",
        "key_biomarker": "MECP2 sequencing + MLPA (deletions); QTc >470 ms cardiac alert",
        "key_aha": "PHT is CONTRAINDICATED in RTT — worsens myoclonus + QTc. MANDATORY: ECG at diagnosis (QTc). Trofinetide FDA-approved 2023.",
        "etiologies": [
            {"etiology": "Classical RTT (MECP2 pathogenic variant)", "category": "Classical-RTT-MECP2", "pct": 71},
            {"etiology": "CDKL5 Deficiency Disorder (Hanefeld variant)", "category": "CDKL5-Atypical-RTT", "pct": 10},
            {"etiology": "FOXG1 Syndrome (congenital RTT variant)", "category": "FOXG1-Congenital-RTT", "pct": 7},
            {"etiology": "MECP2 Duplication Syndrome (males)", "category": "MECP2-Duplication-Male", "pct": 7},
            {"etiology": "Clinical RTT — MECP2-negative", "category": "Clinical-RTT-MECP2-Negative", "pct": 5},
        ],
        "seizure_type_prevalence": {
            "Focal onset + secondary generalisation": 80,
            "Myoclonic seizures": 65,
            "Generalised tonic-clonic (GTCS)": 50,
            "Tonic seizures (nocturnal)": 30,
        },
        "trigger_seizure_rates": {
            "Hyperventilation/breathing irregularities (RTT-specific)": 85,
            "Stress/emotional excitement": 80,
            "Sleep deprivation": 70,
            "Fever/infection": 65,
            "Missed AED dose": 60,
            "Constipation/GI distress": 45,
            "Puberty/catamenial": 35,
            "Vagal/medical procedure": 25,
        },
        "lifecycle_windows": LIFECYCLE_WINDOWS,
        "kpis": {
            "seizure_free_pct": round(seizure_free / n * 100, 1),
            "drug_resistant_pct": round(drug_resistant / n * 100, 1),
            "qtc_prolonged_pct": round(qtc_prolonged / n * 100, 1),
            "kd_responder_pct": round(kd_on / n * 100, 1),
            "trofinetide_pct": round(trofinetide / n * 100, 1),
            "gastrostomy_pct": round(gastrostomy / n * 100, 1),
            "avg_onset_age_months": avg_onset,
        },
        "clinical_alerts": [
            "🚫 ABSOLUTE CONTRAINDICATION: Phenytoin (PHT) — worsens myoclonus + QTc prolongation in RTT",
            "⚠️ RELATIVE CONTRAINDICATION: CBZ/OXC — myoclonus worsening + QTc risk",
            "💊 MANDATORY ECG at diagnosis: QTc >470 ms → cardiology review before any new drug",
            "🧬 First FDA-approved RTT pharmacotherapy: Trofinetide (Daybue®) March 2023",
            "🍳 Ketogenic diet after ≥2 AED failures — effective in RTT refractory epilepsy",
            "💉 Monitor carnitine levels in all VPA-treated RTT patients (G-tube risk)",
            "😮‍💨 Breathing irregularities (hyperventilation/breath-holding) are NOT seizures — do NOT escalate AEDs",
            "🔬 MECP2-negative RTT: order WGS + CDKL5/FOXG1 panel",
        ],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def get_breakdown():
    pts = _generate_patients()
    return {
        "patients": pts,
        "etiology_catalog": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "aed_monitoring": AED_MONITORING,
        "lifecycle_windows": LIFECYCLE_WINDOWS,
        "standards": STANDARDS,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


def get_definitions():
    return {
        "concepts": CONCEPTS,
        "absolute_contraindications": ABSOLUTE_CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "references": REFERENCES,
        "standards": STANDARDS,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
