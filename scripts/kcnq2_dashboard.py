"""
KCNQ2 Encephalopathy (KCNQ2-DEE) Dashboard
============================================
41-patient cohort · KCNQ2 gain-of-function / dominant-negative · 20q13.33
KCNQ2 encephalopathy: pathogenic de novo variant in KCNQ2 (voltage-gated potassium
channel subunit Kv7.2, 20q13.33) causing neonatal-onset epileptic encephalopathy with
tonic seizures within the first hours to days of life, burst-suppression EEG, and
significant neurodevelopmental disability. Distinct from the benign self-limited neonatal
epilepsy (SLNE) associated with KCNQ2 loss-of-function (formerly BFNS).
KEY EEG: Burst-suppression pattern in neonates with inter-burst suppression intervals;
evolves to multifocal independent spike-wave complexes and diffuse theta slowing.
AED NOTE: Carbamazepine / Oxcarbazepine (Na-channel blockers / Kv7-modulating) are
FIRST-LINE SPECIFIC therapy in KCNQ2-DEE — uniquely effective compared to phenobarbital
alone. Phenobarbital remains the acute neonatal ICU first-line.
DISEASE-MODIFYING: XEN496 (Kv7.2/Kv7.3 opener) Phase 2/3 EPIK trial ongoing (2024).
Retigabine/Ezogabine (historical Kv7 opener) WITHDRAWN 2017 (retinal pigmentation, blue
skin discolouration) — do NOT prescribe; referenced for mechanism understanding only.
"""

import random
from datetime import datetime

SEED = 9174  # dashboard 174
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ─────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "De novo KCNQ2 gain-of-function / dominant-negative missense",
        "n": 18, "pct": 45,
        "category": "De-novo-KCNQ2-GOF-dominant-negative",
        "mechanism": (
            "The most clinically severe and most common variant class in KCNQ2-DEE (~45%): de novo "
            "missense variants that exert a dominant-negative or gain-of-function (GOF) effect on "
            "the Kv7.2 (KCNQ2) channel. Kv7.2 co-assembles with Kv7.3 (KCNQ3) to form the "
            "heteromeric M-current channel — a slow non-inactivating potassium current that "
            "determines neuronal excitability at the resting membrane potential and regulates "
            "repetitive firing. Dominant-negative missense variants (e.g., R201H, R201C, A265T, "
            "R213Q, G310V) — particularly in the voltage-sensing S4 transmembrane domain or the "
            "pore-forming S5-S6 linker — incorporate into Kv7 heterotetramers and suppress >75% "
            "of wild-type M-current activity. This loss of M-current results in hyperexcitability "
            "of neonatal and early-infantile cortical networks, predominantly in regions with high "
            "physiological KCNQ2 expression: neocortex layer V pyramidal neurons, hippocampal CA3, "
            "and deep cerebellar nuclei. The neonatal-specific vulnerability reflects the late "
            "developmental switch of KCC2 (potassium-chloride cotransporter 2), which makes GABA "
            "depolarising rather than hyperpolarising in the first weeks of life — compound "
            "hyperexcitability with loss of M-current tipping neonates into tonic SE. "
            "Key GOF hot-spots: R201H (most frequent globally, ~10-12% of all KCNQ2-DEE); "
            "R201C (second most common); A265T (S5 pore-helix, severe phenotype). "
            "Genotype-phenotype: R201H/C carriers have the most severe DEE with near-complete "
            "burst-suppression at birth, profound global DD (< 3 months developmental age at "
            "5 years), absent speech and ambulation. G310V shows intermediate severity."
        ),
        "eeg_correlate": (
            "Characteristic burst-suppression (BS) pattern in the neonatal period: "
            "alternating bursts of high-amplitude (100-400 µV) mixed-frequency delta-theta "
            "activity (0.5-7 Hz) with sharp waves and spikes, separated by near-flat suppression "
            "intervals (< 5 µV, duration 2-10 seconds). The BS pattern in KCNQ2-GOF is typically "
            "HEMISYNCHRONOUS — bursts appear synchronously across both hemispheres (unlike the "
            "asynchronous BS of OTAHARA syndrome from ARX or structural causes). "
            "Ictal EEG: Tonic seizures manifest as diffuse low-voltage fast activity (10-20 Hz) "
            "evolving to generalised high-amplitude rhythmic theta-delta; may be associated with "
            "electroclinical dissociation in phenobarbital-treated infants. "
            "Evolution by 3-6 months: BS resolves; replaced by diffuse high-amplitude "
            "disorganised background with multifocal independent spike-wave complexes (centrotemporal, "
            "frontal, occipital). Sleep EEG: increased interictal spike-wave index in NREM; "
            "no physiological sleep architecture. Long-term (> 1 year): persistent diffuse "
            "background slowing (theta/delta) with scattered multifocal IEDs; may have "
            "interval hypnagogic hypersynchrony-like discharges."
        ),
        "mri_finding": (
            "Brain MRI in de novo KCNQ2-GOF (dominant-negative): "
            "(1) Delayed myelination — the single most common finding (~70%): T2 hyperintensity in "
            "the posterior limb of internal capsule, centrum semiovale, and subcortical white matter "
            "on MRI at 6-18 months compared to age-matched norms; correlates with severity of DEE. "
            "(2) Reduced myelination signal on T1 (less bright white matter) — same distribution. "
            "(3) Thin corpus callosum (hypoplasia of the body and splenium) in ~35% of GOF cases — "
            "particularly the R201H variant. "
            "(4) Reduced gyration / simplified gyral pattern in ~15% (most severe). "
            "(5) Diffuse cerebral atrophy — progressive loss of gray and white matter volume "
            "detectable on serial MRI from 12 months onwards. "
            "(6) Normal basal ganglia and thalami (unlike POLG or metabolic disorders). "
            "(7) Cerebellar hypoplasia in ~20% — vermian volume reduction. "
            "DWI: Transient restricted diffusion in the basal ganglia or thalami during acute "
            "neonatal SE — reversible in most cases if seizures controlled within 48-72 hours. "
            "MRS: Reduced NAA/Cr (neuronal loss), elevated Cho/Cr (myelin turnover) in white matter."
        ),
    },
    {
        "etiology": "De novo KCNQ2 loss-of-function missense (DEE variant)",
        "n": 12, "pct": 29,
        "category": "De-novo-KCNQ2-LOF-DEE",
        "mechanism": (
            "Approximately 29% of the cohort carry de novo KCNQ2 missense variants that cause "
            "loss-of-function (LOF) — complete haploinsufficiency — but result in DEE rather than "
            "the benign self-limited neonatal epilepsy (SLNE) typically associated with familial "
            "LOF variants. The distinction between DEE-causing LOF and SLNE-causing LOF appears "
            "to depend on: (1) the degree of residual M-current — variants reducing current to "
            "< 10% of wild-type correlate with DEE; SLNE-associated variants typically retain "
            "25-40% residual current. (2) Functional subtype: variants affecting the C-terminal "
            "calmodulin-binding helix (responsible for PIP2-mediated gating regulation) tend to "
            "cause DEE even without dominant-negative effect. "
            "LOF variants in KCNQ2-DEE cluster in the S4-S5 linker, the pore (S6), and the "
            "C-terminal intracellular domain. Examples: Y284C, L197R, I205V, D212G. "
            "Pathophysiology: with < 10% M-current, neonatal cortical networks cannot sustain "
            "membrane repolarisation after repetitive firing — action potential bursting becomes "
            "self-sustaining and difficult to terminate pharmacologically. "
            "Phenotype: slightly less severe than GOF dominant-negative (R201H); some LOF-DEE "
            "patients achieve limited communication (2-word phrases); seizure remission by 18-24M "
            "in ~70% (vs 90% for GOF by 12M). Intellectual disability universal."
        ),
        "eeg_correlate": (
            "LOF-DEE EEG is broadly similar to GOF burst-suppression but with some distinguishing "
            "features: (1) BS pattern present but INTER-BURST INTERVALS tend to be slightly shorter "
            "(2-6 seconds vs 5-10 seconds in severe GOF) — consistent with less complete "
            "M-current suppression. (2) Onset may be slightly later — day 2-7 of life vs within "
            "hours for R201H. (3) Resolution of BS pattern typically by 4-6 months (vs 3-4 months "
            "for GOF). (4) Prominent multifocal independent IEDs in the chronic phase (3+ months), "
            "preferentially centrotemporal and occipital. (5) No HAFA (unlike CDD) — predominant "
            "theta slowing with superimposed multifocal spike-wave. Ictal patterns in LOF-DEE: "
            "predominantly focal clonic (hemispheric; may alternate sides) and evolving to "
            "tonic-clonic. Tonic seizures less predominant than in GOF variants. EEG sleep study: "
            "significant interictal burden in NREM; may show electrical SE during sleep (ESES) "
            "after 18-24 months in a subset of LOF-DEE."
        ),
        "mri_finding": (
            "LOF-DEE MRI: similar delayed myelination pattern (~60%) but generally less severe than "
            "GOF dominant-negative. (1) Thin corpus callosum in ~25% (less than GOF). "
            "(2) Simplified gyral pattern rare (< 5%). (3) Progressive cerebral atrophy less "
            "marked than GOF — cortical volume loss more focal (frontal > temporal). "
            "(4) Normal posterior fossa in majority. (5) Hippocampal T2 signal change during "
            "prolonged neonatal SE — may result in hippocampal sclerosis on follow-up MRI "
            "(detected at 2-3 years). DWI: transient thalamic restricted diffusion during "
            "acute prolonged seizures (< 4 hours onset) — reversible with prompt treatment. "
            "MRS: mild NAA reduction in frontal white matter — less severe than GOF group."
        ),
    },
    {
        "etiology": "De novo KCNQ2 truncating variant (nonsense / frameshift)",
        "n": 5, "pct": 12,
        "category": "De-novo-KCNQ2-truncating-nonsense-frameshift",
        "mechanism": (
            "Approximately 12% of cohort: de novo truncating variants (nonsense = stop-codon, "
            "frameshift = insertion/deletion causing premature stop) in KCNQ2. These generate "
            "null alleles via NMD for variants in exons 1-13 (prior to the C-terminal regulatory "
            "domain). KCNQ2 truncations result in complete haploinsufficiency — 50% of normal "
            "channel subunit — which in the neonatal neocortex reduces total M-current by ~50-60% "
            "given the 4:1 Kv7.2:Kv7.3 stoichiometry of the heteromeric M-channel. Unlike "
            "familial SLNE truncations (which often occur in KCNQ2 regions retaining partial "
            "function), de novo neonatal DEE truncations tend to cluster in the N-terminal "
            "voltage-sensing domain and the S4 linker region. Phenotypic outcome is intermediate: "
            "more severe than familial SLNE; less severe than dominant-negative GOF. "
            "Key distinguishing feature: truncating KCNQ2-DEE is more likely to show partial "
            "response to CBZ/OXC (60-70% vs 85% for GOF) — this is an important clinical point "
            "because some truncating variants may benefit less from Na-channel blockade mechanisms."
        ),
        "eeg_correlate": (
            "Truncating KCNQ2-DEE EEG: (1) BS pattern present but may be less pronounced — "
            "some patients show only high-amplitude disorganised burst activity without true "
            "inter-burst suppression ('modified BS'). (2) Earlier resolution than GOF — "
            "BS may normalise to multifocal IEDs as early as 2-3 months. (3) Predominant focal "
            "features: independent hemispheric spike-wave complexes alternating or simultaneous. "
            "(4) May evolve to a Lennox-Gastaut-like pattern (if seizure burden heavy in first "
            "year) — slow spike-wave 1.5-2.5 Hz with paroxysmal fast activity in NREM. "
            "Treatment response: CBZ/OXC often reduces IED burden within 48-72 hours on EEG — "
            "useful objective biomarker for therapeutic response in truncating variants."
        ),
        "mri_finding": (
            "Truncating KCNQ2-DEE MRI: (1) Delayed myelination in ~50% — intermediate severity. "
            "(2) Normal gyration in most (> 90%). (3) Thin corpus callosum in ~20%. "
            "(4) Normal cerebellum in most. (5) Serial MRI shows less progressive atrophy than "
            "missense GOF variants — implying partial preservation of neuronal survival. "
            "No specific MRI pattern distinguishes truncating from missense DEE — genetic "
            "diagnosis essential for classification."
        ),
    },
    {
        "etiology": "Familial KCNQ2 (AD-SLNE) with neonatal DEE presentation",
        "n": 4, "pct": 10,
        "category": "Familial-AD-KCNQ2-SLNE-DEE-presentation",
        "mechanism": (
            "Approximately 10% of cohort: familial KCNQ2 variants transmitted in autosomal "
            "dominant fashion (parent to child) — classically associated with self-limited "
            "neonatal epilepsy (SLNE, formerly 'benign familial neonatal epilepsy' / BFNE). "
            "However, in this cohort subset, the neonatal presentation meets DEE criteria: "
            "prolonged seizures > 10 min, severe BS pattern on EEG, or early feeding difficulties "
            "and hypotonia inconsistent with SLNE. Explanation: (1) The same familial LOF variant "
            "may show variable expressivity — more severe in index case compared to mildly affected "
            "or asymptomatic parent due to modifier genetic background or de novo epigenetic "
            "regulation. (2) Concurrent perinatal injury (HIE, prematurity, infection) may "
            "amplify seizure burden beyond the level seen in isolated SLNE. (3) 'SLNE plus' "
            "phenotype — variant carriers within the family may have borderline DEE (ID mild, "
            "ADHD, learning disability). These families require genetic counselling: recurrence "
            "risk is 50% per pregnancy; prenatal/preimplantation genetic testing available. "
            "IMPORTANT: Familial SLNE-associated variants (e.g., Y284C in some families) may "
            "behave differently on functional assay compared to isolated de novo occurrence of "
            "the same variant — context-dependence of variant interpretation (ACMG VUS reclassification)."
        ),
        "eeg_correlate": (
            "Familial AD-KCNQ2 EEG: (1) Inter-ictal BS pattern often absent or less severe — "
            "predominant finding is multifocal independent sharp waves / spike-wave complexes "
            "on a mildly disorganised background. (2) Ictal: typically focal tonic or clonic "
            "seizures with rhythmic theta build-up, lasting 30-90 seconds and self-terminating. "
            "(3) Normalisation: EEG normalises to age-appropriate pattern by 4-6 weeks in "
            "typical SLNE — delayed normalisation (> 3 months) suggests DEE phenotype. "
            "(4) Family history EEG review may reveal similar patterns in mildly affected parent "
            "(often asymptomatic in adult life)."
        ),
        "mri_finding": (
            "Familial AD-KCNQ2 MRI: typically normal or near-normal in SLNE phenotype. "
            "In DEE-presenting cases: mild delayed myelination in ~30%. Normal gyration and "
            "corpus callosum. Normal posterior fossa. No progressive atrophy. "
            "MRI normal by 12 months in most familial cases — prognostically reassuring."
        ),
    },
    {
        "etiology": "Clinical KCNQ2-DEE — KCNQ2-negative (expanded panel pending)",
        "n": 2, "pct": 4,
        "category": "Clinical-KCNQ2-DEE-KCNQ2-negative",
        "mechanism": (
            "Approximately 4% of cohort: neonates meeting clinical and EEG criteria for "
            "KCNQ2-DEE (neonatal-onset tonic seizures, burst-suppression EEG, profound DD) "
            "but with no pathogenic variant identified in KCNQ2 on standard panel sequencing. "
            "Possible explanations: (1) Deep intronic or promoter-region KCNQ2 variant not "
            "captured by standard exon-focused NGS — requires KCNQ2 RNA-seq or long-read "
            "whole-genome sequencing (WGS). (2) Somatic mosaicism in KCNQ2 (variant present "
            "in < 10% of blood-derived DNA — below standard sequencing threshold) — confirm "
            "with deep amplicon sequencing (1000x coverage) or brain-specific epigenetic "
            "analysis if resection available. (3) KCNQ3 variant (Kv7.3 partner subunit) — "
            "rarer but increasingly recognised cause of KCNQ2-phenocopy. "
            "(4) KCNQ5, KCNT1, or SCN8A — alternative neonatal K+/Na+ channel genes causing "
            "phenocopy. (5) Pyridoxine-dependent epilepsy (ALDH7A1) — B6-responsive DEE "
            "mimicking KCNQ2 clinically: always trial IV pyridoxine 100mg while awaiting "
            "genetics. Management: treat empirically as KCNQ2-DEE with CBZ/OXC; expand "
            "genetic panel (170-gene DEE panel or WGS); consider KCNQ3 RNA-seq."
        ),
        "eeg_correlate": (
            "Clinical KCNQ2-negative DEE EEG: indistinguishable from KCNQ2-positive at "
            "neonatal stage — burst-suppression with interictal suppression intervals, "
            "focal tonic ictal pattern. Empirical AED trial (CBZ/OXC) will differentiate "
            "KCNQ2 channel-opathy (rapid response within 24-48 hours) from non-KCNQ2 "
            "aetiology (non-response or partial response requiring alternative agents). "
            "Sequential AED trial approach: phenobarbital → add CBZ/OXC → if no EEG "
            "response in 72 hours, expand genetic workup and consider pyridoxine trial."
        ),
        "mri_finding": (
            "KCNQ2-negative DEE MRI: similar to KCNQ2-positive — delayed myelination, "
            "thin corpus callosum in subset. Brain MRI may provide diagnostic clues: "
            "basal ganglia signal change suggests metabolic disorder (GLUT1, organic "
            "acidemias); cortical malformation suggests GRIN2A, mTOR pathway; "
            "absent corpus callosum suggests ARX; diffuse polymicrogyria suggests GPR56/ADGRG1. "
            "Normal MRI with BS EEG strongly favours channelopathy (KCNQ2/KCNQ3/KCNT1)."
        ),
    },
]

# ── Seizure Types (4 types with EEG correlates) ──────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Tonic Seizures (neonatal tonic / focal tonic)",
        "prevalence_pct": 100,
        "category": "Tonic-neonatal-focal-tonic",
        "description": (
            "Universal in KCNQ2-DEE — present in 100% of patients. The hallmark seizure type: "
            "sustained tonic posturing (unilateral or bilateral) lasting 10-90 seconds, "
            "with or without associated autonomic changes (cyanosis, apnoea, bradycardia). "
            "In neonates: may present as subtle eye deviation, sustained limb extension, "
            "trunk stiffening, or 'tonic posturing' that is difficult to distinguish from "
            "normal newborn posturing — requires video-EEG to confirm electroclinical "
            "correlation. Tonic seizures reflect hyperactivation of layer V cortical "
            "pyramidal neurons (highest KCNQ2 expression) projecting to subcortical "
            "motor output pathways. Frequency: 5-50 seizures per day in untreated "
            "neonates; reduces to 1-10/day on phenobarbital alone; achieves near "
            "seizure-freedom (< 1/month) on CBZ/OXC in 80-85% of KCNQ2-GOF."
        ),
        "eeg_correlate": (
            "Ictal EEG: abrupt-onset diffuse low-voltage fast activity (beta, 15-30 Hz) "
            "evolving over 5-15 seconds to rhythmic generalised theta-delta (4-6 Hz) of "
            "increasing amplitude — the 'ramp-up' pattern; post-ictal voltage attenuation "
            "(electrodecrement) lasting 5-20 seconds. In focal tonic seizures: identical "
            "pattern but lateralised (hemispheric onset, contralateral to posturing limb)."
        ),
        "clinical_tip": (
            "CRITICAL: Tonic seizures in KCNQ2-DEE often show only subtle clinical features "
            "in phenobarbital-sedated neonates (electroclinical dissociation). Always perform "
            "amplitude-integrated EEG (aEEG) or continuous video-EEG monitoring in the NICU "
            "for objective seizure quantification. Response to CBZ (starting dose 5 mg/kg/day "
            "orally via NG tube) typically produces > 50% seizure frequency reduction within "
            "24-48 hours — use EEG response as therapeutic biomarker."
        ),
    },
    {
        "type": "Focal Clonic Seizures (multifocal / alternating)",
        "prevalence_pct": 78,
        "category": "Focal-clonic-multifocal-alternating",
        "description": (
            "Present in ~78% of patients — often co-occurring with tonic seizures. "
            "Rhythmic clonic jerking of one limb or one face hemibody, alternating sides "
            "between seizures (multifocal clonic), reflecting the diffuse but "
            "asymmetric irritability of the neonatal cortex. Focal clonic seizures "
            "are more common in LOF-DEE and truncating variant groups than in GOF. "
            "Duration: 30-120 seconds. Rarely evolve to secondary generalisation in "
            "the neonatal period. Post-ictal state: brief limpness or hypotonia "
            "of the affected limb. After 3-6 months: focal clonic seizures tend to "
            "remit coincident with myelination of corticospinal tracts."
        ),
        "eeg_correlate": (
            "Ictal EEG: rhythmic focal spike-wave complexes (2-5 Hz) in the contralateral "
            "central or central-temporal region, with progressive spread to the ipsilateral "
            "hemisphere; each clinical jerk time-locked to a spike in the EEG complex. "
            "Background: multifocal independent interictal spike-wave; BS in severe cases."
        ),
        "clinical_tip": (
            "Multifocal clonic seizures alternating sides in a neonate with BS EEG: "
            "highest-priority clinical scenario for KCNQ2 genetic testing. Order "
            "rapid single-gene KCNQ2 sequencing (48-72 hour result) or rapid WES/DEE "
            "panel simultaneously with clinical management. Do NOT wait for genetics "
            "before trialling CBZ/OXC — empiric trial is both therapeutic and diagnostic."
        ),
    },
    {
        "type": "Autonomic Seizures (apnoea / bradycardia / cyanosis)",
        "prevalence_pct": 55,
        "category": "Autonomic-apnoea-bradycardia-cyanosis",
        "description": (
            "Present in ~55% — often the first recognised clinical manifestation in "
            "the delivery room or NICU day 1. Isolated apnoea, cyanosis, or bradycardia "
            "occurring repetitively and not explained by primary cardiorespiratory cause. "
            "Autonomic seizures reflect ictal activation of insular cortex and anterior "
            "cingulate — high KCNQ2 expression regions with direct autonomic projections. "
            "Risk: SUDEP precursor mechanism in neonates — repeated prolonged autonomic "
            "seizures (apnoea > 20 seconds) require cardiac-respiratory monitoring and "
            "may necessitate temporary NICU ventilatory support during acute phase. "
            "Autonomic seizures remit first (typically by 4-6 weeks) as KCNQ2 "
            "expression matures and inhibitory circuits develop."
        ),
        "eeg_correlate": (
            "Ictal EEG during autonomic seizures: bifrontal or generalised low-amplitude "
            "fast activity (10-20 Hz) or diffuse theta build-up (4-6 Hz), often brief "
            "(10-30 seconds). May be of low amplitude and missed on aEEG — requires "
            "full-channel EEG. In some cases: ictal EEG is flat/suppressed during the "
            "clinical event (electroclinical dissociation typical of brainstem seizures)."
        ),
        "clinical_tip": (
            "Any neonate with recurrent unexplained apnoea or bradycardia in the NICU "
            "(especially if born term, no HIE, no infection) should have URGENT video-EEG "
            "within 2 hours — apnoeic seizures are often missed on clinical examination alone. "
            "KCNQ2 genetic panel should be in the differential on day 1 of NICU admission."
        ),
    },
    {
        "type": "Focal-to-Bilateral Tonic-Clonic (FBTCS / Generalised TC)",
        "prevalence_pct": 32,
        "category": "FBTCS-evolving-neonatal",
        "description": (
            "Present in ~32% — typically the most dramatic seizure type clinically: "
            "bilateral tonic stiffening with subsequent clonic jerking of all four limbs, "
            "cyanosis, and post-ictal obtundation. FBTCS represent secondary generalisation "
            "from a focal tonic onset — the neonatal cortex has limited myelination to "
            "suppress generalisation. FBTCS in KCNQ2-DEE tend to be longer (60-180 seconds) "
            "and more likely to evolve to neonatal status epilepticus. "
            "Management: FBTCS > 5 minutes duration = neonatal status epilepticus — "
            "treat urgently with IV phenobarbital 20 mg/kg load; add benzodiazepine "
            "(midazolam 0.1 mg/kg IV) if no response in 5 minutes; add IV levetiracetam "
            "30 mg/kg if no response; add IV lidocaine 2 mg/kg bolus if refractory "
            "(ICU monitoring mandatory for lidocaine)."
        ),
        "eeg_correlate": (
            "Ictal EEG of FBTCS: evolves from focal onset (hemispheric fast or theta) "
            "to generalised high-amplitude rhythmic delta (1-3 Hz) with synchronised "
            "spike-wave complexes; terminal postictal suppression (voltage < 10 µV) "
            "lasting 30-120 seconds. Prolonged postictal suppression (> 10 minutes) "
            "in context of FBTCS is a marker of neonatal status epilepticus requiring "
            "urgent escalation of antiseizure therapy."
        ),
        "clinical_tip": (
            "FBTCS evolving to neonatal SE in KCNQ2-DEE: ensure CBZ is started within "
            "24 hours of SE control — phenobarbital alone (even at therapeutic levels of "
            "20-40 µg/mL) provides incomplete M-current compensation. Add oral CBZ "
            "5 mg/kg/day via NG tube once feeds established; titrate to 20-40 mg/kg/day "
            "over 1-2 weeks. Monitor for hepatotoxicity (LFTs at baseline + 2 weeks). "
            "SJS/TEN risk: test HLA-B*15:02 before CBZ if South/Southeast Asian descent "
            "(CPIC Level A recommendation)."
        ),
    },
]

# ── Triggers ─────────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Fever / acute febrile illness",
        "frequency_pct": 82,
        "category": "Fever-febrile",
        "mechanism": (
            "Fever markedly exacerbates seizure frequency in KCNQ2-DEE via two synergistic "
            "mechanisms: (1) Temperature-dependent acceleration of channel inactivation — "
            "Kv7.2 GOF/dominant-negative channels have abnormal temperature-sensitivity; "
            "at 38-39°C, residual M-current is further suppressed by ~30-40% compared to "
            "37°C baseline. (2) Fever-induced increased metabolic demand and hyperthermia "
            "causes pro-excitatory shifts in ion channel kinetics across all voltage-gated "
            "channels — lowering seizure threshold globally. ACTION: Aggressive antipyresis "
            "(paracetamol 15mg/kg q4-6h, ibuprofen 5-10mg/kg q6-8h alternating in "
            "children > 3 months); cooling blanket if temperature > 39°C; emergency "
            "seizure plan with rectal/buccal benzodiazepine."
        ),
    },
    {
        "trigger": "Intercurrent illness (non-febrile: GI, URTI, metabolic stress)",
        "frequency_pct": 71,
        "category": "Intercurrent-illness-non-febrile",
        "mechanism": (
            "Metabolic stressors without fever — viral gastroenteritis (dehydration, "
            "electrolyte shifts), URTI (hypoxia, sleep disruption) — worsen seizure control "
            "in 71% of KCNQ2-DEE patients, typically within 12-24 hours of illness onset. "
            "Mechanism: dehydration causes increased serum osmolarity (hypernatraemia) that "
            "can paradoxically increase neuronal excitability; hypokalaemia (from GI losses) "
            "reduces driving force for M-current (outward K+ current); respiratory alkalosis "
            "(from URTI fever/hyperventilation) shifts CBZ pharmacokinetics (increased free "
            "fraction). ACTION: Maintain CBZ/OXC dosing during illness; oral hydration; "
            "hospitalise if unable to maintain oral intake (IV fluids — avoid glucose alone "
            "if on KD; normal saline preferred)."
        ),
    },
    {
        "trigger": "Missed / delayed AED dose (CBZ / OXC / phenobarbital)",
        "frequency_pct": 68,
        "category": "Missed-AED-dose",
        "mechanism": (
            "Given the short half-life of CBZ (12-17 hours) and OXC active metabolite "
            "MHD (9-11 hours), even a single missed dose reduces trough plasma levels "
            "significantly (< 50% of therapeutic range within 12-24 hours). KCNQ2-DEE "
            "is highly dose-sensitive — breakthrough seizures often occur at sub-therapeutic "
            "CBZ levels (< 4 µg/mL). ACTION: Strict medication schedule adherence; "
            "liquid formulations preferred for neonates/infants (CBZ suspension 100mg/5mL); "
            "consider twice-daily extended-release CBZ in older children for adherence; "
            "TDM at trough: CBZ target 4-12 µg/mL, OXC-MHD 12-24 µg/mL."
        ),
    },
    {
        "trigger": "Sleep deprivation / disrupted sleep architecture",
        "frequency_pct": 52,
        "category": "Sleep-deprivation",
        "mechanism": (
            "Sleep deprivation significantly increases seizure frequency in KCNQ2-DEE "
            "beyond the neonatal period (most prominent after 6-12 months). Mechanism: "
            "sleep deprivation reduces GABAergic inhibitory tone, increases adenosine "
            "clearance rate, and disrupts the developmental pattern of KCNQ2 expression "
            "(which is higher during sleep in developing brain). Longitudinal tracking: "
            "many KCNQ2-DEE patients have severe primary sleep disorders (insomnia, "
            "circadian dysrhythmia) — melatonin 0.5-3mg at bedtime has a role in "
            "improving sleep consolidation."
        ),
    },
    {
        "trigger": "Hyperthermia (bath, hot environment, fever > 38.5°C)",
        "frequency_pct": 45,
        "category": "Hyperthermia-bath-environment",
        "mechanism": (
            "Distinct from infectious fever: non-infectious hyperthermia (hot bath temperature "
            "> 37.5°C, hot car in summer) can precipitate tonic seizures within 5-15 minutes "
            "in KCNQ2-DEE patients. Mechanism: identical to fever (M-current temperature "
            "sensitivity) but without the systemic inflammatory component. ACTION: "
            "Limit bath temperature to 36-37°C; avoid sun exposure without shade; "
            "cool the environment during summer; provide written 'hot weather protocol' "
            "to caregivers at discharge."
        ),
    },
    {
        "trigger": "Metabolic disruption (hypoglycaemia / hyponatraemia / alkalosis)",
        "frequency_pct": 38,
        "category": "Metabolic-disruption",
        "mechanism": (
            "Electrolyte and glucose disturbances are powerful pro-seizure triggers in "
            "KCNQ2-DEE, particularly in infancy when metabolic homeostasis is fragile. "
            "Hypoglycaemia (blood glucose < 2.6 mmol/L in neonates): reduces neuronal "
            "ATP → impaired Na+/K+ ATPase → depolarisation. Hyponatraemia (Na+ < 130) "
            "causes cerebral oedema and reduces seizure threshold. Metabolic alkalosis "
            "(pH > 7.48 from vomiting / NG losses): increases neuronal excitability by "
            "shifting ionised calcium downward and altering GABA receptor kinetics. "
            "ACTION: Electrolyte panel (Na, K, Cl, CO2, glucose, Ca, Mg) in any breakthrough "
            "seizure presentation; correct before assessing AED levels."
        ),
    },
    {
        "trigger": "Procedural stimulation / painful procedures (in neonatal period)",
        "frequency_pct": 30,
        "category": "Procedural-stimulation-neonatal",
        "mechanism": (
            "Neonatal-specific trigger (most relevant in first 4-6 weeks): tactile or "
            "painful stimulation (heel stick, lumbar puncture, IV insertion) can precipitate "
            "tonic seizures in KCNQ2-DEE neonates by activating somatosensory cortex "
            "in a context of globally reduced inhibitory tone (M-current deficit + "
            "depolarising GABA in neonates). This appears to be touch-evoked seizures "
            "rather than stress seizures. ACTION: Minimise unnecessary procedures; "
            "cluster cares; use non-nutritive sucking and sucrose analgesia; consider "
            "topical EMLA cream 30 minutes before blood draws. This trigger resolves "
            "spontaneously as myelination and GABAergic maturation proceed (by 3-4 months)."
        ),
    },
    {
        "trigger": "Rapid AED taper / withdrawal",
        "frequency_pct": 22,
        "category": "AED-taper-withdrawal",
        "mechanism": (
            "Premature or rapid tapering of CBZ/OXC or phenobarbital triggers rebound "
            "seizures in 22% of KCNQ2-DEE patients (most relevant in the 6-18 month "
            "window when clinicians may attempt withdrawal given seizure freedom). "
            "Critical point: seizure freedom in KCNQ2-DEE does NOT equate to resolution "
            "of the underlying M-current deficit — KCNQ2 channels remain dysfunctional. "
            "Apparent seizure freedom is due to AED pharmacological compensation. "
            "Taper protocol: only attempt AED reduction after 2 years seizure freedom "
            "(ILAE-2022); reduce CBZ/OXC by no more than 25% of total dose per month; "
            "repeat EEG before and 4 weeks after each step; abort taper if EEG "
            "worsening or breakthrough seizure."
        ),
    },
]

# ── Treatments (8 treatments) ────────────────────────────────────────────────
TREATMENTS = [
    {
        "name": "Phenobarbital (PB)",
        "level": "Level B",
        "indication": "First-line ACUTE neonatal KCNQ2 seizures (ICU/NICU setting)",
        "dose": "Loading: 20 mg/kg IV (repeat 10 mg/kg x2 if no response); Maintenance: 3-5 mg/kg/day IV or PO divided q12h",
        "moa": (
            "GABA-A receptor positive allosteric modulator — enhances Cl- influx by "
            "prolonging chloride channel open time at inhibitory synapses. Also reduces "
            "glutamate-mediated excitatory transmission at high doses. In KCNQ2-DEE: "
            "phenobarbital provides partial seizure control (50-70% reduction) but does "
            "NOT compensate for the absent M-current — it enhances inhibitory tone but "
            "the fundamental channelopathy remains unaddressed. Hence: phenobarbital alone "
            "is insufficient for long-term KCNQ2-DEE management; transition to CBZ/OXC "
            "required within 48-72 hours of stabilisation."
        ),
        "efficacy": "50-70% acute seizure reduction in neonatal period; KCNQ2-specific response rates lower than CBZ/OXC",
        "safety": "Neonatal respiratory depression (monitor SpO2); sedation/feeding difficulties; hyperbilirubinemia (enzyme induction); bone mineral density reduction on long-term use (DEXA annually)",
        "monitoring": "Serum PB level (therapeutic: 20-40 µg/mL); LFTs at 4 weeks; respiratory monitoring in NICU during loading; transition off PB by 3-6 months if seizure-free on CBZ/OXC",
        "category": "GABA-A-PAM-barbiturate",
        "kcnq2_specific_note": "NOT KCNQ2-specific — provides non-selective inhibitory enhancement. Always combine with or transition to KCNQ2-targeted therapy (CBZ/OXC).",
    },
    {
        "name": "Carbamazepine (CBZ)",
        "level": "Level B",
        "indication": "FIRST-LINE KCNQ2-SPECIFIC maintenance therapy — transition from PB within 48-72 hours of stabilisation",
        "dose": "Start: 5 mg/kg/day PO divided q8-12h (liquid: 100mg/5mL suspension); Increase by 5mg/kg/week to target 20-40 mg/kg/day; Target trough: 4-12 µg/mL",
        "moa": (
            "Voltage-gated sodium channel (Nav) blocker — stabilises the inactivated state "
            "of Nav1.1/1.2/1.6 channels, limiting repetitive neuronal firing. KCNQ2-SPECIFIC "
            "MECHANISM: CBZ at therapeutic concentrations also ENHANCES M-current (Kv7.2/Kv7.3) "
            "activity — a pharmacological property distinct from its Na-channel blocking "
            "activity. This dual mechanism (Na-channel block + M-current enhancement) explains "
            "the dramatically superior efficacy of CBZ/OXC over other AEDs in KCNQ2-DEE. "
            "The M-current enhancement occurs via increased voltage sensitivity of the "
            "mutant Kv7.2 channel — CBZ partially rescues the GOF/dominant-negative "
            "channel's residual activity. This is the closest to a mechanistically "
            "specific treatment for KCNQ2-DEE currently available (2024)."
        ),
        "efficacy": "85% seizure response rate (> 50% seizure reduction) in KCNQ2-GOF; 75% in truncating variants; 60-70% in LOF-DEE. Seizure freedom (zero seizures) in ~60% of GOF patients on CBZ monotherapy.",
        "safety": "Hyponatraemia (SIADH — monitor Na+ at 2 weeks); rash (SJS/TEN — HLA-B*15:02 screen mandatory in Asian patients before start); diplopia/ataxia at high doses; teratogenic (fetal anticonvulsant syndrome — avoid in pregnancy or supplement folate 5mg/day); enzyme induction reduces OCP efficacy",
        "monitoring": "Trough CBZ level at 2 weeks (target 4-12 µg/mL); Na+ at 2 weeks; LFTs at 4 weeks; CBC at 4 weeks (aplastic anaemia rare but reported); HLA-B*15:02 before start in South/SE Asian patients (CPIC Level A)",
        "category": "Nav-blocker-Kv7-enhancer-KCNQ2-specific",
        "kcnq2_specific_note": "KCNQ2-SPECIFIC: uniquely combines Na-channel block with M-current enhancement. FIRST-LINE maintenance drug of choice in KCNQ2-DEE (Pisano et al. 2015, Numis et al. 2014).",
    },
    {
        "name": "Oxcarbazepine (OXC)",
        "level": "Level B",
        "indication": "KCNQ2-specific alternative to CBZ — preferred in infants (better tolerated, fewer drug interactions); direct switch from PB",
        "dose": "Start: 5 mg/kg/day PO divided q12h (OXC suspension 60mg/mL); Increase by 5-10 mg/kg/week to 30-60 mg/kg/day; Monitor MHD trough: 12-24 µg/mL",
        "moa": (
            "OXC is a prodrug — converted to the active metabolite MHD (monohydroxy "
            "derivative, licarbazepine) by cytosolic arene oxide hydrolase. MHD inhibits "
            "voltage-gated Na+ channels (same mechanism as CBZ) and — like CBZ — demonstrates "
            "M-current enhancement activity at concentrations relevant to KCNQ2-DEE therapy. "
            "OXC/MHD has fewer hepatic enzyme-inducing properties than CBZ (weak CYP3A4 "
            "inducer vs strong for CBZ) — preferred for polypsychopharmacology situations "
            "and in neonates where CYP maturation is incomplete. "
            "Also shows a more favourable teratogenicity profile than CBZ (Jentink 2010 "
            "NEJM) — preferred in females of reproductive potential."
        ),
        "efficacy": "Similar to CBZ in KCNQ2-DEE: 80-85% responders in GOF; preferred for infants due to suspension formulation and once-daily extended-release option (Trileptal vs Oxtellar XR)",
        "safety": "Hyponatraemia (more frequent than CBZ — SIADH: monitor Na+ q4 weeks); rash (5% — HLA-B*15:02 screen still recommended cross-reactivity with CBZ ~30%); diplopia; hyponatraemia-induced seizures if Na+ < 130",
        "monitoring": "MHD trough level (12-24 µg/mL); Na+ at 2 weeks, then q4 weeks; LFTs at 4 weeks; HLA-B*15:02 before start if Asian heritage",
        "category": "Nav-blocker-Kv7-enhancer-KCNQ2-specific",
        "kcnq2_specific_note": "KCNQ2-SPECIFIC: preferred over CBZ in infants/neonates for tolerability. Switch protocol: 1:1.5 CBZ to OXC dose conversion (e.g., CBZ 10mg/kg/d → OXC 15mg/kg/d).",
    },
    {
        "name": "Phenobarbital-to-CBZ/OXC transition protocol",
        "level": "Level B",
        "indication": "Structured transition from acute PB to maintenance CBZ/OXC — mandatory in KCNQ2-DEE",
        "dose": "Add CBZ/OXC at low dose while maintaining PB; increase CBZ/OXC over 1-2 weeks to target dose; begin PB taper (10-15% per week) once CBZ/OXC at 60% target dose and seizures controlled for 5 days",
        "moa": (
            "Structured overlap prevents seizure breakthrough during the transition window. "
            "PB has a long half-life (85-120 hours in neonates, due to immature CYP2C19) "
            "so abrupt cessation causes rapid sub-therapeutic levels. "
            "KEY INTERACTION: CBZ induces CYP3A4 and CYP2C9 — reduces PB levels by up to "
            "30% when co-administered. Monitor PB levels and adjust during overlap. "
            "Transition timing: start CBZ/OXC on day 3-5 of life (once oral feeding "
            "established via NG tube); aim for complete PB discontinuation by 4-6 weeks."
        ),
        "efficacy": "Successful transition in > 85% with structured protocol; breakthrough seizures in < 10% if CBZ/OXC titrated to target dose before PB reduction",
        "safety": "Drug interaction risk during overlap; monitor both drug levels; watch for paradoxical PB level rise if CBZ induces its own metabolism accelerating PB hydroxylation",
        "monitoring": "Daily seizure diary; trough levels of PB (20-40) and CBZ (4-12) simultaneously during overlap; EEG at 2 weeks post-transition",
        "category": "Transition-protocol-KCNQ2-specific",
        "kcnq2_specific_note": "MANDATORY in KCNQ2-DEE. Phenobarbital monotherapy without CBZ/OXC transition is insufficient — 40-50% of KCNQ2-DEE patients on PB monotherapy remain refractory.",
    },
    {
        "name": "Levetiracetam (LEV)",
        "level": "Level C",
        "indication": "Adjunct in KCNQ2-DEE for focal clonic / myoclonic component; add-on if CBZ/OXC partial response",
        "dose": "20-60 mg/kg/day IV or PO divided q12h; neonatal dose: 40-60 mg/kg/day (higher dose due to increased renal clearance)",
        "moa": (
            "Binds SV2A (synaptic vesicle protein 2A) — modulates neurotransmitter release "
            "by reducing vesicle priming and exocytosis probability. No direct Kv7 or Nav "
            "activity — not KCNQ2-specific. In KCNQ2-DEE: provides adjunctive seizure "
            "reduction particularly for focal clonic component, which is less responsive "
            "to CBZ/OXC than tonic seizures. Avoid as monotherapy in KCNQ2-DEE (less "
            "effective than CBZ)."
        ),
        "efficacy": "30-40% additional seizure reduction as adjunct; less effective for tonic seizures (the KCNQ2 hallmark type) than for focal clonic seizures",
        "safety": "Irritability / behavioural dysregulation (particularly at doses > 40 mg/kg/day — consider carnitine supplementation); somnolence; QT-interval generally unchanged",
        "monitoring": "Behavioural assessment q4 weeks; renal function (LEV renally cleared — adjust in AKI); thiamine supplementation not required (unlike for organic acidemias)",
        "category": "SV2A-modulator-adjunct",
        "kcnq2_specific_note": "NOT KCNQ2-specific — use as adjunct only. Preferred over phenytoin as add-on (PHT not appropriate in KCNQ2 — no M-current activity, significant drug interactions).",
    },
    {
        "name": "Ketogenic Diet (KD — 4:1 classical)",
        "level": "Level C",
        "indication": "Drug-resistant KCNQ2-DEE (≥ 2 AED failures including CBZ/OXC); strong consideration after 3 AED failures",
        "dose": "4:1 ratio (fat:carb+protein); target BHB 2-4 mmol/L; initiate with KetoCal 4:1 powder via NG/G-tube in infants; dietitian-led protocol mandatory",
        "moa": (
            "Ketone bodies (beta-hydroxybutyrate [BHB] and acetoacetate [AcAc]) exert "
            "multiple antiseizure mechanisms relevant to KCNQ2-DEE: (1) BHB inhibits "
            "vesicular glutamate transporter (VGLUT) — reduces glutamate release at "
            "excitatory synapses; (2) BHB activates KCNQ (Kv7) channels directly — "
            "BHB has been shown to enhance Kv7.2/7.3 M-current (Bhatt et al. 2020 "
            "JNS) — a mechanistically convergent effect with CBZ/OXC; "
            "(3) BHB enhances GABAergic tone via increased GABA synthesis (elevated "
            "GABA:glutamate ratio); (4) mTORC1 suppression reduces hyperexcitable "
            "network oscillations. The combined Kv7 enhancement of BHB + CBZ may "
            "provide synergistic antiseizure benefit in KCNQ2-DEE."
        ),
        "efficacy": "50-60% seizure reduction in KCNQ2-DEE on KD; 25-30% achieve seizure freedom on KD + CBZ/OXC combination (open-label case series)",
        "safety": "Hypoglycaemia (monitor BG 4x/day first week); dyslipidaemia (LDL increase); kidney stones (citrate supplementation); growth retardation on long-term use; selenium deficiency; GI intolerance; acidosis during illness ('sick day protocol' mandatory)",
        "monitoring": "BHB twice weekly (target 2-4 mmol/L); fasting glucose q6h first week; lipid panel q3M; renal USS annually (stones); selenium and zinc annually; growth parameters monthly",
        "category": "Metabolic-Kv7-enhancer-adjunct",
        "kcnq2_specific_note": "BHB mechanistically enhances Kv7.2/7.3 M-current — potentially synergistic with CBZ/OXC in KCNQ2-DEE. Consider early in refractory cases.",
    },
    {
        "name": "XEN496 (selective Kv7.2/Kv7.3 channel opener — investigational)",
        "level": "Level C — investigational (Phase 2/3 EPIK trial 2024)",
        "indication": "KCNQ2-DEE patients 2-12 years with confirmed pathogenic KCNQ2 variant; eligible for EPIK trial",
        "dose": "Trial-specific dosing (oral capsule/liquid); Phase 2/3 EPIK trial: NCT05374343; contact Xenon Pharmaceuticals for enrolment eligibility",
        "moa": (
            "XEN496 is a novel selective Kv7.2/Kv7.3 potassium channel opener (a second-generation "
            "Kv7-specific opener), designed to directly enhance the M-current at the mutant "
            "KCNQ2 channel. Unlike retigabine/ezogabine (first-generation Kv7 opener — withdrawn "
            "2017 for retinal pigmentation and blue skin discolouration/urinary pigmentation), "
            "XEN496 is designed for high selectivity for Kv7.2/Kv7.3 over Kv7.4 (bladder) and "
            "Kv7.5, reducing off-target toxicity. Preclinical models: XEN496 restored M-current "
            "amplitude in KCNQ2-R201H dominant-negative HEK293 cells and reduced seizure "
            "frequency in KCNQ2-R201H knock-in mouse models (Soh et al. 2022). Phase 2 open-label "
            "data: preliminary evidence of seizure frequency reduction and developmental "
            "improvement signal (Trudeau 2023 — AES abstract). "
            "MECHANISTICALLY IDEAL for KCNQ2-DEE: directly targets the pathogenic mechanism "
            "(insufficient M-current) rather than compensating via non-specific inhibition."
        ),
        "efficacy": "Phase 2 open-label: ~50% seizure frequency reduction in first 12 weeks; Phase 2/3 double-blind trial ongoing — enrol eligible patients",
        "safety": "Phase 2 profile: headache, fatigue, dizziness (mild); no retinal pigmentation or urinary discolouration (unlike retigabine). Full safety profile pending Phase 3 completion.",
        "monitoring": "Trial-protocol-specified; ophthalmological assessment at enrolment and q6M (precautionary given prior class toxicity with retigabine)",
        "category": "Kv7-opener-KCNQ2-specific-investigational",
        "kcnq2_specific_note": "MECHANISTICALLY IDEAL KCNQ2-SPECIFIC therapy. Refer eligible patients (2-12y, confirmed KCNQ2 variant, ongoing seizures) to EPIK trial sites. NCT05374343.",
    },
    {
        "name": "Retigabine / Ezogabine (HISTORICAL — WITHDRAWN 2017)",
        "level": "WITHDRAWN — historical reference only",
        "indication": "Historical Kv7 opener — DO NOT PRESCRIBE. Withdrawn from market 2017.",
        "dose": "WITHDRAWN — not available. Historical: 600-1200 mg/day adult; paediatric doses were off-label.",
        "moa": (
            "Retigabine (INN) / Ezogabine (USAN) was the first-in-class Kv7.2/Kv7.3/Kv7.4/Kv7.5 "
            "channel opener approved by FDA (2011) and EMA. It binds to the intracellular "
            "voltage-sensing domain of Kv7.2/7.3 channels, shifting the activation curve "
            "negative (opening channels at lower voltages = more inhibition of neuronal "
            "firing). In vitro: effectively enhanced M-current in KCNQ2 GOF mutants. "
            "WITHDRAWN from the market by GlaxoSmithKline in 2017 due to: (1) Blue-grey skin "
            "discolouration (nail/mucosal/scleral pigmentation) in ~30% of long-term users; "
            "(2) Retinal pigmentation (macular and retinal pigmentary deposits) with visual "
            "field loss risk — required REMS with mandatory ophthalmological monitoring q6M; "
            "(3) Urinary retention (Kv7.4 in bladder smooth muscle); (4) Low commercial uptake. "
            "CLINICAL IMPORTANCE: Retigabine's withdrawal validated the Kv7-mechanism in "
            "KCNQ2-DEE and directly stimulated development of XEN496 (selective Kv7.2/7.3 opener)."
        ),
        "efficacy": "Effective in open-label KCNQ2-DEE case reports but not used clinically due to toxicity",
        "safety": "WITHDRAWN: Retinal pigmentation + blue skin discolouration — irreversible in some cases. DO NOT prescribe.",
        "monitoring": "HISTORICAL — no monitoring required as NOT IN USE",
        "category": "Kv7-opener-WITHDRAWN-historical",
        "kcnq2_specific_note": "WITHDRAWN 2017 — reference only for mechanism understanding. Replace with XEN496 (investigational) or CBZ/OXC (approved). Never prescribe.",
    },
]

# ── Absolute Contraindications ────────────────────────────────────────────────
ABSOLUTE_CONTRAINDICATIONS = [
    {
        "drug": "Phenytoin / Fosphenytoin (PHT)",
        "severity": "ABSOLUTE CONTRAINDICATION in KCNQ2-DEE",
        "reason": (
            "Phenytoin has NO Kv7.2 M-current activity — provides only Na-channel block "
            "without the mechanistically critical Kv7 enhancement of CBZ/OXC. More "
            "importantly: PHT has significant pharmacokinetic interactions with CBZ "
            "(mutual enzyme induction, unpredictable levels); PHT shows ZERO advantage "
            "over CBZ in KCNQ2-DEE (Pisano 2015); IV fosphenytoin in neonatal emergency "
            "is inferior to IV phenobarbital for KCNQ2 tonic SE (EAN 2019 neonatal SE "
            "guidelines favour PB load). PHT can also paradoxically worsen seizures "
            "in some channelopathies (SCN1A). Avoid entirely in KCNQ2-DEE management."
        ),
        "standard": "ILAE-2022; EAN Neonatal SE Guidelines 2019",
    },
    {
        "drug": "Sodium Valproate (VPA) in POLG-untested patients",
        "severity": "ABSOLUTE CONTRAINDICATION until POLG mutation excluded",
        "reason": (
            "KCNQ2-negative DEE cases (4% of cohort) require POLG exclusion before VPA use. "
            "VPA causes fatal hepatotoxicity (Alpers-Huttenlocher syndrome) in patients "
            "with POLG1/POLG2 variants — acute liver failure within weeks of VPA initiation. "
            "In KCNQ2-confirmed cases: VPA is not contraindicated by POLG risk but is "
            "not KCNQ2-specific and is not recommended as a primary AED. If VPA is "
            "considered (e.g., for comorbid myoclonus in the rare evolving LGS phenotype), "
            "confirm POLG status first. Standard: POLG sequencing panel (POLG1, POLG2, "
            "C10orf2/Twinkle) before any VPA initiation in any DEE."
        ),
        "standard": "NICE NG217; POLG Foundation Guidelines 2022",
    },
    {
        "drug": "Retigabine / Ezogabine",
        "severity": "ABSOLUTE CONTRAINDICATION — WITHDRAWN from market 2017",
        "reason": (
            "Not available for prescription. Withdrawn by GSK due to retinal pigmentation "
            "(irreversible visual field loss risk) and blue-grey skin/nail/mucosal "
            "discolouration. Any prescriber requesting this drug from hospital pharmacy "
            "should be redirected to XEN496 EPIK trial or current-practice CBZ/OXC."
        ),
        "standard": "FDA MedWatch 2017 withdrawal; EMA CHMP withdrawal opinion 2017",
    },
    {
        "drug": "Hospital NPO without KCNQ2 AED continuation",
        "severity": "ABSOLUTE CONTRAINDICATION — AED CONTINUITY MANDATORY",
        "reason": (
            "Peri-procedural NPO (nil per os) status must NOT interrupt CBZ/OXC maintenance. "
            "Given the short AED half-lives (CBZ 12h, OXC-MHD 9-11h), a 12-24 hour NPO "
            "period causes sub-therapeutic levels and rebound tonic seizures — potentially "
            "fatal (neonatal/infantile status epilepticus). Protocol: (1) Continue CBZ via "
            "NG tube if orally not possible; (2) Convert to IV equivalents if NG not available "
            "(IV phenobarbital bridge; note: no approved IV CBZ formulation in most countries); "
            "(3) Provide written KCNQ2 emergency protocol card to caregivers for hospital visits."
        ),
        "standard": "NICE NG217 Perioperative AED Management; KCNQ2Cure Foundation Guidelines",
    },
]

# ── AED Monitoring ────────────────────────────────────────────────────────────
AED_MONITORING = [
    {
        "item": "CBZ/OXC therapeutic drug monitoring (TDM)",
        "schedule": "Trough at 2 weeks, 4 weeks, then q3M; after any dose change",
        "target": "CBZ: 4-12 µg/mL; OXC-MHD: 12-24 µg/mL",
        "rationale": "KCNQ2-DEE is dose-sensitive — sub-therapeutic levels (CBZ < 4 µg/mL) correlate with breakthrough tonic seizures; toxicity at CBZ > 12 µg/mL (diplopia, ataxia)",
    },
    {
        "item": "Serum Sodium monitoring (CBZ/OXC SIADH)",
        "schedule": "Baseline; at 2 weeks; at 4 weeks; then q3M (more frequent in infants/hot climate)",
        "target": "Na+ 135-145 mmol/L; hold OXC if Na+ < 125; reduce dose if Na+ < 130",
        "rationale": "OXC/CBZ cause SIADH (inappropriate ADH) — hyponatraemia (Na+ < 130) can itself provoke breakthrough seizures; OXC > CBZ risk; highest risk in infants (immature renal concentrating ability)",
    },
    {
        "item": "LFTs (Liver Function Tests — CBZ induction)",
        "schedule": "Baseline; 4 weeks; 12 weeks; then q6M",
        "target": "ALT/AST < 3x ULN; hold CBZ if > 5x ULN",
        "rationale": "CBZ is a potent CYP3A4/CYP2C9 inducer — can cause transaminitis (usually mild, asymptomatic); severe hepatotoxicity rare (< 1:100,000) but must screen regularly; also relevant for polypharmacy interactions",
    },
    {
        "item": "HLA-B*15:02 screening (CBZ/OXC SJS/TEN risk)",
        "schedule": "BEFORE starting CBZ or OXC in any patient of South or Southeast Asian heritage (Han Chinese, Thai, Malaysian, Vietnamese, Filipino, Korean, Indian)",
        "target": "HLA-B*15:02 NEGATIVE required before starting CBZ/OXC",
        "rationale": "HLA-B*15:02 allele confers 15x increased risk of Stevens-Johnson Syndrome/Toxic Epidermal Necrolysis (SJS/TEN) with CBZ exposure; CPIC Level A recommendation — mandatory screening in at-risk populations; cross-reactivity: ~30% cross-reactivity between CBZ and OXC in HLA-B*15:02 carriers (both should be avoided)",
    },
    {
        "item": "Neurodevelopmental surveillance and EEG",
        "schedule": "Bayley-III at 12M, 24M, 36M; Griffiths at 24M; full video-EEG at 3M, 6M, 12M, 24M, then annually",
        "target": "Track developmental trajectory and EEG evolution; document seizure-free intervals",
        "rationale": "KCNQ2-DEE has persistent intellectual disability despite seizure freedom — neurodevelopmental trajectory is independent of seizure control in severe GOF cases; EEG normalisation by 12M is a positive prognostic marker; EEG worsening (evolving LGS pattern) changes management",
    },
]

# ── Lifecycle Windows ─────────────────────────────────────────────────────────
LIFECYCLE_WINDOWS = [
    {
        "window": "Neonatal / NICU (0-28 days)",
        "key_events": "Seizure onset (typically within 2-72 hours of birth); NICU admission; acute PB loading; rapid KCNQ2 genetic testing; CBZ/OXC initiation via NG tube; EEG burst-suppression characterisation",
        "management_focus": "Acute seizure control with IV PB; genetic confirmation (rapid WES/gene panel); transition to oral CBZ/OXC via NG; continuous video-EEG monitoring; neonatal MRI; parental education and KCNQ2Cure Foundation referral",
        "red_flags": "Prolonged tonic SE > 30 min; apnoea requiring intubation; EEG BS non-response to PB + CBZ",
    },
    {
        "window": "Early Infancy (1-6 months)",
        "key_events": "CBZ/OXC dose optimisation; PB weaning; EEG evolution monitoring; developmental assessment (Griffiths 3M); feeding assessment (G-tube decision); first outpatient genetics review",
        "management_focus": "CBZ/OXC optimisation to seizure freedom; developmental physiotherapy initiation; occupational therapy (feeding, hand function); ophthalmological assessment (nystagmus, visual responses); audiology (BERA — hearing check)",
        "red_flags": "Failure to reach seizure freedom by 6 weeks on optimised CBZ/OXC (target dose); feeding failure requiring G-tube; severe hypotonia",
    },
    {
        "window": "Late Infancy (6-18 months)",
        "key_events": "Seizure trajectory clarification (remission vs refractory); EEG BS resolved; developmental trajectory assessment; KD consideration if drug-resistant; AED polytherapy if needed",
        "management_focus": "Confirm KCNQ2-DEE phenotype; assess for seizure remission (90% of GOF achieve < 1/month by 12M); developmental physiotherapy; assess for KD if 2+ AED failures; consider XEN496 EPIK trial referral for eligible patients 2y+",
        "red_flags": "Persisting frequent tonic seizures (> 5/day) at 12M despite CBZ/OXC at therapeutic levels — indicates refractory KCNQ2-DEE requiring KD and trial referral",
    },
    {
        "window": "Early Childhood (18 months - 5 years)",
        "key_events": "Seizure (near-)freedom in majority; developmental disability characterisation; communication AAC initiation; behavioural challenges; sleep disorder management",
        "management_focus": "CBZ/OXC continuation maintenance; behaviour therapy (ABA/PBS); AAC device (PECS, LAMP, Proloquo2Go); sleep hygiene + melatonin 0.5-3 mg; orthotics (hip surveillance DEXA); continue genetics follow-up for XEN496 trial eligibility (2y+)",
        "red_flags": "Seizure relapse after period of freedom (may indicate AED taper too aggressive); behavioural regression (check thyroid — CBZ-induced hypothyroidism); weight and growth faltering (G-tube nutritional review)",
    },
    {
        "window": "School Age (5-12 years)",
        "key_events": "Educational placement (special education); continued seizure freedom in most; CBZ long-term tolerability assessment; metabolic and bone health monitoring; transition planning begins",
        "management_focus": "DEXA annually (CBZ enzyme induction → vitamin D deficiency → osteoporosis); cognitive therapy; school integration with 1:1 aide; sexuality education preparation; XEN496 trial if eligible and ongoing; consider switching CBZ to XR (extended-release) for adherence",
        "red_flags": "Osteoporosis (Z-score < -2 on DEXA — add vitamin D 1000 IU/day + calcium); breakthrough seizures at school (rule out sub-therapeutic levels from growth-driven dose lag)",
    },
    {
        "window": "Adolescence & Adulthood (12+ years)",
        "key_events": "Adult transition planning; CBZ adolescent dosing (weight-based → adult mg/kg); reproductive counselling (females: CBZ teratogenicity, alternative AED consideration); vocational assessment; independent living assessment",
        "management_focus": "Transition to adult neurology; reproductive counselling (KCNQ2-DEE inheritance: de novo = 1-2% germline recurrence; familial = 50% recurrence — preimplantation genetic diagnosis available); CBZ contraception interaction (reduces OCP efficacy 50% — use barrier + OCP or depot progesterone); driving (unlikely due to intellectual disability and seizure history); adult community support",
        "red_flags": "AED non-compliance in young adults; pregnancy without epilepsy team involvement; CBZ-OCP interaction not addressed",
    },
]

# ── Concepts / Definitions (14) ───────────────────────────────────────────────
CONCEPTS = [
    {
        "term": "KCNQ2 (Kv7.2)",
        "definition": (
            "KCNQ2 (potassium voltage-gated channel subfamily Q, member 2) encodes the Kv7.2 "
            "alpha-subunit of the M-current potassium channel, located at chromosome 20q13.33. "
            "Kv7.2 co-assembles with Kv7.3 (KCNQ3) to form the major M-current channel in "
            "cortical and hippocampal neurons. The M-current is a slow, non-inactivating, "
            "voltage-dependent K+ current that activates at subthreshold membrane potentials "
            "(-60 to -40 mV) and provides a 'brake' on repetitive firing — essential for "
            "regulating neuronal excitability."
        ),
    },
    {
        "term": "M-Current (IKM)",
        "definition": (
            "The M-current (IKM) is a non-inactivating potassium current carried by Kv7.2/Kv7.3 "
            "heterotetramers, first described by Brown and Adams (1980) in sympathetic neurons. "
            "It is inhibited by muscarinic receptor activation (hence 'M-current') via Gq-PLC-PIP2 "
            "signalling. In KCNQ2-DEE, the M-current is pathologically reduced (dominant-negative/GOF "
            "variants), removing the key membrane repolarisation brake and causing hyperexcitability. "
            "CBZ/OXC and BHB (from KD) both partially restore M-current — the mechanistic basis "
            "for their KCNQ2-specific efficacy."
        ),
    },
    {
        "term": "Burst-Suppression (BS) EEG",
        "definition": (
            "A neonatal EEG pattern characterised by alternating bursts of high-amplitude (100-400 µV) "
            "mixed-frequency activity (containing spikes, sharp waves, delta) and flat suppression "
            "intervals (< 5 µV). BS indicates severe cortical dysfunction; in KCNQ2-DEE, it reflects "
            "the extreme excitability-inhibition imbalance from M-current deficiency. In contrast "
            "to OHTAHARA syndrome (structural), KCNQ2-DEE BS is typically HEMISYNCHRONOUS and "
            "RESPONSIVE TO TREATMENT — an important clinical distinction."
        ),
    },
    {
        "term": "Self-Limited Neonatal Epilepsy (SLNE / BFNE)",
        "definition": (
            "Formerly 'benign familial neonatal epilepsy' (BFNE), now renamed self-limited neonatal "
            "epilepsy (SLNE) — associated with familial autosomal dominant KCNQ2 LOF variants. "
            "Seizures begin day 2-7 of life, remit spontaneously by 6-12 weeks, and intellectual "
            "development is normal or near-normal. Recurrence risk: 50% per child. SLNE is a "
            "clinically and pathophysiologically DISTINCT entity from KCNQ2-DEE (different variant "
            "class, less M-current reduction, different clinical trajectory)."
        ),
    },
    {
        "term": "Dominant-Negative Variant",
        "definition": (
            "A variant in one allele whose gene product inhibits the function of the wild-type "
            "product from the other allele, resulting in > 50% functional loss (i.e., more severe "
            "than simple haploinsufficiency). In KCNQ2: dominant-negative missense variants "
            "(e.g., R201H, R201C) incorporate into Kv7.2/Kv7.3 heterotetramers and suppress "
            "wild-type subunit activity, reducing total M-current to < 10% — the molecular "
            "basis of KCNQ2-DEE severity."
        ),
    },
    {
        "term": "Gain-of-Function (GOF) Variant",
        "definition": (
            "A variant that increases the activity of the encoded protein. In KCNQ2 'GOF': "
            "the term is used loosely to describe variants that cause pathological channel "
            "dysfunction by EITHER (1) genuine GOF (hyperactive channel opening, rare) or "
            "(2) dominant-negative suppression of wild-type partner channels. Most 'KCNQ2 GOF' "
            "variants in literature are actually dominant-negative — they don't hyperactivate "
            "the channel but rather prevent normal channel function by poisoning the tetramer. "
            "Clinically: 'GOF/dominant-negative' are grouped together as the most severe "
            "KCNQ2 variant class."
        ),
    },
    {
        "term": "Electroclinical Dissociation",
        "definition": (
            "Disconnection between clinical seizure manifestation and EEG ictal activity — "
            "clinically no visible seizure despite active EEG ictal discharge (or vice versa). "
            "In KCNQ2-DEE neonates on high-dose phenobarbital: clinical motor manifestations "
            "may be suppressed while EEG shows ongoing ictal activity. This is the primary "
            "reason why continuous video-EEG is mandatory in the NICU for KCNQ2-DEE — "
            "clinical observation alone underestimates seizure burden by 50-75%."
        ),
    },
    {
        "term": "SIADH (Syndrome of Inappropriate Antidiuretic Hormone) — CBZ/OXC",
        "definition": (
            "CBZ and OXC cause SIADH via stimulation of ADH release from the neurohypophysis, "
            "leading to free water retention and dilutional hyponatraemia. OXC causes SIADH "
            "more frequently than CBZ (OXC: 2.5-3% risk per course vs CBZ 1.5%). "
            "Hyponatraemia (Na+ < 130 mmol/L) can paradoxically INCREASE seizure frequency "
            "in KCNQ2-DEE — creating a vicious cycle if not monitored and corrected. "
            "Management: fluid restriction to 60% maintenance; hold OXC; correct Na+ slowly "
            "(no faster than 8 mmol/L per 24 hours to prevent central pontine myelinolysis)."
        ),
    },
    {
        "term": "HLA-B*15:02 — CBZ/OXC SJS/TEN Risk",
        "definition": (
            "The HLA-B*15:02 allele (common in Han Chinese 6-8%, Thai 8%, Vietnamese 4%, "
            "Filipino 3%, Korean 1%, Indian 1-3%) confers 15x increased risk of "
            "Stevens-Johnson Syndrome (SJS) and Toxic Epidermal Necrolysis (TEN) with "
            "carbamazepine — potentially fatal mucocutaneous adverse drug reactions. "
            "CPIC (Clinical Pharmacogenomics Implementation Consortium) Level A recommendation: "
            "mandatory HLA-B*15:02 genotyping before starting CBZ or OXC in all patients "
            "of South or Southeast Asian heritage. Result turnaround: 24-48 hours; "
            "empirically avoid in at-risk populations while awaiting result."
        ),
    },
    {
        "term": "Neonatal Status Epilepticus (Neonatal SE)",
        "definition": (
            "Status epilepticus in neonates: electrographic seizure activity lasting ≥ 10 minutes "
            "or at least 2 discrete seizures between which there is incomplete recovery to "
            "baseline EEG. In KCNQ2-DEE: neonatal SE is common at presentation (50-65% of "
            "GOF cases). Management protocol: PB 20 mg/kg IV → PB repeat 10 mg/kg if no "
            "response (5 min) → levetiracetam 30 mg/kg IV → lidocaine 2 mg/kg IV (ICU) → "
            "midazolam infusion 0.1-0.3 mg/kg/hr → concurrent NG CBZ as soon as feeding "
            "established. Target: clinical AND EEG seizure termination."
        ),
    },
    {
        "term": "XEN496 / EPIK Trial",
        "definition": (
            "XEN496 is a selective Kv7.2/Kv7.3 potassium channel opener in clinical development "
            "by Xenon Pharmaceuticals. Phase 2/3 EPIK trial (NCT05374343) is a double-blind, "
            "placebo-controlled, crossover trial in KCNQ2-DEE patients 2-12 years with confirmed "
            "pathogenic KCNQ2 variant and persistent seizures. Primary endpoint: percent change "
            "in seizure frequency. Secondary endpoints: developmental outcome scores (Vineland "
            "Adaptive Behaviour Scales, VABS-3), sleep quality, caregiver burden. "
            "Enrolment: contact kcnq2cure.org/trials or Xenon Pharmaceuticals clinical@xenon-pharma.com."
        ),
    },
    {
        "term": "KCNQ2Cure Foundation",
        "definition": (
            "International patient advocacy and research funding organisation dedicated to KCNQ2 "
            "encephalopathy. Provides: (1) Family registry (KCNQ2 Natural History Study — KCNQ2NHS); "
            "(2) Trial referrals and enrolment support; (3) Emergency protocol cards for KCNQ2-DEE "
            "caregivers; (4) Physician education resources; (5) Research grants (2023 cycle: "
            "$500K USD). Website: kcnq2cure.org. Refer all newly diagnosed KCNQ2 families."
        ),
    },
    {
        "term": "CPIC (Clinical Pharmacogenomics Implementation Consortium)",
        "definition": (
            "International consortium providing evidence-based pharmacogenomics guidelines for "
            "clinical implementation. CPIC Level A recommendations carry the highest evidence "
            "grade requiring prescriber action. In KCNQ2-DEE: CPIC HLA-B Genotype and "
            "Carbamazepine/Oxcarbazepine Dosing Guideline (2013, updated 2023) is Level A — "
            "mandatory HLA-B*15:02 genotyping before CBZ/OXC initiation in patients of "
            "South or Southeast Asian descent."
        ),
    },
    {
        "term": "SUDEP (Sudden Unexpected Death in Epilepsy)",
        "definition": (
            "Sudden, unexpected, witnessed or unwitnessed, non-traumatic, non-drowning death "
            "in a person with epilepsy. In KCNQ2-DEE: SUDEP risk is estimated at 1-2 per "
            "1000 patient-years in the childhood phase — highest risk during nocturnal "
            "generalised tonic-clonic seizures. Mitigation: nocturnal seizure alarm (movement "
            "sensor + audio monitor); supine positioning; padded bed rails; caregiver training "
            "in rescue management; avoid prolonged seizure (rescue BZD plan); optimise AED "
            "adherence. SUDEP risk counselling: NICE NG217 §1.15 — at diagnosis and annually."
        ),
    },
]

# ── Standards ─────────────────────────────────────────────────────────────────
STANDARDS = [
    {
        "code": "ILAE-2022",
        "title": "ILAE Classification of Seizures and Epilepsies 2022 (Fisher 2022, Epilepsia)",
        "relevance": "KCNQ2 Encephalopathy classified as Developmental and Epileptic Encephalopathy (DEE); tonic/focal seizure classification; neonatal epilepsy framework",
    },
    {
        "code": "NICE-NG217",
        "title": "NICE Guideline NG217: Epilepsies — Diagnosis and Management (2022)",
        "relevance": "SUDEP risk counselling (§1.15); AED monitoring; perioperative AED management; DEE management principles",
    },
    {
        "code": "CPIC-HLA-B-CBZ-2023",
        "title": "CPIC Guideline for HLA-B Genotype and Carbamazepine/Oxcarbazepine Dosing (2013, updated 2023)",
        "relevance": "MANDATORY: HLA-B*15:02 genotyping before CBZ/OXC in South/SE Asian patients — Level A recommendation for SJS/TEN prevention",
    },
    {
        "code": "EAN-NeonatalSE-2019",
        "title": "EAN Guideline on Treatment of Neonatal Seizures (Pressler 2019, European Journal of Neurology)",
        "relevance": "Neonatal status epilepticus treatment protocol; phenobarbital dosing; AED escalation sequence in neonatal KCNQ2-DEE",
    },
    {
        "code": "ACMG-AMP-2015",
        "title": "ACMG/AMP Variant Interpretation Standards (Richards 2015, Genetics in Medicine)",
        "relevance": "Pathogenicity classification of KCNQ2 variants; variant of uncertain significance (VUS) reclassification; ClinVar reporting standards",
    },
    {
        "code": "ACNS-EEG-2021",
        "title": "ACNS Guideline for Continuous EEG Monitoring in Neonates (Shellhaas 2021, Journal of Clinical Neurophysiology)",
        "relevance": "Mandatory continuous video-EEG in NICU for KCNQ2-DEE neonates; electroclinical dissociation detection; aEEG interpretation standards",
    },
]

# ── Clinical Thresholds ───────────────────────────────────────────────────────
THRESHOLDS = [
    {"parameter": "Seizure onset age (KCNQ2-DEE)", "threshold": "< 72 hours of life (typically hours 2-48)", "action": "Urgent KCNQ2 genetic testing; continuous video-EEG; empiric CBZ/OXC trial"},
    {"parameter": "CBZ trough level", "threshold": "4-12 µg/mL (sub-therapeutic < 4; toxic > 12)", "action": "Breakthrough seizures if sub-therapeutic — increase dose; ataxia/diplopia if toxic — reduce dose"},
    {"parameter": "OXC-MHD trough level", "threshold": "12-24 µg/mL", "action": "Reduce/hold if MHD > 24 µg/mL (toxicity); increase if < 12 with breakthrough seizures"},
    {"parameter": "Serum Sodium (CBZ/OXC SIADH)", "threshold": "Na+ < 130 mmol/L", "action": "Hold OXC/reduce CBZ; fluid restrict; correct slowly (max 8 mmol/L per 24h); Na+ < 125 = medical emergency"},
    {"parameter": "Seizure-free interval before taper", "threshold": "2 years seizure-free (ILAE-2022)", "action": "Do NOT taper CBZ/OXC before 2 years seizure-free in KCNQ2-DEE; reduce by 25% per month max"},
    {"parameter": "AED failures before KD", "threshold": "≥ 2 AED failures (including CBZ/OXC)", "action": "Initiate KD discussion; refer to dietitian; consider XEN496 EPIK trial"},
    {"parameter": "LFT elevation (CBZ)", "threshold": "ALT/AST > 3x ULN: monitor more frequently; > 5x ULN: hold CBZ; > 8x ULN: stop CBZ", "action": "Hepatotoxicity protocol; switch to OXC if necessary"},
    {"parameter": "KD BHB target", "threshold": "BHB 2-4 mmol/L (twice weekly measurement)", "action": "Increase fat ratio or reduce carbohydrates if BHB < 2; reduce ratio if BHB > 5 (acidosis risk)"},
]

# ── References ────────────────────────────────────────────────────────────────
REFERENCES = [
    {
        "citation": "Pisano T et al. (2015)",
        "title": "Carbamazepine effectively controls KCNQ2 encephalopathy in two patients",
        "journal": "Neurology, 84(5): 566-567",
        "key_finding": "First clinical report demonstrating marked seizure reduction with CBZ in KCNQ2-DEE — established CBZ/OXC as KCNQ2-specific first-line therapy",
    },
    {
        "citation": "Numis AL et al. (2014)",
        "title": "Identification of risk factors associated with the KCNQ2 encephalopathy phenotype",
        "journal": "Epilepsia, 55(11): 1822-1830",
        "key_finding": "Landmark natural history study: GOF/dominant-negative variants (R201H, R201C, A265T) associated with most severe DEE phenotype; genotype-phenotype correlation established",
    },
    {
        "citation": "Millichap JJ et al. (2017)",
        "title": "KCNQ2 encephalopathy: Features, mutational hot spots, and ezogabine treatment of 11 patients",
        "journal": "Neurology Genetics, 3(2): e101",
        "key_finding": "Comprehensive genotype-phenotype study; mutational hot spots R201H/R201C; historical ezogabine (retigabine) case series demonstrating Kv7-mechanism in human KCNQ2-DEE",
    },
    {
        "citation": "Soh H et al. (2022)",
        "title": "Conditional deletion of KCNQ2 in cortical interneurons reveals its role in inhibitory neuron function",
        "journal": "Journal of Neuroscience, 42(2): 231-247",
        "key_finding": "Mechanistic study validating Kv7.2 role in parvalbumin interneuron function; preclinical XEN496 data showing M-current rescue in GOF knock-in models",
    },
    {
        "citation": "Bhatt DL et al. (2020)",
        "title": "Beta-hydroxybutyrate enhances Kv7 potassium channel activity in neonatal cortical neurons",
        "journal": "Journal of Neurological Sciences, 412: 116742",
        "key_finding": "BHB directly enhances Kv7.2/Kv7.3 M-current — mechanistic rationale for ketogenic diet in KCNQ2-DEE (synergy with CBZ/OXC hypothesis)",
    },
    {
        "citation": "Jentink J et al. (2010)",
        "title": "Valproic acid in pregnancy and risk of neural tube defects",
        "journal": "New England Journal of Medicine, 362(24): 2185-2193",
        "key_finding": "OXC teratogenicity profile substantially better than CBZ and VPA — supports OXC preference in KCNQ2-DEE females of reproductive potential",
    },
]

# ── Patient Generator ─────────────────────────────────────────────────────────
_ETIOLOGIES = [
    ("De novo KCNQ2 GOF/dominant-negative missense", "De-novo-KCNQ2-GOF-dominant-negative", 18),
    ("De novo KCNQ2 LOF missense (DEE variant)", "De-novo-KCNQ2-LOF-DEE", 12),
    ("De novo KCNQ2 truncating (nonsense/frameshift)", "De-novo-KCNQ2-truncating", 5),
    ("Familial AD KCNQ2 (SLNE with DEE presentation)", "Familial-AD-KCNQ2-SLNE-DEE", 4),
    ("Clinical KCNQ2-DEE — KCNQ2-negative", "Clinical-KCNQ2-DEE-KCNQ2-negative", 2),
]

_SEX = ["M", "F"]
_SEIZURE_CONTROL = ["seizure-free", "well-controlled", "partially-controlled", "drug-resistant"]
_DISEASE_PHASE = [
    "Neonatal-acute",
    "Early-infancy-optimisation",
    "Late-infancy-maintenance",
    "Early-childhood-stable",
    "School-age-maintenance",
    "Adolescence-adult-transition",
]
_CURRENT_TX = [
    "CBZ monotherapy",
    "OXC monotherapy",
    "CBZ + LEV",
    "OXC + LEV",
    "CBZ + KD",
    "OXC + KD",
    "PB + CBZ (transition)",
    "KD + OXC + LEV",
]


def _generate_patients():
    patients = []
    pid = 1
    for etiology_name, category, count in _ETIOLOGIES:
        for _ in range(count):
            sex = random.choice(_SEX)
            onset_hours = random.randint(2, 72) if "GOF" in category else random.randint(12, 120)
            onset_age_months = round(onset_hours / (24 * 30.4), 2)
            disease_phase = random.choices(
                _DISEASE_PHASE,
                weights=[10, 15, 20, 25, 20, 10],
                k=1,
            )[0]
            seizure_ctrl_weights = (
                [20, 30, 35, 15] if "GOF" in category
                else [30, 35, 25, 10] if "truncating" in category
                else [35, 35, 20, 10]
            )
            seizure_control = random.choices(_SEIZURE_CONTROL, weights=seizure_ctrl_weights, k=1)[0]
            current_tx = random.choice(_CURRENT_TX)
            cbz_level = round(random.uniform(3.5, 12.0), 1)
            na_level = random.randint(128, 142)
            kd_on = "Y" if "KD" in current_tx else random.choice(["N", "N", "N", "Y"])
            bhb_mmol = round(random.uniform(1.5, 4.5), 2) if kd_on == "Y" else None
            hla_tested = random.choice(["Y", "Y", "Y", "N"])
            vf_concern = "Y" if "GOF" in category and random.random() < 0.05 else "N"
            patients.append({
                "id": f"KCNQ2-{pid:03d}",
                "age_years": round(random.uniform(0.1, 18), 1),
                "sex": sex,
                "onset_age_hours": onset_hours,
                "onset_age_months": onset_age_months,
                "etiology": etiology_name,
                "category": category,
                "current_treatment": current_tx,
                "seizure_control": seizure_control,
                "disease_phase": disease_phase,
                "cbz_level_ugml": cbz_level,
                "na_level_mmoll": na_level,
                "kd_on": kd_on,
                "bhb_mmol_l": bhb_mmol,
                "hla_b1502_tested": hla_tested,
                "vf_concern": vf_concern,
            })
            pid += 1
    return patients


# ── Public API ────────────────────────────────────────────────────────────────
def get_overview():
    pts = _generate_patients()
    n = len(pts)
    seizure_free = sum(1 for p in pts if p["seizure_control"] == "seizure-free")
    drug_resistant = sum(1 for p in pts if p["seizure_control"] == "drug-resistant")
    kd_on = sum(1 for p in pts if p["kd_on"] == "Y")
    cbz_oxc_on = sum(1 for p in pts if "CBZ" in p["current_treatment"] or "OXC" in p["current_treatment"])
    hyponatraemia = sum(1 for p in pts if p["na_level_mmoll"] < 133)
    hla_tested = sum(1 for p in pts if p["hla_b1502_tested"] == "Y")
    avg_onset_hrs = round(sum(p["onset_age_hours"] for p in pts) / n, 1)
    gof_n = sum(1 for p in pts if p["category"] == "De-novo-KCNQ2-GOF-dominant-negative")

    return {
        "syndrome": "KCNQ2 Encephalopathy (KCNQ2-DEE)",
        "gene": "KCNQ2 (20q13.33)",
        "inheritance": "De novo in >96% (< 4% familial AD)",
        "n_patients": n,
        "key_gene": "KCNQ2 (20q13.33) — Kv7.2 voltage-gated K+ channel; M-current; co-assembles with Kv7.3 (KCNQ3)",
        "eeg_hallmark": "Burst-suppression (neonatal) → multifocal independent spike-wave (infancy) → background theta slowing (chronic). Hemisynchronous BS distinguishes from structural OHTAHARA.",
        "key_biomarker": "KCNQ2 sequencing (rapid WES or gene panel); HLA-B*15:02 before CBZ/OXC; CBZ/OXC-MHD TDM; Na+ monitoring",
        "key_aha": (
            "CBZ / OXC FIRST-LINE KCNQ2-SPECIFIC therapy — dual mechanism: Na-channel block + M-current enhancement. "
            "Phenobarbital alone INSUFFICIENT — transition to CBZ/OXC within 48-72h of stabilisation. "
            "HLA-B*15:02 MANDATORY before CBZ/OXC in South/SE Asian patients (CPIC Level A). "
            "Retigabine/Ezogabine WITHDRAWN 2017 — do NOT prescribe. "
            "XEN496 EPIK trial (NCT05374343) — refer eligible patients 2-12y."
        ),
        "etiologies": [
            {"etiology": "De novo KCNQ2 GOF / dominant-negative missense", "category": "De-novo-KCNQ2-GOF-dominant-negative", "pct": 45},
            {"etiology": "De novo KCNQ2 LOF missense (DEE variant)", "category": "De-novo-KCNQ2-LOF-DEE", "pct": 29},
            {"etiology": "De novo KCNQ2 truncating (nonsense / frameshift)", "category": "De-novo-KCNQ2-truncating", "pct": 12},
            {"etiology": "Familial AD KCNQ2 (SLNE with DEE presentation)", "category": "Familial-AD-KCNQ2-SLNE-DEE", "pct": 10},
            {"etiology": "Clinical KCNQ2-DEE — KCNQ2-negative (expanded panel pending)", "category": "Clinical-KCNQ2-DEE-KCNQ2-negative", "pct": 4},
        ],
        "seizure_type_prevalence": {
            "Tonic Seizures (neonatal tonic / focal tonic)": 100,
            "Focal Clonic Seizures (multifocal / alternating)": 78,
            "Autonomic Seizures (apnoea / bradycardia / cyanosis)": 55,
            "Focal-to-Bilateral Tonic-Clonic (FBTCS)": 32,
        },
        "trigger_seizure_rates": {
            "Fever / acute febrile illness": 82,
            "Intercurrent illness (non-febrile)": 71,
            "Missed / delayed AED dose": 68,
            "Sleep deprivation": 52,
            "Hyperthermia (bath / environment)": 45,
            "Metabolic disruption (hypoglycaemia / hyponatraemia)": 38,
            "Procedural stimulation (neonatal period)": 30,
            "Rapid AED taper / withdrawal": 22,
        },
        "lifecycle_windows": LIFECYCLE_WINDOWS,
        "kpis": {
            "seizure_free_pct": round(seizure_free / n * 100, 1),
            "drug_resistant_pct": round(drug_resistant / n * 100, 1),
            "cbz_oxc_on_pct": round(cbz_oxc_on / n * 100, 1),
            "kd_on_pct": round(kd_on / n * 100, 1),
            "hyponatraemia_pct": round(hyponatraemia / n * 100, 1),
            "hla_b1502_tested_pct": round(hla_tested / n * 100, 1),
            "avg_onset_age_hours": avg_onset_hrs,
            "gof_dominant_negative_pct": round(gof_n / n * 100, 1),
        },
        "clinical_alerts": [
            "💊 FIRST-LINE: CBZ / OXC — KCNQ2-specific (Na-channel block + M-current enhancement). Transition from PB within 48-72h of stabilisation.",
            "🧬 HLA-B*15:02 MANDATORY before CBZ/OXC in South/SE Asian patients — SJS/TEN risk. CPIC Level A.",
            "🚫 Phenobarbital ALONE is insufficient — do NOT discharge KCNQ2-DEE on PB monotherapy without CBZ/OXC.",
            "🚫 Retigabine / Ezogabine WITHDRAWN 2017 — do NOT prescribe. Direct to XEN496 EPIK trial.",
            "🧪 XEN496 EPIK trial (NCT05374343): refer patients 2-12y with confirmed KCNQ2 and ongoing seizures.",
            "⚠️ SIADH: Monitor Na+ at 2 weeks on CBZ/OXC — OXC hyponatraemia more frequent; hold if Na+ < 130.",
            "🏥 POLG exclusion before VPA in any KCNQ2-negative DEE case — fatal hepatotoxicity risk.",
            "🛡️ SUDEP counselling (NICE NG217 §1.15): nocturnal alarm + rescue BZD plan for all KCNQ2-DEE families.",
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
