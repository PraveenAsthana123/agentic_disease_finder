"""
CDKL5 Deficiency Disorder (CDD) Dashboard
===========================================
41-patient cohort · CDKL5 loss-of-function · X-linked
CDKL5 Deficiency Disorder: pathogenic variant in CDKL5 (cyclin-dependent kinase-like 5,
Xp22.13) → early-infantile onset epileptic encephalopathy with infantile spasms,
severe intellectual disability, limited hand use, and hypotonia. Formerly classified as
"Hanefeld variant" of atypical Rett Syndrome before gene discovery (2004 — Tao 2004 AJHG;
Weaving 2004 AJHG). Now recognised as an independent entity: CDKL5 Deficiency Disorder.
KEY EEG: Early hypsarrhythmia-like pattern → multifocal high-amplitude spike-wave → diffuse
irregular slow-wave with bursts of synchronised high-amplitude generalised discharge;
characteristic high-amplitude fast activity (HAFA) seen in some CDD patients.
AED NOTE: Vigabatrin (VGB) for IS phase — MANDATORY SHARE REMS visual field monitoring.
VPA commonly used. KD highly effective in CDD (Level B). CBZ/OXC: relative CI (worsens).
DISEASE-MODIFYING: Soticlestat (OV935, cholesterol 24-hydroxylase inhibitor) in Phase 2/3
ARCADE trial (NCT04462003). No FDA-approved disease-modifying therapy as of 2024.
"""

import random
from datetime import datetime

SEED = 9131
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ─────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "De novo CDKL5 missense variant (kinase domain)",
        "n": 16, "pct": 39,
        "category": "De-novo-CDKL5-missense-kinase-domain",
        "mechanism": (
            "The most common variant class in CDD (~39% of N=41): de novo missense variants "
            "within the kinase domain of CDKL5 (Xp22.13, exons 1-18). CDKL5 is a serine/"
            "threonine kinase that autophosphorylates and phosphorylates downstream substrates "
            "including MeCP2 (serine-80), DNMT3A (DNA methyltransferase 3A), MAP1S, and CEP131. "
            "Missense variants in the kinase domain (typically between residues 1-297) result in "
            "near-complete loss of kinase activity, preventing CDKL5-mediated phosphorylation of "
            "its critical neuronal substrates. The CDKL5-MeCP2 signalling axis is essential for "
            "synaptic maturation: CDKL5-phosphorylated MeCP2 (pS80-MeCP2) is more stable and "
            "promotes dendritic spine density and synaptic plasticity. Loss of this axis leads to "
            "reduced dendritic complexity, impaired GABAergic interneuron maturation (especially "
            "parvalbumin-positive interneurons in hippocampus and cortex), altered glutamatergic "
            "transmission, and reduced inhibitory tone — the direct substrate for seizure "
            "susceptibility and intellectual disability in CDD. "
            "Common missense hot spots: R178 (most frequent, ~6%), T288M, A40V, R178Q/W. "
            "Genotype-phenotype: kinase-dead variants (p.T288M, p.K42R) are among the most severe "
            "(earliest IS onset, most refractory epilepsy); C-terminal missense variants tend to "
            "be milder with preserved ambulation and some communicative language."
        ),
        "eeg_correlate": (
            "CDD EEG in the missense/kinase-dead group shows the characteristic three-phase "
            "evolution: Phase 1 (0-4 months, before IS): multifocal independent spike and "
            "sharp-wave discharges on a disorganised, age-inappropriate background; "
            "Phase 2 (IS onset, 2-5 months): hypsarrhythmia or modified hypsarrhythmia — "
            "high-amplitude chaotic polymorphic slow waves with multifocal spikes; electrodecrement "
            "associated with clinical spasms; Phase 3 (post-IS, 6+ months): the background "
            "becomes persistently slow with continuous high-amplitude irregular generalised "
            "delta 1-3 Hz; periodic generalised high-amplitude fast activity bursts (HAFA — "
            "runs of 14-20 Hz recruiting activity) are seen in ~40% of CDD and are considered "
            "a characteristic, though not pathognomonic, CDD EEG biomarker; "
            "interictal multifocal spike-wave complexes (frontal, central, temporal); "
            "absent or markedly attenuated posterior dominant rhythm; "
            "sleep EEG: increased spike-wave index, near-continuous interictal discharges in NREM; "
            "focal and generalised ictal patterns depending on seizure type."
        ),
        "mri_finding": (
            "Brain MRI in de novo CDKL5 kinase-domain missense: often normal in the neonatal "
            "period. Progressive changes on serial MRI: (1) Simplified gyral pattern (reduced "
            "cortical folding) in ~25% — most common in severe kinase-dead variants. "
            "(2) Diffuse cerebral and cerebellar hypoplasia/atrophy (most prominent frontally "
            "and in cerebellar vermis) — appears by 12-18 months of age. "
            "(3) Thin corpus callosum (hypoplasia or thinning of the genu/body/splenium) in "
            "~30% of cases — correlates with intellectual disability severity. "
            "(4) Delayed myelination for age on T2-weighted sequences. "
            "(5) Increased T2 signal in periventricular and subcortical white matter on "
            "later imaging. (6) Hippocampal volume reduction (bilateral) — correlating with "
            "degree of memory and learning impairment. "
            "MRS: Reduced NAA/Cr and Cho/Cr ratios in the basal ganglia and prefrontal cortex — "
            "neuronal/axonal loss marker. Not specific but confirms widespread neuronal dysfunction."
        ),
    },
    {
        "etiology": "De novo CDKL5 nonsense / frameshift (truncating) variant",
        "n": 14, "pct": 34,
        "category": "De-novo-CDKL5-nonsense-frameshift-truncating",
        "mechanism": (
            "The second most common variant class (~34%): de novo nonsense (stop codon) or "
            "frameshift (insertion/deletion causing a premature stop) variants in CDKL5. These "
            "truncating variants result in either (1) null alleles (no protein produced via "
            "nonsense-mediated mRNA decay [NMD] for variants in exons 1-16) or (2) a truncated "
            "CDKL5 protein lacking C-terminal regulatory sequences (for variants beyond exon 16). "
            "Variants subject to NMD produce no kinase — representing complete loss-of-function. "
            "C-terminal truncations beyond residue ~900 may spare the kinase domain but disrupt "
            "CDKL5 localisation signals and regulatory sequences that modulate activity. "
            "The CDKL5 protein has a bipartite nuclear localisation signal and a nuclear export "
            "signal; C-terminal truncations cause cytoplasmic mislocalisation and reduction in "
            "nuclear CDKL5 activity. This class also includes in-frame deletions of critical "
            "regulatory exons. Truncating variants are slightly more common in males (X-hemizygous "
            "affected) than females (X-heterozygous), reflecting higher clinical severity in males. "
            "In females: random X-inactivation (XCI) determines penetrance; skewed XCI (>75% "
            "expression from the mutant X) leads to more severe phenotype. Males with CDD are "
            "rarer (CDKL5 is X-linked — males are hemizygous) but present with severe or "
            "ultra-severe encephalopathy. Key clinical predictor: NMD-susceptible variants predict "
            "the most severe epilepsy burden (highest daily seizure frequency in early life)."
        ),
        "eeg_correlate": (
            "Truncating CDKL5 variants produce the most severe EEG phenotype in CDD: "
            "Neonatal period: normal or disorganised background; occasionally burst-suppression "
            "pattern in the most severe NMD cases (exon 1-8 truncating). "
            "Infantile spasms phase: classic hypsarrhythmia with high inter-hemispheric "
            "asynchrony; the spasm-associated electrodecrement is often preceded by a "
            "generalised high-amplitude slow wave rather than a spike. "
            "Post-IS phase (>6 months): diffuse very-high-amplitude (>200 μV) irregular "
            "delta background with superimposed multifocal spikes; runs of "
            "synchronised generalised high-amplitude fast activity (HAFA, 14-25 Hz) — "
            "more frequent and higher in amplitude than in missense group; "
            "runs of generalised polyspike-wave 3-4 Hz; focal motor seizures have "
            "contralateral frontotemporal onset with rapid generalisation. "
            "Sleep: marked increase in spike-wave burden; suppression-burst-like episodes "
            "in NREM sleep. The very high background amplitude (>300 μV) with HAFA bursts "
            "is considered a distinguishing feature of severe CDD vs other epileptic "
            "encephalopathies and may facilitate diagnosis."
        ),
        "mri_finding": (
            "Severe truncating CDD MRI: markedly abnormal in the majority. Generalised "
            "cerebral and cerebellar atrophy (early, progressive from 6-12 months). "
            "Corpus callosum: hypoplasia/agenesis of the genu and body in ~40%. "
            "Cortical malformations: simplified gyral pattern, pachygyria, or mild "
            "lissencephaly in ~15% of severe NMD cases. White matter signal: extensive "
            "T2 hyperintensity throughout periventricular and deep white matter. "
            "Basal ganglia: increased T2 signal or atrophy of the putamen and caudate. "
            "Brainstem: hypoplastic or atrophic pons/medulla in the most severe cases. "
            "All imaging abnormalities progress over time; serial MRI recommended at "
            "diagnosis, 12 months, and then every 2-3 years."
        ),
    },
    {
        "etiology": "De novo CDKL5 large deletion / duplication (CNV)",
        "n": 7, "pct": 17,
        "category": "De-novo-CDKL5-large-deletion-duplication-CNV",
        "mechanism": (
            "CDKL5 large deletions or duplications (~17% of CDD) represent copy-number "
            "variants (CNVs) detected by chromosomal microarray (CMA) or MLPA rather than "
            "by sequencing alone. Deletions encompassing part or all of CDKL5 (Xp22.13) "
            "result in complete haploinsufficiency in females (absent CDKL5 kinase on one "
            "X chromosome) or hemizygous deletion in males (complete absence). "
            "The deletion size ranges from focal intragenic deletions (single or multiple "
            "exons) to megabase-scale deletions that also encompass neighbouring genes "
            "(IQSEC2, NHS, SCML1) — the latter produce contiguous gene deletion syndromes "
            "with additional features (NHS syndrome with cataracts, IQSEC2-associated ID). "
            "CDKL5 duplications are rarer but documented; their pathogenicity depends on "
            "the duplication orientation and whether the extra copy disrupts regulatory "
            "elements. CDKL5 deletions involving exons 1-5 are the most clinically severe "
            "(complete CDKL5 loss); intragenic deletions of exons 6-16 may preserve some "
            "CDKL5 function. Testing: standard WES may miss CNVs — CMA with ≥200 Kbp "
            "resolution and MLPA targeting CDKL5 exons required in sequencing-negative CDD."
        ),
        "eeg_correlate": (
            "CNV-class CDD EEG is broadly similar to truncating variants but with additional "
            "complexity if neighbouring genes are deleted. Pure CDKL5-only deletions: "
            "hypsarrhythmia in IS phase, followed by persistently disorganised high-amplitude "
            "multifocal-to-generalised background. Contiguous gene deletions (CDKL5+IQSEC2): "
            "even more severe background with earlier epilepsy onset (< 4 weeks in some "
            "CDKL5+IQSEC2 deletions — IQSEC2 mutations independently cause early severe "
            "epilepsy); hypsarrhythmia-variant with increased interictal spike density. "
            "NHS-encompassing deletions: generally similar to pure CDKL5 deletions for "
            "EEG — NHS protein does not appear to modulate seizure threshold independently. "
            "Post-IS: persistent multifocal > generalised pattern. HAFA bursts present in "
            "~35% of CNV cases. Inter-ictal focal slowing (temporal > frontal) commonly seen "
            "on the background of the diffuse high-amplitude slow pattern."
        ),
        "mri_finding": (
            "CNV CDD MRI: pure CDKL5 CNV deletions show similar findings to truncating "
            "variants (progressive atrophy, thin CC, delayed myelination). "
            "Contiguous deletions including IQSEC2: additional basal ganglia signal "
            "abnormalities, more severe cerebral atrophy. "
            "NHS-deletion cases: lens opacities (cataracts) on clinical exam — not MRI. "
            "All CNV cases: routine T1/T2/FLAIR at diagnosis + 12M; DWI during acute "
            "seizure cluster to exclude status-related cytotoxic oedema."
        ),
    },
    {
        "etiology": "Familial (maternal carrier) CDKL5 variant",
        "n": 3, "pct": 7,
        "category": "Familial-maternal-carrier-CDKL5",
        "mechanism": (
            "Familial CDD (~7% of cases) occurs when a mother carries a CDKL5 pathogenic "
            "variant in a mosaic or germline state and transmits it to her daughter (or son). "
            "Maternal mosaicism: the mother carries the CDKL5 variant in a proportion of "
            "cells (blood somatic mosaicism), may be clinically unaffected, and has a "
            "recurrence risk depending on gonadal vs somatic mosaicism extent. "
            "Germline carrier mothers: carry the variant on one X chromosome, typically "
            "protected by favourable X-inactivation (mutant X inactivated in >75% of cells). "
            "If skewed XCI is unfavourable, carrier mothers may have a milder CDD-like "
            "phenotype (mild ID, seizures in some). "
            "Male carriers are extremely rare (if CDKL5 mutation on X, hemizygous males have "
            "CDD by definition — they are affected, not carriers). "
            "Recurrence risk: de novo CDD has ~1% recurrence risk (gonadal mosaicism); "
            "documented maternal carrier → 50% risk of transmission to daughters, "
            "25% to sons (sons are severely affected if inheriting the mutation)."
        ),
        "eeg_correlate": (
            "Familial CDKL5 EEG mirrors the de novo case phenotype; the variant class "
            "(missense vs truncating) drives EEG severity rather than inheritance pattern. "
            "Maternal carriers with favourable XCI and mild phenotype may show mild, "
            "non-diagnostic EEG abnormalities (infrequent focal spikes on otherwise "
            "normal background). In documented carrier mothers with seizure history: "
            "focal temporal or centrotemporal spikes in childhood, normalising with "
            "age — consistent with the observation that favourable XCI protects against "
            "severe seizure phenotype but not against minor EEG abnormalities."
        ),
        "mri_finding": (
            "Familial CDD MRI in affected children: same as de novo cases. "
            "Carrier mothers with mild phenotype: typically normal MRI. "
            "Some documented asymptomatic carriers show subtle reduced white matter "
            "volume on volumetric MRI analysis — subclinical finding not visible on "
            "standard clinical MRI. Not currently clinically actionable."
        ),
    },
    {
        "etiology": "Clinical CDD — CDKL5-negative / atypical (mosaic / VUS)",
        "n": 1, "pct": 3,
        "category": "Clinical-CDD-CDKL5-negative-atypical",
        "mechanism": (
            "A small proportion (~3%) meet clinical criteria for CDD (early-onset IS before "
            "5 months, severe-profound ID, limited/absent hand use, characteristic EEG) "
            "but return a negative result on WES/WGS and CMA. This category includes: "
            "(1) Deep intronic variants in CDKL5 creating cryptic splice sites — not "
            "detectable by standard exon-focused sequencing; RNA sequencing (RNAseq) on "
            "blood or fibroblasts required. (2) Variants of uncertain significance (VUS) "
            "in CDKL5 where functional evidence (kinase activity assay, CDKL5 substrate "
            "phosphorylation) has not yet been obtained. (3) CDKL5 promoter variants — "
            "very rare, requiring 5' regulatory region sequencing. (4) True phenocopies: "
            "early-onset severe epileptic encephalopathy caused by IQSEC2, ARX, SPTAN1, "
            "or other genes mimicking the CDD clinical phenotype. "
            "Approach: if WES negative in clinical CDD, order: (a) MLPA CDKL5 exon-level "
            "CNV analysis; (b) RNA-seq CDKL5 splicing assay; (c) Expanded epilepsy gene "
            "panel targeting early-onset EE; (d) WGS with CDKL5 intronic coverage."
        ),
        "eeg_correlate": (
            "Clinical CDD (CDKL5-negative): EEG phenotype overlaps with CDD but may show "
            "subtle differences. IQSEC2 phenocopies: very early onset epilepsy (neonatal), "
            "extremely high seizure burden (>100 spasms/day), severe EEG background "
            "disorganisation — potentially even more severe than typical CDKL5. "
            "ARX phenocopies (males): neonatal onset, hyperkinetic movement, "
            "burst-suppression evolving to hypsarrhythmia — Ohtahara-West transition. "
            "SPTAN1 phenocopies: very severe, agenesis of CC, neonatal onset, "
            "burst-suppression. All clinical CDD phenocopies share: early-onset (< 6M) "
            "hypsarrhythmia or severe background disorganisation, IS/focal spasms, "
            "high drug resistance. Consider multi-gene epilepsy panel before clinical CDD label."
        ),
        "mri_finding": (
            "Clinical CDD CDKL5-negative MRI: variable — depends on underlying aetiology. "
            "IQSEC2: similar to CDKL5 (progressive atrophy, thin CC). "
            "ARX (males): basal ganglia signal, hypothalamic abnormalities (in X-linked "
            "lissencephaly with abnormal genitalia — XLAG), dysgyria. "
            "SPTAN1: complete/partial agenesis of corpus callosum + olivopontocerebellar "
            "hypoplasia — distinct from typical CDD MRI. "
            "WGS structural analysis recommended for all MRI-abnormal CDKL5-negative cases."
        ),
    },
]

# ── Seizure Types (4 types with EEG correlates + clinical tips) ──────────────
SEIZURE_TYPES = [
    {
        "type": "Infantile Spasms / Early Epileptic Spasms (IS/EES)",
        "prevalence_pct": 90,
        "eeg_correlate": (
            "Classical or modified hypsarrhythmia during IS phase (typically 2-5 months of age "
            "in CDD): high-amplitude (>200 μV), chaotic, disorganised polymorphic slow-wave "
            "activity with multifocal spike and polyspike discharges; marked inter-hemispheric "
            "asynchrony. Ictal: generalised electrodecrement (attenuation) associated with "
            "clinical flexion or extension spasm — the electrodecrement is preceded by a "
            "generalised high-amplitude slow transient (not always a clear spike). "
            "Duration of IS phase: 6-18 months in CDD (longer than idiopathic IS). "
            "Unlike TSC-IS (VGB first-line), CDD-IS responds to ACTH + VGB combo but less "
            "completely (60% spasm-free vs 80% in other IS aetiology). "
            "Key: IS in CDD almost universally recur even after initial ACTH response — "
            "ongoing vigilance and multi-agent AED continuation required."
        ),
        "clinical_tip": (
            "CDD IS: earlier onset (median 5 weeks vs 5-6 months in cryptogenic IS). "
            "ACTH IM + vigabatrin combination recommended (UKISS-like regimen but longer). "
            "Spasm remission does not equal seizure remission — focal motor seizures and GTCS "
            "almost universally persist after IS phase. "
            "MANDATORY: vigabatrin SHARE REMS enrolment + visual field (VF) testing — "
            "formal Goldman perimetry q3M; OCT retinal nerve fibre layer annually. "
            "Infant VF testing: electrophysiological VF (VEP sweep or perimetry); "
            "formal static perimetry deferred until age 3+ years."
        ),
    },
    {
        "type": "Focal Motor Seizures (tonic / hypermotor / clonic)",
        "prevalence_pct": 85,
        "eeg_correlate": (
            "Focal onset with contralateral hemispheric discharge — most commonly "
            "frontotemporal (F7/F8, T3/T4) or frontocentral (Fc3/Fc4, C3/C4). "
            "Ictal pattern: rhythmic alpha-theta (8-12 Hz then slowing) recruiting discharge "
            "with rapid secondary generalisation to tonic phase. "
            "In the context of the diffuse high-amplitude CDD background, focal onset may "
            "be obscured — multi-channel EEG with full 10-20 montage essential. "
            "Hypermotor seizures: bilateral tonic posturing ± hypermotor automatisms "
            "(pedalling, thrashing) with diffuse bilateral ictal discharge. "
            "Clonic seizures: typically hemispheric, later generalising; 3-5 Hz rhythmic "
            "contralateral spike-wave. "
            "Seizure duration: focal motor 30-120 seconds; hypermotor 60-180 seconds. "
            "Breakthrough pattern: common in mornings and at sleep-wake transition."
        ),
        "clinical_tip": (
            "Focal motor seizures in CDD are the dominant seizure type post-IS-phase and "
            "typically persist throughout life despite optimised AED therapy. "
            "CLB (clobazam) is one of the most effective agents for focal motor CDD seizures "
            "(Level B evidence); tolerance can develop over 3-6 months — dose escalation "
            "rotation with CLN (clonazepam) may be needed. "
            "Video-EEG useful when distinguishing hypermotor seizures from non-epileptic "
            "movement stereotypies (common in CDD — hand mouthing, rocking, not seizures). "
            "Seizure safety planning: padded headgear for GTCS-risk patients with "
            "frequent falling during focal-to-bilateral tonic-clonic evolution."
        ),
    },
    {
        "type": "Focal-to-Bilateral Tonic-Clonic Seizures (FBTCS / GTCS)",
        "prevalence_pct": 65,
        "eeg_correlate": (
            "Secondary generalisation from focal onset: the focal discharge (frontotemporal "
            "or frontocentral) spreads bilaterally over 2-5 seconds; EEG transitions from "
            "focal rhythmic alpha-beta recruiting to generalised 8-12 Hz fast activity "
            "(tonic phase) → 3-5 Hz spike-wave (clonic phase) → post-ictal attenuation. "
            "Pure generalised onset GTCS also seen (rarer): simultaneous bilateral "
            "generalised polyspike-wave without clear focal onset — more common with "
            "truncating variants. "
            "Duration: 60-180 seconds (tonic) + 30-120 seconds (clonic). "
            "Post-ictal: marked suppression 20-60 minutes; may be associated with "
            "post-ictal focal neurological deficit (Todd's paresis) reflecting focal onset. "
            "SUDEP relevance: post-ictal EEG suppression + autonomic dysfunction in CDD "
            "suggests elevated SUDEP risk; nocturnal GTCS particularly concerning."
        ),
        "clinical_tip": (
            "FBTCS/GTCS in CDD carry significant SUDEP risk — nocturnal supervision, "
            "padded bed rails, and seizure alert devices (wearable or mattress) recommended "
            "for all CDD patients with nocturnal GTCS (NICE NG217 §1.15.2). "
            "Rescue therapy: buccal midazolam or intranasal diazepam for GTCS >5 min or "
            "cluster (≥2 GTCS within 24h without recovery). "
            "VPA (valproate) is a mainstay for GTCS in CDD — particularly effective as "
            "a broad-spectrum agent covering both focal and generalised patterns. "
            "Avoid CBZ/OXC: sodium channel blockers may worsen the IS background and "
            "have limited GTCS efficacy in CDD; classified as relative contraindication "
            "by IARC 2019 consensus."
        ),
    },
    {
        "type": "Myoclonic / Myoclonic-Atonic Seizures",
        "prevalence_pct": 45,
        "eeg_correlate": (
            "Myoclonic jerks in CDD: brief (< 100ms) generalised spike or polyspike "
            "followed by slow-wave; bilateral synchronous myoclonia — arms > legs, "
            "often axial. Cortical myoclonus: time-locked with spike discharge; "
            "giant SEPs may be present (cortical reflex myoclonus in some). "
            "Myoclonic-atonic (astatic): spike or polyspike followed by >200ms slow "
            "wave with loss of postural tone; electromyographic (EMG) burst followed "
            "by EMG silence; causes sudden falls (drop attacks). "
            "The myoclonic component in CDD is typically worse in the morning and "
            "on awakening from NREM sleep (timing consistent with primary generalised "
            "myoclonus pattern despite overall focal CDD background). "
            "High-amplitude background may obscure myoclonic discharges — look for "
            "semi-rhythmic muscle artefact on EEG suggesting sub-clinical myoclonus."
        ),
        "clinical_tip": (
            "Myoclonic seizures in CDD respond to CLN (clonazepam) — Level B for "
            "cortical myoclonus. KD (ketogenic diet) reduces myoclonic burden "
            "significantly in CDD (50-60% myoclonic seizure reduction in responders). "
            "LEV (levetiracetam): evidence mixed for myoclonus in CDD (effective in JME "
            "but limited data in CDKL5 specifically; adjunct use). "
            "AVOID VGB for myoclonus: vigabatrin is GABA-ergic but may paradoxically "
            "worsen myoclonic features in some CDD patients (worsens in 5-10%). "
            "Myoclonic-atonic drop attacks: protective headgear mandatory; assess "
            "fall risk at every clinic visit."
        ),
    },
]

# ── Triggers (8 with seizure rates) ─────────────────────────────────────────
TRIGGERS = [
    {"trigger": "Sleep transitions (waking / drowsiness onset)", "rate_pct": 90,
     "note": "CDD seizures cluster at sleep-wake transitions — especially tonic and focal motor seizures on arousal from NREM sleep and at the point of falling asleep."},
    {"trigger": "Fever / acute infection", "rate_pct": 80,
     "note": "Any febrile illness significantly increases seizure frequency in CDD. Temperature >38.5°C is a high-risk trigger — antipyretics early, Emergency Care Plan reviewed."},
    {"trigger": "Missed / delayed AED dose", "rate_pct": 75,
     "note": "CDD seizures are exquisitely sensitive to AED levels; even 2-4 hours late on CLB or VPA can precipitate seizure clusters. Consistent medication timing is critical."},
    {"trigger": "Overstimulation / emotional excitement", "rate_pct": 65,
     "note": "Intense positive or negative emotional arousal (excitement, distress, pain, startle) triggers focal motor and tonic seizures in CDD — autonomic arousal lowers seizure threshold."},
    {"trigger": "Hyperthermia (bath water, overheating)", "rate_pct": 55,
     "note": "Hot baths or environmental overheating without systemic fever can trigger seizures in CDD — thermosensitive sodium channels, similar to Dravet (SCN1A). Lukewarm baths; avoid overheating."},
    {"trigger": "Intercurrent illness (GI, URTI — without fever)", "rate_pct": 50,
     "note": "Non-febrile systemic illness (vomiting/diarrhoea with AED absorption disruption, or systemic inflammatory response) increases CDD seizure frequency — monitor AED absorption."},
    {"trigger": "Sleep deprivation", "rate_pct": 45,
     "note": "Reduced sleep duration or severely fragmented sleep increases seizure density — particularly focal motor morning-cluster seizures. Melatonin for CDD sleep disturbance supported by IARC consensus."},
    {"trigger": "Photic stimulation / pattern visual stimulation", "rate_pct": 15,
     "note": "A minority of CDD patients (15%) show photoparoxysmal EEG response (PPR) on IPS — relevant if attending concerts, gaming, or under strobe-type lighting. Photosensitivity screen at diagnosis."},
]

# ── Treatments (8 with dose / MOA / efficacy / safety / monitoring) ──────────
TREATMENTS = [
    {
        "drug": "ACTH (Acthar Gel / synthetic tetracosactide) — IS Phase",
        "evidence": "Level B (IS in CDD)",
        "dose": "ACTH (Acthar): 40-60 IU/day IM or tetracosactide 0.5-1.0 mg/day IM for 2 weeks, then taper over 4-6 weeks",
        "moa": (
            "Adrenocorticotropic hormone (ACTH) acts on melanocortin receptors (MC2R) in the "
            "adrenal cortex to stimulate cortisol secretion AND directly on MC5R in the brain "
            "(extra-adrenal mechanism) to suppress CRH-driven seizure circuits in the "
            "hypothalamus. In IS, ACTH is thought to reduce corticotropin-releasing hormone "
            "(CRH), which is excitatory in the neonatal/infant brain and may directly "
            "promote seizure generation and hypsarrhythmia. In CDD, ACTH targets the "
            "same IS mechanism — the CDD CDKL5-kinase deficit impairs inhibitory circuit "
            "maturation, but ACTH's suppression of CRH provides temporary seizure "
            "suppression regardless of the underlying genetic cause."
        ),
        "efficacy": "Spasm cessation (>72h) in ~60% of CDD-IS (vs 80%+ in cryptogenic IS); EEG hypsarrhythmia resolution in 55%. Lower sustained response rate at 3 months (~35%) compared to cryptogenic IS.",
        "safety": "Hypertension (monitor BP q2wk), immunosuppression (infection risk, avoid live vaccines during course), hyperglycaemia (blood glucose monitoring), growth suppression, irritability, cerebral volume loss (reversible).",
        "monitoring": "BP twice weekly during treatment; blood glucose (random + fasting); growth velocity monthly; infection screening before each visit; renal function (electrolytes weekly).",
    },
    {
        "drug": "Vigabatrin (VGB, Sabril)",
        "evidence": "Level B — IS component (especially combined with ACTH); adjunct for focal motor seizures",
        "dose": "IS: 50-150 mg/kg/day in 2 divided doses (typical infant dose 100 mg/kg/day). Maintenance: 40-80 mg/kg/day (paediatric). Max: 3g/day (adult).",
        "moa": (
            "Vigabatrin is a mechanism-based irreversible inhibitor of GABA-aminotransferase "
            "(GABA-T), the enzyme responsible for GABA catabolism in both neurons and glia. "
            "Inhibition of GABA-T raises synaptic GABA levels by preventing GABA breakdown, "
            "increasing GABAergic inhibitory tone throughout the brain. The IS mechanism of "
            "VGB is particularly well-understood in TSC (mTOR/tuberin pathway upregulates "
            "GABA-T; VGB corrects this); in CDD the mechanism is less specific but VGB "
            "increases net inhibitory tone to compensate for reduced parvalbumin interneuron "
            "activity caused by CDKL5 deficiency."
        ),
        "efficacy": "IS remission in CDD: 40-50% as monotherapy; 60% in combination with ACTH. Post-IS focal motor seizure control: partial responder in 30-40%; rarely seizure-free. Long-term efficacy limited by drug resistance.",
        "safety": "MANDATORY: vigabatrin permanent concentric visual field loss (peripheral vision) in up to 30% of paediatric patients (dose and duration dependent). Irreversible after prolonged exposure. SHARE REMS enrolment mandatory. Also: sedation, irritability, weight gain, peripheral neuropathy with chronic use.",
        "monitoring": "SHARE REMS programme enrolment BEFORE prescribing. Goldman kinetic perimetry q3M during treatment. OCT (retinal nerve fibre layer) q6M. Ophthalmology review at first missed VF test. Discontinue VGB if VF loss confirmed unless IS treatment benefit clearly outweighs risk.",
    },
    {
        "drug": "Valproate (VPA, sodium valproate)",
        "evidence": "Level B — broad-spectrum first-line in CDD (covers IS + focal + GTCS + myoclonic)",
        "dose": "10-40 mg/kg/day in 2-3 divided doses; TDM target 50-100 μg/mL. CDD infants often need higher weight-adjusted doses (>40 mg/kg/day) due to rapid metabolism.",
        "moa": (
            "Valproate is a broad-spectrum AED with multiple mechanisms: (1) Sodium channel "
            "blockade (use-dependent) — slows repetitive firing. (2) T-type calcium channel "
            "blockade — reduces thalamo-cortical oscillations (absence/spike-wave). "
            "(3) GABA-ergic enhancement — inhibits GABA-T, increases GABA synthesis. "
            "(4) Glutamate reduction — inhibits NMDA receptor. (5) Inhibits histone "
            "deacetylase (HDAC) — epigenetic anti-epileptic effect. "
            "In CDD: VPA's multi-target profile makes it effective across the mixed seizure "
            "types; the GABA-T inhibition provides complementary mechanism to LEV."
        ),
        "efficacy": "50-80% seizure frequency reduction in focal motor seizures (highest efficacy of oral AEDs in CDD); GTCS response 50-60%. Limited IS efficacy but maintains background seizure control during and after IS phase.",
        "safety": "Hepatotoxicity (risk highest < 2 years with polytherapy — MANDATORY LFTs). Hyperammonaemia (check ammonia if encephalopathy). Carnitine deficiency (monitor; supplement if low or with G-tube feeding). Teratogenicity (avoid or REMS in childbearing patients — not typically relevant in severe CDD females, but note for carriers). Thrombocytopenia. Pancreatitis (rare).",
        "monitoring": "LFT + ammonia + carnitine at baseline, 3M, 6M, then annually. Platelet count q6M. TDM: VPA level 50-100 μg/mL pre-dose trough. Avoid in suspected POLG mutation (fatal hepatic failure).",
    },
    {
        "drug": "Clobazam (CLB, Onfi)",
        "evidence": "Level B — focal motor seizures, adjunct for IS + GTCS in CDD",
        "dose": "0.1-0.5 mg/kg/day in 1-2 divided doses. Start 0.05 mg/kg/day; titrate over 2-4 weeks. Max: 40 mg/day (adult), 1-2 mg/kg/day (paediatric).",
        "moa": (
            "Clobazam is a 1,5-benzodiazepine (distinct from classical 1,4-BZDs) that acts "
            "as a positive allosteric modulator at GABA-A receptors — but with preference "
            "for the α2- and α3-subunit-containing GABA-A receptors over the sedative "
            "α1-subunit. This profile gives CLB better antiseizure efficacy with less "
            "sedation compared to classical benzodiazepines. The α2/α3 selectivity is "
            "particularly relevant for focal cortical epilepsy networks (relevant to CDD "
            "focal motor seizures). Active metabolite N-desmethylclobazam (norclobazam) "
            "contributes 20-30% of antiseizure activity at steady state."
        ),
        "efficacy": "Focal motor seizure reduction ≥50%: 60-70% of CDD patients; 25-30% become seizure-free for focal motor seizures for periods. Tolerance develops in 40% within 3-6 months — sequential dose escalation, CLB/CLN rotation, or drug holidays may restore efficacy.",
        "safety": "Sedation (dose-dependent; usually transient at initiation), hypotension (at high doses), respiratory depression (with other CNS depressants), paradoxical agitation/aggression (in 5-10% of developmental disorder patients).",
        "monitoring": "Clinical tolerance assessment q3M. CLB + norclobazam TDM if seizure escape (CLB target 30-300 ng/mL; norclobazam 300-3000 ng/mL). Respiratory function if concurrent VPA + CLB at high doses.",
    },
    {
        "drug": "Ketogenic Diet (KD 4:1 classic / MAD / MCT-KD)",
        "evidence": "Level B — strong efficacy evidence in CDD (specifically studied in CDKL5 cohorts)",
        "dose": "Classic 4:1 KD (fat:carbohydrate+protein ratio); MAD: fat 60-65% kcal with 20g/day free carbohydrate (for older/transition patients). Target serum BHB: 2-4 mmol/L. Dietitian-supervised initiation.",
        "moa": (
            "The ketogenic diet produces ketosis (BHB, AcAc) which replaces glucose as the "
            "primary neuronal energy substrate. Antiseizure mechanisms in CDD: "
            "(1) BHB directly inhibits synaptic vesicle exocytosis (blocking VGLUT — vesicular "
            "glutamate transporter) reducing excitatory glutamate release. "
            "(2) BHB activates adenosine A1 receptors (KD increases endogenous adenosine — "
            "an endogenous antiseizure molecule). "
            "(3) KD upregulates GABA synthesis via acetyl-CoA shuttling. "
            "(4) mTOR pathway inhibition (relevant in TSC-CDD overlap). "
            "In CDD specifically: CDKL5 deficiency impairs mitochondrial function in neurons; "
            "BHB as an alternative mitochondrial fuel may partially compensate for this — "
            "the so-called 'CDKL5-metabolic rescue hypothesis' (Cacciagli 2020 NatCommun)."
        ),
        "efficacy": "≥50% seizure reduction in 50-60% of CDD patients (comparable to drug-resistant focal epilepsy KD response). 10-15% seizure-free on KD in CDD cohort studies. Best responders: younger age at KD initiation (<4 years) and truncating/kinase-dead CDKL5 variants.",
        "safety": "Dyslipidaemia (LDL elevation in 30-40%), nephrolithiasis (renal ultrasound annually), selenium/zinc/carnitine deficiency with chronic KD (supplement), growth deceleration (monitor closely in CDD children), constipation (fibre supplementation), G-tube dependency (common in severe CDD — KetoCal formula).",
        "monitoring": "BHB twice weekly (target 2-4 mmol/L). Fasting lipid panel + selenium + zinc + carnitine q6M. Renal USS annually (nephrolithiasis). Growth percentile monthly. Alkaline urine pH + citrate supplementation to reduce stone risk.",
    },
    {
        "drug": "Levetiracetam (LEV, Keppra)",
        "evidence": "Level C — adjunct for focal motor seizures; limited CDD-specific RCT data",
        "dose": "20-60 mg/kg/day in 2 divided doses. Titrate 10 mg/kg/week. Target: 40-60 mg/kg/day in most paediatric CDD patients.",
        "moa": (
            "Levetiracetam binds synaptic vesicle protein 2A (SV2A), modulating the priming "
            "and fusion of synaptic vesicles and reducing excitatory neurotransmitter release "
            "(glutamate and norepinephrine). LEV also inhibits AMPA-receptor-mediated "
            "transmission and negatively modulates high-voltage-activated N-type calcium "
            "channels. In CDD, the SV2A mechanism is relevant because CDKL5 phosphorylates "
            "MAPK1/ERK2 and other kinases that regulate synaptic vesicle cycling — "
            "LEV's SV2A-binding may partially compensate for dysregulated vesicle priming. "
            "LEV is not effective for infantile spasms."
        ),
        "efficacy": "≥50% focal motor seizure reduction in 30-40% of CDD patients as add-on therapy. Limited efficacy for IS or GTCS in CDD. Well tolerated in most; behavioural side effects limit use in CDD.",
        "safety": "Neuropsychiatric side effects (irritability, agitation, aggression) in 15-25% of CDD patients — higher rate than general epilepsy population due to underlying neurobehavioural CDD phenotype. Drowsiness. No hepatotoxicity. No drug interactions (renally cleared).",
        "monitoring": "Behavioural CBCL-6/18 or ABC-C assessment at 4 weeks and 3 months. Renal function if concurrent nephrotoxic agents. No TDM routinely required; consider if seizure escape or toxicity suspected.",
    },
    {
        "drug": "Cannabidiol (CBD, Epidiolex)",
        "evidence": "Level C — open-label/compassionate data; no CDKL5-specific Phase 3 RCT completed",
        "dose": "2.5 mg/kg/day starting dose; titrate to 10-20 mg/kg/day in 2 divided doses. Max studied: 25 mg/kg/day in open-label CDD cohorts.",
        "moa": (
            "Cannabidiol (CBD) is a phytocannabinoid with multiple antiseizure mechanisms: "
            "(1) TRPV1 (transient receptor potential vanilloid 1) desensitisation — reduces "
            "calcium-mediated neuronal excitability. (2) GPR55 inverse agonism — GPR55 is a "
            "pro-convulsant lysophosphatidylinositol receptor; CBD blockade reduces neuronal "
            "excitability. (3) 5-HT1A receptor agonism — serotonergic anti-epileptic effect. "
            "(4) Inhibition of equilibrative nucleoside transporter (ENT1) — increases "
            "endogenous adenosine (pro-inhibitory). CBD is NOT a CB1/CB2 cannabinoid receptor "
            "agonist at clinical doses — no psychoactive effect, no THC-like mechanism. "
            "FDA approval: Epidiolex is approved for Dravet, LGS, and TSC — not specifically "
            "for CDD, but used off-label with compassionate use support in multiple countries."
        ),
        "efficacy": "Open-label CDD cohort studies (Devinsky 2019 Lancet Neurol; CDKL5 subgroup): ≥50% seizure reduction in 40% of CDD patients; 5-10% seizure-free. Myoclonic seizures showed better response than focal motor in some series.",
        "safety": "Hepatotoxicity: significant LFT elevation (>3× ULN) in 15-20% when combined with VPA — additive hepatotoxic risk; dose reduction of VPA if elevated LFTs. Somnolence. Diarrhoea (GI side effects in 30%). Drug interactions: CBD inhibits CYP3A4 and CYP2C19 — affects CLB metabolism (norclobazam levels elevated — may cause excess sedation or improved efficacy).",
        "monitoring": "LFT at baseline, 1M, 3M, 6M, then q6M (more frequent if combined with VPA). CBC. CLB/norclobazam TDM if combined with CLB (CBD-CLB interaction common). Monitor for over-sedation if polytherapy.",
    },
    {
        "drug": "Soticlestat (OV935) — Phase 2/3 Investigational",
        "evidence": "Phase 2/3 — ARCADE trial (NCT04462003) — not yet FDA/EMA approved",
        "dose": "ARCADE Phase 2b: oral 30-300 mg/day; paediatric dose-finding ongoing. Not commercially available.",
        "moa": (
            "Soticlestat is a first-in-class cholesterol 24-hydroxylase (CH24H / CYP46A1) "
            "inhibitor. CH24H converts cholesterol to 24S-hydroxycholesterol (24-HC) in neurons "
            "— the primary pathway for cholesterol elimination from the brain. 24-HC is an "
            "endogenous positive allosteric modulator (PAM) of NMDA receptors: it binds the "
            "GluN2B subunit of NMDArs and increases NMDA-mediated excitatory neurotransmission. "
            "By inhibiting CH24H, soticlestat reduces brain 24-HC levels, reducing NMDA "
            "receptor over-activation, thereby decreasing excitatory tone. "
            "In CDKL5-deficient models: NMDA receptor over-activity contributes to seizure "
            "generation; reducing 24-HC-mediated NMDA PAM activity represents a novel "
            "mechanism distinct from all approved AEDs. Preclinical CDD mouse models showed "
            "significant seizure frequency reduction. ARCADE Phase 2b (CDD + Angelman) "
            "interim results (2023 AES) showed 28-37% median convulsive seizure reduction."
        ),
        "efficacy": "Phase 2b ARCADE trial interim (2023): CDD arm: 37% median convulsive seizure reduction (responder rate 50%; p=0.04 vs placebo). Phase 3 ongoing. Not yet approved.",
        "safety": "Phase 2b: well tolerated; somnolence (5%), headache (4%), diarrhoea (3%); no hepatotoxicity signal; no VF loss. Phase 3 ongoing — safety profile not fully established.",
        "monitoring": "Phase 2b/3 trial protocol: seizure diary, LFTs, ophthalmological exam, cholesterol levels. Access: via named patient program or clinical trial enrolment only.",
    },
]

# ── Absolute / Relative Contraindications (4 items) ──────────────────────────
ABSOLUTE_CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) — Relative CI",
        "reason": (
            "Sodium channel blockers CBZ and OXC may worsen infantile spasms and "
            "focal motor seizures in CDD. The CDKL5-deficient brain has reduced "
            "inhibitory interneuron tone — sodium channel blockade preferentially "
            "suppresses fast-spiking inhibitory interneurons (PV-positive) more than "
            "pyramidal excitatory neurons at therapeutic doses, paradoxically increasing "
            "the excitation-inhibition imbalance. Clinical reports in CDD cohorts "
            "(Mangatt 2016 Orphanet) document worsening in 10-15% of patients. "
            "IARC 2019 consensus: relative contraindication — avoid unless no alternatives. "
            "If used, monitor for IS worsening, increased spasm frequency, or EEG deterioration."
        ),
        "severity": "Relative CI — avoid; use only as last resort with EEG monitoring",
    },
    {
        "drug": "Vigabatrin — MANDATORY SHARE REMS / Visual Field Monitoring",
        "reason": (
            "Vigabatrin causes irreversible concentric peripheral visual field (VF) loss "
            "in up to 30% of paediatric patients on prolonged therapy (>6 months). "
            "This is caused by GABA accumulation in retinal Müller cells and GABA-T "
            "inhibition in photoreceptors. The visual field loss is irreversible, "
            "bilateral, and progressive — cannot be detected by routine clinical exam. "
            "VGB is nonetheless indicated for IS-phase CDD given the severity of epileptic "
            "spasms and absence of other effective IS treatments — but SHARE REMS "
            "enrolment is MANDATORY before prescribing. "
            "Goldman kinetic perimetry q3M during treatment. If VF loss detected: "
            "weigh IS benefit vs irreversible visual deficit; consider early transition "
            "to alternative IS therapies."
        ),
        "severity": "MANDATORY REMS enrolment — VGB can be used but with mandatory VF monitoring programme",
    },
    {
        "drug": "NPO (nil-by-mouth) without KD formula continuation — Metabolic CI",
        "reason": (
            "CDD patients on the ketogenic diet (KD) maintained in hospital who are "
            "made NPO (nil-by-mouth) without continuation of KD-appropriate enteral "
            "formula (KetoCal or equivalent) via nasogastric or G-tube will suffer "
            "rapid loss of ketosis (BHB < 0.5 mmol/L within 2-4 hours of NPO). "
            "Loss of ketosis removes the antiseizure metabolic protection of KD and "
            "is a recognised cause of acute seizure clusters and status epilepticus "
            "in KD-dependent CDD patients. "
            "MANDATORY hospital protocol: KD formula continued at maintenance rate via NG "
            "or G-tube for all surgical/investigational NPO procedures unless general "
            "anaesthesia requires full NPO (in which case IV lipid emulsion [Intralipid] "
            "0.5-1.0 g/kg/hr to maintain ketosis). "
            "All CDD patients on KD must carry an Emergency KD Card."
        ),
        "severity": "ABSOLUTE CI in KD-maintained patients — hospital protocol mandatory",
    },
    {
        "drug": "POLG mutation overlap — VPA absolute CI if POLG suspected",
        "reason": (
            "Alpers-Huttenlocher syndrome (POLG pathogenic variant) causes an early-onset "
            "severe epileptic encephalopathy that can phenocopy CDD, especially the "
            "CDKL5-negative/atypical subgroup. Valproate administration in POLG causes "
            "fatal fulminant hepatic failure (VPA inhibits POLG2 polymerase and depletes "
            "mitochondrial DNA — Alpers triad: seizures + encephalopathy + liver failure). "
            "Before prescribing VPA in any CDD-negative patient with progressive "
            "encephalopathy: EXCLUDE POLG by targeted Sanger/WES sequencing. "
            "If POLG variants identified (or strongly suspected — mtDNA depletion on "
            "muscle biopsy, elevated CSF lactate): VPA is ABSOLUTELY contraindicated. "
            "Alternative: LEV + CLB + KD (avoid any mitochondrial-toxic agent)."
        ),
        "severity": "ABSOLUTE CI if POLG mutation present or suspected",
    },
]

# ── AED Monitoring (5 items) ─────────────────────────────────────────────────
AED_MONITORING = [
    {
        "item": "Vigabatrin (VGB) — SHARE REMS visual field monitoring",
        "schedule": "Goldman kinetic perimetry q3M; OCT retinal nerve fibre layer q6M; full ophthalmological assessment at VGB initiation and annually",
        "target": "VF loss: stop VGB if concentric VF loss confirmed on Goldman; OCT RNFL < 50 μm → urgent ophthalmology",
    },
    {
        "item": "VPA (Valproate) — LFT + ammonia + carnitine",
        "schedule": "LFT (AST/ALT/bilirubin) + ammonia + carnitine at baseline, 1M (paediatric), 3M, 6M, then q6M. Platelet count q6M.",
        "target": "AST/ALT >3× ULN → hold VPA + gastroenterology review. Ammonia >60 μmol/L without VPA levels → carnitine 50-100 mg/kg/day supplementation. Carnitine < 25 μmol/L → supplement.",
    },
    {
        "item": "Ketogenic Diet — BHB, metabolic, growth monitoring",
        "schedule": "BHB (serum or urine ketones) twice weekly during titration, then weekly once stable. Fasting lipid panel + selenium + zinc + carnitine q6M. Renal USS annually. Weight/height/OFC monthly.",
        "target": "BHB target 2-4 mmol/L. LDL >4.0 mmol/L → dietitian review + MCT oil partial substitution. Selenium < 0.7 μmol/L → supplement. Nephrolithiasis on USS → increase fluid intake + alkalinise urine.",
    },
    {
        "item": "ACTH / Steroids — blood pressure, glucose, growth, infection",
        "schedule": "BP twice weekly during ACTH treatment. Random blood glucose twice weekly. Growth velocity monthly. Infection screening before each visit. Electrolytes weekly.",
        "target": "Hypertension (>95th centile for age): dose reduction or antihypertensive. Hyperglycaemia (random >11 mmol/L): endocrine review. GI haemorrhage: ranitidine prophylaxis during ACTH course.",
    },
    {
        "item": "CBD (Epidiolex) — LFT + drug interaction monitoring",
        "schedule": "LFT at baseline, 1M, 2M, 3M, then q3-6M. CLB/norclobazam TDM if CBD + CLB combination. Somnolence and CNS depression assessment at each visit.",
        "target": "LFT >3× ULN (especially with VPA co-administration) → reduce VPA/CBD dose; >8× ULN → stop CBD. CLB toxicity (excess sedation, respiratory depression) → measure norclobazam level; reduce CLB dose by 25-50%.",
    },
]

# ── Lifecycle Windows (6 stages) ─────────────────────────────────────────────
LIFECYCLE_WINDOWS = [
    {
        "window": "Neonatal / Early Infantile (0–4 months)",
        "key_features": [
            "Seizure onset in 90% before 5 months (median ~5 weeks)",
            "Early multifocal EEG discharges; hypotonia; poor feeding",
            "CDKL5 sequencing + MLPA initiated urgently",
            "Emergency VPA loading at seizure onset pending genetic results",
        ],
    },
    {
        "window": "Infantile Spasms Phase (3–18 months)",
        "key_features": [
            "IS onset 2-5 months: ACTH + VGB combination — SHARE REMS enrolment",
            "Hypsarrhythmia on EEG; developmental regression",
            "KD introduction after ≥2 failed AED trials",
            "Multidisciplinary: neurology, genetics, ophthalmology (VGB VF), dietitian",
        ],
    },
    {
        "window": "Early Childhood — Refractory Epilepsy (1–5 years)",
        "key_features": [
            "Post-IS mixed epilepsy (focal motor + GTCS + myoclonic)",
            "Drug resistance established (~75% by age 2 years)",
            "VPA + CLB + KD as core regimen in most",
            "Developmental plateau: limited purposeful hand use, minimal speech",
            "Introduction of augmentative communication (AAC) devices",
        ],
    },
    {
        "window": "School Age — Chronic Management (5–12 years)",
        "key_features": [
            "Stable but chronic epilepsy; AED optimisation rather than cure",
            "VF testing feasible (formal Goldman perimetry from age 3+)",
            "Orthopaedic monitoring: scoliosis screening annually",
            "Seizure first-aid training for school staff; emergency care plan",
            "Consider CBD / soticlestat trial enrolment if eligible",
        ],
    },
    {
        "window": "Adolescence (12–18 years)",
        "key_features": [
            "Puberty: catamenial seizure worsening in some CDD females",
            "AED review: transition from paediatric to adult dosing",
            "SUDEP risk counselling for families",
            "Transition planning: adult neurology, residential care assessment",
            "Melatonin for sleep disturbance (supported by IARC consensus)",
        ],
    },
    {
        "window": "Adult (18 years+)",
        "key_features": [
            "Lifelong severe-profound intellectual disability; independent living not achievable",
            "Adult neurology + internal medicine co-management",
            "AED side effects accumulate: bone density (DEXA q2y), lipids, cardiac",
            "SUDEP risk remains high: nocturnal supervision device (wearable / mattress alert)",
            "Gene therapy trials may offer future disease-modification (AAV9-CDKL5 preclinical)",
        ],
    },
]

# ── Concepts (14 key definitions) ───────────────────────────────────────────
CONCEPTS = [
    {
        "term": "CDKL5",
        "definition": (
            "Cyclin-dependent kinase-like 5: a serine/threonine kinase encoded by the CDKL5 gene "
            "(Xp22.13) that phosphorylates MeCP2, DNMT3A, MAP1S, and CEP131. Essential for "
            "synaptic maturation, dendritic spine development, and GABAergic interneuron "
            "maturation. Loss of CDKL5 kinase activity causes CDD."
        ),
    },
    {
        "term": "CDKL5 Deficiency Disorder (CDD)",
        "definition": (
            "An X-linked epileptic encephalopathy caused by loss-of-function pathogenic variants "
            "in CDKL5 (Xp22.13). Characterised by early-onset epileptic spasms (< 5 months), "
            "severe intellectual disability, limited hand use, hypotonia, and characteristic EEG "
            "features (hypsarrhythmia → HAFA bursts). Incidence ~1:40,000–60,000 live births."
        ),
    },
    {
        "term": "Hanefeld Variant (historical)",
        "definition": (
            "The original designation for atypical Rett Syndrome with early-onset seizures "
            "(before 6 months), later attributed to CDKL5 pathogenic variants. Now superseded "
            "by the term CDKL5 Deficiency Disorder — CDD is a distinct entity from RTT."
        ),
    },
    {
        "term": "Infantile Epileptic Spasms Syndrome (IESS / IS)",
        "definition": (
            "The electroclinical syndrome of epileptic spasms (flexor/extensor/mixed), EEG "
            "hypsarrhythmia, and developmental regression in infancy. In CDD, IS is the "
            "presenting seizure type in ~90% of patients; onset at 2-5 months of age."
        ),
    },
    {
        "term": "Hypsarrhythmia",
        "definition": (
            "A chaotic, very-high-amplitude (>200 μV) EEG pattern with multifocal spikes, "
            "polyspikes, and slow waves on a disorganised background; the characteristic "
            "interictal EEG finding of infantile spasms. In CDD, hypsarrhythmia may be "
            "'modified' (asymmetric, incomplete, or with attenuation periods)."
        ),
    },
    {
        "term": "HAFA (High-Amplitude Fast Activity)",
        "definition": (
            "High-amplitude fast activity: runs of 14-25 Hz synchronised fast activity "
            "on EEG, seen in ~40% of CDD patients. Considered a characteristic (though "
            "not pathognomonic) electrographic biomarker of CDD; useful in distinguishing "
            "CDD from other epileptic encephalopathies in the differential diagnosis."
        ),
    },
    {
        "term": "CDKL5–MeCP2 Signalling Axis",
        "definition": (
            "The molecular pathway by which CDKL5 phosphorylates MeCP2 at serine-80. "
            "Phosphorylated MeCP2 (pS80-MeCP2) is more stable and less prone to "
            "proteasomal degradation; it modulates BDNF expression and GABAergic synapse "
            "maturation. Loss of CDKL5 reduces pS80-MeCP2 — the mechanistic link between "
            "CDD and the partially overlapping Rett Syndrome phenotype."
        ),
    },
    {
        "term": "Soticlestat (OV935)",
        "definition": (
            "A first-in-class cholesterol 24-hydroxylase (CH24H/CYP46A1) inhibitor in "
            "Phase 2/3 development for CDD and Angelman Syndrome (ARCADE NCT04462003). "
            "Reduces brain 24S-hydroxycholesterol levels → reduces NMDA receptor "
            "over-activation → antiseizure effect. Not yet FDA/EMA approved."
        ),
    },
    {
        "term": "Vigabatrin SHARE REMS",
        "definition": (
            "The FDA Risk Evaluation and Mitigation Strategy (REMS) programme for vigabatrin "
            "(Sabril). MANDATORY enrolment of prescriber, pharmacy, and patient/caregiver "
            "before first prescription. Visual field testing (Goldman kinetic perimetry) "
            "required q3M. VGB causes irreversible concentric peripheral visual field loss "
            "in up to 30% of paediatric patients with long-term use."
        ),
    },
    {
        "term": "Drug-Resistant Epilepsy (DRE) in CDD",
        "definition": (
            "Drug resistance: failure of adequate trials of 2 tolerated and appropriately "
            "dosed AEDs to achieve seizure freedom (ILAE 2010 definition). In CDD, "
            "~75% of patients meet DRE criteria by 24 months of age — one of the "
            "highest DRE rates among genetic epilepsies."
        ),
    },
    {
        "term": "POLG Exclusion (before VPA)",
        "definition": (
            "Alpers-Huttenlocher syndrome (POLG pathogenic variant) must be excluded "
            "before prescribing valproate in any patient with progressive encephalopathy "
            "or suspected mitochondrial disease. VPA in POLG patients causes fatal "
            "hepatic failure. POLG sequencing is mandatory in CDKL5-negative early-onset "
            "epileptic encephalopathy."
        ),
    },
    {
        "term": "X-Inactivation (XCI) in CDD Females",
        "definition": (
            "X-chromosome inactivation is the random epigenetic silencing of one X in "
            "female cells. In CDKL5 carrier/affected females: if the mutant X is "
            "preferentially inactivated (favourable skewing > 75%), the phenotype is "
            "mild or subclinical. If the wild-type X is preferentially inactivated "
            "(unfavourable skewing), the phenotype is severe. Skewing can be measured "
            "in blood (methyl-sensitive PCR). Gonadal XCI may differ from somatic."
        ),
    },
    {
        "term": "SUDEP in CDD",
        "definition": (
            "Sudden Unexpected Death in Epilepsy: CDD carries elevated SUDEP risk due to "
            "severe refractory epilepsy (frequent GTCS), autonomic dysregulation, and "
            "nocturnal seizures. NICE NG217 §1.15: night-time supervision device, "
            "SUDEP risk counselling, nocturnal seizure alarm for all high-risk patients."
        ),
    },
    {
        "term": "Gene Therapy — AAV9-CDKL5",
        "definition": (
            "Preclinical studies in CDKL5-deficient mouse models using adeno-associated "
            "virus serotype 9 (AAV9) to deliver functional CDKL5 cDNA show reversal of "
            "seizure phenotype and synaptic deficits. Phase 1 trials anticipated 2025-2026. "
            "Represents the only potential disease-modifying (not symptomatic) approach "
            "to CDD. Currently preclinical/early IND stage only."
        ),
    },
]

# ── Standards (6) ────────────────────────────────────────────────────────────
STANDARDS = [
    {"code": "ILAE-2022", "title": "ILAE Classification of the Epilepsies 2022 (Scheffer et al.)", "relevance": "CDD classification as genetic epileptic encephalopathy; IESS/IS criteria"},
    {"code": "NICE-NG217", "title": "NICE NG217 Epilepsies in Children, Young People and Adults (2022)", "relevance": "AED selection, SUDEP risk, rescue medication protocols, KD guidelines"},
    {"code": "IARC-2019", "title": "International CDKL5 Research Consortium Natural History Consensus 2019 (Fehr, Leonard, et al.)", "relevance": "CDD-specific natural history, AED selection, IARC contraindication guidance (CBZ/OXC relative CI)"},
    {"code": "FDA-SHARE-REMS", "title": "FDA SHARE REMS — Vigabatrin (Sabril) Safety Programme", "relevance": "Mandatory enrolment, Goldman perimetry q3M, VF monitoring protocol for all VGB prescriptions"},
    {"code": "FDA-CDD-DraftGuidance-2022", "title": "FDA Draft Guidance for CDD Clinical Trials (2022)", "relevance": "Primary endpoint selection (motor seizure frequency), trial design guidance for CDD interventional studies"},
    {"code": "ACNS-EEG-2021", "title": "ACNS Guideline for EEG in Epileptic Encephalopathies (2021)", "relevance": "EEG monitoring standards for IS phase, hypsarrhythmia grading, HAFA recognition"},
]

# ── Thresholds (8) ────────────────────────────────────────────────────────────
THRESHOLDS = [
    {"threshold": "Early IS evaluation: IS onset < 5 months → CDKL5 priority genetic testing (alongside TSC, KCNQ2, ARX)", "value": "5 months", "unit": "age at IS onset"},
    {"threshold": "Vigabatrin VF loss monitoring: Goldman perimetry q3M during VGB treatment (SHARE REMS)", "value": "q3M", "unit": "monitoring interval"},
    {"threshold": "VPA target TDM: 50-100 μg/mL pre-dose trough", "value": "50-100", "unit": "μg/mL"},
    {"threshold": "KD BHB target: 2-4 mmol/L for adequate antiseizure ketosis", "value": "2-4", "unit": "mmol/L BHB"},
    {"threshold": "DRE threshold: 2 adequate AED trials → drug-resistant epilepsy label (ILAE 2010)", "value": "2", "unit": "failed AED trials"},
    {"threshold": "Seizure-free for AED taper consideration: 2 years seizure-free (CDD rarely achieved)", "value": "2 years", "unit": "seizure freedom"},
    {"threshold": "LFT threshold for AED hepatotoxicity action: AST/ALT >3× ULN → hold hepatotoxic AED; >8× ULN → stop", "value": ">3× ULN (hold) / >8× ULN (stop)", "unit": "AST/ALT"},
    {"threshold": "Driving: excluded in all CDD patients (severe cognitive impairment + uncontrolled epilepsy in all jurisdictions)", "value": "excluded", "unit": "driving eligibility"},
]

# ── References (6) ────────────────────────────────────────────────────────────
REFERENCES = [
    {"citation": "Fehr S et al. (2015). The CDKL5 disorder is an independent clinical entity associated with early-onset encephalopathy. Eur J Hum Genet 23(9):1253-1260.", "key_finding": "First large natural history study; CDD distinct from RTT; IS in 85%, drug resistance in 75%, median age IS onset 4 months"},
    {"citation": "Mangatt M et al. (2016). Prevalence and onset of comorbidities in the CDKL5 disorder differ from Rett syndrome. Orphanet J Rare Dis 11(1):39.", "key_finding": "AED response in 152 CDD patients; VPA + CLB + KD most effective; CBZ/OXC worsening documented in 12%"},
    {"citation": "Devinsky O et al. (2019). Cannabidiol in patients with treatment-resistant epilepsy: an open-label interventional trial. Lancet Neurol 15(3):270-278.", "key_finding": "Open-label CBD in 214 patients incl CDKL5 subgroup: median 36% seizure reduction; LFT elevation with VPA co-administration"},
    {"citation": "IARC Natural History Consortium (Leonard H, Fehr S, et al., 2019). International CDKL5 Disorder Database consensus natural history. Dev Med Child Neurol 61(8):895-904.", "key_finding": "IARC consensus natural history data; HAFA EEG biomarker; treatment survey across 25 countries; CBZ/OXC relative CI recommendation"},
    {"citation": "Tao J et al. (2004). Mutations in the X-linked cyclin-dependent kinase-like 5 (CDKL5/STK9) gene are associated with severe neurodevelopmental retardation. Am J Hum Genet 75(6):1149-1154.", "key_finding": "CDKL5 discovery paper: first identification of CDKL5 pathogenic variants in patients with early-onset seizures and severe ID"},
    {"citation": "Weissberg S et al. (2023 AES Poster). Soticlestat in CDKL5 Deficiency Disorder — ARCADE Phase 2b interim. Epilepsia (conference abstract).", "key_finding": "ARCADE Phase 2b CDD arm: 37% median convulsive seizure reduction vs placebo; 50% responder rate; p=0.04"},
]


# ── Patient Generator ─────────────────────────────────────────────────────────
def _generate_patients():
    random.seed(SEED)
    etiology_options = [
        ("De novo CDKL5 missense (kinase domain)", "De-novo-CDKL5-missense-kinase-domain", 0.39),
        ("De novo CDKL5 nonsense/frameshift (truncating)", "De-novo-CDKL5-truncating", 0.34),
        ("De novo CDKL5 large deletion/duplication (CNV)", "De-novo-CDKL5-CNV", 0.17),
        ("Familial (maternal carrier) CDKL5", "Familial-maternal-carrier", 0.07),
        ("Clinical CDD — CDKL5-negative/atypical", "Clinical-CDD-CDKL5-negative", 0.03),
    ]
    aed_options = ["VPA", "CLB", "VGB", "LEV", "KD", "CBD", "CLN", "ACTH-history"]
    control_options = ["drug-resistant", "drug-resistant", "drug-resistant",
                       "partial-control", "partial-control", "seizure-free"]

    pts = []
    for i in range(1, 42):
        # etiology
        r = random.random()
        cum = 0.0
        eth_name, eth_cat = etiology_options[-1][0], etiology_options[-1][1]
        for name, cat, p in etiology_options:
            cum += p
            if r < cum:
                eth_name, eth_cat = name, cat
                break

        onset_months = max(1, int(random.gauss(4.5, 1.8)))  # median ~5 weeks → expressed in months
        onset_weeks = max(1, int(random.gauss(5.5, 3.0)))   # age in weeks

        # AEDs
        n_aeds = random.randint(2, 5)
        current_aeds = random.sample(aed_options[:7], min(n_aeds, 7))

        seizure_control = random.choice(control_options)
        kd_on = "Y" if "KD" in current_aeds else "N"
        vgb_ever = "Y" if "VGB" in current_aeds or random.random() < 0.45 else "N"
        vf_loss = "Y" if vgb_ever == "Y" and random.random() < 0.12 else "N"
        acth_history = "Y" if random.random() < 0.55 else "N"
        gastrostomy = "Y" if random.random() < 0.45 else "N"
        soticlestat_trial = "Y" if random.random() < 0.07 else "N"

        # sex: predominantly female (X-linked; males severely affected but rarer)
        sex = "F" if random.random() < 0.82 else "M"
        age_years = random.randint(1, 22)

        pts.append({
            "id": f"CDD{i:03d}",
            "age_years": age_years,
            "sex": sex,
            "etiology": eth_name,
            "etiology_category": eth_cat,
            "onset_age_months": onset_months,
            "onset_age_weeks": onset_weeks,
            "seizure_control": seizure_control,
            "current_aeds": ", ".join(current_aeds),
            "n_aeds": len(current_aeds),
            "kd_on": kd_on,
            "vgb_ever": vgb_ever,
            "vf_loss": vf_loss,
            "acth_history": acth_history,
            "gastrostomy": gastrostomy,
            "soticlestat_trial": soticlestat_trial,
        })
    return pts


# ── Public API ─────────────────────────────────────────────────────────────────
def get_overview():
    pts = _generate_patients()
    n = len(pts)
    seizure_free = sum(1 for p in pts if p["seizure_control"] == "seizure-free")
    drug_resistant = sum(1 for p in pts if p["seizure_control"] == "drug-resistant")
    kd_on = sum(1 for p in pts if p["kd_on"] == "Y")
    vgb_ever = sum(1 for p in pts if p["vgb_ever"] == "Y")
    vf_loss = sum(1 for p in pts if p["vf_loss"] == "Y")
    acth_history = sum(1 for p in pts if p["acth_history"] == "Y")
    gastrostomy = sum(1 for p in pts if p["gastrostomy"] == "Y")
    avg_onset = round(sum(p["onset_age_months"] for p in pts) / n, 1)

    return {
        "syndrome": "CDKL5 Deficiency Disorder (CDD)",
        "gene": "CDKL5 (Xp22.13)",
        "inheritance": "X-linked (de novo in >90%)",
        "n_patients": n,
        "key_gene": "CDKL5 (Xp22.13) — serine/threonine kinase; phosphorylates MeCP2 at Ser-80",
        "eeg_hallmark": "Hypsarrhythmia (IS phase) → High-Amplitude Fast Activity (HAFA 14-25 Hz) bursts + diffuse high-amplitude irregular delta with multifocal spikes",
        "key_biomarker": "CDKL5 sequencing (WES/Sanger) + MLPA for CNVs; RNAseq if sequencing-negative clinical CDD",
        "key_aha": (
            "MANDATORY SHARE REMS for vigabatrin (VGB): Goldman perimetry q3M — VF loss irreversible. "
            "CBZ/OXC: relative CI (IARC 2019 — may worsen spasms). "
            "KD highly effective in CDD (50-60% responder rate). "
            "Soticlestat (OV935) ARCADE trial ongoing — Phase 2/3 — consider enrolment."
        ),
        "etiologies": [
            {"etiology": "De novo CDKL5 missense (kinase domain)", "category": "De-novo-CDKL5-missense-kinase-domain", "pct": 39},
            {"etiology": "De novo CDKL5 nonsense/frameshift (truncating)", "category": "De-novo-CDKL5-truncating", "pct": 34},
            {"etiology": "De novo CDKL5 large deletion/duplication (CNV)", "category": "De-novo-CDKL5-CNV", "pct": 17},
            {"etiology": "Familial (maternal carrier) CDKL5", "category": "Familial-maternal-carrier", "pct": 7},
            {"etiology": "Clinical CDD — CDKL5-negative/atypical", "category": "Clinical-CDD-CDKL5-negative", "pct": 3},
        ],
        "seizure_type_prevalence": {
            "Infantile Spasms / Early Epileptic Spasms (IS/EES)": 90,
            "Focal Motor Seizures (tonic / hypermotor / clonic)": 85,
            "Focal-to-Bilateral Tonic-Clonic (FBTCS / GTCS)": 65,
            "Myoclonic / Myoclonic-Atonic Seizures": 45,
        },
        "trigger_seizure_rates": {
            "Sleep transitions (waking / drowsiness onset)": 90,
            "Fever / acute infection": 80,
            "Missed / delayed AED dose": 75,
            "Overstimulation / emotional excitement": 65,
            "Hyperthermia (bath, overheating)": 55,
            "Intercurrent illness (non-febrile)": 50,
            "Sleep deprivation": 45,
            "Photic stimulation": 15,
        },
        "lifecycle_windows": LIFECYCLE_WINDOWS,
        "kpis": {
            "seizure_free_pct": round(seizure_free / n * 100, 1),
            "drug_resistant_pct": round(drug_resistant / n * 100, 1),
            "kd_responder_pct": round(kd_on / n * 100, 1),
            "vgb_ever_pct": round(vgb_ever / n * 100, 1),
            "vf_loss_pct": round(vf_loss / n * 100, 1),
            "acth_history_pct": round(acth_history / n * 100, 1),
            "gastrostomy_pct": round(gastrostomy / n * 100, 1),
            "avg_onset_age_months": avg_onset,
        },
        "clinical_alerts": [
            "⚠️ MANDATORY SHARE REMS: Vigabatrin (VGB) — Goldman perimetry q3M — irreversible VF loss in up to 30% paediatric patients",
            "🚫 RELATIVE CI: CBZ / OXC — may worsen infantile spasms and focal motor seizures in CDD (IARC 2019 consensus)",
            "🍳 KD highly effective in CDD: 50-60% responder rate — initiate after 2 failed AEDs; G-tube KetoCal formula",
            "💊 MANDATORY: hospital NPO protocol for KD patients — continue KD formula via NG/G-tube; prevent ketosis loss",
            "🧬 POLG exclusion before VPA: rule out Alpers syndrome in CDKL5-negative cases — VPA fatal in POLG",
            "🔬 CDKL5-negative clinical CDD: order MLPA + RNAseq CDKL5 + expanded epilepsy panel (IQSEC2/ARX/SPTAN1)",
            "💊 Soticlestat (OV935) ARCADE trial recruiting: consider CDD patients for Phase 2/3 enrolment",
            "🛡️ SUDEP risk: nocturnal GTCS → wearable seizure alarm + padded rails; SUDEP counselling (NICE NG217 §1.15)",
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
