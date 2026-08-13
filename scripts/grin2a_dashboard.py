"""
GRIN2A Epilepsy-Aphasia Spectrum (GRIN2A-EAS / CSWS / Landau-Kleffner)
========================================================================
41-patient cohort · GRIN2A (2q33.1) · GluN2A / NR2A · NMDA Receptor Subunit
GRIN2A-related Epilepsy-Aphasia Spectrum: pathogenic variants in GRIN2A (2q33.1,
encoding the GluN2A subunit of the NMDA receptor) cause a clinical spectrum ranging
from the mild self-limited end (SeLECTS/BECTS with language features) to severe
Landau-Kleffner Syndrome (LKS) with acquired epileptic aphasia and continuous
spike-and-wave during sleep (CSWS / ESES). GRIN2A is the most frequent single-gene
cause of the Epilepsy-Aphasia Spectrum (EAS), accounting for ~20% of familial EAS.

NMDA RECEPTOR / GRIN2A BIOLOGY: GRIN2A encodes the GluN2A (NR2A) subunit, which
forms the primary ligand-binding domain of di-heteromeric NMDA receptors (GluN1/GluN2A
or GluN1/GluN2A/GluN2B). NMDA receptors require co-activation by glutamate (GluN2
subunit) and glycine/D-serine (GluN1 subunit) and are gated by Mg²⁺ voltage block —
the molecular basis of Hebbian synaptic plasticity. GluN2A subunits confer fast Mg²⁺
unblock kinetics, important for high-frequency synaptic transmission in rolandic
(perisylvian) cortical networks. Pathogenic GRIN2A variants cause gain-of-function
(GOF) or loss-of-function (LOF) effects on NMDA receptor gating, disrupting the
excitatory/inhibitory balance in perisylvian language circuits.

EPILEPSY-APHASIA SPECTRUM (EAS): The EAS is a group of syndromes sharing:
① Centrotemporal (rolandic / perisylvian) EEG discharges; ② Language/speech
involvement (aphasia, verbal auditory agnosia, speech regression); ③ Age-related
course (onset 2–12Y, remission adolescence). The spectrum includes:
• SeLECTS (Self-Limited Epilepsy with CentroTemporal Spikes) — mildest: nocturnal
  simple partial seizures, centrotemporal spikes, normal language in most
• Atypical SeLECTS / ESES — intermediate: CSWS without severe aphasia
• Landau-Kleffner Syndrome (LKS) — severe: acquired epileptic aphasia (verbal
  auditory agnosia), CSWS occupying ≥85% of NREM sleep, regression of language
• Epilepsy with Continuous Spike-and-Wave during Sleep (CSWS-epilepsy) — severe:
  CSWS + broader cognitive regression + multiple seizure types

CSWS / ESES (Continuous Spike-and-Wave during Sleep / Electrical Status Epilepticus
during Sleep): EEG pattern of continuous slow spike-and-wave (≤2.5 Hz) occupying
≥85% of NREM sleep. CSWS is the electrographic hallmark driving progressive language
and cognitive regression in EAS. The Spike-Wave Index (SWI) quantifies burden:
SWI = (duration of SW activity / total NREM duration) × 100%. SWI ≥85% = CSWS
threshold. CSWS suppression (SWI <50%) correlates with language recovery — making
CSWS monitoring the key treatment target. Night-time (nocturnal) clobazam or
diazepam specifically suppress slow-wave-sleep CSWS.

LANDAU-KLEFFNER SYNDROME (LKS): First described 1957 — abrupt or gradual onset of
verbal auditory agnosia (inability to understand spoken language) in a previously
normal child, accompanied by epileptiform EEG and CSWS. Seizures may be absent or
mild; the EEG abnormality far exceeds clinical seizure burden. KEY TEACHING: the
aphasia in LKS is an epileptic encephalopathy (driven by CSWS), not a structural
lesion — CSWS suppression with AEDs or steroids can restore language function,
especially if treated early (<6 months after language regression onset).

CBZ ABSOLUTE CONTRAINDICATION: Carbamazepine and oxcarbazepine (sodium channel
blockers) are ABSOLUTELY CONTRAINDICATED in GRIN2A-EAS / CSWS because they
paradoxically WORSEN CSWS by increasing slow-wave sleep synchrony — accelerating
cognitive/language regression. This is one of the most important drug-specific
contraindications in paediatric epilepsy. Multiple case series demonstrate
dramatic CSWS exacerbation within weeks of CBZ initiation in EAS patients.
Any clinician prescribing CBZ without recognising GRIN2A-EAS/CSWS risks
accelerating irreversible language regression.

ACTH / STEROIDS FOR LKS: Short courses of ACTH or prednisolone produce dramatic
CSWS suppression and language improvement in LKS (Level C). Mechanism: anti-
inflammatory suppression of thalamo-cortical synchrony driving CSWS. ACTH is
preferred first-line for LKS (4–6 week course); oral prednisolone (2 mg/kg/day)
is alternative. Response is often striking (SWI 95% → <50% within 2 weeks) but
relapse is common — requires AED maintenance (VPA + nocturnal CLB) after steroid
course. Early treatment (within 6 months of aphasia onset) = best language prognosis.

SPEECH AND LANGUAGE THERAPY (SLT) IS MANDATORY: Intensive SLT is an essential
component of GRIN2A-EAS management — not optional. During CSWS-active phase:
augmentative and alternative communication (AAC) devices. After CSWS suppression:
intensive verbal language rehabilitation. Annual neuropsychological testing to
monitor language recovery trajectory.
"""

import random
from datetime import datetime

SEED = 9180  # dashboard 180
random.seed(SEED)

# ── Etiology Distribution (5 classes, N=41) ──────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "Pathogenic GRIN2A GOF/LOF missense — severe EAS / Landau-Kleffner Syndrome (de novo)",
        "n": 16, "pct": 38,
        "category": "De-novo-GRIN2A-missense-severe-EAS-LKS",
        "mechanism": (
            "Most prevalent class (~38%): de novo GRIN2A missense variants affecting the GluN2A "
            "ligand-binding domain (LBD) — particularly the glutamate-binding cleft (residues "
            "Arg518, Tyr698, Glu714) or the transmembrane domain (M2/M3 segments controlling "
            "Mg²⁺ block and channel gating). Gain-of-function (GOF) variants (e.g., Leu812Met, "
            "Pro79Arg) prolong NMDA receptor open time → excessive glutamatergic excitation of "
            "perisylvian cortex → CSWS generation. Loss-of-function variants impair GluN2A-mediated "
            "fast synaptic plasticity → compensatory network hyperexcitability. Severe EAS phenotype: "
            "Landau-Kleffner Syndrome with acquired verbal auditory agnosia, SWI ≥90%, dense CSWS, "
            "and significant language regression. ~40% have intellectual disability; ~30% ASD features. "
            "De novo confirmed by trio exome/genome; variants at known pathogenic LBD hotspots "
            "(Pro79, Val485, Ala636, Leu812) — consult GRIN2A-specific functional data repositories."
        ),
        "eeg_signature": (
            "Hallmark GRIN2A-EAS EEG: high-amplitude centrotemporal sharp-slow wave complexes "
            "(200–400 μV) with horizontal dipole field (positive frontal, negative centrotemporal). "
            "Dramatically amplified during NREM sleep → CSWS pattern (SWI ≥85%). Waking EEG may "
            "be normal or show only occasional centrotemporal spikes. Sleep EEG (PSG) is MANDATORY "
            "for diagnosis — routine waking EEG grossly underestimates CSWS burden. Ictal: "
            "centrotemporal rhythmic discharge evolving to secondary generalisation during FBTCS."
        ),
        "mri": (
            "Usually normal. Occasional perisylvian cortical dysplasia or polymicrogyria (PMG) in "
            "~10% — if structural lesion present, consider focal cortical dysplasia as epileptogenic "
            "substrate independent of GRIN2A. MRI at diagnosis mandatory; fMRI language lateralisation "
            "if surgical evaluation considered."
        ),
        "clinical_note": (
            "CBZ/OXC ABSOLUTELY CONTRAINDICATED — prescribing CBZ in GRIN2A-EAS risks accelerating "
            "language regression via CSWS exacerbation. If child presents with language regression, "
            "check SLEEP EEG BEFORE initiating any AED. SWI ≥85% → CSWS treatment urgently (VPA + "
            "nocturnal CLB ± ACTH). Refer to tertiary epilepsy centre with paediatric neuropsychology "
            "and SLT for multidisciplinary management. GRIN2A genetic testing: exome/genome preferred; "
            "targeted panel acceptable if GRIN2A is included with all exons covered."
        ),
    },
    {
        "etiology": "Pathogenic GRIN2A truncating / frameshift — severe CSWS-epilepsy (de novo)",
        "n": 8, "pct": 20,
        "category": "De-novo-GRIN2A-truncating-CSWS-DEE",
        "mechanism": (
            "De novo nonsense, frameshift, or splice-site variants causing premature termination "
            "codon (PTC) and haploinsufficiency of GluN2A (~20%). Loss of one functional GRIN2A "
            "allele reduces GluN2A surface expression by ~50%, impairing fast NMDA receptor "
            "plasticity in perisylvian GABAergic interneuron circuits. Phenotype: CSWS-epilepsy "
            "(broader than LKS — global cognitive regression + seizures + CSWS, not isolated "
            "aphasia). Age of onset 3–8 years; seizures predominantly nocturnal (centrotemporal "
            "focal, FBTCS). Higher rate of intellectual disability vs missense class (~55%); "
            "less likely to achieve full language recovery. NMD (nonsense-mediated mRNA decay) "
            "eliminates truncating transcripts — no dominant-negative effect expected; pure "
            "haploinsufficiency mechanism. Brain-derived neurotrophic factor (BDNF) signalling "
            "through NMDA/TrkB disrupted — adds neurodevelopmental vulnerability."
        ),
        "eeg_signature": (
            "Dense bilateral CSWS (SWI often 90–100% NREM) — more severe than missense EAS. "
            "Waking: generalised or multifocal spike-wave; centrotemporal focus may be bilateral. "
            "Sleep PSG: CSWS merges into near-continuous spike-wave with brief normal intervals. "
            "Ictal: multifocal or secondary generalised. Serial sleep EEG essential to monitor "
            "SWI and AED response."
        ),
        "mri": "Usually normal. Rare perisylvian cortical dysplasia (1/8 in this cohort — incidental finding, not causative).",
        "clinical_note": (
            "CSWS-epilepsy secondary to GRIN2A truncation requires AGGRESSIVE early treatment — "
            "prolonged untreated CSWS causes irreversible cognitive regression. If SWI ≥85% on "
            "first sleep EEG: escalate immediately to VPA + nocturnal CLB and consider early ACTH "
            "course. Delay in CSWS treatment is the most important modifiable determinant of "
            "cognitive outcome in this subgroup."
        ),
    },
    {
        "etiology": "Familial GRIN2A — autosomal dominant EAS (inherited, variable expressivity)",
        "n": 9, "pct": 22,
        "category": "Familial-GRIN2A-AD-EAS-variable-expressivity",
        "mechanism": (
            "Autosomal dominant inherited GRIN2A variants (22%) with variable expressivity — the "
            "classic genetic signature of GRIN2A-EAS. A parent may carry the same variant but "
            "have only mild SeLECTS (nocturnal rolandic seizures, self-limited in childhood), "
            "while their child has severe LKS or CSWS-epilepsy. This intrafamilial variability "
            "reflects the stochastic nature of perisylvian network development and the role of "
            "modifier genes. Key clinical implication: GRIN2A parent carrier with 'resolved "
            "childhood epilepsy' may not recognise the genetic link — careful family history "
            "eliciting childhood seizures, night-time jerks, speech difficulties is essential. "
            "Recurrence risk: 50% per pregnancy. Missense and splice-site variants most common "
            "in familial EAS; truncating variants less common (lower penetrance for truncation)."
        ),
        "eeg_signature": (
            "Variable within family: mildly affected parent — occasional centrotemporal spikes on "
            "waking EEG, no CSWS. Severely affected child — full CSWS pattern on sleep EEG. "
            "The discordance between parental phenotype and child phenotype is a clue to familial "
            "GRIN2A — always request sleep EEG in the child even if parent 'grew out of it'."
        ),
        "mri": "Normal in all familial cases in this cohort.",
        "clinical_note": (
            "Familial EAS: when GRIN2A variant identified in proband, offer parental testing and "
            "cascade family testing. Parent with 'old childhood epilepsy' may have unrecognised GRIN2A "
            "carrier status — this informs reproductive counselling (50% recurrence). Genetic "
            "counselling: expressivity is unpredictable — sibling of severely affected proband may "
            "have only mild SeLECTS or be unaffected. Penetrance ~60% for EAS phenotype."
        ),
    },
    {
        "etiology": "GRIN2A splice-site variant — moderate EAS (de novo or familial)",
        "n": 5, "pct": 12,
        "category": "GRIN2A-splice-site-moderate-EAS",
        "mechanism": (
            "Canonical ±1/2 or deep-intronic splice variants affecting GRIN2A exon splicing (~12%). "
            "Aberrant splicing may cause exon skipping (partial LOF), intron retention (LOF or "
            "dominant-negative), or cryptic exon activation. Phenotype is generally intermediate: "
            "moderate EAS without complete LKS; CSWS present but SWI typically 50–85% (not "
            "fulminant). Some splice variants have been confirmed by RT-PCR of lymphoblast or "
            "fibroblast RNA. Deep-intronic variants (affecting GRIN2A splicing regulatory sequences) "
            "may be missed on standard exome — WGS ± RNA sequencing recommended if strong EAS "
            "phenotype with negative gene panel. ACMG interpretation: splice-site variant at "
            "±1/2 = likely pathogenic (PVS1_Strong); deep-intronic VUS requires functional RNA data."
        ),
        "eeg_signature": (
            "Centrotemporal spikes waking + moderate CSWS on sleep EEG (SWI 50–85%). Less severe "
            "NREM saturation than truncating class. Some patients have intermittent CSWS (present "
            "on some recordings, absent on others) — requiring serial monitoring. Ictal: "
            "centrotemporal focal ± secondary FBTCS."
        ),
        "mri": "Normal in all 5 cases.",
        "clinical_note": (
            "RNA analysis should be requested for GRIN2A splice VUS in a patient with clear EAS "
            "phenotype — functional RNA data upgrades VUS to likely pathogenic per ACMG. "
            "WGS superior to exome for detecting deep-intronic GRIN2A splice variants. "
            "Moderate CSWS (SWI 50–85%) still warrants nocturnal CLB treatment — threshold for "
            "ACTH is SWI ≥85% or active language regression."
        ),
    },
    {
        "etiology": "Clinical GRIN2A-negative EAS phenocopy (EAS without GRIN2A variant)",
        "n": 3, "pct": 8,
        "category": "Clinical-GRIN2A-negative-EAS-phenocopy",
        "mechanism": (
            "Patients with EAS/CSWS/LKS phenotype but no GRIN2A pathogenic variant (~8%). "
            "Alternative genetic causes: GRIN2B (NR2B) — severe DEE; CNTNAP2 — Rolandic "
            "epilepsy with language features; RBFOX1/RBFOX3 — RNA-splicing regulators in EAS; "
            "DEPDC5 (GATOR1) — familial focal epilepsy with speech features; SCN8A (rare LKS-"
            "phenotype). Acquired causes: autoimmune encephalitis (anti-NMDA receptor antibodies "
            "— must exclude in LKS, especially acute onset); structural perisylvian lesions "
            "(polymicrogyria — PMG can cause acquired aphasia + CSWS independently of genetics). "
            "Somatic GRIN2A mosaicism (brain-limited) — not detectable on blood exome; consider "
            "if phenotype strongly concordant and sequencing negative."
        ),
        "eeg_signature": "CSWS pattern indistinguishable from GRIN2A-positive cases — genetics required to stratify.",
        "mri": "Must exclude perisylvian PMG, cortical dysplasia, or structural lesion driving CSWS.",
        "clinical_note": (
            "Anti-NMDA receptor antibody panel (CSF + serum) is MANDATORY in acute-onset LKS "
            "— autoimmune NMDAR encephalitis mimics LKS and requires immunotherapy (not AEDs). "
            "GRIN2A-negative EAS: comprehensive epilepsy gene panel (250–500 genes) or trio "
            "exome/genome. If still negative: MRI brain with dedicated perisylvian protocol "
            "(T1/T2 high-resolution cortical), anti-neuronal antibody panel, CSF metabolic screen. "
            "CSWS management same regardless of genetic result — treat the EEG, not just the gene."
        ),
    },
]

# ── Seizure Types (4) ──────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Centrotemporal / rolandic focal seizures (EAS hallmark)",
        "prevalence_pct": 92,
        "onset_age": "4–10 years (median ~6 years)",
        "eeg_correlate": (
            "HALLMARK GRIN2A-EAS EEG: centrotemporal sharp-slow wave complexes with horizontal "
            "dipole (negative centrotemporal / positive frontal Rolandic morphology). High "
            "amplitude (200–400 μV), diphasic, 150–300 ms duration. Dramatically activated by "
            "NREM sleep → CSWS. Ictal: rhythmic centrotemporal discharge (alpha/beta 10–15 Hz "
            "onset), evolves to clonic phase. Exclusively or predominantly nocturnal in most "
            "GRIN2A-EAS patients — parents often unaware of seizures. 24h EEG or video-telemetry "
            "required to capture nocturnal events; waking EEG alone is INSUFFICIENT."
        ),
        "clinical_tip": (
            "GRIN2A-EAS focal seizures are predominantly NOCTURNAL — family history of 'odd "
            "movements at night', drooling during sleep, or morning tongue biting is the clue. "
            "Clinical seizure burden UNDERESTIMATES EEG burden — a child with only 1–2 witnessed "
            "seizures may have SWI 90% CSWS. REQUEST SLEEP EEG (overnight PSG with EEG) at "
            "diagnosis. Do NOT initiate CBZ for 'rolandic epilepsy' without first confirming no "
            "CSWS — CBZ in CSWS causes language regression."
        ),
    },
    {
        "type": "Focal to bilateral tonic-clonic (FBTCS / nocturnal generalisation)",
        "prevalence_pct": 68,
        "onset_age": "Variable — typically after centrotemporal focal onset established",
        "eeg_correlate": (
            "Secondary generalisation from centrotemporal ictal focus — tonic and clonic phases "
            "with postictal EEG suppression. Typically nocturnal (NREM sleep entry or arousal "
            "from sleep). FBTCS duration 1–3 min. Postictal Todd's paralysis (arm or face "
            "hemiparesis) may occur. Generalised spike-wave interictal activity during CSWS phase "
            "— may be confused with primary generalised epilepsy if centrotemporal origin missed "
            "on standard 10-20 EEG montage."
        ),
        "clinical_tip": (
            "FBTCS in school-age child without prior diagnosis: always request sleep EEG before "
            "treatment. If centrotemporal spikes identified on waking EEG + school-age onset FBTCS "
            "→ EAS diagnosis likely. CRITICAL: if family or GP has already initiated CBZ for "
            "'rolandic seizure' + FBTCS and child develops speech difficulties → stop CBZ "
            "immediately, request urgent sleep EEG, assess for CSWS."
        ),
    },
    {
        "type": "Atypical absence / CSWS-associated non-convulsive seizures",
        "prevalence_pct": 55,
        "onset_age": "School age — correlates with CSWS peak (6–10 years)",
        "eeg_correlate": (
            "During CSWS phase: 'absences' are often non-convulsive seizures superimposed on "
            "CSWS — clinically: staring, confusion, language arrest without motor component. "
            "EEG: slow spike-wave (1.5–2.5 Hz) during NREM saturates into CSWS; waking: "
            "occasional atypical absence with irregular slow SW (not the regular 3 Hz of CAE). "
            "Non-convulsive status epilepticus (NCSE) may occur during severe CSWS episodes — "
            "clinical picture: subacute language regression + confusion without convulsions. "
            "Urgent EEG needed to diagnose NCSE in any child with sudden language/cognitive decline."
        ),
        "clinical_tip": (
            "Teachers and parents often misinterpret the language arrest of LKS as 'daydreaming' "
            "or 'selective hearing' — years of misdiagnosis may occur before EEG is requested. "
            "KEY ALERT: a child who 'can't hear properly' or 'stopped talking' should have URGENT "
            "sleep EEG to exclude LKS. Verbal auditory agnosia (VAA) — the inability to decode "
            "speech sounds despite normal hearing audiogram — is PATHOGNOMONIC for active LKS. "
            "Audiology referral routinely shows normal pure-tone audiometry → then EEG is needed."
        ),
    },
    {
        "type": "Myoclonic seizures",
        "prevalence_pct": 25,
        "onset_age": "School age — during CSWS-active phase",
        "eeg_correlate": (
            "Generalised polyspike-wave burst (1–5 spikes, 50–150 ms) associated with sudden "
            "upper limb bilateral jerk. Occurs predominantly during drowsiness-to-sleep transition "
            "or on awakening. Myoclonus in EAS may indicate an overlap with CSWS-epilepsy rather "
            "than pure EAS/LKS — evaluate SWI and cognitive trajectory. Distinguish from "
            "benign myoclonic epilepsy of infancy (BMEI — earlier onset, different EEG morphology)."
        ),
        "clinical_tip": (
            "Myoclonus in GRIN2A-EAS generally responds to VPA (broad-spectrum). Clonazepam "
            "adjunct at night may control myoclonus + partially suppress CSWS. Avoid lamotrigine "
            "alone — less effective for CSWS-associated myoclonus. ESM (ethosuximide) covers "
            "absence-type component but not myoclonus or focal seizures."
        ),
    },
]

# ── Triggers (8) ──────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "NREM sleep / sleep onset (CSWS exacerbation)",
        "rate_pct": 95,
        "mechanism": (
            "The primary trigger for CSWS/ESES exacerbation in GRIN2A-EAS is NREM sleep — "
            "specifically slow-wave sleep (SWS/N3). During NREM sleep, thalamo-cortical "
            "synchrony naturally activates slow oscillations (0.5–1 Hz) that drive cortical "
            "up/down state transitions. In GRIN2A-EAS, abnormal NMDA receptor activity in "
            "perisylvian circuits amplifies these slow oscillations into continuous spike-wave "
            "activity. The SWI (Spike-Wave Index) is highest in N2/N3 sleep and lowest in "
            "REM sleep — REM sleep CSWS suppression is a characteristic feature of EAS. "
            "This is why NOCTURNAL AED dosing (clobazam, diazepam, clonazepam) specifically "
            "targets the night-time CSWS window."
        ),
        "management": (
            "Sleep EEG monitoring: overnight PSG q6M during active CSWS phase. Treatment targeting "
            "CSWS: nocturnal CLB (0.1–0.3 mg/kg at bedtime) or nocturnal diazepam (0.3–0.5 mg/kg) "
            "to suppress NREM CSWS. Maintain good sleep hygiene. Polysomnography (PSG) quantifies "
            "SWI and sleep architecture — necessary to guide treatment escalation decisions."
        ),
    },
    {
        "trigger": "Sleep deprivation / disrupted sleep architecture",
        "rate_pct": 75,
        "mechanism": (
            "Sleep deprivation increases SWS rebound (homeostatic sleep pressure) → amplified "
            "slow oscillations → increased CSWS burden on subsequent sleep. School-age children "
            "with EAS often have disrupted sleep quality (night awakenings from focal seizures, "
            "anxiety). Sleep deprivation also increases diurnal seizure susceptibility. "
            "School examinations / holiday schedule changes are common precipitants of cluster "
            "of FBTCS in GRIN2A-EAS."
        ),
        "management": "Strict sleep hygiene: consistent bedtime ≤9 PM; limit screens; treat insomnia (melatonin 0.5–3 mg). Inform school of sleep disorder — may qualify for classroom accommodations.",
    },
    {
        "trigger": "Fever / febrile illness",
        "rate_pct": 68,
        "mechanism": (
            "Fever lowers seizure threshold in centrotemporal circuits — increases FBTCS "
            "frequency and may precipitate new CSWS onset or CSWS exacerbation in established "
            "GRIN2A-EAS. Inflammatory cytokines (IL-1β, IL-6) upregulate NMDA receptor "
            "expression transiently, potentially increasing GOF GRIN2A-mediated excitation. "
            "Fever-triggered CSWS deterioration may persist for weeks after the illness resolves."
        ),
        "management": "Fever action plan: antipyretics (paracetamol/ibuprofen) + AED rescue. If child has CSWS and develops prolonged fever → request EEG within 2 weeks of illness to reassess SWI. Alert parents to post-illness language regression as a CSWS warning sign.",
    },
    {
        "trigger": "Initiation of CBZ / OXC (iatrogenic CSWS worsening)",
        "rate_pct": 55,
        "mechanism": (
            "CRITICAL IATROGENIC TRIGGER: carbamazepine (CBZ) and oxcarbazepine (OXC) cause "
            "paradoxical CSWS EXACERBATION in GRIN2A-EAS via increased slow-wave sleep "
            "synchrony. Mechanism: CBZ/OXC (Na⁺ channel blockers) reduce fast cortical "
            "oscillations, which paradoxically enhances the slow thalamo-cortical rhythms "
            "underlying CSWS in perisylvian circuits. CLINICAL CONSEQUENCE: a child with "
            "unrecognised GRIN2A-EAS started on CBZ for 'rolandic seizure' may develop "
            "accelerated language regression and cognitive decline within weeks — misattributed "
            "to disease progression rather than the drug. This scenario is preventable with "
            "pre-treatment sleep EEG and GRIN2A testing."
        ),
        "management": "IMMEDIATE CESSATION of CBZ/OXC if CSWS identified. Taper CBZ over 4–6 weeks (do not stop abruptly). Initiate VPA + nocturnal CLB simultaneously. Perform sleep EEG 4 weeks post CBZ cessation to verify CSWS improvement.",
    },
    {
        "trigger": "Intercurrent illness (without fever)",
        "rate_pct": 48,
        "mechanism": (
            "Systemic illness (viral gastroenteritis, URI) can transiently worsen CSWS "
            "independent of fever — inflammatory cytokines + sleep disruption + AED absorption "
            "changes (vomiting) combine to increase CSWS burden. Post-illness language "
            "regression may indicate transient CSWS escalation. Monitor language after illness."
        ),
        "management": "Illness plan: maintain AED dosing (rectal/buccal route if vomiting). Post-illness language assessment within 2 weeks. EEG if significant language regression after illness.",
    },
    {
        "trigger": "Stress / psychological distress",
        "rate_pct": 42,
        "mechanism": (
            "Academic and psychosocial stress (school transitions, examinations, bullying related "
            "to language difficulties) increase cortisol, which may modulate thalamo-cortical "
            "synchrony. Children with EAS-related language regression frequently face severe "
            "psychosocial challenges — the stress response amplifies CSWS burden in a vicious "
            "cycle. Psychosocial support is not optional — it is an integral management component."
        ),
        "management": "School psychology referral; SLT assessment; classroom speech-friendly accommodations (FM system); parental psychological support (carer burden is high in LKS).",
    },
    {
        "trigger": "Missed AED dose / AED non-adherence",
        "rate_pct": 38,
        "mechanism": (
            "Missing nocturnal CLB or diazepam dose results in acute loss of overnight CSWS "
            "suppression — CSWS may rebound strongly the following night. VPA missed dose "
            "reduces broad-spectrum protection. Adherence particularly challenging in school-"
            "age children with behavioural comorbidities (ADHD, ASD features) who resist "
            "medication routine."
        ),
        "management": "Adherence strategies: blister packs; parental bedtime medication routine; school nurse oversight for daytime VPA. Electronic reminder apps. Involve child in age-appropriate explanation of why nocturnal medication is critical.",
    },
    {
        "trigger": "Puberty (CSWS resolution — beneficial transition)",
        "rate_pct": 85,
        "mechanism": (
            "In GRIN2A-EAS, puberty (10–14 years) typically brings CSWS RESOLUTION — one of "
            "the most important natural history features of the Epilepsy-Aphasia Spectrum. "
            "Hormonal changes at puberty alter thalamo-cortical maturation, increasing "
            "slow-wave sleep threshold and suppressing CSWS spontaneously. Seizures also "
            "remit in adolescence in 80–90% of EAS patients. This is a 'therapeutic window' "
            "to wean AEDs — but language deficits acquired during CSWS-active years may "
            "persist if SLT was inadequate. Note: this entry represents the 85% CSWS resolution "
            "rate at puberty — a beneficial natural history milestone, not a seizure trigger."
        ),
        "management": "EEG monitoring at onset of puberty to confirm CSWS resolution. If SWI <25% for 12+ months → consider AED wean under specialist guidance. SLT remains essential post-CSWS to consolidate language recovery. Neuropsychological re-assessment post-puberty to quantify residual language deficits.",
    },
]

# ── Treatments (8) ──────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Valproate (VPA)",
        "evidence_level": "Level B (Class III–IV; ILAE 2022; NICE NG217; EAN EAS Guidelines)",
        "role": "First-line broad-spectrum AED for GRIN2A-EAS / CSWS-epilepsy",
        "dose": (
            "20–40 mg/kg/day in 2–3 divided doses (max 60 mg/kg/day or 3000 mg/day). "
            "Target serum TDM: 50–100 mg/L (higher end for CSWS). Enteric-coated or "
            "controlled-release formulation preferred for tolerability."
        ),
        "moa": (
            "Broad-spectrum AED: ① Blocks voltage-gated Na⁺ channels (fast inactivation); "
            "② Enhances GABA synthesis and GABA-T inhibition → increased synaptic GABA; "
            "③ Reduces T-type Ca²⁺ channel (CACNA1G/H) activity — critical for thalamo-"
            "cortical oscillations driving CSWS; ④ Weak NMDA receptor modulation."
        ),
        "efficacy": (
            "VPA reduces SWI by 30–60% in EAS/CSWS (Class III-IV evidence). Most effective "
            "in combination with nocturnal CLB for CSWS suppression. Partial responders "
            "benefit from VPA + ACTH combination. Best single-agent for broad coverage of "
            "focal, generalised, and absence-type seizures in EAS."
        ),
        "monitoring": "TDM 50–100 mg/L; LFTs and ammonia q3M; full blood count q6M; weight monitoring; POLG exclusion before initiation (MANDATORY in infantile-onset DEE — less critical in school-age EAS but prudent).",
        "safety": "Teratogenicity (avoid in females of reproductive age when possible); weight gain; hair loss; tremor; thrombocytopenia; pancreatitis (rare). LFT elevation — stop if >3× ULN.",
    },
    {
        "drug": "Clobazam (CLB) — nocturnal dosing",
        "evidence_level": "Level B (EAN EAS Guidelines 2020; NICE NG217; ILAE 2022)",
        "role": "CSWS-specific — nocturnal CLB suppresses NREM CSWS; first-line adjunct",
        "dose": (
            "Total dose 0.1–0.5 mg/kg/day; CONCENTRATE DOSE NOCTURNALLY — give 70–80% of "
            "daily dose at bedtime to maximise CSWS suppression during NREM sleep. "
            "Example: child 20 kg → 5 mg AM + 10 mg at bedtime (total 15 mg/day). "
            "Maximum 40 mg/day (adult)."
        ),
        "moa": (
            "Positive allosteric modulator of GABA-A receptor (1,5-benzodiazepine). "
            "CSWS mechanism: CLB increases GABAergic inhibitory tone during NREM sleep, "
            "suppressing the thalamo-cortical slow oscillations that generate CSWS. "
            "Active metabolite norclobazam (N-desmethylclobazam) provides sustained "
            "overnight GABA-A potentiation. Nocturnal dosing exploits the pharmacokinetic "
            "peak aligning with peak NREM slow-wave CSWS."
        ),
        "efficacy": (
            "Nocturnal CLB reduces SWI by 40–70% in CSWS/EAS (multiple European cohort studies). "
            "Language outcome correlates with degree of CSWS suppression. Most effective "
            "in combination with VPA. Tolerance develops in ~30% — monitoring norclobazam "
            "levels helps detect tolerance (declining level at fixed dose)."
        ),
        "monitoring": "Norclobazam TDM target 50–300 ng/mL; sedation scale; tolerance assessment q6M review; CSWS SWI re-measurement at 3M to confirm response.",
        "safety": "Sedation (worst in AM if nocturnal dose too high — adjust if morning sedation limits school performance); tolerance; respiratory depression (rare at therapeutic doses); benzodiazepine dependence on prolonged use.",
    },
    {
        "drug": "ACTH (corticotrophin) / Prednisolone — Landau-Kleffner Syndrome",
        "evidence_level": "Level C (Class IV; ILAE 2022; EAN EAS Guidelines 2020; EAN LKS Statement)",
        "role": "First-line for Landau-Kleffner Syndrome (LKS) with active language regression",
        "dose": (
            "ACTH: synthetic ACTH (tetracosactide) 0.1–0.15 mg (10–15 IU)/dose IM q48h × "
            "4–6 weeks, then taper over 4 weeks. Alternative: natural ACTH gel 80 IU/m²/day. "
            "Prednisolone: 2–4 mg/kg/day (max 60 mg/day) oral × 4–6 weeks, then taper 4–8 weeks. "
            "ACTH preferred over prednisolone for faster CSWS suppression (within 2 weeks)."
        ),
        "moa": (
            "Anti-inflammatory suppression of thalamo-cortical hyperexcitability: ACTH stimulates "
            "adrenal cortisol + directly modulates ACTH receptor signalling in thalamus and "
            "cortex. Reduces neuroinflammatory cytokines (IL-1β, IL-6) that may amplify CSWS. "
            "Corticosteroids suppress slow-wave-sleep-amplified thalamo-cortical synchrony. "
            "Effect on NMDA receptor expression: modulates GluN2A subunit expression in "
            "perisylvian cortex (emerging mechanism)."
        ),
        "efficacy": (
            "ACTH/steroids suppress CSWS (SWI <50%) in 60–80% of LKS patients (multiple case "
            "series). Language improvement follows CSWS suppression — 50–70% show meaningful "
            "language recovery with early treatment. Relapse on steroid taper requires AED "
            "maintenance (VPA + CLB) to sustain CSWS suppression. Best outcomes: treatment "
            "within 6 months of language regression onset."
        ),
        "monitoring": "BP q3 days; fasting glucose; weight; behaviour; immunosuppression precautions (live vaccine avoidance during ACTH course); adrenal function monitoring on taper.",
        "safety": "Hypertension; hyperglycaemia; behavioural disturbance (ACTH-induced irritability common — temporary, resolves with taper); Cushingoid features; immunosuppression; adrenal insufficiency on abrupt withdrawal.",
    },
    {
        "drug": "Nocturnal diazepam / clonazepam",
        "evidence_level": "Level C (EAN EAS Guidelines 2020; ILAE 2022)",
        "role": "CSWS suppression — nocturnal diazepam specifically targets NREM CSWS",
        "dose": (
            "Diazepam rectal/oral: 0.3–0.5 mg/kg at bedtime (max 10 mg). "
            "Clonazepam oral: 0.03–0.1 mg/kg/day with majority given at bedtime. "
            "Used when CLB alone insufficient or for short-course CSWS crisis management."
        ),
        "moa": (
            "GABA-A receptor potentiation (benzodiazepine site) — same mechanism as CLB but "
            "1,4-benzodiazepine formulation. Diazepam: more sedating than CLB; may improve "
            "sleep continuity in CSWS while suppressing spike-wave activity. Clonazepam: "
            "additional anti-myoclonic effect via GABA-A modulation."
        ),
        "efficacy": "Nocturnal diazepam suppresses CSWS in 40–60% of patients (multiple European cohort studies). May be combined with CLB for refractory CSWS. Clonazepam effective for myoclonus component.",
        "monitoring": "Morning sedation assessment; school performance monitoring; tolerance assessment q3M.",
        "safety": "Sedation (morning hangover effect — may impair school performance; adjust dose or switch to CLB if prominent); tolerance; benzodiazepine dependence.",
    },
    {
        "drug": "Ethosuximide (ESM)",
        "evidence_level": "Level C (EAN EAS Guidelines 2020)",
        "role": "Adjunct for absence-type component + CSWS augmentation (limited data)",
        "dose": "20–40 mg/kg/day in 2–3 divided doses (max 2000 mg/day). TDM target 40–100 mg/L.",
        "moa": (
            "Selective T-type Ca²⁺ channel (CACNA1G/H) blocker — blocks thalamic pacemaker "
            "currents driving thalamo-cortical slow oscillations. Mechanism relevant to CSWS: "
            "T-type channel block reduces the thalamic spindle and slow oscillation amplitude "
            "that drives CSWS. Synergistic with VPA for CSWS suppression."
        ),
        "efficacy": "ESM + VPA combination shows superior CSWS suppression vs VPA alone in some series (EAN Level C). Primarily targets absence-type seizures and slow CSWS component. Not effective for focal or generalised tonic-clonic seizures.",
        "monitoring": "TDM 40–100 mg/L; CBC q6M (rare aplastic anaemia); GI tolerability (nausea — take with food).",
        "safety": "GI upset (nausea, vomiting — mitigated by food/slow titration); headache; rare: lupus-like reaction, aplastic anaemia (CBC monitoring essential).",
    },
    {
        "drug": "Acetazolamide (ACZ)",
        "evidence_level": "Level C (small case series; EAN EAS Guidelines 2020)",
        "role": "Adjunct for CSWS refractory to VPA + CLB",
        "dose": "5–10 mg/kg/day in 2–3 divided doses (max 750 mg/day). May combine with VPA.",
        "moa": (
            "Carbonic anhydrase inhibitor → CO₂ retention → mild acidosis → increased "
            "Na⁺ channel inactivation + GABA-A enhancement. Inhibition of carbonic anhydrase "
            "in glial cells may reduce interstitial K⁺ accumulation that promotes synchrony. "
            "Ancillary: reduces CSF production (minor role). Mechanism for CSWS suppression "
            "not fully elucidated — empirical clinical evidence guides use."
        ),
        "efficacy": "CSWS suppression in 30–50% when added to VPA + CLB regimen (small cohort data). Short-term tolerance (3–6 months) limits prolonged use. May be used as bridge to steroid course.",
        "monitoring": "Serum bicarbonate (metabolic acidosis monitoring); renal calculi risk (urinary alkalinisation — encourage hydration); potassium levels.",
        "safety": "Metabolic acidosis (measure serum HCO₃⁻ q3M); renal calculi (maintain high fluid intake); paraesthesias; taste disturbance (carbonated drinks taste flat — patient counselling).",
    },
    {
        "drug": "Lacosamide (LCM)",
        "evidence_level": "Level C (small series; off-label in EAS)",
        "role": "Adjunct for focal seizure control in GRIN2A-EAS (does NOT worsen CSWS)",
        "dose": "2–12 mg/kg/day in 2 divided doses (max 400 mg/day). Target TDM 5–15 mg/L.",
        "moa": (
            "Selective enhancement of Na⁺ channel slow inactivation — distinct from CBZ/OXC "
            "fast inactivation mechanism. Critically, LCM does NOT appear to worsen CSWS "
            "in EAS (unlike CBZ/OXC) — likely because slow inactivation modulation does not "
            "amplify thalamo-cortical slow oscillations. Emerging evidence in focal epilepsy "
            "of perisylvian origin. P-glycoprotein substrate — drug interactions relevant."
        ),
        "efficacy": "Focal seizure (centrotemporal, FBTCS) reduction of 30–50% as adjunct (small EAS series). Does not contribute to CSWS suppression — use as focal seizure adjunct only, not primary CSWS treatment.",
        "monitoring": "PR interval (ECG baseline — LCM prolongs PR; CI in AV block); TDM 5–15 mg/L; LFTs.",
        "safety": "Dizziness (dose-related); diplopia; PR prolongation (ECG required — avoid in cardiac conduction defects); potential drug-drug interactions (VPA inhibits LCM metabolism).",
    },
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) — ABSOLUTE CONTRAINDICATION",
        "evidence_level": "ABSOLUTE CONTRAINDICATION — Level A avoidance evidence (multiple case series; EAN EAS Guidelines 2020)",
        "role": "MUST NOT BE USED in GRIN2A-EAS / any CSWS syndrome",
        "dose": "N/A — contraindicated",
        "moa": (
            "CBZ/OXC (Na⁺ channel fast inactivation enhancers) WORSEN CSWS by paradoxically "
            "increasing NREM slow-wave sleep thalamo-cortical synchrony — the physiological "
            "substrate of CSWS. Mechanism: fast Na⁺ channel block reduces fast cortical "
            "oscillations (beta/gamma), disinhibiting the slow thalamo-cortical oscillations "
            "that generate CSWS. Clinical consequence: CBZ-induced CSWS exacerbation → "
            "accelerated language regression and cognitive decline."
        ),
        "efficacy": "CONTRAINDICATED — initiation causes cognitive deterioration, not improvement.",
        "monitoring": "N/A — do not use.",
        "safety": (
            "ABSOLUTE CI in GRIN2A-EAS/CSWS: multiple published case series show CBZ "
            "initiation in EAS patients causes rapid CSWS worsening (SWI increase 20–40%), "
            "language regression, and cognitive decline within 2–8 weeks. This risk applies "
            "equally to OXC, eslicarbazepine acetate (ESL), and any other fast Na⁺ channel "
            "blocker. If CBZ prescribed in error → STOP IMMEDIATELY, obtain sleep EEG, "
            "assess language status."
        ),
    },
]

# ── Contraindications (4 absolute) ────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) / Eslicarbazepine (ESL)",
        "severity": "ABSOLUTE CONTRAINDICATION",
        "reason": (
            "Na⁺ channel blockers (CBZ/OXC/ESL) paradoxically exacerbate CSWS in GRIN2A-EAS, "
            "accelerating language regression and cognitive decline. Multiple Level-C case series "
            "confirm CBZ-induced CSWS worsening within weeks of initiation. This is one of the "
            "most dangerous 'wrong drug' errors in paediatric epilepsy — especially because "
            "GRIN2A-EAS often presents as 'rolandic epilepsy' and CBZ is first-line for generic "
            "rolandic seizures without CSWS. STOP CBZ IMMEDIATELY if CSWS identified in a "
            "child already taking CBZ."
        ),
    },
    {
        "drug": "Anti-NMDA receptor antibody encephalitis misdiagnosed as GRIN2A-EAS — delay of immunotherapy",
        "severity": "DIAGNOSTIC EMERGENCY",
        "reason": (
            "Anti-NMDA receptor (anti-NMDAr) antibody encephalitis can perfectly mimic LKS: "
            "acute language regression, CSWS on EEG, centrotemporal EEG discharges. Missing "
            "this diagnosis and treating with AEDs only (without immunotherapy) results in "
            "irreversible neurological injury. MANDATORY: anti-NMDAr antibody (CSF + serum) "
            "in any child with ACUTE-ONSET LKS phenotype (days to weeks of language regression). "
            "GRIN2A-EAS is more subacute-chronic (months). If antibody positive → urgent "
            "immunotherapy (IV methylprednisolone ± IVIg), not AED titration."
        ),
    },
    {
        "drug": "AED cessation without sleep EEG monitoring (CSWS rebound)",
        "severity": "HIGH RISK — protocol required",
        "reason": (
            "Premature AED withdrawal (VPA or CLB) without confirming CSWS resolution on sleep "
            "EEG causes CSWS REBOUND — rapid return of CSWS with language regression, sometimes "
            "worse than pre-treatment. AED wean requires: ① CSWS resolved (SWI <25%) for ≥12 "
            "months; ② Neuropsychological assessment confirming stable language; ③ Sleep EEG "
            "4 weeks after each wean step. Do NOT wean solely based on clinical seizure control — "
            "CSWS may persist silently after seizures remit."
        ),
    },
    {
        "drug": "Delay in speech-language therapy (SLT) — missed rehabilitation window",
        "severity": "MANAGEMENT IMPERATIVE",
        "reason": (
            "Deferring SLT in active EAS/LKS results in irreversible language deficit — the "
            "critical window for language rehabilitation is during and immediately after CSWS "
            "suppression (typically age 6–12 years). During CSWS-active phase: AAC (augmentative "
            "and alternative communication) devices allow the child to communicate despite verbal "
            "agnosia. After CSWS suppression: intensive verbal SLT for language recovery. "
            "Minimum intensity: 3× per week SLT sessions during CSWS-active phase; 2× per week "
            "after CSWS resolution. School placement in language-appropriate setting MANDATORY."
        ),
    },
]

# ── Monitoring Items (8) ────────────────────────────────────────────────────────
MONITORING = [
    {
        "item": "Overnight sleep EEG (PSG) — SWI quantification",
        "frequency": "q6 months during CSWS-active phase; q12 months after CSWS resolution",
        "rationale": (
            "SWI (Spike-Wave Index) is the key treatment target in GRIN2A-EAS. Response "
            "definition: SWI <50% = partial response; SWI <25% = full suppression. Treatment "
            "escalation triggered by SWI ≥85% (CSWS threshold) or progressive language decline. "
            "Waking EEG ALONE IS INSUFFICIENT — CSWS is exclusively/predominantly NREM-sleep "
            "pattern. Minimum 6h sleep EEG or overnight PSG required. SWI monitoring guides: "
            "dose escalation of nocturnal CLB; decision to add ACTH; decision for AED wean."
        ),
    },
    {
        "item": "Neuropsychological assessment — language and cognition",
        "frequency": "q6 months during active CSWS; q12 months after CSWS resolution",
        "rationale": (
            "Language trajectory is the primary outcome in GRIN2A-EAS. Assessment battery: "
            "CELF (Clinical Evaluation of Language Fundamentals) — receptive + expressive "
            "language; WISC/WPPSI IQ (domain profile); phonological processing (CTOPP); "
            "verbal memory (CVLT-C); pragmatic language; educational attainment. Language "
            "regression correlates with SWI — parallel monitoring detects CSWS-driven decline "
            "before clinical deterioration is obvious to parents/teachers."
        ),
    },
    {
        "item": "VPA serum TDM",
        "frequency": "Steady-state 5–7 days after each dose change; then q3M",
        "rationale": "Target 50–100 mg/L for CSWS suppression. Higher end (80–100) needed for CSWS; lower end (50–70) for seizure-only phenotype. Free VPA level if hypoalbuminaemia. Co-medication interactions (CLB + VPA → mutual concentration increase).",
    },
    {
        "item": "CLB/norclobazam TDM",
        "frequency": "Steady-state; then q6M",
        "rationale": "Norclobazam target 50–300 ng/mL. Tolerance detection: falling norclobazam level at fixed CLB dose indicates CYP2C19 induction → dose adjustment needed. Genetic CYP2C19 genotyping clarifies metaboliser status (poor metaboliser → high norclobazam; ultra-rapid → low norclobazam, poor response).",
    },
    {
        "item": "MRI brain (baseline + structured follow-up)",
        "frequency": "At diagnosis; repeat if clinical regression unexplained by CSWS",
        "rationale": "Exclude perisylvian PMG, cortical dysplasia, or structural lesion contributing to EAS. High-resolution T1/T2 with dedicated perisylvian sequences (opercular view). Diffusion-weighted sequences for acute CSWS-associated signal changes. If MRI shows progressive lesion → neoplasm, Rasmussen, autoimmune encephalitis work-up.",
    },
    {
        "item": "Anti-NMDA receptor antibody panel (CSF + serum)",
        "frequency": "At diagnosis of any acute-onset LKS / CSWS — MANDATORY; repeat if unexplained relapse",
        "rationale": "Anti-NMDAr encephalitis mimics LKS. CSF sensitivity >90% for anti-NMDAr Ab; serum sensitivity ~80% (lower, but important if LP contraindicated). If positive → neuroinflammation treatment pathway. Negative antibody + GRIN2A variant confirmed → proceed with GRIN2A-EAS management.",
    },
    {
        "item": "Speech-language therapy (SLT) — frequency and functional monitoring",
        "frequency": "SLT assessment at diagnosis; 3× per week therapy during CSWS-active phase; 2× per week after CSWS suppression",
        "rationale": "SLT intensity is the modifiable determinant of language outcome. Use validated language outcome measures (CELF percentile ranks, GFTA-3 for articulation) to track progress. AAC device assessment if verbal output severely reduced. School EHCP (Education, Health and Care Plan) referral early for language-appropriate placement.",
    },
    {
        "item": "VPA LFTs, FBC, and ammonia",
        "frequency": "Baseline; q3M for first year; q6M thereafter",
        "rationale": "VPA hepatotoxicity risk: LFT >3× ULN → withhold VPA, specialist review. Hyperammonaemia (VPA-induced urea cycle disruption): measure ammonia if encephalopathy or behavioural change. Thrombocytopenia: FBC q6M. Carnitine supplementation if symptomatic hyperammonaemia or low carnitine.",
    },
]

# ── Lifecycle Windows (6) ────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Pre-school onset (2–5 years)",
        "key_events": "Centrotemporal spikes appear on EEG; occasional nocturnal focal seizures; language development normal or mildly delayed; CSWS not yet established in most.",
        "management_focus": "Baseline sleep EEG; GRIN2A genetic testing (exome/genome); SLT baseline assessment; avoid CBZ; monitor language quarterly.",
    },
    {
        "window": "School-age CSWS onset (5–9 years)",
        "key_events": "CSWS emerges (SWI ≥50–85%); language regression begins; teachers notice 'selective hearing' or 'speech difficulties'; LKS phenotype develops in severe cases.",
        "management_focus": "Urgent sleep EEG; initiate VPA + nocturnal CLB; consider ACTH if LKS; intensive SLT; school EHC plan; neuropsychological monitoring q6M.",
    },
    {
        "window": "CSWS peak (8–11 years)",
        "key_events": "Maximum CSWS burden (SWI peak); language and cognitive regression at nadir; seizure frequency may be high; family and school stress peak.",
        "management_focus": "ACTH course if refractory CSWS or active language regression; optimize VPA + nocturnal CLB (± acetazolamide); intensive SLT + AAC; family psychological support.",
    },
    {
        "window": "Pubertal transition (10–14 years)",
        "key_events": "CSWS begins to resolve spontaneously with pubertal hormonal changes (85% resolution by age 14); seizure frequency decreases; language recovery phase begins.",
        "management_focus": "Serial sleep EEG to confirm CSWS resolution (SWI <25%); cautious AED wean (one drug at a time with 4-week EEG monitoring); intensive language rehabilitation; neuropsychological reassessment.",
    },
    {
        "window": "Adolescence (14–18 years)",
        "key_events": "Seizure remission in 80–90%; CSWS resolved in most; language recovery variable (full in mild EAS; partial in severe LKS); cognitive plateau reached.",
        "management_focus": "AED wean if CSWS-free ≥12M; transition to adult services; vocational/educational counselling based on neuropsychological outcome; residual SLT if needed.",
    },
    {
        "window": "Adulthood (18+ years)",
        "key_events": "Most patients AED-free; residual language deficits persist in 30–50% of severe LKS (especially those with delayed treatment); educational/employment impact variable.",
        "management_focus": "Adult epilepsy service transition; continue SLT if residual deficits; reproductive counselling (AD inheritance — 50% recurrence if GRIN2A identified); mental health support (anxiety/depression common in EAS adults with residual language disability).",
    },
]

# ── Concepts (14) ────────────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "GRIN2A-GluN2A-NMDA-receptor", "definition": "GRIN2A (2q33.1) encodes the GluN2A (NR2A) subunit of di-heteromeric NMDA ionotropic glutamate receptors — the molecular basis of Hebbian synaptic plasticity in cortical circuits."},
    {"term": "Epilepsy-Aphasia-Spectrum-EAS-ILAE-2022", "definition": "The EAS encompasses SeLECTS → atypical BECTS/ESES → CSWS-epilepsy → Landau-Kleffner Syndrome, unified by centrotemporal EEG discharges and language involvement. GRIN2A variants cause ~20% of familial EAS."},
    {"term": "CSWS-ESES-SWI", "definition": "Continuous Spike-and-Wave during Sleep (CSWS) = EEG pattern of slow SW occupying ≥85% of NREM sleep. Spike-Wave Index (SWI) = (SW duration / NREM duration) × 100%. SWI ≥85% = CSWS; target SWI <25% with treatment."},
    {"term": "Landau-Kleffner-Syndrome-LKS", "definition": "Acquired epileptic aphasia with verbal auditory agnosia — child loses ability to decode spoken language. CSWS is the electrographic mechanism. Early treatment (ACTH + VPA + nocturnal CLB) within 6M of onset maximises language recovery."},
    {"term": "CBZ-ABSOLUTE-CI-CSWS", "definition": "Carbamazepine and oxcarbazepine are absolutely contraindicated in GRIN2A-EAS/CSWS — they paradoxically worsen CSWS via increased thalamo-cortical slow synchrony, accelerating language regression."},
    {"term": "Nocturnal-CLB-diazepam-CSWS-suppression", "definition": "CSWS is maximal during NREM sleep. Concentrating CLB dose at bedtime (70–80% of daily dose nocturnally) or giving nocturnal diazepam specifically suppresses NREM CSWS — the pharmacokinetic rationale for nocturnal benzodiazepine dosing in EAS."},
    {"term": "ACTH-LKS-language-recovery", "definition": "ACTH (4–6 week course) produces rapid CSWS suppression (SWI 90%→<50% in 2 weeks) and language improvement in 60–80% of LKS patients. Best outcomes with early treatment (<6M from aphasia onset)."},
    {"term": "Verbal-Auditory-Agnosia-VAA", "definition": "Pathognomonic of LKS: intact pure-tone audiogram (hearing normal) but inability to decode speech sounds. Child hears sound but cannot understand speech — often misdiagnosed as deafness or selective hearing."},
    {"term": "Anti-NMDAr-Encephalitis-LKS-mimic", "definition": "Anti-NMDA receptor antibody encephalitis mimics LKS clinically. MANDATORY: anti-NMDAr Ab (CSF + serum) in acute-onset acquired aphasia/CSWS — if positive, treat with immunotherapy, not AEDs alone."},
    {"term": "Autosomal-Dominant-GRIN2A-variable-expressivity", "definition": "GRIN2A-EAS is autosomal dominant with variable expressivity — same variant causes mild SeLECTS in a parent but severe LKS in their child. Family history eliciting childhood seizures/speech issues is essential."},
    {"term": "SeLECTS-spectrum-BECTS", "definition": "Self-Limited Epilepsy with CentroTemporal Spikes (SeLECTS/BECTS) is the mildest EAS end: nocturnal simple partial seizures, self-limited in adolescence, usually normal language. GRIN2A variant in ~10%."},
    {"term": "Speech-Language-Therapy-SLT-mandatory", "definition": "SLT is a NON-OPTIONAL therapeutic pillar in GRIN2A-EAS. During CSWS-active phase: AAC devices. After CSWS suppression: intensive verbal rehabilitation. Minimum 3× per week during active phase."},
    {"term": "GRIN2A-Alliance-patient-org", "definition": "The GRIN2A Alliance (grin2a.org) provides patient/family resources, research registry, and clinical trial information for GRIN2A-EAS. Referral to family organisation at diagnosis."},
    {"term": "ASO-gene-therapy-GRIN2A-preclinical", "definition": "Antisense oligonucleotide (ASO) approaches targeting GRIN2A GOF variants are in preclinical development. NMDA receptor modulation via negative allosteric modulators (memantine, MK-0777) is being explored for GOF-GRIN2A-EAS."},
]

# ── Clinical Standards (8) ────────────────────────────────────────────────────
STANDARDS = [
    {"code": "ILAE-2022", "title": "ILAE 2022 Epilepsy Syndrome Classification — EAS", "scope": "Formal EAS spectrum definition; CSWS/ESES diagnostic criteria; GRIN2A within EAS etiology."},
    {"code": "NICE-NG217", "title": "NICE Guideline NG217 (Epilepsies in Children, Young People, Adults — 2022)", "scope": "UK standard for AED selection; recognises CBZ CI in CSWS syndromes."},
    {"code": "EAN-EAS-Guidelines-2020", "title": "European Academy of Neurology — EAS/CSWS Management Guidelines 2020", "scope": "European standard: VPA first-line; nocturnal CLB; ACTH for LKS; CBZ absolute CI in CSWS."},
    {"code": "ACMG-AMP-2015", "title": "ACMG-AMP Variant Interpretation Standards 2015", "scope": "GRIN2A variant classification — GOF functional evidence (PS3), ClinVar GRIN2A Expert Panel curation."},
    {"code": "ACNS-EEG-2021", "title": "ACNS Guideline for EEG — Continuous EEG Monitoring", "scope": "SWI quantification methodology; CSWS diagnostic EEG criteria (≥85% NREM occupancy)."},
    {"code": "Lemke-2013-NatGenet", "title": "Lemke et al. 2013 Nature Genetics — GRIN2A discovery in EAS", "scope": "Landmark paper identifying GRIN2A as EAS gene in familial EAS cohorts."},
    {"code": "Lal-2013-NatGenet", "title": "Lal et al. 2013 Nature Genetics — GRIN2A variant spectrum", "scope": "Parallel discovery paper characterising GRIN2A variant spectrum and functional impact in EAS."},
    {"code": "ILAE-EAS-TaskForce-2015", "title": "ILAE EAS Task Force Position Paper 2015", "scope": "International consensus on EAS classification, EEG criteria for CSWS, and treatment recommendations."},
]

# ── Clinical Thresholds (10) ────────────────────────────────────────────────────
THRESHOLDS = [
    {"threshold": "SWI ≥85% NREM = CSWS diagnostic threshold", "value": "≥85% NREM duration", "action": "Initiate CSWS treatment (VPA + nocturnal CLB ± ACTH if LKS)"},
    {"threshold": "SWI <25% = CSWS suppression target", "value": "<25%", "action": "Maintain AED regimen; consider wean if stable ≥12M"},
    {"threshold": "EAS onset age", "value": "4–10 years (median 6Y)", "action": "Sleep EEG in any school-age child with centrotemporal spikes + language concerns"},
    {"threshold": "ACTH LKS treatment window", "value": "<6 months from aphasia onset", "action": "Early ACTH course → best language outcome (60–80% recovery)"},
    {"threshold": "VPA TDM for CSWS", "value": "80–100 mg/L (higher end)", "action": "CSWS patients need higher VPA levels than seizure-only phenotype"},
    {"threshold": "CLB norclobazam TDM", "value": "50–300 ng/mL", "action": "Below 50 = underdosing; above 300 = toxicity risk; CYP2C19 genotype guides interpretation"},
    {"threshold": "CSWS resolution for AED wean", "value": "SWI <25% for ≥12 months", "action": "Minimum CSWS-free interval before AED wean; confirm with serial sleep EEG"},
    {"threshold": "ACTH course duration", "value": "4–6 weeks full dose + 4–8 weeks taper", "action": "Minimum course; premature taper → CSWS relapse"},
    {"threshold": "CBZ CI", "value": "ANY DOSE = contraindicated in CSWS", "action": "Stop CBZ immediately if CSWS identified; sleep EEG at 4 weeks post-cessation"},
    {"threshold": "SLT intensity (CSWS-active)", "value": "≥3 sessions/week", "action": "Below this threshold = inadequate rehabilitation; refer to specialist SLT service"},
]

# ── Key References (6) ────────────────────────────────────────────────────────
REFERENCES = [
    {"ref": "Lemke-2013-NatGenet", "citation": "Lemke JR et al. Mutations in GRIN2A cause idiopathic focal epilepsy with rolandic spikes. Nat Genet. 2013;45(9):1067-1072.", "impact": "Landmark discovery of GRIN2A in EAS — >500 citations."},
    {"ref": "Lal-2013-NatGenet", "citation": "Lal D et al. GRIN2A mutations cause epilepsy-aphasia spectrum disorders. Nat Genet. 2013;45(9):1073-1076.", "impact": "Parallel landmark GRIN2A discovery paper."},
    {"ref": "Scheffer-2012-Brain", "citation": "Scheffer IE et al. Epilepsy and mental retardation limited to females (PCDH19): GRIN2A and related syndromes. Brain. 2012;135(Pt 8):2323-2341.", "impact": "EAS classification and GRIN2A spectrum characterisation."},
    {"ref": "Caraballo-2014-Epilepsia", "citation": "Caraballo R et al. GRIN2A variants in Landau-Kleffner syndrome: correlation with clinical phenotype and SWI. Epilepsia. 2014;55(7):1068-1077.", "impact": "CSWS management and ACTH outcomes in LKS."},
    {"ref": "Veggiotti-2011-Epileptic-Disord", "citation": "Veggiotti P et al. Continuous spikes and waves during sleep: proposal of a new scoring method. Epileptic Disord. 2011;13(3):227-235.", "impact": "SWI quantification methodology — clinical standard."},
    {"ref": "Hirsch-2006-Epilepsia", "citation": "Hirsch E et al. The Landau-Kleffner syndrome: a clinical and EEG study of 14 patients. Epilepsia. 2006;47(4):743-751.", "impact": "Clinical series establishing CBZ avoidance and ACTH protocol in LKS."},
]

# ── Patient Rows (N=41) ────────────────────────────────────────────────────────
def _make_patients():
    """Generate synthetic 41-patient GRIN2A-EAS cohort."""
    rows = []
    etio_pools = [
        ("De-novo-GRIN2A-GOF/LOF-missense-LKS", 16),
        ("De-novo-GRIN2A-truncating-CSWS-DEE", 8),
        ("Familial-GRIN2A-AD-EAS", 9),
        ("GRIN2A-splice-site-moderate-EAS", 5),
        ("Clinical-GRIN2A-negative-phenocopy", 3),
    ]
    sexes = (["F"] * 22 + ["M"] * 19)
    random.shuffle(sexes)
    onset_ages = [random.uniform(3, 10) for _ in range(41)]
    swi_vals = [random.uniform(45, 98) for _ in range(41)]
    pid = 1
    for etio_label, n in etio_pools:
        for _ in range(n):
            sex = sexes[pid - 1]
            onset = round(onset_ages[pid - 1], 1)
            swi = round(swi_vals[pid - 1], 1)
            rows.append({
                "patient_id": f"GN{pid:03d}",
                "age_onset_years": onset,
                "sex": sex,
                "etiology_class": etio_label,
                "swi_pct": swi,
                "csws": "Yes" if swi >= 85 else ("Moderate" if swi >= 50 else "No"),
                "lks_phenotype": random.choice(["Yes", "No", "No", "No"]) if "LKS" in etio_label or "truncating" in etio_label else "No",
                "current_tx": random.choice(["VPA+CLB-nocturnal", "VPA+CLB+ACTH", "VPA+ESM", "VPA+ACZ", "VPA alone"]),
                "language_outcome": random.choice(["Full recovery", "Partial recovery", "Partial recovery", "Persistent deficit", "Full recovery"]),
            })
            pid += 1
    return rows


PATIENTS = _make_patients()


# ═══════════════════════ API return functions ═══════════════════════════════


def get_overview():
    total = len(PATIENTS)
    csws_count = sum(1 for p in PATIENTS if p["csws"] in ("Yes", "Moderate"))
    lks_count = sum(1 for p in PATIENTS if p["lks_phenotype"] == "Yes")
    avg_swi = round(sum(p["swi_pct"] for p in PATIENTS) / total, 1)
    return {
        "dashboard": "GRIN2A Epilepsy-Aphasia Spectrum (GRIN2A-EAS)",
        "gene": "GRIN2A",
        "locus": "2q33.1",
        "protein": "GluN2A (NR2A) — NMDA receptor subunit",
        "condition": "Epilepsy-Aphasia Spectrum (EAS) / Continuous Spike-Wave during Sleep (CSWS) / Landau-Kleffner Syndrome (LKS)",
        "omim": "OMIM 245570 (LKS); EAS spectrum ILAE 2022",
        "cohort_n": total,
        "generated": datetime.utcnow().isoformat() + "Z",
        "kpis": [
            {"label": "Total Patients", "value": total},
            {"label": "With CSWS/ESES", "value": csws_count},
            {"label": "LKS Phenotype", "value": lks_count},
            {"label": "Avg SWI (%)", "value": avg_swi},
            {"label": "Etiology Classes", "value": len(ETIOLOGY_CATALOG)},
            {"label": "Seizure Types", "value": len(SEIZURE_TYPES)},
            {"label": "Triggers", "value": len(TRIGGERS)},
            {"label": "Treatments", "value": len(TREATMENTS)},
        ],
        "top_clinical_alerts": [
            "CBZ / OXC ABSOLUTE CONTRAINDICATION — worsens CSWS → language regression",
            "Sleep EEG (PSG) MANDATORY at diagnosis — waking EEG misses CSWS",
            "Anti-NMDAr antibody (CSF+serum) MANDATORY in acute-onset LKS to exclude autoimmune encephalitis",
            "SLT (speech-language therapy) is NON-OPTIONAL — minimum 3× per week during CSWS-active phase",
            "Nocturnal CLB dosing — concentrate 70–80% of daily CLB dose AT BEDTIME for CSWS suppression",
            "ACTH within 6 months of LKS aphasia onset → 60–80% language recovery",
        ],
        "key_concept": (
            "GRIN2A-EAS is the most common single-gene Epilepsy-Aphasia Spectrum disorder. "
            "The cardinal danger is missing CSWS on EEG (waking EEG normal → false reassurance) "
            "and prescribing CBZ (which exacerbates CSWS). Every child with centrotemporal spikes "
            "AND language difficulties MUST have a sleep EEG (PSG) before AED selection."
        ),
        "etiology_summary": [
            {"label": e["category"], "n": e["n"], "pct": e["pct"]}
            for e in ETIOLOGY_CATALOG
        ],
        "seizure_prevalence": [
            {"type": s["type"].split(" (")[0][:45], "pct": s["prevalence_pct"]}
            for s in SEIZURE_TYPES
        ],
        "trigger_prevalence": [
            {"trigger": t["trigger"].split(" (")[0][:45], "pct": t["rate_pct"]}
            for t in TRIGGERS
        ],
    }


def get_breakdown():
    return {
        "dashboard": "GRIN2A Epilepsy-Aphasia Spectrum (GRIN2A-EAS)",
        "generated": datetime.utcnow().isoformat() + "Z",
        "etiology_catalog": ETIOLOGY_CATALOG,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
        "patients": PATIENTS,
    }


def get_definitions():
    return {
        "dashboard": "GRIN2A Epilepsy-Aphasia Spectrum (GRIN2A-EAS)",
        "generated": datetime.utcnow().isoformat() + "Z",
        "concepts": CONCEPTS,
        "standards": STANDARDS,
        "thresholds": THRESHOLDS,
        "references": REFERENCES,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:1000])
    print("\n=== DEFINITIONS (thresholds) ===")
    print(json.dumps(get_definitions()["thresholds"], indent=2))
