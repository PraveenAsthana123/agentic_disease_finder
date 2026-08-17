"""
GABRB2 Epilepsy (GEFS+ / CAE / DEE-GABRB2 / GABA-A β2 Subunit / 5q34)
=========================================================================
40-patient cohort · GABRB2 AD-LOF / de novo GOF-LOF / GEFS+ / CAE / DEE spectrum

GABRB2 BIOLOGY:
GABRB2 (5q34) encodes the β2 subunit of the GABA-A receptor (GABA-A receptor
subunit beta-2), the most abundantly expressed β subunit in the mammalian brain.
GABA-A receptors are pentameric ligand-gated Cl⁻ channels — the primary mediators
of fast inhibitory synaptic transmission in the CNS. The canonical synaptic GABA-A
receptor consists of 2α + 2β + 1γ subunits arranged around a central Cl⁻ ion pore.

KEY GABA-A β2 BIOLOGY:
  - β2 is the dominant β subunit isoform in cortex, hippocampus, thalamus, and cerebellum
  - β2-containing receptors mediate the majority of benzodiazepine-sensitive, phasic
    inhibitory currents at GABAergic synapses on pyramidal cells and interneurons
  - β2 subunit contains the GABA binding site (in collaboration with α subunits) and
    the benzodiazepine binding site (at the α/γ interface, β2 participates in receptor
    assembly and trafficking to the synapse)
  - LOF variants → reduced synaptic GABA-A current → cortical disinhibition → epilepsy
  - GOF variants → complex gain-of-function via altered channel kinetics, trafficking,
    or assembly — may paradoxically increase spontaneous GABA-A activity at extrasynaptic
    sites while reducing phasic synaptic currents (desensitisation)

GABRB2 ALLELIC DISORDER SPECTRUM:
  1. GEFS+ (Generalized Epilepsy with Febrile Seizures Plus) — Autosomal Dominant LOF (40%)
     Most common GABRB2-associated phenotype: febrile seizures + (afebrile generalised,
     absence, or myoclonic seizures persisting beyond 6 years). Familial AD inheritance,
     reduced penetrance (~75%). Onset: 6 months to 6 years. Good prognosis overall.
     Key variants: missense LOF in transmembrane domains or N-terminal ligand-binding.
  2. Childhood Absence Epilepsy (CAE) — de novo or familial (20%)
     GABRB2 is one of the confirmed susceptibility genes for CAE. Typical absence seizures
     (3 Hz spike-wave, 5-20 sec, abrupt onset/offset), age 4-10 years. May co-exist with
     GEFS+ spectrum or evolve from febrile seizures. Treatment: ETX ± VPA first-line.
  3. Dravet-like DEE — de novo severe GOF or LOF (20%)
     Rare but well-described: de novo GABRB2 variants (truncating or severe missense) causing
     persistent infantile epilepsy, intellectual disability, ataxia, hypotonia. Phenotypic
     overlap with Dravet syndrome (fever-sensitive, prolonged seizures, developmental
     regression). SCN1A-negative Dravet-like → GABRB2 panel essential.
  4. DEE / Lennox-Gastaut-like — de novo severe (10%)
     De novo GABRB2 missense with severe GABA-A dysfunction causing refractory DEE with
     multiple seizure types (GTCS, tonic, atonic, absence, myoclonic), EEG slow spike-wave
     (<2.5 Hz), developmental arrest. Rare GABRB2-specific LGS-like phenotype.
  5. Phenocopy / panel-negative — (10%)
     SCN1A (Dravet/GEFS+), GABRG2, GABRA1, GABRD presenting with clinically similar
     GEFS+/absence/DEE phenotype. GABRB2 negative on panel; reclassified.

CRITICAL DRUG INTERACTIONS AND CONTRAINDICATIONS:
  VPA is the preferred broad-spectrum agent for GABRB2-associated epilepsies (GEFS+, DEE):
  POLG MANDATORY before VPA — Alpers-Huttenlocher hepatic failure in POLG carriers.

  CBZ/OXC/PHENYTOIN — AVOID in GABRB2 (especially absence and myoclonic components):
  Sodium channel blockers (CBZ, OXC, PHT, LTG) can WORSEN absence seizures (absence
  acceleration) and myoclonic seizures via NaV1.1-dependent fast-spiking interneuron
  suppression. In Dravet-like GABRB2-DEE, LTG especially HIGH RISK for myoclonic
  aggravation (identical mechanism to SCN1A-Dravet). AVOID CBZ/OXC/PHT/LTG in any
  GABRB2 patient with absence or myoclonic seizure component.

  TIAGABINE ABSOLUTE CI in DEE-GABRB2 and all GABRB2 epileptic encephalopathies:
  GAT-1 inhibition → extrasynaptic GABA accumulation → tonic GABA-A activation → NCSE.
  The already-compromised GABA-A inhibitory architecture in GABRB2-DEE makes this
  catastrophically dangerous (same mechanism as Dravet, LGS, ARX, KCNQ3-DEE).

  FLUMAZENIL CAUTION: GABRB2 patients on benzodiazepines may have paradoxical
  flumazenil sensitivity if their GABRB2 variant affects BDZ binding site.
"""

import random
from datetime import datetime

SEED = 9203  # dashboard 203
random.seed(SEED)

# ── Etiology Catalog ──────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "GABRB2 AD-LOF / GEFS+ (familial, autosomal dominant)",
        "n": 16, "pct": 40,
        "category": "GABRB2-AD-LOF-GEFS-plus",
        "mechanism": (
            "The most common GABRB2 phenotype (40%) — autosomal dominant loss-of-function "
            "variants (missense in transmembrane domains TM1-TM4 or ligand-binding N-terminal, "
            "occasionally truncating with haploinsufficiency) reducing synaptic GABA-A "
            "current amplitude by ~40-60% in β2-containing receptor populations. The residual "
            "~40-60% β2 function is sufficient to maintain baseline cortical inhibition in "
            "most circumstances but creates a vulnerability to seizure-threshold lowering "
            "stimuli — particularly fever (which further reduces GABA-A channel conductance "
            "via temperature-sensitive gating kinetics), resulting in the GEFS+ spectrum: "
            "febrile seizures (onset 6M-6Y), febrile seizures plus (FS extending beyond 6Y), "
            "and occasional afebrile absence, GTCS, or myoclonic seizures in ~35% of affected "
            "family members. Pedigree: autosomal dominant with ~75% penetrance; unaffected "
            "obligate carriers are relatively common. Prognosis: generally favourable — most "
            "family members remit by adolescence. ~20% develop adult-onset sporadic GTCS."
        ),
        "eeg_correlate": (
            "GEFS+ / GABRB2-AD-LOF EEG: (1) Febrile seizure phase: generalised spike-wave "
            "or polyspike-wave; may be focal centrotemporal in some FS presentations. "
            "(2) Afebrile phase (GEFS+): normal interictal EEG in 40-50%; mild background "
            "theta in remainder. Occasional generalised 3-4 Hz spike-wave (absence component). "
            "(3) Hyperventilation: 3 Hz spike-wave provocation in absence subtype. "
            "Hallmark: NORMAL interictal background between episodes distinguishes GEFS+ "
            "from DEE. No burst-suppression, no hypsarrhythmia."
        ),
        "mri_finding": (
            "Brain MRI NORMAL in >95% of GABRB2-GEFS+ patients. Structural epilepsy "
            "excluded on initial MRI. Serial MRI not required if seizures respond to VPA/ETX "
            "and development is normal. If MRI abnormal (cortical dysplasia, hippocampal "
            "sclerosis) → reclassify as structural epilepsy + GABRB2 genetic modifier."
        ),
    },
    {
        "etiology": "GABRB2 de novo / familial Childhood Absence Epilepsy (CAE)",
        "n": 8, "pct": 20,
        "category": "GABRB2-CAE-absence",
        "mechanism": (
            "GABRB2 is one of the confirmed susceptibility genes for childhood absence epilepsy "
            "(CAE), accounting for ~3-5% of GABRB2-positive CAE in large gene-panel series. "
            "Variants are de novo missense or familial AD-LOF in the β2 ligand-binding or "
            "transmembrane domain, reducing thalamo-cortical GABA-A current and altering the "
            "thalamo-cortical 3 Hz oscillatory loop. The thalamic reticular nucleus (TRN) — "
            "the primary GABA-A inhibitory source modulating corticothalamic synchrony — is "
            "particularly sensitive to β2 LOF: reduced TRN → Gα/Gβ disinhibition → "
            "hypersynchronous 3 Hz spike-wave discharge. Presentation: typical absence "
            "seizures age 4-10 years, abrupt onset/offset, 3 Hz spike-wave, mild head drop, "
            "automatisms. May co-exist with GEFS+ history (febrile seizures preceding CAE). "
            "Treatment response: ETX ± VPA usually excellent; ~70% achieve remission by age 12."
        ),
        "eeg_correlate": (
            "CAE / GABRB2 EEG: (1) Ictal: bilateral symmetrical synchronous 3 Hz (2.5-4 Hz) "
            "spike-wave discharge, 5-20 seconds duration, abrupt onset and offset, maximum "
            "amplitude bifrontally. Evolution: burst begins at 3-4 Hz then slows to 2.5-3 Hz "
            "in last 1-2 seconds. (2) Interictal: normal background in 75%; occasional "
            "photoparoxysmal response (IPS). (3) Hyperventilation: reliable 3 Hz SWD "
            "provocation — 3 minutes HV mandated in diagnostic EEG. (4) Sleep: SWD fragments "
            "in drowsiness/Stage N1; complete suppression in N2/N3 (unlike LGS which shows "
            "slow SWD in sleep). GABRB2-CAE vs primary CAE: indistinguishable EEG; gene "
            "panel required for definitive GABRB2 attribution."
        ),
        "mri_finding": (
            "CAE / GABRB2: Brain MRI NORMAL (structural CAE excluded by definition). "
            "MRI performed once at diagnosis to exclude structural lesion. If normal + "
            "3 Hz SWD on EEG: clinical CAE diagnosis. Functional MRI (fMRI): thalamic "
            "BOLD signal changes during SWD documented in research settings but not "
            "required clinically. No serial MRI unless seizures fail to remit."
        ),
    },
    {
        "etiology": "GABRB2 de novo severe (Dravet-like DEE / fever-sensitive)",
        "n": 8, "pct": 20,
        "category": "GABRB2-Dravet-like-DEE",
        "mechanism": (
            "The most clinically severe GABRB2 phenotype (20% of cohort) — de novo missense "
            "or truncating variants with dominant-negative or severe haploinsufficient effect "
            "on GABA-A β2-containing receptor trafficking and function. Reduced synaptic "
            "GABA-A current to <30% of normal → catastrophic loss of cortical inhibitory "
            "tone → Dravet syndrome-like presentation in SCN1A-negative patients: onset in "
            "first year of life with prolonged febrile hemiclonic or GTCS, progressing to "
            "afebrile polymorphic seizures (myoclonic, focal, absence, GTCS), intellectual "
            "disability (universal), ataxia (60%), and fever hypersensitivity (80%). This "
            "subgroup is the GABRB2 population most frequently misdiagnosed as Dravet syndrome "
            "before SCN1A is negative and extended panel detects GABRB2. Key distinction from "
            "SCN1A-Dravet: GABRB2-DEE tends to have earlier onset seizures (2-6 months vs "
            "5-8 months for SCN1A-Dravet), more prominent hypotonia, and less stereotyped "
            "febrile hemiclonic pattern. Treatment: VPA (POLG), CLB, LEV, KD. AVOID LTG "
            "(myoclonic aggravation), CBZ/OXC (absence aggravation), Tiagabine (NCSE)."
        ),
        "eeg_correlate": (
            "Dravet-like GABRB2-DEE EEG: (1) Ictal in first year: generalised or hemispheric "
            "clonic seizure pattern — rhythmic delta with polyspike superimposition. (2) 1-3 "
            "years: polymorphic — generalised polyspike-wave (myoclonic), slow spike-wave "
            "(atonic/tonic), focal onset. Background: progressive delta slowing, loss of "
            "posterior dominant rhythm. (3) Chronic: slow background, multifocal IEDs, "
            "occasional photosensitivity. Unlike SCN1A-Dravet: less CSWS (continuous spike-"
            "wave in slow sleep) in early childhood; more diffuse background slowing. "
            "Ictal EEG video correlation: myoclonic jerks time-locked to polyspike-wave bursts."
        ),
        "mri_finding": (
            "Dravet-like GABRB2-DEE MRI: (1) Typically normal in first 2 years. "
            "(2) Progressive changes: hippocampal T2 signal changes after prolonged SE "
            "(25-30%) — may be transient (DWI restriction → FLAIR hyperintensity → "
            "eventual hippocampal sclerosis in severe cases). (3) Diffuse cerebral volume "
            "loss in severe drug-resistant DEE after age 3. (4) MRS: reduced NAA/Cr ratio "
            "in frontoparietal cortex. Serial MRI at 12M, 24M, then q2Y. 3T preferred."
        ),
    },
    {
        "etiology": "GABRB2 de novo LGS-like DEE (severe encephalopathy)",
        "n": 4, "pct": 10,
        "category": "GABRB2-LGS-DEE",
        "mechanism": (
            "Approximately 10% carry de novo GABRB2 variants with severe GABA-A β2 dysfunction "
            "causing a Lennox-Gastaut-like DEE: multiple seizure types (tonic > atonic > absence > "
            "focal > myoclonic), EEG slow spike-wave <2.5 Hz, intellectual disability (universal), "
            "and non-epileptic falls (drop attacks). These patients typically have complete loss of "
            "β2 subunit function via truncating de novo variants (frameshift, nonsense) causing "
            "NMD-mediated haploinsufficiency, or dominant-negative missense severely impairing "
            "receptor pentamer assembly and membrane trafficking. Severe reduction of GABA-A "
            "phasic inhibition → thalamo-cortical and brainstem disinhibition → LGS seizure "
            "semiology. VPA + CLB ± LEV foundation; rufinamide and perampanel for refractory "
            "tonic/atonic component. KD considered at ≥2 AED failure. SUDEP risk high."
        ),
        "eeg_correlate": (
            "LGS-like GABRB2-DEE EEG: (1) Awake: slow (<2.5 Hz) generalised spike-wave, "
            "diffuse background slowing (theta-delta). (2) Sleep: generalised paroxysmal fast "
            "activity (GPFA) at 10-25 Hz — hallmark tonic seizure correlate in LGS. "
            "(3) Tonic seizures: brief desynchronisation → GPFA burst. "
            "(4) Atonic/drop attacks: generalised polyspike → slow wave → brief electrodecrement. "
            "(5) Background: severe diffuse slow wave abnormality, loss of physiological "
            "sleep features. Video-EEG essential for seizure classification. "
            "Continuous EEG in SE: ICU monitoring for subclinical NCSE."
        ),
        "mri_finding": (
            "LGS-GABRB2-DEE MRI: Progressive cerebral atrophy in severe cases. Corpus callosum "
            "thinning (35%). Diffuse white matter volume loss. MRI typically abnormal by age 3-4 "
            "in severe LGS-DEE. Check for cortical dysplasia (FCD) — occasionally GABRB2 is a "
            "modifier of an underlying structural lesion (dual pathology). 3T MRI with FCD "
            "protocol (thin-slice T1, T2, FLAIR, post-contrast) annually in drug-resistant cases."
        ),
    },
    {
        "etiology": "Phenocopy / panel-negative (SCN1A / GABRG2 / GABRA1 / GABRD)",
        "n": 4, "pct": 10,
        "category": "GABRB2-negative-phenocopy",
        "mechanism": (
            "10% of cohort referred as GABRB2-suspected (GEFS+/absence/DEE) are GABRB2-negative; "
            "reclassified as phenocopy: SCN1A (Dravet/GEFS+ — most common GABRB2-negative GEFS+ "
            "mimic; SCN1A GEFS+ has identical pedigree pattern), GABRG2 (GEFS+/DEE — γ2 subunit "
            "partner of β2 in the α1β2γ2 trimer; GABRG2-GEFS+ phenotype essentially identical "
            "to GABRB2-GEFS+ clinically), GABRA1 (CAE/JME with AD-LOF inheritance), or GABRD "
            "(GEFS+ with mild phenotype, δ-subunit extrasynaptic receptor loss). "
            "Distinction: clinical phenotype alone cannot distinguish; comprehensive GABAergic "
            "epilepsy gene panel essential. GABRB2 panel-negative GEFS+: extend panel to "
            "SCN1A/SCN1B/GABRG2/GABRA1/GABRD."
        ),
        "eeg_correlate": (
            "Phenocopy EEG: Same GEFS+ and absence EEG patterns as GABRB2-positive. "
            "SCN1A-Dravet: tends to have more prominent hemispheric clonic + multifocal IEDs "
            "by 2-3 years. GABRG2-DEE: may show early focal centrotemporoparietal IEDs. "
            "Clinical distinction requires comprehensive panel; EEG not gene-diagnostic."
        ),
        "mri_finding": (
            "Phenocopy MRI: varies by gene. SCN1A-Dravet: same progressive hippocampal "
            "changes. GABRG2: largely normal or similar to GABRB2. Structural investigation "
            "same as GABRB2 panel."
        ),
    },
]

# ── Seizure Types ─────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Febrile Seizures (FS) / Febrile Seizures Plus (GEFS+)",
        "prevalence_pct": 88,
        "eeg_correlate": "Generalised clonic/tonic-clonic during fever; normal interictal EEG between events",
        "clinical_tip": (
            "GEFS+ diagnostic criteria: (1) Febrile seizures before age 6; (2) +/- afebrile GTCS "
            "persisting beyond age 6; (3) family history of febrile seizures or GEFS+. "
            "Fever threshold: onset at ≥38°C in GEFS+ vs ≥37.5°C in GABRB2-DEE (Dravet-like). "
            "Antipyretic protocol: paracetamol at 38.0°C (not waiting for 38.5°C) in all GABRB2 "
            "patients. Rescue midazolam 0.3 mg/kg buccal/nasal for any seizure >5 min. "
            "Distinguish prolonged FS (>15 min) from GEFS+ proper: prolonged FS → risk of "
            "hippocampal injury → serial MRI. Short FS (<5 min) → standard GEFS+ management."
        ),
    },
    {
        "type": "Generalised Tonic-Clonic Seizures (GTCS — afebrile)",
        "prevalence_pct": 72,
        "eeg_correlate": "Generalised polyspike then slow wave; post-ictal suppression; diffuse EEG change",
        "clinical_tip": (
            "Afebrile GTCS in GEFS+: usually rare (≤4/year) and controlled by VPA monotherapy. "
            "In DEE-GABRB2: more frequent and drug-resistant. GTCS risk factors: adolescent growth "
            "spurts (pharmacokinetic changes), sleep deprivation, missed doses. "
            "VPA TDM essential: sub-therapeutic trough → GTCS breakthrough. "
            "First-aid: recovery position, time seizure, call EMS if >5 min or clustering. "
            "SUDEP risk highest in patients with nocturnal unwitnessed GTCS + DRE."
        ),
    },
    {
        "type": "Absence Seizures (typical, 3 Hz SWD)",
        "prevalence_pct": 55,
        "eeg_correlate": "Bilateral 3 Hz spike-wave, 5-20 sec, bifrontal maximum; HV-provoked",
        "clinical_tip": (
            "3 Hz absence in GABRB2: typically responds to ETX + VPA combination (>85% remission). "
            "CRITICAL: Do NOT use CBZ, OXC, or PHT for absence component — these Na-channel blockers "
            "cause ABSENCE ACCELERATION (increase absence frequency by NaV1.1 interneuron suppression). "
            "LTG: MODERATE RISK — may worsen absence in GABRB2 (some reports of absence acceleration). "
            "ETX DOES NOT cover GTCS — always combine ETX + VPA when both GTCS and absence present. "
            "HV 3-minute provocation mandatory in EEG: if no 3 Hz SWD provoked during HV, CAE "
            "diagnosis should be reconsidered."
        ),
    },
    {
        "type": "Myoclonic Seizures (GABRB2-DEE / Dravet-like)",
        "prevalence_pct": 40,
        "eeg_correlate": "Bilateral polyspike-wave 3-6 Hz; time-locked jerks; photosensitivity in 30%",
        "clinical_tip": (
            "Myoclonic seizures in GABRB2-DEE: morning predominant (post-awakening), upper "
            "extremity, often stimulus-sensitive (noise, startle). CRITICAL DRUG RULE: "
            "LTG ABSOLUTE CONTRAINDICATION if myoclonic component present — NaV1.1-PV interneuron "
            "suppression → dramatic myoclonus worsening (same Dravet mechanism). "
            "CBZ/OXC: also AVOIDED — can worsen myoclonic seizures in GABRB2-DEE. "
            "Treatment: VPA (POLG first), CLB, LEV. If photosensitive: sunglasses, anti-glare "
            "filters. Clonazepam: Level C for myoclonic clusters not responding to VPA+CLB."
        ),
    },
    {
        "type": "Focal Seizures (GABRB2-DEE / Dravet-like)",
        "prevalence_pct": 35,
        "eeg_correlate": "Focal ictal discharge (temporal or frontoparietal onset); may secondarily generalise",
        "clinical_tip": (
            "Focal seizures in GABRB2-DEE may mimic temporal lobe epilepsy. EEG focal onset "
            "does not exclude genetic aetiology — GABRB2-DEE frequently produces focal cortical "
            "discharges due to regional GABA-A density gradients. MRI normal → genetic "
            "aetiology likely. If MRI shows focal signal change: dual pathology (structural + "
            "genetic modifier) possible. Focal seizures in GABRB2: partial response to LEV "
            "or CLB; avoid OXC (absence/myoclonic co-aggravation risk)."
        ),
    },
    {
        "type": "Tonic / Atonic / Drop Attacks (LGS-like GABRB2-DEE)",
        "prevalence_pct": 18,
        "eeg_correlate": "Tonic: GPFA 10-25 Hz in sleep; Atonic: generalised polyspike → electrodecrement",
        "clinical_tip": (
            "Tonic and atonic seizures in GABRB2-LGS-DEE: highest injury risk (falls, head "
            "trauma). Protective helmet mandatory. VPA + CLB foundation; add rufinamide for "
            "tonic/atonic component (Level B for LGS). Perampanel (AMPA antagonist, Level B "
            "LGS adjunct) may help focal-onset component. VNS: Level B for drug-resistant LGS. "
            "CCS (corpus callosotomy): Level B for refractory drop attacks with no focal "
            "resectable substrate. AVOID AED dose changes without EEG reassessment."
        ),
    },
]

# ── Triggers ─────────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Fever (≥38°C GEFS+; ≥37.5°C in DEE/Dravet-like)",
        "prevalence_pct": 92,
        "mechanism": (
            "GABRB2 β2 LOF creates a temperature-dependent vulnerability: fever reduces GABA-A "
            "β2-containing channel open probability by ~25-35% per degree above 37°C (via "
            "temperature-sensitive channel kinetics — faster desensitisation, reduced mean open "
            "time). In GABRB2-GEFS+ where baseline β2 current is already ~50% of normal, "
            "fever reduces it further to threshold → generalised seizure. In GABRB2-DEE/"
            "Dravet-like where β2 is <30% of normal, even mild fever (37.5°C) is sufficient. "
            "This fever threshold difference is diagnostically informative: low-threshold "
            "fever sensitivity (<38°C) → Dravet-like; higher threshold (≥38.5°C) → GEFS+."
        ),
        "action": "Written fever action plan: paracetamol at 38.0°C (GEFS+) or 37.5°C (DEE/Dravet-like). Rescue buccal midazolam 0.3 mg/kg prescribed. Emergency plan if seizure >5 min or cluster. Parent trained in rescue administration. Avoid ibuprofen if on VPA (renal/GI interaction).",
    },
    {
        "trigger": "Missed AED dose / sub-therapeutic VPA level",
        "prevalence_pct": 75,
        "mechanism": "VPA half-life 8-20h; sub-therapeutic trough after missed dose reduces GABA-A enhancement → breakthrough GTCS or absence. VPA TDM: target trough 50-100 µg/mL. Breakthrough seizures often occur in morning (overnight pharmacokinetic trough).",
        "action": "Electronic medication reminders. TDM if breakthrough after missed dose. Split-dose VPA (BID or TID) for patients with short half-life pharmacokinetics. Educate: NEVER double-dose after missed VPA. If >12h since missed VPA: resume normal dose schedule.",
    },
    {
        "trigger": "Sleep deprivation / disrupted sleep-wake cycle",
        "prevalence_pct": 68,
        "mechanism": "Slow-wave sleep (SWS) promotes GABA-A receptor synaptic scaling (upregulation of β2-containing receptors at the membrane). Sleep deprivation abolishes SWS → reduced synaptic GABA-A insertion → reduced inhibitory tone → seizure threshold lowering. Myoclonic seizures on waking particularly sensitive to sleep deprivation.",
        "action": "Sleep hygiene counselling mandatory. Seizure diary: correlate seizure clusters with sleep quality. If morning myoclonic clusters consistently follow poor sleep: assess VPA timing (split dosing), consider small melatonin supplementation for sleep-onset insomnia. Avoid night shift work in adults.",
    },
    {
        "trigger": "Photosensitivity / photic stimulation (GABRB2-DEE subset)",
        "prevalence_pct": 45,
        "mechanism": "Photosensitive IEDs in ~30-45% of GABRB2-DEE patients (significantly higher than GEFS+ ~5%). Photic stimulation at 12-18 Hz activates occipital GABA-A networks; in GABRB2 LOF, reduced feedforward inhibition → photoparoxysmal response (PPR). Pattern: occipital → secondary generalised polyspike-wave.",
        "action": "Intermittent photic stimulation (IPS) mandated in diagnostic EEG. If PPR: photosensitivity precautions — polarised/tinted glasses outdoors, anti-glare screen filters, avoid discothèque strobe. VPA reduces photosensitivity in most patients. If refractory photosensitivity: specialist photosensitivity management centre.",
    },
    {
        "trigger": "Intercurrent illness / immune activation",
        "prevalence_pct": 55,
        "mechanism": "Pro-inflammatory cytokines (IL-1β, IL-6, TNF-α) reduce GABA-A receptor surface expression via PKC phosphorylation of β2 subunit, accelerating receptor internalisation and reducing synaptic GABA-A currents. Compound effect with concurrent fever → acute seizure escalation during viral URTI, gastroenteritis.",
        "action": "Illness action plan: monitor temperature q4h during febrile illness. Anticipate seizure escalation. If >3 seizures in 24h during illness → emergency evaluation. Ensure adequate hydration (dehydration further lowers seizure threshold). Avoid azithromycin and other macrolides in patients on VPA (hepatic enzyme interaction).",
    },
    {
        "trigger": "Hyperventilation (absence trigger)",
        "prevalence_pct": 85,
        "mechanism": "Hyperventilation induces hypocapnia → cerebral vasoconstriction → cortical alkalosis → increased GABA-A receptor open probability shift → paradoxically enhanced 3 Hz thalamo-cortical synchrony → absence seizure induction. Classic diagnostic EEG trigger for CAE/GABRB2-absence. 3 minutes HV produces absence in >90% of active CAE.",
        "action": "Mandate 3-minute HV in every diagnostic EEG. Explain to patient/family that HV-induced absence during EEG is expected and monitored (not an emergency). In daily life: vigorous exercise-induced hyperventilation rarely triggers absence (transient; mechanism different from forced HV). No restriction on physical activity.",
    },
    {
        "trigger": "Catamenial exacerbation (females — perimenstrual)",
        "prevalence_pct": 38,
        "mechanism": "Progesterone metabolite allopregnanolone is a potent positive allosteric modulator of GABA-A β2-containing receptors. Perimenstrual progesterone withdrawal → abrupt loss of allopregnanolone → acute GABA-A downregulation → seizure exacerbation in ~38% of post-pubertal females with GABRB2-epilepsy. Neurosteroid hypothesis: explains catamenial clustering.",
        "action": "Catamenial seizure diary mandatory in post-pubertal females. If catamenial pattern: consider clobazam pulse therapy 10 mg/day perimenstrually (3-7 days around menstruation onset). Hormonal contraception (combined OCP): evidence limited but may reduce catamenial exacerbation by stabilising steroid cycle. Neurosteroid therapy (ganaxolone — GABA-A positive modulator): emerging evidence, CDKL5/status epilepticus indication; extrapolation to catamenial GABRB2 is investigational.",
        },
    {
        "trigger": "Drug-drug interactions / enzyme inducers",
        "prevalence_pct": 30,
        "mechanism": "VPA is metabolised by CYP2C9, CYP2C19, and UGT enzymes. Enzyme-inducing AEDs (CBZ, PHT, PB, OXC) dramatically reduce VPA plasma levels by 40-60% via hepatic CYP induction → sub-therapeutic VPA → breakthrough seizures. Conversely: VPA inhibits CBZ epoxide hydrolase → CBZ toxicity if co-prescribed.",
        "action": "AVOID co-prescribing enzyme inducers with VPA. If inducers unavoidable: VPA dose increase + TDM q4W. Monitor for VPA toxicity signs (tremor, somnolence, nausea) if enzyme inhibitors added. Drug interaction check (DrugBank/Lexicomp) before any new prescription in GABRB2 patients on VPA.",
    },
]

# ── Treatments ────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Valproate / Sodium Valproate (VPA)",
        "level": "Level A — First-line GABRB2 GEFS+ / DEE / GTCS + absence",
        "dose": (
            "GEFS+/CAE: 15-30 mg/kg/day PO divided q8-12h; target serum level 50-100 µg/mL. "
            "DEE: 30-40 mg/kg/day PO/IV; target 80-120 µg/mL in drug-resistant cases. "
            "Neonates: 20-40 mg/kg/day IV divided q8h (limited data). "
            "Extended-release (VPA-ER): preferred for adolescents/adults (better tolerance, "
            "once-daily or BID dosing). "
            "MANDATORY: POLG panel completed and documented BEFORE first VPA dose."
        ),
        "moa": (
            "Multiple synergistic mechanisms: (1) GABA-A potentiation — VPA inhibits GABA "
            "transaminase (GABA-T) and succinic semialdehyde dehydrogenase → increased brain "
            "GABA levels → enhanced endogenous GABA-A activation (compensates GABRB2 LOF at "
            "remaining functional receptors). (2) Na+ channel use-dependent inactivation "
            "(secondary; less than CBZ). (3) T-type Ca2+ channel blockade → reduced "
            "thalamo-cortical 3 Hz oscillation → anti-absence. (4) HCN channel upregulation "
            "(Ih current). VPA is the preferred broad-spectrum agent for GABRB2-associated "
            "epilepsies because its GABA enhancement complements GABRB2 β2 LOF."
        ),
        "efficacy": "GEFS+ GTCS: ~80% seizure freedom on VPA monotherapy. CAE: VPA ± ETX achieves 65-75% seizure freedom (CATNAP data). DEE-GABRB2: partial response (~50% >50% seizure reduction). Level A for generalised epilepsies with GTCS.",
        "safety": (
            "POLG MANDATORY — Alpers-Huttenlocher hepatic failure in POLG carriers (fatal). "
            "Teratogenic: neural tube defects (2-10% — absolute CI in pregnancy without "
            "specialist oversight; folic acid 5 mg/day if VPA unavoidable in childbearing age). "
            "Common: tremor, weight gain, hair loss (transient), nausea (improved with ER formulation). "
            "Serious: hepatotoxicity (especially <2 years age), pancreatitis, thrombocytopaenia, "
            "hyperammonaemia. Long-term: polycystic ovary syndrome (PCOS) risk in females "
            "(use with caution; consider alternatives in adolescent females when possible)."
        ),
        "monitoring": "POLG panel (mandatory, document result before VPA). LFTs + ammonia at baseline, 2W, 1M, then q3M. FBC (platelets q3M). TDM q3M (trough). Serum carnitine if ammonia elevated. Growth/weight chart. PCOS screening in adolescent females (q1Y).",
        "gabrb2_note": "GABRB2-SPECIFIC: VPA GABA enhancement directly compensates GABRB2 β2 LOF by increasing endogenous GABA levels and GABA-A activation at remaining functional receptors. First-line for GEFS+, CAE+GTCS, and DEE-GABRB2. POLG mandatory — document result before prescribing.",
    },
    {
        "drug": "Ethosuximide (ETX)",
        "level": "Level A — First-line for GABRB2 absence (CAE / GEFS+ absence component)",
        "dose": "Children: 20-40 mg/kg/day PO divided q12h; target serum level 40-100 µg/mL. Max 1500 mg/day. Start 125-250 mg/day, uptitrate over 2-4 weeks. Liquid formulation available for young children.",
        "moa": "Primary mechanism: T-type (Cav3.1, Cav3.2) Ca2+ channel blockade in thalamic relay neurons (specifically ventrobasal thalamus) → suppression of low-threshold Ca2+ spike (LTS) responsible for 3 Hz thalamo-cortical oscillation underlying absence seizures. DOES NOT affect GABA-A receptors directly. Anti-absence selective — NO effect on GTCS (never use ETX as monotherapy if GTCS component present).",
        "efficacy": "CAE absence: ETX alone achieves ~53% seizure freedom; ETX + VPA 65-75% (CATNAP data). For GABRB2-absence: similar to primary CAE. ETX does NOT protect against GTCS — always combine with VPA if GTCS risk (GEFS+).",
        "safety": "Generally well tolerated. Common: nausea, vomiting (take with food), headache, dizziness. Serious (rare): aplastic anaemia (1:500,000), lupus-like syndrome, psychiatric effects (depression, psychosis). No teratogenicity data. Drug interactions: minimal (no CYP induction); VPA increases ETX levels 15-30% (monitor ETX levels when VPA added).",
        "monitoring": "ETX TDM (target 40-100 µg/mL) at 4W, then q6M. FBC (q6M — aplastic anaemia surveillance). LFTs at baseline. VPA+ETX co-monitoring: recheck ETX level 4W after VPA initiation.",
        "gabrb2_note": "GABRB2-specific absence: ETX ± VPA is the preferred combination. ETX does NOT worsen myoclonic or GTCS components of GABRB2 epilepsy (unlike LTG, CBZ, OXC). Pure ETX monotherapy only if no GTCS component confirmed by EEG.",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B — Adjunct GABRB2 DEE / GEFS+ / add-on",
        "dose": "Children: 20-60 mg/kg/day PO/IV divided q12h. Adults: 1000-3000 mg/day divided q12h. Neonates: 40-60 mg/kg/day IV q12h (limited data). Oral liquid available.",
        "moa": "SV2A synaptic vesicle glycoprotein binding → impairs neurotransmitter exocytosis vesicle priming → reduced excitatory release cycle. No direct GABA-A effect; broad-spectrum. Complementary to VPA in GABRB2 (different mechanism).",
        "efficacy": "Broad-spectrum adjunct: ~40% ≥50% responder rate as add-on in GABRB2-DEE. Particular utility in: focal seizure component, myoclonic seizures (LEV anti-myoclonic, unlike LTG). Non-inferior to phenobarbital as second-line neonatal seizure agent.",
        "safety": "Behavioural effects (irritability, aggression, emotional lability) in 15-25% — most common reason for discontinuation. Psychiatric: depression, anxiety, psychosis (rare). Dose-dependent somnolence. No organ toxicity. No hepatic monitoring required. Note: avoid LEV dose reduction abruptly in DEE patients (withdrawal seizure risk).",
        "monitoring": "Behavioural diary. TDM optional (10-40 µg/mL). Psychiatric review if behavioural effects. Renal dose adjustment (LEV is renally cleared; dose-adjust for eGFR <80).",
        "gabrb2_note": "Add-on to VPA in GABRB2-DEE or GEFS+ with residual focal/GTCS seizures. Preferred over LTG when myoclonic component present (LEV anti-myoclonic; LTG HIGH RISK myoclonus aggravation in GABRB2-DEE). Behavioural effects: counsel parents and patient; assess at 4-week visit.",
    },
    {
        "drug": "Clobazam (CLB)",
        "level": "Level B — Adjunct / pulse therapy (1,5-benzodiazepine; α2/α3 GABA-A)",
        "dose": "Children: 0.1-0.5 mg/kg/day PO divided q12-24h. Max 40 mg/day. Catamenial pulse: 10 mg/day perimenstrually (3-7 days). Titrate over 2-4 weeks.",
        "moa": "Positive allosteric modulation of GABA-A receptors with 1,5-benzodiazepine structure — preference for α2/α3 subunit-containing receptors (limbic/spinal) over α1 (cortical/sedation). The α2/α3 preference reduces sedation liability vs 1,4-BDZs. Note: CLB mechanism depends on FUNCTIONAL β2 subunits — in severe GABRB2 LOF (<30% β2), CLB efficacy may be reduced (insufficient β2 for BDZ allosteric amplification).",
        "efficacy": "Adjunct in GABRB2-DEE and GEFS+ with residual seizures: ~40-45% ≥50% responder rate. Pulse therapy in catamenial exacerbation: ~60% reduction in perimenstrual seizure frequency (observational data). Tolerance in chronic dosing: 25-30% at 6M — drug holiday strategy.",
        "safety": "Sedation (dose-dependent), hypersalivation, drooling. Tolerance and dependence with long-term use. Less respiratory depression than 1,4-BDZs. Withdrawal: taper slowly over ≥4 weeks to prevent withdrawal seizures.",
        "monitoring": "Sedation scale (COMFORT-B in infants). Efficacy review at 3M. If tolerance develops: consider drug holiday (3-4 week gradual reduction, brief CLB-free period, then reinitiation — resets tolerance in ~60%).",
        "gabrb2_note": "Useful adjunct for GABRB2-DEE and catamenial GEFS+. Be aware that CLB efficacy may be reduced in severe β2-LOF (reduced BDZ allosteric site function). In severe GABRB2-LGS-DEE: CLB remains worthwhile but expect partial response. Avoid abrupt discontinuation.",
    },
    {
        "drug": "Rufinamide",
        "level": "Level B — LGS-like GABRB2-DEE (tonic/atonic component)",
        "dose": "Children >4 years: 10 mg/kg/day PO (start), titrate to 45 mg/kg/day divided q12h (max 3200 mg/day). ≥30 kg adults: up to 3200 mg/day. Food increases bioavailability (always with meals).",
        "moa": "Novel mechanism: prolongs inactivation of voltage-gated Na+ channels (distinct kinetics from CBZ/PHT); reduces high-frequency neuronal firing. Particular efficacy in tonic and atonic seizures (LGS hallmark), possibly via brainstem neuromodulation. No GABA-A effect.",
        "efficacy": "LGS tonic/atonic seizures: ~42% ≥50% reduction (Marson 2012 meta-analysis). GABRB2-LGS-like DEE: limited specific data but mechanistically analogous. Level B in ILAE LGS guidelines.",
        "safety": "Somnolence, vomiting, dizziness, nausea. QTc shortening: avoid in congenital short QT syndrome. Appetite decrease. Avoid in hepatic impairment. Interactions: VPA increases rufinamide levels 25-85% (reduce rufinamide starting dose if on VPA; monitor QTc).",
        "monitoring": "QTc at baseline and 4W (short QT CI). LFTs baseline. Seizure diary for tonic/atonic quantification. TDM not routinely required. If VPA co-prescribed: monitor for rufinamide toxicity.",
        "gabrb2_note": "Reserve for GABRB2-LGS-DEE with prominent tonic/atonic drop attacks uncontrolled on VPA+CLB+LEV. Not indicated for GEFS+ or CAE-only phenotypes. VPA interaction: reduce rufinamide starting dose.",
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B — Drug-resistant GABRB2-DEE (≥2 AEDs failed)",
        "dose": "Classical 4:1 ratio or Modified Atkins Diet (MAD). Initiation with KD-trained dietitian + epileptologist. Beta-hydroxybutyrate (β-OHB) target 2-5 mmol/L. Multivitamin/mineral supplementation mandatory.",
        "moa": "Multiple metabolic anti-seizure mechanisms: ketone bodies (β-OHB, acetoacetate) enhance vesicular GABA release (increases synaptic GABA availability → partially compensates GABRB2 β2 LOF); activate KATP channels; inhibit AMPA glutamate receptors; alter mitochondrial biogenesis. KD mechanism is GABA-A-independent — effective even when β2 function is severely reduced.",
        "efficacy": ">50% seizure reduction in ~50-55% of drug-resistant GABRB2-DEE patients. Particularly effective for myoclonic and GTCS components. KD may allow VPA dose reduction (KD+lower VPA = comparable seizure control with less VPA toxicity).",
        "safety": "Dyslipidaemia (LDL elevation), kidney stones (citrate supplementation), constipation, growth impairment, metabolic acidosis. Interaction: KD + VPA → additive hyperammonaemia risk (monitor ammonia q3M). KD + rufinamide + VPA: complex three-way interaction; simplify regimen if possible.",
        "monitoring": "β-OHB daily (urine) or weekly (blood). Lipid panel q3M. Urinalysis + renal US q6M (stones). Growth chart monthly × 6M then q3M. Ammonia if VPA co-prescribed (q3M). Dietitian review monthly × 6M then q3M.",
        "gabrb2_note": "KD enhances GABA availability — mechanistically complementary to GABRB2-LOF. May allow VPA dose reduction in patients with VPA toxicity. Useful when ≥2 AEDs failed in GABRB2-DEE. VPA+KD: monitor ammonia closely (additive hyperammonaemia).",
    },
]

# ── Contraindications ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Tiagabine",
        "severity": "ABSOLUTE CONTRAINDICATION",
        "risk": "Non-convulsive status epilepticus (NCSE) / Status Epilepticus",
        "mechanism": (
            "Tiagabine inhibits GAT-1 (GABA transporter 1), the primary synaptic and glial GABA "
            "reuptake transporter. In GABRB2-DEE where cortical inhibitory architecture is "
            "already compromised (β2 LOF → reduced phasic GABA-A current), GAT-1 inhibition "
            "causes massive extrasynaptic GABA accumulation → sustained tonic GABA-A activation "
            "(via extrasynaptic δ-subunit receptors) → paradoxical depolarising GABA-mediated "
            "block → NCSE. Catastrophic in GABRB2-DEE, Dravet-like, and LGS-like phenotypes. "
            "Class effect in all epileptic encephalopathies with GABAergic dysfunction."
        ),
        "action": "NEVER PRESCRIBE tiagabine in any GABRB2 patient (GEFS+, CAE, or DEE). Document absolute CI in chart. If inadvertently prescribed: STOP IMMEDIATELY, hospitalise, IV BDZ ± LEV for NCSE management.",
    },
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) / Phenytoin (PHT)",
        "severity": "HIGH RISK — CONTRAINDICATED in absence + myoclonic GABRB2",
        "risk": "Absence acceleration / Myoclonic seizure aggravation / Paradoxical worsening",
        "mechanism": (
            "Na-channel blockers (CBZ, OXC, PHT) suppress NaV1.1-dependent parvalbumin-positive "
            "fast-spiking interneurons (PV-FSI) — the primary fast inhibitory neurons controlling "
            "cortical excitability and thalamo-cortical synchrony. PV-FSI suppression → reduced "
            "feedforward inhibition of thalamo-cortical loop → ENHANCED 3 Hz spike-wave → "
            "absence acceleration. Simultaneously, reduced cortical PV-FSI → paradoxical "
            "disinhibition → MYOCLONUS WORSENING (identical Dravet/SCN1A mechanism). "
            "In GABRB2-GEFS+ with absence component: CBZ/OXC may worsen absence dramatically. "
            "In GABRB2-DEE with myoclonic component: CBZ/OXC/PHT may precipitate myoclonic SE."
        ),
        "action": "NEVER use CBZ, OXC, or PHT as first- or second-line agents in any GABRB2 patient with documented absence or myoclonic seizures. If absence or myoclonic seizures present: use VPA ± ETX ± CLB ± LEV. If focal seizures only (no absence/myoclonus confirmed): OXC/CBZ may be considered as last resort with EEG monitoring for absence exacerbation.",
    },
    {
        "drug": "Lamotrigine (LTG) — if myoclonic or absence component",
        "severity": "MODERATE-HIGH RISK",
        "risk": "Myoclonus aggravation (NaV1.1 PV-interneuron block) / Possible absence acceleration",
        "mechanism": (
            "LTG voltage-gated Na-channel blockade (NaV1.1/NaV1.2 fast-inactivation stabilisation) "
            "suppresses PV-FSI — causing myoclonic aggravation in patients with myoclonic seizures "
            "(Dravet-like GABRB2-DEE: high risk; identical to SCN1A-Dravet worsening on LTG). "
            "LTG may also worsen absence in some GABRB2-CAE patients (NaV-dependent mechanism). "
            "LTG IS SAFE in GABRB2-GEFS+ with GTCS ONLY (no myoclonic/absence component) — "
            "but clinical verification of seizure type inventory essential before prescribing."
        ),
        "action": "Before prescribing LTG: verify complete seizure inventory (EEG, video-EEG if needed). If ANY myoclonic seizures documented: DO NOT USE LTG. If GTCS only, no absence, no myoclonus: LTG may be considered as adjunct. If LTG inadvertently started and myoclonus worsens: STOP LTG immediately.",
    },
    {
        "drug": "VPA without POLG screen",
        "severity": "HIGH RISK — MANDATORY POLG EXCLUSION",
        "risk": "Alpers-Huttenlocher Syndrome (fulminant hepatic failure, fatal)",
        "mechanism": (
            "VPA inhibits mitochondrial β-oxidation and depletes NAD+ and CoA-SH. In POLG "
            "(gamma-DNA polymerase) pathogenic variant carriers — POLG variants cause mitochondrial "
            "DNA depletion syndrome → mitochondrial dysfunction → in combination with VPA: "
            "(1) Direct VPA metabolite (4-en-VPA) toxicity to dysfunctional POLG mitochondria; "
            "(2) NAD+ depletion precipitates mitochondrial failure in POLG-deficient hepatocytes. "
            "Result: Alpers-Huttenlocher syndrome — progressive neurodegeneration + fulminant "
            "liver failure. Fatal within weeks. POLG carrier frequency: ~1:3500. Mandatory "
            "exclusion before VPA in any GABRB2 patient at any age."
        ),
        "action": "Order POLG gene panel BEFORE starting VPA. Document result. If POLG pathogenic variant: ABSOLUTE CI for VPA — substitute LEV, CLB, ETX (absent GI issues), KD. If urgent VPA needed before POLG result: daily LFT + ammonia monitoring + explicit informed consent (signed).",
    },
    {
        "drug": "VPA in pregnancy / females of childbearing age (without specialist oversight)",
        "severity": "HIGH RISK — TERATOGENICITY",
        "risk": "Neural tube defects (2-10%), foetal valproate syndrome, cognitive impairment",
        "mechanism": (
            "VPA is the most teratogenic AED — associated with spina bifida (1-2% first trimester "
            "exposure), cardiac defects, cleft palate, and foetal valproate syndrome (craniofacial "
            "anomalies, limb defects, cognitive impairment in 30-40% of exposed offspring). Dose-"
            "dependent: higher VPA dose → higher teratogenic risk. EU PRAC restriction (2018): "
            "VPA CI in women of childbearing age without valproate pregnancy prevention programme "
            "(VPPP). UK MHRA (2021): annual VPPP review mandatory. Requires documented informed "
            "consent, contraception discussion, and specialist-supervised pregnancy planning."
        ),
        "action": "In females of childbearing age: document VPPP discussion at every visit. High-dose folic acid 5 mg/day if VPA unavoidable. Contraception counselling. Planned pregnancy: attempt VPA taper with alternative AED (LEV, LTG-only if no myoclonic/absence) pre-conception. Emergency contraception counselled. If VPA continued in pregnancy: specialist oversight (foetal anomaly scan at 18-20W).",
    },
]

# ── Monitoring ─────────────────────────────────────────────────────────────────
MONITORING = [
    {"item": "POLG panel before VPA (MANDATORY)", "frequency": "Once before first VPA dose", "threshold": "Any pathogenic POLG variant = VPA absolutely contraindicated", "rationale": "Prevent Alpers-Huttenlocher hepatic failure"},
    {"item": "LFT + ammonia (on VPA)", "frequency": "Baseline → 2W → 1M → q3M", "threshold": "ALT >3× ULN → reduce VPA; ammonia >80 µmol/L → add L-carnitine, review VPA", "rationale": "VPA hepatotoxicity and hyperammonaemia surveillance"},
    {"item": "FBC / platelets (on VPA)", "frequency": "Baseline, q3M", "threshold": "Platelets <80,000 → consider VPA dose reduction or change", "rationale": "VPA-associated thrombocytopaenia"},
    {"item": "VPA therapeutic drug monitoring (TDM)", "frequency": "4W after initiation, q3M stable, after dose change", "threshold": "Trough 50-100 µg/mL (GEFS+/CAE); 80-120 µg/mL (DEE)", "rationale": "Ensure therapeutic levels; sub-therapeutic = breakthrough; supra-therapeutic = toxicity"},
    {"item": "ETX TDM (if on ETX)", "frequency": "4W after start, q6M stable", "threshold": "Target 40-100 µg/mL; recheck when VPA added (VPA raises ETX ~20%)", "rationale": "ETX efficacy and safety monitoring"},
    {"item": "EEG monitoring", "frequency": "Baseline diagnostic; 3M on treatment; annually; after any breakthrough", "threshold": "Persistent 3 Hz SWD at 6M on ETX+VPA → treatment review; worsening background → NCSE or drug effect", "rationale": "Treatment response monitoring; exclude subclinical SE"},
    {"item": "Seizure diary + absence counting", "frequency": "Daily (patient/family)", "threshold": "Absence frequency >10/day on treatment → EEG review + dose adjustment", "rationale": "GEFS+ / CAE treatment monitoring; detect absence acceleration from wrong AED"},
    {"item": "Developmental / neuropsychological assessment", "frequency": "At diagnosis, 12M, 24M; then annually in DEE", "threshold": "Regression of milestones → urgent clinical review + MRI + extended EEG", "rationale": "Detect DEE progression; guide educational support"},
    {"item": "PCOS screening (females on VPA >2 years)", "frequency": "Annually from puberty", "threshold": "Menstrual irregularity + raised LH/FSH or testosterone → endocrinology referral", "rationale": "VPA-associated PCOS risk in adolescent females"},
    {"item": "VPPP review (females of childbearing age on VPA)", "frequency": "Annually", "threshold": "Not on VPPP → immediate specialist referral; pregnancy → emergency obstetric + neurology co-management", "rationale": "EU/UK MHRA teratogenicity prevention programme mandatory"},
    {"item": "SUDEP risk assessment (DEE spectrum)", "frequency": "Annually from age 2 (if not seizure-free)", "threshold": "Uncontrolled nocturnal GTCS + no supervision = highest SUDEP risk tier → intensify management", "rationale": "SUDEP highest in GABRB2-DEE with refractory nocturnal GTCS"},
]

# ── Lifecycle ────────────────────────────────────────────────────────────────
LIFECYCLE = [
    {"phase": "Infancy (0-12 months — DEE/Dravet-like onset)", "description": "Prolonged febrile hemiclonic or GTCS in Dravet-like GABRB2-DEE. EEG + SCN1A-negative → extend to GABRB2 panel. Start VPA (after POLG screen) ± CLB. Fever action plan mandatory. Developmental surveillance monthly."},
    {"phase": "Early childhood (1-5 years — GEFS+ / CAE onset)", "description": "Febrile seizures ± afebrile GTCS in GEFS+. Absence onset age 4-10 in CAE. ETX ± VPA for absence. VPA for GEFS+ GTCS. AVOID CBZ/OXC. EEG at diagnosis. Family history pedigree construction."},
    {"phase": "Mid-childhood (5-10 years)", "description": "CAE: 70% seizure freedom on ETX±VPA. GEFS+: occasional breakthrough GTCS (fever or missed dose). DEE: escalating polypharmacy; KD consideration at ≥2 AED failure. Neuropsychological testing for school planning. Seizure management plan for school."},
    {"phase": "Adolescence (10-18 years)", "description": "GEFS+: majority achieve seizure freedom or rare breakthrough. CAE: remission in ~60%; JME-like evolution in ~20% (adolescent-onset myoclonic ± GTCS = JME overlap). DEE: transition planning. VPA teratogenicity counselling for females MANDATORY. VPPP initiation. Driving law counselling (must be seizure-free ≥1 year in most jurisdictions)."},
    {"phase": "Adulthood (>18 years)", "description": "GEFS+: VPA monotherapy long-term or AED-free (trial if ≥2 years seizure-free). DEE: specialist adult epilepsy centre transition; genetic counselling (AD inheritance — 50% offspring risk if AD-LOF variant). SUDEP counselling. VPPP compliance annual. VPA teratogenicity: reproductive counselling at every visit for females."},
    {"phase": "Pregnancy (all ages)", "description": "VPA: highest-risk AED in pregnancy. VPPP mandatory. Attempt switch to LEV pre-conception if possible (lower teratogenic risk). If VPA unavoidable: high-dose folic acid 5 mg/day, foetal anomaly scan 18-20W, obstetric + neurology co-management. Postpartum: neonatal monitoring (VPA neonatal haemorrhage risk — Vitamin K IV at delivery). Breastfeeding: VPA compatible (low milk concentration)."},
]

# ── Definitions / Concepts ────────────────────────────────────────────────────
CONCEPTS = [
    {"term": "GABRB2 (5q34)", "definition": "Encodes the GABA-A receptor β2 subunit — the most abundantly expressed β subunit in cortex, hippocampus, thalamus, and cerebellum. β2-containing GABA-A receptors (typically α1β2γ2 pentamers) mediate the majority of fast phasic GABAergic inhibition at central synapses. LOF variants reduce synaptic Cl⁻ current, lowering seizure threshold."},
    {"term": "GEFS+ (Generalized Epilepsy with Febrile Seizures Plus)", "definition": "Autosomal dominant familial epilepsy syndrome with febrile seizures onset <6 years AND afebrile generalised seizures (GTCS, absence, myoclonic) persisting beyond age 6. GABRB2 is the β2 subunit aetiology; SCN1A (Dravet/GEFS+) and GABRG2, SCN1B, GABRD are other confirmed genes. Excellent prognosis in most family members. Penetrance ~75%."},
    {"term": "Childhood Absence Epilepsy (CAE)", "definition": "Epilepsy syndrome defined by typical absence seizures (3 Hz bilateral spike-wave, 5-20 sec, abrupt onset/offset, no post-ictal), age 4-10 years, normal development. GABRB2 contributes 3-5% of genetically-diagnosed CAE. Treatment: ETX ± VPA. Most achieve remission by adolescence."},
    {"term": "GABA-A Receptor β2 Subunit / α1β2γ2", "definition": "The canonical synaptic GABA-A receptor is a 2α1:2β2:1γ2 heteropentamer arranged around a central Cl⁻ ion pore. Benzodiazepine binding site is at the α1/γ2 interface; GABA binding sites are at the α1/β2 interfaces × 2. β2 subunit is essential for receptor assembly, synaptic targeting, and channel gating kinetics at synapses."},
    {"term": "Absence Acceleration (Na-channel blockers)", "definition": "Paradoxical worsening of absence seizures by sodium channel blockers (CBZ, OXC, PHT, LTG) via NaV1.1-dependent thalamic reticular nucleus (TRN) interneuron suppression → reduced TRN inhibition of thalamo-cortical relay → enhanced 3 Hz spike-wave. A critical prescribing hazard in GABRB2-CAE and GEFS+ with absence component."},
    {"term": "POLG (gamma-DNA Polymerase) / Alpers-Huttenlocher", "definition": "POLG pathogenic variants → mitochondrial DNA depletion. VPA + POLG → Alpers-Huttenlocher syndrome: fulminant hepatic failure (fatal). POLG panel mandatory before VPA in any GABRB2 patient. Frequency ~1:3500-1:10,000."},
    {"term": "VPPP (Valproate Pregnancy Prevention Programme)", "definition": "EU PRAC (2018) and UK MHRA (2021) mandated programme: women of childbearing age prescribed VPA MUST be enrolled in VPPP — annual specialist review, contraception counselling, risk acknowledgement form. VPA is the most teratogenic commonly used AED (NTD risk 2-10%). VPPP annual compliance is mandatory."},
    {"term": "Dravet-like GABRB2-DEE", "definition": "SCN1A-negative infants with Dravet syndrome-like presentation (prolonged febrile hemiclonic seizures, progressive polymorphic epilepsy, intellectual disability, fever hypersensitivity) due to de novo GABRB2 variants. Distinguished from SCN1A-Dravet by earlier onset, more prominent hypotonia, and less stereotyped febrile pattern. Management parallels Dravet but GABRB2-specific."},
    {"term": "Thalamo-cortical 3 Hz oscillation / TRN", "definition": "The thalamic reticular nucleus (TRN) — a shell of GABA-ergic inhibitory neurons surrounding the thalamus — controls thalamo-cortical synchrony. TRN suppression (by Na-channel blockers or GABRB2 β2 LOF reducing TRN GABA-A currents) → disinhibited thalamocortical relay → hypersynchronous 3 Hz spike-wave → absence. ETX suppresses this by blocking T-type Ca2+ channels in thalamic relay neurons."},
    {"term": "Catamenial Exacerbation / Allopregnanolone", "definition": "Perimenstrual seizure exacerbation in females with GABRB2-epilepsy due to progesterone withdrawal → loss of allopregnanolone (endogenous GABA-A β2-subunit positive allosteric modulator). Management: CLB pulse therapy or hormonal cycle stabilisation. Neurosteroid therapy (ganaxolone) is investigational."},
    {"term": "SUDEP (Sudden Unexpected Death in Epilepsy)", "definition": "SUDEP risk in GABRB2-DEE (drug-resistant, nocturnal GTCS): estimated 1:200-1:500/year. Prevention: seizure alarms, supervised sleep, VPA optimisation. Annual SUDEP risk discussion mandatory in drug-resistant GABRB2-DEE."},
]

# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "polg_before_vpa": "POLG panel mandatory before VPA; any pathogenic POLG variant = VPA absolutely CI",
    "fever_threshold_gefs": "GEFS+: antipyretic at ≥38.0°C; rescue BDZ if seizure >5 min",
    "fever_threshold_dee": "GABRB2-DEE/Dravet-like: antipyretic at ≥37.5°C; lower threshold than GEFS+",
    "vpa_trough_gefs_cae": "VPA trough target 50-100 µg/mL for GEFS+/CAE",
    "vpa_trough_dee": "VPA trough target 80-120 µg/mL for drug-resistant DEE",
    "ammonia_threshold": "Ammonia >80 µmol/L on VPA → add L-carnitine, review VPA dose",
    "etx_target": "ETX trough 40-100 µg/mL; recheck 4W after VPA addition (VPA raises ETX level)",
    "hv_provocation": "3 minutes HV mandatory in EEG; 3 Hz SWD induction = absence confirmation",
    "cbz_oxc_pht_avoidance": "AVOID CBZ/OXC/PHT if any absence or myoclonic seizures documented — absence acceleration + myoclonic worsening",
    "ltg_avoidance": "AVOID LTG if ANY myoclonic seizures or significant absence component — NaV1.1 PV-interneuron block → myoclonic aggravation",
    "tiagabine_ci": "Tiagabine ABSOLUTE CI in all GABRB2 epilepsy (GEFS+/CAE/DEE) — NCSE risk",
    "sudep_annual": "Annual SUDEP counselling if drug-resistant DEE; nocturnal GTCS + no supervision = highest risk tier",
}

# ── Standards ─────────────────────────────────────────────────────────────────
STANDARDS = [
    "ILAE-2022 (Scheffer et al. Epilepsia — GEFS+ classification; DEE framework)",
    "NICE-NG217 (2022) Epilepsies in children and young people — VPA, ETX first-line",
    "CATNAP-2010 (Glauser EA et al. NEJM 2010 — ETX vs VPA vs LTG in CAE; ETX superior for attention + absence freedom)",
    "Berkel-2000-NatGenet (Baulac S et al. Nat Genet 2001 — GABRB2 variants in GEFS+ families)",
    "Lachance-Touchette-2011 (Epilepsia — GABRB2 de novo variants in Dravet-like DEE)",
    "Marson-2012-LancetNeurol (Marson AG et al. — Systematic review LGS/DEE treatments; rufinamide Level B)",
    "MHRA-VPPP-2021 (UK Medicines and Healthcare products Regulatory Agency — Valproate Pregnancy Prevention Programme)",
    "ACMG-AMP-2015 (Richards S et al. Genet Med — variant pathogenicity classification framework)",
]

# ── References ────────────────────────────────────────────────────────────────
REFERENCES = [
    "Baulac S et al. (2001) Familial febrile convulsions and epilepsy through mutations in GABRB2. Nat Genet 28:46. (PMID 11326274) — first GABRB2 GEFS+ family.",
    "Glauser EA et al. (2010) Ethosuximide, valproic acid, and lamotrigine in childhood absence epilepsy. NEJM 362:790. (PMID 20200383) — CATNAP trial: ETX superior to LTG; ETX+VPA standard.",
    "Lachance-Touchette P et al. (2011) Novel α1 and γ2 GABA-A receptor subunit mutations in Dravet syndrome. Epilepsia 52:1887. (PMID 21834830) — de novo GABRB2 in Dravet-like DEE.",
    "Macdonald RL, Kang JQ, Gallagher MJ (2010) Mutations in GABA-A receptor subunits associated with genetic epilepsies. J Physiol 588:1861. (PMID 20406795) — comprehensive GABA-A receptor epilepsy genetics review.",
    "Bhatt SS et al. (2023) Evidence-based pharmacological management of genetic epilepsies. Epilepsia — VPA Level A for generalised epilepsies.",
    "MHRA (2021) Valproate: actions required now from GPs, specialists and dispensers. UK Medicines and Healthcare products Regulatory Agency — VPPP mandatory.",
]

# ── Patient Cohort Simulation ─────────────────────────────────────────────────
_FIRST = ["Amara","Ethan","Sophia","Lucas","Mia","Oliver","Emma","Liam","Isla","Noah",
           "Ava","James","Lily","Leo","Nora","Henry","Grace","Sam","Chloe","Ben",
           "Priya","Kai","Zara","Aiden","Elias","Maya","Clara","Felix","Hugo","Ivy",
           "Ravi","Freya","Finn","Ada","Zoe","Max","Ruby","Oscar","Jack","Leila"]
_LAST  = ["Okafor","Chen","Patel","Singh","Kim","Andersen","Müller","Sato","Hassan","Rossi",
           "García","Park","Thompson","Nguyen","Williams","Martin","Jackson","Yamamoto","Silva","Lee",
           "Brown","Taylor","Wilson","Clark","Moore","White","Davis","Anderson","Smith","Johnson",
           "Thomas","Harris","Martinez","Robinson","Lewis","Walker","Hall","Young","King","Carter"]


def _generate_patients():
    pts = []
    for i in range(40):
        random.seed(SEED + i)
        roll = random.random()
        if roll < 0.40:
            etiology = "GEFS+ AD-LOF"
            onset_years = round(random.uniform(0.5, 6), 1)
            drug_resistant = False
            seizure_free = random.random() < 0.70
            on_vpa = random.random() < 0.80
            on_etx = random.random() < 0.30
            on_clb = False
            on_lev = random.random() < 0.20
            on_kd = False
            dravet_like = False
            absence = random.random() < 0.35
            myoclonic = False
            dd_severity = "none"
        elif roll < 0.60:
            etiology = "CAE de novo/familial"
            onset_years = round(random.uniform(4, 10), 1)
            drug_resistant = random.random() < 0.15
            seizure_free = random.random() < 0.65
            on_vpa = random.random() < 0.70
            on_etx = random.random() < 0.75
            on_clb = random.random() < 0.15
            on_lev = random.random() < 0.20
            on_kd = False
            dravet_like = False
            absence = True
            myoclonic = False
            dd_severity = "none"
        elif roll < 0.80:
            etiology = "Dravet-like DEE de novo"
            onset_years = round(random.uniform(0.2, 1.0), 1)
            drug_resistant = random.random() < 0.55
            seizure_free = random.random() < 0.25
            on_vpa = True
            on_etx = False
            on_clb = random.random() < 0.65
            on_lev = random.random() < 0.50
            on_kd = random.random() < 0.30
            dravet_like = True
            absence = random.random() < 0.40
            myoclonic = random.random() < 0.60
            dd_severity = random.choice(["mild", "moderate", "severe"])
        elif roll < 0.90:
            etiology = "LGS-like DEE de novo"
            onset_years = round(random.uniform(0.3, 2.0), 1)
            drug_resistant = random.random() < 0.75
            seizure_free = random.random() < 0.15
            on_vpa = True
            on_etx = False
            on_clb = random.random() < 0.70
            on_lev = random.random() < 0.55
            on_kd = random.random() < 0.40
            dravet_like = False
            absence = random.random() < 0.60
            myoclonic = random.random() < 0.45
            dd_severity = random.choice(["moderate", "severe"])
        else:
            etiology = "Phenocopy / panel-negative"
            onset_years = round(random.uniform(0.5, 8), 1)
            drug_resistant = random.random() < 0.20
            seizure_free = random.random() < 0.50
            on_vpa = random.random() < 0.60
            on_etx = random.random() < 0.30
            on_clb = random.random() < 0.20
            on_lev = random.random() < 0.30
            on_kd = False
            dravet_like = False
            absence = random.random() < 0.40
            myoclonic = random.random() < 0.15
            dd_severity = random.choice(["none", "mild"])

        polg_tested = "Y" if random.random() < 0.85 else "N"
        vpa_without_polg = on_vpa and polg_tested == "N"
        catamenial = (random.random() < 0.35) if random.random() < 0.50 else False  # female subset
        age_years = round(random.uniform(0.5, 20), 1)
        photosensitive = random.random() < 0.20 if dravet_like else random.random() < 0.05

        pts.append({
            "id": f"GB2-{i+1:03d}",
            "name": f"{random.choice(_FIRST)} {random.choice(_LAST)}",
            "age_years": age_years,
            "etiology": etiology,
            "onset_years": onset_years,
            "drug_resistant": drug_resistant,
            "seizure_free": seizure_free,
            "dravet_like": dravet_like,
            "absence": absence,
            "myoclonic": myoclonic,
            "dd_severity": dd_severity,
            "on_vpa": on_vpa,
            "on_etx": on_etx,
            "on_clb": on_clb,
            "on_lev": on_lev,
            "on_kd": on_kd,
            "polg_tested": polg_tested,
            "vpa_without_polg": vpa_without_polg,
            "photosensitive": photosensitive,
            "catamenial": catamenial,
        })
    return pts


# ── Public API ────────────────────────────────────────────────────────────────
def get_overview():
    """Return GABRB2 overview dict."""
    pts = _generate_patients()
    n = len(pts)
    seizure_free = sum(1 for p in pts if p["seizure_free"])
    drug_resistant = sum(1 for p in pts if p["drug_resistant"])
    dravet_like = sum(1 for p in pts if p["dravet_like"])
    absence = sum(1 for p in pts if p["absence"])
    myoclonic = sum(1 for p in pts if p["myoclonic"])
    on_vpa = sum(1 for p in pts if p["on_vpa"])
    on_etx = sum(1 for p in pts if p["on_etx"])
    on_kd = sum(1 for p in pts if p["on_kd"])
    polg_done = sum(1 for p in pts if p["polg_tested"] == "Y")
    vpa_without_polg = sum(1 for p in pts if p["vpa_without_polg"])
    photosensitive = sum(1 for p in pts if p["photosensitive"])
    gefs_n = sum(1 for p in pts if "GEFS" in p["etiology"])

    return {
        "gene": "GABRB2",
        "locus": "5q34",
        "inheritance": "Autosomal dominant (GEFS+ familial) or de novo (DEE/CAE/Dravet-like)",
        "protein": "GABA-A receptor β2 subunit — most abundant β subunit; core of α1β2γ2 pentameric phasic inhibitory synapse",
        "mechanism": (
            "GABRB2 LOF/GOF → reduced or dysfunctional GABA-A β2-containing receptor current "
            "→ cortical and thalamo-cortical disinhibition → GEFS+ (familial, partial LOF, "
            "fever-sensitive) / CAE (thalamo-cortical 3 Hz oscillation, β2 LOF in TRN) / "
            "DEE-Dravet-like (severe de novo, SCN1A-negative, fever-hypersensitive) / "
            "LGS-like DEE (severe truncating de novo, multiple seizure types)."
        ),
        "key_aha": (
            "GABRB2 (5q34) — GABA-A β2 subunit LOF/GOF → GEFS+/CAE/DEE spectrum. "
            "POLG MANDATORY before VPA — Alpers-Huttenlocher hepatic failure risk. "
            "AVOID CBZ/OXC/PHT/LTG if absence or myoclonic seizures — absence acceleration + "
            "myoclonic aggravation (NaV1.1 PV-interneuron block). "
            "Tiagabine ABSOLUTE CI (NCSE). VPA teratogenicity: VPPP mandatory in females. "
            "ETX first-line for absence component (CAE/GEFS+). KD complements VPA in DEE."
        ),
        "n_patients": n,
        "gefs_pct": round(100 * gefs_n / n),
        "seizure_free_pct": round(100 * seizure_free / n),
        "drug_resistant_pct": round(100 * drug_resistant / n),
        "dravet_like_pct": round(100 * dravet_like / n),
        "absence_pct": round(100 * absence / n),
        "myoclonic_pct": round(100 * myoclonic / n),
        "on_vpa_pct": round(100 * on_vpa / n),
        "on_etx_pct": round(100 * on_etx / n),
        "on_kd_pct": round(100 * on_kd / n),
        "polg_done_pct": round(100 * polg_done / n),
        "vpa_without_polg": vpa_without_polg,
        "photosensitive_pct": round(100 * photosensitive / n),
        "tiagabine_alert": "ABSOLUTE CI in ALL GABRB2 epilepsy (GEFS+/CAE/DEE) — GAT-1 inhibition → extrasynaptic GABA → NCSE",
        "cbz_oxc_alert": "AVOID CBZ/OXC/PHT if ANY absence or myoclonic seizures — absence acceleration + myoclonic aggravation (NaV1.1 PV-interneuron suppression)",
        "vpa_alert": "VPA first-line GABRB2 (GABA enhancement). POLG mandatory before use. VPPP mandatory in females of childbearing age (teratogenicity).",
        "polg_alert": "POLG panel MANDATORY before VPA — Alpers-Huttenlocher fatal hepatic failure in POLG carriers",
        "ltg_alert": "AVOID LTG if myoclonic or significant absence component — NaV1.1 PV-interneuron block → myoclonic aggravation (same Dravet mechanism)",
        "contraindications_summary": [
            "Tiagabine — ABSOLUTE CI: NCSE risk (GAT-1 → extrasynaptic GABA → tonic GABA-A block)",
            "CBZ/OXC/PHT — HIGH RISK: absence acceleration + myoclonic worsening (NaV1.1 PV-interneuron suppression)",
            "LTG — MODERATE-HIGH RISK if myoclonic/absence: NaV1.1 block → myoclonic aggravation",
            "VPA without POLG screen — HIGH RISK: Alpers-Huttenlocher fatal hepatic failure",
            "VPA in females of childbearing age without VPPP — HIGH RISK: teratogenicity (NTD 2-10%)",
        ],
        "thresholds": THRESHOLDS,
        "references": [r.split(" (")[0] for r in REFERENCES],
    }


def get_breakdown():
    """Return GABRB2 breakdown dict."""
    pts = _generate_patients()
    n = len(pts)
    seizure_free = sum(1 for p in pts if p["seizure_free"])
    drug_resistant = sum(1 for p in pts if p["drug_resistant"])
    on_vpa = sum(1 for p in pts if p["on_vpa"])
    on_etx = sum(1 for p in pts if p["on_etx"])
    on_kd = sum(1 for p in pts if p["on_kd"])
    polg_done = sum(1 for p in pts if p["polg_tested"] == "Y")
    vpa_without_polg = sum(1 for p in pts if p["vpa_without_polg"])

    etiology_dist = []
    for ec in ETIOLOGY_CATALOG:
        etiology_dist.append({
            "etiology": ec["etiology"],
            "pct": ec["pct"],
            "n": ec["n"],
            "mechanism_short": ec["mechanism"][:120],
            "eeg_signature_short": ec["eeg_correlate"][:80],
        })

    return {
        "summary": {
            "n": n,
            "seizure_free_pct": round(100 * seizure_free / n),
            "drug_resistant_pct": round(100 * drug_resistant / n),
            "on_vpa_pct": round(100 * on_vpa / n),
            "on_etx_pct": round(100 * on_etx / n),
            "on_kd_pct": round(100 * on_kd / n),
            "polg_done_pct": round(100 * polg_done / n),
            "vpa_without_polg": vpa_without_polg,
        },
        "etiology_distribution": etiology_dist,
        "patients_sample": pts[:15],
        "seizure_types": [
            {"type": s["type"], "prevalence_pct": s["prevalence_pct"]}
            for s in SEIZURE_TYPES
        ],
        "seizure_detail": SEIZURE_TYPES,
        "triggers": [
            {"trigger": t["trigger"], "prevalence_pct": t["prevalence_pct"]}
            for t in TRIGGERS
        ],
        "trigger_detail": TRIGGERS,
        "treatment_detail": TREATMENTS,
        "contraindication_detail": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
    }


def get_definitions():
    """Return GABRB2 definitions dict."""
    return {
        "concepts": CONCEPTS,
        "contraindications_full": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
