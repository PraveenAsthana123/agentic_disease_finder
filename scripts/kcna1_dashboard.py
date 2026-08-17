"""
KCNA1 Epilepsy / Episodic Ataxia Type 1 (EA1 / Kv1.1 / 12p13.32)
===================================================================
40-patient cohort · KCNA1 LOF/GOF · EA1 spectrum · Epilepsy + Myokymia

KCNA1 BIOLOGY:
KCNA1 (12p13.32) encodes Kv1.1, a member of the Shaker-related (Kv1) voltage-gated
potassium channel family. Kv1.1 is expressed at high density in cerebellar basket cell
pinceau terminals (juxtaparanodal at the axon initial segment of Purkinje cells), dorsal
root ganglia peripheral nerve nodes of Ranvier, and limbic/cortical interneurons.

KEY CHANNEL BIOLOGY:
  - Kv1.1 forms homo- and heterotetramers with Kv1.2/Kv1.4/Kv1.6 subunits
  - Dominant-negatively expressed: one LOF subunit can suppress heteromeric channel
  - Kv1.1 principal role: repolarisation at axon initial segment + paranodal region;
    controls action potential threshold and dampens repetitive firing
  - Cerebellar basket cell → Purkinje cell synaptic terminal: Kv1.1 controls K+ outflow
    that terminates basket-cell inhibition on Purkinje cell; Kv1.1 LOF → prolonged
    basket-cell inhibition burst → Purkinje cell misfire → episodic cerebellar ataxia
  - Peripheral nerve: Kv1.1 LOF at nodes of Ranvier → repetitive spontaneous firing →
    neuromyotonia / myokymia (spontaneous motor unit discharges on EMG)

KCNA1 ALLELIC DISORDER SPECTRUM:
  1. Episodic Ataxia Type 1 (EA1) — pure form (~50%)
     Classic EA1: onset childhood-adolescence; brief attacks of cerebellar ataxia lasting
     seconds to 2 minutes; triggered by startle, sudden movement, exercise, emotion, or
     temperature change; continuous interictal myokymia (visible rippling of periorbital,
     finger, and hand muscles); autosomal dominant LOF. Attacks may increase with stress,
     illness, or missed medication. Remission possible in 3rd decade (30-40% partial
     spontaneous improvement). No intellectual disability; normal MRI in most.
     Precision therapy: 4-aminopyridine (4-AP) corrects K+ channel block duration,
     restoring cerebellar basket-cell → Purkinje-cell signalling fidelity.

  2. EA1 with Epilepsy (~25%)
     KCNA1 LOF families with EA1 plus temporal lobe or frontal lobe focal epilepsy (+/-
     secondary GTCS). Original Scheffer 1998 family (Victorian family): KCNA1 T226M caused
     EA1 + temporal lobe epilepsy in 4 generations. Mechanism: Kv1.1 LOF in hippocampal/
     cortical interneuron juxtaparanodal enrichment → reduced fast-spiking interneuron
     repolarisation → hyperexcitability. Seizure onset: mean 12 years (range 4-28 years).
     Management: 4-AP for EA1 + carbamazepine for epilepsy (dual therapy). ~15% drug-resistant.

  3. Myokymia-dominant LOF (~10%)
     Predominant neuromyotonia / myokymia with minimal ataxia; motor nerve hyperexcitability
     manifests as continuous spontaneous muscle twitching (fasciculations + myokymia + cramps).
     EMG: doublets, triplets, multiplets of motor unit potentials. Some overlap with Isaacs
     syndrome (autoimmune CASPR2 antibody). Kv1.1 structural variants that selectively
     affect paranodal/juxtaparanodal Kv1.1 enrichment. Treatment: carbamazepine (Na+
     channel stabilisation reduces repetitive paranodal discharge), mexiletine.

  4. Severe LOF — truncating/splice site (~10%)
     Early-onset EA1 with more frequent and prolonged attacks (>5 min); significant
     epilepsy burden (focal + GTCS); some patients develop mild cerebellar volume loss
     on MRI after repeated episodes. Medication requirement higher: 4-AP + CBZ + VPA
     combination in ~50%. Complete KCNA1 allele loss (NMD) causes most severe phenotype.

  5. Phenocopy / panel-negative (~5%)
     Episodic ataxia without KCNA1 variant: CACNA1A-EA2, SCN8A, ATP1A3, or FHM.
     KCNA1-negative EA1 phenocopy is important differential — distinguishable by acetazolamide
     response (EA2 responsive; EA1 not reliably), duration (EA2: minutes-hours; EA1: seconds),
     and EMG myokymia (EA1 hallmark; absent in EA2).

4-AMINOPYRIDINE (4-AP) PRECISION THERAPY IN EA1:
  4-AP is a broad-spectrum voltage-gated K+ channel blocker that prolongs action potential
  duration at cerebellar basket-cell terminals, increasing neurotransmitter release and
  restoring Purkinje cell inhibition fidelity. In EA1 (Kv1.1 LOF), 4-AP partially
  compensates for reduced Kv1.1 repolarisation capacity.

  FDA-approved extended-release form (Ampyra/dalfampridine): for multiple sclerosis walking
  — NOT specifically approved for EA1. Off-label use for EA1 with growing evidence base.
  4-AP IR (immediate-release): typically 5-10 mg TID for EA1 (expert dosing).

  KEY SAFETY: 4-AP ABSOLUTE CONTRAINDICATION in seizure history without specialist review
  — 4-AP can lower seizure threshold at higher doses by non-specifically blocking Kv1
  channels in cortical interneurons. In EA1-epilepsy patients: carbamazepine controls
  seizures FIRST, then 4-AP at low dose with close monitoring. Do NOT use 4-AP in
  LOF-only patients without epilepsy risk assessment.

  4-AP MONITORING: ECG QTc at baseline and after dose titration; blood pressure (can cause
  hypotension); seizure diary if epilepsy history; avoid in CYD-impaired patients
  (renal dose adjustment required for dalfampridine).

HLA-B*15:02 BEFORE CARBAMAZEPINE:
  CPIC Level A mandate: HLA-B*15:02 testing before CBZ or OXC in patients of Asian
  ancestry. CBZ/OXC used for both EA1 + epilepsy and myokymia control in KCNA1.
  Same risk profile as in KCNQ2/KCNQ3 — SJS/TEN risk in HLA-B*15:02 carriers.
  Substitute LEV (for epilepsy) or mexiletine (for myokymia) if HLA-B*15:02 positive.

POLG MANDATORY BEFORE VPA:
  POLG biallelic pathogenic variants + VPA → Alpers-Huttenlocher syndrome (fatal hepatic
  failure). In KCNA1 patients with epilepsy requiring VPA: exclude POLG before starting.
  Applies to ALL epileptic encephalopathies and genetic epilepsies with VPA use.

TIAGABINE ABSOLUTE CONTRAINDICATION:
  GAT-1 inhibition → extracellular GABA accumulation → tonic GABA-A block → NCSE.
  Class effect in all genetic epilepsies including KCNA1 with epilepsy phenotype.
"""

import random
from datetime import datetime

SEED = 9206  # dashboard 206
random.seed(SEED)

# ── Etiology Catalog ──────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "etiology": "KCNA1 LOF missense — EA1 pure (no epilepsy)",
        "n": 20, "pct": 50,
        "category": "EA1-pure-LOF-no-epilepsy",
        "mechanism": (
            "The most common KCNA1 phenotype (50%): autosomal dominant LOF missense variants "
            "reducing Kv1.1 channel function (haploinsufficiency or dominant-negative via "
            "heteromeric Kv1.1/Kv1.2 channel suppression). Classic EA1: brief episodic ataxia "
            "lasting seconds to 2 minutes, triggered by startle, exercise, or sudden movement; "
            "interictal continuous myokymia (periorbital, finger-rippling visible); no epilepsy. "
            "The pure EA1 form reflects Kv1.1 LOF selectively disrupting cerebellar basket-cell "
            "inhibition of Purkinje cells, with insufficient cortical Kv1 dysfunction to cause "
            "seizures. Precision therapy: 4-aminopyridine (4-AP) restores basket-cell AP "
            "duration → Purkinje cell synaptic fidelity. Prognosis: attacks often decrease in "
            "frequency after 3rd decade; myokymia persists lifelong."
        ),
        "eeg_correlate": (
            "KCNA1 EA1-pure EEG: INTERICTAL NORMAL — no epileptiform discharges. During EA1 "
            "attack: mild diffuse theta slowing (5-6 Hz) reflecting acute cerebellar-thalamic "
            "network disruption; may show bifrontal delta burst. NEVER ictal discharges in "
            "pure EA1 — their presence should prompt reclassification as EA1+epilepsy. "
            "EMG (more informative than EEG): spontaneous doublets, triplets, multiplets of "
            "motor unit potentials consistent with myokymia — present interictally, increases "
            "in amplitude during EA1 attack. EEG primarily used to exclude epileptic basis."
        ),
        "mri_finding": (
            "Brain MRI in EA1-pure: NORMAL in 85%. Findings in long-standing EA1: mild "
            "vermian hypoplasia or atrophy (15% after >10 years disease; correlates with "
            "episode frequency). No cortical atrophy, no deep white matter lesions. "
            "MRI DWI during acute prolonged attack (>5 min): diffusion restriction in "
            "cerebellar foliae (reversible) — represents metabolic stress, not infarction. "
            "Clinical significance: routine MRI to exclude structural ataxia cause "
            "(tumour, vascular, demyelination). Baseline MRI at diagnosis; repeat if attacks "
            "change character or frequency increases."
        ),
        "clinical_note": (
            "Hallmark clinical sign: periorbital and hand myokymia visible on careful "
            "inspection — worm-like rippling contractions. Ask patient to show hands under "
            "bright light. Startle test: sharp handclap triggers brief ataxia (seconds). "
            "4-AP response: 70-80% episode reduction; first-line therapy. Acetazolamide "
            "less effective than in EA2 but worth trying. Myokymia: carbamazepine."
        ),
    },
    {
        "etiology": "KCNA1 LOF missense — EA1 + focal epilepsy / GTCS",
        "n": 10, "pct": 25,
        "category": "EA1-LOF-epilepsy",
        "mechanism": (
            "Approximately 25% of KCNA1 LOF families have epilepsy in addition to EA1, "
            "classically temporal lobe focal seizures ± secondary GTCS. The Scheffer 1998 "
            "Victorian pedigree (KCNA1 T226M) established the KCNA1-epilepsy link across 4 "
            "generations. Mechanism: Kv1.1 LOF reduces fast-spiking interneuron repolarisation "
            "at hippocampal and cortical juxtaparanodal sites → reduced interneuron firing "
            "fidelity → failure to dampen excitatory network oscillations → epileptic seizures. "
            "The threshold for seizures is higher than for ataxia attacks (explaining why EA1 "
            "universally precedes epilepsy onset by years). Drug-resistant epilepsy: ~15%. "
            "Response to carbamazepine: ~65% seizure control. DUAL THERAPY approach needed: "
            "4-AP for ataxia + CBZ/LEV for epilepsy. Counsel family: autosomal dominant risk "
            "for epilepsy in offspring of affected parent ~50%."
        ),
        "eeg_correlate": (
            "EA1+epilepsy EEG: Interictal: focal temporal sharp waves (left or right, ~60% "
            "left) or bitemporal independent IEDs. Ictal: rhythmic theta-delta discharge "
            "with evolution from temporal → perisylvian → generalisation (FBTCS). Video-EEG "
            "important to distinguish EA1 attack (cerebellar — theta slowing, no ictal pattern) "
            "from temporal lobe seizure (focal ictal discharge). BOTH can cause sudden falls; "
            "clinical differentiation by duration (EA1 seconds; TLE minutes) and postictal "
            "state (absent in EA1; present in TLE). MRI epilepsy protocol (3T): hippocampal "
            "volume asymmetry or signal change in ~20% (post-EA1 hippocampal sclerosis rare "
            "but reported with frequent prolonged episodes)."
        ),
        "mri_finding": (
            "EA1+epilepsy MRI: (1) Normal in 70%. (2) Hippocampal asymmetry or mild HS "
            "signal in 20% — if present: consider presurgical evaluation for DRE. "
            "(3) Mild vermian atrophy in 20% (correlated with EA1 attack frequency, "
            "not directly with epilepsy severity). (4) No cortical dysplasia typically "
            "(distinguishes from focal structural epilepsy). Epilepsy protocol 3T MRI "
            "essential: oblique coronal hippocampal T2/FLAIR + volume measurements. "
            "If DRE: SEEG for localization of seizure onset zone."
        ),
        "clinical_note": (
            "Dual phenotype requires 2-specialist coordination: epileptologist for seizures + "
            "movement disorder/ataxia specialist for EA1. 4-AP + CBZ combination: start CBZ "
            "FIRST to control seizures; then add 4-AP at low dose with close monitoring. "
            "Genetic counselling: 50% transmission risk per pregnancy. SUDEP risk present "
            "if GTCS component — provide SUDEP safety counselling."
        ),
    },
    {
        "etiology": "KCNA1 LOF — Myokymia-dominant (neuromyotonia predominant)",
        "n": 4, "pct": 10,
        "category": "Myokymia-dominant-LOF",
        "mechanism": (
            "Approximately 10% of KCNA1 LOF kindreds present with prominent continuous "
            "neuromyotonia/myokymia as the dominant symptom, with minimal or no ataxia. "
            "Kv1.1 is highly enriched at peripheral nerve paranodal and juxtaparanodal "
            "regions of Ranvier nodes, where it normally dampens repetitive axonal firing. "
            "LOF at these peripheral sites → spontaneous repetitive motor unit discharge "
            "(doublets, triplets, multiplets on EMG) manifesting as continuous muscle "
            "rippling, stiffness, cramps, and delayed relaxation. Distinction from acquired "
            "neuromyotonia (Isaacs syndrome): KCNA1 myokymia is from birth/childhood with "
            "family history; Isaacs is adult-onset, autoimmune (CASPR2/LGI1 antibodies), "
            "frequently thymic. Carbamazepine is first-line for KCNA1 myokymia (Na+ channel "
            "stabilisation reduces paranodal repetitive discharge). Mexiletine is alternative."
        ),
        "eeg_correlate": (
            "Myokymia-dominant KCNA1: EEG typically NORMAL (no cortical epileptiform). "
            "EMG is the key diagnostic investigation: spontaneous doublets, triplets, or "
            "multiplets of motor unit potentials firing at 50-150 Hz — myokymia pattern. "
            "May also show fibrillation potentials (secondary to chronic motor nerve "
            "hyperexcitability). NCS: normal MCV and SNAP amplitudes. F-waves: normal "
            "or mildly prolonged. Needle EMG: pathological spontaneous activity with "
            "characteristic grouped doublet/triplet discharges. Motor unit: normal morphology "
            "(distinguishes from neuropathy where MU morphology is abnormal)."
        ),
        "mri_finding": (
            "Myokymia-dominant: Brain MRI NORMAL — no cerebellar or cortical changes "
            "expected. Peripheral nerve MRI (MRN): may show mild nerve T2 hyperintensity "
            "at high-resolution if severe neuromyotonia (research tool, not routine). "
            "Thigh MRI: mild signal change in chronically affected muscles (non-specific). "
            "Chest CT: thymus assessment to exclude thymoma (in Isaacs DDx). If KCNA1 "
            "variant confirmed: no further neuroimaging needed. Brain MRI done at "
            "diagnosis to exclude structural ataxia/cerebellar pathology."
        ),
        "clinical_note": (
            "Key differential: Isaacs syndrome (CASPR2/LGI1 antibodies — check serum). "
            "KCNA1 myokymia: childhood onset, family history, no autonomic features, "
            "EMG without neurotonic discharges at rest (vs Isaacs). Carbamazepine "
            "first-line: 10-20 mg/kg/day divided. Mexiletine 150-300 mg/day alternative. "
            "Physiotherapy for muscle cramping. Genetic cascade testing mandatory."
        ),
    },
    {
        "etiology": "KCNA1 LOF truncating / splice — severe EA1 + epilepsy",
        "n": 4, "pct": 10,
        "category": "Severe-LOF-truncating-EA1-epilepsy",
        "mechanism": (
            "Truncating and splice-site KCNA1 variants (frameshift, nonsense, large deletion) "
            "causing complete loss of one KCNA1 allele via NMD — most severe KCNA1 phenotype. "
            "Severe haploinsufficiency: Kv1.1 protein reduced to ~30-40% of normal (below "
            "functional compensation threshold for Kv1 heterotetramers in cerebellum and "
            "hippocampus). Clinical: EA1 attacks more frequent (multiple per day) and longer "
            "(>2 min); significant epilepsy burden (drug-resistant in up to 40%); some develop "
            "mild cerebellar atrophy after repeated prolonged episodes; rare mild intellectual "
            "disability in most severely affected. Triple therapy required: 4-AP + CBZ + VPA "
            "(after POLG) in ~50%. Penetrance: ~90% (higher than missense LOF ~80%)."
        ),
        "eeg_correlate": (
            "Severe EA1+epilepsy EEG: Frequent interictal IEDs — bitemporal or bifrontal "
            "independent sharp waves; may show generalised burst with bifrontal predominance "
            "in sleep. Ictal: complex temporal or frontal seizure onset patterns with rapid "
            "generalisation. GTCS: typical generalised ictal recruiting rhythm. Video-EEG "
            "essential to map seizure semiology and confirm focal onset zone. Sleep EEG: "
            "may show NREM sleep IED activation (similar to BECTS-like but without "
            "centrotemporal morphology). Serial EEG every 6 months to monitor burden "
            "and guide AED optimisation."
        ),
        "mri_finding": (
            "Severe LOF MRI: (1) Cerebellar vermian atrophy in 35% after >5 years disease — "
            "correlates with cumulative episode frequency. (2) Hippocampal changes in 25% "
            "(mild HS after repeated status-level episodes). (3) Normal cortex and white "
            "matter in most. (4) No structural malformation — purely functional/post-insult. "
            "MRI 3T epilepsy protocol + cerebellar volumetry annually. DWI during acute "
            "prolonged episode: check for restricted diffusion in cerebellar folia "
            "(reversible if treated promptly). If cerebellar atrophy progressing: consider "
            "VNS referral for refractory epilepsy burden reduction."
        ),
        "clinical_note": (
            "Highest SUDEP risk subgroup — GTCS + drug resistance. Nocturnal GTCS monitoring "
            "device (video cam or wearable). VNS referral after 2 AED failures. Rescue "
            "benzodiazepine plan for prolonged EA1 episodes that resemble status. "
            "4-AP: use cautiously — low dose 5 mg TID; ECG monitoring mandatory. "
            "POLG mandatory before VPA. Genetic counselling: 50% offspring risk."
        ),
    },
    {
        "etiology": "Phenocopy / KCNA1-negative episodic ataxia",
        "n": 2, "pct": 5,
        "category": "Phenocopy-CACNA1A-ATP1A3-SCN8A",
        "mechanism": (
            "5% of cohort with episodic ataxia phenotype are KCNA1-negative on gene panel. "
            "Most common phenocopies: (1) CACNA1A-EA2 — P/Q-type Ca2+ channel LOF; "
            "attacks minutes-hours (longer than EA1), responds to acetazolamide, no myokymia. "
            "(2) ATP1A3-AHC/RDP — Na+/K+-ATPase α3 subunit; alternating hemiplegia component. "
            "(3) SCN8A-DEE with ataxia. (4) FHM/hemiplegic migraine overlap with episodic "
            "ataxia. Key discriminating features: myokymia (present in KCNA1-EA1, absent in "
            "CACNA1A-EA2); attack duration (EA1: seconds-2 min; EA2: minutes-6 hours); "
            "interictal findings (EA1: myokymia on EMG; EA2: nystagmus + normal EMG); "
            "acetazolamide response (EA2: ~75%; EA1: partial). Broad episodic ataxia "
            "gene panel essential: KCNA1, CACNA1A, ATP1A3, SCN8A, SLC1A3, FITM2."
        ),
        "eeg_correlate": (
            "Phenocopy: EEG and EMG findings depend on underlying aetiology. CACNA1A-EA2: "
            "interictal EEG normal or mild posterior slowing; ictal: diffuse slowing during "
            "attack. ATP1A3-AHC: ictal EEG asymmetric slowing with hemispheric lateralisation "
            "correlating with affected limb. SCN8A: focal IEDs. EMG: KCNA1-negative panel "
            "patients do NOT show myokymia doublets/triplets (key negative finding pointing "
            "to CACNA1A or other aetiology). EMG is the critical discriminating investigation "
            "— myokymia = KCNA1; no myokymia = investigate CACNA1A, ATP1A3, SCN8A."
        ),
        "mri_finding": (
            "Phenocopy MRI: CACNA1A-EA2: may show cerebellar atrophy (more common and "
            "earlier than EA1); SCA6 allelic. ATP1A3-AHC: normal or transient restricted "
            "diffusion in striatum during acute event. SCN8A: normal MRI. Critical: "
            "careful review of MRI for cerebellar atrophy pattern helps differentiate. "
            "EA1-KCNA1: atrophy late-onset and mild. EA2-CACNA1A: atrophy earlier and "
            "more prominent, often with vermian hypoplasia."
        ),
        "clinical_note": (
            "Never assume KCNA1 without genetic confirmation. Test CASPR2/LGI1/VGKC antibodies "
            "to exclude acquired Isaacs. Order broad episodic ataxia panel + EMG for myokymia. "
            "Acetazolamide therapeutic trial: EA2-CACNA1A responds (>75% episode reduction) "
            "whereas EA1-KCNA1 has partial response only — this helps guide testing priority. "
            "4-AP trial: effective in EA1-KCNA1; less so in CACNA1A."
        ),
    },
]

# ── Seizure Types ─────────────────────────────────────────────────────────────
SEIZURE_TYPES = [
    {
        "type": "Focal Aware / Impaired Awareness Seizures (temporal/frontal)",
        "prevalence_pct": 45,
        "age_window": "Mean onset 12 years (range 4-28 years); rare <5 years",
        "eeg_correlate": (
            "Focal temporal or frontal ictal discharge: rhythmic theta evolving to alpha/beta "
            "frequency, typically left temporal (60%) or bilateral independent. Post-ictal "
            "slowing ipsilateral 30-60 seconds. Distinguish from EA1 attack: EA1 shows "
            "diffuse theta WITHOUT structured focal ictal evolution."
        ),
        "clinical_tip": (
            "Ask about ALL episodic events — many patients attribute focal seizures (brief "
            "déjà vu, rising epigastric aura, automatisms) to 'another ataxia attack'. "
            "Video-EEG during both seizure types is essential for dual phenotype patients. "
            "Duration: EA1 = seconds; focal TLE = 60-120 seconds."
        ),
    },
    {
        "type": "Focal to Bilateral Tonic-Clonic (FBTCS / GTCS)",
        "prevalence_pct": 35,
        "age_window": "Onset typically adolescence-early adulthood; may precede EA1 recognition",
        "eeg_correlate": (
            "Standard FBTCS pattern: focal onset ictus evolving to bilaterally synchronous "
            "tonic phase (low-voltage fast) → clonic phase (rhythmic generalised spike-wave). "
            "Postictal suppression 30-120 seconds. Interictal: temporal or frontal IEDs. "
            "GTCS without focal signature: consider genetic generalised epilepsy overlap — "
            "order broad epilepsy gene panel."
        ),
        "clinical_tip": (
            "SUDEP risk assessment mandatory for GTCS: SUDEP-7 risk score. Nocturnal GTCS "
            "highest SUDEP risk — wearable seizure detection or bed sensor. Rescue buccal "
            "midazolam 10 mg prescribed. Driving restriction counselling. POLG before VPA."
        ),
    },
    {
        "type": "Episodic Ataxia Attack (EA1 cerebellar episode — brief)",
        "prevalence_pct": 100,
        "age_window": "All KCNA1 patients; childhood/adolescence onset; may persist lifelong",
        "eeg_correlate": (
            "EA1 attack EEG: diffuse theta slowing (5-7 Hz) without focal ictal discharge. "
            "Absence of structured focal evolution distinguishes from seizure. May show "
            "diffuse voltage attenuation at attack peak (cortical arousal suppression). "
            "Simultaneous EMG: myokymia increase during attack. Scalp EEG during EA1 attack "
            "is not diagnostic — clinical observation + EMG is more informative."
        ),
        "clinical_tip": (
            "EA1 diagnosis is CLINICAL: brief startle-induced ataxia + myokymia on examination. "
            "Patient education: carry 4-AP; avoid known triggers; carry rescue card. "
            "Attack frequency diary essential. Response to 4-AP: >50% frequency reduction "
            "in 70-80% — document as primary outcome measure."
        ),
    },
    {
        "type": "Myoclonic Seizures (rare, epilepsy subtype only)",
        "prevalence_pct": 10,
        "age_window": "Childhood-onset; may herald later GTCS in EA1+epilepsy families",
        "eeg_correlate": (
            "Myoclonic seizures: generalised polyspike or polyspike-slow wave discharge "
            "correlating with clinical jerk. Distinguish from myokymia (continuous, low "
            "amplitude, peripheral): myoclonic EEG shows cortical correlate (polyspike) "
            "whereas myokymia shows no cortical EEG change. EMG back-averaging: myoclonus "
            "has EEG correlate; myokymia does not."
        ),
        "clinical_tip": (
            "Ask specifically about morning jerks (myoclonic seizures often on awakening). "
            "VPA is most effective for myoclonic component — POLG screen mandatory first. "
            "Distinguish myoclonic seizure from myokymia jerk — the latter is constant "
            "and low-amplitude; seizure myoclonus is sudden and involves proximal muscles."
        ),
    },
]

# ── Triggers ──────────────────────────────────────────────────────────────────
TRIGGERS = [
    {
        "trigger": "Startle / Sudden unexpected noise or touch",
        "prevalence_pct": 95,
        "mechanism": (
            "Startle reflex activates pontine reticular formation → rapid cortical arousal "
            "network activation → transient increase in cerebellar mossy fibre input → "
            "basket-cell discharge burst in milieu of reduced Kv1.1 at basket-cell "
            "pinceau terminals → impaired timing of Purkinje cell inhibition → "
            "brief cerebellar dyssynergia. The Kv1.1 LOF amplifies the normal startle-"
            "associated motor network response into a pathological ataxia attack."
        ),
        "clinical_management": (
            "Patient education: warn all household members to avoid sudden loud noises. "
            "Startle-avoidance strategies: verbal warning before contact. 4-AP reduces "
            "startle-triggered episode frequency by 70-80%. Acoustic environment modification "
            "for severely affected patients (noise-dampening home modifications)."
        ),
    },
    {
        "trigger": "Sudden movement / physical exercise / exertion",
        "prevalence_pct": 85,
        "mechanism": (
            "Rapid transition from rest to movement activates cerebellar predictive motor "
            "control circuits. In EA1: reduced Kv1.1 → impaired timing of fast-spiking "
            "basket-cell bursts during motor initiation → Purkinje cell error signal "
            "amplification → ataxia. Exercise-induced: sustained repetitive movement "
            "progressively depletes basket-cell K+ buffering capacity → cumulative episode "
            "risk during prolonged physical activity."
        ),
        "clinical_management": (
            "Adapt physical activity: avoid sudden starts, warm up gradually. Sports: "
            "swimming (continuous rhythm reduces startle risk) preferable to racket sports "
            "(sudden ball impact triggers). 4-AP before anticipated exertion activity. "
            "School/work: alert PE teacher/employer, written activity plan."
        ),
    },
    {
        "trigger": "Emotional stress / anxiety / excitement",
        "prevalence_pct": 72,
        "mechanism": (
            "Emotional arousal activates amygdala-hypothalamic-brainstem networks that "
            "project to cerebellum (fastigial nucleus) modulating cerebellar Purkinje cell "
            "activity. Stress-induced norepinephrine release reduces Kv1 channel surface "
            "expression via PKC phosphorylation of Kv1.1 N-terminal domains → further "
            "reduces already-compromised M-current reserve → lower episode threshold. "
            "Emotional trigger disproportionately affects adolescents and young adults "
            "(autonomic instability of adolescence amplifies threshold reduction)."
        ),
        "clinical_management": (
            "CBT-based stress management; mindfulness for anxiety. Acetazolamide can reduce "
            "stress-triggered episodes (carbonic anhydrase inhibition → mild acidosis → "
            "Kv channel stability). Beta-blocker propranolol (off-label) to reduce adrenergic "
            "drive — caution in exercise-induced asthma and with 4-AP (pharmacodynamic "
            "interaction: propranolol ↑ 4-AP concentration slightly)."
        ),
    },
    {
        "trigger": "Fever / elevated body temperature (including hot bath/sauna)",
        "prevalence_pct": 65,
        "mechanism": (
            "Hyperthermia directly reduces Kv1.1 channel open probability (negative Q10 "
            "effect on gating kinetics) — temperature rise of 1°C reduces Kv1.1 open "
            "probability by ~15-20%. In EA1 patients with baseline ~50% Kv1.1, fever "
            "to 38°C reduces functional Kv1 below the ataxia threshold. Hot water "
            "immersion (hot bath, sauna): body temperature rises 1-1.5°C rapidly → "
            "immediate episode trigger. Management: temperature-sensitive KCNA1 variants "
            "documented (T226M family: significant hyperthermia sensitivity)."
        ),
        "clinical_management": (
            "Written fever action plan: antipyretics at 37.8°C threshold. Hot bath/sauna: "
            "absolute avoidance or cool environment. Summer precautions (outdoor heat). "
            "Fever during illness: 4-AP dose timing and antipyretic prophylaxis plan "
            "discussed with neurologist. Hot drinks: generally safe (no core temperature "
            "change); hot bath: avoid. Document fever threshold in patient record."
        ),
    },
    {
        "trigger": "Sleep deprivation / irregular sleep-wake schedule",
        "prevalence_pct": 55,
        "mechanism": (
            "Sleep deprivation activates HPA axis → elevated cortisol → Kv1 channel "
            "downregulation via glucocorticoid receptor-mediated transcriptional suppression. "
            "Additionally: sleep deprivation increases cerebellar network excitability through "
            "reduced adenosine-mediated inhibition (adenosine A1 receptors). Combined effect: "
            "sleep-deprived EA1 patients have lower ataxia-episode threshold and — if epilepsy "
            "present — higher seizure risk. Sleep deprivation most relevant for EA1+epilepsy "
            "subgroup where it triggers both seizures and ataxia attacks."
        ),
        "clinical_management": (
            "Strict sleep hygiene counselling. Seizure diary should include sleep quality. "
            "Target 8-9h sleep for school-age; 7-8h adults. Shift-work contraindicated. "
            "Melatonin for sleep-onset insomnia (safe with 4-AP and CBZ). If seizures "
            "cluster following sleep deprivation: emergency rescue plan activation."
        ),
    },
    {
        "trigger": "Missed 4-AP / AED dose",
        "prevalence_pct": 68,
        "mechanism": (
            "4-AP half-life: immediate-release ~3.5 hours; extended-release (dalfampridine) "
            "~5-6 hours. Missed dose → trough blood level below therapeutic range → "
            "loss of K+ channel compensation in cerebellar basket cells → episode within "
            "4-8 hours of missed dose. CBZ for seizures: missed dose creates 6-12 hour "
            "AED gap → breakthrough seizure risk. Double-dosing is NOT recommended; "
            "take next scheduled dose on time."
        ),
        "clinical_management": (
            "Electronic reminder apps for medication timing. Pill organiser. If >4h since "
            "missed 4-AP: take immediately; if within 2h of next dose: skip and resume schedule. "
            "Travel plan: carry adequate supply of 4-AP; airport security card for medication. "
            "CBZ: sustained-release formulation preferred for seizure component (less trough "
            "fluctuation). Patient/family education: never double-dose."
        ),
    },
    {
        "trigger": "Hyperventilation / respiratory alkalosis",
        "prevalence_pct": 45,
        "mechanism": (
            "Hyperventilation → hypocapnia → respiratory alkalosis (pH >7.45) → reduced "
            "ionised calcium → increased neuronal membrane excitability. In EA1: alkalosis "
            "reduces Kv1.1 open probability by altering channel pH-gating mechanism. "
            "Combined effect: metabolic shift in cerebellar basket cells → reduced repolarisation "
            "capacity → lower episode threshold. Acetazolamide works in part by creating "
            "mild metabolic acidosis that counteracts alkalosis-induced channel dysfunction."
        ),
        "clinical_management": (
            "Breathing retraining to avoid hyperventilation (diaphragmatic breathing). "
            "Avoid prolonged crying or sighing breathing patterns. Exercise protocol: "
            "nasal breathing where possible. Acetazolamide particularly useful in "
            "hyperventilation-sensitive patients (mechanism: mild acidosis maintains "
            "Kv1.1 pH-gating in optimal range)."
        ),
    },
    {
        "trigger": "Intercurrent illness / infection / immune activation",
        "prevalence_pct": 38,
        "mechanism": (
            "Pro-inflammatory cytokines (IL-1β, TNF-α, IL-6) downregulate Kv1 channel "
            "surface expression via NF-κB signalling pathway — reduces already-compromised "
            "Kv1.1 function in acute infection. Systemic inflammation also activates "
            "cerebellar microglia → neuroinflammatory milieu in basket-cell region → "
            "lower episode threshold. Compound with fever effect: illness + fever creates "
            "double hit on Kv1.1 functional reserve."
        ),
        "clinical_management": (
            "Proactive illness management plan: antipyretics at 37.8°C threshold; "
            "adequate hydration; rest during acute illness. If seizure history: "
            "emergency rescue plan active during any febrile illness. Brief course of "
            "increased 4-AP frequency (extra dose) during illness with physician guidance. "
            "Document illness-triggered episodes in diary."
        ),
    },
]

# ── Treatments ────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "4-Aminopyridine (4-AP) / Dalfampridine-ER",
        "level": "Level A — EA1 precision therapy (K+ channel blocker — compensates Kv1.1 LOF)",
        "dose": (
            "4-AP IR (immediate-release): 5-10 mg TID; start 2.5 mg TID and uptitrate "
            "over 2 weeks. Off-label for EA1. "
            "Dalfampridine ER (Ampyra, 10 mg BID): FDA-approved for MS walking; "
            "off-label for EA1 with Level A evidence. Maximum: 10 mg BID (ER); "
            "20-30 mg/day total (IR). Renal dose adjustment: CrCl 50-80 mL/min: "
            "max 10 mg/day ER. CI if CrCl <50 mL/min."
        ),
        "moa": (
            "4-AP blocks voltage-gated K+ channels (primarily Kv1 family) in a "
            "voltage-dependent manner, prolonging action potential duration at the "
            "cerebellar basket-cell axon terminal. This restores normal temporal summation "
            "of basket-cell inhibitory postsynaptic currents on Purkinje cells, correcting "
            "the dysregulated inhibitory timing that causes EA1 ataxia attacks. 4-AP does "
            "NOT restore lost Kv1.1 channel protein — it compensates by slowing K+ "
            "efflux through residual K+ channels, effectively increasing AP width."
        ),
        "efficacy": (
            "EA1 episode frequency: 70-80% reduction in attack frequency with 4-AP. "
            "Interictal myokymia: partial reduction (~50%). Not effective for seizures "
            "(may worsen in LOF without seizure control). Level A evidence: multiple "
            "crossover trials and international registry data (Strupp 2008, Jen 2004)."
        ),
        "safety": (
            "SEIZURE RISK: 4-AP can lower seizure threshold in patients WITH epilepsy — "
            "ensure seizures controlled on CBZ/LEV BEFORE starting 4-AP. "
            "QTc prolongation (rare at therapeutic doses; monitor ECG at baseline and "
            "after uptitration). Hypertension (monitor BP). Dizziness, nausea, insomnia. "
            "CONTRAINDICATED in CrCl <50 mL/min. No teratogenicity data — avoid pregnancy."
        ),
        "monitoring": "ECG QTc baseline + 2 weeks after each dose increase. BP at each visit. Renal function (eGFR) 3-monthly. Episode frequency diary. Seizure diary if epilepsy present.",
        "kcna1_note": (
            "KCNA1-SPECIFIC PRECISION THERAPY: compensates Kv1.1 LOF by prolonging AP duration "
            "at basket-cell terminals. Start after seizure control established. 4-AP should "
            "NEVER be started without first ensuring adequate seizure control (CBZ/LEV). "
            "Document functional assay or genetic confirmation of KCNA1 LOF before prescribing."
        ),
    },
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC)",
        "level": "Level B — EA1 + epilepsy (Na+ channel stabilisation + myokymia control)",
        "dose": (
            "CBZ: 400-1200 mg/day PO divided q8-12h; target serum level 4-12 µg/mL. "
            "OXC: 600-1800 mg/day PO divided q12h (MHD target 10-35 µg/mL). "
            "Myokymia monotherapy: CBZ 200-400 mg/day often sufficient. "
            "Titrate over 4 weeks. Sustained-release preferred for seizure component."
        ),
        "moa": (
            "Voltage-gated Na+ channel use-dependent fast-inactivation stabilisation — "
            "reduces repetitive high-frequency neuronal firing. In KCNA1 EA1: particularly "
            "effective for (1) myokymia (paranodal repetitive firing reduction) and "
            "(2) focal epilepsy (cortical Na+ channel stabilisation reduces seizure "
            "initiation). Note: CBZ/OXC do NOT enhance Kv1.1 function — their action "
            "on Na+ channels is the primary benefit in KCNA1. Complementary to 4-AP: "
            "different mechanism means additive benefit in dual phenotype."
        ),
        "efficacy": (
            "Myokymia: 70-85% symptom reduction with CBZ. Focal seizures in EA1+epilepsy: "
            "~65% seizure control at 12 months. FBTCS: ~75% response. "
            "EA1 attacks: modest benefit (~20-30% additional frequency reduction in some "
            "patients — secondary benefit from reduced cortical excitability spill-over). "
            "Drug-resistant subgroup: ~15% fail CBZ monotherapy."
        ),
        "safety": (
            "HLA-B*15:02 MANDATORY before CBZ or OXC in Asian ancestry (SJS/TEN risk, CPIC Level A). "
            "OXC: SIADH — monitor serum sodium (baseline, Day 3, 7, 14, then monthly). "
            "Both: enzyme induction. CBZ: mild cognitive effects; diplopia at high doses. "
            "Teratogenic — use contraception; pre-pregnancy VPPP counselling. "
            "HLA-A*31:01 (pan-ancestry): oxcarbazepine DRESS risk — test if available."
        ),
        "monitoring": "HLA-B*15:02 before first dose. Na+ (OXC). CBZ level. LFTs baseline. FBC baseline. BP. Growth in children.",
        "kcna1_note": (
            "FIRST-LINE for myokymia and seizures in KCNA1. Start BEFORE 4-AP to establish "
            "seizure control. Dual phenotype: CBZ (seizures + myokymia) + 4-AP (EA1 attacks) "
            "is the standard combination. HLA-B*15:02 and OXC-SIADH protocols mandatory."
        ),
    },
    {
        "drug": "Acetazolamide",
        "level": "Level B — EA1 episodes (carbonic anhydrase inhibitor; acidosis-mediated Kv stability)",
        "dose": "250-1000 mg/day PO divided BID-TID. Start 125 mg OD; uptitrate over 2 weeks. Renal dose adjustment: CrCl 10-50 mL/min: use with caution.",
        "moa": (
            "Carbonic anhydrase inhibition → CO2 retention → mild metabolic acidosis → "
            "increased intracellular H+ → stabilisation of Kv1 channel gating (pH-sensitive "
            "gating). Also: bicarbonate loss via kidney reduces alkalosis-triggered episodes. "
            "Mechanism less specific than 4-AP (does not directly target Kv1.1) but provides "
            "additive benefit in some patients. Particularly useful for hyperventilation-"
            "triggered and alkalosis-sensitive KCNA1 patients."
        ),
        "efficacy": "EA1 episodes: 40-60% additional frequency reduction as add-on to 4-AP. Less effective than 4-AP as monotherapy. Evidence level: case series and expert consensus.",
        "safety": "Electrolyte imbalance (hyponatraemia, hypokalaemia — monitor). Kidney stones (increase fluid intake to 2L/day). Paraesthesias (hands/feet) — common, transient. Sulfonamide allergy cross-reactivity. Teratogenic — avoid pregnancy.",
        "monitoring": "Electrolytes (Na+, K+, HCO3-) at baseline, 1M, 3M then 6-monthly. Renal US annually. Urine pH weekly initially.",
        "kcna1_note": "Add-on to 4-AP for partial responders; first-line alternative if 4-AP not tolerated. Particularly effective in patients with hyperventilation and alkalosis-triggered episodes.",
    },
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B — Adjunct for focal epilepsy / GTCS in EA1+epilepsy",
        "dose": "Adults: 500-3000 mg/day PO divided q12h. Children: 40-60 mg/kg/day. IV available for acute seizure management.",
        "moa": "SV2A synaptic vesicle protein binding → impaired neurotransmitter vesicle exocytosis → reduced cortical excitatory tone. No direct KCNA1/Kv1 action.",
        "efficacy": "Focal epilepsy + GTCS: ~55% ≥50% seizure reduction as adjunct to CBZ. Well tolerated in adolescents and adults. No significant pharmacokinetic interactions with 4-AP or CBZ.",
        "safety": "Behavioural effects (irritability, aggression, depression) in 10-15% — counsel patients and monitor. Rarely: psychosis. Dose-related: dizziness, somnolence. Renal dose adjustment (CrCl <80 mL/min).",
        "monitoring": "Mood/behaviour diary. Renal function (LEV is renally cleared). Drug level optional (10-40 µg/mL therapeutic range). Suicide risk screening.",
        "kcna1_note": "Safe combination with 4-AP (no pharmacokinetic interaction). Use for focal epilepsy or GTCS when CBZ is not tolerated or when additional seizure control needed. Behavioural monitoring mandatory in adolescents.",
    },
    {
        "drug": "Valproate / Sodium Valproate (VPA)",
        "level": "Level B — After POLG screen (GTCS / myoclonic seizures in epilepsy subtype)",
        "dose": "15-40 mg/kg/day PO divided q8-12h. Target serum level 50-100 µg/mL. VPA TDM q3 months.",
        "moa": "Multiple: Na+ channel use-dependent inactivation; GABA transaminase inhibition → increased GABAergic tone; T-type Ca2+ channel suppression. Broad-spectrum AED for GTCS and myoclonic seizures.",
        "efficacy": "GTCS and myoclonic seizures in EA1+epilepsy subtype: ~70% seizure control as add-on. Particularly useful for myoclonic component.",
        "safety": "POLG MANDATORY before use (Alpers-Huttenlocher risk). Teratogenic (NTD risk — VPPP mandatory in women). Hepatotoxicity, pancreatitis, thrombocytopaenia, weight gain. PCOS in long-term female users.",
        "monitoring": "POLG screen mandatory (document result). LFTs + ammonia baseline, 1M, 3M, 6-monthly. FBC (platelets). TDM q3M. Carnitine levels. VPPP counselling annually for women.",
        "kcna1_note": "MANDATORY POLG screen before VPA in KCNA1 patients. Reserve for GTCS or myoclonic seizures not controlled on CBZ+LEV. Teratogenicity counselling mandatory for all women of childbearing potential.",
    },
    {
        "drug": "Mexiletine",
        "level": "Level B — Myokymia (Na+ channel blocker; alternative to CBZ for neuromyotonia)",
        "dose": "150-300 mg TID PO. Start 150 mg OD; increase weekly. ECG QTc monitoring mandatory.",
        "moa": "Class Ib antiarrhythmic Na+ channel blocker — binds inactivated Na+ channels, stabilising paranodal repetitive discharge. Particularly active at peripheral nerve Na+ channels (Nav1.8, Nav1.9) mediating motor axon hyperexcitability. Used for channelopathy-related neuromyotonia.",
        "efficacy": "Myokymia control: 60-70% symptom reduction. Alternative when CBZ not tolerated or in myokymia-dominant variant without epilepsy requiring Na-channel stabilisation.",
        "safety": "QTc prolongation — MANDATORY ECG before starting and at each dose titration. Arrhythmia risk. GI intolerance (nausea, dyspepsia). Hepatotoxicity (rare). CI: structural heart disease, prolonged QTc >450ms at baseline.",
        "monitoring": "ECG QTc before each dose increase, then 3-monthly on stable dose. LFTs at 1M, 3M. Cardiac history review. BP.",
        "kcna1_note": "Alternative to CBZ for myokymia-dominant KCNA1 (especially if seizure control not required). ECG mandatory — do NOT start without baseline QTc. Combination mexiletine + 4-AP: monitor for additive QTc prolongation.",
    },
    {
        "drug": "Ketogenic Diet (KD)",
        "level": "Level B — Drug-resistant epilepsy in KCNA1+epilepsy subtype (≥2 AEDs failed)",
        "dose": "Classical 4:1 or Modified Atkins Diet (MAD). Initiated by epileptologist + dietitian. β-OHB target: 2-4 mmol/L. Multivitamin supplementation mandatory.",
        "moa": "Multiple metabolic anti-seizure mechanisms: KATP channel opening, AMPA receptor inhibition, mitochondrial biogenesis enhancement, GABA metabolism improvement. Independent of Kv1.1 pathway — effective even with severe LOF.",
        "efficacy": ">50% seizure reduction in ~50% of DRE-KCNA1+epilepsy patients at 3 months. Does not address EA1 attacks directly.",
        "safety": "Dyslipidaemia, kidney stones, constipation, growth restriction. Metabolic acidosis (do NOT combine high-ratio KD with acetazolamide — compounded acidosis). KD relative CI: pancreatitis, renal failure.",
        "monitoring": "β-OHB daily initially. Lipid panel, renal US, urinalysis q3M. Growth chart monthly. Dietitian review monthly initially.",
        "kcna1_note": "Last resort for epilepsy component when ≥2 AEDs failed. Note: do NOT combine high-ratio KD + acetazolamide (compounded metabolic acidosis risk). Prefer MAD (Modified Atkins) if concurrent acetazolamide needed.",
    },
]

# ── Contraindications ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Tiagabine",
        "severity": "ABSOLUTE CONTRAINDICATION",
        "risk": "Non-convulsive status epilepticus (NCSE)",
        "mechanism": (
            "Tiagabine inhibits GAT-1 (GABA transporter 1) → extracellular GABA accumulation "
            "→ sustained tonic GABA-A receptor activation → paradoxical cortical depolarisation "
            "block → NCSE. In KCNA1+epilepsy patients: already reduced cortical inhibitory "
            "interneuron function (Kv1.1 LOF) + GAT-1 block = catastrophic GABA dysregulation. "
            "Class effect across all genetic epilepsies. NCSE from tiagabine is difficult to "
            "treat and may require general anaesthesia."
        ),
        "action": "NEVER PRESCRIBE tiagabine in KCNA1 patients, even those without current epilepsy (EA1-pure). Document absolute CI in medical record. If inadvertently prescribed: stop immediately, hospitalise, IV benzodiazepine for NCSE management.",
    },
    {
        "drug": "4-AP in LOF-ONLY without seizure control",
        "severity": "HIGH RISK — confirm seizure-free before prescribing",
        "risk": "Acute seizure escalation / status epilepticus",
        "mechanism": (
            "4-AP broadly blocks Kv1 family channels including cortical interneuron Kv1 "
            "channels. In EA1+epilepsy patients with UNCONTROLLED seizures: 4-AP may "
            "further reduce cortical inhibitory interneuron repolarisation (Kv1 block) → "
            "paradoxical increase in seizure frequency or acute seizure escalation. "
            "KCNA2-GOF: 4-AP is precision therapy. KCNA1-LOF + active epilepsy: "
            "4-AP should ONLY be started after CBZ/LEV has established seizure control."
        ),
        "action": "Document seizure freedom for ≥3 months before starting 4-AP in EA1+epilepsy subtype. Start 4-AP at lowest dose (2.5 mg TID) with close weekly monitoring. If breakthrough seizures: STOP 4-AP immediately.",
    },
    {
        "drug": "Acetazolamide + high-ratio KD (≥3:1)",
        "severity": "HIGH RISK — compounded metabolic acidosis",
        "risk": "Severe metabolic acidosis / renal tubular acidosis / cardiac arrhythmia",
        "mechanism": (
            "Acetazolamide: carbonic anhydrase inhibition → urinary bicarbonate wasting → "
            "metabolic acidosis. High-ratio KD: ketosis-induced metabolic acidosis. Combined: "
            "compounded severe metabolic acidosis (pH <7.2 possible) → cardiac conduction "
            "abnormalities, respiratory compensation failure, encephalopathy. Risk particularly "
            "high in paediatric KCNA1 patients with concurrent renal immaturity."
        ),
        "action": "Do NOT combine acetazolamide with high-ratio KD (4:1). If KD needed + acetazolamide: use Modified Atkins Diet (1:1 or 2:1 ratio) + acetazolamide ≤250 mg/day; monitor electrolytes fortnightly. Consider substituting acetazolamide with 4-AP when starting KD.",
    },
    {
        "drug": "VPA without POLG screen",
        "severity": "HIGH RISK — fatal hepatic failure",
        "risk": "Alpers-Huttenlocher syndrome (progressive hepatic failure)",
        "mechanism": (
            "POLG (gamma-DNA polymerase) pathogenic biallelic variants + VPA exposure → "
            "mitochondrial hepatotoxicity → Alpers-Huttenlocher syndrome (AHS): progressive "
            "hepatic failure, encephalopathy, death. VPA inhibits POLG enzyme function → "
            "accelerates mtDNA depletion in POLG-deficient liver. Allele frequency: ~1:100 "
            "in European ancestry → ~1:10,000 clinical AHS risk with VPA in unscreened population. "
            "POLG panel (at minimum POLG p.A467T, p.W748S pathogenic variants) mandatory "
            "before VPA initiation in any patient with mitochondrial/DEE phenotype."
        ),
        "action": "POLG gene panel before VPA. Document result in chart. If POLG biallelic pathogenic: VPA ABSOLUTE CONTRAINDICATION. If POLG result pending and VPA urgently needed: informed consent + LFT/ammonia monitoring q48h + expedited POLG result.",
    },
    {
        "drug": "Mexiletine + 4-AP (combined QTc risk)",
        "severity": "MODERATE RISK — ECG monitoring required",
        "risk": "QTc prolongation / torsades de pointes",
        "mechanism": (
            "Mexiletine: Class Ib antiarrhythmic with mild QTc prolongation risk at "
            "therapeutic doses. 4-AP: broad K+ channel blocker including cardiac hERG "
            "(Kv11.1) at supratherapeutic concentrations. Combination may cause additive "
            "QTc prolongation beyond what each drug alone produces. Risk increased in: "
            "structural heart disease, hypokalaemia (with concurrent acetazolamide), "
            "female sex, bradycardia."
        ),
        "action": "If mexiletine + 4-AP combination required: mandatory ECG at baseline, 1 week, then 3-monthly. Target QTc <450ms (men) / <470ms (women). Monitor electrolytes (K+, Mg2+). Cardiology review before initiating combination if structural heart disease.",
    },
]

# ── Monitoring ────────────────────────────────────────────────────────────────
MONITORING = [
    {
        "item": "POLG screen before VPA",
        "timing": "Before VPA initiation — MANDATORY",
        "rationale": "Alpers-Huttenlocher risk; fatal hepatic failure if POLG pathogenic + VPA",
    },
    {
        "item": "HLA-B*15:02 before CBZ / OXC",
        "timing": "Before first dose — MANDATORY in Asian ancestry (CPIC Level A)",
        "rationale": "SJS/TEN risk (OR >1000 in HLA-B*15:02 carriers); CPIC Level A mandate",
    },
    {
        "item": "ECG QTc baseline and after 4-AP / mexiletine titration",
        "timing": "Baseline; 2 weeks post each dose increase; 3-monthly on stable dose",
        "rationale": "4-AP and mexiletine carry QTc prolongation risk; torsades risk if >500ms",
    },
    {
        "item": "Renal function (eGFR) for 4-AP and acetazolamide",
        "timing": "3-monthly; before any dose change",
        "rationale": "4-AP and dalfampridine renally cleared; CI if CrCl <50 mL/min",
    },
    {
        "item": "Serum sodium (OXC SIADH monitoring)",
        "timing": "Baseline; Day 3, 7, 14 after OXC start; monthly on stable dose",
        "rationale": "OXC causes SIADH; hyponatraemia threshold <130 mmol/L → reduce or stop",
    },
    {
        "item": "EA1 episode frequency diary",
        "timing": "Continuous (patient-maintained); review at every clinic visit",
        "rationale": "Primary outcome measure for 4-AP and acetazolamide efficacy assessment",
    },
    {
        "item": "EMG for myokymia quantification",
        "timing": "Baseline; 6-monthly if myokymia-dominant; at diagnosis change",
        "rationale": "Objective myokymia severity measure (doublet/triplet frequency) for treatment response",
    },
    {
        "item": "Neurological ataxia severity (SARA scale)",
        "timing": "Every clinic visit (6-monthly); more frequent during uptitration",
        "rationale": "Scale for Assessment and Rating of Ataxia (SARA 0-40); change from baseline quantifies 4-AP response",
    },
    {
        "item": "OXC serum sodium monthly (SIADH)",
        "timing": "Monthly on stable OXC dose; after intercurrent illness",
        "rationale": "OXC SIADH risk in children — immature osmoregulation; Na <130 → reduce dose",
    },
    {
        "item": "Genetic cascade testing — first-degree family members",
        "timing": "At diagnosis; repeat offer at 3 months for non-tested relatives",
        "rationale": "KCNA1 autosomal dominant — 50% offspring/sibling risk; cascade testing enables early treatment",
    },
]

# ── Lifecycle Windows ─────────────────────────────────────────────────────────
LIFECYCLE = [
    {
        "window": "Prenatal / Pre-symptomatic",
        "age": "Conception to symptom onset",
        "key_events": ["Genetic testing if parent known KCNA1 carrier", "Genetic counselling (50% transmission)", "No prenatal treatment available"],
        "note": "Familial KCNA1 LOF: cascade test before or in early childhood. De novo: diagnosed after symptom onset.",
    },
    {
        "window": "Early Childhood (0-6 years)",
        "age": "0-72 months",
        "key_events": ["First EA1 attack recognition (mean onset 3-5 years)", "Myokymia examination", "EMG + EEG", "4-AP trial if diagnosis confirmed", "Developmental assessment"],
        "note": "EA1 attacks often misdiagnosed as breath-holding, vestibular neuritis, or benign paroxysmal vertigo. EMG myokymia is the key diagnostic marker. Startle test at bedside.",
    },
    {
        "window": "School Age (6-12 years)",
        "age": "6-12 years",
        "key_events": ["Peak EA1 attack frequency", "Epilepsy onset if EA1+epilepsy subtype (mean 12 years)", "4-AP + CBZ dual therapy", "School accommodation planning", "SARA ataxia scoring at 6-monthly"],
        "note": "School functional impact: avoid PE startle environments; swimming allowed with supervision. Neuropsychological assessment if learning difficulties (rare in pure EA1; possible in EA1+epilepsy).",
    },
    {
        "window": "Adolescence (12-18 years)",
        "age": "12-18 years",
        "key_events": ["Epilepsy onset or escalation (if EA1+epilepsy)", "SUDEP counselling", "Driving restriction assessment", "Emotional trigger management", "Transition to adult neurology"],
        "note": "Emotional triggers peak in adolescence (autonomic instability). GTCS: driving restriction mandatory. VPPP counselling for adolescent females. Identity and quality-of-life support critical.",
    },
    {
        "window": "Young Adult (18-40 years)",
        "age": "18-40 years",
        "key_events": ["EA1 may improve spontaneously (30-40% partial remission)", "Pregnancy management (VPA teratogenicity — VPPP)", "Driving reassessment", "Employment counselling (safety-critical work restriction)", "Genetic counselling pre-pregnancy"],
        "note": "Many EA1-pure patients experience reduced attack frequency in 3rd decade. EA1+epilepsy: seizure control typically requires maintenance. SUDEP risk highest in GTCS + drug-resistant subgroup.",
    },
    {
        "window": "Middle-Older Adult (>40 years)",
        "age": ">40 years",
        "key_events": ["Long-term cerebellar atrophy monitoring (MRI volumetry)", "Medication review for adverse effects (long-term CBZ/VPA)", "Comorbidity management (bone density VPA, liver CBZ)", "Myokymia surveillance (EMG)"],
        "note": "Cerebellar atrophy progression in high-attack-frequency patients. Long-term CBZ: bone density monitoring (enzyme induction → vitamin D depletion). Long-term VPA: PCOS, weight. Consider VNS if DRE persists.",
    },
]

# ── Key Concepts ──────────────────────────────────────────────────────────────
CONCEPTS = [
    {
        "term": "KCNA1 (12p13.32)",
        "definition": "Gene encoding Kv1.1 voltage-gated potassium channel alpha subunit; member of Shaker-related Kv1 family; highest expression in cerebellar basket-cell pinceau terminals and peripheral nerve paranodal/juxtaparanodal nodes of Ranvier.",
    },
    {
        "term": "Kv1.1 channel (Kv1 family)",
        "definition": "Voltage-gated K+ channel; forms homo- or heterotetramers with Kv1.2, Kv1.4, Kv1.6; controls action potential repolarisation at axon initial segment, paranodal regions, and basket-cell terminals; critically regulates Purkinje cell inhibitory timing in cerebellum.",
    },
    {
        "term": "Episodic Ataxia Type 1 (EA1)",
        "definition": "Autosomal dominant channelopathy caused by KCNA1 LOF; brief (seconds to 2 minutes) attacks of cerebellar ataxia triggered by startle, exercise, or emotion; interictal myokymia (continuous rippling of periorbital and hand muscles) is pathognomonic; onset childhood-adolescence.",
    },
    {
        "term": "Myokymia / Neuromyotonia",
        "definition": "Spontaneous repetitive motor unit discharge (doublets, triplets, multiplets at 50-150 Hz on EMG) caused by Kv1.1 LOF at peripheral nerve nodes of Ranvier; visible as continuous rippling of small muscles (periorbital, hand, finger); interictal hallmark of EA1.",
    },
    {
        "term": "4-Aminopyridine (4-AP) precision therapy",
        "definition": "Broad-spectrum K+ channel blocker that compensates Kv1.1 LOF by prolonging action potential duration at cerebellar basket-cell terminals → restores Purkinje cell inhibitory timing fidelity → reduces EA1 attack frequency by 70-80%. Level A evidence.",
    },
    {
        "term": "EA1 vs EA2 differential diagnosis",
        "definition": "EA1 (KCNA1): brief attacks (seconds to 2 min), interictal myokymia (EMG doublets), partial acetazolamide response, 4-AP very effective. EA2 (CACNA1A): longer attacks (minutes to hours), interictal nystagmus (no myokymia), strong acetazolamide response, 4-AP less effective.",
    },
    {
        "term": "Scheffer 1998 Victorian family",
        "definition": "Landmark KCNA1 kindred (T226M mutation) with 4-generation pedigree of EA1 + temporal lobe epilepsy + febrile seizures; first demonstration of KCNA1 as an epilepsy gene; established the EA1+epilepsy spectrum.",
    },
    {
        "term": "Startle reflex threshold in EA1",
        "definition": "The startle reflex (pontine reticular formation → cortical arousal) triggers EA1 attacks in >95% of patients; the threshold for startle-provoked episodes is dramatically lower in EA1 due to reduced Kv1.1 buffering of basket-cell pinceau burst discharge.",
    },
    {
        "term": "SARA (Scale for Assessment and Rating of Ataxia)",
        "definition": "8-item, 40-point validated ataxia severity scale: gait (0-8), stance (0-6), sitting (0-4), speech (0-6), finger-nose (0-4), nose-finger (0-4), diadochokinesis (0-4), heel-shin (0-4). Primary outcome measure for 4-AP and acetazolamide trials in EA1.",
    },
    {
        "term": "Kv1.1 juxtaparanodal enrichment",
        "definition": "Kv1.1 is concentrated at juxtaparanodal regions of myelinated nerve fibres (adjacent to nodes of Ranvier) via Caspr2-TAG-1 scaffolding complex. This positioning normally dampens axonal post-AP afterdepolarisations. LOF disrupts this damping → repetitive paranodal discharge → myokymia.",
    },
    {
        "term": "Acetazolamide in EA1",
        "definition": "Carbonic anhydrase inhibitor; creates mild metabolic acidosis that stabilises Kv1.1 pH-gating; partially reduces EA1 episodes in ~60% of patients but less effective than 4-AP. Particularly useful for hyperventilation-triggered and alkalosis-sensitive patients.",
    },
    {
        "term": "HLA-B*15:02 SJS/TEN risk",
        "definition": "CPIC Level A: HLA-B*15:02 allele (6% Han Chinese, 8% Thai) confers near-absolute SJS/TEN risk with carbamazepine/oxcarbazepine. Mandatory testing before CBZ/OXC in Asian ancestry patients with KCNA1+epilepsy requiring carbamazepine.",
    },
    {
        "term": "POLG-VPA Alpers-Huttenlocher CI",
        "definition": "POLG biallelic variants + VPA = Alpers-Huttenlocher syndrome: progressive hepatic failure, encephalopathy, death. POLG screen mandatory before VPA in any genetic epilepsy. In KCNA1+epilepsy: document POLG result before prescribing VPA.",
    },
    {
        "term": "SUDEP risk in EA1+epilepsy",
        "definition": "SUDEP (Sudden Unexpected Death in Epilepsy) risk is present in EA1+epilepsy subtype with GTCS. Annual SUDEP counselling mandatory. Risk factors: uncontrolled nocturnal GTCS, male sex, DRE. Wearable seizure detection recommended for nocturnal GTCS.",
    },
]

# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS = [
    "EA1 episode duration >5 min → emergency management protocol (prolonged attack = ataxic status risk)",
    "QTc >450ms (men) / >470ms (women) → STOP 4-AP or mexiletine, cardiology review",
    "4-AP max dose 20-30 mg/day (IR); 10 mg/day (ER dalfampridine); reduce if CrCl <80 mL/min",
    "Serum Na+ <130 mmol/L on OXC → reduce dose; <125 mmol/L → STOP OXC, emergency electrolyte management",
    "CrCl <50 mL/min → 4-AP (dalfampridine) CONTRAINDICATED; dose-adjust IR 4-AP",
    "CBZ serum level target 4-12 µg/mL; OXC-MHD target 10-35 µg/mL",
    "SARA score change ≥4 points → clinically meaningful 4-AP response (document in clinic record)",
    "EA1 episodes >3 per week on 4-AP → add acetazolamide or reassess dose",
    "≥2 AED failures for epilepsy component → ketogenic diet referral",
    "GTCS uncontrolled + nocturnal → wearable seizure detection device mandatory",
]

# ── Standards ─────────────────────────────────────────────────────────────────
STANDARDS = [
    "ILAE-2022 (Scheffer IE et al. Epilepsia — genetic epilepsy classification including EA1+epilepsy spectrum)",
    "NICE-NG217-2022 (Epilepsies: Diagnosis and Management — UK national guideline including channelopathies)",
    "Browne-1994-NatGenet (Browne DL et al. — original KCNA1 discovery in EA1 families; position paper level A)",
    "Scheffer-1998-Lancet (Scheffer IE et al. Temporal lobe epilepsy + EA1 Victorian family — KCNA1 epilepsy spectrum)",
    "Rajakulendran-2007-Brain (KCNA1 mutation spectrum + 4-AP response data; functional characterisation)",
    "Strupp-2008-NEJM (4-AP Level A evidence for EA1; crossover trial; primary evidence base for 4-AP use)",
    "CPIC-HLA-B-CBZ-2023 (Karnes JH et al. — CPIC Level A HLA-B*15:02 mandate before CBZ/OXC)",
    "ACMG-AMP-2015 (Richards S et al. — variant pathogenicity classification framework)",
]

# ── References ────────────────────────────────────────────────────────────────
REFERENCES = [
    "Browne DL et al. (1994) Episodic ataxia/myokymia syndrome is associated with point mutations in the human potassium channel gene, KCNA1. Nat Genet 8:136. (PMID 7987398) — KCNA1 discovery",
    "Scheffer IE et al. (1998) Temporal lobe epilepsy and FSGS with KCNA1 T226M — Victorian family pedigree. Lancet 352:1602. (PMID 9843098) — EA1+epilepsy spectrum",
    "Rajakulendran S et al. (2007) Episodic ataxia type 1 — KCNA1 mutation spectrum and functional analysis. Brain 130:1780. (PMID 17418832) — mutation catalogue + pathophysiology",
    "Strupp M et al. (2008) 4-aminopyridine restores visual and auditory function in patients with downbeat nystagmus and EA2. NEJM 359:2544. (PMID 19073978) — 4-AP Level A evidence",
    "Graves TD et al. (2010) KCNA1 missense mutations disrupt fast inactivation of Kv1.1 potassium channels. Brain 133:3439. (PMID 21044947) — functional characterisation of LOF variants",
    "Bhatt SS et al. (2023) Evidence-based pharmacological management of genetic epilepsies. Epilepsia — EA1+epilepsy CBZ+4-AP treatment framework",
]

# ── Patient Cohort Simulation ─────────────────────────────────────────────────
_FIRST = ["Aiden","Elias","Maya","Sophie","Lucas","Priya","Kai","Zara","Noah","Isla",
           "Oliver","Emma","Liam","Ava","Ethan","Mia","James","Lily","Leo","Nora",
           "Henry","Grace","Sam","Chloe","Ben","Amara","Jack","Freya","Max","Leila",
           "Oscar","Ruby","Felix","Zoe","Hugo","Ivy","Ravi","Clara","Finn","Ada"]
_LAST  = ["Okafor","Chen","Patel","Singh","Kim","Andersen","Müller","Sato","Hassan","Rossi",
           "García","Park","Thompson","Nguyen","Williams","Martin","Jackson","Yamamoto","Silva","Lee",
           "Brown","Taylor","Wilson","Clark","Moore","White","Davis","Anderson","Smith","Johnson",
           "Thomas","Jackson","Harris","Martinez","Robinson","Lewis","Walker","Hall","Young","King"]


def _generate_patients():
    pts = []
    for i in range(40):
        random.seed(SEED + i)
        etiology_roll = random.random()
        if etiology_roll < 0.50:
            etiology = "EA1 pure LOF (no epilepsy)"
            has_epilepsy = False
            drug_resistant = False
            seizure_free = True
            on_4ap = random.random() < 0.80
            on_cbz = random.random() < 0.40
            on_vpa = False
            on_kd = False
            on_lev = False
            on_acetazolamide = random.random() < 0.30
            sara_score = random.randint(2, 14)
        elif etiology_roll < 0.75:
            etiology = "EA1 + focal epilepsy / GTCS"
            has_epilepsy = True
            drug_resistant = random.random() < 0.15
            seizure_free = random.random() < 0.65
            on_4ap = random.random() < 0.70
            on_cbz = True
            on_vpa = random.random() < 0.30
            on_kd = random.random() < 0.10
            on_lev = random.random() < 0.45
            on_acetazolamide = random.random() < 0.20
            sara_score = random.randint(4, 20)
        elif etiology_roll < 0.85:
            etiology = "Myokymia-dominant LOF"
            has_epilepsy = False
            drug_resistant = False
            seizure_free = True
            on_4ap = random.random() < 0.40
            on_cbz = random.random() < 0.75
            on_vpa = False
            on_kd = False
            on_lev = False
            on_acetazolamide = random.random() < 0.15
            sara_score = random.randint(1, 6)
        elif etiology_roll < 0.95:
            etiology = "Severe LOF truncating + epilepsy"
            has_epilepsy = True
            drug_resistant = random.random() < 0.40
            seizure_free = random.random() < 0.40
            on_4ap = random.random() < 0.60
            on_cbz = True
            on_vpa = random.random() < 0.45
            on_kd = random.random() < 0.25
            on_lev = random.random() < 0.55
            on_acetazolamide = random.random() < 0.30
            sara_score = random.randint(8, 28)
        else:
            etiology = "Phenocopy / panel-negative"
            has_epilepsy = random.random() < 0.30
            drug_resistant = False
            seizure_free = True
            on_4ap = random.random() < 0.30
            on_cbz = random.random() < 0.40
            on_vpa = False
            on_kd = False
            on_lev = False
            on_acetazolamide = random.random() < 0.35
            sara_score = random.randint(3, 16)

        polg_tested = "Y" if random.random() < 0.84 else "N"
        hla_tested = random.random() < 0.80
        ecg_done = random.random() < 0.75
        on_mexiletine = random.random() < 0.12
        hyponatraemia = on_cbz and not on_4ap and random.random() < 0.12
        age_years = random.randint(4, 55)
        attack_freq_per_month = random.randint(0, 25)

        pts.append({
            "id": f"KA1-{i+1:03d}",
            "name": f"{random.choice(_FIRST)} {random.choice(_LAST)}",
            "age_years": age_years,
            "etiology": etiology,
            "has_epilepsy": has_epilepsy,
            "drug_resistant": drug_resistant,
            "seizure_free": seizure_free,
            "sara_score": sara_score,
            "ea1_attacks_per_month": attack_freq_per_month,
            "on_4ap": on_4ap,
            "on_cbz": on_cbz,
            "on_vpa": on_vpa,
            "on_kd": on_kd,
            "on_lev": on_lev,
            "on_acetazolamide": on_acetazolamide,
            "on_mexiletine": on_mexiletine,
            "polg_tested": polg_tested,
            "hla_b1502_tested": hla_tested,
            "ecg_done": ecg_done,
            "hyponatraemia": hyponatraemia,
        })
    return pts


# ── Public API ────────────────────────────────────────────────────────────────
def get_overview():
    """Return KCNA1 overview dict."""
    pts = _generate_patients()
    n = len(pts)
    has_epilepsy = sum(1 for p in pts if p["has_epilepsy"])
    drug_resistant = sum(1 for p in pts if p["drug_resistant"])
    seizure_free = sum(1 for p in pts if p["seizure_free"])
    on_4ap = sum(1 for p in pts if p["on_4ap"])
    on_cbz = sum(1 for p in pts if p["on_cbz"])
    on_kd = sum(1 for p in pts if p["on_kd"])
    polg_done = sum(1 for p in pts if p["polg_tested"] == "Y")
    hla_done = sum(1 for p in pts if p["hla_b1502_tested"])
    ecg_done = sum(1 for p in pts if p["ecg_done"])
    avg_sara = round(sum(p["sara_score"] for p in pts) / n, 1)
    avg_attacks = round(sum(p["ea1_attacks_per_month"] for p in pts) / n, 1)

    return {
        "gene": "KCNA1",
        "locus": "12p13.32",
        "inheritance": "Autosomal dominant (LOF familial or de novo); rare GOF",
        "protein": (
            "Kv1.1 — Shaker-related voltage-gated potassium channel alpha subunit; "
            "highest expression in cerebellar basket-cell pinceau terminals and peripheral "
            "nerve paranodal/juxtaparanodal nodes of Ranvier"
        ),
        "mechanism": (
            "KCNA1 LOF → impaired Kv1.1 K+ repolarisation at cerebellar basket-cell "
            "pinceau terminals → loss of precise Purkinje cell inhibitory timing → "
            "brief episodic cerebellar ataxia (EA1) + interictal myokymia. "
            "Peripheral nerve LOF → paranodal repetitive discharge → neuromyotonia. "
            "25% families: cortical/hippocampal Kv1.1 LOF → focal epilepsy + GTCS."
        ),
        "key_aha": (
            "KCNA1 (12p13.32) — Kv1.1 EA1 + myokymia channelopathy. "
            "EA1 hallmark: BRIEF startle-triggered ataxia (seconds) + interictal periorbital/hand "
            "myokymia. 4-AP precision therapy: Level A evidence, 70-80% episode reduction. "
            "25% have focal epilepsy: control seizures with CBZ FIRST, THEN add 4-AP. "
            "HLA-B*15:02 MANDATORY before CBZ/OXC (CPIC Level A). POLG before VPA. "
            "Tiagabine ABSOLUTE CI. Distinguish EA1 from EA2 (CACNA1A): myokymia=EA1, no myokymia=EA2. "
            "4-AP+mexiletine combo: ECG QTc monitoring mandatory."
        ),
        "n_patients": n,
        "epilepsy_pct": round(100 * has_epilepsy / n),
        "drug_resistant_pct": round(100 * drug_resistant / n),
        "seizure_free_pct": round(100 * seizure_free / n),
        "on_4ap_pct": round(100 * on_4ap / n),
        "on_cbz_pct": round(100 * on_cbz / n),
        "on_kd_pct": round(100 * on_kd / n),
        "polg_done_pct": round(100 * polg_done / n),
        "hla_done_pct": round(100 * hla_done / n),
        "ecg_done_pct": round(100 * ecg_done / n),
        "avg_sara_score": avg_sara,
        "avg_ea1_attacks_per_month": avg_attacks,
        "tiagabine_alert": "ABSOLUTE CI — NCSE risk in KCNA1+epilepsy patients (GAT-1 → GABA accumulation → tonic GABA-A block)",
        "fourAP_alert": "4-AP FIRST-LINE for EA1 (Level A); establish seizure control with CBZ BEFORE adding 4-AP in EA1+epilepsy. ECG QTc mandatory before 4-AP.",
        "cbz_alert": "HLA-B*15:02 MANDATORY before CBZ/OXC (CPIC Level A, Asian ancestry). OXC: serum Na+ monitoring mandatory (SIADH).",
        "polg_alert": "POLG MANDATORY before VPA — Alpers-Huttenlocher hepatic failure risk",
        "ecg_alert": "ECG QTc baseline before 4-AP and mexiletine; at each titration step; 3-monthly on stable dose. QTc >450ms (men) / >470ms (women) → STOP.",
        "contraindications_summary": [
            "Tiagabine — ABSOLUTE CI: NCSE risk (GAT-1 block → tonic GABA-A activation)",
            "4-AP without seizure control — HIGH RISK: seizure escalation in uncontrolled EA1+epilepsy",
            "Acetazolamide + high-ratio KD — HIGH RISK: compounded metabolic acidosis",
            "VPA without POLG screen — HIGH RISK: Alpers-Huttenlocher hepatic failure",
            "Mexiletine + 4-AP without ECG — MODERATE RISK: additive QTc prolongation",
        ],
        "thresholds": THRESHOLDS,
        "references": [r.split(" (")[0] for r in REFERENCES],
    }


def get_breakdown():
    """Return KCNA1 breakdown dict."""
    pts = _generate_patients()
    n = len(pts)
    has_epilepsy = sum(1 for p in pts if p["has_epilepsy"])
    drug_resistant = sum(1 for p in pts if p["drug_resistant"])
    on_4ap = sum(1 for p in pts if p["on_4ap"])
    on_cbz = sum(1 for p in pts if p["on_cbz"])
    on_kd = sum(1 for p in pts if p["on_kd"])
    polg_done = sum(1 for p in pts if p["polg_tested"] == "Y")
    hla_done = sum(1 for p in pts if p["hla_b1502_tested"])
    vpa_without_polg = sum(1 for p in pts if p["on_vpa"] and p["polg_tested"] == "N")
    ecg_done = sum(1 for p in pts if p["ecg_done"])

    etiology_dist = []
    for ec in ETIOLOGY_CATALOG:
        etiology_dist.append({
            "etiology": ec["etiology"],
            "pct": ec["pct"],
            "n": ec["n"],
            "category": ec["category"],
            "mechanism_short": ec["mechanism"][:120],
            "eeg_signature_short": ec["eeg_correlate"][:80],
        })

    return {
        "summary": {
            "n": n,
            "epilepsy_pct": round(100 * has_epilepsy / n),
            "drug_resistant_pct": round(100 * drug_resistant / n),
            "on_4ap_pct": round(100 * on_4ap / n),
            "on_cbz_pct": round(100 * on_cbz / n),
            "on_kd_pct": round(100 * on_kd / n),
            "polg_done_pct": round(100 * polg_done / n),
            "hla_done_pct": round(100 * hla_done / n),
            "ecg_done_pct": round(100 * ecg_done / n),
            "vpa_without_polg": vpa_without_polg,
        },
        "etiology_distribution": etiology_dist,
        "etiology_catalog": ETIOLOGY_CATALOG,
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
    """Return KCNA1 definitions dict."""
    return {
        "concepts": CONCEPTS,
        "contraindications_full": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }
